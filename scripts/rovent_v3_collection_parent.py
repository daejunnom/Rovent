"""Optional original-owner supervisor for one V3 Collection child.

The caller supplies the already-owned input, output lease, binary and cgroup.
This supervisor is not a source-lock, immutable-input or GPU-completion proof.
The Rust Collection writer remains gated until those separate proofs exist.
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
import time

from managed_output_child_handoff import ManagedCollectionOutputHandoffV2
from managed_process import OwnedProcessGroup, run_group
from rovent_cgroup_owner import CgroupOwner, entry_command
from rovent_v3_input_snapshot import OriginalControlInput


def _namespace(pid):
    info = os.stat(f"/proc/{pid}/ns/time")
    return info.st_dev, info.st_ino


def _seconds_not_later(nanoseconds):
    return math.nextafter(nanoseconds / 1_000_000_000, -math.inf)


class OriginalCollectionProcess:
    """Keep the owners used by one opt-in collector execution alive.

    Create this object before calling ``run`` and retain it if dispatch, stop,
    group drain, or output completion is uncertain. No lease is finalized here.
    """

    def __init__(self, lease_key, scope, original_input, binary_fd, work_ns, cleanup_ns,
                 *, bootstrap_budget_ms):
        if (sys.platform != "linux" or type(scope) is not CgroupOwner
                or type(original_input) is not OriginalControlInput
                or type(binary_fd) is not int or binary_fd < 0
                or type(bootstrap_budget_ms) is not int
                or not 0 < bootstrap_budget_ms < 1 << 64
                or type(work_ns) is not int or type(cleanup_ns) is not int
                or not 0 < work_ns <= cleanup_ns < 1 << 64
                or time.monotonic_ns() >= work_ns):
            raise ValueError("original Linux owner and future ordered deadlines required")
        self.scope = scope
        self.input = original_input
        self.group = OwnedProcessGroup()
        self._pid = os.getpid()
        self._namespace = _namespace(self._pid)
        self._work_ns, self._cleanup_ns = work_ns, cleanup_ns
        self._bootstrap_budget_ms = bootstrap_budget_ms
        self._binary_fd = self._barrier_read = self._barrier_write = None
        self._fd_close_state = {}
        self.handoff = None
        self._attempted = False
        try:
            self._binary_fd = os.dup(binary_fd)
            self._barrier_read, self._barrier_write = os.pipe2(os.O_CLOEXEC)
            self.handoff = ManagedCollectionOutputHandoffV2(lease_key, self.group, scope, original_input)
        except BaseException as error:
            try:
                self.close_unstarted()
            except BaseException as close_error:
                error.add_note("unstarted collector cleanup also failed: " + str(close_error))
            raise

    def _check_namespace(self, child_pid=None):
        if os.getpid() != self._pid or _namespace(self._pid) != self._namespace:
            raise RuntimeError("original supervisor time namespace changed")
        if child_pid is not None and _namespace(child_pid) != self._namespace:
            raise RuntimeError("original child time namespace differs")

    def run(self, native_arguments, cwd, env, admission_check, on_stop, *, stdout=None):
        if (self._attempted or self.handoff is None or not callable(admission_check)
                or not callable(on_stop) or not isinstance(native_arguments, (tuple, list))
                or any(not isinstance(value, str) or value.startswith("--v3-")
                       for value in native_arguments)):
            raise ValueError("one original collector execution and explicit stop owner required")
        self._check_namespace()
        work, cleanup = _seconds_not_later(self._work_ns), _seconds_not_later(self._cleanup_ns)
        remaining = work - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("original work deadline expired before dispatch")
        data = self.input.bytes
        channel, output, input_dir, plan = self.handoff.child_fds
        arguments = ["rovent-collect", *native_arguments, "--v3-mode=pc",
                     "--v3-plan=" + self.input.selector,
                     "--v3-plan-sha256=" + hashlib.sha256(data).hexdigest(),
                     "--v3-plan-bytes=" + str(len(data)),
                     "--v3-managed-output-channel-fd=" + str(channel),
                     "--v3-managed-output-dir-fd=" + str(output),
                     "--v3-managed-input-dir-fd=" + str(input_dir),
                     "--v3-managed-plan-fd=" + str(plan),
                     "--v3-bootstrap-budget-ms=" + str(self._bootstrap_budget_ms),
                     "--v3-work-deadline-monotonic-ns=" + str(self._work_ns),
                     "--v3-cleanup-deadline-monotonic-ns=" + str(self._cleanup_ns)]
        command = entry_command(self._binary_fd, self._barrier_read, arguments)
        inherited = (self._binary_fd, self._barrier_read, channel, output, input_dir, plan)

        def admitted():
            child = self.group._child
            if child is None:
                raise RuntimeError("original barrier child is missing")
            self._check_namespace(child.pid)
            self._close_local_fd("_barrier_read")
            self.scope.admit_waiting_child_borrowed(child.pid, self._barrier_write)
            self._close_local_fd("_barrier_write")

        def tick():
            child = self.group._child
            if child is None:
                raise RuntimeError("original collector child is missing")
            try:
                self._check_namespace(child.pid)
            except FileNotFoundError:
                # A reaped/vanished namespace is not itself completion. Only
                # the same unreaped child exit distinguishes the benign race.
                if os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
                    raise

        self._attempted = True
        try:
            return run_group(command, cwd, env, remaining, admission_check,
                             pass_fds=inherited, stdout=stdout, on_spawn=admitted,
                             verify_before_spawn=True, absolute_deadline=work,
                             cleanup_absolute_deadline=cleanup, on_tick=tick,
                             on_stop=on_stop, owned_group=self.group,
                             output_handoff=self.handoff)
        except BaseException as error:
            try:
                unstarted = (self.group._child is None and not self.handoff._dispatch_attempted
                             and not self.scope.output_dispatch_attempted(self.group))
            except BaseException:
                unstarted = False
            if unstarted:
                self._attempted = False
                try:
                    self.close_unstarted()
                except BaseException as close_error:
                    error.add_note("unstarted collector cleanup also failed: " + str(close_error))
            raise

    def close_unstarted(self):
        if self._attempted:
            raise RuntimeError("attempted collector owner requires original group settlement")
        if self.handoff is not None:
            self.handoff.revoke()
        for field in ("_barrier_read", "_barrier_write", "_binary_fd"):
            self._close_local_fd(field)

    def _close_local_fd(self, field):
        if self._fd_close_state.get(field) == "unknown":
            raise RuntimeError("original collector FD close outcome remains unknown")
        fd = getattr(self, field, None)
        if fd is not None:
            # A failed close may already have consumed the descriptor number.
            # Never retry it against a potentially reused unrelated owner.
            setattr(self, field, None)
            self._fd_close_state[field] = "unknown"
            os.close(fd)
            self._fd_close_state[field] = "closed"

    def close_settled_process_fds(self):
        """Release only local launch FDs after this exact group was drained."""
        if not self._attempted or not self.group._completed:
            raise RuntimeError("original collector process group is not confirmed gone")
        for field in ("_barrier_read", "_barrier_write", "_binary_fd"):
            self._close_local_fd(field)
