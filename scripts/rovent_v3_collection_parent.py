"""Optional original-owner supervisor for one V3 Collection child.

The caller supplies the already-owned input, output lease, binary and cgroup.
This supervisor is not a source-lock, immutable-input or GPU-completion proof.
The Rust Collection writer remains gated until those separate proofs exist.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import sys
import threading
import time

from managed_output_child_handoff import ManagedCollectionOutputHandoffV2
from managed_process import OwnedProcessGroup, run_group
from managed_outputs import OutputLease
from rovent_cgroup_owner import CgroupOwner, CgroupSetupError, entry_command
from rovent_v3_input_snapshot import OriginalControlInput, open_directory, relative


def _namespace(pid):
    info = os.stat(f"/proc/{pid}/ns/time")
    return info.st_dev, info.st_ino


def _seconds_not_later(nanoseconds):
    return math.nextafter(nanoseconds / 1_000_000_000, -math.inf)


class OriginalCollectionWindow:
    """One caller-specified budget sealed to the current thread's clock."""

    def __init__(self, *, work_budget_ns, cleanup_budget_ns):
        if (sys.platform != "linux" or type(work_budget_ns) is not int
                or type(cleanup_budget_ns) is not int or work_budget_ns <= 0
                or cleanup_budget_ns <= 0):
            raise ValueError("positive explicit Linux work and cleanup budgets required")
        self._pid, self._thread = os.getpid(), threading.get_native_id()
        self._time_fd = self._children_fd = None
        self._fd_close_state = {}
        try:
            self._time_fd = os.open("/proc/thread-self/ns/time", os.O_RDONLY | os.O_CLOEXEC)
            self._children_fd = os.open("/proc/thread-self/ns/time_for_children", os.O_RDONLY | os.O_CLOEXEC)
            current, children = os.fstat(self._time_fd), os.fstat(self._children_fd)
            self._identity = (current.st_dev, current.st_ino)
            if self._identity != (children.st_dev, children.st_ino):
                raise ValueError("current and child time namespaces differ")
            now = time.monotonic_ns()
            self.work_ns = now + work_budget_ns
            self.cleanup_ns = self.work_ns + cleanup_budget_ns
            if self.cleanup_ns >= 1 << 64:
                raise ValueError("collection deadline overflow")
            self.check()
        except BaseException as error:
            try:
                self.close()
            except BaseException as close_error:
                error.add_note("original window cleanup also failed: " + str(close_error))
            raise

    def check(self):
        if self._time_fd is None or self._children_fd is None:
            raise RuntimeError("original Collection window is closed")
        if (os.getpid() != self._pid or threading.get_native_id() != self._thread
                or _namespace("thread-self") != self._identity):
            raise RuntimeError("original Collection time namespace or thread changed")
        children = os.stat("/proc/thread-self/ns/time_for_children")
        if (children.st_dev, children.st_ino) != self._identity:
            raise RuntimeError("original child time namespace changed")

    def close(self):
        failure = None
        for field in ("_time_fd", "_children_fd"):
            if self._fd_close_state.get(field) == "unknown":
                if failure is None:
                    failure = RuntimeError("original Collection window FD close remains unknown")
                continue
            fd = getattr(self, field, None)
            if fd is not None:
                setattr(self, field, None)
                self._fd_close_state[field] = "unknown"
                try:
                    os.close(fd)
                    self._fd_close_state[field] = "closed"
                except OSError as error:
                    if failure is None:
                        failure = error
        if failure is not None:
            raise failure


class OriginalCollectionSupervisor:
    """Actual creator for an explicitly selected managed Collection launch.

    Construct the owner first and retain it across ``prepare`` and ``run``:
    exceptions may leave original lease, cgroup or child custody in this same
    object. Input immutability, native lock acceptance and GPU completion are
    not inferred from this supervisor or from its output lease.
    """

    def __init__(self, *, input_root, plan_path, output, checkout_root, reserve_bytes,
                 binary_fd, window, bootstrap_budget_ms):
        if type(window) is not OriginalCollectionWindow:
            raise ValueError("original Collection clock window required")
        window.check()
        self.input_root = Path(input_root)
        self.plan_path = relative(plan_path)
        self.output = Path(output)
        self.checkout_root = Path(checkout_root)
        if (not all(path.is_absolute() and ".." not in path.parts
                    for path in (self.input_root, self.output, self.checkout_root))
                or not self.plan_path.as_posix().startswith("inputs/")
                or any(part.lower().startswith(".env") or "credential" in part.lower()
                       or "service-account" in part.lower() or "service_account" in part.lower()
                       or "api_key" in part.lower() or "api-key" in part.lower()
                       or part.lower() in {"id_rsa", "id_ed25519", "id_ecdsa"}
                       for part in self.plan_path.parts)
                or type(reserve_bytes) is not int or reserve_bytes <= 0
                or type(binary_fd) is not int or binary_fd < 0
                or time.monotonic_ns() >= window.work_ns
                or type(bootstrap_budget_ms) is not int
                or not 0 < bootstrap_budget_ms < 1 << 64):
            raise ValueError("absolute original roots and explicit collection reservation required")
        self.reserve_bytes = reserve_bytes
        # Retain the caller's actual open file description now; a raw number
        # could otherwise be closed and reassigned before prepare().
        self._binary_fd = os.dup(binary_fd)
        self._binary_close_state = "open"
        self.window = window
        self.work_ns, self.cleanup_ns = window.work_ns, window.cleanup_ns
        self.bootstrap_budget_ms = bootstrap_budget_ms
        self.input_dir_fd = None
        self._input_dir_close_state = None
        self.input = self.scope = self.lease = self.process = None
        self.setup_failure = None
        self._setup_recovered = False
        self._acquired = False
        self._prepare_attempted = False
        self.outcome = None

    def prepare(self):
        if self._prepare_attempted:
            raise RuntimeError("original Collection supervisor was already prepared or attempted")
        self._prepare_attempted = True
        try:
            self.window.check()
            self.input_dir_fd = open_directory(self.input_root)
            self.input = OriginalControlInput(self.input_dir_fd, self.plan_path)
            self.scope = CgroupOwner.prepare()
            self.lease = OutputLease(self.output, self.checkout_root, reserve=self.reserve_bytes)
            self.lease.acquire()
            self._acquired = True
            key = self.lease.prepare_live_handoff()
            self.lease.suspend_live()
            self.process = OriginalCollectionProcess(
                key, self.scope, self.input, self._binary_fd, self.work_ns, self.cleanup_ns,
                bootstrap_budget_ms=self.bootstrap_budget_ms)
            self._close_original_binary()
        except BaseException as error:
            self.setup_failure = error
            raise
        return self

    def run(self, native_arguments, cwd, env, admission_check, *,
            cleanup_request_start, cleanup_request_poll, **options):
        if self.process is None or self.setup_failure is not None or self.outcome is not None:
            raise RuntimeError("original Collection launch is not prepared or was already run")
        self.window.check()
        try:
            self.outcome = self.process.run(
                native_arguments, cwd, env, admission_check,
                cleanup_request_start=cleanup_request_start,
                cleanup_request_poll=cleanup_request_poll, **options)
        except BaseException as error:
            # The caller retains this same session and original process owner.
            self.outcome = error
            raise
        return self.outcome

    def close_unstarted(self):
        """Release only setup owners for which dispatch was never attempted."""
        if isinstance(self.setup_failure, CgroupSetupError) and not self._setup_recovered:
            raise RuntimeError("original cgroup setup recovery owner must be resolved")
        if self.process is not None:
            self.process.close_unstarted()
        if self.lease is not None:
            if not self._acquired and self.lease.record is not None:
                raise RuntimeError("partially acquired output owner requires original recovery")
            if self._acquired:
                if self.lease._live_suspended:
                    self.lease.resume_live()
                self.lease.finish(status="not-started", tree_gone=True)
        if self.scope is not None:
            self.scope.close()
        self._close_original_binary()
        if self.input is not None:
            self.input.close()
        if self._input_dir_close_state == "unknown":
            raise RuntimeError("original input directory FD close remains unknown")
        if self.input_dir_fd is not None:
            fd, self.input_dir_fd = self.input_dir_fd, None
            self._input_dir_close_state = "unknown"
            os.close(fd)
            self._input_dir_close_state = "closed"

    def recover_failed_scope_setup(self):
        """Retry only this same never-dispatched cgroup setup owner."""
        if not isinstance(self.setup_failure, CgroupSetupError) or self._setup_recovered:
            raise RuntimeError("original cgroup setup recovery is unavailable")
        recovery = self.setup_failure.recovery
        if isinstance(recovery, CgroupOwner):
            recovery.close()
        else:
            recovery.retry()
        self._setup_recovered = True

    def _close_original_binary(self):
        if self._binary_close_state == "unknown":
            raise RuntimeError("original collector binary FD close remains unknown")
        if self._binary_fd is not None:
            fd, self._binary_fd = self._binary_fd, None
            self._binary_close_state = "unknown"
            os.close(fd)
            self._binary_close_state = "closed"


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

    def run(self, native_arguments, cwd, env, admission_check, on_stop=None, *, stdout=None,
            cleanup_signal_ns=None, long_game_preflight=None, cleanup_request_start=None,
            cleanup_request_poll=None):
        """Optionally select a pinned profile for preflight only, not native execution."""
        if (self._attempted or self.handoff is None or not callable(admission_check)
                or (on_stop is not None and not callable(on_stop))
                or ((cleanup_request_start is not None or cleanup_request_poll is not None)
                    and (not callable(cleanup_request_start) or not callable(cleanup_request_poll)
                         or on_stop is not None))
                or (on_stop is None and cleanup_request_start is None)
                or not isinstance(native_arguments, (tuple, list))
                or any(not isinstance(value, str) or value.startswith("--v3-")
                       for value in native_arguments)
                or (cleanup_signal_ns is not None and
                    (type(cleanup_signal_ns) is not int or
                     not self._work_ns <= cleanup_signal_ns <= self._cleanup_ns))):
            raise ValueError("one original collector execution and explicit stop owner required")
        preflight_args = []
        if long_game_preflight is not None:
            if (type(long_game_preflight) is not dict
                    or set(long_game_preflight) != {"path", "sha256", "bytes"}):
                raise ValueError("long-game preflight requires a complete pin")
            path, sha256, size = (long_game_preflight[key] for key in ("path", "sha256", "bytes"))
            if (type(path) is not str or not path.startswith("inputs/")
                    or len(path.encode("utf-8")) > 4096
                    or any(char in path for char in "\\:")
                    or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in path)
                    or any(not part or part.startswith(".") for part in path.split("/"))
                    or type(sha256) is not str or len(sha256) != 64
                    or any(char not in "0123456789abcdef" for char in sha256)
                    or type(size) is not int or not 0 < size < 1 << 64):
                raise ValueError("invalid long-game preflight pin")
            preflight_args = ["--v3-long-game-preflight-path=" + path,
                              "--v3-long-game-preflight-sha256=" + sha256,
                              "--v3-long-game-preflight-bytes=" + str(size)]
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
        arguments.extend(preflight_args)
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
                             graceful_drain_deadline=(None if cleanup_signal_ns is None
                                                      else _seconds_not_later(cleanup_signal_ns)),
                             on_stop=on_stop, owned_group=self.group,
                             output_handoff=self.handoff,
                             cleanup_request_start=cleanup_request_start,
                             cleanup_request_poll=cleanup_request_poll)
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
