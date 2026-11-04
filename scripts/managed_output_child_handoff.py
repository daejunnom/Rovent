"""Original Linux output-lease handoff to one supervised collector child.

The inherited descriptors alone are not a lease credential. The original
supervisor must keep this object and service each authenticated request while
it owns the same live lease and direct process group. This channel does not
establish native/GPU completion or authorize lease finalization.
"""
from __future__ import annotations

import json
import array
import hashlib
from contextlib import contextmanager
import os
from pathlib import Path
import socket
import stat
import struct
import sys

from managed_outputs import original_live_handoff
from managed_process import OwnedProcessGroup
from rovent_cgroup_owner import CgroupOwner, _membership
from managed_storage import CatalogBusy, StorageError

_REQUEST = b"rz-managed-output-lease/1"
_CREDENTIALS = struct.Struct("3i")
_MAX_PACKET = 4096


@contextmanager
def _original_guard(lease):
    """Temporarily verify the same suspended lease without full inventory."""
    with lease._live_gate:
        borrowed = lease._live_suspended
        if borrowed:
            if lease._live_guard_proof != "released" or lease.guard.stream is not None:
                raise StorageError("original live catalog guard state is unknown")
            lease._live_guard_proof = "unknown"
            try:
                lease.guard.acquire()
            except CatalogBusy:
                lease._live_guard_proof = "released"
                raise
            lease._live_guard_proof = "held"
        primary = None
        try:
            lease._check_live_handoff()
            yield lease._live_anchor
        except BaseException as error:
            primary = error
            raise
        finally:
            if borrowed:
                try:
                    lease._live_guard_proof = "unknown"
                    lease.guard.release()
                    lease._live_guard_proof = "released"
                except BaseException as error:
                    if lease._live_guard_release_error is None:
                        lease._live_guard_release_error = error
                    if primary is not None:
                        primary.add_note("original child guard release also failed: " + str(error))
                    else:
                        raise


class ManagedOutputChildHandoff:
    """Process-local issuer bound to an existing lease and one child group.

    Call ``lease.prepare_live_handoff()`` first, pass ``child_fds`` only to
    the corresponding ``run_group(..., owned_group=group, output_handoff=self)``
    child, and admit that child through the original cgroup barrier. The
    process runner calls ``after_spawn`` and ``service`` on this same owner.
    Retain this issuer and lease until the original group has been drained.
    Closing it revokes further checks, not the lease or physical work.
    """

    def __init__(self, lease_key, group, scope):
        if (sys.platform != "linux" or type(group) is not OwnedProcessGroup
                or type(scope) is not CgroupOwner):
            raise StorageError("Linux original managed group and cgroup scope required")
        group._assert_available()
        scope._check_owner()
        self._key = lease_key
        self._group = group
        self._scope = scope
        self._scope_bound = False
        self._owner_pid = os.getpid()
        self._parent = self._child = None
        self._directory_fd = None
        self._spawned = False
        self._dispatch_attempted = False
        self._peer_closed = False
        self._pending_request = None
        self._next_sequence = 1
        self._admission_closed = False
        lease = original_live_handoff(lease_key)
        self._lease = lease
        try:
            scope.reserve_output_group(group)
            with _original_guard(lease) as anchor:
                if anchor is None or lease._live_child_handoff is not None:
                    raise StorageError("original output lease already has a child handoff")
                self._directory_fd = os.dup(anchor["output_fd"])
                info = os.fstat(self._directory_fd)
                if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != anchor["output_id"]:
                    raise StorageError("original output descriptor identity changed")
            self._parent, self._child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            self._parent.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
            self._parent.setblocking(False)
            child_info = os.fstat(self._child.fileno())
            self._child_socket_id = (child_info.st_dev, child_info.st_ino)
            with _original_guard(lease):
                if lease._live_child_handoff is not None:
                    raise StorageError("original output lease already has a child handoff")
                lease._live_child_handoff = self
        except BaseException as error:
            try:
                self.revoke()
            except BaseException as close_error:
                error.add_note("handoff cleanup also failed: " + str(close_error))
            raise

    @property
    def child_fds(self):
        if self._spawned or self._child is None or self._directory_fd is None:
            raise StorageError("original child handoff is no longer available")
        return self._child.fileno(), self._directory_fd

    def after_spawn(self):
        if (os.getpid() != self._owner_pid or self._spawned or not self._dispatch_attempted or self._group._child is None
                or self._group._owner_pid != self._owner_pid or self._group._completed):
            raise StorageError("original direct child is not bound")
        self._spawned = True
        self._child.close()
        self._child = None
        os.close(self._directory_fd)
        self._directory_fd = None

    def mark_dispatch_attempted(self):
        if (os.getpid() != self._owner_pid or self._dispatch_attempted or self._spawned
                or self._parent is None or self._child is None or self._directory_fd is None):
            raise StorageError("original managed output dispatch is unavailable")
        self._group._assert_available()
        lease = original_live_handoff(self._key)
        with _original_guard(lease) as anchor:
            if anchor is None or lease._live_child_handoff is not self:
                raise StorageError("original output child owner changed before dispatch")
            output_info = os.fstat(self._directory_fd)
            child_info = os.fstat(self._child.fileno())
            if (not stat.S_ISDIR(output_info.st_mode)
                    or (output_info.st_dev, output_info.st_ino) != anchor["output_id"]
                    or (child_info.st_dev, child_info.st_ino) != self._child_socket_id
                    or self._child.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_SEQPACKET):
                raise StorageError("original output child descriptors changed before dispatch")
        # The original scope records this first, so interruption before the
        # local flag cannot authorize an unsafe unstarted release.
        self._scope.mark_output_dispatch_attempted(self._group)
        self._dispatch_attempted = True

    def close_admission(self):
        if os.getpid() != self._owner_pid:
            raise StorageError("original managed output supervisor changed")
        self._admission_closed = True

    def confirm_scope_admission(self):
        """Bind this group to the same scope's completed barrier grant."""
        if (not self._spawned or self._group._child is None
                or self._group._owner_pid != self._owner_pid
                or not self._scope.confirms_granted_child(self._group._child.pid)):
            raise StorageError("original collector barrier grant is not confirmed")
        self._scope_bound = True

    def assert_finish_allowed(self):
        if os.getpid() != self._owner_pid or self._group._owner_pid != self._owner_pid:
            raise StorageError("original managed output child owner changed")
        scope_attempted = self._scope.output_dispatch_attempted(self._group)
        if (scope_attempted and self._group._completed and not self._scope_bound
                and self._group._child is not None
                and self._scope.confirms_granted_child(self._group._child.pid)):
            # on_spawn may have failed after the original barrier grant.
            self._scope_bound = True
        if (self._group._completed and self._scope_bound
                and self._scope.close_admission_and_observe_empty(self._group)):
            return
        if not scope_attempted and self._group._child is None:
            return
        raise StorageError("original managed output child process group is not confirmed gone")

    def release_after_lease_finished(self):
        """Release only the scope reserved for this recorded terminal lease."""
        if self._lease._live_finish_receipt is None:
            raise StorageError("original output terminal receipt is unavailable")
        if self._scope.owns_output_group(self._group):
            if not self._scope.output_dispatch_attempted(self._group):
                self._scope.release_unstarted_output_group(self._group)
            else:
                self._scope.release_finished_output_group(self._group)
        self.revoke()
        with self._lease._live_gate:
            if self._lease._live_child_handoff is self:
                self._lease._live_child_handoff = None

    def scope_release_observation(self):
        """Observe this scope reservation separately from output/GPU closure."""
        if self._lease._live_finish_receipt is None:
            return "terminal_not_confirmed"
        try:
            return "retained" if self._scope.owns_output_group(self._group) else "released"
        except Exception:
            return "unknown"

    def service(self):
        """Serve at most one request without waiting for an idle child."""
        if (os.getpid() != self._owner_pid or not self._spawned or self._parent is None
                or self._group._owner_pid != self._owner_pid or self._group._completed
                or self._group._child is None):
            raise StorageError("original output child supervisor is unavailable")
        if self._peer_closed:
            return False
        if self._admission_closed:
            return False
        child = self._group._child
        if self._pending_request is None:
            try:
                packet, ancillary, flags, _ = self._parent.recvmsg(
                    _MAX_PACKET + 1, socket.CMSG_SPACE(_CREDENTIALS.size) + socket.CMSG_SPACE(16 * array.array("i").itemsize),
                    socket.MSG_DONTWAIT | socket.MSG_CMSG_CLOEXEC)
            except BlockingIOError:
                return False
            unexpected = False
            close_failure = None
            for level, kind, data in ancillary:
                if (level == socket.SOL_SOCKET and
                        (kind == socket.SCM_RIGHTS or
                         (hasattr(socket, "SCM_PIDFD") and kind == socket.SCM_PIDFD))):
                    unexpected = True
                    received_fds = array.array("i")
                    received_fds.frombytes(data[:len(data) - (len(data) % received_fds.itemsize)])
                    for fd in received_fds:
                        try:
                            os.close(fd)
                        except OSError as error:
                            if close_failure is None:
                                close_failure = error
                elif (level, kind) != (socket.SOL_SOCKET, socket.SCM_CREDENTIALS):
                    unexpected = True
            # Linux recvmsg can echo MSG_CMSG_CLOEXEC in msg_flags even for
            # EOF. A real zero-byte datagram on this SO_PASSCRED socket still
            # carries sender credentials and must be rejected as a packet.
            semantic_flags = flags & ~socket.MSG_CMSG_CLOEXEC
            if not packet and not ancillary and not semantic_flags:
                # Closure grants no output or physical-completion proof.
                self._peer_closed = True
                return False
            if close_failure is not None:
                raise StorageError("unexpected received descriptor close failed") from close_failure
            if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or unexpected or not packet or len(packet) > _MAX_PACKET:
                raise StorageError("invalid managed output lease request")
            credentials = [data for level, kind, data in ancillary
                           if level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS]
            if len(credentials) != 1 or len(credentials[0]) != _CREDENTIALS.size:
                raise StorageError("missing original child credentials")
            pid, uid, _ = _CREDENTIALS.unpack(credentials[0])
            if pid != child.pid or uid != os.geteuid():
                raise StorageError("managed output request is not from the original child")
            try:
                def unique(pairs):
                    result = {}
                    for key, value in pairs:
                        if key in result:
                            raise ValueError("duplicate field")
                        result[key] = value
                    return result
                request = json.loads(packet.decode("utf-8"), object_pairs_hook=unique,
                                     parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite")))
            except (ValueError, UnicodeError, RecursionError) as error:
                raise StorageError("invalid managed output request encoding") from error
            self._pending_request = self._validate_request(request)
        if os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
            raise StorageError("original collector has already exited")
        self._scope._check_owner()
        expected = "/" + str(self._scope.workload.relative_to("/sys/fs/cgroup"))
        membership = _membership(str(child.pid))
        if (not self._scope_bound or not self._scope.confirms_granted_child(child.pid)
                or not (membership == expected or membership.startswith(expected + "/"))):
            raise StorageError("original collector cgroup admission is not confirmed")
        lease = original_live_handoff(self._key)
        try:
            guard = _original_guard(lease)
            with guard as anchor:
                if anchor is None or lease._live_child_handoff is not self:
                    raise StorageError("original output lease child owner changed")
                info = os.fstat(anchor["output_fd"])
                if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != anchor["output_id"]:
                    raise StorageError("original output descriptor changed")
                reply = json.dumps(self._reply(info),
                                   sort_keys=True, separators=(",", ":")).encode("ascii")
        except CatalogBusy:
            # Keep the authenticated request; never grant based on stale rows.
            return False
        # Do not send success before the borrowed original guard was released.
        # The selected handoff itself prevents lease finish until this group
        # and its bound workload are confirmed gone.
        if os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
            raise StorageError("original collector exited before output reply")
        try:
            written = self._parent.send(reply, socket.MSG_DONTWAIT)
        except BlockingIOError as error:
            raise StorageError("managed output reply could not be delivered") from error
        if written != len(reply):
            raise StorageError("managed output reply was incomplete")
        self._next_sequence += 1
        self._pending_request = None
        return True

    def _validate_request(self, request):
        if (not isinstance(request, dict) or set(request) != {"schema", "sequence"}
                or request["schema"] != _REQUEST.decode("ascii")
                or type(request["sequence"]) is not int or request["sequence"] != self._next_sequence
                or not 0 < request["sequence"] < 1 << 64):
            raise StorageError("managed output request sequence or schema differs")
        return request["sequence"]

    def _reply(self, info):
        return {"schema": _REQUEST.decode("ascii"), "sequence": self._pending_request,
                "owner_pid": self._owner_pid, "device": info.st_dev, "inode": info.st_ino}

    def revoke(self):
        """Close the control channel and local duplicate; preserve the lease."""
        failure = None
        for name in ("_child", "_parent"):
            channel = getattr(self, name, None)
            if channel is not None:
                setattr(self, name, None)
                try:
                    channel.close()
                except OSError as error:
                    if failure is None:
                        failure = error
        fd, self._directory_fd = getattr(self, "_directory_fd", None), None
        if fd is not None:
            try:
                os.close(fd)
            except OSError as error:
                if failure is None:
                    failure = error
        # An unattempted handoff has never left this process. Remove only its
        # own optional lease binding so a constructor rollback can be retried.
        if not getattr(self, "_dispatch_attempted", False) and self._group._child is None:
            try:
                if self._scope.owns_output_group(self._group):
                    self._scope.release_unstarted_output_group(self._group)
            except BaseException as error:
                if failure is None:
                    failure = error
            else:
                try:
                    lease = original_live_handoff(self._key)
                    with lease._live_gate:
                        if lease._live_child_handoff is self:
                            lease._live_child_handoff = None
                except StorageError:
                    # A finished/unavailable original lease is not recreated.
                    pass
        if failure is not None:
            raise failure


class ManagedCollectionOutputHandoffV2(ManagedOutputChildHandoff):
    """Optional original plan and output lease handoff to the same child.

    The input owner must outlive the child. This proves neither exclusive
    access to the input nor physical GPU completion or writer eligibility.
    """

    _schema = "rz-managed-output-lease/2"

    def __init__(self, lease_key, group, scope, original_input):
        from rovent_v3_input_snapshot import OriginalControlInput, unique_json
        if type(original_input) is not OriginalControlInput:
            raise StorageError("original control input owner required")
        try:
            plan = unique_json(original_input.bytes)
            output = plan["output_directory"]
            if (not isinstance(plan, dict)
                    or plan.get("schema") not in ("rz-rovent-collection-lock-v3-launch/1",
                                                   "rz-rovent-own-collection-lock-v3-launch/1")
                    or not isinstance(output, str) or not output):
                raise ValueError("invalid collection plan")
            declared_output = Path(output)
            if (not declared_output.is_absolute() or ".." in declared_output.parts
                    or declared_output != original_live_handoff(lease_key).output):
                raise ValueError("collection output differs from original lease")
        except (KeyError, TypeError, ValueError, UnicodeError, RecursionError, RuntimeError) as error:
            raise StorageError("invalid original collection plan or output lease") from error
        self._input = original_input
        self._input_directory_fd = self._input_plan_fd = None
        self._session_id = None
        try:
            directory_fd, plan_fd = original_input.borrowed_fds()
            self._input_directory_fd = os.dup(directory_fd)
            self._input_plan_fd = os.dup(plan_fd)
            super().__init__(lease_key, group, scope)
        except BaseException as error:
            try:
                self._close_input_duplicates()
            except BaseException as close_error:
                error.add_note("input handoff cleanup also failed: " + str(close_error))
            raise

    @property
    def child_fds(self):
        if self._input_directory_fd is None or self._input_plan_fd is None:
            raise StorageError("original input child handoff is no longer available")
        return super().child_fds + (self._input_directory_fd, self._input_plan_fd)

    def mark_dispatch_attempted(self):
        directory_fd, plan_fd = self._input.borrowed_fds()
        for source, child in ((directory_fd, self._input_directory_fd), (plan_fd, self._input_plan_fd)):
            before, duplicate = os.fstat(source), os.fstat(child)
            if (before.st_dev, before.st_ino) != (duplicate.st_dev, duplicate.st_ino):
                raise StorageError("original input descriptor changed before dispatch")
        super().mark_dispatch_attempted()

    def after_spawn(self):
        super().after_spawn()
        self._close_input_duplicates()

    def _validate_request(self, request):
        if (not isinstance(request, dict) or set(request) != {"schema", "sequence", "session_id"}
                or request["schema"] != self._schema
                or type(request["sequence"]) is not int or request["sequence"] != self._next_sequence
                or not 0 < request["sequence"] < 1 << 64
                or not isinstance(request["session_id"], str)
                or len(request["session_id"]) != 32
                or any(c not in "0123456789abcdef" for c in request["session_id"])):
            raise StorageError("managed collection request sequence or schema differs")
        if self._session_id is None:
            self._session_id = request["session_id"]
        elif self._session_id != request["session_id"]:
            raise StorageError("managed collection session changed")
        return request["sequence"]

    def _reply(self, info):
        directory_fd, plan_fd = self._input.borrowed_fds()
        data = self._input.bytes
        directory, plan = os.fstat(directory_fd), os.fstat(plan_fd)
        return {"schema": self._schema, "sequence": self._pending_request,
                "session_id": self._session_id, "owner_pid": self._owner_pid,
                "output": {"device": info.st_dev, "inode": info.st_ino},
                "input_directory": {"device": directory.st_dev, "inode": directory.st_ino},
                "plan": {"device": plan.st_dev, "inode": plan.st_ino,
                         "path": self._input.selector,
                         "size": plan.st_size, "sha256": hashlib.sha256(data).hexdigest()}}

    def _close_input_duplicates(self):
        failure = None
        for field in ("_input_plan_fd", "_input_directory_fd"):
            fd = getattr(self, field, None)
            if fd is not None:
                setattr(self, field, None)
                try:
                    os.close(fd)
                except OSError as error:
                    if failure is None:
                        failure = error
        if failure is not None:
            raise failure

    def revoke(self):
        failure = None
        try:
            super().revoke()
        except BaseException as error:
            failure = error
        try:
            self._close_input_duplicates()
        except BaseException as error:
            if failure is not None:
                failure.add_note("input handoff cleanup also failed: " + str(error))
            else:
                failure = error
        if failure is not None:
            raise failure
