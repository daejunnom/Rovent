"""First-admission custody for one directly spawned LC0 process.

This is a CPU-side lifecycle component, not an executable entry point. The
caller owns the resource/output lease and must keep this object alive, drain
both pipes, and retain all original errors. A validated first admission does not
prove native entry, process completion, output preservation or a GPU fence.

The held executable FD and a momentary `/proc/PID/exe` inode observation bind
this attempt to the registered bytes at inspection time. They do not prove
source provenance, prevent concurrent same-user writes, or prove that the
child never executes another image later. An uncertain spawn is never retried.
No recovery or forced termination of a stuck child is implemented here.
"""
from __future__ import annotations

import os
import re
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

from lc0_async_output_spool import AsyncOutputSpool
from lc0_parent_observation import NonblockingSessionObservation, OriginalAdmission
from lc0_parent_preflight import (
    FRAME_LIMIT,
    NOTICE_SCHEMA,
    PreflightError,
    _decode,
    _open_pinned_binary,
    _validate_notice,
)
from lc0_parent_transport import NonblockingParentHandshake


_UNCLAIMED_OWNERS = []


class _AttemptSlot:
    """Reserved before a constructor can create or move a socket."""

    def __init__(self):
        self.value = None
        self.failure = None


class _StderrOffer:
    """Retain one original chunk before calling an asynchronous writer."""

    __slots__ = ("index", "chunk", "result", "failure")

    def __init__(self, index, chunk):
        self.index = index
        self.chunk = chunk
        self.result = "offering"
        self.failure = None


class Lc0FirstAdmissionOwner:
    """Single writer for the child, its first handshake, and later controls.

    Construct without side effects, then call `start` once. A failed attempt
    stays owned, including an unknown spawn result. Only stderr bytes read from
    the original Popen pipe may supply the endpoint notice. Successful first
    authentication is transferred directly to this owner's OriginalAdmission;
    callers cannot pass in a pre-existing handshake or admission as the first.

    The caller must arrange a selector and finite deadlines; methods never wait
    for network/pipe readiness. Retained stderr is bounded, and exceeding the
    bound is an explicit observation failure, never an output-success claim.
    The constructor deadline covers only original admission; it is not a
    child-lifetime deadline or a guarantee that the process will terminate.
    """

    def __init__(self, *, binary: Path, binary_bytes: int, binary_sha256: str,
                 native_args: tuple[str, ...], epoch: int, ordinal: int,
                 lock_sha256: str, run_seconds: int, cleanup_seconds: int,
                 deadline: float, max_stderr_bytes: int, max_control_attempts: int):
        if (sys.platform != "linux" or not isinstance(binary, Path) or not binary.is_absolute()
                or type(binary_bytes) is not int or binary_bytes <= 0
                or not isinstance(binary_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", binary_sha256) is None
                or not isinstance(native_args, tuple) or not native_args
                or any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in native_args)
                or any(arg.startswith(("--lc0-parent-session=", "--lc0-parent-preflight="))
                       for arg in native_args)
                or type(epoch) is not int or not 1 <= epoch <= 2**64 - 1
                or type(ordinal) is not int or not 0 <= ordinal <= 2**31 - 1
                or not isinstance(lock_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", lock_sha256) is None
                or type(run_seconds) is not int or run_seconds <= 0
                or type(cleanup_seconds) is not int or cleanup_seconds <= 0
                or type(deadline) not in (int, float) or not time.monotonic() < deadline < float("inf")
                or type(max_stderr_bytes) is not int or max_stderr_bytes <= 0
                or type(max_control_attempts) is not int or max_control_attempts <= 0):
            raise ValueError("invalid LC0 first-admission owner configuration")
        self._binary = binary
        self._binary_bytes = binary_bytes
        self._binary_sha256 = binary_sha256
        self._args = native_args
        self._epoch = epoch
        self._ordinal = ordinal
        self._lock_sha256 = lock_sha256
        self._run_seconds = run_seconds
        self._cleanup_seconds = cleanup_seconds
        self._deadline = float(deadline)
        self._stderr_cap = max_stderr_bytes
        self._max_controls = max_control_attempts
        self._control_capacity_reached = False
        self._state = "new"
        self._binary_fd = None
        self._binary_identity = None
        self._process = None
        self._pidfd = None
        self._spawn_attempted = False
        self._failure = None
        self._cleanup_failure = None
        self._notice = None
        self._handshake = None
        self._pending_admission = None
        self._original_admission = None
        self._controls = []
        self._observations = []
        # Keep the exact immutable objects returned by os.read; the notice
        # parser retains at most one bounded line separately.
        self._stderr_chunks = []
        self._stderr_length = 0
        self._stderr_line = bytearray()
        self._stderr_line_oversized = False
        self._pending_stderr = None
        self._stderr_eof = False
        self._stderr_writer = None
        self._stderr_prefix_token = object()
        self._stderr_offered = 0
        self._stderr_offer = None
        self._stderr_handoff_error = None
        self._stderr_handoff_mark_error = None
        self._stderr_writer_mark = "not_needed"
        self._stderr_handoff_lock = threading.Lock()
        self._image_observations = 0

    @property
    def state(self):
        return self._state

    @property
    def original_failure(self):
        """Original local exception, never print its potentially sensitive text."""
        return self._failure

    @property
    def child_pid(self):
        return None if self._process is None else self._process.pid

    @property
    def cleanup_failure(self):
        return self._cleanup_failure

    def first_admission_poll(self):
        """Selector registration for the one original handshake attempt."""
        if self._state != "first_authentication" or self._handshake is None:
            raise RuntimeError("LC0 first authentication is not active")
        return (self._handshake.fileno(), self._handshake.interest(),
                self._handshake.retry_at)

    @property
    def control_capacity_reached(self):
        return self._control_capacity_reached

    @property
    def stderr_bytes(self):
        """A copy of retained bytes, never a complete-output claim."""
        if self._pending_stderr is not None:
            raise RuntimeError("LC0 pending stderr attachment is unresolved")
        return b"".join(self._stderr_chunks)

    @property
    def pending_stderr(self):
        """The original just-read chunk if attaching it was interrupted."""
        return self._pending_stderr

    def pipe_filenos(self):
        """Borrow original pipe FDs; this is not a transfer.

        The owner alone reads stderr until notice capture is complete. The
        caller must preserve `stderr_bytes` before continuing with that FD.
        """
        if self._process is None:
            raise RuntimeError("LC0 original process is not owned")
        return tuple(stream.fileno() for stream in
                     (self._process.stdin, self._process.stdout, self._process.stderr))

    def release_before_spawn(self):
        """Release only an FD for an attempt that never entered Popen."""
        if self._state != "failed_before_spawn" or self._spawn_attempted:
            raise RuntimeError("LC0 spawn status does not permit pre-spawn release")
        if self._binary_fd is not None:
            try:
                os.close(self._binary_fd)
            except OSError as error:
                self._state = "pre_spawn_cleanup_unknown"
                self._cleanup_failure = error
                return self._state
            self._binary_fd = None
        self._state = "pre_spawn_released"
        return self._state

    def _fail(self, state, error):
        if self._failure is None:
            self._failure = error
        self._state = state

    def start(self):
        if self._state != "new":
            raise RuntimeError("LC0 child start already attempted")
        self._state = "opening"
        try:
            self._binary_fd, self._binary_identity = _open_pinned_binary(
                self._binary, self._binary_bytes, self._binary_sha256, self._deadline)
        except BaseException as error:
            self._fail("failed_before_spawn", error)
            return self._state
        if time.monotonic() >= self._deadline:
            self._fail("failed_before_spawn", TimeoutError("LC0 spawn deadline expired"))
            return self._state
        self._spawn_attempted = True
        self._state = "spawning"
        try:
            argv = [str(self._binary), *self._args,
                    f"--lc0-parent-session={self._epoch}:{self._run_seconds}:{self._cleanup_seconds}"]
            self._process = subprocess.Popen(
                argv, executable=f"/proc/self/fd/{self._binary_fd}",
                pass_fds=(self._binary_fd,), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        except BaseException as error:
            # No usable Popen handle exists. Do not claim the kernel never
            # created a child, close the pin, or start a replacement attempt.
            self._fail("spawn_result_unknown", error)
            return self._state
        self._state = "child_owned"
        try:
            if not hasattr(os, "pidfd_open"):
                raise RuntimeError("LC0 child pidfd capability unavailable")
            self._pidfd = os.pidfd_open(self._process.pid, 0)
            if not hasattr(os, "P_PIDFD") or not hasattr(os, "WNOWAIT"):
                raise RuntimeError("LC0 direct-child pidfd observation unavailable")
            # Non-reaping check: ECHILD is uncertainty, not evidence of exit.
            waited = os.waitid(os.P_PIDFD, self._pidfd, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if waited is not None and waited.si_pid != 0:
                raise PreflightError("child_exit_observed_before_image_check")
            self._observe_child_image()
            os.set_blocking(self._process.stderr.fileno(), False)
        except BaseException as error:
            self._fail("child_observation_unknown", error)
            return self._state
        self._state = "awaiting_first_notice"
        return self._state

    def _observe_child_image(self):
        if self._process is None or self._binary_fd is None or self._pidfd is None:
            raise RuntimeError("LC0 original child is unavailable")
        poller = select.poll()
        poller.register(self._pidfd, select.POLLIN)
        def check_pidfd():
            for _, flags in poller.poll(0):
                if flags & (select.POLLERR | select.POLLNVAL):
                    raise PreflightError("child_pidfd_observation_unknown")
                if flags & (select.POLLIN | select.POLLHUP):
                    raise PreflightError("child_exit_observed_during_image_check")
        check_pidfd()
        pinned = os.fstat(self._binary_fd)
        actual = os.stat(f"/proc/{self._process.pid}/exe")
        if ((pinned.st_dev, pinned.st_ino, pinned.st_size, pinned.st_mtime_ns, pinned.st_ctime_ns)
                != (self._binary_identity.st_dev, self._binary_identity.st_ino,
                    self._binary_identity.st_size, self._binary_identity.st_mtime_ns,
                    self._binary_identity.st_ctime_ns)
                or (actual.st_dev, actual.st_ino) != (pinned.st_dev, pinned.st_ino)):
            raise PreflightError("child_image_identity_mismatch")
        check_pidfd()
        self._image_observations += 1

    def advance_stderr(self):
        """Read one chunk from the original stderr pipe; never wait for data."""
        if self._state != "awaiting_first_notice":
            raise RuntimeError("LC0 first-notice observation is not active")
        if time.monotonic() >= self._deadline:
            self._fail("first_notice_unknown", TimeoutError("LC0 notice deadline expired"))
            return self._state
        remaining = self._stderr_cap - self._stderr_length
        if remaining <= 0:
            self._fail("first_notice_unknown", RuntimeError("LC0 retained stderr bound reached"))
            return self._state
        try:
            chunk = os.read(self._process.stderr.fileno(), min(4096, remaining))
        except BlockingIOError:
            return self._state
        except OSError as error:
            self._fail("first_notice_unknown", error)
            return self._state
        if not chunk:
            self._stderr_eof = True
            self._fail("first_notice_unknown", EOFError("LC0 stderr closed before admission"))
            return self._state
        # Hold the just-read object before any allocation that might fail.
        self._pending_stderr = chunk
        try:
            self._stderr_chunks.append(chunk)
            self._stderr_length += len(chunk)
            self._pending_stderr = None
            self._scan_notice(chunk)
        except BaseException as error:
            self._fail("first_notice_unknown", error)
        return self._state

    def _scan_notice(self, chunk):
        offset = 0
        while offset < len(chunk):
            end = chunk.find(b"\n", offset)
            piece = chunk[offset:] if end < 0 else chunk[offset:end]
            if not self._stderr_line_oversized:
                if len(self._stderr_line) + len(piece) >= FRAME_LIMIT:
                    self._stderr_line_oversized = True
                    self._stderr_line.clear()
                else:
                    self._stderr_line.extend(piece)
            if end < 0:
                return
            offset = end + 1
            line = bytes(self._stderr_line) if not self._stderr_line_oversized else b""
            self._stderr_line.clear()
            self._stderr_line_oversized = False
            # Ignore unrelated/oversized diagnostic lines while retaining raw chunks.
            if not line.startswith(b"{"):
                continue
            try:
                notice = _decode(line)
            except PreflightError:
                continue
            if notice.get("schema") != NOTICE_SCHEMA:
                continue
            _validate_notice(notice, self._process.pid, self._epoch,
                             self._lock_sha256, self._ordinal)
            if time.monotonic() >= self._deadline:
                raise TimeoutError("LC0 first admission deadline expired")
            self._observe_child_image()
            self._notice = notice
            # One attempt is installed before any admission byte is sent.
            self._handshake = NonblockingParentHandshake.__new__(NonblockingParentHandshake)
            self._handshake.__init__(
                notice, child_pid=self._process.pid, epoch=self._epoch,
                lock_sha256=self._lock_sha256, ordinal=self._ordinal,
                deadline=self._deadline)
            self._state = "first_authentication"
            return

    def handoff_stderr_prefix(self, writer):
        """Offer at most one exact original stderr read to the same writer.

        The caller must not read later stderr bytes before ``complete``. An
        exception or unknown offer result is indeterminate: retain the writer,
        attempt and bytes, and never retry that chunk automatically. Acceptance
        is only ownership transfer to the writer, not EOF, durability or a
        completed output receipt. The caller must bind the writer's file to the
        original managed output owner; this component cannot establish that
        identity. ``busy`` means another handoff call owns this boundary and
        this call made no offer. The caller remains responsible for retaining
        and finishing any writer that rejected the first offer, even if a later
        writer is selected. No child/process state is changed here.
        """
        if not self._stderr_handoff_lock.acquire(blocking=False):
            return "busy"
        try:
            return self._handoff_stderr_prefix_owned(writer)
        finally:
            self._stderr_handoff_lock.release()

    def _handoff_stderr_prefix_owned(self, writer):
        if not isinstance(writer, AsyncOutputSpool):
            raise TypeError("LC0 stderr requires its original async output writer")
        if self._process is None:
            raise RuntimeError("LC0 original process is not owned")
        if self._state in ("new", "opening", "spawning", "child_owned", "awaiting_first_notice"):
            raise RuntimeError("LC0 first stderr prefix is still being captured")
        if self._pending_stderr is not None:
            raise RuntimeError("LC0 original stderr read remains unresolved")
        if self._stderr_writer_mark != "not_needed":
            return "indeterminate"
        if not self._stderr_chunks:
            return "complete"
        if self._stderr_writer is None:
            self._stderr_writer = writer
            try:
                if not writer.bind_prefix(self._stderr_prefix_token):
                    self._stderr_writer = None
                    return "unavailable"
            except BaseException as error:
                if self._stderr_handoff_error is None:
                    self._stderr_handoff_error = error
                self._mark_stderr_writer_indeterminate(writer, error)
                return "indeterminate"
        elif self._stderr_writer is not writer:
            raise ValueError("LC0 stderr prefix belongs to another writer")
        attempt = self._stderr_offer
        if attempt is not None and attempt.result == "offering":
            # Interruption can happen after an offer accepted but before the
            # caller received its result. Never issue that chunk a second time.
            attempt.result = "indeterminate"
            self._mark_stderr_writer_indeterminate(
                writer, RuntimeError("LC0 stderr offer result was not observed")
            )
        if attempt is not None and attempt.result == "indeterminate":
            return "indeterminate"
        if attempt is not None and attempt.result == "accepted":
            # A prior call may have been interrupted after receiving accepted.
            # Commit the same attempt without invoking the writer again.
            try:
                self._stderr_offered = attempt.index + 1
                self._stderr_offer = None
            except BaseException as error:
                attempt.failure = error
                if self._stderr_handoff_error is None:
                    self._stderr_handoff_error = error
                return "indeterminate"
        if self._stderr_offered == len(self._stderr_chunks):
            try:
                if not writer.end_prefix(self._stderr_prefix_token):
                    raise RuntimeError("LC0 original stderr prefix release is unknown")
            except BaseException as error:
                if self._stderr_handoff_error is None:
                    self._stderr_handoff_error = error
                self._mark_stderr_writer_indeterminate(writer, error)
                return "indeterminate"
            return "complete"
        chunk = self._stderr_chunks[self._stderr_offered]
        attempt = _StderrOffer(self._stderr_offered, chunk)
        self._stderr_offer = attempt
        try:
            result = writer.offer(chunk, prefix_token=self._stderr_prefix_token)
            if type(result) is not str or result not in ("accepted", "backpressure", "oversized", "unavailable"):
                raise ValueError("LC0 stderr writer returned an unknown offer result")
            attempt.result = result
            if result == "accepted":
                self._stderr_offered = attempt.index + 1
                self._stderr_offer = None
            elif self._stderr_offered == 0:
                # No bytes have been accepted and this was a definitive
                # rejection. A suitably sized original-output writer may be
                # selected later without replaying an accepted chunk.
                if not writer.end_prefix(self._stderr_prefix_token):
                    raise RuntimeError("LC0 rejected stderr prefix release is unknown")
                self._stderr_writer = None
                self._stderr_offer = None
            return result
        except BaseException as error:
            attempt.failure = error
            if self._stderr_handoff_error is None:
                self._stderr_handoff_error = error
            if attempt.result != "accepted":
                attempt.result = "indeterminate"
                self._mark_stderr_writer_indeterminate(writer, error)
            return "indeterminate"

    def _mark_stderr_writer_indeterminate(self, writer, error):
        try:
            result = writer.note_offer_indeterminate(error)
            if result not in ("marked", "too_late"):
                raise ValueError("LC0 stderr writer returned an unknown mark result")
        except BaseException as mark_error:
            if self._stderr_handoff_mark_error is None:
                self._stderr_handoff_mark_error = mark_error
            self._stderr_writer_mark = "unknown"
        else:
            self._stderr_writer_mark = result

    def stderr_prefix_handoff_observation(self):
        """Observe transfer scope, or busy without inferring a partial state."""
        if not self._stderr_handoff_lock.acquire(blocking=False):
            return {"state": "busy"}
        try:
            attempt = self._stderr_offer
            return {
                "state": "observed",
                "original_chunks": len(self._stderr_chunks),
                "accepted_chunks": self._stderr_offered,
                "attempt": None if attempt is None else attempt.result,
                "original_offer_error_retained": self._stderr_handoff_error is not None,
                "writer_indeterminate_mark": self._stderr_writer_mark,
                "original_mark_error_retained": self._stderr_handoff_mark_error is not None,
                "output_eof": "not_established_by_prefix_handoff",
                "durability": "not_established_by_prefix_handoff",
            }
        finally:
            self._stderr_handoff_lock.release()

    @property
    def original_stderr_handoff_error(self):
        """Original offer error for the same owner; never render private text."""
        if not self._stderr_handoff_lock.acquire(blocking=False):
            raise BlockingIOError("LC0 stderr handoff owner is busy")
        try:
            return self._stderr_handoff_error
        finally:
            self._stderr_handoff_lock.release()

    @property
    def original_stderr_writer_mark_error(self):
        if not self._stderr_handoff_lock.acquire(blocking=False):
            raise BlockingIOError("LC0 stderr handoff owner is busy")
        try:
            return self._stderr_handoff_mark_error
        finally:
            self._stderr_handoff_lock.release()

    def advance_first_admission(self, events):
        if self._state != "first_authentication":
            raise RuntimeError("LC0 first authentication is not active")
        state = self._handshake.advance(events)
        if state == "authenticated":
            try:
                self._observe_child_image()
                if time.monotonic() >= self._deadline:
                    raise TimeoutError("LC0 first admission custody deadline expired")
                # Install the actual object before its constructor can move
                # the liveness FD. A partial constructor remains in this owner.
                candidate = OriginalAdmission.__new__(OriginalAdmission)
                self._pending_admission = candidate
                candidate.__init__(self._handshake)
                self._original_admission = candidate
                self._pending_admission = None
            except BaseException as error:
                self._fail("first_admission_unknown", error)
                return self._state
            self._state = "admitted"
        elif state.startswith(("failed_", "cancelled_")):
            self._fail("first_admission_unknown", self._handshake.original_error
                       or RuntimeError(self._handshake.error or "LC0 admission failed"))
        return self._state

    def new_control_handshake(self, deadline):
        """Create only a later control attempt bound to this first owner."""
        if self._state != "admitted" or self._original_admission is None:
            raise RuntimeError("LC0 first admission is not owned")
        if len(self._controls) >= self._max_controls:
            self._control_capacity_reached = True
            # This local observation capacity does not change the engine,
            # original admission, or any independent execution's status.
            raise RuntimeError("LC0 original owner control capacity reached")
        slot = _AttemptSlot()
        self._controls.append(slot)
        try:
            # The pidfd stays pinned, while image identity is only a fresh
            # instant observation, not a no-future-exec guarantee.
            self._observe_child_image()
            control = NonblockingParentHandshake.__new__(NonblockingParentHandshake)
            slot.value = control
            control.__init__(
                self._notice, child_pid=self._process.pid, epoch=self._epoch,
                lock_sha256=self._lock_sha256, ordinal=self._ordinal, deadline=deadline)
        except BaseException as error:
            slot.failure = error
            raise
        return control

    def new_observation(self, control, *, sequence, deadline):
        if (self._state != "admitted" or not isinstance(control, NonblockingParentHandshake)
                or not any(control is item.value for item in self._controls)):
            raise RuntimeError("LC0 control is not from this original owner")
        if len(self._observations) >= self._max_controls:
            self._control_capacity_reached = True
            raise RuntimeError("LC0 original owner observation capacity reached")
        slot = _AttemptSlot()
        self._observations.append(slot)
        try:
            self._observe_child_image()
            observation = NonblockingSessionObservation.__new__(NonblockingSessionObservation)
            slot.value = observation
            observation.__init__(
                self._original_admission, control, sequence=sequence, deadline=deadline)
        except BaseException as error:
            slot.failure = error
            raise
        return observation

    def __del__(self):
        if (getattr(self, "_binary_fd", None) is not None
                or getattr(self, "_pidfd", None) is not None
                or getattr(self, "_process", None) is not None
                or getattr(self, "_spawn_attempted", False)):
            _UNCLAIMED_OWNERS.append(self)
