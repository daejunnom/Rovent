"""Selector-driven original-parent admission for an LC0 child.

This transport does not spawn a process, load a model, or prove physical GPU
completion.  The caller must first verify and retain the original child image,
keep the same parent alive, and continue draining its output while advancing
this state machine.  A failed or cancelled admission after bytes may have been
sent remains uncertain and retains its socket; it is not a pre-native proof.
"""
from __future__ import annotations

import errno
import json
import math
import os
import re
import selectors
import socket
import struct
import time

from lc0_parent_preflight import (
    ACK_SCHEMA, ACK_SCOPE, FRAME_LIMIT, REQUEST_SCHEMA, PreflightError,
    _decode, _validate_notice,
)


# A caller accidentally dropping an uncertain admission must not close the
# original parent stream and partial wire state while the process remains alive. This is quarantine,
# not a completion witness, an automatic retry, or an operating-owner recovery
# capability. The owning process must retain its original child and use its own
# authorized observation/recovery path; this list provides no release API.
_DROPPED_UNCERTAIN_ADMISSIONS = []


class NonblockingParentHandshake:
    """One admission attempt; no implicit retry or fallback.

    `notice` is the LC0 producer's original endpoint notice. The caller must
    start only after checking the child's executable identity and must retain
    this object on uncertainty. On authentication, ownership of the original
    stream is transferred exactly once via `take_authenticated_socket()`. The
    consumer must retain and monitor that stream, including bytes arriving
    after the initially validated ACK; a single recv cannot rule out delayed
    trailing data. If `interest()` is zero in `retry_connect`, call `advance`
    on a timer at `retry_at`; avoid a busy loop. Neither the ACK nor this
    handoff proves native completion.
    """

    def __init__(self, notice, *, child_pid, epoch, lock_sha256, ordinal,
                 deadline):
        if (type(child_pid) is not int or child_pid <= 0
                or type(epoch) is not int or not 1 <= epoch <= 2**64 - 1
                or type(ordinal) is not int or not 0 <= ordinal <= 2**31 - 1
                or not isinstance(lock_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", lock_sha256) is None
                or not isinstance(deadline, (int, float))
                or isinstance(deadline, bool) or not math.isfinite(deadline)
                or not deadline > time.monotonic()):
            raise ValueError("invalid LC0 parent admission parameters")
        route = _validate_notice(notice, child_pid, epoch, lock_sha256, ordinal)
        self._pid = child_pid
        self._epoch = epoch
        self._lock = lock_sha256
        self._ordinal = ordinal
        self._uid = os.geteuid()
        self._deadline = float(deadline)
        self._route = route
        self._retry_at = None
        self._socket = None
        self._state = "connecting"
        self._error = None
        self._sent = 0
        self._send_uncertain = False
        self._response = bytearray()
        self._request = bytearray(json.dumps({
            "schema": REQUEST_SCHEMA, "child_pid": child_pid,
            "process_epoch": epoch, "lock_sha256": lock_sha256,
            "cuda_ordinal": ordinal,
        }, separators=(",", ":")).encode("utf-8") + b"\n")
        self._request_wire = bytes(self._request)
        try:
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.setblocking(False)
            self._connect()
        except OSError:
            self._fail("connect_failed")

    def _connect(self):
        status = self._socket.connect_ex(self._route)
        if status == errno.EAGAIN:
            # AF_UNIX nonblocking connect may not start when the listener's
            # backlog is full. Retry only before any admission byte is sent.
            self._state = "retry_connect"
            self._retry_at = min(self._deadline, time.monotonic() + 0.01)
        elif status in (0, errno.EINPROGRESS, errno.EALREADY, errno.EINTR, errno.EISCONN):
            self._state = "connecting"
            self._retry_at = None
        else:
            self._fail("connect_failed")

    @property
    def state(self):
        return self._state

    @property
    def binding(self):
        """Return the identity carried by this original admission attempt."""
        return (self._pid, self._epoch, self._lock, self._ordinal)

    @property
    def error(self):
        return self._error

    @property
    def bytes_sent(self):
        return self._sent

    @property
    def admission_uncertain(self):
        return self._sent > 0 or self._send_uncertain

    @property
    def retry_at(self):
        return self._retry_at if self._state == "retry_connect" else None

    @property
    def authenticated_wire(self):
        """Copies of this attempt's actual request and validated ACK bytes."""
        if self._state != "authenticated" or self._sent != len(self._request_wire):
            raise RuntimeError("LC0 authenticated wire is not available")
        return (self._request_wire, bytes(self._response))

    def fileno(self):
        if self._socket is None:
            return -1
        return self._socket.fileno()

    def interest(self):
        if self._state in ("connecting", "writing"):
            return selectors.EVENT_WRITE
        if self._state == "reading":
            return selectors.EVENT_READ
        return 0

    def _close_before_send(self):
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def _fail(self, code):
        self._error = self._error or code
        if self.admission_uncertain:
            self._state = "failed_after_send"
        else:
            self._state = "failed_before_send"
            self._close_before_send()

    def cancel(self):
        """Stop sending. An already-sent attempt remains owned and uncertain."""
        if self._state not in ("retry_connect", "connecting", "writing", "reading"):
            return self._state
        self._error = self._error or "cancelled"
        if self.admission_uncertain:
            self._state = "cancelled_after_send"
        else:
            self._state = "cancelled_before_send"
            self._close_before_send()
        return self._state

    def advance(self, events):
        """Advance once for selector readiness; never wait for socket I/O."""
        if self._state not in ("retry_connect", "connecting", "writing", "reading"):
            return self._state
        if time.monotonic() >= self._deadline:
            self._fail("deadline_expired")
            return self._state
        try:
            if self._state == "retry_connect" and time.monotonic() >= self._retry_at:
                self._connect()
            if self._state == "connecting" and events & selectors.EVENT_WRITE:
                if self._socket.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR):
                    self._fail("connect_failed")
                    return self._state
                peer = self._socket.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED,
                                               struct.calcsize("3i"))
                peer_pid, peer_uid, _ = struct.unpack("3i", peer)
                if peer_pid != self._pid or peer_uid != self._uid:
                    self._fail("socket_peer_mismatch")
                    return self._state
                self._state = "writing"
            if self._state == "writing" and events & selectors.EVENT_WRITE:
                try:
                    count = self._socket.send(self._request)
                except BlockingIOError:
                    return self._state
                except InterruptedError:
                    # Do not infer whether a native syscall transferred bytes.
                    self._send_uncertain = True
                    self._fail("send_interrupted")
                    return self._state
                except OSError:
                    self._send_uncertain = True
                    self._fail("send_failed")
                    return self._state
                if count <= 0 or count > len(self._request):
                    if count > len(self._request):
                        self._send_uncertain = True
                    self._fail("send_invalid_count")
                    return self._state
                self._sent += count
                del self._request[:count]
                if not self._request:
                    self._state = "reading"
            elif self._state == "reading" and events & selectors.EVENT_READ:
                chunk = self._socket.recv(FRAME_LIMIT - len(self._response))
                if not chunk:
                    self._fail("ack_eof")
                    return self._state
                self._response.extend(chunk)
                if b"\n" in self._response:
                    frame, _, trailing = self._response.partition(b"\n")
                    if trailing:
                        self._fail("ack_trailing_bytes")
                        return self._state
                    ack = _decode(bytes(frame))
                    expected = {"schema": ACK_SCHEMA, "child_pid": self._pid,
                                "process_epoch": self._epoch, "lock_sha256": self._lock,
                                "cuda_ordinal": self._ordinal, "scope": ACK_SCOPE}
                    if ack != expected or any(type(ack[key]) is not int for key in
                                              ("child_pid", "process_epoch", "cuda_ordinal")):
                        self._fail("ack_identity_mismatch")
                    elif time.monotonic() >= self._deadline:
                        self._fail("deadline_expired")
                    else:
                        self._state = "authenticated"
                elif len(self._response) >= FRAME_LIMIT:
                    self._fail("ack_frame_too_large")
        except BlockingIOError:
            pass
        except (OSError, PreflightError, KeyError, struct.error):
            self._fail("transport_or_ack_invalid")
        return self._state

    def take_authenticated_socket(self):
        """Move the original authenticated stream to its long-lived owner."""
        if self._state != "authenticated" or self._socket is None:
            raise RuntimeError("LC0 admission is not available for handoff")
        owned, self._socket = self._socket, None
        self._state = "handed_off"
        return owned

    def __del__(self):
        sock = getattr(self, "_socket", None)
        if sock is None:
            return
        if getattr(self, "_sent", 0) or getattr(self, "_send_uncertain", False):
            _DROPPED_UNCERTAIN_ADMISSIONS.append(self)
        else:
            sock.close()
            self._socket = None
