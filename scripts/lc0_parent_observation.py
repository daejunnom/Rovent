"""Nonblocking, read-only V2 observation of an original LC0 session.

This consumes a *separately authenticated control connection*. The caller
must keep the initial parent-admission stream and process owner alive. An
observation grants no stop, exit, physical-completion or output-acceptance
authority. The V2 server is available only after UCI service returns.
"""
from __future__ import annotations

import json
import math
import os
import selectors
import time
from copy import deepcopy

from lc0_parent_preflight import FRAME_LIMIT, PreflightError, _decode
from lc0_parent_transport import NonblockingParentHandshake

REQUEST_SCHEMA = "rz-lc0-system-observation/2"
RESPONSE_SCHEMA = "rz-lc0-system-observation-response/2"
SESSION_SCHEMA = "rz-lc0-original-session-observation/1"
_UNCLAIMED_CUSTODIES = []


class OriginalAdmission:
    """Retain the first authenticated parent stream for one original child.

    The future process owner must construct this from the actual first
    admission, keep its ordering and process identity, and retain it until
    that original lifetime is resolved. This class cannot prove that a caller
    supplied the first handshake, nor does it imply native entry or completion.
    """

    def __init__(self, handshake):
        if not isinstance(handshake, NonblockingParentHandshake) or handshake.state != "authenticated":
            raise ValueError("original LC0 admission is not authenticated")
        self._binding = handshake.binding
        self.authenticated_wire = handshake.authenticated_wire
        self._socket = handshake.take_authenticated_socket()
        try:
            st = os.fstat(self._socket.fileno())
            self._identity = (st.st_dev, st.st_ino)
        except BaseException:
            _UNCLAIMED_CUSTODIES.append(self)
            raise

    def check_control(self, handshake):
        if (not isinstance(handshake, NonblockingParentHandshake)
                or handshake.state != "authenticated" or handshake.binding != self.binding):
            raise ValueError("control admission differs from the original")
        original = os.fstat(self._socket.fileno())
        control = os.fstat(handshake.fileno())
        if ((original.st_dev, original.st_ino) != self._identity
                or (control.st_dev, control.st_ino) == self._identity):
            raise ValueError("control must be distinct from the original admission socket")

    @property
    def binding(self):
        return self._binding

    def __del__(self):
        if getattr(self, "_socket", None) is not None:
            _UNCLAIMED_CUSTODIES.append(self)


class ControlCustody:
    """Retain a control socket together with its exact partial wire state."""

    def __init__(self, owner):
        self.socket = owner._socket
        self.original_admission = owner._admission
        self.binding = owner._binding
        self.control_admission_wire = owner._control_admission_wire
        self.sequence = owner._sequence
        self.state = owner._state
        self.error = owner._error
        self.bytes_sent = owner._sent
        self.send_uncertain = owner._send_uncertain
        self.unsent_request = bytes(owner._request)
        self.partial_response = bytes(owner._response)
        self.observation = deepcopy(owner._value)

    def __del__(self):
        if getattr(self, "socket", None) is not None:
            _UNCLAIMED_CUSTODIES.append(self)


def _matches(response, binding, sequence):
    pid, epoch, lock_sha256, ordinal = binding
    if (not isinstance(response, dict) or response.get("schema") != RESPONSE_SCHEMA
            or type(response.get("sequence")) is not int or response["sequence"] != sequence
            or response.get("observation") not in ("snapshot_before_authentication", "historical_replay")):
        return False
    session = response.get("session")
    publications = ("runtime_start_receipt", "runtime_finish_receipt", "original_failure_receipt",
                    "finish_receipt", "reobservation_receipt", "endpoint_retirement")
    return (
        isinstance(session, dict) and session.get("schema") == SESSION_SCHEMA
        and type(session.get("process_id")) is int and session["process_id"] == pid
        and type(session.get("process_epoch")) is int and session["process_epoch"] == epoch
        and session.get("lock_sha256") == lock_sha256
        and type(session.get("cuda_ordinal")) is int and session["cuda_ordinal"] == ordinal
        and type(session.get("ort_entry_observed")) is bool
        and type(session.get("scope_settled")) is bool
        and session.get("service") in ("not_started", "serving", "succeeded", "failed", "unknown")
        and session.get("original_release") in ("not_observed", "pending", "released", "failed")
        and session.get("latest_release") in ("not_observed", "pending", "released", "failed")
        and all(session.get(key) in ("not_attempted", "succeeded", "failed", "mixed")
                for key in publications)
        and session.get("output_preservation") == "unknown"
        and session.get("physical_completion") == "unknown"
    )


class NonblockingSessionObservation:
    """One V2 exchange on the same authenticated control socket.

    If this owner is dropped, its socket is retained, never automatically
    retried or converted into physical-completion evidence. The consumer must
    monitor the handed-back stream for later bytes or disconnection.
    """

    def __init__(self, original_admission, handshake, *, sequence, deadline):
        if (not isinstance(original_admission, OriginalAdmission)
                or not isinstance(handshake, NonblockingParentHandshake)
                or handshake.state != "authenticated"
                or type(sequence) is not int or not 1 <= sequence <= 2**64 - 1
                or not isinstance(deadline, (int, float)) or isinstance(deadline, bool)
                or not math.isfinite(deadline) or deadline <= time.monotonic()):
            raise ValueError("invalid LC0 observation admission")
        original_admission.check_control(handshake)
        self._admission = original_admission
        self._control_admission_wire = handshake.authenticated_wire
        self._binding = handshake.binding
        self._sequence = sequence
        self._deadline = float(deadline)
        self._response = bytearray()
        self._value = None
        self._state = "writing"
        self._error = None
        self._sent = 0
        self._send_uncertain = False
        self._request = bytearray(json.dumps({
            "schema": REQUEST_SCHEMA, "sequence": sequence,
            "process_epoch": self._binding[1], "lock_sha256": self._binding[2],
            "cuda_ordinal": self._binding[3],
        }, separators=(",", ":")).encode("utf-8") + b"\n")
        self._socket = handshake.take_authenticated_socket()

    @property
    def state(self):
        return self._state

    @property
    def error(self):
        return self._error

    @property
    def bytes_sent(self):
        return self._sent

    @property
    def send_uncertain(self):
        return self._send_uncertain

    @property
    def observation(self):
        # A caller cannot mutate the retained observation through the returned
        # value. It is a historical snapshot, not a completion/exit decision.
        return deepcopy(self._value)

    def fileno(self):
        return -1 if self._socket is None else self._socket.fileno()

    def interest(self):
        if self._state == "writing":
            return selectors.EVENT_WRITE
        if self._state == "reading":
            return selectors.EVENT_READ
        return 0

    def _fail(self, reason):
        self._error = self._error or reason
        self._state = "failed_after_send" if self._sent or self._send_uncertain else "failed_before_send"

    def cancel(self):
        if self._state in ("writing", "reading"):
            self._error = self._error or "cancelled"
            self._state = "cancelled_after_send" if self._sent or self._send_uncertain else "cancelled_before_send"
        return self._state

    def advance(self, events):
        if self._state not in ("writing", "reading"):
            return self._state
        if time.monotonic() >= self._deadline:
            self._fail("deadline_expired")
            return self._state
        try:
            if self._state == "writing" and events & selectors.EVENT_WRITE:
                try:
                    count = self._socket.send(self._request)
                except BlockingIOError:
                    return self._state
                except (InterruptedError, OSError):
                    self._send_uncertain = True
                    self._fail("send_unknown")
                    return self._state
                if count <= 0 or count > len(self._request):
                    self._send_uncertain = count > len(self._request)
                    self._fail("send_invalid_count")
                    return self._state
                self._sent += count
                del self._request[:count]
                if not self._request:
                    self._state = "reading"
            elif self._state == "reading" and events & selectors.EVENT_READ:
                chunk = self._socket.recv(FRAME_LIMIT - len(self._response))
                if not chunk:
                    self._fail("response_eof")
                    return self._state
                self._response.extend(chunk)
                if b"\n" in self._response:
                    frame, _, trailing = self._response.partition(b"\n")
                    if trailing:
                        self._fail("response_trailing_bytes")
                        return self._state
                    response = _decode(bytes(frame))
                    if time.monotonic() >= self._deadline:
                        self._fail("deadline_expired")
                    elif not _matches(response, self._binding, self._sequence):
                        self._fail("response_identity_or_scope_mismatch")
                    else:
                        self._value = response
                        self._state = "observed"
                elif len(self._response) >= FRAME_LIMIT:
                    self._fail("response_too_large")
        except BlockingIOError:
            pass
        except (OSError, PreflightError):
            self._fail("transport_or_response_invalid")
        return self._state

    def take_control_custody(self):
        """Move the exact stream and all partial wire evidence together."""
        if self._socket is None or self._state in ("writing", "reading"):
            raise RuntimeError("LC0 observation stream is not ready for handoff")
        owned = ControlCustody(self)
        self._socket = None
        self._state = "handed_off"
        return owned

    def __del__(self):
        sock = getattr(self, "_socket", None)
        if sock is not None:
            # Preserve partial frames and the original admission, not only FD.
            _UNCLAIMED_CUSTODIES.append(self)
