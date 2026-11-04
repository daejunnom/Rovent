"""CPU-only V2 frame handling; actual peer admission is tested separately."""
import hashlib
import errno
import json
from pathlib import Path
import selectors
import socket
import sys
import time
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lc0_parent_observation import (
    FRAME_LIMIT, NonblockingSessionObservation, OriginalAdmission, RESPONSE_SCHEMA, SESSION_SCHEMA,
)
from lc0_parent_transport import NonblockingParentHandshake


class FakeControlSocket:
    def __init__(self, owned, width=7):
        self.owned = owned
        self.width = width
        self.sent = bytearray()
        self.received = []
        self.send_error = None

    def send(self, value):
        if self.send_error is not None:
            raise self.send_error
        count = min(self.width, len(value))
        self.sent.extend(value[:count])
        return count

    def recv(self, _maximum):
        if not self.received:
            raise BlockingIOError()
        return self.received.pop(0)

    def fileno(self):
        return self.owned.fileno()


class SessionObservationTests(unittest.TestCase):
    def setUp(self):
        self.binding = (41, 7, hashlib.sha256(b"test lock").hexdigest(), 0)
        self.original_socket, self.original_peer = socket.socketpair()
        self.control_socket, self.control_peer = socket.socketpair()
        self.socket = FakeControlSocket(self.control_socket)
        # This fixture deliberately isolates frame handling. It does not prove
        # real peer authentication; the handshake suite covers that boundary.
        original = object.__new__(NonblockingParentHandshake)
        original._pid, original._epoch, original._lock, original._ordinal = self.binding
        original._state, original._socket = "authenticated", self.original_socket
        original._request_wire, original._response, original._sent = b"request\n", b"ack\n", 8
        self.admission = OriginalAdmission(original)
        handshake = object.__new__(NonblockingParentHandshake)
        handshake._pid, handshake._epoch, handshake._lock, handshake._ordinal = self.binding
        handshake._state, handshake._socket = "authenticated", self.socket
        handshake._request_wire, handshake._response, handshake._sent = b"control\n", b"ack\n", 8
        self.observe = NonblockingSessionObservation(self.admission, handshake, sequence=1, deadline=time.monotonic() + 2)

    def tearDown(self):
        if self.observe.state in ("writing", "reading"):
            self.observe.cancel()
        custody = self.observe.take_control_custody()
        custody.socket = None
        self.control_socket.close()
        self.control_peer.close()
        self.admission._socket.close()
        self.admission._socket = None
        self.original_peer.close()

    def request_sent(self):
        while self.observe.state == "writing":
            self.observe.advance(selectors.EVENT_WRITE)

    def response(self):
        pid, epoch, lock_sha256, ordinal = self.binding
        session = {"schema": SESSION_SCHEMA, "process_id": pid, "process_epoch": epoch,
                   "lock_sha256": lock_sha256, "cuda_ordinal": ordinal,
                   "ort_entry_observed": False, "scope_settled": True,
                   "service": "not_started", "original_release": "not_observed",
                   "latest_release": "not_observed", "output_preservation": "unknown",
                   "physical_completion": "unknown"}
        for field in ("runtime_start_receipt", "runtime_finish_receipt", "original_failure_receipt",
                      "finish_receipt", "reobservation_receipt", "endpoint_retirement"):
            session[field] = "not_attempted"
        return {"schema": RESPONSE_SCHEMA, "sequence": 1,
                "observation": "snapshot_before_authentication", "session": session}

    def test_partial_exchange_preserves_original_snapshot(self):
        self.request_sent()
        frame = json.dumps(self.response()).encode() + b"\n"
        self.socket.received = [frame[:9], frame[9:]]
        self.observe.advance(selectors.EVENT_READ)
        self.assertEqual(self.observe.state, "reading")
        self.observe.advance(selectors.EVENT_READ)
        self.assertEqual(self.observe.state, "observed")
        snapshot = self.observe.observation
        snapshot["session"]["physical_completion"] = "complete"
        self.assertEqual(self.observe.observation["session"]["physical_completion"], "unknown")

    def test_claimed_physical_completion_is_rejected(self):
        self.request_sent()
        value = self.response()
        value["session"]["physical_completion"] = "complete"
        self.socket.received = [json.dumps(value).encode() + b"\n"]
        self.observe.advance(selectors.EVENT_READ)
        self.assertEqual(self.observe.state, "failed_after_send")
        self.assertIsNone(self.observe.observation)

    def test_trailing_frame_is_not_accepted_as_one_response(self):
        self.request_sent()
        self.socket.received = [json.dumps(self.response()).encode() + b"\nextra"]
        self.observe.advance(selectors.EVENT_READ)
        self.assertEqual(self.observe.error, "response_trailing_bytes")
        self.assertEqual(self.observe.state, "failed_after_send")

    def test_maximum_frame_body_with_newline_is_accepted(self):
        self.request_sent()
        body = json.dumps(self.response()).encode()
        body += b" " * (FRAME_LIMIT - 1 - len(body))
        self.socket.received = [body + b"\n"]
        self.observe.advance(selectors.EVENT_READ)
        self.assertEqual(self.observe.state, "observed")

    def test_unterminated_body_at_limit_is_rejected(self):
        self.request_sent()
        self.socket.received = [b"x" * FRAME_LIMIT]
        self.observe.advance(selectors.EVENT_READ)
        self.assertEqual(self.observe.error, "response_too_large")

    def test_cancel_does_not_transfer_an_inflight_request(self):
        with self.assertRaises(RuntimeError):
            self.observe.take_control_custody()
        self.observe.advance(selectors.EVENT_WRITE)
        self.assertEqual(self.observe.cancel(), "cancelled_after_send")
        self.assertGreater(self.observe.bytes_sent, 0)

    def test_send_crossing_deadline_keeps_partial_request(self):
        deadline = self.observe._deadline
        with mock.patch("lc0_parent_observation.time.monotonic",
                        side_effect=[deadline - 1, deadline - 1, deadline + 1]):
            self.observe.advance(selectors.EVENT_WRITE)
        self.assertEqual(self.observe.state, "failed_after_send")
        self.assertEqual(self.observe.error, "deadline_expired")
        self.assertGreater(self.observe.bytes_sent, 0)

    def test_original_send_exception_is_retained_with_custody(self):
        error = BrokenPipeError(errno.EPIPE, "private transport detail")
        self.socket.send_error = error
        self.observe.advance(selectors.EVENT_WRITE)
        self.assertEqual(self.observe.state, "failed_after_send")
        self.assertIs(self.observe.original_error, error)
        custody = self.observe.take_control_custody()
        self.assertIs(custody.original_error, error)
        self.observe._socket = custody.socket
        custody.socket = None

    def test_same_socket_cannot_be_reused_for_control(self):
        duplicate = object.__new__(NonblockingParentHandshake)
        duplicate._pid, duplicate._epoch, duplicate._lock, duplicate._ordinal = self.binding
        duplicate._state = "authenticated"
        duplicate._socket = self.admission._socket.dup()
        with self.assertRaises(ValueError):
            NonblockingSessionObservation(self.admission, duplicate, sequence=2,
                                          deadline=time.monotonic() + 2)
        duplicate._socket.close()
        duplicate._socket = None

    def test_cancelled_handoff_retains_wire_and_original_owner(self):
        self.observe.advance(selectors.EVENT_WRITE)
        self.observe.cancel()
        custody = self.observe.take_control_custody()
        self.assertEqual(custody.state, "cancelled_after_send")
        self.assertGreater(custody.bytes_sent, 0)
        self.assertTrue(custody.unsent_request)
        self.assertIs(custody.original_admission, self.admission)
        # Give teardown the original owner, not a copy, for cleanup.
        self.observe._socket = custody.socket
        custody.socket = None


if __name__ == "__main__":
    unittest.main()
