"""CPU-only V2 frame handling; actual peer admission is tested separately."""
import hashlib
import json
from pathlib import Path
import selectors
import sys
import time
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lc0_parent_observation import NonblockingSessionObservation, RESPONSE_SCHEMA, SESSION_SCHEMA
from lc0_parent_transport import NonblockingParentHandshake


class FakeControlSocket:
    def __init__(self, width=7):
        self.width = width
        self.sent = bytearray()
        self.received = []

    def send(self, value):
        count = min(self.width, len(value))
        self.sent.extend(value[:count])
        return count

    def recv(self, _maximum):
        if not self.received:
            raise BlockingIOError()
        return self.received.pop(0)

    def fileno(self):
        return 7


class SessionObservationTests(unittest.TestCase):
    def setUp(self):
        self.binding = (41, 7, hashlib.sha256(b"test lock").hexdigest(), 0)
        self.socket = FakeControlSocket()
        # This fixture deliberately isolates frame handling. It does not prove
        # real peer authentication; the handshake suite covers that boundary.
        handshake = object.__new__(NonblockingParentHandshake)
        handshake._pid, handshake._epoch, handshake._lock, handshake._ordinal = self.binding
        handshake._state, handshake._socket = "authenticated", self.socket
        self.observe = NonblockingSessionObservation(handshake, sequence=1, deadline=time.monotonic() + 2)

    def tearDown(self):
        if self.observe.state in ("writing", "reading"):
            self.observe.cancel()
        self.observe.take_control_socket()

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

    def test_cancel_does_not_transfer_an_inflight_request(self):
        with self.assertRaises(RuntimeError):
            self.observe.take_control_socket()
        self.observe.advance(selectors.EVENT_WRITE)
        self.assertEqual(self.observe.cancel(), "cancelled_after_send")
        self.assertGreater(self.observe.bytes_sent, 0)


if __name__ == "__main__":
    unittest.main()
