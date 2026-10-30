"""Bounded IPC protocol checks only; no native lease, Torch or training run."""
import json
import socket
import sys
import threading
import unittest

from test_rovent_training import NeuralImportsForbidden

_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_training_authority as authority
finally:
    sys.meta_path.remove(_guard)


def wire_ack(request, **changes):
    return {"schema": authority.ACK_SCHEMA, "window_id": request["window_id"], "sequence": request["sequence"],
        "operation": request["operation"], "accepted": True, "payload": {}, **changes}


class RoventTrainingProtocolTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def bound_protocol_fixture(self, responder):
        """Test-layer socketpair; never claims the production peer handshake."""
        child, parent = socket.socketpair()
        seen = []
        errors = []
        def peer():
            try:
                with parent:
                    parent.settimeout(1)
                    raw = bytearray()
                    while not raw.endswith(b"\n"):
                        block = parent.recv(4096)
                        if not block:
                            return
                        raw.extend(block)
                    request = json.loads(raw)
                    seen.append(request)
                    response = responder(request)
                    if response is not None:
                        parent.sendall((json.dumps(response) + "\n").encode())
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=peer)
        thread.start()
        startup = {"window_id": "fixture-window", "execution_until_ns": 10**18, "whole_until_ns": 10**18 + 100}
        original = authority.ParentSocketTrainerAuthority(authority._CONNECTED, child, startup, "1" * 64, lambda: 100)
        return original, thread, seen, errors

    def test_startup_json_and_public_constructor_do_not_issue_a_live_authority(self):
        with self.assertRaisesRegex(ValueError, "explicit original socket connection"):
            authority.ParentSocketTrainerAuthority.connect_registered({}, raw_startup=b"{}", expected_sha256="1" * 64)
        with self.assertRaisesRegex(ValueError, "live original parent socket handshake"):
            authority.ParentSocketTrainerAuthority(None, None, {}, "1" * 64, lambda: 0)

    def test_one_request_binds_source_window_sequence_and_exact_ack_operation(self):
        original, thread, seen, errors = self.bound_protocol_fixture(wire_ack)
        try:
            result = original._rpc("begin_window", {"schema": "fixture-only-no-native-authority"})
            self.assertEqual(result, {})
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0]["sequence"], 1)
            self.assertEqual(seen[0]["window_id"], "fixture-window")
            self.assertEqual(seen[0]["startup_sha256"], "1" * 64)
            self.assertEqual(original._sequence, 1)
        finally:
            original.close()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

    def test_foreign_ack_or_unknown_transport_failure_is_quarantined_without_reconnect(self):
        for responder in (lambda request: wire_ack(request, sequence=2), lambda request: None):
            original, thread, seen, errors = self.bound_protocol_fixture(responder)
            try:
                with self.assertRaises((ValueError, EOFError)):
                    original._rpc("completed_update", {"schema": "fixture-only-no-step"})
                with self.assertRaisesRegex(ValueError, "connection is quarantined"):
                    original._rpc("completed_update", {"schema": "must-not-retry"})
                self.assertEqual(original._sequence, 0)
            finally:
                original.close()
                thread.join(2)
            self.assertEqual(len(seen), 1)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])

    def test_oversized_or_expired_requests_stop_before_sending_work(self):
        for expired in (False, True):
            child, parent = socket.socketpair()
            startup = {"window_id": "fixture", "execution_until_ns": 100 if expired else 1000, "whole_until_ns": 1200}
            original = authority.ParentSocketTrainerAuthority(authority._CONNECTED, child, startup, "1" * 64, lambda: 100)
            try:
                with self.assertRaises((ValueError, TimeoutError)):
                    original._rpc("consume_sample", {} if expired else {"oversized": "x" * authority.FRAME_BYTES})
                parent.setblocking(False)
                with self.assertRaises(BlockingIOError):
                    parent.recv(1)
            finally:
                original.close()
                parent.close()

    def test_terminal_and_sample_calls_cannot_borrow_another_window_lease(self):
        child, parent = socket.socketpair()
        original = authority.ParentSocketTrainerAuthority(authority._CONNECTED, child,
            {"window_id": "fixture", "execution_until_ns": 1000, "whole_until_ns": 1200}, "1" * 64, lambda: 100)
        try:
            with self.assertRaisesRegex(ValueError, "same original live window lease"):
                original.consume_sample(object(), {})
            self.assertEqual(original._sequence, 0)
        finally:
            original.close()
            parent.close()


if __name__ == "__main__":
    unittest.main()
