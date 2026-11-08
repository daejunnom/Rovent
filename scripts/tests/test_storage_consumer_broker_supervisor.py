"""Protocol and mock guards only; no real process, catalog, or GPU proof."""
import os
import socket
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from storage_consumer_broker_supervisor import ConsumerBrokerSupervisor
from storage_consumer_broker_transport import BrokerTransportError, open_pair, receive_packet, send_packet


@unittest.skipUnless(os.name == "posix" and hasattr(socket, "SO_PASSCRED"), "Linux credentials required")
class ConsumerBrokerSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.parent, self.child = open_pair()
        self.addCleanup(self.parent.close)
        self.addCleanup(self.child.close)
        self.now = 10.0
        self.locked = False
        self.leases = []
        self.future = None
        self.fail_register = False
        self.child_alive = True
        self.advance_on_peer = False
        self.broker = ConsumerBrokerSupervisor(self.parent, self.transaction, lambda: None,
                                               self.assert_child, self.peer, clock=lambda: self.now)

    def assert_child(self):
        if not self.child_alive:
            raise RuntimeError("original child unavailable")

    def peer(self, pid, uid, gid):
        if (pid, uid, gid) != (os.getpid(), os.getuid(), os.getgid()):
            raise ValueError("unexpected fixture process")
        if self.advance_on_peer:
            self.now = 20.0

    @contextmanager
    def transaction(self):
        self.locked = True
        try:
            yield self
        finally:
            self.locked = False

    def register(self, inputs, *, owner_id, future_inputs):
        if not self.locked:
            raise AssertionError("registration requires transaction")
        if self.fail_register:
            raise OSError("publication and rollback unknown")
        self.future = list(future_inputs)
        lease = Lease()
        self.leases.append(lease)
        return lease

    def request(self, message):
        send_packet(self.child, message)
        self.broker.service()
        return receive_packet(self.child, self.peer)

    def test_transaction_and_unknown_physical_completion(self):
        begin = self.request({"kind": "begin", "request_id": 1, "deadline": 20.0})
        self.assertEqual(begin["phase"], "begin")
        self.assertTrue(self.locked)
        commit = self.request({"kind": "commit", "request_id": 1, "inputs": ["/asset"],
                               "future_inputs": ["/future"]})
        self.assertEqual(commit["phase"], "commit")
        self.assertEqual(self.future, ["/future"])
        self.assertFalse(self.locked)
        self.broker.finish(child_never_dispatched=False, tree_gone=True)
        self.assertEqual(self.leases[0].state, "physical_unknown")

    def test_timeout_revokes_but_does_not_unlock_before_drain(self):
        self.request({"kind": "begin", "request_id": 1, "deadline": 11.0})
        self.now = 11.0
        with self.assertRaises(BrokerTransportError):
            self.broker.service()
        self.assertTrue(self.locked)
        with self.assertRaises(BrokerTransportError):
            self.broker.finish(child_never_dispatched=False, tree_gone=False)
        self.assertTrue(self.locked)
        self.broker.finish(child_never_dispatched=False, tree_gone=True)
        self.assertFalse(self.locked)

    def test_lost_commit_ack_keeps_published_lease(self):
        self.request({"kind": "begin", "request_id": 1, "deadline": 20.0})
        send_packet(self.child, {"kind": "commit", "request_id": 1,
                                 "inputs": ["/asset"], "future_inputs": []})
        self.child.close()
        with self.assertRaises(BrokerTransportError):
            self.broker.service()
        self.assertEqual(self.leases[0].state, "active")
        self.broker.finish(child_never_dispatched=False, tree_gone=True)
        self.assertEqual(self.leases[0].state, "physical_unknown")

    def test_boolean_request_id_is_rejected(self):
        send_packet(self.child, {"kind": "begin", "request_id": True, "deadline": 20.0})
        with self.assertRaises(BrokerTransportError):
            self.broker.service()
        self.assertFalse(self.locked)

    def test_child_identity_loss_during_commit_retains_guard_until_drain(self):
        self.request({"kind": "begin", "request_id": 1, "deadline": 20.0})
        self.child_alive = False
        send_packet(self.child, {"kind": "commit", "request_id": 1,
                                 "inputs": ["/asset"], "future_inputs": []})
        with self.assertRaises(RuntimeError):
            self.broker.service()
        self.assertTrue(self.locked)
        with self.assertRaises(BrokerTransportError):
            self.broker.finish(child_never_dispatched=False, tree_gone=False)
        self.broker.finish(child_never_dispatched=False, tree_gone=True)
        self.assertFalse(self.locked)

    def test_deadline_race_during_commit_retains_guard_until_drain(self):
        self.request({"kind": "begin", "request_id": 1, "deadline": 20.0})
        self.advance_on_peer = True
        send_packet(self.child, {"kind": "commit", "request_id": 1,
                                 "inputs": ["/asset"], "future_inputs": []})
        with self.assertRaises(RuntimeError):
            self.broker.service()
        self.assertTrue(self.locked)
        self.broker.finish(child_never_dispatched=False, tree_gone=True)
        self.assertFalse(self.locked)

    def test_unknown_publication_keeps_guard_until_original_drain(self):
        self.request({"kind": "begin", "request_id": 1, "deadline": 20.0})
        self.fail_register = True
        send_packet(self.child, {"kind": "commit", "request_id": 1,
                                 "inputs": ["/asset"], "future_inputs": []})
        with self.assertRaises(RuntimeError):
            self.broker.service()
        self.assertTrue(self.broker.publication_unknown)
        self.assertIsNotNone(self.broker.unknown_owner_id)
        self.assertTrue(self.locked)
        self.broker.finish(child_never_dispatched=False, tree_gone=True)
        self.assertFalse(self.locked)


class Lease:
    def __init__(self):
        self.state = "active"

    def close(self, *, not_started):
        if not_started is not True:
            raise AssertionError("invalid proof")
        self.state = "closed"

    def retain_unknown(self, *, tree_gone):
        if tree_gone is not True:
            raise AssertionError("invalid tree observation")
        self.state = "physical_unknown"


if __name__ == "__main__":
    unittest.main()
