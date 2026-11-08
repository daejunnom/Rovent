"""Mock owner/group and local sockets; not real process or GPU evidence."""
import os
import socket
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from storage_consumer_broker_launch import ManagedBrokerLaunch
from storage_consumer_broker_transport import receive_packet, send_packet


class FakeDomain:
    def __init__(self, manager, roots):
        self.active = False
        self.closed = False
        self.leases = []

    def verify(self):
        if self.closed:
            raise RuntimeError("domain closed")

    @contextmanager
    def transaction(self):
        self.active = True
        try:
            yield self
        finally:
            self.active = False

    def register(self, inputs, *, owner_id, future_inputs):
        lease = FakeLease()
        self.leases.append(lease)
        return lease

    def close(self):
        if self.active:
            raise RuntimeError("active domain")
        self.closed = True


class FakeLease:
    def __init__(self):
        self.state = "active"

    def retain_unknown(self, *, tree_gone):
        if tree_gone is not True:
            raise AssertionError("missing process observation")
        self.state = "physical_unknown"

    def close(self, *, not_started):
        if not_started is not True:
            raise AssertionError("missing no-dispatch observation")
        self.state = "closed"


@unittest.skipUnless(sys.platform == "linux", "Linux process and socket semantics required")
class ManagedBrokerLaunchTests(unittest.TestCase):
    def setUp(self):
        self.manager = SimpleNamespace(root=Path("/synthetic/manager"), dependency_registry=object())
        self.manager._assert_owner = lambda: None
        self.manager.environment = lambda: {}
        self.fake_group = SimpleNamespace(_owner_pid=os.getpid(), _child=None, _completed=False)
        self.pins = [patch("storage_consumer_broker_launch.PinnedBrokerDomain", FakeDomain),
                     patch("storage_consumer_broker_launch._ordered_roots", return_value=[self.manager.root]),
                     patch("storage_consumer_broker_launch.OwnedProcessGroup", return_value=self.fake_group),
                     patch("storage_consumer_broker_launch.os.waitid", return_value=None)]
        for item in self.pins:
            item.start()
            self.addCleanup(item.stop)
        self.launch = ManagedBrokerLaunch(self.manager, [])
        child_fd = os.dup(self.launch.child_fds[0])
        self.child = socket.socket(fileno=child_fd)
        self.child.setblocking(False)
        self.addCleanup(self.child.close)
        self.launch.mark_dispatch_attempted()
        self.fake_group._child = SimpleNamespace(pid=os.getpid())
        self.launch.on_spawn()

    def peer(self, pid, uid, gid):
        if (pid, uid, gid) != (os.getpid(), os.getuid(), os.getgid()):
            raise ValueError("unexpected fixture peer")

    def exchange(self, message):
        send_packet(self.child, message)
        self.launch.on_tick()
        return receive_packet(self.child, self.peer)

    def test_transaction_defers_budget_and_done_preserves_lease(self):
        import time
        self.exchange({"kind": "begin", "request_id": 1, "deadline": time.monotonic() + 10, "descriptor": None})
        self.assertTrue(self.launch.defer_budget())
        self.exchange({"kind": "commit", "request_id": 1, "inputs": ["/asset"], "future_inputs": []})
        self.assertFalse(self.launch.defer_budget())
        self.exchange({"kind": "done", "request_id": 2})
        self.fake_group._completed = True
        self.assertTrue(self.launch.finish(tree_gone=True))
        self.assertEqual(self.launch.domain.leases[0].state, "physical_unknown")
        self.assertEqual(self.launch.budget_deferred_count, 1)

    def test_unknown_group_keeps_exact_active_domain_until_drain(self):
        import time
        self.exchange({"kind": "begin", "request_id": 1, "deadline": time.monotonic() + 10, "descriptor": None})
        self.assertFalse(self.launch.finish(tree_gone=False))
        self.assertTrue(self.launch.domain.active)
        self.assertFalse(self.launch.domain.closed)
        self.fake_group._completed = True
        self.assertTrue(self.launch.finish(tree_gone=True))
        self.assertFalse(self.launch.domain.active)
        self.assertTrue(self.launch.domain.closed)

    def test_unknown_group_after_done_keeps_domain_without_active_transaction(self):
        self.exchange({"kind": "done", "request_id": 1})
        self.assertFalse(self.launch.broker.transaction_active)
        self.assertFalse(self.launch.finish(tree_gone=False))
        self.assertFalse(self.launch.domain.closed)
        self.fake_group._completed = True
        self.assertTrue(self.launch.finish(tree_gone=True))
        self.assertTrue(self.launch.domain.closed)

    def test_restricted_unknown_releases_domain_but_keeps_lease_active(self):
        import time
        self.launch.reader_scope_proven = True
        self.exchange({"kind": "begin", "request_id": 1, "deadline": time.monotonic() + 10, "descriptor": None})
        self.exchange({"kind": "commit", "request_id": 1, "inputs": ["/asset"], "future_inputs": []})
        self.exchange({"kind": "begin", "request_id": 2, "deadline": time.monotonic() + 10, "descriptor": None})
        self.assertTrue(self.launch.finish(tree_gone=False))
        self.assertFalse(self.fake_group._completed)
        self.assertTrue(self.launch.domain.closed)
        self.assertEqual(self.launch.domain.leases[0].state, "active")


if __name__ == "__main__":
    unittest.main()
