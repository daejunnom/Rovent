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


class FakeProfile:
    def __init__(self, manager):
        self._manager = manager
        self.stable = True
        self.command = ("/synthetic/python", "/synthetic/reader.py")

    def unchanged(self):
        return self.stable

    def close(self):
        pass


class FakeContract:
    def __init__(self, manager):
        self.profile = FakeProfile(manager)
        self.allowed = False

    def allows_unknown_abandon(self):
        return self.allowed and self.profile.unchanged()


@unittest.skipUnless(sys.platform == "linux", "Linux process and socket semantics required")
class ManagedBrokerLaunchTests(unittest.TestCase):
    def setUp(self):
        self.manager = SimpleNamespace(root=Path("/synthetic/manager"), source=Path("/synthetic/source"),
                                       dependency_registry=object())
        self.manager._assert_owner = lambda: None
        self.manager.environment = lambda: {}
        self.fake_group = SimpleNamespace(_owner_pid=os.getpid(), _child=None, _completed=False)
        self.pins = [patch("storage_consumer_broker_launch.PinnedBrokerDomain", FakeDomain),
                     patch("storage_consumer_broker_launch.ObservedReaderProfile", FakeProfile),
                     patch("storage_consumer_broker_launch.ReviewedReaderContract", FakeContract),
                     patch("storage_consumer_broker_launch._ordered_roots", return_value=[self.manager.root]),
                     patch("storage_consumer_broker_launch.OwnedProcessGroup", return_value=self.fake_group),
                     patch("storage_consumer_broker_launch._same_process_cwd", return_value=True),
                     patch("storage_consumer_broker_launch.os.waitid", return_value=None)]
        for item in self.pins:
            item.start()
            self.addCleanup(item.stop)
        self.contract = FakeContract(self.manager)
        self.launch = ManagedBrokerLaunch(self.manager, [], reader_contract=self.contract)
        self.addCleanup(self.cleanup_launch)
        child_fd = os.dup(self.launch.child_fds[0])
        self.child = socket.socket(fileno=child_fd)
        self.child.setblocking(False)
        self.addCleanup(self.child.close)
        self.launch.assert_spawn_command(self.launch.command, self.manager.source)
        self.launch.mark_dispatch_attempted()
        self.fake_group._child = SimpleNamespace(pid=os.getpid(), args=self.launch.command)
        self.launch.on_spawn()

    def cleanup_launch(self):
        # Only synthetic group fixtures are advanced here; no OS owner exists.
        if not self.launch._closed:
            if self.launch._child is not None:
                self.fake_group._completed = True
                self.launch.finish(tree_gone=True)
            elif not self.launch._attempted:
                self.launch.finish(tree_gone=False)

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
        self.launch.reader_contract.allowed = True
        self.exchange({"kind": "begin", "request_id": 1, "deadline": time.monotonic() + 10, "descriptor": None})
        self.exchange({"kind": "commit", "request_id": 1, "inputs": ["/asset"], "future_inputs": []})
        self.exchange({"kind": "begin", "request_id": 2, "deadline": time.monotonic() + 10, "descriptor": None})
        self.assertTrue(self.launch.finish(tree_gone=False))
        self.assertFalse(self.fake_group._completed)
        self.assertTrue(self.launch.domain.closed)
        self.assertEqual(self.launch.domain.leases[0].state, "active")

    def test_changed_reader_profile_keeps_unknown_owner(self):
        import time
        self.launch.reader_contract.allowed = True
        self.launch.reader_contract.profile.stable = False
        self.exchange({"kind": "begin", "request_id": 1, "deadline": time.monotonic() + 10, "descriptor": None})
        self.assertFalse(self.launch.finish(tree_gone=False))
        self.assertTrue(self.launch.domain.active)
        self.fake_group._completed = True
        self.assertTrue(self.launch.finish(tree_gone=True))

    def test_spawn_requires_same_tuple_and_original_cwd(self):
        command = self.launch.command
        with self.assertRaises(RuntimeError):
            self.launch.assert_spawn_command(tuple(list(command)), self.manager.source)
        with self.assertRaises(RuntimeError):
            self.launch.assert_spawn_command(command, Path("/different"))
        with self.assertRaises(AttributeError):
            self.launch.command = ("/different",)
        self.fake_group._completed = True
        self.assertTrue(self.launch.finish(tree_gone=True))

    def test_on_spawn_rejects_unarmed_owner(self):
        group = SimpleNamespace(_owner_pid=os.getpid(), _child=None, _completed=False)
        with patch("storage_consumer_broker_launch.OwnedProcessGroup", return_value=group):
            other = ManagedBrokerLaunch(self.manager, [], reader_contract=FakeContract(self.manager))
            try:
                with self.assertRaises(RuntimeError):
                    other.on_spawn()
            finally:
                self.assertTrue(other.finish(tree_gone=False))

    def test_source_drift_before_spawn_rejects_only_opt_in(self):
        group = SimpleNamespace(_owner_pid=os.getpid(), _child=None, _completed=False)
        with patch("storage_consumer_broker_launch.OwnedProcessGroup", return_value=group):
            contract = FakeContract(self.manager)
            other = ManagedBrokerLaunch(self.manager, [], reader_contract=contract)
            try:
                contract.profile.stable = False
                with self.assertRaises(RuntimeError):
                    other.assert_spawn_command(other.command, self.manager.source)
                self.assertFalse(other._attempted)
            finally:
                self.assertTrue(other.finish(tree_gone=False))

    def test_automatic_profile_has_no_review_contract(self):
        group = SimpleNamespace(_owner_pid=os.getpid(), _child=None, _completed=False)
        with patch("storage_consumer_broker_launch.OwnedProcessGroup", return_value=group):
            profile = FakeProfile(self.manager)
            other = ManagedBrokerLaunch(self.manager, [], reader_profile=profile)
            try:
                self.assertIsNone(other.reader_contract)
                self.assertIs(other.reader_profile, profile)
                other.assert_spawn_command(other.command, self.manager.source)
            finally:
                self.assertTrue(other.finish(tree_gone=False))

    def test_unconfirmed_spawn_revokes_only_unserved_broker_resources(self):
        group = SimpleNamespace(_owner_pid=os.getpid(), _child=None, _completed=False)
        with patch("storage_consumer_broker_launch.OwnedProcessGroup", return_value=group):
            other = ManagedBrokerLaunch(self.manager, [], reader_contract=FakeContract(self.manager))
            other.assert_spawn_command(other.command, self.manager.source)
            other.mark_dispatch_attempted()
            other.revoke_unconfirmed_spawn()
            self.assertTrue(other._closed)
            self.assertTrue(other.domain.closed)
            self.assertFalse(group._completed)  # No process termination is asserted.
            self.assertIsNone(other._parent)
            self.assertIsNone(other._child_endpoint)

    def test_cwd_observation_does_not_assert_exec_cwd(self):
        group = SimpleNamespace(_owner_pid=os.getpid(), _child=None, _completed=False)
        with patch("storage_consumer_broker_launch.OwnedProcessGroup", return_value=group), patch(
                "storage_consumer_broker_launch._same_process_cwd", return_value=False):
            other = ManagedBrokerLaunch(self.manager, [], reader_contract=FakeContract(self.manager))
            other.assert_spawn_command(other.command, self.manager.source)
            other.mark_dispatch_attempted()
            group._child = SimpleNamespace(pid=os.getpid(), args=other.command)
            other.on_spawn()
            self.assertIs(other._child, group._child)
            self.assertEqual(other.child_cwd_observation, "differs")
            group._completed = True
            self.assertTrue(other.finish(tree_gone=True))

    def test_fast_exit_cwd_unavailable_is_only_observation(self):
        group = SimpleNamespace(_owner_pid=os.getpid(), _child=None, _completed=False)
        with patch("storage_consumer_broker_launch.OwnedProcessGroup", return_value=group), patch(
                "storage_consumer_broker_launch._same_process_cwd", side_effect=FileNotFoundError()):
            other = ManagedBrokerLaunch(self.manager, [], reader_contract=FakeContract(self.manager))
            other.assert_spawn_command(other.command, self.manager.source)
            other.mark_dispatch_attempted()
            group._child = SimpleNamespace(pid=os.getpid(), args=other.command)
            other.on_spawn()
            self.assertEqual(other.child_cwd_observation, "unavailable")
            group._completed = True
            self.assertTrue(other.finish(tree_gone=True))


if __name__ == "__main__":
    unittest.main()
