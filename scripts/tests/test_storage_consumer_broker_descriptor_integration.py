"""Temporary registry tests for the state/descriptor coverage boundary."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from managed_storage import StorageError
from storage_consumer_broker_state import ConsumerBrokerState
from storage_consumer_registry import ConsumerRegistry


@unittest.skipUnless(sys.platform == "linux", "Linux no-follow descriptor pin required")
class BrokerDescriptorIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.registry = ConsumerRegistry.initialize(base / "registry", scope="temporary-fixture")
        self.source = base / "registered"
        self.source.mkdir()
        self.descriptor = self.source / "control.json"
        self.descriptor.write_text("{}", encoding="ascii")
        self.state = ConsumerBrokerState(self.registry.guard, lambda: None, lambda: None,
                                         clock=lambda: 10, hold_on_failure=True)

    def register(self, inputs, *, future=()):
        self.state.begin(1, deadline=20)
        self.state.commit(1, inputs, future_inputs=future)

    def test_registered_directory_covers_existing_descriptor(self):
        self.register([self.source])
        self.state.begin(2, deadline=20, descriptor=self.descriptor)
        self.assertTrue(self.state.transaction_active)
        self.state.commit(2, [])
        self.state.finish(child_never_dispatched=False, tree_gone=True)

    def test_future_reservation_alone_is_not_existing_descriptor_evidence(self):
        self.descriptor.unlink()
        self.register([], future=[self.descriptor])
        self.assertEqual(self.state._leases[0].row["inputs"][0]["kind"], "future")
        self.descriptor.write_text("{}", encoding="ascii")
        with self.assertRaises(StorageError):
            self.state.begin(2, deadline=20, descriptor=self.descriptor)
        self.assertFalse(self.state.transaction_active)
        self.state.finish(child_never_dispatched=False, tree_gone=True)

    def test_replaced_descriptor_retains_guard_until_observed_drain(self):
        self.register([self.source])
        self.state.begin(2, deadline=20, descriptor=self.descriptor)
        replacement = self.source / "replacement"
        replacement.write_text("{}", encoding="ascii")
        os.replace(replacement, self.descriptor)
        with self.assertRaises(StorageError):
            self.state.commit(2, [])
        self.assertTrue(self.state.transaction_active)
        self.state.release_after_drain(tree_gone=True)
        self.assertFalse(self.state.transaction_active)


if __name__ == "__main__":
    unittest.main()
