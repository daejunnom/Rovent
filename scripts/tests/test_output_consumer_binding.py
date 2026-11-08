"""Temporary output catalogs; no real user data or GPU execution."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import managed_outputs
import rovezero_usage_snapshot
from managed_storage import StorageError
from storage_consumer_registry import ConsumerRegistry, load_catalog_binding


class OutputConsumerBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.catalog = self.root / "output-catalog"
        self.registry = ConsumerRegistry.initialize(self.root / "registry", scope="temporary-fixture")
        patcher = patch.object(rovezero_usage_snapshot, "warn_cached_total_usage", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def lease(self, name, registry=None):
        return managed_outputs.OutputLease(self.root / name, self.source, catalog=self.catalog,
                                          reserve=1024 * 1024, total_limit=16 * 1024 * 1024,
                                          dependency_registry=registry)

    def test_empty_catalog_binds_under_existing_output_guard(self):
        lease = self.lease("first", self.registry).acquire()
        try:
            self.assertTrue(lease.consumer_catalog_observation["binding_confirmed"])
            binding = load_catalog_binding(self.catalog, lease.guard)
            self.assertEqual(binding.registry_id, self.registry.registry_id)
        finally:
            lease.finish(status="fixture", tree_gone=True)

    def test_populated_legacy_catalog_defers_cleanup_without_adoption(self):
        legacy = self.lease("legacy").acquire()
        legacy.finish(status="fixture", tree_gone=True)
        participant = self.lease("new", self.registry).acquire()
        try:
            self.assertFalse(participant.consumer_catalog_observation["binding_confirmed"])
            self.assertEqual(participant.consumer_catalog_observation["status"], "cleanup_deferred_legacy_rows")
            self.assertTrue((self.catalog / "consumer-registry.required").exists())
            self.assertFalse((self.catalog / "consumer-registry.binding").exists())
            self.assertTrue((self.root / "legacy").exists())
        finally:
            participant.finish(status="fixture", tree_gone=True)

    def test_different_registry_defers_without_rebinding(self):
        managed_outputs.prepare_consumer_output_catalog(self.registry, self.catalog)
        other = ConsumerRegistry.initialize(self.root / "other-registry", scope="other-fixture")
        participant = self.lease("new", other).acquire()
        try:
            self.assertFalse(participant.consumer_catalog_observation["binding_confirmed"])
            self.assertEqual(participant.consumer_catalog_observation["status"], "cleanup_deferred_registry_mismatch")
            self.assertTrue((self.catalog / "consumer-registry.required").exists())
        finally:
            participant.finish(status="fixture", tree_gone=True)

    def test_unavailable_registry_marks_deferral_and_keeps_producer_independent(self):
        self.registry.registry_id = "0" * 32
        participant = self.lease("new", self.registry).acquire()
        try:
            self.assertFalse(participant.consumer_catalog_observation["binding_confirmed"])
            self.assertEqual(participant.consumer_catalog_observation["status"], "cleanup_deferred_registry_unavailable")
            self.assertTrue((self.catalog / "consumer-registry.required").exists())
        finally:
            participant.finish(status="fixture", tree_gone=True)

    def test_unwritable_deferral_fails_only_participating_admission(self):
        participant = self.lease("new", self.registry)
        with patch.object(managed_outputs, "_prepare_consumer_output_catalog_locked", side_effect=OSError("unavailable")), \
                patch("storage_consumer_registry.mark_catalog_registry_unavailable", side_effect=OSError("unavailable")):
            with self.assertRaises(StorageError):
                participant.acquire()
        self.assertIsNone(participant.guard.stream)
        self.assertFalse((self.root / "new").exists())
        unrelated = self.lease("independent").acquire()
        unrelated.finish(status="fixture", tree_gone=True)


if __name__ == "__main__":
    unittest.main()
