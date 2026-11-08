"""Isolated temporary registry/locks; no operational owner is created."""
import os
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from managed_storage import StorageError
from storage_consumer_broker_binding import PinnedBrokerDomain, manager_registry_transaction
from storage_consumer_registry import ConsumerRegistry, bind_catalog
from storage_guard import CatalogGuard


@unittest.skipUnless(os.name == "posix", "POSIX lock fixture")
class ManagerRegistryTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.root, self.other = base / "a-manager", base / "z-other"
        self.root.mkdir()
        self.other.mkdir()
        self.registry = ConsumerRegistry.initialize(base / "registry", scope="synthetic-fixture")
        for root in (self.root, self.other):
            with CatalogGuard(root / "catalog.lock") as guard:
                bind_catalog(root, self.registry, guard)
        self.original = CatalogGuard(self.root / "catalog.lock").acquire()
        self.addCleanup(self.original.release)
        self.manager = SimpleNamespace(root=self.root, guard=self.original, dependency_registry=self.registry)
        self.manager._assert_owner = self.original.assert_held

    def test_original_guard_is_reused_and_both_catalogs_participate(self):
        source = Path(self.temp.name) / "input"
        source.write_bytes(b"fixture")
        with manager_registry_transaction(self.manager, [self.other, self.root, self.other]) as registry_guard:
            lease = registry_guard.register([source])
        self.original.assert_held()
        lease.close(not_started=True)

    def test_reverse_catalog_order_is_rejected_without_releasing_owner(self):
        reverse = Path(self.temp.name) / "0-earlier"
        reverse.mkdir()
        with self.assertRaises(StorageError):
            with manager_registry_transaction(self.manager, [reverse]):
                self.fail("must reject order before taking another lock")
        self.original.assert_held()

    def test_same_id_registry_directory_copy_is_rejected(self):
        domain = PinnedBrokerDomain(self.manager, [self.other])
        self.addCleanup(domain.close)
        old = Path(self.temp.name) / "old-registry"
        os.rename(self.registry.root, old)
        shutil.copytree(old, self.registry.root)
        with self.assertRaises(StorageError):
            with domain.transaction():
                self.fail("same-ID replacement must not be accepted")

    def test_same_content_catalog_binding_replacement_is_rejected(self):
        domain = PinnedBrokerDomain(self.manager, [self.other])
        self.addCleanup(domain.close)
        binding = self.other / "consumer-registry.binding"
        replacement = self.other / "fixture-replacement"
        replacement.write_bytes(binding.read_bytes())
        os.replace(replacement, binding)
        with self.assertRaises(StorageError):
            with domain.transaction():
                self.fail("same-content replacement must not be accepted")

    def test_registry_identity_changed_in_place_is_rejected(self):
        domain = PinnedBrokerDomain(self.manager, [self.other])
        self.addCleanup(domain.close)
        record = self.registry.root / "registry.json"
        content = json.loads(record.read_text(encoding="utf-8"))
        content["scope"] = "different-synthetic-scope"
        record.write_text(json.dumps(content), encoding="utf-8")
        with self.assertRaises(StorageError):
            with domain.transaction():
                self.fail("in-place registry identity change must not be accepted")

    def test_additional_binding_in_place_change_during_transaction_is_rejected(self):
        domain = PinnedBrokerDomain(self.manager, [self.other])
        self.addCleanup(domain.close)
        binding = self.other / "consumer-registry.binding"
        with self.assertRaises(StorageError):
            with domain.transaction():
                content = json.loads(binding.read_text(encoding="utf-8"))
                content["registry_id"] = "0" * 32
                binding.write_text(json.dumps(content), encoding="utf-8")

    def test_close_failure_attempts_every_descriptor_without_retry(self):
        domain = PinnedBrokerDomain(self.manager, [self.other])
        fds = [entry[1] for entry in domain._entries]
        real_close = os.close
        failed = []

        def close_then_fail_once(fd):
            real_close(fd)
            if not failed:
                failed.append(fd)
                raise OSError("simulated uncertain close")

        with patch("storage_consumer_broker_binding.os.close", side_effect=close_then_fail_once):
            with self.assertRaises(StorageError):
                domain.close()
        self.assertTrue(domain.close_unknown)
        self.assertEqual(len(failed), 1)
        for fd in fds:
            with self.assertRaises(OSError):
                os.fstat(fd)
        domain.close()  # No numeric-fd retry after an uncertain close.


if __name__ == "__main__":
    unittest.main()
