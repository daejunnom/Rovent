import os
import json
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from managed_storage import StorageError
from storage_consumer_registry import (ConsumerRegistry, bind_catalog, load_catalog_binding,
                                       mark_catalog_registry_unavailable, recover_empty_required_catalog,
                                       verify_lease_coverage)
from storage_guard import CatalogGuard


class ConsumerRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.a = self.root / "a"
        self.b = self.root / "b"
        self.a.mkdir()
        self.b.mkdir()
        self.registry = ConsumerRegistry.initialize(self.root / "registry", scope="test")
        self.addCleanup(self.tmp.cleanup)

    def test_active_and_interrupted_inputs_block_only_overlaps(self):
        lease = self.registry.register([self.a])
        with self.registry.guard() as guard:
            self.assertEqual(len(guard.conflicts([self.a / "model"])["conflicts"]), 1)
            self.assertFalse(guard.conflicts([self.b])["conflicts"])
            self.assertEqual(guard.conflicts([self.b])["coverage"], "registered_only")
        with self.assertRaises(StorageError):
            lease.close()
        with self.registry.guard() as guard:
            self.assertEqual(len(guard.conflicts([self.a])["conflicts"]), 1)
        lease.close(not_started=True)
        with self.registry.guard() as guard:
            self.assertFalse(guard.conflicts([self.a])["conflicts"])
        self.assertFalse((self.registry.root / ("consumer-" + lease.owner_id + ".json")).exists())
        self.assertTrue((self.registry.root / "registry-history.json").exists())

    def test_file_hardlink_alias_is_a_conflict(self):
        source = self.a / "weights"
        source.write_bytes(b"x")
        alias = self.b / "alias"
        try:
            os.link(source, alias)
        except OSError:
            self.skipTest("hard links unavailable")
        self.registry.register([source])
        with self.registry.guard() as guard:
            result = guard.conflicts([alias])
            self.assertEqual(len(result["conflicts"]), 1)

    def test_future_input_is_lexically_protected_and_tracks_its_identity(self):
        future = self.a / "future-cache"
        self.registry.register([self.b], future_inputs=[future])
        with self.registry.guard() as guard:
            self.assertTrue(guard.conflicts([future])["conflicts"])
        future.write_bytes(b"x")
        alias = self.a / "alias"
        try:
            os.link(future, alias)
        except OSError:
            self.skipTest("hard links unavailable")
        with self.registry.guard() as guard:
            self.assertTrue(guard.conflicts([alias])["conflicts"])

    def test_catalog_binding_and_unconfigured_marker_are_not_interchangeable(self):
        catalog = self.root / "catalog"
        catalog.mkdir()
        other = ConsumerRegistry.initialize(self.root / "other-registry", scope="test")
        with CatalogGuard(catalog / "catalog.lock") as guard:
            self.assertIsNone(load_catalog_binding(catalog, guard))
            bind_catalog(catalog, self.registry, guard)
            self.assertEqual(load_catalog_binding(catalog, guard).registry_id, self.registry.registry_id)
            with self.assertRaises(StorageError):
                bind_catalog(catalog, other, guard)
        unavailable = self.root / "unavailable-catalog"
        unavailable.mkdir()
        with CatalogGuard(unavailable / "catalog.lock") as guard:
            mark_catalog_registry_unavailable(unavailable, guard)
            with self.assertRaises(StorageError):
                load_catalog_binding(unavailable, guard)
            with self.assertRaises(StorageError):
                bind_catalog(unavailable, self.registry, guard)
            recover_empty_required_catalog(unavailable, self.registry, guard)
            self.assertEqual(load_catalog_binding(unavailable, guard).registry_id, self.registry.registry_id)

    def test_output_catalog_binds_empty_or_defers_existing_without_adoption(self):
        from managed_outputs import prepare_consumer_output_catalog
        empty = self.root / "empty-output"
        result = prepare_consumer_output_catalog(self.registry, empty)
        self.assertEqual(result["status"], "bound_new_empty_catalog")
        self.assertEqual(prepare_consumer_output_catalog(self.registry, empty)["status"], "bound")
        legacy = self.root / "legacy-output"
        legacy.mkdir()
        (legacy / "old-row.json").write_text("{}")
        result = prepare_consumer_output_catalog(self.registry, legacy)
        self.assertEqual(result["status"], "cleanup_deferred_legacy_rows")
        self.assertTrue((legacy / "old-row.json").exists())
        self.assertTrue((legacy / "consumer-registry.required").exists())
        other = ConsumerRegistry.initialize(self.root / "output-other", scope="test")
        result = prepare_consumer_output_catalog(other, empty)
        self.assertEqual(result["status"], "cleanup_deferred_registry_mismatch")

    def test_nonempty_unconfigured_catalog_is_not_auto_adopted(self):
        catalog = self.root / "pending"
        catalog.mkdir()
        with CatalogGuard(catalog / "catalog.lock") as guard:
            mark_catalog_registry_unavailable(catalog, guard)
            (catalog / "owner.json").write_text("{}")
            with self.assertRaisesRegex(StorageError, "owner-specific"):
                recover_empty_required_catalog(catalog, self.registry, guard)

    def test_empty_required_catalog_recovery_cli(self):
        catalog = self.root / "cli-catalog"
        catalog.mkdir()
        with CatalogGuard(catalog / "catalog.lock") as guard:
            mark_catalog_registry_unavailable(catalog, guard)
        script = Path(__file__).resolve().parents[1] / "storage_maintenance.py"
        done = subprocess.run([sys.executable, "-B", str(script), "recover-empty-consumer-catalog",
                               "--catalog", str(catalog), "--consumer-registry-root", str(self.registry.root),
                               "--consumer-registry-id", self.registry.registry_id],
                              capture_output=True, text=True, timeout=10)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('"asset_deletion_performed": false', done.stdout)
        with CatalogGuard(catalog / "catalog.lock") as guard:
            self.assertEqual(load_catalog_binding(catalog, guard).registry_id, self.registry.registry_id)

    def test_registry_initialization_cli_does_not_overwrite_existing(self):
        root = self.root / "cli-registry"
        script = Path(__file__).resolve().parents[1] / "storage_maintenance.py"
        command = [sys.executable, "-B", str(script), "init-consumer-registry", "--root", str(root), "--scope", "test"]
        first = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertEqual(first.returncode, 0, first.stderr)
        record = (root / "registry.json").read_bytes()
        second = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(json.loads(first.stdout), json.loads(second.stdout))
        self.assertEqual(record, (root / "registry.json").read_bytes())
        different_scope = subprocess.run(command[:-1] + ["other"], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(different_scope.returncode, 0)
        self.assertEqual(record, (root / "registry.json").read_bytes())

    @unittest.skipUnless(os.name == "posix", "POSIX symlink test")
    def test_broken_binding_symlink_is_not_an_unbound_catalog(self):
        catalog = self.root / "broken"
        catalog.mkdir()
        (catalog / "consumer-registry.binding").symlink_to(catalog / "missing")
        with CatalogGuard(catalog / "catalog.lock") as guard:
            with self.assertRaises(StorageError):
                load_catalog_binding(catalog, guard)

    def test_moved_directory_is_detected_and_alias_scan_is_bounded(self):
        self.registry.register([self.a])
        self.a.rename(self.b / "moved")
        with self.registry.guard() as guard:
            self.assertTrue(guard.conflicts([self.b])["conflicts"])
            self.assertTrue(guard.conflicts([self.b], seconds=1e-12)["unknown"])

    def test_unknown_record_and_identity_mismatch_fail_closed(self):
        (self.registry.root / "bad.json").write_text("{}")
        with self.registry.guard() as guard:
            self.assertTrue(guard.conflicts([self.a])["unknown"])
        with self.assertRaises(StorageError):
            ConsumerRegistry(self.registry.root, registry_id="0" * 32)

    def test_supervisor_lease_coverage_checks_owner_and_input(self):
        lease = self.registry.register([self.a])
        self.assertTrue(verify_lease_coverage(self.registry, lease.owner_id, [self.a / "file"]))
        with self.assertRaisesRegex(StorageError, "does not cover"):
            verify_lease_coverage(self.registry, lease.owner_id, [self.b])
        with self.assertRaisesRegex(StorageError, "owner does not match"):
            verify_lease_coverage(self.registry, lease.owner_id, [self.a],
                                  expected_owner={"pid": -1, "lifetime": {}})
        lease.close(not_started=True)

    def test_tree_end_without_fence_remains_protected(self):
        lease = self.registry.register([self.a])
        with self.assertRaises(StorageError):
            lease.retain_unknown()
        lease.retain_unknown(tree_gone=True)
        with self.registry.guard() as guard:
            result = guard.conflicts([self.a])
            self.assertEqual(result["conflicts"][0]["state"], "physical_unknown_retained")
        with self.assertRaises(StorageError):
            lease.close(not_started=True)

    def test_common_lock_excludes_registration_during_cleanup(self):
        with self.registry.guard():
            with self.assertRaises(StorageError):
                self.registry.register([self.a])

    def test_failed_publication_withdraws_exact_unstarted_row(self):
        original = self.registry._write
        def published_then_failed(path, value):
            original(path, value)
            raise OSError("injected post-publication failure")
        with patch.object(self.registry, "_write", side_effect=published_then_failed):
            with self.assertRaisesRegex(OSError, "post-publication"):
                self.registry.register([self.a])
        self.assertFalse(list(self.registry.root.glob("consumer-*.json")))

    @unittest.skipUnless(os.name == "posix", "POSIX group sharing only")
    def test_group_registry_enables_group_access_to_lock_and_record(self):
        registry = ConsumerRegistry.initialize(self.root / "group-registry", scope="test", group_id=os.getegid())
        lease = registry.register([self.a])
        row = registry.root / ("consumer-" + lease.owner_id + ".json")
        self.assertEqual(row.stat().st_mode & 0o060, 0o060)
        self.assertEqual((registry.root / "registry.lock").stat().st_mode & 0o060, 0o060)
        lease.close(not_started=True)

    @unittest.skipUnless(os.name == "posix" and hasattr(os, "fork") and os.geteuid() == 0,
                         "root POSIX process needed for cross-identity verification")
    def test_registry_lock_and_rows_are_shared_across_users(self):
        import pwd
        try:
            account = pwd.getpwnam("nobody")
        except KeyError:
            self.skipTest("unprivileged nobody account unavailable")
        os.chmod(self.root, 0o755)
        registry = ConsumerRegistry.initialize(self.root / "shared", scope="test", group_id=account.pw_gid)
        lease = registry.register([self.a])
        pid = os.fork()
        if pid == 0:
            try:
                os.setgroups([])
                os.setgid(account.pw_gid)
                os.setuid(account.pw_uid)
                joined = ConsumerRegistry(registry.root, registry_id=registry.registry_id)
                with joined.guard() as guard:
                    result = guard.conflicts([self.a])
                os._exit(0 if len(result["conflicts"]) == 1 and not result["unknown"] else 2)
            except BaseException:
                os._exit(3)
        _, status = os.waitpid(pid, 0)
        self.assertEqual(os.waitstatus_to_exitcode(status), 0)
        lease.close(not_started=True)


if __name__ == "__main__":
    unittest.main()
