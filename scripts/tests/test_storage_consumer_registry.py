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

    def test_original_registry_object_rejects_replaced_lock_before_next_guard(self):
        lease = self.registry.register([self.a])
        lock = self.registry.root / "registry.lock"
        lock.rename(self.registry.root / "original.lock")
        lock.write_bytes(b"")
        with self.assertRaises(StorageError):
            with self.registry.guard():
                self.fail("a replacement lock cannot authorize inspection or cleanup")
        self.assertTrue((self.registry.root / ("consumer-" + lease.owner_id + ".json")).exists())

    def test_missing_original_registry_lock_is_not_recreated_by_guard(self):
        lock = self.registry.root / "registry.lock"
        lock.rename(self.registry.root / "original.lock")
        with self.assertRaises(StorageError):
            with self.registry.guard():
                self.fail("missing lock cannot authorize inspection or cleanup")
        self.assertFalse(lock.exists())

    def test_original_registry_object_rejects_same_id_directory_replacement(self):
        original = self.registry.root
        previous = self.root / "original-registry"
        original.rename(previous)
        original.mkdir()
        (original / "registry.json").write_bytes((previous / "registry.json").read_bytes())
        (original / "registry.lock").write_bytes(b"")
        with self.assertRaises(StorageError):
            with self.registry.guard():
                self.fail("a copied registry ID cannot replace the original owner")

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

    def test_moved_and_replaced_directory_protects_its_actual_child(self):
        nested = self.a / "nested"
        nested.mkdir()
        (nested / "data").write_bytes(b"input")
        unrelated = self.root / "unrelated"
        unrelated.write_bytes(b"other")
        self.registry.register([self.a])
        with self.registry.guard() as guard:
            self.assertEqual(guard.conflicts([unrelated])["unknown"], [])
        moved = self.b / "moved"
        self.a.rename(moved)
        self.a.mkdir()  # Same path, a different directory identity.
        with self.registry.guard() as guard:
            child = guard.conflicts([moved / "nested" / "data"], seconds=10)
            self.assertTrue(child["conflicts"])
            self.assertEqual(child["unknown"], [])
            unresolved = guard.conflicts([unrelated], seconds=10)
            self.assertFalse(unresolved["conflicts"])
            self.assertTrue(any(item.get("target") == str(unrelated) for item in unresolved["unknown"]))

    def test_moved_child_is_unknown_when_registered_directory_was_replaced(self):
        original = self.a / "input"
        original.write_bytes(b"input")
        elsewhere = self.b / "detached-input"
        self.registry.register([self.a])
        original.rename(elsewhere)
        self.a.rename(self.b / "old-directory")
        self.a.mkdir()
        with self.registry.guard() as guard:
            result = guard.conflicts([elsewhere], seconds=10)
        self.assertFalse(result["conflicts"])
        self.assertTrue(any(item.get("target") == str(elsewhere) for item in result["unknown"]))

    def test_failed_ancestor_observation_preserves_only_examined_target(self):
        self.registry.register([self.a])
        target = self.b / "target"
        target.write_bytes(b"target")
        with self.registry.guard() as guard, \
             patch("storage_consumer_registry._ancestor_directories", side_effect=StorageError("unavailable")):
            result = guard.conflicts([target], seconds=10)
        self.assertFalse(result["conflicts"])
        self.assertTrue(any(item.get("target") == str(target) for item in result["unknown"]))

    def test_unknown_record_and_identity_mismatch_fail_closed(self):
        (self.registry.root / "bad.json").write_text("{}")
        with self.registry.guard() as guard:
            self.assertTrue(guard.conflicts([self.a])["unknown"])
        with self.assertRaises(StorageError):
            ConsumerRegistry(self.registry.root, registry_id="0" * 32)

    def test_registry_scan_pages_past_old_entry_limit_and_protects_only_overlap(self):
        # Lower the former threshold rather than creating thousands of files.
        # Each actual row and registry lock still uses the normal code path.
        self.registry.register([self.a])
        for _ in range(4):
            self.registry.register([self.b])
        unrelated = self.root / "unrelated"
        unrelated.mkdir()
        with patch("storage_consumer_registry._LIMIT", 2), \
             patch("storage_consumer_registry._REGISTRY_PAGE_SIZE", 2), \
             self.registry.guard() as guard:
            clear = guard.conflicts([unrelated], seconds=10)
            self.assertEqual(clear["conflicts"], [])
            self.assertEqual(clear["unknown"], [])
            blocked = guard.conflicts([self.a], seconds=10)
            self.assertEqual(len(blocked["conflicts"]), 1)
            self.assertEqual(blocked["unknown"], [])

    def test_registry_scan_detects_directory_change_during_page_read(self):
        import storage_consumer_registry as module
        self.registry.register([self.a])
        original_read = module.read_record_snapshot
        changed = []

        def change_directory(path, **kwargs):
            result = original_read(path, **kwargs)
            if path.name.startswith("consumer-") and not changed:
                (self.registry.root / "unexpected").write_text("x")
                changed.append(True)
            return result

        # Simulate a filesystem whose directory timestamp does not change;
        # the names-only verification must still detect the new entry.
        stable_directory_stamp = object()
        with self.registry.guard() as guard, \
             patch.object(module, "read_record_snapshot", change_directory), \
             patch.object(module, "_registry_directory_stamp", return_value=stable_directory_stamp):
            observed = guard.conflicts([self.b], seconds=10)
        self.assertTrue(changed)
        self.assertEqual(observed["conflicts"], [])
        self.assertEqual([item["reason"] for item in observed["unknown"]],
                         ["registry_directory_changed_during_scan"])

    def test_registry_scan_deadline_keeps_incomplete_result_unknown(self):
        import storage_consumer_registry as module
        self.registry.register([self.a])
        with self.registry.guard() as guard, \
             patch.object(module, "_REGISTRY_PAGE_SIZE", 1), \
             patch.object(module.time, "monotonic", side_effect=[0.0, 0.0, 2.0]):
            result = guard.conflicts([self.b], seconds=1)
        self.assertTrue(any(item["reason"] == "registry_scan_time_budget_exceeded"
                            for item in result["unknown"]))

    def test_registry_scan_caps_diagnostics_without_hiding_conflicts(self):
        for _ in range(3):
            self.registry.register([self.a])
        with patch("storage_consumer_registry._LIMIT", 2), self.registry.guard() as guard:
            result = guard.conflicts([self.a], seconds=10)
        self.assertEqual(len(result["conflicts"]), 2)
        self.assertTrue(any(item["reason"] == "registry_scan_result_limit"
                            for item in result["unknown"]))

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
