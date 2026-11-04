"""SOURCE ONLY: narrow cache-recovery fixtures; main owns every execution.

All paths/catalogs are private authored fixtures. Owner death and historical
chain are explicitly mocked component seams, not real termination, old owner,
source-custody, physical completion or production recovery evidence.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import managed_storage as storage
import managed_output_unattributed_preservation as policy
import managed_outputs as outputs
import test_managed_output_unattributed_preservation as previous


@unittest.skipUnless(sys.platform == "linux", "Linux original FD/gate fixtures")
class UnattributedBuildRecoveryTests(unittest.TestCase):
    setUp = previous.UnattributedInventoryTests.setUp
    _archive = previous.UnattributedInventoryTests._archive
    selection = previous.UnattributedInventoryTests.selection
    _mock_history = previous.UnattributedInventoryTests._mock_history
    _history_patches = previous.UnattributedInventoryTests._history_patches

    def rawpin(self, path):
        raw = path.read_bytes()
        return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    def interrupted_fixture(self):
        manager = storage.ManagedBuild(self.root / "build", self.source, purpose="storage-validation").acquire()
        self.old = manager
        self.owner = copy.deepcopy(manager._identity())
        self.owner_path = manager.root / "owner.json"
        self.owner_raw = self.owner_path.read_bytes()
        self.owner_sha = hashlib.sha256(self.owner_raw).hexdigest()
        storage.write_record(manager.root / "closure.json", {"schema": storage.SCHEMA,
            "owner": self.owner, "tree_gone": True})
        self.scratch_file = manager.scratch / "regenerable-scratch.txt"
        self.target_file = manager.target / "regenerable-build-cache.txt"
        self.scratch_file.write_text("fixture cache only", encoding="utf-8")
        self.target_file.write_text("fixture compiled cache only", encoding="utf-8")
        manager.guard.release()
        # The original OS lock has been released; the persisted lease remains.
        # A later owner_proven_dead mock below is diagnostic, never live proof.
        return manager

    def recover(self, selection):
        with patch.object(storage, "reboot_proves_dead", return_value=False), \
             patch.object(storage, "owner_proven_dead", return_value=True), self._history_patches():
            return storage.recover_build(self.old.root, self.owner_sha, output_preservation=selection)

    def assert_cache_preserved(self):
        self.assertTrue(self.owner_path.exists())
        self.assertTrue(self.scratch_file.exists())
        self.assertTrue(self.target_file.exists())

    def test_research_and_build_recovery_domains_remain_distinct(self):
        self.assertEqual(policy.RECOVERY_SCHEMA, "rovezero.unattributed-research-recovery.v1")
        self.assertEqual(policy.BUILD_RECOVERY_SCHEMA, "rovezero.unattributed-build-cache-recovery.v1")
        self.assertNotEqual(policy.RECOVERY_SCHEMA, policy.BUILD_RECOVERY_SCHEMA)

    def test_default_none_keeps_original_finalizer_and_original_cache_scope(self):
        self.interrupted_fixture()
        with patch.object(storage, "reboot_proves_dead", return_value=False), \
             patch.object(storage, "owner_proven_dead", return_value=True), \
             patch.object(outputs, "finalize_supervised_outputs") as original_finalizer, \
             patch.object(policy, "recovery_catalog_exclusion", side_effect=AssertionError("default must not opt in")):
            audit = storage.recover_build(self.old.root, self.owner_sha)
        original_finalizer.assert_called_once_with(self.old.root, self.owner)
        self.assertTrue(audit["regenerable_cache_removed"])
        self.assertNotIn("output_preservation", audit)
        self.assertFalse(self.owner_path.exists())

    def test_typed_cache_recovery_keeps_all_six_raw_rows_and_markers_unchanged(self):
        self.interrupted_fixture()
        original = {Path(item[key]["path"]): Path(item[key]["path"]).read_bytes()
                    for item in self.items for key in ("catalog", "marker")}
        audit = self.recover(self.selection())
        self.assertTrue(audit["regenerable_cache_removed"])
        self.assertFalse(self.scratch_file.exists())
        self.assertFalse(self.target_file.exists())
        self.assertFalse(self.owner_path.exists())
        scope = audit["output_preservation"]
        self.assertFalse(scope["output_rows_modified"])
        self.assertFalse(scope["old_owner_reissued"])
        self.assertFalse(scope["old_output_finalization_performed"])
        self.assertEqual(scope["physical_completion"], "unknown_unchanged")
        for path, raw in original.items():
            self.assertEqual(path.read_bytes(), raw)
        for item in self.items:
            self.assertTrue(Path(item["research_archive"]["path"]).is_file())

    def test_selected_active_old_owner_output_blocks_recovery_without_finalizing_it(self):
        self.interrupted_fixture()
        item, row, target = self.items[0], self.rows[0], self.targets[0]
        row.update(state="active", outer_supervisor={"root": str(self.old.root), "owner": self.owner,
                   "scope": "inherited_outer_process_group_or_job"})
        raw = (json.dumps(row, sort_keys=True) + "\n").encode()
        record, marker = Path(item["catalog"]["path"]), Path(item["marker"]["path"])
        record.write_bytes(raw)
        marker.write_bytes(raw)
        item["catalog"], item["marker"] = self.rawpin(record), self.rawpin(marker)
        archive = Path(item["research_archive"]["path"])
        self._archive(archive, target, item)
        item["research_archive"] = self.rawpin(archive)
        with self.assertRaisesRegex(storage.StorageError, "old-owner active supervised"):
            self.recover(self.selection())
        self.assert_cache_preserved()
        self.assertEqual(record.read_bytes(), raw)
        self.assertEqual(marker.read_bytes(), raw)

    def test_expired_single_observation_refuses_without_refresh_or_cache_cleanup(self):
        self.interrupted_fixture()
        actual_observe = policy.observe_catalog
        calls = []
        def expired(*args, **kwargs):
            view = actual_observe(*args, **kwargs)
            calls.append(view)
            view.held.until = -1  # Explicit clock fault fixture, no authority materialized from JSON.
            return view
        with patch.object(policy, "observe_catalog", side_effect=expired):
            with self.assertRaisesRegex(storage.StorageError, "expired"):
                self.recover(self.selection())
        self.assertEqual(len(calls), 1)
        self.assert_cache_preserved()

    def test_owner_change_after_initial_observation_is_rejected_before_cleanup(self):
        self.interrupted_fixture()
        original_check = policy._RecoveryCatalogHold.check
        calls = []
        def changed(hold):
            result = original_check(hold)
            if not calls:
                calls.append(True)
                value = {**self.owner, "token": "f" * 32}
                storage.write_record(self.owner_path, value)
            return result
        with patch.object(policy._RecoveryCatalogHold, "check", changed):
            with self.assertRaises(storage.StorageError):
                self.recover(self.selection())
        self.assert_cache_preserved()
        self.assertNotEqual(self.owner_path.read_bytes(), self.owner_raw)
        # The fixture's changed owner is preserved rather than repaired back.

    def test_current_source_change_after_observation_is_rejected_before_cleanup(self):
        self.interrupted_fixture()
        original_check = policy._RecoveryCatalogHold.check
        calls = []
        def changed(hold):
            result = original_check(hold)
            if not calls:
                calls.append(True)
                self.source_paths["unattributed"].write_text("# changed fixture source\n", encoding="utf-8")
            return result
        with patch.object(policy._RecoveryCatalogHold, "check", changed):
            with self.assertRaises(storage.StorageError):
                self.recover(self.selection())
        self.assert_cache_preserved()

    def test_recovery_copies_and_current_source7_must_both_be_outside_cleanup(self):
        self.interrupted_fixture()
        items = copy.deepcopy(self.items)
        archive = self.old.scratch / "protected-research.zip"
        archive.write_bytes(Path(items[0]["research_archive"]["path"]).read_bytes())
        items[0]["research_archive"] = self.rawpin(archive)
        with self.subTest(scope="archive"), self.assertRaisesRegex(storage.StorageError, "overlaps manager cleanup"):
            self.recover(self.selection(items))
        self.assert_cache_preserved()
        live = self.old.target / "current-unattributed.py"
        live.write_bytes(self.source_paths["unattributed"].read_bytes())
        self.source_paths["unattributed"] = live
        self.source_pins["unattributed"] = self.rawpin(live)
        with self.subTest(scope="current-source7"), self.assertRaisesRegex(storage.StorageError, "source7 overlaps build cleanup"):
            self.recover(self.selection())
        self.assert_cache_preserved()
        self.assertTrue(archive.exists())
        self.assertTrue(live.exists())

    def test_exact_closure_and_original_dead_proof_still_precede_new_policy(self):
        self.interrupted_fixture()
        selection = self.selection()
        with patch.object(policy, "recovery_catalog_exclusion") as policy_entry, \
             patch.object(storage, "reboot_proves_dead", return_value=False):
            with self.subTest(proof="owner-still-live"), \
                 patch.object(storage, "owner_proven_dead", return_value=False):
                with self.assertRaisesRegex(storage.StorageError, "termination not proven"):
                    storage.recover_build(self.old.root, self.owner_sha, output_preservation=selection)
            storage.write_record(self.old.root / "closure.json", {"schema": storage.SCHEMA,
                "owner": self.owner, "tree_gone": False})
            with self.subTest(proof="nonterminal-closure"), \
                 patch.object(storage, "owner_proven_dead", return_value=True):
                with self.assertRaisesRegex(storage.StorageError, "termination not proven"):
                    storage.recover_build(self.old.root, self.owner_sha, output_preservation=selection)
        policy_entry.assert_not_called()
        self.assert_cache_preserved()


if __name__ == "__main__":
    unittest.main()
