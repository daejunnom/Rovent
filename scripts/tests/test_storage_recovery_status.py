"""Temp metadata fixtures for a read-only gate; no owner/lease/recovery execution."""
from pathlib import Path
import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from managed_storage import SCHEMA
import storage_recovery_status as status


class BuildRecoveryStatusTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rovezero-recovery-status-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "managed"
        self.root.mkdir()
        try:
            current = status.lifetime()
        except OSError as error:
            self.skipTest("actual observer lifetime unavailable: " + type(error).__name__)
        prior = copy.deepcopy(current)
        # An exact old process incarnation differs from the current process at
        # the same PID. Use the real kernel helper; no child, lease or lock.
        prior["process_birth"] = str(int(prior["process_birth"]) + 1)
        self.owner = {"schema": SCHEMA, "token": "fixture-owner-token-never-report",
            "pid": os.getpid(), "source_id": "a" * 24,
            "source": str(self.root.parent / "source-not-opened"), "lifetime": prior,
            "root_identity": [self.root.stat().st_dev, self.root.stat().st_ino]}
        self.write_owner()
        self.write_closure()

    def write_owner(self):
        raw = (json.dumps(self.owner, sort_keys=True) + "\n").encode()
        (self.root / "owner.json").write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def write_closure(self, *, tree_gone=True):
        raw = {"schema": SCHEMA, "owner": self.owner, "tree_gone": tree_gone}
        (self.root / "closure.json").write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")

    def snapshot(self):
        return {path.name: path.read_bytes() for path in self.root.iterdir()}

    def assert_read_only(self, result, before):
        self.assertEqual(self.snapshot(), before)
        for flag in ("recovery_performed", "all_outputs_termination_proven", "owner_or_lease_created",
                     "locks_acquired", "closure_created", "output_catalog_inspected", "deletion_performed"):
            self.assertIs(result[flag], False)
        self.assertIs(result["read_only"], True)
        self.assertNotIn(self.owner["token"], json.dumps(result))
        self.assertIn("original_output_finalization", result["remaining_recovery_checks_unperformed"])

    def test_original_closure_and_actual_old_pid_incarnation_match_only_initial_gate(self):
        before = self.snapshot()
        expected = hashlib.sha256(before["owner.json"]).hexdigest()
        with patch("managed_storage.recover_build", side_effect=AssertionError("no recovery call")):
            result = status.inspect_build_recovery(self.root, expected)
        self.assertEqual(result["status"], "recovery_eligible_by_existing_gate")
        self.assertTrue(result["owner_proven_dead"])
        self.assertTrue(result["original_closure_matches"])
        self.assertTrue(result["snapshot_consistent"])
        self.assertEqual(result["owner_sha256"], expected)
        self.assertEqual(result["source_id"], self.owner["source_id"])
        self.assert_read_only(result, before)

    def test_old_pid_incarnation_without_original_closure_remains_held(self):
        (self.root / "closure.json").unlink()
        before = self.snapshot()
        result = status.inspect_build_recovery(self.root)
        self.assertTrue(result["owner_proven_dead"])
        self.assertFalse(result["original_closure_matches"])
        self.assertEqual(result["status"], "held_original_tree_termination_unproven")
        self.assertFalse(result["recovery_eligible_by_existing_gate"])
        self.assert_read_only(result, before)

    def test_nonterminal_or_other_original_closure_cannot_prove_tree_termination(self):
        self.write_closure(tree_gone=False)
        before = self.snapshot()
        result = status.inspect_build_recovery(self.root)
        self.assertEqual(result["status"], "held_original_tree_termination_unproven")
        self.assertFalse(result["original_closure_matches"])
        self.assert_read_only(result, before)

    def test_same_host_reboot_uses_existing_helper_without_a_closure(self):
        current = {"host": "b" * 64, "boot": "fixture-current-linux-boot",
                   "process_birth": "41", "platform": "linux"}
        self.owner["lifetime"] = {**current, "boot": "fixture-prior-linux-boot", "process_birth": "17"}
        self.write_owner()
        (self.root / "closure.json").unlink()
        before = self.snapshot()
        with patch.object(status, "lifetime", side_effect=[current, current.copy()]):
            result = status.inspect_build_recovery(self.root)
        self.assertEqual(result["status"], "recovery_eligible_by_existing_gate")
        self.assertTrue(result["same_host_reboot_proven"])
        self.assertFalse(result["original_closure_matches"])
        self.assert_read_only(result, before)

    def test_observer_boot_change_during_read_is_unknown_and_preserved(self):
        current = status.lifetime()
        after = {**current, "boot": current["boot"] + "-changed"}
        before = self.snapshot()
        with patch.object(status, "lifetime", side_effect=[current, after]):
            result = status.inspect_build_recovery(self.root)
        self.assertEqual(result["status"], "unknown_preserved")
        self.assertFalse(result["observer_lifetime_consistent"])
        self.assertIsNone(result["recovery_eligible_by_existing_gate"])
        self.assert_read_only(result, before)

    def test_stable_new_owner_is_distinguished_from_the_expected_old_raw_hash(self):
        expected = hashlib.sha256((self.root / "owner.json").read_bytes()).hexdigest()
        self.owner["token"] = "fixture-new-owner-token-never-report"
        current_hash = self.write_owner()
        self.write_closure()
        before = self.snapshot()
        result = status.inspect_build_recovery(self.root, expected)
        self.assertEqual(result["status"], "changed_owner_preserved")
        self.assertFalse(result["expected_owner_matches"])
        self.assertEqual(result["owner_sha256"], current_hash)
        self.assertNotEqual(current_hash, expected)
        self.assert_read_only(result, before)

    def test_owner_replacement_between_snapshots_never_yields_eligibility(self):
        original = status.read_record_snapshot
        replaced = []
        def replace_after_read(path, **kwargs):
            result = original(path, **kwargs)
            if Path(path).name == "owner.json" and not replaced:
                replaced.append(True)
                self.owner["token"] = "fixture-replacement-token-never-report"
                self.write_owner()
            return result
        with patch.object(status, "read_record_snapshot", side_effect=replace_after_read):
            result = status.inspect_build_recovery(self.root)
        self.assertEqual(result["status"], "unknown_preserved")
        self.assertFalse(result["snapshot_consistent"])
        self.assertIsNone(result["recovery_eligible_by_existing_gate"])
        self.assertNotIn(self.owner["token"], json.dumps(result))

    def test_access_denied_is_unknown_without_mutating_any_record(self):
        before = self.snapshot()
        with patch.object(status, "read_record_snapshot", side_effect=PermissionError("fixture denied")):
            result = status.inspect_build_recovery(self.root)
        self.assertEqual(result["status"], "unknown_preserved")
        self.assertEqual(result["issue"]["error_type"], "PermissionError")
        self.assertIsNone(result["recovery_eligible_by_existing_gate"])
        self.assert_read_only(result, before)

    def test_owner_absence_is_stable_and_does_not_claim_prior_recovery(self):
        (self.root / "owner.json").unlink()
        before = self.snapshot()
        result = status.inspect_build_recovery(self.root)
        self.assertEqual(result["status"], "owner_absent")
        self.assertIsNone(result["owner_sha256"])
        self.assertFalse(result["recovery_eligible_by_existing_gate"])
        self.assert_read_only(result, before)

    def test_root_identity_mismatch_preserves_owner_even_with_original_closure(self):
        self.owner["root_identity"] = [0, 0]
        self.write_owner()
        self.write_closure()
        before = self.snapshot()
        result = status.inspect_build_recovery(self.root)
        self.assertFalse(result["root_identity_matches"])
        self.assertEqual(result["status"], "held_original_tree_termination_unproven")
        self.assert_read_only(result, before)

    def test_windows_filetime_reports_exact_utc_and_original_100ns_integer(self):
        current = {"host": "c" * 64, "boot": "windows-boot-counter-v1:4",
                   "process_birth": "116444736000000001", "platform": "win32"}
        self.owner["lifetime"] = {**current, "process_birth": "116444736000000000"}
        self.write_owner()
        self.write_closure()
        before = self.snapshot()
        with patch.object(status, "lifetime", side_effect=[current, current.copy()]), \
                patch.object(status, "owner_proven_dead", return_value=True):
            result = status.inspect_build_recovery(self.root)
        observed = result["owner_lifetime"]
        self.assertEqual(observed["process_birth"], "116444736000000000")
        self.assertEqual(observed["process_birth_units"], "filetime_100ns_since_1601_utc")
        self.assertEqual(observed["process_created_utc"], "1970-01-01T00:00:00.0000000Z")
        self.assertEqual(result["observer_lifetime"]["process_created_utc"], "1970-01-01T00:00:00.0000001Z")
        self.assert_read_only(result, before)


if __name__ == "__main__":
    unittest.main()
