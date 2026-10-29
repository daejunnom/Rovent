"""CPU/private-file recovery contracts; injected boots never prove a real reboot.

No production catalog, subprocess, model, network or accelerator is used.  Root
executes these tests; their synthetic lifetime changes are fixture inputs only.
"""
import copy
import errno
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import managed_output_quarantine as custody
import managed_output_quarantine_transition as transitions
import managed_output_recovery as recovery
import managed_output_unattributed_preservation as preservation
import managed_outputs as outputs
from managed_storage import GiB, StorageError, read_record_snapshot, write_record
import test_output_quarantine_transition as prior


class SealedRecoveryPredicates(unittest.TestCase):
    def setUp(self):
        self.key, self.digest = "1" * 64, "2" * 64
        self.target = Path("/private/fixture-output")
        self.now = {"host": "fixture-host", "boot": "fixture-new-boot",
                    "platform": "linux", "process_birth": "fixture-new-process"}
        self.row = {"schema": outputs.SCHEMA, "key": self.key, "output": str(self.target),
            "state": "active", "deferred": True, "identity": [7, 10],
            "pid": 123, "token": "original-fixture-token", "reserved_bytes": 1 << 20,
            "catalog_limit_bytes": 18 * GiB, "tree_gone": False,
            "lifetime": dict(self.now, boot="fixture-old-boot", process_birth="fixture-old-process"),
            "error": "original failure unchanged", "physical_completion": "unknown"}

    def eligible(self, *, row=None, digest=None, now=None):
        recovery._eligible(self.row if row is None else row, self.key,
            self.digest if digest is None else digest, self.digest, self.target,
            self.now if now is None else now)

    def test_only_original_bound_deferred_reboot_predicates_admit_without_mutation(self):
        original = copy.deepcopy(self.row)
        self.eligible()
        self.assertEqual(self.row, original)
        self.assertIsNone(recovery._eligible(self.row, self.key, self.digest,
                                             self.digest, self.target, self.now))

    def test_same_boot_foreign_host_platform_and_missing_birth_are_not_dead_proofs(self):
        for now in (dict(self.now, boot=self.row["lifetime"]["boot"]),
                    dict(self.now, host="foreign-host"), dict(self.now, platform="win32"),
                    {key: value for key, value in self.now.items() if key != "process_birth"}):
            with self.subTest(now=now), self.assertRaises(StorageError):
                self.eligible(now=now)

    def test_unbound_non_deferred_retained_digest_and_physical_upgrades_reject(self):
        for changes in ({"identity": None}, {"identity": [True, 10]}, {"identity": [7, 0]},
                        {"deferred": 1}, {"deferred": False}, {"state": "retained"},
                        {"tree_gone": True}, {"physical_completion": "confirmed"},
                        {"reserved_bytes": True}, {"catalog_limit_bytes": 2**63}):
            with self.subTest(changes=changes), self.assertRaises(StorageError):
                self.eligible(row=dict(self.row, **changes))
        with self.assertRaisesRegex(StorageError, "digest changed"):
            self.eligible(digest="3" * 64)

    def test_exact_booleans_explicit_pins_and_one_total_window_are_mandatory(self):
        arguments = dict(quarantine_transition_receipts=[], current_source_pins={},
                         root_exclusive_writer=True)
        for changes in ({"root_exclusive_writer": 1}, {"apply": 1},
                        {"maximum_seconds": True}, {"maximum_seconds": 31},
                        {"maximum_seconds": float("nan")}, {"current_source_pins": None}):
            with self.subTest(changes=changes), self.assertRaises(StorageError):
                recovery.recover_sealed_output_after_reboot(Path("/not-opened"), self.key,
                    self.digest, **dict(arguments, **changes))
        with patch.object(recovery.time, "monotonic", return_value=10):
            with self.assertRaisesRegex(StorageError, "deadline expired"):
                recovery._remaining(10)


@unittest.skipUnless(sys.platform == "linux", "Linux original private FD/mount fixtures; Root executes")
class SealedRecoveryLinuxTests(unittest.TestCase):
    """Original OutputLease seals the row; public transitions validate the view.

    The prior fixture supplies private real directories/source copies, original
    FD/hash/mount walkers and a deliberately injected lifetime.  A fixture reboot
    is not evidence for a real operator reboot or any physical GPU completion.
    """
    pins = prior.TransitionLinuxTests.pins
    anchor = prior.TransitionLinuxTests.anchor
    move = prior.TransitionLinuxTests.move

    def setUp(self):
        prior.TransitionLinuxTests.setUp(self)
        self.stack.enter_context(patch.object(recovery, "lifetime", side_effect=lambda: dict(self.now)))
        self.first = self.move(self.anchor())["receipt"]
        owner = outputs.OutputLease(self.parent / "sealed", self.source, catalog=self.catalog,
            reserve=1 << 20, deferred=True, quarantine_transition_receipts=[self.first]).acquire()
        try:
            owner.create_deferred_directory()
            (owner.output / "failure.log").write_bytes(b"original failed-run evidence; physical unknown\n")
            owner.check_budget()
            self.sealed_target, self.sealed_key = owner.output, owner.key
            self.sealed_path = self.catalog / (owner.key + ".json")
            terminal = owner.finish(status="failed-before-child-fixture", tree_gone=False)
            self.assertEqual(terminal["state"], "active")
            self.assertIs(terminal["tree_gone"], False)
        finally:
            # Original finish(False) leaves an active, bound row and explicitly
            # makes no physical inference.  Close any descriptors still held
            # if the original producer setup itself raised; do not edit its row.
            try:
                owner._close_created_directory()
            finally:
                try:
                    owner._close_quarantine_view()
                finally:
                    owner.guard.release()
        self.original = read_record_snapshot(self.sealed_path, schema=outputs.SCHEMA)
        self.original_raw = self.sealed_path.read_bytes()
        self.payload = (self.sealed_target / "failure.log").read_bytes()
        self.parent_identity = outputs.identity(self.sealed_target.parent)
        self.now.update(boot="fixture-later-boot", process_birth="fixture-later-maintainer")
        self.tip = transitions.publish_custody_transition(self.catalog, self.row_pin, self.first,
            [self.first], self.pins(), reason="reboot_reobserve", attempt_id="2" * 32,
            root_exclusive_writer=True, apply=True)["receipt"]
        self.preserved = {path: path.read_bytes() for path in self.registry.iterdir()}
        self.preserved[self.row_path] = self.raw

    def recover(self, *, apply=False, **changes):
        arguments = dict(quarantine_transition_receipts=[self.tip], current_source_pins=self.pins(),
                         root_exclusive_writer=True, apply=apply)
        arguments.update(changes)
        return recovery.recover_sealed_output_after_reboot(self.catalog, self.sealed_key,
                                                           self.original[1], **arguments)

    def assert_retained_evidence_unchanged(self):
        self.assertEqual((self.sealed_target / "failure.log").read_bytes(), self.payload)
        self.assertEqual(outputs.identity(self.sealed_target), self.original[0]["identity"])
        self.assertEqual(outputs.identity(self.sealed_target.parent), self.parent_identity)
        self.assertEqual(sorted(path.name for path in self.sealed_target.iterdir()), ["failure.log"])
        self.assertFalse((self.sealed_target / ".rovezero-output.json").exists())
        for path, raw in self.preserved.items():
            self.assertEqual(path.read_bytes(), raw)

    def test_dry_run_then_apply_only_one_row_and_keep_unknown_originals_and_error(self):
        result = self.recover()
        self.assertEqual(result["status"], "eligible_but_not_applied")
        self.assertIs(result["applied"], False)
        self.assertEqual(self.sealed_path.read_bytes(), self.original_raw)
        self.assert_retained_evidence_unchanged()
        result = self.recover(apply=True)
        actual, _, _ = read_record_snapshot(self.sealed_path, schema=outputs.SCHEMA)
        self.assertEqual(result["status"], "interrupted_retained")
        self.assertEqual(result["row"], actual)
        self.assertEqual(actual["recovery_reason"], "verified_same_host_reboot")
        self.assertEqual(actual["recovery_source_sha256"], self.original[1])
        self.assertEqual(actual["physical_completion"], "not_inferred")
        self.assertIs(actual["tree_gone"], False)
        self.assertEqual(actual["status"], "failed-before-child-fixture")
        changed = {"state", "recovery_reason", "recovered_ns", "recovery_source_sha256", "physical_completion"}
        self.assertEqual({key: value for key, value in actual.items() if key not in changed},
                         {key: value for key, value in self.original[0].items() if key not in changed})
        self.assert_retained_evidence_unchanged()

    def test_same_boot_foreign_host_stale_tip_and_unknown_row_do_not_write(self):
        good = dict(self.now)
        for changes in ({"boot": self.original[0]["lifetime"]["boot"]}, {"host": "foreign-host"}):
            self.now.update(changes)
            try:
                with self.subTest(changes=changes), self.assertRaises(StorageError):
                    self.recover(apply=True)
            finally:
                self.now.clear()
                self.now.update(good)
        with self.assertRaises(StorageError):
            self.recover(apply=True, quarantine_transition_receipts=[self.first])
        with self.assertRaises(StorageError):
            recovery.recover_sealed_output_after_reboot(self.catalog, self.key, self.row_pin["sha256"],
                quarantine_transition_receipts=[self.tip], current_source_pins=self.pins(),
                root_exclusive_writer=True, apply=True)
        self.assertEqual(self.sealed_path.read_bytes(), self.original_raw)
        self.assert_retained_evidence_unchanged()

    def test_original_readable_catalog_accepts_0755_and_rejects_foreign_write_permissions(self):
        # Exercise the whole public path with the original root-owned catalog
        # permission policy, rather than duplicating its permission expression.
        self.catalog.chmod(0o755)
        result = self.recover()
        self.assertEqual(result["status"], "eligible_but_not_applied")
        self.assertEqual(self.sealed_path.read_bytes(), self.original_raw)
        for mode in (0o775, 0o757, 0o777):
            self.catalog.chmod(mode)
            with self.subTest(mode=oct(mode)), patch.object(recovery, "write_record") as commit:
                with self.assertRaisesRegex(StorageError, "owner-controlled original catalog"):
                    self.recover(apply=True)
                commit.assert_not_called()
            self.assertEqual(self.sealed_path.read_bytes(), self.original_raw)
        self.catalog.chmod(0o755)
        self.assertEqual(self.recover(apply=True)["status"], "interrupted_retained")
        self.assert_retained_evidence_unchanged()

    def test_digest_and_bound_directory_replacement_are_refused_before_commit(self):
        with self.assertRaisesRegex(StorageError, "digest changed"):
            recovery.recover_sealed_output_after_reboot(self.catalog, self.sealed_key, "0" * 64,
                quarantine_transition_receipts=[self.tip], current_source_pins=self.pins(),
                root_exclusive_writer=True, apply=True)
        moved = self.parent / "original-kept"
        self.sealed_target.rename(moved)
        self.sealed_target.mkdir(mode=0o700)
        with patch.object(recovery, "write_record") as commit:
            with self.assertRaises(StorageError):
                self.recover(apply=True)
            commit.assert_not_called()
        self.assertEqual(self.sealed_path.read_bytes(), self.original_raw)
        self.assertEqual((moved / "failure.log").read_bytes(), self.payload)

    def test_current_source_raw_change_and_live_view_mount_change_refuse(self):
        pins = self.pins()
        path = self.source_paths["storage"]
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n# explicit fixture source mismatch\n")
            with patch.object(recovery, "write_record") as commit:
                with self.assertRaises(StorageError):
                    self.recover(apply=True, current_source_pins=pins)
                commit.assert_not_called()
        finally:
            path.write_bytes(original)
        original_mounts = custody._mounts
        calls = [0]
        def changed_mounts(*args):
            value = original_mounts(*args)
            calls[0] += 1
            # The initial transition history sees the original observation;
            # a later actual-view check must not tolerate a changed namespace.
            if calls[0] > 2:
                value = copy.deepcopy(value)
                value["namespace"][1] += 1
            return value
        with patch.object(custody, "_mounts", side_effect=changed_mounts):
            with patch.object(recovery, "write_record") as commit:
                with self.assertRaises(StorageError):
                    self.recover(apply=True)
                commit.assert_not_called()
        self.assertEqual(self.sealed_path.read_bytes(), self.original_raw)
        self.assert_retained_evidence_unchanged()

    def test_original_reservation_overflow_is_not_recovered_or_recharged(self):
        (self.sealed_target / "over-budget.log").write_bytes(b"x" * (1 << 20))
        with patch.object(recovery, "write_record") as commit:
            with self.assertRaisesRegex(StorageError, "reservation"):
                self.recover(apply=True)
            commit.assert_not_called()
        self.assertEqual(self.sealed_path.read_bytes(), self.original_raw)
        self.assertEqual((self.sealed_target / "failure.log").read_bytes(), self.payload)
        self.assertEqual((self.sealed_target / "over-budget.log").stat().st_size, 1 << 20)
        for path, raw in self.preserved.items():
            self.assertEqual(path.read_bytes(), raw)

    def test_precommit_failure_keeps_all_raw_and_first_error(self):
        with patch.object(recovery, "write_record", side_effect=OSError("fixture original commit failure")):
            with self.assertRaisesRegex(OSError, "original commit failure") as caught:
                self.recover(apply=True)
        self.assertTrue(any("write_record returned=False" in note
                            for note in getattr(caught.exception, "__notes__", [])))
        self.assertEqual(self.sealed_path.read_bytes(), self.original_raw)
        self.assert_retained_evidence_unchanged()

    def test_postreplace_failure_preserves_actual_row_and_error_without_rollback(self):
        def fail_after_replace(path, row):
            write_record(path, row)
            raise OSError("fixture post-replace durability failure")
        with patch.object(recovery, "write_record", side_effect=fail_after_replace):
            with self.assertRaisesRegex(OSError, "post-replace durability failure") as caught:
                self.recover(apply=True)
        self.assertTrue(any("commit attempted" in note for note in getattr(caught.exception, "__notes__", [])))
        actual = read_record_snapshot(self.sealed_path, schema=outputs.SCHEMA)[0]
        self.assertEqual(actual["state"], "interrupted_retained")
        self.assertEqual(actual["physical_completion"], "not_inferred")
        self.assertIs(actual["tree_gone"], False)
        self.assert_retained_evidence_unchanged()

    def test_inner_validation_cannot_renew_the_one_outer_window(self):
        original = transitions.validate_transitions
        real_clock = recovery.time.monotonic
        expired = [False]
        def clock():
            return real_clock() + (31 if expired[0] else 0)
        def expire_after_validation(*args, **kwargs):
            self.assertGreater(kwargs["maximum_seconds"], 0)
            self.assertLessEqual(kwargs["maximum_seconds"], 30)
            view = original(*args, **kwargs)
            expired[0] = True
            return view
        with patch.object(recovery.time, "monotonic", side_effect=clock):
            with patch.object(transitions, "validate_transitions", side_effect=expire_after_validation):
                with patch.object(recovery, "write_record") as commit:
                    with self.assertRaisesRegex(StorageError, "deadline expired"):
                        self.recover(apply=True)
                    commit.assert_not_called()
        self.assertEqual(self.sealed_path.read_bytes(), self.original_raw)
        self.assert_retained_evidence_unchanged()


@unittest.skipUnless(sys.platform == "linux", "real Linux private source7 FD fixtures; Root executes")
class UnboundAbsentRecoveryLinuxTests(unittest.TestCase):
    """Authored private interruption/reboot inputs, not live recovery evidence.

    Original OutputLease writes the absent reservation before its owner stops.
    Six separately archived mismatched corpora exercise the actual source7
    catalog reader; the real historical generation deliberately stays stale.
    """
    pins = prior.TransitionLinuxTests.pins
    anchor = prior.TransitionLinuxTests.anchor
    move = prior.TransitionLinuxTests.move

    def setUp(self):
        prior.TransitionLinuxTests.setUp(self)
        self.stack.enter_context(patch.object(recovery, "lifetime", side_effect=lambda: dict(self.now)))
        self.stack.enter_context(patch.object(preservation, "lifetime", side_effect=lambda: dict(self.now)))
        self.stack.enter_context(patch.object(outputs, "canonical_catalog", return_value=self.catalog))
        self.first = self.move(self.anchor())["receipt"]
        owner = outputs.OutputLease(self.parent / "unbound", self.source, catalog=self.catalog,
            reserve=1 << 20, deferred=True, quarantine_transition_receipts=[self.first]).acquire()
        try:
            self.unbound_target, self.unbound_key = owner.output, owner.key
            self.unbound_path = self.catalog / (owner.key + ".json")
            self.assertFalse(owner.output.exists())
            self.assertIs(owner.record["identity"], None)
            # An interrupted fixture owner relinquishes only its OS resources.
            # No row mutation or pretend tree/physical completion is performed.
        finally:
            try:
                owner._close_quarantine_view()
            finally:
                owner.guard.release()
        self.original = read_record_snapshot(self.unbound_path, schema=outputs.SCHEMA)
        self.original_raw = self.unbound_path.read_bytes()
        self.parent_identity = outputs.identity(self.unbound_target.parent)
        self.source7_paths = dict(self.source_paths)
        for role, module in (("session", preservation.legacy), ("unattributed", preservation)):
            path = self.source / Path(module.__file__).name
            path.write_bytes(Path(module.__file__).read_bytes())
            path.chmod(0o600)
            self.source7_paths[role] = path
        self.stack.enter_context(patch.object(preservation, "_source_paths", return_value=self.source7_paths))
        self.items = []
        for index in range(6):
            target = self.parent / ("unattributed-" + str(index))
            target.mkdir(mode=0o700)
            key = hashlib.sha256(os.path.normcase(str(target)).encode()).hexdigest()
            identity = outputs.identity(target)
            row = {"schema": outputs.SCHEMA, "key": key, "output": str(target),
                "identity": [identity[0] + 123, identity[1]], "state": "retained", "deferred": False,
                "source": str(self.source), "reserved_bytes": 1 << 20, "catalog_limit_bytes": 18 * GiB}
            raw = (json.dumps(row, sort_keys=True) + "\n").encode()
            record, marker = self.catalog / (key + ".json"), target / ".rovezero-output.json"
            record.write_bytes(raw)
            marker.write_bytes(raw)
            (target / "unique.log").write_bytes(b"preserved corpus " + str(index).encode())
            item = {"catalog": custody._pin(record, raw), "marker": custody._pin(marker, raw)}
            archive = self.parent / ("research-" + str(index) + ".zip")
            members = []
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as out:
                for path in sorted(target.iterdir()):
                    content, name = path.read_bytes(), "corpus/" + path.name
                    out.writestr(name, content)
                    members.append({"relative": path.name, "member": name, "bytes": len(content),
                                    "sha256": hashlib.sha256(content).hexdigest()})
                out.writestr("inventory.json", json.dumps({"schema": preservation.RECOVERY_SCHEMA,
                    "original_catalog_pin": item["catalog"], "original_marker_pin": item["marker"],
                    "output": str(target), "files": members}, sort_keys=True))
            item["research_archive"] = custody._pin(archive, archive.read_bytes())
            self.items.append(item)
        # No ninth generation, old-owner renewal or current-tip retagging.
        self.now.update(boot="fixture-later-boot", process_birth="fixture-later-maintainer")
        self.preserved = {path: path.read_bytes() for path in self.registry.iterdir()}
        self.preserved[self.row_path] = self.raw
        for item in self.items:
            for pin in item.values():
                path = Path(pin["path"])
                self.preserved[path] = path.read_bytes()
            target = Path(json.loads(Path(item["catalog"]["path"]).read_bytes())["output"])
            self.preserved[target / "unique.log"] = (target / "unique.log").read_bytes()

    def selection(self):
        pins = {role: custody._pin(path, path.read_bytes()) for role, path in self.source7_paths.items()}
        return preservation.UnattributedSelection(self.items, [self.first], pins,
                                                  root_exclusive_writer=True)

    def recover(self, *, apply=False, **changes):
        arguments = dict(unattributed_selection=self.selection(), root_exclusive_writer=True, apply=apply)
        arguments.update(changes)
        return recovery.recover_unbound_absent_output_after_reboot(self.catalog, self.unbound_key,
                                                                  self.original[1], **arguments)

    def assert_preserved(self):
        self.assertFalse(os.path.lexists(self.unbound_target))
        self.assertEqual(outputs.identity(self.unbound_target.parent), self.parent_identity)
        for path, raw in self.preserved.items():
            self.assertEqual(path.read_bytes(), raw)

    def test_stale_history_dry_run_and_apply_change_one_absent_row_only(self):
        result = self.recover()
        self.assertEqual(result["status"], "eligible_but_not_applied")
        self.assertIs(result["output_identity"], None)
        self.assertEqual(self.unbound_path.read_bytes(), self.original_raw)
        self.assert_preserved()
        result = self.recover(apply=True)
        actual = read_record_snapshot(self.unbound_path, schema=outputs.SCHEMA)[0]
        self.assertEqual(result["row"], actual)
        self.assertEqual(actual["state"], "interrupted_retained")
        self.assertEqual(actual["physical_completion"], "not_inferred")
        self.assertIs(actual["identity"], None)
        self.assertNotIn("tree_gone", actual)
        changed = {"state", "recovery_reason", "recovered_ns", "recovery_source_sha256", "physical_completion"}
        self.assertEqual({key: value for key, value in actual.items() if key not in changed},
                         {key: value for key, value in self.original[0].items() if key not in changed})
        self.assert_preserved()

    def test_same_boot_foreign_host_non_linux_and_changed_digest_refuse_without_write(self):
        good = dict(self.now)
        for changes in ({"boot": self.original[0]["lifetime"]["boot"]}, {"host": "foreign-host"},
                        {"platform": "win32"}):
            self.now.update(changes)
            try:
                with self.subTest(changes=changes), patch.object(recovery, "write_record") as commit:
                    with self.assertRaises(StorageError):
                        self.recover(apply=True)
                    commit.assert_not_called()
            finally:
                self.now.clear()
                self.now.update(good)
        with self.assertRaisesRegex(StorageError, "digest changed"):
            recovery.recover_unbound_absent_output_after_reboot(self.catalog, self.unbound_key, "0" * 64,
                unattributed_selection=self.selection(), root_exclusive_writer=True, apply=True)
        self.assertEqual(self.unbound_path.read_bytes(), self.original_raw)
        self.assert_preserved()

    def test_existing_file_directory_and_dangling_link_are_not_absence(self):
        for kind in ("file", "directory", "dangling-link"):
            if kind == "file":
                self.unbound_target.write_bytes(b"unattributed new evidence")
            elif kind == "directory":
                self.unbound_target.mkdir(mode=0o700)
            else:
                self.unbound_target.symlink_to(self.parent / "not-created")
            try:
                with self.subTest(kind=kind), patch.object(recovery, "write_record") as commit:
                    with self.assertRaises(StorageError):
                        self.recover(apply=True)
                    commit.assert_not_called()
                self.assertTrue(os.path.lexists(self.unbound_target))
                self.assertEqual(self.unbound_path.read_bytes(), self.original_raw)
            finally:
                if kind == "directory":
                    self.unbound_target.rmdir()
                else:
                    self.unbound_target.unlink()
        self.assert_preserved()

    def test_no_bare_dict_or_forged_boolean_can_replace_source7_or_deadline(self):
        for changes in ({"unattributed_selection": {}}, {"root_exclusive_writer": 1}, {"apply": 1},
                        {"maximum_seconds": True}, {"maximum_seconds": 31},
                        {"maximum_seconds": float("nan")}):
            arguments = dict(apply=True)
            arguments.update(changes)
            with self.subTest(changes=changes), patch.object(recovery, "write_record") as commit:
                with self.assertRaises(StorageError):
                    self.recover(**arguments)
                commit.assert_not_called()
        self.assertEqual(self.unbound_path.read_bytes(), self.original_raw)
        self.assert_preserved()

    def test_parent_permission_change_and_name_replacement_refuse(self):
        self.parent.chmod(0o777)
        try:
            with patch.object(recovery, "write_record") as commit:
                # Source7 may reject this same unsafe parent while verifying
                # a preserved sibling, before the new absence observer runs.
                with self.assertRaisesRegex(StorageError,
                        "private current custody parent/target required|owner-controlled parent"):
                    self.recover(apply=True)
                commit.assert_not_called()
        finally:
            self.parent.chmod(0o700)
        original_check = recovery._AbsentDeferredOutput.check
        renamed = self.base / "kept-original-parent"
        swapped = [False]
        def replace_parent(observed):
            if not swapped[0]:
                swapped[0] = True
                self.parent.rename(renamed)
                self.parent.mkdir(mode=0o700)
            return original_check(observed)
        with patch.object(recovery._AbsentDeferredOutput, "check", replace_parent):
            with patch.object(recovery, "write_record") as commit:
                with self.assertRaises(StorageError):
                    self.recover(apply=True)
                commit.assert_not_called()
        self.assertEqual(self.unbound_path.read_bytes(), self.original_raw)
        self.assertFalse(self.unbound_target.exists())
        self.assertTrue((renamed / "old" / "original.log").is_file())

    def test_current_source_or_archived_corpus_change_cannot_be_recovered(self):
        selection = self.selection()
        path = self.source7_paths["unattributed"]
        raw = path.read_bytes()
        try:
            path.write_bytes(raw + b"\n# changed fixture live source\n")
            with patch.object(recovery, "write_record") as commit:
                with self.assertRaises(StorageError):
                    self.recover(apply=True, unattributed_selection=selection)
                commit.assert_not_called()
        finally:
            path.write_bytes(raw)
        target = Path(json.loads(Path(self.items[0]["catalog"]["path"]).read_bytes())["output"])
        (target / "unique.log").write_bytes(b"different unarchived evidence")
        with patch.object(recovery, "write_record") as commit:
            with self.assertRaises(StorageError):
                self.recover(apply=True)
            commit.assert_not_called()
        self.assertEqual(self.unbound_path.read_bytes(), self.original_raw)

    def test_enotdir_and_permission_errors_are_not_treated_as_enoent(self):
        original_stat = os.stat
        for code in (errno.ENOTDIR, errno.EACCES, errno.EIO):
            def deny(name, *args, **kwargs):
                if name == self.unbound_target.name and kwargs.get("dir_fd") is not None:
                    raise OSError(code, "fixture stat failure")
                return original_stat(name, *args, **kwargs)
            with self.subTest(code=code), patch.object(recovery.os, "stat", side_effect=deny):
                with patch.object(recovery, "write_record") as commit:
                    with self.assertRaises(OSError) as caught:
                        self.recover(apply=True)
                    self.assertEqual(caught.exception.errno, code)
                    commit.assert_not_called()
        self.assertEqual(self.unbound_path.read_bytes(), self.original_raw)
        self.assert_preserved()

    def test_postreplace_failure_keeps_actual_row_and_primary_error(self):
        def fail_after_replace(path, row):
            write_record(path, row)
            raise OSError("fixture unbound post-replace failure")
        with patch.object(recovery, "write_record", side_effect=fail_after_replace):
            with self.assertRaisesRegex(OSError, "post-replace failure") as caught:
                self.recover(apply=True)
        self.assertTrue(any("unbound recovery catalog commit attempted" in note
                            for note in getattr(caught.exception, "__notes__", [])))
        actual = read_record_snapshot(self.unbound_path, schema=outputs.SCHEMA)[0]
        self.assertEqual(actual["state"], "interrupted_retained")
        self.assertIs(actual["identity"], None)
        self.assertEqual(actual["physical_completion"], "not_inferred")
        self.assertNotIn("tree_gone", actual)
        self.assert_preserved()

    def test_precommit_failure_keeps_raw_row_and_unknown_commit_stage(self):
        with patch.object(recovery, "write_record", side_effect=OSError("fixture unbound commit failure")):
            with self.assertRaisesRegex(OSError, "commit failure") as caught:
                self.recover(apply=True)
        self.assertTrue(any("write_record returned=False" in note
                            for note in getattr(caught.exception, "__notes__", [])))
        self.assertEqual(self.unbound_path.read_bytes(), self.original_raw)
        self.assert_preserved()

    def test_postwrite_name_appearance_is_failed_commit_without_deleting_evidence(self):
        def create_after_replace(path, row):
            write_record(path, row)
            self.unbound_target.write_bytes(b"new evidence after metadata replacement")
        with patch.object(recovery, "write_record", side_effect=create_after_replace):
            with self.assertRaises(StorageError) as caught:
                self.recover(apply=True)
        self.assertTrue(any("write_record returned=True" in note
                            for note in getattr(caught.exception, "__notes__", [])))
        self.assertEqual(read_record_snapshot(self.unbound_path, schema=outputs.SCHEMA)[0]["state"],
                         "interrupted_retained")
        self.assertEqual(self.unbound_target.read_bytes(), b"new evidence after metadata replacement")

    def test_separate_quarantined_and_selected_six_keys_cannot_be_recovered(self):
        cases = [(self.key, self.row_pin["sha256"])]
        cases += [(json.loads(Path(item["catalog"]["path"]).read_bytes())["key"],
                   item["catalog"]["sha256"]) for item in self.items]
        for key, digest in cases:
            with self.subTest(key=key), patch.object(recovery, "write_record") as commit:
                with self.assertRaisesRegex(StorageError, "original rows are immutable"):
                    recovery.recover_unbound_absent_output_after_reboot(self.catalog, key, digest,
                        unattributed_selection=self.selection(), root_exclusive_writer=True, apply=True)
                commit.assert_not_called()
        self.assertEqual(self.unbound_path.read_bytes(), self.original_raw)
        self.assert_preserved()

    def test_catalog_warning_allows_dry_run_but_live_row_mutation_does_not_commit(self):
        actual_usage = preservation.CatalogView.checked_usage
        def overflow(view):
            rows, _ = actual_usage(view)
            return rows, 18 * GiB + 1
        with patch.object(preservation.CatalogView, "checked_usage", side_effect=overflow, autospec=True):
            with patch.object(recovery, "write_record") as commit, \
                    patch.object(recovery, "warn_output_threshold") as warning:
                result = self.recover(apply=False)
                self.assertFalse(result["applied"])
                warning.assert_called()
                commit.assert_not_called()
        original_absence = recovery._AbsentDeferredOutput.check
        modified = [False]
        def change_row(observed):
            if not modified[0]:
                modified[0] = True
                row = dict(self.original[0], status="other writer changed metadata")
                write_record(self.unbound_path, row)
            return original_absence(observed)
        with patch.object(recovery._AbsentDeferredOutput, "check", change_row):
            with patch.object(recovery, "write_record") as commit:
                with self.assertRaisesRegex(StorageError, "raw/hash/stat changed"):
                    self.recover(apply=True)
                commit.assert_not_called()
        self.assertEqual(read_record_snapshot(self.unbound_path, schema=outputs.SCHEMA)[0]["status"],
                         "other writer changed metadata")
        self.assert_preserved()

    def test_inner_source7_observation_cannot_renew_outer_deadline(self):
        original = preservation.observe_catalog
        real_clock = recovery.time.monotonic
        expired = [False]
        def clock():
            return real_clock() + (31 if expired[0] else 0)
        def expire_after_view(*args, **kwargs):
            view = original(*args, **kwargs)
            expired[0] = True
            return view
        with patch.object(recovery.time, "monotonic", side_effect=clock):
            with patch.object(preservation, "observe_catalog", side_effect=expire_after_view):
                with patch.object(recovery, "write_record") as commit:
                    with self.assertRaises(StorageError):
                        self.recover(apply=True)
                    commit.assert_not_called()
        self.assertEqual(self.unbound_path.read_bytes(), self.original_raw)
        self.assert_preserved()


if __name__ == "__main__":
    unittest.main()
