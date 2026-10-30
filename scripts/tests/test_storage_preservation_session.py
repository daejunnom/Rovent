"""Preservation-policy source fixtures; Root alone runs tests/real validation.

Injected lifetimes and mount observations are fault fixtures, not evidence of a
real reboot, live experiment, process closure, or provider completion.
"""
import copy
import hashlib
import json
from pathlib import Path
import os
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import managed_output_preservation_session as session
import managed_output_quarantine as custody
import managed_output_quarantine_transition as transitions
import managed_outputs as outputs
from managed_storage import StorageError, tree_size
from storage_guard import CatalogGuard
import test_output_quarantine_transition as prior


class PreservationPolicyShapeTests(unittest.TestCase):
    def inputs(self):
        tip = {"path": str(Path.cwd() / "historical.receipt.json"), "bytes": 1, "sha256": "0" * 64}
        pins = {role: {"path": str(path), "bytes": 1, "sha256": "1" * 64}
                for role, path in session._source_paths().items()}
        return [tip], pins

    def test_raw_selection_is_copied_but_not_an_admission_capability(self):
        tips, pins = self.inputs()
        selection = session.PreservationSelection(tips, pins, root_exclusive_writer=True)
        tips[0]["sha256"] = "f" * 64
        pins["session"]["sha256"] = "e" * 64
        self.assertEqual(selection._tips[0]["sha256"], "0" * 64)
        self.assertEqual(selection._sources["session"]["sha256"], "1" * 64)
        self.assertFalse(hasattr(selection, "keys"))
        with self.assertRaisesRegex(StorageError, "actual held"):
            session.HistoricalAnchor(None, {})
        with self.assertRaisesRegex(StorageError, "actual held"):
            session.CurrentExclusionSession(None, {}, selection, None, None, None, None, None)

    def test_nonexplicit_unbounded_source5_and_by_value_reports_are_rejected(self):
        tips, pins = self.inputs()
        for changes in ({"root_exclusive_writer": False}, {"root_exclusive_writer": 1},
                        {"maximum_seconds": True}, {"maximum_seconds": 31},
                        {"maximum_seconds": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(StorageError):
                session.PreservationSelection(tips, pins,
                    **dict({"root_exclusive_writer": True}, **changes))
        with self.assertRaisesRegex(StorageError, "source6"):
            session.PreservationSelection(tips, {key: value for key, value in pins.items() if key != "session"},
                                          root_exclusive_writer=True)
        with self.assertRaisesRegex(StorageError, "explicit preservation selection"):
            outputs.OutputLease(Path.cwd() / "not-created", SCRIPTS,
                                preservation_session={"schema": session.SCHEMA})
        selection = session.PreservationSelection(tips, pins, root_exclusive_writer=True)
        for kwargs in ({"quarantine_receipts": []}, {"quarantine_transition_receipts": []},
                       {"outer_supervised": True}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(StorageError, "mutually exclusive"):
                outputs.OutputLease(Path.cwd() / "not-created", SCRIPTS,
                                    preservation_session=selection, **kwargs)

    def test_original_eight_generation_limit_and_current_validator_are_unchanged(self):
        self.assertEqual(transitions.MAX_TRANSITIONS, 8)
        with self.assertRaisesRegex(StorageError, "bounded transition"):
            transitions._prefix("0" * 64 + "-" + "1" * 64, 9, "2" * 32)

    def test_explicit_longer_session_requires_a_finite_future_caller_bound(self):
        tips, pins = self.inputs()
        deadline = time.monotonic() + 60
        selection = session.PreservationSelection(
            tips, pins, root_exclusive_writer=True, maximum_seconds=120,
            observation_hard_until=deadline)
        self.assertEqual(selection.maximum_seconds, 120)
        self.assertEqual(selection.observation_hard_until, deadline)
        for name, value in (("maximum_seconds", 240),
                            ("observation_hard_until", deadline + 60),
                            ("_window", (240, deadline + 60))):
            with self.subTest(name=name), self.assertRaisesRegex(StorageError, "immutable"):
                setattr(selection, name, value)
        with self.assertRaisesRegex(StorageError, "immutable"):
            del selection._window
        for invalid in (True, float("nan"), float("inf"), time.monotonic() - 1):
            with self.subTest(invalid=invalid), self.assertRaises(StorageError):
                session.PreservationSelection(tips, pins, root_exclusive_writer=True,
                    maximum_seconds=120, observation_hard_until=invalid)

    def test_expired_selection_cannot_create_a_new_observation(self):
        tips, pins = self.inputs()
        deadline = time.monotonic() + 60
        selection = session.PreservationSelection(
            tips, pins, root_exclusive_writer=True, maximum_seconds=120,
            observation_hard_until=deadline)
        with tempfile.TemporaryDirectory() as name:
            catalog = Path(name) / "catalog"
            catalog.mkdir()
            guard = SimpleNamespace(path=catalog / "catalog.lock", assert_held=lambda: None)
            with patch.object(outputs, "canonical_catalog", return_value=catalog), \
                    patch.object(session.sys, "platform", "linux"), \
                    patch.object(session.time, "monotonic", return_value=deadline + 1):
                with self.assertRaisesRegex(StorageError, "deadline expired"):
                    session.observe_new_output(catalog, [], selection, guard,
                                               output=Path(name) / "new-output", source=SCRIPTS)

    def test_optional_policy_cannot_open_a_separate_budget_ledger(self):
        tips, pins = self.inputs()
        selection = session.PreservationSelection(tips, pins, root_exclusive_writer=True)
        with self.assertRaisesRegex(StorageError, "canonical output catalog"):
            outputs.OutputLease(Path.cwd() / "not-created", SCRIPTS,
                catalog=Path.cwd() / "different-budget-ledger", preservation_session=selection)


@unittest.skipUnless(sys.platform == "linux", "Linux original held FD/mount fixtures; Root runs")
class PreservationSessionLinuxTests(unittest.TestCase):
    pins = prior.TransitionLinuxTests.pins
    anchor = prior.TransitionLinuxTests.anchor
    move = prior.TransitionLinuxTests.move

    def setUp(self):
        prior.TransitionLinuxTests.setUp(self)
        self.stack.enter_context(patch.object(outputs, "canonical_catalog", return_value=self.catalog))
        copied = self.source / "managed_output_preservation_session.py"
        copied.write_bytes(Path(session.__file__).read_bytes())
        copied.chmod(0o600)
        self.stack.enter_context(patch.object(session, "__file__", str(copied)))
        self.stack.enter_context(patch.object(session, "lifetime", side_effect=lambda: dict(self.now)))
        self.tip = self.move(self.anchor())["receipt"]
        for generation in range(2, 9):
            self.now.update(boot=f"fixture-transition-boot-{generation}")
            self.tip = transitions.publish_custody_transition(self.catalog, self.row_pin, self.tip,
                [self.tip], self.pins(), reason="reboot_reobserve", attempt_id=f"{generation:032x}",
                root_exclusive_writer=True, apply=True)["receipt"]
        self.preserved = {path: path.read_bytes() for path in self.registry.iterdir()}
        self.preserved[self.row_path] = self.raw
        self.now.update(boot="fixture-current-session-after-eight")

    def selection(self, **kwargs):
        source6 = {role: custody._pin(path, path.read_bytes())
                   for role, path in session._source_paths().items()}
        return session.PreservationSelection([self.tip], source6, root_exclusive_writer=True, **kwargs)

    def test_direct_observer_rejects_a_different_ledger_before_gate_or_history(self):
        other = self.parent / "different-ledger"
        with self.assertRaisesRegex(StorageError, "canonical output catalog"):
            session.observe_new_output(other, [], self.selection(), CatalogGuard(other / "catalog.lock"),
                output=self.parent / "fresh", source=self.source)
        self.assertFalse(other.exists())
        self.assert_originals_unchanged()

    def owner(self, *, output=None, reserve=1 << 20, deferred=True, selection=None):
        owner = outputs.OutputLease(output or self.parent / "fresh", self.source,
            catalog=self.catalog, reserve=reserve, deferred=deferred,
            preservation_session=selection or self.selection()).acquire()
        def close_only():
            try:
                owner._close_preservation_journal()
            finally:
                try:
                    owner._close_created_directory()
                finally:
                    try:
                        owner._close_quarantine_view()
                    finally:
                        owner.guard.release()
        self.addCleanup(close_only)
        return owner

    def assert_originals_unchanged(self):
        self.assertEqual(set(self.registry.iterdir()), set(self.preserved) - {self.row_path})
        for path, raw in self.preserved.items():
            self.assertEqual(path.read_bytes(), raw)
        self.assertIsNone(json.loads(self.row_path.read_bytes())["identity"])
        self.assertEqual(json.loads(self.row_path.read_bytes())["original_result"]["physical"], "unknown")

    def journal_rows(self, owner):
        files = list(owner.output.glob(".rovezero-preservation*"))
        self.assertEqual(len(files), 1)
        rows = [json.loads(line) for line in files[0].read_bytes().splitlines()]
        previous = "0" * 64
        for sequence, row in enumerate(rows, 1):
            body = row["observation"]
            self.assertEqual(body["sequence"], sequence)
            self.assertEqual(body["previous_sha256"], previous)
            self.assertEqual(row["sha256"], hashlib.sha256(custody._encode(body)).hexdigest())
            self.assertEqual(body["schema"], session.SCHEMA)
            self.assertEqual(set(body["current_source_pins"]), set(session.SOURCE_ROLES))
            self.assertFalse(body["product_deadline_extended"])
            self.assertFalse(body["original_generation_custody_reissued"])
            self.assertFalse(body["retirement_authority_issued"])
            self.assertEqual(body["physical_completion"], "unknown_unchanged")
            self.assertEqual(body["preserved"][0]["historical_generation"], 8)
            previous = row["sha256"]
        return rows

    def test_stale_tip8_stays_rejected_by_both_existing_admission_paths(self):
        with CatalogGuard(self.catalog / "catalog.lock") as guard:
            with self.assertRaisesRegex(StorageError, "stale"):
                transitions.validate_transitions(self.catalog, outputs._catalog_rows(self.catalog), [self.tip], guard)
        with self.assertRaisesRegex(StorageError, "interrupted output"):
            outputs.OutputLease(self.parent / "legacy-fresh", self.source, catalog=self.catalog,
                                reserve=1 << 20).acquire()
        self.assert_originals_unchanged()

    def test_new_deferred_output_seals_one_append_journal_and_keeps_all_originals(self):
        owner = self.owner()
        self.assertFalse(owner.output.exists())
        owner.create_deferred_directory()
        self.assertEqual(len(self.journal_rows(owner)), 1)
        (owner.output / "new-fixture.log").write_bytes(b"new producer only; no neural execution")
        owner.check_budget()
        self.assertEqual(len(self.journal_rows(owner)), 2)
        registry_bytes = sum(len(raw) for path, raw in self.preserved.items() if path.parent == self.registry)
        self.assertEqual(owner._output_usage(outputs._catalog_rows(self.catalog)),
                         self.row["reserved_bytes"] + registry_bytes + tree_size(owner.output))
        result = owner.finish(status="fixture-no-product-or-provider", tree_gone=True)
        self.assertEqual(result["state"], "retained")
        self.assertEqual(result["physical_completion"], "not_inferred_from_output_lifecycle")
        self.journal_rows(owner)
        self.assert_originals_unchanged()

    def test_nondeferred_new_output_also_seals_before_acquire_returns(self):
        owner = self.owner(deferred=False)
        self.assertEqual(len(self.journal_rows(owner)), 1)
        self.assertEqual(owner.output.stat().st_mode & 0o777, 0o700)
        self.assert_originals_unchanged()

    def test_expiry_requires_new_observation_without_extending_old_deadline(self):
        owner = self.owner()
        owner.create_deferred_directory()
        old = owner._quarantine_view
        old.until = old._held.until = 0
        with self.assertRaisesRegex(StorageError, "deadline expired"):
            old.check()
        owner.check_budget()
        self.assertEqual(old.until, 0)
        self.assertTrue(old._closed)
        self.assertIsNot(owner._quarantine_view, old)
        self.assertGreater(owner._quarantine_view.until, 0)
        self.assertEqual(len(self.journal_rows(owner)), 2)
        self.assert_originals_unchanged()

    def test_current_boot_change_does_not_adopt_an_existing_producer(self):
        owner = self.owner()
        self.now.update(boot="fixture-another-boot")
        with self.assertRaisesRegex(StorageError, "producer"):
            owner.check_budget()
        self.assertFalse(owner.output.exists())
        self.assert_originals_unchanged()

    def test_source6_change_before_or_during_observation_is_rejected(self):
        selection = self.selection()
        implementation = session._source_paths()["session"]
        implementation.write_bytes(implementation.read_bytes() + b"\n# changed-fixture\n")
        with self.assertRaisesRegex(StorageError, "custody metadata FD/type/owner/length changed"):
            self.owner(selection=selection)
        self.assertFalse((self.parent / "fresh").exists())
        self.assert_originals_unchanged()

    def test_preserved_descendant_and_other_active_output_are_not_adopted(self):
        with self.assertRaisesRegex(StorageError, "overlaps preserved"):
            self.owner(output=self.target / "new-child")
        other = self.parent / "other-active"
        key = hashlib.sha256(os.path.normcase(str(other)).encode()).hexdigest()
        (self.catalog / (key + ".json")).write_bytes(custody._encode(
            dict(self.row, key=key, output=str(other))))
        with self.assertRaisesRegex(StorageError, "interrupted output"):
            self.owner()
        self.assertFalse((self.parent / "fresh").exists())
        self.assert_originals_unchanged()

    def test_mandatory_journal_bytes_cannot_be_dropped_to_fit_small_reservation(self):
        owner = self.owner(reserve=1)
        with self.assertRaisesRegex(StorageError, "managed output exceeds reservation"):
            owner.create_deferred_directory()
        self.assertEqual(owner.record["state"], "active")
        self.assertTrue((owner.output / ".rovezero-preservation-session.jsonl").is_file())
        with self.assertRaisesRegex(StorageError, "binding failed"):
            owner.check_budget()
        self.assert_originals_unchanged()

    def test_partial_journal_append_is_sticky_and_preserved_before_payload(self):
        owner = self.owner()
        original_write = os.write
        writes = []
        def partial(fd, raw):
            if not writes:
                writes.append(True)
                return original_write(fd, raw[:7])
            raise OSError("fixture journal append failure")
        with patch.object(session.os, "write", side_effect=partial):
            with self.assertRaisesRegex(OSError, "append failure"):
                owner.create_deferred_directory()
        self.assertEqual((owner.output / ".rovezero-preservation-session.jsonl").stat().st_size, 7)
        self.assertTrue(owner._preservation_journal.failed)
        self.assertEqual(owner.record["state"], "active")
        with self.assertRaises(StorageError):
            owner.check_budget()
        self.assert_originals_unchanged()

    def test_changed_journal_or_exhausted_observations_stop_without_success_repair(self):
        owner = self.owner()
        owner.create_deferred_directory()
        with patch.object(session, "MAX_OBSERVATIONS", 1):
            with self.assertRaisesRegex(StorageError, "count exhausted"):
                owner.check_budget()
        self.assertEqual(len(self.journal_rows(owner)), 1)
        with self.assertRaisesRegex(StorageError, "evidence failed"):
            owner.check_budget()
        self.assert_originals_unchanged()

    def test_deadline_after_partial_append_preserves_bytes_and_stops_next_write(self):
        owner = self.owner()
        original_write = os.write
        writes = []
        def partial_then_expire(fd, raw):
            writes.append(fd)
            count = original_write(fd, raw[:7])
            # Fault-inject the actual observation used by the callback, not a
            # separate report timestamp or a synthetic success return value.
            owner._quarantine_view._held.until = 0
            return count
        with patch.object(session.os, "write", side_effect=partial_then_expire):
            with self.assertRaisesRegex(StorageError, "original custody observation deadline expired"):
                owner.create_deferred_directory()
        self.assertEqual(len(writes), 1)
        self.assertEqual((owner.output / ".rovezero-preservation-session.jsonl").stat().st_size, 7)
        self.assertTrue(owner._preservation_journal.failed)
        self.assertEqual(owner._preservation_journal.entries, 0)
        self.assertEqual(owner.record["state"], "active")
        with self.assertRaisesRegex(StorageError, "binding failed"):
            owner.check_budget()
        self.assert_originals_unchanged()

    def test_deadline_during_append_fsync_does_not_ack_written_observation(self):
        owner = self.owner()
        original_sync = os.fsync
        expired = []
        def sync_then_expire(fd):
            result = original_sync(fd)
            journal = owner._preservation_journal
            if journal is not None and fd == journal.fd and os.fstat(fd).st_size > 0:
                expired.append(fd)
                owner._quarantine_view._held.until = 0
            return result
        with patch.object(session.os, "fsync", side_effect=sync_then_expire):
            with self.assertRaisesRegex(StorageError, "original custody observation deadline expired"):
                owner.create_deferred_directory()
        self.assertEqual(len(expired), 1)
        self.assertGreater((owner.output / ".rovezero-preservation-session.jsonl").stat().st_size, 0)
        self.assertTrue(owner._preservation_journal.failed)
        self.assertEqual(owner._preservation_journal.entries, 0)
        self.assertEqual(owner.record["state"], "active")
        self.assert_originals_unchanged()


if __name__ == "__main__":
    unittest.main()
