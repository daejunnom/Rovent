"""SOURCE ONLY; main owns all imports, Linux tests and integration validation.

The fixture never accesses the canonical user catalog.  These focused tests do
not claim that a saved manifest is a live guard or that legacy physical work is
complete.  A real current-source7/history/journal integration remains required.
"""
from __future__ import annotations

import copy
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import Mock, patch
import zipfile

# Independent discovery uses the checked-out implementation. All storage
# catalogs and historical-source inputs below are private authored fixtures.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import managed_output_unattributed_preservation as policy
import managed_outputs as outputs
from managed_storage import GiB, StorageError, ManagedBuild
from storage_guard import CatalogGuard


def rawpin(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


@unittest.skipUnless(sys.platform == "linux", "requires actual Linux FD/mount observations")
class UnattributedInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.catalog = self.root / "catalog"
        self.catalog.mkdir(mode=0o700)
        self.source = self.root / "source"
        self.source.mkdir(mode=0o700)
        self.source_paths = {}
        for role in policy.SOURCE_ROLES:
            path = self.source / (role + ".py")
            path.write_text("# immutable fixture " + role + "\n", encoding="utf-8")
            self.source_paths[role] = path
        self.source_pins = {role: rawpin(path) for role, path in self.source_paths.items()}
        self.items, self.rows, self.targets = [], [], []
        for index in range(6):
            target = self.root / ("old-" + str(index))
            target.mkdir(mode=0o700)
            key = hashlib.sha256(os.path.normcase(str(target)).encode()).hexdigest()
            actual = outputs.identity(target)
            row = {"schema": outputs.SCHEMA, "key": key, "output": str(target),
                "identity": [actual[0] + 123, actual[1]], "state": "retained", "deferred": False,
                "source": str(self.source), "reserved_bytes": 1 << 20, "catalog_limit_bytes": 18 * GiB}
            raw = (json.dumps(row, sort_keys=True) + "\n").encode()
            record, marker = self.catalog / (key + ".json"), target / ".rovezero-output.json"
            record.write_bytes(raw)
            marker.write_bytes(raw)
            (target / "result.txt").write_bytes(b"unique research result " + str(index).encode())
            item = {"catalog": rawpin(record), "marker": rawpin(marker)}
            archive = self.root / ("recovered-" + str(index) + ".zip")
            self._archive(archive, target, item)
            item["research_archive"] = rawpin(archive)
            self.items.append(item)
            self.rows.append(row)
            self.targets.append(target)
        self.paths_patch = patch.object(policy, "_source_paths", return_value=self.source_paths)
        self.catalog_patch = patch.object(outputs, "canonical_catalog", return_value=self.catalog)
        self.paths_patch.start()
        self.catalog_patch.start()
        self.addCleanup(self.paths_patch.stop)
        self.addCleanup(self.catalog_patch.stop)

    def _archive(self, archive, target, item, *, omit=None):
        files = []
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as out:
            for path in sorted(target.iterdir()):
                if path.name == omit:
                    continue
                raw = path.read_bytes()
                member = "corpus/" + path.name
                files.append({"relative": path.name, "member": member, "bytes": len(raw),
                              "sha256": hashlib.sha256(raw).hexdigest()})
                out.writestr(member, raw)
            out.writestr("inventory.json", json.dumps({"schema": policy.RECOVERY_SCHEMA,
                "original_catalog_pin": item["catalog"], "original_marker_pin": item["marker"],
                "output": str(target), "files": files}, sort_keys=True))

    def selection(self, items=None):
        result = policy.UnattributedSelection(self.items if items is None else items,
            [{"path": str(self.root / "history-tip.json"), "bytes": 1, "sha256": "0" * 64}],
            self.source_pins, root_exclusive_writer=True)
        return policy.bind_original_deadline(result, time.monotonic() + 60)

    def inventory(self, selection=None):
        selection = selection or self.selection()
        held = policy._Held(30, selection.original_hard_until)
        try:
            return policy._inventory(held, self.catalog, selection)
        finally:
            held.close(sys.exc_info()[1])

    def test_raw_catalog_and_every_recovered_file_are_verified_without_old_owner(self):
        rows, details, archives = self.inventory()
        self.assertEqual(len(rows), 6)
        self.assertEqual(archives, sum(item["research_archive"]["bytes"] for item in self.items))
        for detail in details.values():
            self.assertEqual(detail["charge_bytes"], 1 << 20)
            self.assertGreater(detail["charge_bytes"], detail["observed_bytes"])
            self.assertFalse(detail["old_owner_reissued"])
            self.assertFalse(detail["identity_continuity_asserted"])
            self.assertFalse(detail["retirement_authority"])
            self.assertEqual(detail["physical_completion"], "unknown_unchanged")
            self.assertNotEqual(detail["recorded_identity"], detail["current_observed_identity"][:2])

    def test_default_reader_still_rejects_changed_dev(self):
        with self.assertRaisesRegex(StorageError, "custody changed"):
            outputs._catalog_rows(self.catalog)

    def test_missing_full_payload_recovery_cannot_be_replaced_by_row_only_evidence(self):
        items = copy.deepcopy(self.items)
        archive = Path(items[0]["research_archive"]["path"])
        self._archive(archive, self.targets[0], items[0], omit="result.txt")
        items[0]["research_archive"] = rawpin(archive)
        with self.assertRaisesRegex(StorageError, "every actual corpus file"):
            self.inventory(self.selection(items))

    def test_original_marker_edit_is_rejected(self):
        Path(self.items[0]["marker"]["path"]).write_bytes(b"{}\n")
        with self.assertRaises(StorageError):
            self.inventory()

    def test_actual_content_changed_after_archive_is_not_recovered(self):
        (self.targets[0] / "result.txt").write_bytes(b"different unique research result")
        with self.assertRaises(StorageError):
            self.inventory()

    def test_actual_current_source7_change_is_rejected(self):
        self.source_paths["unattributed"].write_text("# changed\n")
        with self.assertRaises(StorageError):
            self.inventory()

    def test_unselected_seventh_mismatch_still_blocks(self):
        target = self.root / "unselected"
        target.mkdir()
        key = hashlib.sha256(os.path.normcase(str(target)).encode()).hexdigest()
        row = {**self.rows[0], "key": key, "output": str(target), "identity": [999999, 999999]}
        raw = json.dumps(row, sort_keys=True).encode()
        (self.catalog / (key + ".json")).write_bytes(raw)
        (target / ".rovezero-output.json").write_bytes(raw)
        with self.assertRaisesRegex(StorageError, "unselected row identity"):
            self.inventory()

    def test_distinct_unselected_target_alias_is_still_rejected(self):
        target = self.root / "alias-fixture"
        target.mkdir()
        key = hashlib.sha256(os.path.normcase(str(target)).encode()).hexdigest()
        alias = outputs.identity(self.targets[0])
        row = {**self.rows[0], "key": key, "output": str(target), "identity": alias}
        raw = json.dumps(row, sort_keys=True).encode()
        (self.catalog / (key + ".json")).write_bytes(raw)
        (target / ".rovezero-output.json").write_bytes(raw)
        actual_identity = outputs.identity
        # Fault fixture only: no real mount, ownership or old continuity claim.
        with patch.object(outputs, "identity", side_effect=lambda path:
                          alias if Path(path) == target else actual_identity(path)):
            with self.assertRaisesRegex(StorageError, "another catalog row aliases"):
                self.inventory()

    def test_corpus_symlink_is_rejected(self):
        (self.targets[0] / "external").symlink_to(self.source)
        with self.assertRaises(StorageError):
            self.inventory()

    def test_dictionary_cannot_construct_a_live_view_or_observer(self):
        with self.assertRaises(StorageError):
            policy.ReadOnlyBudgetObserver(None, {}, {}, {})
        with self.assertRaises(StorageError):
            policy.create_budget_observer({}, self.selection())

    def test_original_deadline_is_not_renewed(self):
        selection = self.selection()
        original = selection.original_hard_until
        with self.assertRaises(StorageError):
            policy.bind_original_deadline(selection, original + 10)
        self.assertEqual(selection.original_hard_until, original)
        with self.assertRaises(StorageError):
            selection._bound_deadline = original + 10

    def test_nested_config_mutation_rejected_before_observation(self):
        selection = self.selection()
        selection.sources["guard"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(StorageError, "selection inputs changed"):
            self.inventory(selection)

    def test_conservative_charge_over18g_is_rejected(self):
        details = {row["key"]: {"charge_bytes": 4 * GiB} for row in self.rows}
        view = object.__new__(policy.CatalogView)
        view.details, view.archive_charge = details, 0
        view.history = types.SimpleNamespace(records=(), registry_bytes=0)
        view.check = lambda: None
        view.selection_document = lambda: {}
        rows = [(self.catalog / (row["key"] + ".json"), row, target)
                for row, target in zip(self.rows, self.targets)]
        with self.assertRaisesRegex(StorageError, "exceed18GiB"):
            view.usage(rows)

    def _mock_history(self, held, catalog, rows, tips, guard):
        # Explicit unit-only historical seam. This is NOT old chain validation,
        # reboot/physical completion or a real source7 product acceptance claim.
        return types.SimpleNamespace(held=held, keys=frozenset(), records=(), registry={},
            registry_bytes=0, observed=(), details={"unit-only": {"generation": 8}},
            check=lambda: (held.check(), guard.assert_held()), close=held.close)

    def _history_patches(self):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(policy.transitions, "_history", side_effect=self._mock_history))
        stack.enter_context(patch.object(policy.legacy, "_source_paths",
            return_value={role: self.source_paths[role] for role in policy.legacy.SOURCE_ROLES}))
        stack.enter_context(patch.object(policy.transitions, "_layout", return_value=({}, 0, {})))
        return stack

    def _observation_owner(self, name):
        owner = outputs.OutputLease(self.root / name, self.source, catalog=self.catalog,
            reserve=1 << 18, unattributed_preservation=self.selection()).acquire()
        def close_inputs_only():
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
        self.addCleanup(close_inputs_only)
        return owner

    def _view_fds(self, view):
        fds = tuple(view.inventory.held.fds) + tuple(view.inventory.history.held.fds)
        self.assertTrue(fds, "the fixture must hold actual observation inputs")
        for fd in fds:
            os.fstat(fd)
        return fds

    def _assert_closed_fds(self, fds):
        # Check before another observer opens files: the OS may then reuse the
        # descriptor numbers, which would not mean the old observation survived.
        for fd in fds:
            with self.assertRaises(OSError) as caught:
                os.fstat(fd)
            self.assertEqual(caught.exception.errno, errno.EBADF)

    def test_replacement_retires_actual_input_fds_before_new_observation(self):
        with self._history_patches():
            owner = self._observation_owner("fd-replacement")
            old = owner._quarantine_view
            old_fds = self._view_fds(old)
            guard_stream = owner.guard.stream
            journal = owner._preservation_journal
            journal_identity = policy.custody._did(os.fstat(journal.fd))
            original_record = copy.deepcopy(owner.record)
            row_path = self.catalog / (owner.key + ".json")
            original_row = row_path.read_bytes()
            observe = policy.observe_fresh_output
            # This independent input shares a pathname with an observation
            # source, but has its own owner and must survive view retirement.
            with self.source_paths["guard"].open("rb") as retained_input:
                retained_identity = policy.custody._stamp(os.fstat(retained_input.fileno()))
                def next_observation(*args, **kwargs):
                    self._assert_closed_fds(old_fds)
                    owner.guard.assert_held()
                    self.assertIs(owner.guard.stream, guard_stream)
                    self.assertIs(owner._preservation_journal, journal)
                    self.assertEqual(policy.custody._did(os.fstat(journal.fd)), journal_identity)
                    self.assertEqual(owner.record, original_record)
                    self.assertEqual(row_path.read_bytes(), original_row)
                    self.assertEqual(policy.custody._stamp(os.fstat(retained_input.fileno())),
                                     retained_identity)
                    return observe(*args, **kwargs)
                with patch.object(policy, "observe_fresh_output", side_effect=next_observation):
                    owner.check_budget()
                self.assertEqual(policy.custody._stamp(os.fstat(retained_input.fileno())),
                                 retained_identity)
            self.assertTrue(old._closed)
            self.assertIsNot(owner._quarantine_view, old)
            self.assertEqual(owner.record, original_record)
            self.assertEqual(row_path.read_bytes(), original_row)

    def test_old_observation_close_failure_preserves_original_error_and_owner(self):
        with self._history_patches():
            owner = self._observation_owner("fd-close-failure")
            old = owner._quarantine_view
            old_fds = self._view_fds(old)
            close = old.close
            failure = OSError(errno.EIO, "injected observation close failure")
            original_record = copy.deepcopy(owner.record)
            row_path = self.catalog / (owner.key + ".json")
            original_row = row_path.read_bytes()
            journal = owner._preservation_journal
            def close_then_fail(primary=None):
                close(primary)
                raise failure
            with patch.object(old, "close", side_effect=close_then_fail), \
                 patch.object(policy, "observe_fresh_output",
                              side_effect=AssertionError("failed close cannot admit a replacement")):
                with self.assertRaises(OSError) as caught:
                    owner.check_budget()
            self.assertIs(caught.exception, failure)
            self._assert_closed_fds(old_fds)
            self.assertIsNone(owner._quarantine_view)
            self.assertIs(owner._preservation_journal, journal)
            owner.guard.assert_held()
            self.assertEqual(owner.record, original_record)
            self.assertEqual(row_path.read_bytes(), original_row)

    def test_new_observation_failure_during_finish_releases_only_owned_inputs(self):
        with self._history_patches():
            owner = self._observation_owner("fd-new-observation-failure")
            old_fds = self._view_fds(owner._quarantine_view)
            partial_fds = []
            failure = OSError(errno.EIO, "injected new inventory failure")
            row_path = self.catalog / (owner.key + ".json")
            original_row = row_path.read_bytes()
            original_record = copy.deepcopy(owner.record)
            observe = policy.observe_fresh_output
            def next_observation(*args, **kwargs):
                self._assert_closed_fds(old_fds)
                return observe(*args, **kwargs)
            def fail_inventory(held, *args, **kwargs):
                held.read(self.items[0]["marker"], policy.MAX_METADATA)
                partial_fds.extend(held.fds)
                raise failure
            with self.source_paths["guard"].open("rb") as retained_input:
                retained_identity = policy.custody._stamp(os.fstat(retained_input.fileno()))
                with patch.object(policy, "observe_fresh_output", side_effect=next_observation), \
                     patch.object(policy, "_inventory", side_effect=fail_inventory):
                    with self.assertRaises(OSError) as caught:
                        owner.finish(status="must_not_be_published", tree_gone=True)
                self.assertIs(caught.exception, failure)
                self.assertTrue(partial_fds)
                self._assert_closed_fds(partial_fds)
                self.assertEqual(policy.custody._stamp(os.fstat(retained_input.fileno())),
                                 retained_identity)
            self.assertIsNone(owner._quarantine_view)
            self.assertIsNone(owner._preservation_journal)
            self.assertIsNone(owner.guard.stream)
            self.assertEqual(owner.record, original_record)
            self.assertEqual(row_path.read_bytes(), original_row)

    def test_new_output_journal_and_old_rows_unchanged_with_explicit_mocked_history(self):
        original = {Path(item["catalog"]["path"]): Path(item["catalog"]["path"]).read_bytes()
                    for item in self.items}
        with self._history_patches():
            owner = outputs.OutputLease(self.root / "fresh-only", self.source, catalog=self.catalog,
                reserve=1 << 18, unattributed_preservation=self.selection()).acquire()
            try:
                (owner.output / "payload.txt").write_text("new payload only", encoding="utf-8")
                owner.check_budget()
                self.assertTrue((owner.output / ".rovezero-preservation-session.jsonl").is_file())
                self.assertFalse(hasattr(owner._quarantine_view, "retire_old_output"))
                owner.finish(status="unit_storage_only", tree_gone=True)
            finally:
                owner.guard.release()
        for path, raw in original.items():
            self.assertEqual(path.read_bytes(), raw)

    def test_partial_journal_failure_retains_new_evidence_without_payload_permission(self):
        with self._history_patches(), patch.object(policy.legacy._EvidenceJournal, "append",
                                                   side_effect=StorageError("injected append failure")):
            owner = outputs.OutputLease(self.root / "partial-only", self.source, catalog=self.catalog,
                reserve=1 << 18, unattributed_preservation=self.selection())
            with self.assertRaisesRegex(StorageError, "injected append"):
                owner.acquire()
            self.assertIsNotNone(owner._preservation_error)
            self.assertTrue((owner.output / ".rovezero-preservation-session.jsonl").is_file())
            self.assertTrue((self.catalog / (owner.key + ".json")).is_file())
            self.assertFalse((owner.output / "payload.txt").exists())

    def test_outer_readonly_sample_does_not_acquire_producer_gate(self):
        with self._history_patches():
            manager = ManagedBuild(self.root / "build", self.source, purpose="storage-validation",
                                   output_preservation=self.selection()).acquire()
            self.addCleanup(manager.guard.release)
            first = manager.sample_output_budget()
            with CatalogGuard(self.catalog / "catalog.lock"), patch.object(policy.custody,
                    "original_custody_gate", side_effect=AssertionError("must not acquire producer gate")):
                second = manager.sample_output_budget()
            self.assertEqual(first, second)
            self.assertFalse(hasattr(manager._output_budget_observer, "acquire"))
            with self.assertRaisesRegex(StorageError, "cleanup unverified"):
                manager.finish(exit_code=0, tree_gone=False)
            self.assertFalse((manager.root / "closure.json").exists())

    def test_nonselected_active_marker_write_window_is_budget_only(self):
        target = self.root / "current-producer"
        target.mkdir()
        key = hashlib.sha256(os.path.normcase(str(target)).encode()).hexdigest()
        row = {**self.rows[0], "key": key, "output": str(target), "identity": outputs.identity(target),
               "state": "active"}
        (self.catalog / (key + ".json")).write_text(json.dumps(row), encoding="utf-8")
        (target / ".rovezero-output.json").write_text(
            json.dumps({**row, "status": "fixture-mid-write-different-field"}), encoding="utf-8")
        selected = {item["catalog"]["path"].split("/")[-1].split(".")[0]:
                    (Path(item["catalog"]["path"]), old, old_target)
                    for item, old, old_target in zip(self.items, self.rows, self.targets)}
        with self.assertRaisesRegex(StorageError, "unselected row marker"):
            policy._rows(self.catalog, selected)
        self.assertEqual(len(policy._rows(self.catalog, selected, read_only_budget=True)), 7)

    def test_recovery_archive_inside_build_cleanup_is_rejected_before_acquire(self):
        items = copy.deepcopy(self.items)
        items[0]["research_archive"]["path"] = str(self.root / "build" / "slots" / "fixture" / "tmp" / "archive.zip")
        with self.assertRaisesRegex(StorageError, "overlaps manager cleanup"):
            ManagedBuild(self.root / "build", self.source, output_preservation=self.selection(items))
        self.assertFalse((self.root / "build").exists())

    def test_actual_held_file_rename_is_rejected(self):
        selection = self.selection()
        held = policy._Held(30, selection.original_hard_until)
        try:
            held.read(self.items[0]["marker"], policy.MAX_METADATA)
            path = Path(self.items[0]["marker"]["path"])
            replacement = path.with_name("replacement.json")
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
            with self.assertRaisesRegex(StorageError, "held file"):
                held.check()
        finally:
            held.close(sys.exc_info()[1])

    def test_corpus_child_requires_its_actual_held_parent_fd(self):
        held = policy._Held(30, time.monotonic() + 30)
        try:
            parent = held.directory(self.targets[0])
            other = held.directory(self.targets[1])
            path = self.targets[0] / "result.txt"
            with self.assertRaisesRegex(StorageError, "exact held parent"):
                held._open_file_at(path, 1 << 20, other)
            fd, _ = held._open_file_at(path, 1 << 20, parent)
            self.assertEqual(os.read(fd, 1024), path.read_bytes())
            held.check()
        finally:
            held.close(sys.exc_info()[1])

    def test_corpus_child_rejects_parent_replacement_before_open(self):
        parent = self.root / "held-corpus-parent"
        parent.mkdir(mode=0o700)
        path = parent / "input.txt"
        path.write_bytes(b"same bytes do not prove the same parent")
        held = policy._Held(30, time.monotonic() + 30)
        try:
            parent_fd = held.directory(parent)
            parent.rename(self.root / "preserved-corpus-parent")
            parent.mkdir(mode=0o700)
            path.write_bytes(b"same bytes do not prove the same parent")
            with self.assertRaisesRegex(StorageError, "held directory"):
                held._open_file_at(path, 1 << 20, parent_fd)
        finally:
            held.close(sys.exc_info()[1])

    def test_held_ancestor_replacement_is_rejected_with_identical_payload(self):
        parent = self.root / "held-ancestor"
        parent.mkdir(mode=0o700)
        source = parent / "input.txt"
        source.write_bytes(b"same bytes do not prove the same ancestor")
        held = policy._Held(30, time.monotonic() + 30)
        try:
            held.read(rawpin(source), policy.MAX_METADATA)
            parent.rename(self.root / "preserved-ancestor")
            parent.mkdir(mode=0o700)
            source.write_bytes(b"same bytes do not prove the same ancestor")
            with self.assertRaisesRegex(StorageError, "held directory"):
                held.check()
        finally:
            held.close(sys.exc_info()[1])

    def test_held_ancestor_symlink_is_rejected_before_admission(self):
        parent = self.root / "held-ancestor"
        parent.mkdir(mode=0o700)
        source = parent / "input.txt"
        source.write_bytes(b"original current input")
        held = policy._Held(30, time.monotonic() + 30)
        try:
            held.read(rawpin(source), policy.MAX_METADATA)
            preserved = self.root / "preserved-ancestor"
            parent.rename(preserved)
            parent.symlink_to(preserved, target_is_directory=True)
            with self.assertRaisesRegex(StorageError, "held directory"):
                held.check()
        finally:
            held.close(sys.exc_info()[1])

    def test_expired_observation_is_not_renewed(self):
        selection = self.selection()
        held = policy._Held(30, selection.original_hard_until)
        try:
            with patch.object(policy.time, "monotonic", return_value=held.until):
                with self.assertRaises(policy.ObservationExpired) as caught:
                    held.tick()
                self.assertIsInstance(caught.exception, StorageError)
                self.assertEqual(caught.exception.failure_scope, "storage_observation")
                self.assertEqual(caught.exception.deadline_monotonic, held.until)
                self.assertEqual(caught.exception.observed_monotonic, held.until)
        finally:
            held.close()

    def test_zip_central_entry_count_is_bounded_before_zipfile(self):
        pin = self.items[0]["research_archive"]
        raw = bytearray(Path(pin["path"]).read_bytes())
        offset = raw.rfind(b"PK\x05\x06")
        raw[offset + 10:offset + 12] = (5000).to_bytes(2, "little")
        path = self.root / "oversize-central.zip"
        path.write_bytes(raw)
        held = policy._Held(30, time.monotonic() + 30)
        try:
            fd, _ = held.open_file(path, policy.MAX_ARCHIVE)
            with self.assertRaisesRegex(StorageError, "central directory"):
                policy._zip_metadata_preflight(held, fd, len(raw))
        finally:
            held.close(sys.exc_info()[1])

    def test_zip64_and_big_central_size_rejected_before_zipfile(self):
        pin = self.items[0]["research_archive"]
        for tag, field, value, length in (("zip64", 10, 0xffff, 2),
                                          ("central-bytes", 12, policy.MAX_CENTRAL_DIRECTORY + 1, 4)):
            with self.subTest(tag=tag):
                raw = bytearray(Path(pin["path"]).read_bytes())
                offset = raw.rfind(b"PK\x05\x06")
                raw[offset + field:offset + field + length] = value.to_bytes(length, "little")
                path = self.root / (tag + ".zip")
                path.write_bytes(raw)
                held = policy._Held(30, time.monotonic() + 30)
                try:
                    fd, _ = held.open_file(path, policy.MAX_ARCHIVE)
                    with self.assertRaises(StorageError):
                        policy._zip_metadata_preflight(held, fd, len(raw))
                finally:
                    held.close(sys.exc_info()[1])


class DefaultBranchTests(unittest.TestCase):
    def test_none_preserves_original_sampler_before_owner_observation(self):
        manager = object.__new__(ManagedBuild)
        manager.output_preservation = None
        manager._assert_owner = Mock(side_effect=AssertionError("new owner observation in default branch"))
        with patch.object(outputs, "sample_output_budget", return_value=123) as sample:
            self.assertEqual(manager.sample_output_budget(), 123)
        sample.assert_called_once_with()
        manager._assert_owner.assert_not_called()

    def test_run_managed_none_keeps_original_sampler_with_legacy_manager_consumer(self):
        import run_managed
        manager = types.SimpleNamespace(acquire=lambda: None, environment=lambda: {},
            finish=lambda **arguments: {"fixture_only": True}, root=Path.cwd(),
            source=Path.cwd(), total_limit=12 * GiB)
        def supervisor(command, source, environment, timeout, check_budget):
            check_budget()
            return 0, True, None  # Mock component scope, no actual child/closure witness.
        with patch.object(run_managed, "ManagedBuild", return_value=manager), \
             patch.object(run_managed, "canonical_root", return_value=Path.cwd()), \
             patch.object(run_managed.shutil, "which", return_value=sys.executable), \
             patch.object(run_managed, "tree_size", return_value=0), \
             patch.object(run_managed, "sample_output_budget", return_value=0) as original_sampler, \
             patch.object(run_managed, "run_group", side_effect=supervisor), \
             patch.object(sys, "argv", ["run_managed.py", "--", sys.executable]), \
             patch.object(sys, "stderr", io.StringIO()):
            self.assertEqual(run_managed.main(), 0)
        original_sampler.assert_called_once_with()
        self.assertFalse(hasattr(manager, "sample_output_budget"))


if __name__ == "__main__":
    unittest.main()
