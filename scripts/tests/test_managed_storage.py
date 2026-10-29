import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from managed_storage import (ManagedBuild, StorageError, _record_stamp, checked,
                             read_record, read_record_snapshot, remove_tree, tree_size)
from managed_process import run_group
import managed_outputs as outputs


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rovezero-storage-contract-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base / "source"
        self.source.mkdir()
        self.root = self.base / "owned-build"
        # ManagedBuild.finish/recover_build call the real output finalizer.
        # Keep its default lookup in this fixture instead of the user's live
        # catalog; explicit catalog arguments remain unaffected.
        self.output_catalog = self.base / "outputs"
        self.output_catalog.mkdir()
        catalog_patch = patch("managed_outputs.canonical_catalog", return_value=self.output_catalog)
        self.default_catalog = catalog_patch.start()
        self.addCleanup(catalog_patch.stop)
        self.model = self.source / "original-model.bin"
        self.model.write_bytes(b"retained-original")

    def acquire(self, source=None, **limits):
        return ManagedBuild(self.root, source or self.source, **limits).acquire()

    def sampled_identity_fixture(self, expected):
        target = self.base / "retained-output"
        target.mkdir(exist_ok=True)
        payload = target / "evidence.txt"
        payload.write_bytes(b"original retained evidence")
        record = self.output_catalog / (("a" * 64) + ".json")
        row = {"schema": outputs.SCHEMA, "key": "a" * 64,
               "output": str(target), "identity": expected, "reserved_bytes": 4096,
               "deferred": True}
        raw = json.dumps(row).encode()
        record.write_bytes(raw)
        return target, payload, record, raw

    def test_sampled_identity_failure_keeps_original_error_and_one_identity_call(self):
        expected, observed = [67, (1 << 100) + 11], [193, (1 << 100) + 11]
        target, payload, record, raw = self.sampled_identity_fixture(expected)
        with patch.object(outputs, "identity", return_value=observed) as identity_call:
            with patch.object(outputs, "tree_size") as size_call:
                with self.assertRaisesRegex(StorageError, "^live output identity changed$") as caught:
                    outputs.sample_output_budget(self.output_catalog)
        self.assertIs(type(caught.exception), StorageError)
        self.assertEqual(caught.exception.args, ("live output identity changed",))
        identity_call.assert_called_once_with(checked(target))
        size_call.assert_not_called()  # The original failure still precedes size sampling.
        observation = caught.exception.managed_output_identity_observation
        self.assertEqual(observation["record_key"], "a" * 64)
        self.assertEqual(observation["normalized_target"], str(checked(target)))
        self.assertEqual(observation["expected_identity"], expected)
        self.assertEqual(observation["observed_identity"], observed)
        self.assertIs(observation["read_only_observation"], True)
        for key in ("identity_equivalence_attested", "storage_lifetime_authority_issued",
                    "recovery_authorized", "deletion_authorized", "physical_completion_attested",
                    "performance_evidence"):
            self.assertIs(observation[key], False)
        self.assertEqual(record.read_bytes(), raw)
        self.assertEqual(payload.read_bytes(), b"original retained evidence")
        self.assertEqual(expected, [67, (1 << 100) + 11])
        self.assertEqual(observed, [193, (1 << 100) + 11])

    def test_sampled_identity_success_and_none_keep_original_identity_call_count(self):
        for expected in ([67, 11], None):
            with self.subTest(expected=expected):
                target, _payload, record, raw = self.sampled_identity_fixture(expected)
                with patch.object(outputs, "identity", return_value=expected) as identity_call:
                    with patch.object(outputs, "tree_size", return_value=23) as size_call:
                        self.assertEqual(outputs.sample_output_budget(self.output_catalog), 23)
                if expected is None:
                    identity_call.assert_not_called()
                else:
                    identity_call.assert_called_once_with(checked(target))
                size_call.assert_called_once_with(checked(target), allow_disappearing=True)
                self.assertEqual(record.read_bytes(), raw)

    def test_supervisor_sampling_keeps_unrelated_identity_and_reservation_failures_isolated(self):
        target, payload, record, raw = self.sampled_identity_fixture([67, 11])
        scope = {"root": str(self.root), "owner": {"token": "current"},
                 "scope": "inherited_outer_process_group_or_job"}
        with patch.object(outputs, "identity") as identity_call, \
             patch.object(outputs, "tree_size") as size_call, \
             patch.object(sys, "stderr", new_callable=io.StringIO) as log:
            cache = {}
            self.assertEqual(outputs.sample_output_budget(self.output_catalog, supervisor_scope=scope,
                                                          foreign_cache=cache), 4096)
        identity_call.assert_not_called()
        size_call.assert_not_called()
        self.assertIn("reserved accounting", log.getvalue())
        self.assertTrue(cache["__unknown_ownership__"])
        self.assertEqual(record.read_bytes(), raw)
        self.assertTrue(payload.exists())

    def test_supervisor_rechecks_own_output_and_never_walks_foreign_tree(self):
        target, _payload, record, _raw = self.sampled_identity_fixture([67, 11])
        scope = {"root": str(self.root), "owner": {"token": "current"},
                 "scope": "inherited_outer_process_group_or_job"}
        cache = {}
        with patch.object(outputs, "identity", return_value=[67, 11]), \
             patch.object(outputs, "tree_size", return_value=23) as size, \
             patch.object(sys, "stderr", new_callable=io.StringIO) as log:
            for _ in range(3):
                self.assertEqual(outputs.sample_output_budget(self.output_catalog,
                                 supervisor_scope=scope, foreign_cache=cache), 4096)
        self.assertEqual(size.call_count, 0)
        self.assertIn("reserved accounting", log.getvalue())
        row = json.loads(record.read_text())
        row["outer_supervisor"] = scope
        record.write_text(json.dumps(row))
        with patch.object(outputs, "identity", return_value=[67, 11]), \
             patch.object(outputs, "tree_size", return_value=23) as size:
            for _ in range(2):
                self.assertEqual(outputs.sample_output_budget(self.output_catalog,
                                 supervisor_scope=scope, foreign_cache=cache), 23)
        self.assertEqual(size.call_count, 2)

    def test_supervised_row_owner_change_or_disappearance_is_not_success(self):
        target, _payload, record, _raw = self.sampled_identity_fixture([67, 11])
        scope = {"root": str(self.root), "owner": {"token": "current"},
                 "scope": "inherited_outer_process_group_or_job"}
        row = json.loads(record.read_text())
        row["outer_supervisor"] = scope
        record.write_text(json.dumps(row))
        cache = {}
        with patch.object(outputs, "identity", return_value=[67, 11]):
            outputs.sample_output_budget(self.output_catalog, supervisor_scope=scope, foreign_cache=cache)
            row["outer_supervisor"] = None
            record.write_text(json.dumps(row))
            with self.assertRaisesRegex(StorageError, "disappeared or changed owner"):
                outputs.sample_output_budget(self.output_catalog, supervisor_scope=scope, foreign_cache=cache)
            record.unlink()
            with self.assertRaisesRegex(StorageError, "disappeared or changed owner"):
                outputs.sample_output_budget(self.output_catalog, supervisor_scope=scope, foreign_cache=cache)

    def test_marker_retaining_owner_detects_forged_catalog_owner(self):
        target, _payload, record, _raw = self.sampled_identity_fixture([67, 11])
        scope = {"root": str(self.root), "owner": {"token": "current"},
                 "scope": "inherited_outer_process_group_or_job"}
        row = json.loads(record.read_text())
        row["deferred"] = False
        row["outer_supervisor"] = scope
        (target / ".rovezero-output.json").write_text(json.dumps(row))
        row["outer_supervisor"] = None
        record.write_text(json.dumps(row))
        with self.assertRaisesRegex(StorageError, "owner or marker changed"):
            outputs.sample_output_budget(self.output_catalog, supervisor_scope=scope, foreign_cache={})

    def test_bounded_reobservation_accepts_transient_marker_publication(self):
        _target, _payload, record, _raw = self.sampled_identity_fixture([67, 11])
        scope = {"root": str(self.root), "owner": {"token": "current"},
                 "scope": "inherited_outer_process_group_or_job"}
        old = json.loads(record.read_text())
        old.update(deferred=False, outer_supervisor=scope, state="active")
        new = {**old, "state": "retained"}
        with patch.object(outputs, "_read", side_effect=[old.copy(), new.copy(), new.copy(), new.copy()]), \
             patch.object(outputs, "identity", return_value=[67, 11]), \
             patch.object(outputs, "tree_size", return_value=23), \
             patch.object(outputs.time, "sleep") as sleep:
            self.assertEqual(outputs.sample_output_budget(self.output_catalog, supervisor_scope=scope,
                                                          foreign_cache={}), 23)
        sleep.assert_called_once_with(0.005)

    def test_unreadable_foreign_marker_is_recorded_as_unknown_custody(self):
        _target, _payload, record, _raw = self.sampled_identity_fixture([67, 11])
        row = json.loads(record.read_text())
        row["deferred"] = False
        record.write_text(json.dumps(row))
        scope = {"root": str(self.root), "owner": {"token": "current"},
                 "scope": "inherited_outer_process_group_or_job"}
        cache = {}
        with patch.object(outputs, "identity", return_value=[67, 11]), \
             patch.object(sys, "stderr", new_callable=io.StringIO):
            outputs.sample_output_budget(self.output_catalog, supervisor_scope=scope, foreign_cache=cache)
        self.assertTrue(cache["__unknown_ownership__"])

    def test_output_parent_identity_alias_of_protected_root_is_rejected(self):
        protected = self.base / "protected"
        alias = self.base / "different-path"
        protected.mkdir()
        alias.mkdir()
        original_stat = Path.stat
        shared = SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_dev=9, st_ino=44)
        def stat_with_alias(path, *args, **kwargs):
            if path == protected or path == alias:
                return shared
            return original_stat(path, *args, **kwargs)
        with patch.object(Path, "stat", autospec=True, side_effect=stat_with_alias):
            with self.assertRaisesRegex(StorageError, "directory identity"):
                outputs._reject_identity_ancestor_overlap(alias / "new", protected)

    def test_run_managed_preserves_budget_failure_after_original_supervisor_stringifies_it(self):
        import run_managed
        target, payload, record, raw = self.sampled_identity_fixture([67, 11])
        original_errors, finished = [], []

        def original_supervisor(_command, _source, _environment, _timeout, check_budget):
            try:
                check_budget()
            except StorageError as error:
                original_errors.append(error)
                # Existing run_group converts the original exception to its
                # message after attempting the original process-tree cleanup.
                return 125, False, str(error)
            self.fail("original sample guard did not reject the changed identity")

        def finish(**arguments):
            finished.append(arguments)
            return {"failure": arguments["supervisor_error"], "tree_gone": arguments["tree_gone"]}

        manager = SimpleNamespace(acquire=lambda: None, environment=lambda: {}, finish=finish,
                                  root=self.root, source=self.source, total_limit=12 << 30,
                                  _identity=lambda: {"token": "current"})
        row = json.loads(record.read_text())
        row["outer_supervisor"] = {"root": str(self.root), "owner": manager._identity(),
                                   "scope": "inherited_outer_process_group_or_job"}
        raw = json.dumps(row).encode()
        record.write_bytes(raw)
        error_stream = io.StringIO()
        with patch.object(run_managed, "ManagedBuild", return_value=manager):
            with patch.object(run_managed, "canonical_root", return_value=self.root):
                with patch.object(run_managed.shutil, "which", return_value=sys.executable):
                    with patch.object(run_managed, "run_group", side_effect=original_supervisor):
                        # Admission sees the original identity; it changes
                        # only after the child was dispatched.
                        with patch.object(outputs, "identity", side_effect=[[67, 11], [193, 11]]) as identity_call:
                            with patch.object(sys, "argv", ["run_managed.py", "--", sys.executable]):
                                with patch.object(sys, "stderr", error_stream):
                                    self.assertEqual(run_managed.main(), 125)
        self.assertEqual(identity_call.call_count, 2)
        self.assertEqual(identity_call.call_args_list[0].args, (checked(target),))
        self.assertEqual(identity_call.call_args_list[1].args, (checked(target),))
        self.assertEqual(len(original_errors), 1)
        self.assertEqual(finished, [{"exit_code": 125, "tree_gone": False,
                                     "supervisor_error": "live output identity changed"}])
        lines = error_stream.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("managed_storage="))
        self.assertEqual(json.loads(lines[1]), original_errors[0].managed_output_identity_observation)
        self.assertEqual(record.read_bytes(), raw)
        self.assertEqual(payload.read_bytes(), b"original retained evidence")

    def test_default_output_budget_failure_is_detected_before_child_dispatch(self):
        import run_managed
        original = StorageError("pre-existing output accounting failure")
        manager = SimpleNamespace(acquire=Mock(), environment=Mock(return_value={}),
                                  finish=Mock(return_value={}), root=self.root,
                                  source=self.source, total_limit=12 << 30,
                                  _identity=lambda: {"token": "current"})
        with patch.object(run_managed, "ManagedBuild", return_value=manager), \
             patch.object(run_managed, "canonical_root", return_value=self.root), \
             patch.object(run_managed.shutil, "which", return_value=sys.executable), \
             patch.object(run_managed, "sample_output_budget", side_effect=original) as sample, \
             patch.object(run_managed, "run_group") as dispatch, \
             patch.object(run_managed, "_emit_storage_failure_observation") as observed, \
             patch.object(sys, "argv", ["run_managed.py", "--", sys.executable]), \
             patch.object(sys, "stderr", io.StringIO()):
            self.assertEqual(run_managed.main(), 125)
        sample.assert_called_once_with(supervisor_scope={"root": str(self.root), "owner": {"token": "current"},
                                                       "scope": "inherited_outer_process_group_or_job"},
                                       foreign_cache={})
        dispatch.assert_not_called()
        manager.acquire.assert_called_once_with()
        manager.finish.assert_called_once_with(exit_code=125, tree_gone=True, _not_dispatched=True)
        observed.assert_called_once_with(original)
        self.assertEqual(original.args, ("pre-existing output accounting failure",))

    def test_default_output_budget_is_still_sampled_during_execution(self):
        import run_managed
        manager = SimpleNamespace(acquire=Mock(), environment=Mock(return_value={}),
                                  finish=Mock(return_value={}), root=self.root,
                                  source=self.source, total_limit=12 << 30,
                                  _identity=lambda: {"token": "current"})
        def supervise(_command, _source, _environment, _timeout, check_budget):
            check_budget()
            return 0, True, None
        with patch.object(run_managed, "ManagedBuild", return_value=manager), \
             patch.object(run_managed, "canonical_root", return_value=self.root), \
             patch.object(run_managed.shutil, "which", return_value=sys.executable), \
             patch.object(run_managed, "sample_output_budget", side_effect=[0, 0]) as sample, \
             patch.object(run_managed, "tree_size", return_value=0), \
             patch.object(run_managed, "run_group", side_effect=supervise) as dispatch, \
             patch.object(sys, "argv", ["run_managed.py", "--", sys.executable]), \
             patch.object(sys, "stderr", io.StringIO()):
            self.assertEqual(run_managed.main(), 0)
        self.assertEqual(sample.call_count, 2)
        dispatch.assert_called_once()
        manager.finish.assert_called_once_with(exit_code=0, tree_gone=True, supervisor_error=None)

    def test_predispatch_corrupt_foreign_catalog_keeps_evidence_and_closes_own_build(self):
        import run_managed
        manager = ManagedBuild(self.root, self.source)
        foreign = self.output_catalog / ("f" * 64 + ".json")
        foreign.write_bytes(b"{broken")
        with patch.object(run_managed, "ManagedBuild", return_value=manager), \
             patch.object(run_managed, "canonical_root", return_value=self.root), \
             patch.object(run_managed.shutil, "which", return_value=sys.executable), \
             patch.object(run_managed, "sample_output_budget", side_effect=StorageError("pre-dispatch observation failed")), \
             patch.object(run_managed, "run_group") as dispatch, \
             patch.object(sys, "argv", ["run_managed.py", "--", sys.executable]), \
             patch.object(sys, "stderr", io.StringIO()):
            self.assertEqual(run_managed.main(), 125)
        dispatch.assert_not_called()
        self.assertFalse(manager.lock.exists())
        self.assertEqual(foreign.read_bytes(), b"{broken")
        receipt = json.loads((manager.slot / "last-run.json").read_text())
        self.assertEqual(receipt["output_finalization"], "not_needed_no_child_dispatched")

    def test_unreadable_unrelated_catalog_row_does_not_stop_execution_or_claim_custody(self):
        import run_managed
        manager = ManagedBuild(self.root, self.source)
        foreign = self.output_catalog / ("f" * 64 + ".json")
        foreign.write_bytes(b"{broken")
        with patch.object(run_managed, "ManagedBuild", return_value=manager), \
             patch.object(run_managed, "canonical_root", return_value=self.root), \
             patch.object(run_managed.shutil, "which", return_value=sys.executable), \
             patch.object(run_managed, "run_group", return_value=(0, True, None)), \
             patch.object(sys, "argv", ["run_managed.py", "--", sys.executable]), \
             patch.object(sys, "stderr", io.StringIO()):
            self.assertEqual(run_managed.main(), 0)
        self.assertFalse(manager.lock.exists())
        self.assertEqual(foreign.read_bytes(), b"{broken")
        receipt = json.loads((manager.slot / "last-run.json").read_text())
        self.assertEqual(receipt["output_custody_scope"], "unknown")

    def test_no_dispatch_closure_refuses_any_launch_attempt(self):
        manager = self.acquire()
        manager._dispatch_started = True
        with self.assertRaisesRegex(StorageError, "no-dispatch closure"):
            manager.finish(exit_code=125, tree_gone=True, _not_dispatched=True)
        self.assertTrue(manager.lock.exists())
        manager.finish(exit_code=125, tree_gone=True)

    def test_run_managed_closes_original_owner_for_every_pre_dispatch_failure(self):
        import run_managed
        for phase in ("environment", "sample_output_budget"):
            for error_type in (StorageError, OSError, RuntimeError, KeyboardInterrupt):
                with self.subTest(phase=phase, error_type=error_type.__name__):
                    original = error_type("original pre-dispatch " + phase)
                    manager = SimpleNamespace(acquire=Mock(), environment=Mock(return_value={}),
                        sample_output_budget=Mock(), finish=Mock(return_value={}),
                        root=self.root, source=self.source, total_limit=12 << 30)
                    getattr(manager, phase).side_effect = original
                    preservation = object()
                    with patch.object(run_managed, "ManagedBuild", return_value=manager), \
                         patch.object(run_managed, "canonical_root", return_value=self.root), \
                         patch.object(run_managed.shutil, "which", return_value=sys.executable), \
                         patch.object(run_managed, "run_group") as dispatch, \
                         patch.object(run_managed, "_emit_storage_failure_observation") as observed, \
                         patch.object(sys, "argv", ["run_managed.py", "--", sys.executable]), \
                         patch.object(sys, "stderr", io.StringIO()):
                        if error_type is KeyboardInterrupt:
                            with self.assertRaises(KeyboardInterrupt) as caught:
                                run_managed.main(output_preservation=preservation)
                            self.assertIs(caught.exception, original)
                            observed.assert_not_called()
                        else:
                            self.assertEqual(run_managed.main(output_preservation=preservation), 125)
                            observed.assert_called_once_with(original)
                    manager.acquire.assert_called_once_with()
                    manager.finish.assert_called_once_with(exit_code=125, tree_gone=True)
                    dispatch.assert_not_called()
                    if phase == "environment":
                        manager.sample_output_budget.assert_not_called()
                    else:
                        manager.sample_output_budget.assert_called_once_with()
                    self.assertEqual(original.args, ("original pre-dispatch " + phase,))

    def test_run_managed_pre_dispatch_error_releases_only_real_fixture_owner(self):
        import run_managed
        actual = ManagedBuild(self.root, self.source)
        original = OSError("actual fixture environment failed before dispatch")
        original_owner = {}

        def fail_environment():
            original_owner.update(actual._identity())
            raise original

        with patch.object(run_managed, "ManagedBuild", return_value=actual), \
             patch.object(run_managed, "canonical_root", return_value=self.root), \
             patch.object(run_managed.shutil, "which", return_value=sys.executable), \
             patch.object(actual, "environment", side_effect=fail_environment) as environment, \
             patch.object(run_managed, "run_group", side_effect=AssertionError("fixture child dispatch forbidden")) as dispatch, \
             patch.object(sys, "argv", ["run_managed.py", "--", sys.executable]), \
             patch.object(sys, "stderr", io.StringIO()):
            self.assertEqual(run_managed.main(), 125)
        environment.assert_called_once_with()
        dispatch.assert_not_called()
        self.assertTrue(actual.finished)
        self.assertFalse(actual.lock.exists())
        self.assertFalse(actual.scratch.exists())
        record = read_record(actual.slot / "record.json")
        self.assertEqual(record["state"], "complete")
        self.assertGreater(record["finished_ns"], 0)
        self.assertEqual({key: value for key, value in record.items()
                          if key not in ("state", "finished_ns")}, original_owner)
        self.assertEqual(read_record(self.root / "closure.json"),
                         {"schema": original_owner["schema"], "owner": original_owner, "tree_gone": True})
        self.assertEqual(self.model.read_bytes(), b"retained-original")
        self.assertEqual({path.name for path in self.output_catalog.iterdir()}, set())

    def test_run_managed_pre_dispatch_cleanup_preserves_first_error_and_cleanup_note(self):
        import run_managed
        for error_type in (OSError, KeyboardInterrupt):
            for cleanup_type in (RuntimeError, KeyboardInterrupt):
                with self.subTest(error_type=error_type.__name__, cleanup_type=cleanup_type.__name__):
                    original = error_type("first environment failure")
                    cleanup = cleanup_type("independent original-manager finish failure")
                    manager = SimpleNamespace(acquire=Mock(), environment=Mock(side_effect=original),
                        finish=Mock(side_effect=cleanup), root=self.root, source=self.source,
                        total_limit=12 << 30)
                    error_stream = io.StringIO()
                    with patch.object(run_managed, "ManagedBuild", return_value=manager), \
                         patch.object(run_managed, "canonical_root", return_value=self.root), \
                         patch.object(run_managed.shutil, "which", return_value=sys.executable), \
                         patch.object(run_managed, "run_group") as dispatch, \
                         patch.object(run_managed, "_emit_storage_failure_observation") as observed, \
                         patch.object(sys, "argv", ["run_managed.py", "--", sys.executable]), \
                         patch.object(sys, "stderr", error_stream):
                        if error_type is KeyboardInterrupt:
                            with self.assertRaises(KeyboardInterrupt) as caught:
                                run_managed.main()
                            self.assertIs(caught.exception, original)
                        else:
                            self.assertEqual(run_managed.main(), 125)
                            observed.assert_called_once_with(original)
                    dispatch.assert_not_called()
                    manager.finish.assert_called_once_with(exit_code=125, tree_gone=True, _not_dispatched=True)
                    self.assertEqual(original.args, ("first environment failure",))
                    self.assertEqual(original.__notes__, [
                        "pre-dispatch original manager cleanup failed: " + cleanup_type.__name__
                        + ": independent original-manager finish failure"])
                    lines = error_stream.getvalue().splitlines()
                    diagnostic = [line for line in lines if line.startswith("managed_storage_cleanup_error=")]
                    self.assertEqual(len(diagnostic), 1)
                    self.assertLessEqual(len(diagnostic[0].encode("utf-8")) + 1, 16 << 10)
                    self.assertEqual(json.loads(diagnostic[0].split("=", 1)[1]), {
                        "stage": "pre_dispatch_original_manager_finish", "type": cleanup_type.__name__,
                        "detail": "independent original-manager finish failure"})

    def test_run_managed_cleanup_diagnostic_closed_pipe_preserves_original_failure(self):
        import run_managed
        for error_type in (OSError, KeyboardInterrupt):
            with self.subTest(error_type=error_type.__name__):
                original = error_type("first pre-dispatch failure with closed stderr")
                cleanup = RuntimeError("cleanup failed with closed stderr")
                manager = SimpleNamespace(acquire=Mock(), environment=Mock(side_effect=original),
                    finish=Mock(side_effect=cleanup), root=self.root, source=self.source, total_limit=12 << 30)
                closed_stream = io.StringIO()
                closed_stream.close()
                with patch.object(run_managed, "ManagedBuild", return_value=manager), \
                     patch.object(run_managed, "canonical_root", return_value=self.root), \
                     patch.object(run_managed.shutil, "which", return_value=sys.executable), \
                     patch.object(run_managed, "run_group") as dispatch, \
                     patch.object(run_managed, "_emit_storage_failure_observation") as observed, \
                     patch.object(sys, "argv", ["run_managed.py", "--", sys.executable]), \
                     patch.object(sys, "stderr", closed_stream):
                    if error_type is KeyboardInterrupt:
                        with self.assertRaises(KeyboardInterrupt) as caught:
                            run_managed.main()
                        self.assertIs(caught.exception, original)
                        observed.assert_not_called()
                    else:
                        self.assertEqual(run_managed.main(), 125)
                        observed.assert_called_once_with(original)
                dispatch.assert_not_called()
                manager.finish.assert_called_once_with(exit_code=125, tree_gone=True, _not_dispatched=True)
                self.assertEqual(original.args, ("first pre-dispatch failure with closed stderr",))
                self.assertEqual(original.__notes__, [
                    "pre-dispatch original manager cleanup failed: RuntimeError: cleanup failed with closed stderr"])

    def test_cleanup_diagnostic_has_one_bounded_line_for_long_non_ascii_detail(self):
        import run_managed
        error_stream = io.StringIO()
        with patch.object(sys, "stderr", error_stream):
            run_managed._emit_storage_cleanup_failure(RuntimeError("\U0001f642" * 10000))
        lines = error_stream.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertLessEqual(len(lines[0].encode("utf-8")) + 1, 16 << 10)
        self.assertEqual(json.loads(lines[0].split("=", 1)[1])["detail"], "\U0001f642" * 1024)

    def test_run_managed_does_not_synthesize_no_child_finish_after_dispatch_exception(self):
        import run_managed
        for error_type in (StorageError, OSError, RuntimeError, KeyboardInterrupt):
            with self.subTest(error_type=error_type.__name__):
                original = error_type("ambiguous dispatched supervisor failure")
                manager = SimpleNamespace(acquire=Mock(), environment=Mock(return_value={}),
                    sample_output_budget=Mock(), finish=Mock(), root=self.root,
                    source=self.source, total_limit=12 << 30)
                with patch.object(run_managed, "ManagedBuild", return_value=manager), \
                     patch.object(run_managed, "canonical_root", return_value=self.root), \
                     patch.object(run_managed.shutil, "which", return_value=sys.executable), \
                     patch.object(run_managed, "run_group", side_effect=original) as dispatch, \
                     patch.object(sys, "argv", ["run_managed.py", "--", sys.executable]), \
                     patch.object(sys, "stderr", io.StringIO()):
                    if error_type is KeyboardInterrupt:
                        with self.assertRaises(KeyboardInterrupt) as caught:
                            run_managed.main(output_preservation=object())
                        self.assertIs(caught.exception, original)
                    else:
                        self.assertEqual(run_managed.main(output_preservation=object()), 125)
                dispatch.assert_called_once()
                manager.finish.assert_not_called()
                manager.sample_output_budget.assert_called_once_with()
                self.assertEqual(original.args, ("ambiguous dispatched supervisor failure",))

    def test_storage_observation_write_failure_does_not_replace_original_error(self):
        import run_managed
        error = StorageError("live output identity changed")
        outputs._attach_sampled_identity_observation(error, "a" * 64, self.base, [67, 11], [193, 11])
        with patch.object(sys, "stderr", SimpleNamespace(write=lambda _value: (_ for _ in ()).throw(OSError("closed fixture pipe")))):
            run_managed._emit_storage_failure_observation(error)
        self.assertEqual(error.args, ("live output identity changed",))

    def test_finish_real_finalizer_uses_only_the_fixture_default_catalog(self):
        owner = self.acquire()
        (owner.scratch / "owned-remnant").write_bytes(b"fixture temporary")
        result = owner.finish(exit_code=0, tree_gone=True)
        self.default_catalog.assert_called_once_with()
        self.assertEqual({path.name for path in self.output_catalog.iterdir()}, {"catalog.lock"})
        self.assertTrue(result["tree_cleanup_verified"])
        self.assertTrue(result["temporary_removed"])
        self.assertFalse(owner.lock.exists())
        self.assertFalse(owner.scratch.exists())
        self.assertEqual(self.model.read_bytes(), b"retained-original")

    def test_snapshot_preserves_raw_hash_and_original_fd_stamp(self):
        record = self.source / "owner.json"
        raw = b'{"schema":"rovezero.managed-build.v1","owner":"original"}\n'
        record.write_bytes(raw)
        with record.open("rb") as stream:
            original = _record_stamp(os.fstat(stream.fileno()))
        snapshot = read_record_snapshot(record)
        self.assertEqual(snapshot, (json.loads(raw), hashlib.sha256(raw).hexdigest(), original))
        self.assertEqual(read_record_snapshot(record), snapshot)

    def test_windows_cross_source_creation_time_does_not_ignore_file_identity(self):
        # The actual CPython 3.12+ path stat uses creation time for ctime,
        # whereas fstat uses ChangeTime. Model both without depending on OS.
        fields = dict(st_dev=7, st_ino=(1 << 100) + 11, st_mode=stat.S_IFREG | 0o666,
                      st_nlink=1, st_size=40, st_mtime_ns=1000,
                      st_ctime_ns=2000, st_birthtime_ns=900)
        opened = SimpleNamespace(**fields)
        named = SimpleNamespace(**{**fields, "st_ctime_ns": fields["st_birthtime_ns"]})
        self.assertNotEqual(_record_stamp(opened), _record_stamp(named))
        self.assertEqual(_record_stamp(opened, creation_time=True),
                         _record_stamp(named, creation_time=True))
        for field in ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                      "st_mtime_ns", "st_birthtime_ns"):
            with self.subTest(field=field):
                changed = SimpleNamespace(**{**vars(named), field: getattr(named, field) + 1})
                self.assertNotEqual(_record_stamp(opened, creation_time=True),
                                    _record_stamp(changed, creation_time=True))

    def test_snapshot_rejects_fd_change_time_mutation(self):
        record = self.source / "owner.json"
        record.write_bytes(b'{"schema":"rovezero.managed-build.v1"}\n')
        fstat = os.fstat
        calls = 0

        def changed_after_read(fd):
            nonlocal calls
            info = fstat(fd)
            calls += 1
            if calls == 1:
                return info
            fields = {name: getattr(info, name) for name in (
                "st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                "st_mtime_ns", "st_ctime_ns")}
            fields["st_ctime_ns"] += 1
            if hasattr(info, "st_birthtime_ns"):
                fields["st_birthtime_ns"] = info.st_birthtime_ns
            return SimpleNamespace(**fields)

        with patch("managed_storage.os.fstat", side_effect=changed_after_read):
            with self.assertRaisesRegex(StorageError, "identity changed while reading"):
                read_record_snapshot(record)
        self.assertEqual(record.read_bytes(), b'{"schema":"rovezero.managed-build.v1"}\n')

    def test_fixed_slot_reuse_and_cleanup_even_after_command_failure(self):
        first = self.acquire()
        (first.target / "reusable").write_bytes(b"cache")
        (first.scratch / "test-remnant").write_bytes(b"temporary")
        with patch.dict(os.environ, {"CARGO_TARGET_DIR": str(first.target)}):
            env = first.environment()
        self.assertEqual(env["CARGO_INCREMENTAL"], "0")
        self.assertEqual(env["TMPDIR"], str(first.scratch))
        first.finish(exit_code=7, tree_gone=True)
        self.assertFalse(first.scratch.exists())
        second = self.acquire()
        self.assertEqual(first.slot, second.slot)
        self.assertEqual((second.target / "reusable").read_bytes(), b"cache")
        second.finish(exit_code=0, tree_gone=True)
        self.assertEqual(self.model.read_bytes(), b"retained-original")

    @unittest.skipUnless(os.name == "nt", "Windows CreateProcessW start boundary")
    def test_definite_windows_non_start_releases_only_its_owned_lease_and_scratch(self):
        owner = self.acquire()
        invalid_program = self.source / "invalid-managed-program.exe"
        invalid_program.write_bytes(b"not a Windows executable\n")
        (owner.target / "retained-cache").write_bytes(b"cache")
        (owner.scratch / "unstarted-remnant").write_bytes(b"owned temporary")
        with patch.dict(os.environ, {"CARGO_TARGET_DIR": str(owner.target)}):
            environment = owner.environment()
        code, gone, failure = run_group(
            [str(invalid_program)], self.source, environment, 5, lambda: None)
        self.assertEqual((code, gone), (125, True))
        self.assertEqual(failure["kind"], "windows_process_not_started")
        self.assertIn(failure["winerror"], (193, 216))
        self.assertIn(str(failure["winerror"]), failure["message"])
        receipt = owner.finish(exit_code=code, tree_gone=gone, supervisor_error=failure)
        self.assertTrue(receipt["temporary_removed"])
        self.assertFalse(owner.scratch.exists())
        self.assertFalse(owner.lock.exists())
        self.assertEqual(receipt["supervisor_error"], failure)
        self.assertEqual((owner.target / "retained-cache").read_bytes(), b"cache")
        self.assertEqual(self.model.read_bytes(), b"retained-original")

    def test_post_run_budget_reset_preserves_source(self):
        owner = self.acquire(slot_limit=16_384, total_limit=49_152)
        (owner.target / "too-large").write_bytes(b"x" * 20_000)
        receipt = owner.finish(exit_code=0, tree_gone=True)
        self.assertTrue(receipt["post_run_budget_reset"])
        self.assertFalse(owner.target.exists())
        self.assertLessEqual(tree_size(owner.slot), owner.slot_limit)
        self.assertTrue(self.model.exists())

    @unittest.skipUnless(os.name == "posix", "POSIX directory deletion permissions")
    def test_readonly_owned_directories_are_removed_after_verified_finish(self):
        owner = self.acquire()
        nested = owner.scratch / "frozen-parent"
        nested.mkdir()
        retained = nested / "private-input"
        retained.write_bytes(b"owned regenerable input")
        retained.chmod(0o400)
        nested.chmod(0o500)
        owner.scratch.chmod(0o500)
        try:
            receipt = owner.finish(exit_code=0, tree_gone=True)
            self.assertTrue(receipt["temporary_removed"])
            self.assertFalse(owner.scratch.exists())
            self.assertFalse(owner.lock.exists())
            self.assertEqual(self.model.read_bytes(), b"retained-original")
        finally:
            # Only test-fixture cleanup if the regression is still present.
            for directory in (owner.scratch, nested):
                if directory.exists():
                    directory.chmod(0o700)

    @unittest.skipUnless(os.name == "posix", "POSIX directory deletion permissions")
    def test_readonly_parent_retry_repairs_directory_without_chmod_shared_file(self):
        owner = self.acquire()
        nested = owner.scratch / "frozen-parent"
        nested.mkdir()
        target = nested / "private-input"
        target.write_bytes(b"owned")
        target.chmod(0o400)
        nested.chmod(0o500)
        original = PermissionError("unlink requires writable parent")
        import managed_storage
        real_rmtree = managed_storage.shutil.rmtree
        retries = []

        def unlink_after_repair(name):
            retries.append(name)
            self.assertTrue(stat.S_IMODE(nested.stat().st_mode) & stat.S_IWUSR)
            self.assertTrue(stat.S_IMODE(nested.stat().st_mode) & stat.S_IXUSR)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o400)
            os.unlink(name)

        def permission_failure(path, *, onerror):
            # Exercise the same failure even when CI runs as root and bypasses
            # normal permission checks; the retry checks the real directory mode.
            onerror(unlink_after_repair, str(target), (PermissionError, original, None))
            real_rmtree(path, onerror=onerror)

        try:
            with patch("managed_storage.shutil.rmtree", side_effect=permission_failure):
                remove_tree(owner.scratch, owner.slot)
            self.assertEqual(retries, [str(target)])
            self.assertFalse(owner.scratch.exists())
            self.assertEqual(self.model.read_bytes(), b"retained-original")
        finally:
            if nested.exists():
                nested.chmod(0o700)

    def test_cleanup_retry_preserves_original_permission_failure(self):
        owner = self.acquire()
        target = owner.scratch / "private-input"
        target.write_bytes(b"owned")
        original = PermissionError("initial unlink failure")
        retry = OSError("retry failed")

        def retry_failure(_name):
            raise retry

        def permission_failure(_path, *, onerror):
            onerror(retry_failure, str(target), (PermissionError, original, None))

        with patch("managed_storage.shutil.rmtree", side_effect=permission_failure):
            with self.assertRaises(OSError) as failure:
                remove_tree(owner.scratch, owner.slot)
        self.assertIs(failure.exception, retry)
        self.assertIs(failure.exception.__cause__, original)
        self.assertTrue(target.exists())
        self.assertTrue(owner.lock.exists())

    @unittest.skipUnless(os.name == "posix", "POSIX no-follow permission repair")
    def test_cleanup_changed_parent_link_is_rejected_before_permission_repair(self):
        owner = self.acquire()
        nested = owner.scratch / "frozen-parent"
        nested.mkdir()
        target = nested / self.model.name
        target.write_bytes(b"owned regenerable copy")
        nested.chmod(0o500)
        saved = owner.scratch / "saved-parent"
        source_mode = stat.S_IMODE(self.source.stat().st_mode)
        original = PermissionError("initial unlink failure")

        def changed_link(_path, *, onerror):
            # The full tree was already preflighted. A changed link must never
            # give permission repair authority over the external source tree.
            nested.rename(saved)
            nested.symlink_to(self.source, target_is_directory=True)
            onerror(os.unlink, str(target), (PermissionError, original, None))

        try:
            with patch("managed_storage.shutil.rmtree", side_effect=changed_link):
                with self.assertRaises(StorageError) as failure:
                    remove_tree(owner.scratch, owner.slot)
            self.assertIs(failure.exception.__cause__, original)
            self.assertEqual(stat.S_IMODE(self.source.stat().st_mode), source_mode)
            self.assertEqual(self.model.read_bytes(), b"retained-original")
            self.assertTrue((saved / self.model.name).exists())
            self.assertTrue(owner.lock.exists())
        finally:
            if nested.is_symlink():
                nested.unlink()  # Test fixture cleanup only; do not follow it.
            if saved.exists():
                saved.rename(nested)
            if nested.exists():
                nested.chmod(0o700)

    def test_pre_run_oversize_cache_and_aggregate_retention(self):
        first = self.acquire(slot_limit=4096, total_limit=6144)
        first.finish(exit_code=0, tree_gone=True)
        (first.target / "oversize-completed-cache").write_bytes(b"x" * 10_000)
        reuse = self.acquire(slot_limit=4096, total_limit=6144)
        self.assertFalse((reuse.target / "oversize-completed-cache").exists())
        (reuse.target / "within-budget").write_bytes(b"x" * 3000)
        reuse.finish(exit_code=0, tree_gone=True)
        other = self.base / "other-source"
        other.mkdir()
        second = self.acquire(other, slot_limit=4096, total_limit=6144)
        (second.target / "within-budget").write_bytes(b"x" * 3000)
        second.finish(exit_code=0, tree_gone=True)
        self.assertFalse(first.slot.exists())
        self.assertTrue(second.slot.exists())
        self.assertLessEqual(tree_size(self.root / "slots"), 6144)

    def test_live_and_abandoned_owner_are_not_stolen(self):
        owner = self.acquire()
        for _ in range(2):
            with self.assertRaises(StorageError):
                self.acquire()
            self.assertTrue(owner.lock.exists())
        owner.finish(exit_code=0, tree_gone=True)

    def test_unverified_descendants_retain_lease_and_scratch(self):
        owner = self.acquire()
        (owner.scratch / "input").write_bytes(b"still-in-use")
        with self.assertRaises(StorageError):
            owner.finish(exit_code=124, tree_gone=False)
        self.assertTrue(owner.lock.exists())
        self.assertTrue((owner.scratch / "input").exists())

    def test_changed_record_or_unknown_content_preserves_all(self):
        owner = self.acquire()
        record = read_record(owner.slot / "record.json")
        record["token"] = "changed"
        (owner.slot / "record.json").write_text(json.dumps(record))
        with self.assertRaises(StorageError):
            owner.finish(exit_code=0, tree_gone=True)
        self.assertTrue(owner.lock.exists())
        self.assertTrue(owner.scratch.exists())

    def test_unknown_catalog_never_adopted(self):
        self.root.mkdir()
        (self.root / "unowned").mkdir()
        with self.assertRaises(StorageError):
            self.acquire()
        self.assertTrue((self.root / "unowned").is_dir())

    def test_checkout_limit_evicts_only_completed_owned_slot(self):
        slots = []
        for index in range(5):
            source = self.base / f"source-{index}"
            source.mkdir()
            owner = self.acquire(source, slots_limit=4)
            slots.append(owner.slot)
            owner.finish(exit_code=0, tree_gone=True)
        self.assertFalse(slots[0].exists())
        self.assertTrue(all(path.exists() for path in slots[1:]))
        self.assertTrue(self.model.exists())

    def test_link_escape_rejected_before_deleting_any_file(self):
        owner = self.acquire()
        (owner.scratch / "ordinary").write_bytes(b"owned")
        link = owner.scratch / "escape"
        try:
            link.symlink_to(self.source, target_is_directory=True)
        except OSError:
            self.skipTest("symlink creation unavailable on this Windows account")
        with self.assertRaises(StorageError):
            owner.finish(exit_code=0, tree_gone=True)
        self.assertTrue((owner.scratch / "ordinary").exists())
        self.assertEqual(self.model.read_bytes(), b"retained-original")
        link.unlink()  # Test fixture cleanup only.

    def test_credential_and_checkout_destinations_rejected_without_read(self):
        with self.assertRaises(StorageError):
            checked(self.base / ".env" / "unopened")
        with self.assertRaises(StorageError):
            ManagedBuild(self.source / "target", self.source)

    def test_internal_venv_link_is_removed_without_traversal(self):
        owner = self.acquire()
        (owner.scratch / "lib").mkdir()
        (owner.scratch / "lib" / "owned-copy").write_bytes(b"owned")
        link = owner.scratch / "lib64"
        try:
            link.symlink_to("lib", target_is_directory=True)
        except OSError:
            self.skipTest("symlink creation unavailable on this Windows account")
        amount = tree_size(owner.slot)
        self.assertGreater(amount, 0)
        owner.finish(exit_code=0, tree_gone=True)
        self.assertFalse(owner.scratch.exists())
        self.assertEqual(self.model.read_bytes(), b"retained-original")

    def test_live_sampling_allows_disappearance_but_cleanup_preflight_does_not(self):
        original_scandir = os.scandir
        class VanishingScan:
            def __enter__(inner):
                with original_scandir(self.source) as scan:
                    entries = list(scan)
                self.model.unlink()
                return iter(entries)
            def __exit__(inner, *args):
                pass
        for live in (True, False):
            self.model.write_bytes(b"owned test input")
            with patch("managed_storage.os.scandir", return_value=VanishingScan()):
                if live:
                    self.assertEqual(tree_size(self.source, allow_disappearing=True), 0)
                else:
                    with self.assertRaises(FileNotFoundError):
                        tree_size(self.source)


class ProcessTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows CreateProcessW start boundary")
    def test_invalid_windows_executable_is_definitely_not_started(self):
        with tempfile.TemporaryDirectory(prefix="rovezero-invalid-process-") as directory:
            program = Path(directory) / "invalid.exe"
            program.write_bytes(b"not a Windows executable\n")
            admitted = []
            code, gone, failure = run_group(
                [str(program)], directory, os.environ.copy(), 5,
                lambda: admitted.append(True))
            self.assertEqual(code, 125)
            self.assertIs(gone, True)
            self.assertEqual(admitted, [])
            self.assertIsInstance(failure, dict)
            self.assertEqual(failure["kind"], "windows_process_not_started")
            self.assertEqual(failure["stage"], "CreateProcessW")
            self.assertIs(failure["process_created"], False)
            self.assertIn(failure["winerror"], (193, 216))
            self.assertIn(str(failure["winerror"]), failure["message"])
            self.assertEqual(json.loads(json.dumps(failure)), failure)

    @unittest.skipUnless(os.name == "nt", "Windows Job-owned stdlib child")
    def test_windows_stdlib_child_runs_and_keeps_success_result_unchanged(self):
        with tempfile.TemporaryDirectory(prefix="rovezero-stdlib-process-") as directory:
            marker = Path(directory) / "completed"
            program = f"from pathlib import Path; Path({str(marker)!r}).write_text('ran',encoding='utf-8')"
            code, gone, failure = run_group(
                [sys.executable, "-I", "-S", "-c", program],
                directory, os.environ.copy(), 5, lambda: None)
            self.assertEqual(code, 0)
            self.assertIs(gone, True)
            self.assertIsNone(failure)
            self.assertEqual(marker.read_text(encoding="utf-8"), "ran")

    def test_normal_exit_and_orphan_descendant_are_drained(self):
        with tempfile.TemporaryDirectory(prefix="rovezero-owned-process-") as directory:
            marker = Path(directory) / "late-write"
            child = f"import time,pathlib; time.sleep(2); pathlib.Path({str(marker)!r}).write_text('escaped')"
            program = f"import subprocess,sys; subprocess.Popen([sys.executable,'-c',{child!r}])"
            code, gone, failure = run_group(
                [sys.executable, "-c", program], directory, os.environ.copy(), 5, lambda: None)
            self.assertEqual(code, 0)
            self.assertTrue(gone)
            self.assertIsNone(failure)
            self.assertFalse(marker.exists())

    def test_timeout_and_budget_error_return_failure_after_owned_cleanup(self):
        for error in (False, True):
            def budget():
                if error:
                    raise StorageError("synthetic storage limit")
            code, gone, failure = run_group(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                Path.cwd(), os.environ.copy(), 0.3, budget)
            self.assertEqual(code, 125 if error else 124)
            self.assertTrue(gone)
            self.assertEqual(failure is not None, error)


if __name__ == "__main__":
    unittest.main()
