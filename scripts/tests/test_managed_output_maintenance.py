"""After-run cache cleanup tests use only temporary owned fixture trees."""
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import managed_output_maintenance as maintenance
import managed_outputs
import storage_code_retirement
import run_managed
from managed_storage import ManagedBuild, write_record


@unittest.skipUnless(sys.platform == "linux", "requires Linux path/census semantics")
class OutputMaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.source = base / "source"
        self.source.mkdir()
        self.root = base / "build"
        (self.root / "slots").mkdir(parents=True)
        self.catalog = base / "outputs"
        self.catalog.mkdir()
        self.current = self._slot("current")
        self.candidate = self._slot("old")
        self.current.finished = True
        self.patches = [
            patch.object(managed_outputs, "canonical_catalog", return_value=self.catalog),
            patch.object(managed_outputs, "_catalog_rows", return_value=[]),
            patch.object(maintenance, "owner_proven_dead", return_value=True),
            patch.object(maintenance, "reboot_proves_dead", return_value=False),
            patch.object(storage_code_retirement, "_process_census", return_value={"complete": True}),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def _slot(self, purpose, source=None):
        manager = ManagedBuild(self.root, source or self.source, purpose=purpose)
        manager.slot.mkdir()
        (manager.slot / "cargo-target").mkdir()
        (manager.slot / "cargo-target" / "cache.o").write_bytes(b"rebuildable")
        row = {"schema": "rovezero.managed-build.v1", "source_id": manager.source_id,
               "source": str(manager.source), "token": "a" * 32, "pid": 123,
               "state": "complete", "finished_ns": 1, "purpose": purpose}
        write_record(manager.slot / "record.json", row)
        write_record(manager.slot / "last-run.json", {
            "schema": "rovezero.managed-build.v1", "source_id": manager.source_id,
            "tree_cleanup_verified": True,
        })
        return manager

    def test_completed_other_purpose_is_cleaned_after_original_finish(self):
        result = maintenance.cleanup_completed_builds_idle(self.current)
        self.assertEqual(result["removed_slots"], 1)
        self.assertGreater(result["removed_bytes"], 0)
        self.assertFalse(result["output_total_changed"])
        self.assertTrue(self.candidate.slot.exists())
        self.assertTrue((self.candidate.slot / "record.json").exists())
        self.assertTrue((self.candidate.slot / "last-run.json").exists())
        self.assertFalse((self.candidate.slot / "cargo-target").exists())
        self.assertTrue(self.current.slot.exists())

    def test_automatic_warning_maintenance_defers_build_cache_to_idle(self):
        result = maintenance.cleanup_completed_builds(self.current)
        self.assertEqual(result["build_cache_maintenance"], "deferred_to_idle")
        self.assertTrue((self.candidate.slot / "cargo-target").exists())

    def test_output_from_same_source_protects_cache_even_if_inactive(self):
        row = {"source": str(self.source), "state": "interrupted_retained",
               "physical_completion": "unknown"}
        with patch.object(managed_outputs, "_catalog_rows", return_value=[(None, row, self.catalog / "data")]):
            result = maintenance.cleanup_completed_builds_idle(self.current)
        self.assertEqual(result["removed_slots"], 0)
        self.assertTrue(self.candidate.slot.exists())

    def test_bound_output_protects_its_slot_but_not_other_completed_slot(self):
        row = {"source": str(self.source), "outer_supervisor": {
            "root": str(self.root), "owner": {"schema": "rovezero.managed-build.v1", "token": "a" * 32,
                                             "pid": 123, "source": str(self.source), "purpose": "current",
                                             "source_id": self.current.source_id}, "scope": "test"}}
        with patch.object(managed_outputs, "_catalog_rows", return_value=[(None, row, self.catalog / "data")]):
            result = maintenance.cleanup_completed_builds_idle(self.current)
        self.assertEqual(result["removed_slots"], 1)
        self.assertFalse((self.candidate.slot / "cargo-target").exists())
        self.assertTrue((self.current.slot / "cargo-target").exists())

    def test_direct_output_uncertain_lifetime_is_never_maintained(self):
        lease = Mock()
        lease.guard.stream = None
        lease.record = {"state": "retained", "tree_gone": False}
        result = maintenance.cleanup_completed_builds_for_output(lease)
        self.assertEqual(result["status"], "unavailable")
        self.assertTrue(self.candidate.slot.exists())

    def test_other_checkout_and_active_slot_are_not_removed(self):
        other = Path(self.temp.name) / "other"
        other.mkdir()
        external = self._slot("outside", other)
        row_path = self.candidate.slot / "record.json"
        import json
        row = json.loads(row_path.read_text())
        row["state"] = "active"
        write_record(row_path, row)
        result = maintenance.cleanup_completed_builds_idle(self.current)
        self.assertEqual(result["removed_slots"], 0)
        self.assertTrue(external.slot.exists())
        self.assertTrue(self.candidate.slot.exists())

    def test_busy_or_unknown_process_census_warns_without_deleting(self):
        with patch.object(storage_code_retirement, "_process_census", side_effect=OSError("no proc access")):
            result = maintenance.cleanup_completed_builds_idle(self.current)
        self.assertEqual(result["removed_slots"], 0)
        self.assertTrue(result["warnings"])
        self.assertTrue(self.candidate.slot.exists())

    def test_cache_deletion_checks_deadline_and_preserves_unvisited_files(self):
        target = self.candidate.slot / "cargo-target"
        (target / "second.o").write_bytes(b"second")
        snapshot, _ = maintenance._snapshot(target, lambda: None)
        calls = 0

        def deadline():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise maintenance.StorageError("deadline")

        with self.assertRaisesRegex(maintenance.StorageError, "deadline") as caught:
            maintenance._remove_snapshot(target, snapshot, deadline)
        self.assertGreater(caught.exception.regenerable_removed_bytes, 0)
        self.assertTrue(target.is_dir())
        self.assertEqual(len(list(target.iterdir())), 1)

    def test_unfinished_original_or_new_owner_blocks_cleanup(self):
        self.current.finished = False
        self.assertEqual(maintenance.cleanup_completed_builds_idle(self.current)["status"], "unavailable")
        self.current.finished = True
        (self.root / "owner.json").write_text("{}")
        self.assertEqual(maintenance.cleanup_completed_builds_idle(self.current)["status"], "unavailable")
        self.assertTrue(self.candidate.slot.exists())


class CompletedCommandMaintenanceTests(unittest.TestCase):
    def _run(self, *, total, gone=True, finish_error=None, cleanup_error=None):
        manager = Mock()
        manager.source = Path("/source")
        manager.finished = False
        events = []

        def finish(**kwargs):
            events.append("finish")
            if finish_error is not None:
                raise finish_error
            if kwargs["tree_gone"] is not True:
                raise maintenance.StorageError("physical child cleanup unknown")
            manager.finished = True
            return {"tree_cleanup_verified": True}

        def cleanup(owner, *, seconds):
            self.assertTrue(owner.finished)
            self.assertEqual(seconds, 5)
            events.append("maintenance")
            if cleanup_error:
                raise cleanup_error
            return {"status": "completed"}

        manager.finish.side_effect = finish
        with patch.object(sys, "argv", ["run_managed.py", "--", "program"]), \
             patch.object(run_managed.shutil, "which", return_value="/bin/program"), \
             patch.object(run_managed, "canonical_root", return_value=Path("/catalog")), \
             patch.object(run_managed, "ManagedBuild", return_value=manager), \
             patch.object(run_managed, "sample_output_budget", return_value=total), \
             patch.object(run_managed, "run_group", return_value=(7, gone, None)), \
             patch.object(maintenance, "cleanup_completed_builds", side_effect=cleanup) as called:
            code = run_managed.main()
        return code, events, called.call_count

    def test_above_threshold_after_finish_once(self):
        code, events, count = self._run(total=run_managed.DEFAULT_OUTPUT_WARNING_BYTES + 1)
        self.assertEqual((code, events, count), (7, ["finish", "maintenance"], 1))

    def test_at_threshold_no_maintenance(self):
        code, events, count = self._run(total=run_managed.DEFAULT_OUTPUT_WARNING_BYTES)
        self.assertEqual((code, events, count), (7, ["finish"], 0))

    def test_unknown_cleanup_or_finish_failure_never_maintains(self):
        for kwargs in ({"gone": False}, {"finish_error": maintenance.StorageError("finish failed")}):
            with self.subTest(kwargs=kwargs):
                code, events, count = self._run(total=run_managed.DEFAULT_OUTPUT_WARNING_BYTES + 1, **kwargs)
                self.assertEqual((code, events, count), (125, ["finish"], 0))

    def test_maintenance_exception_preserves_command_exit(self):
        code, events, count = self._run(total=run_managed.DEFAULT_OUTPUT_WARNING_BYTES + 1,
                                         cleanup_error=RuntimeError("cleanup failed"))
        self.assertEqual((code, events, count), (7, ["finish", "maintenance"], 1))


class DirectOutputMaintenanceTests(unittest.TestCase):
    def _finish(self, *, due, tree_gone=True, outer=None, cleanup_error=None):
        lease = managed_outputs.OutputLease.__new__(managed_outputs.OutputLease)
        lease._maintenance_due = due
        lease.record = {"state": "active"}
        lease.outer = outer
        lease.deferred = False
        lease._created_directory = None
        lease.check_budget = Mock()
        lease._write = Mock()
        lease._close_preservation_journal = Mock()
        lease._close_created_directory = Mock()
        lease._close_quarantine_view = Mock()
        lease.guard = Mock()
        lease.guard.stream = object()
        events = []

        def release(_error):
            lease.guard.stream = None
            events.append("release")

        def cleanup(owner, *, seconds):
            self.assertIsNone(owner.guard.stream)
            self.assertEqual(seconds, 5)
            events.append("maintenance")
            if cleanup_error:
                raise cleanup_error
            return {"status": "protected_or_empty"}

        lease._release_guard = release
        with patch.object(maintenance, "cleanup_completed_builds_for_output", side_effect=cleanup) as called:
            lease.finish(status="completed", tree_gone=tree_gone)
        return events, called.call_count, lease.record

    def test_warning_triggers_once_only_after_guard_release(self):
        events, count, record = self._finish(due=True)
        self.assertEqual((events, count), (["release", "maintenance"], 1))
        self.assertEqual(record["physical_completion"], "not_inferred_from_output_lifecycle")

    def test_no_threshold_unknown_tree_or_outer_lease_never_triggers(self):
        for options in ({"due": False}, {"due": True, "tree_gone": False},
                        {"due": True, "outer": {"owner": {}}}):
            with self.subTest(options=options):
                events, count, _ = self._finish(**options)
                self.assertEqual((events, count), (["release"], 0))

    def test_maintenance_failure_preserves_output_status(self):
        events, count, record = self._finish(due=True, cleanup_error=RuntimeError("unavailable"))
        self.assertEqual((events, count), (["release", "maintenance"], 1))
        self.assertEqual(record["status"], "completed")


if __name__ == "__main__":
    unittest.main()
