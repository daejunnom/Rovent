"""CPU-only generated artifact reclamation preserves all result evidence."""
import hashlib
import json
import os
import select
import subprocess
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import managed_output_regenerable as reclaim
import managed_outputs
import rovezero_usage_snapshot
import storage_code_retirement


@unittest.skipUnless(sys.platform == "linux", "Linux no-follow/census path")
class CpuGraphReclamationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.source = base / "source"
        self.source.mkdir()
        self.catalog = base / "catalog"
        self.root = base / "build"
        self.root.mkdir()
        self.output = base / "output"
        self.owner = {"schema": "rovezero.managed-build.v1", "pid": 1234,
                      "source": str(self.source), "source_id": "a" * 24}
        self.manager = SimpleNamespace(root=self.root, finished=True, _identity=lambda: self.owner)
        self.lease = managed_outputs.OutputLease(self.output, self.source, catalog=self.catalog,
                                                reserve=1024 * 1024)
        self.lease.outer = {"root": str(self.root), "owner": self.owner,
                            "scope": "inherited_outer_process_group_or_job"}
        self.lease.acquire()
        self.packing = self.output / "device-packing"
        self.packing.mkdir()
        self.graph = self.packing / reclaim.RELATIVE.name
        self.graph.write_bytes(b"recreatable-onnx")
        (self.packing / "device-packing.json").write_text("retained evidence")
        self.pin = {"bytes": self.graph.stat().st_size,
                    "sha256": hashlib.sha256(self.graph.read_bytes()).hexdigest()}
        self.lease.register_cpu_static_packing_artifact(self.pin)
        self.receipt = {"schema": "rovezero.pals-cpu-model-validation.v1", "status": "success",
                        "provider": "cpu", "gpu_validation": "not_run",
                        "device_packing_static_interop": {
                            "scope": "python_artifact_rust_fixed_body_check_only",
                            "rust_body_exit_status": 0,
                            "assets": {reclaim.RELATIVE.name: self.pin}}}
        (self.output / "validation.json").write_text(json.dumps(self.receipt))
        self.lease.finish(status="success", tree_gone="outer_managed_supervisor_only")
        managed_outputs.finalize_supervised_outputs(self.root, self.owner, self.catalog)
        self.census = patch.object(storage_code_retirement, "_process_census", return_value={"complete": True})
        self.census.start()
        self.addCleanup(self.census.stop)

    def cleanup_graph(self):
        return reclaim.cleanup_registered_cpu_graphs(self.manager, self.catalog)

    def test_completed_cpu_static_graph_only_is_removed(self):
        result = self.cleanup_graph()
        self.assertEqual(result["removed_files"], 1, result)
        self.assertEqual(result["removed_bytes"], self.pin["bytes"])
        self.assertTrue(result["output_total_changed"])
        self.assertFalse(self.graph.exists())
        self.assertTrue((self.packing / "device-packing.json").exists())
        self.assertTrue((self.output / "validation.json").exists())
        self.assertTrue((self.output / ".rovezero-output.json").exists())
        self.assertTrue((self.output / ".rovezero-cpu-static-packing-retirement.jsonl").exists())

    def test_gpu_or_failed_validation_never_deleted(self):
        self.receipt["gpu_validation"] = "unknown"
        (self.output / "validation.json").write_text(json.dumps(self.receipt))
        self.assertEqual(self.cleanup_graph()["removed_files"], 0)
        self.assertTrue(self.graph.exists())

    def test_replaced_or_open_graph_never_deleted(self):
        self.graph.write_bytes(b"different")
        self.assertEqual(self.cleanup_graph()["removed_files"], 0)
        self.assertTrue(self.graph.exists())

    def test_incomplete_census_preserves_graph(self):
        self.census.stop()
        with patch.object(storage_code_retirement, "_process_census", side_effect=OSError("unavailable")):
            self.assertEqual(self.cleanup_graph()["removed_files"], 0)
        self.assertTrue(self.graph.exists())

    def test_census_incomplete_result_preserves_graph(self):
        with patch.object(storage_code_retirement, "_process_census", return_value={"complete": False}):
            self.assertEqual(self.cleanup_graph()["removed_files"], 0)
        self.assertTrue(self.graph.exists())

    def test_census_includes_output_and_graph_parent_directories(self):
        with patch.object(storage_code_retirement, "_process_census", return_value={"complete": True}) as census:
            self.assertEqual(self.cleanup_graph()["removed_files"], 1)
        identities = census.call_args.args[0]
        for path in (self.output, self.packing):
            info = path.stat()
            self.assertIn((info.st_dev, info.st_ino), identities)

    def test_busy_catalog_protects_graph_and_reports_maintenance_warning(self):
        with managed_outputs.CatalogGuard(self.catalog / "catalog.lock"):
            result = self.cleanup_graph()
            self.assertEqual(result["removed_files"], 0)
            self.assertTrue(result["warnings"])
        self.assertTrue(self.graph.exists())

    def test_active_source_protects_only_overlapping_graph(self):
        base = self.output.parent
        active = managed_outputs.OutputLease(base / "active", self.output, catalog=self.catalog,
                                             reserve=1024 * 1024)
        active.acquire()
        active.prepare_live_handoff()
        active.suspend_live()
        try:
            result = self.cleanup_graph()
            self.assertEqual(result["removed_files"], 0, result)
            self.assertGreaterEqual(result["protected"], 1)
            self.assertTrue(self.graph.exists())
        finally:
            active.resume_live()
            active.finish(status="fixture_done", tree_gone=True)

    def test_live_source_on_different_filesystem_without_worker_mount_proof_is_protected(self):
        if not Path("/dev/shm").is_dir() or not os.access("/dev/shm", os.W_OK):
            self.skipTest("independent tmpfs unavailable")
        with tempfile.TemporaryDirectory(dir="/dev/shm") as source:
            if Path(source).stat().st_dev == self.graph.stat().st_dev:
                self.skipTest("independent filesystem unavailable")
            other = managed_outputs.OutputLease(self.output.parent / "other-active", Path(source),
                                                catalog=self.catalog, reserve=1024 * 1024)
            other.acquire()
            other.prepare_live_handoff()
            other.suspend_live()
            try:
                result = self.cleanup_graph()
                # The source worker could use a distinct mount namespace; an
                # independent device in this process alone is not proof.
                self.assertEqual(result["removed_files"], 0, result)
                self.assertGreaterEqual(result["protected"], 1)
                self.assertTrue(self.graph.exists())
            finally:
                other.resume_live()
                other.finish(status="fixture_done", tree_gone=True)

    def test_live_source_on_same_device_is_conservatively_protected(self):
        other = managed_outputs.OutputLease(self.output.parent / "same-device-active", self.source,
                                            catalog=self.catalog, reserve=1024 * 1024)
        other.acquire()
        other.prepare_live_handoff()
        other.suspend_live()
        try:
            result = self.cleanup_graph()
            self.assertEqual(result["removed_files"], 0, result)
            self.assertGreaterEqual(result["protected"], 1)
            self.assertTrue(self.graph.exists())
        finally:
            other.resume_live()
            other.finish(status="fixture_done", tree_gone=True)

    def test_foreign_active_source_without_original_inode_retains_candidate(self):
        # The original anchor lives in a different process and is not present
        # in this process's private live-handoff registry.
        script = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import managed_outputs, rovezero_usage_snapshot
rovezero_usage_snapshot.claim_total_usage_maintenance = lambda: False
lease = managed_outputs.OutputLease(Path(sys.argv[2]), Path(sys.argv[3]),
    catalog=Path(sys.argv[4]), reserve=1024 * 1024)
lease.acquire()
lease.prepare_live_handoff()
lease.suspend_live()
print('ready', flush=True)
sys.stdin.readline()
lease.resume_live()
lease.finish(status='fixture_done', tree_gone=True)
"""
        worker = subprocess.Popen(
            [sys.executable, "-c", script, str(Path(managed_outputs.__file__).parent),
             str(self.output.parent / "foreign-active"), str(self.source), str(self.catalog)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            ready, _, _ = select.select([worker.stdout], [], [], 5)
            self.assertTrue(ready, "foreign lease did not report readiness")
            self.assertEqual(worker.stdout.readline().strip(), "ready")
            result = self.cleanup_graph()
            self.assertEqual(result["removed_files"], 0, result)
            self.assertGreaterEqual(result["protected"], 1)
            self.assertTrue(any("active_source_scope_unresolved" in item for item in result["warnings"]))
            self.assertTrue(self.graph.exists())
        finally:
            if worker.poll() is None and worker.stdin is not None:
                try:
                    worker.stdin.write("finish\n")
                    worker.stdin.flush()
                except BrokenPipeError:
                    pass
            try:
                worker.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.communicate(timeout=5)
                raise
        self.assertEqual(worker.returncode, 0)

    def test_new_source_registration_cannot_race_census_and_unlink(self):
        base = self.output.parent
        next_lease = managed_outputs.OutputLease(base / "racing-active", self.output,
                                                 catalog=self.catalog, reserve=1024 * 1024)
        contended = threading.Event()
        admitted = threading.Event()
        failures = []
        original_acquire = managed_outputs.CatalogGuard.acquire

        def worker():
            try:
                next_lease.acquire()
                admitted.set()
            except BaseException as error:
                failures.append(error)

        thread = threading.Thread(target=worker)

        def observed_acquire(guard):
            try:
                return original_acquire(guard)
            except managed_outputs.CatalogBusy:
                if threading.current_thread() is thread:
                    contended.set()
                raise

        def census(*_args):
            thread.start()
            # This event is set only after the actual OS lock attempt failed,
            # not merely before the worker requested the lock.
            self.assertTrue(contended.wait(0.3))
            self.assertFalse(admitted.is_set())
            return {"complete": True}

        with patch.object(managed_outputs.CatalogGuard, "acquire", observed_acquire), \
             patch.object(storage_code_retirement, "_process_census", side_effect=census):
            result = self.cleanup_graph()
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assertTrue(admitted.is_set())
        self.assertEqual(result["removed_files"], 1, result)
        next_lease.finish(status="fixture_done", tree_gone=True)

    def test_unlink_is_reported_even_if_directory_fsync_fails(self):
        original_fsync = os.fsync
        packing = self.packing.stat()

        def fail_packing_directory(fd):
            info = os.fstat(fd)
            if (info.st_dev, info.st_ino) == (packing.st_dev, packing.st_ino):
                raise OSError("injected directory durability failure")
            return original_fsync(fd)

        with patch.object(reclaim.os, "fsync", side_effect=fail_packing_directory):
            result = self.cleanup_graph()
        self.assertFalse(self.graph.exists())
        self.assertEqual(result["removed_files"], 1, result)
        self.assertEqual(result["removed_bytes"], self.pin["bytes"])
        self.assertTrue(result["output_total_changed"])
        self.assertEqual(result["directory_sync_unconfirmed"], 1)
        self.assertTrue(any("directory_fsync_unconfirmed" in warning
                            for warning in result["warnings"]))
        journal = self.output / ".rovezero-cpu-static-packing-retirement.jsonl"
        records = [json.loads(line) for line in journal.read_text().splitlines()]
        self.assertEqual([record["status"] for record in records],
                         ["unlink_pending", "unlinked_directory_sync_unconfirmed"])

    def test_started_retirement_finishes_even_if_deadline_elapses_after_journal(self):
        now = [100.0]
        original_append = reclaim._append_journal

        def advance_after_pending(fd, record):
            original_append(fd, record)
            if record["status"] == "unlink_pending":
                now[0] += 1.0

        with patch.object(reclaim.time, "monotonic", side_effect=lambda: now[0]), \
             patch.object(reclaim, "_append_journal", side_effect=advance_after_pending):
            result = self.cleanup_graph()
        self.assertFalse(self.graph.exists())
        self.assertEqual(result["removed_files"], 1, result)
        journal = self.output / ".rovezero-cpu-static-packing-retirement.jsonl"
        records = [json.loads(line) for line in journal.read_text().splitlines()]
        self.assertEqual([record["status"] for record in records], ["unlink_pending", "unlinked"])

    def test_journal_failure_after_unlink_does_not_count_as_protected(self):
        original_append = reclaim._append_journal

        def fail_completion(fd, record):
            if record["status"] == "unlinked":
                raise OSError("injected journal completion failure")
            return original_append(fd, record)

        with patch.object(reclaim, "_append_journal", side_effect=fail_completion):
            result = self.cleanup_graph()
        self.assertFalse(self.graph.exists())
        self.assertEqual(result["removed_files"], 1, result)
        self.assertEqual(result["removed_bytes"], self.pin["bytes"])
        self.assertEqual(result["removed_bytes_scope"], "logical_file_size")
        self.assertEqual(result["protected"], 0, result)
        self.assertTrue(any("journal completion failure" in item for item in result["warnings"]))

    def test_journal_close_error_does_not_hide_original_or_skip_other_fds(self):
        real_close = os.close
        output_stat = self.output.stat()
        parent_stat = self.packing.stat()
        closed = []

        def close_then_fail_journal(fd):
            info = os.fstat(fd)
            identity = (info.st_dev, info.st_ino)
            try:
                name = os.readlink(f"/proc/self/fd/{fd}")
            except OSError:
                name = ""
            real_close(fd)
            if identity == (parent_stat.st_dev, parent_stat.st_ino):
                closed.append("parent")
            if identity == (output_stat.st_dev, output_stat.st_ino):
                closed.append("root")
            if name.endswith("/.rovezero-cpu-static-packing-retirement.jsonl"):
                closed.append("journal")
                raise OSError("injected journal close error")

        def fail_pending_append(_fd, _record):
            raise OSError("original journal write error")

        with patch.object(reclaim, "_append_journal", side_effect=fail_pending_append), \
             patch.object(reclaim.os, "close", side_effect=close_then_fail_journal):
            result = self.cleanup_graph()
        self.assertTrue(self.graph.exists())
        self.assertEqual(result["removed_files"], 0, result)
        self.assertEqual(closed, ["journal", "parent", "root"])
        self.assertTrue(any("original journal write error" in item for item in result["warnings"]))
        self.assertTrue(any("journal_close_unconfirmed" in item for item in result["warnings"]))

    def test_pin_close_error_preserves_original_read_error_and_reports_both(self):
        real_read, real_close = os.read, os.close
        target = self.graph.stat()
        injected_close = []

        def is_graph(fd):
            info = os.fstat(fd)
            return (info.st_dev, info.st_ino) == (target.st_dev, target.st_ino)

        def fail_graph_read(fd, size):
            if is_graph(fd):
                raise OSError("original graph read failure")
            return real_read(fd, size)

        def close_then_fail_graph(fd):
            graph_fd = is_graph(fd)
            real_close(fd)
            if graph_fd:
                injected_close.append(fd)
                raise OSError("secondary graph close failure")

        with patch.object(reclaim.os, "read", side_effect=fail_graph_read), \
             patch.object(reclaim.os, "close", side_effect=close_then_fail_graph):
            result = self.cleanup_graph()
        self.assertTrue(self.graph.exists())
        self.assertEqual(result["removed_files"], 0, result)
        self.assertEqual(len(injected_close), 1)
        self.assertTrue(any("original graph read failure" in item for item in result["warnings"]))
        self.assertTrue(any("file_pin_close_unconfirmed" in item
                            and "secondary graph close failure" in item
                            for item in result["warnings"]))

    def test_catalog_release_error_preserves_completed_unlink_result(self):
        original_release = managed_outputs.CatalogGuard.release

        def release_then_fail(guard):
            original_release(guard)
            raise OSError("injected catalog release error")

        with patch.object(managed_outputs.CatalogGuard, "release", release_then_fail):
            result = self.cleanup_graph()
        self.assertFalse(self.graph.exists())
        self.assertEqual(result["removed_files"], 1, result)
        self.assertEqual(result["removed_bytes"], self.pin["bytes"])
        self.assertTrue(result["output_total_changed"])
        self.assertTrue(any("catalog_guard_release_unconfirmed" in item
                            for item in result["warnings"]))

    def test_unfinished_supervisor_preserves_graph(self):
        self.manager.finished = False
        self.assertEqual(self.cleanup_graph()["removed_files"], 0)
        self.assertTrue(self.graph.exists())


@unittest.skipUnless(sys.platform == "linux", "Linux no-follow/census path")
class DirectLeaseCpuGraphTests(unittest.TestCase):
    def test_direct_original_lease_cleans_graph_after_warning(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source, output, catalog = base / "source", base / "output", base / "catalog"
            source.mkdir()
            lease = managed_outputs.OutputLease(output, source, catalog=catalog,
                                               reserve=1024 * 1024, total_limit=1)
            lease.acquire()
            packing = output / "device-packing"
            packing.mkdir()
            graph = packing / reclaim.RELATIVE.name
            graph.write_bytes(b"cpu-graph")
            (packing / "device-packing.json").write_text("evidence")
            pin = {"bytes": graph.stat().st_size, "sha256": hashlib.sha256(graph.read_bytes()).hexdigest()}
            lease.register_cpu_static_packing_artifact(pin)
            receipt = {"schema": "rovezero.pals-cpu-model-validation.v1", "status": "success",
                       "provider": "cpu", "gpu_validation": "not_run",
                       "device_packing_static_interop": {
                           "scope": "python_artifact_rust_fixed_body_check_only", "rust_body_exit_status": 0,
                           "assets": {reclaim.RELATIVE.name: pin}}}
            (output / "validation.json").write_text(json.dumps(receipt))
            # Maintenance is driven by a separately claimed physical-storage
            # warning, not by this fixture's logical reservation threshold.
            # Keep that trigger deterministic without replacing the real
            # direct-lease, catalog guard, or graph retirement paths.
            with patch.object(rovezero_usage_snapshot, "claim_total_usage_maintenance", return_value=True), \
                 patch.object(storage_code_retirement, "_process_census", return_value={"complete": True}):
                lease.finish(status="success", tree_gone=True)
            self.assertFalse(graph.exists())
            self.assertTrue((packing / "device-packing.json").exists())
            self.assertTrue((output / "validation.json").exists())

    def test_direct_unknown_tree_does_not_cleanup(self):
        # No original lease can be promoted solely from a retained pathname.
        lease = SimpleNamespace(guard=SimpleNamespace(stream=None), record={"state": "active",
                                "tree_gone": False}, outer=None)
        result = reclaim.cleanup_registered_cpu_graphs(None, Path("/nonexistent"), own_lease=lease)
        self.assertEqual(result["removed_files"], 0)
