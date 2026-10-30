"""CPU-only generated artifact reclamation preserves all result evidence."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import managed_output_regenerable as reclaim
import managed_outputs
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

    def test_cpu_graph_cleanup_does_not_take_global_catalog_lock(self):
        with managed_outputs.CatalogGuard(self.catalog / "catalog.lock"):
            self.assertEqual(self.cleanup_graph()["removed_files"], 1)
        self.assertFalse(self.graph.exists())

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
            with patch.object(storage_code_retirement, "_process_census", return_value={"complete": True}):
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
