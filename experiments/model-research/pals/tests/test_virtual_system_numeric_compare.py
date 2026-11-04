"""Focused pure-reader checks; no model, optimizer, ORT or GPU is started."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

from rz_pals_model import numeric_capture as nc
from rz_pals_model import virtual_system_numeric_compare as comparator


class SystemComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).absolute()
        self.ref_dir = self.root / "ref"
        self.native_dir = self.root / "native"
        self.ref_dir.mkdir()
        self.native_dir.mkdir()
        self.candidate = {"from": 8, "to": 16, "promotion": 0}
        self.payload = {"role": "proposer", "candidates": [self.candidate],
                        "divergences": [], "scalar": 0.0}
        self.typed = {"model_input_kind": "virtual_review", "input": self.payload}
        self.bindings = {"source_components": {"model_reference": "a" * 64,
                                                "model_native": "b" * 64}}
        self.plan = {"schema_version": 6, "domain": "rz-pals-numerical-case-plan-v6/1",
                     "bindings": self.bindings, "invocation": "fresh", "batch": 1,
                     "groups": [{"id": "group", "role": "proposer", "line_bucket": 8,
                                 "cases": [{"id": "case", "input": self.typed}]}]}
        self.tensors = self.tensor_for(self.payload)
        digest = nc.tensor_digest(self.tensors, "proposer")
        case = {"id": "case", "input": self.typed, "input_tensors": self.tensors,
                "input_tensor_sha256": digest, "candidate_order": [self.candidate],
                "raw_output": {"candidate_logits": [0.0], "wdl_logits": [0.0, 0.0, 0.0],
                               "private_latent": [0.0] * (16 * 384),
                               "divergence_logits": None, "task_logits": None},
                "physical_completion_id": None}
        common = {"schema_version": 6, "domain": comparator.RAW_DOMAIN,
                  "bindings": self.bindings, "invocation": "fresh", "batch": 1,
                  "public_cache_hits": 0, "cases": [case]}
        self.reference = {**copy.deepcopy(common), "run_id": "reference.group",
                          "producer": "independent_pytorch_fp32",
                          "producer_source_component": "a" * 64, "observed_provider": None,
                          "completed_nn_inputs": 0, "physical_completion_known": False}
        self.native = {**copy.deepcopy(common), "run_id": "native-group", "producer": "rust_ort_native",
                       "producer_source_component": "b" * 64, "observed_provider": "CUDAExecutionProvider",
                       "completed_nn_inputs": 2, "physical_completion_known": True}
        self.native["cases"][0]["physical_completion_id"] = "completion-1"

    @staticmethod
    def tensor_for(payload):
        return [{"name": "scalar", "dtype": "f32_le", "shape": [1],
                 "bytes": list(struct.pack("<f", payload["scalar"]))}]

    def execute(self):
        plan_bytes = json.dumps(self.plan).encode("utf-8")
        plan_path = self.root / "plan.json"
        plan_path.write_bytes(plan_bytes)
        reference_bytes = json.dumps(self.reference).encode("utf-8")
        native_bytes = json.dumps(self.native).encode("utf-8")
        (self.ref_dir / "reference.group.json").write_bytes(reference_bytes)
        (self.native_dir / "native-group.json").write_bytes(native_bytes)
        # This tests the comparator boundary with a tiny isolated typed input;
        # the full V3 plan validator and encoder have their own independent checks.
        with (mock.patch.object(nc, "validate_system_case_plan", return_value=1),
              mock.patch.object(nc, "python_input", side_effect=lambda value: value),
              mock.patch.object(nc, "captured_tensors", side_effect=self.tensor_for)):
            return comparator.compare(plan_path, hashlib.sha256(plan_bytes).hexdigest(),
                                      self.ref_dir, self.native_dir)

    def test_success_reports_declared_plan_and_no_gpu_acceptance(self):
        result = self.execute()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["coverage_scope"], "declared_plan_only")
        self.assertEqual(result["kernel_placement"], "unknown")
        self.assertFalse(result["training_eligible"])

    def test_actual_tensor_byte_type_cannot_be_camouflaged(self):
        self.native["cases"][0]["input_tensors"][0]["bytes"][0] = 0.0
        with self.assertRaises(ValueError):
            self.execute()

    def test_actual_typed_input_signed_zero_is_reencoded(self):
        self.native["cases"][0]["input"]["input"]["scalar"] = -0.0
        with self.assertRaises(ValueError):
            self.execute()

    def test_raw_tolerance_exceeded_is_not_a_success(self):
        self.native["cases"][0]["raw_output"]["private_latent"][0] = 0.01
        with self.assertRaises(ValueError):
            self.execute()

    def test_each_producer_has_its_own_total_limit(self):
        # Each raw file fits; the sum of the independent producer artifacts
        # exceeds this limit. The reader must still accept the declared pair.
        raw_size = max(len(json.dumps(self.reference).encode()), len(json.dumps(self.native).encode()))
        with mock.patch.object(nc, "MAX_TOTAL_BYTES", raw_size + 32):
            result = self.execute()
        self.assertEqual(result["cases"], 1)
        self.assertGreater(result["reference_bytes_read"] + result["native_bytes_read"], raw_size + 32)


if __name__ == "__main__":
    unittest.main()
