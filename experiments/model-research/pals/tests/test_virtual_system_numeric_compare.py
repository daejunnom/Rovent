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

    def execute(self, **kwargs):
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
                                      self.ref_dir, self.native_dir, **kwargs)

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

    def test_invalid_optional_control_preserves_numerical_result(self):
        result = self.execute(control_inventory=self.root / "missing.json")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["control_evidence_status"], "failed")
        self.assertEqual(result["control_evidence"]["error_type"], "ValueError")
        self.assertEqual(result["kernel_placement"], "unknown")

    def test_cli_publishes_numerical_report_before_partial_exit(self):
        output = self.root / "comparison.json"
        partial = {"status": "passed", "control_evidence_status": "failed"}
        argv = ["compare", "--plan", str(self.root / "plan.json"), "--plan-sha256", "ignored",
                "--reference-dir", str(self.ref_dir), "--native-dir", str(self.native_dir),
                "--output", str(output)]
        with (mock.patch("sys.argv", argv), mock.patch.object(comparator, "compare", return_value=partial),
              mock.patch.object(nc, "_publish", return_value={"sha256": "observed", "bytes": 7}) as publish,
              mock.patch("builtins.print")):
            self.assertEqual(comparator.main(), 2)
        publish.assert_called_once_with(partial, output, 0)


class OptionalSystemControlEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).absolute()
        self.plan_sha = self.digest(b"plan")
        self.manifest_sha = self.digest(b"manifest")
        self.ort_sha = self.digest(b"ort")
        self.graphs = {"public": self.digest(b"public"), "shared_pc": self.digest(b"shared")}
        self.bindings = {"export_manifest_sha256": self.manifest_sha,
                         "runtime_library_sha256": self.ort_sha, "graphs": self.graphs,
                         "system_cuda_libraries": [{"role": "ort-core", "required_mapping": True},
                                                   {"role": "optional", "required_mapping": False}]}
        self.inventory = {"schema": "rovezero.pals-virtual-static-control-inventory.v1",
                          "actual_provider_placement": "not_observed", "cpu_allowlist": "not_created",
                          "manifest_sha256": self.manifest_sha, "layout": "virtual_review_shared_pc_if_v1",
                          "artifact_kind": "fresh",
                          "graphs": [{"role": role, "sha256": digest} for role, digest in self.graphs.items()]}
        placements = [{"role": role, "graph_sha256": list(bytes.fromhex(digest)),
                       "log_sha256": self.digest_array(b"log"), "assigned_nodes": 2,
                       "cuda_nodes": 1, "approved_cpu_control_nodes": 1,
                       "optimization": "disable", "approved_transfers": []}
                      for role, digest in self.graphs.items()]
        kernels = {role: {"profile_sha256": self.digest_array(role.encode()), "cuda_kernels": 2,
                          "neural_kernels": 1, "proposer_private_kernels": 1,
                          "critic_private_kernels": 1} for role in self.graphs}
        self.inventory_path, self.inventory_sha = self.write("inventory.json", self.inventory)
        self.witness = {"schema": "rovezero.pals-virtual-system-cuda-placement.v1",
                        "inventory_sha256": list(bytes.fromhex(self.inventory_sha)),
                        "manifest_sha256": list(bytes.fromhex(self.manifest_sha)),
                        "ort_sha256": list(bytes.fromhex(self.ort_sha)),
                        "system_runtime_identity_sha256": self.digest_array(b"system"),
                        "declared_device_id": 0, "mapped_roles": ["ort-core"],
                        "missing_optional_roles": ["optional"], "layout": self.inventory["layout"],
                        "artifact_kind": "fresh", "optimization": "disable", "initialization": placements,
                        "kernels": kernels, "physical_completion": "not_inferred"}

    @staticmethod
    def digest(value):
        return hashlib.sha256(value).hexdigest()

    @staticmethod
    def digest_array(value):
        return list(hashlib.sha256(value).digest())

    def write(self, name, value):
        path = self.root / name
        raw = json.dumps(value, sort_keys=True).encode("utf-8")
        path.write_bytes(raw)
        return path, self.digest(raw)

    def read(self, status="verified", *, witness=None, observation_updates=None):
        witness_path = None
        witness_sha = None
        if status == "verified":
            witness_path, witness_sha = self.write("witness.json", self.witness if witness is None else witness)
        observation = {"schema": "rovezero.virtual-system-cuda-placement-observation.v1",
                       "plan_sha256": self.plan_sha, "inventory_sha256": self.inventory_sha,
                       "manifest_sha256": self.manifest_sha, "final_status": status,
                       "witness_sha256": witness_sha, "profile_scope": "registered_cases",
                       "first_registered_role_coverage_status": "verified"}
        if observation_updates:
            observation.update(observation_updates)
        path, sha = self.write("observation.json", observation)
        return comparator._control_evidence(self.plan_sha, self.bindings, self.inventory_path,
                                            self.inventory_sha, path, sha, witness_path)

    def test_verified_evidence_is_only_a_cross_artifact_observation(self):
        result = self.read()
        self.assertEqual(result["producer_status"], "verified")
        self.assertEqual(result["reader_verification"], "cross_artifact_fields_only")
        self.assertEqual(result["graph_roles"], ["public", "shared_pc"])
        self.assertFalse(result["profiles_replayed"])
        self.assertEqual(result["physical_completion"], "not_attested")
        self.assertEqual(result["process_exit"], "not_attested")
        self.assertEqual(result["current_mapping"], "not_attested")
        self.assertEqual(result["same_numeric_run"], "not_attested")
        self.assertEqual(result["system_runtime_identity_scope"], "producer_reported_not_recomputed")

    def test_unavailable_mapping_stays_unknown_without_a_witness(self):
        result = self.read("mapping_unavailable", observation_updates={
            "initial_partial_observation": "write_failed", "initial_partial_error": "write error"})
        self.assertEqual(result["producer_status"], "mapping_unavailable")
        self.assertNotIn("witness_sha256", result)
        self.assertEqual(result["current_mapping"], "not_attested")
        self.assertEqual(result["initial_partial_observation"], "write_failed")
        self.assertEqual(result["initial_partial_error"], "write error")

    def test_mismatched_plan_and_graph_are_rejected(self):
        with self.assertRaises(ValueError):
            self.read(observation_updates={"plan_sha256": self.digest(b"other")})
        witness = copy.deepcopy(self.witness)
        witness["initialization"][0]["graph_sha256"] = self.digest_array(b"other graph")
        with self.assertRaises(ValueError):
            self.read(witness=witness)

    def test_required_mapping_and_private_branch_cannot_be_fabricated(self):
        witness = copy.deepcopy(self.witness)
        witness["mapped_roles"] = []
        witness["missing_optional_roles"] = ["ort-core", "optional"]
        with self.assertRaises(ValueError):
            self.read(witness=witness)
        witness = copy.deepcopy(self.witness)
        witness["kernels"]["shared_pc"]["critic_private_kernels"] = 0
        with self.assertRaises(ValueError):
            self.read(witness=witness)


if __name__ == "__main__":
    unittest.main()
