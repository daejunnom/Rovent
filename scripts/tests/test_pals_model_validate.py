"""CPU validation dependencies only; no installer, engine, model or owner runs."""
from pathlib import Path
import ast
import subprocess
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pals_model_validate import ValidationStages, ValidationStagesFailed
from managed_storage import StorageError


class ValidationStageTests(unittest.TestCase):
    def setUp(self):
        self.receipt = {"stages": {}}
        self.calls = []
        self.stages = ValidationStages(self.receipt, lambda: None)

    def operation(self, name, error=None):
        def perform():
            self.calls.append(name)
            if error is not None:
                raise error
        return perform

    def failed_gate(self, name):
        return self.stages.perform(name, self.operation(name, subprocess.CalledProcessError(1, [name])))

    def test_returned_unit_failure_keeps_independent_neural_branch(self):
        self.assertFalse(self.failed_gate("unit_suite"))
        self.assertTrue(self.stages.perform("initialize", self.operation("initialize")))
        self.assertTrue(self.stages.perform("export", self.operation("export"), requires=("initialize",)))
        self.assertTrue(self.stages.perform("numeric", self.operation("numeric"), requires=("export",)))
        self.assertEqual(self.calls, ["unit_suite", "initialize", "export", "numeric"])
        self.assertEqual(self.stages.results["unit_suite"]["returncode"], 1)
        with self.assertRaises(ValidationStagesFailed):
            self.stages.require_complete()

    def test_packing_and_private_verifier_are_independent_of_export(self):
        self.failed_gate("packing_rust")
        self.failed_gate("private_verifier")
        self.assertTrue(self.stages.perform("export", self.operation("export")))
        self.assertIsNone(self.stages.shared_failure)
        self.assertEqual(self.stages.results["packing_rust"]["failure_scope"], "stage")

    def test_real_failed_prerequisite_blocks_only_its_consumers(self):
        self.failed_gate("initialize")
        self.assertFalse(self.stages.perform("export", self.operation("export"), requires=("initialize",)))
        self.assertFalse(self.stages.perform("numeric", self.operation("numeric"), requires=("export",)))
        self.assertTrue(self.stages.perform("weight_free_packing", self.operation("packing")))
        self.assertEqual(self.calls, ["initialize", "packing"])
        self.assertEqual(self.stages.results["export"]["blocked_by"], ["initialize"])
        self.assertFalse(self.stages.results["export"]["executed"])

    def test_numeric_failure_does_not_publish_passing_fixture_or_hide_failure(self):
        self.failed_gate("numeric")
        self.assertFalse(self.stages.perform("rust_fixtures", self.operation("fixtures"), requires=("numeric",)))
        self.assertTrue(self.stages.perform("runtime_asset", self.operation("runtime")))
        self.assertNotIn("numeric", self.receipt["stages"])
        self.assertNotIn("rust_fixtures", self.receipt["stages"])
        self.assertIn("runtime_asset", self.receipt["stages"])
        with self.assertRaises(ValidationStagesFailed):
            self.stages.require_complete()

    def test_timeout_preserves_unknown_tree_and_stops_same_shared_scope(self):
        timeout = subprocess.TimeoutExpired(["child"], 1)
        self.assertFalse(self.stages.perform("child", self.operation("child", timeout)))
        self.assertFalse(self.stages.perform("independent", self.operation("later")))
        self.assertEqual(self.calls, ["child"])
        self.assertEqual(self.stages.results["independent"]["failure_scope"], "shared_managed_execution")
        self.assertEqual(self.stages.results["child"]["error_type"], "TimeoutExpired")

    def test_negative_returncode_blocks_same_scope_without_guessing_oom(self):
        self.assertFalse(self.stages.perform("child", self.operation(
            "child", subprocess.CalledProcessError(-9, ["child"]))))
        self.assertFalse(self.stages.perform("independent", self.operation("must_not_run")))
        self.assertEqual(self.calls, ["child"])
        self.assertEqual(self.stages.results["child"]["returncode"], -9)
        self.assertEqual(self.stages.results["child"]["termination_observation"],
                         "negative_returncode_no_cause_inferred")
        self.assertEqual(self.stages.results["independent"]["failure_scope"], "shared_managed_execution")

    def test_shared_io_and_storage_failures_are_not_numerical_mismatches(self):
        for error in (OSError("shared file unavailable"), StorageError("owner/quota failure")):
            with self.subTest(error=type(error).__name__):
                stages = ValidationStages({"stages": {}}, lambda: None)
                self.assertFalse(stages.perform("first", self.operation("first", error)))
                self.assertFalse(stages.perform("second", self.operation("must_not_run")))
                self.assertEqual(stages.results["first"]["failure_scope"], "shared_managed_execution")

    def test_changed_registered_input_never_dispatches_a_command(self):
        def changed():
            raise OSError("registered source digest differs")
        stages = ValidationStages({"stages": {}}, changed)
        self.assertFalse(stages.perform("export", self.operation("must_not_run")))
        self.assertFalse(stages.perform("numeric", self.operation("must_not_run")))
        self.assertEqual(self.calls, [])
        self.assertFalse(stages.results["export"]["executed"])
        self.assertEqual(stages.results["export"]["failure_scope"], "registered_inputs_or_shared_storage")

    def test_local_artifact_rejection_does_not_change_other_branch(self):
        self.assertFalse(self.stages.perform("packing", self.operation("packing", ValueError("bad generated graph"))))
        self.assertFalse(self.stages.perform("packing_reader", self.operation("reader"), requires=("packing",)))
        self.assertTrue(self.stages.perform("initialize", self.operation("initialize")))
        self.assertEqual(self.calls, ["packing", "initialize"])

    def test_duplicate_and_unknown_dependencies_are_rejected(self):
        self.assertTrue(self.stages.perform("first", self.operation("first")))
        with self.assertRaises(ValueError):
            self.stages.perform("first", self.operation("duplicate"))
        with self.assertRaises(ValueError):
            self.stages.perform("second", self.operation("unknown"), requires=("not_registered",))
        self.assertEqual(self.calls, ["first"])

    def test_cancel_is_recorded_and_propagates(self):
        with self.assertRaises(KeyboardInterrupt):
            self.stages.perform("first", self.operation("first", KeyboardInterrupt()))
        self.assertEqual(self.stages.results["first"]["status"], "failed")
        self.assertEqual(self.stages.results["first"]["error_type"], "KeyboardInterrupt")
        self.assertFalse(self.stages.perform("second", self.operation("must_not_run")))

    def test_complete_requires_all_requested_stages_pass(self):
        self.assertTrue(self.stages.perform("first", self.operation("first")))
        self.assertTrue(self.stages.perform("second", self.operation("second"), requires=("first",)))
        self.stages.require_complete()
        self.assertEqual(set(self.receipt["stages"]), {"first", "second"})


class ActualCpuPlanTests(unittest.TestCase):
    """Check critical command dependencies without dispatching the real plan."""
    def setUp(self):
        self.tree = ast.parse((Path(__file__).resolve().parents[1] / "pals_model_validate.py").read_text(encoding="utf-8"))
        self.main = next(node for node in self.tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        self.dependencies = {}
        for node in ast.walk(self.main):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "controller"
                    and node.func.attr == "perform"):
                name = ast.literal_eval(node.args[0])
                required = next((ast.literal_eval(kw.value) for kw in node.keywords if kw.arg == "requires"), ())
                self.dependencies[name] = required

    def ancestors(self, stage):
        result = set()
        for prerequisite in self.dependencies[stage]:
            result.add(prerequisite)
            result.update(self.ancestors(prerequisite))
        return result

    def test_actual_neural_branch_does_not_require_unrelated_gates(self):
        unrelated = {"model_tests_seconds", "device_packing_rust_body_seconds", "private_verifier_seconds",
                     "preparation_check_seconds"}
        for stage in ("initialize_seconds", "export_seconds", "numeric_seconds"):
            with self.subTest(stage=stage):
                self.assertFalse(self.ancestors(stage) & unrelated)
        self.assertIn("initialize_seconds", self.ancestors("numeric_seconds"))
        self.assertIn("numeric_seconds", self.ancestors("rust_fixtures_seconds"))

    def test_actual_weight_free_packing_does_not_require_model_initialization(self):
        prerequisites = self.ancestors("device_packing_rust_body_seconds")
        self.assertIn("reference_install_seconds", prerequisites)
        self.assertNotIn("torch_install_seconds", prerequisites)
        self.assertNotIn("initialize_seconds", prerequisites)

    def test_owner_acquisition_is_covered_by_original_finally(self):
        def calls(node, method):
            return [child for child in ast.walk(node) if isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute) and child.func.attr == method
                    and isinstance(child.func.value, ast.Name) and child.func.value.id == "output_lease"]
        acquisitions = calls(self.main, "acquire")
        self.assertEqual(len(acquisitions), 1)
        protected = [node for node in ast.walk(self.main) if isinstance(node, ast.Try)
                     and acquisitions[0] in [child for item in node.body for child in ast.walk(item)]
                     and any(calls(item, "finish") for item in node.finalbody)]
        self.assertEqual(len(protected), 1)


if __name__ == "__main__":
    unittest.main()
