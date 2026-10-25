"""CPU validation dependencies only; no installer, engine, model or owner runs."""
from pathlib import Path
import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pals_model_validate as wrapper
from pals_model_validate import ValidationStages, ValidationStagesFailed, cpu_stage_dependencies
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

    def test_numeric_failure_preserves_independent_reference_outputs_and_failure(self):
        for name in cpu_stage_dependencies():
            operation = self.operation(name, subprocess.CalledProcessError(1, [name])
                                       if name == "numeric_seconds" else None)
            self.stages.perform(name, operation)
        self.assertFalse(self.stages.passed("numeric_seconds"))
        self.assertTrue(self.stages.passed("rust_fixtures_seconds"))
        self.assertTrue(self.stages.passed("runtime_preservation_seconds"))
        self.assertNotIn("numeric_seconds", self.receipt["stages"])
        self.assertEqual(self.stages.results["numeric_seconds"]["returncode"], 1)
        self.assertIn("rust_fixtures_seconds", self.calls)
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

    def test_missing_executable_keeps_independent_command_without_claiming_execution(self):
        command = ["/absent/cargo", "run"]
        for error in (FileNotFoundError(2, "missing executable", command[0]),
                      PermissionError(13, "executable denied", command[0])):
            with self.subTest(error=type(error).__name__):
                stages = ValidationStages({"stages": {}}, lambda: None)
                with mock.patch.object(wrapper.subprocess, "run", side_effect=error):
                    self.assertFalse(stages.perform("packing_reader", lambda: wrapper.run(
                        command, cwd="/registered/source", environment={})))
                self.assertTrue(stages.perform("neural", self.operation("neural")))
                result = stages.results["packing_reader"]
                self.assertFalse(result["command_exec_started"])
                self.assertEqual(result["failure_scope"], "stage")
                self.assertEqual(result["original_error_type"], type(error).__name__)

    def test_missing_cwd_is_not_reported_as_executable_start_failure(self):
        error = FileNotFoundError(2, "missing cwd", "/missing/source")
        for command in (["/registered/cargo"], ["/missing/source"]):
            with self.subTest(command=command):
                stages = ValidationStages({"stages": {}}, lambda: None)
                with mock.patch.object(wrapper.subprocess, "run", side_effect=error):
                    self.assertFalse(stages.perform("first", lambda: wrapper.run(
                        command, cwd="/missing/source", environment={})))
                self.assertFalse(stages.perform("second", self.operation("must_not_run")))
                self.assertEqual(stages.results["first"]["error_type"], "FileNotFoundError")
                self.assertEqual(stages.results["second"]["failure_scope"], "shared_managed_execution")

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
    """Read the dependency data used by production; do not inspect source shape."""
    def ancestors(self, stage):
        dependencies = cpu_stage_dependencies()
        result = set()
        for prerequisite in dependencies[stage]:
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
        self.assertIn("export_seconds", self.ancestors("rust_fixtures_seconds"))
        self.assertNotIn("numeric_seconds", self.ancestors("rust_fixtures_seconds"))

    def test_actual_weight_free_packing_does_not_require_model_initialization(self):
        prerequisites = self.ancestors("device_packing_rust_body_seconds")
        self.assertIn("reference_install_seconds", prerequisites)
        self.assertNotIn("torch_install_seconds", prerequisites)
        self.assertNotIn("initialize_seconds", prerequisites)

    def test_actual_plan_data_drives_failure_propagation(self):
        stages = ValidationStages({"stages": {}}, lambda: None)
        observed = []
        failures = {"model_tests_seconds", "device_packing_rust_body_seconds", "private_verifier_seconds"}
        for name in cpu_stage_dependencies():
            def operation(name=name):
                observed.append(name)
                if name in failures:
                    raise subprocess.CalledProcessError(1, [name])
            stages.perform(name, operation)
        self.assertTrue(stages.passed("export_seconds"))
        self.assertTrue(stages.passed("numeric_seconds"))
        self.assertTrue(stages.passed("rust_fixtures_seconds"))
        self.assertTrue(failures.issubset(observed))
        with self.assertRaises(ValidationStagesFailed):
            stages.require_complete()
        with self.assertRaises(TypeError):
            cpu_stage_dependencies()["export_seconds"] = ("model_tests_seconds",)


class OwnerBoundaryTests(unittest.TestCase):
    """Exercise main with fake external I/O; no source bytes or owner are read."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pals-validation-boundary-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / "output"
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()
        output = self.output
        class FakeLease:
            def __init__(self):
                self.record = None
                self.calls = []
                self.acquire_error = None
            def acquire(self):
                self.calls.append(("acquire",))
                output.mkdir()
                self.record = {"state": "active"}
                if self.acquire_error is not None:
                    raise self.acquire_error
                return self
            def check_budget(self):
                self.calls.append(("budget",))
            def finish(self, **values):
                self.calls.append(("finish", values))
                self.record = {"state": "retained"}
        self.lease = FakeLease()
        self.args = argparse.Namespace(output=self.output, seed=1, runtime_store=self.root / "runtime",
            source_commit="1" * 40, source_dirty=False, layout="separate_pc", rules_profile_json=None,
            dataset_run=None, dataset_receipt_sha256=None, dataset_encoder_source_sha256=None,
            verifier_cpu_binary=None, verifier_cpu_binary_sha256=None)

    def patched_boundaries(self, *, source_error=None):
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(wrapper.argparse.ArgumentParser, "parse_args", return_value=self.args))
        stack.enter_context(mock.patch.object(wrapper, "OutputLease", return_value=self.lease))
        stack.enter_context(mock.patch.object(wrapper, "checked", side_effect=lambda path: Path(path).absolute()))
        stack.enter_context(mock.patch.object(wrapper, "input_identity", return_value=(1, 2, 3, 1, 7, 8, 9)))
        stack.enter_context(mock.patch.object(Path, "glob", return_value=iter(())))
        stack.enter_context(mock.patch.object(Path, "read_bytes", return_value=b"synthetic fixture bytes", side_effect=source_error))
        stack.enter_context(mock.patch.object(wrapper, "run", side_effect=subprocess.CalledProcessError(1, ["fake venv"])))
        stack.enter_context(mock.patch.object(wrapper.subprocess, "run", side_effect=AssertionError("real subprocess forbidden")))
        stack.enter_context(mock.patch.object(wrapper.subprocess, "check_output", side_effect=AssertionError("real subprocess forbidden")))
        stack.enter_context(mock.patch.dict(os.environ, {"TMPDIR": str(self.scratch), "CARGO_TARGET_DIR": str(self.root / "target")}))
        return stack

    def test_returned_stage_failure_finishes_original_fake_lease_once(self):
        with self.patched_boundaries(), self.assertRaises(ValidationStagesFailed):
            wrapper.main()
        self.assertEqual(self.lease.calls, [("acquire",), ("finish", {
            "status": "failed", "tree_gone": "outer_managed_supervisor_only"})])
        receipt = json.loads((self.output / "validation.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["stage_results"]["venv_seconds"]["status"], "failed")
        self.assertEqual(receipt["stage_results"]["numeric_seconds"]["status"], "blocked")

    def test_partial_acquire_failure_still_attempts_original_finish(self):
        self.lease.acquire_error = RuntimeError("fake acquire failed after record")
        with self.patched_boundaries(), self.assertRaisesRegex(RuntimeError, "after record"):
            wrapper.main()
        self.assertEqual(self.lease.calls, [("acquire",), ("finish", {
            "status": "failed", "tree_gone": "outer_managed_supervisor_only"})])

    def test_source_preparation_failure_never_acquires_owner(self):
        with self.patched_boundaries(source_error=OSError("synthetic source unavailable")), self.assertRaises(OSError):
            wrapper.main()
        self.assertEqual(self.lease.calls, [])
        self.assertIsNone(self.lease.record)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
