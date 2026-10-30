"""Finite wrapper boundary checks. These tests never spawn an engine or load NN."""
import argparse
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pals_followup_run as wrapper


class FollowupPlanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rovezero-followup-contract-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.assets = self.root / "assets"
        self.output = self.root / "output"
        self.assets.mkdir()
        self.output.mkdir()

    def write(self, path, value):
        data = json.dumps(value).encode("utf-8")
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def fixture(self):
        resource = {"cpu_affinity": [2, 3], "cpu_threads": 2,
                    "memory_high_bytes": 6 * wrapper.GIB,
                    "memory_max_bytes": 12 * wrapper.GIB, "swap_max_bytes": 0}
        manifest = {"schema_version": 4, "contract_revision": "pals/0.2",
                    "purpose": "arena_pilot", "training_executed": False,
                    "pilot": {"games": 2, "base_ms": 120000, "increment_ms": 1000,
                              "max_plies": 256, "wall_time_max_ms": 900000,
                              "cleanup_max_ms": 30000},
                    "resources": [resource, copy.deepcopy(resource)]}
        return {"domain": "rz-pals-arena-launch-v4/1", "sha256": "a" * 64,
                "input": {"domain": "rz-pals-arena-launch-v4/1",
                          "semantic_lock": {"manifest": manifest}}}

    def args(self, value):
        path = self.root / "lock.json"
        digest = self.write(path, value)
        return argparse.Namespace(lock=path, lock_sha256=digest,
                                  cpu_affinity=[3, 2], stage="execute", asset_root=self.assets,
                                  output_root=self.output, label="finite-pilot")

    def test_registered_pilot_produces_finite_argv_without_execution(self):
        binary = self.root / "registered-runner"
        command, evidence = wrapper.arena_plan(self.args(self.fixture()), binary)
        self.assertEqual(command, [str(binary), "execute", str(self.root / "lock.json"),
                                   str(self.assets), str(self.output), "finite-pilot"])
        self.assertEqual(evidence["pilot"]["wall_time_max_ms"], 900000)
        self.assertFalse((self.output / "finite-pilot").exists())

    def test_larger_resources_missing_peer_and_diagnostic_are_rejected(self):
        for alteration in ("memory", "resources", "diagnostic", "wall", "ply"):
            value = self.fixture()
            manifest = value["input"]["semantic_lock"]["manifest"]
            if alteration == "memory":
                manifest["resources"][1]["memory_max_bytes"] += 1
            elif alteration == "resources":
                manifest["resources"] = []
            elif alteration == "diagnostic":
                manifest["purpose"] = "diagnostic"
            elif alteration == "wall":
                manifest["pilot"]["wall_time_max_ms"] += 1
            else:
                manifest["pilot"]["max_plies"] += 1
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                wrapper.arena_plan(self.args(value), self.root / "runner")

    def test_duplicate_json_keys_and_mutated_pins_fail_before_plan(self):
        path = self.root / "duplicate.json"
        path.write_text('{"schema":4,"schema":3}', encoding="utf-8")
        with self.assertRaises(ValueError):
            wrapper.bounded_json(path)
        path.write_bytes(b"actual executable")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.write_bytes(b"replaced executable")
        with self.assertRaises(ValueError):
            wrapper.pinned(path, digest, 1024)

    def test_new_model_profile_does_not_reinterpret_legacy_export(self):
        export = self.root / "export.json"
        digest = self.write(export, {"schema": "rovezero.pals-model.v1", "config": {}})
        args = argparse.Namespace(export=export, export_sha256=digest,
                                  model_profile="full_line_interaction_v2")
        with self.assertRaises(ValueError):
            wrapper.model_plan(args, self.root / "validator")

    def test_cuda_control_inventory_is_explicit_pinned_and_bound_to_export(self):
        export = self.root / "control-export.json"
        export_sha = self.write(export, {"config": {}})
        fixtures, runtime, bundle = (self.root / name for name in ("fixtures", "runtime", "bundle"))
        for path in (fixtures, runtime, bundle):
            path.write_bytes(b"pinned input")
        digest = hashlib.sha256(b"pinned input").hexdigest()
        inventory = self.root / "inventory.json"
        value = {"schema": "rovezero.pals-static-control-inventory.v2",
                 "manifest_sha256": export_sha, "actual_provider_placement": "not_observed",
                 "cpu_allowlist": "not_created"}
        inventory_sha = self.write(inventory, value)
        args = argparse.Namespace(export=export, export_sha256=export_sha,
            model_profile="legacy_summary_v1", fixtures=fixtures, fixtures_sha256=digest,
            runtime=runtime, runtime_sha256=digest, model_report=self.output / "numeric.json",
            runtime_cache=self.output, provider="cuda-control", cuda_bundle=bundle,
            cuda_bundle_sha256=digest, control_inventory=inventory,
            control_inventory_sha256=inventory_sha)
        for provider in ("cuda-control", "cuda-control-conv-default"):
            args.provider = provider
            command, evidence = wrapper.model_plan(args, self.root / "validator")
            self.assertEqual(command[-2:], [str(inventory), inventory_sha])
            self.assertEqual(command[7], provider)
            self.assertTrue(evidence["native_placement_audit_required"])
            if provider == "cuda-control":
                self.assertNotIn("cudnn_conv_search_diagnostic", evidence)
            else:
                observation = evidence["cudnn_conv_search_diagnostic"]
                self.assertEqual(observation["cudnn_conv_algo_search_requested"], "DEFAULT")
                self.assertEqual(observation["native_loading_profile_requested"], "eager-cuda12-cudnn9-v1")
                self.assertIs(observation["tf32_requested"], False)
                for name in ("actual_cudnn_plan", "numerical_note", "tf32_actual", "tf32_retry"):
                    self.assertEqual(observation[name], "unknown")
                self.assertEqual(observation["backend_execution_descriptor"], "requires_loaded_checker_receipt")
                for name in ("production_default_changed", "warm_success_claim", "pilot_success_claim"):
                    self.assertIs(observation[name], False)
            for alteration in ("ordinary_provider", "partial_pin", "another_export", "invented_placement"):
                invalid = copy.copy(args)
                if alteration == "ordinary_provider":
                    invalid.provider = "cuda"
                elif alteration == "partial_pin":
                    invalid.control_inventory_sha256 = None
                else:
                    changed = dict(value)
                    changed["manifest_sha256" if alteration == "another_export" else "actual_provider_placement"] = "unknown"
                    invalid.control_inventory_sha256 = self.write(inventory, changed)
                with self.subTest(provider=provider, alteration=alteration), self.assertRaises(ValueError):
                    wrapper.model_plan(invalid, self.root / "validator")
                inventory.write_text(json.dumps(value), encoding="utf-8")

    def gc_args(self, value):
        path = self.root / "external-gc.json"
        digest = self.write(path, value)
        return argparse.Namespace(external_gc_ledger=path, external_gc_ledger_sha256=digest)

    def gc_fixture(self):
        return {"schema": wrapper.EXTERNAL_GC_LEDGER_SCHEMA,
                "requests": [{"id": "request-1", "product_hook": False},
                             {"id": "request-2", "product_hook": False}],
                "actual_executions": [{"id": "actual-1", "request_id": "request-1",
                    "scope": "owned_external_controller", "full_gc_calls": 1, "product_hook": False}],
                "memory_contention_observed": True, "performance_evidence_eligible": False}

    def test_external_gc_requests_are_not_counted_as_actual_calls(self):
        evidence = wrapper.external_gc_observation(self.gc_args(self.gc_fixture()))
        self.assertEqual(evidence["request_count"], 2)
        self.assertEqual(evidence["actual_execution_count"], 1)
        self.assertEqual(evidence["actual_full_gc_calls"], 1)
        self.assertFalse(evidence["performance_evidence_eligible"])
        self.assertFalse(evidence["GC_invoked_by_wrapper"])
        self.assertEqual(evidence["product_GC_hooks"], 0)

    def test_unregistered_gc_is_unknown_and_partial_pin_is_rejected(self):
        evidence = wrapper.external_gc_observation(argparse.Namespace(
            external_gc_ledger=None, external_gc_ledger_sha256=None))
        self.assertEqual(evidence["actual_full_gc_calls"], "unknown")
        self.assertFalse(evidence["performance_evidence_eligible"])
        with self.assertRaises(ValueError):
            wrapper.external_gc_observation(argparse.Namespace(
                external_gc_ledger=self.root / "missing.json", external_gc_ledger_sha256=None))

    def test_external_gc_duplicate_actual_product_hook_and_numeric_boolean_rejected(self):
        for alteration in ("duplicate", "product", "count", "contention", "unlinked", "performance"):
            value = self.gc_fixture()
            if alteration == "duplicate":
                value["actual_executions"].append(copy.deepcopy(value["actual_executions"][0]))
            elif alteration == "product":
                value["actual_executions"][0]["scope"] = "product"
            elif alteration == "count":
                value["actual_executions"][0]["full_gc_calls"] = True
            elif alteration == "contention":
                value["memory_contention_observed"] = 1
            elif alteration == "unlinked":
                value["actual_executions"][0]["request_id"] = "absent-request"
            else:
                value["performance_evidence_eligible"] = True
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                wrapper.external_gc_observation(self.gc_args(value))

    def test_fresh_attempt_output_budget_does_not_scan_runtime_or_models(self):
        report = self.root / "report.json"
        report.write_bytes(b"12345678")
        args = argparse.Namespace(mode="model-check", model_report=report, output_bytes_max=7)
        with self.assertRaises(RuntimeError):
            wrapper.check_colab_output_budget(args)
        args.output_bytes_max = 8
        wrapper.check_colab_output_budget(args)

    def test_colab_planning_uses_separate_schema_and_never_spawns(self):
        lock = self.fixture()
        lock["input"]["resource_profile"] = {"profile": "colab_observed"}
        args = self.args(lock)
        binary = self.root / "runner"
        binary.write_bytes(b"registered fixture")
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        receipt = self.root / "colab-receipt.json"
        argv = ["wrapper", "--binary", str(binary), "--binary-sha256", digest,
                "--receipt", str(receipt), "--cpu-affinity", "2", "3",
                "--resource-profile", "colab-observed", "arena", "--stage", "prepare",
                "--lock", str(args.lock), "--lock-sha256", args.lock_sha256,
                "--asset-root", str(self.assets), "--output-root", str(self.output), "--label", "dry-run"]
        with (mock.patch.object(sys, "argv", argv), mock.patch.object(wrapper, "run_group") as spawn,
              contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(wrapper.main(), 0)
        spawn.assert_not_called()
        evidence = json.loads(receipt.read_text())
        self.assertEqual(evidence["schema"], wrapper.COLAB_SCHEMA)
        self.assertFalse(evidence["strict_local_resources_admitted"])
        self.assertFalse(evidence["performance_evidence_eligible"])
        self.assertFalse(evidence["native_arena_policy_relaxed"])
        self.assertEqual(evidence["actual_resources"], "unknown")
        self.assertEqual(evidence["external_GC"]["actual_full_gc_calls"], "unknown")
        self.assertEqual(evidence["native_arena_resource_profile"], "colab_observed")
        self.assertFalse(evidence["pilot_pair_eligible"])

    def test_arena_wrapper_profile_must_match_the_explicit_native_lock(self):
        value = self.fixture()
        args = self.args(value)
        args.resource_profile = "colab-observed"
        with self.assertRaises(ValueError):
            wrapper.arena_plan(args, self.root / "runner")
        value["input"]["resource_profile"] = {"profile": "colab_observed"}
        args = self.args(value)
        with self.assertRaises(ValueError):
            wrapper.arena_plan(args, self.root / "runner")
        args.resource_profile = "colab-observed"
        _, evidence = wrapper.arena_plan(args, self.root / "runner")
        self.assertFalse(evidence["pilot_pair_eligible"])
        value["input"]["resource_profile"]["skip_lifetime_checks"] = True
        args = self.args(value)
        args.resource_profile = "colab-observed"
        with self.assertRaises(ValueError):
            wrapper.arena_plan(args, self.root / "runner")


class DriverLookupTests(unittest.TestCase):
    """Mocked host lookup and execution gates; never load a CUDA library."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rovezero-driver-boundary-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.receipt = self.root / "receipt.json"
        self.driver = str(self.root / "libcuda.so.580.178.04")

    def probe(self, data=None, result=(0, True, None), remaining=30, error=None):
        data = data if data is not None else json.dumps({
            "status": "available", "resolved_path": self.driver}).encode()
        environment = {"PATH": "registered path", "OMP_NUM_THREADS": "2"}
        original = environment.copy()
        def run(argv, cwd, env, timeout, budget):
            self.assertEqual(argv[:3], [sys.executable, "-I", "-c"])
            self.assertEqual(cwd, self.root)
            self.assertIs(env, environment)
            self.assertEqual(timeout, min(10, remaining))
            budget()
            if error:
                raise error
            os.write(1, data)
            return result
        with mock.patch.object(wrapper, "run_group", side_effect=run) as spawn:
            evidence = wrapper.check_colab_driver_lookup(
                self.receipt, self.root, environment, remaining, lambda: None)
        self.assertEqual(environment, original)
        return evidence, spawn

    def test_default_lookup_is_bounded_separate_and_preserves_environment(self):
        evidence, spawn = self.probe(remaining=3)
        self.assertEqual(evidence["status"], "available")
        self.assertEqual(evidence["resolved_path"], self.driver)
        self.assertTrue(evidence["tree_gone"])
        self.assertTrue(evidence["same_environment_as_native_child"])
        self.assertFalse(evidence["cu_init_performed"])
        self.assertFalse(evidence["model_executed"])
        self.assertFalse(evidence["linker_settings_changed"])
        self.assertTrue(evidence["output_eof_observed"])
        spawn.assert_called_once()

    def test_missing_driver_signal_and_incomplete_cleanup_are_not_success(self):
        for index, result in enumerate(((125, True, None), (-11, True, None), (124, True, None),
                                        (0, False, None), (0, "unknown", None),
                                        (0, True, "budget failure"))):
            self.receipt = self.root / f"failure-{index}.json"
            with self.subTest(result=result):
                evidence, _ = self.probe(result=result)
                self.assertEqual(evidence["status"], "lookup_or_supervision_failed")
                self.assertEqual(evidence["tree_gone"], result[1])
                self.assertFalse(evidence["output_eof_observed"])
                self.assertTrue(Path(evidence["log"]).is_file())

    def test_raised_supervisor_preserves_unknown_ownership(self):
        evidence, _ = self.probe(error=RuntimeError("supervisor did not return"))
        self.assertEqual(evidence["status"], "probe_supervision_raised")
        self.assertEqual(evidence["tree_gone"], "unknown")
        self.assertEqual(evidence["child_spawned"], "unknown")

    def test_bounded_decode_and_format_failures_preserve_unknown_observation(self):
        valid = {"status": "available", "resolved_path": self.driver}
        values = (b"", b"not JSON", b"{}", b"[]", b"\xff",
                  json.dumps({**valid, "resolved_path": "libcuda.so.1"}).encode(),
                  json.dumps({**valid, "resolved_path": self.driver + " (deleted)"}).encode(),
                  json.dumps({**valid, "extra": True}).encode(),
                  ('{"status":"available","status":"available","resolved_path":'
                   + json.dumps(self.driver) + '}').encode())
        for index, data in enumerate(values):
            self.receipt = self.root / f"invalid-{index}.json"
            with self.subTest(index=index):
                evidence, _ = self.probe(data=data)
                self.assertEqual(evidence["status"], "observation_unknown")
                self.assertEqual(evidence["exit_code"], 0)
                self.assertIs(evidence["tree_gone"], True)
                self.assertIsNone(evidence["supervisor_error"])
                self.assertTrue(evidence["output_eof_observed"])
                self.assertNotIn("resolved_path", evidence)
                self.assertEqual(Path(evidence["log"]).read_bytes(), data)

    def test_explicit_failed_observation_is_not_softened_to_unknown(self):
        for index, status in enumerate(("unavailable", "failed", "error")):
            self.receipt = self.root / f"reported-failure-{index}.json"
            with self.subTest(status=status):
                evidence, _ = self.probe(data=json.dumps({
                    "status": status, "error_type": "OSError"}).encode())
                self.assertEqual(evidence["status"], "lookup_reported_failure")
                self.assertEqual(evidence["reported_status"], status)
                self.assertTrue(evidence["output_eof_observed"])
                self.assertNotIn("resolved_path", evidence)

    def test_oversized_and_unreadable_probe_output_remain_failures(self):
        evidence, _ = self.probe(data=b"x" * (64 * 1024 + 1))
        self.assertEqual(evidence["status"], "invalid_probe_output")
        self.assertFalse(evidence["output_eof_observed"])
        self.receipt = self.root / "unreadable-output.json"
        original_open = Path.open
        def open_log(path, mode="r", *args, **kwargs):
            if path == self.receipt.with_name(self.receipt.name + ".driver-lookup.log") and mode == "rb":
                raise OSError("synthetic unreadable probe output")
            return original_open(path, mode, *args, **kwargs)
        with mock.patch.object(Path, "open", autospec=True, side_effect=open_log):
            evidence, _ = self.probe()
        self.assertEqual(evidence["status"], "probe_output_io_failed")
        self.assertEqual(evidence["output_error_type"], "OSError")
        self.assertTrue(evidence["tree_gone"])
        self.assertFalse(evidence["output_eof_observed"])

    def test_expired_lookup_never_spawns_and_existing_log_is_not_overwritten(self):
        evidence, spawn = self.probe(remaining=0)
        spawn.assert_not_called()
        self.assertEqual(evidence["status"], "deadline_expired")
        log = self.receipt.with_name(self.receipt.name + ".driver-lookup.log")
        log.write_bytes(b"retained evidence")
        with self.assertRaises(ValueError):
            self.probe()
        self.assertEqual(log.read_bytes(), b"retained evidence")

    @unittest.skipUnless(sys.platform == "linux", "actual probe supervision requires Linux")
    def test_real_supervisor_captures_stub_output_and_reaps_before_admission(self):
        # Exercise the real inherited-FD/process-group boundary with tiny Python
        # stubs. No CUDA library, NN, optimizer or engine is loaded.
        for code in (0, 125):
            self.receipt = self.root / f"actual-supervisor-{code}.json"
            script = ("import json,os,sys; assert os.environ['RZ_PROBE_STUB']=='1'; "
                      + "print(" + repr(json.dumps({"status": "available",
                                                    "resolved_path": self.driver})) + "); "
                      + f"sys.exit({code})")
            with mock.patch.object(wrapper, "CUDA_DRIVER_LOOKUP", script):
                evidence = wrapper.check_colab_driver_lookup(
                    self.receipt, self.root, {"RZ_PROBE_STUB": "1"}, 3, lambda: None)
            self.assertTrue(evidence["tree_gone"])
            self.assertEqual(evidence["exit_code"], code)
            self.assertEqual(evidence["status"],
                             "available" if code == 0 else "lookup_or_supervision_failed")

    def model_argv(self, provider="cuda", resource="colab-observed", execute=True):
        files = {"binary": b"fixture executable", "export": json.dumps({
            "config": {"profile": "legacy_summary_v1"}}).encode(),
            "runtime": b"fixture runtime", "fixtures": b"{}", "cuda-bundle": b"{}"}
        argv = ["wrapper", "--receipt", str(self.receipt), "--cpu-affinity", "0", "2",
                "--resource-profile", resource, "--seconds", "100"]
        for name in ("binary",):
            path = self.root / name
            path.write_bytes(files[name])
            argv += ["--" + name, str(path), "--" + name + "-sha256",
                     hashlib.sha256(files[name]).hexdigest()]
        if execute:
            argv.append("--run")
        argv += ["model-check", "--model-profile", "legacy_summary_v1", "--provider", provider,
                 "--runtime-cache", str(self.root / "runtime-cache"),
                 "--model-report", str(self.root / "model-report.json")]
        for name in ("export", "runtime", "fixtures", "cuda-bundle"):
            if name == "cuda-bundle" and provider == "cpu":
                continue
            path = self.root / name
            path.write_bytes(files[name])
            argv += ["--" + name, str(path), "--" + name + "-sha256",
                     hashlib.sha256(files[name]).hexdigest()]
        return argv

    def closed_probe(self, **overrides):
        return {"status": "available", "tree_gone": True, "exit_code": 0,
                "supervisor_error": None, "output_eof_observed": True, **overrides}

    def captured_probe(self, data, result=(0, True, None)):
        check = wrapper.check_colab_driver_lookup
        def observe(*args):
            def run(_argv, _cwd, _env, _timeout, budget):
                budget()
                os.write(1, data)
                return result
            with mock.patch.object(wrapper, "run_group", side_effect=run):
                return check(*args)
        return observe

    def execute(self, argv, probe, *, prepared_at=2, lookup_completed_at=None,
                native_completed_at=3, lifecycle_error=None, resource_error=None,
                post_resource_error=None, native_error=None, native_result=(0, True, None),
                output_budget_error_after_preflight=None):
        # Time changes at simulated work boundaries, not at each observation.
        # Additional monotonic() samples must not change the fixture's cost or
        # exhaust a sequence despite preserving the original deadline contract.
        now = [0]
        observations = [0]
        budgets = [0]
        def advance_to(instant):
            if instant is not None:
                self.assertGreaterEqual(instant, now[0])
                now[0] = instant
        def observe(*_args):
            observations[0] += 1
            error = resource_error if observations[0] == 1 else post_resource_error
            if error is not None:
                raise error
            if now[0] < prepared_at:
                advance_to(prepared_at)
            return {}
        def output_budget(*_args):
            budgets[0] += 1
            if budgets[0] > 1 and output_budget_error_after_preflight is not None:
                raise output_budget_error_after_preflight
        def lookup_driver(*args):
            observation = probe(*args) if callable(probe) else probe
            advance_to(lookup_completed_at)
            return observation
        def run_native(*_args):
            advance_to(native_completed_at)
            if native_error is not None:
                raise native_error
            return native_result
        with (mock.patch.object(sys, "argv", argv),
              mock.patch.object(wrapper.sys, "platform", "linux"),
              mock.patch.object(wrapper, "observe_resources", side_effect=observe),
              mock.patch.object(wrapper, "check_resources", side_effect=observe),
              mock.patch.object(wrapper, "check_colab_driver_lookup", side_effect=lookup_driver) as lookup,
              mock.patch.object(wrapper.time, "monotonic", side_effect=lambda: now[0]),
              mock.patch.object(wrapper, "run_group", side_effect=run_native) as native,
              mock.patch.object(wrapper, "OutputLease") as lifecycle,
              mock.patch.object(wrapper, "check_colab_output_budget", side_effect=output_budget),
              contextlib.redirect_stdout(io.StringIO())):
            self.last_lifecycle, self.last_native, self.last_lookup = lifecycle, native, lookup
            lifecycle.return_value.acquire.return_value.finish.side_effect = lifecycle_error
            code = wrapper.main()
        return code, json.loads(self.receipt.read_text()), lookup, native

    def test_final_lifecycle_failure_is_not_recorded_as_completed_native_success(self):
        code, receipt, _, native = self.execute(self.model_argv(provider="cpu"), None,
            lifecycle_error=RuntimeError("synthetic final output quota failure"))
        native.assert_called_once()
        self.assertEqual(code, 125)
        self.assertEqual(receipt['status'], 'failed_output_lifecycle')
        self.assertEqual(receipt['producer_exit_code'], 0)
        self.assertEqual(receipt['exit_code'], 125)
        self.assertEqual(receipt['storage_lifecycle_error'], {
            'type': 'RuntimeError', 'message': 'synthetic final output quota failure'})
        self.last_lifecycle.return_value.acquire.return_value.finish.assert_called_once()
        self.assertFalse(receipt['output_lifecycle']['finish_returned'])

    def test_resource_preflight_failure_issues_no_lease_or_child(self):
        with self.assertRaisesRegex(ValueError, '^synthetic resource mismatch$'):
            self.execute(self.model_argv(provider="cpu", resource="strict-local"), None,
                resource_error=ValueError("synthetic resource mismatch"))
        self.last_lifecycle.assert_not_called()
        self.last_native.assert_not_called()
        self.last_lookup.assert_not_called()

    def test_after_acquire_before_child_failure_finishes_once_with_no_child(self):
        code, receipt, _, native = self.execute(self.model_argv(provider="cpu"), None,
            output_budget_error_after_preflight=RuntimeError("synthetic pre-dispatch quota failure"))
        native.assert_not_called()
        self.last_lifecycle.return_value.acquire.return_value.finish.assert_called_once_with(
            status="failed_wrapper_supervision_or_observation", tree_gone=True)
        self.assertEqual(code, 125)
        self.assertFalse(receipt['producer_dispatch_attempted'])
        self.assertFalse(receipt['producer_result_observed'])
        self.assertIsNone(receipt['producer_exit_code'])
        self.assertEqual(receipt['wrapper_error'], {'stage': 'before_producer_dispatch',
            'type': 'RuntimeError', 'message': 'synthetic pre-dispatch quota failure'})

    def test_raised_supervisor_finishes_once_but_preserves_unknown_and_original_error(self):
        code, receipt, _, native = self.execute(self.model_argv(provider="cpu"), None,
            native_error=RuntimeError("synthetic original supervisor failure"),
            lifecycle_error=RuntimeError("synthetic finish failure"))
        native.assert_called_once()
        self.last_lifecycle.return_value.acquire.return_value.finish.assert_called_once_with(
            status="failed_wrapper_supervision_or_observation", tree_gone="unknown")
        self.assertEqual(code, 125)
        self.assertEqual(receipt['tree_gone'], 'unknown')
        self.assertEqual(receipt['child_spawned'], 'unknown')
        self.assertFalse(receipt['producer_result_observed'])
        self.assertIsNone(receipt['producer_exit_code'])
        self.assertEqual(receipt['wrapper_error'], {'stage': 'producer_supervision',
            'type': 'RuntimeError', 'message': 'synthetic original supervisor failure'})
        self.assertEqual(receipt['storage_lifecycle_error'], {
            'type': 'RuntimeError', 'message': 'synthetic finish failure'})

    def test_post_run_observer_failure_preserves_actual_producer_result(self):
        code, receipt, _, native = self.execute(self.model_argv(provider="cpu"), None,
            post_resource_error=OSError("synthetic post-run resource read failure"))
        native.assert_called_once()
        self.last_lifecycle.return_value.acquire.return_value.finish.assert_called_once_with(
            status="failed_wrapper_supervision_or_observation", tree_gone=True)
        self.assertEqual(code, 125)
        self.assertEqual(receipt['producer_exit_code'], 0)
        self.assertTrue(receipt['producer_tree_gone'])
        self.assertTrue(receipt['producer_result_observed'])
        self.assertIsNone(receipt['producer_supervisor_error'])
        self.assertEqual(receipt['wrapper_error'], {'stage': 'post_producer_observation',
            'type': 'OSError', 'message': 'synthetic post-run resource read failure'})

    def test_returned_supervisor_failure_does_not_become_zero_exit_success(self):
        code, receipt, _, native = self.execute(self.model_argv(provider="cpu"), None,
            native_result=(0, True, "synthetic original cleanup diagnostic"))
        native.assert_called_once()
        self.last_lifecycle.return_value.acquire.return_value.finish.assert_called_once_with(
            status="failed_ownership_preserved", tree_gone=True)
        self.assertEqual(code, 125)
        self.assertEqual(receipt['producer_exit_code'], 0)
        self.assertEqual(receipt['producer_supervisor_error'], "synthetic original cleanup diagnostic")
        self.assertEqual(receipt['supervisor_error'], "synthetic original cleanup diagnostic")
        self.assertEqual(receipt['status'], 'failed_ownership_preserved')

    def test_original_control_flow_is_rethrown_after_unknown_finish_and_receipt(self):
        for index, original in enumerate((KeyboardInterrupt("synthetic cancellation"), SystemExit(7))):
            self.receipt = self.root / f"control-flow-{index}.json"
            with self.subTest(kind=type(original).__name__), self.assertRaises(type(original)) as caught:
                self.execute(self.model_argv(provider="cpu"), None, native_error=original,
                    lifecycle_error=RuntimeError("synthetic finish failure during cancellation"))
            self.assertIs(caught.exception, original)
            self.last_native.assert_called_once()
            self.last_lifecycle.return_value.acquire.return_value.finish.assert_called_once_with(
                status="failed_wrapper_supervision_or_observation", tree_gone="unknown")
            receipt = json.loads(self.receipt.read_text())
            self.assertEqual(receipt['status'], 'interrupted_control_flow')
            self.assertIsNone(receipt['exit_code'])
            self.assertEqual(receipt['tree_gone'], 'unknown')
            self.assertFalse(receipt['producer_result_observed'])
            self.assertIsNone(receipt['producer_exit_code'])
            self.assertEqual(receipt['control_flow_exception'], {
                'type': type(original).__name__, 'message': str(original)})
            self.assertEqual(receipt['wrapper_error'], {'stage': 'producer_supervision',
                'type': type(original).__name__, 'message': str(original)})
            self.assertEqual(receipt['storage_lifecycle_error'], {'type': 'RuntimeError',
                'message': 'synthetic finish failure during cancellation'})

    def test_finish_control_flow_is_not_swallowed_after_original_producer_result(self):
        original = KeyboardInterrupt("synthetic finish cancellation")
        with self.assertRaises(KeyboardInterrupt) as caught:
            self.execute(self.model_argv(provider="cpu"), None, lifecycle_error=original)
        self.assertIs(caught.exception, original)
        self.last_lifecycle.return_value.acquire.return_value.finish.assert_called_once()
        receipt = json.loads(self.receipt.read_text())
        self.assertEqual(receipt['status'], 'interrupted_control_flow')
        self.assertIsNone(receipt['exit_code'])
        self.assertEqual(receipt['producer_exit_code'], 0)
        self.assertTrue(receipt['producer_result_observed'])
        self.assertTrue(receipt['producer_tree_gone'])
        self.assertEqual(receipt['storage_lifecycle_error'], {
            'type': 'KeyboardInterrupt', 'message': 'synthetic finish cancellation'})

    def test_failed_probe_blocks_model_even_after_whole_window_expires(self):
        cases = (
            (b"not JSON", (125, True, None), "lookup_or_supervision_failed"),
            (b"not JSON", (-11, True, None), "lookup_or_supervision_failed"),
            (b"not JSON", (124, True, None), "lookup_or_supervision_failed"),
            (b"not JSON", (0, False, None), "lookup_or_supervision_failed"),
            (b"not JSON", (0, "unknown", None), "lookup_or_supervision_failed"),
            (b"not JSON", (0, True, "budget failure"), "lookup_or_supervision_failed"),
            (b"x" * (64 * 1024 + 1), (0, True, None), "invalid_probe_output"),
            (b'{"status":"unavailable"}', (0, True, None), "lookup_reported_failure"),
        )
        for index, (data, result, status) in enumerate(cases):
            self.receipt = self.root / f"blocked-model-{index}.json"
            with self.subTest(status=status, result=result):
                code, receipt, lookup, native = self.execute(self.model_argv(),
                    self.captured_probe(data, result), lookup_completed_at=101)
                lookup.assert_called_once()
                native.assert_not_called()
                self.assertEqual(code, 125)
                self.assertEqual(receipt["status"], "host_driver_lookup_failed_no_model_spawned")
                self.assertEqual(receipt["host_driver_lookup"]["status"], status)
                self.assertEqual(receipt["tree_gone"], result[1])
                self.assertFalse(receipt["child_spawned"])
                self.assertFalse(receipt["producer_dispatch_attempted"])

    def test_unknown_probe_requires_original_closure_and_complete_output_before_dispatch(self):
        cases = ({"exit_code": None}, {"exit_code": 125}, {"tree_gone": "unknown"},
                 {"supervisor_error": "original supervision failure"},
                 {"output_eof_observed": False}, {"status": "probe_output_io_failed"})
        for index, changed in enumerate(cases):
            self.receipt = self.root / f"incomplete-observation-{index}.json"
            with self.subTest(changed=changed):
                code, receipt, _, native = self.execute(self.model_argv(),
                    self.closed_probe(status="observation_unknown") | changed)
                native.assert_not_called()
                self.assertEqual(code, 125)
                self.assertFalse(receipt["producer_dispatch_attempted"])

    def test_malformed_closed_probe_output_dispatches_original_model_command(self):
        for index, data in enumerate((b"not JSON", b"\xff")):
            self.receipt = self.root / f"unknown-observation-{index}.json"
            argv = self.model_argv()
            flag = lambda name: argv[argv.index("--" + name) + 1]
            expected = [flag(name) for name in ("binary", "export", "export-sha256", "runtime",
                        "runtime-sha256", "fixtures", "model-report", "provider", "runtime-cache", "cuda-bundle")]
            with self.subTest(data=data):
                code, receipt, lookup, native = self.execute(argv, self.captured_probe(data),
                    lookup_completed_at=8, native_completed_at=9)
                native.assert_called_once()
                self.assertEqual(native.call_args.args[0], expected)
                self.assertEqual(native.call_args.args[3], 92)
                self.assertIs(lookup.call_args.args[2], native.call_args.args[2])
                observation = receipt["host_driver_lookup"]
                self.assertEqual(observation["status"], "observation_unknown")
                self.assertTrue(observation["output_eof_observed"])
                self.assertNotIn("resolved_path", observation)
                self.assertEqual(Path(observation["log"]).read_bytes(), data)
                self.assertTrue(receipt["producer_dispatch_attempted"])
                self.assertTrue(receipt["producer_result_observed"])
                self.assertEqual(code, 0)
                self.assertEqual(receipt["status"], "completed_observed_profile")

    def test_successful_probe_cost_is_charged_to_original_model_window(self):
        code, receipt, lookup, native = self.execute(self.model_argv(),
            self.closed_probe(), lookup_completed_at=8, native_completed_at=9)
        lookup.assert_called_once()
        native.assert_called_once()
        self.assertEqual(native.call_args.args[3], 92)
        self.assertIs(lookup.call_args.args[2], native.call_args.args[2])
        self.assertEqual(code, 0)
        self.assertEqual(receipt["status"], "completed_observed_profile")

    def test_successful_probe_cannot_restart_an_expired_original_model_window(self):
        for index, status in enumerate(("available", "observation_unknown")):
            self.receipt = self.root / f"expired-model-window-{index}.json"
            with self.subTest(status=status):
                code, receipt, lookup, native = self.execute(self.model_argv(),
                    self.closed_probe(status=status), lookup_completed_at=101)
                lookup.assert_called_once()
                native.assert_not_called()
                self.assertEqual(code, 124)
                self.assertEqual(receipt["status"], "preparation_deadline_expired")
                self.assertTrue(receipt["tree_gone"])
                self.assertFalse(receipt["child_spawned"])

    def test_planning_cpu_and_strict_local_do_not_run_host_lookup(self):
        cases = (("cuda", "colab-observed", False),
                 ("cpu", "colab-observed", True),
                 ("cuda", "strict-local", True))
        for index, (provider, resource, execute) in enumerate(cases):
            self.receipt = self.root / f"no-lookup-{index}.json"
            with self.subTest(provider=provider, resource=resource, execute=execute):
                code, receipt, lookup, native = self.execute(
                    self.model_argv(provider, resource, execute), None)
                lookup.assert_not_called()
                self.assertNotIn("host_driver_lookup", receipt)
                self.assertEqual(native.call_count, int(execute))
                self.assertEqual(code, 0)


class ResourceObservationTests(unittest.TestCase):
    def reads(self, values):
        def read(path, **unused):
            if path.as_posix() == "/proc/self/cgroup":
                return "0::/colab-observed\n"
            if path.name not in values:
                raise FileNotFoundError(path.name)
            return values[path.name]
        return read

    def observe(self, values, cpus=(0, 2), writable=False):
        with mock.patch.object(wrapper.sys, "platform", "linux"), \
                mock.patch.object(wrapper.os, "sched_getaffinity", return_value=set(cpus), create=True), \
                mock.patch.object(Path, "read_text", self.reads(values)), \
                mock.patch.object(wrapper.os, "access", return_value=writable):
            return wrapper.observe_resources([0, 2])

    def values(self):
        return {"cpu.max": "200000 100000", "memory.high": str(6 * wrapper.GIB),
                "memory.max": str(12 * wrapper.GIB), "memory.swap.max": "0", "pids.max": "128"}

    def test_matching_observations_do_not_promote_to_strict_local(self):
        evidence = self.observe(self.values(), writable=True)
        self.assertTrue(all(row["status"] == "satisfied" for row in evidence["observations"].values()))
        self.assertFalse(evidence["strict_local_resources_admitted"])
        self.assertFalse(evidence["settings_installed"])
        self.assertFalse(evidence["cgroup_write_performed"])
        self.assertEqual(evidence["cgroup_write_capability"]["status"], "observed_capability_only")
        self.assertFalse(evidence["cgroup_write_capability"]["applied_settings_verified"])

    def test_colab_unlimited_or_different_resources_are_unfulfilled(self):
        values = self.values()
        values.update({"cpu.max": "max 100000", "memory.high": "max", "memory.max": "max"})
        evidence = self.observe(values, cpus=(0, 1, 2, 3))
        for name in ("cpu_affinity", "cpu_quota", "memory.high", "memory.max"):
            self.assertEqual(evidence["observations"][name]["status"], "unfulfilled")
        self.assertEqual(evidence["cgroup_write_capability"]["status"], "unfulfilled")
        self.assertFalse(evidence["strict_local_resources_admitted"])

    def test_missing_or_malformed_quota_and_memory_are_unknown(self):
        evidence = self.observe({"cpu.max": "nan 0", "memory.high": "not-a-limit"})
        for name in ("cpu_quota", "memory.high", "memory.max", "memory.swap.max", "pids.max"):
            self.assertEqual(evidence["observations"][name]["status"], "unknown")

    def test_unsafe_membership_is_not_used_for_limit_reads(self):
        read_paths = []
        def read_membership(path, **unused):
            read_paths.append(path)
            self.assertEqual(path, Path("/proc/self/cgroup"))
            return "0::/../outside\n"
        with mock.patch.object(wrapper.sys, "platform", "linux"), \
                mock.patch.object(wrapper.os, "sched_getaffinity", return_value={0, 2}, create=True), \
                mock.patch.object(Path, "read_text", autospec=True, side_effect=read_membership):
            evidence = wrapper.observe_resources([0, 2])
        self.assertIn(Path("/proc/self/cgroup"), read_paths)
        self.assertIsNone(evidence["cgroup_membership"])
        self.assertEqual(evidence["observations"]["cpu_quota"]["status"], "unknown")

    def test_non_linux_colab_environment_is_explicitly_unfulfilled(self):
        with mock.patch.object(wrapper.sys, "platform", "win32"), \
                mock.patch.object(Path, "read_text") as reads:
            evidence = wrapper.observe_resources([0, 2])
        reads.assert_not_called()
        self.assertEqual(evidence["execution_platform_status"], "unfulfilled")
        self.assertFalse(evidence["strict_local_resources_admitted"])

    def test_legacy_strict_local_checks_are_still_fail_closed(self):
        with mock.patch.object(wrapper.sys, "platform", "linux"), \
                mock.patch.object(wrapper.os, "sched_getaffinity", return_value={0, 2}, create=True), \
                mock.patch.object(Path, "read_text", self.reads(self.values())):
            evidence = wrapper.check_resources([0, 2])
        self.assertEqual(evidence["cpu_affinity"], [0, 2])
        self.assertNotIn("strict_local_resources_admitted", evidence)
        with mock.patch.object(wrapper.sys, "platform", "linux"), \
                mock.patch.object(wrapper.os, "sched_getaffinity", return_value={0, 1, 2}, create=True):
            with self.assertRaises(ValueError):
                wrapper.check_resources([0, 2])


if __name__ == "__main__":
    unittest.main()
