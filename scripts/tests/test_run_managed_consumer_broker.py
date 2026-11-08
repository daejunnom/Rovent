"""Temporary real child/locks only; not product reader or GPU acceptance."""
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import managed_outputs
import rovezero_usage_snapshot
import run_managed
import storage_consumer_broker_budget
import storage_consumer_broker_launch
import storage_consumer_broker_quarantine
from storage_consumer_registry import ConsumerRegistry, require_catalog_binding
from managed_process import ProcessNotSpawned
from managed_storage import CatalogBusy, ManagedBuild, StorageError


@unittest.skipUnless(sys.platform == "linux", "Linux child supervision required")
class RunManagedConsumerBrokerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        (self.source / "scripts").mkdir(parents=True)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.registry = ConsumerRegistry.initialize(self.root / "registry", scope="temporary-fixture")
        self.descriptor = self.root / "descriptor.json"
        self.descriptor.write_text("{}", encoding="ascii")
        self.reader_marker = self.root / "reader-started"
        repo_scripts = Path(__file__).resolve().parents[1]
        for name in ("storage_consumer_broker_client.py", "storage_consumer_broker_transport.py"):
            (self.source / "scripts" / name).write_bytes((repo_scripts / name).read_bytes())
        source = textwrap.dedent(f"""
            import os, socket, sys, time
            from pathlib import Path
            from storage_consumer_broker_client import ConsumerBrokerClient
            fd = int(os.environ.pop("RZ_CONSUMER_BROKER_FD"))
            parent = int(os.environ.pop("RZ_CONSUMER_BROKER_PARENT_PID"))
            endpoint = socket.socket(fileno=fd)
            endpoint.set_inheritable(False)
            endpoint.setblocking(False)
            def peer(pid, uid, gid):
                if (pid, uid, gid) != (parent, os.getuid(), os.getgid()) or os.getppid() != parent:
                    raise RuntimeError("synthetic original parent changed")
            client = ConsumerBrokerClient(endpoint, peer, maximum_seconds=15)
            path = Path(sys.argv[1])
            client.add([path])
            def reader():
                Path(sys.argv[2]).write_text("started", encoding="ascii")
                time.sleep(5.3)
                return path.read_text(encoding="ascii"), []
            try:
                if client.read_and_register(reader, descriptor=path) != "{{}}":
                    raise RuntimeError("synthetic descriptor mismatch")
            finally:
                client.finish()
        """)
        self.script = self.source / "scripts" / "virtual_system_numeric_prepare.py"
        self.script.write_text(source, encoding="utf-8")
        self.sha = hashlib.sha256(self.script.read_bytes()).hexdigest()
        self.samples = []

    def execute(self, digest, *, simulate_unknown=False, simulate_spawn_unknown=False,
                simulate_manager_finish_failure=False, simulate_predispatch_finish_failure=False,
                simulate_arm_failure=False, simulate_proven_no_spawn=False, simulate_interruption=False,
                automatic=False):
        argv = ["run_managed", "--source-root", str(self.source), "--timeout", "20",
                "--consumer-registry-root", str(self.registry.root),
                "--consumer-registry-id", self.registry.registry_id]
        if not automatic:
            argv.extend(["--consumer-broker-reviewed-sha256", digest])
        argv.extend(["--", sys.executable, str(self.script), str(self.descriptor), str(self.reader_marker)])
        env = {"XDG_CACHE_HOME": str(self.cache), "PATH": "/usr/bin:/bin", "LANG": "C",
               "PYTHONDONTWRITEBYTECODE": "1"}
        stderr = io.StringIO()
        original_run_group = run_managed.run_group
        original_gate = storage_consumer_broker_budget.BrokerBudgetGate
        original_launch = storage_consumer_broker_launch.ManagedBrokerLaunch
        original_manager_finish = run_managed.ManagedBuild.finish
        self.quarantine = storage_consumer_broker_quarantine.BrokerQuarantine()
        fixture = self
        interruption = {"signal": None}

        class RecordingLaunch(original_launch):
            def __init__(self, *arguments, **options):
                super().__init__(*arguments, **options)
                fixture.last_launch = self

            def assert_spawn_command(self, command, cwd):
                if simulate_arm_failure:
                    raise StorageError("synthetic arm failure")
                return super().assert_spawn_command(command, cwd)

        def observe_group(*arguments, **options):
            if simulate_spawn_unknown:
                raise OSError("synthetic Popen outcome unknown")
            if simulate_proven_no_spawn:
                raise ProcessNotSpawned("synthetic proof of no spawn")
            code, gone, failure = original_run_group(*arguments, **options)
            # Conservative synthetic observation: the actual fixture child has
            # finished, but the caller is told that the group is unknown.
            return code, False if simulate_unknown else gone, failure

        def finish_manager(manager, *arguments, **options):
            fixture.last_manager = manager
            if simulate_predispatch_finish_failure and options.get("_not_dispatched") is True:
                raise OSError("synthetic pre-dispatch finalization failure")
            if simulate_manager_finish_failure and options.get("tree_gone") is True and not options.get("_not_dispatched"):
                raise OSError("synthetic manager finalization failure")
            return original_manager_finish(manager, *arguments, **options)

        def sample_budget(**_):
            self.samples.append(time.monotonic())
            if simulate_interruption:
                interruption["signal"] = 15

        class DeferredFixtureGate(original_gate):
            def after_tick(self):
                return False  # Exercise the finalization path with dirty state.

        with patch.object(sys, "argv", argv), patch.dict(os.environ, env, clear=True), \
                patch.object(run_managed, "run_group", side_effect=observe_group), \
                patch.object(storage_consumer_broker_launch, "ManagedBrokerLaunch", RecordingLaunch), \
                patch.object(run_managed.ManagedBuild, "finish", autospec=True, side_effect=finish_manager), \
                patch.object(storage_consumer_broker_quarantine, "IN_PROCESS_BROKER_QUARANTINE", self.quarantine), \
                patch.object(storage_consumer_broker_budget, "BrokerBudgetGate",
                             DeferredFixtureGate if simulate_unknown else original_gate), \
                patch.object(run_managed, "sample_output_budget", side_effect=sample_budget), \
                patch.object(managed_outputs, "finalize_supervised_outputs", return_value=None), \
                patch.object(rovezero_usage_snapshot, "warn_cached_total_usage", return_value=None), \
                patch.object(rovezero_usage_snapshot, "claim_total_usage_maintenance", return_value=False), \
                contextlib.redirect_stderr(stderr):
            code = (run_managed._main(interruption=interruption) if simulate_interruption else run_managed.main())
        return code, stderr.getvalue()

    def test_real_child_registers_and_resamples_deferred_budget(self):
        code, receipt = self.execute(self.sha)
        self.assertEqual(code, 0, receipt)
        line = next(item for item in receipt.splitlines() if item.startswith("managed_consumer_broker_observation="))
        observation = json.loads(line.split("=", 1)[1])
        self.assertIs(observation["original_group_gone"], True)
        self.assertGreaterEqual(observation["deferred_budget_count"], 1)
        self.assertGreaterEqual(len(self.samples), 3)
        self.assertFalse((self.cache / "rovezero/build/managed-v1/owner.json").exists())
        with self.registry.guard() as guard:
            self.assertTrue(guard.conflicts([self.descriptor])["conflicts"])

    def test_bound_catalog_automatically_brokers_supported_reader_without_sha_input(self):
        code, receipt = self.execute(self.sha, automatic=True)
        self.assertEqual(code, 0, receipt)
        line = next(item for item in receipt.splitlines() if item.startswith("managed_consumer_broker_profile="))
        observation = json.loads(line.split("=", 1)[1])
        self.assertEqual(observation["reader_contract"], "automatic_unreviewed")
        self.assertEqual(observation["imports"], "unknown")
        with self.registry.guard() as guard:
            self.assertTrue(guard.conflicts([self.descriptor])["conflicts"])

    def test_automatic_unknown_does_not_use_reviewed_guard_release(self):
        code, receipt = self.execute(self.sha, automatic=True, simulate_unknown=True)
        self.assertEqual(code, 125, receipt)
        line = next(item for item in receipt.splitlines() if item.startswith("managed_consumer_broker_quarantine="))
        handle = json.loads(line.split("=", 1)[1])["handle"]
        self.assertFalse(self.last_launch._closed)
        self.assertTrue((self.cache / "rovezero/build/managed-v1/owner.json").exists())
        with patch.object(managed_outputs, "finalize_supervised_outputs", return_value=None), \
                patch.object(run_managed, "sample_output_budget", return_value=None), \
                patch.object(rovezero_usage_snapshot, "warn_cached_total_usage", return_value=None):
            result = self.quarantine.reobserve(handle, seconds=0.1)
        self.assertEqual(result["state"], "group_reobserved_owner_finalized")
        self.assertEqual(result["physical_completion"], "unknown")

    def test_supported_reader_without_registry_keeps_legacy_path(self):
        self.script.write_text("raise SystemExit(0)\n", encoding="utf-8")
        argv = ["run_managed", "--source-root", str(self.source), "--timeout", "20", "--",
                sys.executable, str(self.script)]
        env = {"XDG_CACHE_HOME": str(self.cache), "PATH": "/usr/bin:/bin", "LANG": "C"}
        stderr = io.StringIO()
        with patch.object(sys, "argv", argv), patch.dict(os.environ, env, clear=True), \
                patch.object(run_managed, "sample_output_budget", return_value=None), \
                patch.object(managed_outputs, "finalize_supervised_outputs", return_value=None), \
                patch.object(rovezero_usage_snapshot, "warn_cached_total_usage", return_value=None), \
                patch.object(rovezero_usage_snapshot, "claim_total_usage_maintenance", return_value=False), \
                contextlib.redirect_stderr(stderr):
            code = run_managed.main()
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertNotIn("managed_consumer_broker_profile=", stderr.getvalue())

    def test_legacy_checkout_without_client_module_keeps_normal_result(self):
        (self.source / "scripts" / "storage_consumer_broker_client.py").unlink()
        self.script.write_text("raise SystemExit(0)\n", encoding="utf-8")
        code, receipt = self.execute(self.sha, automatic=True)
        self.assertEqual(code, 0, receipt)
        self.assertNotIn("managed_consumer_broker_profile=", receipt)

    def test_clean_eof_without_handshake_preserves_child_result_not_protection(self):
        self.script.write_text("import os, time\nos.close(int(os.environ['RZ_CONSUMER_BROKER_FD']))\n"
                               "time.sleep(0.5)\n", encoding="utf-8")
        code, receipt = self.execute(self.sha, automatic=True)
        self.assertEqual(code, 0, receipt)
        line = next(item for item in receipt.splitlines() if item.startswith("managed_consumer_broker_observation="))
        observation = json.loads(line.split("=", 1)[1])
        self.assertEqual(observation["participation"], "no_messages_observed")
        self.assertEqual(observation["registered_scope"], "no_consumer_lease_observed")
        self.assertEqual(observation["registered_lease_count"], 0)

    def test_help_skips_automatic_broker_with_bound_registry(self):
        self.script.write_text("import argparse\nargparse.ArgumentParser().parse_args()\n", encoding="utf-8")
        argv = ["run_managed", "--source-root", str(self.source), "--timeout", "20",
                "--consumer-registry-root", str(self.registry.root), "--consumer-registry-id",
                self.registry.registry_id, "--", sys.executable, str(self.script), "--help"]
        env = {"XDG_CACHE_HOME": str(self.cache), "PATH": "/usr/bin:/bin", "LANG": "C"}
        stderr = io.StringIO()
        with patch.object(sys, "argv", argv), patch.dict(os.environ, env, clear=True), \
                patch.object(run_managed, "sample_output_budget", return_value=None), \
                patch.object(managed_outputs, "finalize_supervised_outputs", return_value=None), \
                patch.object(rovezero_usage_snapshot, "warn_cached_total_usage", return_value=None), \
                patch.object(rovezero_usage_snapshot, "claim_total_usage_maintenance", return_value=False), \
                contextlib.redirect_stderr(stderr):
            code = run_managed.main()
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertNotIn("managed_consumer_broker_profile=", stderr.getvalue())

    def test_same_registry_cleaner_preserves_input_during_and_after_read(self):
        with patch.object(rovezero_usage_snapshot, "warn_cached_total_usage", return_value=None):
            cleaner = ManagedBuild(self.root / "other-managed-catalog", self.source,
                                   purpose="cleanup-fixture", dependency_registry=self.registry).acquire()

        def close_cleaner():
            if not cleaner.finished:
                cleaner.finish(exit_code=0, tree_gone=True, _not_dispatched=True)

        self.addCleanup(close_cleaner)
        self.descriptor = cleaner.target / "descriptor.json"
        self.descriptor.write_text("{}", encoding="ascii")
        stop = threading.Event()
        attempt = {}

        def observe_reader_and_clean():
            while not stop.wait(0.01):
                if self.reader_marker.exists():
                    try:
                        try:
                            with self.registry.guard():
                                attempt["registry_lock_busy"] = False
                        except CatalogBusy:
                            attempt["registry_lock_busy"] = True
                        attempt["removed"] = cleaner._remove_with_dependencies(cleaner.target, cleaner.target.parent)
                    except BaseException as error:
                        attempt["error"] = type(error).__name__
                    return

        observer = threading.Thread(target=observe_reader_and_clean)
        observer.start()
        try:
            code, receipt = self.execute(self.sha)
        finally:
            stop.set()
            observer.join(timeout=2)
        self.assertFalse(observer.is_alive())
        self.assertEqual(code, 0, receipt)
        self.assertEqual(attempt, {"registry_lock_busy": True, "removed": False})
        self.assertTrue(self.descriptor.exists())
        # The original broker retains its lease because process end is not a
        # GPU fence, so the same production cleanup remains deferred afterward.
        require_catalog_binding(cleaner.root, self.registry, cleaner.guard)
        with self.registry.guard() as guard:
            conflict = guard.conflicts([cleaner.target])
        self.assertEqual(conflict["coverage"], "registered_only")
        self.assertTrue(any(item["input"] == str(self.descriptor)
                            and item["target"] == str(cleaner.target)
                            and item["state"] == "physical_unknown_retained"
                            for item in conflict["conflicts"]))
        self.assertFalse(conflict["unknown"])
        self.assertFalse(cleaner._remove_with_dependencies(cleaner.target, cleaner.target.parent))
        self.assertTrue(self.descriptor.exists())

    def test_synthetic_unknown_group_resamples_dirty_budget_and_preserves_owner(self):
        code, receipt = self.execute(self.sha, simulate_unknown=True)
        self.assertEqual(code, 125, receipt)
        line = next(item for item in receipt.splitlines() if item.startswith("managed_consumer_broker_budget="))
        observation = json.loads(line.split("=", 1)[1])
        self.assertEqual(observation["status"], "deferred_check_completed")
        self.assertEqual(observation["group_scope"], "unknown")
        self.assertIs(observation["final_peak_attested"], False)
        self.assertTrue((self.cache / "rovezero/build/managed-v1/owner.json").exists())
        with self.registry.guard() as guard:
            self.assertTrue(guard.conflicts([self.descriptor])["conflicts"])
        line = next(item for item in receipt.splitlines() if item.startswith("managed_consumer_broker_quarantine="))
        handle = json.loads(line.split("=", 1)[1])["handle"]
        with patch.object(managed_outputs, "finalize_supervised_outputs", return_value=None):
            result = self.quarantine.reobserve(handle, seconds=0.1)
        self.assertEqual(result["state"], "group_reobserved_owner_finalized")
        self.assertEqual(result["physical_completion"], "unknown")
        self.assertFalse((self.cache / "rovezero/build/managed-v1/owner.json").exists())

    def test_nonmatching_reviewed_source_rejects_only_opt_in_without_owner(self):
        code, receipt = self.execute("0" * 64)
        self.assertEqual(code, 125, receipt)
        self.assertFalse((self.cache / "rovezero/build/managed-v1/owner.json").exists())

    def test_synthetic_unknown_spawn_revokes_pins_and_other_catalog_runs(self):
        code, receipt = self.execute(self.sha, simulate_spawn_unknown=True)
        self.assertEqual(code, 125, receipt)
        owner = self.cache / "rovezero/build/managed-v1/owner.json"
        self.assertTrue(owner.exists())  # Original process outcome is unknown.
        line = next(item for item in receipt.splitlines() if item.startswith("managed_consumer_broker_quarantine="))
        handle = json.loads(line.split("=", 1)[1])["handle"]
        self.assertEqual(self.quarantine.reobserve(handle, seconds=0.1)["state"],
                         "original_popen_unavailable")
        self.assertTrue(self.last_launch._closed)
        # Synthetic fixture did not spawn a child; release only this test's
        # live guard after checking that the unknown owner record remains.
        self.addCleanup(self.last_launch.manager.guard.release)
        self.assertEqual(self.last_launch.domain._entries, [])
        self.assertIsNone(self.last_launch._parent)
        self.assertIsNone(self.last_launch._child_endpoint)

        separate = self.root / "independent-cache"
        separate.mkdir()
        env = {"XDG_CACHE_HOME": str(separate), "PATH": "/usr/bin:/bin", "LANG": "C"}
        argv = ["run_managed", "--source-root", str(self.source), "--timeout", "20", "--", "/bin/true"]
        with patch.object(sys, "argv", argv), patch.dict(os.environ, env, clear=True), \
                patch.object(run_managed, "sample_output_budget", return_value=None), \
                patch.object(managed_outputs, "finalize_supervised_outputs", return_value=None), \
                patch.object(rovezero_usage_snapshot, "warn_cached_total_usage", return_value=None), \
                patch.object(rovezero_usage_snapshot, "claim_total_usage_maintenance", return_value=False), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run_managed.main(), 0)
        self.assertTrue(owner.exists())
        self.assertFalse((separate / "rovezero/build/managed-v1/owner.json").exists())

    def test_manager_finish_failure_retains_original_without_automatic_retry(self):
        code, receipt = self.execute(self.sha, simulate_manager_finish_failure=True)
        self.assertEqual(code, 125, receipt)
        line = next(item for item in receipt.splitlines() if item.startswith("managed_consumer_broker_quarantine="))
        observation = json.loads(line.split("=", 1)[1])
        self.assertEqual(observation["process_exit_lock_survival"], "not_claimed")
        self.assertEqual(observation["shared_catalog_guard"], "may_block_until_release")
        with self.assertRaises(RuntimeError):
            self.quarantine.reobserve(observation["handle"], seconds=0.1)
        self.assertTrue((self.cache / "rovezero/build/managed-v1/owner.json").exists())
        # Synthetic failure happened before the mock touched the original
        # manager; close only the fixture's guard after observing the record.
        self.addCleanup(self.last_launch.manager.guard.release)

    def assert_predispatch_manager_retained(self, result):
        code, receipt = result
        self.assertEqual(code, 125, receipt)
        line = next(item for item in receipt.splitlines() if item.startswith("managed_consumer_broker_quarantine="))
        observation = json.loads(line.split("=", 1)[1])
        self.assertEqual(observation["owner_kind"], "manager_finalization_unknown")
        with self.assertRaises(RuntimeError):
            self.quarantine.reobserve(observation["handle"], seconds=0.1)
        self.assertTrue((self.cache / "rovezero/build/managed-v1/owner.json").exists())
        self.addCleanup(self.last_manager.guard.release)

    def test_preparation_cleanup_failure_retains_original_manager(self):
        self.assert_predispatch_manager_retained(
            self.execute("0" * 64, simulate_predispatch_finish_failure=True))

    def test_arm_cleanup_failure_retains_original_manager(self):
        self.assert_predispatch_manager_retained(
            self.execute(self.sha, simulate_arm_failure=True, simulate_predispatch_finish_failure=True))

    def test_proven_no_spawn_cleanup_failure_retains_original_manager(self):
        self.assert_predispatch_manager_retained(
            self.execute(self.sha, simulate_proven_no_spawn=True, simulate_predispatch_finish_failure=True))

    def test_predispatch_interruption_cleanup_failure_retains_original_manager(self):
        self.assert_predispatch_manager_retained(
            self.execute(self.sha, simulate_interruption=True, simulate_predispatch_finish_failure=True))


if __name__ == "__main__":
    unittest.main()
