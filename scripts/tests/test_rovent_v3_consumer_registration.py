"""V3 input ownership at the original parent boundary; no model or GPU."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import signal
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rovent_v3_parent as parent
import managed_process
from storage_consumer_registry import ConsumerRegistry, bind_catalog
from storage_guard import CatalogGuard


@unittest.skipUnless(sys.platform == "linux", "V3 original owner is Linux only")
class V3ConsumerRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.catalog = self.root / "catalog"
        self.catalog.mkdir()
        self.registry = ConsumerRegistry.initialize(self.root / "registry", scope="v3-input-test")
        with CatalogGuard(self.catalog / "catalog.lock") as guard:
            bind_catalog(self.catalog, self.registry, guard)
        self.options = argparse.Namespace(consumer_registry_root=self.registry.root,
                                          consumer_registry_id=self.registry.registry_id,
                                          consumer_registry_required=True,
                                          consumer_input_catalog=[self.catalog])
        self.environment = patch.dict(os.environ, {"RZ_CONSUMER_REGISTRY_REQUIRED": "0"}, clear=True)
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temp.cleanup()

    def _protected(self, *paths):
        with self.registry.guard() as guard:
            for path in paths:
                result = guard.conflicts([path])
                self.assertTrue(result["conflicts"], str(path))

    def _plan(self):
        checkout = self.root / "checkout"
        assets = self.root / "assets"
        checkout.mkdir()
        assets.mkdir()
        native = self.root / "engine"
        registration = self.root / "registration"
        library = self.root / "cuda-library"
        for path in (native, registration, library):
            path.write_bytes(b"synthetic input; never executed")
        request = self.root / "request.json"
        request_bytes = json.dumps({"system_runtime": {"libraries": [{"path": str(library)}]}}).encode()
        request.write_bytes(request_bytes)
        output = self.root / "future-output"
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"schema": "rz-rovent-v3-managed-parent-plan/1",
                                    "checkout_root": str(checkout), "asset_root": str(assets),
                                    "output": str(output),
                                    "request": {"path": str(request), "bytes": len(request_bytes),
                                                "sha256": hashlib.sha256(request_bytes).hexdigest()},
                                    "native_binary": {"path": str(native)},
                                    "registration_binary": {"path": str(registration)}}), encoding="utf8")
        return plan, request, checkout, assets, native, registration, library, output

    def test_managed_registers_dynamic_inputs_and_closes_without_dispatch(self):
        plan, request, checkout, assets, native, registration, library, output = self._plan()

        def before_dispatch(_plan, *, consumer, prepared, started, interruption):
            self.assertIsNotNone(prepared)
            self.assertFalse(consumer.dispatched)
            self._protected(plan, request, checkout, assets, native, registration, library, output)
            return 0

        with patch.object(parent, "_run_managed", side_effect=before_dispatch):
            self.assertEqual(parent.run_managed(plan, consumer_args=self.options), 0)
        with self.registry.guard() as guard:
            self.assertFalse(guard.conflicts([library])["conflicts"])

    def test_managed_retains_dispatched_inputs_when_only_group_end_is_known(self):
        plan, _, _, _, _, _, library, output = self._plan()

        def dispatched(_plan, *, consumer, prepared, started, interruption):
            consumer.dispatched = True
            return 0

        with patch.object(parent, "_run_managed", side_effect=dispatched):
            self.assertEqual(parent.run_managed(plan, consumer_args=self.options), 0)
        with self.registry.guard() as guard:
            matches = guard.conflicts([library])["conflicts"]
            self.assertTrue(matches)
            self.assertTrue(all(row["state"] == "physical_unknown_retained" for row in matches))
            self.assertTrue(guard.conflicts([output])["conflicts"])

    def test_pre_dispatch_failure_closes_registration(self):
        plan, _, _, _, _, _, library, _ = self._plan()
        with patch.object(parent, "_run_managed", side_effect=ValueError("before dispatch")):
            with self.assertRaisesRegex(ValueError, "before dispatch"):
                parent.run_managed(plan, consumer_args=self.options)
        with self.registry.guard() as guard:
            self.assertFalse(guard.conflicts([library])["conflicts"])

    def test_interruption_during_initial_registration_prevents_dispatch_and_closes(self):
        plan, _, _, _, _, _, _, _ = self._plan()

        @contextmanager
        def interrupted_scope():
            yield {"signal": signal.SIGTERM}

        with patch.object(managed_process, "supervisor_signal_scope", interrupted_scope):
            with patch.object(parent, "_run_managed") as dispatch:
                with self.assertRaises(InterruptedError):
                    parent.run_managed(plan, consumer_args=self.options)
                dispatch.assert_not_called()
        with self.registry.guard() as guard:
            self.assertFalse(guard.conflicts([plan])["conflicts"])

    def test_popen_exception_keeps_spawn_and_input_completion_unknown(self):
        binary = self.root / "engine"
        binary.write_bytes(b"synthetic")
        output = self.root / "output"
        output.mkdir()
        args = argparse.Namespace(
            seconds=5, cleanup_seconds=2, native_args=[
                "--resource-system-runtime-v3", "--custody-control", "--process-epoch=1",
                "--input-lock=lock", "--resource-plan=plan", "--resource-plan-sha256=hash",
                "--resource-plan-bytes=1"], epoch=1, output_dir=output,
            binary=binary, binary_bytes=9, binary_sha256="unused", numeric_observation=False,
            cwd=self.root)
        outcome = {"spawned": False}
        registration = argparse.Namespace(dispatched=False)

        def pinned(*_args):
            return os.open(binary, os.O_RDONLY), None

        with patch.object(parent, "_open_pinned_binary", side_effect=pinned):
            with patch.object(parent.subprocess, "Popen", side_effect=OSError("spawn outcome unknown")):
                with self.assertRaisesRegex(OSError, "spawn outcome unknown"):
                    parent.run(args, outcome=outcome, input_registration=registration)
        self.assertIsNone(outcome["spawned"])
        self.assertTrue(registration.dispatched)

    def test_direct_lock_registers_actual_runtime_before_run(self):
        cwd = self.root / "cwd"
        cwd.mkdir()
        binary = self.root / "engine"
        binary.write_bytes(b"synthetic")
        library = self.root / "cuda-library"
        library.write_bytes(b"synthetic")
        requested = self.root / "cuda-soname"
        requested.symlink_to(library)
        lock = cwd / "lock.json"
        lock.write_text(json.dumps({"manifest": {"system_runtime_lock": {"runtime": {
            "kind": "system_cuda", "configuration": {"libraries": [{"requested_path": str(requested),
                                                                          "canonical_path": str(library)}]}}}}}),
            encoding="utf8")
        args = argparse.Namespace(**vars(self.options), binary=binary, cwd=cwd, output_dir=cwd,
                                  native_args=["--input-lock=lock.json"])

        def observed(_args, *, started, outcome, input_registration, interruption):
            self._protected(binary, cwd, lock, library)
            input_registration.dispatched = True
            outcome["group_gone"] = True
            return 0

        with patch.object(parent, "run", side_effect=observed):
            self.assertEqual(parent._run_direct_registered(args), 0)
        with self.registry.guard() as guard:
            matches = guard.conflicts([library])["conflicts"]
            self.assertTrue(matches)
            self.assertTrue(all(row["state"] == "active" for row in matches))


if __name__ == "__main__":
    unittest.main()
