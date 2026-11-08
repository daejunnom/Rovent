"""Consumer registration at the real numeric wrapper boundary, without GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from storage_consumer_registry import ConsumerRegistry, bind_catalog
from storage_guard import CatalogGuard
from virtual_system_numeric_prepare import NumericInputRegistration


class NumericConsumerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.catalog = self.root / "outputs"
        self.catalog.mkdir()
        self.registry = ConsumerRegistry.initialize(self.root / "registry", scope="numeric-test")
        with CatalogGuard(self.catalog / "catalog.lock") as guard:
            bind_catalog(self.catalog, self.registry, guard)
        self.args = argparse.Namespace(consumer_registry_root=self.registry.root,
                                       consumer_registry_id=self.registry.registry_id,
                                       consumer_registry_required=True,
                                       consumer_input_catalog=[self.catalog])
        self.environment = patch.dict(os.environ, {"RZ_CONSUMER_REGISTRY_REQUIRED": "0"}, clear=True)
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temp.cleanup()

    def test_before_dispatch_closes_original_input_lease(self):
        source = self.root / "source"
        source.mkdir()
        owner = NumericInputRegistration(self.args, [source])
        with self.registry.guard() as guard:
            self.assertTrue(guard.conflicts([source / "file"])["conflicts"])
        owner.close_if_not_dispatched()
        with self.registry.guard() as guard:
            self.assertFalse(guard.conflicts([source / "file"])["conflicts"])

    def test_dispatched_unknown_remains_protected(self):
        source = self.root / "source"
        source.mkdir()
        owner = NumericInputRegistration(self.args, [source])
        owner.dispatched = True
        owner.close_if_not_dispatched()
        with self.registry.guard() as guard:
            self.assertTrue(guard.conflicts([source])["conflicts"])

    def test_descriptor_discovery_holds_registry_lock(self):
        source = self.root / "source"
        source.mkdir()
        library = self.root / "lib.so"
        library.write_bytes(b"library")
        owner = NumericInputRegistration(self.args, [source])
        def discover():
            with self.assertRaises(Exception):
                with self.registry.guard():
                    pass
            return "parsed", [library]
        self.assertEqual(owner.read_and_register(discover), "parsed")
        with self.registry.guard() as guard:
            self.assertTrue(guard.conflicts([library])["conflicts"])
        owner.close_if_not_dispatched()

    def test_mismatched_or_missing_catalog_cannot_claim_protection(self):
        source = self.root / "source"
        source.mkdir()
        self.args.consumer_input_catalog = []
        with self.assertRaisesRegex(Exception, "catalog"):
            NumericInputRegistration(self.args, [source])
        other = self.root / "unbound"
        other.mkdir()
        self.args.consumer_input_catalog = [other]
        with self.assertRaisesRegex(Exception, "binding"):
            NumericInputRegistration(self.args, [source])

    def test_supervisor_catalogs_are_rechecked_and_combined(self):
        source = self.root / "source"
        source.mkdir()
        self.args.consumer_input_catalog = []
        with patch.dict(os.environ, {"RZ_CONSUMER_INPUT_CATALOGS": json.dumps([str(self.catalog)])}):
            owner = NumericInputRegistration(self.args, [source])
            with self.registry.guard() as guard:
                self.assertTrue(guard.conflicts([source])["conflicts"])
            owner.close_if_not_dispatched()
        with patch.dict(os.environ, {"RZ_CONSUMER_INPUT_CATALOGS": json.dumps(["relative"])}):
            with self.assertRaisesRegex(Exception, "absolute"):
                NumericInputRegistration(self.args, [source])

    @unittest.skipUnless(sys.platform == "linux", "numeric supervisor uses Linux memfd")
    def test_real_run_registers_descriptor_paths_before_getter(self):
        import virtual_system_numeric_run as runner
        binary = self.root / "engine"
        binary.write_bytes(b"synthetic binary, never executed")
        plan = self.root / "plan.json"
        plan.write_bytes(b"{}")
        export_dir = self.root / "model"
        export_dir.mkdir()
        manifest = export_dir / "manifest.json"
        manifest.write_bytes(b"{}")
        core = self.root / "ort.so"
        core.write_bytes(b"runtime")
        cuda_library = self.root / "cuda.so"
        cuda_library.write_bytes(b"cuda")
        system = self.root / "system.json"
        system.write_text(json.dumps({"libraries": [{"path": str(cuda_library)}]}), encoding="utf8")
        config = self.root / "config.json"
        config.write_text(json.dumps({
            "provider": "cuda-system", "output_root": str(self.root / "out"),
            "case_plan": str(plan), "case_plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
            "export_manifest": str(manifest), "runtime_library": str(core),
            "system_cuda_config": str(system),
            "system_cuda_config_sha256": hashlib.sha256(system.read_bytes()).hexdigest(),
        }), encoding="utf8")

        def getter(*_args, **_kwargs):
            with self.registry.guard() as guard:
                for path in (binary, config, plan, manifest, core, system, cuda_library):
                    self.assertTrue(guard.conflicts([path])["conflicts"], str(path))
            raise RuntimeError("stop before synthetic getter dispatch")

        argv = ["--binary", str(binary), "--binary-sha256", hashlib.sha256(binary.read_bytes()).hexdigest(),
                "--binary-bytes", str(binary.stat().st_size), "--config", str(config),
                "--config-sha256", hashlib.sha256(config.read_bytes()).hexdigest(),
                "--receipt", str(self.root / "receipt.json"), "--timeout", "30",
                "--consumer-registry-root", str(self.registry.root),
                "--consumer-registry-id", self.registry.registry_id,
                "--consumer-input-catalog", str(self.catalog)]
        with patch.object(runner, "require_capture_window", return_value={}), \
                patch.object(runner, "observe_resources", return_value={}), \
                patch.object(runner, "metadata_getter", side_effect=getter):
            self.assertEqual(runner.main(argv), 1)
        # A dispatched-or-possibly-dispatched external binary must stay
        # protected; the synthetic exception is not a physical fence.
        with self.registry.guard() as guard:
            self.assertTrue(guard.conflicts([cuda_library])["conflicts"])


if __name__ == "__main__":
    unittest.main()
