"""The recovery CLI forwards the same pinned scope as the build runner."""
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import storage_maintenance
from storage_consumer_registry import ConsumerRegistry


class RecoveryRegistryCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="rovezero-recovery-registry-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.registry = ConsumerRegistry.initialize(self.root / "registry", scope="fixture")

    def invoke(self, extra):
        args = ["storage_maintenance.py", "recover-build", "--root", str(self.root / "build"),
                "--owner-sha256", "a" * 64, *extra]
        env = {k: v for k, v in os.environ.items() if k not in
               ("RZ_CONSUMER_REGISTRY_ROOT", "RZ_CONSUMER_REGISTRY_ID", "RZ_CONSUMER_REGISTRY_REQUIRED")}
        with patch.dict(os.environ, env, clear=True), patch.object(sys, "argv", args), \
             patch.object(storage_maintenance, "recover_build", return_value={"ok": True}) as recover, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(storage_maintenance.main(), 0)
        return recover.call_args

    def test_pinned_registry_is_forwarded(self):
        call = self.invoke(["--consumer-registry-root", str(self.registry.root),
                            "--consumer-registry-id", self.registry.registry_id])
        self.assertEqual(call.kwargs["dependency_registry"].registry_id, self.registry.registry_id)
        self.assertTrue(call.kwargs["dependency_registry_required"])

    def test_missing_required_registry_remains_required(self):
        call = self.invoke(["--consumer-registry-required"])
        self.assertIsNone(call.kwargs["dependency_registry"])
        self.assertTrue(call.kwargs["dependency_registry_required"])

    def test_legacy_call_has_no_new_scope_arguments(self):
        self.assertEqual(self.invoke([]).kwargs, {})


if __name__ == "__main__":
    unittest.main()
