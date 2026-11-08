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
import pals_followup_run as runner
from storage_consumer_registry import ConsumerRegistry


class FollowupConsumerTests(unittest.TestCase):
    def test_cuda_config_library_is_registered_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = ConsumerRegistry.initialize(root / "registry", scope="test")
            assets = root / "assets"
            assets.mkdir()
            asset = assets / "asset"
            asset.write_bytes(b"input")
            libraries = root / "libraries"
            libraries.mkdir()
            library = libraries / "libcudnn.so"
            library.write_bytes(b"test-library")
            config = assets / "cuda.json"
            raw = json.dumps({"libraries": [{"path": str(library)}]}).encode()
            config.write_bytes(raw)
            args = argparse.Namespace(consumer_registry_root=reg.root, consumer_registry_id=reg.registry_id,
                                      consumer_registry_required=True, mode="model-check", run=False, provider="cuda-system",
                                      binary=asset, export=asset, fixtures=asset, runtime=asset,
                                      external_gc_ledger=None, runtime_cache=None, cuda_bundle=None,
                                      system_cuda_config=config, system_cuda_config_sha256=hashlib.sha256(raw).hexdigest(),
                                      control_inventory=None)
            def ready(_args, _parser):
                with reg.guard() as guard:
                    self.assertTrue(guard.conflicts([library])["conflicts"])
                return 0
            env = {key: value for key, value in os.environ.items() if not key.startswith("RZ_CONSUMER_REGISTRY_")}
            with patch.dict(os.environ, env, clear=True), patch.object(runner, "_main_after_parse", ready):
                self.assertEqual(runner.run_with_registered_inputs(args, argparse.ArgumentParser()), 0)
            with reg.guard() as guard:
                self.assertFalse(guard.conflicts([library])["conflicts"])

    def test_registered_before_input_read_and_planning_closes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = ConsumerRegistry.initialize(root / "registry", scope="test")
            paths = [root / name for name in ("binary", "export", "fixtures", "runtime")]
            for path in paths:
                path.write_bytes(b"input")
            graph = root / "graph.onnx"
            graph.write_bytes(b"graph")
            cache = root / "new-cache"
            args = argparse.Namespace(consumer_registry_root=reg.root, consumer_registry_id=reg.registry_id,
                                      consumer_registry_required=True, mode="model-check", run=False, provider="cpu",
                                      binary=paths[0], export=paths[1], fixtures=paths[2], runtime=paths[3],
                                      external_gc_ledger=None, runtime_cache=cache, cuda_bundle=None,
                                      system_cuda_config=None, control_inventory=None)

            def read_after_registration(_args, _parser):
                with reg.guard() as guard:
                    for path in paths:
                        self.assertTrue(guard.conflicts([path])["conflicts"])
                    self.assertTrue(guard.conflicts([graph])["conflicts"])
                    self.assertTrue(guard.conflicts([cache])["conflicts"])
                return 0

            env = {key: value for key, value in os.environ.items() if not key.startswith("RZ_CONSUMER_REGISTRY_")}
            with patch.dict(os.environ, env, clear=True), patch.object(runner, "_main_after_parse", read_after_registration):
                self.assertEqual(runner.run_with_registered_inputs(args, argparse.ArgumentParser()), 0)
            with reg.guard() as guard:
                self.assertFalse(guard.conflicts(paths)["conflicts"])

    def test_cuda_completion_remains_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = ConsumerRegistry.initialize(root / "registry", scope="test")
            asset = root / "asset"
            asset.write_bytes(b"input")
            args = argparse.Namespace(consumer_registry_root=reg.root, consumer_registry_id=reg.registry_id,
                                      consumer_registry_required=True, mode="model-check", run=True, provider="cuda-system",
                                      binary=asset, export=asset, fixtures=asset, runtime=asset,
                                      external_gc_ledger=None, runtime_cache=root / "unused-cache", cuda_bundle=root / "unused-bundle",
                                      system_cuda_config=None, control_inventory=root / "unused-control")
            env = {key: value for key, value in os.environ.items() if not key.startswith("RZ_CONSUMER_REGISTRY_")}
            def dispatched(call_args, _parser):
                call_args._consumer_dispatch_state["started"] = True
                return 0
            with patch.dict(os.environ, env, clear=True), patch.object(runner, "_main_after_parse", dispatched), \
                    patch.object(runner, "prepare_consumer_output_catalog", return_value={"status": "test"}):
                self.assertEqual(runner.run_with_registered_inputs(args, argparse.ArgumentParser()), 0)
            with reg.guard() as guard:
                self.assertTrue(guard.conflicts([asset])["conflicts"])

    def test_predispatch_exception_closes_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = ConsumerRegistry.initialize(root / "registry", scope="test")
            asset = root / "asset"
            asset.write_bytes(b"input")
            args = argparse.Namespace(consumer_registry_root=reg.root, consumer_registry_id=reg.registry_id,
                                      consumer_registry_required=True, mode="model-check", run=True, provider="cpu",
                                      binary=asset, export=asset, fixtures=asset, runtime=asset,
                                      external_gc_ledger=None, runtime_cache=None, cuda_bundle=None,
                                      system_cuda_config=None, control_inventory=None)
            env = {key: value for key, value in os.environ.items() if not key.startswith("RZ_CONSUMER_REGISTRY_")}
            with patch.dict(os.environ, env, clear=True), patch.object(runner, "_main_after_parse", side_effect=ValueError("before dispatch")), \
                    patch.object(runner, "prepare_consumer_output_catalog", return_value={"status": "test"}):
                with self.assertRaisesRegex(ValueError, "before dispatch"):
                    runner.run_with_registered_inputs(args, argparse.ArgumentParser())
            with reg.guard() as guard:
                self.assertFalse(guard.conflicts([asset])["conflicts"])

    def test_dispatched_cpu_flag_alone_does_not_release_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = ConsumerRegistry.initialize(root / "registry", scope="test")
            asset = root / "asset"
            asset.write_bytes(b"input")
            args = argparse.Namespace(consumer_registry_root=reg.root, consumer_registry_id=reg.registry_id,
                                      consumer_registry_required=True, mode="model-check", run=True, provider="cpu",
                                      binary=asset, export=asset, fixtures=asset, runtime=asset,
                                      external_gc_ledger=None, runtime_cache=None, cuda_bundle=None,
                                      system_cuda_config=None, control_inventory=None)
            def dispatched(call_args, _parser):
                call_args._consumer_dispatch_state["started"] = True
                return 0
            env = {key: value for key, value in os.environ.items() if not key.startswith("RZ_CONSUMER_REGISTRY_")}
            with patch.dict(os.environ, env, clear=True), patch.object(runner, "_main_after_parse", dispatched), \
                    patch.object(runner, "prepare_consumer_output_catalog", return_value={"status": "test"}):
                self.assertEqual(runner.run_with_registered_inputs(args, argparse.ArgumentParser()), 0)
            with reg.guard() as guard:
                self.assertTrue(guard.conflicts([asset])["conflicts"])


if __name__ == "__main__":
    unittest.main()
