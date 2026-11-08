import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pals_followup_run as runner
from storage_consumer_registry import ConsumerRegistry


class FollowupConsumerTests(unittest.TestCase):
    def test_windows_file_attributes_must_be_observed_before_link_classification(self):
        for info in (SimpleNamespace(), SimpleNamespace(st_file_attributes=None),
                     SimpleNamespace(st_file_attributes="0"), SimpleNamespace(st_file_attributes=True)):
            with self.subTest(info=info), self.assertRaisesRegex(ValueError, "attributes unavailable"):
                runner._observed_windows_reparse(info, is_windows=True)
        self.assertFalse(runner._observed_windows_reparse(SimpleNamespace(st_file_attributes=0), is_windows=True))
        self.assertTrue(runner._observed_windows_reparse(SimpleNamespace(st_file_attributes=0x400), is_windows=True))
        self.assertFalse(runner._observed_windows_reparse(SimpleNamespace(), is_windows=False))

    def test_secret_like_library_spelling_is_rejected_before_any_lstat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("client_secret.json", "token.json", "private_key.bin",
                         "private-key.bin", "identity.pem", "identity.key", "identity.p12",
                         "identity.pfx", "passwords.kdbx"):
                with self.subTest(name=name), patch.object(Path, "lstat", side_effect=AssertionError("lstat forbidden")):
                    with self.assertRaisesRegex(ValueError, "secret-like"):
                        runner._observe_system_library(root / name, deadline=float("inf"))

    @unittest.skipUnless(os.name == "posix", "POSIX system-library symlink fixture")
    def test_secret_like_link_target_is_rejected_without_observing_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            link = root / "libtest.so"
            link.symlink_to("token.json")
            original_lstat = Path.lstat

            def forbid_secret_lstat(path, *args, **kwargs):
                if path.name == "token.json":
                    raise AssertionError("secret target observation forbidden")
                return original_lstat(path, *args, **kwargs)

            with patch.object(Path, "lstat", forbid_secret_lstat):
                with self.assertRaisesRegex(ValueError, "secret-like"):
                    runner._observe_system_library(link, deadline=float("inf"))

    def test_expired_library_deadline_is_checked_before_lstat(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(Path, "lstat", side_effect=AssertionError("lstat forbidden")):
            with self.assertRaises(TimeoutError):
                runner._observe_system_library(Path(tmp) / "libtest.so", deadline=0.0)

    @unittest.skipUnless(os.name == "posix", "POSIX system-library symlink fixture")
    def test_cuda_config_registers_each_link_and_its_canonical_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = ConsumerRegistry.initialize(root / "registry", scope="test")
            assets = root / "assets"
            assets.mkdir()
            asset = assets / "asset"
            asset.write_bytes(b"input")
            library_root = root / "library-root"
            library_root.mkdir()
            library = library_root / "libcudnn.so.9.1"
            library.write_bytes(b"test-library")
            soname = library_root / "libcudnn.so.9"
            soname.symlink_to(library.name)
            link_dir = root / "library-link"
            link_dir.symlink_to(library_root.name, target_is_directory=True)
            unrelated = root / "unrelated"
            unrelated.mkdir()
            requested = link_dir / soname.name
            config = assets / "cuda.json"
            raw = json.dumps({"libraries": [{"path": str(requested)}]}).encode()
            config.write_bytes(raw)
            args = argparse.Namespace(consumer_registry_root=reg.root, consumer_registry_id=reg.registry_id,
                                      consumer_registry_required=True, mode="model-check", run=False,
                                      provider="cuda-system", binary=asset, export=asset, fixtures=asset,
                                      runtime=asset, external_gc_ledger=None, runtime_cache=None,
                                      cuda_bundle=None, system_cuda_config=config,
                                      system_cuda_config_sha256=hashlib.sha256(raw).hexdigest(),
                                      control_inventory=None)

            def ready(_args, _parser):
                with reg.guard() as guard:
                    # Deleting either enclosing directory would destroy a
                    # recorded link; the distinct target is protected too.
                    self.assertTrue(guard.conflicts([library_root])["conflicts"])
                    self.assertTrue(guard.conflicts([library])["conflicts"])
                    self.assertFalse(guard.conflicts([unrelated])["conflicts"])
                rows = [json.loads(path.read_text(encoding="utf8"))
                        for path in reg.root.glob("consumer-*.json")]
                recorded = {(entry["kind"], entry["path"])
                            for row in rows for entry in row["inputs"]}
                self.assertIn(("symlink", str(link_dir)), recorded)
                self.assertIn(("symlink", str(soname)), recorded)
                self.assertIn(("file", str(library)), recorded)
                return 0

            env = {key: value for key, value in os.environ.items()
                   if not key.startswith("RZ_CONSUMER_REGISTRY_")}
            with patch.dict(os.environ, env, clear=True), patch.object(runner, "_main_after_parse", ready):
                self.assertEqual(runner.run_with_registered_inputs(args, argparse.ArgumentParser()), 0)
            with reg.guard() as guard:
                self.assertFalse(guard.conflicts([library])["conflicts"])

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
