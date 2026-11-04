"""Small real-writer checks; importing a model or neural library is forbidden."""
import builtins
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


_original_import = builtins.__import__


def _without_neural_imports(name, *args, **kwargs):
    if name.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"} or name == "model" or name.endswith(".model"):
        raise AssertionError("write-budget checks must not import a model or neural library")
    return _original_import(name, *args, **kwargs)


with mock.patch("builtins.__import__", side_effect=_without_neural_imports):
    from rz_pals_model import artifacts, cli
    from rz_pals_model.onnx_virtual_review import _save


class ExportWriteBudgetTests(unittest.TestCase):
    def setUp(self):
        blocker = mock.patch("builtins.__import__", side_effect=_without_neural_imports)
        blocker.start()
        self.addCleanup(blocker.stop)
        temporary = tempfile.TemporaryDirectory(prefix="rz-export-write-budget-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def test_only_positive_integer_limits_and_nonnegative_integer_charges(self):
        for value in (None, True, False, 0, -1, 1.0, "1"):
            with self.subTest(limit=value), self.assertRaises(ValueError):
                artifacts.ExportWriteBudget(value)
        budget = artifacts.ExportWriteBudget(10)
        for value in (None, True, -1, 1.0, "1"):
            with self.subTest(charge=value), self.assertRaises(ValueError):
                budget(value)
        self.assertEqual(budget.reserved_bytes, 0)

    def test_exact_limit_is_shared_and_rejected_credit_is_not_refunded(self):
        budget = artifacts.ExportWriteBudget(10)
        budget(4)
        budget(6)
        budget(0)
        with self.assertRaisesRegex(ValueError, "before file creation"):
            budget(1)
        self.assertEqual((budget.maximum_bytes, budget.reserved_bytes), (10, 10))

    def test_metadata_rejection_precedes_both_partial_and_destination(self):
        value = {"input": "한국어", "unchanged": True}
        raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        path = self.root / "metadata.json"
        with self.assertRaises(ValueError):
            artifacts.atomic_json(path, value, output_guard=artifacts.ExportWriteBudget(len(raw) - 1))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_graph_rejection_precedes_partial_creation(self):
        class MetadataDocument:
            def SerializeToString(self):
                return b"source-only-serialized-byte-fixture"
        document = MetadataDocument()
        budget = artifacts.ExportWriteBudget(len(document.SerializeToString()) - 1)
        with mock.patch.object(artifacts, "output_directory", return_value=self.root):
            with self.assertRaises(ValueError):
                _save(document, {}, self.root / "mock.onnx", output_guard=budget)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_real_owner_marker_and_metadata_consume_one_budget(self):
        owner = {"schema": artifacts.SCHEMA, "owner": "rz-pals-model"}
        owner_raw = (json.dumps(owner, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
        budget = artifacts.ExportWriteBudget(len(owner_raw))
        self.assertEqual(artifacts.output_directory(self.root, output_guard=budget), self.root)
        self.assertEqual(budget.reserved_bytes, len(owner_raw))
        with self.assertRaises(ValueError):
            artifacts.atomic_json(self.root / "export.json", {}, output_guard=budget)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["pals-artifact-owner.json"])

    def test_partial_io_failure_keeps_original_bytes_and_full_charge(self):
        value = {"data": "unchanged"}
        raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
        budget = artifacts.ExportWriteBudget(len(raw))
        path = self.root / "failed.json"
        original_error = OSError("source-only serializer failure")
        def fail_after_partial(_value, stream, **_kwargs):
            stream.write("owned partial")
            raise original_error
        with mock.patch.object(artifacts.json, "dump", side_effect=fail_after_partial):
            with self.assertRaises(OSError) as failure:
                artifacts.atomic_json(path, value, output_guard=budget)
        self.assertIs(failure.exception, original_error)
        self.assertEqual(path.with_name("failed.json.partial").read_text(), "owned partial")
        self.assertFalse(path.exists())
        self.assertEqual(budget.reserved_bytes, len(raw))
        with self.assertRaises(ValueError):
            artifacts.atomic_json(self.root / "another.json", value, output_guard=budget)
        self.assertFalse((self.root / "another.json.partial").exists())

    def test_unguarded_metadata_writer_keeps_its_existing_output(self):
        path = self.root / "legacy.json"
        value = {"unchanged": True}
        artifacts.atomic_json(path, value)
        self.assertEqual(json.loads(path.read_text()), value)
        self.assertFalse(path.with_name("legacy.json.partial").exists())

    def test_cli_rejects_invalid_cap_or_layout_before_neural_import(self):
        base = ["export", "--checkpoint", "not-read.pt", "--output", str(self.root)]
        for tail in (["--max-export-bytes", "0"], ["--max-export-bytes", "-1"],
                     ["--max-export-bytes", "x"], ["--max-export-bytes", "1"]):
            with self.subTest(tail=tail), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    cli.main(base + tail)
                self.assertEqual(failure.exception.code, 2)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_export_help_remains_available_without_neural_import(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as result:
                cli.main(["export", "--help"])
        self.assertEqual(result.exception.code, 0)
        self.assertIn("--max-export-bytes", output.getvalue())

    def test_direct_api_rejects_uncovered_layout_or_noncallable_before_import(self):
        with self.assertRaises(TypeError):
            artifacts.export_checkpoint("not-read.pt", self.root, output_guard=1)
        for layout in ("separate_pc", "shared_pc_if"):
            with self.subTest(layout=layout), self.assertRaises(ValueError):
                artifacts.export_checkpoint("not-read.pt", self.root, layout=layout,
                                            output_guard=artifacts.ExportWriteBudget(1))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_positive_cli_passes_one_budget_through_actual_v3_writers(self):
        # Explicit in-memory stand-ins exercise dispatch, not a model or ONNX.
        # Every actual filesystem writer and both export dispatchers stay real.
        from rz_pals_model import onnx_virtual_review as virtual
        from rz_pals_model.config import ModelConfig
        config = ModelConfig.for_profile("virtual_review_v1")
        checkpoint = {"checkpoint_sha256": "a" * 64, "training_steps": 0, "trained": False}
        model = SimpleNamespace(config=config, experts={"validator": object()},
                                virtual_review_pc_export_view=lambda: object())
        torch = ModuleType("torch")
        torch.set_num_threads, torch.set_num_interop_threads = mock.Mock(), mock.Mock()
        torch.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                         cudnn=SimpleNamespace(allow_tf32=True))
        onnx = ModuleType("onnx")
        onnx.helper = SimpleNamespace(set_model_props=mock.Mock())
        onnx.checker = SimpleNamespace(check_model=mock.Mock())
        model_module = ModuleType("rz_pals_model.model")
        model_module.fixture_input = mock.Mock(side_effect=AssertionError("legacy model input must not run"))
        stand_ins = {"torch": torch, "onnx": onnx, "rz_pals_model.model": model_module}
        def synthetic_import(name, *args, **kwargs):
            key = "rz_pals_model.model" if name == "model" else name
            if key in stand_ins and sys.modules.get(key) is stand_ins[key]:
                return stand_ins[key]
            return _without_neural_imports(name, *args, **kwargs)
        class MetadataDocument:
            def __init__(self, raw):
                self.raw = raw
            def SerializeToString(self):
                return self.raw
        documents = {role: (MetadataDocument((role + "-source-only-byte-fixture").encode()), {})
                     for role in ("public", "shared_pc", "validator")}
        rules = {"rules_input_profile": "source-only-routing-fixture",
                 "rules_input_semantic_sha256": "b" * 64, "rules_encoder_source_sha256": "c" * 64}
        original_budget = artifacts.ExportWriteBudget
        created = []
        def budget_factory(maximum):
            value = original_budget(maximum)
            created.append(value)
            return value
        output = self.root / "export"
        with mock.patch.dict(sys.modules, stand_ins), \
                mock.patch("builtins.__import__", side_effect=synthetic_import), \
                mock.patch.object(artifacts, "ExportWriteBudget", side_effect=budget_factory), \
                mock.patch.object(artifacts, "load_checkpoint", return_value=(model, checkpoint)) as load, \
                mock.patch.object(artifacts, "read_rules_profile", return_value=rules), \
                mock.patch.object(artifacts, "validate_virtual_export_manifest", return_value=["proposer", "critic", "validator"]), \
                mock.patch.object(virtual, "build_virtual_public_graph", return_value=documents["public"]), \
                mock.patch.object(virtual, "build_virtual_shared_pc_graph", return_value=documents["shared_pc"]), \
                mock.patch.object(virtual, "build_virtual_validator_graph", return_value=documents["validator"]), \
                mock.patch.object(artifacts, "export_checkpoint", wraps=artifacts.export_checkpoint) as dispatch, \
                mock.patch.object(artifacts, "export_virtual_checkpoint", wraps=artifacts.export_virtual_checkpoint) as v3, \
                mock.patch.object(virtual, "_save", wraps=virtual._save) as graph_writes, \
                mock.patch.object(artifacts, "atomic_json", wraps=artifacts.atomic_json) as json_writes, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["export", "--checkpoint", "not-read.pt", "--output", str(output),
                "--layout", "virtual_review_shared_pc_if_v1", "--rules-profile-json", "not-read.json",
                "--include-validator", "--max-export-bytes", "65536"]), 0)
        self.assertEqual(len(created), 1)
        budget = created[0]
        load.assert_called_once_with("not-read.pt")
        self.assertIs(dispatch.call_args.kwargs["output_guard"], budget)
        self.assertIs(v3.call_args.kwargs["output_guard"], budget)
        self.assertEqual(graph_writes.call_count, 3)
        self.assertEqual(json_writes.call_count, 2)  # owner marker and manifest
        for call in graph_writes.call_args_list + json_writes.call_args_list:
            self.assertIs(call.kwargs["output_guard"], budget)
        self.assertEqual(budget.reserved_bytes, sum(path.stat().st_size for path in output.iterdir()))
        self.assertEqual(sorted(path.name for path in output.iterdir()),
            ["export.json", "pals-artifact-owner.json", "public_memory.onnx", "role_validator.onnx", "shared_pc_if.onnx"])
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)


if __name__ == "__main__":
    unittest.main()
