"""Isolated entry metadata/receipt checks only; no live trainer or Torch."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from test_rovent_training import NeuralImportsForbidden, membership_fixture, recipe_fixture
from test_rovent_training_window import spec_fixture

_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_training_child as child
finally:
    sys.meta_path.remove(_guard)


class OriginalRoventChildContractTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_original_stdin_is_exactly_bounded_and_requires_eof(self):
        self.assertEqual(child._read_startup(io.BytesIO(b"{}"), 2), b"{}")
        for payload, size in ((b"{}x", 2), (b"{", 2), (b"", 0), (b"{}", True)):
            with self.subTest(payload=payload, size=size), self.assertRaisesRegex(ValueError, "startup"):
                child._read_startup(io.BytesIO(payload), size)

    def test_no_execute_is_rejected_before_stdin_or_tensor_import(self):
        with self.assertRaisesRegex(ValueError, "explicitly authorize"):
            child.original_rovent_child_main(["--startup-sha256", "1" * 64, "--startup-bytes", "2",
                "--execution-until-ns", "100", "--whole-until-ns", "200"], stdin=io.BytesIO(b"{}"))

    def test_pinned_metadata_keeps_registered_thread_recipe_and_rejects_foreign_runtime(self):
        from rz_pals_model.artifacts import VIRTUAL_RULES_BOUNDS, VIRTUAL_RULES_FEATURES
        from rz_pals_model.config import ModelConfig
        from rz_pals_model.virtual_review import RECORD_FEATURES, QUERY_FEATURES
        membership = membership_fixture()
        recipe = recipe_fixture(membership)
        spec = spec_fixture(recipe)
        profile, base = "rz-pals-rules-virtual-review-v1", "fixture_contract_not_Rules_execution"
        metadata, divergence = ["metadata" + str(i) for i in range(16)], ["divergence" + str(i) for i in range(8)]
        semantic_fields = [profile, base, *metadata, *RECORD_FEATURES, *QUERY_FEATURES, *divergence, *VIRTUAL_RULES_FEATURES]
        semantic = hashlib.sha256()
        for field in semantic_fields:
            raw = field.encode("utf-8")
            semantic.update(len(raw).to_bytes(8, "little")); semantic.update(raw)
        rules = {"schema": "rovezero.pals-rules-descriptor.v3", "rules_input_profile": profile,
            "config": ModelConfig.for_profile("virtual_review_v1").to_dict(), "rules_base_semantics": base,
            "metadata_features": metadata, "divergence_features": divergence,
            "record_features": list(RECORD_FEATURES), "query_features": list(QUERY_FEATURES),
            "virtual_review_features": VIRTUAL_RULES_FEATURES, "bounds": VIRTUAL_RULES_BOUNDS,
            "semantic_fields": semantic_fields, "semantic_digest_algorithm": "sha256_u64le_length_prefixed_utf8_fields",
            "rules_input_semantic_sha256": semantic.hexdigest(), "rules_encoder_source_sha256": "7" * 64,
            "learned_input_compatibility": "unverified_declaration_only"}
        payloads = {"recipe": recipe.to_dict(), "membership": membership, "window": spec, "rules": rules}
        class Reads:
            def json(self, pin, *, maximum):
                return copy.deepcopy(payloads[pin])
        counts = dict.fromkeys(("proposer", "critic", "validator"), 0)
        startup = {"window_id": spec["window_id"], "registration": {"recipe": "recipe", "membership": "membership",
            "window_spec": "window", "rules_profile": "rules", "python_runtime_sha256": spec["runtime_sha256"],
            "model_semantics_sha256": spec["model_semantics_sha256"], "resources_policy": {"device": "cpu"},
            "expected_initial_completed": counts, "expected_ancestor_completed": counts}}
        checked = child._metadata_inputs(startup, Reads())
        self.assertEqual(checked[2]["cpu_threads"], 2)
        startup["registration"]["python_runtime_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "window/runtime/model/resource"):
            child._metadata_inputs(startup, Reads())

    def test_missing_imported_torch_pin_fails_before_any_endpoint_file_is_read(self):
        class Reads:
            def verify(self, *_args, **_kwargs):
                raise AssertionError("unpinned module must fail before file access")
        fixture_module = SimpleNamespace(__file__="/explicit/not-a-runtime/torch.py", _C=None)
        with self.assertRaisesRegex(ValueError, "runtime pin missing for imported endpoint"):
            child._verify_imported_runtime({"runtime_module_pins": []}, Reads(), fixture_module)

    def test_bounded_failure_chain_preserves_primary_type_without_copying_message(self):
        original = ValueError("fixture private message is not a report field")
        cleanup = RuntimeError("fixture cleanup message is not a report field")
        cleanup.__cause__ = original
        original.__context__ = cleanup
        self.assertEqual(child._error_chain_types(cleanup), ["RuntimeError", "ValueError"])

    @unittest.skipUnless(os.name == "posix" and hasattr(os, "O_NOFOLLOW"), "original child receipt writer requires Linux nofollow descriptors")
    def test_terminal_is_new_file_and_byte_pin_without_replacing_prior_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory).resolve() / "terminal.json"
            payload = {"schema": child.RESULT_DOMAIN, "status": "failed", "error_type": "FixtureFailure"}
            with self.assertRaisesRegex(ValueError, "remaining output credit"):
                child._write_terminal(target, payload, 1)
            self.assertFalse(target.exists())
            pin = child._write_terminal(target, payload, 4096)
            raw = target.read_bytes()
            self.assertEqual(pin["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(pin["bytes"], len(raw))
            self.assertEqual(json.loads(raw), payload)
            with self.assertRaises(FileExistsError):
                child._write_terminal(target, {"status": "changed"}, 4096)
            self.assertEqual(target.read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
