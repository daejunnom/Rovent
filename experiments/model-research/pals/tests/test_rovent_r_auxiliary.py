"""R auxiliary recipe/cursor fixtures; no optimizer or neural dispatch."""
import copy
import importlib.abc
import hashlib
import json
import struct
import sys
import unittest


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("R auxiliary preparation cannot import a neural framework")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_r_auxiliary as auxiliary
    from test_rovent_training import membership_fixture, recipe_fixture
finally:
    sys.meta_path.remove(_guard)


def recipe_value():
    return {"schema": auxiliary.RECIPE_DOMAIN, "manifest_file_sha256": "a" * 64, "index_file_sha256": "b" * 64,
        "sampling_seed": 19, "rows_per_pc_update": 2, "weight": 0.25,
        "normalization": "valid_exact_trajectory_rows", "optimizer_owner": "nnuec",
        "supervision_source": "r_auxiliary_only_preserve_pc_evidence"}


def sample_value(sequence=1, *, known=True):
    word = format(sequence, "064x")
    return {"input_sha256": "c" * 64, "origin_sha256": word, "record_sha256": format(sequence + 100, "064x"),
        "nnuec_input": {"schema": "fixture-only-numeric-not-a-rules-proof"}, "nnuec_model_sha256": "d" * 64,
        "nnuec_weights_file_sha256": "e" * 64, "nnuec_semantics": "fixture-only", "split": "train",
        "retirement_sha256": "f" * 64, "nnuec_wdl": [1.0, 0.0, 0.0] if known else None,
        "wdl_raw_sha256": "9" * 64 if known else None}


class RAuxiliaryPreparationTests(unittest.TestCase):
    def setUp(self):
        sys.meta_path.insert(0, _guard)

    def tearDown(self):
        sys.meta_path.remove(_guard)

    def test_explicit_weight_normalization_and_exclusive_nnuec_owner_are_required(self):
        value = recipe_value()
        before = copy.deepcopy(value)
        auxiliary.validate_auxiliary_recipe(value)
        self.assertEqual(value, before)
        self.assertIsNone(auxiliary.validate_auxiliary_recipe(None))
        for change in ({"weight": float("nan")}, {"weight": 10**1000}, {"weight": 0},
                       {"normalization": "all_rows"}, {"rows_per_pc_update": 0}, {"rows_per_pc_update": 65}):
            with self.subTest(change=tuple(change)), self.assertRaises(ValueError):
                auxiliary.validate_auxiliary_recipe({**value, **change})
        recipe = recipe_fixture(membership_fixture())
        with self.assertRaisesRegex(ValueError, "exclusive NNUEC"):
            auxiliary.validate_auxiliary_recipe(value, recipe=recipe)

    def test_r_rows_never_create_pc_updates_and_repeated_inputs_do_not_inflate_unique_count(self):
        rows = [sample_value(1), sample_value(2)]
        with self.assertRaisesRegex(ValueError, "without a valid P/C target"):
            auxiliary.update_projection(recipe_value(), rows, pc_valid_rows=0)
        result = auxiliary.update_projection(recipe_value(), rows, pc_valid_rows=3)
        self.assertEqual(result["valid_rows"], 2)
        self.assertEqual(result["unique_valid_inputs"], 1)
        self.assertEqual(len(result["occurrences"]), 2)
        self.assertFalse(result["execution_authorized"])
        self.assertNotIn("role", result)
        self.assertNotIn("completed_updates", result)

    def test_all_unknown_r_targets_stay_zero_and_cannot_claim_cp_or_mate_truth(self):
        rows = [sample_value(1, known=False), sample_value(2, known=False)]
        result = auxiliary.update_projection(recipe_value(), rows, pc_valid_rows=1)
        self.assertEqual(result["valid_rows"], 0)
        self.assertEqual(result["unique_valid_inputs"], 0)
        for invalid in ([0.7, 0.2, 0.1], {"cp": 30}, [1, 0, 0, 0]):
            with self.subTest(target=invalid), self.assertRaisesRegex(ValueError, "actual one-hot Rules"):
                auxiliary.validate_auxiliary_sample({**rows[0], "nnuec_wdl": invalid})
        with self.assertRaises(ValueError):
            auxiliary.validate_auxiliary_sample({**sample_value(), "split": "holdout"})

    def test_sampler_resume_keeps_the_exact_r_source_cursor_and_does_not_use_pc_roles(self):
        rows = tuple(range(7))
        continuous = auxiliary.RAuxiliarySampler(rows, seed=23, pool_sha256="a" * 64)
        first = continuous.take(8)
        state = continuous.state()
        resumed = auxiliary.RAuxiliarySampler(rows, seed=23, pool_sha256="a" * 64, state=state)
        expected = continuous.take(8)
        self.assertEqual(resumed.take(8), expected)
        self.assertEqual(resumed.state(), continuous.state())
        self.assertEqual(len(first), 8)
        self.assertNotIn("role", state)
        for change in ({"pool_sha256": "b" * 64}, {"cursor": state["cursor"] + 1},
                       {"permutation": [False] + state["permutation"][1:]}):
            with self.subTest(change=tuple(change)), self.assertRaises(ValueError):
                auxiliary.RAuxiliarySampler(rows, seed=23, pool_sha256="a" * 64, state={**state, **change})

    def test_pool_seal_preserves_actual_occurrences_and_refuses_duplicate_origin(self):
        rows = [sample_value(1), sample_value(2)]
        identity = auxiliary.auxiliary_pool_digest("b" * 64, rows)
        self.assertNotEqual(identity, auxiliary.auxiliary_pool_digest("b" * 64, list(reversed(rows))))
        with self.assertRaisesRegex(ValueError, "duplicate R occurrence"):
            auxiliary.auxiliary_pool_digest("b" * 64, [rows[0], copy.deepcopy(rows[0])])

    def test_recipe_seal_uses_explicit_f64_bits_and_preserves_raw_recipe(self):
        value = {**recipe_value(), "weight": 1e-8}
        before = copy.deepcopy(value)
        projection = {"algorithm": "coefficient-f64-bits-v1",
            "recipe": {**value, "weight": struct.pack(">d", 1e-8).hex()}}
        raw = json.dumps(["rz-rovent-r-auxiliary-recipe-seal/1", projection],
            sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.assertEqual(auxiliary.auxiliary_recipe_digest(value), hashlib.sha256(raw).hexdigest())
        self.assertEqual(value, before)
        self.assertEqual(auxiliary.auxiliary_recipe_digest({**value, "weight": 1}),
            auxiliary.auxiliary_recipe_digest({**value, "weight": 1.0}))


if __name__ == "__main__":
    unittest.main()
