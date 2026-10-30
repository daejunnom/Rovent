"""Neural-blocked window/continuation contracts; no native or learner run."""
import copy
import sys
import unittest

from test_rovent_training import NeuralImportsForbidden, membership_fixture, recipe_fixture

_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_training_window as window
    from rz_pals_model import rovent_tensor_training as tensors
    from rz_pals_model.generation_training import generation_spec, OPTIMIZER
finally:
    sys.meta_path.remove(_guard)


def spec_fixture(recipe):
    registered = recipe.to_dict()
    return {"schema": window.WINDOW_DOMAIN, "window_id": "window-one", "recipe_sha256": recipe.sha256,
        "stage_id": "common", "source_sha256": registered["source_sha256"], "runtime_sha256": "1" * 64,
        "model_semantics_sha256": registered["model_semantics_sha256"], "nnuec_collection_model_sha256": "2" * 64,
        "nnuec_semantics": "board800-exactb192-causal-line-exactc256-fp32-v1", "model_epoch": 0, "context_epoch": 0,
        "phase": "pc", "frozen_pc_sha256": None, "device": "cpu", "precision": "fp32", "tf32": False,
        "deterministic": True, "cpu_threads": 2, "interop_threads": 1, "microbatch": 1, "accumulation": 8,
        "optimizer": copy.deepcopy(OPTIMIZER), "sampling_seed": 31, "training_seed": 31,
        "loss_weights": {"policy": 1.0, "wdl": 1.0, "divergence": 1.0, "task": 1.0},
        "nnuec_loss": {"loss": "wdl_cross_entropy", "target": "same_exact_rules_anchor",
                       "normalization": "valid_rows", "weight": 0.5},
        "max_updates": 2, "max_groups": 8, "max_samples": 64, "max_wall_ns": 10**9,
        "max_output_bytes": 1024**2, "checkpoint_interval": 2}


class RoventWindowContractTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)
        self.membership = membership_fixture()
        self.recipe = recipe_fixture(self.membership)
        self.spec = spec_fixture(self.recipe)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_window_requires_its_own_complete_recipe_and_explicit_nnuec_learning_semantics(self):
        self.assertEqual(window.validate_window_spec(self.recipe, self.spec), self.spec)
        for name, value in (("max_wall_ns", 0), ("max_samples", 7), ("max_updates", True),
                            ("device", "cuda"), ("device", "cuda:01"), ("tf32", True)):
            changed = copy.deepcopy(self.spec)
            changed[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                window.validate_window_spec(self.recipe, changed)
        unresolved = copy.deepcopy(self.spec)
        unresolved["nnuec_loss"]["weight"] = None
        with self.assertRaisesRegex(ValueError, "explicitly resolved"):
            window.validate_window_spec(self.recipe, unresolved)
        legacy = generation_spec(generation_id="g0", source_sha256="1" * 64, dataset_sha256="2" * 64,
            split_sha256="3" * 64, model_sha256="4" * 64, runtime_sha256="5" * 64)
        with self.assertRaisesRegex(ValueError, "Rovent training window fields mismatch"):
            window.validate_window_spec(self.recipe, legacy)

    def test_window_renewal_preserves_fixed_learning_profile_and_fresh_v_requires_a_freeze(self):
        renewed = copy.deepcopy(self.spec)
        renewed.update(window_id="window-two", max_wall_ns=2 * 10**9, max_updates=1)
        self.assertEqual(window.training_profile_digest(renewed), window.training_profile_digest(self.spec))
        changed = copy.deepcopy(renewed)
        changed["nnuec_loss"]["weight"] = 0.75
        self.assertNotEqual(window.training_profile_digest(changed), window.training_profile_digest(self.spec))
        v = copy.deepcopy(self.spec)
        v.update(phase="v", model_epoch=1)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            window.validate_window_spec(self.recipe, v)
        v["frozen_pc_sha256"] = "4" * 64
        window.validate_window_spec(self.recipe, v)
        self.assertEqual(window.training_profile_digest(v), window.training_profile_digest(self.spec))

    def test_sampler_resume_retains_exact_order_epoch_cursor_and_consumed_samples(self):
        first = window.StageSampler(["a", "b", "c"], role="proposer", seed=31, pool_sha256="2" * 64)
        prefix = first.take(8)
        self.assertEqual(len(prefix), 8)
        saved = first.state()
        expected = first.take(8)
        resumed = window.StageSampler(["a", "b", "c"], role="proposer", seed=31, pool_sha256="2" * 64, state=saved)
        self.assertEqual(resumed.take(8), expected)
        self.assertEqual(resumed.state(), first.state())
        bad = copy.deepcopy(saved)
        bad["samples"] += 1
        with self.assertRaisesRegex(ValueError, "exact consumed position"):
            window.StageSampler(["a", "b", "c"], role="proposer", seed=31, pool_sha256="2" * 64, state=bad)
        with self.assertRaisesRegex(ValueError, "role/seed/pinned pool"):
            window.StageSampler(["a", "b", "c"], role="critic", seed=31, pool_sha256="2" * 64, state=saved)

    def test_head_counts_keep_wdl_rows_and_v_known_pairs_as_distinct_normalizers(self):
        def sample(**updates):
            result = {"targets": {"policy": None, "wdl": None, "divergence": None, "task": None}, "nnuec_wdl": None}
            result.update(updates)
            return result
        one = sample(targets={"policy": [1.0], "wdl": [1.0, 0.0, 0.0], "divergence": 1.0, "task": None}, nnuec_wdl=[1.0, 0.0, 0.0])
        two = sample(targets={"policy": None, "wdl": None, "divergence": None, "task": {"pairs": [(0, 1), (2, 3)]}})
        self.assertEqual(window._valid_counts([one, sample()], "critic"),
            {"policy": 1, "wdl": 1, "divergence": 1, "task": 0, "nnuec_wdl": 1})
        self.assertEqual(window._valid_counts([two], "validator"),
            {"policy": 0, "wdl": 0, "divergence": 0, "task": 2, "nnuec_wdl": 0})
        self.assertFalse(any(window._valid_counts([sample()], "proposer").values()))

    def test_serialized_admission_cannot_construct_an_original_training_authority(self):
        with self.assertRaises(TypeError):
            window.OriginalTrainerAuthority()
        owner = tensors.OwnedTensorState(None, self.recipe, self.membership, "common", {}, {}, {}, {}, {}, 8, ())
        with self.assertRaisesRegex(ValueError, "original teacher-aware native corpus owner"):
            window.train_rovent_window(owner, {"execution_authorized": True}, self.spec, "/not-used", execute=False)

    def test_fork_preparation_names_parent_and_never_materializes_tensor_bytes(self):
        spec = copy.deepcopy(self.spec)
        stage = next(stage for stage in self.recipe.to_dict()["stages"] if stage["parent_stage_id"] == "common")
        spec["stage_id"] = stage["stage_id"]
        prepared = window.stage_rovent_fork("/not-read/source.pt", "3" * 64, None, self.recipe, self.membership,
            None, spec, max_file_bytes=1024)
        self.assertEqual(prepared["source_stage_id"], "common")
        self.assertTrue(prepared["reset_optimizer_state"])
        self.assertFalse(prepared["actual_tensor_bytes_loaded"])
        self.assertFalse(prepared["execution_authorized"])
        with self.assertRaisesRegex(ValueError, "original parent checkpoint authority"):
            window.stage_rovent_fork("/not-read/source.pt", "3" * 64, None, self.recipe, self.membership,
                None, spec, max_file_bytes=1024, execute=True)
        with self.assertRaisesRegex(ValueError, "common initialization"):
            window.stage_rovent_fork("/not-read/source.pt", "3" * 64, None, self.recipe, self.membership,
                None, self.spec, max_file_bytes=1024)


if __name__ == "__main__":
    unittest.main()
