"""R update/checkpoint integration contracts; no neural or optimizer dispatch."""
import copy
import importlib.abc
import sys
from types import SimpleNamespace
import unittest


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("R training contracts cannot dispatch a neural runtime")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_r_auxiliary as auxiliary
    from rz_pals_model import rovent_training_window as window
    from rz_pals_model import rovent_training_window_v2 as v2
    from rz_pals_model import rovent_nnuec_training_state as nnuec
    from rz_pals_model.rovent_generation import SeriesRecipe
    from rz_pals_model.rovent_tensor_training import _schedule_count
    from test_rovent_training import membership_fixture, recipe_fixture
    from test_rovent_training_window_v2 import spec_fixture
    from test_rovent_r_auxiliary import recipe_value as auxiliary_recipe, sample_value
finally:
    sys.meta_path.remove(_guard)


def fixture(*, known=True):
    recipe = recipe_fixture(membership_fixture())
    raw = recipe.to_dict()
    raw["parameter_ownership"]["nnuec"] = "nnuec"
    recipe = SeriesRecipe(raw)
    spec = spec_fixture(recipe)
    spec["nnuec_r_auxiliary"] = auxiliary_recipe()
    journal = window.initial_training_journal(recipe, spec, {"corpus_sha256": "a" * 64})
    rows = [sample_value(1, known=known), sample_value(2, known=known)]
    sampler = auxiliary.RAuxiliarySampler((0, 1), seed=19, pool_sha256="b" * 64)
    sampler.take(2)
    journal["r_auxiliary"]["pool_sha256"] = "b" * 64
    projection = auxiliary.update_projection(spec["nnuec_r_auxiliary"], rows, pc_valid_rows=1)
    step = int(known)
    projection.update({name: step for name in ("r_auxiliary_nnuec_optimizer_steps", "r_auxiliary_nnuec_scheduler_steps",
        "r_auxiliary_nnuec_moments_steps")})
    heads = {"policy": 1, "wdl": 0, "divergence": 0, "task": 0, "nnuec_wdl": 8}
    event = {"schema": "rz-rovent-recovered-learner-update/1", "window_id": spec["window_id"], "stage_id": "common",
        "role": "proposer", "completed_role_updates": 1, "profile_sha256": window.training_profile_digest(spec),
        "corpus_sha256": "a" * 64, "phase": "pc", "sample_origins": ["c" * 64] * 8,
        "sample_inputs": ["d" * 64] * 8, "head_counts": heads, "actual_model_state_sha256": "e" * 64,
        "original_completion": {"sha256": "f" * 64, "bytes": 10}, "r_auxiliary": projection}
    journal["updates"].append(event)
    journal["head_counts"] = heads
    auxiliary.complete_auxiliary_update(journal["r_auxiliary"], projection, sampler.state())
    owner = SimpleNamespace(recipe=recipe, stage_id="common", completed={"proposer": 1, "critic": 0, "validator": 0},
        inherited_input_sha256=["d" * 64], inherited_origin_sha256=["c" * 64], samplers=journal)
    return recipe, spec, owner


class RTrainingContractTests(unittest.TestCase):
    def setUp(self):
        sys.meta_path.insert(0, _guard)

    def tearDown(self):
        sys.meta_path.remove(_guard)

    def test_window_preserves_absent_null_enabled_and_only_normalizes_weight_bits(self):
        recipe, enabled, _owner = fixture()
        absent = {name: value for name, value in enabled.items() if name != "nnuec_r_auxiliary"}
        null = {**absent, "nnuec_r_auxiliary": None}
        for item in (absent, null, enabled):
            before = copy.deepcopy(item)
            v2.validate_window_spec(recipe, item)
            v2.training_profile_digest(item)
            self.assertEqual(item, before)
        self.assertNotEqual(v2.training_profile_digest(absent), v2.training_profile_digest(null))
        self.assertNotEqual(v2.training_profile_digest(null), v2.training_profile_digest(enabled))
        integer = copy.deepcopy(enabled)
        integer["nnuec_r_auxiliary"]["weight"] = 1
        floating = copy.deepcopy(integer)
        floating["nnuec_r_auxiliary"]["weight"] = 1.0
        self.assertEqual(v2.training_profile_digest(integer), v2.training_profile_digest(floating))

    def test_pc_head_evidence_survives_but_zero_r_targets_keep_all_nnuec_counts_zero(self):
        _recipe, spec, owner = fixture(known=False)
        checked = window._journal(owner, spec, {"corpus_sha256": "a" * 64})
        self.assertEqual(checked["head_counts"]["nnuec_wdl"], 8)
        self.assertEqual(checked["r_auxiliary"]["current_nnuec_updates"], 0)
        self.assertEqual(_schedule_count(owner, ["nnuec.board_weights"]), 0)
        state = nnuec.derive_nnuec_tensor_training_state(source_declaration={"kind": "untrained"},
            parameter_updates={"nnuec.board_weights": 0}, samplers=checked)
        self.assertTrue(nnuec.nnuec_is_unmodified(state))
        self.assertEqual(nnuec.nnuec_known_updates(state), 0)

    def test_valid_r_updates_drive_private_scheduler_and_tensor_derived_state(self):
        _recipe, spec, owner = fixture(known=True)
        checked = window._journal(owner, spec, {"corpus_sha256": "a" * 64})
        self.assertEqual(_schedule_count(owner, ["nnuec.board_weights"]), 1)
        self.assertEqual(_schedule_count(owner, ["public_encoder.record_field_embedding"]), 1)
        state = nnuec.derive_nnuec_tensor_training_state(source_declaration={"kind": "untrained"},
            parameter_updates={"nnuec.board_weights": 1}, samplers=checked)
        self.assertEqual(nnuec.nnuec_known_updates(state), 1)
        self.assertFalse(nnuec.nnuec_is_unmodified(state))
        with self.assertRaisesRegex(ValueError, "maximum parameter steps"):
            nnuec.derive_nnuec_tensor_training_state(source_declaration={"kind": "untrained"},
                parameter_updates={"nnuec.board_weights": 0}, samplers=checked)

    def test_projection_and_sampler_cannot_erase_occurrences_or_fake_zero_target_step(self):
        _recipe, spec, owner = fixture(known=False)
        for mutate in (
            lambda value: value["updates"][0]["r_auxiliary"].update(r_auxiliary_nnuec_optimizer_steps=1),
            lambda value: value["r_auxiliary"].update(current_consumed_rows=1),
            lambda value: value["r_auxiliary"]["sampler"].update(samples=1),
            lambda value: value["r_auxiliary"].update(retained_origin_sha256=[]),
        ):
            bad = copy.deepcopy(owner)
            mutate(bad.samplers)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                window._journal(bad, spec, {"corpus_sha256": "a" * 64})

    def test_fork_resets_cursor_and_current_steps_but_retains_original_r_ancestry(self):
        _recipe, _spec, owner = fixture(known=True)
        parent = owner.samplers["r_auxiliary"]
        before = copy.deepcopy(parent)
        target = {**auxiliary_recipe(), "manifest_file_sha256": "1" * 64, "index_file_sha256": "2" * 64}
        child = auxiliary.fork_auxiliary_journal(parent, target)
        self.assertEqual(parent, before)
        self.assertIsNone(child["sampler"])
        self.assertIsNone(child["pool_sha256"])
        self.assertEqual(child["current_nnuec_updates"], 0)
        self.assertEqual(child["ancestor_nnuec_updates"], 1)
        self.assertEqual(child["ancestor_origin_sha256"], parent["retained_origin_sha256"])
        auxiliary.validate_auxiliary_journal(child, [])

    def test_v_event_cannot_carry_auxiliary_consumption_or_nnuec_step(self):
        _recipe, _spec, owner = fixture()
        bad = copy.deepcopy(owner.samplers["updates"][0])
        bad["role"] = "validator"
        with self.assertRaisesRegex(ValueError, "V update cannot consume"):
            auxiliary.validate_auxiliary_journal(owner.samplers["r_auxiliary"], [bad])

    def test_checkpoint_projection_keeps_distinct_r_seals_and_does_not_issue_authority(self):
        _recipe, _spec, owner = fixture(known=True)
        result = auxiliary.checkpoint_projection(owner.samplers["r_auxiliary"], owner.samplers["updates"])
        self.assertEqual(result["current_consumed_rows"], 2)
        self.assertEqual(result["retained_input"]["count"], 1)
        self.assertEqual(result["retained_origin"]["count"], 2)
        self.assertFalse(result["execution_authorized"])
        self.assertFalse(result["pc_nnuec_evidence_optimized"])

    def test_recovered_retirement_preserves_compact_auxiliary_and_rejects_old_domain_alias(self):
        from rz_pals_model import rovent_historical_checkpoint as historical
        from test_rovent_historical_checkpoint import fixture as checkpoint_fixture, project
        _recipe, _spec, owner = fixture(known=True)
        receipt, observed, recovered = checkpoint_fixture()
        projection = auxiliary.checkpoint_projection(owner.samplers["r_auxiliary"], owner.samplers["updates"])
        receipt.update(schema=historical.RECOVERED_RECEIPT_DOMAIN, source_kind="historical_exact_pins",
            past_native_authority_reissued=False)
        receipt["checkpoint"].update(schema=historical.RECOVERED_CHECKPOINT_DOMAIN, r_auxiliary=copy.deepcopy(projection))
        observed["r_auxiliary"] = copy.deepcopy(projection)
        checked = project(receipt, observed, recovered)
        self.assertEqual(checked["r_auxiliary"], projection)
        self.assertFalse(checked["execution_authorized"])
        observed["r_auxiliary"]["current_valid_rows"] += 1
        with self.assertRaises(ValueError):
            project(receipt, observed, recovered)
        receipt.update(schema=historical.RECEIPT_DOMAIN)
        receipt.pop("source_kind")
        receipt.pop("past_native_authority_reissued")
        receipt["checkpoint"]["schema"] = historical.CHECKPOINT_DOMAIN
        with self.assertRaisesRegex(ValueError, "legacy checkpoint retirement"):
            project(receipt, observed, recovered)


if __name__ == "__main__":
    unittest.main()
