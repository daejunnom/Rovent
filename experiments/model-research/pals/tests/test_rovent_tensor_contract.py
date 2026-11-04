"""Neural-blocked admission checks, not actual tensor or learning evidence."""
import importlib.abc
import sys
import unittest

from test_rovent_training import NeuralImportsForbidden, membership_fixture, membership_v2_fixture, recipe_fixture


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_tensor_training as tensors
finally:
    sys.meta_path.remove(_guard)


class RoventTensorAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_all_materializing_boundaries_require_explicit_dispatch_before_import_or_io(self):
        calls = (
            lambda: tensors.create_tensor_owner(None, None, None, stage_id=None, samplers=None,
                bucket=None, sampler_validator=None),
            lambda: tensors.tensor_payload(None, sampler_validator=None),
            lambda: tensors.save_tensor_checkpoint(None, None, sampler_validator=None, max_file_bytes=0),
            lambda: tensors.stage_tensor_resume(None, None, None, None, None, sampler_validator=None,
                max_file_bytes=0, expected_stage_id=None),
            lambda: tensors.stage_tensor_fork(None, None, None, None, None, source_sampler_validator=None,
                target_sampler_validator=None, target_stage_id=None, reset_seed=None, reset_samplers=None,
                reset_bucket=None, max_file_bytes=0),
            lambda: tensors.TensorOwnerSlot(None).publish(None, sampler_validator=None),
            lambda: tensors.export_tensor_owner(None, None, kind=None, rules_profile_json=None,
                initialization_declaration=None, sampler_validator=None, max_output_bytes=0),
        )
        for call in calls:
            with self.subTest(call=call), self.assertRaisesRegex(ValueError, "explicit tensor lifecycle execution"):
                call()

    def test_plain_sidecars_reject_objects_nonfinite_and_excessive_nesting(self):
        tensors._validate_json_state({"epoch": 0, "cursor": [0, 1], "seed": 31})
        for bad in ({"cursor": object()}, {"loss": float("nan")}, {"loss": float("inf")}, {1: "cursor"}):
            with self.subTest(bad=type(bad)), self.assertRaisesRegex(ValueError, "sidecar"):
                tensors._validate_json_state(bad)
        nested = []
        for _ in range(66):
            nested = [nested]
        with self.assertRaisesRegex(ValueError, "bounded plain checkpoint sidecar"):
            tensors._validate_json_state(nested)

    def test_prototype_membership_is_rejected_before_any_runtime_import_or_caller_mutation(self):
        from types import SimpleNamespace
        membership = membership_v2_fixture(
            semantics="board800-exactb192-causal-line-kind4-directedpair-exactc256-fp32-v2")
        owner = SimpleNamespace(membership=membership)
        calls = (
            lambda: tensors.create_tensor_owner(None, None, membership, stage_id=None, samplers=None,
                bucket=None, sampler_validator=None, execute=True),
            lambda: tensors.tensor_payload(owner, sampler_validator=None, execute=True),
            lambda: tensors.stage_tensor_resume(None, None, None, None, membership, sampler_validator=None,
                max_file_bytes=0, expected_stage_id=None, execute=True),
            lambda: tensors.stage_tensor_fork(None, None, None, None, membership, source_sampler_validator=None,
                target_sampler_validator=None, target_stage_id=None, reset_seed=None, reset_samplers=None,
                reset_bucket=None, max_file_bytes=0, execute=True),
            lambda: tensors.export_tensor_owner(owner, None, kind=None, rules_profile_json=None,
                sampler_validator=None, max_output_bytes=0, execute=True),
        )
        for call in calls:
            with self.subTest(call=call), self.assertRaisesRegex(ValueError, "prototype membership.*originfirst"):
                call()

    def test_ledger_preserves_ancestral_cost_and_rejects_uncertain_accumulation_or_early_v(self):
        membership = membership_fixture()
        recipe = recipe_fixture(membership)
        owner = tensors.OwnedTensorState(None, recipe, membership, "common", {}, {}, {}, {}, {}, 8, ())
        owner.ancestor_cost["samples"] = 8
        owner.cost["samples"] = 3
        self.assertEqual(owner.total_cost()["samples"], 11)
        self.assertEqual(tensors._ledger(owner)["ancestor_cost"]["samples"], 8)
        owner.pending_microbatches = 1
        with self.assertRaisesRegex(ValueError, "completed accumulation boundary"):
            tensors._ledger(owner)
        owner.pending_microbatches = 0
        owner.active_role = "validator"
        with self.assertRaisesRegex(ValueError, "declared stage freeze boundary"):
            tensors._ledger(owner)
        owner.active_role = "proposer"
        owner.dispatched["proposer"] = 0
        owner.completed["proposer"] = 1
        with self.assertRaisesRegex(ValueError, "completed updates exceed dispatches"):
            tensors._ledger(owner)


if __name__ == "__main__":
    unittest.main()
