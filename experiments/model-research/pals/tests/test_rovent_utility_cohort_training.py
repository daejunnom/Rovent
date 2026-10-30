"""Distinct cohort supervision preparation; these are metadata fixtures only."""
import copy
import hashlib
import json
import unittest

from rz_pals_model import rovent_utility_training as training
from test_rovent_utility_training import membership, registration


COHORT_SCOPE = "cohort_route_with_bounded_materialization_and_physical_pin_zero_excluding_durable_dataset_io_v1"


def fixture():
    spec = membership()
    source = registration(spec)
    source["cost_scope"] = COHORT_SCOPE
    recipe = training.make_recipe(spec["optimizer_owner"])
    return recipe, source, spec


def raw_digest(domain, value):
    raw = json.dumps([domain, value], sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class UtilityCohortPreparationTests(unittest.TestCase):
    def test_new_outer_domain_keeps_source_eight_and_numerical_recipe_membership_meanings(self):
        recipe, source, spec = fixture()
        prepared = training.prepare_cohort_training(recipe, source, spec)
        self.assertEqual(prepared["schema"], "rz-rovent-utility-training-preparation/2")
        self.assertEqual(prepared["registration"]["cost_scope"], COHORT_SCOPE)
        self.assertEqual(set(prepared["registration"]), {"collection_source_sha256", "trainer_source_sha256",
            "cost_epoch_sha256", "resource_profile_sha256", "cost_scope", "objective", "frozen_base", "utility"})
        legacy_source = copy.deepcopy(source)
        legacy_source["cost_scope"] = "exclusive_submitted_global_target_route_and_descendants_with_required_drain"
        legacy = training.prepare_training(recipe, legacy_source, spec)
        self.assertEqual(prepared["recipe_sha256"], legacy["recipe_sha256"])
        self.assertEqual(prepared["parameter_spec_sha256"], legacy["parameter_spec_sha256"])
        self.assertNotEqual(prepared["registration_sha256"], legacy["registration_sha256"])
        self.assertNotEqual(prepared["preparation_sha256"], legacy["preparation_sha256"])

    def test_cohort_registration_and_whole_preparation_have_explicit_v2_framing(self):
        recipe, source, spec = fixture()
        prepared = training.prepare_cohort_training(recipe, source, spec)
        self.assertEqual(prepared["registration_sha256"], raw_digest(
            "rz-rovent-utility-training-preparation/2", {"kind": "source_registration", "value": source}))
        original = {name: value for name, value in prepared.items() if name != "preparation_sha256"}
        self.assertEqual(prepared["preparation_sha256"], raw_digest("rz-rovent-utility-training-preparation/2", original))
        self.assertEqual(training.validate_training_preparation(prepared), prepared)

    def test_original_v1_parser_and_domain_remain_explicit_and_readable(self):
        spec = membership(); source = registration(spec)
        recipe = training.make_recipe(spec["optimizer_owner"])
        prepared = training.prepare_training(recipe, source, spec)
        self.assertEqual(prepared["schema"], "rz-rovent-utility-training-preparation/1")
        self.assertEqual(prepared["registration_sha256"], raw_digest(
            "rz-rovent-utility-training-preparation/1", {"kind": "source_registration", "value": source}))
        self.assertEqual(training.validate_training_preparation(prepared), prepared)
        with self.assertRaises(ValueError):
            training.prepare_cohort_training(recipe, source, spec)

    def test_old_cost_is_never_accepted_as_cohort_or_new_cost_as_root_local(self):
        recipe, source, spec = fixture()
        with self.assertRaises(ValueError):
            training.prepare_training(recipe, source, spec)
        for scope in ("root_wall_interval", "cohort_route", "exclusive_submitted_global_target_route_and_descendants_with_required_drain"):
            changed = copy.deepcopy(source); changed["cost_scope"] = scope
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                training.prepare_cohort_training(recipe, changed, spec)

    def test_schema_retag_or_changed_source_without_new_seal_is_rejected_without_conversion(self):
        recipe, source, spec = fixture()
        prepared = training.prepare_cohort_training(recipe, source, spec)
        for mutate in (lambda value: value.update(schema="rz-rovent-utility-training-preparation/1"),
                       lambda value: value.update(schema="rz-rovent-utility-training-preparation/3"),
                       lambda value: value["registration"].update(cost_epoch_sha256="61"*32),
                       lambda value: value.update(execution_authorized=True),
                       lambda value: value.update(extra="ignored")):
            changed = copy.deepcopy(prepared); mutate(changed)
            before = copy.deepcopy(changed)
            with self.assertRaises(ValueError):
                training.validate_training_preparation(changed)
            self.assertEqual(changed, before)

    def test_preparation_and_caller_are_detached_and_issue_no_known_pair_or_execution_authority(self):
        recipe, source, spec = fixture()
        before = copy.deepcopy((recipe, source, spec))
        prepared = training.prepare_cohort_training(recipe, source, spec)
        for flag in ("tensor_membership_verified", "source_authority_verified", "optimizer_state_present",
                     "checkpoint_tensor_present", "export_tensor_present", "execution_authorized"):
            self.assertIs(prepared[flag], False)
        self.assertEqual(prepared["performed_optimizer_updates"], 0)
        self.assertNotIn("known_pairs", prepared)
        self.assertEqual((recipe, source, spec), before)
        prepared["registration"]["frozen_base"]["config"]["board_width"] = 3
        self.assertEqual((recipe, source, spec), before)

    def test_source_cost_and_frozen_identity_pins_are_not_inferred_from_each_other(self):
        recipe, source, spec = fixture()
        original = training.prepare_cohort_training(recipe, source, spec)
        for name in ("collection_source_sha256", "trainer_source_sha256", "cost_epoch_sha256", "resource_profile_sha256"):
            changed = copy.deepcopy(source); changed[name] = "71"*32
            with self.subTest(field=name):
                current = training.prepare_cohort_training(recipe, changed, spec)
                self.assertNotEqual(current["preparation_sha256"], original["preparation_sha256"])
        missing = copy.deepcopy(source); missing.pop("cost_epoch_sha256")
        with self.assertRaises(ValueError):
            training.prepare_cohort_training(recipe, missing, spec)


if __name__ == "__main__":
    unittest.main()
