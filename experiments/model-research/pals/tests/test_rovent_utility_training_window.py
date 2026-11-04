"""Pure utility window/sampler fixtures; these create no live authority."""
import copy
import hashlib
import json
import unittest

from rz_pals_model import rovent_utility_training as training
from rz_pals_model import rovent_utility_training_window as window


def preparation():
    config = {"board_width": 2, "line_width": 3, "context_width": 4, "local_width": 5,
              "hidden_width": 6, "causal_window": 3, "max_lines": 64, "max_plies": 256}
    spec = {"hidden_width": 2, "input_width": 17, "optimizer_owner": "utility-owner",
        "frozen_components": ["nnuec_base", "p", "c"],
        "wire_to_parameter": {name: "utility." + name for name in training.PARAMETER_FIELDS},
        "parameters": {name: {"checkpoint_path": "parameters." + name, "shape": shape,
            "dtype": "float32", "planned_requires_grad": True} for name, shape in
            zip(training.PARAMETER_FIELDS, ([2, 17], [2], [7, 2], [7]), strict=True)}}
    registration = {"collection_source_sha256": "10" * 32, "trainer_source_sha256": "11" * 32,
        "resource_profile_sha256": "40" * 32, "cost_epoch_sha256": "50" * 32,
        "cost_scope": "exclusive_submitted_global_target_route_and_descendants_with_required_drain",
        "objective": "exclusive_wall_nanoseconds",
        "frozen_base": {"schema": training.BASE_SCHEMA, "semantics": training.BASE_SEMANTICS,
            "weights_sha256": "20" * 32, "config": config},
        "utility": {"schema": training.UTILITY_SCHEMA, "semantics": training.UTILITY_SEMANTICS,
            "initial_weights_sha256": "30" * 32, "base_weights_sha256": "20" * 32,
            "parameter_spec_sha256": training.parameter_spec_sha256(spec, config, "utility-owner")}}
    return training.prepare_training(training.make_recipe("utility-owner"), registration, spec)


def spec(prepared):
    return {"schema": "rz-rovent-utility-train-window/1", "window_id": "utility-window", "run_id": "utility-run",
        "preparation_sha256": prepared["preparation_sha256"], "dataset_sha256": "60" * 32,
        "device": "cpu", "precision": "fp32", "tf32": False, "deterministic": True,
        "cpu_threads": 2, "interop_threads": 1, "training_seed": 31, "microbatch": 1, "accumulation": 8,
        "max_optimizer_updates": 4, "checkpoint_every_updates": 2, "max_wall_time_ns": 60000000000,
        "cleanup_time_ns": 30000000000, "max_read_bytes": 1048576, "max_output_bytes": 1048576,
        "max_groups": 16, "max_contexts": 128}


class UtilityWindowTests(unittest.TestCase):
    def test_finite_window_seals_the_separate_preparation_without_granting_permission(self):
        prepared = preparation(); value = spec(prepared)
        before = copy.deepcopy((prepared, value))
        self.assertEqual(window.validate_window_spec(prepared, value), value)
        self.assertEqual(len(window.utility_window_digest(value)), 64)
        self.assertEqual((prepared, value), before)
        self.assertFalse(prepared["execution_authorized"])
        value["max_optimizer_updates"] = 0
        self.assertEqual(window.validate_window_spec(prepared, value)["max_optimizer_updates"], 0)

    def test_changed_preparation_domain_or_authorization_flag_is_not_reissued(self):
        original = preparation()
        for mutate in (lambda value: value.update(execution_authorized=True),
                       lambda value: value.update(schema="rz-rovent-train-window/1"),
                       lambda value: value.update(performed_optimizer_updates=1)):
            changed = copy.deepcopy(original); mutate(changed)
            with self.assertRaises(ValueError): window.validate_window_spec(changed, spec(original))

    def test_explicit_v1_v2_preparations_are_checked_without_automatic_conversion(self):
        original = preparation()
        registration = copy.deepcopy(original["registration"])
        registration["cost_scope"] = "cohort_route_with_bounded_materialization_and_physical_pin_zero_excluding_durable_dataset_io_v1"
        cohort = training.prepare_cohort_training(original["recipe"], registration, original["parameter_spec"])
        self.assertEqual(cohort["schema"], "rz-rovent-utility-training-preparation/2")
        self.assertEqual(window.validate_window_spec(cohort, spec(cohort))["preparation_sha256"], cohort["preparation_sha256"])
        self.assertEqual(original["schema"], "rz-rovent-utility-training-preparation/1")
        with self.assertRaises(ValueError): window.validate_window_spec(original, spec(cohort))
        changed = copy.deepcopy(original); changed["schema"] = "rz-rovent-utility-training-preparation/2"
        with self.assertRaises(ValueError): window.validate_window_spec(changed, spec(cohort))

    def test_exact_window_fields_device_precision_integer_and_finite_credits_are_required(self):
        prepared = preparation()
        for mutate in (lambda value: value.update(schema="rz-rovent-train-window/2"),
                       lambda value: value.update(device="cuda:00"),
                       lambda value: value.update(tf32=0),
                       lambda value: value.update(deterministic=1),
                       lambda value: value.update(precision="fp16"),
                       lambda value: value.update(microbatch=True),
                       lambda value: value.update(accumulation=8.0),
                       lambda value: value.update(training_seed=2**32),
                       lambda value: value.update(max_groups=0),
                       lambda value: value.update(max_contexts=False),
                       lambda value: value.update(max_wall_time_ns=0),
                       lambda value: value.update(execution_authorized=True)):
            changed = spec(prepared); mutate(changed)
            with self.assertRaises(ValueError): window.validate_window_spec(prepared, changed)

    def test_window_v2_has_an_explicit_positive_reserve_without_extending_old_budgets(self):
        prepared = preparation(); old = spec(prepared)
        value = {**old, "schema": "rz-rovent-utility-train-window/2", "checkpoint_reserve_ns": 5000000000}
        before = copy.deepcopy(value)
        self.assertEqual(window.validate_utility_window_v2(prepared, value), before)
        with self.assertRaises(ValueError): window.validate_window_spec(prepared, value)
        for name in ("max_wall_time_ns", "cleanup_time_ns", "max_read_bytes", "max_output_bytes"):
            self.assertEqual(value[name], old[name])
        self.assertEqual(value, before)
        self.assertNotIn("execution_authorized", value)

    def test_window_v2_rejects_missing_noninteger_zero_and_full_window_reserve(self):
        prepared = preparation()
        original = {**spec(prepared), "schema": "rz-rovent-utility-train-window/2", "checkpoint_reserve_ns": 1}
        for mutate in (lambda value: value.pop("checkpoint_reserve_ns"),
                       lambda value: value.update(checkpoint_reserve_ns=True),
                       lambda value: value.update(checkpoint_reserve_ns=1.0),
                       lambda value: value.update(checkpoint_reserve_ns=0),
                       lambda value: value.update(checkpoint_reserve_ns=-1),
                       lambda value: value.update(checkpoint_reserve_ns=value["max_wall_time_ns"]),
                       lambda value: value.update(checkpoint_reserve_ns=value["max_wall_time_ns"] + 1),
                       lambda value: value.update(execution_authorized=True)):
            changed = copy.deepcopy(original); mutate(changed)
            with self.assertRaises(ValueError): window.validate_utility_window_v2(prepared, changed)
        boundary = {**original, "checkpoint_reserve_ns": original["max_wall_time_ns"] - 1}
        self.assertEqual(window.validate_utility_window_v2(prepared, boundary), boundary)

    def test_original_window_v1_bytes_and_parser_do_not_gain_a_reserve_or_auto_upgrade(self):
        prepared = preparation(); old = spec(prepared)
        independent = hashlib.sha256(json.dumps(["rz-rovent-utility-train-window/1", old],
            sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.assertEqual(window.utility_window_digest(old), independent)
        self.assertEqual(window.validate_window_spec(prepared, old), old)
        with self.assertRaises(ValueError): window.validate_utility_window_v2(prepared, old)
        old_with_reserve = {**old, "checkpoint_reserve_ns": 1}
        with self.assertRaises(ValueError): window.validate_window_spec(prepared, old_with_reserve)
        retagged = {**old, "schema": "rz-rovent-utility-train-window/2"}
        with self.assertRaises(ValueError): window.validate_utility_window_v2(prepared, retagged)
        with self.assertRaises(ValueError): window.utility_window_v2_digest(old)

    def test_new_reserve_is_sealed_with_window_v2_domain_and_not_window_v1_domain(self):
        prepared = preparation()
        value = {**spec(prepared), "schema": "rz-rovent-utility-train-window/2", "checkpoint_reserve_ns": 2000000000}
        independent = hashlib.sha256(json.dumps(["rz-rovent-utility-train-window/2", value],
            sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        wrong_domain = hashlib.sha256(json.dumps(["rz-rovent-utility-train-window/1", value],
            sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.assertEqual(window.utility_window_v2_digest(value), independent)
        self.assertNotEqual(independent, wrong_domain)
        self.assertNotEqual(independent, window.utility_window_v2_digest({**value, "checkpoint_reserve_ns": 2000000001}))
        with self.assertRaises(ValueError): window.utility_window_digest(value)
        for unknown in ({**value, "schema": "rz-rovent-utility-train-window/3"}, {}, None):
            with self.assertRaises(ValueError): window.utility_window_v2_digest(unknown)


class UtilitySamplerTests(unittest.TestCase):
    CONTEXTS = ("a1" * 32, "b2" * 32, "c3" * 32)
    ARGS = {"context_ids": CONTEXTS, "pool_sha256": "60" * 32, "run_id": "utility-run", "seed": 31}

    def make(self):
        return window.make_utility_sampler(self.CONTEXTS, "60" * 32, 31, run_id="utility-run")

    def test_sampler_owns_original_order_and_complete_permutation_without_mutation(self):
        initial = self.make(); original = copy.deepcopy(initial)
        self.assertEqual(initial["context_ids"], list(self.CONTEXTS))
        self.assertEqual(sorted(initial["permutation"]), [0, 1, 2])
        checked = window.validate_utility_sampler(initial, **self.ARGS)
        index, following = window.next_utility_sample(initial, **self.ARGS)
        self.assertEqual(index, initial["permutation"][0])
        self.assertEqual(following["cursor"], 1)
        self.assertEqual(checked, original)
        self.assertEqual(initial, original)

    def test_continuous_and_serialized_resume_choose_the_same_exact_indices(self):
        import json
        left = self.make(); right = self.make()
        observed = []
        for _ in range(11):
            left_index, left = window.next_utility_sample(left, **self.ARGS)
            right_index, right = window.next_utility_sample(json.loads(json.dumps(right)), **self.ARGS)
            self.assertEqual((left_index, left), (right_index, right))
            observed.append(left_index)
        self.assertEqual(sorted(observed[:3]), [0, 1, 2])
        self.assertEqual(sorted(observed[3:6]), [0, 1, 2])
        self.assertEqual(left["epoch"], 3)

    def test_seed_pool_context_order_and_run_cannot_be_relabelled(self):
        initial = self.make()
        for args in ({**self.ARGS, "seed": 32}, {**self.ARGS, "pool_sha256": "61" * 32},
                     {**self.ARGS, "context_ids": tuple(reversed(self.CONTEXTS))},
                     {**self.ARGS, "run_id": "different-run"}):
            with self.assertRaises(ValueError): window.validate_utility_sampler(initial, **args)

    def test_forged_permutation_cursor_epoch_and_legacy_fields_fail_closed(self):
        for mutate in (lambda value: value.update(permutation=[0, 0, 1]),
                       lambda value: value.update(permutation=[False, 1, 2]),
                       lambda value: value.update(cursor=4),
                       lambda value: value.update(epoch=True),
                       lambda value: value.update(schema="rz-rovent-native-stage-sampler/1"),
                       lambda value: value.update(execution_authorized=True)):
            changed = self.make(); mutate(changed)
            with self.assertRaises(ValueError): window.validate_utility_sampler(changed, **self.ARGS)

    def test_context_inventory_and_numpy_seed_width_are_bounded(self):
        for contexts in ((), ("a1" * 32, "a1" * 32), ("AA" * 32,)):
            with self.assertRaises(ValueError):
                window.make_utility_sampler(contexts, "60" * 32, 31, run_id="utility-run")
        with self.assertRaises(ValueError):
            window.make_utility_sampler(self.CONTEXTS, "60" * 32, 2**32, run_id="utility-run")


if __name__ == "__main__":
    unittest.main()
