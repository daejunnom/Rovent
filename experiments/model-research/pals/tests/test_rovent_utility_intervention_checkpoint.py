"""Detached codec fixtures only: no tensor, real process, update or peer lease."""
import copy
import hashlib
import json
import unittest
from unittest.mock import patch

from rz_pals_model.generation_data import digest
from rz_pals_model import rovent_utility_historical_checkpoint as legacy
from rz_pals_model import rovent_utility_intervention_checkpoint as checkpoint
from rz_pals_model.rovent_utility_intervention_preparation import prepare_intervention_metadata
from rz_pals_model.rovent_utility_intervention_numerical import validate_intervention_numerical_parts
from rz_pals_model.rovent_utility_training_window import make_utility_sampler
from test_rovent_utility_intervention_preparation import preparation_parts


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def projection():
    preparation = prepare_intervention_metadata(*preparation_parts())
    parts = validate_intervention_numerical_parts(preparation)
    registered, spec = parts.registration, parts.parameter_spec
    base, utility = registered["frozen_base"], registered["utility"]
    metadata = {"schema": "rz-nnuec-utility-torch-metadata/1", "model_schema": utility["schema"],
        "model_semantics": utility["semantics"], "base_schema": base["schema"], "base_semantics": base["semantics"],
        "base_weights_sha256": base["weights_sha256"], "frozen_base_mapping_sha256": "21" * 32,
        "frozen_base_config": base["config"], "hidden_width": spec["hidden_width"],
        "input_width": spec["input_width"], "initial_training": {"kind": "untrained"}}
    sampler = make_utility_sampler(["61" * 32], "62" * 32, 31, run_id="intervention-run")
    zero = dict.fromkeys(("forward_calls", "backward_calls", "samples", "wall_time_ns",
        "partial_accumulation_ns", "replay_ns", "masked_groups", "output_bytes"), 0)
    result = {"schema": checkpoint.PROJECTION_DOMAIN, "payload_state_sha256": "63" * 32,
        "preparation": preparation, "metadata": metadata, "sampler": sampler, "cuda_devices": [],
        "parameter_names": [spec["wire_to_parameter"][name] for name in parts.recipe["active_parameter_names"]],
        "completed_updates": 0, "dispatched_updates": 0, "ancestor_updates": 0,
        "pending_microbatches": 0, "cost": dict(zero), "ancestor_cost": dict(zero),
        "completion_receipts": [], "ancestor_completion_sha256": [], "parent_checkpoint_sha256": None,
        "source_cost_observation": None, "model_state_sha256": "64" * 32,
        "optimizer_state_sha256": "65" * 32, "scheduler_state_sha256": "66" * 32,
        "rng_state_sha256": "67" * 32, "sampler_state_sha256": legacy._plain_state_sha256(sampler)}
    result["derived_training_state"] = {"schema": checkpoint.STATE_DOMAIN,
        "initial_source_declaration": {"kind": "untrained"}, "current_tensor_kind": "initial_source_only",
        "completed_updates": 0, "ancestor_updates": 0, "total_updates": 0,
        "completion_chain_sha256": digest(checkpoint.STATE_DOMAIN, []),
        "recipe_sha256": parts.recipe_sha256, "registration_sha256": parts.source_registration_sha256,
        "frozen_base_weights_sha256": base["weights_sha256"], "frozen_base_mapping_sha256": "21" * 32,
        "live_training_authority_reissued": False}
    return result


def parsed(value, parser=checkpoint.parse_intervention_utility_tensor_projection):
    raw = encoded(value)
    return parser(raw, expected_sha256=hashlib.sha256(raw).hexdigest(), expected_bytes=len(raw))


class InterventionCheckpointCodecTests(unittest.TestCase):
    def test_every_materializing_entry_denies_before_owner_path_authority_or_metadata(self):
        class Untouchable:
            def __getattribute__(self, name):
                raise AssertionError("denied execution inspected an owner, authority or input")

            def __fspath__(self):
                raise AssertionError("denied execution inspected a path")

        value = Untouchable()
        for flag in (False, None, 0, 1, "true", [], {}, object()):
            calls = {
                "create": lambda: checkpoint.create_intervention_utility_owner(value, value,
                    authority=value, sampler=value, sampler_validator=None, execute=flag),
                "payload": lambda: checkpoint.intervention_utility_tensor_payload(value,
                    sampler_validator=None, execute=flag),
                "projection": lambda: checkpoint.intervention_utility_tensor_projection(value,
                    sampler_validator=None, execute=flag),
                "save": lambda: checkpoint.save_intervention_utility_checkpoint(value, value,
                    authority=value, lease=value, sampler_validator=None, post_snapshot_observer=value,
                    max_file_bytes=value, read_accounting=value, execute=flag),
                "resume": lambda: checkpoint.stage_intervention_utility_resume(value, value, value, value,
                    authority=value, historical_checkpoint=value, sampler_validator=None,
                    max_file_bytes=value, read_accounting=value, snapshot_directory=value, execute=flag),
                "fork": lambda: checkpoint.stage_intervention_utility_fork(value, preparation=value,
                    authority=value, sampler=value, source_sampler_validator=None, sampler_validator=None,
                    reset_seed=value, execute=flag),
                "export": lambda: checkpoint.export_intervention_utility_owner(value, value,
                    authority=value, kind=value, max_output_bytes=value, execute=flag),
            }
            for boundary, call in calls.items():
                with self.subTest(boundary=boundary, flag=type(flag).__name__):
                    with self.assertRaisesRegex(ValueError, "explicit utility tensor execution required"):
                        call()

    def test_exact9_projection_keeps_original_preparation_and_emits_no_authority(self):
        value = projection()
        before = copy.deepcopy(value)
        self.assertEqual(parsed(value), before)
        self.assertEqual(len(value["preparation"]), 9)
        self.assertNotIn("recipe_sha256", value["preparation"])
        self.assertNotIn("registration_sha256", value["preparation"])
        observed = checkpoint.project_intervention_utility_tensor_scratch(value, observed_state=before)
        self.assertEqual(observed["schema"], "rz-rovent-utility-historical-scratch-observation/2")
        for name in ("execution_authorized", "old_run_reclamation_issued", "past_physical_completion_reissued"):
            self.assertFalse(observed[name])
        self.assertEqual(value, before)

    def test_original_public_reader_remains_old_only_and_new_reader_refuses_old_projection(self):
        value = projection()
        with self.assertRaises(ValueError):
            parsed(value, legacy.parse_utility_tensor_projection)
        value["schema"] = legacy.TENSOR_PROJECTION_DOMAIN
        with self.assertRaises(ValueError):
            parsed(value)

    def test_new_projection_cannot_add_synthesized_old_preparation_fields(self):
        value = projection()
        for field in ("recipe_sha256", "registration_sha256", "optimizer_owner"):
            changed = copy.deepcopy(value)
            changed["preparation"][field] = "90" * 32
            with self.subTest(field=field), self.assertRaises(ValueError):
                parsed(changed)

    def test_new_scratch_compares_all_five_state_families_and_cost(self):
        value = projection()
        for name in legacy.STATE_FIELDS:
            changed = copy.deepcopy(value)
            changed[name] = "91" * 32
            with self.subTest(name=name), self.assertRaises(ValueError):
                checkpoint.project_intervention_utility_tensor_scratch(value, observed_state=changed)
        changed = copy.deepcopy(value)
        changed["cost"]["replay_ns"] = 1
        with self.assertRaises(ValueError):
            checkpoint.project_intervention_utility_tensor_scratch(value, observed_state=changed)

    def test_projection_original_raw_hash_and_duplicate_keys_are_not_rebuilt(self):
        raw = encoded(projection())
        with self.assertRaises(ValueError):
            checkpoint.parse_intervention_utility_tensor_projection(raw, expected_sha256="00" * 32,
                expected_bytes=len(raw))
        raw = b'{"schema":"one","schema":"two"}'
        with self.assertRaises(ValueError):
            checkpoint.parse_intervention_utility_tensor_projection(raw,
                expected_sha256=hashlib.sha256(raw).hexdigest(), expected_bytes=len(raw))

    def test_unrelated_authority_cannot_reach_checkpoint_export_or_model_runtime(self):
        with patch.object(checkpoint.tensors, "_runtime", side_effect=AssertionError("no neural runtime")):
            with self.assertRaises(ValueError):
                checkpoint.export_intervention_utility_owner(object(), "/unused/export.json",
                    authority=object(), kind="deployment", execute=True)
            with self.assertRaises(ValueError):
                checkpoint.save_intervention_utility_checkpoint(object(), "/unused/checkpoint.pt",
                    authority=object(), lease=object(), sampler_validator=lambda _: None,
                    post_snapshot_observer=lambda _: None, execute=True)


if __name__ == "__main__":
    unittest.main()
