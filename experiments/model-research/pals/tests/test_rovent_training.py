"""Tiny parameter/state inventories; never model weights or training proof."""
import copy
import importlib.abc
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}
                or fullname in {"rz_pals_model.model", "rz_pals_model.training", "rz_pals_model.nonzero_training"}):
            raise AssertionError("model/optimizer import forbidden in trainer preparation tests")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_training as training
    from rz_pals_model import rovent_generation_cli as cli
    from rz_pals_model.generation_data import digest
    from rz_pals_model.rovent_generation import SeriesRecipe
    from test_rovent_generation import checkpoint as series_checkpoint
    from test_rovent_generation import json_pin, recipe_value, request_value, seal
finally:
    sys.meta_path.remove(_guard)


def membership_fixture(*, validator=True):
    config = {"board_width": 2, "line_width": 3, "context_width": 4, "local_width": 5,
              "hidden_width": 6, "causal_window": 3, "max_lines": 4, "max_plies": 8}
    names = ["public_encoder.record_field_embedding", "experts.proposer.initial_latent", "experts.critic.initial_latent"]
    if validator:
        names.append("experts.validator.initial_latent")
    parameters = [{"name": name, "wire_name": name, "shape": [1, 2], "dtype": "float32",
                   "semantics": seal("semantics"), "layout": "native_parameter_shape", "trainable": True} for name in names]
    parameters.extend(training.nnuec_parameter_descriptors(config))
    return {"schema": training.MEMBERSHIP_DOMAIN, "model_semantics_sha256": seal("semantics"),
            "nnuec_config": config, "validator_present": validator, "parameters": parameters}


def recipe_fixture(membership):
    value = recipe_value()
    value["parameter_ownership"]["nnuec"] = "pc"
    value["parameter_ownership"]["membership_sha256"] = training.membership_digest(membership)
    return SeriesRecipe(value)


def membership_v2_fixture(*, validator=True, semantics=None):
    semantics = training.NNUEC_V2_SEMANTICS if semantics is None else semantics
    original = membership_fixture(validator=validator)
    return {**original, "schema": training.MEMBERSHIP_V2_DOMAIN,
        "nnuec_schema": training.NNUEC_V2_SCHEMA, "nnuec_semantics": semantics,
        "parameters": [item for item in original["parameters"] if not item["name"].startswith("nnuec.")]
            + training.nnuec_v2_parameter_descriptors(original["nnuec_config"], semantics=semantics)}


def checkpoint(recipe):
    value = series_checkpoint(recipe)
    value["parameter_membership_sha256"] = recipe.to_dict()["parameter_ownership"]["membership_sha256"]
    return value


def tensor_pin(name, shape):
    return {"shape": list(shape), "dtype": "float32", "sha256": seal(name), "bytes": math.prod(shape) * 4}


def inventory_fixture(recipe, membership):
    """Inventories are schema fixtures; none of the advertised bytes exist."""
    summary = checkpoint(recipe)
    parameters = {item["name"]: item for item in membership["parameters"]}
    groups = training.optimizer_ownership(recipe, membership)
    optimizers, schedules = {}, {}
    for owner, names in groups.items():
        states = {}
        for name in names:
            count = 0 if name.startswith("experts.validator.") else 2 if name.startswith("experts.") else 4
            shape = parameters[name]["shape"]
            states[name] = {"step": count,
                "exp_avg": tensor_pin(name + "moment1", shape) if count else None,
                "exp_avg_sq": tensor_pin(name + "moment2", shape) if count else None}
        optimizers[owner] = {"algorithm": "adamw", "hyperparameters": copy.deepcopy(training.OPTIMIZER), "parameters": states}
        schedules[owner] = {"kind": "constant", "completed_steps": 0 if owner == "v" else 4, "state_sha256": seal(owner + "schedule")}
    summary["state_sha256"]["optimizer"] = digest("rz-rovent-optimizer-inventory/1", optimizers)
    summary["state_sha256"]["scheduler"] = digest("rz-rovent-scheduler-inventory/1", schedules)
    return {"schema": training.STATE_DOMAIN, "checkpoint": summary,
            "membership_sha256": training.membership_digest(membership),
            "model_tensors": {name: tensor_pin(name + "model", item["shape"]) for name, item in parameters.items()},
            "optimizers": optimizers, "schedulers": schedules,
            **{part + "_sha256": summary["state_sha256"][part] for part in ("rng", "sampler", "cost")},
            "bucket_sha256": seal("bucket")}


class RoventTrainingPreparationTests(unittest.TestCase):
    def test_windows_name_comparison_uses_birthtime_but_fd_keeps_changetime(self):
        common = dict(st_dev=1, st_ino=2, st_mode=0o100600, st_nlink=1,
                      st_size=8, st_mtime_ns=20, st_birthtime_ns=10)
        held = SimpleNamespace(**common, st_ctime_ns=30)
        named = SimpleNamespace(**common, st_ctime_ns=10)
        changed = SimpleNamespace(**common, st_ctime_ns=31)
        with patch.object(cli.os, "name", "nt"):
            self.assertEqual(cli._stamp(held, named_comparison=True),
                             cli._stamp(named, named_comparison=True))
            self.assertNotEqual(cli._stamp(held), cli._stamp(changed))
            for field in ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns"):
                replaced = SimpleNamespace(**vars(named))
                setattr(replaced, field, getattr(replaced, field) + 1)
                self.assertNotEqual(cli._stamp(held, named_comparison=True),
                                    cli._stamp(replaced, named_comparison=True))

    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_prototype_v2_metadata_keeps_its_literal_and_digest_but_cannot_train_current_model(self):
        old_literal = "board800-exactb192-causal-line-kind4-directedpair-exactc256-fp32-v2"
        prototype = membership_v2_fixture(semantics=old_literal)
        before = copy.deepcopy(prototype)
        expected = digest("rz-rovent-trainable-parameter-membership/2", before)
        self.assertEqual(training.membership_digest(prototype), expected)
        self.assertEqual(training.nnuec_membership(prototype)[1], old_literal)
        with self.assertRaisesRegex(ValueError, "prototype membership.*originfirst"):
            training.validate_training_membership(prototype)
        self.assertEqual(prototype, before)
        current = membership_v2_fixture()
        training.validate_training_membership(current)
        self.assertNotEqual(training.membership_digest(current), expected)
        self.assertEqual(training.membership_digest(prototype), expected)

    def test_current_guard_uses_independent_reference_metadata_and_exact_parameter_spec(self):
        # The independent scalar reference imports only the standard library;
        # importing its public metadata constructs no weights or model. Its
        # assignment/declaration shape is not part of the training contract.
        from rz_pals_model.nnuec_v2_reference import SEMANTICS as semantics
        current = membership_v2_fixture()
        self.assertEqual(current["nnuec_semantics"], semantics)
        metadata = {"schema": "rz-nnuec-torch-metadata/1", "model_schema": "rz-nnuec-fp32-v2",
            "model_semantics": semantics, "config": copy.deepcopy(current["nnuec_config"]),
            "output_scale": 1.0, "training": {"kind": "untrained"}}
        spec = {item["wire_name"]: {"shape": tuple(item["shape"]), "dtype": "float32"}
            for item in current["parameters"] if item["name"].startswith("nnuec.")}
        self.assertEqual(spec["base.board_weights"]["shape"], (800, 2))
        self.assertEqual(spec["kind_embeddings"]["shape"], (4, 3))
        self.assertEqual(spec["pair_projection"]["shape"], (4, 3))
        training.validate_training_membership(current, nnuec_metadata=metadata, nnuec_parameter_spec=spec)
        for field, value in (("model_semantics", "board800-exactb192-causal-line-kind4-directedpair-exactc256-fp32-v2"),
                              ("model_schema", "rz-nnuec-fp32-v1"), ("config", {**metadata["config"], "line_width": 4})):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "model metadata differs"):
                training.validate_training_membership(current, nnuec_metadata={**metadata, field: value}, nnuec_parameter_spec=spec)
        altered = copy.deepcopy(spec)
        altered["pair_projection"]["shape"] = (3, 4)
        with self.assertRaisesRegex(ValueError, "parameter spec differs"):
            training.validate_training_membership(current, nnuec_metadata=metadata, nnuec_parameter_spec=altered)

    def test_v2_membership_is_explicit_base19_plus_relation5_without_v1_or_utility_conversion(self):
        original = membership_fixture()
        before = copy.deepcopy(original)
        original_digest = training.membership_digest(original)
        revised = membership_v2_fixture()
        training.validate_membership(revised)
        parameters = {item["name"]: item for item in revised["parameters"] if item["name"].startswith("nnuec.")}
        self.assertEqual(len(parameters), 24)
        self.assertEqual(parameters["nnuec.base.board_weights"]["shape"], [800, 2])
        self.assertEqual(parameters["nnuec.kind_embeddings"]["shape"], [4, 3])
        self.assertEqual(parameters["nnuec.pair_projection"]["shape"], [4, 3])
        self.assertEqual(parameters["nnuec.pair_projection"]["layout"], "output_rows")
        self.assertNotEqual(training.membership_digest(revised), original_digest)
        self.assertEqual(original, before)
        self.assertEqual(training.membership_digest(original), original_digest)
        for name in ("nnuec_schema", "nnuec_semantics"):
            bad = copy.deepcopy(revised)
            del bad[name]
            with self.subTest(missing=name), self.assertRaisesRegex(ValueError, "fields mismatch"):
                training.validate_membership(bad)
        for name in ("nnuec.pair_parent", "nnuec.base.board_weights"):
            bad = copy.deepcopy(revised)
            next(item for item in bad["parameters"] if item["name"] == name)["shape"] = [1, 1]
            with self.subTest(shape=name), self.assertRaisesRegex(ValueError, "differs from Rust FP32 wire"):
                training.validate_membership(bad)
        bad = copy.deepcopy(revised)
        bad["parameters"].append({**parameters["nnuec.pair_projection"], "name": "nnuec.utility_weights", "wire_name": "utility_weights"})
        with self.assertRaisesRegex(ValueError, "differs from Rust FP32 wire"):
            training.validate_membership(bad)

    def test_nnuec_embedding_and_dense_shapes_match_the_rust_wire_orientation(self):
        membership = membership_fixture()
        descriptors = {item["wire_name"]: item for item in membership["parameters"] if item["name"].startswith("nnuec.")}
        expected = {"board_weights": [800, 2], "anchor_weights": [800, 3], "from_weights": [64, 3],
                    "to_weights": [64, 3], "promotion_weights": [5, 3], "ply_weights": [8, 3],
                    "causal_weights": [3, 3], "context_projection": [4, 3], "local_projection": [5, 3],
                    "hidden_weights": [6, 13], "output_weights": [11, 6]}
        for name, shape in expected.items():
            self.assertEqual(descriptors[name]["shape"], shape)
            self.assertEqual(descriptors[name]["semantics"], "board800-exactb192-causal-line-exactc256-fp32-v1")
        self.assertEqual(descriptors["board_weights"]["layout"], "feature_rows")
        self.assertEqual(descriptors["output_weights"]["layout"], "output_rows")
        transposed = copy.deepcopy(membership)
        next(item for item in transposed["parameters"] if item["name"] == "nnuec.board_weights")["shape"] = [2, 800]
        with self.assertRaisesRegex(ValueError, "differs from Rust FP32 wire"):
            training.validate_membership(transposed)
        self.assertNotIn("output_scale", descriptors)

    def test_role_activation_keeps_one_pc_optimizer_and_private_v_separate(self):
        membership = membership_fixture()
        recipe = recipe_fixture(membership)
        prepared = training.prepare_training_stage(recipe, membership, stage_id="common", mode="initialize")
        self.assertEqual(set(prepared["optimizer_groups"]), {"pc", "v"})
        self.assertEqual(prepared["role_activation"]["proposer"]["optimizer_ids"], ["pc"])
        self.assertEqual(prepared["role_activation"]["critic"]["optimizer_ids"], ["pc"])
        self.assertEqual(prepared["role_activation"]["validator"]["optimizer_ids"], ["v"])
        self.assertFalse(any(name.startswith("experts.validator.") for name in prepared["optimizer_groups"]["pc"]))
        self.assertFalse(any(name.startswith("nnuec.") for name in prepared["optimizer_groups"]["v"]))
        self.assertIs(prepared["actual_optimizer_created"], False)
        self.assertIs(prepared["legacy_generation_trainer_compatible"], False)

    def test_fork_and_resume_consume_the_same_registered_membership_without_state_aliases(self):
        membership = membership_fixture()
        recipe = recipe_fixture(membership)
        inventory = inventory_fixture(recipe, membership)
        common = inventory["checkpoint"]
        before = copy.deepcopy(common)
        sf = training.prepare_training_stage(recipe, membership, stage_id="sf-ft", mode="fork", checkpoint=common, inventory=inventory)
        reckless = training.prepare_training_stage(recipe, membership, stage_id="reckless-ft", mode="fork", checkpoint=common, inventory=inventory)
        self.assertEqual(sf["state_operation"]["parent_checkpoint_sha256"], reckless["state_operation"]["parent_checkpoint_sha256"])
        self.assertEqual(sf["lineage_teacher_actor_ids"], ["reckless", "stockfish"])
        resumed = training.prepare_training_stage(recipe, membership, stage_id="common", mode="resume", checkpoint=common, inventory=inventory)
        self.assertEqual(resumed["state_operation"]["checkpoint"], before)
        resumed["state_operation"]["checkpoint"]["state_sha256"]["rng"] = seal("foreign-rng")
        self.assertEqual(common, before)
        wrong = copy.deepcopy(membership)
        wrong["parameters"][0]["shape"] = [2, 2]
        with self.assertRaisesRegex(ValueError, "membership differs"):
            training.prepare_training_stage(recipe, wrong, stage_id="common", mode="initialize")
        with self.assertRaisesRegex(ValueError, "checkpoint inventory fields mismatch"):
            training.prepare_training_stage(recipe, membership, stage_id="common", mode="resume", checkpoint=common)
        foreign = copy.deepcopy(common)
        foreign["state_sha256"]["rng"] = seal("foreign")
        with self.assertRaisesRegex(ValueError, "source checkpoint differs"):
            training.prepare_training_stage(recipe, membership, stage_id="common", mode="resume", checkpoint=foreign, inventory=inventory)

    def test_full_checkpoint_inventory_checks_moments_scheduler_and_shape_before_any_mutation(self):
        membership = membership_fixture()
        recipe = recipe_fixture(membership)
        inventory = inventory_fixture(recipe, membership)
        before = copy.deepcopy(inventory)
        result = training.validate_checkpoint_inventory(recipe, membership, inventory)
        self.assertEqual(result["optimizer_owner_count"], 2)
        self.assertIs(result["actual_tensor_bytes_verified"], False)
        self.assertIs(result["caller_state_mutated"], False)
        self.assertEqual(inventory, before)
        cases = [("moment", lambda value: value["optimizers"]["pc"]["parameters"]["nnuec.board_weights"].update(exp_avg=None)),
                 ("scheduler", lambda value: value["schedulers"]["pc"].update(completed_steps=3)),
                 ("rng", lambda value: value.update(rng_sha256=seal("different-rng"))),
                 ("shape", lambda value: value["model_tensors"]["experts.proposer.initial_latent"].update(shape=[True, 2])),
                 ("diagnostic", lambda value: value.update(schema="rz-pals-python-nonzero-checkpoint/1"))]
        for name, mutate in cases:
            with self.subTest(case=name):
                value = copy.deepcopy(inventory)
                mutate(value)
                with self.assertRaises(ValueError):
                    training.validate_checkpoint_inventory(recipe, membership, value)
                self.assertEqual(inventory, before)

    def test_collection_and_v_free_deployment_exports_do_not_flip_actual_eligibility(self):
        membership = membership_fixture()
        recipe = recipe_fixture(membership)
        inventory = inventory_fixture(recipe, membership)
        collection = training.prepare_export(recipe, membership, inventory, kind="collection")
        deployment = training.prepare_export(recipe, membership, inventory, kind="deployment")
        self.assertEqual(collection["roles"], ["proposer", "critic", "validator"])
        self.assertEqual(deployment["roles"], ["proposer", "critic"])
        self.assertFalse(any(name.startswith("experts.validator.") for name in deployment["parameter_names"]))
        self.assertEqual(deployment["nnuec"]["schema"], "rz-nnuec-fp32-v1")
        self.assertIs(deployment["nnuec"]["output_scale"]["trainable"], False)
        for prepared in (collection, deployment):
            self.assertIs(prepared["actual_export_written"], False)
            self.assertIs(prepared["training_provenance_verified"], False)
            self.assertIs(prepared["arena_eligible"], False)
        no_v_membership = membership_fixture(validator=False)
        no_v_recipe = recipe_fixture(no_v_membership)
        no_v_collection = training.prepare_export(no_v_recipe, no_v_membership,
            inventory_fixture(no_v_recipe, no_v_membership), kind="collection")
        self.assertEqual(no_v_collection["roles"], ["proposer", "critic"])
        self.assertIs(no_v_collection["validator_present"], False)

    def test_v2_file_consumer_loads_membership_and_inventory_only_as_pinned_json(self):
        membership = membership_fixture()
        recipe = recipe_fixture(membership)
        inventory = inventory_fixture(recipe, membership)
        with tempfile.TemporaryDirectory(prefix="rz-rovent-trainer-prep-") as temporary:
            root = Path(temporary)
            request = request_value(root, recipe, action="prepare_training_stage")
            request.update(schema=cli.TRAINING_REQUEST_DOMAIN, stage_id="common", training_preparation={
                "membership": json_pin(root, "membership.json", membership), "inventory": None,
                "mode": "initialize", "export_kind": None})
            stage = cli.prepare_request(request, budget=cli.PreparationBudget(max_wall_ns=10**9))
            self.assertEqual(stage["result"]["stage_id"], "common")
            request.update(action="prepare_model_export", stage_id=None)
            request["training_preparation"].update(inventory=json_pin(root, "checkpoint-inventory.json", inventory),
                mode=None, export_kind="deployment")
            exported = cli.prepare_request(request, budget=cli.PreparationBudget(max_wall_ns=10**9))
            self.assertEqual(exported["result"]["roles"], ["proposer", "critic"])
            self.assertIs(exported["execution_authorized"], False)
            self.assertEqual(exported["input_pins"]["training_preparation"], request["training_preparation"])
            request.update(action="prepare_training_stage", stage_id="common",
                           checkpoint=json_pin(root, "source-checkpoint-metadata.json", inventory["checkpoint"]))
            request["training_preparation"].update(mode="resume", export_kind=None)
            resumed = cli.prepare_request(request, budget=cli.PreparationBudget(max_wall_ns=10**9))
            self.assertEqual(resumed["result"]["state_operation"]["mode"], "resume_exact_state")
            self.assertEqual(resumed["result"]["state_operation"]["checkpoint"], inventory["checkpoint"])
            old = copy.deepcopy(request)
            old["schema"] = cli.REQUEST_DOMAIN
            old.pop("training_preparation")
            with self.assertRaisesRegex(ValueError, "require request V2"):
                cli.validate_request(old)


if __name__ == "__main__":
    unittest.main()
