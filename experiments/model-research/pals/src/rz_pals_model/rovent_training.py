"""Rovent trainer/checkpoint/export preparation with no neural imports.

The existing PALS actual trainer is a closed first-generation implementation;
its fixed recipe and diagnostic loaders are not repurposed here. These consumers
compile explicit parameter ownership and validate checkpoint state inventories
before a future live trainer can load bytes or mutate model/RNG/optimizer state.
An inventory is metadata, never an original native intake or numeric proof.
"""
from __future__ import annotations

import copy
import math

from .generation_data import digest, fields, identifier, sha, uint
from .generation_training import OPTIMIZER, _typed_equal, parameter_owner
from .rovent_generation import (SeriesRecipe, StageKind, fork_plan, resume_plan,
                                validate_checkpoint_metadata)

MEMBERSHIP_DOMAIN = "rz-rovent-trainable-parameter-membership/1"
MEMBERSHIP_V2_DOMAIN = "rz-rovent-trainable-parameter-membership/2"
STATE_DOMAIN = "rz-rovent-checkpoint-state-inventory/1"
WINDOW_DOMAIN = "rz-rovent-training-stage-preparation/1"
EXPORT_DOMAIN = "rz-rovent-export-preparation/1"
NNUEC_SCHEMA = "rz-nnuec-fp32-v1"
NNUEC_SEMANTICS = "board800-exactb192-causal-line-exactc256-fp32-v1"
NNUEC_V2_SCHEMA = "rz-nnuec-fp32-v2"
NNUEC_V2_PROTOTYPE_SEMANTICS = "board800-exactb192-causal-line-kind4-directedpair-exactc256-fp32-v2"
NNUEC_V2_SEMANTICS = "board800-exactb192-causal-line-kind4-directedpair-originfirst-exactc256-fp32-v2"
CONFIG_FIELDS = ("board_width", "line_width", "context_width", "local_width", "hidden_width",
                 "causal_window", "max_lines", "max_plies")
COMPONENTS = ("pc_shared", "p_private", "c_private", "v_private", "nnuec")
NNUEC_FIELDS = ("board_weights", "board_bias", "anchor_weights", "anchor_bias", "from_weights",
    "to_weights", "promotion_weights", "ply_weights", "token_bias", "causal_weights", "causal_bias",
    "context_projection", "context_bias", "local_projection", "local_bias", "hidden_weights",
    "hidden_bias", "output_weights", "output_bias")


def nnuec_parameter_shapes(config):
    """Wire shapes agreed with the Rust FP32 checkpoint and scalar reference.

    Embeddings are feature-row-major, unlike the output-row-major projection
    and head matrices. This describes the existing flattened wire layout and
    does not transpose weights or change exact B192/C256 accumulation semantics.
    """
    fields(config, CONFIG_FIELDS, "NNUEC FP32 configuration")
    for name in CONFIG_FIELDS[:4]:
        if not 1 <= uint(config[name], 256):
            raise ValueError("required bounded NNUEC width")
    if not 1 <= uint(config["hidden_width"], 64) or type(config["causal_window"]) is not int or config["causal_window"] not in (3, 5):
        raise ValueError("bounded NNUEC hidden width and registered causal window required")
    if not 1 <= uint(config["max_lines"], 128) or not 1 <= uint(config["max_plies"], 256):
        raise ValueError("bounded NNUEC line/ply capacity required")
    b, e, c, local, hidden = (config[name] for name in CONFIG_FIELDS[:5])
    return dict(zip(NNUEC_FIELDS, ([800, b], [b], [800, e], [e], [64, e], [64, e], [5, e],
        [config["max_plies"], e], [e], [config["causal_window"], e], [e], [c, e], [c], [local, e], [local],
        [hidden, 2 * b + c + local], [hidden], [11, hidden], [11]), strict=True))


def nnuec_parameter_descriptors(config):
    shapes = nnuec_parameter_shapes(config)
    embeddings = {"board_weights", "anchor_weights", "from_weights", "to_weights", "promotion_weights",
                  "ply_weights", "causal_weights"}
    return [{"name": "nnuec." + name, "wire_name": name, "shape": shape, "dtype": "float32",
             "semantics": NNUEC_SEMANTICS, "layout": "feature_rows" if name in embeddings else
             "output_rows" if len(shape) == 2 else "vector", "trainable": True}
            for name, shape in shapes.items()]


def nnuec_v2_parameter_descriptors(config, *, semantics=NNUEC_V2_SEMANTICS):
    """Explicit base19+relation5 domain; V1 is never silently upgraded."""
    if semantics not in (NNUEC_V2_PROTOTYPE_SEMANTICS, NNUEC_V2_SEMANTICS):
        raise ValueError("explicit recognized NNUEC V2 membership semantics required")
    parameters = [{**item, "name": "nnuec.base." + item["wire_name"],
        "wire_name": "base." + item["wire_name"], "semantics": semantics}
        for item in nnuec_parameter_descriptors(config)]
    e, c = config["line_width"], config["context_width"]
    for name, shape in (("kind_embeddings", [4, e]), ("pair_parent", [2, e]), ("pair_child", [2, e]),
                         ("pair_bias", [2, e]), ("pair_projection", [c, e])):
        parameters.append({"name": "nnuec." + name, "wire_name": name, "shape": shape, "dtype": "float32",
            "semantics": semantics, "layout": "output_rows" if name == "pair_projection" else "feature_rows", "trainable": True})
    return parameters


def nnuec_membership(membership):
    if membership["schema"] == MEMBERSHIP_DOMAIN:
        return NNUEC_SCHEMA, NNUEC_SEMANTICS, nnuec_parameter_descriptors(membership["nnuec_config"])
    if membership["schema"] == MEMBERSHIP_V2_DOMAIN:
        semantics = membership["nnuec_semantics"]
        return NNUEC_V2_SCHEMA, semantics, nnuec_v2_parameter_descriptors(membership["nnuec_config"], semantics=semantics)
    raise ValueError("unsupported NNUEC trainable membership domain")


def _component(name):
    if name.startswith("nnuec."):
        return "nnuec"
    # Reuse the existing P/C shared/private classifier, not a second prefix map.
    return {"shared": "pc_shared", "proposer": "p_private", "critic": "c_private", "validator": "v_private"}[parameter_owner(name)]


def validate_membership(value):
    if type(value) is not dict:
        raise ValueError("Rovent trainable membership fields mismatch")
    domain = value.get("schema")
    fields(value, ("schema", "model_semantics_sha256", "nnuec_config", "validator_present", "parameters")
        + (("nnuec_schema", "nnuec_semantics") if domain == MEMBERSHIP_V2_DOMAIN else ()), "Rovent trainable membership")
    if domain not in (MEMBERSHIP_DOMAIN, MEMBERSHIP_V2_DOMAIN) or type(value["validator_present"]) is not bool:
        raise ValueError("new explicit Rovent membership domain required")
    if domain == MEMBERSHIP_V2_DOMAIN and (value["nnuec_schema"] != NNUEC_V2_SCHEMA
            or value["nnuec_semantics"] not in (NNUEC_V2_PROTOTYPE_SEMANTICS, NNUEC_V2_SEMANTICS)):
        raise ValueError("explicit V2 NNUEC schema/semantics required")
    sha(value["model_semantics_sha256"])
    expected_nnuec = {item["name"]: item for item in nnuec_membership(value)[2]}
    parameters = value["parameters"]
    if type(parameters) is not list or not 1 <= len(parameters) <= 4096:
        raise ValueError("bounded explicit parameter membership required")
    seen, components, total = set(), set(), 0
    for item in parameters:
        fields(item, ("name", "wire_name", "shape", "dtype", "semantics", "layout", "trainable"), "trainable tensor descriptor")
        name = identifier(item["name"])
        if name in seen or item["dtype"] != "float32" or item["trainable"] is not True:
            raise ValueError("duplicate/non-FP32/nontrainable parameter descriptor")
        seen.add(name)
        component = _component(name)
        components.add(component)
        shape = item["shape"]
        if type(shape) is not list or not 1 <= len(shape) <= 4 or any(not 1 <= uint(dimension, 65536) for dimension in shape):
            raise ValueError("nonempty bounded tensor shape required")
        total += math.prod(shape)
        if total > 2**30:
            raise ValueError("declared trainable parameter byte capacity exceeded")
        if component == "nnuec":
            if name not in expected_nnuec or not _typed_equal(item, expected_nnuec[name]):
                raise ValueError("NNUEC tensor name/shape/layout/semantics differs from Rust FP32 wire")
        elif (item["wire_name"] != name or item["semantics"] != value["model_semantics_sha256"]
                or item["layout"] != "native_parameter_shape"):
            raise ValueError("P/C/V parameter descriptor changed source model semantics")
    required = {"pc_shared", "p_private", "c_private", "nnuec"} | ({"v_private"} if value["validator_present"] else set())
    if components != required or not set(expected_nnuec) <= seen:
        raise ValueError("incomplete shared/private/NNUEC parameter membership")
    return value


def validate_training_membership(value, *, nnuec_metadata=None, nnuec_parameter_spec=None):
    """Admit current model meaning, separately from historical metadata reads.

    The prototype /2 literal remains readable and keeps its exact canonical
    SHA. It cannot load into the current origin-first model by an alias. The
    caller repeats this guard against the live model's metadata and actual
    parameter spec before constructing moments or applying restored state.
    """
    checked = validate_membership(value)
    schema, semantics, descriptors = nnuec_membership(checked)
    if schema == NNUEC_V2_SCHEMA and semantics != NNUEC_V2_SEMANTICS:
        raise ValueError("historical NNUEC V2 prototype membership is not current originfirst training semantics")
    if (nnuec_metadata is None) != (nnuec_parameter_spec is None):
        raise ValueError("actual NNUEC metadata and parameter spec must be checked together")
    if nnuec_metadata is not None:
        fields(nnuec_metadata, ("schema", "model_schema", "model_semantics", "config", "output_scale", "training"),
            "actual trainable NNUEC metadata")
        if (nnuec_metadata["schema"] != "rz-nnuec-torch-metadata/1"
                or nnuec_metadata["model_schema"] != schema or nnuec_metadata["model_semantics"] != semantics
                or not _typed_equal(nnuec_metadata["config"], checked["nnuec_config"])):
            raise ValueError("actual NNUEC model metadata differs from exact registered schema/semantics/config")
        expected = {item["wire_name"]: item for item in descriptors}
        fields(nnuec_parameter_spec, expected, "actual trainable NNUEC parameter spec")
        for name, item in nnuec_parameter_spec.items():
            fields(item, ("shape", "dtype"), "actual NNUEC parameter shape/dtype")
            shape = item["shape"]
            if (type(shape) not in (tuple, list) or any(type(size) is not int for size in shape)
                    or list(shape) != expected[name]["shape"] or item["dtype"] != "float32"):
                raise ValueError("actual NNUEC parameter spec differs from registered wire shape/dtype")
    return checked


def membership_digest(value):
    checked = validate_membership(value)
    return digest(checked["schema"], checked)


def optimizer_ownership(recipe, membership):
    registered = recipe.to_dict()
    if membership_digest(membership) != registered["parameter_ownership"]["membership_sha256"]:
        raise ValueError("recipe parameter membership differs from trainer/export inventory")
    if membership["model_semantics_sha256"] != registered["model_semantics_sha256"]:
        raise ValueError("recipe/model semantic identity mismatch")
    groups = {}
    for parameter in membership["parameters"]:
        component = _component(parameter["name"])
        owner = registered["parameter_ownership"][component]
        groups.setdefault(owner, []).append(parameter["name"])
    return {owner: sorted(names) for owner, names in groups.items()}


def prepare_training_stage(recipe, membership, *, stage_id, mode, checkpoint=None, inventory=None):
    """Compile the future trainer's role activation and exact state operation."""
    if type(recipe) is not SeriesRecipe:
        raise ValueError("registered Rovent recipe required")
    groups = optimizer_ownership(recipe, membership)
    stage = recipe.stage(stage_id)
    if not membership["validator_present"] and stage.role_updates[2]:
        raise ValueError("V-free training inventory cannot dispatch V updates")
    if mode == "initialize":
        if stage.kind is not StageKind.COMMON or checkpoint is not None or inventory is not None:
            raise ValueError("initialization starts the common student without a foreign checkpoint")
        state = {"mode": "initialize_explicit_state", "completed_updates": dict.fromkeys(("proposer", "critic", "validator"), 0)}
    elif mode in ("resume", "fork"):
        checked = validate_checkpoint_inventory(recipe, membership, inventory)
        if not _typed_equal(inventory["checkpoint"], checkpoint):
            raise ValueError("stage source checkpoint differs from its full owned state inventory")
        state = (resume_plan(recipe, checkpoint, stage_id=stage_id) if mode == "resume" else
                 fork_plan(recipe, checkpoint, stage_id=stage_id))
        state["source_inventory_sha256"] = checked["inventory_sha256"]
    else:
        raise ValueError("explicit initialize/fork/resume operation required")
    active = {}
    for role, components in (("proposer", ("pc_shared", "p_private", "nnuec")),
                             ("critic", ("pc_shared", "c_private", "nnuec")), ("validator", ("v_private",))):
        names = [item["name"] for item in membership["parameters"] if _component(item["name"]) in components]
        active[role] = {"parameters": names, "optimizer_ids": sorted(owner for owner, owned in groups.items() if set(names) & set(owned))}
    return {"schema": WINDOW_DOMAIN, "recipe_sha256": recipe.sha256, "stage_id": stage_id,
            "student_id": recipe.to_dict()["student_id"], "learner_id": recipe.to_dict()["learner_id"],
            "round_id": stage.round_id, "teacher_actor_ids": list(stage.teacher_actor_ids),
            "lineage_teacher_actor_ids": list(recipe.inherited_teachers(stage_id)),
            "membership_sha256": membership_digest(membership), "optimizer_groups": groups, "role_activation": active,
            "optimizer_recipe": copy.deepcopy(OPTIMIZER), "state_operation": state,
            "legacy_generation_trainer_compatible": False,
            "requires_original_native_intake": True, "requires_global_lineage_audit": True,
            "actual_model_state_loaded": False,
            "actual_optimizer_created": False, "execution_authorized": False}


def _tensor_pin(value, expected_shape):
    fields(value, ("shape", "dtype", "sha256", "bytes"), "checkpoint FP32 tensor pin")
    if not _typed_equal(value["shape"], expected_shape) or value["dtype"] != "float32":
        raise ValueError("checkpoint tensor shape/dtype differs from registered parameter")
    sha(value["sha256"])
    if uint(value["bytes"]) != 4 * math.prod(expected_shape):
        raise ValueError("checkpoint tensor byte count differs from FP32 shape")


def validate_checkpoint_inventory(recipe, membership, value):
    """Check full owned state descriptors without loading tensors or caller RNG.

    State pins describe the original future writer's inventories, not the PALS
    diagnostic `_state_digest` or a torch.save archive. A runtime must still
    recover and compare actual tensor bytes before restoring caller state.
    """
    groups = optimizer_ownership(recipe, membership)
    fields(value, ("schema", "checkpoint", "membership_sha256", "model_tensors", "optimizers", "schedulers",
                   "rng_sha256", "sampler_sha256", "cost_sha256", "bucket_sha256"), "Rovent checkpoint inventory")
    if value["schema"] != STATE_DOMAIN or value["membership_sha256"] != membership_digest(membership):
        raise ValueError("new Rovent checkpoint inventory/membership domain required")
    checkpoint = validate_checkpoint_metadata(recipe, value["checkpoint"])
    parameters = {item["name"]: item for item in membership["parameters"]}
    fields(value["model_tensors"], parameters, "complete model tensor inventory")
    for name, pin in value["model_tensors"].items():
        _tensor_pin(pin, parameters[name]["shape"])
    fields(value["optimizers"], groups, "owned optimizer inventory")
    fields(value["schedulers"], groups, "owned constant scheduler inventory")
    for owner, names in groups.items():
        optimizer = fields(value["optimizers"][owner], ("algorithm", "hyperparameters", "parameters"), "AdamW owner")
        if optimizer["algorithm"] != "adamw" or not _typed_equal(optimizer["hyperparameters"], OPTIMIZER):
            raise ValueError("checkpoint optimizer recipe differs from the prepared AdamW recipe")
        fields(optimizer["parameters"], names, "single-owner AdamW parameter states")
        for name, state in optimizer["parameters"].items():
            fields(state, ("step", "exp_avg", "exp_avg_sq"), "AdamW moments and parameter step")
            step = uint(state["step"])
            component = _component(name)
            max_steps = checkpoint["completed_updates"]["validator"] if component == "v_private" else sum(
                checkpoint["completed_updates"][role] for role in ("proposer", "critic") if component in
                (("pc_shared", "p_private", "nnuec") if role == "proposer" else ("pc_shared", "c_private", "nnuec")))
            if step > max_steps:
                raise ValueError("optimizer parameter steps exceed completed active role updates")
            if step:
                _tensor_pin(state["exp_avg"], parameters[name]["shape"])
                _tensor_pin(state["exp_avg_sq"], parameters[name]["shape"])
            elif state["exp_avg"] is not None or state["exp_avg_sq"] is not None:
                raise ValueError("uninitialized AdamW parameter cannot claim existing moments")
        schedule = fields(value["schedulers"][owner], ("kind", "completed_steps", "state_sha256"), "constant scheduler owner")
        active_roles = set()
        for name in names:
            component = _component(name)
            active_roles.update(("validator",) if component == "v_private" else
                                ("proposer",) if component == "p_private" else
                                ("critic",) if component == "c_private" else ("proposer", "critic"))
        if schedule["kind"] != "constant" or uint(schedule["completed_steps"]) != sum(checkpoint["completed_updates"][role] for role in active_roles):
            raise ValueError("scheduler steps differ from completed owner updates")
        sha(schedule["state_sha256"])
    for key in ("rng_sha256", "sampler_sha256", "cost_sha256", "bucket_sha256"):
        sha(value[key])
    for part, contents in (("optimizer", value["optimizers"]), ("scheduler", value["schedulers"])):
        if checkpoint["state_sha256"][part] != digest("rz-rovent-" + part + "-inventory/1", contents):
            raise ValueError("checkpoint state pin differs from complete owned state inventory")
    for part in ("rng", "sampler", "cost"):
        if checkpoint["state_sha256"][part] != value[part + "_sha256"]:
            raise ValueError("checkpoint auxiliary state pin differs from original inventory")
    return {"schema": STATE_DOMAIN, "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "inventory_sha256": digest(STATE_DOMAIN, value), "model_tensor_count": len(parameters),
            "optimizer_owner_count": len(groups), "actual_tensor_bytes_verified": False,
            "caller_state_mutated": False, "execution_authorized": False}


def prepare_export(recipe, membership, checkpoint_inventory, *, kind):
    """Consume the same checked inventory for Collection or V-free Deployment."""
    checked = validate_checkpoint_inventory(recipe, membership, checkpoint_inventory)
    if kind not in ("collection", "deployment"):
        raise ValueError("explicit collection/deployment export kind required")
    include_v = kind == "collection" and membership["validator_present"]
    selected = [item for item in membership["parameters"] if kind == "collection" or _component(item["name"]) != "v_private"]
    return {"schema": EXPORT_DOMAIN, "kind": kind, "recipe_sha256": recipe.sha256,
            "checkpoint_sha256": checked["checkpoint_sha256"], "inventory_sha256": checked["inventory_sha256"],
            "membership_sha256": membership_digest(membership), "validator_present": include_v,
            "roles": ["proposer", "critic"] + (["validator"] if include_v else []),
            "parameter_names": [item["name"] for item in selected],
            "nnuec": {"schema": nnuec_membership(membership)[0], "semantics": nnuec_membership(membership)[1],
                      "parameters": nnuec_membership(membership)[2],
                      "output_scale": {"trainable": False, "source": "original_checkpoint_scalar"}},
            "legacy_export_domain_compatible": False, "actual_checkpoint_bytes_loaded": False,
            "actual_export_written": False, "training_provenance_verified": False,
            "numeric_acceptance": "not_run", "arena_eligible": False, "execution_authorized": False}
