"""N05 utility recipe and metadata preparation, without execution authority.

This module uses only the standard library. It does not import a tensor model,
create an optimizer, perform an update, write a checkpoint, or authorize an
action. The future actual-source adapter must independently validate the opaque
action/cost/witness owner before projecting any known route comparisons here.
The existing numerical utility V1 and P/C/V recipes keep their own meanings.
"""
from __future__ import annotations

import hashlib
import json
import math


RECIPE_DOMAIN = "rz-rovent-utility-training-recipe/1"
PREPARATION_DOMAIN = "rz-rovent-utility-training-preparation/1"
COHORT_PREPARATION_DOMAIN = "rz-rovent-utility-training-preparation/2"
RECIPE_VERSION = "actual-route-known-pairwise-logistic/1"
ROUTE_ACTIONS = (4, 5, 6)
PARAMETER_FIELDS = ("hidden_weights", "hidden_bias", "output_weights", "output_bias")
FROZEN_COMPONENTS = ("nnuec_base", "p", "c")
BASE_SCHEMA = "rz-nnuec-fp32-v2"
BASE_SEMANTICS = "board800-exactb192-causal-line-kind4-directedpair-originfirst-exactc256-fp32-v2"
UTILITY_SCHEMA = "rz-nnuec-utility-fp32-v1"
UTILITY_SEMANTICS = "nnuec-v2-targetbundle-risk-route-shadow-fp32-v1"
COST_SCOPE = "exclusive_submitted_global_target_route_and_descendants_with_required_drain"
COST_OBJECTIVE = "exclusive_wall_nanoseconds"
COHORT_COST_SCOPE = "cohort_route_with_bounded_materialization_and_physical_pin_zero_excluding_durable_dataset_io_v1"
MAX_PREPARATION_BYTES = 64 * 1024
MAX_CONTEXTS = 8192
MAX_PAIR_ENTRIES_PER_CONTEXT = 64
_UINT64 = 2**64 - 1
_FP32_MAX = float.fromhex("0x1.fffffep+127")
_CONFIG_FIELDS = ("board_width", "line_width", "context_width", "local_width", "hidden_width",
                  "causal_window", "max_lines", "max_plies")
_OPTIMIZER = {
    "kind": "adamw", "learning_rate": "0.0001", "betas": ["0.9", "0.999"],
    "epsilon": "0.00000001", "weight_decay": "0.01", "gradient_clip_norm": "1.0",
    "schedule": "constant",
}
_LOSS = {"kind": "pairwise_logistic", "route_actions": list(ROUTE_ACTIONS),
         "normalization": "mean_unique_known_pairs", "weight": "1.0"}


def _fields(value, names, label):
    if type(value) is not dict or len(value) != len(names) or set(value) != set(names):
        raise ValueError(f"{label}: exact fields required")
    return value


def _uint(value, maximum=_UINT64):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("bounded utility unsigned integer required")
    return value


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("lowercase utility SHA-256 pin required")
    return value


def _identifier(value):
    if (type(value) is not str or not value.strip() or len(value.encode("utf-8")) > 256
            or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in value)):
        raise ValueError("bounded utility metadata identifier required")
    return value


def _sequence(value, maximum, label):
    if type(value) not in (list, tuple) or len(value) > maximum:
        raise ValueError(f"bounded {label} sequence required")
    return value


def _json(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    if len(raw) > MAX_PREPARATION_BYTES:
        raise ValueError("utility preparation metadata exceeds 64KiB")
    return raw


def _copy(value):
    return json.loads(_json(value).decode("utf-8"))


def _digest(domain, value):
    return hashlib.sha256(_json([domain, value])).hexdigest()


def make_recipe(optimizer_owner):
    """Declare a separate future optimizer owner; a name grants no authority."""
    return {
        "schema": RECIPE_DOMAIN, "version": RECIPE_VERSION, "loss": _copy(_LOSS),
        "optimizer": _copy(_OPTIMIZER), "optimizer_owner": _identifier(optimizer_owner),
        "active_parameter_names": list(PARAMETER_FIELDS), "frozen_components": list(FROZEN_COMPONENTS),
        "limits": {"max_contexts": MAX_CONTEXTS, "max_pair_entries_per_context": MAX_PAIR_ENTRIES_PER_CONTEXT,
                   "max_preparation_bytes": MAX_PREPARATION_BYTES},
    }


def validate_recipe(value):
    _fields(value, ("schema", "version", "loss", "optimizer", "optimizer_owner", "active_parameter_names",
                    "frozen_components", "limits"), "utility recipe")
    _identifier(value["optimizer_owner"])
    loss = _fields(value["loss"], tuple(_LOSS), "utility loss recipe")
    _fields(value["optimizer"], tuple(_OPTIMIZER), "utility optimizer recipe")
    limits = _fields(value["limits"], ("max_contexts", "max_pair_entries_per_context", "max_preparation_bytes"),
                     "utility preparation limits")
    if type(loss["route_actions"]) is not list or any(type(action) is not int for action in loss["route_actions"]):
        raise ValueError("registered integer route action indices required")
    for limit in limits.values():
        _uint(limit)
    expected = make_recipe(value["optimizer_owner"])
    if value != expected:
        raise ValueError("utility recipe differs from the explicitly registered fixed recipe")
    return _copy(value)


def recipe_sha256(value):
    return _digest(RECIPE_DOMAIN, validate_recipe(value))


def _base_config(value):
    """Check the actual frozen base config projection, without loading weights."""
    _fields(value, _CONFIG_FIELDS, "frozen NNUEC base config")
    for name in _CONFIG_FIELDS[:4]:
        if not 1 <= _uint(value[name], 256):
            raise ValueError("positive frozen NNUEC feature width required")
    if not 1 <= _uint(value["hidden_width"], 64):
        raise ValueError("frozen NNUEC hidden width required")
    if type(value["causal_window"]) is not int or value["causal_window"] not in (3, 5):
        raise ValueError("registered frozen NNUEC causal window required")
    if not 1 <= _uint(value["max_lines"], 128) or not 1 <= _uint(value["max_plies"], 256):
        raise ValueError("bounded frozen NNUEC line/ply config required")
    return _copy(value)


def validate_parameter_membership(spec, frozen_base_config, optimizer_owner):
    """Validate a planned wire-to-parameter mapping, never observed tensor state.

    The existing checkpoint paths are exactly parameters.<wire-name>. Actual
    future module names are explicitly supplied, not inferred from P/C names or
    silently assumed to exist. A tensor consumer must check names, ownership,
    aliasing, dtype and requires_grad again against the actual model.
    """
    config = _base_config(frozen_base_config)
    _identifier(optimizer_owner)
    _fields(spec, ("hidden_width", "input_width", "wire_to_parameter", "parameters", "optimizer_owner",
                   "frozen_components"), "utility parameter membership")
    hidden = _uint(spec["hidden_width"], 64)
    width = 2 * config["board_width"] + 2 * config["context_width"] + config["local_width"]
    if not hidden or type(spec["input_width"]) is not int or spec["input_width"] != width:
        raise ValueError("utility H/I must join the actual frozen base I=2B+2C+L")
    if spec["optimizer_owner"] != optimizer_owner or spec["frozen_components"] != list(FROZEN_COMPONENTS):
        raise ValueError("utility has a distinct optimizer owner and frozen NNUEC/P/C components")
    names = _fields(spec["wire_to_parameter"], PARAMETER_FIELDS, "utility wire-to-parameter names")
    if len({_identifier(names[name]) for name in PARAMETER_FIELDS}) != 4:
        raise ValueError("utility parameter names must be distinct")
    if any(names[name].split(".", 1)[0] in ("base", *FROZEN_COMPONENTS) for name in PARAMETER_FIELDS):
        raise ValueError("frozen component parameter names cannot become utility members")
    parameters = _fields(spec["parameters"], PARAMETER_FIELDS, "utility private parameters")
    shapes = ((hidden, width), (hidden,), (7, hidden), (7,))
    for name, shape in zip(PARAMETER_FIELDS, shapes, strict=True):
        supplied = _fields(parameters[name], ("checkpoint_path", "shape", "dtype", "planned_requires_grad"),
                           "utility parameter spec")
        if (supplied["checkpoint_path"] != "parameters." + name or supplied["shape"] != list(shape)
                or any(type(dimension) is not int for dimension in supplied["shape"])
                or supplied["dtype"] != "float32" or supplied["planned_requires_grad"] is not True):
            raise ValueError("utility requires only its four exact FP32 private parameter arrays")
    return _copy(spec)


def parameter_spec_sha256(spec, frozen_base_config, optimizer_owner):
    checked = validate_parameter_membership(spec, frozen_base_config, optimizer_owner)
    return _digest(PREPARATION_DOMAIN, {"kind": "utility_parameter_spec", "value": checked})


def normalize_known_pairs(pairs):
    """Normalize a pure projected comparison list, without attesting its source.

    None is a masked entry. Repeated identical direction is counted as an
    occurrence but contributes once to a context's loss. Both directions of one
    comparison are contradictory and rejected. The raw owner adapter remains
    responsible for actual action support, complete cost and causal witnesses.
    """
    pairs = _sequence(pairs, MAX_PAIR_ENTRIES_PER_CONTEXT, "known route pair")
    counts, masked = {}, 0
    for pair in pairs:
        if pair is None:
            masked += 1
            continue
        if type(pair) not in (list, tuple) or len(pair) != 2:
            raise ValueError("known route comparison requires two indices")
        preferred, other = pair
        if (type(preferred) is not int or type(other) is not int or preferred not in ROUTE_ACTIONS
                or other not in ROUTE_ACTIONS or preferred == other):
            raise ValueError("known route comparison is restricted to distinct actions 4/5/6")
        if (other, preferred) in counts:
            raise ValueError("contradictory directions for the same route comparison")
        counts[(preferred, other)] = counts.get((preferred, other), 0) + 1
    ordered = sorted(counts, key=lambda pair: (min(pair), max(pair)))
    return {"pairs": [list(pair) for pair in ordered], "occurrence_counts": [counts[pair] for pair in ordered],
            "unique_known_pairs": len(ordered), "known_occurrences": sum(counts.values()),
            "masked_entries": masked, "input_entries": len(pairs), "execution_authorized": False}


def route_pairwise_loss(logits_rows, pairs_rows):
    """Scalar reference sums; this performs no tensor/model/optimizer operation."""
    logits_rows = _sequence(logits_rows, MAX_CONTEXTS, "utility logit row")
    pairs_rows = _sequence(pairs_rows, MAX_CONTEXTS, "utility pair row")
    if len(logits_rows) != len(pairs_rows):
        raise ValueError("utility logit and known-pair context counts differ")
    terms, occurrences, masked = [], 0, 0
    for logits, pairs in zip(logits_rows, pairs_rows, strict=True):
        if type(logits) not in (list, tuple) or len(logits) != 7:
            raise ValueError("seven original utility logits required")
        for value in logits:
            if type(value) not in (int, float) or abs(value) > _FP32_MAX or not math.isfinite(value):
                raise ValueError("finite FP32-range utility logits required")
        checked = normalize_known_pairs(pairs)
        occurrences += checked["known_occurrences"]
        masked += checked["masked_entries"]
        for preferred, other in checked["pairs"]:
            difference = float(logits[other]) - float(logits[preferred])
            terms.append(max(difference, 0.0) + math.log1p(math.exp(-abs(difference))))
    total, denominator = math.fsum(terms), len(terms)
    return {"loss_sum": total, "known_pair_denominator": denominator, "known_occurrences": occurrences,
            "masked_entries": masked, "normalized_loss": total / denominator if denominator else None,
            "loss_weight": "1.0", "performed_optimizer_updates": 0, "execution_authorized": False}


def _registration(value, spec, optimizer_owner, *, cost_scope=COST_SCOPE):
    _fields(value, ("collection_source_sha256", "trainer_source_sha256", "cost_epoch_sha256",
                    "resource_profile_sha256", "cost_scope", "objective", "frozen_base", "utility"),
            "utility source registration")
    _sha(value["collection_source_sha256"])
    _sha(value["trainer_source_sha256"])
    _sha(value["cost_epoch_sha256"])
    _sha(value["resource_profile_sha256"])
    if value["cost_scope"] != cost_scope or value["objective"] != COST_OBJECTIVE:
        raise ValueError("registered exclusive whole-route/drain cost scope and wall-nanosecond objective required")
    frozen = _fields(value["frozen_base"], ("schema", "semantics", "weights_sha256", "config"), "frozen V2 base")
    if frozen["schema"] != BASE_SCHEMA or frozen["semantics"] != BASE_SEMANTICS:
        raise ValueError("explicit NNUEC V2 frozen base semantics required")
    _sha(frozen["weights_sha256"])
    config = _base_config(frozen["config"])
    utility = _fields(value["utility"], ("schema", "semantics", "initial_weights_sha256", "base_weights_sha256",
                                        "parameter_spec_sha256"), "utility initial parameter declaration")
    if utility["schema"] != UTILITY_SCHEMA or utility["semantics"] != UTILITY_SEMANTICS:
        raise ValueError("explicit existing numerical utility semantics required; no promotion")
    _sha(utility["initial_weights_sha256"])
    _sha(utility["base_weights_sha256"])
    _sha(utility["parameter_spec_sha256"])
    if (utility["base_weights_sha256"] != frozen["weights_sha256"]
            or utility["parameter_spec_sha256"] != parameter_spec_sha256(spec, config, optimizer_owner)):
        raise ValueError("utility parameter declaration differs from its frozen base/spec")
    return _copy(value)


def _prepare_training(recipe, registration, spec, *, domain, cost_scope):
    checked_recipe = validate_recipe(recipe)
    owner = checked_recipe["optimizer_owner"]
    checked_registration = _registration(registration, spec, owner, cost_scope=cost_scope)
    config = checked_registration["frozen_base"]["config"]
    checked_spec = validate_parameter_membership(spec, config, owner)
    result = {
        "schema": domain, "status": "prepared_metadata_only",
        "recipe": checked_recipe, "recipe_sha256": recipe_sha256(checked_recipe),
        "registration": checked_registration,
        "registration_sha256": _digest(domain, {"kind": "source_registration", "value": checked_registration}),
        "parameter_spec": checked_spec, "parameter_spec_sha256": parameter_spec_sha256(checked_spec, config, owner),
        "optimizer_owner": owner, "prepared_artifacts": ["checkpoint_metadata", "export_metadata"],
        "tensor_membership_verified": False, "source_authority_verified": False,
        "optimizer_state_present": False, "checkpoint_tensor_present": False, "export_tensor_present": False,
        "performed_optimizer_updates": 0, "execution_authorized": False,
    }
    result["preparation_sha256"] = _digest(domain, result)
    return _copy(result)


def prepare_training(recipe, registration, spec):
    """Read/seal the original V1 declaration without retagging its cost scope."""
    return _prepare_training(recipe, registration, spec, domain=PREPARATION_DOMAIN, cost_scope=COST_SCOPE)


def prepare_cohort_training(recipe, registration, spec):
    """Seal the distinct cohort-only supervision domain, without live rights.

    Parameter membership and loss arithmetic have unchanged meanings. That
    does not make root-local cost observations a cohort target. The current
    parent must additionally verify original cohort/trial/physical-release
    evidence; this metadata parser cannot issue a known-pair capability.
    """
    return _prepare_training(recipe, registration, spec,
        domain=COHORT_PREPARATION_DOMAIN, cost_scope=COHORT_COST_SCOPE)


def validate_training_preparation(value):
    """Explicit V1/V2 parser dispatch; never fill, retag, or resign a record."""
    if type(value) is not dict:
        raise ValueError("explicit utility training preparation object required")
    schema = value.get("schema")
    if schema == PREPARATION_DOMAIN:
        checked = prepare_training(value.get("recipe"), value.get("registration"), value.get("parameter_spec"))
    elif schema == COHORT_PREPARATION_DOMAIN:
        checked = prepare_cohort_training(value.get("recipe"), value.get("registration"), value.get("parameter_spec"))
    else:
        raise ValueError("registered explicit utility preparation V1/V2 domain required")
    if _json(value) != _json(checked):
        raise ValueError("utility training preparation differs from its original exact codec")
    return checked
