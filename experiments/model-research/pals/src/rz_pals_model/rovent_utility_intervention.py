"""Closed /2 intervention metadata, separate from the original Ready cohort.

The declared arms are experimental interventions, not an observation that a
normal scheduler offered all recipients at once. Decoding or hashing metadata
cannot create a neutral target, role query, trial, native or learner authority.
"""
from __future__ import annotations

import copy
import hashlib
import json

from .generation_data import fields, sha, uint


PROFILE_DOMAIN = "rz-rovent-utility-comparison-cohort/2"
TARGET_DOMAIN = "rz-rovent-utility-intervention-target/1"
TRIAL_DOMAIN = "rz-rovent-utility-cohort-trial/2"
MATERIALIZATION_DOMAIN = "rz-rovent-utility-cohort-materialization/2"
PAIR_DOMAIN = "rz-rovent-utility-cohort-pair/2"
POLICY = "frozen_proposal_review_v1"
COST_SCOPE = "cohort_route_with_bounded_materialization_and_physical_pin_zero_excluding_durable_dataset_io_v1"
MAX_PROFILE_BYTES = 64 << 10
MAX_WINDOW_NS = 86_400_000_000_000
PROFILE_INPUT_FIELDS = (
    "input_lock_sha256", "source_sha256", "frozen_nnuec_model_sha256",
    "frozen_nnuec_namespace_sha256", "frozen_pc_model_sha256",
    "frozen_pc_namespace_sha256", "mode", "requested_recipients",
    "prelude_ns", "action_ns", "cleanup_ns", "logical_budget_sha256",
    "ready_policy_sha256", "cache_policy_sha256", "cost_class_sha256",
    "resource_plan_sha256", "resource_epoch", "usage", "cost_scope",
)
RECIPIENTS = ("proposer", "critic", "researcher")
ROUTES = ("route_p", "route_c", "route_r")


def canonical_bytes(value):
    """Canonical metadata spelling only; this is not a source/file pin."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")


def _positive(value):
    value = uint(value)
    if not value:
        raise ValueError("positive intervention unsigned integer required")
    return value


def validate_intervention_profile(value):
    """Validate the exact Root-owned /2 DTO without decoding a /1 object."""
    fields(value, ("domain", "input", "target_record", "policy"), "intervention profile/2")
    if value["domain"] != PROFILE_DOMAIN or value["policy"] != POLICY:
        raise ValueError("explicit intervention/2 domain and policy required")
    body = fields(value["input"], PROFILE_INPUT_FIELDS, "intervention original input")
    for name in PROFILE_INPUT_FIELDS:
        if name.endswith("_sha256"):
            sha(body[name])
    if body["mode"] not in ("cpu_pc", "cpu_pcr"):
        raise ValueError("intervention CPU-only mode required")
    recipients = body["requested_recipients"]
    if (type(recipients) is not list or not 2 <= len(recipients) <= 3
            or any(type(item) is not str or item not in RECIPIENTS for item in recipients)
            or len(set(recipients)) != len(recipients)
            or body["mode"] == "cpu_pc" and "researcher" in recipients):
        raise ValueError("two or three unique ordered supported intervention arms required")
    duration = sum(_positive(body[name]) for name in ("prelude_ns", "action_ns", "cleanup_ns"))
    if duration > MAX_WINDOW_NS:
        raise ValueError("intervention declared duration sum exceeds 24 hours")
    _positive(body["resource_epoch"])
    if body["usage"] not in ("validation_only", "collection_candidate") or body["cost_scope"] != COST_SCOPE:
        raise ValueError("explicit intervention usage and whole-route cost scope required")
    uint(value["target_record"], 63)
    if len(canonical_bytes(value)) > MAX_PROFILE_BYTES:
        raise ValueError("intervention canonical metadata exceeds 64KiB")
    return copy.deepcopy(value)


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate intervention JSON key")
        value[key] = item
    return value


def _nonfinite(_value):
    raise ValueError("finite intervention JSON required")


def parse_intervention_profile(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_PROFILE_BYTES:
        raise ValueError("nonempty bounded original intervention profile bytes required")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_object, parse_constant=_nonfinite)
    return validate_intervention_profile(value)


def intervention_profile_sha256(value):
    """Rust DTO domain + NUL + canonical bytes, without a length prefix."""
    value = validate_intervention_profile(value)
    return hashlib.sha256(PROFILE_DOMAIN.encode("ascii") + b"\0" + canonical_bytes(value)).hexdigest()
