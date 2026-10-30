"""Separate immutable source declarations from mutable NNUEC tensor history.

This standard-library metadata consumer never imports a model, issues an
optimizer update, verifies an original process capability, or grants export
eligibility. Journal-shaped JSON and checkpoint pins remain evidence to join
with the caller's opaque original learner/checkpoint authority.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

STATE_DOMAIN = "rz-rovent-nnuec-tensor-training-state/1"
ANCESTOR_DOMAIN = "rz-rovent-nnuec-tensor-training-ancestor/1"
SOURCE_DOMAIN = "rz-rovent-nnuec-initial-source-declaration/1"
_JOURNAL_DOMAIN = "rz-rovent-original-learner-journal/1"
_UPDATE_DOMAIN = "rz-rovent-original-learner-update/1"
_RECOVERED_JOURNAL_DOMAIN = "rz-rovent-recovered-learner-journal/1"
_RECOVERED_UPDATE_DOMAIN = "rz-rovent-recovered-learner-update/1"
_HEADS = ("policy", "wdl", "divergence", "task", "nnuec_wdl")
_ROLES = ("proposer", "critic", "validator")
_COSTS = ("forward_calls", "backward_calls", "samples", "wall_time_ns", "partial_accumulation_ns",
          "replay_ns", "masked_groups", "output_bytes")
_CLASSES = ("unmodified_from_source", "modified_requires_original_authority", "modified_unverified")
_UINT64 = 2**64 - 1
_MAX_PARAMETERS = 256
_MAX_EVENTS = 262144
_MAX_METADATA_BYTES = 64 * 1024 * 1024


def _fields(value, names, description):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError(f"{description} fields mismatch")
    return value


def _uint(value, maximum=_UINT64):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("bounded NNUEC unsigned integer required")
    return value


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("lowercase NNUEC SHA-256 pin required")
    return value


def _identifier(value, maximum=256):
    if (type(value) is not str or not value.strip() or len(value.encode("utf-8")) > maximum
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)):
        raise ValueError("bounded NNUEC metadata identifier required")
    return value


def _plain(value):
    """Bound JSON-shaped input before copying or canonical hash traversal."""
    active, nodes, text_bytes = set(), 0, 0

    def visit(item, depth):
        nonlocal nodes, text_bytes
        nodes += 1
        if nodes > 1048576 or depth > 64:
            raise ValueError("NNUEC metadata aggregate/depth budget exceeded")
        if item is None or type(item) is bool:
            return
        if type(item) is int:
            if not -(2**63) <= item <= _UINT64:
                raise ValueError("NNUEC metadata integer budget exceeded")
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("NNUEC metadata must be finite")
            return
        if type(item) is str:
            length = len(item.encode("utf-8"))
            text_bytes += length
            if length > 1024 * 1024 or text_bytes > _MAX_METADATA_BYTES:
                raise ValueError("NNUEC metadata string/aggregate budget exceeded")
            return
        if type(item) not in (dict, list):
            raise ValueError("plain JSON NNUEC metadata required")
        if len(item) > _MAX_EVENTS or id(item) in active:
            raise ValueError("NNUEC metadata container/cycle budget exceeded")
        active.add(id(item))
        try:
            if type(item) is dict:
                for key, child in item.items():
                    if type(key) is not str or len(key.encode("utf-8")) > 1024:
                        raise ValueError("bounded NNUEC metadata JSON key required")
                    visit(key, depth + 1)
                    visit(child, depth + 1)
            else:
                for child in item:
                    visit(child, depth + 1)
        finally:
            active.remove(id(item))

    visit(value, 0)


def _digest(domain, value):
    hashed, total = hashlib.sha256(), 0
    encoder = json.JSONEncoder(sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for chunk in encoder.iterencode([domain, value]):
        raw = chunk.encode("utf-8")
        total += len(raw)
        if total > _MAX_METADATA_BYTES:
            raise ValueError("NNUEC canonical metadata byte budget exceeded")
        hashed.update(raw)
    return hashed.hexdigest()


def _source(value):
    if type(value) is not dict:
        raise ValueError("immutable initial NNUEC source declaration required")
    if value.get("kind") == "untrained":
        _fields(value, ("kind",), "initial untrained NNUEC declaration")
    elif value.get("kind") == "learned":
        _fields(value, ("kind", "run_id", "steps", "dataset_sha256"), "initial learned NNUEC declaration")
        _identifier(value["run_id"], 128)
        if _uint(value["steps"]) == 0:
            raise ValueError("initial learned NNUEC declaration needs positive declared steps")
        _sha(value["dataset_sha256"])
    else:
        raise ValueError("explicit initial untrained/learned NNUEC declaration required")
    return copy.deepcopy(value)


def _parameter_counts(value):
    if type(value) is not dict or len(value) > _MAX_PARAMETERS:
        raise ValueError("bounded NNUEC-only parameter update map required")
    result = {}
    for name, count in value.items():
        _identifier(name)
        if (not name.startswith("nnuec.") or any(not part or any(char not in
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for char in part)
                for part in name.split("."))):
            raise ValueError("explicit nnuec parameter namespace required")
        result[name] = _uint(count)
    return dict(sorted(result.items()))


def _head_counts(value):
    _fields(value, _HEADS, "NNUEC original learner head counts")
    return {name: _uint(value[name], 4096) for name in _HEADS}


def _journal_groups(value):
    """Validate original-shaped journal data, without verifying its authority."""
    recovered = type(value) is dict and value.get("schema") == _RECOVERED_JOURNAL_DOMAIN
    auxiliary = type(value) is dict and "r_auxiliary" in value
    if auxiliary and not recovered:
        raise ValueError("legacy NNUEC journal cannot acquire an auxiliary stream")
    _fields(value, ("schema", "recipe_sha256", "stage_id", "profile_sha256", "phase_inputs", "updates", "head_counts",
                    "frozen_pc_sha256", "resume_cost_receipts") + (("recovery_cost_state",) if recovered else ())
            + (("r_auxiliary",) if auxiliary else ())
            + (("r_auxiliary_ancestry",) if recovered and "r_auxiliary_ancestry" in value else ()),
            "NNUEC original learner journal")
    if "r_auxiliary_ancestry" in value:
        from .rovent_r_ancestry import ancestry_from_journal
        ancestry_from_journal(value)
    if value["schema"] not in (_JOURNAL_DOMAIN, _RECOVERED_JOURNAL_DOMAIN):
        raise ValueError("explicit original learner journal domain required")
    if recovered and value["recovery_cost_state"] is not None:
        from .rovent_training_recovery import validate_recovered_cost
        validate_recovered_cost(value["recovery_cost_state"])
    _sha(value["recipe_sha256"])
    _sha(value["profile_sha256"])
    _identifier(value["stage_id"])
    phases = value["phase_inputs"]
    if type(phases) is not dict or not phases or set(phases) - {"pc", "v"}:
        raise ValueError("bounded original learner phases required")
    for phase, source in phases.items():
        _fields(source, ("corpus_sha256", "nnuec_model_sha256", "model_epoch", "context_epoch", "samplers")
                + (("training_model_epoch",) if recovered else ()),
                "NNUEC original collection phase")
        _sha(source["corpus_sha256"])
        _sha(source["nnuec_model_sha256"])
        _uint(source["model_epoch"])
        _uint(source["context_epoch"])
        if recovered:
            _uint(source["training_model_epoch"])
        roles = {"validator"} if phase == "v" else {"proposer", "critic"}
        if type(source["samplers"]) is not dict or set(source["samplers"]) - roles:
            raise ValueError("original learner phase has a foreign role sampler")
    updates = value["updates"]
    if type(updates) is not list or len(updates) > _MAX_EVENTS:
        raise ValueError("bounded original learner update events required")
    counted, heads, groups, receipt_bytes = dict.fromkeys(_ROLES, 0), dict.fromkeys(_HEADS, 0), 0, 0
    for event in updates:
        _fields(event, ("schema", "window_id", "stage_id", "role", "completed_role_updates", "profile_sha256", "corpus_sha256",
            "phase", "sample_origins", "sample_inputs", "head_counts", "actual_model_state_sha256", "original_completion")
            + (("r_auxiliary",) if auxiliary and event.get("role") != "validator" else ()),
            "NNUEC original learner event")
        role = event["role"]
        if type(role) is not str or role not in _ROLES:
            raise ValueError("registered original learner event role required")
        counted[role] += 1
        phase = "v" if role == "validator" else "pc"
        if (event["schema"] != (_RECOVERED_UPDATE_DOMAIN if recovered else _UPDATE_DOMAIN) or event["stage_id"] != value["stage_id"]
                or _uint(event["completed_role_updates"]) != counted[role] or event["phase"] != phase
                or phase not in phases or event["profile_sha256"] != value["profile_sha256"]
                or event["corpus_sha256"] != phases[phase]["corpus_sha256"]):
            raise ValueError("original learner event role/phase/order/source mismatch")
        _identifier(event["window_id"])
        _sha(event["actual_model_state_sha256"])
        for name in ("sample_origins", "sample_inputs"):
            values = event[name]
            if type(values) is not list or len(values) != 8:
                raise ValueError("original learner event requires eight sealed samples")
            for identity in values:
                _sha(identity)
        event_heads = _head_counts(event["head_counts"])
        if (not any(event_heads.values()) or role == "validator" and any(event_heads[head] for head in _HEADS if head != "task")
                or role != "validator" and event_heads["task"] or role == "proposer" and event_heads["divergence"]):
            raise ValueError("original learner event violates valid private head routing")
        for head, count in event_heads.items():
            heads[head] = _uint(heads[head] + count)
        if not auxiliary and role in ("proposer", "critic") and event_heads["nnuec_wdl"] > 0:
            groups += 1
        receipt = _fields(event["original_completion"], ("sha256", "bytes"), "original learner receipt pin")
        _sha(receipt["sha256"])
        if _uint(receipt["bytes"], 2 * 1024 * 1024 * 1024) == 0:
            raise ValueError("nonempty bounded original learner receipt required")
        receipt_bytes = _uint(receipt_bytes + receipt["bytes"])
    _fields(value["head_counts"], _HEADS, "cumulative original learner head counts")
    if {head: _uint(value["head_counts"][head]) for head in _HEADS} != heads:
        raise ValueError("original learner cumulative head counts disagree")
    if value["frozen_pc_sha256"] is not None:
        _sha(value["frozen_pc_sha256"])
    receipts = value["resume_cost_receipts"]
    if type(receipts) is not list or len(receipts) > 4096:
        raise ValueError("bounded original learner resume receipts required")
    for receipt in receipts:
        _fields(receipt, ("checkpoint_sha256", "original_receipt_sha256", "cost_delta"), "original learner resume receipt")
        _sha(receipt["checkpoint_sha256"])
        _sha(receipt["original_receipt_sha256"])
        for count in _fields(receipt["cost_delta"], _COSTS, "original learner retained cost delta").values():
            _uint(count)
    if auxiliary:
        from .rovent_r_auxiliary import validate_auxiliary_journal
        groups = validate_auxiliary_journal(value["r_auxiliary"], updates)["current_nnuec_updates"]
    return groups


def validate_nnuec_ancestor(ancestor):
    """Check a compact parent description, never reconstruct parent authority."""
    if ancestor is None:
        return None
    _plain(ancestor)
    _fields(ancestor, ("schema", "parent_state_sha256", "parent_checkpoint_sha256", "source_declaration_sha256",
                       "classification", "known_updates", "actual_training_authority_verified"), "compact NNUEC ancestor")
    if ancestor["schema"] != ANCESTOR_DOMAIN or ancestor["actual_training_authority_verified"] is not False:
        raise ValueError("compact NNUEC ancestry cannot claim actual training authority")
    for name in ("parent_state_sha256", "parent_checkpoint_sha256", "source_declaration_sha256"):
        _sha(ancestor[name])
    _classification(ancestor["classification"], ancestor["known_updates"])
    return copy.deepcopy(ancestor)


def _classification(kind, known):
    if type(kind) is not str or kind not in _CLASSES:
        raise ValueError("registered NNUEC tensor training classification required")
    if kind == "modified_unverified":
        if known is not None:
            raise ValueError("unverified NNUEC updates must remain unknown")
    elif kind == "unmodified_from_source":
        if type(known) is not int or known != 0:
            raise ValueError("unmodified NNUEC history must have zero tracked updates")
    elif _uint(known) == 0:
        raise ValueError("modified NNUEC history needs positive tracked updates")


def _aggregate(ancestor, current_known):
    previous = 0 if ancestor is None else ancestor["known_updates"]
    if previous is None or current_known is None:
        return "modified_unverified", None
    total = previous + current_known
    if total > _UINT64:
        raise ValueError("NNUEC tracked update total exceeds uint64")
    return ("modified_requires_original_authority" if total else "unmodified_from_source"), total


def derive_nnuec_tensor_training_state(*, source_declaration, parameter_updates, samplers, ancestor=None):
    """Derive lifecycle metadata only; initial learned declarations are not added.

    Parameter counts are NNUEC-only current-stage AdamW steps. The caller owns
    actual tensor membership, optimizer validation, and original journal proof.
    An untrained export also needs an initially untrained source declaration;
    ``nnuec_is_unmodified`` alone does not erase an initially learned asset.
    """
    _plain(source_declaration)
    _plain(parameter_updates)
    _plain(samplers)
    source, counts = _source(source_declaration), _parameter_counts(parameter_updates)
    parent = validate_nnuec_ancestor(ancestor)
    if parent is not None and parent["source_declaration_sha256"] != _digest(SOURCE_DOMAIN, source):
        raise ValueError("NNUEC fork cannot change its immutable initial source declaration")
    if type(samplers) is not dict:
        raise ValueError("plain NNUEC learner sampler state required")
    maximum = max(counts.values(), default=0)
    if samplers.get("schema") in (_JOURNAL_DOMAIN, _RECOVERED_JOURNAL_DOMAIN):
        groups = _journal_groups(samplers)
        if maximum != groups:
            raise ValueError("NNUEC valid journal update groups differ from maximum parameter steps")
        journal_sha = _digest(samplers["schema"], samplers)
    else:
        groups, journal_sha = (None if maximum else 0), None
    classification, known = _aggregate(parent, groups)
    return {"schema": STATE_DOMAIN, "source_declaration_scope": "initial_source_asset_only", "source_declaration": source,
            "ancestor": parent, "current": {"nnuec_parameter_update_counts": counts, "valid_update_groups": groups,
                                           "journal_sha256": journal_sha},
            "classification": classification, "known_updates": known, "actual_training_authority_verified": False}


def _state(value):
    _plain(value)
    _fields(value, ("schema", "source_declaration_scope", "source_declaration", "ancestor", "current", "classification",
                    "known_updates", "actual_training_authority_verified"), "NNUEC tensor training state")
    if (value["schema"] != STATE_DOMAIN or value["source_declaration_scope"] != "initial_source_asset_only"
            or value["actual_training_authority_verified"] is not False):
        raise ValueError("NNUEC tensor metadata cannot claim actual training authority or mutable source scope")
    source, ancestor = _source(value["source_declaration"]), validate_nnuec_ancestor(value["ancestor"])
    if ancestor is not None and ancestor["source_declaration_sha256"] != _digest(SOURCE_DOMAIN, source):
        raise ValueError("NNUEC ancestor changed immutable initial source declaration")
    current = _fields(value["current"], ("nnuec_parameter_update_counts", "valid_update_groups", "journal_sha256"),
                      "current NNUEC tensor update summary")
    counts = _parameter_counts(current["nnuec_parameter_update_counts"])
    maximum, groups, journal_sha = max(counts.values(), default=0), current["valid_update_groups"], current["journal_sha256"]
    if journal_sha is not None:
        _sha(journal_sha)
        _uint(groups, _MAX_EVENTS)
        if maximum != groups:
            raise ValueError("NNUEC journal summary differs from maximum parameter steps")
    elif maximum:
        if groups is not None:
            raise ValueError("modified NNUEC without original journal must remain unverified")
    elif type(groups) is not int or groups != 0:
        raise ValueError("unmodified current NNUEC parameters need zero update groups")
    expected_class, expected_known = _aggregate(ancestor, groups)
    _classification(value["classification"], value["known_updates"])
    if value["classification"] != expected_class or value["known_updates"] != expected_known:
        raise ValueError("NNUEC tensor classification disagrees with current/ancestor history")
    return copy.deepcopy(value)


def fork_nnuec_ancestor(state, parent_checkpoint_sha256):
    """Retain compact parent history across a reset; no recursive state copy."""
    checked = _state(state)
    return {"schema": ANCESTOR_DOMAIN, "parent_state_sha256": _digest(STATE_DOMAIN, checked),
            "parent_checkpoint_sha256": _sha(parent_checkpoint_sha256),
            "source_declaration_sha256": _digest(SOURCE_DOMAIN, checked["source_declaration"]),
            "classification": checked["classification"], "known_updates": checked["known_updates"],
            "actual_training_authority_verified": False}


def nnuec_is_unmodified(state):
    """Whether this tensor lineage changed its source; not an export authority."""
    return _state(state)["classification"] == "unmodified_from_source"


def nnuec_known_updates(state):
    """Tracked tensor-lineage update count, or unknown; not source-declared steps."""
    return _state(state)["known_updates"]
