"""Rovent submitted input /1, separate from Generation V3 input semantics.

Attack slots are sealed at submission, before a future counter or Repair
outcome exists. Only their original proposal-relative ply and optional selected
counter metadata are consumed here. Rust owns Rules, all eight divergence
features, and the exact selected-counter revision/line replay. This codec never
turns a future counter into a past selected record or grants native authority.
"""
from __future__ import annotations

import copy
import struct

from .generation_data import INPUT_FIELDS, TASKS, digest, fields, identifier, sha, uint
from .generation_data import _anchor, _moves, _validate_encoded_input, validate_view, view_digest

INPUT_DOMAIN = "rz-rovent-submitted-input/1"
SUBMITTED_INPUT_FIELDS = ("schema", *(name for name in INPUT_FIELDS if name != "divergence_record_ids"), "attack_mapping")


def _f32_bits(number):
    try:
        return struct.pack(">f", float(number)).hex()
    except (OverflowError, ValueError, struct.error) as error:
        raise ValueError("submitted model payload value cannot be represented as f32") from error


def _validate_attack_mapping(value):
    mapping = value["attack_mapping"]
    features = value["encoded_model_input"]["divergences"]
    if type(mapping) is not list or len(mapping) != len(features):
        raise ValueError("submitted attack slot/feature ordering mismatch")
    records, proposal = value["selected_view"]["records"], value["selected_view"]["query"]["proposal"]
    seen = set()
    for slot, (entry, row) in enumerate(zip(mapping, features)):
        fields(entry, ("slot", "attack_ply", "selected_counter_local_index", "selected_counter_revision"),
               "submitted attack slot")
        if uint(entry["slot"], 65535) != slot:
            raise ValueError("submitted attack slots must preserve exact 0-based order")
        ply = uint(entry["attack_ply"], 65535)
        if not 0 < ply < len(proposal) or ply > 255 or ply in seen:
            raise ValueError("submitted attack ply must be a unique interior proposal-relative ply in 1..255")
        seen.add(ply)
        if _f32_bits(row[0]) != _f32_bits(ply / 256.0):
            raise ValueError("submitted attack ply differs from original first FP32 feature")
        index, revision = entry["selected_counter_local_index"], entry["selected_counter_revision"]
        if (index is None) != (revision is None):
            raise ValueError("submitted selected counter index/revision must be a null pair")
        if index is not None:
            uint(index, 65535)
            if uint(revision) == 0:
                raise ValueError("submitted selected counter revision must be positive")
            query = value["selected_view"]["query"]
            if (index >= len(records) or records[index]["kind"] != "counter" or not query["counter_present"]
                    or records[index]["moves"] != query["counter"]):
                raise ValueError("submitted attack slot counter must already exist in the exact selected view/query")


def validate_input(value):
    """Check the new exact input and shared pure input contracts only."""
    fields(value, SUBMITTED_INPUT_FIELDS, "Rovent submitted input")
    if value["schema"] != INPUT_DOMAIN:
        raise ValueError("explicit Rovent submitted input domain required")
    for name in ("game_id", "opening_id", "line_genealogy_id", "question_id", "worker_id", "job_id", "slice_id"):
        identifier(value[name])
    sha(value["transposition_sha256"])
    if value["role"] not in ("proposer", "critic"):
        raise ValueError("V-free Rovent submitted input requires P/C role")
    for name in ("model_epoch", "context_epoch", "capture_sequence"):
        uint(value[name])
    _anchor(value["rules_anchor"])
    _moves(value["actual_history"], 16384)
    validate_view(value["selected_view"])
    if (value["selected_view"]["query"]["white_to_move"] != value["rules_anchor"]["white_to_move"]
            or sha(value["view_sha256"]) != view_digest(value["selected_view"])):
        raise ValueError("submitted exact query endpoint/view digest mismatch")
    if any(record["observed_sequence"] > value["capture_sequence"] for record in value["selected_view"]["records"]):
        raise ValueError("future observation leaked into submitted Rovent view")
    _moves(value["legal_moves"])
    if len(set(value["legal_moves"])) != len(value["legal_moves"]):
        raise ValueError("duplicate submitted legal candidate")
    mask = value["action_support_mask"]
    if type(mask) is not list or len(mask) != len(TASKS) or any(type(item) is not bool for item in mask):
        raise ValueError("invalid submitted action support mask")
    if any(mask):
        raise ValueError("V-free Rovent submitted action support mask must be entirely false")
    _validate_encoded_input(value)
    _validate_attack_mapping(value)
    return value


def seal_input(value):
    """Keep numeric model input intact; seal its exact FP32 projection in /1."""
    validate_input(value)
    projected = copy.deepcopy(value)
    payload = projected["encoded_model_input"]
    for anchor in payload["anchors"]:
        anchor["metadata"] = [_f32_bits(number) for number in anchor["metadata"]]
    payload["divergences"] = [[_f32_bits(number) for number in row] for row in payload["divergences"]]
    return digest(INPUT_DOMAIN, projected)


def submitted_collate_and_forward(model, role, snapshot):
    """Use the unchanged encoded payload only after new input validation.

    The trainer caller owns execution authorization. Validation and role joins
    happen before importing the tensor/model module; no future target or
    attack outcome is passed into this input-only collator.
    """
    validate_input(snapshot)
    if role != snapshot["role"]:
        raise ValueError("submitted model role differs from sealed input role")
    from .model import VirtualReviewTensorInput
    batch = VirtualReviewTensorInput.from_payload(snapshot["encoded_model_input"], device=next(model.parameters()).device)
    batch.validate(model.config)
    memory = model.public_encoder(*batch.public_args(model.config))
    output = model.role_graph(role)(*batch.role_args(memory, model.config))
    return batch, output
