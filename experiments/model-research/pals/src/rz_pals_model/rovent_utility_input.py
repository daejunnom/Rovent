"""Submitted utility input bytes and three original capture seals only.

The producer is ``NnuecUtilityInputCapture``. Its UTF8 input bytes, ordered
numeric target and view-bound submission target have distinct domains from
the shadow utility reference's output/numeric seals. Parsing this historical
data never reconstructs the original model, Request, Bound, Store or lease.
Exact Rules anchors and whole-bundle completeness remain Rust-owned.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import struct

from .generation_data import sha, uint
from .nnuec_reference import NnuecConfig
from .nnuec_utility_reference import ReferenceInput


INPUT_CAPTURE_DOMAIN = "rz-nnuec-submitted-utility-input/1"
TARGET_CAPTURE_DOMAIN = "rz-nnuec-submitted-utility-target/1"
SUBMISSION_TARGET_CAPTURE_DOMAIN = "rz-nnuec-submitted-utility-submission-target/1"
MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_SELECTED_RECORDS = 64
OBSERVATION_SCOPE = "submitted_utility_input_bytes_and_numeric_structure_only"


@dataclass(frozen=True)
class SubmittedUtilityInputObservation:
    """Detached immutable data observation, never an original producer cap."""

    input_json: str
    input_bytes: bytes
    reference_input: ReferenceInput
    config: NnuecConfig
    base_weights_sha256: str
    context_sha256: str
    target_ids: tuple
    view_seal_sha256: str
    input_sha256: str
    target_sha256: str
    submission_target_sha256: str

    def observation(self):
        return {
            "scope": OBSERVATION_SCOPE,
            "input_bytes": len(self.input_bytes),
            "base_weights_sha256": self.base_weights_sha256,
            "context_sha256": self.context_sha256,
            "target_ids": list(self.target_ids),
            "view_seal_sha256": self.view_seal_sha256,
            "input_sha256": self.input_sha256,
            "target_sha256": self.target_sha256,
            "submission_target_sha256": self.submission_target_sha256,
            "exact_rules_bundle_completeness": "requires_original_Rust_BoundFactory_evidence",
            "original_preparation_authority_reissued": False,
            "runtime_lease_authority": False,
            "execution_authorized": False,
            "training_target_authorized": False,
        }


def _object(pairs):
    value = {}
    for name, item in pairs:
        if name in value:
            raise ValueError("submitted utility input has duplicate JSON keys")
        value[name] = item
    return value


def _constant(_name):
    raise ValueError("submitted utility input has a nonfinite JSON constant")


def _original_input(input_json):
    if type(input_json) is not str:
        raise ValueError("submitted utility original JSON text required")
    if not 0 < len(input_json) <= MAX_INPUT_BYTES:
        raise ValueError("submitted utility original input byte bound")
    try:
        raw = input_json.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("submitted utility original input is not UTF8") from error
    if not 0 < len(raw) <= MAX_INPUT_BYTES:
        raise ValueError("submitted utility original input byte bound")
    try:
        value = json.loads(input_json, object_pairs_hook=_object, parse_constant=_constant)
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError("submitted utility original input is not bounded JSON") from error
    if type(value) is not dict:
        raise ValueError("submitted utility original input object required")
    return raw, value


def _targets(target_ids):
    if type(target_ids) not in (tuple, list) or not 1 <= len(target_ids) <= MAX_SELECTED_RECORDS:
        raise ValueError("submitted utility bounded original target IDs required")
    result = tuple(uint(item, MAX_SELECTED_RECORDS - 1) for item in target_ids)
    # The original producer enumerates selected metadata and filters it in
    # order. Do not sort a caller's changed target into an accepted target.
    if result != tuple(sorted(set(result))):
        raise ValueError("submitted utility original target order or duplicate differs")
    return result


def validate_submitted_utility_input(*, input_json, base_weights_sha256,
        context_sha256, target_ids, view_seal_sha256, input_sha256,
        target_sha256, submission_target_sha256, config):
    """Check actual capture framing without forward, Rules or live authority.

    ``config`` is the explicit source model configuration. No checkpoint is
    initialized, inferred or loaded here. Equal sparse board features do not
    prove equal Rules history or completeness of the original target bundle.
    """
    for pin in (base_weights_sha256, context_sha256, view_seal_sha256,
                input_sha256, target_sha256, submission_target_sha256):
        sha(pin)
    if type(config) is not NnuecConfig:
        raise ValueError("submitted utility explicit source NnuecConfig required")
    checked_config = NnuecConfig.from_mapping(config.to_mapping())
    ordered_ids = _targets(target_ids)
    raw, value = _original_input(input_json)
    checked = ReferenceInput.from_mapping(value, checked_config)
    if checked.target_ids != ordered_ids:
        raise ValueError("submitted utility original JSON target order differs from capture")
    records = checked.base_input.context_lines
    if (len(records) > MAX_SELECTED_RECORDS
            or tuple(record.id for record in records) != tuple(range(len(records)))):
        raise ValueError("submitted utility selected record IDs differ from original view order")
    # Capture builds this line from the root Position with an empty suffix.
    # Its actual board/history and root anchor are independently replayed by
    # Rust; this scalar guard does not claim that replay has happened.
    if checked.base_input.local_line.moves:
        raise ValueError("submitted utility original root local suffix must be empty")
    if checked.base_input.context_sha256 != context_sha256:
        raise ValueError("submitted utility numeric context seal differs from capture")

    input_state = hashlib.sha256(INPUT_CAPTURE_DOMAIN.encode("ascii") + b"\0")
    input_state.update(base_weights_sha256.encode("ascii") + b"\0")
    input_state.update(struct.pack("<Q", len(raw)))
    input_state.update(raw)
    if input_state.hexdigest() != input_sha256:
        raise ValueError("submitted utility original UTF8 input capture seal mismatch")

    target_state = hashlib.sha256(TARGET_CAPTURE_DOMAIN.encode("ascii") + b"\0")
    target_state.update(context_sha256.encode("ascii") + b"\0")
    target_state.update(struct.pack("<I", len(ordered_ids)))
    for identifier in ordered_ids:
        target_state.update(struct.pack("<Q", identifier))
    if target_state.hexdigest() != target_sha256:
        raise ValueError("submitted utility original ordered target capture seal mismatch")

    submission_state = hashlib.sha256(SUBMISSION_TARGET_CAPTURE_DOMAIN.encode("ascii") + b"\0")
    submission_state.update(bytes.fromhex(view_seal_sha256))
    submission_state.update(target_sha256.encode("ascii"))
    if submission_state.hexdigest() != submission_target_sha256:
        raise ValueError("submitted utility original view-bound submission target seal mismatch")

    return SubmittedUtilityInputObservation(input_json, raw, checked, checked_config,
        base_weights_sha256, context_sha256, ordered_ids, view_seal_sha256,
        input_sha256, target_sha256, submission_target_sha256)
