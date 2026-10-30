"""Historical checkpoint integrity projection, with no live authority recovery.

Only the explicitly supplied receipt bytes are decoded. Historical paths are
retained as strings and are never opened or resolved. The caller separately
owns verified checkpoint-byte acquisition and computes observed state pins from
its scratch tensor loader. Matching metadata does not reclaim an old process,
prove physical closure, issue a native capability, or authorize new execution.
"""
from __future__ import annotations

import copy
import hashlib
import json

RECEIPT_DOMAIN = "rz-rovent-original-checkpoint-retirement/1"
CHECKPOINT_DOMAIN = "rz-rovent-trainer-checkpoint/1"
RECOVERED_RECEIPT_DOMAIN = "rz-rovent-recovered-checkpoint-retirement/1"
RECOVERED_CHECKPOINT_DOMAIN = "rz-rovent-trainer-checkpoint/2"
PROJECTION_DOMAIN = "rz-rovent-historical-checkpoint-projection/1"
STATE_HASH_FIELDS = ("sampler_state_sha256", "model_state_sha256", "optimizer_state_sha256",
                     "scheduler_state_sha256", "rng_state_sha256")
_ROLES = ("proposer", "critic", "validator")
_COSTS = ("forward_calls", "backward_calls", "samples", "wall_time_ns", "partial_accumulation_ns",
          "replay_ns", "masked_groups", "output_bytes")
_UINT64 = 2**64 - 1
_MAX_RECEIPT_BYTES = 1024 * 1024


def _fields(value, names, description):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError(f"{description} fields mismatch")
    return value


def _uint(value, maximum=_UINT64):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("bounded historical checkpoint unsigned integer required")
    return value


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("lowercase historical checkpoint SHA-256 required")
    return value


def _text(value, maximum, description):
    if (type(value) is not str or not value.strip() or len(value.encode("utf-8")) > maximum
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)):
        raise ValueError(f"bounded historical {description} text required")
    return value


def _checkpoint_pin(value):
    _fields(value, ("path", "sha256", "bytes"), "historical/recovered checkpoint pin")
    _text(value["path"], 4096, "checkpoint path")
    _sha(value["sha256"])
    if _uint(value["bytes"]) == 0:
        raise ValueError("nonempty historical/recovered checkpoint byte pin required")
    return value


def _counts(value):
    _fields(value, _ROLES, "historical completed optimizer updates")
    for count in value.values():
        _uint(count)
    return value


def _cost(value):
    _fields(value, _COSTS, "historical checkpoint snapshot cost")
    for count in value.values():
        _uint(count)
    return value


def _state(value, *, checkpoint_envelope=False, checkpoint_domain=CHECKPOINT_DOMAIN):
    extras = ("schema", "checkpoint") if checkpoint_envelope else ()
    auxiliary = type(value) is dict and "r_auxiliary" in value
    if auxiliary and checkpoint_envelope and checkpoint_domain != RECOVERED_CHECKPOINT_DOMAIN:
        raise ValueError("legacy checkpoint retirement cannot acquire R auxiliary state")
    _fields(value, (*extras, *STATE_HASH_FIELDS, "completed_updates", "cost", "pending_microbatches")
            + (("r_auxiliary",) if auxiliary else ()),
            "historical trainer checkpoint" if checkpoint_envelope else "observed recovered tensor state")
    if auxiliary:
        from .rovent_r_auxiliary import validate_checkpoint_projection
        validate_checkpoint_projection(value["r_auxiliary"])
    if checkpoint_envelope:
        if value["schema"] != checkpoint_domain:
            raise ValueError("original trainer checkpoint receipt domain required")
        _checkpoint_pin(value["checkpoint"])
    for field in STATE_HASH_FIELDS:
        _sha(value[field])
    _counts(value["completed_updates"])
    _cost(value["cost"])
    if type(value["pending_microbatches"]) is not int or value["pending_microbatches"] != 0:
        raise ValueError("historical checkpoint needs a completed optimizer boundary")
    return value


def _strict_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate historical checkpoint JSON field")
        result[name] = value
    return result


def _nonfinite(_):
    raise ValueError("nonfinite historical checkpoint JSON rejected")


def parse_historical_checkpoint_receipt(raw_receipt, *, receipt_sha256, receipt_bytes):
    """Check the original raw byte pin before strict receipt/schema decoding.

    The result is plain historical data. No path in it is read, and the result
    cannot replace the caller's original source or current runtime authority.
    """
    _sha(receipt_sha256)
    if type(raw_receipt) is not bytes or not 1 <= _uint(receipt_bytes, _MAX_RECEIPT_BYTES):
        raise ValueError("bounded immutable historical checkpoint receipt bytes required")
    if len(raw_receipt) != receipt_bytes or hashlib.sha256(raw_receipt).hexdigest() != receipt_sha256:
        raise ValueError("historical checkpoint receipt raw byte pin mismatch")
    try:
        value = json.loads(raw_receipt.decode("utf-8"), object_pairs_hook=_strict_object, parse_constant=_nonfinite)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError("strict UTF-8 historical checkpoint receipt JSON required") from error
    recovered = type(value) is dict and value.get("schema") == RECOVERED_RECEIPT_DOMAIN
    _fields(value, ("schema", "execution_id", "window_id", "checkpoint")
            + (("source_kind", "past_native_authority_reissued") if recovered else ()),
            "original checkpoint retirement receipt")
    if value["schema"] not in (RECEIPT_DOMAIN, RECOVERED_RECEIPT_DOMAIN):
        raise ValueError("original historical checkpoint retirement domain required")
    if recovered and (value["source_kind"] != "historical_exact_pins" or value["past_native_authority_reissued"] is not False):
        raise ValueError("recovered retirement cannot reissue historical native authority")
    _text(value["execution_id"], 256, "execution id")
    _text(value["window_id"], 256, "window id")
    _state(value["checkpoint"], checkpoint_envelope=True,
           checkpoint_domain=RECOVERED_CHECKPOINT_DOMAIN if recovered else CHECKPOINT_DOMAIN)
    return copy.deepcopy(value)


def project_historical_checkpoint(raw_receipt, *, receipt_sha256, receipt_bytes,
                                  recovered_checkpoint_pin, observed_state):
    """Join historical receipt to relocated bytes and a scratch-state summary.

    Only SHA-256 and byte length bind the old checkpoint to the recovered one;
    path equality would incorrectly forbid verified relocation. Snapshot cost
    is compared exactly. It is not a total of unknown post-snapshot expenses.
    """
    receipt = parse_historical_checkpoint_receipt(raw_receipt, receipt_sha256=receipt_sha256,
                                                 receipt_bytes=receipt_bytes)
    recovered = _checkpoint_pin(recovered_checkpoint_pin)
    historical = receipt["checkpoint"]
    if (historical["checkpoint"]["sha256"] != recovered["sha256"]
            or historical["checkpoint"]["bytes"] != recovered["bytes"]):
        raise ValueError("historical checkpoint differs from recovered exact hash/bytes")
    observed = _state(observed_state)
    for field in STATE_HASH_FIELDS:
        if historical[field] != observed[field]:
            raise ValueError(f"historical checkpoint {field} differs from recovered tensor scratch")
    if historical["completed_updates"] != observed["completed_updates"]:
        raise ValueError("historical completed_updates differ from recovered tensor scratch")
    if historical["cost"] != observed["cost"]:
        raise ValueError("historical snapshot cost differs from recovered tensor scratch")
    if ("r_auxiliary" in historical) != ("r_auxiliary" in observed) or historical.get("r_auxiliary") != observed.get("r_auxiliary"):
        raise ValueError("historical R auxiliary tensor projection differs from recovered scratch")
    # Both pending values have independently passed the strict integer-zero
    # boundary check. The result deliberately makes no old-run drain claim.
    result = {"schema": PROJECTION_DOMAIN, "status": "historical_integrity_only",
            "receipt_pin": {"sha256": receipt_sha256, "bytes": receipt_bytes},
            "historical_execution_id": receipt["execution_id"], "historical_window_id": receipt["window_id"],
            "historical_checkpoint_pin": copy.deepcopy(historical["checkpoint"]),
            "recovered_checkpoint_pin": copy.deepcopy(recovered),
            **{field: historical[field] for field in STATE_HASH_FIELDS},
            "completed_updates": copy.deepcopy(historical["completed_updates"]),
            "snapshot_cost": copy.deepcopy(historical["cost"]), "pending_microbatches": 0,
            "past_native_authority_reissued": False, "execution_authorized": False,
            "physical_completion_scope": "checkpoint_does_not_prove_old_run_closure"}
    if "r_auxiliary" in historical:
        result["r_auxiliary"] = copy.deepcopy(historical["r_auxiliary"])
    return result
