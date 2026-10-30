"""Strict persisted Collection CPU_T bindings, never live teacher authority.

Collection TeacherJobId is independent of reciprocal R QuestionEnvelope and
native ExecutionId. This module checks original data joins without making any
of those IDs or replaying Rules. Rust retains the original SubmittedInputSeal,
loaded InputLock, root owner and private worker dispatch/completion. Matching
JSON cannot replace those capabilities or authorize a new teacher go/label.
"""
from __future__ import annotations

import hashlib

from .generation_data import digest, fields, sha, uint, _anchor, _moves, _move16_uci
from .rovent_root_descriptor import _positive, validate_root_descriptor

COLLECTION_BINDING_DOMAIN = "rz-rovent-collection-teacher-binding/1"
FROZEN_ACTOR_DOMAIN = "rz-rovent-frozen-teacher-actor/1"

def _root_join(manifest, record, binding, *, native_domain):
    """Join one explicitly selected raw family; loaded lock stays Rust-owned."""
    from .rovent_native_v2 import _root_admission

    native = record["native"]
    submitted, retired = native["raw_submission"], native["raw_retirement"]
    if submitted.get("schema") != native_domain or retired.get("schema") != native_domain:
        revision = "/2" if native_domain == "rz-rovent-native-owner-event/2" else "/3"
        raise ValueError(f"Collection teacher descriptor requires original native {revision} root data")
    if native_domain == "rz-rovent-raw-record/3":
        from .rovent_external_root import validate_submitted_root
        validate_submitted_root(manifest, record, retired)
    else:
        _root_admission(manifest, record, retired)
    root = retired["root_admission"]
    descriptor, descriptor_sha = binding["root_descriptor"], binding["root_descriptor_sha256"]
    inp = validate_root_descriptor(descriptor, descriptor_sha)
    ticket, origin = submitted["ticket"], root["native_origin"]
    authority = inp["authority"]
    for key, original in (("epoch", "process_epoch"), ("game", "game_generation"), ("root", "root_generation")):
        if authority[key] != native[original] or root[original] != native[original]:
            raise ValueError("Collection teacher descriptor crossed original root authority")
    if (authority["implementation_sha256"] != ticket["implementation_sha256"]
            or inp["startup_lock_sha256"] != origin["input_lock_sha256"]
            or inp["state_and_history_sha256"] != root["root_state"]["semantic"]
            or descriptor["root_budget_sha256"] != root["root_budget_sha256"]
            or inp["clock"]["hard"] != {"clock_epoch": root["deadline_clock"], "at_ns": root["deadline_tick"]}
            or inp["allowed_root_moves"] != root["root_moves"]
            or any(inp["original_go"][key] != root[key] for key in ("requested_nodes", "depth_plies", "mate_moves"))
            or root.get("root_descriptor") != descriptor or root.get("root_descriptor_sha256") != descriptor_sha):
        raise ValueError("Collection teacher descriptor changed original lock/root/budget/go/deadline")
    _anchor(binding["position_anchor"])
    if binding["position_anchor"] != record["input"]["rules_anchor"]:
        raise ValueError("Collection teacher changed original submitted exact Rules endpoint")
    return inp


def validate_legacy_teacher_root(manifest, record, binding):
    """Explicit descriptor-bearing legacy variant; no old-body rewriting."""
    from .rovent_native_v2 import NATIVE_DOMAIN
    inp = _root_join(manifest, record, binding, native_domain=NATIVE_DOMAIN)
    question, hard = binding["question"], inp["clock"]["hard"]
    if (question["root_budget"] != binding["root_budget"]
            or question["root_budget"] != binding["root_descriptor"]["root_budget_sha256"]
            or question["deadline_clock"] != hard["clock_epoch"] or question["deadline_tick"] > hard["at_ns"]
            or binding["mode"] != record["native"]["raw_retirement"]["root_admission"]["native_origin"]["mode"]):
        raise ValueError("legacy teacher descriptor changed original question budget/deadline/mode")


def validate_collection_teacher_binding(manifest, record, actor_id, teacher_pin, binding):
    """Validate the original raw /2 entry only; do not infer another family."""
    from .rovent_native_v2 import NATIVE_DOMAIN
    _validate_collection_teacher_binding(manifest, record, actor_id, teacher_pin, binding, native_domain=NATIVE_DOMAIN)


def validate_collection_teacher_binding_v3(manifest, record, actor_id, teacher_pin, binding):
    """Validate raw /3 against its new submitted-input seal, never retagging it."""
    from .rovent_corpus_codec import SUBMITTED_CODEC, validate_record_codec
    from .rovent_native_v3 import NATIVE_DOMAIN
    from .rovent_submitted_input import validate_input, seal_input

    if validate_record_codec(manifest, record) is not SUBMITTED_CODEC:
        raise ValueError("Collection teacher raw /3 entry requires the submitted corpus family")
    snapshot = validate_input(record["input"])
    if sha(record["input_sha256"]) != seal_input(snapshot):
        raise ValueError("Collection teacher raw /3 original submitted input seal changed")
    _validate_collection_teacher_binding(manifest, record, actor_id, teacher_pin, binding, native_domain=NATIVE_DOMAIN)


def _validate_collection_teacher_binding(manifest, record, actor_id, teacher_pin, binding, *, native_domain):
    """Shared persisted-data joins only; this returns no execution capability."""
    fields(binding, ("schema", "job_id", "input_scope", "requested", "process", "root_restriction",
        "position_anchor", "root_moves", "soft_quantum_ns", "original_deadline", "root_descriptor", "root_descriptor_sha256"),
        "original Collection teacher dispatch binding")
    if binding["schema"] != COLLECTION_BINDING_DOMAIN:
        raise ValueError("explicit Collection teacher binding domain required")
    job = fields(binding["job_id"], ("collection_owner", "sequence"), "original Collection TeacherJobId")
    _positive(job["collection_owner"], 2**32 - 1)
    _positive(job["sequence"])
    scope = fields(binding["input_scope"], ("authority", "input_sha256", "view_sha256", "source_sha256", "role",
        "rules_semantic", "input_lock", "root_admission", "root_budget", "frozen_actor"), "original Collection teacher input scope")
    authority = fields(scope["authority"], ("process_epoch", "game_generation", "root_generation", "implementation_sha256"),
                       "original Collection teacher source authority")
    native, snapshot = record["native"], record["input"]
    for key in ("process_epoch", "game_generation", "root_generation"):
        if _positive(authority[key]) != native[key]:
            raise ValueError("Collection teacher input crossed original source authority")
    if sha(authority["implementation_sha256"]) != native["raw_submission"]["ticket"]["implementation_sha256"]:
        raise ValueError("Collection teacher input changed original implementation")
    for key in set(scope) - {"authority", "role"}:
        sha(scope[key])
    frozen = digest(FROZEN_ACTOR_DOMAIN, {"actor_id": actor_id, "teacher_pin": teacher_pin})
    if (scope["input_sha256"] != record["input_sha256"] or scope["view_sha256"] != snapshot["view_sha256"]
            or scope["source_sha256"] != manifest["source_sha256"]
            or scope["role"] not in ("proposer", "critic") or scope["role"] != record["role"]
            or scope["rules_semantic"] != snapshot["rules_anchor"]["rules_state_sha256"]
            or scope["frozen_actor"] != frozen):
        raise ValueError("Collection teacher changed original submitted input/view/source/role/Rules/frozen actor")
    if any(scope[key] == "0" * 64 for key in ("input_lock", "root_admission", "root_budget", "frozen_actor")):
        raise ValueError("Collection teacher original root/actor scope cannot be zero")
    inp = _root_join(manifest, record, binding, native_domain=native_domain)
    if (scope["input_lock"] != inp["startup_lock_sha256"]
            or scope["root_admission"] != binding["root_descriptor_sha256"]
            or scope["root_budget"] != binding["root_descriptor"]["root_budget_sha256"]):
        raise ValueError("Collection teacher scope changed original InputLock/root descriptor/budget")
    deadline = fields(binding["original_deadline"], ("clock_epoch", "at_ns"), "original Collection teacher Job deadline")
    hard = inp["clock"]["hard"]
    if uint(deadline["clock_epoch"]) != hard["clock_epoch"] or uint(deadline["at_ns"]) > hard["at_ns"]:
        raise ValueError("Collection teacher Job deadline crossed original root deadline")
    _positive(binding["soft_quantum_ns"])
    roots = binding["root_moves"]
    framed = b"rz-external-root-restriction/1\0" + bytes([roots is not None])
    if roots is not None:
        _moves(roots, 218)
        if not roots or len(set(roots)) != len(roots) or any(move not in snapshot["legal_moves"] for move in roots):
            raise ValueError("Collection teacher root restriction contains duplicate/illegal candidate")
        framed += len(roots).to_bytes(8, "little")
        for move in roots:
            raw = _move16_uci(move).encode("ascii")
            framed += len(raw).to_bytes(8, "little") + raw
    if hashlib.sha256(framed).hexdigest() != sha(binding["root_restriction"]):
        raise ValueError("Collection teacher original root restriction seal changed")

