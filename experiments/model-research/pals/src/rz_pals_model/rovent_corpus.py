"""Bounded original Rovent corpus intake; this reader grants no execution right.

Rust owns Rules, native retirement and external-teacher process lifetimes. This
module verifies recovered bytes and their joins without turning JSON claims
into a live producer capability, without reconstructing Rules, and without
importing a tensor framework. Legacy/diagnostic records are not converted.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
from types import SimpleNamespace
from collections.abc import Sequence

from .generation_data import digest, fields, identifier, sha, uint, validate_input, seal_input
from .generation_data import wdl_target, divergence_target, task_pairs, trajectory_digest
from .rovent_generation import GlobalLineageIndex, SeriesRecipe, index_original_input
from .rovent_corpus_codec import (LEGACY_CODEC, SUBMITTED_CODEC, codec_from_manifest,
    validate_container_pair, validate_record_codec)

MANIFEST_DOMAIN = "rz-rovent-corpus-manifest/1"
RECORD_DOMAIN = "rz-rovent-training-record/1"
INDEX_DOMAIN = "rz-rovent-corpus-index/1"
ORIGIN_DOMAIN = "rz-rovent-training-origin/1"
HEAD_DOMAIN = "rz-rovent-head-source/1"
NATIVE_RAW_DOMAIN = "rz-rovent-native-owner-event/1"
NNUEC_INPUT_DOMAIN = "rz-rovent-nnuec-reference-input/1"


def _sha_list(value, maximum, description):
    if (type(value) is not list or len(value) > maximum
            or value != sorted(set(sha(item) for item in value))):
        raise ValueError(f"bounded sorted unique {description} required")
    return value


def _manifest(value, limits, *, codec=LEGACY_CODEC):
    fields(value, ("schema", "corpus_id", "source_commit", "source_sha256", "runtime_sha256",
                   "model_semantics_sha256", "model_epoch", "context_epoch", "recipe_sha256",
                   "stage_id", "round", "round_id", "student", "learner", "one_learner", "inherited_input_sha256",
                   "inherited_origin_sha256", "teachers", "opening_splits", "ancestor_index_sha256"),
           "original Rovent corpus manifest")
    if codec not in (LEGACY_CODEC, SUBMITTED_CODEC) or value["schema"] != codec.manifest_domain or value["one_learner"] is not True:
        raise ValueError("new single-learner Rovent corpus domain required")
    for name in ("corpus_id", "stage_id", "round_id", "student", "learner"):
        identifier(value[name])
    uint(value["round"])
    revision = value["source_commit"]
    if type(revision) is not str or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("corpus must pin an original full source commit")
    for name in ("source_sha256", "runtime_sha256", "model_semantics_sha256", "recipe_sha256"):
        sha(value[name])
    for name in ("model_epoch", "context_epoch"):
        if uint(value[name]) == 0:
            raise ValueError("positive original model/context epochs required")
    for name in ("inherited_input_sha256", "inherited_origin_sha256"):
        _sha_list(value[name], 262144, name)
    _sha_list(value["ancestor_index_sha256"], 256, "ancestor indices")
    teachers = value["teachers"]
    if type(teachers) is not dict or len(teachers) > 16:
        raise ValueError("bounded frozen corpus teachers required")
    for actor, pin in teachers.items():
        identifier(actor)
        _teacher_pin(pin)
    splits = value["opening_splits"]
    if type(splits) is not dict or not 1 <= len(splits) <= 8192:
        raise ValueError("bounded original global opening splits required")
    for opening, split in splits.items():
        identifier(opening)
        if split not in ("train", "validation", "holdout"):
            raise ValueError("unregistered original corpus split")
    return value


def _origin(manifest, record):
    snapshot, lineage = record["input"], record["lineage"]
    return {"source_sha256": manifest["source_sha256"], "recipe_sha256": manifest["recipe_sha256"],
            "stage_id": manifest["stage_id"], "round": manifest["round"], "round_id": manifest["round_id"],
            "student": manifest["student"],
            "learner": manifest["learner"], "actor_id": lineage["actor_id"],
            "teacher_actor_id": lineage["teacher_actor_id"], "input_sha256": record["input_sha256"],
            "game_id": snapshot["game_id"], "opening_id": snapshot["opening_id"],
            "line_genealogy_id": snapshot["line_genealogy_id"],
            "transposition_sha256": snapshot["transposition_sha256"],
            "ancestor_origin_sha256": lineage["ancestor_origin_sha256"]}


def _record_digest(record):
    projected = {**record, "record_sha256": "", "input": {"sealed_input_sha256": record["input_sha256"]}}
    domain = record.get("schema")
    if domain not in (LEGACY_CODEC.record_domain, SUBMITTED_CODEC.record_domain):
        raise ValueError("explicit original record domain required for its semantic seal")
    return digest(domain, projected)


def _teacher_pin(value):
    fields(value, ("engine", "source_commit", "binary_sha256", "binary_bytes", "profile_sha256",
                   "canonical_sha256", "capability_sha256", "options", "assets"), "Rovent frozen teacher pin")
    identifier(value["engine"])
    if value["engine"] not in ("stockfish", "reckless"):
        raise ValueError("registered Stockfish/Reckless teacher family required")
    revision = value["source_commit"]
    if type(revision) is not str or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("original teacher source commit required")
    if uint(value["binary_bytes"]) == 0:
        raise ValueError("original teacher binary byte pin required")
    for name in ("binary_sha256", "profile_sha256", "canonical_sha256", "capability_sha256"):
        sha(value[name])
    for name in ("options", "assets"):
        if type(value[name]) is not dict or len(value[name]) > 128:
            raise ValueError("bounded original teacher options/assets required")
        for key, item in value[name].items():
            identifier(key)
            if name == "assets":
                sha(item)
            elif type(item) is not str or len(item.encode()) > 8192:
                raise ValueError("bounded original teacher option string required")
    return value


def _validate_teachers(recipe, stage, manifest):
    actors = {actor.actor_id: actor for actor in recipe.actors}
    if set(manifest["teachers"]) != set(stage.teacher_actor_ids):
        raise ValueError("original corpus teacher mixture differs from frozen stage")
    for actor_id, pin in manifest["teachers"].items():
        _teacher_pin(pin)
        actor = actors[actor_id]
        if (pin["engine"] != actor.family or pin["source_commit"] != actor.source_revision
                or pin["binary_sha256"] != actor.binary_sha256 or pin["profile_sha256"] != actor.profile_sha256):
            raise ValueError("original frozen teacher source/binary/profile/family changed")


def _native(manifest, record):
    native = record.get("native")
    raw = native.get("raw_submission") if type(native) is dict else None
    if type(raw) is dict and raw.get("schema") == "rz-rovent-raw-record/3":
        from .rovent_native_v3 import validate_native_v3
        return validate_native_v3(manifest, record)
    if type(raw) is dict and raw.get("schema") == "rz-rovent-native-owner-event/2":
        from .rovent_native_v2 import validate_native_v2
        return validate_native_v2(manifest, record)
    return _native_common(manifest, record, NATIVE_RAW_DOMAIN)


def _native_common(manifest, record, raw_domain):
    native = fields(record["native"], ("head", "source_sha256", "model_sha256", "encoding_sha256", "input_digest",
        "view_digest", "output_digest", "process_epoch", "game_generation", "root_generation", "request_sequence",
        "execution_epoch", "execution_sequence", "completed_nn_inputs", "submitted_sequence", "completed_sequence",
        "retired_sequence", "owner_run_invoked", "physical_confirmed", "accepted_retired", "raw_submission",
        "raw_completion", "raw_retirement", "raw_submission_sha256", "raw_completion_sha256", "raw_retirement_sha256"),
        "original native lifecycle")
    if (native["head"] != record["role"] or native["source_sha256"] != manifest["source_sha256"]
            or any(native[name] is not True for name in ("owner_run_invoked", "physical_confirmed", "accepted_retired"))):
        raise ValueError("original native lifecycle source/role/accepted physical state mismatch")
    for name in ("source_sha256", "model_sha256", "encoding_sha256", "input_digest", "view_digest", "output_digest"):
        sha(native[name])
    for name in ("process_epoch", "game_generation", "root_generation", "request_sequence", "execution_epoch",
                 "execution_sequence", "completed_nn_inputs"):
        if uint(native[name]) == 0:
            raise ValueError("positive original native owner/execution identity required")
    for name in ("submitted_sequence", "completed_sequence", "retired_sequence"):
        uint(native[name])
    if (native["execution_epoch"] != native["process_epoch"]
            or not native["submitted_sequence"] < native["completed_sequence"] < native["retired_sequence"]):
        raise ValueError("original native completion/retirement chronology mismatch")
    from .virtual_review import input_digest
    if native["input_digest"] != input_digest(record["input"]["encoded_model_input"]):
        raise ValueError("original native lifecycle changed the sealed submitted input/view")
    for name, kind in (("raw_submission", "submitted"), ("raw_completion", "completed"), ("raw_retirement", "retired")):
        raw = native[name]
        if (type(raw) is not dict or raw.get("schema") != raw_domain or raw.get("kind") != kind
                or raw.get("process_epoch") != native["process_epoch"]
                or raw.get("request_sequence") != native["request_sequence"]
                or digest(raw_domain, raw) != sha(native[name + "_sha256"])):
            raise ValueError("original native raw owner body/seal/request mismatch")
    submitted, completed, retired = (native[name] for name in ("raw_submission", "raw_completion", "raw_retirement"))
    if (submitted.get("model_sha256") != native["model_sha256"]
            or submitted.get("input_digest") != native["input_digest"]
            or submitted.get("view_digest") != native["view_digest"]
            or completed.get("output_digest") != native["output_digest"]
            or retired.get("output_digest") != native["output_digest"]
            or completed.get("physical_confirmed") is not True or retired.get("accepted_retired") is not True):
        raise ValueError("original native raw receipt changed lifecycle summary")
    for raw, sequence in ((submitted, native["submitted_sequence"]), (completed, native["completed_sequence"]),
                          (retired, native["retired_sequence"])):
        for key in ("process_epoch", "game_generation", "root_generation", "request_sequence", "head", "source_sha256"):
            if type(raw.get(key)) is not type(native[key]) or raw[key] != native[key]:
                raise ValueError("original native raw owner scope/source changed")
        if (type(raw.get("sequence")) is not int or raw["sequence"] != sequence
                or raw.get("input_sha256") != record["input_sha256"]
                or raw.get("ticket") != submitted.get("ticket")):
            raise ValueError("original native raw event chronology/input/ticket changed")
    if (submitted["sequence"] != record["input"]["capture_sequence"]
            or completed.get("submitted_sha256") != native["raw_submission_sha256"]
            or retired.get("submitted_sha256") != native["raw_submission_sha256"]
            or retired.get("completed_sha256") != native["raw_completion_sha256"]
            or completed.get("execution_epoch") != native["execution_epoch"]
            or completed.get("execution_sequence") != native["execution_sequence"]
            or completed.get("completed_nn_inputs") != native["completed_nn_inputs"]
            or completed.get("owner_run_invoked") is not True
            or completed.get("measurement_complete") is not True or completed.get("physical_fresh") is not True):
        raise ValueError("original native source/run/completion/retirement join failed")
    ticket = submitted.get("ticket")
    fields(ticket, ("process_epoch", "game_generation", "root_generation", "request_sequence", "implementation_sha256",
                    "question_digest", "view_sequence", "view_digest", "context_policy", "actual_tokens", "selected_records",
                    "anchors", "length_bucket", "root_state", "representation", "deadline_clock", "deadline_tick"),
           "original native frozen ticket")
    for key in ("process_epoch", "game_generation", "root_generation", "request_sequence", "view_digest"):
        if type(ticket[key]) is not type(native[key]) or ticket[key] != native[key]:
            raise ValueError("original native frozen ticket scope/view changed")
    snapshot, view = record["input"], record["input"]["selected_view"]
    if (ticket["question_digest"] != snapshot["question_id"]
            or ticket["context_policy"] != view["context_policy_sha256"]
            or ticket["selected_records"] != len(view["records"]) or ticket["anchors"] != len(view["anchors"])
            or ticket["length_bucket"] != view["length_bucket"]
            or ticket["actual_tokens"] != sum(len(row["moves"]) for row in view["records"])
                + sum(len(view["query"][key]) for key in ("prefix", "proposal", "counter"))):
        raise ValueError("original native frozen ticket changed selected question/context allocation")
    for key in ("implementation_sha256", "question_digest", "view_digest", "context_policy"):
        sha(ticket[key])
    for key in ("view_sequence", "deadline_clock", "deadline_tick", "actual_tokens", "selected_records", "anchors", "length_bucket"):
        uint(ticket[key])
    root = fields(ticket["root_state"], ("owner", "revision", "semantic"), "original native root state")
    uint(root["owner"])
    uint(root["revision"])
    sha(root["semantic"])
    representation = fields(ticket["representation"], ("input", "history", "candidates", "mask_positions", "model", "encoding",
        "precision", "frozen_epoch", "record_revision", "role"), "original native input representation")
    for key in ("input", "history", "candidates", "mask_positions", "model", "encoding"):
        sha(representation[key])
    for key in ("frozen_epoch", "record_revision"):
        uint(representation[key])
    if (representation["role"] != record["role"] or representation["input"] != native["input_digest"]
            or representation["model"] != native["model_sha256"] or representation["encoding"] != native["encoding_sha256"]
            or submitted.get("encoding_sha256") != native["encoding_sha256"]
            or representation["precision"] != "Fp32"):
        raise ValueError("original native model/encoding/FP32 role representation changed")
    _native_output(native, record, completed.get("output"))
    return native


def _bit_values(value, count, description):
    if type(value) is not list or len(value) != count:
        raise ValueError(f"original native {description} allocation mismatch")
    numbers = [struct.unpack("<f", uint(bits, 2**32 - 1).to_bytes(4, "little"))[0] for bits in value]
    if any(not math.isfinite(number) for number in numbers):
        raise ValueError(f"original native {description} must be finite FP32")
    return b"".join(bits.to_bytes(4, "little") for bits in value), numbers


def _native_output(native, record, output):
    fields(output, ("request_epoch", "request_sequence", "game_generation", "root_generation", "implementation_sha256",
                    "representation", "payload", "private_latent_bits"),
           "original native output")
    ticket = native["raw_submission"]["ticket"]
    if (output["representation"] != ticket["representation"]
            or output["implementation_sha256"] != ticket["implementation_sha256"]):
        raise ValueError("original native output changed representation or implementation authority")
    for key, expected in (("request_epoch", native["process_epoch"]), ("request_sequence", native["request_sequence"]),
                          ("game_generation", native["game_generation"]), ("root_generation", native["root_generation"])):
        if type(output[key]) is not int or output[key] != expected:
            raise ValueError("original native output belongs to another request/root")
    _bit_values(output["private_latent_bits"], 6144, "private latent")
    payload, role = output["payload"], record["role"]
    if role == "validator":
        fields(payload, ("kind", "task_logits_bits"), "original native validator output")
        if payload["kind"] != "task_ranking":
            raise ValueError("original native private role output mismatch")
        raw, _ = _bit_values(payload["task_logits_bits"], 7, "task logits")
        framed = b"rz-pals-accepted-validator-head/1\0" + raw
    else:
        extra = ("divergence_logits_bits",) if role == "critic" else ()
        fields(payload, ("kind", "candidate_logits_bits", "wdl_bits", *extra), "original native P/C output")
        if payload["kind"] != ("counterexample" if role == "critic" else "proposal"):
            raise ValueError("original native private role output mismatch")
        candidates = len(record["input"]["legal_moves"])
        raw, _ = _bit_values(payload["candidate_logits_bits"], candidates, "candidate logits")
        wdl_raw, wdl = _bit_values(payload["wdl_bits"], 3, "WDL")
        if any(value < 0.0 or value > 1.0 for value in wdl) or abs(sum(wdl) - 1.0) > 1e-4:
            raise ValueError("original native WDL probability shape/normalization mismatch")
        if role == "critic":
            count = len(record["input"]["encoded_model_input"]["divergences"])
            divergence_raw, _ = _bit_values(payload["divergence_logits_bits"], count, "divergence logits")
            framed = (b"rz-rovent-accepted-critic-head/1\0" + candidates.to_bytes(8, "little")
                      + count.to_bytes(8, "little") + raw + divergence_raw + wdl_raw)
        else:
            framed = b"rz-pals-accepted-proposer-head/1\0" + candidates.to_bytes(8, "little") + raw + wdl_raw
    if hashlib.sha256(framed).hexdigest() != native["output_digest"]:
        raise ValueError("original native raw output differs from accepted head digest")


def _target_policy(snapshot, label):
    fields(label, ("observed_sequence", "question_id", "view_sha256", "raw_evidence_sha256", "source", "completed",
                   "stale", "anchor", "ranked_moves", "teacher_binary_sha256", "teacher_profile_sha256",
                   "go_sequence", "completed_depth"), "Rovent conditional teacher opinion")
    if (label["source"] != "rovent_completed_teacher_multipv" or label["completed"] is not True
            or label["stale"] is not False or uint(label["observed_sequence"]) < snapshot["capture_sequence"]
            or label["question_id"] != snapshot["question_id"] or label["view_sha256"] != snapshot["view_sha256"]
            or label["anchor"] != snapshot["rules_anchor"] or uint(label["go_sequence"]) == 0
            or uint(label["completed_depth"], 2**32 - 1) == 0):
        raise ValueError("Rovent policy requires completed same-anchor original teacher opinion")
    for name in ("raw_evidence_sha256", "teacher_binary_sha256", "teacher_profile_sha256"):
        sha(label[name])
    ranked = label["ranked_moves"]
    if (type(ranked) is not list or not 1 <= len(ranked) <= 4 or len(set(ranked)) != len(ranked)
            or any(type(move) is not int or move not in snapshot["legal_moves"] for move in ranked)):
        raise ValueError("Rovent teacher policy contains duplicate/illegal/unsupported ranked candidates")
    weights = [math.exp(-rank) for rank in range(len(ranked))]
    total = sum(weights)
    return [weights[ranked.index(move)] / total if move in ranked else 0.0 for move in snapshot["legal_moves"]]


def _constraint(value):
    if type(value) is not dict:
        raise ValueError("original typed teacher go constraint required")
    kind = value.get("kind")
    keys = {"combined": ("depth", "reported_nodes"), "nodes": ("reported_nodes",),
            "depth": ("depth",), "move_time": ("milliseconds",)}.get(kind)
    if keys is None:
        raise ValueError("unsupported original teacher go constraint")
    fields(value, ("kind", *keys), "original teacher go constraint")
    for key in keys:
        if uint(value[key], 65535 if key == "depth" else 2**64 - 1) == 0:
            raise ValueError("positive original teacher go limit required")


def _raw_moves(value):
    if type(value) is not list or len(value) > 256:
        raise ValueError("bounded original teacher PV required")
    for move in value:
        uint(move, 65535)
        if move >> 12 > 4 or (move & 63) == ((move >> 6) & 63):
            raise ValueError("original teacher PV Move16 serialization mismatch")


def _raw_score(value):
    if type(value) is not dict or value.get("kind") not in ("centipawns", "mate_moves", "unknown"):
        raise ValueError("typed original external CP/mate observation required")
    if value["kind"] == "unknown":
        fields(value, ("kind",), "unknown original external score")
    else:
        fields(value, ("kind", "value"), "original external score")
        if type(value["value"]) is not int or not -(2**31) <= value["value"] < 2**31:
            raise ValueError("original external raw score exceeds i32")


def _teacher_question(record, binding):
    question = fields(binding["question"], ("process_epoch", "game_generation", "root_generation", "implementation_sha256",
        "sequence", "parent", "sender", "recipient", "purpose", "anchor", "anchor_root_ply", "original_first_move", "prefix_digest",
        "line_digest", "line_offset", "question_digest", "evidence_revision", "root_budget", "deadline_clock", "deadline_tick", "view"),
        "original reciprocal teacher question")
    native = record["native"]
    for key in ("process_epoch", "game_generation", "root_generation"):
        if type(question[key]) is not int or question[key] != native[key]:
            raise ValueError("original teacher question crossed process/game/root")
    if (question["sequence"] != binding["question_sequence"] or question["question_digest"] != binding["question_digest"]
            or question["root_budget"] != binding["root_budget"]
            or question["implementation_sha256"] != native["raw_submission"]["ticket"]["implementation_sha256"]
            or binding["position_anchor"] != record["input"]["rules_anchor"]):
        raise ValueError("original teacher frozen question/position/budget binding changed")
    for key in ("sequence", "anchor_root_ply", "line_offset", "evidence_revision", "deadline_clock", "deadline_tick"):
        uint(question[key], 65535 if key in ("anchor_root_ply", "line_offset") else 2**64 - 1)
    for key in ("implementation_sha256", "prefix_digest", "question_digest", "root_budget"):
        sha(question[key])
    if question["line_digest"] is not None:
        sha(question["line_digest"])
    for key in ("sender", "recipient", "purpose"):
        identifier(question[key])
    if question["parent"] is not None:
        parent = fields(question["parent"], ("process_epoch", "game_generation", "root_generation", "sequence"),
                        "original teacher parent question")
        for key in ("process_epoch", "game_generation", "root_generation"):
            if type(parent[key]) is not int or parent[key] != native[key]:
                raise ValueError("original teacher parent crossed process/game/root")
        uint(parent["sequence"])
    anchor = fields(question["anchor"], ("owner", "revision", "semantic"), "original teacher exact question anchor")
    uint(anchor["owner"])
    uint(anchor["revision"])
    if sha(anchor["semantic"]) != record["input"]["rules_anchor"]["rules_state_sha256"]:
        raise ValueError("original teacher question did not ask the exact submitted Rules endpoint")
    if question["original_first_move"] is not None:
        _raw_moves([question["original_first_move"]])
    view = fields(question["view"], ("view_digest", "view_sequence", "context_policy", "actual_tokens", "selected_records",
                                    "anchors", "length_bucket"), "original teacher question view seal")
    sha(view["view_digest"])
    sha(view["context_policy"])
    for key in ("view_sequence", "actual_tokens", "selected_records", "anchors", "length_bucket"):
        uint(view[key])
    if (view["actual_tokens"] > 512 or view["selected_records"] > 64 or not 1 <= view["anchors"] <= 4
            or view["length_bucket"] not in (8, 16, 32, 64, 128, 256)):
        raise ValueError("original teacher question view capacity mismatch")
    if (binding["mode"] not in ("HybridPcr", "HybridPc", "CpuPcr", "CpuPc")
            or uint(binding["soft_quantum_ns"]) == 0 or binding["context_token_budget"] not in (128, 256, 512)):
        raise ValueError("original teacher execution mode/quantum/context budget invalid")
    roots = binding["root_moves"]
    framed = b"rz-external-root-restriction/1\0" + bytes([roots is not None])
    if roots is not None:
        _raw_moves(roots)
        if not roots or len(set(roots)) != len(roots) or any(move not in record["input"]["legal_moves"] for move in roots):
            raise ValueError("original teacher root restriction contains duplicate/illegal candidate")
        from .generation_data import _move16_uci
        framed += len(roots).to_bytes(8, "little")
        for move in roots:
            raw = _move16_uci(move).encode("ascii")
            framed += len(raw).to_bytes(8, "little") + raw
    if hashlib.sha256(framed).hexdigest() != binding["root_restriction"]:
        raise ValueError("original teacher root restriction seal changed")


def _teacher_head(manifest, record, source, label):
    from .rovent_collection_teacher import COLLECTION_BINDING_DOMAIN
    from .rovent_collection_teacher import validate_collection_teacher_binding, validate_legacy_teacher_root

    raw = fields(source["raw_receipt"], ("schema", "kind", "actor_id", "input_sha256", "anchor", "teacher_pin",
                                        "binding", "completion"), "original teacher policy owner body")
    actor = source["actor_id"]
    pin = manifest["teachers"].get(actor)
    if (pin is None or raw["kind"] != "teacher_policy" or raw["actor_id"] != actor
            or raw["teacher_pin"] != pin or record["lineage"]["teacher_actor_id"] != actor
            or source["source"] != "rovent_completed_teacher_multipv"
            or source["binary_sha256"] != pin["binary_sha256"] or source["profile_sha256"] != pin["profile_sha256"]
            or label["teacher_binary_sha256"] != pin["binary_sha256"]
            or label["teacher_profile_sha256"] != pin["profile_sha256"]):
        raise ValueError("original teacher actor/source/binary/profile attribution mismatch")
    _teacher_pin(pin)
    binding = raw["binding"]
    collection = type(binding) is dict and binding.get("schema") == COLLECTION_BINDING_DOMAIN
    submitted = record["native"]["raw_submission"].get("schema") == "rz-rovent-raw-record/3"
    if submitted and not collection:
        raise ValueError("submitted corpus requires its independent Collection teacher binding")
    if collection:
        if submitted:
            from .rovent_collection_teacher import validate_collection_teacher_binding_v3
            validate_collection_teacher_binding_v3(manifest, record, actor, pin, binding)
        else:
            validate_collection_teacher_binding(manifest, record, actor, pin, binding)
    else:
        # Preserve the historical exact body and its digest. Descriptor-bearing
        # legacy dispatches are a second explicit variant, never loose extras.
        descriptor = type(binding) is dict and bool({"root_descriptor", "root_descriptor_sha256"} & set(binding))
        extra = ("root_descriptor", "root_descriptor_sha256") if descriptor else ()
        fields(binding, ("question_sequence", "question_digest", "root_budget", "root_restriction",
            "execution_epoch", "execution_sequence", "owner", "process", "requested", "question", "position_anchor", "root_moves",
            "mode", "soft_quantum_ns", "context_token_budget", *extra), "original teacher dispatch binding")
        for key in ("question_sequence", "execution_epoch", "execution_sequence", "owner"):
            uint(binding[key])
        for key in ("question_digest", "root_budget", "root_restriction"):
            sha(binding[key])
    process = fields(binding["process"], ("pid", "process_group", "proc_start_ticks"), "original teacher process")
    for key in process:
        if uint(process[key], 2**32 - 1 if key != "proc_start_ticks" else 2**64 - 1) == 0:
            raise ValueError("known original teacher physical process required")
    _constraint(binding["requested"])
    if not collection:
        _teacher_question(record, binding)
        if descriptor:
            validate_legacy_teacher_root(manifest, record, binding)
    completed = fields(raw["completion"], ("profile", "purpose", "issued_go", "requested", "issued",
        "readiness_barrier_completed", "go_issued", "raw", "opinions", "requested_multipv",
        "conditional_policy_target_known", "option_application_observed", "model_loading_observed", "attempt", "stop_control", "timing"),
        "original teacher completion")
    expected_profile = {"stockfish": "stockfish19_embedded_nnue", "reckless": "reckless090_embedded_nnue"}.get(pin["engine"])
    if (completed["profile"] != expected_profile or completed["purpose"] != "teacher"
            or completed["requested"] != binding["requested"] or completed["go_issued"] is not True
            or completed["readiness_barrier_completed"] is not True
            or completed["conditional_policy_target_known"] is not True
            or not collection and binding["execution_epoch"] != record["native"]["process_epoch"]):
        raise ValueError("original teacher request/go/readiness/profile completion was not joined")
    _constraint(completed["issued"])
    if (type(completed["issued_go"]) is not str or not completed["issued_go"].startswith("go ")
            or len(completed["issued_go"].encode()) > 8192 or any(ord(char) < 32 for char in completed["issued_go"])):
        raise ValueError("bounded original teacher issued go text required")
    for key in ("option_application_observed", "model_loading_observed"):
        if completed[key] is not None and type(completed[key]) is not bool:
            raise ValueError("teacher option/model loading observation must preserve known/unknown")
    timing_value = completed["timing"]
    timing_keys = ("go_issued", "stop_sent", "bestmove_drained", "ready_completed")
    has_durations = type(timing_value) is dict and "durations_ns" in timing_value
    timing = fields(timing_value, (*timing_keys, *(("durations_ns",) if has_durations else ())),
                    "original teacher stop/drain/readiness observations")
    if (any(type(timing[key]) is not bool for key in timing_keys) or timing["go_issued"] is not True
            or timing["bestmove_drained"] is not True or timing["ready_completed"] is not True):
        raise ValueError("original teacher physical go/bestmove/readiness observations missing")
    if has_durations:
        duration_keys = ("admitted_to_run_started", "run_started_to_go_issued", "go_issued_to_stop_sent",
                         "go_issued_to_bestmove_drained", "stop_sent_to_bestmove_drained",
                         "bestmove_drained_to_ready_completed", "ready_completed_to_returned", "admitted_to_returned")
        durations = fields(timing["durations_ns"], duration_keys, "original teacher relative durations")
        for value in durations.values():
            if value is not None:
                uint(value)
        if not timing["stop_sent"] and (durations["go_issued_to_stop_sent"] is not None
                                        or durations["stop_sent_to_bestmove_drained"] is not None):
            raise ValueError("teacher stop duration without observed stop")
    if completed["stop_control"] is not None:
        control = fields(completed["stop_control"], ("sequence", "action"), "original teacher stop control")
        uint(control["sequence"])
        identifier(control["action"])
    observed = fields(completed["raw"], ("identity", "observed_uci", "request_id", "best_move", "pv", "score", "bound",
        "wdl_per_mille", "perspective", "requested_depth", "reported_depth", "seldepth", "root_restricted", "completion",
        "work", "elapsed_ns"), "original teacher raw UCI report")
    from .generation_data import _teacher_identity
    _teacher_identity(observed["identity"])
    identity = observed["identity"]
    assets = {item["purpose"]: item["sha256"] for item in identity["assets"]}
    if (identity["binary_sha256"] != pin["binary_sha256"] or identity["options"] != pin["options"]
            or len(assets) != len(identity["assets"]) or assets != pin["assets"]
            or observed["request_id"] != label["go_sequence"] or observed["completion"] != "BestMove"):
        raise ValueError("original teacher raw binary/options/assets/go completion changed")
    uci = fields(observed["observed_uci"], ("name", "author"), "observed original teacher UCI identity")
    identifier(uci["name"])
    if uci["author"] is not None:
        identifier(uci["author"])
    _raw_moves(observed["pv"])
    _raw_score(observed["score"])
    work = fields(observed["work"], ("nodes", "qnodes", "tt_hits"), "original reported external work")
    for count in work.values():
        if count is not None:
            uint(count)
    uint(observed["elapsed_ns"])
    attempt = fields(completed["attempt"], ("elapsed_ns", "work", "request_id", "partial_report_matches", "process"),
                     "original physical teacher attempt")
    if (uint(attempt["request_id"]) != label["go_sequence"] or attempt["partial_report_matches"] is not True
            or attempt["work"] != observed["work"] or uint(attempt["elapsed_ns"]) < observed["elapsed_ns"]):
        raise ValueError("original teacher attempt/work/request/raw report join failed")
    physical = fields(attempt["process"], ("identity", "stop_sent", "quit_sent", "exit_observed", "stdout_drained",
        "stderr_drained", "exit_code", "exit_signal", "cleanup_complete", "quarantined", "ownership_lost", "stdout_bytes", "stderr_bytes"),
        "original physical teacher process receipt")
    if (physical["identity"] != binding["process"] or physical["quarantined"] is not False
            or physical["ownership_lost"] is not False):
        raise ValueError("original teacher process owner is unknown/lost/quarantined")
    for key in ("stop_sent", "quit_sent", "exit_observed", "stdout_drained", "stderr_drained", "cleanup_complete",
                "quarantined", "ownership_lost"):
        if type(physical[key]) is not bool:
            raise ValueError("original teacher lifecycle observation must be a boolean")
    for key in ("stdout_bytes", "stderr_bytes"):
        uint(physical[key])
    for key in ("exit_code", "exit_signal"):
        if physical[key] is not None and (type(physical[key]) is not int or not -(2**31) <= physical[key] < 2**31):
            raise ValueError("original teacher process exit observation must retain signed integer/unknown")
    opinions = completed["opinions"]
    if (type(opinions) is not list or not 1 <= len(opinions) <= 4
            or uint(completed["requested_multipv"], 4) != len(opinions)):
        raise ValueError("original completed teacher MultiPV allocation mismatch")
    ranked = []
    for rank, opinion in enumerate(opinions, 1):
        fields(opinion, ("rank", "depth", "seldepth", "pv", "score", "bound", "reported_wdl"),
               "original completed teacher ranked opinion")
        _raw_moves(opinion["pv"])
        _raw_score(opinion["score"])
        if (type(opinion["rank"]) is not int or opinion["rank"] != rank or not opinion["pv"]
                or uint(opinion["depth"], 65535) != label["completed_depth"] or opinion["bound"] != "ExactReported"):
            raise ValueError("original teacher rank/depth/bound/PV completion mismatch")
        ranked.append(opinion["pv"][0])
    if ranked != label["ranked_moves"] or observed["best_move"] != ranked[0]:
        raise ValueError("original teacher policy changed ranked first moves")
    if binding["root_moves"] is not None and any(move not in binding["root_moves"] for move in ranked):
        raise ValueError("original teacher ranked policy violates its root restriction")


def _rules_head(record, source, label):
    native_v2 = record["native"]["raw_submission"].get("schema") in (
        "rz-rovent-native-owner-event/2", "rz-rovent-raw-record/3")
    extra = ("rules_replay",) if native_v2 else ()
    raw = fields(source["raw_receipt"], ("schema", "kind", "input_sha256", "anchor", "trajectory", "rules_receipt", *extra),
                 "original actual Rules outcome body")
    if native_v2:
        from .rovent_native_v2 import validate_wdl_replay_v2
        validate_wdl_replay_v2(record, raw, label)
    authority = raw["trajectory"]
    if (raw["kind"] != "rules_wdl" or source["source"] != "rules_selfplay_trajectory"
            or source["binary_sha256"] is not None or source["profile_sha256"] is not None
            or source["actor_id"] != record["lineage"]["actor_id"]
            or trajectory_digest(authority) != label["trajectory_membership_sha256"]
            or authority["game_id"] != record["input"]["game_id"] or authority["ending"] != label["ending"]
            or authority["outcome"] != label["outcome"] or authority["result_evidence_sha256"] != label["rules_result_sha256"]
            or record["input"]["rules_anchor"] not in authority["anchors"]):
        raise ValueError("original WDL lacks the exact actual Rules trajectory anchor/result")
    expected = {"schema": "rz-pals-rules-outcome-evidence/1", "game_id": authority["game_id"],
                "ending": authority["ending"], "outcome": authority["outcome"], "rules_terminal": True}
    if (raw["rules_receipt"] != expected
            or digest(expected["schema"], expected) != authority["result_evidence_sha256"]):
        raise ValueError("original Rules terminal receipt changed or is missing")


def _head_sources(manifest, record, labels):
    sources, snapshot = record["head_sources"], record["input"]
    if type(sources) is not dict or not set(sources) <= {"policy", "wdl", "divergence", "task", "nnuec_wdl"}:
        raise ValueError("bounded typed original head provenance required")
    if any(label is not None and head not in sources for head, label in labels.items()):
        raise ValueError("known original target has no head owner provenance")
    for head, source in sources.items():
        fields(source, ("actor_id", "source", "binary_sha256", "profile_sha256", "raw_receipt", "raw_receipt_sha256",
                        "observed_sequence", "anchor"), "original head source")
        identifier(source["actor_id"])
        identifier(source["source"])
        for key in ("binary_sha256", "profile_sha256"):
            if source[key] is not None:
                sha(source[key])
        raw = source["raw_receipt"]
        if (type(raw) is not dict or raw.get("schema") != HEAD_DOMAIN
                or digest(HEAD_DOMAIN, raw) != sha(source["raw_receipt_sha256"])
                or source["anchor"] != snapshot["rules_anchor"] or raw.get("anchor") != snapshot["rules_anchor"]
                or raw.get("input_sha256") != record["input_sha256"]
                or uint(source["observed_sequence"]) < snapshot["capture_sequence"]):
            raise ValueError("original head body/anchor/chronology/raw seal mismatch")
        label = labels[head]
        if head == "divergence" and codec_from_manifest(manifest) is SUBMITTED_CODEC:
            # A preserved Unknown witness still has a strict original-owner
            # shape and causal chain. Do not skip those checks just because
            # it intentionally supplies no optimizer target.
            from .rovent_divergence_owner import validate_pc_divergence_head
            validate_pc_divergence_head(record, source, label, manifest=manifest)
        if label is None:
            continue
        if (source["raw_receipt_sha256"] != label["raw_evidence_sha256"]
                or source["observed_sequence"] != label["observed_sequence"]):
            raise ValueError("known original head target differs from source receipt")
        if head == "divergence" and codec_from_manifest(manifest) is SUBMITTED_CODEC:
            continue  # The exact raw /3 owner witness was joined above.
        if head == "policy":
            _teacher_head(manifest, record, source, label)
        elif head in ("wdl", "nnuec_wdl"):
            _rules_head(record, source, label)
        else:
            raise ValueError("UnsupportedRoventHeadSource: original divergence/task owner producer is not connected")


def _nnuec_input(record):
    names = ("nnuec_input", "nnuec_input_sha256", "nnuec_model_sha256", "nnuec_semantics")
    values = [record[name] for name in names]
    if all(value is None for value in values):
        if record["nnuec_wdl"] is not None:
            raise ValueError("NNUEC WDL has no original Rust reference input/model")
        return
    if any(value is None for value in values):
        raise ValueError("partial original NNUEC input/model identity")
    sha(record["nnuec_input_sha256"])
    sha(record["nnuec_model_sha256"])
    if digest(NNUEC_INPUT_DOMAIN, record["nnuec_input"]) != record["nnuec_input_sha256"]:
        raise ValueError("original Rust NNUEC reference input seal mismatch")
    from . import nnuec_reference as base
    from . import nnuec_v2_reference as relation
    payload = record["nnuec_input"]
    if type(payload) is not dict:
        raise ValueError("typed original Rust NNUEC reference input required")
    validator = {base.INPUT_SCHEMA: base, relation.INPUT_SCHEMA: relation}.get(payload.get("schema"))
    if validator is None or record["nnuec_semantics"] != validator.SEMANTICS:
        raise ValueError("explicit matching NNUEC V1/V2 semantics required; no conversion")
    # Syntax/ABI budget only. The actual learner additionally checks its own
    # model configuration and the registered collection-source model pin.
    validator.ReferenceInput.from_mapping(payload, SimpleNamespace(max_lines=128, max_plies=256))


def _validate_row(manifest, record, indexed_origin, limits):
    fields(record, ("schema", "role", "split", "input", "input_sha256", "origin_sha256", "record_sha256", "lineage",
                    "native", "head_sources", "targets", "divergence_index", "nnuec_input", "nnuec_input_sha256",
                    "nnuec_model_sha256", "nnuec_semantics", "nnuec_wdl"), "original Rovent training record")
    codec = codec_from_manifest(manifest)
    if codec is SUBMITTED_CODEC:
        validate_record_codec(manifest, record)
    elif record["schema"] != RECORD_DOMAIN:
        raise ValueError("new original Rovent corpus record required; legacy/diagnostic records are not converted")
    if codec is SUBMITTED_CODEC and (type(record["targets"]) is not dict or record["targets"].get("task") is not None
            or any(record[name] is not None for name in ("nnuec_input", "nnuec_input_sha256", "nnuec_model_sha256",
                "nnuec_semantics", "nnuec_wdl"))
            or type(record["head_sources"]) is not dict or set(record["head_sources"]) - {"policy", "wdl", "divergence"}):
        raise ValueError("submitted P/C record cannot manufacture V or separate R/NNUEC evidence")
    input_validator, input_seal = validate_input, seal_input
    if codec is SUBMITTED_CODEC:
        from .rovent_submitted_input import validate_input as input_validator, seal_input as input_seal
    snapshot = input_validator(record["input"])
    if (record["role"] != snapshot["role"] or record["split"] != manifest["opening_splits"].get(snapshot["opening_id"])
            or snapshot["model_epoch"] != manifest["model_epoch"] or snapshot["context_epoch"] != manifest["context_epoch"]
            or sha(record["input_sha256"]) != input_seal(snapshot)):
        raise ValueError("original record role/input/model/context/global split mismatch")
    lineage = fields(record["lineage"], ("actor_id", "teacher_actor_id", "ancestor_origin_sha256"), "original row lineage")
    identifier(lineage["actor_id"])
    actor = lineage["teacher_actor_id"]
    if actor is not None and identifier(actor) not in manifest["teachers"]:
        raise ValueError("original row teacher actor absent from frozen manifest")
    _sha_list(lineage["ancestor_origin_sha256"], 256, "row ancestor origins")
    if not set(lineage["ancestor_origin_sha256"]) <= set(manifest["inherited_origin_sha256"]):
        raise ValueError("original row references missing/future/sibling ancestor origins")
    if (sha(record["origin_sha256"]) != digest(ORIGIN_DOMAIN, _origin(manifest, record))
            or sha(record["record_sha256"]) != _record_digest(record)):
        raise ValueError("original row origin/record semantic seal mismatch")
    expected_origin = {key: record[key] for key in ("record_sha256", "input_sha256", "origin_sha256", "split")}
    expected_origin.update({key: snapshot[key] for key in ("game_id", "opening_id", "line_genealogy_id", "transposition_sha256")})
    expected_origin.update(lineage)
    if indexed_origin != expected_origin or type(indexed_origin) is not dict:
        raise ValueError("original row missing/changed in recovered corpus origin index")
    _native(manifest, record)
    labels = fields(record["targets"], ("policy", "wdl", "divergence", "task"), "original head-separated future targets")
    result = dict.fromkeys(labels)
    for head, function in (("policy", _target_policy), ("wdl", wdl_target),
                           ("divergence", divergence_target), ("task", task_pairs)):
        if labels[head] is None:
            continue
        if ((record["role"] == "validator" and head != "task")
                or (record["role"] != "validator" and head == "task")
                or (head == "divergence" and record["role"] != "critic")):
            raise ValueError("original corpus target violates private role routing")
        if codec is SUBMITTED_CODEC and head == "divergence":
            from .rovent_divergence import divergence_target as submitted_divergence_target
            result[head] = submitted_divergence_target(snapshot, labels[head])
        else:
            result[head] = function(snapshot, labels[head])
    divergence_index = record["divergence_index"]
    if labels["divergence"] is None:
        if divergence_index is not None:
            raise ValueError("divergence target index without an original known target")
    elif codec is SUBMITTED_CODEC:
        if type(divergence_index) is not int or divergence_index != labels["divergence"]["slot"]:
            raise ValueError("submitted divergence target changed its original attack head slot")
    elif (type(divergence_index) is not int or divergence_index != snapshot["divergence_record_ids"].index(
            labels["divergence"]["counter_record_id"])):
        raise ValueError("original divergence candidate order/target index mismatch")
    _nnuec_input(record)
    nnuec_wdl = None if record["nnuec_wdl"] is None else wdl_target(snapshot, record["nnuec_wdl"])
    _head_sources(manifest, record, {**labels, "nnuec_wdl": record["nnuec_wdl"]})
    return {"input_sha256": record["input_sha256"], "origin_sha256": record["origin_sha256"],
            "record_sha256": record["record_sha256"], "targets": result,
            "role": record["role"], "split": record["split"],
            "native_retirement_sha256": record["native"]["raw_retirement_sha256"],
            "head_raw_sha256": {head: source["raw_receipt_sha256"] for head, source in record["head_sources"].items()},
            "divergence_index": divergence_index,
            "nnuec_wdl": nnuec_wdl, "nnuec_model_sha256": record["nnuec_model_sha256"],
            "nnuec_semantics": record["nnuec_semantics"]}


@dataclass(frozen=True)
class CorpusLimits:
    max_file_bytes: int = 32 * 1024 * 1024
    max_total_bytes: int = 64 * 1024 * 1024
    max_records: int = 4096
    max_receipts: int = 32768

    def __post_init__(self):
        uint(self.max_file_bytes, 1024 * 1024 * 1024)
        uint(self.max_total_bytes, 4 * 1024 * 1024 * 1024)
        uint(self.max_records, 262144)
        uint(self.max_receipts, 1048576)
        if min(self.max_file_bytes, self.max_total_bytes, self.max_records, self.max_receipts) == 0:
            raise ValueError("positive bounded corpus limits required")
        if self.max_file_bytes > self.max_total_bytes:
            raise ValueError("per-file limit exceeds corpus byte limit")


def _secret_basename(name):
    value = name.casefold()
    return (value == ".env" or value.startswith(".env.") or value.startswith("client_secret")
            or value in {"credentials", "credentials.json", "application_default_credentials.json",
                         "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"})


def _regular_path(path, *, suffixes=(".json",)):
    """Explicit source JSON only; no symlink/reparse ancestors or secret paths."""
    path = Path(path)
    if not path.is_absolute() or path.suffix.casefold() not in suffixes:
        raise ValueError("absolute pinned corpus source path required")
    for item in (path, *path.parents):
        if _secret_basename(item.name):
            raise ValueError("secret path is not a corpus source")
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("corpus symlink/reparse path rejected")
    return path


def _strict_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate JSON field in original corpus")
        result[name] = value
    return result


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _read_pinned_bytes(path, expected_sha256, expected_bytes, maximum, *, suffixes=(".json",), read_accounting=None):
    """Pin raw bytes through one descriptor before payload decoding."""
    sha(expected_sha256)
    expected_bytes = uint(expected_bytes, maximum)
    if expected_bytes == 0:
        raise ValueError("nonempty original corpus file required")
    path = _regular_path(path, suffixes=suffixes)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != expected_bytes:
        raise ValueError("original corpus is not a bounded single-link regular file")
    lease = None
    if read_accounting is not None:
        if not all(callable(getattr(read_accounting, name, None)) for name in
                   ("reserve_external", "charge_external", "finish_external", "fail_external")):
            raise ValueError("local original corpus read accounting callbacks required")
        # A one-byte probe proves EOF while bounding a growing-file failure.
        lease = read_accounting.reserve_external(expected_bytes + 1)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        # Unbuffered reads keep returned block lengths equal to descriptor read
        # work for the accounting path, including the one-byte growth probe.
        # The legacy None path preserves its original buffering behavior.
        with os.fdopen(descriptor, "rb", buffering=0 if read_accounting is not None else -1) as source:
            opened = os.fstat(source.fileno())
            if _identity(opened) != _identity(before):
                raise ValueError("original corpus file changed before read")
            chunks, observed, total = [], hashlib.sha256(), 0
            read_maximum = expected_bytes if read_accounting is not None else maximum
            while block := source.read(min(1024 * 1024, read_maximum + 1 - total)):
                if read_accounting is not None:
                    read_accounting.charge_external(lease, len(block))
                total += len(block)
                if total > read_maximum:
                    raise ValueError("original corpus byte limit exceeded")
                observed.update(block)
                chunks.append(block)
            after_fd = os.fstat(source.fileno())
        after_path = path.lstat()
        if (_identity(before) != _identity(after_fd) or _identity(before) != _identity(after_path)
                or total != expected_bytes or observed.hexdigest() != expected_sha256):
            raise ValueError("original corpus raw byte pin/identity mismatch")
        _regular_path(path, suffixes=suffixes)
        payload = b"".join(chunks)
        if read_accounting is not None:
            read_accounting.finish_external(lease, expected_bytes=expected_bytes)
        return payload
    except BaseException:
        if read_accounting is not None:
            read_accounting.fail_external(lease)
        raise


def _decode_json(raw):
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite corpus JSON")))
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError("original corpus strict UTF-8 JSON required") from error


def _read_pinned_json(path, expected_sha256, expected_bytes, maximum, *, read_accounting=None):
    return _decode_json(_read_pinned_bytes(path, expected_sha256, expected_bytes, maximum,
                                          read_accounting=read_accounting))


def _read_pinned_records(path, expected_sha256, expected_bytes, maximum, expected_records, *, read_accounting=None):
    raw = _read_pinned_bytes(path, expected_sha256, expected_bytes, maximum, suffixes=(".jsonl",),
                             read_accounting=read_accounting)
    lines = raw.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    if len(lines) != expected_records:
        raise ValueError("original corpus chunk record count mismatch")
    result = []
    for line in lines:
        if line.endswith(b"\r"):
            line = line[:-1]
        if not line or len(line) > 1024 * 1024:
            raise ValueError("bounded nonempty original corpus JSONL row required")
        result.append(_decode_json(line))
    return result


@dataclass(frozen=True)
class CorpusRow:
    """Opaque owner-bound row reference; it is neither a row nor an authority."""
    _owner: object
    _index: int
    _split: str
    _optimization: bool


class _RowView(Sequence):
    def __init__(self, owner, indices, split, optimization):
        self._owner, self._indices = owner, tuple(indices)
        self._split, self._optimization = split, optimization

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _RowView(self._owner, self._indices[index], self._split, self._optimization)
        return CorpusRow(self._owner, self._indices[index], self._split, self._optimization)


def _relative_chunk(root, value):
    if (type(value) is not str or not value or len(value.encode()) > 256
            or "\\" in value or ":" in value or value.startswith("/")
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError("explicit relative original corpus chunk path required")
    path = root.joinpath(*value.split("/"))
    _regular_path(path, suffixes=(".jsonl",))
    return path


class RoventTrainingCorpus:
    """Recovered original-byte corpus owner, distinct from a live training owner.

    A successful admission proves data joins only. The host must separately
    provide its opaque original producer/trainer authority before any update.
    There is intentionally no ``authorized=True`` or authority callback input.
    """
    @classmethod
    def from_files(cls, manifest_path, *, manifest_sha256, index_path, index_sha256,
                   ancestors=(), limits=CorpusLimits(), read_accounting=None):
        return cls._from_files(manifest_path, manifest_sha256=manifest_sha256, index_path=index_path,
            index_sha256=index_sha256, ancestors=ancestors, limits=limits, read_accounting=read_accounting,
            codec=LEGACY_CODEC)

    @classmethod
    def from_submitted_files(cls, manifest_path, *, manifest_sha256, index_path, index_sha256,
                             ancestors=(), limits=CorpusLimits(), read_accounting=None):
        """Explicit /2 reader; no legacy file or checkpoint is converted."""
        return cls._from_files(manifest_path, manifest_sha256=manifest_sha256, index_path=index_path,
            index_sha256=index_sha256, ancestors=ancestors, limits=limits, read_accounting=read_accounting,
            codec=SUBMITTED_CODEC)

    @classmethod
    def _from_files(cls, manifest_path, *, manifest_sha256, index_path, index_sha256,
                    ancestors, limits, read_accounting, codec):
        if type(limits) is not CorpusLimits:
            raise ValueError("validated original corpus limits required")
        manifest_path, index_path = _regular_path(manifest_path), _regular_path(index_path)
        used = manifest_path.stat().st_size + index_path.stat().st_size
        if used > limits.max_total_bytes:
            raise ValueError("original corpus aggregate byte limit exceeded")
        manifest = _read_pinned_json(manifest_path, manifest_sha256, manifest_path.stat().st_size,
                                     limits.max_file_bytes, read_accounting=read_accounting)
        _manifest(manifest, limits, codec=codec)
        index = _read_pinned_json(index_path, index_sha256, index_path.stat().st_size, limits.max_file_bytes,
                                  read_accounting=read_accounting)
        fields(index, ("schema", "manifest_sha256", "record_count", "chunks", "origins"),
               "original Rovent corpus index")
        if (validate_container_pair(manifest, index) is not codec
                or index["manifest_sha256"] != digest(codec.manifest_domain, manifest)):
            raise ValueError("original corpus index/manifest semantic seal mismatch")
        uint(index["record_count"], limits.max_records)
        if (type(index["chunks"]) is not list or len(index["chunks"]) > limits.max_records
                or type(index["origins"]) is not list or len(index["origins"]) != index["record_count"]):
            raise ValueError("original corpus bounded record/chunk allocation mismatch")
        by_record = {}
        for origin in index["origins"]:
            if type(origin) is not dict or "record_sha256" not in origin:
                raise ValueError("typed original corpus origin index required")
            identity = sha(origin["record_sha256"])
            if identity in by_record:
                raise ValueError("duplicate original corpus record index")
            by_record[identity] = origin
        records, seen_paths, seen_records = [], set(), set()
        for chunk in index["chunks"]:
            fields(chunk, ("path", "bytes", "sha256", "records"), "original corpus chunk pin")
            path = _relative_chunk(index_path.parent, chunk["path"])
            identity = str(path).casefold() if os.name == "nt" else str(path)
            if identity in seen_paths or path in (manifest_path, index_path):
                raise ValueError("duplicate/overlapping original corpus source file")
            seen_paths.add(identity)
            chunk_limit = min(limits.max_file_bytes, 32 * 1024 * 1024)
            used += uint(chunk["bytes"], chunk_limit)
            count = uint(chunk["records"], limits.max_records)
            if used > limits.max_total_bytes or len(records) + count > limits.max_records:
                raise ValueError("original corpus aggregate byte/record limit exceeded")
            if count == 0:
                raise ValueError("nonempty original corpus chunk required")
            payload = _read_pinned_records(path, chunk["sha256"], chunk["bytes"], chunk_limit, count,
                                            read_accounting=read_accounting)
            for record in payload:
                identity = record.get("record_sha256") if type(record) is dict else None
                if identity not in by_record or identity in seen_records:
                    raise ValueError("original corpus chunk has duplicate/unindexed record")
                seen_records.add(identity)
            records.extend(payload)
        if len(records) != index["record_count"]:
            raise ValueError("original corpus index total record count mismatch")
        if (not isinstance(ancestors, (tuple, list)) or len(ancestors) > 256
                or any(type(owner) is not cls for owner in ancestors)):
            raise ValueError("bounded recovered original ancestor corpus owners required")
        owner = cls.__new__(cls)
        # These objects have just been decoded from private pinned descriptors
        # and have never been returned to the caller. Transfer their ownership
        # rather than retaining a second full corpus snapshot in RAM.
        owner._manifest, owner._index, owner._records, owner._codec = manifest, index, tuple(records), codec
        owner.corpus_sha256, owner.index_sha256 = sha(manifest_sha256), sha(index_sha256)
        owner.manifest_semantic_sha256, owner.index_semantic_sha256 = digest(codec.manifest_domain, manifest), digest(codec.index_domain, index)
        owner._pins = (owner.corpus_sha256, owner.index_sha256, owner.manifest_semantic_sha256,
                       owner.index_semantic_sha256)
        owner._ancestors, owner._limits, owner._admission = tuple(ancestors), limits, None
        owner._source_manifest_sha256 = digest("rz-rovent-corpus-owner-state/1", [manifest, index])
        owner._samples = tuple(_validate_row(manifest, row, by_record[row["record_sha256"]], limits)
                               for row in owner._records)
        if 3 * len(records) + sum(len(row["head_sources"]) for row in records) > limits.max_receipts:
            raise ValueError("original corpus recovered receipt budget exceeded")
        if len({row["origin_sha256"] for row in records}) != len(records):
            raise ValueError("duplicate original corpus row origin")
        if len({(row["native"]["process_epoch"], row["native"]["game_generation"], row["native"]["root_generation"],
                 row["native"]["request_sequence"]) for row in records}) != len(records):
            raise ValueError("duplicate original retired native ticket")
        return owner

    def _check_owner(self, *, full=False):
        if getattr(self, "_codec", LEGACY_CODEC) is not codec_from_manifest(self._manifest):
            raise ValueError("original corpus owner changed its explicit wire family")
        if self._pins != (self.corpus_sha256, self.index_sha256, self.manifest_semantic_sha256,
                          self.index_semantic_sha256):
            raise ValueError("recovered original corpus owner pins changed")
        if full and (digest("rz-rovent-corpus-owner-state/1", [self._manifest, self._index])
                     != self._source_manifest_sha256):
            raise ValueError("recovered original corpus owner metadata changed")

    def admit_stage(self, recipe, stage_id, model_epoch, context_epoch,
                    inherited_input_sha256, inherited_origin_sha256):
        self._check_owner(full=True)
        if type(recipe) is not SeriesRecipe:
            raise ValueError("validated frozen Rovent series required")
        stage, declared, manifest = recipe.stage(stage_id), recipe.to_dict(), self._manifest
        for key, expected in (("recipe_sha256", recipe.sha256), ("stage_id", stage_id),
                              ("round_id", stage.round_id), ("student", declared["student_id"]),
                              ("learner", declared["learner_id"]), ("source_sha256", declared["source_sha256"]),
                              ("model_semantics_sha256", declared["model_semantics_sha256"]),
                              ("model_epoch", uint(model_epoch)), ("context_epoch", uint(context_epoch)),
                              ("opening_splits", declared["opening_splits"])):
            if type(manifest[key]) is not type(expected) or manifest[key] != expected:
                raise ValueError("original corpus differs from frozen stage/source/model/split")
        for key, values in (("inherited_input_sha256", inherited_input_sha256),
                            ("inherited_origin_sha256", inherited_origin_sha256)):
            _sha_list(values, 262144, key)
            # A frozen collection manifest owns only the ancestor base. A
            # resumed tensor owner can additionally retain rows consumed by
            # earlier windows of this same stage. Data checks below separate
            # them; a JSON list cannot attest that those updates happened.
            if not set(manifest[key]) <= set(values):
                raise ValueError("original corpus erased/changed ancestor-base training seals")
        _validate_teachers(recipe, stage, manifest)
        owners = (*self._ancestors, self)
        if sum(len(owner._records) for owner in owners) > 262144:
            raise ValueError("original global lineage record budget exceeded")
        if len({owner.index_semantic_sha256 for owner in owners}) != len(owners):
            raise ValueError("duplicate original ancestor corpus index")
        declared_ancestors = set(manifest["ancestor_index_sha256"])
        if declared_ancestors != {owner.index_semantic_sha256 for owner in self._ancestors}:
            raise ValueError("complete original ancestor corpus indices required")
        original, raw_by_origin = [], {}
        for owner in owners:
            owner._check_owner(full=True)
            _manifest(owner._manifest, owner._limits, codec=getattr(owner, "_codec", LEGACY_CODEC))
            ancestor_manifest = owner._manifest
            if not set(ancestor_manifest["ancestor_index_sha256"]) <= declared_ancestors:
                raise ValueError("original ancestor index dependency is missing")
            if (ancestor_manifest["recipe_sha256"] != recipe.sha256
                    or ancestor_manifest["opening_splits"] != declared["opening_splits"]
                    or ancestor_manifest["student"] != declared["student_id"]
                    or ancestor_manifest["learner"] != declared["learner_id"]):
                raise ValueError("original ancestor corpus crossed series/student/learner/split")
            ancestor_stage = recipe.stage(ancestor_manifest["stage_id"])
            if ancestor_stage.round_id != ancestor_manifest["round_id"]:
                raise ValueError("original ancestor corpus changed round")
            _validate_teachers(recipe, ancestor_stage, ancestor_manifest)
            for row in owner._records:
                if _record_digest(row) != row["record_sha256"]:
                    raise ValueError("recovered original corpus row changed")
                projection = index_original_input(recipe, input_snapshot=row["input"], input_sha256=row["input_sha256"],
                    record_sha256=row["record_sha256"], producer_sha256=owner.manifest_semantic_sha256,
                    stage_id=ancestor_manifest["stage_id"], round_id=ancestor_manifest["round_id"],
                    teacher_actor_id=row["lineage"]["teacher_actor_id"])
                origin = row["origin_sha256"]
                if origin in raw_by_origin:
                    raise ValueError("duplicate original origin across corpus indices")
                raw_by_origin[origin] = (owner, row)
                original.append(projection)
        lineage = GlobalLineageIndex(recipe, original)
        allowed_records = {row["record_sha256"] for row in lineage.rows("train", optimization=True)}
        inherited = set(inherited_origin_sha256)
        base_origins = set(manifest["inherited_origin_sha256"])
        base_inputs = set(manifest["inherited_input_sha256"])
        same_stage_origins = inherited - base_origins
        parent_scope = {item.stage_id for item in recipe.lineage(stage_id)[:-1]}
        if any(origin not in raw_by_origin or raw_by_origin[origin][0]._manifest["stage_id"] not in parent_scope
               or raw_by_origin[origin][1]["record_sha256"] not in allowed_records for origin in base_origins):
            raise ValueError("ancestor-base training origin is missing, cross-split, future or sibling")
        if {raw_by_origin[origin][1]["input_sha256"] for origin in base_origins} != base_inputs:
            raise ValueError("ancestor-base original origin/input seal projection mismatch")
        if any(origin not in raw_by_origin or raw_by_origin[origin][0] is not self
               or raw_by_origin[origin][0]._manifest["stage_id"] != stage_id
               or raw_by_origin[origin][1]["split"] != "train"
               or raw_by_origin[origin][1]["record_sha256"] not in allowed_records for origin in same_stage_origins):
            raise ValueError("same-stage retained origin is not an original current-stage self corpus train row")
        # These are projections of origin sets, not a set difference between
        # input lists: a base origin and a retained origin may share one input.
        same_stage_inputs = {raw_by_origin[origin][1]["input_sha256"] for origin in same_stage_origins}
        if {raw_by_origin[origin][1]["input_sha256"] for origin in inherited} != set(inherited_input_sha256):
            raise ValueError("retained original origin/input seal projection mismatch")
        for owner in self._ancestors:
            if owner._manifest["stage_id"] in parent_scope:
                if not set(owner._manifest["inherited_origin_sha256"]) <= base_origins:
                    raise ValueError("original corpus erased an ancestor training origin")
        base_seals = {"input_sha256": sorted(base_inputs), "origin_sha256": sorted(base_origins)}
        retained_seals = {"input_sha256": sorted(same_stage_inputs), "origin_sha256": sorted(same_stage_origins)}
        base_digest = digest("rz-rovent-corpus-ancestor-base-seals/1", base_seals)
        retained_digest = digest("rz-rovent-corpus-same-stage-retained-seals/1", retained_seals)
        checkpoint_authority = {
            "schema": "rz-rovent-same-stage-checkpoint-authority-requirement/1",
            "kind": "original_checkpoint_authority_required" if same_stage_origins else "no_same_stage_rows_retained",
            "required": bool(same_stage_origins), "verified": False, "execution_authorized": False,
            "recipe_sha256": recipe.sha256, "stage_id": stage_id, "corpus_sha256": self.corpus_sha256,
            "base_inherited_seals_sha256": base_digest, "same_stage_retained_seals_sha256": retained_digest,
        }
        split_records = {split: {row["record_sha256"] for row in lineage.rows(split)}
                         for split in ("train", "validation", "holdout")}
        self._admission = {"recipe_sha256": recipe.sha256, "stage_id": stage_id,
                           "allowed_records": allowed_records, "splits": split_records,
                           "lineage": lineage}
        return {"schema": "rz-rovent-corpus-intake-audit/1", "corpus_sha256": self.corpus_sha256,
                "index_sha256": self.index_sha256, "manifest_semantic_sha256": self.manifest_semantic_sha256,
                "recipe_sha256": recipe.sha256, "stage_id": stage_id,
                **{key: manifest[key] for key in ("source_sha256", "runtime_sha256", "model_semantics_sha256",
                                                  "model_epoch", "context_epoch")},
                "records": len(self._records), "global_index_sha256": lineage.sha256,
                "cross_split_identities": lineage.cross_split_identities,
                "base_inherited_input_sha256": base_seals["input_sha256"],
                "base_inherited_origin_sha256": base_seals["origin_sha256"],
                "base_inherited_input_count": len(base_inputs), "base_inherited_origin_count": len(base_origins),
                "base_inherited_seals_sha256": base_digest,
                "same_stage_retained_input_sha256": retained_seals["input_sha256"],
                "same_stage_retained_origin_sha256": retained_seals["origin_sha256"],
                "same_stage_retained_input_count": len(same_stage_inputs),
                "same_stage_retained_origin_count": len(same_stage_origins),
                "same_stage_retained_seals_sha256": retained_digest,
                "same_stage_checkpoint_authority": checkpoint_authority,
                "source_receipts_verified": True, "original_producer_authority_present": False,
                "execution_authorized": False, "actual_training_admissible": False,
                "actual_training_executed": False, "optimizer_updates": 0, "arena_eligible": False}

    def rows(self, role, split="train", optimization=True):
        self._check_owner()
        if self._admission is None:
            raise ValueError("original corpus stage intake must precede row sampling")
        if role not in ("proposer", "critic", "validator"):
            raise ValueError("registered original corpus role required")
        if (type(optimization) is not bool or split not in ("train", "validation", "holdout")
                or optimization and split != "train"):
            raise ValueError("validation/global holdout cannot enter optimization")
        allowed = self._admission["splits"][split]
        return _RowView(self, (index for index, row in enumerate(self._records)
                              if row["role"] == role and row["record_sha256"] in allowed), split, optimization)

    def training_sample(self, row):
        self._check_owner()
        if type(row) is not CorpusRow or row._owner is not self or self._admission is None:
            raise ValueError("same admitted original corpus row owner required")
        index = uint(row._index, len(self._records) - 1)
        original = self._records[index]
        if _record_digest(original) != original["record_sha256"]:
            raise ValueError("recovered original corpus row changed")
        if (type(row._optimization) is not bool or row._split not in self._admission["splits"]
                or row._optimization and row._split != "train"):
            raise ValueError("validation/global holdout cannot enter optimization")
        allowed = self._admission["splits"][row._split]
        if original["record_sha256"] not in allowed:
            raise ValueError("nontraining/cross-split original corpus row cannot enter this sample view")
        return copy.deepcopy({**self._samples[index], "input": original["input"],
                              "nnuec_input": original["nnuec_input"], "head_sources": original["head_sources"]})
