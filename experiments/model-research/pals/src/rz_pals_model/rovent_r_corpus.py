"""Pinned actual-R readout data, independent of P/C rows and live authority.

Only explicitly pinned regular files are read. The original UTF8 input seal,
parsed FP32 semantic seal, parameter identity and raw weights-file identity
remain separate. Python checks durable data joins; Rust must independently
replay Rules and retain original CPU/loaded-lock/trainer custody. Opening this
owner cannot revive an old Job, prove old process closure, or authorize updates.
"""
from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct

from .generation_data import digest, fields, identifier, sha, uint, _anchor, _moves
from .generation_data import trajectory_digest, RULES_ENDINGS, DRAW_ENDINGS
from . import rovent_corpus as pc
from .rovent_generation import SeriesRecipe, index_original_input, validate_record_origin
from .rovent_historical_corpus import HistoricalReadBudget, _input_lock, _source_aggregate, _read_observation
from .rovent_lineage_union import LineageUnion, R_SOURCE, pc_source_domain
from .rovent_corpus_codec import codec_from_manifest
from .rovent_native_v2 import validate_position_replay, _artifact, _profile
from . import rovent_r_readout as wire

ROOT_DOMAIN = "rz-rovent-collection-root-admission/1"
ORIGIN_DOMAIN = "rz-rovent-collection-native-origin/1"
DESCRIPTOR_DOMAIN = "rz-rovent-root-admission/1"
BUDGET_DOMAIN = "rz-rovent-root-budget/1"
DEADLINE_DOMAIN = "rz-rovent-root-deadline/1"
RETIREMENT_DOMAIN = "rz-rovent-r-readout-retirement/1"
WDL_DOMAIN = "rz-rovent-r-readout-wdl-evidence/1"
OBSERVATION_DOMAIN = "rz-rovent-r-corpus-observation/1"
POOL_DOMAIN = "rz-rovent-r-auxiliary-pool/1"
_SPLITS = ("train", "validation", "holdout")
_U64 = 2**64 - 1
_MAX_ROW_BYTES = 16 * 1024 * 1024
_MAX_CHUNK_BYTES = 32 * 1024 * 1024


def _positive(value, maximum=_U64):
    if uint(value, maximum) == 0:
        raise ValueError("positive bounded R declaration required")
    return value


def _absolute(value):
    if (type(value) is not str or not value or len(value.encode("utf-8")) > 4096
            or any(ord(char) < 32 for char in value)):
        raise ValueError("explicit bounded absolute R corpus root required")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("explicit absolute R corpus root required")
    return path


def _relative(value):
    if (type(value) is not str or not value or len(value.encode("utf-8")) > 256
            or "\\" in value or ":" in value or value.startswith("/")
            or any(part in ("", ".", "..") for part in value.split("/"))
            or any(ord(char) < 32 for char in value)):
        raise ValueError("normalized relative R corpus path required")
    return value


@dataclass(frozen=True)
class RFilePin:
    path: str
    sha256: str
    bytes: int

    @classmethod
    def from_mapping(cls, value):
        fields(value, ("path", "sha256", "bytes"), "R corpus file pin")
        return cls(_relative(value["path"]), sha(value["sha256"]), _positive(value["bytes"], 1 << 30))

    def to_mapping(self):
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True)
class RCorpusFiles:
    """Explicit immutable file registration; this is not an execution lease."""
    root: str
    manifest: RFilePin
    index: RFilePin
    input_locks: tuple  # sorted (canonical lock SHA, raw file pin)

    @classmethod
    def from_mapping(cls, value):
        fields(value, ("root", "manifest", "index", "input_locks"), "R corpus files")
        _absolute(value["root"])
        locks = value["input_locks"]
        if type(locks) is not dict or len(locks) > 256:
            raise ValueError("bounded explicit original R input lock pins required")
        return cls(value["root"], RFilePin.from_mapping(value["manifest"]),
                   RFilePin.from_mapping(value["index"]),
                   tuple(sorted((sha(key), RFilePin.from_mapping(pin)) for key, pin in locks.items())))

    def to_mapping(self):
        return {"root": self.root, "manifest": self.manifest.to_mapping(), "index": self.index.to_mapping(),
                "input_locks": {key: pin.to_mapping() for key, pin in self.input_locks}}


def _canonical_domain(domain, value):
    """Root descriptors use domain NUL + JSON, not generation_digest."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
    if len(payload) > 64 * 1024:
        raise ValueError("R root descriptor exceeds 64 KiB")
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + payload).hexdigest()


def _uci_moves(value, *, empty=False, duplicates=False):
    if type(value) is not list or not (0 if empty else 1) <= len(value) <= (8192 if duplicates else 218):
        raise ValueError("bounded R UCI move declaration required")
    for move in value:
        if (type(move) is not str or len(move) not in (4, 5)
                or move[0] not in "abcdefgh" or move[1] not in "12345678"
                or move[2] not in "abcdefgh" or move[3] not in "12345678" or move[:2] == move[2:4]
                or len(move) == 5 and move[4] not in "nbrq"):
            raise ValueError("R UCI move syntax required")
    if len(set(value)) > 218 or (not duplicates and len(set(value)) != len(value)):
        raise ValueError("distinct bounded R resolved root moves required")
    return value


def _root_descriptor(value, expected_sha, lock):
    fields(value, ("domain", "input", "root_budget_sha256", "deadline_seal_sha256"), "R root descriptor")
    inp = fields(value["input"], ("policy", "startup_lock_sha256", "state_and_history_sha256", "rules_profile_sha256",
        "authority", "node_unit", "search_configuration_sha256", "coordinator_policy_sha256", "original_go",
        "allowed_root_moves", "clock", "declared_ceilings"), "R root descriptor input")
    if value["domain"] != DESCRIPTOR_DOMAIN or inp["policy"] not in ("exact_registered", "dynamic_go"):
        raise ValueError("explicit R root descriptor domain/policy required")
    for name in ("startup_lock_sha256", "state_and_history_sha256", "rules_profile_sha256",
                 "search_configuration_sha256", "coordinator_policy_sha256"):
        sha(inp[name])
    authority = fields(inp["authority"], ("epoch", "game", "root", "implementation_sha256"), "R root authority")
    _positive(authority["epoch"])
    uint(authority["game"])
    uint(authority["root"])
    sha(authority["implementation_sha256"])
    go = fields(inp["original_go"], ("coverage", "requested_nodes", "depth_plies", "mate_moves", "searchmoves",
        "transport_simulation_cap", "original_parsed_limits"), "R original resolved go")
    for key in ("requested_nodes", "depth_plies", "mate_moves"):
        if go[key] is not None:
            _positive(go[key], _U64 if key == "requested_nodes" else 65535)
    uint(go["transport_simulation_cap"])
    _uci_moves(inp["allowed_root_moves"], empty=True)
    if go["searchmoves"] is not None and _uci_moves(go["searchmoves"]) != inp["allowed_root_moves"]:
        raise ValueError("R original root restriction/order differs")
    if go["coverage"] == "semantic_context_only":
        if go["original_parsed_limits"] is not None:
            raise ValueError("semantic-only R go cannot claim original parsed limits")
    elif go["coverage"] == "full_original_go":
        raw = fields(go["original_parsed_limits"], ("searchmoves", "depth_plies", "mate_moves", "movetime_ms",
            "white_time_ms", "black_time_ms", "white_increment_ms", "black_increment_ms", "moves_to_go",
            "nodes", "infinite", "ponder"), "R original parsed go")
        for key in ("infinite", "ponder"):
            if type(raw[key]) is not bool:
                raise ValueError("plain original R go boolean required")
        for key in set(raw) - {"searchmoves", "infinite", "ponder"}:
            if raw[key] is not None:
                uint(raw[key], 65535 if key in ("depth_plies", "mate_moves") else 2**32-1 if key == "moves_to_go" else _U64)
        if any(raw[a] != go[b] for a, b in (("nodes", "requested_nodes"), ("depth_plies", "depth_plies"), ("mate_moves", "mate_moves"))):
            raise ValueError("R original parsed limits differ from resolved go")
        if (raw["searchmoves"] is None) != (go["searchmoves"] is None):
            raise ValueError("R raw/resolved restriction nullness differs")
        if raw["searchmoves"] is not None:
            _uci_moves(raw["searchmoves"], duplicates=True)
            if sum(len(v.encode()) + 3 for v in raw["searchmoves"]) > 32 * 1024 or set(raw["searchmoves"]) != set(go["searchmoves"]):
                raise ValueError("R raw/resolved move set differs")
    else:
        raise ValueError("registered R original-go coverage required")
    clocks = fields(inp["clock"], ("hard", "soft", "admission", "cleanup"), "R root clocks")
    for clock in clocks.values():
        fields(clock, ("clock_epoch", "at_ns"), "R root deadline")
        if uint(clock["clock_epoch"]) != authority["epoch"]:
            raise ValueError("R root clock domain differs")
        uint(clock["at_ns"])
    if not clocks["soft"]["at_ns"] <= clocks["admission"]["at_ns"] <= clocks["hard"]["at_ns"] <= clocks["cleanup"]["at_ns"]:
        raise ValueError("R original deadline order differs")
    ceilings = fields(inp["declared_ceilings"], ("resource_plan", "registered_max_chess_plies", "cpu_credit_limit",
        "questions_max", "ancestor_depth_max", "waiting_jobs_max", "role_tickets_max", "wall_time_max_ms",
        "cleanup_max_ms", "output_bytes_max"), "R root declared ceilings")
    resource = fields(ceilings["resource_plan"], ("cpu_quota_millicpus", "pools", "memory_high_bytes", "memory_max_bytes",
        "gpu_devices", "physical_nn_concurrency", "topology_policy"), "R declared resource plan")
    _positive(resource["cpu_quota_millicpus"], 256000)
    _positive(resource["memory_high_bytes"], 1 << 50)
    _positive(resource["memory_max_bytes"], 1 << 50)
    uint(resource["gpu_devices"], 256)
    if uint(resource["physical_nn_concurrency"]) != 1 or resource["memory_high_bytes"] > resource["memory_max_bytes"]:
        raise ValueError("R declared memory/physical concurrency relation differs")
    identifier(resource["topology_policy"])
    if type(resource["pools"]) is not list or not 1 <= len(resource["pools"]) <= 32:
        raise ValueError("bounded R declared physical thread pools required")
    for pool in resource["pools"]:
        fields(pool, ("actors", "max_threads", "cpu_charge_millicpus"), "R declared thread pool")
        if type(pool["actors"]) is not list or not pool["actors"] or len(pool["actors"]) > 16:
            raise ValueError("bounded R declared pool actors required")
        for actor in pool["actors"]:
            identifier(actor)
        if len(set(pool["actors"])) != len(pool["actors"]):
            raise ValueError("R declared pool actors must be distinct")
        _positive(pool["max_threads"], 256)
        uint(pool["cpu_charge_millicpus"], 256000)
    for name, maximum in (("registered_max_chess_plies", 65535), ("cpu_credit_limit", _U64), ("questions_max", 65536),
        ("ancestor_depth_max", 1024), ("waiting_jobs_max", 1024), ("role_tickets_max", 1024),
        ("wall_time_max_ms", 86400000), ("cleanup_max_ms", 30000), ("output_bytes_max", 64 << 30)):
        _positive(ceilings[name], maximum)
    if ceilings["ancestor_depth_max"] > ceilings["questions_max"] or clocks["cleanup"]["at_ns"] - clocks["hard"]["at_ns"] > ceilings["cleanup_max_ms"] * 1000000:
        raise ValueError("R root declared relation/cleanup bound differs")
    if (go["depth_plies"] is not None and go["depth_plies"] > ceilings["registered_max_chess_plies"]
            or go["mate_moves"] is not None and 2 * go["mate_moves"] - 1 > ceilings["registered_max_chess_plies"]):
        raise ValueError("R goal exceeds registered chess ply bound")
    stable = {key: part for key, part in inp.items() if key != "clock"}
    budget = _canonical_domain(BUDGET_DOMAIN, stable)
    deadline = _canonical_domain(DEADLINE_DOMAIN, {"domain": DEADLINE_DOMAIN, "startup_lock_sha256": inp["startup_lock_sha256"],
        "policy": inp["policy"], "authority": authority, "root_budget_sha256": budget,
        "clock": clocks, "cleanup_max_ms": ceilings["cleanup_max_ms"]})
    if (value["root_budget_sha256"] != budget or value["deadline_seal_sha256"] != deadline
            or _canonical_domain(DESCRIPTOR_DOMAIN, value) != sha(expected_sha)):
        raise ValueError("R root descriptor canonical domain-NUL seals differ")
    manifest = lock["manifest"]
    search, coordinator = manifest["engine"]["search"], manifest["coordinator"]
    for key, expected in (("startup_lock_sha256", lock["canonical_sha256"]), ("node_unit", "rovent-position-visits/1"),
        ("rules_profile_sha256", search["rules_profile_sha256"]), ("search_configuration_sha256", search["search_configuration_sha256"]),
        ("coordinator_policy_sha256", coordinator["policy_sha256"])):
        if inp[key] != expected:
            raise ValueError("R root descriptor differs from original lock configuration")
    for key in ceilings:
        expected = (manifest["resources"] if key == "resource_plan" else search[key] if key == "registered_max_chess_plies"
                    else manifest[key] if key in ("wall_time_max_ms", "cleanup_max_ms", "output_bytes_max") else coordinator[key])
        if type(ceilings[key]) is not type(expected) or ceilings[key] != expected:
            raise ValueError("R root descriptor changes original resource/credit ceiling")
    if inp["policy"] == "exact_registered":
        if inp["state_and_history_sha256"] != search["root_state_and_history_sha256"]:
            raise ValueError("R exact root descriptor changes original workload")
        for key in ("requested_nodes", "depth_plies", "mate_moves"):
            if go[key] != search[key]:
                raise ValueError("R exact root descriptor changes original go")
        if ((go["searchmoves"] is None) != (search["searchmoves"] is None)
                or go["searchmoves"] is not None and set(go["searchmoves"]) != set(search["searchmoves"])):
            raise ValueError("R exact root descriptor changes original restriction")
    return inp


def _rules(value):
    fields(value, ("anchor", "replay", "replay_sha256"), "R Rules evidence")
    _anchor(value["anchor"])
    replay = validate_position_replay(value["replay"])
    if replay["anchor"] != value["anchor"] or digest(replay["schema"], replay) != sha(value["replay_sha256"]):
        raise ValueError("R original Rules replay/anchor seal differs")
    return value


def _bound(value):
    fields(value, ("model_semantics", "parameter_sha256", "training_state", "weights_file_sha256", "collection_model_epoch", "collection_context_epoch",
        "root_state", "generation", "sequence", "full_view_sha256", "view_seal_sha256", "record_scope_sha256",
        "policy_sha256", "context_sha256", "context_query", "selection_policy", "actual_tokens", "token_budget", "length_bucket", "context_records",
        "context_anchors", "context_source_anchors", "context_anchor_states", "local_anchor", "local_encoded_perspective", "current_rules"), "R Bound")
    from .nnuec_reference import _training
    _training(value["training_state"])
    identifier(value["model_semantics"])
    for key in ("parameter_sha256", "weights_file_sha256", "full_view_sha256", "view_seal_sha256", "record_scope_sha256", "policy_sha256", "context_sha256"):
        sha(value[key])
    for key in ("collection_model_epoch", "collection_context_epoch", "sequence"):
        _positive(value[key])
    for key in ("root_state", "generation"):
        uint(value[key])
    _positive(value["token_budget"], 512)
    if uint(value["actual_tokens"], 512) > value["token_budget"] or uint(value["length_bucket"], 256) not in (8, 16, 32, 64, 128, 256):
        raise ValueError("R original review token/bucket bounds differ")
    records, anchors = value["context_records"], value["context_anchors"]
    if type(records) is not list or len(records) > 64 or type(anchors) is not list or not 1 <= len(anchors) <= 4:
        raise ValueError("bounded exact R selected records/anchors required")
    source_anchors = value["context_source_anchors"]
    if type(source_anchors) is not list or len(source_anchors) != len(records):
        raise ValueError("R original source anchors must match selected record order/count")
    states = value["context_anchor_states"]
    if type(states) is not list or len(states) != len(anchors):
        raise ValueError("R original selected anchor state/order differs")
    for state in states:
        uint(state)
    if states[0] != value["root_state"]:
        raise ValueError("R original current anchor zero must retain source root state")
    policy = fields(value["selection_policy"], ("mode", "token_budget", "max_records", "max_anchors", "required_bundle"), "R original selection policy")
    if (policy["mode"] not in ("current_root_only", "anchored_compact")
            or uint(policy["token_budget"], 512) not in (128, 256, 512) or policy["token_budget"] != value["token_budget"]
            or not len(records) <= _positive(policy["max_records"], 64)
            or not len(anchors) <= _positive(policy["max_anchors"], 4)
            or policy["mode"] == "current_root_only" and policy["max_anchors"] != 1):
        raise ValueError("R original selection policy budget/capacity differs")
    if policy["required_bundle"] is not None:
        uint(policy["required_bundle"])
    for anchor in (*anchors, *source_anchors, value["local_anchor"], value["current_rules"]):
        _rules(anchor)
    for index, record in enumerate(records):
        fields(record, ("local_id", "record_id", "record_revision", "parent_revision", "supersedes_revision", "origin_state",
            "kind", "anchor", "moves", "original_first_move", "parent", "supersedes", "source_state", "source_revision", "source_line_offset",
            "record_perspective", "encoded_perspective", "completed_depth", "score_scope", "value", "cpu_observation", "critical"), "R selected context record")
        if uint(record["local_id"]) != index:
            raise ValueError("R selected context local ID/order differs")
        identifier(record["record_id"])
        uint(record["kind"], 3)
        uint(record["anchor"], len(anchors) - 1)
        _moves(record["moves"])
        for key in ("record_revision", "origin_state"):
            uint(record[key])
        if record["origin_state"] != states[record["anchor"]]:
            raise ValueError("R original selected anchor state differs from record origin")
        if record["original_first_move"] is not None:
            _moves([record["original_first_move"]], 1)
        for key in ("parent", "supersedes"):
            if (record[key] is None) != (record[key + "_revision"] is None):
                raise ValueError("R context relation and original revision nullness differs")
            if record[key] is not None:
                if uint(record[key], len(records)-1) >= index or uint(record[key + "_revision"]) != records[record[key]]["record_revision"]:
                    raise ValueError("R context relation original revision/order differs")
        for key in ("source_state", "source_revision"):
            uint(record[key])
        uint(record["source_line_offset"], 65535)
        if record["source_line_offset"] + len(record["moves"]) > 256:
            raise ValueError("R original source suffix exceeds bounded complete line")
        if (type(record["record_perspective"]) is not str or record["record_perspective"] not in ("white", "black")
                or record["encoded_perspective"] != ("white" if anchors[record["anchor"]]["anchor"]["white_to_move"] else "black")):
            raise ValueError("R raw and encoded record perspectives differ from their explicit domains")
        uint(record["completed_depth"], 65535)
        if record["score_scope"] is not None:
            identifier(record["score_scope"])
        if record["value"] is not None and (type(record["value"]) is not int or not -(2**31) <= record["value"] < 2**31):
            raise ValueError("bounded raw R signed CPU observation required")
        if record["cpu_observation"] is not None:
            uint(record["cpu_observation"])
        if type(record["critical"]) is not bool:
            raise ValueError("plain raw R critical observation required")
    query = fields(value["context_query"], ("purpose", "prefix", "proposal", "counterexample", "legal", "divergences"), "R original context query")
    identifier(query["purpose"])
    for key in ("prefix", "proposal", "legal"):
        _moves(query[key])
    if query["counterexample"] is not None:
        _moves(query["counterexample"])
    if type(query["divergences"]) is not list or len(query["divergences"]) > 256:
        raise ValueError("bounded R original divergence positions required")
    for position in query["divergences"]:
        uint(position, 256)
    lengths = [len(record["moves"]) for record in records] + [len(query[key]) for key in ("prefix", "proposal")] + [0 if query["counterexample"] is None else len(query["counterexample"])]
    if sum(lengths) != value["actual_tokens"] or next(bucket for bucket in (8, 16, 32, 64, 128, 256) if max(lengths) <= bucket) != value["length_bucket"]:
        raise ValueError("R original query/record token count or minimum length bucket differs")
    if value["local_encoded_perspective"] != ("white" if value["local_anchor"]["anchor"]["white_to_move"] else "black"):
        raise ValueError("R explicit local encoder perspective differs from anchor")
    # The Rust parent reuses review_context's original private framing for
    # full-view/record-scope/builder-seal reconstruction. Do not replace that
    # owner with a second Python ReviewView or a reconstructed Store pin.
    return value


def _move(value):
    return {"from_square": value & 63, "to_square": value >> 6 & 63, "promotion": value >> 12}


def _input_context(parsed, bound, config):
    # Only the pure scalar ABI parser is imported; no checkpoint/model exists.
    from .nnuec_v2_reference import ReferenceInput, SEMANTICS
    if bound["model_semantics"] != SEMANTICS:
        raise ValueError("explicit registered origin-first V2 R semantics required")
    admitted = ReferenceInput.from_mapping(parsed, config)
    if admitted.context_sha256 != bound["context_sha256"] or len(parsed["context_lines"]) != len(bound["context_records"]):
        raise ValueError("R original input context seal/order differs from Bound")
    expected_edges = []
    for record, line in zip(bound["context_records"], parsed["context_lines"], strict=True):
        if (line["id"] != record["local_id"] or line["kind"] != record["kind"]
                or line["moves"] != [_move(move) for move in record["moves"]]
                or line["original_first_move"] != (None if record["original_first_move"] is None else _move(record["original_first_move"]))
                or line["perspective"] != record["encoded_perspective"]):
            raise ValueError("R input line changed exact selected record relation/anchor mapping")
        for key, kind in (("parent", 0), ("supersedes", 1)):
            if record[key] is not None:
                expected_edges.append({"source": record[key], "target": record["local_id"], "kind": kind})
    if sorted(parsed["context_edges"], key=lambda x: (x["source"], x["target"], x["kind"])) != sorted(expected_edges, key=lambda x: (x["source"], x["target"], x["kind"])):
        raise ValueError("R input edges differ from exact selected parent/supersedes graph")
    if parsed["side_to_move"] != ("white" if bound["current_rules"]["anchor"]["white_to_move"] else "black"):
        raise ValueError("R input side differs from exact current Rules anchor")
    local, current = bound["local_anchor"]["replay"], bound["current_rules"]["replay"]
    if (any(local[key] != current[key] for key in ("origin", "completeness", "start_fen"))
            or current["moves"][:len(local["moves"])] != local["moves"]
            or parsed["local_line"]["moves"] != [_move(move) for move in current["moves"][len(local["moves"]):]]
            or parsed["local_line"]["perspective"] != bound["local_encoded_perspective"]):
        raise ValueError("R local suffix differs from original Rules replay history")
    # Numeric board/anchor features are validated by the existing ABI. Only the
    # Rust Rules owner can confirm they encode these exact boards and histories.
    return parsed


def _job(value):
    fields(value, ("authority", "job_sequence", "state_owner", "state_revision", "state_semantic_sha256", "question_sha256",
        "root_restriction_sha256", "evaluation_namespace_sha256", "search_conditions_sha256", "total_native_node_credit",
        "hard_deadline_clock_epoch", "hard_deadline_ns"), "R original CpuJob")
    authority = fields(value["authority"], ("process_epoch", "game_generation", "root_generation", "implementation_sha256"), "R Job authority")
    _positive(authority["process_epoch"])
    for key in ("game_generation", "root_generation"):
        uint(authority[key])
    sha(authority["implementation_sha256"])
    for key in ("job_sequence", "total_native_node_credit"):
        _positive(value[key])
    for key in ("state_owner", "state_revision", "hard_deadline_clock_epoch", "hard_deadline_ns"):
        uint(value[key])
    for key in ("state_semantic_sha256", "question_sha256", "root_restriction_sha256", "evaluation_namespace_sha256", "search_conditions_sha256"):
        sha(value[key])
    if value["hard_deadline_clock_epoch"] != authority["process_epoch"]:
        raise ValueError("R Job deadline clock differs from original owner")
    return value


def _numeric(value, bound, config):
    fields(value, ("schema", "parameter_sha256", "board_accumulators_bits", "context_accumulator_bits", "local_accumulator_bits",
        "hidden_bits", "raw_outputs_bits", "value_bits", "wdl_logits_bits", "wdl_bits"), "R original numeric bits")
    if value["schema"] != "rz-nnuec-fp32-output/2" or value["parameter_sha256"] != bound["parameter_sha256"]:
        raise ValueError("R numeric output schema/parsed parameter identity differs")
    boards = value["board_accumulators_bits"]
    if type(boards) is not list or len(boards) != 2:
        raise ValueError("R numeric output needs both board accumulators")
    arrays = [(part, config.board_width) for part in boards] + [(value[key], count) for key, count in (
        ("context_accumulator_bits", config.context_width), ("local_accumulator_bits", config.local_width),
        ("hidden_bits", config.hidden_width), ("raw_outputs_bits", 11), ("wdl_logits_bits", 3), ("wdl_bits", 3))]
    for values, count in arrays:
        if type(values) is not list or len(values) != count:
            raise ValueError("R original numeric accumulator/readout shape differs")
        for bits in values:
            if not math.isfinite(struct.unpack("<f", struct.pack("<I", uint(bits, 2**32-1)))[0]):
                raise ValueError("finite original R FP32 bits required")
    if not math.isfinite(struct.unpack("<f", struct.pack("<I", uint(value["value_bits"], 2**32-1)))[0]):
        raise ValueError("finite original R value bits required")
    wdl = [struct.unpack("<f", struct.pack("<I", bits))[0] for bits in value["wdl_bits"]]
    if any(not 0 <= item <= 1 for item in wdl) or abs(sum(wdl) - 1) > 1e-4:
        raise ValueError("R predicted WDL probability shape differs")


def _retirement(value, question, input_bytes):
    fields(value, ("job", "worker_owner", "frame_sequence", "capture_sequence", "ending", "limits", "attempted", "selected",
        "retained_readouts", "overflow", "retained_input_bytes", "selected_input_preparation_ns", "error"), "R original capture retirement")
    _job(value["job"])
    for key in ("worker_owner", "frame_sequence", "capture_sequence"):
        if uint(value[key], 65535 if key == "worker_owner" else _U64) != question[key]:
            raise ValueError("R retirement crossed original worker/frame/capture")
    if value["job"] != question["job"]:
        raise ValueError("R retirement changed original Job")
    ending = value["ending"]
    if type(ending) is not dict or ending.get("kind") not in ("cpu_terminal", "not_started", "failed", "unknown"):
        raise ValueError("explicit original R capture ending required")
    fields(ending, ("kind", "completion") if ending["kind"] == "cpu_terminal" else ("kind",), "R capture ending")
    if ending["kind"] == "cpu_terminal":
        identifier(ending["completion"])
    limits = fields(value["limits"], ("maximum_readouts", "every_nth_score", "maximum_input_bytes"), "R capture limits")
    _positive(limits["maximum_readouts"], 64)
    _positive(limits["every_nth_score"], 2**32-1)
    _positive(limits["maximum_input_bytes"], _MAX_ROW_BYTES)
    for key in ("attempted", "selected", "overflow", "retained_input_bytes", "selected_input_preparation_ns"):
        uint(value[key])
    _positive(value["retained_readouts"], 64)
    if (not value["retained_readouts"] <= value["selected"] <= min(value["attempted"], limits["maximum_readouts"])
            or question["readout_sequence"] >= value["attempted"]
            or question["readout_sequence"] % limits["every_nth_score"] != 0
            or not input_bytes <= value["retained_input_bytes"] <= limits["maximum_input_bytes"]):
        raise ValueError("R original zero-based score sampling/retention limits differ")
    if value["error"] is not None:
        identifier(value["error"])


def _wdl(value, record):
    if value is None:
        return None
    fields(value, ("target", "raw_receipt", "raw_receipt_sha256"), "R actual Rules WDL evidence")
    label = fields(value["target"], ("observed_sequence", "question_id", "view_sha256", "raw_evidence_sha256", "source", "game_id",
        "anchor", "trajectory_membership_sha256", "rules_result_sha256", "ending", "outcome"), "R actual trajectory WDL target")
    uint(label["observed_sequence"])
    identifier(label["question_id"])
    for key in ("view_sha256", "raw_evidence_sha256", "trajectory_membership_sha256", "rules_result_sha256"):
        sha(label[key])
    _anchor(label["anchor"])
    raw = fields(value["raw_receipt"], ("schema", "kind", "native_input_sha256", "semantic_input_sha256", "anchor", "trajectory",
        "rules_receipt", "rules_replay"), "R raw actual Rules WDL receipt")
    if (label["source"] != "rules_selfplay_trajectory" or label["game_id"] != record["lineage"]["game_id"]
            or label["anchor"] != record["bound"]["current_rules"]["anchor"] or raw["anchor"] != label["anchor"]
            or raw["schema"] != WDL_DOMAIN or raw["kind"] != "rules_wdl"
            or raw["native_input_sha256"] != record["input"]["native_input_sha256"]
            or raw["semantic_input_sha256"] != record["input"]["semantic_input_sha256"]
            or digest(WDL_DOMAIN, raw) != sha(value["raw_receipt_sha256"])
            or label["raw_evidence_sha256"] != value["raw_receipt_sha256"]):
        raise ValueError("R WDL requires original current-anchor Rules evidence, not numeric/CP/mate scores")
    trajectory = raw["trajectory"]
    if (trajectory_digest(trajectory) != label["trajectory_membership_sha256"]
            or label["anchor"] not in trajectory["anchors"]
            or any(trajectory[key] != label[key] for key in ("game_id", "ending", "outcome"))
            or trajectory["result_evidence_sha256"] != label["rules_result_sha256"]):
        raise ValueError("R WDL exact actual trajectory membership differs")
    result = fields(raw["rules_receipt"], ("schema", "game_id", "ending", "outcome", "rules_terminal"), "R raw Rules result")
    if (result["schema"] != "rz-pals-rules-outcome-evidence/1" or result["rules_terminal"] is not True
            or any(result[key] != label[key] for key in ("game_id", "ending", "outcome"))
            or digest(result["schema"], result) != label["rules_result_sha256"]):
        raise ValueError("R actual Rules terminal receipt differs")
    trace = fields(raw["rules_replay"], ("schema", "initial", "actual_selected_moves", "final_position", "completion", "claim"), "R actual Rules trajectory replay")
    initial, final = validate_position_replay(trace["initial"]), validate_position_replay(trace["final_position"])
    moves = _moves(trace["actual_selected_moves"], 512)
    if (trace["schema"] != "rz-rovent-actual-rules-trajectory-replay/1" or trace["completion"] not in ("automatic", "current_claim")
            or trace["completion"] == "automatic" and trace["claim"] is not None
            or trace["completion"] == "current_claim" and trace["claim"] not in ("threefold_repetition", "fifty_move")
            or len(trajectory["anchors"]) != len(moves) + 1 or trajectory["anchors"][0] != initial["anchor"]
            or trajectory["anchors"][-1] != final["anchor"]
            or any(initial[key] != final[key] for key in ("origin", "completeness", "start_fen"))
            or final["moves"] != initial["moves"] + moves):
        raise ValueError("R actual Rules trajectory history/completion differs")
    if label["ending"] not in RULES_ENDINGS or (label["ending"] in DRAW_ENDINGS) != (label["outcome"] == "draw"):
        raise ValueError("R actual Rules outcome/termination differs")
    if label["outcome"] == "draw":
        return [0.0, 1.0, 0.0]
    win = (label["outcome"] == "white_win") == label["anchor"]["white_to_move"]
    return [1.0, 0.0, 0.0] if win else [0.0, 0.0, 1.0]


def _native_admission(record, manifest, lock, expected_parameter):
    root = fields(record["native_admission"], ("schema", "native_origin", "native_origin_sha256", "process_epoch",
        "game_generation", "root_generation", "implementation_sha256", "deadline_clock", "deadline_tick", "root_budget_sha256",
        "selective_policy_sha256", "root_state", "store_root_state", "root_moves", "depth_plies", "mate_moves", "requested_nodes",
        "root_descriptor", "root_descriptor_sha256"), "R original native root admission")
    if root["schema"] != ROOT_DOMAIN or digest(ROOT_DOMAIN, root) != sha(record["native_admission_sha256"]):
        raise ValueError("R original native root admission seal differs")
    job, bound = record["question"]["job"], record["bound"]
    authority = job["authority"]
    for key in ("process_epoch", "game_generation", "root_generation", "implementation_sha256"):
        if key != "implementation_sha256":
            uint(root[key])
        if root[key] != authority[key]:
            raise ValueError("R native root admission differs from original Job authority")
    for key in ("deadline_clock", "deadline_tick", "store_root_state"):
        uint(root[key])
    for key in ("implementation_sha256", "root_budget_sha256", "selective_policy_sha256"):
        sha(root[key])
    state = fields(root["root_state"], ("owner", "revision", "semantic"), "R original root state handle")
    uint(state["owner"])
    uint(state["revision"])
    sha(state["semantic"])
    for key in ("depth_plies", "mate_moves", "requested_nodes"):
        if root[key] is not None:
            _positive(root[key], _U64 if key == "requested_nodes" else 65535)
    _uci_moves(root["root_moves"], empty=True)
    if root["root_descriptor"] != record["root_descriptor"] or root["root_descriptor_sha256"] != record["root_descriptor_sha256"]:
        raise ValueError("R native admission changed original root descriptor")
    descriptor = _root_descriptor(record["root_descriptor"], record["root_descriptor_sha256"], lock)
    if (descriptor["authority"] != {"epoch": authority["process_epoch"], "game": authority["game_generation"],
            "root": authority["root_generation"], "implementation_sha256": authority["implementation_sha256"]}
            or descriptor["startup_lock_sha256"] != record["startup_input_lock_sha256"]
            or root["root_budget_sha256"] != record["root_descriptor"]["root_budget_sha256"]
            or root["deadline_clock"] != descriptor["clock"]["hard"]["clock_epoch"]
            or root["deadline_tick"] != descriptor["clock"]["hard"]["at_ns"]
            or job["hard_deadline_clock_epoch"] != root["deadline_clock"]
            or job["hard_deadline_ns"] > root["deadline_tick"]
            or descriptor["state_and_history_sha256"] != state["semantic"]
            or root["root_moves"] != descriptor["allowed_root_moves"]
            or any(root[key] != descriptor["original_go"][key] for key in ("depth_plies", "mate_moves", "requested_nodes"))):
        raise ValueError("R original Job/root descriptor/go/deadline join differs")
    origin = fields(root["native_origin"], ("schema", "purpose", "process_epoch", "mode", "input_lock_sha256", "source_commit",
        "source_components", "source_sha256", "weights_sha256", "export_manifest_sha256", "backend_sha256", "runtime_library_sha256",
        "runtime_bundle_sha256", "native_configuration_sha256", "model_semantics", "model_semantics_sha256", "encoding", "encoding_sha256",
        "encoding_registry_sha256", "model_implementation_sha256", "frozen_epoch", "representation_model_domain", "runtime_sha256_domain",
        "runtime_sha256", "teacher_profiles"), "R original native collection origin")
    if (origin["schema"] != ORIGIN_DOMAIN or origin["purpose"] != "collection"
            or digest(ORIGIN_DOMAIN, origin) != sha(root["native_origin_sha256"])
            or uint(origin["process_epoch"]) != authority["process_epoch"]):
        raise ValueError("R original collection native origin seal/scope differs")
    for key in ("input_lock_sha256", "source_sha256", "weights_sha256", "export_manifest_sha256", "backend_sha256",
        "runtime_library_sha256", "native_configuration_sha256", "model_semantics_sha256", "encoding_sha256", "encoding_registry_sha256",
        "model_implementation_sha256", "runtime_sha256"):
        sha(origin[key])
    if origin["runtime_bundle_sha256"] is not None:
        sha(origin["runtime_bundle_sha256"])
    for key in ("model_semantics", "encoding"):
        identifier(origin[key])
    _positive(origin["frozen_epoch"])
    engine = lock["manifest"]["engine"]
    if (origin["input_lock_sha256"] != lock["canonical_sha256"] or origin["input_lock_sha256"] != record["startup_input_lock_sha256"]
            or origin["source_commit"] != manifest["source_commit"] or engine.get("source_commit") != origin["source_commit"]
            or engine.get("source_components") != origin["source_components"]
            or _source_aggregate(origin["source_commit"], origin["source_components"]) != manifest["source_sha256"]
            or origin["source_sha256"] != manifest["source_sha256"]
            or origin["runtime_sha256_domain"] != "loaded_runtime_library_file_sha256"
            or origin["runtime_sha256"] != origin["runtime_library_sha256"] or origin["runtime_sha256"] != manifest["runtime_sha256"]
            or origin["model_semantics_sha256"] != manifest["model_semantics_sha256"]
            or origin["representation_model_domain"] != "loaded_checkpoint_sha256"
            or origin["mode"] not in ("HybridPcr", "HybridPc", "CpuPcr", "CpuPc")
            or engine.get("mode") != {"HybridPcr": "hybrid_pcr", "HybridPc": "hybrid_pc", "CpuPcr": "cpu_pcr", "CpuPc": "cpu_pc"}[origin["mode"]]):
        raise ValueError("R original collection source/runtime/model domains differ from lock")
    profiles = origin["teacher_profiles"]
    if type(profiles) is not list or len(profiles) > 2 or profiles != engine.get("teachers"):
        raise ValueError("R collection teacher profiles changed original lock")
    for profile in profiles:
        _profile(profile)
    for pin in manifest["teachers"].values():
        if not any(all(profile[key] == pin[key] for key in ("canonical_sha256", "capability_sha256", "source_commit"))
                   and profile["profile"]["sha256"] == pin["profile_sha256"]
                   and profile["binary"]["sha256"] == pin["binary_sha256"]
                   and profile["binary"]["bytes"] == pin["binary_bytes"] for profile in profiles):
            raise ValueError("R collection corpus teacher absent from original lock")
    nnuec = fields(engine.get("nnuec"), ("schema", "semantics", "encoding", "implementation_sha256", "configuration_sha256", "weights",
        "provenance", "diagnostic", "context_namespace_policy_sha256", "local_namespace_policy_sha256", "search_conditions_sha256"), "R original lock NNUEC")
    for key in ("implementation_sha256", "configuration_sha256", "context_namespace_policy_sha256", "local_namespace_policy_sha256", "search_conditions_sha256"):
        sha(nnuec[key])
    _artifact(nnuec["weights"])
    _artifact(nnuec["provenance"])
    if type(nnuec["diagnostic"]) is not bool:
        raise ValueError("R original NNUEC diagnostic declaration must be plain boolean")
    if (nnuec["schema"] != "rz-nnuec-fp32-v2" or nnuec["semantics"] != bound["model_semantics"]
            or bound["weights_file_sha256"] != nnuec["weights"]["sha256"]
            or bound["parameter_sha256"] != expected_parameter):
        raise ValueError("R parsed parameter identity/raw NNUEC weights file pins differ")
    # Parsed parameter SHA is not a raw checkpoint SHA. No equality is imposed
    # between either NNUEC identity and the separate P/C model weight domain.


def _lineage(value, manifest, split):
    fields(value, ("game_id", "opening_id", "line_genealogy_id", "transposition_sha256", "actor_id", "teacher_actor_id",
        "ancestor_origin_sha256", "pc_genealogy"), "R original role-free lineage")
    for key in ("game_id", "opening_id", "line_genealogy_id", "actor_id"):
        identifier(value[key])
    sha(value["transposition_sha256"])
    pc._sha_list(value["ancestor_origin_sha256"], 256, "R ancestor origins")
    if split not in _SPLITS or manifest["opening_splits"].get(value["opening_id"]) != split:
        raise ValueError("R opening differs from frozen global split")
    if value["teacher_actor_id"] is not None and value["teacher_actor_id"] not in manifest["teachers"]:
        raise ValueError("R raw origin cannot invent a teacher actor")
    link = value["pc_genealogy"]
    if link is not None:
        fields(link, ("record_sha256", "input_sha256", "origin_sha256"), "R original PC genealogy")
        for pin in link.values():
            sha(pin)


def _validate_record(record, manifest, origin, locks, config, expected_parameter):
    fields(record, ("schema", "manifest_sha256", "split", "lineage", "native_admission", "native_admission_sha256", "root_descriptor",
        "root_descriptor_sha256", "startup_input_lock_sha256", "question", "bound", "input", "numeric", "retirement", "retirement_sha256",
        "semantic_consumption", "whole_action_cost", "wdl", "utility", "origin_sha256", "record_sha256"), "actual R readout record")
    if record["schema"] != R_SOURCE or record["manifest_sha256"] != digest(wire.MANIFEST_DOMAIN, manifest):
        raise ValueError("explicit actual R record/manifest domain required; no P/C row conversion")
    collection = manifest["collection"]
    _lineage(record["lineage"], collection, record["split"])
    question = fields(record["question"], ("job", "worker_owner", "frame_sequence", "capture_sequence", "readout_sequence", "observed_tick_ns"), "R exact question")
    _job(question["job"])
    uint(question["worker_owner"], 65535)
    _positive(question["frame_sequence"])
    _positive(question["capture_sequence"])
    uint(question["readout_sequence"])
    uint(question["observed_tick_ns"])
    # A late observed tick is retained raw. It is not semantic acceptance, a
    # reason to erase the record, or evidence of current worker authority.
    bound = _bound(record["bound"])
    if (bound["collection_model_epoch"] != collection["model_epoch"] or bound["collection_context_epoch"] != collection["context_epoch"]
            or bound["generation"] != question["job"]["authority"]["root_generation"]
            or question["job"]["state_semantic_sha256"] != bound["local_anchor"]["anchor"]["rules_state_sha256"]):
        raise ValueError("R collection epoch/Job/Bound local anchor differs")
    parsed = _input_context(wire.validate_input_evidence(record["input"], bound), bound, config)
    _numeric(record["numeric"], bound, config)
    _retirement(record["retirement"], question, record["input"]["reference_input_bytes"])
    if digest(RETIREMENT_DOMAIN, record["retirement"]) != sha(record["retirement_sha256"]):
        raise ValueError("R original retirement data projection seal differs")
    lock = locks.get(sha(record["startup_input_lock_sha256"]))
    if lock is None:
        raise ValueError("R original startup lock artifact was not explicitly pinned")
    _native_admission(record, collection, lock, expected_parameter)
    if record["semantic_consumption"] is not None or record["whole_action_cost"] is not None:
        raise ValueError("R missing semantic consumption/whole action cost must remain unknown")
    utility = fields(record["utility"], ("support_mask", "target_known_mask", "unknown_reason"), "V-free R utility state")
    for key in ("support_mask", "target_known_mask"):
        if type(utility[key]) is not list or len(utility[key]) != 7 or any(type(bit) is not bool or bit for bit in utility[key]):
            raise ValueError("V-free R utility support/known masks must remain false")
    identifier(utility["unknown_reason"])
    wdl = _wdl(record["wdl"], record)
    origin_value = {"manifest_sha256": record["manifest_sha256"], "lineage": record["lineage"], "question": question,
        "native_admission_sha256": record["native_admission_sha256"], "root_descriptor_sha256": record["root_descriptor_sha256"],
        "native_input_sha256": record["input"]["native_input_sha256"], "semantic_input_sha256": record["input"]["semantic_input_sha256"]}
    if digest(wire.ORIGIN_DOMAIN, origin_value) != sha(record["origin_sha256"]) or wire.record_digest(record) != sha(record["record_sha256"]):
        raise ValueError("R original origin/record seals differ")
    expected_origin = {"source_domain": R_SOURCE, "record_sha256": record["record_sha256"],
        "input_sha256": record["input"]["semantic_input_sha256"], "origin_sha256": record["origin_sha256"],
        "split": record["split"], "lineage": record["lineage"]}
    if origin != expected_origin:
        raise ValueError("R full original index occurrence differs from raw row")
    return {"input_sha256": record["input"]["semantic_input_sha256"], "origin_sha256": record["origin_sha256"],
        "record_sha256": record["record_sha256"], "nnuec_input": parsed, "nnuec_model_sha256": bound["parameter_sha256"],
        "nnuec_weights_file_sha256": bound["weights_file_sha256"], "nnuec_semantics": bound["model_semantics"],
        "nnuec_wdl": wdl, "retirement_sha256": record["retirement_sha256"],
        "wdl_raw_sha256": None if record["wdl"] is None else record["wdl"]["raw_receipt_sha256"], "split": record["split"]}


def _manifest(value, limits, recipe):
    fields(value, ("schema", "collection", "pc_ancestor_index_sha256", "r_ancestor_index_sha256"), "R corpus manifest")
    if value["schema"] != wire.MANIFEST_DOMAIN:
        raise ValueError("new R corpus manifest domain required")
    collection = pc._manifest(value["collection"], limits)
    pc._sha_list(value["pc_ancestor_index_sha256"], 256, "PC ancestor index pins")
    pc._sha_list(value["r_ancestor_index_sha256"], 256, "R ancestor index pins")
    stage, registered = recipe.stage(collection["stage_id"]), recipe.to_dict()
    for key, expected in (("recipe_sha256", recipe.sha256), ("round_id", stage.round_id), ("student", registered["student_id"]),
        ("learner", registered["learner_id"]), ("source_sha256", registered["source_sha256"]),
        ("model_semantics_sha256", registered["model_semantics_sha256"]), ("opening_splits", registered["opening_splits"])):
        if type(collection[key]) is not type(expected) or collection[key] != expected:
            raise ValueError("R corpus differs from frozen recipe/source/model/splits")
    pc._validate_teachers(recipe, stage, collection)
    return value


@dataclass(frozen=True)
class RCorpusRow:
    _owner: object
    _index: int
    _split: str
    _optimization: bool


class _RRows(Sequence):
    def __init__(self, owner, indices, split, optimization):
        self._owner, self._indices = owner, tuple(indices)
        self._split, self._optimization = split, optimization

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _RRows(self._owner, self._indices[index], self._split, self._optimization)
        return RCorpusRow(self._owner, self._indices[index], self._split, self._optimization)


class RoventRCorpus:
    """Bounded recovered R data owner. No JSON input creates live authority."""
    def __init__(self):
        raise TypeError("use explicit pinned RCorpusFiles")

    @classmethod
    def from_files(cls, files, *, recipe, nnuec_config, expected_nnuec_parameter_sha256,
                   pc_corpora=(), r_ancestors=(), limits=pc.CorpusLimits(), read_accounting=None):
        from .nnuec_reference import NnuecConfig
        if type(files) is not RCorpusFiles or type(recipe) is not SeriesRecipe or type(limits) is not pc.CorpusLimits:
            raise ValueError("typed R file pins, frozen recipe and bounded corpus limits required")
        if RCorpusFiles.from_mapping(files.to_mapping()) != files or type(nnuec_config) is not NnuecConfig:
            raise ValueError("strict original R file registration/configuration required")
        parameter = sha(expected_nnuec_parameter_sha256)
        if (not isinstance(pc_corpora, (tuple, list)) or len(pc_corpora) > 256
                or any(type(owner) is not pc.RoventTrainingCorpus for owner in pc_corpora)
                or not isinstance(r_ancestors, (tuple, list)) or len(r_ancestors) > 256
                or any(type(owner) is not cls for owner in r_ancestors)):
            raise ValueError("explicit typed complete PC/R ancestor owners required")
        budget = HistoricalReadBudget(limits.max_total_bytes) if read_accounting is None else read_accounting
        if not all(callable(getattr(budget, method, None)) for method in
                   ("reserve_external", "charge_external", "finish_external", "fail_external", "observation")):
            raise ValueError("bounded shared pinned read accounting required")
        root, used, sources = _absolute(files.root), 0, {}

        def read(pin, *, maximum, suffixes=(".json",)):
            nonlocal used
            if type(pin) is not RFilePin or RFilePin.from_mapping(pin.to_mapping()) != pin:
                raise ValueError("strict R source file pin required")
            path = root.joinpath(*pin.path.split("/"))
            identity = str(path).casefold() if os.name == "nt" else str(path)
            if identity in sources:
                raise ValueError("R source path repeated or overlaps another pinned source")
            sources[identity] = pin
            used += pin.bytes
            if used > limits.max_total_bytes:
                raise ValueError("R aggregate pinned byte limit exceeded")
            return pc._read_pinned_bytes(path, pin.sha256, pin.bytes, maximum,
                                         suffixes=suffixes, read_accounting=budget)

        manifest = _manifest(pc._decode_json(read(files.manifest, maximum=limits.max_file_bytes)), limits, recipe)
        manifest_sha = digest(wire.MANIFEST_DOMAIN, manifest)
        index = pc._decode_json(read(files.index, maximum=limits.max_file_bytes))
        fields(index, ("schema", "manifest_sha256", "record_count", "chunks", "origins"), "R corpus index")
        if index["schema"] != wire.INDEX_DOMAIN or index["manifest_sha256"] != manifest_sha:
            raise ValueError("R index/manifest canonical seal differs")
        count = uint(index["record_count"], limits.max_records)
        if (type(index["chunks"]) is not list or len(index["chunks"]) > min(8192, limits.max_records)
                or type(index["origins"]) is not list or len(index["origins"]) != count):
            raise ValueError("R bounded index allocation/count mismatch")
        by_record, origin_ids = {}, set()
        for origin in index["origins"]:
            fields(origin, ("source_domain", "record_sha256", "input_sha256", "origin_sha256", "split", "lineage"), "R indexed occurrence")
            identity, origin_sha = sha(origin["record_sha256"]), sha(origin["origin_sha256"])
            sha(origin["input_sha256"])
            if origin["source_domain"] != R_SOURCE or identity in by_record or origin_sha in origin_ids:
                raise ValueError("duplicate/non-R original index occurrence")
            _lineage(origin["lineage"], manifest["collection"], origin["split"])
            by_record[identity] = origin
            origin_ids.add(origin_sha)
        locks, raw_lock_paths = {}, {}
        for canonical, pin in files.input_locks:
            if pin.path not in raw_lock_paths:
                raw_lock_paths[pin.path] = (pin, pc._decode_json(read(pin, maximum=limits.max_file_bytes)))
            original_pin, raw = raw_lock_paths[pin.path]
            if pin != original_pin:
                raise ValueError("same R input lock path has conflicting raw pin")
            locks[canonical] = _input_lock(raw, canonical)
        rows, samples, seen, executions, capture_groups, total_declared = [], [], set(), set(), {}, 0
        for chunk in index["chunks"]:
            fields(chunk, ("path", "bytes", "sha256", "records"), "R chunk pin")
            expected = _positive(chunk["records"], limits.max_records)
            total_declared += expected
            if total_declared > count:
                raise ValueError("R chunk record sum exceeds original index")
            pin = RFilePin.from_mapping({key: chunk[key] for key in ("path", "sha256", "bytes")})
            raw = read(pin, maximum=min(_MAX_CHUNK_BYTES, limits.max_file_bytes), suffixes=(".jsonl",))
            lines = raw.splitlines()
            if len(lines) != expected or any(not line or len(line) > _MAX_ROW_BYTES for line in lines):
                raise ValueError("R JSONL rows exceed exact count/16 MiB cap")
            for line in lines:
                record = pc._decode_json(line)
                record_sha = record.get("record_sha256") if type(record) is dict else None
                if record_sha not in by_record or record_sha in seen:
                    raise ValueError("R chunk has duplicate/unindexed original record")
                sample = _validate_record(record, manifest, by_record[record_sha], locks, nnuec_config, parameter)
                question, authority = record["question"], record["question"]["job"]["authority"]
                occurrence = tuple(authority[key] for key in ("process_epoch", "game_generation", "root_generation")) + (question["job"]["job_sequence"],) + tuple(
                    question[key] for key in ("worker_owner", "frame_sequence", "capture_sequence", "readout_sequence"))
                if occurrence in executions:
                    raise ValueError("duplicate original R capture/score attempt occurrence")
                # A capture retirement publishes all retained rows atomically;
                # chunk boundaries may split that group, but no row may be
                # silently dropped or borrow another retirement's counters.
                capture = occurrence[:-1]
                group = capture_groups.setdefault(capture, {"sha256": record["retirement_sha256"],
                    "retained": record["retirement"]["retained_readouts"], "count": 0})
                if group["sha256"] != record["retirement_sha256"] or group["retained"] != record["retirement"]["retained_readouts"]:
                    raise ValueError("same original R capture has conflicting retirement evidence")
                group["count"] += 1
                executions.add(occurrence)
                seen.add(record_sha)
                rows.append(record)
                samples.append(sample)
            del raw, lines
        if total_declared != count or len(rows) != count or seen != set(by_record):
            raise ValueError("R original full chunk/index coverage differs")
        if any(group["count"] != group["retained"] for group in capture_groups.values()):
            raise ValueError("R atomic capture publication omits retained original readouts")
        if {row["startup_input_lock_sha256"] for row in rows} != set(locks):
            raise ValueError("R original complete startup input lock artifact set differs")
        owner = cls.__new__(cls)
        owner._manifest, owner._index, owner._records, owner._samples = manifest, index, tuple(rows), tuple(samples)
        owner._record_indices = {record["record_sha256"]: index for index, record in enumerate(owner._records)}
        owner._pc_corpora, owner._r_ancestors = tuple(pc_corpora), tuple(r_ancestors)
        owner._recipe, owner._limits = recipe, limits
        owner.corpus_sha256, owner.index_sha256 = files.manifest.sha256, files.index.sha256
        owner.manifest_semantic_sha256, owner.index_semantic_sha256 = manifest_sha, digest(wire.INDEX_DOMAIN, index)
        owner._metadata_sha = digest("rz-rovent-r-corpus-owner-state/1", [manifest, index])
        owner._pins = (owner.corpus_sha256, owner.index_sha256, owner.manifest_semantic_sha256, owner.index_semantic_sha256)
        owner._union = owner._build_union()
        # Freeze allowed source identities once. Per-sample lookup must not
        # re-hash/sort the complete global lineage for every microbatch.
        owner._allowed = tuple((split, frozenset(entry.record_sha256 for entry in
            owner._union.rows(split, optimization=split == "train", source_domain=R_SOURCE))) for split in _SPLITS)
        owner.all_occurrences_sha256 = digest(POOL_DOMAIN, {"index_sha256": owner.index_semantic_sha256,
            "occurrences": [{key: item[key] for key in ("source_domain", "record_sha256", "input_sha256", "origin_sha256")} for item in index["origins"]]})
        train_records = dict(owner._allowed)["train"]
        owner.pool_sha256 = digest(POOL_DOMAIN, {"index_sha256": owner.index_semantic_sha256,
            "occurrences": [{"source_domain": R_SOURCE, "record_sha256": row["record_sha256"],
                "input_sha256": row["input"]["semantic_input_sha256"], "origin_sha256": row["origin_sha256"]}
                for row in owner._records if row["record_sha256"] in train_records]})
        owner._pool_pin = owner.pool_sha256
        owner._all_occurrences_pin = owner.all_occurrences_sha256
        owner._read_observation = _read_observation(budget)
        return owner

    def _check_owner(self, *, full=False):
        if self._pins != (self.corpus_sha256, self.index_sha256, self.manifest_semantic_sha256, self.index_semantic_sha256):
            raise ValueError("R original corpus owner pins changed")
        if full and digest("rz-rovent-r-corpus-owner-state/1", [self._manifest, self._index]) != self._metadata_sha:
            raise ValueError("R original manifest/index metadata changed")

    def _build_union(self):
        manifest = self._manifest
        if (len({owner.index_semantic_sha256 for owner in self._pc_corpora}) != len(self._pc_corpora)
                or set(manifest["pc_ancestor_index_sha256"]) != {owner.index_semantic_sha256 for owner in self._pc_corpora}
                or len({owner.index_semantic_sha256 for owner in self._r_ancestors}) != len(self._r_ancestors)
                or set(manifest["r_ancestor_index_sha256"]) != {owner.index_semantic_sha256 for owner in self._r_ancestors}):
            raise ValueError("complete distinct original PC/R ancestor index pins required")
        occurrences, pending, by_origin, pc_records, declarations = [], [], {}, {}, []
        total = len(self._records) + sum(len(owner._records) for owner in (*self._pc_corpora, *self._r_ancestors))
        if total > 262144:
            raise ValueError("R global PC/R lineage occurrence budget exceeded")

        def remember(domain, origin, projection, ancestors, link=None):
            key = (domain, origin)
            if key in by_origin:
                raise ValueError("duplicate source origin across original PC/R owners")
            by_origin[key] = projection
            pending.append((domain, origin, projection, ancestors, link))

        for owner in self._pc_corpora:
            owner._check_owner(full=True)
            declaration = owner._manifest
            source_domain = pc_source_domain(owner)
            declarations.append(declaration)
            pc._manifest(declaration, owner._limits, codec=codec_from_manifest(declaration))
            if (declaration["recipe_sha256"] != self._recipe.sha256
                    or not set(declaration["ancestor_index_sha256"]) <= set(manifest["pc_ancestor_index_sha256"])):
                raise ValueError("PC ancestor source/closure differs from R collection recipe")
            for row in owner._records:
                if row["schema"] != source_domain or pc._record_digest(row) != row["record_sha256"]:
                    raise ValueError("original PC ancestor record changed")
                projection = index_original_input(self._recipe, input_snapshot=row["input"], input_sha256=row["input_sha256"],
                    record_sha256=row["record_sha256"], producer_sha256=owner.manifest_semantic_sha256,
                    stage_id=declaration["stage_id"], round_id=declaration["round_id"], teacher_actor_id=row["lineage"]["teacher_actor_id"])
                pc_records[(source_domain, row["origin_sha256"])] = row
                remember(source_domain, row["origin_sha256"], projection, row["lineage"]["ancestor_origin_sha256"])
        for owner in (*self._r_ancestors, self):
            owner._check_owner(full=True)
            declaration = owner._manifest
            _manifest(declaration, owner._limits, self._recipe)
            declarations.append(declaration["collection"])
            if (not set(declaration["r_ancestor_index_sha256"]) <= set(manifest["r_ancestor_index_sha256"])
                    or not set(declaration["pc_ancestor_index_sha256"]) <= set(manifest["pc_ancestor_index_sha256"])):
                raise ValueError("R ancestor index closure is incomplete")
            for row in owner._records:
                if wire.record_digest(row) != row["record_sha256"]:
                    raise ValueError("original R ancestor record changed")
                lineage, collection = row["lineage"], declaration["collection"]
                projection = validate_record_origin(self._recipe, {"input_sha256": row["input"]["semantic_input_sha256"],
                    "record_sha256": row["record_sha256"], "producer_sha256": owner.manifest_semantic_sha256,
                    "stage_id": collection["stage_id"], "round_id": collection["round_id"], "teacher_actor_id": lineage["teacher_actor_id"],
                    "opening_id": lineage["opening_id"], "game_id": lineage["game_id"], "line_genealogy_id": lineage["line_genealogy_id"],
                    "rules_state_sha256": row["bound"]["current_rules"]["anchor"]["rules_state_sha256"],
                    "transposition_sha256": lineage["transposition_sha256"]})
                remember(R_SOURCE, row["origin_sha256"], projection, lineage["ancestor_origin_sha256"], lineage["pc_genealogy"])
        for domain, origin, projection, ancestor_shas, link in pending:
            references = set()
            for ancestor in ancestor_shas:
                keys = [key for key in by_origin if key[1] == ancestor]
                if len(keys) != 1:
                    raise ValueError("PC/R original ancestor origin is missing or source-ambiguous")
                references.add(keys[0])
            if link is not None:
                matches = [source for source, original in pc_records.items()
                    if source[1] == link["origin_sha256"] and all(original[key] == link[key]
                        for key in ("record_sha256", "input_sha256", "origin_sha256"))]
                if len(matches) != 1:
                    raise ValueError("R causal PC genealogy lacks original exact PC row")
                # The old genealogy wire carries three pins, not a domain.
                # Resolve them to exactly one original PC source; never retag
                # a submitted row as the legacy PC family.
                references.add(matches[0])
            occurrences.append({"source_domain": domain, "origin_sha256": origin, "projection": projection,
                "ancestor_origins": [{"source_domain": key[0], "origin_sha256": key[1]} for key in sorted(references)]})
        union = LineageUnion.from_projections(self._recipe, occurrences, max_occurrences=self._limits.max_records)
        for declaration in declarations:
            stage_id = declaration["stage_id"]
            prior = {stage.stage_id for stage in self._recipe.lineage(stage_id)[:-1]}
            references, inputs = [], set()
            for origin in declaration["inherited_origin_sha256"]:
                matches = [key for key in by_origin if key[1] == origin]
                if len(matches) != 1 or by_origin[matches[0]]["stage_id"] not in prior:
                    raise ValueError("R/PC inherited ancestor base is missing, source-ambiguous, same-stage, future or sibling")
                source = matches[0]
                references.append({"source_domain": source[0], "origin_sha256": source[1]})
                inputs.add(by_origin[source]["input_sha256"])
            if inputs != set(declaration["inherited_input_sha256"]):
                raise ValueError("R/PC original inherited origin/input projection differs")
            union.require_training_origins(stage_id, sorted(references, key=lambda ref: (ref["source_domain"], ref["origin_sha256"])))
        return union

    def rows(self, split="train", optimization=True):
        self._check_owner()
        if type(optimization) is not bool or split not in _SPLITS or optimization and split != "train":
            raise ValueError("validation/global holdout cannot enter R optimization")
        allowed = dict(self._allowed)[split]
        return _RRows(self, (index for index, row in enumerate(self._records) if row["record_sha256"] in allowed), split, optimization)

    def training_sample(self, token):
        self._check_owner()
        if type(token) is not RCorpusRow or token._owner is not self:
            raise ValueError("same recovered R corpus token owner required")
        index = uint(token._index, len(self._records) - 1)
        if type(token._optimization) is not bool or token._split not in _SPLITS or token._optimization and token._split != "train":
            raise ValueError("validation/global holdout cannot enter R optimization")
        row = self._records[index]
        allowed = dict(self._allowed)[token._split]
        if wire.record_digest(row) != row["record_sha256"] or row["record_sha256"] not in allowed:
            raise ValueError("changed or globally excluded R source row cannot enter sampling")
        return copy.deepcopy(self._samples[index])

    def pool_metadata(self, split="train", optimization=True):
        """Seal actual sampler order after global PC/R leakage exclusions."""
        self._check_owner(full=True)
        if self.pool_sha256 != self._pool_pin:
            raise ValueError("original R auxiliary pool seal changed")
        tokens = self.rows(split, optimization=optimization)
        occurrences = [{"source_domain": R_SOURCE, "record_sha256": self._records[token._index]["record_sha256"],
            "input_sha256": self._records[token._index]["input"]["semantic_input_sha256"],
            "origin_sha256": self._records[token._index]["origin_sha256"]} for token in tokens]
        pool = digest(POOL_DOMAIN, {"index_sha256": self.index_semantic_sha256, "occurrences": occurrences})
        if split == "train" and optimization and pool != self.pool_sha256:
            raise ValueError("original R training eligible pool/order changed")
        return {"schema": POOL_DOMAIN, "index_sha256": self.index_semantic_sha256, "pool_sha256": pool,
            "occurrences": occurrences, "execution_authorized": False}

    def all_occurrence_metadata(self):
        """Raw original index order, including excluded observations."""
        self._check_owner(full=True)
        if self.all_occurrences_sha256 != self._all_occurrences_pin:
            raise ValueError("original R all-occurrence metadata seal changed")
        return {"schema": POOL_DOMAIN, "index_sha256": self.index_semantic_sha256,
            "pool_sha256": self.all_occurrences_sha256,
            "occurrences": [{key: item[key] for key in ("source_domain", "record_sha256", "input_sha256", "origin_sha256")} for item in self._index["origins"]],
            "execution_authorized": False}

    def original_auxiliary_evidence(self, record_sha256):
        """Original metadata projection for the parent's separate opaque RPC.

        These rows were already charged to the shared bounded file reader; no
        new read is performed here. No caller-supplied target-known flag or
        native/physical/training permission is manufactured from this lookup.
        """
        self._check_owner()
        identity = sha(record_sha256)
        index = self._record_indices.get(identity)
        if index is None:
            raise ValueError("original R auxiliary record is absent from this owner")
        row = self._records[uint(index, len(self._records) - 1)]
        if row["record_sha256"] != identity or wire.record_digest(row) != identity:
            raise ValueError("original R auxiliary record evidence changed")
        question, authority = row["question"], row["question"]["job"]["authority"]
        return {"schema": "rz-rovent-r-auxiliary-evidence/1", "source_domain": R_SOURCE,
            "record_sha256": identity, "origin_sha256": row["origin_sha256"],
            "native_input_sha256": row["input"]["native_input_sha256"],
            "semantic_input_sha256": row["input"]["semantic_input_sha256"],
            "occurrence": {**{key: authority[key] for key in ("process_epoch", "game_generation", "root_generation")},
                "job_sequence": question["job"]["job_sequence"],
                **{key: question[key] for key in ("worker_owner", "frame_sequence", "capture_sequence", "readout_sequence")}},
            "retirement_sha256": row["retirement_sha256"],
            "wdl_raw_sha256": None if row["wdl"] is None else row["wdl"]["raw_receipt_sha256"],
            "execution_authorized": False}

    def original_semantic_evidence(self, record_sha256):
        """Detached original fields for a separate Job-level observation join.

        The auxiliary RPC above remains unchanged. This lookup performs no
        file read, changes no input/target, and grants no historical worker,
        Store, physical or per-readout consumption authority.
        """
        self._check_owner(full=True)
        identity = sha(record_sha256)
        index = self._record_indices.get(identity)
        if index is None:
            raise ValueError("original R semantic record is absent from this owner")
        row = self._records[uint(index, len(self._records) - 1)]
        if row["record_sha256"] != identity or wire.record_digest(row) != identity:
            raise ValueError("original R semantic record evidence changed")
        return copy.deepcopy({"source_domain": R_SOURCE, "record_sha256": identity,
            "origin_sha256": row["origin_sha256"], "native_admission_sha256": row["native_admission_sha256"],
            "root_descriptor_sha256": row["root_descriptor_sha256"],
            "startup_input_lock_sha256": row["startup_input_lock_sha256"],
            "question": row["question"], "bound": row["bound"],
            "input": {key: row["input"][key] for key in ("native_input_sha256", "semantic_input_sha256")},
            "retirement": row["retirement"], "retirement_sha256": row["retirement_sha256"],
            "semantic_consumption": row["semantic_consumption"], "whole_action_cost": row["whole_action_cost"],
            "execution_authorized": False})

    def observation(self):
        self._check_owner(full=True)
        allowed = tuple(self.rows("train", optimization=True))
        known = [self._samples[token._index] for token in allowed if self._samples[token._index]["nnuec_wdl"] is not None]
        return {"schema": OBSERVATION_DOMAIN, "manifest_sha256": self.corpus_sha256, "index_sha256": self.index_sha256,
            "canonical_manifest_sha256": self.manifest_semantic_sha256, "canonical_index_sha256": self.index_semantic_sha256,
            "occurrence_count": len(self._records), "unique_input_count": len({sample["input_sha256"] for sample in self._samples}),
            "candidate_wdl_train_occurrence_count": len(known), "candidate_wdl_train_unique_input_count": len({sample["input_sha256"] for sample in known}),
            "lineage": self._union.observation().to_mapping(), "reads": copy.deepcopy(self._read_observation),
            "historical_data_integrity_only": True, "actual_rules_replay_verified": False,
            "full_review_view_reconstructed": False,
            "wdl_original_timeline_authority_verified": False,
            "original_native_authority_present": False, "physical_completion_scope": "historical_record_only",
            "old_run_closure_proven": False, "execution_authorized": False, "actual_training_executed": False}
