"""Persisted Rust Rovent root descriptor shape and canonical seals only.

This pure wire boundary never grants a live InputLock, Rules, root, clock or
resource lease. Rust independently validates its loaded lock and actual state.
"""
from __future__ import annotations

import hashlib
import json

from .generation_data import fields, sha, uint

DESCRIPTOR_DOMAIN = "rz-rovent-root-admission/1"
BUDGET_DOMAIN = "rz-rovent-root-budget/1"
DEADLINE_DOMAIN = "rz-rovent-root-deadline/1"
_U64 = 2**64 - 1

def _positive(value, maximum=_U64):
    if uint(value, maximum) == 0:
        raise ValueError("positive bounded Collection teacher identity required")
    return value


def _canonical_domain(domain, value):
    # RootDescriptor is domain NUL canonical JSON, unlike generation_digest.
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                     allow_nan=False).encode("utf-8")
    if len(raw) > 64 << 10:
        raise ValueError("Collection teacher root descriptor exceeds 64 KiB")
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + raw).hexdigest()


def _uci_moves(value, *, empty=False, raw=False):
    if type(value) is not list or not (0 if empty else 1) <= len(value) <= (8192 if raw else 218):
        raise ValueError("bounded Collection teacher declared UCI moves required")
    for move in value:
        if (type(move) is not str or len(move) not in (4, 5)
                or move[0] not in "abcdefgh" or move[1] not in "12345678"
                or move[2] not in "abcdefgh" or move[3] not in "12345678" or move[:2] == move[2:4]
                or len(move) == 5 and move[4] not in "nbrq"):
            raise ValueError("Collection teacher declared UCI move syntax differs")
    if len(set(value)) > 218 or (not raw and len(set(value)) != len(value)):
        raise ValueError("Collection teacher resolved root moves must be distinct")
    if raw and sum(len(move.encode("ascii")) + 3 for move in value) > 32 << 10:
        raise ValueError("Collection teacher original searchmoves exceed raw byte bound")
    return value


def validate_root_descriptor(value, expected_sha):
    """Check the Rust closed DTO and canonical seals; no InputLock authority."""
    fields(value, ("domain", "input", "root_budget_sha256", "deadline_seal_sha256"),
           "Collection teacher original root descriptor")
    inp = fields(value["input"], ("policy", "startup_lock_sha256", "state_and_history_sha256", "rules_profile_sha256",
        "authority", "node_unit", "search_configuration_sha256", "coordinator_policy_sha256", "original_go",
        "allowed_root_moves", "clock", "declared_ceilings"), "Collection teacher root descriptor input")
    if (value["domain"] != DESCRIPTOR_DOMAIN or inp["policy"] not in ("exact_registered", "dynamic_go")
            or inp["node_unit"] != "rovent-position-visits/1"):
        raise ValueError("Collection teacher original descriptor domain/policy/node unit differs")
    for name in ("startup_lock_sha256", "state_and_history_sha256", "rules_profile_sha256",
                 "search_configuration_sha256", "coordinator_policy_sha256"):
        sha(inp[name])
    authority = fields(inp["authority"], ("epoch", "game", "root", "implementation_sha256"),
                       "Collection teacher root descriptor authority")
    _positive(authority["epoch"])
    uint(authority["game"])
    uint(authority["root"])
    sha(authority["implementation_sha256"])
    go = fields(inp["original_go"], ("coverage", "requested_nodes", "depth_plies", "mate_moves", "searchmoves",
        "transport_simulation_cap", "original_parsed_limits"), "Collection teacher original resolved go")
    for key in ("requested_nodes", "depth_plies", "mate_moves"):
        if go[key] is not None:
            # Resolved requested_nodes is Option<u64>, not NonZeroU64.
            if key == "requested_nodes":
                uint(go[key])
            else:
                _positive(go[key], 65535)
    uint(go["transport_simulation_cap"])
    _uci_moves(inp["allowed_root_moves"], empty=True)
    if go["searchmoves"] is not None and _uci_moves(go["searchmoves"]) != inp["allowed_root_moves"]:
        raise ValueError("Collection teacher original resolved searchmoves/order differs")
    if go["coverage"] == "semantic_context_only":
        if go["original_parsed_limits"] is not None:
            raise ValueError("Collection teacher semantic go cannot claim original parsed limits")
    elif go["coverage"] == "full_original_go":
        raw = fields(go["original_parsed_limits"], ("searchmoves", "depth_plies", "mate_moves", "movetime_ms",
            "white_time_ms", "black_time_ms", "white_increment_ms", "black_increment_ms", "moves_to_go",
            "nodes", "infinite", "ponder"), "Collection teacher original parsed go")
        for key in ("infinite", "ponder"):
            if type(raw[key]) is not bool:
                raise ValueError("Collection teacher original go boolean differs")
        for key in set(raw) - {"searchmoves", "infinite", "ponder"}:
            if raw[key] is not None:
                maximum = 65535 if key in ("depth_plies", "mate_moves") else 2**32 - 1 if key == "moves_to_go" else _U64
                uint(raw[key], maximum)
                if key in ("depth_plies", "mate_moves") and raw[key] == 0:
                    raise ValueError("Collection teacher original go depth/mate must be positive")
        if any(raw[a] != go[b] for a, b in (("nodes", "requested_nodes"), ("depth_plies", "depth_plies"), ("mate_moves", "mate_moves"))):
            raise ValueError("Collection teacher parsed go differs from original resolved limits")
        if ((raw["searchmoves"] is None) != (go["searchmoves"] is None)
                or raw["searchmoves"] is not None and set(_uci_moves(raw["searchmoves"], raw=True)) != set(go["searchmoves"])):
            raise ValueError("Collection teacher parsed/resolved root restriction differs")
    else:
        raise ValueError("Collection teacher original-go coverage differs")
    clocks = fields(inp["clock"], ("hard", "soft", "admission", "cleanup"), "Collection teacher root clocks")
    for clock in clocks.values():
        fields(clock, ("clock_epoch", "at_ns"), "Collection teacher original root deadline")
        if uint(clock["clock_epoch"]) != authority["epoch"]:
            raise ValueError("Collection teacher root descriptor clock domain differs")
        uint(clock["at_ns"])
    if not clocks["soft"]["at_ns"] <= clocks["admission"]["at_ns"] <= clocks["hard"]["at_ns"] <= clocks["cleanup"]["at_ns"]:
        raise ValueError("Collection teacher original root deadline order differs")
    ceilings = fields(inp["declared_ceilings"], ("resource_plan", "registered_max_chess_plies", "cpu_credit_limit",
        "questions_max", "ancestor_depth_max", "waiting_jobs_max", "role_tickets_max", "wall_time_max_ms",
        "cleanup_max_ms", "output_bytes_max"), "Collection teacher declared root ceilings")
    resources = fields(ceilings["resource_plan"], ("cpu_quota_millicpus", "pools", "memory_high_bytes", "memory_max_bytes",
        "gpu_devices", "physical_nn_concurrency", "topology_policy"), "Collection teacher declared resource plan")
    _positive(resources["cpu_quota_millicpus"], 256000)
    _positive(resources["memory_high_bytes"], 1 << 50)
    _positive(resources["memory_max_bytes"], 1 << 50)
    uint(resources["gpu_devices"], 2**32 - 1)
    if uint(resources["physical_nn_concurrency"]) != 1 or resources["memory_high_bytes"] > resources["memory_max_bytes"]:
        raise ValueError("Collection teacher declared memory/physical concurrency differs")
    topology = resources["topology_policy"]
    if (type(topology) is not str or not topology or len(topology.encode("utf-8")) > 4096
            or topology.strip() != topology or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in topology)
            or topology.lower() in ("unknown", "latest", "placeholder", "todo", "tbd", "unset")):
        raise ValueError("Collection teacher declared topology text differs")
    pools = resources["pools"]
    if type(pools) is not list or not 1 <= len(pools) <= 32:
        raise ValueError("Collection teacher bounded declared thread pools required")
    for pool in pools:
        fields(pool, ("actors", "max_threads", "cpu_charge_millicpus"), "Collection teacher declared thread pool")
        actors = pool["actors"]
        if type(actors) is not list or not 1 <= len(actors) <= 16:
            raise ValueError("Collection teacher bounded declared pool actors required")
        # serde serializes BTreeSet<RoventActor> in enum declaration order.
        order = ("manager", "pc_caller", "ort_intra", "ort_inter", "own_r", "external_r", "teacher", "runtime_helper")
        if any(type(actor) is not str or actor not in order for actor in actors):
            raise ValueError("Collection teacher declared pool actor enum differs")
        if actors != sorted(set(actors), key=order.index):
            raise ValueError("Collection teacher declared pool actor canonical set order differs")
        _positive(pool["max_threads"], 1024)
        uint(pool["cpu_charge_millicpus"], 256000)
    for name, maximum in (("registered_max_chess_plies", 65535), ("questions_max", 65536),
        ("ancestor_depth_max", 1024), ("waiting_jobs_max", 1024), ("role_tickets_max", 1024),
        ("wall_time_max_ms", 86400000), ("cleanup_max_ms", 30000), ("output_bytes_max", 64 << 30)):
        _positive(ceilings[name], maximum)
    uint(ceilings["cpu_credit_limit"])
    if (ceilings["ancestor_depth_max"] > ceilings["questions_max"]
            or clocks["cleanup"]["at_ns"] - clocks["hard"]["at_ns"] > ceilings["cleanup_max_ms"] * 1000000
            or go["depth_plies"] is not None and go["depth_plies"] > ceilings["registered_max_chess_plies"]
            or go["mate_moves"] is not None and 2 * go["mate_moves"] - 1 > ceilings["registered_max_chess_plies"]):
        raise ValueError("Collection teacher declared goal/cleanup relation differs")
    budget = _canonical_domain(BUDGET_DOMAIN, {key: part for key, part in inp.items() if key != "clock"})
    deadline = _canonical_domain(DEADLINE_DOMAIN, {"domain": DEADLINE_DOMAIN, "startup_lock_sha256": inp["startup_lock_sha256"],
        "policy": inp["policy"], "authority": authority, "root_budget_sha256": budget,
        "clock": clocks, "cleanup_max_ms": ceilings["cleanup_max_ms"]})
    if (sha(value["root_budget_sha256"]) != budget or sha(value["deadline_seal_sha256"]) != deadline
            or _canonical_domain(DESCRIPTOR_DOMAIN, value) != sha(expected_sha)):
        raise ValueError("Collection teacher original descriptor domain-NUL seals differ")
    return inp


def validate_root_admission_descriptor(root):
    """Join current descriptor data to its separately sealed native wrapper.

    The native reader validates the closed wrapper/source body first. Its
    generation digest and this domain-NUL descriptor digest stay distinct.
    Original InputLock validation and Rules replay remain with Rust.
    """
    inp = validate_root_descriptor(root["root_descriptor"], root["root_descriptor_sha256"])
    authority, origin = inp["authority"], root["native_origin"]
    for key, original in (("epoch", "process_epoch"), ("game", "game_generation"), ("root", "root_generation")):
        if authority[key] != root[original]:
            raise ValueError("native root descriptor changed original authority")
    if (authority["implementation_sha256"] != root["implementation_sha256"]
            or inp["startup_lock_sha256"] != origin["input_lock_sha256"]
            or inp["state_and_history_sha256"] != root["root_state"]["semantic"]
            or root["root_descriptor"]["root_budget_sha256"] != root["root_budget_sha256"]
            or inp["clock"]["hard"] != {"clock_epoch": root["deadline_clock"], "at_ns": root["deadline_tick"]}
            or inp["allowed_root_moves"] != root["root_moves"]
            or any(inp["original_go"][key] != root[key] for key in ("requested_nodes", "depth_plies", "mate_moves"))):
        raise ValueError("native root descriptor changed original lock/root/budget/go/deadline")
    return inp


