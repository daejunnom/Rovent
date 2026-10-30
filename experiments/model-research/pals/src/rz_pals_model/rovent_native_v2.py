"""Explicit native raw /2 data joins; no Rules or native authority recreation.

The Rust parent replays exact chess state/history and validates original loaded
locks. Python checks typed wire structure, bounded replay data and seals. No
field below proves that a historical producer or device is currently alive.
"""
from __future__ import annotations

from .generation_data import digest, fields, identifier, sha, uint, _anchor, _moves

NATIVE_DOMAIN = "rz-rovent-native-owner-event/2"
ROOT_DOMAIN = "rz-rovent-collection-root-admission/1"
ORIGIN_DOMAIN = "rz-rovent-collection-native-origin/1"
POSITION_DOMAIN = "rz-rovent-rules-position-replay/1"
SUBMISSION_DOMAIN = "rz-rovent-submitted-rules-replay/1"
TRAJECTORY_DOMAIN = "rz-rovent-actual-rules-trajectory-replay/1"
_COMMON = ("schema", "kind", "sequence", "process_epoch", "game_generation", "root_generation",
           "request_sequence", "head", "source_sha256", "input_sha256", "ticket")


def _text(value, maximum, description):
    if (type(value) is not str or not value or len(value.encode("utf-8")) > maximum
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)):
        raise ValueError(f"bounded native /2 {description} required")
    return value


def _commit(value):
    if type(value) is not str or len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("native /2 full collection commit required")
    return value


def _artifact(value):
    fields(value, ("path", "sha256", "bytes", "source", "license"), "native /2 declared artifact")
    for key in ("path", "source", "license"):
        _text(value[key], 4096, "artifact metadata")
    sha(value["sha256"])
    if uint(value["bytes"]) == 0:
        raise ValueError("native /2 nonempty declared artifact required")


def _profile(value):
    fields(value, ("domain", "profile", "canonical_sha256", "capability_sha256", "purpose", "binary",
                   "source_commit", "requested_threads", "requested_hash_mib"), "native /2 teacher profile")
    if value["domain"] != "rz-rovent-external-engine-profile/1" or value["purpose"] != "teacher":
        raise ValueError("native /2 original teacher profile domain/purpose required")
    _artifact(value["profile"])
    _artifact(value["binary"])
    sha(value["canonical_sha256"])
    sha(value["capability_sha256"])
    _commit(value["source_commit"])
    if uint(value["requested_threads"], 256) == 0 or uint(value["requested_hash_mib"], 1048576) == 0:
        raise ValueError("native /2 positive teacher resource declaration required")


def validate_position_replay(value):
    """Check replay bytes/anchor shape only; Rust performs legal replay."""
    fields(value, ("schema", "origin", "completeness", "start_fen", "moves", "observer_revision", "anchor"),
           "native /2 Rules position replay")
    if (value["schema"] != POSITION_DOMAIN or value["origin"] not in ("start_position", "fen")
            or value["completeness"] not in ("complete", "unknown_prefix")
            or (value["origin"] == "start_position" and value["completeness"] != "complete")):
        raise ValueError("native /2 explicit Rules history origin/completeness required")
    _text(value["start_fen"], 512, "Rules replay start FEN")
    _moves(value["moves"], 16384)
    uint(value["observer_revision"])
    _anchor(value["anchor"])
    return value


def _submission(record, value):
    fields(value, ("schema", "actual_root", "selected_anchors"), "native /2 submitted Rules replay")
    snapshot = record["input"]
    if value["schema"] != SUBMISSION_DOMAIN:
        raise ValueError("native /2 submitted Rules replay domain required")
    actual = validate_position_replay(value["actual_root"])
    anchors = value["selected_anchors"]
    if (type(anchors) is not list or not 1 <= len(anchors) <= 4
            or len(anchors) != len(snapshot["selected_view"]["anchors"])):
        raise ValueError("native /2 submitted Rules replay anchor allocation mismatch")
    for replay, anchor in zip(anchors, snapshot["selected_view"]["anchors"]):
        if validate_position_replay(replay)["anchor"] != anchor:
            raise ValueError("native /2 selected replay anchor differs from original input")
    if (actual["moves"] != snapshot["actual_history"] or actual["anchor"] != anchors[0]["anchor"]
            or any(actual[key] != anchors[0][key] for key in
                   ("origin", "completeness", "start_fen", "moves", "observer_revision"))):
        raise ValueError("native /2 original actual history differs from anchor zero")


def _root_admission(manifest, record, retired):
    root = retired["root_admission"]
    descriptor = type(root) is dict and bool({"root_descriptor", "root_descriptor_sha256"} & set(root))
    extra = ("root_descriptor", "root_descriptor_sha256") if descriptor else ()
    root = fields(root, ("schema", "native_origin", "native_origin_sha256",
        "process_epoch", "game_generation", "root_generation", "implementation_sha256", "deadline_clock",
        "deadline_tick", "root_budget_sha256", "selective_policy_sha256", "root_state", "store_root_state",
        "root_moves", "depth_plies", "mate_moves", "requested_nodes", *extra), "native /2 original root admission")
    if root["schema"] != ROOT_DOMAIN or digest(ROOT_DOMAIN, root) != sha(retired["root_admission_sha256"]):
        raise ValueError("native /2 original root admission seal mismatch")
    _root_admission_common(manifest, record, root, descriptor=descriptor)


def _root_admission_common(manifest, record, root, *, descriptor):
    """Share exact owner/source joins after a codec checks its own wire seal.

    This body does not choose a root schema, project one into another, or issue
    native authority. The caller preserves its original object and decides the
    descriptor variant only from that codec's already checked exact roster.
    """
    native, ticket = record["native"], record["native"]["raw_submission"]["ticket"]
    for key in ("process_epoch", "game_generation", "root_generation", "deadline_clock", "deadline_tick"):
        if uint(root[key]) != ticket[key]:
            raise ValueError("native /2 root admission crossed original request scope/deadline")
    if root["implementation_sha256"] != ticket["implementation_sha256"] or root["root_state"] != ticket["root_state"]:
        raise ValueError("native /2 root admission changed implementation/root state")
    for key in ("implementation_sha256", "root_budget_sha256", "selective_policy_sha256"):
        sha(root[key])
    uint(root["store_root_state"])
    for key in ("depth_plies", "mate_moves", "requested_nodes"):
        if root[key] is not None:
            parsed = uint(root[key], 2**64 - 1 if key == "requested_nodes" else 65535)
            # Current RootDescriptor carries Option<u64>, including zero, for
            # nodes. Preserve the historical no-descriptor admission rule.
            if parsed == 0 and (key != "requested_nodes" or not descriptor):
                raise ValueError("native /2 explicit positive root search limit required")
    moves = root["root_moves"]
    if (type(moves) is not list or not 1 <= len(moves) <= 218
            or any(type(move) is not str for move in moves) or len(set(moves)) != len(moves)):
        raise ValueError("native /2 bounded unique declared root moves required")
    for move in moves:
        if (type(move) is not str or len(move) not in (4, 5)
                or move[0] not in "abcdefgh" or move[1] not in "12345678"
                or move[2] not in "abcdefgh" or move[3] not in "12345678"
                or (len(move) == 5 and move[4] not in "nbrq")):
            raise ValueError("native /2 declared root move UCI shape required")
    origin = fields(root["native_origin"], ("schema", "purpose", "process_epoch", "mode", "input_lock_sha256",
        "source_commit", "source_components", "source_sha256", "weights_sha256", "export_manifest_sha256",
        "backend_sha256", "runtime_library_sha256", "runtime_bundle_sha256", "native_configuration_sha256",
        "model_semantics", "model_semantics_sha256", "encoding", "encoding_sha256", "encoding_registry_sha256",
        "model_implementation_sha256", "frozen_epoch", "representation_model_domain", "runtime_sha256_domain",
        "runtime_sha256", "teacher_profiles"), "native /2 original loaded origin")
    if (origin["schema"] != ORIGIN_DOMAIN or origin["purpose"] != "collection"
            or digest(ORIGIN_DOMAIN, origin) != sha(root["native_origin_sha256"])
            or uint(origin["process_epoch"]) != native["process_epoch"]
            or origin["mode"] not in ("HybridPcr", "HybridPc", "CpuPcr", "CpuPc")):
        raise ValueError("native /2 original loaded collection origin mismatch")
    for key in ("input_lock_sha256", "source_sha256", "weights_sha256", "export_manifest_sha256",
                "backend_sha256", "runtime_library_sha256", "native_configuration_sha256",
                "model_semantics_sha256", "encoding_sha256", "encoding_registry_sha256",
                "model_implementation_sha256", "runtime_sha256"):
        sha(origin[key])
    if origin["runtime_bundle_sha256"] is not None:
        sha(origin["runtime_bundle_sha256"])
    _commit(origin["source_commit"])
    components = origin["source_components"]
    if type(components) is not dict or not 1 <= len(components) <= 64:
        raise ValueError("native /2 bounded original source component map required")
    for name, value_sha in components.items():
        _text(name, 4096, "source component name")
        sha(value_sha)
    for key in ("model_semantics", "encoding"):
        _text(origin[key], 4096, "model semantics/encoding")
    if (origin["source_commit"] != manifest["source_commit"] or origin["source_sha256"] != manifest["source_sha256"]
            or origin["runtime_sha256_domain"] != "loaded_runtime_library_file_sha256"
            or origin["runtime_sha256"] != origin["runtime_library_sha256"]
            or origin["runtime_library_sha256"] != manifest["runtime_sha256"]
            or origin["representation_model_domain"] != "loaded_checkpoint_sha256"
            or origin["weights_sha256"] != native["model_sha256"]
            or origin["encoding_sha256"] != native["encoding_sha256"]
            or origin["model_semantics_sha256"] != manifest["model_semantics_sha256"]
            or uint(origin["frozen_epoch"]) != ticket["representation"]["frozen_epoch"]):
        raise ValueError("native /2 original collection source/runtime/model domain mismatch")
    profiles = origin["teacher_profiles"]
    if type(profiles) is not list or len(profiles) > 2:
        raise ValueError("native /2 bounded original teacher profiles required")
    for profile in profiles:
        _profile(profile)
    for pin in manifest["teachers"].values():
        if not any(profile["canonical_sha256"] == pin["canonical_sha256"]
                   and profile["capability_sha256"] == pin["capability_sha256"]
                   and profile["profile"]["sha256"] == pin["profile_sha256"]
                   and profile["binary"]["sha256"] == pin["binary_sha256"]
                   and profile["binary"]["bytes"] == pin["binary_bytes"]
                   and profile["source_commit"] == pin["source_commit"] for profile in profiles):
            raise ValueError("native /2 original loaded lock lacks frozen corpus teacher")
    if descriptor:
        # Explicit current variant only. Historical no-descriptor bytes and
        # digest remain unchanged; neither variant reconstructs a live lock.
        from .rovent_root_descriptor import validate_root_admission_descriptor
        validate_root_admission_descriptor(root)


def validate_native_v2(manifest, record):
    """Validate /2 itself; never rewrite it into /1 to reuse checks."""
    native = record["native"]
    extras = {
        "raw_submission": ("model_sha256", "encoding_sha256", "input_digest", "view_digest", "rules_replay"),
        "raw_completion": ("submitted_sha256", "execution_epoch", "execution_sequence", "completed_nn_inputs",
            "physical_confirmed", "owner_run_invoked", "measurement_complete", "physical_fresh", "output_digest",
            "owner_service_elapsed_ns", "output"),
        "raw_retirement": ("submitted_sha256", "completed_sha256", "output_digest", "accepted_retired",
            "acceptance_kind", "root_admission", "root_admission_sha256"),
    }
    for name, names in extras.items():
        raw = fields(native[name], (*_COMMON, *names), "native /2 original " + name)
        if raw["schema"] != NATIVE_DOMAIN:
            raise ValueError("mixed native raw versions rejected; explicit /2 required")
        for key in ("sequence", "process_epoch", "game_generation", "root_generation", "request_sequence"):
            uint(raw[key])
    for key in ("execution_epoch", "execution_sequence", "completed_nn_inputs"):
        uint(native["raw_completion"][key])
    elapsed = native["raw_completion"]["owner_service_elapsed_ns"]
    if elapsed is not None:
        uint(elapsed)
    retired = native["raw_retirement"]
    if (retired["acceptance_kind"] not in ("accepted_role_step", "root_ranking_published")
            or (retired["acceptance_kind"] == "root_ranking_published" and record["role"] != "proposer")):
        raise ValueError("native /2 original acceptance kind/role mismatch")
    from .rovent_corpus import _native_common
    _native_common(manifest, record, NATIVE_DOMAIN)
    _submission(record, native["raw_submission"]["rules_replay"])
    _root_admission(manifest, record, retired)
    return native


def validate_wdl_replay_v2(record, raw, label):
    """Join actual trace metadata; only Rust checks legality and termination."""
    trace = fields(raw["rules_replay"], ("schema", "initial", "actual_selected_moves", "final_position",
                                        "completion", "claim"), "native /2 actual Rules trajectory replay")
    if trace["schema"] != TRAJECTORY_DOMAIN or trace["completion"] not in ("automatic", "current_claim"):
        raise ValueError("native /2 explicit actual Rules trajectory domain/completion required")
    initial = validate_position_replay(trace["initial"])
    final = validate_position_replay(trace["final_position"])
    _moves(trace["actual_selected_moves"], 512)
    if (trace["completion"] == "automatic" and trace["claim"] is not None
            or trace["completion"] == "current_claim" and trace["claim"] not in ("threefold_repetition", "fifty_move")):
        raise ValueError("native /2 actual Rules completion/claim mismatch")
    authority = raw["trajectory"]
    anchors = authority.get("anchors") if type(authority) is dict else None
    if (type(anchors) is not list or len(anchors) != len(trace["actual_selected_moves"]) + 1
            or not anchors or anchors[0] != initial["anchor"] or anchors[-1] != final["anchor"]
            or record["input"]["rules_anchor"] not in anchors or label["anchor"] != record["input"]["rules_anchor"]
            or any(initial[key] != final[key] for key in ("origin", "completeness", "start_fen"))
            or final["moves"] != initial["moves"] + trace["actual_selected_moves"]):
        raise ValueError("native /2 actual WDL trace does not join original trajectory anchors/history")
    for anchor in anchors:
        _anchor(anchor)
