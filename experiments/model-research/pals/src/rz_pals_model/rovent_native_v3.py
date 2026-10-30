"""Exact raw /3 submission metadata, distinct from legacy raw /1 and /2.

Rules replay, loaded native ownership and numerical re-encoding belong to the
Rust owner. These bounded joins retain the original submitted counter mapping;
they cannot create an old native lease or insert a future counter into input.
"""
from __future__ import annotations

from .generation_data import fields, sha, uint, _moves
from .rovent_native_v2 import _COMMON, _submission, validate_position_replay
from .rovent_submitted_input import validate_input

NATIVE_DOMAIN = "rz-rovent-raw-record/3"
REVIEW_DOMAIN = "rz-rovent-submitted-review-evidence/1"


def validate_submitted_record_metadata(snapshot, value):
    validate_input(snapshot)
    records = snapshot["selected_view"]["records"]
    if type(value) is not list or len(value) != len(records) or len(value) > 64:
        raise ValueError("raw /3 original selected metadata allocation differs")
    for index, metadata in enumerate(value):
        fields(metadata, ("local_index", "record_revision", "parent_revision", "supersedes_revision", "origin_state", "source_state",
            "source_revision", "source_line_offset", "original_first_move", "value", "completed_depth", "score_scope", "critical"),
            "raw /3 submitted selected-record metadata")
        if uint(metadata["local_index"], 63) != index:
            raise ValueError("raw /3 original selected metadata order differs")
        for key in ("record_revision", "origin_state", "source_state", "source_revision"):
            uint(metadata[key])
        uint(metadata["source_line_offset"], 65535)
        if metadata["source_line_offset"] + len(records[index]["moves"]) > 256:
            raise ValueError("raw /3 selected suffix exceeds its complete line bound")
        if metadata["original_first_move"] is not None:
            _moves([metadata["original_first_move"]], 1)
        raw_value = metadata["value"]
        if raw_value is not None and (type(raw_value) is not int or not -(2**31) <= raw_value < 2**31):
            raise ValueError("raw /3 selected original value must be a nullable i32")
        uint(metadata["completed_depth"], 65535)
        scope = metadata["score_scope"]
        if scope is not None and (type(scope) is not str or scope not in
                ("FrontierOnly", "CompletedIteration", "RulesTerminal")):
            raise ValueError("raw /3 selected original score scope is not an exact supported variant")
        if type(metadata["critical"]) is not bool:
            raise ValueError("raw /3 selected original critical flag must be a bool")
        # This one flag is copied directly from the original CPU scope by the
        # encoder. rules_terminal instead comes from Rules replay and must not
        # be inferred from score_scope or the original raw value/depth.
        if records[index]["completed_iteration"] != (scope == "CompletedIteration"):
            raise ValueError("raw /3 selected completed-iteration flag differs from its original score scope")
    # Validate every entry before following original relation references.
    for index, metadata in enumerate(value):
        for relation in ("parent", "supersedes"):
            local = records[index][relation]
            revision = metadata[relation + "_revision"]
            if (local is None) != (revision is None):
                raise ValueError("raw /3 selected relation revision nullness differs")
            if local is not None:
                uint(revision)
                if revision != value[local]["record_revision"]:
                    raise ValueError("raw /3 selected relation does not retain its original revision")
    for mapping in snapshot["attack_mapping"]:
        local = mapping["selected_counter_local_index"]
        if local is not None and value[local]["source_revision"] != mapping["selected_counter_revision"]:
            raise ValueError("raw /3 selected counter revision differs from its submitted source")
    return value


def validate_submitted_review(record, value):
    """Join original view metadata without recomputing Rules or view hashes."""
    fields(value, ("schema", "sequence", "generation", "root_state", "policy", "input_sha256", "view_seal_sha256",
        "record_scope_sha256", "anchor_states", "source_anchors", "record_metadata"), "raw /3 original submitted ReviewView evidence")
    if value["schema"] != REVIEW_DOMAIN:
        raise ValueError("raw /3 original submitted ReviewView evidence domain required")
    snapshot, native = record["input"], record["native"]
    ticket = native["raw_submission"]["ticket"]
    for name in ("sequence", "generation", "root_state"):
        uint(value[name])
    for name in ("input_sha256", "view_seal_sha256", "record_scope_sha256"):
        sha(value[name])
    if value["sequence"] != ticket["view_sequence"] or value["view_seal_sha256"] != ticket["view_digest"]:
        raise ValueError("raw /3 original submitted view sequence/seal differs from its frozen ticket")
    root = native["raw_retirement"].get("root_admission")
    if type(root) is not dict or "store_root_state" not in root:
        raise ValueError("raw /3 submitted view requires its original Store root admission")
    if (value["generation"] != uint(native["root_generation"])
            or value["root_state"] != uint(root["store_root_state"])):
        raise ValueError("raw /3 submitted Store generation/root differs from its original root admission")
    view = snapshot["selected_view"]
    policy = fields(value["policy"], ("mode", "token_budget", "max_records", "max_anchors", "required_bundle"),
        "raw /3 submitted review selection policy")
    if (policy["mode"] not in ("current_root_only", "anchored_compact")
            or uint(policy["token_budget"], 512) not in (128, 256, 512) or policy["token_budget"] != view["token_budget"]
            or not 1 <= uint(policy["max_records"], 64) or len(view["records"]) > policy["max_records"]
            or not 1 <= uint(policy["max_anchors"], 4) or len(view["anchors"]) > policy["max_anchors"]
            or policy["mode"] == "current_root_only" and policy["max_anchors"] != 1):
        raise ValueError("raw /3 submitted policy changed original selected allocations")
    if policy["required_bundle"] is not None:
        uint(policy["required_bundle"])
    anchors = value["source_anchors"]
    if type(anchors) is not list or len(anchors) != len(view["records"]):
        raise ValueError("raw /3 selected original source replay coverage differs")
    for anchor in anchors:
        validate_position_replay(anchor)
    validate_submitted_record_metadata(snapshot, value["record_metadata"])
    states = value["anchor_states"]
    if type(states) is not list or len(states) != len(view["anchors"]):
        raise ValueError("raw /3 submitted anchor StateId allocation differs from selected anchor order/count")
    for state in states:
        uint(state)
    if states[0] != value["root_state"]:
        raise ValueError("raw /3 submitted anchor zero StateId differs from its original Store root")
    for selected, metadata in zip(view["records"], value["record_metadata"], strict=True):
        if metadata["origin_state"] != states[selected["anchor"]]:
            raise ValueError("raw /3 selected record origin StateId differs from its explicit original anchor")
    return value


def validate_native_v3(manifest, record):
    """Consume only actual raw /3 bodies; never retag them as a /2 receipt."""
    from .rovent_corpus_codec import SUBMITTED_CODEC, validate_record_codec
    if validate_record_codec(manifest, record) is not SUBMITTED_CODEC:
        raise ValueError("raw /3 validator requires the explicit submitted corpus family")
    return _validate_native_v3_body(manifest, record)


def validate_native_v3_evidence(manifest, value):
    """Check a bounded future invocation projection without creating a row.

    The projection contains only the original input and native lifecycle. It
    neither imports future labels into an earlier input nor reconstructs an
    accepted owner. Rust separately verifies legal replay and original custody.
    """
    fields(value, ("input_json", "input_sha256", "native"), "raw /3 future invocation evidence")
    from .rovent_corpus_codec import SUBMITTED_CODEC, codec_from_manifest
    from .rovent_submitted_input import seal_input
    from .rovent_corpus import _decode_json
    if codec_from_manifest(manifest) is not SUBMITTED_CODEC:
        raise ValueError("raw /3 future evidence requires the submitted corpus family")
    if (type(value["native"]) is not dict
            or not {"raw_submission", "raw_completion", "raw_retirement"} <= set(value["native"])):
        raise ValueError("raw /3 future invocation requires its original native lifecycle object")
    # Keep the producer's original JSON spelling inside the outer raw seal.
    # In particular nested f32 values must not be canonicalized a second time
    # using another language's exponent/integral-float formatting rules.
    if type(value["input_json"]) is not str:
        raise ValueError("raw /3 future invocation input requires its original UTF-8 JSON string")
    try:
        raw = value["input_json"].encode("utf-8")
    except UnicodeError as error:
        raise ValueError("raw /3 future invocation input requires strict UTF-8") from error
    if not 1 <= len(raw) <= 1024 * 1024:
        raise ValueError("raw /3 future invocation input exceeds its original row bound")
    snapshot = validate_input(_decode_json(raw))
    if seal_input(snapshot) != sha(value["input_sha256"]):
        raise ValueError("raw /3 future evidence changed its original sealed input")
    # An ephemeral input-only adapter for shared pure lifecycle checks; this
    # object is never serialized, admitted as a training record, or published.
    projection = {"input": snapshot, "input_sha256": value["input_sha256"],
                  "native": value["native"], "role": snapshot["role"]}
    _validate_native_v3_body(manifest, projection)
    return projection


def _validate_native_v3_body(manifest, record):
    validate_input(record["input"])
    native = record["native"]
    extra = {
        "raw_submission": ("model_sha256", "encoding_sha256", "input_digest", "view_digest", "rules_replay",
            "submitted_review"),
        "raw_completion": ("submitted_sha256", "execution_epoch", "execution_sequence", "completed_nn_inputs",
            "physical_confirmed", "owner_run_invoked", "measurement_complete", "physical_fresh", "output_digest",
            "owner_service_elapsed_ns", "output"),
        "raw_retirement": ("submitted_sha256", "completed_sha256", "output_digest", "accepted_retired",
            "acceptance_kind", "root_admission", "root_admission_sha256"),
    }
    for name, keys in extra.items():
        raw = fields(native[name], (*_COMMON, *keys), "raw /3 original " + name)
        if raw["schema"] != NATIVE_DOMAIN:
            raise ValueError("mixed raw versions rejected; explicit raw /3 required")
        for key in ("sequence", "process_epoch", "game_generation", "root_generation", "request_sequence"):
            uint(raw[key])
    for key in ("execution_epoch", "execution_sequence", "completed_nn_inputs"):
        uint(native["raw_completion"][key])
    elapsed = native["raw_completion"]["owner_service_elapsed_ns"]
    if elapsed is not None:
        uint(elapsed)
    retired = native["raw_retirement"]
    if (retired["acceptance_kind"] not in ("accepted_role_step", "root_ranking_published")
            or retired["acceptance_kind"] == "root_ranking_published" and record["role"] != "proposer"):
        raise ValueError("raw /3 original acceptance kind/role differs")
    from .rovent_corpus import _native_common
    _native_common(manifest, record, NATIVE_DOMAIN)
    _submission(record, native["raw_submission"]["rules_replay"])
    from .rovent_external_root import validate_submitted_root
    validate_submitted_root(manifest, record, retired)
    validate_submitted_review(record, native["raw_submission"]["submitted_review"])
    return native
