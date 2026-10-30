"""Strict historical root-local utility observations, never cohort targets.

The original publisher closes a private action and its whole-route cost. This
reader checks the published raw-byte frame and metadata joins; it cannot prove
Rules replay, reopen retained artifacts, recover a live source or physical
fence, or promote a root-local observation into known utility supervision.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .generation_data import digest, fields, sha, uint, _moves
from . import nnuec_reference as numeric
from . import nnuec_v2_reference as relation
from .rovent_utility_input import validate_submitted_utility_input
from .rovent_external_root import ROOT_DOMAIN as EXTERNAL_ROOT, ROOT_FIELDS, validate_external_source
from .rovent_native_v2 import _artifact, _commit, _profile
from .rovent_root_descriptor import validate_root_admission_descriptor
from .rovent_divergence_owner import _origin as _accepted_origin, _ticket


ACTION_DOMAIN = "rz-rovent-utility-root-local-action/1"
PAIR_DOMAIN = "rz-rovent-utility-root-local-pair/1"
MAX_PUBLICATION_BYTES = 32 << 20
_AUTHORITY = ("process_epoch", "game_generation", "root_generation", "implementation_sha256")
_ACTION_FIELDS = ("schema", "collection_sequence", "source", "input", "action", "semantic", "cost", "support_mask", "target_known_mask")
_PAIR_FIELDS = ("schema", "collection_sequence", "left_action_sha256", "right_action_sha256", "original_comparison_sha256",
    "support_mask", "target_known_mask", "preferred", "unknown_reason")
_SOURCE_FIELDS = ("root_admission_sha256", "root_admission", "native_origin_sha256", "input_lock_sha256", "nnuec_registration",
    "weights_artifact", "provenance_artifact")
_INPUT_FIELDS = ("reference_json", "bytes", "input_sha256", "target_sha256", "submission_target_sha256", "context_sha256",
    "target_record", "target_record_ids", "base_identity", "root_rules", "records_metadata", "source_rules", "provenance")
_ACTION_BODY = ("authority", "frame_sequence", "question_sha256", "recipient", "output_index", "manager_sequence",
    "original_first_move", "root_budget_sha256", "original_deadline", "mode", "budget", "frozen_context_sha256",
    "comparison_group_sha256", "original_action_sha256")
_SEMANTIC_FIELDS = ("baseline_witnesses", "new_witnesses", "pc_consumption", "r_selected_pc_consumption", "dependencies", "native_executions")
_COST_FIELDS = ("scope", "objective", "resource_profile_sha256", "cost_epoch_sha256", "whole_wall_ns", "baseline_counters",
    "final_counters", "native_credit_baseline", "native_credit_final", "cpu_thread_time_ns", "energy", "retained_host_bytes")
_COUNTERS = ("native_preparations", "submitted_role_tickets", "physical_role_completions", "completed_nn_inputs", "nn_input_count_unknown",
    "consumed_role_outputs", "accepted_nn_inputs", "accepted_nn_input_count_unknown", "consumed_nn_inputs", "consumed_nn_input_count_unknown",
    "zero_nn_outputs_consumed", "unused_role_results", "canceled_preparations", "cpu_slice_receipts", "cpu_progress_publications",
    "cpu_yields", "cpu_yield_requests", "cpu_yield_acknowledgements", "cpu_natural_completions", "validator_ticket_admissions",
    "foreign_reports", "foreign_reported_nodes", "foreign_node_count_unknown", "foreign_yield_requests", "foreign_yield_acknowledgements",
    "rejected_events", "notification_waits")
_UNKNOWN_COUNTERS = ("nn_input_count_unknown", "accepted_nn_input_count_unknown", "consumed_nn_input_count_unknown", "foreign_node_count_unknown")
_CONTEXT_POLICY = "exact-original-review-view:anchor-rules-history:kind4-parent-supersedes-origin-first:immutable-model:cancel-deadline-epoch:typed-owner-v2"
_LOCAL_POLICY = "exact-begin-rules-history:ordered-checked-move16:append-undo:fixed-context-authority:same-owner-resume-v2"
_COST_SCOPE = "exclusive_submitted_global_target_route_and_descendants_with_required_drain"
_OBJECTIVE = "exclusive_wall_nanoseconds"
_ROUTES = {"route_p": 4, "route_c": 5, "route_r": 6}
_UNKNOWN_REASONS = ("different_frozen_group", "duplicate_actual_action", "same_recipient", "no_new_valid_witness",
    "equal_observation", "pareto_incomparable", "missing_comparison_cohort")


def _positive(value, maximum=2**64 - 1):
    if not uint(value, maximum):
        raise ValueError("positive root-local original sequence required")
    return value


def _text(value, maximum=4096):
    if (type(value) is not str or not value or len(value.encode("utf-8")) > maximum
            or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in value)):
        raise ValueError("bounded root-local metadata text required")
    return value


def _array(value, maximum=64):
    if type(value) is not list or len(value) > maximum:
        raise ValueError("bounded root-local original array required")
    return value


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("root-local original JSON contains duplicate keys")
        value[key] = item
    return value


def _constant(_name):
    raise ValueError("root-local metadata contains a nonfinite JSON constant")


def _decode(raw, expected_sha256, domain):
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_PUBLICATION_BYTES:
        raise ValueError("bounded immutable root-local original bytes required")
    actual = hashlib.sha256(domain.encode("ascii") + b"\0" + len(raw).to_bytes(8, "little") + raw).hexdigest()
    if actual != sha(expected_sha256):
        raise ValueError("root-local original domain/length/raw-byte seal mismatch")
    try:
        value = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=_object, parse_constant=_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("root-local original UTF8 JSON required") from error
    remaining = [MAX_PUBLICATION_BYTES]
    def walk(item, depth=0):
        remaining[0] -= 1
        if depth > 64 or remaining[0] < 0:
            raise ValueError("root-local JSON structure capacity exceeded")
        if type(item) is dict:
            for nested in item.values():
                walk(nested, depth + 1)
        elif type(item) is list:
            for nested in item:
                walk(nested, depth + 1)
        elif type(item) not in (str, int, bool, type(None)):
            # Native numeric input floats belong to reference_json's original
            # string, not a second outer JSON number spelling.
            raise ValueError("root-local outer metadata requires integer/string scalar wire")
    walk(value)
    return value


def _authority(value, expected=None):
    fields(value, _AUTHORITY, "root-local original authority")
    _positive(value["process_epoch"])
    uint(value["game_generation"])
    uint(value["root_generation"])
    sha(value["implementation_sha256"])
    if expected is not None and value != expected:
        raise ValueError("root-local original authority differs")
    return value


def _deadline(value, expected=None):
    fields(value, ("clock", "tick"), "root-local original deadline")
    uint(value["clock"])
    uint(value["tick"])
    if expected is not None and value != expected:
        raise ValueError("root-local original deadline differs")
    return value


def _identity(value):
    fields(value, ("semantics", "weights_sha256", "training"), "root-local frozen base identity")
    if value["semantics"] != relation.SEMANTICS:
        raise ValueError("root-local requires original origin-first V2 base semantics")
    sha(value["weights_sha256"])
    numeric._training(value["training"])
    return value


def _registration(value, identity):
    fields(value, ("domain", "schema", "semantics", "encoding", "implementation_sha256", "configuration_sha256",
        "context_namespace_policy", "local_namespace_policy", "context_namespace_policy_sha256", "local_namespace_policy_sha256",
        "model_identity"), "root-local original NNUEC registration")
    if (value["domain"] != "rz-nnuec-v2-registration/1" or value["schema"] != relation.SCHEMA
            or value["semantics"] != relation.SEMANTICS or value["encoding"] != relation.INPUT_SCHEMA
            or value["context_namespace_policy"] != _CONTEXT_POLICY or value["local_namespace_policy"] != _LOCAL_POLICY
            or _identity(value["model_identity"]) != identity):
        raise ValueError("root-local original NNUEC registration/model namespace mismatch")
    for key in ("implementation_sha256", "configuration_sha256", "context_namespace_policy_sha256", "local_namespace_policy_sha256"):
        sha(value[key])
    for name, domain in (("context", "rz-rovent-nnuec-context-policy/1"), ("local", "rz-rovent-nnuec-local-policy/1")):
        encoded = json.dumps(value[name + "_namespace_policy"], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        expected = hashlib.sha256(domain.encode("ascii") + b"\0" + encoded).hexdigest()
        if value[name + "_namespace_policy_sha256"] != expected:
            raise ValueError("root-local original NNUEC policy declaration seal mismatch")


def _root(source):
    fields(source, _SOURCE_FIELDS, "root-local original source")
    root = source["root_admission"]
    schema = root.get("schema") if type(root) is dict else None
    if schema == EXTERNAL_ROOT:
        fields(root, ROOT_FIELDS, "root-local original external root")
    elif schema == "rz-rovent-collection-root-admission/1":
        fields(root, tuple(name for name in ROOT_FIELDS if name not in ("external_source", "external_source_sha256")),
            "root-local original root")
    else:
        raise ValueError("root-local original descriptor-bearing Collection root required")
    if digest(schema, root) != sha(source["root_admission_sha256"]):
        raise ValueError("root-local original root admission seal mismatch")
    origin = fields(root["native_origin"], ("schema", "purpose", "process_epoch", "mode", "input_lock_sha256",
        "source_commit", "source_components", "source_sha256", "weights_sha256", "export_manifest_sha256", "backend_sha256",
        "runtime_library_sha256", "runtime_bundle_sha256", "native_configuration_sha256", "model_semantics", "model_semantics_sha256",
        "encoding", "encoding_sha256", "encoding_registry_sha256", "model_implementation_sha256", "frozen_epoch",
        "representation_model_domain", "runtime_sha256_domain", "runtime_sha256", "teacher_profiles"), "root-local original native origin")
    if (origin["schema"] != "rz-rovent-collection-native-origin/1" or origin["purpose"] != "collection"
            or digest(origin["schema"], origin) != sha(root["native_origin_sha256"])
            or source["native_origin_sha256"] != root["native_origin_sha256"]
            or sha(source["input_lock_sha256"]) != origin["input_lock_sha256"]
            or origin["process_epoch"] != root["process_epoch"]
            or origin["mode"] not in ("HybridPcr", "HybridPc", "CpuPcr", "CpuPc")
            or origin["representation_model_domain"] != "loaded_checkpoint_sha256"
            or origin["runtime_sha256_domain"] != "loaded_runtime_library_file_sha256"
            or origin["runtime_sha256"] != origin["runtime_library_sha256"]):
        raise ValueError("root-local original native source/lock/domain mismatch")
    _commit(origin["source_commit"])
    _positive(origin["process_epoch"])
    for key in ("input_lock_sha256", "source_sha256", "weights_sha256", "export_manifest_sha256", "backend_sha256",
            "runtime_library_sha256", "native_configuration_sha256", "model_semantics_sha256", "encoding_sha256",
            "encoding_registry_sha256", "model_implementation_sha256", "runtime_sha256"):
        sha(origin[key])
    if origin["runtime_bundle_sha256"] is not None:
        sha(origin["runtime_bundle_sha256"])
    for key in ("model_semantics", "encoding"):
        _text(origin[key])
    uint(origin["frozen_epoch"])
    components = origin["source_components"]
    if type(components) is not dict or not 1 <= len(components) <= 64:
        raise ValueError("root-local bounded original source components required")
    aggregate = b"rz-rovent-source-aggregate/1\0" + origin["source_commit"].encode("ascii") + len(components).to_bytes(4, "little")
    for name, pin in sorted(components.items()):
        encoded = _text(name).encode("utf-8")
        aggregate += len(encoded).to_bytes(4, "little") + encoded + sha(pin).encode("ascii")
    if hashlib.sha256(aggregate).hexdigest() != origin["source_sha256"]:
        raise ValueError("root-local original source component aggregate mismatch")
    for profile in _array(origin["teacher_profiles"], 2):
        _profile(profile)
    authority = _authority({name: root[name] for name in _AUTHORITY})
    for key in ("deadline_clock", "deadline_tick", "store_root_state"):
        uint(root[key])
    for key in ("depth_plies", "mate_moves", "requested_nodes"):
        if root[key] is not None:
            uint(root[key], 2**64 - 1 if key == "requested_nodes" else 65535)
            if key != "requested_nodes" and not root[key]:
                raise ValueError("root-local original positive depth/mate limit required")
    for key in ("root_budget_sha256", "selective_policy_sha256"):
        sha(root[key])
    fields(root["root_state"], ("owner", "revision", "semantic"), "root-local original root state")
    uint(root["root_state"]["owner"])
    uint(root["root_state"]["revision"])
    sha(root["root_state"]["semantic"])
    validate_root_admission_descriptor(root)
    if schema == EXTERNAL_ROOT:
        if origin["mode"] not in ("HybridPcr", "CpuPcr"):
            raise ValueError("root-local external root requires its original P/C/R mode")
        validate_external_source(root)
    for key, cap in (("weights_artifact", 32 << 20), ("provenance_artifact", 256 << 10)):
        _artifact(source[key])
        if source[key]["bytes"] > cap:
            raise ValueError("root-local original NNUEC artifact byte cap exceeded")
    return root, authority


def _rules(value):
    fields(value, ("rules_version", "variant", "semantic_sha256", "fen", "history_origin", "history_completeness", "known_history"),
        "root-local full Rules snapshot")
    for name in ("rules_version", "variant", "fen"):
        _text(value[name])
    if value["rules_version"] != "rz-position/0.1.0" or value["variant"] != "standard_chess":
        raise ValueError("root-local original Rules version/variant differs")
    sha(value["semantic_sha256"])
    if (value["history_origin"] not in ("StartPosition", "Fen")
            or value["history_completeness"] not in ("Complete", "UnknownPrefix")
            or value["history_origin"] == "StartPosition" and value["history_completeness"] != "Complete"):
        raise ValueError("root-local Rules history origin/completeness mismatch")
    history = _array(value["known_history"], MAX_PUBLICATION_BYTES)
    for fen in history:
        _text(fen)
    if not history or history[0] != value["fen"]:
        raise ValueError("root-local full Rules history must retain newest-first current FEN")
    return value


def _input(value, root, config):
    fields(value, _INPUT_FIELDS, "root-local original submitted utility input")
    if type(config) is not numeric.NnuecConfig:
        raise ValueError("root-local explicit source NnuecConfig required")
    identity = _identity(value["base_identity"])
    p = fields(value["provenance"], ("root_state", "generation", "sequence", "full_view_sha256", "view_seal_sha256",
        "record_scope_sha256", "policy_sha256", "model"), "root-local original input provenance")
    for key in ("root_state", "generation", "sequence"):
        uint(p[key])
    for key in ("full_view_sha256", "view_seal_sha256", "record_scope_sha256", "policy_sha256"):
        sha(p[key])
    if (p["root_state"] != root["store_root_state"] or p["generation"] != root["root_generation"]
            or _identity(p["model"]) != identity):
        raise ValueError("root-local input provenance changed original root/model")
    captured = validate_submitted_utility_input(input_json=value["reference_json"], base_weights_sha256=identity["weights_sha256"],
        context_sha256=value["context_sha256"], target_ids=value["target_record_ids"], view_seal_sha256=p["view_seal_sha256"],
        input_sha256=value["input_sha256"], target_sha256=value["target_sha256"],
        submission_target_sha256=value["submission_target_sha256"], config=config)
    if uint(value["bytes"], 8 << 20) != len(captured.input_bytes):
        raise ValueError("root-local original reference JSON byte count differs")
    rules = _rules(value["root_rules"])
    if rules["semantic_sha256"] != root["root_state"]["semantic"]:
        raise ValueError("root-local input full Rules root differs from admission")
    metadata, sources, lines = _array(value["records_metadata"]), _array(value["source_rules"]), captured.reference_input.base_input.context_lines
    if len(metadata) != len(lines) or len(sources) != len(lines):
        raise ValueError("root-local selected record/source Rules allocation mismatch")
    expected_edges, exact_sources = [], {}
    for index, (row, source, line) in enumerate(zip(metadata, sources, lines, strict=True)):
        fields(row, ("anchor", "parent", "supersedes", "source_state", "source_revision", "source_line_offset", "original_first_move"),
            "root-local selected record metadata")
        uint(row["anchor"], 3)
        for key in ("source_state", "source_revision"):
            uint(row[key])
        uint(row["source_line_offset"], 65535)
        if row["original_first_move"] is not None:
            _moves([row["original_first_move"]], 1)
        first = line.original_first_move
        packed = None if first is None else first.from_square | first.to_square << 6 | first.promotion << 12
        if packed != row["original_first_move"]:
            raise ValueError("root-local numeric line differs from original first move")
        for kind, key in enumerate(("parent", "supersedes")):
            if row[key] is not None:
                if uint(row[key], 63) >= index:
                    raise ValueError("root-local original parent/supersedes order differs")
                expected_edges.append((row[key], index, kind))
        source = _rules(source)
        previous = exact_sources.setdefault(row["source_state"], source)
        if previous != source:
            raise ValueError("root-local original source StateId acquired different Rules history")
    actual_edges = [(edge.source, edge.target, edge.kind) for edge in captured.reference_input.base_input.context_edges]
    if actual_edges != expected_edges:
        raise ValueError("root-local numeric relations differ from selected metadata")
    target = uint(value["target_record"], 63)
    if target >= len(metadata) or metadata[target]["original_first_move"] is None:
        raise ValueError("root-local original target record is missing")
    first, source_state = metadata[target]["original_first_move"], metadata[target]["source_state"]
    selected = [index for index, row in enumerate(metadata)
        if row["source_state"] == source_state and row["original_first_move"] == first]
    if selected != value["target_record_ids"]:
        raise ValueError("root-local original exact source/first-move target bundle differs")
    return captured


def _view(value, authority):
    fields(value, ("authority", "sequence", "context_policy_sha256", "view_digest", "root_state", "actual_tokens", "selected_records", "anchors", "length_bucket"),
        "root-local original selected view")
    _authority(value["authority"], authority)
    uint(value["sequence"])
    for name in ("context_policy_sha256", "view_digest"):
        sha(value[name])
    state = fields(value["root_state"], ("owner", "revision", "semantic_sha256"), "root-local view root state")
    uint(state["owner"])
    uint(state["revision"])
    sha(state["semantic_sha256"])
    uint(value["actual_tokens"], 512)
    uint(value["selected_records"], 64)
    if not 1 <= uint(value["anchors"], 4) or value["length_bucket"] not in (8, 16, 32, 64, 128, 256) or type(value["length_bucket"]) is not int:
        raise ValueError("root-local original selected view allocation differs")
    return value


def _legacy_view(value):
    return {"authority": value["authority"], "view_sequence": value["sequence"], "context_policy": value["context_policy_sha256"],
        **{name: value[name] for name in ("view_digest", "actual_tokens", "selected_records", "anchors", "length_bucket")},
        "root_state": {"owner": value["root_state"]["owner"], "revision": value["root_state"]["revision"],
            "semantic": value["root_state"]["semantic_sha256"]}}


def _peer(value, authority):
    fields(value, ("authority", "id", "parent", "sender", "recipient", "purpose", "anchor", "anchor_root_ply", "original_first_move",
        "prefix_sha256", "line_sha256", "line_offset", "question_sha256", "view", "evidence_revision", "root_budget_sha256", "deadline"),
        "root-local original reciprocal peer")
    _authority(value["authority"], authority)
    _positive(value["id"])
    if value["parent"] is not None and uint(value["parent"]) >= value["id"]:
        raise ValueError("root-local original peer parent chronology differs")
    if (value["sender"] not in ("Proposer", "Critic", "Researcher") or value["recipient"] not in ("Proposer", "Critic", "Researcher")
            or value["sender"] == value["recipient"] or value["purpose"] not in ("Propose", "Counter", "Repair", "TacticalReview", "Continue", "FreshValue")):
        raise ValueError("root-local original peer role/purpose differs")
    state = fields(value["anchor"], ("owner", "revision", "semantic_sha256"), "root-local peer anchor")
    uint(state["owner"])
    uint(state["revision"])
    sha(state["semantic_sha256"])
    for key in ("anchor_root_ply", "line_offset"):
        uint(value[key], 65535)
    uint(value["evidence_revision"])
    if value["original_first_move"] is not None:
        _moves([value["original_first_move"]], 1)
    for key in ("prefix_sha256", "question_sha256", "root_budget_sha256"):
        sha(value[key])
    if value["line_sha256"] is not None:
        sha(value["line_sha256"])
    elif value["line_offset"]:
        raise ValueError("root-local peer line offset lacks its original line")
    _deadline(value["deadline"])
    _view(value["view"], authority)
    return {"id": {"authority": authority, "sequence": value["id"]},
        "parent": None if value["parent"] is None else {"authority": authority, "sequence": value["parent"]},
        **{name: value[name] for name in ("sender", "recipient", "purpose", "anchor_root_ply", "original_first_move", "line_offset", "evidence_revision")},
        "anchor": {"owner": state["owner"], "revision": state["revision"], "semantic": state["semantic_sha256"]},
        "prefix_digest": value["prefix_sha256"], "line_digest": value["line_sha256"], "question_digest": value["question_sha256"],
        "view": _legacy_view(value["view"]), "root_budget": value["root_budget_sha256"], "original_deadline": value["deadline"]}


def _origin(value, authority, root):
    fields(value, ("frame_sequence", "ticket", "output_sha256", "kind", "peer", "seal_sha256"), "root-local original accepted P/C origin")
    ticket = _ticket(value["ticket"])
    _authority({name: ticket[name] for name in _AUTHORITY}, authority)
    if (ticket["deadline_clock"] != root["deadline_clock"] or ticket["deadline_tick"] != root["deadline_tick"]
            or ticket["root_state"] != root["root_state"]):
        raise ValueError("root-local accepted P/C ticket changed root/deadline")
    # Explicit metadata projection of two persisted DTO spellings only. The
    # shared binary checker never constructs an AcceptedRoleOrigin capability.
    peer = None if value["peer"] is None else _peer(value["peer"], authority)
    if value["peer"] is not None and value["peer"]["root_budget_sha256"] != root["root_budget_sha256"]:
        raise ValueError("root-local accepted peer crossed original root budget")
    projected = {"frame": {**authority, "sequence": value["frame_sequence"]}, "ticket": ticket,
        "output_digest": value["output_sha256"], "kind": value["kind"], "peer": peer, "seal_sha256": value["seal_sha256"]}
    _accepted_origin(projected)
    return ticket


def _semantic(value, authority, root, action, original_input):
    fields(value, _SEMANTIC_FIELDS, "root-local original semantic consumption")
    baseline = _array(value["baseline_witnesses"], MAX_PUBLICATION_BYTES)
    for item in baseline:
        sha(item)
    if baseline != sorted(set(baseline)):
        raise ValueError("root-local baseline witness set must preserve sorted unique identities")
    identities = set()
    for witness in _array(value["new_witnesses"]):
        fields(witness, ("kind", "observation", "identity_sha256", "raw_evidence_sha256", "line"), "root-local original new witness")
        if witness["kind"] not in ("counter", "repair"):
            raise ValueError("root-local original witness kind differs")
        uint(witness["observation"])
        identity = sha(witness["identity_sha256"])
        sha(witness["raw_evidence_sha256"])
        _moves(witness["line"], 256)
        if identity in identities or identity in baseline:
            raise ValueError("root-local repeated/baseline evidence is not a new witness")
        identities.add(identity)
    pc = _array(value["pc_consumption"])
    tickets = {}
    for consumed in pc:
        fields(consumed, ("origin", "question_sha256", "view", "full_view_sha256"), "root-local original P/C consumption")
        ticket = _origin(consumed["origin"], authority, root)
        view = _view(consumed["view"], authority)
        if (sha(consumed["question_sha256"]) != ticket["question_digest"]
                or _legacy_view(view) != {"authority": authority, **{name: ticket[name] for name in
                    ("view_sequence", "context_policy", "view_digest", "root_state", "actual_tokens", "selected_records", "anchors", "length_bucket")}}):
            raise ValueError("root-local original P/C consumed question/view differs from its accepted ticket")
        sha(consumed["full_view_sha256"])
        key = ticket["request_sequence"]
        if key in tickets:
            raise ValueError("root-local duplicate original P/C consumption")
        tickets[key] = consumed
        if (ticket["representation"]["model"] != root["native_origin"]["weights_sha256"]
                or ticket["representation"]["encoding"] != root["native_origin"]["encoding_sha256"]
                or ticket["representation"]["frozen_epoch"] != root["native_origin"]["frozen_epoch"]):
            raise ValueError("root-local P/C consumption changed frozen loaded model")
        if (consumed["origin"]["frame_sequence"] == action["frame_sequence"] and consumed["question_sha256"] == action["question_sha256"]
                and action["recipient"] in ("route_p", "route_c")
                and ticket["representation"]["role"] != {"route_p": "proposer", "route_c": "critic"}[action["recipient"]]):
            raise ValueError("root-local original P/C route consumed a different private role")
    executions, previous = set(), None
    for item in _array(value["native_executions"]):
        fields(item, ("epoch", "sequence"), "root-local original native execution")
        key = (_positive(item["epoch"]), _positive(item["sequence"]))
        if key[0] != authority["process_epoch"] or key in executions or previous is not None and key <= previous:
            raise ValueError("root-local original native execution scope/order differs")
        executions.add(key)
        previous = key
    r_sources = set()
    for consumed in _array(value["r_selected_pc_consumption"]):
        key = _r_consumption(consumed, authority, root, tickets, executions, original_input)
        if key in r_sources:
            raise ValueError("root-local duplicate original selected R/P-C consumption")
        r_sources.add(key)
    if (action["recipient"] in ("route_p", "route_c") and not pc
            or action["recipient"] == "route_r" and not value["r_selected_pc_consumption"]):
        raise ValueError("root-local original route lacks its actual semantic consumption")
    seen = set()
    for dependency in _array(value["dependencies"]):
        fields(dependency, ("frame_sequence", "question_sha256", "view", "full_view_sha256", "parent_frame_sequence",
            "alias", "canonical", "source_origin", "role_cause", "cpu_cause"), "root-local original dependency")
        frame = _positive(dependency["frame_sequence"])
        key = (frame, sha(dependency["question_sha256"]))
        if key in seen:
            raise ValueError("root-local duplicate dependency frame/question")
        seen.add(key)
        _view(dependency["view"], authority)
        sha(dependency["full_view_sha256"])
        if dependency["parent_frame_sequence"] is not None and uint(dependency["parent_frame_sequence"]) >= frame:
            raise ValueError("root-local dependency parent chronology differs")
        if (dependency["alias"] is None) != (dependency["canonical"] is None):
            raise ValueError("root-local reciprocal dependency lacks alias/canonical pair")
        for name in ("alias", "canonical"):
            if dependency[name] is not None:
                _peer(dependency[name], authority)
                if (dependency[name]["root_budget_sha256"] != root["root_budget_sha256"]
                        or dependency[name]["deadline"] != action["original_deadline"]):
                    raise ValueError("root-local dependency crossed root budget/deadline")
        if dependency["source_origin"] is not None:
            if dependency["alias"] is None:
                raise ValueError("root-local dependency source origin lacks original peer")
            _origin(dependency["source_origin"], authority, root)
        if dependency["role_cause"] is not None and dependency["cpu_cause"] is not None:
            raise ValueError("root-local dependency cannot invent two continuation causes")
        cause = dependency["role_cause"]
        if cause is not None:
            fields(cause, ("request_epoch", "request_sequence", "frame_sequence", "question_sha256", "view_sha256"), "root-local role continuation cause")
            if _positive(cause["request_epoch"]) != authority["process_epoch"]:
                raise ValueError("root-local role continuation crossed original process")
            for name in ("request_sequence", "frame_sequence"):
                _positive(cause[name])
            sha(cause["question_sha256"])
            sha(cause["view_sha256"])
            source = tickets.get(cause["request_sequence"])
            if (source is None or source["origin"]["frame_sequence"] != cause["frame_sequence"]
                    or source["question_sha256"] != cause["question_sha256"] or source["view"]["view_digest"] != cause["view_sha256"]):
                raise ValueError("root-local role continuation lacks its original consumed P/C source")
        cause = dependency["cpu_cause"]
        if cause is not None:
            fields(cause, ("job_sequence", "native_execution_epoch", "native_execution_sequence", "frame_sequence", "question_sha256", "observation"),
                "root-local CPU continuation cause")
            for name in ("job_sequence", "native_execution_epoch", "native_execution_sequence", "frame_sequence"):
                _positive(cause[name])
            uint(cause["observation"])
            sha(cause["question_sha256"])
            if cause["native_execution_epoch"] != authority["process_epoch"]:
                raise ValueError("root-local CPU continuation crossed original process")
            if (cause["native_execution_epoch"], cause["native_execution_sequence"]) not in executions:
                raise ValueError("root-local CPU continuation lacks its original committed native execution")
    allowed = seen | {(action["frame_sequence"], action["question_sha256"])}
    if any((source["origin"]["frame_sequence"], source["question_sha256"]) not in allowed for source in pc):
        raise ValueError("root-local P/C consumption lacks its original action/dependency question")
    if any((source["frame_sequence"], source["question_sha256"]) not in allowed for source in value["r_selected_pc_consumption"]):
        raise ValueError("root-local R consumption lacks its original action/dependency question")


def _r_consumption(value, authority, root, tickets, executions, original_input):
    fields(value, ("stage", "authority", "frame_sequence", "question_sha256", "submitted_view", "submitted_full_view_sha256", "job", "owner",
        "task_key", "store_execution", "native_execution", "observation", "capture_sequence", "binding_sha256", "capture_full_view_sha256",
        "capture_view_seal_sha256", "attested_slice_work", "selected_pc_ticket", "selected_output_sha256", "selected_record", "selected_full_view_sha256"),
        "root-local original selected R/P-C consumption")
    if value["stage"] != "SelectedPcViewConsumed":
        raise ValueError("root-local R feedback was not consumed by its original selected P/C view")
    _authority(value["authority"], authority)
    _positive(value["frame_sequence"])
    sha(value["question_sha256"])
    _view(value["submitted_view"], authority)
    for name in ("submitted_full_view_sha256", "binding_sha256", "capture_full_view_sha256", "capture_view_seal_sha256", "selected_full_view_sha256"):
        sha(value[name])
    uint(value["owner"], 65535)
    for name in ("store_execution", "observation", "capture_sequence"):
        uint(value[name])
    native = fields(value["native_execution"], ("epoch", "sequence"), "root-local R native execution")
    if (uint(native["epoch"]), uint(native["sequence"])) not in executions:
        raise ValueError("root-local selected R/P-C source lacks its original native execution")
    from .rovent_r_corpus import _job
    job = _job(value["job"])
    if (job["authority"] != authority or job["question_sha256"] != value["question_sha256"]
            or job["hard_deadline_clock_epoch"] != root["deadline_clock"] or job["hard_deadline_ns"] > root["deadline_tick"]
            or job["evaluation_namespace_sha256"] != value["binding_sha256"]):
        raise ValueError("root-local R Job changed original authority/question/deadline")
    task = fields(value["task_key"], ("state", "line", "question", "root_moves", "model", "epoch", "value_identity", "checker_identity",
        "cpu_condition", "profile", "condition", "input_revision", "requested_depth", "node_budget"), "root-local R exact task key")
    for name in ("state", "model", "epoch", "profile", "condition", "input_revision", "node_budget"):
        uint(task[name])
    uint(task["requested_depth"], 65535)
    if task["line"] is not None:
        uint(task["line"])
    _moves(task["root_moves"], 218)
    if task["value_identity"] is not None:
        _identity(task["value_identity"])
        if task["value_identity"] != original_input["base_identity"]:
            raise ValueError("root-local R task changed original NNUEC model")
    if task["checker_identity"] is not None or task["value_identity"] is None:
        raise ValueError("root-local native R task cannot acquire an external checker identity")
    if task["cpu_condition"] is None:
        raise ValueError("root-local native R task lacks exact search conditions")
    _text(task["cpu_condition"], MAX_PUBLICATION_BYTES)
    question = task["question"]
    if question not in ("AnalyzePosition", "AnalyzeRootMoves"):
        if type(question) is not dict or set(question) != {"FindAlternative"}:
            raise ValueError("root-local R task question is not a native search task")
        fields(question["FindAlternative"], ("divergence_ply",), "root-local R alternative question")
        uint(question["FindAlternative"]["divergence_ply"], 65535)
    work = fields(value["attested_slice_work"], ("nodes", "quiescence_nodes", "tt_hits"), "root-local original R slice work")
    for count in work.values():
        uint(count)
    ticket = _ticket(value["selected_pc_ticket"])
    consumed = tickets.get(ticket["request_sequence"])
    selected = fields(value["selected_record"], ("revision", "origin_state"), "root-local original R selected record")
    uint(selected["revision"])
    uint(selected["origin_state"])
    p = original_input["provenance"]
    if (consumed is None or consumed["origin"]["ticket"] != ticket
            or sha(value["selected_output_sha256"]) != consumed["origin"]["output_sha256"]
            or value["selected_full_view_sha256"] != consumed["full_view_sha256"]
            or value["capture_full_view_sha256"] != p["full_view_sha256"] or value["capture_view_seal_sha256"] != p["view_seal_sha256"]):
        raise ValueError("root-local selected R/P-C source changed original capture/accepted output")
    return native["epoch"], native["sequence"], ticket["request_sequence"]


def _cost(value):
    fields(value, _COST_FIELDS, "root-local original whole-route cost")
    if value["scope"] != _COST_SCOPE or value["objective"] != _OBJECTIVE:
        raise ValueError("root-local original exclusive whole-route/drain cost scope differs")
    for name in ("resource_profile_sha256", "cost_epoch_sha256"):
        sha(value[name])
    uint(value["whole_wall_ns"])
    if any(value[name] is not None for name in ("cpu_thread_time_ns", "energy", "retained_host_bytes")):
        raise ValueError("root-local unobserved cost metrics must remain unknown")
    baseline = fields(value["baseline_counters"], _COUNTERS, "root-local original baseline counters")
    final = fields(value["final_counters"], _COUNTERS, "root-local original final counters")
    for name in _COUNTERS:
        uint(baseline[name])
        uint(final[name])
        if name in _UNKNOWN_COUNTERS:
            if baseline[name] != final[name]:
                raise ValueError("root-local whole-cost unknown counter state changed")
        else:
            if final[name] < baseline[name]:
                raise ValueError("root-local whole-cost counter moved backwards")
    if final["rejected_events"] != baseline["rejected_events"]:
        raise ValueError("root-local successful closure includes a rejected event")
    for name in ("native_credit_baseline", "native_credit_final"):
        fields(value[name], ("spent", "reserved"), "root-local original native credit")
        uint(value[name]["spent"])
        if uint(value[name]["reserved"]) != 0:
            raise ValueError("root-local physical closure retains outstanding native reservation")
    if value["native_credit_final"]["spent"] < value["native_credit_baseline"]["spent"]:
        raise ValueError("root-local original native spent credit moved backwards")


def _masks(value, supported):
    for name in ("support_mask", "target_known_mask"):
        if type(value[name]) is not list or len(value[name]) != 7 or any(type(bit) is not bool for bit in value[name]):
            raise ValueError("root-local exact seven-boolean masks required")
    if value["support_mask"] != supported or value["target_known_mask"] != [False] * 7:
        raise ValueError("root-local support must preserve actual routes and known targets remain false")


def _group(action, cost):
    raw = (b"rz-rovent-utility-route-comparison-group/1\0" + bytes.fromhex(action["frozen_context_sha256"])
        + bytes.fromhex(cost["resource_profile_sha256"]) + bytes.fromhex(cost["cost_epoch_sha256"])
        + cost["scope"].encode("ascii") + b"\0" + cost["objective"].encode("ascii"))
    return hashlib.sha256(raw).hexdigest()


def _action_digest(value):
    action, cost, semantic, inp = (value[name] for name in ("action", "cost", "semantic", "input"))
    authority = action["authority"]
    raw = b"rz-rovent-utility-actual-route-action/1\0"
    raw += b"".join(authority[name].to_bytes(8, "little") for name in _AUTHORITY[:3])
    raw += bytes.fromhex(authority["implementation_sha256"]) + action["frame_sequence"].to_bytes(8, "little")
    raw += bytes.fromhex(action["question_sha256"]) + bytes([action["output_index"]]) + action["manager_sequence"].to_bytes(8, "little")
    raw += bytes.fromhex(action["root_budget_sha256"]) + bytes.fromhex(action["comparison_group_sha256"])
    raw += inp["submission_target_sha256"].encode("ascii") + cost["whole_wall_ns"].to_bytes(16, "little")
    raw += len(semantic["new_witnesses"]).to_bytes(8, "little")
    for witness in semantic["new_witnesses"]:
        raw += bytes.fromhex(witness["identity_sha256"]) + bytes.fromhex(witness["raw_evidence_sha256"])
    raw += len(semantic["pc_consumption"]).to_bytes(8, "little")
    for source in semantic["pc_consumption"]:
        raw += bytes.fromhex(source["origin"]["seal_sha256"])
    raw += len(semantic["r_selected_pc_consumption"]).to_bytes(8, "little")
    for source in semantic["r_selected_pc_consumption"]:
        raw += source["native_execution"]["epoch"].to_bytes(8, "little") + source["native_execution"]["sequence"].to_bytes(8, "little")
        raw += source["selected_pc_ticket"]["request_sequence"].to_bytes(8, "little")
    return hashlib.sha256(raw).hexdigest()


def _report(value, raw, pin):
    return {"schema": value["schema"], "sha256": pin, "bytes": len(raw), "collection_sequence": value["collection_sequence"],
        "scope": "historical_root_local_metadata_only_not_comparison_cohort", "support_mask": list(value["support_mask"]),
        "target_known_mask": list(value["target_known_mask"]), "past_native_authority_reissued": False,
        "past_physical_completion_reissued": False, "execution_authorized": False, "training_target_authorized": False,
        "known_pair_eligible": False, "actual_training_executed": False, "rules_history_replayed": False,
        "source_artifact_bytes_verified": False, "physical_completion_verified": False}


@dataclass(frozen=True)
class RootLocalActionObservation:
    raw_bytes: bytes
    sha256: str
    _config: numeric.NnuecConfig

    @property
    def wire(self):
        return json.loads(self.raw_bytes.decode("utf-8"))

    def observation(self):
        value = self.wire
        return {**_report(value, self.raw_bytes, self.sha256), "recipient": value["action"]["recipient"],
            "reference_json_preserved": True, "input_group_seals_scope": "reader_checked_historical_metadata_only"}


@dataclass(frozen=True)
class RootLocalPairObservation:
    raw_bytes: bytes
    sha256: str

    @property
    def wire(self):
        return json.loads(self.raw_bytes.decode("utf-8"))

    def observation(self):
        value = self.wire
        return {**_report(value, self.raw_bytes, self.sha256), **{name: value[name] for name in
            ("left_action_sha256", "right_action_sha256", "original_comparison_sha256", "preferred", "unknown_reason")}}


def read_root_local_action(raw, expected_sha256, *, config):
    value = fields(_decode(raw, expected_sha256, ACTION_DOMAIN), _ACTION_FIELDS, "root-local original action")
    if value["schema"] != ACTION_DOMAIN:
        raise ValueError("explicit root-local action domain required; no cohort conversion")
    _positive(value["collection_sequence"])
    root, authority = _root(value["source"])
    captured = _input(value["input"], root, config)
    _registration(value["source"]["nnuec_registration"], value["input"]["base_identity"])
    action = fields(value["action"], _ACTION_BODY, "root-local original actual action")
    _authority(action["authority"], authority)
    for name in ("frame_sequence", "manager_sequence"):
        _positive(action[name])
    for name in ("question_sha256", "root_budget_sha256", "frozen_context_sha256", "comparison_group_sha256", "original_action_sha256"):
        sha(action[name])
    _moves([action["original_first_move"]], 1)
    if (type(action["recipient"]) is not str or action["recipient"] not in _ROUTES
            or type(action["output_index"]) is not int or action["output_index"] != _ROUTES[action["recipient"]]
            or action["mode"] != root["native_origin"]["mode"] or action["root_budget_sha256"] != root["root_budget_sha256"]
            or action["original_first_move"] != value["input"]["records_metadata"][value["input"]["target_record"]]["original_first_move"]
            or action["recipient"] == "route_r" and action["mode"] not in ("HybridPcr", "CpuPcr")):
        raise ValueError("root-local original action route/mode/target/budget mismatch")
    _deadline(action["original_deadline"], {"clock": root["deadline_clock"], "tick": root["deadline_tick"]})
    budget = fields(action["budget"], ("role_submissions_remaining", "original_cpu_credit", "original_store_byte_limit"), "root-local original frame budget")
    uint(budget["role_submissions_remaining"])
    uint(budget["original_cpu_credit"])
    uint(budget["original_store_byte_limit"])
    _masks(value, [index == action["output_index"] for index in range(7)])
    _semantic(value["semantic"], authority, root, action, value["input"])
    _cost(value["cost"])
    if (_group(action, value["cost"]) != action["comparison_group_sha256"]
            or _action_digest(value) != action["original_action_sha256"]):
        raise ValueError("root-local original comparison group/action binary seal mismatch")
    # The validated capture is deliberately not stored as a training sample.
    if captured.input_bytes != value["input"]["reference_json"].encode("utf-8"):
        raise ValueError("root-local original submitted JSON bytes changed")
    return RootLocalActionObservation(raw, expected_sha256, numeric.NnuecConfig.from_mapping(config.to_mapping()))


def _comparison(left, right):
    l, r = left["action"], right["action"]
    lw = {w["identity_sha256"] for w in left["semantic"]["new_witnesses"]}
    rw = {w["identity_sha256"] for w in right["semantic"]["new_witnesses"]}
    preferred, unknown = 255, 255
    if l["original_action_sha256"] == r["original_action_sha256"]:
        unknown = 1
    elif l["comparison_group_sha256"] != r["comparison_group_sha256"]:
        unknown = 0
    elif l["recipient"] == r["recipient"]:
        unknown = 2
    elif not lw and not rw:
        unknown = 3
    else:
        lc, rc = left["cost"]["whole_wall_ns"], right["cost"]["whole_wall_ns"]
        if lw == rw and lc == rc:
            unknown = 4
        elif rw <= lw and lc <= rc:
            preferred = l["output_index"]
        elif lw <= rw and rc <= lc:
            preferred = r["output_index"]
        else:
            unknown = 5
    raw = (b"rz-rovent-utility-actual-route-pair/1\0" + bytes.fromhex(l["original_action_sha256"])
        + bytes.fromhex(r["original_action_sha256"]) + bytes((preferred, unknown)))
    return hashlib.sha256(raw).hexdigest(), "missing_comparison_cohort" if unknown == 255 else _UNKNOWN_REASONS[unknown]


def read_root_local_pair(raw, expected_sha256, *, left, right):
    value = fields(_decode(raw, expected_sha256, PAIR_DOMAIN), _PAIR_FIELDS, "root-local original pair")
    if value["schema"] != PAIR_DOMAIN:
        raise ValueError("explicit root-local pair domain required; no cohort conversion")
    _positive(value["collection_sequence"])
    if type(left) is not RootLocalActionObservation or type(right) is not RootLocalActionObservation:
        raise ValueError("root-local pair requires two separately read historical actions")
    # These detached observations remain public metadata, not original caps.
    # Recheck the complete reader, not only the outer SHA: a public Python
    # constructor is not a substitute for strict historical input/source joins.
    lw = read_root_local_action(left.raw_bytes, left.sha256, config=left._config).wire
    rw = read_root_local_action(right.raw_bytes, right.sha256, config=right._config).wire
    if sha(value["left_action_sha256"]) != left.sha256 or sha(value["right_action_sha256"]) != right.sha256:
        raise ValueError("root-local pair changed original left/right raw publication pins")
    support = [a or b for a, b in zip(lw["support_mask"], rw["support_mask"], strict=True)]
    _masks(value, support)
    comparison, unknown = _comparison(lw, rw)
    if (sha(value["original_comparison_sha256"]) != comparison or value["preferred"] is not None
            or value["unknown_reason"] != unknown):
        raise ValueError("root-local original comparison must remain unknown without its comparison cohort")
    return RootLocalPairObservation(raw, expected_sha256)
