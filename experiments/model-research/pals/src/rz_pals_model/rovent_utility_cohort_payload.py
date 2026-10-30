"""Strict historical utility cohort bodies, without original trial authority.

The file owner supplies retained immutable raw bytes. Native seals, raw file
pins and canonical metadata hashes deliberately keep separate domains here.
This consumer never reconstructs a ready offer, physical fence, independent
trial, Rules owner or training permission from public JSON. The Rust parent
must separately compare the original source projection and replay Rules.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import struct

from .generation_data import digest, fields, identifier, sha, uint, _moves
from .rovent_utility_cohort_files import (
    COST_SCOPE, HistoricalUtilityCohortFiles, MAX_MATERIALIZATION_BYTES,
)
from .rovent_utility_training import (
    COHORT_PREPARATION_DOMAIN, COST_OBJECTIVE, validate_training_preparation,
)
from . import rovent_utility_root_local as structure
from . import rovent_corpus as corpus
from . import rovent_historical_r_corpus as r_source
from . import nnuec_reference as numeric
from . import nnuec_v2_reference as relation
from .rovent_corpus_codec import SUBMITTED_CODEC
from .rovent_historical_corpus import _input_lock, _source_aggregate
from .rovent_native_v2 import _artifact, _commit, validate_position_replay
from .rovent_utility_input import validate_submitted_utility_input


PROFILE_DOMAIN = "rz-rovent-utility-comparison-cohort/1"
TRIAL_DOMAIN = "rz-rovent-utility-cohort-trial/1"
MATERIALIZATION_DOMAIN = "rz-rovent-utility-cohort-materialization/1"
PAIR_DOMAIN = "rz-rovent-utility-cohort-pair/1"
BASIS_DOMAIN = "rz-rovent-utility-cohort-conditional-basis/1"
COST_DOMAIN = "rz-rovent-utility-cohort-cost/1"
MAX_PROFILE_BYTES = 64 << 10
MAX_WINDOW_NS = 86_400_000_000_000
_PROFILE_FIELDS = (
    "input_lock_sha256", "source_sha256", "frozen_nnuec_model_sha256",
    "frozen_nnuec_namespace_sha256", "frozen_pc_model_sha256",
    "frozen_pc_namespace_sha256", "mode", "requested_recipients",
    "prelude_ns", "action_ns", "cleanup_ns", "logical_budget_sha256",
    "ready_policy_sha256", "cache_policy_sha256", "cost_class_sha256",
    "resource_plan_sha256", "resource_epoch", "usage", "cost_scope",
)
_SOURCE_FIELDS = (
    "input_lock_sha256", "source_sha256", "frozen_nnuec_model_sha256",
    "frozen_nnuec_namespace_sha256", "frozen_pc_model_sha256",
    "frozen_pc_namespace_sha256", "resource_plan_sha256", "logical_budget_sha256",
)
_FROZEN_FIELDS = (
    "input_sha256", "target_sha256", "context_sha256", "target_record_ids",
    "input_bytes", "root_rules_sha256", "original_first_move", "first_trial_sequence",
)
_TRIAL_FIELDS = (
    "schema", "cohort_id", "trial_sequence", "recipient", "profile",
    "profile_sha256", "source", "frozen_context", "materialized_sha256",
    "materialized_json", "historical_unbound_input", "unresolved_source",
    "cost", "original_root", "ready_prelude", "physical_closure", "previous_durable_ack",
    "resolved", "comparable", "unknown_reason", "support_mask", "target_known_mask",
)
_IO_FIELDS = (
    "clock_epoch", "started_tick_ns", "finished_tick_ns", "elapsed_ns",
    "written_bytes", "readback_bytes", "synced_files", "synced_directories",
)
_COST_FIELDS = (
    "domain", "scope", "objective", "cost_class_sha256", "cost_epoch_sha256",
    "resource_plan_sha256", "resource_epoch", "clock_epoch", "started_tick", "root_fence_tick",
    "actor_release_tick", "physical_pin_zero_tick", "whole_wall_ns",
    "original_deadline_tick", "cohort_action_cutoff_tick", "cohort_cleanup_cutoff_tick",
    "baseline_counters", "final_counters", "native_credit_baseline",
    "native_credit_final", "cpu_thread_time_ns", "energy", "complete_scope",
)
_COUNTER_FIELDS = (
    "native_preparations", "submitted_role_tickets", "physical_role_completions",
    "completed_nn_inputs", "nn_input_count_unknown", "consumed_role_outputs",
    "accepted_nn_inputs", "accepted_nn_input_count_unknown", "consumed_nn_inputs",
    "consumed_nn_input_count_unknown", "zero_nn_outputs_consumed", "unused_role_results",
    "canceled_preparations", "cpu_slice_receipts", "cpu_progress_publications",
    "cpu_yields", "cpu_yield_requests", "cpu_yield_acknowledgements",
    "cpu_natural_completions", "validator_ticket_admissions", "foreign_reports",
    "foreign_reported_nodes", "foreign_node_count_unknown", "foreign_yield_requests",
    "foreign_yield_acknowledgements", "rejected_events", "notification_waits",
)
_RECIPIENTS = ("proposer", "critic", "researcher")
_ROUTES = ("route_p", "route_c", "route_r")
_ROUTE_FOR_RECIPIENT = dict(zip(_RECIPIENTS, _ROUTES, strict=True))
_RECIPIENT_FOR_ROUTE = dict(zip(_ROUTES, _RECIPIENTS, strict=True))


class UnsupportedCohortPayload(ValueError):
    """An unregistered original proof body stays unknown, never a target."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def _boolean(value, label):
    if type(value) is not bool:
        raise ValueError(f"utility cohort exact boolean required: {label}")
    return value


def _text(value, maximum=4096):
    if (type(value) is not str or not value or len(value.encode("utf-8")) > maximum
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)):
        raise ValueError("utility cohort bounded original metadata text required")
    return value


def _positive(value, maximum=2**64 - 1):
    if not uint(value, maximum):
        raise ValueError("utility cohort positive bounded integer required")
    return value


def _array(value, maximum):
    if type(value) is not list or len(value) > maximum:
        raise ValueError("utility cohort bounded original array required")
    return value


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("utility cohort original JSON contains duplicate keys")
        value[key] = item
    return value


def _nonfinite(_value):
    raise ValueError("utility cohort nonfinite JSON constant rejected")


def _decode(raw, maximum, *, source_operands=False, allow_null=False):
    if type(raw) is not bytes or not 0 < len(raw) <= maximum:
        raise ValueError("utility cohort bounded immutable raw bytes required")
    try:
        value = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=_object,
            parse_constant=_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("utility cohort original UTF8 JSON required") from error
    remaining = [maximum]

    def walk(item, depth=0):
        remaining[0] -= 1
        if depth > 64 or remaining[0] < 0:
            raise ValueError("utility cohort original JSON structure bound")
        if type(item) is dict:
            for nested in item.values():
                walk(nested, depth + 1)
        elif type(item) is list:
            for nested in item:
                walk(nested, depth + 1)
        elif type(item) is float and source_operands and math.isfinite(item):
            return
        elif type(item) not in (str, int, bool, type(None)):
            # Floating input spellings belong in reference_json, not a second
            # serialization of the outer source/clock/retirement metadata.
            raise ValueError("utility cohort outer metadata requires integer/string wire")
    walk(value)
    if type(value) is not dict and not (allow_null and value is None):
        raise ValueError("utility cohort original JSON object required")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")


def _native_digest(domain, raw):
    return hashlib.sha256(domain.encode("ascii") + b"\0"
        + len(raw).to_bytes(8, "little") + raw).hexdigest()


def _value_digest(domain, value):
    return _native_digest(domain, _canonical(value))


def validate_cohort_profile(value):
    """Check the exact declared profile; this does not observe ready trials."""
    fields(value, ("domain", "input"), "utility cohort profile")
    if value["domain"] != PROFILE_DOMAIN:
        raise ValueError("utility cohort explicit profile domain required")
    body = fields(value["input"], _PROFILE_FIELDS, "utility cohort profile input")
    for name in _PROFILE_FIELDS:
        if name.endswith("_sha256"):
            sha(body[name])
    if body["mode"] not in ("cpu_pc", "cpu_pcr"):
        raise ValueError("utility cohort CPU-only mode required")
    recipients = _array(body["requested_recipients"], 3)
    if (not recipients or any(type(item) is not str or item not in _RECIPIENTS for item in recipients)
            or len(set(recipients)) != len(recipients)
            or body["mode"] == "cpu_pc" and "researcher" in recipients):
        raise ValueError("utility cohort unique ordered supported recipients required")
    total = sum(_positive(body[name]) for name in ("prelude_ns", "action_ns", "cleanup_ns"))
    if total > MAX_WINDOW_NS:
        raise ValueError("utility cohort declared window exceeds 24 hours")
    _positive(body["resource_epoch"])
    if body["usage"] not in ("validation_only", "collection_candidate") or body["cost_scope"] != COST_SCOPE:
        raise ValueError("utility cohort explicit usage/cost scope required")
    if len(_canonical(value)) > MAX_PROFILE_BYTES:
        raise ValueError("utility cohort canonical profile exceeds 64KiB")
    return copy.deepcopy(value)


def cohort_profile_sha256(profile):
    """Return the declared profile's domain seal, distinct from raw SHA."""
    profile = validate_cohort_profile(profile)
    # This profile domain deliberately has NO length prefix, unlike native
    # trial/materialization/pair seals. A raw file SHA is different again.
    return hashlib.sha256(PROFILE_DOMAIN.encode("ascii") + b"\0" + _canonical(profile)).hexdigest()


def _source(value, profile):
    fields(value, _SOURCE_FIELDS, "utility cohort original source pins")
    for name in _SOURCE_FIELDS:
        if sha(value[name]) != profile["input"][name]:
            raise ValueError("utility cohort original source/profile pin differs")


def _source_parameter_sha(weights):
    """Hash validated FP32 operands without constructing/evaluating a model."""
    config, training = r_source._weights_metadata(weights)
    base = weights["base"]
    state = hashlib.sha256((numeric.SCHEMA + "\0" + numeric.SEMANTICS + "\0").encode("ascii"))
    for name in numeric.CONFIG_FIELDS:
        state.update(struct.pack("<I", getattr(config, name)))
    for name in numeric.PARAMETER_FIELDS:
        for value in base["parameters"][name]:
            state.update(struct.pack("<f", numeric.f32(value)))
    state.update(struct.pack("<f", numeric.f32(base["output_scale"])))
    base_sha = state.hexdigest()
    state = hashlib.sha256((relation.SCHEMA + "\0" + relation.SEMANTICS + "\0" + base_sha + "\0").encode("ascii"))
    for name in relation.PARAMETER_FIELDS:
        for value in weights["parameters"][name]:
            state.update(struct.pack("<f", numeric.f32(value)))
    return state.hexdigest(), config, training


def _ancestor_descriptors(value):
    ancestors = _array(value, 256)
    seen = set()
    for item in ancestors:
        fields(item, ("kind", "publication_sha256", "metadata"), "utility cohort original ancestor descriptor")
        kind = item["kind"]
        if kind in ("pc_legacy", "pc_submitted_input_v1", "r_auxiliary"):
            keys = ("manifest_file_sha256", "index_file_sha256", "canonical_manifest_sha256", "canonical_index_sha256")
        elif kind == "utility_cohort":
            keys = ("dataset_sha256", "pool_sha256", "preparation_sha256", "context_sha256")
        else:
            raise ValueError("utility cohort original ancestor family required")
        publication_sha = sha(item["publication_sha256"])
        fields(item["metadata"], keys, "utility cohort original ancestor metadata")
        for pin in item["metadata"].values():
            sha(pin)
        if publication_sha in seen:
            raise ValueError("utility cohort duplicate original ancestor publication")
        seen.add(publication_sha)
    return copy.deepcopy(ancestors)


def _resource_scope(value):
    fields(value, ("schema", "plan_pin", "evidence_directory", "profile", "admission_attempted", "finished_attempted",
        "original_plan_descriptor_retained", "original_delegated_descriptor_retained", "original_scope_retained",
        "resource_failure", "admission", "finished_live", "finished_live_error", "controlled_comparison_accepted",
        "automatic_cleanup"), "utility cohort original resource custody")
    if value["schema"] != "rz-rovent-resource-custody/1" or value["profile"] not in ("functional_validation", "strict_comparison"):
        raise ValueError("utility cohort original resource custody schema/profile differs")
    pin = fields(value["plan_pin"], ("path", "sha256", "bytes"), "utility cohort original resource plan pin")
    from .rovent_utility_cohort_files import _relative
    if not _relative(pin["path"]).startswith("inputs/"):
        raise ValueError("utility cohort original resource plan path differs")
    sha(pin["sha256"])
    _positive(pin["bytes"], 64 << 10)
    _relative(value["evidence_directory"])
    for name in ("admission_attempted", "finished_attempted", "original_plan_descriptor_retained",
            "original_delegated_descriptor_retained", "original_scope_retained", "controlled_comparison_accepted", "automatic_cleanup"):
        _boolean(value[name], name)
    if value["automatic_cleanup"] is not False or value["controlled_comparison_accepted"] is not False:
        raise ValueError("utility cohort resource projection cannot claim comparison or cleanup authority")
    if (not value["admission_attempted"] or not value["original_plan_descriptor_retained"]
            or not value["original_delegated_descriptor_retained"] or not value["original_scope_retained"]
            or value["admission"] is None):
        raise ValueError("utility cohort original admitted resource custody required")
    if not value["finished_attempted"] and (value["finished_live"] is not None or value["finished_live_error"] is not None):
        raise ValueError("utility cohort resource finish observation lacks its original attempt")

    def observed(item, kind):
        if type(item) is not dict or item.get("status") not in ("unknown", "observed"):
            raise ValueError("utility cohort explicit actual resource observation required")
        if item["status"] == "unknown":
            fields(item, ("status",), "utility cohort unknown resource observation")
            return
        fields(item, ("status", "value", "method"), "utility cohort actual resource readback")
        _text(item["method"])
        if kind == "bool":
            _boolean(item["value"], "actual resource boolean")
        elif kind == "sha":
            sha(item["value"])
        elif kind == "u32":
            _positive(item["value"], 2**32 - 1)
        elif kind == "positive":
            _positive(item["value"])
        else:
            uint(item["value"])

    for name in ("admission", "finished_live"):
        original = value[name]
        if original is None:
            continue
        fields(original, ("limits_applied", "process_tree_quota_millicpus", "memory_high_bytes", "memory_max_bytes",
            "inventory_complete", "kernel_threads", "verified_plan_readback_sha256", "host_peak_bytes", "vram_peak_bytes", "evidence"),
            "utility cohort actual aggregate resource readback")
        for key in ("limits_applied", "inventory_complete"):
            observed(original[key], "bool")
        observed(original["process_tree_quota_millicpus"], "u32")
        observed(original["verified_plan_readback_sha256"], "sha")
        for key in ("memory_high_bytes", "memory_max_bytes", "kernel_threads", "host_peak_bytes"):
            observed(original[key], "positive")
        observed(original["vram_peak_bytes"], "u64")
        _artifact(original["evidence"])
    if value["resource_failure"] is not None:
        raise UnsupportedCohortPayload("cohort_resource_failure_cannot_issue_target")
    if value["finished_live_error"] is not None:
        _text(value["finished_live_error"])
    return copy.deepcopy(value)


def _source_enrollment(files, profile, preparation):
    source = _decode(files.raw_file("source.json"), 1 << 20)
    fields(source, ("schema", "source_pins", "nnuec_registration", "nnuec_weights_artifact", "nnuec_provenance_artifact",
        "native_origin", "collection_manifest", "collection_metadata", "rules_root", "frozen_pc_sources",
        "ancestor_datasets", "resource_observation", "registration_root", "transposition_sha256"),
        "utility cohort original collection source")
    if source["schema"] != "rz-rovent-utility-cohort-collection-source/1":
        raise ValueError("utility cohort explicit collection source enrollment required")
    _source(source["source_pins"], profile)
    registered = preparation["registration"]
    pins = source["source_pins"]
    if (pins["source_sha256"] != registered["collection_source_sha256"]
            or pins["resource_plan_sha256"] != registered["resource_profile_sha256"]
            or pins["frozen_nnuec_model_sha256"] != registered["frozen_base"]["weights_sha256"]):
        raise ValueError("utility cohort source/preparation source/resource/frozen model differs")
    lock = _input_lock(_decode(files.raw_file("input-lock.json"), 4 << 20), pins["input_lock_sha256"])
    engine = fields(lock["manifest"]["engine"], ("source_commit", "source_components", "binary", "mode", "pc", "researcher",
        "teachers", "nnuec", "search"), "utility cohort original lock engine")
    _commit(engine["source_commit"])
    provenance = _array(lock["manifest"]["provenance"], 32)
    if not provenance:
        raise ValueError("utility cohort original lock provenance required")
    for artifact in provenance:
        _artifact(artifact)
    if (lock["manifest"]["purpose"] != "collection" or engine["mode"] != profile["input"]["mode"]
            or _source_aggregate(engine["source_commit"], engine["source_components"]) != pins["source_sha256"]
            or digest("rz-rovent-resource-plan/1", lock["manifest"]["resources"]) != pins["resource_plan_sha256"]):
        raise ValueError("utility cohort original lock collection/source/mode/resource plan differs")
    declared_nnuec = r_source._nnuec_pin(engine["nnuec"])
    for name, declared in (("nnuec_weights_artifact", "weights"), ("nnuec_provenance_artifact", "provenance")):
        _artifact(source[name])
        filename = "nnuec-weights.json" if declared == "weights" else "nnuec-provenance.json"
        original = files.file_pin(filename)
        if (source[name] != declared_nnuec[declared] or source[name]["sha256"] != original["sha256"]
                or source[name]["bytes"] != original["bytes"]):
            raise ValueError("utility cohort NNUEC raw artifact/lock/publication differs")
    weights = _decode(files.raw_file("nnuec-weights.json"), 32 << 20, source_operands=True)
    parameter_sha, config, training = _source_parameter_sha(weights)
    if parameter_sha != pins["frozen_nnuec_model_sha256"] or config.to_mapping() != registered["frozen_base"]["config"]:
        raise ValueError("utility cohort raw weights differ from original frozen semantic parameters")
    r_source._nnuec_registration(source["nnuec_registration"], declared_nnuec, parameter_sha, training)
    structure._registration(source["nnuec_registration"], source["nnuec_registration"]["model_identity"])
    if _value_digest("rz-rovent-utility-cohort-nnuec-namespace/1", source["nnuec_registration"]) != pins["frozen_nnuec_namespace_sha256"]:
        raise ValueError("utility cohort original frozen NNUEC namespace differs")
    r_source._provenance(_decode(files.raw_file("nnuec-provenance.json"), 1 << 20), declared_nnuec,
        parameter_sha, training, lock)
    manifest = corpus._manifest(source["collection_manifest"], corpus.CorpusLimits(), codec=SUBMITTED_CODEC)
    if manifest["source_sha256"] != pins["source_sha256"] or manifest["source_commit"] != engine["source_commit"]:
        raise ValueError("utility cohort original corpus/source enrollment differs")
    metadata = fields(source["collection_metadata"], ("game_id", "opening_id", "line_genealogy_id", "lineage"),
        "utility cohort original collection metadata")
    for name in ("game_id", "opening_id", "line_genealogy_id"):
        identifier(metadata[name])
    lineage = fields(metadata["lineage"], ("actor_id", "teacher_actor_id", "ancestor_origin_sha256"),
        "utility cohort original collection lineage")
    identifier(lineage["actor_id"])
    if lineage["teacher_actor_id"] is not None:
        identifier(lineage["teacher_actor_id"])
    corpus._sha_list(lineage["ancestor_origin_sha256"], 256, "cohort original ancestor origins")
    if metadata["opening_id"] not in manifest["opening_splits"]:
        raise ValueError("utility cohort opening lacks its original registered split")
    validate_position_replay(source["rules_root"])
    sha(source["transposition_sha256"])
    _ancestor_descriptors(source["ancestor_datasets"])
    scopes = _array(source["resource_observation"], 3)
    if len(scopes) != files.publication()["trial_count"]:
        raise ValueError("utility cohort original per-trial resource custody coverage differs")
    for scope in scopes:
        _resource_scope(scope)
    return source, lock, config


def _source_pc_models(source, lock):
    origin = source["native_origin"]
    if type(origin) is not dict:
        raise ValueError("utility cohort original native origin object required")
    pins, manifest = source["source_pins"], source["collection_manifest"]
    for name, expected in (("source_sha256", pins["source_sha256"]),
            ("input_lock_sha256", pins["input_lock_sha256"]),
            ("weights_sha256", pins["frozen_pc_model_sha256"]),
            ("model_semantics_sha256", manifest["model_semantics_sha256"]),
            ("runtime_library_sha256", manifest["runtime_sha256"]),
            ("source_commit", manifest["source_commit"]),
            ("source_components", lock["manifest"]["engine"]["source_components"])):
        if origin.get(name) != expected:
            raise ValueError("utility cohort original P/C native namespace/source differs")
    namespace_names = ("model_semantics", "model_semantics_sha256", "encoding", "encoding_sha256",
        "encoding_registry_sha256", "model_implementation_sha256", "representation_model_domain", "frozen_epoch",
        "weights_sha256", "export_manifest_sha256", "backend_sha256")
    if not set(namespace_names) <= set(origin):
        raise ValueError("utility cohort original P/C namespace preimage is incomplete")
    namespace = {name: origin[name] for name in namespace_names}
    if _value_digest("rz-rovent-utility-cohort-pc-namespace/1", namespace) != pins["frozen_pc_namespace_sha256"]:
        raise ValueError("utility cohort frozen P/C namespace seal differs")
    pc = fields(lock["manifest"]["engine"]["pc"], ("proposer", "critic", "validator", "backend", "device",
        "backend_configuration_sha256", "runtime_artifacts", "fp32", "tf32", "batch_width", "ort_intra_threads",
        "ort_inter_threads", "ort_spinning"), "utility cohort original lock P/C configuration")
    if (pc["device"] != {"device": "cpu"} or pc["backend"] not in ("rust_cpu", "ort_cpu")
            or pc["fp32"] is not True or pc["tf32"] is not False or uint(pc["batch_width"]) != 1):
        raise ValueError("utility cohort original CPU FP32/B1 P/C configuration differs")
    sha(pc["backend_configuration_sha256"])
    runtime_artifacts = _array(pc["runtime_artifacts"], 32)
    for artifact in runtime_artifacts:
        _artifact(artifact)
    if pc["backend"] == "rust_cpu":
        if any(pc[name] is not None for name in ("ort_intra_threads", "ort_inter_threads", "ort_spinning")):
            raise ValueError("utility cohort Rust CPU cannot acquire ORT settings")
    else:
        _positive(pc["ort_intra_threads"], 256)
        _positive(pc["ort_inter_threads"], 256)
        _boolean(pc["ort_spinning"], "original ORT spinning setting")
        if not runtime_artifacts:
            raise ValueError("utility cohort ORT CPU requires original runtime artifacts")
    if pc["validator"] is not None:
        validator = fields(pc["validator"], ("semantics", "encoding", "policy_head", "value_head", "implementation_sha256",
            "weights", "provenance", "diagnostic"), "utility cohort original optional V pin")
        for name in ("semantics", "encoding", "policy_head", "value_head"):
            _text(validator[name])
        sha(validator["implementation_sha256"])
        _boolean(validator["diagnostic"], "original optional V diagnostic declaration")
        for artifact in (validator["weights"], validator["provenance"]):
            _artifact(artifact)
    frozen = _array(source["frozen_pc_sources"], 2)
    if len(frozen) != 2:
        raise ValueError("utility cohort both frozen P/C source declarations required")
    for role, item in zip(("proposer", "critic"), frozen, strict=True):
        fields(item, ("role", "artifact", "model_semantics_sha256", "model_identity_sha256", "encoding_sha256"),
            "utility cohort original frozen P/C source")
        declared = fields(pc[role], ("semantics", "encoding", "policy_head", "value_head", "implementation_sha256",
            "weights", "provenance", "diagnostic"), "utility cohort original frozen P/C model pin")
        for name in ("semantics", "encoding", "policy_head", "value_head"):
            _text(declared[name])
        sha(declared["implementation_sha256"])
        if _boolean(declared["diagnostic"], "original frozen P/C diagnostic declaration") is not False:
            raise ValueError("utility cohort actual frozen P/C source cannot be diagnostic")
        for artifact in (item["artifact"], declared["weights"], declared["provenance"]):
            _artifact(artifact)
        if (item["role"] != role or item["artifact"] != declared["weights"]
                or sha(item["model_semantics_sha256"]) != origin["model_semantics_sha256"]
                or sha(item["model_identity_sha256"]) != origin["weights_sha256"]
                or sha(item["encoding_sha256"]) != origin["encoding_sha256"]
                or declared["weights"]["sha256"] != origin["weights_sha256"]
                or declared["semantics"] != origin["model_semantics"] or declared["encoding"] != origin["encoding"]
                or declared["implementation_sha256"] != origin["model_implementation_sha256"]):
            raise ValueError("utility cohort original frozen P/C artifact/model/encoding differs")
    return copy.deepcopy(frozen)


def _root_structure(root, source):
    if type(root) is not dict or type(root.get("schema")) is not str:
        raise ValueError("utility cohort original descriptor-bearing root required")
    # This calculated wrapper hash is used solely by the shared closed-shape
    # validator. It is not returned as an original cap or independent pin.
    projection = {"root_admission_sha256": digest(root["schema"], root), "root_admission": root,
        "native_origin_sha256": root.get("native_origin_sha256"),
        "input_lock_sha256": source["source_pins"]["input_lock_sha256"],
        "nnuec_registration": source["nnuec_registration"],
        "weights_artifact": source["nnuec_weights_artifact"],
        "provenance_artifact": source["nnuec_provenance_artifact"]}
    root, authority = structure._root(projection)
    original = source["native_origin"]
    if (set(root["native_origin"]) != set(original)
            or any(root["native_origin"][name] != original[name] for name in original if name != "process_epoch")):
        raise ValueError("utility cohort independent trial changed original frozen source namespace")
    _positive(original["process_epoch"])
    return root, authority


def _materialization_input(body, trial, source, config):
    """Reuse exact wire validators, never the root-local action decoder."""
    root, authority = _root_structure(body["original_root"], source)
    original_input = body["input"]
    if type(original_input) is not dict:
        raise ValueError("utility cohort original materialized input object required")
    projected_input = {**original_input, "root_rules": _rules_projection(original_input.get("root_rules")),
        "source_rules": [_rules_projection(item) for item in _array(original_input.get("source_rules"), 64)]}
    captured = structure._input(projected_input, root, config)
    _input_submission_seal(original_input)
    if captured.base_weights_sha256 != source["source_pins"]["frozen_nnuec_model_sha256"]:
        raise ValueError("utility cohort input changed original frozen NNUEC parameters")
    frozen = trial["frozen_context"]
    if frozen is None:
        raise ValueError("utility cohort materialized input lacks its original frozen decision")
    inp = body["input"]
    expected = {"input_sha256": captured.input_sha256, "target_sha256": captured.target_sha256,
        "context_sha256": captured.context_sha256, "target_record_ids": list(captured.target_ids),
        "input_bytes": len(captured.input_bytes), "root_rules_sha256": inp["root_rules"]["semantic_sha256"],
        "original_first_move": inp["records_metadata"][inp["target_record"]]["original_first_move"]}
    if any(frozen[name] != item for name, item in expected.items()):
        raise ValueError("utility cohort original frozen input/target/Rules metadata differs")
    if frozen["root_rules_sha256"] != source["rules_root"]["anchor"]["rules_state_sha256"]:
        raise ValueError("utility cohort exact original Rules root differs from source replay")
    return captured, root, authority


def _input_submission_seal(value):
    """Match the producer's immutable view framing, not numerical NN input."""
    provenance = value["provenance"]
    seal = hashlib.sha256(b"rz-pals-review-view-seal/1\0")
    seal.update(bytes.fromhex(sha(provenance["full_view_sha256"])))
    for name in ("sequence", "generation", "root_state"):
        seal.update(uint(provenance[name]).to_bytes(8, "little"))
    for record in _array(value["records_metadata"], 64):
        seal.update(uint(record["source_revision"]).to_bytes(8, "little"))
        seal.update(uint(record["source_state"]).to_bytes(8, "little"))
        seal.update(uint(record["source_line_offset"], 65535).to_bytes(2, "little"))
    if seal.hexdigest() != sha(provenance["view_seal_sha256"]):
        raise ValueError("utility cohort original submitted view seal/source revision framing differs")


def _rules_projection(value):
    """Validate new replay operands, then reuse only the old common shape."""
    common = ("rules_version", "variant", "semantic_sha256", "fen", "history_origin", "history_completeness", "known_history")
    fields(value, (*common, "observer_revision", "uci_replay"), "utility cohort full original Rules snapshot")
    uint(value["observer_revision"])
    replay = fields(value["uci_replay"], ("origin", "completeness", "start_fen", "moves"),
        "utility cohort original bounded Rules replay")
    if type(value["history_origin"]) is not str or type(value["history_completeness"]) is not str:
        raise ValueError("utility cohort exact Rules history enums required")
    if (replay["origin"] != {"StartPosition": "start_position", "Fen": "fen"}.get(value["history_origin"])
            or replay["completeness"] != {"Complete": "complete", "UnknownPrefix": "unknown_prefix"}.get(value["history_completeness"])):
        raise ValueError("utility cohort original Rules replay origin/completeness differs")
    _text(replay["start_fen"], 512)
    _moves(replay["moves"], 16384)
    projected = {name: value[name] for name in common}
    structure._rules(projected)
    # The complete new nine-field body remains in the pinned original JSON.
    # Legal replay and observer revision reconstruction stay with Rust.
    return projected


def _materialization_scope(body, trial, source, config, profile, *, cohort_id):
    captured, root, authority = _materialization_input(body, trial, source, config)
    resource_known = _prelude(body["prelude"], profile=profile, root=root, authority=authority)
    action = fields(body["action"], ("authority", "frame_sequence", "question_sha256", "root_budget_sha256",
        "original_first_move", "deadline_clock", "deadline_tick", "frozen_context_sha256", "cost_epoch_sha256", "cost_scope"),
        "utility cohort original materialized action")
    structure._authority(action["authority"], authority)
    _positive(action["frame_sequence"])
    for name in ("question_sha256", "root_budget_sha256", "frozen_context_sha256", "cost_epoch_sha256"):
        sha(action[name])
    _moves([action["original_first_move"]], 1)
    if (action["root_budget_sha256"] != root["root_budget_sha256"]
            or uint(action["deadline_clock"]) != root["deadline_clock"]
            or uint(action["deadline_tick"]) != root["deadline_tick"]
            or action["original_first_move"] != trial["frozen_context"]["original_first_move"]
            or action["cost_scope"] != COST_SCOPE):
        raise ValueError("utility cohort original action root/budget/deadline/input differs")
    cost = trial["cost"]
    prelude = body["prelude"]
    expected_epoch = hashlib.sha256(b"rz-rovent-utility-cohort-cost-epoch/1\0"
        + bytes.fromhex(cohort_id) + bytes.fromhex(cohort_profile_sha256(profile))
        + bytes.fromhex(prelude["ready_group_sha256"])
        + captured.input_sha256.encode("ascii") + captured.target_sha256.encode("ascii")
        + COST_SCOPE.encode("ascii")).hexdigest()
    if (action["cost_epoch_sha256"] != expected_epoch or cost["cost_epoch_sha256"] != expected_epoch
            or cost["clock_epoch"] != root["deadline_clock"]
            or cost["original_deadline_tick"] != root["deadline_tick"]
            or cost["started_tick"] < prelude["captured_tick"]
            or cost["baseline_counters"] != prelude["counters"]
            or cost["native_credit_baseline"] != [prelude["native_credit"]["spent"], prelude["native_credit"]["reserved"]]):
        raise ValueError("utility cohort original cost epoch/prelude/deadline differs")
    declared = root["root_descriptor"]["input"]
    if (_value_digest("rz-rovent-utility-cohort-logical-budget/1", declared["declared_ceilings"])
            != source["source_pins"]["logical_budget_sha256"]
            or digest("rz-rovent-resource-plan/1", declared["declared_ceilings"]["resource_plan"])
                != source["source_pins"]["resource_plan_sha256"]
            or cost["cohort_action_cutoff_tick"] > root["deadline_tick"]
            or cost["cohort_cleanup_cutoff_tick"] > declared["clock"]["cleanup"]["at_ns"]):
        raise ValueError("utility cohort action window exceeds original root budget/deadlines")
    selected = [offer for offer in prelude["offers"] if offer["recipient"] == trial["recipient"]]
    if len(selected) != 1 or any(selected[0][key] != action[key] for key in (
            "frame_sequence", "question_sha256", "original_first_move", "deadline_clock", "deadline_tick")):
        raise ValueError("utility cohort action differs from its original selected ready offer")
    offer, provenance = selected[0], body["input"]["provenance"]
    if (provenance["root_state"] != root["store_root_state"]
            or provenance["generation"] != root["root_generation"]
            or provenance["sequence"] != offer["view"]["sequence"]
            or provenance["view_seal_sha256"] != offer["view"]["view_digest"]
            or provenance["full_view_sha256"] != offer["full_view_sha256"]
            or provenance["policy_sha256"] != offer["view"]["context_policy_sha256"]):
        raise ValueError("utility cohort input provenance differs from its original root/selected ready view")
    return {"input": captured, "root": root, "authority": authority, "resource_known": resource_known,
        "materialization": body, "trial": trial}


def _trial_original_scope(trial, source, profile):
    """Read the actual checkpoint's root/prelude, never synthesize a root.

    These bodies are frozen by the original trial factory even if no bounded
    semantic materialization exists. A later trial does not inherit the first
    registration receipt's authority, Store identifiers or clock ticks.
    """
    root, authority = _root_structure(trial["original_root"], source)
    if root["root_state"]["semantic"] != source["rules_root"]["anchor"]["rules_state_sha256"]:
        raise ValueError("utility cohort checkpoint changed original Rules root")
    prelude, cost = trial["ready_prelude"], trial["cost"]
    resource_known = _prelude(prelude, profile=profile, root=root, authority=authority)
    if (cost["clock_epoch"] != root["deadline_clock"]
            or cost["original_deadline_tick"] != root["deadline_tick"]
            or cost["started_tick"] < prelude["captured_tick"]
            or cost["baseline_counters"] != prelude["counters"]
            or cost["native_credit_baseline"] != [prelude["native_credit"]["spent"], prelude["native_credit"]["reserved"]]):
        raise ValueError("utility cohort checkpoint cost differs from its original root/prelude")
    declared = root["root_descriptor"]["input"]
    if (_value_digest("rz-rovent-utility-cohort-logical-budget/1", declared["declared_ceilings"])
            != source["source_pins"]["logical_budget_sha256"]
            or digest("rz-rovent-resource-plan/1", declared["declared_ceilings"]["resource_plan"])
                != source["source_pins"]["resource_plan_sha256"]
            or cost["cohort_action_cutoff_tick"] > root["deadline_tick"]
            or cost["cohort_cleanup_cutoff_tick"] > declared["clock"]["cleanup"]["at_ns"]):
        raise ValueError("utility cohort checkpoint exceeds its original root budget/deadlines")
    if trial["recipient"] not in {offer["recipient"] for offer in prelude["offers"]}:
        raise ValueError("utility cohort checkpoint lacks its original ready recipient")
    return {"root": root, "authority": authority, "resource_known": resource_known}


def _ticket_projection(value, authority):
    fields(value, ("authority", "request_epoch", "request_sequence", "role", "question_sha256", "view", "representation",
        "deadline_clock", "deadline_tick"), "utility cohort original consumed role ticket")
    structure._authority(value["authority"], authority)
    if _positive(value["request_epoch"]) != authority["process_epoch"]:
        raise ValueError("utility cohort consumed ticket crossed its original process")
    _positive(value["request_sequence"])
    if value["role"] not in ("Proposer", "Critic"):
        raise ValueError("utility cohort semantic P/C ticket role differs")
    view = structure._view(value["view"], authority)
    representation = value["representation"]
    if type(representation) is not dict or representation.get("role") != value["role"].lower():
        raise ValueError("utility cohort ticket/private representation role differs")
    projected = {**authority, "request_sequence": value["request_sequence"],
        "question_digest": value["question_sha256"], "view_sequence": view["sequence"],
        "context_policy": view["context_policy_sha256"], "view_digest": view["view_digest"],
        "root_state": {"owner": view["root_state"]["owner"], "revision": view["root_state"]["revision"],
            "semantic": view["root_state"]["semantic_sha256"]},
        **{name: view[name] for name in ("actual_tokens", "selected_records", "anchors", "length_bucket")},
        "representation": representation, "deadline_clock": value["deadline_clock"], "deadline_tick": value["deadline_tick"]}
    # Explicit spelling projection only. This checker validates the original
    # same binary ticket frame and never creates an accepted native ticket.
    structure._ticket(projected)
    return projected


def _peer_projection(value, authority):
    fields(value, ("authority", "id", "parent", "sender", "recipient", "purpose", "anchor", "anchor_root_ply",
        "original_first_move", "prefix_sha256", "line_sha256", "line_offset", "question_sha256", "view",
        "evidence_revision", "root_budget_sha256", "deadline_clock", "deadline_tick"),
        "utility cohort original reciprocal peer")
    projected = {name: item for name, item in value.items() if name not in ("deadline_clock", "deadline_tick")}
    projected["deadline"] = {"clock": value["deadline_clock"], "tick": value["deadline_tick"]}
    structure._peer(projected, authority)
    return projected


def _accepted(value, authority, root):
    fields(value, ("frame_sequence", "ticket", "kind", "peer", "output_sha256", "seal_sha256"),
        "utility cohort original accepted P/C origin")
    projected = {**value, "ticket": _ticket_projection(value["ticket"], authority),
        "peer": None if value["peer"] is None else _peer_projection(value["peer"], authority)}
    return structure._origin(projected, authority, root)


def _semantic_core(scope):
    body, trial, root, authority = (scope[name] for name in ("materialization", "trial", "root", "authority"))
    value, action = body["semantic"], body["action"]
    fields(value, ("baseline_witnesses", "new_witnesses", "witness_preimages", "witness_basis",
        "pc_consumption", "dependencies", "r_selected_pc_consumption", "native_executions"),
        "utility cohort original semantic consumption")
    baseline = corpus._sha_list(value["baseline_witnesses"], 65536, "cohort baseline witness set")
    identities = set()
    for witness in _array(value["new_witnesses"], 64):
        fields(witness, ("kind", "observation", "identity_sha256", "raw_evidence_sha256",
            "source_question_sha256", "source_frame_sequence", "line"),
            "utility cohort original conditional witness")
        if witness["kind"] not in ("Counter", "Repair"):
            raise ValueError("utility cohort original witness kind differs")
        uint(witness["observation"])
        identity = sha(witness["identity_sha256"])
        sha(witness["raw_evidence_sha256"])
        sha(witness["source_question_sha256"])
        _positive(witness["source_frame_sequence"])
        _moves(witness["line"], 256)
        if identity in identities or identity in baseline:
            raise ValueError("utility cohort duplicate/baseline witness is not a new outcome")
        identities.add(identity)
    tickets = {}
    for consumed in _array(value["pc_consumption"], 64):
        fields(consumed, ("origin", "question_sha256", "view", "full_view_sha256"), "utility cohort actual P/C consumption")
        ticket = _accepted(consumed["origin"], authority, root)
        structure._view(consumed["view"], authority)
        sha(consumed["full_view_sha256"])
        if (sha(consumed["question_sha256"]) != ticket["question_digest"]
                or consumed["view"] != consumed["origin"]["ticket"]["view"]
                or ticket["representation"]["model"] != root["native_origin"]["weights_sha256"]
                or ticket["representation"]["encoding"] != root["native_origin"]["encoding_sha256"]
                or ticket["representation"]["frozen_epoch"] != root["native_origin"]["frozen_epoch"]):
            raise ValueError("utility cohort P/C semantic consumption changed original question/view/model")
        if ticket["request_sequence"] in tickets:
            raise ValueError("utility cohort duplicate original P/C consumption")
        tickets[ticket["request_sequence"]] = consumed
    executions = set()
    previous = None
    for execution in _array(value["native_executions"], 64):
        fields(execution, ("epoch", "sequence"), "utility cohort original native execution")
        key = (_positive(execution["epoch"]), _positive(execution["sequence"]))
        if key[0] != authority["process_epoch"] or key in executions or previous is not None and key <= previous:
            raise ValueError("utility cohort original native execution scope/order differs")
        executions.add(key)
        previous = key
    seen = set()
    for dependency in _array(value["dependencies"], 64):
        fields(dependency, ("frame_sequence", "question_sha256", "view", "full_view_sha256", "parent_frame_sequence",
            "alias", "canonical", "origin", "role_cause", "cpu_cause"), "utility cohort original dependency")
        frame = _positive(dependency["frame_sequence"])
        key = (frame, sha(dependency["question_sha256"]))
        if key in seen:
            raise ValueError("utility cohort duplicate original dependency")
        seen.add(key)
        structure._view(dependency["view"], authority)
        sha(dependency["full_view_sha256"])
        if dependency["parent_frame_sequence"] is not None and uint(dependency["parent_frame_sequence"]) >= frame:
            raise ValueError("utility cohort original dependency parent chronology differs")
        if (dependency["alias"] is None) != (dependency["canonical"] is None):
            raise ValueError("utility cohort dependency lacks original alias/canonical peer pair")
        for name in ("alias", "canonical"):
            if dependency[name] is not None:
                _peer_projection(dependency[name], authority)
                if (dependency[name]["root_budget_sha256"] != root["root_budget_sha256"]
                        or dependency[name]["deadline_clock"] != root["deadline_clock"]
                        or dependency[name]["deadline_tick"] != root["deadline_tick"]):
                    raise ValueError("utility cohort dependency crossed original budget/deadline")
        if dependency["origin"] is not None:
            if dependency["alias"] is None:
                raise ValueError("utility cohort dependency origin lacks original peer")
            _accepted(dependency["origin"], authority, root)
        if dependency["role_cause"] is not None and dependency["cpu_cause"] is not None:
            raise ValueError("utility cohort dependency invented multiple continuation causes")
        cause = dependency["role_cause"]
        if cause is not None:
            fields(cause, ("request_epoch", "request_sequence", "frame_sequence", "question_sha256", "view_sha256"),
                "utility cohort original role continuation cause")
            if _positive(cause["request_epoch"]) != authority["process_epoch"]:
                raise ValueError("utility cohort continuation crossed original process")
            for name in ("request_sequence", "frame_sequence"):
                _positive(cause[name])
            sha(cause["question_sha256"])
            sha(cause["view_sha256"])
            consumed = tickets.get(cause["request_sequence"])
            if (consumed is None or consumed["origin"]["frame_sequence"] != cause["frame_sequence"]
                    or consumed["question_sha256"] != cause["question_sha256"]
                    or consumed["view"]["view_digest"] != cause["view_sha256"]):
                raise ValueError("utility cohort role cause lacks original consumed P/C source")
        cause = dependency["cpu_cause"]
        if cause is not None:
            fields(cause, ("job_sequence", "native_epoch", "native_sequence", "frame_sequence", "question_sha256", "observation"),
                "utility cohort original CPU continuation cause")
            for name in ("job_sequence", "native_epoch", "native_sequence", "frame_sequence"):
                _positive(cause[name])
            uint(cause["observation"])
            sha(cause["question_sha256"])
            if (cause["native_epoch"], cause["native_sequence"]) not in executions:
                raise ValueError("utility cohort CPU cause lacks original committed execution")
    allowed = seen | {(action["frame_sequence"], action["question_sha256"])}
    if any((item["origin"]["frame_sequence"], item["question_sha256"]) not in allowed for item in tickets.values()):
        raise ValueError("utility cohort P/C source lacks its original action/dependency")
    r_sources = _array(value["r_selected_pc_consumption"], 64)
    seen_r = set()
    for consumed in r_sources:
        key = _r_semantic(consumed, scope, tickets, executions, allowed)
        if key in seen_r:
            raise ValueError("utility cohort duplicate original selected R/P-C source")
        seen_r.add(key)
    if (trial["recipient"] in ("route_p", "route_c") and not tickets
            or trial["recipient"] == "route_r" and not r_sources):
        raise ValueError("utility cohort original route lacks actual semantic consumption")
    scope["committed_sources"] = {(item["origin"]["frame_sequence"], item["question_sha256"]) for item in tickets.values()}
    scope["committed_sources"].update((item["frame_sequence"], item["question_sha256"]) for item in r_sources)
    if any((witness["source_frame_sequence"], witness["source_question_sha256"])
            not in scope["committed_sources"] for witness in value["new_witnesses"]):
        raise ValueError("utility cohort witness lacks its actual committed source question")
    return identities


def _witness_bodies(scope, *, original_store_byte_limit=None):
    semantic = scope["materialization"]["semantic"]
    if "witness_preimages" not in semantic:
        raise UnsupportedCohortPayload("cohort_original_conditional_witness_wire_not_registered")
    if semantic["witness_preimages"] is None:
        return False
    bodies = _array(semantic["witness_preimages"], 64)
    witnesses = semantic["new_witnesses"]
    if len(bodies) != len(witnesses):
        raise ValueError("utility cohort original conditional witness body coverage differs")
    total = 0
    for body, witness in zip(bodies, witnesses, strict=True):
        fields(body, ("observation", "kind", "identity_sha256", "source_question_sha256", "source_frame_sequence",
            "raw_evidence_sha256", "preimage_bytes", "preimage_hex"), "utility cohort original witness preimage")
        for name in ("observation", "kind", "identity_sha256", "raw_evidence_sha256",
                "source_question_sha256", "source_frame_sequence"):
            if body[name] != witness[name]:
                raise ValueError("utility cohort original witness/preimage metadata differs")
        uint(body["observation"])
        question = sha(body["source_question_sha256"])
        frame = _positive(body["source_frame_sequence"])
        if (frame, question) not in scope["committed_sources"]:
            raise ValueError("utility cohort witness lacks its actual committed source question")
        expected_identity = hashlib.sha256(b"rz-pals-action-unique-witness/1\0"
            + bytes.fromhex(scope["root"]["root_state"]["semantic"])
            + bytes([0 if body["kind"] == "Counter" else 1])
            + b"".join(move.to_bytes(2, "little") for move in witness["line"])).hexdigest()
        if expected_identity != sha(body["identity_sha256"]):
            raise ValueError("utility cohort witness identity changed original Rules root/kind/line")
        if not witness["line"] or witness["line"][0] != scope["materialization"]["action"]["original_first_move"]:
            raise ValueError("utility cohort witness line changed original first move")
        size = _positive(body["preimage_bytes"], 16 << 20)
        text = body["preimage_hex"]
        if type(text) is not str or len(text) != 2 * size or re.fullmatch("[0-9a-f]+", text) is None:
            raise ValueError("utility cohort original witness preimage lowercase hex/bytes differs")
        raw = bytes.fromhex(text)
        total += len(raw)
        prefix = (b"rz-rovent-utility-actual-conditional-witness/1\0" + bytes.fromhex(question)
            + frame.to_bytes(8, "little") + bytes.fromhex(body["identity_sha256"]))
        if (not raw.startswith(prefix) or len(raw) <= len(prefix)
                or hashlib.sha256(raw).hexdigest() != sha(body["raw_evidence_sha256"])):
            raise ValueError("utility cohort original conditional witness binary frame/raw seal differs")
    if original_store_byte_limit is None:
        selected = next(offer for offer in scope["materialization"]["prelude"]["offers"]
            if offer["recipient"] == scope["trial"]["recipient"])
        original_store_byte_limit = selected["budget"]["original_store_byte_limit"]
    # /2 callers pass the separately checked actual arm budget. They never
    # insert an alternative arm into the original neutral Ready offers.
    if total > min(uint(original_store_byte_limit), 16 << 20):
        raise ValueError("utility cohort original witness body budget exceeded")
    # The Rust parent alone consumes the remaining observation traversal and
    # replays its conditional Rules certificate before admitting a learner.
    return True


def _basis_digest_bytes(value):
    values = _array(value, 32)
    if len(values) != 32:
        raise ValueError("utility cohort original basis digest bytes required")
    return bytes(uint(item, 255) for item in values).hex()


def _basis_snapshot(value):
    """Check replay operands only; Rust alone replays their Rules meaning."""
    fields(value, ("origin", "start_fen", "endpoint_fen", "moves", "revision"),
        "utility cohort original basis Rules snapshot")
    uint(value["origin"], 1)
    _text(value["start_fen"], 4096)
    _text(value["endpoint_fen"], 4096)
    _moves(value["moves"], 16384)
    uint(value["revision"])
    return value


def _basis_observation(value):
    return fields(value, ("id", "state_id", "state", "line", "source", "epoch", "value", "checker", "work", "external",
        "model", "input", "condition", "cpu_pv", "scope", "score", "budget", "kind", "supersedes", "execution"),
        "utility cohort original basis observation")


def _basis_pc_origin(value, observation, scope, authority):
    """Join the original endpoint projection to an already checked P/C ticket."""
    fields(value, ("authority", "frame_sequence", "request_epoch", "request_sequence", "question", "view", "view_sequence",
        "full_view", "input", "history", "candidates", "mask_positions", "model", "encoding", "precision", "frozen_epoch",
        "revision", "role", "deadline_clock", "deadline_at", "peer", "output", "origin_seal", "namespace"),
        "utility cohort original basis endpoint P/C origin")
    original = [entry for entry in scope["materialization"]["semantic"]["pc_consumption"]
        if entry["origin"]["ticket"]["request_epoch"] == value["request_epoch"]
        and entry["origin"]["ticket"]["request_sequence"] == value["request_sequence"]]
    if len(original) != 1:
        raise ValueError("utility cohort basis endpoint lacks its original consumed P/C ticket")
    consumed = original[0]
    accepted, ticket = consumed["origin"], consumed["origin"]["ticket"]
    representation = ticket["representation"]
    integers = {"frame_sequence": accepted["frame_sequence"], "request_epoch": ticket["request_epoch"],
        "request_sequence": ticket["request_sequence"], "view_sequence": ticket["view"]["sequence"],
        "frozen_epoch": representation["frozen_epoch"], "revision": representation["record_revision"],
        "role": 0 if ticket["role"] == "Proposer" else 1, "deadline_clock": ticket["deadline_clock"],
        "deadline_at": ticket["deadline_tick"]}
    if (value["authority"] != authority or value["precision"] != "fp32"
            or representation["precision"] != "Fp32"
            or any(uint(value[name]) != expected for name, expected in integers.items())):
        raise ValueError("utility cohort basis endpoint changed original P/C authority/context/precision")
    pins = {"question": ticket["question_sha256"], "view": ticket["view"]["view_digest"],
        "full_view": consumed["full_view_sha256"], "output": accepted["output_sha256"], "origin_seal": accepted["seal_sha256"],
        **{name: representation[name] for name in ("input", "history", "candidates", "mask_positions", "model", "encoding")}}
    if any(_basis_digest_bytes(value[name]) != expected for name, expected in pins.items()):
        raise ValueError("utility cohort basis endpoint changed original P/C source/representation/output")
    declared_namespace = _basis_digest_bytes(value["namespace"])
    if value["peer"] is None:
        if accepted["peer"] is not None:
            raise ValueError("utility cohort basis endpoint lost its original reciprocal source")
    else:
        peer = _array(value["peer"], 2)
        if (len(peer) != 2 or accepted["peer"] is None or uint(peer[0]) != accepted["peer"]["id"]
                or _basis_digest_bytes(peer[1]) != accepted["peer"]["question_sha256"]):
            raise ValueError("utility cohort basis endpoint changed original reciprocal source")
    observation = _basis_observation(observation)
    _basis_snapshot(observation["state"])
    model = fields(observation["model"], ("semantics", "model", "encoding", "precision", "epoch"),
        "utility cohort original basis endpoint model")
    for name in ("semantics", "model", "encoding", "precision"):
        _text(model[name])
    model_epoch = _basis_digest_bytes(model["epoch"])
    if (model["precision"] not in ("fp32", "fp32_tf32_off")
            or _basis_digest_bytes(observation["input"]) != representation["input"]):
        raise ValueError("utility cohort basis endpoint changed original model/input namespace")
    # Display value-identity strings and representation digest keys are
    # separate original operands. The explicit native namespace bridge
    # binds both; neither is inferred by hashing or comparing the other.
    namespace = hashlib.sha256(b"rz-rovent-utility-cohort-model-namespace/1\0")
    for name in ("semantics", "model", "encoding", "precision"):
        text = model[name].encode("utf-8")
        namespace.update(len(text).to_bytes(8, "little"))
        namespace.update(text)
    namespace.update(bytes.fromhex(model_epoch))
    namespace.update(bytes.fromhex(representation["model"]))
    namespace.update(bytes.fromhex(representation["encoding"]))
    namespace.update(b"\0")
    namespace.update(representation["frozen_epoch"].to_bytes(8, "little"))
    if namespace.hexdigest() != declared_namespace:
        raise ValueError("utility cohort original explicit model representation namespace differs")
    score = fields(observation["score"], ("kind", "bits", "side", "revision"),
        "utility cohort original basis endpoint ContextWDL")
    bits = _array(score["bits"], 3)
    if len(bits) != 3 or score["kind"] != "ContextWdl" or uint(score["revision"]) != representation["record_revision"]:
        raise ValueError("utility cohort basis endpoint changed original ContextWDL revision")
    for bits_value in bits:
        uint(bits_value, 2**32 - 1)
    uint(score["side"], 1)
    # Numerical WDL validity, exact native namespace framing and conditional
    # score meaning remain Core checks against these original endpoint bytes.
    return model


def _basis_metadata(value, scope, witness, *, original_store_byte_limit=None):
    """Join original metadata, not the Core conditional comparison proof.

    The original JSON remains sealed as bytes. Nested observation scores,
    accepted endpoint namespaces, task completion and comparison lineage
    still require Core's complete pure validator and the Rust parent's held
    original source projection before any learner authority can open.
    """
    fields(value, ("schema", "domain", "root", "authority", "root_state", "kind", "line",
        "source_question", "source_frame_sequence", "identity", "raw_evidence", "root_budget",
        "original_store_byte_limit", "observation", "comparison"), "utility cohort original basis")
    if value["schema"] != BASIS_DOMAIN or value["domain"] != BASIS_DOMAIN:
        raise ValueError("utility cohort original basis schema/domain differs")
    authority = fields(value["authority"], ("epoch", "game", "root", "implementation"),
        "utility cohort original basis authority")
    if (uint(authority["epoch"]) != scope["authority"]["process_epoch"]
            or uint(authority["game"]) != scope["authority"]["game_generation"]
            or uint(authority["root"]) != scope["authority"]["root_generation"]
            or _basis_digest_bytes(authority["implementation"]) != scope["authority"]["implementation_sha256"]
            or uint(value["root_state"]) != scope["root"]["store_root_state"]
            or uint(value["kind"], 1) != (0 if witness["kind"] == "Counter" else 1)
            or uint(value["source_frame_sequence"]) != witness["source_frame_sequence"]):
        raise ValueError("utility cohort original basis authority/root/kind/source differs")
    for name, expected in (("source_question", witness["source_question_sha256"]),
            ("identity", witness["identity_sha256"]), ("raw_evidence", witness["raw_evidence_sha256"]),
            ("root_budget", scope["root"]["root_budget_sha256"])):
        if _basis_digest_bytes(value[name]) != expected:
            raise ValueError("utility cohort original basis source/witness/budget digest differs")
    _moves(value["line"], 256)
    if value["line"] != witness["line"]:
        raise ValueError("utility cohort original basis witness line differs")
    if original_store_byte_limit is None:
        selected = next(offer for offer in scope["materialization"]["prelude"]["offers"]
            if offer["recipient"] == scope["trial"]["recipient"])
        original_store_byte_limit = selected["budget"]["original_store_byte_limit"]
    if uint(value["original_store_byte_limit"]) != uint(original_store_byte_limit):
        raise ValueError("utility cohort original basis Store byte limit differs")
    snapshot = _basis_snapshot(value["root"])
    rules = scope["materialization"]["input"]["root_rules"]
    replay = rules["uci_replay"]
    if (snapshot["origin"] != (0 if replay["origin"] == "start_position" else 1)
            or snapshot["start_fen"] != replay["start_fen"] or snapshot["endpoint_fen"] != rules["fen"]
            or snapshot["moves"] != replay["moves"]):
        raise ValueError("utility cohort original basis Rules replay metadata differs")
    # Observer revisions remain original operands. Core compares restored
    # Rules state/history; equal Rules states do not imply equal revisions.
    observation = _basis_observation(value["observation"])
    if uint(observation["id"]) != witness["observation"] or uint(observation["state_id"]) != value["root_state"]:
        raise ValueError("utility cohort original basis observation/root identity differs")
    _basis_snapshot(observation["state"])
    comparison = value["comparison"]
    if type(comparison) is not dict:
        raise ValueError("utility cohort original basis comparison object required")
    if comparison.get("kind") == "Model":
        fields(comparison, ("kind", "before", "after", "origins", "superseded"),
            "utility cohort original model comparison")
        origins = _array(comparison["origins"], 2)
        if len(origins) != 2 or any(type(origin) is not dict for origin in origins):
            raise ValueError("utility cohort original model endpoint origins required")
        models = [_basis_pc_origin(origin, endpoint, scope, authority) for origin, endpoint
            in zip(origins, (comparison["before"], comparison["after"]), strict=True)]
        if models[0] != models[1]:
            raise ValueError("utility cohort original model endpoint namespaces differ")
    elif comparison.get("kind") == "Finite":
        fields(comparison, ("kind", "recipe"), "utility cohort original finite comparison")
        fields(comparison["recipe"], ("root", "before", "after", "superseded", "previous"),
            "utility cohort original finite comparison recipe")
        _basis_snapshot(comparison["recipe"]["root"])
    else:
        raise UnsupportedCohortPayload("cohort_original_conditional_basis_variant_not_registered")


def _witness_basis(scope, *, original_store_byte_limit=None):
    """Check original raw basis seals/metadata; unavailable stays masked."""
    semantic = scope["materialization"]["semantic"]
    rows = semantic["witness_basis"]
    if rows is None:
        if semantic["witness_preimages"] is not None:
            raise ValueError("utility cohort original witness/basis availability differs")
        return False
    if semantic["witness_preimages"] is None:
        raise ValueError("utility cohort original witness/basis availability differs")
    rows = _array(rows, 64)
    if len(rows) != len(semantic["new_witnesses"]):
        raise ValueError("utility cohort original conditional basis coverage differs")
    unavailable = ("missing_original_comparison", "missing_resident_evidence", "incomplete_context",
        "namespace_mismatch", "capacity", "invalid_original_basis")
    complete = True
    total = sum(body["preimage_bytes"] for body in semantic["witness_preimages"])
    for row, witness in zip(rows, semantic["new_witnesses"], strict=True):
        fields(row, ("observation", "kind", "identity_sha256", "source_question_sha256", "source_frame_sequence",
            "raw_evidence_sha256", "basis", "unavailable_reason"), "utility cohort original conditional basis")
        for name in ("observation", "kind", "identity_sha256", "source_question_sha256", "source_frame_sequence",
                "raw_evidence_sha256"):
            if row[name] != witness[name]:
                raise ValueError("utility cohort original witness/basis metadata differs")
        basis, reason = row["basis"], row["unavailable_reason"]
        if basis is None:
            if type(reason) is not str or reason not in unavailable:
                raise ValueError("utility cohort original unavailable basis reason required")
            complete = False
            continue
        fields(basis, ("domain", "sha256", "bytes", "original_json"), "utility cohort original basis bytes")
        if basis["domain"] != BASIS_DOMAIN or reason is not None:
            raise ValueError("utility cohort original basis domain/unavailable reason differs")
        sha(basis["sha256"])
        expected_bytes = _positive(basis["bytes"], 16 << 20)
        if type(basis["original_json"]) is not str:
            raise ValueError("utility cohort original basis UTF8 JSON string required")
        try:
            raw = basis["original_json"].encode("utf-8", "strict")
        except UnicodeEncodeError as error:
            raise ValueError("utility cohort original basis UTF8 JSON string required") from error
        if len(raw) != expected_bytes:
            raise ValueError("utility cohort original basis byte count differs")
        # The actual issuer hashes the complete original JSON bytes directly.
        # This is neither the native domain/length seal nor generation_digest;
        # do not reserialize parsed FP spellings before this comparison.
        if hashlib.sha256(raw).hexdigest() != basis["sha256"]:
            raise ValueError("utility cohort original basis raw byte SHA differs")
        _basis_metadata(_decode(raw, 16 << 20, source_operands=True), scope, witness,
            original_store_byte_limit=original_store_byte_limit)
        total += len(raw)
    if original_store_byte_limit is None:
        selected = next(offer for offer in scope["materialization"]["prelude"]["offers"]
            if offer["recipient"] == scope["trial"]["recipient"])
        original_store_byte_limit = selected["budget"]["original_store_byte_limit"]
    if total > min(uint(original_store_byte_limit), 16 << 20):
        raise ValueError("utility cohort original shared witness/basis body budget exceeded")
    return complete


def _physical_closure(trial, source, *, root, authority):
    value = fields(trial["physical_closure"], ("root_fence_receipt", "review_pin_count", "engine_release",
        "context_cache_release", "input_cache_prelude", "native_product"), "utility cohort original physical closure")
    receipt = fields(value["root_fence_receipt"], ("authority", "native_owner_joined", "native_owner_quiesced",
        "native_physical_shutdown_confirmed", "own_cpu_joined", "expected_own_cpu_owners", "own_cpu_panicked",
        "external_shutdown_confirmed", "external_owner_quiesced", "auxiliary_shutdown_confirmed", "native_cpu_spent",
        "native_cpu_reserved", "pending_tickets", "quarantined", "request_sequence_high_water", "permits_semantic_closure"),
        "utility cohort original root fence receipt")
    authority = structure._authority(receipt["authority"], authority)
    for name in ("native_owner_joined", "native_owner_quiesced", "native_physical_shutdown_confirmed",
            "auxiliary_shutdown_confirmed", "quarantined", "permits_semantic_closure"):
        _boolean(receipt[name], name)
    for name in ("own_cpu_joined", "expected_own_cpu_owners", "own_cpu_panicked", "native_cpu_spent", "native_cpu_reserved",
            "pending_tickets", "request_sequence_high_water"):
        uint(receipt[name])
    for name in ("external_shutdown_confirmed", "external_owner_quiesced"):
        if receipt[name] is not None:
            _boolean(receipt[name], name)
    external_closed = (receipt["external_shutdown_confirmed"] is None and receipt["external_owner_quiesced"] is None
        or receipt["external_shutdown_confirmed"] is True or receipt["external_owner_quiesced"] is True)
    necessary = ((receipt["native_owner_joined"] and receipt["native_physical_shutdown_confirmed"]
            or receipt["native_owner_quiesced"]) and receipt["own_cpu_panicked"] == 0
        and receipt["own_cpu_joined"] == receipt["expected_own_cpu_owners"] and receipt["native_cpu_reserved"] == 0
        and receipt["pending_tickets"] == 0 and not receipt["quarantined"] and receipt["auxiliary_shutdown_confirmed"]
        and external_closed)
    if (not necessary or receipt["permits_semantic_closure"] is not True or uint(value["review_pin_count"]) != 0
            or trial["cost"]["native_credit_final"] != [receipt["native_cpu_spent"], 0]
            or trial["cost"]["clock_epoch"] != authority["process_epoch"]):
        raise ValueError("utility cohort original root/actor/pin-zero closure metadata differs")
    release = fields(value["engine_release"], ("owner", "actual_stack_owner_id", "profile", "binding_sha256", "model_identity",
        "preparation_epoch", "release_duration_ns", "provenance"), "utility cohort original engine release")
    uint(release["owner"], 65535)
    uint(release["actual_stack_owner_id"])
    uint(release["preparation_epoch"])
    release_profile = fields(release["profile"], ("owner", "max_depth", "evaluation_namespace_sha256", "search_conditions_sha256"),
        "utility cohort original released CPU profile")
    if uint(release_profile["owner"], 65535) != release["owner"]:
        raise ValueError("utility cohort release changed original CPU owner")
    uint(release_profile["max_depth"], 65535)
    sha(release_profile["search_conditions_sha256"])
    if sha(release_profile["evaluation_namespace_sha256"]) != sha(release["binding_sha256"]):
        raise ValueError("utility cohort release changed original value binding")
    if structure._identity(release["model_identity"]) != source["nnuec_registration"]["model_identity"]:
        raise ValueError("utility cohort released engine changed frozen NNUEC model")
    duration = release["release_duration_ns"]
    if (type(duration) is not str or re.fullmatch(r"0|[1-9][0-9]{0,38}", duration) is None
            or int(duration) > 2**128 - 1):
        raise ValueError("utility cohort original engine release duration string differs")
    provenance = fields(release["provenance"], ("root", "generation", "sequence", "full_view_sha256", "view_seal_sha256",
        "record_scope_sha256", "policy_sha256", "model"), "utility cohort original engine release provenance")
    for name in ("root", "generation", "sequence"):
        uint(provenance[name])
    for name in ("full_view_sha256", "view_seal_sha256", "record_scope_sha256", "policy_sha256"):
        sha(provenance[name])
    if (structure._identity(provenance["model"]) != release["model_identity"]
            or provenance["generation"] != authority["root_generation"]):
        raise ValueError("utility cohort release provenance changed original model/generation")
    if provenance["root"] != root["store_root_state"]:
        raise ValueError("utility cohort engine release changed original root state owner")
    # Context revisions can advance during this action. The release's original
    # provenance is retained separately; equality with the earlier capture is
    # not implied by their common cache-release owner/epoch.
    cache = fields(value["context_cache_release"], ("epoch", "matches_engine_release"), "utility cohort original context cache release")
    uint(cache["epoch"])
    if _boolean(cache["matches_engine_release"], "original context cache release matches engine") is not True:
        raise ValueError("utility cohort context cache release differs from original engine")
    cache_prelude = value["input_cache_prelude"]
    cache_known = False
    if cache_prelude is not None:
        fields(cache_prelude, ("epoch", "matches_input"), "utility cohort original input cache prelude")
        uint(cache_prelude["epoch"])
        cache_known = _boolean(cache_prelude["matches_input"], "original input cache prelude matches input")
    product = fields(value["native_product"], ("original_lifetime_present", "first_product_root"),
        "utility cohort original product lifetime")
    for name in product:
        _boolean(product[name], name)
    return {"authority": authority, "cache_known": cache_known,
        "fresh_product": product["original_lifetime_present"] and product["first_product_root"]}


def _historical_input(trial, source, config):
    value = trial["historical_unbound_input"]
    if value is None:
        return None
    return validate_submitted_utility_input(input_json=value["reference_json"],
        base_weights_sha256=source["nnuec_registration"]["model_identity"]["weights_sha256"],
        context_sha256=value["context_sha256"], target_ids=value["target_record_ids"],
        view_seal_sha256=value["original_view_sha256"], input_sha256=value["input_sha256"],
        target_sha256=value["target_sha256"], submission_target_sha256=value["submission_target_sha256"], config=config)


def _pair_candidates(files, trials, scopes, physical):
    """Compute declared pair candidates; Rust still validates certificates."""
    publication = files.publication()
    known, known_sha, all_sha, seen = [], [], [], set()
    pair_fields = ("schema", "cohort_id", "left_trial_sequence", "right_trial_sequence", "left_trial_sha256", "right_trial_sha256",
        "left_recipient", "right_recipient", "preferred", "unknown_reason", "support_mask", "target_known_mask",
        "left_new_witnesses", "right_new_witnesses", "left_whole_wall_ns", "right_whole_wall_ns", "cost_scope", "objective")
    for pin in publication["pairs"]:
        raw = files.raw_file(pin["path"])
        pair = fields(_decode(raw, 64 << 10), pair_fields, "utility cohort original pair")
        if (pair["schema"] != PAIR_DOMAIN or pair["cohort_id"] != publication["cohort_id"]
                or pair["cost_scope"] != COST_SCOPE or pair["objective"] != COST_OBJECTIVE):
            raise ValueError("utility cohort original pair domain/group/cost differs")
        left_number = _positive(pair["left_trial_sequence"], len(trials))
        right_number = _positive(pair["right_trial_sequence"], len(trials))
        key = tuple(sorted((left_number, right_number)))
        if key in seen:
            raise ValueError("utility cohort duplicate original unordered comparison")
        seen.add(key)
        left, right = trials[left_number - 1], trials[right_number - 1]
        for side, number, trial in (("left", left_number, left), ("right", right_number, right)):
            trial_raw = files.raw_file(publication["trials"][number - 1]["path"])
            if (sha(pair[side + "_trial_sha256"]) != _native_digest(TRIAL_DOMAIN, trial_raw)
                    or pair[side + "_recipient"] != trial["recipient"]
                    or uint(pair[side + "_whole_wall_ns"]) != trial["cost"]["whole_wall_ns"]):
                raise ValueError("utility cohort pair changed original trial/native seal/route/cost")
            declared = corpus._sha_list(pair[side + "_new_witnesses"], 64, "cohort pair new witnesses")
            if number in scopes and declared != sorted(scopes[number]["witnesses"]):
                raise ValueError("utility cohort pair changed original new witness set")
        left_set, right_set = set(pair["left_new_witnesses"]), set(pair["right_new_witnesses"])
        preferred, reason = None, None
        if (left_number == right_number or pair["left_trial_sha256"] == pair["right_trial_sha256"]
                or physical[left_number]["authority"] == physical[right_number]["authority"]):
            reason = "duplicate_independent_trial"
        elif left["recipient"] == right["recipient"]:
            reason = "same_recipient"
        elif not left["comparable"] or not right["comparable"]:
            reason = "unresolved_or_masked_trial"
        elif not left_set and not right_set:
            reason = "no_new_valid_witness"
        elif left_set == right_set and pair["left_whole_wall_ns"] == pair["right_whole_wall_ns"]:
            reason = "equal_observation"
        elif left_set.issuperset(right_set) and pair["left_whole_wall_ns"] <= pair["right_whole_wall_ns"]:
            preferred = left["recipient"]
        elif right_set.issuperset(left_set) and pair["right_whole_wall_ns"] <= pair["left_whole_wall_ns"]:
            preferred = right["recipient"]
        else:
            reason = "pareto_incomparable"
        support = [a or b for a, b in zip(left["support_mask"], right["support_mask"], strict=True)]
        target = [False] * 7
        if preferred is not None:
            target[4 + _ROUTES.index(left["recipient"])] = True
            target[4 + _ROUTES.index(right["recipient"])] = True
        for name in ("support_mask", "target_known_mask"):
            bits = _array(pair[name], 7)
            if len(bits) != 7 or any(type(bit) is not bool for bit in bits):
                raise ValueError("utility cohort exact original pair masks required")
        if (pair["preferred"] != preferred or pair["unknown_reason"] != reason
                or pair["support_mask"] != support or pair["target_known_mask"] != target):
            raise ValueError("utility cohort original pair preference/mask/unknown arithmetic differs")
        native_sha = _native_digest(PAIR_DOMAIN, raw)
        all_sha.append(native_sha)
        if preferred is not None:
            # Both known trials have already passed physical, cache, fresh
            # product and original witness-byte joins below. The remaining
            # conditional observation/Rules proof belongs to Rust's intake.
            if left_number not in scopes or right_number not in scopes:
                raise ValueError("utility cohort known pair lacks original materialized trials")
            other = right["recipient"] if preferred == left["recipient"] else left["recipient"]
            known.append([4 + _ROUTES.index(preferred), 4 + _ROUTES.index(other)])
            known_sha.append(native_sha)
    return known, known_sha, all_sha


def _r_semantic(value, scope, tickets, executions, allowed):
    root, authority, original_input = scope["root"], scope["authority"], scope["materialization"]["input"]
    fields(value, ("stage", "authority", "frame_sequence", "question_sha256", "submitted_view", "submitted_full_view_sha256",
        "job", "owner", "store_execution", "observation", "native_execution_epoch", "native_execution_sequence",
        "capture_sequence", "binding_sha256", "capture_full_view_sha256", "capture_view_seal_sha256", "slice_work",
        "selected_pc_ticket", "selected_output_sha256", "selected_record", "selected_full_view_sha256"),
        "utility cohort original selected R/P-C consumption")
    if value["stage"] != "SelectedPcViewConsumed":
        raise ValueError("utility cohort native R feedback lacks original selected P/C consumption")
    structure._authority(value["authority"], authority)
    frame = _positive(value["frame_sequence"])
    question = sha(value["question_sha256"])
    if (frame, question) not in allowed:
        raise ValueError("utility cohort native R source lacks original action/dependency")
    structure._view(value["submitted_view"], authority)
    for name in ("submitted_full_view_sha256", "binding_sha256", "capture_full_view_sha256", "capture_view_seal_sha256",
            "selected_full_view_sha256"):
        sha(value[name])
    uint(value["owner"], 65535)
    for name in ("store_execution", "observation", "capture_sequence"):
        uint(value[name])
    execution = (_positive(value["native_execution_epoch"]), _positive(value["native_execution_sequence"]))
    if execution not in executions:
        raise ValueError("utility cohort R consumption lacks original committed native execution")
    job = fields(value["job"], ("authority", "sequence", "state_owner", "state_revision", "state_sha256", "question_sha256",
        "root_restriction_sha256", "evaluation_namespace_sha256", "search_conditions_sha256", "node_credit", "deadline_clock",
        "deadline_tick"), "utility cohort original native R job")
    structure._authority(job["authority"], authority)
    _positive(job["sequence"])
    for name in ("state_owner", "state_revision", "node_credit", "deadline_clock", "deadline_tick"):
        uint(job[name])
    for name in ("state_sha256", "question_sha256", "root_restriction_sha256", "evaluation_namespace_sha256", "search_conditions_sha256"):
        sha(job[name])
    if (job["question_sha256"] != question or job["evaluation_namespace_sha256"] != value["binding_sha256"]
            or job["deadline_clock"] != root["deadline_clock"] or job["deadline_tick"] > root["deadline_tick"]):
        raise ValueError("utility cohort R job changed original question/namespace/deadline")
    work = fields(value["slice_work"], ("nodes", "quiescence_nodes", "tt_hits"), "utility cohort original R slice work")
    for count in work.values():
        uint(count)
    ticket = _ticket_projection(value["selected_pc_ticket"], authority)
    selected = fields(value["selected_record"], ("revision", "origin_state"), "utility cohort original selected R record")
    uint(selected["revision"])
    uint(selected["origin_state"])
    consumed = tickets.get(ticket["request_sequence"])
    captured = original_input["provenance"]
    if (consumed is None or consumed["origin"]["ticket"] != value["selected_pc_ticket"]
            or sha(value["selected_output_sha256"]) != consumed["origin"]["output_sha256"]
            or value["selected_full_view_sha256"] != consumed["full_view_sha256"]
            or value["capture_full_view_sha256"] != captured["full_view_sha256"]
            or value["capture_view_seal_sha256"] != captured["view_seal_sha256"]):
        raise ValueError("utility cohort R feedback changed original capture/accepted P/C output")
    return execution, ticket["request_sequence"]


def _frozen(value):
    fields(value, _FROZEN_FIELDS, "utility cohort frozen decision")
    for name in ("input_sha256", "target_sha256", "context_sha256", "root_rules_sha256"):
        sha(value[name])
    identifiers = _array(value["target_record_ids"], 64)
    for item in identifiers:
        uint(item, 63)
    if not identifiers or identifiers != sorted(set(identifiers)):
        raise ValueError("utility cohort frozen original ordered target IDs required")
    _positive(value["input_bytes"], 8 << 20)
    uint(value["original_first_move"], 65535)
    _positive(value["first_trial_sequence"], 3)


def _io(value):
    fields(value, _IO_FIELDS, "utility cohort original trial ACK collection I/O")
    for item in value.values():
        uint(item)
    if value["finished_tick_ns"] - value["started_tick_ns"] != value["elapsed_ns"]:
        raise ValueError("utility cohort original trial ACK I/O clock differs")
    return value


def _resource_observations(value):
    fields(value, ("native", "own"), "utility cohort actual resource observations")
    methods = ("LinuxProcThreadStatus", "LinuxProcProcessStatus", "LinuxSysfsSiblings",
        "LinuxCgroupV2Direct", "LinuxCgroupV2VisibleAncestors", "WindowsProcessAndThreadGroupAffinity",
        "WindowsLogicalProcessorInformationEx", "WindowsImmediateJobObject", "WorkerReadyBoundary",
        "OrtAppliedSessionSettings")
    unknowns = ("UnsupportedPlatform", "ReadUnavailable", "InvalidFormat", "BudgetExceeded",
        "NamespaceOrMountAmbiguous", "ChangedDuringProbe", "HiddenAncestors",
        "WindowsJobHierarchyUnavailable", "WindowsGroupAffinityAmbiguous", "UnobservedThread",
        "UnobservedOrtSettings")

    def one(observed):
        if observed is None:
            return False
        if type(observed) is dict and set(observed) == {"error"}:
            _text(observed["error"])
            return False
        fields(observed, ("actor", "thread_id_present", "affinity", "probe_method", "unknown_reason"),
            "utility cohort actual thread observation")
        actor = _text(observed["actor"])
        if actor not in ("Manager", "NativeCaller") and re.fullmatch(
                r"(?:OwnWorker|ExternalEngine|RuntimeHelper)\([0-9]+\)", actor) is None:
            raise ValueError("utility cohort original CPU actor differs")
        present = _boolean(observed["thread_id_present"], "original thread ID observed")
        if observed["affinity"] is None:
            if observed["probe_method"] is not None or observed["unknown_reason"] not in unknowns:
                raise ValueError("utility cohort unknown resource readback differs")
            return False
        affinity = _array(observed["affinity"], 65536)
        for cpu in affinity:
            if type(cpu) is not list or len(cpu) != 2:
                raise ValueError("utility cohort original CPU affinity pair required")
            uint(cpu[0], 65535)
            uint(cpu[1], 2**32 - 1)
        if observed["unknown_reason"] is not None or observed["probe_method"] not in methods:
            raise ValueError("utility cohort known resource readback differs")
        return present and bool(affinity)

    native_known = one(value["native"])
    own = _array(value["own"], 4)
    own_known = [one(item) for item in own]
    return native_known and bool(own_known) and all(own_known)


def _prelude(value, *, profile, root, authority):
    """Check the actual ready-group preimage; no ready offer is issued here."""
    fields(value, ("captured_tick", "native_credit", "role_credit", "counters", "offers",
        "requested_common_nodes", "common_node_work", "ready_group_sha256", "actual_resource_observations",
        "native_thread_readback", "own_thread_readbacks"),
        "utility cohort original ready prelude")
    captured = uint(value["captured_tick"])
    if captured >= root["deadline_tick"]:
        raise ValueError("utility cohort ready prelude exceeds original deadline")
    credit = fields(value["native_credit"], ("limit", "spent", "reserved", "highest_job_sequence", "highest_execution_sequence"),
        "utility cohort original native credit prelude")
    for item in credit.values():
        uint(item)
    if credit["spent"] + credit["reserved"] > credit["limit"]:
        raise ValueError("utility cohort original native prelude credit exceeds limit")
    roles = fields(value["role_credit"], ("spent", "reserved"), "utility cohort original role prelude credit")
    for item in roles.values():
        uint(item)
    counters = fields(value["counters"], _COUNTER_FIELDS, "utility cohort original prelude counters")
    for item in counters.values():
        uint(item)
    if value["requested_common_nodes"] is not None:
        uint(value["requested_common_nodes"])
    if value["requested_common_nodes"] != root["requested_nodes"]:
        raise ValueError("utility cohort original common node request differs")
    common = fields(value["common_node_work"], ("root_entries", "checked_transitions", "native_nodes",
        "native_qsearch_nodes", "reserved_native_nodes", "outstanding_native_executions", "native_usage_known"),
        "utility cohort original common node work")
    for name in set(common) - {"native_usage_known"}:
        uint(common[name])
    _boolean(common["native_usage_known"], "original native common work known")
    resource_known = _resource_observations(value["actual_resource_observations"])
    for name in ("native_thread_readback", "own_thread_readbacks"):
        _text(value[name], MAX_MATERIALIZATION_BYTES)
    offers = _array(value["offers"], 3)
    if not offers:
        raise ValueError("utility cohort original nonempty ready offers required")
    stages = {
        "proposer": ("ProposerRanking", "ProposalContinuation", "Repair", "ProposalValueFresh", "CounterValueFresh",
            "RepairValueFresh", "FreshRepairedEndpoint", "FreshCounterEndpoint", "RecheckRepairedValueFresh"),
        "critic": ("CriticDivergence", "CriticReply", "CounterContinuation", "CriticRecheck"),
        "researcher": ("CpuValidation", "CpuDiscovery", "CounterValidation", "RepairValidation", "RecheckCounterValidation"),
    }
    seen, ready = set(), []
    for offer in offers:
        fields(offer, ("frame_sequence", "recipient", "stage", "original_first_move", "question_sha256", "view",
            "full_view_sha256", "public_context_sha256", "semantic_sha256", "budget", "deadline_clock", "deadline_tick"),
            "utility cohort original ready offer")
        _positive(offer["frame_sequence"])
        route = offer["recipient"]
        if type(route) is not str or route not in _ROUTES or route in seen:
            raise ValueError("utility cohort original ready recipient/stage differs")
        recipient = _RECIPIENT_FOR_ROUTE[route]
        if offer["stage"] not in stages[recipient]:
            raise ValueError("utility cohort original ready recipient/stage differs")
        if profile["input"]["mode"] == "cpu_pc" and route == "route_r":
            raise ValueError("utility cohort CPU P-C prelude cannot own native R offer")
        seen.add(route)
        uint(offer["original_first_move"], 65535)
        for name in ("question_sha256", "full_view_sha256", "public_context_sha256", "semantic_sha256"):
            sha(offer[name])
        structure._view(offer["view"], authority)
        budget = fields(offer["budget"], ("role_submissions_remaining", "original_cpu_credit", "original_store_byte_limit"),
            "utility cohort original ready offer budget")
        for item in budget.values():
            uint(item)
        if (uint(offer["deadline_clock"]) != root["deadline_clock"]
                or uint(offer["deadline_tick"]) != root["deadline_tick"]):
            raise ValueError("utility cohort original ready offer deadline differs")
        ready.append((4 + _ROUTES.index(route), offer["semantic_sha256"]))
    preimage = b"rz-rovent-utility-cohort-ready-offer-group/1\0" + len(ready).to_bytes(8, "little")
    for route, semantic in sorted(ready):
        preimage += bytes([route]) + bytes.fromhex(semantic)
    preimage += bytes.fromhex(_value_digest("rz-rovent-utility-cohort-common-node-work/1",
        {"requested_nodes": value["requested_common_nodes"], "work": common}))
    preimage += b"".join(item.to_bytes(8, "little") for item in (
        credit["limit"], credit["spent"], credit["reserved"], roles["spent"], roles["reserved"]))
    # Wait-loop iterations are preserved as original provenance, but only
    # this explicitly documented comparison-allocation digest excludes them.
    allocation_counters = {name: item for name, item in counters.items() if name != "notification_waits"}
    preimage += bytes.fromhex(_value_digest("rz-rovent-utility-cohort-prelude-counters/1", allocation_counters))
    preimage += bytes.fromhex(_value_digest("rz-rovent-utility-cohort-actual-resource-observations/1",
        value["actual_resource_observations"]))
    if hashlib.sha256(preimage).hexdigest() != sha(value["ready_group_sha256"]):
        raise ValueError("utility cohort original actual ready-group preimage differs")
    return resource_known


def _cost(value, profile):
    fields(value, _COST_FIELDS, "utility cohort original whole cost")
    declared = profile["input"]
    if (value["domain"] != COST_DOMAIN or value["scope"] != COST_SCOPE
            or value["objective"] != COST_OBJECTIVE
            or sha(value["cost_class_sha256"]) != declared["cost_class_sha256"]
            or sha(value["resource_plan_sha256"]) != declared["resource_plan_sha256"]
            or _positive(value["resource_epoch"]) != declared["resource_epoch"]):
        raise ValueError("utility cohort original whole cost profile differs")
    if value["cost_epoch_sha256"] is not None:
        sha(value["cost_epoch_sha256"])
    _boolean(value["complete_scope"], "complete whole cost scope")
    ticks = tuple(uint(value[name]) for name in (
        "started_tick", "root_fence_tick", "actor_release_tick", "physical_pin_zero_tick"))
    if (ticks != tuple(sorted(ticks))
            or uint(value["whole_wall_ns"]) != ticks[-1] - ticks[0]
            or uint(value["cohort_action_cutoff_tick"]) != ticks[0] + declared["action_ns"]
            or uint(value["cohort_cleanup_cutoff_tick"]) != value["cohort_action_cutoff_tick"] + declared["cleanup_ns"]):
        raise ValueError("utility cohort original whole cost clock/cutoff differs")
    uint(value["original_deadline_tick"])
    uint(value["clock_epoch"])
    for name in ("cpu_thread_time_ns", "energy"):
        if value[name] is not None:
            uint(value[name])
    for name in ("baseline_counters", "final_counters"):
        fields(value[name], _COUNTER_FIELDS, "utility cohort original work counters")
        for item in value[name].values():
            uint(item)
    if any(value["final_counters"][name] < value["baseline_counters"][name] for name in _COUNTER_FIELDS):
        raise ValueError("utility cohort original whole cost counter reversal")
    for name in ("native_credit_baseline", "native_credit_final"):
        credit = _array(value[name], 2)
        if len(credit) != 2:
            raise ValueError("utility cohort native credit tuple required")
        for item in credit:
            uint(item)
    if value["native_credit_final"][0] < value["native_credit_baseline"][0]:
        raise ValueError("utility cohort native credit consumed total reversal")


def _trial_metadata(files, profile):
    """Check closed original clock/materialization linkage, not proof meaning."""
    publication = files.publication()
    profile_sha = cohort_profile_sha256(profile)
    materializations = {int(pin["path"].split("/")[1][:6]): pin for pin in publication["materializations"]}
    observed = []
    frozen = None
    for ordinal, pin in enumerate(publication["trials"]):
        body = _decode(files.raw_file(pin["path"]), MAX_MATERIALIZATION_BYTES)
        fields(body, _TRIAL_FIELDS, "utility cohort original trial checkpoint")
        if (body["schema"] != TRIAL_DOMAIN or body["cohort_id"] != publication["cohort_id"]
                or _positive(body["trial_sequence"], 3) != ordinal + 1
                or body["profile"] != profile or sha(body["profile_sha256"]) != profile_sha
                or body["recipient"] not in [_ROUTE_FOR_RECIPIENT[item] for item in profile["input"]["requested_recipients"]]):
            raise ValueError("utility cohort original trial profile/order/recipient differs")
        _source(body["source"], profile)
        for name in ("resolved", "comparable"):
            _boolean(body[name], name)
        if body["frozen_context"] is not None:
            _frozen(body["frozen_context"])
            if frozen is None:
                frozen = body["frozen_context"]
            elif any(body["frozen_context"][name] != frozen[name] for name in _FROZEN_FIELDS[:-1]):
                raise ValueError("utility cohort original frozen decision differs between trials")
            if body["frozen_context"]["first_trial_sequence"] > ordinal + 1:
                raise ValueError("utility cohort frozen first trial chronology differs")
        elif body["resolved"] or body["comparable"] or body["materialized_json"] is not None:
            raise ValueError("utility cohort unbound trial cannot synthesize a frozen decision")
        known = _array(body["target_known_mask"], 7)
        if len(known) != 7 or any(type(item) is not bool or item for item in known):
            raise ValueError("utility cohort trial cannot issue a known pair mask")
        support = _array(body["support_mask"], 7)
        route_index = 4 + _ROUTES.index(body["recipient"])
        if (len(support) != 7 or any(type(item) is not bool for item in support)
                or any(bit for index, bit in enumerate(support) if index != route_index)
                or body["resolved"] and not support[route_index]):
            raise ValueError("utility cohort original actual route support mask differs")
        reason = body["unknown_reason"]
        if reason is not None and (type(reason) is not str or not reason or len(reason.encode("utf-8")) > 256):
            raise ValueError("utility cohort bounded original unknown reason required")
        if body["comparable"] != (reason is None):
            raise ValueError("utility cohort original comparable/unknown observation differs")
        _cost(body["cost"], profile)
        previous_ack = body["previous_durable_ack"]
        if ordinal == 0:
            if previous_ack is not None:
                raise ValueError("utility cohort first trial cannot invent a preceding durable ACK")
        else:
            fields(previous_ack, ("trial_sequence", "trial_sha256", "file_sha256"),
                "utility cohort original preceding durable ACK")
            previous_pin = publication["trials"][ordinal - 1]
            previous_raw = files.raw_file(previous_pin["path"])
            if (uint(previous_ack["trial_sequence"]) != ordinal
                    or sha(previous_ack["trial_sha256"]) != _native_digest(TRIAL_DOMAIN, previous_raw)
                    or sha(previous_ack["file_sha256"]) != previous_pin["sha256"]):
                raise ValueError("utility cohort original preceding durable ACK causal chain differs")
        unresolved = body["unresolved_source"]
        if unresolved is not None:
            fields(unresolved, ("question_sha256", "frame_sequence", "recipient", "root_budget_sha256", "frozen_context_sha256",
                "reason", "actual_pc_consumptions", "actual_dependencies", "actual_r_pc_consumptions", "new_witnesses"),
                "utility cohort original unresolved source")
            for name in ("question_sha256", "root_budget_sha256", "frozen_context_sha256"):
                sha(unresolved[name])
            _positive(unresolved["frame_sequence"])
            if unresolved["recipient"] != body["recipient"] or unresolved["reason"] not in (
                    "cancelled", "deadline", "replaced_root", "semantic_failure", "physical_unknown", "missing_actual_consumption",
                    "missing_exclusive_whole_cost"):
                raise ValueError("utility cohort original unresolved route/reason differs")
            for name in ("actual_pc_consumptions", "actual_dependencies", "actual_r_pc_consumptions"):
                uint(unresolved[name])
            for witness in _array(unresolved["new_witnesses"], 64):
                fields(witness, ("observation", "identity_sha256", "raw_evidence_sha256"), "utility cohort unresolved witness")
                uint(witness["observation"])
                sha(witness["identity_sha256"])
                sha(witness["raw_evidence_sha256"])
            if body["resolved"] or body["comparable"]:
                raise ValueError("utility cohort unresolved source cannot become a comparable completed route")
        historical_input = body["historical_unbound_input"]
        if historical_input is not None:
            fields(historical_input, ("reference_json", "input_sha256", "target_sha256", "submission_target_sha256", "context_sha256",
                "target_record_ids", "original_view_sha256", "original_full_view_sha256", "materialized_inside_complete_scope"),
                "utility cohort original historical-only input")
            if historical_input["materialized_inside_complete_scope"] is not False:
                raise ValueError("utility cohort historical-only input cannot claim measured materialization")
            for name in ("input_sha256", "target_sha256", "submission_target_sha256", "context_sha256", "original_view_sha256",
                    "original_full_view_sha256"):
                sha(historical_input[name])
            if type(historical_input["reference_json"]) is not str:
                raise ValueError("utility cohort original historical reference JSON string required")
            for identifier in _array(historical_input["target_record_ids"], 64):
                uint(identifier, 63)
        materialized_pin = materializations.get(ordinal)
        if body["materialized_json"] is None:
            if body["materialized_sha256"] is not None or materialized_pin is not None:
                raise ValueError("utility cohort missing original materialization cannot be synthesized")
        else:
            if type(body["materialized_json"]) is not str or materialized_pin is None:
                raise ValueError("utility cohort original embedded materialization required")
            try:
                original = body["materialized_json"].encode("utf-8", "strict")
            except UnicodeEncodeError as error:
                raise ValueError("utility cohort original materialization UTF8 required") from error
            if (original != files.raw_file(materialized_pin["path"])
                    or sha(body["materialized_sha256"]) != _native_digest(MATERIALIZATION_DOMAIN, original)):
                raise ValueError("utility cohort embedded original materialization raw/native seal differs")
            materialized = _decode(original, MAX_MATERIALIZATION_BYTES)
            fields(materialized, ("schema", "cohort_id", "trial_sequence", "recipient", "profile_sha256",
                "source", "original_root", "prelude", "input", "action", "semantic"),
                "utility cohort original materialization")
            if (materialized["schema"] != MATERIALIZATION_DOMAIN
                    or any(materialized[name] != body[name] for name in (
                        "cohort_id", "trial_sequence", "recipient", "profile_sha256", "source", "original_root"))
                    or materialized["prelude"] != body["ready_prelude"]):
                raise ValueError("utility cohort original materialization/trial linkage differs")
        observed.append(body)
    return tuple(observed)


def validate_trial_acknowledgements(files):
    """Join ACK4/IO8 with original checkpoint raw pins and native trial seals.

    This deliberately returns detached observations, not live FD-bearing ACKs.
    It does not infer physical pin-zero or independent trial authority.
    """
    if type(files) is not HistoricalUtilityCohortFiles:
        raise ValueError("utility cohort original pinned file owner required")
    publication, ack = files.publication(), files.durable_ack()
    acknowledgements = _array(ack["trial_acknowledgements"], 3)
    if len(acknowledgements) != len(publication["trials"]):
        raise ValueError("utility cohort original trial ACK complete coverage required")
    materialized = {int(pin["path"].split("/")[1][:6]): pin for pin in publication["materializations"]}
    result = []
    for ordinal, (item, pin) in enumerate(zip(acknowledgements, publication["trials"])):
        fields(item, ("trial_sequence", "native_trial_sha256", "checkpoint", "collection_io"),
            "utility cohort original trial ACK")
        fields(item["checkpoint"], ("path", "sha256", "bytes"), "utility cohort original trial ACK pin")
        if (uint(item["trial_sequence"], 3) != ordinal + 1
                or item["checkpoint"] != pin or type(item["checkpoint"]["path"]) is not str):
            raise ValueError("utility cohort original trial ACK checkpoint/order differs")
        sha(item["checkpoint"]["sha256"])
        _positive(item["checkpoint"]["bytes"], MAX_MATERIALIZATION_BYTES)
        raw = files.raw_file(pin["path"])
        body = _decode(raw, MAX_MATERIALIZATION_BYTES)
        if (body.get("schema") != TRIAL_DOMAIN or type(body.get("trial_sequence")) is not int
                or body["trial_sequence"] != ordinal + 1 or body.get("cohort_id") != publication["cohort_id"]
                or sha(item["native_trial_sha256"]) != _native_digest(TRIAL_DOMAIN, raw)):
            raise ValueError("utility cohort original trial ACK native body seal differs")
        interval = _io(item["collection_io"])
        materialized_bytes = materialized.get(ordinal, {}).get("bytes", 0)
        if (interval["written_bytes"] != materialized_bytes + pin["bytes"]
                or interval["readback_bytes"] != materialized_bytes + 2 * pin["bytes"]
                or interval["synced_files"] != 2 + int(materialized_bytes > 0)
                or interval["synced_directories"] != 3 * (1 + int(materialized_bytes > 0))):
            raise ValueError("utility cohort original trial ACK I/O accounting differs")
        result.append(copy.deepcopy(item))
    return tuple(result)


def validate_cohort_payloads(files, *, preparation):
    """Consume only the independently registered cohort/PREP2 path.

    Known pairs are historical data candidates, not live/Rules authority.
    Rust must validate the remaining conditional observation traversal and
    full Rules replay, then compare the original held source projection,
    before the current learner's authority gate can open. This entry never
    substitutes root-local evidence for the independently registered cohort.
    """
    checked = validate_training_preparation(preparation)
    if checked["schema"] != COHORT_PREPARATION_DOMAIN:
        raise ValueError("utility cohort requires explicit preparation/2")
    if type(files) is not HistoricalUtilityCohortFiles:
        raise ValueError("utility cohort original pinned file owner required")
    profile = validate_cohort_profile(_decode(files.raw_file("cohort-profile.json"), MAX_PROFILE_BYTES))
    acknowledgements = validate_trial_acknowledgements(files)
    trials = _trial_metadata(files, profile)
    source, lock, config = _source_enrollment(files, profile, checked)
    frozen_pc_sources = _source_pc_models(source, lock)
    registration_root, registration_authority = _root_structure(source["registration_root"], source)
    if (registration_root["native_origin"] != source["native_origin"]
            or registration_root["root_state"]["semantic"] != source["rules_root"]["anchor"]["rules_state_sha256"]):
        raise ValueError("utility cohort original registration root/source/Rules differs")
    frozen = _decode(files.raw_file("frozen-context.json"), MAX_MATERIALIZATION_BYTES, allow_null=True)
    if frozen is None:
        if any(trial["frozen_context"] is not None for trial in trials):
            raise ValueError("utility cohort final frozen context lost its original bound input")
        raise UnsupportedCohortPayload("cohort_has_no_original_submitted_context")
    _frozen(frozen)
    if frozen["root_rules_sha256"] != source["rules_root"]["anchor"]["rules_state_sha256"]:
        raise ValueError("utility cohort frozen context changed original source Rules root")
    originals = {}
    for trial in trials:
        originals[trial["trial_sequence"]] = _trial_original_scope(trial, source, profile)
        if trial["frozen_context"] is not None and trial["frozen_context"] != frozen:
            raise ValueError("utility cohort original final frozen decision differs from checkpoint")
        if (trial["cost"]["cost_epoch_sha256"] is not None
                and trial["cost"]["cost_epoch_sha256"] != checked["registration"]["cost_epoch_sha256"]):
            raise ValueError("utility cohort original cost epoch differs from preparation/2")
    scopes = {}
    for trial in trials:
        if trial["materialized_json"] is None:
            continue
        body = _decode(trial["materialized_json"].encode("utf-8"), MAX_MATERIALIZATION_BYTES)
        scope = _materialization_scope(body, trial, source, config, profile,
            cohort_id=files.publication()["cohort_id"])
        scope["witnesses"] = _semantic_core(scope)
        scope["witness_bodies_complete"] = _witness_bodies(scope)
        scope["witness_basis_complete"] = _witness_basis(scope)
        scopes[trial["trial_sequence"]] = scope
    first_prelude = trials[0]["ready_prelude"]
    ready = {offer["recipient"] for offer in first_prelude["offers"]}
    planned = [_ROUTE_FOR_RECIPIENT[recipient] for recipient in profile["input"]["requested_recipients"]
        if _ROUTE_FOR_RECIPIENT[recipient] in ready]
    expected_cohort = hashlib.sha256(b"rz-rovent-utility-cohort-owner/1\0"
        + bytes.fromhex(cohort_profile_sha256(profile)) + bytes.fromhex(registration_root["root_descriptor_sha256"])
        + bytes.fromhex(first_prelude["ready_group_sha256"])
        + bytes(4 + _ROUTES.index(recipient) for recipient in planned)).hexdigest()
    if expected_cohort != files.publication()["cohort_id"]:
        raise ValueError("utility cohort original registration root/ready-group seal differs")
    for trial in trials:
        if trial["ready_prelude"]["ready_group_sha256"] != first_prelude["ready_group_sha256"]:
            raise ValueError("utility cohort independent trials changed original ready allocation")
    physical = {}
    historical = {}
    for trial, acknowledgement in zip(trials, acknowledgements, strict=True):
        number = trial["trial_sequence"]
        scope = scopes.get(number)
        physical[number] = _physical_closure(trial, source,
            root=originals[number]["root"], authority=originals[number]["authority"])
        historical[number] = _historical_input(trial, source, config)
        if historical[number] is not None and trial["frozen_context"] is not None:
            original = historical[number]
            if (original.input_sha256 != frozen["input_sha256"] or original.target_sha256 != frozen["target_sha256"]
                    or original.context_sha256 != frozen["context_sha256"]
                    or list(original.target_ids) != frozen["target_record_ids"]
                    or len(original.input_bytes) != frozen["input_bytes"]):
                raise ValueError("utility cohort historical input differs from its original frozen capture")
        if scope is not None and historical[number] is not None and historical[number] != scope["input"]:
            raise ValueError("utility cohort duplicated historical input differs from original materialized capture")
        interval = acknowledgement["collection_io"]
        if (interval["clock_epoch"] != trial["cost"]["clock_epoch"]
                or interval["started_tick_ns"] < trial["cost"]["physical_pin_zero_tick"]):
            raise ValueError("utility cohort durable I/O did not follow its own original physical boundary")
        if trial["comparable"]:
            cost = trial["cost"]
            unknown_counters = ("nn_input_count_unknown", "accepted_nn_input_count_unknown",
                "consumed_nn_input_count_unknown", "rejected_events")
            if (profile["input"]["usage"] != "collection_candidate" or not trial["resolved"] or scope is None
                    or not cost["complete_scope"] or not scope["witness_bodies_complete"]
                    or not scope["witness_basis_complete"]
                    or not scope["resource_known"] or not physical[number]["cache_known"]
                    or not physical[number]["fresh_product"]
                    or cost["physical_pin_zero_tick"] > cost["cohort_cleanup_cutoff_tick"]
                    or any(cost["final_counters"][name] != cost["baseline_counters"][name] for name in unknown_counters)):
                raise ValueError("utility cohort comparable trial lacks its original complete known scope")
    final_interval = _io(files.durable_ack()["collection_io"])
    last_interval = acknowledgements[-1]["collection_io"]
    if (final_interval["clock_epoch"] != trials[-1]["cost"]["clock_epoch"]
            or final_interval["started_tick_ns"] < last_interval["finished_tick_ns"]):
        raise ValueError("utility cohort final durable I/O did not follow its last original trial ACK")
    authorities = [tuple(physical[number]["authority"][name] for name in (
        "process_epoch", "game_generation", "root_generation", "implementation_sha256")) for number in physical]
    if len(set(authorities)) != len(authorities):
        raise ValueError("utility cohort trials reused an original authority")
    if physical[1]["authority"] != registration_authority:
        raise ValueError("utility cohort first trial changed original registration authority")
    known, known_sha, all_sha = _pair_candidates(files, trials, scopes, physical)
    canonical_number = frozen["first_trial_sequence"]
    captured = scopes[canonical_number]["input"] if canonical_number in scopes else historical.get(canonical_number)
    if captured is None:
        raise UnsupportedCohortPayload("cohort_original_frozen_capture_not_available")
    if (captured.input_sha256 != frozen["input_sha256"] or captured.target_sha256 != frozen["target_sha256"]
            or captured.context_sha256 != frozen["context_sha256"] or list(captured.target_ids) != frozen["target_record_ids"]
            or len(captured.input_bytes) != frozen["input_bytes"]):
        raise ValueError("utility cohort report input differs from original frozen capture")
    expected_epoch = hashlib.sha256(b"rz-rovent-utility-cohort-cost-epoch/1\0"
        + bytes.fromhex(files.publication()["cohort_id"]) + bytes.fromhex(cohort_profile_sha256(profile))
        + bytes.fromhex(first_prelude["ready_group_sha256"])
        + captured.input_sha256.encode("ascii") + captured.target_sha256.encode("ascii")
        + COST_SCOPE.encode("ascii")).hexdigest()
    if expected_epoch != checked["registration"]["cost_epoch_sha256"]:
        raise ValueError("utility cohort original frozen capture/cost epoch differs from preparation/2")
    # A later view has its own submission target. Preserve the original first
    # frozen capture's input string and three seals; never reserialize it.
    metadata = source["collection_metadata"]
    manifest = source["collection_manifest"]
    ancestors = source["ancestor_datasets"]
    report = {"cohort_id": files.publication()["cohort_id"], "profile_sha256": cohort_profile_sha256(profile),
        "usage": profile["input"]["usage"], "collection_source_sha256": source["source_pins"]["source_sha256"],
        "cost_epoch_sha256": checked["registration"]["cost_epoch_sha256"],
        "resource_profile_sha256": source["source_pins"]["resource_plan_sha256"],
        "frozen_base_weights_sha256": captured.base_weights_sha256, "input_reference_json": captured.input_json,
        "input_sha256": captured.input_sha256, "target_sha256": captured.target_sha256,
        "submission_target_sha256": captured.submission_target_sha256, "utility_context_sha256": captured.context_sha256,
        "known_pairs": known, "known_pair_sha256": known_sha, "all_pair_sha256": all_sha,
        "frozen_pc_sources": frozen_pc_sources, "split": manifest["opening_splits"][metadata["opening_id"]],
        "opening_id": metadata["opening_id"], "game_id": metadata["game_id"], "line_genealogy_id": metadata["line_genealogy_id"],
        "transposition_sha256": source["transposition_sha256"], "rules_state_sha256": source["rules_root"]["anchor"]["rules_state_sha256"],
        "ancestor_dataset_publication_sha256": sorted(item["publication_sha256"] for item in ancestors),
        "ancestor_datasets": copy.deepcopy(ancestors)}
    return {"input": captured, "report": report}
