"""Strict cohort profile/ACK observations, never a fabricated trial/target.

The file owner below comes from actual tiny temporary-file pin readback. All
payloads remain synthetic test data; historical source/witness/final closure
metadata cannot issue live authority merely by recomputing hashes.
Profile, native byte framing and typed ACK expectations follow the Rust source
independently of the new payload reader's helpers. No model is initialized.
"""
import copy
from contextlib import contextmanager
import hashlib
import importlib.abc
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime", "tensorflow", "jax"}
                or fullname in {"rz_pals_model.model", "rz_pals_model.training", "rz_pals_model.nonzero_training",
                    "rz_pals_model.nnuec_trainable", "rz_pals_model.nnuec_utility_trainable"}):
            raise AssertionError("cohort payload tests cannot load a model or tensor runtime")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_utility_cohort_payload as reader
    from rz_pals_model import rovent_utility_cohort_files as files_reader
    from rz_pals_model import rovent_utility_training as training
    from test_rovent_utility_cohort_files import publication_fixture, write_fixture, file_pin, payloads, raw, budget
    from test_rovent_utility_cohort_training import fixture as preparation_fixture
    from test_rovent_utility_root_local import action_fixture as common_action_fixture, origin_seal, source_aggregate, view_wire
    from test_rovent_collection_teacher import submitted_fixture, seal_descriptor
    from test_nnuec_v2_reference import checkpoint_mapping as small_weights_mapping
finally:
    sys.meta_path.remove(_guard)


PROFILE_DOMAIN = "rz-rovent-utility-comparison-cohort/1"
TRIAL_DOMAIN = "rz-rovent-utility-cohort-trial/1"
COHORT_SCOPE = "cohort_route_with_bounded_materialization_and_physical_pin_zero_excluding_durable_dataset_io_v1"
PUBLICATION_FILE = "utility-cohort-publication.json"
ACK_SCHEMA = "rz-rovent-utility-cohort-durable-ack/1"
PROFILE_INPUT_FIELDS = ("input_lock_sha256", "source_sha256", "frozen_nnuec_model_sha256", "frozen_nnuec_namespace_sha256",
    "frozen_pc_model_sha256", "frozen_pc_namespace_sha256", "mode", "requested_recipients", "prelude_ns", "action_ns",
    "cleanup_ns", "logical_budget_sha256", "ready_policy_sha256", "cache_policy_sha256", "cost_class_sha256",
    "resource_plan_sha256", "resource_epoch", "usage", "cost_scope")


def h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def native_bytes_sha(domain, body):
    # UCI trial/materialization/pair bytes use a distinct length-framed ABI.
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + len(body).to_bytes(8, "little") + body).hexdigest()


def preparation():
    recipe, registration, spec = preparation_fixture()
    return training.prepare_cohort_training(recipe, registration, spec)


def profile_fixture(prepared):
    source = prepared["registration"]
    return {"domain": PROFILE_DOMAIN, "input": {
        "input_lock_sha256": h("synthetic original startup lock"), "source_sha256": source["collection_source_sha256"],
        "frozen_nnuec_model_sha256": source["frozen_base"]["weights_sha256"], "frozen_nnuec_namespace_sha256": h("synthetic NNUEC namespace"),
        "frozen_pc_model_sha256": h("synthetic P/C model"), "frozen_pc_namespace_sha256": h("synthetic P/C namespace"),
        "mode": "cpu_pc", "requested_recipients": ["proposer", "critic"], "prelude_ns": 1000000,
        "action_ns": 2000000, "cleanup_ns": 3000000, "logical_budget_sha256": h("synthetic logical budget"),
        "ready_policy_sha256": h("synthetic ready policy"), "cache_policy_sha256": h("synthetic cache policy"),
        "cost_class_sha256": h("synthetic cohort cost class"), "resource_plan_sha256": source["resource_profile_sha256"],
        "resource_epoch": 1, "usage": "collection_candidate", "cost_scope": COHORT_SCOPE}}


def typed_ack(publication, publication_body, blobs):
    """Trial ACK/4, collection ACK/6 and IO/8, never a live FD ACK."""
    publication_pin = file_pin(PUBLICATION_FILE, publication_body)
    checkpoints = []
    for ordinal, trial in enumerate(publication["trials"]):
        materialized = next((pin["bytes"] for pin in publication["materializations"]
            if pin["path"] == f"materializations/{ordinal:06d}.json"), 0)
        # Each independently launched trial owns its own ProcessClock. These
        # numeric ranges deliberately reverse/overlap and can exceed the final
        # publisher's ticks; only an interval's own elapsed is comparable here.
        start = (150, 0, 151)[ordinal]
        checkpoints.append({"trial_sequence": ordinal + 1,
            "native_trial_sha256": native_bytes_sha(TRIAL_DOMAIN, blobs[trial["path"]]), "checkpoint": copy.deepcopy(trial),
            "collection_io": {"clock_epoch": 8 + ordinal, "started_tick_ns": start, "finished_tick_ns": start + 2,
                "elapsed_ns": 2, "written_bytes": materialized + trial["bytes"],
                "readback_bytes": materialized + 2 * trial["bytes"], "synced_files": 2 + int(materialized > 0),
                "synced_directories": 3 * (1 + int(materialized > 0))}})
    readback = publication["total_payload_bytes"] + len(publication_body)
    earlier = sum(pin["bytes"] for pin in (*publication["trials"], *publication["materializations"]))
    return {"schema": ACK_SCHEMA, "cohort_id": publication["cohort_id"], "cost_scope": COHORT_SCOPE,
        "publication": publication_pin, "collection_io": {"clock_epoch": 7, "started_tick_ns": 100,
            "finished_tick_ns": 103, "elapsed_ns": 3, "written_bytes": readback - earlier, "readback_bytes": readback,
            "synced_files": len(payloads(publication)) + 1, "synced_directories": 4},
        "trial_acknowledgements": checkpoints}


@contextmanager
def pinned_files(prepared, *, change_profile=None, change_ack=None,
                 change_trial_body=None, change_trial_raw=None, trial_count=2,
                 materialization_ordinals=(0, 1), pair_count=1):
    """Integrity-valid files with intentionally incomplete historical bodies.

    The minimal trial dictionaries expose only schema/cohort/sequence for the
    ACK-specific parser. All remaining bodies are opaque fixture text. None
    represent a resolved action, witness, materialization or final fence.
    """
    publication, blobs = publication_fixture(trial_count=trial_count,
        materialization_ordinals=materialization_ordinals, pair_count=pair_count)
    profile = profile_fixture(prepared)
    if change_profile is not None:
        change_profile(profile)
    blobs[publication["cohort_profile"]["path"]] = raw(profile)
    for ordinal, pin in enumerate(publication["trials"]):
        body = {"schema": TRIAL_DOMAIN, "cohort_id": publication["cohort_id"], "trial_sequence": ordinal + 1}
        if change_trial_body is not None:
            change_trial_body(body, ordinal)
        original = raw(body)
        if change_trial_raw is not None:
            original = change_trial_raw(original, ordinal)
        blobs[pin["path"]] = original
    for pin in payloads(publication):
        pin.update(file_pin(pin["path"], blobs[pin["path"]]))
    publication["total_payload_bytes"] = sum(pin["bytes"] for pin in payloads(publication))
    publication_body = raw(publication)
    ack = typed_ack(publication, publication_body, blobs)
    if change_ack is not None:
        change_ack(ack)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        reference, _publication, _ack, _blobs = write_fixture(root, publication=publication, blobs=blobs,
            publication_body=publication_body, ack=ack, ack_body=raw(ack))
        owner = files_reader.read_utility_cohort_files(reference, read_accounting=budget())
        yield owner, profile, ack


# Full bodies below are synthetic original-wire test operands. Even a complete
# comparable metadata record is not a real Product, Rules replay, conditional
# certificate, physical fence or permission to update the utility learner.
BASE_PARAMETERS = ("board_weights", "board_bias", "anchor_weights", "anchor_bias", "from_weights", "to_weights",
    "promotion_weights", "ply_weights", "token_bias", "causal_weights", "causal_bias", "context_projection",
    "context_bias", "local_projection", "local_bias", "hidden_weights", "hidden_bias", "output_weights", "output_bias")
RELATION_PARAMETERS = ("kind_embeddings", "pair_parent", "pair_child", "pair_bias", "pair_projection")
DIMENSIONS = ("board_width", "line_width", "context_width", "local_width", "hidden_width", "causal_window", "max_lines", "max_plies")
FULL_COUNTERS = ("native_preparations", "submitted_role_tickets", "physical_role_completions", "completed_nn_inputs",
    "nn_input_count_unknown", "consumed_role_outputs", "accepted_nn_inputs", "accepted_nn_input_count_unknown",
    "consumed_nn_inputs", "consumed_nn_input_count_unknown", "zero_nn_outputs_consumed", "unused_role_results",
    "canceled_preparations", "cpu_slice_receipts", "cpu_progress_publications", "cpu_yields", "cpu_yield_requests",
    "cpu_yield_acknowledgements", "cpu_natural_completions", "validator_ticket_admissions", "foreign_reports",
    "foreign_reported_nodes", "foreign_node_count_unknown", "foreign_yield_requests", "foreign_yield_acknowledgements",
    "rejected_events", "notification_waits")
REPORT_FIELDS = ("cohort_id", "profile_sha256", "usage", "collection_source_sha256", "cost_epoch_sha256",
    "resource_profile_sha256", "frozen_base_weights_sha256", "input_reference_json", "input_sha256", "target_sha256",
    "submission_target_sha256", "utility_context_sha256", "known_pairs", "known_pair_sha256", "all_pair_sha256",
    "frozen_pc_sources", "split", "opening_id", "game_id", "line_genealogy_id", "transposition_sha256",
    "rules_state_sha256", "ancestor_dataset_publication_sha256", "ancestor_datasets")


def canonical_body(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def plain_digest(domain, value):
    return hashlib.sha256(canonical_body([domain, value])).hexdigest()


def framed_value(domain, value):
    return native_bytes_sha(domain, canonical_body(value))


def independent_semantic_weights(weights):
    """The Rust V1/V2 binary operand frames, without creating a model."""
    base = weights["base"]
    original = hashlib.sha256((base["schema"] + "\0" + base["semantics"] + "\0").encode("ascii"))
    for name in DIMENSIONS:
        original.update(base["config"][name].to_bytes(4, "little"))
    for name in BASE_PARAMETERS:
        for operand in base["parameters"][name]:
            original.update(struct.pack("<f", operand))
    original.update(struct.pack("<f", base["output_scale"]))
    relation = hashlib.sha256((weights["schema"] + "\0" + weights["semantics"] + "\0"
        + original.hexdigest() + "\0").encode("ascii"))
    for name in RELATION_PARAMETERS:
        for operand in weights["parameters"][name]:
            relation.update(struct.pack("<f", operand))
    return relation.hexdigest()


def wire_artifact(path, body):
    return {**file_pin(path, body), "source": "synthetic parser fixture; no loaded source owner", "license": "fixture-only"}


def seal_fixture_submitted_view(value):
    """Independent original producer frame for a NEW synthetic submission."""
    provenance = value["provenance"]
    frame = b"rz-pals-review-view-seal/1\0" + bytes.fromhex(provenance["full_view_sha256"])
    frame += b"".join(provenance[key].to_bytes(8, "little") for key in ("sequence", "generation", "root_state"))
    for record in value["records_metadata"]:
        frame += record["source_revision"].to_bytes(8, "little")
        frame += record["source_state"].to_bytes(8, "little")
        frame += record["source_line_offset"].to_bytes(2, "little")
    provenance["view_seal_sha256"] = hashlib.sha256(frame).hexdigest()
    value["submission_target_sha256"] = hashlib.sha256(b"rz-nnuec-submitted-utility-submission-target/1\0"
        + bytes.fromhex(provenance["view_seal_sha256"]) + value["target_sha256"].encode("ascii")).hexdigest()


def full_cpu_plan():
    # Independently declared full CPU_PC plan; no external fixture's partial
    # lock/resources object is treated as a valid full execution manifest.
    return {"cpu_quota_millicpus": 2000, "pools": [{"actors": ["manager", "pc_caller"],
        "max_threads": 2, "cpu_charge_millicpus": 2000}], "memory_high_bytes": 6 << 30,
        "memory_max_bytes": 12 << 30, "gpu_devices": 0, "physical_nn_concurrency": 1,
        "topology_policy": "SharedControl"}


def thread_observations(known):
    def thread(actor):
        return {"actor": actor, "thread_id_present": True, "affinity": [[0, 0]] if known else None,
            "probe_method": "LinuxProcThreadStatus" if known else None,
            "unknown_reason": None if known else "ReadUnavailable"}
    return {"native": thread("NativeCaller"), "own": [thread("OwnWorker(0)")]}


def resource_custody(plan, known, ordinal):
    def observed(value):
        return {"status": "observed", "value": value, "method": "synthetic strict wire readback"} if known else {"status": "unknown"}
    resource = {"limits_applied": observed(True), "process_tree_quota_millicpus": observed(2000),
        "memory_high_bytes": observed(plan["memory_high_bytes"]), "memory_max_bytes": observed(plan["memory_max_bytes"]),
        "inventory_complete": observed(True), "kernel_threads": observed(2),
        "verified_plan_readback_sha256": observed(plain_digest("rz-rovent-resource-plan/1", plan)),
        "host_peak_bytes": observed(4096), "vram_peak_bytes": {"status": "unknown"},
        "evidence": wire_artifact(f"evidence/trial-{ordinal}.json", b"synthetic observed resource bytes")}
    return {"schema": "rz-rovent-resource-custody/1", "plan_pin": file_pin("inputs/resource-plan.json", raw(plan)),
        "evidence_directory": "evidence", "profile": "functional_validation", "admission_attempted": True,
        "finished_attempted": True, "original_plan_descriptor_retained": True,
        "original_delegated_descriptor_retained": True, "original_scope_retained": True, "resource_failure": None,
        "admission": copy.deepcopy(resource), "finished_live": copy.deepcopy(resource), "finished_live_error": None,
        "controlled_comparison_accepted": False, "automatic_cleanup": False}


def physical_wire(root, inp):
    authority = {name: root[name] for name in ("process_epoch", "game_generation", "root_generation", "implementation_sha256")}
    binding = h("synthetic original released NNUEC binding")
    identity = copy.deepcopy(inp["base_identity"])
    return {"root_fence_receipt": {"authority": authority, "native_owner_joined": True, "native_owner_quiesced": True,
        "native_physical_shutdown_confirmed": True, "own_cpu_joined": 1, "expected_own_cpu_owners": 1,
        "own_cpu_panicked": 0, "external_shutdown_confirmed": None, "external_owner_quiesced": None,
        "auxiliary_shutdown_confirmed": True, "native_cpu_spent": 0, "native_cpu_reserved": 0,
        "pending_tickets": 0, "quarantined": False, "request_sequence_high_water": 1, "permits_semantic_closure": True},
        "review_pin_count": 0, "engine_release": {"owner": 0, "actual_stack_owner_id": root["process_epoch"],
            "profile": {"owner": 0, "max_depth": 8, "evaluation_namespace_sha256": binding,
                "search_conditions_sha256": h("synthetic original CPU search conditions")},
            "binding_sha256": binding, "model_identity": identity, "preparation_epoch": root["process_epoch"],
            "release_duration_ns": "20", "provenance": {"root": root["store_root_state"],
                "generation": root["root_generation"], "sequence": inp["provenance"]["sequence"],
                **{name: inp["provenance"][name] for name in ("full_view_sha256", "view_seal_sha256", "record_scope_sha256", "policy_sha256")},
                "model": copy.deepcopy(identity)}},
        "context_cache_release": {"epoch": root["process_epoch"], "matches_engine_release": True},
        "input_cache_prelude": {"epoch": root["process_epoch"], "matches_input": True},
        "native_product": {"original_lifetime_present": True, "first_product_root": True}}


def full_cohort_case(*, resource_known=True, materialized=True, change_trial=None):
    """Build closed synthetic data, with zero new witnesses and known pairs.

    All expected seals are independent fixture framing. A successful parser
    result remains a candidate; no original Root/Product/certificate exists.
    """
    include_materializations = materialized
    weights = small_weights_mapping()
    semantic_weights = independent_semantic_weights(weights)
    weights_raw = raw(weights)
    weights_artifact = wire_artifact("inputs/nnuec-weights.json", weights_raw)
    originals = [common_action_fixture(route) for route in ("route_p", "route_c")]
    manifest, record = submitted_fixture()
    registration = copy.deepcopy(originals[0]["source"]["nnuec_registration"])
    registration["model_identity"]["weights_sha256"] = semantic_weights
    identity = copy.deepcopy(registration["model_identity"])
    nnuec_pin = {name: registration[name] for name in ("schema", "semantics", "encoding", "implementation_sha256", "configuration_sha256",
        "context_namespace_policy_sha256", "local_namespace_policy_sha256")}
    nnuec_pin.update(weights=weights_artifact, diagnostic=False, search_conditions_sha256=h("synthetic source search conditions"))
    provenance = {"domain": "rz-rovent-nnuec-provenance/1", "weights_raw_sha256": weights_artifact["sha256"],
        "semantic_weights_sha256": semantic_weights, **{name: nnuec_pin[name] for name in ("schema", "semantics", "encoding",
            "implementation_sha256", "configuration_sha256", "context_namespace_policy_sha256", "local_namespace_policy_sha256")},
        "training": {"kind": "untrained"}, "diagnostic": False, "provenance": []}
    provenance_raw = raw(provenance)
    provenance_artifact = wire_artifact("inputs/nnuec-provenance.json", provenance_raw)
    nnuec_pin["provenance"] = provenance_artifact
    origin = copy.deepcopy(originals[0]["source"]["root_admission"]["native_origin"])
    components = {"synthetic-cohort-wire": h("independent synthetic full cohort source component")}
    aggregate = source_aggregate(manifest["source_commit"], components)
    pc_artifact = wire_artifact("inputs/frozen-pc.bin", b"synthetic PC checkpoint metadata only; never loaded")
    pc_provenance = wire_artifact("inputs/frozen-pc-provenance.json", b"synthetic PC provenance metadata")
    model_pin = {"semantics": origin["model_semantics"], "encoding": origin["encoding"], "policy_head": "legal_candidates",
        "value_head": "wdl", "implementation_sha256": origin["model_implementation_sha256"],
        "weights": pc_artifact, "provenance": pc_provenance, "diagnostic": False}
    pc = {"proposer": copy.deepcopy(model_pin), "critic": copy.deepcopy(model_pin), "validator": None,
        "backend": "rust_cpu", "device": {"device": "cpu"}, "backend_configuration_sha256": h("synthetic Rust CPU backend"),
        "runtime_artifacts": [], "fp32": True, "tf32": False, "batch_width": 1,
        "ort_intra_threads": None, "ort_inter_threads": None, "ort_spinning": None}
    plan = full_cpu_plan()
    search = {"node_unit": "rovent-position-visits/1", "requested_nodes": 4096, "depth_plies": None, "mate_moves": None,
        "registered_max_chess_plies": 256, "searchmoves": None, "search_configuration_sha256": h("search configuration"),
        "root_state_and_history_sha256": originals[0]["source"]["root_admission"]["root_state"]["semantic"],
        "rules_profile_sha256": h("Rules profile")}
    coordinator = {"policy_sha256": h("coordinator"), "questions_max": 64, "ancestor_depth_max": 8,
        "waiting_jobs_max": 64, "role_tickets_max": 16, "cpu_credit_limit": 4096,
        "original_root_budget_sha256": h("registered startup root budget"),
        "original_deadline_seal_sha256": h("registered startup deadline")}
    execution = {"domain": "rz-rovent-execution/1", "contract_revision": "0.1", "run_id": "synthetic-complete-cohort-wire",
        "purpose": "collection", "engine": {"source_commit": manifest["source_commit"], "source_components": components,
            "binary": wire_artifact("inputs/rovent", b"synthetic engine metadata; never executed"), "mode": "cpu_pc", "pc": pc,
            "researcher": None, "teachers": [], "nnuec": nnuec_pin, "search": search},
        "resources": plan, "coordinator": coordinator, "wall_time_max_ms": 1000, "cleanup_max_ms": 500,
        "output_bytes_max": 1048576, "agent_renewable": True,
        "provenance": [wire_artifact("inputs/source.txt", b"synthetic original source metadata")]}
    lock = {"domain": "rz-rovent-input-lock/1", "manifest": execution,
        "canonical_sha256": plain_digest("rz-rovent-execution-canonical/1", execution)}
    origin.update(mode="CpuPc", input_lock_sha256=lock["canonical_sha256"], source_commit=manifest["source_commit"],
        source_components=components, source_sha256=aggregate, weights_sha256=pc_artifact["sha256"], teacher_profiles=[])
    roots, inputs, accepted, views = [], [], [], []
    for ordinal, original in enumerate(originals):
        epoch = ordinal + 1
        root = copy.deepcopy(original["source"]["root_admission"])
        root.update(process_epoch=epoch, deadline_clock=epoch, native_origin={**copy.deepcopy(origin), "process_epoch": epoch})
        descriptor = root["root_descriptor"]
        declared = descriptor["input"]
        declared.update(startup_lock_sha256=lock["canonical_sha256"])
        declared["authority"]["epoch"] = epoch
        declared["declared_ceilings"]["resource_plan"] = copy.deepcopy(plan)
        for deadline in declared["clock"].values():
            deadline["clock_epoch"] = epoch
        root["root_descriptor_sha256"] = seal_descriptor(descriptor)
        root["root_budget_sha256"] = descriptor["root_budget_sha256"]
        root["native_origin_sha256"] = plain_digest("rz-rovent-collection-native-origin/1", root["native_origin"])
        inp = copy.deepcopy(original["input"])
        inp["base_identity"] = copy.deepcopy(identity)
        inp["provenance"]["model"] = copy.deepcopy(identity)
        original_raw = inp["reference_json"].encode("utf-8")
        inp["input_sha256"] = hashlib.sha256(b"rz-nnuec-submitted-utility-input/1\0" + semantic_weights.encode("ascii")
            + b"\0" + len(original_raw).to_bytes(8, "little") + original_raw).hexdigest()
        for rules in [inp["root_rules"], *inp["source_rules"]]:
            rules.update(observer_revision=1, uci_replay={"origin": "fen", "completeness": "unknown_prefix",
                "start_fen": rules["fen"], "moves": []})
        legacy_origin = copy.deepcopy(original["semantic"]["pc_consumption"][0]["origin"])
        legacy = legacy_origin["ticket"]
        # The root-local fixture deliberately gives its capture a separate
        # policy label. This new cohort fixture seals one immutable Request:
        # its materialized capture and original selected Ready view must name
        # that same policy before either the trial or source is registered.
        inp["provenance"]["policy_sha256"] = legacy["context_policy"]
        seal_fixture_submitted_view(inp)
        legacy.update(process_epoch=epoch, deadline_clock=epoch, question_digest=h("synthetic cohort question " + str(ordinal)),
            view_digest=inp["provenance"]["view_seal_sha256"], view_sequence=inp["provenance"]["sequence"])
        legacy["representation"].update(model=pc_artifact["sha256"], encoding=origin["encoding_sha256"], frozen_epoch=origin["frozen_epoch"])
        legacy_origin["frame_sequence"] = 10 + ordinal
        legacy_origin["seal_sha256"] = origin_seal(legacy_origin)
        authority = {name: root[name] for name in ("process_epoch", "game_generation", "root_generation", "implementation_sha256")}
        view = view_wire(legacy)
        ticket = {"authority": authority, "request_epoch": epoch, "request_sequence": legacy["request_sequence"],
            "role": "Proposer" if ordinal == 0 else "Critic", "question_sha256": legacy["question_digest"],
            "view": view, "representation": copy.deepcopy(legacy["representation"]),
            "deadline_clock": epoch, "deadline_tick": root["deadline_tick"]}
        accepted.append({**legacy_origin, "ticket": ticket})
        roots.append(root)
        inputs.append(inp)
        views.append(view)
    manifest.update(source_sha256=aggregate, source_commit=execution["engine"]["source_commit"], teachers={})
    source_pins = {"input_lock_sha256": lock["canonical_sha256"], "source_sha256": aggregate,
        "frozen_nnuec_model_sha256": semantic_weights,
        "frozen_nnuec_namespace_sha256": framed_value("rz-rovent-utility-cohort-nnuec-namespace/1", registration),
        "frozen_pc_model_sha256": origin["weights_sha256"],
        "frozen_pc_namespace_sha256": framed_value("rz-rovent-utility-cohort-pc-namespace/1", {name: origin[name] for name in (
            "model_semantics", "model_semantics_sha256", "encoding", "encoding_sha256", "encoding_registry_sha256",
            "model_implementation_sha256", "representation_model_domain", "frozen_epoch", "weights_sha256", "export_manifest_sha256", "backend_sha256")}),
        "resource_plan_sha256": plain_digest("rz-rovent-resource-plan/1", plan),
        "logical_budget_sha256": framed_value("rz-rovent-utility-cohort-logical-budget/1", roots[0]["root_descriptor"]["input"]["declared_ceilings"])}
    profile = {"domain": PROFILE_DOMAIN, "input": {**source_pins, "mode": "cpu_pc", "requested_recipients": ["proposer", "critic"],
        "prelude_ns": 10, "action_ns": 200, "cleanup_ns": 100, "ready_policy_sha256": h("synthetic fixed ready policy"),
        "cache_policy_sha256": h("synthetic fresh cache policy"), "cost_class_sha256": h("synthetic cohort cost class"),
        "resource_epoch": 1, "usage": "collection_candidate", "cost_scope": COHORT_SCOPE}}
    profile_digest = hashlib.sha256(PROFILE_DOMAIN.encode("ascii") + b"\0" + canonical_body(profile)).hexdigest()
    before = dict.fromkeys(FULL_COUNTERS, 0)
    after = dict(before)
    after.update(native_preparations=1, submitted_role_tickets=1, physical_role_completions=1,
        completed_nn_inputs=1, consumed_role_outputs=1, accepted_nn_inputs=1, consumed_nn_inputs=1)
    common = {"root_entries": 1, "checked_transitions": 0, "native_nodes": 0, "native_qsearch_nodes": 0,
        "reserved_native_nodes": 0, "outstanding_native_executions": 0, "native_usage_known": True}
    observations = thread_observations(resource_known)
    semantic_offers = [h("synthetic original semantic offer " + role) for role in ("proposer", "critic")]
    group = b"rz-rovent-utility-cohort-ready-offer-group/1\0" + (2).to_bytes(8, "little")
    for index, digest in enumerate(semantic_offers):
        group += bytes([4 + index]) + bytes.fromhex(digest)
    group += bytes.fromhex(framed_value("rz-rovent-utility-cohort-common-node-work/1", {"requested_nodes": 4096, "work": common}))
    group += b"".join(item.to_bytes(8, "little") for item in (4096, 0, 0, 0, 0))
    group += bytes.fromhex(framed_value("rz-rovent-utility-cohort-prelude-counters/1",
        {name: value for name, value in before.items() if name != "notification_waits"}))
    group += bytes.fromhex(framed_value("rz-rovent-utility-cohort-actual-resource-observations/1", observations))
    ready_group = hashlib.sha256(group).hexdigest()
    cohort_id = hashlib.sha256(b"rz-rovent-utility-cohort-owner/1\0" + bytes.fromhex(profile_digest)
        + bytes.fromhex(roots[0]["root_descriptor_sha256"]) + bytes.fromhex(ready_group) + bytes([4, 5])).hexdigest()
    first = inputs[0]
    frozen = {"input_sha256": first["input_sha256"], "target_sha256": first["target_sha256"], "context_sha256": first["context_sha256"],
        "target_record_ids": [0, 1], "input_bytes": first["bytes"], "root_rules_sha256": first["root_rules"]["semantic_sha256"],
        "original_first_move": 64, "first_trial_sequence": 1}
    cost_epoch = hashlib.sha256(b"rz-rovent-utility-cohort-cost-epoch/1\0" + bytes.fromhex(cohort_id) + bytes.fromhex(profile_digest)
        + bytes.fromhex(ready_group) + first["input_sha256"].encode("ascii") + first["target_sha256"].encode("ascii")
        + COHORT_SCOPE.encode("ascii")).hexdigest()
    source = {"schema": "rz-rovent-utility-cohort-collection-source/1", "source_pins": source_pins,
        "nnuec_registration": registration, "nnuec_weights_artifact": weights_artifact, "nnuec_provenance_artifact": provenance_artifact,
        "native_origin": copy.deepcopy(roots[0]["native_origin"]), "registration_root": copy.deepcopy(roots[0]),
        "collection_manifest": manifest, "collection_metadata": {"game_id": record["input"]["game_id"],
            "opening_id": record["input"]["opening_id"], "line_genealogy_id": record["input"]["line_genealogy_id"],
            "lineage": {"actor_id": "synthetic-cohort-student", "teacher_actor_id": None, "ancestor_origin_sha256": []}},
        "rules_root": {"schema": "rz-rovent-rules-position-replay/1", "origin": "fen", "completeness": "unknown_prefix",
            "start_fen": first["root_rules"]["fen"], "moves": [], "observer_revision": 1,
            "anchor": copy.deepcopy(record["input"]["rules_anchor"])},
        "transposition_sha256": h("synthetic transposition metadata; Rust replay still required"),
        "frozen_pc_sources": [{"role": role, "artifact": copy.deepcopy(pc_artifact),
            "model_semantics_sha256": origin["model_semantics_sha256"], "model_identity_sha256": origin["weights_sha256"],
            "encoding_sha256": origin["encoding_sha256"]} for role in ("proposer", "critic")],
        "ancestor_datasets": [], "resource_observation": [resource_custody(plan, resource_known, i) for i in range(2)]}
    blobs = {"cohort-profile.json": raw(profile), "frozen-context.json": raw(frozen), "source.json": raw(source),
        "input-lock.json": raw(lock), "nnuec-weights.json": weights_raw, "nnuec-provenance.json": provenance_raw}
    trials = []
    for index, (root, inp, accepted_origin, view) in enumerate(zip(roots, inputs, accepted, views, strict=True)):
        epoch = index + 1
        authority = copy.deepcopy(accepted_origin["ticket"]["authority"])
        offers = [{"frame_sequence": 10 + role_index, "recipient": route, "stage": stage, "original_first_move": 64,
            "question_sha256": h("synthetic cohort question " + str(role_index)), "view": copy.deepcopy(view),
            "full_view_sha256": inp["provenance"]["full_view_sha256"], "public_context_sha256": inp["context_sha256"],
            "semantic_sha256": semantic_offers[role_index], "budget": {"role_submissions_remaining": 4,
                "original_cpu_credit": 4096, "original_store_byte_limit": 1048576},
            "deadline_clock": epoch, "deadline_tick": root["deadline_tick"]}
            for role_index, (route, stage) in enumerate((("route_p", "ProposerRanking"), ("route_c", "CriticReply")))]
        materialized = {"schema": "rz-rovent-utility-cohort-materialization/1", "cohort_id": cohort_id, "trial_sequence": epoch,
            "recipient": ("route_p", "route_c")[index], "profile_sha256": profile_digest, "source": source_pins,
            "original_root": root, "prelude": {"captured_tick": 50, "native_credit": {"limit": 4096, "spent": 0, "reserved": 0,
                "highest_job_sequence": 0, "highest_execution_sequence": 0}, "role_credit": {"spent": 0, "reserved": 0},
                "counters": before, "offers": offers, "requested_common_nodes": 4096, "common_node_work": common,
                "ready_group_sha256": ready_group, "actual_resource_observations": copy.deepcopy(observations),
                "native_thread_readback": "synthetic original NativeCaller wire", "own_thread_readbacks": "synthetic original OwnWorker wire"},
            "input": inp, "action": {"authority": authority, "frame_sequence": 10 + index,
                "question_sha256": accepted_origin["ticket"]["question_sha256"], "root_budget_sha256": root["root_budget_sha256"],
                "original_first_move": 64, "deadline_clock": epoch, "deadline_tick": root["deadline_tick"],
                "frozen_context_sha256": h("synthetic original common frozen semantic scope"), "cost_epoch_sha256": cost_epoch,
                "cost_scope": COHORT_SCOPE},
            "semantic": {"baseline_witnesses": [], "new_witnesses": [], "witness_preimages": [], "witness_basis": [],
                "pc_consumption": [{"origin": accepted_origin, "question_sha256": accepted_origin["ticket"]["question_sha256"],
                    "view": copy.deepcopy(view), "full_view_sha256": inp["provenance"]["full_view_sha256"]}],
                "dependencies": [], "r_selected_pc_consumption": [], "native_executions": []}}
        original_materialization = raw(materialized)
        if include_materializations:
            blobs[f"materializations/{index:06d}.json"] = original_materialization
        previous = None if index == 0 else {"trial_sequence": 1,
            "trial_sha256": native_bytes_sha(TRIAL_DOMAIN, blobs["trials/000000.json"]),
            "file_sha256": hashlib.sha256(blobs["trials/000000.json"]).hexdigest()}
        cost = {"domain": "rz-rovent-utility-cohort-cost/1", "scope": COHORT_SCOPE, "objective": "exclusive_wall_nanoseconds",
            "cost_class_sha256": profile["input"]["cost_class_sha256"], "cost_epoch_sha256": cost_epoch,
            "resource_plan_sha256": source_pins["resource_plan_sha256"], "resource_epoch": 1, "clock_epoch": epoch,
            "started_tick": 100, "root_fence_tick": 180, "actor_release_tick": 200, "physical_pin_zero_tick": 220,
            "whole_wall_ns": 120, "original_deadline_tick": root["deadline_tick"], "cohort_action_cutoff_tick": 300,
            "cohort_cleanup_cutoff_tick": 400, "baseline_counters": before, "final_counters": after,
            "native_credit_baseline": [0, 0], "native_credit_final": [0, 0], "cpu_thread_time_ns": None,
            "energy": None, "complete_scope": include_materializations}
        historical = None if include_materializations else {
            **{name: inp[name] for name in ("reference_json", "input_sha256", "target_sha256", "submission_target_sha256", "context_sha256", "target_record_ids")},
            "original_view_sha256": inp["provenance"]["view_seal_sha256"],
            "original_full_view_sha256": inp["provenance"]["full_view_sha256"],
            "materialized_inside_complete_scope": False}
        trial = {"schema": TRIAL_DOMAIN, "cohort_id": cohort_id, "trial_sequence": epoch, "recipient": materialized["recipient"],
            "profile": profile, "profile_sha256": profile_digest, "source": source_pins, "frozen_context": frozen,
            "original_root": copy.deepcopy(materialized["original_root"]), "ready_prelude": copy.deepcopy(materialized["prelude"]),
            "materialized_sha256": native_bytes_sha("rz-rovent-utility-cohort-materialization/1", original_materialization) if include_materializations else None,
            "materialized_json": original_materialization.decode("utf-8") if include_materializations else None,
            "historical_unbound_input": historical, "unresolved_source": None,
            "cost": cost, "physical_closure": physical_wire(root, inp), "previous_durable_ack": previous,
            "resolved": include_materializations, "comparable": resource_known and include_materializations,
            "unknown_reason": ("missing_bounded_historical_materialization" if not include_materializations else
                None if resource_known else "unknown_original_resource_observation"),
            "support_mask": [i == 4 + index for i in range(7)], "target_known_mask": [False] * 7}
        if change_trial is not None:
            change_trial(trial, materialized, index)
        if include_materializations:
            # Mutation cases still reach the typed body checker with exact
            # original raw/native/file pins, not an unrelated stale outer SHA.
            original_materialization = raw(materialized)
            blobs[f"materializations/{index:06d}.json"] = original_materialization
            trial["materialized_sha256"] = native_bytes_sha("rz-rovent-utility-cohort-materialization/1", original_materialization)
            trial["materialized_json"] = original_materialization.decode("utf-8")
        blobs[f"trials/{index:06d}.json"] = raw(trial)
        trials.append(trial)
    pair = {"schema": "rz-rovent-utility-cohort-pair/1", "cohort_id": cohort_id, "left_trial_sequence": 1,
        "right_trial_sequence": 2, "left_trial_sha256": native_bytes_sha(TRIAL_DOMAIN, blobs["trials/000000.json"]),
        "right_trial_sha256": native_bytes_sha(TRIAL_DOMAIN, blobs["trials/000001.json"]), "left_recipient": "route_p",
        "right_recipient": "route_c", "preferred": None,
        "unknown_reason": "no_new_valid_witness" if resource_known and include_materializations else "unresolved_or_masked_trial",
        "support_mask": [False, False, False, False, True, True, False], "target_known_mask": [False] * 7,
        "left_new_witnesses": [], "right_new_witnesses": [], "left_whole_wall_ns": 120, "right_whole_wall_ns": 120,
        "cost_scope": COHORT_SCOPE, "objective": "exclusive_wall_nanoseconds"}
    blobs["pairs/000000.json"] = raw(pair)
    publication, _ = publication_fixture(trial_count=2,
        materialization_ordinals=(0, 1) if include_materializations else (), pair_count=1)
    publication["cohort_id"] = cohort_id
    for pin in payloads(publication):
        pin.update(file_pin(pin["path"], blobs[pin["path"]]))
    publication["total_payload_bytes"] = sum(pin["bytes"] for pin in payloads(publication))
    publication_body = raw(publication)
    ack = typed_ack(publication, publication_body, blobs)
    for ordinal, item in enumerate(ack["trial_acknowledgements"]):
        item["collection_io"].update(clock_epoch=ordinal + 1, started_tick_ns=230, finished_tick_ns=232)
    # Only the full entry joins the final publisher to the last trial clock.
    # The independent ACK-only fixtures retain their separate clock epochs.
    final_start = ack["trial_acknowledgements"][-1]["collection_io"]["finished_tick_ns"] + 1
    ack["collection_io"].update(clock_epoch=trials[-1]["cost"]["clock_epoch"],
        started_tick_ns=final_start, finished_tick_ns=final_start + ack["collection_io"]["elapsed_ns"])
    config = copy.deepcopy(weights["base"]["config"])
    spec = {"hidden_width": 2, "input_width": 5,
        "wire_to_parameter": {name: "utility." + name for name in ("hidden_weights", "hidden_bias", "output_weights", "output_bias")},
        "parameters": {name: {"checkpoint_path": "parameters." + name, "shape": shape, "dtype": "float32", "planned_requires_grad": True}
            for name, shape in zip(("hidden_weights", "hidden_bias", "output_weights", "output_bias"), ([2, 5], [2], [7, 2], [7]), strict=True)},
        "optimizer_owner": "synthetic-cohort-utility-owner", "frozen_components": ["nnuec_base", "p", "c"]}
    registered = {"collection_source_sha256": aggregate, "trainer_source_sha256": h("synthetic trainer source"),
        "cost_epoch_sha256": cost_epoch, "resource_profile_sha256": source_pins["resource_plan_sha256"],
        "cost_scope": COHORT_SCOPE, "objective": "exclusive_wall_nanoseconds",
        "frozen_base": {"schema": weights["schema"], "semantics": weights["semantics"], "weights_sha256": semantic_weights, "config": config},
        "utility": {"schema": training.UTILITY_SCHEMA, "semantics": training.UTILITY_SEMANTICS,
            "initial_weights_sha256": h("synthetic utility parameter declaration; no model"), "base_weights_sha256": semantic_weights,
            "parameter_spec_sha256": training.parameter_spec_sha256(spec, config, spec["optimizer_owner"])}}
    prepared = training.prepare_cohort_training(training.make_recipe(spec["optimizer_owner"]), registered, spec)
    expected = {"source": source, "pair": pair, "trials": trials, "profile": profile, "profile_sha256": profile_digest,
        "semantic_weights_sha256": semantic_weights, "weights_raw_sha256": weights_artifact["sha256"],
        "pair_sha256": native_bytes_sha("rz-rovent-utility-cohort-pair/1", blobs["pairs/000000.json"]),
        "input": first, "cost_epoch_sha256": cost_epoch}
    return prepared, publication, blobs, publication_body, ack, expected


@contextmanager
def full_cohort_files(*, resource_known=True, materialized=True, change_trial=None):
    prepared, publication, blobs, publication_body, ack, expected = full_cohort_case(
        resource_known=resource_known, materialized=materialized, change_trial=change_trial)
    with tempfile.TemporaryDirectory() as directory:
        reference, _publication, _ack, _blobs = write_fixture(Path(directory).resolve(), publication=publication, blobs=blobs,
            publication_body=publication_body, ack=ack, ack_body=raw(ack))
        owner = files_reader.read_utility_cohort_files(reference, read_accounting=budget(2 << 20))
        yield owner, prepared, expected


class UtilityCohortPayloadStrictTests(unittest.TestCase):
    def setUp(self):
        sys.meta_path.insert(0, _guard)

    def tearDown(self):
        sys.meta_path.remove(_guard)

    def strict_rejection(self, files, prepared):
        with self.assertRaises(ValueError) as rejected:
            reader.validate_cohort_payloads(files, preparation=prepared)
        self.assertNotIsInstance(rejected.exception, reader.UnsupportedCohortPayload)

    def test_profile_declaration_accepts_exact_cpu_modes_and_detaches_metadata(self):
        for mode, recipients in (("cpu_pc", ["proposer", "critic"]),
                ("cpu_pcr", ["researcher", "critic", "proposer"])):
            value = profile_fixture(preparation())
            value["input"].update(mode=mode, requested_recipients=recipients)
            self.assertEqual(set(value["input"]), set(PROFILE_INPUT_FIELDS))
            with self.subTest(mode=mode):
                checked = reader.validate_cohort_profile(value)
                self.assertEqual(checked, value)
                checked["input"]["requested_recipients"].clear()
                self.assertEqual(value["input"]["requested_recipients"], recipients)
        # Zero is valid for some observed counters, but not declared windows.
        value = profile_fixture(preparation())
        value["input"].update(prelude_ns=1, action_ns=86400000000000 - 2, cleanup_ns=1)
        self.assertEqual(reader.validate_cohort_profile(value), value)

    def test_preparation_is_validated_before_any_files_or_legacy_conversion(self):
        class NoFiles:
            def __getattr__(self, _name):
                raise AssertionError("invalid preparation cannot inspect a payload")
        spec = preparation_fixture()[2]
        from test_rovent_utility_training import registration
        legacy = training.prepare_training(training.make_recipe(spec["optimizer_owner"]), registration(spec), spec)
        for prepared in (None, {}, legacy, {**preparation(), "performed_optimizer_updates": 1}):
            with self.subTest(schema=prepared.get("schema") if type(prepared) is dict else None):
                self.strict_rejection(NoFiles(), prepared)

    def test_profile_seal_is_canonical_domain_nul_without_native_length_or_raw_sha(self):
        profile = profile_fixture(preparation())
        original = raw(profile)
        compact = json.dumps(profile, sort_keys=True, ensure_ascii=False,
            allow_nan=False, separators=(",", ":")).encode("utf-8")
        expected = hashlib.sha256(PROFILE_DOMAIN.encode("ascii") + b"\0" + compact).hexdigest()
        self.assertEqual(reader.cohort_profile_sha256(profile), expected)
        self.assertNotEqual(expected, hashlib.sha256(original).hexdigest())
        self.assertNotEqual(expected, native_bytes_sha(PROFILE_DOMAIN, compact))
        reversed_profile = {"input": dict(reversed(tuple(profile["input"].items()))), "domain": PROFILE_DOMAIN}
        self.assertEqual(reader.cohort_profile_sha256(reversed_profile), expected)
        changed = copy.deepcopy(profile)
        changed["input"]["requested_recipients"].reverse()
        self.assertNotEqual(reader.cohort_profile_sha256(changed), expected)

    def test_preparation_two_cannot_accept_a_duck_typed_or_success_flag_file_owner(self):
        class MetadataOnlyFiles:
            publication = lambda self: {"claimed_payloads_verified": True}
            raw_file = lambda self, _name: b"{}"
            durable_ack = lambda self: {"claimed_physical_complete": True}
        for value in (None, {}, MetadataOnlyFiles()):
            with self.subTest(kind=type(value).__name__):
                self.strict_rejection(value, preparation())

    def test_profile_roster_domain_and_unknown_fields_fail_before_incomplete_sources(self):
        changes = ((lambda value: value.update(domain="rz-rovent-utility-root-local-action/1"), "explicit profile domain"),
            (lambda value: value.update(extra="not a cohort declaration"), "profile fields mismatch"),
            (lambda value: value["input"].pop("resource_epoch"), "profile input fields mismatch"),
            (lambda value: value["input"].update(observed_available_recipients=["proposer"]), "profile input fields mismatch"))
        for index, (change, message) in enumerate(changes):
            value = profile_fixture(preparation())
            change(value)
            with self.subTest(public_profile=index), self.assertRaisesRegex(ValueError, message):
                reader.validate_cohort_profile(value)
            with self.subTest(case=index), pinned_files(preparation(), change_profile=change) as (files, _profile, _ack):
                self.strict_rejection(files, preparation())

    def test_cpu_modes_requested_order_and_exact_integer_windows_are_strict(self):
        changes = ((lambda value: value["input"].update(mode="hybrid_pc"), "CPU-only mode"),
            (lambda value: value["input"].update(requested_recipients=["proposer", "researcher"]), "unique ordered supported recipients"),
            (lambda value: value["input"].update(requested_recipients=["proposer", "proposer"]), "unique ordered supported recipients"),
            (lambda value: value["input"].update(requested_recipients=[]), "unique ordered supported recipients"),
            (lambda value: value["input"].update(prelude_ns=0), "positive bounded integer"),
            (lambda value: value["input"].update(action_ns=True), "invalid bounded unsigned integer"),
            (lambda value: value["input"].update(cleanup_ns=1.0), "invalid bounded unsigned integer"),
            (lambda value: value["input"].update(action_ns=2**64), "invalid bounded unsigned integer"),
            (lambda value: value["input"].update(action_ns=86400000000000), "declared window exceeds 24 hours"),
            (lambda value: value["input"].update(resource_epoch=0), "positive bounded integer"))
        for index, (change, message) in enumerate(changes):
            value = profile_fixture(preparation())
            change(value)
            with self.subTest(case=index), self.assertRaisesRegex(ValueError, message):
                reader.validate_cohort_profile(value)

    def test_collection_candidate_does_not_allow_root_local_or_unknown_cost_policies(self):
        changes = ((lambda value: value["input"].update(cost_scope="exclusive_submitted_global_target_route_and_descendants_with_required_drain"), "explicit usage/cost scope"),
            (lambda value: value["input"].update(usage="training_authorized"), "explicit usage/cost scope"),
            (lambda value: value["input"].update(frozen_nnuec_model_sha256="ab" * 33), "invalid SHA-256"),
            (lambda value: value["input"].update(source_sha256="AB" * 32), "invalid SHA-256"))
        for index, (change, message) in enumerate(changes):
            value = profile_fixture(preparation())
            change(value)
            with self.subTest(case=index), self.assertRaisesRegex(ValueError, message):
                reader.validate_cohort_profile(value)

    def test_trial_ack_returns_only_detached_data_and_respects_sparse_materializations(self):
        for count, materializations, pair_count in ((1, (), 0), (2, (0, 1), 1), (3, (0, 2), 3)):
            with self.subTest(trials=count), pinned_files(preparation(), trial_count=count,
                    materialization_ordinals=materializations, pair_count=pair_count) as (files, _profile, ack):
                checked = reader.validate_trial_acknowledgements(files)
                self.assertIs(type(checked), tuple)
                self.assertEqual(checked, tuple(ack["trial_acknowledgements"]))
                epochs = [item["collection_io"]["clock_epoch"] for item in checked]
                self.assertEqual(len(set(epochs)), count)
                self.assertNotIn(ack["collection_io"]["clock_epoch"], epochs)
                self.assertGreater(checked[0]["collection_io"]["finished_tick_ns"],
                    ack["collection_io"]["started_tick_ns"])
                if count > 1:
                    self.assertLess(checked[1]["collection_io"]["finished_tick_ns"],
                        checked[0]["collection_io"]["started_tick_ns"])
                if count > 2:
                    self.assertLess(checked[2]["collection_io"]["started_tick_ns"],
                        checked[0]["collection_io"]["finished_tick_ns"])
                self.assertFalse(files.observation()["physical_completion_verified"])
                self.assertFalse(files.observation()["training_target_authorized"])
                checked[0]["collection_io"]["written_bytes"] = -1
                self.assertEqual(reader.validate_trial_acknowledgements(files), tuple(ack["trial_acknowledgements"]))

    def test_ack_minimal_body_acceptance_cannot_be_promoted_to_a_full_payload_target(self):
        with pinned_files(preparation()) as (files, _profile, _ack):
            self.assertEqual(len(reader.validate_trial_acknowledgements(files)), 2)
            with self.assertRaisesRegex(ValueError, "utility cohort original trial checkpoint fields mismatch"):
                reader.validate_cohort_payloads(files, preparation=preparation())

    def test_trial_ack_requires_original_file_owner_not_a_detached_flag(self):
        class MetadataOnlyAck:
            def publication(self):
                raise AssertionError("duck typed file owner must not be inspected")
        for value in (None, {}, MetadataOnlyAck()):
            with self.subTest(kind=type(value).__name__), self.assertRaisesRegex(ValueError, "original pinned file owner"):
                reader.validate_trial_acknowledgements(value)

    def test_trial_ack_body_is_closed_and_cannot_invent_complete_or_free_io(self):
        prefix = "utility cohort original trial ACK"
        changes = ((lambda value: value["trial_acknowledgements"].pop(), prefix + " complete coverage"),
            (lambda value: value["trial_acknowledgements"][0].update(physical_completion_verified=True), prefix + " fields"),
            (lambda value: value["trial_acknowledgements"][0].pop("checkpoint"), prefix + " fields"),
            (lambda value: value["trial_acknowledgements"][0].update(trial_sequence=2), prefix + " checkpoint/order"),
            (lambda value: value["trial_acknowledgements"][0].update(trial_sequence=True), "invalid bounded unsigned integer"),
            (lambda value: value["trial_acknowledgements"][0].update(native_trial_sha256="f" * 64), prefix + " native body seal"),
            (lambda value: value["trial_acknowledgements"][0]["checkpoint"].update(extra=True), prefix + " pin fields"),
            (lambda value: value["trial_acknowledgements"][0]["checkpoint"].update(bytes=1), prefix + " checkpoint/order"),
            (lambda value: value["trial_acknowledgements"][0]["checkpoint"].update(sha256="0" * 64), prefix + " checkpoint/order"),
            (lambda value: value["trial_acknowledgements"][0]["checkpoint"].update(bytes=True), prefix + " checkpoint/order"),
            (lambda value: value["trial_acknowledgements"][0]["collection_io"].update(written_bytes=0), prefix + " I/O accounting"),
            (lambda value: value["trial_acknowledgements"][0]["collection_io"].update(readback_bytes=1), prefix + " I/O accounting"),
            (lambda value: value["trial_acknowledgements"][0]["collection_io"].update(synced_files=True), "invalid bounded unsigned integer"),
            (lambda value: value["trial_acknowledgements"][0]["collection_io"].update(readback_bytes=2**64), "invalid bounded unsigned integer"),
            (lambda value: value["trial_acknowledgements"][0]["collection_io"].update(elapsed_ns=3), prefix + " I/O clock"),
            (lambda value: value["trial_acknowledgements"][0]["collection_io"].update(physical_zero_tick=12), prefix + " collection I/O fields"))
        for index, (change, message) in enumerate(changes):
            with self.subTest(case=index), pinned_files(preparation(), change_ack=change) as (files, _profile, _ack):
                with self.assertRaisesRegex(ValueError, message):
                    reader.validate_trial_acknowledgements(files)

    def test_trial_ack_native_seal_uses_original_raw_bytes_not_file_or_canonical_sha(self):
        def wrong_seal(kind):
            def change(ack):
                first = ack["trial_acknowledgements"][0]
                if kind == "file":
                    first["native_trial_sha256"] = first["checkpoint"]["sha256"]
                else:
                    body = {"schema": TRIAL_DOMAIN, "cohort_id": ack["cohort_id"], "trial_sequence": 1}
                    compact = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    if kind == "canonical":
                        first["native_trial_sha256"] = native_bytes_sha(TRIAL_DOMAIN, compact)
                    else:
                        original = raw(body)
                        first["native_trial_sha256"] = hashlib.sha256(TRIAL_DOMAIN.encode("ascii") + b"\0" + original).hexdigest()
            return change
        for kind in ("file", "canonical", "missing length"):
            with self.subTest(kind=kind), pinned_files(preparation(), change_ack=wrong_seal(kind)) as (files, _profile, _ack):
                with self.assertRaisesRegex(ValueError, "utility cohort original trial ACK native body seal"):
                    reader.validate_trial_acknowledgements(files)

    def test_trial_ack_rejects_body_order_or_domain_even_when_raw_pins_are_recomputed(self):
        changes = (lambda body, ordinal: body.update(schema="rz-rovent-utility-root-local-action/1") if ordinal == 0 else None,
            lambda body, ordinal: body.update(trial_sequence=True) if ordinal == 0 else None,
            lambda body, ordinal: body.update(trial_sequence=2) if ordinal == 0 else None,
            lambda body, ordinal: body.update(cohort_id="f" * 64) if ordinal == 0 else None)
        for index, change in enumerate(changes):
            with self.subTest(case=index), pinned_files(preparation(), change_trial_body=change) as (files, _profile, _ack):
                with self.assertRaisesRegex(ValueError, "utility cohort original trial ACK native body seal"):
                    reader.validate_trial_acknowledgements(files)

    def test_trial_ack_rejects_duplicate_json_keys_and_non_utf8_original_bytes(self):
        changes = (lambda original, ordinal: b'{"schema":"other",' + original.lstrip()[1:] if ordinal == 0 else original,
            lambda original, ordinal: b"\xff" + original if ordinal == 0 else original)
        for index, change in enumerate(changes):
            message = "duplicate keys" if index == 0 else "original UTF8 JSON"
            with self.subTest(case=index), pinned_files(preparation(), change_trial_raw=change) as (files, _profile, _ack):
                with self.assertRaisesRegex(ValueError, message):
                    reader.validate_trial_acknowledgements(files)

    def test_full_cohort_complete_metadata_without_new_witnesses_returns_no_known_target(self):
        with full_cohort_files() as (files, prepared, expected):
            result = reader.validate_cohort_payloads(files, preparation=prepared)
            self.assertEqual(set(result), {"input", "report"})
            report, captured = result["report"], result["input"]
            self.assertEqual(set(report), set(REPORT_FIELDS))
            self.assertEqual(report["known_pairs"], [])
            self.assertEqual(report["known_pair_sha256"], [])
            self.assertEqual(report["all_pair_sha256"], [expected["pair_sha256"]])
            self.assertIsNone(expected["pair"]["preferred"])
            self.assertEqual(expected["pair"]["unknown_reason"], "no_new_valid_witness")
            self.assertEqual(expected["pair"]["target_known_mask"], [False] * 7)
            self.assertTrue(all(trial["resolved"] and trial["comparable"] for trial in expected["trials"]))
            self.assertTrue(all(trial["unknown_reason"] is None for trial in expected["trials"]))
            for trial in expected["trials"]:
                materialized = json.loads(trial["materialized_json"])
                provenance = materialized["input"]["provenance"]
                selected = next(offer for offer in trial["ready_prelude"]["offers"]
                    if offer["recipient"] == trial["recipient"])
                self.assertEqual(provenance["root_state"], trial["original_root"]["store_root_state"])
                self.assertEqual(provenance["generation"], trial["original_root"]["root_generation"])
                self.assertEqual(provenance["sequence"], selected["view"]["sequence"])
                self.assertEqual(provenance["view_seal_sha256"], selected["view"]["view_digest"])
                self.assertEqual(provenance["full_view_sha256"], selected["full_view_sha256"])
                self.assertEqual(provenance["policy_sha256"], selected["view"]["context_policy_sha256"])
            source = expected["source"]
            self.assertEqual(report["cohort_id"], files.publication()["cohort_id"])
            self.assertEqual(report["profile_sha256"], expected["profile_sha256"])
            self.assertEqual(report["collection_source_sha256"], source["source_pins"]["source_sha256"])
            self.assertEqual(report["resource_profile_sha256"], source["source_pins"]["resource_plan_sha256"])
            self.assertEqual(report["cost_epoch_sha256"], expected["cost_epoch_sha256"])
            self.assertEqual(report["frozen_base_weights_sha256"], expected["semantic_weights_sha256"])
            self.assertNotEqual(expected["semantic_weights_sha256"], expected["weights_raw_sha256"])
            self.assertEqual(report["input_reference_json"], expected["input"]["reference_json"])
            self.assertEqual(captured.input_json, expected["input"]["reference_json"])
            for name in ("input_sha256", "target_sha256", "submission_target_sha256"):
                self.assertEqual(report[name], expected["input"][name])
            self.assertEqual(report["utility_context_sha256"], expected["input"]["context_sha256"])
            self.assertEqual(report["frozen_pc_sources"], source["frozen_pc_sources"])
            self.assertEqual(report["rules_state_sha256"], source["rules_root"]["anchor"]["rules_state_sha256"])
            self.assertEqual(report["transposition_sha256"], source["transposition_sha256"])
            self.assertEqual(report["ancestor_dataset_publication_sha256"], [])
            self.assertEqual(report["ancestor_datasets"], [])
            metadata = source["collection_metadata"]
            for name in ("opening_id", "game_id", "line_genealogy_id"):
                self.assertEqual(report[name], metadata[name])
            self.assertEqual(report["split"], source["collection_manifest"]["opening_splits"][metadata["opening_id"]])
            self.assertEqual(report["usage"], "collection_candidate")
            observation = captured.observation()
            for name in ("execution_authorized", "training_target_authorized",
                    "original_preparation_authority_reissued", "runtime_lease_authority"):
                self.assertIs(observation[name], False)
            self.assertIs(files.observation()["physical_completion_verified"], False)

    def test_full_cohort_unknown_resource_readback_remains_censored_and_has_no_known_target(self):
        with full_cohort_files(resource_known=False) as (files, prepared, expected):
            result = reader.validate_cohort_payloads(files, preparation=prepared)
            self.assertEqual(set(result["report"]), set(REPORT_FIELDS))
            self.assertEqual(result["report"]["known_pairs"], [])
            self.assertEqual(result["report"]["known_pair_sha256"], [])
            self.assertEqual(result["report"]["all_pair_sha256"], [expected["pair_sha256"]])
            self.assertIsNone(expected["pair"]["preferred"])
            self.assertEqual(expected["pair"]["unknown_reason"], "unresolved_or_masked_trial")
            for trial in expected["trials"]:
                self.assertIs(trial["resolved"], True)
                self.assertIs(trial["comparable"], False)
                self.assertEqual(trial["unknown_reason"], "unknown_original_resource_observation")
                self.assertEqual(trial["target_known_mask"], [False] * 7)
            self.assertEqual(result["input"].input_json, expected["input"]["reference_json"])
            self.assertIs(result["input"].observation()["training_target_authorized"], False)
            self.assertIs(files.observation()["physical_completion_verified"], False)

        def missing_original_bodies(trial, materialization, _ordinal):
            materialization["semantic"]["witness_preimages"] = None
            materialization["semantic"]["witness_basis"] = None
            trial["comparable"] = False
            trial["unknown_reason"] = "missing_original_conditional_witness_bodies"

        with self.subTest(case="missing_original_witness_bodies"), full_cohort_files(
                resource_known=False, change_trial=missing_original_bodies) as (files, prepared, expected):
            # The original bodies were not captured. Re-sealing all enclosing
            # files and causal ACKs does not turn this absence into a target.
            result = reader.validate_cohort_payloads(files, preparation=prepared)
            self.assertEqual(result["report"]["known_pairs"], [])
            self.assertEqual(result["report"]["known_pair_sha256"], [])
            self.assertEqual(result["report"]["all_pair_sha256"], [expected["pair_sha256"]])
            self.assertIsNone(expected["pair"]["preferred"])
            self.assertEqual(expected["pair"]["unknown_reason"], "unresolved_or_masked_trial")
            self.assertEqual(expected["pair"]["target_known_mask"], [False] * 7)
            for trial in expected["trials"]:
                self.assertIs(trial["resolved"], True)
                self.assertIs(trial["comparable"], False)
                self.assertEqual(trial["unknown_reason"], "missing_original_conditional_witness_bodies")
                semantic = json.loads(trial["materialized_json"])["semantic"]
                self.assertEqual(semantic["new_witnesses"], [])
                self.assertIsNone(semantic["witness_preimages"])
                self.assertIsNone(semantic["witness_basis"])
            self.assertEqual(result["input"].input_json, expected["input"]["reference_json"])
            self.assertIs(result["input"].observation()["training_target_authorized"], False)
            self.assertIs(files.observation()["physical_completion_verified"], False)

    def test_original_conditional_basis_absence_and_raw_corruption_never_become_complete(self):
        # Isolate the byte/metadata adapter. These synthetic bytes are not a
        # Core comparison certificate, a live witness or a known learner pair.
        witness = {"observation": 7, "kind": "Counter", "identity_sha256": h("synthetic witness"),
            "source_question_sha256": h("synthetic question"), "source_frame_sequence": 2,
            "raw_evidence_sha256": h("synthetic raw evidence"), "line": [64]}
        row = {name: copy.deepcopy(value) for name, value in witness.items() if name != "line"}
        row.update(basis=None, unavailable_reason="missing_original_comparison")
        scope = {"trial": {"recipient": "route_p"}, "materialization": {
            "prelude": {"offers": [{"recipient": "route_p", "budget": {"original_store_byte_limit": 1024}}]},
            "semantic": {"new_witnesses": [witness], "witness_preimages": [{"preimage_bytes": 1}],
                "witness_basis": [row]}}}
        self.assertIs(reader._witness_basis(scope), False)
        for reason in (None, "", "accepted", True):
            altered = copy.deepcopy(scope)
            altered["materialization"]["semantic"]["witness_basis"][0]["unavailable_reason"] = reason
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, "unavailable basis reason"):
                reader._witness_basis(altered)
        altered = copy.deepcopy(scope)
        altered["materialization"]["semantic"]["witness_basis"][0]["observation"] = 8
        with self.assertRaisesRegex(ValueError, "witness/basis metadata"):
            reader._witness_basis(altered)
        for change, message in ((lambda basis: basis.update(bytes=3), "byte count"),
                (lambda basis: basis.update(sha256=h("different raw bytes")), "raw byte SHA"),
                (lambda basis: basis.update(domain="rz-rovent-utility-actual-conditional-witness/1"), "domain"),
                (lambda basis: basis.update(original_json=chr(0xD800)), "UTF8 JSON string")):
            altered = copy.deepcopy(scope)
            target = altered["materialization"]["semantic"]["witness_basis"][0]
            target.update(basis={"domain": reader.BASIS_DOMAIN, "sha256": hashlib.sha256(b"{}").hexdigest(),
                "bytes": 2, "original_json": "{}"}, unavailable_reason=None)
            change(target["basis"])
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                reader._witness_basis(altered)

    def test_materialized_input_cannot_rebind_its_selected_ready_view_provenance(self):
        for key in ("sequence", "full_view_sha256", "policy_sha256"):
            def alter(_trial, materialization, ordinal):
                if ordinal == 0:
                    provenance = materialization["input"]["provenance"]
                    provenance[key] = provenance[key] + 1 if key == "sequence" else h("different " + key)
                    seal_fixture_submitted_view(materialization["input"])
            with self.subTest(key=key), full_cohort_files(change_trial=alter) as (files, prepared, _expected):
                # The fixture recomputes enclosing file/native/ACK seals. The
                # input remains distinct from the original immutable offer.
                with self.assertRaisesRegex(ValueError, "input provenance differs from its original root/selected ready view"):
                    reader.validate_cohort_payloads(files, preparation=prepared)

    def test_original_view_seal_preserves_source_revision_state_and_u16_offset(self):
        for key in ("source_revision", "source_state", "source_line_offset"):
            def alter(_trial, materialization, ordinal):
                if ordinal == 0:
                    # Keep target/source grouping intact while altering the
                    # original metadata covered by the unchanged view seal.
                    for record in materialization["input"]["records_metadata"]:
                        record[key] += 1
            with self.subTest(key=key), full_cohort_files(change_trial=alter) as (files, prepared, _expected):
                with self.assertRaisesRegex(ValueError, "submitted view seal/source revision framing"):
                    reader.validate_cohort_payloads(files, preparation=prepared)

        def overflow_offset(_trial, materialization, ordinal):
            if ordinal == 0:
                materialization["input"]["records_metadata"][0]["source_line_offset"] = 65536
        with full_cohort_files(change_trial=overflow_offset) as (files, prepared, _expected):
            with self.assertRaises(ValueError):
                reader.validate_cohort_payloads(files, preparation=prepared)

    def test_full_cohort_without_materializations_preserves_original_frozen_capture_as_unknown(self):
        with full_cohort_files(materialized=False) as (files, prepared, expected):
            result = reader.validate_cohort_payloads(files, preparation=prepared)
            report = result["report"]
            self.assertEqual(set(report), set(REPORT_FIELDS))
            self.assertEqual(files.publication()["materializations"], [])
            self.assertEqual(report["known_pairs"], [])
            self.assertEqual(report["known_pair_sha256"], [])
            self.assertEqual(report["all_pair_sha256"], [expected["pair_sha256"]])
            self.assertIsNone(expected["pair"]["preferred"])
            self.assertEqual(expected["pair"]["unknown_reason"], "unresolved_or_masked_trial")
            for trial in expected["trials"]:
                self.assertIsNone(trial["materialized_json"])
                self.assertIsNone(trial["materialized_sha256"])
                self.assertIs(trial["resolved"], False)
                self.assertIs(trial["comparable"], False)
                self.assertIs(trial["cost"]["complete_scope"], False)
                self.assertEqual(trial["unknown_reason"], "missing_bounded_historical_materialization")
                self.assertEqual(trial["target_known_mask"], [False] * 7)
                self.assertEqual(len(trial["ready_prelude"]), 11)
                self.assertIs(type(trial["original_root"]), dict)
                historical = trial["historical_unbound_input"]
                self.assertEqual(set(historical), {"reference_json", "input_sha256", "target_sha256", "submission_target_sha256",
                    "context_sha256", "target_record_ids", "original_view_sha256", "original_full_view_sha256",
                    "materialized_inside_complete_scope"})
                self.assertIs(historical["materialized_inside_complete_scope"], False)
            first = expected["trials"][0]
            self.assertEqual(first["original_root"], expected["source"]["registration_root"])
            self.assertNotEqual(first["original_root"]["process_epoch"], expected["trials"][1]["original_root"]["process_epoch"])
            self.assertEqual(first["frozen_context"]["first_trial_sequence"], 1)
            historical = first["historical_unbound_input"]
            self.assertEqual(report["input_reference_json"], historical["reference_json"])
            self.assertEqual(result["input"].input_json, historical["reference_json"])
            for name in ("input_sha256", "target_sha256", "submission_target_sha256"):
                self.assertEqual(report[name], historical[name])
            self.assertEqual(report["utility_context_sha256"], historical["context_sha256"])
            self.assertIs(result["input"].observation()["training_target_authorized"], False)
            self.assertIs(files.observation()["physical_completion_verified"], False)

        def altered_group(trial, materialization, ordinal):
            if ordinal == 0:
                trial["ready_prelude"]["common_node_work"]["root_entries"] += 1
                materialization["prelude"]["common_node_work"]["root_entries"] += 1

        def mismatched_materialization(trial, materialization, ordinal):
            if ordinal == 0:
                materialization["prelude"]["captured_tick"] += 1

        changes = (
            ("missing_ready", lambda trial, body, ordinal: trial.pop("ready_prelude") if ordinal == 0 else None,
                "utility cohort original trial checkpoint fields mismatch"),
            ("changed_actual_group", altered_group, "utility cohort.*ready-group preimage"),
            ("different_materialized_prelude", mismatched_materialization, "utility cohort original materialization/trial linkage differs"),
            ("missing_original_root", lambda trial, body, ordinal: trial.pop("original_root") if ordinal == 0 else None,
                "utility cohort original trial checkpoint fields mismatch"),
            ("mixed_original_root", lambda trial, body, ordinal: trial["original_root"].update(process_epoch=2) if ordinal == 0 else None,
                "utility cohort original materialization/trial linkage differs"),
        )
        for name, change, message in changes:
            with self.subTest(case=name), full_cohort_files(change_trial=change) as (files, prepared, _expected):
                # Every body/file/native/causal ACK pin has already been
                # recomputed by the fixture. The original typed join must
                # still reject the changed root/prelude rather than accepting
                # a self-consistent outer hash as execution authority.
                with self.assertRaisesRegex(ValueError, message) as rejected:
                    reader.validate_cohort_payloads(files, preparation=prepared)
                self.assertNotIsInstance(rejected.exception, reader.UnsupportedCohortPayload)


if __name__ == "__main__":
    unittest.main()
