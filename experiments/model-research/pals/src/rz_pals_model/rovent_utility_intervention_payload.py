"""Strict /2 original neutral/arm payloads; no restored execution authority.

The /1 admission, Ready selection and cohort codec are never invoked. Only
unchanged raw Rules/root/weight/witness/physical arithmetic leaves are shared.
Actual legal replay and conditional comparison proof remain Rust-owned.
"""
from __future__ import annotations

import copy
import hashlib
import json

from .generation_data import digest, fields, identifier, sha, uint, _moves
from . import rovent_utility_cohort_payload as proof
from . import rovent_utility_root_local as structure
from . import rovent_corpus as corpus
from . import rovent_historical_r_corpus as r_source
from .rovent_historical_corpus import _input_lock, _source_aggregate
from .rovent_native_v2 import _artifact, _commit, validate_position_replay
from .rovent_corpus_codec import SUBMITTED_CODEC
from .rovent_utility_input import validate_submitted_utility_input
from .rovent_utility_intervention import (COST_SCOPE, PROFILE_DOMAIN, TRIAL_DOMAIN,
    MATERIALIZATION_DOMAIN, PAIR_DOMAIN, TARGET_DOMAIN, ROUTES, canonical_bytes,
    validate_intervention_profile, intervention_profile_sha256)
from .rovent_utility_intervention_files import (HistoricalInterventionFiles,
    MAX_MATERIALIZATION_BYTES, MAX_METADATA_BYTES, ordinal)


SOURCE_DOMAIN = "rz-rovent-utility-cohort-collection-source/2"
SOURCE_FIELDS = ("schema", "source_pins", "nnuec_registration", "nnuec_weights_artifact",
    "nnuec_provenance_artifact", "native_origin", "registration_root", "collection_manifest",
    "collection_metadata", "rules_root", "transposition_sha256", "frozen_pc_sources",
    "ancestor_datasets", "resource_observation")
NEUTRAL_FIELDS = ("domain", "input_sha256", "target_sha256", "context_sha256", "target_record",
    "target_record_ids", "input_bytes", "reference_json", "root_snapshot_sha256", "source_snapshot_sha256",
    "record_scope_sha256", "model_identity", "original_first_move", "root_rules", "records_metadata",
    "source_rules", "original_view_sha256", "original_full_view_sha256")
ARM_FIELDS = ("neutral_target_sha256", "arm_query_sha256", "recipient", "frame_sequence",
    "source_frame_sequence", "stage", "purpose", "original_first_move", "view", "full_view_sha256",
    "public_context_sha256", "budget", "deadline_clock", "deadline_tick", "query", "prepared_input")
TRIAL_FIELDS = ("schema", "cohort_id", "trial_sequence", "recipient", "profile", "profile_sha256", "source",
    "neutral_target", "neutral_target_sha256", "arm_query", "materialized_sha256", "materialized_json",
    "historical_unbound_input", "unresolved_source", "cost", "original_root", "preparation", "physical_closure",
    "previous_durable_ack", "resolved", "comparable", "unknown_reason", "support_mask", "target_known_mask")
MATERIALIZATION_FIELDS = ("schema", "cohort_id", "trial_sequence", "recipient", "profile_sha256", "source",
    "original_root", "preparation", "neutral_target", "arm_query", "action", "semantic")
_RECIPIENTS = ("proposer", "critic", "researcher")
_ROUTE = dict(zip(_RECIPIENTS, ROUTES, strict=True))


class UnsupportedInterventionPayload(ValueError):
    """Missing original operands stay unknown; no fallback target is made."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def _source_enrollment(files, profile, registration):
    """Independent /2 entry; the /1 source admission is deliberately unused."""
    source = fields(proof._decode(files.raw_file("source.json"), 1 << 20), SOURCE_FIELDS,
        "intervention original collection source/2")
    if source["schema"] != SOURCE_DOMAIN:
        raise ValueError("explicit intervention original collection source/2 required")
    proof._source(source["source_pins"], profile)
    pins = source["source_pins"]
    if (pins["source_sha256"] != registration["collection_source_sha256"]
            or pins["resource_plan_sha256"] != registration["resource_profile_sha256"]
            or pins["frozen_nnuec_model_sha256"] != registration["frozen_base"]["weights_sha256"]):
        raise ValueError("intervention source/registration/frozen parameters differ")
    lock = _input_lock(proof._decode(files.raw_file("input-lock.json"), 4 << 20), pins["input_lock_sha256"])
    engine = fields(lock["manifest"]["engine"], ("source_commit", "source_components", "binary", "mode", "pc",
        "researcher", "teachers", "nnuec", "search"), "intervention original lock engine")
    _commit(engine["source_commit"])
    provenance = proof._array(lock["manifest"]["provenance"], 32)
    if not provenance:
        raise ValueError("intervention original lock provenance required")
    for artifact in provenance:
        _artifact(artifact)
    if (lock["manifest"]["purpose"] != "collection" or engine["mode"] != profile["input"]["mode"]
            or _source_aggregate(engine["source_commit"], engine["source_components"]) != pins["source_sha256"]
            or digest("rz-rovent-resource-plan/1", lock["manifest"]["resources"]) != pins["resource_plan_sha256"]):
        raise ValueError("intervention original lock/source/mode/resource differs")
    declared = r_source._nnuec_pin(engine["nnuec"])
    for name, key, filename in (("nnuec_weights_artifact", "weights", "nnuec-weights.json"),
            ("nnuec_provenance_artifact", "provenance", "nnuec-provenance.json")):
        _artifact(source[name])
        pin = files.file_pin(filename)
        if (source[name] != declared[key] or source[name]["sha256"] != pin["sha256"]
                or source[name]["bytes"] != pin["bytes"]):
            raise ValueError("intervention original NNUEC artifact/lock/raw pin differs")
    parameter_sha, config, training = proof._source_parameter_sha(proof._decode(
        files.raw_file("nnuec-weights.json"), 32 << 20, source_operands=True))
    if parameter_sha != pins["frozen_nnuec_model_sha256"] or config.to_mapping() != registration["frozen_base"]["config"]:
        raise ValueError("intervention actual frozen FP32 parameter bytes differ")
    r_source._nnuec_registration(source["nnuec_registration"], declared, parameter_sha, training)
    structure._registration(source["nnuec_registration"], source["nnuec_registration"]["model_identity"])
    if proof._value_digest("rz-rovent-utility-cohort-nnuec-namespace/1", source["nnuec_registration"]) != pins["frozen_nnuec_namespace_sha256"]:
        raise ValueError("intervention original NNUEC namespace differs")
    r_source._provenance(proof._decode(files.raw_file("nnuec-provenance.json"), 1 << 20),
        declared, parameter_sha, training, lock)
    manifest = corpus._manifest(source["collection_manifest"], corpus.CorpusLimits(), codec=SUBMITTED_CODEC)
    if manifest["source_sha256"] != pins["source_sha256"] or manifest["source_commit"] != engine["source_commit"]:
        raise ValueError("intervention original collection manifest/source differs")
    metadata = fields(source["collection_metadata"], ("game_id", "opening_id", "line_genealogy_id", "lineage"),
        "intervention original collection metadata")
    for name in ("game_id", "opening_id", "line_genealogy_id"):
        identifier(metadata[name])
    lineage = fields(metadata["lineage"], ("actor_id", "teacher_actor_id", "ancestor_origin_sha256"),
        "intervention original lineage")
    identifier(lineage["actor_id"])
    if lineage["teacher_actor_id"] is not None:
        identifier(lineage["teacher_actor_id"])
    corpus._sha_list(lineage["ancestor_origin_sha256"], 256, "intervention ancestor origin")
    if metadata["opening_id"] not in manifest["opening_splits"]:
        raise ValueError("intervention opening lacks original registered split")
    validate_position_replay(source["rules_root"])
    sha(source["transposition_sha256"])
    ancestors = proof._array(source["ancestor_datasets"], 256)
    ancestor_publications = set()
    for descriptor in ancestors:
        if type(descriptor) is dict and descriptor.get("kind") == "utility_intervention":
            fields(descriptor, ("kind", "publication_sha256", "metadata"), "original intervention ancestor")
            fields(descriptor["metadata"], ("dataset_sha256", "pool_sha256", "preparation_sha256", "context_sha256"),
                "original intervention ancestor source seals")
            for seal in descriptor["metadata"].values(): sha(seal)
            sha(descriptor["publication_sha256"])
        else:
            proof._ancestor_descriptors([descriptor])
        if descriptor["publication_sha256"] in ancestor_publications:
            raise ValueError("intervention original ancestor publication repeated")
        ancestor_publications.add(descriptor["publication_sha256"])
    scopes = proof._array(source["resource_observation"], 3)
    if len(scopes) != files.publication()["trial_count"]:
        raise ValueError("intervention original per-trial resource custody coverage differs")
    for scope in scopes:
        proof._resource_scope(scope)
    return source, lock, config


def snapshot_sha256(rules):
    """Original Search snapshot_digest bytes; this does not replay Rules."""
    proof._rules_projection(rules)
    replay = rules["uci_replay"]
    start = replay["start_fen"].encode("utf-8", "strict")
    moves = replay["moves"]
    if len(moves) > 65535:
        raise ValueError("intervention exact Rules replay exceeds original u16 framing")
    framed = (b"rz-pals-exact-review-anchor/1\0"
        + bytes((0 if replay["origin"] == "start_position" else 1,
            0 if replay["completeness"] == "complete" else 1))
        + len(start).to_bytes(8, "little") + start + len(moves).to_bytes(2, "little")
        + b"".join(move.to_bytes(2, "little") for move in moves) + rules["fen"].encode("utf-8", "strict"))
    return hashlib.sha256(framed).hexdigest()


def _identity_bytes(value):
    structure._identity(value)
    # CpuValueIdentity is a serde struct in this declared field order. Its
    # internally tagged training body also retains the producer field order.
    training = value["training"]
    ordered_training = {name: training[name] for name in ("kind", "run_id", "steps", "dataset_sha256") if name in training}
    return json.dumps({"semantics": value["semantics"], "weights_sha256": value["weights_sha256"],
        "training": ordered_training}, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def neutral_target_sha256(value):
    """Pure original neutral cap preimage, independent of arm/authority/tick."""
    fields(value, NEUTRAL_FIELDS, "intervention original neutral target")
    if value["domain"] != TARGET_DOMAIN:
        raise ValueError("explicit original intervention neutral target domain required")
    for name in ("input_sha256", "target_sha256", "context_sha256", "root_snapshot_sha256",
            "record_scope_sha256", "original_view_sha256", "original_full_view_sha256"):
        sha(value[name])
    if type(value["reference_json"]) is not str:
        raise ValueError("intervention original neutral input string required")
    raw = value["reference_json"].encode("utf-8", "strict")
    if uint(value["input_bytes"], 8 << 20) != len(raw):
        raise ValueError("intervention original neutral input byte count differs")
    snapshots = proof._array(value["source_snapshot_sha256"], 64)
    for snapshot in snapshots:
        sha(snapshot)
    identity = _identity_bytes(value["model_identity"])
    _moves([value["original_first_move"]], 1)
    framed = (TARGET_DOMAIN.encode("ascii") + b"\0" + bytes.fromhex(value["root_snapshot_sha256"])
        + bytes.fromhex(value["record_scope_sha256"]) + uint(value["target_record"], 63).to_bytes(2, "little")
        + value["original_first_move"].to_bytes(2, "little") + len(raw).to_bytes(8, "little") + raw
        + len(identity).to_bytes(8, "little") + identity + len(snapshots).to_bytes(8, "little")
        + b"".join(bytes.fromhex(item) for item in snapshots))
    return hashlib.sha256(framed).hexdigest()


def _neutral(value, *, config, source):
    target = neutral_target_sha256(value)
    if (value["model_identity"] != source["nnuec_registration"]["model_identity"]
            or snapshot_sha256(value["root_rules"]) != value["root_snapshot_sha256"]
            or value["root_rules"]["semantic_sha256"] != source["rules_root"]["anchor"]["rules_state_sha256"]):
        raise ValueError("intervention neutral original Rules/frozen namespace differs")
    # The original input parser's submission seal is calculated for local
    # numerical validation only; it is never a newly issued arm request.
    submission = hashlib.sha256(b"rz-nnuec-submitted-utility-submission-target/1\0"
        + bytes.fromhex(value["original_view_sha256"]) + value["target_sha256"].encode("ascii")).hexdigest()
    captured = validate_submitted_utility_input(input_json=value["reference_json"],
        base_weights_sha256=value["model_identity"]["weights_sha256"], context_sha256=value["context_sha256"],
        target_ids=value["target_record_ids"], view_seal_sha256=value["original_view_sha256"],
        input_sha256=value["input_sha256"], target_sha256=value["target_sha256"],
        submission_target_sha256=submission, config=config)
    metadata, snapshots = proof._array(value["records_metadata"], 64), proof._array(value["source_rules"], 64)
    lines = captured.reference_input.base_input.context_lines
    if not (len(metadata) == len(snapshots) == len(lines) == len(value["source_snapshot_sha256"])):
        raise ValueError("intervention neutral original record/Rules allocations differ")
    edges, exact_sources = [], {}
    for index, (row, rules, line) in enumerate(zip(metadata, snapshots, lines, strict=True)):
        fields(row, ("anchor", "parent", "supersedes", "source_state", "source_revision", "source_line_offset",
            "original_first_move"), "intervention original neutral record metadata")
        uint(row["anchor"], 3)
        for name in ("source_state", "source_revision"):
            uint(row[name])
        uint(row["source_line_offset"], 65535)
        first = line.original_first_move
        packed = None if first is None else first.from_square | first.to_square << 6 | first.promotion << 12
        if row["original_first_move"] != packed:
            raise ValueError("intervention neutral numeric/original first move differs")
        for kind, name in enumerate(("parent", "supersedes")):
            if row[name] is not None:
                if uint(row[name], 63) >= index:
                    raise ValueError("intervention neutral relation chronology differs")
                edges.append((row[name], index, kind))
        if snapshot_sha256(rules) != value["source_snapshot_sha256"][index]:
            raise ValueError("intervention neutral original source Rules replay seal differs")
        projected = proof._rules_projection(rules)
        if exact_sources.setdefault(row["source_state"], projected) != projected:
            raise ValueError("intervention neutral StateId changed exact original Rules history")
    if edges != [(e.source, e.target, e.kind) for e in captured.reference_input.base_input.context_edges]:
        raise ValueError("intervention neutral relation graph differs")
    selected = value["target_record"]
    if selected >= len(metadata) or metadata[selected]["original_first_move"] != value["original_first_move"]:
        raise ValueError("intervention original selected target/first move differs")
    if lines[selected].kind != 0 or metadata[selected]["anchor"] != 0 or not lines[selected].line.moves:
        raise ValueError("intervention target requires its original current-root Proposal record")
    first = lines[selected].line.moves[0]
    if first.from_square | first.to_square << 6 | first.promotion << 12 != value["original_first_move"]:
        raise ValueError("intervention original Proposal must start with its declared first move")
    expected = [index for index, row in enumerate(metadata) if row["source_state"] == metadata[selected]["source_state"]
        and row["original_first_move"] == value["original_first_move"]]
    if value["target_record_ids"] != expected:
        raise ValueError("intervention exact original target bundle differs")
    return captured, target


def validate_intervention_trial_acknowledgements(files):
    """Own-clock ACK joins only; no cross-process absolute tick ordering."""
    if type(files) is not HistoricalInterventionFiles:
        raise ValueError("original intervention pinned-byte owner required")
    publication, ack = files.publication(), files.durable_ack()
    result = []
    for sequence, (entry, pin) in enumerate(zip(ack["trial_acknowledgements"], publication["trials"], strict=True), 1):
        raw = files.raw_file(pin["path"])
        body = proof._decode(raw, MAX_MATERIALIZATION_BYTES)
        if (entry["checkpoint"] != pin or entry["trial_sequence"] != sequence
                or body.get("schema") != TRIAL_DOMAIN or type(body.get("trial_sequence")) is not int
                or body["trial_sequence"] != sequence or body.get("cohort_id") != publication["cohort_id"]
                or entry["native_trial_sha256"] != proof._native_digest(TRIAL_DOMAIN, raw)):
            raise ValueError("intervention original ACK/checkpoint/native body seal differs")
        result.append(copy.deepcopy(entry))
    return tuple(result)


def _trials(files, profile):
    publication = files.publication()
    profile_sha = intervention_profile_sha256(profile)
    routes = [_ROUTE[item] for item in profile["input"]["requested_recipients"]]
    if len(routes) != publication["trial_count"]:
        raise ValueError("intervention original declared arm count differs")
    materializations = {ordinal(pin["path"], "materializations"): pin for pin in publication["materializations"]}
    trials = []
    for index, pin in enumerate(publication["trials"]):
        trial = fields(proof._decode(files.raw_file(pin["path"]), MAX_MATERIALIZATION_BYTES), TRIAL_FIELDS,
            "intervention original trial/2")
        if (trial["schema"] != TRIAL_DOMAIN or trial["cohort_id"] != publication["cohort_id"]
                or proof._positive(trial["trial_sequence"], 3) != index + 1 or trial["recipient"] != routes[index]
                or trial["profile"] != profile or sha(trial["profile_sha256"]) != profile_sha):
            raise ValueError("intervention original trial/group/profile/order/arm differs")
        proof._source(trial["source"], profile)
        for name in ("resolved", "comparable"):
            proof._boolean(trial[name], name)
        known, support = trial["target_known_mask"], trial["support_mask"]
        declared_support = [index >= 4 and ROUTES[index - 4] in routes for index in range(7)]
        if (type(known) is not list or len(known) != 7 or any(type(bit) is not bool or bit for bit in known)
                or type(support) is not list or len(support) != 7
                or any(type(bit) is not bool for bit in support)
                or any(bit and not declared_support[index] for index, bit in enumerate(support))):
            raise ValueError("intervention original actual support/trial-unknown masks differ")
        reason = trial["unknown_reason"]
        if reason is not None:
            proof._text(reason, 256)
        if trial["comparable"] != (reason is None):
            raise ValueError("intervention original comparable/unknown observation differs")
        selected_supported = support[4 + ROUTES.index(trial["recipient"])]
        if ((trial["resolved"] or trial["comparable"] or trial["arm_query"] is not None)
                and (sum(support) < 2 or not selected_supported)):
            raise ValueError("intervention actual arm lacks two original eligible declared actions or selected support")
        if reason == "InsufficientEligibleActions" and (sum(support) >= 2 or trial["arm_query"] is not None):
            raise ValueError("intervention insufficient eligible actions cannot synthesize a supported arm")
        if reason == "UnsupportedInterventionArm" and (sum(support) < 2 or selected_supported or trial["arm_query"] is not None):
            raise ValueError("intervention unsupported original arm cannot become supported")
        proof._cost(trial["cost"], profile)
        if trial["historical_unbound_input"] is not None:
            raise ValueError("intervention/2 cannot import a legacy unbound input projection")
        previous = trial["previous_durable_ack"]
        if index == 0:
            if previous is not None:
                raise ValueError("intervention first trial cannot have a preceding ACK")
        else:
            fields(previous, ("trial_sequence", "trial_sha256", "file_sha256"), "intervention original preceding ACK")
            old_pin = publication["trials"][index - 1]
            if (uint(previous["trial_sequence"], 3) != index
                    or sha(previous["file_sha256"]) != old_pin["sha256"]
                    or sha(previous["trial_sha256"]) != proof._native_digest(TRIAL_DOMAIN, files.raw_file(old_pin["path"]))):
                raise ValueError("intervention original preceding durable ACK causal chain differs")
        materialized_pin = materializations.get(index)
        raw = trial["materialized_json"]
        if raw is None:
            if trial["materialized_sha256"] is not None or materialized_pin is not None or trial["resolved"] or trial["comparable"]:
                raise ValueError("intervention absent original materialization cannot be resolved")
        else:
            if type(raw) is not str or materialized_pin is None:
                raise ValueError("intervention original materialization string/file required")
            raw = raw.encode("utf-8", "strict")
            if (raw != files.raw_file(materialized_pin["path"])
                    or sha(trial["materialized_sha256"]) != proof._native_digest(MATERIALIZATION_DOMAIN, raw)):
                raise ValueError("intervention original embedded/raw/native materialization differs")
            body = fields(proof._decode(raw, MAX_MATERIALIZATION_BYTES), MATERIALIZATION_FIELDS,
                "intervention original materialization/2")
            if body["schema"] != MATERIALIZATION_DOMAIN or any(body[name] != trial[name] for name in (
                    "cohort_id", "trial_sequence", "recipient", "profile_sha256", "source", "original_root",
                    "preparation", "neutral_target", "arm_query")):
                raise ValueError("intervention original materialization/trial linkage differs")
        unresolved = trial["unresolved_source"]
        if unresolved is not None:
            fields(unresolved, ("question_sha256", "frame_sequence", "recipient", "root_budget_sha256", "frozen_context_sha256",
                "reason", "actual_pc_consumptions", "actual_dependencies", "actual_r_pc_consumptions", "new_witnesses"),
                "intervention original unresolved source")
            for name in ("question_sha256", "root_budget_sha256", "frozen_context_sha256"):
                sha(unresolved[name])
            proof._positive(unresolved["frame_sequence"])
            if (unresolved["recipient"] != trial["recipient"] or unresolved["reason"] not in (
                    "cancelled", "deadline", "replaced_root", "semantic_failure", "physical_unknown", "missing_actual_consumption",
                    "missing_exclusive_whole_cost") or trial["resolved"] or trial["comparable"]):
                raise ValueError("intervention unresolved source cannot become a resolved comparable arm")
            for name in ("actual_pc_consumptions", "actual_dependencies", "actual_r_pc_consumptions"):
                uint(unresolved[name])
            for witness in proof._array(unresolved["new_witnesses"], 64):
                fields(witness, ("observation", "identity_sha256", "raw_evidence_sha256"), "intervention unresolved witness")
                uint(witness["observation"])
                sha(witness["identity_sha256"])
                sha(witness["raw_evidence_sha256"])
        trials.append(trial)
    return tuple(trials)


def _original_scope(trial, source, profile):
    root, authority = proof._root_structure(trial["original_root"], source)
    if root["root_state"]["semantic"] != source["rules_root"]["anchor"]["rules_state_sha256"]:
        raise ValueError("intervention checkpoint changed original Rules root")
    preparation, cost = trial["preparation"], trial["cost"]
    # This is the unmodified *neutral source* Ready observation; no requested
    # alternative arm is added to these actual offers or treated as Ready.
    resource_known = proof._prelude(preparation, profile=profile, root=root, authority=authority)
    if (cost["clock_epoch"] != root["deadline_clock"] or cost["original_deadline_tick"] != root["deadline_tick"]
            or cost["started_tick"] < preparation["captured_tick"] or cost["baseline_counters"] != preparation["counters"]
            or cost["native_credit_baseline"] != [preparation["native_credit"]["spent"], preparation["native_credit"]["reserved"]]):
        raise ValueError("intervention original root/preparation/own clock/cost differs")
    declared = root["root_descriptor"]["input"]
    if (proof._value_digest("rz-rovent-utility-cohort-logical-budget/1", declared["declared_ceilings"])
                != source["source_pins"]["logical_budget_sha256"]
            or digest("rz-rovent-resource-plan/1", declared["declared_ceilings"]["resource_plan"])
                != source["source_pins"]["resource_plan_sha256"]
            or cost["cohort_action_cutoff_tick"] > root["deadline_tick"]
            or cost["cohort_cleanup_cutoff_tick"] > declared["clock"]["cleanup"]["at_ns"]):
        raise ValueError("intervention original action exceeds own root ceilings/deadlines")
    return {"root": root, "authority": authority, "resource_known": resource_known}


def _arm(arm, *, trial, original, neutral, source, config):
    fields(arm, ARM_FIELDS, "intervention original role arm query")
    root, authority = original["root"], original["authority"]
    if (sha(arm["neutral_target_sha256"]) != trial["neutral_target_sha256"]
            or arm["recipient"] != trial["recipient"]
            or arm["original_first_move"] != neutral["original_first_move"]
            or uint(arm["deadline_clock"]) != root["deadline_clock"]
            or uint(arm["deadline_tick"]) != root["deadline_tick"]):
        raise ValueError("intervention actual arm changed neutral/role/first move/own deadline")
    proof._positive(arm["frame_sequence"])
    proof._positive(arm["source_frame_sequence"])
    for name in ("arm_query_sha256", "full_view_sha256", "public_context_sha256"):
        sha(arm[name])
    stages = {"route_p": ("ProposalContinuation", "ProposePolicy"),
        "route_c": ("CriticDivergence", "DivergencePolicy"), "route_r": ("CpuValidation", "ProposePolicy")}
    stage, purpose = stages[trial["recipient"]]
    if arm["stage"] != stage or arm["purpose"] != purpose:
        raise ValueError("intervention actual role query stage/purpose differs")
    view = structure._view(arm["view"], authority)
    if (view["root_state"]["semantic_sha256"] != root["root_state"]["semantic"]
            or view["root_state"]["owner"] != root["root_state"]["owner"]
            or view["root_state"]["revision"] != root["root_state"]["revision"]):
        raise ValueError("intervention actual arm view crossed its original Rules root")
    budget = fields(arm["budget"], ("role_submissions_remaining", "original_cpu_credit", "original_store_byte_limit"),
        "intervention original actual arm budget")
    for amount in budget.values():
        uint(amount)
    origins = [offer for offer in trial["preparation"]["offers"]
        if offer["frame_sequence"] == arm["source_frame_sequence"]]
    if len(origins) != 1 or budget != origins[0]["budget"]:
        raise ValueError("intervention arm lacks its original neutral source frame/budget")
    if origins[0] != _source_offer(trial):
        raise ValueError("intervention actual arm changed its original fair neutral source")
    origin = origins[0]
    if (origin["original_first_move"] != arm["original_first_move"]
            or origin["deadline_clock"] != arm["deadline_clock"] or origin["deadline_tick"] != arm["deadline_tick"]
            or origin["view"]["root_state"] != view["root_state"]
            or origin["view"]["context_policy_sha256"] != view["context_policy_sha256"]
            or origin["view"]["selected_records"] != view["selected_records"]
            or origin["view"]["view_digest"] != neutral["original_view_sha256"]
            or origin["full_view_sha256"] != neutral["original_full_view_sha256"]):
        raise ValueError("intervention arm changed original source first move/deadline/context allocation")
    query = fields(arm["query"], ("endpoint_rules", "prefix", "proposal", "counterexample", "legal", "divergences", "question_preimage"),
        "intervention actual role query operands")
    preimage = query["question_preimage"]
    if (question_sha256(preimage) != arm["arm_query_sha256"]
            or preimage["stage_code"] != {"route_p": 2, "route_c": 5, "route_r": 3}[trial["recipient"]]
            or preimage["view_digest"] != arm["full_view_sha256"]
            or preimage["view_seal_digest"] != view["view_digest"]):
        raise ValueError("intervention original question preimage changed actual stage/query view")
    proof._rules_projection(query["endpoint_rules"])
    for name in ("prefix", "proposal", "legal"):
        _moves(query[name], 256)
    neutral_capture = original.get("neutral_input")
    if neutral_capture is None:
        neutral_capture, _ = _neutral(neutral, config=config, source=source)
    target_line = neutral_capture.reference_input.base_input.context_lines[neutral["target_record"]].line.moves
    full_proposal = [move.from_square | move.to_square << 6 | move.promotion << 12 for move in target_line]
    _query_prefix(query, trial["recipient"], neutral["original_first_move"], full_proposal)
    if query["counterexample"] is not None:
        raise ValueError("intervention frozen-proposal arm cannot contain a counterexample")
    divergences = proof._array(query["divergences"], 256)
    for divergence in divergences:
        uint(divergence, 65535)
    if len(set(query["legal"])) != len(query["legal"]) or len(set(divergences)) != len(divergences):
        raise ValueError("intervention actual query candidate/divergence order contains duplicates")
    if trial["recipient"] == "route_c" and query["legal"]:
        raise ValueError("intervention C divergence query cannot import move-policy candidates")
    prepared = arm["prepared_input"]
    if prepared is None:
        if trial["resolved"] or trial["comparable"] or trial["materialized_json"] is not None:
            raise ValueError("intervention original arm preparation unavailable for resolved source")
        return None
    projected = {**prepared, "root_rules": proof._rules_projection(prepared.get("root_rules")),
        "source_rules": [proof._rules_projection(item) for item in proof._array(prepared.get("source_rules"), 64)]}
    captured = structure._input(projected, root, config)
    proof._input_submission_seal(prepared)
    provenance = prepared["provenance"]
    if (prepared["base_identity"] != source["nnuec_registration"]["model_identity"]
            or prepared["root_rules"] != neutral["root_rules"]
            or prepared["records_metadata"] != neutral["records_metadata"]
            or prepared["source_rules"] != neutral["source_rules"]
            or prepared["target_record"] != neutral["target_record"]
            or prepared["target_record_ids"] != neutral["target_record_ids"]
            or provenance["record_scope_sha256"] != neutral["record_scope_sha256"]
            or provenance["sequence"] != view["sequence"] or provenance["view_seal_sha256"] != view["view_digest"]
            or provenance["full_view_sha256"] != arm["full_view_sha256"]
            or provenance["policy_sha256"] != view["context_policy_sha256"]
            or arm["public_context_sha256"] != neutral["record_scope_sha256"]):
        raise ValueError("intervention actual arm preparation crossed neutral records or its own original query view")
    return captured


def _query_prefix(query, recipient, first, full_proposal):
    # /2 P re-proposes from the exact first endpoint; it does not extend an
    # already completed full target at its limit. Neutral/full target stays
    # intact. C starts at root and R verifies the full target endpoint.
    if not full_proposal or full_proposal[0] != first or query["proposal"] != full_proposal:
        raise ValueError("intervention actual query lost its original full Proposal target")
    expected = [first] if recipient == "route_p" else [] if recipient == "route_c" else full_proposal
    if query["prefix"] != expected:
        raise ValueError("intervention actual bounded P/C/R prefix differs from its registered role policy")


def question_sha256(value):
    """Actual private question sequence framing, never coordinator FrameId."""
    fields(value, ("sequence", "stage_code", "view_digest", "view_seal_digest", "focus_prefix_sha256",
        "goal_local_sha256", "legacy_full_records_sha256"), "intervention original question digest preimage")
    framed = (b"rz-pals-adaptive-checked-question/1\0" + uint(value["sequence"]).to_bytes(8, "little")
        + bytes([uint(value["stage_code"], 18)]))
    for name in ("view_digest", "view_seal_digest", "focus_prefix_sha256"):
        framed += bytes.fromhex(sha(value[name]))
    for name, tag in (("goal_local_sha256", b"goal-local-rules-origin\0"),
            ("legacy_full_records_sha256", b"legacy-full-raw-snapshot\0")):
        if value[name] is not None:
            framed += tag + bytes.fromhex(sha(value[name]))
    return hashlib.sha256(framed).hexdigest()


def _materialized_scope(trial, original, neutral, source, config, profile, cohort_id):
    body = proof._decode(trial["materialized_json"].encode("utf-8", "strict"), MAX_MATERIALIZATION_BYTES)
    arm = body["arm_query"]
    captured = _arm(arm, trial=trial, original=original, neutral=neutral, source=source, config=config)
    if captured is None:
        raise ValueError("intervention materialization lacks original actual arm capture")
    root, authority, cost = original["root"], original["authority"], trial["cost"]
    action = fields(body["action"], ("authority", "frame_sequence", "question_sha256", "root_budget_sha256",
        "original_first_move", "deadline_clock", "deadline_tick", "frozen_context_sha256", "cost_epoch_sha256", "cost_scope"),
        "intervention original materialized action")
    structure._authority(action["authority"], authority)
    if (action["frame_sequence"] != arm["frame_sequence"] or sha(action["question_sha256"]) != arm["arm_query_sha256"]
            or sha(action["root_budget_sha256"]) != root["root_budget_sha256"]
            or action["original_first_move"] != arm["original_first_move"]
            or action["deadline_clock"] != arm["deadline_clock"] or action["deadline_tick"] != arm["deadline_tick"]
            or sha(action["frozen_context_sha256"]) != trial["neutral_target_sha256"] or action["cost_scope"] != COST_SCOPE
            or sha(action["cost_epoch_sha256"]) != cost["cost_epoch_sha256"]):
        raise ValueError("intervention actual action/arm/neutral/own source scope differs")
    # A lexical projection for unchanged proof leaves, with the REAL arm
    # input. No /1 schema, profile, Ready offer or source admission is created.
    proof_body = {"semantic": body["semantic"], "action": action, "input": arm["prepared_input"]}
    scope = {**original, "input": captured, "materialization": proof_body, "trial": trial,
        "original_materialization": body, "arm_query": arm}
    scope["witnesses"] = proof._semantic_core(scope)
    limit = arm["budget"]["original_store_byte_limit"]
    scope["witness_bodies_complete"] = proof._witness_bodies(scope, original_store_byte_limit=limit)
    scope["witness_basis_complete"] = proof._witness_basis(scope, original_store_byte_limit=limit)
    return scope


def _pairs(files, trials, scopes, physical):
    publication = files.publication()
    fields_pair = ("schema", "cohort_id", "neutral_target_sha256", "left_trial_sequence", "right_trial_sequence", "left_trial_sha256",
        "right_trial_sha256", "left_recipient", "right_recipient", "preferred", "unknown_reason", "support_mask",
        "target_known_mask", "left_new_witnesses", "right_new_witnesses", "left_whole_wall_ns", "right_whole_wall_ns",
        "cost_scope", "objective")
    known, known_sha, all_sha, seen = [], [], [], set()
    for pin in publication["pairs"]:
        raw = files.raw_file(pin["path"])
        pair = fields(proof._decode(raw, MAX_METADATA_BYTES), fields_pair, "intervention original pair/2")
        if (pair["schema"] != PAIR_DOMAIN or pair["cohort_id"] != publication["cohort_id"]
                or pair["cost_scope"] != COST_SCOPE or pair["objective"] != proof.COST_OBJECTIVE):
            raise ValueError("intervention original pair group/domain/cost differs")
        left_num = proof._positive(pair["left_trial_sequence"], len(trials))
        right_num = proof._positive(pair["right_trial_sequence"], len(trials))
        key = tuple(sorted((left_num, right_num)))
        if key in seen:
            raise ValueError("intervention duplicate unordered actual trial comparison")
        seen.add(key)
        left, right = trials[left_num - 1], trials[right_num - 1]
        if (left_num == right_num or physical[left_num]["authority"] == physical[right_num]["authority"]
                or left["cost"]["clock_epoch"] == right["cost"]["clock_epoch"]
                or pair["left_trial_sha256"] == pair["right_trial_sha256"]
                or left["recipient"] == right["recipient"]):
            raise ValueError("intervention pair repeats an original trial/authority/clock/recipient")
        neutral_left, neutral_right = left["neutral_target_sha256"], right["neutral_target_sha256"]
        if neutral_left is not None and neutral_right is not None and neutral_left != neutral_right:
            raise ValueError("intervention pair original neutral content differs")
        if pair["neutral_target_sha256"] != (neutral_left if neutral_left is not None else neutral_right):
            raise ValueError("intervention pair lost its original optional neutral content seal")
        if pair["neutral_target_sha256"] is not None:
            sha(pair["neutral_target_sha256"])
        left_offer, right_offer = _source_offer(left), _source_offer(right)
        if left_offer["budget"] != right_offer["budget"]:
            raise ValueError("intervention pair changed original neutral source FrameBudget")
        if left_num in scopes and right_num in scopes:
            left_baseline = set(scopes[left_num]["materialization"]["semantic"]["baseline_witnesses"])
            right_baseline = set(scopes[right_num]["materialization"]["semantic"]["baseline_witnesses"])
            if left_baseline != right_baseline:
                raise ValueError("intervention pair changed original semantic baseline witness set")
        for side, number, trial in (("left", left_num, left), ("right", right_num, right)):
            if (sha(pair[side + "_trial_sha256"]) != proof._native_digest(TRIAL_DOMAIN,
                        files.raw_file(publication["trials"][number - 1]["path"]))
                    or pair[side + "_recipient"] != trial["recipient"]
                    or uint(pair[side + "_whole_wall_ns"]) != trial["cost"]["whole_wall_ns"]):
                raise ValueError("intervention pair original native trial/arm/whole cost differs")
            declared = corpus._sha_list(pair[side + "_new_witnesses"], 64, "intervention original pair witness set")
            if number in scopes and declared != sorted(scopes[number]["witnesses"]):
                raise ValueError("intervention pair changed actual conditional witness source set")
        a, b = set(pair["left_new_witnesses"]), set(pair["right_new_witnesses"])
        preferred, reason = None, None
        if (not left["comparable"] or not right["comparable"]
                or neutral_left is None or neutral_right is None):
            reason = "unresolved_or_masked_trial"
        elif not a and not b:
            reason = "no_new_valid_witness"
        elif a == b and pair["left_whole_wall_ns"] == pair["right_whole_wall_ns"]:
            reason = "equal_observation"
        elif a.issuperset(b) and pair["left_whole_wall_ns"] <= pair["right_whole_wall_ns"]:
            preferred = left["recipient"]
        elif b.issuperset(a) and pair["right_whole_wall_ns"] <= pair["left_whole_wall_ns"]:
            preferred = right["recipient"]
        else:
            reason = "pareto_incomparable"
        support = [a or b for a, b in zip(left["support_mask"], right["support_mask"], strict=True)]
        target = [False] * 7
        if preferred is not None:
            target[4 + ROUTES.index(left["recipient"])] = True
            target[4 + ROUTES.index(right["recipient"])] = True
            if left_num not in scopes or right_num not in scopes:
                raise ValueError("intervention known pair lacks actual materialized arm sources")
        for name in ("support_mask", "target_known_mask"):
            bits = pair[name]
            if type(bits) is not list or len(bits) != 7 or any(type(bit) is not bool for bit in bits):
                raise ValueError("intervention exact original pair mask required")
        if (pair["preferred"] != preferred or pair["unknown_reason"] != reason
                or pair["support_mask"] != support or pair["target_known_mask"] != target):
            raise ValueError("intervention original Pareto/unknown/mask arithmetic differs")
        native = proof._native_digest(PAIR_DOMAIN, raw)
        all_sha.append(native)
        if preferred is not None:
            other = right["recipient"] if preferred == left["recipient"] else left["recipient"]
            known.append([4 + ROUTES.index(preferred), 4 + ROUTES.index(other)])
            known_sha.append(native)
    return known, known_sha, all_sha


def _source_offer(trial):
    """The producer's original fair neutral source, never an arm offer."""
    offers = trial["preparation"]["offers"]
    for route in ("route_p", "route_r", "route_c"):
        selected = [offer for offer in offers if offer["recipient"] == route]
        if selected:
            if trial["arm_query"] is not None and trial["arm_query"]["source_frame_sequence"] != selected[0]["frame_sequence"]:
                raise ValueError("intervention actual arm lost its original fair neutral source offer")
            return selected[0]
    raise ValueError("intervention original neutral source offer unavailable")


def validate_intervention_payloads(files, *, registration):
    """Return saved data candidates only; current Rust authority stays closed."""
    if type(files) is not HistoricalInterventionFiles:
        raise ValueError("intervention original strict pinned file owner required")
    profile = validate_intervention_profile(proof._decode(files.raw_file("cohort-profile.json"), MAX_METADATA_BYTES))
    acknowledgements = validate_intervention_trial_acknowledgements(files)
    trials = _trials(files, profile)
    # This separate preparation field is only the first original trial's
    # cost-epoch anchor. Other processes retain their own actual epochs.
    if trials[0]["cost"]["cost_epoch_sha256"] != registration["cost_epoch_sha256"]:
        raise ValueError("intervention first_trial_cost_epoch_anchor differs from original first trial")
    source, lock, config = _source_enrollment(files, profile, registration)
    frozen_pc = proof._source_pc_models(source, lock)
    registration_root, registration_authority = proof._root_structure(source["registration_root"], source)
    if (registration_root["native_origin"] != source["native_origin"]
            or registration_root["root_state"]["semantic"] != source["rules_root"]["anchor"]["rules_state_sha256"]):
        raise ValueError("intervention original registration source/Rules differs")
    expected_owner = proof._value_digest("rz-rovent-utility-intervention-owner/2", {
        "profile": intervention_profile_sha256(profile), "original_root": registration_root["root_descriptor_sha256"],
        "source": source["source_pins"]})
    if expected_owner != files.publication()["cohort_id"]:
        raise ValueError("intervention original cohort registration root/profile/source seal differs")
    final_neutral = proof._decode(files.raw_file("neutral-target.json"), MAX_MATERIALIZATION_BYTES, allow_null=True)
    if final_neutral is None:
        if any(trial["neutral_target"] is not None for trial in trials):
            raise ValueError("intervention final neutral file lost an original submitted target")
        raise UnsupportedInterventionPayload("intervention_has_no_original_neutral_target")
    captured, common_target = _neutral(final_neutral, config=config, source=source)
    if final_neutral["target_record"] != profile["target_record"]:
        raise ValueError("intervention common original target differs from registered record")
    originals, scopes, physical, role_queries = {}, {}, {}, []
    original_support = None
    for trial, ack in zip(trials, acknowledgements, strict=True):
        number = trial["trial_sequence"]
        original = originals[number] = _original_scope(trial, source, profile)
        neutral = trial["neutral_target"]
        if neutral is not None:
            own_input, own_target = _neutral(neutral, config=config, source=source)
            original["neutral_input"] = own_input
            if (own_target != common_target or sha(trial["neutral_target_sha256"]) != own_target
                    or own_input.input_bytes != captured.input_bytes
                    or neutral["model_identity"] != final_neutral["model_identity"]):
                raise ValueError("intervention independent trials changed common neutral input/Rules/target/model")
            if original_support is None:
                original_support = trial["support_mask"]
            elif trial["support_mask"] != original_support:
                raise ValueError("intervention same neutral content changed actual supported declared actions")
        elif (trial["neutral_target_sha256"] is not None or trial["arm_query"] is not None
                or trial["materialized_json"] is not None or trial["resolved"] or trial["comparable"]):
            raise ValueError("intervention unbound trial cannot synthesize an original neutral cap")
        arm = trial["arm_query"]
        if arm is not None:
            _arm(arm, trial=trial, original=original, neutral=neutral, source=source, config=config)
            role_queries.append({"trial_sequence": number, "recipient": trial["recipient"],
                "question_sha256": arm["arm_query_sha256"], "view_sha256": arm["view"]["view_digest"],
                "full_view_sha256": arm["full_view_sha256"]})
        elif trial["resolved"] or trial["comparable"] or trial["materialized_json"] is not None:
            raise ValueError("intervention resolved trial lacks its original actual role query")
        if trial["materialized_json"] is not None:
            scopes[number] = _materialized_scope(trial, original, neutral, source, config, profile,
                files.publication()["cohort_id"])
        physical[number] = proof._physical_closure(trial, source, root=original["root"], authority=original["authority"])
        interval = ack["collection_io"]
        if interval["clock_epoch"] != trial["cost"]["clock_epoch"] or interval["started_tick_ns"] < trial["cost"]["physical_pin_zero_tick"]:
            raise ValueError("intervention durable ACK precedes its own original physical pin-zero")
        if trial["comparable"]:
            scope, cost = scopes.get(number), trial["cost"]
            unknown = ("nn_input_count_unknown", "accepted_nn_input_count_unknown", "consumed_nn_input_count_unknown", "rejected_events")
            if (profile["input"]["usage"] != "collection_candidate" or not trial["resolved"] or scope is None
                    or not cost["complete_scope"] or not scope["witness_bodies_complete"] or not scope["witness_basis_complete"]
                    or not original["resource_known"] or not physical[number]["cache_known"] or not physical[number]["fresh_product"]
                    or cost["physical_pin_zero_tick"] > cost["cohort_cleanup_cutoff_tick"]
                    or any(cost["final_counters"][name] != cost["baseline_counters"][name] for name in unknown)):
                raise ValueError("intervention comparable arm lacks original consumed/proof/known whole scope")
    identities = [tuple(physical[number]["authority"][name] for name in (
        "process_epoch", "game_generation", "root_generation", "implementation_sha256")) for number in physical]
    if len(set(identities)) != len(identities) or physical[1]["authority"] != registration_authority:
        raise ValueError("intervention independent root authority reused or original registration retagged")
    final_io = files.durable_ack()["collection_io"]
    if (final_io["clock_epoch"] != trials[-1]["cost"]["clock_epoch"]
            or final_io["started_tick_ns"] < acknowledgements[-1]["collection_io"]["finished_tick_ns"]):
        raise ValueError("intervention final publisher did not follow its own final trial ACK")
    known, known_sha, all_sha = _pairs(files, trials, scopes, physical)
    metadata, manifest = source["collection_metadata"], source["collection_manifest"]
    report = {"schema": "rz-rovent-utility-intervention-payload-observation/1", "cohort_id": files.publication()["cohort_id"],
        "profile_sha256": intervention_profile_sha256(profile), "usage": profile["input"]["usage"],
        "neutral_input_sha256": captured.input_sha256, "neutral_target_sha256": common_target,
        "neutral_context_sha256": captured.context_sha256, "root_rules_sha256": final_neutral["root_rules"]["semantic_sha256"],
        "role_queries": role_queries, "known_pairs": known, "known_pair_sha256": known_sha, "all_pair_sha256": all_sha,
        "frozen_pc_sources": frozen_pc, "ancestor_datasets": copy.deepcopy(source["ancestor_datasets"]),
        "split": manifest["opening_splits"][metadata["opening_id"]], "lineage": copy.deepcopy(metadata),
        "transposition_sha256": source["transposition_sha256"], "native_authority_issued": False,
        "current_ack_issued": False, "physical_completion_verified": False, "rules_verified": False,
        "conditional_comparison_verified": False, "training_target_authorized": False, "execution_authorized": False}
    return {"input": captured, "report": report, "source": source, "profile": profile,
        "trials": trials, "original_files": files}
