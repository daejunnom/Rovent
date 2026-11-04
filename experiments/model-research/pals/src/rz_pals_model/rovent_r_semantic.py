"""Separate historical R Job/slice semantic observations.

The original R record/publication remains unchanged, including its unknown
semantic-consumption and whole-action-cost fields. The Rust producer retains
the actual CpuSemanticWitness, Store and attested physical slice. This module
joins their persisted observations to an already read historical R corpus;
matching JSON cannot issue those capabilities or any new training target.
"""
from __future__ import annotations

import copy
import json

from .generation_data import digest, fields, sha, uint
from .rovent_corpus import _decode_json
from .rovent_historical_r_corpus import HistoricalRoventRCorpus, _retirement_body
from . import rovent_r_corpus as r


DOMAIN = "rz-rovent-r-semantic-observations/1"
TASK_DOMAIN = "rz-rovent-r-semantic-task/1"
OBSERVATION_DOMAIN = "rz-rovent-r-semantic-data-observation/1"
MAX_BYTES = 4 << 20
MAX_TASK_BYTES = 64 << 10
_U64 = 2**64 - 1
_STAGES = ("raw_published", "search_feedback_consumed", "selected_pc_view_consumed")
_TOP = ("schema", "r_publication_file_sha256", "native_admission_sha256", "authority", "retirements", "observations",
    "unrecorded_witness_attempts", "manager_failures", "per_readout_consumption", "whole_action_cost", "wdl_target",
    "utility_target", "semantic_sha256")
_OBSERVATION = ("retirement_sha256", "stage", "native_execution_epoch", "native_execution_sequence", "store_execution",
    "observation", "task_key", "task_key_sha256", "job_sequence", "worker_owner", "frame_sequence", "capture_sequence",
    "question_sha256", "binding_sha256", "submitted_view_sha256", "submitted_full_view_sha256", "native_slice_work", "selected_pc")
_SELECTED = ("request_epoch", "request_sequence", "question_sha256", "view_sha256", "full_view_sha256", "output_sha256",
    "record_revision", "record_origin_state")
_TASK = ("state", "line", "question", "root_moves", "model", "epoch", "value_identity", "checker_identity", "cpu_condition",
    "profile", "condition", "input_revision", "requested_depth", "node_budget")


def _encoded(value, maximum, description):
    try:
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, OverflowError, UnicodeError, RecursionError) as error:
        raise ValueError(f"{description} must be bounded plain JSON") from error
    if not 0 < len(raw) <= maximum:
        raise ValueError(f"{description} byte cap exceeded")
    return raw


def _list(value, maximum, description):
    if type(value) is not list or len(value) > maximum:
        raise ValueError(f"bounded {description} list required")
    return value


def _sha_list(value, maximum, description):
    result = _list(value, maximum, description)
    for identity in result:
        sha(identity)
    if result != sorted(set(result)):
        raise ValueError(f"{description} must preserve its sorted unique original SHA list")
    return result


def _authority(value):
    fields(value, ("process_epoch", "game_generation", "root_generation", "implementation_sha256"), "R semantic authority")
    if not uint(value["process_epoch"]):
        raise ValueError("R semantic positive original process epoch required")
    uint(value["game_generation"])
    uint(value["root_generation"])
    sha(value["implementation_sha256"])


def _question(value):
    # Rust's externally tagged TaskQuestion enum roundtrips these exact shapes.
    if type(value) is str:
        if value in ("AnalyzePosition", "AnalyzeRootMoves", "ModelProposal", "ModelRefutation", "ModelRepair"):
            return
        raise ValueError("R semantic unknown original TaskQuestion enum")
    if type(value) is not dict or len(value) != 1:
        raise ValueError("R semantic exact original TaskQuestion enum required")
    if set(value) == {"FindAlternative"}:
        part = fields(value["FindAlternative"], ("divergence_ply",), "R semantic alternative task question")
        uint(part["divergence_ply"], 65535)
        return
    if set(value) == {"ProfiledExternal"}:
        part = fields(value["ProfiledExternal"], ("requested",), "R semantic profiled task question")
        requested = part["requested"]
        kind = requested.get("kind") if type(requested) is dict else None
        roster = {"combined": ("depth", "reported_nodes"), "nodes": ("reported_nodes",),
            "depth": ("depth",), "move_time": ("milliseconds",)}.get(kind)
        if roster is None:
            raise ValueError("R semantic exact profiled task constraint required")
        fields(requested, ("kind", *roster), "R semantic original profiled constraint")
        for name in roster:
            uint(requested[name], 65535 if name == "depth" else _U64)
        return
    raise ValueError("R semantic unknown original TaskQuestion enum")


def _task(value, expected_sha, job, model_identity):
    fields(value, _TASK, "R semantic original TaskKey")
    _encoded(value, MAX_TASK_BYTES, "R semantic original TaskKey")
    for name in ("state", "model", "epoch", "profile", "condition", "input_revision", "node_budget"):
        uint(value[name])
    uint(value["requested_depth"], 65535)
    if value["line"] is not None:
        uint(value["line"])
    _question(value["question"])
    roots = _list(value["root_moves"], MAX_TASK_BYTES // 2, "R semantic original TaskKey root move")
    for movement in roots:
        # Move16 is a transparent u16 here; Rules owns move legality.
        uint(movement, 65535)
    identity = fields(value["value_identity"], ("semantics", "weights_sha256", "training"), "R semantic original value identity")
    if type(identity["semantics"]) is not str:
        raise ValueError("R semantic original value semantics must be text")
    if identity["weights_sha256"] is not None:
        sha(identity["weights_sha256"])
    # Actual captures are the original NNUEC source; there is no bootstrap or
    # external-checker conversion and a learned declaration proves no training.
    from .nnuec_reference import _training
    _training(identity["training"])
    condition = value["cpu_condition"]
    if condition is not None and type(condition) is not str:
        raise ValueError("R semantic original CPU condition must be nullable text")
    if (digest(TASK_DOMAIN, value) != sha(expected_sha)
            or identity != model_identity or value["node_budget"] != job["total_native_node_credit"]
            or value["checker_identity"] is not None or value["line"] is not None
            or value["model"] != 0 or value["epoch"] != 0):
        raise ValueError("R semantic exact original TaskKey/model/Job credit join differs")


def _retirements(value, source, publication, admission_sha, authority):
    declarations = {item["retirement_sha256"]: item for item in publication["retirements"]}
    parts = _list(value, 64, "R semantic retirement")
    if len(parts) != len(declarations):
        raise ValueError("R semantic complete publication retirement coverage differs")
    terminals, records, empty = {}, {}, 0
    for terminal in parts:
        fields(terminal, ("retirement_sha256", "retirement", "readout_record_sha256"), "R semantic original retirement")
        identity = sha(terminal["retirement_sha256"])
        declared = declarations.get(identity)
        if declared is None or identity in terminals:
            raise ValueError("R semantic original retirement is absent or duplicated")
        readouts = _sha_list(terminal["readout_record_sha256"], 64, "R semantic retained readout")
        original = source.original_retirement_evidence(identity)["observation"]
        _retirement_body(terminal["retirement"], identity, readouts, original["native_admission"])
        if (terminal["retirement"] != original["retirement"]
                or readouts != original["readout_record_sha256"] or readouts != declared["readout_record_sha256"]
                or original["native_admission_sha256"] != admission_sha
                or terminal["retirement"]["job"]["authority"] != authority):
            raise ValueError("R semantic whole retirement/readout/original admission differs")
        terminals[identity] = terminal
        empty += not readouts
        for record_sha in readouts:
            if record_sha in records:
                raise ValueError("R semantic original readout occurs in multiple retirements")
            evidence = source.corpus.original_semantic_evidence(record_sha)
            if (evidence["native_admission_sha256"] != admission_sha or evidence["retirement_sha256"] != identity
                    or evidence["retirement"] != terminal["retirement"]
                    or evidence["semantic_consumption"] is not None or evidence["whole_action_cost"] is not None):
                raise ValueError("R semantic readout changed its original unknown retirement scope")
            records[record_sha] = evidence
    if set(terminals) != set(declarations) or len(records) != publication["record_count"]:
        raise ValueError("R semantic full original publication readout coverage differs")
    return terminals, records, empty


def _selected(value, epoch):
    fields(value, _SELECTED, "R semantic selected P/C original record scope")
    for name in ("request_epoch", "request_sequence", "record_revision", "record_origin_state"):
        uint(value[name])
    if value["request_epoch"] != epoch or value["request_sequence"] == 0:
        raise ValueError("R semantic selected P/C original request authority differs")
    for name in ("question_sha256", "view_sha256", "full_view_sha256", "output_sha256"):
        sha(value[name])


def _observations(value, terminals, records, authority, model_identity):
    executions, seen = {}, set()
    stage_counts = dict.fromkeys(_STAGES, 0)
    records_by_retirement = {}
    for evidence in records.values():
        records_by_retirement.setdefault(evidence["retirement_sha256"], []).append(evidence)
    for observation in _list(value, 256, "R semantic observation"):
        fields(observation, _OBSERVATION, "R semantic original observation")
        retirement_sha = sha(observation["retirement_sha256"])
        terminal = terminals.get(retirement_sha)
        if terminal is None:
            raise ValueError("R semantic observation has no original retirement")
        retired, job = terminal["retirement"], terminal["retirement"]["job"]
        for name in ("native_execution_epoch", "native_execution_sequence", "store_execution", "observation", "job_sequence",
                "worker_owner", "frame_sequence", "capture_sequence"):
            uint(observation[name], 65535 if name == "worker_owner" else _U64)
        for name in ("question_sha256", "binding_sha256", "submitted_view_sha256", "submitted_full_view_sha256"):
            sha(observation[name])
        _task(observation["task_key"], observation["task_key_sha256"], job, model_identity)
        work = fields(observation["native_slice_work"], ("nodes", "quiescence_nodes", "tt_hits"), "R semantic attested slice work")
        for count in work.values():
            uint(count)
        if (retired["ending"]["kind"] == "not_started"
                or observation["native_execution_epoch"] != authority["process_epoch"]
                or observation["native_execution_sequence"] == 0
                or observation["job_sequence"] != job["job_sequence"]
                or any(observation[name] != retired[name] for name in ("worker_owner", "frame_sequence", "capture_sequence"))
                or observation["question_sha256"] != job["question_sha256"]
                or observation["binding_sha256"] != job["evaluation_namespace_sha256"]
                or work["quiescence_nodes"] > work["nodes"] or work["nodes"] > job["total_native_node_credit"]):
            raise ValueError("R semantic original Job/frame/capture/physical execution/work join differs")
        for evidence in records_by_retirement.get(retirement_sha, ()):
            if (observation["submitted_view_sha256"] != evidence["bound"]["view_seal_sha256"]
                    or observation["submitted_full_view_sha256"] != evidence["bound"]["full_view_sha256"]):
                raise ValueError("R semantic submitted view differs from its original readout Bound")
        key = (observation["native_execution_epoch"], observation["native_execution_sequence"])
        invariant = {name: observation[name] for name in ("retirement_sha256", "store_execution", "observation", "task_key",
            "task_key_sha256", "submitted_view_sha256", "submitted_full_view_sha256", "native_slice_work")}
        first, stages = executions.get(key, (invariant, 0))
        if invariant != first:
            raise ValueError("R semantic stages changed the same native execution evidence or slice work")
        stage = observation["stage"]
        if type(stage) is not str or stage not in _STAGES:
            raise ValueError("R semantic explicit original stage required")
        selected = observation["selected_pc"]
        if stage == "raw_published":
            if stages != 0 or selected is not None:
                raise ValueError("R semantic RawPublished must be first with no selected P/C claim")
            bit = 1
        elif stage == "search_feedback_consumed":
            if stages != 1 or selected is not None:
                raise ValueError("R semantic feedback requires the earlier raw fact only")
            bit = 2
        else:
            if not stages & 2 or selected is None:
                raise ValueError("R semantic selected P/C requires original feedback and selected record scope")
            _selected(selected, authority["process_epoch"])
            bit = 4
        selected_id = None if selected is None else (selected["request_epoch"], selected["request_sequence"])
        occurrence = (key, bit, selected_id)
        if occurrence in seen:
            raise ValueError("duplicate R semantic original occurrence")
        seen.add(occurrence)
        executions[key] = (first, stages | bit)
        stage_counts[stage] += 1
    return len(executions), stage_counts


class RoventRSemanticObservation:
    """Metadata joined to an original historical source, with no live rights."""
    def __init__(self):
        raise TypeError("use validate_r_semantic_observations")

    def _check(self):
        self._source.observation()
        if (self._value["semantic_sha256"] != self._semantic_sha256
                or digest(DOMAIN, {**self._value, "semantic_sha256": ""}) != self._semantic_sha256):
            raise ValueError("R semantic retained observation changed")

    def to_mapping(self):
        self._check()
        return copy.deepcopy(self._value)

    def observation(self):
        self._check()
        return {"schema": OBSERVATION_DOMAIN, "r_publication_file_sha256": self._value["r_publication_file_sha256"],
            "semantic_sha256": self._semantic_sha256, "retirement_count": len(self._value["retirements"]),
            "empty_retirement_count": self._empty, "retained_readout_reference_count": self._readout_count,
            "distinct_attested_execution_count": self._execution_count, "stages": dict(self._stage_counts),
            "unrecorded_witness_attempts": self._value["unrecorded_witness_attempts"],
            "manager_failure_count": len(self._value["manager_failures"]), "per_readout_consumption": False,
            "whole_action_cost": None, "wdl_target": None, "utility_target": None,
            "historical_data_integrity_only": True, "original_native_authority_reissued": False,
            "past_physical_completion_reissued": False, "actual_rules_replay_verified": False,
            "execution_authorized": False, "training_target_authorized": False}


def validate_r_semantic_observations(value, *, source):
    """Validate a separate supplement without changing the original R corpus."""
    if type(source) is not HistoricalRoventRCorpus:
        raise ValueError("original typed historical R corpus required for semantic observations")
    fields(value, _TOP, "R semantic supplement")
    _encoded(value, MAX_BYTES, "R semantic supplement")
    if (value["schema"] != DOMAIN or type(value["per_readout_consumption"]) is not bool
            or value["per_readout_consumption"] or any(value[name] is not None for name in ("whole_action_cost", "wdl_target", "utility_target"))):
        raise ValueError("R semantic explicit domain/unknown target and per-readout scope required")
    sha(value["r_publication_file_sha256"])
    admission_sha = sha(value["native_admission_sha256"])
    _authority(value["authority"])
    uint(value["unrecorded_witness_attempts"])
    original = source.original_publication_metadata()
    publication = original["publication"]
    if original["publication_file"]["sha256"] != value["r_publication_file_sha256"]:
        raise ValueError("R semantic raw publication file pin differs from original source")
    admission = next((item for item in publication["admissions"] if item["native_admission_sha256"] == admission_sha), None)
    if admission is None:
        raise ValueError("R semantic admission is absent from original publication")
    model_identity = admission["nnuec_registration"]["model_identity"]
    terminals, records, empty = _retirements(value["retirements"], source, publication, admission_sha, value["authority"])
    execution_count, stage_counts = _observations(value["observations"], terminals, records, value["authority"], model_identity)
    for failure in _list(value["manager_failures"], 64, "R semantic manager failure"):
        fields(failure, ("stage", "code", "detail"), "R semantic original manager failure")
        if failure["stage"] not in ("Dispatch", "NativeEvent", "Accepted", "Abandoned", "Settled"):
            raise ValueError("R semantic original manager failure stage differs")
        for name, maximum, empty_allowed in (("code", 128, False), ("detail", 2048, True)):
            text = failure[name]
            if type(text) is not str or not empty_allowed and not text or len(text.encode("utf-8")) > maximum:
                raise ValueError("R semantic bounded original manager failure text required")
    expected = sha(value["semantic_sha256"])
    if digest(DOMAIN, {**value, "semantic_sha256": ""}) != expected:
        raise ValueError("R semantic original supplement digest mismatch")
    result = RoventRSemanticObservation.__new__(RoventRSemanticObservation)
    result._source, result._value, result._semantic_sha256 = source, copy.deepcopy(value), expected
    result._empty, result._readout_count = empty, len(records)
    result._execution_count, result._stage_counts = execution_count, tuple(stage_counts.items())
    return result


def parse_r_semantic_observations(input_json, *, source):
    """Bounded original strict JSON entry; source files are never reopened here."""
    if type(input_json) is not str:
        raise ValueError("R semantic original JSON text required")
    if not 0 < len(input_json) <= MAX_BYTES:
        raise ValueError("R semantic original JSON byte cap exceeded")
    try:
        raw = input_json.encode("utf-8")
    except UnicodeError as error:
        raise ValueError("R semantic original JSON must be UTF8") from error
    if not 0 < len(raw) <= MAX_BYTES:
        raise ValueError("R semantic original JSON byte cap exceeded")
    return validate_r_semantic_observations(_decode_json(raw), source=source)
