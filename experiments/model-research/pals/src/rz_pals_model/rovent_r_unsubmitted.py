"""Detached historical observations of original R unsubmitted/NotStarted custody.

The Rust producer requires original manager post-fence custody or the original
worker terminal. This JSON consumer checks persisted fields and seals only. In
particular an absent Job is never a fabricated Job, zero usage, input row,
physical completion, Rules proof, or training source. A publication-free root
retains its unverified asset declarations without opening the original paths.
"""
from __future__ import annotations

import copy
import json

from .generation_data import digest, fields, identifier, sha, uint
from .rovent_corpus import _decode_json
from .rovent_historical_corpus import _input_lock, _source_aggregate
from .rovent_historical_r_corpus import HistoricalRoventRCorpus, _nnuec_pin, _retirement_body
from .rovent_native_v2 import _commit
from . import rovent_r_corpus as r


DOMAIN = "rz-rovent-r-unsubmitted-observations/1"
OBSERVATION_DOMAIN = "rz-rovent-r-unsubmitted-data-observation/1"
MAX_BYTES = 1 << 20
MAX_LOCK_BYTES = 16 << 20
_TOP = ("schema", "native_admission", "native_admission_sha256", "input_lock_sha256", "nnuec_weights_raw_sha256",
    "nnuec_parameter_sha256", "authority", "r_publication_file_sha256", "observations", "per_readout_consumption",
    "native_node_usage", "whole_action_cost", "wdl_target", "utility_target", "observations_sha256")
_ITEM = ("kind", "frame_sequence", "question_sha256", "full_view_sha256", "view_seal_sha256", "binding_sha256",
    "capture_sequence", "original_deadline_clock_epoch", "original_deadline_ns", "job", "worker_owner",
    "retirement_sha256", "retirement", "activation_observed")
_ROOT = ("schema", "native_origin", "native_origin_sha256", "process_epoch", "game_generation", "root_generation",
    "implementation_sha256", "deadline_clock", "deadline_tick", "root_budget_sha256", "selective_policy_sha256", "root_state",
    "store_root_state", "root_moves", "depth_plies", "mate_moves", "requested_nodes", "root_descriptor", "root_descriptor_sha256")
_ORIGIN = ("schema", "purpose", "process_epoch", "mode", "input_lock_sha256", "source_commit", "source_components",
    "source_sha256", "weights_sha256", "export_manifest_sha256", "backend_sha256", "runtime_library_sha256", "runtime_bundle_sha256",
    "native_configuration_sha256", "model_semantics", "model_semantics_sha256", "encoding", "encoding_sha256", "encoding_registry_sha256",
    "model_implementation_sha256", "frozen_epoch", "representation_model_domain", "runtime_sha256_domain", "runtime_sha256", "teacher_profiles")
_NO_TARGET = ("native_node_usage", "whole_action_cost", "wdl_target", "utility_target")


def _encoded(value, maximum, description):
    try:
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, OverflowError, UnicodeError, RecursionError) as error:
        raise ValueError(f"{description} must be bounded plain JSON") from error
    if not 0 < len(raw) <= maximum:
        raise ValueError(f"{description} byte cap exceeded")
    return raw


def _authority(value):
    fields(value, ("process_epoch", "game_generation", "root_generation", "implementation_sha256"), "R unsubmitted authority")
    r._positive(value["process_epoch"])
    uint(value["game_generation"])
    uint(value["root_generation"])
    sha(value["implementation_sha256"])
    return value


def _admission(value, lock, expected_sha, authority, weights_raw_sha):
    """Root/lock declaration joins; no raw asset, runtime or Rules verification."""
    fields(value, _ROOT, "R unsubmitted original root admission")
    if (value["schema"] != r.ROOT_DOMAIN or digest(r.ROOT_DOMAIN, value) != sha(expected_sha)
            or lock["manifest"]["purpose"] != "collection"):
        raise ValueError("R unsubmitted explicit original root domain/seal differs")
    inp = r._root_descriptor(value["root_descriptor"], value["root_descriptor_sha256"], lock)
    for name, key in (("process_epoch", "epoch"), ("game_generation", "game"), ("root_generation", "root")):
        if uint(value[name]) != authority[name] or value[name] != inp["authority"][key]:
            raise ValueError("R unsubmitted original root authority differs")
    state = fields(value["root_state"], ("owner", "revision", "semantic"), "R unsubmitted original Rules state declaration")
    uint(state["owner"])
    uint(state["revision"])
    sha(state["semantic"])
    uint(value["store_root_state"])
    sha(value["selective_policy_sha256"])
    if (value["implementation_sha256"] != authority["implementation_sha256"]
            or value["implementation_sha256"] != inp["authority"]["implementation_sha256"]
            or value["root_budget_sha256"] != value["root_descriptor"]["root_budget_sha256"]
            or uint(value["deadline_clock"]) != inp["clock"]["hard"]["clock_epoch"]
            or uint(value["deadline_tick"]) != inp["clock"]["hard"]["at_ns"]
            or state["semantic"] != inp["state_and_history_sha256"] or value["root_moves"] != inp["allowed_root_moves"]
            or any(type(value[key]) is not type(inp["original_go"][key]) or value[key] != inp["original_go"][key]
                for key in ("depth_plies", "mate_moves", "requested_nodes"))):
        raise ValueError("R unsubmitted original root go/deadline/state differs")
    origin = fields(value["native_origin"], _ORIGIN, "R unsubmitted original native origin")
    if (origin["schema"] != r.ORIGIN_DOMAIN or origin["purpose"] != "collection"
            or digest(r.ORIGIN_DOMAIN, origin) != sha(value["native_origin_sha256"])
            or uint(origin["process_epoch"]) != authority["process_epoch"]):
        raise ValueError("R unsubmitted original native origin domain/seal differs")
    for name in ("input_lock_sha256", "source_sha256", "weights_sha256", "export_manifest_sha256", "backend_sha256",
            "runtime_library_sha256", "native_configuration_sha256", "model_semantics_sha256", "encoding_sha256",
            "encoding_registry_sha256", "model_implementation_sha256", "runtime_sha256"):
        sha(origin[name])
    if origin["runtime_bundle_sha256"] is not None:
        sha(origin["runtime_bundle_sha256"])
    for name in ("model_semantics", "encoding"):
        identifier(origin[name])
    r._positive(origin["frozen_epoch"])
    _commit(origin["source_commit"])
    engine = lock["manifest"]["engine"]
    nnuec = _nnuec_pin(engine.get("nnuec"))
    if (origin["input_lock_sha256"] != lock["canonical_sha256"] or inp["startup_lock_sha256"] != lock["canonical_sha256"]
            or origin["source_commit"] != engine.get("source_commit") or origin["source_components"] != engine.get("source_components")
            or _source_aggregate(origin["source_commit"], origin["source_components"]) != origin["source_sha256"]
            or origin["runtime_sha256_domain"] != "loaded_runtime_library_file_sha256"
            or origin["runtime_sha256"] != origin["runtime_library_sha256"]
            or origin["representation_model_domain"] != "loaded_checkpoint_sha256"
            or origin["teacher_profiles"] != engine.get("teachers")
            or origin["mode"] not in ("HybridPcr", "CpuPcr")
            or engine.get("mode") != {"HybridPcr": "hybrid_pcr", "CpuPcr": "cpu_pcr"}[origin["mode"]]
            or nnuec["weights"]["sha256"] != weights_raw_sha):
        raise ValueError("R unsubmitted original source/runtime/NNUEC lock declarations differ")
    return nnuec


def _publication(source, value, nnuec, lock):
    if source is None:
        if value["r_publication_file_sha256"] is not None:
            raise ValueError("R unsubmitted original publication requires its typed historical owner")
        return None
    if type(source) is not HistoricalRoventRCorpus:
        raise ValueError("R unsubmitted exact historical R owner required")
    original = source.original_publication_metadata()
    publication = original["publication"]
    if original["publication_file"]["sha256"] != value["r_publication_file_sha256"]:
        raise ValueError("R unsubmitted original publication raw SHA differs")
    selected = [item for item in publication["admissions"] if item["native_admission_sha256"] == value["native_admission_sha256"]]
    if len(selected) != 1:
        raise ValueError("R unsubmitted original publication admission is absent")
    item = selected[0]
    if (item["canonical_input_lock_sha256"] != lock["canonical_sha256"]
            or item["nnuec_parameter_sha256"] != value["nnuec_parameter_sha256"]
            or any(item[name][key] != nnuec[decl][key] for name, decl in (("nnuec_weights", "weights"), ("nnuec_provenance", "provenance"))
                for key in ("sha256", "bytes"))):
        raise ValueError("R unsubmitted original publication NNUEC/raw weights/parameter declarations differ")
    return publication


def _items(values, value, source, publication):
    if type(values) is not list or not 1 <= len(values) <= 64:
        raise ValueError("bounded nonempty R unsubmitted observation list required")
    sequences, counts = set(), {"unsubmitted": 0, "not_started": 0}
    root, authority = value["native_admission"], value["authority"]
    for item in values:
        fields(item, _ITEM, "R unsubmitted original observation")
        if item["kind"] not in counts or type(item["activation_observed"]) is not bool or item["activation_observed"]:
            raise ValueError("R unsubmitted explicit kind and activation-free observation required")
        r._positive(item["frame_sequence"])
        capture = r._positive(item["capture_sequence"])
        if capture in sequences:
            raise ValueError("R unsubmitted original capture sequence repeated")
        sequences.add(capture)
        for name in ("question_sha256", "full_view_sha256", "view_seal_sha256", "binding_sha256"):
            sha(item[name])
        if (uint(item["original_deadline_clock_epoch"]) != authority["process_epoch"]
                or item["original_deadline_clock_epoch"] != root["deadline_clock"]
                or uint(item["original_deadline_ns"]) > root["deadline_tick"]):
            raise ValueError("R unsubmitted original reservation deadline differs")
        if item["kind"] == "unsubmitted":
            if any(item[key] is not None for key in ("job", "worker_owner", "retirement_sha256", "retirement")):
                raise ValueError("unsubmitted absence cannot invent a Job/worker/terminal")
        else:
            if source is None or publication is None:
                raise ValueError("NotStarted requires its original typed historical R publication")
            job = r._job(item["job"])
            owner = uint(item["worker_owner"], 65535)
            identity = sha(item["retirement_sha256"])
            terminal = item["retirement"]
            _retirement_body(terminal, identity, [], root)
            if (job["authority"] != authority or job["question_sha256"] != item["question_sha256"]
                    or job["evaluation_namespace_sha256"] != item["binding_sha256"]
                    or job["hard_deadline_clock_epoch"] != item["original_deadline_clock_epoch"]
                    or job["hard_deadline_ns"] > item["original_deadline_ns"]
                    or terminal["ending"] != {"kind": "not_started"} or terminal["job"] != job
                    or terminal["worker_owner"] != owner or terminal["frame_sequence"] != item["frame_sequence"]
                    or terminal["capture_sequence"] != item["capture_sequence"]):
                raise ValueError("NotStarted original Job/owner/terminal/frame/capture differs")
            original = source.original_retirement_evidence(identity)
            body = original["observation"]
            declaration = next((entry for entry in publication["retirements"] if entry["retirement_sha256"] == identity), None)
            if (declaration is None or declaration["readout_record_sha256"] != []
                    or original["publication_file_sha256"] != value["r_publication_file_sha256"]
                    or body["native_admission"] != root or body["native_admission_sha256"] != value["native_admission_sha256"]
                    or body["retirement"] != terminal or body["retirement_sha256"] != identity
                    or body["readout_record_sha256"] != []):
                raise ValueError("NotStarted original publication terminal body/readout coverage differs")
        counts[item["kind"]] += 1
    return counts


class RoventRUnsubmittedObservation:
    """Detached data observation; matching metadata issues no original capability."""
    def __init__(self, *args, **kwargs):
        raise TypeError("use validate_r_unsubmitted_observations")

    def _check(self):
        if (self._value["observations_sha256"] != self._sha256
                or digest(DOMAIN, {**self._value, "observations_sha256": ""}) != self._sha256
                or digest("rz-rovent-execution-canonical/1", self._lock["manifest"]) != self._value["input_lock_sha256"]
                or self._lock["canonical_sha256"] != self._value["input_lock_sha256"]):
            raise ValueError("R unsubmitted detached original observation changed")
        if self._source is not None:
            original = self._source.original_publication_metadata()
            if (original["publication_file"]["sha256"] != self._value["r_publication_file_sha256"]
                    or _items(self._value["observations"], self._value, self._source, original["publication"]) != self._counts):
                raise ValueError("R unsubmitted original source observation changed")

    def to_mapping(self):
        self._check()
        return copy.deepcopy(self._value)

    def observation(self):
        self._check()
        return {"schema": OBSERVATION_DOMAIN, "observations_sha256": self._sha256,
            "native_admission_sha256": self._value["native_admission_sha256"], "input_lock_sha256": self._value["input_lock_sha256"],
            "r_publication_file_sha256": self._value["r_publication_file_sha256"], "observation_count": sum(self._counts.values()),
            "unsubmitted_count": self._counts["unsubmitted"], "not_started_count": self._counts["not_started"],
            "nnuec_asset_bytes_scope": "unknown_without_publication" if self._source is None else "verified_historical_source_bytes",
            "input_lock_file_bytes_verified": False, "nnuec_parameter_identity_recomputed": False, "original_manager_custody_verified": False,
            "original_native_authority_reissued": False, "past_physical_completion_reissued": False,
            "actual_rules_replay_verified": False, "execution_authorized": False, "training_source_authorized": False,
            "training_target_authorized": False, "input_rows_issued": False, "per_readout_consumption": False,
            "native_node_usage": None, "whole_action_cost": None, "wdl_target": None, "utility_target": None}


def _validate_r_unsubmitted_observations(value, *, input_lock, source):
    """Validate original declarations and optional typed published terminal joins.

    ``input_lock`` is the original parsed snapshot supplied by a pinned caller;
    its canonical seal is checked here. This function performs no file I/O and
    does not claim to have independently checked that snapshot's raw file pin.
    """
    _encoded(value, MAX_BYTES, "R unsubmitted observations")
    fields(value, _TOP, "R unsubmitted observations")
    if (value["schema"] != DOMAIN or type(value["per_readout_consumption"]) is not bool
            or value["per_readout_consumption"] or any(value[key] is not None for key in _NO_TARGET)):
        raise ValueError("R unsubmitted explicit domain/no-usage/no-target scope required")
    for name in ("native_admission_sha256", "input_lock_sha256", "nnuec_weights_raw_sha256", "nnuec_parameter_sha256", "observations_sha256"):
        sha(value[name])
    if value["r_publication_file_sha256"] is not None:
        sha(value["r_publication_file_sha256"])
    _encoded(input_lock, MAX_LOCK_BYTES, "R unsubmitted original input lock")
    lock = _input_lock(input_lock, value["input_lock_sha256"])
    authority = _authority(value["authority"])
    nnuec = _admission(value["native_admission"], lock, value["native_admission_sha256"], authority, value["nnuec_weights_raw_sha256"])
    publication = _publication(source, value, nnuec, lock)
    counts = _items(value["observations"], value, source, publication)
    expected = digest(DOMAIN, {**value, "observations_sha256": ""})
    if expected != value["observations_sha256"]:
        raise ValueError("R unsubmitted observations digest differs")
    result = object.__new__(RoventRUnsubmittedObservation)
    result._value, result._lock, result._source = copy.deepcopy(value), copy.deepcopy(lock), source
    result._sha256, result._counts = expected, counts
    return result


def validate_r_unsubmitted_observations(value, *, input_lock, source=None):
    """Strict mapping boundary, with no file or original capability issuance."""
    try:
        return _validate_r_unsubmitted_observations(value, input_lock=input_lock, source=source)
    except (KeyError, TypeError, OverflowError, RecursionError) as error:
        raise ValueError("R unsubmitted malformed original typed metadata") from error


def parse_r_unsubmitted_observations(input_json, *, input_lock, source=None):
    if type(input_json) is not str:
        raise ValueError("R unsubmitted original UTF8 JSON text required")
    try:
        raw = input_json.encode("utf-8")
    except UnicodeError as error:
        raise ValueError("R unsubmitted original UTF8 JSON text required") from error
    if not 0 < len(raw) <= MAX_BYTES:
        raise ValueError("R unsubmitted original JSON byte cap exceeded")
    return validate_r_unsubmitted_observations(_decode_json(raw), input_lock=input_lock, source=source)
