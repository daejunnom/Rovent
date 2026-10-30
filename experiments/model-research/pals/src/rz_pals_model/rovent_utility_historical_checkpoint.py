"""Original utility checkpoint evidence; never a recovered execution lease.

Numerical tensor bytes are not decoded here. The immutable producer projection
is independently joined to original file pins, ordered update/consume evidence
and a separately loaded scratch state. Historical paths are never PIDs, process
handles or permission to reclaim an old run. P/C/V codecs are not reinterpreted.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from pathlib import PurePosixPath, PureWindowsPath

from .generation_data import digest
from .nnuec_reference import _training
from .rovent_training_bootstrap import (PinnedReadBudget, _strict_json, _absolute,
    _pin as _absolute_pin, _open_directory, _open_file, _identity as _file_identity)
from .rovent_utility_training import COHORT_PREPARATION_DOMAIN, validate_training_preparation
from .rovent_utility_training_authority import (COST_FIELDS, MAX_COMPLETION_RECEIPT_BYTES,
    validate_utility_checkpoint_cost, _cost, _fields, _sha, _uint, _identifier, _cost_evidence)
from .rovent_utility_training_window import validate_utility_sampler, next_utility_sample, validate_utility_window_v2, utility_window_v2_digest


TENSOR_PROJECTION_DOMAIN = "rz-rovent-utility-tensor-projection/1"
STATE_DOMAIN = "rz-rovent-utility-derived-training-state/1"
MAX_PROJECTION_BYTES = 16 * 1024**2
MAX_TENSOR_BYTES = 2 * 1024**3
MAX_JOURNAL_BYTES = 2 * 1024**3
MAX_UPDATES = 131072
PUBLICATION_DOMAIN = "rz-rovent-utility-checkpoint-publication/1"
RETIREMENT_DOMAIN = "rz-rovent-utility-checkpoint-retirement/1"
INITIAL_DOMAIN = "rz-rovent-utility-initial-ledger-projection/1"
JOURNAL_DOMAIN = "rz-rovent-utility-ledger-event/1"
CLOSURE_DOMAIN = "rz-rovent-utility-original-training-closure/1"
TERMINAL_DOMAIN = "rz-rovent-utility-parent-terminal-receipt/1"
PUBLICATION_FIELDS = ("schema", "execution_id", "window_id", "run_id", "preparation_sha256", "dataset_sha256",
    "window_spec_sha256", "trainer_source_sha256", "trainer_python_executable_sha256", "trainer_dependency_set_sha256",
    "tensor", "retirement", "tensor_projection", "initial_ledger_state", "original_preparation", "original_window",
    "original_registration", "original_rpc_journal", "completion_receipts", "ancestor_completion_receipts",
    "parent_publication", "cost", "closure", "historical_evidence_only", "execution_authorized")
STATE_FIELDS = ("model_state_sha256", "optimizer_state_sha256", "scheduler_state_sha256",
                "rng_state_sha256", "sampler_state_sha256")
TENSOR_PROJECTION_FIELDS = ("schema", "payload_state_sha256", "preparation", "metadata", "sampler", "cuda_devices",
    "parameter_names", "completed_updates", "dispatched_updates", "ancestor_updates", "pending_microbatches", "cost",
    "ancestor_cost", "completion_receipts", "ancestor_completion_sha256", "parent_checkpoint_sha256",
    "source_cost_observation", "derived_training_state", *STATE_FIELDS)
METADATA_FIELDS = ("schema", "model_schema", "model_semantics", "base_schema", "base_semantics", "base_weights_sha256",
    "frozen_base_mapping_sha256", "frozen_base_config", "hidden_width", "input_width", "initial_training")


def _codec(value):
    from .rovent_utility_training_family import LEGACY, require_family
    return require_family(LEGACY if value is None else value)


def _history(value):
    if value["historical_evidence_only"] is not True or value["execution_authorized"] is not False:
        raise ValueError("utility historical observations cannot issue execution permission")


def _text(value, maximum=1024):
    if (type(value) is not str or not value.strip() or len(value.encode("utf-8")) > maximum
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)):
        raise ValueError("bounded original utility historical text required")
    return value


def _relative_ref(value, maximum=2 * 1024**3):
    _cost_evidence(value)
    if PureWindowsPath(value["path"]).drive:
        raise ValueError("utility archive reference cannot use a drive prefix")
    _uint(value["bytes"], maximum, positive=True)
    return value


def _identity(value):
    _fields(value, ("boot_id", "pid", "start_ticks", "cgroup_path", "cgroup_device", "cgroup_inode"), "original utility child identity")
    _text(value["boot_id"], 128); _uint(value["pid"], 2**32 - 1, positive=True); _uint(value["start_ticks"], positive=True)
    if not isinstance(_absolute(value["cgroup_path"]), PurePosixPath):
        raise ValueError("original utility cgroup identity requires its Linux path spelling")
    _uint(value["cgroup_device"]); _uint(value["cgroup_inode"], positive=True)
    return value


def _process_receipt(value):
    optional = tuple(name for name in ("cleanup_elapsed_ns", "cleanup_timing_unavailable_reason") if type(value) is dict and name in value)
    _fields(value, ("supervisor_version", "pid", "elapsed_ns", "stop", "exit_code", "exit_signal", "group_cleanup",
        "stdout_bytes", "stderr_bytes", "observed_output_bytes", "descendant_cleanup_required", "child_limit_enforcement",
        "watched_artifact_bytes", "artifact_limit_enforcement", "errors", *optional), "original ProcessReceipt")
    _uint(value["supervisor_version"], 2**32 - 1); _uint(value["pid"], 2**32 - 1, positive=True)
    for name in ("elapsed_ns", "stdout_bytes", "stderr_bytes", "observed_output_bytes"):
        _uint(value[name])
    if value["stop"] not in ("exited", "wall_limit", "output_limit", "cancelled", "io_failure", "child_limit", "artifact_limit"):
        raise ValueError("original process stop kind required")
    if value["group_cleanup"] not in ("gone", "unverified") or type(value["descendant_cleanup_required"]) is not bool:
        raise ValueError("original process cleanup observation required")
    for name in ("exit_code", "exit_signal"):
        if value[name] is not None and (type(value[name]) is not int or not -(2**31) <= value[name] < 2**31):
            raise ValueError("original nullable process status requires an exact i32")
    if value["watched_artifact_bytes"] is not None:
        _uint(value["watched_artifact_bytes"])
    _text(value["child_limit_enforcement"], 1024)
    if value["artifact_limit_enforcement"] is not None:
        _text(value["artifact_limit_enforcement"], 1024)
    if "cleanup_elapsed_ns" in value and value["cleanup_elapsed_ns"] is not None:
        _uint(value["cleanup_elapsed_ns"])
    if "cleanup_timing_unavailable_reason" in value and value["cleanup_timing_unavailable_reason"] is not None:
        _text(value["cleanup_timing_unavailable_reason"], 1024)
    if type(value["errors"]) is not list or len(value["errors"]) > 1024:
        raise ValueError("bounded original process error roster required")
    for error in value["errors"]:
        _text(error, 1024)
    return value


def _closure(value, *, _family=None):
    family = _codec(_family)
    _fields(value, ("schema", "execution_id", "window_id", "run_id", "kind", "child_identity", "process_receipt",
        "terminal_receipt", "boundary_publication", "child_reap", "physical_completion", "historical_evidence_only", "execution_authorized"), "original utility closure")
    _history(value)
    if value["schema"] != family.domain(CLOSURE_DOMAIN):
        raise ValueError("separate original utility closure domain required")
    for name in ("execution_id", "window_id", "run_id"):
        _identifier(value[name])
    _identity(value["child_identity"])
    if value["child_reap"] not in ("owned_child_reaped", "unresolved") or value["physical_completion"] not in ("confirmed", "unknown"):
        raise ValueError("original utility reap/physical observation required")
    if value["kind"] == "optimizer_boundary":
        if (value["process_receipt"] is not None or value["terminal_receipt"] is not None or value["boundary_publication"] is not None
                or value["child_reap"] != "unresolved" or value["physical_completion"] != "unknown"):
            raise ValueError("a live optimizer boundary cannot claim process closure")
    elif value["kind"] == "process_terminal":
        _process_receipt(value["process_receipt"]); _relative_ref(value["terminal_receipt"], MAX_PROJECTION_BYTES)
        _relative_ref(value["boundary_publication"], MAX_PROJECTION_BYTES)
        if value["process_receipt"]["pid"] != value["child_identity"]["pid"]:
            raise ValueError("original process receipt changed child identity")
    else:
        raise ValueError("registered utility boundary or terminal closure required")
    return value


def _parse_utility_checkpoint_publication(raw, *, publication_sha256, publication_bytes, _family=None):
    family = _codec(_family)
    value = _raw_json(raw, expected_sha256=publication_sha256, expected_bytes=publication_bytes, maximum=MAX_PROJECTION_BYTES)
    _fields(value, PUBLICATION_FIELDS, "utility checkpoint publication")
    _history(value)
    if value["schema"] != family.domain(PUBLICATION_DOMAIN):
        raise ValueError("original utility publication required; P/C/V conversion forbidden")
    for name in ("execution_id", "window_id", "run_id"):
        _identifier(value[name])
    for name in ("preparation_sha256", "dataset_sha256", "window_spec_sha256", "trainer_source_sha256",
                 "trainer_python_executable_sha256", "trainer_dependency_set_sha256"):
        _sha(value[name])
    for name in ("tensor", "retirement", "tensor_projection", "initial_ledger_state", "original_preparation",
                 "original_window", "original_registration", "original_rpc_journal"):
        _relative_ref(value[name], MAX_TENSOR_BYTES if name == "tensor" else MAX_JOURNAL_BYTES if name == "original_rpc_journal" else MAX_PROJECTION_BYTES)
    all_receipts = []
    for name in ("completion_receipts", "ancestor_completion_receipts"):
        refs = value[name]
        if type(refs) is not list or len(refs) > MAX_UPDATES:
            raise ValueError("bounded original utility completion-file chain required")
        for reference in refs:
            _relative_ref(reference, MAX_COMPLETION_RECEIPT_BYTES)
            all_receipts.append(reference["sha256"])
    if len(all_receipts) > MAX_UPDATES or len(set(all_receipts)) != len(all_receipts):
        raise ValueError("original utility full receipt chains cannot repeat updates")
    if value["parent_publication"] is not None:
        _relative_ref(value["parent_publication"], MAX_PROJECTION_BYTES)
    _relative_ref(value["cost"], MAX_PROJECTION_BYTES)
    _relative_ref(value["closure"], MAX_PROJECTION_BYTES)
    return copy.deepcopy(value)


def parse_utility_checkpoint_publication(raw, *, publication_sha256, publication_bytes):
    return _parse_utility_checkpoint_publication(raw, publication_sha256=publication_sha256,
        publication_bytes=publication_bytes)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _equal(left, right):
    return _json(left) == _json(right)


def _raw_json(raw, *, expected_sha256, expected_bytes, maximum):
    _sha(expected_sha256); _uint(expected_bytes, maximum, positive=True)
    if type(raw) is not bytes or len(raw) != expected_bytes or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("original utility historical raw byte pin differs")
    return _strict_json(raw, maximum=maximum)


def _plain_state_sha256(value):
    """Exact typed state framing for the validated ASCII sampler, no Torch."""
    result = hashlib.sha256()
    def visit(item, depth=0):
        if depth > 16:
            raise ValueError("bounded utility sampler nesting required")
        if type(item) is dict:
            result.update(b"dict")
            for key in sorted(item):
                visit(key, depth + 1); visit(item[key], depth + 1)
        elif type(item) is list:
            result.update(("list" + str(len(item))).encode("ascii"))
            for child in item:
                visit(child, depth + 1)
        elif type(item) is str and item.isascii():
            result.update(json.dumps(["str", item], separators=(",", ":")).encode("ascii"))
        elif type(item) is int:
            _uint(item)
            result.update(json.dumps(["int", item], separators=(",", ":")).encode("ascii"))
        else:
            raise ValueError("only original plain ASCII utility sampler state is supported")
        result.update(b";")
    visit(value)
    return result.hexdigest()


def _metadata(value, preparation):
    _fields(value, METADATA_FIELDS, "utility original tensor metadata")
    registration, membership = preparation["registration"], preparation["parameter_spec"]
    base, utility = registration["frozen_base"], registration["utility"]
    expected = {"schema": "rz-nnuec-utility-torch-metadata/1", "model_schema": utility["schema"],
        "model_semantics": utility["semantics"], "base_schema": base["schema"], "base_semantics": base["semantics"],
        "base_weights_sha256": base["weights_sha256"], "frozen_base_mapping_sha256": _sha(value["frozen_base_mapping_sha256"]),
        "frozen_base_config": base["config"], "hidden_width": membership["hidden_width"], "input_width": membership["input_width"],
        "initial_training": dict(_training(value["initial_training"]))}
    if not _equal(value, expected):
        raise ValueError("utility original metadata differs from its numerical source/preparation")
    return value


def _tensor_projection(value, *, _family=None):
    family = _codec(_family)
    _fields(value, TENSOR_PROJECTION_FIELDS, "original utility tensor projection")
    if value["schema"] != family.domain(TENSOR_PROJECTION_DOMAIN):
        raise ValueError("actual utility producer projection required; no historical wrapper conversion")
    _sha(value["payload_state_sha256"])
    parts = family.parts(value["preparation"])
    preparation = parts.preparation
    if not family.intervention and preparation["schema"] != COHORT_PREPARATION_DOMAIN:
        raise ValueError("actual historical utility source requires its separate cohort preparation/2")
    metadata = _metadata(value["metadata"], preparation)
    for name in STATE_FIELDS:
        _sha(value[name])
    for name in ("completed_updates", "dispatched_updates", "ancestor_updates"):
        _uint(value[name], MAX_UPDATES)
    completed, ancestor = value["completed_updates"], value["ancestor_updates"]
    total = _uint(completed + ancestor, MAX_UPDATES)
    if completed > value["dispatched_updates"] or type(value["pending_microbatches"]) is not int or value["pending_microbatches"] != 0:
        raise ValueError("utility projection must retain an actual completed optimizer boundary")
    _cost(value["cost"]); _cost(value["ancestor_cost"])
    for name in COST_FIELDS:
        _uint(value["cost"][name] + value["ancestor_cost"][name])
    receipts = value["completion_receipts"]
    if type(receipts) is not list or len(receipts) != completed:
        raise ValueError("each original utility update requires one ACK pin")
    seen = set()
    for pin in receipts:
        _fields(pin, ("sha256", "bytes"), "utility original tensor ACK pin")
        _sha(pin["sha256"]); _uint(pin["bytes"], MAX_COMPLETION_RECEIPT_BYTES, positive=True)
        if pin["sha256"] in seen:
            raise ValueError("duplicate original utility update ACK")
        seen.add(pin["sha256"])
    ancestor_shas = value["ancestor_completion_sha256"]
    if type(ancestor_shas) is not list or len(ancestor_shas) != ancestor:
        raise ValueError("complete original utility ancestor update chain required")
    for identity in ancestor_shas:
        _sha(identity)
        if identity in seen:
            raise ValueError("utility current/ancestor update chain contains duplicates")
        seen.add(identity)
    if value["parent_checkpoint_sha256"] is not None:
        _sha(value["parent_checkpoint_sha256"])
    if value["source_cost_observation"] is not None:
        validate_utility_checkpoint_cost(value["source_cost_observation"])
    devices = value["cuda_devices"]
    if (type(devices) is not list or len(devices) > 1
            or any(type(device) is not int or not 0 <= device < 64 for device in devices)):
        raise ValueError("utility projection requires its one original device roster")
    membership = preparation["parameter_spec"]
    expected_names = [membership["wire_to_parameter"][name] for name in preparation["recipe"]["active_parameter_names"]]
    if type(value["parameter_names"]) is not list or value["parameter_names"] != expected_names:
        raise ValueError("utility projection changed its four private optimizer members/order")
    sampler = value["sampler"]
    if type(sampler) is not dict:
        raise ValueError("original utility sampler required")
    validate_utility_sampler(sampler, context_ids=sampler.get("context_ids"), pool_sha256=sampler.get("pool_sha256"),
        run_id=sampler.get("run_id"), seed=sampler.get("seed"))
    if _plain_state_sha256(sampler) != value["sampler_state_sha256"]:
        raise ValueError("utility original sampler bytes differ from its typed state seal")
    state_domain = family.domain(STATE_DOMAIN)
    derived = {"schema": state_domain, "initial_source_declaration": metadata["initial_training"],
        "current_tensor_kind": "learned" if total else "initial_source_only", "completed_updates": completed,
        "ancestor_updates": ancestor, "total_updates": total,
        "completion_chain_sha256": digest(state_domain, [*ancestor_shas, *(pin["sha256"] for pin in receipts)]),
        "recipe_sha256": parts.recipe_sha256, "registration_sha256": parts.source_registration_sha256,
        "frozen_base_weights_sha256": metadata["base_weights_sha256"],
        "frozen_base_mapping_sha256": metadata["frozen_base_mapping_sha256"], "live_training_authority_reissued": False}
    if not _equal(value["derived_training_state"], derived):
        raise ValueError("utility original derived state changed counts, frozen source or full ACK chain")
    return copy.deepcopy(value)


def parse_utility_tensor_projection(raw, *, expected_sha256, expected_bytes):
    """Decode the actual producer bytes, without opening or restoring a tensor."""
    return _tensor_projection(_raw_json(raw, expected_sha256=expected_sha256,
        expected_bytes=expected_bytes, maximum=MAX_PROJECTION_BYTES))


def project_utility_tensor_scratch(projection, *, observed_state):
    """Join a separately measured scratch; this returns historical data only."""
    expected = _tensor_projection(projection)
    observed = _tensor_projection(observed_state)
    if not _equal(expected, observed):
        raise ValueError("actual utility scratch tensor/state/journal projection differs from the original producer")
    return {"schema": "rz-rovent-utility-historical-scratch-observation/1", "status": "historical_integrity_only",
        "payload_state_sha256": expected["payload_state_sha256"],
        **{name: expected[name] for name in STATE_FIELDS}, "completed_updates": expected["completed_updates"],
        "ancestor_updates": expected["ancestor_updates"], "snapshot_cost": copy.deepcopy(expected["cost"]),
        "past_native_authority_reissued": False, "past_physical_completion_reissued": False,
        "old_run_reclamation_issued": False, "execution_authorized": False}


def _states(value):
    _fields(value, STATE_FIELDS, "five original utility states")
    for identity in value.values():
        _sha(identity)
    return value


def _initial(value, *, _family=None):
    family = _codec(_family)
    _fields(value, ("schema", "state_admission_scope", "states", "sampler", "completed_updates", "dispatched_updates",
        "ancestor_updates", "cost", "ancestor_cost", "completion_receipts", "ancestor_completion_sha256",
        "initial_source_declaration", "frozen_base_weights_sha256", "frozen_base_mapping_sha256", "source_checkpoint",
        "source_payload_state_sha256", "source_snapshot_cost_observation", "source_snapshot_cost", "source_snapshot_ancestor_cost",
        "checkpoint_cost", "historical_evidence_only", "execution_authorized"), "utility initial ledger projection")
    _history(value)
    if value["schema"] != family.domain(INITIAL_DOMAIN) or value["state_admission_scope"] not in (
            "actual_original_child_initial_observation_not_independent_torch_recomputation",
            "original_held_checkpoint_and_full_completion_chain",
            "original_checkpoint_model_and_chain_with_actual_child_reset_state_observation"):
        raise ValueError("original utility initial admission scope required")
    _states(value["states"]); _cost(value["cost"]); _cost(value["ancestor_cost"])
    _training(value["initial_source_declaration"])
    _sha(value["frozen_base_weights_sha256"]); _sha(value["frozen_base_mapping_sha256"])
    for name in ("completed_updates", "dispatched_updates", "ancestor_updates"):
        _uint(value[name], MAX_UPDATES)
    if value["completed_updates"] > value["dispatched_updates"] or value["completed_updates"] + value["ancestor_updates"] > MAX_UPDATES:
        raise ValueError("bounded original utility initial counts required")
    receipts, ancestors = value["completion_receipts"], value["ancestor_completion_sha256"]
    if type(receipts) is not list or len(receipts) != value["completed_updates"] or type(ancestors) is not list or len(ancestors) != value["ancestor_updates"]:
        raise ValueError("original initial utility completion chain differs from counts")
    identities = []
    for pin in receipts:
        _absolute_pin(pin, MAX_COMPLETION_RECEIPT_BYTES); identities.append(pin["sha256"])
    identities.extend(_sha(identity) for identity in ancestors)
    if len(set(identities)) != len(identities):
        raise ValueError("original initial utility completion chain repeats an update")
    source_fields = ("source_checkpoint", "source_payload_state_sha256", "source_snapshot_cost_observation",
                     "source_snapshot_cost", "source_snapshot_ancestor_cost", "checkpoint_cost")
    if value["source_checkpoint"] is None:
        if (any(value[name] is not None for name in source_fields) or receipts or ancestors
                or value["completed_updates"] or value["dispatched_updates"] or value["ancestor_updates"]
                or value["state_admission_scope"] != "actual_original_child_initial_observation_not_independent_torch_recomputation"):
            raise ValueError("fresh original observation cannot create prior utility completion/source evidence")
        if any(value["cost"].values()) or any(value["ancestor_cost"].values()):
            raise ValueError("fresh utility initial observation cannot create recorded prior expense")
    else:
        _absolute_pin(value["source_checkpoint"], MAX_TENSOR_BYTES); _sha(value["source_payload_state_sha256"])
        _cost(value["source_snapshot_cost"]); _cost(value["source_snapshot_ancestor_cost"])
        validate_utility_checkpoint_cost(value["checkpoint_cost"])
        if value["source_snapshot_cost_observation"] is not None:
            validate_utility_checkpoint_cost(value["source_snapshot_cost_observation"])
        if value["state_admission_scope"] not in ("original_held_checkpoint_and_full_completion_chain",
                "original_checkpoint_model_and_chain_with_actual_child_reset_state_observation"):
            raise ValueError("a restored original utility state requires its full checkpoint admission")
        if (value["state_admission_scope"] == "original_checkpoint_model_and_chain_with_actual_child_reset_state_observation"
                and (value["completed_updates"] or value["dispatched_updates"] or receipts or any(value["cost"].values()))):
            raise ValueError("original fork must retain ancestry and reset current counts, receipts and expense")
    return value


def _checkpoint_request(value, *, _family=None):
    family = _codec(_family)
    _fields(value, ("schema", "checkpoint", "tensor_projection", "completed_updates", "ancestor_updates", "snapshot_cost",
        "post_snapshot_cost", "pending_microbatches", "derived_training_state_sha256", *STATE_FIELDS), "original utility checkpoint request")
    if value["schema"] != family.domain("rz-rovent-utility-trainer-checkpoint/1") or type(value["pending_microbatches"]) is not int or value["pending_microbatches"] != 0:
        raise ValueError("original utility no-gradient checkpoint request required")
    _absolute_pin(value["checkpoint"], MAX_TENSOR_BYTES); _absolute_pin(value["tensor_projection"], MAX_PROJECTION_BYTES)
    _uint(value["completed_updates"], MAX_UPDATES); _uint(value["ancestor_updates"], MAX_UPDATES)
    for name in (*STATE_FIELDS, "derived_training_state_sha256"):
        _sha(value[name])
    _cost(value["snapshot_cost"]); _cost(value["post_snapshot_cost"])
    for name in COST_FIELDS:
        if value["post_snapshot_cost"][name] < value["snapshot_cost"][name]:
            raise ValueError("original checkpoint post-snapshot expense decreased")
    if value["post_snapshot_cost"]["output_bytes"] - value["snapshot_cost"]["output_bytes"] != value["checkpoint"]["bytes"] + value["tensor_projection"]["bytes"]:
        raise ValueError("original checkpoint must charge its two published files once")
    return value


def _parse_utility_checkpoint_retirement(raw, *, expected_sha256, expected_bytes, _family=None):
    family = _codec(_family)
    value = _raw_json(raw, expected_sha256=expected_sha256, expected_bytes=expected_bytes, maximum=MAX_PROJECTION_BYTES)
    _fields(value, ("schema", "checkpoint", "original_request", "parent_projection", "tensor_projection_sha256",
        "historical_evidence_only", "execution_authorized"), "original utility checkpoint retirement")
    _history(value)
    if value["schema"] != family.domain(RETIREMENT_DOMAIN):
        raise ValueError("separate original utility retirement domain required")
    _absolute_pin(value["checkpoint"], MAX_TENSOR_BYTES); _sha(value["tensor_projection_sha256"])
    request = _checkpoint_request(value["original_request"], _family=family)
    parent = value["parent_projection"]
    _fields(parent, ("schema", "checkpoint", "tensor_projection", "states", "completed_updates", "ancestor_updates",
        "snapshot_cost", "post_snapshot_cost", "unupdated_context_sha256", "pending_receipt",
        "historical_evidence_only", "execution_authorized"), "original utility parent checkpoint projection")
    _history(parent); _states(parent["states"])
    if (parent["schema"] != family.domain("rz-rovent-utility-parent-checkpoint-projection/1") or parent["pending_receipt"] is not False
            or not _equal(parent["checkpoint"], request["checkpoint"]) or not _equal(value["checkpoint"], request["checkpoint"])
            or not _equal(parent["tensor_projection"], request["tensor_projection"])
            or value["tensor_projection_sha256"] != request["tensor_projection"]["sha256"]):
        raise ValueError("utility retirement changed original held file/projection boundaries")
    for name in ("completed_updates", "ancestor_updates", "snapshot_cost", "post_snapshot_cost"):
        if not _equal(parent[name], request[name]):
            raise ValueError("utility parent checkpoint projection differs from its original request")
    if any(parent["states"][name] != request[name] for name in STATE_FIELDS):
        raise ValueError("utility original parent checkpoint states differ")
    if type(parent["unupdated_context_sha256"]) is not list or len(parent["unupdated_context_sha256"]) > 8:
        raise ValueError("utility checkpoint cannot exceed one unupdated accumulation")
    for identity in parent["unupdated_context_sha256"]:
        _sha(identity)
    return copy.deepcopy(value)


def parse_utility_checkpoint_retirement(raw, *, expected_sha256, expected_bytes):
    return _parse_utility_checkpoint_retirement(raw, expected_sha256=expected_sha256, expected_bytes=expected_bytes)


def _monotone(before, after):
    _cost(before); _cost(after)
    if any(after[name] < before[name] for name in COST_FIELDS):
        raise ValueError("original utility recorded cost decreased")


def _sum_cost(*values):
    return {name: _uint(sum(_cost(value)[name] for value in values)) for name in COST_FIELDS}


def _pin_identity(value):
    return {name: value[name] for name in ("sha256", "bytes")}


class _Archive:
    """One same-root, nofollow, shared-budget archive read transaction."""
    def __init__(self, root, reads):
        if type(reads) is not PinnedReadBudget:
            raise ValueError("original shared finite read account required")
        self.root = _absolute(root)
        if not isinstance(self.root, PurePosixPath):
            raise ValueError("historical utility archive requires a Linux absolute root")
        self.reads, self.paths, self.completed, self.active = reads, {}, {}, set()
        self.fd = _open_directory(self.root)
        info = os.fstat(self.fd)
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            os.close(self.fd)
            raise ValueError("original utility archive root must be private and owned")
        self.identity = _file_identity(info)

    def close(self):
        if self.fd is not None:
            os.close(self.fd); self.fd = None

    def _root(self):
        if self.fd is None:
            raise ValueError("closed original utility archive")
        self.reads._check()
        fd = _open_directory(self.root)
        try:
            if _file_identity(os.fstat(fd)) != self.identity or _file_identity(os.fstat(self.fd)) != self.identity:
                raise ValueError("original utility archive root identity changed")
        finally:
            os.close(fd)

    def absolute(self, reference, maximum=MAX_PROJECTION_BYTES):
        self._root(); _relative_ref(reference, maximum)
        spelling = reference["path"]
        prior = self.paths.setdefault(spelling, _pin_identity(reference))
        if not _equal(prior, _pin_identity(reference)) or len(self.paths) > 1048576:
            raise ValueError("utility archive path pin conflicts or exceeds its finite inventory")
        return {"path": str(self.root / spelling), **_pin_identity(reference)}

    def raw(self, reference, maximum=MAX_PROJECTION_BYTES):
        pin = self.absolute(reference, maximum)
        raw, _readback = self.reads.read(pin, maximum=maximum)
        self._root()
        return raw

    def json(self, reference, maximum=MAX_PROJECTION_BYTES):
        return _strict_json(self.raw(reference, maximum), maximum=maximum)

    def verify(self, reference, maximum):
        self.reads.verify(self.absolute(reference, maximum), maximum=maximum)
        self._root()

    def journal(self, reference):
        """Replay bounded JSONL without retaining a multi-GiB journal in RAM."""
        pin = self.absolute(reference, MAX_JOURNAL_BYTES)
        lease = self.reads.reserve_external(pin["bytes"] + 1)
        fd, total, pending, hasher = None, 0, b"", hashlib.sha256()
        try:
            fd = _open_file(pin["path"])
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != pin["bytes"]:
                raise ValueError("original utility journal regular file/size differs")
            while total < pin["bytes"]:
                self.reads._check()
                block = os.read(fd, min(65536, pin["bytes"] - total))
                self.reads.charge_external(lease, len(block))
                if not block:
                    raise ValueError("original utility journal ended before its pinned boundary")
                total += len(block); hasher.update(block); pending += block
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if not line or len(line) + 1 > 256 * 1024:
                        raise ValueError("bounded original utility JSONL event required")
                    yield _strict_json(line, maximum=256 * 1024)
                if len(pending) > 256 * 1024:
                    raise ValueError("original utility journal event exceeds its finite boundary")
            extra = os.read(fd, 1); self.reads.charge_external(lease, len(extra))
            current_fd = _open_file(pin["path"])
            try:
                current = os.fstat(current_fd)
            finally:
                os.close(current_fd)
            if (pending or extra or total != pin["bytes"] or hasher.hexdigest() != pin["sha256"]
                    or _file_identity(before) != _file_identity(os.fstat(fd))
                    or _file_identity(before) != _file_identity(current)):
                raise ValueError("original utility journal bytes/EOF/identity changed")
            self._root(); self.reads.finish_external(lease, expected_bytes=pin["bytes"])
        except BaseException:
            self.reads.fail_external(lease)
            raise
        finally:
            if fd is not None:
                os.close(fd)


def _dataset_source(dataset, registration, *, _family=None):
    # Importing this module is still stdlib-only. A raw cohort file observation
    # cannot substitute this separately validated historical target dataset.
    family = _codec(_family)
    observation = family.require_dataset(dataset)
    for name, source in (("publication_pin", "publication"), ("durable_ack_pin", "durable_ack")):
        expected = _pin_identity(registration["current_dataset"][source])
        if not _equal(observation[name], expected):
            raise ValueError("historical utility dataset changed its original raw publication/ACK pins")
    return dataset


class _Replay:
    """Metadata replay only; no ACK, optimizer, native or physical capability."""
    def __init__(self, archive, publication, preparation, window, initial, dataset, *, _family=None):
        self.family = _codec(_family)
        self.parts = self.family.parts(preparation)
        self.archive, self.publication, self.preparation, self.window, self.initial, self.dataset = (
            archive, publication, preparation, window, initial, dataset)
        self.sampler = copy.deepcopy(initial["sampler"])
        validate_utility_sampler(self.sampler, context_ids=dataset.context_ids, pool_sha256=dataset.pool_sha256,
            run_id=window["run_id"], seed=window["training_seed"])
        if _plain_state_sha256(self.sampler) != initial["states"]["sampler_state_sha256"]:
            raise ValueError("original initial utility sampler differs from its five-state pin")
        self.states = copy.deepcopy(initial["states"])
        self.completed, self.dispatched = initial["completed_updates"], initial["dispatched_updates"]
        self.receipts = [_pin_identity(pin) for pin in initial["completion_receipts"]]
        self.ancestors = initial["ancestor_completion_sha256"]
        self.cost, self.highwater = copy.deepcopy(initial["cost"]), copy.deepcopy(initial["cost"])
        self.pending, self.consumed, self.groups, self.masked, self.begun, self.sequence = [], 0, 0, 0, False, 0

    def derived(self):
        total = _uint(self.completed + self.initial["ancestor_updates"], MAX_UPDATES)
        domain = self.family.domain(STATE_DOMAIN)
        return {"schema": domain, "initial_source_declaration": self.initial["initial_source_declaration"],
            "current_tensor_kind": "learned" if total else "initial_source_only", "completed_updates": self.completed,
            "ancestor_updates": self.initial["ancestor_updates"], "total_updates": total,
            "completion_chain_sha256": digest(domain, [*self.ancestors, *(pin["sha256"] for pin in self.receipts)]),
            "recipe_sha256": self.parts.recipe_sha256, "registration_sha256": self.parts.source_registration_sha256,
            "frozen_base_weights_sha256": self.initial["frozen_base_weights_sha256"],
            "frozen_base_mapping_sha256": self.initial["frozen_base_mapping_sha256"], "live_training_authority_reissued": False}

    def counts(self, value):
        if (type(value["completed_updates"]) is not int or value["completed_updates"] != self.completed
                or type(value["ancestor_updates"]) is not int or value["ancestor_updates"] != self.initial["ancestor_updates"]):
            raise ValueError("original utility current/ancestor completion count differs")

    def observed_cost(self, value):
        _monotone(self.highwater, value)
        if (value["samples"] != self.initial["cost"]["samples"] + self.consumed
                or value["masked_groups"] != self.initial["cost"]["masked_groups"] + self.masked
                or value["output_bytes"] - self.initial["cost"]["output_bytes"] > self.window["max_output_bytes"]):
            raise ValueError("original utility samples/masked/output cost differs from full consumption replay")

    def begin(self, value):
        _fields(value, ("schema", "preparation_sha256", "window_spec_sha256", "dataset_sha256",
            "initial_source_declaration", "frozen_base_mapping_sha256", "frozen_base_weights_sha256", "completed_updates",
            "ancestor_updates", "derived_training_state_sha256", *STATE_FIELDS), "original utility begin")
        self.counts(value)
        expected = {"schema": self.family.domain("rz-rovent-utility-trainer-begin/1"), "preparation_sha256": self.window["preparation_sha256"],
            "window_spec_sha256": self.family.window_digest(self.window), "dataset_sha256": self.window["dataset_sha256"],
            "initial_source_declaration": self.initial["initial_source_declaration"],
            "frozen_base_mapping_sha256": self.initial["frozen_base_mapping_sha256"],
            "frozen_base_weights_sha256": self.initial["frozen_base_weights_sha256"], "completed_updates": self.completed,
            "ancestor_updates": self.initial["ancestor_updates"], "derived_training_state_sha256": digest(self.family.domain(STATE_DOMAIN), self.derived()),
            **self.states}
        if self.begun or not _equal(value, expected):
            raise ValueError("one original utility begin with exact frozen metadata/5state required")
        self.begun = True

    def consume(self, value):
        _fields(value, ("schema", "source_projection"), "original utility context consumption")
        if (not self.begun or value["schema"] != self.family.domain("rz-rovent-utility-trainer-consume/1") or len(self.pending) >= 8
                or self.consumed >= self.window["max_contexts"] or self.groups >= self.window["max_groups"]
                or self.completed - self.initial["completed_updates"] >= self.window["max_optimizer_updates"]):
            raise ValueError("original utility full consume journal exceeds its registered credits")
        index, sampler = next_utility_sample(self.sampler, context_ids=self.dataset.context_ids,
            pool_sha256=self.dataset.pool_sha256, run_id=self.window["run_id"], seed=self.window["training_seed"])
        context = self.dataset.historical_context(self.dataset.context_ids[index])
        _fields(context, ("context_sha256", "source_projection", "pair_sha256", "unique_known_pairs"), "original dataset context")
        if context["context_sha256"] != self.dataset.context_ids[index] or not _equal(context["source_projection"], value["source_projection"]):
            raise ValueError("original utility consumed context differs from exact sampler/source projection")
        pairs = context["pair_sha256"]
        if (type(pairs) is not list or len(pairs) > 64 or len(set(pairs)) != len(pairs)
                or type(context["unique_known_pairs"]) is not int or context["unique_known_pairs"] != len(pairs)):
            raise ValueError("original utility known comparison IDs/unique count differs")
        for identity in pairs:
            _sha(identity)
        self.sampler = sampler; self.consumed += 1; self.pending.append(copy.deepcopy(context))
        if len(self.pending) == 8:
            self.groups += 1
            if not any(context["unique_known_pairs"] for context in self.pending):
                self.masked += 1; self.pending.clear()

    def update(self, value, original_pin, ack):
        _fields(value, ("schema", "event", "cost", "pending_microbatches", "frozen_base_weights_sha256",
            "frozen_base_mapping_sha256", "prior_completion_chain_sha256", *STATE_FIELDS), "original utility update")
        if (not self.begun or len(self.pending) != 8 or value["schema"] != self.family.domain("rz-rovent-utility-trainer-update/1")
                or type(value["pending_microbatches"]) is not int or value["pending_microbatches"] != 0):
            raise ValueError("original update requires eight actual prior journal consumes and no pending gradient")
        for name in ("frozen_base_weights_sha256", "frozen_base_mapping_sha256"):
            if value[name] != self.initial[name]:
                raise ValueError("original utility update changed frozen source observation")
        if value["prior_completion_chain_sha256"] != self.derived()["completion_chain_sha256"]:
            raise ValueError("original utility update omitted/repeated a prior completion")
        denominator = sum(context["unique_known_pairs"] for context in self.pending)
        event = {"schema": self.family.domain("rz-rovent-utility-learner-update/1"), "completed_updates": self.completed + 1,
            "dispatched_updates": self.dispatched + 1, "unique_known_pair_denominator": denominator,
            "consumed_context_sha256": [context["context_sha256"] for context in self.pending], "groups": 1, "contexts": 8}
        if (not denominator or not _equal(value["event"], event)
                or self.completed + 1 - self.initial["completed_updates"] > self.window["max_optimizer_updates"]):
            raise ValueError("original utility update denominator/count/order differs")
        states = _states({name: value[name] for name in STATE_FIELDS})
        if (states["sampler_state_sha256"] != _plain_state_sha256(self.sampler)
                or any(states[name] == self.states[name] for name in STATE_FIELDS[:3])):
            raise ValueError("original successful utility update did not change active moments/model/scheduler")
        self.observed_cost(value["cost"])
        active = sum(context["unique_known_pairs"] > 0 for context in self.pending)
        if any(value["cost"][name] != self.cost[name] + active for name in ("forward_calls", "backward_calls")):
            raise ValueError("original utility forward/backward count differs from known context occurrences")
        expected = {"schema": self.family.domain("rz-rovent-utility-learner-completion/1"), "window_id": self.window["window_id"],
            "run_id": self.window["run_id"], "dataset_sha256": self.window["dataset_sha256"],
            "preparation_sha256": self.window["preparation_sha256"], "source_registration_sha256": self.parts.source_registration_sha256,
            "optimizer_owner": self.parts.optimizer_owner, "initial_state_admission_scope": self.initial["state_admission_scope"],
            "completed_updates": self.completed + 1, "dispatched_updates": self.dispatched + 1,
            "ancestor_updates": self.initial["ancestor_updates"], "window_completed_updates": self.completed + 1 - self.initial["completed_updates"],
            "window_dispatched_updates": self.dispatched + 1 - self.initial["dispatched_updates"],
            "consumed_contexts": self.pending, "original_request": value, "historical_evidence_only": True, "execution_authorized": False}
        _absolute_pin(original_pin, MAX_COMPLETION_RECEIPT_BYTES)
        if self.completed >= len(self.publication["completion_receipts"]):
            raise ValueError("original utility journal contains an unpublished completed update")
        reference = self.publication["completion_receipts"][self.completed]
        if not _equal(_pin_identity(reference), _pin_identity(original_pin)) or not _equal(ack, _pin_identity(reference)):
            raise ValueError("original utility journal ACK differs from its held full receipt file")
        body = self.archive.json(reference, MAX_COMPLETION_RECEIPT_BYTES)
        if not _equal(body, expected):
            raise ValueError("full original utility completion body differs from consumption/5state replay")
        self.completed += 1; self.dispatched += 1; self.receipts.append(_pin_identity(reference))
        self.states = copy.deepcopy(states); self.cost = copy.deepcopy(value["cost"])
        self.highwater = copy.deepcopy(self.cost); self.pending.clear()

    def checkpoint(self, request):
        _checkpoint_request(request, _family=self.family); self.counts(request)
        if (not self.begun or any(request[name] != self.states[name] for name in STATE_FIELDS[:3])
                or request["sampler_state_sha256"] != _plain_state_sha256(self.sampler)
                or request["derived_training_state_sha256"] != digest(self.family.domain(STATE_DOMAIN), self.derived())):
            raise ValueError("original utility checkpoint changed exact state, sampler or full completion chain")
        self.observed_cost(request["snapshot_cost"])
        _monotone(request["snapshot_cost"], request["post_snapshot_cost"])
        if request["post_snapshot_cost"]["output_bytes"] - self.initial["cost"]["output_bytes"] > self.window["max_output_bytes"]:
            raise ValueError("original utility checkpoint exceeds registered output credit")
        self.highwater = copy.deepcopy(request["post_snapshot_cost"])

    def replay(self):
        for event in self.archive.journal(self.publication["original_rpc_journal"]):
            _fields(event, ("schema", "sequence", "kind", "request", "completion_receipt", "ack"), "utility journal event")
            self.sequence += 1
            if (event["schema"] != self.family.domain(JOURNAL_DOMAIN) or type(event["sequence"]) is not int
                    or event["sequence"] != self.sequence or self.sequence > 2000000):
                raise ValueError("original utility journal sequence/domain differs")
            kind = event["kind"]
            if kind == "committed_update":
                self.update(event["request"], event["completion_receipt"], event["ack"])
            else:
                if event["completion_receipt"] is not None or not _equal(event["ack"], {}):
                    raise ValueError("non-update utility journal event cannot contain a completion receipt")
                if kind == "begin_window":
                    self.begin(event["request"])
                elif kind == "consume_pair_context":
                    self.consume(event["request"])
                elif kind == "checkpoint_published":
                    self.checkpoint(event["request"])
                else:
                    raise ValueError("explicit original utility journal operation required")
        if not self.begun:
            raise ValueError("original utility checkpoint journal omitted its begin event")


def _parent_join(initial, registration, parent):
    mode = registration["initialization_source"]["mode"]
    if mode == "initialize":
        if parent is not None or initial["source_checkpoint"] is not None:
            raise ValueError("fresh utility source cannot acquire an old parent checkpoint")
        if initial["sampler"]["epoch"] != 0 or initial["sampler"]["cursor"] != 0:
            raise ValueError("fresh utility sampler must begin at its original epoch/cursor")
        return
    if parent is None:
        raise ValueError("utility resume/fork omitted its complete original parent publication")
    checkpoint = registration["initialization_source"]["checkpoint"]
    source = parent["projection"]
    if (not _equal(_pin_identity(checkpoint["publication"]), _pin_identity(parent["publication_pin"]))
            or not _equal(_pin_identity(initial["source_checkpoint"]), _pin_identity(parent["publication"]["tensor"]))
            or initial["source_payload_state_sha256"] != source["payload_state_sha256"]
            or not _equal(initial["source_snapshot_cost"], source["cost"])
            or not _equal(initial["source_snapshot_ancestor_cost"], source["ancestor_cost"])
            or not _equal(initial["source_snapshot_cost_observation"], source["source_cost_observation"])
            or not _equal(initial["checkpoint_cost"], parent["cost"])
            or not _equal(initial["initial_source_declaration"], source["metadata"]["initial_training"])
            or initial["frozen_base_weights_sha256"] != source["metadata"]["base_weights_sha256"]
            or initial["frozen_base_mapping_sha256"] != source["metadata"]["frozen_base_mapping_sha256"]):
        raise ValueError("original utility parent tensor/cost/frozen metadata source changed")
    if mode == "resume":
        additional = parent["cost"]["post_snapshot_delta"]
        if parent["cost"]["closure_tail"]["status"] == "known":
            additional = _sum_cost(additional, parent["cost"]["closure_tail"]["delta"])
        if (initial["state_admission_scope"] != "original_held_checkpoint_and_full_completion_chain"
                or any(initial["states"][name] != source[name] for name in STATE_FIELDS)
                or not _equal(initial["sampler"], source["sampler"])
                or any(initial[name] != source[name] for name in ("completed_updates", "dispatched_updates", "ancestor_updates"))
                or not _equal(initial["cost"], _sum_cost(source["cost"], additional))
                or not _equal(initial["ancestor_cost"], source["ancestor_cost"])
                or not _equal([_pin_identity(pin) for pin in initial["completion_receipts"]], source["completion_receipts"])
                or initial["ancestor_completion_sha256"] != source["ancestor_completion_sha256"]):
            raise ValueError("utility resume changed original five states, sampler, counts or restored known expense")
    elif mode == "fork":
        ancestry = [*source["ancestor_completion_sha256"], *(pin["sha256"] for pin in source["completion_receipts"])]
        if (initial["state_admission_scope"] != "original_checkpoint_model_and_chain_with_actual_child_reset_state_observation"
                or initial["states"]["model_state_sha256"] != source["model_state_sha256"]
                or initial["ancestor_updates"] != len(ancestry) or initial["ancestor_completion_sha256"] != ancestry
                or not _equal(initial["ancestor_cost"], parent["cost"]["known_lower_bound"])
                or initial["sampler"]["epoch"] != 0 or initial["sampler"]["cursor"] != 0):
            raise ValueError("utility fork must retain full source model/ancestry and reset its original new sampler")
    else:
        raise ValueError("original utility initialization tag required")


def _closure_join(archive, publication, closure, cost, replay, parent_unknown, retirement, projection):
    family = replay.family
    _closure(closure, _family=family)
    if any(closure[name] != publication[name] for name in ("execution_id", "window_id", "run_id")):
        raise ValueError("utility publication changed original process closure identity")
    if closure["kind"] == "optimizer_boundary":
        if cost["closure_tail"]["status"] != "unknown":
            raise ValueError("a running original boundary cannot claim known terminal expense")
        return "unresolved", None
    boundary_reference = closure["boundary_publication"]
    boundary = _parse_utility_checkpoint_publication(archive.raw(boundary_reference),
        publication_sha256=boundary_reference["sha256"], publication_bytes=boundary_reference["bytes"], _family=family)
    if not _equal({key: value for key, value in publication.items() if key not in ("closure", "cost")},
            {key: value for key, value in boundary.items() if key not in ("closure", "cost")}):
        raise ValueError("final utility publication changed its preserved original running boundary")
    boundary_closure = _closure(archive.json(boundary["closure"]), _family=family)
    boundary_cost = validate_utility_checkpoint_cost(archive.json(boundary["cost"]))
    if (boundary_closure["kind"] != "optimizer_boundary"
            or any(not _equal(boundary_closure[name], closure[name]) for name in ("execution_id", "window_id", "run_id", "child_identity"))
            or boundary_cost["closure_tail"]["status"] != "unknown" or cost["closure_tail"]["status"] != "known"
            or any(not _equal(boundary_cost[name], cost[name]) for name in ("snapshot_cost", "ancestor_cost", "post_snapshot_delta"))
            or not _equal(boundary_cost["inherited_unknown_tails"], parent_unknown)):
        raise ValueError("final utility closure must preserve its original boundary identity and Unknown cost evidence")
    boundary_unknown = {"checkpoint_publication_sha256": boundary_reference["sha256"],
        "execution_id": boundary["execution_id"], "window_id": boundary["window_id"],
        "tail": copy.deepcopy(boundary_cost["closure_tail"])}
    if boundary_cost["closure_tail"]["evidence"] is not None:
        archive.verify(boundary_cost["closure_tail"]["evidence"], MAX_PROJECTION_BYTES)
    terminal = archive.json(closure["terminal_receipt"])
    _fields(terminal, ("schema", "execution_id", "window_id", "run_id", "end_payload", "process_receipt",
        "child_identity", "original_child_terminal", "historical_evidence_only", "execution_authorized"), "original utility parent terminal")
    _history(terminal)
    if (terminal["schema"] != family.domain(TERMINAL_DOMAIN)
            or any(terminal[name] != closure[name] for name in ("execution_id", "window_id", "run_id"))
            or not _equal(terminal["process_receipt"], closure["process_receipt"])
            or not _equal(terminal["child_identity"], closure["child_identity"])):
        raise ValueError("original utility terminal changed held process/child identity")
    end = terminal["end_payload"]
    _fields(end, ("schema", "status", "completed_updates", "dispatched_updates", "window_dispatched_updates", "ancestor_updates",
        "cost", "pending_microbatches", "quarantined", "physical_completion", "checkpoint", "result"), "original utility end payload")
    replay.counts(end); replay.observed_cost(end["cost"])
    if (end["schema"] != family.domain("rz-rovent-utility-trainer-end/1") or type(end["dispatched_updates"]) is not int
            or end["dispatched_updates"] != replay.dispatched or type(end["window_dispatched_updates"]) is not int
            or end["window_dispatched_updates"] != replay.dispatched - replay.initial["dispatched_updates"]
            or type(end["pending_microbatches"]) is not int or end["pending_microbatches"] != 0
            or type(end["quarantined"]) is not bool or end["physical_completion"] not in ("confirmed", "unknown")
            or end["status"] not in ("window_complete", "window_limit", "partial_accumulation_discarded", "export_complete", "export_failed")
            or replay.pending and end["status"] != "partial_accumulation_discarded"):
        raise ValueError("original utility end changed actual dispatch, physical or partial-discard observations")
    checkpoint = end["checkpoint"]
    _fields(checkpoint, ("path", "sha256", "bytes", "tensor_projection", "completed_updates", "state_sha256",
        "snapshot_cost", "post_snapshot_cost"), "original saved utility terminal checkpoint")
    original_request = retirement["original_request"]
    if (not _equal({name: checkpoint[name] for name in ("path", "sha256", "bytes")}, original_request["checkpoint"])
            or not _equal(checkpoint["tensor_projection"], original_request["tensor_projection"])
            or type(checkpoint["completed_updates"]) is not int or checkpoint["completed_updates"] != projection["completed_updates"]
            or checkpoint["state_sha256"] != projection["payload_state_sha256"]
            or any(not _equal(checkpoint[name], original_request[name]) for name in ("snapshot_cost", "post_snapshot_cost"))):
        raise ValueError("terminal utility checkpoint differs from its actual selected retirement boundary")
    child_reference = _relative_ref(terminal["original_child_terminal"], MAX_PROJECTION_BYTES)
    child = archive.json(child_reference)
    optional = ("error_type", "error_chain_types") if type(child) is dict and "error_type" in child else ()
    _fields(child, ("schema", "execution_id", "window_id", "status", "result", "bootstrap", "completed_updates",
        "window_completed_updates", "dispatched_updates", "window_dispatched_updates", "physical_completion",
        "physical_process_retirement", "arena_eligible", *optional), "original raw utility child terminal")
    if (child["schema"] != family.domain("rz-rovent-utility-trainer-result/1") or child["execution_id"] != publication["execution_id"]
            or child["window_id"] != publication["window_id"] or child["status"] != "completed" or optional
            or child["arena_eligible"] is not False or child["physical_completion"] != "confirmed"
            or child["physical_process_retirement"] != "original_parent_join_required"
            or any(type(child[name]) is not int or child[name] != end[name] for name in ("completed_updates", "dispatched_updates", "window_dispatched_updates"))
            or type(child["window_completed_updates"]) is not int
            or child["window_completed_updates"] != replay.completed - replay.initial["completed_updates"]):
        raise ValueError("raw utility child terminal differs from its original parent end/physical evidence")
    tail = cost["closure_tail"]
    if tail["status"] == "known":
        delta = {name: _uint(end["cost"][name] - replay.highwater[name]) for name in COST_FIELDS}
        if not _equal(tail["evidence"], closure["terminal_receipt"]) or not _equal(tail["delta"], delta):
            raise ValueError("known utility tail differs from the original terminal expense evidence")
    return ("recorded_closed" if closure["process_receipt"]["group_cleanup"] == "gone"
        and closure["child_reap"] == "owned_child_reaped" and closure["physical_completion"] == "confirmed"
        and end["physical_completion"] == "confirmed" and end["quarantined"] is False else "unresolved"), boundary_unknown


def _projection_source_join(projection, window, registration, parent):
    expected_devices = [] if window["device"] == "cpu" else [int(window["device"][5:])]
    if projection["cuda_devices"] != expected_devices:
        raise ValueError("original utility tensor CUDA RNG roster differs from its exact window device")
    if parent is None:
        expected_source_cost, expected_parent_tensor = None, None
    else:
        expected_source_cost = parent["cost"]
        expected_parent_tensor = (parent["publication"]["tensor"]["sha256"] if registration["initialization_source"]["mode"] == "fork"
            else parent["projection"]["parent_checkpoint_sha256"])
    if (not _equal(projection["source_cost_observation"], expected_source_cost)
            or projection["parent_checkpoint_sha256"] != expected_parent_tensor):
        raise ValueError("original utility source cost or inherited fork tensor pointer changed")


def _source(archive, reference, datasets, *, depth=0, _family=None):
    family = _codec(_family)
    if depth > 256 or len(archive.completed) + len(archive.active) >= 257:
        raise ValueError("bounded original utility checkpoint publication ancestry required")
    _relative_ref(reference, MAX_PROJECTION_BYTES)
    identity = (family.intervention, reference["sha256"], reference["bytes"])
    if identity in archive.active:
        raise ValueError("cyclic original utility checkpoint publication chain")
    if identity in archive.completed:
        return archive.completed[identity]
    archive.active.add(identity)
    try:
        publication = _parse_utility_checkpoint_publication(archive.raw(reference), publication_sha256=reference["sha256"],
            publication_bytes=reference["bytes"], _family=family)
        preparation = family.parts(archive.json(publication["original_preparation"])).preparation
        if not family.intervention and preparation["schema"] != COHORT_PREPARATION_DOMAIN:
            raise ValueError("historical actual utility source requires its original cohort preparation/2")
        window = family.validate_window(preparation, archive.json(publication["original_window"]))
        if family.intervention:
            from .rovent_utility_intervention_execution import validate_intervention_registration as validate_registration
        else:
            from .rovent_utility_training_bootstrap import validate_utility_registration_v2 as validate_registration
        registration_value = archive.json(publication["original_registration"])
        # The old /1 parser remains available as its original declaration
        # codec. It is not an actual cohort learner checkpoint source.
        registration = validate_registration(registration_value)
        if (any(publication[name] != registration[name] for name in ("execution_id", "trainer_source_sha256", "trainer_dependency_set_sha256"))
                or publication["trainer_python_executable_sha256"] != registration["trainer_python_executable"]["sha256"]
                or any(publication[name] != window[name] for name in ("window_id", "run_id", "dataset_sha256", "preparation_sha256"))
                or publication["window_spec_sha256"] != family.window_digest(window)
                or preparation["preparation_sha256"] != publication["preparation_sha256"]
                or preparation["registration"]["trainer_source_sha256"] != registration["trainer_source_sha256"]
                or not _equal(_pin_identity(registration["preparation"]), _pin_identity(publication["original_preparation"]))
                or not _equal(_pin_identity(registration["window_spec"]), _pin_identity(publication["original_window"]))):
            raise ValueError("original utility publication changed source/runtime/window/preparation pins")
        candidates = [dataset for dataset in datasets if dataset.observation()["dataset_sha256"] == publication["dataset_sha256"]]
        if len(candidates) != 1:
            raise ValueError("one exact original historical utility dataset for each checkpoint stage required")
        dataset = _dataset_source(candidates[0], registration, _family=family)
        dataset.validate_execution_sources(registration)
        initial = _initial(archive.json(publication["initial_ledger_state"]), _family=family)
        parent = None
        if publication["parent_publication"] is not None:
            parent = _source(archive, publication["parent_publication"], datasets, depth=depth + 1, _family=family)
        _parent_join(initial, registration, parent)
        projection = _tensor_projection(_raw_json(archive.raw(publication["tensor_projection"]),
            expected_sha256=publication["tensor_projection"]["sha256"], expected_bytes=publication["tensor_projection"]["bytes"],
            maximum=MAX_PROJECTION_BYTES), _family=family)
        retirement = _parse_utility_checkpoint_retirement(archive.raw(publication["retirement"]),
            expected_sha256=publication["retirement"]["sha256"], expected_bytes=publication["retirement"]["bytes"], _family=family)
        for name, original in (("tensor", retirement["checkpoint"]),
                ("tensor_projection", retirement["original_request"]["tensor_projection"])):
            if not _equal(_pin_identity(publication[name]), _pin_identity(original)):
                raise ValueError("historical utility relocated files changed original raw byte pins")
        if not _equal(projection["preparation"], preparation):
            raise ValueError("historical utility tensor projection changed original preparation")
        _projection_source_join(projection, window, registration, parent)
        if (not _equal(initial["initial_source_declaration"], projection["metadata"]["initial_training"])
                or initial["frozen_base_weights_sha256"] != projection["metadata"]["base_weights_sha256"]
                or initial["frozen_base_mapping_sha256"] != projection["metadata"]["frozen_base_mapping_sha256"]):
            raise ValueError("historical utility tensor changed original frozen source observations")
        current, ancestor = publication["completion_receipts"], publication["ancestor_completion_receipts"]
        if (len(current) != projection["completed_updates"] or len(ancestor) != projection["ancestor_updates"]
                or not _equal([_pin_identity(pin) for pin in current], projection["completion_receipts"])
                or [pin["sha256"] for pin in ancestor] != projection["ancestor_completion_sha256"]
                or not _equal([_pin_identity(pin) for pin in initial["completion_receipts"]],
                    [_pin_identity(pin) for pin in current[:initial["completed_updates"]]])
                or [pin["sha256"] for pin in ancestor] != initial["ancestor_completion_sha256"]):
            raise ValueError("utility publication omitted/reordered original current or ancestor completion files")
        for receipt in (*ancestor, *current[:initial["completed_updates"]]):
            archive.verify(receipt, MAX_COMPLETION_RECEIPT_BYTES)
        replay = _Replay(archive, publication, preparation, window, initial, dataset, _family=family)
        replay.replay()
        request = retirement["original_request"]
        replay.checkpoint(request)
        parent_projection = retirement["parent_projection"]
        if (replay.completed != projection["completed_updates"] or replay.dispatched != projection["dispatched_updates"]
                or not _equal(replay.sampler, projection["sampler"])
                or not _equal(replay.receipts, projection["completion_receipts"])
                or not _equal(replay.derived(), projection["derived_training_state"])
                or not _equal(request["snapshot_cost"], projection["cost"])
                or not _equal(initial["ancestor_cost"], projection["ancestor_cost"])
                or any(request[name] != projection[name] for name in STATE_FIELDS)
                or parent_projection["unupdated_context_sha256"] != [row["context_sha256"] for row in replay.pending]):
            raise ValueError("utility full journal differs from selected original tensor/checkpoint boundary")
        cost = validate_utility_checkpoint_cost(archive.json(publication["cost"]))
        expected_unknown = [] if parent is None else copy.deepcopy(parent["cost"]["inherited_unknown_tails"])
        if parent is not None and parent["cost"]["closure_tail"]["status"] == "unknown":
            expected_unknown.append({"checkpoint_publication_sha256": parent["publication_pin"]["sha256"],
                "execution_id": parent["publication"]["execution_id"], "window_id": parent["publication"]["window_id"],
                "tail": copy.deepcopy(parent["cost"]["closure_tail"])})
        expected_unknown.sort(key=lambda item: (item["checkpoint_publication_sha256"], item["execution_id"], item["window_id"]))
        if (not _equal(cost["snapshot_cost"], projection["cost"]) or not _equal(cost["ancestor_cost"], projection["ancestor_cost"])
                or not _equal(cost["post_snapshot_delta"], {name: _uint(request["post_snapshot_cost"][name] - projection["cost"][name]) for name in COST_FIELDS})):
            raise ValueError("original utility snapshot/ancestor/post expense or unknown-tail conservation differs")
        for tail in (cost["closure_tail"], *(item["tail"] for item in cost["inherited_unknown_tails"])):
            if tail["evidence"] is not None:
                archive.verify(tail["evidence"], MAX_PROJECTION_BYTES)
        closure = _closure(archive.json(publication["closure"]), _family=family)
        classification, boundary_unknown = _closure_join(archive, publication, closure, cost, replay, expected_unknown, retirement, projection)
        if boundary_unknown is not None:
            expected_unknown.append(boundary_unknown)
            expected_unknown.sort(key=lambda item: (item["checkpoint_publication_sha256"], item["execution_id"], item["window_id"]))
        if not _equal(cost["inherited_unknown_tails"], expected_unknown):
            raise ValueError("original utility parent/boundary Unknown tails were omitted, repeated or relabeled")
        archive.verify(publication["tensor"], MAX_TENSOR_BYTES)
        source = {"publication_pin": copy.deepcopy(reference), "publication": publication, "preparation": preparation,
            "window": window, "registration": registration, "initial": initial, "projection": projection,
            "retirement": retirement, "cost": cost, "closure": closure, "classification": classification,
            "parent": parent}
        archive.completed[identity] = source
        return source
    finally:
        archive.active.remove(identity)


class HistoricalUtilityCheckpoint:
    """Detached verified evidence; current admission still needs the live parent."""
    def __init__(self):
        raise TypeError("read the exact original utility checkpoint publication")

    def projection(self):
        self._check()
        return copy.deepcopy(self._source["projection"])

    def cost_observation(self):
        self._check()
        return copy.deepcopy(self._source["cost"])

    def tensor_pin(self):
        self._check()
        return {"path": str(self._root / self._source["publication"]["tensor"]["path"]),
            **_pin_identity(self._source["publication"]["tensor"])}

    def _check(self):
        if digest("rz-rovent-utility-historical-checkpoint-owner/1", self._source) != self._seal:
            raise ValueError("original utility historical checkpoint evidence changed")

    def observation(self):
        return _historical_checkpoint_observation(self)


def _historical_checkpoint_observation(owner, *, _family=None):
    family = _codec(_family)
    owner._check()
    source, ancestors = owner._source, []
    current = source
    while current is not None:
        ancestors.append({"publication_pin": _pin_identity(current["publication_pin"]),
            "execution_id": current["publication"]["execution_id"], "window_id": current["publication"]["window_id"],
            "classification": current["classification"], "child_identity": copy.deepcopy(current["closure"]["child_identity"])})
        current = current["parent"]
    return {"schema": family.domain("rz-rovent-utility-historical-checkpoint-observation/1"),
        "publication_pin": _pin_identity(source["publication_pin"]), "tensor_pin": _pin_identity(source["publication"]["tensor"]),
        "payload_state_sha256": source["projection"]["payload_state_sha256"], "completed_updates": source["projection"]["completed_updates"],
        "ancestor_updates": source["projection"]["ancestor_updates"], "original_closure_chain": ancestors,
        "historical_evidence_only": True, "past_native_authority_reissued": False,
        "past_physical_completion_reissued": False, "old_run_reclamation_issued": False,
        "execution_authorized": False}


def _read_utility_historical_checkpoint(reference, *, read_accounting, datasets, _family=None):
    """Read full original archive evidence, with one shared bounded read budget.

    This neither unpickles `.pt` nor reclaims an old process. The actual new
    parent separately gates old-run closure and its new peer execution lease.
    A scratch tensor loader must independently join the immutable projection.
    """
    family = _codec(_family)
    _fields(reference, ("root", "publication"), "original utility checkpoint reference")
    if type(datasets) not in (tuple, list) or not 1 <= len(datasets) <= 257:
        raise ValueError("bounded original typed utility source datasets required")
    for dataset in datasets:
        family.require_dataset(dataset)
    _absolute_pin(reference["publication"], MAX_PROJECTION_BYTES)
    archive = _Archive(reference["root"], read_accounting)
    try:
        path = _absolute(reference["publication"]["path"])
        if type(path) is not type(archive.root) or not path.is_relative_to(archive.root) or path == archive.root:
            raise ValueError("original utility publication escaped its registered archive root")
        relative = {"path": path.relative_to(archive.root).as_posix(), **_pin_identity(reference["publication"])}
        source = _source(archive, relative, datasets, _family=family)
        if family.intervention:
            from .rovent_utility_intervention_checkpoint import HistoricalUtilityInterventionCheckpoint
            owner_type = HistoricalUtilityInterventionCheckpoint
        else:
            owner_type = HistoricalUtilityCheckpoint
        owner = owner_type.__new__(owner_type)
        owner._source, owner._root = source, archive.root
        owner._seal = digest(family.domain("rz-rovent-utility-historical-checkpoint-owner/1"), source)
        return owner
    finally:
        archive.close()


def read_utility_historical_checkpoint(reference, *, read_accounting, datasets):
    """Original /1 only; intervention history has its separate typed reader."""
    return _read_utility_historical_checkpoint(reference, read_accounting=read_accounting, datasets=datasets)
