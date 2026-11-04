"""Explicit durable-recovery metadata; no runtime or old authority is revived.

The original tensor /1 bytes remain an exact externally pinned base. This new
codec preserves publication/chain/reclamation references and cost uncertainty
beside that base. All APIs below are pure stdlib validation. File IO, complete
corpus/chain verification, scratch tensor validation and admission of a *new*
owned window belong to the Rust recovery owner and separate runtime consumer.
"""
from __future__ import annotations

import copy

from .generation_data import digest, fields, identifier, sha, uint

RECOVERED_CHECKPOINT_DOMAIN = "rz-rovent-recovered-checkpoint/1"
COST_DOMAIN = "rz-rovent-recovered-cost-state/1"
BASE_DOMAIN = "rz-rovent-tensor-checkpoint/1"
MAX_ARTIFACT_BYTES = 18 * 1024**3
MAX_WINDOWS = 65536
COST_FIELDS = ("forward_calls", "backward_calls", "samples", "wall_time_ns", "partial_accumulation_ns",
               "replay_ns", "masked_groups", "output_bytes")


def artifact_reference(value):
    """A portable reference under the explicit publication root; never opened."""
    fields(value, ("path", "sha256", "bytes"), "historical publication artifact reference")
    path = value["path"]
    if (type(path) is not str or not path or len(path.encode("utf-8")) > 4096
            or path.startswith("/") or "\\" in path or ":" in path
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in path)
            or any(part in ("", ".", "..") for part in path.split("/"))):
        raise ValueError("normalized relative historical artifact path required")
    sha(value["sha256"])
    if not uint(value["bytes"], MAX_ARTIFACT_BYTES):
        raise ValueError("positive exact historical artifact byte pin required")
    return copy.deepcopy(value)


def _cost(value):
    fields(value, COST_FIELDS, "recorded cost counters")
    return {name: uint(value[name]) for name in COST_FIELDS}


def recorded_tail_cost(value):
    """Unknown tail stays unknown, with its evidence/reason intact."""
    if type(value) is not dict:
        raise ValueError("typed recorded tail cost required")
    if value.get("status") == "known":
        fields(value, ("status", "delta", "evidence"), "known recorded tail cost")
        _cost(value["delta"])
        artifact_reference(value["evidence"])
    elif value.get("status") == "unknown":
        fields(value, ("status", "reason", "evidence"), "unknown recorded tail cost")
        identifier(value["reason"])
        if value["evidence"] is not None:
            artifact_reference(value["evidence"])
    else:
        raise ValueError("recorded tail cost status must be known or unknown")
    return copy.deepcopy(value)


def _add(left, right):
    return {name: uint(left[name] + right[name]) for name in COST_FIELDS}


def _inherited_unknown_tails(value):
    if type(value) is not list or len(value) > MAX_WINDOWS:
        raise ValueError("bounded inherited unknown publication tails required")
    seen = set()
    publications = set()
    for row in value:
        fields(row, ("checkpoint_publication_sha256", "execution_id", "window_id", "tail"),
            "inherited unknown publication tail")
        publication = sha(row["checkpoint_publication_sha256"])
        identifier(row["execution_id"])
        identifier(row["window_id"])
        identity = (publication, row["execution_id"], row["window_id"])
        if identity in seen or publication in publications:
            raise ValueError("inherited unknown publication tails cannot be duplicated or relabelled")
        seen.add(identity)
        publications.add(publication)
        tail = recorded_tail_cost(row["tail"])
        if tail["status"] != "unknown":
            raise ValueError("inherited tail list preserves unknowns; known deltas belong to the selected snapshot")
    return copy.deepcopy(value)


def _cost_projection(snapshot, tail, windows, inherited):
    lower_bound = dict(snapshot)
    if tail["status"] == "known":
        lower_bound = _add(lower_bound, tail["delta"])
    for window in windows:
        lower_bound = _add(lower_bound, window["delta"])
    unknown = tail["status"] == "unknown" or bool(inherited)
    return lower_bound, "lower_bound_only" if unknown else "recorded_scope_known"


def initial_recovered_cost(snapshot_cost, tail_cost, *, inherited_unknown_tails=None):
    """Choose one confirmed snapshot without erasing older chain uncertainty.

    The historical parent must join the complete publication chain and supply
    every inherited unknown. This pure codec cannot prove that the supplied
    chain is complete and does not turn a snapshot into a closure proof.
    """
    snapshot, tail = _cost(snapshot_cost), recorded_tail_cost(tail_cost)
    inherited = _inherited_unknown_tails([] if inherited_unknown_tails is None else inherited_unknown_tails)
    lower_bound, status = _cost_projection(snapshot, tail, [], inherited)
    return {"schema": COST_DOMAIN, "historical_snapshot_cost": snapshot, "historical_tail_cost": tail,
        "inherited_unknown_tails": inherited,
        "new_window_deltas": [], "known_lower_bound": lower_bound, "total_status": status,
        "scope": "recorded_cost_only_not_old_process_closure_or_new_window_authority",
        "past_native_authority_reissued": False, "execution_authorized": False}


def validate_recovered_cost(value):
    fields(value, ("schema", "historical_snapshot_cost", "historical_tail_cost", "inherited_unknown_tails", "new_window_deltas",
        "known_lower_bound", "total_status", "scope", "past_native_authority_reissued", "execution_authorized"), "recovered cost state")
    if (value["schema"] != COST_DOMAIN
            or value["scope"] != "recorded_cost_only_not_old_process_closure_or_new_window_authority"
            or value["past_native_authority_reissued"] is not False or value["execution_authorized"] is not False):
        raise ValueError("cost metadata cannot issue old or new execution authority")
    snapshot, tail = _cost(value["historical_snapshot_cost"]), recorded_tail_cost(value["historical_tail_cost"])
    inherited = _inherited_unknown_tails(value["inherited_unknown_tails"])
    windows = value["new_window_deltas"]
    if type(windows) is not list or len(windows) > MAX_WINDOWS:
        raise ValueError("bounded explicit new-window cost deltas required")
    seen = set()
    for window in windows:
        fields(window, ("execution_id", "window_id", "delta", "evidence"), "new-window recorded cost delta")
        identifier(window["execution_id"])
        identifier(window["window_id"])
        identity = (window["execution_id"], window["window_id"])
        if identity in seen:
            raise ValueError("a new-window cost delta cannot be charged twice")
        seen.add(identity)
        _cost(window["delta"])
        artifact_reference(window["evidence"])
    lower_bound, status = _cost_projection(snapshot, tail, windows, inherited)
    if _cost(value["known_lower_bound"]) != lower_bound or value["total_status"] != status:
        raise ValueError("recovered cost changed its recorded uncertainty or known lower bound")
    return copy.deepcopy(value)


def append_new_window_cost(value, *, execution_id, window_id, delta, evidence):
    """Append an explicit delta; caller still needs the new window's owner."""
    result = validate_recovered_cost(value)
    identifier(execution_id)
    identifier(window_id)
    if len(result["new_window_deltas"]) >= MAX_WINDOWS:
        raise ValueError("new-window cost observation capacity exhausted")
    if any((row["execution_id"], row["window_id"]) == (execution_id, window_id)
            for row in result["new_window_deltas"]):
        raise ValueError("a new-window cost delta cannot be charged twice")
    result["new_window_deltas"].append({"execution_id": execution_id, "window_id": window_id,
        "delta": _cost(delta), "evidence": artifact_reference(evidence)})
    result["known_lower_bound"], result["total_status"] = _cost_projection(result["historical_snapshot_cost"],
        result["historical_tail_cost"], result["new_window_deltas"], result["inherited_unknown_tails"])
    return result


def validate_recovered_checkpoint(value):
    """Validate the explicit new codec without converting a legacy checkpoint.

    Passing this function means only that a metadata envelope is well formed.
    It proves no artifact exists and no old run has been reclaimed. The actual
    recovery reader must join every referenced byte and original chain before
    its separate new-process owner considers admission.
    """
    fields(value, ("schema", "base", "checkpoint_publication", "dataset_publication",
        "ancestor_dataset_publications", "update_chain", "reclamation_evidence", "cost_state",
        "source_enrollment_scope", "past_native_authority_reissued", "execution_authorized"), "explicit recovered checkpoint")
    if (value["schema"] != RECOVERED_CHECKPOINT_DOMAIN
            or value["source_enrollment_scope"] != "historical_exact_pins_new_window_authority_required"
            or value["past_native_authority_reissued"] is not False or value["execution_authorized"] is not False):
        raise ValueError("new recovered checkpoint domain required without restored live authority")
    fields(value["base"], ("schema", "artifact"), "exact unmodified tensor checkpoint base")
    if value["base"]["schema"] != BASE_DOMAIN:
        raise ValueError("an exact original tensor /1 base is required; automatic conversion is forbidden")
    references = [artifact_reference(value["base"]["artifact"])]
    for name in ("checkpoint_publication", "dataset_publication", "update_chain", "reclamation_evidence"):
        references.append(artifact_reference(value[name]))
    ancestors = value["ancestor_dataset_publications"]
    if type(ancestors) is not list or len(ancestors) > 256:
        raise ValueError("bounded complete historical dataset ancestry required")
    references.extend(artifact_reference(item) for item in ancestors)
    if len({pin["path"] for pin in references}) != len(references):
        raise ValueError("distinct recovery artifacts cannot alias one publication path")
    publications = [value["dataset_publication"], *ancestors]
    if len({pin["sha256"] for pin in publications}) != len(publications):
        raise ValueError("duplicate historical dataset publication ancestry")
    validate_recovered_cost(value["cost_state"])
    return copy.deepcopy(value)


def recovered_checkpoint_digest(value):
    return digest(RECOVERED_CHECKPOINT_DOMAIN, validate_recovered_checkpoint(value))
