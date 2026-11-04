"""Explicit intervention_v2 checkpoint lineage and current lifecycle boundary.

The original /1 readers remain fixed to their source family. Only these entry
points select the closed INTERVENTION private codec. Exact9 preparation bytes
are preserved; numerical seals come from the validated numerical-parts owner.
Historical reads grant no current peer lease, physical fence or learned export.
"""
from __future__ import annotations

import copy

from .generation_data import digest, uint
from .rovent_utility_training_family import INTERVENTION
from . import rovent_utility_historical_checkpoint as history
from . import rovent_utility_tensor_training as tensors


CHECKPOINT_DOMAIN = "rz-rovent-utility-tensor-checkpoint/2"
STATE_DOMAIN = "rz-rovent-utility-derived-training-state/2"
PROJECTION_DOMAIN = "rz-rovent-utility-tensor-projection/2"
PUBLICATION_DOMAIN = "rz-rovent-utility-checkpoint-publication/2"
RETIREMENT_DOMAIN = "rz-rovent-utility-checkpoint-retirement/2"
EXPORT_DOMAIN = "rz-rovent-utility-lifecycle-export/2"
HISTORICAL_OWNER_DOMAIN = "rz-rovent-utility-historical-checkpoint-owner/2"
HISTORICAL_SCRATCH_DOMAIN = "rz-rovent-utility-historical-scratch-observation/2"


def parse_intervention_utility_checkpoint_publication(raw, *, publication_sha256, publication_bytes):
    return history._parse_utility_checkpoint_publication(raw, publication_sha256=publication_sha256,
        publication_bytes=publication_bytes, _family=INTERVENTION)


def parse_intervention_utility_tensor_projection(raw, *, expected_sha256, expected_bytes):
    value = history._raw_json(raw, expected_sha256=expected_sha256, expected_bytes=expected_bytes,
        maximum=history.MAX_PROJECTION_BYTES)
    return history._tensor_projection(value, _family=INTERVENTION)


def parse_intervention_utility_checkpoint_retirement(raw, *, expected_sha256, expected_bytes):
    return history._parse_utility_checkpoint_retirement(raw, expected_sha256=expected_sha256,
        expected_bytes=expected_bytes, _family=INTERVENTION)


def project_intervention_utility_tensor_scratch(projection, *, observed_state):
    expected = history._tensor_projection(projection, _family=INTERVENTION)
    observed = history._tensor_projection(observed_state, _family=INTERVENTION)
    if not history._equal(expected, observed):
        raise ValueError("intervention original tensor/moments/sampler/RNG/expense projection differs")
    return {"schema": HISTORICAL_SCRATCH_DOMAIN, "status": "historical_integrity_only",
        "payload_state_sha256": expected["payload_state_sha256"],
        **{name: expected[name] for name in history.STATE_FIELDS},
        "completed_updates": expected["completed_updates"], "ancestor_updates": expected["ancestor_updates"],
        "snapshot_cost": copy.deepcopy(expected["cost"]), "past_native_authority_reissued": False,
        "past_physical_completion_reissued": False, "old_run_reclamation_issued": False,
        "execution_authorized": False}


class HistoricalUtilityInterventionCheckpoint(history.HistoricalUtilityCheckpoint):
    """Detached full /2 source replay, never a live authority constructor."""
    def __init__(self):
        raise TypeError("read the complete original intervention checkpoint archive")

    def _check(self):
        if digest(HISTORICAL_OWNER_DOMAIN, self._source) != self._seal:
            raise ValueError("original intervention historical checkpoint evidence changed")

    def observation(self):
        return history._historical_checkpoint_observation(self, _family=INTERVENTION)


def read_intervention_utility_historical_checkpoint(reference, *, read_accounting, datasets):
    return history._read_utility_historical_checkpoint(reference, read_accounting=read_accounting,
        datasets=datasets, _family=INTERVENTION)


def create_intervention_utility_owner(model, preparation, *, authority, sampler,
        sampler_validator, cuda_devices=(), execute=False):
    tensors._require_tensor_execution(execute)
    INTERVENTION.require_authority(authority, allow_unleased=True)
    parts = INTERVENTION.parts(preparation)
    return tensors._create_utility_owner(model, parts.preparation, sampler=sampler,
        sampler_validator=sampler_validator, cuda_devices=cuda_devices, execute=execute,
        numerical_parts=parts)


def derived_intervention_training_state(owner):
    return tensors._derived_training_state(owner, _family=INTERVENTION)


def intervention_utility_tensor_payload(owner, *, sampler_validator, execute=False):
    return tensors._utility_tensor_payload(owner, sampler_validator=sampler_validator,
        execute=execute, _family=INTERVENTION)


def intervention_utility_tensor_projection(owner, *, sampler_validator, execute=False):
    runtime = tensors._runtime(execute)
    payload = intervention_utility_tensor_payload(owner, sampler_validator=sampler_validator, execute=True)
    return tensors._tensor_projection(payload, runtime, _family=INTERVENTION)


def save_intervention_utility_checkpoint(owner, path, *, authority, lease, sampler_validator,
        post_snapshot_observer, max_file_bytes=tensors.MAX_CHECKPOINT_BYTES,
        read_accounting=None, execute=False):
    """Return only after the current peer ACK; uncertain files stay quarantined."""
    tensors._require_tensor_execution(execute)
    INTERVENTION.require_authority(authority, lease=lease, owner=owner)
    INTERVENTION.parts(owner.preparation)
    tensors._counts(owner)
    if (owner.physical_completion != "confirmed" or owner.quarantined
            or owner.completed_updates <= lease.initial_completed
            or not callable(post_snapshot_observer)):
        raise ValueError("intervention checkpoint needs this live lease's completed ACKed update and physical boundary")
    if read_accounting is not None and read_accounting is not authority.read_accounting:
        raise ValueError("intervention checkpoint must use its current authority's original read budget")
    try:
        boundary_states = authority._state(owner)
        boundary_derived = derived_intervention_training_state(owner)
        checkpoint = tensors._save_utility_checkpoint(owner, path, sampler_validator=sampler_validator,
            max_file_bytes=max_file_bytes, read_accounting=authority.read_accounting,
            execute=execute, _family=INTERVENTION)
        original = copy.deepcopy(checkpoint)
        # The shared runner charges serialization/elapsed work here, before
        # the parent observes post-snapshot cost. This callback cannot alter
        # the published file pins, snapshot preimage or completed state.
        post_snapshot_observer(checkpoint)
        if any(checkpoint[name] != original[name] for name in original if name != "post_snapshot_cost"):
            raise ValueError("intervention elapsed observer changed a published checkpoint preimage")
        if checkpoint["post_snapshot_cost"] != owner.cost:
            raise ValueError("intervention checkpoint omitted its actual post-publication expense")
        if (any(owner.cost[name] < checkpoint["snapshot_cost"][name] for name in tensors.COST_FIELDS)
                or owner.cost["output_bytes"] - checkpoint["snapshot_cost"]["output_bytes"]
                    != checkpoint["bytes"] + checkpoint["tensor_projection"]["bytes"]
                or authority._state(owner) != boundary_states
                or derived_intervention_training_state(owner) != boundary_derived):
            raise ValueError("intervention observer changed tensor boundary, completion chain or actual output expense")
        tensors._counts(owner)
        INTERVENTION.require_authority(authority, lease=lease, owner=owner)
        authority.checkpoint_published(lease, checkpoint, owner)
        return checkpoint
    except BaseException:
        owner.quarantined = True
        owner.physical_completion = "unknown"
        raise


def stage_intervention_utility_resume(path, expected_sha256, template, preparation, *, authority,
        historical_checkpoint, sampler_validator, max_file_bytes=tensors.MAX_CHECKPOINT_BYTES,
        read_accounting=None, snapshot_directory=None, execute=False):
    tensors._require_tensor_execution(execute)
    INTERVENTION.require_authority(authority, allow_unleased=True)
    parts = INTERVENTION.parts(preparation)
    if type(historical_checkpoint) is not HistoricalUtilityInterventionCheckpoint:
        raise ValueError("exact typed original intervention checkpoint required before scratch restore")
    if not history._equal(historical_checkpoint.projection()["preparation"], parts.preparation):
        raise ValueError("intervention resume cannot replace the original exact9 preparation")
    if read_accounting is not None and read_accounting is not authority.read_accounting:
        raise ValueError("intervention resume requires one current original read budget")
    return tensors._stage_utility_resume(path, expected_sha256, template, parts.preparation,
        sampler_validator=sampler_validator, max_file_bytes=max_file_bytes,
        read_accounting=authority.read_accounting, snapshot_directory=snapshot_directory,
        execute=execute, authority=authority, historical_checkpoint=historical_checkpoint,
        require_historical=True, _family=INTERVENTION)


def stage_intervention_utility_fork(source, *, preparation, authority, sampler,
        source_sampler_validator, sampler_validator, reset_seed, execute=False):
    tensors._require_tensor_execution(execute)
    INTERVENTION.require_authority(authority, allow_unleased=True)
    parts = INTERVENTION.parts(preparation)
    return tensors._stage_utility_fork(source, parts.preparation, sampler=sampler,
        source_sampler_validator=source_sampler_validator, sampler_validator=sampler_validator,
        reset_seed=reset_seed, execute=execute, _family=INTERVENTION)


def export_intervention_utility_owner(owner, path, *, authority, kind,
        max_output_bytes=8 * 1024**2, execute=False):
    tensors._require_tensor_execution(execute)
    INTERVENTION.require_authority(authority, owner=owner)
    INTERVENTION.parts(owner.preparation)
    if not uint(owner.total_updates(), tensors.MAX_UPDATES):
        raise ValueError("intervention learned export requires actual completed utility updates")
    return tensors._export_utility_owner(owner, path, authority=authority, kind=kind,
        max_output_bytes=max_output_bytes, execute=execute, _family=INTERVENTION)
