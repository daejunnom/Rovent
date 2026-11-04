"""A new live process lease consuming historical data, never old authority.

Only the existing kernel-authenticated Unix transport is reused. This protocol
has separate startup/RPC/ack domains and explicit /2 learner payloads. The
parent joins historical projections against its held publication and current
reclamation owner before admitting the new window.
"""
from __future__ import annotations

import copy

from .generation_data import fields
from .rovent_training import membership_digest
from .rovent_training_authority import ParentSocketTrainerAuthority, _WindowLease
from .rovent_training_window import OriginalTrainingAuthorityRequired, retained_seal_set
from .rovent_training_window_v2 import training_profile_digest
from .rovent_historical_checkpoint import project_historical_checkpoint


class RecoveredParentSocketTrainerAuthority(ParentSocketTrainerAuthority):
    STARTUP_SCHEMA = "rz-rovent-recovered-training-startup/1"
    RPC_SCHEMA = "rz-rovent-recovered-trainer-rpc/1"
    ACK_SCHEMA = "rz-rovent-recovered-trainer-ack/1"

    @classmethod
    def connect_registered(cls, *args, read_accounting=None, **kwargs):
        from .rovent_training_bootstrap import PinnedReadBudget
        if type(read_accounting) is not PinnedReadBudget:
            raise OriginalTrainingAuthorityRequired("the new process's one original file-read budget is required")
        result = super().connect_registered(*args, **kwargs)
        result._read_budget = read_accounting
        return result

    def attach_historical_checkpoint(self, source):
        """Retain exact historical operands; attachment grants no permission."""
        from .rovent_recovered_training_window import HistoricalCheckpointInput
        if type(source) is not HistoricalCheckpointInput or getattr(self, "_historical_source", None) is not None:
            raise ValueError("one explicit historical checkpoint input per new process required")
        if self._lease is not None:
            raise OriginalTrainingAuthorityRequired("checkpoint operands must precede new-window dispatch")
        if source.read_budget is not self._read_budget:
            raise ValueError("all historical and tensor reads must share the new process's original credit")
        self._historical_source = source

    def begin_window(self, owner, intake, spec, admission):
        if self._lease is not None:
            raise OriginalTrainingAuthorityRequired("one recovered process cannot issue a second window lease")
        payload = {"schema": "rz-rovent-trainer-begin/2", "recipe_sha256": owner.recipe.sha256,
            "stage_id": owner.stage_id, "training_profile_sha256": training_profile_digest(spec),
            "corpus_sha256": admission["corpus_sha256"],
            **{name: spec[name] for name in ("collection_source_sha256", "collection_runtime_library_sha256",
                "trainer_source_sha256", "trainer_python_executable_sha256", "trainer_dependency_set_sha256",
                "training_model_semantics_sha256")},
            "completed_updates": dict(owner.completed), "ancestor_updates": dict(owner.ancestor_updates),
            "frozen_pc_sha256": spec["frozen_pc_sha256"], **self._auxiliary_state(owner)}
        if self._rpc("begin_window", payload):
            raise ValueError("recovered begin acknowledgement cannot serialize a capability")
        self._lease = _WindowLease(self)
        return self._lease

    def consume_sample(self, lease, sample):
        self._checked_lease(lease)
        payload = {"schema": "rz-rovent-trainer-consume/2", **{name: copy.deepcopy(sample[name]) for name in
            ("input_sha256", "origin_sha256", "record_sha256", "native_retirement_sha256", "head_raw_sha256", "role", "split")}}
        if self._rpc("consume_sample", payload):
            raise ValueError("recovered consume acknowledgement must be empty")

    def consume_auxiliary(self, lease, evidence):
        self._checked_lease(lease)
        payload = {"schema": "rz-rovent-trainer-consume-auxiliary/1", "evidence": copy.deepcopy(evidence)}
        if self._rpc("consume_auxiliary", payload):
            raise ValueError("separate R auxiliary acknowledgement must be empty")

    @staticmethod
    def _auxiliary_state(owner):
        if "r_auxiliary" not in owner.samplers:
            return {}
        from .rovent_r_auxiliary import checkpoint_projection
        return {"r_auxiliary": checkpoint_projection(owner.samplers["r_auxiliary"], owner.samplers["updates"])}

    def completed_update(self, lease, event, owner):
        self._checked_lease(lease)
        return self._rpc("completed_update", {"schema": "rz-rovent-trainer-update/2", "event": copy.deepcopy(event),
            **self._state(owner), "cost": dict(owner.cost), "pending_microbatches": owner.pending_microbatches})

    def checkpoint_published(self, lease, checkpoint, owner):
        self._checked_lease(lease)
        payload = {"schema": "rz-rovent-trainer-checkpoint/2",
            "checkpoint": {name: checkpoint[name] for name in ("path", "sha256", "bytes")},
            "completed_updates": dict(owner.completed), **self._state(owner),
            "cost": dict(owner.cost), "pending_microbatches": owner.pending_microbatches, **self._auxiliary_state(owner)}
        if self._rpc("checkpoint_published", payload):
            raise ValueError("recovered checkpoint acknowledgement must be empty")

    def end_window(self, lease, *, status, owner):
        self._checked_lease(lease)
        payload = {"schema": "rz-rovent-trainer-end/2", "status": status,
            "completed_updates": dict(owner.completed), "dispatched_updates": dict(owner.dispatched),
            "cost": dict(owner.cost), "pending_microbatches": owner.pending_microbatches,
            "quarantined": owner.quarantined, "physical_completion": owner.physical_completion}
        try:
            if self._rpc("end_window", payload, cleanup=True):
                raise ValueError("recovered end acknowledgement must be empty")
        finally:
            lease.closed = True

    def _historical_request(self, owner, checkpoint):
        source = getattr(self, "_historical_source", None)
        if source is None:
            raise OriginalTrainingAuthorityRequired("exact historical checkpoint operands were not attached")
        if any(checkpoint[name] != source.tensor_pin[name] for name in ("sha256", "bytes")):
            raise ValueError("scratch tensor differs from the exact historical checkpoint input")
        observed = {**self._state(owner), "completed_updates": dict(owner.completed),
            "cost": dict(owner.cost), "pending_microbatches": owner.pending_microbatches, **self._auxiliary_state(owner)}
        projection = project_historical_checkpoint(source.retirement_raw,
            receipt_sha256=source.retirement_pin["sha256"], receipt_bytes=source.retirement_pin["bytes"],
            recovered_checkpoint_pin=source.tensor_pin, observed_state=observed)
        inputs = retained_seal_set(owner.inherited_input_sha256, "input")
        origins = retained_seal_set(owner.inherited_origin_sha256, "origin")
        return {"projection": projection, "recipe_sha256": owner.recipe.sha256, "stage_id": owner.stage_id,
            "membership_sha256": membership_digest(owner.membership), "ancestor_updates": dict(owner.ancestor_updates),
            "retained_input_set_sha256": inputs["sha256"], "retained_input_count": inputs["count"],
            "retained_origin_set_sha256": origins["sha256"], "retained_origin_count": origins["count"],
            **self._auxiliary_state(owner)}

    def validate_checkpoint_state(self, owner, intake, checkpoint):
        return self._rpc("validate_checkpoint_state", {"schema": "rz-rovent-trainer-resume/2",
            **self._historical_request(owner, checkpoint)})

    def validate_learned_export(self, owner, intake, checkpoint):
        return self._rpc("validate_learned_export", {"schema": "rz-rovent-trainer-export/2",
            **self._historical_request(owner, checkpoint)})
