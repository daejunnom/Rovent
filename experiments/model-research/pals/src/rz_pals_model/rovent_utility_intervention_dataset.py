"""Original /2 data custody, separate from the current learner authority."""
from __future__ import annotations

import copy

from .generation_data import digest, sha
from .rovent_utility_intervention import COST_SCOPE
from .rovent_utility_intervention_files import (ACK_FILE, PUBLICATION_FILE, HistoricalInterventionFiles,
    read_intervention_files)
from .rovent_utility_intervention_payload import validate_intervention_payloads
from .rovent_utility_intervention_preparation import (_validate_ancestor_owners,
    validate_intervention_source)
from .rovent_utility_intervention_numerical import DATA_FAMILY, validate_intervention_numerical_parts
from .rovent_utility_route import _capture_projection


DATASET_DOMAIN = "rz-rovent-utility-training-dataset/2"
CONTEXT_DOMAIN = "rz-rovent-utility-cohort-context-source/2"
POOL_DOMAIN = "rz-rovent-utility-training-context-pool/2"
OBSERVATION_DOMAIN = "rz-rovent-historical-utility-dataset-admission/2"


def _raw_pin(files, name):
    return {key: files.file_pin(name)[key] for key in ("sha256", "bytes")}


def source_metadata(original):
    """Projection of already checked original operands, never an admission."""
    report, files = original["report"], original["original_files"]
    value = {name: copy.deepcopy(report[name]) for name in ("cohort_id", "profile_sha256", "neutral_input_sha256",
        "neutral_target_sha256", "neutral_context_sha256", "root_rules_sha256", "role_queries", "frozen_pc_sources",
        "ancestor_datasets", "known_pair_sha256", "all_pair_sha256")}
    value.update(publication=_raw_pin(files, PUBLICATION_FILE), durable_ack=_raw_pin(files, ACK_FILE))
    return validate_intervention_source(value)


def _projection(original, parts):
    report, files = original["report"], original["original_files"]
    registered, lineage = parts.registration, report["lineage"]
    value = {"schema": CONTEXT_DOMAIN, "data_family": DATA_FAMILY,
        "publication_sha256": files.file_pin(PUBLICATION_FILE)["sha256"],
        "durable_ack_sha256": files.file_pin(ACK_FILE)["sha256"],
        **{name: copy.deepcopy(report[name]) for name in ("cohort_id", "profile_sha256", "neutral_input_sha256",
            "neutral_target_sha256", "neutral_context_sha256", "role_queries", "root_rules_sha256", "transposition_sha256",
            "all_pair_sha256", "known_pair_sha256", "known_pairs", "split")},
        "collection_source_sha256": registered["collection_source_sha256"],
        "first_trial_cost_epoch_anchor_sha256": registered["cost_epoch_sha256"],
        "resource_profile_sha256": registered["resource_profile_sha256"],
        "frozen_base_weights_sha256": registered["frozen_base"]["weights_sha256"],
        **{name: lineage[name] for name in ("opening_id", "game_id", "line_genealogy_id")},
        "ancestor_dataset_publication_sha256": sorted(item["publication_sha256"] for item in report["ancestor_datasets"]),
        "cost_scope": COST_SCOPE, "execution_authorized": False}
    value["context_sha256"] = digest(CONTEXT_DOMAIN, value)
    return value


def read_intervention_utility_dependencies(reference, *, preparation, read_budget):
    """Read original dependencies without admitting ancestors or a learner.

    Full dataset admission opens and charges these files again. This metadata
    pass keeps the original /2 FD, payload and exact9 source joins intact; it
    never replaces an explicit typed ancestor owner with a cached declaration.
    """
    parts = validate_intervention_numerical_parts(preparation)
    files = read_intervention_files(reference, read_accounting=read_budget)
    original = validate_intervention_payloads(files, registration=parts.registration)
    if source_metadata(original) != parts.preparation["intervention_source"]:
        raise ValueError("intervention dependencies differ from original preparation source")
    read_budget._check()
    return {"ancestor_dataset_publication_sha256": [
        item["publication_sha256"] for item in original["report"]["ancestor_datasets"]]}


class HistoricalUtilityInterventionDataset:
    """One original cohort data owner; no constructor for live authority."""
    def __init__(self):
        raise TypeError("use the strict original intervention publication/ACK reader")

    @classmethod
    def from_registration(cls, reference, *, preparation, read_budget, ancestors=()):
        if cls is not HistoricalUtilityInterventionDataset:
            raise TypeError("exact original intervention dataset type required")
        parts = validate_intervention_numerical_parts(preparation)
        files = read_intervention_files(reference, read_accounting=read_budget)
        read_budget._check()
        original = validate_intervention_payloads(files, registration=parts.registration)
        if source_metadata(original) != parts.preparation["intervention_source"]:
            raise ValueError("intervention preparation differs from complete original source publication")
        _validate_ancestor_owners(original, ancestors, read_budget)
        projection = _projection(original, parts)
        ids = (projection["context_sha256"],)
        pool = digest(POOL_DOMAIN, list(ids))
        publication, ack = _raw_pin(files, PUBLICATION_FILE), _raw_pin(files, ACK_FILE)
        dataset = digest(DATASET_DOMAIN, {"data_family": DATA_FAMILY, "publication": publication, "durable_ack": ack,
            "contexts": list(ids), "pool_sha256": pool, "source_registration_sha256": parts.registration_sha256})
        owner = cls.__new__(cls)
        owner._files, owner._parts, owner._original = files, parts, original
        owner._projection, owner._context_ids, owner._pool_sha256, owner._dataset_sha256 = projection, ids, pool, dataset
        owner._publication_pin, owner._durable_ack_pin = publication, ack
        owner._state_sha256 = owner._state_digest()
        read_budget._check()
        return owner

    def _state_digest(self):
        return digest(CONTEXT_DOMAIN, {"kind": "owner_integrity", "preparation": self._parts.preparation,
            "report": self._original["report"], "projection": self._projection, "context_ids": list(self._context_ids),
            "pool_sha256": self._pool_sha256, "dataset_sha256": self._dataset_sha256,
            "publication": self._publication_pin, "durable_ack": self._durable_ack_pin,
            "capture": _capture_projection(self._original["input"])})

    def _check(self):
        if type(self._files) is not HistoricalInterventionFiles or self._state_digest() != self._state_sha256:
            raise ValueError("original intervention dataset custody changed")
        if (_raw_pin(self._files, PUBLICATION_FILE) != self._publication_pin
                or _raw_pin(self._files, ACK_FILE) != self._durable_ack_pin):
            raise ValueError("original intervention publication/ACK pin changed")
        self._files.raw_file(PUBLICATION_FILE)
        self._files.raw_file(ACK_FILE)

    @property
    def context_ids(self):
        self._check()
        return self._context_ids

    @property
    def pool_sha256(self):
        self._check()
        return self._pool_sha256

    @property
    def publication_pin(self):
        self._check()
        return copy.deepcopy(self._publication_pin)

    @property
    def durable_ack_pin(self):
        self._check()
        return copy.deepcopy(self._durable_ack_pin)

    def source_ancestor_descriptors(self):
        self._check()
        return copy.deepcopy(self._original["report"]["ancestor_datasets"])

    def historical_context(self, context_sha256):
        self._check()
        if sha(context_sha256) != self._context_ids[0]:
            raise ValueError("context outside original intervention publication")
        return {"context_sha256": context_sha256, "source_projection": copy.deepcopy(self._projection),
            "pair_sha256": copy.deepcopy(self._original["report"]["known_pair_sha256"]),
            "unique_known_pairs": len(self._original["report"]["known_pairs"])}

    def sample(self, index):
        self._check()
        if type(index) is not int or index != 0 or self._original["report"]["split"] != "train":
            raise ValueError("only the original train intervention can provide a sample")
        return {"input": self._original["input"], "known_pairs": copy.deepcopy(self._original["report"]["known_pairs"]),
            "source_projection": copy.deepcopy(self._projection)}

    def validate_execution_sources(self, registration):
        self._check()
        if registration.get("schema") != "rz-rovent-utility-training-registration/3" or registration.get("data_family") != DATA_FAMILY:
            raise ValueError("explicit original intervention execution registration/3 required")
        current = registration["current_dataset"]
        for name, original in (("publication", self._publication_pin), ("durable_ack", self._durable_ack_pin)):
            if any(current[name][key] != original[key] for key in ("sha256", "bytes")):
                raise ValueError("intervention execution changed original publication/ACK")
        if any(registration["frozen_nnuec_base"][key] != self._files.file_pin("nnuec-weights.json")[key]
                for key in ("sha256", "bytes")):
            raise ValueError("intervention execution changed original frozen NNUEC raw weights")
        frozen = registration["frozen_pc_sources"]
        originals = self._original["report"]["frozen_pc_sources"]
        if type(frozen) is not list or len(frozen) != 2 or len({item["role"] for item in frozen}) != 2:
            raise ValueError("both original frozen P/C execution sources required")
        for source in frozen:
            original = next((item for item in originals if item["role"] == source["role"]), None)
            if original is None or any(source[key] != original[key] for key in (
                    "model_semantics_sha256", "model_identity_sha256", "encoding_sha256")) or any(
                    source["artifact"][key] != original["artifact"][key] for key in ("sha256", "bytes")):
                raise ValueError("intervention execution changed original frozen P/C identity")

    def observation(self):
        self._check()
        report = self._original["report"]
        return {"schema": OBSERVATION_DOMAIN, "data_family": DATA_FAMILY, "dataset_sha256": self._dataset_sha256,
            "publication_sha256": self._publication_pin["sha256"], "publication_pin": self.publication_pin,
            "durable_ack_pin": self.durable_ack_pin, "pool_sha256": self._pool_sha256,
            "preparation_sha256": self._parts.preparation_sha256, "context_ids": list(self._context_ids),
            "record_count": 1, "trial_count": self._files.publication()["trial_count"],
            "pair_count": self._files.publication()["pair_count"], "split": report["split"],
            **{name: report["lineage"][name] for name in ("opening_id", "game_id", "line_genealogy_id")},
            "transposition_sha256": report["transposition_sha256"], "rules_state_sha256": report["root_rules_sha256"],
            "unique_known_pairs": len(report["known_pairs"]), "execution_authorized": False,
            "training_target_authorized": False, "past_native_authority_reissued": False,
            "past_physical_completion_reissued": False, "rules_replay_scope": "original_rust_parent_required",
            "pair_arithmetic_scope": "original_candidate_values_rust_conditional_certificate_required"}
