"""Historical cohort data for the separate, current utility learner.

The original payload verifier checks native framing and independent trial
metadata. The Rust parent separately replays Rules and owns original FDs,
trial closure and the new learner lease. Neither this detached dataset nor a
root-local observation can reissue an old native/physical capability.
"""
from __future__ import annotations

import copy
import hashlib

from .generation_data import digest, fields, identifier, sha
from .rovent_utility_cohort_files import (HistoricalUtilityCohortFiles,
    PUBLICATION_FILE, ACK_FILE, COST_SCOPE, read_utility_cohort_files)
from .rovent_utility_input import SubmittedUtilityInputObservation, validate_submitted_utility_input
from .rovent_native_v2 import _artifact
from .rovent_utility_training import (COHORT_PREPARATION_DOMAIN,
    normalize_known_pairs, validate_training_preparation)


DATASET_DOMAIN = "rz-rovent-utility-training-dataset/1"
CONTEXT_DOMAIN = "rz-rovent-utility-cohort-context-source/1"
POOL_DOMAIN = "rz-rovent-utility-training-context-pool/1"
OBSERVATION_DOMAIN = "rz-rovent-historical-utility-dataset-admission/1"
_OWNER_DOMAIN = "rz-rovent-historical-utility-python-owner/1"
_REPORT_FIELDS = ("cohort_id", "profile_sha256", "usage", "collection_source_sha256",
    "cost_epoch_sha256", "resource_profile_sha256", "frozen_base_weights_sha256",
    "input_reference_json", "input_sha256", "target_sha256", "submission_target_sha256",
    "utility_context_sha256", "known_pairs", "known_pair_sha256", "all_pair_sha256",
    "frozen_pc_sources", "split", "opening_id", "game_id", "line_genealogy_id",
    "transposition_sha256", "rules_state_sha256", "ancestor_dataset_publication_sha256", "ancestor_datasets")
_LINEAGE_FIELDS = ("opening_id", "game_id", "line_genealogy_id", "transposition_sha256")
_MAX_ANCESTORS = 256
_MAX_LINEAGE_ROWS = 262144


def _pins(value, maximum):
    if type(value) is not list or len(value) > maximum:
        raise ValueError("bounded original utility seal list required")
    for pin in value:
        sha(pin)
    if len(set(value)) != len(value):
        raise ValueError("distinct original utility seals required")
    return value


def _raw_pin(files, name):
    return {key: files.file_pin(name)[key] for key in ("sha256", "bytes")}


def _capture_projection(capture):
    from .nnuec_reference import NnuecConfig
    from .nnuec_utility_reference import ReferenceInput
    if (type(capture) is not SubmittedUtilityInputObservation or type(capture.input_bytes) is not bytes
            or type(capture.reference_input) is not ReferenceInput or type(capture.config) is not NnuecConfig):
        raise ValueError("complete immutable original utility capture required")
    return {"input_json": capture.input_json, "input_bytes": len(capture.input_bytes),
        "input_bytes_sha256": hashlib.sha256(capture.input_bytes).hexdigest(),
        "reference_input": capture.reference_input.to_mapping(), "config": capture.config.to_mapping(),
        "base_weights_sha256": capture.base_weights_sha256, "context_sha256": capture.context_sha256,
        "target_ids": list(capture.target_ids), "view_seal_sha256": capture.view_seal_sha256,
        "input_sha256": capture.input_sha256, "target_sha256": capture.target_sha256,
        "submission_target_sha256": capture.submission_target_sha256}


def _report(value, captured, preparation, publication):
    fields(value, _REPORT_FIELDS, "original utility cohort historical report")
    if type(captured) is not SubmittedUtilityInputObservation:
        raise ValueError("original checked utility capture required")
    checked_capture = validate_submitted_utility_input(**{key: getattr(captured, key) for key in
        ("input_json", "base_weights_sha256", "context_sha256", "target_ids", "view_seal_sha256",
         "input_sha256", "target_sha256", "submission_target_sha256", "config")})
    if captured != checked_capture:
        raise ValueError("complete original utility capture differs from checked input bytes")
    for key in ("cohort_id", "profile_sha256", "collection_source_sha256", "cost_epoch_sha256",
            "resource_profile_sha256", "frozen_base_weights_sha256", "input_sha256", "target_sha256",
            "submission_target_sha256", "utility_context_sha256", "transposition_sha256"):
        sha(value[key])
    sha(value["rules_state_sha256"])
    for key in ("opening_id", "game_id", "line_genealogy_id"):
        identifier(value[key])
    registration = preparation["registration"]
    if (value["cohort_id"] != publication["cohort_id"] or value["usage"] != "collection_candidate"
            or value["split"] not in ("train", "validation", "holdout")
            or any(value[key] != registration[key] for key in
                ("collection_source_sha256", "cost_epoch_sha256", "resource_profile_sha256"))
            or value["frozen_base_weights_sha256"] != registration["frozen_base"]["weights_sha256"]
            or value["input_reference_json"] != captured.input_json
            or value["input_sha256"] != captured.input_sha256 or value["target_sha256"] != captured.target_sha256
            or value["submission_target_sha256"] != captured.submission_target_sha256
            or value["utility_context_sha256"] != captured.context_sha256
            or value["frozen_base_weights_sha256"] != captured.base_weights_sha256
            or captured.config.to_mapping() != registration["frozen_base"]["config"]):
        raise ValueError("original cohort source/capture/split/frozen base differs from preparation")
    if type(value["known_pairs"]) is not list or len(value["known_pairs"]) > 3:
        raise ValueError("bounded original cohort known comparison list required")
    normalized = normalize_known_pairs(value["known_pairs"])
    if (normalized["masked_entries"] or normalized["unique_known_pairs"] != len(value["known_pairs"])
            or normalized["known_occurrences"] != len(value["known_pairs"])):
        raise ValueError("original cohort pair inventory must not repeat or mask known pair entries")
    all_pairs, known = _pins(value["all_pair_sha256"], 3), _pins(value["known_pair_sha256"], 3)
    if (len(known) != len(value["known_pairs"]) or len(all_pairs) != publication["pair_count"]
            or not set(known) <= set(all_pairs)):
        raise ValueError("original cohort known pair source coverage differs")
    ancestors = _pins(value["ancestor_dataset_publication_sha256"], _MAX_ANCESTORS)
    if ancestors != sorted(ancestors):
        raise ValueError("sorted original cohort ancestor publication inventory required")
    descriptors = value["ancestor_datasets"]
    if type(descriptors) is not list or len(descriptors) > _MAX_ANCESTORS:
        raise ValueError("bounded original mixed ancestor descriptors required")
    declared = set()
    for descriptor in descriptors:
        fields(descriptor, ("kind", "publication_sha256", "metadata"), "original mixed ancestor descriptor")
        kind = descriptor["kind"]
        if kind in ("pc_legacy", "pc_submitted_input_v1", "r_auxiliary"):
            names = ("manifest_file_sha256", "index_file_sha256", "canonical_manifest_sha256", "canonical_index_sha256")
        elif kind == "utility_cohort":
            names = ("dataset_sha256", "pool_sha256", "preparation_sha256", "context_sha256")
        else:
            raise ValueError("explicit original mixed ancestor kind required")
        fields(descriptor["metadata"], names, "original ancestor kind-specific four seals")
        for identity in descriptor["metadata"].values():
            sha(identity)
        pin = sha(descriptor["publication_sha256"])
        if pin in declared:
            raise ValueError("repeated original ancestor descriptor publication")
        declared.add(pin)
    if declared != set(ancestors):
        raise ValueError("original mixed ancestor descriptors differ from full publication coverage")
    sources = value["frozen_pc_sources"]
    if type(sources) is not list or len(sources) != 2:
        raise ValueError("two original frozen P/C source descriptors required")
    roles = set()
    for source in sources:
        fields(source, ("role", "artifact", "model_semantics_sha256", "model_identity_sha256", "encoding_sha256"),
            "original utility frozen P/C source")
        if source["role"] not in ("proposer", "critic") or source["role"] in roles:
            raise ValueError("distinct original proposer and critic source roles required")
        roles.add(source["role"])
        # Collection preserves ArtifactRef provenance as well as raw file pins.
        # The execution registration separately supplies an exact FilePin.
        _artifact(source["artifact"])
        for key in ("model_semantics_sha256", "model_identity_sha256", "encoding_sha256"):
            sha(source[key])
    return copy.deepcopy(value)


def _projection(report, files):
    body = {"schema": CONTEXT_DOMAIN,
        "publication_sha256": files.file_pin(PUBLICATION_FILE)["sha256"],
        "durable_ack_sha256": files.file_pin(ACK_FILE)["sha256"],
        **{key: copy.deepcopy(report[key]) for key in ("cohort_id", "profile_sha256", "input_sha256",
            "target_sha256", "submission_target_sha256", "utility_context_sha256", "collection_source_sha256",
            "cost_epoch_sha256", "resource_profile_sha256", "frozen_base_weights_sha256", "split",
            *_LINEAGE_FIELDS, "rules_state_sha256", "ancestor_dataset_publication_sha256", "all_pair_sha256", "known_pair_sha256", "known_pairs")},
        "cost_scope": COST_SCOPE, "execution_authorized": False}
    body["context_sha256"] = digest(CONTEXT_DOMAIN, body)
    return body


def _lineage_keys(report):
    # Input formats are never retagged. Equality is only a conservative leak
    # edge, not equivalence of tensor meaning or evidence for another target.
    result = {(key, report[key]) for key in _LINEAGE_FIELDS}
    result.add(("rules_state_sha256", report["rules_state_sha256"]))
    result.update(("input_seal", report[key]) for key in
        ("input_sha256", "target_sha256", "submission_target_sha256", "utility_context_sha256"))
    return result


def _ancestor_descriptor(owner):
    from .rovent_historical_corpus import HistoricalRoventCorpus
    from .rovent_historical_r_corpus import HistoricalRoventRCorpus
    from .rovent_corpus_codec import LEGACY_CODEC, SUBMITTED_CODEC
    if type(owner) is HistoricalUtilityRouteDataset:
        owner._check()
        metadata = {"dataset_sha256": owner._dataset_sha256, "pool_sha256": owner._pool_sha256,
            "preparation_sha256": owner._preparation["preparation_sha256"], "context_sha256": owner._context_ids[0]}
        kind, publication = "utility_cohort", owner._publication_pin["sha256"]
    elif type(owner) in (HistoricalRoventCorpus, HistoricalRoventRCorpus):
        corpus, observed = owner.corpus, owner.observation()
        metadata = {"manifest_file_sha256": corpus.corpus_sha256, "index_file_sha256": corpus.index_sha256,
            "canonical_manifest_sha256": corpus.manifest_semantic_sha256, "canonical_index_sha256": corpus.index_semantic_sha256}
        publication = observed["publication_sha256"]
        if type(owner) is HistoricalRoventRCorpus:
            kind = "r_auxiliary"
        elif owner._codec is LEGACY_CODEC:
            kind = "pc_legacy"
        elif owner._codec is SUBMITTED_CODEC:
            kind = "pc_submitted_input_v1"
        else:
            raise ValueError("original P/C ancestor lost its exact source family")
    else:
        raise ValueError("original typed PC/R/cohort ancestor owner required")
    for identity in metadata.values():
        sha(identity)
    return {"kind": kind, "publication_sha256": sha(publication), "metadata": metadata}


def _ancestor_rows(owner):
    from .rovent_historical_corpus import HistoricalRoventCorpus
    from .rovent_historical_r_corpus import HistoricalRoventRCorpus
    if type(owner) is HistoricalUtilityRouteDataset:
        owner._check()
        yield owner._report["split"], _lineage_keys(owner._report)
        return
    if type(owner) not in (HistoricalRoventCorpus, HistoricalRoventRCorpus):
        raise ValueError("original typed PC/R/cohort ancestor owner required")
    corpus = owner.corpus  # This invokes its original full row integrity check.
    for row in corpus._records:
        if type(owner) is HistoricalRoventCorpus:
            lineage = row["input"]
            seals = (row["input_sha256"], row["input"]["view_sha256"])
            state = row["input"]["rules_anchor"]["rules_state_sha256"]
        else:
            lineage = row["lineage"]
            seals = (row["input"]["semantic_input_sha256"], row["input"]["native_input_sha256"])
            state = row["bound"]["current_rules"]["anchor"]["rules_state_sha256"]
        keys = {(key, lineage[key]) for key in _LINEAGE_FIELDS}
        keys.add(("rules_state_sha256", state))
        keys.update(("input_seal", seal) for seal in seals)
        yield row["split"], keys


def _read_cohort(reference, preparation, read_budget):
    checked = validate_training_preparation(preparation)
    if checked["schema"] != COHORT_PREPARATION_DOMAIN:
        raise ValueError("actual cohort data requires explicit utility preparation/2")
    files = read_utility_cohort_files(reference, read_accounting=read_budget)
    from .rovent_utility_cohort_payload import validate_cohort_payloads
    original = fields(validate_cohort_payloads(files, preparation=checked), ("input", "report"),
        "original historical cohort body verification")
    report = _report(original["report"], original["input"], checked, files.publication())
    return checked, files, original["input"], report


def read_historical_utility_dependencies(reference, *, preparation, read_budget):
    """Read the original dependency list through the strict payload verifier.

    This returns historical metadata only. It neither admits the dependencies
    nor supplies a known-target or execution capability. A subsequent dataset
    read is charged again to the same budget; no cached declaration receives
    unaccounted read credit or replaces a separately opened ancestor owner.
    """
    _, _, _, report = _read_cohort(reference, preparation, read_budget)
    return {"ancestor_dataset_publication_sha256": copy.deepcopy(report["ancestor_dataset_publication_sha256"])}


class HistoricalUtilityRouteDataset:
    """One original cohort/context, data custody rather than a trainer lease."""
    def __init__(self):
        raise TypeError("use the exact pinned cohort publication and durable ACK")

    @classmethod
    def from_registration(cls, reference, *, preparation, read_budget, ancestors=()):
        checked, files, capture, report = _read_cohort(reference, preparation, read_budget)
        projection = _projection(report, files)
        context_ids = (projection["context_sha256"],)
        pool = digest(POOL_DOMAIN, list(context_ids))
        publication, ack = _raw_pin(files, PUBLICATION_FILE), _raw_pin(files, ACK_FILE)
        dataset = digest(DATASET_DOMAIN, {"publication": publication, "durable_ack": ack,
            "contexts": list(context_ids), "pool_sha256": pool,
            "source_registration_sha256": checked["registration_sha256"]})
        owner = cls.__new__(cls)
        owner._files, owner._preparation, owner._report = files, checked, report
        owner._capture, owner._projection = capture, projection
        owner._context_ids, owner._pool_sha256, owner._dataset_sha256 = context_ids, pool, dataset
        owner._publication_pin, owner._durable_ack_pin = publication, ack
        owner._publication_metadata, owner._durable_ack_metadata = files.publication(), files.durable_ack()
        owner._state_sha256 = owner._state_digest()
        owner.validate_ancestors(ancestors)
        return owner

    def _state_digest(self):
        return digest(_OWNER_DOMAIN, [self._preparation, self._report, self._projection,
            list(self._context_ids), self._pool_sha256, self._dataset_sha256,
            self._publication_pin, self._durable_ack_pin, self._publication_metadata,
            self._durable_ack_metadata, _capture_projection(self._capture)])

    def _check(self):
        if type(self._files) is not HistoricalUtilityCohortFiles or self._state_digest() != self._state_sha256:
            raise ValueError("original utility cohort dataset metadata changed")
        if (_raw_pin(self._files, PUBLICATION_FILE) != self._publication_pin
                or _raw_pin(self._files, ACK_FILE) != self._durable_ack_pin
                or self._files.publication() != self._publication_metadata
                or self._files.durable_ack() != self._durable_ack_metadata):
            raise ValueError("original utility cohort file owner differs from retained raw pins")
        self._files.raw_file(PUBLICATION_FILE)
        self._files.raw_file(ACK_FILE)
        if (self._capture.input_json != self._report["input_reference_json"]
                or self._capture.input_sha256 != self._report["input_sha256"]):
            raise ValueError("original utility cohort input changed")

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
        return copy.deepcopy(self._report["ancestor_datasets"])

    def sample(self, index):
        self._check()
        if type(index) is not int or index != 0 or self._report["split"] != "train":
            raise ValueError("only the original train cohort context is a learner sample")
        return {"input": self._capture, "known_pairs": copy.deepcopy(self._report["known_pairs"]),
            "source_projection": copy.deepcopy(self._projection)}

    def historical_context(self, context_sha256):
        self._check()
        if sha(context_sha256) != self._context_ids[0]:
            raise ValueError("utility context is outside the original cohort publication")
        return {"context_sha256": context_sha256, "source_projection": copy.deepcopy(self._projection),
            "pair_sha256": copy.deepcopy(self._report["known_pair_sha256"]),
            "unique_known_pairs": len(self._report["known_pairs"])}

    def validate_ancestors(self, ancestors):
        self._check()
        if type(ancestors) not in (list, tuple) or len(ancestors) > _MAX_ANCESTORS:
            raise ValueError("bounded explicit mixed utility ancestor owners required")
        actual, identities, inspected = set(), [], 0
        declarations = {item["publication_sha256"]: item for item in self._report["ancestor_datasets"]}
        for owner in ancestors:
            descriptor = _ancestor_descriptor(owner)
            pin = descriptor["publication_sha256"]
            if pin == self._publication_pin["sha256"] or pin in actual:
                raise ValueError("self or repeated original utility ancestor publication")
            if descriptor != declarations.get(pin):
                raise ValueError("original ancestor kind/four seals differ from the retained historical owner")
            actual.add(pin)
            for split, keys in _ancestor_rows(owner):
                if split not in ("train", "validation", "holdout"):
                    raise ValueError("original ancestor split was not registered")
                inspected += 1
                if inspected > _MAX_LINEAGE_ROWS:
                    raise ValueError("bounded mixed utility ancestry row limit exceeded")
                identities.append((split, keys))
        if actual != set(self._report["ancestor_dataset_publication_sha256"]):
            raise ValueError("original mixed ancestor publication coverage differs")
        # Union all keys first: a validation bridge may connect a utility row
        # indirectly to holdout even when those two share no direct key.
        rows = [(self._report["split"], _lineage_keys(self._report)), *identities]
        parents, keyed = list(range(len(rows))), {}
        def root(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index
        for index, (_, keys) in enumerate(rows):
            for key in keys:
                if key in keyed:
                    parents[root(index)] = root(keyed[key])
                else:
                    keyed[key] = index
        components = {}
        for index, (split, _) in enumerate(rows):
            components.setdefault(root(index), set()).add(split)
        if len(components[root(0)]) > 1:
            raise ValueError("known PC/R/cohort leakage component crosses global splits")
        return {"ancestor_publication_sha256": sorted(actual), "inspected_occurrences": inspected,
            "ancestor_cross_split_components": sum(len(splits) > 1 for splits in components.values()),
            "execution_authorized": False, "past_native_authority_reissued": False}

    def validate_execution_sources(self, registration):
        self._check()
        current = registration["current_dataset"]
        if any(current[name][key] != pin[key] for name, pin in
                (("publication", self._publication_pin), ("durable_ack", self._durable_ack_pin))
                for key in ("sha256", "bytes")):
            raise ValueError("utility execution selected different original cohort metadata")
        base = self._files.publication()["nnuec_weights"]
        if any(registration["frozen_nnuec_base"][key] != base[key] for key in ("sha256", "bytes")):
            raise ValueError("utility execution frozen NNUEC raw file differs from original cohort")
        sources = registration["frozen_pc_sources"]
        if type(sources) is not list or len(sources) != len(self._report["frozen_pc_sources"]):
            raise ValueError("utility execution frozen P/C inventory differs")
        roles = set()
        for source in sources:
            if source["role"] in roles:
                raise ValueError("utility execution repeats an original P/C source role")
            roles.add(source["role"])
            original = next((item for item in self._report["frozen_pc_sources"] if item["role"] == source["role"]), None)
            if original is None or any(source[key] != original[key] for key in
                    ("model_semantics_sha256", "model_identity_sha256", "encoding_sha256")) or any(
                    source["artifact"][key] != original["artifact"][key] for key in ("sha256", "bytes")):
                raise ValueError("utility execution changed original P/C raw/model/encoding source")

    def observation(self):
        self._check()
        report, publication = self._report, self._files.publication()
        return {"schema": OBSERVATION_DOMAIN, "dataset_sha256": self._dataset_sha256,
            "publication_sha256": self._publication_pin["sha256"], "publication_pin": self.publication_pin,
            "durable_ack_pin": self.durable_ack_pin, "pool_sha256": self._pool_sha256,
            "preparation_sha256": self._preparation["preparation_sha256"], "context_ids": list(self._context_ids),
            "record_count": 1, "trial_count": publication["trial_count"], "pair_count": publication["pair_count"],
            **{key: report[key] for key in ("split", *_LINEAGE_FIELDS, "rules_state_sha256")},
            "unique_known_pairs": len(report["known_pairs"]), "source_kind": "historical_exact_cohort_pins",
            "past_native_authority_reissued": False, "past_physical_completion_reissued": False,
            "execution_authorized": False, "training_target_authorized": False,
            "pair_arithmetic_scope": "original_candidate_values_rust_conditional_certificate_required",
            "rules_replay_scope": "original_rust_parent_required",
            "frozen_pc_scope": "not_instantiated_pinned_original_source_preserved",
            "collection_io": self._files.durable_ack()["collection_io"],
            "retained_bytes": publication["total_payload_bytes"] + self._publication_pin["bytes"] + self._durable_ack_pin["bytes"],
            "retained_bytes_scope": "original_file_bytes_not_process_peak_memory"}
