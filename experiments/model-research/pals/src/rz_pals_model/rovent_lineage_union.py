"""Immutable P/C and R occurrence metadata union, without source authority.

Source-specific readers validate original records before projecting them here.
This module never invents a P/C input for R, reads a corpus, verifies a target,
or issues native/trainer authority. Repeated numerical inputs remain distinct
occurrences; unique input counts are reported separately from occurrences.
The existing global-lineage /1 canonical bytes and duplicate rules are intact.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .generation_data import digest, fields, sha, uint
from .rovent_generation import SeriesRecipe, validate_record_origin
from .rovent_corpus_codec import LEGACY_CODEC, SUBMITTED_CODEC, codec_from_manifest

UNION_DOMAIN = "rz-rovent-pc-r-lineage-union/1"
OBSERVATION_DOMAIN = "rz-rovent-pc-r-lineage-observation/1"
INHERITED_DOMAIN = "rz-rovent-pc-r-inherited-origin-observation/1"
PC_SOURCE = "rz-rovent-training-record/1"
PC_SOURCE_V2 = "rz-rovent-training-record/2"
PC_SOURCES = (PC_SOURCE, PC_SOURCE_V2)
R_SOURCE = "rz-rovent-r-readout-record/1"
_LEGACY_SOURCES = (PC_SOURCE, R_SOURCE)
_SOURCES = (*_LEGACY_SOURCES, PC_SOURCE_V2)
_PROJECTION_FIELDS = ("input_sha256", "record_sha256", "producer_sha256", "stage_id", "round_id",
                      "teacher_actor_id", "opening_id", "game_id", "line_genealogy_id",
                      "rules_state_sha256", "transposition_sha256")
_MAX_OCCURRENCES = 262144
_MAX_METADATA_BYTES = 256 * 1024 * 1024
_MAX_EDGES = 1048576


def pc_source_domain(owner):
    """Retain an already-open PC owner's explicit wire family.

    This is a metadata join, not a substitute for the original opaque owner
    type or its integrity check. Callers must perform both before using rows.
    An old owner may omit the field; a submitted owner must explicitly carry
    the new codec instead of acquiring it from relabelled manifest JSON.
    """
    codec = codec_from_manifest(owner._manifest)
    if codec is not getattr(owner, "_codec", LEGACY_CODEC):
        raise ValueError("original P/C owner codec differs from its manifest family")
    if codec is LEGACY_CODEC:
        return PC_SOURCE
    if codec is SUBMITTED_CODEC:
        return PC_SOURCE_V2
    raise ValueError("unsupported original P/C source family")


@dataclass(frozen=True, order=True)
class SourceOrigin:
    source_domain: str
    origin_sha256: str

    @classmethod
    def from_mapping(cls, value):
        fields(value, ("source_domain", "origin_sha256"), "PC/R source origin reference")
        if type(value["source_domain"]) is not str or value["source_domain"] not in _SOURCES:
            raise ValueError("explicit supported PC/R source domain required")
        return cls(value["source_domain"], sha(value["origin_sha256"]))

    def to_mapping(self):
        return {"source_domain": self.source_domain, "origin_sha256": self.origin_sha256}


@dataclass(frozen=True)
class LineageOccurrence:
    """Data provenance only; no role, native ticket, target or capability."""
    source: SourceOrigin
    _projection_values: tuple
    ancestor_origins: tuple

    def projection(self):
        return dict(zip(_PROJECTION_FIELDS, self._projection_values, strict=True))

    @property
    def input_sha256(self):
        return self._projection_values[0]

    @property
    def record_sha256(self):
        return self._projection_values[1]

    def to_mapping(self):
        return {**self.source.to_mapping(), "projection": self.projection(),
                "ancestor_origins": [ancestor.to_mapping() for ancestor in self.ancestor_origins]}


@dataclass(frozen=True)
class SourceCount:
    source_domain: str
    occurrence_count: int
    unique_input_count: int


@dataclass(frozen=True)
class LineageUnionObservation:
    schema: str
    recipe_sha256: str
    index_sha256: str
    occurrence_count: int
    unique_input_count: int
    cross_split_component_count: int
    excluded_occurrence_count: int
    sources: tuple
    execution_authorized: bool = False
    original_source_authority_verified: bool = False
    target_validity_verified: bool = False

    def to_mapping(self):
        return {"schema": self.schema, "recipe_sha256": self.recipe_sha256, "index_sha256": self.index_sha256,
                "occurrence_count": self.occurrence_count, "unique_input_count": self.unique_input_count,
                "unique_input_identity": "source_domain_and_input_sha256",
                "cross_split_component_count": self.cross_split_component_count,
                "excluded_occurrence_count": self.excluded_occurrence_count,
                "sources": [{"source_domain": item.source_domain, "occurrence_count": item.occurrence_count,
                             "unique_input_count": item.unique_input_count} for item in self.sources],
                "execution_authorized": self.execution_authorized,
                "original_source_authority_verified": self.original_source_authority_verified,
                "target_validity_verified": self.target_validity_verified}


@dataclass(frozen=True)
class InheritedOriginObservation:
    schema: str
    index_sha256: str
    stage_id: str
    origins: tuple
    unique_input_count: int
    same_stage_original_checkpoint_authority_required: bool
    execution_authorized: bool = False
    original_checkpoint_authority_verified: bool = False


def _occurrence(recipe, value):
    fields(value, ("source_domain", "origin_sha256", "projection", "ancestor_origins"),
           "PC/R occurrence metadata")
    source = SourceOrigin.from_mapping({key: value[key] for key in ("source_domain", "origin_sha256")})
    projection = validate_record_origin(recipe, value["projection"])
    if projection["teacher_actor_id"] is not None and type(projection["teacher_actor_id"]) is not str:
        raise ValueError("plain original PC/R teacher actor or null required")
    ancestors = value["ancestor_origins"]
    if type(ancestors) is not list or len(ancestors) > 256:
        raise ValueError("bounded explicit PC/R ancestor origins required")
    refs = tuple(SourceOrigin.from_mapping(item) for item in ancestors)
    if refs != tuple(sorted(set(refs))) or source in refs:
        raise ValueError("sorted unique non-self PC/R ancestor references required")
    return LineageOccurrence(source, tuple(projection[key] for key in _PROJECTION_FIELDS), refs)


def _seal(recipe_sha256, entries, maximum):
    """New domain: canonical JSON [domain,{occurrences,recipe_sha256}].

    Stream one small immutable projection at a time instead of making another
    complete metadata tree. Source order is deliberately part of this seal.
    """
    hasher, used = hashlib.sha256(), 0
    def emit(block):
        nonlocal used
        used += len(block)
        if used > maximum:
            raise ValueError("PC/R lineage metadata byte limit exceeded")
        hasher.update(block)
    emit(("[" + json.dumps(UNION_DOMAIN) + ',{"occurrences":[').encode("utf-8"))
    for index, entry in enumerate(entries):
        if index:
            emit(b",")
        emit(json.dumps(entry.to_mapping(), sort_keys=True, ensure_ascii=False,
                        allow_nan=False, separators=(",", ":")).encode("utf-8"))
    emit(('],"recipe_sha256":' + json.dumps(recipe_sha256) + "}]").encode("utf-8"))
    return hasher.hexdigest()


class LineageUnion:
    def __init__(self):
        raise TypeError("use bounded PC/R occurrence projections")

    @classmethod
    def from_projections(cls, recipe, occurrences, *, max_occurrences=_MAX_OCCURRENCES,
                         max_metadata_bytes=64 * 1024 * 1024):
        if type(recipe) is not SeriesRecipe or type(occurrences) is not list:
            raise ValueError("validated series and explicit occurrence metadata list required")
        if (uint(max_occurrences, _MAX_OCCURRENCES) == 0 or len(occurrences) > max_occurrences
                or uint(max_metadata_bytes, _MAX_METADATA_BYTES) == 0):
            raise ValueError("bounded positive PC/R union limits required")
        entries, by_source, records, edge_count = [], {}, set(), 0
        metadata_bytes = len(("[" + json.dumps(UNION_DOMAIN) + ',{"occurrences":['
            + '],"recipe_sha256":' + json.dumps(recipe.sha256) + "}]").encode("utf-8"))
        for value in occurrences:
            entry = _occurrence(recipe, value)
            key = (entry.source.source_domain, entry.record_sha256)
            if entry.source in by_source or key in records:
                raise ValueError("duplicate PC/R source origin or source record occurrence")
            metadata_bytes += len(json.dumps(entry.to_mapping(), sort_keys=True, ensure_ascii=False,
                allow_nan=False, separators=(",", ":")).encode("utf-8")) + bool(entries)
            edge_count += len(entry.ancestor_origins)
            if metadata_bytes > max_metadata_bytes or edge_count > _MAX_EDGES:
                raise ValueError("PC/R lineage metadata byte/edge limit exceeded")
            by_source[entry.source] = len(entries)
            records.add(key)
            entries.append(entry)
        entries = tuple(entries)
        projection = tuple(entry.projection() for entry in entries)
        scopes = {stage.stage_id: frozenset(item.stage_id for item in recipe.lineage(stage.stage_id))
                  for stage in recipe.stages}
        splits = dict(recipe.to_dict()["opening_splits"])
        parents, adjacency, indegree = list(range(len(entries))), [[] for _ in entries], [0] * len(entries)
        def root(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index
        def join(first, second):
            parents[root(first)] = root(second)
        identities = {}
        for index, entry in enumerate(entries):
            row = projection[index]
            for name in ("input_sha256", "game_id", "opening_id", "line_genealogy_id", "rules_state_sha256",
                         "transposition_sha256"):
                # These are conservative leakage keys, never a claim that P/C
                # and R tensor formats or outputs are interchangeable.
                key = (name, row[name])
                if key in identities:
                    join(index, identities[key])
                else:
                    identities[key] = index
            for ancestor in entry.ancestor_origins:
                if ancestor not in by_source:
                    raise ValueError("PC/R ancestor origin missing from full source union")
                parent_index = by_source[ancestor]
                if projection[parent_index]["stage_id"] not in scopes[row["stage_id"]]:
                    raise ValueError("PC/R ancestor origin belongs to a future or sibling stage")
                adjacency[parent_index].append(index)
                indegree[index] += 1
                join(index, parent_index)
        ready = [index for index, degree in enumerate(indegree) if degree == 0]
        visited = 0
        while ready:
            index = ready.pop()
            visited += 1
            for child in adjacency[index]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if visited != len(entries):
            raise ValueError("PC/R occurrence ancestor cycle rejected")
        components = {}
        for index, row in enumerate(projection):
            components.setdefault(root(index), set()).add(splits[row["opening_id"]])
        contaminated = {index for index, parts in components.items() if len(parts) > 1}
        excluded = frozenset(index for index in range(len(entries)) if root(index) in contaminated)
        owner = cls.__new__(cls)
        owner._entries, owner._by_source, owner._excluded = entries, by_source, excluded
        owner._splits, owner._scopes = tuple(sorted(splits.items())), tuple(sorted(scopes.items()))
        owner._recipe_sha256, owner._maximum_metadata = recipe.sha256, max_metadata_bytes
        owner.sha256 = _seal(recipe.sha256, entries, max_metadata_bytes)
        # Keep the old PC/R observation and integrity projection byte-for-byte
        # unchanged when no submitted-format occurrence exists. The new source
        # appears explicitly only in unions which actually contain that source.
        observed_sources = (*_LEGACY_SOURCES, *((PC_SOURCE_V2,) if any(
            entry.source.source_domain == PC_SOURCE_V2 for entry in entries) else ()))
        source_counts = tuple(SourceCount(domain, sum(entry.source.source_domain == domain for entry in entries),
            len({entry.input_sha256 for entry in entries if entry.source.source_domain == domain})) for domain in observed_sources)
        owner._observation = LineageUnionObservation(OBSERVATION_DOMAIN, recipe.sha256, owner.sha256, len(entries),
            len({(entry.source.source_domain, entry.input_sha256) for entry in entries}), len(contaminated), len(excluded), source_counts)
        owner._integrity_sha256 = owner._integrity()
        return owner

    def _integrity(self):
        return digest("rz-rovent-pc-r-lineage-owner-state/1", {"index_sha256": self.sha256,
            "projection_sha256": _seal(self._recipe_sha256, self._entries, self._maximum_metadata),
            "recipe_sha256": self._recipe_sha256, "excluded": sorted(self._excluded),
            "by_source": [[ref.source_domain, ref.origin_sha256, index]
                          for ref, index in sorted(self._by_source.items())],
            "splits": self._splits, "scopes": [[stage, sorted(scope)] for stage, scope in self._scopes],
            "observation": self._observation.to_mapping()})

    def _check(self):
        if self._integrity() != self._integrity_sha256:
            raise ValueError("immutable PC/R occurrence projection seal changed")

    def all_occurrences(self):
        self._check()
        return self._entries

    def rows(self, split, *, optimization=False, source_domain=None):
        self._check()
        if (type(optimization) is not bool or split not in ("train", "validation", "holdout")
                or optimization and split != "train" or source_domain is not None and source_domain not in _SOURCES):
            raise ValueError("PC/R validation/global holdout cannot enter optimization")
        splits = dict(self._splits)
        return tuple(entry for index, entry in enumerate(self._entries)
                     if index not in self._excluded and splits[entry.projection()["opening_id"]] == split
                     and (source_domain is None or entry.source.source_domain == source_domain))

    def observation(self):
        self._check()
        return self._observation

    def require_training_origins(self, stage_id, references):
        """Check metadata scope only; same-stage checkpoint proof is external."""
        self._check()
        scopes, splits = dict(self._scopes), dict(self._splits)
        if (type(stage_id) is not str or stage_id not in scopes or type(references) is not list
                or len(references) > _MAX_OCCURRENCES):
            raise ValueError("bounded PC/R inherited source references and registered stage required")
        refs = tuple(SourceOrigin.from_mapping(value) for value in references)
        if refs != tuple(sorted(set(refs))):
            raise ValueError("sorted unique PC/R inherited source references required")
        inputs, same_stage = set(), False
        for ref in refs:
            if ref not in self._by_source:
                raise ValueError("PC/R inherited source origin is unknown")
            index = self._by_source[ref]
            row = self._entries[index].projection()
            if index in self._excluded or splits[row["opening_id"]] != "train":
                raise ValueError("PC/R inherited origin has global holdout/cross-split contamination")
            if row["stage_id"] not in scopes[stage_id]:
                raise ValueError("PC/R inherited origin belongs to a future or sibling stage")
            same_stage |= row["stage_id"] == stage_id
            inputs.add((ref.source_domain, row["input_sha256"]))
        return InheritedOriginObservation(INHERITED_DOMAIN, self.sha256, stage_id, refs, len(inputs), same_stage)
