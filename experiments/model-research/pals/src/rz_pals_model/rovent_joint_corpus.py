"""Current P/C and R sampler views over one conservative leakage union.

Canonical datasets and original tokens remain unchanged. This owner narrows
their sampling references after joining both source domains, including P/C
roots that only the R corpus declared. It issues no native/trainer authority.
"""
from __future__ import annotations

import copy
from collections.abc import Sequence

from .generation_data import digest, sha
from . import rovent_corpus as pc
from . import rovent_r_corpus as r
from .rovent_generation import index_original_input
from .rovent_lineage_union import LineageUnion, PC_SOURCES, R_SOURCE, pc_source_domain

JOINT_DOMAIN = "rz-rovent-joint-pc-r-intake/1"
_SPLITS = ("train", "validation", "holdout")


class _TokenView(Sequence):
    """Retain original opaque tokens; do not relabel a R row as P/C."""
    def __init__(self, tokens):
        self._tokens = tuple(tokens)

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _TokenView(self._tokens[index])
        return self._tokens[index]


def _seals(values, description):
    if not isinstance(values, (tuple, list)) or len(values) > 262144:
        raise ValueError("bounded joint retained " + description + " seals required")
    result = [sha(item) for item in values]
    if result != sorted(set(result)):
        raise ValueError("sorted unique joint retained " + description + " seals required")
    return result


def _combine(pc_intake, r_intake):
    """Merge equal source occurrences once; reject conflicting projections."""
    recipe = r_intake._recipe
    existing = {}
    for occurrence in r_intake._union.all_occurrences():
        value = occurrence.to_mapping()
        existing[(value["source_domain"], value["origin_sha256"])] = value
    pending = []
    owners = (*pc_intake._ancestors, pc_intake)
    if len(owners) > 257 or len({owner.index_semantic_sha256 for owner in owners}) != len(owners):
        raise ValueError("joint original P/C ancestor closure has duplicate/too many indices")
    manifests = [r_intake._manifest["collection"]]
    manifests.extend(owner._manifest["collection"] for owner in r_intake._r_ancestors)
    # R-only referenced P/C roots may be outside the current P/C closure.
    # Preserve their already-learned base exposure in this larger union too.
    for r_owner in (*r_intake._r_ancestors, r_intake):
        manifests.extend(owner._manifest for owner in r_owner._pc_corpora)
    for owner in owners:
        if type(owner) is not pc.RoventTrainingCorpus:
            raise ValueError("joint P/C closure requires original corpus owners")
        owner._check_owner(full=True)
        manifest = owner._manifest
        source_domain = pc_source_domain(owner)
        if manifest["recipe_sha256"] != recipe.sha256:
            raise ValueError("joint P/C and R owners belong to different frozen recipes")
        manifests.append(manifest)
        for row in owner._records:
            if row["schema"] != source_domain or pc._record_digest(row) != row["record_sha256"]:
                raise ValueError("joint original P/C record changed")
            projection = index_original_input(recipe, input_snapshot=row["input"], input_sha256=row["input_sha256"],
                record_sha256=row["record_sha256"], producer_sha256=owner.manifest_semantic_sha256,
                stage_id=manifest["stage_id"], round_id=manifest["round_id"], teacher_actor_id=row["lineage"]["teacher_actor_id"])
            pending.append((source_domain, row["origin_sha256"], projection, row["lineage"]["ancestor_origin_sha256"]))
    available = set(existing) | {(domain, origin) for domain, origin, _projection, _ancestors in pending}
    origins = {}
    for source in available:
        origins.setdefault(source[1], []).append(source)
    for source_domain, origin, projection, ancestors in pending:
        references = []
        for ancestor in ancestors:
            candidates = origins.get(ancestor, ())
            if len(candidates) != 1:
                raise ValueError("joint original ancestor origin is missing or source-ambiguous")
            domain, identity = candidates[0]
            references.append({"source_domain": domain, "origin_sha256": identity})
        value = {"source_domain": source_domain, "origin_sha256": origin, "projection": projection,
            "ancestor_origins": sorted(references, key=lambda item: (item["source_domain"], item["origin_sha256"]))}
        key = (source_domain, origin)
        if key in existing and existing[key] != value:
            raise ValueError("joint same original source origin has conflicting projection or ancestry")
        existing[key] = value
    ordered = [existing[key] for key in sorted(existing)]
    union = LineageUnion.from_projections(recipe, ordered)
    # Frozen inherited declarations may already refer to learned samples. A
    # newly discovered P/C↔R holdout leak cannot be fixed by dropping only a
    # descendant row or silently forgetting its inherited training exposure.
    for manifest in manifests:
        references, inputs = [], set()
        prior = {stage.stage_id for stage in recipe.lineage(manifest["stage_id"])[:-1]}
        for origin in manifest["inherited_origin_sha256"]:
            candidates = origins.get(origin, ())
            if len(candidates) != 1:
                raise ValueError("joint inherited original source is missing or ambiguous")
            source = candidates[0]
            projected = existing[source]["projection"]
            if projected["stage_id"] not in prior:
                raise ValueError("joint inherited ancestor base is same-stage, future or sibling")
            references.append({"source_domain": source[0], "origin_sha256": source[1]})
            inputs.add(projected["input_sha256"])
        if inputs != set(manifest["inherited_input_sha256"]):
            raise ValueError("joint inherited original origin/input projection differs")
        union.require_training_origins(manifest["stage_id"], sorted(references, key=lambda item: (item["source_domain"], item["origin_sha256"])))
    return union, existing


def _retained(union, by_source, domains, stage_id, inputs, origins):
    input_values = _seals(inputs, "input")
    origin_values = _seals(origins, "origin")
    references, actual_inputs = [], set()
    for origin in origin_values:
        matches = [source for source in by_source if source[0] in domains and source[1] == origin]
        if len(matches) != 1:
            raise ValueError("joint retained training origin is absent from its original source domain")
        source = matches[0]
        references.append({"source_domain": source[0], "origin_sha256": origin})
        actual_inputs.add(by_source[source]["projection"]["input_sha256"])
    if actual_inputs != set(input_values):
        raise ValueError("joint retained training origin/input projection differs")
    union.require_training_origins(stage_id, sorted(references, key=lambda item: (item["source_domain"], item["origin_sha256"])))
    return {"input_sha256": input_values, "origin_sha256": origin_values}


def build_pc_r_companion(pc_intake, r_intake, *, training_input_sha256, training_origin_sha256,
                         retained_r_input_sha256=(), retained_r_origin_sha256=()):
    if type(pc_intake) is not pc.RoventTrainingCorpus or type(r_intake) is not r.RoventRCorpus:
        raise ValueError("exact original P/C and R corpus owner types required")
    pc_intake._check_owner(full=True)
    r_intake._check_owner(full=True)
    admitted = pc_intake._admission
    if admitted is None:
        raise ValueError("original P/C stage admission must precede the joint companion")
    collection = r_intake._manifest["collection"]
    if (admitted["recipe_sha256"] != r_intake._recipe.sha256 or admitted["stage_id"] != collection["stage_id"]
            or pc_intake._manifest["context_epoch"] != collection["context_epoch"]):
        raise ValueError("joint P/C and R current stage/context declaration differ")
    # This is a leakage union, not a claim that the two model epochs are one
    # inference namespace. Frozen P/C may advance before a V phase which must
    # retain old R provenance/pool while performing zero R consumption. The
    # actual training window separately binds each original epoch exactly.
    union, by_source = _combine(pc_intake, r_intake)
    pc_retained = _retained(union, by_source, PC_SOURCES, collection["stage_id"], training_input_sha256, training_origin_sha256)
    r_retained = _retained(union, by_source, (R_SOURCE,), collection["stage_id"], retained_r_input_sha256, retained_r_origin_sha256)
    all_origins = set(pc_retained["origin_sha256"]) | set(r_retained["origin_sha256"])
    if not set(pc_intake._manifest["inherited_origin_sha256"]) <= all_origins:
        raise ValueError("joint retained training seals erased an original P/C inherited base")
    owner = JointPcRIntake.__new__(JointPcRIntake)
    owner._pc, owner._r, owner._union = pc_intake, r_intake, union
    owner._pc_source_domain = pc_source_domain(pc_intake)
    owner._allowed = tuple((split, domain, frozenset(entry.record_sha256 for entry in union.rows(split,
        optimization=split == "train", source_domain=domain))) for split in _SPLITS for domain in (*PC_SOURCES, R_SOURCE))
    owner._retained = {"pc": pc_retained, "r": r_retained}
    owner._pins = (pc_intake.corpus_sha256, pc_intake.index_sha256, r_intake.corpus_sha256, r_intake.index_sha256, union.sha256)
    owner._report = {"schema": JOINT_DOMAIN, "union_sha256": union.sha256,
        "pc_manifest_file_sha256": pc_intake.corpus_sha256, "pc_index_file_sha256": pc_intake.index_sha256,
        "r_manifest_file_sha256": r_intake.corpus_sha256, "r_index_file_sha256": r_intake.index_sha256,
        "allowed_pc_record_sha256": sorted(owner._allowed_pc_records("train")),
        "allowed_r_record_sha256": sorted(owner._allowed_records("train", R_SOURCE)),
        "retained_training": copy.deepcopy(owner._retained), "lineage": union.observation().to_mapping(),
        "execution_authorized": False, "original_checkpoint_authority_verified": False}
    owner._metadata_sha256 = digest(JOINT_DOMAIN, owner._report)
    return owner


class JointPcRIntake:
    """Local sampling dispatcher, never an original corpus or live capability."""
    def __init__(self):
        raise TypeError("use original owner-bound build_pc_r_companion")

    def _allowed_records(self, split, domain):
        return next(records for part, source, records in self._allowed if part == split and source == domain)

    def _allowed_pc_records(self, split):
        return frozenset(record for domain in PC_SOURCES for record in self._allowed_records(split, domain))

    def _check(self, *, full=False):
        self._pc._check_owner(full=full)
        self._r._check_owner(full=full)
        if pc_source_domain(self._pc) != self._pc_source_domain:
            raise ValueError("joint original P/C source family changed")
        if self._pins != (self._pc.corpus_sha256, self._pc.index_sha256, self._r.corpus_sha256, self._r.index_sha256, self._union.sha256):
            raise ValueError("joint original corpus/union pins changed")
        if full and digest(JOINT_DOMAIN, self._report) != self._metadata_sha256:
            raise ValueError("joint immutable sampling projection changed")
        if full and (self._report["allowed_pc_record_sha256"] != sorted(self._allowed_pc_records("train"))
                     or self._report["allowed_r_record_sha256"] != sorted(self._allowed_records("train", R_SOURCE))):
            raise ValueError("joint eligible source record sets changed")

    def pc_rows(self, role, split="train", optimization=True):
        self._check()
        tokens = self._pc.rows(role, split=split, optimization=optimization)
        allowed = self._allowed_records(split, self._pc_source_domain)
        return _TokenView(token for token in tokens if self._pc._records[token._index]["record_sha256"] in allowed)

    def r_rows(self, split="train", optimization=True):
        self._check()
        tokens = self._r.rows(split=split, optimization=optimization)
        allowed = self._allowed_records(split, R_SOURCE)
        return _TokenView(token for token in tokens if self._r._records[token._index]["record_sha256"] in allowed)

    def r_training_sample(self, token):
        self._check()
        sample = self._r.training_sample(token)
        if sample["record_sha256"] not in self._allowed_records(token._split, R_SOURCE):
            raise ValueError("R row is excluded by the joint current P/C leakage union")
        return sample

    def original_auxiliary_evidence(self, record_sha256):
        self._check()
        identity = sha(record_sha256)
        if identity not in self._allowed_records("train", R_SOURCE):
            raise ValueError("R auxiliary evidence is not joint training eligible")
        return self._r.original_auxiliary_evidence(identity)

    def pool_metadata(self, split="train", optimization=True):
        self._check(full=True)
        occurrences = []
        for token in self.r_rows(split=split, optimization=optimization):
            row = self._r._records[token._index]
            occurrences.append({"source_domain": R_SOURCE, "record_sha256": row["record_sha256"],
                "input_sha256": row["input"]["semantic_input_sha256"], "origin_sha256": row["origin_sha256"]})
        value = {"index_sha256": self._r.index_semantic_sha256, "occurrences": occurrences}
        return {"schema": r.POOL_DOMAIN, **value, "pool_sha256": digest(r.POOL_DOMAIN, value), "execution_authorized": False}

    def to_mapping(self):
        self._check(full=True)
        return copy.deepcopy(self._report)

    def observation(self):
        return self.to_mapping()
