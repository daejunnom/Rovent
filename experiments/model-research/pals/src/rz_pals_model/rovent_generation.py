"""Rovent multi-teacher series preparation; standard library and no execution.

This domain is independent of the closed PALS first-generation recipe. Its
metadata neither authorizes training nor certifies an actual checkpoint, a
teacher completion, a Rules outcome, or a native intake. Original row/head
evidence remains with each producer. In particular, a Reckless row is never
renamed to the legacy Stockfish-only policy source.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum

from .generation_data import (GenerationDataset, digest, fields, identifier, seal_input, sha, uint,
                              validate_input, validate_record)

RECIPE_DOMAIN = "rz-rovent-generation-series/1"
CHECKPOINT_DOMAIN = "rz-rovent-series-checkpoint-metadata/1"
SPLIT_DOMAIN = "rz-rovent-global-opening-split/1"
INDEX_DOMAIN = "rz-rovent-global-lineage-index/1"
ORIGIN_DOMAIN = "rz-rovent-original-row-origin/1"
ROLES = ("proposer", "critic", "validator")
STATE_PARTS = ("optimizer", "scheduler", "rng", "sampler", "cost")
MAX_STAGES = 256
MAX_RECORDS = 262144


class StageKind(str, Enum):
    COMMON = "common_student"
    TEACHER_FT = "teacher_finetune"
    MIXED = "mixed"
    SELF = "self"


def _plain_id(value):
    identifier(value)
    if not value.isascii() or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
        raise ValueError("bounded plain Rovent identifier required")
    return value


def _unique_strings(values, validator, maximum, description):
    if type(values) is not list or not 1 <= len(values) <= maximum:
        raise ValueError(f"bounded nonempty {description} required")
    result = [validator(value) for value in values]
    if len(set(result)) != len(result):
        raise ValueError(f"duplicate {description}")
    return result


@dataclass(frozen=True)
class FrozenTeacherActor:
    actor_id: str
    family: str
    source_revision: str
    source_sha256: str
    binary_sha256: str
    profile_sha256: str
    options_sha256: str
    assets_sha256: str
    rights_sha256: str

    @classmethod
    def from_dict(cls, value):
        fields(value, cls.__dataclass_fields__, "frozen Rovent CPU_T actor")
        _plain_id(value["actor_id"])
        if type(value["family"]) is not str or value["family"] not in ("stockfish", "reckless"):
            raise ValueError("registered frozen Stockfish/Reckless actor required")
        revision = value["source_revision"]
        if type(revision) is not str or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("teacher source must pin a full commit revision")
        for key in ("source_sha256", "binary_sha256", "profile_sha256", "options_sha256", "assets_sha256", "rights_sha256"):
            sha(value[key])
        return cls(**value)


@dataclass(frozen=True)
class SeriesStage:
    stage_id: str
    kind: StageKind
    parent_stage_id: str | None
    teacher_actor_ids: tuple[str, ...]
    round_id: str
    role_updates: tuple[int, int, int]

    @classmethod
    def from_dict(cls, value):
        fields(value, ("stage_id", "kind", "parent_stage_id", "teacher_actor_ids", "round_id", "role_updates"), "Rovent stage")
        _plain_id(value["stage_id"])
        _plain_id(value["round_id"])
        if value["parent_stage_id"] is not None:
            _plain_id(value["parent_stage_id"])
        if type(value["kind"]) is not str:
            raise ValueError("registered stage kind required")
        kind = StageKind(value["kind"])
        teachers = value["teacher_actor_ids"]
        if type(teachers) is not list or len(teachers) > 2:
            raise ValueError("at most two frozen teacher actors per stage")
        if len(set(_plain_id(v) for v in teachers)) != len(teachers):
            raise ValueError("duplicate stage teacher actor")
        updates = fields(value["role_updates"], ROLES, "explicit stage update declaration")
        counts = tuple(uint(updates[role], 2**32 - 1) for role in ROLES)
        if not any(counts):
            raise ValueError("an empty stage cannot stand in for a learning recipe")
        return cls(value["stage_id"], kind, value["parent_stage_id"], tuple(teachers), value["round_id"], counts)

    def to_dict(self):
        return {"stage_id": self.stage_id, "kind": self.kind.value, "parent_stage_id": self.parent_stage_id,
                "teacher_actor_ids": list(self.teacher_actor_ids), "round_id": self.round_id,
                "role_updates": dict(zip(ROLES, self.role_updates))}


def validate_parameter_ownership(value):
    """Declare ownership without creating or cloning optimizer moments.

    NNUEC membership is explicit; its optimizer may be the common P/C owner or
    a separate owner. V's private owner cannot own shared P/C or NNUEC state.
    The membership seal must later be checked against the actual model.
    """
    fields(value, ("pc_shared", "p_private", "c_private", "v_private", "nnuec", "membership_sha256"), "Rovent parameter ownership")
    for name in ("pc_shared", "p_private", "c_private", "v_private", "nnuec"):
        _plain_id(value[name])
    sha(value["membership_sha256"])
    if not value["pc_shared"] == value["p_private"] == value["c_private"]:
        raise ValueError("one P/C optimizer must own shared and P/C private moments")
    if value["v_private"] in (value["pc_shared"], value["nnuec"]):
        raise ValueError("V private optimizer cannot own shared P/C or NNUEC moments")
    return value


class SeriesRecipe:
    """Immutable preparation for one student/learner and an explicit fork DAG."""
    def __init__(self, value):
        fields(value, ("schema", "series_id", "student_id", "learner_id", "model_semantics_sha256", "source_sha256",
                       "actors", "parameter_ownership", "stages", "opening_splits", "global_holdout_openings"), "Rovent series")
        if value["schema"] != RECIPE_DOMAIN:
            raise ValueError("new Rovent recipe domain required; legacy recipes are not converted")
        for key in ("series_id", "student_id", "learner_id"):
            _plain_id(value[key])
        for key in ("model_semantics_sha256", "source_sha256"):
            sha(value[key])
        if type(value["actors"]) is not list or len(value["actors"]) != 2:
            raise ValueError("exactly two frozen CPU_T actors feed the common student")
        actors = tuple(FrozenTeacherActor.from_dict(actor) for actor in value["actors"])
        if {a.family for a in actors} != {"stockfish", "reckless"} or len({a.actor_id for a in actors}) != 2:
            raise ValueError("distinct frozen Stockfish and Reckless actors required")
        validate_parameter_ownership(value["parameter_ownership"])
        if type(value["stages"]) is not list or not 3 <= len(value["stages"]) <= MAX_STAGES:
            raise ValueError("common stage and both teacher-specific forks required")
        stages = tuple(SeriesStage.from_dict(stage) for stage in value["stages"])
        self._validate_stages(stages, actors)
        splits = value["opening_splits"]
        if type(splits) is not dict or not 1 <= len(splits) <= 65536:
            raise ValueError("bounded predeclared global opening split required")
        for opening, split in splits.items():
            identifier(opening)
            if split not in ("train", "validation", "holdout"):
                raise ValueError("unregistered global split")
        holdout = _unique_strings(value["global_holdout_openings"], identifier, 65536, "global holdout openings")
        if any(splits.get(opening) != "holdout" for opening in holdout):
            raise ValueError("global holdout must be excluded before all ancestor training")
        self._value = copy.deepcopy(value)
        self.actors, self.stages = actors, stages
        self.sha256 = digest(RECIPE_DOMAIN, self._value)
        self.split_sha256 = digest(SPLIT_DOMAIN, {"opening_splits": splits, "global_holdout_openings": holdout})

    def _check_integrity(self):
        if (digest(RECIPE_DOMAIN, self._value) != self.sha256
                or self.actors != tuple(FrozenTeacherActor.from_dict(actor) for actor in self._value["actors"])
                or self.stages != tuple(SeriesStage.from_dict(stage) for stage in self._value["stages"])):
            raise ValueError("frozen recipe changed after registration")
        expected_split = digest(SPLIT_DOMAIN, {key: self._value[key] for key in ("opening_splits", "global_holdout_openings")})
        if expected_split != self.split_sha256:
            raise ValueError("global split changed after registration")

    @staticmethod
    def _validate_stages(stages, actors):
        known, common, forks = {}, None, set()
        actor_ids = {actor.actor_id for actor in actors}
        for stage in stages:
            if stage.stage_id in known or any(actor not in actor_ids for actor in stage.teacher_actor_ids):
                raise ValueError("duplicate stage or unfrozen teacher actor")
            if stage.kind is StageKind.COMMON:
                if common is not None or known or stage.parent_stage_id is not None or set(stage.teacher_actor_ids) != actor_ids:
                    raise ValueError("one initial common student must consume both frozen teachers")
                common = stage.stage_id
            else:
                if stage.parent_stage_id not in known:
                    raise ValueError("stage parent must precede its child; cyclic/unbound lineage rejected")
                if stage.kind is StageKind.TEACHER_FT:
                    if stage.parent_stage_id != common or len(stage.teacher_actor_ids) != 1:
                        raise ValueError("teacher FT must fork the same common checkpoint stage")
                    actor = stage.teacher_actor_ids[0]
                    if actor in forks:
                        raise ValueError("duplicate teacher-specific FT fork")
                    forks.add(actor)
                elif stage.kind is StageKind.MIXED:
                    if forks != actor_ids or set(stage.teacher_actor_ids) != actor_ids:
                        raise ValueError("Mixed follows both teacher FT declarations and uses both actors")
                elif stage.kind is StageKind.SELF and (forks != actor_ids or stage.teacher_actor_ids):
                    raise ValueError("Self follows teacher FT and declares no external teacher")
            known[stage.stage_id] = stage
        if forks != actor_ids:
            raise ValueError("both teacher-specific FT forks must be declared")

    def to_dict(self):
        self._check_integrity()
        return copy.deepcopy(self._value)

    def stage(self, stage_id):
        self._check_integrity()
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        raise ValueError("stage absent from frozen recipe")

    def lineage(self, stage_id):
        stages, result = {stage.stage_id: stage for stage in self.stages}, []
        current = self.stage(stage_id)
        while current is not None:
            result.append(current)
            current = stages.get(current.parent_stage_id)
        return tuple(reversed(result))

    def inherited_teachers(self, stage_id):
        return tuple(sorted({actor for stage in self.lineage(stage_id) for actor in stage.teacher_actor_ids}))


def validate_checkpoint_metadata(recipe, value):
    """Validate a pinned description before a future loader mutates caller state.

    Byte verification, native producer authority and actual optimizer state are
    separate required intake work. This function never opens a checkpoint.
    """
    if type(recipe) is not SeriesRecipe:
        raise ValueError("validated Rovent recipe required")
    fields(value, ("schema", "recipe_sha256", "stage_id", "checkpoint_sha256", "model_sha256", "parent_checkpoint_sha256",
                   "parameter_membership_sha256", "split_sha256", "state_sha256", "completed_updates", "pending_microbatches",
                   "lineage_teacher_actor_ids", "training_input_sha256", "training_origin_sha256"), "Rovent checkpoint metadata")
    if value["schema"] != CHECKPOINT_DOMAIN or value["recipe_sha256"] != recipe.sha256 or value["split_sha256"] != recipe.split_sha256:
        raise ValueError("checkpoint recipe/global split domain mismatch")
    stage = recipe.stage(value["stage_id"])
    for key in ("checkpoint_sha256", "model_sha256", "parameter_membership_sha256"):
        sha(value[key])
    if value["parameter_membership_sha256"] != recipe.to_dict()["parameter_ownership"]["membership_sha256"]:
        raise ValueError("checkpoint parameter/optimizer membership changed")
    if stage.parent_stage_id is None:
        if value["parent_checkpoint_sha256"] is not None:
            raise ValueError("common stage has no parent checkpoint")
    else:
        sha(value["parent_checkpoint_sha256"])
        if value["parent_checkpoint_sha256"] == value["checkpoint_sha256"]:
            raise ValueError("checkpoint cannot parent itself")
    states = fields(value["state_sha256"], STATE_PARTS, "complete resume state pins")
    for part in states.values():
        sha(part)
    counts = fields(value["completed_updates"], ROLES, "checkpoint completed optimizer updates")
    for role, limit in zip(ROLES, stage.role_updates):
        uint(counts[role], limit)
    if uint(value["pending_microbatches"]) != 0:
        raise ValueError("checkpoint/fork requires a completed optimizer boundary")
    if value["lineage_teacher_actor_ids"] != list(recipe.inherited_teachers(stage.stage_id)):
        raise ValueError("teacher-specific FT cannot erase common ancestor teacher exposure")
    for name in ("training_input_sha256", "training_origin_sha256"):
        values = value[name]
        if type(values) is not list or len(values) > MAX_RECORDS or len(set(sha(v) for v in values)) != len(values):
            raise ValueError("bounded unique inherited training input/origin seals required")
    return value


def resume_plan(recipe, checkpoint, *, stage_id):
    validate_checkpoint_metadata(recipe, checkpoint)
    if checkpoint["stage_id"] != stage_id:
        raise ValueError("resume cannot change stage, teacher mixture, or optimizer ownership")
    return {"schema": "rz-rovent-series-resume-plan/1", "mode": "resume_exact_state", "stage_id": stage_id,
            "checkpoint": copy.deepcopy(checkpoint), "reset_state": [], "execution_authorized": False}


def fork_plan(recipe, checkpoint, *, stage_id):
    validate_checkpoint_metadata(recipe, checkpoint)
    stage = recipe.stage(stage_id)
    if stage.parent_stage_id != checkpoint["stage_id"]:
        raise ValueError("fork must use the declared parent stage checkpoint")
    parent = recipe.stage(checkpoint["stage_id"])
    if checkpoint["completed_updates"] != dict(zip(ROLES, parent.role_updates)):
        raise ValueError("fork requires the completed common/parent checkpoint")
    return {"schema": "rz-rovent-series-fork-plan/1", "mode": "fork_weights_only", "stage_id": stage_id,
            "parent_checkpoint_sha256": checkpoint["checkpoint_sha256"], "model_sha256": checkpoint["model_sha256"],
            "inherited_training_input_sha256": list(checkpoint["training_input_sha256"]),
            "inherited_training_origin_sha256": list(checkpoint["training_origin_sha256"]),
            "lineage_teacher_actor_ids": list(recipe.inherited_teachers(stage_id)),
            "reset_state": list(STATE_PARTS), "completed_updates": dict.fromkeys(ROLES, 0),
            "execution_authorized": False}


def common_teacher_forks(recipe, checkpoint):
    """Prepare both teacher FT branches from one identical common checkpoint."""
    validate_checkpoint_metadata(recipe, checkpoint)
    if recipe.stage(checkpoint["stage_id"]).kind is not StageKind.COMMON:
        raise ValueError("both teacher forks require the completed common student")
    plans = [fork_plan(recipe, checkpoint, stage_id=stage.stage_id)
             for stage in recipe.stages if stage.kind is StageKind.TEACHER_FT]
    return {"schema": "rz-rovent-common-teacher-forks/1", "recipe_sha256": recipe.sha256,
            "common_checkpoint_sha256": checkpoint["checkpoint_sha256"], "forks": plans,
            "execution_authorized": False}


def index_original_input(recipe, *, input_snapshot, input_sha256, record_sha256, producer_sha256,
                         stage_id, round_id, teacher_actor_id):
    """Bind original input semantics without accepting or editing future labels."""
    input_validator, input_seal = validate_input, seal_input
    if type(input_snapshot) is dict and input_snapshot.get("schema") == "rz-rovent-submitted-input/1":
        from .rovent_submitted_input import validate_input as input_validator, seal_input as input_seal
    input_validator(input_snapshot)
    if sha(input_sha256) != input_seal(input_snapshot):
        raise ValueError("original selected input seal mismatch")
    value = {"input_sha256": input_sha256, "record_sha256": record_sha256, "producer_sha256": producer_sha256,
             "stage_id": stage_id, "round_id": round_id, "teacher_actor_id": teacher_actor_id,
             "opening_id": input_snapshot["opening_id"], "game_id": input_snapshot["game_id"],
             "line_genealogy_id": input_snapshot["line_genealogy_id"],
             "rules_state_sha256": input_snapshot["rules_anchor"]["rules_state_sha256"],
             "transposition_sha256": input_snapshot["transposition_sha256"]}
    return copy.deepcopy(validate_record_origin(recipe, value))


def validate_record_origin(recipe, value):
    """Index an original sealed row without editing its inputs, heads or masks.

    producer/record pins preserve the origin; native head and Rules provenance
    must still pass the source-specific corpus consumer. CP/mate is deliberately
    absent from this index and can never become a WDL target here.
    """
    registered = recipe.to_dict()
    return _validate_record_origin(value, {stage.stage_id: stage for stage in recipe.stages}, registered["opening_splits"])


def _validate_record_origin(value, stages, opening_splits):
    fields(value, ("input_sha256", "record_sha256", "producer_sha256", "stage_id", "round_id", "teacher_actor_id",
                   "opening_id", "game_id", "line_genealogy_id", "rules_state_sha256", "transposition_sha256"), "Rovent original row index")
    _plain_id(value["stage_id"])
    stage = stages.get(value["stage_id"])
    if stage is None:
        raise ValueError("row stage absent from frozen recipe")
    # Own/Rules-WDL/V observations retain a null teacher even inside a stage
    # with teacher actors. This origin index cannot issue policy supervision;
    # the source-specific native head consumer must verify that separately.
    if value["round_id"] != stage.round_id or (value["teacher_actor_id"] is not None
            and value["teacher_actor_id"] not in stage.teacher_actor_ids):
        raise ValueError("row differs from its frozen round/teacher origin")
    for key in ("input_sha256", "record_sha256", "producer_sha256", "rules_state_sha256", "transposition_sha256"):
        sha(value[key])
    for key in ("opening_id", "game_id", "line_genealogy_id"):
        identifier(value[key])
    if value["opening_id"] not in opening_splits:
        raise ValueError("row opening absent from globally registered split")
    return value


def record_origin_digest(recipe, value):
    validate_record_origin(recipe, value)
    return digest(ORIGIN_DOMAIN, value)


class GlobalLineageIndex:
    """Audit all teacher/round/branch inputs together, including ancestors.

    This index is bounded metadata, not a replacement corpus or training
    authority. Cross-split identities are excluded symmetrically. An already
    trained contaminated ancestor is rejected instead of relabelled or fixed by
    dropping only a descendant row.
    """
    def __init__(self, recipe, records):
        if type(recipe) is not SeriesRecipe or type(records) is not list or len(records) > MAX_RECORDS:
            raise ValueError("validated series and bounded global origin index required")
        splits = recipe.to_dict()["opening_splits"]
        stages = {stage.stage_id: stage for stage in recipe.stages}
        identities, seen, parents = {}, set(), list(range(len(records)))

        def root(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        for index, record in enumerate(records):
            _validate_record_origin(record, stages, splits)
            origin = (record["input_sha256"], record["producer_sha256"], record["stage_id"], record["teacher_actor_id"])
            if origin in seen:
                raise ValueError("duplicate original row index")
            seen.add(origin)
            for key in self._keys(record):
                if key in identities:
                    parents[root(index)] = root(identities[key])
                else:
                    identities[key] = index
        components = {}
        for index, record in enumerate(records):
            components.setdefault(root(index), set()).add(splits[record["opening_id"]])
        contaminated = {component for component, parts in components.items() if len(parts) > 1}
        leaked = {key for key, index in identities.items() if root(index) in contaminated}
        self._recipe_sha256 = recipe.sha256
        self._records = copy.deepcopy(records)
        self._by_origin = {digest(ORIGIN_DOMAIN, record): record for record in self._records}
        self._splits = copy.deepcopy(splits)
        self._excluded = {record["input_sha256"] for index, record in enumerate(records) if root(index) in contaminated}
        self.sha256 = digest(INDEX_DOMAIN, {"recipe_sha256": recipe.sha256, "records": records})
        self.cross_split_identities = len(leaked)
        self._integrity_sha256 = self._integrity()

    def _integrity(self):
        return digest("rz-rovent-global-lineage-index-state/1", {"recipe_sha256": self._recipe_sha256,
            "records": self._records, "opening_splits": self._splits, "excluded": sorted(self._excluded),
            "by_origin": self._by_origin, "index_sha256": self.sha256,
            "cross_split_identities": self.cross_split_identities})

    def _check_integrity(self):
        if self._integrity() != self._integrity_sha256:
            raise ValueError("immutable global lineage index changed")

    @staticmethod
    def _keys(record):
        return tuple((name, record[name]) for name in ("input_sha256", "game_id", "line_genealogy_id", "rules_state_sha256", "transposition_sha256"))

    def rows(self, split, *, optimization=False):
        self._check_integrity()
        if split not in ("train", "validation", "holdout") or optimization and split != "train":
            raise ValueError("validation/global holdout cannot enter optimization")
        return copy.deepcopy([record for record in self._records if self._splits[record["opening_id"]] == split
                              and record["input_sha256"] not in self._excluded])

    def admit_checkpoint_lineage(self, recipe, checkpoint, *, ancestors):
        self._check_integrity()
        validate_checkpoint_metadata(recipe, checkpoint)
        if recipe.sha256 != self._recipe_sha256 or type(ancestors) is not list or len(ancestors) > MAX_STAGES:
            raise ValueError("same globally indexed recipe and complete checkpoint ancestors required")
        chain = ancestors + [checkpoint]
        for item in chain:
            validate_checkpoint_metadata(recipe, item)
        expected = recipe.lineage(checkpoint["stage_id"])
        if [item.get("stage_id") for item in chain] != [stage.stage_id for stage in expected]:
            raise ValueError("all common/branch ancestor checkpoints must be audited")
        prior, seen_training = None, set()
        by_origin = self._by_origin
        allowed = {origin for origin, record in by_origin.items() if self._splits[record["opening_id"]] == "train"
                   and record["input_sha256"] not in self._excluded}
        for index, item in enumerate(chain):
            validate_checkpoint_metadata(recipe, item)
            if item["parent_checkpoint_sha256"] != prior:
                raise ValueError("checkpoint parent digest chain mismatch")
            training = set(item["training_origin_sha256"])
            if not seen_training <= training or not training <= allowed:
                raise ValueError("global holdout/cross-split contamination or erased ancestor training input")
            if {by_origin[origin]["input_sha256"] for origin in training} != set(item["training_input_sha256"]):
                raise ValueError("training origin/input seal projection mismatch")
            stage_scope = {stage.stage_id for stage in recipe.lineage(item["stage_id"])}
            covered = {origin for origin, record in by_origin.items() if record["stage_id"] in stage_scope}
            if not training <= covered:
                raise ValueError("checkpoint trained on an unrelated future or sibling branch")
            if index < len(chain) - 1 and item["completed_updates"] != dict(zip(ROLES, expected[index].role_updates)):
                raise ValueError("child checkpoint requires a completed parent stage")
            seen_training = training
            prior = item["checkpoint_sha256"]
        return {"schema": "rz-rovent-global-lineage-audit/1", "recipe_sha256": recipe.sha256,
                "index_sha256": self.sha256, "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "training_origins": len(seen_training), "training_inputs": len(checkpoint["training_input_sha256"]),
                "cross_split_identities": self.cross_split_identities,
                "excluded_input_seals": len(self._excluded), "native_intake_verified": False,
                "execution_authorized": False}


class LegacyIntakeStatus(str, Enum):
    AUDITED = "legacy_provenance_audited_not_trainable"
    UNSUPPORTED_TEACHER = "unsupported_legacy_teacher_family"
    UNSUPPORTED_FORMAT = "unsupported_legacy_format"
    REJECTED = "rejected_legacy_provenance"


def _legacy_result(recipe, stage_id, actor_id, status, code, **observations):
    return {"schema": "rz-rovent-legacy-intake-audit/1", "recipe_sha256": recipe.sha256,
            "stage_id": stage_id, "teacher_actor_id": actor_id, "status": status.value, "code": code,
            "actual_training_admissible": False, "execution_authorized": False,
            "actual_training_executed": False, "optimizer_updates": 0, "arena_eligible": False,
            **observations}


def _legacy_actor(recipe, stage_id, teacher_actor_id):
    stage = recipe.stage(stage_id)
    if teacher_actor_id not in stage.teacher_actor_ids:
        raise ValueError("legacy intake actor is not assigned to its stage")
    return next(actor for actor in recipe.actors if actor.actor_id == teacher_actor_id)


def _audit_legacy_dataset(recipe, dataset, *, stage_id, actor, expected_model_sha256,
                          expected_runtime_sha256, max_records, check_budget):
    producer = dataset.authority
    for key, expected in (("teacher_binary_sha256", actor.binary_sha256),
                          ("teacher_profile_sha256", actor.profile_sha256),
                          ("model_sha256", sha(expected_model_sha256)),
                          ("runtime_sha256", sha(expected_runtime_sha256))):
        if producer[key] != expected:
            raise ValueError("legacy producer differs from its frozen teacher/model/runtime pin: " + key)
    splits = recipe.to_dict()["opening_splits"]
    stages = {stage.stage_id: stage for stage in recipe.stages}
    selected = recipe.stage(stage_id)
    producer_sha256 = digest(producer["schema"], producer)
    if any(splits.get(opening) != split for opening, split in dataset.opening_splits.items()):
        raise ValueError("legacy producer split differs from preregistered global split")
    origins, heads, counts = [], [], dict.fromkeys(ROLES, 0)
    for split in ("train", "validation", "holdout"):
        for record in dataset.rows(split):
            check_budget()
            if len(origins) >= max_records:
                raise ValueError("legacy audit record capacity exceeded; no sampled admission")
            targets = validate_record(record)
            snapshot = record["input"]  # Already validated against its original seal above.
            origin = {"input_sha256": record["input_sha256"],
                "record_sha256": digest("rz-rovent-legacy-hydrated-record-snapshot/1", record),
                "producer_sha256": producer_sha256, "stage_id": stage_id, "round_id": selected.round_id,
                "teacher_actor_id": actor.actor_id, **{key: snapshot[key] for key in
                    ("opening_id", "game_id", "line_genealogy_id", "transposition_sha256")},
                "rules_state_sha256": snapshot["rules_anchor"]["rules_state_sha256"]}
            _validate_record_origin(origin, stages, splits)
            origins.append(origin)
            labels = record["future_targets"]
            heads.append({"input_sha256": record["input_sha256"], "record_sha256": origin["record_sha256"],
                "split": split, "role": record["input"]["role"], "capture_sequence": record["input"]["capture_sequence"],
                "action_support_mask": list(record["input"]["action_support_mask"]),
                "heads": {head: {"masked": targets[head] is None,
                    "source": None if label is None else label["source"],
                    "evidence_sha256": None if label is None else label["raw_evidence_sha256"],
                    "observed_sequence": None if label is None else label["observed_sequence"]}
                    for head, label in labels.items()},
                "task_target_known_mask": None if labels["task"] is None else list(labels["task"]["target_known_mask"])})
            counts[record["input"]["role"]] += 1
    check_budget()
    # A valid legacy row is still not a new native Rovent producer, nor does a
    # serialized receipt prove the live game/trainer authority of its origin.
    return _legacy_result(recipe, stage_id, actor.actor_id, LegacyIntakeStatus.AUDITED,
        "LegacyReadOnlyProvenanceAudit", dataset_sha256=dataset.sha256, producer=producer,
        record_snapshot_domain="rz-rovent-legacy-hydrated-record-snapshot/1", origins=origins,
        head_observations=heads, role_records=counts, exclusions=dict(dataset.exclusions),
        limitations=["legacy_domain_preserved_no_automatic_conversion",
                     "native_rovent_collector_authority_not_issued",
                     "serialized_provenance_does_not_certify_live_original_authority",
                     "actual_parameter_membership_and_checkpoint_bytes_not_loaded"])


def audit_legacy_bundle(recipe, bundle, *, stage_id, teacher_actor_id, expected_model_sha256,
                        expected_runtime_sha256, max_records=4096, check_budget=lambda: None):
    """Reuse the original whole-JSON parser; never promote its eligibility.

    Legacy first-generation code itself excludes this format from actual
    optimizer dispatch because its producer does not seal the physical native
    collection chain. Successful head evidence joins remain useful for audit.
    """
    actor = _legacy_actor(recipe, stage_id, teacher_actor_id)
    if actor.family != "stockfish":
        return _legacy_result(recipe, stage_id, actor.actor_id, LegacyIntakeStatus.UNSUPPORTED_TEACHER,
            "LegacyPolicySourceIsStockfish19Only", required="new_teacher_aware_native_corpus_domain")
    try:
        check_budget()
        if not 1 <= uint(max_records, 4096):
            raise ValueError("bounded legacy audit record ceiling required")
        fields(bundle, ("producer", "trajectories", "opening_splits", "allowed_evidence", "records"), "legacy generation input bundle")
        if type(bundle["records"]) is not list or len(bundle["records"]) > max_records:
            raise ValueError("legacy audit record capacity exceeded")
        dataset = GenerationDataset(bundle["records"], producer=bundle["producer"], trajectories=bundle["trajectories"],
            opening_splits=bundle["opening_splits"], allowed_evidence=bundle["allowed_evidence"])
        result = _audit_legacy_dataset(recipe, dataset, stage_id=stage_id, actor=actor,
            expected_model_sha256=expected_model_sha256, expected_runtime_sha256=expected_runtime_sha256,
            max_records=max_records, check_budget=check_budget)
        result["data_format"] = "legacy_whole_json_bundle"
        result["limitations"].append("legacy_bundle_lacks_native_physical_receipt_chain")
        return result
    except ValueError as error:
        return _legacy_result(recipe, stage_id, actor.actor_id, LegacyIntakeStatus.REJECTED,
            "LegacyProvenanceRejected", error_type=type(error).__name__, reason=str(error))


def audit_legacy_chunks(recipe, manifest, *, stage_id, teacher_actor_id, expected_model_sha256,
                        expected_runtime_sha256, expected_manifest_sha256, corpus_root, index_root,
                        max_records=4096, check_budget=lambda: None):
    """Use original bounded chunk/native/head validators and their exclusions."""
    actor = _legacy_actor(recipe, stage_id, teacher_actor_id)
    if actor.family != "stockfish":
        return _legacy_result(recipe, stage_id, actor.actor_id, LegacyIntakeStatus.UNSUPPORTED_TEACHER,
            "LegacyPolicySourceIsStockfish19Only", required="new_teacher_aware_native_corpus_domain")
    from .generation_chunks import ChunkGenerationDataset, validate_manifest
    try:
        check_budget()
        if not 1 <= uint(max_records, 4096):
            raise ValueError("bounded legacy audit record ceiling required")
        validate_manifest(manifest)
        if manifest["total_records"] > max_records or manifest["total_bytes"] > 64 * 1024**2:
            raise ValueError("legacy preparation audit corpus exceeds bounded records/64MiB")
        with ChunkGenerationDataset(corpus_root, manifest, expected_manifest_sha256=expected_manifest_sha256,
                                    index_root=index_root, max_index_bytes=16 * 1024**2) as dataset:
            return _audit_legacy_dataset(recipe, dataset, stage_id=stage_id, actor=actor,
                expected_model_sha256=expected_model_sha256, expected_runtime_sha256=expected_runtime_sha256,
                max_records=max_records, check_budget=check_budget) | {"data_format": manifest["schema"]}
    except ValueError as error:
        return _legacy_result(recipe, stage_id, actor.actor_id, LegacyIntakeStatus.REJECTED,
            "LegacyProvenanceRejected", error_type=type(error).__name__, reason=str(error))
