"""Explicit R auxiliary recipe/sampler, separate from P/C role samples.

This module is stdlib-only. R observations never create a proposer/critic
update, native capability, optimizer or V task label. A live trainer may use
this source only alongside an independently valid P/C update boundary.
"""
from __future__ import annotations

import copy
import math
import random
import struct

from .generation_data import digest, fields, identifier, sha, uint

RECIPE_DOMAIN = "rz-rovent-r-auxiliary-recipe/1"
RECIPE_SEAL_DOMAIN = "rz-rovent-r-auxiliary-recipe-seal/1"
SAMPLER_DOMAIN = "rz-rovent-r-auxiliary-sampler/1"
POOL_DOMAIN = "rz-rovent-r-auxiliary-pool/1"
UPDATE_DOMAIN = "rz-rovent-r-auxiliary-update/1"
JOURNAL_DOMAIN = "rz-rovent-r-auxiliary-journal/1"
R_SOURCE_DOMAIN = "rz-rovent-r-readout-record/1"
MAX_ROWS = 262144
MAX_AUX_ROWS_PER_UPDATE = 64


def validate_auxiliary_recipe(value, *, recipe=None):
    """Validate an explicitly present auxiliary object; None stays None.

    The window codec retains the distinction between an absent field and an
    explicit null. This function never adds a field or re-signs either form.
    """
    if value is None:
        return None
    fields(value, ("schema", "manifest_file_sha256", "index_file_sha256", "sampling_seed", "rows_per_pc_update",
        "weight", "normalization", "optimizer_owner", "supervision_source"), "explicit R auxiliary recipe")
    if (value["schema"] != RECIPE_DOMAIN or value["normalization"] != "valid_exact_trajectory_rows"
            or value["supervision_source"] != "r_auxiliary_only_preserve_pc_evidence"):
        raise ValueError("explicit exact-trajectory R auxiliary recipe required")
    sha(value["manifest_file_sha256"])
    sha(value["index_file_sha256"])
    uint(value["sampling_seed"], 2**32 - 1)
    if not 1 <= uint(value["rows_per_pc_update"], MAX_AUX_ROWS_PER_UPDATE):
        raise ValueError("R auxiliary rows per valid P/C update must be bounded 1..64")
    weight = value["weight"]
    try:
        finite = type(weight) in (int, float) and math.isfinite(weight)
    except OverflowError:
        finite = False
    if not finite or weight <= 0:
        raise ValueError("finite explicitly positive R auxiliary loss weight required")
    identifier(value["optimizer_owner"])
    if recipe is not None:
        owners = recipe.to_dict()["parameter_ownership"]
        if (owners["nnuec"] != value["optimizer_owner"]
                or owners["nnuec"] in {owners[name] for name in ("pc_shared", "p_private", "c_private", "v_private")}):
            raise ValueError("R auxiliary requires its registered exclusive NNUEC optimizer owner")
    return value


def auxiliary_recipe_digest(value):
    """Seal the new recipe with an explicit cross-language f64 coefficient."""
    registration = validate_auxiliary_recipe(value)
    if registration is None:
        raise ValueError("a disabled auxiliary has no enabled recipe seal")
    coefficient = struct.pack(">d", float(registration["weight"])).hex()
    return digest(RECIPE_SEAL_DOMAIN, {"algorithm": "coefficient-f64-bits-v1",
        "recipe": {**registration, "weight": coefficient}})


def validate_auxiliary_sample(value):
    fields(value, ("input_sha256", "origin_sha256", "record_sha256", "nnuec_input", "nnuec_model_sha256",
        "nnuec_weights_file_sha256", "nnuec_semantics", "nnuec_wdl", "retirement_sha256", "wdl_raw_sha256", "split"),
        "separate original R auxiliary sample")
    for name in ("input_sha256", "origin_sha256", "record_sha256", "nnuec_model_sha256", "nnuec_weights_file_sha256",
                 "retirement_sha256"):
        sha(value[name])
    identifier(value["nnuec_semantics"])
    if value["split"] != "train" or type(value["nnuec_input"]) is not dict:
        raise ValueError("only original train-split R reference input may enter auxiliary optimization")
    target = value["nnuec_wdl"]
    if target is None:
        if value["wdl_raw_sha256"] is not None:
            sha(value["wdl_raw_sha256"])
    else:
        if (type(target) is not list or len(target) != 3
                or any(type(item) not in (float, int) or type(item) is bool or item not in (0, 1) for item in target)
                or sum(target) != 1):
            raise ValueError("R auxiliary target must be an actual one-hot Rules outcome, never CP/mate")
        sha(value["wdl_raw_sha256"])
    return value


def auxiliary_pool_digest(canonical_index_sha256, samples):
    """Seal the ordered eligible pool, retaining the original index identity."""
    sha(canonical_index_sha256)
    if type(samples) not in (list, tuple) or not 1 <= len(samples) <= MAX_ROWS:
        raise ValueError("bounded nonempty original R auxiliary pool required")
    projected, records, origins = [], set(), set()
    for sample in samples:
        validate_auxiliary_sample(sample)
        record, origin = sample["record_sha256"], sample["origin_sha256"]
        if record in records or origin in origins:
            raise ValueError("duplicate R occurrence in auxiliary pool")
        records.add(record)
        origins.add(origin)
        projected.append({"source_domain": R_SOURCE_DOMAIN,
            **{name: sample[name] for name in ("input_sha256", "origin_sha256", "record_sha256")}})
    return digest(POOL_DOMAIN, {"index_sha256": canonical_index_sha256, "occurrences": projected})


class RAuxiliarySampler:
    """A separate seeded R cursor; it never impersonates a P/C role sampler."""
    def __init__(self, rows, *, seed, pool_sha256, state=None):
        if type(rows) not in (tuple, list) or not 1 <= len(rows) <= MAX_ROWS:
            raise ValueError("bounded nonempty original R row tokens required")
        self.rows = tuple(rows)
        self.seed, self.pool_sha256 = uint(seed, 2**32 - 1), sha(pool_sha256)
        self.epoch, self.cursor, self.samples = 0, 0, 0
        if state is not None:
            fields(state, ("schema", "seed", "pool_sha256", "epoch", "cursor", "samples", "permutation"), "R auxiliary sampler state")
            if state["schema"] != SAMPLER_DOMAIN or state["seed"] != seed or state["pool_sha256"] != pool_sha256:
                raise ValueError("R auxiliary sampler changed source or seed")
            self.epoch, self.cursor, self.samples = uint(state["epoch"]), uint(state["cursor"], len(rows)), uint(state["samples"])
            if self.samples != self.epoch * len(rows) + self.cursor:
                raise ValueError("R auxiliary sampler lost exact consumed position")
        self.permutation = self._order()
        if state is not None and (type(state["permutation"]) is not list
                or any(type(index) is not int for index in state["permutation"])
                or state["permutation"] != self.permutation):
            raise ValueError("R auxiliary seeded permutation differs from checkpoint")

    def _order(self):
        permutation = list(range(len(self.rows)))
        random.Random(digest(SAMPLER_DOMAIN, [self.seed, self.pool_sha256, self.epoch])).shuffle(permutation)
        return permutation

    def take(self, count):
        count = uint(count, MAX_AUX_ROWS_PER_UPDATE)
        result = []
        for _ in range(count):
            if self.cursor == len(self.rows):
                self.epoch, self.cursor = self.epoch + 1, 0
                self.permutation = self._order()
            result.append(self.rows[self.permutation[self.cursor]])
            self.cursor += 1
            self.samples += 1
        return result

    def state(self):
        return {"schema": SAMPLER_DOMAIN, "seed": self.seed, "pool_sha256": self.pool_sha256,
            "epoch": self.epoch, "cursor": self.cursor, "samples": self.samples, "permutation": list(self.permutation)}


def update_projection(auxiliary_recipe, samples, *, pc_valid_rows):
    """Project original R seals only after a real P/C target is available.

    The projection is evidence to join with the parent's actual update owner,
    not a completed update or training authorization. Unknown R targets remain
    observations and never advance NNUEC moments on their own.
    """
    registration = validate_auxiliary_recipe(auxiliary_recipe)
    if registration is None:
        raise ValueError("disabled R auxiliary stream cannot publish consumed rows")
    if not 1 <= uint(pc_valid_rows, 4096):
        raise ValueError("R auxiliary stream cannot create an update without a valid P/C target")
    if type(samples) not in (tuple, list) or len(samples) != registration["rows_per_pc_update"]:
        raise ValueError("R auxiliary sample count differs from the registered P/C boundary")
    rows, valid, unique = [], 0, set()
    for sample in samples:
        validate_auxiliary_sample(sample)
        target_known = sample["nnuec_wdl"] is not None
        valid += target_known
        if target_known:
            unique.add(sample["input_sha256"])
        rows.append({name: copy.deepcopy(sample[name]) for name in ("input_sha256", "origin_sha256", "record_sha256",
            "retirement_sha256", "wdl_raw_sha256")})
    return {"schema": UPDATE_DOMAIN, "manifest_file_sha256": registration["manifest_file_sha256"],
        "index_file_sha256": registration["index_file_sha256"], "optimizer_owner": registration["optimizer_owner"],
        "occurrences": rows, "valid_rows": valid, "unique_valid_inputs": len(unique),
        "pc_valid_rows": pc_valid_rows, "normalization": registration["normalization"],
        "supervision_source": registration["supervision_source"],
        "pc_nnuec_evidence_preserved_not_optimized": True,
        "execution_authorized": False}


def enabled(spec):
    return type(spec) is dict and spec.get("schema") == "rz-rovent-train-window/2" and spec.get("nnuec_r_auxiliary") is not None


def initial_auxiliary_journal(recipe):
    registration = copy.deepcopy(validate_auxiliary_recipe(recipe))
    if registration is None:
        raise ValueError("disabled R auxiliary has no sampler or update journal")
    return {"schema": JOURNAL_DOMAIN, "recipe": registration, "pool_sha256": None, "sampler": None,
        "current_consumed_rows": 0, "current_valid_rows": 0, "current_nnuec_updates": 0,
        "retained_input_sha256": [], "retained_origin_sha256": [],
        "ancestor_input_sha256": [], "ancestor_origin_sha256": [], "ancestor_nnuec_updates": 0,
        "pc_nnuec_evidence_optimized": False}


def _seals(value):
    if (type(value) is not list or len(value) > MAX_ROWS
            or value != sorted(set(sha(item) for item in value))):
        raise ValueError("sorted unique bounded R retained seal set required")
    return value


def validate_update_projection(value, *, registration=None):
    fields(value, ("schema", "manifest_file_sha256", "index_file_sha256", "optimizer_owner", "occurrences",
        "valid_rows", "unique_valid_inputs", "pc_valid_rows", "normalization", "supervision_source",
        "pc_nnuec_evidence_preserved_not_optimized", "execution_authorized",
        "r_auxiliary_nnuec_optimizer_steps", "r_auxiliary_nnuec_scheduler_steps", "r_auxiliary_nnuec_moments_steps"),
        "original R auxiliary update projection")
    if (value["schema"] != UPDATE_DOMAIN or value["normalization"] != "valid_exact_trajectory_rows"
            or value["supervision_source"] != "r_auxiliary_only_preserve_pc_evidence"
            or value["pc_nnuec_evidence_preserved_not_optimized"] is not True or value["execution_authorized"] is not False):
        raise ValueError("R auxiliary projection cannot change supervision or issue authority")
    for name in ("manifest_file_sha256", "index_file_sha256"):
        sha(value[name])
    identifier(value["optimizer_owner"])
    rows = value["occurrences"]
    if type(rows) is not list or not 1 <= len(rows) <= MAX_AUX_ROWS_PER_UPDATE:
        raise ValueError("bounded original R auxiliary occurrence list required")
    for row in rows:
        fields(row, ("input_sha256", "origin_sha256", "record_sha256", "retirement_sha256", "wdl_raw_sha256"),
            "R auxiliary original occurrence seals")
        for name in ("input_sha256", "origin_sha256", "record_sha256", "retirement_sha256"):
            sha(row[name])
        if row["wdl_raw_sha256"] is not None:
            sha(row["wdl_raw_sha256"])
    valid = uint(value["valid_rows"], len(rows))
    uint(value["unique_valid_inputs"], valid)
    if not uint(value["pc_valid_rows"], 4096):
        raise ValueError("R auxiliary cannot complete without a valid P/C update")
    step = int(valid > 0)
    if any(type(value[name]) is not int or value[name] != step for name in
           ("r_auxiliary_nnuec_optimizer_steps", "r_auxiliary_nnuec_scheduler_steps", "r_auxiliary_nnuec_moments_steps")):
        raise ValueError("R target zero must preserve NNUEC optimizer/scheduler/moments steps")
    if registration is not None:
        validate_auxiliary_recipe(registration)
        for key in ("manifest_file_sha256", "index_file_sha256", "optimizer_owner", "normalization", "supervision_source"):
            if value[key] != registration[key]:
                raise ValueError("R auxiliary update changed the registered source or optimizer owner")
        if len(rows) != registration["rows_per_pc_update"]:
            raise ValueError("R auxiliary consumed row count changed registered boundary")
    return value


def validate_auxiliary_journal(value, updates):
    fields(value, ("schema", "recipe", "pool_sha256", "sampler", "current_consumed_rows", "current_valid_rows",
        "current_nnuec_updates", "retained_input_sha256", "retained_origin_sha256", "ancestor_input_sha256",
        "ancestor_origin_sha256", "ancestor_nnuec_updates", "pc_nnuec_evidence_optimized"), "R auxiliary checkpoint journal")
    if value["schema"] != JOURNAL_DOMAIN or value["pc_nnuec_evidence_optimized"] is not False:
        raise ValueError("explicit R-only supervision journal required")
    registration = validate_auxiliary_recipe(value["recipe"])
    if registration is None:
        raise ValueError("R journal cannot use a disabled recipe")
    for name in ("retained_input_sha256", "retained_origin_sha256", "ancestor_input_sha256", "ancestor_origin_sha256"):
        _seals(value[name])
    uint(value["ancestor_nnuec_updates"])
    consumed, valid, steps, inputs, origins = 0, 0, 0, set(), set()
    for event in updates:
        projection = event.get("r_auxiliary")
        if event["role"] == "validator":
            if projection is not None:
                raise ValueError("V update cannot consume R auxiliary or mutate NNUEC")
            continue
        validate_update_projection(projection, registration=registration)
        consumed += len(projection["occurrences"])
        valid += projection["valid_rows"]
        steps += projection["r_auxiliary_nnuec_optimizer_steps"]
        inputs.update(row["input_sha256"] for row in projection["occurrences"])
        origins.update(row["origin_sha256"] for row in projection["occurrences"])
    expected = {"current_consumed_rows": consumed, "current_valid_rows": valid, "current_nnuec_updates": steps,
        "retained_input_sha256": sorted(inputs), "retained_origin_sha256": sorted(origins)}
    if any(type(value[name]) is not type(expected_value) or value[name] != expected_value for name, expected_value in expected.items()):
        raise ValueError("R auxiliary checkpoint lost consumed rows, seals or actual NNUEC step count")
    if value["pool_sha256"] is None:
        if value["sampler"] is not None or consumed:
            raise ValueError("R consumed rows require a bound original sampling pool")
    else:
        sha(value["pool_sha256"])
        state = value["sampler"]
        if state is not None and (type(state) is not dict or state.get("pool_sha256") != value["pool_sha256"]
                or state.get("schema") != SAMPLER_DOMAIN or state.get("samples") != consumed):
            raise ValueError("R sampler checkpoint cursor does not match actual consumed rows")
        if consumed and state is None:
            raise ValueError("R consumed rows require an exact sampler checkpoint")
    return value


def complete_auxiliary_update(journal, projection, sampler_state):
    validate_update_projection(projection, registration=journal["recipe"])
    journal["sampler"] = copy.deepcopy(sampler_state)
    journal["current_consumed_rows"] += len(projection["occurrences"])
    journal["current_valid_rows"] += projection["valid_rows"]
    journal["current_nnuec_updates"] += projection["r_auxiliary_nnuec_optimizer_steps"]
    journal["retained_input_sha256"] = sorted(set(journal["retained_input_sha256"]) | {row["input_sha256"] for row in projection["occurrences"]})
    journal["retained_origin_sha256"] = sorted(set(journal["retained_origin_sha256"]) | {row["origin_sha256"] for row in projection["occurrences"]})


def fork_auxiliary_journal(parent, registration):
    child = initial_auxiliary_journal(registration)
    child["ancestor_nnuec_updates"] = uint(parent["ancestor_nnuec_updates"] + parent["current_nnuec_updates"])
    for kind in ("input", "origin"):
        child["ancestor_" + kind + "_sha256"] = sorted(set(parent["ancestor_" + kind + "_sha256"]) | set(parent["retained_" + kind + "_sha256"]))
    return child


def checkpoint_projection(journal, updates):
    """Bound source/count/set projection; full state stays in the tensor file."""
    checked = validate_auxiliary_journal(journal, updates)
    from .rovent_training_window import retained_seal_set
    return {"schema": "rz-rovent-r-auxiliary-checkpoint-projection/1",
        "recipe_sha256": auxiliary_recipe_digest(checked["recipe"]),
        "manifest_file_sha256": checked["recipe"]["manifest_file_sha256"],
        "index_file_sha256": checked["recipe"]["index_file_sha256"], "pool_sha256": checked["pool_sha256"],
        "sampler_sha256": None if checked["sampler"] is None else digest(SAMPLER_DOMAIN, checked["sampler"]),
        "current_consumed_rows": checked["current_consumed_rows"], "current_valid_rows": checked["current_valid_rows"],
        "current_nnuec_updates": checked["current_nnuec_updates"], "ancestor_nnuec_updates": checked["ancestor_nnuec_updates"],
        "retained_input": retained_seal_set(checked["retained_input_sha256"], "input"),
        "retained_origin": retained_seal_set(checked["retained_origin_sha256"], "origin"),
        "ancestor_input": retained_seal_set(checked["ancestor_input_sha256"], "input"),
        "ancestor_origin": retained_seal_set(checked["ancestor_origin_sha256"], "origin"),
        "pc_nnuec_evidence_optimized": False, "execution_authorized": False}


def validate_checkpoint_projection(value):
    fields(value, ("schema", "recipe_sha256", "manifest_file_sha256", "index_file_sha256", "pool_sha256",
        "sampler_sha256", "current_consumed_rows", "current_valid_rows", "current_nnuec_updates", "ancestor_nnuec_updates",
        "retained_input", "retained_origin", "ancestor_input", "ancestor_origin", "pc_nnuec_evidence_optimized",
        "execution_authorized"), "R auxiliary tensor checkpoint projection")
    if (value["schema"] != "rz-rovent-r-auxiliary-checkpoint-projection/1" or value["pc_nnuec_evidence_optimized"] is not False
            or value["execution_authorized"] is not False):
        raise ValueError("R checkpoint projection cannot authorize execution or optimize preserved PC evidence")
    for key in ("recipe_sha256", "manifest_file_sha256", "index_file_sha256"):
        sha(value[key])
    for key in ("pool_sha256", "sampler_sha256"):
        if value[key] is not None:
            sha(value[key])
    for key in ("current_consumed_rows", "current_valid_rows", "current_nnuec_updates", "ancestor_nnuec_updates"):
        uint(value[key])
    if value["current_valid_rows"] > value["current_consumed_rows"] or value["current_nnuec_updates"] > value["current_valid_rows"]:
        raise ValueError("R checkpoint actual step/valid counts exceed consumed rows")
    for key in ("retained_input", "retained_origin", "ancestor_input", "ancestor_origin"):
        fields(value[key], ("sha256", "count"), "R auxiliary compact retained set")
        sha(value[key]["sha256"])
        uint(value[key]["count"], MAX_ROWS)
    return value
