"""Separate no-execution /2 supervision preparation metadata.

Only existing recipe/registration/parameter mathematics are shared. Existing
PREP1/PREP2 parsers, native Ready intake and history allow-lists are unchanged.
This record is not a tensor, optimizer, dataset eligibility or source grant.
"""
from __future__ import annotations

import copy
import hashlib

from .generation_data import fields, sha, uint
from .rovent_native_v2 import _artifact
from . import rovent_utility_training as mathematics
from .rovent_utility_intervention import COST_SCOPE, ROUTES, canonical_bytes


PREPARATION_DOMAIN = "rz-rovent-utility-intervention-preparation/1"
MAX_PREPARATION_BYTES = 64 << 10
PREPARATION_FIELDS = ("schema", "status", "recipe", "registration", "parameter_spec",
    "intervention_source", "preparation_sha256", "execution_authorized", "performed_optimizer_updates")
SOURCE_FIELDS = ("cohort_id", "profile_sha256", "publication", "durable_ack", "neutral_input_sha256",
    "neutral_target_sha256", "neutral_context_sha256", "root_rules_sha256", "role_queries",
    "frozen_pc_sources", "ancestor_datasets", "known_pair_sha256", "all_pair_sha256")


def _raw_pin(value):
    fields(value, ("sha256", "bytes"), "intervention original metadata raw pin")
    sha(value["sha256"])
    if not uint(value["bytes"], MAX_PREPARATION_BYTES):
        raise ValueError("nonempty bounded intervention metadata raw pin required")


def _seals(value):
    if type(value) is not list or len(value) > 3:
        raise ValueError("bounded original intervention pair seals required")
    for seal in value:
        sha(seal)
    if len(set(value)) != len(value):
        raise ValueError("distinct original intervention pair seals required")
    return value


def validate_intervention_source(value):
    """Check detached source shape; original FD/proof joins belong to intake."""
    fields(value, SOURCE_FIELDS, "intervention preparation original source")
    for name in ("cohort_id", "profile_sha256", "neutral_input_sha256", "neutral_target_sha256",
            "neutral_context_sha256", "root_rules_sha256"):
        sha(value[name])
    _raw_pin(value["publication"])
    _raw_pin(value["durable_ack"])
    queries = value["role_queries"]
    if type(queries) is not list or not 2 <= len(queries) <= 3:
        raise ValueError("complete original two or three intervention query sources required")
    recipients = set()
    for sequence, query in enumerate(queries, 1):
        fields(query, ("trial_sequence", "recipient", "question_sha256", "view_sha256", "full_view_sha256"),
            "original intervention role query source")
        if (uint(query["trial_sequence"], 3) != sequence or type(query["recipient"]) is not str
                or query["recipient"] not in ROUTES or query["recipient"] in recipients):
            raise ValueError("distinct original role queries in trial order required")
        recipients.add(query["recipient"])
        for name in ("question_sha256", "view_sha256", "full_view_sha256"):
            sha(query[name])
    frozen = value["frozen_pc_sources"]
    if type(frozen) is not list or len(frozen) != 2:
        raise ValueError("both original frozen P/C source descriptors required")
    for role, source in zip(("proposer", "critic"), frozen, strict=True):
        fields(source, ("role", "artifact", "model_semantics_sha256", "model_identity_sha256", "encoding_sha256"),
            "original frozen P/C intervention source")
        if source["role"] != role:
            raise ValueError("original proposer/critic source order required")
        _artifact(source["artifact"])
        for name in ("model_semantics_sha256", "model_identity_sha256", "encoding_sha256"):
            sha(source[name])
    ancestors = value["ancestor_datasets"]
    if type(ancestors) is not list or len(ancestors) > 256:
        raise ValueError("bounded original intervention ancestor descriptors required")
    publications = set()
    for descriptor in ancestors:
        fields(descriptor, ("kind", "publication_sha256", "metadata"), "original intervention ancestor")
        kind = descriptor["kind"]
        if kind in ("pc_legacy", "pc_submitted_input_v1", "r_auxiliary"):
            names = ("manifest_file_sha256", "index_file_sha256", "canonical_manifest_sha256", "canonical_index_sha256")
        elif kind in ("utility_cohort", "utility_intervention"):
            names = ("dataset_sha256", "pool_sha256", "preparation_sha256", "context_sha256")
        else:
            raise ValueError("explicit original P/C, R or utility ancestor kind required")
        publication = sha(descriptor["publication_sha256"])
        if publication in publications or publication == value["publication"]["sha256"]:
            raise ValueError("repeated or current intervention ancestor publication")
        publications.add(publication)
        fields(descriptor["metadata"], names, "kind-specific original ancestor seals")
        for seal in descriptor["metadata"].values():
            sha(seal)
    known, all_pairs = _seals(value["known_pair_sha256"]), _seals(value["all_pair_sha256"])
    if not set(known) <= set(all_pairs) or len(all_pairs) > len(queries) * (len(queries) - 1) // 2:
        raise ValueError("original intervention known pair source coverage differs")
    return copy.deepcopy(value)


def _seal(unsigned):
    # Same canonical [domain, body] rule as generation_data.digest. This is
    # deliberately distinct from native domain/NUL/u64-length/raw framing.
    raw = canonical_bytes([PREPARATION_DOMAIN, unsigned])
    if len(raw) > MAX_PREPARATION_BYTES:
        raise ValueError("intervention preparation envelope exceeds 64KiB")
    return hashlib.sha256(raw).hexdigest()


def prepare_intervention_metadata(recipe, registration, parameter_spec, intervention_source):
    """Seal a passive declaration, never re-tag a previously admitted one.

    registration.cost_epoch_sha256 is a first_trial_cost_epoch_anchor for
    this separate format. Other trial epochs remain in their own original
    payloads; this field does not assert a common clock or cost epoch.
    """
    checked_recipe = mathematics.validate_recipe(recipe)
    owner = checked_recipe["optimizer_owner"]
    checked_registration = mathematics._registration(registration, parameter_spec, owner, cost_scope=COST_SCOPE)
    checked_spec = mathematics.validate_parameter_membership(parameter_spec,
        checked_registration["frozen_base"]["config"], owner)
    source = validate_intervention_source(intervention_source)
    unsigned = {"schema": PREPARATION_DOMAIN, "status": "prepared_metadata_only", "recipe": checked_recipe,
        "registration": checked_registration, "parameter_spec": checked_spec, "intervention_source": source,
        "execution_authorized": False, "performed_optimizer_updates": 0}
    result = {**unsigned, "preparation_sha256": _seal(unsigned)}
    if len(canonical_bytes(result)) > MAX_PREPARATION_BYTES:
        raise ValueError("intervention preparation exceeds 64KiB")
    return result


def validate_intervention_preparation(value):
    fields(value, PREPARATION_FIELDS, "intervention no-execution preparation")
    if (value["schema"] != PREPARATION_DOMAIN or value["status"] != "prepared_metadata_only"
            or value["execution_authorized"] is not False
            or type(value["performed_optimizer_updates"]) is not int or value["performed_optimizer_updates"] != 0):
        raise ValueError("separate intervention metadata-only preparation required")
    sha(value["preparation_sha256"])
    checked = prepare_intervention_metadata(value["recipe"], value["registration"],
        value["parameter_spec"], value["intervention_source"])
    if canonical_bytes(value) != canonical_bytes(checked):
        raise ValueError("intervention preparation differs from its exact original codec")
    return checked


def _validate_ancestor_owners(original, ancestor_owners, read_budget):
    """Join already-opened typed PC/R/utility ancestors, without re-admission."""
    from .rovent_utility_route import _ancestor_descriptor, _ancestor_rows
    if type(ancestor_owners) not in (tuple, list) or len(ancestor_owners) > 256:
        raise ValueError("bounded original typed intervention ancestor owners required")
    report = original["report"]
    declarations = {item["publication_sha256"]: item for item in report["ancestor_datasets"]}
    actual, rows = set(), []
    for owner in ancestor_owners:
        read_budget._check()
        from .rovent_utility_intervention_dataset import HistoricalUtilityInterventionDataset
        if type(owner) is HistoricalUtilityInterventionDataset:
            owner._check()
            descriptor = {"kind": "utility_intervention", "publication_sha256": owner.publication_pin["sha256"],
                "metadata": {"dataset_sha256": owner._dataset_sha256, "pool_sha256": owner.pool_sha256,
                    "preparation_sha256": owner._parts.preparation_sha256, "context_sha256": owner.context_ids[0]}}
            nested = owner._original["report"]
            keys = {(name, nested["lineage"][name]) for name in ("opening_id", "game_id", "line_genealogy_id")}
            keys.update((("transposition_sha256", nested["transposition_sha256"]), ("rules_state_sha256", nested["root_rules_sha256"])))
            keys.update(("input_seal", nested[name]) for name in (
                "neutral_input_sha256", "neutral_target_sha256", "neutral_context_sha256"))
            keys.update(("input_seal", query[name]) for query in nested["role_queries"]
                for name in ("question_sha256", "view_sha256", "full_view_sha256"))
            ancestor_rows = ((nested["split"], keys),)
        else:
            descriptor = _ancestor_descriptor(owner)
            ancestor_rows = _ancestor_rows(owner)
        publication = descriptor["publication_sha256"]
        if (publication == original["original_files"].file_pin("utility-cohort-publication.json")["sha256"]
                or publication in actual or descriptor != declarations.get(publication)):
            raise ValueError("intervention original ancestor kind/seals/publication differs")
        actual.add(publication)
        for split, keys in ancestor_rows:
            read_budget._check()
            if split not in ("train", "validation", "holdout") or len(rows) >= 262144:
                raise ValueError("bounded original intervention ancestor split/row inventory required")
            rows.append((split, keys))
    if actual != set(declarations):
        raise ValueError("intervention original ancestor owner coverage incomplete")
    lineage = report["lineage"]
    current = {(name, lineage[name]) for name in ("opening_id", "game_id", "line_genealogy_id")}
    current.add(("transposition_sha256", report["transposition_sha256"]))
    current.add(("rules_state_sha256", report["root_rules_sha256"]))
    current.update(("input_seal", report[name]) for name in (
        "neutral_input_sha256", "neutral_target_sha256", "neutral_context_sha256"))
    current.update(("input_seal", query[name]) for query in report["role_queries"]
        for name in ("question_sha256", "view_sha256", "full_view_sha256"))
    rows.insert(0, (report["split"], current))
    parents, keyed = list(range(len(rows))), {}

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    # Union first, including indirect validation/holdout bridges. New input
    # formats are conservative leak edges, never evidence of equivalence.
    for index, (_, keys) in enumerate(rows):
        read_budget._check()
        for key in keys:
            if key in keyed:
                parents[root(index)] = root(keyed[key])
            else:
                keyed[key] = index
    connected = {split for index, (split, _) in enumerate(rows) if root(index) == root(0)}
    if len(connected) > 1:
        raise ValueError("intervention original PC/R/utility lineage crosses global split")


def prepare_intervention_from_files(reference, recipe, registration, parameter_spec, *, read_budget, ancestor_owners=()):
    """Standalone original /2 file intake; no trainer or optimizer entry.

    The nine-field result is still passive. Saved Rules/conditional operands
    require the Rust source owner's pure proof joins before any later learner
    can be authorized; no historical current ACK is revived here.
    """
    from .rovent_utility_intervention_files import read_intervention_files
    from .rovent_utility_intervention_payload import validate_intervention_payloads
    checked_recipe = mathematics.validate_recipe(recipe)
    checked_registration = mathematics._registration(registration, parameter_spec,
        checked_recipe["optimizer_owner"], cost_scope=COST_SCOPE)
    mathematics.validate_parameter_membership(parameter_spec, checked_registration["frozen_base"]["config"],
        checked_recipe["optimizer_owner"])
    files = read_intervention_files(reference, read_accounting=read_budget)
    read_budget._check()
    original = validate_intervention_payloads(files, registration=checked_registration)
    report = original["report"]
    _validate_ancestor_owners(original, ancestor_owners, read_budget)
    source = {name: copy.deepcopy(report[name]) for name in ("cohort_id", "profile_sha256", "neutral_input_sha256",
        "neutral_target_sha256", "neutral_context_sha256", "root_rules_sha256", "role_queries", "frozen_pc_sources",
        "ancestor_datasets", "known_pair_sha256", "all_pair_sha256")}
    for name, filename in (("publication", "utility-cohort-publication.json"), ("durable_ack", "utility-cohort-durable-ack.json")):
        pin = files.file_pin(filename)
        source[name] = {key: pin[key] for key in ("sha256", "bytes")}
    result = prepare_intervention_metadata(checked_recipe, checked_registration, parameter_spec, source)
    read_budget._check()
    return result
