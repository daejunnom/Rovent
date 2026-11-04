"""Explicit /2 learner consumers for retained historical publications.

The existing tensor/scheduler/optimizer and bounded learner algorithms are
reused after separate codec validation. Historical files never become an old
native run or physical lease. A fresh kernel-authenticated parent must join
the scratch projection, full update chain and current OS reclamation evidence.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from .generation_data import digest, fields, sha, uint
from .generation_training import _typed_equal
from .rovent_training import membership_digest, validate_training_membership
from .rovent_training_window_v2 import validate_window_spec, training_profile_digest
from .rovent_training_recovery import artifact_reference, initial_recovered_cost, recorded_tail_cost
from . import rovent_training_window as core
from .rovent_tensor_training import (MAX_BYTES, ROLES, _RNG_LOCK, _runtime,
    _load_pinned_tensor_file, _restore_scratch, tensor_payload, stage_tensor_fork, _nnuec_tensor_training_state)
from .rovent_nnuec_training_state import nnuec_is_unmodified, nnuec_known_updates

LearnedExportRequest = core.LearnedExportRequest
PUBLICATION_DOMAIN = "rz-rovent-training-checkpoint-publication/1"
_PUBLICATION_FIELDS = ("schema", "execution_id", "window_id", "collection_source_sha256",
    "collection_runtime_library_sha256", "trainer_source_sha256", "trainer_python_executable_sha256",
    "trainer_dependency_set_sha256", "training_model_semantics_sha256", "initialization_source",
    "collection_model_epoch", "collection_context_epoch", "training_model_epoch", "recipe_sha256", "stage_id", "training_profile_sha256",
    "membership_sha256", "current_dataset_publication_sha256", "ancestor_dataset_publication_sha256", "tensor",
    "checkpoint_retirement", "update_receipts", "terminal_receipt", "original_recipe", "original_membership",
    "original_window_spec", "parent_publication", "boundary", "recorded_tail_cost", "recorded_closure")
_ACK_FIELDS = ("schema", "source_kind", "past_native_authority_reissued", "past_physical_completion_reissued",
    "historical_physical_scope", "current_execution_id", "current_window_id", "checkpoint_publication_sha256",
    "checkpoint_sha256", "checkpoint_retirement_sha256", "original_update_chain_sha256", "closure_sha256",
    "new_os_reclamation", "recipe_sha256", "stage_id", "training_profile_sha256", "completed_updates",
    "retained_input_set_sha256", "retained_input_count", "retained_origin_set_sha256", "retained_origin_count",
    "historical_tail_cost_records", "current_peer_tensor_projection", "ancestor_updates", "snapshot_cost",
    "recorded_tail_cost", "inherited_unknown_tails")
_EXPORT_FIELDS = ("pc_training_steps", "collection_source_sha256", "collection_runtime_library_sha256",
    "trainer_source_sha256", "trainer_python_executable_sha256", "trainer_dependency_set_sha256",
    "training_model_semantics_sha256", "lineage_teacher_actor_ids", "nnuec_training")


@dataclass(frozen=True)
class HistoricalCheckpointInput:
    """Pinned operands only; neither construction nor attachment grants a lease."""
    tensor_pin: dict
    publication: dict
    publication_pin: dict
    retirement_raw: bytes
    retirement_pin: dict
    source_window: dict
    source_intake: object
    read_budget: object
    snapshot_directory: str
    source_r_auxiliary_intake: object = None


def validate_checkpoint_publication_metadata(value, *, original_window=None):
    """Check the actual publication DTO without IO, tensor loads or a lease."""
    fields(value, _PUBLICATION_FIELDS + (("r_auxiliary",) if type(value) is dict and "r_auxiliary" in value else ()), "historical checkpoint publication")
    if value["schema"] != PUBLICATION_DOMAIN:
        raise ValueError("explicit historical checkpoint publication domain required")
    for name in ("collection_model_epoch", "collection_context_epoch", "training_model_epoch"):
        uint(value[name])
    auxiliary_publication = value.get("r_auxiliary")
    if auxiliary_publication is not None:
        fields(auxiliary_publication, ("schema", "dataset_publication_sha256", "manifest_file_sha256", "index_file_sha256",
            "parent_ledger_projection"), "checkpoint's separate R auxiliary source")
        if auxiliary_publication["schema"] != "rz-rovent-checkpoint-r-auxiliary/1":
            raise ValueError("explicit checkpoint R source sidecar required")
        for name in ("dataset_publication_sha256", "manifest_file_sha256", "index_file_sha256"):
            sha(auxiliary_publication[name])
        artifact_reference(auxiliary_publication["parent_ledger_projection"])
    if original_window is not None:
        if type(original_window) is not dict:
            raise ValueError("original pinned training window metadata required")
        domain = original_window.get("schema")
        from .rovent_r_auxiliary import enabled
        if enabled(original_window):
            if (auxiliary_publication is None or any(auxiliary_publication[name] != original_window["nnuec_r_auxiliary"][name]
                    for name in ("manifest_file_sha256", "index_file_sha256"))):
                raise ValueError("original enabled window lost its exact R auxiliary publication")
        elif auxiliary_publication is not None:
            raise ValueError("disabled original window cannot acquire an R publication")
        if domain == "rz-rovent-train-window/2":
            names = ("collection_source_sha256", "collection_runtime_library_sha256", "collection_model_epoch",
                "collection_context_epoch", "training_model_epoch", "trainer_source_sha256",
                "trainer_python_executable_sha256", "trainer_dependency_set_sha256", "training_model_semantics_sha256")
            if any(name not in original_window or not _typed_equal(original_window[name], value[name]) for name in names):
                raise ValueError("historical publication changed original separate collection/training identity or epoch")
        elif domain == "rz-rovent-train-window/1":
            # The old unseparated fields are retained as old evidence. They
            # cannot identify separate Python and collection runtimes by alias.
            if original_window.get("model_semantics_sha256") != value["training_model_semantics_sha256"]:
                raise ValueError("legacy historical model semantics changed")
        else:
            raise ValueError("historical original window domain unsupported")
        if (original_window.get("window_id") != value["window_id"]
                or original_window.get("stage_id") != value["stage_id"]
                or original_window.get("recipe_sha256") != value["recipe_sha256"]
                or core.training_profile_digest(original_window) != value["training_profile_sha256"]):
            raise ValueError("historical original window/profile/stage publication mismatch")
    return value


def _relative_pin(root, value):
    pin = artifact_reference(value)
    root = Path(root)
    if not root.is_absolute() or root.resolve(strict=True) != root or root.is_symlink():
        raise ValueError("explicit existing unlinked historical checkpoint root required")
    path = root.joinpath(*pin["path"].split("/"))
    return {"path": str(path), "sha256": pin["sha256"], "bytes": pin["bytes"]}


def read_historical_checkpoint_source(startup, reads, *, corpus_by_publication, r_corpus_by_publication=None):
    """Read exact registered metadata; no sibling discovery or tensor loading."""
    registration = startup["registration"]
    initialization = registration["initialization_source"]
    if initialization["kind"] not in ("resume", "fork"):
        raise ValueError("historical checkpoint input requires explicit resume or fork")
    source = initialization["checkpoint"]
    publication = reads.json(source["publication"], maximum=16 << 20)
    validate_checkpoint_publication_metadata(publication)
    observation = startup["checkpoint_admission"]
    if (publication["tensor"]["sha256"] != observation["tensor_sha256"]
            or source["publication"]["sha256"] != observation["publication_sha256"]):
        raise ValueError("historical metadata differs from the current parent's retained checkpoint")
    tensor = _relative_pin(source["root"], publication["tensor"])
    if tensor["bytes"] > MAX_BYTES:
        raise ValueError("historical tensor exceeds explicit allocation/read ceiling")
    retirement = _relative_pin(source["root"], publication["checkpoint_retirement"])
    raw, _identity = reads.read(retirement, maximum=1024**2)
    original_recipe = reads.json(_relative_pin(source["root"], publication["original_recipe"]), maximum=1024**2)
    original_membership = reads.json(_relative_pin(source["root"], publication["original_membership"]), maximum=16 << 20)
    original_window = reads.json(_relative_pin(source["root"], publication["original_window_spec"]), maximum=1024**2)
    from .rovent_generation import SeriesRecipe
    original = SeriesRecipe(original_recipe)
    if (original.sha256 != publication["recipe_sha256"]
            or membership_digest(original_membership) != publication["membership_sha256"]):
        raise ValueError("historical recipe/membership exact bytes changed canonical identity")
    core.validate_window_spec(original, original_window)
    validate_checkpoint_publication_metadata(publication, original_window=original_window)
    if (core.training_profile_digest(original_window) != publication["training_profile_sha256"]
            or original_window["stage_id"] != publication["stage_id"]):
        raise ValueError("historical window changed its original profile/stage")
    if type(corpus_by_publication) is not dict or len(corpus_by_publication) > 257:
        raise ValueError("bounded already-validated historical corpus graph required")
    intake = corpus_by_publication.get(publication["current_dataset_publication_sha256"])
    from .rovent_corpus import RoventTrainingCorpus
    if type(intake) is not RoventTrainingCorpus:
        raise ValueError("historical checkpoint's original corpus was not retained")
    source_r = None
    from . import rovent_r_auxiliary as auxiliary
    if auxiliary.enabled(original_window):
        if type(r_corpus_by_publication) is not dict or len(r_corpus_by_publication) > 257:
            raise ValueError("historical auxiliary source requires its complete separate original R publication map")
        registered_r = original_window["nnuec_r_auxiliary"]
        original_r = publication["r_auxiliary"]
        matches = [owner for identity, owner in r_corpus_by_publication.items()
            if identity == original_r["dataset_publication_sha256"] and owner.corpus_sha256 == registered_r["manifest_file_sha256"]
                and owner.index_sha256 == registered_r["index_file_sha256"]]
        if len(matches) != 1:
            raise ValueError("original checkpoint R source is missing, duplicated or replaced by the current dataset")
        source_r = matches[0]
        # The retained parent ledger's raw file is part of the shared read
        # budget. Its historical JSON does not reissue a current capability.
        parent_projection = reads.json(_relative_pin(source["root"], original_r["parent_ledger_projection"]), maximum=16 << 20)
        if type(parent_projection) is not dict:
            raise ValueError("original R auxiliary parent ledger metadata must be retained")
    return HistoricalCheckpointInput(tensor, copy.deepcopy(publication), copy.deepcopy(source["publication"]),
        raw, retirement, copy.deepcopy(original_window), intake, reads, str(Path(registration["output_root"]) / "tmp"), source_r)


def _authority(value):
    from .rovent_recovered_training_authority import RecoveredParentSocketTrainerAuthority
    if not isinstance(value, RecoveredParentSocketTrainerAuthority):
        raise core.OriginalTrainingAuthorityRequired("a new kernel-authenticated recovered parent is required")
    return value


def _source(authority):
    source = getattr(_authority(authority), "_historical_source", None)
    if type(source) is not HistoricalCheckpointInput:
        raise core.OriginalTrainingAuthorityRequired("original historical checkpoint operands are not attached")
    return source


def _read_admission(owner, intake, spec, *, r_auxiliary_intake=None):
    validate_window_spec(owner.recipe, spec)
    return core._read_admission(owner, intake, spec)


def _journal(owner, spec, admission, *, r_auxiliary_intake=None):
    validate_window_spec(owner.recipe, spec)
    return core._journal(owner, spec, admission, r_auxiliary_intake=r_auxiliary_intake)


def initial_training_journal(recipe, spec, admission, *, r_auxiliary_intake=None):
    validate_window_spec(recipe, spec)
    return core.initial_training_journal(recipe, spec, admission, r_auxiliary_intake=r_auxiliary_intake)


def _checked_admission(owner, value, authority, *, exported=False):
    fields(value, _ACK_FIELDS + (_EXPORT_FIELDS if exported else ())
        + (("r_auxiliary",) if "r_auxiliary" in owner.samplers else ()), "new-process historical checkpoint admission")
    if "r_auxiliary" in owner.samplers:
        from .rovent_r_auxiliary import checkpoint_projection
        expected_aux = checkpoint_projection(owner.samplers["r_auxiliary"], owner.samplers["updates"])
        if not _typed_equal(value["r_auxiliary"], expected_aux):
            raise ValueError("current parent changed original R auxiliary source/count/seal checkpoint state")
    source = _source(authority)
    inputs = core.retained_seal_set(owner.inherited_input_sha256, "input")
    origins = core.retained_seal_set(owner.inherited_origin_sha256, "origin")
    counts = ({role: uint(owner.completed[role] + owner.ancestor_updates[role]) for role in ROLES}
              if exported else dict(owner.completed))
    expected = {"schema": "rz-rovent-recovered-learned-export-admission/1" if exported
            else "rz-rovent-recovered-checkpoint-admission/1", "source_kind": "historical_exact_pins",
        "past_native_authority_reissued": False, "past_physical_completion_reissued": False,
        "historical_physical_scope": "checkpoint_does_not_prove_old_run_closure",
        "current_execution_id": authority._startup["execution_id"], "current_window_id": authority._startup["window_id"],
        "checkpoint_publication_sha256": source.publication_pin["sha256"], "checkpoint_sha256": source.tensor_pin["sha256"],
        "checkpoint_retirement_sha256": source.retirement_pin["sha256"], "recipe_sha256": owner.recipe.sha256,
        "stage_id": owner.stage_id, "training_profile_sha256": owner.samplers["profile_sha256"],
        "completed_updates": counts, "ancestor_updates": dict(owner.ancestor_updates), "snapshot_cost": dict(owner.cost),
        "retained_input_set_sha256": inputs["sha256"], "retained_input_count": inputs["count"],
        "retained_origin_set_sha256": origins["sha256"], "retained_origin_count": origins["count"]}
    if any(not _typed_equal(value[name], item) for name, item in expected.items()):
        raise ValueError("new process admission changed historical scratch/update/input ancestry")
    for name in ("original_update_chain_sha256", "closure_sha256"):
        sha(value[name])
    for name in ("new_os_reclamation", "current_peer_tensor_projection"):
        if type(value[name]) is not dict or not value[name]:
            raise ValueError("current original parent must retain its separate reclamation/projection evidence")
        core._validate_json_state(value[name])
    tail = recorded_tail_cost(value["recorded_tail_cost"])
    if not _typed_equal(tail, source.publication["recorded_tail_cost"]):
        raise ValueError("current parent changed original recorded tail uncertainty")
    records = value["historical_tail_cost_records"]
    if type(records) is not list or not 1 <= len(records) <= 64:
        raise ValueError("bounded complete original publication tail chain required")
    seen, expected_unknowns = set(), []
    for row in records:
        fields(row, ("checkpoint_publication_sha256", "execution_id", "window_id", "tail"), "historical publication tail")
        sha(row["checkpoint_publication_sha256"])
        core.identifier(row["execution_id"])
        core.identifier(row["window_id"])
        identity = (row["execution_id"], row["window_id"])
        if identity in seen:
            raise ValueError("historical chain repeats a cost execution/window")
        seen.add(identity)
        recorded_tail_cost(row["tail"])
        if row["checkpoint_publication_sha256"] != source.publication_pin["sha256"] and row["tail"]["status"] == "unknown":
            expected_unknowns.append(row)
    selected = records[-1]
    if (selected["checkpoint_publication_sha256"] != source.publication_pin["sha256"]
            or selected["execution_id"] != source.publication["execution_id"]
            or selected["window_id"] != source.publication["window_id"] or not _typed_equal(selected["tail"], tail)
            or not _typed_equal(value["inherited_unknown_tails"], expected_unknowns)):
        raise ValueError("historical cost ancestry omitted or relabelled selected/unknown tail")
    # This validates shape/uncertainty only; complete chain authority remains
    # the freshly connected parent's exact publication/OS owner.
    initial_recovered_cost(owner.cost, tail, inherited_unknown_tails=value["inherited_unknown_tails"])
    return copy.deepcopy(value)


def _sampler_validator(owner, intake, spec, *, r_auxiliary_intake=None):
    admission = core._read_admission(owner, intake, spec)
    def validate(value):
        probe = copy.copy(owner)
        probe.samplers = value
        journal = core._journal(probe, spec, admission, r_auxiliary_intake=r_auxiliary_intake)
        dispatch = None
        if r_auxiliary_intake is not None:
            dispatch, metadata = core.joint_r_auxiliary_dispatch(probe, intake, spec, journal, r_auxiliary_intake)
        for role, state in journal["phase_inputs"][spec["phase"]]["samplers"].items():
            rows = (dispatch.pc_rows(role, split="train", optimization=True) if dispatch is not None
                else intake.rows(role, split="train", optimization=True))
            pool = digest("rz-rovent-native-role-pool/1", {"corpus_sha256": admission["corpus_sha256"], "role": role,
                "inputs": [(sample["input_sha256"], sample["origin_sha256"]) for sample in
                           (intake.training_sample(row) for row in rows)]})
            core.StageSampler(rows, role=role, seed=spec["sampling_seed"], pool_sha256=pool, state=state)
        if r_auxiliary_intake is not None and spec["phase"] == "pc":
            from .rovent_r_auxiliary import RAuxiliarySampler
            sidecar = journal["r_auxiliary"]
            if sidecar["sampler"] is not None:
                RAuxiliarySampler(tuple(dispatch.r_rows(split="train", optimization=True)),
                    seed=spec["nnuec_r_auxiliary"]["sampling_seed"], pool_sha256=metadata["pool_sha256"], state=sidecar["sampler"])
    return validate


def _raw_scratch(path, expected_sha256, template, recipe, membership, authority, maximum):
    validate_training_membership(membership)
    source = _source(authority)
    if str(Path(path)) != source.tensor_pin["path"] or expected_sha256 != source.tensor_pin["sha256"]:
        raise ValueError("only the exact registered recovered tensor input may be loaded")
    runtime = _runtime(True)
    payload, size = _load_pinned_tensor_file(path, expected_sha256, maximum, runtime[0], read_accounting=source.read_budget,
        snapshot_directory=source.snapshot_directory)
    if size != source.tensor_pin["bytes"]:
        raise ValueError("recovered tensor byte length differs from the publication")
    if type(payload) is not dict or not {"ledger", "samplers", "stage_id", "inherited_input_sha256", "inherited_origin_sha256"} <= payload.keys():
        raise ValueError("full historical tensor checkpoint state required")
    probe = SimpleNamespace(recipe=recipe, stage_id=payload["stage_id"], completed=payload["ledger"]["completed"],
        samplers=payload["samplers"], inherited_input_sha256=payload["inherited_input_sha256"],
        inherited_origin_sha256=payload["inherited_origin_sha256"])
    validator = _sampler_validator(probe, source.source_intake, source.source_window,
        r_auxiliary_intake=source.source_r_auxiliary_intake)
    owner = _restore_scratch(payload, template, recipe, membership, sampler_validator=validator, runtime=runtime)
    owner.source_checkpoint_sha256, owner.source_checkpoint_path = expected_sha256, str(path)
    owner.source_model_state_sha256 = runtime[5](payload["model"])
    owner.source_nnuec_metadata_sha256 = digest("rz-rovent-original-nnuec-metadata/1", payload["nnuec_metadata"])
    owner.source_payload_state_sha256 = runtime[5](payload)
    return owner, payload, validator, runtime


def _stage_recovered_checkpoint(path, expected_sha256, template, recipe, membership, native_intake, window_spec, *,
                                execution_authority=None, max_file_bytes, purpose, execute=False, r_auxiliary_intake=None):
    spec = copy.deepcopy(validate_window_spec(recipe, window_spec))
    if purpose not in ("resume", "export"):
        raise ValueError("explicit recovered resume or raw export purpose required")
    sha(expected_sha256)
    maximum = uint(max_file_bytes, MAX_BYTES)
    if not maximum:
        raise ValueError("positive original tensor read/allocation credit required")
    if execute is False:
        return {"schema": "rz-rovent-recovered-learner-" + purpose + "-preparation/1",
            "recipe_sha256": recipe.sha256, "stage_id": spec["stage_id"], "checkpoint_sha256": expected_sha256,
            "actual_tensor_bytes_loaded": False, "caller_state_mutated": False, "execution_authorized": False}
    if execute is not True:
        raise ValueError("exact bool recovered execution selector required")
    authority = _authority(execution_authority)
    source = _source(authority)
    if source.source_window["schema"] != "rz-rovent-train-window/2":
        raise ValueError("exact resume/export of an unseparated /1 window needs a separate explicit migration policy")
    owner, payload, _source_validator, runtime = _raw_scratch(path, expected_sha256, template, recipe, membership, authority, maximum)
    if owner.stage_id != spec["stage_id"]:
        raise ValueError("recovered resume cannot change stage or teacher branch")
    original = _checked_admission(owner, authority.validate_checkpoint_state(owner, native_intake,
        {"sha256": expected_sha256, "bytes": source.tensor_pin["bytes"]}), authority)
    if purpose == "resume":
        cost_state = initial_recovered_cost(owner.cost, original["recorded_tail_cost"],
            inherited_unknown_tails=original["inherited_unknown_tails"])
        if original["recorded_tail_cost"]["status"] == "known":
            delta = original["recorded_tail_cost"]["delta"]
            for name, count in delta.items():
                owner.cost[name] = uint(owner.cost[name] + count)
            owner.samplers["resume_cost_receipts"].append({"checkpoint_sha256": expected_sha256,
                "original_receipt_sha256": original["checkpoint_retirement_sha256"], "cost_delta": copy.deepcopy(delta)})
        owner.samplers["recovery_cost_state"] = cost_state
        if spec["phase"] == "v" and "v" not in owner.samplers["phase_inputs"]:
            owner.samplers = core.prepare_frozen_v_journal(owner, native_intake, spec)
    validator = _sampler_validator(owner, native_intake, spec, r_auxiliary_intake=r_auxiliary_intake)
    current = tensor_payload(owner, sampler_validator=validator, execute=True)
    return core.StagedLearnerResume(owner, validator, original, runtime[5](current))


def stage_rovent_resume(path, expected_sha256, template, recipe, membership, native_intake, window_spec, **kwargs):
    return _stage_recovered_checkpoint(path, expected_sha256, template, recipe, membership, native_intake, window_spec,
        purpose="resume", **kwargs)


def stage_rovent_export(path, expected_sha256, template, recipe, membership, native_intake, window_spec, **kwargs):
    return _stage_recovered_checkpoint(path, expected_sha256, template, recipe, membership, native_intake, window_spec,
        purpose="export", **kwargs)


def stage_recovered_rovent_fork(path, expected_sha256, template, recipe, membership, native_intake, window_spec, *,
                                execution_authority=None, max_file_bytes, execute=False, r_auxiliary_intake=None):
    spec = copy.deepcopy(validate_window_spec(recipe, window_spec))
    target = recipe.stage(spec["stage_id"])
    if target.parent_stage_id is None:
        raise ValueError("common initialization cannot be represented as a recovered fork")
    sha(expected_sha256)
    maximum = uint(max_file_bytes, MAX_BYTES)
    if not maximum:
        raise ValueError("positive original parent tensor byte credit required")
    if execute is False:
        return {"schema": "rz-rovent-recovered-learner-fork-preparation/1", "recipe_sha256": recipe.sha256,
            "source_stage_id": target.parent_stage_id, "stage_id": target.stage_id, "checkpoint_sha256": expected_sha256,
            "reset_optimizer_state": True, "actual_tensor_bytes_loaded": False, "execution_authorized": False}
    if execute is not True:
        raise ValueError("exact bool recovered fork execution selector required")
    authority = _authority(execution_authority)
    source = _source(authority)
    parent, payload, source_validator, runtime = _raw_scratch(path, expected_sha256, template, recipe, membership, authority, maximum)
    if (parent.stage_id != target.parent_stage_id
            or parent.completed != dict(zip(ROLES, recipe.stage(parent.stage_id).role_updates))):
        raise ValueError("a recovered fork requires its declared completed parent stage")
    original = _checked_admission(parent, authority.validate_checkpoint_state(parent, native_intake,
        {"sha256": expected_sha256, "bytes": source.tensor_pin["bytes"]}), authority)
    probe = SimpleNamespace(recipe=recipe, stage_id=target.stage_id, completed=dict.fromkeys(ROLES, 0),
        inherited_input_sha256=list(parent.inherited_input_sha256), inherited_origin_sha256=list(parent.inherited_origin_sha256))
    admission = _read_admission(probe, native_intake, spec)
    journal = initial_training_journal(recipe, spec, admission, r_auxiliary_intake=r_auxiliary_intake)
    from . import rovent_r_auxiliary as auxiliary
    from .rovent_r_ancestry import ancestry_from_journal
    ancestry = ancestry_from_journal(parent.samplers)
    if auxiliary.enabled(spec):
        if "r_auxiliary" in parent.samplers:
            journal["r_auxiliary"] = auxiliary.fork_auxiliary_journal(parent.samplers["r_auxiliary"], spec["nnuec_r_auxiliary"])
        tracked = nnuec_known_updates(_nnuec_tensor_training_state(parent))
        if tracked is None:
            raise ValueError("unverified NNUEC ancestry cannot become a known auxiliary fork")
        # All NNUEC tensor steps and R-only source seals have distinct owners.
        # A disabled middle stage may add valid old-PC NNUEC updates without
        # creating any new R source occurrence.
        journal["r_auxiliary"]["ancestor_nnuec_updates"] = tracked
        journal["r_auxiliary"]["ancestor_input_sha256"] = ancestry["input_sha256"]
        journal["r_auxiliary"]["ancestor_origin_sha256"] = ancestry["origin_sha256"]
        core.bind_r_auxiliary(recipe, spec, journal["r_auxiliary"], r_auxiliary_intake)
    elif ancestry["input_sha256"] or ancestry["origin_sha256"]:
        journal["r_auxiliary_ancestry"] = ancestry
    journal["recovery_cost_state"] = initial_recovered_cost(parent.cost, original["recorded_tail_cost"],
        inherited_unknown_tails=original["inherited_unknown_tails"])
    probe.samplers = journal
    validator = _sampler_validator(probe, native_intake, spec, r_auxiliary_intake=r_auxiliary_intake)
    child = stage_tensor_fork(path, expected_sha256, template, recipe, membership,
        source_sampler_validator=source_validator, target_sampler_validator=validator, target_stage_id=target.stage_id,
        reset_seed=spec["training_seed"], reset_samplers=journal, reset_bucket=8, max_file_bytes=maximum,
        read_accounting=source.read_budget, snapshot_directory=source.snapshot_directory, execute=True)
    if original["recorded_tail_cost"]["status"] == "known":
        for name, count in original["recorded_tail_cost"]["delta"].items():
            child.ancestor_cost[name] = uint(child.ancestor_cost[name] + count)
    final_validator = _sampler_validator(child, native_intake, spec, r_auxiliary_intake=r_auxiliary_intake)
    current = tensor_payload(child, sampler_validator=final_validator, execute=True)
    return core.StagedLearnerResume(child, final_validator, original, runtime[5](current))


stage_rovent_fork = stage_recovered_rovent_fork


def train_rovent_window(owner, native_intake, window_spec, checkpoint_directory, *, execution_authority=None,
                        execute=False, **kwargs):
    validate_window_spec(owner.recipe, window_spec)
    if execute is True:
        _authority(execution_authority)
    return core.train_rovent_window(owner, native_intake, window_spec, checkpoint_directory,
        execution_authority=execution_authority, execute=execute, **kwargs)


train_recovered_rovent_window = train_rovent_window


def admit_recovered_learned_export(owner, checkpoint, request):
    """Derive writer declarations from the current peer's exact historical ack.

    The raw parent acknowledgement is retained without re-tagging or editing.
    Source NNUEC training metadata remains immutable; the declaration below is
    based on current tensor-derived state and the original update chain.
    """
    if type(request) is not LearnedExportRequest:
        raise core.OriginalTrainingAuthorityRequired("an explicit recovered learned-export request is required")
    authority = _authority(request.execution_authority)
    spec = validate_window_spec(owner.recipe, request.window_spec)
    admission = _read_admission(owner, request.native_intake, spec)
    journal = _journal(owner, spec, admission, r_auxiliary_intake=request.r_auxiliary_intake)
    stage_limits = dict(zip(ROLES, owner.recipe.stage(owner.stage_id).role_updates))
    if owner.completed != stage_limits or owner.pending_microbatches:
        raise ValueError("recovered learned export requires all declared current-stage update boundaries")
    original = _checked_admission(owner, authority.validate_learned_export(owner, request.native_intake, checkpoint),
        authority, exported=True)
    expected = {name: spec[name] for name in ("collection_source_sha256", "collection_runtime_library_sha256",
        "trainer_source_sha256", "trainer_python_executable_sha256", "trainer_dependency_set_sha256",
        "training_model_semantics_sha256")}
    expected["lineage_teacher_actor_ids"] = list(owner.recipe.inherited_teachers(owner.stage_id))
    totals = {role: uint(owner.completed[role] + owner.ancestor_updates[role]) for role in ROLES}
    expected["pc_training_steps"] = totals["proposer"] + totals["critic"]
    if not any(totals.values()) or any(not _typed_equal(original[name], item) for name, item in expected.items()):
        raise ValueError("recovered learned export changed explicit source/runtime/model or actual teacher ancestry")
    lineage = original["nnuec_training"]
    fields(lineage, ("schema", "completed_nnuec_updates", "dataset_manifest_sha256", "last_execution_id",
        "last_window_id", "historical_metadata_only", "past_native_authority_reissued", "execution_authorized"),
        "historical NNUEC update lineage")
    if (lineage["schema"] != "rz-rovent-historical-learned-nnuec-lineage/1"
            or lineage["historical_metadata_only"] is not True or lineage["past_native_authority_reissued"] is not False
            or lineage["execution_authorized"] is not False):
        raise ValueError("historical NNUEC lineage cannot issue execution or native authority")
    steps = uint(lineage["completed_nnuec_updates"], expected["pc_training_steps"])
    datasets = lineage["dataset_manifest_sha256"]
    if (type(datasets) is not list or len(datasets) > 256 or datasets != sorted(set(datasets))):
        raise ValueError("sorted unique historical NNUEC corpus set required")
    for value in datasets:
        sha(value)
    derived = _nnuec_tensor_training_state(owner)
    if steps == 0:
        if (datasets or lineage["last_execution_id"] is not None or lineage["last_window_id"] is not None
                or not nnuec_is_unmodified(derived) or derived["source_declaration"] != {"kind": "untrained"}
                or (journal["r_auxiliary"]["current_nnuec_updates"] if "r_auxiliary" in journal
                    else journal["head_counts"]["nnuec_wdl"])
                or any(count for name, count in owner.parameter_updates.items() if name.startswith("nnuec."))):
            raise ValueError("learned or modified NNUEC tensors cannot be relabelled as untrained")
        declaration = {"kind": "untrained"}
    else:
        if not datasets or nnuec_known_updates(derived) != steps:
            raise ValueError("actual NNUEC tensor update count differs from the historical source chain")
        core.identifier(lineage["last_execution_id"])
        core.identifier(lineage["last_window_id"])
        dataset = datasets[0] if len(datasets) == 1 else digest("rz-rovent-nnuec-training-corpus-set/1", datasets)
        declaration = {"kind": "learned", "run_id": lineage["last_execution_id"], "steps": steps, "dataset_sha256": dataset}
    return {"schema": "rz-rovent-recovered-learned-export-proof/1", "source_kind": "historical_exact_pins",
        "original_parent_admission": original, "completed_updates": totals, "pc_training_steps": expected["pc_training_steps"],
        "nnuec_training": declaration, "historical_nnuec_lineage": copy.deepcopy(lineage),
        "cost_state": initial_recovered_cost(owner.cost, original["recorded_tail_cost"],
            inherited_unknown_tails=original["inherited_unknown_tails"]),
        "past_native_authority_reissued": False, "past_physical_completion_reissued": False,
        "scope": "current_peer_checked_historical_updates_not_numeric_gpu_or_arena_acceptance"}
