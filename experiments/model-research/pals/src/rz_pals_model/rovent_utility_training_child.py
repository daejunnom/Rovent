"""Isolated utility learner with explicitly registered historical source owners.

Each PC/R/utility ancestor retains its own exact codec and original recipe or
preparation. Root-local observations and old P/C/V publications never become
utility targets. OS/runtime/terminal primitives are explicitly shared; current
startup, preparation, window, authority and result boundaries remain separate.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import sys
import time


RESULT_DOMAIN = "rz-rovent-utility-trainer-result/1"
MAX_STARTUP = 16 * 1024**2


class SourceIntegrationPending(ValueError):
    """An original utility cohort/checkpoint owner has not been connected yet."""


def _parser():
    parser = argparse.ArgumentParser(description="Original registered utility-only learner child")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--startup-sha256", required=True)
    parser.add_argument("--startup-bytes", type=int, required=True)
    parser.add_argument("--execution-until-ns", type=int, required=True)
    parser.add_argument("--whole-until-ns", type=int, required=True)
    return parser


def _read_startup(stream, count):
    if type(count) is not int or not 0 < count <= MAX_STARTUP:
        raise ValueError("bounded original utility startup byte count required")
    raw = stream.read(count + 1)
    if type(raw) is not bytes or len(raw) != count:
        raise ValueError("original utility startup EOF/bytes differ from pinned stdin")
    return raw


def _checked_family(value):
    from .rovent_utility_training_family import LEGACY, require_family
    return require_family(LEGACY if value is None else value)


def _actual_registration(startup, *, family=None):
    family = _checked_family(family)
    registration = startup["registration"]
    expected = "rz-rovent-utility-training-registration/3" if family.intervention else "rz-rovent-utility-training-registration/2"
    if (registration["schema"] != expected
            or family.intervention and registration.get("data_family") != "intervention_v2"):
        raise ValueError("actual utility child requires the separate mixed-ancestry registration/2")
    return registration


def _metadata_inputs(startup, reads, *, family=None):
    family = _checked_family(family)
    registration = _actual_registration(startup, family=family)
    preparation = reads.json(registration["preparation"], maximum=65536)
    if (not family.intervention and (type(preparation) is not dict
            or preparation.get("schema") != "rz-rovent-utility-training-preparation/2")):
        raise ValueError("actual cohort utility child requires the separate preparation/2 codec")
    checked = family.parts(preparation).preparation
    if json.dumps(checked, sort_keys=True, separators=(",", ":"), allow_nan=False) != json.dumps(preparation, sort_keys=True, separators=(",", ":"), allow_nan=False):
        raise ValueError("utility preparation changed from its exact source codec")
    spec = family.validate_window(checked, reads.json(registration["window_spec"], maximum=1024**2))
    if (spec["window_id"] != startup["window_id"]
            or checked["registration"]["trainer_source_sha256"] != registration["trainer_source_sha256"]
            or spec["device"] != registration["resources_policy"]["device"]
            or spec["max_read_bytes"] != registration["max_child_read_bytes"]
            or spec["max_output_bytes"] > registration["max_output_bytes"]):
        raise ValueError("utility source/window/runtime resource declaration differs from original startup")
    preserved = [reads.verify(source["artifact"], maximum=2 * 1024**3) for source in registration["frozen_pc_sources"]]
    return checked, spec, {"scope": "not_instantiated_pinned_original_source_preserved", "files": preserved,
                           "model_or_tensor_instantiated": False, "actual_tensor_unchanged_observed": False}


def _load_actual_cohort(startup, reads, preparation, *, family=None):
    from .generation_data import sha
    from .rovent_generation import SeriesRecipe
    from . import rovent_historical_corpus as pc
    from . import rovent_historical_r_corpus as r
    from .rovent_utility_route import HistoricalUtilityRouteDataset, read_historical_utility_dependencies
    from .rovent_utility_training import validate_training_preparation
    from .rovent_corpus_codec import LEGACY_CODEC, SUBMITTED_CODEC, codec_from_publication
    family = _checked_family(family)
    registration = startup["registration"]
    entries, metadata, by_index = {}, {}, {}
    for entry in registration["ancestor_datasets"]:
        source, kind = entry["registration"], entry["kind"]
        identity = sha(source["publication"]["sha256"])
        if identity in entries:
            raise ValueError("repeated original utility ancestor publication across families")
        entries[identity] = entry
        if kind in ("pc_legacy", "pc_submitted_input_v1"):
            root = pc._registration(source)
            body = pc._read_file(root, source["publication"], reads, relative=False, json_metadata=True)
            codec = codec_from_publication(body)
            if codec is not (LEGACY_CODEC if kind == "pc_legacy" else SUBMITTED_CODEC):
                raise ValueError("declared original PC ancestor kind differs from its actual publication family")
            publication = pc._publication(body, pc.original.CorpusLimits()) if codec is LEGACY_CODEC else pc._submitted_publication(body, pc.original.CorpusLimits())
            manifest = pc._read_file(root, publication["manifest"], reads, json_metadata=True)
            dependencies = pc.original._sha_list(manifest["ancestor_index_sha256"], 256, "PC original ancestor indices")
            metadata[identity] = {"index": publication["canonical_index_sha256"], "pc": tuple(dependencies), "r": (), "publication": ()}
        elif kind == "r_auxiliary":
            graph = r.read_historical_r_dependencies(source, read_budget=reads)
            recipe = SeriesRecipe(reads.json(entry["recipe"], maximum=16 * 1024**2))
            metadata[identity] = {"index": graph["canonical_index_sha256"], "pc": tuple(graph["pc_ancestor_index_sha256"]),
                "r": tuple(graph["r_ancestor_index_sha256"]), "publication": (), "recipe": recipe}
        elif kind == "utility_cohort":
            original_preparation = validate_training_preparation(reads.json(entry["preparation"], maximum=65536))
            graph = read_historical_utility_dependencies(source, preparation=original_preparation, read_budget=reads)
            metadata[identity] = {"index": None, "pc": (), "r": (),
                "publication": tuple(graph["ancestor_dataset_publication_sha256"]), "preparation": original_preparation}
        elif family.intervention and kind == "utility_intervention":
            from .rovent_utility_intervention_dataset import read_intervention_utility_dependencies
            original_preparation = family.parts(reads.json(entry["preparation"], maximum=65536)).preparation
            graph = read_intervention_utility_dependencies(source, preparation=original_preparation, read_budget=reads)
            metadata[identity] = {"index": None, "pc": (), "r": (),
                "publication": tuple(graph["ancestor_dataset_publication_sha256"]), "preparation": original_preparation}
        else:
            raise ValueError("explicit original mixed ancestor kind required")
        index = metadata[identity]["index"]
        if index is not None:
            sha(index)
            if index in by_index:
                raise ValueError("original PC/R ancestor canonical index repeated across families")
            by_index[index] = identity
    owners, pending = {}, list(entries)
    while pending:
        progressed = False
        for identity in tuple(pending):
            entry, graph = entries[identity], metadata[identity]
            dependencies = [*graph["publication"]]
            for index in (*graph["pc"], *graph["r"]):
                if index not in by_index:
                    raise ValueError("missing explicitly registered original PC/R dependency")
                dependencies.append(by_index[index])
            if any(dependency not in entries for dependency in dependencies):
                raise ValueError("missing original mixed utility ancestor publication")
            if any(dependency not in owners for dependency in dependencies):
                continue
            source, kind = entry["registration"], entry["kind"]
            if kind in ("pc_legacy", "pc_submitted_input_v1"):
                parents = tuple(owners[by_index[index]] for index in graph["pc"])
                if any(type(owner) is not pc.HistoricalRoventCorpus for owner in parents):
                    raise ValueError("original PC index must resolve to its own historical family")
                owner = pc.HistoricalRoventCorpus.from_registration(source, read_budget=reads, ancestors=parents)
                expected = LEGACY_CODEC if kind == "pc_legacy" else SUBMITTED_CODEC
                if owner._codec is not expected:
                    raise ValueError("original PC family changed between dependency and complete source reads")
            elif kind == "r_auxiliary":
                pc_parents = tuple(owners[by_index[index]] for index in graph["pc"])
                r_parents = tuple(owners[by_index[index]] for index in graph["r"])
                if (any(type(item) is not pc.HistoricalRoventCorpus for item in pc_parents)
                        or any(type(item) is not r.HistoricalRoventRCorpus for item in r_parents)):
                    raise ValueError("original R dependency kind differs from its registered source family")
                owner = r.HistoricalRoventRCorpus.from_registration(source, recipe=graph["recipe"], read_budget=reads,
                    pc_corpora=tuple(item.corpus for item in pc_parents), r_ancestors=r_parents)
            elif kind == "utility_cohort":
                owner = HistoricalUtilityRouteDataset.from_registration(source, preparation=graph["preparation"],
                    read_budget=reads, ancestors=tuple(owners[dependency] for dependency in graph["publication"]))
            else:
                if not family.intervention or kind != "utility_intervention":
                    raise ValueError("explicit typed original intervention ancestor required")
                from .rovent_utility_intervention_dataset import HistoricalUtilityInterventionDataset
                owner = HistoricalUtilityInterventionDataset.from_registration(source, preparation=graph["preparation"],
                    read_budget=reads, ancestors=tuple(owners[dependency] for dependency in graph["publication"]))
            owners[identity] = owner; pending.remove(identity); progressed = True
        if not progressed:
            raise ValueError("original mixed utility ancestry is cyclic or incomplete")
    if family.intervention:
        from .rovent_utility_intervention_dataset import HistoricalUtilityInterventionDataset
        dataset_type = HistoricalUtilityInterventionDataset
    else:
        dataset_type = HistoricalUtilityRouteDataset
    intake = dataset_type.from_registration(registration["current_dataset"], preparation=preparation,
        read_budget=reads, ancestors=tuple(owners.values()))
    intake.validate_execution_sources(registration)
    if intake.observation()["split"] != "train":
        raise ValueError("only the original current train cohort can enter the utility learner")
    return intake, tuple([intake, *(owner for owner in owners.values() if type(owner) is dataset_type)])


def _load_historical_owner(startup, reads, authority, preparation, spec, intake, base, torch, datasets, *, family=None):
    from .nnuec_utility_trainable import NnuecUtilityTrainableV1
    from .rovent_tensor_training import _load_pinned_tensor_file
    from .rovent_utility_historical_checkpoint import read_utility_historical_checkpoint
    from .rovent_utility_tensor_training import (CHECKPOINT_DOMAIN, MAX_CHECKPOINT_BYTES,
        stage_cohort_utility_resume, stage_utility_fork)
    from .rovent_utility_training_window import make_utility_sampler, validate_utility_sampler
    family = _checked_family(family)
    source = startup["registration"]["initialization_source"]
    if family.intervention:
        from .rovent_utility_intervention_checkpoint import read_intervention_utility_historical_checkpoint
        historical = read_intervention_utility_historical_checkpoint(source["checkpoint"], read_accounting=reads, datasets=datasets)
    else:
        historical = read_utility_historical_checkpoint(source["checkpoint"], read_accounting=reads, datasets=datasets)
    projection, pin = historical.projection(), historical.tensor_pin()
    # The tensor parser itself uses a nofollow/hash-pinned temporary snapshot
    # and weights_only. Its independent second lifecycle pass is charged again.
    payload, size = _load_pinned_tensor_file(pin["path"], pin["sha256"], MAX_CHECKPOINT_BYTES, torch,
        read_accounting=reads, snapshot_directory=startup["registration"]["checkpoint_directory"])
    if (type(payload) is not dict or payload.get("schema") != family.domain(CHECKPOINT_DOMAIN) or size != pin["bytes"]
            or payload.get("state_sha256") != projection["payload_state_sha256"]
            or json.dumps(payload.get("metadata"), sort_keys=True, separators=(",", ":"), allow_nan=False)
                != json.dumps(projection["metadata"], sort_keys=True, separators=(",", ":"), allow_nan=False)):
        raise ValueError("original utility template bytes/metadata differ from the historical producer")
    template = NnuecUtilityTrainableV1.from_tensor_state(projection["metadata"], payload["model"],
        frozen_base=base, device=spec["device"])
    del payload
    original_sampler = projection["sampler"]
    def source_sampler_validator(value):
        return validate_utility_sampler(value, context_ids=original_sampler["context_ids"],
            pool_sha256=original_sampler["pool_sha256"], run_id=original_sampler["run_id"], seed=original_sampler["seed"])
    original_preparation = projection["preparation"]
    if family.intervention:
        from .rovent_utility_intervention_checkpoint import stage_intervention_utility_resume
        staged = stage_intervention_utility_resume(pin["path"], pin["sha256"], template, original_preparation,
            sampler_validator=source_sampler_validator, max_file_bytes=MAX_CHECKPOINT_BYTES,
            read_accounting=reads, snapshot_directory=startup["registration"]["checkpoint_directory"], execute=True,
            authority=authority, historical_checkpoint=historical)
    else:
        staged = stage_cohort_utility_resume(pin["path"], pin["sha256"], template,
            original_preparation["recipe"], original_preparation["registration"], original_preparation["parameter_spec"],
            sampler_validator=source_sampler_validator, max_file_bytes=MAX_CHECKPOINT_BYTES,
            read_accounting=reads, snapshot_directory=startup["registration"]["checkpoint_directory"], execute=True,
            authority=authority, historical_checkpoint=historical)
    def current_sampler_validator(value):
        return validate_utility_sampler(value, context_ids=intake.context_ids, pool_sha256=intake.pool_sha256,
            run_id=spec["run_id"], seed=spec["training_seed"])
    if source["mode"] == "resume":
        if (json.dumps(original_preparation, sort_keys=True, separators=(",", ":"), allow_nan=False)
                != json.dumps(preparation, sort_keys=True, separators=(",", ":"), allow_nan=False)):
            raise ValueError("utility resume cannot change the original preparation")
        current_sampler_validator(staged.sampler)
        return staged
    if source["mode"] != "fork":
        raise ValueError("original utility resume/fork tag required")
    sampler = make_utility_sampler(intake.context_ids, intake.pool_sha256, spec["training_seed"], run_id=spec["run_id"])
    if family.intervention:
        from .rovent_utility_intervention_checkpoint import stage_intervention_utility_fork
        return stage_intervention_utility_fork(staged, preparation=preparation, authority=authority, sampler=sampler,
            source_sampler_validator=source_sampler_validator, sampler_validator=current_sampler_validator,
            reset_seed=spec["training_seed"], execute=True)
    return stage_utility_fork(staged, recipe=preparation["recipe"], registration=preparation["registration"],
        membership=preparation["parameter_spec"], sampler=sampler,
        source_sampler_validator=source_sampler_validator, sampler_validator=current_sampler_validator,
        reset_seed=spec["training_seed"], execute=True)


def _frozen_base(startup, reads, preparation):
    from .nnuec_reference import MAX_JSON_BYTES, decode_json
    from .nnuec_v2_reference import NnuecV2Checkpoint
    raw, readback = reads.read(startup["registration"]["frozen_nnuec_base"], maximum=MAX_JSON_BYTES)
    base = NnuecV2Checkpoint.from_mapping(decode_json(raw))
    declared = preparation["registration"]["frozen_base"]
    if base.weights_sha256 != declared["weights_sha256"] or base.base.config.to_mapping() != declared["config"]:
        raise ValueError("actual frozen utility base semantic weights/config differ from preparation")
    return base, readback


def _initial_owner(startup, reads, preparation, spec, intake, base, torch, *, authority=None, family=None):
    import random
    import numpy as np
    from .nnuec_reference import decode_json
    from .nnuec_utility_reference import MAX_CHECKPOINT_BYTES
    from .nnuec_utility_trainable import NnuecUtilityTrainableV1
    from .rovent_utility_tensor_training import create_cohort_utility_owner
    from .rovent_utility_training_window import make_utility_sampler, validate_utility_sampler
    family = _checked_family(family)
    if family.intervention:
        family.require_authority(authority, allow_unleased=True)
    raw, _identity = reads.read(startup["registration"]["initialization_source"]["utility"], maximum=MAX_CHECKPOINT_BYTES)
    seed = spec["training_seed"]
    random.seed(seed); np.random.seed(seed); torch.random.default_generator.manual_seed(seed)
    devices = (int(spec["device"][5:]),) if spec["device"].startswith("cuda:") else ()
    for device in devices:
        torch.cuda.default_generators[device].manual_seed(seed)
    model = NnuecUtilityTrainableV1.from_checkpoint(decode_json(raw), frozen_base=base, device=spec["device"])
    sampler = make_utility_sampler(intake.context_ids, intake.pool_sha256, seed, run_id=spec["run_id"])
    def validate_sampler(value):
        return validate_utility_sampler(value, context_ids=intake.context_ids, pool_sha256=intake.pool_sha256,
            run_id=spec["run_id"], seed=seed)
    if family.intervention:
        from .rovent_utility_intervention_checkpoint import create_intervention_utility_owner
        return create_intervention_utility_owner(model, preparation, authority=authority,
            sampler=sampler, sampler_validator=validate_sampler, cuda_devices=devices, execute=True)
    return create_cohort_utility_owner(model, preparation["recipe"], preparation["registration"], preparation["parameter_spec"],
        sampler=sampler, sampler_validator=validate_sampler, cuda_devices=devices, execute=True)


def _export_request(startup, reads, *, family=None):
    from .rovent_training_bootstrap import _child_path, _fields, _uint
    family = _checked_family(family)
    registration = startup["registration"]
    request = reads.json(registration["export_request"], maximum=65536)
    _fields(request, ("schema", "kind", "path", "max_output_bytes"), "utility export request")
    if request["schema"] != family.domain("rz-rovent-utility-export-request/1") or request["kind"] not in ("collection", "deployment"):
        raise ValueError("separate explicit utility export request required")
    target = Path(_child_path(registration["output_root"], request["path"], "utility export"))
    if target.suffix != ".json" or target.parent.resolve(strict=True) != target.parent or target.exists() or target.is_symlink():
        raise ValueError("new original plain utility JSON export target required")
    _uint(request["max_output_bytes"], registration["max_output_bytes"], positive=True)
    return request


def _execute_export(owner, intake, spec, request, authority, torch, *, clock, family=None):
    from .generation_data import uint
    from .rovent_utility_tensor_training import export_utility_owner
    family = _checked_family(family)
    family.require_authority(authority, allow_unleased=True)
    lease = authority.begin_window(owner, intake, spec)
    started, status, result = clock(), "export_failed", None
    try:
        if authority.should_stop(lease):
            raise TimeoutError("utility export cancelled/expired before its original writer")
        for device in owner.cuda_devices:
            torch.cuda.synchronize(device)
        owner.physical_completion = "confirmed"
        if family.intervention:
            from .rovent_utility_intervention_checkpoint import export_intervention_utility_owner
            export = export_intervention_utility_owner
        else:
            export = export_utility_owner
        result = export(owner, request["path"], authority=authority, kind=request["kind"],
            max_output_bytes=request["max_output_bytes"], execute=True)
        status = "export_complete"
        return result
    except BaseException:
        owner.quarantined = True
        raise
    finally:
        try:
            for device in owner.cuda_devices:
                torch.cuda.synchronize(device)
            owner.physical_completion = "confirmed"
        except BaseException:
            owner.physical_completion = "unknown"
            owner.quarantined = True
            raise
        finally:
            owner.cost["wall_time_ns"] = uint(owner.cost["wall_time_ns"] + max(0, clock() - started))
            authority.end_window(lease, owner, status, physical_completion=owner.physical_completion,
                dispatched_updates=0, result=result)


def original_utility_child_main(argv=None, *, stdin=None, clock=time.monotonic_ns):
    from .rovent_utility_training_family import LEGACY
    return _utility_child_main(argv, stdin=stdin, clock=clock, family=LEGACY)


def _utility_child_main(argv=None, *, stdin=None, clock=time.monotonic_ns, family):
    """Shared process machinery reached only through a closed family entry."""
    family = _checked_family(family)
    args = _parser().parse_args(argv)
    if not args.execute:
        raise ValueError("one original finite utility child requires explicit execution")
    raw = _read_startup(sys.stdin.buffer if stdin is None else stdin, args.startup_bytes)
    if family.intervention:
        from .rovent_utility_intervention_execution import validate_intervention_startup as validate
        from .rovent_utility_intervention_execution import bootstrap_intervention_child as bootstrap_child
        from .rovent_utility_intervention_authority import ParentSocketInterventionUtilityTrainerAuthority as Authority
    else:
        from .rovent_utility_training_bootstrap import validate_startup as validate, bootstrap_utility_child as bootstrap_child
        from .rovent_utility_training_authority import ParentSocketUtilityTrainerAuthority as Authority
    startup = validate(raw, expected_sha256=args.startup_sha256, expected_bytes=args.startup_bytes,
        execution_until_ns=args.execution_until_ns, whole_until_ns=args.whole_until_ns, now_ns=clock())
    _actual_registration(startup, family=family)
    bootstrap = bootstrap_child(startup, until_ns=args.execution_until_ns, clock=clock)
    report, reads = bootstrap.report, bootstrap.budget
    authority = Authority.connect_registered(startup, raw_startup=raw,
        expected_sha256=args.startup_sha256, read_accounting=reads, execute=True, clock=clock)
    owner, result, initial_output, initial_completed, initial_dispatched = None, None, 0, 0, 0
    tensor_admission_started = False
    old_handlers = {}
    for name in (signal.SIGTERM, signal.SIGINT):
        old_handlers[name] = signal.signal(name, lambda _signal, _frame: authority.observe_cancel())
    try:
        preparation, spec, preserved_pc = _metadata_inputs(startup, reads, family=family)
        report["frozen_pc_sources"] = preserved_pc
        request = _export_request(startup, reads, family=family) if startup["registration"]["action"] == "export" else None
        # No tensor/runtime import is permitted until this actual new cohort
        # data-integrity owner exists and its publication matches the parent.
        intake, utility_source_datasets = _load_actual_cohort(startup, reads, preparation, family=family)
        from .rovent_training_child import _configure_runtime, _verify_imported_runtime
        tensor_admission_started = True
        torch = _configure_runtime(spec)
        report["imported_runtime"] = _verify_imported_runtime(startup, reads, torch)
        base, report["frozen_nnuec_base"] = _frozen_base(startup, reads, preparation)
        if startup["registration"]["initialization_source"]["mode"] == "initialize":
            owner = _initial_owner(startup, reads, preparation, spec, intake, base, torch, authority=authority, family=family)
        else:
            owner = _load_historical_owner(startup, reads, authority, preparation, spec, intake, base, torch,
                utility_source_datasets, family=family)
        initial_output, initial_completed, initial_dispatched = owner.cost["output_bytes"], owner.completed_updates, owner.dispatched_updates
        if startup["registration"]["action"] == "export":
            result = _execute_export(owner, intake, spec, request, authority, torch, clock=clock, family=family)
        else:
            from .rovent_utility_training_run import _train_utility_window_core
            result = _train_utility_window_core(owner, intake, spec, authority=authority,
                checkpoint_directory=startup["registration"]["checkpoint_directory"], execute=True,
                clock=clock, family=family)
        status, error = "completed", None
    except BaseException as failure:
        if owner is not None:
            owner.quarantined = True
        status = "source_integration_pending" if isinstance(failure, SourceIntegrationPending) else "failed"
        from .rovent_training_child import _error_chain_types
        error = {"error_type": type(failure).__name__, "error_chain_types": _error_chain_types(failure)}
    finally:
        authority.close()
        for name, handler in old_handlers.items():
            signal.signal(name, handler)
    report["read_bytes"] = reads.spent
    terminal = {"schema": family.domain(RESULT_DOMAIN), "execution_id": startup["execution_id"], "window_id": startup["window_id"],
        "status": status, "result": result, "bootstrap": report,
        "completed_updates": 0 if owner is None else owner.completed_updates,
        "window_completed_updates": 0 if owner is None else owner.completed_updates - initial_completed,
        "dispatched_updates": 0 if owner is None else owner.dispatched_updates,
        "window_dispatched_updates": 0 if owner is None else owner.dispatched_updates - initial_dispatched,
        "physical_completion": ("unknown" if tensor_admission_started else "not_started") if owner is None else owner.physical_completion,
        "physical_process_retirement": "original_parent_join_required", "arena_eligible": False}
    if error is not None:
        terminal.update(error)
    if clock() >= args.whole_until_ns:
        raise TimeoutError("utility cleanup deadline expired; original parent retains partial evidence")
    maximum = startup["registration"]["max_output_bytes"] - (0 if owner is None else owner.cost["output_bytes"] - initial_output)
    from .rovent_training_child import _write_terminal
    receipt = _write_terminal(startup["registration"]["receipt_path"], terminal, maximum)
    summary = {name: terminal[name] for name in ("schema", "execution_id", "window_id", "completed_updates",
        "window_completed_updates", "dispatched_updates", "window_dispatched_updates", "physical_completion", "arena_eligible")}
    summary["result"] = {"status": status, "receipt": receipt, "learner_result": result, **({} if error is None else error)}
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0 if status == "completed" else 2 if status == "source_integration_pending" else 1


if __name__ == "__main__":
    raise SystemExit(original_utility_child_main())
