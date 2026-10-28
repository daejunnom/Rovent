"""Isolated recovered Rovent learner entry; module import is stdlib only.

Historical files are data and recorded cost, never a revived process/native
lease. A new OS parent must own this child and its live socket admission before
any tensor import. This entry accepts only the explicit /2 window and does not
retag the original /1 registration, window, checkpoint, or termination evidence.
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

RESULT_DOMAIN = "rz-rovent-recovered-trainer-result/1"
MAX_STARTUP = 16 * 1024**2
_ROLES = ("proposer", "critic", "validator")


def _parser():
    parser = argparse.ArgumentParser(description="New parent-owned recovered Rovent learner child")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--startup-sha256", required=True)
    parser.add_argument("--startup-bytes", type=int, required=True)
    parser.add_argument("--execution-until-ns", type=int, required=True)
    parser.add_argument("--whole-until-ns", type=int, required=True)
    return parser


def _read_startup(stream, count):
    if type(count) is not int or not 0 < count <= MAX_STARTUP:
        raise ValueError("bounded recovered startup byte count required")
    raw = stream.read(count + 1)
    if type(raw) is not bytes or len(raw) != count:
        raise ValueError("recovered startup bytes or EOF differ from pinned stdin")
    return raw


def _metadata_inputs(startup, reads):
    from .generation_data import fields
    from .rovent_generation import SeriesRecipe
    from .rovent_training import validate_training_membership, nnuec_membership
    from .rovent_training_window_v2 import validate_window_spec
    from .rovent_training_bootstrap import validate_recovered_window_binding
    from .artifacts import validate_rules_declaration
    from .config import ModelConfig
    registration = startup["registration"]
    recipe = SeriesRecipe(reads.json(registration["recipe"], maximum=1024**2))
    membership = reads.json(registration["membership"], maximum=16 * 1024**2)
    validate_training_membership(membership)
    spec = reads.json(registration["window_spec"], maximum=1024**2)
    # The explicit /2 parser precedes all runtime/model access. Do not project
    # its fields into an old window just to make the old child accept them.
    validate_window_spec(recipe, spec)
    binding = validate_recovered_window_binding(startup, spec)
    if spec["nnuec_semantics"] != nnuec_membership(membership)[1]:
        raise ValueError("recovered NNUEC membership and declared window semantics differ")
    rules = reads.json(registration["rules_profile"], maximum=65536)
    if type(rules) is not dict:
        raise ValueError("pinned recovered Rules profile object required")
    validate_rules_declaration(rules, ModelConfig.for_profile("virtual_review_v1"))
    for key in ("expected_initial_completed", "expected_ancestor_completed"):
        fields(registration[key], _ROLES, "registered recovered update counts")
        if any(type(value) is not int or value < 0 for value in registration[key].values()):
            raise ValueError("exact nonnegative recovered update counts required")
    return recipe, membership, spec, rules, binding


def _publication_metadata(files, reads, limits):
    from .rovent_historical_corpus import _publication
    publication = _publication(reads.json(files["publication"], maximum=16 * 1024**2), limits)
    root = Path(files["root"])
    pin = publication["manifest"]
    manifest = reads.json({"path": str(root.joinpath(*pin["path"].split("/"))),
        "bytes": pin["bytes"], "sha256": pin["sha256"]}, maximum=16 * 1024**2)
    parents = manifest.get("ancestor_index_sha256") if type(manifest) is dict else None
    from .generation_data import sha
    if (type(parents) is not list or len(parents) > 256
            or parents != sorted(set(sha(value) for value in parents))):
        raise ValueError("complete sorted historical ancestor index graph required")
    return publication, parents


def _intake(startup, reads):
    """Read exact durable files; never reconstruct a current producer token."""
    from .rovent_corpus import CorpusLimits
    from .rovent_historical_corpus import HistoricalRoventCorpus
    registration, limits = startup["registration"], CorpusLimits()
    pending, owners = [], {}
    for files in registration["ancestor_datasets"]:
        publication, parents = _publication_metadata(files, reads, limits)
        pending.append((files, publication, parents))
    while pending:
        progressed = False
        for entry in tuple(pending):
            files, publication, parents = entry
            if any(identity not in owners for identity in parents):
                continue
            owner = HistoricalRoventCorpus.from_registration(files, read_budget=reads,
                ancestors=tuple(owners[identity] for identity in parents), limits=limits)
            identity = owner.corpus.index_semantic_sha256
            if identity != publication["canonical_index_sha256"] or identity in owners:
                raise ValueError("duplicate or changed historical ancestor corpus identity")
            owners[identity] = owner
            pending.remove(entry)
            progressed = True
        if not progressed:
            raise ValueError("historical dataset graph is cyclic, foreign, or incomplete")
    files = registration["current_dataset"]
    _publication, parents = _publication_metadata(files, reads, limits)
    if parents != sorted(owners):
        raise ValueError("registered historical ancestors differ from the complete current dataset ancestry")
    owner = HistoricalRoventCorpus.from_registration(files, read_budget=reads,
        ancestors=tuple(owners[identity] for identity in parents), limits=limits)
    ancestors = tuple(owners[identity] for identity in sorted(owners))
    corpus_by_publication = {item.observation()["publication_sha256"]: item.corpus for item in (*ancestors, owner)}
    if len(corpus_by_publication) != len(ancestors) + 1:
        raise ValueError("historical dataset publication cannot alias another corpus owner")
    return owner, ancestors, corpus_by_publication


def _r_auxiliary_presence(startup, spec):
    """Preserve absent/null/enabled without importing or opening an R source.

    This comparison is neither a corpus validation nor a training capability.
    The registration and window codecs separately validate enabled objects;
    their presence must also match before considering the typed file owner.
    """
    registration = startup["registration"]
    declared, configured = "r_auxiliary_dataset" in registration, "nnuec_r_auxiliary" in spec
    if declared != configured:
        raise ValueError("registered R auxiliary presence differs from the exact window presence")
    if not declared:
        return "absent", None
    request, auxiliary = registration["r_auxiliary_dataset"], spec["nnuec_r_auxiliary"]
    if request is None and auxiliary is None:
        return "null", None
    if type(request) is not dict or type(auxiliary) is not dict:
        raise ValueError("registered R auxiliary null/enabled state differs from the exact window state")
    return "enabled", request


def _intake_r_auxiliary(startup, reads, recipe, spec, *, pc_corpus_by_publication):
    """Keep the optional R owner distinct and disabled paths free of R IO.

    The historical reader selects the exact declared PC ancestor set from the
    available validated owners. Its result still grants no old native or new
    trainer authority; only the current parent's per-sample admission can do so.
    """
    state, request = _r_auxiliary_presence(startup, spec)
    if state != "enabled":
        return None
    from .rovent_historical_r_corpus import HistoricalRoventRCorpus, read_historical_r_dependencies
    if type(pc_corpus_by_publication) is not dict or len(pc_corpus_by_publication) > 257:
        raise ValueError("bounded validated PC publication owners are required for R lineage")
    pending, ancestors, available_pc = [], {}, tuple(pc_corpus_by_publication.values())
    registrations = startup["registration"].get("ancestor_r_auxiliary_datasets")
    for source in () if registrations is None else registrations:
        dependency = read_historical_r_dependencies(source, read_budget=reads)
        pending.append((source, dependency))
    while pending:
        progressed = False
        for entry in tuple(pending):
            source, dependency = entry
            if any(identity not in ancestors for identity in dependency["r_ancestor_index_sha256"]):
                continue
            owner = HistoricalRoventRCorpus.from_registration(source, recipe=recipe, read_budget=reads,
                pc_corpora=available_pc, r_ancestors=tuple(ancestors.values()))
            identity = owner.corpus.index_semantic_sha256
            if identity != dependency["canonical_index_sha256"] or identity in ancestors:
                raise ValueError("duplicate or changed historical R ancestor corpus identity")
            ancestors[identity] = owner
            pending.remove(entry)
            progressed = True
        if not progressed:
            raise ValueError("historical R auxiliary graph is cyclic, foreign, or incomplete")
    dependency = read_historical_r_dependencies(request, read_budget=reads)
    if dependency["r_ancestor_index_sha256"] != sorted(ancestors):
        raise ValueError("registered historical R ancestors differ from the complete current R lineage")
    owner = HistoricalRoventRCorpus.from_registration(request, recipe=recipe, read_budget=reads,
        pc_corpora=available_pc, r_ancestors=tuple(ancestors.values()))
    corpus, configured = owner.corpus, spec["nnuec_r_auxiliary"]
    if (corpus.index_semantic_sha256 != dependency["canonical_index_sha256"]
            or corpus.corpus_sha256 != configured["manifest_file_sha256"]
            or corpus.index_sha256 != configured["index_file_sha256"]):
        raise ValueError("typed R auxiliary corpus differs from the exact window file pins")
    ancestors = tuple(ancestors[identity] for identity in sorted(ancestors))
    corpus_by_publication = {item.observation()["publication_sha256"]: item.corpus for item in (*ancestors, owner)}
    if len(corpus_by_publication) != len(ancestors) + 1:
        raise ValueError("historical R publication cannot alias another R corpus owner")
    return owner, ancestors, corpus_by_publication


def _join_r_auxiliary_admission(startup, observation, ancestor_observations):
    """Compare durable observations to this parent's pinned startup report.

    Neither side is turned into historical live authority. The actual current
    socket owner remains responsible for original custody and update admission.
    """
    report = startup["r_auxiliary_admission"]
    expected = {"current_publication_sha256": observation["publication_sha256"],
        "ancestor_publication_sha256": sorted(item["publication_sha256"] for item in ancestor_observations),
        **{name: observation[name] for name in ("manifest_file_sha256", "index_file_sha256",
            "nnuec_parameter_sha256", "record_count")}}
    if (any(name not in report or type(report[name]) is not type(value) or report[name] != value
            for name, value in expected.items()) or observation.get("execution_authorized") is not False
            or observation.get("past_native_authority_reissued") is not False
            or observation.get("past_physical_completion_reissued") is not False):
        raise ValueError("typed historical R observation differs from the current parent's pinned admission")
    return {"schema": "rz-rovent-recovered-r-child-binding/1", **expected,
        "scope": "durable_observation_and_current_startup_projection_only_not_historical_native_or_execution_authority",
        "past_native_authority_reissued": False, "past_physical_completion_reissued": False,
        "execution_authorized": False}


def _verify_imported_runtime(startup, reads, torch):
    # Endpoint byte/inode observation is independent of collection's ORT pin.
    # The old low-level observer has no lease or window codec conversion.
    from .rovent_training_child import _verify_imported_runtime as verify
    result = verify(startup, reads, torch)
    endpoints = {row["name"]: row["artifact"] for row in startup["registration"]["runtime_endpoints"]}
    for observed in result["endpoints"]:
        key = observed["module"] + ".__file__"
        expected = endpoints.get(key)
        if (expected is None or observed["sha256"] != expected["sha256"]
                or observed["bytes"] != expected["bytes"]):
            raise ValueError("imported recovered Torch endpoint differs from its logical registered pin")
    result["schema"] = "rz-rovent-recovered-imported-runtime-readback/1"
    result["collection_runtime_identity_scope"] = "historical_loaded_ORT_library_not_this_trainer_python_or_torch"
    result["past_native_authority_reissued"] = False
    result["past_physical_completion_reissued"] = False
    return result


def _initial_owner(startup, recipe, membership, spec, native_intake, reads, torch, *, r_auxiliary_intake=None):
    from types import SimpleNamespace
    from .generation_data import fields
    from .rovent_tensor_training import _load_pinned_tensor_file, create_tensor_owner
    from .rovent_recovered_training_window import _read_admission, _journal, initial_training_journal
    from .rovent_training_child import _pc_template, _roles, _check_pc_state, _nnuec_class
    source = startup["registration"]["initialization_source"]
    if source["kind"] != "initialize":
        raise ValueError("explicit recovered initialize source required")
    pin = source["pc_checkpoint"]
    payload, size = _load_pinned_tensor_file(pin["path"], pin["sha256"], pin["bytes"], torch,
        read_accounting=reads, snapshot_directory=Path(startup["registration"]["output_root"]) / "tmp")
    if size != pin["bytes"]:
        raise ValueError("pinned recovered initial P/C byte count changed")
    fields(payload, ("state_dict", "config", "seed", "schema", "training_steps", "trained"), "untrained recovered P/C initialization")
    if (type(payload["training_steps"]) is not int or payload["training_steps"] != 0
            or payload["trained"] is not False or type(payload["seed"]) is not int):
        raise ValueError("recovered initialize cannot relabel learned or diagnostic tensors")
    model = _pc_template(payload["config"], _roles(membership), torch, device=spec["device"])
    if payload["schema"] != model.config.model_semantics:
        raise ValueError("recovered initial P/C model semantics/configuration mismatch")
    _check_pc_state(model, payload["state_dict"], torch)
    model.load_state_dict(payload["state_dict"], strict=True)
    nnuec = reads.json(source["nnuec_checkpoint"], maximum=startup["registration"]["max_read_bytes"])
    model.nnuec = _nnuec_class(nnuec["schema"]).from_checkpoint(nnuec, device=spec["device"])
    if model.nnuec.metadata_state()["training"] != {"kind": "untrained"}:
        raise ValueError("recovered initialize requires an explicit untrained NNUEC declaration")
    probe = SimpleNamespace(recipe=recipe, stage_id=spec["stage_id"], inherited_input_sha256=[], inherited_origin_sha256=[])
    auxiliary = {} if r_auxiliary_intake is None else {"r_auxiliary_intake": r_auxiliary_intake}
    admission = _read_admission(probe, native_intake, spec, **auxiliary)
    journal = initial_training_journal(recipe, spec, admission, **auxiliary)
    probe.samplers, probe.completed = journal, dict.fromkeys(_ROLES, 0)
    def validate_sampler(value):
        candidate = copy.copy(probe)
        candidate.samplers = value
        _journal(candidate, spec, admission, **auxiliary)
    import random
    import numpy as np
    random.seed(spec["training_seed"])
    np.random.seed(spec["training_seed"])
    devices = (int(spec["device"][5:]),) if spec["device"].startswith("cuda:") else ()
    torch.random.default_generator.manual_seed(spec["training_seed"])
    for device in devices:
        torch.cuda.default_generators[device].manual_seed(spec["training_seed"])
    return create_tensor_owner(model, recipe, membership, stage_id=spec["stage_id"], samplers=journal,
        bucket=8, sampler_validator=validate_sampler, cuda_devices=devices, execute=True)


def _recover_owner(startup, recipe, membership, spec, native_intake, authority, reads, torch, *, corpus_by_publication,
                   r_auxiliary_intake=None, r_corpus_by_publication=None):
    from .rovent_tensor_training import _load_pinned_tensor_file
    from .rovent_training_child import _template_from_tensor_payload
    from .rovent_recovered_training_window import (read_historical_checkpoint_source,
        stage_rovent_resume, stage_rovent_fork, stage_rovent_export)
    registration = startup["registration"]
    historical = {} if r_corpus_by_publication is None else {"r_corpus_by_publication": r_corpus_by_publication}
    source = read_historical_checkpoint_source(startup, reads, corpus_by_publication=corpus_by_publication, **historical)
    # The actual current socket owner separately checks history and scratch
    # state; the durable publication cannot fabricate this live relationship.
    authority.attach_historical_checkpoint(source)
    pin = source.tensor_pin
    payload, size = _load_pinned_tensor_file(pin["path"], pin["sha256"], pin["bytes"], torch,
        read_accounting=reads, snapshot_directory=Path(registration["output_root"]) / "tmp")
    if size != pin["bytes"]:
        raise ValueError("recovered exact tensor checkpoint byte count changed")
    kind = registration["initialization_source"]["kind"]
    if (kind == "fork" and payload.get("stage_id")
            != registration["initialization_source"]["parent_stage_id"]):
        raise ValueError("recovered fork tensor differs from its explicitly registered parent stage")
    template = _template_from_tensor_payload(payload, membership, torch, spec["device"])
    if kind == "resume":
        factory = stage_rovent_export if registration["action"] == "export" else stage_rovent_resume
    elif kind == "fork":
        factory = stage_rovent_fork
    else:
        raise ValueError("recovered scratch requires explicit resume or fork source")
    auxiliary = {} if r_auxiliary_intake is None else {"r_auxiliary_intake": r_auxiliary_intake}
    return factory(pin["path"], pin["sha256"], template, recipe, membership, native_intake, spec,
        execution_authority=authority, max_file_bytes=pin["bytes"], execute=True, **auxiliary)


def _export_request(startup, reads):
    from .generation_data import fields, uint
    registration = startup["registration"]
    request = reads.json(registration["export_request"], maximum=65536)
    fields(request, ("schema", "kind", "directory", "max_output_bytes"), "new recovered export request")
    directory, root = Path(request["directory"]), Path(registration["output_root"])
    if (request["schema"] != "rz-rovent-recovered-export-request/1"
            or request["kind"] not in ("collection", "deployment") or not directory.is_absolute()
            or directory.parent != root or directory == Path(registration["checkpoint_directory"])
            or directory == Path(startup["source_root"]) or directory.parent.resolve(strict=True) != directory.parent
            or directory.exists() or directory.is_symlink()
            or not 1 <= uint(request["max_output_bytes"], min(registration["max_output_bytes"], 2 * 1024**3))):
        raise ValueError("bounded new recovered collection/deployment export request required")
    return request


def _execute_export(startup, owner, staged, native_intake, spec, request, authority, reads, torch, *, clock, cancelled,
                    r_auxiliary_intake=None):
    from .generation_data import uint
    from .generation_export import _hash_file
    from .rovent_tensor_training import export_tensor_owner
    from .rovent_recovered_training_window import _read_admission, LearnedExportRequest
    from .rovent_training_child import _rules_export_snapshot
    started, initial_output = clock(), owner.cost["output_bytes"]
    lease, status = None, "failed"
    owner.physical_completion = "unknown"
    try:
        if cancelled() or clock() >= startup["execution_until_ns"]:
            raise TimeoutError("recovered export cancelled or expired before writer admission")
        auxiliary = {} if r_auxiliary_intake is None else {"r_auxiliary_intake": r_auxiliary_intake}
        lease = authority.begin_window(owner, native_intake, spec, _read_admission(owner, native_intake, spec, **auxiliary))
        rules, auxiliary_bytes = _rules_export_snapshot(startup, reads)
        available = request["max_output_bytes"] - auxiliary_bytes
        if available <= 0:
            raise ValueError("recovered export cannot fit its pinned Rules snapshot")
        result = export_tensor_owner(owner, request["directory"], kind=request["kind"], rules_profile_json=rules,
            learned_request=LearnedExportRequest(native_intake, spec, authority, **auxiliary),
            sampler_validator=staged.sampler_validator, max_output_bytes=available, execute=True)
        owner.cost["output_bytes"] = uint(owner.cost["output_bytes"] + auxiliary_bytes)
        receipt_path = Path(request["directory"]) / "rovent-export.json"
        identity, size = _hash_file(receipt_path, request["max_output_bytes"], read_accounting=reads)
        pin = {"path": str(receipt_path), "sha256": identity, "bytes": size}
        status = "export_complete"
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
            if lease is not None:
                authority.end_window(lease, status=status, owner=owner)
    return result, pin, {"wall_time_ns": clock() - started,
        "output_bytes": owner.cost["output_bytes"] - initial_output,
        "scope": "new_export_through_fence_and_end_ack_not_historical_drain_or_total_unknown_cost"}


def _history_scope(startup):
    return {"collection_admission": copy.deepcopy(startup["collection_admission"]),
        "checkpoint_admission": copy.deepcopy(startup["checkpoint_admission"]),
        "past_native_authority_reissued": False, "past_physical_completion_reissued": False,
        "scope": "recorded_historical_integrity_only_new_parent_owns_this_window"}


def _execute_ipc_check(startup, owner, native_intake, spec, authority, torch, *, clock, cancelled,
                       r_auxiliary_intake=None):
    """Check this child and its live parent without dispatching training work."""
    from .rovent_recovered_training_window import train_rovent_window

    auxiliary = {} if r_auxiliary_intake is None else {"r_auxiliary_intake": r_auxiliary_intake}
    before_completed, before_dispatched = dict(owner.completed), dict(owner.dispatched)
    if owner.pending_microbatches != 0 or owner.quarantined:
        raise ValueError("IPC check requires an idle nonquarantined tensor owner")
    if cancelled() or clock() >= startup["execution_until_ns"]:
        raise TimeoutError("IPC check cancelled or expired before admission")
    validation = train_rovent_window(owner, native_intake, spec,
        startup["registration"]["checkpoint_directory"], execute=False, **auxiliary)
    if validation.get("actual_training_executed") is not False or validation.get("optimizer_updates") != 0:
        raise ValueError("IPC check requires validation-only training admission")
    if cancelled() or clock() >= startup["execution_until_ns"]:
        raise TimeoutError("IPC check cancelled or expired before parent handshake")
    owner.physical_completion = "unknown"
    try:
        lease = authority.begin_window(owner, native_intake, spec, validation["admission"])
        for device in owner.cuda_devices:
            torch.cuda.synchronize(device)
        owner.physical_completion = "confirmed"
        if (owner.completed != before_completed or owner.dispatched != before_dispatched
                or owner.pending_microbatches != 0):
            raise ValueError("IPC check unexpectedly changed update state")
        if cancelled() or clock() >= startup["execution_until_ns"]:
            raise TimeoutError("IPC check cancelled or expired before terminal acknowledgement")
        authority.end_window(lease, status="ipc_check_complete", owner=owner)
    except BaseException:
        owner.quarantined = True
        raise
    return {"schema": "rz-rovent-recovered-trainer-ipc-check/1", "optimizer_updates": 0,
        "actual_training_executed": False, "scope": "current_child_dataset_and_ipc_only",
        "validation": validation, "model_owner_created": True,
        "optimizer_objects_may_be_created": True, "cuda_synchronization_performed": bool(owner.cuda_devices)}


def recovered_rovent_child_main(argv=None, *, stdin=None, clock=time.monotonic_ns):
    args = _parser().parse_args(argv)
    if not args.execute:
        raise ValueError("new original parent must explicitly authorize one finite recovered execution")
    raw = _read_startup(sys.stdin.buffer if stdin is None else stdin, args.startup_bytes)
    from .rovent_training_bootstrap import validate_recovered_startup, bootstrap_recovered_child
    startup = validate_recovered_startup(raw, expected_sha256=args.startup_sha256, expected_bytes=args.startup_bytes,
        execution_until_ns=args.execution_until_ns, whole_until_ns=args.whole_until_ns, now_ns=clock())
    bootstrap = bootstrap_recovered_child(startup, until_ns=args.execution_until_ns, clock=clock)
    readback, reads = bootstrap.report, bootstrap.budget
    from .rovent_recovered_training_authority import RecoveredParentSocketTrainerAuthority
    authority = RecoveredParentSocketTrainerAuthority.connect_registered(startup, raw_startup=raw,
        expected_sha256=args.startup_sha256, execute=True, clock=clock, read_accounting=reads)
    owner, result, cancelled, start_output = None, None, False, 0
    staged, export_pin, export_expense = None, None, None
    old_handlers = {}
    def receive_cancel(_signal, _frame):
        nonlocal cancelled
        cancelled = True
    for name in (signal.SIGTERM, signal.SIGINT):
        old_handlers[name] = signal.signal(name, receive_cancel)
    try:
        registration = startup["registration"]
        recipe, membership, spec, rules, binding = _metadata_inputs(startup, reads)
        readback["window_binding"] = binding
        from .rovent_training_child import _configure_runtime
        request = _export_request(startup, reads) if registration["action"] == "export" else None
        historical, ancestors, corpus_by_publication = _intake(startup, reads)
        native_intake = historical.corpus
        readback["historical_dataset"] = historical.observation()
        readback["historical_ancestor_publications"] = [item.observation()["publication_sha256"] for item in ancestors]
        r_bundle = _intake_r_auxiliary(startup, reads, recipe, spec, pc_corpus_by_publication=corpus_by_publication)
        auxiliary, r_sources = {}, {}
        if r_bundle is not None:
            historical_r, r_ancestors, r_corpus_by_publication = r_bundle
            r_intake = historical_r.corpus
            auxiliary["r_auxiliary_intake"] = r_intake
            r_sources["r_corpus_by_publication"] = r_corpus_by_publication
            r_observation = historical_r.observation()
            ancestor_r_observations = [item.observation() for item in r_ancestors]
            readback["r_auxiliary_current_binding"] = _join_r_auxiliary_admission(startup, r_observation, ancestor_r_observations)
            readback["historical_r_auxiliary_dataset"] = r_observation
            readback["historical_ancestor_r_publications"] = [item["publication_sha256"] for item in ancestor_r_observations]
            # Candidate targets remain a separate inventory. In particular,
            # captured R rows with unknown WDL never count as valid NNUEC
            # updates merely because their files and lineage were accepted.
            readback["r_auxiliary_target_inventory"] = r_intake.observation()
        if cancelled or clock() >= args.execution_until_ns:
            raise TimeoutError("recovered child cancelled or expired before tensor admission")
        torch = _configure_runtime(spec)
        readback["imported_runtime"] = _verify_imported_runtime(startup, reads, torch)
        readback["precision_environment"] = {"precision": "fp32", "tf32": False,
            "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
        if registration["initialization_source"]["kind"] == "initialize":
            owner = _initial_owner(startup, recipe, membership, spec, native_intake, reads, torch, **auxiliary)
        else:
            staged = _recover_owner(startup, recipe, membership, spec, native_intake, authority, reads, torch,
                corpus_by_publication=corpus_by_publication, **auxiliary, **r_sources)
            owner = staged.owner
        start_output = owner.cost["output_bytes"]
        if (owner.completed != registration["expected_initial_completed"]
                or owner.ancestor_updates != registration["expected_ancestor_completed"]):
            raise ValueError("recovered actual scratch update ancestry differs from current registration")
        from .artifacts import validate_rules_declaration
        rules_semantic = validate_rules_declaration(rules, owner.model.config)
        readback["rules_profile"] = {"descriptor_sha256": registration["rules_profile"]["sha256"],
            "rules_input_semantic_sha256": rules_semantic, "rules_input_profile": rules["rules_input_profile"],
            "rules_encoder_source_sha256": rules["rules_encoder_source_sha256"],
            "scope": "declared_model_input_contract_not_independent_Rules_execution_proof"}
        if registration["action"] == "export":
            result, export_pin, export_expense = _execute_export(startup, owner, staged, native_intake, spec,
                request, authority, reads, torch, clock=clock, cancelled=lambda: cancelled, **auxiliary)
        elif registration["action"] == "ipc_check":
            result = _execute_ipc_check(startup, owner, native_intake, spec, authority, torch,
                clock=clock, cancelled=lambda: cancelled, **auxiliary)
        else:
            from .rovent_recovered_training_window import train_rovent_window
            result = train_rovent_window(owner, native_intake, spec, registration["checkpoint_directory"],
                execution_authority=authority, cancelled=lambda: cancelled,
                execution_until_ns=args.execution_until_ns, execute=True, **auxiliary)
        terminal = {"schema": RESULT_DOMAIN, "execution_id": startup["execution_id"], "window_id": startup["window_id"],
            "status": "completed", "result": result, "bootstrap": readback, "historical": _history_scope(startup),
            "completed_updates": dict(owner.completed), "physical_completion": owner.physical_completion,
            "physical_completion_scope": "this_new_child_only_not_historical_physical_completion",
            "physical_process_retirement": "new_original_parent_join_required", "arena_eligible": False}
        if export_pin is not None:
            terminal.update(export_receipt=export_pin, export_expense=export_expense)
    except BaseException as error:
        from .rovent_training_child import _error_chain_types
        if owner is not None:
            owner.quarantined = True
        terminal = {"schema": RESULT_DOMAIN, "execution_id": startup["execution_id"], "window_id": startup["window_id"],
            "status": "failed", "error_type": type(error).__name__, "error_chain_types": _error_chain_types(error),
            "completed_updates": None if owner is None else dict(owner.completed),
            "dispatched_updates": None if owner is None else dict(owner.dispatched),
            "cost": None if owner is None else dict(owner.cost),
            "physical_completion": "unknown" if owner is None else getattr(owner, "physical_completion", "unknown"),
            "physical_completion_scope": "this_new_child_only_not_historical_physical_completion",
            "quarantined": owner is not None, "bootstrap": readback, "historical": _history_scope(startup),
            "physical_process_retirement": "new_original_parent_join_required", "arena_eligible": False}
    finally:
        authority.close()
        for name, previous in old_handlers.items():
            signal.signal(name, previous)
    readback["read_bytes"] = reads.spent
    readback["child_read_accounting"] = reads.observation()
    terminal["child_read_accounting"] = reads.observation()
    if clock() >= args.whole_until_ns:
        raise TimeoutError("new original cleanup deadline expired; partial evidence stays with its parent")
    remaining = startup["registration"]["max_output_bytes"] - (0 if owner is None else owner.cost["output_bytes"] - start_output)
    from .rovent_training_child import _write_terminal
    pin = _write_terminal(startup["registration"]["receipt_path"], terminal, remaining)
    summary = {"schema": RESULT_DOMAIN, "execution_id": startup["execution_id"], "window_id": startup["window_id"],
        "status": terminal["status"], "completed_updates": terminal.get("completed_updates"),
        "physical_completion": terminal["physical_completion"],
        "child_read_accounting": reads.observation(),
        "past_native_authority_reissued": False, "past_physical_completion_reissued": False,
        "arena_eligible": False, "receipt": pin}
    if "error_type" in terminal:
        summary["error_type"] = terminal["error_type"]
        summary["error_chain_types"] = terminal["error_chain_types"]
    if export_pin is not None:
        summary["export_receipt"] = export_pin
        summary["result"] = result
    print(json.dumps(summary, sort_keys=True))
    return 0 if terminal["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(recovered_rovent_child_main())
