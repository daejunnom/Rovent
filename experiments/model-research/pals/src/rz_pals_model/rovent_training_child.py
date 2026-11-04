"""Isolated original Rovent trainer child; importing this module is stdlib only.

The Rust parent owns live producer custody and process retirement. Startup JSON
alone never authorizes training. The resource bootstrap and live Unix parent
handshake precede every tensor/model import. This entry is separate from the
fixed historical first-generation trainer and does not reinterpret its recipe.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

RESULT_DOMAIN = "rz-rovent-original-trainer-result/1"
MAX_STARTUP = 16 * 1024**2


def _parser():
    parser = argparse.ArgumentParser(description="Original registered Rovent learner child")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--startup-sha256", required=True)
    parser.add_argument("--startup-bytes", type=int, required=True)
    parser.add_argument("--execution-until-ns", type=int, required=True)
    parser.add_argument("--whole-until-ns", type=int, required=True)
    return parser


def _read_startup(stream, count):
    if type(count) is not int or not 0 < count <= MAX_STARTUP:
        raise ValueError("bounded original startup byte count required")
    raw = stream.read(count + 1)
    if type(raw) is not bytes or len(raw) != count:
        raise ValueError("original startup bytes or EOF differ from pinned stdin")
    return raw


def _metadata_inputs(startup, reads):
    # No sibling checkpoint metadata is inferred from a filename. Every read
    # consumes an explicitly registered original pin.
    from .generation_data import fields
    from .rovent_generation import SeriesRecipe
    from .rovent_training import validate_training_membership, nnuec_membership
    from .rovent_training_window import validate_window_spec
    from .artifacts import validate_rules_declaration
    from .config import ModelConfig
    registration = startup["registration"]
    recipe = SeriesRecipe(reads.json(registration["recipe"], maximum=1024**2))
    membership = reads.json(registration["membership"], maximum=16 * 1024**2)
    validate_training_membership(membership)
    window = reads.json(registration["window_spec"], maximum=1024**2)
    validate_window_spec(recipe, window)
    rules = reads.json(registration["rules_profile"], maximum=65536)
    if type(rules) is not dict:
        raise ValueError("pinned Rules profile object required")
    validate_rules_declaration(rules, ModelConfig.for_profile("virtual_review_v1"))
    if (window["window_id"] != startup["window_id"]
            or window["runtime_sha256"] != registration["python_runtime_sha256"]
            or window["model_semantics_sha256"] != registration["model_semantics_sha256"]
            or window["device"] != registration["resources_policy"]["device"]
            or window["nnuec_semantics"] != nnuec_membership(membership)[1]):
        raise ValueError("registered original window/runtime/model/resource differs")
    for key in ("expected_initial_completed", "expected_ancestor_completed"):
        fields(registration[key], ("proposer", "critic", "validator"), "registered update counts")
        if any(type(value) is not int or value < 0 for value in registration[key].values()):
            raise ValueError("exact nonnegative registered update counts required")
    return recipe, membership, window, rules


def _intake(startup, reads):
    from .rovent_corpus import RoventTrainingCorpus
    registration = startup["registration"]
    pending, ancestors = [], {}
    for files in registration["ancestor_corpora"]:
        manifest = reads.json(files["manifest"], maximum=16 * 1024**2)
        pending.append((files, manifest))
    def load(files, parents):
        root = Path(files["root"])
        if any(Path(files[name]["path"]).parent != root for name in ("manifest", "index")):
            raise ValueError("original corpus manifest/index must share its registered chunk root")
        index = reads.json(files["index"], maximum=16 * 1024**2)
        amount = files["manifest"]["bytes"] + files["index"]["bytes"]
        if type(index.get("chunks")) is not list:
            raise ValueError("original corpus bounded chunk pin list required")
        for chunk in index["chunks"]:
            count = chunk.get("bytes")
            if type(count) is not int or count <= 0:
                raise ValueError("positive exact corpus chunk byte credit required")
            amount += count
        reads.account_external(amount)
        return RoventTrainingCorpus.from_files(files["manifest"]["path"],
            manifest_sha256=files["manifest"]["sha256"], index_path=files["index"]["path"],
            index_sha256=files["index"]["sha256"], ancestors=tuple(parents))
    while pending:
        progressed = False
        for entry in list(pending):
            files, manifest = entry
            parents = manifest.get("ancestor_index_sha256")
            if type(parents) is not list:
                raise ValueError("complete original ancestor index dependencies required")
            if any(identity not in ancestors for identity in parents):
                continue
            owner = load(files, [ancestors[identity] for identity in parents])
            if owner.index_semantic_sha256 in ancestors:
                raise ValueError("duplicate registered ancestor corpus identity")
            ancestors[owner.index_semantic_sha256] = owner
            pending.remove(entry)
            progressed = True
        if not progressed:
            raise ValueError("original corpus ancestor graph is cyclic or incomplete")
    files = registration["current_corpus"]
    return load(files, list(ancestors.values()))


def _configure_runtime(spec):
    # Called only after cgroup/affinity readback and original parent handshake.
    if spec["device"].startswith("cuda:"):
        # The original supervisor clears inherited environment. Install this
        # registered deterministic requirement before importing CUDA/Torch.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if (os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
                or os.environ.get("NVIDIA_TF32_OVERRIDE") != "0"):
            raise ValueError("original deterministic FP32 CUDA environment differs")
    import torch
    torch.set_num_threads(spec["cpu_threads"])
    torch.set_num_interop_threads(spec["interop_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if spec["device"].startswith("cuda:"):
        device = int(spec["device"][5:])
        if not torch.cuda.is_available() or device >= torch.cuda.device_count():
            raise ValueError("registered CUDA device unavailable; no CPU fallback")
        torch.cuda.set_device(device)
    return torch


def _verify_imported_runtime(startup, reads, torch):
    """Pin the imported Torch endpoints, leaving transitive mapping unknown."""
    pins = {str(Path(pin["path"])): pin for pin in startup["runtime_module_pins"]}
    observed = []
    for name, module in (("torch", torch), ("torch._C", getattr(torch, "_C", None))):
        path = getattr(module, "__file__", None)
        if (module is None or sys.modules.get(name) is not module or type(path) is not str
                or not Path(path).is_absolute() or str(Path(path)) not in pins):
            raise ValueError("original runtime pin missing for imported endpoint: " + name)
        identity = reads.verify(pins[str(Path(path))], maximum=2 * 1024**3)
        observed.append({"module": name, **identity})
    # The extension is a real native loaded image. The Python source endpoint
    # is byte-pinned but is not expected to have an executable memory mapping.
    extension = observed[1]
    fd = os.open("/proc/self/maps", os.O_RDONLY | os.O_NOFOLLOW)
    try:
        chunks, size = [], 0
        while True:
            block = os.read(fd, min(65536, 8 * 1024**2 + 1 - size))
            reads.account_external(len(block))
            size += len(block)
            if size > 8 * 1024**2:
                raise ValueError("runtime mapped-image observation exceeds bounded snapshot")
            if not block:
                break
            chunks.append(block)
    finally:
        os.close(fd)
    matched = False
    for line in b"".join(chunks).decode("utf-8", "strict").splitlines():
        values = line.split(maxsplit=5)
        if len(values) < 5 or ":" not in values[3]:
            raise ValueError("malformed original runtime mapped-image observation")
        major, minor = (int(value, 16) for value in values[3].split(":"))
        if (os.makedev(major, minor), int(values[4])) == (extension["device"], extension["inode"]):
            matched = True
    if not matched:
        raise ValueError("pinned Torch extension is absent from actual mapped images")
    return {"schema": "rz-rovent-imported-runtime-readback/1", "endpoints": observed,
        "torch_extension_mapping": "confirmed_device_inode",
        "transitive_native_cuda_mapping": "unknown", "physical_completion": "not_an_observation"}


def _pc_template(config, roles, torch, *, device):
    from .config import ModelConfig
    from .model import PalsModel
    selected = ModelConfig(**config)
    selected.validate()
    if not selected.virtual_review:
        raise ValueError("Rovent requires the explicitly registered compact P/C model")
    # Empty construction avoids overwriting global RNG while making a scratch
    # shape owner. All actual values subsequently come from pinned tensors.
    with torch.device("meta"):
        model = PalsModel(selected, roles=tuple(roles))
    return model.to_empty(device=device)


def _roles(membership):
    names = [row["name"] for row in membership["parameters"]]
    result = tuple(role for role in ("proposer", "critic", "validator")
                   if any(name.startswith("experts." + role + ".") for name in names))
    if not {"proposer", "critic"} <= set(result):
        raise ValueError("explicit common P/C parameter owners required")
    return result


def _nnuec_class(schema):
    from .nnuec_trainable import NnuecTrainableV1, NnuecTrainableV2
    if schema == "rz-nnuec-fp32-v1":
        return NnuecTrainableV1
    if schema == "rz-nnuec-fp32-v2":
        return NnuecTrainableV2
    raise ValueError("explicit registered NNUEC tensor schema required")


def _check_pc_state(model, state, torch):
    from .generation_data import fields
    expected = model.state_dict()
    fields(state, expected, "complete original P/C tensor state")
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.dtype != expected[name].dtype or tuple(value.shape) != tuple(expected[name].shape)
                or value.is_floating_point() and not bool(torch.all(torch.isfinite(value)))):
            raise ValueError("original P/C state shape/dtype/finite mismatch")


def _template_from_tensor_payload(payload, membership, torch, device):
    from .generation_data import fields
    from .rovent_tensor_training import validate_nnuec_tensor_checkpoint_state
    validate_nnuec_tensor_checkpoint_state(payload)
    metadata = fields(payload["model_metadata"], ("class", "config"), "source model metadata")
    # The registered class is an identifier, never dynamic code to import.
    if metadata["class"] != "rz_pals_model.model.PalsModel":
        raise ValueError("original P/C implementation class differs")
    model = _pc_template(metadata["config"], _roles(membership), torch, device=device)
    nnuec = payload["nnuec_metadata"]
    factory = _nnuec_class(nnuec["model_schema"])
    nnuec_state = {name[len("nnuec."):]: value for name, value in payload["model"].items() if name.startswith("nnuec.")}
    model.nnuec = factory.from_tensor_state(nnuec, nnuec_state, device=device)
    return model


def _initial_owner(startup, recipe, membership, spec, native_intake, reads, torch):
    from types import SimpleNamespace
    from .generation_data import fields
    from .rovent_tensor_training import _load_pinned_tensor_file, create_tensor_owner
    from .rovent_training_window import _read_admission, _journal, initial_training_journal
    registration = startup["registration"]
    pin = registration["pc_initial"]
    reads.account_external(pin["bytes"])
    payload, size = _load_pinned_tensor_file(pin["path"], pin["sha256"], pin["bytes"], torch)
    if size != pin["bytes"]:
        raise ValueError("original P/C checkpoint byte count changed")
    fields(payload, ("state_dict", "config", "seed", "schema", "training_steps", "trained"), "untrained P/C initialization")
    if (type(payload["training_steps"]) is not int or payload["training_steps"] != 0
            or payload["trained"] is not False or type(payload["seed"]) is not int):
        raise ValueError("initialize cannot relabel learned/diagnostic tensors")
    model = _pc_template(payload["config"], _roles(membership), torch, device=spec["device"])
    if payload["schema"] != model.config.model_semantics:
        raise ValueError("initial P/C semantics/configuration mismatch")
    _check_pc_state(model, payload["state_dict"], torch)
    model.load_state_dict(payload["state_dict"], strict=True)
    nnuec = reads.json(registration["nnuec_initial"], maximum=registration["max_read_bytes"])
    model.nnuec = _nnuec_class(nnuec["schema"]).from_checkpoint(nnuec, device=spec["device"])
    if model.nnuec.metadata_state()["training"] != {"kind": "untrained"}:
        raise ValueError("initialize requires an explicit untrained NNUEC declaration")
    probe = SimpleNamespace(recipe=recipe, stage_id=spec["stage_id"], inherited_input_sha256=[], inherited_origin_sha256=[])
    admission = _read_admission(probe, native_intake, spec)
    journal = initial_training_journal(recipe, spec, admission)
    probe.samplers, probe.completed = journal, dict.fromkeys(("proposer", "critic", "validator"), 0)
    def validate_sampler(value):
        candidate = copy.copy(probe)
        candidate.samplers = value
        _journal(candidate, spec, admission)
    import random
    import numpy as np
    random.seed(spec["training_seed"])
    np.random.seed(spec["training_seed"])
    devices = (int(spec["device"][5:]),) if spec["device"].startswith("cuda:") else ()
    torch.random.default_generator.manual_seed(spec["training_seed"])
    for device in devices:
        torch.cuda.default_generators[device].manual_seed(spec["training_seed"])
    owner = create_tensor_owner(model, recipe, membership, stage_id=spec["stage_id"], samplers=journal,
        bucket=8, sampler_validator=validate_sampler, cuda_devices=devices, execute=True)
    return owner


def _recover_owner(startup, recipe, membership, spec, native_intake, authority, reads, torch):
    from .rovent_tensor_training import _load_pinned_tensor_file
    from .rovent_training_window import stage_rovent_resume, stage_rovent_fork, stage_rovent_export
    registration, pin = startup["registration"], startup["registration"]["checkpoint"]
    reads.account_external(pin["bytes"])
    payload, size = _load_pinned_tensor_file(pin["path"], pin["sha256"], pin["bytes"], torch)
    if size != pin["bytes"]:
        raise ValueError("original checkpoint byte count changed")
    template = _template_from_tensor_payload(payload, membership, torch, spec["device"])
    if registration["checkpoint_mode"] == "resume":
        reads.account_external(pin["bytes"])
        factory = stage_rovent_export if registration["action"] == "export" else stage_rovent_resume
        staged = factory(pin["path"], pin["sha256"], template, recipe, membership,
            native_intake, spec, execution_authority=authority, max_file_bytes=pin["bytes"], execute=True)
    else:
        # The fork checks the original scratch source before its independent
        # weights-only reset read. Both original reads consume credit.
        reads.account_external(2 * pin["bytes"])
        staged = stage_rovent_fork(pin["path"], pin["sha256"], template, recipe, membership,
            native_intake, spec, execution_authority=authority, max_file_bytes=pin["bytes"], execute=True)
    return staged


def _export_request(startup, reads):
    from .generation_data import fields, uint
    registration = startup["registration"]
    request = reads.json(registration["export_request"], maximum=65536)
    fields(request, ("schema", "kind", "directory", "max_output_bytes"), "original export request")
    directory, root = Path(request["directory"]), Path(registration["output_root"])
    if (request["schema"] != "rz-rovent-original-export-request/1" or request["kind"] not in ("collection", "deployment")
            or not directory.is_absolute() or directory == root or root not in directory.parents
            or directory.parent.resolve(strict=True) != directory.parent or directory.exists() or directory.is_symlink()
            or not 1 <= uint(request["max_output_bytes"], min(registration["max_output_bytes"], 2 * 1024**3))):
        raise ValueError("bounded new original collection/deployment export request required")
    return request


def _rules_export_snapshot(startup, reads):
    raw, _identity = reads.read(startup["registration"]["rules_profile"], maximum=65536)
    path = Path(startup["registration"]["output_root"]) / "export-rules-profile.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(fd)
            os.fchmod(fd, 0o400)
    finally:
        os.close(fd)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return path, len(raw)


def _execute_export(startup, owner, staged, native_intake, spec, request, authority, reads, torch, *, clock, cancelled):
    from .generation_data import uint
    from .generation_export import _hash_file
    from .rovent_tensor_training import export_tensor_owner
    from .rovent_training_window import _read_admission, LearnedExportRequest
    started, initial_output = clock(), owner.cost["output_bytes"]
    lease, status = None, "failed"
    owner.physical_completion = "unknown"
    try:
        if cancelled() or clock() >= startup["execution_until_ns"]:
            raise TimeoutError("original export cancelled/expired before writer admission")
        # This lease permits only the registered export action. The parent
        # rejects sample/update/checkpoint RPC on this action.
        lease = authority.begin_window(owner, native_intake, spec, _read_admission(owner, native_intake, spec))
        rules, auxiliary_bytes = _rules_export_snapshot(startup, reads)
        available = request["max_output_bytes"] - auxiliary_bytes
        if available <= 0:
            raise ValueError("export cannot fit its pinned Rules snapshot")
        result = export_tensor_owner(owner, request["directory"], kind=request["kind"], rules_profile_json=rules,
            learned_request=LearnedExportRequest(native_intake, spec, authority),
            sampler_validator=staged.sampler_validator, max_output_bytes=available, execute=True)
        owner.cost["output_bytes"] = uint(owner.cost["output_bytes"] + auxiliary_bytes)
        receipt_path = Path(request["directory"]) / "rovent-export.json"
        reads.account_external(receipt_path.stat(follow_symlinks=False).st_size)
        identity, size = _hash_file(receipt_path, request["max_output_bytes"])
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
        "scope": "current_export_through_fence_and_end_ack_training_drain_cost_is_separate"}


def _write_terminal(path, result, maximum):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or path.parent.resolve(strict=True) != path.parent:
        raise ValueError("original plain terminal receipt path required")
    raw = (json.dumps(result, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    if len(raw) > maximum:
        raise ValueError("terminal receipt exceeds remaining output credit")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _error_chain_types(error):
    """Preserve failure boundaries without copying arbitrary exception text."""
    seen, result = set(), []
    while error is not None and id(error) not in seen and len(result) < 8:
        seen.add(id(error))
        result.append(type(error).__name__)
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return result


def original_rovent_child_main(argv=None, *, stdin=None, clock=time.monotonic_ns):
    args = _parser().parse_args(argv)
    if not args.execute:
        raise ValueError("original parent must explicitly authorize one finite child execution")
    raw = _read_startup(sys.stdin.buffer if stdin is None else stdin, args.startup_bytes)
    from .rovent_training_bootstrap import validate_startup, bootstrap_original_child
    startup = validate_startup(raw, expected_sha256=args.startup_sha256, expected_bytes=args.startup_bytes,
        execution_until_ns=args.execution_until_ns, whole_until_ns=args.whole_until_ns, now_ns=clock())
    bootstrap = bootstrap_original_child(startup, until_ns=args.execution_until_ns, clock=clock)
    readback, reads = bootstrap.report, bootstrap.budget
    from .rovent_training_authority import ParentSocketTrainerAuthority
    authority = ParentSocketTrainerAuthority.connect_registered(startup, raw_startup=raw,
        expected_sha256=args.startup_sha256, execute=True, clock=clock)
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
        recipe, membership, spec, rules = _metadata_inputs(startup, reads)
        request = _export_request(startup, reads) if registration["action"] == "export" else None
        native_intake = _intake(startup, reads)
        if cancelled or clock() >= args.execution_until_ns:
            raise TimeoutError("original child cancelled/expired before tensor admission")
        torch = _configure_runtime(spec)
        readback["imported_runtime"] = _verify_imported_runtime(startup, reads, torch)
        readback["precision_environment"] = {"precision": "fp32", "tf32": False,
            "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
        if registration["checkpoint_mode"] == "initialize":
            owner = _initial_owner(startup, recipe, membership, spec, native_intake, reads, torch)
        else:
            staged = _recover_owner(startup, recipe, membership, spec, native_intake, authority, reads, torch)
            owner = staged.owner
        start_output = owner.cost["output_bytes"]
        if (owner.completed != registration["expected_initial_completed"]
                or owner.ancestor_updates != registration["expected_ancestor_completed"]):
            raise ValueError("actual original checkpoint update ancestry differs from registration")
        from .artifacts import validate_rules_declaration
        rules_semantic = validate_rules_declaration(rules, owner.model.config)
        readback["rules_profile"] = {"descriptor_sha256": registration["rules_profile"]["sha256"],
            "rules_input_semantic_sha256": rules_semantic, "rules_input_profile": rules["rules_input_profile"],
            "rules_encoder_source_sha256": rules["rules_encoder_source_sha256"],
            "scope": "declared_model_input_contract_not_independent_Rules_execution_proof"}
        if registration["action"] == "export":
            result, export_pin, export_expense = _execute_export(startup, owner, staged, native_intake, spec,
                request, authority, reads, torch, clock=clock, cancelled=lambda: cancelled)
        else:
            from .rovent_training_window import train_rovent_window
            result = train_rovent_window(owner, native_intake, spec, registration["checkpoint_directory"],
                execution_authority=authority, cancelled=lambda: cancelled,
                execution_until_ns=args.execution_until_ns, execute=True)
        terminal = {"schema": RESULT_DOMAIN, "execution_id": startup["execution_id"], "window_id": startup["window_id"],
            "status": "completed", "result": result, "bootstrap": readback,
            "completed_updates": dict(owner.completed), "physical_completion": owner.physical_completion,
            "physical_process_retirement": "original_parent_join_required", "arena_eligible": False}
        if export_pin is not None:
            terminal.update(export_receipt=export_pin, export_expense=export_expense)
    except BaseException as error:
        if owner is not None:
            owner.quarantined = True
        terminal = {"schema": RESULT_DOMAIN, "execution_id": startup["execution_id"], "window_id": startup["window_id"],
            "status": "failed", "error_type": type(error).__name__,
            "error_chain_types": _error_chain_types(error),
            "completed_updates": None if owner is None else dict(owner.completed),
            "dispatched_updates": None if owner is None else dict(owner.dispatched),
            "cost": None if owner is None else dict(owner.cost),
            "physical_completion": "unknown" if owner is None else getattr(owner, "physical_completion", "unknown"),
            "quarantined": owner is not None, "bootstrap": readback,
            "physical_process_retirement": "original_parent_join_required", "arena_eligible": False}
    finally:
        authority.close()
        for name, previous in old_handlers.items():
            signal.signal(name, previous)
    readback["read_bytes"] = reads.spent
    if clock() >= args.whole_until_ns:
        raise TimeoutError("original cleanup deadline expired; partial evidence remains with original parent")
    remaining = startup["registration"]["max_output_bytes"] - (0 if owner is None else owner.cost["output_bytes"] - start_output)
    pin = _write_terminal(startup["registration"]["receipt_path"], terminal, remaining)
    # Native parent stdout remains a small terminal receipt; raw admission,
    # source inventory and full result stay in the bounded pinned file.
    summary = {"schema": RESULT_DOMAIN, "execution_id": startup["execution_id"], "window_id": startup["window_id"],
        "status": terminal["status"], "completed_updates": terminal.get("completed_updates"),
        "physical_completion": terminal["physical_completion"], "arena_eligible": False, "receipt": pin}
    if "error_type" in terminal:
        summary["error_type"] = terminal["error_type"]
        summary["error_chain_types"] = terminal["error_chain_types"]
    if export_pin is not None:
        summary["export_receipt"] = export_pin
        summary["result"] = result
    print(json.dumps(summary, sort_keys=True))
    return 0 if terminal["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(original_rovent_child_main())
