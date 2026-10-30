"""Actual-generation export authority, separate from diagnostic weight formats.

The JSON validators are dependency-free. They validate a separately pinned
receipt, not a claim inferred from ``trained=True`` in an ONNX manifest. Only
the explicit execution API may load the owned, preregistered checkpoint and
call the existing V3 exporter. Failed/partial output is preserved for diagnosis.
Implementing or testing this module does not execute that API.
"""
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from contextlib import contextmanager

from .artifacts import publish_noreplace
from .generation_data import ACTUAL_SOURCE, digest, fields, identifier, sha, uint, strict_json, _admit_coverage
from .config import MODEL_SCHEMA_V3, MODEL_SEMANTICS_V3, ENCODING_V3, ModelConfig
from .generation_chunks import MANIFEST_DOMAIN, MAX_CORPUS_BYTES, ChunkGenerationDataset
from .generation_training import (CHECKPOINT_DOMAIN, ELIGIBILITY, EXPORT_DOMAIN, SPEC_DOMAIN, UPDATES,
                                  GenerationLedger, generation_admission, parameter_owner,
                                  spec_digest, validate_spec, INFERENCE_PROFILE)

SUMMARY_DOMAIN = "rz-pals-verified-generation-checkpoint/1"
EVALUATION_PENDING = "evaluation_candidate_pending_gpu_acceptance"
OWNERS = ("shared", "proposer", "critic", "validator")
STATE_COMPONENTS = ("model", "optimizers", "schedulers", "rng", "sampler", "parameter_updates", "membership", "bucket")
COST_FIELDS = ("forward_calls", "backward_calls", "samples", "wall_time_ns", "partial_accumulation_ns",
               "replay_ns", "masked_groups", "output_bytes")


def _positive(value, maximum=MAX_CORPUS_BYTES):
    if uint(value, maximum) == 0:
        raise ValueError("actual artifact/cost must be positive")
    return value


def _file(value, expected_path):
    fields(value, ("file", "sha256", "bytes"), "actual artifact file pin")
    if value["file"] != expected_path:
        raise ValueError("actual artifact uses an unregistered file path")
    sha(value["sha256"])
    _positive(value["bytes"])


def _checkpoint(summary, spec):
    fields(summary, ("schema", "checkpoint_domain", "eligibility", "file", "sha256", "bytes", "phase",
                     "pending_microbatches", "completed_updates", "dispatched_updates", "ledger_sha256", "cost",
                     "state_sha256", "config_sha256", "frozen_pc_sha256", "frozen_pc_model_epoch", "context_epoch",
                     "nonzero_parameter_owners", "cuda_devices"), "verified actual checkpoint summary")
    if (summary["schema"] != SUMMARY_DOMAIN or summary["checkpoint_domain"] != CHECKPOINT_DOMAIN
            or summary["eligibility"] != ELIGIBILITY or summary["phase"] != "complete"
            or type(summary["pending_microbatches"]) is not int or summary["pending_microbatches"] != 0
            or summary["nonzero_parameter_owners"] != list(OWNERS)
            or summary["cuda_devices"] != [0] or type(summary["cuda_devices"][0]) is not int):
        raise ValueError("diagnostic/partial/all-masked checkpoint cannot become an actual candidate")
    _file({key: summary[key] for key in ("file", "sha256", "bytes")}, "checkpoint.pt")
    fields(summary["completed_updates"], tuple(UPDATES), "actual completed updates")
    fields(summary["dispatched_updates"], tuple(UPDATES), "actual dispatched updates")
    for role, expected in UPDATES.items():
        if type(summary["completed_updates"][role]) is not int or summary["completed_updates"][role] != expected:
            raise ValueError("actual first-generation update count is incomplete")
        if uint(summary["dispatched_updates"][role]) < expected:
            raise ValueError("actual dispatch ledger cannot refund failed work")
    fields(summary["cost"], COST_FIELDS, "actual accumulated cost")
    for name, value in summary["cost"].items():
        uint(value)
    # Each earned update required eight forward/backward calls. Individual
    # masked rows still cost work; this is not eight valid targets per update.
    required = sum(UPDATES.values()) * spec["accumulation"]
    if any(summary["cost"][name] < required for name in ("forward_calls", "backward_calls", "samples")):
        raise ValueError("all-masked/unexecuted work cannot earn actual optimizer updates")
    if summary["cost"]["backward_calls"] > summary["cost"]["samples"] or summary["cost"]["samples"] > summary["cost"]["forward_calls"]:
        raise ValueError("actual sample/backward cost exceeds dispatched neural forward work")
    _positive(summary["cost"]["wall_time_ns"], 2**64 - 1)
    if summary["cost"]["output_bytes"] > MAX_CORPUS_BYTES:
        raise ValueError("actual generation output exceeds managed cap")
    fields(summary["state_sha256"], STATE_COMPONENTS, "actual state evidence")
    for value in summary["state_sha256"].values():
        sha(value)
    for name in ("ledger_sha256", "config_sha256", "frozen_pc_sha256"):
        sha(summary[name])
    ledger = GenerationLedger(spec, completed=copy.deepcopy(summary["completed_updates"]),
        dispatched=copy.deepcopy(summary["dispatched_updates"]), phase="complete", next_role="validator",
        pending_microbatches=0, cost=copy.deepcopy(summary["cost"]))
    if summary["ledger_sha256"] != digest("rz-pals-generation-ledger-state/1", ledger.state()):
        raise ValueError("actual checkpoint summary changed the earned update/cost ledger")
    if summary["config_sha256"] != digest("rz-pals-generation-model-config/1", ModelConfig.for_profile(spec["profile"]).to_dict()):
        raise ValueError("actual checkpoint has another compact model configuration")
    if uint(summary["frozen_pc_model_epoch"]) <= spec["model_epoch"] or summary["context_epoch"] != spec["context_epoch"] or type(summary["context_epoch"]) is not int:
        raise ValueError("actual P/C freeze epoch is not separately registered")


def _admission(report, spec, phase, summary):
    fields(report, ("schema", "spec_sha256", "dataset_sha256", "frozen_pc_sha256", "phase", "coverage", "data_format",
                   "producer", "opening_splits", "model_epoch", "context_epoch", "native_chain_verified",
                   "inference_profile", "profile_exclusions", "actual_training_executed", "optimizer_updates", "arena_eligible"), "actual dataset admission")
    if (report["schema"] != SPEC_DOMAIN or report["spec_sha256"] != spec_digest(spec) or report["phase"] != phase
            or report["data_format"] != MANIFEST_DOMAIN or report["native_chain_verified"] is not True
            or report["actual_training_executed"] is not False or type(report["optimizer_updates"]) is not int
            or report["optimizer_updates"] != 0 or report["arena_eligible"] is not False):
        raise ValueError("actual export needs separately verified native-chain data, not diagnostic admission")
    from .generation_training import _typed_equal
    if not _typed_equal(report["inference_profile"], INFERENCE_PROFILE):
        raise ValueError("first generation requires registered Fresh B1; no Warm seed conversion")
    fields(report["profile_exclusions"], ("cross_split_records_excluded", "cross_split_identities", "duplicate_inputs_excluded",
                                        "duplicate_semantic_inputs_excluded", "warm_seed_inputs_excluded"), "actual optimization profile exclusions")
    for count in report["profile_exclusions"].values():
        uint(count)
    sha(report["dataset_sha256"])
    if phase == "pc" and (report["dataset_sha256"] != spec["dataset_sha256"] or report["frozen_pc_sha256"] is not None):
        raise ValueError("P/C actual corpus differs from registered G0 spec")
    if phase == "v" and report["frozen_pc_sha256"] != summary["frozen_pc_sha256"]:
        raise ValueError("V corpus does not belong to exact frozen P/C")
    expected_epoch = spec["model_epoch"] if phase == "pc" else summary["frozen_pc_model_epoch"]
    if type(report["model_epoch"]) is not int or report["model_epoch"] != expected_epoch or type(report["context_epoch"]) is not int or report["context_epoch"] != spec["context_epoch"]:
        raise ValueError("actual data/model/context epoch mismatch")
    producer = report["producer"]
    fields(producer, ("schema", "source_kind", "source_sha256", "model_sha256", "runtime_sha256", "teacher_binary_sha256",
                      "teacher_profile_sha256", "opening_split_sha256", "registered_before_outcomes", "actual_game_ids"), "actual dataset authority")
    if producer["schema"] != "rz-pals-actual-collection-authority/2" or producer["source_kind"] != ACTUAL_SOURCE or producer["registered_before_outcomes"] is not True:
        raise ValueError("actual export cannot admit renamed diagnostic authority")
    for name in ("source_sha256", "model_sha256", "runtime_sha256", "teacher_binary_sha256", "teacher_profile_sha256", "opening_split_sha256"):
        sha(producer[name])
    expected_model = spec["model_sha256"] if phase == "pc" else summary["frozen_pc_sha256"]
    if (producer["source_sha256"] != spec["source_sha256"] or producer["runtime_sha256"] != spec["runtime_sha256"]
            or producer["opening_split_sha256"] != spec["split_sha256"] or producer["model_sha256"] != expected_model):
        raise ValueError("actual source/runtime/model/split provenance mismatch")
    games = producer["actual_game_ids"]
    if type(games) is not list or not 1 <= len(games) <= 8192:
        raise ValueError("actual corpus game identities are not unique/bounded")
    for game in games:
        identifier(game)
    if len(set(games)) != len(games):
        raise ValueError("actual corpus game identities are not unique/bounded")
    if phase == "pc" and len(games) != 128:
        raise ValueError("first actual P/C generation must retain 128 collected games")
    splits = report["opening_splits"]
    if type(splits) is not dict or len(splits) != 64 or {name: list(splits.values()).count(name) for name in ("train", "validation", "holdout")} != {"train": 48, "validation": 8, "holdout": 8}:
        raise ValueError("actual corpus split was not preregistered 48/8/8")
    for opening in splits:
        identifier(opening)
    if digest("rz-pals-opening-split/1", splits) != spec["split_sha256"]:
        raise ValueError("actual opening mapping differs from registered split pin")
    if type(report["coverage"]) is not dict or len(report["coverage"]) > 128:
        raise ValueError("bounded actual coverage evidence required")
    for key, value in report["coverage"].items():
        identifier(key)
        uint(value)
    _admit_coverage(report["coverage"], phase)
    if phase == "v" and report["coverage"].get("validator_unique_inputs", 0) > 8192:
        raise ValueError("actual V collection context cap exceeded")


def _artifacts(value, kind):
    fields(value, ("roles", "validator_present", "manifest", "graphs"), "actual role export artifacts")
    include_v = kind == "collection"
    if value["roles"] != ["proposer", "critic"] + (["validator"] if include_v else []) or value["validator_present"] is not include_v:
        raise ValueError("Deployment must contain P/C and no V bank")
    _file(value["manifest"], kind + "/export.json")
    expected = [("public", "public_memory.onnx"), ("shared_pc", "shared_pc_if.onnx")] + ([("validator", "role_validator.onnx")] if include_v else [])
    if type(value["graphs"]) is not list or len(value["graphs"]) != len(expected):
        raise ValueError("actual export graph ownership mismatch")
    for graph, (role, filename) in zip(value["graphs"], expected):
        fields(graph, ("role", "file", "sha256", "bytes"), "actual graph file pin")
        if graph["role"] != role:
            raise ValueError("actual graph role order/ownership mismatch")
        _file({key: graph[key] for key in ("file", "sha256", "bytes")}, kind + "/" + filename)


def provenance_digest(value):
    return digest(EXPORT_DOMAIN, value)


def validate_actual_generation_provenance(value, *, expected_sha256, expected_spec_sha256=None, expected_checkpoint_sha256=None):
    """Validate a preregistered JSON receipt without loading a neural library.

    File pins are verified by ``verify_actual_generation_artifacts``. This
    structural check alone is not actual training/GPU/arena acceptance.
    """
    sha(expected_sha256)
    fields(value, ("schema", "source_kind", "eligibility", "spec", "spec_sha256", "generation_id", "source_sha256", "runtime_sha256",
                   "checkpoint", "pc_admission", "v_admission", "collection", "deployment", "numeric_status", "cuda_status", "arena_accepted"), "actual generation provenance")
    validate_spec(value["spec"])
    spec = value["spec"]
    if (value["schema"] != EXPORT_DOMAIN or value["source_kind"] != ACTUAL_SOURCE or value["eligibility"] != EVALUATION_PENDING
            or value["spec_sha256"] != spec_digest(spec) or value["generation_id"] != spec["generation_id"]
            or value["source_sha256"] != spec["source_sha256"] or value["runtime_sha256"] != spec["runtime_sha256"]
            or value["numeric_status"] != "not_run" or value["cuda_status"] != "not_run" or value["arena_accepted"] is not False):
        raise ValueError("unaccepted actual export source/domain/quality mismatch")
    _checkpoint(value["checkpoint"], spec)
    for phase in ("pc", "v"):
        _admission(value[phase + "_admission"], spec, phase, value["checkpoint"])
    if (value["pc_admission"]["opening_splits"] != value["v_admission"]["opening_splits"]
            or any(value["pc_admission"]["producer"][key] != value["v_admission"]["producer"][key] for key in ("teacher_binary_sha256", "teacher_profile_sha256"))):
        raise ValueError("actual frozen generation changed split/teacher configuration")
    for kind in ("collection", "deployment"):
        _artifacts(value[kind], kind)
    if expected_spec_sha256 is not None and sha(expected_spec_sha256) != value["spec_sha256"]:
        raise ValueError("actual generation spec differs from externally registered pin")
    if expected_checkpoint_sha256 is not None and sha(expected_checkpoint_sha256) != value["checkpoint"]["sha256"]:
        raise ValueError("actual checkpoint differs from externally registered pin")
    if provenance_digest(value) != expected_sha256:
        raise ValueError("actual export provenance changed after registration")
    total = value["checkpoint"]["bytes"] + sum(value[kind]["manifest"]["bytes"] + sum(graph["bytes"] for graph in value[kind]["graphs"]) for kind in ("collection", "deployment"))
    if total > MAX_CORPUS_BYTES:
        raise ValueError("actual independent export exceeds managed output cap")
    return copy.deepcopy(value)


def _owned_file(root, relative):
    parts = PurePosixPath(relative)
    if parts.is_absolute() or str(parts) != relative or "\\" in relative or any(p in (".", "..") for p in parts.parts):
        raise ValueError("actual artifact pin escapes owned output")
    current = root
    for component in parts.parts:
        current = current / component
        if current.is_symlink() or current.exists() and getattr(current.stat(), "st_file_attributes", 0) & 0x400:
            raise ValueError("linked/reparse actual artifact cannot be verified")
    target = current.resolve(strict=True)
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError("actual artifact is not an owned regular file")
    return target


def _hash_file(path, maximum=MAX_CORPUS_BYTES, *, snapshot=None, read_accounting=None):
    path = Path(path)
    if path.is_symlink() or getattr(path.stat(), "st_file_attributes", 0) & 0x400:
        raise ValueError("linked/reparse actual file cannot be pinned")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb", buffering=0 if read_accounting is not None else -1) as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= maximum:
            raise ValueError("actual file must be one bounded unlinked regular file")
        lease = None if read_accounting is None else read_accounting.reserve_external(before.st_size + 1)
        try:
            result, count = hashlib.sha256(), 0
            while True:
                amount = 1024 * 1024 if lease is None else min(1024 * 1024, before.st_size + 1 - count)
                block = stream.read(amount)
                if read_accounting is not None:
                    read_accounting.charge_external(lease, len(block))
                if not block:
                    break
                count += len(block)
                if count > maximum or lease is not None and count > before.st_size:
                    raise ValueError("actual file grew past registered output cap")
                result.update(block)
                if snapshot is not None:
                    snapshot.write(block)
            after = os.fstat(stream.fileno())
            now = Path(path).stat()
            identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if identity(before) != identity(after) or identity(after) != identity(now) or count != before.st_size:
                raise ValueError("actual file changed during verification")
            if read_accounting is not None:
                read_accounting.finish_external(lease, expected_bytes=count)
        except BaseException:
            if lease is not None:
                read_accounting.fail_external(lease)
            raise
    return result.hexdigest(), count


@contextmanager
def _pinned_checkpoint_snapshot(path, expected_sha256):
    """Decode and copy only a private byte snapshot whose hash was verified.

    A source path replaced between hash and decode cannot substitute bytes.
    The snapshot size and source are both counted while live; growth is stopped
    before writing a block beyond the bounded snapshot budget.
    """
    with tempfile.TemporaryFile(mode="w+b", prefix="rz-generation-pinned-", dir=path.parent) as snapshot:
        actual, size = _hash_file(path, MAX_CORPUS_BYTES // 2, snapshot=snapshot)
        if actual != expected_sha256:
            raise ValueError("actual checkpoint failed separately registered binary pin")
        snapshot.flush()
        snapshot.seek(0)
        yield snapshot, size


def _pinned_json(path, expected_sha256, expected_bytes, maximum=4 * 1024**2):
    """Hash exactly the bounded bytes that will be parsed, never reopen them."""
    import io
    snapshot = io.BytesIO()
    actual_sha256, actual_bytes = _hash_file(path, maximum, snapshot=snapshot)
    if (actual_sha256, actual_bytes) != (expected_sha256, expected_bytes):
        raise ValueError("pinned metadata bytes changed before parsing")
    return strict_json(snapshot.getvalue().decode("utf-8", "strict"))


def read_actual_generation_provenance(path, *, expected_file_sha256, expected_spec_sha256=None, expected_checkpoint_sha256=None):
    """Read the externally pinned raw JSON file used by non-Python consumers.

    Raw file hashes are language-independent. The semantic helper hash remains
    a Python-side canonical identity and is not a substitute for this file pin.
    """
    sha(expected_file_sha256)
    target = Path(path)
    if target.name != "actual-generation.json":
        raise ValueError("actual generation reader requires its registered provenance filename")
    # Size is read only to bound the stream, not to authenticate its bytes.
    size = target.stat().st_size
    value = _pinned_json(target, expected_file_sha256, size)
    return validate_actual_generation_provenance(value, expected_sha256=provenance_digest(value),
        expected_spec_sha256=expected_spec_sha256, expected_checkpoint_sha256=expected_checkpoint_sha256)


def verify_actual_generation_artifacts(directory, value, *, expected_sha256, expected_spec_sha256=None, expected_checkpoint_sha256=None):
    receipt = validate_actual_generation_provenance(value, expected_sha256=expected_sha256,
                expected_spec_sha256=expected_spec_sha256, expected_checkpoint_sha256=expected_checkpoint_sha256)
    raw_root = Path(directory)
    if raw_root.is_symlink() or getattr(raw_root.stat(), "st_file_attributes", 0) & 0x400:
        raise ValueError("actual artifact root cannot be a link/reparse path")
    root = raw_root.resolve(strict=True)
    pins = [{key: receipt["checkpoint"][key] for key in ("file", "sha256", "bytes")}]
    for kind in ("collection", "deployment"):
        pins.append(receipt[kind]["manifest"])
        pins.extend(receipt[kind]["graphs"])
    for pin in pins:
        actual_sha, size = _hash_file(_owned_file(root, pin["file"]))
        if (actual_sha, size) != (pin["sha256"], pin["bytes"]):
            raise ValueError("actual artifact binary differs from its preregistered receipt")
    for kind in ("collection", "deployment"):
        pin = receipt[kind]["manifest"]
        if pin["bytes"] > 4 * 1024**2:
            raise ValueError("actual role manifest exceeded bounded metadata limit")
        manifest = _pinned_json(_owned_file(root, pin["file"]), pin["sha256"], pin["bytes"])
        config = ModelConfig.for_profile(receipt["spec"]["profile"])
        required = {"schema": MODEL_SCHEMA_V3, "model_semantics": MODEL_SEMANTICS_V3, "encoding_schema": ENCODING_V3,
                    "model_profile": config.profile, "config": config.to_dict(),
                    "checkpoint_sha256": receipt["checkpoint"]["sha256"], "trained": True,
                    "training_steps": sum(UPDATES.values()), "roles": receipt[kind]["roles"],
                    "validator_present": receipt[kind]["validator_present"], "precision": "fp32", "tf32": False,
                    "numeric_status": "not_run", "cuda_status": "not_run"}
        from .generation_training import _typed_equal
        if type(manifest) is not dict or any(not _typed_equal(manifest.get(key), expected) for key, expected in required.items()):
            raise ValueError("actual role manifest does not match completed generation provenance")
        graph_pins = receipt[kind]["graphs"]
        if type(manifest.get("graphs")) is not list or len(manifest["graphs"]) != len(graph_pins):
            raise ValueError("actual manifest graph ownership differs from receipt")
        for declaration, graph_pin in zip(manifest["graphs"], graph_pins):
            if (type(declaration) is not dict or declaration.get("role") != graph_pin["role"]
                    or declaration.get("file") != PurePosixPath(graph_pin["file"]).name
                    or declaration.get("sha256") != graph_pin["sha256"]):
                raise ValueError("actual manifest graph/file pin join failed")
    return receipt


def _atomic_json(path, value, *, output_guard=None):
    """Exclusive publication; on failure leave the owned partial file visible."""
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError("actual generation evidence already exists")
    if output_guard is not None:
        raw = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        output_guard(len(raw))
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    publish_noreplace(temporary, path)


def _summary(payload, ledger, config, checkpoint_sha256, checkpoint_bytes, frozen_pc_model_epoch):
    from .nonzero_training import _state_digest
    owners = {parameter_owner(name) for name, count in payload["parameter_updates"].items() if count > 0}
    if owners != set(OWNERS):
        raise ValueError("actual generation lacks an updated shared/private parameter owner")
    return {"schema": SUMMARY_DOMAIN, "checkpoint_domain": CHECKPOINT_DOMAIN, "eligibility": ELIGIBILITY,
            "file": "checkpoint.pt", "sha256": checkpoint_sha256, "bytes": checkpoint_bytes,
            "phase": ledger.phase, "pending_microbatches": ledger.pending_microbatches,
            "completed_updates": dict(ledger.completed), "dispatched_updates": dict(ledger.dispatched),
            "ledger_sha256": digest("rz-pals-generation-ledger-state/1", ledger.state()), "cost": dict(ledger.cost),
            "state_sha256": {name: _state_digest(payload[name]) for name in STATE_COMPONENTS},
            "config_sha256": digest("rz-pals-generation-model-config/1", config),
            "frozen_pc_sha256": payload["frozen_pc_sha256"], "frozen_pc_model_epoch": frozen_pc_model_epoch,
            "context_epoch": ledger.spec["context_epoch"], "nonzero_parameter_owners": list(OWNERS), "cuda_devices": [0]}


def _export_descriptors(root, kind, manifest):
    graphs = []
    for graph in manifest["graphs"]:
        relative = kind + "/" + graph["file"]
        identity, size = _hash_file(_owned_file(root, relative))
        if identity != graph["sha256"]:
            raise ValueError("actual graph changed after native exporter publication")
        graphs.append({"role": graph["role"], "file": relative, "sha256": identity, "bytes": size})
    relative = kind + "/export.json"
    identity, size = _hash_file(_owned_file(root, relative))
    return {"roles": list(manifest["roles"]), "validator_present": manifest["validator_present"],
            "manifest": {"file": relative, "sha256": identity, "bytes": size}, "graphs": graphs}


def export_actual_generation(checkpoint_path, *, expected_checkpoint_sha256, model, optimizers, spec,
                             pc_dataset, v_dataset, frozen_pc_model_epoch, sampler_validator,
                             rules_profile_json, directory, execute=False, output_guard=None):
    """Explicit owned-checkpoint export; never dispatches a training update.

    The caller must supply its already instantiated, eval-mode G1 model and the
    validated optimizer owner; this function does not construct a model, load
    old diagnostic formats, mutate model/RNG, or waive numerical acceptance.
    The checkpoint is an owned torch archive with a separately registered hash.
    It is loaded only after that hash is verified and execute=True is explicit.
    """
    if not execute:
        raise ValueError("actual generation export requires explicit authorization")
    validate_spec(spec)
    sha(expected_checkpoint_sha256)
    if type(pc_dataset) is not ChunkGenerationDataset or type(v_dataset) is not ChunkGenerationDataset:
        raise ValueError("actual export requires native-chain chunk corpus owners")
    source = Path(checkpoint_path)
    if source.suffix != ".pt" or source.name.lower().startswith(".env"):
        raise ValueError("only an owned actual .pt checkpoint is an export input")
    if source.is_symlink() or getattr(source.stat(), "st_file_attributes", 0) & 0x400:
        raise ValueError("actual checkpoint cannot be a link/reparse file")
    source = source.resolve(strict=True)
    identity = expected_checkpoint_sha256
    # Retain a private pinned snapshot through both decode and independent copy.
    # The owned snapshot is never made a public artifact before validation.
    with _pinned_checkpoint_snapshot(source, identity) as (snapshot, size):
        import torch
        payload = torch.load(snapshot, map_location="cpu", weights_only=True)
        return _export_validated_snapshot(snapshot, size, identity, payload, model, optimizers, spec,
            pc_dataset, v_dataset, frozen_pc_model_epoch, sampler_validator, rules_profile_json, directory, output_guard)


def _export_validated_snapshot(snapshot, size, identity, payload, model, optimizers, spec, pc_dataset, v_dataset,
                               frozen_pc_model_epoch, sampler_validator, rules_profile_json, directory, output_guard=None):
    from .generation_training import validate_checkpoint
    from .nonzero_training import _state_digest
    ledger = validate_checkpoint(payload, model, optimizers, spec, sampler_validator)
    if ledger.phase != "complete" or ledger.pending_microbatches:
        raise ValueError("incomplete actual checkpoint cannot export a candidate")
    model.config.validate()
    if model.config.profile != "virtual_review_v1" or model.training is not False or optimizers.model is not model or optimizers.spec != spec:
        raise ValueError("actual export requires the matching eval-mode compact model owner")
    if _state_digest(model.state_dict()) != _state_digest(payload["model"]):
        raise ValueError("caller model differs from completed actual checkpoint")
    pc_report = generation_admission(spec, pc_dataset, "pc")
    v_report = generation_admission(spec, v_dataset, "v", frozen_pc_sha256=payload["frozen_pc_sha256"], model_epoch=frozen_pc_model_epoch)
    summary = _summary(payload, ledger, model.config.to_dict(), identity, size, frozen_pc_model_epoch)
    _checkpoint(summary, spec)
    from .artifacts import export_virtual_checkpoint, read_rules_profile
    read_rules_profile(rules_profile_json, model.config)  # Fail before creating output.
    root = Path(directory).expanduser().resolve()
    repository = next((p for p in Path(__file__).resolve().parents if (p / ".git").exists()), None)
    if repository is not None and (root == repository or root.is_relative_to(repository)):
        raise ValueError("actual generation export must be outside repository")
    if root.exists():
        raise FileExistsError("actual generation output already exists; preserve prior work")
    if ledger.cost["output_bytes"] + 2 * size > MAX_CORPUS_BYTES:
        raise ValueError("independent actual checkpoint exceeds remaining managed cap")
    root.mkdir(parents=True, exist_ok=False)
    try:
        _atomic_json(root / "generation-export-owner.json", {"schema": EXPORT_DOMAIN, "owner": "rz-pals-actual-generation",
                     "generation_id": spec["generation_id"], "source_checkpoint_sha256": identity}, output_guard=output_guard)
        temporary = root / "checkpoint.pt.partial"
        snapshot.seek(0)
        if output_guard is not None:
            output_guard(size)
        with temporary.open("xb") as writer:
            copied = hashlib.sha256()
            written = 0
            for block in iter(lambda: snapshot.read(1024 * 1024), b""):
                written += len(block)
                if written > size or ledger.cost["output_bytes"] + size + written > MAX_CORPUS_BYTES:
                    raise ValueError("pinned checkpoint copy exceeded registered size/output cap")
                if output_guard is not None:
                    output_guard(len(block))
                copied.update(block)
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        if copied.hexdigest() != identity or temporary.stat().st_size != size:
            raise ValueError("independent actual checkpoint copy failed integrity")
        publish_noreplace(temporary, root / "checkpoint.pt")
        metadata = {"config": model.config.to_dict(), "checkpoint_sha256": identity,
                    "trained": True, "training_steps": sum(ledger.completed.values())}
        exports = {}
        for kind, include_v in (("collection", True), ("deployment", False)):
            manifest = export_virtual_checkpoint(model, metadata, root / kind, include_validator=include_v,
                rules_profile_json=rules_profile_json, output_guard=output_guard)
            exports[kind] = _export_descriptors(root, kind, manifest)
            size_used = size + sum(entry["manifest"]["bytes"] + sum(g["bytes"] for g in entry["graphs"]) for entry in exports.values())
            if ledger.cost["output_bytes"] + size + size_used > MAX_CORPUS_BYTES:
                raise ValueError("actual role export exceeded managed staging cap")
        receipt = {"schema": EXPORT_DOMAIN, "source_kind": ACTUAL_SOURCE, "eligibility": EVALUATION_PENDING,
                   "spec": copy.deepcopy(spec), "spec_sha256": spec_digest(spec), "generation_id": spec["generation_id"],
                   "source_sha256": spec["source_sha256"], "runtime_sha256": spec["runtime_sha256"], "checkpoint": summary,
                   "pc_admission": pc_report, "v_admission": v_report, **exports,
                   "numeric_status": "not_run", "cuda_status": "not_run", "arena_accepted": False}
        pin = provenance_digest(receipt)
        verify_actual_generation_artifacts(root, receipt, expected_sha256=pin,
             expected_spec_sha256=spec_digest(spec), expected_checkpoint_sha256=identity)
        _atomic_json(root / "actual-generation.json", receipt, output_guard=output_guard)
        file_sha256, file_bytes = _hash_file(root / "actual-generation.json", 4 * 1024**2)
        return {"provenance": receipt, "provenance_sha256": pin, "directory": str(root),
                "provenance_file_sha256": file_sha256, "provenance_file_bytes": file_bytes,
                "actual_export_executed": True, "optimizer_updates_executed": 0, "arena_accepted": False}
    except BaseException as error:
        # Preserve both original evidence and any owned partial artifacts.
        failure = root / "export-incomplete.json"
        if not failure.exists() and not failure.with_name(failure.name + ".partial").exists():
            try:
                _atomic_json(failure, {"schema": EXPORT_DOMAIN, "status": "incomplete", "error_type": type(error).__name__,
                             "generation_id": spec["generation_id"], "source_checkpoint_sha256": identity, "arena_accepted": False}, output_guard=output_guard)
            except BaseException as evidence_error:
                # Exhausted credit cannot erase the original export failure.
                # Existing partial files and the child terminal remain owned.
                raise error from evidence_error
        raise
