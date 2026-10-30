"""Separate N05 utility tensor ownership, checkpoint and export lifecycle.

Import is stdlib-only. Tensor materialization requires explicit ``execute``;
no function in this module calls forward, backward, or optimizer.step. The
initial numerical declaration and current update journal have different
meanings. Only a live utility trainer authority may attest nonzero updates or
authorize a learned export; ordinary JSON pins cannot do so.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import os
from pathlib import Path
import tempfile
import threading

from .generation_data import digest as metadata_digest, fields, sha, uint
from .generation_export import COST_FIELDS
from .generation_training import _typed_equal
from .rovent_utility_training import (PARAMETER_FIELDS, PREPARATION_DOMAIN, COHORT_PREPARATION_DOMAIN,
    prepare_training, prepare_cohort_training, validate_recipe, validate_training_preparation)


CHECKPOINT_DOMAIN = "rz-rovent-utility-tensor-checkpoint/1"
STATE_DOMAIN = "rz-rovent-utility-derived-training-state/1"
EXPORT_DOMAIN = "rz-rovent-utility-lifecycle-export/1"
PROJECTION_DOMAIN = "rz-rovent-utility-tensor-projection/1"
MAX_CHECKPOINT_BYTES = 256 * 1024**2
MAX_PROJECTION_BYTES = 16 * 1024**2
MAX_UPDATES = 131072
_LOCK = threading.RLock()
_FORK_FROM_PINNED_SOURCE = object()
_RESUME_COST_STATE_DOMAIN = "rz-rovent-utility-resume-cost-state/1"


def _require_tensor_execution(execute):
    """Deny before consulting an owner, path, source family or neural import."""
    if execute is not True:
        raise ValueError("explicit utility tensor execution required; no implicit neural import")


def _runtime(execute):
    _require_tensor_execution(execute)
    from .rovent_tensor_training import _runtime as load_runtime
    return load_runtime(True)


def _plain(value):
    from .rovent_tensor_training import _validate_json_state
    _validate_json_state(value)
    return copy.deepcopy(value)


def _metadata(model):
    if not all(callable(getattr(model, name, None)) for name in
               ("metadata_state", "validate_metadata_state", "restore_metadata_state",
                "parameter_spec", "export_checkpoint", "write_checkpoint", "current_weights_sha256")):
        raise ValueError("actual separate utility trainable lifecycle required")
    state = _plain(model.metadata_state())
    model.validate_metadata_state(copy.deepcopy(state))
    if state.get("schema") != "rz-nnuec-utility-torch-metadata/1":
        raise ValueError("explicit utility metadata domain required")
    return state


def _validate_actual(model, preparation, torch, *, numerical_parts=None):
    from .nnuec_utility_trainable import NnuecUtilityTrainableV1
    if type(model) is not NnuecUtilityTrainableV1:
        raise ValueError("actual concrete utility tensor owner required")
    if numerical_parts is None:
        if preparation.get("schema") not in (PREPARATION_DOMAIN, COHORT_PREPARATION_DOMAIN):
            raise ValueError("explicit separate utility preparation required")
        # The original named path remains PREP1/PREP2 only.
        checked = validate_training_preparation(preparation)
    else:
        from .rovent_utility_intervention_numerical import ValidatedUtilityNumericalParts
        if type(numerical_parts) is not ValidatedUtilityNumericalParts:
            raise ValueError("sealed original intervention numerical parts required")
        checked = numerical_parts.preparation
    if not _typed_equal(checked, preparation):
        raise ValueError("utility preparation seal changed")
    metadata = _metadata(model)
    registered = checked["registration"]
    frozen = registered["frozen_base"]
    if (metadata["model_schema"] != registered["utility"]["schema"]
            or metadata["model_semantics"] != registered["utility"]["semantics"]
            or metadata["base_schema"] != frozen["schema"]
            or metadata["base_semantics"] != frozen["semantics"]
            or metadata["base_weights_sha256"] != frozen["weights_sha256"]
            or metadata["frozen_base_config"] != frozen["config"]):
        raise ValueError("actual utility metadata differs from frozen source registration")
    spec = checked["parameter_spec"]
    mapping = spec["wire_to_parameter"]
    named = list(model.named_parameters(remove_duplicate=False))
    if (len(named) != 4 or len({name for name, _ in named}) != 4
            or {name for name, _ in named} != set(mapping.values())):
        raise ValueError("actual utility named parameters must contain exactly four registered private arrays")
    parameters = dict(named)
    storages, identities, device = set(), set(), None
    actual_spec = model.parameter_spec()
    if type(actual_spec) is not dict or set(actual_spec) != set(mapping.values()):
        raise ValueError("actual utility parameter spec differs from four registered names")
    for wire in PARAMETER_FIELDS:
        name, expected = mapping[wire], spec["parameters"][wire]
        value = parameters[name]
        item = actual_spec[name]
        if (tuple(value.shape) != tuple(expected["shape"]) or value.dtype != torch.float32
                or value.layout != torch.strided or not value.is_contiguous()
                or value.device.type not in ("cpu", "cuda") or value.requires_grad is not True
                or tuple(item["shape"]) != tuple(expected["shape"]) or item["dtype"] != "float32"
                or not bool(torch.isfinite(value).all().item())
                or bool((torch.abs(value) > 1000.0).any().item())):
            raise ValueError("actual utility private tensor shape/dtype/layout/freeze/value mismatch")
        storage = (str(value.device), value.untyped_storage().data_ptr())
        if id(value) in identities or storage in storages:
            raise ValueError("utility parameter object/storage alias is forbidden")
        identities.add(id(value)); storages.add(storage)
        if device is None:
            device = value.device
        if value.device != device:
            raise ValueError("utility parameters require one registered device")
    # The frozen base is immutable reference data, never a hidden nn.Module
    # parameter or optimizer member. Its mapping pin is in metadata above.
    if any(name.startswith(("base.", "p.", "c.", "nnuec.")) for name in parameters):
        raise ValueError("frozen NNUEC/P/C parameters cannot become utility optimizer members")
    return parameters, metadata


def _fresh_optimizer(parameters, preparation, torch):
    recipe = validate_recipe(preparation["recipe"])
    config = recipe["optimizer"]
    names = [preparation["parameter_spec"]["wire_to_parameter"][wire] for wire in PARAMETER_FIELDS]
    optimizer = torch.optim.AdamW([{"params": [parameters[name] for name in names],
        "group_name": recipe["optimizer_owner"], "weight_decay": float(config["weight_decay"])}],
        lr=float(config["learning_rate"]), betas=tuple(float(value) for value in config["betas"]),
        eps=float(config["epsilon"]), foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return optimizer, scheduler, tuple(names)


def _device_roster(parameters, cuda_devices):
    devices = tuple(cuda_devices)
    if any(type(device) is not int or device < 0 for device in devices) or len(set(devices)) != len(devices):
        raise ValueError("explicit unique utility CUDA RNG device IDs required")
    actual = next(iter(parameters.values())).device
    expected = () if actual.type == "cpu" else (actual.index,)
    if devices != expected:
        raise ValueError("utility RNG roster differs from actual private tensor device")


@dataclass
class UtilityTensorOwner:
    model: object
    preparation: dict
    metadata: dict
    optimizer: object
    scheduler: object
    parameter_names: tuple
    rng: dict
    sampler: dict
    cuda_devices: tuple
    completed_updates: int = 0
    dispatched_updates: int = 0
    ancestor_updates: int = 0
    pending_microbatches: int = 0
    cost: dict = field(default_factory=lambda: dict.fromkeys(COST_FIELDS, 0))
    ancestor_cost: dict = field(default_factory=lambda: dict.fromkeys(COST_FIELDS, 0))
    completion_receipts: list = field(default_factory=list)
    ancestor_completion_sha256: list = field(default_factory=list)
    parent_checkpoint_sha256: str | None = None
    # Original pinned load identity is a live caller fact, separate from the
    # ancestral fork pointer carried in checkpoint bytes.
    source_checkpoint_sha256: str | None = None
    source_payload_state_sha256: str | None = None
    source_snapshot_cost_observation: dict | None = None
    source_snapshot_cost: dict | None = None
    source_snapshot_ancestor_cost: dict | None = None
    # A local staged-load observation, not serialized checkpoint data or a
    # substitute for the live parent's cost/closure authority. It seals the
    # actual admitted delta separately from the unchanged snapshot preimage.
    source_admitted_cost_state_sha256: str | None = None
    # Post-snapshot original parent costs are separately admitted. An unknown
    # tail remains unknown and is never converted to a numeric zero delta.
    source_cost_observation: dict | None = None
    physical_completion: str = "unknown"
    quarantined: bool = False

    def total_updates(self):
        return uint(self.ancestor_updates + self.completed_updates, MAX_UPDATES)

    def total_cost(self):
        return {name: uint(self.ancestor_cost[name] + self.cost[name]) for name in COST_FIELDS}


def _counts(owner):
    for value in (owner.completed_updates, owner.dispatched_updates, owner.ancestor_updates):
        uint(value, MAX_UPDATES)
    if owner.completed_updates != owner.dispatched_updates or owner.pending_microbatches:
        raise ValueError("utility checkpoint requires a completed optimizer boundary")
    if owner.total_updates() > MAX_UPDATES:
        raise ValueError("bounded total utility updates required")
    fields(owner.cost, COST_FIELDS, "utility current cost")
    fields(owner.ancestor_cost, COST_FIELDS, "utility ancestor cost")
    for value in (*owner.cost.values(), *owner.ancestor_cost.values()):
        uint(value)
    if len(owner.completion_receipts) != owner.completed_updates:
        raise ValueError("each completed utility update requires one original completion receipt")
    seen = set()
    for receipt in owner.completion_receipts:
        fields(receipt, ("sha256", "bytes"), "original utility completion receipt pin")
        sha(receipt["sha256"])
        if not uint(receipt["bytes"]) or receipt["sha256"] in seen:
            raise ValueError("distinct original utility completion receipts required")
        seen.add(receipt["sha256"])
    if type(owner.ancestor_completion_sha256) is not list or len(owner.ancestor_completion_sha256) != owner.ancestor_updates:
        raise ValueError("complete utility ancestor update chain required")
    for value in owner.ancestor_completion_sha256:
        sha(value)
        if value in seen:
            raise ValueError("repeated utility ancestor completion receipt")
        seen.add(value)
    if owner.parent_checkpoint_sha256 is not None:
        sha(owner.parent_checkpoint_sha256)


def _resume_cost_state_sha256(owner):
    return metadata_digest(_RESUME_COST_STATE_DOMAIN, {
        "cost": _plain(owner.cost), "ancestor_cost": _plain(owner.ancestor_cost),
        "source_cost_observation": _plain(owner.source_cost_observation)})


def _optimizer_state(owner, torch, clone):
    named = dict(owner.model.named_parameters())
    if (len(owner.optimizer.param_groups) != 1
            or tuple(id(value) for value in owner.optimizer.param_groups[0]["params"])
                != tuple(id(named[name]) for name in owner.parameter_names)):
        raise ValueError("utility optimizer lost exact private tensor ownership/order")
    state = clone(owner.optimizer.state_dict())
    fields(state, ("state", "param_groups"), "utility AdamW state")
    config = owner.preparation["recipe"]["optimizer"]
    group = state["param_groups"][0]
    if (group["group_name"] != owner.preparation["recipe"]["optimizer_owner"]
            or group["lr"] != float(config["learning_rate"])
            or group["weight_decay"] != float(config["weight_decay"])
            or group["eps"] != float(config["epsilon"])
            or tuple(group["betas"]) != tuple(float(value) for value in config["betas"])
            or group.get("foreach") is not False or group.get("amsgrad") is not False
            or group.get("maximize") is not False or group.get("capturable") is not False
            or group.get("differentiable") is not False or group.get("fused") not in (False, None)):
        raise ValueError("utility AdamW options differ from the fixed explicit recipe")
    slots = group["params"]
    if len(slots) != 4 or len(set(slots)) != 4:
        raise ValueError("utility AdamW requires four unique slots")
    if set(state["state"]) != (set(slots) if owner.completed_updates else set()):
        raise ValueError("utility AdamW moments differ from completed update boundary")
    for name, slot in zip(owner.parameter_names, slots, strict=True):
        if not owner.completed_updates:
            continue
        item = state["state"][slot]
        fields(item, ("step", "exp_avg", "exp_avg_sq"), "utility AdamW moment state")
        step = item["step"]
        if (not isinstance(step, torch.Tensor) or step.shape != torch.Size([])
                or step.dtype != torch.float32 or not bool(torch.isfinite(step).item())
                or float(step.item()) != owner.completed_updates):
            raise ValueError("utility AdamW actual step differs from update count")
        for key in ("exp_avg", "exp_avg_sq"):
            value = item[key]
            if (not isinstance(value, torch.Tensor) or value.shape != named[name].shape
                    or value.dtype != torch.float32 or value.layout != torch.strided
                    or not bool(torch.isfinite(value).all().item())
                    or key == "exp_avg_sq" and bool((value < 0).any().item())):
                raise ValueError("utility AdamW moment shape/dtype/finite value mismatch")
    scheduler = clone(owner.scheduler.state_dict())
    if (scheduler["last_epoch"] != owner.completed_updates
            or scheduler["_step_count"] != owner.completed_updates + 1
            or scheduler["base_lrs"] != [float(config["learning_rate"])]
            or scheduler["_last_lr"] != scheduler["base_lrs"]):
        raise ValueError("utility constant scheduler differs from completed update boundary")
    return state, scheduler


def _create_utility_owner(model, preparation, *, sampler,
                          sampler_validator, cuda_devices=(), execute=False,
                          _fork_source=None, numerical_parts=None):
    _require_tensor_execution(execute)
    if numerical_parts is None:
        preparation = validate_training_preparation(preparation)
    else:
        from .rovent_utility_intervention_numerical import ValidatedUtilityNumericalParts
        if type(numerical_parts) is not ValidatedUtilityNumericalParts or not _typed_equal(
                numerical_parts.preparation, preparation):
            raise ValueError("sealed original intervention numerical parts/preparation differ")
        preparation = numerical_parts.preparation
    registration = preparation["registration"]
    _plain(sampler); sampler_validator(copy.deepcopy(sampler))
    runtime = _runtime(execute)
    torch, clone, validate_rng, capture_rng, restore_rng, _, _ = runtime
    with _LOCK:
        caller_rng = capture_rng(cuda_devices)
        try:
            # Parameter deepcopy creates fresh storages and does not preserve
            # an existing .grad. Inspect the caller's original ownership first
            # so cloning cannot conceal aliases or discard unfinished work.
            original_parameters, _ = _validate_actual(model, preparation, torch, numerical_parts=numerical_parts)
            _device_roster(original_parameters, cuda_devices)
            if any(value.grad is not None for value in original_parameters.values()):
                raise ValueError("utility initialization cannot discard existing gradients")
            scratch = copy.deepcopy(model)
            parameters, metadata = _validate_actual(scratch, preparation, torch, numerical_parts=numerical_parts)
            _device_roster(parameters, cuda_devices)
            if _fork_source is not _FORK_FROM_PINNED_SOURCE:
                if scratch.current_weights_sha256() != registration["utility"]["initial_weights_sha256"]:
                    raise ValueError("utility initialization tensors differ from exact registered source weights")
            if any(value.grad is not None for value in parameters.values()):
                raise ValueError("utility initialization cannot discard existing gradients")
            optimizer, scheduler, names = _fresh_optimizer(parameters, preparation, torch)
            rng = clone(caller_rng)
            validate_rng(rng, cuda_devices)
            return UtilityTensorOwner(scratch, preparation, metadata, optimizer, scheduler,
                names, rng, copy.deepcopy(sampler), tuple(cuda_devices))
        finally:
            restore_rng(caller_rng, cuda_devices)


def create_utility_owner(model, recipe, registration, membership, *, sampler,
                         sampler_validator, cuda_devices=(), execute=False):
    """Original V1 lifecycle-only path; cannot authorize cohort supervision."""
    _require_tensor_execution(execute)
    preparation = prepare_training(recipe, registration, membership)
    return _create_utility_owner(model, preparation, sampler=sampler,
        sampler_validator=sampler_validator, cuda_devices=cuda_devices, execute=execute)


def create_cohort_utility_owner(model, recipe, registration, membership, *, sampler,
                                sampler_validator, cuda_devices=(), execute=False):
    """Explicit preparation/2 factory; still needs the current parent lease."""
    _require_tensor_execution(execute)
    preparation = prepare_cohort_training(recipe, registration, membership)
    return _create_utility_owner(model, preparation, sampler=sampler,
        sampler_validator=sampler_validator, cuda_devices=cuda_devices, execute=execute)


def _checkpoint_family(value):
    from .rovent_utility_training_family import LEGACY, require_family
    return require_family(LEGACY if value is None else value)


def _family_resume_cost_state_sha256(owner, family):
    family = _checkpoint_family(family)
    return metadata_digest(family.domain(_RESUME_COST_STATE_DOMAIN), {
        "cost": _plain(owner.cost), "ancestor_cost": _plain(owner.ancestor_cost),
        "source_cost_observation": _plain(owner.source_cost_observation)})


def _validate_checkpoint_actual(model, preparation, torch, family, parts):
    if family.intervention:
        return _validate_actual(model, preparation, torch, numerical_parts=parts)
    return _validate_actual(model, preparation, torch)


def _derived_training_state(owner, *, _family=None):
    _counts(owner)
    family = _checkpoint_family(_family)
    parts = family.parts(owner.preparation)
    from .generation_data import digest
    domain = family.domain(STATE_DOMAIN)
    return {"schema": domain,
        "initial_source_declaration": copy.deepcopy(owner.metadata["initial_training"]),
        "current_tensor_kind": "learned" if owner.total_updates() else "initial_source_only",
        "completed_updates": owner.completed_updates, "ancestor_updates": owner.ancestor_updates,
        "total_updates": owner.total_updates(),
        "completion_chain_sha256": digest(domain,
            [*owner.ancestor_completion_sha256, *(item["sha256"] for item in owner.completion_receipts)]),
        "recipe_sha256": parts.recipe_sha256,
        "registration_sha256": parts.source_registration_sha256,
        "frozen_base_weights_sha256": owner.metadata["base_weights_sha256"],
        "frozen_base_mapping_sha256": owner.metadata["frozen_base_mapping_sha256"],
        "live_training_authority_reissued": False}


def derived_training_state(owner):
    """Original source-family projection; never accepts intervention metadata."""
    return _derived_training_state(owner)


def _utility_tensor_payload(owner, *, sampler_validator, execute=False, _family=None):
    _require_tensor_execution(execute)
    family = _checkpoint_family(_family)
    parts = family.parts(owner.preparation)
    runtime = _runtime(execute)
    torch, clone, validate_rng, _, _, state_digest, _ = runtime
    _counts(owner)
    if owner.quarantined or any(value.grad is not None for value in owner.model.parameters()):
        raise ValueError("quarantined/partial utility owner cannot publish a checkpoint")
    parameters, _ = _validate_checkpoint_actual(owner.model, owner.preparation, torch, family, parts)
    _device_roster(parameters, owner.cuda_devices)
    if not _typed_equal(_metadata(owner.model), owner.metadata):
        raise ValueError("immutable utility/frozen-base metadata changed")
    _plain(owner.sampler); sampler_validator(copy.deepcopy(owner.sampler))
    validate_rng(owner.rng, owner.cuda_devices)
    model = clone(owner.model.state_dict())
    optimizer, scheduler = _optimizer_state(owner, torch, clone)
    payload = {"schema": family.domain(CHECKPOINT_DOMAIN), "preparation": _plain(owner.preparation),
        "metadata": _plain(owner.metadata), "model": model, "optimizer": optimizer,
        "scheduler": scheduler, "rng": clone(owner.rng), "sampler": _plain(owner.sampler),
        "cuda_devices": list(owner.cuda_devices), "parameter_names": list(owner.parameter_names),
        "completed_updates": owner.completed_updates, "dispatched_updates": owner.dispatched_updates,
        "ancestor_updates": owner.ancestor_updates, "pending_microbatches": 0,
        "cost": dict(owner.cost), "ancestor_cost": dict(owner.ancestor_cost),
        "completion_receipts": _plain(owner.completion_receipts),
        "ancestor_completion_sha256": list(owner.ancestor_completion_sha256),
        "parent_checkpoint_sha256": owner.parent_checkpoint_sha256,
        "source_cost_observation": _plain(owner.source_cost_observation),
        "derived_training_state": _derived_training_state(owner, _family=family)}
    payload["state_sha256"] = state_digest(payload)
    return payload


def utility_tensor_payload(owner, *, sampler_validator, execute=False):
    return _utility_tensor_payload(owner, sampler_validator=sampler_validator, execute=execute)


def _tensor_projection(payload, runtime, *, _family=None):
    """Saved-payload observation only; this does not decode or attest tensors."""
    state_digest = runtime[5]
    family = _checkpoint_family(_family)
    if payload["schema"] != family.domain(CHECKPOINT_DOMAIN):
        raise ValueError("utility tensor projection cannot retag another source-family payload")
    family.parts(payload["preparation"])
    result = {"schema": family.domain(PROJECTION_DOMAIN), "payload_state_sha256": payload["state_sha256"]}
    for name in ("preparation", "metadata", "sampler", "cuda_devices", "parameter_names", "completed_updates",
                 "dispatched_updates", "ancestor_updates", "pending_microbatches", "cost", "ancestor_cost",
                 "completion_receipts", "ancestor_completion_sha256", "parent_checkpoint_sha256",
                 "source_cost_observation", "derived_training_state"):
        result[name] = _plain(payload[name])
    result.update({"model_state_sha256": state_digest(payload["model"]),
        "optimizer_state_sha256": state_digest(payload["optimizer"]),
        "scheduler_state_sha256": state_digest(payload["scheduler"]),
        "rng_state_sha256": state_digest(payload["rng"]),
        "sampler_state_sha256": state_digest(payload["sampler"])})
    return result


def utility_tensor_projection(owner, *, sampler_validator, execute=False):
    """Measure the complete scratch state, without changing its cost ledger."""
    runtime = _runtime(execute)
    payload = utility_tensor_payload(owner, sampler_validator=sampler_validator, execute=True)
    return _tensor_projection(payload, runtime)


def _write_projection(path, value, *, maximum, read_accounting):
    import hashlib
    import json
    from .artifacts import publish_noreplace
    from .generation_export import _hash_file
    maximum = uint(maximum, MAX_PROJECTION_BYTES)
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    if not maximum or len(raw) > maximum:
        raise ValueError("utility tensor projection exceeds original remaining output credit")
    descriptor, partial = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    publish_noreplace(partial, path)
    observed, size = _hash_file(path, maximum=maximum, read_accounting=read_accounting)
    if observed != hashlib.sha256(raw).hexdigest() or size != len(raw):
        raise ValueError("utility tensor projection changed after publication")
    return {"path": str(path), "sha256": observed, "bytes": size}


def _save_utility_checkpoint(owner, path, *, sampler_validator,
                            max_file_bytes=MAX_CHECKPOINT_BYTES, read_accounting=None, execute=False, _family=None):
    _require_tensor_execution(execute)
    family = _checkpoint_family(_family)
    maximum = uint(max_file_bytes, MAX_CHECKPOINT_BYTES)
    if not maximum:
        raise ValueError("positive bounded utility checkpoint byte credit required")
    payload = _utility_tensor_payload(owner, sampler_validator=sampler_validator, execute=execute, _family=family)
    # Reuse the bounded serialization and no-replace primitives, not the
    # fixed P/C/V writer's ledger codec or its output-byte mutation loop.
    from .artifacts import publish_noreplace
    from .generation_training import _BoundedCheckpointWriter, _serialize_checkpoint
    target = Path(path)
    if (not target.is_absolute() or target.suffix != ".pt"
            or target.parent.resolve(strict=True) != target.parent or target.is_symlink()):
        raise ValueError("utility checkpoint requires a plain existing absolute .pt parent")
    repository = next((parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()), None)
    if repository is not None and (repository == target or repository in target.parents):
        raise ValueError("utility generated tensor checkpoints must remain outside source")
    if target.exists():
        raise FileExistsError("utility checkpoint already exists")
    projection_path = target.with_name(target.name + ".projection.json")
    if projection_path.exists() or projection_path.is_symlink():
        raise FileExistsError("utility tensor projection already exists")
    descriptor, partial = tempfile.mkstemp(prefix=target.name + ".", suffix=".partial", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            _serialize_checkpoint(payload, _BoundedCheckpointWriter(stream, maximum))
            stream.flush()
            os.fsync(stream.fileno())
        publish_noreplace(partial, target)
        from .generation_export import _hash_file
        digest, size = _hash_file(path, maximum=maximum, read_accounting=read_accounting)
        projection = _tensor_projection(payload, _runtime(True), _family=family)
        projection_pin = _write_projection(projection_path, projection,
            maximum=min(MAX_PROJECTION_BYTES, maximum - size), read_accounting=read_accounting)
        snapshot_cost = dict(owner.cost)
        published_output_bytes = uint(owner.cost["output_bytes"] + size + projection_pin["bytes"])
    except BaseException:
        # The partial or published output belongs to the failed run and is
        # retained. A caller may not continue from an uncertain clean owner.
        owner.quarantined = True
        owner.physical_completion = "unknown"
        raise
    owner.cost["output_bytes"] = published_output_bytes
    return {"path": str(target), "sha256": digest, "bytes": size,
            "tensor_projection": projection_pin,
            "completed_updates": owner.completed_updates, "state_sha256": payload["state_sha256"],
            "snapshot_cost": snapshot_cost, "post_snapshot_cost": dict(owner.cost)}


def save_utility_checkpoint(owner, path, *, sampler_validator,
                            max_file_bytes=MAX_CHECKPOINT_BYTES, read_accounting=None, execute=False):
    return _save_utility_checkpoint(owner, path, sampler_validator=sampler_validator,
        max_file_bytes=max_file_bytes, read_accounting=read_accounting, execute=execute)


def _restore(payload, template, preparation, *, sampler_validator, runtime, _family=None):
    family = _checkpoint_family(_family)
    parts = family.parts(preparation)
    torch, clone, validate_rng, _, _, state_digest, equal = runtime
    names = ("schema", "preparation", "metadata", "model", "optimizer", "scheduler", "rng", "sampler",
        "cuda_devices", "parameter_names", "completed_updates", "dispatched_updates", "ancestor_updates",
        "pending_microbatches", "cost", "ancestor_cost", "completion_receipts", "ancestor_completion_sha256",
        "parent_checkpoint_sha256", "source_cost_observation", "derived_training_state", "state_sha256")
    fields(payload, names, "separate utility tensor checkpoint")
    if payload["schema"] != family.domain(CHECKPOINT_DOMAIN) or not _typed_equal(payload["preparation"], preparation):
        raise ValueError("utility checkpoint recipe/source/frozen-base preparation differs")
    sha(payload["state_sha256"])
    if state_digest({name: value for name, value in payload.items() if name != "state_sha256"}) != payload["state_sha256"]:
        raise ValueError("utility full tensor/state bytes digest differs")
    if not _typed_equal(_metadata(template), payload["metadata"]):
        raise ValueError("utility frozen base/initial metadata differs before scratch restore")
    scratch = copy.deepcopy(template)
    parameters, metadata = _validate_checkpoint_actual(scratch, preparation, torch, family, parts)
    _device_roster(parameters, payload["cuda_devices"])
    expected = scratch.state_dict()
    if (type(payload["model"]) is not dict or set(payload["model"]) != set(expected)
            or any(not isinstance(payload["model"][name], torch.Tensor)
                   or tuple(payload["model"][name].shape) != tuple(value.shape)
                   or payload["model"][name].dtype != torch.float32
                   or payload["model"][name].layout != torch.strided
                   or not bool(torch.isfinite(payload["model"][name]).all().item())
                   for name, value in expected.items())):
        raise ValueError("complete finite utility FP32 tensor state required before load")
    source_storages = set()
    for value in payload["model"].values():
        storage = (str(value.device), value.untyped_storage().data_ptr())
        if not value.is_contiguous() or storage in source_storages:
            raise ValueError("utility incoming tensor storage alias/layout is forbidden")
        source_storages.add(storage)
    scratch.load_state_dict(clone(payload["model"]), strict=True)
    _validate_checkpoint_actual(scratch, preparation, torch, family, parts)
    optimizer, scheduler, canonical = _fresh_optimizer(parameters, preparation, torch)
    if payload["parameter_names"] != list(canonical):
        raise ValueError("utility optimizer canonical parameter order differs")
    optimizer.load_state_dict(clone(payload["optimizer"]))
    scheduler.load_state_dict(clone(payload["scheduler"]))
    if not equal(clone(optimizer.state_dict()), payload["optimizer"]) or not equal(clone(scheduler.state_dict()), payload["scheduler"]):
        raise ValueError("utility optimizer/scheduler state changed while loading")
    validate_rng(payload["rng"], tuple(payload["cuda_devices"]))
    _plain(payload["sampler"]); sampler_validator(copy.deepcopy(payload["sampler"]))
    owner = UtilityTensorOwner(scratch, copy.deepcopy(preparation), metadata, optimizer, scheduler,
        canonical, clone(payload["rng"]), copy.deepcopy(payload["sampler"]), tuple(payload["cuda_devices"]),
        completed_updates=payload["completed_updates"], dispatched_updates=payload["dispatched_updates"],
        ancestor_updates=payload["ancestor_updates"], pending_microbatches=payload["pending_microbatches"],
        cost=copy.deepcopy(payload["cost"]), ancestor_cost=copy.deepcopy(payload["ancestor_cost"]),
        completion_receipts=copy.deepcopy(payload["completion_receipts"]),
        ancestor_completion_sha256=list(payload["ancestor_completion_sha256"]),
        parent_checkpoint_sha256=payload["parent_checkpoint_sha256"],
        source_cost_observation=_plain(payload["source_cost_observation"]))
    if not _typed_equal(_derived_training_state(owner, _family=family), payload["derived_training_state"]):
        raise ValueError("utility current/ancestor derived training state differs")
    # Complete round-trip checks include every tensor byte and all five state
    # families, not just a caller-supplied state digest.
    if not equal(_utility_tensor_payload(owner, sampler_validator=sampler_validator, execute=True, _family=family), payload):
        raise ValueError("utility scratch checkpoint round-trip differs")
    return owner


def _stage_utility_resume(path, expected_sha256, template, preparation, *,
                          sampler_validator, max_file_bytes=MAX_CHECKPOINT_BYTES,
                          read_accounting=None, snapshot_directory=None, execute=False,
                          authority=None, historical_checkpoint=None,
                          require_historical=False, _family=None):
    _require_tensor_execution(execute)
    family = _checkpoint_family(_family)
    preparation = family.parts(preparation).preparation
    if require_historical and authority is not None and historical_checkpoint is None:
        raise ValueError("actual cohort resume requires the original historical checkpoint owner")
    if historical_checkpoint is not None:
        if family.intervention:
            from .rovent_utility_intervention_checkpoint import HistoricalUtilityInterventionCheckpoint
            expected_type = HistoricalUtilityInterventionCheckpoint
        else:
            from .rovent_utility_historical_checkpoint import HistoricalUtilityCheckpoint
            expected_type = HistoricalUtilityCheckpoint
        if (not family.intervention and preparation["schema"] != COHORT_PREPARATION_DOMAIN
                or type(historical_checkpoint) is not expected_type):
            raise ValueError("explicit typed cohort historical checkpoint source required")
        original_pin = historical_checkpoint.tensor_pin()
        if original_pin["sha256"] != expected_sha256:
            raise ValueError("utility scratch selected a different original historical tensor")
    runtime = _runtime(execute)
    from .rovent_tensor_training import _load_pinned_tensor_file
    _, _, _, capture_rng, restore_rng, _, _ = runtime
    maximum = uint(max_file_bytes, MAX_CHECKPOINT_BYTES)
    if not maximum:
        raise ValueError("positive utility checkpoint read credit required")
    payload, source_bytes = _load_pinned_tensor_file(path, expected_sha256, maximum, runtime[0],
        read_accounting=read_accounting, snapshot_directory=snapshot_directory)
    if type(payload) is not dict or type(payload.get("cuda_devices")) is not list:
        raise ValueError("complete utility RNG device roster required")
    devices = tuple(payload["cuda_devices"])
    if any(type(device) is not int or device < 0 for device in devices) or len(set(devices)) != len(devices):
        raise ValueError("explicit unique utility CUDA RNG devices required")
    with _LOCK:
        caller_rng = capture_rng(devices)
        try:
            owner = _restore(payload, template, preparation, sampler_validator=sampler_validator, runtime=runtime, _family=family)
            if historical_checkpoint is not None:
                if family.intervention:
                    from .rovent_utility_intervention_checkpoint import project_intervention_utility_tensor_scratch as project_scratch
                else:
                    from .rovent_utility_historical_checkpoint import project_utility_tensor_scratch as project_scratch
                if original_pin["bytes"] != source_bytes:
                    raise ValueError("utility historical raw tensor byte count differs from scratch")
                measured = _tensor_projection(_utility_tensor_payload(owner,
                    sampler_validator=sampler_validator, execute=True, _family=family), runtime, _family=family)
                project_scratch(historical_checkpoint.projection(), observed_state=measured)
            owner.source_checkpoint_sha256 = expected_sha256
            owner.source_payload_state_sha256 = payload["state_sha256"]
            owner.source_snapshot_cost_observation = _plain(payload["source_cost_observation"])
            owner.source_snapshot_cost = dict(owner.cost)
            owner.source_snapshot_ancestor_cost = dict(owner.ancestor_cost)
            if authority is None:
                # Raw staged bytes remain lifecycle-only. They cannot be
                # dispatched or exported as a learned resumed window.
                owner.source_cost_observation = {"status": "unknown", "scope": "raw_snapshot_only",
                    "inherited": _plain(payload["source_cost_observation"])}
            else:
                family.require_authority(authority, allow_unleased=True)
                owner.source_cost_observation = _plain(authority.validate_checkpoint_state(owner, {
                    "path": str(path), "sha256": expected_sha256, "bytes": source_bytes}))
                # The concrete parent has verified original snapshot, output
                # delta and closure separately. Apply each known delta once
                # while retaining every unknown tail outside numeric counts.
                from .rovent_utility_training_authority import validate_utility_checkpoint_cost
                cost = validate_utility_checkpoint_cost(owner.source_cost_observation)
                if cost["snapshot_cost"] != owner.cost or cost["ancestor_cost"] != owner.ancestor_cost:
                    raise ValueError("utility original checkpoint cost scope differs from scratch snapshot")
                owner.cost = {name: uint(cost["known_lower_bound"][name] - owner.ancestor_cost[name])
                    for name in COST_FIELDS}
            owner.source_admitted_cost_state_sha256 = _family_resume_cost_state_sha256(owner, family)
            return owner
        finally:
            restore_rng(caller_rng, devices)


def stage_utility_resume(path, expected_sha256, template, recipe, registration, membership, *,
                         sampler_validator, max_file_bytes=MAX_CHECKPOINT_BYTES,
                         read_accounting=None, snapshot_directory=None, execute=False, authority=None):
    _require_tensor_execution(execute)
    return _stage_utility_resume(path, expected_sha256, template,
        prepare_training(recipe, registration, membership), sampler_validator=sampler_validator,
        max_file_bytes=max_file_bytes, read_accounting=read_accounting,
        snapshot_directory=snapshot_directory, execute=execute, authority=authority)


def stage_cohort_utility_resume(path, expected_sha256, template, recipe, registration, membership, *,
                                sampler_validator, max_file_bytes=MAX_CHECKPOINT_BYTES,
                                read_accounting=None, snapshot_directory=None, execute=False, authority=None,
                                historical_checkpoint=None):
    _require_tensor_execution(execute)
    return _stage_utility_resume(path, expected_sha256, template,
        prepare_cohort_training(recipe, registration, membership), sampler_validator=sampler_validator,
        max_file_bytes=max_file_bytes, read_accounting=read_accounting,
        snapshot_directory=snapshot_directory, execute=execute, authority=authority,
        historical_checkpoint=historical_checkpoint, require_historical=True)


def _stage_utility_fork(source, preparation, *, sampler,
                       source_sampler_validator, sampler_validator, reset_seed, execute=False, _family=None):
    """A fork resets moments/sampler/counts; a resume preserves every state."""
    _require_tensor_execution(execute)
    family = _checkpoint_family(_family)
    parts = family.parts(preparation)
    preparation = parts.preparation
    family.parts(source.preparation)
    runtime = _runtime(execute)
    _counts(source)
    if source.source_checkpoint_sha256 is None:
        raise ValueError("utility fork requires an exact pinned checkpoint source")
    if source.source_admitted_cost_state_sha256 != _family_resume_cost_state_sha256(source, family):
        raise ValueError("utility fork source changed after its exact pinned cost admission")
    source_payload = _utility_tensor_payload(source, sampler_validator=source_sampler_validator, execute=True, _family=family)
    # Parent source must still be precisely the tensor/journal snapshot that
    # was loaded. A later update cannot claim that older checkpoint as parent.
    original_projection = dict(source_payload)
    original_projection["source_cost_observation"] = _plain(source.source_snapshot_cost_observation)
    original_projection["cost"] = _plain(source.source_snapshot_cost)
    original_projection["ancestor_cost"] = _plain(source.source_snapshot_ancestor_cost)
    original_projection.pop("state_sha256")
    if runtime[5](original_projection) != source.source_payload_state_sha256:
        raise ValueError("utility fork source changed after its exact pinned checkpoint")
    uint(reset_seed, 2**32 - 1)
    import random
    import numpy as np
    torch, _, _, capture_rng, restore_rng, _, _ = runtime
    with _LOCK:
        caller_rng = capture_rng(source.cuda_devices)
        try:
            random.seed(reset_seed)
            np.random.seed(reset_seed)
            torch.random.default_generator.manual_seed(reset_seed)
            for device in source.cuda_devices:
                torch.cuda.default_generators[device].manual_seed(reset_seed)
            child = _create_utility_owner(source.model, preparation, sampler=sampler,
                sampler_validator=sampler_validator, cuda_devices=source.cuda_devices, execute=True,
                _fork_source=_FORK_FROM_PINNED_SOURCE, **({"numerical_parts": parts} if family.intervention else {}))
        finally:
            restore_rng(caller_rng, source.cuda_devices)
    if (child.metadata != source.metadata or child.preparation["registration"] != source.preparation["registration"]):
        raise ValueError("utility fork cannot change frozen group/source registration")
    child.ancestor_updates = source.total_updates()
    child.ancestor_cost = source.total_cost()
    child.ancestor_completion_sha256 = [*source.ancestor_completion_sha256,
        *(item["sha256"] for item in source.completion_receipts)]
    child.parent_checkpoint_sha256 = source.source_checkpoint_sha256
    child.source_cost_observation = copy.deepcopy(source.source_cost_observation)
    # A valid source snapshot, not a fabricated JSON flag, was consumed above.
    if not source_payload["derived_training_state"]["completion_chain_sha256"]:
        raise ValueError("utility fork source update chain missing")
    return child


def stage_utility_fork(source, *, recipe, registration, membership, sampler,
                       source_sampler_validator, sampler_validator, reset_seed, execute=False):
    _require_tensor_execution(execute)
    preparation = (prepare_cohort_training(recipe, registration, membership)
        if source.preparation["schema"] == COHORT_PREPARATION_DOMAIN
        else prepare_training(recipe, registration, membership))
    return _stage_utility_fork(source, preparation, sampler=sampler,
        source_sampler_validator=source_sampler_validator, sampler_validator=sampler_validator,
        reset_seed=reset_seed, execute=execute)


def _export_utility_owner(owner, path, *, authority, kind,
                         max_output_bytes=8 * 1024**2, execute=False, _family=None):
    """Call the numerical writer only after original learned-source admission."""
    _require_tensor_execution(execute)
    family = _checkpoint_family(_family)
    family.require_authority(authority, owner=owner)
    family.parts(owner.preparation)
    torch = _runtime(execute)[0]
    if kind not in ("collection", "deployment"):
        raise ValueError("registered utility export kind required")
    _counts(owner)
    if owner.quarantined or owner.pending_microbatches or owner.physical_completion != "confirmed":
        raise ValueError("utility export requires quiescent confirmed physical completion")
    if not family.intervention and owner.preparation["schema"] != COHORT_PREPARATION_DOMAIN:
        raise ValueError("learned utility export requires explicit cohort preparation/2")
    training = authority.validate_learned_export(owner, kind)
    fields(training, ("kind", "steps", "run_id", "dataset_sha256"), "original utility learned training projection")
    if training["kind"] != "learned" or training["steps"] != owner.total_updates() or not owner.total_updates():
        raise ValueError("original learned utility projection differs from actual completed updates")
    sha(training["dataset_sha256"])
    if type(training["run_id"]) is not str or not training["run_id"]:
        raise ValueError("actual utility learned run identity required")
    maximum = uint(max_output_bytes, 8 * 1024**2)
    if not maximum:
        raise ValueError("positive bounded utility export byte credit required")
    # Check the exact writer's UTF8 representation before publishing any byte.
    # Declared output credit cannot be enforced only after an oversized write.
    import json
    mapping = owner.model.export_checkpoint(training=training)
    raw = (json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    if len(raw) > maximum:
        raise ValueError("utility export exceeds original output credit before writing")
    model_versions = tuple((name, id(value), value._version) for name, value in owner.model.named_parameters())
    # Deployment has no V and no change to the frozen base/P/C exports. This
    # writer emits the separate utility artifact only, never a P/C/V checkpoint.
    try:
        pin = owner.model.write_checkpoint(path, training=training, max_output_bytes=maximum,
            read_accounting=authority.read_accounting)
        if (pin["bytes"] != len(raw) or pin["bytes"] > maximum
                or tuple((name, id(value), value._version) for name, value in owner.model.named_parameters()) != model_versions):
            raise ValueError("utility export tensor version/registered byte representation changed")
        import hashlib
        if pin["sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError("utility exported file differs from original bounded snapshot")
        owner.cost["output_bytes"] = uint(owner.cost["output_bytes"] + pin["bytes"])
        for device in owner.cuda_devices:
            torch.cuda.synchronize(device)
        owner.physical_completion = "confirmed"
    except BaseException:
        owner.quarantined = True
        owner.physical_completion = "unknown"
        raise
    return {"schema": family.domain(EXPORT_DOMAIN), "kind": kind, "artifact": pin,
        "derived_training_state": _derived_training_state(owner, _family=family),
        "registration": copy.deepcopy(owner.preparation["registration"]),
        "snapshot_cost_lower_bound": owner.total_cost(),
        "historical_cost_observation": _plain(owner.source_cost_observation),
        "v_present": False, "learned_admission": copy.deepcopy(training),
        "physical_completion": "confirmed"}


def export_utility_owner(owner, path, *, authority, kind,
                         max_output_bytes=8 * 1024**2, execute=False):
    return _export_utility_owner(owner, path, authority=authority, kind=kind,
        max_output_bytes=max_output_bytes, execute=execute)
