"""Actual tensor lifecycle for a host-owned Rovent learner; no update method.

Import is standard-library-only. Explicit ``execute=True`` operations may
materialize scratch tensors/AdamW owners, save or resume a pinned checkpoint,
or call the original exporters. They never call forward/backward/optimizer.step.
This domain proves tensor lifecycle only and cannot certify actual training or
turn a diagnostic checkpoint into a trained/evaluation asset.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import threading
import zipfile

from .generation_data import digest, fields, sha, uint
from .generation_export import COST_FIELDS, _hash_file
from .generation_training import OPTIMIZER, _typed_equal, atomic_checkpoint
from .rovent_generation import SeriesRecipe
from .rovent_training import membership_digest, optimizer_ownership, prepare_training_stage, validate_training_membership
from .rovent_nnuec_training_state import (derive_nnuec_tensor_training_state,
    fork_nnuec_ancestor, nnuec_is_unmodified)

CHECKPOINT_DOMAIN = "rz-rovent-tensor-checkpoint/1"
ELIGIBILITY = "tensor_lifecycle_only_not_training_evidence"
MAX_BYTES = 2 * 1024**3
ROLES = ("proposer", "critic", "validator")
BUCKETS = (8, 16, 32, 64, 128, 256)
_RNG_LOCK = threading.RLock()


def _require_execution(execute):
    if execute is not True:
        raise ValueError("explicit tensor lifecycle execution required; no implicit model/optimizer import")


def _runtime(execute):
    _require_execution(execute)
    import torch
    from .training import _clone_state, _validate_rng, capture_rng, restore_rng
    from .nonzero_training import _state_digest, state_equal
    return torch, _clone_state, _validate_rng, capture_rng, restore_rng, _state_digest, state_equal


def _owner_component(name):
    from .rovent_training import _component
    return _component(name)


def _validate_json_state(value, *, depth=0, allowance=None):
    """Sampler/cost sidecars cannot smuggle executable objects into a payload."""
    if allowance is None:
        allowance = [1048576]
    allowance[0] -= 1
    if depth > 64 or allowance[0] < 0:
        raise ValueError("bounded plain checkpoint sidecar required")
    if type(value) in (dict, list, tuple):
        if len(value) > 262144:
            raise ValueError("checkpoint sidecar entry capacity exceeded")
        if type(value) is dict:
            if any(type(key) is not str or len(key.encode()) > 4096 for key in value):
                raise ValueError("plain bounded sidecar keys required")
            values = value.values()
        else:
            values = value
        for child in values:
            _validate_json_state(child, depth=depth + 1, allowance=allowance)
    elif type(value) is str:
        if len(value.encode()) > 1024**2:
            raise ValueError("checkpoint sidecar string capacity exceeded")
    elif type(value) not in (int, float, bool, type(None)):
        raise ValueError("checkpoint sidecar is not plain data")
    elif type(value) is float:
        import math
        if not math.isfinite(value):
            raise ValueError("checkpoint sidecar scalar is nonfinite")


def _live_parameters(model, membership, torch):
    module = getattr(model, "nnuec", None)
    if not callable(getattr(module, "parameter_spec", None)):
        raise ValueError("actual NNUEC trainable parameter spec required")
    validate_training_membership(membership, nnuec_metadata=_nnuec_metadata(model),
        nnuec_parameter_spec=module.parameter_spec())
    expected = {item["name"]: item for item in membership["parameters"]}
    parameters = dict(model.named_parameters())
    if set(parameters) != set(expected):
        raise ValueError("actual model parameter names differ from registered membership")
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        component = _owner_component(name)
        previous = aliases.setdefault(id(parameter), component)
        if previous != component:
            raise ValueError("cross-component actual parameter alias")
    for name, parameter in parameters.items():
        spec = expected[name]
        if (list(parameter.shape) != spec["shape"] or parameter.dtype != torch.float32
                or parameter.layout != torch.strided or not bool(torch.all(torch.isfinite(parameter)))):
            raise ValueError("actual model parameter shape/dtype/value differs from membership")
    return parameters


def _fresh_optimizers(model, recipe, membership, torch):
    parameters = _live_parameters(model, membership, torch)
    groups = optimizer_ownership(recipe, membership)
    optimizers, schedulers = {}, {}
    for owner, names in groups.items():
        optimizers[owner] = torch.optim.AdamW([{"params": [parameters[name] for name in names], "group_name": owner,
            "weight_decay": OPTIMIZER["weight_decay"]}], lr=OPTIMIZER["lr"], betas=tuple(OPTIMIZER["betas"]),
            eps=OPTIMIZER["epsilon"], foreach=False)
        schedulers[owner] = torch.optim.lr_scheduler.LambdaLR(optimizers[owner], lambda _: 1.0)
    return parameters, groups, optimizers, schedulers


@dataclass
class OwnedTensorState:
    """A new scratch owner; it is never reconstructed from a JSON capability."""
    model: object
    recipe: SeriesRecipe
    membership: dict
    stage_id: str
    optimizers: dict
    schedulers: dict
    parameter_names: dict
    rng: dict
    samplers: dict
    bucket: int
    cuda_devices: tuple
    nnuec_metadata: dict = field(default_factory=dict)
    model_metadata: dict = field(default_factory=dict)
    optimizer_configuration: dict = field(default_factory=dict)
    scheduler_configuration: dict = field(default_factory=dict)
    active_role: str = "proposer"
    completed: dict = field(default_factory=lambda: dict.fromkeys(ROLES, 0))
    dispatched: dict = field(default_factory=lambda: dict.fromkeys(ROLES, 0))
    parameter_updates: dict = field(default_factory=dict)
    cost: dict = field(default_factory=lambda: dict.fromkeys(COST_FIELDS, 0))
    ancestor_cost: dict = field(default_factory=lambda: dict.fromkeys(COST_FIELDS, 0))
    ancestor_updates: dict = field(default_factory=lambda: dict.fromkeys(ROLES, 0))
    pending_microbatches: int = 0
    parent_checkpoint_sha256: str | None = None
    source_checkpoint_sha256: str | None = None
    source_checkpoint_path: str | None = None
    source_model_state_sha256: str | None = None
    source_nnuec_metadata_sha256: str | None = None
    source_payload_state_sha256: str | None = None
    inherited_input_sha256: list = field(default_factory=list)
    inherited_origin_sha256: list = field(default_factory=list)
    quarantined: bool = False
    # Live process state only, never a serialized CUDA completion capability.
    physical_completion: str = "unknown"
    nnuec_ancestor_training_state: dict | None = None

    def activate_role(self, role):
        if role not in ROLES or not self.recipe.stage(self.stage_id).role_updates[ROLES.index(role)]:
            raise ValueError("unknown learner role")
        if self.pending_microbatches or any(parameter.grad is not None for parameter in self.model.parameters()):
            raise ValueError("role change cannot discard pending accumulation/gradients")
        if role == "validator":
            stage = self.recipe.stage(self.stage_id)
            if (not self.membership["validator_present"] or not stage.role_updates[2]
                    or any(self.completed[name] != limit for name, limit in zip(ROLES[:2], stage.role_updates[:2]))):
                raise ValueError("V activation requires complete frozen P/C and declared private V")
        active = {"validator": {"v_private"}, "proposer": {"pc_shared", "p_private", "nnuec"},
                  "critic": {"pc_shared", "c_private", "nnuec"}}[role]
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(_owner_component(name) in active)
        self.active_role = role

    def total_cost(self):
        return {name: uint(self.ancestor_cost[name] + self.cost[name]) for name in COST_FIELDS}


def _nnuec_metadata(model):
    module = getattr(model, "nnuec", None)
    if module is None or not all(callable(getattr(module, name, None)) for name in
                                 ("metadata_state", "validate_metadata_state", "restore_metadata_state")):
        raise ValueError("registered NNUEC model must expose its nonparameter metadata lifecycle")
    metadata = module.metadata_state()
    _validate_json_state(metadata)
    module.validate_metadata_state(copy.deepcopy(metadata))
    return copy.deepcopy(metadata)


def _model_metadata(model):
    config = getattr(model, "config", None)
    if not callable(getattr(config, "to_dict", None)):
        raise ValueError("original composed model must expose immutable source configuration")
    result = {"class": type(model).__module__ + "." + type(model).__qualname__, "config": config.to_dict()}
    _validate_json_state(result)
    return copy.deepcopy(result)


def _nnuec_parameter_counts(values):
    if type(values) is not dict or any(type(name) is not str for name in values):
        raise ValueError("complete named parameter update counts required before NNUEC state validation")
    for value in values.values():
        uint(value)
    return {name: value for name, value in values.items() if name.startswith("nnuec.")}


def _nnuec_tensor_training_state(owner):
    return derive_nnuec_tensor_training_state(source_declaration=owner.nnuec_metadata["training"],
        parameter_updates=_nnuec_parameter_counts(owner.parameter_updates),
        samplers=owner.samplers, ancestor=owner.nnuec_ancestor_training_state)


def validate_nnuec_tensor_checkpoint_state(payload):
    """Validate current tensor provenance before loading any caller's model.

    NNUEC's nested metadata declaration identifies the initial source asset.
    Current/ancestor updates have their own domain and may never be inferred
    to be untrained merely because that immutable source said untrained.
    """
    state = payload["nnuec_tensor_training_state"]
    if type(state) is not dict or "ancestor" not in state:
        raise ValueError("explicit current NNUEC tensor training state required")
    actual = derive_nnuec_tensor_training_state(source_declaration=payload["nnuec_metadata"]["training"],
        parameter_updates=_nnuec_parameter_counts(payload["parameter_updates"]),
        samplers=payload["samplers"], ancestor=state["ancestor"])
    if not _typed_equal(state, actual):
        raise ValueError("NNUEC current tensor update/ancestor state differs from source declaration scope")
    return actual


def create_tensor_owner(model, recipe, membership, *, stage_id, samplers, bucket, sampler_validator,
                        cuda_devices=(), execute=False):
    """Clone a caller model and create unused AdamW owners; caller stays intact."""
    _require_execution(execute)
    validate_training_membership(membership)
    runtime = _runtime(execute)
    torch, _, validate_rng, capture_rng, restore_rng, _, _ = runtime
    prepare_training_stage(recipe, membership, stage_id=stage_id, mode="initialize")
    if bucket not in BUCKETS or type(bucket) is not int:
        raise ValueError("registered physical line bucket required")
    _validate_json_state(samplers)
    sampler_validator(copy.deepcopy(samplers))
    with _RNG_LOCK:
        original_rng = capture_rng(tuple(cuda_devices))
        try:
            scratch = copy.deepcopy(model)
            parameters, groups, optimizers, schedulers = _fresh_optimizers(scratch, recipe, membership, torch)
            validate_rng(original_rng, tuple(cuda_devices))
            owner = OwnedTensorState(scratch, recipe, copy.deepcopy(membership), stage_id, optimizers, schedulers,
                groups, copy.deepcopy(original_rng), copy.deepcopy(samplers), bucket, tuple(cuda_devices),
                nnuec_metadata=_nnuec_metadata(scratch),
                model_metadata=_model_metadata(scratch),
                optimizer_configuration={name: copy.deepcopy(value.state_dict()["param_groups"]) for name, value in optimizers.items()},
                scheduler_configuration={name: copy.deepcopy(value.state_dict()) for name, value in schedulers.items()},
                parameter_updates=dict.fromkeys(parameters, 0))
            owner.activate_role(next(role for role, limit in zip(ROLES, recipe.stage(stage_id).role_updates) if limit))
            return owner
        finally:
            restore_rng(original_rng, tuple(cuda_devices))


def _ledger(owner):
    stage = owner.recipe.stage(owner.stage_id)
    fields(owner.completed, ROLES, "completed stage updates")
    fields(owner.dispatched, ROLES, "dispatched stage updates")
    fields(owner.cost, COST_FIELDS, "actual stage cost")
    fields(owner.ancestor_cost, COST_FIELDS, "retained ancestor cost")
    fields(owner.ancestor_updates, ROLES, "retained ancestor optimizer updates")
    for role, limit in zip(ROLES, stage.role_updates):
        uint(owner.completed[role], limit)
        if owner.completed[role] > uint(owner.dispatched[role]):
            raise ValueError("completed updates exceed dispatches")
    for count in list(owner.cost.values()) + list(owner.ancestor_cost.values()):
        uint(count)
    for count in owner.ancestor_updates.values():
        uint(count)
    if owner.pending_microbatches != 0 or type(owner.pending_microbatches) is not int:
        raise ValueError("checkpoint requires a completed accumulation boundary")
    if owner.active_role not in ROLES or owner.bucket not in BUCKETS or type(owner.bucket) is not int:
        raise ValueError("checkpoint role/bucket is not registered")
    if (not stage.role_updates[ROLES.index(owner.active_role)] or owner.active_role == "validator"
            and any(owner.completed[role] != limit for role, limit in zip(ROLES[:2], stage.role_updates[:2]))):
        raise ValueError("checkpoint active role violates its declared stage freeze boundary")
    for name in ("inherited_input_sha256", "inherited_origin_sha256"):
        values = getattr(owner, name)
        if type(values) is not list or len(values) > 262144 or len(set(sha(value) for value in values)) != len(values):
            raise ValueError("bounded unique inherited training seals required")
    if stage.parent_stage_id is None:
        if owner.parent_checkpoint_sha256 is not None:
            raise ValueError("common tensor state cannot have a foreign parent")
    else:
        sha(owner.parent_checkpoint_sha256)
    return {"completed": dict(owner.completed), "dispatched": dict(owner.dispatched), "cost": dict(owner.cost),
            "ancestor_cost": dict(owner.ancestor_cost), "ancestor_updates": dict(owner.ancestor_updates),
            "pending_microbatches": 0, "active_role": owner.active_role}


def _optimizer_snapshot(owner, torch, clone):
    parameters = _live_parameters(owner.model, owner.membership, torch)
    if set(owner.parameter_updates) != set(parameters):
        raise ValueError("per-parameter update membership differs from actual model")
    result = {}
    for identity, names in owner.parameter_names.items():
        optimizer = owner.optimizers[identity]
        if (len(optimizer.param_groups) != 1 or [id(p) for p in optimizer.param_groups[0]["params"]]
                != [id(parameters[name]) for name in names]):
            raise ValueError("actual optimizer does not exclusively own its registered parameters")
        incoming = optimizer.state_dict()
        if not _typed_equal(incoming["param_groups"], owner.optimizer_configuration[identity]):
            raise ValueError("actual AdamW options/order changed after owner creation")
        ids = incoming["param_groups"][0]["params"]
        if type(incoming["state"]) is not dict or set(incoming["state"]) - set(ids):
            raise ValueError("foreign optimizer parameter state")
        for name, slot in zip(names, ids):
            count = uint(owner.parameter_updates[name])
            component = _owner_component(name)
            maximum = owner.completed["validator"] if component == "v_private" else sum(
                owner.completed[role] for role in ("proposer", "critic") if component in
                (("pc_shared", "p_private", "nnuec") if role == "proposer" else ("pc_shared", "c_private", "nnuec")))
            if count > maximum or bool(count) != (slot in incoming["state"]):
                raise ValueError("unearned/missing AdamW moments")
            if not count:
                continue
            state = fields(incoming["state"][slot], ("step", "exp_avg", "exp_avg_sq"), "complete AdamW state")
            step = state["step"]
            if (not isinstance(step, torch.Tensor) or step.dtype != torch.float32 or step.ndim != 0
                    or not bool(torch.isfinite(step)) or step.item() != count):
                raise ValueError("AdamW scalar step differs from actual parameter update count")
            for part in ("exp_avg", "exp_avg_sq"):
                moment = state[part]
                if (not isinstance(moment, torch.Tensor) or moment.shape != parameters[name].shape
                        or moment.dtype != torch.float32 or moment.layout != torch.strided
                        or not bool(torch.all(torch.isfinite(moment)))
                        or part == "exp_avg_sq" and bool(torch.any(moment < 0))):
                    raise ValueError("invalid AdamW moment shape/dtype/value")
        result[identity] = clone(incoming)
    return result


def _schedule_count(owner, names):
    if (type(owner.samplers) is dict and "r_auxiliary" in owner.samplers
            and all(_owner_component(name) == "nnuec" for name in names)):
        from .rovent_r_auxiliary import validate_auxiliary_journal
        sidecar = validate_auxiliary_journal(owner.samplers["r_auxiliary"], owner.samplers["updates"])
        return sidecar["current_nnuec_updates"]
    active = set()
    for name in names:
        component = _owner_component(name)
        active.update(("validator",) if component == "v_private" else ("proposer",) if component == "p_private" else
                      ("critic",) if component == "c_private" else ("proposer", "critic"))
    return sum(owner.completed[role] for role in active)


def _scheduler_snapshot(owner, clone, equal):
    result = {}
    for identity, names in owner.parameter_names.items():
        expected = copy.deepcopy(owner.scheduler_configuration[identity])
        count = _schedule_count(owner, names)
        expected.update(last_epoch=count, _step_count=count + 1, _last_lr=expected["base_lrs"])
        actual = owner.schedulers[identity].state_dict()
        if not equal(actual, expected):
            raise ValueError("constant scheduler does not match completed owner updates")
        result[identity] = clone(actual)
    return result


def tensor_payload(owner, *, sampler_validator, execute=False):
    _require_execution(execute)
    validate_training_membership(owner.membership)
    torch, clone, validate_rng, _, _, _, equal = _runtime(execute)
    if type(owner) is not OwnedTensorState or owner.quarantined:
        raise ValueError("live nonquarantined original tensor owner required")
    ledger = _ledger(owner)
    if any(parameter.grad is not None for parameter in owner.model.parameters()):
        raise ValueError("checkpoint cannot discard pending gradients")
    _validate_json_state(owner.samplers)
    sampler_validator(copy.deepcopy(owner.samplers))
    validate_rng(owner.rng, owner.cuda_devices)
    if owner.parameter_names != optimizer_ownership(owner.recipe, owner.membership):
        raise ValueError("optimizer ownership changed after owner creation")
    for values in (owner.optimizers, owner.schedulers, owner.optimizer_configuration, owner.scheduler_configuration):
        fields(values, owner.parameter_names, "complete owned optimizer/scheduler configuration")
    if not _typed_equal(_nnuec_metadata(owner.model), owner.nnuec_metadata):
        raise ValueError("NNUEC nonparameter state changed outside its owner checkpoint boundary")
    if not _typed_equal(_model_metadata(owner.model), owner.model_metadata):
        raise ValueError("source model configuration changed outside its owner checkpoint boundary")
    active = {"validator": {"v_private"}, "proposer": {"pc_shared", "p_private", "nnuec"},
              "critic": {"pc_shared", "c_private", "nnuec"}}[owner.active_role]
    if any(parameter.requires_grad != (_owner_component(name) in active)
           for name, parameter in owner.model.named_parameters()):
        raise ValueError("actual trainable/frozen parameter state differs from active role")
    return {"schema": CHECKPOINT_DOMAIN, "eligibility": ELIGIBILITY, "recipe": owner.recipe.to_dict(),
            "recipe_sha256": owner.recipe.sha256, "membership": copy.deepcopy(owner.membership),
            "membership_sha256": membership_digest(owner.membership), "stage_id": owner.stage_id,
            "nnuec_metadata": copy.deepcopy(owner.nnuec_metadata),
            "nnuec_tensor_training_state": _nnuec_tensor_training_state(owner),
            "model_metadata": copy.deepcopy(owner.model_metadata),
            "model": clone(owner.model.state_dict()), "optimizers": _optimizer_snapshot(owner, torch, clone),
            "schedulers": _scheduler_snapshot(owner, clone, equal),
            "parameter_names": copy.deepcopy(owner.parameter_names), "parameter_updates": dict(owner.parameter_updates),
            "ledger": ledger, "samplers": copy.deepcopy(owner.samplers), "bucket": owner.bucket,
            "rng": clone(owner.rng), "cuda_devices": list(owner.cuda_devices),
            "parent_checkpoint_sha256": owner.parent_checkpoint_sha256,
            "inherited_input_sha256": list(owner.inherited_input_sha256),
            "inherited_origin_sha256": list(owner.inherited_origin_sha256)}


def save_tensor_checkpoint(owner, path, *, sampler_validator, max_file_bytes, execute=False, read_accounting=None):
    """Use the existing bounded atomic torch serializer; no update dispatch."""
    runtime = _runtime(execute)
    maximum = uint(max_file_bytes, MAX_BYTES)
    if not maximum:
        raise ValueError("positive checkpoint byte credit required")
    target = Path(path)
    if (not target.is_absolute() or target.suffix != ".pt" or target.is_symlink()
            or target.parent.resolve() != target.parent):
        raise ValueError("explicit original regular .pt output required")
    repository = next((parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()), None)
    if repository is not None and (target == repository or repository in target.parents):
        raise ValueError("generated tensor checkpoints must remain outside source")
    with _RNG_LOCK:
        payload = tensor_payload(owner, sampler_validator=sampler_validator, execute=True)
        try:
            atomic_checkpoint(target, payload, max_file_bytes=maximum)
            actual, size = _hash_file(target, maximum, read_accounting=read_accounting)
            owner.cost["output_bytes"] = payload["ledger"]["cost"]["output_bytes"]
            owner.source_checkpoint_sha256, owner.source_checkpoint_path = actual, str(target)
            owner.source_model_state_sha256 = runtime[5](payload["model"])
            owner.source_nnuec_metadata_sha256 = digest("rz-rovent-original-nnuec-metadata/1", payload["nnuec_metadata"])
            owner.source_payload_state_sha256 = runtime[5](payload)
            return {"schema": CHECKPOINT_DOMAIN, "path": str(target), "sha256": actual, "bytes": size,
                    "stage_id": owner.stage_id, "eligibility": ELIGIBILITY, "optimizer_updates_executed": 0,
                    "arena_eligible": False, "training_provenance_verified": False}
        except BaseException:
            # Partial/uncertain writes remain owned. Stop this owner instead of
            # refunding unknown storage/work and continuing with a clean ledger.
            owner.quarantined = True
            raise


def _load_pinned_tensor_file(path, expected_sha256, maximum, torch, *, read_accounting=None, snapshot_directory=None):
    source = Path(path)
    if not source.is_absolute() or source.suffix != ".pt" or source.is_symlink():
        raise ValueError("explicit pinned original .pt input required")
    sha(expected_sha256)
    directory = source.parent if snapshot_directory is None else Path(snapshot_directory)
    if snapshot_directory is not None and (not directory.is_absolute() or directory.resolve(strict=True) != directory
            or directory.is_symlink() or not directory.is_dir()):
        raise ValueError("explicit original owned snapshot directory required")
    with tempfile.TemporaryFile(mode="w+b", prefix="rz-rovent-pinned-", dir=directory) as snapshot:
        identity, size = _hash_file(source, maximum, snapshot=snapshot, read_accounting=read_accounting)
        if identity != expected_sha256:
            raise ValueError("original tensor checkpoint byte pin mismatch")
        snapshot.flush()
        snapshot.seek(0)
        with zipfile.ZipFile(snapshot) as archive:
            entries = archive.infolist()
            if (not entries or len(entries) > 65536 or len({entry.filename for entry in entries}) != len(entries)
                    or any(entry.compress_type != zipfile.ZIP_STORED for entry in entries)
                    or sum(entry.file_size for entry in entries) > maximum):
                raise ValueError("checkpoint archive allocation/identity exceeds declared credit")
        snapshot.seek(0)
        return torch.load(snapshot, map_location="cpu", weights_only=True), size


def _restore_scratch(payload, template, recipe, membership, *, sampler_validator, runtime):
    validate_training_membership(membership)
    torch, clone, validate_rng, capture_rng, restore_rng, _, equal = runtime
    fields(payload, ("schema", "eligibility", "recipe", "recipe_sha256", "membership", "membership_sha256", "stage_id",
        "nnuec_metadata", "nnuec_tensor_training_state", "model_metadata",
        "model", "optimizers", "schedulers", "parameter_names", "parameter_updates", "ledger", "samplers", "bucket",
        "rng", "cuda_devices", "parent_checkpoint_sha256", "inherited_input_sha256", "inherited_origin_sha256"), "Rovent tensor checkpoint")
    if (payload["schema"] != CHECKPOINT_DOMAIN or payload["eligibility"] != ELIGIBILITY
            or payload["recipe_sha256"] != recipe.sha256 or not _typed_equal(payload["recipe"], recipe.to_dict())
            or payload["membership_sha256"] != membership_digest(membership)
            or not _typed_equal(payload["membership"], membership)):
        raise ValueError("diagnostic/foreign checkpoint cannot be relabelled as Rovent tensor state")
    stage = recipe.stage(payload["stage_id"])
    groups = optimizer_ownership(recipe, membership)
    if payload["parameter_names"] != groups:
        raise ValueError("checkpoint changed single-owner parameter membership")
    devices = payload["cuda_devices"]
    if type(devices) is not list or any(type(device) is not int or device < 0 for device in devices) or len(set(devices)) != len(devices):
        raise ValueError("explicit unique checkpoint RNG devices required")
    validate_rng(payload["rng"], tuple(devices))
    _validate_json_state(payload["nnuec_metadata"])
    nnuec_tensor_state = validate_nnuec_tensor_checkpoint_state(payload)
    module = getattr(template, "nnuec", None)
    if not callable(getattr(module, "parameter_spec", None)):
        raise ValueError("actual NNUEC restore parameter spec required")
    validate_training_membership(membership, nnuec_metadata=_nnuec_metadata(template),
        nnuec_parameter_spec=module.parameter_spec())
    validate_training_membership(membership, nnuec_metadata=payload["nnuec_metadata"],
        nnuec_parameter_spec=module.parameter_spec())
    template.nnuec.validate_metadata_state(copy.deepcopy(payload["nnuec_metadata"]))
    if not _typed_equal(_model_metadata(template), payload["model_metadata"]):
        raise ValueError("source model immutable configuration differs from caller template")
    _validate_json_state(payload["samplers"])
    sampler_validator(copy.deepcopy(payload["samplers"]))
    fields(payload["ledger"], ("completed", "dispatched", "cost", "ancestor_cost", "ancestor_updates", "pending_microbatches", "active_role"), "checkpoint stage ledger")
    expected = template.state_dict()
    fields(payload["model"], expected, "complete original model state")
    for name, value in payload["model"].items():
        if (not isinstance(value, torch.Tensor) or value.shape != expected[name].shape or value.dtype != expected[name].dtype
                or value.layout != torch.strided or value.is_floating_point() and not bool(torch.all(torch.isfinite(value)))):
            raise ValueError("model tensor shape/dtype/finite value differs from caller template")
    fields(payload["optimizers"], groups, "checkpoint optimizer owners")
    fields(payload["schedulers"], groups, "checkpoint scheduler owners")
    if any(type(payload[name]) is not list for name in ("inherited_input_sha256", "inherited_origin_sha256")):
        raise ValueError("original checkpoint seal lists required without coercion")
    with _RNG_LOCK:
        original_rng = capture_rng(tuple(devices))
        try:
            model = copy.deepcopy(template)
            model.load_state_dict(clone(payload["model"]), strict=True)
            model.nnuec.restore_metadata_state(copy.deepcopy(payload["nnuec_metadata"]))
            _, _, optimizers, schedulers = _fresh_optimizers(model, recipe, membership, torch)
            ledger = payload["ledger"]
            owner = OwnedTensorState(model, recipe, copy.deepcopy(membership), stage.stage_id, optimizers, schedulers,
                groups, clone(payload["rng"]), copy.deepcopy(payload["samplers"]), payload["bucket"], tuple(devices),
                nnuec_metadata=copy.deepcopy(payload["nnuec_metadata"]),
                nnuec_ancestor_training_state=copy.deepcopy(nnuec_tensor_state["ancestor"]),
                model_metadata=copy.deepcopy(payload["model_metadata"]),
                optimizer_configuration={name: copy.deepcopy(value.state_dict()["param_groups"]) for name, value in optimizers.items()},
                scheduler_configuration={name: copy.deepcopy(value.state_dict()) for name, value in schedulers.items()},
                active_role=ledger["active_role"], completed=copy.deepcopy(ledger["completed"]),
                dispatched=copy.deepcopy(ledger["dispatched"]), parameter_updates=copy.deepcopy(payload["parameter_updates"]),
                cost=copy.deepcopy(ledger["cost"]), ancestor_cost=copy.deepcopy(ledger["ancestor_cost"]),
                ancestor_updates=copy.deepcopy(ledger["ancestor_updates"]),
                pending_microbatches=ledger["pending_microbatches"], parent_checkpoint_sha256=payload["parent_checkpoint_sha256"],
                inherited_input_sha256=list(payload["inherited_input_sha256"]), inherited_origin_sha256=list(payload["inherited_origin_sha256"]))
            _ledger(owner)
            for name in ("inherited_input_sha256", "inherited_origin_sha256"):
                values = getattr(owner, name)
                if len(values) > 262144 or len(set(sha(value) for value in values)) != len(values):
                    raise ValueError("bounded unique inherited training seals required")
            if stage.parent_stage_id is None:
                if owner.parent_checkpoint_sha256 is not None:
                    raise ValueError("common tensor state cannot have a foreign parent")
            else:
                sha(owner.parent_checkpoint_sha256)
            for identity, optimizer in optimizers.items():
                incoming = payload["optimizers"][identity]
                fields(incoming, ("state", "param_groups"), "AdamW checkpoint state")
                if not equal(incoming["param_groups"], optimizer.state_dict()["param_groups"]):
                    raise ValueError("AdamW options/order/slots differ from registered owner")
                optimizer.load_state_dict(clone(incoming))
                schedule = payload["schedulers"][identity]
                current = schedulers[identity].state_dict()
                fields(schedule, current, "constant scheduler state")
                count = _schedule_count(owner, groups[identity])
                current.update(last_epoch=count, _step_count=count + 1, _last_lr=current["base_lrs"])
                if not equal(schedule, current):
                    raise ValueError("constant scheduler/RNG owner update boundary mismatch")
                schedulers[identity].load_state_dict(clone(schedule))
            # Preserve the raw incoming moments and reject any silent cast by
            # the scratch optimizer loader before publishing an owner.
            for identity, optimizer in optimizers.items():
                loaded = optimizer.state_dict()
                if not equal(loaded, payload["optimizers"][identity]):
                    raise ValueError("optimizer load coerced incoming dtype/value")
            _optimizer_snapshot(owner, torch, clone)
            owner.activate_role(owner.active_role)
            tensor_payload(owner, sampler_validator=sampler_validator, execute=True)
            return owner
        finally:
            restore_rng(original_rng, tuple(devices))


def stage_tensor_resume(path, expected_sha256, template, recipe, membership, *, sampler_validator,
                        max_file_bytes, expected_stage_id, execute=False, read_accounting=None, snapshot_directory=None):
    """Return a fully checked new owner. Original model/global RNG are intact."""
    _require_execution(execute)
    validate_training_membership(membership)
    runtime = _runtime(execute)
    maximum = uint(max_file_bytes, MAX_BYTES)
    if not maximum:
        raise ValueError("positive checkpoint input byte credit required")
    payload, _ = _load_pinned_tensor_file(path, expected_sha256, maximum, runtime[0], read_accounting=read_accounting,
        snapshot_directory=snapshot_directory)
    owner = _restore_scratch(payload, template, recipe, membership, sampler_validator=sampler_validator, runtime=runtime)
    if owner.stage_id != expected_stage_id:
        raise ValueError("resume cannot change stage/teacher lineage")
    owner.source_checkpoint_sha256 = expected_sha256
    owner.source_checkpoint_path = str(path)
    owner.source_model_state_sha256 = runtime[5](payload["model"])
    owner.source_nnuec_metadata_sha256 = digest("rz-rovent-original-nnuec-metadata/1", payload["nnuec_metadata"])
    owner.source_payload_state_sha256 = runtime[5](payload)
    return owner


def stage_tensor_fork(path, expected_sha256, template, recipe, membership, *, source_sampler_validator,
                      target_sampler_validator, target_stage_id, reset_seed, reset_samplers, reset_bucket,
                      max_file_bytes, execute=False, read_accounting=None, snapshot_directory=None):
    """Load the pinned completed parent; retain weights, reset all child state.

    Both teacher branches can name this one common checkpoint byte pin. Parent
    cost and update history remain in separate ancestry counters; resetting the
    child's stage counters does not erase the cost or relabel its learned origin.
    """
    _require_execution(execute)
    validate_training_membership(membership)
    runtime = _runtime(execute)
    torch, _, _, capture_rng, restore_rng, _, _ = runtime
    target = recipe.stage(target_stage_id)
    if target.parent_stage_id is None:
        raise ValueError("common initialization is not a checkpoint fork")
    parent = stage_tensor_resume(path, expected_sha256, template, recipe, membership,
        sampler_validator=source_sampler_validator, max_file_bytes=max_file_bytes,
        expected_stage_id=target.parent_stage_id, execute=True, read_accounting=read_accounting,
        snapshot_directory=snapshot_directory)
    source_stage = recipe.stage(parent.stage_id)
    if parent.completed != dict(zip(ROLES, source_stage.role_updates)):
        raise ValueError("fork requires a completed declared parent checkpoint")
    uint(reset_seed, 2**32 - 1)
    if type(reset_bucket) is not int or reset_bucket not in BUCKETS:
        raise ValueError("explicit registered child bucket required")
    _validate_json_state(reset_samplers)
    target_sampler_validator(copy.deepcopy(reset_samplers))
    with _RNG_LOCK:
        original_rng = capture_rng(parent.cuda_devices)
        try:
            import random
            import numpy as np
            random.seed(reset_seed)
            np.random.seed(reset_seed)
            torch.random.default_generator.manual_seed(reset_seed)
            for device in parent.cuda_devices:
                torch.cuda.default_generators[device].manual_seed(reset_seed)
            child_rng = capture_rng(parent.cuda_devices)
            model = copy.deepcopy(parent.model)
            parameters, groups, optimizers, schedulers = _fresh_optimizers(model, recipe, membership, torch)
            child = OwnedTensorState(model, recipe, copy.deepcopy(membership), target_stage_id, optimizers, schedulers,
                groups, child_rng, copy.deepcopy(reset_samplers), reset_bucket, parent.cuda_devices,
                nnuec_metadata=_nnuec_metadata(model),
                nnuec_ancestor_training_state=fork_nnuec_ancestor(_nnuec_tensor_training_state(parent), expected_sha256),
                model_metadata=_model_metadata(model),
                optimizer_configuration={name: copy.deepcopy(value.state_dict()["param_groups"]) for name, value in optimizers.items()},
                scheduler_configuration={name: copy.deepcopy(value.state_dict()) for name, value in schedulers.items()},
                parameter_updates=dict.fromkeys(parameters, 0), ancestor_cost=parent.total_cost(),
                ancestor_updates={role: uint(parent.ancestor_updates[role] + parent.completed[role]) for role in ROLES},
                parent_checkpoint_sha256=expected_sha256,
                inherited_input_sha256=list(parent.inherited_input_sha256),
                inherited_origin_sha256=list(parent.inherited_origin_sha256))
            first = next(role for role, limit in zip(ROLES, target.role_updates) if limit)
            child.activate_role(first)
            tensor_payload(child, sampler_validator=target_sampler_validator, execute=True)
            return child
        finally:
            restore_rng(original_rng, parent.cuda_devices)


class TensorOwnerSlot:
    """Host-side atomic publication, with admission closed by the sole learner.

    This is not a CUDA fence or an interprocess lock. The host must finish all
    outstanding learner work before replacing the owner and retain the old
    owner's resources until its physical lifetime has ended.
    """
    def __init__(self, owner):
        self.owner = owner

    def publish(self, staged, *, sampler_validator, execute=False):
        _, _, validate_rng, capture_rng, restore_rng, _, _ = _runtime(execute)
        if type(staged) is not OwnedTensorState or staged.quarantined:
            raise ValueError("checked staged owner required")
        with _RNG_LOCK:
            tensor_payload(staged, sampler_validator=sampler_validator, execute=True)
            validate_rng(staged.rng, staged.cuda_devices)
            original = capture_rng(staged.cuda_devices)
            old = self.owner
            try:
                restore_rng(staged.rng, staged.cuda_devices)
                self.owner = staged
            except BaseException:
                self.owner = old
                restore_rng(original, staged.cuda_devices)
                raise
            return old


def export_tensor_owner(owner, directory, *, kind, rules_profile_json, initialization_declaration=None,
                        learned_request=None, sampler_validator, max_output_bytes, execute=False, read_accounting=None):
    """Invoke the original PC and NNUEC writers from a pinned lifecycle owner.

    Untrained declarations remain declarations. Learned export additionally
    requires a live original trainer/native intake admission; reset fork
    counters never make that ancestry untrained again. The old fixed-recipe
    trainer and diagnostic eligibility flags do not grant that admission.
    No optimizer update, model forward or numerical inference is dispatched.
    """
    _require_execution(execute)
    validate_training_membership(owner.membership)
    runtime = _runtime(execute)
    if kind not in ("collection", "deployment"):
        raise ValueError("explicit collection/deployment export kind required")
    maximum = uint(max_output_bytes, MAX_BYTES)
    if not maximum:
        raise ValueError("positive export byte credit required")
    target = Path(directory)
    if not target.is_absolute() or target.exists() or target.is_symlink():
        raise ValueError("new explicit owned export directory required")
    if learned_request is None:
        fields(initialization_declaration, ("kind", "checkpoint_sha256"), "initialization export declaration")
        if initialization_declaration["kind"] != "untrained" or initialization_declaration["checkpoint_sha256"] != owner.source_checkpoint_sha256:
            raise ValueError("export requires explicit untrained declaration for this original checkpoint pin")
    elif initialization_declaration is not None:
        raise ValueError("learned and untrained source declarations cannot be combined")
    with _RNG_LOCK:
        payload = tensor_payload(owner, sampler_validator=sampler_validator, execute=True)
        if learned_request is None and (any(payload["ledger"]["completed"].values()) or any(payload["ledger"]["ancestor_updates"].values())
                or payload["inherited_input_sha256"] or payload["inherited_origin_sha256"]
                or payload["nnuec_metadata"].get("training") != {"kind": "untrained"}
                or not nnuec_is_unmodified(payload["nnuec_tensor_training_state"])):
            raise ValueError("learned tensor export requires original native training provenance, not lifecycle counters")
        if (owner.source_checkpoint_path is None or owner.source_checkpoint_sha256 is None
                or owner.source_payload_state_sha256 != runtime[5](payload)):
            raise ValueError("export requires an unchanged saved/resumed original tensor checkpoint")
        recovered = learned_request is not None and learned_request.window_spec.get("schema") == "rz-rovent-train-window/2"
        if recovered and read_accounting is None:
            read_accounting = learned_request.execution_authority._read_budget
        actual, checkpoint_bytes = _hash_file(Path(owner.source_checkpoint_path), MAX_BYTES, read_accounting=read_accounting)
        if actual != owner.source_checkpoint_sha256:
            raise ValueError("original checkpoint changed before export")
        original_training = None
        nnuec_training = {"kind": "untrained"}
        training_steps = 0
        if learned_request is not None:
            from .rovent_training_window import admit_learned_export
            original_training = admit_learned_export(owner, {"sha256": actual, "bytes": checkpoint_bytes}, learned_request)
            nnuec_training = original_training["nnuec_training"]
            training_steps = sum(original_training["completed_updates"].values())
        # The full immutable configuration and metadata were validated above.
        # Work only on a clone so export/eval mode cannot alter the live owner.
        original_rng = runtime[3](owner.cuda_devices)
        written = 0
        def charge(count):
            nonlocal written
            uint(count)
            if count > maximum - written:
                raise ValueError("owned export byte credit exceeded")
            written += count
        try:
            from .artifacts import export_virtual_checkpoint, output_directory
            from .generation_export import _atomic_json
            scratch = copy.deepcopy(owner.model)
            scratch.eval()
            nnuec_count = sum(parameter.numel() for name, parameter in scratch.named_parameters() if name.startswith("nnuec."))
            # JSON's numeric lexical expansion is bounded before the writer
            # materializes its document. The writer separately enforces 32MiB.
            nnuec_reserve = min(32 * 1024**2, nnuec_count * 32 + 131072)
            if nnuec_reserve > maximum:
                raise ValueError("insufficient predeclared NNUEC JSON export credit")
            output = output_directory(target, output_guard=charge)
            manifest = export_virtual_checkpoint(scratch, {"config": scratch.config.to_dict(),
                "checkpoint_sha256": actual, "trained": original_training is not None, "training_steps": training_steps}, output / "pc",
                include_validator=kind == "collection" and owner.membership["validator_present"],
                rules_profile_json=rules_profile_json, output_guard=charge)
            wanted_roles = ["proposer", "critic"] + (["validator"] if kind == "collection" and owner.membership["validator_present"] else [])
            if manifest["roles"] != wanted_roles or manifest["validator_present"] != ("validator" in wanted_roles):
                raise ValueError("actual PC exporter did not enforce collection/deployment V membership")
            graph_pins = []
            for graph in manifest["graphs"]:
                name = graph["file"]
                if type(name) is not str or Path(name).name != name:
                    raise ValueError("PC exporter graph must be one owned regular filename")
                identity, size = _hash_file(output / "pc" / name, maximum, read_accounting=read_accounting)
                if identity != graph["sha256"]:
                    raise ValueError("actual PC graph changed after exporter publication")
                graph_pins.append({"role": graph["role"], "file": "pc/" + name, "sha256": identity, "bytes": size})
            manifest_identity, manifest_size = _hash_file(output / "pc" / "export.json", maximum, read_accounting=read_accounting)
            if nnuec_reserve > maximum - written:
                raise ValueError("insufficient remaining NNUEC JSON export credit")
            nnuec = scratch.nnuec.write_checkpoint(output / "nnuec.json", training=nnuec_training)
            identity, size = _hash_file(output / "nnuec.json", nnuec_reserve, read_accounting=read_accounting)
            if identity != nnuec["sha256"] or size != nnuec["bytes"]:
                raise ValueError("actual NNUEC JSON writer byte receipt mismatch")
            charge(size)
            sha(nnuec["weights_sha256"])
            if nnuec["schema"] != payload["nnuec_metadata"]["model_schema"]:
                raise ValueError("NNUEC writer changed its independently registered model domain")
            receipt = {"schema": "rz-rovent-tensor-lifecycle-export/2" if recovered else "rz-rovent-tensor-lifecycle-export/1", "eligibility": ELIGIBILITY if original_training is None
                    else "actual_training_pending_numeric_and_gpu_acceptance",
                "kind": kind, "recipe_sha256": owner.recipe.sha256, "stage_id": owner.stage_id,
                "checkpoint": {"sha256": actual, "bytes": checkpoint_bytes},
                "membership_sha256": membership_digest(owner.membership),
                "initialization_declaration": copy.deepcopy(initialization_declaration),
                ("historical_training_admission" if recovered else "original_training_admission"): original_training,
                "pc": {"roles": wanted_roles, "validator_present": "validator" in wanted_roles,
                    "manifest": {"file": "pc/export.json", "sha256": manifest_identity, "bytes": manifest_size},
                    "graphs": graph_pins},
                "nnuec": {"file": "nnuec.json", **nnuec},
                "output_bytes_before_receipt": written, "actual_export_written": True,
                "optimizer_updates_executed": 0, "numeric_status": "not_run", "cuda_status": "not_run",
                "training_provenance_verified": original_training is not None, "arena_eligible": False}
            # Detect unsupported concurrent mutation before publishing evidence.
            if runtime[5](tensor_payload(owner, sampler_validator=sampler_validator, execute=True)) != owner.source_payload_state_sha256:
                raise ValueError("live tensor owner changed during export")
            _atomic_json(output / "rovent-export.json", receipt, output_guard=charge)
            owner.cost["output_bytes"] = uint(owner.cost["output_bytes"] + written)
            return receipt
        except BaseException:
            # Files and failed reservations remain attributable to this owner.
            # A new explicit owner/run must investigate partial publication.
            owner.quarantined = True
            raise
        finally:
            runtime[4](original_rng, owner.cuda_devices)
