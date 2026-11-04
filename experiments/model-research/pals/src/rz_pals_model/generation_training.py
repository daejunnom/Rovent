"""First actual generation trainer, separate from preparation/smoke domains.

Import and recipe/ledger validation use the standard library only. Torch and
model operations are lazy and require an explicit execute=True dispatch. Source
tests can forbid those imports; implementing this module does not train a model.
"""
import copy
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import tempfile
import time

from .artifacts import publish_noreplace
from .generation_data import (ROLES, TASKS, GenerationDataset, admit_phase, digest, fields, identifier,
    sha, uint, validate_record, coverage)
from .generation_chunks import ChunkGenerationDataset, ChunkRows, MANIFEST_DOMAIN

SPEC_DOMAIN = "rz-pals-train-generation/1"
CHECKPOINT_DOMAIN = "rz-pals-actual-generation-checkpoint/1"
EXPORT_DOMAIN = "rz-pals-actual-generation-export/1"
ELIGIBILITY = "actual_training_pending_acceptance"
OPTIMIZER = {"name": "adamw", "lr": 1e-4, "betas": [0.9, 0.999], "epsilon": 1e-8,
             "weight_decay": 0.01, "clip_norm": 1.0, "schedule": "constant"}
UPDATES = {"proposer": 2000, "critic": 2000, "validator": 1000}
HEADS = ("policy", "wdl", "divergence", "task")
INFERENCE_PROFILE = {"mode": "fresh", "batch_size": 1}


def generation_spec(*, generation_id, source_sha256, dataset_sha256, split_sha256,
                    model_sha256, runtime_sha256, model_epoch=0, context_epoch=0):
    value = {"schema": SPEC_DOMAIN, "generation_id": generation_id, "source_sha256": source_sha256,
             "dataset_sha256": dataset_sha256, "split_sha256": split_sha256,
             "model_sha256": model_sha256, "runtime_sha256": runtime_sha256,
             "profile": "virtual_review_v1", "model_epoch": model_epoch, "context_epoch": context_epoch,
             "device": "cuda:0", "precision": "fp32", "tf32": False, "cpu_threads": 2,
             "microbatch": 1, "accumulation": 8, "optimizer": copy.deepcopy(OPTIMIZER),
             "collection_inference": dict(INFERENCE_PROFILE),
             "role_updates": dict(UPDATES), "checkpoint_interval": 250,
             "seeds": {"initialization": 23, "sampling": 29, "training": 31, "arena": 1},
             "loss_weights": dict.fromkeys(HEADS, 1.0), "eligibility": ELIGIBILITY}
    validate_spec(value)
    return value


def validate_spec(spec):
    fields(spec, ("schema", "generation_id", "source_sha256", "dataset_sha256", "split_sha256", "model_sha256", "runtime_sha256",
                  "profile", "model_epoch", "context_epoch", "device", "precision", "tf32", "cpu_threads",
                  "microbatch", "accumulation", "optimizer", "role_updates", "checkpoint_interval", "seeds", "loss_weights", "eligibility", "collection_inference"), "generation spec")
    identifier(spec["generation_id"])
    for name in ("source_sha256", "dataset_sha256", "split_sha256", "model_sha256", "runtime_sha256"):
        sha(spec[name])
    for name in ("model_epoch", "context_epoch"):
        uint(spec[name])
    expected = {"schema": SPEC_DOMAIN, "profile": "virtual_review_v1", "device": "cuda:0",
                "precision": "fp32", "tf32": False, "cpu_threads": 2, "microbatch": 1, "accumulation": 8,
                "optimizer": OPTIMIZER, "role_updates": UPDATES, "checkpoint_interval": 250,
                "seeds": {"initialization": 23, "sampling": 29, "training": 31, "arena": 1},
                "loss_weights": dict.fromkeys(HEADS, 1.0), "eligibility": ELIGIBILITY}
    expected["collection_inference"] = INFERENCE_PROFILE
    if any(not _typed_equal(spec[k], v) for k, v in expected.items()):
        raise ValueError("unregistered first-generation configuration")
    return spec


def _typed_equal(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(_typed_equal(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(_typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def spec_digest(spec):
    validate_spec(spec)
    return digest(SPEC_DOMAIN, spec)


@dataclass
class GenerationLedger:
    """Dispatch and successful optimizer boundaries are different counters.

    Partial accumulation costs remain charged after interruption. Checkpoint is
    forbidden until the accumulation has either completed or been explicitly
    discarded with its replay cost preserved.
    """
    spec: dict
    completed: dict = field(default_factory=lambda: dict.fromkeys(ROLES, 0))
    dispatched: dict = field(default_factory=lambda: dict.fromkeys(ROLES, 0))
    phase: str = "pc"
    next_role: str = "proposer"
    pending_microbatches: int = 0
    cost: dict = field(default_factory=lambda: dict.fromkeys(("forward_calls", "backward_calls", "samples",
                           "wall_time_ns", "partial_accumulation_ns", "replay_ns", "masked_groups", "output_bytes"), 0))

    def __post_init__(self):
        validate_spec(self.spec)
        self.spec = copy.deepcopy(self.spec)
        self.validate()

    def validate(self):
        validate_spec(self.spec)
        fields(self.completed, ROLES, "completed role updates")
        fields(self.dispatched, ROLES, "dispatched role updates")
        fields(self.cost, ("forward_calls", "backward_calls", "samples", "wall_time_ns", "partial_accumulation_ns", "replay_ns", "masked_groups", "output_bytes"), "generation cost")
        for role in ROLES:
            uint(self.completed[role], UPDATES[role])
            uint(self.dispatched[role])
            if self.completed[role] > self.dispatched[role]:
                raise ValueError("completed updates exceed actual dispatches")
        for value in self.cost.values():
            uint(value)
        uint(self.pending_microbatches, 8)
        if self.phase not in ("pc", "v", "complete") or self.next_role not in ROLES:
            raise ValueError("invalid generation phase/role")
        p, c, v = (self.completed[r] for r in ROLES)
        if self.phase == "pc" and (v or c == 2000 or p < c or p > c + 1 or self.next_role != ("proposer" if p == c else "critic")):
            raise ValueError("P/C role alternation mismatch")
        if self.phase in ("v", "complete") and (p != 2000 or c != 2000 or self.next_role != "validator"):
            raise ValueError("V phase requires frozen complete P/C")
        if self.phase == "v" and v == 1000:
            raise ValueError("V terminal update count requires complete phase")
        if self.phase == "complete" and v != 1000:
            raise ValueError("incomplete generation cannot report complete")
        return self

    def dispatch_update(self, role):
        self.validate()
        if role != self.next_role or self.completed[role] == UPDATES[role] or self.pending_microbatches:
            raise ValueError("wrong/exhausted role or outstanding accumulation")
        self.dispatched[role] += 1

    def complete_update(self, role):
        if role != self.next_role or self.pending_microbatches != 8 or self.completed[role] >= self.dispatched[role]:
            raise ValueError("optimizer completion requires dispatched full accumulation")
        self.completed[role] += 1
        self.pending_microbatches = 0
        if role == "proposer":
            self.next_role = "critic"
        elif role == "critic":
            self.next_role = "proposer"
            if self.completed["critic"] == 2000:
                self.phase, self.next_role = "v", "validator"
        elif self.completed["validator"] == 1000:
            self.phase = "complete"
        self.validate()

    def checkpoint_boundary(self):
        self.validate()
        return (self.pending_microbatches == 0 and sum(self.completed.values()) > 0
                and sum(self.completed.values()) % self.spec["checkpoint_interval"] == 0)

    def discard_partial(self, elapsed_ns, replay_ns=0):
        self.cost["partial_accumulation_ns"] += uint(elapsed_ns)
        self.cost["replay_ns"] += uint(replay_ns)
        self.pending_microbatches = 0

    def state(self):
        self.validate()
        return copy.deepcopy(asdict(self))


def head_counts(records, role):
    """Count actual supervised rows/pairs across the whole accumulation group."""
    result = dict.fromkeys(HEADS, 0)
    if role not in ROLES or len(records) != 8:
        raise ValueError("registered accumulation requires exactly eight microbatch rows")
    for record in records:
        if record["input"]["role"] != role:
            raise ValueError("heterogeneous role accumulation prohibited")
        targets = validate_record(record)
        for head in ("policy", "wdl", "divergence"):
            result[head] += int(targets[head] is not None)
        if targets["task"] is not None:
            result["task"] += len(targets["task"]["pairs"])
    return result


def parameter_owner(name):
    if name.startswith(("public_encoder.", "reader_blocks.", "move_embedding.")):
        return "shared"
    for role in ROLES:
        if name.startswith(f"experts.{role}."):
            return role
    raise ValueError(f"unmapped generation parameter responsibility: {name}")


class GenerationOptimizers:
    """One AdamW owns shared/P/C moments; V owns only its private moments."""
    def __init__(self, model, spec, *, execute=False):
        if not execute:
            raise ValueError("actual model/optimizer execution requires explicit authorization")
        validate_spec(spec)
        import torch
        model.config.validate()
        if (model.config.profile != "virtual_review_v1" or torch.get_num_threads() != 2 or torch.get_num_interop_threads() != 1
                or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
                or not torch.are_deterministic_algorithms_enabled()):
            raise ValueError("registered compact/CPU2/TF32-off deterministic runtime required")
        self.model, self.spec = model, copy.deepcopy(spec)
        self.members, self.parameters, aliases = dict.fromkeys(("shared",) + ROLES, ()), {}, {}
        grouped = {r: [] for r in self.members}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            owner = parameter_owner(name)
            if parameter.dtype != torch.float32 or parameter.device != torch.device(spec["device"]) or not bool(torch.all(torch.isfinite(parameter))):
                raise ValueError("registered generation requires finite CUDA FP32 parameters")
            previous = aliases.get(id(parameter))
            if previous is not None:
                if previous != owner:
                    raise ValueError("cross-owner parameter alias")
                continue
            aliases[id(parameter)] = owner
            grouped[owner].append((name, parameter))
            self.parameters[name] = parameter
        if any(not grouped[owner] for owner in grouped):
            raise ValueError("shared and all private experts must be observed")
        self.members = {owner: tuple(name for name, _ in pairs) for owner, pairs in grouped.items()}
        self.parameter_updates = dict.fromkeys(self.parameters, 0)
        self.pc = torch.optim.AdamW([{"params": [p for _, p in grouped[r]], "group_name": r, "weight_decay": 0.01}
                                    for r in ("shared", "proposer", "critic")], lr=1e-4, betas=(0.9, 0.999), eps=1e-8, foreach=False)
        self.v = torch.optim.AdamW([{"params": [p for _, p in grouped["validator"]], "group_name": "validator", "weight_decay": 0.01}],
                                  lr=1e-4, betas=(0.9, 0.999), eps=1e-8, foreach=False)
        self.schedulers = {"pc": torch.optim.lr_scheduler.LambdaLR(self.pc, lambda _: 1.0),
                           "v": torch.optim.lr_scheduler.LambdaLR(self.v, lambda _: 1.0)}
        self.frozen_pc_sha256 = None
        self.set_role("proposer")

    def set_role(self, role):
        if role not in ROLES or (role != "validator" and self.frozen_pc_sha256 is not None):
            raise ValueError("frozen P/C cannot be reactivated during V acceptance")
        active = {role} | ({"shared"} if role != "validator" else set())
        for name, parameter in self.parameters.items():
            parameter.requires_grad_(parameter_owner(name) in active)
            parameter.grad = None
        return self.v if role == "validator" else self.pc

    def freeze_pc(self):
        from .nonzero_training import _state_digest
        self.frozen_pc_sha256 = _state_digest({n: p.detach().cpu().clone() for n, p in self.parameters.items() if parameter_owner(n) != "validator"})
        self.set_role("validator")
        return self.frozen_pc_sha256

    def verify_frozen_pc(self):
        from .nonzero_training import _state_digest
        if self.frozen_pc_sha256 is not None and self.frozen_pc_sha256 != _state_digest({n: p.detach().cpu().clone() for n, p in self.parameters.items() if parameter_owner(n) != "validator"}):
            raise ValueError("shared/P/C changed during V phase")


def loss_sums(outputs, batch, targets, role, *, divergence_index=None):
    """Per-head sums; caller divides by that head's group-wide valid count.

    batch supplies actual prepared tensors and candidate masks. Input encoding
    is independent of future labels; the collator has to validate the input seal.
    """
    import torch
    from torch.nn import functional as F
    if role not in ROLES or len(outputs) != (3 if role == "proposer" else 4):
        raise ValueError("generation hard-routed output mismatch")
    if any(v.dtype != torch.float32 or not bool(torch.all(torch.isfinite(v))) for v in outputs):
        raise ValueError("nonfinite/non-FP32 generation output")
    zero = outputs[0].sum() * 0.0 + outputs[1].sum() * 0.0
    result = dict.fromkeys(HEADS, zero)
    device = outputs[0].device
    if role != "validator":
        for head, output in (("policy", outputs[0]), ("wdl", outputs[1])):
            if targets[head] is None:
                continue
            target = torch.tensor([targets[head]], dtype=torch.float32, device=device)
            if target.shape != output.shape:
                raise ValueError("target/output shape mismatch; broadcasting prohibited")
            logits = output
            if head == "policy":
                if batch.candidate_mask.shape != output.shape or batch.candidate_mask.dtype != torch.bool or not bool(batch.candidate_mask.any()):
                    raise ValueError("invalid legal policy support")
                logits = output.masked_fill(~batch.candidate_mask, -1e9)
            result[head] = -(target * F.log_softmax(logits, dim=-1)).sum()
    if role == "critic" and targets["divergence"] is not None:
        # V3 record carries one exact visible challenged counter target.
        index = divergence_index
        uint(index, outputs[3].shape[1] - 1)
        if not bool(batch.divergence_mask[0, index]):
            raise ValueError("divergence target points to masked input candidate")
        result["divergence"] = F.binary_cross_entropy_with_logits(outputs[3][0, index],
                                torch.tensor(targets["divergence"], device=device, dtype=torch.float32), reduction="sum")
    if role == "validator" and targets["task"] is not None:
        if outputs[3].shape != (1, len(TASKS)):
            raise ValueError("V task logits shape mismatch")
        pairs = targets["task"]["pairs"]
        result["task"] = sum((F.softplus(-(outputs[3][0, i] - outputs[3][0, j])) for i, j in pairs), zero)
    return result


def train_accumulation(model, optimizers, ledger, records, collate_and_forward, *, execute=False):
    """Eight rows, one role, one real optimizer update; all-masked groups skip."""
    if not execute:
        raise ValueError("actual generation updates are disabled without explicit execution")
    role = ledger.next_role
    counts = head_counts(records, role)
    if not any(counts.values()):
        ledger.cost["masked_groups"] += 1
        return {"optimizer_updated": False, "all_masked": True, "head_counts": counts}
    import torch
    if role == "validator" and optimizers.frozen_pc_sha256 is None:
        optimizers.freeze_pc()
    optimizer = optimizers.set_role(role)
    optimizer.zero_grad(set_to_none=True)
    ledger.dispatch_update(role)
    started = time.monotonic_ns()
    try:
        for record in records:
            targets = validate_record(record)
            batch, outputs = collate_and_forward(model, role, copy.deepcopy(record["input"]))
            ledger.cost["forward_calls"] += 1
            ledger.cost["samples"] += 1
            label = record["future_targets"]["divergence"]
            divergence_index = None if label is None else record["input"]["divergence_record_ids"].index(label["counter_record_id"])
            sums = loss_sums(outputs, batch, targets, role, divergence_index=divergence_index)
            loss = sum((sums[head] / counts[head] for head in HEADS if counts[head]), outputs[0].sum() * 0.0)
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite generation loss")
            loss.backward()
            ledger.cost["backward_calls"] += 1
            ledger.pending_microbatches += 1
        active = [p for p in optimizers.parameters.values() if p.requires_grad]
        if any(p.grad is not None and not bool(torch.all(torch.isfinite(p.grad))) for p in active):
            raise ValueError("nonfinite generation gradient")
        torch.nn.utils.clip_grad_norm_(active, 1.0, error_if_nonfinite=True)
        optimizer.step()
        for name, parameter in optimizers.parameters.items():
            if parameter.grad is not None:
                optimizers.parameter_updates[name] += 1
        optimizers.schedulers["v" if role == "validator" else "pc"].step()
        optimizer.zero_grad(set_to_none=True)
        ledger.complete_update(role)
        if ledger.phase == "v" and optimizers.frozen_pc_sha256 is None:
            optimizers.freeze_pc()
        optimizers.verify_frozen_pc()
    except BaseException:
        # Keep partial count/cost and dispatched update; no implicit retry/refund.
        raise
    finally:
        ledger.cost["wall_time_ns"] += time.monotonic_ns() - started
    return {"optimizer_updated": True, "role": role, "head_counts": counts,
            "completed_updates": dict(ledger.completed)}


def checkpoint_payload(model, optimizers, ledger, sampler, bucket, *, cuda_devices=(0,)):
    """Full AdamW moments, scheduler, RNG and owned sampler at update boundary."""
    if not ledger.checkpoint_boundary() or any(p.grad is not None for p in optimizers.parameters.values()):
        raise ValueError("checkpoint requires clean registered optimizer update boundary")
    if tuple(cuda_devices) != (0,) or type(tuple(cuda_devices)[0]) is not int:
        raise ValueError("checkpoint must preserve registered L4 CUDA RNG owner")
    from .training import _clone_state, capture_rng
    optimizers.verify_frozen_pc()
    return {"schema": CHECKPOINT_DOMAIN, "eligibility": ELIGIBILITY, "spec": copy.deepcopy(ledger.spec),
            "config": model.config.to_dict(),
            "spec_sha256": spec_digest(ledger.spec), "model": _clone_state(model.state_dict()),
            "optimizers": {"pc": _clone_state(optimizers.pc.state_dict()), "v": _clone_state(optimizers.v.state_dict())},
            "schedulers": {k: _clone_state(s.state_dict()) for k, s in optimizers.schedulers.items()},
            "membership": copy.deepcopy(optimizers.members), "frozen_pc_sha256": optimizers.frozen_pc_sha256,
            "parameter_updates": dict(optimizers.parameter_updates),
            "ledger": ledger.state(), "sampler": copy.deepcopy(sampler), "bucket": bucket,
            "rng": capture_rng(cuda_devices), "cuda_devices": list(cuda_devices)}


class _BoundedCheckpointWriter:
    """Stop checkpoint serialization before it crosses declared disk credit."""
    def __init__(self, stream, maximum):
        self.stream, self.maximum = stream, uint(maximum)

    def write(self, value):
        if self.stream.tell() + len(value) > self.maximum:
            raise ValueError("generation checkpoint exceeds remaining declared disk credit")
        return self.stream.write(value)

    def flush(self):
        return self.stream.flush()

    def tell(self):
        return self.stream.tell()


def _serialize_checkpoint(payload, stream):
    """The existing serializer; publication tests can use small ASCII bytes."""
    import torch
    torch.save(payload, stream)


def atomic_checkpoint(path, payload, *, max_file_bytes=None):
    """Fsync then no-replace rename; preserve failed partials within output credit.

    Never unlink the old temporary name after publication: it may have been
    recreated by another owner, including after a known/unknown publish error.
    """
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError("generation checkpoint already registered")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    base_bytes = payload["ledger"]["cost"]["output_bytes"]
    available = 18 * 1024**3 - uint(base_bytes, 18 * 1024**3)
    if max_file_bytes is not None:
        available = min(available, uint(max_file_bytes, 18 * 1024**3))
    if not available:
        raise ValueError("generation checkpoint has no remaining declared disk credit")
    previous_size = 0
    for _ in range(8):
        payload["ledger"]["cost"]["output_bytes"] = base_bytes + previous_size
        if payload["ledger"]["cost"]["output_bytes"] > 18 * 1024**3:
            raise ValueError("managed generation staging/cache 18GiB cap exceeded")
        with open(temporary, "wb") as stream:
            _serialize_checkpoint(payload, _BoundedCheckpointWriter(stream, available))
            stream.flush()
            os.fsync(stream.fileno())
        size = os.path.getsize(temporary)
        if size == previous_size:
            break
        previous_size = size
    else:
        raise ValueError("generation checkpoint byte accounting did not stabilize")
    publish_noreplace(temporary, target)
    return target


def validate_checkpoint(payload, model, optimizers, spec, sampler_validator):
    """Validate scratch state before caller model, optimizers or RNG mutate."""
    import torch
    from .training import _validate_rng
    fields(payload, ("schema", "eligibility", "spec", "spec_sha256", "config", "model", "optimizers", "schedulers", "membership", "parameter_updates",
                     "frozen_pc_sha256", "ledger", "sampler", "bucket", "rng", "cuda_devices"), "actual generation checkpoint")
    if payload["schema"] != CHECKPOINT_DOMAIN or payload["eligibility"] != ELIGIBILITY or payload["spec"] != spec or payload["spec_sha256"] != spec_digest(spec) or not _typed_equal(payload["config"], model.config.to_dict()):
        raise ValueError("diagnostic/other generation checkpoint cannot resume this generation")
    ledger = GenerationLedger(**copy.deepcopy(payload["ledger"]))
    if ledger.spec != spec or not ledger.checkpoint_boundary() or payload["membership"] != optimizers.members or payload["bucket"] not in (8, 16, 32, 64, 128, 256):
        raise ValueError("generation checkpoint boundary/membership/bucket mismatch")
    validate_checkpoint_phase(ledger, payload["frozen_pc_sha256"])
    if payload["cuda_devices"] != [0] or type(payload["cuda_devices"][0]) is not int:
        raise ValueError("checkpoint RNG devices differ from registered L4 owner")
    expected = model.state_dict()
    fields(payload["parameter_updates"], tuple(optimizers.parameters), "per-parameter actual update counts")
    for name, count in payload["parameter_updates"].items():
        owner = parameter_owner(name)
        maximum = sum(ledger.completed[r] for r in ("proposer", "critic")) if owner == "shared" else ledger.completed[owner]
        uint(count, maximum)
    if type(payload["model"]) is not dict or set(payload["model"]) != set(expected):
        raise ValueError("model checkpoint fields mismatch")
    for name, value in payload["model"].items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape or value.dtype != expected[name].dtype or not bool(torch.all(torch.isfinite(value))):
            raise ValueError("invalid model checkpoint tensor")
    fields(payload["optimizers"], ("pc", "v"), "generation optimizer owners")
    fields(payload["schedulers"], ("pc", "v"), "generation schedules")
    for owner in ("pc", "v"):
        scratch = copy.deepcopy(getattr(optimizers, owner))
        incoming = payload["optimizers"][owner]
        current = scratch.state_dict()
        if (type(incoming) is not dict or set(incoming) != {"state", "param_groups"}
                or incoming["param_groups"] != current["param_groups"]):
            raise ValueError("optimizer group/configuration mismatch")
        parameters = [p for group in scratch.param_groups for p in group["params"]]
        ids = [pid for group in current["param_groups"] for pid in group["params"]]
        names = [name for group in current["param_groups"] for name in optimizers.members[group["group_name"]]]
        if set(incoming["state"]) - set(ids):
            raise ValueError("foreign optimizer moment ownership")
        for pid, state in incoming["state"].items():
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("incomplete AdamW moments")
            parameter = parameters[ids.index(pid)]
            for name in ("exp_avg", "exp_avg_sq"):
                if state[name].shape != parameter.shape or state[name].dtype != torch.float32 or not bool(torch.all(torch.isfinite(state[name]))):
                    raise ValueError("invalid AdamW moment shape/dtype/value")
            if bool(torch.any(state["exp_avg_sq"] < 0)):
                raise ValueError("AdamW second moment cannot be negative")
            step = state["step"]
            actual_steps = payload["parameter_updates"][names[ids.index(pid)]]
            if not isinstance(step, torch.Tensor) or step.dtype != torch.float32 or step.ndim != 0 or not bool(torch.isfinite(step)) or step.item() != actual_steps or actual_steps <= 0:
                raise ValueError("invalid actual AdamW step")
        for group in current["param_groups"]:
            for pid in group["params"]:
                if bool(payload["parameter_updates"][names[ids.index(pid)]]) != (pid in incoming["state"]):
                    raise ValueError("missing/unearned optimizer moments for per-parameter update count")
        scratch.load_state_dict(copy.deepcopy(incoming))
        schedule = copy.deepcopy(optimizers.schedulers[owner])
        schedule_state = payload["schedulers"][owner]
        expected_schedule = schedule.state_dict()
        if type(schedule_state) is not dict or set(schedule_state) != set(expected_schedule):
            raise ValueError("scheduler checkpoint fields mismatch")
        steps = sum(ledger.completed[r] for r in ("proposer", "critic")) if owner == "pc" else ledger.completed["validator"]
        if (schedule_state["last_epoch"] != steps or schedule_state["_step_count"] != steps + 1
                or schedule_state["base_lrs"] != expected_schedule["base_lrs"]
                or schedule_state["_last_lr"] != expected_schedule["base_lrs"]):
            raise ValueError("constant schedule/update counter mismatch")
        schedule.load_state_dict(copy.deepcopy(schedule_state))
    if payload["frozen_pc_sha256"] is not None:
        from .nonzero_training import _state_digest
        sha(payload["frozen_pc_sha256"])
        actual = _state_digest({n: p for n, p in payload["model"].items() if parameter_owner(n) != "validator"})
        if actual != payload["frozen_pc_sha256"]:
            raise ValueError("P/C freeze digest mismatch")
    sampler_validator(copy.deepcopy(payload["sampler"]))
    _validate_rng(payload["rng"], tuple(payload["cuda_devices"]))
    return ledger


def validate_checkpoint_phase(ledger, frozen_pc_sha256):
    """Reject contradictory freeze state before any caller state is restored."""
    ledger.validate()
    if ledger.phase == "pc" and frozen_pc_sha256 is not None:
        raise ValueError("P/C checkpoint cannot already be frozen")
    if ledger.phase in ("v", "complete") and frozen_pc_sha256 is None:
        raise ValueError("V checkpoint lacks P/C freeze evidence")
    if frozen_pc_sha256 is not None:
        sha(frozen_pc_sha256)


def restore_checkpoint(payload, model, optimizers, spec, sampler_validator):
    ledger = validate_checkpoint(payload, model, optimizers, spec, sampler_validator)
    from .training import restore_rng
    model.load_state_dict(copy.deepcopy(payload["model"]), strict=True)
    optimizers.pc.load_state_dict(copy.deepcopy(payload["optimizers"]["pc"]))
    optimizers.v.load_state_dict(copy.deepcopy(payload["optimizers"]["v"]))
    for k, scheduler in optimizers.schedulers.items():
        scheduler.load_state_dict(copy.deepcopy(payload["schedulers"][k]))
    optimizers.frozen_pc_sha256 = payload["frozen_pc_sha256"]
    optimizers.parameter_updates = dict(payload["parameter_updates"])
    optimizers.set_role(ledger.next_role)
    restore_rng(payload["rng"], tuple(payload["cuda_devices"]))
    return ledger, copy.deepcopy(payload["sampler"]), payload["bucket"]


def generation_admission(spec, dataset, phase, *, frozen_pc_sha256=None, model_epoch=None):
    """Validation-only CLI path: no model or optimizer loading."""
    validate_spec(spec)
    if type(dataset) is not ChunkGenerationDataset:
        raise ValueError("actual generation requires independently admitted chunk producer/trajectory/native-chain dataset")
    splits = dataset.opening_splits
    if len(splits) != 64 or {name: list(splits.values()).count(name) for name in ("train", "validation", "holdout")} != {"train": 48, "validation": 8, "holdout": 8}:
        raise ValueError("actual generation requires preregistered 64 openings split 48/8/8")
    if phase == "pc" and dataset.sha256 != spec["dataset_sha256"]:
        raise ValueError("P/C actual dataset differs from registered generation spec")
    if phase == "pc" and dataset.authority["model_sha256"] != spec["model_sha256"]:
        raise ValueError("P/C actual collection used a different G0 model")
    if phase == "pc" and len(dataset.authority["actual_game_ids"]) != 128:
        raise ValueError("first-generation P/C collection requires actual fixed 128 games")
    if (dataset.authority["source_sha256"] != spec["source_sha256"]
            or dataset.authority["runtime_sha256"] != spec["runtime_sha256"]
            or dataset.authority["opening_split_sha256"] != spec["split_sha256"]):
        raise ValueError("actual generation producer/source/runtime/split mismatch")
    if phase == "v":
        sha(frozen_pc_sha256)
        if type(model_epoch) is not int or model_epoch <= spec["model_epoch"] or dataset.authority["model_sha256"] != frozen_pc_sha256:
            raise ValueError("V actual collection must use registered newly frozen P/C")
    elif phase != "pc" or frozen_pc_sha256 is not None or model_epoch is not None:
        raise ValueError("unexpected generation phase/freeze override")
    records = dataset.rows("train", optimization=True)
    observed = dataset.admit(phase)
    if phase == "v" and observed.get("validator_unique_inputs", 0) > 8192:
        raise ValueError("V action collection exceeds 8192 context bound")
    expected_epoch = spec["model_epoch"] if phase == "pc" else model_epoch
    for record in records:
        if (record["input"]["model_epoch"] != expected_epoch or record["input"]["context_epoch"] != spec["context_epoch"]):
            raise ValueError("generation/model/context epoch mismatch")
    return {"schema": SPEC_DOMAIN, "spec_sha256": spec_digest(spec), "dataset_sha256": dataset.sha256,
            "frozen_pc_sha256": frozen_pc_sha256, "phase": phase, "coverage": observed,
            "data_format": MANIFEST_DOMAIN, "producer": dataset.authority,
            "opening_splits": splits, "model_epoch": expected_epoch, "context_epoch": spec["context_epoch"],
            "native_chain_verified": True,
            "inference_profile": dict(INFERENCE_PROFILE), "profile_exclusions": dataset.exclusions,
            "actual_training_executed": False, "optimizer_updates": 0, "arena_eligible": False}


class GenerationSampler:
    """Owned deterministic role stream with sealed pool, epoch and exact cursor."""
    def __init__(self, records, role, *, seed=31, state=None):
        import random
        if role not in ROLES or type(seed) is not int or seed != 31:
            raise ValueError("registered training role/seed required")
        self.role, self.seed = role, seed
        if type(records) is ChunkRows:
            self.records = records
            self.indices, self.pool_sha256 = records.role_pool(role)
            self.algorithm = "python-mt19937-generation-chunk-role-v1"
        else:
            self.records = copy.deepcopy(records)
            self.indices = [i for i, r in enumerate(records) if r["input"]["role"] == role]
            for i in self.indices:
                validate_record(records[i])
            self.pool_sha256 = digest("rz-pals-generation-sampler-pool/1", [records[i]["input_sha256"] for i in self.indices])
            self.algorithm = "python-mt19937-generation-role-v1"
        if not self.indices or len(self.indices) > 262144:
            raise ValueError("bounded nonempty generation role pool required")
        self.epoch, self.cursor, self.samples = 0, 0, 0
        self.permutation = list(self.indices)
        if state is not None:
            fields(state, ("algorithm", "role", "seed", "pool_sha256", "epoch", "cursor", "samples", "permutation"), "generation sampler checkpoint")
            if (state["algorithm"] != self.algorithm or state["role"] != role
                    or state["seed"] != seed or state["pool_sha256"] != self.pool_sha256
                    or type(state["permutation"]) is not list or sorted(state["permutation"]) != sorted(self.indices)):
                raise ValueError("generation sampler role/pool/permutation mismatch")
            self.epoch = uint(state["epoch"])
            self.cursor = uint(state["cursor"], len(self.indices))
            self.samples = uint(state["samples"])
            if self.samples != self.epoch * len(self.indices) + self.cursor:
                raise ValueError("generation sampler cumulative position mismatch")
            expected = list(self.indices)
            random.Random(digest("rz-pals-generation-sampler-seed/1", [seed, role, self.epoch])).shuffle(expected)
            if state["permutation"] != expected:
                raise ValueError("generation sampler seeded order mismatch")
            self.permutation = list(state["permutation"])
        else:
            random.Random(digest("rz-pals-generation-sampler-seed/1", [seed, role, self.epoch])).shuffle(self.permutation)

    def next_accumulation(self):
        import random
        result = []
        for _ in range(8):
            if self.cursor == len(self.permutation):
                self.epoch += 1
                self.cursor = 0
                self.permutation = list(self.indices)
                random.Random(digest("rz-pals-generation-sampler-seed/1", [self.seed, self.role, self.epoch])).shuffle(self.permutation)
            result.append(copy.deepcopy(self.records[self.permutation[self.cursor]]))
            self.cursor += 1
            self.samples += 1
        return result

    def state(self):
        return {"algorithm": self.algorithm, "role": self.role, "seed": self.seed,
                "pool_sha256": self.pool_sha256, "epoch": self.epoch, "cursor": self.cursor,
                "samples": self.samples, "permutation": list(self.permutation)}


def compact_collate_and_forward(model, role, snapshot):
    """Use MI's selected-payload encoder directly; future targets cannot enter."""
    from .generation_data import validate_input
    from .model import VirtualReviewTensorInput
    validate_input(snapshot)
    batch = VirtualReviewTensorInput.from_payload(snapshot["encoded_model_input"], device=next(model.parameters()).device)
    batch.validate(model.config)
    memory = model.public_encoder(*batch.public_args(model.config))
    output = model.role_graph(role)(*batch.role_args(memory, model.config))
    # Caller sets the target index from exact sealed divergence provenance;
    # a target-free encoder never sees that future label.
    return batch, output


def configure_generation_runtime(spec, *, execute=False, initialize_rng=False):
    """Explicit first-window setup; resumed windows never reset their RNG."""
    if not execute:
        raise ValueError("actual generation runtime setup requires explicit authorization")
    validate_spec(spec)
    import torch
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("registered deterministic CUBLAS_WORKSPACE_CONFIG must be set before CUDA initialization")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("registered sole CUDA L4 device unavailable; no CPU fallback")
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if initialize_rng:
        import random
        import numpy as np
        random.seed(spec["seeds"]["training"])
        np.random.seed(spec["seeds"]["training"])
        torch.random.default_generator.manual_seed(spec["seeds"]["training"])
        torch.cuda.default_generators[0].manual_seed(spec["seeds"]["training"])
    return {"device": "cuda:0", "cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "rng_initialized": initialize_rng, "device_name": torch.cuda.get_device_name(0)}


def deployment_state(model, ledger, *, checkpoint_sha256):
    """V is removed, not merely disabled, in the actual deployment export."""
    ledger.validate()
    if ledger.phase != "complete" or ledger.pending_microbatches:
        raise ValueError("incomplete actual generation cannot export deployment candidate")
    sha(checkpoint_sha256)
    from .training import _clone_state
    state = {name: value for name, value in model.state_dict().items() if not name.startswith("experts.validator.")}
    if any(name.startswith("experts.validator.") for name in state):
        raise ValueError("V entered deployment export")
    return {"schema": EXPORT_DOMAIN, "eligibility": "evaluation_candidate_pending_gpu_acceptance",
            "generation_id": ledger.spec["generation_id"], "source_checkpoint_sha256": checkpoint_sha256,
            "spec_sha256": spec_digest(ledger.spec), "actual_updates": dict(ledger.completed),
            "include_validator": False, "model": _clone_state(state), "arena_accepted": False}


def train_generation_window(spec, dataset, model, *, optimizers=None, ledger=None, sampler_states=None,
                            frozen_pc_sha256=None, model_epoch=None, checkpoint_directory,
                            max_wall_ns=60 * 60 * 10**9, execute=False, clock=time.monotonic_ns,
                            collate_and_forward=compact_collate_and_forward, output_guard=None,
                            progress_observer=None, checkpoint_observer=None):
    """One bounded 250-update window of the real generation; never an auto-loop.

    Collection is a separate Rust producer. The caller resumes with validated
    checkpoint state, and must collect a fresh V dataset after the P/C freeze.
    A window deadline returns the owned state and charged partial work; it does
    not claim the generation complete or silently restart from zero.
    """
    if not execute:
        raise ValueError("actual train-generation execution requires explicit authorization")
    uint(max_wall_ns)
    if max_wall_ns == 0:
        raise ValueError("finite positive generation window required")
    ledger = GenerationLedger(spec) if ledger is None else ledger.validate()
    if ledger.spec != spec or ledger.phase == "complete" or ledger.pending_microbatches:
        raise ValueError("resume requires matching generation and clean accumulation boundary")
    generation_admission(spec, dataset, ledger.phase, frozen_pc_sha256=frozen_pc_sha256, model_epoch=model_epoch)
    optimizers = GenerationOptimizers(model, spec, execute=True) if optimizers is None else optimizers
    if optimizers.model is not model or optimizers.spec != spec:
        raise ValueError("optimizer/model generation owner mismatch")
    if ledger.phase == "v" and (optimizers.frozen_pc_sha256 != frozen_pc_sha256 or frozen_pc_sha256 is None):
        raise ValueError("V window must retain exact frozen P/C owner")
    records = dataset.rows("train", optimization=True)
    roles = ("proposer", "critic") if ledger.phase == "pc" else ("validator",)
    if sampler_states is not None and set(sampler_states) != set(roles):
        raise ValueError("sampler roles differ from generation phase")
    samplers = {role: GenerationSampler(records, role, state=None if sampler_states is None else sampler_states[role]) for role in roles}
    def publish_owned_progress():
        if progress_observer is not None:
            # Same-process owner references retain cursors without copying a
            # large permutation every update. Serialize only at a boundary.
            progress_observer(samplers, ledger)
    publish_owned_progress()
    started, start_completed = clock(), sum(ledger.completed.values())
    max_groups, groups, checkpoints = 65536, 0, []
    status = "window_complete"
    while sum(ledger.completed.values()) - start_completed < 250:
        elapsed = clock() - started
        if type(elapsed) is not int or elapsed < 0:
            raise ValueError("generation window clock moved backwards")
        if elapsed >= max_wall_ns:
            status = "window_expired"
            break
        if ledger.phase == "complete" or ledger.phase == "v" and roles == ("proposer", "critic"):
            status = "await_actual_v_collection" if ledger.phase == "v" else "generation_updates_complete"
            break
        if groups >= max_groups:
            raise ValueError("NoValidTargetProgress: bounded sampler groups exhausted")
        role = ledger.next_role
        if output_guard is not None:
            output_guard()
        group = samplers[role].next_accumulation()
        groups += 1
        try:
            # Original process transports use an absolute-deadline clock.
            # Admission/collation before this point cannot renew that deadline.
            clock()
            result = train_accumulation(model, optimizers, ledger, group, collate_and_forward, execute=True)
        finally:
            # The original process owner retains exact consumed sampler cursors
            # and charged partial work even if forward/backward/step fails.
            publish_owned_progress()
        if result["optimizer_updated"] and ledger.checkpoint_boundary():
            from .virtual_review import validate_payload
            bucket = validate_payload(group[-1]["input"]["encoded_model_input"])["line_bucket"]
            sampler = {r: s.state() for r, s in samplers.items()}
            payload = checkpoint_payload(model, optimizers, ledger, sampler, bucket)
            path = Path(checkpoint_directory) / f"{spec['generation_id']}-{sum(ledger.completed.values()):05d}.pt"
            available = None if output_guard is None else output_guard()
            saved = atomic_checkpoint(path, payload, max_file_bytes=available)
            ledger.cost["output_bytes"] = payload["ledger"]["cost"]["output_bytes"]
            checkpoints.append(str(saved))
            # Publish the durable boundary before any later disk/deadline
            # check can fail. Original process owners must retain this artifact
            # even when a terminal check prevents a normal return.
            if checkpoint_observer is not None:
                checkpoint_observer(str(saved), ledger)
            if output_guard is not None:
                output_guard()
    if ledger.phase == "complete":
        status = "generation_updates_complete"
    elif ledger.phase == "v" and roles == ("proposer", "critic"):
        status = "await_actual_v_collection"
    return {"schema": "rz-pals-generation-window/1", "status": status, "spec_sha256": spec_digest(spec),
            "dataset_sha256": dataset.sha256, "completed_updates": dict(ledger.completed), "usage": dict(ledger.cost),
            "window_wall_ns": clock() - started, "checkpoints": checkpoints,
            "sampler": {r: s.state() for r, s in samplers.items()}, "ledger": ledger.state(),
            "actual_training_executed": sum(ledger.completed.values()) > start_completed,
            "arena_eligible": False}, optimizers, ledger


def validate_generation_bundle(spec, bundle, phase, *, frozen_pc_sha256=None, model_epoch=None):
    """Preserve legacy evidence parsing; never admit it as actual training data.

    The legacy authority does not seal physical submission/completion/retirement.
    Its coverage is useful for a bounded diagnostic, not first-generation
    optimizer dispatch or actual export provenance.
    """
    validate_spec(spec)
    fields(bundle, ("producer", "trajectories", "opening_splits", "allowed_evidence", "records"), "generation input bundle")
    dataset = GenerationDataset(bundle["records"], producer=bundle["producer"], trajectories=bundle["trajectories"],
                                opening_splits=bundle["opening_splits"], allowed_evidence=bundle["allowed_evidence"])
    if phase not in ("pc", "v"):
        raise ValueError("unknown generation phase")
    return {"schema": SPEC_DOMAIN, "spec_sha256": spec_digest(spec), "dataset_sha256": dataset.sha256,
            "phase": phase, "coverage": coverage(dataset.rows("train", optimization=True)), "actual_training_admissible": False,
            "reason": "legacy_bundle_lacks_native_physical_receipt_chain", "actual_training_executed": False,
            "optimizer_updates": 0, "arena_eligible": False}


def validate_generation_chunks(spec, manifest, phase, *, corpus_root, index_root,
                               frozen_pc_sha256=None, model_epoch=None, expected_manifest_sha256=None):
    """Validation-only manifest ingestion with global joins and source pins.

P/C uses the existing spec dataset pin. V collection is independently pinned
by its caller because it is collected after frozen P/C, never substituted into
the registered G0 dataset identity. No optimizer/model is created here.
"""
    validate_spec(spec)
    expected = spec["dataset_sha256"] if phase == "pc" else expected_manifest_sha256
    with ChunkGenerationDataset(corpus_root, manifest, expected_manifest_sha256=expected, index_root=index_root) as dataset:
        result = generation_admission(spec, dataset, phase, frozen_pc_sha256=frozen_pc_sha256, model_epoch=model_epoch)
        result["data_format"] = MANIFEST_DOMAIN
        result["exclusions"] = dict(dataset.exclusions)
        return result


def main(argv=None):
    """Metadata by default; actual dispatch uses the separate opt-in child.

    The source-owned original-trainer child requires pinned startup bytes and
    original deadlines from its process owner. A metadata receipt cannot issue
    native game authority. No smoke command is reused for generation execution.
    """
    import argparse
    import sys
    actual = list(sys.argv[1:] if argv is None else argv)
    if "--execute" in actual:
        from .generation_training_cli import main as execute_generation
        return execute_generation(actual)
    from .generation_data import strict_json
    parser = argparse.ArgumentParser(prog="train-generation", description="actual generation data/recipe registration; no updates")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase", choices=("pc", "v"), required=True)
    parser.add_argument("--frozen-pc-sha256")
    parser.add_argument("--model-epoch", type=int)
    parser.add_argument("--dataset-manifest-sha256", help="externally registered semantic manifest digest, required for V chunks")
    args = parser.parse_args(actual)
    def read(path):
        target = Path(path)
        if target.name.lower().startswith(".env") or target.suffix.lower() in (".key", ".pem"):
            raise ValueError("secret file is not a generation input")
        with target.open("rb") as stream:
            raw = stream.read(16 * 1024 * 1024 + 1)
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError("bounded generation input file exceeded")
        return strict_json(raw.decode("utf-8", "strict"))
    spec, bundle = read(args.spec), read(args.dataset)
    if type(bundle) is dict and bundle.get("schema") == MANIFEST_DOMAIN:
        # CLI stays validation-only. P/C keeps its original spec pin; V is
        # collected after P/C freeze and requires a separately supplied pin.
        if args.phase == "v":
            if args.dataset_manifest_sha256 is None:
                raise ValueError("V chunk validation requires an externally registered manifest digest")
            sha(args.dataset_manifest_sha256)
        elif args.dataset_manifest_sha256 is not None:
            raise ValueError("P/C chunk validation uses the dataset pin in the spec")
        if os.name == "nt":
            parent = Path(os.environ["APPDATA"]) / "RoveZero" / "tmp"
        else:
            parent = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "rovezero" / "generation-index"
        parent.mkdir(parents=True, exist_ok=True)
        result = validate_generation_chunks(spec, bundle, args.phase, corpus_root=Path(args.dataset).resolve().parent,
                                            index_root=parent, frozen_pc_sha256=args.frozen_pc_sha256, model_epoch=args.model_epoch,
                                            expected_manifest_sha256=args.dataset_manifest_sha256)
    else:
        if args.dataset_manifest_sha256 is not None:
            raise ValueError("a manifest digest applies only to chunk datasets")
        result = validate_generation_bundle(spec, bundle, args.phase,
                                           frozen_pc_sha256=args.frozen_pc_sha256, model_epoch=args.model_epoch)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
