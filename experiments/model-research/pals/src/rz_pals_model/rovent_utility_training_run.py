"""Finite N05 utility-only learner dispatch, separate from P/C/V recipes.

Source implementation only does not authorize training. The execute boundary
requires the concrete current parent socket owner and original sealed pair
dataset. No raw JSON assertion or generic callback can substitute either.
"""
from __future__ import annotations

import copy
from pathlib import Path
import time

from .generation_data import uint
from .rovent_utility_training import COHORT_PREPARATION_DOMAIN, normalize_known_pairs
from .rovent_utility_tensor_training import (_runtime, _validate_actual, _device_roster,
    MAX_UPDATES, save_utility_checkpoint)
from .rovent_utility_training_window import (validate_utility_window_v2, validate_utility_sampler,
    next_utility_sample)


RESULT_DOMAIN = "rz-rovent-utility-learner-window-result/1"
UPDATE_DOMAIN = "rz-rovent-utility-learner-update/1"
WINDOW_DOMAIN = "rz-rovent-utility-train-window/2"


def _intake(value):
    # The reader issues a data-integrity owner, not execution authority. The
    # current Rust parent's held action/cost/publication owner separately joins
    # every source_projection before a sample is admitted to a live window.
    from .rovent_utility_route import HistoricalUtilityRouteDataset
    if type(value) is not HistoricalUtilityRouteDataset:
        raise ValueError("original typed utility action/pair dataset required")
    return value.observation()


def _sampler_validator(intake, spec):
    return lambda value: validate_utility_sampler(value, context_ids=intake.context_ids,
        pool_sha256=intake.pool_sha256, run_id=spec["run_id"], seed=spec["training_seed"])


def _verify_runtime(owner, spec, torch, *, family=None):
    from .rovent_utility_training_family import LEGACY, require_family
    family = require_family(LEGACY if family is None else family)
    parts = family.parts(owner.preparation) if family.intervention else None
    parameters, _ = _validate_actual(owner.model, owner.preparation, torch, numerical_parts=parts)
    _device_roster(parameters, owner.cuda_devices)
    device = next(iter(parameters.values())).device
    if (str(device) != spec["device"] or torch.get_num_threads() != spec["cpu_threads"]
            or torch.get_num_interop_threads() != spec["interop_threads"]
            or not torch.are_deterministic_algorithms_enabled()
            or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
            or torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda")):
        raise ValueError("actual utility device/thread/FP32/determinism runtime differs from window")


def _fence(owner, torch):
    try:
        for device in owner.cuda_devices:
            torch.cuda.synchronize(device)
        owner.physical_completion = "confirmed"
    except BaseException:
        owner.physical_completion = "unknown"
        owner.quarantined = True
        raise


def _charge_wall(owner, started, clock):
    now = clock()
    if now < started:
        raise ValueError("original utility monotonic clock moved backwards")
    owner.cost["wall_time_ns"] = uint(owner.cost["wall_time_ns"] + now - started)
    return now


def _remaining_output(owner, initial_output_bytes, spec):
    # Window output credit applies to bytes published by this process/window,
    # while the checkpoint cost ledger retains prior-window numeric history.
    spent = owner.cost["output_bytes"] - initial_output_bytes
    if spent < 0 or spent >= spec["max_output_bytes"]:
        raise ValueError("utility current-window output credit exhausted")
    return min(256 * 1024**2, spec["max_output_bytes"] - spent)


def _execution_cutoffs(spec, *, started_ns, execution_until_ns, expected_domain=WINDOW_DOMAIN):
    """Reserve final publication inside, never beyond, the original window."""
    if type(spec) is not dict or spec.get("schema") != expected_domain:
        raise ValueError("explicit checkpoint-reserved utility window/2 required")
    if not all(key in spec for key in ("max_wall_time_ns", "checkpoint_reserve_ns")):
        raise ValueError("explicit utility wall window and checkpoint reserve required")
    started, original = uint(started_ns), uint(execution_until_ns)
    duration, reserve = uint(spec["max_wall_time_ns"]), uint(spec["checkpoint_reserve_ns"])
    if not 0 < reserve < duration:
        raise ValueError("positive utility checkpoint reserve below original wall window required")
    execution = min(original, uint(started + duration))
    if execution < reserve:
        raise ValueError("original utility deadline cannot hold its registered checkpoint reserve")
    return execution, execution - reserve


def train_utility_window(owner, intake, spec, *, authority=None, checkpoint_directory=None,
                         execute=False, clock=time.monotonic_ns):
    from .rovent_utility_training_family import LEGACY
    return _train_utility_window_core(owner, intake, spec, authority=authority,
        checkpoint_directory=checkpoint_directory, execute=execute, clock=clock, family=LEGACY)


def _train_utility_window_core(owner, intake, spec, *, authority, checkpoint_directory,
                               execute, clock, family):
    """Consume only original known route pairs; all-masked work never steps.

    Each completed accumulation has exactly eight actual context occurrences.
    Its gradient denominator is the total number of unique known pairs across
    those contexts, rather than the number of rows or repeated discoveries.
    An interrupted accumulation is discarded with its actual cost retained.
    """
    from .rovent_utility_training_family import require_family
    family = require_family(family)
    checked = family.validate_window(owner.preparation, spec)
    expected_window = "rz-rovent-utility-train-window/3" if family.intervention else WINDOW_DOMAIN
    if checked["schema"] != expected_window:
        raise ValueError("actual utility loop requires explicit checkpoint-reserved window/2")
    if not family.intervention and owner.preparation["schema"] != COHORT_PREPARATION_DOMAIN:
        raise ValueError("actual utility supervision requires explicit cohort preparation/2")
    numerical_parts = family.parts(owner.preparation) if family.intervention else None
    remaining_updates = MAX_UPDATES - owner.total_updates()
    if checked["max_optimizer_updates"] > remaining_updates:
        raise ValueError("utility window exceeds remaining cumulative optimizer update credit")
    observation = family.require_dataset(intake)
    if observation["dataset_sha256"] != checked["dataset_sha256"]:
        raise ValueError("utility window dataset differs from original publication")
    sampler_validator = _sampler_validator(intake, checked)
    sampler_validator(copy.deepcopy(owner.sampler))
    if execute is not True:
        return {"schema": family.domain(RESULT_DOMAIN), "status": "prepared_only", "window_id": checked["window_id"],
            "completed_updates": 0, "dispatched_updates": 0, "groups": 0, "contexts": 0,
            "execution_authorized": False, "actual_training_executed": False}
    authority = family.require_authority(authority, allow_unleased=True)
    runtime = _runtime(True)
    torch, _, _, capture_rng, restore_rng, _, _ = runtime
    _verify_runtime(owner, checked, torch, family=family)
    if owner.quarantined or owner.pending_microbatches or any(value.grad is not None for value in owner.model.parameters()):
        raise ValueError("utility learner cannot dispatch a quarantined or partial owner")
    target = Path(checkpoint_directory)
    if (not target.is_absolute() or target.resolve(strict=True) != target or not target.is_dir()
            or target.is_symlink()):
        raise ValueError("explicit original owned utility checkpoint directory required")
    lease = authority.begin_window(owner, intake, checked)
    started = clock()
    charged_at = started
    deadline, admission_deadline = _execution_cutoffs(checked, started_ns=started,
        execution_until_ns=authority.window_deadline_ns(lease), expected_domain=expected_window)

    def post_snapshot_observer(checkpoint):
        nonlocal charged_at
        charged_at = _charge_wall(owner, charged_at, clock)
        checkpoint["post_snapshot_cost"] = dict(owner.cost)
    # The reserve is inside the unchanged original execution window. It is
    # never a new lease or permission to run forward/update during cleanup.
    updates, groups, contexts, last_checkpoint = 0, 0, 0, None
    initial_output_bytes = owner.cost["output_bytes"]
    status = "window_limit"
    group_started = None
    # The parent/child bootstrap owns exclusive process RNG admission. Resume
    # restores all checkpoint RNG state before the first admitted sample.
    restore_rng(owner.rng, owner.cuda_devices)
    try:
        while updates < checked["max_optimizer_updates"]:
            if (groups >= checked["max_groups"] or contexts + checked["accumulation"] > checked["max_contexts"]
                    or clock() >= admission_deadline or authority.should_stop(lease)):
                break
            samples, denominator = [], 0
            group_started = clock()
            for _ in range(checked["accumulation"]):
                if clock() >= admission_deadline or authority.should_stop(lease):
                    break
                index, sampler = next_utility_sample(owner.sampler, context_ids=intake.context_ids,
                    pool_sha256=intake.pool_sha256, run_id=checked["run_id"], seed=checked["training_seed"])
                sample = intake.sample(index)
                pairs = normalize_known_pairs(sample["known_pairs"])
                # Consumption is acknowledged against the original held source
                # before its tensor is ever used, and journaled once per row.
                authority.consume_pair_context(lease, sample["source_projection"])
                owner.sampler = sampler
                samples.append((sample, pairs["pairs"]))
                denominator += pairs["unique_known_pairs"]
                contexts += 1
                owner.cost["samples"] = uint(owner.cost["samples"] + 1)
                owner.pending_microbatches += 1
            if len(samples) != checked["accumulation"]:
                owner.cost["partial_accumulation_ns"] = uint(owner.cost["partial_accumulation_ns"] + clock() - group_started)
                owner.pending_microbatches = 0
                status = "partial_accumulation_discarded"
                break
            groups += 1
            if not denominator:
                owner.cost["masked_groups"] = uint(owner.cost["masked_groups"] + 1)
                owner.pending_microbatches = 0
                group_started = None
                continue
            owner.optimizer.zero_grad(set_to_none=True)
            for sample, pairs in samples:
                if not pairs:
                    continue
                if clock() >= admission_deadline or authority.should_stop(lease):
                    break
                # These count actual call admissions, including failed calls;
                # physical completion is independently fenced and attested.
                owner.cost["forward_calls"] = uint(owner.cost["forward_calls"] + 1)
                outputs = owner.model(sample["input"].reference_input)
                logits = outputs.raw_outputs
                if (logits.shape != (7,) or logits.dtype != torch.float32
                        or not bool(torch.isfinite(logits).all().item())):
                    raise ValueError("finite original utility seven-logit output required")
                terms = [torch.nn.functional.softplus(logits[other] - logits[preferred])
                    for preferred, other in pairs]
                loss = torch.stack(terms).sum() / denominator
                if not bool(torch.isfinite(loss).item()):
                    raise ValueError("utility known-pair loss is nonfinite")
                if clock() >= admission_deadline or authority.should_stop(lease):
                    break
                owner.cost["backward_calls"] = uint(owner.cost["backward_calls"] + 1)
                loss.backward()
            else:
                parameters, _ = _validate_actual(owner.model, owner.preparation, torch, numerical_parts=numerical_parts)
                if any(value.grad is None or not bool(torch.isfinite(value.grad).all().item())
                       for value in parameters.values()):
                    raise ValueError("all four active utility parameters require finite gradients")
                norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()),
                    float(owner.preparation["recipe"]["optimizer"]["gradient_clip_norm"]), error_if_nonfinite=True)
                if not bool(torch.isfinite(norm).item()):
                    raise ValueError("utility gradient norm is nonfinite")
                if clock() >= admission_deadline or authority.should_stop(lease):
                    owner.cost["partial_accumulation_ns"] = uint(owner.cost["partial_accumulation_ns"] + clock() - group_started)
                    owner.optimizer.zero_grad(set_to_none=True)
                    owner.pending_microbatches = 0
                    status = "partial_accumulation_discarded"
                    break
                if owner.total_updates() >= MAX_UPDATES:
                    raise ValueError("utility cumulative update credit exhausted before optimizer dispatch")
                before_model_sha256 = runtime[5](owner.model.state_dict())
                # Measuring the current tensor bytes also costs time. Check
                # again at the actual optimizer admission boundary.
                if clock() >= admission_deadline or authority.should_stop(lease):
                    owner.cost["partial_accumulation_ns"] = uint(owner.cost["partial_accumulation_ns"] + clock() - group_started)
                    owner.optimizer.zero_grad(set_to_none=True)
                    owner.pending_microbatches = 0
                    status = "partial_accumulation_discarded"
                    break
                owner.dispatched_updates += 1
                # This is the only N05 optimizer dispatch. Exceptions before an
                # original parent completion ACK quarantine the partial owner.
                owner.optimizer.step()
                owner.scheduler.step()
                owner.optimizer.zero_grad(set_to_none=True)
                _fence(owner, torch)
                if runtime[5](owner.model.state_dict()) == before_model_sha256:
                    raise ValueError("utility optimizer dispatch changed no active parameter bytes")
                owner.completed_updates += 1
                owner.pending_microbatches = 0
                owner.rng = capture_rng(owner.cuda_devices)
                charged_at = _charge_wall(owner, charged_at, clock)
                event = {"schema": family.domain(UPDATE_DOMAIN), "completed_updates": owner.completed_updates,
                    "dispatched_updates": owner.dispatched_updates, "unique_known_pair_denominator": denominator,
                    "consumed_context_sha256": [sample["source_projection"]["context_sha256"] for sample, _ in samples],
                    "groups": 1, "contexts": len(samples)}
                receipt = authority.completed_update(lease, event, owner)
                owner.completion_receipts.append(receipt)
                updates += 1
                group_started = None
                _validate_actual(owner.model, owner.preparation, torch, numerical_parts=numerical_parts)
                if owner.completed_updates % checked["checkpoint_every_updates"] == 0:
                    charged_at = _charge_wall(owner, charged_at, clock)
                    maximum = _remaining_output(owner, initial_output_bytes, checked)
                    last_checkpoint = family.save_checkpoint(owner,
                        target / f"utility-{owner.completed_updates:08d}.pt", sampler_validator=sampler_validator,
                        max_file_bytes=maximum, read_accounting=authority.read_accounting, execute=True,
                        authority=authority, lease=lease, post_snapshot_observer=post_snapshot_observer)
                    charged_at = _charge_wall(owner, charged_at, clock)
                    if not family.intervention:
                        last_checkpoint["post_snapshot_cost"] = dict(owner.cost)
                        authority.checkpoint_published(lease, last_checkpoint, owner)
                continue
            # A stop while accumulating gradients cannot become a checkpoint
            # of partially updated moments or an extra optimizer step.
            owner.cost["partial_accumulation_ns"] = uint(owner.cost["partial_accumulation_ns"] + clock() - group_started)
            owner.optimizer.zero_grad(set_to_none=True)
            owner.pending_microbatches = 0
            status = "partial_accumulation_discarded"
            break
        if updates == checked["max_optimizer_updates"]:
            status = "window_complete"
        _fence(owner, torch)
        owner.rng = capture_rng(owner.cuda_devices)
        charged_at = _charge_wall(owner, charged_at, clock)
        # The final boundary includes skipped/partial sampler progress and its
        # cost even when the last successful update was already checkpointed.
        if clock() >= deadline:
            raise TimeoutError("original utility execution deadline expired before final checkpoint")
        if family.intervention and owner.completed_updates <= lease.initial_completed:
            # A quiescent all-masked/stopped window is an observed zero-update
            # window, not a newly learned checkpoint or publication ACK.
            # Original cost and consumed sampler progress still reach end.
            last_checkpoint = None
            status = "no_update_checkpoint_not_published"
        else:
            maximum = _remaining_output(owner, initial_output_bytes, checked)
            last_checkpoint = family.save_checkpoint(owner, target / f"utility-final-{checked['window_id']}.pt",
                sampler_validator=sampler_validator, max_file_bytes=maximum,
                read_accounting=authority.read_accounting, execute=True, authority=authority, lease=lease,
                post_snapshot_observer=post_snapshot_observer)
            charged_at = _charge_wall(owner, charged_at, clock)
            if not family.intervention:
                last_checkpoint["post_snapshot_cost"] = dict(owner.cost)
                authority.checkpoint_published(lease, last_checkpoint, owner)
        charged_at = _charge_wall(owner, charged_at, clock)
        authority.end_window(lease, owner, status, physical_completion="confirmed",
            dispatched_updates=updates, checkpoint=last_checkpoint)
        return {"schema": family.domain(RESULT_DOMAIN), "status": status, "window_id": checked["window_id"],
            "completed_updates": updates, "dispatched_updates": updates, "groups": groups, "contexts": contexts,
            "checkpoint": last_checkpoint, "physical_completion": owner.physical_completion,
            "cost": copy.deepcopy(owner.cost), "actual_training_executed": bool(updates),
            "arena_eligible": False}
    except BaseException:
        owner.quarantined = True
        owner.physical_completion = "unknown"
        if group_started is not None:
            owner.cost["partial_accumulation_ns"] = uint(owner.cost["partial_accumulation_ns"] + max(0, clock() - group_started))
        _charge_wall(owner, charged_at, clock)
        # The original supervisor retains live unknown execution and partials;
        # this error is never converted to a clean learned/checkpoint receipt.
        raise
