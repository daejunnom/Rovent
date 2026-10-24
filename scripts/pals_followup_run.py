#!/usr/bin/env python3
"""Finite local V4 preparation, native model validation and 120+1 pilot.

No compilation, package installation, remote GPU creation or automatic retry.
Planning is the default. --run consumes the independently pinned executable
under existing inherited CPU/cgroup limits and preserves an outside-Git receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
from managed_process import run_group
from managed_storage import checked, tree_size
from managed_outputs import OutputLease

GIB = 1024 ** 3
SCHEMA = "rz-pals-followup-local-wrapper/1"
COLAB_SCHEMA = "rz-pals-followup-colab-observed-wrapper/1"
RESOURCE_PROFILES = ("strict-local", "colab-observed")
EXTERNAL_GC_LEDGER_SCHEMA = "rovezero.external-gc-ledger.v1"
PROFILES = ("legacy_summary_v1", "full_line_v2", "interaction_head_v2", "full_line_interaction_v2")

# Run in a separate, supervised process. ORT 1.22's CUDA provider constructs
# CUDA driver wrappers during dlopen, before model Run. An absolute-path driver
# probe does not establish that the provider can resolve this SONAME. Keep the
# same environment as the native child and never repair the linker or load NN.
CUDA_DRIVER_LOOKUP = """
import ctypes, json
from pathlib import Path
try:
    driver = ctypes.CDLL('libcuda.so.1')
    with Path('/proc/self/maps').open('rb') as stream:
        data = stream.read(2 * 1024 * 1024 + 1)
    if len(data) > 2 * 1024 * 1024:
        raise ValueError('driver mapping observation exceeds bound')
    paths = set()
    for line in data.splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6:
            path = Path(fields[5].decode('utf-8'))
            if path.name.startswith('libcuda.so'):
                if not path.is_absolute() or str(path).endswith(' (deleted)'):
                    raise ValueError('driver mapping origin unavailable')
                paths.add(str(path))
    if len(paths) != 1:
        raise ValueError('one observed system driver image required')
    print(json.dumps({'status': 'available', 'resolved_path': paths.pop()}))
except (OSError, ValueError, UnicodeError) as error:
    print(json.dumps({'status': 'unavailable', 'error_type': type(error).__name__}))
    raise SystemExit(125)
"""


def sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("an independently registered lowercase SHA256 is required")
    return value


def regular(path, cap):
    if not path.is_absolute():
        raise ValueError("absolute paths are required")
    path = checked(path)
    if not path.is_file() or not 0 < path.stat().st_size <= cap:
        raise ValueError("bounded regular file required")
    return path


def pinned(path, digest, cap):
    path = regular(path, cap)
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    if hasher.hexdigest() != sha(digest):
        raise ValueError("actual pinned file bytes differ from registration")
    return path


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def bounded_json(path, maximum=256 * 1024):
    path = regular(path, maximum)
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def outside_git(path):
    if not path.is_absolute():
        raise ValueError("absolute output required")
    path = checked(path)
    for parent in (path, *path.parents):
        if (parent / ".git").exists():
            raise ValueError("generated output must remain outside Git")
    return path


def check_resources(cpus):
    """Read actual inherited policy; install no global or per-host setting."""
    if sys.platform != "linux" or len(cpus) != 2 or len(set(cpus)) != 2:
        raise ValueError("execution requires Linux and exactly two registered CPUs")
    if sorted(os.sched_getaffinity(0)) != sorted(cpus):
        raise ValueError("actual inherited affinity differs from the two registered CPUs")
    membership = Path("/proc/self/cgroup").read_text(encoding="utf-8")
    relative = next((line[3:] for line in membership.splitlines() if line.startswith("0::")), None)
    if relative is None or not relative.startswith("/") or any(p in (".", "..") for p in relative.split("/")):
        raise ValueError("actual cgroup-v2 membership required")
    root = Path("/sys/fs/cgroup") / relative.lstrip("/")
    limits = {"memory.high": 6 * GIB, "memory.max": 12 * GIB, "memory.swap.max": 0}
    for name, value in limits.items():
        if (root / name).read_text(encoding="ascii").strip() != str(value):
            raise ValueError("inherited memory.high/max/swap differs from 6GiB/12GiB/0")
    return {"cpu_affinity": sorted(cpus), **limits, "method": "actual inherited affinity and cgroup-v2 reads"}


def observe_resources(cpus):
    """Colab observations are never an admission of the strict local policy.

    Read only this process's inherited cgroup and permission capabilities. No
    affinity, quota, memory or cgroup setting is installed or silently changed.
    """
    requested = {"cpu_affinity": sorted(cpus), "cpu_quota": 2,
                 "memory.high": 6 * GIB, "memory.max": 12 * GIB,
                 "memory.swap.max": 0, "pids.max": 128}
    observations = {name: {"requested": value, "observed": None, "status": "unknown"}
                    for name, value in requested.items()}
    result = {"profile": "colab-observed", "platform": sys.platform,
              "strict_local_resources_admitted": False,
              "method": "read-only current inherited affinity/cgroup-v2 observations",
              "ancestor_effective_limits": "unknown", "settings_installed": False,
              "cgroup_write_performed": False, "observations": observations,
              "cgroup_membership": None, "cgroup_write_capability": {
                  "observed": None, "status": "unknown", "applied_settings_verified": False}}
    if sys.platform != "linux":
        result["execution_platform_status"] = "unfulfilled"
        return result
    result["execution_platform_status"] = "satisfied"
    try:
        affinity = sorted(os.sched_getaffinity(0))
        observations["cpu_affinity"].update(observed=affinity,
            status="satisfied" if affinity == sorted(cpus) else "unfulfilled")
    except (AttributeError, OSError) as error:
        observations["cpu_affinity"]["error_type"] = type(error).__name__
    try:
        membership = Path("/proc/self/cgroup").read_text(encoding="utf-8")
        relative = next((line[3:] for line in membership.splitlines() if line.startswith("0::")), None)
        if (relative is None or not relative.startswith("/")
                or any(part in (".", "..") for part in relative.split("/"))):
            raise ValueError("actual safe cgroup-v2 membership unavailable")
        root = Path("/sys/fs/cgroup") / relative.lstrip("/")
        result["cgroup_membership"] = relative
    except (OSError, ValueError) as error:
        result["cgroup_observation_error_type"] = type(error).__name__
        return result
    names = ("cpu.max", "memory.high", "memory.max", "memory.swap.max", "pids.max")
    for name in names:
        target = "cpu_quota" if name == "cpu.max" else name
        row = observations[target]
        try:
            raw = (root / name).read_text(encoding="ascii").strip()
            row["raw"] = raw
            if name == "cpu.max":
                fields = raw.split()
                if (len(fields) != 2 or not fields[1].isdecimal() or int(fields[1]) <= 0
                        or (fields[0] != "max" and not fields[0].isdecimal())):
                    raise ValueError("malformed cgroup CPU quota")
                value = "unlimited" if fields[0] == "max" else int(fields[0]) / int(fields[1])
            else:
                if raw != "max" and not raw.isdecimal():
                    raise ValueError("malformed cgroup limit")
                value = "unlimited" if raw == "max" else int(raw)
            row.update(observed=value, status="satisfied" if value == row["requested"] else "unfulfilled")
        except (OSError, ValueError) as error:
            row["error_type"] = type(error).__name__
    try:
        writable = os.access(root, os.W_OK) and all(os.access(root / name, os.W_OK) for name in names)
        result["cgroup_write_capability"].update(observed=writable,
            status="observed_capability_only" if writable else "unfulfilled")
    except OSError as error:
        result["cgroup_write_capability"]["error_type"] = type(error).__name__
    return result


def external_gc_observation(args):
    """Record requests and actual external GC separately; never invoke GC."""
    path, digest = args.external_gc_ledger, args.external_gc_ledger_sha256
    if (path is None) != (digest is None):
        raise ValueError("external GC ledger and independently registered SHA256 are required together")
    if path is None:
        return {"ledger": None, "request_count": "unknown", "actual_execution_count": "unknown",
                "actual_full_gc_calls": "unknown", "memory_contention_observed": "unknown",
                "product_GC_hooks": 0, "GC_invoked_by_wrapper": False,
                "performance_evidence_eligible": False}
    path = pinned(path, digest, 256 * 1024)
    value = bounded_json(path)
    requests, executions = value.get("requests"), value.get("actual_executions")
    if (value.get("schema") != EXTERNAL_GC_LEDGER_SCHEMA
            or not isinstance(requests, list) or not isinstance(executions, list)
            or len(requests) > 64 or len(executions) > 64
            or not (type(value.get("memory_contention_observed")) is bool
                    or value.get("memory_contention_observed") == "unknown")
            or value.get("performance_evidence_eligible") is not False):
        raise ValueError("bounded separate request/actual external GC ledger required")
    identifiers = set()
    for request in requests:
        identifier = request.get("id") if isinstance(request, dict) else None
        if (not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", identifier)
                or identifier in identifiers or request.get("product_hook") is not False):
            raise ValueError("unique external GC request IDs and product_hook=false required")
        identifiers.add(identifier)
    actual_calls = 0
    execution_ids = set()
    for execution in executions:
        identifier = execution.get("id") if isinstance(execution, dict) else None
        if (not isinstance(execution, dict) or execution.get("request_id") not in identifiers
                or not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", identifier)
                or identifier in execution_ids
                or execution.get("scope") != "owned_external_controller"
                or type(execution.get("full_gc_calls")) is not int
                or not 0 <= execution["full_gc_calls"] <= 64
                or execution.get("product_hook") is not False):
            raise ValueError("actual external GC events must reference requests and exclude product hooks")
        execution_ids.add(identifier)
        actual_calls += execution["full_gc_calls"]
    return {"ledger": {"path": str(path), "sha256": digest, "bytes": path.stat().st_size},
            "request_count": len(requests), "actual_execution_count": len(executions),
            "actual_full_gc_calls": actual_calls,
            "memory_contention_observed": value["memory_contention_observed"],
            "product_GC_hooks": 0, "GC_invoked_by_wrapper": False,
            "performance_evidence_eligible": False}


def check_colab_output_budget(args):
    """Bound only this attempt's fresh output, never models or shared runtime."""
    target = args.output_root / args.label if args.mode == "arena" else args.model_report
    if tree_size(target, allow_disappearing=True) > args.output_bytes_max:
        raise RuntimeError("owned Colab attempt exceeds registered finite output budget")


def check_colab_driver_lookup(receipt_path, cwd, environment, remaining, budget):
    """Check the provider's default host-driver lookup before any model child.

    This is host integration evidence, separate from the pinned 19-file vendor
    bundle, device readiness, physical completion and numerical acceptance.
    No cuInit, environment mutation, linker-cache change or retry is performed.
    """
    evidence = {"soname": "libcuda.so.1", "status": "unknown",
                "same_environment_as_native_child": True, "cu_init_performed": False,
                "model_executed": False, "linker_settings_changed": False,
                "tree_gone": True, "child_spawned": False, "exit_code": None}
    if remaining <= 0:
        evidence["status"] = "deadline_expired"
        return evidence
    log = outside_git(receipt_path.with_name(receipt_path.name + ".driver-lookup.log"))
    if log.exists():
        raise ValueError("driver lookup log must be a fresh outside-Git file")
    descriptors = []
    def bounded_probe():
        budget()
        if log.stat().st_size > 64 * 1024:
            raise RuntimeError("driver lookup log exceeds finite bound")
    try:
        with log.open("xb") as stream:
            sys.stdout.flush()
            sys.stderr.flush()
            for target in (1, 2):
                descriptors.append(os.dup(target))
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            code, gone, failure = run_group(
                [sys.executable, "-I", "-c", CUDA_DRIVER_LOOKUP], cwd, environment,
                min(10, remaining), bounded_probe)
    except (OSError, RuntimeError) as error:
        # A raised supervisor may have started a child. Do not invent a clean
        # process tree or allow the model child after an unconfirmed cleanup.
        evidence.update(status="probe_supervision_raised", tree_gone="unknown",
                        child_spawned="unknown", log=str(log),
                        supervisor_error_type=type(error).__name__)
        return evidence
    finally:
        for target, original in zip((1, 2), descriptors):
            os.dup2(original, target)
            os.close(original)
    evidence.update(exit_code=code, tree_gone=gone, child_spawned=True,
                    log=str(log), supervisor_error=failure)
    if not gone or failure is not None or code != 0:
        evidence["status"] = "lookup_or_supervision_failed"
        return evidence
    if log.stat().st_size > 64 * 1024:
        evidence["status"] = "invalid_probe_output"
        return evidence
    try:
        with log.open("rb") as stream:
            data = stream.read(64 * 1024 + 1)
        if len(data) > 64 * 1024:
            raise ValueError("driver lookup log exceeds finite bound")
        value = json.loads(data.decode("utf-8"), object_pairs_hook=unique_object)
        path = value["resolved_path"]
        if (set(value) != {"status", "resolved_path"} or value["status"] != "available"
                or not isinstance(path, str) or not Path(path).is_absolute()
                or not Path(path).name.startswith("libcuda.so") or path.endswith(" (deleted)")):
            raise ValueError("invalid driver lookup observation")
    except (OSError, ValueError, KeyError, TypeError, UnicodeError):
        evidence["status"] = "invalid_probe_output"
    else:
        evidence.update(status="available", resolved_path=path)
    return evidence


def arena_plan(args, binary):
    lock_path = pinned(args.lock, args.lock_sha256, 256 * 1024)
    lock = bounded_json(lock_path)
    if (lock.get("domain") != "rz-pals-arena-launch-v4/1"
            or lock.get("input", {}).get("domain") != "rz-pals-arena-launch-v4/1"):
        raise ValueError("an actual locked V4 arena envelope is required")
    sha(lock["sha256"])
    native_profile = lock["input"].get("resource_profile")
    wrapper_profile = getattr(args, "resource_profile", "strict-local")
    if wrapper_profile == "colab-observed":
        if native_profile != {"profile": "colab_observed"}:
            raise ValueError("Colab arena requires an explicitly locked native colab_observed resource profile")
    elif native_profile is not None:
        raise ValueError("strict-local wrapper cannot select a Colab-observed native arena lock")
    value = lock["input"]["semantic_lock"]["manifest"]
    pilot = value["pilot"]
    if (value.get("schema_version") != 4 or value.get("contract_revision") != "pals/0.2"
            or value.get("purpose") != "arena_pilot" or value.get("training_executed") is not False
            or pilot.get("games") != 2 or pilot.get("base_ms") != 120000 or pilot.get("increment_ms") != 1000
            or not 1 <= pilot.get("max_plies", 0) <= 256
            or not 1 <= pilot.get("wall_time_max_ms", 0) <= 900000
            or not 1 <= pilot.get("cleanup_max_ms", 0) <= 30000):
        raise ValueError("V4 pilot is frozen to exchanged 120+1 games, max256 ply, 15min+30s")
    cpus = sorted(args.cpu_affinity)
    if len(value["resources"]) != 2:
        raise ValueError("exactly two exchanged-game resource declarations required")
    for resource in value["resources"]:
        if (sorted(resource["cpu_affinity"]) != cpus or resource["cpu_threads"] != 2
                or resource["memory_high_bytes"] != 6 * GIB or resource["memory_max_bytes"] != 12 * GIB
                or resource["swap_max_bytes"] != 0):
            raise ValueError("V4 resource declaration differs from finite local CPU2/6GiB/12GiB policy")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", args.label):
        raise ValueError("bounded fresh attempt label required")
    assets = checked(args.asset_root)
    output = outside_git(args.output_root)
    if not assets.is_absolute() or not assets.is_dir() or not output.is_dir() or (output / args.label).exists():
        raise ValueError("existing absolute asset/output roots and fresh attempt label required")
    evidence = {
        "lock_file_sha256": args.lock_sha256, "arena_launch_sha256": lock["sha256"],
        "pilot": pilot, "resource_declarations_checked": True, "native_policy_admission": "Rust V4 owner before spawn",
    }
    if native_profile is not None:
        evidence.update(native_arena_resource_profile="colab_observed",
                        native_policy_admission="Rust V4 owner: explicit inherited observations; strict Core resource match remains Unknown",
                        pilot_pair_eligible=False, strength_eligible=False)
    return [str(binary), args.stage, str(lock_path), str(assets), str(output), args.label], evidence


def model_plan(args, binary):
    export = pinned(args.export, args.export_sha256, 256 * 1024)
    descriptor = bounded_json(export)
    profile = descriptor.get("config", {}).get("profile", "legacy_summary_v1")
    if profile != args.model_profile:
        raise ValueError("actual export profile differs from explicit registered profile")
    if profile != "legacy_summary_v1" and (
            descriptor.get("schema") != "rovezero.pals-model.v2"
            or descriptor.get("model_semantics") != "rovezero.pals-model-semantics.v2"
            or descriptor.get("encoding_schema") != "rovezero.pals-board-records.v2"):
        raise ValueError("new model semantics and encoding must be actual V2")
    fixtures = pinned(args.fixtures, args.fixtures_sha256, 32 * 1024 * 1024)
    runtime = pinned(args.runtime, args.runtime_sha256, 1024 * 1024 * 1024)
    report = outside_git(args.model_report)
    cache = outside_git(args.runtime_cache)
    if report.exists() or not report.parent.is_dir():
        raise ValueError("new model report under existing outside-Git parent required")
    command = [str(binary), str(export), args.export_sha256, str(runtime), args.runtime_sha256,
               str(fixtures), str(report), args.provider, str(cache)]
    if args.provider != "cpu":
        if args.cuda_bundle is None or args.cuda_bundle_sha256 is None:
            raise ValueError("CUDA validation needs an independently pinned actual runtime bundle")
        bundle = pinned(args.cuda_bundle, args.cuda_bundle_sha256, 256 * 1024)
        command.append(str(bundle))
    elif args.cuda_bundle is not None or args.cuda_bundle_sha256 is not None:
        raise ValueError("CPU validation cannot silently inherit CUDA assets")
    inventory_path = getattr(args, "control_inventory", None)
    inventory_sha = getattr(args, "control_inventory_sha256", None)
    control = args.provider in ("cuda-control", "cuda-control-shim", "cuda-control-conv-default")
    if control:
        if inventory_path is None or inventory_sha is None:
            raise ValueError("CUDA control validation requires an explicit pinned metadata inventory")
        inventory = pinned(inventory_path, inventory_sha, 1024 * 1024)
        static = bounded_json(inventory, 1024 * 1024)
        if (static.get("schema") != "rovezero.pals-static-control-inventory.v2"
                or static.get("manifest_sha256") != args.export_sha256
                or static.get("actual_provider_placement") != "not_observed"
                or static.get("cpu_allowlist") != "not_created"):
            raise ValueError("static control inventory does not describe this exact export")
        # Native Rust reconstructs provenance and audits placement before Run.
        # Static input alone grants neither blanket CPU fallback nor CUDA success.
        command.extend((str(inventory), inventory_sha))
    elif inventory_path is not None or inventory_sha is not None:
        raise ValueError("metadata inventory requires an explicit CUDA control provider")
    details = {"model_profile": profile, "export_sha256": args.export_sha256,
               "fixture_sha256": args.fixtures_sha256, "provider": args.provider,
               "cuda_control_inventory_sha256": inventory_sha if control else None,
               "native_placement_audit_required": control,
               "numeric_reference_only": True, "strength_eligible": False}
    if args.provider == "cuda-control-conv-default":
        details["cudnn_conv_search_diagnostic"] = {
            "scope": "single_requested_option_no_observed_cudnn_plan",
            "cudnn_conv_algo_search_requested": "DEFAULT",
            "native_loading_profile_requested": "eager-cuda12-cudnn9-v1",
            "backend_execution_descriptor": "requires_loaded_checker_receipt",
            "actual_cudnn_plan": "unknown", "numerical_note": "unknown",
            "tf32_requested": False, "tf32_actual": "unknown", "tf32_retry": "unknown",
            "production_default_changed": False, "warm_success_claim": False,
            "pilot_success_claim": False, "failure_cause": "not_proven",
        }
    return command, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--cpu-affinity", nargs=2, type=int, required=True)
    parser.add_argument("--seconds", type=int, default=960)
    parser.add_argument("--resource-profile", choices=RESOURCE_PROFILES, default="strict-local")
    parser.add_argument("--output-bytes-max", type=int, default=2 * GIB)
    parser.add_argument("--external-gc-ledger", type=Path)
    parser.add_argument("--external-gc-ledger-sha256")
    parser.add_argument("--run", action="store_true")
    commands = parser.add_subparsers(dest="mode", required=True)
    arena = commands.add_parser("arena")
    arena.add_argument("--stage", choices=("prepare", "execute"), default="prepare")
    arena.add_argument("--lock", type=Path, required=True)
    arena.add_argument("--lock-sha256", required=True)
    arena.add_argument("--asset-root", type=Path, required=True)
    arena.add_argument("--output-root", type=Path, required=True)
    arena.add_argument("--label", required=True)
    model = commands.add_parser("model-check")
    model.add_argument("--model-profile", choices=PROFILES, default="full_line_interaction_v2")
    model.add_argument("--export", type=Path, required=True)
    model.add_argument("--export-sha256", required=True)
    model.add_argument("--fixtures", type=Path, required=True)
    model.add_argument("--fixtures-sha256", required=True)
    model.add_argument("--runtime", type=Path, required=True)
    model.add_argument("--runtime-sha256", required=True)
    model.add_argument("--runtime-cache", type=Path, required=True)
    model.add_argument("--model-report", type=Path, required=True)
    model.add_argument("--provider", choices=("cpu", "cuda", "cuda-device", "cuda-control", "cuda-control-shim", "cuda-control-conv-default"), default="cpu")
    model.add_argument("--cuda-bundle", type=Path)
    model.add_argument("--cuda-bundle-sha256")
    model.add_argument("--control-inventory", type=Path)
    model.add_argument("--control-inventory-sha256")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 3600 or any(cpu < 0 for cpu in args.cpu_affinity) or len(set(args.cpu_affinity)) != 2:
        parser.error("finite 1..3600s window and two distinct nonnegative CPU IDs required")
    if not 1 <= args.output_bytes_max <= 2 * GIB:
        parser.error("finite output budget of 1 byte..2GiB required")
    receipt_path = outside_git(args.receipt)
    if receipt_path.exists() or not receipt_path.parent.is_dir():
        parser.error("receipt must be a fresh file under existing outside-Git parent")
    started = time.monotonic()
    gc_observation = external_gc_observation(args)
    binary = pinned(args.binary, args.binary_sha256, 256 * 1024 * 1024)
    command, details = arena_plan(args, binary) if args.mode == "arena" else model_plan(args, binary)
    receipt = {"schema": SCHEMA if args.resource_profile == "strict-local" else COLAB_SCHEMA,
               "mode": args.mode, "planning_only": not args.run,
               "command": command, "binary_sha256": args.binary_sha256, "window_seconds": args.seconds,
               "gpu_validation_window_seconds_max": 3600, "resource_policy": {"cpu_affinity": sorted(args.cpu_affinity),
                   "memory_high_bytes": 6 * GIB, "memory_max_bytes": 12 * GIB, "swap_max_bytes": 0},
               "actual_resources": "unknown", "training_executed": False, "strength_eligible": False,
               "automatic_retry": False, "tree_gone": "unknown", "exit_code": None, **details}
    if args.resource_profile == "colab-observed":
        receipt.update(resource_profile="colab-observed", strict_local_resources_admitted=False,
                       resource_policy_interpretation="requested targets, actual capability is independently observed",
                       cleanup_seconds_max=30, physical_workers_max=1, concurrent_wrapper_children_max=1,
                       output_bytes_max=args.output_bytes_max,
                       output_budget_scope="fresh attempt generated files only; inherited stdout/stderr bytes unknown",
                       inherited_stdio_bytes="unknown", process_exit_is_provider_fence=False,
                       native_arena_policy_relaxed=False, external_GC=gc_observation,
                       performance_evidence_eligible=False, remote_creation_performed=False,
                       new_cloud_cost_authorized=False, preparation_is_in_work_window=True)
    elif args.external_gc_ledger is not None:
        receipt.update(external_GC=gc_observation, performance_evidence_eligible=False)
    # Preserve the strict-local timeout start and receipt behavior. Colab's
    # independently named observation profile includes preparation in its window.
    if args.resource_profile == "strict-local":
        started = time.monotonic()
    pending_control = None
    if args.run:
        target = args.output_root / args.label if args.mode == "arena" else args.model_report
        observer = check_resources if args.resource_profile == "strict-local" else observe_resources
        # These checks do not launch a child or issue an output owner. A failed
        # resource observation must not leave an otherwise unused active lease.
        receipt["actual_resources"] = observer(args.cpu_affinity)
        check_colab_output_budget(args)
        environment = os.environ.copy()
        environment.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"})
        # Preserve the Rust consumer's fresh-path contract: it creates its own
        # attempt directory/file after this reservation, never a borrowed asset.
        receipt.update(tree_gone=True, child_spawned=False, producer_dispatch_attempted=False,
            producer_result_observed=False, producer_exit_code=None,
            producer_result_scope="original_supervisor_return_only_not_NN_execution_proof",
            status="failed_before_child_dispatch", exit_code=125,
            output_lifecycle={"acquired": False, "finish_attempted": False, "finish_returned": False})
        stage = "before_child_dispatch"
        output_lease = OutputLease(target, Path(__file__).resolve().parent.parent,
                                   reserve=args.output_bytes_max, deferred=True).acquire()
        try:
            receipt["output_lifecycle"]["acquired"] = True
            if args.resource_profile == "strict-local":
                def strict_budget():
                    check_resources(args.cpu_affinity)
                    check_colab_output_budget(args)
                    output_lease.check_budget()
                remaining = args.seconds
                budget = strict_budget
            else:
                remaining = args.seconds - (time.monotonic() - started)
                if sys.platform == "linux" and args.mode == "model-check" and args.provider != "cpu":
                    stage = "host_driver_lookup"
                    # A raised supervisor can have started a probe. Only its
                    # returned original closure observation may make this known.
                    receipt["tree_gone"] = "unknown"
                    receipt["host_driver_lookup"] = check_colab_driver_lookup(
                        receipt_path, binary.parent, environment, remaining,
                        lambda: check_colab_output_budget(args))
                    receipt["tree_gone"] = receipt["host_driver_lookup"]["tree_gone"]
                    remaining = args.seconds - (time.monotonic() - started)
                if receipt.get("host_driver_lookup", {}).get("status", "available") != "available":
                    receipt.update(exit_code=125, status="host_driver_lookup_failed_no_model_spawned")
                    remaining = None
                elif sys.platform != "linux" or remaining <= 0:
                    receipt.update(exit_code=125 if sys.platform != "linux" else 124,
                        status="unsupported_execution_platform" if sys.platform != "linux" else "preparation_deadline_expired")
                    remaining = None
                def observed_budget():
                    check_colab_output_budget(args)
                    output_lease.check_budget()
                budget = observed_budget
            if remaining is not None:
                stage = "before_producer_dispatch"
                check_colab_output_budget(args)
                receipt.update(tree_gone="unknown", child_spawned="unknown", producer_dispatch_attempted=True)
                stage = "producer_supervision"
                code, gone, failure = run_group(command, binary.parent, environment, remaining, budget)
                # Preserve the original result before any post-run observer can
                # fail. Neither a zero exit nor process closure proves NN work.
                receipt.update(producer_result_observed=True, producer_exit_code=code,
                    producer_tree_gone=gone, producer_supervisor_error=failure,
                    exit_code=code if failure is None and gone is True else 125,
                    tree_gone=gone, supervisor_error=failure,
                    status=("completed" if args.resource_profile == "strict-local" else "completed_observed_profile")
                        if code == 0 and gone is True and failure is None else "failed_ownership_preserved")
                stage = "post_producer_observation"
                if args.resource_profile == "colab-observed":
                    if gone is True:
                        check_colab_output_budget(args)
                    receipt["actual_resources_after"] = observe_resources(args.cpu_affinity)
        except BaseException as error:
            receipt.update(exit_code=125, status="failed_wrapper_supervision_or_observation",
                wrapper_error={"stage": stage, "type": type(error).__name__, "message": str(error)})
            if not isinstance(error, Exception):
                # Keep the original cancellation object through cleanup and
                # receipt publication, then propagate it rather than return 125.
                pending_control = error
        finally:
            # One attempt for every successfully acquired lease. Unknown child
            # ownership remains unknown and keeps the original active record.
            receipt["output_lifecycle"].update(finish_attempted=True, tree_gone_argument=receipt["tree_gone"])
            try:
                output_lease.finish(status=receipt["status"], tree_gone=receipt["tree_gone"])
                receipt["output_lifecycle"]["finish_returned"] = True
            except BaseException as error:
                receipt.update(status="failed_output_lifecycle", exit_code=125,
                    storage_lifecycle_error={"type": type(error).__name__, "message": str(error)})
                if pending_control is None and not isinstance(error, Exception):
                    pending_control = error
    else:
        receipt["status"] = "prepared_no_child_spawned"
    if pending_control is not None:
        receipt.update(status="interrupted_control_flow", exit_code=None,
            control_flow_exception={"type": type(pending_control).__name__, "message": str(pending_control)})
    try:
        receipt["elapsed_seconds"] = time.monotonic() - started
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(f"wrapper_receipt={receipt_path} planning_only={not args.run} training_executed=false strength_eligible=false")
    finally:
        if pending_control is not None:
            # A cleanup/publication error must not replace the original control
            # flow. The unknown tree guard and producer result stay in receipt.
            raise pending_control
    if args.run and receipt["tree_gone"] is not True:
        return 125
    return receipt["exit_code"] or 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"pals_followup_wrapper_error={error}; no automatic retry; existing artifacts/ownership retained", file=sys.stderr)
        raise SystemExit(125)
