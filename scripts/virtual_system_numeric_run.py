#!/usr/bin/env python3
"""Run one V3 system-CUDA numeric capture under its original output owner.

This wrapper starts no training and does not retry. Process exit is not a GPU
fence. The caller supplies a previously registered binary, config and cgroup.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

sys.dont_write_bytecode = True
from managed_outputs import OutputLease
from managed_process import ProcessNotSpawned, run_group
from managed_storage import checked
from pals_followup_run import outside_git, sha, unique_object, observe_resources
from virtual_custody_transport import CustodyTransport


def require_capture_window(config, supervisor_deadline_ns, *, now_ns=None):
    """Check sealed child windows against the original supervisor deadline.

    Child-side comparison against the inherited absolute deadline is also
    required: process startup occurs after this admission check.
    """
    work = config.get("deadline_seconds")
    cleanup = config.get("cleanup_seconds")
    if type(work) is not int or not 1 <= work <= 7200:
        raise ValueError("sealed capture work deadline is invalid")
    if type(cleanup) is not int or not 1 <= cleanup <= 30:
        raise ValueError("sealed capture cleanup deadline is invalid")
    now = time.monotonic_ns() if now_ns is None else now_ns
    remaining = supervisor_deadline_ns - now
    required = (work + cleanup) * 1_000_000_000
    if remaining < required:
        raise TimeoutError("supervisor time remaining cannot contain sealed child work and cleanup")
    return {"work_seconds": work, "cleanup_seconds": cleanup,
            "remaining_ns_at_check": remaining, "required_ns": required,
            "child_startup_recheck_required": True}


def capture_process_status(code, gone, failure, observation):
    """Keep exit evidence even when its observation does not establish timing."""
    if code != 0 or gone is not True or failure is not None:
        return "capture_process_failed_or_incomplete"
    if observation.get("exit_observation_timing") != "before_deadline":
        return "capture_process_completed_deadline_unknown"
    return "capture_process_completed"


def snapshot_file(path: Path, expected: str, maximum: int, deadline: float, *, content=False, exact_size=None, seal=True):
    """Hold the source and pass a sealed anonymous copy of the exact bytes."""
    if not path.is_absolute():
        raise ValueError("absolute input path required")
    path = checked(path)
    expected = sha(expected)
    info = path.lstat()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags)
    image_fd = None
    try:
        opened = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or not stat.S_ISREG(opened.st_mode)
                or (info.st_dev, info.st_ino) != (opened.st_dev, opened.st_ino)
                or not 0 < opened.st_size <= maximum
                or (exact_size is not None and opened.st_size != exact_size)):
            raise ValueError("original bounded regular input required")
        digest = hashlib.sha256()
        count = 0
        captured = bytearray() if content else None
        if seal:
            image_fd = os.memfd_create("rovezero-pinned-input", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        while chunk := os.read(fd, 1024 * 1024):
            if time.monotonic() >= deadline:
                raise TimeoutError("preparation window elapsed during pinned read")
            count += len(chunk)
            if count > opened.st_size:
                raise ValueError("input grew during identity observation")
            digest.update(chunk)
            if captured is not None:
                captured.extend(chunk)
            if image_fd is not None:
                view = memoryview(chunk)
                while view:
                    written = os.write(image_fd, view)
                    if written <= 0:
                        raise OSError("incomplete sealed image write")
                    view = view[written:]
        after = os.fstat(fd)
        if (count != opened.st_size or digest.hexdigest() != expected
                or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)):
            raise ValueError("original input changed or differs from its registered SHA")
        if image_fd is not None:
            os.fchmod(image_fd, 0o400 if content else 0o500)
            seals = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
            fcntl.fcntl(image_fd, fcntl.F_ADD_SEALS, seals)
            if fcntl.fcntl(image_fd, fcntl.F_GET_SEALS) & seals != seals:
                raise RuntimeError("input snapshot sealing failed")
            os.lseek(image_fd, 0, os.SEEK_SET)
        return fd, image_fd, bytes(captured) if captured is not None else None
    except BaseException:
        os.close(fd)
        if image_fd is not None:
            os.close(image_fd)
        raise


def metadata_getter(image_fd, label, cwd, env, deadline, receipt):
    """Read metadata from the same sealed executable; output is bounded post-exit."""
    output = os.memfd_create("rovezero-metadata", os.MFD_CLOEXEC)
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("preparation window elapsed before metadata getter")
        observation = {"dispatch_attempted": True, "tree_gone": "unknown", "spawned": False, "result_observed": False}
        receipt.setdefault("metadata_getters", {})[label] = observation
        def spawned():
            observation.update(tree_gone="unknown", spawned=True)
        def budget():
            if os.fstat(output).st_size > 64 * 1024:
                raise ValueError("metadata getter output exceeds bound")
        try:
            code, gone, failure = run_group([f"/proc/self/fd/{image_fd}", label], cwd, env,
                                            remaining, budget, pass_fds=(image_fd,), stdout=output,
                                            on_spawn=spawned, verify_before_spawn=True)
        except ProcessNotSpawned as error:
            observation.update(tree_gone=True, not_spawned_cause={
                "type": type(error.__cause__).__name__, "message": str(error.__cause__)})
            raise
        observation.update(exit_code=code, tree_gone=gone, supervisor_error=failure, result_observed=True)
        if code != 0 or gone is not True or failure is not None:
            raise RuntimeError(f"metadata getter failed: {label}, exit={code}, tree_gone={gone}")
        size = os.fstat(output).st_size
        if not 0 < size <= 64 * 1024:
            raise ValueError("metadata getter output exceeds bound")
        os.lseek(output, 0, os.SEEK_SET)
        raw = os.read(output, size + 1)
        if len(raw) != size:
            raise OSError("metadata getter output changed")
        return json.loads(raw, object_pairs_hook=unique_object,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    finally:
        os.close(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--binary-bytes", required=True, type=int)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--timeout", required=True, type=int)
    parser.add_argument("--output-bytes-max", type=int, default=128 * 1024 * 1024)
    args = parser.parse_args(argv)
    if sys.platform != "linux" or not 1 <= args.timeout <= 7200 or not 1 <= args.binary_bytes <= 2**63 - 1:
        parser.error("Linux, finite timeout and declared binary size are required")
    if not 1 <= args.output_bytes_max <= 2**63 - 1:
        parser.error("finite output allowance required")
    start_ns = time.monotonic_ns()
    start = start_ns / 1_000_000_000
    deadline_ns = start_ns + args.timeout * 1_000_000_000
    deadline = deadline_ns / 1_000_000_000
    source = Path(__file__).resolve().parent.parent
    receipt_path = outside_git(args.receipt)
    if not receipt_path.parent.is_dir():
        raise ValueError("receipt requires an existing parent")
    # Reserve the receipt first; an existing user file is never replaced.
    receipt_file = receipt_path.open("x", encoding="utf-8")
    receipt = {
        "schema": "rovezero.virtual-system-numeric-supervisor.v1",
        "status": "failed_before_dispatch", "producer_exit_code": None,
        "producer_result_observed": False, "producer_tree_gone": True,
        "producer_supervisor_error": None, "dispatch_attempted": False,
        "native_binary_sha256": args.binary_sha256, "capture_config_sha256": args.config_sha256,
        "binary_identity_scope": "same_sha_sealed_anonymous_image_not_original_inode",
        "window_scope": "preparation_metadata_and_child; individual blocking OS reads cannot be preempted",
        "gpu_physical_completion": "unknown", "numerical_acceptance": False,
        "kernel_placement": "unknown", "training_executed": False,
        "output_lease": {"acquired": False, "finish_attempted": False, "finish_returned": False},
    }
    lease = None
    pending = None
    first_error = None
    descriptors = []
    try:
        env = os.environ.copy()
        env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1",
                    "NVIDIA_TF32_OVERRIDE": "0"})
        try:
            resource = observe_resources(sorted(os.sched_getaffinity(0)))
            affinity = resource.get("observations", {}).get("cpu_affinity", {})
            affinity.update(requested="not_registered_by_this_wrapper", status="observed_only")
            receipt["resources_before"] = resource
        except Exception as error:
            receipt["resources_before"] = {"status": "unknown", "error_type": type(error).__name__}
        source_fd, binary_fd, _ = snapshot_file(args.binary, args.binary_sha256, args.binary_bytes,
                                                deadline, exact_size=args.binary_bytes)
        descriptors.extend((source_fd, binary_fd))
        config_fd, config_image, config_bytes = snapshot_file(args.config, args.config_sha256, 64 * 1024,
                                                             deadline, content=True)
        descriptors.extend((config_fd, config_image))
        config = json.loads(config_bytes, object_pairs_hook=unique_object,
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        if type(config) is not dict or config.get("provider") != "cuda-system" or not isinstance(config.get("output_root"), str):
            raise ValueError("explicit V3 system-CUDA config/output required")
        receipt["capture_window_preparation"] = require_capture_window(config, deadline_ns)
        target = outside_git(Path(config["output_root"]))
        if receipt_path == target or receipt_path.is_relative_to(target):
            raise ValueError("receipt must be outside the fresh capture output")
        plan_fd, _, plan_bytes = snapshot_file(Path(config["case_plan"]), config["case_plan_sha256"],
                                               4 * 1024 * 1024, deadline, content=True, seal=False)
        descriptors.append(plan_fd)
        plan = json.loads(plan_bytes, object_pairs_hook=unique_object,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        pins = metadata_getter(binary_fd, "--source-pins", args.binary.parent, env, deadline, receipt)
        registry = metadata_getter(binary_fd, "--model-registry", args.binary.parent, env, deadline, receipt)
        binding = plan["bindings"]
        if (binding["source_components"] != pins
                or binding["model_semantic_sha256"] != registry["model_semantic_sha256"]
                or binding["encoding_sha256"] != registry["encoding_sha256"]):
            raise ValueError("same sealed ELF metadata differs from registered V6 plan")
        receipt.update(source_pins=pins, model_registry=registry,
                       sealed_image_bytes=args.binary_bytes,
                       input_identity_scope="sealed_binary_and_config_bytes; plan_rechecked_by_native")
        lease = OutputLease(target, source, reserve=args.output_bytes_max, deferred=True).acquire()
        receipt["output_lease"]["acquired"] = True
        receipt["capture_window_dispatch"] = require_capture_window(config, deadline_ns)
        remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
        receipt["dispatch_attempted"] = True
        receipt["producer_tree_gone"] = "unknown"
        transport = CustodyTransport()
        receipt["custody_transport"] = transport.state
        observation = {}
        receipt["producer_deadline_observation"] = observation
        env["ROVEZERO_CUSTODY_FD"] = str(transport.child_fd())
        env["ROVEZERO_SUPERVISOR_DEADLINE_MONOTONIC_NS"] = str(deadline_ns)
        def producer_spawned():
            receipt["producer_spawned"] = True
            transport.child_started()
        def producer_check():
            transport.poll()
            lease.check_budget()
        try:
            code, gone, failure = run_group([f"/proc/self/fd/{binary_fd}", f"/proc/self/fd/{config_image}"],
                                            args.binary.parent, env, remaining, producer_check,
                                            pass_fds=(binary_fd, config_image, transport.child_fd()), on_spawn=producer_spawned,
                                            verify_before_spawn=True, absolute_deadline=deadline,
                                            deadline_observation=observation)
        except ProcessNotSpawned as error:
            receipt["producer_tree_gone"] = True
            receipt["producer_not_spawned_cause"] = {
                "type": type(error.__cause__).__name__, "message": str(error.__cause__)}
            raise
        finally:
            transport.poll()
            transport.close()
        receipt.update(producer_result_observed=True, producer_exit_code=code,
                       producer_tree_gone=gone, producer_supervisor_error=failure,
                       status=capture_process_status(code, gone, failure, observation))
    except BaseException as error:
        first_error = error
        receipt.update(status="supervisor_error", wrapper_error={"type": type(error).__name__, "message": str(error)})
        if not isinstance(error, Exception):
            pending = error
    finally:
        if lease is not None:
            receipt["output_lease"]["finish_attempted"] = True
            try:
                lease.finish(status=receipt["status"], tree_gone=receipt["producer_tree_gone"])
                receipt["output_lease"]["finish_returned"] = True
            except BaseException as error:
                receipt["output_lease"]["error"] = {"type": type(error).__name__, "message": str(error)}
                if first_error is None:
                    first_error = error
                if pending is None and not isinstance(error, Exception):
                    pending = error
        for fd in reversed(descriptors):
            try:
                os.close(fd)
            except OSError as error:
                receipt.setdefault("descriptor_close_errors", []).append(type(error).__name__)
        receipt["elapsed_seconds"] = time.monotonic() - start
        receipt_error = None
        try:
            json.dump(receipt, receipt_file, sort_keys=True, ensure_ascii=False, allow_nan=False)
            receipt_file.write("\n")
            receipt_file.flush()
            os.fsync(receipt_file.fileno())
        except BaseException as error:
            receipt_error = error
        finally:
            try:
                receipt_file.close()
            except BaseException as error:
                if receipt_error is None:
                    receipt_error = error
        if receipt_error is not None:
            # A producer can fail via its returned tuple without raising a
            # Python exception. Preserve that original outcome even when the
            # dedicated receipt itself could not be durably written.
            try:
                print("virtual_system_receipt_write_failure=" + json.dumps({
                    "producer_exit_code": receipt["producer_exit_code"],
                    "producer_tree_gone": receipt["producer_tree_gone"],
                    "producer_supervisor_error": receipt["producer_supervisor_error"],
                    "receipt_error_type": type(receipt_error).__name__,
                    "receipt_error": str(receipt_error),
                }, ensure_ascii=False, sort_keys=True), file=sys.stderr)
            except Exception:
                pass
            if first_error is not None:
                raise first_error from receipt_error
            raise receipt_error
    if pending is not None:
        raise pending
    print(f"virtual_system_supervisor_receipt={receipt_path}")
    return 0 if receipt["status"] == "capture_process_completed" and receipt["output_lease"]["finish_returned"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
