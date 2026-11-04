#!/usr/bin/env python3
"""Prepare a V6 system-CUDA numerical plan from the actual CPU-built ELF.

Only static metadata getters run; this command never loads ORT, CUDA, or a
model. The registered source ELF is copied once to a sealed in-memory image
with matching SHA, and both getters execute that same image. Its inode is not
the source ELF inode. All output belongs to one fresh managed lease.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

from managed_outputs import OutputLease
from managed_process import ProcessNotSpawned, run_group
from managed_storage import checked

sys.dont_write_bytecode = True


def _regular_fd(path, limit):
    path = checked(path)
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags)
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or (limit is not None and before.st_size > limit):
        os.close(fd)
        raise ValueError("bounded regular source file required")
    return path, fd, before


def _digest(fd, before, deadline):
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    read = 0
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("input verification window expired")
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        read += len(chunk)
        if read > before.st_size:
            raise ValueError("input grew while reading")
        digest.update(chunk)
    after = os.fstat(fd)
    if (read != before.st_size or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ValueError("input changed while reading")
    return digest.hexdigest()


def _sealed_binary(fd, before, expected_sha, deadline):
    """Capture one immutable in-memory image, without claiming inode identity."""
    import fcntl
    image = os.memfd_create("rovezero-virtual-numeric-metadata", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        total = 0
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("binary capture window expired")
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > before.st_size:
                raise ValueError("binary grew while capturing")
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(image, view)
                if written <= 0:
                    raise OSError("binary image write did not progress")
                view = view[written:]
        after = os.fstat(fd)
        if (total != before.st_size or digest.hexdigest() != expected_sha
                or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
            raise ValueError("binary changed or differs from registered SHA")
        os.fchmod(image, 0o500)
        seals = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
        fcntl.fcntl(image, fcntl.F_ADD_SEALS, seals)
        if fcntl.fcntl(image, fcntl.F_GET_SEALS) & seals != seals:
            raise OSError("binary image seal not confirmed")
        return image
    except BaseException:
        os.close(image)
        raise


def _read_json(path, limit, deadline, decode):
    path, fd, before = _regular_fd(path, limit)
    try:
        digest = _digest(fd, before, deadline)
        os.lseek(fd, 0, os.SEEK_SET)
        raw = b""
        while len(raw) < before.st_size:
            block = os.read(fd, min(65536, before.st_size - len(raw)))
            if not block:
                break
            raw += block
        if len(raw) != before.st_size or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("input JSON changed before decode")
        return decode(raw), digest
    finally:
        os.close(fd)


def _pinned_metadata(image_fd, output, switch, source, deadline, check_budget, custody):
    # The child receives the original sealed image FD; no pathname can select
    # a replacement image. Its inode is not the original source ELF inode.
    binary = f"/proc/self/fd/{image_fd}"
    out = output / (switch[2:] + ".json")
    out_fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("getter window expired before dispatch")
        custody["dispatch_attempted"] = True
        custody["current_getter"] = switch
        # Popen and its spawn callback are not an atomic operation. If the
        # supervisor raises anywhere in that gap, a child may already exist.
        # Only its returned original process-tree observation resolves this.
        custody["tree_gone"] = "unknown"
        def spawned():
            custody["child_spawned"] = True
        try:
            result = run_group([binary, switch], str(source), os.environ.copy(), remaining, check_budget,
                               pass_fds=(image_fd,), stdout=out_fd, on_spawn=spawned,
                               verify_before_spawn=True)
        except ProcessNotSpawned as error:
            # This typed error is issued exclusively before Popen is called.
            # Re-raise the original exception unchanged, including __cause__.
            custody["tree_gone"] = True
            cause = error.__cause__
            custody["last_result"] = {
                "getter": switch, "dispatch": "preflight_rejected_before_popen",
                "cause_type": type(cause).__name__ if cause is not None else None,
                "cause": str(cause)[:2048] if cause is not None else None,
            }
            raise
        custody["tree_gone"] = result[1]
        custody["last_result"] = {"getter": switch, "exit_code": result[0],
                                  "tree_gone": result[1], "supervisor_error": result[2]}
        return result, out
    finally:
        os.close(out_fd)


def source_head(source, git_executable, path_format):
    """Read HEAD from the explicitly selected Git without changing its config."""
    git_source = str(source)
    if path_format == "windows":
        git_source = subprocess.run(["wslpath", "-w", str(source)], check=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=10).stdout.decode("utf-8").strip()
        if not git_source:
            raise ValueError("Windows Git source path conversion returned empty output")
    elif path_format != "native":
        raise ValueError("explicit native/windows Git path format required")
    return subprocess.run([git_executable, "-C", git_source, "rev-parse", "HEAD"],
                          check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=10).stdout.decode("ascii").strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--source-head", required=True)
    p.add_argument("--git-executable", default="git",
                   help="Git executable for this checkout; defaults to native Linux git")
    p.add_argument("--git-path-format", choices=("native", "windows"), default="native",
                   help="explicit checkout path format expected by the selected Git")
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--binary-sha256", required=True)
    p.add_argument("--binary-bytes", type=int, required=True)
    p.add_argument("--export-manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--system-config", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--model-epoch", type=int, required=True)
    p.add_argument("--seconds", type=int, default=600)
    args = p.parse_args()
    if (sys.platform != "linux" or not 1 <= args.seconds <= 3600
            or not 1 <= args.binary_bytes < 2 ** 63
            or not 1 <= args.model_epoch < 2 ** 64):
        raise ValueError("finite Linux preparation required")
    if any(not path.is_absolute() for path in (args.source_root, args.binary, args.export_manifest,
                                               args.checkpoint, args.system_config, args.output_root)):
        raise ValueError("absolute preparation paths required")
    started = time.monotonic()
    deadline = started + args.seconds
    source, output = checked(args.source_root), checked(args.output_root)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("fresh output leaf and existing parent required")
    if len(args.source_head) != 40 or any(c not in "0123456789abcdef" for c in args.source_head):
        raise ValueError("full source commit required")
    head = source_head(source, args.git_executable, args.git_path_format)
    if head != args.source_head:
        raise ValueError("source commit differs")
    sys.path.insert(0, str(source / "experiments/model-research/pals/src"))
    from rz_pals_model import numeric_capture as nc
    if checked(Path(nc.__file__)).parents[5] != source:
        raise ValueError("imported numerical source belongs to another checkout")
    if not nc._sha(args.binary_sha256):
        raise ValueError("registered binary SHA required")
    binary_path, binary_fd, binary_stat = _regular_fd(args.binary, args.binary_bytes)
    image_fd = None
    lease = None
    created_output = False
    custody = {"tree_gone": True, "dispatch_attempted": False, "child_spawned": False,
               "current_getter": None, "last_result": None}
    status = "failed_before_child_dispatch"
    pending = None
    result_summary = None
    try:
        if binary_stat.st_size != args.binary_bytes:
            raise ValueError("actual ELF size differs from registered build receipt")
        image_fd = _sealed_binary(binary_fd, binary_stat, args.binary_sha256, deadline)
        # This 64KiB bound is the V6 native capture config wire limit, not an
        # independently imposed limit on the ELF or the system libraries.
        config, config_sha = _read_json(args.system_config, 64 * 1024, deadline, nc._strict_json)
        if type(config) is not dict or set(config) != {"max_libraries", "max_file_bytes", "max_total_bytes", "max_capture_ms", "libraries"}:
            raise ValueError("exact system configuration required")
        if (type(config["max_libraries"]) is not int or not 1 <= config["max_libraries"] < 2 ** 64
                or any(type(config[k]) is not int or not 1 <= config[k] < 2 ** 64
                       for k in ("max_file_bytes", "max_total_bytes"))
                or type(config["max_capture_ms"]) is not int or not 1 <= config["max_capture_ms"] < 2 ** 64
                or type(config["libraries"]) is not list or not 1 <= len(config["libraries"]) <= min(64, config["max_libraries"])):
            raise ValueError("bounded system configuration values required")
        rows, total, identities = [], 0, set()
        for entry in config["libraries"]:
            if type(entry) is not dict or set(entry) != {"role", "path", "required_mapping"}:
                raise ValueError("exact system library declaration required")
            if (type(entry["path"]) is not str or not Path(entry["path"]).is_absolute()
                    or type(entry["role"]) is not str or not nc._id(entry["role"])
                    or type(entry["required_mapping"]) is not bool):
                raise ValueError("absolute system path, valid role and mapping flag required")
            path, fd, before = _regular_fd(Path(entry["path"]), config["max_file_bytes"])
            try:
                identity = (before.st_dev, before.st_ino)
                if identity in identities:
                    raise ValueError("system library roles must name distinct files")
                identities.add(identity)
                digest = _digest(fd, before, deadline)
            finally:
                os.close(fd)
            total += before.st_size
            rows.append({"role": entry["role"], "sha256": digest, "bytes": before.st_size,
                         "required_mapping": entry["required_mapping"]})
        if len(rows) > config["max_libraries"] or total > config["max_total_bytes"]:
            raise ValueError("actual system libraries exceed configured limits")
        rows.sort(key=lambda row: row["role"])
        manifest, manifest_sha = _read_json(args.export_manifest, 256 * 1024, deadline, nc._strict_json)
        _, checkpoint_fd, checkpoint_stat = _regular_fd(args.checkpoint, None)
        try:
            checkpoint_sha = _digest(checkpoint_fd, checkpoint_stat, deadline)
        finally:
            os.close(checkpoint_fd)
        if checkpoint_sha != manifest.get("checkpoint_sha256"):
            raise ValueError("actual checkpoint differs from export manifest")
        graph_pins = {}
        for graph in manifest["graphs"]:
            name = graph["file"]
            if type(name) is not str or Path(name).name != name:
                raise ValueError("export graph must be a direct filename")
            _, fd, before = _regular_fd(args.export_manifest.parent / name, None)
            try:
                digest = _digest(fd, before, deadline)
            finally:
                os.close(fd)
            if digest != graph["sha256"] or graph["role"] in graph_pins:
                raise ValueError("actual export graph or role differs")
            graph_pins[graph["role"]] = digest
        actual_sources = nc.source_pins()
        lease = OutputLease(output, source, reserve=16 * 1024 ** 2, deferred=True).acquire()
        lease.create_deferred_directory()
        created_output = True
        values = []
        for switch in ("--source-pins", "--model-registry"):
            (code, gone, failure), getter = _pinned_metadata(image_fd, output, switch, source, deadline, lease.check_budget, custody)
            if code != 0 or gone is not True or failure is not None:
                raise RuntimeError(f"metadata getter failed: {switch}; exit={code}; tree_gone={gone}; error={failure}")
            value, _ = _read_json(getter, 64 * 1024, deadline, nc._strict_json)
            values.append(value)
        compiled, registry = values
        if compiled != actual_sources:
            raise ValueError("same actual ELF and Python source components differ")
        if nc.source_pins() != actual_sources:
            raise ValueError("source changed after metadata getters")
        core = next((row for row in rows if row["role"] == "ort-core"), None)
        if core is None:
            raise ValueError("system ORT core missing")
        bindings = {
            "model_profile": registry["model_profile"], "model_schema": registry["model_schema"],
            "model_semantics": registry["model_semantics"], "encoding_schema": registry["encoding_schema"],
            "model_semantic_sha256": registry["model_semantic_sha256"], "encoding_sha256": registry["encoding_sha256"],
            "model_epoch": args.model_epoch, "weights_sha256": checkpoint_sha, "export_manifest_sha256": manifest_sha,
            "graphs": graph_pins, "runtime_library_sha256": core["sha256"],
            "system_cuda_config_sha256": config_sha, "system_cuda_libraries": rows,
            "backend": "ort_cuda_system", "precision": "fp32", "tf32": False,
            "source_components": actual_sources,
        }
        nc.validate_system_bindings(bindings)
        bindings_item = nc._publish(bindings, output / "bindings.json", 0)
        plan_item = nc.write_system_case_plan(bindings, output / "plan.json")
        status = "prepared_no_model_or_gpu_execution"
        receipt = {"schema": "rovezero.virtual-system-numeric-preparation.v1", "status": status,
                   "source_head": head, "binary_path": str(binary_path), "binary_sha256": args.binary_sha256,
                   "getter_identity": "sealed_memfd_copy_sha_matches_open_source_elf_not_same_inode", "getter_tree_gone": gone,
                   "source_head_binding_scope": "checkout_head_and_compiled_core5_only",
                   "same_full_head_build_evidence": "external_build_receipt_required",
                   "system_config_sha256": config_sha, "bindings": bindings_item, "plan": plan_item,
                   "model_execution": False, "gpu_execution": False, "runtime_mapping": "not_observed",
                   "output_lease_completion": "requires_separate_original_owner_record"}
        item = nc._publish(receipt, output / "preparation-receipt.json", 0)
        result_summary = {"status": status, "receipt": item, "bindings": bindings_item, "plan": plan_item}
    except BaseException as error:
        if custody["child_spawned"]:
            status = "failed_after_metadata_dispatch"
        elif custody["tree_gone"] is not True:
            status = "failed_metadata_dispatch_ownership_unknown"
        if created_output:
            failure = {"schema": "rovezero.virtual-system-numeric-preparation-failure.v1",
                       "status": status, "error_type": type(error).__name__, "error": str(error)[:2048],
                       "custody": custody, "physical_completion": "not_applicable_no_gpu_started"}
            try:
                nc._publish(failure, output / "preparation-failure.json", 0)
            except BaseException as publication_error:
                print(f"preparation failure receipt also failed: {type(publication_error).__name__}: {publication_error}", file=sys.stderr)
        pending = error
        raise
    finally:
        if image_fd is not None:
            os.close(image_fd)
        os.close(binary_fd)
        if lease is not None:
            try:
                lease.finish(status=status, tree_gone=custody["tree_gone"])
            except BaseException as cleanup:
                if pending is None:
                    raise
                print(f"preparation lease finish also failed: {type(cleanup).__name__}: {cleanup}", file=sys.stderr)
    print(json.dumps(result_summary, sort_keys=True))


if __name__ == "__main__":
    main()
