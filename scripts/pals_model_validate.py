#!/usr/bin/env python3
"""Bounded CPU-only PALS initialization/export validation; no training.

Run under run_managed.py so the dependency environment and pip scratch are
owned, bounded, and removed after all children exit. Preserve only the explicit
output directory outside the checkout. GPU validation is a different stage.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
import re
import stat
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import MappingProxyType

sys.dont_write_bytecode = True
from managed_storage import checked, StorageError
from managed_outputs import OutputLease, canonical_catalog, publish_runtime


_CPU_STAGE_DEPENDENCIES = MappingProxyType({
    "venv_seconds": (),
    "torch_install_seconds": ("venv_seconds",),
    "reference_install_seconds": ("venv_seconds",),
    "model_tests_seconds": ("torch_install_seconds", "reference_install_seconds"),
    "device_packing_export_seconds": ("reference_install_seconds",),
    "device_packing_files_seconds": ("device_packing_export_seconds",),
    "device_packing_rust_body_seconds": ("device_packing_files_seconds",),
    "initialize_seconds": ("torch_install_seconds",),
    "preparation_check_seconds": ("initialize_seconds", "reference_install_seconds"),
    # This producer revalidates its own collection, checkpoint and CPU ELF.
    # It does not consume preparation-check's separate row/mask verdict.
    "private_verifier_seconds": ("initialize_seconds", "reference_install_seconds"),
    "export_seconds": ("initialize_seconds", "reference_install_seconds"),
    "numeric_seconds": ("export_seconds",),
    # Fixture generation reads the checkpoint/export, not numeric-check's
    # verdict. Preserve a failed comparison while collecting independent raw
    # reference outputs; neither result admits the other as numerically valid.
    "rust_fixtures_seconds": ("export_seconds",),
    "runtime_preservation_seconds": ("reference_install_seconds",),
})


def cpu_stage_dependencies():
    """Read-only dependencies consumed by the actual managed CPU plan."""
    return _CPU_STAGE_DEPENDENCIES


def cpu_source_cohorts(source, package):
    """Bind source integrity to the stages that actually consume those files."""
    shared = [source / "scripts/pals_model_validate.py",
              source / "crates/rz-eval/src/pals_model.rs",
              source / "crates/rz-eval/Cargo.toml"]
    shared.extend(sorted((package / "src/rz_pals_model").glob("*.py")))
    packing = tuple(source / logical for logical in (
        "crates/rz-eval/examples/device_packing_graph_check.rs",
        "crates/rz-eval/src/pals_onnx/device_packing_admission.rs",
        "crates/rz-eval/src/pals_onnx/device_packing_graph.rs",
        "crates/rz-eval/src/pals_onnx.rs"))
    return tuple(shared), MappingProxyType({"device_packing_rust_body_seconds": packing})


class CommandNotStarted(RuntimeError):
    """Popen reported that this executable could not be started."""
    def __init__(self, command, error):
        super().__init__(str(error))
        self.command = list(command)
        self.original_error_type = type(error).__name__
        self.errno = error.errno


def run(command, *, cwd, environment, seconds=300):
    started = time.monotonic()
    try:
        subprocess.run(command, cwd=cwd, env=environment, check=True, timeout=seconds)
    except (FileNotFoundError, PermissionError) as error:
        # Popen reaps its failed exec child before reporting these errors. A
        # missing cwd or arbitrary I/O error is not executable-start evidence.
        if (error.filename != os.fspath(command[0])
                or error.filename == os.fspath(cwd)):
            raise
        raise CommandNotStarted(command, error) from error
    return time.monotonic() - started


def input_identity(path):
    """Cheap boundary observation; full byte hashes remain initial/final gates."""
    value = path.lstat()
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns)


class StageSourceInputFailure(ValueError):
    """A known unavailable or changed source input belongs to its own stage."""


def _stage_source_snapshot(source, paths):
    try:
        selected = tuple(sorted(paths()))
        if not selected:
            raise StageSourceInputFailure("stage source cohort is empty")
        observed = {}
        for path in selected:
            try:
                relative = path.relative_to(source).as_posix()
            except ValueError as error:
                raise StageSourceInputFailure("stage source is outside the checkout") from error
            before = input_identity(path)
            if not stat.S_ISREG(before[2]):
                raise StageSourceInputFailure("stage source must be a regular file")
            raw = path.read_bytes()
            if input_identity(path) != before or len(raw) != before[4]:
                raise StageSourceInputFailure("stage source changed while being read")
            observed[relative] = (before, hashlib.sha256(raw).hexdigest())
        return observed
    except (FileNotFoundError, NotADirectoryError) as error:
        raise StageSourceInputFailure("stage source is unavailable") from error


def run_with_stage_sources(operation, *, receipt, name, source, paths):
    """Seal only this branch's source inputs around its operation, not another NN stage."""
    record = {"status": "registering", "files": {}}
    receipt.setdefault("stage_source_files", {})[name] = record
    try:
        before = _stage_source_snapshot(source, paths)
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:2048])
        raise
    record.update(status="registered", files={path: value[1] for path, value in before.items()})
    primary = None
    try:
        return operation()
    except BaseException as error:
        primary = error
        record["operation_error"] = {"type": type(error).__name__, "error": str(error)[:2048]}
        if isinstance(error, subprocess.CalledProcessError):
            record["operation_error"]["returncode"] = error.returncode
        raise
    finally:
        try:
            if _stage_source_snapshot(source, paths) != before:
                raise StageSourceInputFailure("stage source cohort changed during execution")
        except StageSourceInputFailure as error:
            record.update(status="failed", error_type=type(error).__name__, error=str(error)[:2048])
            if primary is None:
                raise
            # Never replace an original timeout/cancel/unknown failure with a local input error.
            BaseException.add_note(primary, "stage source verification also failed: " + str(error))
        except BaseException as error:
            # Unknown I/O/storage still propagates to the shared-scope controller.
            record.update(status="failed", error_type=type(error).__name__, error=str(error)[:2048])
            raise
        else:
            record["status"] = "verified"


def run_with_rules_profile(operation, *, receipt, path):
    """Register and bind the export-only Rules input to its execution."""
    record = {"status": "registering"}
    receipt["rules_profile_verification"] = record

    try:
        path = checked(path)
        identity = input_identity(path)
        if not stat.S_ISREG(identity[2]) or not 1 <= identity[4] <= 64 * 1024:
            raise StageSourceInputFailure("Rules descriptor must be a bounded regular file")
        raw = path.read_bytes()
        if len(raw) != identity[4] or input_identity(path) != identity:
            raise StageSourceInputFailure("Rules descriptor changed while being registered")
        digest = hashlib.sha256(raw).hexdigest()
        receipt["rules_profile"] = {"sha256": digest, "bytes": len(raw)}
    except (FileNotFoundError, NotADirectoryError) as error:
        failure = StageSourceInputFailure("registered Rules descriptor unavailable")
        record.update(status="failed", error_type=type(failure).__name__, error=str(failure))
        raise failure from error
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:2048])
        raise

    def check():
        try:
            if (input_identity(path) != identity
                    or hashlib.sha256(path.read_bytes()).hexdigest() != digest
                    or input_identity(path) != identity):
                raise StageSourceInputFailure("registered Rules descriptor changed")
        except (FileNotFoundError, NotADirectoryError) as error:
            raise StageSourceInputFailure("registered Rules descriptor unavailable") from error

    try:
        check()
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:2048])
        raise
    record["status"] = "registered"
    primary = None
    try:
        return operation(path)
    except BaseException as error:
        primary = error
        record["operation_error"] = {"type": type(error).__name__, "error": str(error)[:2048]}
        if isinstance(error, subprocess.CalledProcessError):
            record["operation_error"]["returncode"] = error.returncode
        raise
    finally:
        try:
            check()
        except StageSourceInputFailure as error:
            record.update(status="failed", error_type=type(error).__name__, error=str(error)[:2048])
            if primary is None:
                raise
            BaseException.add_note(primary, "Rules descriptor verification also failed: " + str(error))
        except BaseException as error:
            record.update(status="failed", error_type=type(error).__name__, error=str(error)[:2048])
            if primary is not None:
                BaseException.add_note(error, "export also failed: " + str(primary))
            raise
        else:
            record["status"] = "verified"


class ValidationStagesFailed(RuntimeError):
    """Independent results were preserved, but required validation is incomplete."""


class ValidationStages:
    """Record real dependencies inside one shared managed CPU execution scope.

    A returned nonzero test/gate is a failed result, not a prerequisite for an
    unrelated model command. A failed executable start is also local to that
    command. Timeout, storage failure, or changed registered
    inputs make this shared execution scope unsafe for further commands. The
    outer managed supervisor remains the sole process-tree closure authority.
    """
    def __init__(self, receipt, check_inputs, *, clock=time.monotonic):
        self.receipt, self.check_inputs, self.clock = receipt, check_inputs, clock
        self.results = receipt.setdefault("stage_results", {})
        self.shared_failure = None

    def passed(self, name):
        return self.results.get(name, {}).get("status") == "passed"

    def perform(self, name, operation, *, requires=None):
        if requires is None:
            requires = cpu_stage_dependencies().get(name, ())
        if name in self.results or any(dep not in self.results for dep in requires):
            raise ValueError("unique stage and already registered dependencies required")
        result = {"status": "blocked", "depends_on": list(requires), "executed": False}
        self.results[name] = result
        if self.shared_failure is not None:
            result.update(blocked_by=[self.shared_failure], failure_scope="shared_managed_execution")
            return False
        blocked = [dep for dep in requires if not self.passed(dep)]
        if blocked:
            result.update(blocked_by=blocked, failure_scope="prerequisites")
            return False
        started = self.clock()
        try:
            self.check_inputs()
        except (OSError, StorageError, ValueError) as error:
            self.shared_failure = name
            result.update(status="failed", failure_scope="registered_inputs_or_shared_storage",
                          error_type=type(error).__name__, error=str(error)[:2048])
            return False
        result["executed"] = True
        try:
            operation()
        except (subprocess.TimeoutExpired, StorageError, OSError) as error:
            # A timeout never certifies descendant exit. An I/O/ownership
            # failure is not converted to an ordinary numerical mismatch.
            self.shared_failure = name
            result.update(status="failed", failure_scope="shared_managed_execution",
                          error_type=type(error).__name__, error=str(error)[:2048])
        except (subprocess.CalledProcessError, CommandNotStarted, ValueError) as error:
            result.update(status="failed", failure_scope="stage",
                          error_type=type(error).__name__, error=str(error)[:2048])
            if isinstance(error, CommandNotStarted):
                result.update(command_exec_started=False, original_error_type=error.original_error_type,
                              errno=error.errno)
            if isinstance(error, subprocess.CalledProcessError):
                result["returncode"] = error.returncode
                if error.returncode < 0:
                    self.shared_failure = name
                    result.update(failure_scope="shared_managed_execution",
                                  termination_observation="negative_returncode_no_cause_inferred")
        except BaseException as error:
            self.shared_failure = name
            result.update(status="failed", failure_scope="shared_managed_execution",
                          error_type=type(error).__name__, error=str(error)[:2048],
                          seconds=self.clock() - started)
            raise
        else:
            result["status"] = "passed"
        result["seconds"] = self.clock() - started
        if self.passed(name):
            self.receipt["stages"][name] = result["seconds"]
        return self.passed(name)

    def require_complete(self):
        incomplete = [name for name in self.results if not self.passed(name)]
        if incomplete:
            raise ValidationStagesFailed("required validation stages incomplete: " + ", ".join(incomplete))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--runtime-store", type=Path,
                        help="External immutable content-addressed CPU runtime asset store")
    parser.add_argument("--source-commit", help="Coordinator-verified HEAD for a cross-OS worktree")
    parser.add_argument("--source-dirty", action="store_true")
    parser.add_argument("--layout", choices=("separate_pc", "shared_pc_if"), default="separate_pc")
    parser.add_argument("--rules-profile-json", type=Path,
                        help="Registered descriptor emitted by the Rust Rules encoder")
    parser.add_argument("--dataset-run", type=Path,
                        help="Registered own-data collection directory for no-step preparation")
    parser.add_argument("--dataset-receipt-sha256")
    parser.add_argument("--dataset-encoder-source-sha256")
    parser.add_argument("--verifier-cpu-binary", type=Path,
                        help="Registered own Rust training-private CPU task dispatcher")
    parser.add_argument("--verifier-cpu-binary-sha256")
    args = parser.parse_args()
    if not 0 <= args.seed <= 2**63 - 1:
        parser.error("seed must fit nonnegative signed int64")
    if args.source_commit and not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        parser.error("source commit must be a full lowercase SHA-1")
    dataset_args = (args.dataset_run, args.dataset_receipt_sha256,
                    args.dataset_encoder_source_sha256)
    if any(dataset_args) and not all(dataset_args):
        parser.error("dataset run, receipt digest, and encoder source digest must be supplied together")
    for digest in dataset_args[1:]:
        if digest is not None and not re.fullmatch(r"[0-9a-f]{64}", digest):
            parser.error("dataset identity requires lowercase SHA-256")
    verifier_args = (args.verifier_cpu_binary, args.verifier_cpu_binary_sha256)
    if any(verifier_args) and (not all(verifier_args) or not all(dataset_args)):
        parser.error("private verifier requires binary/digest and a registered own dataset")
    if args.verifier_cpu_binary_sha256 and not re.fullmatch(r"[0-9a-f]{64}", args.verifier_cpu_binary_sha256):
        parser.error("private verifier binary requires lowercase SHA-256")
    source = Path(__file__).resolve().parent.parent
    output = checked(args.output)
    runtime_store = checked(args.runtime_store or
                            canonical_catalog().parent.parent / "cache" / "pals-cpu-runtime")
    rules_profile = args.rules_profile_json
    dataset_run = checked(args.dataset_run) if args.dataset_run else None
    verifier_binary = checked(args.verifier_cpu_binary) if args.verifier_cpu_binary else None
    if args.layout == "shared_pc_if" and rules_profile is None:
        parser.error("shared_pc_if requires a registered Rules descriptor")
    if runtime_store and (runtime_store.is_relative_to(source) or source.is_relative_to(runtime_store)):
        parser.error("runtime store must be outside the checkout")
    if output.is_relative_to(source) or source.is_relative_to(output) or output.exists():
        parser.error("output must be a fresh directory outside the checkout")
    scratch = os.environ.get("TMPDIR")
    if not scratch or os.environ.get("CARGO_TARGET_DIR") is None:
        parser.error("run through scripts/run_managed.py with managed scratch")
    output_lease = OutputLease(output, source, outer_supervised=True)
    receipt = {"schema": "rovezero.pals-cpu-model-validation.v1", "status": "running",
               "training_executed": False, "training_steps": 0,
               "provider": "cpu", "gpu_validation": "not_run", "stages": {}}
    receipt["seed"] = args.seed
    receipt["layout"] = args.layout
    if dataset_run:
        receipt["preparation_dataset"] = {
            "receipt_sha256": args.dataset_receipt_sha256,
            "encoder_source_sha256": args.dataset_encoder_source_sha256,
            "optimizer_updates": 0,
        }
    if args.source_commit:
        receipt["source_commit"] = args.source_commit
        receipt["source_dirty"] = args.source_dirty
        receipt["source_commit_method"] = "coordinator_registered_cross_os_worktree"
    else:
        receipt["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        receipt["source_dirty"] = subprocess.run(
            ["git", "diff", "--quiet", "--no-ext-diff"], cwd=source, check=False).returncode != 0
        receipt["source_commit_method"] = "local_git"
    package = source / "experiments/model-research/pals"
    source_files, stage_source_paths = cpu_source_cohorts(source, package)
    receipt["source_files"], source_identities = {}, {}
    for path in source_files:
        relative = str(path.relative_to(source)).replace("\\", "/")
        original = input_identity(path)
        receipt["source_files"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        if input_identity(path) != original:
            raise OSError("validation source changed while being registered")
        source_identities[relative] = original
    receipt["versions"] = {"torch": "2.8.0", "numpy": "2.2.6", "onnx": "1.19.0",
                           "onnxruntime": "1.22.0"}
    started = time.monotonic()
    environment = os.environ.copy()
    environment.update({"PYTHONDONTWRITEBYTECODE": "1", "PIP_NO_CACHE_DIR": "1",
                        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    def check_inputs(*, full=False):
        # Do not rescan output trees or rehash every source body per command.
        # These stat observations supplement the original byte-hash gates.
        if any(input_identity(source / relative) != original
               for relative, original in source_identities.items()):
            raise OSError("validation source changed during the registered run")
        if full:
            if any(hashlib.sha256((source / relative).read_bytes()).hexdigest() != digest
                   for relative, digest in receipt["source_files"].items()):
                raise OSError("validation source bytes changed during the registered run")
            check_inputs()

    receipt["stage_failure_policy"] = "dependency_scoped_cpu_validation_v1"
    controller = ValidationStages(receipt, check_inputs)
    try:
        output_lease.acquire()
        # Only the outer supervisor may certify tree closure. A timeout blocks
        # further commands in this same managed scope; returned gate failures
        # keep their own dependency scope and do not suppress independent work.
        with nullcontext(tempfile.mkdtemp(prefix="pals-validation-", dir=scratch)) as temporary:
            venv = Path(temporary) / "venv"
            python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            controller.perform("venv_seconds", lambda: run([
                sys.executable, "-m", "venv", "--copies", str(venv)],
                cwd=source, environment=environment))
            controller.perform("torch_install_seconds", lambda: run([
                str(python), "-m", "pip", "install", "torch==2.8.0",
                "--index-url", "https://download.pytorch.org/whl/cpu"],
                cwd=source, environment=environment, seconds=600))
            controller.perform("reference_install_seconds", lambda: run([
                str(python), "-m", "pip", "install", "numpy==2.2.6", "onnx==1.19.0",
                "onnxruntime==1.22.0"], cwd=source, environment=environment))
            environment["PYTHONPATH"] = str(package / "src")
            controller.perform("model_tests_seconds", lambda: run_with_stage_sources(
                lambda: run([str(python), "-m", "unittest", "discover", "-s", str(package / "tests"), "-v"],
                            cwd=source, environment=environment, seconds=600),
                receipt=receipt, name="model_tests_seconds", source=source,
                paths=lambda: [path for path in (package / "tests").iterdir() if path.suffix == ".py"]))
            # Weight-free static packing is a separate branch. It never creates
            # an ORT Session/provider/Run or authorizes a later GPU execution.
            packing = output / "device-packing"
            packing_files = {name: packing / name for name in
                             ("device-packing.json", "device_public_pack.onnx")}
            packing_pins = {}
            controller.perform("device_packing_export_seconds", lambda: run([
                str(python), "-c",
                "from rz_pals_model.device_packing_artifacts import export_device_packing_artifact\n"
                "import sys\nexport_device_packing_artifact(sys.argv[1])\n",
                str(packing)], cwd=source, environment=environment))

            def inspect_packing():
                for name, asset_path in packing_files.items():
                    cap = 512 * 1024 if name.endswith(".json") else 2 * 1024 * 1024
                    try:
                        size = asset_path.stat().st_size
                    except FileNotFoundError as error:
                        raise ValueError("generated packing artifact is missing") from error
                    if asset_path.is_symlink() or not asset_path.is_file() or not 1 <= size <= cap:
                        raise ValueError("generated packing artifact violates static file bounds")
                    with asset_path.open("rb") as handle:
                        sha = hashlib.file_digest(handle, "sha256").hexdigest()
                    packing_pins[name] = {"bytes": size, "sha256": sha}

            controller.perform("device_packing_files_seconds", inspect_packing)
            controller.perform("device_packing_rust_body_seconds", lambda: run_with_stage_sources(
                lambda: run(["cargo", "run", "--locked", "-p", "rz-eval", "--features", "onnx,contracts",
                             "--example", "device_packing_graph_check", "--quiet", "--",
                             str(packing_files["device-packing.json"]), str(packing_files["device_public_pack.onnx"]),
                             packing_pins["device-packing.json"]["sha256"], packing_pins["device_public_pack.onnx"]["sha256"]],
                            cwd=source, environment=environment, seconds=300),
                receipt=receipt, name="device_packing_rust_body_seconds", source=source,
                paths=lambda: stage_source_paths["device_packing_rust_body_seconds"]))
            if controller.passed("device_packing_rust_body_seconds"):
                receipt["device_packing_static_interop"] = {
                    "assets": packing_pins, "scope": "python_artifact_rust_fixed_body_check_only",
                    "rust_body_exit_status": 0, "native_validation": "not_performed",
                    "provider_run_fence_authority": False, "registration_authenticity_authority": False}
            cli = [str(python), "-m", "rz_pals_model.cli"]
            initialization = output / "initialization"
            export = output / "export-pc"
            controller.perform("initialize_seconds", lambda: run(cli + ["init", "--seed", str(args.seed),
                "--output", str(initialization)], cwd=source, environment=environment))
            checkpoint = initialization / "untrained.pt"
            if dataset_run:
                controller.perform("preparation_check_seconds", lambda: run([
                    str(python), "-m", "rz_pals_model.preparation_check", "--collection", str(dataset_run),
                    "--receipt-sha256", args.dataset_receipt_sha256,
                    "--encoder-source-sha256", args.dataset_encoder_source_sha256,
                    "--checkpoint", str(checkpoint), "--output", str(output / "preparation-check.json"),
                    "--max-records", "256", "--max-wall-time-ms", "120000", "--max-output-bytes", "1048576",
                    "--threads", "2", "--batch-size", "1", "--require-masked-value"],
                    cwd=source, environment=environment))
                if verifier_binary:
                    def private_verifier():
                        if (not verifier_binary.is_file()
                                or not 1 <= verifier_binary.stat().st_size <= 1024**3):
                            raise ValueError("private verifier dispatcher must be a bounded regular executable")
                        with checkpoint.open("rb") as stream:
                            checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
                        run([str(python), "-B", "-m", "rz_pals_model.verifier_producer",
                            "--collection", str(dataset_run), "--receipt-sha256", args.dataset_receipt_sha256,
                            "--encoder-source-sha256", args.dataset_encoder_source_sha256,
                            "--checkpoint", str(checkpoint), "--checkpoint-sha256", checkpoint_sha,
                            "--cpu-binary", str(verifier_binary), "--cpu-binary-sha256", args.verifier_cpu_binary_sha256,
                            "--output", str(output / "private-verifier"), "--allowed-task", "resume_task",
                            "--max-games", "1", "--max-steps", "1", "--max-nodes", "8192", "--max-wall-time-ms", "120000",
                            "--max-output-bytes", "4194304", "--max-forward-flops", "10000000000",
                            "--baseline-depth", "1", "--requested-depth", "2", "--max-nodes-per-check", "4096",
                            "--max-task-wall-time-ms", "5000", "--max-cpu-output-bytes", "65536"],
                            cwd=source, environment=environment, seconds=150)
                    controller.perform("private_verifier_seconds", private_verifier)
            export_command = cli + ["export", "--checkpoint", str(checkpoint), "--output", str(export), "--layout", args.layout]
            def export_model():
                if rules_profile:
                    return run_with_rules_profile(
                        lambda path: run(export_command + ["--rules-profile-json", str(path)],
                                         cwd=source, environment=environment),
                        receipt=receipt, path=rules_profile)
                return run(export_command, cwd=source, environment=environment)
            controller.perform("export_seconds", export_model)
            controller.perform("numeric_seconds", lambda: run(cli + ["numeric-check", "--checkpoint", str(checkpoint),
                "--export", str(export)], cwd=source, environment=environment))
            controller.perform("rust_fixtures_seconds", lambda: run(cli + ["rust-fixtures", "--checkpoint", str(checkpoint),
                "--export", str(export), "--output", str(output / "rust-fixtures.json")],
                cwd=source, environment=environment))

            def preserve_runtime():
                runtime_path = subprocess.check_output([str(python), "-c",
                    "import onnxruntime,pathlib; p=pathlib.Path(onnxruntime.__file__).parent/'capi'; "
                    "print(next(p.glob('libonnxruntime.so.1.22.0')))"],
                    cwd=source, env=environment, text=True).strip()
                output_lease.check_budget()
                runtime_asset, runtime_sha256 = publish_runtime(Path(runtime_path), runtime_store)
                receipt["runtime"] = {"path": str(runtime_asset), "sha256": runtime_sha256,
                    "bytes": runtime_asset.stat().st_size, "storage": "external_shared_content_addressed_store"}

            controller.perform("runtime_preservation_seconds", preserve_runtime)
            check_inputs(full=True)
            controller.require_complete()
            receipt["status"] = "success"
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["failure_type"] = type(error).__name__
        raise
    finally:
        receipt["total_seconds"] = time.monotonic() - started
        storage_error = None
        try:
            if output_lease.record is not None:
                output_lease.finish(status=receipt["status"], tree_gone="outer_managed_supervisor_only")
        except (StorageError, OSError) as error:
            storage_error = error
            receipt.update(status="failed", storage_lifecycle_error={
                "type": type(error).__name__, "message": str(error)})
        if output.is_dir():
            (output / "validation.json").write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if storage_error is not None and sys.exc_info()[0] is None:
            raise storage_error


if __name__ == "__main__":
    main()
