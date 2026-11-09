#!/usr/bin/env python3
"""Own one isolated V3 pre-native identity-mismatch integration run.

Inputs are copied under a fresh managed output lease. No runtime is loaded by
this wrapper. The supplied test executable must be the current, reviewed Linux
rz-uci test binary containing the specified ignored integration test.
"""
import argparse
from contextlib import ExitStack
from enum import Enum
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import resource
import stat
import sys
import time

from managed_outputs import OutputLease
from managed_process import run_group, supervisor_signal_scope
from virtual_system_numeric_prepare import NumericInputRegistration, add_consumer_arguments

TEST = ("rovent_bootstrap::launch::v3::before_native_asset_integration::"
        "registered_v3_factory_pre_native_reclaims_seventeen_in_one_process")
MAX_CONTROL = 1024 * 1024
MAX_ASSET = 2 * 1024 ** 3
RESERVE = 1024 ** 3
OVERHEAD = 32 * 1024 ** 2


class OriginalOwnerSupervisorCapability(Enum):
    """Capability of this research runner, not a GPU or process observation."""

    NOT_CONNECTED = "not_connected"
    CONNECTED = "connected"


def original_owner_supervisor_capability():
    """Report the implemented boundary before allocating any input or output.

    CONNECTED must only be returned by a future implementation that actually
    binds this test to its original owner and can retain and reobserve that
    owner. A subprocess PID, a successful process join, or an environment flag
    cannot establish that capability.
    """
    return OriginalOwnerSupervisorCapability.NOT_CONNECTED


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def unique_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def relative(value):
    require(isinstance(value, str), "artifact path must be text")
    path = PurePosixPath(value)
    require(not path.is_absolute() and path.parts and
            all(part not in ("", ".", "..") for part in path.parts), "unsafe artifact path")
    return Path(*path.parts)


def open_directory(path):
    """Pin each path component without following links, including ancestors."""
    require(path.is_absolute() and all(p != ".." for p in path.parts), "unsafe directory path")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[1:]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def open_source(base_fd, rel):
    require(not rel.is_absolute() and all(p not in ("", ".", "..") for p in rel.parts),
            "unsafe source path")
    parent_fd = os.dup(base_fd)
    try:
        for component in rel.parts[:-1]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        fd = os.open(rel.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(fd)
            raise RuntimeError("source must be a regular file")
        return fd, info
    finally:
        os.close(parent_fd)


def metadata(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def read_control(base, rel):
    fd, before = open_source(base, rel)
    try:
        require(before.st_size <= MAX_CONTROL, "control input exceeds limit")
        with os.fdopen(fd, "rb", closefd=False) as source:
            data = source.read(MAX_CONTROL + 1)
        require(len(data) == before.st_size and metadata(before) == metadata(os.fstat(fd)),
                "control input changed while reading")
        return data
    finally:
        os.close(fd)


def hash_stream(source, deadline, consumer):
    digest = hashlib.sha256()
    while True:
        check_consumer_interruption(consumer)
        require(time.monotonic() < deadline, "original deadline expired while hashing")
        data = source.read(1024 * 1024)
        if not data:
            return digest.hexdigest()
        digest.update(data)


def copy_pin(base, root, item, deadline, progress, consumer):
    rel = relative(item["path"])
    expected_size, expected_sha = item["bytes"], item["sha256"]
    require(type(expected_size) is int and 0 < expected_size <= MAX_ASSET,
            "invalid declared artifact size")
    require(isinstance(expected_sha, str) and len(expected_sha) == 64 and
            all(c in "0123456789abcdef" for c in expected_sha), "invalid declared digest")
    fd, before = open_source(base, rel)
    try:
        require(before.st_size == expected_size, "source artifact size differs")
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        out_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        digest, total = hashlib.sha256(), 0
        try:
            while True:
                check_consumer_interruption(consumer)
                require(time.monotonic() < deadline, "original deadline expired while copying")
                data = os.read(fd, 1024 * 1024)
                if not data:
                    break
                digest.update(data)
                total += len(data)
                require(total <= expected_size, "source grew during copy")
                progress[0] += len(data)
                require(progress[0] <= progress[1], "aggregate copy reservation exceeded")
                view = memoryview(data)
                while view:
                    written = os.write(out_fd, view)
                    require(written > 0, "copy made no progress")
                    view = view[written:]
            os.fsync(out_fd)
        finally:
            os.close(out_fd)
        require(total == expected_size and digest.hexdigest() == expected_sha and
                metadata(before) == metadata(os.fstat(fd)), "artifact bytes or source identity changed")
        with open(target, "rb") as output:
            require(hash_stream(output, deadline, consumer) == expected_sha,
                    "copied artifact digest differs")
        return {"path": str(rel), "bytes": total, "sha256": expected_sha}
    finally:
        os.close(fd)


def check_consumer_interruption(consumer):
    state = getattr(consumer, "interruption", None)
    if state is not None and state.get("signal") is not None:
        raise KeyboardInterrupt(f"interrupted by signal {state['signal']}")


def protect_output(lease, output, parent_fd):
    # Do not chmod a substituted path or bypass a sealed directory capability.
    require(getattr(lease, "_created_directory", None) is None,
            "managed output capability already seals its mode")
    expected = lease.record.get("identity")
    require(isinstance(expected, list) and len(expected) == 2, "missing owner identity")
    fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        opened, named = os.fstat(fd), os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        require(stat.S_ISDIR(opened.st_mode) and stat.S_ISDIR(named.st_mode) and
                [opened.st_dev, opened.st_ino] == expected == [named.st_dev, named.st_ino] and
                opened.st_uid == os.geteuid(), "registered output identity changed")
        os.fchmod(fd, 0o700)
        named = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        require([named.st_dev, named.st_ino] == expected and
                stat.S_IMODE(os.fstat(fd).st_mode) == 0o700, "output protection failed")
    finally:
        os.close(fd)


def system_runtime_inputs(consumer, manifest):
    """Identify canonical libraries before launching the test ELF.

    The manifest is already protected by the template-root lease. The
    requested paths may be symlinks; the registration uses the independently
    declared canonical paths, whose inode identities also cover aliases.
    """
    if consumer.registry is None:
        return []
    runtime = manifest["system_runtime_lock"]["runtime"]
    libraries = runtime["configuration"]["libraries"]
    require(isinstance(libraries, list) and 0 < len(libraries) <= 128,
            "bounded registered system libraries required")
    paths = []
    for library in libraries:
        require(isinstance(library, dict) and isinstance(library.get("canonical_path"), str),
                "canonical system library path required")
        path = Path(library["canonical_path"])
        require(path.is_absolute() and ".." not in path.parts,
                "absolute canonical system library path required")
        paths.append(path)
    return list(dict.fromkeys(paths))


def run(args, template_fd, asset_fd, repo_fd, parent_fd, consumer):
    paths = [Path(getattr(args, name)) for name in
             ("repo", "output", "template_root", "asset_root", "test_binary")]
    require(all(path.is_absolute() for path in paths), "absolute paths required")
    repo, output, template_root, asset_root, binary = paths
    require(all(".." not in path.parts for path in paths), "parent traversal is not allowed")
    # Pin and inspect the output parent without resolving links. OutputLease
    # supplies the ownership check when the new output is actually created.
    try:
        os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise RuntimeError("output root must be new; links are not accepted")
    for source in (template_root, asset_root):
        require(not output.is_relative_to(source) and not source.is_relative_to(output),
                "output must be disjoint from source roots")
    binary_info = binary.stat()
    require(stat.S_ISREG(binary_info.st_mode) and os.access(binary, os.R_OK | os.X_OK),
            "test executable must be readable and executable")
    template_rel = relative(args.template)
    def read_template():
        # NumericInputRegistration keeps all participating catalog locks and
        # the registry lock across this descriptor read and publication of the
        # dynamically discovered system-library paths.
        lock_bytes = read_control(template_fd, template_rel)
        lock = unique_json(lock_bytes)
        manifest = lock["manifest"]
        require(manifest["system_runtime_lock"]["runtime"]["kind"] == "system_cuda",
                "registered system CUDA lock required")
        return (lock_bytes, lock), system_runtime_inputs(consumer, manifest)

    lock_bytes, lock = consumer.read_and_register(read_template, descriptor=template_root / template_rel)
    manifest = lock["manifest"]
    startup = manifest["startup_document"]
    model = manifest["engine"]["pc"]
    require(model["proposer"]["weights"] == model["critic"]["weights"] and
            model["proposer"]["provenance"] == model["critic"]["provenance"],
            "P/C weight and checkpoint pins must agree")
    direct_paths = ("inputs/export/export.json", "inputs/model/rules-descriptor.json",
                    "inputs/export/public_memory.onnx", "inputs/export/shared_pc_if.onnx")
    provenance = manifest["provenance"]
    artifacts = []
    for path in direct_paths:
        matches = [item for item in provenance if item.get("path") == path]
        require(len(matches) == 1, "unique original direct asset pin required")
        artifacts.append(matches[0])
    artifacts += [model["proposer"]["weights"], model["proposer"]["provenance"]]
    require(len({item["path"] for item in artifacts}) == len(artifacts), "asset paths must be distinct")
    sizes = [item.get("bytes") for item in [*artifacts, startup]]
    require(all(type(size) is int and 0 < size <= MAX_ASSET for size in sizes),
            "invalid declared copy size")
    copy_total = sum(sizes)
    # The hardlink does not allocate another file's data, but the lifecycle
    # accounting may charge its full logical size. Include both capped logs,
    # generated controls and a bounded metadata allowance in OVERHEAD.
    require(copy_total + len(lock_bytes) + binary_info.st_size + OVERHEAD <= RESERVE,
            "declared assets and test overhead exceed this run's reservation")
    progress = [0, copy_total]
    source_rel = Path("crates/rz-uci/src/rovent_bootstrap/launch/v3.rs")
    source_bytes = read_control(repo_fd, source_rel)
    deadline = time.monotonic() + args.seconds
    check_consumer_interruption(consumer)
    lease = OutputLease(output, repo, reserve=RESERVE, retention_days=1,
                        dependency_registry=consumer.registry)
    status, gone = "failed_before_test", True
    receipt = {"schema": "rovent-before-native-isolated-owner/1", "status": status,
               "test": TEST, "provider_ready": "not_observed", "gpu_inference": "not_observed",
               "physical_completion": "not_inferred_from_process_exit",
               "consumer_input_coverage": ("registered_only" if consumer.registry is not None
                                           else "legacy_unregistered"),
               "registered_test_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
               "source_binary_relationship": "not_proven_by_this_owner",
               "executed_image_matches_hashed_path": "unknown_path_execution_race",
               "process_join": "not_dispatched",
               "declared_copy_bytes": copy_total, "test_binary_bytes": binary_info.st_size}
    # The outer supervisor scope's non-raising handlers remain installed from
    # registration through child drain and terminal output publication.
    lease.acquire()
    try:
        check_consumer_interruption(consumer)
        protect_output(lease, output, parent_fd)
        require(os.stat(output).st_dev == binary_info.st_dev,
                "test image and output must be on the same filesystem")
        binary_parent_fd = open_directory(binary.parent)
        try:
            binary_fd, binary_opened = open_source(binary_parent_fd, Path(binary.name))
            try:
                require(metadata(binary_opened) == metadata(binary_info), "test binary changed")
                with os.fdopen(binary_fd, "rb", closefd=False) as image:
                    receipt["test_binary_sha256"] = hash_stream(image, deadline, consumer)
                require(metadata(binary_opened) == metadata(os.fstat(binary_fd)),
                        "test binary changed while hashing")
            finally:
                os.close(binary_fd)
        finally:
            os.close(binary_parent_fd)
        for item in artifacts:
            receipt.setdefault("assets", []).append(copy_pin(asset_fd, output, item, deadline, progress, consumer))
        receipt["startup"] = copy_pin(template_fd, output, startup, deadline, progress, consumer)
        require(progress[0] == copy_total, "aggregate copy length mismatch")
        lock_target = output / template_rel
        lock_target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with open(lock_target, "xb") as stream:
            stream.write(lock_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        receipt["template_sha256"] = hashlib.sha256(lock_bytes).hexdigest()
        run_args = ["--input-lock=" + str(template_rel),
                    "--native-export=" + direct_paths[0], "--native-rules=" + direct_paths[1],
                    "--native-public-graph=" + direct_paths[2],
                    "--native-shared-pc-graph=" + direct_paths[3],
                    "--source-verification=observe", f"--process-epoch={args.process_epoch}",
                    "--startup-ms=120000", "--root-policy=exact-registered"]
        env = os.environ.copy()
        env.update(ROVENT_PRENATIVE_ROOT=str(output), ROVENT_PRENATIVE_TEMPLATE=str(template_rel),
                   ROVENT_PRENATIVE_ARGS_JSON=json.dumps(run_args))
        require(metadata(binary_info) == metadata(binary.stat()), "test executable changed before run")
        require(deadline > time.monotonic(), "deadline expired before test")
        check_consumer_interruption(consumer)
        # Redirect inherited descriptors for the existing bounded process owner.
        out_fd = os.open(output / "test.stdout", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        err_fd = os.open(output / "test.stderr", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        saved_out, saved_err = os.dup(1), os.dup(2)
        original_fsize = resource.getrlimit(resource.RLIMIT_FSIZE)
        try:
            # Bound each test log even if a faulty child writes faster than
            # periodic output-budget observation. The child inherits this cap.
            soft = original_fsize[0]
            capped = 8 * 1024 * 1024 if soft == resource.RLIM_INFINITY else min(soft, 8 * 1024 * 1024)
            resource.setrlimit(resource.RLIMIT_FSIZE, (capped, original_fsize[1]))
            os.dup2(out_fd, 1)
            os.dup2(err_fd, 2)
            # A stop after this check may race with Popen, so mark dispatch as
            # attempted before invoking the original group supervisor.
            check_consumer_interruption(consumer)
            gone = False  # Unknown until the original group supervisor reports termination.
            receipt["process_join"] = "unknown_supervisor_has_not_returned"
            # The externally supplied executable has no independently proven
            # CPU-only boundary. Once dispatch is attempted, process-group
            # completion cannot authorize release of its input lease.
            consumer.dispatched = True
            code, gone, failure = run_group([str(binary), TEST, "--exact", "--ignored", "--nocapture",
                                              "--test-threads=1"], str(output), env,
                                             max(0.001, deadline - time.monotonic()), lease.check_budget)
            # This is the same group-termination observation used by the
            # original output owner. It is not a GPU physical fence.
            consumer.original_group_gone = gone is True
            receipt["process_join"] = ("supervisor_reported_group_gone_and_leader_reaped"
                                       if gone else "unknown_group_not_proven_gone")
        finally:
            resource.setrlimit(resource.RLIMIT_FSIZE, original_fsize)
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            for fd in (saved_out, saved_err, out_fd, err_fd):
                os.close(fd)
        receipt.update(exit_code=code, tree_gone=gone, supervision_error=failure)
        if code == 0 and gone and failure is None:
            log = (output / "test.stdout").read_text(encoding="utf-8", errors="replace")
            # These libtest observations are not a permanent correctness gate:
            # a changed format is unknown, and the original log remains intact.
            summary = re.search(r"test result: ok\.\s+(\d+) passed;\s+(\d+) failed;", log)
            attempted = [int(value) for value in re.findall(r"\battempt (\d+):", log)]
            receipt["attempt_numbers_observed"] = attempted
            if summary and summary.groups() == ("0", "0"):
                status = "zero_tests_matched"
            elif summary and summary.groups() == ("1", "0") and TEST in log:
                status = "one_test_passed_source_identity_separate"
            else:
                status = "test_result_format_unconfirmed"
        else:
            status = "test_failed_or_unconfirmed"
    except BaseException as error:
        receipt["error_type"] = type(error).__name__
        receipt["error"] = str(error)
        raise
    finally:
        # The outer supervisor signal handler is non-raising, including for
        # repeated signals during the original terminal publication.
        receipt["status"] = status
        try:
            with open(output / "owner-receipt.json", "x", encoding="utf-8") as stream:
                json.dump(receipt, stream, ensure_ascii=False, sort_keys=True, indent=2)
                stream.write("\n")
        finally:
            lease.finish(status=status, tree_gone=gone)
    print(json.dumps({"status": status, "receipt": str(output / "owner-receipt.json")}))
    return 0 if status == "one_test_passed_source_identity_separate" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("repo", "output", "template-root", "template", "asset-root", "test-binary"):
        parser.add_argument("--" + arg, required=True)
    parser.add_argument("--process-epoch", type=int, required=True)
    parser.add_argument("--seconds", type=int, default=1860)
    add_consumer_arguments(parser)
    args = parser.parse_args()
    require(sys.platform == "linux", "Linux required")
    require(60 <= args.seconds <= 3600 and args.process_epoch > 0, "finite window/epoch required")
    # The signal scope starts before a consumer lease is issued, and remains
    # installed through its terminal observation. A recorded pre-dispatch stop
    # is checked before any test child can be launched.
    with supervisor_signal_scope() as interruption:
        return run_registered(args, interruption)


def run_registered(args, interruption):
    capability = original_owner_supervisor_capability()
    if capability is OriginalOwnerSupervisorCapability.NOT_CONNECTED:
        print(json.dumps({
            "schema": "rovent-before-native-capability/1",
            "status": "unsupported",
            "reason": "original_owner_supervisor_not_connected",
            "input_registration": "not_started",
            "test_dispatch": "not_started",
            "physical_completion": "not_observed",
        }, sort_keys=True))
        return 2
    if capability is not OriginalOwnerSupervisorCapability.CONNECTED:
        raise RuntimeError("unrecognized original-owner supervisor capability")
    # Register all original input roots and the binary before opening any of
    # them. The output is a future input for the original test child and must
    # remain protected as well if child completion is uncertain.
    inputs = [Path(args.template_root), Path(args.asset_root),
              Path(args.repo), Path(args.test_binary)]
    consumer = NumericInputRegistration(args, list(dict.fromkeys(inputs)))
    consumer.interruption = interruption
    try:
        if consumer.registry is not None:
            consumer.add_future([Path(args.output)])
        check_consumer_interruption(consumer)
        with ExitStack() as stack:
            pinned = []
            for value in (args.template_root, args.asset_root, args.repo, str(Path(args.output).parent)):
                fd = open_directory(Path(value))
                stack.callback(os.close, fd)
                pinned.append(fd)
            result = run(args, *pinned, consumer)
    except BaseException as error:
        try:
            finish_consumer_observation(consumer)
        except BaseException as close_error:
            if hasattr(error, "add_note"):
                error.add_note("consumer input close also failed: " + str(close_error))
        raise
    finish_consumer_observation(consumer)
    return result


def finish_consumer_observation(consumer):
    try:
        if not consumer.dispatched:
            consumer.close_if_not_dispatched()
            return
        if getattr(consumer, "original_group_gone", False) is not True:
            return
        # Process-tree completion alone is not a GPU fence.
        for lease in reversed(consumer.leases):
            try:
                lease.retain_unknown(tree_gone=True)
            except Exception as error:
                try:
                    print("consumer_input_observation_warning=" + type(error).__name__, file=sys.stderr)
                except Exception:
                    pass
    finally:
        try:
            consumer.finish_broker()
        except Exception as error:
            try:
                print("consumer_broker_close_warning=" + type(error).__name__, file=sys.stderr)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
