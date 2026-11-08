#!/usr/bin/env python3
"""Own one explicit V3 system-CUDA UCI smoke child and its parent stream.

This is a functional smoke driver, not numerical acceptance, a game, or a
device-wide GPU completion proof. An unresolved original child remains owned
after the finite run/cleanup window; the quarantine is reported separately.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import errno
import json
import os
from pathlib import Path
import re
import selectors
import signal
import socket
import struct
import subprocess
import sys
import time
import uuid

from lc0_parent_preflight import (
    ACK_SCHEMA, ACK_SCOPE, REQUEST_SCHEMA, PreflightError, _decode, _open_pinned_binary, _unique_object,
)
from managed_process import _group_has_live_members

NOTICE_SCHEMA = "rz-rovent-direct-custody-startup/1"
QUARANTINE_SCHEMA = "rz-rovent-direct-custody-quarantined/1"
REPORT_SCHEMA = "rz-rovent-v3-parent-smoke/1"
LINE_LIMIT = 64 * 1024
LOG_LIMIT = 16 * 1024 * 1024


def _notice_route(notice, pid, epoch, lock_sha256, ordinal):
    route = notice.get("control_path")
    if (notice.get("schema") != NOTICE_SCHEMA
            or type(notice.get("process_id")) is not int or notice["process_id"] != pid
            or type(notice.get("process_epoch")) is not int or notice["process_epoch"] != epoch
            or notice.get("lock_sha256") != lock_sha256
            or type(notice.get("cuda_ordinal")) is not int or notice["cuda_ordinal"] != ordinal
            or notice.get("physical_completion") != "unknown_until_original_owner_confirms"
            or not isinstance(route, str)
            or re.fullmatch(rf"/proc/{pid}/fd/[0-9]+/custody\.sock", route) is None):
        raise PreflightError("v3_notice_identity_mismatch")
    return route


def _once(arguments, prefix):
    values = [item[len(prefix):] for item in arguments if item.startswith(prefix)]
    if len(values) != 1 or not values[0]:
        raise ValueError("required V3 argument missing or repeated")
    return values[0]


def _report(data):
    fd = None
    blocking = None
    try:
        payload = (json.dumps(data, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
        fd = sys.stdout.fileno()
        blocking = os.get_blocking(fd)
        os.set_blocking(fd, False)
        os.write(fd, payload)
    except (BrokenPipeError, OSError):
        # A disconnected observer cannot release a possibly active owner.
        pass
    finally:
        if fd is not None and blocking is not None:
            try:
                os.set_blocking(fd, blocking)
            except OSError:
                pass


def _exit_observed(pid):
    """Inspect the exact direct child without reaping its PID reservation."""
    status = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    return status is not None and status.si_pid == pid


def _pre_native_evidence(value, pid, epoch, lock_sha256):
    """Decode only the original V3 producer's typed before-native records.

    The admission record has no input-lock digest; the caller retains that
    limitation rather than treating it as a factory or device-fence proof.
    """
    if (not isinstance(value, dict)
            or value.get("schema") != "rz-rovent-native-system-resource-run-failure/3"
            or value.get("resource_state_poisoned") is not False
            or value.get("control_panic_payload_retained") is not False
            or value.get("automatic_cleanup") is not False
            or value.get("training_executed") is not False
            or value.get("product_finish") is not None):
        return None
    admission = value.get("pre_native_admission")
    factory = value.get("factory_pre_native")
    if (admission is None) == (factory is None):
        return None
    observation = admission if admission is not None else factory
    if (not isinstance(observation, dict)
            or type(observation.get("process_id")) is not int or observation["process_id"] != pid
            or type(observation.get("process_generation")) is not int
            or observation["process_generation"] != epoch):
        return None
    if admission is not None:
        if (observation.get("schema") == "rz-rovent-resource-pre-native-admission/1"
                and observation.get("reason") == "resource_plan_provenance"
                and observation.get("native_scope_started") is False
                and value.get("session_evidence") == "scope_unavailable"):
            return "resource_admission_before_scope"
        return None
    if (observation.get("schema") == "rz-rovent-factory-pre-native-observation/1"
            and observation.get("input_lock_sha256") == lock_sha256
            and observation.get("resource_scope_started") is True
            and observation.get("native_entry") == "not_entered_by_original_factory"
            and observation.get("physical_completion") == "not_inferred"
            and value.get("session_evidence") == "before_native"):
        return "factory_before_native"
    return None


def run(args, *, containment=None, started=None, outcome=None, input_registration=None, interruption=None):
    started = time.monotonic() if started is None else started
    run_until = started + args.seconds
    cleanup_until = run_until + args.cleanup_seconds
    raw_arguments = args.native_args[1:] if args.native_args[:1] == ["--"] else args.native_args
    if ("--resource-system-runtime-v3" not in raw_arguments
            or raw_arguments.count("--resource-system-runtime-v3") != 1
            or raw_arguments.count("--custody-control") != 1
            or _once(raw_arguments, "--process-epoch=") != str(args.epoch)):
        raise ValueError("explicit system V3 control and matching epoch required")
    _once(raw_arguments, "--input-lock=")
    _once(raw_arguments, "--resource-plan=")
    _once(raw_arguments, "--resource-plan-sha256=")
    _once(raw_arguments, "--resource-plan-bytes=")

    output = os.open(args.output_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    binary_fd = None
    process = None
    selector = selectors.DefaultSelector()
    streams = {}
    channel = None
    pending_route = None
    auth_phase = "not_started"
    auth_out = bytearray()
    auth_in = bytearray()
    auth_bytes_sent = 0
    notice_seen = False
    authenticated = False
    image_verified = False
    quarantine_notice = False
    barrier_not_released = False
    error = None
    quarantine_reported = False
    exited = False
    owner_reaped = False
    output_drain_unknown = False
    quit_queued = False
    state = "waiting_for_notice"
    input_buffer = bytearray()
    pending = {"stdout": bytearray(), "stderr": bytearray()}
    bytes_seen = {"stdout": 0, "stderr": 0}
    evidence = {"uciok": False, "readyok": False, "bestmove": None}
    log_buffers = {"stdout": bytearray(), "stderr": bytearray()}
    log_truncated = {"stdout": False, "stderr": False}
    line_overflow = {"stdout": False, "stderr": False}
    log_write_errors = {}
    product_finish = None
    pre_native_evidence = None
    pre_native_conflicting = False
    pre_native_accepted = False
    cancelled = False
    previous_handlers = {}

    def request_cancel(_signum, _frame):
        nonlocal cancelled
        cancelled = True
        if interruption is not None and interruption.get("signal") is None:
            interruption["signal"] = _signum

    def stop_requested():
        return cancelled or (interruption is not None and interruption.get("signal") is not None)

    def add_input(value):
        input_buffer.extend(value)
        if process is not None and process.stdin is not None:
            try:
                selector.get_key(process.stdin)
            except KeyError:
                selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")

    def queue_quit():
        nonlocal quit_queued
        if not quit_queued:
            input_buffer.clear()
            add_input(b"stop\nquit\n")
            quit_queued = True

    def line(name, raw):
        nonlocal notice_seen, pending_route, error, state, quarantine_notice, product_finish
        nonlocal pre_native_evidence, pre_native_conflicting
        text = raw.decode("utf-8", errors="replace").strip()
        if name == "stderr" and raw.startswith(b"{"):
            try:
                value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
                if not isinstance(value, dict):
                    value = None
            except (UnicodeDecodeError, ValueError, RecursionError):
                value = None
            if value and value.get("schema") == NOTICE_SCHEMA:
                if notice_seen:
                    error = error or "duplicate_v3_notice"
                    return
                notice_seen = True
                try:
                    pending_route = _notice_route(value, process.pid, args.epoch, args.lock_sha256, args.cuda_ordinal)
                except Exception as exc:
                    error = error or (str(exc) if isinstance(exc, PreflightError) else type(exc).__name__)
            elif value and value.get("schema") == QUARANTINE_SCHEMA:
                quarantine_notice = True
                product_finish = None
                error = error or "child_quarantine"
            elif value and value.get("schema") in ("rz-rovent-native-system-resource-run/3", "rz-rovent-native-system-resource-run-failure/3"):
                failed_receipt = value.get("schema") == "rz-rovent-native-system-resource-run-failure/3"
                if failed_receipt:
                    error = error or "child_resource_run_failed"
                    observed = _pre_native_evidence(value, process.pid, args.epoch, args.lock_sha256)
                    if observed is not None:
                        if pre_native_evidence is not None:
                            pre_native_conflicting = True
                        else:
                            pre_native_evidence = observed
                finish = value.get("product_finish")
                if (authenticated and image_verified and not quarantine_notice
                        and (failed_receipt or (type(value.get("process_id")) is int and value["process_id"] == process.pid
                             and type(value.get("process_generation")) is int and value["process_generation"] == args.epoch))
                        and isinstance(finish, dict)
                        and finish.get("schema") == "rz-rovent-product-finish-observation/1"
                        and type(finish.get("process_id")) is int and finish["process_id"] == process.pid
                        and type(finish.get("process_epoch")) is int and finish["process_epoch"] == args.epoch
                        and finish.get("input_lock_sha256") == args.lock_sha256
                        and finish.get("state") == "confirmed_by_original_product"
                        and finish.get("device_fence") is False):
                    product_finish = finish
        if name != "stdout":
            return
        if state == "uci" and text == "uciok":
            evidence["uciok"] = True
            state = "ready"
            add_input(b"isready\n")
        elif state == "ready" and text == "readyok":
            evidence["readyok"] = True
            state = "search"
            add_input(f"position startpos\ngo nodes {args.go_nodes}\n".encode("ascii"))
        elif state == "search" and text.startswith("bestmove "):
            words = text.split()
            move = words[1] if len(words) >= 2 else ""
            evidence["bestmove"] = move if re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", move) else None
            if evidence["bestmove"] is None:
                error = error or "bestmove_syntax_invalid"
            state = "quit"
            if not exited:
                queue_quit()

    try:
        for name in ("stdout", "stderr"):
            streams[name] = os.fdopen(os.open(
                f"rovent-v3-{name}.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600, dir_fd=output), "wb", buffering=0)
        binary_fd, expected = _open_pinned_binary(args.binary, args.binary_bytes, args.binary_sha256, run_until)
        env = {"PATH": "/usr/bin:/bin", "LANG": "C", "NVIDIA_TF32_OVERRIDE": "0"}
        if args.numeric_observation:
            env["RZ_V3_SYSTEM_NUMERIC_CAPTURE"] = "1"
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, request_cancel)
        if stop_requested() or time.monotonic() >= run_until:
            raise InterruptedError("cancelled or deadline elapsed before child spawn")
        native_argv = [str(args.binary), *raw_arguments]
        barrier_read = barrier_write = None
        try:
            # Popen may fail after creating a child. Preserve the original
            # input leases unless a separate physical completion is proven.
            if containment is None:
                if stop_requested():
                    raise InterruptedError("cancelled before child spawn")
                if input_registration is not None:
                    input_registration.dispatched = True
                if outcome is not None:
                    outcome["spawned"] = None
                process = subprocess.Popen(
                    native_argv, executable=f"/proc/self/fd/{binary_fd}",
                    pass_fds=(binary_fd,), cwd=args.cwd, env=env, start_new_session=True,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
                if outcome is not None:
                    outcome["spawned"] = True
            else:
                from rovent_cgroup_owner import entry_command
                barrier_read, barrier_write = os.pipe2(os.O_CLOEXEC)
                if stop_requested():
                    raise InterruptedError("cancelled before child spawn")
                if input_registration is not None:
                    input_registration.dispatched = True
                if outcome is not None:
                    outcome["spawned"] = None
                process = subprocess.Popen(
                    entry_command(binary_fd, barrier_read, native_argv),
                    pass_fds=(binary_fd, barrier_read), cwd=args.cwd, env=env, start_new_session=True,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
                if outcome is not None:
                    outcome["spawned"] = True
                os.close(barrier_read)
                barrier_read = None
                if stop_requested() or time.monotonic() >= run_until:
                    os.close(barrier_write)
                    barrier_write = None
                    barrier_not_released = True
                    raise InterruptedError("native cancelled before barrier release")
                release_fd, barrier_write = barrier_write, None
                containment.admit_waiting_child(process.pid, release_fd)
        finally:
            if barrier_read is not None:
                os.close(barrier_read)
            if barrier_write is not None:
                os.close(barrier_write)
                barrier_not_released = True
        for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        os.set_blocking(process.stdin.fileno(), False)
        while True:
            now = time.monotonic()
            if not image_verified and now < run_until:
                try:
                    actual = os.stat(f"/proc/{process.pid}/exe")
                    image_verified = (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino)
                except OSError:
                    pass
            if (pending_route is not None and image_verified and channel is None
                    and not stop_requested() and error is None and now < run_until):
                channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                channel.setblocking(False)
                status = channel.connect_ex(pending_route)
                if status not in (0, errno.EINPROGRESS, errno.EALREADY, errno.EAGAIN):
                    error = error or f"parent_connect_errno:{status}"
                else:
                    auth_phase = "connecting"
                    selector.register(channel, selectors.EVENT_WRITE, "auth")
            if authenticated and image_verified and state == "waiting_for_notice" and now < run_until:
                state = "uci"
                add_input(b"uci\n")
            if (stop_requested() or error is not None or now >= run_until) and not quit_queued:
                error = error or ("cancelled" if stop_requested() else "run_deadline_elapsed")
                queue_quit()
            if now >= cleanup_until:
                error = error or "cleanup_deadline_elapsed"
            if (stop_requested() or error is not None or now >= run_until) and auth_phase in ("connecting", "write", "read"):
                # Already-written admission bytes can have authorized native
                # entry. Keep that same stream and owner, but never send more.
                try:
                    selector.unregister(channel)
                except KeyError:
                    pass
                auth_phase = "aborted_after_send" if auth_bytes_sent else "aborted_before_send"
                if not auth_bytes_sent:
                    channel.close()
                    channel = None
            try:
                exited = _exit_observed(process.pid)
            except ChildProcessError:
                error = error or "original_child_wait_relationship_lost"
                exited = False
                quarantine_notice = True
            if exited:
                # Hold the unreaped child while draining all available output.
                # Escaped descendants keeping pipes open are not evidence of
                # completion; do not wait for them beyond the cleanup window.
                try:
                    selector.unregister(process.stdin)
                except KeyError:
                    pass
                readers_open = any(key.data in ("stdout", "stderr") for key in selector.get_map().values())
                if not readers_open or now >= cleanup_until:
                    if readers_open:
                        # Preserve the already observed session claim while
                        # marking trailing pipe content unobserved. This is
                        # not proof that no later diagnostic was written.
                        output_drain_unknown = True
                    # Child output alone is a reported session observation,
                    # not a GPU fence. The PID reservation is released only
                    # after this original product reports session finish and
                    # the same process group has no live members.
                    workload_empty = containment is None or containment.workload_empty()
                    # A pre-native producer record is useful only for the
                    # same pinned image and a fully drained, contained child.
                    # An admission record predates parent authentication and
                    # does not itself bind an input-lock digest.
                    accepted_pre_native = (
                        containment is not None and pre_native_evidence is not None
                        and not pre_native_conflicting and not quarantine_notice
                        and image_verified and not readers_open
                        and not line_overflow["stderr"] and not log_truncated["stderr"]
                        and not pending["stderr"]
                        and (pre_native_evidence != "factory_before_native" or authenticated)
                    )
                    if ((product_finish is not None or accepted_pre_native)
                            and not _group_has_live_members(process.pid) and workload_empty):
                        process.wait()
                        owner_reaped = True
                        pre_native_accepted = accepted_pre_native and product_finish is None
                        if outcome is not None:
                            outcome.update(session_reported=product_finish is not None,
                                           before_native=pre_native_accepted,
                                           pre_native_evidence=pre_native_evidence if pre_native_accepted else None,
                                           stderr_tail="incomplete_line" if pending["stderr"] else "none",
                                           output_drain="unknown_trailing_bytes" if output_drain_unknown else "eof_observed",
                                           group_gone=True, workload_empty=workload_empty)
                        break
                    if not quarantine_reported:
                        quarantine_reported = True
                        _report({"schema": REPORT_SCHEMA, "status": "quarantined_original_owner_retained",
                                 "child_pid": process.pid, "physical_completion": "unknown",
                                 "reason": "session_or_group_completion_unobserved"})
            if (now >= cleanup_until or quarantine_notice) and not quarantine_reported:
                quarantine_reported = True
                _report({"schema": REPORT_SCHEMA, "status": "quarantined_original_owner_retained",
                         "child_pid": process.pid, "physical_completion": "unknown",
                         "reason": error or ("child_quarantine" if quarantine_notice else "cleanup_deadline_elapsed")})
            for key, _ in selector.select(0.05):
                if key.data == "auth":
                    if stop_requested() or error is not None or time.monotonic() >= run_until:
                        continue
                    try:
                        if auth_phase == "connecting":
                            status = channel.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                            if status:
                                raise OSError(status, "parent socket connect failed")
                            pid, uid, _ = struct.unpack(
                                "3i", channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
                            if pid != process.pid or uid != os.geteuid():
                                raise PreflightError("socket_peer_mismatch")
                            auth_out.extend(json.dumps({"schema": REQUEST_SCHEMA,
                                "child_pid": process.pid, "process_epoch": args.epoch,
                                "lock_sha256": args.lock_sha256, "cuda_ordinal": args.cuda_ordinal},
                                separators=(",", ":")).encode("utf-8") + b"\n")
                            auth_phase = "write"
                        if auth_phase == "write" and not stop_requested() and time.monotonic() < run_until:
                            count = channel.send(auth_out)
                            if count == 0:
                                raise PreflightError("parent_write_zero")
                            auth_bytes_sent += count
                            del auth_out[:count]
                            if not auth_out:
                                auth_phase = "read"
                                selector.modify(channel, selectors.EVENT_READ, "auth")
                        elif auth_phase == "read":
                            chunk = channel.recv(4096 - len(auth_in))
                            if not chunk:
                                raise PreflightError("parent_ack_eof")
                            auth_in.extend(chunk)
                            if len(auth_in) >= 4096:
                                raise PreflightError("parent_ack_too_large")
                            if b"\n" in auth_in:
                                frame, _, trailing = auth_in.partition(b"\n")
                                ack = _decode(frame)
                                expected_ack = {"schema": ACK_SCHEMA, "child_pid": process.pid,
                                    "process_epoch": args.epoch, "lock_sha256": args.lock_sha256,
                                    "cuda_ordinal": args.cuda_ordinal, "scope": ACK_SCOPE}
                                if trailing or ack != expected_ack or any(type(ack[key]) is not int for key in ("child_pid", "process_epoch", "cuda_ordinal")):
                                    raise PreflightError("parent_ack_identity_mismatch")
                                authenticated = True
                                auth_phase = "authenticated"
                                selector.unregister(channel)
                    except BlockingIOError:
                        pass
                    except (OSError, PreflightError, KeyError) as exc:
                        error = error or f"parent_authentication:{type(exc).__name__}"
                    continue
                if key.data == "stdin":
                    try:
                        count = os.write(key.fileobj.fileno(), input_buffer)
                        if count == 0:
                            raise OSError("stdin write returned zero")
                        del input_buffer[:count]
                        if not input_buffer:
                            selector.unregister(key.fileobj)
                    except (BrokenPipeError, OSError):
                        error = error or "stdin_write_failed"
                        selector.unregister(key.fileobj)
                    continue
                try:
                    chunk = os.read(key.fileobj.fileno(), 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                name = key.data
                bytes_seen[name] += len(chunk)
                remaining = max(0, LOG_LIMIT - len(log_buffers[name]))
                log_buffers[name].extend(chunk[:remaining])
                if len(chunk) > remaining:
                    log_truncated[name] = True
                pending[name].extend(chunk)
                while b"\n" in pending[name]:
                    item, _, rest = pending[name].partition(b"\n")
                    pending[name] = bytearray(rest)
                    if len(item) <= LINE_LIMIT:
                        line(name, item)
                    else:
                        line_overflow[name] = True
                if len(pending[name]) > LINE_LIMIT:
                    pending[name].clear()
                    line_overflow[name] = True

        for name, stream in streams.items():
            try:
                stream.write(log_buffers[name])
            except OSError as failure:
                log_write_errors[name] = type(failure).__name__
        if not authenticated or not image_verified:
            error = error or "parent_or_image_not_verified"
        if not all((evidence["uciok"], evidence["readyok"], evidence["bestmove"])):
            error = error or "uci_smoke_incomplete"
        if process.returncode != 0:
            error = error or "child_failed"
        _report({"schema": REPORT_SCHEMA, "status": "uci_exchange_observed" if error is None else "failed_or_incomplete",
                 "error_type": error, "child_pid": process.pid, "exit_code": process.returncode,
                 "binary_sha256": args.binary_sha256, "input_lock_sha256": args.lock_sha256,
                 "elapsed_ms": int((time.monotonic() - started) * 1000),
                 "execution_budget_seconds": args.seconds, "cleanup_budget_seconds": args.cleanup_seconds,
                 "parent_authenticated": authenticated, "child_image_observed": image_verified,
                 "uci": evidence, "stdout_bytes": bytes_seen["stdout"], "stderr_bytes": bytes_seen["stderr"],
                 "quarantine_observed": quarantine_notice,
                 "output_drain": "unknown_trailing_bytes" if output_drain_unknown else "eof_observed",
                 "stderr_tail": "incomplete_line" if pending["stderr"] else "none",
                 "physical_completion": ("native_not_entered_by_original_product"
                                         if pre_native_accepted else "session_reported_by_original_product"),
                 "pre_native_evidence": pre_native_evidence if pre_native_accepted else None,
                 "pre_native_lock_binding": ("not_in_admission_record" if pre_native_accepted
                                             and pre_native_evidence == "resource_admission_before_scope"
                                             else "factory_reported" if pre_native_accepted else "not_applicable"),
                 "move_legality": "not_evaluated", "log_truncated": log_truncated,
                 "line_overflow": line_overflow, "log_write_errors": log_write_errors,
                 "numeric_acceptance": "not_evaluated", "gpu_device_fence": "not_observed",
                 "group_observation_scope": "same_process_group_no_live_members; escaped_descendants_unknown"})
        return 0 if error is None else 1
    except BaseException as failure:
        if process is not None and not owner_reaped:
            _report({"schema": REPORT_SCHEMA, "status": "quarantined_original_owner_retained",
                     "child_pid": process.pid, "physical_completion": "unknown",
                     "reason": "parent_exception:" + type(failure).__name__})
            # The actual child may have entered native code already. Keep this
            # exact parent, its connection, and the child handle alive even if
            # diagnostics or a local disk operation failed. The frozen
            # native window is not extended and no signal/fence is inferred.
            for pipe in (process.stdout, process.stderr):
                try:
                    if pipe is not None:
                        os.set_blocking(pipe.fileno(), False)
                except OSError:
                    pass
            while True:
                try:
                    before_native = (containment is not None and
                                     (barrier_not_released or
                                      containment.admission_outcome == "rejected_before_grant"))
                    workload_empty = containment is None or containment.workload_empty()
                    if (_exit_observed(process.pid) and (product_finish is not None or before_native)
                            and not _group_has_live_members(process.pid) and workload_empty):
                        process.wait()
                        if outcome is not None:
                            outcome.update(session_reported=product_finish is not None,
                                           before_native=before_native, group_gone=True,
                                           workload_empty=workload_empty)
                        break
                except BaseException:
                    # An unverified/reaped wait relationship is not proof
                    # that the native operation physically completed. This
                    # also retains the owner on cgroup observation failure.
                    pass
                for pipe in (process.stdout, process.stderr):
                    try:
                        if pipe is not None:
                            chunk = os.read(pipe.fileno(), 65536)
                            if pipe is process.stderr and chunk:
                                pending["stderr"].extend(chunk)
                                while b"\n" in pending["stderr"]:
                                    item, _, rest = pending["stderr"].partition(b"\n")
                                    pending["stderr"] = bytearray(rest)
                                    if len(item) <= LINE_LIMIT:
                                        try:
                                            line("stderr", item)
                                        except Exception:
                                            pass
                                if len(pending["stderr"]) > LINE_LIMIT:
                                    pending["stderr"].clear()
                    except (BlockingIOError, OSError):
                        pass
                time.sleep(0.25)
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if channel is not None:
            channel.close()
        selector.close()
        # Popen's PIPE objects outlive the selector. Close them only after the
        # original child was actually waited; unresolved owners stay in the
        # quarantine loop above with their transport intact.
        if process is not None and process.returncode is not None:
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError as close_error:
                        _report({"schema": REPORT_SCHEMA, "status": "stdio_close_failed",
                                 "error_type": type(close_error).__name__})
        for stream in streams.values():
            stream.close()
        if binary_fd is not None:
            os.close(binary_fd)
        os.close(output)


def _managed_control(path, maximum=1024 * 1024):
    from rovent_v3_input_snapshot import open_directory, read_control, relative
    path = Path(path)
    if not path.is_absolute() or _secret_name(path.name):
        raise ValueError("explicit non-secret absolute control file required")
    directory = open_directory(path.parent)
    try:
        data = read_control(directory, relative(path.name))
    finally:
        os.close(directory)
    if len(data) > maximum:
        raise ValueError("control file exceeds bound")
    return data


def _secret_name(name):
    lower = name.lower()
    return (lower.startswith(".env") or "credential" in lower or "service-account" in lower
            or "service_account" in lower or lower in {"id_rsa", "id_ed25519", "id_ecdsa"}
            or "api_key" in lower or "api-key" in lower)


def _failed_acquire_retained(lease):
    """Recognize only this exact durably retained, never-dispatched lease."""
    from managed_outputs import _read, identity, lifetime
    try:
        record = lease.record
        return (record is not None and record.get("state") == "retained"
                and record.get("status") == "acquire_failed_before_dispatch"
                and record.get("pid") == os.getpid() and record.get("lifetime") == lifetime()
                and lease.guard.stream is None and identity(lease.output) == record.get("identity")
                and _read(lease.catalog / (lease.key + ".json")) == record
                and _read(lease.output / ".rovezero-output.json") == record)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError):
        return False


def _recover_same_live_finish(lease, status):
    """Resume only this owner's already-known terminal operation."""
    phase = lease.live_finish_observation()["phase"]
    if phase == "terminal_pending":
        lease.recover_live_terminal()
        return True
    if phase == "terminal_verified_released":
        return True
    if phase == "active_before_intent" and status is not None:
        if lease.live_handoff_observation()["state"] == "suspended":
            lease.resume_live(deadline=time.monotonic() + 2.0)
        lease.finish(status=status, tree_gone=True)
        return True
    return False


def _scope_observation(owner, stage):
    """Optional readback; absence never substitutes for a configured limit."""
    value = {"schema": "rz-rovent-v3-managed-scope-observation/1", "stage": stage,
             "scope": "original_manager_common_scope", "limits": {}, "affinity": "unknown"}
    try:
        owner._check_owner()
        for name in ("cpu.max", "memory.high", "memory.max", "pids.max", "cpuset.cpus.effective"):
            try:
                value["limits"][name] = (owner.scope / name).read_text(encoding="ascii").strip()
            except (OSError, UnicodeError):
                value["limits"][name] = "unknown"
        value["affinity"] = sorted(os.sched_getaffinity(0))
    except (OSError, RuntimeError, ValueError):
        value["owner_identity"] = "unknown"
    return value


def _registered_read(consumer, reader):
    """Read a protected descriptor and publish discovered paths atomically."""
    return consumer.read_and_register_with_future(reader)


def _registered_system_libraries(runtime, *, path_fields=("path",)):
    """Describe actual system-library targets, without opening their contents."""
    if not isinstance(runtime, dict) or not isinstance(runtime.get("libraries"), list):
        raise ValueError("explicit system-runtime library declarations required")
    entries = runtime["libraries"]
    if not 1 <= len(entries) <= 64:
        raise ValueError("bounded system-runtime library declarations required")
    existing, future = [], []
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("absolute system-library path required")
        for field in path_fields:
            if not isinstance(item.get(field), str):
                raise ValueError("absolute system-library path required")
            original = Path(item[field])
            if not original.is_absolute() or ".." in original.parts or any(_secret_name(part) for part in original.parts):
                raise ValueError("non-secret absolute system-library path required")
            # SystemRuntime resolves normal SONAME symlinks before opening.
            # Protect both the currently resolved request and the registered
            # canonical target when a lock supplies both identities.
            target = Path(os.path.realpath(original))
            (existing if target.exists() else future).append(target)
    return existing, future


def _consumer_plan(plan_path, consumer):
    from rovent_v3_input_snapshot import unique_json

    def read_plan():
        plan = unique_json(_managed_control(plan_path))
        if not isinstance(plan, dict) or plan.get("schema") != "rz-rovent-v3-managed-parent-plan/1":
            raise ValueError("explicit managed V3 plan required")
        paths = [plan["checkout_root"], plan["asset_root"], plan["request"]["path"],
                 plan["native_binary"]["path"], plan["registration_binary"]["path"]]
        return plan, paths, [plan["output"]]

    plan = _registered_read(consumer, read_plan)

    def read_request():
        import hashlib
        pin = plan["request"]
        raw = _managed_control(Path(pin["path"]))
        if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
            raise ValueError("request source bytes differ")
        document = unique_json(raw)
        if not isinstance(document, dict):
            raise ValueError("request document must be an object")
        request = document.get("request") if document.get("status") == "draft_incomplete_do_not_execute" else document
        if not isinstance(request, dict):
            raise ValueError("request declaration is unavailable")
        inputs, future = _registered_system_libraries(request.get("system_runtime"))
        return raw, inputs, future

    return plan, _registered_read(consumer, read_request)


def _finish_consumer_observation(consumer, *, original_scope_closed):
    try:
        if not consumer.dispatched:
            consumer.close_if_not_dispatched()
        elif original_scope_closed:
            # Process/cgroup exit is not a GPU fence.
            for lease in reversed(consumer.leases):
                lease.retain_unknown(tree_gone=True)
    finally:
        consumer.finish_broker()


def run_managed(plan_path, *, consumer_args=None):
    """Run one V3 owner, optionally registering its actual input scope."""
    from managed_process import supervisor_signal_scope

    with supervisor_signal_scope() as interruption:
        return _run_managed_registered(plan_path, consumer_args=consumer_args, interruption=interruption)


def _check_pre_dispatch_interruption(interruption):
    if interruption is not None and interruption.get("signal") is not None:
        raise InterruptedError("cancelled before child dispatch")


def _run_managed_registered(plan_path, *, consumer_args, interruption):
    from virtual_system_numeric_prepare import NumericInputRegistration

    started = time.monotonic()
    if consumer_args is None:
        consumer_args = argparse.Namespace(consumer_registry_root=None, consumer_registry_id=None,
                                           consumer_registry_required=False, consumer_input_catalog=[])
    consumer = NumericInputRegistration(consumer_args, [Path(plan_path)])
    try:
        _check_pre_dispatch_interruption(interruption)
        prepared = _consumer_plan(plan_path, consumer) if consumer.registry is not None else None
        _check_pre_dispatch_interruption(interruption)
        result = _run_managed(plan_path, consumer=consumer, prepared=prepared, started=started,
                              interruption=interruption)
    except BaseException as error:
        if not consumer.dispatched:
            try:
                consumer.close_if_not_dispatched()
            except BaseException as close_error:
                error.add_note("consumer input close also failed: " + str(close_error))
        try:
            consumer.finish_broker()
        except BaseException as close_error:
            error.add_note("broker admission close unconfirmed: " + type(close_error).__name__)
        raise
    try:
        _finish_consumer_observation(consumer, original_scope_closed=True)
    except Exception as error:
        # The original execution result remains authoritative; the prior
        # active registration still protects inputs if this observation fails.
        _report({"schema": REPORT_SCHEMA, "status": "consumer_observation_unavailable",
                 "error_type": type(error).__name__, "physical_completion": "unknown"})
    return result


def _run_managed(plan_path, *, consumer, prepared, started, interruption=None):
    """Own the fresh output, snapshots, registration and V3 child together.

    This entry is opt-in. It never accepts a draft request as executable and
    never converts a group-only observation into a device-wide GPU fence.
    """
    import hashlib
    from managed_outputs import OutputLease
    from managed_process import OwnedProcessGroup, ProcessNotSpawned, run_group
    from rovent_v3_input_snapshot import copy_pin, open_directory, protect_output, relative, unique_json
    from rovent_cgroup_owner import CgroupOwner, CgroupSetupError, entry_command

    plan = prepared[0] if prepared is not None else unique_json(_managed_control(plan_path))
    if not isinstance(plan, dict) or plan.get("schema") != "rz-rovent-v3-managed-parent-plan/1":
        raise ValueError("explicit managed V3 plan required")
    seconds, cleanup_seconds = plan["seconds"], plan["cleanup_seconds"]
    epoch, ordinal = plan["process_epoch"], plan["cuda_ordinal"]
    reserve = plan["reserve_bytes"]
    if (type(seconds) is not int or not 1 <= seconds <= 7200
            or type(cleanup_seconds) is not int or not 1 <= cleanup_seconds <= 30
            or type(epoch) is not int or not 1 <= epoch <= 2**64 - 1
            or type(ordinal) is not int or not 0 <= ordinal <= 2**31 - 1
            or type(reserve) is not int or reserve <= 0
            or type(plan["go_nodes"]) is not int or not 1 <= plan["go_nodes"] <= 1_000_000):
        raise ValueError("invalid finite managed plan limits")
    run_until = started + seconds
    cleanup_until = run_until + cleanup_seconds
    output = Path(plan["output"])
    checkout = Path(plan["checkout_root"])
    asset_root = Path(plan["asset_root"])
    if not all(path.is_absolute() and ".." not in path.parts for path in (output, checkout, asset_root)):
        raise ValueError("absolute managed roots required")
    request_pin, native_pin, registration_pin = (plan[key] for key in ("request", "native_binary", "registration_binary"))
    request_source = Path(request_pin["path"])
    request_bytes = prepared[1] if prepared is not None else _managed_control(request_source)
    if (len(request_bytes) != request_pin["bytes"]
            or hashlib.sha256(request_bytes).hexdigest() != request_pin["sha256"]):
        raise ValueError("request source bytes differ")
    source_request_bytes = request_bytes
    document = unique_json(source_request_bytes)
    if not isinstance(document, dict):
        raise ValueError("request document must be an object")
    from_draft = document.get("status") == "draft_incomplete_do_not_execute"
    if from_draft:
        if plan.get("request_resolution") != "validate_under_original_owner":
            raise ValueError("draft resolution must be explicitly selected")
        request = document.get("request")
    else:
        request = document
    if not isinstance(request, dict):
        raise ValueError("request declaration is unavailable")
    engine = request["engine"]
    if (request["purpose"] != "validation" or engine["mode"] != "hybrid_pc"
            or request["position_command"] != "position startpos"
            or engine["search"]["requested_nodes"] != plan["go_nodes"]):
        raise ValueError("managed smoke question differs from request")
    if (request["wall_time_max_ms"] != seconds * 1000
            or request["cleanup_max_ms"] != cleanup_seconds * 1000):
        raise ValueError("managed window differs from registration")
    if request["system_runtime"].get("cuda_device_id") != ordinal:
        raise ValueError("declared CUDA ordinal differs from managed plan")
    binary_path = Path(native_pin["path"])
    registration_path = Path(registration_pin["path"])
    if (not binary_path.is_absolute() or not registration_path.is_absolute()
            or _secret_name(binary_path.name) or _secret_name(registration_path.name)):
        raise ValueError("absolute registered binaries required")
    binary_declaration = engine["binary"]
    if (binary_declaration["path"] != "inputs/bin/" + binary_path.name
            or binary_declaration["bytes"] != native_pin["bytes"]
            or binary_declaration["sha256"] != native_pin["sha256"]):
        raise ValueError("native binary differs from registration")
    artifacts = list(request["provenance"])
    for role in ("proposer", "critic"):
        artifacts.extend((engine["pc"][role]["weights"], engine["pc"][role]["provenance"]))
    unique = {}
    for artifact in artifacts:
        path = str(relative(artifact["path"]))
        if not path.startswith("inputs/") or _secret_name(Path(path).name):
            raise ValueError("invalid artifact snapshot path")
        identity = (artifact["bytes"], artifact["sha256"])
        if path in unique and unique[path] != identity:
            raise ValueError("conflicting artifact snapshot")
        unique[path] = identity
    total = sum(size for size, _ in unique.values()) + native_pin["bytes"] + registration_pin["bytes"]
    if total > reserve:
        raise ValueError("declared snapshots exceed original reservation")

    owner = None
    lease = None
    suspended = False
    original_child_unsettled = False
    native_outcome = {}
    native_attempted = False
    registration_group = None
    native_leaf = "rovent-" + uuid.uuid4().hex
    acquired = False
    partial_publication_unknown = False
    finish_attempted = False
    finish_status = None
    cancel_requested = [False]
    previous_handlers = {}

    def request_cancel(_signum, _frame):
        cancel_requested[0] = True
        if interruption is not None and interruption.get("signal") is None:
            interruption["signal"] = _signum

    def stop_requested():
        return cancel_requested[0] or (interruption is not None and interruption.get("signal") is not None)

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, request_cancel)
        if stop_requested() or time.monotonic() >= run_until:
            raise InterruptedError("managed run cancelled before owner setup")
        # Scope and output remain held by this exact manager process.
        try:
            owner = CgroupOwner.prepare()
        except CgroupSetupError as error:
            recovered = False
            for _ in range(3):
                try:
                    error.recovery.retry()
                    recovered = True
                    break
                except BaseException:
                    time.sleep(0.1)
            if recovered:
                raise error.original from error
            _report({"schema": REPORT_SCHEMA, "status": "cgroup_setup_owner_retained",
                     "reason": type(error.original).__name__, "physical_completion": "not_started"})
            # The exception retains the original rollback capability. These
            # bounded retries are bookkeeping only, never a new native run.
            while not recovered:
                time.sleep(5)
                try:
                    error.recovery.retry()
                    recovered = True
                except BaseException:
                    pass
            raise error.original from error
        _report(_scope_observation(owner, "before_snapshots"))
        parent_fd = open_directory(output.parent)
        try:
            lease = OutputLease(output, checkout, reserve=reserve, retention_days=14,
                                dependency_registry=consumer.registry)
            try:
                lease.acquire()
                acquired = True
            except BaseException:
                # A partially published active row still belongs to this
                # original object; failed acquire is not finish authority.
                partial_publication_unknown = (lease.record is not None and not _failed_acquire_retained(lease))
                raise
            protect_output(lease, output, parent_fd)
        finally:
            os.close(parent_fd)
        lease.prepare_live_handoff()
        lease.suspend_live()
        suspended = True
        if from_draft:
            # Resolving a draft is an explicit plan action, performed only
            # after this process owns both output and workload cgroup. The
            # original draft bytes stay alongside the independently generated
            # request; typed registration still validates the declaration.
            draft_target = output / "inputs" / "system-v3-request-draft-source.json"
            draft_target.parent.mkdir(parents=True, mode=0o700)
            draft_fd = os.open(draft_target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(draft_fd, "wb") as stream:
                stream.write(source_request_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            request_bytes = json.dumps(request, separators=(",", ":"), sort_keys=True).encode("utf-8")
        (output / "inputs" / "bin").mkdir(parents=True, mode=0o700)
        progress = [0, total]
        asset_fd = open_directory(asset_root)
        try:
            for path, (size, sha256) in unique.items():
                if stop_requested():
                    raise InterruptedError("managed snapshot cancelled")
                copy_pin(asset_fd, output, {"path": path, "bytes": size, "sha256": sha256},
                         run_until, progress, stop_requested=stop_requested)
        finally:
            os.close(asset_fd)
        for pin, source in ((native_pin, binary_path), (registration_pin, registration_path)):
            if stop_requested():
                raise InterruptedError("managed snapshot cancelled")
            source_fd = open_directory(source.parent)
            try:
                copy_pin(source_fd, output / "inputs" / "bin",
                         {"path": source.name, "bytes": pin["bytes"], "sha256": pin["sha256"]},
                         run_until, progress, stop_requested=stop_requested)
            finally:
                os.close(source_fd)
            executable_fd = os.open(output / "inputs" / "bin" / source.name, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                os.fchmod(executable_fd, 0o700)
            finally:
                os.close(executable_fd)
        if progress[0] != total:
            raise ValueError("managed snapshot byte count differs")
        request_target = output / "inputs" / "system-v3-registration-request.json"
        request_fd = os.open(request_target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(request_fd, "wb") as stream:
            stream.write(request_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        resource = {"schema": "rz-rovent-native-resource-launch/1",
                    "delegated_cgroup_path": str(owner.delegated_path), "leaf_name": native_leaf,
                    "options": plan["resource_options"], "profile": "functional_validation",
                    "evidence_directory": "resource-evidence"}
        resource_bytes = json.dumps(resource, separators=(",", ":"), sort_keys=True).encode()
        resource_path = output / "inputs" / "system-v3-resource-plan.json"
        resource_fd = os.open(resource_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(resource_fd, "wb") as stream:
            stream.write(resource_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        lease.resume_live(deadline=run_until)
        suspended = False
        lease.check_budget()
        lease.suspend_live()
        suspended = True
        if stop_requested() or time.monotonic() >= run_until:
            raise TimeoutError("managed window expired before registration")

        registration_image = output / "inputs" / "bin" / registration_path.name
        reg_fd, _ = _open_pinned_binary(registration_image, registration_pin["bytes"], registration_pin["sha256"], run_until)
        barrier_read, barrier_write = os.pipe2(os.O_CLOEXEC)
        group = OwnedProcessGroup()
        registration_group = group
        registration_stdout = os.open(output / "registration.stdout", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            argv = entry_command(reg_fd, barrier_read, [str(registration_image), str(request_target), str(output)])
            def admit_registration():
                nonlocal barrier_read, barrier_write, original_child_unsettled
                original_child_unsettled = True
                os.close(barrier_read)
                barrier_read = None
                if stop_requested() or time.monotonic() >= run_until:
                    os.close(barrier_write)
                    barrier_write = None
                    raise InterruptedError("registration cancelled before barrier release")
                release_fd, barrier_write = barrier_write, None
                owner.admit_waiting_child(group._child.pid, release_fd)
            def registration_tick():
                if stop_requested():
                    raise InterruptedError("registration cancelled")
                if os.fstat(registration_stdout).st_size > 1024 * 1024:
                    raise RuntimeError("registration stdout exceeds bound")
            original_child_unsettled = True
            consumer.dispatched = True
            try:
                code, gone, failure = run_group(
                    argv, output, {"PATH": "/usr/bin:/bin", "LANG": "C"},
                    max(0.001, run_until - time.monotonic()), lambda: None,
                    pass_fds=(reg_fd, barrier_read), stdout=registration_stdout,
                    on_spawn=admit_registration, owned_group=group,
                    absolute_deadline=cleanup_until, graceful_drain_deadline=run_until,
                    on_tick=registration_tick)
            except ProcessNotSpawned:
                original_child_unsettled = False
                consumer.dispatched = False
                raise
            if not gone or not owner.workload_empty():
                raise RuntimeError("original registration process scope remains unknown")
            original_child_unsettled = False
            if code != 0 or failure is not None or time.monotonic() >= run_until:
                raise RuntimeError("registration failed or exceeded the original window")
        finally:
            for fd in (barrier_read, barrier_write, reg_fd, registration_stdout):
                if fd is not None:
                    os.close(fd)

        lock_relative = str(relative(request["outputs"]["input_lock"]))
        if not lock_relative.startswith("inputs/"):
            raise ValueError("registered lock is outside inputs")
        lock = unique_json(_managed_control(output / lock_relative))
        lock_sha = lock.get("canonical_sha256")
        if not isinstance(lock_sha, str) or re.fullmatch(r"[0-9a-f]{64}", lock_sha) is None:
            raise ValueError("registered input lock digest missing")
        asset_arguments = plan["native_asset_arguments"]
        allowed_assets = ("--native-export=", "--native-rules=", "--native-public-graph=",
                          "--native-shared-pc-graph=", "--native-cuda-control-inventory=")
        if (not isinstance(asset_arguments, list) or not asset_arguments
                or any(not isinstance(item, str) or not item.startswith(allowed_assets)
                       or not item.split("=", 1)[1].startswith("inputs/")
                       or _secret_name(Path(item.split("=", 1)[1]).name)
                       for item in asset_arguments)):
            raise ValueError("explicit native asset arguments required")
        arguments = ["--resource-system-runtime-v3", "--custody-control",
                     "--resource-plan=inputs/system-v3-resource-plan.json",
                     "--resource-plan-sha256=" + hashlib.sha256(resource_bytes).hexdigest(),
                     "--resource-plan-bytes=" + str(len(resource_bytes)),
                     "--input-lock=" + lock_relative, "--process-epoch=" + str(epoch),
                     "--startup-ms=" + str(plan["startup_ms"]), "--source-verification=observe",
                     "--root-policy=exact-registered", *asset_arguments]
        smoke = argparse.Namespace(binary=output / binary_declaration["path"], binary_bytes=native_pin["bytes"],
                                   binary_sha256=native_pin["sha256"], cwd=output, output_dir=output,
                                   epoch=epoch, lock_sha256=lock_sha, cuda_ordinal=ordinal,
                                   seconds=seconds, cleanup_seconds=cleanup_seconds,
                                   numeric_observation=plan.get("numeric_observation") is True,
                                   go_nodes=plan["go_nodes"], native_args=arguments)
        if stop_requested():
            raise InterruptedError("managed run cancelled before native dispatch")
        original_child_unsettled = True
        native_attempted = True
        native_outcome["spawned"] = False
        result = run(smoke, containment=owner, started=started, outcome=native_outcome,
                     input_registration=consumer, interruption=interruption)
        original_child_unsettled = not ((native_outcome.get("session_reported") is True
                                         or native_outcome.get("before_native") is True)
                                        and native_outcome.get("group_gone") is True
                                        and native_outcome.get("workload_empty") is True)
        if original_child_unsettled:
            raise RuntimeError("original native session remains unknown")
        observation = {
            "schema": "rz-rovent-v3-managed-parent-observation/1",
            "output": str(output), "input_lock_sha256": lock_sha,
            "smoke_result": result,
            "session": ("not_entered_by_original_product" if native_outcome.get("before_native")
                        else "reported_by_original_product"),
            "pre_native_evidence": native_outcome.get("pre_native_evidence"),
            "pre_native_lock_binding": ("not_in_admission_record"
                                        if native_outcome.get("pre_native_evidence") == "resource_admission_before_scope"
                                        else "factory_reported" if native_outcome.get("before_native") else "not_applicable"),
            "stderr_tail": native_outcome.get("stderr_tail", "unknown"),
            "output_drain": native_outcome.get("output_drain", "unknown"),
            "original_process_group": "no_live_members_observed",
            "original_workload_cgroup": "empty_observed",
            "gpu_device_fence": "not_observed",
            "lease_finish": "pending_at_observation",
            "limits_before_close": _scope_observation(owner, "after_native_before_close"),
        }
        receipt_path = output / "managed-parent-observation.json"
        receipt_error = None
        try:
            receipt_fd = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(receipt_fd, "wb") as stream:
                stream.write(json.dumps(observation, sort_keys=True, separators=(",", ":")).encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            # Optional evidence failure is separate from the observed session.
            receipt_error = type(error).__name__
        owner.close(native_leaf)
        owner = None
        # This finite catalog operation happens after native/session closure;
        # it is not extra native or GPU execution time.
        lease.resume_live(deadline=time.monotonic() + 2.0)
        suspended = False
        finish_status = "v3_parent_smoke_complete" if result == 0 else "v3_parent_smoke_failed"
        finish_attempted = True
        lease.finish(status=finish_status, tree_gone=True)
        lease_observation = lease.live_finish_observation()
        _report({"schema": "rz-rovent-v3-managed-parent-result/1", "output": str(output),
                 "smoke_result": result, "session": ("not_entered_by_original_product"
                 if native_outcome.get("before_native") else "reported_by_original_product"),
                 "pre_native_evidence": native_outcome.get("pre_native_evidence"),
                 "gpu_device_fence": "not_observed", "lease_finish": lease_observation["phase"],
                 "lease_marker": str(output / ".rovezero-output.json"),
                 "managed_observation": str(receipt_path) if receipt_error is None else None,
                 "managed_observation_error": receipt_error})
        lease = None
        return result
    except BaseException as error:
        # Before child start or after the exact scope closes, normal cleanup
        # can finish this lease. Unknown original child/cgroup retains owners.
        if native_attempted and (native_outcome.get("spawned") is False or
                (native_outcome.get("before_native") is True
                 and native_outcome.get("group_gone") is True
                 and native_outcome.get("workload_empty") is True)):
            original_child_unsettled = False
        if owner is not None and not original_child_unsettled:
            try:
                if owner.workload_empty():
                    owner.close(native_leaf)
                    owner = None
            except BaseException:
                pass
        if (lease is not None and acquired and finish_attempted
                and owner is None and not original_child_unsettled):
            try:
                if _recover_same_live_finish(lease, finish_status):
                    lease = None
            except BaseException:
                pass
        if (lease is not None and acquired and not finish_attempted
                and owner is None and not original_child_unsettled):
            try:
                if suspended:
                    lease.resume_live(deadline=time.monotonic() + 2.0)
                    suspended = False
                finish_status = "v3_parent_cleanup_failed" if native_attempted else "v3_parent_prepare_failed"
                finish_attempted = True
                lease.finish(status=finish_status, tree_gone=True)
                lease = None
            except BaseException:
                pass
        if owner is not None or partial_publication_unknown or (lease is not None and acquired):
            _report({"schema": REPORT_SCHEMA, "status": "managed_owner_retained",
                     "reason": type(error).__name__, "physical_completion": "unknown",
                     "group_scope": "owned_workload_cgroup", "output": str(output)})
            while owner is not None or partial_publication_unknown or (lease is not None and acquired):
                # Only bookkeeping on the same retained owners is retried.
                # An unresolved child never receives another registration,
                # native request, signal, or extended execution deadline.
                time.sleep(5)
                if (original_child_unsettled and not native_attempted and registration_group is not None
                        and owner is not None):
                    try:
                        gone, _, failure = registration_group.reobserve_and_drain(
                            absolute_deadline=time.monotonic() + 0.25, terminate=False)
                        if gone and failure is None and owner.workload_empty():
                            original_child_unsettled = False
                    except BaseException:
                        pass
                if partial_publication_unknown and _failed_acquire_retained(lease):
                    partial_publication_unknown = False
                    if not acquired:
                        lease = None
                if owner is not None and not original_child_unsettled:
                    try:
                        if owner.workload_empty():
                            owner.close(native_leaf)
                            owner = None
                    except BaseException:
                        pass
                if lease is not None and acquired and owner is None and not original_child_unsettled:
                    try:
                        if finish_attempted:
                            if _recover_same_live_finish(lease, finish_status):
                                lease = None
                        else:
                            if suspended:
                                lease.resume_live(deadline=time.monotonic() + 2.0)
                                suspended = False
                            finish_status = "v3_parent_cleanup_failed" if native_attempted else "v3_parent_prepare_failed"
                            finish_attempted = True
                            lease.finish(status=finish_status, tree_gone=True)
                            lease = None
                    except BaseException:
                        pass
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def _run_direct_registered(args):
    """Protect direct launch inputs before opening the lock or executable."""
    from managed_process import supervisor_signal_scope

    with supervisor_signal_scope() as interruption:
        return _run_direct_registered_scoped(args, interruption)


def _run_direct_registered_scoped(args, interruption):
    from rovent_v3_input_snapshot import relative, unique_json
    from virtual_system_numeric_prepare import NumericInputRegistration

    started = time.monotonic()
    inputs = list(dict.fromkeys((args.binary, args.cwd, args.output_dir)))
    consumer = NumericInputRegistration(args, inputs)
    if consumer.registry is None:
        # Preserve the original non-participating direct path and its clock.
        _check_pre_dispatch_interruption(interruption)
        return run(args, interruption=interruption)
    outcome = {}
    try:
        _check_pre_dispatch_interruption(interruption)
        if consumer.registry is not None:
            arguments = args.native_args[1:] if args.native_args[:1] == ["--"] else args.native_args
            lock_path = args.cwd / relative(_once(arguments, "--input-lock="))

            def read_lock():
                lock = unique_json(_managed_control(lock_path))
                if not isinstance(lock, dict) or not isinstance(lock.get("manifest"), dict):
                    raise ValueError("explicit system V3 lock required")
                runtime = lock["manifest"].get("system_runtime_lock")
                runtime = runtime.get("runtime") if isinstance(runtime, dict) else None
                if not isinstance(runtime, dict) or runtime.get("kind") != "system_cuda":
                    raise ValueError("system CUDA runtime lock required")
                existing, future = _registered_system_libraries(
                    runtime.get("configuration"), path_fields=("requested_path", "canonical_path"))
                return None, existing, future

            _registered_read(consumer, read_lock)
        _check_pre_dispatch_interruption(interruption)
        result = run(args, started=started, outcome=outcome, input_registration=consumer,
                     interruption=interruption)
    except BaseException as error:
        if not consumer.dispatched:
            try:
                consumer.close_if_not_dispatched()
            except BaseException as close_error:
                error.add_note("consumer input close also failed: " + str(close_error))
        try:
            consumer.finish_broker()
        except BaseException as close_error:
            error.add_note("broker admission close unconfirmed: " + type(close_error).__name__)
        raise
    try:
        # Direct mode observes a process group, not a contained original tree;
        # an escaped descendant therefore cannot be excluded. Keep a
        # dispatched lease active instead of asserting tree termination.
        _finish_consumer_observation(consumer, original_scope_closed=False)
    except Exception as error:
        _report({"schema": REPORT_SCHEMA, "status": "consumer_observation_unavailable",
                 "error_type": type(error).__name__, "physical_completion": "unknown"})
    return result


def main():
    from virtual_system_numeric_prepare import add_consumer_arguments

    if len(sys.argv) > 1 and sys.argv[1] == "--managed-plan":
        managed = argparse.ArgumentParser(description=__doc__)
        managed.add_argument("--managed-plan", required=True, type=Path)
        add_consumer_arguments(managed)
        managed_args = managed.parse_args()
        try:
            return run_managed(managed_args.managed_plan, consumer_args=managed_args)
        except Exception as exc:
            _report({"schema": REPORT_SCHEMA, "status": "managed_failed_before_owner",
                     "error_type": type(exc).__name__, "physical_completion": "unknown"})
            return 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-bytes", type=int, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--lock-sha256", required=True)
    parser.add_argument("--cuda-ordinal", type=int, required=True)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--cleanup-seconds", type=int, default=30)
    parser.add_argument("--numeric-observation", action="store_true")
    parser.add_argument("--go-nodes", type=int, default=64)
    add_consumer_arguments(parser)
    parser.add_argument("native_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if (sys.platform != "linux" or not all(path.is_absolute() for path in (args.binary, args.cwd, args.output_dir))
            or not (1 <= args.binary_bytes <= 1024 * 1024 * 1024)
            or not re.fullmatch(r"[0-9a-f]{64}", args.binary_sha256)
            or not re.fullmatch(r"[0-9a-f]{64}", args.lock_sha256)
            or not (1 <= args.epoch <= 2**64 - 1) or not (0 <= args.cuda_ordinal <= 2**31 - 1)
            or not (1 <= args.seconds <= 7200) or not (1 <= args.cleanup_seconds <= 30)
            or not (1 <= args.go_nodes <= 1_000_000)):
        parser.error("invalid bounded Linux V3 smoke configuration")
    try:
        return _run_direct_registered(args)
    except Exception as exc:
        _report({"schema": REPORT_SCHEMA, "status": "failed_before_owner_or_unexpected",
                 "error_type": type(exc).__name__, "physical_completion": "unknown"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
