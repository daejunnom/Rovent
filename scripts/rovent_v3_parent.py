#!/usr/bin/env python3
"""Own one explicit V3 system-CUDA UCI smoke child and its parent stream.

This is a functional smoke driver, not numerical acceptance, a game, or a
device-wide GPU completion proof. An unresolved original child remains owned
after the finite run/cleanup window; the quarantine is reported separately.
"""
from __future__ import annotations

import argparse
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


def run(args):
    started = time.monotonic()
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
    error = None
    quarantine_reported = False
    exited = False
    owner_reaped = False
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
    cancelled = False
    previous_handlers = {}

    def request_cancel(_signum, _frame):
        nonlocal cancelled
        cancelled = True

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
                error = error or "child_quarantine"
            elif value and value.get("schema") in ("rz-rovent-native-system-resource-run/3", "rz-rovent-native-system-resource-run-failure/3"):
                failed_receipt = value.get("schema") == "rz-rovent-native-system-resource-run-failure/3"
                if failed_receipt:
                    error = error or "child_resource_run_failed"
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
            add_input(b"position startpos\ngo movetime 1000\n")
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
        if cancelled or time.monotonic() >= run_until:
            raise InterruptedError("cancelled or deadline elapsed before child spawn")
        process = subprocess.Popen(
            [str(args.binary), *raw_arguments], executable=f"/proc/self/fd/{binary_fd}",
            pass_fds=(binary_fd,), cwd=args.cwd, env=env, start_new_session=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
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
                    and not cancelled and error is None and now < run_until):
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
            if (cancelled or error is not None or now >= run_until) and not quit_queued:
                error = error or ("cancelled" if cancelled else "run_deadline_elapsed")
                queue_quit()
            if now >= cleanup_until:
                error = error or "cleanup_deadline_elapsed"
            if (cancelled or error is not None or now >= run_until) and auth_phase in ("connecting", "write", "read"):
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
                        error = error or "output_drain_incomplete"
                    # Child output alone is a reported session observation,
                    # not a GPU fence. The PID reservation is released only
                    # after this original product reports session finish and
                    # the same process group has no live members.
                    if product_finish is not None and not _group_has_live_members(process.pid):
                        process.wait()
                        owner_reaped = True
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
                    if cancelled or error is not None or time.monotonic() >= run_until:
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
                        if auth_phase == "write" and not cancelled and time.monotonic() < run_until:
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
                 "physical_completion": "session_reported_by_original_product",
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
                    if (_exit_observed(process.pid) and product_finish is not None
                            and not _group_has_live_members(process.pid)):
                        process.wait()
                        break
                except (ChildProcessError, OSError):
                    # An unverified/reaped wait relationship is not proof
                    # that the native operation physically completed.
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
        for stream in streams.values():
            stream.close()
        if binary_fd is not None:
            os.close(binary_fd)
        os.close(output)


def main():
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
    parser.add_argument("native_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if (sys.platform != "linux" or not all(path.is_absolute() for path in (args.binary, args.cwd, args.output_dir))
            or not (1 <= args.binary_bytes <= 1024 * 1024 * 1024)
            or not re.fullmatch(r"[0-9a-f]{64}", args.binary_sha256)
            or not re.fullmatch(r"[0-9a-f]{64}", args.lock_sha256)
            or not (1 <= args.epoch <= 2**64 - 1) or not (0 <= args.cuda_ordinal <= 2**31 - 1)
            or not (1 <= args.seconds <= 7200) or not (1 <= args.cleanup_seconds <= 30)):
        parser.error("invalid bounded Linux V3 smoke configuration")
    try:
        return run(args)
    except Exception as exc:
        _report({"schema": REPORT_SCHEMA, "status": "failed_before_owner_or_unexpected",
                 "error_type": type(exc).__name__, "physical_completion": "unknown"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
