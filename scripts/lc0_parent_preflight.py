#!/usr/bin/env python3
"""Own one Linux LC0 parent-authentication preflight child.

The caller supplies a pinned executable and its normal NativeConfig arguments,
and owns any required output lease and resource scope. The same opened ELF is
hashed, held through execution, and compared to the child image. This verifies
the supplied file identity, not its source/build provenance or immunity to
concurrent writes. The execution FD is also inherited by the child and is not
proof against a further exec inheriting it. No model files or GPU completion
are inspected. The owner and connection remain alive until the original child
naturally terminates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import selectors
import socket
import stat
import struct
import subprocess
import sys
import time
from pathlib import Path

FRAME_LIMIT = 4096
STDERR_LIMIT = 1024 * 1024
CLEANUP_RESERVE_SECONDS = 30
MAX_BINARY_BYTES = 1024 * 1024 * 1024
NOTICE_SCHEMA = "rz-lc0-parent-endpoint/1"
REQUEST_SCHEMA = "rz-rovent-parent-admission/1"
ACK_SCHEMA = "rz-rovent-parent-admission-ack/1"
ACK_SCOPE = "parent-child-connection-not-gpu-fence"


class PreflightError(Exception):
    """A bounded stage label, never native text or caller arguments."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate object member")
        value[key] = item
    return value


def _decode(frame: bytes) -> dict[str, object]:
    if not frame or len(frame) >= FRAME_LIMIT:
        raise PreflightError("invalid_frame_size")
    try:
        value = json.loads(frame.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise PreflightError("invalid_json") from exc
    if not isinstance(value, dict):
        raise PreflightError("invalid_object")
    return value


def _remaining(deadline: float) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise PreflightError("deadline_expired")
    return value


def _open_pinned_binary(path: Path, expected_bytes: int, expected_sha256: str,
                        deadline: float) -> tuple[int, os.stat_result]:
    """Open and hash one ELF without reopening its name after validation."""
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
            raise PreflightError("binary_type_or_size_mismatch")
        digest = hashlib.sha256()
        remaining = expected_bytes
        first = True
        while remaining:
            _remaining(deadline)
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise PreflightError("binary_short_read")
            if first:
                if not chunk.startswith(b"\x7fELF"):
                    raise PreflightError("binary_not_elf")
                first = False
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise PreflightError("binary_grew_during_read")
        after = os.fstat(fd)
        identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                  value.st_mtime_ns, value.st_ctime_ns)
        if identity(before) != identity(after) or digest.hexdigest() != expected_sha256:
            raise PreflightError("binary_identity_mismatch")
        return fd, after
    except BaseException:
        os.close(fd)
        raise


def _recv_frame(channel: socket.socket, deadline: float) -> bytes:
    data = bytearray()
    while len(data) < FRAME_LIMIT:
        channel.settimeout(_remaining(deadline))
        chunk = channel.recv(1)
        if not chunk:
            raise PreflightError("handshake_eof")
        if chunk == b"\n":
            return bytes(data)
        data.extend(chunk)
    raise PreflightError("handshake_frame_too_large")


def _validate_notice(
    notice: dict[str, object], pid: int, epoch: int, lock_sha256: str, ordinal: int
) -> str:
    route = notice.get("control_path")
    prefix = rf"/proc/{pid}/fd/[0-9]+/lc0-system-control\.sock"
    if (
        notice.get("schema") != NOTICE_SCHEMA
        or type(notice.get("process_id")) is not int
        or notice.get("process_id") != pid
        or type(notice.get("process_epoch")) is not int
        or notice.get("process_epoch") != epoch
        or notice.get("lock_sha256") != lock_sha256
        or type(notice.get("cuda_ordinal")) is not int
        or notice.get("cuda_ordinal") != ordinal
        or notice.get("native_started") is not False
        or not isinstance(route, str)
        or re.fullmatch(prefix, route) is None
    ):
        raise PreflightError("notice_identity_mismatch")
    return route


def _authenticate(
    route: str, pid: int, epoch: int, lock_sha256: str, ordinal: int, deadline: float
) -> socket.socket:
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        channel.settimeout(_remaining(deadline))
        channel.connect(route)
        peer_pid, peer_uid, _ = struct.unpack(
            "3i", channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        )
        if peer_pid != pid or peer_uid != os.geteuid():
            raise PreflightError("socket_peer_mismatch")
        request = {
            "schema": REQUEST_SCHEMA,
            "child_pid": pid,
            "process_epoch": epoch,
            "lock_sha256": lock_sha256,
            "cuda_ordinal": ordinal,
        }
        payload = json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\n"
        channel.settimeout(_remaining(deadline))
        channel.sendall(payload)
        ack = _decode(_recv_frame(channel, deadline))
        expected = {
            "schema": ACK_SCHEMA,
            "child_pid": pid,
            "process_epoch": epoch,
            "lock_sha256": lock_sha256,
            "cuda_ordinal": ordinal,
            "scope": ACK_SCOPE,
        }
        if ack != expected or any(
            type(ack[key]) is not int for key in ("child_pid", "process_epoch", "cuda_ordinal")
        ):
            raise PreflightError("ack_identity_mismatch")
        return channel
    except BaseException:
        channel.close()
        raise


def _drain_child(
    process: subprocess.Popen[bytes],
    deadline: float,
    expected_epoch: int,
    lock_sha256: str,
    ordinal: int,
) -> tuple[bool, bool]:
    assert process.stderr is not None
    fd = process.stderr.fileno()
    os.set_blocking(fd, False)
    selector = selectors.DefaultSelector()
    selector.register(fd, selectors.EVENT_READ)
    pending = bytearray()
    total = 0
    channel: socket.socket | None = None
    receipt_unavailable = False
    got_notice = False
    try:
        while True:
            if process.poll() is not None and not selector.get_map():
                if pending:
                    raise PreflightError("unterminated_stderr_frame")
                return got_notice, receipt_unavailable
            events = selector.select(min(_remaining(deadline), 0.25))
            for _, _ in events:
                data = os.read(fd, 8192)
                if not data:
                    selector.unregister(fd)
                    continue
                total += len(data)
                if total > STDERR_LIMIT:
                    raise PreflightError("stderr_limit_exceeded")
                pending.extend(data)
                while b"\n" in pending:
                    line, _, rest = pending.partition(b"\n")
                    pending = bytearray(rest)
                    if b"LC0 parent preflight receipt unavailable" in line:
                        receipt_unavailable = True
                    if got_notice or not line.startswith(b"{"):
                        continue
                    try:
                        notice = _decode(bytes(line))
                    except PreflightError:
                        continue
                    if notice.get("schema") != NOTICE_SCHEMA:
                        continue
                    route = _validate_notice(notice, process.pid, expected_epoch, lock_sha256, ordinal)
                    channel = _authenticate(route, process.pid, expected_epoch, lock_sha256, ordinal, deadline)
                    got_notice = True
                if len(pending) >= FRAME_LIMIT:
                    raise PreflightError("stderr_frame_too_large")
    except BaseException:
        # Preserve this process as the child's real parent, and preserve any
        # authenticated admission stream, even when observation fails. The
        # controlling process must use a retained managed service/handle and
        # may independently stop waiting; this owner does not kill the child.
        announced = False
        while process.poll() is None:
            if not announced and time.monotonic() >= deadline:
                _report_retained(process.pid)
                announced = True
            try:
                events = selector.select(0.5)
            except OSError:
                time.sleep(0.5)
                continue
            for _, _ in events:
                try:
                    data = os.read(fd, 8192)
                except BlockingIOError:
                    continue
                except OSError:
                    selector.unregister(fd)
                    continue
                if not data:
                    selector.unregister(fd)
        raise
    finally:
        # The original admission socket is retained throughout the child's run.
        if channel is not None:
            channel.close()
        selector.close()


def _report_retained(pid: int) -> None:
    try:
        print(json.dumps({
            "schema": "rz-lc0-parent-preflight-wrapper/1",
            "status": "deadline_expired_owner_retained",
            "child_pid": pid,
            "physical_completion": "unknown",
        }, separators=(",", ":")), flush=True)
    except (BrokenPipeError, OSError):
        # A disconnected observer is not permission to release the owner.
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--binary-bytes", required=True, type=int)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--seconds", required=True, type=int)
    parser.add_argument("--lock-sha256", required=True)
    parser.add_argument("--cuda-ordinal", required=True, type=int)
    parser.add_argument("native_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    native_args = args.native_args
    if native_args[:1] == ["--"]:
        native_args = native_args[1:]
    if (
        sys.platform != "linux"
        or not args.binary.is_absolute()
        or not (1 <= args.binary_bytes <= MAX_BINARY_BYTES)
        or re.fullmatch(r"[0-9a-f]{64}", args.binary_sha256) is None
        or not (1 <= args.epoch <= 2**64 - 1)
        or not (1 <= args.seconds <= 3600)
        or not (0 <= args.cuda_ordinal <= 2**31 - 1)
        or re.fullmatch(r"[0-9a-f]{64}", args.lock_sha256) is None
        or not native_args
        or any(value.startswith("--lc0-parent-preflight=") for value in native_args)
    ):
        parser.error("invalid Linux preflight configuration")
    deadline = time.monotonic() + args.seconds + CLEANUP_RESERVE_SECONDS
    process: subprocess.Popen[bytes] | None = None
    binary_fd: int | None = None
    binary_identity = "not_verified"
    error: str | None = None
    notice = False
    receipt_unavailable = False
    termination = "not_started"
    try:
        binary_fd, expected_image = _open_pinned_binary(
            args.binary, args.binary_bytes, args.binary_sha256, deadline)
        argv = [str(args.binary), *native_args, f"--lc0-parent-preflight={args.epoch}:{args.seconds}"]
        process = subprocess.Popen(
            argv, executable=f"/proc/self/fd/{binary_fd}", pass_fds=(binary_fd,),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        observed_image = os.stat(f"/proc/{process.pid}/exe")
        if (observed_image.st_dev, observed_image.st_ino) != (expected_image.st_dev, expected_image.st_ino):
            raise PreflightError("child_image_identity_mismatch")
        binary_identity = "held_fd_hash_and_child_inode_observed"
        notice, receipt_unavailable = _drain_child(process, deadline, args.epoch, args.lock_sha256, args.cuda_ordinal)
        if not notice:
            error = "notice_not_received"
        elif process.returncode != 0:
            error = "child_failed"
        termination = "exited"
    except BaseException as exc:
        error = str(exc) if isinstance(exc, PreflightError) else type(exc).__name__
        if process is not None:
            # If setup failed before the observation loop, retain the parent
            # and drain stderr until the child exits naturally; never signal an
            # unverified binary or let a full pipe manufacture a deadlock.
            announced = False
            stderr_fd = process.stderr.fileno() if process.stderr is not None else None
            if stderr_fd is not None:
                os.set_blocking(stderr_fd, False)
            while process.poll() is None:
                if not announced and time.monotonic() >= deadline:
                    _report_retained(process.pid)
                    announced = True
                if stderr_fd is not None:
                    try:
                        os.read(stderr_fd, 8192)
                    except (BlockingIOError, OSError):
                        pass
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
            termination = "exited_after_error"
    finally:
        if process is not None and process.stderr is not None:
            process.stderr.close()
        if binary_fd is not None:
            os.close(binary_fd)
    result = {
        "schema": "rz-lc0-parent-preflight-wrapper/1",
        "status": "success" if error is None else "failed",
        "error_type": error,
        "child_pid": process.pid if process is not None else None,
        "child_exit_code": process.returncode if process is not None else None,
        "authenticated_notice": notice,
        "receipt_unavailable_observed": receipt_unavailable,
        "termination": termination,
        "binary_identity": binary_identity,
        "model_gpu_and_device_completion": "not_verified",
    }
    print(json.dumps(result, separators=(",", ":")))
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
