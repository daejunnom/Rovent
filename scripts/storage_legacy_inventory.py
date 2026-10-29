#!/usr/bin/env python3
"""Observe one explicitly named legacy build directory, without adopting it.

This opt-in inventory creates a separate record only. It does not change the
source tree, issue ownership, establish reproducibility, or authorize removal.
It is not called by the ordinary build or inference paths.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
from pathlib import Path
import re
import stat
import sys
import uuid
import hashlib

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

from managed_storage import StorageError
from storage_artifact_review import decode, no_secret
from storage_code_retirement import _Inputs, _file_identity, _need, _note, _path

SCHEMA = "rovezero.legacy-build-observation.v1"
MAX_ENTRIES = 4096
MAX_RECORD_BYTES = 4 << 20
WARNING_BYTES = 18 << 30
_NAME = re.compile(r"(?:original-binaries-build-[0-9]+|binaries-build-[0-9]+|binaries)\Z")


def _reject_secret_like(path):
    no_secret(path)
    for part in Path(path).parts:
        name = part.lower()
        _need(not (any(tag in name for tag in ("secret", "token", "private_key", "private-key"))
                   or name.endswith((".p12", ".pfx", ".kdbx"))
                   or name.startswith(("id_dsa", "id_ecdsa"))),
              "secret-like paths are never observation inputs")


def _rename_noreplace(directory_fd, temporary, name):
    """Linux atomic no-replace publication, or fail without a fallback."""
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        rename = libc.renameat2
    except AttributeError as error:
        raise OSError(errno.ENOSYS, "renameat2 unavailable") from error
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(directory_fd, os.fsencode(temporary), directory_fd, os.fsencode(name), 1) != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), name)


def _publish(directory_fd, name, raw, tick):
    """Publish complete bytes without replacing an existing registration."""
    temporary = ".legacy-observation-" + uuid.uuid4().hex + ".tmp"
    fd = None
    identity = None
    first = None
    published = False
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                     0o600, dir_fd=directory_fd)
        info = os.fstat(fd)
        identity = (info.st_dev, info.st_ino)
        _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "temporary observation identity invalid")
        offset = 0
        while offset < len(raw):
            tick()
            written = os.write(fd, raw[offset:])
            _need(written > 0, "observation write made no progress")
            offset += written
        os.fsync(fd)
        opened = os.fstat(fd)
        named = os.stat(temporary, dir_fd=directory_fd, follow_symlinks=False)
        _need(_file_identity(opened) == _file_identity(named) and opened.st_nlink == 1
              and opened.st_size == len(raw) and (opened.st_dev, opened.st_ino) == identity,
              "temporary observation changed before publication")
        tick()
        _rename_noreplace(directory_fd, temporary, name)
        published = True
        named_publication = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        _need((named_publication.st_dev, named_publication.st_ino) == identity
              and named_publication.st_nlink == 1,
              "published observation identity changed")
        os.fsync(directory_fd)
    except BaseException as error:
        first = error
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except BaseException as error:
                if first is None:
                    first = error
                else:
                    _note(first, "observation FD close also failed: " + str(error))
        if identity is not None and not published:
            try:
                named = os.stat(temporary, dir_fd=directory_fd, follow_symlinks=False)
                _need((named.st_dev, named.st_ino) == identity and stat.S_ISREG(named.st_mode),
                      "temporary observation identity changed; preserve it")
                os.unlink(temporary, dir_fd=directory_fd)
                os.fsync(directory_fd)
            except BaseException as error:
                if first is None:
                    first = error
                else:
                    _note(first, "temporary observation cleanup also failed: " + str(error))
    if first is not None:
        if published:
            raise StorageError("observation publication may exist; durability is unknown") from first
        raise first


def observe(inputs_root: Path, directory: Path, output: Path, *,
            max_bytes: int, max_entries: int = MAX_ENTRIES, seconds: int = 60):
    """Write a create-only snapshot; byte thresholds warn, not grant authority.

    This inventory only observes file metadata, not file contents. Time and
    entry count bound its actual work; the selected files' total size does not.
    ``max_bytes`` is retained as a caller-selected warning threshold.
    """
    _need(sys.platform == "linux", "legacy observation requires Linux")
    _need(type(max_bytes) is int and max_bytes > 0,
          "positive byte warning threshold required")
    _need(type(max_entries) is int and 0 < max_entries <= MAX_ENTRIES,
          "explicit bounded entry limit required")
    for path in (inputs_root, directory, output):
        _reject_secret_like(path)
    paths = [_path(p) for p in (inputs_root, directory, output)]
    inputs_root, directory, output = paths
    _need(inputs_root.name == "inputs" and directory.parent == inputs_root
          and _NAME.fullmatch(directory.name) is not None,
          "explicit allowlisted binaries child of inputs required")
    _need(output.name.endswith(".json") and not output.is_relative_to(inputs_root)
          and not inputs_root.is_relative_to(output),
          "separate explicit JSON observation path required")
    inputs = _Inputs(seconds)
    first = None
    result = None
    try:
        inputs.directory(inputs_root, controlled=True)
        root_fd = inputs.directory(directory, controlled=True)
        destination_fd = inputs.directory(output.parent, controlled=True)
        root_info = os.fstat(root_fd)
        rows = []
        pending = [directory]
        directory_names = {}
        total = 0
        while pending:
            inputs.tick()
            current = pending.pop()
            fd = inputs.directory(current, controlled=True)
            names = []
            with os.scandir(fd) as entries:
                for entry in entries:
                    inputs.tick()
                    _need(len(names) + len(rows) < max_entries,
                          "observation entry bound exceeded")
                    names.append(entry.name)
            names.sort()
            directory_names[current] = names
            for name in names:
                inputs.tick()
                path = current / name
                _reject_secret_like(path)  # Refuse before inspecting any secret-like entry.
                _need(len(rows) < max_entries, "observation entry bound exceeded")
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                _need(info.st_dev == root_info.st_dev and info.st_uid == os.geteuid(),
                      "cross-device or unowned observation entry")
                _need(not stat.S_ISLNK(info.st_mode), "links are not observed")
                if stat.S_ISDIR(info.st_mode):
                    inputs.directory(path, controlled=True)
                    kind = "directory"
                    pending.append(path)
                else:
                    _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                          "ordinary single-link files required")
                    kind = "file"
                    total += info.st_size
                rows.append({"relative_path": str(path.relative_to(directory)), "kind": kind,
                             "device": info.st_dev, "inode": info.st_ino,
                             "mode": info.st_mode, "uid": info.st_uid, "gid": info.st_gid,
                             "links": info.st_nlink, "bytes": info.st_size,
                             "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns})
        # Recheck all names before publication. This is an observation, not a
        # held proof that another process cannot alter them after publication.
        inputs.check()
        for path, before in directory_names.items():
            inputs.tick()
            with os.scandir(inputs.directory(path)) as entries:
                after = []
                for entry in entries:
                    inputs.tick()
                    _need(len(after) < max_entries, "observation entry bound exceeded")
                    after.append(entry.name)
            _need(sorted(after) == before, "observation directory entries changed during scan")
        for row in rows:
            inputs.tick()
            path = directory / row["relative_path"]
            current = os.stat(path.name, dir_fd=inputs.directory(path.parent), follow_symlinks=False)
            _need((current.st_dev, current.st_ino, current.st_mode, current.st_uid,
                   current.st_gid, current.st_nlink, current.st_size, current.st_mtime_ns,
                   current.st_ctime_ns) == (row["device"], row["inode"], row["mode"], row["uid"],
                   row["gid"], row["links"], row["bytes"], row["mtime_ns"], row["ctime_ns"]),
                  "observation entry changed during scan")
        _need(_file_identity(os.fstat(root_fd)) == _file_identity(root_info),
              "observation root changed during scan")
        result = {"schema": SCHEMA, "source": str(directory), "entries": rows,
                  "root_identity": {"device": root_info.st_dev, "inode": root_info.st_ino,
                                    "mode": root_info.st_mode, "uid": root_info.st_uid,
                                    "gid": root_info.st_gid, "links": root_info.st_nlink,
                                    "mtime_ns": root_info.st_mtime_ns, "ctime_ns": root_info.st_ctime_ns},
                  "total_file_bytes": total, "entry_count": len(rows),
                  "registration_only": True, "ownership_adopted": False,
                  "deletion_authorized": False, "reproducibility_proven": False,
                  "contents_hashed": False, "physical_completion_attested": False,
                  "observation_is_snapshot": True}
        raw = (json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
        _need(len(raw) <= MAX_RECORD_BYTES, "observation record bound exceeded")
        inputs.check()
        _publish(destination_fd, output.name, raw, inputs.tick)
    except BaseException as error:
        first = error
    finally:
        first = inputs.close(first)
    if first is not None:
        raise first
    threshold = min(max_bytes, WARNING_BYTES)
    if total > threshold:
        try:
            print(f"rovezero_storage_warning: legacy observation accounts for {total} bytes "
                  f"(warning threshold {threshold} bytes); no cleanup authority was granted",
                  file=sys.stderr)
        except (OSError, ValueError):
            pass
    return result


def inspect_observation(path: Path):
    """Read a bounded immutable observation; never authorize deletion."""
    _reject_secret_like(path)
    path = _path(path)
    _need(path.name.endswith(".json"), "explicit JSON observation required")
    inputs = _Inputs(60)
    first = result = None
    try:
        parent = inputs.directory(path.parent, controlled=True)
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
        try:
            before = os.fstat(fd)
            _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                  and before.st_uid == os.geteuid() and 0 < before.st_size <= MAX_RECORD_BYTES,
                  "bounded single-link owned observation required")
            blocks, count = [], 0
            while count < before.st_size:
                inputs.tick()
                chunk = os.read(fd, min(65536, before.st_size - count))
                _need(bool(chunk), "observation file truncated")
                blocks.append(chunk)
                count += len(chunk)
            _need(not os.read(fd, 1), "observation file grew")
            named = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            _need(_file_identity(before) == _file_identity(os.fstat(fd)) == _file_identity(named),
                  "observation file changed during read")
            raw = b"".join(blocks)
            result = decode(raw)
            _need(type(result) is dict and result.get("schema") == SCHEMA
                  and result.get("registration_only") is True
                  and all(result.get(k) is False for k in ("ownership_adopted", "deletion_authorized",
                          "reproducibility_proven", "physical_completion_attested"))
                  and type(result.get("entries")) is list
                  and type(result.get("entry_count")) is int
                  and result["entry_count"] == len(result["entries"]) <= MAX_ENTRIES,
                  "invalid observation or attempted authority claim")
            _need(type(result.get("source")) is str and type(result.get("root_identity")) is dict,
                  "observation source and root identity required")
            _reject_secret_like(result["source"])
            result = {"schema": SCHEMA, "source": result["source"],
                      "root_identity": result["root_identity"],
                      "entry_count": result["entry_count"],
                      "total_file_bytes": result.get("total_file_bytes"),
                      "record_bytes": len(raw), "record_sha256": hashlib.sha256(raw).hexdigest(),
                      "registration_only": True, "deletion_authorized": False,
                      "current_source_revalidated": False}
        finally:
            os.close(fd)
    except BaseException as error:
        first = error
    finally:
        first = inputs.close(first)
    if first is not None:
        raise first
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-root", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, required=True,
                        help="caller-selected warning threshold; no byte-based rejection")
    parser.add_argument("--max-entries", type=int, default=MAX_ENTRIES)
    parser.add_argument("--seconds", type=int, default=60)
    args = parser.parse_args()
    try:
        result = observe(args.inputs_root, args.directory, args.output,
                         max_bytes=args.max_bytes, max_entries=args.max_entries, seconds=args.seconds)
    except (StorageError, OSError, ValueError) as error:
        print("legacy_observation_error=" + str(error)[:2048], file=sys.stderr)
        return 125
    print(json.dumps({"schema": result["schema"], "source": result["source"],
                      "entry_count": result["entry_count"], "total_file_bytes": result["total_file_bytes"],
                      "registration_only": True, "deletion_authorized": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
