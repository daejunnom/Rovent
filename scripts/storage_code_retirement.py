"""One-shot retirement of exact duplicate ELF leaves in an already closed run.

The authority is the live held catalog gate, unchanged terminal row/marker,
original-owner death, external held keepers and a current Linux /proc census.
An old audit JSON is never authority. No lease, owner, expiry, reservation or
custody record is created, adopted or changed. Only the journal and selected
leaf names are written. A failure retains its journal and first exception.

Linux offers no atomic 'not mapped/open anywhere and unlink this exact inode'
operation. The census is bounded to the explicitly checked current proc/PID
namespace. Another process may open/map a leaf after its last observation; a
same-UID writer may race the final name check. Held parent directories and
O_NOFOLLOW prevent following a substituted link, but do not erase that final
check/unlink interval. These limits are reported, never physical-completion or
disk/VHD space-return proof.
"""
from __future__ import annotations

import ast
import __future__
import contextlib
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
from types import CodeType, FunctionType, MethodType, ModuleType
import zipfile

import storage_artifact_review as review
from managed_storage import GiB, StorageError, checked, read_record_snapshot
from managed_outputs import _catalog_rows, _refuse_custody_registry
from storage_guard import CatalogGuard, lifetime, owner_proven_dead

_JOURNAL_CAP = 256 << 10
_MAX_PROC_PIDS = 65536
_MAX_PROC_FDS = 131072
_MAX_PROC_BYTES = 64 << 20
_MAX_TREE_ENTRIES = 100000
_LIMITS = ["current_visible_proc_pid_namespace_only", "census_is_not_atomic_with_unlink",
           "same_uid_name_race_after_final_check", "no_global_physical_completion_proof"]


class CodeCacheRetirementUnsupported(StorageError):
    code = "CodeCacheRetirementUnsupported"


def _need(condition, message):
    if condition is not True:
        raise StorageError(message)


def _path(value):
    review.no_secret(value)
    path = Path(value)
    _need(path.is_absolute() and str(path) == str(value) and len(os.fsencode(path)) <= 4096
          and ".." not in path.parts, "bounded canonical absolute retirement path required")
    _need(checked(path) == path, "retirement path alias or link is forbidden")
    return path


def _inode(info):
    return info.st_dev, info.st_ino


def _directory_identity(info):
    return info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid


def _file_identity(info):
    return (_inode(info), info.st_mode, info.st_uid, info.st_gid, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _note(error, text):
    try:
        error.add_note(text[:2048])
    except BaseException:
        pass


def _describe(error):
    try:
        return str(error)[:1024]
    except BaseException:
        return "<" + type(error).__name__[:128] + ": detail unavailable>"


class _Inputs:
    """Original component-walk directory FDs and independently held files."""
    def __init__(self, seconds):
        _need(type(seconds) is int and 1 <= seconds <= 600,
              "finite <=600s retirement budget required")
        self.until = time.monotonic() + seconds
        self.directories = {}
        self.files = {}
        self.candidates = set()
        self.deleted = set()

    def tick(self):
        _need(time.monotonic() < self.until, "retirement time budget exceeded; originals preserved")

    def directory(self, path, *, controlled=False):
        self.tick()
        path = _path(path)
        if path not in self.directories:
            parent = None if path == Path("/") else self.directory(path.parent)
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
            fd = os.open(str(path) if parent is None else path.name, flags, dir_fd=parent)
            # Register before any fallible check so cleanup retains this FD.
            self.directories[path] = [fd, None]
            self.directories[path][1] = os.fstat(fd)
        fd, before = self.directories[path]
        self.check_directory(path)
        if controlled:
            _need(before.st_uid == os.geteuid() and before.st_mode & 0o022 == 0,
                  "retirement directory must be owner-controlled and not group/world writable")
        return fd

    def check_directory(self, path):
        self.tick()
        fd, before = self.directories[path]
        opened, named = os.fstat(fd), path.lstat()
        _need(stat.S_ISDIR(opened.st_mode) and opened.st_nlink > 0
              and _directory_identity(before) == _directory_identity(opened) == _directory_identity(named),
              "held retirement directory name/inode changed")

    def open(self, pin, maximum):
        self.tick()
        review.raw_pin(pin, maximum)
        path = _path(pin["path"])
        if path in self.files:
            item = self.files[path]
            _need(item["pin"] == pin, "one held file cannot have replacement byte pins")
        else:
            parent = self.directory(path.parent)
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
            try:
                stream = os.fdopen(fd, "rb")
            except BaseException as error:
                try:
                    os.close(fd)
                except BaseException as cleanup:
                    _note(error, "untransferred retirement FD close also failed: " + _describe(cleanup))
                raise
            item = {"path": path, "stream": stream, "before": None, "pin": dict(pin)}
            self.files[path] = item
            before = os.fstat(stream.fileno())
            item["before"] = before
            _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                  and before.st_uid == os.geteuid() and before.st_mode & 0o022 == 0
                  and 0 < before.st_size <= maximum,
                  "retirement FD/type/identity or writable file is forbidden")
            self.hash_file(path)
        self.check_file(path)
        item["stream"].seek(0)
        return item["stream"]

    def hash_file(self, path):
        self.check_file(path)
        item = self.files[path]
        stream, pin = item["stream"], item["pin"]
        digest, offset = hashlib.sha256(), 0
        while offset < pin["bytes"]:
            self.tick()
            raw = os.pread(stream.fileno(), min(65536, pin["bytes"] - offset), offset)
            _need(bool(raw), "retirement held file truncated during hash")
            digest.update(raw)
            offset += len(raw)
        _need(not os.pread(stream.fileno(), 1, offset)
              and (offset, digest.hexdigest()) == (pin["bytes"], pin["sha256"]),
              "retirement original file bytes/hash changed")
        self.check_file(path)

    def check_file(self, path):
        self.tick()
        item = self.files[path]
        before = item["before"]
        _need(before is not None, "retirement file admission did not complete")
        opened = os.fstat(item["stream"].fileno())
        if path in self.deleted:
            _need(_inode(opened) == _inode(before) and opened.st_nlink == 0
                  and opened.st_size == before.st_size and opened.st_mtime_ns == before.st_mtime_ns,
                  "unlinked original ELF still has a name or its held bytes changed")
            try:
                os.stat(path.name, dir_fd=self.directories[path.parent][0], follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise StorageError("unlinked candidate name reappeared; preserve its replacement")
        else:
            named = os.stat(path.name, dir_fd=self.directories[path.parent][0], follow_symlinks=False)
            _need(_file_identity(before) == _file_identity(opened) == _file_identity(named),
                  "held retirement file name/inode/mtime changed")

    def check(self):
        for path in self.directories:
            self.check_directory(path)
        for path in self.files:
            self.check_file(path)

    def fds(self):
        return ({item["stream"].fileno() for item in self.files.values()}
                | {item[0] for item in self.directories.values()})

    def close(self, first=None):
        for item in reversed(list(self.files.values())):
            try:
                item["stream"].close()
            except BaseException as error:
                if first is None:
                    first = error
                else:
                    _note(first, "retirement file FD close also failed: " + _describe(error))
        for item in reversed(list(self.directories.values())):
            fd, item[0] = item[0], None
            if fd is not None:
                try:
                    os.close(fd)
                except BaseException as error:
                    if first is None:
                        first = error
                    else:
                        _note(first, "retirement directory FD close also failed: " + _describe(error))
        return first


def _proc_read(path, maximum, tick, meter):
    tick()
    stream, first, blocks, count = None, None, [], 0
    try:
        stream = open(path, "rb", buffering=0)
        while True:
            tick()
            raw = stream.read(min(65536, maximum + 1 - count))
            if not raw:
                break
            count += len(raw)
            meter["bytes"] += len(raw)
            _need(count <= maximum, "current /proc census read bound exceeded")
            _need(meter["bytes"] <= _MAX_PROC_BYTES, "aggregate /proc census byte bound exceeded")
            blocks.append(raw)
    except BaseException as error:
        first = error
    finally:
        if stream is not None:
            try:
                stream.close()
            except BaseException as error:
                if first is None:
                    first = error
                else:
                    _note(first, "current proc census FD close also failed: " + _describe(error))
    if first is not None:
        raise first
    tick()
    return b"".join(blocks)


def _proc_namespace(tick, meter):
    raw = _proc_read("/proc/self/mountinfo", 1 << 20, tick, meter)
    mounts = []
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) >= 10 and fields[4] == b"/proc":
            split = fields.index(b"-") if b"-" in fields else -1
            _need(split > 5 and len(fields) > split + 3 and fields[split + 1] == b"proc",
                  "current proc mount identity is unknown")
            options = fields[5].split(b",") + fields[split + 3].split(b",")
            _need(all(not item.startswith(b"hidepid=") or item == b"hidepid=0" for item in options),
                  "current proc census has hidden or unknown process visibility")
            mounts.append(fields)
    _need(len(mounts) == 1, "one explicit current proc mount required")
    own, init = os.stat("/proc/self/ns/pid"), os.stat("/proc/1/ns/pid")
    _need(_inode(own) == _inode(init), "current proc/PID namespace is ambiguous")
    status = _proc_read("/proc/self/status", 65536, tick, meter)
    identifiers = [line.split()[1:] for line in status.splitlines() if line.startswith(b"NSpid:")]
    _need(identifiers == [[str(os.getpid()).encode()]],
          "current proc namespace visibility is unknown or nested")
    return {"device": own.st_dev, "inode": own.st_ino,
            "scope": "explicit_current_visible_proc_pid_namespace",
            "host_initial_namespace_attested": False}


def _process_census(targets, excluded_fds, tick):
    """Fail closed on use or inaccessible observations; never skip a whole PID."""
    meter = {"bytes": 0, "pids": 0, "fds": 0}
    try:
        namespace = _proc_namespace(tick, meter)
        with os.scandir("/proc") as entries:
            pids = []
            for entry in entries:
                tick()
                if entry.name.isascii() and entry.name.isdecimal():
                    pids.append(entry.name)
                    _need(len(pids) <= _MAX_PROC_PIDS, "current proc PID bound exceeded")
        for pid in pids:
            tick()
            root = Path("/proc") / pid
            try:
                before = _proc_read(root / "stat", 8192, tick, meter)
                birth = before.rsplit(b")", 1)[1].split()[19]
                for kind in ("exe", "cwd"):
                    try:
                        observed = os.stat(root / kind)
                    except OSError as error:
                        if error.errno not in (errno.ENOENT, errno.ESRCH):
                            raise
                    else:
                        _need(_inode(observed) not in targets, "code cache is in use by current proc " + kind)
                with os.scandir(root / "fd") as entries:
                    for entry in entries:
                        tick()
                        meter["fds"] += 1
                        _need(meter["fds"] <= _MAX_PROC_FDS, "current proc FD bound exceeded")
                        # Only our actual retained FD numbers are excluded.
                        if int(pid) == os.getpid() and entry.name.isdecimal() and int(entry.name) in excluded_fds:
                            continue
                        try:
                            observed = os.stat(entry.path)
                        except OSError as error:
                            if error.errno not in (errno.ENOENT, errno.ESRCH):
                                raise
                        else:
                            _need(_inode(observed) not in targets, "code cache is in use by current proc FD")
                maps = _proc_read(root / "maps", 4 << 20, tick, meter)
                for line in maps.splitlines():
                    tick()
                    fields = line.split(None, 5)
                    _need(len(fields) >= 5, "current proc maps format is unknown")
                    major, minor = fields[3].split(b":")
                    observed = (os.makedev(int(major, 16), int(minor, 16)), int(fields[4]))
                    _need(observed not in targets, "code cache is in use by current proc maps")
                after = _proc_read(root / "stat", 8192, tick, meter)
                _need(after.rsplit(b")", 1)[1].split()[19] == birth,
                      "current proc PID identity changed during census")
                meter["pids"] += 1
            except OSError as error:
                if error.errno not in (errno.ENOENT, errno.ESRCH):
                    raise
                # Only a genuinely gone PID can make a failed census harmless.
                try:
                    root.stat()
                except FileNotFoundError:
                    continue
                raise StorageError("current proc census incomplete for a still-visible PID") from error
        _need(_proc_namespace(tick, meter) == namespace, "current proc namespace changed")
        return {**meter, "namespace": namespace, "target_inodes": len(targets),
                "own_held_fd_numbers_only_excluded": True, "complete": True,
                "limits": list(_LIMITS)}
    except (OSError, ValueError, IndexError) as error:
        raise StorageError("current proc census visibility/permission/format unknown; preserve code cache") from error


def _tree_bytes(root, inputs):
    # Metadata only, under original nofollow directory FDs. In particular a
    # lstat/scandir pathname race cannot walk an unrelated payload or secret.
    seen, meter = set(), {"bytes": 0, "entries": 0}

    def walk(fd, path, depth):
        inputs.tick()
        _need(depth <= 128, "retirement budget directory depth exceeded")
        original = _directory_identity(os.fstat(fd))
        first, children = None, None
        try:
            children = os.scandir(fd)
            for child in children:
                inputs.tick()
                child_path = path / child.name
                review.no_secret(child_path)
                meter["entries"] += 1
                _need(meter["entries"] <= _MAX_TREE_ENTRIES, "retirement output budget entry bound exceeded")
                info = os.stat(child.name, dir_fd=fd, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    child_fd = os.open(child.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                       dir_fd=fd)
                    child_error = None
                    try:
                        _need(_directory_identity(os.fstat(child_fd)) == _directory_identity(info),
                              "retirement budget child directory changed")
                        walk(child_fd, child_path, depth + 1)
                    except BaseException as error:
                        child_error = error
                    finally:
                        try:
                            os.close(child_fd)
                        except BaseException as error:
                            if child_error is None:
                                child_error = error
                            else:
                                _note(child_error, "retirement budget directory close also failed: " + _describe(error))
                    if child_error is not None:
                        raise child_error
                    _need(_directory_identity(os.stat(child.name, dir_fd=fd, follow_symlinks=False))
                          == _directory_identity(info), "retirement budget directory name changed")
                elif stat.S_ISREG(info.st_mode):
                    if _inode(info) not in seen:
                        seen.add(_inode(info))
                        meter["bytes"] += info.st_size
                elif stat.S_ISLNK(info.st_mode):
                    meter["bytes"] += info.st_size  # Metadata counted; never followed.
                else:
                    raise StorageError("unknown output inode type in retirement budget")
            _need(_directory_identity(os.fstat(fd)) == original,
                  "retirement budget original directory changed")
        except BaseException as error:
            first = error
        finally:
            if children is not None:
                try:
                    children.close()
                except BaseException as error:
                    if first is None:
                        first = error
                    else:
                        _note(first, "retirement budget iterator close also failed: " + _describe(error))
        if first is not None:
            raise first

    walk(inputs.directory(root), root, 0)
    return meter["bytes"]


def _budget(rows, selected, reserved, written, inputs):
    total, extra = 0, max(0, reserved - written)
    for _, row, output in rows:
        inputs.tick()
        _need(type(row.get("reserved_bytes")) is int and 0 < row["reserved_bytes"] <= 18 * GiB
              and type(row.get("catalog_limit_bytes")) is int
              and row["reserved_bytes"] <= row["catalog_limit_bytes"] <= 18 * GiB,
              "original output reservation and total limit required")
        amount = _tree_bytes(output, inputs)
        if row["key"] == selected:
            amount += extra
        _need(amount <= row["reserved_bytes"], "retirement journal exceeds original output reservation")
        total += amount
    _need(all(total <= row["catalog_limit_bytes"] for _, row, _ in rows),
          "retirement journal exceeds original aggregate output budget")
    return total


def _encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


class _Journal:
    def __init__(self, inputs, output, name, reserved):
        self.path, self.parent = output / name, inputs.directory(output, controlled=True)
        self.fd, self.written, self.reserved, self.digest = None, 0, reserved, hashlib.sha256()
        self.fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                          0o600, dir_fd=self.parent)
        try:
            self.before = self.current = os.fstat(self.fd)
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                _note(error, "untransferred retirement journal FD close also failed: " + _describe(cleanup))
            raise

    def check(self):
        opened = os.fstat(self.fd)
        named = os.stat(self.path.name, dir_fd=self.parent, follow_symlinks=False)
        _need(stat.S_ISREG(opened.st_mode) and opened.st_nlink == 1
              and _inode(opened) == _inode(self.before) == _inode(named)
              and _directory_identity(opened) == _directory_identity(self.before)
              and _file_identity(opened) == _file_identity(named) == _file_identity(self.current)
              and opened.st_size == self.written,
              "original retirement journal name/inode changed")
        digest, offset = hashlib.sha256(), 0
        while offset < self.written:
            raw = os.pread(self.fd, min(65536, self.written - offset), offset)
            _need(bool(raw), "retirement journal readback truncated")
            digest.update(raw)
            offset += len(raw)
        _need(not os.pread(self.fd, 1, offset) and digest.hexdigest() == self.digest.hexdigest()
              and _file_identity(os.fstat(self.fd)) == _file_identity(self.current),
              "retirement journal held readback hash changed")

    def append(self, event):
        self.check()
        raw = _encode(event)
        _need(self.written + len(raw) <= self.reserved <= _JOURNAL_CAP,
              "retirement journal reservation exhausted")
        offset = 0
        while offset < len(raw):
            written = os.write(self.fd, raw[offset:])
            _need(type(written) is int and written > 0, "retirement journal write made no progress")
            self.digest.update(raw[offset:offset + written])
            self.written += written
            offset += written
        self.current = os.fstat(self.fd)
        os.fsync(self.fd)
        self.check()

    def pin(self):
        self.check()
        return {"path": str(self.path), "bytes": self.written, "sha256": self.digest.hexdigest()}

    def close(self):
        fd, self.fd = self.fd, None
        if fd is not None:
            os.close(fd)


def retire_code_cache(plan_pin, *, seconds=300):
    """Apply one exact closed audit plan. A failure is never automatically retried."""
    if sys.platform != "linux":
        raise CodeCacheRetirementUnsupported("code-cache retirement is supported only by native Linux observations")
    inputs = _Inputs(seconds)
    guard = journal = None
    first = result = None
    retired, removed, last_attempt, census = [], 0, None, None
    evidence = {"schema": "rovezero.code-cache-retirement-failure.v1",
                "status": "FAILED_OR_PARTIAL_PRESERVED", "plan": plan_pin,
                "retired_files": retired, "logical_removed_bytes": 0, "partial": False,
                "journal": None, "journal_path": None, "last_attempt": None,
                "physical_completion_attested": False, "physical_space_reclaimed": "unknown",
                "vhd_return": "unknown", "observation_limits": list(_LIMITS)}
    try:
        plan, catalog, code, expected, archive_member = review.read_code_plan(inputs, plan_pin)
        catalog = _path(catalog)
        inputs.directory(catalog, controlled=True)
        guard = CatalogGuard(catalog / "catalog.lock")
        guard.acquire()
        _refuse_custody_registry(catalog)
        rows = _catalog_rows(catalog)
        _need(not any(row["state"] == "active" for _, row, _ in rows),
              "another active catalog row blocks code retirement")
        matches = [entry for entry in rows if entry[1]["key"] == plan["key"]]
        _need(len(matches) == 1, "exact original retirement output row missing")
        record_path, row, output = matches[0]
        output = _path(output)
        inputs.directory(output, controlled=True)
        original = read_record_snapshot(record_path, schema="rovezero.managed-output.v1")
        _need(original[1] == plan["record_sha256"] and original[0] == row,
              "original output row raw pin changed before retirement")
        _need(row["state"] == "retained" and row.get("tree_gone") is True
              and row.get("budget_error") is None and row.get("deferred") is False,
              "only original completed nondeferred budget-clean output may be retired")
        observer = lifetime()
        _need(owner_proven_dead(row.get("pid"), row.get("lifetime"), observer),
              "original owner death not proven; no adoption")
        snapshots = {str(path): read_record_snapshot(path, schema="rovezero.managed-output.v1")
                     for path, _, _ in rows}
        for path in (record_path, output / ".rovezero-output.json"):
            inputs.open({"path": str(path), "bytes": path.lstat().st_size,
                         "sha256": plan["record_sha256"]}, 8192)
        keepers = [_path(plan[name]["path"]) for name in ("artifact", "research")]
        for keeper in keepers:
            _need(not keeper.is_relative_to(catalog) and not catalog.is_relative_to(keeper)
                  and not keeper.is_relative_to(output) and not output.is_relative_to(keeper),
                  "preservation keeper archive must be outside selected output and catalog")
        verified_research = review.research(inputs, plan["research"])
        candidates = []
        for target in code:
            path = _path(output / target["relative"])
            _need(path.is_relative_to(output), "candidate escapes original output")
            inputs.directory(path.parent, controlled=True)
            stream = inputs.open({"path": str(path), "bytes": target["bytes"], "sha256": target["sha256"]}, review.ELF_CAP)
            _need(stream.read(4) == b"\x7fELF", "candidate must contain actual ELF bytes")
            inputs.candidates.add(path)
            candidates.append(path)
        review.canonical_code(inputs, plan["artifact"], archive_member, expected)
        target_inodes = {_inode(inputs.files[path]["before"]) for path in candidates}
        _need(len(target_inodes) == len(candidates), "candidate ELF inode aliases are forbidden")
        _need(all(_inode(item["before"]) not in target_inodes
                  for path, item in inputs.files.items() if path not in inputs.candidates),
              "keeper/control and selected candidate inode overlap is forbidden")
        # Directories protect future use through an existing output cwd/FD too.
        targets = target_inodes | {_inode(inputs.directories[path][1])
                                   for path in {output, *(candidate.parent for candidate in candidates)}}

        def recheck():
            inputs.check()
            guard.assert_held()
            _refuse_custody_registry(catalog)
            current = _catalog_rows(catalog)
            _need(current == rows and all(read_record_snapshot(path, schema="rovezero.managed-output.v1")
                      == snapshots[str(path)] for path, _, _ in current),
                  "original catalog row/marker roster changed during retirement")
            _need(lifetime() == observer
                  and owner_proven_dead(row.get("pid"), row.get("lifetime"), observer),
                  "original owner death or observer lifetime changed")
            if journal is not None:
                journal.check()

        recheck()
        census = _process_census(targets, inputs.fds() | {guard.stream.fileno()}, inputs.tick)
        recheck()
        header = {"schema": "rovezero.code-cache-retirement-journal.v1", "event": "preflight_complete",
                  "plan": plan_pin, "record_sha256": plan["record_sha256"], "key": plan["key"],
                  "artifact": plan["artifact"], "research": plan["research"], "code": code,
                  "candidate_inodes": [list(_inode(inputs.files[path]["before"])) for path in candidates],
                  "census": census, "physical_completion_attested": False,
                  "observation_limits": list(_LIMITS)}
        reserved = len(_encode(header)) + len(code) * 1024 + 16384
        _need(reserved <= _JOURNAL_CAP, "bounded retirement journal plan required")
        _budget(rows, plan["key"], reserved, 0, inputs)
        name = ".rovezero-code-retirement-" + plan_pin["sha256"] + ".jsonl"
        evidence["journal_path"] = str(output / name)
        journal = _Journal(inputs, output, name, reserved)
        journal.append(header)
        os.fsync(inputs.directory(output))  # Durable O_EXCL name before any unlink.
        for index, path in enumerate(candidates):
            recheck()
            _budget(rows, plan["key"], reserved, journal.written, inputs)
            last_attempt = {"index": index, "relative": code[index]["relative"],
                            "unlink_attempted": False, "unlink_returned": False, "directory_fsynced": False}
            journal.append({"event": "unlink_intent", "index": index})
            # Final original FD/name/keeper/catalog checks follow the durable
            # intent. No archive reopen, owner substitution or budget renewal.
            recheck()
            inputs.hash_file(path)
            _process_census(targets, inputs.fds() | {guard.stream.fileno(), journal.fd}, inputs.tick)
            recheck()
            last_attempt["unlink_attempted"] = True
            os.unlink(path.name, dir_fd=inputs.directories[path.parent][0])
            last_attempt["unlink_returned"] = True
            inputs.deleted.add(path)
            retired.append(code[index]["relative"])
            removed += code[index]["bytes"]
            os.fsync(inputs.directories[path.parent][0])
            last_attempt["directory_fsynced"] = True
            inputs.check_file(path)
            journal.append({"event": "unlinked_directory_fsynced", "index": index,
                            "logical_removed_bytes": code[index]["bytes"]})
        recheck()
        journal.append({"event": "complete", "retired_files": len(retired), "logical_removed_bytes": removed})
        result = {"schema": "rovezero.code-cache-retirement-result.v1", "status": "CODE_CACHE_RETIRED",
                  "plan": plan_pin, "retired_files": list(retired), "retired_bytes": removed,
                  "logical_removed_bytes": removed, "journal": journal.pin(),
                  "research": verified_research, "deletion_performed": bool(retired),
                  "original_row_unchanged": True, "reservations_changed": False,
                  "expiry_changed": False, "physical_completion_attested": False,
                  "physical_space_reclaimed": "unknown", "vhd_return": "unknown",
                  "census": census, "observation_limits": list(_LIMITS)}
    except BaseException as error:
        first = error
        if journal is not None:
            try:
                journal.append({"event": "failed_or_partial_preserved", "type": type(error).__name__[:128],
                                "detail": _describe(error), "retired_files": len(retired),
                                "logical_removed_bytes": removed, "last_attempt": last_attempt})
            except BaseException as secondary:
                _note(first, "partial retirement journal append/fsync also failed: " + _describe(secondary))
    finally:
        evidence.update(logical_removed_bytes=removed, partial=bool(retired) or bool(last_attempt and last_attempt["unlink_attempted"]),
                        last_attempt=last_attempt)
        if journal is not None:
            try:
                evidence["journal"] = journal.pin()
            except BaseException as error:
                if first is None:
                    first = error
                else:
                    _note(first, "retirement journal pin also unavailable: " + _describe(error))
            try:
                journal.close()
            except BaseException as error:
                if first is None:
                    first = error
                else:
                    _note(first, "retirement journal FD close also failed: " + _describe(error))
        first = inputs.close(first)
        if guard is not None:
            try:
                guard.release()
            except BaseException as error:
                if first is None:
                    first = error
                else:
                    _note(first, "original catalog guard release also failed: " + _describe(error))
    if first is not None:
        try:
            first.code_cache_retirement = evidence
        except BaseException:
            pass
        _note(first, "retirement evidence preserved: journal=" + str(evidence["journal_path"])
              + "; logical_removed_bytes=" + str(removed) + "; no automatic retry or owner reissue")
        raise first
    return result


_NUMERIC_PREPARED_TOKEN = object()


def _numeric_source_function(module, name):
    value = vars(module).get(name)
    _need(type(module) is ModuleType and type(value) is FunctionType
          and value.__globals__ is vars(module)
          and value.__code__.co_filename == module.__file__, "actual source function required: " + name)
    return value


def _numeric_normalized_code(code):
    return code.replace(co_flags=code.co_flags & ~__future__.annotations.compiler_flag,
        co_consts=tuple(_numeric_normalized_code(value) if type(value) is CodeType else value
                        for value in code.co_consts))


def _numeric_attest_source(pack, originals):
    _need(type(pack) is tuple and len(pack) == 3, "actual borrowed source pack required")
    module, held, pin = pack
    review.raw_pin(pin, 128 << 10)
    _need(type(module) is ModuleType and module.__file__ == pin["path"]
          and type(held) is tuple and len(held) == 3,
          "same actual source module/FD/path required")
    _need(any(type(item) is tuple and len(item) == 3 and item[0] == held[0]
              and item[1] is held[1] and item[2] is held[2] for item in originals.held),
          "borrowed source FD must belong to actual Originals")
    import fcntl
    _need(type(held[0]) is int and held[0] >= 0
          and fcntl.fcntl(held[0], fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY,
          "actual readonly borrowed source FD required")
    before = _file_identity(held[2])
    _need(before == _file_identity(os.fstat(held[0])) == _file_identity(held[1].lstat())
          and held[2].st_size == pin["bytes"], "borrowed source FD/name changed")
    raw = os.pread(held[0], pin["bytes"] + 1, 0)
    _need((len(raw), hashlib.sha256(raw).hexdigest()) == (pin["bytes"], pin["sha256"]),
          "actual borrowed source raw pin changed")
    tree = ast.parse(raw)
    code = compile(raw, module.__file__, "exec", dont_inherit=True)
    compiled = {value.co_name: value for value in code.co_consts if type(value) is CodeType}
    definitions = {node.name: node for node in tree.body
                   if type(node) in (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)}
    for name, definition in definitions.items():
        wanted, actual = compiled[name], vars(module).get(name)
        if type(definition) in (ast.FunctionDef, ast.AsyncFunctionDef):
            candidate = actual
            if definition.decorator_list:
                decorators = definition.decorator_list
                decorator = decorators[0]
                stdlib_contextmanager = (type(decorator) is ast.Name and decorator.id == "contextmanager"
                    and vars(module).get("contextmanager") is contextlib.contextmanager) or (
                    type(decorator) is ast.Attribute and decorator.attr == "contextmanager"
                    and type(decorator.value) is ast.Name and decorator.value.id == "contextlib"
                    and vars(module).get("contextlib") is contextlib)
                _need(len(decorators) == 1 and stdlib_contextmanager,
                      "only the original explicit stdlib contextmanager decorator is supported")
                wrapped = getattr(actual, "__wrapped__", None)
                expected_wrapper = contextlib.contextmanager(FunctionType(wanted, vars(module), name))
                _need(type(actual) is FunctionType and actual.__globals__ is vars(contextlib)
                      and actual.__code__ is expected_wrapper.__code__
                      and actual.__closure__ is not None and len(actual.__closure__) == 1
                      and actual.__closure__[0].cell_contents is wrapped,
                      "actual stdlib contextmanager wrapper and original closure required")
                candidate = wrapped
            _need(type(candidate) is FunctionType and candidate.__globals__ is vars(module)
                  and _numeric_normalized_code(candidate.__code__) == _numeric_normalized_code(wanted),
                  "borrowed source function missing/replaced: " + name)
        else:
            _need(type(actual) is type and actual.__module__ == module.__name__,
                  "borrowed source class missing/replaced: " + name)
            names = {node.name for node in definition.body
                     if type(node) in (ast.FunctionDef, ast.AsyncFunctionDef)}
            for method in (value for value in wanted.co_consts
                           if type(value) is CodeType and value.co_name in names):
                member = vars(actual).get(method.co_name)
                if type(member) in (staticmethod, classmethod):
                    member = member.__func__
                choices = (member.fget, member.fset, member.fdel) if type(member) is property else (member,)
                _need(any(type(choice) is FunctionType and choice.__globals__ is vars(module)
                          and _numeric_normalized_code(choice.__code__) == _numeric_normalized_code(method)
                          for choice in choices),
                      "borrowed source method code changed: " + name + "." + method.co_name)
    for node in ast.parse(raw).body:
        if type(node) is ast.Assign and len(node.targets) == 1 and type(node.targets[0]) is ast.Name:
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            _need(vars(module).get(node.targets[0].id) == value, "borrowed source literal changed")
    _need(before == _file_identity(os.fstat(held[0])) == _file_identity(held[1].lstat()),
          "borrowed source changed during code validation")
    return module


def _numeric_originals_current(originals):
    method = getattr(originals, "recheck", None)
    original = vars(type(originals)).get("recheck")
    _need(type(method) is MethodType and method.__self__ is originals
          and method.__func__ is original and type(original) is FunctionType
          and original.__globals__.get("Originals") is type(originals)
          and original.__code__.co_filename == original.__globals__.get("__file__")
          and type(originals.held) is list, "actual source-owned Originals required; never JSON")
    method()


def _numeric_borrowed_view(view, source_pack, originals, guard, catalog):
    """Check the common G/W public surface; never create/close/renew a view."""
    _numeric_originals_current(originals)
    policy = _numeric_attest_source(source_pack, originals)
    _need(type(view) is policy.CatalogView and type(view.selection) is policy.UnattributedSelection
          and view.guard is guard and type(guard) is policy.CatalogGuard
          and Path(view.catalog) == catalog and Path(guard.path) == catalog / "catalog.lock"
          and view.closed is False and view.failed is False and policy.MAX_SECONDS == 30
          and time.monotonic() < view.until <= time.monotonic() + 30,
          "same live typed policy view, original guard and finite until required")
    _need(source_pack[2] == view.selection.sources["unattributed"], "current source7 cohort source pin required")
    for name in ("rows", "checked_usage", "check", "selection_document"):
        method = getattr(view, name)
        _need(type(method) is MethodType and method.__self__ is view
              and method.__func__ is vars(policy.CatalogView)[name], "original policy bound method required")
    guard_init = vars(type(guard)).get("__init__")
    _need(type(guard_init) is FunctionType, "actual original guard type required")
    namespace = guard_init.__globals__
    guard_module = sys.modules.get(namespace.get("__name__"))
    _need(type(guard_module) is ModuleType and vars(guard_module) is namespace
          and guard_module.CatalogGuard is type(guard), "same actual source7 guard module")
    guard_pin = view.selection.sources["guard"]
    _, guard_held = originals.read(guard_pin, body=False, maximum=128 << 10)
    _numeric_attest_source((guard_module, guard_held, guard_pin), originals)
    guard.assert_held()
    document = view.selection_document()
    _need(document.get("retirement_authority") is False and document.get("old_owner_reissued") is False
          and document.get("original_rows_unchanged") is True
          and document.get("product_deadline_extended") is False, "observation is not borrowed owner authority")
    return guard_module


def _selected_code_keeper(inputs, plan, expected, archive_member, spec):
    """Use the existing CI ZIP and original canonical-code reader; create no archive."""
    stream = inputs.open(plan["artifact"], review.ARCHIVE_CAP)
    with zipfile.ZipFile(stream) as archive:
        entries = review.zip_entries(archive, 1 << 30)
        _need("build-receipt.json" in entries and entries["build-receipt.json"].file_size <= 4 << 20,
              "actual CI producer receipt required")
        receipt = review.decode(archive.read("build-receipt.json"))
        _need(receipt.get("schema") == "rovezero.ci-cpu-validation-bundle/1"
              and receipt.get("source_commit") == plan["source_commit"]
              and all(receipt.get(name) is False for name in ("neural_execution", "gpu_execution", "training_execution")),
              "original selected-code source evidence required; no runtime authority")
        selected = receipt["binaries"][spec["target"]]
        code = plan["code"][0]
        _need({key: selected["pin"][key] for key in ("bytes", "sha256")}
              == {key: code[key] for key in ("bytes", "sha256")}
              and selected.get("requested_features") == list(spec["requested_features"]),
              "same direct selected code target bytes and features")
        commands = [row for row in receipt["commands"]
                    if row.get("label") == "build-" + spec["target"]]
        _need(len(commands) == 1 and type(commands[0].get("exit_code")) is int and commands[0]["exit_code"] == 0
              and commands[0].get("original_tree_cleanup_verified") is True
              and commands[0].get("supervisor_error") is None, "original selected compiler command closure")
        _need(archive_member in entries and entries[archive_member].file_size == receipt["binary_bundle"]["bytes"],
              "actual original canonical code bundle member required")
    review.canonical_code(inputs, plan["artifact"], archive_member, expected)
    return {"artifact": plan["artifact"], "source_commit": plan["source_commit"],
            "target": spec["target"], "original_compiler_exit": 0,
            "producer_status_observed": receipt.get("status"), "full_bundle_or_runtime_admission_issued": False,
            "canonical_code_member_verified": True, "new_archive_created": False}


def _numeric_keeper(inputs, plan, expected, archive_member):
    return _selected_code_keeper(inputs, plan, expected, archive_member,
        review._normal_deferred_selected_code_spec(review.NUMERIC_CODE_PROFILE))


def _numeric_open_raw_evidence(inputs, pin):
    """Only this explicit profile admits a genuinely empty original log FD."""
    review._numeric_raw_pin(pin, review.RESEARCH_CAP, allow_empty=True)
    if pin["bytes"]:
        return inputs.open(pin, review.RESEARCH_CAP)
    inputs.tick()
    path = _path(pin["path"])
    if path in inputs.files:
        _need(inputs.files[path]["pin"] == pin, "one raw evidence FD cannot replace its original pin")
    else:
        parent = inputs.directory(path.parent)
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
        try:
            stream = os.fdopen(fd, "rb")
        except BaseException as error:
            try:
                os.close(fd)
            except BaseException as cleanup:
                _note(error, "untransferred empty evidence FD close also failed: " + _describe(cleanup))
            raise
        item = {"path": path, "stream": stream, "before": None, "pin": dict(pin)}
        inputs.files[path] = item
        before = os.fstat(stream.fileno())
        item["before"] = before
        _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
              and before.st_uid == os.geteuid() and before.st_mode & 0o022 == 0
              and before.st_size == 0, "actual empty ordinary owner-controlled evidence required")
    import fcntl
    stream = inputs.files[path]["stream"]
    _need(fcntl.fcntl(stream.fileno(), fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY,
          "actual readonly raw evidence FD required")
    inputs.hash_file(path)  # Also checks empty SHA, EOF, original FD/name/stat.
    stream.seek(0)
    return stream


def _numeric_raw_evidence(inputs, plan):
    result = review.research(inputs, plan["research"])
    required = [{"pin": plan["record_pin"], "member": "original/catalog-row.json"}, *plan["raw_evidence"]]
    stream = inputs.open(plan["research"], review.RESEARCH_CAP)
    with zipfile.ZipFile(stream) as archive:
        for item in required:
            member = review.member_name(item["member"])
            pin = item["pin"]
            original = (_numeric_open_raw_evidence(inputs, pin) if item in plan["raw_evidence"]
                        else inputs.open(pin, review.RESEARCH_CAP))
            raw = original.read(pin["bytes"] + 1)
            _need(len(raw) == pin["bytes"] and archive.read(member) == raw,
                  "research keeper must preserve exact original row and raw evidence")
    inputs.check()
    return result


class _PreparedNormalDeferredSelectedCode:
    """Current-process held data proof only. A saved audit JSON cannot recreate it."""
    __slots__ = ("_inputs", "_plan", "_pin", "_path", "_keeper", "_research", "_pid",
                 "_hard_until", "_closed", "_consumed")

    def __init__(self, token, **values):
        _need(token is _NUMERIC_PREPARED_TOKEN, "only live FD preparation may issue this proof")
        for key, value in values.items():
            object.__setattr__(self, key, value)

    def __setattr__(self, name, value):
        raise StorageError("prepared selected-code proof is immutable")

    def __copy__(self):
        raise StorageError("prepared selected-code proof is not transferable")

    def __deepcopy__(self, memo):
        raise StorageError("prepared selected-code proof is not transferable")

    def check(self):
        _need(not self._closed and os.getpid() == self._pid
              and self._inputs.until <= self._hard_until, "same original live FD preparation only")
        self._inputs.check()

    def close(self, primary=None):
        if not self._closed:
            object.__setattr__(self, "_closed", True)
            return self._inputs.close(primary)
        return primary


class PreparedNormalDeferredSelectedNumericCode(_PreparedNormalDeferredSelectedCode):
    __slots__ = ()


class PreparedNormalDeferredSelectedNativeCode(_PreparedNormalDeferredSelectedCode):
    __slots__ = ()


def _selected_code_proof_type(profile):
    review._normal_deferred_selected_code_spec(profile)
    return (PreparedNormalDeferredSelectedNumericCode if profile == review.NUMERIC_CODE_PROFILE
            else PreparedNormalDeferredSelectedNativeCode)


def _prepare_normal_deferred_selected_code(plan_pin, *, profile, hard_until, seconds=300):
    """Perform expensive readonly keeper/hash work before the caller opens its view."""
    _need(sys.platform == "linux" and type(hard_until) in (int, float)
          and time.monotonic() < hard_until <= time.monotonic() + 600,
          "original finite native Linux preparation window required")
    inputs = _Inputs(seconds)
    inputs.until = min(inputs.until, hard_until)
    try:
        spec = review._normal_deferred_selected_code_spec(profile)
        plan, catalog, code, expected, archive_member = review._read_normal_deferred_selected_code_plan(inputs, plan_pin, profile)
        inputs.directory(catalog, controlled=True)
        row_stream = inputs.open(plan["record_pin"], 8192)
        row = review.decode(row_stream.read(8193))
        _numeric_target_row(plan, row)
        output = _path(plan["output"])
        inputs.directory(output, controlled=True)
        _need(list(_inode(os.fstat(inputs.directories[output][0]))) == plan["output_identity"]
              and not os.path.lexists(output / ".rovezero-output.json"), "original deferred-directory format required")
        target = code[0]
        path = _path(output / target["relative"])
        inputs.directory(path.parent, controlled=True)
        stream = inputs.open({"path": str(path), "bytes": target["bytes"], "sha256": target["sha256"]}, review.ELF_CAP)
        info = inputs.files[path]["before"]
        _need(stream.read(4) == b"\x7fELF" and info.st_mode & 0o222 == 0,
              "selected candidate must be actual readonly ELF")
        inputs.candidates.add(path)
        for pin in plan["protected_reference_pins"]:
            _need(Path(pin["path"]) != path, "candidate is a declared current reference")
            inputs.open(pin, review.ELF_CAP)
        if plan["duplicate_keeper"] is not None:
            inputs.open(plan["duplicate_keeper"], review.ELF_CAP)
        keeper = _selected_code_keeper(inputs, plan, expected, archive_member, spec)
        research = _numeric_raw_evidence(inputs, plan)
        target_inode = _inode(info)
        _need(all(_inode(item["before"]) != target_inode for name, item in inputs.files.items() if name != path),
              "candidate aliases keeper/control/current reference inode")
        for name in ("artifact", "research", "duplicate_keeper"):
            if plan[name] is not None:
                external = _path(plan[name]["path"])
                _need(not external.is_relative_to(output) and not output.is_relative_to(external)
                      and not external.is_relative_to(catalog) and not catalog.is_relative_to(external),
                      "keepers must be outside original output and catalog")
        inputs.check()
        return _selected_code_proof_type(profile)(_NUMERIC_PREPARED_TOKEN,
            _inputs=inputs, _plan=_encode(plan), _pin=dict(plan_pin), _path=path,
            _keeper=keeper, _research=research, _pid=os.getpid(), _hard_until=inputs.until,
            _closed=False, _consumed=False)
    except BaseException as primary:
        inputs.close(primary)
        raise


def prepare_normal_deferred_selected_numeric_code(plan_pin, *, hard_until, seconds=300):
    return _prepare_normal_deferred_selected_code(plan_pin, profile=review.NUMERIC_CODE_PROFILE,
        hard_until=hard_until, seconds=seconds)


def prepare_normal_deferred_selected_native_code(plan_pin, *, hard_until, seconds=300):
    return _prepare_normal_deferred_selected_code(plan_pin, profile=review.NATIVE_CODE_PROFILE,
        hard_until=hard_until, seconds=seconds)


def _numeric_target_row(plan, row):
    _need(type(row) is dict and row.get("schema") == "rovezero.managed-output.v1"
          and row.get("key") == plan["key"] and row.get("output") == plan["output"]
          and row.get("identity") == plan["output_identity"] and row.get("state") == "retained"
          and row.get("tree_gone") is True and row.get("deferred") is True
          and row.get("budget_error") is None, "same original normal retained/deferred row required")


def _numeric_original_owner_dead(guard, row, now):
    return (_numeric_source_function(guard, "reboot_proves_dead")(row.get("lifetime"), now)
            or _numeric_source_function(guard, "owner_proven_dead")(row.get("pid"), row.get("lifetime"), now))


def _numeric_producer_state(rows, protected_keys):
    return {"no_unprotected_active_producers": not any(
                row["state"] == "active" and row["key"] not in protected_keys for _, row, _ in rows),
            "protected_active_row_count": sum(
                row["state"] == "active" and row["key"] in protected_keys for _, row, _ in rows)}


def _audit_prepared_normal_deferred_selected_code(prepared, profile):
    _need(type(prepared) is _selected_code_proof_type(profile), "actual matching-profile live prepared FD proof required")
    prepared.check()
    _need(review.decode(prepared._plan)["profile"] == profile, "prepared code profile changed")
    return {"schema": "rovezero." + profile + "-audit.v1",
            "status": "READONLY_FD_AND_KEEPERS_VERIFIED_NO_RETIREMENT_AUTHORITY",
            "plan": dict(prepared._pin), "keeper": prepared._keeper, "research": prepared._research,
            "deletion_performed": False, "retirement_authority_issued": False,
            "still_required": ["first_actual_borrowed_current_view", "no_unprotected_active_producers", "original_owner_death",
                               "initial_and_final_current_proc_census", "durable_intent_and_parent_fsync"],
            "physical_completion_attested": False}


def audit_prepared_normal_deferred_selected_numeric_code(prepared):
    return _audit_prepared_normal_deferred_selected_code(prepared, review.NUMERIC_CODE_PROFILE)


def audit_prepared_normal_deferred_selected_native_code(prepared):
    return _audit_prepared_normal_deferred_selected_code(prepared, review.NATIVE_CODE_PROFILE)


def _retire_prepared_normal_deferred_selected_code(prepared, *, profile, borrowed_live_view,
        source_pack, actual_originals, same_guard):
    """One effect attempt with a caller-owned view; never acquire/renew/close its gate."""
    _need(type(prepared) is _selected_code_proof_type(profile) and not prepared._consumed,
          "one actual live preparation, never an audit JSON or automatic retry")
    object.__setattr__(prepared, "_consumed", True)
    inputs, journal, first, result = prepared._inputs, None, None, None
    plan = None
    path = prepared._path
    attempt = {"unlink_attempted": False, "unlink_returned": False, "directory_fsynced": False}
    evidence = {"schema": "rovezero." + profile + "-failure.v1",
        "status": "FAILED_OR_PARTIAL_PRESERVED", "plan": dict(prepared._pin),
        "last_attempt": attempt, "journal": None, "journal_path": None, "logical_removed_bytes": 0,
        "no_unprotected_active_producers": None, "protected_active_row_count": None,
        "physical_completion_attested": False, "physical_space_reclaimed": "unknown",
        "vhd_return": "unknown", "observation_limits": list(_LIMITS)}
    try:
        plan = review.decode(prepared._plan)
        _need(plan["profile"] == profile, "prepared code profile changed")
        prepared.check()
        catalog = _path(plan["catalog"])
        current_guard = _numeric_borrowed_view(borrowed_live_view, source_pack, actual_originals, same_guard, catalog)
        inputs.until = min(inputs.until, borrowed_live_view.until)
        prepared.check()
        rows, usage = borrowed_live_view.checked_usage()
        producer_state = _numeric_producer_state(rows, borrowed_live_view.keys)
        evidence.update(producer_state)
        _need(producer_state["no_unprotected_active_producers"], "unprotected active producer blocks code retirement")
        matches = [item for item in rows if item[1]["key"] == plan["key"]]
        _need(len(matches) == 1 and plan["key"] not in borrowed_live_view.keys,
              "exact unprotected original selected code output required")
        record_path, row, output = matches[0]
        _need(str(record_path) == plan["record_pin"]["path"] and str(output) == plan["output"], "canonical target paths changed")
        _numeric_target_row(plan, row)
        _need(not os.path.lexists(output / ".rovezero-output.json"), "original deferred marker absence changed")
        original = review.decode(inputs.open(plan["record_pin"], 8192).read(8193))
        _need(original == row, "actual canonical row differs from held original bytes")
        now = _numeric_source_function(current_guard, "lifetime")()
        _need(_numeric_original_owner_dead(current_guard, row, now), "original owner death not proven; no adoption")
        targets = {_inode(inputs.files[path]["before"]),
                   _inode(inputs.directories[output][1]), _inode(inputs.directories[path.parent][1])}
        census = _process_census(targets, inputs.fds() | {same_guard.stream.fileno()}, inputs.tick)
        header = {"schema": "rovezero." + profile + "-journal.v1", "event": "preflight_complete",
            "plan": dict(prepared._pin), "key": plan["key"], "record_pin": plan["record_pin"],
            "code": plan["code"], "artifact": plan["artifact"], "research": plan["research"],
            "census": census, **producer_state, "physical_completion_attested": False, "observation_limits": list(_LIMITS)}
        reserved = len(_encode(header)) + 1024 + 16384
        _need(reserved <= _JOURNAL_CAP and usage + reserved <= 18 * GiB
              and all(usage + reserved <= item[1]["catalog_limit_bytes"] for item in rows)
              and _tree_bytes(output, inputs) + reserved <= row["reserved_bytes"], "original journal/output/18GiB budget required")
        name = ".rovezero-code-retirement-" + prepared._pin["sha256"] + ".jsonl"
        evidence["journal_path"] = str(output / name)
        journal = _Journal(inputs, output, name, reserved)
        journal.append(header)
        os.fsync(inputs.directory(output))
        journal.append({"event": "unlink_intent", "relative": plan["code"][0]["relative"]})
        # Exactly one final current boundary; no whole inventory/hash/census per chunk.
        prepared.check()
        _numeric_borrowed_view(borrowed_live_view, source_pack, actual_originals, same_guard, catalog)
        final_rows, final_usage = borrowed_live_view.checked_usage()
        final_producer_state = _numeric_producer_state(final_rows, borrowed_live_view.keys)
        evidence.update(final_producer_state)
        _need(final_rows == rows and final_producer_state == producer_state
              and final_producer_state["no_unprotected_active_producers"], "original catalog rows/history or protected producers changed")
        _need(_numeric_source_function(current_guard, "lifetime")() == now
              and _numeric_original_owner_dead(current_guard, row, now), "original owner or observer lifetime changed")
        _need(final_usage + max(0, reserved - journal.written) <= 18 * GiB
              and all(final_usage + max(0, reserved - journal.written) <= item[1]["catalog_limit_bytes"] for item in rows)
              and _tree_bytes(output, inputs) + max(0, reserved - journal.written) <= row["reserved_bytes"],
              "original final journal/output/18GiB budget required")
        inputs.hash_file(path)
        _process_census(targets, inputs.fds() | {same_guard.stream.fileno(), journal.fd}, inputs.tick)
        inputs.check(); same_guard.assert_held(); journal.check()
        _need(borrowed_live_view.rows() == rows
              and _numeric_source_function(current_guard, "lifetime")() == now
              and _numeric_original_owner_dead(current_guard, row, now),
              "final original rows/history/owner changed after census")
        attempt["unlink_attempted"] = True
        os.unlink(path.name, dir_fd=inputs.directories[path.parent][0])
        attempt["unlink_returned"] = True
        inputs.deleted.add(path)
        os.fsync(inputs.directories[path.parent][0])
        attempt["directory_fsynced"] = True
        inputs.check_file(path)
        journal.append({"event": "unlinked_directory_fsynced", "logical_removed_bytes": plan["code"][0]["bytes"]})
        _need(borrowed_live_view.rows() == rows, "original rows/history changed after leaf retirement")
        inputs.check(); same_guard.assert_held()
        journal.append({"event": "complete", "retired_files": 1})
        result = {"schema": "rovezero." + profile + "-result.v1",
            "status": "CODE_CACHE_RETIRED", "plan": dict(prepared._pin), "journal": journal.pin(),
            "retired_files": [plan["code"][0]["relative"]], "logical_removed_bytes": plan["code"][0]["bytes"],
            **producer_state,
            "original_row_unchanged": True, "reservations_changed": False, "expiry_changed": False,
            "borrowed_guard_or_view_closed": False, "physical_completion_attested": False,
            "physical_space_reclaimed": "unknown", "vhd_return": "unknown", "observation_limits": list(_LIMITS)}
    except BaseException as error:
        first = error
        if journal is not None:
            try:
                journal.append({"event": "failed_or_partial_preserved", "type": type(error).__name__[:128],
                                "detail": _describe(error), "last_attempt": attempt})
            except BaseException as secondary:
                _note(first, "partial retirement journal also failed: " + _describe(secondary))
    finally:
        if journal is not None:
            try:
                evidence["journal"] = journal.pin()
            except BaseException as error:
                if first is None: first = error
                else: _note(first, "retirement journal pin also failed: " + _describe(error))
            try:
                journal.close()
            except BaseException as error:
                if first is None: first = error
                else: _note(first, "retirement journal close also failed: " + _describe(error))
        if attempt["unlink_returned"]:
            evidence["logical_removed_bytes"] = plan["code"][0]["bytes"]
        evidence["partial"] = bool(attempt["unlink_attempted"])
        first = prepared.close(first)
    if first is not None:
        try: first.code_cache_retirement = evidence
        except BaseException: pass
        _note(first, "one exact selected-code attempt; partial effects preserved, no retry or owner/view reissue")
        raise first
    return result


def retire_prepared_normal_deferred_selected_numeric_code(prepared, *, borrowed_live_view,
        source_pack, actual_originals, same_guard):
    return _retire_prepared_normal_deferred_selected_code(prepared, profile=review.NUMERIC_CODE_PROFILE,
        borrowed_live_view=borrowed_live_view, source_pack=source_pack,
        actual_originals=actual_originals, same_guard=same_guard)


def retire_prepared_normal_deferred_selected_native_code(prepared, *, borrowed_live_view,
        source_pack, actual_originals, same_guard):
    return _retire_prepared_normal_deferred_selected_code(prepared, profile=review.NATIVE_CODE_PROFILE,
        borrowed_live_view=borrowed_live_view, source_pack=source_pack,
        actual_originals=actual_originals, same_guard=same_guard)
