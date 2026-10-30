"""SOURCE PROPOSAL ONLY: unattributed evidence is a charge and an exclusion.

No legacy row/identity/owner or transition is repaired.  The default catalog
reader remains strict.  This opt-in has its own source7 domain and requires
actual research-archive readback before a new, disjoint producer may proceed.
Archived sources are historical bytes, never a substitute for live sources.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
import time
import zipfile

import managed_output_preservation_session as legacy
import managed_output_quarantine as custody
import managed_output_quarantine_transition as transitions
from managed_storage import (GiB, StorageError, checked, read_record_snapshot,
                             reject_credential_names, tree_size, write_record,
                             warn_output_threshold)
from storage_guard import CatalogGuard, lifetime

SCHEMA = "rovezero.unattributed-output-preservation.v1"
RECOVERY_SCHEMA = "rovezero.unattributed-research-recovery.v1"
POLICY = "fresh_output_only_unattributed_corpus_preserved_v1"
EXACT_SELECTED_ROWS = 6  # This proposal is deliberately narrower than a generic migration.
MAX_ENTRIES = 4096
MAX_ARCHIVE = 512 << 20
MAX_MANIFEST = 1 << 20
MAX_METADATA = 8192
MAX_CENTRAL_DIRECTORY = 4 << 20
MAX_SECONDS = 30
SOURCE_ROLES = (*legacy.SOURCE_ROLES, "unattributed")
_TOKEN = object()


class ObservationExpired(StorageError):
    """A finite storage observation did not complete; no NN result is implied."""
    code = "StorageObservationExpired"
    failure_scope = "storage_observation"

    def __init__(self, deadline, observed):
        super().__init__("unattributed-preservation: original finite observation expired")
        self.deadline_monotonic = deadline
        self.observed_monotonic = observed


def need(value, reason):
    if value is not True:
        raise StorageError("unattributed-preservation: " + reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate metadata key")
            result[key] = value
        return result
    def constant(_):
        raise StorageError("nonfinite metadata")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _fields(value, fields):
    need(type(value) is dict and set(value) == set(fields), "closed fields required")


def _pin(pin, maximum):
    _fields(pin, ("path", "bytes", "sha256"))
    need(type(pin["path"]) is str and Path(pin["path"]).is_absolute(), "absolute raw pin required")
    need(type(pin["bytes"]) is int and 0 < pin["bytes"] <= maximum, "bounded raw byte count required")
    digest = pin["sha256"]
    need(type(digest) is str and len(digest) == 64
         and all(c in "0123456789abcdef" for c in digest), "raw sha256 required")
    checked(Path(pin["path"]))  # Existing credential/link policy is retained.
    return copy.deepcopy(pin)


def _source_paths():
    return {**legacy._source_paths(), "unattributed": Path(__file__).resolve()}


class UnattributedSelection:
    """Raw, explicit source7 inputs only.  Neither a lease nor a recovery token.

    Each row names its ORIGINAL catalog/marker bytes and an independent archive
    which actually includes every current plain file.  Missing archives cannot
    be replaced by a JSON `recovered=true` assertion or a diagnostic fixture.
    """
    def __init__(self, rows, historical_tips, current_source_pins, *,
                 root_exclusive_writer=False, maximum_seconds=30):
        need(root_exclusive_writer is True, "explicit exclusive observation required")
        need(type(maximum_seconds) in (int, float) and math.isfinite(maximum_seconds)
             and 0 < maximum_seconds <= MAX_SECONDS, "finite <=30s observation required")
        need(type(rows) in (tuple, list) and len(rows) == EXACT_SELECTED_ROWS,
             "this policy requires exactly the six separately selected rows")
        need(type(current_source_pins) is dict and set(current_source_pins) == set(SOURCE_ROLES),
             "new source7 closure required; old source6 is not an alias")
        self.rows, names = [], set()
        for value in rows:
            _fields(value, ("catalog", "marker", "research_archive"))
            item = {name: _pin(value[name], MAX_ARCHIVE if name == "research_archive" else MAX_METADATA)
                    for name in value}
            need(item["catalog"]["path"] not in names, "duplicate selected catalog row")
            names.add(item["catalog"]["path"])
            self.rows.append(item)
        self.rows = tuple(self.rows)
        self.sources = {role: _pin(current_source_pins[role], transitions.SOURCE_RAW_LIMIT)
                        for role in SOURCE_ROLES}
        # This is configuration construction only.  Live observation reads all
        # seven actual files and the original history under its own OS gate.
        need(type(historical_tips) in (tuple, list) and 0 < len(historical_tips) <= custody.MAX_RUNS,
             "bounded explicit historical tip pins required")
        self.tips = tuple(_pin(pin, custody.RECEIPT_LIMIT) for pin in historical_tips)
        self.maximum_seconds = maximum_seconds
        self._configuration_digest = _sha(custody._encode({"rows": self.rows, "sources": self.sources,
            "tips": self.tips, "maximum_seconds": self.maximum_seconds}))
        self._bound_deadline = None
        self._frozen = True

    def __setattr__(self, name, value):
        need(not getattr(self, "_frozen", False), "selection is immutable; create a new configuration")
        object.__setattr__(self, name, value)

    @property
    def original_hard_until(self):
        need(self._bound_deadline is not None, "original caller deadline not bound")
        return self._bound_deadline


class _Held:
    """One actual finite observation; target UID observations grant no ownership."""
    def __init__(self, seconds, hard_until):
        need(sys.platform == "linux", "Linux-only proposal")
        need(type(hard_until) in (int, float) and math.isfinite(hard_until), "original finite deadline required")
        self.until = min(time.monotonic() + seconds, hard_until)
        self.fds, self.files, self.directories, self.trees = [], [], {}, []
        self.file_parents = {}
        self.blobs = []
        try:
            self.tick()
            self.now = lifetime()
            self.namespace_fd = self._own(os.open("/proc/self/ns/mnt", os.O_RDONLY | os.O_CLOEXEC))
            self.namespace = (os.fstat(self.namespace_fd).st_dev, os.fstat(self.namespace_fd).st_ino)
            self.mounts = self._mounts()
        except BaseException as error:
            self.close(error)
            raise

    def _own(self, fd):
        self.fds.append(fd)
        return fd

    def tick(self):
        observed = time.monotonic()
        if not observed < self.until:
            raise ObservationExpired(self.until, observed)

    def _mounts(self):
        self.tick()
        with open("/proc/self/mountinfo", "rb") as stream:
            raw = stream.read((1 << 20) + 1)
        need(0 < len(raw) <= 1 << 20, "bounded mount observation required")
        return _sha(raw)

    def directory(self, path):
        self.tick()
        path = Path(path)
        if path not in self.directories:
            path = checked(path)
            need(path.is_absolute(), "absolute held directory required")
            need(len(self.directories) < 512, "held directory bound exceeded")
            if path == Path("/"):
                fd = self._own(os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC))
                named = path.lstat()
            else:
                parent = self.directory(path.parent)
                need(len(self.directories) < 512, "held directory bound exceeded")
                named = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
                fd = self._own(os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                       dir_fd=parent))
            before = custody._did(os.fstat(fd))
            need(stat.S_ISDIR(named.st_mode) and before == custody._did(named), "directory FD/name changed")
            self.directories[path] = (fd, before)
        fd, before = self.directories[path]
        named = (path.lstat() if path == Path("/") else
                 os.stat(path.name, dir_fd=self.directories[path.parent][0], follow_symlinks=False))
        need(custody._did(os.fstat(fd)) == before == custody._did(named), "held directory changed")
        return fd

    def open_file(self, path, maximum, *, expected=None, root_owned=False):
        self.tick()
        path = checked(Path(path))
        parent = self.directory(path.parent)
        return self._open_file_at(path, maximum, parent, expected=expected, root_owned=root_owned)

    def _open_file_at(self, path, maximum, parent, *, expected=None, root_owned=False):
        """Open one child of this observation's already held directory.

        Corpus traversal supplies a real scandir basename and this exact FD.
        Do not rescan every path ancestor for every child: the held directory
        FD/name checks and final ancestor/roster checks remain authoritative.
        """
        self.tick()
        path = Path(path)
        reject_credential_names(path)
        known_parent = self.directories.get(path.parent)
        need(path.is_absolute() and known_parent is not None
             and type(parent) is int and known_parent[0] == parent,
             "child file requires this observation's exact held parent")
        need(self.directory(path.parent) == parent, "held child parent changed")
        fd = self._own(os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent))
        info = os.fstat(fd)
        stamp = custody._stamp(info)
        need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
             and 0 <= info.st_size <= maximum, "plain bounded single-link file required")
        if root_owned:
            need(info.st_uid == os.geteuid(), "actual current policy metadata/source owner required")
        need(stamp == custody._stamp(os.stat(path.name, dir_fd=parent, follow_symlinks=False)), "file FD/name changed")
        if expected is not None:
            _pin(expected, maximum)
            need(str(path) == expected["path"] and info.st_size == expected["bytes"], "raw pin path/length changed")
        self.files.append((path, fd, stamp, expected))
        self.file_parents[fd] = parent
        return fd, stamp

    def bytes(self, fd, size):
        self.tick()
        os.lseek(fd, 0, os.SEEK_SET)
        chunks, remaining = [], size
        while remaining:
            self.tick()
            raw = os.read(fd, min(65536, remaining))
            need(bool(raw), "early EOF")
            chunks.append(raw)
            remaining -= len(raw)
        need(os.read(fd, 1) == b"", "file exceeded actual bound")
        return b"".join(chunks)

    def read(self, pin, maximum, *, root_owned=False):
        fd, before = self.open_file(Path(pin["path"]), maximum, expected=pin, root_owned=root_owned)
        raw = self.bytes(fd, pin["bytes"])
        need(_sha(raw) == pin["sha256"] and custody._stamp(os.fstat(fd)) == before,
             "raw pin/FD changed while reading")
        self.blobs.append((fd, before, dict(pin)))
        return raw

    def digest(self, fd, size):
        self.tick()
        os.lseek(fd, 0, os.SEEK_SET)
        digest, remaining = hashlib.sha256(), size
        while remaining:
            self.tick()
            raw = os.read(fd, min(65536, remaining))
            need(bool(raw), "early payload EOF")
            digest.update(raw)
            remaining -= len(raw)
        need(os.read(fd, 1) == b"", "payload exceeded bound")
        return digest.hexdigest()

    def check(self):
        self.tick()
        need(lifetime() == self.now and self._mounts() == self.mounts, "current observer boot/mount changed")
        current = os.stat("/proc/self/ns/mnt")
        need((current.st_dev, current.st_ino) == self.namespace, "current mount namespace changed")
        for path, (fd, before) in self.directories.items():
            self.tick()
            named = (path.lstat() if path == Path("/") else
                     os.stat(path.name, dir_fd=self.directories[path.parent][0], follow_symlinks=False))
            need(custody._did(os.fstat(fd)) == before == custody._did(named), "held directory changed")
        for path, fd, before, _ in self.files:
            self.tick()
            named = os.stat(path.name, dir_fd=self.file_parents[fd], follow_symlinks=False)
            need(custody._stamp(os.fstat(fd)) == before == custody._stamp(named), "held file/owner/length changed")
        for fd, before, pin in self.blobs:
            self.tick()
            need(_sha(self.bytes(fd, pin["bytes"])) == pin["sha256"]
                 and custody._stamp(os.fstat(fd)) == before, "held metadata/source changed")
        for tree in self.trees:
            tree.check()

    def close(self, primary=None):
        first = None
        for fd in reversed(self.fds):
            try:
                os.close(fd)
            except OSError as error:
                first = first or error
        self.fds.clear()
        if first is not None:
            if primary is None:
                raise first
            if hasattr(primary, "add_note"):
                primary.add_note("unattributed observation close also failed")


class _Tree:
    """Observed corpus only.  Its UID/dev/inode is never the old owner's proof."""
    def __init__(self, held, target, reserve):
        self.held, self.target = held, checked(target)
        self.parent = held.directory(self.target.parent)
        self.root = held.directory(self.target)
        root_info = os.fstat(self.root)
        self.identity = custody._did(root_info)
        self.entries, self.payload, self.files = [], 0, {}
        self._walk(self.target, self.root, PurePosixPath(), root_info.st_dev, set(), reserve)
        need(self.payload <= reserve, "observed corpus exceeds original reservation; no reduced charge")
        self.digest = _sha(custody._encode(self.entries))
        self.expected = {item["name"]: item["stat"] for item in self.entries}
        held.trees.append(self)

    def check(self):
        """Detect new/deleted entries too, not only replaced already-open FDs."""
        seen = set()
        def walk(path, fd, relative):
            self.held.tick()
            name = relative.as_posix()
            need(name in self.expected and custody._stamp(os.fstat(fd)) == self.expected[name],
                 "corpus directory content/metadata changed")
            seen.add(name)
            with os.scandir(fd) as scan:
                names = [item.name for item in scan]
            need(len(names) + len(seen) <= MAX_ENTRIES, "changed corpus roster exceeded bound")
            for child_name in sorted(names):
                self.held.tick()
                child, rel = path / child_name, relative / child_name
                key = rel.as_posix()
                info = os.stat(child_name, dir_fd=fd, follow_symlinks=False)
                need(key in self.expected and custody._stamp(info) == self.expected[key],
                     "corpus roster/file metadata changed")
                if stat.S_ISDIR(info.st_mode):
                    walk(child, self.held.directory(child), rel)
                else:
                    seen.add(key)
            need(custody._stamp(os.fstat(fd)) == self.expected[name], "corpus directory changed during recheck")
        walk(self.target, self.root, PurePosixPath())
        need(seen == set(self.expected), "corpus entry disappeared")

    def _walk(self, path, fd, relative, device, seen, reserve):
        held = self.held
        held.tick()
        before = custody._stamp(os.fstat(fd))
        need(before[0] == device and stat.S_ISDIR(before[2]), "cross-device/non-directory corpus")
        need(tuple(before[:2]) not in seen, "corpus directory alias")
        seen.add(tuple(before[:2]))
        self.entries.append({"name": relative.as_posix(), "stat": before})
        need(len(self.entries) <= MAX_ENTRIES, "bounded corpus roster required")
        with os.scandir(fd) as scan:
            names = [item.name for item in scan]
        need(len(names) + len(self.entries) <= MAX_ENTRIES, "bounded corpus entries required")
        for name in sorted(names):
            held.tick()
            current, rel = path / name, relative / name
            reject_credential_names(current)
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            need(info.st_dev == device, "cross-device corpus entry")
            if stat.S_ISDIR(info.st_mode):
                child = held.directory(current)
                need(custody._stamp(os.fstat(child)) == custody._stamp(info), "corpus directory open race")
                self._walk(current, child, rel, device, seen, reserve)
            else:
                need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "corpus links/special entries forbidden")
                child, stamp = held._open_file_at(current, reserve, fd)
                need(stamp == custody._stamp(info), "corpus file open race")
                need(tuple(stamp[:2]) not in seen, "corpus file alias")
                seen.add(tuple(stamp[:2]))
                self.payload += stamp[6]
                need(self.payload <= reserve, "corpus exceeds original reservation")
                self.entries.append({"name": rel.as_posix(), "stat": stamp})
                need(len(self.entries) <= MAX_ENTRIES, "bounded complete corpus roster required")
                self.files[rel.as_posix()] = (child, stamp)
        need(custody._stamp(os.fstat(fd)) == before, "corpus directory mutated during observation")


def _member(name):
    need(type(name) is str and 0 < len(name) <= 512 and "\\" not in name and ":" not in name,
         "bounded plain archive member required")
    path = PurePosixPath(name)
    need(name not in (".", "..") and not path.is_absolute() and path.as_posix() == name
         and all(part not in (".", "..") for part in path.parts), "archive member escapes or normalizes")
    checked(Path(name))  # Credential-name guard, no member extraction.
    return name


def _zip_metadata_preflight(held, fd, size):
    """Bound central metadata before ZipFile allocates its entry objects.

    Our archives are smaller than512MiB and contain <=4097 entries. ZIP64,
    split disks, prepended stubs and nonstandard trailing central records are
    explicitly unsupported by this first evidence format, rather than guessed.
    """
    held.tick()
    need(size >= 22, "ZIP end record missing")
    tail_size = min(size, 65557)
    tail = os.pread(fd, tail_size, size - tail_size)
    need(len(tail) == tail_size, "ZIP tail read truncated")
    offset = tail.rfind(b"PK\x05\x06")
    need(offset >= 0 and offset + 22 <= len(tail), "ZIP end record missing")
    end = struct.unpack_from("<4s4H2LH", tail, offset)
    _, disk, central_disk, disk_entries, entries, central_size, central_offset, comment = end
    need(disk == central_disk == 0 and disk_entries == entries and 0 < entries <= MAX_ENTRIES + 1
         and entries != 0xffff and central_size != 0xffffffff and central_offset != 0xffffffff
         and 0 < central_size <= MAX_CENTRAL_DIRECTORY and comment <= 512
         and offset + 22 + comment == len(tail)
         and central_offset + central_size == size - tail_size + offset,
         "bounded single-disk non-ZIP64 central directory required")
    central = os.pread(fd, central_size, central_offset)
    need(len(central) == central_size, "ZIP central directory short read")
    position = 0
    for _ in range(entries):
        held.tick()
        need(position + 46 <= len(central), "ZIP central header truncated")
        fields = struct.unpack_from("<4s6H3L5H2L", central, position)
        name, extra, member_comment = fields[10:13]
        need(fields[0] == b"PK\x01\x02" and fields[13] == 0 and fields[2] <= 20 and fields[4] in (0, 8)
             and 0 < name <= 512 and extra <= 4096 and member_comment <= 512
             and fields[8] != 0xffffffff and fields[9] != 0xffffffff
             and 0 <= fields[16] < central_offset,
             "unsupported/big ZIP central entry")
        next_position = position + 46 + name + extra + member_comment
        need(next_position <= len(central), "ZIP central entry overrun")
        extra_bytes = central[position + 46 + name:position + 46 + name + extra]
        cursor = 0
        while cursor < len(extra_bytes):
            need(cursor + 4 <= len(extra_bytes), "ZIP extra field truncated")
            kind, length = struct.unpack_from("<2H", extra_bytes, cursor)
            need(kind != 1 and cursor + 4 + length <= len(extra_bytes), "ZIP64/invalid extra field unsupported")
            cursor += 4 + length
        position = next_position
    need(position == len(central), "unparsed ZIP central metadata")
    held.tick()


def _recovery(held, selected, row, tree):
    pin = selected["research_archive"]
    archive, before = held.open_file(Path(pin["path"]), MAX_ARCHIVE, expected=pin)
    need(held.digest(archive, pin["bytes"]) == pin["sha256"], "actual research archive raw pin changed")
    need(not (Path(pin["path"]).is_relative_to(tree.target) or tree.target.is_relative_to(Path(pin["path"]))),
         "recovery archive must be independently preserved")
    _zip_metadata_preflight(held, archive, pin["bytes"])
    with os.fdopen(os.dup(archive), "rb") as stream, zipfile.ZipFile(stream) as archive_zip:
        entries, total = {}, 0
        for info in archive_zip.infolist():
            held.tick()
            name = _member(info.filename)
            unix_kind = stat.S_IFMT(info.external_attr >> 16)
            need(name not in entries and not info.is_dir() and not info.flag_bits & 1
                 and info.create_system in (0, 3)
                 and ((info.create_system == 3 and unix_kind in (0, stat.S_IFREG))
                      or (info.create_system == 0 and unix_kind == 0))
                 and not (info.external_attr & 0x10), "plain unique unencrypted ZIP members required")
            total += info.file_size
            need(len(entries) < MAX_ENTRIES + 1 and total <= row["reserved_bytes"] + MAX_MANIFEST,
                 "bounded recovery archive contents required")
            entries[name] = info
        need("inventory.json" in entries and entries["inventory.json"].file_size <= MAX_MANIFEST,
             "independent research inventory required")
        manifest = _decode(archive_zip.read(entries["inventory.json"]))
        _fields(manifest, ("schema", "original_catalog_pin", "original_marker_pin", "output", "files"))
        need(manifest["schema"] == RECOVERY_SCHEMA and manifest["original_catalog_pin"] == selected["catalog"]
             and manifest["original_marker_pin"] == selected["marker"] and manifest["output"] == str(tree.target),
             "archive must bind original row/marker/target raw bytes")
        files = manifest["files"]
        need(type(files) is list and len(files) == len(tree.files), "every actual corpus file must be recovered")
        listed = set()
        for item in files:
            held.tick()
            _fields(item, ("relative", "member", "bytes", "sha256"))
            name = _member(item["member"])
            relative = _member(item["relative"])
            need(name == "corpus/" + relative and name not in listed and relative in tree.files and name in entries,
                 "archive/corpus member mismatch")
            fd, stamp = tree.files[relative]
            need(type(item["bytes"]) is int and item["bytes"] == entries[name].file_size == stamp[6],
                 "recovery member length mismatch")
            digest, count = hashlib.sha256(), 0
            with archive_zip.open(entries[name]) as member:
                while True:
                    held.tick()
                    raw = member.read(65536)
                    if not raw:
                        break
                    digest.update(raw)
                    count += len(raw)
                    need(count <= stamp[6], "recovery member exceeded original length")
            actual = held.digest(fd, stamp[6])
            need(count == stamp[6] and actual == digest.hexdigest() == item["sha256"],
                 "actual corpus and actual recovered archive bytes differ")
            listed.add(name)
        need(set(entries) == listed | {"inventory.json"}, "unlisted recovery members forbidden")
    need(custody._stamp(os.fstat(archive)) == before, "recovery archive changed during verification")
    return {"archive": copy.deepcopy(pin), "files": len(tree.files), "current_content_readback_verified": True,
            "original_custody_restored": False, "retirement_authority": False}


def _rows(catalog, selected, *, read_only_budget=False):
    import managed_outputs as outputs
    entries = list(catalog.iterdir())
    need(len(entries) <= outputs.MAX_RUNS + 1, "original catalog bound exceeded")
    rows = []
    for path in entries:
        if path.name == "catalog.lock":
            continue
        row, digest, _ = read_record_snapshot(path, schema=outputs.SCHEMA)
        need(type(row.get("output")) is str and Path(row["output"]).is_absolute(), "absolute original output required")
        target = checked(Path(row["output"]))
        key = _sha(os.path.normcase(str(target)).encode())
        pending = row.get("deferred") is True and row.get("identity") is None
        need(path.name == key + ".json" and row.get("key") == key
             and row.get("state") in {"active", "retained", "interrupted_retained"}
             and not (pending and row["state"] != "active" and target.exists()), "original catalog schema/key/state changed")
        need(type(row.get("reserved_bytes")) is int and type(row.get("catalog_limit_bytes")) is int
             and 0 < row["reserved_bytes"] < 2**63 and 0 < row["catalog_limit_bytes"] < 2**63,
             "every current/preserved row requires its positive original reservation")
        if key not in selected:
            need(pending or row.get("identity") == outputs.identity(target), "unselected row identity changed")
            # A producer atomically replaces marker and row in two separate
            # writes.  Only the authority-free outer sampler may ignore that
            # active-marker window, just like the unchanged default sampler.
            # Identity/reservation/size remain checked and no row is accepted
            # for execution or finalization by this read-only branch.
            need((read_only_budget and row["state"] == "active") or row.get("deferred") is True
                 or outputs._read(target / ".rovezero-output.json") == row,
                 "unselected row marker changed")
        else:
            item = selected[key]
            expected = item[1] if type(item) is tuple else None
            need((row == expected if expected is not None else
                  digest == item["original_catalog_pin"]["sha256"]), "selected original catalog raw changed")
        rows.append((path, row, target))
    need(set(selected) <= {row["key"] for _, row, _ in rows}, "a selected original row disappeared")
    return rows


def _inventory(held, catalog, selection, *, read_only_budget=False):
    import managed_outputs as outputs
    need(type(selection) is UnattributedSelection, "exact typed configuration required; dicts grant no authority")
    need(selection._configuration_digest == _sha(custody._encode({"rows": selection.rows,
         "sources": selection.sources, "tips": selection.tips, "maximum_seconds": selection.maximum_seconds})),
         "selection inputs changed; no implicit configuration refresh")
    need(catalog == checked(outputs.canonical_catalog()), "same original canonical catalog required")
    paths = _source_paths()
    need(set(paths) == set(SOURCE_ROLES), "current source7 closure changed")
    source_bytes = 0
    for role, path in paths.items():
        need(str(path) == selection.sources[role]["path"], "actual live source7 path differs")
        source_bytes += len(held.read(selection.sources[role], transitions.SOURCE_RAW_LIMIT, root_owned=True))
        need(source_bytes <= 256 << 10, "bounded complete current source7 closure required")
    selected, details, archive_charge, seen = {}, {}, 0, set()
    for item in selection.rows:
        raw = held.read(item["catalog"], MAX_METADATA, root_owned=True)
        row = _decode(raw)
        need(row.get("schema") == outputs.SCHEMA, "original row schema differs")
        target = checked(Path(row["output"]))
        key = _sha(os.path.normcase(str(target)).encode())
        need(Path(row["output"]).is_absolute() and row["output"] == str(target) and row.get("key") == key
             and item["catalog"]["path"] == str(catalog / (key + ".json")) and key not in selected,
             "exact six original catalog rows required")
        need(item["marker"]["path"] == str(target / ".rovezero-output.json")
             and type(row.get("identity")) is list and len(row["identity"]) == 2,
             "only previously bound marker-backed rows belong to this policy")
        marker = held.read(item["marker"], MAX_METADATA)
        need(marker == raw, "original catalog and marker raw bytes differ")
        reserve = row.get("reserved_bytes")
        need(type(reserve) is int and 0 < reserve < 2**63
             and type(row.get("catalog_limit_bytes")) is int
             and 0 < row["catalog_limit_bytes"] < 2**63, "original finite reservation required")
        tree = _Tree(held, target, reserve)
        # A later namespace may report the recorded numbers again.  Neither
        # equality nor inequality authenticates an old owner; both observations
        # remain unattributed.  No device number is an allowlist or rewritten.
        actual = tuple(tree.identity[:2])
        need(actual not in seen, "selected corpus root aliases another selected root")
        seen.add(actual)
        for other in (catalog, custody.registry_path(catalog), *paths.values()):
            need(not (target.is_relative_to(other) or other.is_relative_to(target)), "corpus overlaps control/live source")
        recovery = _recovery(held, item, row, tree)
        archive_charge += item["research_archive"]["bytes"]  # Also count separate recovery bytes; never refund original.
        selected[key] = (Path(item["catalog"]["path"]), row, target)
        details[key] = {"original_catalog_pin": copy.deepcopy(item["catalog"]),
            "original_marker_pin": copy.deepcopy(item["marker"]), "output": str(target),
            "recorded_identity": copy.deepcopy(row["identity"]), "current_observed_identity": list(tree.identity),
            "recorded_numbers_equal_current_observation": row["identity"] == list(tree.identity[:2]),
            "current_snapshot_sha256": tree.digest, "observed_bytes": tree.payload, "charge_bytes": reserve,
            "recovery": recovery, "classification": "unattributed_preserved_corpus",
            "old_owner_reissued": False, "identity_continuity_asserted": False,
            "reservation_refund_allowed": False, "retirement_authority": False, "physical_completion": "unknown_unchanged"}
    rows = _rows(catalog, selected, read_only_budget=read_only_budget)
    for _, row, target in rows:
        for key, (_, _, blocked) in selected.items():
            if row["key"] != key:
                need(not (target.is_relative_to(blocked) or blocked.is_relative_to(target)), "corpus overlaps another catalog output")
                if target.exists():
                    need(tuple(outputs.identity(target)) != tuple(details[key]["current_observed_identity"][:2]),
                         "another catalog row aliases unattributed corpus")
    held.check()
    return rows, details, archive_charge


class CatalogView:
    """Actual same-gate metadata/history view.  It is not the old owner's lease."""
    def __init__(self, token, catalog, selection, guard, held, history, details, archive_charge):
        need(token is _TOKEN, "only live original-gate observation constructs a view")
        self.catalog, self.selection, self.guard, self.held = catalog, selection, guard, held
        self.history, self.details, self.archive_charge = history, details, archive_charge
        self.keys = frozenset(history.keys | set(details))
        self.until, self.closed, self.failed = min(held.until, history.held.until), False, False

    def rows(self):
        return self._checked_rows()

    def check(self):
        self._checked_rows()

    def _checked_rows(self):
        need(not self.closed and not self.failed, "observation closed/failed; preserve original failure")
        try:
            self.held.tick()
            self.guard.assert_held()
            self.held.check()
            self.history.check()  # Existing historical chain/records1..8/MAX8 remain unmodified.
            return _rows(self.catalog, self.details)
        except BaseException:
            self.failed = True
            raise

    def usage(self, rows):
        self.check()
        return self._usage_after_check(rows)

    def checked_usage(self):
        """One full current observation, then its rows and unchanged charge.

        The original catalog gate stays held throughout. This is not a cached
        result, permission renewal, or a way to skip a later cleanup boundary.
        """
        rows = self._checked_rows()
        return rows, self._usage_after_check(rows)

    def _usage_after_check(self, rows):
        charges = {record.key: record.charge_bytes for record in self.history.records}
        need(not set(charges) & set(self.details), "distinct history and unattributed classification required")
        charges.update({key: value["charge_bytes"] for key, value in self.details.items()})
        total = self.history.registry_bytes + self.archive_charge
        # Persisted proposal/selection metadata is conservative additional charge.
        total += len(custody._encode(self.selection_document()))
        for _, row, target in rows:
            if row["key"] in charges:
                total += charges[row["key"]]
            elif row.get("state") == "active":
                # Reserve the original maximum, not a transient small live sample.
                amount = tree_size(target, allow_disappearing=True)
                need(amount <= row["reserved_bytes"], "current producer exceeds reservation")
                total += row["reserved_bytes"]
            else:
                amount = tree_size(target, allow_disappearing=True)
                need(amount <= row["reserved_bytes"], "retained producer exceeds reservation")
                total += amount
        warn_output_threshold(total, scope="unattributed-preservation")
        return total

    def selection_document(self):
        return {"schema": SCHEMA, "policy": POLICY, "rows": copy.deepcopy(self.details),
            "current_source7_pins": copy.deepcopy(self.selection.sources),
            "historical_tip_pins": copy.deepcopy(self.selection.tips),
            "original_rows_unchanged": True, "physical_completion": "unknown_unchanged",
            "old_owner_reissued": False, "retirement_authority": False,
            "product_deadline_extended": False}

    def close(self, primary=None):
        if not self.closed:
            self.closed = True
            try:
                self.history.close(primary)
            finally:
                self.held.close(primary)


def observe_catalog(catalog, selection, guard, *, hard_until):
    """New explicit policy; it never calls/weakens the default `_catalog_rows`."""
    catalog = checked(Path(catalog))
    need(type(selection) is UnattributedSelection, "typed source7 selection required")
    need(Path(guard.path) == catalog / "catalog.lock", "same original catalog gate required")
    guard.assert_held()
    held, historical = _Held(selection.maximum_seconds, hard_until), None
    try:
        rows, details, archives = _inventory(held, catalog, selection)
        historical = custody._Held(min(selection.maximum_seconds, held.until - time.monotonic()))
        history = transitions._history(historical, catalog, rows, selection.tips, guard)
        need(bool(history.details) and all(item["generation"] > 0 for item in history.details.values()),
             "existing registered completed history required; not a new generation")
        view = CatalogView(_TOKEN, catalog, selection, guard, held, history, details, archives)
        view.usage(rows)
        return view
    except BaseException as error:
        try:
            if historical is not None:
                historical.close(error)
        finally:
            held.close(error)
        raise


class FreshOutputView(legacy.CurrentExclusionSession):
    """Original new producer only; every old corpus remains an excluded charge."""
    def __init__(self, token, view, output, source, producer_record):
        need(token is _TOKEN and type(view) is CatalogView, "actual original-gate view required")
        self.inventory = view
        self._hard_until = view.until
        legacy_selection = legacy.PreservationSelection(view.selection.tips,
            {role: view.selection.sources[role] for role in legacy.SOURCE_ROLES},
            root_exclusive_writer=True, maximum_seconds=view.selection.maximum_seconds)
        anchor = legacy.HistoricalAnchor(legacy._TOKEN, view.history)
        super().__init__(legacy._TOKEN, anchor, legacy_selection, view.catalog, view.guard,
                         output, source, producer_record)
        self.keys = view.keys

    def check(self):
        need(not self._closed, "fresh output view closed")
        self.inventory.check()
        self._held.tick()
        need(lifetime() == self.now, "new producer lifetime changed")
        need(custody._did(os.fstat(self._parent_fd)) == self._parent_identity
             == custody._did(checked(self.output.parent).lstat()), "private new-output parent changed")
        need(custody._mounts(self._held, self.output.parent, self.output, self._parent_identity[0]) == self._mounts,
             "new-output current mount changed")
        self._scope(_rows(self.catalog, self.inventory.details))
        current = self.output.lstat() if self.output.exists() else None
        if current is not None:
            need((current.st_dev, current.st_ino) not in
                 {tuple(item["current_observed_identity"][:2]) for item in self.inventory.details.values()},
                 "new output aliases unattributed corpus")

    def rows(self):
        self.check()
        return _rows(self.catalog, self.inventory.details)

    def usage(self, rows):
        self.check()
        self._scope(rows)
        return self.inventory.usage(rows)

    def create_journal(self, record):
        import managed_outputs as outputs
        self.bind_producer(record)
        self.check()
        need(record.get("identity") == outputs.identity(self.output)
             and outputs._read(self.catalog / (record["key"] + ".json")) == record,
             "actual bound new output required before journal creation")
        info = os.fstat(self._held.directory(self.output))
        need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
             and stat.S_IMODE(info.st_mode) & 0o077 == 0, "private actual new output required")
        return legacy._EvidenceJournal(legacy._TOKEN, self.output, record, self.now, self.inventory.held.tick)

    def seal(self, record, journal):
        need(not self._sealed and type(journal) is legacy._EvidenceJournal, "original new journal required")
        self.bind_producer(record)
        self.check()
        import managed_outputs as outputs
        need(record.get("identity") == outputs.identity(self.output)
             and outputs._read(self.catalog / (record["key"] + ".json")) == record,
             "same bound new original producer required before publication")
        document = self.inventory.selection_document()
        document.update(new_output_key=record["key"], new_output=str(self.output),
                        new_producer_token=record["token"], current_lifetime=dict(self.now),
                        original_until_monotonic=self.inventory.until,
                        old_source6_proof_reinterpreted=False)
        pin = journal.append(record, document, tick=self.inventory.held.tick)
        self.check()
        self._sealed = True
        return pin

    def close(self, primary=None):
        if not self._closed:
            self._closed = True
            self.inventory.close(primary)


def observe_fresh_output(catalog, selection, guard, *, output, source, producer_record=None):
    """Each call reobserves.  Observation refresh never extends a product clock."""
    output, source = checked(Path(output)), checked(Path(source))
    need(producer_record is not None or not output.exists(), "fresh output only; no historical adoption")
    # The caller must supply its original work/cleanup deadline explicitly.
    need(type(selection) is UnattributedSelection and selection._bound_deadline is not None,
         "original caller hard deadline was not bound")
    view = observe_catalog(catalog, selection, guard, hard_until=selection.original_hard_until)
    try:
        result = FreshOutputView(_TOKEN, view, output, source, producer_record)
        result.check()
        return result
    except BaseException as error:
        view.close(error)
        raise


def bind_original_deadline(selection, original_hard_until):
    """Only initial live caller registration may set this immutable clock."""
    need(type(selection) is UnattributedSelection, "exact typed selection required")
    need(selection._bound_deadline is None, "original deadline cannot be renewed")
    need(type(original_hard_until) in (int, float) and math.isfinite(original_hard_until)
         and time.monotonic() < original_hard_until <= time.monotonic() + 3630,
         "finite original work+cleanup clock required")
    object.__setattr__(selection, "_bound_deadline", original_hard_until)
    return selection


def _mount_table():
    with open("/proc/self/mountinfo", "rb") as stream:
        raw = stream.read((1 << 20) + 1)
    need(0 < len(raw) <= 1 << 20, "bounded cleanup-scope mount table required")
    table = {}
    def unescape(value):
        return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)
    for line in raw.decode("utf-8", errors="strict").splitlines():
        fields = line.split()
        need(len(fields) >= 10 and "-" in fields and fields[0].isdigit(), "kernel mount record malformed")
        mount_id = int(fields[0])
        need(mount_id not in table and len(table) < 16384, "bounded unique kernel mount IDs required")
        table[mount_id] = (fields[2], Path(unescape(fields[3])), Path(unescape(fields[4])))
    return raw, table


def _physical_scope(path, table):
    """Current-namespace DENY coordinate only, never old identity continuity."""
    path = checked(Path(path))
    existing = path
    while not existing.exists():
        need(existing.parent != existing, "existing cleanup-scope parent required")
        existing = existing.parent
    fd = os.open(existing, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = custody._stamp(os.fstat(fd))
        with open("/proc/self/fdinfo/" + str(fd), "rb") as stream:
            raw = stream.read(4097)
        need(len(raw) <= 4096, "bounded current mount FD observation required")
        values = [line.split(b":", 1)[1].strip() for line in raw.splitlines() if line.startswith(b"mnt_id:")]
        need(len(values) == 1 and values[0].isdigit(), "actual current FD mount ID required")
        mount_id = int(values[0])
        need(mount_id in table, "current mount disappeared")
        device, root, point = table[mount_id]
        need(path.is_relative_to(point) and custody._stamp(os.fstat(fd)) == before
             == custody._stamp(checked(existing).lstat()), "cleanup-scope FD/name/mount changed")
        return device, root / path.relative_to(point)
    finally:
        os.close(fd)


def assert_manager_preservation_scope(root, selection):
    """Deny before acquire/prune, and again before observation/final cleanup.

    Raw input paths grant no ownership; they can only make this stricter.  The
    actual selected row/marker/archive joins are still verified under a view.
    Regenerable live source7 copies have separate source custody and are not
    research archives.  They never make a selected corpus into build cache.
    """
    need(type(selection) is UnattributedSelection, "exact typed preservation selection required")
    need(sys.platform == "linux", "Linux cleanup-scope observation required")
    root = checked(Path(root))
    protected = []
    for item in selection.rows:
        protected.extend(Path(item[name]["path"]) for name in ("catalog", "marker", "research_archive"))
        protected.append(Path(item["marker"]["path"]).parent)
    protected.extend(Path(pin["path"]) for pin in selection.tips)
    before, table = _mount_table()
    root_device, root_coordinate = _physical_scope(root, table)
    for value in protected:
        target = checked(value)
        need(not (target.is_relative_to(root) or root.is_relative_to(target)),
             "preserved corpus/archive/control input overlaps manager cleanup root")
        device, coordinate = _physical_scope(target, table)
        need(not (device == root_device and (coordinate.is_relative_to(root_coordinate)
                 or root_coordinate.is_relative_to(coordinate))),
             "current bind/namespace alias overlaps manager cleanup root")
    after, _ = _mount_table()
    need(before == after, "cleanup-scope mount observation changed")


class ReadOnlyBudgetObserver:
    """No gate/admission/finish authority; an actual original manager is retained.

    Each sample independently observes the exact raw source/corpus/archives.
    Historical roots/registry are revalidated against the actual pre-child
    held-gate baseline.  It does NOT acquire a busy producer's catalog gate.
    """
    def __init__(self, token, manager, selection, baseline):
        need(token is _TOKEN and type(baseline) is CatalogView, "actual original-gate baseline required")
        self.manager, self.selection, self.catalog = manager, selection, baseline.catalog
        self.baseline = baseline.selection_document()
        self.now = lifetime()
        self.history_charges = {record.key: record.charge_bytes for record in baseline.history.records}
        self.registry = copy.deepcopy(baseline.history.registry)
        self.registry_bytes = baseline.history.registry_bytes
        self.history_raw_pins = [copy.deepcopy(pin) for _, _, _, pin in baseline.history.held.blobs]
        self.history_observations = [{"output": str(item.target), "identity": list(item.identity),
            "parent_identity": list(item.parent_identity), "snapshot": copy.deepcopy(item.snapshot)}
            for item in baseline.history.observed]
        self.closed, self.failed = False, False

    def sample(self):
        need(not self.closed and not self.failed, "read-only observer closed/failed")
        held = old = None
        try:
            self.manager._assert_owner()  # Same actual in-process manager, not a JSON owner reconstruction.
            assert_manager_preservation_scope(self.manager.root, self.selection)
            need(lifetime() == self.now, "original manager observer lifetime changed")
            held = _Held(self.selection.maximum_seconds, self.selection.original_hard_until)
            rows, details, archives = _inventory(held, self.catalog, self.selection, read_only_budget=True)
            need(details == self.baseline["rows"], "preserved corpus observation/content changed")
            old = custody._Held(min(self.selection.maximum_seconds, held.until - time.monotonic()))
            registry, size, _ = transitions._layout(old, self.catalog)
            need(registry == self.registry and size == self.registry_bytes, "historical registry changed")
            for pin in self.history_raw_pins:
                old.read(pin, Path(pin["path"]), max(custody.SNAPSHOT_LIMIT, transitions.SOURCES_LIMIT))
            for item in self.history_observations:
                observed = custody._Observed(old, Path(item["output"]))
                need(list(observed.identity) == item["identity"]
                     and list(observed.parent_identity) == item["parent_identity"]
                     and observed.snapshot == item["snapshot"], "preserved historical subtree changed")
            charges = dict(self.history_charges)
            charges.update({key: item["charge_bytes"] for key, item in details.items()})
            total = self.registry_bytes + archives + len(custody._encode(self.baseline))
            for _, row, target in rows:
                amount = charges.get(row["key"])
                if amount is None:
                    actual = tree_size(target, allow_disappearing=True)
                    need(actual <= row["reserved_bytes"], "sampled current output exceeds reservation")
                    amount = row["reserved_bytes"] if row["state"] == "active" else actual
                total += amount
            warn_output_threshold(total, scope="unattributed-read-only")
            held.check()
            self.manager._assert_owner()
            return total
        except BaseException:
            self.failed = True
            raise
        finally:
            primary = sys.exc_info()[1]
            try:
                if old is not None:
                    old.close(primary)
            finally:
                if held is not None:
                    held.close(primary)

    def close(self):
        self.closed = True  # It owns no lease or persisted owner to retire.


def create_budget_observer(manager, selection):
    """Called before any child.  Never construct a manager from a saved owner."""
    from managed_storage import ManagedBuild
    import managed_outputs as outputs
    need(type(manager) is ManagedBuild and manager.acquired is True and manager.finished is False,
         "actual acquired original ManagedBuild object required")
    manager._assert_owner()
    assert_manager_preservation_scope(manager.root, selection)
    need(type(selection) is UnattributedSelection and selection._bound_deadline is not None,
         "typed original-deadline selection required")
    catalog = checked(outputs.canonical_catalog())
    with custody.original_custody_gate(catalog) as guard:
        view = observe_catalog(catalog, selection, guard, hard_until=selection.original_hard_until)
        try:
            result = ReadOnlyBudgetObserver(_TOKEN, manager, selection, view)
            result.sample()
            return result
        finally:
            view.close(sys.exc_info()[1])


def finalize_original_manager(manager, selection):
    """Post-tree-gone original supervisor only.  Old selected rows never change."""
    from managed_storage import ManagedBuild
    import managed_outputs as outputs
    need(type(manager) is ManagedBuild and manager.acquired is True and manager.finished is False,
         "actual original acquired manager required")
    manager._assert_owner()
    assert_manager_preservation_scope(manager.root, selection)
    # Caller has already emitted its original closure.json after run_group's
    # tree-gone witness.  Verify that exact original witness, not a supplied bool.
    from managed_storage import read_record
    closure = read_record(manager.root / "closure.json")
    need(closure.get("owner") == manager._identity() and closure.get("tree_gone") is True,
         "original manager closure missing")
    catalog = checked(outputs.canonical_catalog())
    with custody.original_custody_gate(catalog) as guard:
        view = observe_catalog(catalog, selection, guard, hard_until=selection.original_hard_until)
        try:
            rows = view.rows()
            view.usage(rows)
            expected = {"root": str(manager.root), "owner": manager._identity(),
                        "scope": "inherited_outer_process_group_or_job"}
            for path, row, target in rows:
                if row["key"] in view.keys:
                    continue  # No old owner, identity, state, reservation or physical fact is rewritten.
                if row.get("outer_supervisor") != expected or row.get("state") != "active":
                    continue
                need(row.get("identity") == outputs.identity(target)
                     and outputs._read(target / ".rovezero-output.json") == row,
                     "new original supervised output identity changed")
                guard.assert_held()
                manager._assert_owner()
                row.update(state="retained", tree_gone=True, closure="original_outer_supervisor",
                           physical_completion="not_inferred_from_output_lifecycle")
                write_record(target / ".rovezero-output.json", row)
                write_record(path, row)
            view.usage(view.rows())
        finally:
            view.close(sys.exc_info()[1])


BUILD_RECOVERY_SCHEMA = "rovezero.unattributed-build-cache-recovery.v1"


def _assert_recovery_source_scope(root, selection):
    """Recovery may delete its build cache; every live source7 must be outside."""
    assert_manager_preservation_scope(root, selection)
    before, table = _mount_table()
    device, coordinate = _physical_scope(root, table)
    for pin in selection.sources.values():
        path = checked(Path(pin["path"]))
        need(not (path.is_relative_to(root) or root.is_relative_to(path)),
             "current recovery source7 overlaps build cleanup root")
        source_device, source_coordinate = _physical_scope(path, table)
        need(not (source_device == device and
                  (source_coordinate.is_relative_to(coordinate) or coordinate.is_relative_to(source_coordinate))),
             "current recovery source7 aliases build cleanup root")
    after, _ = _mount_table()
    need(before == after, "recovery source cleanup-scope mount changed")


class _RecoveryCatalogHold:
    """One read-only catalog exclusion during ORIGINAL cache recovery only.

    Constructed after recover_build's original dead/root/closure/slot checks.
    It never acquires a ManagedBuild, issues a current execution owner, closes
    an old output row or interprets the old owner JSON as execution authority.
    """
    def __init__(self, token, root, owner_sha256, selection, build_guard, verify_recovery):
        need(token is _TOKEN and type(selection) is UnattributedSelection,
             "actual recovery call and exact typed preservation selection required")
        root = checked(Path(root))
        need(type(build_guard) is CatalogGuard and Path(build_guard.path) == root / "catalog.lock"
             and callable(verify_recovery), "actual original build guard/verifier required")
        need(selection._bound_deadline is not None, "original finite maintenance clock required")
        self.root, self.selection = root, selection
        self._build_guard, self._verify = build_guard, verify_recovery
        self._held = self._view = self._catalog_context = self._catalog_guard = None
        self._closed, self._failed = False, False
        try:
            build_guard.assert_held()
            verify_recovery()  # Existing actual root/slot/raw-owner proof, not a saved ready flag.
            _assert_recovery_source_scope(root, selection)
            self._held = _Held(selection.maximum_seconds, selection.original_hard_until)
            owner_path = root / "owner.json"
            owner, digest, stamp = read_record_snapshot(owner_path)
            need(digest == owner_sha256, "original recovery owner digest changed")
            owner_pin = {"path": str(owner_path), "bytes": stamp[4], "sha256": digest}
            owner_raw = self._held.read(owner_pin, MAX_METADATA, root_owned=True)
            need(_decode(owner_raw) == owner and read_record_snapshot(owner_path) == (owner, digest, stamp),
                 "actual original owner raw/FD changed")
            self._original_owner = owner  # Comparison data only. Never loaded as a current owner.
            self._original_owner_pin = owner_pin
            import managed_outputs as outputs
            self._catalog_context = custody.original_custody_gate(outputs.canonical_catalog())
            self._catalog_guard = self._catalog_context.__enter__()
            self._view = observe_catalog(outputs.canonical_catalog(), selection, self._catalog_guard,
                                         hard_until=self._held.until)
            self.check()
        except BaseException as error:
            self.close(error)
            raise

    def check(self):
        need(not self._closed and not self._failed, "recovery observation closed/failed; no refresh")
        try:
            self._build_guard.assert_held()
            self._verify()
            self._held.check()
            _assert_recovery_source_scope(self.root, self.selection)
            # One complete observation per boundary; do not recursively repeat
            # the same held-file walk through check -> rows -> usage.
            # The original <=30s observation is never replaced or extended.
            rows, _ = self._view.checked_usage()
            expected = {"root": str(self.root), "owner": self._original_owner,
                        "scope": "inherited_outer_process_group_or_job"}
            before, table = _mount_table()
            device, coordinate = _physical_scope(self.root, table)
            for _, row, target in rows:
                # Selected/unattributed rows receive NO exception to this rule.
                need(not (row.get("state") == "active" and row.get("outer_supervisor") == expected),
                     "old-owner active supervised output prevents cache recovery")
                need(not (target.is_relative_to(self.root) or self.root.is_relative_to(target)),
                     "registered evidence overlaps recovery cleanup root")
                output_device, output_coordinate = _physical_scope(target, table)
                need(not (output_device == device and (output_coordinate.is_relative_to(coordinate)
                         or coordinate.is_relative_to(output_coordinate))),
                     "registered evidence aliases recovery cleanup root")
            after, _ = _mount_table()
            need(before == after, "recovery output-scope mount changed")
            self._held.check()
            self._verify()
            self._catalog_guard.assert_held()
            self._build_guard.assert_held()
        except BaseException:
            self._failed = True
            raise

    def audit_fields(self):
        """Small metadata, never a saved owner/lease/admission capability."""
        need(not self._closed and not self._failed, "recovery observation closed/failed")
        self._held.tick()
        self._build_guard.assert_held()
        self._catalog_guard.assert_held()
        return {"schema": BUILD_RECOVERY_SCHEMA, "scope": "original_cache_recovery_only",
                "policy": POLICY, "selected_unattributed_rows": len(self._view.details),
                "output_rows_modified": False, "old_output_finalization_performed": False,
                "old_owner_reissued": False, "source6_historical_authority_reissued": False,
                "physical_completion": "unknown_unchanged", "observation_refresh": False}

    def close(self, primary=None):
        if self._closed:
            return
        self._closed = True
        first = None
        for name in ("_view", "_held"):
            value = getattr(self, name, None)
            setattr(self, name, None)
            if value is not None:
                try:
                    value.close(primary)
                except BaseException as error:
                    first = first or error
        context, self._catalog_context = self._catalog_context, None
        if context is not None:
            try:
                if self._catalog_guard is not None:
                    context.__exit__(type(primary) if primary is not None else None, primary,
                                     primary.__traceback__ if primary is not None else None)
            except BaseException as error:
                first = first or error
        self._catalog_guard = None
        if first is not None:
            if primary is None:
                raise first
            if hasattr(primary, "add_note"):
                primary.add_note("read-only recovery observation close also failed")


@contextmanager
def recovery_catalog_exclusion(root, owner_sha256, selection, build_guard, verify_recovery):
    """Explicit typed selection for original recover_build's existing cleanup.

    Saved JSON is not accepted as a live guard/verifier/owner.  All original
    termination checks must already have succeeded in recover_build.  There is
    no generation update, output finalizer, observation refresh or new run.
    """
    hold = _RecoveryCatalogHold(_TOKEN, root, owner_sha256, selection, build_guard, verify_recovery)
    try:
        yield hold
    finally:
        hold.close(sys.exc_info()[1])
