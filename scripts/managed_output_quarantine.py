"""Opt-in current preservation custody; never repairs an unbound V1 lease.

The original catalog row is immutable. Only explicitly pinned final receipts may
classify it, under its original catalog gate and the current exclusive-writer
contract. Linux metadata/descriptor observations are not original ownership or
provider completion. Default V1 callers do not import or consume this module.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time

from managed_storage import StorageError, checked, tree_size, GiB, warn_output_threshold
from storage_guard import CatalogGuard, lifetime, reboot_proves_dead

SCHEMA = "rovezero.interrupted-unbound-custody.v1"
MAX_ENTRIES = 4096
SNAPSHOT_LIMIT = 1 << 20
RECEIPT_LIMIT = 8192
REGISTRY_LIMIT = 16 << 20
MAX_RUNS = 1024
_NAME = re.compile(r"([0-9a-f]{64})-([0-9a-f]{64})\.(original|snapshot|receipt)\.json")


def _need(value, message):
    if not value:
        raise StorageError(message)


def _encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def _decode(raw):
    def unique(items):
        value = {}
        for key, item in items:
            _need(key not in value, "duplicate custody metadata field")
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(StorageError("nonfinite custody metadata")))


def _close_fd(fd, primary=None):
    try:
        os.close(fd)
    except OSError:
        if primary is None:
            raise
        if hasattr(primary, "add_note"):
            primary.add_note("custody descriptor close also failed")


@contextmanager
def original_custody_gate(catalog):
    """The original gate, with secondary release failure kept behind primary."""
    guard = CatalogGuard(checked(Path(catalog)) / "catalog.lock").acquire()
    try:
        yield guard
    finally:
        primary = sys.exc_info()[1]
        try:
            guard.release()
        except BaseException:
            if primary is None:
                raise
            if hasattr(primary, "add_note"):
                primary.add_note("original custody catalog gate release also failed")


def _stamp(info):
    # Access-time changes caused by metadata/reference reads are not mutations.
    return [info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _did(info):
    return _stamp(info)[:5]


def registry_path(catalog):
    catalog = checked(Path(catalog))
    return catalog.with_name(catalog.name + ".interrupted-unbound-custody-v1")


def _pin(path, raw):
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _pin_shape(value, expected, maximum):
    _need(type(value) is dict and set(value) == {"path", "bytes", "sha256"}
          and value["path"] == str(expected) and type(value["bytes"]) is int
          and 0 < value["bytes"] <= maximum and isinstance(value["sha256"], str)
          and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]), "exact custody raw pin required")


class _Held:
    def __init__(self, seconds):
        _need(sys.platform == "linux" and 0 < seconds <= 30, "Linux custody observation requires a finite <=30s window")
        started = time.monotonic()
        self._initialize(started, started + seconds)

    @classmethod
    def _for_bounded_session(cls, seconds, hard_until):
        """Create an observation bounded by its explicitly supplied caller deadline."""
        _need(sys.platform == "linux" and type(seconds) in (int, float)
              and math.isfinite(seconds) and seconds > 0
              and type(hard_until) in (int, float) and math.isfinite(hard_until),
              "bounded custody observation requires finite positive bounds")
        started = time.monotonic()
        _need(started < hard_until, "original custody observation deadline expired")
        held = cls.__new__(cls)
        held._initialize(started, min(started + seconds, hard_until))
        return held

    def _initialize(self, started, until):
        self.started, self.until, self.fds, self.blobs = started, until, [], []
        self.directories = {}
        self._stage = "observation"

    def mark_stage(self, stage):
        _need(stage in {"directory", "metadata_read", "metadata_recheck", "mount_observation",
                        "tree_snapshot", "observation_check", "registry_inventory", "output_accounting"},
              "invalid custody observation stage")
        self._stage = stage

    def tick(self):
        now = time.monotonic()
        if now >= self.until:
            error = StorageError("original custody observation deadline expired")
            error.custody_deadline_observation = {
                "schema": "rovezero.custody-deadline-observation.v1",
                "stage": self._stage,
                "elapsed_seconds": round(max(0.0, now - self.started), 3),
                "window_seconds": round(max(0.0, self.until - self.started), 3),
            }
            raise error

    def directory(self, path):
        self.mark_stage("directory")
        self.tick()
        path = checked(path)
        if path in self.directories:
            fd = self.directories[path]
            _need(_did(os.fstat(fd)) == _did(path.lstat()), "held custody directory name changed")
            return fd
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.fds.append(fd)  # Own the descriptor before any fallible metadata read.
        self.directories[path] = fd
        return fd

    def read(self, pin, path, maximum):
        self.mark_stage("metadata_read")
        self.tick()
        _pin_shape(pin, path, maximum)
        fd = os.open(checked(path), os.O_RDONLY | os.O_NOFOLLOW)
        self.fds.append(fd)
        before = _stamp(os.fstat(fd))
        regular = stat.S_ISREG(before[2])
        owner_matches = before[3] == os.geteuid()
        single_link = before[5] == 1
        length_matches = before[6] == pin["bytes"]
        if not (regular and owner_matches and single_link and length_matches):
            error = StorageError("custody metadata FD/type/owner/length changed")
            # No path, file contents, raw pin, or ownership override. The
            # caller may add a known public source role before reporting.
            error.custody_metadata_failure_observation = {
                "schema": "rovezero.custody-metadata-failure-observation.v1",
                "stage": "metadata_read",
                "domain": "unknown",
                "source_role": None,
                "regular_file": regular,
                "owner_matches_euid": owner_matches,
                "single_link": single_link,
                "length_matches_pin": length_matches,
            }
            raise error
        raw = self._bytes(fd, before[6])
        _need(_pin(path, raw) == pin and _stamp(os.fstat(fd)) == before
              == _stamp(Path(path).lstat()), "custody metadata raw pin/FD/name changed")
        self.blobs.append((Path(path), fd, before, pin))
        return raw

    def _bytes(self, fd, size):
        result, remaining = [], size
        self.mark_stage("metadata_read")
        while remaining:
            self.tick()
            block = os.read(fd, min(65536, remaining))
            _need(bool(block), "custody metadata early EOF")
            result.append(block)
            remaining -= len(block)
        _need(os.read(fd, 1) == b"", "custody metadata exceeded bound")
        return b"".join(result)

    def check(self):
        self.mark_stage("metadata_recheck")
        for path, fd, before, pin in self.blobs:
            self.tick()
            _need(_stamp(os.fstat(fd)) == before == _stamp(checked(path).lstat()), "held custody metadata identity changed")
            os.lseek(fd, 0, os.SEEK_SET)
            _need(_pin(path, self._bytes(fd, before[6])) == pin
                  and _stamp(os.fstat(fd)) == before, "held custody metadata content changed")

    def close(self, primary=None):
        first = None
        for fd in reversed(self.fds):
            try:
                os.close(fd)
            except OSError as error:
                first = first or error
        self.fds.clear()
        self.directories.clear()
        if first is not None:
            if primary is None:
                raise first
            if hasattr(primary, "add_note"):
                primary.add_note("custody descriptor close also failed")


def _kernel(held, path, maximum):
    held.tick()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        chunks, size = [], 0
        while True:
            held.tick()
            block = os.read(fd, min(65536, maximum + 1 - size))
            if not block:
                break
            chunks.append(block)
            size += len(block)
            _need(size <= maximum, "custody kernel metadata bound exceeded")
        return b"".join(chunks)
    finally:
        _close_fd(fd, sys.exc_info()[1])


def _mark_stage(held, stage):
    """Keep the existing tick() contract for observation implementations."""
    marker = getattr(held, "mark_stage", None)
    if marker is not None:
        marker(stage)


def _mounts(held, parent, target, device):
    _mark_stage(held, "mount_observation")
    held.tick()
    raw = _kernel(held, Path("/proc/self/mountinfo"), SNAPSHOT_LIMIT)
    namespace = os.stat("/proc/self/ns/mnt")  # A named kernel namespace link, never target data.
    rows, origin_lines = [], {}
    def unescape(text):
        return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), text)
    for line in raw.decode("utf-8", "strict").splitlines():
        fields = line.split()
        _need(len(fields) >= 10 and "-" in fields[6:], "unrecognized custody mount metadata")
        # Per-mount options precede the separator; superblock options do not
        # establish whether this particular alias is writable.
        rows.append((fields[0], fields[2], Path(unescape(fields[3])), Path(unescape(fields[4])), fields[5]))
        origin_lines[fields[0]] = line.encode("utf-8")
    def containing(path):
        options = [row for row in rows if path == row[3] or row[3] in path.parents]
        _need(bool(options), "custody mount origin unavailable")
        length = max(len(row[3].parts) for row in options)
        options = [row for row in options if len(row[3].parts) == length]
        _need(len(options) == 1, "stacked custody mount is ambiguous")
        return options[0]
    origin, target_origin = containing(parent), containing(target)
    _need(origin == target_origin and origin[1] == f"{os.major(device)}:{os.minor(device)}", "cross-device or boundary custody mount")
    # Keep only a digest for failure diagnostics. The full mountinfo remains
    # part of the actual comparison and no path is disclosed in that record.
    if isinstance(held, _Held):
        held._last_mount_origin_sha256 = hashlib.sha256(origin_lines[origin[0]]).hexdigest()
    native_parent = origin[2] / parent.relative_to(origin[3])
    readonly_aliases = []
    for row in rows:
        if row[0] == origin[0]:
            continue
        _need(not (row[3] == target or target in row[3].parents), "nested custody mount is not allowed")
        if row[1] == origin[1] and (row[2] == native_parent or row[2] in native_parent.parents
                                     or native_parent in row[2].parents):
            options = row[4].split(",")
            _need("ro" in options and "rw" not in options, "writable or unknown custody mount alias")
            _need(sum(other[3] == row[3] for other in rows) == 1, "stacked readonly custody mount alias")
            _need(not any(row[3] in other[3].parents for other in rows), "nested readonly custody mount alias")
            readonly_aliases.append({"mount_id": row[0], "root": str(row[2]),
                                     "mountpoint": str(row[3]), "mount_options": row[4]})
    return {"namespace": [namespace.st_dev, namespace.st_ino],
            "mountinfo_bytes": len(raw), "mountinfo_sha256": hashlib.sha256(raw).hexdigest(),
            "readonly_aliases": sorted(readonly_aliases, key=lambda row: (row["mountpoint"], row["mount_id"]))}


def _snapshot(held, target, fd):
    _mark_stage(held, "tree_snapshot")
    held.tick()
    root = os.fstat(fd)
    device, seen, entries, regular_bytes = root.st_dev, set(), [], 0
    encoded_bound = 1024  # Conservative fixed header/numeric overhead.
    def add_entry(relative, stamp):
        nonlocal encoded_bound
        item = {"name": relative.as_posix(), "stat": stamp}
        encoded_bound += len(_encode(item)) + 2
        _need(len(entries) < MAX_ENTRIES, "custody snapshot entry bound exceeded")
        _need(encoded_bound <= SNAPSHOT_LIMIT, "custody snapshot metadata bound exceeded")
        entries.append(item)
    def walk(directory, relative):
        nonlocal regular_bytes
        before = _stamp(os.fstat(directory))
        _need(before[0] == device and before[3] == os.geteuid() and stat.S_ISDIR(before[2]), "custody directory device/type/owner changed")
        identity = tuple(before[:2])
        _need(identity not in seen, "multiple custody directory aliases")
        seen.add(identity)
        add_entry(relative, before)
        names = []
        with os.scandir(directory) as children:
            for child in children:
                held.tick()
                names.append(child.name)
                _need(len(names) + len(entries) <= MAX_ENTRIES, "custody snapshot roster bound exceeded")
        for name in sorted(names):
            held.tick()
            rel = relative / name
            checked(target / rel)  # Preserve the credential/link guard before opening metadata.
            info = os.stat(name, dir_fd=directory, follow_symlinks=False)
            _need(info.st_dev == device and info.st_uid == os.geteuid(), "custody cross-device/owner entry")
            if stat.S_ISDIR(info.st_mode):
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                try:
                    _need(_stamp(os.fstat(child)) == _stamp(info), "custody child directory open race")
                    walk(child, rel)
                    _need(_stamp(os.fstat(child)) == _stamp(os.stat(name, dir_fd=directory, follow_symlinks=False)), "custody child directory name changed")
                finally:
                    _close_fd(child, sys.exc_info()[1])
            else:
                _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "custody symlink/hardlink/special entry is forbidden")
                child = os.open(name, os.O_PATH | os.O_NOFOLLOW, dir_fd=directory)
                try:
                    stamp = _stamp(os.fstat(child))
                    _need(stamp == _stamp(info) == _stamp(os.stat(name, dir_fd=directory, follow_symlinks=False)), "custody regular metadata/name changed")
                    identity = tuple(stamp[:2])
                    _need(identity not in seen, "multiple custody file aliases")
                    seen.add(identity)
                    add_entry(rel, stamp)
                    regular_bytes += stamp[6]
                finally:
                    _close_fd(child, sys.exc_info()[1])
        _need(_stamp(os.fstat(directory)) == before, "custody directory changed during snapshot")
    walk(fd, Path())
    result = {"schema": SCHEMA, "kind": "snapshot", "entries": entries,
              "entry_count": len(entries), "regular_bytes": regular_bytes}
    _need(len(_encode(result)) <= SNAPSHOT_LIMIT, "custody snapshot metadata bound exceeded")
    return result


class _Observed:
    def __init__(self, held, target):
        self.held, self.target = held, checked(target)
        self.parent_fd = held.directory(self.target.parent)
        self.fd = held.directory(self.target)
        parent, root = os.fstat(self.parent_fd), os.fstat(self.fd)
        for info in (parent, root):
            _need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
                  and stat.S_IMODE(info.st_mode) & 0o077 == 0, "private current custody parent/target required")
        _need(parent.st_dev == root.st_dev, "cross-device custody root")
        self.parent_identity, self.identity = _did(parent), _did(root)
        self.mounts = _mounts(held, self.target.parent, self.target, root.st_dev)
        self.mount_origin_sha256 = getattr(held, "_last_mount_origin_sha256", None)
        self.snapshot = _snapshot(held, self.target, self.fd)
        self.check()

    def check(self):
        _mark_stage(self.held, "observation_check")
        self.held.tick()
        _need(_did(os.fstat(self.parent_fd)) == _did(checked(self.target.parent).lstat()) == self.parent_identity, "custody parent FD/name changed")
        _need(_did(os.fstat(self.fd)) == _did(os.stat(self.target.name, dir_fd=self.parent_fd, follow_symlinks=False))
              == _did(checked(self.target).lstat()) == self.identity, "custody target FD/name changed")
        current_mounts = _mounts(self.held, self.target.parent, self.target, self.identity[0])
        if current_mounts != self.mounts:
            current_origin = getattr(self.held, "_last_mount_origin_sha256", None)
            error = StorageError("custody namespace/mount observation changed")
            error.custody_mount_change_observation = {
                "schema": "rovezero.custody-mount-change-observation.v1",
                "namespace_changed": self.mounts.get("namespace") != current_mounts.get("namespace"),
                "mountinfo_changed": (self.mounts.get("mountinfo_bytes"), self.mounts.get("mountinfo_sha256"))
                                     != (current_mounts.get("mountinfo_bytes"), current_mounts.get("mountinfo_sha256")),
                "readonly_aliases_changed": self.mounts.get("readonly_aliases") != current_mounts.get("readonly_aliases"),
                "origin_changed": None if self.mount_origin_sha256 is None or current_origin is None
                                  else self.mount_origin_sha256 != current_origin,
                "before_mountinfo_sha256": self.mounts.get("mountinfo_sha256"),
                "after_mountinfo_sha256": current_mounts.get("mountinfo_sha256"),
                "before_origin_sha256": self.mount_origin_sha256,
                "after_origin_sha256": current_origin,
            }
            raise error
        _need(_snapshot(self.held, self.target, self.fd) == self.snapshot, "custody snapshot changed")


def _registry(held, catalog, complete=True):
    _mark_stage(held, "registry_inventory")
    held.tick()
    root = registry_path(catalog)
    try:
        info = root.lstat()
    except FileNotFoundError:
        return {}, 0
    _need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
          and stat.S_IMODE(info.st_mode) & 0o077 == 0, "private custody registry required")
    fd = held.directory(root)
    _need(_did(os.fstat(fd)) == _did(info), "custody registry FD/name changed")
    result, groups, total = {}, {}, 0
    with os.scandir(fd) as children:
        for child in children:
            held.tick()
            match = _NAME.fullmatch(child.name)
            _need(match is not None, "unknown custody registry artifact")
            item = os.stat(child.name, dir_fd=fd, follow_symlinks=False)
            limit = SNAPSHOT_LIMIT if match[3] == "snapshot" else RECEIPT_LIMIT
            _need(stat.S_ISREG(item.st_mode) and item.st_nlink == 1 and item.st_uid == os.geteuid()
                  and 0 < item.st_size <= limit, "custody registry file type/owner/bound")
            result[child.name] = _stamp(item)
            group = match[1] + "-" + match[2]
            groups.setdefault(group, set()).add(match[3])
            total += item.st_size
            _need(len(groups) <= MAX_RUNS and len(result) <= 3 * MAX_RUNS and total <= REGISTRY_LIMIT, "custody registry bound exceeded")
    if complete:
        _need(all(kinds == {"original", "snapshot", "receipt"} for kinds in groups.values()), "partial custody publication remains blocked")
    return result, total


def _original(held, catalog, pin, now):
    _need(type(pin) is dict, "exact original catalog raw pin required")
    key = Path(pin.get("path", "")).stem
    _need(re.fullmatch(r"[0-9a-f]{64}", key), "exact original catalog key required")
    path = checked(catalog) / (key + ".json")
    raw = held.read(pin, path, RECEIPT_LIMIT)
    row = _decode(raw)
    target = checked(Path(row["output"]))
    _need(str(target) == row["output"] and hashlib.sha256(os.path.normcase(str(target)).encode()).hexdigest() == key
          and row["key"] == key and row["schema"] == "rovezero.managed-output.v1"
          and row["state"] == "active" and row.get("deferred") is True and row.get("identity") is None
          and type(row.get("reserved_bytes")) is int and 0 < row["reserved_bytes"] < 2**63
          and type(row.get("catalog_limit_bytes")) is int and 0 < row["catalog_limit_bytes"] < 2**63
          and type(row.get("pid")) is int and row["pid"] > 0
          and reboot_proves_dead(row.get("lifetime"), now), "original unbound active row/reboot contract is not proven")
    for other in (checked(catalog), registry_path(catalog), checked(Path(row["source"]))):
        _need(not (target == other or target in other.parents or other in target.parents), "custody target overlaps control/source storage")
    return row, raw, target


@dataclass(frozen=True)
class InterruptedUnboundCustody:
    key: str
    output: Path
    charge_bytes: int
    original_pin: dict


class QuarantineValidation:
    def __init__(self, held, catalog, guard, now, registry, registry_bytes, records, observed):
        self.held, self.catalog, self.guard, self.now = held, catalog, guard, now
        self.registry, self.registry_bytes = registry, registry_bytes
        self.records, self.observed = tuple(records), tuple(observed)
        self.keys = frozenset(record.key for record in records)

    def check(self):
        self.guard.assert_held()
        current = lifetime()
        _need(all(current[key] == self.now[key] for key in ("host", "boot", "platform")), "current custody boot changed")
        self.held.check()
        for observation in self.observed:
            observation.check()
        _need(_registry(self.held, self.catalog) == (self.registry, self.registry_bytes), "custody registry changed")
        self.guard.assert_held()

    def usage(self, rows):
        _mark_stage(self.held, "output_accounting")
        self.held.tick()
        charges = {record.key: record.charge_bytes for record in self.records}
        _need(set(charges) <= {row["key"] for _, row, _ in rows}, "quarantined original row disappeared")
        total = self.registry_bytes
        for _, row, target in rows:
            amount = charges.get(row["key"])
            if amount is None:
                self.held.tick()
                amount = tree_size(target, allow_disappearing=True)
                self.held.tick()
                _need(amount <= row["reserved_bytes"], "managed output exceeds reservation")
            total += amount
        warn_output_threshold(total, scope="custody")
        return total

    def close(self, primary=None):
        self.held.close(primary)


def validate_quarantines(catalog, rows, receipt_pins, guard, *, maximum_seconds=30):
    """One verifier for aware admission, writer checks and sampled accounting."""
    catalog = checked(Path(catalog))
    _need(Path(guard.path) == catalog / "catalog.lock", "original shared catalog gate required")
    guard.assert_held()
    registry_root = registry_path(catalog)
    for _, _, target in rows:
        _need(not (target.is_relative_to(registry_root) or registry_root.is_relative_to(target)),
              "catalog row overlaps custody registry")
    held = _Held(maximum_seconds)
    try:
        now = lifetime()
        registry, registry_bytes = _registry(held, catalog)
        _need(isinstance(receipt_pins, (tuple, list)) and len(receipt_pins) <= MAX_RUNS, "bounded explicit custody receipt pins required")
        pins = tuple(receipt_pins)
        selected = {Path(pin["path"]).name for pin in pins}
        _need(len(selected) == len(pins) and selected == {name for name in registry if name.endswith(".receipt.json")}, "every final custody marker requires an explicit exact selection")
        old_rows = {row["key"]: row for _, row, _ in rows}
        records, observations, identities = [], [], set()
        for pin in pins:
            name = Path(pin["path"]).name
            match = _NAME.fullmatch(name)
            _need(match is not None and match[3] == "receipt", "final custody receipt name required")
            path = registry_path(catalog) / name
            receipt = _decode(held.read(pin, path, RECEIPT_LIMIT))
            _need(type(receipt) is dict and set(receipt) == {
                "schema", "kind", "classification", "original_catalog_row", "original_catalog_stat",
                "original_copy", "snapshot", "output", "observed_lifetime", "publisher_source",
                "current_parent_identity", "current_postboot_observed_identity", "mount_observation",
                "quarantine_charge_bytes", "root_exclusive_writer_acknowledged",
                "original_generation_identity_known", "original_generation_custody_reissued",
                "physical_completion", "numeric_success_issued", "exclusion_scope"}, "exact custody receipt fields required")
            _need(type(receipt["observed_lifetime"]) is dict
                  and set(receipt["observed_lifetime"]) == {"host", "boot", "platform", "process_birth"}
                  and all(isinstance(item, str) and item for item in receipt["observed_lifetime"].values()),
                  "exact custody observer lifetime required")
            held.read(receipt["publisher_source"], Path(__file__).resolve(), SNAPSHOT_LIMIT)
            _need(receipt["schema"] == SCHEMA and receipt["kind"] == "receipt"
                  and receipt["classification"] == "interrupted_unbound_custody_quarantined"
                  and receipt["root_exclusive_writer_acknowledged"] is True
                  and receipt["original_generation_identity_known"] is False
                  and receipt["original_generation_custody_reissued"] is False
                  and receipt["physical_completion"] == "unknown_unchanged"
                  and receipt["numeric_success_issued"] is False
                  and receipt["exclusion_scope"] == "output_and_all_ancestors_descendants"
                  and all(receipt["observed_lifetime"][key] == now[key] for key in ("host", "boot", "platform")), "custody receipt meaning/current boot mismatch")
            row, raw, target = _original(held, catalog, receipt["original_catalog_row"], now)
            _need(receipt["original_catalog_stat"] == _stamp(Path(receipt["original_catalog_row"]["path"]).lstat()),
                  "original catalog metadata identity changed")
            stem = row["key"] + "-" + hashlib.sha256(raw).hexdigest()
            _need(name == stem + ".receipt.json" and old_rows.get(row["key"]) == row
                  and row["key"] not in {record.key for record in records}, "custody receipt original row join changed")
            copy_raw = held.read(receipt["original_copy"], registry_path(catalog) / (stem + ".original.json"), RECEIPT_LIMIT)
            _need(copy_raw == raw, "custody original byte copy changed")
            manifest = _decode(held.read(receipt["snapshot"], registry_path(catalog) / (stem + ".snapshot.json"), SNAPSHOT_LIMIT))
            observation = _Observed(held, target)
            _need(_encode(observation.snapshot) == _encode(manifest) and receipt["output"] == str(target)
                  and observation.parent_identity == receipt["current_parent_identity"]
                  and observation.identity == receipt["current_postboot_observed_identity"]
                  and observation.mounts == receipt["mount_observation"], "current custody snapshot/identity changed")
            identity = tuple(observation.identity[:2])
            _need(identity not in identities, "multiple retained custody root aliases")
            identities.add(identity)
            for _, other, other_target in rows:
                if other["key"] != row["key"] and other_target.exists():
                    _need(tuple(_stamp(other_target.lstat())[:2]) != identity, "custody root aliases another output")
            _need(manifest["regular_bytes"] <= row["reserved_bytes"]
                  and type(receipt["quarantine_charge_bytes"]) is int
                  and receipt["quarantine_charge_bytes"] == row["reserved_bytes"], "custody reservation charge changed")
            records.append(InterruptedUnboundCustody(row["key"], target, row["reserved_bytes"], receipt["original_catalog_row"]))
            observations.append(observation)
        result = QuarantineValidation(held, catalog, guard, now, registry, registry_bytes, records, observations)
        result.usage(rows)
        result.check()
        return result
    except BaseException as error:
        held.close(error)
        raise


def _exclusive(held, directory_fd, name, raw, maximum):
    held.tick()
    _need(0 < len(raw) <= maximum, "custody publication artifact bound")
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
    try:
        offset = 0
        while offset < len(raw):
            held.tick()
            written = os.write(fd, raw[offset:])
            _need(written > 0, "short custody publication write")
            offset += written
        os.fsync(fd)
    finally:
        _close_fd(fd, sys.exc_info()[1])
    os.fsync(directory_fd)


def publish_custody_quarantine(catalog, original_row_pin, publisher_source_pin, *,
                               root_exclusive_writer=False, existing_receipts=(), apply=False, maximum_seconds=30):
    """Explicit maintenance only. Partial artifacts remain; final marker is last.

    target payload is never opened for reading. The caller must exclude other
    privileged writers/namespaces; the receipt only binds this current namespace.
    """
    from managed_outputs import _catalog_rows
    _need(root_exclusive_writer is True, "explicit exclusive preservation writer required")
    catalog = checked(Path(catalog))
    stage, held, previous = "not_started", _Held(maximum_seconds), None
    try:
        held.read(publisher_source_pin, Path(__file__).resolve(), SNAPSHOT_LIMIT)
        with original_custody_gate(catalog) as guard:
            rows = _catalog_rows(catalog)
            previous = validate_quarantines(catalog, rows, existing_receipts, guard, maximum_seconds=held.until - time.monotonic())
            now = lifetime()
            row, raw, target = _original(held, catalog, original_row_pin, now)
            _need(any(other == row for _, other, _ in rows), "original row absent from original shared catalog")
            _need(row["key"] not in previous.keys, "custody receipt is immutable; no republishing")
            observation = _Observed(held, target)
            _need(observation.snapshot["regular_bytes"] <= row["reserved_bytes"], "unbound custody exceeds original reservation")
            stem = row["key"] + "-" + hashlib.sha256(raw).hexdigest()
            registry = registry_path(catalog)
            snapshot_raw = _encode(observation.snapshot)
            copy_pin = _pin(registry / (stem + ".original.json"), raw)
            snapshot_pin = _pin(registry / (stem + ".snapshot.json"), snapshot_raw)
            receipt = {"schema": SCHEMA, "kind": "receipt",
                "classification": "interrupted_unbound_custody_quarantined",
                "original_catalog_row": original_row_pin, "original_catalog_stat": _stamp(Path(original_row_pin["path"]).lstat()),
                "original_copy": copy_pin, "snapshot": snapshot_pin,
                "output": str(target), "observed_lifetime": now, "publisher_source": publisher_source_pin,
                "current_parent_identity": observation.parent_identity,
                "current_postboot_observed_identity": observation.identity, "mount_observation": observation.mounts,
                "quarantine_charge_bytes": row["reserved_bytes"], "root_exclusive_writer_acknowledged": True,
                "original_generation_identity_known": False, "original_generation_custody_reissued": False,
                "physical_completion": "unknown_unchanged", "numeric_success_issued": False,
                "exclusion_scope": "output_and_all_ancestors_descendants"}
            receipt_raw = _encode(receipt)
            _need(len(receipt_raw) <= RECEIPT_LIMIT, "custody receipt bound exceeded")
            added = len(raw) + len(snapshot_raw) + len(receipt_raw)
            _need(previous.registry_bytes + added <= REGISTRY_LIMIT, "custody registry metadata bound exceeded")
            warn_output_threshold(previous.usage(rows) - observation.snapshot["regular_bytes"]
                                  + row["reserved_bytes"] + added, scope="custody-publication")
            held.check()
            previous.check()
            observation.check()
            if not apply:
                return {"applied": False, "original_row_unchanged": True, "quarantine_charge_bytes": row["reserved_bytes"],
                        "registry_metadata_bytes_after": previous.registry_bytes + added, "receipt_issued": False}
            stage = "registry_creation_pending"
            try:
                os.mkdir(registry, 0o700)
                parent_fd = held.directory(registry.parent)
                os.fsync(parent_fd)
            except FileExistsError:
                pass
            directory_fd = held.directory(registry)
            _need(stat.S_IMODE(os.fstat(directory_fd).st_mode) & 0o077 == 0, "private registry publication required")
            expected_registry = dict(previous.registry)
            for kind, content, limit in (("original", raw, RECEIPT_LIMIT), ("snapshot", snapshot_raw, SNAPSHOT_LIMIT), ("receipt", receipt_raw, RECEIPT_LIMIT)):
                held.check()
                observation.check()
                _need(_did(os.fstat(directory_fd)) == _did(checked(registry).lstat()), "registry publication FD/name changed")
                _need(_registry(held, catalog, complete=False)[0] == expected_registry, "registry changed during publication")
                guard.assert_held()
                current = lifetime()
                _need(all(current[key] == now[key] for key in ("host", "boot", "platform")), "boot changed during custody publication")
                stage = kind + "_publication_pending"
                _exclusive(held, directory_fd, stem + "." + kind + ".json", content, limit)
                stage = kind + "_durable"
                artifact_path = registry / (stem + "." + kind + ".json")
                held.read(_pin(artifact_path, content), artifact_path, limit)
                expected_registry[artifact_path.name] = _stamp(artifact_path.lstat())
            held.check()
            result_pin = _pin(registry / (stem + ".receipt.json"), receipt_raw)
            previous.close()
            previous = None
            final = validate_quarantines(catalog, _catalog_rows(catalog), (*existing_receipts, result_pin), guard, maximum_seconds=held.until - time.monotonic())
            try:
                final.check()
            finally:
                final.close(sys.exc_info()[1])
            return {"applied": True, "receipt": result_pin, "original_row_unchanged": True,
                    "original_generation_custody_reissued": False, "physical_completion": "unknown_unchanged"}
    except BaseException as error:
        if hasattr(error, "add_note"):
            error.add_note("custody publication stage=" + stage + "; no rollback/delete attempted")
        raise
    finally:
        primary = sys.exc_info()[1]
        try:
            if previous is not None:
                previous.close(primary)
        finally:
            held.close(sys.exc_info()[1] or primary)
