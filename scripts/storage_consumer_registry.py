"""Scoped input leases for cooperating consumers and cleanup operations.

An empty registry only speaks for participants in this registry. It is never
evidence that legacy or unregistered consumers do not exist. Registration
and retirement must use the same pinned registry identity and OS lock.

A directory input identifies the registered directory and its subtree at
inspection time; it does not track former descendants moved outside that
subtree. Register individual files before use when their identity must remain
protected after a move. If the registered directory identity changes, a
same-device cleanup candidate whose alias status cannot be established is
retained as unknown. Recovery requires restoring the original identity or
the original consumer's supported lease completion or recovery procedure;
age, a pathname, or process exit alone does not establish physical completion.
"""
from __future__ import annotations

import hashlib
from itertools import islice
import os
from pathlib import Path
import re
import stat
import time
import uuid

from managed_storage import StorageError, checked, read_record_snapshot, reject_credential_names, write_record
from storage_guard import CatalogGuard, lifetime

REGISTRY_SCHEMA = "rovezero.consumer-registry.v1"
LEASE_SCHEMA = "rovezero.consumer-input-lease.v1"
_ID = re.compile(r"[0-9a-f]{32}\Z")
_LIMIT = 4096
_REGISTRY_PAGE_SIZE = 128
BINDING_NAME = "consumer-registry.binding"
BINDING_SCHEMA = "rovezero.catalog-consumer-binding.v1"
REQUIRED_MARKER_NAME = "consumer-registry.required"
REQUIRED_MARKER_SCHEMA = "rovezero.catalog-consumer-required.v1"


class _BoundedScanResults(list):
    """Bound diagnostic storage without treating omitted conflicts as safe."""

    def __init__(self):
        super().__init__()
        self.overflow = False

    def append(self, value):
        if len(self) < _LIMIT:
            super().append(value)
        else:
            self.overflow = True


def _registry_directory_stamp(root):
    info = root.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode):
        raise StorageError("registry directory unavailable")
    return (info.st_dev, info.st_ino, info.st_mode, info.st_mtime_ns, info.st_ctime_ns)


def _registry_owner_identity(root):
    """Observe the registry directory and the actual shared lock, without links.

    This pins an existing owner's identity for this object lifetime only; it
    does not make a newly constructed object a recovery authority.
    """
    checked(root)
    checked(root / "registry.lock")
    try:
        directory = root.stat(follow_symlinks=False)
        lock = (root / "registry.lock").stat(follow_symlinks=False)
    except OSError as error:
        raise StorageError("consumer registry owner identity unavailable") from error
    if (not stat.S_ISDIR(directory.st_mode) or not stat.S_ISREG(lock.st_mode)
            or lock.st_nlink != 1 or not directory.st_ino or not lock.st_ino):
        raise StorageError("consumer registry owner identity unavailable")
    return ((directory.st_dev, directory.st_ino), (lock.st_dev, lock.st_ino))


def _absolute(value):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise StorageError("absolute consumer input path required")
    return checked(path)


def _hardlinked_inodes(root, deadline):
    """Bounded metadata-only scan of a proposed directory cleanup target."""
    root_info = root.stat()
    pending, result, directories, count = [root], set(), {(root_info.st_dev, root_info.st_ino)}, 0
    device = root_info.st_dev
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as children:
            for entry in children:
                count += 1
                if count % 128 == 0 and time.monotonic() >= deadline:
                    raise StorageError("cleanup alias scan time budget exceeded")
                if count > 100_000:
                    raise StorageError("cleanup alias scan limit exceeded")
                path = Path(entry.path)
                name = entry.name.lower()
                if (name == ".ssh" or name == "id_ecdsa" or "credential" in name
                        or "service-account" in name or "service_account" in name
                        or "api-key" in name or "api_key" in name):
                    raise StorageError("private-named entry is outside alias observation")
                reject_credential_names(path)
                # Windows DirEntry.stat may report device/inode as zero even
                # when os.stat supplies stable identifiers.
                info = os.stat(entry.path, follow_symlinks=False) if os.name == "nt" else entry.stat(follow_symlinks=False)
                if info.st_dev != device:
                    raise StorageError("cleanup alias scan crosses a device")
                if stat.S_ISDIR(info.st_mode):
                    directories.add((info.st_dev, info.st_ino))
                    pending.append(path)
                elif stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
                    result.add((info.st_dev, info.st_ino))
    return result, directories


def _ancestor_directories(target, deadline):
    """Observe target ancestors for a moved registered directory identity."""
    identities = set()
    for parent in target.parents:
        if time.monotonic() >= deadline:
            raise StorageError("cleanup ancestor scan time budget exceeded")
        info = parent.stat(follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode):
            raise StorageError("cleanup ancestor is not a directory")
        identities.add((info.st_dev, info.st_ino))
    return identities


def _catalog_guard(root, guard):
    root = _absolute(root)
    if not isinstance(guard, CatalogGuard) or _absolute(guard.path) != root / "catalog.lock":
        raise StorageError("original catalog guard required for consumer binding")
    guard.assert_held()
    return root


def load_catalog_binding(root, guard):
    """Return the pinned registry for a bound catalog, or None if unbound.

    A present but invalid or unavailable binding is an error, not an unbound
    catalog. Caller holds the catalog lock throughout this operation.
    """
    root = _catalog_guard(root, guard)
    path = root / BINDING_NAME
    if os.path.lexists(root / REQUIRED_MARKER_NAME):
        read_record_snapshot(root / REQUIRED_MARKER_NAME, schema=REQUIRED_MARKER_SCHEMA)
        raise StorageError("catalog requires an unconfigured consumer registry; preserve cleanup targets")
    if not os.path.lexists(path):
        return None
    record, _, _ = read_record_snapshot(path, schema=BINDING_SCHEMA)
    if set(record) != {"schema", "registry_root", "registry_id"}:
        raise StorageError("invalid catalog consumer binding")
    result = ConsumerRegistry(record["registry_root"], registry_id=record["registry_id"])
    guard.assert_held()
    return result


def mark_catalog_registry_unavailable(root, guard, *, force_bound=False):
    """Persist cleanup deferral when a required domain lacks a usable registry.

    This marker does not authorize adoption or identify active consumers.
    A bound or malformed catalog is never overwritten.
    """
    root = _catalog_guard(root, guard)
    if type(force_bound) is not bool:
        raise StorageError("explicit consumer deferral policy required")
    if os.path.lexists(root / BINDING_NAME) and not force_bound:
        return
    path = root / REQUIRED_MARKER_NAME
    if os.path.lexists(path):
        read_record_snapshot(path, schema=REQUIRED_MARKER_SCHEMA)
    else:
        write_record(path, {"schema": REQUIRED_MARKER_SCHEMA, "status": "registry_unavailable"})
    guard.assert_held()


def require_catalog_binding(root, registry, guard):
    if type(registry) is not ConsumerRegistry:
        raise StorageError("pinned consumer registry required")
    registry._check_identity()
    existing = load_catalog_binding(root, guard)
    if (existing is None or existing.root != registry.root
            or existing.registry_id != registry.registry_id
            or existing._owner_identity != registry._owner_identity):
        raise StorageError("catalog consumer registry binding missing or differs")
    registry._check_identity()
    return existing


def bind_catalog(root, registry, guard):
    """Bind once while holding the catalog's actual OS lock; no rebinding."""
    root = _catalog_guard(root, guard)
    if type(registry) is not ConsumerRegistry:
        raise StorageError("pinned consumer registry required")
    existing = load_catalog_binding(root, guard)
    if existing is not None:
        return require_catalog_binding(root, registry, guard)
    identity = registry._check_identity()
    write_record(root / BINDING_NAME, {"schema": BINDING_SCHEMA,
                 "registry_root": str(registry.root), "registry_id": registry.registry_id})
    if identity["sharing"] == "group" and os.name == "posix":
        os.chmod(root / BINDING_NAME, 0o660)
    guard.assert_held()
    return require_catalog_binding(root, registry, guard)


def recover_empty_required_catalog(root, registry, guard):
    """Recover an unconfigured marker only for an empty, quiescent catalog.

    Nonempty catalogs may contain prior unregistered consumers and require an
    evidence-bearing owner-specific recovery; they are not silently adopted.
    """
    root = _catalog_guard(root, guard)
    if type(registry) is not ConsumerRegistry:
        raise StorageError("pinned consumer registry required")
    marker = root / REQUIRED_MARKER_NAME
    read_record_snapshot(marker, schema=REQUIRED_MARKER_SCHEMA)
    for item in root.iterdir():
        if item.name in ("catalog.lock", REQUIRED_MARKER_NAME, BINDING_NAME):
            continue
        if item.name == "slots":
            safe_slots = checked(item)
            if stat.S_ISDIR(safe_slots.lstat().st_mode) and not any(safe_slots.iterdir()):
                continue
        raise StorageError("nonempty catalog requires owner-specific recovery")
    binding = root / BINDING_NAME
    if os.path.lexists(binding):
        record, _, _ = read_record_snapshot(binding, schema=BINDING_SCHEMA)
        if record != {"schema": BINDING_SCHEMA, "registry_root": str(registry.root),
                      "registry_id": registry.registry_id}:
            raise StorageError("pending catalog binding differs")
    else:
        identity = registry._check_identity()
        write_record(binding, {"schema": BINDING_SCHEMA, "registry_root": str(registry.root),
                              "registry_id": registry.registry_id})
        if identity["sharing"] == "group" and os.name == "posix":
            os.chmod(binding, 0o660)
    guard.assert_held()
    marker.unlink()
    if os.name == "posix":
        fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return require_catalog_binding(root, registry, guard)


def verify_lease_coverage(registry, owner_id, paths, *, expected_owner=None):
    """Verify a parent's live registration covers these child input paths.

    This is not evidence that unrelated or legacy consumers participate, nor
    that every catalog with access to these paths is bound to this registry.
    """
    if (type(registry) is not ConsumerRegistry or not isinstance(owner_id, str)
            or not _ID.fullmatch(owner_id) or not isinstance(paths, (list, tuple))
            or not 1 <= len(paths) <= _LIMIT):
        raise StorageError("bounded pinned consumer lease coverage required")
    targets = [_absolute(path) for path in paths]
    with registry.guard() as guard:
        row, _, _ = read_record_snapshot(registry.root / ("consumer-" + owner_id + ".json"), schema=LEASE_SCHEMA)
        if (row.get("registry_id") != registry.registry_id or row.get("owner_id") != owner_id
                or row.get("state") != "active" or type(row.get("inputs")) is not list):
            raise StorageError("active parent consumer lease required")
        owner = expected_owner if expected_owner is not None else {"pid": os.getpid(), "lifetime": lifetime()}
        if (not isinstance(owner, dict) or type(owner.get("pid")) is not int
                or not isinstance(owner.get("lifetime"), dict)
                or row.get("pid") != owner["pid"] or row.get("lifetime") != owner["lifetime"]):
            raise StorageError("consumer lease owner does not match original supervisor")
        for target in targets:
            covered = False
            for item in row["inputs"]:
                if (type(item) is not dict or set(item) != {"path", "kind", "device", "inode"}
                        or item.get("kind") not in ("file", "directory", "future")
                        or not isinstance(item.get("path"), str)):
                    raise StorageError("invalid parent consumer input")
                source = _absolute(item["path"])
                if not (source == target or (item["kind"] in ("directory", "future") and target.is_relative_to(source))):
                    continue
                if item["kind"] != "future":
                    info = source.stat()
                    if (info.st_dev, info.st_ino) != (item.get("device"), item.get("inode")):
                        raise StorageError("parent consumer input identity changed")
                covered = True
                break
            if not covered:
                raise StorageError("parent consumer lease does not cover input")
        guard.check()
    return True


class ConsumerRegistry:
    def __init__(self, root, *, registry_id):
        self.root = _absolute(root)
        if not isinstance(registry_id, str) or not _ID.fullmatch(registry_id):
            raise StorageError("pinned consumer registry identity required")
        self.registry_id = registry_id
        self._owner_identity = _registry_owner_identity(self.root)
        self._check_identity()

    @classmethod
    def initialize(cls, root, *, scope, group_id=None):
        root = _absolute(root)
        if not isinstance(scope, str) or not scope or len(scope) > 256:
            raise StorageError("bounded explicit registry scope required")
        if group_id is not None and (os.name != "posix" or type(group_id) is not int or group_id < 0):
            raise StorageError("explicit POSIX registry group required")
        root.mkdir(mode=0o700, parents=False, exist_ok=False)
        identity = uuid.uuid4().hex
        try:
            if group_id is not None:
                os.chown(root, -1, group_id)
                os.chmod(root, 0o2770)
            write_record(root / "registry.json", {"schema": REGISTRY_SCHEMA,
                         "registry_id": identity, "scope": scope,
                         "sharing": "group" if group_id is not None else "owner"})
            if group_id is not None:
                os.chmod(root / "registry.json", 0o660)
            fd = os.open(root / "registry.lock", os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            try:
                if group_id is not None:
                    os.fchmod(fd, 0o660)
            finally:
                os.close(fd)
        except BaseException:
            # Preserve a partially initialized registry for explicit recovery.
            raise
        return cls(root, registry_id=identity)

    def _check_identity(self):
        if _registry_owner_identity(self.root) != self._owner_identity:
            raise StorageError("consumer registry original owner identity changed")
        record, _, _ = read_record_snapshot(self.root / "registry.json", schema=REGISTRY_SCHEMA)
        if (record.get("registry_id") != self.registry_id or
                record.get("sharing") not in ("owner", "group") or
                not isinstance(record.get("scope"), str) or not record["scope"]):
            raise StorageError("consumer registry identity changed")
        if _registry_owner_identity(self.root) != self._owner_identity:
            raise StorageError("consumer registry original owner identity changed")
        return record

    def _write(self, path, value):
        identity = self._check_identity()
        write_record(path, value)
        if identity["sharing"] == "group":
            os.chmod(path, 0o660)

    def guard(self):
        return _RegistryGuard(self)

    def register(self, inputs, *, owner_id=None, future_inputs=(), _guard=None):
        if (not isinstance(inputs, (list, tuple)) or not isinstance(future_inputs, (list, tuple))
                or not 1 <= len(inputs) + len(future_inputs) <= _LIMIT):
            raise StorageError("bounded explicit consumer inputs required")
        paths = [_absolute(path) for path in inputs] + [_absolute(path) for path in future_inputs]
        future = set(_absolute(path) for path in future_inputs)
        if len(set(paths)) != len(paths):
            raise StorageError("duplicate consumer input")
        owner_id = uuid.uuid4().hex if owner_id is None else owner_id
        if not isinstance(owner_id, str) or not _ID.fullmatch(owner_id):
            raise StorageError("valid consumer owner identity required")
        if _guard is None:
            with self.guard() as guard:
                return self.register(inputs, owner_id=owner_id, future_inputs=future_inputs, _guard=guard)
        guard = _guard
        if type(guard) is not _RegistryGuard or guard.registry is not self:
            raise StorageError("original consumer registry guard required")
        guard.check()
        entries = []
        for path in paths:
            try:
                info = path.stat()
            except FileNotFoundError:
                if path not in future:
                    raise StorageError("consumer input missing at registration")
                entries.append({"path": str(path), "kind": "future", "device": None, "inode": None})
                continue
            if not stat.S_ISDIR(info.st_mode) and not stat.S_ISREG(info.st_mode):
                raise StorageError("regular file or directory consumer input required")
            entries.append({"path": str(path), "kind": "file" if stat.S_ISREG(info.st_mode) else "directory",
                            "device": info.st_dev, "inode": info.st_ino})
        row = {"schema": LEASE_SCHEMA, "registry_id": self.registry_id,
               "owner_id": owner_id, "pid": os.getpid(), "lifetime": lifetime(),
               "state": "active", "inputs": entries}
        path = self.root / ("consumer-" + owner_id + ".json")
        if os.path.lexists(path):
            raise StorageError("consumer owner identity already registered")
        try:
            self._write(path, row)
            guard.check()
        except BaseException as error:
            # No consumer has been dispatched or received this lease.
            # Withdraw only our exact row while the original lock and
            # registry identity are still verifiably held.
            try:
                guard.check()
                if os.path.lexists(path):
                    published, _, _ = read_record_snapshot(path, schema=LEASE_SCHEMA)
                    if published != row:
                        raise StorageError("partially published consumer row changed")
                    path.unlink()
            except BaseException as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note("unstarted consumer row cleanup unavailable: " + str(cleanup_error))
            raise
        return ConsumerLease(self, owner_id, row)

    def prune_closed(self, owner_ids):
        """Remove only explicitly named completed lease records under the shared lock."""
        if not isinstance(owner_ids, (list, tuple)) or not 1 <= len(owner_ids) <= _LIMIT:
            raise StorageError("bounded explicit closed consumer identities required")
        with self.guard() as guard:
            for owner_id in owner_ids:
                if not isinstance(owner_id, str) or not _ID.fullmatch(owner_id):
                    raise StorageError("valid consumer owner identity required")
                path = self.root / ("consumer-" + owner_id + ".json")
                row, _, _ = read_record_snapshot(path, schema=LEASE_SCHEMA)
                if (row.get("registry_id") != self.registry_id or row.get("owner_id") != owner_id
                        or row.get("state") != "closed"
                        or row.get("completion") not in ("not_started", "physical_complete")):
                    raise StorageError("only explicitly completed consumer records may be pruned")
                guard.check()
                path.unlink()


class _RegistryGuard:
    def __init__(self, registry):
        self.registry = registry
        self.lock = CatalogGuard(registry.root / "registry.lock")

    def __enter__(self):
        # Check before acquiring, then check again while holding the opened
        # lock. CatalogGuard can create a lock removed between these checks;
        # the postcheck rejects that replacement before cleanup is allowed.
        self.registry._check_identity()
        self.lock.acquire()
        try:
            self.check()
        except BaseException:
            self.lock.release()
            raise
        return self

    def __exit__(self, *args):
        self.lock.release()

    def check(self):
        self.lock.assert_held()
        self.registry._check_identity()

    def register(self, inputs, *, owner_id=None, future_inputs=()):
        return self.registry.register(inputs, owner_id=owner_id, future_inputs=future_inputs, _guard=self)

    def _registry_pages(self, deadline, unknown):
        """Scan bounded pages under the original registry lock and identity.

        Cooperating writers cannot alter rows while the guard is held. A
        directory change during the scan makes the result incomplete. A
        second names-only pass detects membership changes even on filesystems
        with coarse directory timestamps; row contents are read only once.
        This is not isolation from nonparticipating writers or a filesystem
        snapshot.
        """
        root = self.registry.root
        try:
            before = _registry_directory_stamp(root)
            def add_name(fingerprint, name):
                count, xor, total = fingerprint
                digest = int.from_bytes(hashlib.sha256(os.fsencode(name)).digest(), "big")
                return count + 1, xor ^ digest, (total + digest) % (1 << 256)
            first = (0, 0, 0)
            with os.scandir(root) as entries:
                while True:
                    if time.monotonic() >= deadline:
                        unknown.append({"reason": "registry_scan_time_budget_exceeded"})
                        return
                    page = list(islice(entries, _REGISTRY_PAGE_SIZE))
                    if not page:
                        break
                    self.check()
                    for entry in page:
                        first = add_name(first, entry.name)
                        yield root / entry.name
            self.check()
            if before != _registry_directory_stamp(root):
                unknown.append({"reason": "registry_directory_changed_during_scan"})
                return
            second = (0, 0, 0)
            with os.scandir(root) as entries:
                while True:
                    if time.monotonic() >= deadline:
                        unknown.append({"reason": "registry_scan_time_budget_exceeded"})
                        return
                    page = list(islice(entries, _REGISTRY_PAGE_SIZE))
                    if not page:
                        break
                    self.check()
                    for entry in page:
                        second = add_name(second, entry.name)
            self.check()
            if first != second or before != _registry_directory_stamp(root):
                unknown.append({"reason": "registry_directory_changed_during_scan"})
        except (OSError, StorageError):
            unknown.append({"reason": "registry_scan_incomplete"})

    def conflicts(self, targets, *, seconds=1.0):
        self.check()
        if not isinstance(targets, (list, tuple)) or not 1 <= len(targets) <= _LIMIT:
            raise StorageError("bounded cleanup targets required")
        if type(seconds) not in (int, float) or not 0 < seconds <= 30:
            raise StorageError("bounded alias scan time required")
        deadline = time.monotonic() + seconds
        candidates = [_absolute(item) for item in targets]
        matches, unknown, directory_aliases = _BoundedScanResults(), _BoundedScanResults(), {}
        target_ancestors = {}
        comparisons = 0
        for path in self._registry_pages(deadline, unknown):
            if path.name in ("registry.json", "registry.lock", "registry-history.json"):
                continue
            if not path.name.startswith("consumer-") or not path.name.endswith(".json"):
                unknown.append({"reason": "unexpected_registry_entry", "record": str(path)})
                continue
            try:
                row, _, _ = read_record_snapshot(path, schema=LEASE_SCHEMA)
                owner = row.get("owner_id")
                values = row.get("inputs")
                if (row.get("registry_id") != self.registry.registry_id or
                        not isinstance(owner, str) or not _ID.fullmatch(owner) or
                        path.name != "consumer-" + owner + ".json" or
                        row.get("state") not in ("active", "physical_unknown_retained", "closed") or
                        type(values) is not list or not 1 <= len(values) <= _LIMIT):
                    raise StorageError("invalid consumer lease")
                if row["state"] == "closed":
                    continue
                for value in values:
                    if (type(value) is not dict or set(value) != {"path", "kind", "device", "inode"}
                            or not isinstance(value["path"], str)
                            or value["kind"] not in ("file", "directory", "future")
                            or (value["kind"] == "future" and (value["device"] is not None or value["inode"] is not None))
                            or (value["kind"] != "future" and
                                (type(value["device"]) is not int or type(value["inode"]) is not int))):
                        raise StorageError("invalid consumer input")
                    source = _absolute(value["path"])
                    if value["kind"] == "future":
                        try:
                            created = source.stat()
                        except FileNotFoundError:
                            created = None
                        if created is not None:
                            if not (stat.S_ISDIR(created.st_mode) or stat.S_ISREG(created.st_mode)):
                                raise StorageError("future consumer input has invalid type")
                            value = {"path": value["path"],
                                     "kind": "file" if stat.S_ISREG(created.st_mode) else "directory",
                                     "device": created.st_dev, "inode": created.st_ino}
                    directory_identity_changed = False
                    if value["kind"] == "directory":
                        try:
                            source_info = source.stat(follow_symlinks=False)
                            directory_identity_changed = (
                                not stat.S_ISDIR(source_info.st_mode)
                                or (source_info.st_dev, source_info.st_ino)
                                != (value["device"], value["inode"])
                            )
                        except OSError:
                            directory_identity_changed = True
                    for target in candidates:
                        comparisons += 1
                        if comparisons % _REGISTRY_PAGE_SIZE == 0 and time.monotonic() >= deadline:
                            self.check()
                            return {"conflicts": list(matches),
                                    "unknown": [*unknown, {"reason": "registry_scan_time_budget_exceeded"}],
                                    "coverage": "registered_only"}
                        path_overlap = source == target or source.is_relative_to(target) or target.is_relative_to(source)
                        if path_overlap:
                            matches.append({"record": str(path), "input": str(source),
                                            "target": str(target), "state": row["state"]})
                            continue
                        try:
                            info = target.stat()
                        except FileNotFoundError:
                            # Cleanup rechecks its pinned list after unlinking a
                            # leaf. An absent disjoint pathname has no inode to
                            # alias; the cleanup owner separately verifies the
                            # expected presence or absence of every candidate.
                            continue
                        same_identity = ((value["kind"] == "file" and stat.S_ISREG(info.st_mode)
                                          or value["kind"] == "directory" and stat.S_ISDIR(info.st_mode))
                                         and (value["device"], value["inode"]) == (info.st_dev, info.st_ino))
                        if value["kind"] == "directory" and not same_identity:
                            try:
                                if target not in target_ancestors:
                                    target_ancestors[target] = _ancestor_directories(target, deadline)
                                if (value["device"], value["inode"]) in target_ancestors[target]:
                                    matches.append({"record": str(path), "input": str(source),
                                                    "target": str(target), "state": row["state"]})
                                    continue
                            except (OSError, StorageError):
                                unknown.append({"reason": "directory_ancestor_scan_incomplete", "target": str(target)})
                                continue
                        if same_identity:
                            matches.append({"record": str(path), "input": str(source),
                                            "target": str(target), "state": row["state"]})
                        elif value["kind"] == "directory" and info.st_dev == value["device"]:
                            if directory_identity_changed and not stat.S_ISDIR(info.st_mode):
                                # An unrecorded child may have moved separately.
                                # Limit uncertainty to this same-device target.
                                unknown.append({"reason": "registered_directory_identity_changed_alias_unknown",
                                                "target": str(target)})
                                continue
                            if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
                                unknown.append({"reason": "directory_hardlink_alias_unknown", "target": str(target)})
                                continue
                            if stat.S_ISDIR(info.st_mode):
                                try:
                                    if target not in directory_aliases:
                                        if time.monotonic() >= deadline:
                                            raise StorageError("cleanup alias scan time budget exceeded")
                                        directory_aliases[target] = _hardlinked_inodes(target, deadline)
                                    aliases, directories = directory_aliases[target]
                                except (OSError, StorageError):
                                    unknown.append({"reason": "directory_alias_scan_incomplete", "target": str(target)})
                                    continue
                                if (value["device"], value["inode"]) in directories:
                                    matches.append({"record": str(path), "input": str(source),
                                                    "target": str(target), "state": row["state"]})
                                elif directory_identity_changed:
                                    unknown.append({"reason": "registered_directory_identity_changed_alias_unknown",
                                                    "target": str(target)})
                                elif aliases:
                                    unknown.append({"reason": "directory_contents_alias_unknown", "target": str(target)})
                        elif value["kind"] == "directory" and stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
                            # A registered directory may contain another name
                            # for this inode; discovering that would require an
                            # unbounded recursive scan of consumer inputs.
                            unknown.append({"reason": "directory_hardlink_alias_unknown", "target": str(target)})
                        elif stat.S_ISDIR(info.st_mode):
                            try:
                                if target not in directory_aliases:
                                    if time.monotonic() >= deadline:
                                        raise StorageError("cleanup alias scan time budget exceeded")
                                    directory_aliases[target] = _hardlinked_inodes(target, deadline)
                                aliases, directories = directory_aliases[target]
                            except (OSError, StorageError):
                                unknown.append({"reason": "directory_alias_scan_incomplete", "target": str(target)})
                                continue
                            if value["kind"] == "file" and (value["device"], value["inode"]) in aliases:
                                matches.append({"record": str(path), "input": str(source),
                                                "target": str(target), "state": row["state"]})
                            elif value["kind"] == "directory" and (value["device"], value["inode"]) in directories:
                                matches.append({"record": str(path), "input": str(source),
                                                "target": str(target), "state": row["state"]})
                            elif value["kind"] == "directory" and aliases:
                                unknown.append({"reason": "directory_contents_alias_unknown", "target": str(target)})
            except (OSError, StorageError, ValueError):
                unknown.append({"reason": "invalid_or_unreadable_consumer", "record": str(path)})
        self.check()
        if matches.overflow or unknown.overflow:
            return {"conflicts": list(matches),
                    "unknown": [*unknown, {"reason": "registry_scan_result_limit"}],
                    "coverage": "registered_only"}
        return {"conflicts": list(matches), "unknown": list(unknown), "coverage": "registered_only"}


class ConsumerLease:
    def __init__(self, registry, owner_id, row):
        self.registry, self.owner_id, self.row = registry, owner_id, row
        self._closed = False

    def close(self, *, physical_complete=False, not_started=False):
        if self._closed or (physical_complete is not True and not_started is not True):
            raise StorageError("original consumer requires confirmed physical end or no dispatch")
        if not_started and self.row.get("state") != "active":
            raise StorageError("a dispatched retained consumer cannot be marked not started")
        if os.getpid() != self.row["pid"] or lifetime() != self.row["lifetime"]:
            raise StorageError("only the original consumer can close its lease")
        path = self.registry.root / ("consumer-" + self.owner_id + ".json")
        with self.registry.guard() as guard:
            current, _, _ = read_record_snapshot(path, schema=LEASE_SCHEMA)
            if current != self.row:
                raise StorageError("consumer lease changed")
            closed = dict(current, state="closed", completion=("not_started" if not_started else "physical_complete"))
            self.registry._write(path, closed)
            # Keep bounded terminal evidence without accumulating an owner
            # record for every successful run. Unknown/active records are
            # never included or removed by this path.
            history_path = self.registry.root / "registry-history.json"
            if os.path.lexists(history_path):
                history, _, _ = read_record_snapshot(history_path, schema="rovezero.consumer-history.v1")
                if (type(history.get("closed_count")) is not int or history["closed_count"] < 0
                        or type(history.get("recent")) is not list):
                    raise StorageError("invalid consumer history")
            else:
                history = {"schema": "rovezero.consumer-history.v1", "closed_count": 0, "recent": []}
            history["closed_count"] += 1
            history["recent"] = (history["recent"] + [{"owner_id": self.owner_id,
                                   "completion": closed["completion"]}])[-32:]
            self.registry._write(history_path, history)
            path.unlink()
            if os.name == "posix":
                fd = os.open(self.registry.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            guard.check()
        self._closed = True

    def retain_unknown(self, *, tree_gone=False):
        """Record supervisor tree termination without inferring a GPU fence."""
        if self._closed or tree_gone is not True or self.row.get("state") != "active":
            raise StorageError("original consumer with confirmed tree end required")
        if os.getpid() != self.row["pid"] or lifetime() != self.row["lifetime"]:
            raise StorageError("only the original consumer can update its lease")
        path = self.registry.root / ("consumer-" + self.owner_id + ".json")
        with self.registry.guard() as guard:
            current, _, _ = read_record_snapshot(path, schema=LEASE_SCHEMA)
            if current != self.row:
                raise StorageError("consumer lease changed")
            retained = dict(current, state="physical_unknown_retained", tree_gone=True)
            self.registry._write(path, retained)
            guard.check()
        self.row = retained
