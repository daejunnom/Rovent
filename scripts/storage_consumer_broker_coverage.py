"""Pin a previously registered descriptor under the original registry guard.

This only covers cooperating consumers and bound cleanup catalogs.  The pin
does not prove file content, the child's actual open/read, source immutability,
or GPU completion. The original supervisor must keep it through the reader's
transaction and verify it again before accepting the result.
"""
from __future__ import annotations

import os
from pathlib import Path
import stat

from managed_storage import StorageError, read_record_snapshot
from storage_consumer_registry import ConsumerLease, LEASE_SCHEMA, _ID, _RegistryGuard, _absolute
from storage_guard import lifetime

_MAX_LEASES = 128
_MAX_PARTS = 128


def _nonsecret_path(value):
    path = Path(value)
    if len(path.parts) > _MAX_PARTS:
        raise StorageError("descriptor path component limit exceeded")
    for component in path.parts:
        name = component.lower()
        if (name.startswith(".env") or "credential" in name or "service-account" in name
                or "service_account" in name or "api_key" in name or "api-key" in name
                or name in {".ssh", "id_rsa", "id_ed25519", "id_ecdsa"}
                or name.endswith((".pem", ".key", ".p12", ".pfx"))):
            raise StorageError("private-named descriptor path is unavailable")
    return _absolute(path)


def _open_nofollow(path, *, directory):
    """Open by component, refusing symlinks and blocking special files."""
    if os.name != "posix" or not all(hasattr(os, flag) for flag in ("O_NOFOLLOW", "O_DIRECTORY")):
        raise StorageError("POSIX no-follow descriptor traversal required")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    current = os.open("/", flags)
    try:
        components = path.parts[1:]
        if not components:
            if not directory:
                raise StorageError("regular descriptor file required")
            return current
        for component in components[:-1]:
            nxt = os.open(component, flags, dir_fd=current)
            old, current = current, nxt
            os.close(old)
        final_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        final_flags |= os.O_DIRECTORY if directory else getattr(os, "O_NONBLOCK", 0)
        result = os.open(components[-1], final_flags, dir_fd=current)
        old, current = current, None
        try:
            os.close(old)
        except BaseException:
            os.close(result)
            raise
        return result
    except BaseException:
        if current is not None:
            os.close(current)
        raise


def _identity(fd, *, directory):
    info = os.fstat(fd)
    if (directory and not stat.S_ISDIR(info.st_mode)) or (not directory and not stat.S_ISREG(info.st_mode)):
        raise StorageError("descriptor or registered source has wrong type")
    return (info.st_dev, info.st_ino)


def _current_row(guard, lease):
    if type(guard) is not _RegistryGuard or type(lease) is not ConsumerLease or lease.registry is not guard.registry:
        raise StorageError("original registry guard and supervisor lease required")
    guard.check()
    if (lease._closed or not isinstance(lease.owner_id, str) or not _ID.fullmatch(lease.owner_id)
            or not isinstance(lease.row, dict)):
        raise StorageError("active original lease required")
    row, _, _ = read_record_snapshot(guard.registry.root / ("consumer-" + lease.owner_id + ".json"),
                                     schema=LEASE_SCHEMA)
    if (row != lease.row or set(row) != {"schema", "registry_id", "owner_id", "pid", "lifetime", "state", "inputs"}
            or row.get("registry_id") != guard.registry.registry_id or row.get("owner_id") != lease.owner_id
            or type(row.get("pid")) is not int or row["pid"] != os.getpid()
            or row.get("lifetime") != lifetime() or row.get("state") != "active"
            or type(row.get("inputs")) is not list or not 1 <= len(row["inputs"]) <= 4096):
        raise StorageError("original active supervisor lease changed")
    guard.check()
    return row


class DescriptorCoveragePin:
    """Open source and descriptor identities; no content or child read claim."""

    def __init__(self, guard, lease, source, target, source_kind, source_identity):
        self._guard, self._lease = guard, lease
        self._source, self._target = source, target
        self._source_kind, self._source_identity = source_kind, source_identity
        self._source_fd = self._target_fd = None
        self._closed = False
        self._close_unknown = False
        try:
            self._source_fd = _open_nofollow(source, directory=source_kind == "directory")
            self._target_fd = _open_nofollow(target, directory=False)
            self._target_identity = _identity(self._target_fd, directory=False)
            self.verify()
        except BaseException as error:
            try:
                self.close()
            except BaseException as close_error:
                if hasattr(error, "add_note"):
                    error.add_note("descriptor pin close outcome unknown: " + type(close_error).__name__)
            raise

    def verify(self):
        if self._closed or self._close_unknown:
            raise StorageError("descriptor coverage pin is closed")
        row = _current_row(self._guard, self._lease)
        if not any(item.get("path") == str(self._source) and item.get("kind") == self._source_kind
                   and (item.get("device"), item.get("inode")) == self._source_identity
                   for item in row["inputs"] if isinstance(item, dict)):
            raise StorageError("original descriptor coverage no longer registered")
        if (_identity(self._source_fd, directory=self._source_kind == "directory") != self._source_identity
                or _identity(self._target_fd, directory=False) != self._target_identity):
            raise StorageError("pinned descriptor identity changed")
        if _nonsecret_path(self._source) != self._source or _nonsecret_path(self._target) != self._target:
            raise StorageError("descriptor path changed")
        source_fd = target_fd = None
        primary = None
        try:
            source_fd = _open_nofollow(self._source, directory=self._source_kind == "directory")
            target_fd = _open_nofollow(self._target, directory=False)
            if (_identity(source_fd, directory=self._source_kind == "directory") != self._source_identity
                    or _identity(target_fd, directory=False) != self._target_identity):
                raise StorageError("named descriptor or source was replaced")
        except BaseException as error:
            primary = error
            if isinstance(error, OSError):
                raise StorageError("descriptor or registered source unavailable") from error
            raise
        finally:
            close_error = None
            for fd in (target_fd, source_fd):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError as error:
                        self._close_unknown = True
                        close_error = close_error or error
            if close_error is not None:
                if primary is not None and hasattr(primary, "add_note"):
                    primary.add_note("descriptor verification fd close outcome unknown")
                elif primary is None:
                    raise StorageError("descriptor verification fd close outcome unknown") from close_error
        self._guard.check()

    def close(self):
        if self._closed:
            return
        self._closed = True
        error = None
        for name in ("_target_fd", "_source_fd"):
            fd = getattr(self, name)
            setattr(self, name, None)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError as exc:
                    self._close_unknown = True
                    error = error or exc
        if error is not None:
            raise StorageError("descriptor pin close outcome unknown") from error

    def __enter__(self):
        if self._closed:
            raise StorageError("descriptor coverage pin is closed")
        return self

    def __exit__(self, *_):
        self.close()


def pin_descriptor_coverage(guard, leases, descriptor):
    """Pin one existing regular file already covered by a live owned lease."""
    if type(guard) is not _RegistryGuard or not isinstance(leases, (list, tuple)) or len(leases) > _MAX_LEASES:
        raise StorageError("bounded original supervisor leases required")
    target = _nonsecret_path(descriptor)
    guard.check()
    for lease in leases:
        row = _current_row(guard, lease)
        for item in row["inputs"]:
            if (type(item) is not dict or set(item) != {"path", "kind", "device", "inode"}
                    or item.get("kind") not in ("file", "directory", "future")
                    or type(item.get("path")) is not str):
                raise StorageError("malformed original consumer input")
            # A future reservation cannot prove an existing file's original inode.
            if item["kind"] == "future":
                continue
            source = _nonsecret_path(item["path"])
            if not (source == target or (item["kind"] == "directory" and target.is_relative_to(source))):
                continue
            identity = (item["device"], item["inode"])
            if any(type(value) is not int for value in identity):
                raise StorageError("registered input identity unavailable")
            try:
                return DescriptorCoveragePin(guard, lease, source, target, item["kind"], identity)
            except OSError as error:
                raise StorageError("descriptor or registered source unavailable") from error
    raise StorageError("descriptor is not covered by an existing supervisor lease")
