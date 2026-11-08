"""Owned, bounded regenerable build storage; no historical-tree adoption."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import time
import uuid
from pathlib import Path
from storage_guard import CatalogGuard, lifetime, reboot_proves_dead, owner_proven_dead

GiB = 1024 ** 3
SCHEMA = "rovezero.managed-build.v1"
MAX_ENTRIES = 100_000
DEFAULT_OUTPUT_WARNING_BYTES = 18 * GiB
def warn_output_threshold(total, threshold=DEFAULT_OUTPUT_WARNING_BYTES, *, scope="output", observed_bytes=None):
    """Classify accounting only; it is not a physical-storage warning.

    Accounting can include reservations or logical lengths.  Keep its result
    available to callers without implying that the device crossed a threshold.
    Actual storage warnings come from the bounded total-usage observation.
    """
    return (type(total) is int and type(threshold) is int and total >= 0
            and threshold > 0 and total > threshold)


class StorageError(RuntimeError):
    pass


class CatalogBusy(StorageError):
    """A catalog's non-blocking OS lock is held by another owner."""


def reject_credential_names(path: Path) -> None:
    for part in (path, *path.parents):
        if part.name.lower().startswith(".env") or part.name.lower() in {
            "credentials", "service-account.json", "id_rsa", "id_ed25519",
        }:
            raise StorageError("credential path is not a managed storage target")


def checked(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    reject_credential_names(path)
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise StorageError("managed path contains a symlink or junction")
    # Windows AppContainer aliases can name the same inode without a reparse
    # attribute. Normalize their physical path before containment comparisons.
    return Path(os.path.realpath(path))


def tree_size(path: Path, *, allow_disappearing=False) -> int:
    path = checked(path)
    if not path.exists():
        return 0
    total = 0
    pending = [path]
    seen = set()
    entries = 0
    while pending:
        current = pending.pop()
        try:
            reject_credential_names(current)
            info = current.lstat()
            entries += 1
            if entries > MAX_ENTRIES:
                raise StorageError("managed tree entry bound exceeded")
            if stat.S_ISLNK(info.st_mode):
                checked(current.parent)
                if not allow_disappearing:
                    # The link itself is retired. Never traverse it or inspect
                    # an external target (venv lib64 -> lib stays in this root).
                    target = Path(os.path.abspath(current.parent / os.readlink(current)))
                    if not target.is_relative_to(path) or not checked(target).is_relative_to(path):
                        raise StorageError("managed symlink escapes its owned tree")
                total += info.st_size
                continue
            checked(current)
            if stat.S_ISDIR(info.st_mode):
                with os.scandir(current) as children:
                    for item in children:
                        if entries + len(pending) >= MAX_ENTRIES:
                            raise StorageError("managed tree entry bound exceeded")
                        pending.append(Path(item.path))
            elif stat.S_ISREG(info.st_mode):
                identity = (info.st_dev, info.st_ino)
                if identity not in seen:
                    total += info.st_size
                    seen.add(identity)
            elif not allow_disappearing:
                raise StorageError("unsupported managed tree entry")
        except FileNotFoundError:
            if not allow_disappearing:
                raise
    return total


def remove_tree(path: Path, parent: Path) -> None:
    path = checked(path)
    parent = checked(parent)
    if path.parent != parent:
        raise StorageError("deletion must name a direct owned child")
    tree_size(path)  # Preflight the whole tree before any deletion.
    if not path.exists():
        return

    first_permission_error = None

    def writable_remove(func, name, error):
        nonlocal first_permission_error
        original = error[1]
        if not isinstance(original, PermissionError):
            raise original.with_traceback(error[2])
        if first_permission_error is None:
            first_permission_error = original
        try:
            # Only owned, preflighted entries; do not follow a changed link.
            target = checked(Path(name))
            if not target.is_relative_to(path):
                raise StorageError("cleanup escaped owned tree")
            target_info = target.lstat()
            if not (stat.S_ISDIR(target_info.st_mode) or stat.S_ISREG(target_info.st_mode)):
                raise StorageError("unsupported cleanup permission target")

            def writable_directory(directory):
                directory = checked(directory)
                if not directory.is_relative_to(path):
                    raise StorageError("cleanup chmod escaped owned tree")
                info = directory.lstat()
                if not stat.S_ISDIR(info.st_mode):
                    raise StorageError("cleanup permission parent is not a directory")
                mode = stat.S_IMODE(info.st_mode) | stat.S_IRWXU
                if os.name == "posix":
                    # chmod the verified directory inode, never a link target.
                    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                    descriptor = os.open(directory, flags)
                    try:
                        opened = os.fstat(descriptor)
                        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                            raise StorageError("cleanup directory identity changed")
                        os.fchmod(descriptor, mode)
                    finally:
                        os.close(descriptor)
                else:
                    os.chmod(directory, mode)

            # POSIX unlink/rmdir require write + search on the parent, even
            # when the file itself is writable. Never chmod the outside owner.
            if target.parent.is_relative_to(path):
                writable_directory(target.parent)
            if stat.S_ISDIR(target_info.st_mode):
                writable_directory(target)
            elif os.name == "nt":
                # Windows read-only files need their own attribute cleared.
                # POSIX file mode does not govern unlink; leave shared inodes alone.
                os.chmod(target, stat.S_IMODE(target_info.st_mode) | stat.S_IRUSR | stat.S_IWUSR)
            checked(target)
            retried_info = target.lstat()
            if (retried_info.st_dev, retried_info.st_ino) != (target_info.st_dev, target_info.st_ino):
                raise StorageError("cleanup target identity changed")
            func(name)
        except Exception as failure:
            # Preserve the first failure when permission repair/retry fails.
            raise failure from original
    shutil.rmtree(path, onerror=writable_remove)
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise StorageError("cleanup did not remove the owned tree") from first_permission_error


def write_record(path: Path, value: dict) -> None:
    checked(path)
    encoded = (json.dumps(value, sort_keys=True) + "\n").encode()
    if len(encoded) > 8192:
        raise StorageError("ownership record exceeds bound")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                                | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _record_stamp(info, *, creation_time=False):
    # CPython 3.12+ on Windows exposes ChangeTime through fstat().st_ctime,
    # while path stat retains creation time there. Only the cross-source
    # comparison uses their common birthtime; FD stability and the retained
    # snapshot still include the original change-time observation.
    time = getattr(info, "st_birthtime_ns", info.st_ctime_ns) if creation_time else info.st_ctime_ns
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, time)


def read_record_snapshot(path: Path, *, schema=SCHEMA):
    path = checked(path)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate ownership field")
            result[key] = value
        return result
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= 8192:
                raise StorageError("bounded single-link ownership record required")
            raw = stream.read(8193)
            if (len(raw) != before.st_size
                    or _record_stamp(before) != _record_stamp(os.fstat(stream.fileno()))
                    or _record_stamp(before, creation_time=os.name == "nt")
                    != _record_stamp(checked(path).lstat(), creation_time=os.name == "nt")):
                raise StorageError("ownership file identity changed while reading")
        value = json.loads(raw, object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite record")))
    except (ValueError, OSError) as exc:
        raise StorageError("ownership record is invalid; preserve it") from exc
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise StorageError("unrecognized ownership record; preserve it")
    return value, hashlib.sha256(raw).hexdigest(), _record_stamp(before)


def read_record(path: Path) -> dict:
    return read_record_snapshot(path)[0]


def canonical_root() -> Path:
    if os.environ.get("RUNNER_TEMP"):
        base = Path(os.environ["RUNNER_TEMP"]) / "RoveZero" / "build"
    elif os.name == "nt":
        if not os.environ.get("APPDATA"):
            raise StorageError("APPDATA is unavailable")
        base = Path(os.environ["APPDATA"]) / "RoveZero" / "build"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "rovezero" / "build"
    return checked(base / "managed-v1")


class ManagedBuild:
    """One catalog owner serializes reuse/eviction and owns one fixed scratch slot.

    Interrupted/unknown leases are deliberately not stolen. Recovery is manual,
    after checking the original process tree, paths and source ownership.
    """
    def __init__(self, root: Path, source: Path, *, slot_limit=8 * GiB,
                 total_limit=18 * GiB, slots_limit=4, purpose=None, output_preservation=None,
                 dependency_registry=None, dependency_registry_required=False):
        if type(dependency_registry_required) is not bool:
            raise StorageError("dependency registry policy must be explicit")
        if dependency_registry is not None:
            from storage_consumer_registry import ConsumerRegistry
            if type(dependency_registry) is not ConsumerRegistry:
                raise StorageError("pinned consumer registry required")
        # None preserves the legacy domain. A required domain refuses removal
        # if its participating consumer registry was not supplied.
        self.dependency_registry = dependency_registry
        self.dependency_registry_required = dependency_registry_required
        self._dependency_removal_deferred = False
        if output_preservation is not None:
            from managed_output_unattributed_preservation import UnattributedSelection
            if (type(output_preservation) is not UnattributedSelection
                    or output_preservation._bound_deadline is None):
                raise StorageError('exact source7 original-deadline selection required')
        self.output_preservation = output_preservation
        self._output_budget_observer = None
        self.root = checked(root)
        if output_preservation is not None:
            from managed_output_unattributed_preservation import assert_manager_preservation_scope
            assert_manager_preservation_scope(self.root, output_preservation)
        self.source = Path(os.path.realpath(source))
        if self.root.is_relative_to(self.source) or self.source.is_relative_to(self.root):
            raise StorageError("build storage must be outside the checkout")
        if min(slot_limit, total_limit, slots_limit) <= 0 or slot_limit > total_limit:
            raise StorageError("invalid storage budget")
        self.slot_limit = slot_limit
        self.total_limit = total_limit
        self.slots_limit = slots_limit
        if purpose is not None and (not isinstance(purpose, str) or not 1 <= len(purpose) <= 64
                or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in purpose)):
            raise StorageError("purpose requires 1..64 lowercase safe characters")
        self.purpose = purpose
        identity = os.path.normcase(str(self.source)) + ("\0" + purpose if purpose else "")
        self.source_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
        self.slot = self.root / "slots" / self.source_id
        self.target = self.slot / "cargo-target"
        self.scratch_root = self.slot / "tmp"
        self.scratch = self.scratch_root
        self.lock = self.root / "owner.json"
        self.token = uuid.uuid4().hex
        self._scoped_lifecycle = False
        self._isolated_target = False
        self._binding_guard = None
        self.acquired = False
        self.finished = False
        # Only the original runner can transition this before calling its
        # process launcher. An attempted launch is conservatively dispatched.
        self._dispatch_started = False
        self._foreign_output_cache = None
        self.record = {}
        self.before = 0
        self.lifetime = lifetime()
        self.guard = CatalogGuard(self.root / "catalog.lock")

    def _identity(self) -> dict:
        info = self.root.stat()
        value = {"schema": SCHEMA, "token": self.token, "pid": os.getpid(),
                 "source_id": self.source_id, "source": str(self.source),
                 "lifetime": self.lifetime, "root_identity": [info.st_dev, info.st_ino]}
        if self.purpose is not None:
            value["purpose"] = self.purpose
        if self._scoped_lifecycle:
            value["scratch_layout"] = "owner-token-v1"
        if self._isolated_target:
            value["target_layout"] = "scratch-token-v1"
        return value

    def _set_scoped_paths(self, *, isolated_target=False, token=None):
        self._scoped_lifecycle = True
        self.scratch = self.scratch_root / (token or self.token)
        self._isolated_target = isolated_target
        if isolated_target:
            self.target = self.scratch / "cargo-target"

    def _empty_catalog_for_binding(self, guard):
        """Automatic adoption is limited to a freshly empty build catalog."""
        guard.assert_held()
        for entry in self.root.iterdir():
            if entry.name == "catalog.lock":
                continue
            if entry.name == "slots":
                checked(entry)
                if not entry.is_dir() or any(entry.iterdir()):
                    return False
                continue
            # Prior owner, closure, maintenance or any unknown entry is not
            # evidence that legacy consumers finished using the catalog.
            return False
        return True

    def _resolve_dependency_binding(self, guard):
        """Pin this catalog to one registry; an unknown binding defers cleanup."""
        from storage_consumer_registry import (bind_catalog, load_catalog_binding,
                                                mark_catalog_registry_unavailable)
        self._binding_guard = guard
        requested = self.dependency_registry
        try:
            bound = load_catalog_binding(self.root, guard)
            if bound is None and requested is not None:
                if self._empty_catalog_for_binding(guard):
                    bound = bind_catalog(self.root, requested, guard)
                else:
                    # Existing rows can have unregistered legacy consumers.
                    # Do not adopt them into a new registry automatically.
                    mark_catalog_registry_unavailable(self.root, guard)
                    self.dependency_registry = None
                    self.dependency_registry_required = True
                    self._dependency_removal_deferred = True
            elif bound is None and self.dependency_registry_required:
                mark_catalog_registry_unavailable(self.root, guard)
            if bound is not None:
                if requested is not None and (bound.root != requested.root
                                              or bound.registry_id != requested.registry_id):
                    raise StorageError("consumer registry differs from catalog binding")
                self.dependency_registry = bound
                self.dependency_registry_required = True
        except (StorageError, OSError):
            # A bound but inaccessible/mismatched registry cannot be replaced
            # by an arbitrary registry. Keep unrelated execution available.
            self.dependency_registry = None
            self.dependency_registry_required = True
            if not (os.path.lexists(self.root / "consumer-registry.binding")
                    or os.path.lexists(self.root / "consumer-registry.required")):
                # Without a durable marker a later legacy cleaner could erase
                # this owner's still-registered inputs. Never launch in that gap.
                mark_catalog_registry_unavailable(self.root, guard)

    def _target_requires_isolation(self):
        if not self.dependency_registry_required and self.dependency_registry is None:
            return False
        if os.path.lexists(self.scratch_root):
            try:
                checked(self.scratch_root)
                if not self.scratch_root.is_dir() or any(self.scratch_root.iterdir()):
                    # A prior retained scoped run may still own the shared
                    # target even if no consumer registration was available.
                    return True
            except (StorageError, OSError):
                return True
        if self.dependency_registry is None:
            return True
        try:
            with self.dependency_registry.guard() as guard:
                observation = guard.conflicts([self.target])
                return bool(observation.get("conflicts") or observation.get("unknown"))
        except (StorageError, OSError):
            return True

    def _assert_owner(self) -> None:
        self.guard.assert_held()
        if read_record(self.lock) != self._identity():
            raise StorageError("catalog owner changed; preserve all artifacts")
        if self.slot.exists():
            record = read_record(self.slot / "record.json")
            if record != self.record:
                raise StorageError("slot ownership record changed; preserve it")

    def _records(self) -> list[tuple[Path, dict]]:
        catalog = self.root / "slots"
        rows = []
        for item in catalog.iterdir():
            checked(item)
            if not item.is_dir():
                raise StorageError("unknown catalog entry; preserve it")
            record = read_record(item / "record.json")
            if record.get("source_id") != item.name or set(record) - {
                "lifetime", "root_identity", "purpose", "scratch_layout", "target_layout",
            } != {
                "schema", "source_id", "source", "token", "pid", "state", "finished_ns",
            }:
                raise StorageError("unrecognized slot identity; preserve it")
            if (not isinstance(record["source"], str) or not Path(record["source"]).is_absolute()
                    or ("purpose" in record and (not isinstance(record["purpose"], str)
                        or not 1 <= len(record["purpose"]) <= 64
                        or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in record["purpose"])))
                    or hashlib.sha256((os.path.normcase(record["source"]) + (
                        "\0" + record["purpose"] if "purpose" in record else "")).encode()).hexdigest()[:24] != item.name
                    or record["state"] not in {"active", "complete"}
                    or not isinstance(record["finished_ns"], int) or record["finished_ns"] < 0
                    or not isinstance(record["pid"], int) or record["pid"] <= 0
                    or not isinstance(record["token"], str) or len(record["token"]) != 32
                    or ("scratch_layout" in record and record["scratch_layout"] != "owner-token-v1")
                    or ("target_layout" in record and (record["target_layout"] != "scratch-token-v1"
                        or record.get("scratch_layout") != "owner-token-v1"))
                    or any(c not in "0123456789abcdef" for c in record["token"])):
                raise StorageError("invalid slot provenance; preserve it")
            allowed = {"record.json", "last-run.json", "cargo-target", "tmp"}
            if any(p.name not in allowed for p in item.iterdir()):
                raise StorageError("unknown slot content; preserve it")
            rows.append((item, record))
        return rows

    def _remove_with_dependencies(self, path: Path, parent: Path) -> bool:
        """Remove a known tree only while its participating registry is locked.

        The scope covers registered consumers only; it does not attest that
        unregistered or legacy readers are absent. False means preserved.
        Callers already hold the build catalog guard, establishing the common
        catalog -> registry order. The registry remains held through removal.
        """
        if not path.exists():
            return True
        registry = self.dependency_registry
        if self.dependency_registry_required or registry is not None:
            from storage_consumer_registry import require_catalog_binding
            try:
                if registry is None:
                    raise StorageError("consumer registry unavailable")
                require_catalog_binding(self.root, registry, self._binding_guard)
            except (StorageError, OSError):
                self._dependency_removal_deferred = True
                return False
        if registry is None:
            if self.dependency_registry_required:
                self._dependency_removal_deferred = True
                return False
            remove_tree(path, parent)
            return True
        scope = registry.guard()
        try:
            guard = scope.__enter__()
        except (StorageError, OSError):
            # An unavailable or replaced registry is not authority to delete.
            self._dependency_removal_deferred = True
            return False
        try:
            try:
                result = guard.conflicts([path])
                if result.get("conflicts") or result.get("unknown"):
                    self._dependency_removal_deferred = True
                    return False
                guard.check()
            except (StorageError, OSError):
                self._dependency_removal_deferred = True
                return False
            # Removal failures themselves propagate: they may be partial and
            # must not be misreported as an ordinary protected candidate.
            remove_tree(path, parent)
            return True
        finally:
            scope.__exit__(None, None, None)

    def _retained_scratch_marker(self, token):
        return self.scratch_root / (".rovezero-retained-scratch-" + token + ".json")

    def _record_retained_scratch(self, *, not_dispatched):
        marker = self._retained_scratch_marker(self.token)
        if os.path.lexists(marker):
            raise StorageError("scoped scratch receipt already exists; preserve it")
        write_record(marker, {"schema": "rovezero.retained-scoped-scratch.v1",
                              "token": self.token, "scratch": str(self.scratch),
                              "tree_gone": True, "physical_completion": "not_inferred",
                              "no_child_dispatched": not_dispatched})

    def _prune_retained_scoped_scratch(self):
        """Reclaim only completed owned scratch after consumer release."""
        if not self.scratch_root.is_dir() or self.dependency_registry is None:
            return
        checked(self.scratch_root)
        entries = list(self.scratch_root.iterdir())
        if len(entries) > 4096:
            return  # A bounded, independent cleanup can handle this later.
        prefix = ".rovezero-retained-scratch-"
        for marker in entries:
            if not marker.name.startswith(prefix) or not marker.name.endswith(".json"):
                continue
            token = marker.name[len(prefix):-5]
            if len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
                continue
            scratch = self.scratch_root / token
            try:
                original = read_record_snapshot(marker, schema="rovezero.retained-scoped-scratch.v1")
                expected = {"schema": "rovezero.retained-scoped-scratch.v1", "token": token,
                            "scratch": str(scratch), "tree_gone": True,
                            "physical_completion": "not_inferred", "no_child_dispatched": True}
                if original[0] != expected or not self._remove_with_dependencies(scratch, self.scratch_root):
                    continue
                # The small receipt can itself be a declared input. Protect it
                # with the same lock and recheck it before unlink.
                with self.dependency_registry.guard() as guard:
                    from storage_consumer_registry import require_catalog_binding
                    require_catalog_binding(self.root, self.dependency_registry, self._binding_guard)
                    result = guard.conflicts([marker])
                    if (result.get("conflicts") or result.get("unknown")
                            or read_record_snapshot(marker, schema="rovezero.retained-scoped-scratch.v1") != original):
                        continue
                    guard.check()
                    marker.unlink()
            except (StorageError, OSError):
                # Malformed or inaccessible receipts are retained, not fatal
                # to a new independent run.
                self._dependency_removal_deferred = True

    def _prune(self) -> None:
        rows = self._records()
        for slot, record in rows:
            if record["state"] == "active" and slot != self.slot:
                raise StorageError("interrupted or active slot requires owner-aware maintenance")
        completed = sorted(
            ((slot, record) for slot, record in rows
             if slot != self.slot and record["state"] == "complete"),
            key=lambda row: row[1]["finished_ns"],
        )
        total = sum(tree_size(slot) for slot, _ in rows)
        count = len(rows)
        while completed and (total > self.total_limit or count > self.slots_limit):
            slot, expected = completed.pop(0)
            if read_record(slot / "record.json") != expected:
                raise StorageError("eviction ownership changed")
            if self.dependency_registry_required or self.dependency_registry is not None:
                scratch_root = slot / "tmp"
                if scratch_root.exists():
                    checked(scratch_root)
                    if any(scratch_root.iterdir()):
                        # Any retained scoped scratch can include a dispatched
                        # run whose physical completion is unknown. A missing
                        # or incomplete receipt is not authority to evict it.
                        self._dependency_removal_deferred = True
                        continue
            amount = tree_size(slot)
            if self._remove_with_dependencies(slot, slot.parent):
                total -= amount
                count -= 1
        # A still-active or retained owned slot may itself exceed the aggregate
        # threshold.  Warn rather than terminate its unrelated work; this is
        # never authority to delete active/unknown data.  The slot-count
        # boundary and per-slot post-run cleanup remain independent.
        warn_output_threshold(total, self.total_limit, scope="managed_build_catalog",
                              observed_bytes=total)
        if count > self.slots_limit:
            if self.dependency_registry_required or self.dependency_registry is not None:
                # The disk threshold is advisory. Keep protected slots and let
                # this independent owner run; active owners were rejected above.
                try:
                    print("managed_storage_warning=protected_slot_count_exceeded", file=sys.stderr)
                except (OSError, ValueError):
                    pass
            else:
                raise StorageError("slot limit cannot be met without deleting active/unknown data")

    def acquire(self) -> "ManagedBuild":
        # Diagnostic only: a single bounded cached read before any catalog lock.
        try:
            from rovezero_usage_snapshot import warn_cached_total_usage
            warn_cached_total_usage()
        except Exception:
            pass
        checked(self.root)
        if self.output_preservation is not None:
            from managed_output_unattributed_preservation import assert_manager_preservation_scope
            assert_manager_preservation_scope(self.root, self.output_preservation)
        self.root.mkdir(parents=True, exist_ok=True)
        self.guard.acquire()
        if any(p.name not in {"owner.json", "slots", "catalog.lock", "maintenance.json", "closure.json",
                              "consumer-registry.binding", "consumer-registry.required"} for p in self.root.iterdir()):
            self.guard.release()
            raise StorageError("unknown build root content; preserve it")
        try:
            self._resolve_dependency_binding(self.guard)
            if self.dependency_registry_required or self.dependency_registry is not None:
                self._set_scoped_paths(isolated_target=self._target_requires_isolation())
        except BaseException:
            self.guard.release()
            raise
        try:
            with self.lock.open("xb") as stream:
                stream.write((json.dumps(self._identity(), sort_keys=True) + "\n").encode())
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as exc:
            self.guard.release()
            raise StorageError("managed build owner exists; do not steal a live/interrupted lease") from exc
        except BaseException:
            self.guard.release()
            raise
        self.acquired = True
        try:
            (self.root / "slots").mkdir(exist_ok=True)
            if self.slot.exists():
                self._records()  # Validate ownership before resetting this slot.
                previous = read_record(self.slot / "record.json")
                if previous["source"] != str(self.source) or previous["state"] != "complete":
                    raise StorageError("source mismatch or interrupted slot; preserve it")
                if self._scoped_lifecycle:
                    self._prune_retained_scoped_scratch()
                if not self._isolated_target and tree_size(self.slot) > self.slot_limit:
                    if not self._remove_with_dependencies(self.slot / "cargo-target", self.slot):
                        raise StorageError("build target required by a registered consumer; preserve it")
                if not self._remove_with_dependencies(self.scratch, self.scratch.parent):
                    raise StorageError("build scratch required by a registered consumer; preserve it")
            else:
                self._prune()
                self.slot.mkdir()
            self.record = {**self._identity(), "state": "active", "finished_ns": 0}
            write_record(self.slot / "record.json", self.record)
            checked(self.scratch.parent)
            self.scratch.parent.mkdir(exist_ok=True)
            checked(self.scratch.parent)
            self.scratch.mkdir()
            checked(self.target)
            self.target.mkdir(exist_ok=True)
            self._prune()
            self.before = tree_size(self.slot)
            return self
        except BaseException as original:
            # A failed admission may expose an interrupted existing slot; do not
            # delete that slot. If this owner already published its own active
            # row, no child has been dispatched; close exactly that row before
            # removing the owner. A failed closure retains the original owner.
            try:
                if self.record:
                    self._assert_owner()
                    terminal = {**self.record, "state": "complete", "finished_ns": time.time_ns()}
                    write_record(self.slot / "record.json", terminal)
                    if read_record(self.slot / "record.json") != terminal:
                        raise StorageError("pre-dispatch slot closure changed")
                    self.record = terminal
                if read_record(self.lock) == self._identity():
                    self.lock.unlink()
            except (StorageError, OSError, ValueError) as secondary:
                original.add_note("pre-dispatch slot closure failed; original owner retained: "
                                  + type(secondary).__name__)
            finally:
                self.acquired = False
                self.guard.release()
            raise

    def environment(self) -> dict:
        self._assert_owner()
        env = os.environ.copy()
        expected = str(self.target)
        if env.get("CARGO_TARGET_DIR") and checked(Path(env["CARGO_TARGET_DIR"])) != self.target:
            raise StorageError("CARGO_TARGET_DIR conflicts with the managed target")
        env.update(CARGO_TARGET_DIR=expected, CARGO_INCREMENTAL="0",
                   TMPDIR=str(self.scratch), TMP=str(self.scratch),
                   TEMP=str(self.scratch), PYTHONDONTWRITEBYTECODE="1",
                   RZ_MANAGED_BUILD_ROOT=str(self.root), RZ_MANAGED_BUILD_TOKEN=self.token)
        if self.dependency_registry is not None:
            env.update(RZ_CONSUMER_REGISTRY_ROOT=str(self.dependency_registry.root),
                       RZ_CONSUMER_REGISTRY_ID=self.dependency_registry.registry_id,
                       RZ_CONSUMER_REGISTRY_REQUIRED="1")
        elif self.dependency_registry_required:
            env.pop("RZ_CONSUMER_REGISTRY_ROOT", None)
            env.pop("RZ_CONSUMER_REGISTRY_ID", None)
            env["RZ_CONSUMER_REGISTRY_REQUIRED"] = "1"
        return env

    def sample_output_budget(self):
        """Explicit original object only; default sampler retains its strict guards."""
        if self.output_preservation is None:
            from managed_outputs import sample_output_budget
            return sample_output_budget()
        self._assert_owner()
        if self._output_budget_observer is None:
            from managed_output_unattributed_preservation import create_budget_observer
            self._output_budget_observer = create_budget_observer(self, self.output_preservation)
        return self._output_budget_observer.sample()

    def finish(self, *, exit_code: int, tree_gone: bool, supervisor_error=None,
               _not_dispatched: bool = False) -> dict:
        self._assert_owner()
        if _not_dispatched and (self._dispatch_started or self.output_preservation is not None
                                or self._output_budget_observer is not None):
            raise StorageError("no-dispatch closure is unavailable; preserve original owner")
        if tree_gone is not True:
            # Keep lease and inputs: uncertain cleanup is not ordinary success.
            raise StorageError("process cleanup unverified; slot and lease retained")
        # Durable original-supervisor witness precedes destructive cleanup.
        # It is bound to this exact token, owner birth, boot and root inode.
        expected_closure = {"schema": SCHEMA, "owner": self._identity(), "tree_gone": True}
        closure_path = self.root / "closure.json"
        write_record(closure_path, expected_closure)
        closure_snapshot = read_record_snapshot(closure_path)
        if closure_snapshot[0] != expected_closure:
            raise StorageError("original terminal closure changed; preserve owner")
        if not _not_dispatched:
            from managed_outputs import finalize_supervised_outputs
            if self.output_preservation is None:
                # The original owner remains held. Only a typed OS lock
                # contention is retried, with a bounded deadline; other
                # custody or I/O errors retain their original failure.
                until = time.monotonic() + 2.0
                while True:
                    self._assert_owner()
                    if read_record_snapshot(closure_path) != closure_snapshot:
                        raise StorageError("original terminal closure changed; preserve owner")
                    try:
                        finalize_supervised_outputs(self.root, self._identity(), foreign_cache=self._foreign_output_cache)
                        break
                    except CatalogBusy:
                        remaining = until - time.monotonic()
                        if remaining <= 0:
                            raise
                        time.sleep(min(0.1, remaining))
            else:
                from managed_output_unattributed_preservation import finalize_original_manager
                finalize_original_manager(self, self.output_preservation)
        observed = tree_size(self.slot)
        physical_unknown = self._scoped_lifecycle and not _not_dispatched
        if physical_unknown:
            temporary_removed = False
            self._dependency_removal_deferred = True
        else:
            temporary_removed = self._remove_with_dependencies(self.scratch, self.scratch.parent)
        scratch_receipt = None
        if self._scoped_lifecycle and not temporary_removed:
            try:
                self._record_retained_scratch(not_dispatched=_not_dispatched)
                scratch_receipt = "recorded"
            except (StorageError, OSError, ValueError):
                # Tree termination is already established. The data remains
                # preserved, while a missing receipt merely prevents later
                # automatic scratch reclamation; it is not a live owner.
                scratch_receipt = "unavailable_preserved"
        reset_requested = tree_size(self.slot) > self.slot_limit
        reset = (self._remove_with_dependencies(self.target, self.target.parent)
                 if reset_requested and not self._isolated_target and not physical_unknown else False)
        self.record = {**self.record, "state": "complete", "finished_ns": time.time_ns()}
        write_record(self.slot / "record.json", self.record)
        result = {"schema": SCHEMA, "source_id": self.source_id,
                  "command_exit_code": exit_code, "tree_cleanup_verified": tree_gone,
                  "supervisor_error": supervisor_error,
                  "pre_run_bytes": self.before, "observed_end_bytes": observed,
                  "temporary_removed": temporary_removed, "post_run_budget_reset": reset,
                  "retained_slot_bytes": tree_size(self.slot),
                  "slot_limit_bytes": self.slot_limit,
                  "catalog_limit_bytes": self.total_limit, "maximum_slots": self.slots_limit}
        if self.dependency_registry is not None or self.dependency_registry_required:
            result["consumer_dependency_coverage"] = ("registered_only" if self.dependency_registry is not None
                                                      else "required_registry_unavailable")
            result["consumer_dependency_removal_deferred_before_prune"] = self._dependency_removal_deferred
            if scratch_receipt is not None:
                result["retained_scratch_receipt"] = scratch_receipt
        if _not_dispatched:
            result["output_finalization"] = "not_needed_no_child_dispatched"
        elif self._foreign_output_cache is not None:
            result["output_custody_scope"] = ("unknown" if self._foreign_output_cache.get("__unknown_ownership__")
                                               else "sampled_without_conflict")
        write_record(self.slot / "last-run.json", result)
        self._prune()
        self._assert_owner()
        self.lock.unlink()
        self.finished = True
        if self._output_budget_observer is not None:
            self._output_budget_observer.close()
        self.guard.release()
        return result


def recover_build(root: Path, owner_sha256: str, *, output_preservation=None,
                  dependency_registry=None, dependency_registry_required=False) -> dict:
    """Explicit exact-owner recovery after reboot or original terminal closure.

    PID absence/reuse cannot establish that no orphan detached from its group.
    Legacy owners without durable lifetime evidence remain preserved.
    """
    root = checked(root)
    if not root.is_dir():
        raise StorageError("existing managed build root required")
    with CatalogGuard(root / "catalog.lock") as guard:
        if any(p.name not in {"owner.json", "slots", "catalog.lock", "maintenance.json", "closure.json",
                              "consumer-registry.binding", "consumer-registry.required"}
               for p in root.iterdir()):
            raise StorageError("unknown build root content; preserve it")
        owner, digest, owner_stamp = read_record_snapshot(root / "owner.json")
        if digest != owner_sha256:
            raise StorageError("original owner digest changed")
        info = root.stat()
        current = lifetime()
        reboot = reboot_proves_dead(owner.get("lifetime"), current)
        closure = read_record(root / "closure.json") if (root / "closure.json").exists() else None
        terminal = (closure == {"schema": SCHEMA, "owner": owner, "tree_gone": True}
                    and owner_proven_dead(owner.get("pid"), owner.get("lifetime"), current))
        if owner.get("root_identity") != [info.st_dev, info.st_ino] or not (reboot or terminal):
            raise StorageError("original tree termination not proven; preserve lease and all inputs")
        manager = ManagedBuild(root, Path(owner["source"]), purpose=owner.get("purpose"),
                               dependency_registry=dependency_registry,
                               dependency_registry_required=dependency_registry_required)
        manager._resolve_dependency_binding(guard)
        if owner.get("scratch_layout") == "owner-token-v1":
            manager.token = owner["token"]
            manager._set_scoped_paths(isolated_target=owner.get("target_layout") == "scratch-token-v1",
                                      token=owner["token"])
        rows = manager._records()
        matches = [(slot, record) for slot, record in rows if slot == manager.slot]
        if len(matches) != 1 or {k: v for k, v in matches[0][1].items()
                                if k not in {"state", "finished_ns"}} != owner:
            raise StorageError("slot and original owner do not match")
        tree_size(manager.slot)
        slot_stamp = (manager.slot.stat().st_dev, manager.slot.stat().st_ino)
        def verify_recovery():
            guard.assert_held()
            if (owner["root_identity"] != [root.stat().st_dev, root.stat().st_ino]
                    or slot_stamp != (manager.slot.stat().st_dev, manager.slot.stat().st_ino)
                    or read_record_snapshot(root / "owner.json") != (owner, digest, owner_stamp)):
                raise StorageError("original recovery custody changed")
        audit = {"schema": SCHEMA, "operation": "same_host_reboot_recovery",
                 "owner_sha256": digest, "source_id": manager.source_id,
                 "recovered": False, "reason": "prior_boot_processes_cannot_survive" if reboot
                 else "original_terminal_closure_and_owner_dead"}
        write_record(root / "maintenance.json", audit)
        def complete_original_recovery(preservation_check=None):
            # None executes the original cache-recovery operations/checks in
            # their original order. Extra checks belong only to the opt-in.
            if preservation_check is not None:
                preservation_check()
            verify_recovery()
            if manager._scoped_lifecycle:
                # OS tree end is not a device fence. Record a durable retained
                # generation before closing this exact owner; later managers
                # isolate their scratch/target and never auto-delete it.
                checked(manager.scratch_root)
                manager.scratch_root.mkdir(exist_ok=True)
                checked(manager.scratch_root)
                marker = manager._retained_scratch_marker(manager.token)
                if os.path.lexists(marker):
                    prior, _, _ = read_record_snapshot(marker, schema="rovezero.retained-scoped-scratch.v1")
                    expected = {"schema": "rovezero.retained-scoped-scratch.v1", "token": manager.token,
                                "scratch": str(manager.scratch), "tree_gone": True,
                                "physical_completion": "not_inferred",
                                "no_child_dispatched": prior.get("no_child_dispatched")}
                    if type(expected["no_child_dispatched"]) is not bool:
                        raise StorageError("retained scratch completion is invalid; preserve owner")
                    if prior != expected:
                        raise StorageError("retained scratch receipt differs; preserve owner")
                else:
                    manager._record_retained_scratch(not_dispatched=False)
                if preservation_check is not None:
                    preservation_check()
                verify_recovery()
                record = {**matches[0][1], "state": "complete", "finished_ns": time.time_ns()}
                write_record(manager.slot / "record.json", record)
                audit.update(recovered=True, regenerable_cache_removed=False, temporary_removed=False,
                             physical_completion="unknown_preserved")
                write_record(root / "maintenance.json", audit)
                verify_recovery()
                if read_record(root / "owner.json") != owner:
                    raise StorageError("recovery owner changed; preserve lock")
                if preservation_check is not None:
                    preservation_check()
                (root / "owner.json").unlink()
                return audit
            if not manager._remove_with_dependencies(manager.scratch, manager.scratch.parent):
                raise StorageError("recovery scratch required by a registered consumer; preserve it")
            if preservation_check is not None:
                preservation_check()
            verify_recovery()
            if not manager._remove_with_dependencies(manager.target, manager.target.parent):
                raise StorageError("recovery target required by a registered consumer; preserve it")
            if preservation_check is not None:
                preservation_check()
            record = {**matches[0][1], "state": "complete", "finished_ns": time.time_ns()}
            write_record(manager.slot / "record.json", record)
            audit.update(recovered=True, regenerable_cache_removed=True, temporary_removed=True)
            write_record(root / "maintenance.json", audit)
            verify_recovery()
            if read_record(root / "owner.json") != owner:
                raise StorageError("recovery owner changed; preserve lock")
            if preservation_check is not None:
                preservation_check()  # Both actual gates/FDs still held immediately before unlink.
            (root / "owner.json").unlink()
            return audit
        if output_preservation is None:
            from managed_outputs import finalize_supervised_outputs
            finalize_supervised_outputs(root, owner)
            return complete_original_recovery()
        from managed_output_unattributed_preservation import recovery_catalog_exclusion
        with recovery_catalog_exclusion(root, digest, output_preservation, guard, verify_recovery) as preserved:
            audit["output_preservation"] = preserved.audit_fields()
            write_record(root / "maintenance.json", audit)
            return complete_original_recovery(preserved.check)
