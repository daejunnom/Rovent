"""Owned, bounded regenerable build storage; no historical-tree adoption."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
import uuid
from pathlib import Path
from storage_guard import CatalogGuard, lifetime, reboot_proves_dead, owner_proven_dead

GiB = 1024 ** 3
SCHEMA = "rovezero.managed-build.v1"
MAX_ENTRIES = 100_000
DEFAULT_OUTPUT_WARNING_BYTES = 18 * GiB
_warned_output_scopes: set[str] = set()


def warn_output_threshold(total, threshold=DEFAULT_OUTPUT_WARNING_BYTES, *, scope="output", observed_bytes=None):
    """Warn once per over-threshold interval; never grant cleanup authority.

    Diagnostic delivery failures cannot stop the producer or change custody.
    Per-producer reservations and physical completion checks remain separate.
    """
    if type(total) is not int or type(threshold) is not int or total < 0 or threshold <= 0:
        return False
    label = scope if type(scope) is str and scope.isascii() and len(scope) <= 64 else "output"
    if total <= threshold:
        _warned_output_scopes.discard(label)
        return False
    if label not in _warned_output_scopes:
        if len(_warned_output_scopes) < 64:
            _warned_output_scopes.add(label)
        try:
            import sys
            observed = (str(observed_bytes) if type(observed_bytes) is int and observed_bytes >= 0
                        else "unknown")
            sys.stderr.write(
                f"rovezero_storage_warning: scope={label} accounted_bytes={total} "
                f"(may include reservations; not actual disk usage); "
                f"observed_logical_bytes={observed}; remaining_actual_bytes=unknown; "
                f"warning_threshold_bytes={threshold}; safe cleanup is due\n"
            )
        except Exception:
            pass
    return True


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
                 total_limit=18 * GiB, slots_limit=4, purpose=None, output_preservation=None):
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
        self.scratch = self.slot / "tmp"
        self.lock = self.root / "owner.json"
        self.token = uuid.uuid4().hex
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
        return value

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
                "lifetime", "root_identity", "purpose",
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
                    or any(c not in "0123456789abcdef" for c in record["token"])):
                raise StorageError("invalid slot provenance; preserve it")
            allowed = {"record.json", "last-run.json", "cargo-target", "tmp"}
            if any(p.name not in allowed for p in item.iterdir()):
                raise StorageError("unknown slot content; preserve it")
            rows.append((item, record))
        return rows

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
            amount = tree_size(slot)
            remove_tree(slot, slot.parent)
            total -= amount
            count -= 1
        # A still-active or retained owned slot may itself exceed the aggregate
        # threshold.  Warn rather than terminate its unrelated work; this is
        # never authority to delete active/unknown data.  The slot-count
        # boundary and per-slot post-run cleanup remain independent.
        warn_output_threshold(total, self.total_limit, scope="managed_build_catalog",
                              observed_bytes=total)
        if count > self.slots_limit:
            raise StorageError("slot limit cannot be met without deleting active/unknown data")

    def acquire(self) -> "ManagedBuild":
        checked(self.root)
        if self.output_preservation is not None:
            from managed_output_unattributed_preservation import assert_manager_preservation_scope
            assert_manager_preservation_scope(self.root, self.output_preservation)
        self.root.mkdir(parents=True, exist_ok=True)
        self.guard.acquire()
        if any(p.name not in {"owner.json", "slots", "catalog.lock", "maintenance.json", "closure.json"} for p in self.root.iterdir()):
            self.guard.release()
            raise StorageError("unknown build root content; preserve it")
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
                if tree_size(self.slot) > self.slot_limit:
                    remove_tree(self.slot / "cargo-target", self.slot)
                remove_tree(self.scratch, self.slot)
            else:
                self._prune()
                self.slot.mkdir()
            self.record = {**self._identity(), "state": "active", "finished_ns": 0}
            write_record(self.slot / "record.json", self.record)
            self.target.mkdir(exist_ok=True)
            self.scratch.mkdir()
            self._prune()
            self.before = tree_size(self.slot)
            return self
        except BaseException:
            # A failed admission may expose an interrupted existing slot; do not
            # delete that slot. Release only this invocation's verified lock.
            if read_record(self.lock) == self._identity():
                self.lock.unlink()
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
        remove_tree(self.scratch, self.slot)
        reset = tree_size(self.slot) > self.slot_limit
        if reset:
            remove_tree(self.target, self.slot)
        self.record = {**self.record, "state": "complete", "finished_ns": time.time_ns()}
        write_record(self.slot / "record.json", self.record)
        result = {"schema": SCHEMA, "source_id": self.source_id,
                  "command_exit_code": exit_code, "tree_cleanup_verified": tree_gone,
                  "supervisor_error": supervisor_error,
                  "pre_run_bytes": self.before, "observed_end_bytes": observed,
                  "temporary_removed": True, "post_run_budget_reset": reset,
                  "retained_slot_bytes": tree_size(self.slot),
                  "slot_limit_bytes": self.slot_limit,
                  "catalog_limit_bytes": self.total_limit, "maximum_slots": self.slots_limit}
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


def recover_build(root: Path, owner_sha256: str, *, output_preservation=None) -> dict:
    """Explicit exact-owner recovery after reboot or original terminal closure.

    PID absence/reuse cannot establish that no orphan detached from its group.
    Legacy owners without durable lifetime evidence remain preserved.
    """
    root = checked(root)
    if not root.is_dir():
        raise StorageError("existing managed build root required")
    with CatalogGuard(root / "catalog.lock") as guard:
        if any(p.name not in {"owner.json", "slots", "catalog.lock", "maintenance.json", "closure.json"}
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
        manager = ManagedBuild(root, Path(owner["source"]), purpose=owner.get("purpose"))
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
            remove_tree(manager.scratch, manager.slot)
            if preservation_check is not None:
                preservation_check()
            verify_recovery()
            remove_tree(manager.target, manager.slot)
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
