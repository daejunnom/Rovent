"""Best-effort maintenance after an output warning.

Registered CPU-only static graphs can be reclaimed after a warning.
Build-cache retirement is reserved for explicit idle maintenance. Output
records and validation receipts are always retained.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import stat
import sys
import time

from managed_storage import MAX_ENTRIES, StorageError, checked, read_record
from storage_guard import CatalogGuard, lifetime, owner_proven_dead, reboot_proves_dead


def _snapshot(root: Path, tick):
    """Bounded metadata-only census, without opening file contents or links."""
    pending = [root]
    rows = {}
    total = 0
    while pending:
        tick()
        item = pending.pop()
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise StorageError("linked cache entry requires separate review")
        if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise StorageError("special cache entry requires separate review")
        if len(rows) >= MAX_ENTRIES:
            raise StorageError("cache census entry bound exceeded")
        identity = (info.st_dev, info.st_ino)
        rows[str(item)] = (identity, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if stat.S_ISDIR(info.st_mode):
            with os.scandir(item) as entries:
                for entry in entries:
                    tick()
                    if len(rows) + len(pending) >= MAX_ENTRIES:
                        raise StorageError("cache census entry bound exceeded")
                    pending.append(Path(entry.path))
        else:
            total += info.st_size
    return rows, total


def _remove_snapshot(root: Path, snapshot, tick):
    """Remove the validated tree in small, deadline-checked operations.

    Open every path component relative to a held directory using O_NOFOLLOW and
    compare its device/inode to the snapshot.  An interrupted pass leaves a
    smaller regenerable tree, rather than crossing into an unrelated path.
    """
    root = Path(root)
    expected_root = snapshot[str(root)]
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_fd = os.open(root.parent, directory_flags)
    root_fd = None
    removed_bytes = 0
    try:
        root_fd = os.open(root.name, directory_flags, dir_fd=parent_fd)
        info = os.fstat(root_fd)
        if (info.st_dev, info.st_ino) != expected_root[0]:
            raise StorageError("cache root changed before cleanup")
        paths = sorted((Path(name) for name in snapshot if name != str(root)),
                       key=lambda p: len(p.relative_to(root).parts), reverse=True)
        for path in paths:
            tick()
            parts = path.relative_to(root).parts
            current = root_fd
            opened = []
            try:
                prefix = root
                for name in parts[:-1]:
                    prefix = prefix / name
                    fd = os.open(name, directory_flags, dir_fd=current)
                    opened.append(fd)
                    stat_row = os.fstat(fd)
                    if (stat_row.st_dev, stat_row.st_ino) != snapshot[str(prefix)][0]:
                        raise StorageError("cache parent changed during cleanup")
                    current = fd
                info = os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
                expected = snapshot[str(path)]
                if ((info.st_dev, info.st_ino) != expected[0]
                        or stat.S_IFMT(info.st_mode) != stat.S_IFMT(expected[1])):
                    raise StorageError("cache entry changed during cleanup")
                if stat.S_ISDIR(info.st_mode):
                    os.rmdir(parts[-1], dir_fd=current)
                elif stat.S_ISREG(info.st_mode):
                    if (info.st_size, info.st_mtime_ns, info.st_ctime_ns) != expected[2:]:
                        raise StorageError("cache file changed during cleanup")
                    os.unlink(parts[-1], dir_fd=current)
                    removed_bytes += info.st_size
                else:
                    raise StorageError("unknown cache entry during cleanup")
            finally:
                for fd in reversed(opened):
                    os.close(fd)
        tick()
        named = os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
        if (named.st_dev, named.st_ino) != expected_root[0]:
            raise StorageError("cache root name changed during cleanup")
        os.rmdir(root.name, dir_fd=parent_fd)
        return removed_bytes
    except BaseException as error:
        # Do not pretend a partially removed tree is fully reclaimed.
        error.regenerable_removed_bytes = removed_bytes
        raise
    finally:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)


def cleanup_completed_builds(manager, *, seconds=30):
    """One bounded attempt after original process cleanup; never raises to caller.

    Called at most once by the completed run after its owner has been released.
    Uncertainty protects each candidate. Separate build-cache retirement is
    deferred to idle; the output ledger itself is never removed.
    """
    try:
        from managed_outputs import canonical_catalog
        from managed_output_regenerable import cleanup_registered_cpu_graphs
        cpu_graphs = cleanup_registered_cpu_graphs(manager, checked(canonical_catalog()),
                                                   seconds=min(seconds / 2, 2))
    except Exception as error:
        cpu_graphs = {"removed_files": 0, "removed_bytes": 0, "protected": 0,
                      "warnings": [type(error).__name__], "output_total_changed": False}
    return {"schema": "rovezero.output-warning-cache-maintenance.v1",
            "status": "completed" if cpu_graphs["removed_files"] else "protected_or_empty",
            "removed_slots": 0, "removed_bytes": 0, "protected_slots": 0,
            "warnings": cpu_graphs["warnings"], "cpu_static_graphs": cpu_graphs,
            "output_total_changed": cpu_graphs["output_total_changed"],
            "build_cache_maintenance": "deferred_to_idle"}


def cleanup_completed_builds_idle(manager, *, seconds=30):
    """Explicit idle maintenance; not run automatically after an output warning."""
    return _cleanup_completed_builds(manager, seconds=seconds, requires_finished_manager=True)


def cleanup_completed_builds_for_output(lease, *, seconds=30):
    """Reclaim explicit CPU graphs after this original direct lease closes."""
    try:
        if (lease.guard.stream is not None or lease.record.get("state") != "retained"
                or lease.record.get("tree_gone") is not True):
            raise StorageError("output lease has not closed with a verified process tree")
        from managed_output_regenerable import cleanup_registered_cpu_graphs
        cpu_graphs = cleanup_registered_cpu_graphs(None, lease.catalog, seconds=min(seconds / 2, 2),
                                                  own_lease=lease)
        return {"schema": "rovezero.output-warning-cache-maintenance.v1",
                "status": "completed" if cpu_graphs["removed_files"] else "protected_or_empty",
                "removed_slots": 0, "removed_bytes": 0, "protected_slots": 0,
                "warnings": cpu_graphs["warnings"], "output_total_changed": cpu_graphs["output_total_changed"],
                "cpu_static_graphs": cpu_graphs, "build_cache_maintenance": "deferred_to_idle"}
    except Exception as error:
        return {"schema": "rovezero.output-warning-cache-maintenance.v1", "status": "unavailable",
                "removed_slots": 0, "removed_bytes": 0, "protected_slots": 0,
                "warnings": [f"{type(error).__name__}: {str(error)[:256]}"],
                "output_total_changed": False,
                "observation_limit": "current-visible-proc-namespace-only; census is not atomic with deletion"}


def _cleanup_completed_builds(manager, *, seconds, requires_finished_manager, catalog_override=None):
    result = {"schema": "rovezero.output-warning-cache-maintenance.v1",
              "status": "protected_or_empty", "removed_slots": 0, "removed_bytes": 0,
              "protected_slots": 0, "warnings": [], "output_total_changed": False,
              "observation_limit": "current-visible-proc-namespace-only; census is not atomic with deletion"}
    try:
        if requires_finished_manager and manager.finished is not True:
            raise StorageError("original managed build has not finished")
        if (sys.platform != "linux" or type(seconds) not in (int, float)
                or not math.isfinite(seconds) or not 0 < seconds <= 120):
            raise StorageError("bounded Linux process census is unavailable")
        until = time.monotonic() + seconds

        def tick():
            if time.monotonic() >= until:
                raise StorageError("cache maintenance observation deadline exceeded")

        root, source = checked(Path(manager.root)), checked(Path(manager.source))
        if not root.is_dir():
            raise StorageError("original build catalog is unavailable")
        from managed_outputs import _catalog_rows, canonical_catalog
        from storage_code_retirement import _process_census

        catalog = checked(canonical_catalog() if catalog_override is None else Path(catalog_override))
        if not catalog.is_dir():
            raise StorageError("output evidence catalog is unavailable")
        with CatalogGuard(root / "catalog.lock") as guard, CatalogGuard(catalog / "catalog.lock") as output_guard:
            tick()
            if (root / "owner.json").exists():
                raise StorageError("build catalog has an active or interrupted owner")
            rows = manager._records()
            # Preserve same-source outputs without an exact outer slot binding;
            # the output lifecycle alone cannot prove GPU physical completion.
            outputs = _catalog_rows(catalog)
            output_guard.assert_held()
            for _, row, _ in outputs:
                if not isinstance(row.get("source"), str) or not Path(row["source"]).is_absolute():
                    raise StorageError("output source provenance unknown")
            slots_by_id = {record["source_id"]: record for _, record in rows}
            now = lifetime()
            for slot, record in rows:
                tick()
                if (slot.name == manager.source_id or record["state"] != "complete"
                        or os.path.normcase(record["source"]) != os.path.normcase(str(source))):
                    result["protected_slots"] += 1
                    continue
                def protects_slot(row, target):
                    if target == slot or target.is_relative_to(slot) or slot.is_relative_to(target):
                        return True
                    if os.path.normcase(row["source"]) != os.path.normcase(str(source)):
                        return False
                    outer = row.get("outer_supervisor")
                    owner = outer.get("owner") if isinstance(outer, dict) else None
                    bound_slot = owner.get("source_id") if isinstance(owner, dict) else None
                    bound_root = outer.get("root") if isinstance(outer, dict) else None
                    if (not isinstance(bound_root, str) or not Path(bound_root).is_absolute()
                            or os.path.normcase(bound_root) != os.path.normcase(str(root))
                            or not isinstance(owner, dict) or owner.get("source") != str(source)
                            or not isinstance(bound_slot, str) or len(bound_slot) != 24
                            or any(c not in "0123456789abcdef" for c in bound_slot)):
                        return True
                    registered = slots_by_id.get(bound_slot)
                    if (not registered or any(owner.get(key) != registered.get(key)
                                              for key in ("schema", "token", "pid", "source_id", "source",
                                                          "lifetime", "root_identity", "purpose"))):
                        return True
                    return bound_slot == record["source_id"]

                if any(protects_slot(row, target) for _, row, target in outputs):
                    result["protected_slots"] += 1
                    continue
                try:
                    if not (reboot_proves_dead(record.get("lifetime"), now)
                            or owner_proven_dead(record["pid"], record.get("lifetime"), now)):
                        result["protected_slots"] += 1
                        continue
                    receipt = read_record(slot / "last-run.json")
                    if (receipt.get("source_id") != record["source_id"]
                            or receipt.get("tree_cleanup_verified") is not True):
                        result["protected_slots"] += 1
                        continue
                    target = slot / "cargo-target"
                    if not target.exists():
                        continue
                    before, amount = _snapshot(target, tick)
                    identities = {entry[0] for entry in before.values()}
                    census = _process_census(identities, set(), tick)
                    if not isinstance(census, dict) or census.get("complete") is not True:
                        raise StorageError("cache process census incomplete")
                    guard.assert_held()
                    output_guard.assert_held()
                    tick()
                    if (root / "owner.json").exists() or read_record(slot / "record.json") != record:
                        raise StorageError("cache owner changed during maintenance")
                    after, _ = _snapshot(target, tick)
                    if before != after:
                        raise StorageError("cache identity changed during maintenance")
                    # Keep the slot and durable records even if deletion is
                    # interrupted.  A future build can regenerate this cache.
                    removed = _remove_snapshot(target, after, tick)
                    result["removed_slots"] += 1
                    result["removed_bytes"] += removed
                    result["status"] = "completed"
                except (OSError, StorageError, RuntimeError, ValueError) as error:
                    result["removed_bytes"] += getattr(error, "regenerable_removed_bytes", 0)
                    result["protected_slots"] += 1
                    result["warnings"].append(f"{type(error).__name__}: {str(error)[:256]}")
    except (OSError, StorageError, RuntimeError, ValueError, KeyError, TypeError) as error:
        result["warnings"].append(f"{type(error).__name__}: {str(error)[:256]}")
        if result["status"] != "completed":
            result["status"] = "unavailable"
    return result
