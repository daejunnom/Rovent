#!/usr/bin/env python3
"""Bounded, read-only aggregate storage observation; never manages ownership.

Run outside catalog locks, preferably in a separately supervised worker.
Roots are explicit.  A discovery parent contributes only its immediate
``rovezero-*`` directories.  This is a best-effort snapshot of regular-file
allocation, not filesystem, VHDX, cloud, or host-disk accounting.  A soft
deadline cannot interrupt an OS call that is already blocked.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import stat
import sys
import time

GiB = 1024 ** 3
DEFAULT_THRESHOLD = 18 * GiB
SCHEMA = "rovezero.total-usage-observation.v1"

# Never stat or open these names.  Skipping a possible subtree makes the
# observation partial rather than implying that it consumes no space.
_PRIVATE_DIRS = frozenset({".ssh", ".gnupg", ".aws", ".azure", ".kube", ".pki", ".config", ".netrc"})
_PRIVATE_PARTS = ("credential", "password", "secret", "api_key", "api-key",
                  "apikey", "private_key", "id_rsa", "id_ed25519", "id_ecdsa",
                  "service-account", "service_account")
_PRIVATE_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".jks", ".kdbx")


def _private_name(name: str) -> bool:
    name = name.lower()
    return (name in _PRIVATE_DIRS or name == ".env" or name.startswith(".env.")
            or name.endswith(_PRIVATE_SUFFIXES)
            or any(part in name for part in _PRIVATE_PARTS))


def _private_path(path: Path) -> bool:
    return any(_private_name(part) for part in path.parts if part not in (path.anchor, ""))


def _is_reparse(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _identity(info: os.stat_result):
    # A zero inode is not reliable enough to avoid double-counting.
    return (info.st_dev, info.st_ino) if info.st_ino else None


@lru_cache(maxsize=1)
def _windows_api():
    """Load metadata-only Win32 calls lazily; no file contents are read."""
    import ctypes
    from ctypes import wintypes

    class FileStandardInfo(ctypes.Structure):
        _fields_ = [("allocation_size", ctypes.c_longlong),
                    ("end_of_file", ctypes.c_longlong),
                    ("number_of_links", wintypes.DWORD),
                    ("delete_pending", ctypes.c_ubyte),
                    ("directory", ctypes.c_ubyte)]

    class FileIdInfo(ctypes.Structure):
        _fields_ = [("volume_serial", ctypes.c_ulonglong),
                    ("file_id", ctypes.c_ubyte * 16)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel32.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    query = kernel32.GetFileInformationByHandleEx
    query.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    query.restype = wintypes.BOOL
    close = kernel32.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    return ctypes, create, query, close, FileStandardInfo, FileIdInfo


def _windows_allocation_bytes(path: Path, expected: os.stat_result):
    """Get FILE_STANDARD_INFO allocation after checking the handle's file ID.

    FILE_READ_ATTRIBUTES and OPEN_REPARSE_POINT avoid reading contents or
    following the final symlink.  Unsupported filesystems, inaccessible
    metadata, or changed identities return None and make coverage partial.
    """
    try:
        ctypes, create, query, close, standard_type, id_type = _windows_api()
        # FILE_READ_ATTRIBUTES, share read/write/delete, OPEN_EXISTING,
        # FILE_FLAG_OPEN_REPARSE_POINT.
        handle = create(str(path), 0x80, 7, None, 3, 0x00200000, None)
        if handle == ctypes.c_void_p(-1).value:
            return None
        try:
            standard, file_id = standard_type(), id_type()
            if (not query(handle, 1, ctypes.byref(standard), ctypes.sizeof(standard))
                    or not query(handle, 18, ctypes.byref(file_id), ctypes.sizeof(file_id))):
                return None
            current = path.lstat()
            actual_id = int.from_bytes(bytes(file_id.file_id), "little")
            if (_is_reparse(current) or _identity(current) != _identity(expected)
                    or actual_id != expected.st_ino
                    or standard.directory or standard.delete_pending
                    or standard.end_of_file != expected.st_size
                    or standard.allocation_size < 0):
                return None
            return standard.allocation_size
        finally:
            close(handle)
    except (OSError, AttributeError, ImportError):
        return None


def _allocated_bytes(path: Path, info: os.stat_result):
    blocks = getattr(info, "st_blocks", None)
    if blocks is not None and blocks >= 0:
        return blocks * 512
    if os.name == "nt":
        return _windows_allocation_bytes(path, info)
    return None


def _safe_root(path: Path) -> None:
    """Reject links/junctions in an explicit root before directory traversal."""
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if _is_reparse(current.lstat()):
            raise OSError("root contains a symbolic link or reparse point")


def _same_device_subtree(parent: Path, child: Path):
    """True only if every component to the nested root stays on one device.

    Comparing endpoint devices alone misses an intervening mount.  A link,
    missing component, permission error, or unreliable identity is unknown.
    """
    try:
        _safe_root(parent)
        initial = parent.lstat()
        if not stat.S_ISDIR(initial.st_mode) or _identity(initial) is None:
            return None
        current = parent
        for part in child.relative_to(parent).parts:
            current /= part
            info = current.lstat()
            if _is_reparse(info) or not stat.S_ISDIR(info.st_mode) or _identity(info) is None:
                return None
            if info.st_dev != initial.st_dev:
                return False
        return True
    except (OSError, ValueError):
        return None


def _entry_stat(directory: Path, item):
    info = item.stat(follow_symlinks=False)
    # On Windows DirEntry.stat may omit the inode; a path stat supplies it.
    if os.name == "nt" and _identity(info) is None:
        info = (directory / item.name).lstat()
    return info


def _open_directory(path: Path, expected: os.stat_result):
    """Open a checked POSIX directory without following its final component.

    Windows lacks a portable no-follow directory fd; check immediately before
    scandir there.  Neither branch makes concurrent path replacement atomic.
    """
    if os.name == "nt":
        actual = path.lstat()
        if _is_reparse(actual) or _identity(actual) != _identity(expected):
            raise OSError("directory identity changed")
        return os.scandir(path), None
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        actual = os.fstat(fd)
        if not stat.S_ISDIR(actual.st_mode) or _identity(actual) != _identity(expected):
            raise OSError("directory identity changed")
        return os.scandir(fd), fd
    except BaseException:
        os.close(fd)
        raise


def scan_total_usage(roots, *, discover_parents=(), threshold_bytes=DEFAULT_THRESHOLD,
                     max_entries=1_000_000, max_seconds=60.0):
    """Observe explicitly scoped roots, without deleting or modifying any file.

    The global inode set covers overlapping roots and hardlinks across roots on
    one host.  Secret-like names and unvisited mounts are reported as partial.
    Logical and allocated totals are *observed lower bounds*; reflinks/shared
    extents, filesystem metadata, snapshots, and host VHDX allocation are not
    known from portable stat data.  The observation is not atomic.
    """
    if (type(threshold_bytes) is not int or threshold_bytes <= 0
            or type(max_entries) is not int or max_entries <= 0
            or not isinstance(max_seconds, (int, float)) or isinstance(max_seconds, bool)
            or not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("positive finite threshold, entry count, and deadline required")
    started = time.monotonic()
    deadline = started + max_seconds
    errors = Counter()
    stats = Counter()
    root_results = []
    candidates = []
    seen_roots = set()
    seen_dirs = set()
    opened_dirs = set()
    seen_files = set()
    pending_links = []
    allocated = 0
    logical = 0
    # The deadline includes discovery and cannot be reset by another root.
    def bounded():
        if time.monotonic() >= deadline:
            errors["deadline"] += 1
            return False
        if stats["entries"] >= max_entries:
            errors["entry_limit"] += 1
            return False
        return True

    def add_root(raw):
        path = Path(os.path.abspath(os.fspath(raw)))
        if _private_path(path):
            errors["private_root_excluded"] += 1
            return
        if path not in seen_roots:
            seen_roots.add(path)
            candidates.append(path)

    for root in roots:
        add_root(root)
    for raw in discover_parents:
        if not bounded():
            break
        parent = Path(os.path.abspath(os.fspath(raw)))
        if _private_path(parent):
            errors["private_discovery_excluded"] += 1
            continue
        iterator = None
        fd = None
        try:
            _safe_root(parent)
            info = parent.lstat()
            if not stat.S_ISDIR(info.st_mode):
                raise OSError("discovery parent is not a directory")
            iterator, fd = _open_directory(parent, info)
            with iterator:
                for item in iterator:
                    if not bounded():
                        break
                    stats["entries"] += 1
                    if not item.name.startswith("rovezero-"):
                        continue
                    if _private_name(item.name):
                        errors["private_entries_excluded"] += 1
                        continue
                    child = item.stat(follow_symlinks=False)
                    if _is_reparse(child):
                        stats["symlinks_excluded"] += 1
                        # The matching child could identify an unobserved root.
                        errors["discovery_symlink_unresolved"] += 1
                    elif stat.S_ISDIR(child.st_mode):
                        add_root(parent / item.name)
        except OSError:
            errors["discovery_unavailable"] += 1
        finally:
            if fd is not None:
                os.close(fd)
    if not candidates:
        errors["no_roots"] += 1

    # Parent-before-child traversal.  A nested root is already covered only
    # when each intermediate directory is verifiably on the parent's device.
    # Hardlinks and aliases are deduplicated below across all selected roots.
    candidates.sort(key=lambda p: (len(p.parts), os.fspath(p)))
    selected = []
    for candidate in candidates:
        covered = False
        for prior in selected:
            if candidate == prior:
                covered = True
                break
            if candidate.is_relative_to(prior):
                same_device = _same_device_subtree(prior, candidate)
                if same_device is True:
                    covered = True
                    break
                if same_device is None:
                    errors["nested_root_identity_unavailable"] += 1
        if covered:
            stats["overlapping_roots_excluded"] += 1
        else:
            selected.append(candidate)

    for root in selected:
        if not bounded():
            break
        before_errors = sum(errors.values())
        before_allocated, before_logical = allocated, logical
        root_record = {"root": str(root), "scan_device": None, "scan_inode": None,
                       "identity_unchanged": False}
        root_results.append(root_record)
        root_identity = None
        root_opened = False
        try:
            _safe_root(root)
            info = root.lstat()
            if not stat.S_ISDIR(info.st_mode):
                raise OSError("root is not a directory")
            root_identity = _identity(info)
            if root_identity is None:
                raise OSError("root identity is unavailable")
            root_record["scan_device"], root_record["scan_inode"] = root_identity
            device = info.st_dev
            pending = [(root, info)]
            while pending:
                if not bounded():
                    break
                directory, expected = pending.pop()
                identity = _identity(expected)
                if identity is None:
                    errors["unreliable_identity"] += 1
                    continue
                if identity in seen_dirs:
                    stats["directory_aliases_excluded"] += 1
                    continue
                seen_dirs.add(identity)
                fd = None
                try:
                    iterator, fd = _open_directory(directory, expected)
                    opened_dirs.add(identity)
                    if directory == root:
                        root_opened = True
                    with iterator:
                        for item in iterator:
                            if not bounded():
                                break
                            stats["entries"] += 1
                            if _private_name(item.name):
                                errors["private_entries_excluded"] += 1
                                continue
                            try:
                                item_info = _entry_stat(directory, item)
                                if _is_reparse(item_info):
                                    stats["symlinks_excluded"] += 1
                                    # Resolve only after all selected roots have
                                    # been traversed; never traverse this link.
                                    pending_links.append((directory / item.name, root_record))
                                elif item_info.st_dev != device:
                                    if stat.S_ISDIR(item_info.st_mode) and directory / item.name in selected:
                                        # The selected nested root is scanned
                                        # independently under its own device.
                                        stats["registered_mounts_deferred"] += 1
                                    else:
                                        errors["other_mount_excluded"] += 1
                                elif stat.S_ISDIR(item_info.st_mode):
                                    pending.append((directory / item.name, item_info))
                                elif stat.S_ISREG(item_info.st_mode):
                                    file_identity = _identity(item_info)
                                    if file_identity is None:
                                        errors["unreliable_identity"] += 1
                                    elif file_identity in seen_files:
                                        stats["hardlinks_deduplicated"] += 1
                                    else:
                                        seen_files.add(file_identity)
                                        logical += item_info.st_size
                                        file_allocation = _allocated_bytes(directory / item.name, item_info)
                                        if file_allocation is None:
                                            errors["allocated_bytes_unavailable"] += 1
                                        else:
                                            allocated += file_allocation
                                        stats["unique_regular_files"] += 1
                                else:
                                    stats["nonregular_excluded"] += 1
                            except OSError:
                                errors["entry_unavailable"] += 1
                except OSError:
                    errors["directory_unavailable"] += 1
                finally:
                    if fd is not None:
                        os.close(fd)
                if errors["deadline"] or errors["entry_limit"]:
                    break
        except OSError:
            errors["root_unavailable"] += 1
        if root_opened and root_identity is not None:
            try:
                _safe_root(root)
                current = root.lstat()
                root_record["identity_unchanged"] = (
                    stat.S_ISDIR(current.st_mode) and not _is_reparse(current)
                    and _identity(current) == root_identity
                )
                if not root_record["identity_unchanged"]:
                    errors["root_identity_changed"] += 1
            except OSError:
                errors["root_identity_unavailable"] += 1
        else:
            errors["root_not_opened"] += 1
        root_record.update({"allocated_bytes": allocated - before_allocated,
                            "logical_bytes": logical - before_logical,
                            "partial": sum(errors.values()) != before_errors})
        if errors["deadline"] or errors["entry_limit"]:
            break

    # A skipped link is harmless only when its actual non-link target has
    # already been observed under a selected root.  A target outside the
    # selected roots can itself be unregistered owned data, so its location
    # alone cannot prove completeness.  Never open target contents.
    for index, (link, owner) in enumerate(pending_links):
        if not bounded():
            for _, remaining_owner in pending_links[index:]:
                remaining_owner["partial"] = True
            break
        try:
            raw_target = os.readlink(link)
            target = Path(os.path.abspath(os.path.join(link.parent, raw_target)))
            if _private_path(target):
                raise OSError("private link target")
            _safe_root(target)
            target_info = target.lstat()
            target_id = _identity(target_info)
            already_seen = target_id is not None and (target_id in seen_files or target_id in opened_dirs)
            if _is_reparse(target_info) or not already_seen:
                raise OSError("link target scope or identity is unresolved")
            stats["symlinks_verified_seen"] += 1
        except (OSError, ValueError):
            errors["symlink_scope_unresolved"] += 1
            owner["partial"] = True

    partial = bool(errors)
    decision = ("exceeded" if allocated > threshold_bytes else
                "unknown" if partial else "within")
    return {
        "schema": SCHEMA,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "threshold_bytes": threshold_bytes,
        "allocated_bytes_lower_bound": allocated,
        "logical_bytes_lower_bound": logical,
        "decision": decision,
        "coverage": "partial" if partial else "complete_for_scoped_regular_files",
        "consistency": "best_effort_non_atomic",
        "allocated_measure": "sum_of_observed_unique_file_reported_allocations",
        "host_physical_bytes": "unknown",
        "filesystem_metadata_bytes": "unknown",
        "shared_extents": "unknown",
        "roots": root_results,
        "counts": dict(stats),
        "partial_reasons": dict(errors),
    }


def emit_usage_warning(result, stream=None):
    """Emit an advisory warning; delivery failure never affects a producer."""
    if result.get("decision") not in {"exceeded", "unknown"}:
        return False
    try:
        target = sys.stderr if stream is None else stream
        target.write("rovezero_storage_total_warning: " + json.dumps({
            "decision": result["decision"], "coverage": result["coverage"],
            "allocated_measure": result["allocated_measure"],
            "allocated_bytes_lower_bound": result["allocated_bytes_lower_bound"],
            "logical_bytes_lower_bound": result["logical_bytes_lower_bound"],
            "threshold_bytes": result["threshold_bytes"],
            "observed_at_utc": result["observed_at_utc"],
        }, sort_keys=True) + "\n")
    except Exception:
        pass
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", default=[], help="explicit storage root; repeatable")
    parser.add_argument("--discover-parent", action="append", default=[],
                        help="discover only direct rovezero-* directories under this parent")
    parser.add_argument("--threshold-bytes", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--max-entries", type=int, default=1_000_000)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    try:
        result = scan_total_usage(args.root, discover_parents=args.discover_parent,
                                  threshold_bytes=args.threshold_bytes,
                                  max_entries=args.max_entries, max_seconds=args.max_seconds)
        emit_usage_warning(result)
        print(json.dumps(result, sort_keys=True))
    except (OSError, ValueError) as exc:
        # This is a standalone diagnostic: no storage mutation has occurred.
        print("rovezero_storage_total_observation_failed: " + type(exc).__name__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
