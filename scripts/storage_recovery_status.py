"""Read-only initial build-recovery gate observations, never recovery authority.

No manager, owner, lease, lock, closure or output-catalog capability is created.
The observations are not atomic with another process and do not replace the
remaining locked slot/cache/output checks in managed_storage.recover_build.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import re
import stat

from managed_storage import SCHEMA, StorageError, checked, read_record_snapshot
from storage_guard import lifetime, owner_proven_dead, reboot_proves_dead


_SHA = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_ID = re.compile(r"[0-9a-f]{24}\Z")
_LIFETIME_KEYS = {"host", "boot", "process_birth", "platform"}
_ROOT_NAMES = {"owner.json", "slots", "catalog.lock", "maintenance.json", "closure.json"}


def _valid_observer(value):
    return (type(value) is dict and set(value) == _LIFETIME_KEYS
            and all(type(item) is str and 0 < len(item) <= 1024 for item in value.values()))


def _lifetime_view(value):
    if (type(value) is not dict or not _LIFETIME_KEYS <= set(value)
            or any(type(value[key]) is not str or not 0 < len(value[key]) <= 1024 for key in _LIFETIME_KEYS)):
        return None
    result = {key: value[key] for key in sorted(_LIFETIME_KEYS)}
    result.update(process_birth_units="opaque", process_created_utc=None)
    if value["platform"] == "linux":
        result["process_birth_units"] = "proc_start_ticks_since_boot"
    elif value["platform"] == "win32":
        result["process_birth_units"] = "filetime_100ns_since_1601_utc"
        birth = value["process_birth"]
        if birth.isascii() and birth.isdecimal() and len(birth) <= 20:
            ticks = int(birth)
            if 0 <= ticks < 1 << 64:
                try:
                    seconds, fraction = divmod(ticks, 10_000_000)
                    date = datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)
                    result["process_created_utc"] = date.strftime("%Y-%m-%dT%H:%M:%S") + f".{fraction:07d}Z"
                except (OverflowError, ValueError):
                    pass  # Keep the original integer domain; no guessed date.
    return result


def _root_snapshot(root):
    root = checked(root)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise StorageError("existing managed build directory required")
    names = set()
    with os.scandir(root) as entries:
        for entry in entries:
            names.add(entry.name)
            if len(names) > len(_ROOT_NAMES) or entry.name not in _ROOT_NAMES:
                raise StorageError("unknown build root content is preserved")
    stamp = (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
             info.st_mtime_ns, info.st_ctime_ns)
    return stamp, frozenset(names)


def _optional_record(path):
    try:
        checked(path).lstat()
    except FileNotFoundError:
        return None
    return read_record_snapshot(path)


def inspect_build_recovery(root, expected_owner_sha256=None):
    """Observe the existing initial owner/root/lifetime gate without changing it.

    Eligibility is a stable read-only observation of that initial gate only.
    It is neither a recovery permit nor proof that every output has terminated.
    """
    report = {"schema": "rovezero.build-recovery-status.v1", "operation": "inspect_build_recovery",
        "status": "unknown_preserved", "eligibility": "unknown", "root": None,
        "expected_owner_sha256": (expected_owner_sha256 if type(expected_owner_sha256) is str
                                  and _SHA.fullmatch(expected_owner_sha256) else None),
        "owner_sha256": None, "expected_owner_matches": None, "source_id": None, "owner_pid": None,
        "owner_lifetime": None, "observer_lifetime": None, "closure_sha256": None,
        "root_identity_matches": None, "original_closure_matches": None,
        "owner_proven_dead": None, "same_host_reboot_proven": None,
        "snapshot_consistent": False, "observer_lifetime_consistent": False,
        "recovery_eligible_by_existing_gate": None,
        "existing_gate_scope": "initial_exact_owner_root_identity_and_reboot_or_original_closure",
        "remaining_recovery_checks_unperformed": ["catalog_lock", "slot_owner_and_custody",
            "cache_tree_validation", "original_output_finalization"],
        "read_only": True, "observation_is_not_atomic": True,
        "recovery_performed": False, "all_outputs_termination_proven": False,
        "owner_or_lease_created": False, "locks_acquired": False, "closure_created": False,
        "output_catalog_inspected": False, "deletion_performed": False, "issue": None}
    phase = "input"
    try:
        if (expected_owner_sha256 is not None
                and (type(expected_owner_sha256) is not str or _SHA.fullmatch(expected_owner_sha256) is None)):
            raise StorageError("exact expected owner SHA required")
        root = checked(Path(root))
        report["root"] = str(root)
        phase = "observer_lifetime_before"
        current = lifetime()
        if not _valid_observer(current):
            raise StorageError("verified observer lifetime unavailable")
        report["observer_lifetime"] = _lifetime_view(current)
        phase = "root_before"
        root_before = _root_snapshot(root)
        phase = "owner_before"
        owner_before = _optional_record(root / "owner.json")
        phase = "closure_before"
        closure_before = _optional_record(root / "closure.json")
        if owner_before is not None:
            owner, digest, _ = owner_before
            report["owner_sha256"] = digest
            report["expected_owner_matches"] = expected_owner_sha256 is None or digest == expected_owner_sha256
            source_id = owner.get("source_id")
            report["source_id"] = source_id if type(source_id) is str and _SOURCE_ID.fullmatch(source_id) else None
            pid = owner.get("pid")
            report["owner_pid"] = pid if type(pid) is int and pid > 0 else None
            report["owner_lifetime"] = _lifetime_view(owner.get("lifetime"))
            report["closure_sha256"] = None if closure_before is None else closure_before[1]
            report["root_identity_matches"] = owner.get("root_identity") == list(root_before[0][:2])
            report["original_closure_matches"] = (closure_before is not None
                and closure_before[0] == {"schema": SCHEMA, "owner": owner, "tree_gone": True})
            if report["expected_owner_matches"]:
                phase = "original_owner_lifetime_gate"
                report["same_host_reboot_proven"] = reboot_proves_dead(owner.get("lifetime"), current)
                report["owner_proven_dead"] = owner_proven_dead(pid, owner.get("lifetime"), current)
        phase = "root_after"
        root_after = _root_snapshot(root)
        phase = "owner_after"
        owner_after = _optional_record(root / "owner.json")
        phase = "closure_after"
        closure_after = _optional_record(root / "closure.json")
        phase = "observer_lifetime_after"
        after = lifetime()
        report["observer_lifetime_consistent"] = _valid_observer(after) and after == current
        report["snapshot_consistent"] = (root_before == root_after and owner_before == owner_after
                                         and closure_before == closure_after)
        if not report["observer_lifetime_consistent"]:
            report["issue"] = {"phase": phase, "code": "observer_lifetime_changed_or_unknown"}
        elif not report["snapshot_consistent"]:
            report["issue"] = {"phase": "snapshot", "code": "root_owner_or_closure_changed"}
        elif owner_before is None:
            report.update(status="owner_absent", eligibility="not_applicable",
                          recovery_eligible_by_existing_gate=False)
        elif not report["expected_owner_matches"]:
            report.update(status="changed_owner_preserved", eligibility="held",
                          recovery_eligible_by_existing_gate=False)
        else:
            eligible = bool(report["root_identity_matches"] and (report["same_host_reboot_proven"]
                or (report["original_closure_matches"] and report["owner_proven_dead"])))
            report.update(status="recovery_eligible_by_existing_gate" if eligible
                else "held_original_tree_termination_unproven", eligibility="eligible" if eligible else "held",
                recovery_eligible_by_existing_gate=eligible)
    except (OSError, StorageError, ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
        report["issue"] = {"phase": phase, "code": "observation_unavailable_preserved",
                           "error_type": type(error).__name__}
    return report
