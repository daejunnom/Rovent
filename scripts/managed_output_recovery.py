"""Explicit preservation of one sealed deferred output after a real reboot.

This companion does not change the legacy recover_output admission boundary.
The original catalog gate and current transition view remain mandatory.  No
lease, producer, physical fence, or output payload is reconstructed here.
"""
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time

import managed_output_quarantine as custody
import managed_output_quarantine_transition as transitions
import managed_outputs as outputs
from managed_storage import GiB, StorageError, checked, read_record_snapshot, write_record, warn_output_threshold
from storage_guard import lifetime, reboot_proves_dead


def _need(condition, message):
    if not condition:
        raise StorageError(message)


def _remaining(until):
    remaining = until - time.monotonic()
    _need(remaining > 0, "original sealed recovery deadline expired")
    return remaining


def _life(value):
    _need(type(value) is dict and set(value) == {"host", "boot", "process_birth", "platform"}
          and all(type(item) is str and item for item in value.values()),
          "exact original/current lifetime required")


def _eligible(row, key, digest, expected_sha256, target, now):
    """Pure rejection predicates, not an ownership or reboot capability."""
    _need(digest == expected_sha256, "original catalog raw digest changed")
    _need(row.get("schema") == outputs.SCHEMA and row.get("key") == key
          and row.get("output") == str(target), "canonical original output row required")
    identity = row.get("identity")
    _need(row.get("state") == "active" and row.get("deferred") is True
          and type(identity) is list and len(identity) == 2
          and all(type(item) is int for item in identity)
          and identity[0] >= 0 and identity[1] > 0,
          "only active deferred bound output can be preserved")
    _need(type(row.get("pid")) is int and row["pid"] > 0
          and type(row.get("token")) is str and bool(row["token"]),
          "original producer attribution required")
    _need(type(row.get("reserved_bytes")) is int and 0 < row["reserved_bytes"] < 2**63
          and type(row.get("catalog_limit_bytes")) is int
          and 0 < row["catalog_limit_bytes"] < 2**63,
          "original bounded reservation required")
    _need("tree_gone" not in row or row["tree_gone"] is False,
          "active output must not assert completed tree retirement")
    _need(row.get("physical_completion") in (None, "unknown", "not_inferred",
                                             "not_inferred_from_output_lifecycle"),
          "physical completion cannot be inferred by sealed recovery")
    _life(row.get("lifetime"))
    _life(now)
    _need(reboot_proves_dead(row["lifetime"], now),
          "original same-host real reboot proof required; same-boot death is insufficient")


def _catalog_snapshot(catalog, until):
    """Original canonical parser plus the raw digest/stat of every catalog row."""
    _remaining(until)
    rows = outputs._catalog_rows(catalog)
    snapshots = {}
    for path, row, target in rows:
        _remaining(until)
        snapshot = read_record_snapshot(path, schema=outputs.SCHEMA)
        info = path.lstat()
        _need(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid()
              and info.st_nlink == 1 and snapshot[0] == row,
              "original catalog row owner/raw changed")
        snapshots[row["key"]] = snapshot
    _remaining(until)
    return rows, snapshots


def _disjoint(rows, selected_key, observed, catalog, view, until):
    target = observed.target
    # No payload read: the original snapshot already holds regular-file and
    # directory metadata with no links, cross-device entries or inode aliases.
    identities = {tuple(item["stat"][:2]) for item in observed.snapshot["entries"]}
    for protected in (catalog, custody.registry_path(catalog)):
        _need(not (target.is_relative_to(protected) or protected.is_relative_to(target)),
              "selected output overlaps catalog or custody registry")
    for _, row, other in rows:
        _remaining(until)
        if row["key"] == selected_key:
            continue
        _need(not (target.is_relative_to(other) or other.is_relative_to(target)),
              "selected output overlaps another catalog output")
        if other.exists():
            _need(tuple(custody._stamp(other.lstat())[:2]) not in identities,
                  "selected output aliases another catalog output")
    for other in view.observed:
        other_ids = {tuple(item["stat"][:2]) for item in other.snapshot["entries"]}
        _need(not identities.intersection(other_ids), "selected output aliases quarantined evidence")


def _check_catalog_directory(held, catalog, fd, original_identity):
    _need(custody._did(os.fstat(fd)) == custody._did(checked(catalog).lstat()) == original_identity,
          "original catalog directory FD/name changed")
    held.tick()


def _close_all(resources, primary):
    first = primary
    for resource in reversed(resources):
        if resource is None:
            continue
        try:
            resource.close(first)
        except BaseException as error:
            if first is None:
                first = error
            elif hasattr(first, "add_note"):
                first.add_note("sealed recovery descriptor cleanup also failed: " + repr(error))
    if primary is None and first is not None:
        raise first


def recover_sealed_output_after_reboot(catalog, key, expected_sha256, *,
        quarantine_transition_receipts, current_source_pins,
        root_exclusive_writer=False, apply=False, maximum_seconds=30):
    """Preserve one bound deferred row; only its catalog record may be replaced.

    root_exclusive_writer is an explicit operator exclusion acknowledgement,
    not authority supplied by JSON.  The actual gate, held directories/row,
    actual lifetime and original typed transition validator prove their own
    boundaries.  Dry-run performs the same observations without writing.

    This function never closes or adopts a producer, issues a lease, changes a
    transition, deletes metadata, or attests physical completion.  If the sole
    original write_record call fails after replacement, its first exception is
    retained with commit-stage notes; there is no rollback or success repair.
    """
    _need(root_exclusive_writer is True and type(apply) is bool,
          "explicit exclusive Root maintenance and exact apply boolean required")
    _need(type(key) is str and re.fullmatch(r"[0-9a-f]{64}", key)
          and type(expected_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", expected_sha256),
          "exact original catalog key and raw SHA256 required")
    _need(type(maximum_seconds) in (int, float) and math.isfinite(maximum_seconds)
          and 0 < maximum_seconds <= 30, "one finite sealed recovery window <=30s required")
    _need(type(current_source_pins) is dict and set(current_source_pins) == set(transitions.ROLES),
          "explicit current source5 raw pins required")
    _need(sys.platform == "linux", "sealed custody recovery requires original Linux observations")
    until = time.monotonic() + maximum_seconds
    catalog = checked(Path(catalog))
    # Do not create catalog.lock, a catalog, or any output on a mistaken path.
    info, lock_info = catalog.lstat(), (catalog / "catalog.lock").lstat()
    _need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
          and stat.S_IMODE(info.st_mode) & 0o022 == 0
          and stat.S_ISREG(lock_info.st_mode) and lock_info.st_uid == os.geteuid()
          and lock_info.st_nlink == 1, "existing owner-controlled original catalog and gate required")
    _remaining(until)
    with custody.original_custody_gate(catalog) as guard:
        resources = []
        write_attempted = write_returned = False
        try:
            guard.assert_held()
            _need(custody._did(catalog.lstat()) == custody._did(info)
                  and custody._stamp(os.fstat(guard.stream.fileno())) == custody._stamp(lock_info),
                  "original catalog/gate identity changed before acquisition")
            rows, snapshots = _catalog_snapshot(catalog, until)
            _need(key in snapshots, "explicit original catalog row not found")
            row, digest, record_stamp = snapshots[key]
            target = checked(Path(row["output"]))
            now = lifetime()
            _eligible(row, key, digest, expected_sha256, target, now)
            row_path = catalog / (key + ".json")
            selected_held = custody._Held(_remaining(until))
            selected_held.until = min(selected_held.until, until)
            resources.append(selected_held)
            row_pin = {"path": str(row_path), "bytes": row_path.lstat().st_size, "sha256": digest}
            original_raw = selected_held.read(row_pin, row_path, 8192)
            _need(read_record_snapshot(row_path, schema=outputs.SCHEMA) == (row, digest, record_stamp),
                  "held original catalog snapshot changed")
            view = transitions.validate_transitions(catalog, rows, quarantine_transition_receipts, guard,
                current_source_pins=current_source_pins, maximum_seconds=_remaining(until))
            resources.append(view)
            _need(type(view) is transitions.TransitionValidation and view.guard is guard
                  and view.catalog == catalog and view.require_current is True,
                  "original typed current transition validation required")
            view.held.until = min(view.held.until, until)
            _need(key not in view.keys, "quarantined original unknown row is immutable")
            observation_held = custody._Held(_remaining(until))
            observation_held.until = min(observation_held.until, until)
            resources.append(observation_held)
            catalog_fd = observation_held.directory(catalog)
            catalog_identity = custody._did(os.fstat(catalog_fd))
            _need(catalog_identity == custody._did(info), "original catalog directory changed")
            observed = custody._Observed(observation_held, target)
            _need(observed.identity[:2] == row["identity"], "original sealed directory identity changed")

            def recheck(*, after_write=False, replacement=None):
                _remaining(until)
                guard.assert_held()
                _life(lifetime())
                _need(lifetime() == now, "recovery observer lifetime changed")
                _check_catalog_directory(observation_held, catalog, catalog_fd, catalog_identity)
                view.check()
                observed.check()
                current_rows, current_snapshots = _catalog_snapshot(catalog, until)
                _need(set(current_snapshots) == set(snapshots), "catalog row roster changed")
                for other_key, snapshot in snapshots.items():
                    if after_write and other_key == key:
                        encoded = (json.dumps(replacement, sort_keys=True) + "\n").encode()
                        _need(current_snapshots[key][0] == replacement
                              and current_snapshots[key][1] == hashlib.sha256(encoded).hexdigest()
                              and current_snapshots[key][2][4] == len(encoded),
                              "recovery canonical catalog write readback changed")
                    else:
                        _need(current_snapshots[other_key] == snapshot, "original catalog raw/hash/stat changed")
                if not after_write:
                    selected_held.check()
                    _eligible(row, key, digest, expected_sha256, target, lifetime())
                _disjoint(current_rows, key, observed, catalog, view, until)
                used = view.usage(current_rows)
                warn_output_threshold(used, row["catalog_limit_bytes"], scope="sealed-recovery")
                _need(observed.snapshot["regular_bytes"] <= row["reserved_bytes"],
                      "original sealed recovery reservation exceeded")
                _remaining(until)
                guard.assert_held()
                return used

            used = recheck()
            result = {"status": "eligible_but_not_applied", "applied": False,
                "original_catalog_row": row_pin, "original_catalog_stat": list(record_stamp),
                "output_identity": list(row["identity"]), "catalog_usage_bytes": used,
                "physical_completion": "not_inferred", "row": row}
            if not apply:
                return result
            proposed = dict(row, state="interrupted_retained", recovery_reason="verified_same_host_reboot",
                recovered_ns=time.time_ns(), recovery_source_sha256=expected_sha256,
                physical_completion="not_inferred")
            _need(len((json.dumps(proposed, sort_keys=True) + "\n").encode()) <= 8192,
                  "recovered ownership record exceeds original bound")
            recheck()
            # The selected old FD remains owned until cleanup, but atomic
            # replacement intentionally unlinks its old name.  Never revalidate
            # it as a live producer or run selected_held.check after this point.
            write_attempted = True
            write_record(row_path, proposed)
            write_returned = True
            recheck(after_write=True, replacement=proposed)
            result.update(status="interrupted_retained", applied=True, row=proposed,
                original_catalog_raw_bytes=len(original_raw))
            return result
        except BaseException as error:
            if write_attempted and hasattr(error, "add_note"):
                error.add_note("sealed recovery catalog commit attempted; write_record returned="
                               + str(write_returned) + "; preserve and explicitly inspect the one row; no rollback")
            raise
        finally:
            _close_all(resources, sys.exc_info()[1])


def _unbound_eligible(row, key, digest, expected_sha256, target, now):
    """Separate null-identity predicates; the bound/legacy paths stay unchanged."""
    _need(digest == expected_sha256, "original catalog raw digest changed")
    _need(row.get("schema") == outputs.SCHEMA and row.get("key") == key
          and row.get("output") == str(target), "canonical original output row required")
    _need(row.get("state") == "active" and row.get("deferred") is True
          and "identity" in row and row["identity"] is None,
          "only an active unbound deferred reservation can be preserved")
    _need(type(row.get("pid")) is int and row["pid"] > 0
          and type(row.get("token")) is str and bool(row["token"]),
          "original producer attribution required")
    _need(type(row.get("reserved_bytes")) is int and 0 < row["reserved_bytes"] < 2**63
          and type(row.get("catalog_limit_bytes")) is int
          and 0 < row["catalog_limit_bytes"] < 2**63,
          "original bounded reservation required")
    _need("tree_gone" not in row or row["tree_gone"] is False,
          "unbound recovery must not infer completed tree retirement")
    _need(row.get("physical_completion") in (None, "unknown", "not_inferred",
                                             "not_inferred_from_output_lifecycle"),
          "physical completion cannot be inferred by unbound recovery")
    _life(row.get("lifetime"))
    _life(now)
    _need(reboot_proves_dead(row["lifetime"], now),
          "original same-host real reboot proof required; same-boot death is insufficient")


class _AbsentDeferredOutput:
    """A held original parent and ENOENT observation, never an output capability."""
    def __init__(self, held, target):
        self.held, self.target = held, target
        self.parent_fd = held.directory(checked(target.parent))
        self.parent_identity = custody._did(os.fstat(self.parent_fd))
        self._check_parent()
        self.check()

    def _check_parent(self):
        self.held.tick()
        current = os.fstat(self.parent_fd)
        _need(stat.S_ISDIR(current.st_mode) and current.st_uid == os.geteuid()
              and stat.S_IMODE(current.st_mode) & 0o022 == 0
              and custody._did(current) == self.parent_identity
              == custody._did(checked(self.target.parent).lstat()),
              "unbound output requires its unchanged owner-controlled parent FD/name")

    def check(self):
        self._check_parent()
        try:
            os.stat(self.target.name, dir_fd=self.parent_fd, follow_symlinks=False)
        except OSError as error:
            if error.errno != errno.ENOENT:
                raise
        else:
            raise StorageError("unbound output name exists; preserve it without adoption")
        # Source7 holds the live namespace FD and all parent descriptors and
        # checks the original lifetime/mount digest; no historical owner or
        # old transition boot is promoted to this current observation.
        self.held.check()
        self._check_parent()


def recover_unbound_absent_output_after_reboot(catalog, key, expected_sha256, *,
        unattributed_selection,
        root_exclusive_writer=False, apply=False, maximum_seconds=30):
    """Preserve one failed metadata-only admission after an actual Linux reboot.

    No bound path, producer, lease or physical fence is recovered.  The named
    output must remain absent under its held, non-writable-by-others parent.
    The explicit source7 CatalogView preserves its six unattributed corpora
    and completed historical chain, using current FD/boot/mount observations.
    Only this original, separate catalog row may be replaced. Historical rows,
    transition limits and the legacy/bound recovery contracts are intact.
    The one outer deadline is never renewed, including on commit failure.
    """
    _need(root_exclusive_writer is True and type(apply) is bool,
          "explicit exclusive Root maintenance and exact apply boolean required")
    _need(type(key) is str and re.fullmatch(r"[0-9a-f]{64}", key)
          and type(expected_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", expected_sha256),
          "exact original catalog key and raw SHA256 required")
    _need(type(maximum_seconds) in (int, float) and math.isfinite(maximum_seconds)
          and 0 < maximum_seconds <= 30, "one finite unbound recovery window <=30s required")
    import managed_output_unattributed_preservation as preservation
    _need(type(unattributed_selection) is preservation.UnattributedSelection,
          "explicit typed current source7 preservation selection required")
    _need(sys.platform == "linux", "unbound custody recovery requires original Linux observations")
    until = time.monotonic() + maximum_seconds
    if unattributed_selection._bound_deadline is not None:
        until = min(until, unattributed_selection.original_hard_until)
    catalog = checked(Path(catalog))
    info, lock_info = catalog.lstat(), (catalog / "catalog.lock").lstat()
    _need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
          and stat.S_IMODE(info.st_mode) & 0o022 == 0
          and stat.S_ISREG(lock_info.st_mode) and lock_info.st_uid == os.geteuid()
          and lock_info.st_nlink == 1, "existing owner-controlled original catalog and gate required")
    _remaining(until)
    with custody.original_custody_gate(catalog) as guard:
        resources = []
        write_attempted = write_returned = False
        try:
            guard.assert_held()
            _need(custody._did(catalog.lstat()) == custody._did(info)
                  and custody._stamp(os.fstat(guard.stream.fileno())) == custody._stamp(lock_info),
                  "original catalog/gate identity changed before acquisition")
            view = preservation.observe_catalog(catalog, unattributed_selection, guard, hard_until=until)
            resources.append(view)
            _need(type(view) is preservation.CatalogView and view.guard is guard
                  and view.catalog == catalog and view.selection is unattributed_selection,
                  "original typed source7 same-gate observation required")
            _need(key not in view.keys, "historical and unattributed original rows are immutable")
            view.until = min(view.until, until)
            view.held.until = min(view.held.until, until)
            view.history.held.until = min(view.history.held.until, until)

            def snapshot_rows(rows):
                snapshots = {}
                for path, item, _ in rows:
                    _remaining(until)
                    snapshot = read_record_snapshot(path, schema=outputs.SCHEMA)
                    current = path.lstat()
                    _need(stat.S_ISREG(current.st_mode) and current.st_uid == os.geteuid()
                          and current.st_nlink == 1 and snapshot[0] == item,
                          "original source7 catalog row owner/raw changed")
                    _need(item["key"] not in snapshots, "duplicate catalog row key")
                    snapshots[item["key"]] = snapshot
                _remaining(until)
                return snapshots

            rows = view.rows()
            snapshots = snapshot_rows(rows)
            _need(key in snapshots, "explicit original catalog row not found")
            row, digest, record_stamp = snapshots[key]
            target, now = checked(Path(row["output"])), lifetime()
            _unbound_eligible(row, key, digest, expected_sha256, target, now)
            row_path = catalog / (key + ".json")
            selected_held = custody._Held(_remaining(until))
            selected_held.until = min(selected_held.until, until)
            resources.append(selected_held)
            row_pin = {"path": str(row_path), "bytes": row_path.lstat().st_size, "sha256": digest}
            original_raw = selected_held.read(row_pin, row_path, 8192)
            _need(read_record_snapshot(row_path, schema=outputs.SCHEMA) == (row, digest, record_stamp),
                  "held original catalog snapshot changed")
            observation_held = preservation._Held(_remaining(until), until)
            resources.append(observation_held)
            catalog_fd = observation_held.directory(catalog)
            catalog_identity = custody._did(os.fstat(catalog_fd))
            _need(catalog_identity == custody._did(info), "original catalog directory changed")
            absent = _AbsentDeferredOutput(observation_held, target)

            def recheck(*, after_write=False, replacement=None):
                _remaining(until)
                guard.assert_held()
                _life(lifetime())
                _need(lifetime() == now, "recovery observer lifetime changed")
                _check_catalog_directory(observation_held, catalog, catalog_fd, catalog_identity)
                absent.check()
                current_rows, used = view.checked_usage()
                current_snapshots = snapshot_rows(current_rows)
                _need(set(current_snapshots) == set(snapshots), "catalog row roster changed")
                for other_key, snapshot in snapshots.items():
                    if after_write and other_key == key:
                        encoded = (json.dumps(replacement, sort_keys=True) + "\n").encode()
                        _need(current_snapshots[key][0] == replacement
                              and current_snapshots[key][1] == hashlib.sha256(encoded).hexdigest()
                              and current_snapshots[key][2][4] == len(encoded),
                              "recovery canonical catalog write readback changed")
                    else:
                        _need(current_snapshots[other_key] == snapshot, "original catalog raw/hash/stat changed")
                if not after_write:
                    selected_held.check()
                    _unbound_eligible(row, key, digest, expected_sha256, target, lifetime())
                for other in (catalog, custody.registry_path(catalog),
                              *(other for _, item, other in current_rows if item["key"] != key)):
                    _need(not (target.is_relative_to(other) or other.is_relative_to(target)),
                          "unbound output overlaps protected catalog or output")
                warn_output_threshold(used, row["catalog_limit_bytes"], scope="unbound-recovery")
                # Usage traversals are observations, not authority to skip the
                # final held-parent absence/namespace and current-view checks.
                absent.check()
                view.check()
                _remaining(until)
                guard.assert_held()
                return used

            used = recheck()
            result = {"status": "eligible_but_not_applied", "applied": False,
                "scope": "unbound_absent_reservation_only", "original_catalog_row": row_pin,
                "original_catalog_stat": list(record_stamp), "output_identity": None,
                "output_absent": True, "catalog_usage_bytes": used,
                "physical_completion": "not_inferred", "row": row}
            if not apply:
                return result
            proposed = dict(row, state="interrupted_retained", recovery_reason="verified_same_host_reboot",
                recovered_ns=time.time_ns(), recovery_source_sha256=expected_sha256,
                physical_completion="not_inferred")
            _need(len((json.dumps(proposed, sort_keys=True) + "\n").encode()) <= 8192,
                  "recovered ownership record exceeds original bound")
            recheck()
            # Replacement intentionally retires only the original metadata
            # name.  The old FD is kept for cleanup, never read as a new owner.
            write_attempted = True
            write_record(row_path, proposed)
            write_returned = True
            recheck(after_write=True, replacement=proposed)
            result.update(status="interrupted_retained", applied=True, row=proposed,
                original_catalog_raw_bytes=len(original_raw))
            return result
        except BaseException as error:
            if write_attempted and hasattr(error, "add_note"):
                error.add_note("unbound recovery catalog commit attempted; write_record returned="
                               + str(write_returned) + "; preserve and explicitly inspect the one row; no rollback")
            raise
        finally:
            _close_all(resources, sys.exc_info()[1])
