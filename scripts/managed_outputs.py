"""Bounded, registered run outputs; unique evidence is never an eviction cache.

The catalog serializes producers and reserves their maximum output before they
start. Retained results count toward admission even after expiry. Expiry means
review/archival is due, not permission to delete evidence or mapped GPU inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
import uuid

from managed_storage import (GiB, CatalogBusy, StorageError, canonical_root, checked, tree_size,
                             write_record, read_record_snapshot, warn_output_threshold)
from storage_guard import CatalogGuard, lifetime, reboot_proves_dead

SCHEMA = "rovezero.managed-output.v1"
MAX_RUNS = 1024


class OutputBudgetError(StorageError):
    pass


def _warn_unrelated_output(message):
    """Report a protected independent row; this is not a recovery authority."""
    try:
        print("managed_output_unrelated_warning=" + message, file=sys.stderr)
    except Exception:
        pass


def _scope_owns_row(row, target, supervisor_scope, row_path=None, *, strict=False):
    """Cross-check the bound marker before classifying a directory as foreign."""
    declared = row.get("outer_supervisor") == supervisor_scope
    if row.get("deferred") is True:
        # Without a directory marker this cannot prove a foreign owner.
        return True if declared else None
    try:
        marker = _read(target / ".rovezero-output.json")
    except (StorageError, OSError):
        if declared:
            raise
        _warn_unrelated_output("foreign marker unavailable; custody remains unknown")
        return None
    marker_owned = marker.get("outer_supervisor") == supervisor_scope
    if marker != row:
        if declared or marker_owned:
            if declared != marker_owned:
                raise StorageError("supervised output owner or marker changed")
            # The writer publishes marker and catalog in separate atomic
            # renames. Allow only a short, bounded re-observation of that
            # transition; a persistent mismatch or changed owner stays fatal.
            if row_path is not None:
                for _ in range(2):
                    time.sleep(0.005)
                    refreshed = _read(row_path)
                    refreshed_marker = _read(target / ".rovezero-output.json")
                    if (refreshed.get("output") != row.get("output") or
                            refreshed.get("outer_supervisor") != supervisor_scope):
                        break
                    if refreshed == refreshed_marker:
                        row.clear()
                        row.update(refreshed)
                        return True
            if strict:
                raise StorageError("supervised output owner or marker changed")
            _warn_unrelated_output("live owner marker transition unresolved; finalization will recheck")
            return "unresolved_owned"
        _warn_unrelated_output("foreign owner and marker differ; custody remains unknown")
        return None
    return declared


def _reject_identity_ancestor_overlap(output, target):
    """Reject bind-mounted aliases of an existing protected directory."""
    try:
        protected = target.stat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(protected.st_mode):
        return
    for ancestor in (output.parent, *output.parent.parents):
        try:
            observed = ancestor.stat()
        except FileNotFoundError:
            continue
        if (observed.st_dev, observed.st_ino) == (protected.st_dev, protected.st_ino):
            raise StorageError("overlapping output directory identity")


def canonical_catalog():
    return canonical_root().parent.parent / "storage" / "outputs-v1"


def _read(path):
    return read_record_snapshot(path, schema=SCHEMA)[0]


def identity(path):
    info = checked(path).stat()
    return [info.st_dev, info.st_ino]


def _attach_sampled_identity_observation(error, record_key, target, expected, observed):
    """Best-effort scalar diagnostics; never a custody or recovery authority.

    Called only after the original sample guard failed. No path lookup, stat,
    row copy or identity retry is performed. A malformed/unbounded source stays
    an ordinary original StorageError rather than a manufactured observation.
    """
    try:
        if (type(record_key) is not str or len(record_key) != 64
                or any(character not in "0123456789abcdef" for character in record_key)):
            return
        normalized = str(target)  # Already normalized by the original checked().
        if len(normalized) > 4096:
            return
        for pair in (expected, observed):
            if (type(pair) is not list or len(pair) != 2
                    or any(type(value) is not int or not 0 <= value < 1 << 128 for value in pair)):
                return
        error.managed_output_identity_observation = {
            "schema": "rovezero.managed-output-identity-observation.v1",
            "record_key": record_key,
            "normalized_target": normalized,
            "expected_identity": list(expected),
            "observed_identity": list(observed),
            "read_only_observation": True,
            "identity_equivalence_attested": False,
            "storage_lifetime_authority_issued": False,
            "recovery_authorized": False,
            "deletion_authorized": False,
            "physical_completion_attested": False,
            "performance_evidence": False,
        }
    except Exception:
        # Observation failure cannot replace or suppress the original guard.
        pass


def _catalog_rows(catalog, *, independent_output=None):
    rows = []
    entries = list(catalog.iterdir())
    if len(entries) > MAX_RUNS + 1:
        raise StorageError("output catalog entry bound exceeded; archive review required")
    for path in entries:
        if path.name == "catalog.lock":
            continue
        row = _read(path)
        target = checked(Path(row["output"]))
        key = hashlib.sha256(os.path.normcase(str(target)).encode()).hexdigest()
        pending = row.get("deferred") is True and row.get("identity") is None
        if (path.name != key + ".json" or row.get("key") != key
                or row.get("state") not in {"active", "retained", "interrupted_retained"}):
            raise StorageError("output custody changed or missing; preserve catalog")
        try:
            if ((pending and row["state"] != "active" and target.exists())
                    or (not pending and row.get("identity") != identity(target))
                    or (not row.get("deferred") and _read(target / ".rovezero-output.json") != row)):
                raise StorageError("output custody changed or missing; preserve catalog")
        except (StorageError, OSError) as error:
            if (independent_output is None or independent_output.is_relative_to(target)
                    or target.is_relative_to(independent_output)):
                raise
            # Keep the row in the path census so it cannot be adopted or
            # overwritten. No identity, completion or deletion is inferred.
            _warn_unrelated_output(type(error).__name__ + "; custody retained; total may be incomplete")
        rows.append((path, row, target))
    return rows


class _CreatedOutputDirectory:
    """In-process custody from exclusive mkdir; never reconstructed from a pathname."""
    def __init__(self, parent_fd, parent_identity, name, owner):
        self.parent_fd, self.fd = parent_fd, None
        self.parent_identity, self.name = parent_identity, name
        self.identity, self.directory_identity = None, None
        self.owner, self.sealed = owner, False

    @staticmethod
    def directory_identity_of(info):
        return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid)

    def check(self, output):
        parent = self.directory_identity_of(os.fstat(self.parent_fd))
        named_parent = self.directory_identity_of(checked(output.parent).lstat())
        if parent != self.parent_identity or named_parent != parent:
            raise StorageError("created output parent changed; preserve evidence")
        if self.fd is not None:
            opened = os.fstat(self.fd)
            named = os.stat(self.name, dir_fd=self.parent_fd, follow_symlinks=False)
            if (self.directory_identity_of(opened) != self.directory_identity
                    or self.directory_identity_of(named) != self.directory_identity
                    or not stat.S_ISDIR(opened.st_mode) or opened.st_nlink == 0):
                raise StorageError("created output directory changed; preserve evidence")

    def close(self):
        first = None
        for name in ("fd", "parent_fd"):
            fd = getattr(self, name)
            setattr(self, name, None)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError as error:
                    if first is None:
                        first = error
        if first is not None:
            raise first


class OutputLease:
    def __init__(self, output, source, *, catalog=None, reserve=8 * GiB,
                 total_limit=18 * GiB, retention_days=14, deferred=False, outer_supervised=False,
                 quarantine_receipts=None, quarantine_transition_receipts=None,
                 preservation_session=None, unattributed_preservation=None):
        if unattributed_preservation is not None:
            from managed_output_unattributed_preservation import UnattributedSelection
            if (type(unattributed_preservation) is not UnattributedSelection
                    or unattributed_preservation._bound_deadline is None
                    or preservation_session is not None or quarantine_receipts is not None
                    or quarantine_transition_receipts is not None or outer_supervised):
                raise StorageError('explicit source7 unattributed policy is mutually exclusive')
        if preservation_session is not None:
            from managed_output_preservation_session import PreservationSelection
            if (type(preservation_session) is not PreservationSelection or quarantine_receipts is not None
                    or quarantine_transition_receipts is not None or outer_supervised):
                raise StorageError("explicit preservation selection is mutually exclusive with legacy selectors/outer finalizer")
        if quarantine_transition_receipts is not None:
            if quarantine_receipts is not None:
                raise StorageError("V1 and transition custody selectors are mutually exclusive")
            if not isinstance(quarantine_transition_receipts, (tuple, list)) or len(quarantine_transition_receipts) > MAX_RUNS:
                raise StorageError("bounded explicit transition tip pins required")
            if outer_supervised:
                raise StorageError("outer finalizer does not carry explicit transition tip pins")
        if quarantine_receipts is not None:
            if not isinstance(quarantine_receipts, (tuple, list)) or len(quarantine_receipts) > MAX_RUNS:
                raise StorageError("bounded explicit custody receipt pins required")
            if outer_supervised:
                raise StorageError("outer finalizer does not carry explicit custody receipt pins")
        self.output, self.source = checked(output), checked(source)
        self.catalog = checked(catalog or canonical_catalog())
        if (preservation_session is not None or unattributed_preservation is not None) and self.catalog != checked(canonical_catalog()):
            raise StorageError("preservation policy requires the canonical output catalog")
        if (quarantine_receipts is not None or quarantine_transition_receipts is not None
                or preservation_session is not None or unattributed_preservation is not None):
            from managed_output_quarantine import registry_path
            registry = registry_path(self.catalog)
            if self.output.is_relative_to(registry) or registry.is_relative_to(self.output):
                raise StorageError("output overlaps custody registry; no producer admission")
        for other in (self.source, self.catalog):
            if self.output.is_relative_to(other) or other.is_relative_to(self.output):
                raise StorageError("output, checkout and catalog must be disjoint")
        if (type(reserve) is not int or type(total_limit) is not int
                or not 0 < reserve < 1 << 63 or not 0 < total_limit < 1 << 63
                or not 1 <= retention_days <= 365):
            raise StorageError("finite output reservation and retention required")
        self.reserve, self.total_limit, self.days = reserve, total_limit, retention_days
        self._maintenance_due = False
        self._foreign_usage_snapshot = {}
        self._foreign_estimate_reported = False
        self._last_own_usage = None
        self.deferred = deferred
        self.quarantine_receipts = None if quarantine_receipts is None else tuple(dict(pin) for pin in quarantine_receipts)
        self.quarantine_transition_receipts = (None if quarantine_transition_receipts is None
                                               else tuple(dict(pin) for pin in quarantine_transition_receipts))
        self.preservation_session = preservation_session
        self.unattributed_preservation = unattributed_preservation
        self._preservation_enabled = preservation_session is not None or unattributed_preservation is not None
        self._preservation_journal = None
        self._preservation_error = None
        self._quarantine_view = None
        self.key = hashlib.sha256(os.path.normcase(str(self.output)).encode()).hexdigest()
        self.guard = CatalogGuard(self.catalog / "catalog.lock")
        self.record = None
        self._created_directory = None
        self._directory_binding_error = None
        self._directory_binding_stage = "not_started"
        self.outer = None
        if outer_supervised:
            from managed_storage import read_record
            parent_root = checked(Path(os.environ["RZ_MANAGED_BUILD_ROOT"]))
            parent = read_record(parent_root / "owner.json")
            if parent.get("token") != os.environ.get("RZ_MANAGED_BUILD_TOKEN") or parent["source"] != str(self.source):
                raise StorageError("original outer managed supervisor binding missing")
            self.outer = {"root": str(parent_root), "owner": parent,
                          "scope": "inherited_outer_process_group_or_job"}

    def _rows(self):
        if self.unattributed_preservation is not None:
            from managed_output_unattributed_preservation import observe_fresh_output
            previous, self._quarantine_view = self._quarantine_view, None
            # Retire only the superseded observation's input FDs before the
            # next full corpus/history observation. The lease keeps its guard
            # and append journal; a failed replacement has no old-view fallback.
            if previous is not None:
                previous.close()
            self._quarantine_view = observe_fresh_output(self.catalog,
                self.unattributed_preservation, self.guard, output=self.output,
                source=self.source, producer_record=self.record)
            return self._quarantine_view.rows()
        explicit = self._preservation_enabled or self.quarantine_receipts is not None or self.quarantine_transition_receipts is not None
        rows = _catalog_rows(self.catalog) if explicit else _catalog_rows(self.catalog, independent_output=self.output)
        if self._preservation_enabled:
            from managed_output_preservation_session import observe_new_output
            previous, self._quarantine_view = self._quarantine_view, None
            try:
                self._quarantine_view = observe_new_output(self.catalog, rows, self.preservation_session,
                    self.guard, output=self.output, source=self.source, producer_record=self.record)
            finally:
                if previous is not None:
                    previous.close(sys.exc_info()[1])
            return rows

        elif self.quarantine_transition_receipts is not None:
            from managed_output_quarantine_transition import validate_transitions
            validator, pins = validate_transitions, self.quarantine_transition_receipts
        elif self.quarantine_receipts is not None:
            from managed_output_quarantine import validate_quarantines
            validator, pins = validate_quarantines, self.quarantine_receipts
        else:
            return rows
        previous, self._quarantine_view = self._quarantine_view, None
        try:
            self._quarantine_view = validator(self.catalog, rows, pins, self.guard)
        finally:
            if previous is not None:
                previous.close(sys.exc_info()[1])
        return rows

    def register_cpu_static_packing_artifact(self, pin):
        """Register one weight-free CPU graph, not model or GPU evidence."""
        from managed_output_regenerable import register_cpu_static_packing_artifact
        return register_cpu_static_packing_artifact(self, pin)


    def _check_quarantine_view(self):
        if (self.quarantine_receipts is not None or self.quarantine_transition_receipts is not None
                or self._preservation_enabled):
            if self._quarantine_view is None:
                raise StorageError("explicit custody validation has not completed")
            self._quarantine_view.check()

    def _close_quarantine_view(self, primary=None):
        view, self._quarantine_view = getattr(self, "_quarantine_view", None), None
        if view is not None:
            view.close(primary)

    def _seal_preservation_session(self):
        if not self._preservation_enabled:
            return
        if self._preservation_error is not None:
            raise StorageError("preservation evidence failed; no further payload permission") from self._preservation_error
        # An unbound deferred reservation is metadata only. Its original
        # create_deferred_directory call must publish evidence before return.
        if self.record.get("identity") is None and self.deferred and not self.output.exists():
            return
        try:
            view = self._quarantine_view
            if self._preservation_journal is None:
                self._preservation_journal = view.create_journal(self.record)
            if not view._sealed:
                view.seal(self.record, self._preservation_journal)
        except BaseException as error:
            self._preservation_error = error
            raise

    def _close_preservation_journal(self, primary=None):
        journal, self._preservation_journal = self._preservation_journal, None
        if journal is not None:
            journal.close(primary)

    def _release_guard(self, primary=None):
        try:
            self.guard.release()
        except BaseException:
            if (self.quarantine_receipts is None and self.quarantine_transition_receipts is None
                    and not self._preservation_enabled) or primary is None:
                raise
            if hasattr(primary, "add_note"):
                primary.add_note("custody output catalog gate release also failed")

    def _output_usage(self, rows, *, allow_disappearing=False):
        if self._quarantine_view is not None:
            return self._quarantine_view.usage(rows)
        foreign = [(row, target) for _, row, target in rows if target != self.output]
        self._foreign_usage_snapshot = {row["key"]: row["reserved_bytes"] for row, _ in foreign}
        if foreign and not self._foreign_estimate_reported:
            _warn_unrelated_output("other-owner usage is reserved accounting; actual usage is unknown")
            self._foreign_estimate_reported = True
        own_total = sum(tree_size(target, allow_disappearing=allow_disappearing)
                        for _, _, target in rows if target == self.output)
        self._last_own_usage = own_total
        return own_total + sum(value for value in self._foreign_usage_snapshot.values() if value is not None)

    def acquire(self):
        owner_lifetime = lifetime() if not self.deferred else None
        self.catalog.mkdir(parents=True, exist_ok=True)
        # Maintenance may hold the catalog briefly. Only actual OS lock
        # contention is retried; a failed identity check or any other error
        # remains an immediate failure. Never steal an active owner.
        deadline = time.monotonic() + 2.0
        while True:
            try:
                self.guard.acquire()
                break
            except CatalogBusy:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                time.sleep(min(0.1, remaining))
        try:
            rows = self._rows()
            custody_keys = self._quarantine_view.keys if self._quarantine_view is not None else frozenset()
            # Explicit historical custody selections retain their original
            # stricter contract. In the ordinary path an unrelated interrupted
            # row remains protected by the disjoint-path check below; it is
            # not an authority to cancel another producer.
            explicit_custody = self._quarantine_view is not None
            if explicit_custody and any(row["state"] == "active" and row["key"] not in custody_keys for _, row, _ in rows):
                raise StorageError("interrupted output requires explicit recovery")
            if len(rows) >= MAX_RUNS:
                raise StorageError("output catalog full; archive review required")
            if any(self.output.is_relative_to(target) or target.is_relative_to(self.output)
                   for _, _, target in rows):
                raise StorageError("overlapping output registration")
            for _, _, target in rows:
                _reject_identity_ancestor_overlap(self.output, target)
            used = self._output_usage(rows)
            for _, row, target in rows:
                if not explicit_custody:
                    # The other owner's actual bytes are unknown. Its record
                    # is preserved and its reservation is accounted above.
                    continue
                try:
                    amount = tree_size(target)
                    if amount is None:
                        continue
                    exceeds_reservation = amount > row["reserved_bytes"]
                except (StorageError, OSError) as error:
                    if explicit_custody:
                        raise
                    _warn_unrelated_output(type(error).__name__ + "; custody retained; total is incomplete")
                    continue
                if row["key"] not in custody_keys and exceeds_reservation:
                    if explicit_custody:
                        raise StorageError("previous output exceeds its reservation; archive review required")
                    _warn_unrelated_output("reservation exceeded; retained for owner review")
            warn_output_threshold(used + self.reserve, self.total_limit, scope="output_admission_reserved")
            if self.output.exists():
                raise StorageError("only fresh output can be registered; no historical adoption")
            if not self.deferred:
                self.output.mkdir(parents=True, mode=0o700 if self._preservation_enabled else 0o777)
            now = time.time_ns()
            self.record = {"schema": SCHEMA, "key": self.key, "output": str(self.output),
                "identity": None if self.deferred else identity(self.output), "token": uuid.uuid4().hex,
                "deferred": self.deferred,
                "outer_supervisor": self.outer,
                "source": str(self.source), "pid": os.getpid(),
                "lifetime": lifetime() if self.deferred else owner_lifetime,
                "state": "active", "created_ns": now,
                "review_after_ns": now + self.days * 86400 * 10**9,
                "reserved_bytes": self.reserve, "catalog_limit_bytes": self.total_limit,
                "classification": "retained_evidence", "automatic_expiry_deletion": False}
            if self._preservation_enabled:
                self._quarantine_view.bind_producer(self.record)
            self._write()
            self._seal_preservation_session()
            if self._preservation_enabled:
                # Mandatory journal bytes also consume the original reservation.
                if tree_size(self.output) > self.reserve:
                    raise OutputBudgetError("preservation evidence exceeds original output reservation")
                warn_output_threshold(self._output_usage(self._quarantine_view.rows() if self.unattributed_preservation is not None else _catalog_rows(self.catalog)),
                                      self.total_limit, scope="output_accounted")
            return self
        except BaseException as error:
            try:
                self._close_preservation_journal(error)
            finally:
                try:
                    self._close_quarantine_view(error)
                finally:
                    self._release_guard(error)
            raise

    def create_deferred_directory(self):
        """Create and seal a fresh Linux directory before allowing producer work.

        The acquired original producer must exclusively control the parent name.
        No existing path/FD is adopted. A crash between mkdir and catalog fsync
        remains unrecoverable by the unchanged identity-null recovery guard.
        Publication failure is sticky: preserve the path and primary error, and
        do not start payload/children or call success finish. No rollback occurs.
        """
        self.guard.assert_held()
        if self.record is None or not self.deferred:
            raise StorageError("an acquired deferred output lease is required")
        if self._created_directory is not None or self._directory_binding_error is not None:
            raise StorageError("created directory binding already attempted; preserve evidence")
        try:
            self._directory_binding_stage = "validating_original_reservation"
            if (sys.platform != "linux" or self.record.get("state") != "active"
                    or self.record.get("identity") is not None
                    or self.record.get("deferred") is not True
                    or self.record.get("key") != self.key
                    or self.record.get("output") != str(self.output)
                    or self.record.get("source") != str(self.source)
                    or self.record.get("pid") != os.getpid()
                    or self.record.get("lifetime") != lifetime()):
                raise StorageError("original active Linux deferred producer required")
            path = self.catalog / (self.key + ".json")
            original = read_record_snapshot(path, schema=SCHEMA)
            if original[0] != self.record:
                raise StorageError("output owner changed; preserve all records")
            self.check_budget()
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            owner = (self.record["token"], self.record["pid"], dict(self.record["lifetime"]))
            cap = _CreatedOutputDirectory(None, None, self.output.name, owner)
            self._created_directory = cap
            # Transfer the descriptor before fstat/identity setup can fail.
            cap.parent_fd = os.open(checked(self.output.parent), flags)
            parent_fd = cap.parent_fd
            parent = os.fstat(parent_fd)
            cap.parent_identity = cap.directory_identity_of(parent)
            if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.geteuid()
                    or stat.S_IMODE(parent.st_mode) & 0o022):
                raise StorageError("producer-owned parent without group/other writers required")
            cap.check(self.output)
            try:
                os.stat(cap.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise StorageError("only exclusive fresh directory creation is allowed")
            self._directory_binding_stage = "mkdir_pending"
            os.mkdir(cap.name, 0o700, dir_fd=parent_fd)
            self._directory_binding_stage = "directory_created_unsealed"
            cap.fd = os.open(cap.name, flags, dir_fd=parent_fd)
            directory = os.fstat(cap.fd)
            if (not stat.S_ISDIR(directory.st_mode) or directory.st_uid != os.geteuid()
                    or stat.S_IMODE(directory.st_mode) & 0o077):
                raise StorageError("private original directory descriptor required")
            cap.directory_identity = cap.directory_identity_of(directory)
            cap.identity = [directory.st_dev, directory.st_ino]
            cap.check(self.output)
            os.fsync(cap.fd)
            os.fsync(cap.parent_fd)
            self.guard.assert_held()
            if (read_record_snapshot(path, schema=SCHEMA) != original
                    or owner != (self.record["token"], os.getpid(), lifetime())):
                raise StorageError("original producer changed before directory seal")
            cap.check(self.output)
            proposed = dict(self.record, identity=cap.identity)
            # The deferred row stays outside the payload tree; its actual bytes
            # still obey the existing record bound before any metadata write.
            if len((json.dumps(proposed, sort_keys=True) + "\n").encode()) > 8192:
                raise StorageError("ownership record exceeds bound")
            self._check_quarantine_view()
            self._directory_binding_stage = "catalog_write_pending"
            write_record(path, proposed)
            self._directory_binding_stage = "catalog_written_readback_pending"
            actual, digest, _ = read_record_snapshot(path, schema=SCHEMA)
            if actual != proposed:
                raise StorageError("created output catalog readback changed")
            self.guard.assert_held()
            cap.check(self.output)
            self.record, cap.sealed = proposed, True
            self._directory_binding_sha256 = digest
            self.check_budget()
            self._directory_binding_stage = "sealed"
            return self.output
        except BaseException as error:
            self._directory_binding_error = error
            raise

    def _check_created_directory(self):
        if self._directory_binding_error is not None:
            raise StorageError("directory binding failed; no finish or further work is allowed") from self._directory_binding_error
        cap = self._created_directory
        if cap is not None:
            if (not cap.sealed or self.record["identity"] != cap.identity
                    or cap.owner != (self.record["token"], os.getpid(), lifetime())):
                raise StorageError("original created directory lease changed")
            cap.check(self.output)

    def _close_created_directory(self, primary=None):
        cap = getattr(self, "_created_directory", None)
        if cap is not None:
            try:
                cap.close()
            except OSError as error:
                self._directory_close_error = error
                if primary is None:
                    raise
                if hasattr(primary, "add_note"):
                    primary.add_note("created directory descriptor close also failed")

    def __del__(self):
        try:
            self._close_preservation_journal()
        except BaseException:
            pass
        try:
            self._close_created_directory()
        except BaseException:
            pass  # Descriptor release only; no pathname mutation or row finish.
        try:
            self._close_quarantine_view()
        except BaseException:
            pass

    def _write(self):
        self.guard.assert_held()
        self._check_created_directory()
        self._check_quarantine_view()
        if not self.deferred:
            write_record(self.output / ".rovezero-output.json", self.record)
        write_record(self.catalog / (self.key + ".json"), self.record)
        if (self.quarantine_receipts is not None or self.quarantine_transition_receipts is not None
                or self._preservation_enabled):
            self._check_quarantine_view()  # A late error keeps written metadata; no payload permission.

    def check_budget(self):
        self.guard.assert_held()
        self._check_created_directory()
        if self.record is None:
            raise StorageError("output lease not acquired")
        if (_read(self.catalog / (self.key + ".json")) != self.record
                or (not self.deferred and _read(self.output / ".rovezero-output.json") != self.record)):
            raise StorageError("output owner changed; preserve all records")
        rows = self._rows()
        self._seal_preservation_session()
        actual = self._output_usage(rows, allow_disappearing=True)
        self._maintenance_due |= actual > self.total_limit
        own_usage = (tree_size(self.output, allow_disappearing=True)
                     if self._quarantine_view is not None else self._last_own_usage)
        if own_usage > self.reserve:
            raise OutputBudgetError("managed output byte budget exceeded; evidence retained")
        warn_output_threshold(actual, self.total_limit, scope="output_accounted",
                              observed_bytes=own_usage)
        self._check_created_directory()
        self._check_quarantine_view()

    def finish(self, *, status, tree_gone):
        # No file cleanup is implied by this method. In particular OS exit is
        # never promoted to a provider's physical completion fence.
        error = None
        try:
            try:
                self.check_budget()
            except OutputBudgetError as exc:
                error = exc
            self.record = {**self.record, "state": "retained" if tree_gone is True else "active",
                           "status": "failed_output_budget" if error else str(status), "producer_status": str(status),
                           "tree_gone": tree_gone, "finished_ns": time.time_ns(),
                           "budget_error": str(error) if error else None,
                           "physical_completion": "not_inferred_from_output_lifecycle"}
            if self.deferred and self._created_directory is None and self.output.exists():
                self.record["identity"] = identity(self.output)
            self._write()
            try:
                self.check_budget()  # Terminal metadata itself consumes bytes.
            except OutputBudgetError as late_error:
                error = late_error
                self.record.update(status="failed_output_budget", budget_error=str(late_error))
                self._write()
        finally:
            try:
                self._close_preservation_journal(sys.exc_info()[1] or error)
            finally:
                try:
                    self._close_created_directory(sys.exc_info()[1] or error)
                finally:
                    try:
                        self._close_quarantine_view(sys.exc_info()[1] or error)
                    finally:
                        self._release_guard(sys.exc_info()[1] or error)
        if error:
            raise error
        if self._maintenance_due and tree_gone is True and self.outer is None:
            # The lease guard has been released. This best-effort maintenance
            # never removes output evidence or treats OS exit as a GPU fence.
            try:
                from managed_output_maintenance import cleanup_completed_builds_for_output
                result = cleanup_completed_builds_for_output(self, seconds=5)
                try:
                    print("managed_storage_maintenance=" + json.dumps(result, sort_keys=True), file=sys.stderr)
                except (OSError, ValueError):
                    pass
            except Exception as maintenance_error:
                try:
                    print("managed_storage_maintenance_warning=" + type(maintenance_error).__name__, file=sys.stderr)
                except (OSError, ValueError):
                    pass
        return self.record


def _refuse_custody_registry(catalog):
    # Deny only: legacy recovery never interprets a receipt or adopts custody.
    registry = catalog.with_name(catalog.name + ".interrupted-unbound-custody-v1")
    try:
        registry.lstat()  # A broken symlink/special/empty/partial registry also blocks.
    except FileNotFoundError:
        return
    raise StorageError("custody registry present; legacy output recovery is forbidden")


def recover_output(catalog, key, expected_sha256):
    catalog = checked(catalog)
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise StorageError("exact output catalog key required")
    with CatalogGuard(catalog / "catalog.lock") as guard:
        guard.assert_held()
        _refuse_custody_registry(catalog)
        path = catalog / (key + ".json")
        row, digest, stamp = read_record_snapshot(path, schema=SCHEMA)
        if digest != expected_sha256:
            raise StorageError("output lease digest changed")
        target = checked(Path(row["output"]))
        if (row.get("state") != "active"
                or (row.get("identity") is None and target.exists())
                or (row.get("identity") is not None and row["identity"] != identity(target))
                or (not row.get("deferred") and _read(target / ".rovezero-output.json") != row)
                or not reboot_proves_dead(row.get("lifetime"), lifetime())):
            raise StorageError("interrupted output termination not proven")
        guard.assert_held()
        _refuse_custody_registry(catalog)
        if (read_record_snapshot(path, schema=SCHEMA) != (row, digest, stamp)
                or (row.get("identity") is not None and row["identity"] != identity(target))):
            raise StorageError("output identity changed before recovery commit")
        guard.assert_held()
        _refuse_custody_registry(catalog)
        row.update(state="interrupted_retained", recovery_reason="verified_same_host_reboot",
                   recovered_ns=time.time_ns(), recovery_source_sha256=expected_sha256)
        if not row.get("deferred"):
            write_record(target / ".rovezero-output.json", row)
        write_record(path, row)
        return row


def finalize_supervised_outputs(root, owner, catalog=None, *, foreign_cache=None):
    """Called only by the original supervisor after its owned tree is gone."""
    catalog = checked(catalog or canonical_catalog())
    if not catalog.exists():
        if foreign_cache is not None and foreign_cache.get("__owned_rows__"):
            raise StorageError("previously observed supervised output catalog disappeared")
        return
    with CatalogGuard(catalog / "catalog.lock") as guard:
        expected = {"root": str(root), "owner": owner,
                    "scope": "inherited_outer_process_group_or_job"}
        sample_output_budget(catalog, supervisor_scope=expected, foreign_cache=foreign_cache, strict_scope=True)
        entries = list(catalog.iterdir())
        if len(entries) > MAX_RUNS + 1:
            raise StorageError("output catalog bound exceeded")
        for path in entries:
            if path.name == "catalog.lock":
                continue
            try:
                row = _read(path)
            except (StorageError, OSError):
                if foreign_cache is None or path.name in foreign_cache.get("__owned_rows__", set()):
                    raise
                foreign_cache["__unknown_ownership__"] = True
                _warn_unrelated_output("unreadable output row retained; ownership unknown")
                continue
            try:
                target = checked(Path(row["output"]))
            except (StorageError, OSError):
                if (foreign_cache is None or row.get("outer_supervisor") == expected
                        or path.name in foreign_cache.get("__owned_rows__", set())):
                    raise
                foreign_cache["__unknown_ownership__"] = True
                _warn_unrelated_output("unreadable output path retained; ownership unknown")
                continue
            if not _scope_owns_row(row, target, expected, path, strict=True) or row.get("state") != "active":
                continue
            if row.get("identity") != identity(target) or _read(target / ".rovezero-output.json") != row:
                raise StorageError("supervised output custody changed; preserve it")
            row.update(state="retained", tree_gone=True, closure="original_outer_supervisor",
                       physical_completion="not_inferred_from_output_lifecycle")
            guard.assert_held()
            write_record(target / ".rovezero-output.json", row)
            write_record(path, row)
        sample_output_budget(catalog, supervisor_scope=expected, foreign_cache=foreign_cache, strict_scope=True)


def sample_output_budget(catalog=None, *, quarantine_receipts=None, quarantine_guard=None,
                         quarantine_transition_receipts=None, supervisor_scope=None,
                         foreign_cache=None, strict_scope=False):
    """Sample lengths, never a quota or physical completion witness.

    V1 calls remain read-only without acquiring the producer's held gate. Aware
    calls use an explicit original guard (live owner: prefer check_budget); only
    independent sampling may omit it and acquire that gate. A supervisor scope
    isolates unrelated retained output errors. With a foreign cache, unrelated
    totals are estimates and may omit unreadable rows; the caller's own output
    remains sampled on every call. Busy/own-custody errors propagate.
    """
    if supervisor_scope is not None and (type(supervisor_scope) is not dict or
            quarantine_receipts is not None or quarantine_transition_receipts is not None or quarantine_guard is not None):
        raise StorageError("supervisor scope cannot be combined with custody selectors")
    if foreign_cache is not None and (supervisor_scope is None or type(foreign_cache) is not dict):
        raise StorageError("foreign accounting cache requires an explicit supervisor scope")
    if quarantine_receipts is not None and quarantine_transition_receipts is not None:
        raise StorageError("V1 and transition custody selectors are mutually exclusive")
    catalog = checked(catalog or canonical_catalog())
    if quarantine_transition_receipts is not None:
        from managed_output_quarantine import original_custody_gate
        from managed_output_quarantine_transition import validate_transitions
        if quarantine_guard is None:
            with original_custody_gate(catalog) as held:
                return sample_output_budget(catalog, quarantine_guard=held,
                                            quarantine_transition_receipts=quarantine_transition_receipts)
        view = validate_transitions(catalog, _catalog_rows(catalog), quarantine_transition_receipts, quarantine_guard)
        try:
            total = view.usage(_catalog_rows(catalog))
            view.check()
            warn_output_threshold(total, scope="output_accounted")
            return total
        finally:
            view.close(sys.exc_info()[1])
    if quarantine_receipts is not None:
        from managed_output_quarantine import original_custody_gate, validate_quarantines
        if quarantine_guard is None:
            with original_custody_gate(catalog) as held:
                return sample_output_budget(catalog, quarantine_receipts=quarantine_receipts, quarantine_guard=held)
        view = validate_quarantines(catalog, _catalog_rows(catalog), quarantine_receipts, quarantine_guard)
        try:
            total = view.usage(_catalog_rows(catalog))
            view.check()
            warn_output_threshold(total, scope="output_accounted")
            return total
        finally:
            view.close(sys.exc_info()[1])
    if quarantine_guard is not None:
        raise StorageError("custody guard without explicit receipt pins is not a typed selection")
    if not catalog.exists():
        if foreign_cache is not None and foreign_cache.get("__owned_rows__"):
            raise StorageError("previously observed supervised output catalog disappeared")
        return 0
    total = 0
    observed_bytes = 0
    owned_seen = set()
    entries = list(catalog.iterdir())
    if len(entries) > MAX_RUNS + 1:
        raise StorageError("output catalog entry bound exceeded")
    for path in entries:
        if path.name == "catalog.lock":
            continue
        try:
            row = _read(path)
        except (StorageError, OSError) as error:
            previously_owned = foreign_cache.get("__owned_rows__", set()) if foreign_cache is not None else set()
            if supervisor_scope is None or (strict_scope and (foreign_cache is None or path.name in previously_owned)):
                raise
            _warn_unrelated_output(type(error).__name__ + "; live catalog observation unknown; finalization will recheck")
            if foreign_cache is not None:
                foreign_cache["__unknown_ownership__"] = True
                if path.name in foreign_cache.get("__owned_rows__", set()):
                    owned_seen.add(path.name)
            continue
        try:
            target = checked(Path(row["output"]))
        except (StorageError, OSError) as error:
            previously_owned = foreign_cache.get("__owned_rows__", set()) if foreign_cache is not None else set()
            if (supervisor_scope is None or row.get("outer_supervisor") == supervisor_scope
                    or path.name in previously_owned):
                raise
            _warn_unrelated_output(type(error).__name__ + "; protected foreign path; actual usage unknown")
            if foreign_cache is not None:
                foreign_cache["__unknown_ownership__"] = True
            total += row["reserved_bytes"]
            continue
        owned = (_scope_owns_row(row, target, supervisor_scope, path)
                 if supervisor_scope is not None else False)
        if owned == "unresolved_owned":
            if strict_scope:
                raise StorageError("supervised output owner or marker changed")
            owned_seen.add(path.name)
            if foreign_cache is not None:
                foreign_cache["__unknown_ownership__"] = True
            total += row["reserved_bytes"]
            continue
        if owned is None and foreign_cache is not None:
            foreign_cache["__unknown_ownership__"] = True
        if owned:
            owned_seen.add(path.name)
        unrelated = supervisor_scope is not None and not owned
        if unrelated:
            if foreign_cache is not None and not foreign_cache.get("__estimate_reported__"):
                _warn_unrelated_output("other-owner usage is reserved accounting; actual usage is unknown")
                foreign_cache["__estimate_reported__"] = True
            total += row["reserved_bytes"]
            continue
        try:
            if row.get("identity") is not None and row["identity"] != (observed_identity := identity(target)):
                error = StorageError("live output identity changed")
                _attach_sampled_identity_observation(error, row.get("key"), target,
                                                     row["identity"], observed_identity)
                raise error
            amount = tree_size(target, allow_disappearing=True)
        except (StorageError, OSError) as error:
            if not unrelated:
                raise
            _warn_unrelated_output(type(error).__name__ + "; retained; total is incomplete")
            continue
        if amount > row["reserved_bytes"]:
            if not unrelated:
                raise StorageError("managed output exceeds reservation")
            _warn_unrelated_output("reservation exceeded; retained for owner review")
        total += amount
        observed_bytes += amount
    if supervisor_scope is not None and foreign_cache is not None:
        previously_owned = foreign_cache.setdefault("__owned_rows__", set())
        if previously_owned - owned_seen:
            raise StorageError("previously observed supervised output disappeared or changed owner")
        previously_owned.update(owned_seen)
    warn_output_threshold(total, scope="output_accounted", observed_bytes=observed_bytes)
    return total


def publish_runtime(source, store, *, limit=4 * GiB):
    """One immutable CPU DSO per digest, with bounded aggregate admission.

    This is the source-library store, not ORT's extracted loader cache. No
    model, result or original runtime is evicted to make room.
    """
    source, store = checked(source), checked(store)
    if store == source or store.is_relative_to(source) or source.is_relative_to(store):
        raise StorageError("source runtime and shared store must be disjoint")
    store.mkdir(parents=True, exist_ok=True)
    with CatalogGuard(store / "catalog.lock") as guard:
        before = source.stat()
        if not source.is_file() or not 0 < before.st_size <= GiB:
            raise StorageError("bounded runtime file required")
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        digest = digest.hexdigest()
        folder = checked(store / digest)
        target = checked(folder / "libonnxruntime.so.1.22.0")
        if target.exists():
            target_info = target.stat()
            if (not stat.S_ISREG(target_info.st_mode) or target_info.st_nlink != 1
                    or target_info.st_size != before.st_size):
                raise StorageError("immutable runtime store type or size mismatch")
            actual = hashlib.sha256()
            with target.open("rb") as stream:
                remaining = before.st_size
                while chunk := stream.read(min(1024 * 1024, remaining + 1)):
                    remaining -= len(chunk)
                    if remaining < 0:
                        raise StorageError("runtime store grew while reading")
                    actual.update(chunk)
            if actual.hexdigest() != digest or target.stat().st_size != before.st_size:
                raise StorageError("immutable runtime store digest mismatch")
        else:
            if tree_size(store) + before.st_size > limit:
                raise StorageError("shared source runtime store budget exhausted; preserve assets")
            guard.assert_held()
            folder.mkdir(exist_ok=True)
            # An interrupted exclusive copy stays visible as invalid; never
            # repair/overwrite it under an already registered identity.
            with source.open("rb") as origin, target.open("xb") as destination:
                copied = hashlib.sha256()
                while chunk := origin.read(1024 * 1024):
                    destination.write(chunk)
                    copied.update(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            if copied.hexdigest() != digest:
                raise StorageError("runtime source changed during copy")
        after = source.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise StorageError("runtime source identity changed")
        if tree_size(store) > limit:
            raise StorageError("shared source runtime store budget exceeded")
        guard.assert_held()
        return target, digest
