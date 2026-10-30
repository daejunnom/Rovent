"""Opt-in current exclusion for a fresh output beside immutable old evidence.

The eight-transition history remains historical evidence. This policy does not
publish another transition, relax its current-tip validator, recover an owner,
or infer physical completion. Only a newly observed, short-lived in-process
session may exclude the preserved paths from one new producer's scope.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
from pathlib import Path
import stat
import sys
import time
import uuid

import managed_output_quarantine as custody
import managed_output_quarantine_transition as transitions
from managed_storage import StorageError, checked
from storage_guard import lifetime

SCHEMA = "rovezero.preservation-exclusion-session.v1"
POLICY = "fresh_output_only_preserved_history_excluded"
SOURCE_ROLES = (*transitions.ROLES, "session")
JOURNAL_LIMIT = 1 << 20
JOURNAL_TOTAL_CAP = 16 << 20
MAX_OBSERVATIONS = 4096
_TOKEN = object()


def _need(value, message):
    if value is not True:
        raise StorageError(message)


def _source_paths():
    return {**transitions._source_paths(), "session": Path(__file__).resolve()}


class PreservationSelection:
    """Copied raw input pins, never a current validation or admission token."""

    __slots__ = ("_tips", "_sources", "_window")

    def __setattr__(self, name, value):
        if (name in ("maximum_seconds", "observation_hard_until")
                or (name == "_window" and hasattr(self, "_window"))):
            raise StorageError("original observation bounds are immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in ("maximum_seconds", "observation_hard_until", "_window"):
            raise StorageError("original observation bounds are immutable")
        object.__delattr__(self, name)

    @property
    def maximum_seconds(self):
        return self._window[0]

    @property
    def observation_hard_until(self):
        return self._window[1]

    def __init__(self, historical_tips, current_source_pins, *,
                 root_exclusive_writer=False, maximum_seconds=30,
                 observation_hard_until=None):
        _need(root_exclusive_writer is True, "explicit exclusive Root preservation observation required")
        _need(type(maximum_seconds) in (int, float) and math.isfinite(maximum_seconds)
              and 0 < maximum_seconds, "one finite positive preservation session required")
        # A longer first observation is permitted only with an explicit
        # caller-supplied monotonic bound. The default and all existing calls
        # retain the 30-second limit; a live or expired view is never extended.
        if observation_hard_until is None:
            _need(maximum_seconds <= 30, "longer observation requires an explicit caller deadline")
        else:
            _need(type(observation_hard_until) in (int, float)
                  and math.isfinite(observation_hard_until)
                  and time.monotonic() < observation_hard_until,
                  "future finite caller observation deadline required")
        _need(type(historical_tips) in (tuple, list)
              and 0 < len(historical_tips) <= custody.MAX_RUNS,
              "explicit bounded historical tip pins required")
        _need(type(current_source_pins) is dict and set(current_source_pins) == set(SOURCE_ROLES),
              "explicit current source6 pins required")
        self._tips = tuple(copy.deepcopy(pin) for pin in historical_tips)
        self._sources = copy.deepcopy(current_source_pins)
        self._window = (maximum_seconds, observation_hard_until)
        # Shape validation is not evidence that any pathname was inspected.
        for pin in self._tips:
            _need(type(pin) is dict and type(pin.get("path")) is str,
                  "historical receipt raw pin required")
            path = Path(pin["path"])
            _need(path.is_absolute(), "absolute historical tip path required")
            custody._pin_shape(pin, path, custody.RECEIPT_LIMIT)
        for role, path in _source_paths().items():
            custody._pin_shape(self._sources[role], path, transitions.SOURCE_RAW_LIMIT)


class HistoricalAnchor:
    """Verified preserved history, with no lease/admission/publication methods."""
    def __init__(self, token, history):
        _need(token is _TOKEN, "historical anchor requires actual held history validation")
        self._history = history

    def _check(self):
        self._history.check()


class _EvidenceJournal:
    """One original owner-held append file, not reconstructed from disk JSON."""
    def __init__(self, token, output, record, now, tick):
        _need(token is _TOKEN, "journal requires the actual current session")
        _need(callable(tick), "actual held preservation deadline check required")
        self.output, self.now = output, dict(now)
        self.owner = (record["token"], record["pid"], dict(record["lifetime"]))
        self.fd = self.directory = None
        self.size = self.entries = 0
        self.head = "0" * 64
        self.failed = False
        self.name = ".rovezero-preservation-session.jsonl"
        try:
            tick()
            self.directory = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            self.directory_identity = custody._did(os.fstat(self.directory))
            tick()
            self.fd = os.open(self.name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                              0o600, dir_fd=self.directory)
            tick()
            os.fsync(self.fd)
            tick()
            os.fsync(self.directory)
            tick()
            self.stamp = custody._stamp(os.fstat(self.fd))
            self.check(record)
            tick()
        except BaseException as error:
            self.close(error)
            raise

    def check(self, record):
        _need(not self.failed and self.fd is not None and self.directory is not None,
              "preservation append journal failed or closed; preserve partial evidence")
        _need((record["token"], record["pid"], record["lifetime"]) == self.owner
              and lifetime() == self.now, "original journal producer changed")
        _need(custody._did(os.fstat(self.directory))
              == custody._did(checked(self.output).lstat()) == self.directory_identity,
              "original journal output directory changed")
        opened = os.fstat(self.fd)
        named = os.stat(self.name, dir_fd=self.directory, follow_symlinks=False)
        _need(custody._stamp(opened) == custody._stamp(named) == self.stamp
              and stat.S_ISREG(opened.st_mode) and opened.st_nlink == 1
              and opened.st_uid == os.geteuid() and stat.S_IMODE(opened.st_mode) == 0o600
              and opened.st_size == self.size, "original append journal FD/name/length changed")

    def append(self, record, value, *, tick):
        _need(callable(tick), "actual held preservation deadline check required")
        try:
            tick()
            self.check(record)
            _need(self.entries < MAX_OBSERVATIONS, "preservation observation count exhausted")
            body = {**value, "sequence": self.entries + 1, "previous_sha256": self.head}
            digest = hashlib.sha256(custody._encode(body)).hexdigest()
            raw = custody._encode({"observation": body, "sha256": digest})
            _need(len(raw) <= JOURNAL_LIMIT and self.size + len(raw) <= JOURNAL_TOTAL_CAP,
                  "preservation append journal byte budget exceeded")
            tick()
            _need(os.lseek(self.fd, 0, os.SEEK_END) == self.size, "journal append offset changed")
            tick()
            written = 0
            while written < len(raw):
                tick()
                count = os.write(self.fd, raw[written:])
                tick()
                _need(count > 0, "preservation journal short write")
                written += count
            tick()
            os.fsync(self.fd)
            tick()
            _need(os.pread(self.fd, len(raw), self.size) == raw,
                  "preservation journal append readback changed")
            tick()
            self.size += len(raw)
            self.entries += 1
            self.head = digest
            self.stamp = custody._stamp(os.fstat(self.fd))
            self.check(record)
            tick()
        except BaseException:
            self.failed = True
            raise  # Never truncate, retry a partial append, or repair success.
        return {"path": str(self.output / self.name), "entries": self.entries,
                "bytes": self.size, "last_observation_sha256": self.head}

    def close(self, primary=None):
        first = None
        for name in ("fd", "directory"):
            fd = getattr(self, name, None)
            setattr(self, name, None)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError as error:
                    first = first or error
        if first is not None:
            if primary is None:
                raise first
            if hasattr(primary, "add_note"):
                primary.add_note("preservation journal descriptor close also failed")


def _owner_fields(row, output, source, now):
    import managed_outputs as outputs
    key = hashlib.sha256(os.path.normcase(str(output)).encode()).hexdigest()
    _need(type(row) is dict and row.get("schema") == outputs.SCHEMA and row.get("key") == key
          and row.get("output") == str(output) and row.get("source") == str(source)
          and row.get("pid") == os.getpid() and row.get("lifetime") == now
          and type(row.get("token")) is str and bool(row["token"])
          and row.get("state") in ("active", "retained"),
          "only the original new in-process producer can own a preservation session")
    return row["token"]


class CurrentExclusionSession:
    """Actual held observation; not constructible from a saved JSON report."""
    def __init__(self, token, anchor, selection, catalog, guard, output, source, producer_record):
        _need(token is _TOKEN and type(anchor) is HistoricalAnchor,
              "actual held historical anchor and current session required")
        history = anchor._history
        self._anchor, self._held = anchor, history.held
        self.catalog, self.guard = catalog, guard
        self.output, self.source = output, source
        self.now = lifetime()
        self.session_id = uuid.uuid4().hex
        self.until = self._held.until
        self.keys = frozenset(history.keys)
        self._tips = copy.deepcopy(selection._tips)
        self._sources = copy.deepcopy(selection._sources)
        self._producer_token = None
        self._closed, self._sealed = False, False
        self._parent_fd = self._held.directory(output.parent)
        parent = os.fstat(self._parent_fd)
        _need(stat.S_ISDIR(parent.st_mode) and parent.st_uid == os.geteuid()
              and stat.S_IMODE(parent.st_mode) & 0o077 == 0,
              "private original new-output parent required")
        self._parent_identity = custody._did(parent)
        self._mounts = custody._mounts(self._held, output.parent, output, parent.st_dev)
        if producer_record is not None:
            self.bind_producer(producer_record)

    def bind_producer(self, record):
        token = _owner_fields(record, self.output, self.source, self.now)
        _need(self._producer_token in (None, token), "new producer token cannot change")
        self._producer_token = token

    def _scope(self, rows):
        from managed_output_quarantine import registry_path
        for protected in (self.catalog, registry_path(self.catalog), self.source):
            _need(not (self.output.is_relative_to(protected) or protected.is_relative_to(self.output)),
                  "new output overlaps preserved control/source scope")
        for _, row, target in rows:
            if target == self.output:
                _need(self._producer_token is not None
                      and _owner_fields(row, self.output, self.source, self.now) == self._producer_token,
                      "existing output is not this new original producer")
                continue
            _need(not (self.output.is_relative_to(target) or target.is_relative_to(self.output)),
                  "new output overlaps preserved output/ancestor/descendant")
        self._anchor._check()

    def check(self):
        import managed_outputs as outputs
        _need(not self._closed, "preservation session is closed")
        self._held.tick()  # The original deadline is never moved forward.
        self.guard.assert_held()
        _need(lifetime() == self.now, "current preservation observer lifetime changed")
        _need(custody._did(os.fstat(self._parent_fd))
              == custody._did(checked(self.output.parent).lstat()) == self._parent_identity,
              "new-output held parent changed")
        _need(custody._mounts(self._held, self.output.parent, self.output, self._parent_identity[0])
              == self._mounts, "current preservation namespace/mount changed")
        _need(_source_paths() == {role: Path(pin["path"]) for role, pin in self._sources.items()},
              "current preservation source paths changed")
        self._held.check()  # Also hashes every retained current source FD.
        self._scope(outputs._catalog_rows(self.catalog))
        self.guard.assert_held()

    def usage(self, rows):
        self.check()
        self._scope(rows)
        # Reuses the original reserved charge, registry metadata, and 18GiB cap.
        return self._anchor._history.usage(rows)

    def create_journal(self, record):
        import managed_outputs as outputs
        self.bind_producer(record)
        self.check()
        _need(record.get("identity") == outputs.identity(self.output)
              and outputs._read(self.catalog / (record["key"] + ".json")) == record,
              "original bound new output required before journal creation")
        info = os.fstat(self._held.directory(self.output))
        _need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
              and stat.S_IMODE(info.st_mode) & 0o077 == 0,
              "private new output required before journal creation")
        return _EvidenceJournal(_TOKEN, self.output, record, self.now, self._held.tick)

    def seal(self, record, journal):
        """Persist this actual observation in this producer's fresh output only."""
        import managed_outputs as outputs
        _need(not self._sealed and type(journal) is _EvidenceJournal,
              "actual original owner-held journal and one-time session seal required")
        self.bind_producer(record)
        self.check()
        _need(record.get("identity") == outputs.identity(self.output),
              "new output must be bound before preservation evidence publication")
        actual = outputs._read(self.catalog / (record["key"] + ".json"))
        _need(actual == record, "new original catalog row changed before observation seal")
        directory = self._held.directory(self.output)
        info = os.fstat(directory)
        _need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
              and stat.S_IMODE(info.st_mode) & 0o077 == 0,
              "private original bound output required for observation evidence")
        observations = [{"key": key, "historical_tip": detail["tip_pin"],
                         "historical_generation": detail["generation"],
                         "current_parent_identity": detail["observation"].parent_identity,
                         "current_identity": detail["observation"].identity,
                         "current_mount": detail["observation"].mounts,
                         "snapshot_sha256": hashlib.sha256(custody._encode(detail["tree"])).hexdigest(),
                         "reserved_charge_bytes": detail["charge"]}
                        for key, detail in sorted(self._anchor._history.details.items())]
        value = {"schema": SCHEMA, "policy": POLICY, "session_id": self.session_id,
                 "new_output_key": record["key"], "new_output": str(self.output),
                 "new_producer_token": record["token"], "current_lifetime": dict(self.now),
                 "current_source_pins": copy.deepcopy(self._sources),
                 "observed_monotonic": time.monotonic(), "original_until_monotonic": self.until,
                 "new_output_mount": self._mounts, "preserved": observations,
                 "original_rows_and_transition_history_unchanged": True,
                 "product_deadline_extended": False,
                 "original_generation_custody_reissued": False,
                 "retirement_authority_issued": False, "physical_completion": "unknown_unchanged"}
        _need(journal.output == self.output, "journal belongs to another output")
        pin = journal.append(record, value, tick=self._held.tick)
        self.check()
        self._sealed = True
        return pin

    def close(self, primary=None):
        if not self._closed:
            self._closed = True
            self._anchor._history.close(primary)


def observe_new_output(catalog, rows, selection, guard, *, output, source, producer_record=None):
    """Reobserve the immutable history; publish no transition or old ownership."""
    _need(sys.platform == "linux" and type(selection) is PreservationSelection,
          "explicit typed Linux preservation selection required; JSON is not authority")
    catalog, output, source = checked(Path(catalog)), checked(Path(output)), checked(Path(source))
    from managed_outputs import canonical_catalog
    _need(catalog == checked(canonical_catalog()),
          "preservation policy requires the canonical output catalog")
    _need(Path(guard.path) == catalog / "catalog.lock", "same original catalog gate required")
    guard.assert_held()
    _need(output.is_absolute() and source.is_absolute(), "exact new output/source required")
    if producer_record is None:
        _need(not output.exists(), "only a fresh output may start a preservation session")
    held = (custody._Held(selection.maximum_seconds)
            if selection.observation_hard_until is None
            else custody._Held._for_bounded_session(selection.maximum_seconds,
                                                     selection.observation_hard_until))
    try:
        # This is a historical parser, not validate_transitions and never a
        # current-tip capability. Its returned flag is not modified or handed
        # to OutputLease. The independent current session owns all admission.
        history = transitions._history(held, catalog, rows, selection._tips, guard)
        _need(bool(history.details)
              and all(detail["generation"] > 0 for detail in history.details.values()),
              "registered completed transition history required")
        anchor = HistoricalAnchor(_TOKEN, history)
        for role, path in _source_paths().items():
            try:
                held.read(selection._sources[role], path, transitions.SOURCE_RAW_LIMIT)
            except StorageError as error:
                observation = getattr(error, "custody_metadata_failure_observation", None)
                if type(observation) is dict:
                    observation["domain"] = "current_management_source"
                    observation["source_role"] = role
                raise
        result = CurrentExclusionSession(_TOKEN, anchor, selection, catalog, guard, output, source, producer_record)
        result.check()
        return result
    except BaseException as error:
        held.close(error)
        raise
