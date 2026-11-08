"""Bounded, original-process custody for unresolved broker executions.

Handles are deliberately in-memory only. Process exit releases OS descriptors;
this module does not claim lock survival across process exit or GPU completion.
It never reconstructs a Popen object from a PID or steals an owner.
Catalogs that share locks or dependencies may still be affected. The supplied
reobservation window bounds group observation only; subsequent finalization
I/O is not covered by that wall-time bound.
"""
from __future__ import annotations

import math
import os
import threading
import time
import uuid


class BrokerQuarantineError(RuntimeError):
    pass


class BrokerQuarantine:
    def __init__(self, maximum=8):
        if type(maximum) is not int or not 1 <= maximum <= 8:
            raise ValueError("bounded in-process quarantine required")
        self._maximum = maximum
        self._lock = threading.Lock()
        self._entries = {}

    def reserve(self):
        """Reserve before dispatch; exhaustion affects this opt-in only."""
        with self._lock:
            if len(self._entries) >= self._maximum:
                raise BrokerQuarantineError("in-process broker quarantine capacity unavailable")
            key = uuid.uuid4().hex
            self._entries[key] = {"pid": os.getpid(), "thread": threading.current_thread(), "state": "reserved"}
            return key

    def _entry(self, key):
        entry = self._entries.get(key)
        if (entry is None or entry["pid"] != os.getpid()
                or entry["thread"] is not threading.current_thread()):
            raise BrokerQuarantineError("original in-process quarantine owner required")
        return entry

    def release_reservation(self, key):
        with self._lock:
            entry = self._entry(key)
            if entry["state"] != "reserved":
                raise BrokerQuarantineError("retained original owner cannot be released as reservation")
            del self._entries[key]

    def retain(self, key, launch, budget_gate, *, reason, finalization_unknown=False):
        with self._lock:
            entry = self._entry(key)
            if entry["state"] != "reserved" or not isinstance(reason, str) or not reason or len(reason) > 256:
                raise BrokerQuarantineError("original reservation and bounded reason required")
            if launch is None or budget_gate is None or type(finalization_unknown) is not bool:
                raise BrokerQuarantineError("original launch and budget owner required")
            entry.update(state="retained", launch=launch, budget_gate=budget_gate, reason=reason,
                         launch_closed=bool(launch._closed), finalization_attempted=finalization_unknown,
                         owner_kind="broker_group")

    def retain_manager(self, key, manager, *, reason, launch=None):
        """Keep an original partially finalized manager without retrying it."""
        with self._lock:
            entry = self._entry(key)
            if (entry["state"] != "reserved" or manager is None or not isinstance(reason, str)
                    or not reason or len(reason) > 256):
                raise BrokerQuarantineError("original reservation and manager required")
            entry.update(state="retained", manager=manager, launch=launch, reason=reason,
                         finalization_attempted=True, owner_kind="manager_finalization_unknown")

    def observation(self, key):
        with self._lock:
            entry = self._entry(key)
            launch = entry.get("launch")
            broker = getattr(launch, "broker", None)
            descriptor_close_unknown = getattr(broker, "descriptor_close_unknown", None)
            return {"handle": key, "state": entry["state"], "scope": "original_process_only",
                    "process_exit_lock_survival": "not_claimed", "physical_completion": "unknown",
                    "owner_kind": entry.get("owner_kind", "reservation"),
                    "descriptor_close_unknown": descriptor_close_unknown,
                    "shared_catalog_guard": "may_block_until_release",
                    "other_catalog_independence": "requires_disjoint_locks_and_dependencies"}

    def reobserve(self, key, *, seconds):
        """Bound group observation only; later finalization I/O may exceed it."""
        if (type(seconds) not in (int, float) or not math.isfinite(seconds)
                or not 0 < seconds <= 30):
            raise ValueError("bounded reobservation time required")
        with self._lock:
            entry = self._entry(key)
            if entry["state"] != "retained" or entry["finalization_attempted"]:
                raise BrokerQuarantineError("original retained owner is not eligible for reobservation")
            launch, gate = entry["launch"], entry["budget_gate"]
        if launch.group._child is None:
            return {"handle": key, "state": "original_popen_unavailable", "retained": True,
                    "physical_completion": "unknown", "window_scope": "group_observation_not_available"}
        gone, result, error = launch.group.reobserve_and_drain(
            absolute_deadline=time.monotonic() + seconds, terminate=False)
        if gone is not True:
            return {"handle": key, "state": "group_unknown", "retained": True,
                    "physical_completion": "unknown", "observation_error": error,
                    "window_scope": "group_observation_only"}
        try:
            if not entry["launch_closed"]:
                if launch.finish(tree_gone=True) is not True:
                    raise BrokerQuarantineError("original launch finalization unavailable")
                entry["launch_closed"] = True
        except BaseException:
            # Descriptor close or owner state may be partially changed: never
            # retry it as though the original transition had not happened.
            entry["finalization_attempted"] = True
            raise
        budget_error = None
        try:
            gate.finish()
        except Exception as failure:
            budget_error = type(failure).__name__
        entry["finalization_attempted"] = True
        try:
            launch.manager.finish(exit_code=125, tree_gone=True,
                                  supervisor_error="original broker group reobserved after unknown")
        except BaseException:
            # Original manager transition may be partial; keep this entry and
            # let its existing owner-specific recovery handle the evidence.
            raise
        with self._lock:
            if self._entry(key) is not entry:
                raise BrokerQuarantineError("original retained owner changed")
            del self._entries[key]
        return {"handle": key, "state": "group_reobserved_owner_finalized", "retained": False,
                "original_exit_code": result, "budget_error": budget_error,
                "physical_completion": "unknown", "window_scope": "group_observation_only"}


IN_PROCESS_BROKER_QUARANTINE = BrokerQuarantine()
