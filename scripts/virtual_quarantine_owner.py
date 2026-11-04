"""Keep one original capture process alive after a physical-unknown stop.

This helper never starts, signals, reconstructs or adopts a process.  It is
called by the same supervisor that already owns the Popen and control channel.
Its exceptional wait has no fixed end while the original owner is unresolved;
normal captures do not enter this path.  Process exit is not a GPU fence.
"""
from __future__ import annotations

import os
import time


def retain_original_owner(child, transport, publish, *, poll_interval=1.0,
                          waitid=None, sleeper=time.sleep):
    """Wait for the original owner's release or loss, without signaling it.

    ``publish`` receives the same mutable incident record and should preserve
    it outside the capture output. Its failure is recorded and does not justify
    dropping the original owner. A reported session release is only the same
    child's control observation, never a device-wide completion claim.
    """
    if waitid is None:
        waitid = getattr(os, "waitid", None)
        if waitid is None:
            raise RuntimeError("non-reaping original-owner observation requires Linux waitid")
    record = {
        "state": "quarantined",
        "reason": "original_native_owner_unresolved_after_stop",
        "owner_pid": child.pid,
        "gpu_physical_completion": "unknown",
        "process_tree_completion": "unknown",
        "native_admission": "new_requests_closed_prior_grant_may_exist",
        "observation_errors": [],
        "interruptions": 0,
    }

    def observe_failure(stage, error):
        # A diagnostic failure must not cause the original owner to be dropped.
        if len(record["observation_errors"]) < 16:
            record["observation_errors"].append({
                "stage": stage, "type": type(error).__name__, "message": str(error)[:1024],
            })

    def save():
        try:
            publish(record)
        except BaseException as error:
            observe_failure("publish", error)

    save()
    while True:
        try:
            transport.poll()
        except BaseException as error:
            observe_failure("transport", error)
        state = transport.state
        if (state.get("peer_pid_verified") is True
                and state.get("status") in {"session_released", "owner_recovered", "no_native_entry"}
                and state.get("terminal_observation_valid") is True):
            record["state"] = "original_owner_release_observed"
            record["release_observation"] = state["status"]
            save()
            return record
        try:
            result = waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError as error:
            observe_failure("waitid", error)
            # ECHILD is not evidence that the original process has exited.
            record["state"] = "quarantined_waitability_unknown"
            save()
        except BaseException as error:
            # An unobservable process is not proof of exit or release.
            observe_failure("waitid", error)
        else:
            if result is not None:
                record["state"] = "original_owner_exited_without_release"
                save()
                return record
        try:
            sleeper(poll_interval)
        except BaseException as error:
            # Ctrl-C / normal Python termination is not physical completion.
            record["interruptions"] += 1
            observe_failure("wait", error)
            save()
