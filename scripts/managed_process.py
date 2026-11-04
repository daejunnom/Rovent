"""Finite owned command trees used by the build-storage boundary."""
from __future__ import annotations
import ctypes
import math
import os
import signal
import stat
import subprocess
import time


class ProcessNotSpawned(RuntimeError):
    """Opt-in preflight failed before Popen was invoked; retains original cause."""


def _group_has_live_members(pgid):
    with os.scandir("/proc") as entries:
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            try:
                with open(f"{entry.path}/stat", encoding="utf-8") as stream:
                    fields = stream.read().rsplit(")", 1)[1].split()
                if int(fields[2]) == pgid and fields[0] not in {"Z", "X"}:
                    return True
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (OSError, ValueError, IndexError):
                return True
    return False


def _finish_owned_group(child, on_tick, graceful_deadline, absolute_deadline):
    """Drain this unreaped group while retaining the first cleanup error."""
    tick_enabled = on_tick is not None
    cleanup_error = None

    def tick():
        nonlocal tick_enabled, cleanup_error
        if not tick_enabled:
            return
        try:
            on_tick()
        except BaseException as exc:
            tick_enabled = False
            cleanup_error = {"kind": "callback", "error_type": type(exc).__name__,
                             "message": str(exc)}

    deadline = absolute_deadline if absolute_deadline is not None else time.monotonic() + 3
    try:
        # The V3 child may need the registered cleanup window to finish the
        # original native owner. Keep servicing its channel before signaling.
        while graceful_deadline is not None and time.monotonic() < graceful_deadline:
            tick()
            if not _group_has_live_members(child.pid):
                status = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                if status is not None:
                    return True, child.wait(), cleanup_error
            time.sleep(min(0.05, max(0.0, graceful_deadline - time.monotonic())))
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if time.monotonic() >= deadline:
                return False, None, cleanup_error
            tick()
            if time.monotonic() >= deadline:
                return False, None, cleanup_error
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                break
            if sig == signal.SIGTERM:
                time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
        while time.monotonic() < deadline:
            tick()
            if not _group_has_live_members(child.pid):
                status = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                if status is not None:
                    return True, child.wait(), cleanup_error
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        return False, None, cleanup_error
    except BaseException as exc:
        error = {"kind": "observation", "error_type": type(exc).__name__, "message": str(exc)}
        if cleanup_error is not None:
            cleanup_error["additional_observation_error"] = error
            return False, None, cleanup_error
        return False, None, error


class OwnedProcessGroup:
    """Keep the original Popen and unreaped group identity across an unknown drain.

    This is process-local ownership, not a serializable PID-based recovery token.
    A caller must retain this object and any associated transport and leases.
    """

    def __init__(self):
        self._owner_pid = os.getpid()
        self._child = None
        self._completed = False
        self._exit_code = None
        self.cleanup_error = None
        self.stop_error = None
        self.stop_cause = None
        self.output_handoff_close_error = None

    def _assert_available(self):
        if os.getpid() != self._owner_pid or self._child is not None or self._completed:
            raise ValueError("process group owner is not available in this process")

    def _bind(self, child):
        self._child = child

    def _mark_completed(self, exit_code):
        self._exit_code = exit_code
        self._completed = True

    def reobserve_and_drain(self, *, absolute_deadline, on_tick=None, terminate=False):
        """Bounded same-owner drain; return (group_gone, reaped_exit, error).

        Absolute deadline uses time.monotonic(). A gone process group is not
        itself proof of completion of asynchronous GPU work. The /proc scan
        is limited to the owned process group, not escaped descendants.
        Individual OS
        observations cannot be preempted and can exceed the deadline.
        """
        try:
            valid = (type(absolute_deadline) in (int, float)
                     and math.isfinite(absolute_deadline) and absolute_deadline > 0)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("absolute deadline must be finite positive monotonic seconds")
        if on_tick is not None and not callable(on_tick):
            raise ValueError("on_tick must be callable")
        if type(terminate) is not bool:
            raise ValueError("terminate must be bool")
        if os.getpid() != self._owner_pid:
            raise RuntimeError("process group belongs to a different owner process")
        if self._completed:
            return True, self._exit_code, None
        child = self._child
        if child is None:
            raise RuntimeError("no original process has been attached")
        if time.monotonic() >= absolute_deadline:
            return False, None, {"kind": "deadline", "error_type": "TimeoutError",
                                 "message": "reobservation deadline expired"}
        def tick():
            if on_tick is None:
                return None
            try:
                on_tick()
                return None
            except BaseException as exc:
                return {"kind": "callback", "error_type": type(exc).__name__, "message": str(exc)}
        try:
            # A waitable leader, not a reconstructed PID, pins the group ID.
            os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            for sig in ((signal.SIGTERM, signal.SIGKILL) if terminate else ()):
                if time.monotonic() >= absolute_deadline:
                    return False, None, {"kind": "deadline", "error_type": "TimeoutError",
                                         "message": "reobservation deadline expired"}
                error = tick()
                if error is not None:
                    return False, None, error
                if time.monotonic() >= absolute_deadline:
                    return False, None, {"kind": "deadline", "error_type": "TimeoutError",
                                         "message": "reobservation deadline expired"}
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    break
                if sig == signal.SIGTERM:
                    time.sleep(min(0.2, max(0.0, absolute_deadline - time.monotonic())))
            while time.monotonic() < absolute_deadline:
                error = tick()
                if error is not None:
                    return False, None, error
                if time.monotonic() >= absolute_deadline:
                    return False, None, {"kind": "deadline", "error_type": "TimeoutError",
                                         "message": "reobservation deadline expired"}
                if not _group_has_live_members(child.pid):
                    status = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                    if status is not None:
                        result = child.wait()
                        self._mark_completed(result)
                        return True, result, None
                time.sleep(min(0.05, max(0.0, absolute_deadline - time.monotonic())))
            return False, None, {"kind": "deadline", "error_type": "TimeoutError",
                                 "message": "reobservation deadline expired"}
        except (OSError, ValueError, IndexError) as exc:
            return False, None, {"kind": "observation", "error_type": type(exc).__name__,
                                 "message": str(exc)}


def _check_before_spawn(argv, cwd, pass_fds, stdout):
    try:
        if not argv or not os.path.isabs(os.fspath(argv[0])):
            raise ValueError("preflight requires an absolute executable path")
        executable = os.stat(argv[0])
        if not stat.S_ISREG(executable.st_mode) or not os.access(argv[0], os.X_OK):
            raise ValueError("preflight requires an executable regular file")
        if not stat.S_ISDIR(os.stat(cwd).st_mode):
            raise ValueError("preflight requires an existing directory")
        for fd in pass_fds:
            os.fstat(fd)
        if stdout is not None:
            if stdout == subprocess.PIPE:
                raise ValueError("bounded stdout must be an opened file descriptor")
            os.fstat(stdout if isinstance(stdout, int) else stdout.fileno())
    except (OSError, ValueError, TypeError) as error:
        raise ProcessNotSpawned("original pre-dispatch preflight failed") from error


def run_group(argv, cwd, env, timeout, admission_check, *, pass_fds=(), stdout=None, on_spawn=None,
              verify_before_spawn=False, absolute_deadline=None, deadline_observation=None,
              on_tick=None, owned_group=None, graceful_drain_deadline=None, on_stop=None,
              output_handoff=None, cleanup_absolute_deadline=None):
    if os.name == "nt":
        if (pass_fds or stdout is not None or on_spawn is not None or verify_before_spawn
                or absolute_deadline is not None or deadline_observation is not None
                or on_tick is not None or owned_group is not None or graceful_drain_deadline is not None
                or on_stop is not None or output_handoff is not None
                or cleanup_absolute_deadline is not None):
            raise RuntimeError("explicit descriptor handoff requires Linux")
        return _run_windows(argv, cwd, env, timeout, admission_check)
    if not hasattr(os, "waitid") or not hasattr(os, "WNOWAIT"):
        raise RuntimeError("non-reaping process supervision requires Linux waitid")
    if on_tick is not None and not callable(on_tick):
        raise ValueError("on_tick must be callable")
    if on_stop is not None and (owned_group is None or not callable(on_stop)):
        raise ValueError("on_stop must be callable and requires owned_group")
    if owned_group is not None:
        if type(owned_group) is not OwnedProcessGroup:
            raise ValueError("owned_group must be an OwnedProcessGroup")
        owned_group._assert_available()
    if output_handoff is not None:
        # Import lazily: the handoff module itself names OwnedProcessGroup.
        # Other process users retain their original code and dependencies.
        from managed_output_child_handoff import ManagedOutputChildHandoff, ManagedCollectionOutputHandoffV2
        if (type(output_handoff) not in (ManagedOutputChildHandoff, ManagedCollectionOutputHandoffV2)
                or owned_group is None or output_handoff._group is not owned_group
                or output_handoff._owner_pid != os.getpid()):
            raise ValueError("output handoff requires its original owned process group")
        pass_fds = tuple(pass_fds)
        output_fds = output_handoff.child_fds
        expected_count = 2 if type(output_handoff) is ManagedOutputChildHandoff else 4
        if (len(output_fds) != expected_count or any(type(fd) is not int or fd < 0 for fd in output_fds)
                or len(set(output_fds)) != expected_count
                or any(pass_fds.count(fd) != 1 for fd in output_fds)):
            raise ValueError("exact original output child descriptors must be passed once")
        for fd in output_fds:
            os.fstat(fd)
        previous_on_spawn, previous_on_tick = on_spawn, on_tick

        def on_spawn():
            output_handoff.after_spawn()
            if previous_on_spawn is not None:
                previous_on_spawn()
            output_handoff.confirm_scope_admission()

        def on_tick():
            output_handoff.service()
            if previous_on_tick is not None:
                previous_on_tick()
    if cleanup_absolute_deadline is not None:
        try:
            valid_cleanup = (owned_group is not None and absolute_deadline is not None
                             and type(cleanup_absolute_deadline) in (int, float)
                             and math.isfinite(cleanup_absolute_deadline)
                             and math.isfinite(absolute_deadline)
                             and 0 < absolute_deadline <= cleanup_absolute_deadline)
        except (TypeError, OverflowError):
            valid_cleanup = False
        if not valid_cleanup:
            raise ValueError("cleanup deadline requires an original work deadline and owner")
    if graceful_drain_deadline is not None:
        try:
            valid_grace = (owned_group is not None and absolute_deadline is not None
                           and type(graceful_drain_deadline) in (int, float)
                           and math.isfinite(graceful_drain_deadline)
                           and 0 < graceful_drain_deadline <= (
                               cleanup_absolute_deadline if cleanup_absolute_deadline is not None
                               else absolute_deadline))
        except (TypeError, OverflowError):
            valid_grace = False
        if not valid_grace:
            raise ValueError("graceful drain needs an owner and a deadline within the absolute deadline")
    if deadline_observation is not None:
        if type(deadline_observation) is not dict or absolute_deadline is None:
            raise ValueError("deadline observation requires a dict and an absolute deadline")
        deadline_observation.update(absolute_deadline_monotonic=absolute_deadline,
                                    exit_observed_monotonic=None,
                                    exit_observation_timing="not_observed",
                                    actual_exit_time="unknown",
                                    deadline_elapsed_while_running_observed=False,
                                    reaped_exit_code=None)
    if absolute_deadline is not None:
        try:
            valid_deadline = (type(absolute_deadline) in (int, float)
                              and math.isfinite(absolute_deadline) and absolute_deadline > 0)
        except OverflowError:
            valid_deadline = False
        if not valid_deadline:
            raise ValueError("absolute deadline must be finite positive monotonic seconds")
        if time.monotonic() >= absolute_deadline:
            raise ProcessNotSpawned("absolute deadline expired before process spawn") from TimeoutError()
    # The selected output owner can be released after a proven pre-dispatch
    # error. This cheap check does not turn a later Popen exception into proof.
    if verify_before_spawn or output_handoff is not None:
        _check_before_spawn(argv, cwd, pass_fds, stdout)
    if absolute_deadline is not None and time.monotonic() >= absolute_deadline:
        raise ProcessNotSpawned("absolute deadline expired before process spawn") from TimeoutError()
    requested_exit = None
    failure = None
    if output_handoff is not None:
        # A Popen attempt is not evidence that no child exists if it raises.
        # The same handoff/lease must remain retained until original custody
        # independently establishes that the process scope is gone.
        if output_handoff.child_fds != output_fds:
            raise ProcessNotSpawned("original output child descriptors changed")
        for fd in output_fds:
            os.fstat(fd)
        output_handoff.mark_dispatch_attempted()
    child = subprocess.Popen(argv, cwd=cwd, env=env, start_new_session=True,
                             pass_fds=tuple(pass_fds), stdout=stdout)
    try:
        if owned_group is not None:
            owned_group._bind(child)
        started = time.monotonic()
        next_check = started
        if on_spawn is not None:
            on_spawn()
        while True:
            status = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if status is not None:
                if deadline_observation is not None:
                    observed = time.monotonic()
                    deadline_observation.update(
                        exit_observed_monotonic=observed,
                        exit_observation_timing=("before_deadline" if observed < absolute_deadline
                                                 else "at_or_after_deadline"),
                    )
                break
            now = time.monotonic()
            if absolute_deadline is not None and now >= absolute_deadline:
                # Reobserve after the deadline was read: the child could have
                # exited between the earlier waitid and the clock read.
                status = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                if status is not None:
                    if deadline_observation is not None:
                        deadline_observation.update(exit_observed_monotonic=time.monotonic(),
                                                    exit_observation_timing="at_or_after_deadline")
                    break
                if deadline_observation is not None:
                    deadline_observation["deadline_elapsed_while_running_observed"] = True
                requested_exit = 124
                break
            # Fast nonblocking control I/O must not wait for the heavier
            # admission/budget scan, which remains on its existing schedule.
            if on_tick is not None:
                on_tick()
                now = time.monotonic()
                if now - started > timeout or (absolute_deadline is not None and now >= absolute_deadline):
                    check_started = now
                    status = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                    if status is not None:
                        if deadline_observation is not None:
                            observed = time.monotonic()
                            deadline_observation.update(
                                exit_observed_monotonic=observed,
                                exit_observation_timing=("before_deadline" if observed < absolute_deadline
                                                         else "at_or_after_deadline"),
                            )
                    else:
                        if deadline_observation is not None and check_started >= absolute_deadline:
                            deadline_observation["deadline_elapsed_while_running_observed"] = True
                        requested_exit = 124
                    break
            if now >= next_check:
                admission_check()
                # A full custody observation may itself exceed the sampling
                # interval. Schedule from completion to avoid a scan hot loop,
                # and count its elapsed time toward the original deadline.
                now = time.monotonic()
                next_check = now + 5
            if now - started > timeout or (absolute_deadline is not None and now >= absolute_deadline):
                if absolute_deadline is not None:
                    # The admission callback can be slow; the process may
                    # have exited during it. Preserve the observed exit.
                    check_started = now
                    status = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                    now = time.monotonic()
                    if status is not None:
                        if deadline_observation is not None:
                            deadline_observation.update(
                                exit_observed_monotonic=now,
                                exit_observation_timing=("before_deadline" if now < absolute_deadline
                                                         else "at_or_after_deadline"),
                            )
                        break
                    if deadline_observation is not None and check_started >= absolute_deadline:
                        deadline_observation["deadline_elapsed_while_running_observed"] = True
                requested_exit = 124
                break
            time.sleep(0.1)
    except KeyboardInterrupt as exc:
        requested_exit = 130
        if str(exc):
            failure = str(exc)
    except Exception as exc:
        requested_exit, failure = 125, str(exc)
    except BaseException as exc:
        # Child ownership already exists: even an uncommon control-flow
        # exception must pass through the original group drain below.
        if on_spawn is None and on_tick is None and owned_group is None:
            raise
        requested_exit, failure = 125, f"{type(exc).__name__}: {exc}"
    if owned_group is not None:
        # Bind again after an exceptional setup path if its first attempt was
        # interrupted; do not infer anything about a Popen that raised itself.
        if owned_group._child is None:
            owned_group._bind(child)
        if requested_exit is not None and output_handoff is not None:
            try:
                # No new output grants after cancellation or timeout. The
                # original lease and group remain owned through the drain.
                output_handoff.close_admission()
            except BaseException as exc:
                owned_group.output_handoff_close_error = {
                    "kind": "callback", "error_type": type(exc).__name__, "message": str(exc)
                }
                # Even if revocation fails, never dispatch another handoff
                # tick. Preserve any unrelated caller tick and first failure.
                on_tick = previous_on_tick
                if failure is None:
                    failure = f"{type(exc).__name__}: {exc}"
        stop_result = None
        if requested_exit is not None and on_stop is not None:
            owned_group.stop_cause = {"requested_exit_code": requested_exit,
                                      "original_supervisor_error": failure}
            try:
                stop_result = on_stop()
            except BaseException as exc:
                owned_group.stop_error = {"kind": "callback", "error_type": type(exc).__name__,
                                          "message": str(exc)}
                # A failed stop hook cannot establish that admission closed.
                # Never run another transport tick on that uncertain channel.
                on_tick = None
        if stop_result is False:
            # The original child exited without an owner-release observation.
            # Do not turn that loss into permission to signal its group or to
            # assert process-tree/GPU completion.
            return requested_exit, False, failure
        if stop_result is True:
            # A retained original owner was released after the work deadline.
            # Give that same child a separate, bounded process-drain window.
            if cleanup_absolute_deadline is None:
                absolute_deadline = time.monotonic() + 3.0
                graceful_drain_deadline = time.monotonic() + 1.0
            else:
                graceful_drain_deadline = min(time.monotonic() + 1.0, cleanup_absolute_deadline)
        gone, result, cleanup_error = _finish_owned_group(
            child, on_tick, graceful_drain_deadline,
            cleanup_absolute_deadline if cleanup_absolute_deadline is not None else absolute_deadline)
        owned_group.cleanup_error = cleanup_error
        if cleanup_error is not None and failure is None:
            failure = f"{cleanup_error['error_type']}: {cleanup_error['message']}"
            if requested_exit is None:
                requested_exit = 125
        if gone:
            owned_group._mark_completed(result)
            if deadline_observation is not None:
                deadline_observation["reaped_exit_code"] = result
        else:
            result = 125
        return requested_exit if requested_exit is not None else result, gone, failure
    # Popen and individual observations cannot be preempted. The finite
    # SIGTERM/SIGKILL drain below may extend beyond the work deadline.
    # The unreaped leader reserves this exact process/group ID while signaling.
    # Normal exits must also drain descendants before scratch is reclaimed.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            break
        if sig == signal.SIGTERM:
            time.sleep(0.2)
    deadline = time.monotonic() + 3
    gone = False
    while time.monotonic() < deadline:
        if not _group_has_live_members(child.pid):
            gone = True
            break
        time.sleep(0.05)
    if gone:
        result = child.wait()
        if deadline_observation is not None:
            deadline_observation["reaped_exit_code"] = result
    else:
        # Do not reap an unresolved leader and lose its reserved group identity.
        result = 125
        child.returncode = 125
    return requested_exit if requested_exit is not None else result, gone, failure


def _run_windows(argv, cwd, env, timeout, admission_check):
    # Assign the suspended root to a kill-on-close Job before it can fork.
    # No cmd.exe, taskkill by a recycled PID, or unrelated process termination.
    from ctypes import wintypes as w
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    ULONG_PTR = ctypes.c_size_t

    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

    class Basic(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                    ("flags", w.DWORD), ("minimum", ULONG_PTR), ("maximum", ULONG_PTR),
                    ("processes", w.DWORD), ("affinity", ULONG_PTR),
                    ("priority", w.DWORD), ("scheduling", w.DWORD)]

    class Extended(ctypes.Structure):
        _fields_ = [("basic", Basic), ("io", IO), ("process_memory", ULONG_PTR),
                    ("job_memory", ULONG_PTR), ("peak_process", ULONG_PTR), ("peak_job", ULONG_PTR)]

    class Accounting(ctypes.Structure):
        _fields_ = [(name, ctypes.c_int64) for name in ("user", "kernel", "period_user", "period_kernel")] + [
            ("faults", w.DWORD), ("total", w.DWORD), ("active", w.DWORD), ("terminated", w.DWORD)]

    class Startup(ctypes.Structure):
        _fields_ = [("cb", w.DWORD), ("reserved", w.LPWSTR), ("desktop", w.LPWSTR), ("title", w.LPWSTR)] + [
            (name, w.DWORD) for name in ("x", "y", "width", "height", "chars_x", "chars_y", "fill", "flags")] + [
            ("show", w.WORD), ("reserved_bytes", w.WORD), ("reserved_data", ctypes.POINTER(ctypes.c_byte)),
            ("stdin", w.HANDLE), ("stdout", w.HANDLE), ("stderr", w.HANDLE)]

    class ProcessInfo(ctypes.Structure):
        _fields_ = [("process", w.HANDLE), ("thread", w.HANDLE), ("pid", w.DWORD), ("tid", w.DWORD)]

    signatures = {
        "CreateJobObjectW": ([ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
        "SetInformationJobObject": ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
        "QueryInformationJobObject": ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p], w.BOOL),
        "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
        "TerminateJobObject": ([w.HANDLE, w.UINT], w.BOOL),
        "CloseHandle": ([w.HANDLE], w.BOOL),
        "GetStdHandle": ([w.DWORD], w.HANDLE),
        "CreateProcessW": ([w.LPCWSTR, w.LPWSTR, ctypes.c_void_p, ctypes.c_void_p, w.BOOL, w.DWORD,
                            ctypes.c_void_p, w.LPCWSTR, ctypes.POINTER(Startup), ctypes.POINTER(ProcessInfo)], w.BOOL),
        "ResumeThread": ([w.HANDLE], w.DWORD),
        "WaitForSingleObject": ([w.HANDLE, w.DWORD], w.DWORD),
        "GetExitCodeProcess": ([w.HANDLE, ctypes.POINTER(w.DWORD)], w.BOOL),
        "TerminateProcess": ([w.HANDLE, w.UINT], w.BOOL),
        "OpenProcess": ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
        "IsProcessInJob": ([w.HANDLE, w.HANDLE, ctypes.POINTER(w.BOOL)], w.BOOL),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(k, name)
        fn.argtypes, fn.restype = args, result

    def require(value):
        if not value:
            raise ctypes.WinError(ctypes.get_last_error())
        return value

    job = require(k.CreateJobObjectW(None, None))
    info = ProcessInfo()
    created = False
    assigned = False
    failure = None
    requested_exit = None
    descendants = {}

    def pin_job_members():
        # ActiveProcesses may reach zero while asynchronous termination is still
        # releasing a descendant's cwd/I/O. Hold exact process objects and wait
        # for their signaled state before claiming scratch is reclaimable.
        capacity = 64
        while capacity <= 4096:
            class ProcessIds(ctypes.Structure):
                _fields_ = [("assigned", w.DWORD), ("listed", w.DWORD),
                            ("ids", ULONG_PTR * capacity)]
            ids = ProcessIds()
            ok = k.QueryInformationJobObject(job, 3, ctypes.byref(ids), ctypes.sizeof(ids), None)
            if not ok and ctypes.get_last_error() != 234:  # ERROR_MORE_DATA
                require(ok)
            if ok and ids.listed == ids.assigned:
                break
            capacity *= 2
        else:
            raise RuntimeError("owned Windows job exceeds 4096 process observation bound")
        for pid in ids.ids[:ids.listed]:
            if pid == info.pid or pid in descendants:
                continue
            handle = k.OpenProcess(0x00100000 | 0x1000, False, pid)  # SYNCHRONIZE | QUERY_LIMITED_INFORMATION
            if not handle and ctypes.get_last_error() == 87:  # Process already fully removed.
                continue
            require(handle)
            member = w.BOOL()
            try:
                require(k.IsProcessInJob(handle, job, ctypes.byref(member)))
                if not member.value:
                    raise RuntimeError("process identity changed outside the owned Windows job")
            except BaseException:
                k.CloseHandle(handle)
                raise
            descendants[pid] = handle

    def process_finished(handle):
        status = k.WaitForSingleObject(handle, 0)
        if status not in (0, 0x102):
            raise ctypes.WinError(ctypes.get_last_error())
        return status == 0

    try:
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        require(k.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
        startup = Startup()
        startup.cb, startup.flags = ctypes.sizeof(startup), 0x100
        startup.stdin, startup.stdout, startup.stderr = [k.GetStdHandle(n & 0xffffffff) for n in (-10, -11, -12)]
        command = ctypes.create_unicode_buffer(subprocess.list2cmdline([str(arg) for arg in argv]))
        block = ctypes.create_unicode_buffer("\0".join(f"{key}={value}" for key, value in sorted(env.items(), key=lambda p: p[0].upper())) + "\0\0")
        # suspended | unicode environment | new process group | no new window
        process_created = k.CreateProcessW(
            None, command, None, None, True, 0x4 | 0x400 | 0x200 | 0x8000000,
            block, str(cwd), ctypes.byref(startup), ctypes.byref(info))
        if not process_created:
            # Only an explicit CreateProcessW failure proves that no child was
            # created. Do not apply this classification to setup exceptions or
            # failures after creation/assignment; their existing preservation
            # path must retain the original lease and uncertain artifacts.
            start_error = ctypes.WinError(ctypes.get_last_error())
            return 125, True, {
                "kind": "windows_process_not_started",
                "stage": "CreateProcessW",
                "process_created": False,
                "winerror": start_error.winerror,
                "error_type": type(start_error).__name__,
                "message": str(start_error),
            }
        created = True
        require(k.AssignProcessToJobObject(job, info.process))
        assigned = True
        if k.ResumeThread(info.thread) == 0xffffffff:
            raise ctypes.WinError(ctypes.get_last_error())
        started = time.monotonic()
        next_check = started
        try:
            while True:
                status = k.WaitForSingleObject(info.process, 100)
                if status == 0:
                    break
                if status != 0x102:
                    raise ctypes.WinError(ctypes.get_last_error())
                now = time.monotonic()
                if now >= next_check:
                    admission_check()
                    now = time.monotonic()
                    next_check = now + 5
                if now - started > timeout:
                    requested_exit = 124
                    break
        except KeyboardInterrupt:
            requested_exit = 130
        except Exception as exc:
            requested_exit, failure = 125, str(exc)
        result = w.DWORD()
        if requested_exit is None:
            require(k.GetExitCodeProcess(info.process, ctypes.byref(result)))
        pin_job_members()
        require(k.TerminateJobObject(job, requested_exit or result.value))
        gone = False
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            pin_job_members()
            accounting = Accounting()
            require(k.QueryInformationJobObject(job, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None))
            if accounting.active == 0 and all(process_finished(handle) for handle in [info.process, *descendants.values()]):
                gone = True
                break
            time.sleep(0.05)
        return requested_exit if requested_exit is not None else result.value, gone, failure
    finally:
        if created and not assigned:
            k.TerminateProcess(info.process, 125)  # Still suspended; never executed.
            k.WaitForSingleObject(info.process, 3000)
        for handle in (*descendants.values(), info.thread, info.process, job):
            if handle:
                k.CloseHandle(handle)
