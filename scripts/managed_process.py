"""Finite owned command trees used by the build-storage boundary."""
from __future__ import annotations
import ctypes
import os
import signal
import subprocess
import time


def run_group(argv, cwd, env, timeout, admission_check):
    if os.name == "nt":
        return _run_windows(argv, cwd, env, timeout, admission_check)
    if not hasattr(os, "waitid") or not hasattr(os, "WNOWAIT"):
        raise RuntimeError("non-reaping process supervision requires Linux waitid")
    child = subprocess.Popen(argv, cwd=cwd, env=env, start_new_session=True)
    started = time.monotonic()
    next_check = started
    requested_exit = None
    failure = None
    try:
        while True:
            status = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if status is not None:
                break
            now = time.monotonic()
            if now >= next_check:
                admission_check()
                next_check = now + 5
            if now - started > timeout:
                requested_exit = 124
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        requested_exit = 130
    except Exception as exc:
        requested_exit, failure = 125, str(exc)
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
        live = False
        for entry in os.scandir("/proc"):
            if not entry.name.isdecimal():
                continue
            try:
                with open(f"{entry.path}/stat", encoding="utf-8") as stream:
                    fields = stream.read().rsplit(")", 1)[1].split()
                # /proc stat: state, ppid, pgrp after the closing comm parenthesis.
                if int(fields[2]) == child.pid and fields[0] not in {"Z", "X"}:
                    live = True
                    break
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (OSError, ValueError, IndexError):
                live = True
                break
        if not live:
            gone = True
            break
        time.sleep(0.05)
    if gone:
        result = child.wait()
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
