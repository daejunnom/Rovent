"""OS-held catalog serialization and conservative interrupted-owner evidence.

No PID signaling or age-based lease stealing. A missing owner is insufficient:
unobserved descendants may still own files until a verified reboot.
"""
from __future__ import annotations

import ctypes
import errno
import hashlib
import os
from pathlib import Path
import stat
import sys


WINDOWS_BOOT_COUNTER_PREFIX = "windows-boot-counter-v1:"
WINDOWS_BOOT_COUNTER_KEY = (
    r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management\PrefetchParameters"
)


def _windows_boot_identifier(registry):
    """Read the OS-owned successful-boot counter; never estimate from clocks.

    Microsoft's memory-integrity enablement documentation defines this BootId
    as a counter incremented on each successful boot.  The BootIdentifier in
    SystemBootEnvironmentInformation identifies a boot loader entry and is not
    sufficient to distinguish successive boots of the same Windows install.
    Missing, denied or malformed counters remain unavailable; no fallback can
    turn such an observation into a recovery proof.

    Reference: https://learn.microsoft.com/windows-hardware/design/device-experiences/oem-hvci-enablement
    """
    access = registry.KEY_READ | registry.KEY_WOW64_64KEY
    with registry.OpenKey(registry.HKEY_LOCAL_MACHINE, WINDOWS_BOOT_COUNTER_KEY, 0, access) as key:
        count, kind = registry.QueryValueEx(key, "BootId")
    if kind != registry.REG_DWORD or type(count) is not int or not 0 <= count <= 0xffffffff:
        raise OSError("Windows successful-boot counter unavailable or malformed")
    return WINDOWS_BOOT_COUNTER_PREFIX + str(count)


def _windows_boot_count(identifier):
    """Parse only the new domain; legacy loader GUIDs never imply a reboot."""
    if not isinstance(identifier, str) or not identifier.startswith(WINDOWS_BOOT_COUNTER_PREFIX):
        return None
    value = identifier[len(WINDOWS_BOOT_COUNTER_PREFIX):]
    if not value or len(value) > 10 or not value.isascii() or not value.isdecimal():
        return None
    count = int(value)
    if str(count) != value or not 0 <= count <= 0xffffffff:
        return None
    return count


class CatalogGuard:
    def __init__(self, path):
        self.path, self.stream = Path(path), None
        self.parent_identity = None

    def acquire(self):
        from managed_storage import CatalogBusy, StorageError, checked
        checked(self.path)
        parent = self.path.parent.stat()
        self.parent_identity = (parent.st_dev, parent.st_ino)
        stream = self.path.open("a+b")
        try:
            info = os.fstat(stream.fileno())
            named = self.path.lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                    or (info.st_dev, info.st_ino) != (named.st_dev, named.st_ino)):
                raise StorageError("catalog guard identity changed")
            if os.name == "nt":
                import msvcrt
                if info.st_size == 0:
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise CatalogBusy("catalog lock is held; retry may be possible") from exc
                    raise
            else:
                import fcntl
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    if exc.errno in (errno.EACCES, errno.EAGAIN):
                        raise CatalogBusy("catalog lock is held; retry may be possible") from exc
                    raise
        except OSError as exc:
            stream.close()
            raise StorageError("catalog is active or lock unavailable; preserve it") from exc
        except BaseException:
            stream.close()
            raise
        self.stream = stream
        return self

    def release(self):
        if self.stream is not None:
            self.stream.close()  # OS unlock, including abnormal process exit.
            self.stream = None

    def assert_held(self):
        from managed_storage import StorageError, checked
        checked(self.path)
        if self.stream is None:
            raise StorageError("catalog gate not held")
        opened, named = os.fstat(self.stream.fileno()), self.path.lstat()
        parent = self.path.parent.stat()
        if (self.parent_identity != (parent.st_dev, parent.st_ino) or opened.st_nlink != 1
                or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)):
            raise StorageError("catalog gate replaced; preserve artifacts")

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.release()

    def __del__(self):
        if getattr(self, "stream", None) is not None:
            self.release()  # Persistent lease remains; only the OS-held gate closes.


def lifetime():
    """Host-local proof; identifiers stay in local maintenance records only."""
    if sys.platform == "linux":
        host = Path("/etc/machine-id").read_text(encoding="ascii").strip()
        boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        birth = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
    elif os.name == "nt":
        import winreg
        from ctypes import wintypes as w
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
            host = winreg.QueryValueEx(key, "MachineGuid")[0]
        boot = _windows_boot_identifier(winreg)
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetCurrentProcess.restype = w.HANDLE
        k.GetProcessTimes.argtypes = [w.HANDLE, *([ctypes.POINTER(w.FILETIME)] * 4)]
        times = [w.FILETIME() for _ in range(4)]
        if not k.GetProcessTimes(k.GetCurrentProcess(), *[ctypes.byref(t) for t in times]):
            raise ctypes.WinError(ctypes.get_last_error())
        birth = str(times[0].dwLowDateTime | (times[0].dwHighDateTime << 32))
    else:
        raise OSError("verified owner lifetime unavailable on this OS")
    if not host or not boot or not birth:
        raise OSError("empty owner lifetime")
    return {"host": hashlib.sha256(host.encode()).hexdigest(), "boot": boot,
            "process_birth": birth, "platform": sys.platform}


def reboot_proves_dead(previous, current):
    # Same-boot PID death/reuse says nothing about orphan descendants. Legacy
    # owners without this evidence are deliberately not recoverable here.
    keys = {"host", "boot", "process_birth", "platform"}
    compatible = (isinstance(previous, dict) and set(previous) == keys
            and isinstance(current, dict) and set(current) == keys
            and all(isinstance(v, str) and v for v in previous.values())
            and all(isinstance(v, str) and v for v in current.values())
            and previous["host"] == current["host"]
            and previous["platform"] == current["platform"])
    if not compatible:
        return False
    if previous["platform"] == "win32":
        old_count, new_count = _windows_boot_count(previous["boot"]), _windows_boot_count(current["boot"])
        # A format change, rollback, wrap or reset is not a reboot proof.  Old
        # loader-GUID records stay preserved for explicit legacy maintenance.
        return old_count is not None and new_count is not None and new_count > old_count
    return previous["boot"] != current["boot"]


def owner_proven_dead(pid, previous, current):
    """Only useful with a lease-bound terminal tree-closure witness."""
    if (not isinstance(previous, dict) or previous.get("host") != current["host"]
            or previous.get("boot") != current["boot"]
            or previous.get("platform") != current["platform"] or type(pid) is not int or pid <= 0):
        return False
    try:
        if sys.platform == "linux":
            birth = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        elif os.name == "nt":
            from ctypes import wintypes as w
            k = ctypes.WinDLL("kernel32", use_last_error=True)
            k.OpenProcess.argtypes, k.OpenProcess.restype = [w.DWORD, w.BOOL, w.DWORD], w.HANDLE
            k.CloseHandle.argtypes = [w.HANDLE]
            k.GetProcessTimes.argtypes = [w.HANDLE, *([ctypes.POINTER(w.FILETIME)] * 4)]
            k.WaitForSingleObject.argtypes, k.WaitForSingleObject.restype = [w.HANDLE, w.DWORD], w.DWORD
            handle = k.OpenProcess(0x1000 | 0x00100000, False, pid)
            if not handle:
                return ctypes.get_last_error() == 87  # Only ERROR_INVALID_PARAMETER.
            try:
                status = k.WaitForSingleObject(handle, 0)
                if status == 0:
                    return True  # Exact retained process object is terminated.
                if status != 0x102:
                    return False
                times = [w.FILETIME() for _ in range(4)]
                if not k.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
                    return False
                birth = str(times[0].dwLowDateTime | (times[0].dwHighDateTime << 32))
            finally:
                k.CloseHandle(handle)
        else:
            return False
        return isinstance(previous.get("process_birth"), str) and birth != previous["process_birth"]
    except FileNotFoundError:
        return sys.platform == "linux"
    except (OSError, ValueError, IndexError):
        return False
