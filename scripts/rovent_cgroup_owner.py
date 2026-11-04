#!/usr/bin/env python3
"""Optional same-owner Linux cgroup containment for a V3 smoke child.

The caller must run as the only process in a fresh, delegated cgroup-v2 scope.
The common scope's CPU and memory limits continue to cover both manager and
workload. The caller owns process wait, session completion and output leases;
an empty workload alone proves neither GPU completion nor hostile-root safety.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import uuid


class CgroupOwnerError(RuntimeError):
    pass


class CgroupSetupError(CgroupOwnerError):
    """The initial error and an owner capable of retrying setup rollback."""

    def __init__(self, original: BaseException, recovery: object):
        super().__init__("cgroup setup failed and its rollback is incomplete")
        self.original = original
        self.recovery = recovery


def _write(path: Path, value: str) -> None:
    with path.open("w", encoding="ascii") as handle:
        handle.write(value)


def _read(path: Path) -> str:
    return path.read_text(encoding="ascii").strip()


def _membership(pid: str) -> str:
    entries = [line[3:] for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines()
               if line.startswith("0::")]
    if len(entries) != 1 or not entries[0].startswith("/"):
        raise CgroupOwnerError("unified cgroup membership unavailable")
    return entries[0]


def _identity(path: Path) -> tuple[int, int]:
    value = os.stat(path, follow_symlinks=False)
    return value.st_dev, value.st_ino


class _SetupRecovery:
    def __init__(self, scope: Path, manager: Path, workload: Path, entry: Path):
        self.scope, self.manager, self.workload, self.entry = scope, manager, workload, entry
        self.pid = os.getpid()
        self.scope_fd = os.open(scope, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        held = os.fstat(self.scope_fd)
        self.scope_id = held.st_dev, held.st_ino
        self.created: dict[Path, tuple[int, int]] = {}

    def remember(self, path: Path) -> None:
        self.created[path] = _identity(path)

    def release_fd(self) -> None:
        if self.scope_fd is not None:
            os.close(self.scope_fd)
            self.scope_fd = None

    def retry(self) -> None:
        held = os.fstat(self.scope_fd)
        if (os.getpid() != self.pid or _identity(self.scope) != self.scope_id
                or (held.st_dev, held.st_ino) != self.scope_id):
            raise CgroupOwnerError("original setup owner is unavailable")
        for path in (self.entry, self.workload):
            if path.exists():
                if self.created.get(path) != _identity(path):
                    raise CgroupOwnerError("setup child identity is unknown")
                path.rmdir()
        controls = set(_read(self.scope / "cgroup.subtree_control").split())
        if not controls.issubset({"cpu", "memory", "pids"}):
            raise CgroupOwnerError("scope controller state changed")
        if controls:
            _write(self.scope / "cgroup.subtree_control", " ".join("-" + c for c in sorted(controls)))
        scope_membership = "/" + str(self.scope.relative_to(Path("/sys/fs/cgroup")))
        manager_membership = "/" + str(self.manager.relative_to(Path("/sys/fs/cgroup")))
        current = _membership("self")
        if current == manager_membership:
            _write(self.scope / "cgroup.procs", str(self.pid))
        elif current != scope_membership:
            raise CgroupOwnerError("manager membership changed during setup")
        if self.manager.exists():
            if self.created.get(self.manager) != _identity(self.manager):
                raise CgroupOwnerError("manager identity is unknown")
            self.manager.rmdir()
        self.release_fd()


class CgroupOwner:
    """Own one fresh workload group below the caller's delegated scope."""

    def __init__(self, scope: Path, manager: Path, workload: Path, entry: Path):
        self.scope, self.manager, self.workload, self.entry = scope, manager, workload, entry
        self._pid = os.getpid()
        self._fds: dict[str, int] = {}
        try:
            for name, path in (("scope", scope), ("manager", manager),
                               ("workload", workload), ("entry", entry)):
                self._fds[name] = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            self._ids = {name: (os.fstat(fd).st_dev, os.fstat(fd).st_ino)
                         for name, fd in self._fds.items()}
        except BaseException:
            for fd in self._fds.values():
                os.close(fd)
            raise
        self._entry_removed = False
        self._workload_removed = False
        self._manager_removed = False
        self._entry_remove_intent = False
        self._workload_remove_intent = False
        self._manager_remove_intent = False
        self._controls_disabled = False
        self._manager_moved = False
        self.admission_outcome = "not_attempted"
        self._closed = False

    @classmethod
    def prepare(cls) -> "CgroupOwner":
        root = Path("/sys/fs/cgroup")
        if sys.platform != "linux" or not (root / "cgroup.controllers").is_file():
            raise CgroupOwnerError("Linux unified cgroup is required")
        if len(list(Path("/proc/self/task").iterdir())) != 1:
            raise CgroupOwnerError("scope setup requires a single-threaded manager")
        membership = _membership("self")
        scope = root.joinpath(membership.lstrip("/"))
        if not membership.startswith("/system.slice/") or not membership.endswith(".scope"):
            raise CgroupOwnerError("a dedicated delegated systemd scope is required")
        if scope.resolve() != scope or _read(scope / "cgroup.procs").split() != [str(os.getpid())]:
            raise CgroupOwnerError("scope does not exclusively contain the manager")
        if _read(scope / "cgroup.subtree_control"):
            raise CgroupOwnerError("scope controllers are already delegated")
        if not {"cpu", "memory", "pids"}.issubset(_read(scope / "cgroup.controllers").split()):
            raise CgroupOwnerError("required delegated controllers are unavailable")
        if any(item.is_dir() for item in scope.iterdir()):
            raise CgroupOwnerError("scope already contains child cgroups")
        suffix = uuid.uuid4().hex
        manager = scope / ("rovent-manager-" + suffix)
        workload = scope / ("rovent-workload-" + suffix)
        entry = workload / "entry"
        recovery = _SetupRecovery(scope, manager, workload, entry)
        owner = None
        try:
            manager.mkdir(mode=0o700)
            recovery.remember(manager)
            _write(manager / "cgroup.procs", str(os.getpid()))
            _write(scope / "cgroup.subtree_control", "+cpu +memory +pids")
            workload.mkdir(mode=0o700)
            recovery.remember(workload)
            _write(workload / "cgroup.subtree_control", "+cpu +memory +pids")
            entry.mkdir(mode=0o700)
            recovery.remember(entry)
            owner = cls(scope, manager, workload, entry)
            recovery.release_fd()
            return owner
        except BaseException as failure:
            try:
                if owner is None:
                    recovery.retry()
                else:
                    try:
                        owner.close()
                    finally:
                        recovery.release_fd()
            except BaseException as rollback_failure:
                failure.add_note("setup rollback failed: " + type(rollback_failure).__name__)
                raise CgroupSetupError(failure, recovery if owner is None else owner) from failure
            raise

    @property
    def delegated_path(self) -> Path:
        return self.workload

    def _check_owner(self) -> None:
        if self._closed or self._pid != os.getpid():
            raise CgroupOwnerError("original cgroup owner is unavailable")
        for name, path, removed in (("scope", self.scope, False),
                                    ("manager", self.manager, self._manager_removed),
                                    ("workload", self.workload, self._workload_removed),
                                    ("entry", self.entry, self._entry_removed)):
            if not removed:
                held = os.fstat(self._fds[name])
                if _identity(path) != self._ids[name] or (held.st_dev, held.st_ino) != self._ids[name]:
                    raise CgroupOwnerError("owned cgroup identity changed")

    def _reconcile_removals(self) -> None:
        # An interrupt can arrive after rmdir completed but before Python set
        # its flag. Only a previously recorded intent and ENOENT relative to
        # the still-held original parent descriptor can complete that step.
        for intent, removed, parent_fd, name, field in (
            (self._entry_remove_intent, self._entry_removed, self._fds["workload"],
             self.entry.name, "_entry_removed"),
            (self._workload_remove_intent, self._workload_removed, self._fds["scope"],
             self.workload.name, "_workload_removed"),
            (self._manager_remove_intent, self._manager_removed, self._fds["scope"],
             self.manager.name, "_manager_removed"),
        ):
            if intent and not removed:
                try:
                    os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    setattr(self, field, True)

    def admit_waiting_child(self, pid: int, release_fd: int) -> None:
        """Move the direct, unreaped barrier child before permitting its exec."""
        self.admission_outcome = "rejected_before_grant"
        try:
            self._check_owner()
            if type(pid) is not int or pid <= 0:
                raise CgroupOwnerError("invalid child pid")
            status = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if status is not None:
                raise CgroupOwnerError("barrier child already exited")
            _write(self.entry / "cgroup.procs", str(pid))
            expected = "/" + str(self.entry.relative_to(Path("/sys/fs/cgroup")))
            if _membership(str(pid)) != expected or str(pid) not in _read(self.entry / "cgroup.procs").split():
                raise CgroupOwnerError("barrier child placement could not be confirmed")
            if os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
                raise CgroupOwnerError("barrier child exited during placement")
            # An interruption during write cannot establish whether the child
            # received the grant. The caller must retain the original Popen.
            self.admission_outcome = "grant_unknown"
            if os.write(release_fd, b"R") != 1:
                raise CgroupOwnerError("barrier release was incomplete")
            self.admission_outcome = "granted"
        finally:
            # EOF makes a still-waiting shim exit without executing native.
            os.close(release_fd)

    def workload_empty(self) -> bool:
        """Observe this same cgroup tree, not a device fence or host-wide state."""
        self._reconcile_removals()
        self._check_owner()
        if self._workload_removed:
            # The kernel only permits removal of an unpopulated cgroup. This
            # owner attempted rmdir after observing empty; a later cleanup
            # step can fail, so retain that original tree's empty outcome for
            # retry without making any claim about a newly created pathname.
            return True
        fd = os.open("cgroup.events", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                     dir_fd=self._fds["workload"])
        with os.fdopen(fd, "r", encoding="ascii") as handle:
            events = dict(line.split() for line in handle.read(4096).splitlines())
        if events.get("populated") not in ("0", "1"):
            raise CgroupOwnerError("workload populated observation unavailable")
        return events["populated"] == "0"

    def close(self, native_leaf: str | None = None) -> None:
        """Remove only fresh, empty directories after caller confirms closure."""
        self._reconcile_removals()
        self._check_owner()
        if not self._workload_removed:
            if not self.workload_empty():
                raise CgroupOwnerError("workload is still populated")
            allowed = {"entry"}
            if native_leaf is not None:
                if not native_leaf.startswith("rovent-") or "/" in native_leaf or native_leaf in (".", ".."):
                    raise CgroupOwnerError("invalid native leaf name")
                allowed.add(native_leaf)
            children = {item.name for item in self.workload.iterdir() if item.is_dir()}
            if not children.issubset(allowed):
                raise CgroupOwnerError("unexpected child cgroup is retained")
            if native_leaf in children:
                (self.workload / native_leaf).rmdir()
            if not self._entry_removed:
                self._entry_remove_intent = True
                self.entry.rmdir()
                self._entry_removed = True
            self._workload_remove_intent = True
            self.workload.rmdir()
            self._workload_removed = True
        if not self._controls_disabled:
            controls = set(_read(self.scope / "cgroup.subtree_control").split())
            if not controls.issubset({"cpu", "memory", "pids"}):
                raise CgroupOwnerError("scope controller state changed")
            if controls:
                _write(self.scope / "cgroup.subtree_control", " ".join("-" + c for c in sorted(controls)))
            self._controls_disabled = True
        if not self._manager_moved:
            current = _membership("self")
            manager_path = "/" + str(self.manager.relative_to(Path("/sys/fs/cgroup")))
            scope_path = "/" + str(self.scope.relative_to(Path("/sys/fs/cgroup")))
            if current == manager_path:
                _write(self.scope / "cgroup.procs", str(self._pid))
            elif current != scope_path:
                raise CgroupOwnerError("manager membership changed during cleanup")
            self._manager_moved = True
        if not self._manager_removed:
            self._manager_remove_intent = True
            self.manager.rmdir()
            self._manager_removed = True
        self._closed = True
        for fd in self._fds.values():
            os.close(fd)

    def retry(self) -> None:
        """Retry an unstarted setup cleanup returned by CgroupSetupError."""
        self.close()


def entry_command(binary_fd: int, barrier_fd: int, argv: list[str]) -> list[str]:
    if type(binary_fd) is not int or type(barrier_fd) is not int or not argv:
        raise CgroupOwnerError("entry needs held binary and barrier descriptors")
    return [sys.executable, "-I", "-B", __file__, "--entry", str(binary_fd), str(barrier_fd), *argv]


def _entry(binary_fd: int, barrier_fd: int, argv: list[str]) -> None:
    try:
        if os.read(barrier_fd, 1) != b"R":
            raise CgroupOwnerError("native entry was not admitted")
    finally:
        os.close(barrier_fd)
    os.execve(f"/proc/self/fd/{binary_fd}", argv, os.environ)


if __name__ == "__main__":
    if len(sys.argv) < 5 or sys.argv[1] != "--entry":
        raise SystemExit("internal entry arguments required")
    _entry(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:])
