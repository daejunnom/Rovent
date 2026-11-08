"""Observed script identity for the narrow, opt-in broker reader profiles.

This records the selected script bytes and path, not proof of Python imports,
arbitrary callbacks, other writers, or the child's exact descriptor FD. The
caller must separately review the reader's protocol and keep its original
manager and child identities. It uses no hard-coded expected content hashes.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import sys

from managed_storage import StorageError, checked


_READER_NAMES = {"rovent_v3_parent.py", "rovent_before_native_owner.py",
                 "virtual_system_numeric_prepare.py"}
_SCRIPT_BYTES_LIMIT = 8 * 1024 * 1024


def _stamp(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _digest(fd, before):
    digest = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(fd, min(65536, before.st_size - offset), offset)
        if not chunk:
            raise StorageError("reader source truncated while observing")
        digest.update(chunk)
        offset += len(chunk)
    if _stamp(os.fstat(fd)) != _stamp(before):
        raise StorageError("reader source changed while observing")
    return digest.hexdigest()


class ObservedReaderProfile:
    """Pin a selected source script; caller separately validates its scope."""

    def __init__(self, manager, command):
        if sys.platform != "linux" or not hasattr(os, "O_NOFOLLOW"):
            raise StorageError("Linux no-follow reader profile required")
        if not isinstance(command, (list, tuple)) or len(command) < 2:
            raise StorageError("direct Python reader command required")
        try:
            immutable_command = tuple(os.fspath(argument) for argument in command)
        except TypeError as error:
            raise StorageError("string reader command required") from error
        if any(type(argument) is not str or "\0" in argument for argument in immutable_command):
            raise StorageError("string reader command required")
        if (not Path(command[0]).is_absolute()
                or os.path.realpath(os.fspath(command[0])) != os.path.realpath(sys.executable)):
            raise StorageError("original Python interpreter required for reader profile")
        manager._assert_owner()
        source = checked(Path(manager.source))
        script = Path(command[1])
        if not script.is_absolute():
            script = source / script
        script = checked(script)
        if script.name not in _READER_NAMES or script != source / "scripts" / script.name:
            raise StorageError("reviewed direct reader path required")
        self._manager = manager
        self._command = immutable_command
        self._script = script
        self._fd = None
        self._closed = False
        self.close_unknown = False
        try:
            self._fd = os.open(script, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
            info = os.fstat(self._fd)
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= _SCRIPT_BYTES_LIMIT:
                raise StorageError("bounded regular reader source required")
            named = script.lstat()
            if _stamp(named) != _stamp(info):
                raise StorageError("reader source path changed during observation")
            self._stamp = _stamp(info)
            self._sha256 = _digest(self._fd, info)
            manager._assert_owner()
        except BaseException as error:
            try:
                self.close()
            except BaseException as close_error:
                if hasattr(error, "add_note"):
                    error.add_note("reader source close unknown: " + type(close_error).__name__)
            raise

    def unchanged(self):
        if self._closed or self._fd is None:
            return False
        try:
            self._manager._assert_owner()
            opened = os.fstat(self._fd)
            named = self._script.lstat()
            return (_stamp(opened) == self._stamp and _stamp(named) == self._stamp
                    and _digest(self._fd, opened) == self._sha256)
        except (OSError, StorageError):
            return False

    @property
    def command(self):
        return self._command

    def observation(self):
        return {"reader": self._script.name, "source_sha256": self._sha256,
                "scope": "observed_script_only", "imports": "unknown",
                "same_descriptor_fd": "unknown", "nonparticipating_writers": "unknown"}

    def close(self):
        if not self._closed:
            self._closed = True
            fd, self._fd = self._fd, None
            if fd is not None:
                try:
                    os.close(fd)
                except OSError as error:
                    self.close_unknown = True
                    raise StorageError("reader source descriptor close unknown") from error


class ReviewedReaderContract:
    """Bind a caller-reviewed narrow reader contract to observed script bytes.

    Construction records a caller declaration; it does not independently
    audit Python imports, nonparticipating writers, or the child's opened FD.
    """

    def __init__(self, profile, *, reviewed_sha256, descriptor_content_only, derived_content_after_ack):
        if (type(profile) is not ObservedReaderProfile or not profile.unchanged()
                or type(reviewed_sha256) is not str or reviewed_sha256 != profile.observation()["source_sha256"]
                or descriptor_content_only is not True or derived_content_after_ack is not True):
            raise StorageError("reviewed narrow reader contract bound to observed source required")
        self.profile = profile
        self.reviewed_sha256 = reviewed_sha256

    def allows_unknown_abandon(self):
        return (self.profile.unchanged()
                and self.profile.observation()["source_sha256"] == self.reviewed_sha256)
