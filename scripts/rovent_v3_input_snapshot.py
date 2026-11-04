"""Owned input snapshot helpers for the V3 managed parent.

These helpers do not start a process, create a lease, or infer native/GPU
completion. Callers retain the original source and output owners.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import time


MAX_CONTROL = 1024 * 1024
MAX_ASSET = 2 * 1024 ** 3


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def unique_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def relative(value):
    require(isinstance(value, str), "artifact path must be text")
    path = PurePosixPath(value)
    require(not path.is_absolute() and path.parts and
            all(part not in ("", ".", "..") for part in path.parts), "unsafe artifact path")
    return Path(*path.parts)


def open_directory(path):
    """Pin each path component without following links, including ancestors."""
    require(path.is_absolute() and all(part != ".." for part in path.parts), "unsafe directory path")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[1:]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def open_source(base_fd, rel):
    require(not rel.is_absolute() and all(part not in ("", ".", "..") for part in rel.parts),
            "unsafe source path")
    parent_fd = os.dup(base_fd)
    try:
        for component in rel.parts[:-1]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        fd = os.open(rel.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(fd)
            raise RuntimeError("source must be a regular file")
        return fd, info
    finally:
        os.close(parent_fd)


def metadata(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def read_control(base, rel):
    fd, before = open_source(base, rel)
    try:
        require(before.st_size <= MAX_CONTROL, "control input exceeds limit")
        with os.fdopen(fd, "rb", closefd=False) as source:
            data = source.read(MAX_CONTROL + 1)
        require(len(data) == before.st_size and metadata(before) == metadata(os.fstat(fd)),
                "control input changed while reading")
        return data
    finally:
        os.close(fd)


class OriginalControlInput:
    """Retain the first no-follow control-file open and its source directory.

    This is an input ownership primitive, not permission to write to an output
    lease. The creating supervisor must retain it until its child has stopped.
    Rechecking detects observed mutation but does not make the file immutable.
    """

    def __init__(self, base, rel):
        self._directory = self._file = None
        try:
            self._directory = os.dup(base)
            directory = os.fstat(self._directory)
            require(stat.S_ISDIR(directory.st_mode), "control source is not a directory")
            self._directory_id = (directory.st_dev, directory.st_ino)
            self._file, before = open_source(self._directory, rel)
            require(before.st_size <= MAX_CONTROL, "control input exceeds limit")
            self._metadata = metadata(before)
            self._bytes = self._read_stable()
        except BaseException as error:
            try:
                self.close()
            except BaseException as close_error:
                error.add_note("control input cleanup also failed: " + str(close_error))
            raise

    def _read_stable(self):
        require(self._file is not None, "original control input is closed")
        before = os.fstat(self._file)
        require(stat.S_ISREG(before.st_mode) and metadata(before) == self._metadata,
                "original control input changed")
        chunks, offset = [], 0
        while offset < before.st_size:
            chunk = os.pread(self._file, min(65536, before.st_size - offset), offset)
            require(bool(chunk), "original control input was truncated")
            chunks.append(chunk)
            offset += len(chunk)
        require(metadata(os.fstat(self._file)) == self._metadata,
                "original control input changed while reading")
        return b"".join(chunks)

    def recheck(self):
        require(self._directory is not None, "original control input is closed")
        directory = os.fstat(self._directory)
        require(stat.S_ISDIR(directory.st_mode) and
                (directory.st_dev, directory.st_ino) == self._directory_id,
                "original control source directory changed")
        require(self._read_stable() == self._bytes, "original control input bytes changed")

    @property
    def bytes(self):
        self.recheck()
        return self._bytes

    def borrowed_fds(self):
        """Borrow original descriptors; the caller must retain this owner."""
        self.recheck()
        return self._directory, self._file

    def close(self):
        failure = None
        for field in ("_file", "_directory"):
            fd = getattr(self, field, None)
            if fd is not None:
                setattr(self, field, None)
                try:
                    os.close(fd)
                except OSError as error:
                    if failure is None:
                        failure = error
        if failure is not None:
            raise failure

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()


def copy_pin(base, root, item, deadline, progress):
    """Copy one pinned regular file without replacing an existing output."""
    rel = relative(item["path"])
    expected_size, expected_sha = item["bytes"], item["sha256"]
    require(type(expected_size) is int and 0 < expected_size <= MAX_ASSET,
            "invalid declared artifact size")
    require(isinstance(expected_sha, str) and len(expected_sha) == 64 and
            all(c in "0123456789abcdef" for c in expected_sha), "invalid declared digest")
    fd, before = open_source(base, rel)
    try:
        require(before.st_size == expected_size, "source artifact size differs")
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        out_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        digest, total = hashlib.sha256(), 0
        try:
            while True:
                require(time.monotonic() < deadline, "original deadline expired while copying")
                data = os.read(fd, 1024 * 1024)
                if not data:
                    break
                digest.update(data)
                total += len(data)
                require(total <= expected_size, "source grew during copy")
                progress[0] += len(data)
                require(progress[0] <= progress[1], "aggregate copy reservation exceeded")
                view = memoryview(data)
                while view:
                    written = os.write(out_fd, view)
                    require(written > 0, "copy made no progress")
                    view = view[written:]
            os.fsync(out_fd)
        finally:
            os.close(out_fd)
        require(total == expected_size and digest.hexdigest() == expected_sha and
                metadata(before) == metadata(os.fstat(fd)), "artifact bytes or source identity changed")
        with open(target, "rb") as output:
            require(hashlib.file_digest(output, "sha256").hexdigest() == expected_sha,
                    "copied artifact digest differs")
        return {"path": str(rel), "bytes": total, "sha256": expected_sha}
    finally:
        os.close(fd)


def protect_output(lease, output, parent_fd):
    """Restrict only the directory inode owned by the acquired lease."""
    require(getattr(lease, "_created_directory", None) is None,
            "managed output capability already seals its mode")
    expected = lease.record.get("identity")
    require(isinstance(expected, list) and len(expected) == 2, "missing owner identity")
    fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        named = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        require(stat.S_ISDIR(opened.st_mode) and stat.S_ISDIR(named.st_mode) and
                [opened.st_dev, opened.st_ino] == expected == [named.st_dev, named.st_ino] and
                opened.st_uid == os.geteuid(), "registered output identity changed")
        os.fchmod(fd, 0o700)
        named = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        require([named.st_dev, named.st_ino] == expected and
                stat.S_IMODE(os.fstat(fd).st_mode) == 0o700, "output protection failed")
    finally:
        os.close(fd)
