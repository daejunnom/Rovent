"""Bounded raw-output custody for a future Linux LC0 parent.

This component only records bytes and their I/O status.  It does not own a
process, authenticate a peer, decide whether a session may exit, or prove GPU
completion.  The caller owns the managed output directory and the original
stream descriptor; this object retains its own descriptor until explicitly
closed or handed off together with any pending bytes.
"""
from __future__ import annotations

import os
import stat


class OutputSpool:
    """Keep a bounded prefix, including bytes awaiting a failed write.

    The descriptor must name a new empty regular file owned by the caller.
    Writes use a positional offset so changes to the caller's file position do
    not reorder the captured stream.  A disk error is sticky evidence, not a
    claim of EOF; retrying is explicit and only retries the retained bytes.
    """

    def __init__(self, fd: int, max_bytes: int):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("output spool requires a positive byte bound")
        self._fd = os.dup(fd)
        try:
            info = os.fstat(self._fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size != 0:
                raise ValueError("output spool requires a new empty regular file")
            self._identity = (info.st_dev, info.st_ino)
        except BaseException:
            os.close(self._fd)
            raise
        self.max_bytes = max_bytes
        self.bytes_seen = 0
        self.bytes_persisted = 0
        self.bytes_truncated = 0
        self._pending = bytearray()
        self._first_write_error = None
        self._write_blocked = False
        self._read_error = None
        self._eof = False

    def accept(self, data: bytes) -> None:
        if self._fd is None or self._eof or self._read_error is not None:
            raise ValueError("output stream no longer accepts bytes")
        if not isinstance(data, bytes):
            raise TypeError("output data must be bytes")
        self.bytes_seen += len(data)
        remaining = self.max_bytes - self.bytes_persisted - len(self._pending)
        retained = data[:remaining]
        self._pending.extend(retained)
        self.bytes_truncated += len(data) - len(retained)
        if not self._write_blocked:
            self._write_pending()

    def _write_pending(self) -> None:
        while self._pending:
            try:
                current = os.fstat(self._fd)
                if ((current.st_dev, current.st_ino) != self._identity
                        or current.st_size != self.bytes_persisted):
                    raise OSError("output file identity or size changed")
                count = os.pwrite(self._fd, self._pending, self.bytes_persisted)
                if count <= 0 or count > len(self._pending):
                    raise OSError("output file write did not advance")
                self.bytes_persisted += count
                del self._pending[:count]
            except OSError as error:
                if self._first_write_error is None:
                    self._first_write_error = type(error).__name__
                self._write_blocked = True
                break
        else:
            self._write_blocked = False

    def retry_pending(self) -> None:
        """Retry only the original retained bytes; keep the first error record."""
        if self._fd is None:
            raise ValueError("output spool is closed")
        self._write_pending()

    def observe_eof(self) -> None:
        if self._fd is None:
            raise ValueError("output spool is closed")
        self._eof = True

    def observe_read_error(self, error: OSError) -> None:
        if self._fd is None or self._eof:
            raise ValueError("output stream is already closed")
        if self._read_error is None:
            self._read_error = type(error).__name__

    def observation(self) -> dict[str, object]:
        return {
            "bytes_seen": self.bytes_seen,
            "bytes_persisted": self.bytes_persisted,
            "bytes_pending": len(self._pending),
            "bytes_truncated": self.bytes_truncated,
            "first_write_error": self._first_write_error,
            "write_blocked": self._write_blocked,
            "read_error": self._read_error,
            "eof_observed": self._eof,
        }

    def close(self) -> None:
        if self._pending:
            raise RuntimeError("output bytes remain owned by the spool")
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
