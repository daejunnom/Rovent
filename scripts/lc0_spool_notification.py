"""Coalesced selector wakeup for a process-local LC0 spool writer.

This is only a wakeup, never an output result or a physical-completion proof.
The writer publishes its state first, then signals. The consumer drains the
FD and immediately re-observes the original writer state. A full pipe means
an earlier signal remains readable; other errors are retained for the owner
and require the consumer's bounded-tick fallback.
"""
from __future__ import annotations

import errno
import os
import threading


class CoalescedSpoolNotification:
    def __init__(self, writer):
        if not isinstance(writer, threading.Thread):
            raise TypeError("notification requires its original writer thread")
        self._read, self._write = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
        self._writer = writer
        self._lock = threading.Lock()
        self._first_error = None
        self._coalesced = 0
        self._close_started = False
        self._close_status = "open"

    def fileno(self):
        with self._lock:
            return -1 if self._close_started else self._read

    def _retain_error(self, error):
        with self._lock:
            if self._first_error is None:
                self._first_error = error

    def signal(self):
        """Signal once after publishing state, without waiting for a reader."""
        try:
            count = os.write(self._write, b"\x01")
            if count != 1:
                raise OSError("notification write did not advance")
            return "signalled"
        except BlockingIOError as error:
            if error.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                self._retain_error(error)
                return "failed"
            with self._lock:
                self._coalesced += 1
            return "coalesced"
        except OSError as error:
            self._retain_error(error)
            return "failed"

    def drain(self):
        """Drain a bounded amount, then have the caller re-read writer state."""
        with self._lock:
            if self._close_started:
                return None
            try:
                # This pipe is nonblocking; holding this small lock prevents
                # concurrent release from turning the read into an FD reuse.
                data = os.read(self._read, 8192)
                if not data:
                    raise OSError("notification pipe closed before release")
                return len(data)
            except BlockingIOError:
                return 0
            except OSError as error:
                if self._first_error is None:
                    self._first_error = error
                return None

    def observation(self):
        with self._lock:
            return {"first_error_type": (type(self._first_error).__name__
                                         if self._first_error is not None else None),
                    "first_error_errno": (self._first_error.errno
                                          if self._first_error is not None else None),
                    "coalesced": self._coalesced,
                    "close_status": self._close_status}

    def original_error(self):
        with self._lock:
            return self._first_error

    def close_after_writer(self):
        """Close once only after the original writer thread is not alive.

        The owner additionally verifies that its own output FD was closed.
        A close error is unknown and must not be retried by descriptor number.
        """
        if self._writer.ident is None or self._writer.is_alive():
            raise RuntimeError("original spool writer is still alive")
        return self._close_once()

    def close_before_writer(self):
        """Release a setup failure only if the original writer never started."""
        if self._writer.ident is not None:
            raise RuntimeError("original spool writer was started")
        return self._close_once()

    def _close_once(self):
        with self._lock:
            if self._close_started:
                return False
            self._close_started = True
            self._close_status = "unknown"
        error = None
        for fd in (self._read, self._write):
            try:
                os.close(fd)
            except OSError as caught:
                if error is None:
                    error = caught
        if error is not None:
            self._retain_error(error)
            return False
        with self._lock:
            self._close_status = "closed"
        return True
