"""Bounded asynchronous raw-output writer for a future LC0 parent.

The selector must retain a chunk refused by ``offer`` and pause only that
child pipe until there is capacity.  Such backpressure can also stall the UCI
child.  Other pipes, parent admission and cancellation remain independently
serviceable.  This object does not own a process or establish GPU completion.
"""
from __future__ import annotations

from collections import deque
import os
import threading

from lc0_output_spool import OutputSpool


# A stopped writer with unknown ownership must not lose its FD or pending data
# merely because a caller drops its handle. This process-local quarantine is
# not a recovery API and does not establish any GPU or storage completion.
_QUARANTINED_WRITERS = []


class AsyncOutputSpool:
    """One original FD and one non-daemon writer; no implicit timeout cleanup.

    ``max_queue_bytes`` and ``max_queue_items`` count both queued data and the
    currently writing chunk. Offer is nonblocking and transfers ownership only
    when it returns ``accepted``. Logical cancellation is an observation, not
    a request to close the writer or discard data. The caller must explicitly
    mark input EOF/read failure and request finish; close is attempted only
    after all accepted bytes leave the writer queue. A failed or blocked write
    keeps the same writer, FD and pending bytes until an explicit retry. An
    unexpected exception leaves the exact in-flight input retained and does
    not retry an uncertain write. The spool's own pending write and a chunk
    refused to the caller are separately bounded, not included in this queue
    limit. Callers must limit each read to ``max_queue_bytes`` or split it.
    """

    def __init__(self, fd, *, max_bytes, max_queue_bytes, max_queue_items):
        if any(type(value) is not int or value <= 0 for value in
               (max_bytes, max_queue_bytes, max_queue_items)):
            raise ValueError("spool and queue bounds must be positive integers")
        self._fd = os.dup(fd)
        self._max_bytes = max_bytes
        self._max_queue_bytes = max_queue_bytes
        self._max_queue_items = max_queue_items
        self._condition = threading.Condition()
        self._queue = deque()
        self._reserved_bytes = 0
        self._reserved_items = 0
        self._inflight = None
        self._state = "initializing"
        self._spool = None
        self._last_spool = None
        self._failure = None
        self._original_failure = None
        self._terminal = None
        self._terminal_applied = False
        self._finish_requested = False
        self._retry_requested = False
        self._cancel_observed = False
        self._backpressure_count = 0
        self._thread = threading.Thread(target=self._run, name="lc0-output-writer", daemon=False)
        try:
            self._thread.start()
        except BaseException:
            os.close(self._fd)
            self._fd = None
            raise

    def offer(self, data):
        """Return ``accepted``, ``backpressure``, ``oversized`` or ``unavailable``.

        A refused byte string remains entirely owned by the caller.  No disk
        operation is made under the queue lock or on the caller's thread.
        """
        if not isinstance(data, bytes):
            raise TypeError("output data must be bytes")
        with self._condition:
            if (self._terminal is not None or self._finish_requested
                    or self._state in ("write_blocked", "writer_fault", "init_failed",
                                       "closing", "close_unknown", "stopped")):
                return "unavailable"
            if not data:
                return "accepted"
            if len(data) > self._max_queue_bytes:
                return "oversized"
            if (self._reserved_items >= self._max_queue_items
                    or self._reserved_bytes + len(data) > self._max_queue_bytes):
                self._backpressure_count += 1
                return "backpressure"
            self._queue.append(data)
            self._reserved_bytes += len(data)
            self._reserved_items += 1
            self._condition.notify()
            return "accepted"

    def observe_eof(self):
        self._mark_terminal(("eof", None))

    def observe_read_error(self, error):
        if not isinstance(error, OSError):
            raise TypeError("source read failure must be an OSError")
        self._mark_terminal(("read_error", error))

    def _mark_terminal(self, terminal):
        with self._condition:
            if self._terminal is not None:
                raise ValueError("source stream already ended")
            self._terminal = terminal
            self._condition.notify()

    def note_logical_cancel(self):
        with self._condition:
            self._cancel_observed = True

    def request_retry(self):
        """Request one retry of known pending bytes, never uncertain writes."""
        with self._condition:
            if self._state != "write_blocked" or self._retry_requested:
                return False
            self._retry_requested = True
            self._condition.notify()
            return True

    def request_finish(self):
        """Request drain and close, without waiting or claiming completion."""
        with self._condition:
            if self._terminal is None:
                raise ValueError("source EOF or read failure must be observed first")
            if self._finish_requested or self._state in ("init_failed", "writer_fault", "close_unknown", "stopped"):
                return False
            self._finish_requested = True
            self._condition.notify()
            return True

    def observation(self):
        """Return a bounded snapshot; in-flight write progress is unknown."""
        with self._condition:
            return {
                "writer_state": self._state,
                "writer_thread_alive": self._thread.is_alive(),
                "queued_items_including_inflight": self._reserved_items,
                "queued_bytes_including_inflight": self._reserved_bytes,
                "inflight_bytes": len(self._inflight) if self._inflight is not None else 0,
                "inflight_write_progress": "unknown" if self._inflight is not None else "not_applicable",
                "backpressure_count": self._backpressure_count,
                "logical_cancel_observed": self._cancel_observed,
                "source_terminal": self._terminal[0] if self._terminal else "unknown",
                "source_terminal_applied": self._terminal_applied,
                "finish_requested": self._finish_requested,
                "failure_type": self._failure,
                "original_failure_retained": self._original_failure is not None,
                "last_completed_spool_observation": (self._last_spool.copy()
                                                      if self._last_spool is not None else None),
                "physical_gpu_completion": "not_observed",
            }

    def _run(self):
        try:
            spool = OutputSpool(self._fd, self._max_bytes)
        except BaseException as error:
            with self._condition:
                self._failure = type(error).__name__
                self._original_failure = error
                self._state = "init_failed"
                _QUARANTINED_WRITERS.append(self)
            return
        with self._condition:
            self._spool = spool
            self._last_spool = spool.observation()
            self._state = "idle"

        while True:
            with self._condition:
                while True:
                    if self._state == "write_blocked":
                        if self._retry_requested:
                            self._retry_requested = False
                            self._state = "writing"
                            operation = "retry"
                            break
                    elif self._queue:
                        self._inflight = self._queue.popleft()
                        self._state = "writing"
                        operation = "write"
                        break
                    elif self._terminal is not None and not self._terminal_applied:
                        self._state = "applying_terminal"
                        operation = "terminal"
                        break
                    elif self._finish_requested and self._terminal_applied:
                        self._state = "closing"
                        operation = "close"
                        break
                    self._condition.wait()

            try:
                if operation == "write":
                    spool.accept(self._inflight)
                elif operation == "retry":
                    spool.retry_pending()
                elif operation == "terminal":
                    kind, error = self._terminal
                    if kind == "eof":
                        spool.observe_eof()
                    else:
                        spool.observe_read_error(error)
                else:
                    spool.close()
                    os.close(self._fd)
                    self._fd = None
            except BaseException as error:
                with self._condition:
                    self._failure = self._failure or type(error).__name__
                    if self._original_failure is None:
                        self._original_failure = error
                    self._state = "close_unknown" if operation == "close" else "writer_fault"
                    _QUARANTINED_WRITERS.append(self)
                return

            with self._condition:
                self._last_spool = spool.observation()
                if operation == "close":
                    self._state = "stopped"
                    return
                if operation == "terminal":
                    self._terminal_applied = True
                elif self._last_spool["write_blocked"]:
                    self._state = "write_blocked"
                    continue
                elif self._inflight is not None:
                    self._reserved_bytes -= len(self._inflight)
                    self._reserved_items -= 1
                    self._inflight = None
                self._state = "idle"
