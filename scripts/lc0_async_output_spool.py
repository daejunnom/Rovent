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
from lc0_spool_notification import CoalescedSpoolNotification


# A stopped writer with unknown ownership must not lose its FD or pending data
# merely because a caller drops its handle. This process-local quarantine is
# not a recovery API and does not establish any GPU or storage completion.
_QUARANTINED_WRITERS = []


class AsyncSpoolInitializationCleanupError(Exception):
    """Retain the setup error and every uncertain one-shot cleanup failure."""

    def __init__(self, original, cleanup_failures):
        super().__init__("output writer initialization cleanup is uncertain")
        self.original = original
        self.cleanup_failures = cleanup_failures


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
        self._fd = None
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
        self._last_errors = (None, None)
        self._failure = None
        self._original_failure = None
        self._initialization_cleanup_failures = ()
        self._terminal = None
        self._terminal_applied = False
        self._finish_requested = False
        self._retry_requested = False
        self._cancel_observed = False
        self._backpressure_count = 0
        self._thread = threading.Thread(target=self._run_guarded, name="lc0-output-writer", daemon=False)
        self._notification = None
        self._notification_init_error = None
        start_invoked = False
        created_notification = None
        owned_fd = os.dup(fd)
        try:
            self._fd = owned_fd
            try:
                created_notification = CoalescedSpoolNotification(self._thread)
            except (OSError, AttributeError) as error:
                # The owning selector can use its bounded tick. This failure
                # does not alter another execution or the output operation.
                self._notification_init_error = error
            else:
                self._notification = created_notification
            # Once start is called, even `ident is None` does not establish
            # that no native thread was created (start may be interrupted).
            start_invoked = True
            self._thread.start()
        except BaseException as original:
            if start_invoked:
                # No assumed stop, close, or retry: retain the same owner even
                # when the attempted start's actual physical result is unknown.
                self._failure = type(original).__name__
                self._original_failure = original
                _QUARANTINED_WRITERS.append(self)
                raise
            failures = []
            if created_notification is not None:
                try:
                    if not created_notification.close_before_writer():
                        failures.extend(created_notification.original_close_errors())
                        if not failures:
                            failures.append((None, RuntimeError("notification cleanup unknown")))
                except BaseException as error:
                    failures.append((None, error))
            try:
                os.close(owned_fd)
            except BaseException as error:
                failures.append((owned_fd, error))
            else:
                self._fd = None
            if failures:
                self._failure = type(original).__name__
                self._original_failure = original
                self._initialization_cleanup_failures = tuple(failures)
                _QUARANTINED_WRITERS.append(self)
                raise AsyncSpoolInitializationCleanupError(original, self._initialization_cleanup_failures) from original
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
        """Re-read state after drain; notification and writer snapshots are separate."""
        with self._condition:
            result = {
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
        result["notification"] = (self._notification.observation() if self._notification
                                  else {"first_error_type": (type(self._notification_init_error).__name__
                                                             if self._notification_init_error is not None else None),
                                        "first_error_errno": getattr(self._notification_init_error, "errno", None),
                                        "coalesced": 0, "close_status": "unavailable"})
        return result

    def notification_fileno(self):
        return self._notification.fileno() if self._notification else -1

    def drain_notification(self):
        """Drain one bounded wakeup batch, then call ``observation``."""
        return self._notification.drain() if self._notification else None

    def notification_original_error(self):
        return (self._notification.original_error() if self._notification
                else self._notification_init_error)

    def release_notification(self):
        """Unregister from the selector first; close only after writer/FD finish."""
        with self._condition:
            if self._state != "stopped" or self._thread.is_alive() or self._fd is not None:
                raise RuntimeError("original output writer has not fully stopped")
        return self._notification.close_after_writer() if self._notification else False

    def _signal(self):
        if self._notification is not None:
            self._notification.signal()

    def original_errors(self):
        """Return retained worker, read and write errors for this same owner.

        Read/write errors are the last completed spool observation; a write
        currently in flight can still fail later. No disk operation or wait is
        performed while acquiring these references.
        """
        with self._condition:
            return (self._original_failure, *self._last_errors)

    def original_initialization_cleanup_failures(self):
        """Return the exact one-shot FD cleanup failures, if any."""
        with self._condition:
            return self._initialization_cleanup_failures

    def _run_guarded(self):
        # An unexpected failure in bookkeeping or notification must retain
        # the same output owner just like an exception from the disk call.
        try:
            self._run()
        except BaseException as error:
            with self._condition:
                self._failure = self._failure or type(error).__name__
                if self._original_failure is None:
                    self._original_failure = error
                self._state = "close_unknown" if self._state == "closing" else "writer_fault"
                _QUARANTINED_WRITERS.append(self)
            try:
                self._signal()
            except BaseException:
                # The original writer exception and all pending bytes remain
                # retained; notification is not a completion authority.
                pass

    def _run(self):
        try:
            spool = OutputSpool(self._fd, self._max_bytes)
        except BaseException as error:
            with self._condition:
                self._failure = type(error).__name__
                self._original_failure = error
                self._state = "init_failed"
                _QUARANTINED_WRITERS.append(self)
            self._signal()
            return
        with self._condition:
            self._spool = spool
            self._last_spool = spool.observation()
            self._last_errors = spool.original_errors()
            self._state = "idle"
        self._signal()

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

            self._signal()
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
                    self._last_errors = spool.original_errors()
                    self._state = "close_unknown" if operation == "close" else "writer_fault"
                    _QUARANTINED_WRITERS.append(self)
                self._signal()
                return

            with self._condition:
                self._last_spool = spool.observation()
                self._last_errors = spool.original_errors()
                if operation == "close":
                    self._state = "stopped"
                    stop = True
                elif operation == "terminal":
                    self._terminal_applied = True
                    self._state = "idle"
                    stop = False
                elif self._last_spool["write_blocked"]:
                    self._state = "write_blocked"
                    stop = False
                else:
                    if self._inflight is not None:
                        self._reserved_bytes -= len(self._inflight)
                        self._reserved_items -= 1
                        self._inflight = None
                    self._state = "idle"
                    stop = False
            self._signal()
            if stop:
                return
