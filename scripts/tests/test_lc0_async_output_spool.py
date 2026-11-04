import os
import errno
import selectors
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lc0_async_output_spool as async_spool
import lc0_output_spool
import lc0_spool_notification


def wait_state(spool, wanted, timeout=3):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        observed = spool.observation()
        if observed["writer_state"] == wanted:
            if wanted != "stopped" or not observed["writer_thread_alive"]:
                return observed
        time.sleep(0.005)
    raise AssertionError(f"writer did not reach {wanted}: {spool.observation()}")


@unittest.skipUnless(sys.platform == "linux", "requires Linux positional writes and selector pipe")
class AsyncOutputSpoolTests(unittest.TestCase):
    def test_blocked_write_does_not_block_offer_cancel_or_retain_late_bytes(self):
        entered = threading.Event()
        release = threading.Event()
        original = os.pwrite

        def delayed(fd, data, offset):
            entered.set()
            if not release.wait(3):
                raise OSError("fixture writer deadline")
            return original(fd, data, offset)

        with tempfile.TemporaryFile() as output, mock.patch.object(lc0_output_spool.os, "pwrite", delayed):
            spool = async_spool.AsyncOutputSpool(output.fileno(), max_bytes=64,
                                                  max_queue_bytes=8, max_queue_items=2)
            try:
                self.assertEqual(spool.offer(b"abcd"), "accepted")
                self.assertTrue(entered.wait(2))
                self.assertEqual(spool.offer(b"efgh"), "accepted")
                self.assertEqual(spool.offer(b"i"), "backpressure")
                self.assertEqual(spool.offer(b"123456789"), "oversized")
                spool.note_logical_cancel()
                spool.observe_eof()
                self.assertTrue(spool.request_finish())
                blocked = spool.observation()
                self.assertEqual(blocked["writer_state"], "writing")
                self.assertEqual(blocked["queued_bytes_including_inflight"], 8)
                self.assertEqual(blocked["inflight_write_progress"], "unknown")
                self.assertTrue(blocked["logical_cancel_observed"])
                self.assertEqual(blocked["backpressure_count"], 1)
                self.assertGreaterEqual(os.fstat(spool._fd).st_size, 0)
            finally:
                release.set()
                try:
                    spool.observe_eof()
                except ValueError:
                    pass
                try:
                    spool.request_finish()
                except ValueError:
                    pass
                spool._thread.join(3)
            final = wait_state(spool, "stopped")
            self.assertEqual(final["queued_bytes_including_inflight"], 0)
            self.assertTrue(final["last_completed_spool_observation"]["eof_observed"])
            self.assertEqual(os.pread(output.fileno(), 64, 0), b"abcdefgh")
            self.assertTrue(spool.release_notification())

    def test_partial_write_error_retains_original_and_requires_explicit_retry(self):
        original = os.pwrite
        calls = []
        failing = True
        disk_error = OSError(errno.EIO, "fixture disk error")

        def partial(fd, data, offset):
            calls.append(offset)
            if failing and offset:
                raise disk_error
            return original(fd, data[:2] if failing else data, offset)

        with tempfile.TemporaryFile() as output, mock.patch.object(lc0_output_spool.os, "pwrite", partial):
            spool = async_spool.AsyncOutputSpool(output.fileno(), max_bytes=64,
                                                  max_queue_bytes=16, max_queue_items=2)
            try:
                self.assertEqual(spool.offer(b"abcde"), "accepted")
                blocked = wait_state(spool, "write_blocked")
                self.assertEqual(blocked["inflight_bytes"], 5)
                self.assertEqual(blocked["last_completed_spool_observation"]["bytes_pending"], 3)
                self.assertIs(spool.original_errors()[2], disk_error)
                self.assertEqual(spool.offer(b"later"), "unavailable")
                self.assertEqual(calls, [0, 2])
                failing = False
                self.assertTrue(spool.request_retry())
                wait_state(spool, "idle")
                spool.observe_eof()
                spool.request_finish()
            finally:
                failing = False
                spool.request_retry()
                try:
                    spool.observe_eof()
                except ValueError:
                    pass
                try:
                    spool.request_finish()
                except ValueError:
                    pass
                spool._thread.join(3)
            final = wait_state(spool, "stopped")
            self.assertEqual(final["last_completed_spool_observation"]["first_write_error"], "OSError")
            self.assertIs(spool.original_errors()[2], disk_error)
            self.assertEqual(os.pread(output.fileno(), 64, 0), b"abcde")
            self.assertTrue(spool.release_notification())

    def test_unexpected_writer_exception_quarantines_fd_and_inflight(self):
        with tempfile.TemporaryFile() as output, mock.patch.object(
            lc0_output_spool.OutputSpool, "accept", side_effect=MemoryError
        ):
            spool = async_spool.AsyncOutputSpool(output.fileno(), max_bytes=64,
                                                  max_queue_bytes=16, max_queue_items=2)
            self.assertEqual(spool.offer(b"source"), "accepted")
            failed = wait_state(spool, "writer_fault")
            self.assertEqual(failed["inflight_bytes"], 6)
            self.assertEqual(failed["failure_type"], "MemoryError")
            self.assertTrue(failed["original_failure_retained"])
            self.assertEqual(spool.offer(b"next"), "unavailable")
            self.assertIs(async_spool._QUARANTINED_WRITERS.pop(), spool)
            # Fixture cleanup only after the original writer thread has exited.
            spool._thread.join(1)
            self.assertFalse(spool._thread.is_alive())
            os.close(spool._spool._fd)
            os.close(spool._fd)
            spool._spool._fd = None
            spool._fd = None
            self.assertTrue(spool._notification.close_after_writer())

    def test_bookkeeping_failure_after_write_retains_original_owner(self):
        original_observation = lc0_output_spool.OutputSpool.observation

        def fail_after_write(owner):
            if owner.bytes_written:
                raise RuntimeError("fixture observation failure")
            return original_observation(owner)

        with tempfile.TemporaryFile() as output, mock.patch.object(
            lc0_output_spool.OutputSpool, "observation", fail_after_write
        ):
            spool = async_spool.AsyncOutputSpool(output.fileno(), max_bytes=64,
                                                  max_queue_bytes=16, max_queue_items=2)
            self.assertEqual(spool.offer(b"source"), "accepted")
            failed = wait_state(spool, "writer_fault")
            self.assertEqual(failed["inflight_bytes"], 6)
            self.assertTrue(failed["original_failure_retained"])
            self.assertEqual(os.pread(output.fileno(), 64, 0), b"source")
            self.assertIs(async_spool._QUARANTINED_WRITERS.pop(), spool)
            spool._thread.join(1)
            self.assertFalse(spool._thread.is_alive())
            os.close(spool._spool._fd)
            os.close(spool._fd)
            spool._spool._fd = None
            spool._fd = None
            self.assertTrue(spool._notification.close_after_writer())

    def test_read_error_is_not_reported_as_eof(self):
        with tempfile.TemporaryFile() as output:
            spool = async_spool.AsyncOutputSpool(output.fileno(), max_bytes=64,
                                                  max_queue_bytes=16, max_queue_items=2)
            self.assertEqual(spool.offer(b"data"), "accepted")
            read_error = OSError(errno.EIO, "fixture read error")
            spool.observe_read_error(read_error)
            self.assertTrue(spool.request_finish())
            final = wait_state(spool, "stopped")
            observed = final["last_completed_spool_observation"]
            self.assertEqual(observed["read_error"], "OSError")
            self.assertFalse(observed["eof_observed"])
            self.assertIs(spool.original_errors()[1], read_error)
            self.assertEqual(os.pread(output.fileno(), 64, 0), b"data")
            self.assertTrue(spool.release_notification())

    def test_selector_wakes_then_reobserves_original_writer(self):
        entered, release = threading.Event(), threading.Event()
        original = os.pwrite

        def delayed(fd, data, offset):
            entered.set()
            if not release.wait(3):
                raise OSError("fixture writer deadline")
            return original(fd, data, offset)

        with tempfile.TemporaryFile() as output, mock.patch.object(lc0_output_spool.os, "pwrite", delayed):
            spool = async_spool.AsyncOutputSpool(output.fileno(), max_bytes=64,
                                                  max_queue_bytes=8, max_queue_items=1)
            selector = selectors.DefaultSelector()
            try:
                selector.register(spool.notification_fileno(), selectors.EVENT_READ)
                self.assertEqual(spool.offer(b"data"), "accepted")
                self.assertTrue(entered.wait(2))
                while spool.drain_notification():
                    pass
                self.assertEqual(spool.observation()["writer_state"], "writing")
                with self.assertRaises(RuntimeError):
                    spool.release_notification()
                release.set()
                self.assertTrue(selector.select(2))
                self.assertIsNotNone(spool.drain_notification())
                wait_state(spool, "idle")
                spool.observe_eof()
                spool.request_finish()
                wait_state(spool, "stopped")
                selector.unregister(spool.notification_fileno())
                self.assertTrue(spool.release_notification())
                self.assertEqual(spool.notification_fileno(), -1)
                self.assertIsNone(spool.drain_notification())
            finally:
                release.set()
                selector.close()
                try:
                    spool.observe_eof()
                except ValueError:
                    pass
                try:
                    spool.request_finish()
                except ValueError:
                    pass
                spool._thread.join(3)
                if not spool._thread.is_alive() and spool.notification_fileno() >= 0:
                    spool.release_notification()

    def test_full_notification_pipe_coalesces_without_losing_readability(self):
        writer = threading.Thread(target=lambda: None)
        writer.start()
        writer.join()
        notification = lc0_spool_notification.CoalescedSpoolNotification(writer)
        try:
            while True:
                try:
                    os.write(notification._write, b"x" * 4096)
                except BlockingIOError:
                    break
            self.assertEqual(notification.signal(), "coalesced")
            self.assertEqual(notification.observation()["coalesced"], 1)
            self.assertGreater(notification.drain(), 0)
        finally:
            self.assertTrue(notification.close_after_writer())

    def test_notification_failure_does_not_fail_writer(self):
        notification_error = OSError(errno.ENFILE, "fixture notification unavailable")
        with tempfile.TemporaryFile() as output, mock.patch.object(
            async_spool, "CoalescedSpoolNotification", side_effect=notification_error
        ):
            spool = async_spool.AsyncOutputSpool(output.fileno(), max_bytes=64,
                                                  max_queue_bytes=16, max_queue_items=2)
            self.assertEqual(spool.offer(b"data"), "accepted")
            spool.observe_eof()
            spool.request_finish()
            final = wait_state(spool, "stopped")
            self.assertEqual(final["notification"]["close_status"], "unavailable")
            self.assertIs(spool.notification_original_error(), notification_error)
            self.assertEqual(os.pread(output.fileno(), 64, 0), b"data")
            self.assertFalse(spool.release_notification())

    def test_notification_signal_failure_retains_original_error(self):
        writer = threading.Thread(target=lambda: None)
        writer.start()
        writer.join()
        notification = lc0_spool_notification.CoalescedSpoolNotification(writer)
        failure = OSError(errno.EIO, "fixture pipe write failure")
        try:
            with mock.patch.object(lc0_spool_notification.os, "write", side_effect=failure):
                self.assertEqual(notification.signal(), "failed")
            self.assertIs(notification.original_error(), failure)
            self.assertEqual(notification.observation()["first_error_errno"], errno.EIO)
        finally:
            self.assertTrue(notification.close_after_writer())


if __name__ == "__main__":
    unittest.main()
