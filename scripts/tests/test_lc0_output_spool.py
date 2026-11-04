"""CPU-only tests for bounded original output custody, without an engine."""
import os
import errno
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lc0_output_spool import OutputSpool, OutputSpoolInitError


@unittest.skipUnless(hasattr(os, "pwrite"), "positional file writes required")
class OutputSpoolTests(unittest.TestCase):
    def setUp(self):
        self.file = tempfile.TemporaryFile()
        self.addCleanup(self.file.close)

    def test_prefix_and_truncation_are_independent_of_eof(self):
        spool = OutputSpool(self.file.fileno(), 5)
        spool.accept(b"abcdefg")
        observed = spool.observation()
        self.assertEqual((observed["bytes_seen"], observed["bytes_written"],
                          observed["bytes_truncated"]), (7, 5, 2))
        self.assertFalse(observed["eof_observed"])
        self.assertEqual(os.pread(self.file.fileno(), 10, 0), b"abcde")
        spool.observe_eof()
        self.assertTrue(spool.observation()["eof_observed"])
        spool.close()

    def test_partial_write_and_error_retain_original_suffix_for_retry(self):
        spool = OutputSpool(self.file.fileno(), 20)
        pwrite = os.pwrite
        original_error = OSError(errno.ENOSPC, "disk failed")

        def short_then_error(fd, data, offset):
            if offset == 0:
                return pwrite(fd, data[:2], offset)
            raise original_error

        with mock.patch("lc0_output_spool.os.pwrite", side_effect=short_then_error):
            spool.accept(b"abcdef")
        self.assertEqual(spool.observation()["bytes_written"], 2)
        self.assertEqual(spool.observation()["bytes_pending"], 4)
        with self.assertRaises(RuntimeError):
            spool.close()
        spool.retry_pending()
        spool.accept(b"g")
        self.assertEqual(os.pread(self.file.fileno(), 20, 0), b"abcdefg")
        self.assertEqual(spool.observation()["first_write_error"], "OSError")
        self.assertEqual(spool.observation()["first_write_errno"], errno.ENOSPC)
        self.assertIs(spool.original_errors()[1], original_error)
        self.assertFalse(spool.observation()["write_blocked"])
        spool.close()

    def test_external_file_change_does_not_write_over_it(self):
        spool = OutputSpool(self.file.fileno(), 20)
        os.pwrite(self.file.fileno(), b"outside", 0)
        spool.accept(b"inside")
        self.assertTrue(spool.observation()["write_blocked"])
        self.assertEqual(spool.observation()["bytes_pending"], len(b"inside"))
        self.assertEqual(os.pread(self.file.fileno(), 20, 0), b"outside")
        os.ftruncate(self.file.fileno(), 0)
        spool.retry_pending()
        self.assertEqual(os.pread(self.file.fileno(), 20, 0), b"inside")
        spool.close()

    def test_read_failure_is_not_eof(self):
        spool = OutputSpool(self.file.fileno(), 20)
        original_error = OSError(errno.EIO, "pipe failed")
        spool.observe_read_error(original_error)
        self.assertFalse(spool.observation()["eof_observed"])
        self.assertEqual(spool.observation()["read_error"], "OSError")
        self.assertEqual(spool.observation()["read_errno"], errno.EIO)
        self.assertIs(spool.original_errors()[0], original_error)
        with self.assertRaises(ValueError):
            spool.accept(b"late")
        spool.close()

    def test_failed_initialization_preserves_original_and_ambiguous_close(self):
        original = OSError(errno.EIO, "private initialization detail")
        cleanup = OSError(errno.EIO, "private cleanup detail")
        close = os.close

        def close_then_fail(fd):
            close(fd)
            raise cleanup

        with mock.patch("lc0_output_spool.os.fstat", side_effect=original), \
                mock.patch("lc0_output_spool.os.close", side_effect=close_then_fail):
            with self.assertRaises(OutputSpoolInitError) as caught:
                OutputSpool(self.file.fileno(), 20)
        self.assertIs(caught.exception.original_error, original)
        self.assertIs(caught.exception.close_error, cleanup)
        self.assertNotIn("private", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
