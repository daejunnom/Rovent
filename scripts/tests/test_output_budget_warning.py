"""Soft output-limit diagnostics cannot revoke ownership or stop work."""
from __future__ import annotations

import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import managed_storage


class OutputBudgetWarningTests(unittest.TestCase):
    def test_accounting_exceedance_is_not_a_disk_warning(self):
        sink = io.StringIO()
        with patch.object(sys, "stderr", sink):
            self.assertFalse(managed_storage.warn_output_threshold(18, 18, scope="test"))
            self.assertTrue(managed_storage.warn_output_threshold(19, 18, scope="test"))
        self.assertEqual(sink.getvalue(), "")

    def test_failed_warning_stream_does_not_block_the_observed_execution(self):
        class BrokenStream:
            def write(self, _value):
                raise OSError("closed diagnostic")

        with patch.object(sys, "stderr", BrokenStream()):
            self.assertTrue(managed_storage.warn_output_threshold(19, 18, scope="broken"))

    def test_reservations_do_not_claim_actual_disk_exceedance(self):
        sink = io.StringIO()
        with patch.object(sys, "stderr", sink):
            self.assertTrue(managed_storage.warn_output_threshold(
                51_609_195_897, 18 * managed_storage.GiB,
                scope="output_accounted", observed_bytes=1234))
            self.assertTrue(managed_storage.warn_output_threshold(
                19, 18, scope="unattributed"))
        self.assertEqual(sink.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
