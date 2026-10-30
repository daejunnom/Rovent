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
    def setUp(self):
        self.old_scopes = set(managed_storage._warned_output_scopes)
        managed_storage._warned_output_scopes.clear()

    def tearDown(self):
        managed_storage._warned_output_scopes.clear()
        managed_storage._warned_output_scopes.update(self.old_scopes)

    def test_warning_is_emitted_once_per_exceedance_and_rearmed_after_recovery(self):
        sink = io.StringIO()
        with patch.object(sys, "stderr", sink):
            self.assertFalse(managed_storage.warn_output_threshold(18, 18, scope="test"))
            self.assertTrue(managed_storage.warn_output_threshold(19, 18, scope="test"))
            self.assertTrue(managed_storage.warn_output_threshold(20, 18, scope="test"))
            self.assertFalse(managed_storage.warn_output_threshold(17, 18, scope="test"))
            self.assertTrue(managed_storage.warn_output_threshold(19, 18, scope="test"))
        self.assertEqual(sink.getvalue().count("rovezero_storage_warning"), 2)

    def test_failed_warning_stream_does_not_block_the_observed_execution(self):
        class BrokenStream:
            def write(self, _value):
                raise OSError("closed diagnostic")

        with patch.object(sys, "stderr", BrokenStream()):
            self.assertTrue(managed_storage.warn_output_threshold(19, 18, scope="broken"))

    def test_reservations_are_not_reported_as_actual_usage(self):
        sink = io.StringIO()
        with patch.object(sys, "stderr", sink):
            self.assertTrue(managed_storage.warn_output_threshold(
                51_609_195_897, 18 * managed_storage.GiB,
                scope="output_accounted", observed_bytes=1234))
            self.assertTrue(managed_storage.warn_output_threshold(
                19, 18, scope="unattributed"))
        message = sink.getvalue()
        self.assertIn("accounted_bytes=51609195897", message)
        self.assertIn("may include reservations; not actual disk usage", message)
        self.assertIn("observed_logical_bytes=1234", message)
        self.assertIn("observed_logical_bytes=unknown", message)
        self.assertIn("remaining_actual_bytes=unknown", message)
        self.assertNotIn("output_actual", message)


if __name__ == "__main__":
    unittest.main()
