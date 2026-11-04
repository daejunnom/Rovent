"""Finite custody deadlines and non-sensitive failure observations."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import managed_output_quarantine as custody
import run_managed
from managed_storage import StorageError


class CustodyDeadlineDiagnosticTests(unittest.TestCase):
    def test_original_custody_limit_remains_thirty_seconds(self):
        with patch.object(custody.sys, "platform", "linux"):
            with self.assertRaisesRegex(StorageError, "finite <=30s window"):
                custody._Held(31)

    def test_explicit_session_is_bounded_by_caller_deadline_and_reports_stage(self):
        with patch.object(custody.sys, "platform", "linux"), \
                patch.object(custody.time, "monotonic", side_effect=[100.0, 149.9, 150.0]):
            held = custody._Held._for_bounded_session(90, 150.0)
            self.assertEqual(held.until, 150.0)
            held.mark_stage("tree_snapshot")
            held.tick()
            with self.assertRaisesRegex(StorageError, "original custody observation deadline expired") as caught:
                held.tick()
        observation = caught.exception.custody_deadline_observation
        self.assertEqual(observation["stage"], "tree_snapshot")
        self.assertEqual(observation["elapsed_seconds"], 50.0)
        self.assertEqual(observation["window_seconds"], 50.0)
        self.assertEqual(held.until, 150.0)
        stream = io.StringIO()
        with patch.object(run_managed.sys, "stderr", stream):
            run_managed._emit_custody_deadline_observation(caught.exception)
        prefix, raw = stream.getvalue().strip().split("=", 1)
        self.assertEqual(prefix, "custody_deadline_observation")
        self.assertEqual(json.loads(raw), observation)
        self.assertNotIn("path", raw)

    def test_explicit_session_rejects_nonfinite_expired_and_unbounded_inputs(self):
        with patch.object(custody.sys, "platform", "linux"), \
                patch.object(custody.time, "monotonic", return_value=100.0):
            for duration, deadline in [(0, 150.0), (float("inf"), 150.0),
                                       (1, float("nan")), (1, 100.0)]:
                with self.subTest(duration=duration, deadline=deadline), self.assertRaises(StorageError):
                    custody._Held._for_bounded_session(duration, deadline)
            self.assertEqual(custody._Held._for_bounded_session(10, 150.0).until, 110.0)

    def test_broken_diagnostic_stream_does_not_replace_error(self):
        class Broken:
            def write(self, _text):
                raise OSError("closed")

        error = StorageError("original custody observation deadline expired")
        error.custody_deadline_observation = {
            "schema": "rovezero.custody-deadline-observation.v1", "stage": "observation",
            "elapsed_seconds": 1.0, "window_seconds": 1.0,
        }
        with patch.object(run_managed.sys, "stderr", Broken()):
            run_managed._emit_custody_deadline_observation(error)

    def test_output_tree_expiry_reports_accounting_before_admission(self):
        with patch.object(custody.sys, "platform", "linux"), \
                patch.object(custody.time, "monotonic", side_effect=[100.0, 100.1, 100.2, 101.0]), \
                patch.object(custody, "tree_size", return_value=1):
            held = custody._Held._for_bounded_session(1, 200.0)
            view = custody.QuarantineValidation(held, None, None, None, {}, 0, (), ())
            with self.assertRaises(StorageError) as caught:
                view.usage([(None, {"key": "row", "reserved_bytes": 10}, Path("unread"))])
        self.assertEqual(caught.exception.custody_deadline_observation["stage"], "output_accounting")

    @unittest.skipUnless(sys.platform == "linux", "requires real no-follow Linux descriptor")
    def test_mismatched_source_owner_still_fails_with_bounded_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.py"
            raw = b"fixture only\n"
            path.write_bytes(raw)
            pin = custody._pin(path, raw)
            held = custody._Held(10)
            try:
                actual_uid = os.geteuid()
                with patch.object(custody.os, "geteuid", return_value=actual_uid + 1):
                    with self.assertRaisesRegex(StorageError, "FD/type/owner/length changed") as caught:
                        held.read(pin, path, len(raw))
                observation = caught.exception.custody_metadata_failure_observation
                self.assertFalse(observation["owner_matches_euid"])
                self.assertTrue(observation["regular_file"])
                self.assertTrue(observation["single_link"])
                self.assertTrue(observation["length_matches_pin"])
                self.assertNotIn(str(path), json.dumps(observation))
            finally:
                held.close()


if __name__ == "__main__":
    unittest.main()
