"""In-process mock custody checks, not OS lock or GPU evidence."""
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from storage_consumer_broker_quarantine import BrokerQuarantine, BrokerQuarantineError


class FakeGroup:
    def __init__(self, *, has_child=True, gone=False):
        self._child = object() if has_child else None
        self.gone = gone

    def reobserve_and_drain(self, *, absolute_deadline, terminate):
        if terminate is not False:
            raise AssertionError("fixture must not signal")
        return self.gone, 0 if self.gone else None, None


class FakeLaunch:
    def __init__(self, group):
        self.group = group
        self._closed = False
        self.manager = SimpleNamespace(finish=self.finish_manager)
        self.manager_finished = False
        self.fail_launch = False

    def finish(self, *, tree_gone):
        if self.fail_launch:
            raise RuntimeError("synthetic partial close")
        if tree_gone is not True:
            raise AssertionError("no group completion")
        self._closed = True
        return True

    def finish_manager(self, *, exit_code, tree_gone, supervisor_error):
        if exit_code != 125 or tree_gone is not True:
            raise AssertionError("invalid original finalization")
        self.manager_finished = True


class FakeGate:
    def __init__(self, *, fail=False):
        self.fail = fail

    def finish(self):
        if self.fail:
            raise OSError("synthetic budget observation failure")


class BrokerQuarantineTests(unittest.TestCase):
    def test_manager_only_cleanup_unknown_is_retained_without_retry(self):
        table = BrokerQuarantine(maximum=2)
        key = table.reserve()
        table.retain_manager(key, object(), reason="pre-dispatch cleanup unknown")
        self.assertEqual(table.observation(key)["owner_kind"], "manager_finalization_unknown")
        with self.assertRaises(BrokerQuarantineError):
            table.reobserve(key, seconds=0.1)
        other = table.reserve()
        table.release_reservation(other)

    def test_bound_applies_only_to_reserved_opt_in_and_release(self):
        table = BrokerQuarantine(maximum=1)
        key = table.reserve()
        with self.assertRaises(BrokerQuarantineError):
            table.reserve()
        table.release_reservation(key)
        self.assertIsInstance(table.reserve(), str)

    def test_missing_original_popen_remains_unknown_and_retained(self):
        table = BrokerQuarantine()
        key = table.reserve()
        table.retain(key, FakeLaunch(FakeGroup(has_child=False)), FakeGate(), reason="spawn unknown")
        observation = table.reobserve(key, seconds=0.1)
        self.assertEqual(observation["state"], "original_popen_unavailable")
        self.assertTrue(observation["retained"])
        receipt = table.observation(key)
        self.assertEqual(receipt["state"], "retained")
        self.assertEqual(receipt["process_exit_lock_survival"], "not_claimed")
        self.assertEqual(receipt["other_catalog_independence"], "requires_disjoint_locks_and_dependencies")

    def test_same_owner_reobservation_records_budget_failure_separately(self):
        table = BrokerQuarantine()
        key = table.reserve()
        launch = FakeLaunch(FakeGroup(gone=True))
        table.retain(key, launch, FakeGate(fail=True), reason="group unknown")
        result = table.reobserve(key, seconds=0.1)
        self.assertTrue(launch.manager_finished)
        self.assertEqual(result["budget_error"], "OSError")
        self.assertEqual(result["physical_completion"], "unknown")
        self.assertFalse(result["retained"])
        with self.assertRaises(BrokerQuarantineError):
            table.observation(key)

    def test_partial_finalization_cannot_be_retried_as_new_owner(self):
        table = BrokerQuarantine()
        key = table.reserve()
        launch = FakeLaunch(FakeGroup(gone=True))
        launch.fail_launch = True
        table.retain(key, launch, FakeGate(), reason="group unknown")
        with self.assertRaises(RuntimeError):
            table.reobserve(key, seconds=0.1)
        with self.assertRaises(BrokerQuarantineError):
            table.reobserve(key, seconds=0.1)
        self.assertEqual(table.observation(key)["state"], "retained")

    def test_previously_uncertain_finalization_is_not_retried(self):
        table = BrokerQuarantine()
        key = table.reserve()
        launch = FakeLaunch(FakeGroup(gone=True))
        launch._closed = True
        table.retain(key, launch, FakeGate(), reason="partial close", finalization_unknown=True)
        with self.assertRaises(BrokerQuarantineError):
            table.reobserve(key, seconds=0.1)
        self.assertFalse(launch.manager_finished)

    def test_other_thread_cannot_take_original_owner(self):
        table = BrokerQuarantine()
        key = table.reserve()
        errors = []

        def other_thread():
            try:
                table.release_reservation(key)
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join()
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], BrokerQuarantineError)
        table.release_reservation(key)

    def test_same_numeric_thread_id_does_not_reassign_original_owner(self):
        table = BrokerQuarantine()
        key = table.reserve()
        with patch("storage_consumer_broker_quarantine.threading.current_thread", return_value=object()):
            with self.assertRaises(BrokerQuarantineError):
                table.release_reservation(key)
        table.release_reservation(key)


if __name__ == "__main__":
    unittest.main()
