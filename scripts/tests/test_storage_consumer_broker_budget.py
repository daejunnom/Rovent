"""Mock timing/order checks; no OS resource enforcement is exercised."""
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from storage_consumer_broker_budget import BrokerBudgetGate


class BrokerBudgetGateTests(unittest.TestCase):
    def test_deferred_original_sampler_runs_immediately_after_transaction(self):
        status = {"active": True}
        samples = []
        gate = BrokerBudgetGate(lambda: samples.append("original"), lambda: status["active"])
        self.assertFalse(gate.sample())
        self.assertTrue(gate.deferred)
        status["active"] = False
        self.assertTrue(gate.after_tick())
        self.assertEqual(samples, ["original"])
        self.assertFalse(gate.deferred)

    def test_final_done_requires_original_sampler_if_dirty(self):
        status = {"active": True}
        samples = []
        gate = BrokerBudgetGate(lambda: samples.append("original"), lambda: status["active"])
        gate.sample()
        with self.assertRaises(RuntimeError):
            gate.finish()
        self.assertTrue(gate.deferred)
        status["active"] = False
        self.assertTrue(gate.finish())
        self.assertEqual(samples, ["original"])

    def test_sampler_failure_propagates_and_keeps_dirty(self):
        def fail():
            raise OSError("original budget failure")
        gate = BrokerBudgetGate(fail, lambda: False)
        with self.assertRaisesRegex(OSError, "original budget failure"):
            gate.sample()
        self.assertTrue(gate.deferred)
        with self.assertRaises(OSError):
            gate.finish()
        self.assertTrue(gate.deferred)

    def test_active_observation_error_requires_later_original_sample(self):
        observations = iter((OSError("observation failed"), False))
        samples = []

        def active():
            result = next(observations)
            if isinstance(result, Exception):
                raise result
            return result

        gate = BrokerBudgetGate(lambda: samples.append("original"), active)
        with self.assertRaises(OSError):
            gate.sample()
        self.assertTrue(gate.deferred)
        # finish checks active once and sample checks it again.
        observations = iter((False, False))
        self.assertTrue(gate.finish())
        self.assertEqual(samples, ["original"])

    def test_other_thread_cannot_sample_between_original_owner_operations(self):
        samples, errors = [], []
        gate = BrokerBudgetGate(lambda: samples.append("original"), lambda: False)

        def worker():
            try:
                gate.sample()
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertEqual(samples, [])

    def test_reused_numeric_thread_id_does_not_imply_same_thread_object(self):
        gate = BrokerBudgetGate(lambda: None, lambda: False)
        with patch("storage_consumer_broker_budget.threading.current_thread", return_value=object()):
            with self.assertRaises(RuntimeError):
                gate.sample()


if __name__ == "__main__":
    unittest.main()
