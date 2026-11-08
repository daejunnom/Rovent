"""CPU-only facade checks; mocks do not attest a real catalog lock."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from virtual_system_numeric_prepare import NumericInputRegistration
from rovent_v3_parent import _registered_read


class FakeGuard:
    def __init__(self):
        self.locked = False
        self.recorded = None

    def __enter__(self):
        self.locked = True
        return self

    def __exit__(self, *args):
        self.locked = False

    def register(self, inputs, *, future_inputs):
        if not self.locked:
            raise AssertionError("descriptor and registration require one guard")
        self.recorded = (inputs, future_inputs)
        return object()


class NumericConsumerFutureTests(unittest.TestCase):
    def test_v3_reader_registers_existing_and_future_under_same_guard(self):
        consumer = NumericInputRegistration.__new__(NumericInputRegistration)
        consumer.registry = object()
        consumer.leases = []
        guard = FakeGuard()
        consumer._locked = lambda stack: stack.enter_context(guard)

        def reader():
            self.assertTrue(guard.locked)
            return "value", ["/asset", "/asset"], ["/asset", "/output", "/output"]

        self.assertEqual(_registered_read(consumer, reader), "value")
        self.assertEqual(guard.recorded, ([Path("/asset")], [Path("/output")]))
        self.assertFalse(guard.locked)
        self.assertEqual(len(consumer.leases), 1)

    def test_nonparticipating_legacy_reader_is_unchanged(self):
        consumer = NumericInputRegistration.__new__(NumericInputRegistration)
        consumer.registry = None
        self.assertEqual(_registered_read(consumer, lambda: ("value", [], [])), "value")


if __name__ == "__main__":
    unittest.main()
