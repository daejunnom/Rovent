"""CPU-only facade checks; mocks do not attest a real catalog lock."""
import sys
import unittest
import os
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from virtual_system_numeric_prepare import NumericInputRegistration
from rovent_v3_parent import _registered_read
from managed_storage import StorageError


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
    def test_constructor_failure_finishes_only_validated_broker_and_preserves_error(self):
        args = SimpleNamespace(consumer_registry_root=None, consumer_registry_id=None,
                               consumer_registry_required=False, consumer_input_catalog=[])
        env = {"RZ_CONSUMER_REGISTRY_ROOT": "/synthetic/registry", "RZ_CONSUMER_REGISTRY_ID": "id",
               "RZ_CONSUMER_BROKER_FD": "3", "RZ_CONSUMER_BROKER_PARENT_PID": "123",
               "RZ_CONSUMER_INPUT_CATALOGS": json.dumps(["/synthetic/allowed"])}
        actions = []

        class Broker:
            def add(self, inputs):
                raise ValueError("initial input failure")

            def finish(self):
                actions.append("done")
                raise RuntimeError("ack unavailable")

        with patch.dict(os.environ, env, clear=True), patch(
                "storage_consumer_registry.ConsumerRegistry", return_value=object()), patch(
                "storage_consumer_broker_client.inherited_supervisor_client", return_value=Broker()):
            with self.assertRaisesRegex(ValueError, "initial input failure") as caught:
                NumericInputRegistration(args, [Path("/synthetic/input")])
        self.assertEqual(actions, ["done"])
        self.assertTrue(any("broker admission close unconfirmed" in note
                            for note in getattr(caught.exception, "__notes__", [])))

    def test_broker_rejects_child_catalog_not_authorized_by_supervisor(self):
        args = SimpleNamespace(consumer_registry_root=None, consumer_registry_id=None,
                               consumer_registry_required=False,
                               consumer_input_catalog=[Path("/synthetic/disallowed")])
        env = {"RZ_CONSUMER_REGISTRY_ROOT": "/synthetic/registry", "RZ_CONSUMER_REGISTRY_ID": "id",
               "RZ_CONSUMER_BROKER_FD": "3", "RZ_CONSUMER_BROKER_PARENT_PID": "123",
               "RZ_CONSUMER_INPUT_CATALOGS": json.dumps(["/synthetic/allowed"])}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(StorageError):
                NumericInputRegistration(args, [Path("/synthetic/input")])

    def test_brokered_facade_delegates_without_reacquiring_catalog(self):
        consumer = NumericInputRegistration.__new__(NumericInputRegistration)
        consumer.registry = object()
        calls = []

        class Broker:
            def add(self, inputs):
                calls.append(("add", list(inputs)))

            def read_and_register_with_future(self, reader):
                calls.append(("future", None))
                return reader()[0]

            def finish(self):
                calls.append(("done", None))

        consumer._broker = Broker()
        consumer.add(["/source"])
        self.assertEqual(_registered_read(consumer, lambda: ("value", [], ["/future"])), "value")
        consumer.finish_broker()
        self.assertEqual(calls, [("add", [Path("/source")]), ("future", None), ("done", None)])
        with self.assertRaises(RuntimeError):
            consumer._locked(None)

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
