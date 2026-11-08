"""Mock-only checks; no catalog, process, or GPU ownership is created."""
import sys
import threading
import tempfile
import unittest
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from storage_consumer_broker_state import (BrokerPublicationUnknown, BrokerRegistrationRejected,
                                           BrokerStateError, ConsumerBrokerState,
                                           _exact_registry_row_absent)


class FakeLease:
    def __init__(self):
        self.state = "active"

    def close(self, *, not_started):
        if not_started is not True:
            raise AssertionError("unsupported close")
        self.state = "closed"

    def retain_unknown(self, *, tree_gone):
        if tree_gone is not True:
            raise AssertionError("unsupported observation")
        self.state = "physical_unknown"


class Fixture:
    def __init__(self):
        self.now = 10.0
        self.locked = False
        self.owner_alive = True
        self.child_alive = True
        self.leases = []
        self.rows = {}
        self.fail_register = False
        self.future_inputs = None

    def assert_owner(self):
        if not self.owner_alive:
            raise RuntimeError("original owner unavailable")

    def assert_child(self):
        if not self.child_alive:
            raise RuntimeError("original child unavailable")

    @contextmanager
    def transaction(self):
        if self.locked:
            raise RuntimeError("already locked")
        self.locked = True
        try:
            yield self
        finally:
            self.locked = False

    def register(self, inputs, *, owner_id, future_inputs=()):
        if not self.locked:
            raise AssertionError("registration must hold the guards")
        if self.fail_register:
            raise RuntimeError("registration failed")
        self.future_inputs = list(future_inputs)
        lease = FakeLease()
        self.rows[owner_id] = lease
        self.leases.append(lease)
        return lease

    def prove_absent(self, guard, owner_id):
        if guard is not self or not self.locked:
            raise RuntimeError("original guard unavailable")
        return owner_id not in self.rows

    def state(self):
        with patch("storage_consumer_broker_state._exact_registry_row_absent", self.prove_absent):
            return ConsumerBrokerState(self.transaction, self.assert_owner, self.assert_child,
                                       clock=lambda: self.now)


class ConsumerBrokerStateTests(unittest.TestCase):
    def test_hold_policy_keeps_guard_on_expiry_and_owner_failure(self):
        fixture = Fixture()
        state = ConsumerBrokerState(fixture.transaction, fixture.assert_owner, fixture.assert_child,
                                    clock=lambda: fixture.now, hold_on_failure=True)
        state.begin(1, deadline=11)
        fixture.now = 11
        self.assertTrue(state.expire())
        self.assertTrue(fixture.locked)
        fixture.owner_alive = False
        with self.assertRaises(RuntimeError):
            state.expire()
        self.assertTrue(fixture.locked)
        fixture.owner_alive = True
        state.release_after_drain(tree_gone=True)
        self.assertFalse(fixture.locked)

    def test_boolean_and_float_do_not_match_integer_request(self):
        for invalid_id in (True, 1.0):
            with self.subTest(invalid_id=invalid_id):
                fixture = Fixture()
                state = fixture.state()
                state.begin(1, deadline=20)
                with self.assertRaises(BrokerStateError):
                    state.commit(invalid_id, ["/input"])
                with self.assertRaises(BrokerStateError):
                    state.abort(invalid_id)
                self.assertTrue(fixture.locked)
                state.abort(1)
                self.assertFalse(fixture.locked)

    def test_future_inputs_are_registered_under_same_guard(self):
        fixture = Fixture()
        state = fixture.state()
        state.begin(1, deadline=20)
        state.commit(1, ["/source"], future_inputs=["/future"])
        self.assertEqual(fixture.future_inputs, ["/future"])
        state.finish(child_never_dispatched=False, tree_gone=False)
        self.assertEqual(fixture.leases[0].state, "active")

    @unittest.skipUnless(os.name == "posix", "POSIX registry fixture")
    def test_absence_proof_uses_original_registry_guard(self):
        from storage_consumer_registry import ConsumerRegistry
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "input"
            source.write_bytes(b"fixture")
            registry = ConsumerRegistry.initialize(base / "registry", scope="temporary-test")
            owner_id = uuid.uuid4().hex
            with registry.guard() as guard:
                self.assertTrue(_exact_registry_row_absent(guard, owner_id))
                lease = guard.register([source], owner_id=owner_id)
                self.assertFalse(_exact_registry_row_absent(guard, owner_id))
            lease.close(not_started=True)
            with self.assertRaises(Exception):
                _exact_registry_row_absent(guard, owner_id)

    def test_descriptor_read_is_inside_guard_and_commit_retains_lease(self):
        fixture = Fixture()
        state = fixture.state()
        state.begin(1, deadline=20)
        self.assertTrue(fixture.locked)
        # Simulated descriptor reading and path discovery occur here.
        self.assertEqual(state.commit(1, ["/input"]), 1)
        self.assertFalse(fixture.locked)
        self.assertEqual(state.lease_count, 1)
        state.finish(child_never_dispatched=False, tree_gone=True)
        self.assertEqual(fixture.leases[0].state, "physical_unknown")

    def test_timeout_and_register_exception_release_guards(self):
        fixture = Fixture()
        state = fixture.state()
        state.begin(1, deadline=11)
        fixture.now = 11
        self.assertTrue(state.expire())
        self.assertFalse(fixture.locked)
        state.begin(2, deadline=12)
        fixture.fail_register = True
        with self.assertRaises(BrokerRegistrationRejected):
            state.commit(2, ["/input"])
        self.assertFalse(fixture.locked)
        self.assertEqual(state.lease_count, 0)
        self.assertFalse(state.publication_unknown)
        fixture.fail_register = False
        state.begin(3, deadline=12)
        state.abort(3)

    def test_partially_published_row_remains_unknown_and_blocks_dispatch(self):
        fixture = Fixture()
        state = fixture.state()
        unidentifiable_rows = []

        def partial_publish(inputs, *, owner_id, future_inputs=()):
            row = {"state": "active", "inputs": list(inputs)}
            unidentifiable_rows.append(row)
            fixture.rows[owner_id] = row
            raise OSError("simulated publication followed by failed rollback")

        fixture.register = partial_publish
        state.begin(1, deadline=20)
        with self.assertRaises(BrokerPublicationUnknown) as caught:
            state.commit(1, ["/input"])
        self.assertIsInstance(caught.exception.__cause__, OSError)
        self.assertFalse(fixture.locked)
        self.assertTrue(state.publication_unknown)
        self.assertEqual(unidentifiable_rows[0]["state"], "active")
        self.assertIs(fixture.rows[state.unknown_owner_id], unidentifiable_rows[0])
        with self.assertRaises(BrokerStateError):
            state.begin(2, deadline=20)
        with self.assertRaises(BrokerStateError):
            state.finish(child_never_dispatched=False, tree_gone=True)

    def test_unverifiable_absence_remains_unknown(self):
        fixture = Fixture()
        fixture.fail_register = True

        def unavailable_proof(guard, owner_id):
            raise OSError("registry identity could not be verified")

        with patch("storage_consumer_broker_state._exact_registry_row_absent", unavailable_proof):
            state = ConsumerBrokerState(fixture.transaction, fixture.assert_owner, fixture.assert_child,
                                        clock=lambda: fixture.now)
        state.begin(1, deadline=20)
        with self.assertRaises(BrokerPublicationUnknown):
            state.commit(1, ["/input"])
        self.assertTrue(state.publication_unknown)
        self.assertIsNotNone(state.unknown_owner_id)
        with self.assertRaises(BrokerStateError):
            state.begin(2, deadline=20)

    def test_lost_response_leaves_registered_lease_protected(self):
        fixture = Fixture()
        state = fixture.state()
        state.begin(1, deadline=20)
        state.commit(1, ["/input"])
        # The transport dies after commit but before delivering its response.
        state.finish(child_never_dispatched=False, tree_gone=False)
        self.assertEqual(fixture.leases[0].state, "active")

    def test_late_registration_does_not_ack_or_drop_lease(self):
        fixture = Fixture()
        state = fixture.state()
        register = fixture.register

        def late_register(inputs, *, owner_id, future_inputs=()):
            lease = register(inputs, owner_id=owner_id, future_inputs=future_inputs)
            fixture.now = 20
            return lease

        fixture.register = late_register
        state.begin(1, deadline=20)
        with self.assertRaises(BrokerStateError):
            state.commit(1, ["/input"])
        self.assertFalse(fixture.locked)
        self.assertEqual(state.lease_count, 1)
        state.finish(child_never_dispatched=False, tree_gone=False)
        self.assertEqual(fixture.leases[0].state, "active")

    def test_unstarted_close_and_duplicate_request_rejection(self):
        fixture = Fixture()
        state = fixture.state()
        state.begin(1, deadline=20)
        state.commit(1, ["/input"])
        with self.assertRaises(BrokerStateError):
            state.begin(1, deadline=20)
        state.finish(child_never_dispatched=True, tree_gone=True)
        self.assertEqual(fixture.leases[0].state, "closed")

    def test_original_owner_loss_releases_pending_guards(self):
        fixture = Fixture()
        state = fixture.state()
        state.begin(1, deadline=20)
        fixture.owner_alive = False
        with self.assertRaises(RuntimeError):
            state.commit(1, ["/input"])
        self.assertFalse(fixture.locked)
        self.assertEqual(state.lease_count, 0)

    def test_guard_release_failure_does_not_claim_success(self):
        @contextmanager
        def bad_transaction():
            yield object()
            raise RuntimeError("guard release failed")

        state = ConsumerBrokerState(bad_transaction, lambda: None, lambda: None, clock=lambda: 10)
        state.begin(1, deadline=20)
        with self.assertRaises(RuntimeError):
            state.abort(1)
        self.assertTrue(state.transaction_active)
        self.assertTrue(state.lock_release_unknown)
        with self.assertRaises(BrokerStateError):
            state.begin(2, deadline=20)

    def test_supervisor_finishes_after_child_exit(self):
        fixture = Fixture()
        state = fixture.state()
        state.begin(1, deadline=20)
        state.commit(1, ["/input"])
        fixture.child_alive = False
        state.finish(child_never_dispatched=False, tree_gone=True)
        self.assertEqual(fixture.leases[0].state, "physical_unknown")

    def test_other_thread_cannot_mutate_owner_state(self):
        fixture = Fixture()
        state = fixture.state()
        errors = []

        def worker():
            try:
                state.begin(1, deadline=20)
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], BrokerStateError)
        self.assertFalse(fixture.locked)

    def test_request_count_is_bounded(self):
        fixture = Fixture()
        state = ConsumerBrokerState(fixture.transaction, fixture.assert_owner, fixture.assert_child,
                                    clock=lambda: fixture.now, maximum_requests=1)
        state.begin(1, deadline=20)
        state.abort(1)
        with self.assertRaises(BrokerStateError):
            state.begin(2, deadline=20)


if __name__ == "__main__":
    unittest.main()
