"""Supervisor-local transaction state for a future input-registration broker.

This module performs no IPC, catalog discovery, or process observation.  Its
caller must own the catalog guard, validate the direct child and obtain all
remaining catalog guards in the common order before supplying a transaction.
It does not establish physical GPU completion or protect legacy consumers.
"""
from __future__ import annotations

from contextlib import ExitStack
import os
import threading
import time
import uuid


class BrokerStateError(RuntimeError):
    pass


class BrokerPublicationUnknown(BrokerStateError):
    """A registration may have left a row whose outcome is unknown."""


class BrokerRegistrationRejected(BrokerStateError):
    """Registration failed and its exact designated row is absent."""


def _exact_registry_row_absent(guard, owner_id):
    """Prove only one designated row absent under its original registry lock.

    This makes no claim about temporary files, other rows, or physical work.
    """
    from storage_consumer_registry import _RegistryGuard
    if type(guard) is not _RegistryGuard:
        raise BrokerStateError("original registry guard required for absence proof")
    guard.check()
    path = guard.registry.root / ("consumer-" + owner_id + ".json")
    try:
        os.lstat(path)
    except FileNotFoundError:
        guard.check()
        return True
    guard.check()
    return False


class ConsumerBrokerState:
    """Hold a registry transaction across a child's bounded descriptor read.

    The original supervisor owns this object and every returned lease.  The
    transaction factory must return a context manager yielding a registry
    guard with ``register``; its context must cover all catalog guards in the
    established order and the registry guard. Separate callbacks validate the
    original supervisor and live requesting child, not merely reusable PIDs.
    State changes are confined to the constructing event-loop thread. Deadline
    enforcement while waiting for input requires that loop to call ``expire``.
    Expiry cannot stop an already-running child read. An IPC integration must
    independently prevent further access or dispatch until the child has
    observed cancellation; this object alone is not that safety boundary.
    """

    def __init__(self, transaction_factory, assert_supervisor, assert_child, *, clock=time.monotonic,
                 maximum_seconds=30.0, maximum_requests=128, hold_on_failure=False):
        if not all(callable(fn) for fn in (transaction_factory, assert_supervisor, assert_child,
                                            clock)):
            raise ValueError("transaction factory, identity checks and clock are required")
        if type(maximum_seconds) not in (int, float) or not 0 < maximum_seconds <= 30:
            raise ValueError("bounded transaction duration required")
        if type(maximum_requests) is not int or not 1 <= maximum_requests <= 128:
            raise ValueError("bounded request count required")
        if type(hold_on_failure) is not bool:
            raise ValueError("explicit failure retention policy required")
        self._factory = transaction_factory
        self._assert_supervisor = assert_supervisor
        self._assert_child = assert_child
        self._clock = clock
        self._prove_absent = _exact_registry_row_absent
        self._maximum_seconds = float(maximum_seconds)
        self._maximum_requests = maximum_requests
        self._hold_on_failure = hold_on_failure
        self._requests_started = 0
        self._pid = os.getpid()
        self._thread = threading.get_ident()
        self._stack = None
        self._guard = None
        self._request = None
        self._deadline = None
        self._descriptor_pin = None
        self._last_request = -1
        self._leases = []
        self._broken = False
        self._lock_release_unknown = False
        self._publication_unknown = False
        self._unknown_owner_id = None
        self._finalized = False
        self._admission_closed = False

    def _check(self, *, child=False, preserve_on_failure=False):
        if (os.getpid() != self._pid or threading.get_ident() != self._thread
                or self._broken or self._finalized):
            raise BrokerStateError("original broker owner unavailable")
        try:
            self._assert_supervisor()
            if child:
                self._assert_child()
        except BaseException:
            # The original process may still close its own transaction when
            # child identity or the original owner scope has disappeared.
            if not preserve_on_failure:
                self._release_transaction()
            raise

    @property
    def transaction_active(self):
        # An interrupted release is not proof that any underlying lock remains.
        return self._stack is not None

    @property
    def lock_release_unknown(self):
        return self._lock_release_unknown

    @property
    def publication_unknown(self):
        """True means the designated registry row may require preservation."""
        return self._publication_unknown

    @property
    def unknown_owner_id(self):
        """Expose only the row ID whose publication or rollback is unknown."""
        return self._unknown_owner_id

    @property
    def lease_count(self):
        return len(self._leases)

    def _release_transaction(self):
        stack = self._stack
        if stack is not None:
            try:
                stack.close()
            except BaseException:
                # ExitStack may already have popped a failed callback. Neither
                # lock ownership nor successful unlock can be inferred here.
                self._broken = True
                self._lock_release_unknown = True
                raise
        self._stack = self._guard = self._request = self._deadline = self._descriptor_pin = None

    def begin(self, request_id, *, deadline, descriptor=None):
        self._check(child=True, preserve_on_failure=self._hold_on_failure)
        if self._admission_closed:
            raise BrokerStateError("consumer admission already closed")
        if self._stack is not None:
            raise BrokerStateError("transaction already active")
        if type(request_id) is not int or request_id <= self._last_request:
            raise BrokerStateError("request sequence must increase")
        if self._requests_started >= self._maximum_requests:
            raise BrokerStateError("request limit reached")
        now = self._clock()
        if type(deadline) not in (int, float) or not now < deadline <= now + self._maximum_seconds:
            raise BrokerStateError("invalid transaction deadline")
        self._last_request = request_id
        self._requests_started += 1
        stack = ExitStack()
        try:
            guard = stack.enter_context(self._factory())
            self._assert_supervisor()
            self._assert_child()
            descriptor_pin = None
            if descriptor is not None:
                from storage_consumer_broker_coverage import pin_descriptor_coverage
                descriptor_pin = stack.enter_context(pin_descriptor_coverage(guard, self._leases, descriptor))
            if self._clock() >= deadline:
                raise BrokerStateError("transaction expired while acquiring guards")
            self._stack, self._guard = stack, guard
            self._request, self._deadline = request_id, deadline
            self._descriptor_pin = descriptor_pin
        except BaseException:
            try:
                stack.close()
            except BaseException:
                self._broken = True
                self._lock_release_unknown = True
                self._stack = stack
                raise
            raise

    def commit(self, request_id, inputs, *, future_inputs=()):
        self._check(child=True, preserve_on_failure=self._hold_on_failure)
        if self._stack is None or type(request_id) is not int or request_id != self._request:
            raise BrokerStateError("no matching transaction")
        if (not isinstance(inputs, (list, tuple)) or not isinstance(future_inputs, (list, tuple))
                or len(inputs) + len(future_inputs) > 4096):
            if not self._hold_on_failure:
                self._release_transaction()
            raise BrokerStateError("bounded input list required")
        if self._clock() >= self._deadline:
            if not self._hold_on_failure:
                self._release_transaction()
            raise BrokerStateError("transaction expired")
        succeeded = False
        try:
            if self._descriptor_pin is not None:
                self._descriptor_pin.verify()
            # Empty input lists are valid when the descriptor names no assets.
            if inputs or future_inputs:
                owner_id = uuid.uuid4().hex
                try:
                    lease = self._guard.register(inputs, owner_id=owner_id, future_inputs=future_inputs)
                    # Keep custody before unlocking or attempting a response.
                    self._leases.append(lease)
                except BaseException as error:
                    # The row may have been published before an error and its
                    # rollback may have failed. Prove only our chosen ID absent
                    # while the same original registry guard is held.
                    try:
                        absent = self._prove_absent(self._guard, owner_id) is True
                    except BaseException:
                        absent = False
                    if not absent:
                        self._publication_unknown = True
                        self._unknown_owner_id = owner_id
                        self._broken = True
                    if not isinstance(error, Exception):
                        raise
                    if absent:
                        raise BrokerRegistrationRejected(
                            "registration failed; designated consumer row is absent") from error
                    raise BrokerPublicationUnknown(
                        "consumer row publication or rollback unknown; preserve registry") from error
            self._assert_supervisor()
            self._assert_child()
            if self._descriptor_pin is not None:
                self._descriptor_pin.verify()
            if self._clock() >= self._deadline:
                raise BrokerStateError("transaction expired during registration; lease preserved")
            succeeded = True
        finally:
            if succeeded or not self._hold_on_failure:
                self._release_transaction()
        return request_id

    def abort(self, request_id):
        self._check(child=True, preserve_on_failure=self._hold_on_failure)
        if self._stack is None or type(request_id) is not int or request_id != self._request:
            raise BrokerStateError("no matching transaction")
        self._release_transaction()

    def expire(self):
        self._check(preserve_on_failure=self._hold_on_failure)
        if self._stack is None or self._clock() < self._deadline:
            return False
        if not self._hold_on_failure:
            self._release_transaction()
        return True

    def done(self, request_id):
        """Close request admission without deciding any lease's lifetime."""
        self._check(child=True, preserve_on_failure=self._hold_on_failure)
        if (self._admission_closed or self._stack is not None or type(request_id) is not int
                or request_id <= self._last_request or self._requests_started >= self._maximum_requests):
            raise BrokerStateError("done requires a new request and no active transaction")
        self._last_request = request_id
        self._requests_started += 1
        self._admission_closed = True

    def release_after_drain(self, *, tree_gone):
        """Unlock a failed transaction only after the original group is gone.

        This does not repair a broken registry or close/transition any lease.
        """
        if (os.getpid() != self._pid or threading.get_ident() != self._thread
                or self._finalized or self._lock_release_unknown or tree_gone is not True):
            raise BrokerStateError("original supervisor and observed drain required")
        self._assert_supervisor()
        self._release_transaction()

    def finish(self, *, child_never_dispatched, tree_gone):
        """Release guards; close only proven unstarted leases, else preserve.

        ``tree_gone`` must be the original supervisor's own observation; it
        permits only the physical-unknown transition, never a GPU fence claim.
        If it is false, active rows are intentionally left untouched.
        """
        self._check()
        if type(child_never_dispatched) is not bool or type(tree_gone) is not bool:
            raise BrokerStateError("explicit original-owner observations required")
        self._release_transaction()
        errors = []
        for lease in reversed(self._leases):
            try:
                if child_never_dispatched:
                    lease.close(not_started=True)
                elif tree_gone:
                    lease.retain_unknown(tree_gone=True)
            except Exception as error:
                errors.append(error)
        self._finalized = True
        if errors:
            raise BrokerStateError("one or more consumer lease transitions failed; preserve rows") from errors[0]
