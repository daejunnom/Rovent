"""Nonblocking protocol adapter for the original supervisor's broker state.

This adapter is not wired to run_managed. Its caller must supply the original
catalog/registry transaction, process-lifetime identity validators, original
owned process-group drain, budget scheduling and lease retention.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

from storage_consumer_broker_state import ConsumerBrokerState
from storage_consumer_broker_transport import BrokerTransportError, receive_packet, send_packet


class ConsumerBrokerSupervisor:
    """Service at most a bounded number of verified child packets per tick.

    On timeout or a transport/protocol failure, admission is revoked but the
    active transaction is deliberately retained. The caller must stop and
    drain the original child group before releasing that transaction. EOF is
    not process exit, and process exit is not GPU physical completion.
    """

    def __init__(self, endpoint, transaction_factory, assert_supervisor, assert_child,
                 assert_peer, *, clock=time.monotonic):
        if not callable(assert_peer) or not callable(clock):
            raise ValueError("original peer validator and clock required")
        self._endpoint = endpoint
        self._assert_peer = assert_peer
        self._clock = clock
        self._state = ConsumerBrokerState(transaction_factory, assert_supervisor,
                                          assert_child, clock=clock, hold_on_failure=True)
        self._deadline = None
        self._revoked = False

    @property
    def transaction_active(self):
        return self._state.transaction_active

    @property
    def publication_unknown(self):
        return self._state.publication_unknown

    @property
    def unknown_owner_id(self):
        return self._state.unknown_owner_id

    def revoke(self):
        """Close admission only; never infer child or GPU completion."""
        if not self._revoked:
            self._revoked = True
            self._endpoint.close()

    @staticmethod
    def _paths(value):
        if (type(value) is not list or len(value) > 4096
                or any(type(path) is not str or not Path(path).is_absolute() for path in value)):
            raise BrokerTransportError("bounded absolute input paths required")
        return value

    def service(self, *, maximum_packets=8):
        if self._revoked:
            return
        if type(maximum_packets) is not int or not 1 <= maximum_packets <= 8:
            raise ValueError("bounded packet count required")
        try:
            if self._deadline is not None and self._clock() >= self._deadline:
                raise BrokerTransportError("child transaction deadline elapsed; drain required")
            for _ in range(maximum_packets):
                message = receive_packet(self._endpoint, self._assert_peer)
                if message is None:
                    return
                kind = message.get("kind")
                sequence = message.get("request_id")
                if type(kind) is not str or type(sequence) is not int:
                    raise BrokerTransportError("exact broker message types required")
                if kind == "begin" and set(message) == {"kind", "request_id", "deadline"}:
                    deadline = message["deadline"]
                    if type(deadline) not in (int, float) or not math.isfinite(deadline):
                        raise BrokerTransportError("finite deadline required")
                    self._state.begin(sequence, deadline=deadline)
                    self._deadline = deadline
                elif kind == "commit" and set(message) == {"kind", "request_id", "inputs", "future_inputs"}:
                    inputs = self._paths(message["inputs"])
                    future = self._paths(message["future_inputs"])
                    self._state.commit(sequence, inputs, future_inputs=future)
                    self._deadline = None
                elif kind == "abort" and set(message) == {"kind", "request_id"}:
                    self._state.abort(sequence)
                    self._deadline = None
                else:
                    raise BrokerTransportError("unexpected broker message")
                send_packet(self._endpoint, {"kind": "ack", "phase": kind, "request_id": sequence})
        except BaseException:
            self.revoke()
            raise

    def finish(self, *, child_never_dispatched, tree_gone):
        """Finalize only after original group drain, or proven no dispatch."""
        self.revoke()
        if self.transaction_active and child_never_dispatched is not True and tree_gone is not True:
            raise BrokerTransportError("original child drain unknown; retain transaction owner")
        if self._state.publication_unknown:
            self._state.release_after_drain(tree_gone=tree_gone)
        else:
            self._state.finish(child_never_dispatched=child_never_dispatched, tree_gone=tree_gone)
