"""Child-side bounded transaction client; no process or GPU proof is inferred."""
from __future__ import annotations

import os
from pathlib import Path
import select
import time

from storage_consumer_broker_transport import BrokerTransportError, receive_packet, send_packet


class ConsumerBrokerClient:
    """Require matching supervisor ACK before any discovered input is used.

    The caller must validate the actual original supervisor on every received
    packet. EOF, timeout or uncertain delivery closes this client permanently;
    only the supervisor owns and can transition the resulting lease records.
    """

    def __init__(self, endpoint, assert_supervisor, *, maximum_seconds=25.0, clock=time.monotonic):
        if not callable(assert_supervisor) or not callable(clock):
            raise ValueError("original supervisor validator and clock required")
        if type(maximum_seconds) not in (int, float) or not 0 < maximum_seconds <= 25:
            raise ValueError("bounded request duration required")
        self._endpoint = endpoint
        self._assert_supervisor = assert_supervisor
        self._maximum_seconds = float(maximum_seconds)
        self._clock = clock
        self._sequence = 0
        self._closed = False

    def close(self):
        if not self._closed:
            self._closed = True
            self._endpoint.close()

    def _exchange(self, message, phase, sequence, deadline):
        if self._closed:
            raise BrokerTransportError("consumer broker channel is closed")
        try:
            if self._clock() >= deadline:
                raise BrokerTransportError("consumer broker deadline expired")
            send_packet(self._endpoint, message)
            while True:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise BrokerTransportError("consumer broker response timed out")
                ready, _, _ = select.select([self._endpoint], [], [], remaining)
                if not ready:
                    raise BrokerTransportError("consumer broker response timed out")
                response = receive_packet(self._endpoint, self._assert_supervisor)
                if response is None:
                    continue
                if self._clock() >= deadline:
                    raise BrokerTransportError("consumer broker response arrived after deadline")
                if (set(response) != {"kind", "phase", "request_id"}
                        or type(response["kind"]) is not str or response["kind"] != "ack"
                        or type(response["phase"]) is not str or response["phase"] != phase
                        or type(response["request_id"]) is not int or response["request_id"] != sequence):
                    raise BrokerTransportError("unexpected consumer broker response")
                return
        except BaseException:
            self.close()
            raise

    def read_and_register(self, reader):
        def with_future():
            value, inputs = reader()
            return value, inputs, ()
        return self.read_and_register_with_future(with_future)

    def read_and_register_with_future(self, reader):
        if not callable(reader):
            raise ValueError("descriptor reader required")
        self._sequence += 1
        if self._sequence > 128:
            self.close()
            raise BrokerTransportError("consumer broker request limit reached")
        sequence = self._sequence
        deadline = self._clock() + self._maximum_seconds
        self._exchange({"kind": "begin", "request_id": sequence, "deadline": deadline},
                       "begin", sequence, deadline)
        try:
            value, paths, future_paths = reader()
            paths = list(dict.fromkeys(os.fspath(Path(path)) for path in paths))
            future_paths = [path for path in dict.fromkeys(os.fspath(Path(path)) for path in future_paths)
                            if path not in paths]
            if (len(paths) + len(future_paths) > 4096
                    or any(not Path(path).is_absolute() for path in paths + future_paths)):
                raise BrokerTransportError("bounded absolute consumer paths required")
        except BaseException as original:
            # If abort cannot be acknowledged, parent timeout/drain owns the
            # pending transaction; do not replace the original reader error.
            try:
                abort_deadline = min(deadline, self._clock() + 1.0)
                self._exchange({"kind": "abort", "request_id": sequence},
                               "abort", sequence, abort_deadline)
            except BaseException as error:
                if hasattr(original, "add_note"):
                    original.add_note("broker abort unconfirmed: " + type(error).__name__)
            raise
        self._exchange({"kind": "commit", "request_id": sequence, "inputs": paths,
                        "future_inputs": future_paths},
                       "commit", sequence, deadline)
        return value

    def add(self, paths):
        self.read_and_register(lambda: (None, paths))
