"""Child-side bounded transaction client; no process or GPU proof is inferred."""
from __future__ import annotations

import os
from pathlib import Path
import select
import socket
import time

from storage_consumer_broker_transport import BrokerTransportError, receive_packet, send_packet


def inherited_supervisor_client(*, source, registry):
    """Admit a Linux inherited socket only with a verified direct owner.

    This is not an adversarial same-UID security boundary. The connected
    supervisor must itself verify and hold its original manager/catalog and
    registry guards. No broker environment means the legacy path remains.
    Before parent validation an environment-supplied number is not treated as
    an owned descriptor and is not closed; process exit remains its owner.
    """
    raw_fd = os.environ.pop("RZ_CONSUMER_BROKER_FD", None)
    raw_parent = os.environ.pop("RZ_CONSUMER_BROKER_PARENT_PID", None)
    if raw_fd is None and raw_parent is None:
        return None
    if (raw_fd is None or raw_parent is None or not raw_fd.isascii() or not raw_fd.isdecimal()
            or not raw_parent.isascii() or not raw_parent.isdecimal()):
        raise BrokerTransportError("complete inherited broker identity required")
    fd, parent_pid = int(raw_fd), int(raw_parent)
    if fd < 3 or parent_pid <= 1 or parent_pid != os.getppid():
        raise BrokerTransportError("original direct broker supervisor required")
    root = os.environ.get("RZ_MANAGED_BUILD_ROOT")
    token = os.environ.get("RZ_MANAGED_BUILD_TOKEN")
    if not root or not token or registry is None:
        raise BrokerTransportError("original managed owner and registry required")
    endpoint = None
    try:
        endpoint = socket.socket(fileno=fd)
        if (endpoint.family != socket.AF_UNIX
                or endpoint.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_SEQPACKET
                or endpoint.getsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED) != 1):
            raise BrokerTransportError("original credentialled broker socket required")
        endpoint.set_inheritable(False)
        endpoint.setblocking(False)
        from storage_consumer_broker_identity import OriginalSupervisorIdentity
        identity = OriginalSupervisorIdentity(root=root, source=source, token=token, parent_pid=parent_pid,
                                              registry_root=registry.root, registry_id=registry.registry_id)
        return ConsumerBrokerClient(endpoint, identity.check_peer)
    except BaseException:
        if endpoint is not None:
            endpoint.close()
        raise


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

    def read_and_register(self, reader, *, descriptor):
        def with_future():
            value, inputs = reader()
            return value, inputs, ()
        return self.read_and_register_with_future(with_future, descriptor=descriptor)

    def read_and_register_with_future(self, reader, *, descriptor):
        if descriptor is None or not Path(descriptor).is_absolute():
            raise BrokerTransportError("protected absolute descriptor path required")
        return self._transaction(reader, descriptor=os.fspath(Path(descriptor)))

    def _transaction(self, reader, *, descriptor):
        if not callable(reader):
            raise ValueError("descriptor reader required")
        self._sequence += 1
        # Reserve one sequence for an orderly DONE handshake.
        if self._sequence > 127:
            self.close()
            raise BrokerTransportError("consumer broker request limit reached")
        sequence = self._sequence
        deadline = self._clock() + self._maximum_seconds
        self._exchange({"kind": "begin", "request_id": sequence, "deadline": deadline,
                        "descriptor": descriptor},
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
        self._transaction(lambda: (None, paths, ()), descriptor=None)

    def add_future(self, paths):
        self._transaction(lambda: (None, (), paths), descriptor=None)

    def finish(self):
        """Close request admission; no lease or physical completion is claimed."""
        if self._closed:
            return
        self._sequence += 1
        deadline = self._clock() + min(self._maximum_seconds, 2.0)
        try:
            self._exchange({"kind": "done", "request_id": self._sequence},
                           "done", self._sequence, deadline)
        finally:
            self.close()
