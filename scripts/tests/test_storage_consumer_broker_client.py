"""Local channel fixtures; not proof of real supervisor ownership."""
import os
import select
import socket
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from storage_consumer_broker_client import ConsumerBrokerClient, inherited_supervisor_client
from storage_consumer_broker_transport import BrokerTransportError, open_pair, receive_packet, send_packet


@unittest.skipUnless(os.name == "posix" and hasattr(socket, "SO_PASSCRED"), "Linux credentials required")
class ConsumerBrokerClientTests(unittest.TestCase):
    def setUp(self):
        self.server, self.child = open_pair()
        self.addCleanup(self.server.close)
        self.addCleanup(self.child.close)
        self.client = ConsumerBrokerClient(self.child, self.same_process, maximum_seconds=2)
        self.addCleanup(self.client.close)
        self.errors = []

    def same_process(self, pid, uid, gid):
        if (pid, uid, gid) != (os.getpid(), os.getuid(), os.getgid()):
            raise ValueError("unexpected fixture peer")

    def server_exchange(self, expected, observed):
        try:
            for phase in expected:
                ready, _, _ = select.select([self.server], [], [], 2)
                if not ready:
                    raise TimeoutError("fixture did not receive a request")
                request = receive_packet(self.server, self.same_process)
                if request["kind"] != phase:
                    raise ValueError("unexpected phase")
                observed.append(request)
                send_packet(self.server, {"kind": "ack", "phase": phase,
                                          "request_id": request["request_id"]})
        except BaseException as error:
            self.errors.append(error)

    def launch(self, phases, observed):
        thread = threading.Thread(target=self.server_exchange, args=(phases, observed))
        thread.start()
        return thread

    def test_read_occurs_between_begin_and_commit_ack(self):
        observed = []
        thread = self.launch(("begin", "commit"), observed)

        def reader():
            self.assertEqual([item["kind"] for item in observed], ["begin"])
            return "sealed-value", ["/asset", "/asset"]

        self.assertEqual(self.client.read_and_register(reader, descriptor="/descriptor"), "sealed-value")
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertFalse(self.errors)
        self.assertEqual(observed[1]["inputs"], ["/asset"])
        self.assertEqual(observed[1]["future_inputs"], [])
        self.assertEqual(observed[0]["descriptor"], "/descriptor")

    def test_existing_and_future_paths_share_one_commit(self):
        observed = []
        thread = self.launch(("begin", "commit"), observed)
        result = self.client.read_and_register_with_future(
            lambda: ("value", ["/asset"], ["/asset", "/future"]), descriptor="/descriptor")
        thread.join(timeout=2)
        self.assertEqual(result, "value")
        self.assertFalse(thread.is_alive())
        self.assertFalse(self.errors)
        self.assertEqual(observed[1]["inputs"], ["/asset"])
        self.assertEqual(observed[1]["future_inputs"], ["/future"])

    def test_future_registration_is_not_a_descriptor_read(self):
        observed = []
        thread = self.launch(("begin", "commit"), observed)
        self.client.add_future(["/future"])
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertFalse(self.errors)
        self.assertIsNone(observed[0]["descriptor"])
        self.assertEqual(observed[1]["inputs"], [])
        self.assertEqual(observed[1]["future_inputs"], ["/future"])

    def test_done_closes_admission_without_claiming_completion(self):
        observed = []
        thread = self.launch(("done",), observed)
        self.client.finish()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertFalse(self.errors)
        with self.assertRaises(BrokerTransportError):
            self.client.add(["/asset"])

    def test_reader_failure_sends_abort_and_preserves_exception(self):
        observed = []
        thread = self.launch(("begin", "abort"), observed)

        def reader():
            raise ValueError("bad descriptor")

        with self.assertRaisesRegex(ValueError, "bad descriptor"):
            self.client.read_and_register(reader, descriptor="/descriptor")
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertFalse(self.errors)

    def test_lost_ack_closes_channel_and_rejects_later_requests(self):
        self.server.close()
        with self.assertRaises(BrokerTransportError):
            self.client.add(["/asset"])
        with self.assertRaises(BrokerTransportError):
            self.client.add(["/asset"])

    def test_boolean_and_float_request_ids_are_not_integer_ack(self):
        for invalid_id in (True, 1.0):
            with self.subTest(invalid_id=invalid_id):
                server, child = open_pair()
                client = ConsumerBrokerClient(child, self.same_process, maximum_seconds=2)

                def wrong_ack():
                    ready, _, _ = select.select([server], [], [], 2)
                    if ready:
                        receive_packet(server, self.same_process)
                        send_packet(server, {"kind": "ack", "phase": "begin", "request_id": invalid_id})

                thread = threading.Thread(target=wrong_ack)
                thread.start()
                try:
                    with self.assertRaises(BrokerTransportError):
                        client.add(["/asset"])
                finally:
                    thread.join(timeout=2)
                    client.close()
                    server.close()
                self.assertFalse(thread.is_alive())

    def test_incomplete_inherited_broker_is_rejected_without_fallback(self):
        with patch.dict(os.environ, {"RZ_CONSUMER_BROKER_FD": "3"}, clear=True):
            with self.assertRaises(BrokerTransportError):
                inherited_supervisor_client(source="/synthetic", registry=None)

    def test_wrong_parent_does_not_close_an_unverified_descriptor(self):
        read_fd, write_fd = os.pipe()
        self.addCleanup(os.close, read_fd)
        self.addCleanup(os.close, write_fd)
        env = {"RZ_CONSUMER_BROKER_FD": str(write_fd),
               "RZ_CONSUMER_BROKER_PARENT_PID": str(os.getppid() + 1)}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(BrokerTransportError):
                inherited_supervisor_client(source="/synthetic", registry=None)
        os.fstat(write_fd)  # Unverified descriptor was not stolen or closed.

    def test_inherited_socket_is_cloexec_and_identity_is_constructed(self):
        parent, child = open_pair()
        self.addCleanup(parent.close)
        fd = child.detach()
        env = {"RZ_CONSUMER_BROKER_FD": str(fd),
               "RZ_CONSUMER_BROKER_PARENT_PID": str(os.getppid()),
               "RZ_MANAGED_BUILD_ROOT": "/synthetic/root",
               "RZ_MANAGED_BUILD_TOKEN": "synthetic-token"}
        registry = SimpleNamespace(root=Path("/synthetic/registry"), registry_id="synthetic-id")
        identity = SimpleNamespace(check_peer=lambda *_: None)
        with patch.dict(os.environ, env, clear=True), patch(
                "storage_consumer_broker_identity.OriginalSupervisorIdentity", return_value=identity) as created:
            client = inherited_supervisor_client(source=Path("/synthetic/source"), registry=registry)
            try:
                self.assertFalse(os.get_inheritable(fd))
                self.assertNotIn("RZ_CONSUMER_BROKER_FD", os.environ)
                created.assert_called_once()
            finally:
                client.close()
if __name__ == "__main__":
    unittest.main()
