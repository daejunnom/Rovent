"""Small local sockets only; not an original-owner or GPU integration test."""
import array
import os
import socket
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from storage_consumer_broker_transport import (BrokerPeerEOF, BrokerTransportError, MAX_PACKET_BYTES,
                                                open_pair, receive_packet, send_packet)


@unittest.skipUnless(os.name == "posix" and hasattr(socket, "SO_PASSCRED"), "Linux credentials required")
class ConsumerBrokerTransportTests(unittest.TestCase):
    def setUp(self):
        self.left, self.right = open_pair()
        self.addCleanup(self.left.close)
        self.addCleanup(self.right.close)

    def _same_process(self, pid, uid, gid):
        self.assertEqual((pid, uid, gid), (os.getpid(), os.getuid(), os.getgid()))

    def test_packet_has_kernel_sender_identity(self):
        self.assertFalse(self.left.get_inheritable())
        self.assertFalse(self.right.get_inheritable())
        self.assertIsNone(receive_packet(self.right, self._same_process))
        send_packet(self.left, {"kind": "begin", "sequence": 1})
        self.assertEqual(receive_packet(self.right, self._same_process), {"kind": "begin", "sequence": 1})

    def test_wrong_original_peer_is_rejected(self):
        send_packet(self.left, {"kind": "begin"})
        def rejected(pid, uid, gid):
            raise RuntimeError("not the original child")
        with self.assertRaises(BrokerTransportError):
            receive_packet(self.right, rejected)

    def test_malformed_duplicate_and_oversized_packets_are_rejected(self):
        self.left.send(b'{"kind":1,"kind":2}')
        with self.assertRaises(BrokerTransportError):
            receive_packet(self.right, self._same_process)
        self.left.send(b'{"nested":[{"value":1e999}]}')
        with self.assertRaises(BrokerTransportError):
            receive_packet(self.right, self._same_process)
        with self.assertRaises(BrokerTransportError):
            send_packet(self.left, {"value": "x" * MAX_PACKET_BYTES})

    def test_eof_is_not_an_empty_success(self):
        self.left.close()
        with self.assertRaises(BrokerPeerEOF):
            receive_packet(self.right, self._same_process)

    def test_empty_packet_is_malformed_not_clean_eof(self):
        self.left.send(b"")
        with self.assertRaises(BrokerTransportError) as caught:
            receive_packet(self.right, self._same_process)
        self.assertNotIsInstance(caught.exception, BrokerPeerEOF)

    def test_unexpected_descriptor_is_rejected_and_closed(self):
        read_fd, write_fd = os.pipe()
        self.addCleanup(os.close, read_fd)
        os.set_blocking(read_fd, False)
        self.left.sendmsg([b'{}'], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [write_fd]))])
        os.close(write_fd)
        with self.assertRaises(BrokerTransportError):
            receive_packet(self.right, self._same_process)
        self.assertEqual(os.read(read_fd, 1), b"")

    def test_truncated_ancillary_data_closes_received_descriptors(self):
        read_fd, write_fd = os.pipe()
        self.addCleanup(os.close, read_fd)
        os.set_blocking(read_fd, False)
        self.left.sendmsg([b'{}'], [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                                    array.array("i", [write_fd] * 64))])
        os.close(write_fd)
        with self.assertRaises(BrokerTransportError):
            receive_packet(self.right, self._same_process)
        self.assertEqual(os.read(read_fd, 1), b"")


if __name__ == "__main__":
    unittest.main()
