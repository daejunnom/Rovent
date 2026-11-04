import errno
import hashlib
import json
import os
from pathlib import Path
import selectors
import socket
import struct
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lc0_parent_transport as transport
from lc0_parent_preflight import ACK_SCHEMA, ACK_SCOPE, FRAME_LIMIT, NOTICE_SCHEMA


class FakeSocket:
    def __init__(self, pid, *, chunk=4096, peer_uid=None):
        self.pid = pid
        self.peer_uid = os.geteuid() if peer_uid is None else peer_uid
        self.chunk = chunk
        self.sent = bytearray()
        self.received = []
        self.closed = False
        self.send_error = None
        self.connect_statuses = []

    def setblocking(self, _value):
        pass

    def connect_ex(self, _route):
        return self.connect_statuses.pop(0) if self.connect_statuses else 0

    def fileno(self):
        return -1 if self.closed else 73

    def getsockopt(self, _level, option, *_args):
        if option == socket.SO_ERROR:
            return 0
        return struct.pack("3i", self.pid, self.peer_uid, 0)

    def send(self, data):
        if self.send_error:
            raise self.send_error()
        count = min(self.chunk, len(data))
        self.sent.extend(data[:count])
        return count

    def recv(self, _length):
        if not self.received:
            raise BlockingIOError(errno.EAGAIN, "not yet readable")
        return self.received.pop(0)

    def close(self):
        self.closed = True


@unittest.skipUnless(sys.platform == "linux", "requires Linux Unix socket peer credentials")
class ParentTransportTests(unittest.TestCase):
    def setUp(self):
        self.pid = os.getpid()
        self.epoch = 17
        self.ordinal = 0
        self.digest = hashlib.sha256(b"transport fixture input").hexdigest()
        self.notice = {"schema": NOTICE_SCHEMA, "process_id": self.pid,
                       "process_epoch": self.epoch, "lock_sha256": self.digest,
                       "cuda_ordinal": self.ordinal, "native_started": False,
                       "control_path": f"/proc/{self.pid}/fd/10/lc0-system-control.sock"}

    def handshake(self, fake):
        with mock.patch.object(transport.socket, "socket", return_value=fake):
            return transport.NonblockingParentHandshake(
                self.notice, child_pid=self.pid, epoch=self.epoch,
                lock_sha256=self.digest, ordinal=self.ordinal,
                deadline=time.monotonic() + 2)

    def ack(self):
        return {"schema": ACK_SCHEMA, "child_pid": self.pid,
                "process_epoch": self.epoch, "lock_sha256": self.digest,
                "cuda_ordinal": self.ordinal, "scope": ACK_SCOPE}

    def write_request(self, handshake):
        while handshake.state in ("connecting", "writing"):
            handshake.advance(selectors.EVENT_WRITE)

    def release_quarantined(self, expected):
        custody = transport._DROPPED_UNCERTAIN_ADMISSIONS.pop()
        self.assertIs(custody._socket, expected)
        expected.close()
        custody._socket = None
        return custody

    def test_partial_request_and_ack_transfer_original_socket_once(self):
        fake = FakeSocket(self.pid, chunk=7)
        handshake = self.handshake(fake)
        self.assertEqual(handshake.binding, (self.pid, self.epoch, self.digest, self.ordinal))
        self.write_request(handshake)
        self.assertEqual(handshake.state, "reading")
        self.assertEqual(handshake.bytes_sent, len(fake.sent))
        payload = json.dumps(self.ack()).encode() + b"\n"
        fake.received = [payload[:5], payload[5:]]
        handshake.advance(selectors.EVENT_READ)
        self.assertEqual(handshake.state, "reading")
        handshake.advance(selectors.EVENT_READ)
        self.assertEqual(handshake.state, "authenticated")
        self.assertEqual(handshake.authenticated_wire, (bytes(fake.sent), payload))
        self.assertIs(handshake.take_authenticated_socket(), fake)
        self.assertEqual(handshake.state, "handed_off")
        with self.assertRaises(RuntimeError):
            handshake.take_authenticated_socket()
        fake.close()

    def test_maximum_ack_body_with_newline_is_accepted(self):
        fake = FakeSocket(self.pid)
        handshake = self.handshake(fake)
        self.write_request(handshake)
        body = json.dumps(self.ack()).encode()
        body += b" " * (FRAME_LIMIT - 1 - len(body))
        fake.received = [body + b"\n"]
        handshake.advance(selectors.EVENT_READ)
        self.assertEqual(handshake.state, "authenticated")
        handshake.take_authenticated_socket().close()

    def test_ack_processed_after_deadline_keeps_original_socket(self):
        fake = FakeSocket(self.pid)
        handshake = self.handshake(fake)
        self.write_request(handshake)
        fake.received = [json.dumps(self.ack()).encode() + b"\n"]
        deadline = handshake._deadline
        with mock.patch.object(transport.time, "monotonic", side_effect=[deadline - 1, deadline + 1]):
            handshake.advance(selectors.EVENT_READ)
        self.assertEqual(handshake.error, "deadline_expired")
        self.assertEqual(handshake.state, "failed_after_send")
        self.assertFalse(fake.closed)
        del handshake
        custody = self.release_quarantined(fake)
        self.assertTrue(custody._response.endswith(b"\n"))

    def test_send_crossing_deadline_retains_sent_wire(self):
        fake = FakeSocket(self.pid, chunk=1)
        handshake = self.handshake(fake)
        deadline = handshake._deadline
        with mock.patch.object(transport.time, "monotonic", side_effect=[deadline - 1, deadline - 1, deadline + 1]):
            handshake.advance(selectors.EVENT_WRITE)
        self.assertEqual(handshake.state, "failed_after_send")
        self.assertEqual(handshake.error, "deadline_expired")
        self.assertEqual(handshake.bytes_sent, 1)
        del handshake
        custody = self.release_quarantined(fake)
        self.assertTrue(custody._request)

    def test_peer_mismatch_is_rejected_before_sending(self):
        for fake in (FakeSocket(self.pid + 1), FakeSocket(self.pid, peer_uid=os.geteuid() + 1)):
            with self.subTest(peer=(fake.pid, fake.peer_uid)):
                handshake = self.handshake(fake)
                handshake.advance(selectors.EVENT_WRITE)
                self.assertEqual(handshake.state, "failed_before_send")
                self.assertEqual(handshake.error, "socket_peer_mismatch")
                self.assertEqual(handshake.bytes_sent, 0)
                self.assertTrue(fake.closed)

    def test_cancel_before_and_after_partial_send_differ(self):
        before = FakeSocket(self.pid)
        handshake = self.handshake(before)
        self.assertEqual(handshake.cancel(), "cancelled_before_send")
        self.assertTrue(before.closed)

        after = FakeSocket(self.pid, chunk=1)
        handshake = self.handshake(after)
        handshake.advance(selectors.EVENT_WRITE)
        self.assertEqual(handshake.bytes_sent, 1)
        self.assertEqual(handshake.cancel(), "cancelled_after_send")
        self.assertTrue(handshake.admission_uncertain)
        self.assertFalse(after.closed)
        del handshake
        self.release_quarantined(after)

    def test_interrupted_send_retains_uncertain_socket(self):
        fake = FakeSocket(self.pid)
        fake.send_error = InterruptedError
        handshake = self.handshake(fake)
        handshake.advance(selectors.EVENT_WRITE)
        self.assertEqual(handshake.state, "failed_after_send")
        self.assertEqual(handshake.bytes_sent, 0)
        self.assertTrue(handshake.admission_uncertain)
        self.assertIsInstance(handshake.original_error, InterruptedError)
        self.assertFalse(fake.closed)
        del handshake
        self.release_quarantined(fake)

    def test_deadline_before_or_after_send_preserves_the_correct_scope(self):
        before = FakeSocket(self.pid)
        handshake = self.handshake(before)
        handshake._deadline = time.monotonic() - 1
        handshake.advance(selectors.EVENT_WRITE)
        self.assertEqual(handshake.state, "failed_before_send")
        self.assertEqual(handshake.error, "deadline_expired")
        self.assertTrue(before.closed)

        after = FakeSocket(self.pid, chunk=1)
        handshake = self.handshake(after)
        handshake.advance(selectors.EVENT_WRITE)
        handshake._deadline = time.monotonic() - 1
        handshake.advance(selectors.EVENT_WRITE)
        self.assertEqual(handshake.state, "failed_after_send")
        self.assertFalse(after.closed)
        del handshake
        self.release_quarantined(after)

    def test_full_unix_backlog_retries_before_admission_bytes(self):
        fake = FakeSocket(self.pid)
        fake.connect_statuses = [errno.EAGAIN, 0]
        handshake = self.handshake(fake)
        self.assertEqual(handshake.state, "retry_connect")
        self.assertEqual(handshake.interest(), 0)
        self.assertEqual(handshake.bytes_sent, 0)
        handshake._retry_at = time.monotonic() - 1
        handshake.advance(0)
        self.assertEqual(handshake.state, "connecting")
        handshake.advance(selectors.EVENT_WRITE)
        self.assertEqual(handshake.state, "reading")
        self.assertGreater(handshake.bytes_sent, 0)
        handshake.cancel()
        del handshake
        self.release_quarantined(fake)

    def test_ack_rejects_wrong_identity_or_boolean_integer(self):
        for changed in ({"child_pid": self.pid + 1}, {"cuda_ordinal": False}):
            with self.subTest(changed=changed):
                fake = FakeSocket(self.pid)
                handshake = self.handshake(fake)
                self.write_request(handshake)
                ack = self.ack()
                ack.update(changed)
                fake.received = [json.dumps(ack).encode() + b"\n"]
                handshake.advance(selectors.EVENT_READ)
                self.assertEqual(handshake.state, "failed_after_send")
                self.assertEqual(handshake.error, "ack_identity_mismatch")
                del handshake
                self.release_quarantined(fake)

    def test_bad_ack_trailing_or_duplicate_does_not_authenticate(self):
        ack = json.dumps(self.ack()).encode()
        duplicate = ack[:-1] + b',"scope":"duplicate"}'
        for reply in (ack + b"\nextra", duplicate + b"\n", b"{" + b"x" * transport.FRAME_LIMIT):
            with self.subTest(length=len(reply)):
                fake = FakeSocket(self.pid)
                handshake = self.handshake(fake)
                self.write_request(handshake)
                fake.received = [reply]
                handshake.advance(selectors.EVENT_READ)
                self.assertEqual(handshake.state, "failed_after_send")
                self.assertFalse(fake.closed)
                del handshake
                self.release_quarantined(fake)

    @unittest.skipUnless(sys.platform == "linux", "requires Linux peer credentials")
    def test_real_unix_peer_credentials_and_ack(self):
        with tempfile.TemporaryDirectory() as root:
            directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                route = f"/proc/{self.pid}/fd/{directory}/lc0-system-control.sock"
                listener.bind(route)
                listener.listen(1)
                listener.settimeout(2)
                self.notice["control_path"] = route
                handshake = transport.NonblockingParentHandshake(
                    self.notice, child_pid=self.pid, epoch=self.epoch,
                    lock_sha256=self.digest, ordinal=self.ordinal,
                    deadline=time.monotonic() + 2)
                connection, _ = listener.accept()
                try:
                    self.write_request(handshake)
                    request = connection.recv(4096)
                    self.assertEqual(json.loads(request)["child_pid"], self.pid)
                    connection.sendall(json.dumps(self.ack()).encode() + b"\n")
                    handshake.advance(selectors.EVENT_READ)
                    self.assertEqual(handshake.state, "authenticated")
                    handshake.take_authenticated_socket().close()
                finally:
                    connection.close()
            finally:
                listener.close()
                os.close(directory)


if __name__ == "__main__":
    unittest.main()
