import json
import os
from pathlib import Path
import select
import socket
import sys
import threading
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rovent_go_admission as observer


@unittest.skipUnless(sys.platform == 'linux', 'SCM_CREDENTIALS requires Linux')
class GoAdmissionTest(unittest.TestCase):
    def run_sender(self, mutate, *, expected_pid=True):
        channel = observer.GoAdmissionChannel()
        sequence = channel.begin_go()
        epoch, lock, uid = 17, 'a' * 64, os.geteuid()
        pid = os.fork()
        if pid == 0:
            try:
                channel.receiver.close()
                event = {'schema': 'rz-rovent-go-admission/1', 'process_id': os.getpid(),
                         'process_epoch': epoch, 'input_lock_sha256': lock,
                         'go_sequence': sequence, 'game_generation': 1, 'root_generation': 1,
                         'classification': 'driver_rejected_before_ticket',
                         'prior_search': 'no_owner_search_or_drain_observed',
                         'stage': 'admission', 'code': 'invalid_input',
                         'physical_completion': 'not_inferred'}
                mutate(event)
                raw = json.dumps(event).encode()
                channel._child.send(raw)
                channel._child.send(raw)
            finally:
                os._exit(0)
        channel.after_spawn()
        os.waitpid(pid, 0)
        identity = {'child_pid': pid if expected_pid else pid + 1,
                    'child_uid': uid, 'epoch': epoch, 'lock_sha256': lock}
        try:
            return channel, channel.receive(**identity), identity
        except BaseException:
            channel.close_after_child_completion()
            raise

    def test_valid_packet_is_one_shot(self):
        channel, event, identity = self.run_sender(lambda _: None)
        try:
            self.assertIsNotNone(event)
            self.assertIsNone(channel.receive(**identity))
            with self.assertRaises(RuntimeError):
                channel.begin_go()
        finally:
            channel.close_after_child_completion()

    def test_active_search_or_wrong_kernel_pid_cannot_authorize_stop(self):
        for mutation, correct_pid in ((lambda e: e.update(prior_search='search_drain_or_failure_pending'), True),
                                      (lambda _: None, False),
                                      (lambda e: e.update(go_sequence=e['go_sequence'] + 1), True)):
            channel, event, _ = self.run_sender(mutation, expected_pid=correct_pid)
            try:
                self.assertIsNone(event)
            finally:
                channel.close_after_child_completion()

    def test_original_stdin_sealed_until_rejected_go_drains(self):
        class FakeChannel:
            def __init__(self):
                self.observation_error = None
                self.event = {'prior_search': 'no_owner_search_or_drain_observed'}
            def begin_go(self):
                pass
            def completed_go(self):
                pass
            def revoke(self):
                pass
            def receive(self, **_):
                return self.event

        process = SimpleNamespace(pid=123)
        writes = []
        engine_io = SimpleNamespace(process=process, write=writes.append)
        supervisor = observer.OriginalOwnerGoSupervisor(
            process, engine_io, FakeChannel(), epoch=17, lock_sha256='a' * 64)
        # Socket credentials are exercised separately with a real fork above;
        # this test isolates command ordering and one-shot shutdown behavior.
        supervisor._same_live_owner = lambda: True
        supervisor.write(' \tgo\tnodes 16')
        with self.assertRaises(RuntimeError):
            supervisor.write('go nodes 32')
        self.assertIsNotNone(supervisor.poll_rejection())
        supervisor.stop_and_quit_after_rejection()
        self.assertEqual(writes, [' \tgo\tnodes 16', 'stop', 'quit'])
        with self.assertRaises(RuntimeError):
            supervisor.write('go nodes 32')

    def test_diagnostic_failure_does_not_invalidate_normal_bestmove(self):
        class BrokenChannel:
            observation_error = None
            last_trusted_rejection = None
            revoked = False
            def begin_go(self):
                if self.revoked:
                    raise RuntimeError('disabled channel cannot track another go')
            def completed_go(self):
                raise AssertionError('damaged observer must not affect product')
            def receive(self, **_):
                self.observation_error = {'type': 'OSError', 'errno': 5}
                self.revoke()
                return None
            def revoke(self):
                self.revoked = True
        process = SimpleNamespace(pid=123)
        writes = []
        supervisor = observer.OriginalOwnerGoSupervisor(
            process, SimpleNamespace(process=process, write=writes.append), BrokenChannel(),
            epoch=17, lock_sha256='a' * 64)
        supervisor._same_live_owner = lambda: True
        supervisor.write('go nodes 16')
        self.assertIsNone(supervisor.poll_rejection())
        supervisor.bestmove()
        supervisor.write('go nodes 32')
        supervisor.bestmove()
        supervisor.write('quit')
        self.assertEqual(writes, ['go nodes 16', 'go nodes 32', 'quit'])

    def test_stop_preserves_pending_go_until_bestmove(self):
        class Channel:
            last_trusted_rejection = None
            def begin_go(self):
                pass
            def completed_go(self):
                pass
            def revoke(self):
                pass
        process = SimpleNamespace(pid=123)
        writes = []
        supervisor = observer.OriginalOwnerGoSupervisor(
            process, SimpleNamespace(process=process, write=writes.append), Channel(),
            epoch=17, lock_sha256='a' * 64)
        supervisor.write('go nodes 16')
        supervisor.write('isready')
        supervisor.write('ponderhit')
        supervisor.write(' \tstop\t')
        with self.assertRaises(RuntimeError):
            supervisor.write('go nodes 32')
        supervisor.bestmove()
        supervisor.write('go nodes 32')
        self.assertEqual(writes, ['go nodes 16', 'isready', 'ponderhit', ' \tstop\t', 'go nodes 32'])

    def test_quit_during_go_revokes_observer_and_preserves_drain(self):
        class Channel:
            last_trusted_rejection = None
            revoked = False
            def begin_go(self):
                pass
            def completed_go(self):
                raise AssertionError('quit revoked observer tracking')
            def revoke(self):
                self.revoked = True
        process = SimpleNamespace(pid=123)
        writes = []
        channel = Channel()
        supervisor = observer.OriginalOwnerGoSupervisor(
            process, SimpleNamespace(process=process, write=writes.append), channel,
            epoch=17, lock_sha256='a' * 64)
        supervisor.write('go nodes 16')
        supervisor.write('quit')
        self.assertTrue(channel.revoked)
        self.assertIsNone(supervisor.observation_error)
        supervisor.bestmove()
        with self.assertRaises(RuntimeError):
            supervisor.write('go nodes 32')
        self.assertEqual(writes, ['go nodes 16', 'quit'])

    def test_foreign_thread_and_multiline_commands_are_rejected(self):
        class Channel:
            def revoke(self):
                pass
        process = SimpleNamespace(pid=123)
        writes = []
        supervisor = observer.OriginalOwnerGoSupervisor(
            process, SimpleNamespace(process=process, write=writes.append), Channel(),
            epoch=17, lock_sha256='a' * 64)
        failures = []
        def foreign_write():
            try:
                supervisor.write('isready')
            except RuntimeError as exc:
                failures.append(str(exc))
        worker = threading.Thread(target=foreign_write)
        worker.start()
        worker.join()
        self.assertEqual(len(failures), 1)
        with self.assertRaises(ValueError):
            supervisor.write('isready\ngo nodes 16')
        self.assertEqual(writes, [])

    def test_direct_child_peer_and_go_rejection_shutdown(self):
        channel = observer.GoAdmissionChannel()
        owner_to_child_r, owner_to_child_w = os.pipe()
        ready_r, ready_w = os.pipe()
        pid = os.fork()
        if pid == 0:
            try:
                os.close(owner_to_child_w)
                os.close(ready_r)
                channel.receiver.close()
                listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                address = '\0go-observer-test-' + str(os.getpid())
                listener.bind(address)
                listener.listen(1)
                os.write(ready_w, address.encode())
                peer, _ = listener.accept()
                data = os.read(owner_to_child_r, 128)
                if data != b'go nodes 16\n':
                    os._exit(2)
                event = {'schema': 'rz-rovent-go-admission/1', 'process_id': os.getpid(),
                         'process_epoch': 17, 'input_lock_sha256': 'a' * 64,
                         'go_sequence': 1, 'game_generation': 1, 'root_generation': 1,
                         'classification': 'driver_rejected_before_ticket',
                         'prior_search': 'no_owner_search_or_drain_observed',
                         'stage': 'admission', 'code': 'invalid_input',
                         'physical_completion': 'not_inferred'}
                channel._child.send(json.dumps(event).encode())
                commands = bytearray()
                while b'quit\n' not in commands:
                    block = os.read(owner_to_child_r, 128)
                    if not block:
                        os._exit(3)
                    commands.extend(block)
                os._exit(0 if bytes(commands) == b'stop\nquit\n' else 4)
            except BaseException:
                os._exit(5)
        os.close(owner_to_child_r)
        os.close(ready_w)
        channel.after_spawn()
        peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.assertTrue(select.select([ready_r], [], [], 3)[0], 'child listener did not start')
            peer.connect(os.read(ready_r, 128).decode())
            process = SimpleNamespace(pid=pid)
            admission = SimpleNamespace(process=process, confirmed=True, epoch=17,
                                        lock_sha='a' * 64, peer=peer)
            engine_io = SimpleNamespace(process=process, parent_admission=admission,
                                        write=lambda cmd: os.write(owner_to_child_w, (cmd + '\n').encode()))
            supervisor = observer.OriginalOwnerGoSupervisor(
                process, engine_io, channel, epoch=17, lock_sha256='a' * 64)
            supervisor.write('go nodes 16')
            self.assertTrue(select.select([channel.receiver], [], [], 3)[0], 'rejection not delivered')
            self.assertIsNotNone(supervisor.poll_rejection())
            supervisor.stop_and_quit_after_rejection()
            with self.assertRaises(RuntimeError):
                supervisor.stop_and_quit_after_rejection()
            _, status = os.waitpid(pid, 0)
            pid = None
            self.assertEqual(os.waitstatus_to_exitcode(status), 0)
        finally:
            if pid is not None:
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass
                os.waitpid(pid, 0)
            peer.close()
            os.close(ready_r)
            os.close(owner_to_child_w)
            channel.close_after_child_completion()


if __name__ == '__main__':
    unittest.main()
