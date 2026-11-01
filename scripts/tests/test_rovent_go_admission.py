import json
import os
from pathlib import Path
import sys
import unittest

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


if __name__ == '__main__':
    unittest.main()
