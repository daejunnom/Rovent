"""CPU-only checks for retaining the original unknown capture owner."""
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if sys.platform == "linux":
    from virtual_quarantine_owner import retain_original_owner


@unittest.skipUnless(sys.platform == "linux", "Linux non-reaping child observation")
class OriginalQuarantineOwnerTests(unittest.TestCase):
    def test_same_original_owner_waits_without_signal_until_release(self):
        child = mock.Mock(pid=1234)
        transport = mock.Mock()
        transport.state = {"status": "unknown", "error": None, "peer_pid_verified": True,
                           "terminal_observation_valid": True}
        rounds = []

        def poll():
            rounds.append(1)
            if len(rounds) == 3:
                transport.state["status"] = "session_released"
        transport.poll.side_effect = poll
        waitid = mock.Mock(return_value=None)
        publish = mock.Mock()
        result = retain_original_owner(child, transport, publish, waitid=waitid, sleeper=lambda _: None)
        self.assertEqual(result["state"], "original_owner_release_observed")
        self.assertEqual(result["owner_pid"], child.pid)
        self.assertEqual(result["gpu_physical_completion"], "unknown")
        self.assertEqual(result["process_tree_completion"], "unknown")
        self.assertEqual(len(rounds), 3)
        self.assertEqual(waitid.call_args_list[0].args[1], child.pid)
        child.terminate.assert_not_called()
        child.kill.assert_not_called()

    def test_child_exit_without_release_remains_unknown(self):
        child = mock.Mock(pid=4321)
        transport = mock.Mock()
        transport.state = {"status": "unresolved", "error": None}
        result = retain_original_owner(child, transport, lambda _: None,
                                       waitid=lambda *_: object(), sleeper=lambda _: None)
        self.assertEqual(result["state"], "original_owner_exited_without_release")
        self.assertEqual(result["gpu_physical_completion"], "unknown")
        self.assertEqual(result["process_tree_completion"], "unknown")

    def test_interrupt_and_failed_report_do_not_drop_owner(self):
        child = mock.Mock(pid=3456)
        transport = mock.Mock()
        transport.state = {"status": "unknown", "error": None, "peer_pid_verified": True,
                           "terminal_observation_valid": True}
        rounds = []

        def sleep(_):
            rounds.append(1)
            transport.state["status"] = "owner_recovered"
            raise KeyboardInterrupt()
        result = retain_original_owner(child, transport, mock.Mock(side_effect=OSError("report failed")),
                                       waitid=lambda *_: None, sleeper=sleep)
        self.assertEqual(result["state"], "original_owner_release_observed")
        self.assertEqual(result["interruptions"], 1)
        self.assertTrue(any(item["stage"] == "publish" for item in result["observation_errors"]))
        child.kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
