"""Linux process ownership checks without model, GPU or runner dependencies."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from managed_process import ProcessNotSpawned, run_group


@unittest.skipUnless(sys.platform == "linux", "Linux process-group FD handoff only")
class ManagedProcessSpawnTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="rovezero-managed-spawn-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.python = str(Path(sys.executable).resolve())

    def test_preflight_error_keeps_original_cause_and_does_not_spawn(self):
        callbacks = []
        missing = str(self.root / "missing-executable")
        with mock.patch("managed_process.subprocess.Popen") as popen:
            with self.assertRaises(ProcessNotSpawned) as raised:
                run_group([missing], self.root, os.environ.copy(), 10, lambda: None,
                          verify_before_spawn=True, on_spawn=lambda: callbacks.append(1))
        self.assertIsInstance(raised.exception.__cause__, FileNotFoundError)
        popen.assert_not_called()
        self.assertEqual(callbacks, [])

    def test_success_reports_spawn_once_and_preserves_fd_and_stdout(self):
        callbacks = []
        with tempfile.TemporaryFile(mode="w+b", dir=self.root) as data, \
             tempfile.TemporaryFile(mode="w+b", dir=self.root) as output:
            data.write(b"owned input")
            data.flush()
            data.seek(0)
            program = "import json,os,sys; print(json.dumps(os.read(int(sys.argv[1]),100).decode()))"
            code, gone, failure = run_group(
                [self.python, "-c", program, str(data.fileno())], self.root,
                os.environ.copy(), 10, lambda: None, pass_fds=(data.fileno(),), stdout=output,
                verify_before_spawn=True, on_spawn=lambda: callbacks.append(1),
            )
            output.seek(0)
            self.assertEqual(json.loads(output.read()), "owned input")
        self.assertEqual((code, gone, failure), (0, True, None))
        self.assertEqual(callbacks, [1])

    def test_callback_exception_drains_and_reaps_the_original_child(self):
        original = subprocess.Popen
        for exception, expected in ((ValueError("callback-failed"), "callback-failed"),
                                    (SystemExit("callback-stop"), "SystemExit: callback-stop")):
            with self.subTest(exception=type(exception).__name__):
                children = []

                def record_child(*args, **kwargs):
                    child = original(*args, **kwargs)
                    children.append(child)
                    return child

                def failed_callback():
                    raise exception

                with mock.patch("managed_process.subprocess.Popen", side_effect=record_child):
                    code, gone, failure = run_group(
                        [self.python, "-c", "import time; time.sleep(30)"], self.root,
                        os.environ.copy(), 10, lambda: None,
                        verify_before_spawn=True, on_spawn=failed_callback,
                    )
                self.assertEqual((code, gone, failure), (125, True, expected))
                self.assertEqual(len(children), 1)
                self.assertIsNotNone(children[0].poll(), "owned child must be reaped")

    def test_default_system_exit_behavior_remains_unchanged(self):
        # No actual process is created: the legacy branch intentionally
        # propagates SystemExit, so a real child could be left behind.
        fake_child = mock.Mock(pid=123456)
        with mock.patch("managed_process.subprocess.Popen", return_value=fake_child), \
             mock.patch("managed_process.os.waitid", return_value=None), \
             mock.patch("managed_process.os.killpg") as killpg:
            with self.assertRaises(SystemExit) as raised:
                run_group([self.python], self.root, os.environ.copy(), 10,
                          lambda: (_ for _ in ()).throw(SystemExit("unchanged")))
        self.assertEqual(str(raised.exception), "unchanged")
        killpg.assert_not_called()

    def test_absolute_deadline_validation_and_pre_spawn_expiry(self):
        for deadline in (True, 0, float("nan"), float("inf"), "1"):
            with self.subTest(deadline=deadline), mock.patch("managed_process.subprocess.Popen") as popen:
                with self.assertRaises(ValueError):
                    run_group([self.python], self.root, os.environ.copy(), 10, lambda: None,
                              absolute_deadline=deadline)
                popen.assert_not_called()
        with mock.patch("managed_process.subprocess.Popen") as popen, \
             mock.patch("managed_process.time.monotonic", return_value=10.0):
            with self.assertRaises(ProcessNotSpawned) as raised:
                run_group([self.python], self.root, os.environ.copy(), 10, lambda: None,
                          absolute_deadline=9.0)
            self.assertIsInstance(raised.exception.__cause__, TimeoutError)
            popen.assert_not_called()

    def test_preflight_cannot_dispatch_after_absolute_deadline(self):
        clock = {"now": 10.0}

        def slow_preflight(*args):
            clock["now"] = 20.0

        with mock.patch("managed_process._check_before_spawn", side_effect=slow_preflight), \
             mock.patch("managed_process.time.monotonic", side_effect=lambda: clock["now"]), \
             mock.patch("managed_process.subprocess.Popen") as popen:
            with self.assertRaises(ProcessNotSpawned):
                run_group([self.python], self.root, os.environ.copy(), 10, lambda: None,
                          verify_before_spawn=True, absolute_deadline=15.0)
            popen.assert_not_called()

    def test_spawn_delay_counts_toward_absolute_deadline(self):
        clock = {"now": 10.0}
        children = []
        observation = {}
        child = mock.Mock(pid=123456)
        child.wait.return_value = -15

        def slow_spawn(*args, **kwargs):
            clock["now"] = 20.0
            children.append(child)
            return child

        with mock.patch("managed_process.subprocess.Popen", side_effect=slow_spawn), \
             mock.patch("managed_process.time.monotonic", side_effect=lambda: clock["now"]), \
             mock.patch("managed_process.os.waitid", return_value=None), \
             mock.patch("managed_process.os.killpg", side_effect=ProcessLookupError), \
             mock.patch("managed_process.os.scandir", return_value=[]):
            code, gone, failure = run_group(
                [self.python, "-c", "import time; time.sleep(30)"], self.root,
                os.environ.copy(), 10, lambda: None, on_spawn=lambda: None,
                absolute_deadline=15.0, deadline_observation=observation,
            )
        self.assertEqual((code, gone, failure), (124, True, None))
        self.assertTrue(observation["deadline_elapsed_while_running_observed"])
        self.assertEqual(observation["exit_observation_timing"], "not_observed")
        self.assertEqual(observation["actual_exit_time"], "unknown")
        self.assertIsInstance(observation["reaped_exit_code"], int)
        self.assertEqual(len(children), 1)
        child.wait.assert_called_once_with()

    def test_late_observed_exit_preserves_real_code_and_marks_timing_unknown(self):
        observation = {}
        clock = {"now": 10.0}
        child = mock.Mock(pid=123456)
        child.wait.return_value = 7

        def spawn_after_exit(*args, **kwargs):
            clock["now"] = 20.0
            return child

        with mock.patch("managed_process.subprocess.Popen", side_effect=spawn_after_exit), \
             mock.patch("managed_process.time.monotonic", side_effect=lambda: clock["now"]), \
             mock.patch("managed_process.os.waitid", side_effect=[None, object()]), \
             mock.patch("managed_process.os.killpg", side_effect=ProcessLookupError), \
             mock.patch("managed_process.os.scandir", return_value=[]):
            code, gone, failure = run_group(
                [self.python, "-c", "import sys; sys.exit(7)"], self.root,
                os.environ.copy(), 10, lambda: None, on_spawn=lambda: None,
                absolute_deadline=15.0, deadline_observation=observation,
            )
        self.assertEqual((code, gone, failure), (7, True, None))
        self.assertEqual(observation["reaped_exit_code"], 7)
        self.assertEqual(observation["exit_observation_timing"], "at_or_after_deadline")
        self.assertEqual(observation["actual_exit_time"], "unknown")
        self.assertFalse(observation["deadline_elapsed_while_running_observed"])

    def test_admission_delay_rechecks_exited_child_before_timeout(self):
        observation = {}
        clock = {"now": 10.0}
        child = mock.Mock(pid=123456)
        child.wait.return_value = 7

        def slow_admission():
            clock["now"] = 20.0

        with mock.patch("managed_process.subprocess.Popen", return_value=child), \
             mock.patch("managed_process.time.monotonic", side_effect=lambda: clock["now"]), \
             mock.patch("managed_process.os.waitid", side_effect=[None, object()]), \
             mock.patch("managed_process.os.killpg", side_effect=ProcessLookupError), \
             mock.patch("managed_process.os.scandir", return_value=[]):
            code, gone, failure = run_group(
                [self.python], self.root, os.environ.copy(), 10, slow_admission,
                on_spawn=lambda: None, absolute_deadline=15.0, deadline_observation=observation,
            )
        self.assertEqual((code, gone, failure), (7, True, None))
        self.assertEqual(observation["reaped_exit_code"], 7)
        self.assertEqual(observation["exit_observation_timing"], "at_or_after_deadline")
        self.assertFalse(observation["deadline_elapsed_while_running_observed"])

    def test_relative_timeout_can_precede_absolute_deadline(self):
        code, gone, failure = run_group(
            [self.python, "-c", "import time; time.sleep(30)"], self.root,
            os.environ.copy(), 0.01, lambda: None, on_spawn=lambda: None,
            absolute_deadline=time.monotonic() + 10,
        )
        self.assertEqual((code, gone, failure), (124, True, None))

    def test_fast_tick_runs_each_loop_without_extra_budget_scans(self):
        clock = {"now": 10.0}
        ticks, scans = [], []
        child = mock.Mock(pid=123456)
        child.wait.return_value = 0

        def advance(seconds):
            clock["now"] += seconds

        with mock.patch("managed_process.subprocess.Popen", return_value=child), \
             mock.patch("managed_process.time.monotonic", side_effect=lambda: clock["now"]), \
             mock.patch("managed_process.time.sleep", side_effect=advance), \
             mock.patch("managed_process.os.waitid", side_effect=[None, None, None, object()]), \
             mock.patch("managed_process.os.killpg", side_effect=ProcessLookupError), \
             mock.patch("managed_process.os.scandir", return_value=[]):
            result = run_group([self.python], self.root, os.environ.copy(), 20,
                               lambda: scans.append(1), absolute_deadline=15.0,
                               on_tick=lambda: ticks.append(1))
        self.assertEqual(result, (0, True, None))
        self.assertEqual(len(ticks), 3)
        self.assertEqual(len(scans), 1)

    def test_tick_failures_drain_original_child(self):
        for exception, expected in ((ValueError("tick-failed"), "tick-failed"),
                                    (SystemExit("tick-stop"), "SystemExit: tick-stop")):
            child = mock.Mock(pid=123456)
            child.wait.return_value = -15

            def fail():
                raise exception

            with self.subTest(exception=type(exception).__name__), \
                 mock.patch("managed_process.subprocess.Popen", return_value=child), \
                 mock.patch("managed_process.os.waitid", return_value=None), \
                 mock.patch("managed_process.os.killpg", side_effect=ProcessLookupError), \
                 mock.patch("managed_process.os.scandir", return_value=[]):
                result = run_group([self.python], self.root, os.environ.copy(), 10,
                                   lambda: None, on_tick=fail)
            self.assertEqual(result, (125, True, expected))
            child.wait.assert_called_once_with()

    def test_slow_tick_preserves_exit_and_unknown_completion_time(self):
        clock, observation = {"now": 10.0}, {}
        child = mock.Mock(pid=123456)
        child.wait.return_value = 7
        scans = []

        def slow_tick():
            clock["now"] = 20.0

        with mock.patch("managed_process.subprocess.Popen", return_value=child), \
             mock.patch("managed_process.time.monotonic", side_effect=lambda: clock["now"]), \
             mock.patch("managed_process.os.waitid", side_effect=[None, object()]), \
             mock.patch("managed_process.os.killpg", side_effect=ProcessLookupError), \
             mock.patch("managed_process.os.scandir", return_value=[]):
            result = run_group([self.python], self.root, os.environ.copy(), 10,
                               lambda: scans.append(1), absolute_deadline=15.0,
                               deadline_observation=observation, on_tick=slow_tick)
        self.assertEqual(result, (7, True, None))
        self.assertEqual(scans, [])
        self.assertEqual(observation["exit_observation_timing"], "at_or_after_deadline")
        self.assertEqual(observation["actual_exit_time"], "unknown")
        self.assertFalse(observation["deadline_elapsed_while_running_observed"])

    def test_invalid_tick_is_rejected_before_spawn(self):
        with mock.patch("managed_process.subprocess.Popen") as popen:
            with self.assertRaises(ValueError):
                run_group([self.python], self.root, os.environ.copy(), 10,
                          lambda: None, on_tick=1)
            popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
