"""Linux process ownership checks without model, GPU or runner dependencies."""
import json
from contextlib import nullcontext
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
from managed_process import OwnedProcessGroup, ProcessNotSpawned, run_group


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

    def test_opt_in_work_deadline_uses_original_cleanup_deadline_for_drain(self):
        import managed_process
        start = time.monotonic()
        work, cleanup = start + 0.2, start + 2.0
        group = OwnedProcessGroup()
        with mock.patch("managed_process._finish_owned_group", wraps=managed_process._finish_owned_group) as drain:
            result = run_group(
                [self.python, "-c", "import time; time.sleep(30)"], self.root,
                os.environ.copy(), 30, lambda: None, owned_group=group,
                absolute_deadline=work, cleanup_absolute_deadline=cleanup,
                graceful_drain_deadline=start + 0.3,
            )
        self.assertEqual(result, (124, True, None))
        self.assertEqual(drain.call_args.args[3], cleanup)

    def test_owner_release_does_not_extend_explicit_cleanup_deadline(self):
        import managed_process
        start = time.monotonic()
        work, cleanup = start + 0.2, start + 2.0
        group = OwnedProcessGroup()
        with mock.patch("managed_process._finish_owned_group", wraps=managed_process._finish_owned_group) as drain:
            result = run_group(
                [self.python, "-c", "import time; time.sleep(30)"], self.root,
                os.environ.copy(), 30, lambda: None, owned_group=group,
                absolute_deadline=work, cleanup_absolute_deadline=cleanup,
                on_stop=lambda: True,
            )
        self.assertEqual(result, (124, True, None))
        self.assertEqual(drain.call_args.args[3], cleanup)
        self.assertLessEqual(drain.call_args.args[2], cleanup)

    def test_cleanup_before_work_deadline_never_spawns(self):
        start = time.monotonic()
        with mock.patch("managed_process.subprocess.Popen") as popen:
            with self.assertRaisesRegex(ValueError, "cleanup deadline"):
                run_group([self.python], self.root, os.environ.copy(), 10, lambda: None,
                          owned_group=OwnedProcessGroup(), absolute_deadline=start + 2,
                          cleanup_absolute_deadline=start + 1)
        popen.assert_not_called()

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
             mock.patch("managed_process.os.scandir", side_effect=lambda _path: nullcontext([])):
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
             mock.patch("managed_process.os.scandir", side_effect=lambda _path: nullcontext([])):
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
             mock.patch("managed_process.os.scandir", side_effect=lambda _path: nullcontext([])):
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
             mock.patch("managed_process.os.scandir", side_effect=lambda _path: nullcontext([])):
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
                 mock.patch("managed_process.os.scandir", side_effect=lambda _path: nullcontext([])):
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
             mock.patch("managed_process.os.scandir", side_effect=lambda _path: nullcontext([])):
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

    def test_unresolved_owner_retains_original_unreaped_child_then_reobserves(self):
        owner = OwnedProcessGroup()
        child = mock.Mock(pid=123456, returncode=None)
        child.wait.return_value = 7
        clock = {"now": 10.0}

        def advance(seconds):
            clock["now"] += seconds

        with mock.patch("managed_process.subprocess.Popen", return_value=child), \
             mock.patch("managed_process.time.monotonic", side_effect=lambda: clock["now"]), \
             mock.patch("managed_process.time.sleep", side_effect=advance), \
             mock.patch("managed_process.os.waitid", return_value=object()), \
             mock.patch("managed_process.os.killpg", side_effect=ProcessLookupError), \
             mock.patch("managed_process._group_has_live_members", return_value=True):
            result = run_group([self.python], self.root, os.environ.copy(), 10,
                               lambda: None, owned_group=owner)
        self.assertEqual(result, (125, False, None))
        self.assertIsNone(child.returncode)
        child.wait.assert_not_called()

        ticks = []
        with mock.patch("managed_process.time.monotonic", return_value=20.0), \
             mock.patch("managed_process.os.waitid", return_value=object()), \
             mock.patch("managed_process.os.killpg", side_effect=ProcessLookupError), \
             mock.patch("managed_process._group_has_live_members", return_value=False):
            observed = owner.reobserve_and_drain(absolute_deadline=25.0,
                                                 on_tick=lambda: ticks.append(1))
            repeated = owner.reobserve_and_drain(absolute_deadline=25.0)
        self.assertEqual(observed, (True, 7, None))
        self.assertEqual(repeated, observed)
        self.assertTrue(ticks)
        child.wait.assert_called_once_with()
        with self.assertRaises(ValueError):
            run_group([self.python], self.root, os.environ.copy(), 10,
                      lambda: None, owned_group=owner)

    def test_reobserve_errors_never_signal_reconstructed_identity(self):
        owner = OwnedProcessGroup()
        child = mock.Mock(pid=123456, returncode=None)
        owner._bind(child)
        with mock.patch("managed_process.time.monotonic", return_value=10.0), \
             mock.patch("managed_process.os.waitid", side_effect=ChildProcessError("not a child")), \
             mock.patch("managed_process.os.killpg") as killpg:
            gone, code, error = owner.reobserve_and_drain(absolute_deadline=15.0)
        self.assertFalse(gone)
        self.assertIsNone(code)
        self.assertEqual(error["error_type"], "ChildProcessError")
        killpg.assert_not_called()
        child.wait.assert_not_called()

    def test_reobserve_expiry_and_callback_error_keep_owner(self):
        owner = OwnedProcessGroup()
        child = mock.Mock(pid=123456, returncode=None)
        owner._bind(child)
        with mock.patch("managed_process.time.monotonic", return_value=20.0), \
             mock.patch("managed_process.os.killpg") as killpg:
            gone, code, error = owner.reobserve_and_drain(absolute_deadline=15.0)
        self.assertEqual((gone, code, error["kind"]), (False, None, "deadline"))
        killpg.assert_not_called()
        with mock.patch("managed_process.time.monotonic", return_value=10.0), \
             mock.patch("managed_process.os.waitid", return_value=None), \
             mock.patch("managed_process.os.killpg") as killpg:
            gone, code, error = owner.reobserve_and_drain(
                absolute_deadline=15.0, on_tick=lambda: (_ for _ in ()).throw(ValueError("broken")))
        self.assertEqual((gone, code, error["kind"], error["message"]),
                         (False, None, "callback", "broken"))
        killpg.assert_not_called()
        child.wait.assert_not_called()

    def test_owned_drain_polls_and_preserves_original_and_cleanup_errors(self):
        owner = OwnedProcessGroup()
        child = mock.Mock(pid=123456, returncode=None)
        child.wait.return_value = -15
        calls = []

        def tick():
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("original")
            raise RuntimeError("cleanup")

        with mock.patch("managed_process.subprocess.Popen", return_value=child), \
             mock.patch("managed_process.os.waitid", side_effect=[None, object()]), \
             mock.patch("managed_process.os.killpg", side_effect=ProcessLookupError), \
             mock.patch("managed_process._group_has_live_members", return_value=False):
            result = run_group([self.python], self.root, os.environ.copy(), 10,
                               lambda: None, on_tick=tick, owned_group=owner)
        self.assertEqual(result, (125, True, "original"))
        self.assertEqual(owner.cleanup_error["message"], "cleanup")
        self.assertEqual(len(calls), 2)

    def test_owned_graceful_cleanup_does_not_signal_and_keeps_original_timeout(self):
        owner = OwnedProcessGroup()
        child = mock.Mock(pid=123456, returncode=None)
        child.wait.return_value = 7
        clock = {"now": 10.0}
        ticks = []

        def tick():
            ticks.append(1)
            if len(ticks) == 1:
                clock["now"] = 11.0

        def advance(seconds):
            clock["now"] += seconds

        with mock.patch("managed_process.subprocess.Popen", return_value=child), \
             mock.patch("managed_process.time.monotonic", side_effect=lambda: clock["now"]), \
             mock.patch("managed_process.time.sleep", side_effect=advance), \
             mock.patch("managed_process.os.waitid", side_effect=[None, None, object()]), \
             mock.patch("managed_process.os.killpg") as killpg, \
             mock.patch("managed_process._group_has_live_members", side_effect=[True, False]):
            result = run_group([self.python], self.root, os.environ.copy(), 0.5,
                               lambda: None, on_tick=tick, owned_group=owner,
                               absolute_deadline=15.0, graceful_drain_deadline=14.0)
        self.assertEqual(result, (124, True, None))
        self.assertGreaterEqual(len(ticks), 3)
        killpg.assert_not_called()
        child.wait.assert_called_once_with()
        self.assertEqual(owner.reobserve_and_drain(absolute_deadline=15.0), (True, 7, None))

    def test_owned_graceful_deadline_expiry_never_adds_signals(self):
        owner = OwnedProcessGroup()
        child = mock.Mock(pid=123456, returncode=None)
        clock = {"now": 10.0}

        def advance(seconds):
            clock["now"] += seconds

        def tick():
            clock["now"] = min(12.0, clock["now"] + 0.5)

        with mock.patch("managed_process.subprocess.Popen", return_value=child), \
             mock.patch("managed_process.time.monotonic", side_effect=lambda: clock["now"]), \
             mock.patch("managed_process.time.sleep", side_effect=advance), \
             mock.patch("managed_process.os.waitid", return_value=None), \
             mock.patch("managed_process.os.killpg") as killpg, \
             mock.patch("managed_process._group_has_live_members", return_value=True):
            result = run_group([self.python], self.root, os.environ.copy(), 0.1,
                               lambda: None, on_tick=tick, owned_group=owner,
                               absolute_deadline=12.0, graceful_drain_deadline=12.0)
        self.assertEqual(result, (124, False, None))
        self.assertIsNone(child.returncode)
        child.wait.assert_not_called()
        killpg.assert_not_called()

    def test_stop_closes_admission_before_cleanup_ticks_and_keeps_error(self):
        owner = OwnedProcessGroup()
        child = mock.Mock(pid=123456, returncode=None)
        child.wait.return_value = -15
        events = []

        def tick():
            events.append("tick")
            if len(events) == 1:
                raise ValueError("original")

        with mock.patch("managed_process.subprocess.Popen", return_value=child), \
             mock.patch("managed_process.os.waitid", side_effect=[None, object()]), \
             mock.patch("managed_process.os.killpg", side_effect=ProcessLookupError), \
             mock.patch("managed_process._group_has_live_members", return_value=False):
            result = run_group([self.python], self.root, os.environ.copy(), 10, lambda: None,
                               owned_group=owner, on_tick=tick, on_stop=lambda: events.append("stop"))
        self.assertEqual(result, (125, True, "original"))
        self.assertEqual(events[:3], ["tick", "stop", "tick"])
        self.assertIsNone(owner.stop_error)

    def test_failed_stop_does_not_poll_uncertain_admission(self):
        owner = OwnedProcessGroup()
        child = mock.Mock(pid=123456, returncode=None)
        child.wait.return_value = -15
        ticks = []

        def tick():
            ticks.append(1)
            raise ValueError("original")

        def fail_stop():
            raise RuntimeError("stop failed")

        with mock.patch("managed_process.subprocess.Popen", return_value=child), \
             mock.patch("managed_process.os.waitid", side_effect=[None, object()]), \
             mock.patch("managed_process.os.killpg", side_effect=ProcessLookupError), \
             mock.patch("managed_process._group_has_live_members", return_value=False):
            result = run_group([self.python], self.root, os.environ.copy(), 10, lambda: None,
                               owned_group=owner, on_tick=tick, on_stop=fail_stop)
        self.assertEqual(result, (125, True, "original"))
        self.assertEqual(ticks, [1])
        self.assertEqual(owner.stop_error["message"], "stop failed")

    def test_unreleased_original_owner_is_never_signalled_during_stop(self):
        owner = OwnedProcessGroup()
        child = mock.Mock(pid=123456, returncode=None)

        def fail_tick():
            raise ValueError("original")

        with mock.patch("managed_process.subprocess.Popen", return_value=child), \
             mock.patch("managed_process.os.waitid", return_value=None), \
             mock.patch("managed_process.os.killpg") as killpg:
            result = run_group([self.python], self.root, os.environ.copy(), 10, lambda: None,
                               owned_group=owner, on_tick=fail_tick, on_stop=lambda: False)
        self.assertEqual(result, (125, False, "original"))
        killpg.assert_not_called()
        child.wait.assert_not_called()


if __name__ == "__main__":
    unittest.main()
