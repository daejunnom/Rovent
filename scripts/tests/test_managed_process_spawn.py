"""Linux process ownership checks without model, GPU or runner dependencies."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
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


if __name__ == "__main__":
    unittest.main()
