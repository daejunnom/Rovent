"""Catchable supervisor signals drain only the original owned test child."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from managed_process import supervisor_signal_scope


@unittest.skipUnless(sys.platform == "linux", "Linux signal and process-group ownership")
class ManagedSupervisorSignalTests(unittest.TestCase):
    def test_scope_restores_all_original_handlers(self):
        before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        with supervisor_signal_scope() as state:
            self.assertIsNone(state["signal"])
            self.assertTrue(all(signal.getsignal(sig) is not before[sig] for sig in before))
        self.assertEqual({sig: signal.getsignal(sig) for sig in before}, before)

    def test_catchable_signals_drain_original_child_and_close_owner(self):
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            with self.subTest(signal=sig), tempfile.TemporaryDirectory(prefix="rz-managed-signal-") as tmp:
                root = Path(tmp).resolve()
                source, cache = root / "source", root / "cache"
                source.mkdir()
                cache.mkdir()
                started = root / "child-started.json"
                child = root / "child.py"
                child.write_text(
                    "import json,os,sys,time\n"
                    "from pathlib import Path\n"
                    "p=Path(sys.argv[1]); q=p.with_suffix('.tmp')\n"
                    "q.write_text(json.dumps({'pid':os.getpid()})); q.replace(p)\n"
                    "while True: time.sleep(.05)\n", encoding="utf-8")
                env = os.environ.copy()
                for key in ("RUNNER_TEMP", "RZ_CONSUMER_REGISTRY_ROOT", "RZ_CONSUMER_REGISTRY_ID",
                            "RZ_CONSUMER_REGISTRY_REQUIRED", "RZ_CONSUMER_INPUT_CATALOGS"):
                    env.pop(key, None)
                env["XDG_CACHE_HOME"] = str(cache)
                supervisor = subprocess.Popen(
                    [sys.executable, "-B", str(SCRIPTS / "run_managed.py"),
                     "--source-root", str(source), "--timeout", "20", "--purpose", "signal-test",
                     "--", sys.executable, str(child), str(started)],
                    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                child_pid = None
                try:
                    deadline = time.monotonic() + 15
                    while not started.exists() and supervisor.poll() is None and time.monotonic() < deadline:
                        time.sleep(.05)
                    self.assertTrue(started.exists(), "original test child did not start")
                    child_pid = json.loads(started.read_text(encoding="utf-8"))["pid"]
                    os.kill(supervisor.pid, sig)
                    # Repeated signals must not interrupt the original drain.
                    try:
                        os.kill(supervisor.pid, sig)
                    except ProcessLookupError:
                        pass  # It may already have completed the drain.
                    _, stderr = supervisor.communicate(timeout=15)
                    self.assertEqual(supervisor.returncode, 128 + sig, stderr.decode(errors="replace"))
                    self.assertFalse((cache / "rovezero/build/managed-v1/owner.json").exists())
                    lines = stderr.decode(errors="replace").splitlines()
                    rows = [json.loads(line.removeprefix("managed_storage=")) for line in lines
                            if line.startswith("managed_storage=")]
                    self.assertEqual(len(rows), 1, lines)
                    self.assertTrue(rows[0]["tree_cleanup_verified"])
                    self.assertEqual(rows[0]["command_exit_code"], 128 + sig)
                    self.assertIsNone(rows[0]["supervisor_error"])
                    with self.assertRaises(ProcessLookupError):
                        os.kill(child_pid, 0)
                    child_pid = None  # The original child is already reaped.
                finally:
                    if supervisor.poll() is None:
                        supervisor.kill()
                        supervisor.communicate(timeout=5)
                    # This PID is exclusively our test child; if the supervisor
                    # failed, avoid leaving the process it spawned running.
                    if child_pid is not None:
                        try:
                            os.killpg(child_pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass


if __name__ == "__main__":
    unittest.main()
