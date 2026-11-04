"""Verify the explicit Git path boundary without a repository or subprocess."""
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from virtual_system_numeric_prepare import source_head


class PrepareGitBoundaryTests(unittest.TestCase):
    def test_native_git_uses_source_directly(self):
        with patch("virtual_system_numeric_prepare.subprocess.run", return_value=types.SimpleNamespace(stdout=b"abc\n")) as run:
            self.assertEqual(source_head(Path("/tmp/source"), "git", "native"), "abc")
        self.assertEqual(run.call_args.args[0], ["git", "-C", "/tmp/source", "rev-parse", "HEAD"])

    def test_windows_git_receives_converted_path(self):
        responses = [types.SimpleNamespace(stdout="C:\\Users\\강민수\\source\n".encode()),
                     types.SimpleNamespace(stdout=b"abc\r\n")]
        with patch("virtual_system_numeric_prepare.subprocess.run", side_effect=responses) as run:
            self.assertEqual(source_head(Path("/mnt/c/source"), "/git/git.exe", "windows"), "abc")
        self.assertEqual(run.call_args_list[0].args[0], ["wslpath", "-w", "/mnt/c/source"])
        self.assertEqual(run.call_args_list[1].args[0], ["/git/git.exe", "-C", "C:\\Users\\강민수\\source", "rev-parse", "HEAD"])

    def test_conversion_failure_does_not_run_git(self):
        error = subprocess.CalledProcessError(1, "wslpath")
        with patch("virtual_system_numeric_prepare.subprocess.run", side_effect=error) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                source_head(Path("/mnt/c/source"), "git.exe", "windows")
        self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
