"""Focused aggregate accounting tests using only private temporary fixtures."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rovezero_total_usage as usage


class TotalUsageTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="rovezero-total-usage-")
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name).resolve()

    def root(self, name):
        path = self.base / name
        path.mkdir()
        return path

    def scan(self, *roots, **kwargs):
        return usage.scan_total_usage(roots, threshold_bytes=10**12, **kwargs)

    def test_unique_inode_across_roots_and_overlapping_roots(self):
        first = self.root("first")
        second = self.root("second")
        (first / "nested").mkdir()
        original = first / "nested" / "data.bin"
        original.write_bytes(b"abcdef")
        try:
            os.link(original, second / "copy.bin")
        except OSError as exc:
            self.skipTest("hardlinks unavailable: " + str(exc))
        result = self.scan(first, first / "nested", second)
        self.assertEqual(result["logical_bytes_lower_bound"], 6)
        self.assertEqual(result["counts"]["unique_regular_files"], 1)
        self.assertEqual(result["counts"]["hardlinks_deduplicated"], 1)
        self.assertEqual(result["counts"]["overlapping_roots_excluded"], 1)
        for observed in result["roots"]:
            actual = Path(observed["root"]).stat()
            self.assertEqual((observed["scan_device"], observed["scan_inode"]),
                             (actual.st_dev, actual.st_ino))
            self.assertTrue(observed["identity_unchanged"])
        allocated = usage._allocated_bytes(original, original.stat())
        if allocated is None:
            self.assertEqual(result["decision"], "unknown")
        else:
            self.assertEqual(result["allocated_bytes_lower_bound"], allocated)
            self.assertEqual(result["coverage"], "complete_for_scoped_regular_files")

    def test_discover_only_matching_children(self):
        parent = self.root("tmp")
        selected = parent / "rovezero-one"
        selected.mkdir()
        (selected / "file").write_bytes(b"123")
        unrelated = parent / "unrelated"
        unrelated.mkdir()
        (unrelated / "file").write_bytes(b"other")
        result = usage.scan_total_usage([], discover_parents=[parent], threshold_bytes=10**12)
        self.assertEqual(result["logical_bytes_lower_bound"], 3)
        self.assertEqual([x["root"] for x in result["roots"]], [str(selected)])

    def test_different_device_nested_root_is_scanned_independently(self):
        parent = self.root("parent")
        child = parent / "mounted"
        child.mkdir()
        (parent / "outer").write_bytes(b"outer")
        (child / "inner").write_bytes(b"inner")
        original = usage._entry_stat

        class OtherDevice:
            def __init__(self, info):
                self.info = info
                self.st_dev = info.st_dev + 1
            def __getattr__(self, name):
                return getattr(self.info, name)

        def mount_entry(directory, item):
            info = original(directory, item)
            return OtherDevice(info) if directory == parent and item.name == "mounted" else info

        # Simulate a distinct mounted filesystem without privileged mount calls.
        with patch.object(usage, "_same_device_subtree", return_value=False), \
             patch.object(usage, "_entry_stat", side_effect=mount_entry):
            result = self.scan(parent, child)
        self.assertEqual(result["logical_bytes_lower_bound"], 10)
        self.assertEqual(result["counts"]["registered_mounts_deferred"], 1)
        self.assertEqual(len(result["roots"]), 2)
        self.assertEqual(result["coverage"], "complete_for_scoped_regular_files")

    def test_unobservable_nested_root_is_not_elided(self):
        parent = self.root("uncertain-parent")
        child = parent / "child"
        child.mkdir()
        (child / "data").write_bytes(b"payload")
        with patch.object(usage, "_same_device_subtree", return_value=None):
            result = self.scan(parent, child)
        self.assertEqual(len(result["roots"]), 2)
        self.assertEqual(result["logical_bytes_lower_bound"], len(b"payload"))
        self.assertEqual(result["coverage"], "partial")
        self.assertIn("nested_root_identity_unavailable", result["partial_reasons"])

    def test_intervening_mount_is_not_hidden_by_same_endpoint_device(self):
        parent = self.root("parent-with-mount")
        mid = parent / "mid"
        child = mid / "child"
        child.mkdir(parents=True)
        original = Path.lstat

        class OtherDevice:
            def __init__(self, info):
                self.info = info
                self.st_dev = info.st_dev + 1
            def __getattr__(self, name):
                return getattr(self.info, name)

        def synthetic_lstat(path, *args, **kwargs):
            info = original(path, *args, **kwargs)
            return OtherDevice(info) if path == mid else info

        with patch.object(Path, "lstat", synthetic_lstat):
            self.assertFalse(usage._same_device_subtree(parent, child))

    def test_private_names_are_excluded_and_force_partial(self):
        root = self.root("data")
        (root / "ordinary.txt").write_bytes(b"abc")
        (root / ".env").write_bytes(b"ignored")
        (root / "service-account.json").write_bytes(b"ignored")
        (root / "tls.key").write_bytes(b"ignored")
        private_dir = root / ".ssh"
        private_dir.mkdir()
        (private_dir / "file").write_bytes(b"ignored")
        result = self.scan(root)
        self.assertEqual(result["logical_bytes_lower_bound"], 3)
        self.assertEqual(result["partial_reasons"]["private_entries_excluded"], 4)
        self.assertEqual(result["coverage"], "partial")
        self.assertEqual(result["decision"], "unknown")

    def test_entry_limit_and_deadline_do_not_claim_within_threshold(self):
        root = self.root("bounded")
        for name in ("one", "two", "three"):
            (root / name).write_bytes(b"x")
        limited = self.scan(root, max_entries=1)
        self.assertEqual(limited["coverage"], "partial")
        self.assertEqual(limited["decision"], "unknown")
        self.assertGreater(limited["partial_reasons"]["entry_limit"], 0)
        with patch.object(usage.time, "monotonic", side_effect=[0.0, 2.0, 2.0]):
            expired = self.scan(root, max_seconds=1)
        self.assertEqual(expired["decision"], "unknown")
        self.assertGreater(expired["partial_reasons"]["deadline"], 0)

    def test_missing_or_inaccessible_root_is_not_zero_usage(self):
        missing = self.scan(self.base / "absent")
        self.assertEqual(missing["decision"], "unknown")
        self.assertIn("root_unavailable", missing["partial_reasons"])
        root = self.root("unreadable")
        with patch.object(usage, "_open_directory", side_effect=PermissionError("fixture")):
            denied = self.scan(root)
        self.assertEqual(denied["decision"], "unknown")
        self.assertIn("directory_unavailable", denied["partial_reasons"])
        self.assertFalse(denied["roots"][0]["identity_unchanged"])
        self.assertIn("root_not_opened", denied["partial_reasons"])

    def test_root_identity_change_after_scan_marks_partial(self):
        root = self.root("changing-root")
        (root / "data").write_bytes(b"payload")
        original_identity = usage._identity
        calls = 0
        root_identity = (root.stat().st_dev, root.stat().st_ino)

        def changed_identity(info):
            nonlocal calls
            result = original_identity(info)
            if result == root_identity:
                calls += 1
                # Beginning: record and directory check; ending: path check.
                if calls >= 3:
                    return (result[0], result[1] + 1)
            return result

        with patch.object(usage, "_identity", side_effect=changed_identity):
            result = self.scan(root)
        self.assertFalse(result["roots"][0]["identity_unchanged"])
        self.assertEqual(result["coverage"], "partial")
        self.assertEqual(result["decision"], "unknown")
        self.assertIn("root_identity_changed", result["partial_reasons"])

    def test_allocation_unavailable_never_substitutes_logical_bytes(self):
        root = self.root("unknown-allocation")
        (root / "file").write_bytes(b"abc")
        with patch.object(usage, "_allocated_bytes", return_value=None):
            result = self.scan(root)
        self.assertEqual(result["logical_bytes_lower_bound"], 3)
        self.assertEqual(result["allocated_bytes_lower_bound"], 0)
        self.assertEqual(result["decision"], "unknown")
        self.assertIn("allocated_bytes_unavailable", result["partial_reasons"])

    @unittest.skipUnless(os.name == "nt", "Windows metadata API only")
    def test_windows_file_standard_information_reports_fixture_allocation(self):
        root = self.root("windows-metadata")
        path = root / "ordinary.bin"
        path.write_bytes(b"abc")
        actual = usage._windows_allocation_bytes(path, path.stat())
        self.assertIsNotNone(actual)
        self.assertGreaterEqual(actual, 3)
        result = self.scan(root)
        self.assertEqual(result["allocated_bytes_lower_bound"], actual)
        self.assertEqual(result["coverage"], "complete_for_scoped_regular_files")
        different = root / "different.bin"
        different.write_bytes(b"abc")
        self.assertIsNone(usage._windows_allocation_bytes(different, path.stat()))

    def test_allocated_threshold_warning_preserves_the_file(self):
        root = self.root("actual-threshold")
        path = root / "data.bin"
        path.write_bytes(b"payload")
        allocated = usage._allocated_bytes(path, path.stat())
        if allocated is None or allocated <= 1:
            self.skipTest("fixture filesystem does not expose nonzero allocation")
        result = usage.scan_total_usage([root], threshold_bytes=allocated - 1)
        self.assertEqual(result["decision"], "exceeded")
        self.assertEqual(result["allocated_measure"], "sum_of_observed_unique_file_reported_allocations")
        self.assertEqual(path.read_bytes(), b"payload")

    def test_opt_in_threshold_stop_keeps_scanned_root_proof_and_marks_unvisited(self):
        first = self.root("a-first")
        second = self.root("z-second")
        first_file = first / "data"
        first_file.write_bytes(b"first")
        (second / "data").write_bytes(b"second")
        allocated = usage._allocated_bytes(first_file, first_file.stat())
        if allocated is None or allocated <= 1:
            self.skipTest("fixture filesystem does not expose nonzero allocation")
        result = usage.scan_total_usage([first, second], threshold_bytes=allocated - 1,
                                        stop_when_exceeded=True)
        self.assertEqual(result["decision"], "exceeded")
        self.assertEqual(result["coverage"], "partial")
        self.assertEqual(result["partial_reasons"]["threshold_short_circuit"], 1)
        self.assertEqual(result["unvisited_roots"], [str(second)])
        self.assertEqual(len(result["roots"]), 1)
        self.assertEqual(result["roots"][0]["root"], str(first))
        self.assertTrue(result["roots"][0]["identity_unchanged"])
        self.assertTrue(result["roots"][0]["partial"])

    def test_default_and_below_threshold_opt_in_finish_all_scopes(self):
        first = self.root("a-all")
        second = self.root("z-all")
        first_file = first / "data"
        first_file.write_bytes(b"first")
        (second / "data").write_bytes(b"second")
        allocated = usage._allocated_bytes(first_file, first_file.stat())
        if allocated is None or allocated <= 1:
            self.skipTest("fixture filesystem does not expose nonzero allocation")
        ordinary = usage.scan_total_usage([first, second], threshold_bytes=allocated - 1)
        self.assertEqual(ordinary["decision"], "exceeded")
        self.assertEqual(ordinary["coverage"], "complete_for_scoped_regular_files")
        self.assertEqual(ordinary["logical_bytes_lower_bound"], len(b"firstsecond"))
        self.assertEqual(ordinary["unvisited_roots"], [])
        below = usage.scan_total_usage([first, second], threshold_bytes=10**12,
                                      stop_when_exceeded=True)
        self.assertEqual(below["decision"], "within")
        self.assertEqual(below["coverage"], "complete_for_scoped_regular_files")
        self.assertEqual(below["logical_bytes_lower_bound"], len(b"firstsecond"))
        self.assertEqual(below["unvisited_roots"], [])

    def test_priority_reorders_only_selected_roots_and_keeps_partial_honest(self):
        first = self.root("a-priority")
        preferred = self.root("z-priority")
        (first / "data").write_bytes(b"first")
        chosen = preferred / "data"
        chosen.write_bytes(b"second")
        allocated = usage._allocated_bytes(chosen, chosen.stat())
        if allocated is None or allocated <= 1:
            self.skipTest("fixture filesystem does not expose nonzero allocation")
        stopped = usage.scan_total_usage([first, preferred], threshold_bytes=allocated - 1,
                                        stop_when_exceeded=True, priority_roots=[preferred])
        self.assertEqual(stopped["roots"][0]["root"], str(preferred))
        self.assertEqual(stopped["unvisited_roots"], [str(first)])
        self.assertEqual(stopped["coverage"], "partial")
        self.assertEqual(stopped["decision"], "exceeded")
        full = usage.scan_total_usage([first, preferred], threshold_bytes=10**12,
                                     stop_when_exceeded=True, priority_roots=[preferred])
        baseline = usage.scan_total_usage([first, preferred], threshold_bytes=10**12)
        self.assertEqual(full["allocated_bytes_lower_bound"], baseline["allocated_bytes_lower_bound"])
        self.assertEqual(full["unvisited_roots"], [])

    def test_symlink_is_not_followed(self):
        root = self.root("scope")
        outside = self.root("outside")
        (outside / "data").write_bytes(b"not in scope")
        try:
            (root / "linked").symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest("symlinks unavailable: " + str(exc))
        result = self.scan(root)
        self.assertEqual(result["logical_bytes_lower_bound"], 0)
        self.assertEqual(result["counts"]["symlinks_excluded"], 1)
        self.assertEqual(result["coverage"], "partial")
        self.assertEqual(result["decision"], "unknown")
        self.assertEqual(result["partial_reasons"]["symlink_scope_unresolved"], 1)

    def test_internal_link_to_visited_target_does_not_count_twice(self):
        root = self.root("inside-scope")
        target = root / "data"
        target.write_bytes(b"payload")
        try:
            (root / "alias").symlink_to(target)
        except OSError as exc:
            self.skipTest("symlinks unavailable: " + str(exc))
        result = self.scan(root)
        self.assertEqual(result["logical_bytes_lower_bound"], len(b"payload"))
        self.assertEqual(result["counts"]["symlinks_verified_seen"], 1)
        self.assertEqual(result["coverage"], "complete_for_scoped_regular_files")

    def test_link_to_another_registered_root_is_not_double_counted(self):
        root = self.root("primary")
        registered = self.root("registered")
        target = registered / "data"
        target.write_bytes(b"payload")
        try:
            (root / "alias").symlink_to(target)
        except OSError as exc:
            self.skipTest("symlinks unavailable: " + str(exc))
        result = self.scan(root, registered)
        self.assertEqual(result["logical_bytes_lower_bound"], len(b"payload"))
        self.assertEqual(result["counts"]["symlinks_verified_seen"], 1)
        self.assertEqual(result["coverage"], "complete_for_scoped_regular_files")

    def test_unresolved_link_is_partial_without_following_it(self):
        root = self.root("broken-scope")
        try:
            (root / "unresolved").symlink_to(root / "absent")
        except OSError as exc:
            self.skipTest("symlinks unavailable: " + str(exc))
        result = self.scan(root)
        self.assertEqual(result["coverage"], "partial")
        self.assertEqual(result["decision"], "unknown")
        self.assertEqual(result["partial_reasons"]["symlink_scope_unresolved"], 1)
        self.assertTrue(result["roots"][0]["partial"])

    def test_matching_discovery_symlink_is_partial_not_silently_ignored(self):
        parent = self.root("discovery-parent")
        target = self.root("actual-storage")
        try:
            (parent / "rovezero-alias").symlink_to(target, target_is_directory=True)
        except OSError as exc:
            self.skipTest("symlinks unavailable: " + str(exc))
        result = usage.scan_total_usage([], discover_parents=[parent], threshold_bytes=10**12)
        self.assertEqual(result["coverage"], "partial")
        self.assertEqual(result["decision"], "unknown")
        self.assertEqual(result["partial_reasons"]["discovery_symlink_unresolved"], 1)

    def test_warning_is_advisory_even_if_output_fails(self):
        root = self.root("warning")
        (root / "data").write_bytes(b"payload")
        result = self.scan(root)
        result["decision"] = "exceeded"
        sink = io.StringIO()
        self.assertTrue(usage.emit_usage_warning(result, sink))
        self.assertIn("rovezero_storage_total_warning", sink.getvalue())
        class Broken:
            def write(self, value):
                raise OSError("fixture failure")
        self.assertTrue(usage.emit_usage_warning(result, Broken()))
        result["decision"] = "within"
        self.assertFalse(usage.emit_usage_warning(result, sink))

    def test_cli_json_contains_auditable_non_atomic_scope(self):
        root = self.root("cli")
        (root / "data").write_bytes(b"payload")
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = usage.main(["--root", str(root), "--max-entries", "20"])
        self.assertEqual(code, 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["consistency"], "best_effort_non_atomic")
        self.assertEqual(result["host_physical_bytes"], "unknown")
        self.assertEqual(result["logical_bytes_lower_bound"], 7)


if __name__ == "__main__":
    unittest.main()
