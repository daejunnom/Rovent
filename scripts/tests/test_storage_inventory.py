"""Inventory accounting remains compatible when each tree is walked once."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import managed_storage
import storage_maintenance


class InventorySingleTraversalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "results"
        self.root.mkdir()

    def _legacy_report(self):
        return {
            "root": str(managed_storage.checked(self.root)),
            "length_bytes": managed_storage.tree_size(self.root),
            "children": [
                {"name": child.name, "length_bytes": managed_storage.tree_size(child),
                 "classification": "unclassified_preserved"}
                for child in sorted(self.root.iterdir())
            ],
        }

    def test_nested_files_and_hardlinks_keep_root_and_child_accounting(self):
        a = self.root / "a"
        b = self.root / "b"
        (a / "nested").mkdir(parents=True)
        b.mkdir()
        original = a / "nested" / "original"
        original.write_bytes(b"1234567")
        try:
            os.link(original, a / "same-child")
            os.link(original, b / "other-child")
        except OSError as error:
            self.skipTest(f"hardlinks unavailable: {error}")
        (b / "other").write_bytes(b"123")
        (self.root / "top-level").write_bytes(b"12")
        expected = self._legacy_report()
        with mock.patch.object(storage_maintenance.os, "scandir", wraps=os.scandir) as scan:
            result = storage_maintenance.inventory([self.root])
        self.assertEqual(result["roots"], [expected])
        self.assertEqual(result["sum_root_length_bytes"], 12)
        children = {row["name"]: row["length_bytes"] for row in expected["children"]}
        self.assertEqual(children, {"a": 7, "b": 10, "top-level": 2})
        self.assertEqual(scan.call_count, 3)
        self.assertFalse(result["deletion_performed"])

    def test_nested_internal_link_is_counted_without_following(self):
        child = self.root / "a"
        (child / "target").mkdir(parents=True)
        (child / "target" / "data").write_bytes(b"data")
        try:
            os.symlink("target", child / "link", target_is_directory=True)
        except OSError as error:
            self.skipTest(f"symlinks unavailable: {error}")
        expected = self._legacy_report()
        self.assertEqual(storage_maintenance.inventory([self.root])["roots"], [expected])

    def test_nested_link_to_sibling_child_is_rejected(self):
        a = self.root / "a"
        b = self.root / "b"
        a.mkdir()
        b.mkdir()
        try:
            os.symlink("../b", a / "link", target_is_directory=True)
        except OSError as error:
            self.skipTest(f"symlinks unavailable: {error}")
        with self.assertRaises(managed_storage.StorageError):
            self._legacy_report()
        with self.assertRaises(managed_storage.StorageError):
            storage_maintenance.inventory([self.root])

    def test_direct_child_link_and_overlapping_roots_are_rejected(self):
        target = self.root / "target"
        target.mkdir()
        with self.assertRaises(managed_storage.StorageError):
            storage_maintenance.inventory([self.root, target])
        try:
            os.symlink("target", self.root / "link", target_is_directory=True)
        except OSError as error:
            self.skipTest(f"symlinks unavailable: {error}")
        with self.assertRaises(managed_storage.StorageError):
            storage_maintenance.inventory([self.root])


if __name__ == "__main__":
    unittest.main()
