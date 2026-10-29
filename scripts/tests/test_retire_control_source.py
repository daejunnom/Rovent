"""Temporary, fake-only source-image retirement safety checks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import retire_control_source as control


@unittest.skipUnless(os.name == 'posix' and os.geteuid() == 0,
                     'temporary root-owned file fixtures only')
class ControlRetirementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir='/root')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.parent = self.root / 'images'
        self.catalog = self.root / 'catalog'
        self.registry = self.root / 'registry'
        self.artifacts = self.root / 'artifacts'
        for path in (self.parent, self.catalog, self.registry, self.artifacts):
            path.mkdir(mode=0o700)
        (self.catalog / 'catalog.lock').write_bytes(b'guard')
        os.chmod(self.catalog / 'catalog.lock', 0o600)
        self._record(self.catalog / 'original.json', {'physical_completion': 'unknown'})
        self._record(self.registry / 'original.json', {'custody': 'preserve'})
        self.leaf = self.parent / 'failed-image'
        self.sibling = self.parent / 'completed-image'
        self.leaf.mkdir(mode=0o700)
        self.sibling.mkdir(mode=0o700)
        self.identity = [self.leaf.stat().st_dev, self.leaf.stat().st_ino]
        sibling_identity = [self.sibling.stat().st_dev, self.sibling.stat().st_ino]
        start = {'status': 'preparing_source_input_cache', 'cache_leaf': str(self.leaf),
                 'cache_leaf_identity': self.identity,
                 'copy_process_observation': {'boot': '00000000-0000-0000-0000-000000000000'}}
        failure = {'cache_leaf': str(self.leaf), 'status': 'partial_cache_retained',
                   'terminal_sealed': False}
        self._record(self.leaf / 'preparation-start.json', start)
        self._record(self.leaf / 'failure.json', failure)
        self._record(self.sibling / 'preparation-start.json',
                     {'cache_leaf': str(self.sibling), 'cache_leaf_identity': sibling_identity})
        pins = {}
        for name in sorted(control.SOURCE_NAMES):
            raw = (name + '\n').encode()
            self._write(self.leaf / name, raw)
            self._write(self.sibling / name, raw)
            pins[name] = {'root_owned_copy': {'raw_pin': {
                'path': str(self.sibling / name), 'bytes': len(raw),
                'sha256': hashlib.sha256(raw).hexdigest()}}}
        start_raw = (self.sibling / 'preparation-start.json').read_bytes()
        keys = list(pins)
        manifest = {'status': 'prepared_source_input_cache_only', 'cache_preparation_complete': True,
                    'cache_leaf': str(self.sibling), 'cache_leaf_identity': sibling_identity,
                    'start_metadata': {'raw_pin': {'path': str(self.sibling / 'preparation-start.json'),
                                      'bytes': len(start_raw), 'sha256': hashlib.sha256(start_raw).hexdigest()}},
                    'policy_sources7': {k: pins[k] for k in keys[:7]},
                    'supervisor_sources2': {k: pins[k] for k in keys[7:]}}
        self._record(self.sibling / 'manifest.json', manifest)
        self.args = control.argparse.Namespace(
            parent=str(self.parent), leaf=self.leaf.name, sibling=self.sibling.name,
            catalog=str(self.catalog), registry=str(self.registry), artifacts=str(self.artifacts),
            artifact_prefix='original', expected_device=self.identity[0], expected_inode=self.identity[1],
            expected_start_sha256=self._sha(self.leaf / 'preparation-start.json'),
            expected_failure_sha256=self._sha(self.leaf / 'failure.json'),
            timeout=60, assume_no_external_writers=True, apply=True)
        for name, value in (('CANONICAL_PARENT', self.parent), ('CANONICAL_CATALOG', self.catalog),
                            ('CANONICAL_REGISTRY', self.registry)):
            patcher = patch.object(control, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(control, 'check_open_fds')
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _write(path, raw):
        path.write_bytes(raw)
        path.chmod(0o600)

    def _record(self, path, value):
        self._write(path, json.dumps(value).encode())

    def test_preserves_exact_originals_before_retiring_only_selected_leaf(self):
        original = {p.name: p.read_bytes() for p in self.leaf.iterdir()}
        result = control.retire(self.args)
        self.assertTrue(result['retired'])
        self.assertFalse(self.leaf.exists())
        self.assertTrue(self.sibling.is_dir())
        self.assertEqual(json.loads((self.catalog / 'original.json').read_text())['physical_completion'],
                         'unknown')
        with zipfile.ZipFile(self.artifacts / 'original.zip') as archive:
            for name, raw in original.items():
                self.assertEqual(archive.read(name), raw)
        self.assertIn('"event":"retired"', (self.artifacts / 'original.journal.jsonl').read_text())

    def test_archive_name_loss_stops_before_another_unlink(self):
        original_unlink = os.unlink
        count = 0
        def unlink_then_remove_archive(path, *args, **kwargs):
            nonlocal count
            original_unlink(path, *args, **kwargs)
            count += 1
            if count == 1:
                original_unlink(self.artifacts / 'original.zip')
        with patch.object(control.os, 'unlink', side_effect=unlink_then_remove_archive):
            with self.assertRaises((control.Refusal, FileNotFoundError)):
                control.retire(self.args)
        self.assertEqual(len(list(self.leaf.iterdir())), len(control.ROSTER) - 1)
        self.assertIn('"event":"unlinked"', (self.artifacts / 'original.journal.jsonl').read_text())

    def test_json_escaped_protected_reference_blocks_before_removal(self):
        raw = json.dumps({'source': str(self.leaf)}).replace('/', r'\/').encode()
        self._write(self.catalog / 'original.json', raw)
        with self.assertRaisesRegex(control.Refusal, 'metadata references selected leaf'):
            control.retire(self.args)
        self.assertEqual(len(list(self.leaf.iterdir())), len(control.ROSTER))
        self.assertEqual(list(self.artifacts.iterdir()), [])

    def test_unexpected_file_type_never_removed(self):
        path = self.leaf / sorted(control.SOURCE_NAMES)[0]
        path.unlink()
        path.symlink_to(self.sibling / path.name)
        with self.assertRaises(OSError):
            control.retire(self.args)
        self.assertTrue(path.is_symlink())
        self.assertEqual(list(self.artifacts.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
