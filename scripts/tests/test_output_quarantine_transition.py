"""Closed Windows mocks and Linux private fixtures; no real user catalog apply."""
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import managed_outputs as outputs
import managed_output_quarantine as v1
import managed_output_quarantine_transition as successor
import test_storage_lifecycle as prior_tests
from managed_storage import StorageError


class TransitionTests(unittest.TestCase):
    # Reuse the existing closed FD/metadata fixture without inheriting/rerunning
    # its test methods or modifying the existing 80-test source.
    missing = staticmethod(prior_tests.InterruptedUnboundQuarantineTests.missing)
    info = staticmethod(prior_tests.InterruptedUnboundQuarantineTests.info)
    put = prior_tests.InterruptedUnboundQuarantineTests.put
    reset_old = prior_tests.InterruptedUnboundQuarantineTests.reset_old
    make = prior_tests.InterruptedUnboundQuarantineTests.make
    manifest = prior_tests.InterruptedUnboundQuarantineTests.manifest
    stat_at = prior_tests.InterruptedUnboundQuarantineTests.stat_at
    scandir = prior_tests.InterruptedUnboundQuarantineTests.scandir
    exclusive = prior_tests.InterruptedUnboundQuarantineTests.exclusive

    def tree_bytes(self, path, **kwargs):
        if path in getattr(self, "payload_sizes", {}):
            return self.payload_sizes[path]
        return prior_tests.InterruptedUnboundQuarantineTests.tree_bytes(self, path, **kwargs)

    def setUp(self):
        prior_tests.InterruptedUnboundQuarantineTests.setUp(self)
        self.payload_sizes = {self.target: 8}
        self.mounts = {"namespace": [1, 2], "mountinfo_bytes": 10, "mountinfo_sha256": "a" * 64, "readonly_aliases": []}
        self.paths = {role: self.base / "implementation" / name for role, name in successor._SOURCE_NAMES.items()}
        for role, path in self.paths.items():
            self.put(path, ("# pinned " + role + "\nraise RuntimeError('historical bytes must never execute')\n").encode())
        self.stack.enter_context(patch.object(successor, "checked", side_effect=lambda path: Path(path)))
        self.stack.enter_context(patch.object(successor, "lifetime", side_effect=lambda: dict(self.now)))
        self.stack.enter_context(patch.object(successor, "os", self.fake_os))
        self.stack.enter_context(patch.object(successor, "_source_paths", side_effect=lambda: dict(self.paths)))
        test = self
        class Observed:
            def __init__(self, held, target):
                self.held, self.target = held, target
                self.parent_identity = v1._did(test.infos[target.parent])
                self.identity = v1._did(test.infos[target])
                self.mounts = copy.deepcopy(test.mounts)
                self.snapshot = test.tree(target)
            def check(self):
                self.held.tick()
                successor._need(self.snapshot == test.tree(self.target)
                    and self.parent_identity == v1._did(test.infos[self.target.parent])
                    and self.identity == v1._did(test.infos[self.target])
                    and self.mounts == test.mounts, "observed metadata/namespace changed")
        self.stack.enter_context(patch.object(v1, "_Observed", Observed))
        class Held:
            def __init__(self, seconds):
                successor._need(0 < seconds <= 30, "original transition deadline")
                self.until, self.fds, self.blobs = prior_tests.time.monotonic() + seconds, [], []
            def tick(self):
                successor._need(prior_tests.time.monotonic() < self.until, "original transition deadline")
            def directory(self, path):
                self.tick()
                path = Path(path)
                successor._need(path in test.infos, "missing metadata directory")
                fd = max([99, *test.fd_paths]) + 1
                test.fd_paths[fd] = path
                self.fds.append(fd)
                return fd
            def read(self, pin, path, maximum):
                self.tick()
                path = Path(path)
                test.assertFalse(any(path == target or target in path.parents for target in test.payload_sizes), "target content read")
                v1._pin_shape(pin, path, maximum)
                raw = test.files[path]
                successor._need(v1._pin(path, raw) == pin, "raw pin mismatch")
                fd = max([99, *test.fd_paths]) + 1
                test.fd_paths[fd] = path
                self.fds.append(fd)
                self.blobs.append((path, fd, v1._stamp(test.infos[path]), dict(pin)))
                return raw
            def check(self):
                self.tick()
                for path, fd, before, pin in self.blobs:
                    successor._need(path in test.files and test.fd_paths.get(fd) == path
                        and v1._stamp(test.infos[path]) == before and v1._pin(path, test.files[path]) == pin,
                        "held metadata changed")
            def close(self, primary=None):
                for fd in self.fds:
                    test.fd_paths.pop(fd, None)
                self.fds.clear()
                self.blobs.clear()
        self.stack.enter_context(patch.object(v1, "_Held", Held))

        self.original_rows = {self.key: (self.row_path, self.old_pin, self.old_raw)}

    def tree(self, target):
        return {"schema": v1.SCHEMA, "kind": "snapshot", "entry_count": 1,
            "regular_bytes": self.payload_sizes[target], "entries": [{"name": ".", "stat": v1._stamp(self.infos[target])}]}

    def pins(self):
        return {role: v1._pin(path, self.files[path]) for role, path in self.paths.items()}

    def anchor(self):
        return prior_tests.InterruptedUnboundQuarantineTests.publish(self)

    def other_anchor(self, existing, name="old-two"):
        target = self.target.parent / name
        self.infos[target] = self.info(700, stat.S_IFDIR | 0o700)
        self.payload_sizes[target] = 8
        key = hashlib.sha256(os.path.normcase(str(target)).encode()).hexdigest()
        row = dict(self.old, key=key, output=str(target))
        raw = (json.dumps(row, indent=2) + chr(10)).encode()
        path = self.catalog / (key + ".json")
        self.put(path, raw)
        pin = v1._pin(path, raw)
        self.original_rows[key] = (path, pin, raw)
        return self.q.publish_custody_quarantine(self.catalog, pin, self.source_pin,
            existing_receipts=existing, root_exclusive_writer=True, apply=True)["receipt"]

    def move(self, predecessor, *, tips=None, original=None, reason="publisher_upgrade", attempt="1" * 32,
             historical=None, partial=(), apply=True):
        return successor.publish_custody_transition(self.catalog, original or self.old_pin, predecessor,
            [predecessor] if tips is None else tips, self.pins(), reason=reason, attempt_id=attempt,
            historical_source_snapshot=historical, existing_artifact_pins=partial,
            root_exclusive_writer=True, apply=apply)

    def view(self, tips):
        return successor.validate_transitions(self.catalog, outputs._catalog_rows(self.catalog), tips, self.gate)

    def assert_originals_unchanged(self):
        for path, pin, raw in self.original_rows.values():
            self.assertEqual(self.files[path], raw)
            self.assertEqual(v1._pin(path, raw), pin)
            self.assertIsNone(json.loads(raw)["identity"])
            self.assertEqual(json.loads(raw)["state"], "active")

    def change_boot(self):
        self.now.update(boot="later-boot", process_birth="later-observer")
        self.mounts.update(namespace=[1, 200], mountinfo_sha256="b" * 64)

    def change_source(self, role="transition", *, relocate=False):
        if relocate:
            self.paths[role] = self.base / "relocated" / successor._SOURCE_NAMES[role]
        self.put(self.paths[role], self.files.get(self.paths[role], b"# relocated") + b"\n# explicit new publisher\n")

    def rewrite_tip(self, pin, **changes):
        path = Path(pin["path"])
        value = v1._decode(self.files[path])
        value.update(changes)
        self.put(path, v1._encode(value))
        return v1._pin(path, self.files[path])

    def test_first_transition_preserves_anchor_and_all_sources_marker_last(self):
        anchor = self.anchor()
        before = dict(self.files)
        result = self.move(anchor)
        self.assertTrue(result["admission_ready"])
        self.assertEqual(result["generation"], 1)
        self.assertEqual([name.rsplit(".", 2)[1] for name in self.actions[-3:]], ["sources", "snapshot", "receipt"])
        for path, raw in before.items():
            self.assertEqual(self.files[path], raw)
        receipt = v1._decode(self.files[Path(result["receipt"]["path"])])
        self.assertEqual(receipt["predecessor"], anchor)
        self.assertEqual(receipt["reason"], "publisher_upgrade")
        sources = v1._decode(self.files[Path(receipt["sources"]["path"])])
        self.assertEqual(set(sources["roles"]), set(successor.ROLES))
        self.assertEqual(sources["historical_v1_publisher"]["recorded_pin"], self.source_pin)
        view = self.view([result["receipt"]])
        try:
            self.assertEqual(view.keys, frozenset([self.key]))
            self.assertEqual(view.usage(outputs._catalog_rows(self.catalog)), 4096 + sum(len(raw) for path, raw in self.files.items() if path.parent == self.registry))
        finally:
            view.close()
        self.assert_originals_unchanged()

    def test_reboot_reobserve_requires_new_tip_and_preserves_charge(self):
        anchor = self.anchor()
        first = self.move(anchor)["receipt"]
        self.change_boot()
        with self.assertRaises(StorageError):
            self.view([first])
        result = self.move(first, reason="reboot_reobserve", attempt="2" * 32)
        self.assertTrue(result["admission_ready"])
        view = self.view([result["receipt"]])
        try:
            self.assertEqual(view.records[0].charge_bytes, 4096)
        finally:
            view.close()
        self.assert_originals_unchanged()

    def test_source_upgrade_and_source_relocation_use_recorded_history_only(self):
        first = self.move(self.anchor())["receipt"]
        old_raw = self.files[Path(first["path"])]
        self.change_source(relocate=True)
        with self.assertRaises(StorageError):
            self.view([first])
        second = self.move(first, attempt="2" * 32)["receipt"]
        self.assertEqual(self.files[Path(first["path"])], old_raw)
        view = self.view([second])
        view.close()
        self.assert_originals_unchanged()

    def test_both_reason_and_reason_mismatch_never_autorefresh(self):
        anchor = self.anchor()
        self.change_boot()
        with self.assertRaisesRegex(StorageError, "reason mismatch"):
            self.move(anchor, reason="reboot_reobserve")
        first = self.move(anchor, reason="both")["receipt"]
        with self.assertRaisesRegex(StorageError, "no supported explicit reason"):
            self.move(first, attempt="2" * 32)
        self.assert_originals_unchanged()

    def test_per_key_publication_does_not_open_other_stale_key(self):
        anchor = self.anchor()
        other = self.other_anchor([anchor])
        first = self.move(anchor, tips=[anchor, other])
        self.assertFalse(first["admission_ready"])
        with self.assertRaisesRegex(StorageError, "explicit first transition"):
            self.view([first["receipt"], other])
        other_key = Path(other["path"]).name[:64]
        second = self.move(other, tips=[first["receipt"], other], original=self.original_rows[other_key][1], attempt="2" * 32)
        self.assertTrue(second["admission_ready"])
        view = self.view([first["receipt"], second["receipt"]])
        view.close()
        self.assert_originals_unchanged()

    def test_historical_source_snapshot_can_move_without_relabeling_current_file(self):
        anchor = self.anchor()
        old_body = self.files.pop(self.source_path)
        self.infos.pop(self.source_path)
        with self.assertRaises((StorageError, KeyError, FileNotFoundError)):
            self.move(anchor)
        snapshot = self.base / "history" / "publisher-original.py"
        self.put(snapshot, old_body)
        evidence = v1._pin(snapshot, old_body)
        result = self.move(anchor, historical=evidence)
        receipt = v1._decode(self.files[Path(result["receipt"]["path"])])
        document = v1._decode(self.files[Path(receipt["sources"]["path"])])
        proof = document["historical_v1_publisher"]
        self.assertEqual(proof["recorded_pin"]["path"], str(self.source_path))
        self.assertEqual(proof["evidence"]["pin"], evidence)
        self.assertNotEqual(document["roles"]["v1"]["pin"]["path"], str(self.source_path))
        self.files.pop(snapshot)
        self.infos.pop(snapshot)
        self.view([result["receipt"]]).close()  # Embedded raw evidence, not the old name.
        self.assert_originals_unchanged()

    def test_wrong_historical_source_or_current_role_path_refused_before_publication(self):
        anchor = self.anchor()
        before = dict(self.files)
        wrong = self.base / "history" / "wrong.py"
        self.put(wrong, b"unrelated")
        with self.assertRaisesRegex(StorageError, "historical snapshot"):
            self.move(anchor, historical=v1._pin(wrong, self.files[wrong]))
        pins = self.pins()
        pins["v1"] = dict(pins["transition"])
        with self.assertRaises(StorageError):
            successor.publish_custody_transition(self.catalog, self.old_pin, anchor, [anchor], pins,
                reason="publisher_upgrade", attempt_id="1" * 32, root_exclusive_writer=True, apply=True)
        self.assertEqual(len(self.actions), 3)  # Only original V1 publication.
        for path, raw in before.items():
            self.assertEqual(self.files[path], raw)

    def test_explicit_tip_selection_rejects_missing_duplicate_and_rollback(self):
        anchor = self.anchor()
        first = self.move(anchor)["receipt"]
        for tips in ([], [anchor], [first, first]):
            with self.subTest(tips=tips), self.assertRaises(StorageError):
                self.view(tips)
        self.assert_originals_unchanged()

    def test_metadata_identity_change_remains_blocked_across_boot(self):
        first = self.move(self.anchor())["receipt"]
        self.change_boot()
        before = dict(self.files)
        for path, field in ((self.target, "st_dev"), (self.target, "st_ino"), (self.target, "st_mtime_ns"),
                            (self.target, "st_ctime_ns"), (self.target.parent, "st_mode")):
            previous = getattr(self.infos[path], field)
            setattr(self.infos[path], field, previous + 1)
            with self.subTest(field=field), self.assertRaises(StorageError):
                self.move(first, reason="reboot_reobserve", attempt="2" * 32)
            setattr(self.infos[path], field, previous)
        self.assertEqual(self.files, before)
        self.assert_originals_unchanged()


    def test_mount_reobserve_needs_explicit_new_tip_and_keeps_active_view_strict(self):
        anchor = self.anchor()
        first = self.move(anchor)["receipt"]
        before, original_life, original_sources = dict(self.files), dict(self.now), self.pins()
        self.mounts.update(namespace=[1, 3], mountinfo_sha256="c" * 64)
        with self.assertRaisesRegex(StorageError, "stale"):
            self.view([first])
        with self.assertRaisesRegex(StorageError, "reason mismatch"):
            self.move(first, reason="publisher_upgrade", attempt="2" * 32)
        self.assertEqual(self.files, before)
        result = self.move(first, reason="mount_reobserve", attempt="2" * 32)
        self.assertTrue(result["admission_ready"])
        self.assertEqual(result["generation"], 2)
        self.assertEqual(self.now, original_life)
        self.assertEqual(self.pins(), original_sources)
        receipt = v1._decode(self.files[Path(result["receipt"]["path"])])
        self.assertEqual(receipt["reason"], "mount_reobserve")
        self.assertEqual(receipt["predecessor"], first)
        self.assertEqual(receipt["original_v1_receipt"], anchor)
        self.assertEqual(receipt["mount_observation"], self.mounts)
        self.assertEqual(receipt["physical_completion"], "unknown_unchanged")
        self.assertIs(receipt["original_generation_custody_reissued"], False)
        self.assertIs(receipt["numeric_success_issued"], False)
        self.assertEqual([name.rsplit(".", 2)[1] for name in self.actions[-3:]], ["sources", "snapshot", "receipt"])
        for path, raw in before.items():
            self.assertEqual(self.files[path], raw)
        owner = outputs.OutputLease(self.base / "mount-fresh", self.base / "source", catalog=self.catalog,
            reserve=4096, quarantine_transition_receipts=[result["receipt"]]).acquire()
        self.addCleanup(owner._close_quarantine_view)
        owner.check_budget()
        owner.finish(status="closed-mount-fixture-no-child", tree_gone=True)
        view = self.view([result["receipt"]])
        try:
            self.mounts.update(namespace=[1, 4], mountinfo_sha256="d" * 64)
            with self.assertRaisesRegex(StorageError, "namespace"):
                view.check()
        finally:
            view.close()
        self.assert_originals_unchanged()

    def test_mount_reason_never_overrides_no_change_or_original_three_reasons(self):
        first = self.move(self.anchor())["receipt"]
        before = dict(self.files)
        with self.assertRaisesRegex(StorageError, "no supported explicit reason"):
            self.move(first, reason="mount_reobserve", attempt="2" * 32)
        self.assertEqual(self.files, before)
        self.mounts.update(namespace=[1, 3], mountinfo_sha256="c" * 64)
        self.change_source()
        with self.assertRaisesRegex(StorageError, "reason mismatch"):
            self.move(first, reason="mount_reobserve", attempt="2" * 32)
        second = self.move(first, reason="publisher_upgrade", attempt="2" * 32)["receipt"]
        self.change_boot()
        with self.assertRaisesRegex(StorageError, "reason mismatch"):
            self.move(second, reason="mount_reobserve", attempt="3" * 32)
        third = self.move(second, reason="reboot_reobserve", attempt="3" * 32)["receipt"]
        self.now.update(boot="third-boot", process_birth="third-observer")
        self.mounts.update(namespace=[1, 300], mountinfo_sha256="d" * 64)
        self.change_source()
        with self.assertRaisesRegex(StorageError, "reason mismatch"):
            self.move(third, reason="mount_reobserve", attempt="4" * 32)
        fourth = self.move(third, reason="both", attempt="4" * 32)["receipt"]
        self.view([fourth]).close()
        self.assert_originals_unchanged()

    def test_mount_reason_refuses_malformed_previous_observation_before_publication(self):
        first = self.move(self.anchor())["receipt"]
        original = self.files[Path(first["path"])]
        old_mounts = copy.deepcopy(self.mounts)
        alias = {"mount_id": "82", "root": "/", "mountpoint": "/mnt/wslg/distro", "mount_options": "ro,relatime"}
        bad_values = [
            {}, dict(old_mounts, namespace=[True, 2]), dict(old_mounts, namespace=[1, 0]),
            dict(old_mounts, mountinfo_bytes=True), dict(old_mounts, mountinfo_sha256="A" * 64),
            dict(old_mounts, readonly_aliases={}),
            dict(old_mounts, readonly_aliases=[dict(alias, root="relative")]),
            dict(old_mounts, readonly_aliases=[dict(alias, mount_options="rw,relatime")]),
            dict(old_mounts, readonly_aliases=[dict(alias, extra=True)]),
            dict(old_mounts, readonly_aliases=[alias, dict(alias)]),
        ]
        self.mounts.update(namespace=[1, 3], mountinfo_sha256="c" * 64)
        try:
            for bad in bad_values:
                self.put(Path(first["path"]), original)
                changed = self.rewrite_tip(first, mount_observation=bad)
                before = dict(self.files)
                with self.subTest(bad=bad), self.assertRaisesRegex(StorageError, "mount"):
                    self.move(changed, reason="mount_reobserve", attempt="2" * 32)
                self.assertEqual(self.files, before)
        finally:
            self.put(Path(first["path"]), original)
        self.assert_originals_unchanged()

    def test_mount_history_recomputes_reason_and_rejects_malformed_recorded_mount(self):
        first = self.move(self.anchor())["receipt"]
        previous_mounts = copy.deepcopy(self.mounts)
        self.mounts.update(namespace=[1, 3], mountinfo_sha256="c" * 64)
        second = self.move(first, reason="mount_reobserve", attempt="2" * 32)["receipt"]
        original = self.files[Path(second["path"])]
        try:
            for changes in ({"reason": "reboot_reobserve"}, {"mount_observation": previous_mounts},
                            {"mount_observation": dict(self.mounts, namespace=[True, 3])}):
                self.put(Path(second["path"]), original)
                changed = self.rewrite_tip(second, **changes)
                with self.subTest(changes=changes), self.assertRaises(StorageError):
                    self.view([changed])
        finally:
            self.put(Path(second["path"]), original)
        self.view([second]).close()
        self.assert_originals_unchanged()

    def test_mount_reason_keeps_original_identity_alias_and_eight_generation_limits(self):
        first = self.move(self.anchor())["receipt"]
        self.mounts.update(namespace=[1, 3], mountinfo_sha256="c" * 64)
        old_device = self.infos[self.target].st_dev
        self.infos[self.target].st_dev += 1
        before = dict(self.files)
        try:
            with self.assertRaises(StorageError):
                self.move(first, reason="mount_reobserve", attempt="2" * 32)
        finally:
            self.infos[self.target].st_dev = old_device
        self.assertEqual(self.files, before)
        self.mounts["readonly_aliases"] = [{"mount_id": "82", "root": "/", "mountpoint": "/mnt/wslg/distro",
                                            "mount_options": "rw,relatime"}]
        with self.assertRaisesRegex(StorageError, "readonly mount"):
            self.move(first, reason="mount_reobserve", attempt="2" * 32)
        self.assertEqual(self.files, before)
        self.mounts["readonly_aliases"] = []
        predecessor = first
        for generation in range(2, 9):
            self.mounts.update(namespace=[1, 100 + generation], mountinfo_sha256=f"{generation:064x}")
            result = self.move(predecessor, reason="mount_reobserve", attempt=f"{generation:032x}")
            self.assertEqual(result["generation"], generation)
            predecessor = result["receipt"]
        self.view([predecessor]).close()
        self.mounts.update(namespace=[1, 109], mountinfo_sha256="9" * 64)
        before = dict(self.files)
        with self.assertRaisesRegex(StorageError, "bounded transition generation"):
            self.move(predecessor, reason="mount_reobserve", attempt="9" * 32)
        self.assertEqual(self.files, before)
        self.assert_originals_unchanged()

    def leave_partial_sources(self, anchor):
        original = self.exclusive
        def fail_snapshot(held, fd, name, raw, maximum):
            if ".transition-" in name and name.endswith(".snapshot.json"):
                raise OSError("snapshot publication failed")
            return original(held, fd, name, raw, maximum)
        with patch.object(v1, "_exclusive", side_effect=fail_snapshot):
            with self.assertRaisesRegex(OSError, "snapshot publication failed"):
                self.move(anchor)
        paths = [path for path in self.files if ".transition-" in path.name]
        self.assertEqual(len(paths), 1)
        return v1._pin(paths[0], self.files[paths[0]])

    def test_same_explicit_attempt_resumes_only_exact_partial_artifacts(self):
        anchor = self.anchor()
        artifact = self.leave_partial_sources(anchor)
        self.assert_originals_unchanged()
        with self.assertRaisesRegex(StorageError, "partial transition"):
            self.view([anchor])
        self.now["process_birth"] = "new-maintainer-same-boot"
        original_partial = self.files[Path(artifact["path"])]
        with patch.object(successor.os, "fsync") as sync:
            result = self.move(anchor, partial=[artifact])
            self.assertEqual(sync.call_count, 2)
            self.assertNotEqual(sync.call_args_list[0].args[0], sync.call_args_list[1].args[0])
        self.assertTrue(result["admission_ready"])
        self.assertEqual(self.files[Path(artifact["path"])], original_partial)
        self.assertEqual(sum(name == Path(artifact["path"]).name for name in self.actions), 1)
        self.view([result["receipt"]]).close()
        self.assert_originals_unchanged()

    def test_partial_without_pins_foreign_attempt_or_changed_bytes_stays_blocked(self):
        anchor = self.anchor()
        artifact = self.leave_partial_sources(anchor)
        before = dict(self.files)
        for arguments in ({}, {"partial": [artifact], "attempt": "2" * 32},
                          {"partial": [dict(artifact, sha256="0" * 64)]}):
            with self.subTest(arguments=arguments), self.assertRaises(StorageError):
                self.move(anchor, **arguments)
        self.assertEqual(self.files, before)
        path = Path(artifact["path"])
        self.put(path, b"partial-corrupt")
        wrong = v1._pin(path, self.files[path])
        with self.assertRaisesRegex(StorageError, "content differs"):
            self.move(anchor, partial=[wrong])
        self.assertEqual(self.files[path], b"partial-corrupt")
        self.assert_originals_unchanged()

    def test_orphan_fork_missing_unknown_registry_artifacts_never_select_maximum(self):
        anchor = self.anchor()
        first = self.move(anchor)["receipt"]
        baseline = dict(self.files)
        prefix = Path(first["path"]).name.removesuffix(".receipt.json")
        for fault in ("fork", "missing", "unknown", "orphan"):
            self.files.clear()
            self.files.update(baseline)
            if fault == "unknown":
                self.put(self.registry / "unknown.json", b"{}")
            elif fault == "orphan":
                name = next(path for path in self.files if path.name.endswith(".original.json"))
                self.files.pop(name)
            else:
                altered = prefix.replace("-01-", "-03-") if fault == "missing" else prefix[:-32] + "2" * 32
                for kind in ("sources", "snapshot", "receipt"):
                    self.put(self.registry / (altered + "." + kind + ".json"), self.files[self.registry / (prefix + "." + kind + ".json")])
            with self.subTest(fault=fault), self.assertRaises(StorageError):
                self.view([first])
        self.assert_originals_unchanged()

    def test_sources_fixed_roles_raw_hash_and_historical_join_are_strict(self):
        anchor = self.anchor()
        first = self.move(anchor)["receipt"]
        receipt = v1._decode(self.files[Path(first["path"])])
        original = v1._decode(self.files[Path(receipt["sources"]["path"])])
        base = v1._decode(self.files[Path(anchor["path"])])
        for fault in ("role", "raw", "historical"):
            value = copy.deepcopy(original)
            if fault == "role":
                value["roles"]["arbitrary"] = value["roles"]["transition"]
            elif fault == "raw":
                value["roles"]["transition"]["raw_base64"] = "d3Jvbmc="
            else:
                value["historical_v1_publisher"]["recorded_pin"] = value["roles"]["v1"]["pin"]
            with self.subTest(fault=fault), self.assertRaises(StorageError):
                successor._verify_sources(value, 1, base)
        with self.assertRaises(StorageError):
            successor._historical_pin({"path": str(self.base / ".env.py"), "bytes": 1, "sha256": "0" * 64}, 100)
        self.assert_originals_unchanged()

    def test_eight_generations_are_bounded_and_no_duplicate_completion(self):
        predecessor = self.anchor()
        for generation in range(1, 9):
            if generation > 1:
                self.change_source()
            result = self.move(predecessor, attempt=f"{generation:032x}")
            self.assertEqual(result["generation"], generation)
            predecessor = result["receipt"]
        self.change_source()
        before = dict(self.files)
        with self.assertRaisesRegex(StorageError, "bounded transition generation"):
            self.move(predecessor, attempt="9" * 32)
        self.assertEqual(self.files, before)
        self.assert_originals_unchanged()

    def test_registry_and_source_bounds_refuse_before_new_writes(self):
        anchor = self.anchor()
        before = dict(self.files)
        existing = sum(len(raw) for path, raw in self.files.items() if path.parent == self.registry)
        with patch.object(successor, "REGISTRY_LIMIT", existing + 1):
            with self.assertRaisesRegex(StorageError, "metadata bound exceeded"):
                self.move(anchor)
        self.assertEqual(self.files, before)
        self.put(self.paths["transition"], b"x" * successor.SOURCE_RAW_LIMIT)
        with self.assertRaisesRegex(StorageError, "closure raw bound"):
            self.move(anchor)
        self.assertEqual(len(self.actions), 3)
        self.assert_originals_unchanged()

    def test_charge_authority_and_reason_tampering_are_refused(self):
        first = self.move(self.anchor())["receipt"]
        raw = self.files[Path(first["path"])]
        for changes in ({"quarantine_charge_bytes": 0}, {"original_generation_custody_reissued": True},
                        {"physical_completion": "complete"}, {"reason": "both"}, {"generation": True}):
            self.put(Path(first["path"]), raw)
            changed = self.rewrite_tip(first, **changes)
            with self.subTest(changes=changes), self.assertRaises(StorageError):
                self.view([changed])
        self.assert_originals_unchanged()

    def test_upgrade_of_one_current_key_keeps_other_current_key_stale(self):
        anchor = self.anchor()
        other = self.other_anchor([anchor])
        first = self.move(anchor, tips=[anchor, other])["receipt"]
        other_key = Path(other["path"]).name[:64]
        second = self.move(other, tips=[first, other], original=self.original_rows[other_key][1], attempt="2" * 32)["receipt"]
        self.change_source()
        upgraded = self.move(first, tips=[first, second], attempt="3" * 32)
        self.assertFalse(upgraded["admission_ready"])
        with self.assertRaises(StorageError):
            self.view([upgraded["receipt"], second])
        final = self.move(second, tips=[upgraded["receipt"], second], original=self.original_rows[other_key][1], attempt="4" * 32)
        self.assertTrue(final["admission_ready"])
        self.assert_originals_unchanged()

    def test_output_lease_and_sampler_reject_changed_transition_inputs_without_mutation(self):
        first = self.move(self.anchor())["receipt"]
        with self.assertRaisesRegex(StorageError, "interrupted output"):
            outputs.OutputLease(self.base / "legacy", self.base / "source", catalog=self.catalog).acquire()
        owner = outputs.OutputLease(self.base / "new-run", self.base / "source", catalog=self.catalog,
            reserve=4096, quarantine_transition_receipts=[first]).acquire()
        self.addCleanup(owner._close_quarantine_view)
        owner_path = self.catalog / (owner.key + ".json")
        owner_record, owner_raw = copy.deepcopy(owner.record), self.files[owner_path]
        owner.check_budget()
        with patch.object(v1, "original_custody_gate", side_effect=AssertionError("no gate reacquire")):
            self.assertGreater(outputs.sample_output_budget(self.catalog, quarantine_guard=owner.guard,
                quarantine_transition_receipts=[first]), 4096)
        self.assertEqual(owner.record, owner_record)
        self.assertEqual(self.files[owner_path], owner_raw)
        owner.finish(status="mock-finite", tree_gone=True)
        self.assert_originals_unchanged()

        # Each boundary gets its own closed in-memory fixture: a rejected held
        # view remains failed, and must not be revived just for the next case.
        for boundary in ("acquire", "check_budget", "sample_output_budget"):
            for changed in ("receipt_bytes", "target_identity", "publisher_source"):
                with self.subTest(boundary=boundary, changed=changed):
                    case = type(self)("runTest")
                    try:
                        case.setUp()
                        tip = case.move(case.anchor())["receipt"]
                        candidate = outputs.OutputLease(case.base / "candidate", case.base / "source",
                            catalog=case.catalog, reserve=4096, quarantine_transition_receipts=[tip])
                        if boundary != "acquire":
                            candidate.acquire()
                            case.addCleanup(candidate._close_quarantine_view)
                        if changed == "receipt_bytes":
                            case.files[Path(tip["path"])] += b" "
                        elif changed == "target_identity":
                            case.infos[case.target].st_ino += 1
                        else:
                            case.change_source()
                        files_before = dict(case.files)
                        metadata_before = {path: vars(info).copy() for path, info in case.infos.items()}
                        actions_before = list(case.actions)
                        record_before = copy.deepcopy(candidate.record)
                        dependent_execution = []
                        with self.assertRaises(StorageError):
                            if boundary == "acquire":
                                candidate.acquire()
                            elif boundary == "check_budget":
                                candidate.check_budget()
                            else:
                                outputs.sample_output_budget(case.catalog, quarantine_guard=candidate.guard,
                                    quarantine_transition_receipts=[tip])
                            dependent_execution.append("consume_unverified_transition")
                        self.assertEqual(dependent_execution, [])
                        self.assertEqual(candidate.record, record_before)
                        self.assertEqual(case.files, files_before)
                        self.assertEqual({path: vars(info) for path, info in case.infos.items()}, metadata_before)
                        self.assertEqual(case.actions, actions_before)
                        if boundary == "acquire":
                            self.assertNotIn(candidate.output, case.infos)
                            self.assertNotIn(candidate.catalog / (candidate.key + ".json"), case.files)
                        case.assert_originals_unchanged()
                    finally:
                        case.doCleanups()

    def test_selector_mutex_outer_and_registry_overlap_refuse_before_mutation(self):
        with patch.object(outputs, "checked", side_effect=AssertionError("no path IO")) as read:
            for keywords in ({"quarantine_receipts": [], "quarantine_transition_receipts": []},
                             {"outer_supervised": True, "quarantine_transition_receipts": []}):
                with self.subTest(keywords=keywords), self.assertRaises(StorageError):
                    outputs.OutputLease(self.base / "unused", self.base / "source", **keywords)
            with self.assertRaises(StorageError):
                outputs.sample_output_budget(self.catalog, quarantine_receipts=[], quarantine_transition_receipts=[])
            read.assert_not_called()
        first = self.move(self.anchor())["receipt"]
        for path in (self.registry, self.registry.parent, self.registry / "child"):
            with self.subTest(path=path), self.assertRaisesRegex(StorageError, "overlaps custody registry"):
                outputs.OutputLease(path, self.base / "source", catalog=self.catalog, quarantine_transition_receipts=[first])
        with patch.object(v1, "original_custody_gate", side_effect=StorageError("busy gate")):
            with self.assertRaisesRegex(StorageError, "busy gate"):
                outputs.sample_output_budget(self.catalog, quarantine_transition_receipts=[first])
        self.assert_originals_unchanged()

    def test_successor_postwrite_source_change_keeps_new_active_row_and_originals(self):
        first = self.move(self.anchor())["receipt"]
        owner = outputs.OutputLease(self.base / "late", self.base / "source", catalog=self.catalog,
            reserve=4096, deferred=True, quarantine_transition_receipts=[first])
        self.addCleanup(owner._close_quarantine_view)
        row_path = self.catalog / (owner.key + ".json")
        def write_then_change(path, value):
            self.put(path, v1._encode(value))
            if path == row_path:
                self.change_source()
        with patch.object(outputs, "write_record", side_effect=write_then_change):
            with self.assertRaisesRegex(StorageError, "held metadata changed"):
                owner.acquire()
        self.assertEqual(json.loads(self.files[row_path])["state"], "active")
        self.assertFalse(owner.output.exists())
        self.assert_originals_unchanged()


    def test_resume_file_and_registry_fsync_failure_preserve_partial_and_primary(self):
        anchor = self.anchor()
        artifact = self.leave_partial_sources(anchor)
        baseline = dict(self.files)
        for directory_failure in (False, True):
            primary = OSError("resume durability failed")
            effects = [None, primary] if directory_failure else primary
            self.gate.release.side_effect = OSError("secondary gate close failed")
            try:
                with patch.object(successor.os, "fsync", side_effect=effects):
                    with self.assertRaisesRegex(OSError, "resume durability failed") as failure:
                        self.move(anchor, partial=[artifact])
                self.assertIs(failure.exception, primary)
                self.assertTrue(any("gate release also failed" in note for note in primary.__notes__))
                self.assertEqual(self.files, baseline)
                self.assertFalse(self.fd_paths)
            finally:
                self.gate.release.side_effect = None
        self.assert_originals_unchanged()

    def test_resume_name_replacement_refused_before_durability_or_marker(self):
        anchor = self.anchor()
        artifact = self.leave_partial_sources(anchor)
        path = Path(artifact["path"])
        original_read = v1._Held.read
        count = 0
        def read_then_replace(capability, pin, selected, maximum):
            nonlocal count
            raw = original_read(capability, pin, selected, maximum)
            if selected == path:
                count += 1
                if count == 2:
                    self.infos[path].st_ino += 1
            return raw
        with patch.object(v1._Held, "read", read_then_replace), patch.object(successor.os, "fsync") as sync:
            with self.assertRaisesRegex(StorageError, "held metadata changed"):
                self.move(anchor, partial=[artifact])
            sync.assert_not_called()
        self.assertFalse(any(".transition-" in path.name and path.name.endswith(".receipt.json") for path in self.files))
        self.assert_originals_unchanged()

    def test_wrong_actual_module_basename_is_rejected_before_any_transition_artifact(self):
        anchor = self.anchor()
        wrong = self.base / "implementation" / "wrong-module.py"
        self.put(wrong, self.files[self.paths["v1"]])
        self.paths["v1"] = wrong
        with self.assertRaisesRegex(StorageError, "source role/path|module filename"):
            self.move(anchor)
        self.assertEqual(len(self.actions), 3)
        self.assert_originals_unchanged()

    def test_history_metadata_can_exhaust_global18gib_before_publication(self):
        anchor = self.anchor()
        registry_bytes = sum(len(raw) for path, raw in self.files.items() if path.parent == self.registry)
        other = self.base / "retained-large"
        self.infos[other] = self.info(900, stat.S_IFDIR | 0o700)
        self.payload_sizes[other] = 18 * v1.GiB - self.old["reserved_bytes"] - registry_bytes - 1
        key = hashlib.sha256(os.path.normcase(str(other)).encode()).hexdigest()
        row = dict(self.old, key=key, output=str(other), identity=v1._did(self.infos[other])[:2],
                   state="retained", reserved_bytes=self.payload_sizes[other])
        self.put(self.catalog / (key + ".json"), v1._encode(row))
        before = dict(self.files)
        with self.assertRaisesRegex(StorageError, "metadata exceeds18GiB"):
            self.move(anchor)
        self.assertEqual(self.files, before)
        self.assert_originals_unchanged()

    def test_custody_readiness_does_not_bypass_other_active_row_acquire(self):
        anchor = self.anchor()
        other = self.base / "unclassified-active"
        key = hashlib.sha256(os.path.normcase(str(other)).encode()).hexdigest()
        self.put(self.catalog / (key + ".json"), v1._encode(dict(self.old, key=key, output=str(other))))
        result = self.move(anchor)
        self.assertTrue(result["admission_ready"])
        self.assertEqual(result["admission_scope"], successor.ADMISSION_SCOPE)
        with self.assertRaisesRegex(StorageError, "interrupted output"):
            outputs.OutputLease(self.base / "not-authorized", self.base / "source", catalog=self.catalog,
                                quarantine_transition_receipts=[result["receipt"]]).acquire()
        self.assert_originals_unchanged()

    def test_host_change_old_lifetime_and_expired_observer_refuse_without_new_artifacts(self):
        anchor = self.anchor()
        before = dict(self.files)
        original_host = self.now["host"]
        self.now["host"] = "different-host"
        with self.assertRaises(StorageError):
            self.move(anchor)
        self.now["host"] = original_host
        original_boot = self.now["boot"]
        self.now["boot"] = self.old["lifetime"]["boot"]
        with self.assertRaises(StorageError):
            self.move(anchor)
        self.now["boot"] = original_boot
        with patch.object(v1._Held, "tick", side_effect=StorageError("original deadline expired")):
            with self.assertRaisesRegex(StorageError, "deadline expired"):
                self.move(anchor)
        self.assertEqual(self.files, before)
        self.assert_originals_unchanged()

    def test_partial_marker_or_zero_length_artifact_stays_typed_blocked(self):
        anchor = self.anchor()
        artifact = self.leave_partial_sources(anchor)
        prefix = Path(artifact["path"]).name.removesuffix(".sources.json")
        self.put(self.registry / (prefix + ".receipt.json"), b"{")
        with self.assertRaisesRegex(successor.TransitionBlocked, "partial transition"):
            self.move(anchor, partial=[artifact])
        self.files.pop(self.registry / (prefix + ".receipt.json"))
        self.put(Path(artifact["path"]), b"")
        with self.assertRaisesRegex(successor.TransitionBlocked, "artifact type/owner/size"):
            self.move(anchor, partial=[artifact])
        self.assert_originals_unchanged()


@unittest.skipUnless(sys.platform == "linux", "Linux real private FD/fsync fixtures; Root executes")
class TransitionLinuxTests(unittest.TestCase):
    """Real source/row/artifact/FDs; injected lifetimes are not reboot evidence."""
    def setUp(self):
        # Method reference avoids inheriting/rerunning the original four cases.
        prior_tests.InterruptedUnboundQuarantineLinuxTests.setUp(self)
        import managed_storage
        import storage_guard
        modules = {"transition": successor, "v1": v1, "outputs": outputs,
                   "storage": managed_storage, "guard": storage_guard}
        self.source_paths = {}
        for role in successor.ROLES:
            module = modules[role]
            origin = Path(module.__file__).resolve()
            copied = self.source / successor._SOURCE_NAMES[role]
            if origin != copied:
                copied.write_bytes(origin.read_bytes())
                copied.chmod(0o600)
            self.source_paths[role] = copied
            self.stack.enter_context(patch.object(module, "__file__", str(copied)))
        self.stack.enter_context(patch.object(successor, "lifetime", side_effect=lambda: dict(self.now)))

    def pins(self):
        return {role: v1._pin(path, path.read_bytes()) for role, path in self.source_paths.items()}

    def anchor(self):
        return prior_tests.InterruptedUnboundQuarantineLinuxTests.publish(self)

    def move(self, predecessor, partial=()):
        return successor.publish_custody_transition(self.catalog, self.row_pin, predecessor, [predecessor], self.pins(),
            reason="publisher_upgrade", attempt_id="1" * 32, existing_artifact_pins=partial,
            root_exclusive_writer=True, apply=True)

    def owner(self, tip):
        owner = outputs.OutputLease(self.parent / "fresh", self.source, catalog=self.catalog,
            reserve=1 << 20, deferred=True, quarantine_transition_receipts=[tip]).acquire()
        def close_only():
            try:
                owner._close_created_directory()
            finally:
                try:
                    owner._close_quarantine_view()
                finally:
                    owner.guard.release()
        self.addCleanup(close_only)
        return owner

    def test_real_v1_transition_new_lease_uses_original_guard_and_preserves_raw(self):
        anchor = self.anchor()
        old_marker = Path(anchor["path"]).read_bytes()
        result = self.move(anchor)
        self.assertTrue(result["admission_ready"])
        self.assertEqual(result["admission_scope"], successor.ADMISSION_SCOPE)
        owner = self.owner(result["receipt"])
        owner.create_deferred_directory()
        (owner.output / "new.log").write_bytes(b"new fixture payload")
        owner.check_budget()
        self.assertGreaterEqual(outputs.sample_output_budget(self.catalog, quarantine_guard=owner.guard,
            quarantine_transition_receipts=[result["receipt"]]), self.row["reserved_bytes"])
        record = owner.finish(status="private-fixture-no-child", tree_gone=True)
        self.assertEqual(record["state"], "retained")
        self.assertEqual(record["physical_completion"], "not_inferred_from_output_lifecycle")
        self.assertEqual(Path(anchor["path"]).read_bytes(), old_marker)
        self.assertEqual(self.row_path.read_bytes(), self.raw)
        self.assertIsNone(json.loads(self.row_path.read_bytes())["identity"])

    def test_real_mount_reobserve_preserves_raw_and_new_lease_but_active_view_stays_exact(self):
        anchor = self.anchor()
        first = self.move(anchor)["receipt"]
        anchor_raw, first_raw = Path(anchor["path"]).read_bytes(), Path(first["path"]).read_bytes()
        prior = v1._decode(first_raw)["mount_observation"]
        changed = copy.deepcopy(prior)
        changed["namespace"][1] += 1
        changed["mountinfo_sha256"] = "0" * 64 if prior["mountinfo_sha256"] != "0" * 64 else "1" * 64
        current = [changed]
        # Real private files/FDs/fsync; only mount observations are injected.
        # This fixture never changes the kernel namespace or proves a reboot.
        with patch.object(v1, "_mounts", side_effect=lambda *args: copy.deepcopy(current[0])):
            with prior_tests.CatalogGuard(self.catalog / "catalog.lock") as guard:
                with self.assertRaisesRegex(StorageError, "stale"):
                    successor.validate_transitions(self.catalog, outputs._catalog_rows(self.catalog), [first], guard)
            result = successor.publish_custody_transition(self.catalog, self.row_pin, first, [first], self.pins(),
                reason="mount_reobserve", attempt_id="2" * 32, root_exclusive_writer=True, apply=True)
            self.assertTrue(result["admission_ready"])
            receipt = json.loads(Path(result["receipt"]["path"]).read_bytes())
            self.assertEqual(receipt["reason"], "mount_reobserve")
            self.assertEqual(receipt["physical_completion"], "unknown_unchanged")
            self.assertIs(receipt["original_generation_custody_reissued"], False)
            owner = self.owner(result["receipt"])
            owner.create_deferred_directory()
            record = owner.finish(status="private-mount-fixture-no-child", tree_gone=True)
            self.assertEqual(record["state"], "retained")
            with prior_tests.CatalogGuard(self.catalog / "catalog.lock") as guard:
                view = successor.validate_transitions(self.catalog, outputs._catalog_rows(self.catalog), [result["receipt"]], guard)
                try:
                    current[0] = dict(changed, namespace=[changed["namespace"][0], changed["namespace"][1] + 1])
                    with self.assertRaisesRegex(StorageError, "namespace/mount"):
                        view.check()
                finally:
                    view.close()
        self.assertEqual(Path(anchor["path"]).read_bytes(), anchor_raw)
        self.assertEqual(Path(first["path"]).read_bytes(), first_raw)
        self.assertEqual(self.row_path.read_bytes(), self.raw)

    def test_real_partial_receipt_failure_resumes_existing_fd_fsync_without_overwrite(self):
        anchor = self.anchor()
        original = v1._exclusive
        def fail_final(held, fd, name, raw, maximum):
            if ".transition-" in name and name.endswith(".receipt.json"):
                raise OSError("fixture successor receipt write failure")
            return original(held, fd, name, raw, maximum)
        with patch.object(v1, "_exclusive", side_effect=fail_final):
            with self.assertRaisesRegex(OSError, "successor receipt write failure"):
                self.move(anchor)
        partial = sorted(self.registry.glob("*.transition-01-" + "1" * 32 + ".*.json"))
        self.assertEqual({path.name.rsplit(".", 2)[1] for path in partial}, {"sources", "snapshot"})
        previous = {path: (path.stat().st_ino, path.read_bytes()) for path in partial}
        pins = [v1._pin(path, raw) for path, (_, raw) in previous.items()]
        result = self.move(anchor, pins)  # Production _sync_existing on real O_RDONLY FDs.
        self.assertTrue(result["admission_ready"])
        for path, (inode, raw) in previous.items():
            self.assertEqual(path.stat().st_ino, inode)
            self.assertEqual(path.read_bytes(), raw)
        self.assertEqual(self.row_path.read_bytes(), self.raw)

    def test_real_held_source_name_replacement_refused_even_with_identical_bytes(self):
        anchor = self.anchor()
        tip = self.move(anchor)["receipt"]
        with prior_tests.CatalogGuard(self.catalog / "catalog.lock") as guard:
            view = successor.validate_transitions(self.catalog, outputs._catalog_rows(self.catalog), [tip], guard)
            try:
                source = self.source_paths["transition"]
                raw = source.read_bytes()
                source.rename(source.with_suffix(".preserved"))
                source.write_bytes(raw)
                source.chmod(0o600)
                with self.assertRaises(StorageError):
                    view.check()
            finally:
                view.close()
        self.assertEqual(self.row_path.read_bytes(), self.raw)

    def test_real_target_replacement_cannot_admit_new_output(self):
        anchor = self.anchor()
        tip = self.move(anchor)["receipt"]
        self.target.rename(self.parent / "preserved-old")
        self.target.mkdir(mode=0o700)
        (self.target / "original.log").write_bytes(b"retained fixture payload")
        with self.assertRaises(StorageError):
            self.owner(tip)
        self.assertFalse((self.parent / "fresh").exists())
        self.assertEqual(self.row_path.read_bytes(), self.raw)


if __name__ == "__main__":
    unittest.main()
