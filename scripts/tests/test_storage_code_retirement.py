"""Temporary original-owner fixtures only; no project deletion or ELF execution."""
from pathlib import Path
import contextlib
import ctypes
import errno
import hashlib
import io
import json
import mmap
import os
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from managed_storage import StorageError, tree_size
from storage_guard import lifetime
import storage_code_retirement as retirement
import storage_maintenance as maintenance
import test_storage_artifact_review as original_fixture


# Complete /proc namespace visibility is a prerequisite for these integration
# cases. Preserve the product's rootless EACCES refusal; a CI skip cannot reuse
# or replace the separate actual privileged integration result.
privileged_linux_census = unittest.skipUnless(
    sys.platform == "linux" and getattr(os, "geteuid", lambda: -1)() == 0,
    "privileged Linux integration requires complete current /proc census; a skip is not acceptance",
)


@unittest.skipUnless(sys.platform == "linux", "code retirement requires Linux custody/census")
class CodeRetirementTests(unittest.TestCase):
    # Reuse the actual bounded producer/OutputLease setup without inheriting or
    # collecting the separate audit-only test class a second time.
    write_canonical = original_fixture.ArtifactReviewTests.write_canonical
    write_research = original_fixture.ArtifactReviewTests.write_research
    replace_record = original_fixture.ArtifactReviewTests.replace_record

    def setUp(self):
        original_fixture.ArtifactReviewTests.setUp(self)

    def retire(self):
        selected = original_fixture.dump(self.plan_path, self.plan)
        return retirement.retire_code_cache(selected, seconds=30)

    def keepers(self):
        paths = (self.artifact, self.research, self.record_path,
                 self.output / ".rovezero-output.json", self.output / "native-result.json")
        self.assertTrue(all(path.is_relative_to(self.base) for path in paths))
        return {path: path.read_bytes() for path in paths}

    def assert_preserved(self, before):
        for path, raw in before.items():
            with self.subTest(preserved=path.name):
                self.assertEqual(path.read_bytes(), raw)

    def assert_refused(self, pattern):
        before, raw = self.keepers(), self.code.read_bytes()
        with self.assertRaisesRegex(StorageError, pattern):
            self.retire()
        self.assertEqual(self.code.read_bytes(), raw)
        self.assert_preserved(before)

    @privileged_linux_census
    def test_exact_canonical_elf_retired_and_unique_evidence_preserved(self):
        before, expected_bytes = self.keepers(), self.code.stat().st_size
        result = self.retire()
        self.assertEqual(result["schema"], "rovezero.code-cache-retirement-result.v1")
        self.assertEqual(result["status"], "CODE_CACHE_RETIRED")
        self.assertTrue(result["deletion_performed"])
        self.assertEqual(result["retired_files"], ["selected4/rz-rovent-native"])
        self.assertEqual(result["retired_bytes"], expected_bytes)
        self.assertEqual(result["logical_removed_bytes"], expected_bytes)
        self.assertTrue(result["original_row_unchanged"])
        self.assertFalse(result["physical_completion_attested"])
        self.assertEqual(result["physical_space_reclaimed"], "unknown")
        self.assertEqual(result["vhd_return"], "unknown")
        self.assertFalse(self.code.exists())
        self.assertTrue(self.code.parent.is_dir())
        self.assert_preserved(before)
        self.assertEqual(original_fixture.pin(Path(result["journal"]["path"])), result["journal"])
        self.assertTrue(Path(result["journal"]["path"]).is_relative_to(self.base))

    def test_active_original_row_is_preserved(self):
        self.replace_record(state="active")
        self.assert_refused("completed|retained|active")

    def test_live_original_owner_is_not_adopted(self):
        self.replace_record(pid=os.getpid(), lifetime=lifetime())
        self.assert_refused("owner death|owner.*dead|owner.*alive")

    def test_unknown_original_tree_closure_is_preserved(self):
        self.replace_record(tree_gone="unknown")
        self.assert_refused("completed|closure|tree")

    def test_special_custody_registry_is_not_adopted(self):
        registry = self.catalog.with_name(self.catalog.name + ".interrupted-unbound-custody-v1")
        registry.mkdir()
        self.assert_refused("custody registry")
        self.assertTrue(registry.is_dir())

    def test_unknown_census_stops_before_any_candidate_deletion(self):
        with patch.object(retirement, "_process_census", side_effect=StorageError("fixture census unknown")):
            self.assert_refused("fixture census unknown")

    @privileged_linux_census
    def test_actual_held_candidate_fd_is_in_use_and_preserved(self):
        with self.code.open("rb") as held:
            self.assertEqual(held.read(4), b"\x7fELF")
            self.assert_refused("census|in.use|open.*FD|referenc")

    @privileged_linux_census
    def test_actual_mmap_with_original_fd_closed_is_rejected_by_maps_reference(self):
        # Python 3.12's mmap object retains a duplicate FD. Use the actual libc
        # mapping so the census must find /proc/maps after the sole FD closes.
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, ctypes.c_long]
        libc.mmap.restype = ctypes.c_void_p
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.munmap.restype = ctypes.c_int
        length = self.code.stat().st_size
        with self.code.open("rb") as held:
            address = libc.mmap(None, length, mmap.PROT_READ, mmap.MAP_PRIVATE, held.fileno(), 0)
        self.assertNotEqual(address, ctypes.c_void_p(-1).value,
                            "fixture mmap failed with errno " + str(ctypes.get_errno()))
        try:
            self.assertIsNotNone(address)
            self.assertEqual(ctypes.string_at(address, 4), b"\x7fELF")
            self.assert_refused("in use by current proc maps")
        finally:
            self.assertEqual(libc.munmap(address, length), 0)

    def test_candidate_hardlink_is_rejected_without_retiring_either_name(self):
        alias = self.base / "candidate-hardlink"
        os.link(self.code, alias)
        before = alias.read_bytes()
        self.assert_refused("FD/type/identity|hardlink|link count")
        self.assertEqual(alias.read_bytes(), before)

    def test_candidate_name_swap_after_archive_review_is_rejected(self):
        original = retirement.review.canonical_code
        def swap(*args):
            original(*args)
            raw = self.code.read_bytes()
            self.code.unlink()
            self.code.write_bytes(raw)
        with patch.object(retirement.review, "canonical_code", side_effect=swap):
            self.assert_refused("identity changed|candidate.*changed|held.*changed")

    def test_invalid_research_archive_is_preserved_without_deletion(self):
        self.research.write_bytes(b"not a ZIP archive")
        self.plan["research"] = original_fixture.pin(self.research)
        before = self.keepers()
        with self.assertRaises((StorageError, zipfile.BadZipFile)):
            self.retire()
        self.assertTrue(self.code.exists())
        self.assert_preserved(before)

    def test_archive_keeper_cannot_be_a_selected_elf_even_when_both_formats_are_valid(self):
        # ZIP permits an executable prefix. This temp polyglot has both the ELF
        # magic and a valid research inventory, so overlap is the deciding gate.
        raw = b"\x7fELF-fixture-only\n" + self.research.read_bytes()
        self.code.write_bytes(raw)
        self.plan["research"] = original_fixture.pin(self.code)
        self.plan["code"][0].update({key: value for key, value in original_fixture.pin(self.code).items()
                                      if key != "path"})
        self.write_canonical(raw)
        self.plan["artifact"] = original_fixture.pin(self.artifact)
        self.assert_refused("keeper|overlap|selected.*archive|archive.*selected|research.*candidate")

    def add_second_candidate(self):
        second = self.output / "selected4/rz-rovent-collect"
        second.write_bytes(b"\x7fELF-second-fixture-only-never-executed")
        self.plan["code"].append({"relative": "selected4/rz-rovent-collect",
            "tar_member": "bin/rz-rovent-collect",
            **{key: value for key, value in original_fixture.pin(second).items() if key != "path"}})
        tar_raw = io.BytesIO()
        with tarfile.open(fileobj=tar_raw, mode="w:gz") as archive:
            for row in self.plan["code"]:
                raw = (self.output / row["relative"]).read_bytes()
                item = tarfile.TarInfo(row["tar_member"])
                item.size = len(raw)
                archive.addfile(item, io.BytesIO(raw))
        with zipfile.ZipFile(self.artifact, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("native-validation-binaries.tar.gz", tar_raw.getvalue())
        self.plan["artifact"] = original_fixture.pin(self.artifact)
        return second

    def partial_unlink_failure(self, *, fail_failure_journal=False):
        second = self.add_second_candidate()
        before = self.keepers()
        candidates = {self.code.name, second.name}
        actual_unlink, actual_fsync = os.unlink, os.fsync
        calls = []
        journal_failures = []
        unlink_failed = False

        def unlink(path, *args, **kwargs):
            nonlocal unlink_failed
            name = Path(os.fsdecode(path)).name
            if name in candidates:
                calls.append(name)
                if len(calls) == 2:
                    unlink_failed = True
                    raise OSError("fixture second candidate unlink failure")
            return actual_unlink(path, *args, **kwargs)

        def fsync(fd):
            if fail_failure_journal and unlink_failed:
                journal_failures.append(fd)
                raise OSError("fixture failure-journal fsync failure")
            return actual_fsync(fd)

        with patch.object(retirement.os, "unlink", side_effect=unlink), \
                patch.object(retirement.os, "fsync", side_effect=fsync):
            with self.assertRaises((StorageError, OSError)) as caught:
                self.retire()
        self.assertEqual(len(calls), 2)
        self.assertIn("fixture second candidate unlink failure", str(caught.exception))
        actual_retired = [str(path.relative_to(self.output)) for path in (self.code, second)
                          if not path.exists()]
        self.assertEqual(len(actual_retired), 1)
        partial = caught.exception.code_cache_retirement
        self.assertEqual(partial["status"], "FAILED_OR_PARTIAL_PRESERVED")
        self.assertTrue(partial["partial"])
        self.assertEqual(partial["retired_files"], actual_retired)
        self.assertEqual(partial["logical_removed_bytes"], next(row["bytes"] for row in self.plan["code"]
                                                              if row["relative"] == actual_retired[0]))
        last = partial["last_attempt"]
        self.assertEqual(last["relative"], "selected4/" + calls[-1])
        self.assertTrue(last["unlink_attempted"])
        self.assertFalse(last["unlink_returned"])
        self.assertFalse(last["directory_fsynced"])
        if fail_failure_journal:
            self.assertTrue(journal_failures)
        self.assert_preserved(before)
        return caught.exception

    @privileged_linux_census
    def test_partial_unlink_preserves_actual_retired_prefix_and_surviving_candidate(self):
        failure = self.partial_unlink_failure()
        partial = failure.code_cache_retirement
        self.assertEqual(original_fixture.pin(Path(partial["journal"]["path"])), partial["journal"])

    @privileged_linux_census
    def test_failure_journal_error_cannot_overwrite_original_partial_unlink_failure(self):
        failure = self.partial_unlink_failure(fail_failure_journal=True)
        notes = "\n".join(getattr(failure, "__notes__", ()))
        self.assertRegex(notes, "journal|fsync")

    @privileged_linux_census
    def test_parent_directory_fsync_failure_preserves_returned_unlink_prefix_and_first_error(self):
        second = self.add_second_candidate()
        before = self.keepers()
        candidate_bytes = {path: path.read_bytes() for path in (self.code, second)}
        candidates = {self.code.name, second.name}
        primary = OSError(errno.EIO, "fixture parent-directory fsync failure")
        original_unlink, original_fsync, original_fstat = os.unlink, os.fsync, os.fstat
        removed = []

        def unlink(path, *args, **kwargs):
            result = original_unlink(path, *args, **kwargs)
            name = Path(os.fsdecode(path)).name
            if name in candidates:
                removed.append(name)
            return result

        def fsync(fd):
            if removed and stat.S_ISDIR(original_fstat(fd).st_mode):
                raise primary
            return original_fsync(fd)

        with patch.object(retirement.os, "unlink", side_effect=unlink), \
                patch.object(retirement.os, "fsync", side_effect=fsync):
            with self.assertRaises(OSError) as caught:
                self.retire()
        self.assertIs(caught.exception, primary)
        self.assertEqual(len(removed), 1)
        actual_retired = [str(path.relative_to(self.output)) for path in (self.code, second)
                          if not path.exists()]
        self.assertEqual(actual_retired, ["selected4/" + removed[0]])
        partial = caught.exception.code_cache_retirement
        self.assertTrue(partial["partial"])
        self.assertEqual(partial["retired_files"], actual_retired)
        self.assertEqual(partial["logical_removed_bytes"], next(row["bytes"] for row in self.plan["code"]
                                                              if row["relative"] == actual_retired[0]))
        self.assertTrue(partial["last_attempt"]["unlink_attempted"])
        self.assertTrue(partial["last_attempt"]["unlink_returned"])
        self.assertFalse(partial["last_attempt"]["directory_fsynced"])
        self.assertEqual(partial["last_attempt"]["relative"], actual_retired[0])
        self.assertFalse(partial["physical_completion_attested"])
        self.assertEqual(original_fixture.pin(Path(partial["journal"]["path"])), partial["journal"])
        for path, raw in candidate_bytes.items():
            if path.exists():
                self.assertEqual(path.read_bytes(), raw)
        self.assert_preserved(before)

    def test_first_fstat_failures_close_directory_and_untransferred_journal_fds(self):
        before = self.keepers()
        for boundary in ("directory", "journal"):
            with self.subTest(boundary=boundary):
                inputs = retirement._Inputs(30)
                primary = OSError(errno.EIO, "fixture first " + boundary + " fstat failure")
                opened = []
                original_open, original_fstat = os.open, os.fstat
                # Parent directory acquisition must be completed before the
                # journal-specific fault, so only its newly created FD fails.
                if boundary == "journal":
                    inputs.directory(self.output, controlled=True)

                def track_open(*args, **kwargs):
                    fd = original_open(*args, **kwargs)
                    opened.append(fd)
                    return fd

                def fail_new_fstat(fd):
                    if fd in opened:
                        raise primary
                    return original_fstat(fd)

                try:
                    with patch.object(retirement.os, "open", side_effect=track_open), \
                            patch.object(retirement.os, "fstat", side_effect=fail_new_fstat):
                        with self.assertRaises(OSError) as caught:
                            if boundary == "directory":
                                inputs.directory(self.output, controlled=True)
                            else:
                                retirement._Journal(inputs, self.output, ".fixture-journal-initial-fstat.jsonl", 4096)
                    self.assertIs(caught.exception, primary)
                finally:
                    retained = inputs.close(primary)
                self.assertIs(retained, primary)
                self.assertEqual(len(opened), 1)
                for fd in opened:
                    with self.assertRaises(OSError) as closed:
                        original_fstat(fd)
                    self.assertEqual(closed.exception.errno, errno.EBADF)
        self.assert_preserved(before)

    @privileged_linux_census
    def test_journal_write_failure_prevents_the_first_unlink(self):
        with patch.object(retirement._Journal, "append", side_effect=StorageError("fixture intent journal write failure")):
            self.assert_refused("fixture intent journal write failure")

    @privileged_linux_census
    def test_exact_duplicate_attempt_is_refused_after_durable_intent_and_failed_unlink(self):
        selected = original_fixture.dump(self.plan_path, self.plan)
        before, raw = self.keepers(), self.code.read_bytes()
        original_unlink = os.unlink
        def refuse(path, *args, **kwargs):
            if Path(os.fsdecode(path)).name == self.code.name:
                raise OSError("fixture first candidate unlink failure")
            return original_unlink(path, *args, **kwargs)
        with patch.object(retirement.os, "unlink", side_effect=refuse):
            with self.assertRaisesRegex(OSError, "fixture first candidate unlink failure"):
                retirement.retire_code_cache(selected, seconds=30)
        with self.assertRaises((StorageError, FileExistsError)) as duplicate:
            retirement.retire_code_cache(selected, seconds=30)
        if isinstance(duplicate.exception, FileExistsError):
            self.assertEqual(duplicate.exception.errno, errno.EEXIST)
        else:
            self.assertRegex(str(duplicate.exception), "duplicate|already|attempt")
        self.assertEqual(self.code.read_bytes(), raw)
        self.assert_preserved(before)

    @privileged_linux_census
    def test_full_original_reservation_cannot_be_expanded_for_a_retirement_journal(self):
        remaining = self.record["reserved_bytes"] - tree_size(self.output)
        self.assertGreater(remaining, 0)
        raw = b"r" * remaining
        unique = self.output / "unique-raw-reservation-full.log"
        unique.write_bytes(raw)
        self.assertEqual(tree_size(self.output), self.record["reserved_bytes"])
        self.assert_refused("reservation|budget|journal")
        self.assertEqual(unique.read_bytes(), raw)


@unittest.skipUnless(sys.platform == "win32", "native Windows refusal contract")
class WindowsRetirementUnsupportedTests(unittest.TestCase):
    def test_windows_refuses_before_reading_or_creating_any_path(self):
        # The unsupported boundary must precede plan I/O; no temp or project
        # path needs to be created or enumerated to prove the rejection.
        with self.assertRaises(retirement.CodeCacheRetirementUnsupported) as caught:
            retirement.retire_code_cache({}, seconds=30)
        self.assertEqual(caught.exception.code, "CodeCacheRetirementUnsupported")

    def test_windows_cli_unsupported_has_exit_125_empty_stdout_and_no_traceback(self):
        with tempfile.TemporaryDirectory(prefix="rovezero-retirement-unsupported-") as directory:
            plan = Path(directory) / "missing-plan.json"
            result = subprocess.run([sys.executable, "-I", "-S", "-B",
                str(SCRIPTS / "storage_maintenance.py"), "retire-code-cache", "--plan", str(plan),
                "--plan-bytes", "1", "--plan-sha256", "0" * 64, "--seconds", "1"],
                capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 125)
            self.assertEqual(result.stdout, b"")
            self.assertIn(b"supported only by native Linux", result.stderr)
            self.assertNotIn(b"Traceback", result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])


class RetirementFailureReportingTests(unittest.TestCase):
    def test_selected_code_facades_reject_foreign_proof_before_fd_or_effect_checks(self):
        # Opposite-kind instances deliberately have no live state. Rejection
        # must precede check(), plan access, cleanup, or any effect boundary.
        for foreign_type, audit, retire in (
                (retirement.PreparedNormalDeferredSelectedNativeCode,
                 retirement.audit_prepared_normal_deferred_selected_numeric_code,
                 retirement.retire_prepared_normal_deferred_selected_numeric_code),
                (retirement.PreparedNormalDeferredSelectedNumericCode,
                 retirement.audit_prepared_normal_deferred_selected_native_code,
                 retirement.retire_prepared_normal_deferred_selected_native_code)):
            foreign = object.__new__(foreign_type)
            with self.subTest(foreign=foreign_type.__name__):
                with self.assertRaises(StorageError):
                    audit(foreign)
                with self.assertRaises(StorageError):
                    retire(foreign, borrowed_live_view=None, source_pack=None,
                           actual_originals=None, same_guard=None)

    def test_selected_code_decode_and_profile_failures_close_accepted_inputs_once(self):
        class HeldInputs:
            until = 1

            def __init__(self):
                self.closed_with = []

            def check(self):
                pass

            def close(self, primary=None):
                self.closed_with.append(primary)
                return primary

        for proof_type, retire, other_profile in (
                (retirement.PreparedNormalDeferredSelectedNumericCode,
                 retirement.retire_prepared_normal_deferred_selected_numeric_code,
                 retirement.review.NATIVE_CODE_PROFILE),
                (retirement.PreparedNormalDeferredSelectedNativeCode,
                 retirement.retire_prepared_normal_deferred_selected_native_code,
                 retirement.review.NUMERIC_CODE_PROFILE)):
            for boundary, raw in (("decode", b"{"),
                                  ("profile", json.dumps({"profile": other_profile}).encode("utf-8"))):
                with self.subTest(proof=proof_type.__name__, boundary=boundary):
                    inputs = HeldInputs()
                    selected = {"path": "/fixture/selected-code-plan.json", "bytes": len(raw),
                                "sha256": hashlib.sha256(raw).hexdigest()}
                    prepared = proof_type(retirement._NUMERIC_PREPARED_TOKEN,
                        _inputs=inputs, _plan=raw, _pin=selected, _path=Path("/fixture/selected-code"),
                        _keeper={}, _research={}, _pid=os.getpid(), _hard_until=inputs.until,
                        _closed=False, _consumed=False)
                    with patch.object(retirement, "_numeric_borrowed_view") as effect_guard:
                        with self.assertRaises((StorageError, ValueError)) as caught:
                            retire(prepared, borrowed_live_view=None, source_pack=None,
                                   actual_originals=None, same_guard=None)
                        effect_guard.assert_not_called()
                    self.assertEqual(len(inputs.closed_with), 1)
                    self.assertIs(inputs.closed_with[0], caught.exception)
                    self.assertIs(prepared._closed, True)
                    self.assertIs(prepared._consumed, True)
                    evidence = caught.exception.code_cache_retirement
                    self.assertEqual(evidence["plan"], selected)
                    self.assertIs(evidence["journal"], None)
                    self.assertEqual(evidence["logical_removed_bytes"], 0)
                    self.assertIs(evidence["partial"], False)
                    self.assertIs(evidence["physical_completion_attested"], False)

    def test_report_failure_preserves_partial_json_when_original_exception_str_fails(self):
        class BrokenDetail(StorageError):
            def __str__(self):
                raise RuntimeError("fixture exception string failure")

        failure = BrokenDetail()
        partial = {"status": "FAILED_OR_PARTIAL_PRESERVED", "retired_files": ["selected4/rz-rovent-native"],
                   "logical_removed_bytes": 37, "partial": True}
        failure.code_cache_retirement = partial
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            maintenance.report_failure(failure)
        self.assertEqual(stdout.getvalue(), "")
        lines = stderr.getvalue().splitlines()
        self.assertTrue(lines[0].startswith("code_cache_retirement_partial="))
        self.assertEqual(json.loads(lines[0].removeprefix("code_cache_retirement_partial=")), partial)
        self.assertIn("storage_maintenance_error=BrokenDetail;", lines[1])
        self.assertNotIn("Traceback", stderr.getvalue())

        class BrokenStderr(io.StringIO):
            def write(self, value):
                raise OSError("fixture stderr write failure")

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(BrokenStderr()):
            maintenance.report_failure(failure)  # Diagnostic I/O cannot replace the first failure.
        self.assertEqual(stdout.getvalue(), "")
        self.assertIs(failure.code_cache_retirement, partial)


if __name__ == "__main__":
    unittest.main()
