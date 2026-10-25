"""Real bounded temp archives/owners; no ELF execution, models, GPU, or real cleanup."""
from pathlib import Path
import copy
import hashlib
import gzip
import io
import json
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
from managed_storage import StorageError, write_record
from storage_guard import lifetime
import storage_artifact_review as review


def pin(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return pin(path)


class NumericCodePlanDeclarationTests(unittest.TestCase):
    """Pure DTOs only; no filesystem, owner, View or producer is created."""

    @staticmethod
    def raw_pin(path, payload):
        return {"path": path, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}

    def plan(self, label):
        data = label.encode("utf-8")
        root = "/fixture-" + label
        catalog, output = root + "/catalog", root + "/inputs/candidate-" + label
        key = hashlib.sha256(output.encode("utf-8")).hexdigest()
        code = b"synthetic code declaration " + data
        keeper = self.raw_pin(root + "/inputs/keeper-" + label + "/" + review.NUMERIC_CODE_RELATIVE, code)
        return {"schema": review.NUMERIC_CODE_PLAN_SCHEMA, "profile": review.NUMERIC_CODE_PROFILE,
            "source_commit": hashlib.sha1(b"synthetic source " + data).hexdigest(),
            "catalog": catalog, "key": key, "output": output,
            "output_identity": [len(data) + 1, sum(data) + 1],
            "record_pin": self.raw_pin(catalog + "/" + key + ".json", b"synthetic record " + data),
            "artifact": self.raw_pin(root + "/canonical.zip", b"synthetic archive " + data),
            "archive_member": "fixture-binaries.tar.gz",
            "code": [{"relative": review.NUMERIC_CODE_RELATIVE, "tar_member": review.NUMERIC_CODE_TAR_MEMBER,
                      "bytes": len(code), "sha256": hashlib.sha256(code).hexdigest()}],
            "research": self.raw_pin(root + "/research.zip", b"synthetic research " + data),
            "raw_evidence": [{"pin": self.raw_pin(root + "/reports/result.stderr", b"synthetic raw " + data),
                              "member": "raw/result.stderr"}],
            "protected_reference_pins": [dict(keeper)], "duplicate_keeper": dict(keeper)}

    def test_plan_preserves_distinct_declared_inputs_without_retirement_authority(self):
        plans = [self.plan("alpha"), self.plan("beta")]
        for field in ("catalog", "output", "source_commit", "output_identity"):
            self.assertNotEqual(plans[0][field], plans[1][field])
        self.assertNotEqual(plans[0]["code"][0]["sha256"], plans[1]["code"][0]["sha256"])
        self.assertNotEqual(plans[0]["duplicate_keeper"]["sha256"], plans[1]["duplicate_keeper"]["sha256"])
        for value in plans:
            with self.subTest(output=value["output"]):
                original = copy.deepcopy(value)
                proposal = review.prepare_normal_deferred_selected_numeric_code_plan(value)
                self.assertEqual(proposal["status"], "PLAN_DATA_PREPARED")
                self.assertEqual(proposal["required_inputs"], [])
                self.assertEqual(proposal["plan"], original)
                self.assertEqual(proposal["proposed_plan"], original)
                self.assertEqual(value, original)
                self.assertIs(proposal["retirement_authority_issued"], False)
                self.assertIs(proposal["deletion_performed"], False)

    def test_empty_raw_evidence_requires_empty_hash_and_other_payloads_stay_positive(self):
        value = self.plan("empty-evidence")
        raw_path = value["raw_evidence"][0]["pin"]["path"]
        value["raw_evidence"][0]["pin"] = self.raw_pin(raw_path, b"")
        proposal = review.prepare_normal_deferred_selected_numeric_code_plan(value)
        self.assertEqual(proposal["plan"]["raw_evidence"], value["raw_evidence"])
        self.assertIs(proposal["retirement_authority_issued"], False)
        self.assertIs(proposal["deletion_performed"], False)
        wrong_hash = copy.deepcopy(value)
        wrong_hash["raw_evidence"][0]["pin"]["sha256"] = hashlib.sha256(b"not empty").hexdigest()
        with self.assertRaises(StorageError):
            review.prepare_normal_deferred_selected_numeric_code_plan(wrong_hash)
        for field in ("code", "artifact", "research", "record_pin"):
            with self.subTest(positive_payload=field):
                empty_payload = copy.deepcopy(value)
                if field == "code":
                    empty_payload["code"][0].update(bytes=0, sha256=hashlib.sha256(b"").hexdigest())
                    empty_payload["duplicate_keeper"] = None
                else:
                    empty_payload[field] = self.raw_pin(empty_payload[field]["path"], b"")
                with self.assertRaises(StorageError):
                    review.prepare_normal_deferred_selected_numeric_code_plan(empty_payload)


class ArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="rovezero-artifact-review-")
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name).resolve()
        self.source = self.base / "source"
        self.source.mkdir()
        self.catalog = self.base / "catalog"
        self.output = self.base / "run"
        # A separate original producer closes its own lease and exits. Parent
        # audit cannot reuse the still-live parent as a fake dead owner.
        program = """
import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from managed_outputs import OutputLease
base=Path(sys.argv[2])
lease=OutputLease(base/"run",base/"source",catalog=base/"catalog",reserve=32768,total_limit=131072).acquire()
(lease.output/"selected4").mkdir()
(lease.output/"selected4/rz-rovent-native").write_bytes(b"\\x7fELF-fixture-only-never-executed")
(lease.output/"native-result.json").write_text('{"scope":"fixture_no_native_execution"}')
lease.finish(status="fixture_original_children_closed",tree_gone=True)
"""
        subprocess.run([sys.executable, "-I", "-S", "-B", "-c", program, str(SCRIPTS), str(self.base)],
                       check=True, capture_output=True, timeout=15)
        self.record_path = next(self.catalog.glob("*.json"))
        self.record = json.loads(self.record_path.read_bytes())
        self.code = self.output / "selected4/rz-rovent-native"
        self.artifact = self.base / "canonical.zip"
        self.write_canonical(self.code.read_bytes())
        self.research = self.base / "research.zip"
        self.write_research()
        self.plan = {"schema": review.PLAN_SCHEMA, "catalog": str(self.catalog),
                     "key": self.record["key"], "record_sha256": pin(self.record_path)["sha256"],
                     "artifact": pin(self.artifact), "archive_member": "native-validation-binaries.tar.gz",
                     "code": [{"relative": "selected4/rz-rovent-native", "tar_member": "bin/rz-rovent-native",
                               **{k: v for k, v in pin(self.code).items() if k != "path"}}],
                     "research": pin(self.research)}
        self.plan_path = self.base / "plan.json"

    def write_canonical(self, raw, *, duplicate=False):
        tar_raw = io.BytesIO()
        with tarfile.open(fileobj=tar_raw, mode="w:gz") as tar:
            for _ in range(2 if duplicate else 1):
                item = tarfile.TarInfo("bin/rz-rovent-native")
                item.size = len(raw)
                tar.addfile(item, io.BytesIO(raw))
        with zipfile.ZipFile(self.artifact, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("native-validation-binaries.tar.gz", tar_raw.getvalue())

    def write_research(self, *, mismatch=False, unlisted=False):
        raw = b'{"scope":"fixture_no_native_execution"}'
        rows = [{"member": "original/result.json", "bytes": len(raw),
                 "sha256": ("0" * 64 if mismatch else hashlib.sha256(raw).hexdigest())}]
        with zipfile.ZipFile(self.research, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr(rows[0]["member"], raw)
            z.writestr("inventory.json", json.dumps(rows))
            if unlisted:
                z.writestr("unlisted.log", b"preserve me")

    def audit(self):
        return review.audit_code_cache(dump(self.plan_path, self.plan), seconds=30)

    def replace_record(self, **updates):
        self.record.update(updates)
        write_record(self.record_path, self.record)
        write_record(self.output / ".rovezero-output.json", self.record)
        self.plan["record_sha256"] = pin(self.record_path)["sha256"]

    def test_research_crc_inventory_and_every_member_no_deletion(self):
        actual = review.verify_research(pin(self.research), seconds=30)
        self.assertEqual(actual["files"], 1)
        self.assertTrue(actual["crc_and_every_member_verified"])
        self.assertFalse(actual["deletion_performed"])
        self.assertTrue(self.code.exists())

    def test_real_closed_owner_and_code_match_do_not_authorize_unlink(self):
        before = self.record_path.read_bytes()
        actual = self.audit()
        self.assertEqual(actual["status"], "CODE_CACHE_AND_RESEARCH_AUDIT_MATCHED")
        self.assertFalse(actual["retirement_authority_issued"])
        self.assertFalse(actual["deletion_performed"])
        self.assertIn("current_exe_fd_maps_census", actual["still_required_for_retirement"])
        self.assertEqual(self.record_path.read_bytes(), before)
        self.assertTrue(self.code.exists())

    def test_corrupt_research_hash_and_unlisted_files_are_rejected(self):
        for kw, expected in (({"mismatch": True}, "hash"), ({"unlisted": True}, "unlisted")):
            self.write_research(**kw)
            with self.assertRaisesRegex(StorageError, expected):
                review.verify_research(pin(self.research), seconds=30)
        self.assertTrue(self.code.exists())

    def test_zip_member_types_reject_links_special_and_dos_directories(self):
        raw = b'{"scope":"fixture_no_native_execution"}'
        rows = [{"member": "original/result.json", "bytes": len(raw),
                 "sha256": hashlib.sha256(raw).hexdigest()}]

        def archive(external_attr, create_system):
            member = zipfile.ZipInfo(rows[0]["member"])
            member.create_system = create_system
            member.external_attr = external_attr
            with zipfile.ZipFile(self.research, "w") as z:
                z.writestr(member, raw)
                z.writestr("inventory.json", json.dumps(rows))

        # Unix mode declarations must not be laundered by a DOS host marker.
        for create_system in (0, 3):
            for kind in (stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR, stat.S_IFBLK,
                         stat.S_IFSOCK, stat.S_IFDIR, 0o150000):
                with self.subTest(create_system=create_system, kind=kind):
                    archive((kind | 0o600) << 16, create_system)
                    with self.assertRaisesRegex(StorageError, "plain unencrypted ZIP"):
                        review.verify_research(pin(self.research), seconds=30)
        for dos_type in (0x10, 0x08):
            archive((stat.S_IFREG | 0o600) << 16 | dos_type, 0)
            with self.assertRaisesRegex(StorageError, "plain unencrypted ZIP"):
                review.verify_research(pin(self.research), seconds=30)
        for create_system, attrs in ((0, 0), (3, 0o600 << 16),
                                     (3, (stat.S_IFREG | 0o644) << 16), (0, 0x20)):
            with self.subTest(create_system=create_system, attrs=attrs):
                archive(attrs, create_system)
                self.assertTrue(review.verify_research(pin(self.research), seconds=30)
                                ["crc_and_every_member_verified"])
        self.assertTrue(self.code.exists())

    def test_original_owner_alive_even_after_tree_closed_remains_rejected(self):
        self.replace_record(pid=os.getpid(), lifetime=lifetime())
        with self.assertRaisesRegex(StorageError, "owner death"):
            self.audit()

    def test_unknown_closure_and_active_rows_remain_rejected(self):
        for updates in ({"tree_gone": "unknown"}, {"tree_gone": True, "state": "active"}):
            self.replace_record(**updates)
            with self.assertRaisesRegex(StorageError, "completed"):
                self.audit()

    def test_special_custody_registry_cannot_be_adopted_even_if_empty(self):
        self.catalog.with_name(self.catalog.name + ".interrupted-unbound-custody-v1").mkdir()
        with self.assertRaisesRegex(StorageError, "custody registry"):
            self.audit()
        self.assertTrue(self.code.exists())

    def test_original_row_digest_mismatch_is_preserved(self):
        self.plan["record_sha256"] = "f" * 64
        with self.assertRaisesRegex(StorageError, "raw pin"):
            self.audit()

    def test_different_canonical_bytes_and_duplicate_tar_alias_are_rejected(self):
        for duplicate in (False, True):
            self.write_canonical(self.code.read_bytes() if duplicate else b"x" * self.code.stat().st_size,
                                 duplicate=duplicate)
            self.plan["artifact"] = pin(self.artifact)
            with self.assertRaises(StorageError):
                self.audit()

    def test_candidate_hardlink_is_not_a_private_elf(self):
        try:
            os.link(self.code, self.base / "other-link")
        except OSError as e:
            self.skipTest("hardlinks unavailable: " + str(e))
        with self.assertRaisesRegex(StorageError, "FD/type/identity"):
            self.audit()

    def test_nonplain_tar_headers_are_rejected_before_extended_payload_read(self):
        for kind, declared in ((tarfile.SYMTYPE, 0), (tarfile.XHDTYPE, 2 << 30)):
            item = tarfile.TarInfo("unaccepted-header")
            item.type = kind
            item.size = declared
            item.linkname = "outside-the-reviewed-artifact"
            raw = gzip.compress(item.tobuf() + bytes(1024))
            with zipfile.ZipFile(self.artifact, "w") as z:
                z.writestr("native-validation-binaries.tar.gz", raw)
            self.plan["artifact"] = pin(self.artifact)
            with self.assertRaisesRegex(StorageError, "plain files"):
                self.audit()
        self.assertTrue(self.code.exists())

    def test_unknown_code_names_and_traversal_are_not_scanned(self):
        for relative in ("selected4/runtime.so", "../rz-rovent-native", "selected4//rz-rovent-native"):
            self.plan["code"][0]["relative"] = relative
            with self.assertRaises(StorageError):
                self.audit()

    def test_plan_hash_and_secret_paths_are_rejected_before_open(self):
        selected = dump(self.plan_path, self.plan)
        selected["sha256"] = "0" * 64
        with self.assertRaisesRegex(StorageError, "bytes changed"):
            review.audit_code_cache(selected, seconds=30)
        for name in (".env", "service-account.json", "id_ed25519", "private.key"):
            with self.assertRaisesRegex(StorageError, "secret"):
                review.raw_pin({"path": str(self.base / name), "bytes": 1, "sha256": "0" * 64}, 1024)

    def test_replaced_candidate_during_canonical_review_is_detected(self):
        original = review.canonical_code
        prevented_by_os = []
        def race(*args):
            original(*args)
            raw = self.code.read_bytes()
            try:
                self.code.unlink()
            except PermissionError:
                if os.name != "nt":
                    raise
                prevented_by_os.append(True)  # Original retained Windows FD denies replacement.
                return
            self.code.write_bytes(raw)
        with patch.object(review, "canonical_code", side_effect=race):
            if os.name == "nt":
                result = self.audit()
                self.assertEqual(prevented_by_os, [True])
                self.assertFalse(result["retirement_authority_issued"])
            else:
                with self.assertRaisesRegex(StorageError, "identity changed"):
                    self.audit()
        self.assertTrue(self.code.exists())

    def test_changed_catalog_after_archive_read_is_detected(self):
        original = review.canonical_code
        def race(*args):
            original(*args)
            self.replace_record(status="changed-after-original-audit")
        with patch.object(review, "canonical_code", side_effect=race):
            with self.assertRaisesRegex(StorageError, "catalog/row/output changed"):
                self.audit()

    def test_nonfinite_budget_and_bad_manifest_types_are_rejected(self):
        for seconds in (0, 601, True):
            with self.assertRaisesRegex(StorageError, "budget"):
                review.verify_research(pin(self.research), seconds=seconds)
        self.plan["key"] = 7
        with self.assertRaisesRegex(StorageError, "registered catalog"):
            self.audit()

    def test_cli_reports_invalid_archive_without_success_or_traceback(self):
        self.research.write_bytes(b"not a ZIP archive")
        selected = pin(self.research)
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", str(SCRIPTS / "storage_maintenance.py"), "verify-research",
             "--archive", selected["path"], "--archive-bytes", str(selected["bytes"]),
             "--archive-sha256", selected["sha256"]], capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 125)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"unknown artifacts and leases preserved", result.stderr)
        self.assertNotIn(b"Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()

