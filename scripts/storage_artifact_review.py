"""Bounded research/archive and regenerable-code audits; no retirement authority.

The original output/catalog and native-custody handlers remain responsible for
deletion. These readers never recover leases, interpret expiry as permission,
signal processes, adopt quarantine records, or modify catalog rows.
"""
from __future__ import annotations

import contextlib
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import time
import zipfile

from ci_native_bundle import ELF_CAP, ELF_PAYLOAD_CAP, FUNCTIONAL_SPECS, pinned_file, same_file_identity
from managed_storage import StorageError, checked, read_record_snapshot
from managed_outputs import _catalog_rows, _refuse_custody_registry, identity
from storage_guard import CatalogGuard, lifetime, owner_proven_dead

PLAN_SCHEMA = "rovezero.code-cache-audit.v1"
RESEARCH_CAP = 32 << 20
INVENTORY_CAP = 4 << 20
ARCHIVE_CAP = 512 << 20
MAX_FILES = 512
CODE_NAMES = {spec[2] for spec in FUNCTIONAL_SPECS}
HEX = re.compile(r"[0-9a-f]{64}\Z")
NUMERIC_CODE_PROFILE = "normal-deferred-selected-numeric-code"
NUMERIC_CODE_PLAN_SCHEMA = "rovezero.normal-deferred-selected-numeric-code-plan.v1"
NUMERIC_CODE_RELATIVE = "selected-numeric/virtual_review_numeric_capture"
NUMERIC_CODE_TAR_MEMBER = "bin/virtual_review_numeric_capture"
NATIVE_CODE_PROFILE = "normal-deferred-selected-native-code"
NATIVE_CODE_PLAN_SCHEMA = "rovezero.normal-deferred-selected-native-code-plan.v1"
NATIVE_CODE_RELATIVE = "selected-native/rz-rovent-native"
NATIVE_CODE_TAR_MEMBER = "bin/rz-rovent-native"
UTILITY_CODE_PROFILE = "normal-deferred-selected-utility-code"
UTILITY_CODE_PLAN_SCHEMA = "rovezero.normal-deferred-selected-utility-code-plan.v1"
UTILITY_CODE_RELATIVE = "rz-rovent-utility-trainer"
UTILITY_CODE_TAR_MEMBER = "bin/rz-rovent-utility-trainer"


def need(value, message):
    if value is not True:
        raise StorageError(message)


def fields(value, names):
    need(type(value) is dict and set(value) == set(names), "closed audit fields required")


def no_secret(path):
    for part in Path(path).parts:
        name = part.lower()
        need(not (name.startswith(".env") or name.endswith((".key", ".pem"))
                  or any(tag in name for tag in ("credential", "service-account", "service_account",
                                                 "id_rsa", "id_ed25519", "api_key", "api-key"))),
             "secret paths are never audit inputs")


def member_name(value):
    need(type(value) is str and 0 < len(value) <= 512 and "\\" not in value and ":" not in value,
         "bounded relative archive member required")
    p = PurePosixPath(value)
    need(not p.is_absolute() and p.as_posix() == value and all(v not in (".", "..") for v in p.parts),
         "archive member must not escape or normalize")
    no_secret(value)
    return value


def raw_pin(value, maximum):
    fields(value, ("path", "bytes", "sha256"))
    need(type(value["path"]) is str and Path(value["path"]).is_absolute(), "absolute explicit input required")
    need(type(value["bytes"]) is int and 0 < value["bytes"] <= maximum
         and type(value["sha256"]) is str and HEX.fullmatch(value["sha256"]) is not None,
         "bounded raw byte/hash pin required")
    no_secret(value["path"])
    return value


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise StorageError("nonfinite JSON constant")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


class Inputs:
    def __init__(self, seconds):
        need(type(seconds) is int and 1 <= seconds <= 600, "finite <=600s audit budget required")
        self.until = time.monotonic() + seconds
        self.held = []
        self.stack = contextlib.ExitStack()

    def tick(self):
        need(time.monotonic() < self.until, "audit time budget exceeded; originals preserved")

    def open(self, value, maximum):
        self.tick()
        raw_pin(value, maximum)
        path = checked(Path(value["path"]))
        try:
            stream, actual, before, _ = pinned_file(path, maximum)
        except RuntimeError as error:
            raise StorageError("audit FD/type/identity failure: " + str(error)) from error
        self.stack.callback(stream.close)
        need(actual == {k: value[k] for k in ("bytes", "sha256")}, "original audit input bytes changed")
        self.held.append((path, stream, before))
        self.tick()
        return stream

    def check(self):
        self.tick()
        for path, stream, before in self.held:
            checked(path)
            need(same_file_identity(before, os.fstat(stream.fileno()), path.lstat()),
                 "held audit input identity changed")

    def __enter__(self):
        return self

    def __exit__(self, kind, value, tb):
        # A close failure must not erase the original audit failure.
        try:
            self.stack.close()
        except OSError:
            if value is None:
                raise
            if hasattr(value, "add_note"):
                value.add_note("audit descriptor close also failed")
        return False


def zip_entries(z, maximum_total):
    infos = z.infolist()
    need(0 < len(infos) <= MAX_FILES + 1 and len({v.filename for v in infos}) == len(infos),
         "bounded unique ZIP inventory required")
    total = 0
    for item in infos:
        member_name(item.filename)
        # ZIP has no dedicated symlink predicate. Unix producers declare the
        # file type in the upper external-attribute bits; DOS producers often
        # leave those bits zero. Permissions without a type also occur in
        # Python ZIPs. Preserve those ordinary files, but never accept a link,
        # special inode, DOS directory/volume label, or an unknown file type.
        # Check declared Unix types even when create_system names another OS:
        # a mismatched host marker must not turn a symlink into a plain file.
        file_type = stat.S_IFMT(item.external_attr >> 16)
        need(not item.is_dir() and file_type in (0, stat.S_IFREG)
             and not item.external_attr & (0x10 | 0x08)
             and item.file_size >= 0 and not item.flag_bits & 1,
             "plain unencrypted ZIP files required")
        total += item.file_size
        need(total <= maximum_total, "ZIP uncompressed budget exceeded")
    return {item.filename: item for item in infos}


def research(inputs, pin):
    stream = inputs.open(pin, RESEARCH_CAP)
    with zipfile.ZipFile(stream) as z:
        entries = zip_entries(z, RESEARCH_CAP + INVENTORY_CAP)
        need("inventory.json" in entries and entries["inventory.json"].file_size <= INVENTORY_CAP,
             "bounded research inventory.json required")
        rows = decode(z.read("inventory.json"))
        need(type(rows) is list and 0 < len(rows) <= MAX_FILES, "bounded research members required")
        selected = set()
        for row in rows:
            inputs.tick()
            need(type(row) is dict and {"member", "bytes", "sha256"} <= set(row), "research member pin required")
            name = member_name(row["member"])
            need(name != "inventory.json" and name not in selected, "research member aliases are forbidden")
            need(type(row["bytes"]) is int and 0 <= row["bytes"] <= RESEARCH_CAP
                 and type(row["sha256"]) is str and HEX.fullmatch(row["sha256"]) is not None,
                 "bounded research member bytes/hash required")
            need(name in entries and entries[name].file_size == row["bytes"], "research member size differs")
            raw = z.read(name)  # CRC checked by ZipExtFile; already bounded by inventory.
            need((len(raw), hashlib.sha256(raw).hexdigest()) == (row["bytes"], row["sha256"]),
                 "research member hash differs")
            selected.add(name)
        need(set(entries) == selected | {"inventory.json"}, "unlisted ZIP evidence is not silently ignored")
    inputs.check()
    return {"archive": pin, "files": len(rows), "crc_and_every_member_verified": True}


def verify_research(pin, *, seconds=180):
    with Inputs(seconds) as inputs:
        result = research(inputs, pin)
        inputs.check()
        return {"schema": "rovezero.research-audit.v1", "status": "RESEARCH_ARCHIVE_VERIFIED",
                **result, "deletion_performed": False, "physical_completion_attested": False}


def canonical_code(inputs, pin, archive_member, expected):
    stream = inputs.open(pin, ARCHIVE_CAP)
    found = set()
    with zipfile.ZipFile(stream) as z:
        entries = zip_entries(z, 1 << 30)
        need(archive_member in entries, "registered nested code archive missing")
        with z.open(archive_member) as nested, gzip.GzipFile(fileobj=nested) as tar:
            count = total = 0
            def read(size):
                nonlocal total
                inputs.tick()
                need(0 <= size <= 1 << 20 and total + size <= 2 << 30, "canonical TAR stream budget exceeded")
                raw = tar.read(size)
                need(len(raw) == size, "truncated canonical TAR stream")
                total += size
                return raw
            while True:
                header = read(512)
                if header == bytes(512):
                    need(read(512) == bytes(512), "two canonical TAR end blocks required")
                    while True:
                        inputs.tick()
                        raw = tar.read(65536)
                        if not raw:
                            break
                        total += len(raw)
                        need(total <= 2 << 30 and not any(raw), "nonzero or oversized canonical TAR tail")
                    break
                item = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
                member_name(item.name)
                count += 1
                need(count <= 256 and item.size >= 0 and item.isfile() and not item.issparse(),
                     "canonical producer supports only bounded plain files, no links/PAX/sparse metadata")
                selected = item.name in expected
                if selected:
                    need(item.name not in found, "canonical ELF alias")
                    need(item.size == expected[item.name]["bytes"], "canonical ELF member size differs")
                h = hashlib.sha256()
                remaining = item.size
                while remaining:
                    block = read(min(1 << 20, remaining))
                    if selected:
                        h.update(block)
                    remaining -= len(block)
                need(not any(read((-item.size) % 512)), "nonzero canonical TAR padding")
                if selected:
                    need(h.hexdigest() == expected[item.name]["sha256"], "canonical ELF member hash differs")
                    found.add(item.name)
    need(found == set(expected), "all canonical ELF members required")
    inputs.check()


def read_code_plan(inputs, plan_pin):
    """Validate the existing closed plan; this issues no retirement authority."""
    plan = decode(inputs.open(plan_pin, INVENTORY_CAP).read(INVENTORY_CAP + 1))
    fields(plan, ("schema", "catalog", "key", "record_sha256", "artifact", "archive_member", "code", "research"))
    need(plan["schema"] == PLAN_SCHEMA and type(plan["catalog"]) is str
         and Path(plan["catalog"]).is_absolute() and type(plan["key"]) is str
         and HEX.fullmatch(plan["key"]) is not None and type(plan["record_sha256"]) is str
         and HEX.fullmatch(plan["record_sha256"]) is not None, "exact registered catalog/output required")
    no_secret(plan["catalog"])
    catalog = checked(Path(plan["catalog"]))
    need(catalog.is_dir() and (catalog / "catalog.lock").is_file(), "existing original catalog gate required")
    code = plan["code"]
    need(type(code) is list and 1 <= len(code) <= 32, "bounded explicit code list required")
    expected = {}
    paths = set()
    for row in code:
        fields(row, ("relative", "tar_member", "bytes", "sha256"))
        relative = member_name(row["relative"])
        name = PurePosixPath(relative)
        need(len(name.parts) == 2 and name.parts[0] in ("selected4", "private-elf")
             and name.name in CODE_NAMES and relative not in paths, "only explicit product code-cache leaves allowed")
        raw_pin({"path": str(Path(os.path.abspath(relative))), "bytes": row["bytes"], "sha256": row["sha256"]}, ELF_CAP)
        canonical = member_name(row["tar_member"])
        need(canonical not in expected, "canonical ELF alias")
        expected[canonical] = row
        paths.add(relative)
    need(sum(row["bytes"] for row in code) <= ELF_PAYLOAD_CAP, "aggregate ELF budget exceeded")
    archive_member = member_name(plan["archive_member"])
    raw_pin(plan["research"], RESEARCH_CAP)
    raw_pin(plan["artifact"], ARCHIVE_CAP)
    return plan, catalog, code, expected, archive_member


def _numeric_raw_pin(value, maximum, *, allow_empty=False):
    """Pure POSIX declaration checks; the held-FD readers own actual admission."""
    fields(value, ("path", "bytes", "sha256"))
    need(type(value["path"]) is str and "\\" not in value["path"]
         and not any(ord(c) < 32 for c in value["path"]), "plain POSIX raw pin path required")
    path = PurePosixPath(value["path"])
    need(path.is_absolute() and str(path) == value["path"] and ".." not in path.parts,
         "canonical absolute POSIX raw pin path required")
    need(type(allow_empty) is bool, "explicit raw evidence empty-file policy required")
    need(type(value["bytes"]) is int and (0 if allow_empty else 1) <= value["bytes"] <= maximum
         and type(value["sha256"]) is str and HEX.fullmatch(value["sha256"]) is not None
         and value["sha256"] != "0" * 64, "actual bounded bytes/hash required; no placeholder pins")
    if value["bytes"] == 0:
        need(value["sha256"] == hashlib.sha256(b"").hexdigest(), "actual empty raw evidence SHA required")
    no_secret(value["path"])
    return value


def _normal_deferred_selected_code_spec(profile):
    if profile == NUMERIC_CODE_PROFILE:
        schema, relative, member, target, features = (NUMERIC_CODE_PLAN_SCHEMA, NUMERIC_CODE_RELATIVE,
            NUMERIC_CODE_TAR_MEMBER, "virtual_review_numeric_capture", ("onnx", "contracts"))
    elif profile == NATIVE_CODE_PROFILE:
        schema, relative, member, target, features = (NATIVE_CODE_PLAN_SCHEMA, NATIVE_CODE_RELATIVE,
            NATIVE_CODE_TAR_MEMBER, "rz-rovent-native",
            ("pals-collection-onnx", "native-cuda", "native-validation-diagnostics"))
    else:
        need(profile == UTILITY_CODE_PROFILE, "explicit normal deferred selected code profile required")
        schema, relative, member, target, features = (UTILITY_CODE_PLAN_SCHEMA, UTILITY_CODE_RELATIVE,
            UTILITY_CODE_TAR_MEMBER, "rz-rovent-utility-trainer",
            ("pals-collection-onnx", "native-cuda"))
    need(target in CODE_NAMES, "registered CI selected code target required")
    return {"profile": profile, "schema": schema, "relative": relative, "tar_member": member,
            "target": target, "requested_features": features}


def _prepare_normal_deferred_selected_code_plan(value, profile):
    """Prepare closed plan data; current View/FD/owner checks own admission."""
    spec = _normal_deferred_selected_code_spec(profile)
    names = ("schema", "profile", "source_commit", "catalog", "key", "output", "output_identity",
             "record_pin", "artifact", "archive_member", "code", "research", "raw_evidence",
             "protected_reference_pins", "duplicate_keeper")
    fields(value, names)
    need(value["schema"] == spec["schema"] and value["profile"] == spec["profile"]
         and type(value["source_commit"]) is str
         and re.fullmatch("[0-9a-f]{40}", value["source_commit"]) is not None
         and value["source_commit"] != "0" * 40, "explicit selected code source/profile required")
    paths = {}
    for name in ("catalog", "output"):
        text = value[name]
        need(type(text) is str and "\\" not in text and not any(ord(c) < 32 for c in text),
             "plain selected catalog/output path required")
        path = PurePosixPath(text)
        need(path.is_absolute() and str(path) == text and ".." not in path.parts
             and str(path) != "/", "canonical selected catalog/output path required")
        no_secret(text)
        paths[name] = path
    catalog, output = paths["catalog"], paths["output"]
    need(not output.is_relative_to(catalog) and not catalog.is_relative_to(output)
         and output.parent.name == "inputs", "distinct registered input output/catalog required")
    need(type(value["key"]) is str and HEX.fullmatch(value["key"]) is not None
         and value["key"] == hashlib.sha256(str(output).encode()).hexdigest(),
         "canonical selected output key must derive from its actual normalized path")
    identity = value["output_identity"]
    need(type(identity) is list and len(identity) == 2
         and all(type(item) is int and item > 0 for item in identity), "actual output identity required")
    code = value["code"]
    need(type(code) is list and len(code) == 1 and type(code[0]) is dict,
         "one explicit selected code ELF required")
    fields(code[0], ("relative", "tar_member", "bytes", "sha256"))
    need(code[0]["relative"] == spec["relative"] and code[0]["tar_member"] == spec["tar_member"],
         "one registered selected code kind matching its explicit profile required")
    candidate = str(output / member_name(code[0]["relative"]))
    _numeric_raw_pin({"path": candidate, "bytes": code[0]["bytes"], "sha256": code[0]["sha256"]}, ELF_CAP)
    missing = []
    for name, maximum in (("record_pin", 8192), ("artifact", ARCHIVE_CAP), ("research", RESEARCH_CAP)):
        pin = value[name]
        if pin is None:
            missing.append(name)
        else:
            _numeric_raw_pin(pin, maximum)
    record_path = str(catalog / (value["key"] + ".json"))
    if value["record_pin"] is not None:
        need(value["record_pin"]["path"] == record_path, "actual selected canonical row raw pin required")
    if value["archive_member"] is None:
        missing.append("archive_member")
    else:
        member_name(value["archive_member"])
    evidence = value["raw_evidence"]
    if evidence is None:
        missing.append("raw_evidence")
    else:
        need(type(evidence) is list and 1 <= len(evidence) <= 32, "explicit original raw evidence required")
        members, evidence_paths, total = set(), set(), 0
        for item in evidence:
            fields(item, ("pin", "member"))
            pin = _numeric_raw_pin(item["pin"], RESEARCH_CAP, allow_empty=True)
            member = member_name(item["member"])
            need(member not in members | {"inventory.json", "original/catalog-row.json"}
                 and pin["path"] not in evidence_paths and pin["path"] != record_path,
                 "distinct original raw evidence, separate from canonical row, required")
            members.add(member)
            evidence_paths.add(pin["path"])
            total += pin["bytes"]
        need(total <= RESEARCH_CAP, "original raw evidence exceeds unchanged32MiB research bound")
    references = value["protected_reference_pins"]
    if references is None:
        missing.append("protected_reference_pins")
    else:
        need(type(references) is list and 1 <= len(references) <= 32,
             "nonempty explicit current protected input references required")
        reference_paths = set()
        for pin in references:
            _numeric_raw_pin(pin, ELF_CAP)
            reference = PurePosixPath(pin["path"])
            need(reference.is_relative_to(output.parent), "protected references are registered code inputs only")
            relative = reference.relative_to(output.parent)
            nested_code = (len(relative.parts) == 3 and relative.parts[1] in
                 ("selected-native", "selected-numeric", "selected-numerical-gate", "selected4", "private-elf")
                 and relative.name in CODE_NAMES)
            utility_root_code = (profile == UTILITY_CODE_PROFILE and len(relative.parts) == 2
                                 and relative.name == UTILITY_CODE_RELATIVE)
            need(nested_code or utility_root_code,
                 "model/runtime payloads are not selected code retirement inputs")
            need(pin["path"] not in reference_paths and pin["path"] != candidate,
                 "candidate is a current protected reference or reference path is duplicated")
            reference_paths.add(pin["path"])
    if value["duplicate_keeper"] is not None:
        keeper = _numeric_raw_pin(value["duplicate_keeper"], ELF_CAP)
        location = PurePosixPath(keeper["path"])
        need({key: keeper[key] for key in ("bytes", "sha256")} ==
             {key: code[0][key] for key in ("bytes", "sha256")}
             and not location.is_relative_to(output) and not location.is_relative_to(catalog),
             "optional duplicate keeper must be a distinct external same-content ELF")
    proposed = copy.deepcopy(value)
    return {"schema": "rovezero." + profile + "-proposal.v1",
            "status": "PLAN_DATA_PREPARED" if not missing else "ACTUAL_INPUT_PINS_REQUIRED",
            "profile": profile, "required_inputs": sorted(missing),
            "plan": proposed if not missing else None, "proposed_plan": proposed,
            "retirement_authority_issued": False, "deletion_performed": False,
            "owner_created_or_recovered": False, "physical_completion_attested": False}


def prepare_normal_deferred_selected_numeric_code_plan(value):
    return _prepare_normal_deferred_selected_code_plan(value, NUMERIC_CODE_PROFILE)


def prepare_normal_deferred_selected_native_code_plan(value):
    return _prepare_normal_deferred_selected_code_plan(value, NATIVE_CODE_PROFILE)


def prepare_normal_deferred_selected_utility_code_plan(value):
    return _prepare_normal_deferred_selected_code_plan(value, UTILITY_CODE_PROFILE)


def _read_normal_deferred_selected_code_plan(inputs, plan_pin, profile):
    """Read a complete data plan; live FD/View/owner checks remain mandatory."""
    plan = decode(inputs.open(plan_pin, INVENTORY_CAP).read(INVENTORY_CAP + 1))
    proposal = _prepare_normal_deferred_selected_code_plan(plan, profile)
    need(not proposal["required_inputs"], "actual selected code retirement input pins are incomplete")
    catalog = checked(Path(plan["catalog"]))
    need(catalog.is_dir() and (catalog / "catalog.lock").is_file(), "existing original catalog gate required")
    code = plan["code"]
    expected = {code[0]["tar_member"]: code[0]}
    return plan, catalog, code, expected, plan["archive_member"]


def read_normal_deferred_selected_numeric_code_plan(inputs, plan_pin):
    return _read_normal_deferred_selected_code_plan(inputs, plan_pin, NUMERIC_CODE_PROFILE)


def read_normal_deferred_selected_native_code_plan(inputs, plan_pin):
    return _read_normal_deferred_selected_code_plan(inputs, plan_pin, NATIVE_CODE_PROFILE)


def read_normal_deferred_selected_utility_code_plan(inputs, plan_pin):
    return _read_normal_deferred_selected_code_plan(inputs, plan_pin, UTILITY_CODE_PROFILE)


def audit_code_cache(plan_pin, *, seconds=300):
    """Only audit. Even a matching report cannot authorize unlink or lease recovery."""
    with Inputs(seconds) as inputs:
        plan, catalog, code, expected, archive_member = read_code_plan(inputs, plan_pin)
        with CatalogGuard(catalog / "catalog.lock") as guard:
            _refuse_custody_registry(catalog)  # Existing special custody must use its original validator.
            rows = _catalog_rows(catalog)
            matches = [row for row in rows if row[1]["key"] == plan["key"]]
            need(len(matches) == 1, "exact original output row missing")
            record_path, row, output = matches[0]
            original = read_record_snapshot(record_path, schema="rovezero.managed-output.v1")
            need(original[1] == plan["record_sha256"] and original[0] == row, "original output row raw pin changed")
            need(row["state"] == "retained" and row.get("tree_gone") is True and row.get("budget_error") is None
                 and row.get("deferred") is False, "only original completed nondeferred output can be audited")
            need(owner_proven_dead(row.get("pid"), row.get("lifetime"), lifetime()),
                 "original owner death not proven; no adoption")
            verified_research = research(inputs, plan["research"])
            for target in code:
                path = checked(output / target["relative"])
                need(path.is_relative_to(output) and identity(output) == row["identity"], "original output identity changed")
                stream = inputs.open({"path": str(path), "bytes": target["bytes"], "sha256": target["sha256"]}, ELF_CAP)
                need(stream.read(4) == b"\x7fELF", "candidate must contain actual ELF bytes")
            canonical_code(inputs, plan["artifact"], archive_member, expected)
            guard.assert_held()
            _refuse_custody_registry(catalog)
            need(read_record_snapshot(record_path, schema="rovezero.managed-output.v1") == original
                 and identity(output) == row["identity"] and _catalog_rows(catalog) == rows,
                 "original catalog/row/output changed during audit")
            inputs.check()
        return {"schema": "rovezero.code-cache-audit-result.v1", "status": "CODE_CACHE_AND_RESEARCH_AUDIT_MATCHED",
                "plan": plan_pin, "code_files": len(code), "code_bytes": sum(v["bytes"] for v in code),
                "research": verified_research, "original_row_unchanged": True, "deletion_performed": False,
                "retirement_authority_issued": False, "physical_completion_attested": False,
                "still_required_for_retirement": ["original_custody_revalidation", "current_exe_fd_maps_census",
                                                 "held_name_and_inode_check_at_unlink", "durable_retirement_journal"]}

