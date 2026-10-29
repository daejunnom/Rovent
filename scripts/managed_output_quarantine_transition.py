"""Explicit append-only successor for preservation custody, never old ownership.

V1 artifacts and code stay immutable. A caller selects every current tip by raw
pin; a historical marker is never a current boot/source capability. This module
uses the original catalog gate and V1 FD/metadata primitives without executing
historical source bytes.
"""
from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import time

import managed_output_quarantine as v1
from managed_storage import StorageError, checked, reject_credential_names, warn_output_threshold
from storage_guard import lifetime, reboot_proves_dead

SCHEMA = "rovezero.interrupted-unbound-custody-transition.v1"
MAX_TRANSITIONS = 8
MAX_RUNS = v1.MAX_RUNS
ADMISSION_SCOPE = "selected_custody_tips_only_new_lease_revalidation_required"
REGISTRY_LIMIT = v1.REGISTRY_LIMIT
SOURCE_RAW_LIMIT = 512 << 10
SOURCES_LIMIT = 1 << 20
ROLES = ("transition", "v1", "outputs", "storage", "guard")
_SOURCE_NAMES = dict(zip(ROLES, ("managed_output_quarantine_transition.py", "managed_output_quarantine.py",
                                "managed_outputs.py", "managed_storage.py", "storage_guard.py")))
_TRANSITION = re.compile(r"([0-9a-f]{64})-([0-9a-f]{64})\.transition-(0[1-8])-([0-9a-f]{32})\.(sources|snapshot|receipt)\.json")
_V1_FIELDS = {"schema", "kind", "classification", "original_catalog_row", "original_catalog_stat",
    "original_copy", "snapshot", "output", "observed_lifetime", "publisher_source", "current_parent_identity",
    "current_postboot_observed_identity", "mount_observation", "quarantine_charge_bytes", "root_exclusive_writer_acknowledged",
    "original_generation_identity_known", "original_generation_custody_reissued", "physical_completion",
    "numeric_success_issued", "exclusion_scope"}
_TRANSITION_FIELDS = {"schema", "kind", "classification", "key", "original_catalog_row", "original_catalog_stat",
    "original_v1_receipt", "predecessor", "generation", "attempt_id", "reason", "output", "observed_lifetime",
    "current_parent_identity", "current_postboot_observed_identity", "mount_observation", "snapshot", "sources",
    "quarantine_charge_bytes", "root_exclusive_writer_acknowledged", "original_generation_identity_known",
    "original_generation_custody_reissued", "physical_completion", "numeric_success_issued", "exclusion_scope"}


class TransitionBlocked(StorageError):
    """Preserved partial/stale state; no authority or automatic repair issued."""


def _need(value, message):
    if not value:
        raise TransitionBlocked(message)


def _fields(value, names, message):
    _need(type(value) is dict and set(value) == set(names), message)


def _life(value):
    _fields(value, ("host", "boot", "platform", "process_birth"), "exact transition lifetime required")
    _need(all(isinstance(item, str) and item for item in value.values()), "nonempty transition lifetime required")


def _authority(value):
    _need(value["root_exclusive_writer_acknowledged"] is True
          and value["original_generation_identity_known"] is False
          and value["original_generation_custody_reissued"] is False
          and value["physical_completion"] == "unknown_unchanged"
          and value["numeric_success_issued"] is False
          and value["exclusion_scope"] == "output_and_all_ancestors_descendants", "transition cannot reissue historical authority")


def _source_paths():
    import managed_outputs
    import managed_storage
    import storage_guard
    return dict(zip(ROLES, map(lambda value: Path(value).resolve(),
        (__file__, v1.__file__, managed_outputs.__file__, managed_storage.__file__, storage_guard.__file__))))


def _historical_pin(pin, maximum, *, role=None):
    _need(type(pin) is dict and isinstance(pin.get("path"), str) and 0 < len(pin["path"]) <= 4096,
          "bounded historical source path required")
    path = Path(pin["path"])
    _need(path.is_absolute(), "absolute historical source path required")
    reject_credential_names(path)  # Name guard only; never follow a historical pathname.
    v1._pin_shape(pin, path, maximum)
    if role is not None:
        _need(path.name == _SOURCE_NAMES[role], "fixed historical source role/path required")
    return path


def _source_entry(pin, raw):
    return {"pin": dict(pin), "raw_base64": base64.b64encode(raw).decode("ascii")}


def _source_raw(entry, role=None):
    _fields(entry, ("pin", "raw_base64"), "exact preserved source entry required")
    _historical_pin(entry["pin"], SOURCE_RAW_LIMIT, role=role)
    _need(isinstance(entry["raw_base64"], str) and len(entry["raw_base64"]) <= 4 * SOURCE_RAW_LIMIT // 3 + 4,
          "bounded preserved source bytes required")
    try:
        raw = base64.b64decode(entry["raw_base64"], validate=True)
    except (ValueError, UnicodeError) as error:
        raise TransitionBlocked("invalid preserved source encoding") from error
    _need(base64.b64encode(raw).decode("ascii") == entry["raw_base64"]
          and v1._pin(Path(entry["pin"]["path"]), raw) == entry["pin"], "preserved source raw bytes changed")
    return raw


def _current_sources(held, pins):
    _fields(pins, ROLES, "exact five current source pins required")
    paths, result, total = _source_paths(), {}, 0
    for role in ROLES:
        _historical_pin(pins[role], SOURCE_RAW_LIMIT, role=role)
        _need(paths[role].name == _SOURCE_NAMES[role], "fixed current source module filename required")
        raw = held.read(pins[role], paths[role], SOURCE_RAW_LIMIT)
        total += len(raw)
        _need(total <= SOURCE_RAW_LIMIT, "current source closure raw bound exceeded")
        result[role] = _source_entry(pins[role], raw)
    return result


def _historical_first_source(held, receipt, snapshot_pin):
    recorded = receipt["publisher_source"]
    _historical_pin(recorded, SOURCE_RAW_LIMIT, role="v1")
    evidence = recorded if snapshot_pin is None else snapshot_pin
    path = _historical_pin(evidence, SOURCE_RAW_LIMIT)
    _need(evidence["bytes"] == recorded["bytes"]
          and evidence["sha256"] == recorded["sha256"], "historical snapshot does not match V1 publisher bytes")
    raw = held.read(evidence, path, SOURCE_RAW_LIMIT)
    return {"recorded_pin": dict(recorded), "evidence": _source_entry(evidence, raw)}


def _sources_document(roles, historical):
    result = {"schema": SCHEMA, "kind": "sources", "roles": roles, "historical_v1_publisher": historical}
    _need(len(v1._encode(result)) <= SOURCES_LIMIT, "transition sources artifact bound exceeded")
    return result


def _verify_sources(document, generation, base):
    _fields(document, ("schema", "kind", "roles", "historical_v1_publisher"), "exact sources artifact fields required")
    _need(document["schema"] == SCHEMA and document["kind"] == "sources", "transition sources domain mismatch")
    _fields(document["roles"], ROLES, "fixed five preserved source roles required")
    total = sum(len(_source_raw(document["roles"][role], role)) for role in ROLES)
    _need(total <= SOURCE_RAW_LIMIT, "preserved source closure raw bound exceeded")
    proof = document["historical_v1_publisher"]
    if generation == 1:
        _fields(proof, ("recorded_pin", "evidence"), "first transition historical source proof required")
        _need(proof["recorded_pin"] == base["publisher_source"], "historical V1 source join changed")
        _historical_pin(proof["recorded_pin"], SOURCE_RAW_LIMIT, role="v1")
        raw = _source_raw(proof["evidence"])
        _need(proof["evidence"]["pin"]["bytes"] == proof["recorded_pin"]["bytes"]
              and proof["evidence"]["pin"]["sha256"] == proof["recorded_pin"]["sha256"]
              and hashlib.sha256(raw).hexdigest() == proof["recorded_pin"]["sha256"], "historical source bytes are not the V1 publisher")
    else:
        _need(proof is None, "later transition must reference preserved predecessor sources")
    return {role: dict(document["roles"][role]["pin"]) for role in ROLES}


def _mount_observation(value):
    """Validate the recorded kernel observation before using a mount-only delta."""
    _fields(value, ("namespace", "mountinfo_bytes", "mountinfo_sha256", "readonly_aliases"),
            "exact mount observation fields required")
    namespace = value["namespace"]
    _need(type(namespace) is list and len(namespace) == 2
          and all(type(item) is int and 0 <= item < 1 << 64 for item in namespace)
          and namespace[1] > 0, "bounded mount namespace identity required")
    _need(type(value["mountinfo_bytes"]) is int and 0 < value["mountinfo_bytes"] <= v1.SNAPSHOT_LIMIT
          and type(value["mountinfo_sha256"]) is str
          and re.fullmatch(r"[0-9a-f]{64}", value["mountinfo_sha256"]), "bounded mountinfo raw observation required")
    aliases = value["readonly_aliases"]
    _need(type(aliases) is list and len(aliases) <= v1.RECEIPT_LIMIT, "bounded readonly mount aliases required")
    order = []
    for alias in aliases:
        _fields(alias, ("mount_id", "root", "mountpoint", "mount_options"), "exact readonly mount alias fields required")
        _need(type(alias["mount_id"]) is str and re.fullmatch(r"[1-9][0-9]{0,19}", alias["mount_id"]),
              "bounded original mount ID required")
        for key in ("root", "mountpoint"):
            text = alias[key]
            _need(type(text) is str and 0 < len(text) <= v1.RECEIPT_LIMIT
                  and "\0" not in text and PurePosixPath(text).is_absolute()
                  and str(PurePosixPath(text)) == text and ".." not in PurePosixPath(text).parts,
                  "canonical absolute mount alias path required")
        options = alias["mount_options"]
        _need(type(options) is str and 0 < len(options) <= v1.RECEIPT_LIMIT,
              "bounded readonly mount options required")
        options = options.split(",")
        _need("ro" in options and "rw" not in options
              and all(option and not any(char.isspace() for char in option) for option in options),
              "readonly mount alias options required")
        order.append((alias["mountpoint"], alias["mount_id"]))
    _need(order == sorted(order) and len({row[0] for row in order}) == len(order)
          and len({row[1] for row in order}) == len(order), "canonical nonduplicate readonly mount aliases required")
    try:
        _need(len(v1._encode(value)) <= v1.RECEIPT_LIMIT, "recorded mount observation byte bound exceeded")
    except (ValueError, UnicodeError) as error:
        raise TransitionBlocked("invalid recorded mount observation encoding") from error


def _reason(previous, now, previous_sources, current_sources, generation, previous_mounts, current_mounts):
    _life(previous)
    _life(now)
    _need(previous["host"] == now["host"] and previous["platform"] == now["platform"], "transition host/platform changed")
    boot_changed = previous["boot"] != now["boot"]
    source_changed = generation == 1 or previous_sources != current_sources
    if boot_changed or source_changed:
        return "both" if boot_changed and source_changed else "reboot_reobserve" if boot_changed else "publisher_upgrade"
    _mount_observation(previous_mounts)
    _mount_observation(current_mounts)
    _need(previous_mounts != current_mounts, "transition has no supported explicit reason")
    return "mount_reobserve"


def _stem(key, digest):
    _need(re.fullmatch(r"[0-9a-f]{64}", key) and re.fullmatch(r"[0-9a-f]{64}", digest), "exact original key/raw digest required")
    return key + "-" + digest


def _prefix(stem, generation, attempt_id):
    _need(type(generation) is int and 1 <= generation <= MAX_TRANSITIONS
          and isinstance(attempt_id, str) and re.fullmatch(r"[0-9a-f]{32}", attempt_id), "bounded transition generation/attempt required")
    return stem + ".transition-" + f"{generation:02d}" + "-" + attempt_id


def _layout(held, catalog, allow_partial=None):
    root = v1.registry_path(catalog)
    try:
        info = root.lstat()
    except FileNotFoundError as error:
        raise TransitionBlocked("original V1 anchor registry required; no new catalog") from error
    _need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
          and stat.S_IMODE(info.st_mode) & 0o077 == 0, "private original custody registry required")
    fd = held.directory(root)
    _need(v1._did(os.fstat(fd)) == v1._did(info), "transition registry FD/name changed")
    registry, groups, total = {}, {}, 0
    with os.scandir(fd) as children:
        for child in children:
            held.tick()
            old, new = v1._NAME.fullmatch(child.name), _TRANSITION.fullmatch(child.name)
            _need(old is not None or new is not None, "unknown transition registry artifact")
            match = old if old is not None else new
            stem = _stem(match[1], match[2])
            group = groups.setdefault(stem, {"base": set(), "transitions": {}})
            kind = match[3] if old is not None else match[5]
            maximum = v1.RECEIPT_LIMIT if kind in {"original", "receipt"} else SOURCES_LIMIT if kind == "sources" else v1.SNAPSHOT_LIMIT
            item = os.stat(child.name, dir_fd=fd, follow_symlinks=False)
            _need(stat.S_ISREG(item.st_mode) and item.st_uid == os.geteuid() and item.st_nlink == 1
                  and 0 < item.st_size <= maximum, "transition artifact type/owner/size unsupported; preserve partial")
            registry[child.name] = v1._stamp(item)
            total += item.st_size
            if old is not None:
                group["base"].add(kind)
            else:
                identity = (int(match[3]), match[4])
                group["transitions"].setdefault(identity, set()).add(kind)
            _need(len(groups) <= MAX_RUNS and len(registry) <= MAX_RUNS * (3 + 3 * MAX_TRANSITIONS)
                  and total <= REGISTRY_LIMIT, "transition registry bound exceeded")
    for stem, group in groups.items():
        _need(group["base"] == {"original", "snapshot", "receipt"}, "partial or orphan V1 group remains blocked")
        generations = {}
        completed = []
        for identity, kinds in group["transitions"].items():
            generation, attempt = identity
            _need(generation not in generations, "forked or duplicate transition generation")
            generations[generation] = attempt
            if kinds == {"sources", "snapshot", "receipt"}:
                completed.append(generation)
            else:
                _need(allow_partial == (stem, generation, attempt) and kinds <= {"sources", "snapshot"},
                      "partial transition requires same explicit attempt; admission blocked")
        _need(sorted(completed) == list(range(1, len(completed) + 1)), "missing/orphan transition generation")
        pending = set(generations) - set(completed)
        _need(not pending or pending == {len(completed) + 1}, "partial transition cannot skip generation")
        group["completed"] = sorted(completed)
        group["attempts"] = generations
    return registry, total, groups


def _read_json(held, pin, path, maximum):
    return v1._decode(held.read(pin, path, maximum))


def _snapshot_document(stem, generation, attempt, tree):
    return {"schema": SCHEMA, "kind": "snapshot", "stem": stem, "generation": generation, "attempt_id": attempt, "tree": tree}


def _transition_tree(document, stem, generation, attempt):
    _fields(document, ("schema", "kind", "stem", "generation", "attempt_id", "tree"), "exact transition snapshot fields required")
    _need(document["schema"] == SCHEMA and document["kind"] == "snapshot" and document["stem"] == stem
          and type(document["generation"]) is int and document["generation"] == generation
          and document["attempt_id"] == attempt, "transition snapshot join changed")
    return document["tree"]


class TransitionValidation(v1.QuarantineValidation):
    """One held view for successor admission, mutation and sampled accounting."""
    def __init__(self, held, catalog, guard, now, registry, total, records, observations, details, allow_partial):
        super().__init__(held, catalog, guard, now, registry, total, records, observations)
        self.details, self.allow_partial = details, allow_partial
        self.current_source_pins, self.require_current = None, False

    def admission_ready(self):
        if not self.details or self.current_source_pins is None:
            return False
        for detail in self.details.values():
            tip = detail["tip"]
            if (detail["generation"] == 0 or detail["source_pins"] != self.current_source_pins
                    or any(tip["observed_lifetime"][key] != self.now[key] for key in ("host", "boot", "platform"))
                    or tip["mount_observation"] != detail["observation"].mounts):
                return False
        return True

    def check(self):
        self.guard.assert_held()
        current = lifetime()
        _need(all(current[key] == self.now[key] for key in ("host", "boot", "platform")), "current transition observation boot changed")
        self.held.check()
        for observation in self.observed:
            observation.check()
        registry, total, _ = _layout(self.held, self.catalog, self.allow_partial)
        _need((registry, total) == (self.registry, self.registry_bytes), "transition registry changed")
        if self.require_current:
            _need(self.admission_ready(), "selected transition tip is stale; explicit maintenance required")
        self.guard.assert_held()


def _history(held, catalog, rows, tip_pins, guard, allow_partial=None):
    catalog = checked(Path(catalog))
    _need(Path(guard.path) == catalog / "catalog.lock", "original shared catalog gate required")
    guard.assert_held()
    root = v1.registry_path(catalog)
    for _, _, target in rows:
        _need(not (target.is_relative_to(root) or root.is_relative_to(target)), "catalog row overlaps transition registry")
    _need(isinstance(tip_pins, (tuple, list)) and len(tip_pins) <= MAX_RUNS, "bounded explicit current tip pins required")
    now = lifetime()
    registry, total, groups = _layout(held, catalog, allow_partial)
    selected = {}
    for pin in tip_pins:
        _need(type(pin) is dict and isinstance(pin.get("path"), str), "exact selected tip raw pin required")
        name = Path(pin["path"]).name
        _need(name not in selected, "duplicate current tip selection")
        v1._pin_shape(pin, root / name, v1.RECEIPT_LIMIT)
        selected[name] = dict(pin)
    expected = set()
    for stem, group in groups.items():
        generation = len(group["completed"])
        prefix = stem if generation == 0 else _prefix(stem, generation, group["attempts"][generation])
        expected.add(prefix + ".receipt.json")
    _need(set(selected) == expected, "every original row requires its explicitly selected current tip; no rollback")
    records, observations, details, identities = [], [], {}, set()
    old_rows = {row["key"]: row for _, row, _ in rows}
    for stem, group in groups.items():
        held.tick()
        last_generation = len(group["completed"])
        last_prefix = stem if last_generation == 0 else _prefix(stem, last_generation, group["attempts"][last_generation])
        tip_pin = selected[last_prefix + ".receipt.json"]
        pin, backwards = tip_pin, {}
        for generation in range(last_generation, 0, -1):
            attempt = group["attempts"][generation]
            prefix = _prefix(stem, generation, attempt)
            value = _read_json(held, pin, root / (prefix + ".receipt.json"), v1.RECEIPT_LIMIT)
            _fields(value, _TRANSITION_FIELDS, "exact transition receipt fields required")
            _need(value["schema"] == SCHEMA and value["kind"] == "receipt"
                  and value["classification"] == "interrupted_unbound_preservation_transition"
                  and type(value["generation"]) is int and value["generation"] == generation
                  and value["attempt_id"] == attempt and value["key"] == stem[:64], "transition receipt generation/domain changed")
            _life(value["observed_lifetime"])
            _authority(value)
            previous_prefix = stem if generation == 1 else _prefix(stem, generation - 1, group["attempts"][generation - 1])
            v1._pin_shape(value["predecessor"], root / (previous_prefix + ".receipt.json"), v1.RECEIPT_LIMIT)
            backwards[generation] = (dict(pin), value)
            pin = value["predecessor"]
        base_pin = dict(pin)
        base = _read_json(held, base_pin, root / (stem + ".receipt.json"), v1.RECEIPT_LIMIT)
        _fields(base, _V1_FIELDS, "exact historical V1 receipt fields required")
        _need(base["schema"] == v1.SCHEMA and base["kind"] == "receipt"
              and base["classification"] == "interrupted_unbound_custody_quarantined", "historical V1 receipt domain changed")
        _authority(base)
        _life(base["observed_lifetime"])
        _historical_pin(base["publisher_source"], SOURCE_RAW_LIMIT, role="v1")
        _need(all(base["observed_lifetime"][key] == now[key] for key in ("host", "platform")), "historical receipt host/platform changed")
        row, original_raw, target = v1._original(held, catalog, base["original_catalog_row"], now)
        key = row["key"]
        _need(stem == _stem(key, hashlib.sha256(original_raw).hexdigest()) and key not in details
              and old_rows.get(key) == row and base["output"] == str(target)
              and base["original_catalog_stat"] == v1._stamp(Path(base["original_catalog_row"]["path"]).lstat()),
              "original immutable row/raw/metadata join changed")
        copy_raw = held.read(base["original_copy"], root / (stem + ".original.json"), v1.RECEIPT_LIMIT)
        _need(copy_raw == original_raw, "historical original raw copy changed")
        tree = _read_json(held, base["snapshot"], root / (stem + ".snapshot.json"), v1.SNAPSHOT_LIMIT)
        observation = v1._Observed(held, target)
        _need(v1._encode(observation.snapshot) == v1._encode(tree)
              and observation.parent_identity == base["current_parent_identity"]
              and observation.identity == base["current_postboot_observed_identity"], "preservation subtree/parent identity changed; separate review required")
        identity = tuple(observation.identity[:2])
        _need(identity not in identities, "multiple retained transition root aliases")
        identities.add(identity)
        for _, other, other_target in rows:
            if other["key"] != key and other_target.exists():
                _need(tuple(v1._stamp(other_target.lstat())[:2]) != identity, "transition target aliases another output")
        charge = base["quarantine_charge_bytes"]
        _need(type(charge) is int and charge == row["reserved_bytes"] and tree["regular_bytes"] <= charge,
              "historical V1 reservation charge changed")
        previous, previous_pin, source_pins = base, base_pin, None
        for generation in range(1, last_generation + 1):
            current_pin, value = backwards[generation]
            attempt = group["attempts"][generation]
            prefix = _prefix(stem, generation, attempt)
            sources = _read_json(held, value["sources"], root / (prefix + ".sources.json"), SOURCES_LIMIT)
            current_sources = _verify_sources(sources, generation, base)
            snapshot_document = _read_json(held, value["snapshot"], root / (prefix + ".snapshot.json"), v1.SNAPSHOT_LIMIT)
            next_tree = _transition_tree(snapshot_document, stem, generation, attempt)
            _need(value["original_catalog_row"] == base["original_catalog_row"]
                  and value["original_catalog_stat"] == base["original_catalog_stat"]
                  and value["original_v1_receipt"] == base_pin and value["predecessor"] == previous_pin
                  and value["output"] == str(target)
                  and value["current_parent_identity"] == previous["current_parent_identity"]
                  and value["current_postboot_observed_identity"] == previous["current_postboot_observed_identity"]
                  and v1._encode(next_tree) == v1._encode(tree), "transition immutable preservation comparison failed")
            _need(value["reason"] == _reason(previous["observed_lifetime"], value["observed_lifetime"],
                                              source_pins, current_sources, generation,
                                              previous["mount_observation"], value["mount_observation"]), "explicit transition reason mismatch")
            next_charge = value["quarantine_charge_bytes"]
            _need(type(next_charge) is int and next_charge == max(row["reserved_bytes"], charge)
                  and next_tree["regular_bytes"] <= next_charge, "transition charge reduced or exceeded")
            charge, tree, previous, previous_pin, source_pins = next_charge, next_tree, value, current_pin, current_sources
        records.append(v1.InterruptedUnboundCustody(key, target, charge, dict(base["original_catalog_row"])))
        observations.append(observation)
        details[key] = {"stem": stem, "row": row, "base": base, "base_pin": base_pin,
            "tip": previous, "tip_pin": previous_pin, "generation": last_generation,
            "source_pins": source_pins, "tree": tree, "observation": observation, "charge": charge}
    result = TransitionValidation(held, catalog, guard, now, registry, total, records, observations, details, allow_partial)
    result.usage(rows)
    result.check()
    return result


def validate_transitions(catalog, rows, tip_pins, guard, *, current_source_pins=None, maximum_seconds=30):
    """Current-tip admission only. Historical/stale tips never become capabilities."""
    held = v1._Held(maximum_seconds)
    try:
        result = _history(held, catalog, rows, tip_pins, guard)
        _need(result.details and all(detail["generation"] > 0 for detail in result.details.values()),
              "V1 tip needs an explicit first transition before successor admission")
        expected = current_source_pins
        if expected is None:
            expected = next(iter(result.details.values()))["source_pins"]
        roles = _current_sources(held, expected)
        result.current_source_pins = {role: dict(roles[role]["pin"]) for role in ROLES}
        result.require_current = True
        result.check()
        return result
    except BaseException as error:
        held.close(error)
        raise



def _sync_existing(held, pin, path, content, limit, directory_fd):
    count = len(held.blobs)
    _need(held.read(pin, path, limit) == content, "resumed artifact content changed")
    _need(len(held.blobs) == count + 1 and held.blobs[-1][0] == Path(path)
          and type(held.blobs[-1][1]) is int, "original held artifact read-FD required")
    held.check()  # Recheck the descriptor/name/raw join before its durability fence.
    os.fsync(held.blobs[-1][1])
    os.fsync(directory_fd)
    held.check()


def publish_custody_transition(catalog, original_row_pin, predecessor_pin, current_tip_pins, current_source_pins, *,
                               reason, attempt_id, historical_source_snapshot=None, existing_artifact_pins=(),
                               root_exclusive_writer=False, apply=False, maximum_seconds=30):
    """Explicit maintenance; one selected predecessor, never an automatic refresh.

    Complete prior generations stay immutable. Only exact existing sources/tree
    bytes of this same attempt may be resumed. A partial receipt, foreign attempt
    or mismatching artifact remains blocked; nothing is removed or overwritten.
    Per-key publication is not whole-catalog admission readiness.
    """
    from managed_outputs import _catalog_rows
    _need(root_exclusive_writer is True, "explicit exclusive preservation writer required")
    _need(isinstance(reason, str) and reason in {"reboot_reobserve", "publisher_upgrade", "both", "mount_reobserve"}, "explicit supported transition reason required")
    catalog = checked(Path(catalog))
    root = v1.registry_path(catalog)
    _need(type(predecessor_pin) is dict and isinstance(predecessor_pin.get("path"), str), "explicit predecessor raw pin required")
    predecessor_name = Path(predecessor_pin["path"]).name
    old, new = v1._NAME.fullmatch(predecessor_name), _TRANSITION.fullmatch(predecessor_name)
    _need((old is not None and old[3] == "receipt") or (new is not None and new[5] == "receipt"), "predecessor must be a final marker")
    match = old if old is not None else new
    stem = _stem(match[1], match[2])
    generation = 1 if old is not None else int(new[3]) + 1
    prefix = _prefix(stem, generation, attempt_id)
    _need(isinstance(existing_artifact_pins, (tuple, list)) and len(existing_artifact_pins) <= 2,
          "bounded same-attempt partial artifact pins required")
    stage, held = "not_started", v1._Held(maximum_seconds)
    try:
        current_roles = _current_sources(held, current_source_pins)
        current_pins = {role: dict(current_roles[role]["pin"]) for role in ROLES}
        with v1.original_custody_gate(catalog) as guard:
            rows = _catalog_rows(catalog)
            view = _history(held, catalog, rows, current_tip_pins, guard, (stem, generation, attempt_id))
            key = stem[:64]
            _need(key in view.details, "explicit original row is not a known V1 anchor")
            detail = view.details[key]
            _need(detail["stem"] == stem and detail["tip_pin"] == predecessor_pin
                  and detail["row"]["key"] == key and detail["generation"] + 1 == generation
                  and detail["base"]["original_catalog_row"] == original_row_pin,
                  "predecessor/original row is not the explicitly selected current tip")
            _need(reason == _reason(detail["tip"]["observed_lifetime"], view.now,
                                     detail["source_pins"], current_pins, generation,
                                     detail["tip"]["mount_observation"], detail["observation"].mounts), "explicit transition reason mismatch")
            if generation == 1:
                historical = _historical_first_source(held, detail["base"], historical_source_snapshot)
            else:
                _need(historical_source_snapshot is None, "later generations use preserved predecessor sources only")
                historical = None
            sources_raw = v1._encode(_sources_document(current_roles, historical))
            snapshot_raw = v1._encode(_snapshot_document(stem, generation, attempt_id, detail["observation"].snapshot))
            _need(len(snapshot_raw) <= v1.SNAPSHOT_LIMIT, "transition snapshot wrapper bound exceeded")
            sources_pin = v1._pin(root / (prefix + ".sources.json"), sources_raw)
            snapshot_pin = v1._pin(root / (prefix + ".snapshot.json"), snapshot_raw)
            receipt = {"schema": SCHEMA, "kind": "receipt", "classification": "interrupted_unbound_preservation_transition",
                "key": key, "original_catalog_row": dict(original_row_pin),
                "original_catalog_stat": detail["base"]["original_catalog_stat"], "original_v1_receipt": detail["base_pin"],
                "predecessor": dict(predecessor_pin), "generation": generation, "attempt_id": attempt_id, "reason": reason,
                "output": str(detail["observation"].target), "observed_lifetime": dict(view.now),
                "current_parent_identity": detail["observation"].parent_identity,
                "current_postboot_observed_identity": detail["observation"].identity,
                "mount_observation": detail["observation"].mounts, "snapshot": snapshot_pin, "sources": sources_pin,
                "quarantine_charge_bytes": max(detail["row"]["reserved_bytes"], detail["charge"]),
                "root_exclusive_writer_acknowledged": True, "original_generation_identity_known": False,
                "original_generation_custody_reissued": False, "physical_completion": "unknown_unchanged",
                "numeric_success_issued": False, "exclusion_scope": "output_and_all_ancestors_descendants"}
            receipt_raw = v1._encode(receipt)
            _need(len(receipt_raw) <= v1.RECEIPT_LIMIT, "transition receipt bound exceeded")
            contents = {"sources": (sources_raw, SOURCES_LIMIT), "snapshot": (snapshot_raw, v1.SNAPSHOT_LIMIT),
                        "receipt": (receipt_raw, v1.RECEIPT_LIMIT)}
            existing = {}
            for pin in existing_artifact_pins:
                _need(type(pin) is dict and isinstance(pin.get("path"), str), "exact same-attempt artifact raw pin required")
                name = Path(pin["path"]).name
                _need(name not in existing and name in {prefix + ".sources.json", prefix + ".snapshot.json"},
                      "duplicate/foreign partial artifact selection")
                existing[name] = pin
            actual = {name for name in view.registry if name.startswith(prefix + ".")}
            _need(set(existing) == actual, "every partial artifact needs an explicit same-attempt raw pin")
            for name, pin in existing.items():
                kind = name.rsplit(".", 2)[1]
                content, limit = contents[kind]
                raw = held.read(pin, root / name, limit)
                _need(raw == content, "partial artifact content differs; preserve blocked attempt")
            additional = sum(len(content) for kind, (content, _) in contents.items() if prefix + "." + kind + ".json" not in existing)
            _need(view.registry_bytes + additional <= REGISTRY_LIMIT, "transition registry metadata bound exceeded")
            warn_output_threshold(view.usage(rows) + additional, scope="custody-transition")
            view.check()
            if not apply:
                return {"applied": False, "receipt_issued": False, "generation": generation,
                        "original_row_unchanged": True, "admission_ready": False, "admission_scope": ADMISSION_SCOPE,
                        "quarantine_charge_bytes": receipt["quarantine_charge_bytes"],
                        "registry_metadata_bytes_after": view.registry_bytes + additional}
            directory_fd = held.directory(root)
            expected_registry = dict(view.registry)
            for kind in ("sources", "snapshot", "receipt"):
                content, limit = contents[kind]
                name = prefix + "." + kind + ".json"
                view.registry = dict(expected_registry)
                view.registry_bytes = sum(stamp[6] for stamp in expected_registry.values())
                view.check()
                guard.assert_held()
                _need(v1._did(os.fstat(directory_fd)) == v1._did(checked(root).lstat()), "transition registry publication FD/name changed")
                if name in existing:
                    stage = kind + "_resume_fsync_pending"
                    _sync_existing(held, existing[name], root / name, content, limit, directory_fd)
                    stage = kind + "_resumed_durable"
                else:
                    stage = kind + "_publication_pending"
                    v1._exclusive(held, directory_fd, name, content, limit)
                    stage = kind + "_durable"
                    held.read(v1._pin(root / name, content), root / name, limit)
                    expected_registry[name] = v1._stamp((root / name).lstat())
            result_pin = v1._pin(root / (prefix + ".receipt.json"), receipt_raw)
            next_tips = [result_pin if pin == predecessor_pin else dict(pin) for pin in current_tip_pins]
            final = _history(held, catalog, _catalog_rows(catalog), next_tips, guard)
            final.current_source_pins = current_pins
            final.check()
            return {"applied": True, "receipt": result_pin, "generation": generation,
                    "original_row_unchanged": True, "admission_ready": final.admission_ready(), "admission_scope": ADMISSION_SCOPE,
                    "original_generation_custody_reissued": False, "physical_completion": "unknown_unchanged"}
    except BaseException as error:
        if hasattr(error, "add_note"):
            error.add_note("transition publication stage=" + stage + "; all evidence retained, no rollback/delete")
        raise
    finally:
        held.close(sys.exc_info()[1])
