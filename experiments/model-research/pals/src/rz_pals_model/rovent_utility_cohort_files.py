"""Pinned utility-cohort files, without recovering original execution owners.

Publication coverage and original file bytes are checked in one shared bounded
reader. Payloads, including weights, are opaque bytes here. The newer ACK's
trial acknowledgement bodies remain explicitly unverified until their separate
typed consumer is registered. No Rules, native, physical, target or learner
authority is reconstructed by this historical data owner.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import re

from .generation_data import fields, sha, uint
from . import rovent_corpus as pc
from .rovent_training_bootstrap import PinnedReadBudget


PUBLICATION_DOMAIN = "rz-rovent-utility-cohort-publication/1"
ACK_DOMAIN = "rz-rovent-utility-cohort-durable-ack/1"
OBSERVATION_DOMAIN = "rz-rovent-utility-cohort-files-observation/1"
COST_SCOPE = "cohort_route_with_bounded_materialization_and_physical_pin_zero_excluding_durable_dataset_io_v1"
PUBLICATION_FILE = "utility-cohort-publication.json"
ACK_FILE = "utility-cohort-durable-ack.json"
MAX_METADATA_BYTES = 64 << 10
MAX_PAYLOAD_BYTES = 256 << 20
MAX_MATERIALIZATION_BYTES = 32 << 20
FIXED_FILES = (
    ("cohort_profile", "cohort-profile.json", 64 << 10),
    ("frozen_context", "frozen-context.json", 32 << 20),
    ("source", "source.json", 1 << 20),
    ("input_lock", "input-lock.json", 4 << 20),
    ("nnuec_weights", "nnuec-weights.json", 32 << 20),
    ("nnuec_provenance", "nnuec-provenance.json", 1 << 20),
)
PUBLICATION_FIELDS = ("schema", "cohort_id", "cost_scope", *(item[0] for item in FIXED_FILES),
    "materializations", "trials", "pairs", "trial_count", "pair_count", "total_payload_bytes")
ACK_FIELDS = ("schema", "cohort_id", "cost_scope", "publication", "collection_io", "trial_acknowledgements")
_IO_FIELDS = ("clock_epoch", "started_tick_ns", "finished_tick_ns", "elapsed_ns",
    "written_bytes", "readback_bytes", "synced_files", "synced_directories")


def _relative(value):
    if (type(value) is not str or not value or len(value.encode("utf-8")) > 4096
            or "\\" in value or ":" in value or value.startswith("/")
            or any(part in ("", ".", "..") for part in value.split("/"))
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)):
        raise ValueError("normalized relative utility cohort path required")
    return value


def _pin(value, maximum):
    fields(value, ("path", "sha256", "bytes"), "utility cohort original file pin")
    _relative(value["path"])
    sha(value["sha256"])
    if not uint(value["bytes"], maximum):
        raise ValueError("nonempty bounded utility cohort raw file required")
    return value


def _ordinal(path, directory):
    match = re.fullmatch(re.escape(directory) + r"/([0-9]{6})\.json", path)
    if match is None:
        raise ValueError("utility cohort normalized six-digit ordinal path required")
    return int(match.group(1))


def _payloads(publication):
    return tuple(publication[name] for name, _, _ in FIXED_FILES) + tuple(
        pin for category in ("materializations", "trials", "pairs") for pin in publication[category])


def validate_utility_cohort_publication(value):
    """Validate the closed publication's declared coverage, never its targets."""
    fields(value, PUBLICATION_FIELDS, "utility cohort publication")
    if value["schema"] != PUBLICATION_DOMAIN or value["cost_scope"] != COST_SCOPE:
        raise ValueError("explicit utility cohort publication domain/cost scope required")
    sha(value["cohort_id"])
    for category in ("materializations", "trials", "pairs"):
        if type(value[category]) is not list:
            raise ValueError("original utility cohort ordered file lists required")
    trials, pairs = value["trials"], value["pairs"]
    if (not 1 <= len(trials) <= 3 or len(value["materializations"]) > len(trials)
            or len(pairs) > 3 or len(pairs) > len(trials) * (len(trials) - 1) // 2
            or uint(value["trial_count"], 3) != len(trials)
            or uint(value["pair_count"], 3) != len(pairs)):
        raise ValueError("utility cohort exact bounded trial/pair coverage required")
    paths, total = set(), 0

    def add(pin, maximum):
        nonlocal total
        _pin(pin, maximum)
        identity = pin["path"].casefold()
        if identity in paths:
            raise ValueError("utility cohort duplicate/case-alias payload path")
        paths.add(identity)
        total = uint(total + pin["bytes"], MAX_PAYLOAD_BYTES)

    for name, filename, maximum in FIXED_FILES:
        pin = _pin(value[name], maximum)
        if pin["path"] != filename:
            raise ValueError("utility cohort fixed original file path differs")
        add(pin, maximum)
    previous = -1
    for pin in value["materializations"]:
        _pin(pin, MAX_MATERIALIZATION_BYTES)
        ordinal = _ordinal(pin["path"], "materializations")
        if not previous < ordinal < len(trials):
            raise ValueError("utility cohort materialization ordinal/order differs")
        previous = ordinal
        add(pin, MAX_MATERIALIZATION_BYTES)
    for directory, entries, maximum in (("trials", trials, MAX_MATERIALIZATION_BYTES), ("pairs", pairs, 64 << 10)):
        for index, pin in enumerate(entries):
            _pin(pin, maximum)
            if _ordinal(pin["path"], directory) != index:
                raise ValueError("utility cohort original trial/pair order differs")
            add(pin, maximum)
    if uint(value["total_payload_bytes"], MAX_PAYLOAD_BYTES) != total:
        raise ValueError("utility cohort exact unique payload byte total differs")
    return copy.deepcopy(value)


def _plain_json(value, *, depth=0, allowance=None):
    """Bound opaque ACK operands without assigning an unregistered body type."""
    if allowance is None:
        allowance = [MAX_METADATA_BYTES, MAX_METADATA_BYTES]
    allowance[0] -= 1
    allowance[1] -= 1
    if depth > 64 or min(allowance) < 0:
        raise ValueError("bounded plain utility trial acknowledgement data required")
    if type(value) is str:
        allowance[1] -= len(value.encode("utf-8"))
    elif type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise ValueError("plain utility acknowledgement JSON keys required")
            _plain_json(key, depth=depth + 1, allowance=allowance)
            _plain_json(child, depth=depth + 1, allowance=allowance)
    elif type(value) is list:
        for child in value:
            _plain_json(child, depth=depth + 1, allowance=allowance)
    elif type(value) not in (int, float, bool, type(None)) or type(value) is float and not math.isfinite(value):
        raise ValueError("plain finite utility trial acknowledgement JSON required")
    if min(allowance) < 0:
        raise ValueError("utility trial acknowledgement data exceeds metadata bounds")


def validate_utility_cohort_ack(value, *, publication, publication_file):
    """Validate ACK6's closed envelope and final-pass I/O; bodies stay unknown.

    The final write interval excludes trial/materialization files already
    published between trials. Its readback and re-sync cover every payload.
    The original inter-trial intervals remain opaque acknowledgement operands.
    """
    publication = validate_utility_cohort_publication(publication)
    _pin(publication_file, MAX_METADATA_BYTES)
    if publication_file["path"] != PUBLICATION_FILE:
        raise ValueError("original utility cohort publication filename required")
    fields(value, ACK_FIELDS, "utility cohort durable ACK6")
    if (value["schema"] != ACK_DOMAIN or value["cohort_id"] != publication["cohort_id"]
            or value["cost_scope"] != publication["cost_scope"]
            or value["publication"] != publication_file):
        raise ValueError("utility cohort durable ACK original publication binding differs")
    # Validate the nested pin's exact types too; equality would accept True=1.
    _pin(value["publication"], MAX_METADATA_BYTES)
    io = fields(value["collection_io"], _IO_FIELDS, "utility cohort final collection I/O")
    for observed in io.values():
        uint(observed)
    expected_read = uint(publication["total_payload_bytes"] + publication_file["bytes"])
    expected_write = uint(sum(publication[name]["bytes"] for name, _, _ in FIXED_FILES)
        + sum(pin["bytes"] for pin in publication["pairs"]) + publication_file["bytes"])
    if (io["finished_tick_ns"] < io["started_tick_ns"]
            or io["finished_tick_ns"] - io["started_tick_ns"] != io["elapsed_ns"]
            or io["written_bytes"] != expected_write or io["readback_bytes"] != expected_read
            or io["synced_files"] != len(_payloads(publication)) + 1 or not io["synced_directories"]):
        raise ValueError("utility cohort original final collection I/O receipt differs")
    _plain_json(value["trial_acknowledgements"])
    return copy.deepcopy(value)


def _absolute(value):
    if (type(value) is not str or not value or len(value.encode("utf-8")) > 4096
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)):
        raise ValueError("bounded absolute utility cohort path required")
    result = Path(value)
    if not result.is_absolute() or ".." in result.parts:
        raise ValueError("bounded absolute utility cohort path required")
    return result


def _reference(value):
    fields(value, ("root", "publication", "durable_ack"), "utility cohort exact file references")
    root = _absolute(value["root"])
    if not root.is_dir() or root.resolve(strict=True) != root:
        raise ValueError("original plain utility cohort root directory required")
    pins = {}
    for field, filename in (("publication", PUBLICATION_FILE), ("durable_ack", ACK_FILE)):
        pin = fields(value[field], ("path", "sha256", "bytes"), "absolute utility cohort metadata pin")
        path = _absolute(pin["path"])
        if path.parent != root or path.name != filename:
            raise ValueError("utility cohort metadata must name its exact root file")
        sha(pin["sha256"])
        if not uint(pin["bytes"], MAX_METADATA_BYTES):
            raise ValueError("nonempty bounded utility cohort metadata required")
        pins[filename] = {"path": filename, "sha256": pin["sha256"], "bytes": pin["bytes"]}
    return root, pins


class HistoricalUtilityCohortFiles:
    """Detached file integrity observation with no live or known-target grant."""
    __slots__ = ("_publication", "_ack", "_raw", "_pins", "_paths", "_observation")

    @property
    def payload_paths(self):
        return self._paths

    def publication(self):
        return copy.deepcopy(self._publication)

    def durable_ack(self):
        return copy.deepcopy(self._ack)

    def observation(self):
        return copy.deepcopy(self._observation)

    def file_pin(self, relative_path):
        _relative(relative_path)
        if relative_path not in self._pins:
            raise ValueError("utility cohort file was not in the original pinned inventory")
        return copy.deepcopy(self._pins[relative_path])

    def raw_file(self, relative_path):
        pin = self.file_pin(relative_path)
        raw = self._raw[relative_path]
        if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
            raise ValueError("retained utility cohort original bytes changed")
        return raw


def read_utility_cohort_files(reference, *, read_accounting):
    """Read only caller-pinned files, retaining the exact original raw bytes."""
    if type(read_accounting) is not PinnedReadBudget or read_accounting.strict_growth_credit is not True:
        raise ValueError("one original strict PinnedReadBudget required for utility cohort files")
    if read_accounting.observation()["unknown"]:
        raise ValueError("uncertain utility cohort shared read budget cannot issue an observation")
    root, pins = _reference(reference)
    raw = {}
    for filename in (PUBLICATION_FILE, ACK_FILE):
        pin = pins[filename]
        raw[filename] = pc._read_pinned_bytes(root / filename, pin["sha256"], pin["bytes"],
            MAX_METADATA_BYTES, read_accounting=read_accounting)
    publication = validate_utility_cohort_publication(pc._decode_json(raw[PUBLICATION_FILE]))
    ack = validate_utility_cohort_ack(pc._decode_json(raw[ACK_FILE]), publication=publication,
        publication_file=pins[PUBLICATION_FILE])
    for pin in _payloads(publication):
        path = pin["path"]
        # Closed publication validation already proved each per-file cap; the
        # exact pin size is the reader allocation ceiling, plus its EOF probe.
        raw[path] = pc._read_pinned_bytes(root.joinpath(*path.split("/")), pin["sha256"], pin["bytes"],
            pin["bytes"], read_accounting=read_accounting)
        pins[path] = copy.deepcopy(pin)
    observation = {"schema": OBSERVATION_DOMAIN, "scope": "publication_and_raw_file_pins_only",
        "cohort_id": publication["cohort_id"], "publication_sha256": pins[PUBLICATION_FILE]["sha256"],
        "durable_ack_sha256": pins[ACK_FILE]["sha256"], "payload_count": len(_payloads(publication)),
        "trial_count": publication["trial_count"], "pair_count": publication["pair_count"],
        "total_payload_bytes": publication["total_payload_bytes"], "known_pair_count": None,
        "trial_acknowledgements_body_verified": False, "payload_bodies_verified": False,
        "native_authority_verified": False, "rules_verified": False, "physical_completion_verified": False,
        "training_target_authorized": False, "training_authority_verified": False, "execution_authorized": False,
        "read_accounting": copy.deepcopy(read_accounting.observation())}
    owner = HistoricalUtilityCohortFiles()
    owner._publication, owner._ack, owner._raw, owner._pins = publication, ack, raw, pins
    owner._paths = tuple(pin["path"] for pin in _payloads(publication))
    owner._observation = observation
    return owner
