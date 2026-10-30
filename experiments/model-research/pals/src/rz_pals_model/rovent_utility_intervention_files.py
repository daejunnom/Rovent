"""Original /2 publication and separate ACK bytes through bounded file reads.

This is a detached data owner. The original no-follow FD reader checks each
file before decoding; retained immutable bytes do not restore its live native
owner, current ACK, physical completion, Rules or training authority. The /1
publication reader and its accepted domains remain unchanged.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import re

from .generation_data import fields, sha, uint
from . import rovent_corpus as corpus
from .rovent_training_bootstrap import PinnedReadBudget
from .rovent_utility_intervention import COST_SCOPE


PUBLICATION_DOMAIN = "rz-rovent-utility-cohort-publication/2"
ACK_DOMAIN = "rz-rovent-utility-cohort-durable-ack/2"
OBSERVATION_DOMAIN = "rz-rovent-utility-intervention-files-observation/1"
PUBLICATION_FILE = "utility-cohort-publication.json"
ACK_FILE = "utility-cohort-durable-ack.json"
MAX_METADATA_BYTES = 64 << 10
MAX_PAYLOAD_BYTES = 256 << 20
MAX_MATERIALIZATION_BYTES = 32 << 20
FIXED_FILES = (
    ("cohort_profile", "cohort-profile.json", 64 << 10),
    ("neutral_target", "neutral-target.json", 32 << 20),
    ("source", "source.json", 1 << 20),
    ("input_lock", "input-lock.json", 4 << 20),
    ("nnuec_weights", "nnuec-weights.json", 32 << 20),
    ("nnuec_provenance", "nnuec-provenance.json", 1 << 20),
)
PUBLICATION_FIELDS = ("schema", "cohort_id", "cost_scope", *(item[0] for item in FIXED_FILES),
    "materializations", "trials", "pairs", "trial_count", "pair_count", "total_payload_bytes")
ACK_FIELDS = ("schema", "cohort_id", "cost_scope", "publication", "collection_io", "trial_acknowledgements")
IO_FIELDS = ("clock_epoch", "started_tick_ns", "finished_tick_ns", "elapsed_ns",
    "written_bytes", "readback_bytes", "synced_files", "synced_directories")


def relative(value):
    if (type(value) is not str or not value or len(value.encode("utf-8")) > 4096
            or "\\" in value or ":" in value or value.startswith("/")
            or any(part in ("", ".", "..") for part in value.split("/"))
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)):
        raise ValueError("normalized relative intervention file path required")
    return value


def file_pin(value, maximum):
    fields(value, ("path", "sha256", "bytes"), "original intervention file pin")
    relative(value["path"])
    sha(value["sha256"])
    if not uint(value["bytes"], maximum):
        raise ValueError("nonempty bounded original intervention file required")
    return value


def ordinal(path, directory):
    found = re.fullmatch(re.escape(directory) + r"/([0-9]{6})\.json", path)
    if found is None:
        raise ValueError("six-digit original intervention payload ordinal required")
    return int(found.group(1))


def payload_pins(publication):
    return tuple(publication[name] for name, _, _ in FIXED_FILES) + tuple(
        pin for group in ("materializations", "trials", "pairs") for pin in publication[group])


def validate_intervention_publication(value):
    fields(value, PUBLICATION_FIELDS, "intervention publication/2")
    if value["schema"] != PUBLICATION_DOMAIN or value["cost_scope"] != COST_SCOPE:
        raise ValueError("explicit intervention publication/2 and original cost scope required")
    sha(value["cohort_id"])
    for group in ("materializations", "trials", "pairs"):
        if type(value[group]) is not list:
            raise ValueError("ordered original intervention payload pins required")
    count = len(value["trials"])
    if (not 2 <= count <= 3 or len(value["materializations"]) > count
            or len(value["pairs"]) > count * (count - 1) // 2
            or uint(value["trial_count"], 3) != count
            or uint(value["pair_count"], 3) != len(value["pairs"])):
        raise ValueError("two or three original intervention trials and exact pair counts required")
    paths, total = set(), 0

    def add(pin, maximum):
        nonlocal total
        file_pin(pin, maximum)
        identity = pin["path"].casefold()
        if identity in paths:
            raise ValueError("duplicate or case-aliased intervention payload path")
        paths.add(identity)
        total = uint(total + pin["bytes"], MAX_PAYLOAD_BYTES)

    for name, filename, maximum in FIXED_FILES:
        pin = file_pin(value[name], maximum)
        if pin["path"] != filename:
            raise ValueError("original intervention fixed payload filename differs")
        add(pin, maximum)
    previous = -1
    for pin in value["materializations"]:
        file_pin(pin, MAX_MATERIALIZATION_BYTES)
        current = ordinal(pin["path"], "materializations")
        if not previous < current < count:
            raise ValueError("intervention materialization original ordinal/order differs")
        previous = current
        add(pin, MAX_MATERIALIZATION_BYTES)
    for group, maximum in (("trials", MAX_MATERIALIZATION_BYTES), ("pairs", MAX_METADATA_BYTES)):
        for current, pin in enumerate(value[group]):
            file_pin(pin, maximum)
            if ordinal(pin["path"], group) != current:
                raise ValueError("intervention trial/pair original ordinal differs")
            add(pin, maximum)
    if uint(value["total_payload_bytes"], MAX_PAYLOAD_BYTES) != total:
        raise ValueError("intervention unique payload byte total differs")
    return copy.deepcopy(value)


def collection_io(value):
    fields(value, IO_FIELDS, "original intervention collection I/O")
    for number in value.values():
        uint(number)
    if (not value["clock_epoch"] or value["finished_tick_ns"] < value["started_tick_ns"]
            or value["finished_tick_ns"] - value["started_tick_ns"] != value["elapsed_ns"]):
        raise ValueError("intervention original I/O interval differs")
    return value


def validate_intervention_ack(value, *, publication, publication_file):
    """Check exact saved FD-ACK projections; this issues no live UCI ACK."""
    publication = validate_intervention_publication(publication)
    file_pin(publication_file, MAX_METADATA_BYTES)
    if publication_file["path"] != PUBLICATION_FILE:
        raise ValueError("original intervention publication filename required")
    fields(value, ACK_FIELDS, "intervention durable ACK/2")
    file_pin(value["publication"], MAX_METADATA_BYTES)
    if (value["schema"] != ACK_DOMAIN or value["cohort_id"] != publication["cohort_id"]
            or value["cost_scope"] != COST_SCOPE or value["publication"] != publication_file):
        raise ValueError("intervention original durable ACK/publication binding differs")
    io = collection_io(value["collection_io"])
    expected_read = uint(publication["total_payload_bytes"] + publication_file["bytes"])
    expected_write = uint(sum(publication[name]["bytes"] for name, _, _ in FIXED_FILES)
        + sum(pin["bytes"] for pin in publication["pairs"]) + publication_file["bytes"])
    if (io["written_bytes"] != expected_write or io["readback_bytes"] != expected_read
            or io["synced_files"] != len(payload_pins(publication)) + 1 or not io["synced_directories"]):
        raise ValueError("intervention original final publication I/O coverage differs")
    acknowledgements = value["trial_acknowledgements"]
    if type(acknowledgements) is not list or len(acknowledgements) != publication["trial_count"]:
        raise ValueError("complete original intervention checkpoint ACK roster required")
    for index, entry in enumerate(acknowledgements):
        fields(entry, ("trial_sequence", "native_trial_sha256", "checkpoint", "collection_io"),
            "original intervention FD ACK projection")
        sha(entry["native_trial_sha256"])
        file_pin(entry["checkpoint"], MAX_MATERIALIZATION_BYTES)
        if uint(entry["trial_sequence"], 3) != index + 1 or entry["checkpoint"] != publication["trials"][index]:
            raise ValueError("intervention ACK original checkpoint order differs")
        interval = collection_io(entry["collection_io"])
        materialized = next((pin["bytes"] for pin in publication["materializations"]
            if ordinal(pin["path"], "materializations") == index), 0)
        if (interval["written_bytes"] != materialized + entry["checkpoint"]["bytes"]
                or interval["readback_bytes"] != materialized + 2 * entry["checkpoint"]["bytes"]
                or interval["synced_files"] != 2 + int(materialized > 0)
                or interval["synced_directories"] != 3 * (1 + int(materialized > 0))):
            raise ValueError("intervention original checkpoint I/O arithmetic differs")
    return copy.deepcopy(value)


def _absolute(value):
    if (type(value) is not str or not value or len(value.encode("utf-8")) > 4096
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)):
        raise ValueError("bounded original intervention absolute path required")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("original intervention absolute path required")
    return path


def _reference(value):
    fields(value, ("root", "publication", "durable_ack"), "intervention exact original file references")
    root = _absolute(value["root"])
    if not root.is_dir() or root.resolve(strict=True) != root:
        raise ValueError("plain original intervention root directory required")
    pins = {}
    for name, filename in (("publication", PUBLICATION_FILE), ("durable_ack", ACK_FILE)):
        pin = fields(value[name], ("path", "sha256", "bytes"), "intervention absolute metadata pin")
        path = _absolute(pin["path"])
        if path.parent != root or path.name != filename:
            raise ValueError("intervention metadata is outside its exact original root")
        pins[filename] = dict(pin, path=filename)
        file_pin(pins[filename], MAX_METADATA_BYTES)
    return root, pins


class HistoricalInterventionFiles:
    """Original checked bytes only; no JSON path can issue native authority."""
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

    def file_pin(self, path):
        relative(path)
        if path not in self._pins:
            raise ValueError("intervention file is outside its original pinned inventory")
        return copy.deepcopy(self._pins[path])

    def raw_file(self, path):
        pin = self.file_pin(path)
        raw = self._raw[path]
        if type(raw) is not bytes or len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
            raise ValueError("retained intervention raw bytes changed")
        return raw


def read_intervention_files(reference, *, read_accounting):
    if type(read_accounting) is not PinnedReadBudget or read_accounting.strict_growth_credit is not True:
        raise ValueError("original strict PinnedReadBudget required for intervention files")
    if read_accounting.observation()["unknown"]:
        raise ValueError("uncertain original intervention read budget")
    root, pins = _reference(reference)
    raw = {}
    for filename in (PUBLICATION_FILE, ACK_FILE):
        pin = pins[filename]
        raw[filename] = corpus._read_pinned_bytes(root / filename, pin["sha256"], pin["bytes"],
            MAX_METADATA_BYTES, read_accounting=read_accounting)
    publication = validate_intervention_publication(corpus._decode_json(raw[PUBLICATION_FILE]))
    ack = validate_intervention_ack(corpus._decode_json(raw[ACK_FILE]), publication=publication,
        publication_file=pins[PUBLICATION_FILE])
    for pin in payload_pins(publication):
        path = pin["path"]
        raw[path] = corpus._read_pinned_bytes(root.joinpath(*path.split("/")), pin["sha256"], pin["bytes"],
            pin["bytes"], read_accounting=read_accounting)
        pins[path] = copy.deepcopy(pin)
    observed = {"schema": OBSERVATION_DOMAIN, "scope": "original_publication_ack_and_raw_file_pins_only",
        "cohort_id": publication["cohort_id"], "publication_sha256": pins[PUBLICATION_FILE]["sha256"],
        "durable_ack_sha256": pins[ACK_FILE]["sha256"], "payload_count": len(payload_pins(publication)),
        "trial_count": publication["trial_count"], "pair_count": publication["pair_count"],
        "total_payload_bytes": publication["total_payload_bytes"], "known_pair_count": None,
        "payload_bodies_verified": False, "current_ack_issued": False, "native_authority_issued": False,
        "rules_verified": False, "physical_completion_verified": False,
        "training_target_authorized": False, "execution_authorized": False,
        "read_accounting": copy.deepcopy(read_accounting.observation())}
    result = HistoricalInterventionFiles()
    result._publication, result._ack, result._raw, result._pins = publication, ack, raw, pins
    result._paths = tuple(pin["path"] for pin in payload_pins(publication))
    result._observation = observed
    return result
