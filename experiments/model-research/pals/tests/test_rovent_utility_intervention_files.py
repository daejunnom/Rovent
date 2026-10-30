"""Synthetic original file pins only; no native, Rules or known-target proof."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from rz_pals_model import rovent_utility_intervention_files as files
from rz_pals_model.rovent_training_bootstrap import PinnedReadBudget
from rz_pals_model.rovent_utility_cohort_files import validate_utility_cohort_publication


def raw(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=1) + "\n").encode()


def pin(path, body):
    return {"path": path, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}


def fixture():
    blobs = {filename: ("opaque synthetic original /2 payload " + filename).encode()
        for _, filename, _ in files.FIXED_FILES}
    blobs["nnuec-weights.json"] = b"\xffopaque bytes, not initialized model or optimizer\x00"
    for name in ("trials/000000.json", "trials/000001.json", "materializations/000001.json", "pairs/000000.json"):
        blobs[name] = ("opaque source pin only " + name + "\r\n").encode()
    publication = {"schema": files.PUBLICATION_DOMAIN, "cohort_id": "ab" * 32, "cost_scope": files.COST_SCOPE,
        **{field: pin(filename, blobs[filename]) for field, filename, _ in files.FIXED_FILES},
        "materializations": [pin("materializations/000001.json", blobs["materializations/000001.json"])],
        "trials": [pin(f"trials/{index:06d}.json", blobs[f"trials/{index:06d}.json"]) for index in range(2)],
        "pairs": [pin("pairs/000000.json", blobs["pairs/000000.json"])], "trial_count": 2, "pair_count": 1,
        "total_payload_bytes": sum(map(len, blobs.values()))}
    publication_raw = raw(publication)
    publication_file = pin(files.PUBLICATION_FILE, publication_raw)
    acknowledgements = []
    for index, checkpoint in enumerate(publication["trials"]):
        materialized = 0 if index == 0 else publication["materializations"][0]["bytes"]
        # These are distinct original clocks, deliberately descending ticks.
        started = 9000 if index == 0 else 10
        acknowledgements.append({"trial_sequence": index + 1, "native_trial_sha256": "cd" * 32,
            "checkpoint": copy.deepcopy(checkpoint), "collection_io": {"clock_epoch": 17 + index,
                "started_tick_ns": started, "finished_tick_ns": started + 10, "elapsed_ns": 10,
                "written_bytes": materialized + checkpoint["bytes"],
                "readback_bytes": materialized + 2 * checkpoint["bytes"],
                "synced_files": 2 + int(materialized > 0), "synced_directories": 3 * (1 + int(materialized > 0))}})
    ack = {"schema": files.ACK_DOMAIN, "cohort_id": publication["cohort_id"], "cost_scope": files.COST_SCOPE,
        "publication": publication_file, "trial_acknowledgements": acknowledgements,
        "collection_io": {"clock_epoch": 18, "started_tick_ns": 30, "finished_tick_ns": 40, "elapsed_ns": 10,
            "written_bytes": sum(publication[field]["bytes"] for field, _, _ in files.FIXED_FILES)
                + sum(item["bytes"] for item in publication["pairs"]) + len(publication_raw),
            "readback_bytes": publication["total_payload_bytes"] + len(publication_raw),
            "synced_files": len(blobs) + 1, "synced_directories": 7}}
    return publication, ack, {**blobs, files.PUBLICATION_FILE: publication_raw, files.ACK_FILE: raw(ack)}


def write(root):
    publication, ack, blobs = fixture()
    for name, body in blobs.items():
        path = root.joinpath(*name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    reference = {"root": str(root), "publication": pin(str(root / files.PUBLICATION_FILE), blobs[files.PUBLICATION_FILE]),
        "durable_ack": pin(str(root / files.ACK_FILE), blobs[files.ACK_FILE])}
    return reference, publication, ack, blobs


def budget(maximum=1 << 20):
    return PinnedReadBudget(maximum, until_ns=1000, clock=lambda: 100, strict_growth_credit=True)


class InterventionFilesTests(unittest.TestCase):
    def test_exact_new_publication_and_independent_ack_clocks(self):
        publication, ack, blobs = fixture()
        self.assertEqual(files.validate_intervention_publication(publication), publication)
        self.assertEqual(files.validate_intervention_ack(ack, publication=publication,
            publication_file=pin(files.PUBLICATION_FILE, blobs[files.PUBLICATION_FILE])), ack)
        self.assertGreater(ack["trial_acknowledgements"][0]["collection_io"]["started_tick_ns"],
            ack["trial_acknowledgements"][1]["collection_io"]["started_tick_ns"])

    def test_old_publication_decoder_stays_closed(self):
        publication, _, _ = fixture()
        with self.assertRaises(ValueError):
            validate_utility_cohort_publication(publication)
        old = copy.deepcopy(publication)
        old["schema"] = "rz-rovent-utility-cohort-publication/1"
        old["frozen_context"] = old.pop("neutral_target")
        with self.assertRaises(ValueError):
            files.validate_intervention_publication(old)

    def test_original_shape_ordinals_and_counts_are_not_filled(self):
        for axis in ("single_trial", "trial_order", "false_count", "extra_field", "escaped_neutral"):
            publication, _, _ = fixture()
            if axis == "single_trial":
                publication["trials"].pop(); publication["trial_count"] = 1
            elif axis == "trial_order":
                publication["trials"].reverse()
            elif axis == "false_count":
                publication["trial_count"] = True
            elif axis == "extra_field":
                publication["known"] = True
            else:
                publication["neutral_target"]["path"] = "../neutral-target.json"
            with self.subTest(axis=axis), self.assertRaises(ValueError):
                files.validate_intervention_publication(publication)

    def test_ack_exact_pin_roster_and_real_io_counts_are_required(self):
        for axis in ("old_domain", "missing_ack", "wrong_pin", "foreign_clock", "false_count", "io_bytes"):
            publication, ack, blobs = fixture()
            if axis == "old_domain": ack["schema"] = "rz-rovent-utility-cohort-durable-ack/1"
            elif axis == "missing_ack": ack["trial_acknowledgements"].pop()
            elif axis == "wrong_pin": ack["trial_acknowledgements"][0]["checkpoint"]["sha256"] = "ef" * 32
            elif axis == "foreign_clock": ack["trial_acknowledgements"][0]["collection_io"]["clock_epoch"] = 0
            elif axis == "false_count": ack["trial_acknowledgements"][0]["trial_sequence"] = True
            else: ack["trial_acknowledgements"][1]["collection_io"]["readback_bytes"] += 1
            with self.subTest(axis=axis), self.assertRaises(ValueError):
                files.validate_intervention_ack(ack, publication=publication,
                    publication_file=pin(files.PUBLICATION_FILE, blobs[files.PUBLICATION_FILE]))

    def test_fd_consumer_keeps_opaque_original_bytes_and_no_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, publication, _, blobs = write(Path(directory).resolve())
            accounting = budget()
            owner = files.read_intervention_files(reference, read_accounting=accounting)
            self.assertIs(type(owner), files.HistoricalInterventionFiles)
            self.assertEqual(owner.publication(), publication)
            self.assertEqual(owner.raw_file("nnuec-weights.json"), blobs["nnuec-weights.json"])
            self.assertEqual(set(owner.payload_paths), {item["path"] for item in files.payload_pins(publication)})
            observed = owner.observation()
            self.assertIsNone(observed["known_pair_count"])
            for name in ("payload_bodies_verified", "current_ack_issued", "native_authority_issued",
                    "rules_verified", "physical_completion_verified", "training_target_authorized", "execution_authorized"):
                self.assertIs(observed[name], False)
            owner._raw["source.json"] += b"tampered"
            with self.assertRaises(ValueError):
                owner.raw_file("source.json")

    def test_changed_pinned_payload_and_insufficient_original_read_credit_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            reference, _, _, _ = write(root)
            with self.assertRaises(ValueError):
                files.read_intervention_files(reference, read_accounting=budget(1))
            (root / "source.json").write_bytes(b"unrelated replacement")
            with self.assertRaises(ValueError):
                files.read_intervention_files(reference, read_accounting=budget())

    def test_hardlinked_original_payload_is_not_an_independent_owned_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            reference, _, _, _ = write(root)
            try:
                os.link(root / "source.json", root / "source-link.json")
            except (OSError, NotImplementedError):
                self.skipTest("hardlink creation unavailable")
            with self.assertRaises(ValueError):
                files.read_intervention_files(reference, read_accounting=budget())
