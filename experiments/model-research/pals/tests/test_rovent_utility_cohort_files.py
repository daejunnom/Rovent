"""Pure cohort file-pin fixtures, never Rules, trial or learner authority.

Payloads include deliberately opaque non-JSON bytes. They are synthetic test
data, not model weights or original materializations. Only publication shape,
raw file identity and the final collection-I/O arithmetic are under test.
"""
import copy
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime", "tensorflow", "jax"}
                or fullname in {"rz_pals_model.model", "rz_pals_model.training",
                    "rz_pals_model.nonzero_training", "rz_pals_model.nnuec_trainable",
                    "rz_pals_model.nnuec_utility_trainable"}):
            raise AssertionError("neural import forbidden in utility cohort file-pin fixtures")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_utility_cohort_files as cohort
    from rz_pals_model import rovent_training_bootstrap as bootstrap
finally:
    sys.meta_path.remove(_guard)


PUBLICATION = "utility-cohort-publication.json"
ACK = "utility-cohort-durable-ack.json"
PUBLICATION_SCHEMA = "rz-rovent-utility-cohort-publication/1"
ACK_SCHEMA = "rz-rovent-utility-cohort-durable-ack/1"
COST_SCOPE = "cohort_route_with_bounded_materialization_and_physical_pin_zero_excluding_durable_dataset_io_v1"
FIXED = (
    ("cohort_profile", "cohort-profile.json", 64 << 10),
    ("frozen_context", "frozen-context.json", 32 << 20),
    ("source", "source.json", 1 << 20),
    ("input_lock", "input-lock.json", 4 << 20),
    ("nnuec_weights", "nnuec-weights.json", 32 << 20),
    ("nnuec_provenance", "nnuec-provenance.json", 1 << 20),
)
FALSE_OBSERVATIONS = (
    "trial_acknowledgements_body_verified", "payload_bodies_verified",
    "native_authority_verified", "rules_verified", "physical_completion_verified",
    "training_target_authorized", "training_authority_verified", "execution_authorized",
)


def raw(value):
    # Preserve this noncanonical spacing/newline in the original file SHA.
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=1) + "\n").encode("utf-8")


def file_pin(path, body):
    return {"path": path, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}


def payloads(publication):
    return ([publication[field] for field, _, _ in FIXED]
        + publication["materializations"] + publication["trials"] + publication["pairs"])


def publication_fixture(*, trial_count=3, materialization_ordinals=(0, 2), pair_count=2):
    blobs = {name: ("synthetic opaque payload: " + name + "\r\n").encode("utf-8")
        for _, name, _ in FIXED}
    # A raw-pin consumer must not decode or load this deliberately non-JSON body.
    blobs["nnuec-weights.json"] = b"\xffsynthetic fixture bytes\x00not a model\r\n"
    materializations = [f"materializations/{index:06d}.json" for index in materialization_ordinals]
    trials = [f"trials/{index:06d}.json" for index in range(trial_count)]
    pairs = [f"pairs/{index:06d}.json" for index in range(pair_count)]
    for name in (*materializations, *trials, *pairs):
        blobs[name] = ("opaque original-shaped fixture: " + name + "\n").encode("utf-8")
    value = {"schema": PUBLICATION_SCHEMA, "cohort_id": "23" * 32, "cost_scope": COST_SCOPE,
        **{field: file_pin(name, blobs[name]) for field, name, _ in FIXED},
        "materializations": [file_pin(name, blobs[name]) for name in materializations],
        "trials": [file_pin(name, blobs[name]) for name in trials],
        "pairs": [file_pin(name, blobs[name]) for name in pairs],
        "trial_count": trial_count, "pair_count": pair_count,
        "total_payload_bytes": sum(len(body) for body in blobs.values())}
    return value, blobs


def ack_fixture(publication, publication_body):
    pinned = file_pin(PUBLICATION, publication_body)
    final_written = sum(publication[field]["bytes"] for field, _, _ in FIXED)
    final_written += sum(item["bytes"] for item in publication["pairs"]) + len(publication_body)
    return {"schema": ACK_SCHEMA, "cohort_id": publication["cohort_id"], "cost_scope": COST_SCOPE,
        "publication": pinned,
        "collection_io": {"clock_epoch": 7, "started_tick_ns": 100, "finished_tick_ns": 125,
            "elapsed_ns": 25, "written_bytes": final_written,
            "readback_bytes": publication["total_payload_bytes"] + len(publication_body),
            "synced_files": len(payloads(publication)) + 1, "synced_directories": 4},
        # This field has no typed trial-body contract yet. Preserve, do not attest.
        "trial_acknowledgements": [{"opaque_trial": "fixture-only", "ordinal": 0,
            "inter_trial_io": {"claimed_ns": 17}, "claimed_known_pair_count": 2}]}


def write_fixture(root, *, publication=None, blobs=None, publication_body=None,
                  ack=None, ack_body=None):
    if publication is None:
        publication, blobs = publication_fixture()
    if publication_body is None:
        publication_body = raw(publication)
    if ack is None:
        ack = ack_fixture(publication, publication_body)
    if ack_body is None:
        ack_body = raw(ack)
    all_blobs = {**blobs, PUBLICATION: publication_body, ACK: ack_body}
    for name, body in all_blobs.items():
        target = root.joinpath(*name.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    reference = {"root": str(root),
        "publication": file_pin(str(root / PUBLICATION), publication_body),
        "durable_ack": file_pin(str(root / ACK), ack_body)}
    return reference, publication, ack, all_blobs


def budget(maximum=1 << 20, *, strict=True, now=None):
    current = [100] if now is None else now
    return bootstrap.PinnedReadBudget(maximum, until_ns=1000, clock=lambda: current[0],
        strict_growth_credit=strict)


class UtilityCohortFileTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_publication_validates_exact_shape_sparse_ordinals_and_detaches_its_copy(self):
        value, _ = publication_fixture()
        checked = cohort.validate_utility_cohort_publication(value)
        self.assertEqual(checked, value)
        self.assertEqual([item["path"] for item in checked["materializations"]],
            ["materializations/000000.json", "materializations/000002.json"])
        checked["trials"][0]["sha256"] = "99" * 32
        self.assertNotEqual(checked, value)
        self.assertEqual(value["trials"][0]["path"], "trials/000000.json")

    def test_publication_rejects_missing_extra_or_legacy_fields_and_inexact_types(self):
        original, _ = publication_fixture()
        changes = (
            lambda value: value.pop("cost_scope"),
            lambda value: value.update(known_pair_count=2),
            lambda value: value.update(schema="rz-rovent-utility-cohort-publication/0"),
            lambda value: value.update(cost_scope="route-only"),
            lambda value: value.update(cohort_id="AB" * 32),
            lambda value: value.update(trial_count=True),
            lambda value: value.update(pair_count=2.0),
            lambda value: value["source"].update(bytes=True),
            lambda value: value["source"].update(bytes=0),
            lambda value: value["source"].update(extra="not-a-pin"),
        )
        for index, change in enumerate(changes):
            value = copy.deepcopy(original)
            change(value)
            with self.subTest(case=index), self.assertRaises(ValueError):
                cohort.validate_utility_cohort_publication(value)

    def test_closed_trial_and_pair_counts_do_not_imply_known_targets(self):
        for trials, materializations, pairs in ((1, (), 0), (2, (1,), 1), (3, (0, 2), 3)):
            value, _ = publication_fixture(trial_count=trials,
                materialization_ordinals=materializations, pair_count=pairs)
            self.assertEqual(cohort.validate_utility_cohort_publication(value), value)
        for trials, materializations, pairs in ((0, (), 0), (4, (), 0), (1, (), 1), (2, (), 2), (3, (), 4)):
            value, _ = publication_fixture(trial_count=trials,
                materialization_ordinals=materializations, pair_count=pairs)
            with self.subTest(trials=trials, pairs=pairs), self.assertRaises(ValueError):
                cohort.validate_utility_cohort_publication(value)
        value, _ = publication_fixture()
        value["trial_count"] += 1
        with self.assertRaises(ValueError):
            cohort.validate_utility_cohort_publication(value)

    def test_trial_pair_and_materialization_order_cannot_be_renumbered_or_duplicated(self):
        original, _ = publication_fixture()
        changes = (
            lambda value: value["trials"].reverse(),
            lambda value: value["pairs"].reverse(),
            lambda value: value["materializations"].reverse(),
            lambda value: value["materializations"][1].update(path="materializations/000003.json"),
            lambda value: value["materializations"][1].update(path="materializations/000000.json"),
            lambda value: value["trials"][0].update(path="trials/0.json"),
            lambda value: value["pairs"][0].update(path="pairs/000001.json"),
        )
        for index, change in enumerate(changes):
            value = copy.deepcopy(original)
            change(value)
            with self.subTest(case=index), self.assertRaises(ValueError):
                cohort.validate_utility_cohort_publication(value)

    def test_fixed_paths_and_ordinal_paths_reject_escape_case_aliases_and_backslashes(self):
        original, _ = publication_fixture()
        for name in ("../source.json", "/source.json", "SOURCE.json", "./source.json", "x/../source.json",
                     "C:/source.json", "source.json/", "source\\json"):
            changed = copy.deepcopy(original)
            changed["source"]["path"] = name
            with self.subTest(path=name), self.assertRaises(ValueError):
                cohort.validate_utility_cohort_publication(changed)
        for name in ("trials/../000000.json", "Trials/000000.json", "trials\\000000.json",
                     "trials/000000.JSON", "trials//000000.json", "trials/000000.json/extra"):
            changed = copy.deepcopy(original)
            changed["trials"][0]["path"] = name
            with self.subTest(path=name), self.assertRaises(ValueError):
                cohort.validate_utility_cohort_publication(changed)

    def test_each_payload_class_has_a_separate_cap_and_total_bytes_are_exact(self):
        original, _ = publication_fixture()
        targets = [(field, None, maximum) for field, _, maximum in FIXED]
        targets += [("materializations", 0, 32 << 20), ("trials", 0, 32 << 20), ("pairs", 0, 64 << 10)]
        for field, index, maximum in targets:
            accepted = copy.deepcopy(original)
            target = accepted[field] if index is None else accepted[field][index]
            accepted["total_payload_bytes"] += maximum - target["bytes"]
            target["bytes"] = maximum
            with self.subTest(field=field, boundary="exact cap"):
                self.assertEqual(cohort.validate_utility_cohort_publication(accepted), accepted)
            value = copy.deepcopy(original)
            target = value[field] if index is None else value[field][index]
            value["total_payload_bytes"] += maximum + 1 - target["bytes"]
            target["bytes"] = maximum + 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                cohort.validate_utility_cohort_publication(value)
        for total in (original["total_payload_bytes"] - 1, original["total_payload_bytes"] + 1,
                      (256 << 20) + 1, True):
            value = copy.deepcopy(original)
            value["total_payload_bytes"] = total
            with self.subTest(total=total), self.assertRaises(ValueError):
                cohort.validate_utility_cohort_publication(value)
        # No large payload is allocated: these declared pins independently fit
        # their classes but their exact sum exceeds the global publication cap.
        combined, _ = publication_fixture(materialization_ordinals=(0, 1, 2), pair_count=3)
        for field, _, maximum in FIXED:
            combined[field]["bytes"] = maximum
        for category, maximum in (("materializations", 32 << 20), ("trials", 32 << 20), ("pairs", 64 << 10)):
            for item in combined[category]:
                item["bytes"] = maximum
        combined["total_payload_bytes"] = sum(item["bytes"] for item in payloads(combined))
        self.assertGreater(combined["total_payload_bytes"], 256 << 20)
        with self.assertRaises(ValueError):
            cohort.validate_utility_cohort_publication(combined)
        combined["trials"][-1]["bytes"] -= combined["total_payload_bytes"] - (256 << 20)
        combined["total_payload_bytes"] = 256 << 20
        self.assertEqual(cohort.validate_utility_cohort_publication(combined), combined)

    def test_ack6_checks_only_final_publication_io_and_preserves_the_untyped_trial_body(self):
        publication, _ = publication_fixture()
        body = raw(publication)
        original = ack_fixture(publication, body)
        checked = cohort.validate_utility_cohort_ack(original, publication=publication,
            publication_file=file_pin(PUBLICATION, body))
        self.assertEqual(checked, original)
        io = checked["collection_io"]
        excluded = sum(item["bytes"] for item in publication["trials"] + publication["materializations"])
        self.assertEqual(io["written_bytes"], io["readback_bytes"] - excluded)
        self.assertGreater(io["readback_bytes"], io["written_bytes"])
        checked["trial_acknowledgements"][0]["inter_trial_io"]["claimed_ns"] = 999
        self.assertEqual(original["trial_acknowledgements"][0]["inter_trial_io"]["claimed_ns"], 17)

    def test_stale_ack5_unknown_fields_and_changed_publication_join_are_rejected(self):
        publication, _ = publication_fixture()
        body = raw(publication)
        original = ack_fixture(publication, body)
        changes = (
            lambda value: value.pop("trial_acknowledgements"),
            lambda value: value.update(extra="not-an-ack-field"),
            lambda value: value.update(cohort_id="99" * 32),
            lambda value: value.update(cost_scope="route-only"),
            lambda value: value["publication"].update(sha256="99" * 32),
            lambda value: value["publication"].update(path="other-publication.json"),
            lambda value: value["publication"].update(bytes=len(body) + 1),
        )
        for index, change in enumerate(changes):
            value = copy.deepcopy(original)
            change(value)
            with self.subTest(case=index), self.assertRaises(ValueError):
                cohort.validate_utility_cohort_ack(value, publication=publication,
                    publication_file=file_pin(PUBLICATION, body))

    def test_collection_io_is_exact_typed8_and_cannot_include_ack_or_prior_trial_writes(self):
        publication, _ = publication_fixture()
        body = raw(publication)
        original = ack_fixture(publication, body)
        changes = (
            lambda io: io.pop("clock_epoch"),
            lambda io: io.update(extra=0),
            lambda io: io.update(clock_epoch=True),
            lambda io: io.update(started_tick_ns=-1),
            lambda io: io.update(finished_tick_ns=99),
            lambda io: io.update(elapsed_ns=26),
            lambda io: io.update(written_bytes=io["readback_bytes"]),
            lambda io: io.update(readback_bytes=io["readback_bytes"] + 1),
            lambda io: io.update(synced_files=io["synced_files"] + 1),
            lambda io: io.update(synced_directories=0),
            lambda io: io.update(elapsed_ns=25.0),
        )
        for index, change in enumerate(changes):
            value = copy.deepcopy(original)
            change(value["collection_io"])
            with self.subTest(case=index), self.assertRaises(ValueError):
                cohort.validate_utility_cohort_ack(value, publication=publication,
                    publication_file=file_pin(PUBLICATION, body))

    def test_untyped_trial_acknowledgements_remain_bounded_plain_json(self):
        publication, _ = publication_fixture()
        body = raw(publication)
        original = ack_fixture(publication, body)
        for opaque in (None, [], {"unverified_native_claim": True, "known_pairs": [[4, 5]]}, [1, "raw", False]):
            value = copy.deepcopy(original)
            value["trial_acknowledgements"] = opaque
            self.assertEqual(cohort.validate_utility_cohort_ack(value, publication=publication,
                publication_file=file_pin(PUBLICATION, body))["trial_acknowledgements"], opaque)
        for opaque in ({"not_json": object()}, {"nonfinite": float("nan")}, "x" * (64 << 10)):
            value = copy.deepcopy(original)
            value["trial_acknowledgements"] = opaque
            with self.subTest(value=type(opaque).__name__), self.assertRaises(ValueError):
                cohort.validate_utility_cohort_ack(value, publication=publication,
                    publication_file=file_pin(PUBLICATION, body))

    def test_readback_preserves_original_bytes_pins_order_and_partial_observation_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            reference, publication, ack, blobs = write_fixture(root)
            shared = budget()
            owner = cohort.read_utility_cohort_files(reference, read_accounting=shared)
            self.assertIs(type(owner), cohort.HistoricalUtilityCohortFiles)
            expected_paths = tuple(item["path"] for item in payloads(publication))
            self.assertEqual(owner.payload_paths, expected_paths)
            self.assertEqual(owner.publication(), publication)
            self.assertEqual(owner.durable_ack(), ack)
            for name, body in blobs.items():
                self.assertEqual(owner.raw_file(name), body)
                self.assertEqual(owner.file_pin(name), file_pin(name, body))
            self.assertNotEqual(blobs[PUBLICATION], json.dumps(publication, sort_keys=True,
                separators=(",", ":")).encode("utf-8"))
            observation = owner.observation()
            self.assertEqual(observation["schema"], "rz-rovent-utility-cohort-files-observation/1")
            self.assertEqual(observation["scope"], "publication_and_raw_file_pins_only")
            self.assertEqual(observation["cohort_id"], publication["cohort_id"])
            self.assertEqual(observation["publication_sha256"], reference["publication"]["sha256"])
            self.assertEqual(observation["durable_ack_sha256"], reference["durable_ack"]["sha256"])
            self.assertEqual(observation["payload_count"], len(expected_paths))
            self.assertEqual(observation["trial_count"], 3)
            self.assertEqual(observation["pair_count"], 2)
            self.assertEqual(observation["total_payload_bytes"], publication["total_payload_bytes"])
            self.assertIsNone(observation["known_pair_count"])
            for field in FALSE_OBSERVATIONS:
                self.assertIs(observation[field], False)
            self.assertEqual(observation["read_accounting"], shared.observation())
            self.assertEqual(shared.spent, sum(len(body) for body in blobs.values()))

    def test_accessors_return_detached_metadata_and_do_not_read_files_again(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, publication, _, blobs = write_fixture(Path(directory).resolve())
            shared = budget()
            owner = cohort.read_utility_cohort_files(reference, read_accounting=shared)
            charged = shared.spent
            owner.publication()["source"]["sha256"] = "99" * 32
            owner.durable_ack()["trial_acknowledgements"].clear()
            owner.file_pin("source.json")["sha256"] = "99" * 32
            owner.observation()["read_accounting"]["charged_bytes"] = 0
            self.assertEqual(owner.publication(), publication)
            self.assertTrue(owner.durable_ack()["trial_acknowledgements"])
            self.assertEqual(owner.file_pin("source.json"), file_pin("source.json", blobs["source.json"]))
            self.assertEqual(owner.raw_file("nnuec-weights.json"), blobs["nnuec-weights.json"])
            self.assertEqual(owner.observation()["read_accounting"]["charged_bytes"], charged)
            self.assertEqual(shared.spent, charged)
            for path in ("../source.json", "SOURCE.json", "trials/000003.json", "/source.json"):
                with self.subTest(path=path), self.assertRaises((ValueError, KeyError)):
                    owner.raw_file(path)

    def test_reference_requires_exact_absolute_in_root_publication_and_ack_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            original, _, _, _ = write_fixture(root)
            changes = (
                lambda value: value.update(extra="not-a-reference-field"),
                lambda value: value.pop("durable_ack"),
                lambda value: value.update(root="relative-root"),
                lambda value: value["publication"].update(path=PUBLICATION),
                lambda value: value["publication"].update(path=str(root / "OTHER.json")),
                lambda value: value["publication"].update(path=str(root / PUBLICATION.upper())),
                lambda value: value["durable_ack"].update(path=str(root.parent / ACK)),
                lambda value: value["durable_ack"].update(extra="not-a-file-pin"),
            )
            for index, change in enumerate(changes):
                value = copy.deepcopy(original)
                change(value)
                with self.subTest(case=index), self.assertRaises(ValueError):
                    cohort.read_utility_cohort_files(value, read_accounting=budget())

    def test_reader_rejects_duplicate_json_keys_even_with_the_new_matching_raw_pin(self):
        for target in (PUBLICATION, ACK):
            with self.subTest(file=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                reference, _, _, blobs = write_fixture(root)
                original = blobs[target]
                changed = original.replace(b'"schema":', b'"schema":"duplicate", "schema":', 1)
                (root / target).write_bytes(changed)
                key = "publication" if target == PUBLICATION else "durable_ack"
                reference[key] = file_pin(str(root / target), changed)
                shared = budget()
                with self.assertRaises(ValueError):
                    cohort.read_utility_cohort_files(reference, read_accounting=shared)
                self.assertGreater(shared.spent, 0)

    def test_matching_raw_hash_does_not_relax_the_metadata_size_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            reference, _, _, _ = write_fixture(root)
            oversized = b" " * ((64 << 10) + 1)
            (root / PUBLICATION).write_bytes(oversized)
            reference["publication"] = file_pin(str(root / PUBLICATION), oversized)
            with self.assertRaises(ValueError):
                cohort.read_utility_cohort_files(reference, read_accounting=budget())

    def test_shared_budget_must_be_the_original_exact_strict_growth_account(self):
        class SubclassedBudget(bootstrap.PinnedReadBudget):
            pass

        with tempfile.TemporaryDirectory() as directory:
            reference, _, _, _ = write_fixture(Path(directory).resolve())
            invalid = (None, {}, object(), budget(strict=False),
                SubclassedBudget(1 << 20, until_ns=1000, clock=lambda: 100, strict_growth_credit=True))
            for account in invalid:
                with self.subTest(account=type(account).__name__), self.assertRaises(ValueError):
                    cohort.read_utility_cohort_files(reference, read_accounting=account)

    def test_previously_charged_reads_remain_in_the_same_budget_without_refund(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, _, _, blobs = write_fixture(Path(directory).resolve())
            total = sum(len(body) for body in blobs.values())
            shared = budget(total + 18)
            shared.account_external(17)
            owner = cohort.read_utility_cohort_files(reference, read_accounting=shared)
            self.assertEqual(shared.spent, total + 17)
            self.assertEqual(owner.observation()["read_accounting"], shared.observation())
            self.assertFalse(shared.observation()["unknown"])

    def test_aggregate_credit_must_cover_the_last_file_growth_probe_too(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, _, _, blobs = write_fixture(Path(directory).resolve())
            total = sum(len(body) for body in blobs.values())
            shared = budget(total)
            with self.assertRaises(ValueError):
                cohort.read_utility_cohort_files(reference, read_accounting=shared)
            self.assertLess(shared.spent, total)
            self.assertGreater(shared.spent, 0)

    def test_raw_payload_mismatch_is_charged_and_quarantines_the_shared_account(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            reference, _, _, blobs = write_fixture(root)
            name = "source.json"
            replacement = b"!" + blobs[name][1:]
            self.assertEqual(len(replacement), len(blobs[name]))
            (root / name).write_bytes(replacement)
            shared = budget()
            with self.assertRaises(ValueError):
                cohort.read_utility_cohort_files(reference, read_accounting=shared)
            charged = shared.spent
            self.assertTrue(shared.failed)
            self.assertGreaterEqual(charged, len(replacement))
            with self.assertRaises(ValueError):
                cohort.read_utility_cohort_files(reference, read_accounting=shared)
            self.assertEqual(shared.spent, charged)

    def test_growth_during_read_is_detected_by_the_reserved_extra_byte_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            reference, _, _, blobs = write_fixture(root)
            publication_body = blobs[PUBLICATION]
            original_charge = bootstrap.PinnedReadBudget.charge_external
            grew = [False]

            def grow_after_returned_body(account, lease, count):
                result = original_charge(account, lease, count)
                if not grew[0] and lease.maximum == len(publication_body) + 1 and count == len(publication_body):
                    with (root / PUBLICATION).open("ab") as stream:
                        stream.write(b"!")
                    grew[0] = True
                return result

            shared = budget()
            with patch.object(bootstrap.PinnedReadBudget, "charge_external", new=grow_after_returned_body):
                with self.assertRaises(ValueError):
                    cohort.read_utility_cohort_files(reference, read_accounting=shared)
            self.assertTrue(grew[0])
            self.assertTrue(shared.failed)
            self.assertGreaterEqual(shared.spent, len(publication_body) + 1)

    def test_expired_or_already_failed_budget_cannot_begin_a_new_read(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, _, _, _ = write_fixture(Path(directory).resolve())
            expired = budget(now=[1000])
            with self.assertRaises(TimeoutError):
                cohort.read_utility_cohort_files(reference, read_accounting=expired)
            self.assertEqual(expired.spent, 0)
            failed = budget()
            failed.failed = True
            with self.assertRaises(ValueError):
                cohort.read_utility_cohort_files(reference, read_accounting=failed)
            self.assertEqual(failed.spent, 0)


if __name__ == "__main__":
    unittest.main()
