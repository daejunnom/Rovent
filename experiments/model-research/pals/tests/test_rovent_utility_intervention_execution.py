"""Closed family and original byte boundaries, with no learner execution."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rz_pals_model.generation_data import digest
from rz_pals_model import rovent_utility_training_bootstrap as legacy
from rz_pals_model import rovent_utility_training_child as common_child
from rz_pals_model import rovent_utility_intervention_execution as execution
from rz_pals_model.rovent_utility_intervention_authority import (
    ParentSocketInterventionUtilityTrainerAuthority, require_intervention_utility_authority)
from rz_pals_model.rovent_utility_intervention_child import intervention_utility_child_main
from rz_pals_model.rovent_utility_intervention_preparation import prepare_intervention_from_files
from rz_pals_model.rovent_utility_training_authority import (OriginalUtilityAuthorityRequired,
    ParentSocketUtilityTrainerAuthority, require_original_utility_authority)
from rz_pals_model.rovent_utility_training_family import LEGACY, INTERVENTION, require_family
from rz_pals_model.rovent_utility_training_window import validate_utility_window_v2
from test_rovent_utility_intervention_files import budget
from test_rovent_utility_intervention_payload import intervention_case, write_case
from test_rovent_utility_training_bootstrap import fixture as declaration_fixture
from test_rovent_utility_training_window import spec as window_fields


NEW_SOURCE_NAMES = ("rovent_utility_intervention.py", "rovent_utility_intervention_files.py",
    "rovent_utility_intervention_payload.py", "rovent_utility_intervention_preparation.py",
    "rovent_utility_intervention_numerical.py", "rovent_utility_intervention_dataset.py",
    "rovent_utility_intervention_execution.py", "rovent_utility_intervention_authority.py",
    "rovent_utility_intervention_training.py", "rovent_utility_intervention_checkpoint.py",
    "rovent_utility_intervention_child.py", "rovent_utility_training_family.py")


def _seal_declaration(value):
    inventory = {item["relative_path"]: {name: item[name] for name in ("sha256", "bytes")}
        for item in value["source_files"]}
    registration = value["registration"]
    registration["trainer_source_sha256"] = digest(legacy.SOURCE_DOMAIN, inventory)
    value["registration_sha256"] = digest(execution.REGISTRATION_DOMAIN, registration)
    return value


def intervention_declaration():
    # New synthetic declaration, not a converted historical publication. No
    # FD, native execution evidence, account or current peer is issued here.
    value = declaration_fixture()
    value.update(schema=execution.STARTUP_DOMAIN, data_family="intervention_v2")
    value["registration"].update(schema=execution.REGISTRATION_DOMAIN, data_family="intervention_v2")
    prefix = "experiments/model-research/pals/src/rz_pals_model/"
    value["source_files"].extend({"relative_path": prefix + name, "sha256": "62" * 32, "bytes": 10}
        for name in NEW_SOURCE_NAMES)
    return _seal_declaration(value)


def _validate(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return execution.validate_intervention_startup(raw, expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_bytes=len(raw), execution_until_ns=1000000100, whole_until_ns=2000000100, now_ns=200)


class InterventionExecutionTests(unittest.TestCase):
    def test_new_startup_and_registration_are_closed_declarations_not_current_authority(self):
        value = intervention_declaration()
        self.assertEqual(len(value), 19)
        self.assertEqual(len(value["registration"]), 30)
        self.assertEqual(_validate(value), value)
        self.assertEqual(execution.validate_intervention_registration(value["registration"]), value["registration"])
        with self.assertRaises(ValueError): legacy.validate_utility_registration_v2(value["registration"])
        raw = json.dumps(value).encode()
        with self.assertRaises(ValueError):
            legacy.validate_startup(raw, expected_sha256=hashlib.sha256(raw).hexdigest(), expected_bytes=len(raw),
                execution_until_ns=1000000100, whole_until_ns=2000000100, now_ns=200)

    def test_new_startup_keeps_raw_byte_registration_family_deadline_and_source_pins(self):
        mutations = (lambda v: v.update(data_family="utility_cohort_v1"),
            lambda v: v["registration"].update(data_family="utility_cohort_v1"),
            lambda v: v.update(registration_sha256="ff" * 32),
            lambda v: v.update(execution_until_ns=1000000101),
            lambda v: v.update(execution_authorized=True))
        for mutate in mutations:
            value = intervention_declaration(); mutate(value)
            with self.assertRaises(ValueError): _validate(value)
        for removed in ("rovent_utility_intervention_child.py", "rovent_utility_intervention_dataset.py"):
            value = intervention_declaration()
            value["source_files"] = [item for item in value["source_files"] if not item["relative_path"].endswith("/" + removed)]
            _seal_declaration(value)
            with self.assertRaises(ValueError): _validate(value)
        value = intervention_declaration()
        raw = json.dumps(value).encode()
        with self.assertRaises(ValueError):
            execution.validate_intervention_startup(raw, expected_sha256="00" * 32, expected_bytes=len(raw),
                execution_until_ns=1000000100, whole_until_ns=2000000100, now_ns=200)

    def test_intervention_ancestor_kind_is_not_added_to_old_registration(self):
        from test_rovent_utility_training_bootstrap import pin
        value = intervention_declaration()["registration"]
        value["ancestor_datasets"] = [{"kind": "utility_intervention", "registration": {"root": "/data/previous",
            "publication": pin("/data/previous/utility-cohort-publication.json", "71"),
            "durable_ack": pin("/data/previous/utility-cohort-durable-ack.json", "72")},
            "preparation": pin("/data/previous-preparation.json", "73")}]
        self.assertEqual(execution.validate_intervention_registration(value), value)
        old = copy.deepcopy(value); old.pop("data_family"); old["schema"] = legacy.REGISTRATION_V2_DOMAIN
        with self.assertRaises(ValueError): legacy.validate_utility_registration_v2(old)
        changed = copy.deepcopy(value); changed["ancestor_datasets"][0].pop("preparation")
        with self.assertRaises(ValueError): execution.validate_intervention_registration(changed)

    def test_new_window_uses_original_exact_nine_preparation_and_same_numerical_limits(self):
        recipe, registered, spec, blobs = intervention_case()
        with tempfile.TemporaryDirectory() as directory:
            reference = write_case(Path(directory).resolve(), blobs)
            preparation = prepare_intervention_from_files(reference, recipe, registered, spec, read_budget=budget(4 << 20))
            value = {**window_fields(preparation), "schema": execution.WINDOW_DOMAIN, "checkpoint_reserve_ns": 2000000000}
            self.assertEqual(len(value), 23)
            self.assertEqual(execution.validate_intervention_window(preparation, value), value)
            self.assertEqual(execution.intervention_window_digest(value), digest(execution.WINDOW_DOMAIN, value))
            self.assertEqual(len(preparation), 9)
            with self.assertRaises(ValueError): validate_utility_window_v2(preparation, value)
            for mutate in (lambda v: v.update(accumulation=4), lambda v: v.update(precision="fp16"),
                    lambda v: v.update(checkpoint_reserve_ns=v["max_wall_time_ns"]),
                    lambda v: v.update(preparation_sha256="ff" * 32)):
                changed = copy.deepcopy(value); mutate(changed)
                with self.assertRaises(ValueError): execution.validate_intervention_window(preparation, changed)

    def test_closed_family_and_json_are_not_socket_or_lease_capabilities(self):
        self.assertIs(require_family(INTERVENTION), INTERVENTION)
        self.assertEqual(INTERVENTION.domain("rz-rovent-utility-trainer-rpc/1"), "rz-rovent-utility-trainer-rpc/2")
        self.assertEqual(LEGACY.domain("rz-rovent-utility-trainer-rpc/1"), "rz-rovent-utility-trainer-rpc/1")
        for stem in ("trainer-source-bundle", "sampler", "checkpoint-cost"):
            domain = "rz-rovent-utility-" + stem + "/1"
            self.assertEqual(INTERVENTION.domain(domain), domain)
        for candidate in (None, "intervention_v2", {"data_family": "intervention_v2"}, object()):
            with self.assertRaises(TypeError): require_family(candidate)
        for candidate in (None, {"execution_authorized": True}, object(),
                object.__new__(ParentSocketUtilityTrainerAuthority),
                object.__new__(ParentSocketInterventionUtilityTrainerAuthority)):
            with self.assertRaises(OriginalUtilityAuthorityRequired):
                require_intervention_utility_authority(candidate, allow_unleased=True)
            with self.assertRaises(OriginalUtilityAuthorityRequired):
                require_original_utility_authority(candidate, allow_unleased=True)

    def test_new_named_child_requires_execute_and_rejects_old_registration_before_bootstrap(self):
        arguments = ["--startup-sha256", "10" * 32, "--startup-bytes", "1",
            "--execution-until-ns", "100", "--whole-until-ns", "200"]
        with patch.object(execution, "validate_intervention_startup") as validate:
            with self.assertRaises(ValueError): intervention_utility_child_main(arguments, stdin=io.BytesIO(b"x"))
            validate.assert_not_called()
        old = {"registration": {"schema": legacy.REGISTRATION_V2_DOMAIN}}
        with patch.object(execution, "validate_intervention_startup", return_value=old), \
                patch.object(execution, "bootstrap_intervention_child") as bootstrap:
            with self.assertRaises(ValueError):
                intervention_utility_child_main(["--execute", *arguments], stdin=io.BytesIO(b"x"), clock=lambda: 1)
            bootstrap.assert_not_called()
        with self.assertRaises(ValueError): common_child._actual_registration(intervention_declaration())


if __name__ == "__main__":
    unittest.main()
