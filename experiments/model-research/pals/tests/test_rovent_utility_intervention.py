"""Authored /2 metadata tests; no engine, model, optimizer or live authority."""
import copy
import hashlib
import json
import unittest

from rz_pals_model import rovent_utility_intervention as intervention
from rz_pals_model.rovent_utility_cohort_payload import validate_cohort_profile


def profile():
    pins = {name: "a" * 64 for name in intervention.PROFILE_INPUT_FIELDS if name.endswith("_sha256")}
    return {"domain": "rz-rovent-utility-comparison-cohort/2", "input": {
        **pins, "mode": "cpu_pcr", "requested_recipients": ["proposer", "critic", "researcher"],
        "prelude_ns": 1, "action_ns": 2, "cleanup_ns": 3, "resource_epoch": 1,
        "usage": "collection_candidate", "cost_scope": intervention.COST_SCOPE},
        "target_record": 0, "policy": "frozen_proposal_review_v1"}


class InterventionProfileTests(unittest.TestCase):
    def test_exact_new_dto_roundtrip_and_independent_domain_framing(self):
        value = profile()
        raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(intervention.parse_intervention_profile(raw), value)
        expected = hashlib.sha256(b"rz-rovent-utility-comparison-cohort/2\0" + raw).hexdigest()
        self.assertEqual(intervention.intervention_profile_sha256(value), expected)

    def test_existing_ready_codec_does_not_accept_the_new_intervention(self):
        with self.assertRaises(ValueError):
            validate_cohort_profile(profile())
        previous = {"domain": "rz-rovent-utility-comparison-cohort/1", "input": profile()["input"]}
        with self.assertRaises(ValueError):
            intervention.validate_intervention_profile(previous)

    def test_declared_arms_are_unique_and_supported_not_fake_available_flags(self):
        for arms in (["proposer"], ["critic", "critic"], ["proposer", "validator"], [], [True, "critic"]):
            value = profile()
            value["input"]["requested_recipients"] = arms
            with self.subTest(arms=arms), self.assertRaises(ValueError):
                intervention.validate_intervention_profile(value)
        value = profile()
        value["input"]["mode"] = "cpu_pc"
        with self.assertRaises(ValueError):
            intervention.validate_intervention_profile(value)
        value["input"]["requested_recipients"] = ["proposer", "critic"]
        self.assertEqual(intervention.validate_intervention_profile(value), value)

    def test_target_and_order_have_different_original_seals(self):
        value = profile()
        changed = copy.deepcopy(value)
        changed["target_record"] = 1
        self.assertNotEqual(intervention.intervention_profile_sha256(value), intervention.intervention_profile_sha256(changed))
        changed = copy.deepcopy(value)
        changed["input"]["requested_recipients"].reverse()
        self.assertNotEqual(intervention.intervention_profile_sha256(value), intervention.intervention_profile_sha256(changed))

    def test_unknown_fields_positional_objects_and_duplicate_keys_are_rejected(self):
        value = profile()
        for changed in ([value["domain"], value["input"], 0, value["policy"]], {**value, "known": True}):
            with self.assertRaises(ValueError):
                intervention.validate_intervention_profile(changed)
        raw = json.dumps(value).encode()
        raw = raw.replace(b'"target_record": 0', b'"target_record": 0, "target_record": 1')
        with self.assertRaises(ValueError):
            intervention.parse_intervention_profile(raw)

    def test_exact_integer_limits_and_fixed_policy(self):
        for key, invalid in (("target_record", 64), ("target_record", True), ("policy", "pretend_existing_ready")):
            value = profile()
            value[key] = invalid
            with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                intervention.validate_intervention_profile(value)
        for key, invalid in (("prelude_ns", 0), ("action_ns", 86_400_000_000_000), ("cleanup_ns", 1.5),
                ("resource_epoch", False), ("source_sha256", "A" * 64)):
            value = profile()
            value["input"][key] = invalid
            with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                intervention.validate_intervention_profile(value)
