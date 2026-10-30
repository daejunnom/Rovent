"""Synthetic masked preparation only; actual strategic utility groups remain0.

Existing Query/2 and native whole-witness factories are reused. Their fixtures
have synthetic caller/build/Rules/CPU/launch observations, not actual action
execution. No causal action bridge, closure producer, NN forward, utility
target, no-step loss, backward, optimizer or product V is run here.
"""

import copy
import tempfile
import unittest
from unittest.mock import patch

from rz_pals_model import strategic_verifier_query as query
from rz_pals_model import strategic_verifier_utility as utility
from rz_pals_model import native_recheck_witness as native
from rz_pals_model import verifier_producer as legacy
from test_strategic_verifier_query import QueryFixture
from test_native_recheck_witness import SyntheticWholeRecheckFixture
from native_source_review_fixture import synthetic_custody_fixture_review_scope as fixture_review_scope


class MaskedUtilityFixture:
    def __init__(self, directory):
        self.base = QueryFixture(directory)
        self.checked = self.base.admit()
        assets = self.checked.raw_assets()
        catalogue = self.checked.catalogue()
        original_before = query._parse(assets["before_result"])
        self.criterion = {"schema": utility.CRITERION_SCHEMA, "policy": utility.POLICY,
            "scope": utility.SCOPE, "cost_axes": list(utility._COST_AXES), "witness_scope": native.SCOPE,
            "requires_conditional_publication": True, "requires_same_pre_result_question_and_prior": True,
            "requires_independent_completed_actions": True, "requires_whole_causal_cost": True}
        self.criterion_raw = query.canonical(self.criterion)
        self.before = {"schema": utility.BEFORE_SCHEMA, "query_sha256": self.checked.sha256,
            "parent": self.checked.audit()["parent"], "catalogue": query.byte_pin(assets["catalogue"]),
            "query_before_result": query.byte_pin(assets["before_result"]),
            "prior_ledger_sha256": original_before["prior_ledger_sha256"], "criterion": query.byte_pin(self.criterion_raw),
            "actions": [{"action_index": index, "semantic_input_sha256": checked.sha256,
                         "registered_profile": {name: query.byte_pin(data) for name, data in
                             assets["profiles"][catalogue["actions"][index]["profile_registration"]].items()}}
                        for index, checked in enumerate(self.checked.action_semantic_inputs())],
            "assurance_scope": utility.CALLER_SCOPE}
        self.values = [{"schema": utility.OBSERVATION_SCHEMA, "query_sha256": self.checked.sha256,
            "before_pair_sha256": "0" * 64, "action_index": index, "semantic_input_sha256": checked.sha256,
            "request": None, "response": None, "whole_witness_sha256": None,
            "state": "missing", "reason": "no actual causal action observation", "assurance_scope": utility.CALLER_SCOPE}
                       for index, checked in enumerate(self.checked.action_semantic_inputs())]
        self.bundles = [{"observation": b"", "request": None, "response": None} for _ in range(2)]
        self.witnesses = [None, None]
        self.seal()

    def seal(self):
        # Fixture-only independent expectations. These pins are not supplied by
        # a producer or advertised as runtime proof; product always masks them.
        self.criterion_raw = query.canonical(self.criterion)
        self.before_raw = query.canonical(self.before)
        for value, bundle in zip(self.values, self.bundles):
            value["before_pair_sha256"] = query.byte_pin(self.before_raw)["sha256"]
            for name in ("request", "response"):
                value[name] = None if bundle[name] is None else query.byte_pin(bundle[name])
            bundle["observation"] = query.canonical(value)
        self.pins = {"query_sha256": self.checked.sha256, "criterion": query.byte_pin(self.criterion_raw),
            "before_pair": query.byte_pin(self.before_raw),
            "observations": [{**{name: None if raw is None else query.byte_pin(raw) for name, raw in bundle.items()},
                              "whole_witness_sha256": value["whole_witness_sha256"]}
                             for value, bundle in zip(self.values, self.bundles)]}

    def cpu(self, index=0, *, partial=False, deadline=False, state="declared_completed"):
        spec = self.checked.catalogue()["actions"][index]
        request = self.base.feedback.request(spec["task"])
        for name in ("baseline_depth", "requested_depth", "max_nodes_per_check", "max_wall_time_ms", "max_output_bytes"):
            request[name] = spec[name]
        request.pop("context_sha256")
        request["context_sha256"] = legacy._hash(legacy.CPU_SCHEMA, request)
        response = self.base.feedback.response(request, partial=partial)
        if deadline:
            response.update(deadline_exceeded=True, elapsed_ms=request["max_wall_time_ms"] + 1)
        self.bundles[index]["request"] = query.canonical(request) + b"\n"
        self.bundles[index]["response"] = query.canonical(response) + b"\n"
        self.values[index].update(state=state, reason=None if state in ("declared_completed", "deferred") else "synthetic unfinished CPU observation")
        self.seal()
        return request, response

    def bind_witness(self, index, checked):
        self.witnesses[index] = checked
        self.values[index]["whole_witness_sha256"] = checked.sha256
        self.seal()

    def admit(self):
        return utility.admit_same_witness_cost_dominance(checked_query=self.checked,
            criterion_bytes=self.criterion_raw, before_pair_bytes=self.before_raw,
            action_observations=self.bundles, whole_witnesses=self.witnesses, expected_pins=self.pins)


class StrategicMaskedUtilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.f = MaskedUtilityFixture(self.temp.name)

    def assert_masked(self, checked):
        self.assertIs(checked.verify(), checked)
        value = checked.observation()
        self.assertEqual(value["status"], "masked_unresolved")
        self.assertEqual(value["reason"], "native_action_causal_bridge_unobserved")
        self.assertIs(value["known"], False)
        self.assertIs(value["mask"], False)
        self.assertIsNone(value["preference"])
        self.assertIsNone(value["sign"])
        self.assertIsNone(value["whole_action_cost"])
        self.assertEqual(value["actual_utility_groups"], 0)
        self.assertIs(value["same_conditional_fact_admitted"], False)
        for name in utility._DENIED:
            self.assertIs(value[name], False)
        return value

    def test_missing_factory_is_immutable_detached_and_always_masked(self):
        checked = self.f.admit()
        self.assert_masked(checked)
        identity = checked.sha256
        view = checked.observation()
        view["observations"][0]["declared_state"] = "declared_completed"
        raw = checked.raw_assets()
        raw["observations"][0]["request"] = b"forged"
        raw["expected_pins"]["query_sha256"] = "0" * 64
        self.assertEqual(checked.sha256, identity)
        self.assertEqual(checked.raw_assets()["observations"][0]["request"], None)
        with self.assertRaises(AttributeError):
            checked._identity = "0" * 64
        for name in ("collate", "pairwise_loss", "frozen_preparation", "select_action"):
            self.assertFalse(hasattr(checked, name))

    def test_public_constructor_and_unchecked_query_cannot_bypass(self):
        with self.assertRaises(ValueError):
            utility.CheckedStrategicUtilityPair()
        self.f.checked = object()
        with self.assertRaisesRegex(ValueError, "exact CheckedStrategicQuery"):
            self.f.admit()

    def test_criterion_is_fixed_policy_not_depth_agreement_or_paper_reward(self):
        for field, value in (("policy", "coverage_reward"), ("cost_axes", ["depth"]),
                             ("requires_conditional_publication", False), ("requires_whole_causal_cost", False)):
            with self.subTest(field=field):
                original = copy.deepcopy(self.f.criterion)
                self.f.criterion[field] = value
                self.f.seal()
                with self.assertRaises(ValueError):
                    self.f.admit()
                self.f.criterion = original
        self.f.seal()
        self.assert_masked(self.f.admit())

    def test_before_result_and_outcome_cost_fields_are_forbidden(self):
        for name, value in (("sign", 1), ("whole_witness_sha256", "1" * 64),
                            ("whole_action_cost", {"whole_elapsed_ms": 1}), ("reward", 0)):
            with self.subTest(name=name):
                self.f.before[name] = value
                self.f.seal()
                with self.assertRaisesRegex(ValueError, "closed fields"):
                    self.f.admit()
                self.f.before.pop(name)
        self.f.seal()

    def test_current_prior_catalogue_and_registered_source_pins_are_fixed(self):
        changes = (("prior_ledger_sha256", "0" * 64), ("catalogue", {"bytes": 1, "sha256": "0" * 64}),
                   ("query_before_result", {"bytes": 1, "sha256": "0" * 64}))
        for name, value in changes:
            with self.subTest(name=name):
                old = copy.deepcopy(self.f.before[name])
                self.f.before[name] = value
                self.f.seal()
                with self.assertRaises(ValueError):
                    self.f.admit()
                self.f.before[name] = old
        self.f.before["actions"][0]["registered_profile"]["source"]["sha256"] = "0" * 64
        self.f.seal()
        with self.assertRaisesRegex(ValueError, "registered profile"):
            self.f.admit()

    def test_duplicate_and_bool_action_indices_are_refused(self):
        for index in (True, 1):
            with self.subTest(index=index):
                original = copy.deepcopy(self.f.before["actions"][0])
                self.f.before["actions"][0] = copy.deepcopy(self.f.before["actions"][1])
                self.f.before["actions"][0]["action_index"] = index
                self.f.seal()
                with self.assertRaises(ValueError):
                    self.f.admit()
                self.f.before["actions"][0] = original
        self.f.seal()

    def test_independent_pin_preserves_original_whitespace_bytes(self):
        checked = self.f.admit()
        self.assertEqual(checked.raw_assets()["criterion"], self.f.criterion_raw)
        self.f.criterion_raw += b"\n"
        with self.assertRaisesRegex(ValueError, "independent original raw pin"):
            self.f.admit()

    def test_duplicate_json_float_and_unknown_schema_are_refused(self):
        variants = (b'{"schema":"x","schema":"y"}', b'{"schema":0.5}',
                    query.canonical({"schema": "old-coverage-utility/1"}))
        for raw in variants:
            with self.subTest(raw=raw):
                original = self.f.criterion_raw
                pin = self.f.pins["criterion"]
                self.f.criterion_raw = raw
                self.f.pins["criterion"] = query.byte_pin(raw)
                with self.assertRaises(ValueError):
                    self.f.admit()
                self.f.criterion_raw, self.f.pins["criterion"] = original, pin

    def test_supplementary_raw_credit_refused_before_query_views(self):
        self.f.criterion_raw = b"x" * utility.MAX_NEW_RAW_BYTES
        with patch.object(query.CheckedStrategicQuery, "verify", side_effect=AssertionError("no query view")):
            with self.assertRaisesRegex(ValueError, "aggregate new immutable raw"):
                self.f.admit()

    def test_oversized_caller_bundle_is_refused_before_key_copy_or_query_views(self):
        self.f.bundles[0] = dict.fromkeys(range(100_000), b"x")
        with patch.object(utility, "set", side_effect=AssertionError("no arbitrary key set copy"), create=True), \
                patch.object(query.CheckedStrategicQuery, "verify", side_effect=AssertionError("no query view")):
            with self.assertRaisesRegex(ValueError, "closed fields"):
                self.f.admit()

    def test_pin_metadata_credit_refused_before_json_serialization(self):
        repeated = "x" * 4096
        self.f.pins["unaccepted_metadata"] = [repeated] * 2048
        with patch.object(query.json, "dumps", side_effect=AssertionError("no oversized serialization")) as serialize:
            with self.assertRaises(ValueError):
                self.f.admit()
        serialize.assert_not_called()

    def test_observation_identity_and_closed_wire_refused(self):
        for name, value in (("query_sha256", "0" * 64), ("semantic_input_sha256", "0" * 64),
                            ("action_index", True), ("state", "successful_utility"),
                            ("final_closure", {"completed": True}), ("whole_elapsed_ms", 1)):
            with self.subTest(name=name):
                original = copy.deepcopy(self.f.values[0])
                self.f.values[0][name] = value
                self.f.seal()
                with self.assertRaises(ValueError):
                    self.f.admit()
                self.f.values[0] = original
        self.f.seal()

    def test_declared_completion_without_original_cpu_bytes_refused(self):
        self.f.values[0].update(state="declared_completed", reason=None)
        self.f.seal()
        with self.assertRaisesRegex(ValueError, "completion lacks"):
            self.f.admit()

    def test_completed_legacy_receipt_never_proves_action_or_whole_cost(self):
        self.f.cpu()
        value = self.assert_masked(self.f.admit())
        first = value["observations"][0]
        self.assertEqual(first["declared_state"], "declared_completed")
        self.assertEqual(first["raw_cpu_receipt"]["status"], "observed")
        self.assertIs(first["raw_cpu_is_whole_action_cost"], False)
        self.assertIs(first["action_completion_admitted"], False)
        self.assertTrue(self.f.admit().raw_assets()["observations"][0]["response"].endswith(b"\n"))

    def test_partial_and_honest_deadline_receipts_remain_unknown_original_bytes(self):
        for state in ("partial", "canceled", "failed"):
            with self.subTest(state=state):
                self.f.cpu(partial=True, deadline=True, state=state)
                original = self.f.bundles[0]["response"]
                checked = self.f.admit()
                value = self.assert_masked(checked)
                self.assertTrue(value["observations"][0]["raw_cpu_receipt"]["deadline_exceeded"])
                self.assertEqual(checked.raw_assets()["observations"][0]["response"], original)

    def test_deadline_raw_cannot_be_declared_completed(self):
        self.f.cpu(partial=True, deadline=True)
        with self.assertRaisesRegex(ValueError, "contradicts raw deadline"):
            self.f.admit()

    def test_request_response_identity_budget_and_profile_drift_refused(self):
        self.f.cpu()
        original = self.f.bundles[0]["response"]
        for name, value in (("context_sha256", "0" * 64), ("cpu_binary_sha256", "0" * 64),
                            ("nodes", 200001), ("resume_kind", "cpu_stack_restored")):
            with self.subTest(name=name):
                response = query._parse(original)
                response[name] = value
                self.f.bundles[0]["response"] = query.canonical(response)
                self.f.seal()
                with self.assertRaises(ValueError):
                    self.f.admit()
        self.f.bundles[0]["response"] = original
        request = query._parse(self.f.bundles[0]["request"])
        request["tt_entries"] = 32
        request.pop("context_sha256")
        request["context_sha256"] = legacy._hash(legacy.CPU_SCHEMA, request)
        self.f.bundles[0]["request"] = query.canonical(request)
        self.f.seal()
        with self.assertRaises(ValueError):
            self.f.admit()

    def test_response_byte_budget_refused_before_receipt_parse(self):
        request, _ = self.f.cpu()
        self.f.bundles[0]["response"] += b" " * request["max_output_bytes"]
        self.f.seal()
        with self.assertRaisesRegex(ValueError, "response output extent"):
            self.f.admit()

    def test_defer_remains_masked_without_zero_reward_or_whole_cost(self):
        self.f.cpu(1, state="deferred")
        value = self.assert_masked(self.f.admit())
        self.assertIn("defer_has_no_independent_conditional_witness_action", value["additional_reasons"])
        self.assertEqual(value["observations"][1]["raw_cpu_receipt"]["status"], "deferred")
        self.assertIsNone(value["whole_action_cost"])

    def test_unchecked_witness_and_unobserved_checked_pin_refused(self):
        self.f.witnesses[0] = object()
        with self.assertRaisesRegex(ValueError, "exact whole native"):
            self.f.admit()
        self.f.witnesses[0] = None
        self.f.values[0]["whole_witness_sha256"] = "0" * 64
        self.f.seal()
        with self.assertRaisesRegex(ValueError, "unobserved whole witness"):
            self.f.admit()

    def test_actual_whole_factory_observation_is_not_action_completion(self):
        with fixture_review_scope("legacy_single_reply"), tempfile.TemporaryDirectory() as directory:
            whole_fixture = SyntheticWholeRecheckFixture(directory)
            whole = whole_fixture.admit()
            self.f.bind_witness(0, whole)
            self.f.bind_witness(1, whole)
            self.f.cpu()
            value = self.assert_masked(self.f.admit())
            self.assertEqual(value["observations"][0]["whole_witness_observation"]["status"], "conditional_cp_observation")
            self.assertIs(value["observations"][0]["whole_witness_observation"]["conditional_counter_lower"], True)
            self.assertIs(value["observations"][0]["whole_witness_observation"]["final_search_envelope_observed"], False)
            self.assertIn("shared_witness_does_not_prove_independent_actions", value["additional_reasons"])
            self.assertIn("conditional_witness_parent_source_or_current_differs_from_query", value["additional_reasons"])

    def test_partial_whole_factory_is_preserved_unresolved(self):
        with fixture_review_scope("legacy_single_reply"), tempfile.TemporaryDirectory() as directory:
            whole_fixture = SyntheticWholeRecheckFixture(directory)
            whole = whole_fixture.admit(rules_missing=True)
            self.f.bind_witness(0, whole)
            value = self.assert_masked(self.f.admit())
            self.assertEqual(value["observations"][0]["whole_witness_observation"]["status"], "unresolved")
            self.assertIn("missing_partial_or_unknown_conditional_witness", value["additional_reasons"])

    def test_checked_parent_mutation_after_admission_refused(self):
        checked = self.f.admit()
        self.f.base.parents.records[self.f.base.index]["input"]["snapshot"]["input_revision"] += 1
        with self.assertRaises(ValueError):
            checked.verify()

    def test_no_pareto_or_loss_consumer_is_exported(self):
        checked = self.f.admit()
        for name in ("pareto_preference", "pairwise_loss", "frozen_preparation", "admit_completed_action"):
            self.assertFalse(hasattr(utility, name))
        self.assert_masked(checked)


if __name__ == "__main__":
    unittest.main()
