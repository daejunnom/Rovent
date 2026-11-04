"""Synthetic source/build/event wiring only; no actual compiler/child proof.

The whole positive fixture uses static synthetic source-custody bytes, then
uses the public strict-loader, Repair, Rules, anchor and whole factories.
Its build/process/Rules/CPU records are synthetic caller observations, never
evidence that a compiler, child, model, loss, optimizer or chess replay executed.
Production gates and historical literal profiles are never patched here. New
banks use the separate private registry only in a scoped test context, with
explicit unit byte pins and a synthetic compiled fact. They never read or
enroll current file hashes or claim an actual compiler artifact. Historical
literal reviewed banks remain separate and unchanged.
"""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from rz_pals_model import native_recheck_witness as witness
from rz_pals_model import frozen_producer as frozen
from rz_pals_model import repair_context as repair
from rz_pals_model import semantic_verifier as semantic
from rz_pals_model import training
from rz_pals_model.semantic_verifier import byte_pin, canonical
from test_repair_context import RepairFixture
from test_semantic_verifier import SemanticFixture, tokens
import test_training as fixtures
from native_source_review_fixture import (
    SYNTHETIC_CUSTODY_SOURCE_ID as FIXTURE_SOURCE_ID,
    SYNTHETIC_CUSTODY_TRANSITION_PINS as FIXTURE_TRANSITION_PINS,
    SYNTHETIC_CUSTODY_COMPONENT_PINS as FIXTURE_COMPONENT_PINS,
    SYNTHETIC_CUSTODY_CLOSURE_SHA256 as FIXTURE_CLOSURE_SHA256,
    SYNTHETIC_CUSTODY_COMPILED_FACT as FIXTURE_COMPILED_FACT,
    synthetic_custody_fixture_manifest as fixture_manifest,
    synthetic_custody_fixture_review_scope as fixture_review_scope,
    synthetic_custody_fixture_sources,
)


def jsonl(values):
    return b"".join(canonical(value) + b"\n" for value in values)


class SyntheticBuild:
    """Declared caller observations used to test binding, never a real build."""
    def __init__(self):
        self.binary = "b" * 64
        self.source = {"implementation_sha256": self.binary}
        self.raws = {"engine_source": b"synthetic engine source\n", "native_source": b"synthetic native source\n"}
        self.manifest = {"source_commit": "a" * 40, "files": [
            {"path": path, **byte_pin(self.raws[name])}
            for path, name in zip(witness._SOURCE_PATHS, ("engine_source", "native_source"))]}
        self.build = {"schema": witness.BUILD_SCHEMA, "source_commit": "a" * 40,
                      "actual_build_exit_code": 0, "source_verified_before_and_after_build": True,
                      "binary": {"bytes": 17, "sha256": self.binary},
                      "compiler_artifact": {"reason": "compiler-artifact", "target": {"name": "pals_collect"},
                                            "features": ["pals-collection-onnx"], "fresh": False}}
        self.reseal()

    def reseal(self):
        self.raws["source_manifest"] = canonical(self.manifest)
        self.build["source_manifest"] = byte_pin(self.raws["source_manifest"])
        self.raws["build_registration"] = canonical(self.build)


class SyntheticPolicy:
    """Small policy/source declarations; not complete NativeCollectionFacts."""
    def __init__(self):
        self.policy = copy.deepcopy(witness._POLICY)
        self.registry = {"version": "rz-pals-native-collection-registry/1", "collector_binary_sha256": "b" * 64}
        self.registration = {"version": witness.POLICY_REGISTRATION_SCHEMA,
                             "base_registry_canonical_sha256": byte_pin(canonical(self.registry))["sha256"],
                             "collector_binary_sha256": "b" * 64, "search_policy": copy.deepcopy(self.policy)}
        self.source = {"implementation_sha256": "b" * 64, "native": {
            "independent_registry": self.registry, "pals_search_policy": copy.deepcopy(self.policy),
            "search_version": self.policy["search_identity"], "search_configuration": {}}}
        self.reseal()

    def reseal(self):
        self.raw = canonical(self.registration)
        self.source["native"]["refinement_registration"] = copy.deepcopy(self.registration)
        self.source["native"]["refinement_registration_sha256"] = byte_pin(self.raw)["sha256"]


class SyntheticClosureBuild(SyntheticBuild):
    """Static fixture source text, never current files or a compiled build.

    The complete expected roster is independent of the production helper's
    required-components table. Only individual tests patch its private review
    registry; these pins never register the checkout or authorize execution.
    """
    def __init__(self, lane="legacy_single_reply"):
        super().__init__()
        texts = {
            "crates/rz-search/src/pals/engine.rs": b"// synthetic closed engine fixture\n",
            "crates/rz-arena/src/pals_collect/native.rs": b"// synthetic closed collector fixture\n",
            "crates/rz-search/src/pals/engine/semantic_state.rs": b"// synthetic semantic state fixture\n",
            "crates/rz-search/src/pals/engine/researcher_admission.rs": b"// synthetic researcher admission fixture\n",
            "crates/rz-search/src/pals/engine/researcher_profile.rs": b"// synthetic researcher profile fixture\n",
            "crates/rz-search/src/pals/engine/review_lifecycle.rs": b"// synthetic review lifecycle fixture\n",
            "crates/rz-search/src/pals/engine/cpu_lifecycle.rs": b"// synthetic CPU lifecycle fixture\n",
            "crates/rz-search/src/pals/engine/followup_execution.rs": b"// synthetic followup execution fixture\n",
            "crates/rz-search/src/pals/engine/archive_lifecycle.rs": b"// synthetic archive lifecycle fixture\n",
            "crates/rz-arena/src/pals_collect.rs": b"// synthetic collector entry fixture\n",
            "crates/rz-arena/src/pals_collect/full_line.rs": b"// synthetic full line fixture\n",
            "crates/rz-arena/src/pals_collect/divergence.rs": b"// synthetic divergence fixture\n",
            "crates/rz-arena/src/pals_collect/continuation.rs": b"// synthetic continuation fixture\n",
            "crates/rz-arena/src/pals_collect/producer.rs": b"// synthetic producer fixture\n",
        }
        self.raws.update(engine_source=texts[witness._SOURCE_PATHS[0]], native_source=texts[witness._SOURCE_PATHS[1]])
        components = {path: byte_pin(text) for path, text in texts.items()}
        compiled = byte_pin(b"synthetic independently declared compiled closure fact\n")["sha256"]
        self.profile = {"schema": "rz-pals-native-reviewed-transition-profile/1",
            "review_id": "synthetic-full-closure-only", "lane": lane,
            "transition_sources": {path: copy.deepcopy(components[path]) for path in witness._SOURCE_PATHS},
            "component_pins": components, "compiled_search_implementation_sha256": compiled,
            "closure_sha256": semantic.digest("rz-pals-native-reviewed-transition-closure/1", components)}
        self.manifest["files"] = [{"path": path, **pin} for path, pin in sorted(components.items())]
        self.source["native"] = {"search_implementation_sha256": compiled}
        self.reseal()

    def opt_in(self):
        policy = SyntheticPolicy()
        policy.source["native"]["search_implementation_sha256"] = self.profile["compiled_search_implementation_sha256"]
        self.source = copy.deepcopy(policy.source)
        return policy.raw


class SyntheticReply:
    """Retained raw bytes, not an actual ONNX, engine or OS observation."""
    def __init__(self):
        self.identity = "c" * 64
        self.request = [1, 7]
        self.events = []
        details = [
            ("prepared", {"prepared_before_submit": True, "native_query_kind": "Reply", "producer_metadata_admitted": True}),
            ("physically_completed", {"success": True, "logical_acceptance_inferred": False}),
            ("delivered", {"search_consumed": False}), ("search_consumed", {"search_consumed": True})]
        for tick, (stage, detail) in enumerate(details, 1):
            self.events.append({"domain": "rz-pals-native-call-event/1", "game_id": "synthetic-g1",
                                "process_epoch": 1, "request_sequence": 7, "input_sha256": self.identity,
                                "stage": stage, "observer_elapsed_us": tick, "detail": detail})
        self.outputs = [{"domain": "rz-pals-native-physical-raw/1", "process_epoch": 1, "request_sequence": 7,
                         "input_sha256": self.identity, "physical_completion_confirmed": True, "success": True,
                         "raw": {"representation": "f32_ieee754_bits", "candidate_logits_bits": [0, 0],
                                  "wdl_logits_bits": [0, 0, 0], "divergence_logits_bits": [], "task_logits_bits": None,
                                 "private_latent_bits": [0] * 6144, "prediction_is_future_label": False}}]

    def audit(self, **changes):
        events, outputs = jsonl(self.events), jsonl(self.outputs)
        args = {"events_bytes": events, "output_bytes": outputs, "game_id": "synthetic-g1",
                "input_sha256": self.identity, "native_request": self.request, "candidate_count": 2,
                "expected_pins": {"events": byte_pin(events), "outputs": byte_pin(outputs)}}
        args.update(changes)
        return witness._raw_reply_events(**args)


class SyntheticTrace:
    """Seal/order fixture only; no engine endpoint or parent-chain proof."""
    def __init__(self):
        self.identity = {"game_generation": 1, "search_generation": 2, "root": {"slot": 0, "generation": 1},
                         "root_revision": 3, "repair_record_revision": 4, "repaired_line": 5}
        self.rows = []
        prepared = {"synthetic_descriptor_only": True}
        self.descriptor = byte_pin(canonical([witness.TRACE_DOMAIN, "prepared-descriptor", self.identity, prepared]))["sha256"]
        for tick, (stage, data) in enumerate((("prepared", prepared), ("reply_bound", {"synthetic_binding_only": True}),
                                             ("finished", {"synthetic_endpoint_missing": True})), 1):
            self.rows.append({"domain": witness.TRACE_DOMAIN, "stage": stage, "game_id": "synthetic-g1",
                              "identity": copy.deepcopy(self.identity), "descriptor_sha256": self.descriptor,
                              "payload_sha256": byte_pin(canonical([witness.TRACE_DOMAIN, stage, self.identity, self.descriptor, data]))["sha256"],
                              "observer_elapsed_us": tick, "data": data})

    def read(self):
        raw = jsonl(self.rows)
        return witness._sealed_trace_rows(raw, expected_pin=byte_pin(raw), identity=self.identity, descriptor_sha256=self.descriptor)


class SyntheticEndpoint:
    """Wire/type/parity fixture; not an actual CPU task or Rust Rules replay."""
    def __init__(self):
        self.value_identity = {"semantics":"synthetic-bootstrap", "weights_sha256":None, "training":{"kind":"bootstrap"}}
        self.prepared = {"cpu_condition":"synthetic-exact-owned-condition", "limits":{"max_rounds":2,"max_cpu_nodes":8192,"cpu_depth":2}}
        self.observation = {name:None for name in witness._OBSERVATION_FIELDS}
        self.observation.update(state=9,source=7,epoch=0,value_identity=copy.deepcopy(self.value_identity),
            cpu_condition=self.prepared["cpu_condition"],scope={"kind":"DepthLimited","depth":2,"profile":17,"condition":19},
            score={"kind":"Cpu","value":8,"white_perspective":False,"bound":"ExactWithinSearch"},budget=100,kind="CpuAnalysis",execution=5)
        self.key = {"state":9,"line":None,"question":"AnalyzePosition","root_moves":[],"model":0,"epoch":0,
                    "value_identity":copy.deepcopy(self.value_identity),"checker_identity":None,"cpu_condition":self.prepared["cpu_condition"],
                    "profile":17,"condition":19,"input_revision":0,"requested_depth":2,"node_budget":1024}
        self.evidence = {"kind":"OwnCpu","observation_id":4,"execution_id":5,"admitted_scope":"CompletedIteration", "observation":self.observation,
                         "task":{"status":{"kind":"Completed","observation_id":4},"resumed_from":None,"key":self.key},
                         "task_consumer_observation":"private-consumers-not-observed; engine-borrowed-completed-provenance-only"}
        self.endpoint = {"state":9,"situation":{"slot":1,"generation":2},"rules_state_sha256":"e"*64,"board_fen":"synthetic FEN", "evidence":self.evidence}

    def read(self):
        return witness._endpoint(self.endpoint,prepared=self.prepared,value_identity=self.value_identity,root_white=True,plies=5)


class NativeRecheckClosureConsumerTests(unittest.TestCase):
    def test_new_pair_requires_full_literal_closure_and_returns_detached_metadata_only(self):
        from rz_pals_model import native_source_review as reviewed
        fixture = SyntheticClosureBuild()
        policy = fixture.opt_in()
        with patch.object(reviewed, "REVIEWED_CLOSURE_PROFILES", ()):
            with self.assertRaisesRegex(witness.UnsupportedNativeRecheck, "unreviewed_opt_in_collector_transition_source_pair"):
                witness._source_review(fixture.raws, fixture.source, policy)
        registered = copy.deepcopy(fixture.profile)
        with patch.object(reviewed, "REVIEWED_CLOSURE_PROFILES", (registered,)):
            result = witness._source_review(fixture.raws, fixture.source, policy)
            proof = result["reviewed_transition_closure"]
            self.assertEqual(set(proof), {"schema", "review_id", "lane", "transition_sources", "component_pins",
                "compiled_search_implementation_sha256", "closure_sha256", "unreviewed_manifest_files",
                "historical_metadata_only", "execution_authorized", "original_native_authority_reissued",
                "rules_proof_authorized", "training_target_authorized"})
            self.assertEqual(proof["lane"], "legacy_single_reply")
            self.assertEqual(proof["component_pins"], fixture.profile["component_pins"])
            self.assertEqual(result["reviewed_transition_sources"], fixture.profile["transition_sources"])
            self.assertEqual(result["source_review_profile"], fixture.profile["review_id"])
            self.assertIs(proof["historical_metadata_only"], True)
            for key in ("execution_authorized", "original_native_authority_reissued", "rules_proof_authorized", "training_target_authorized"):
                self.assertIs(proof[key], False)
            original_bytes = canonical(result)
            proof["component_pins"][witness._SOURCE_PATHS[0]]["sha256"] = "0" * 64
            self.assertEqual(canonical(witness._source_review(fixture.raws, fixture.source, policy)), original_bytes)
            self.assertEqual(registered, fixture.profile)

    def test_full_helper_roster_and_actual_compiled_fact_are_required_by_consumer(self):
        from rz_pals_model import native_source_review as reviewed
        helper = "crates/rz-search/src/pals/engine/semantic_state.rs"
        for change in ("replacement", "missing_helper", "profile_omission", "compiled_mismatch", "compiled_missing", "wrong_lane"):
            fixture = SyntheticClosureBuild()
            policy = fixture.opt_in()
            registered = copy.deepcopy(fixture.profile)
            if change == "replacement":
                next(row for row in fixture.manifest["files"] if row["path"] == helper).update(byte_pin(b"changed helper fixture\n"))
            elif change == "missing_helper":
                fixture.manifest["files"] = [row for row in fixture.manifest["files"] if row["path"] != helper]
            elif change == "profile_omission":
                del registered["component_pins"][helper]
                registered["closure_sha256"] = semantic.digest("rz-pals-native-reviewed-transition-closure/1", registered["component_pins"])
            elif change == "compiled_mismatch":
                fixture.source["native"]["search_implementation_sha256"] = "e" * 64
            elif change == "compiled_missing":
                del fixture.source["native"]["search_implementation_sha256"]
            else:
                registered["lane"] = "default_disabled"
            fixture.reseal()
            boundary = {"replacement": "helper closure differs", "missing_helper": "helper closure differs",
                "profile_omission": "omits a required source helper", "compiled_mismatch": "compiled search closure differs",
                "compiled_missing": "original_compiled_search_closure_fact_missing", "wrong_lane": "unreviewed_opt_in_collector_transition_source_pair"}[change]
            with self.subTest(change=change), patch.object(reviewed, "REVIEWED_CLOSURE_PROFILES", (registered,)):
                with self.assertRaisesRegex(ValueError, boundary):
                    witness._source_review(fixture.raws, fixture.source, policy)

    def test_build_binary_source_and_registered_policy_gates_precede_new_closure(self):
        from rz_pals_model import native_source_review as reviewed
        for change in ("exit", "verified", "binary", "unbound_component", "policy"):
            fixture = SyntheticClosureBuild()
            policy = fixture.opt_in()
            if change == "exit":
                fixture.build["actual_build_exit_code"] = False
            elif change == "verified":
                fixture.build["source_verified_before_and_after_build"] = False
            elif change == "binary":
                fixture.build["binary"]["sha256"] = "e" * 64
            elif change == "unbound_component":
                fixture.raws["native_source"] += b"// not in original manifest\n"
            else:
                fixture.source["native"]["refinement_registration_sha256"] = "e" * 64
            fixture.reseal()
            boundary = {"exit": "successful original build observation", "verified": "successful original build observation",
                "binary": "registered collector binary binding", "unbound_component": "source bytes absent from original",
                "policy": "policy registration/base registry/binary binding"}[change]
            with self.subTest(change=change), patch.object(reviewed, "review_transition_sources",
                    side_effect=AssertionError("source closure must not hide an earlier failed gate")) as lookup:
                with self.assertRaisesRegex(ValueError, boundary):
                    witness._source_review(fixture.raws, fixture.source, policy)
                lookup.assert_not_called()

    def test_new_pair_cannot_hybridize_one_registered_component(self):
        from rz_pals_model import native_source_review as reviewed
        for name, path in zip(("engine_source", "native_source"), witness._SOURCE_PATHS):
            fixture = SyntheticClosureBuild()
            policy = fixture.opt_in()
            fixture.raws[name] += b"// separately bound successor fixture\n"
            next(row for row in fixture.manifest["files"] if row["path"] == path).update(byte_pin(fixture.raws[name]))
            fixture.reseal()
            with self.subTest(component=name), patch.object(reviewed, "REVIEWED_CLOSURE_PROFILES", (copy.deepcopy(fixture.profile),)):
                with self.assertRaisesRegex(witness.UnsupportedNativeRecheck, "unreviewed_opt_in_collector_transition_source_pair"):
                    witness._source_review(fixture.raws, fixture.source, policy)

    def test_historical_literal_lookup_remains_detached_and_does_not_enroll_a_closure(self):
        from rz_pals_model import native_source_review as reviewed
        expected = copy.deepcopy(REVIEWED_WHOLE_SOURCE_PINS)
        with patch.object(reviewed, "review_transition_sources", side_effect=AssertionError("historical lookup must stay separate")) as lookup:
            name, retained = witness._reviewed_source_pair(expected)
            before = canonical([name, expected])
            retained[witness._SOURCE_PATHS[0]]["sha256"] = "0" * 64
            self.assertEqual(canonical(list(witness._reviewed_source_pair(expected))), before)
            lookup.assert_not_called()


class NativeRecheckBoundaryTests(unittest.TestCase):
    def test_policy_is_exact_detached_four_field_identity(self):
        value = copy.deepcopy(witness._POLICY)
        admitted = witness._policy_identity(value)
        admitted["conditions_sha256"][0] ^= 1
        self.assertEqual(value, witness._POLICY)

    def test_policy_digest_rejects_bool_range_and_extent(self):
        for invalid in (True, -1, 256, 1.0):
            with self.subTest(invalid=invalid):
                value = copy.deepcopy(witness._POLICY)
                value["conditions_sha256"][0] = invalid
                with self.assertRaises(ValueError):
                    witness._policy_identity(value)
        for invalid in ([], [0] * 31, [0] * 33, "bea44b7e"):
            value = copy.deepcopy(witness._POLICY)
            value["conditions_sha256"] = invalid
            with self.assertRaises(ValueError):
                witness._policy_identity(value)

    def test_policy_rejects_unknown_disabled_and_extra_field(self):
        for policy in ("Disabled", "same-repaired-line-once-v1", "actual_opponent_continuation_v1",
                       "actual-opponent-continuation-v1", "frozen_model_wdl_v2",
                       "iterative_frozen_model_wdl_v2", None):
            value = copy.deepcopy(witness._POLICY)
            value["policy"] = policy
            with self.assertRaises(witness.UnsupportedNativeRecheck):
                witness._policy_identity(value)
        actual_continuation = {"version": "pals-post-repair-continuation/1",
            "policy": "actual_opponent_continuation_v1",
            "search_identity": "pals-restricted-refinement-post-repair-continuation/1",
            "conditions_sha256": list(bytes.fromhex("876c7104c28183131243101df86c386865bf903f26545edde01d48de4403d4d1"))}
        with self.assertRaises(witness.UnsupportedNativeRecheck):
            witness._policy_identity(actual_continuation)
        value = copy.deepcopy(witness._POLICY)
        value["completed"] = True
        with self.assertRaises(ValueError):
            witness._policy_identity(value)

    def test_policy_registration_binds_original_raw_bytes(self):
        fixture = SyntheticPolicy()
        result = witness._registered_policy(fixture.raw, fixture.source)
        self.assertEqual(result["registration"], byte_pin(fixture.raw))
        self.assertIn("selection_only", result["assurance"])
        spaced = json.dumps(fixture.registration, indent=2).encode()
        with self.assertRaises(ValueError):
            witness._registered_policy(spaced, fixture.source)

    def test_registration_rejects_base_registry_binary_or_raw_pin_drift(self):
        for axis in ("base_registry", "binary", "raw_pin", "actual_marker"):
            with self.subTest(axis=axis):
                fixture = SyntheticPolicy()
                if axis == "base_registry":
                    fixture.registry["collector_binary_sha256"] = "e" * 64
                elif axis == "binary":
                    fixture.source["implementation_sha256"] = "e" * 64
                elif axis == "raw_pin":
                    fixture.source["native"]["refinement_registration_sha256"] = "e" * 64
                else:
                    fixture.source["native"]["pals_search_policy"]["conditions_sha256"][0] ^= 1
                with self.assertRaises(ValueError):
                    witness._registered_policy(fixture.raw, fixture.source)

    def test_all_policy_owners_reject_contradictory_version_or_alias(self):
        for owner in ("source", "native", "configuration", "registry"):
            for axis in ("search_version", "refinement_policy"):
                with self.subTest(owner=owner, axis=axis):
                    fixture = SyntheticPolicy()
                    selected = {"source": fixture.source, "native": fixture.source["native"],
                                "configuration": fixture.source["native"]["search_configuration"],
                                "registry": fixture.registry}[owner]
                    selected[axis] = "Disabled"
                    # Bind deliberate registry mutations to isolate the policy
                    # check; this never supplies actual build/collection proof.
                    if owner == "registry":
                        fixture.registration["base_registry_canonical_sha256"] = byte_pin(canonical(fixture.registry))["sha256"]
                        fixture.reseal()
                    with self.assertRaises(ValueError):
                        witness._registered_policy(fixture.raw, fixture.source)

    def test_registration_duplicate_json_key_is_malformed(self):
        fixture = SyntheticPolicy()
        raw = fixture.raw[:-1] + b',"version":"rz-pals-native-refinement-registration/1"}'
        with self.assertRaises(ValueError):
            witness._registered_policy(raw, fixture.source)

    def test_policy_registration_nonempty_and_native_byte_cap(self):
        fixture = SyntheticPolicy()
        for raw in (b"", "not bytes", b" " * (witness.MAX_POLICY_REGISTRATION_BYTES + 1)):
            with self.assertRaises(ValueError):
                witness._registered_policy(raw, fixture.source)

    def test_equal_policy_declaration_outside_actual_native_location_is_unsupported(self):
        for owner in ("source", "configuration", "registry"):
            for axis in ("pals_search_policy", "search_version", "refinement_registration", "refinement_registration_sha256"):
                with self.subTest(owner=owner, axis=axis):
                    fixture = SyntheticPolicy()
                    selected = {"source": fixture.source, "configuration": fixture.source["native"]["search_configuration"],
                                "registry": fixture.registry}[owner]
                    selected[axis] = copy.deepcopy(fixture.source["native"][axis])
                    if owner == "registry":
                        fixture.registration["base_registry_canonical_sha256"] = byte_pin(canonical(fixture.registry))["sha256"]
                        fixture.reseal()
                    with self.assertRaises(witness.UnsupportedNativeRecheck):
                        witness._registered_policy(fixture.raw, fixture.source)

    def test_original_manifest_binds_both_source_bytes_before_review(self):
        fixture = SyntheticBuild()
        pair, observation = witness._original_build_sources(fixture.raws, fixture.source)
        self.assertEqual(pair, {path: byte_pin(fixture.raws[name]) for path, name in
                               zip(witness._SOURCE_PATHS, ("engine_source", "native_source"))})
        self.assertIn("caller_build_observation", observation["assurance"])
        self.assertNotIn("execution_certified", observation)

    def test_build_rejects_bool_exit_and_failed_or_unverified_observation(self):
        for key, bad in (("actual_build_exit_code", False), ("actual_build_exit_code", 1),
                         ("source_verified_before_and_after_build", False)):
            fixture = SyntheticBuild()
            fixture.build[key] = bad
            fixture.reseal()
            with self.assertRaises(ValueError):
                witness._original_build_sources(fixture.raws, fixture.source)

    def test_unknown_engine_cannot_hide_unbound_native_source(self):
        fixture = SyntheticBuild()
        fixture.raws["engine_source"] = b"unknown engine source"
        fixture.manifest["files"][0].update(byte_pin(fixture.raws["engine_source"]))
        fixture.raws["native_source"] = b"unbound native source"
        fixture.reseal()
        with self.assertRaisesRegex(ValueError, "absent from original"):
            witness._original_build_sources(fixture.raws, fixture.source)

    def test_registered_binary_mismatch_precedes_review(self):
        fixture = SyntheticBuild()
        fixture.source["implementation_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "binary binding"):
            witness._original_build_sources(fixture.raws, fixture.source)

    def test_synthetic_hashes_are_not_enrolled_without_literal_review(self):
        fixture = SyntheticBuild()
        pair, _ = witness._original_build_sources(fixture.raws, fixture.source)
        self.assertIsInstance(witness._REVIEWED_OPT_IN_SOURCE_PROFILES, tuple)
        with self.assertRaises(witness.UnsupportedNativeRecheck):
            witness._reviewed_source_pair(pair)

    def test_selected_identity_successor_preserves_prior_single_reply_review_and_rejects_drift(self):
        old = {
            witness._SOURCE_PATHS[0]: {
                "bytes": 355881, "sha256": "32e393b4ad83bfd33a1b2205cd110cdc0cf007909e7e9c7f617819ccf5c4e93c"},
            witness._SOURCE_PATHS[1]: {
                "bytes": 220199, "sha256": "eea0654a5c11d6e0d69b5c5df71363a8e2d8f4fb7ad02cf5d766fb4b06ec456f"},
        }
        _, retained = witness._reviewed_source_pair(old)
        self.assertEqual(retained, old)
        _, successor = witness._reviewed_source_pair(REVIEWED_WHOLE_SOURCE_PINS)
        self.assertEqual(successor, REVIEWED_WHOLE_SOURCE_PINS)
        for path in witness._SOURCE_PATHS:
            changed = copy.deepcopy(REVIEWED_WHOLE_SOURCE_PINS)
            changed[path]["bytes"] += 1
            with self.assertRaises(witness.UnsupportedNativeRecheck):
                witness._reviewed_source_pair(changed)

    def test_optional_review_pair_remains_a_separate_historical_profile(self):
        name, reviewed = witness._reviewed_source_pair(REVIEWED_OPTIONAL_REVIEW_WHOLE_SOURCE_PINS)
        self.assertEqual(name, "opt_in_single_reply_followup_optional_review_ce40b6f9_native_02af59b9")
        self.assertEqual(reviewed, REVIEWED_OPTIONAL_REVIEW_WHOLE_SOURCE_PINS)
        self.assertNotEqual(reviewed, REVIEWED_CURRENT_WHOLE_SOURCE_PINS)
        reviewed[witness._SOURCE_PATHS[0]]["sha256"] = "0" * 64
        self.assertEqual(witness._reviewed_source_pair(REVIEWED_OPTIONAL_REVIEW_WHOLE_SOURCE_PINS)[1],
                         REVIEWED_OPTIONAL_REVIEW_WHOLE_SOURCE_PINS)

    def test_followup_core_review_preserves_history_and_rejects_component_hybrids(self):
        _, historical = witness._reviewed_source_pair(REVIEWED_WHOLE_SOURCE_PINS)
        _, current = witness._reviewed_source_pair(REVIEWED_CURRENT_WHOLE_SOURCE_PINS)
        self.assertEqual(historical, REVIEWED_WHOLE_SOURCE_PINS)
        self.assertEqual(current, REVIEWED_CURRENT_WHOLE_SOURCE_PINS)
        _, pre_lint = witness._reviewed_source_pair(REVIEWED_PRE_LINT_WHOLE_SOURCE_PINS)
        self.assertEqual(pre_lint, REVIEWED_PRE_LINT_WHOLE_SOURCE_PINS)
        _, lint = witness._reviewed_source_pair(REVIEWED_LINT_WHOLE_SOURCE_PINS)
        self.assertEqual(lint, REVIEWED_LINT_WHOLE_SOURCE_PINS)
        _, controlled = witness._reviewed_source_pair(REVIEWED_CONTROLLED_WHOLE_SOURCE_PINS)
        self.assertEqual(controlled, REVIEWED_CONTROLLED_WHOLE_SOURCE_PINS)
        _, fallback = witness._reviewed_source_pair(REVIEWED_CONTROLLED_FALLBACK_WHOLE_SOURCE_PINS)
        self.assertEqual(fallback, REVIEWED_CONTROLLED_FALLBACK_WHOLE_SOURCE_PINS)
        _, live_binding = witness._reviewed_source_pair(REVIEWED_LIVE_BINDING_WHOLE_SOURCE_PINS)
        self.assertEqual(live_binding, REVIEWED_LIVE_BINDING_WHOLE_SOURCE_PINS)
        _, observed = witness._reviewed_source_pair(REVIEWED_OBSERVED_PROGRESS_WHOLE_SOURCE_PINS)
        self.assertEqual(observed, REVIEWED_OBSERVED_PROGRESS_WHOLE_SOURCE_PINS)
        _, ordered = witness._reviewed_source_pair(REVIEWED_ORDERED_SCOPE_WHOLE_SOURCE_PINS)
        self.assertEqual(ordered, REVIEWED_ORDERED_SCOPE_WHOLE_SOURCE_PINS)
        self.assertNotEqual(historical, current)
        for path in witness._SOURCE_PATHS:
            with self.subTest(component=path):
                hybrid = copy.deepcopy(current)
                hybrid[path] = copy.deepcopy(historical[path])
                with self.assertRaises(witness.UnsupportedNativeRecheck):
                    witness._reviewed_source_pair(hybrid)
                hybrid = copy.deepcopy(current)
                hybrid[path] = copy.deepcopy(ordered[path])
                with self.assertRaises(witness.UnsupportedNativeRecheck):
                    witness._reviewed_source_pair(hybrid)
                changed = copy.deepcopy(current)
                changed[path]["bytes"] += 1
                with self.assertRaises(witness.UnsupportedNativeRecheck):
                    witness._reviewed_source_pair(changed)

    def test_unchecked_repair_anchor_cannot_be_constructed(self):
        with self.assertRaises(ValueError):
            witness.CheckedNativeRecheckRepairAnchor()
        with self.assertRaisesRegex(ValueError, "exact checked"):
            witness._validate_anchor(object(), {}, {})

    def test_additional_trace_is_bound_to_original_complete_receipt(self):
        raw = b"synthetic recheck trace\n"
        receipt = canonical({"version": "rz-pals-own-collector/1", "artifacts": {
            "native-recheck-traces.jsonl": byte_pin(raw), "unrelated-retained.jsonl": byte_pin(b"preserved\n")}})
        result = witness._original_receipt_asset(receipt, byte_pin(receipt), "native-recheck-traces.jsonl", raw, byte_pin(raw))
        self.assertEqual(result, byte_pin(raw))
        with self.assertRaises(ValueError):
            witness._original_receipt_asset(receipt, byte_pin(receipt), "native-recheck-traces.jsonl", raw + b"late", byte_pin(raw + b"late"))

    def test_sanitized_or_retroactive_receipt_cannot_replace_original_bytes(self):
        raw = b"synthetic recheck trace\n"
        original = canonical({"version": "rz-pals-own-collector/1", "artifacts": {"unrelated-retained.jsonl": byte_pin(b"preserved\n")}})
        changed = canonical({"version": "rz-pals-own-collector/1", "artifacts": {"native-recheck-traces.jsonl": byte_pin(raw)}})
        with self.assertRaises(ValueError):
            witness._original_receipt_asset(changed, byte_pin(original), "native-recheck-traces.jsonl", raw, byte_pin(raw))
        with self.assertRaises(witness.UnsupportedNativeRecheck):
            witness._original_receipt_asset(original, byte_pin(original), "native-recheck-traces.jsonl", raw, byte_pin(raw))

    def test_consumed_raw_reply_does_not_admit_endpoint_or_target(self):
        result = SyntheticReply().audit()
        self.assertEqual(result["status"], "consumed_raw_observation")
        self.assertEqual(result["scope"], "raw_reply_attribution_only")
        for field in witness._DENIED_AUTHORITIES:
            self.assertIs(result[field], False)

    def test_trace_seals_are_only_attributed_rows_without_endpoint_admission(self):
        fixture = SyntheticTrace()
        rows = fixture.read()
        self.assertEqual([row["stage"] for row in rows], ["prepared", "reply_bound", "finished"])
        rows[0]["data"]["synthetic_descriptor_only"] = False
        self.assertIs(fixture.rows[0]["data"]["synthetic_descriptor_only"], True)

    def test_partial_or_finished_only_trace_retains_incomplete_raw_rows(self):
        for extent in ("prepared_only", "bound_only", "finished_only"):
            fixture = SyntheticTrace()
            fixture.rows = fixture.rows[:1] if extent == "prepared_only" else fixture.rows[:2] if extent == "bound_only" else fixture.rows[-1:]
            self.assertGreater(len(fixture.read()), 0)
        fixture = SyntheticTrace()
        fixture.rows = []
        self.assertEqual(fixture.read(), [])

    def test_empty_optional_journal_does_not_relax_nonempty_exact_jsonl(self):
        self.assertEqual(witness._optional_jsonl_rows(b""), [])
        for raw in (b'{"stage":"prepared"}', b"\n"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                witness._optional_jsonl_rows(raw)

    def test_trace_payload_or_descriptor_mutation_cannot_be_resealed_as_observation(self):
        fixture = SyntheticTrace()
        fixture.rows[-1]["data"]["synthetic_endpoint_missing"] = False
        with self.assertRaises(ValueError):
            fixture.read()
        fixture = SyntheticTrace()
        fixture.rows[0]["descriptor_sha256"] = "e" * 64
        with self.assertRaises(ValueError):
            fixture.read()

    def test_duplicate_or_out_of_order_trace_stage_is_not_partial_evidence(self):
        for mutation in ("duplicate", "order", "clock"):
            fixture = SyntheticTrace()
            if mutation == "duplicate":
                fixture.rows.append(copy.deepcopy(fixture.rows[-1]))
            elif mutation == "order":
                fixture.rows[1], fixture.rows[2] = fixture.rows[2], fixture.rows[1]
            else:
                fixture.rows[-1]["observer_elapsed_us"] = 0
            with self.assertRaises(ValueError):
                fixture.read()

    def test_trace_unsigned_identity_does_not_accept_boolean_alias(self):
        fixture = SyntheticTrace()
        fixture.identity["root"]["slot"] = False
        with self.assertRaises(ValueError):
            fixture.read()

    def test_missing_and_prepared_only_reply_stay_unresolved(self):
        for count in (0, 1):
            fixture = SyntheticReply()
            fixture.events = fixture.events[:count]
            fixture.outputs = []
            self.assertEqual(fixture.audit()["status"], "unresolved")

    def test_delivery_without_consumption_stays_unresolved(self):
        fixture = SyntheticReply()
        fixture.events.pop()
        self.assertEqual(fixture.audit()["status"], "unresolved")

    def test_canceled_or_partial_reply_is_not_consumed(self):
        fixture = SyntheticReply()
        fixture.events[-1]["stage"] = "logically_rejected"
        fixture.events[-1]["detail"] = {"reason": "Canceled", "search_consumed": False}
        self.assertEqual(fixture.audit()["status"], "unresolved")
        fixture.events = fixture.events[:2]
        self.assertEqual(fixture.audit()["status"], "unresolved")

    def test_failed_physical_output_stays_unresolved_and_cannot_be_delivered(self):
        fixture = SyntheticReply()
        fixture.events = fixture.events[:2]
        fixture.events[-1]["detail"]["success"] = False
        fixture.outputs[0]["success"] = False
        fixture.outputs[0]["raw"] = {"failure": "synthetic backend failure", "kind": "Backend"}
        self.assertEqual(fixture.audit()["status"], "unresolved")
        fixture.events.append({**copy.deepcopy(fixture.events[-1]), "stage": "delivered", "observer_elapsed_us": 3,
                               "detail": {"search_consumed": False}})
        with self.assertRaises(ValueError):
            fixture.audit()

    def test_misordered_duplicate_or_regressed_events_are_malformed(self):
        for mode in ("order", "duplicate", "clock"):
            fixture = SyntheticReply()
            if mode == "order":
                fixture.events[1], fixture.events[2] = fixture.events[2], fixture.events[1]
            elif mode == "duplicate":
                fixture.events.append(copy.deepcopy(fixture.events[-1]))
            else:
                fixture.events[-1]["observer_elapsed_us"] = 0
            with self.assertRaises(ValueError):
                fixture.audit()

    def test_request_boolean_alias_is_rejected_on_each_raw_axis(self):
        fixture = SyntheticReply()
        with self.assertRaises(ValueError):
            fixture.audit(native_request=[True, 7])
        for axis in ("events", "outputs"):
            fixture = SyntheticReply()
            getattr(fixture, axis)[0]["process_epoch"] = True
            with self.assertRaises(ValueError):
                fixture.audit()

    def test_wrong_input_or_request_cannot_supply_selected_reply(self):
        for axis in ("events", "outputs"):
            fixture = SyntheticReply()
            getattr(fixture, axis)[0]["input_sha256"] = "e" * 64
            with self.assertRaises(ValueError):
                fixture.audit()
            fixture = SyntheticReply()
            getattr(fixture, axis)[0]["request_sequence"] = 8
            with self.assertRaises(ValueError):
                fixture.audit()

    def test_physical_success_or_completion_contradiction_rejects_consumption(self):
        for axis in ("success", "physical_completion_confirmed"):
            fixture = SyntheticReply()
            fixture.outputs[0][axis] = False
            with self.assertRaises(ValueError):
                fixture.audit()

    def test_rejected_prepared_metadata_cannot_reach_physical_or_consumed_work(self):
        fixture = SyntheticReply()
        fixture.events[0]["detail"]["producer_metadata_admitted"] = False
        with self.assertRaises(ValueError):
            fixture.audit()

    def test_raw_reply_predictions_require_finite_exact_heads(self):
        for axis, bad in (("candidate_logits_bits", [0]), ("wdl_logits_bits", [0, 0, 0x7F800000]),
                          ("private_latent_bits", [0]), ("prediction_is_future_label", True),
                          ("task_logits_bits", [0] * 7)):
            fixture = SyntheticReply()
            fixture.outputs[0]["raw"][axis] = bad
            with self.assertRaises(ValueError):
                fixture.audit()

    def test_reply_critic_requires_exact_empty_divergence_head(self):
        fixture = SyntheticReply()
        self.assertEqual(fixture.audit()["status"],"consumed_raw_observation")
        for consumed in (True,False):
            for bad in (None,[0],[[]],{},0,False,""):
                with self.subTest(consumed=consumed,bad=bad):
                    fixture = SyntheticReply()
                    if not consumed:
                        fixture.events = fixture.events[:3]
                    fixture.outputs[0]["raw"]["divergence_logits_bits"] = bad
                    with self.assertRaisesRegex(ValueError,"exact empty divergence head"):
                        fixture.audit()

    def test_independent_byte_pin_mismatch_is_not_an_unresolved_result(self):
        fixture = SyntheticReply()
        events, outputs = jsonl(fixture.events), jsonl(fixture.outputs)
        pins = {"events": byte_pin(events), "outputs": byte_pin(outputs)}
        pins["outputs"]["sha256"] = "e" * 64
        with self.assertRaises(ValueError):
            fixture.audit(expected_pins=pins)


class NativeRecheckWholeGateTests(unittest.TestCase):
    def test_whole_witness_constructor_and_factory_refuse_unchecked_anchor(self):
        with self.assertRaises(ValueError):
            witness.CheckedNativeRecheckWitness()
        with self.assertRaisesRegex(ValueError,"exact factory-checked"):
            witness.admit_native_recheck_witness(anchor=object(),trace_bytes=b"",expected_pins={})

    def test_actual_strict_parent_prepared_binding_reused_without_opt_in_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RepairFixture(directory)
            causal = fixture.admit()
            identity = fixture.rows[1]["input"]["sha256"]
            input_raw = next(raw for raw,value in witness.native._rows(fixture.artifacts["inputs.jsonl"]) if value["sha256"] == identity)
            sidecar_raw = next(raw for raw,value in witness.native._rows(fixture.artifacts["native-inputs.jsonl"]) if value["input_sha256"] == identity)
            lineage_raw = next(raw for raw,value in witness.native._rows(fixture.artifacts["input-lineage.jsonl"]) if value["input_sha256"] == identity)
            anchors = {"input_row_sha256":byte_pin(input_raw)["sha256"],
                       "sidecar_row_sha256":byte_pin(sidecar_raw)["sha256"],
                       "lineage_row_sha256":byte_pin(lineage_raw)["sha256"],
                       "sidecar_sha256":fixture.sidecars[1]["sha256"],"canonical_tensor_sha256":fixture.sidecars[1]["canonical_tensor_sha256"]}
            result = witness._prepared_input_rows(causal,input_sha256=identity,native_request=[1,8],kind="Repair",anchors=anchors)
            self.assertEqual(result[0],fixture.repair_index)
            self.assertEqual(result[3]["sha256"],fixture.journals[1]["sha256"])
            # Recovered synthetic byte wiring is not a reviewed opt-in source,
            # engine execution or whole repaired-line authority.
            unknown = {path:byte_pin(value) for path,value in zip(witness._SOURCE_PATHS,(b"synthetic unreviewed engine",b"synthetic unreviewed native"))}
            with self.assertRaises(witness.UnsupportedNativeRecheck): witness._reviewed_source_pair(unknown)
            mutated = copy.deepcopy(anchors)
            mutated["lineage_row_sha256"] = "e"*64
            with self.assertRaisesRegex(ValueError,"descriptor byte anchor"):
                witness._prepared_input_rows(causal,input_sha256=identity,native_request=[1,8],kind="Repair",anchors=mutated)
            with self.assertRaises(ValueError):
                witness._prepared_input_rows(causal,input_sha256=identity,native_request=[True,8],kind="Repair",anchors=anchors)

    def test_finite_policy_rank_preserves_rules_ties_and_signed_zero(self):
        ranked = witness._ranked_policy_moves({"candidate_logits_bits":[0x80000000,0,0]},[1025,1026,1027])
        self.assertEqual(ranked,[1026,1027,1025])
        with self.assertRaises(ValueError):
            witness._ranked_policy_moves({"candidate_logits_bits":[0,0]},[1025,1025])
        with self.assertRaises(ValueError):
            witness._ranked_policy_moves({"candidate_logits_bits":[0x7f800000]},[1025])

    def test_whole_line_rules_reuses_exact_registered_semantic_capability(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RepairFixture(directory)
            causal, check = fixture.admit(), fixture.proposal_check
            final = check.rules_receipt()["claimed_line"]["final_state"]
            endpoint = {name:final[name] for name in ("rules_state_sha256","board_fen")}
            arguments = {"prefix":[],"claim":fixture.lineages[1]["proposal"],"endpoint":endpoint,"expected_sha256":check.sha256}
            receipt = witness._rules_line(check,causal,**arguments)
            self.assertEqual(receipt["claimed_line"]["claim_truth"],"unknown")
            # The real admission factory validates this finite synthetic wire;
            # no Rust child was launched by the fixture and no claim is proven.
            for axis in ("prefix","claim","endpoint","independent_pin","unchecked"):
                local, selected = copy.deepcopy(arguments), check
                if axis == "prefix": local["prefix"] = fixture.lineages[1]["virtual_prefix"]
                elif axis == "claim": local["claim"] = local["claim"][:-1]
                elif axis == "endpoint": local["endpoint"]["rules_state_sha256"] = "e"*64
                elif axis == "independent_pin": local["expected_sha256"] = "e"*64
                else: selected = object()
                with self.subTest(axis=axis), self.assertRaises(ValueError):
                    witness._rules_line(selected,causal,**local)

    def test_last_repair_request_is_not_a_complete_suffix_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RepairFixture(directory)
            causal = fixture.admit()
            prefix = fixture.lineages[1]["virtual_prefix"]
            repaired = prefix + fixture.lineages[1]["proposal"][len(prefix):]
            data = {"repaired":repaired,"refutation":fixture.lineages[1]["counterexample"],
                    "parent_repair_chain":{"initial_prefix":prefix,"initial_prefix_len":len(prefix),
                        "full_repaired_line":repaired,"calls":[{"process_epoch":1,"request_sequence":8}],
                        "journal_observation":"independent-exact-producer-journal-entry-required; raw-entry-SHA-not-returned-by-producer-API"}}
            with self.assertRaisesRegex(ValueError,"entire actual Repair suffix"):
                witness._parent_chain(causal,data,{})

    def test_endpoint_stored_completion_and_parity_are_not_proof_or_target(self):
        result = SyntheticEndpoint().read()
        self.assertEqual(result["status"],"completed_cp")
        self.assertEqual((result["raw_score"],result["root_score"]),(8,-8))
        self.assertNotIn("wdl",result)
        self.assertNotIn("completion",result)
        self.assertNotIn("private_consumers",result)

    def test_endpoint_execution_state_profile_condition_and_budget_mismatch_fail(self):
        for axis in ("execution","state","profile","condition","budget","value_identity","cpu_condition","root_restriction","question","epoch"):
            with self.subTest(axis=axis):
                fixture = SyntheticEndpoint()
                if axis == "execution": fixture.observation["execution"] = 6
                elif axis == "state": fixture.key["state"] = 10
                elif axis in ("profile","condition"): fixture.key[axis] += 1
                elif axis == "budget": fixture.observation["budget"] = 1025
                elif axis == "value_identity": fixture.key[axis]["semantics"] = "different"
                elif axis == "cpu_condition": fixture.key[axis] = "different"
                elif axis == "root_restriction": fixture.key["root_moves"] = [1025]
                elif axis == "question": fixture.key[axis] = "AnalyzeRootMoves"
                elif axis == "epoch": fixture.key[axis] = 1
                with self.assertRaises(ValueError): fixture.read()

    def test_endpoint_completed_task_id_scope_depth_and_bound_are_not_declarations(self):
        for axis in ("id","scope","depth","bound","perspective"):
            with self.subTest(axis=axis):
                fixture = SyntheticEndpoint()
                if axis == "id": fixture.evidence["task"]["status"]["observation_id"] = 6
                elif axis == "scope": fixture.evidence["admitted_scope"] = "FrontierOnly"
                elif axis == "depth": fixture.observation["scope"]["depth"] = 1
                elif axis == "bound": fixture.observation["score"]["bound"] = "Lower"
                elif axis == "perspective": fixture.observation["score"]["white_perspective"] = True
                with self.assertRaises(ValueError): fixture.read()

    def test_failed_paused_and_frontier_endpoint_are_unresolved(self):
        fixture = SyntheticEndpoint()
        fixture.evidence["task"]["status"] = {"kind":"Failed"}
        self.assertEqual(fixture.read()["reason"],"endpoint_task_not_completed")
        fixture.evidence["task"]["status"] = {"kind":"Paused","checkpoint":3,"evidence":4}
        self.assertEqual(fixture.read()["status"],"unresolved")
        fixture.evidence["admitted_scope"] = "FrontierOnly"
        fixture.observation["score"] = {"kind":"Estimate","value_f32_bits":0,"white_perspective":False}
        self.assertEqual(fixture.read()["reason"],"frontier_only_endpoint")

    def test_new_retired_or_model_endpoint_cannot_gain_legacy_completed_cp_authority(self):
        fixture = SyntheticEndpoint()
        fixture.evidence["task"]["status"] = {"kind":"RetiredPaused","checkpoint":3,"evidence":4}
        with self.assertRaisesRegex(ValueError, "typed actual task status"):
            fixture.read()
        fixture = SyntheticEndpoint()
        fixture.endpoint["evidence"] = {"kind":"ModelWdl"}
        with self.assertRaisesRegex(witness.UnsupportedNativeRecheck, "unsupported_native_recheck_endpoint_evidence"):
            fixture.read()

    def test_unobserved_terminal_and_mate_band_do_not_gain_cp_authority(self):
        fixture = SyntheticEndpoint()
        fixture.observation["score"]["value"] = 30000
        self.assertEqual(fixture.read()["status"],"unresolved")
        fixture.endpoint["evidence"] = {"kind":"Unobserved"}
        self.assertEqual(fixture.read()["reason"],"endpoint_provenance_unobserved")
        fixture.endpoint["evidence"] = {"kind":"RulesTerminal","reason":"Checkmate","value":-30000,"white_perspective":False}
        self.assertEqual(fixture.read()["reason"],"rules_terminal_not_cp_comparison")
        fixture.endpoint["evidence"] = {"kind":"InvalidProvenance","error":"synthetic stale handle"}
        with self.assertRaisesRegex(ValueError,"invalid store provenance"): fixture.read()

    def test_endpoint_numeric_ids_budget_and_score_reject_boolean_aliases(self):
        for owner,key in (("endpoint","state"),("evidence","observation_id"),("key","node_budget"),("observation","budget"),("score","value")):
            fixture = SyntheticEndpoint()
            selected = {"endpoint":fixture.endpoint,"evidence":fixture.evidence,"key":fixture.key,"observation":fixture.observation,"score":fixture.observation["score"]}[owner]
            selected[key] = True
            with self.assertRaises(ValueError): fixture.read()

    def test_engine_and_native_clock_domains_must_stay_distinct(self):
        fixture = SyntheticEndpoint()
        controls = {**fixture.prepared,"engine_deadline_tick":9007199254740993,"engine_deadline_tick_unit":"nanoseconds",
                    "engine_deadline_tick_origin":"engine_monotonic_clock_origin","observer_elapsed_origin":"native_capture_start",
                    "observer_elapsed_unit":"microseconds","cancelled_at_observer":False,"deadline_expired_at_observer":False}
        witness._clock_controls(controls)
        for axis,value in (("engine_deadline_tick",True),("engine_deadline_tick_unit","microseconds"),("observer_elapsed_origin","engine_monotonic_clock_origin")):
            mutated = copy.deepcopy(controls)
            mutated[axis] = value
            with self.assertRaises(ValueError): witness._clock_controls(mutated)

    def test_publication_is_restricted_estimate_and_has_no_cpu_or_rules_scope(self):
        identity, prepared = {"repaired_line":3,"root_revision":7}, {"root_state":1}
        observation = {name:None for name in witness._OBSERVATION_FIELDS}
        observation.update(state=1,line=3,source=witness._engine_summary_id(witness._POLICY["search_identity"]),epoch=0,budget=0,kind="Refutation",
                           scope={"kind":"Model","model":witness._engine_summary_id("pals-onnx-pc-fp32-"+"d"*64),
                                  "encoding":witness._engine_summary_id("pals-restricted-summary-v1"),"input":9},
                           score={"kind":"Estimate","value_f32_bits":struct.unpack("<I",struct.pack("<f",-8))[0],"white_perspective":True})
        publication = {"observation_id":10,"observation":observation,"conclusion":{"status":"Refuted","evidence":10,"revision":8}}
        args = {"prepared":prepared,"identity":identity,"root_white":True,"disposition":"ConditionalRefutationPublished","public_revision":8,
                "export_manifest_sha256":"d"*64}
        witness._publication_shape(publication,**args)
        for axis in ("cpu_scope","state_bool","score_bool","conclusion_id","disposition","search_identity","encoding","model"):
            mutated, local = copy.deepcopy(publication), dict(args)
            if axis == "cpu_scope": mutated["observation"]["scope"] = {"kind":"RulesTerminal"}
            elif axis == "state_bool": mutated["observation"]["state"] = True
            elif axis == "score_bool": mutated["observation"]["score"]["value_f32_bits"] = False
            elif axis == "conclusion_id": mutated["conclusion"]["evidence"] = 11
            elif axis == "disposition": local["disposition"] = "CounterNotLower"
            elif axis == "search_identity": mutated["observation"]["source"] += 1
            elif axis == "encoding": mutated["observation"]["scope"]["encoding"] += 1
            else: mutated["observation"]["scope"]["model"] += 1
            with self.assertRaises(ValueError): witness._publication_shape(mutated,**local)


# Historical closed-table pins. They remain independent of the live checkout.
# Preserving an old profile is not a historical build or execution observation.
REVIEWED_WHOLE_SOURCE_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 355881,
        "sha256": "32e393b4ad83bfd33a1b2205cd110cdc0cf007909e7e9c7f617819ccf5c4e93c",
    },
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 224380,
        "sha256": "a286a2403317be7f5f058b65488e7c9722d2b7f4b9bf683ea95e5c9f51a7aaf8",
    },
}

# Independent current-fixture pins: do not derive these from the production
# table or live hashes. A successor needs review and a deliberate fixture edit.
REVIEWED_PRE_LINT_WHOLE_SOURCE_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 374235,
        "sha256": "b86af1f0e55f55add0d4cca277772f8d1c8edf3b3d93d147629c4ab6f881103a",
    },
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 233158,
        "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9",
    },
}

REVIEWED_LINT_WHOLE_SOURCE_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 374216,
        "sha256": "4729c3595731d82e7252dd4aec316bf051092449d95f4fe8a4a83017038ee018",
    },
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 233158,
        "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9",
    },
}

REVIEWED_CONTROLLED_WHOLE_SOURCE_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 382810,
        "sha256": "752d328ae53dc1b4c06a1f3b7cbbd311834dbb59961e598c4752d8fcb9222cf4",
    },
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 233158,
        "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9",
    },
}

REVIEWED_CONTROLLED_FALLBACK_WHOLE_SOURCE_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 391894,
        "sha256": "4770a55361a994210a2238240ed4d2dea1a24d7a8b85524cc7c2db31d19f5e19",
    },
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 233158,
        "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9",
    },
}

# Independent final live-binding pin, never derived from production enrollment.
REVIEWED_LIVE_BINDING_WHOLE_SOURCE_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 396086,
        "sha256": "004db31f9be5133bd1925b373e6df60e97da3d50979a7eb5a031b1dad4ca397b",
    },
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 233158,
        "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9",
    },
}


REVIEWED_OBSERVED_PROGRESS_WHOLE_SOURCE_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 402637,
        "sha256": "4b272ed825d98f51ace408b2798093e88487f6b29f94e38b9d2a753f8fc263d1",
    },
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 233158,
        "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9",
    },
}


REVIEWED_ORDERED_SCOPE_WHOLE_SOURCE_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 402843,
        "sha256": "70d4bae5ad4e19e86137e47cc66922182c5232cbe35b9b30d42bf43ec6850171",
    },
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 233158,
        "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9",
    },
}


# Independent historical pair; preserving the profile certifies no old build.
REVIEWED_OPTIONAL_REVIEW_WHOLE_SOURCE_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 414372,
        "sha256": "ce40b6f9ff8d13a41da8c9482b48528c1437896e9571c9630790c4b11b262726",
    },
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 233208,
        "sha256": "02af59b9edbc350e6e1a204eb6543033bb4a859729eb2ed79c8b470cc6823a01",
    },
}


# Independent current fixture after the compiled-source inventory-only review.
# Source/build/roster/trace bytes are sealed for a new synthetic bank before its
# admission. No old receipt is re-sealed and no adaptive execution is certified.
REVIEWED_CURRENT_WHOLE_SOURCE_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 414554,
        "sha256": "0ba7c9e41b106f35516b8b234872a839f5b2f4227153ed79aa7ff1c503d12837",
    },
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 233208,
        "sha256": "02af59b9edbc350e6e1a204eb6543033bb4a859729eb2ed79c8b470cc6823a01",
    },
}


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


class SyntheticWholeRecheckFixture:
    """One NEW synthetic receipt, two Repair calls, one Reply, public factories.

    The seed supplies reusable model/producer *fixture* declarations only. All
    rows, source/registration/roster/journal/envelope pins and the entire trace
    are fixed BEFORE this new bank is admitted. No already-admitted receipt is
    modified into a positive witness. No production gate or reviewed profile is
    mocked; the only real filesystem source observation is the literal Rust pair.
    Synthetic build, launch, Rules and OwnCpu wire are conditional schema input,
    not actual compilation, process, search, NN or Rules execution evidence.
    """

    def __init__(self, directory, *, trace_mutation=None, raw_mutation=None):
        self.seed = RepairFixture(directory)
        self.root = Path(directory)
        self.rows = copy.deepcopy(self.seed.rows)
        self.lineages = copy.deepcopy(self.seed.lineages)
        self.source = copy.deepcopy(self.seed.source)
        self.repaired = self.lineages[1]["virtual_prefix"] + [fixtures.move(4, 3), fixtures.move(51, 43)]
        self.refutation = copy.deepcopy(self.lineages[1]["counterexample"])
        self.counter = self.repaired[:3] + [fixtures.move(51, 52)]
        self.identity = {"game_generation": 1, "search_generation": 1,
                         "root": {"slot": 0, "generation": 1}, "root_revision": 7,
                         "repair_record_revision": 4, "repaired_line": 4}
        self._register_source()
        self._extend_rows()
        self._prepare_rows()
        if raw_mutation is not None:
            raw_mutation(self)
        self._make_trace()
        if trace_mutation is not None:
            trace_mutation(self)
        self._seal_trace()
        self._admit_new_bank()
        self.causal = self.seed.admit()
        self.anchor = witness.admit_native_recheck_repair_anchor(**self.anchor_arguments())
        self.repaired_check, self.reply_check = self._whole_rules_checks()

    def _register_source(self):
        self.value_identity = {"semantics": "bootstrap-material-pst-v1", "weights_sha256": None,
                               "training": {"kind": "bootstrap"}}
        cpu_config = {"value_identity": copy.deepcopy(self.value_identity), "value_precision": "integer_cp"}
        cpu_sha = byte_pin(canonical(cpu_config))["sha256"]
        self.source["cpu_profile_sha256"] = cpu_sha
        native_source = self.source["native"]
        native_source["search_implementation_sha256"] = FIXTURE_COMPILED_FACT
        native_source["cpu_task_source"] = {"configuration": cpu_config,
            "implementation_sha256": self.source["implementation_sha256"], "cpu_profile_sha256": cpu_sha}
        registry = native_source["independent_registry"]
        registry["cpu_configuration_sha256"] = cpu_sha
        native_source["search_configuration"] = {"max_rounds": 2, "max_cpu_nodes": 8192, "cpu_depth": 2}
        policy_registration = {"version": witness.POLICY_REGISTRATION_SCHEMA,
            "base_registry_canonical_sha256": byte_pin(canonical(registry))["sha256"],
            "collector_binary_sha256": self.source["implementation_sha256"],
            "search_policy": copy.deepcopy(witness._POLICY)}
        self.policy_raw = canonical(policy_registration)
        native_source.update(refinement_registration=policy_registration,
            refinement_registration_sha256=byte_pin(self.policy_raw)["sha256"],
            pals_search_policy=copy.deepcopy(witness._POLICY), search_version=witness._POLICY["search_identity"])
        self.source_raw = canonical([training.CHECKED_SOURCE_DOMAIN, self.source])
        registration = json.loads(self.seed.registration_raw)
        registration["checked_source_sha256"] = byte_pin(self.source_raw)["sha256"]
        self.registration_raw = canonical(registration)
        producer = copy.deepcopy(self.seed.options["producer_registrations"][0]["pin"])
        producer["registration_sha256"] = byte_pin(self.registration_raw)["sha256"]
        self.roster = frozen.seal_roster({"version": frozen.ROSTER_DOMAIN, "game_producers": [producer]})
        self.producer = producer
        reviewed_sources = synthetic_custody_fixture_sources()
        self.source_raws = {name: reviewed_sources[path] for path, name in
            zip(FIXTURE_TRANSITION_PINS, ("engine_source", "native_source"))}
        manifest = fixture_manifest()
        self.source_raws["source_manifest"] = canonical(manifest)
        build = {"schema": witness.BUILD_SCHEMA, "source_commit": FIXTURE_SOURCE_ID,
                 "actual_build_exit_code": 0, "source_verified_before_and_after_build": True,
                 "binary": {"bytes": 17, "sha256": self.source["implementation_sha256"]},
                 "source_manifest": byte_pin(self.source_raws["source_manifest"]),
                 "compiler_artifact": {"reason": "compiler-artifact", "target": {"name": "pals_collect"},
                                       "features": ["pals-collection-onnx"], "fresh": False},
                 "assurance": "synthetic caller compiler/binary observations; no actual build or process execution"}
        self.source_raws["build_registration"] = canonical(build)

    def _extend_rows(self):
        snapshot = copy.deepcopy(self.rows[1]["input"]["snapshot"])
        snapshot.update(capture_sequence=10, white_to_move=False,
            position_command=self.rows[0]["input"]["snapshot"]["position_command"] + " moves e2e4 e8d7 e1d1",
            board_fen="8/3k4/8/8/4P3/8/8/3K4 b - - 2 2",
            rules_state_sha256=fixtures.sha("synthetic whole ply3 Rules state"),
            rules_history_sha256=fixtures.sha("synthetic whole ply3 Rules history"),
            transposition_sha256=fixtures.sha("synthetic whole ply3 transposition"),
            legal_moves=[fixtures.move(51, 43), fixtures.move(51, 52), fixtures.move(51, 50)])
        self.rows.append({"input": {"snapshot": snapshot, "sha256": ""}, "future_label": None, "verifier_private": None})
        second = copy.deepcopy(self.lineages[1])
        second.update(request_sequence=9, virtual_prefix=self.repaired[:3])
        self.lineages.append(second)
        public = {"domain": "rz-pals-native-public-source/1", "game_id": snapshot["game_id"],
                  "record_index": 1, "revision": 4, "origin_state_id": 0, "origin_state_id_is_advisory": True,
                  "origin_rules_state_sha256": None, "origin_rules_identity_observation": "unknown",
                  "kind": "Repair", "line": copy.deepcopy(self.repaired), "value": None,
                  "completed_depth": 0, "scope": None, "white_score_perspective": True, "critical": True,
                  "source_cpu_profile_sha256": self.source["cpu_profile_sha256"]}
        self.public_raw = canonical(public)
        reply = copy.deepcopy(snapshot)
        reply.update(role="critic", input_revision=4, capture_sequence=11,
                     public_records=[{"observation_sha256": byte_pin(self.public_raw)["sha256"], "situation_revision": 4}])
        self.rows.append({"input": {"snapshot": reply, "sha256": ""}, "future_label": None, "verifier_private": None})
        branch = copy.deepcopy(second)
        branch.update(request_sequence=10, native_query_kind="Reply", proposal=copy.deepcopy(self.repaired))
        self.lineages.append(branch)

    def _prepare_rows(self):
        self.sidecars, self.tensors, self.journals, self.events, self.outputs = [], [], [], [], []
        template = json.loads(self.seed.base["producer-prepared.jsonl"])
        for index, (row, lineage) in enumerate(zip(self.rows, self.lineages)):
            snapshot = row["input"]["snapshot"]
            row["input"]["sha256"] = training.seal_snapshot(snapshot)
            input_sha = row["input"]["sha256"]
            lineage["input_sha256"] = input_sha
            sidecar = fixtures.native_sidecar(row)
            tensor = json.loads(sidecar["tensor_json"])
            prefix, proposal, counter = lineage["virtual_prefix"], lineage["proposal"], lineage["counterexample"]
            query = [0.0] * 16
            query[:9] = [{"Propose": 0, "Repair": 2, "Reply": 1}[lineage["native_query_kind"]],
                len(prefix) / 256, len(proposal) / 256, int(counter is not None),
                0 if counter is None else len(counter) / 256, len(snapshot["legal_moves"]) / 256,
                snapshot["input_revision"], 9.25, int(snapshot["white_to_move"])]
            for offset, movement in ((9, prefix[-1] if prefix else None), (12, proposal[0] if proposal else None)):
                if movement is not None:
                    start, end, promotion = training.move_components(movement)
                    query[offset:offset + 3] = [start / 63, end / 63, promotion / 4]
            query[15] = 1
            records = []
            if index == 3:
                first = training.move_components(self.repaired[0])
                last = training.move_components(self.repaired[-1])
                features = [2, 0, 0, 0, -1, 1, 0, len(self.repaired) / 256,
                    first[0] / 63, first[1] / 63, first[2] / 4, 1,
                    last[0] / 63, last[1] / 63, last[2] / 4, 1]
                records = [{"record_id": 1, "revision": 4, "critical": True, "features": features}]
            tensor.update(role=snapshot["role"], query=query, records=records,
                required_critical_records=[1] if records else [],
                model_epoch=list(bytes.fromhex(snapshot["source"]["model_weights_sha256"])),
                situation_revision=snapshot["input_revision"], history_digest=list(bytes.fromhex(snapshot["rules_history_sha256"])))
            sidecar.update(model_epoch_kind="frozen_model_epoch", record_sources=copy.deepcopy(snapshot["public_records"]),
                canonical_tensor_sha256=fixtures.sha("synthetic whole canonical tensor " + str(index)),
                tensor_json=json.dumps(tensor, separators=(",", ":"), allow_nan=False))
            fixtures.reseal_sidecar(sidecar)
            journal = copy.deepcopy(template)
            journal["prepared"].update(input_sha256=input_sha, capture_sequence=snapshot["capture_sequence"],
                native_request=[lineage["process_epoch"], lineage["request_sequence"]], input_json=byte_pin(canonical(row["input"])),
                tensor_sidecar_json=byte_pin(canonical(sidecar)), lineage_json=byte_pin(canonical(lineage)), learning_input=True,
                roster_sha256=self.roster["sha256"], registration_sha256=byte_pin(self.registration_raw)["sha256"],
                checked_source_sha256=byte_pin(self.source_raw)["sha256"])
            journal["sha256"] = training._sorted_canonical(training.PREPARED_PRODUCER_DOMAIN, journal["prepared"])
            self.sidecars.append(sidecar)
            self.tensors.append(tensor)
            self.journals.append(journal)
            details = [{"prepared_before_submit": True, "native_query_kind": lineage["native_query_kind"], "producer_metadata_admitted": True},
                       {"success": True, "logical_acceptance_inferred": False}, {"search_consumed": False}, {"search_consumed": True}]
            for tick, (stage, detail) in enumerate(zip(("prepared", "physically_completed", "delivered", "search_consumed"), details)):
                self.events.append({"domain": "rz-pals-native-call-event/1", "game_id": snapshot["game_id"],
                    "process_epoch": 1, "request_sequence": lineage["request_sequence"], "input_sha256": input_sha,
                    "stage": stage, "observer_elapsed_us": index * 40 + tick + 1, "detail": detail})
            selected = 1 if index in (1, 3) else 0
            self.outputs.append({"domain": "rz-pals-native-physical-raw/1", "process_epoch": 1,
                "request_sequence": lineage["request_sequence"], "input_sha256": input_sha,
                "physical_completion_confirmed": True, "success": True,
                "raw": {"representation": "f32_ieee754_bits",
                        "candidate_logits_bits": [f32_bits(1 if slot == selected else -1) for slot in range(len(snapshot["legal_moves"]))],
                        "wdl_logits_bits": [0, 0, 0], "divergence_logits_bits": [] if index == 3 else None,
                        "task_logits_bits": None, "private_latent_bits": [0] * 6144, "prediction_is_future_label": False}})

    def _row_anchors(self, index):
        return {"input_row_sha256": byte_pin(canonical(self.rows[index]["input"]))["sha256"],
                "sidecar_row_sha256": byte_pin(canonical(self.sidecars[index]))["sha256"],
                "lineage_row_sha256": byte_pin(canonical(self.lineages[index]))["sha256"],
                "sidecar_sha256": self.sidecars[index]["sha256"],
                "canonical_tensor_sha256": self.sidecars[index]["canonical_tensor_sha256"]}

    def _logical_context(self, index, purpose, prefix):
        return {"game_generation": 1, "search_generation": 1,
                "situation": {"slot": len(prefix), "generation": 1}, "state": len(prefix), "focus": len(prefix),
                "purpose": purpose, "prefix": copy.deepcopy(prefix), "focus_sha256": fixtures.sha("synthetic focus " + str(index)),
                "prefix_sha256": byte_pin(canonical(prefix))["sha256"], "proposal_sha256": fixtures.sha("synthetic proposal " + str(index)),
                "refutation_sha256": byte_pin(canonical(self.refutation))["sha256"], "divergence_sha256": fixtures.sha("synthetic empty divergence"),
                "public_revision": self.rows[index]["input"]["snapshot"]["input_revision"], "situation_revision": 1}

    def _endpoint(self, state, execution, observation_id, score, fen, name):
        endpoint = SyntheticEndpoint().endpoint
        endpoint.update(state=state, situation={"slot": state, "generation": 1},
                        rules_state_sha256=fixtures.sha("synthetic " + name + " endpoint state"), board_fen=fen)
        evidence = endpoint["evidence"]
        evidence.update(observation_id=observation_id, execution_id=execution)
        evidence["observation"].update(state=state, execution=execution,
            value_identity=copy.deepcopy(self.value_identity), cpu_condition="synthetic exact OwnedCpu recheck condition",
            score={"kind": "Cpu", "value": score, "white_perspective": True, "bound": "ExactWithinSearch"})
        evidence["task"]["key"].update(state=state, value_identity=copy.deepcopy(self.value_identity),
                                       cpu_condition="synthetic exact OwnedCpu recheck condition")
        evidence["task"]["status"]["observation_id"] = observation_id
        return endpoint

    def _make_trace(self):
        chain = []
        for index, ply in ((1, 2), (2, 3)):
            context = self._logical_context(index, "RepairPolicy", self.repaired[:ply])
            chain.append({"process_epoch": 1, "request_sequence": self.lineages[index]["request_sequence"],
                "input_sha256": self.rows[index]["input"]["sha256"], **self._row_anchors(index),
                "prefix_len": ply, "prefix": self.repaired[:ply], "chosen_move": self.repaired[ply],
                "position_rules_state_sha256": self.rows[index]["input"]["snapshot"]["rules_state_sha256"],
                "proposal": self.lineages[1]["proposal"], "counterexample": self.refutation,
                "logical_context_sha256": byte_pin(canonical(context))["sha256"], "logical_context": context,
                "producer_metadata_admitted": True,
                "selection_observation": "actual-raw-policy-ranked-first-and-accepted-context-matches-repaired-ply"})
        anticipated = self._logical_context(3, "ReplyPolicy", self.repaired[:3])
        record = {"revision": 4, "origin_state": 0, "kind": "Repair", "line": copy.deepcopy(self.repaired),
                  "value": None, "completed_depth": 0, "score_scope": None, "cpu_observation": None,
                  "white_perspective": True, "critical": True}
        repaired_endpoint = self._endpoint(4, 5, 4, 8, "8/8/3k4/8/4P3/8/8/3K4 w - - 3 3", "repaired")
        counter_endpoint = self._endpoint(5, 6, 5, -8, "8/4k3/8/8/4P3/8/8/3K4 w - - 3 3", "counter")
        self.clock = {"engine_deadline_tick": 1000000, "engine_deadline_tick_unit": "nanoseconds",
                      "engine_deadline_tick_origin": "engine_monotonic_clock_origin", "observer_elapsed_origin": "native_capture_start",
                      "observer_elapsed_unit": "microseconds", "cancelled_at_observer": False, "deadline_expired_at_observer": False,
                      "limits": {"max_rounds": 2, "max_cpu_nodes": 8192, "cpu_depth": 2}}
        self.prepared = {"engine_observer_version": "pals-post-repair-recheck-observer/1",
            "checked_source_sha256": byte_pin(self.source_raw)["sha256"], "refinement_registration_sha256": byte_pin(self.policy_raw)["sha256"],
            "root_state": 0, "root_rules_state_sha256": self.rows[0]["input"]["snapshot"]["rules_state_sha256"],
            "anchor_state": 3, "anchor_situation": {"slot": 3, "generation": 1},
            "anchor_rules_state_sha256": self.rows[3]["input"]["snapshot"]["rules_state_sha256"], "anchor_ply": 3,
            "anticipated_reply_context": anticipated, "anticipated_reply_context_sha256": byte_pin(canonical(anticipated))["sha256"],
            "accepted_repair_record": record, "repaired": copy.deepcopy(self.repaired), "refutation": copy.deepcopy(self.refutation),
            "parent_repair_chain": {"initial_prefix": self.repaired[:2], "initial_prefix_len": 2, "full_repaired_line": copy.deepcopy(self.repaired),
                "calls": chain, "journal_observation": "independent-exact-producer-journal-entry-required; raw-entry-SHA-not-returned-by-producer-API"},
            "repaired_endpoint": repaired_endpoint, **copy.deepcopy(self.clock),
            "checker_identity": {"kind": "owned", "identity": copy.deepcopy(self.value_identity)},
            "cpu_condition": "synthetic exact OwnedCpu recheck condition", "prepared_before_reply_submit": True,
            "trace_persistence": "seal-before-submit; prepaid-buffer-drain-after-search",
            "assurance": "conditional-search-observer; no-whole-game-proof; no-training-target; task-private-consumers-not-observed"}
        self.bound = {"prepared_payload_sha256": "", "process_epoch": 1, "request_sequence": 10,
                      "input_sha256": self.rows[3]["input"]["sha256"], **self._row_anchors(3),
                      "logical_context": copy.deepcopy(anticipated), "producer_metadata_admitted": True, "bound_before_submit": True,
                      "physical_completion_observed": False, "delivery_observed": False, "search_consumption_observed": False}
        observation = {name: None for name in witness._OBSERVATION_FIELDS}
        observation.update(state=0, line=4, source=witness._engine_summary_id(witness._POLICY["search_identity"]), epoch=0, budget=0, kind="Refutation",
            scope={"kind": "Model", "model": witness._engine_summary_id("pals-onnx-pc-fp32-" + self.source["native"]["independent_registry"]["export_manifest_sha256"]),
                   "encoding": witness._engine_summary_id("pals-restricted-summary-v1"), "input": 5},
            score={"kind": "Estimate", "value_f32_bits": f32_bits(-8), "white_perspective": True})
        self.finished = {"prepared_payload_sha256": "", "bound_payload_sha256": "", "prepared_accepted": True,
            "reply_call_attempted": True, "reply_accepted": True,
            "reply": {"process_epoch": 1, "request_sequence": 10, "input_sha256": self.rows[3]["input"]["sha256"],
                      "physical": True, "delivered": True, "search_consumed": True, "rejected": False, "physical_unknown": False},
            "reply_context": copy.deepcopy(anticipated), "selected_response": self.counter[3], "counterline": copy.deepcopy(self.counter),
            "full_suffix_replayed": True, "repaired_endpoint": copy.deepcopy(repaired_endpoint), "counter_endpoint": counter_endpoint,
            "comparable": True, "publication": {"observation_id": 10, "observation": observation,
                "conclusion": {"status": "Refuted", "evidence": 10, "revision": 8}},
            "disposition": "ConditionalRefutationPublished", "original_error": None, **copy.deepcopy(self.clock),
            "assurance": "conditional-search-observer; marker-is-not-refutation; no-training-target; physical-and-delivery-events-required"}
        self.stages = ["prepared", "reply_bound", "finished"]

    def _seal_trace(self):
        self.descriptor = byte_pin(canonical([witness.TRACE_DOMAIN, "prepared-descriptor", self.identity, self.prepared]))["sha256"]
        payloads = {}
        rows = []
        for stage, tick in (("prepared", 100), ("reply_bound", 110), ("finished", 200)):
            if stage not in self.stages:
                continue
            data = {"prepared": self.prepared, "reply_bound": self.bound, "finished": self.finished}[stage]
            if stage != "prepared":
                data["prepared_payload_sha256"] = payloads["prepared"]
            if stage == "finished":
                data["bound_payload_sha256"] = payloads.get("reply_bound")
            payload = byte_pin(canonical([witness.TRACE_DOMAIN, stage, self.identity, self.descriptor, data]))["sha256"]
            payloads[stage] = payload
            rows.append({"domain": witness.TRACE_DOMAIN, "stage": stage, "game_id": self.rows[0]["input"]["snapshot"]["game_id"],
                         "identity": copy.deepcopy(self.identity), "descriptor_sha256": self.descriptor, "payload_sha256": payload,
                         "observer_elapsed_us": tick, "data": copy.deepcopy(data)})
        self.trace_raw = jsonl(rows)

    def _admit_new_bank(self):
        bindings = [{"input_sha256": journal["prepared"]["input_sha256"], "game_id": journal["prepared"]["game_id"],
                     "producer_id": journal["prepared"]["producer_id"], "capture_sequence": journal["prepared"]["capture_sequence"],
                     "capture_evidence_sha256": journal["sha256"]} for journal in self.journals]
        capture = frozen.seal_capture({"version": frozen.CAPTURE_DOMAIN, "bindings": bindings})
        registry = json.loads(self.seed.base["source-registry.jsonl"])
        split = json.loads(self.seed.base["split.jsonl"])
        view = training.ValidatedDataset(self.rows, split, registry, {}).current_view
        envelope = json.loads(self.seed.base["producer-envelope.json"])["envelope"]
        envelope.update(raw_records=4, unique_inputs=4, current_view_sha256=view.sha256, roster_sha256=self.roster["sha256"],
                        capture_sha256=capture["sha256"], capture_artifact=byte_pin(canonical(capture)))
        envelope = frozen.seal_envelope(envelope)
        audit = json.loads(self.seed.base["producer-audit.json"])
        audit.update(raw_records=4, unique_inputs=4, native_exact_metadata_inputs=4, roster_sha256=self.roster["sha256"],
                     envelope_sha256=envelope["sha256"], capture_sha256=capture["sha256"])
        assets = dict(self.seed.base)
        assets.update({"records.jsonl": jsonl(self.rows), "inputs.jsonl": jsonl([row["input"] for row in self.rows]),
            "native-inputs.jsonl": jsonl(self.sidecars), "input-lineage.jsonl": jsonl(self.lineages),
            "producer-source.json": self.source_raw, "producer-registration.json": self.registration_raw,
            "producer-roster.json": canonical(self.roster), "producer-prepared.jsonl": jsonl(self.journals),
            "producer-captures.json": canonical(capture), "producer-envelope.json": canonical(envelope), "producer-audit.json": canonical(audit),
            "public-record-sources.jsonl": self.public_raw + b"\n", "native-events.jsonl": jsonl(self.events),
            "native-raw-outputs.jsonl": jsonl(self.outputs), witness.TRACE_ARTIFACT: self.trace_raw})
        # This is the FIRST receipt for this bank, not a re-signing of an admitted
        # historical collection. Every trace stage is already in its inventory.
        receipt = json.loads(self.seed.base["receipt.json"])
        receipt.update(source=copy.deepcopy(self.source), artifacts={name: byte_pin(raw) for name, raw in assets.items() if name != "receipt.json"})
        receipt["audit"]["records"] = 4
        receipt["native_finish"] = {"receipt": {"physical_shutdown_confirmed": True, "native_buffers_released": True,
            "quarantined": False, "physical_runs_in_flight": 0, "observer_failures": 0},
            "finish_error": None, "_collection_failure": None, "collection_accepted": True}
        assets["receipt.json"] = canonical(receipt)
        # Preserve the seed directory. The newly frozen bank has a distinct owner.
        collection = self.root / "synthetic-whole-bank"
        collection.mkdir()
        for name, raw in assets.items():
            (collection / name).write_bytes(raw)
        options = {"expected_receipt_sha256": byte_pin(assets["receipt.json"])["sha256"],
                   "producer_registrations": [{"pin": copy.deepcopy(self.producer),
                       "registration_path": collection / "producer-registration.json", "checked_source_path": collection / "producer-source.json"}]}
        parents = training.load_frozen_collected_dataset(collection, **options)
        seed = self.seed
        seed.collection, seed.options, seed.parents = collection, options, parents
        seed.rows, seed.lineages, seed.sidecars, seed.tensors, seed.journals = self.rows, self.lineages, self.sidecars, self.tensors, self.journals
        seed.source, seed.source_raw, seed.registration_raw = self.source, self.source_raw, self.registration_raw
        seed.root_index = next(index for index in parents.current_view.current_indices if parents.records[index]["input"]["sha256"] == self.rows[0]["input"]["sha256"])
        seed.repair_index = next(index for index in parents.current_view.current_indices if parents.records[index]["input"]["sha256"] == self.rows[1]["input"]["sha256"])
        seed.artifacts = {name: assets[name] for name in repair._NAMES}
        launch = {"schema": witness.native.LAUNCH_SCHEMA, "receipt_artifact": byte_pin(assets["receipt.json"]),
                  "checked_source_artifact": byte_pin(self.source_raw), "collector_binary_sha256": self.source["implementation_sha256"],
                  "assurance_scope": "independently_pinned_caller_collection_observation", "spawned": True, "reaped": True, "exit_code": 0, "timed_out": False}
        seed.launch_raw = canonical(launch)
        seed._rules_checks()
        seed.context_and_pins()
        self.assets = assets

    def _whole_rules_checks(self):
        checks = []
        for name, prefix, claim, endpoint in (("whole-repaired", [], self.repaired, self.prepared["repaired_endpoint"]),
                ("whole-reply", self.repaired[:3], self.counter[3:], self.finished["counter_endpoint"])):
            # A missing endpoint is handled as unresolved before these capabilities
            # are consumed; keep the independent full-line Rules fixture available.
            if endpoint is None:
                endpoint = self._endpoint(5, 6, 5, -8, "8/4k3/8/8/4P3/8/8/3K4 w - - 3 3", "counter")
            folder = self.root / ("synthetic-semantic-" + name)
            folder.mkdir()
            check = SemanticFixture(folder)
            check.parents, check.index = self.seed.parents, self.seed.root_index
            check.snapshot = copy.deepcopy(self.rows[0]["input"]["snapshot"])
            check.common.update(parent_input_sha256=self.rows[0]["input"]["sha256"], current_view_sha256=self.seed.parents.current_view.sha256,
                question="continuation_challenge", prefix=copy.deepcopy(prefix), claimed_line=copy.deepcopy(claim),
                root_moves=[], allowed_tasks=["attack_repair"] if prefix else ["resume_task"])
            check.receipt["root"] = copy.deepcopy(self.seed.prefix_check.rules_receipt()["root"])
            target = copy.deepcopy(check.receipt["root"])
            if prefix:
                self.seed._descriptor(target, self.rows[3]["input"]["snapshot"], "target_legal", 4)
                target.update(halfmove_clock=2, fullmove_number=2)
            else:
                target["legal_tokens"] = tokens(target["legal_moves"], "target_legal", target["legal_moves"])
            check.receipt["target"] = target
            final = copy.deepcopy(target)
            final.update(board_fen=endpoint["board_fen"], board64_piece_codes=list(training._fen_board(endpoint["board_fen"])),
                side_to_move="white", rules_state_sha256=endpoint["rules_state_sha256"],
                rules_history_sha256=fixtures.sha("synthetic " + name + " endpoint history"), known_history_positions=5,
                halfmove_clock=3, fullmove_number=3,
                legal_moves=[fixtures.move(3, 4), fixtures.move(3, 11), fixtures.move(28, 36)])
            final["legal_order_sha256"] = semantic.digest(semantic.MOVE_DOMAIN, final["legal_moves"])
            final["legal_tokens"] = tokens(final["legal_moves"], "claim_end_legal", final["legal_moves"])
            check.receipt["claimed_line"] = {"status": "legal_continuation", "legality_verified": True, "claim_truth": "unknown",
                "movements": tokens(claim, "claimed_continuation"), "restriction_checked": False, "final_state": final}
            check.reseal()
            checks.append(check.admit())
        return checks

    def anchor_arguments(self):
        return {"initial_repair": self.causal, "policy_registration_bytes": self.policy_raw,
                "build_registration_bytes": self.source_raws["build_registration"], "source_manifest_bytes": self.source_raws["source_manifest"],
                "engine_source_bytes": self.source_raws["engine_source"], "native_source_bytes": self.source_raws["native_source"],
                "expected_pins": {**{name: byte_pin(raw) for name, raw in self.source_raws.items()},
                    "policy_registration": byte_pin(self.policy_raw), "initial_repair_sha256": self.causal.sha256}}

    def arguments(self, *, rules_missing=False):
        return {"anchor": self.anchor, "trace_bytes": self.trace_raw,
                "repaired_rules_check": None if rules_missing else self.repaired_check,
                "reply_rules_check": None if rules_missing else self.reply_check,
                "expected_pins": {"anchor_sha256": self.anchor.sha256, "trace": byte_pin(self.trace_raw),
                    "identity": copy.deepcopy(self.identity), "descriptor_sha256": self.descriptor,
                    "repaired_rules_sha256": None if rules_missing else self.repaired_check.sha256,
                    "reply_rules_sha256": None if rules_missing else self.reply_check.sha256}}

    def admit(self, **kwargs):
        return witness.admit_native_recheck_witness(**self.arguments(**kwargs))


class NativeRecheckPublicWholeTests(unittest.TestCase):
    """Public-factory synthetic wiring; execution/effectiveness remain untested."""

    def setUp(self):
        self.enterContext(fixture_review_scope("legacy_single_reply"))

    def assert_denied_authorities(self, observation):
        for name in witness._DENIED_AUTHORITIES:
            self.assertIs(observation[name], False, name)
        self.assertIs(observation["search_completion_admitted"], False)
        self.assertIs(observation["final_search_envelope_observed"], False)

    def test_public_anchor_and_whole_positive_two_call_chain_original_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticWholeRecheckFixture(directory)
            self.assertIs(fixture.anchor.verify(), fixture.anchor)
            value = fixture.admit()
            self.assertIs(value.verify(), value)
            observation = value.observation()
            self.assertEqual(observation["status"], "conditional_cp_observation")
            self.assertEqual(observation["reason"], None)
            self.assertEqual([call["native_request"] for call in observation["parent_repair_chain"]], [[1, 8], [1, 9]])
            self.assertEqual((observation["raw_repaired_cp"], observation["raw_counter_cp"]), (8, -8))
            self.assertEqual((observation["root_repaired_cp"], observation["root_counter_cp"]), (8, -8))
            self.assertIs(observation["conditional_counter_lower"], True)
            self.assertIs(observation["publication_observed"], True)
            self.assertEqual(observation["reply_observation"]["status"], "consumed_raw_observation")
            self.assertEqual(observation["publication"]["observation"]["score"]["kind"], "Estimate")
            self.assertEqual(observation["publication"]["conclusion"]["revision"], 8)
            self.assert_denied_authorities(observation)
            original = json.loads(fixture.seed.artifacts["receipt.json"])
            self.assertEqual(original["artifacts"][witness.TRACE_ARTIFACT], byte_pin(fixture.trace_raw))
            self.assertEqual(set(original["artifacts"]), set(fixture.assets) - {"receipt.json"})
            source_review = fixture.anchor.admission()["source_observation"]
            self.assertEqual(source_review["reviewed_transition_sources"], FIXTURE_TRANSITION_PINS)
            closure = source_review["reviewed_transition_closure"]
            self.assertEqual(closure["component_pins"], FIXTURE_COMPONENT_PINS)
            self.assertEqual(closure["closure_sha256"], FIXTURE_CLOSURE_SHA256)
            self.assertEqual(closure["compiled_search_implementation_sha256"], FIXTURE_COMPILED_FACT)
            self.assertIs(closure["historical_metadata_only"], True)
            self.assertIs(closure["execution_authorized"], False)
            self.assertEqual(fixture.repaired_check.rules_receipt()["claimed_line"]["claim_truth"], "unknown")
            self.assertEqual(fixture.reply_check.rules_receipt()["claimed_line"]["claim_truth"], "unknown")
            # Capabilities are immutable and their returned views are detached.
            observation["publication"]["conclusion"]["revision"] = 99
            self.assertEqual(value.observation()["publication"]["conclusion"]["revision"], 8)
            with self.assertRaises(AttributeError):
                value._trace = b""
            for name in ("collate", "loss", "target", "policy_target", "wdl_target"):
                self.assertFalse(hasattr(value, name))

    def test_public_whole_rejects_trace_bytes_or_independent_rules_pin_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticWholeRecheckFixture(directory)
            arguments = fixture.arguments()
            arguments["trace_bytes"] += b"\n"
            with self.assertRaises(ValueError):
                witness.admit_native_recheck_witness(**arguments)
            arguments = fixture.arguments()
            arguments["expected_pins"]["reply_rules_sha256"] = "e" * 64
            with self.assertRaisesRegex(ValueError, "Rules capability/independent pin"):
                witness.admit_native_recheck_witness(**arguments)

    def test_public_anchor_rejects_source_byte_mutation_even_with_fresh_independent_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticWholeRecheckFixture(directory)
            arguments = fixture.anchor_arguments()
            arguments["engine_source_bytes"] += b"\n"
            arguments["expected_pins"]["engine_source"] = byte_pin(arguments["engine_source_bytes"])
            with self.assertRaisesRegex(ValueError, "original registered build manifest"):
                witness.admit_native_recheck_repair_anchor(**arguments)

    def test_public_whole_rejects_resealed_last_call_substitution(self):
        def mutate(fixture):
            fixture.prepared["parent_repair_chain"]["calls"] = fixture.prepared["parent_repair_chain"]["calls"][-1:]
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticWholeRecheckFixture(directory, trace_mutation=mutate)
            with self.assertRaisesRegex(ValueError, "entire actual Repair suffix"):
                fixture.admit()

    def test_public_whole_rejects_resealed_repair_policy_choice_or_reply_context_drift(self):
        def wrong_choice(fixture):
            fixture.outputs[1]["raw"]["candidate_logits_bits"] = [f32_bits(1), f32_bits(-1), f32_bits(-1)]
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticWholeRecheckFixture(directory, raw_mutation=wrong_choice)
            with self.assertRaisesRegex(ValueError, "actual policy rank"):
                fixture.admit()
        def wrong_context(fixture):
            fixture.bound["logical_context"]["public_revision"] = 5
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticWholeRecheckFixture(directory, trace_mutation=wrong_context)
            with self.assertRaisesRegex(ValueError, "bound row drifted"):
                fixture.admit()

    def test_public_whole_cancel_and_expired_preserve_unresolved(self):
        for flag in ("cancelled_at_observer", "deadline_expired_at_observer"):
            def mutate(fixture, selected=flag):
                fixture.finished[selected] = True
                fixture.finished.update(disposition="Interrupted", comparable=False, publication=None)
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as directory:
                fixture = SyntheticWholeRecheckFixture(directory, trace_mutation=mutate)
                observation = fixture.admit().observation()
                self.assertEqual(observation["status"], "unresolved")
                self.assertEqual(observation["reason"], "cancelled_expired_or_failed_search_preserved")
                self.assertIs(observation["publication_observed"], False)
                self.assert_denied_authorities(observation)

    def test_public_whole_original_service_error_preserved_without_stop_flags(self):
        def mutate(fixture):
            fixture.finished.update(
                original_error="synthetic retained original service error",
                disposition="Interrupted", comparable=False, publication=None)
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticWholeRecheckFixture(directory, trace_mutation=mutate)
            observation = fixture.admit().observation()
            self.assertEqual(observation["status"], "unresolved")
            self.assertEqual(observation["reason"], "cancelled_expired_or_failed_search_preserved")
            self.assertEqual(observation["original_error"], "synthetic retained original service error")
            self.assertIs(observation["finished_cancelled"], False)
            self.assertIs(observation["finished_deadline_expired"], False)
            self.assertIs(observation["publication_observed"], False)
            self.assert_denied_authorities(observation)

    def test_public_whole_rejects_resealed_cpu_execution_and_publication_line_drift(self):
        for axis, expected in (("cpu_execution", "stored CPU execution"), ("publication_line", "restricted estimate/conclusion")):
            def mutate(fixture, selected=axis):
                if selected == "cpu_execution":
                    fixture.finished["counter_endpoint"]["evidence"]["observation"]["execution"] = 99
                else:
                    fixture.finished["publication"]["observation"]["line"] = 99
            with self.subTest(axis=axis), tempfile.TemporaryDirectory() as directory:
                fixture = SyntheticWholeRecheckFixture(directory, trace_mutation=mutate)
                with self.assertRaisesRegex(ValueError, expected):
                    fixture.admit()

    def test_public_whole_prepared_bound_without_finished_remains_unresolved(self):
        def mutate(fixture):
            fixture.stages = ["prepared", "reply_bound"]
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticWholeRecheckFixture(directory, trace_mutation=mutate)
            observation = fixture.admit().observation()
            self.assertEqual(observation["status"], "unresolved")
            self.assertEqual(observation["reason"], "missing_prepared_or_finished_trace")
            self.assertEqual(observation["reply_observation"]["status"], "consumed_raw_observation")
            self.assertIs(observation["publication_observed"], False)
            self.assert_denied_authorities(observation)

    def test_public_whole_failed_counter_endpoint_is_not_completed_evidence(self):
        def mutate(fixture):
            fixture.finished["counter_endpoint"]["evidence"]["task"]["status"] = {"kind": "Failed"}
            fixture.finished.update(disposition="MissingCompletedCounterValue", comparable=False, publication=None)
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticWholeRecheckFixture(directory, trace_mutation=mutate)
            observation = fixture.admit().observation()
            self.assertEqual(observation["status"], "unresolved")
            self.assertEqual(observation["counter_endpoint"]["reason"], "endpoint_task_not_completed")
            self.assertEqual(observation["reason"], "partial_missing_terminal_or_incomparable_endpoint_observation")
            self.assert_denied_authorities(observation)

    def test_public_whole_missing_full_rules_capabilities_remain_unresolved(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticWholeRecheckFixture(directory)
            observation = fixture.admit(rules_missing=True).observation()
            self.assertEqual(observation["status"], "unresolved")
            self.assertEqual(observation["reason"], "full_line_registered_Rules_capability_missing")
            self.assert_denied_authorities(observation)


if __name__ == "__main__":
    unittest.main()
