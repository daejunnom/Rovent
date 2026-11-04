"""Synthetic utility dataset assembly, never original known-pair authority.

The file reader and payload verifier are replaced with data-only fixtures.
The capture itself uses the original pure submitted-input validator. No file,
Rules, native, physical or trainer admission is proved by these assembly tests.
"""
import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib.abc
import json
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest import mock


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}
                or fullname in {"rz_pals_model.model", "rz_pals_model.training",
                    "rz_pals_model.nonzero_training", "rz_pals_model.nnuec_trainable",
                    "rz_pals_model.nnuec_utility_trainable"}):
            raise AssertionError("utility route assembly cannot import a neural runtime")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_utility_route as route
    from rz_pals_model import rovent_utility_training as training
    from rz_pals_model.rovent_utility_cohort_files import HistoricalUtilityCohortFiles
    from rz_pals_model.rovent_utility_input import validate_submitted_utility_input
    from rz_pals_model.rovent_historical_corpus import HistoricalRoventCorpus
    from rz_pals_model.rovent_historical_r_corpus import HistoricalRoventRCorpus
    from rz_pals_model.rovent_corpus_codec import LEGACY_CODEC, SUBMITTED_CODEC
    from test_rovent_utility_input import fixture as capture_fixture, input_mapping
    from test_rovent_utility_cohort_files import (publication_fixture, ack_fixture,
        raw as json_file_bytes, file_pin, payloads)
finally:
    sys.meta_path.remove(_guard)


def seal(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def framed(domain, body):
    """Independent compact UTF8 framing, not the production digest helper."""
    return hashlib.sha256(json.dumps([domain, body], sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def capture(*, different_context=False):
    value = input_mapping()
    if different_context:
        value["base_input"]["context_lines"][0]["kind"] = 2
    return validate_submitted_utility_input(**capture_fixture(value=value))


def preparation_fixture(captured, *, cohort=True):
    owner = "synthetic-utility-route-owner"
    config = captured.config.to_mapping()
    width = 2 * config["board_width"] + 2 * config["context_width"] + config["local_width"]
    spec = {"hidden_width": 2, "input_width": width,
        "wire_to_parameter": {name: "utility." + name for name in training.PARAMETER_FIELDS},
        "parameters": {name: {"checkpoint_path": "parameters." + name, "shape": list(shape),
            "dtype": "float32", "planned_requires_grad": True}
            for name, shape in zip(training.PARAMETER_FIELDS, ((2, width), (2,), (7, 2), (7,)), strict=True)},
        "optimizer_owner": owner, "frozen_components": ["nnuec_base", "p", "c"]}
    registered = {"collection_source_sha256": seal("collection-source"), "trainer_source_sha256": seal("trainer-source"),
        "cost_epoch_sha256": seal("cost-epoch"), "resource_profile_sha256": seal("resource-profile"),
        "cost_scope": training.COHORT_COST_SCOPE if cohort else training.COST_SCOPE,
        "objective": training.COST_OBJECTIVE,
        "frozen_base": {"schema": training.BASE_SCHEMA, "semantics": training.BASE_SEMANTICS,
            "weights_sha256": captured.base_weights_sha256, "config": config},
        "utility": {"schema": training.UTILITY_SCHEMA, "semantics": training.UTILITY_SEMANTICS,
            "initial_weights_sha256": seal("synthetic-initial-utility-declaration"),
            "base_weights_sha256": captured.base_weights_sha256,
            "parameter_spec_sha256": training.parameter_spec_sha256(spec, config, owner)}}
    factory = training.prepare_cohort_training if cohort else training.prepare_training
    return factory(training.make_recipe(owner), registered, spec)


def synthetic_files(label):
    """Fill the detached file owner's data fields without reading real files."""
    publication, blobs = publication_fixture(pair_count=3)
    publication["cohort_id"] = seal("cohort:" + label)
    original_publication = json_file_bytes(publication)
    ack = ack_fixture(publication, original_publication)
    original_ack = json_file_bytes(ack)
    blobs = {**blobs, route.PUBLICATION_FILE: original_publication, route.ACK_FILE: original_ack}
    owner = HistoricalUtilityCohortFiles()
    owner._publication, owner._ack = publication, ack
    owner._raw = blobs
    owner._pins = {name: file_pin(name, body) for name, body in blobs.items()}
    owner._paths = tuple(pin["path"] for pin in payloads(publication))
    owner._observation = {"scope": "synthetic_files_only", "execution_authorized": False}
    return owner


def report_fixture(files, captured, prepared, *, split="train", ancestors=(), ancestor_descriptors=(), label="current"):
    registered = prepared["registration"]
    all_pairs = [seal(label + ":pair:" + str(index)) for index in range(3)]
    return {"cohort_id": files.publication()["cohort_id"], "profile_sha256": seal(label + ":profile"),
        "usage": "collection_candidate", "collection_source_sha256": registered["collection_source_sha256"],
        "cost_epoch_sha256": registered["cost_epoch_sha256"], "resource_profile_sha256": registered["resource_profile_sha256"],
        "frozen_base_weights_sha256": captured.base_weights_sha256, "input_reference_json": captured.input_json,
        "input_sha256": captured.input_sha256, "target_sha256": captured.target_sha256,
        "submission_target_sha256": captured.submission_target_sha256, "utility_context_sha256": captured.context_sha256,
        "known_pairs": [[6, 4], [5, 4]], "known_pair_sha256": [all_pairs[2], all_pairs[0]], "all_pair_sha256": all_pairs,
        "frozen_pc_sources": [{"role": role, "artifact": {"path": "/synthetic-only/" + role + ".onnx",
            "sha256": seal(role + ":raw-file"), "bytes": 37, "source": "synthetic-only",
            "license": "synthetic-only"}, "model_semantics_sha256": seal(role + ":semantics"),
            "model_identity_sha256": seal(role + ":identity"), "encoding_sha256": seal(role + ":encoding")}
            for role in ("critic", "proposer")],
        "split": split, "opening_id": label + "-opening", "game_id": label + "-game",
        "line_genealogy_id": label + "-line", "transposition_sha256": seal(label + ":transposition"),
        "rules_state_sha256": seal(label + ":rules-state"),
        "ancestor_dataset_publication_sha256": sorted(ancestors),
        "ancestor_datasets": copy.deepcopy(list(ancestor_descriptors))}


def synthetic_corpus(owner):
    metadata = owner._synthetic_metadata
    return SimpleNamespace(_records=owner._synthetic_records,
        corpus_sha256=metadata["manifest_file_sha256"], index_sha256=metadata["index_file_sha256"],
        manifest_semantic_sha256=metadata["canonical_manifest_sha256"],
        index_semantic_sha256=metadata["canonical_index_sha256"])


def ancestor_descriptor(owner):
    """Expected descriptor from fixture getters, not the route's assembler."""
    if type(owner) is route.HistoricalUtilityRouteDataset:
        observed = owner.observation()
        return {"kind": "utility_cohort", "publication_sha256": observed["publication_sha256"],
            "metadata": {"dataset_sha256": observed["dataset_sha256"], "pool_sha256": owner.pool_sha256,
                "preparation_sha256": observed["preparation_sha256"], "context_sha256": owner.context_ids[0]}}
    return {"kind": owner._synthetic_kind, "publication_sha256": owner._synthetic_publication_sha256,
        "metadata": copy.deepcopy(owner._synthetic_metadata)}


class UtilityRouteAssemblyTests(unittest.TestCase):
    def setUp(self):
        sys.meta_path.insert(0, _guard)
        self.addCleanup(lambda: sys.meta_path.remove(_guard))
        self.captured = capture()
        self.preparation = preparation_fixture(self.captured)
        self.reader = mock.Mock(name="synthetic_file_reader")
        reader_patch = mock.patch.object(route, "read_utility_cohort_files", self.reader)
        reader_patch.start()
        self.addCleanup(reader_patch.stop)
        self.verifier = mock.Mock(name="synthetic_body_report_verifier")
        body = ModuleType("rz_pals_model.rovent_utility_cohort_payload")
        body.validate_cohort_payloads = self.verifier
        body_patch = mock.patch.dict(sys.modules, {body.__name__: body})
        body_patch.start()
        self.addCleanup(body_patch.stop)
        # These exact typed PC/R fixtures exercise only the route's traversal
        # and union algorithm. Their original readers are explicitly replaced.
        for owner_type in (HistoricalRoventCorpus, HistoricalRoventRCorpus):
            for patcher in (mock.patch.object(owner_type, "observation", lambda item: {
                    "publication_sha256": item._synthetic_publication_sha256}),
                    mock.patch.object(owner_type, "corpus", property(synthetic_corpus))):
                patcher.start()
                self.addCleanup(patcher.stop)

    def build(self, *, files=None, report=None, captured=None, prepared=None, ancestors=(), label="current"):
        files = synthetic_files(label) if files is None else files
        captured = self.captured if captured is None else captured
        prepared = self.preparation if prepared is None else prepared
        report = report_fixture(files, captured, prepared, label=label) if report is None else report
        self.reader.return_value = files
        self.verifier.return_value = {"input": captured, "report": report}
        reference, shared_budget = object(), object()
        result = route.HistoricalUtilityRouteDataset.from_registration(reference,
            preparation=prepared, read_budget=shared_budget, ancestors=ancestors)
        self.reader.assert_called_with(reference, read_accounting=shared_budget)
        self.verifier.assert_called_with(files, preparation=prepared)
        return result

    def typed_ancestor(self, kind, *, publication, rows):
        owner_type = HistoricalRoventCorpus if kind in ("pc", "pc_legacy", "pc_submitted_input_v1") else HistoricalRoventRCorpus
        owner = owner_type.__new__(owner_type)
        owner._synthetic_publication_sha256, owner._synthetic_records = publication, copy.deepcopy(rows)
        owner._synthetic_kind = ("pc_submitted_input_v1" if kind == "pc_submitted_input_v1" else
            "pc_legacy" if owner_type is HistoricalRoventCorpus else "r_auxiliary")
        owner._synthetic_metadata = {name: seal(publication + ":" + name) for name in
            ("manifest_file_sha256", "index_file_sha256", "canonical_manifest_sha256", "canonical_index_sha256")}
        if owner_type is HistoricalRoventCorpus:
            owner._codec = SUBMITTED_CODEC if kind == "pc_submitted_input_v1" else LEGACY_CODEC
        return owner

    def row(self, kind, *, split="train", label="other", opening=None, rules_state=None):
        lineage = {"opening_id": opening if opening is not None else label + "-opening",
            "game_id": label + "-game", "line_genealogy_id": label + "-line",
            "transposition_sha256": seal(label + ":transposition")}
        state = rules_state if rules_state is not None else seal(label + ":rules-state")
        if kind == "pc":
            return {"split": split, "input_sha256": seal(label + ":input"),
                "input": {**lineage, "view_sha256": seal(label + ":view"), "rules_anchor": {"rules_state_sha256": state}}}
        return {"split": split, "lineage": lineage,
            "input": {"semantic_input_sha256": seal(label + ":semantic-input"), "native_input_sha256": seal(label + ":native-input")},
            "bound": {"current_rules": {"anchor": {"rules_state_sha256": state}}}}

    def test_cohort_preparation_is_explicit_and_legacy_preparation_is_rejected_before_read(self):
        owner = self.build()
        self.assertEqual(owner.observation()["preparation_sha256"], self.preparation["preparation_sha256"])
        legacy = preparation_fixture(self.captured, cohort=False)
        before = copy.deepcopy(legacy)
        self.reader.reset_mock()
        self.verifier.reset_mock()
        with self.assertRaisesRegex(ValueError, "preparation/2"):
            route.HistoricalUtilityRouteDataset.from_registration(object(), preparation=legacy, read_budget=object())
        self.reader.assert_not_called()
        self.verifier.assert_not_called()
        self.assertEqual(legacy, before)

    def test_assembly_uses_original_raw_metadata_pins_and_independent_context_pool_dataset_framing(self):
        files = synthetic_files("framing")
        report = report_fixture(files, self.captured, self.preparation)
        owner = self.build(files=files, report=report)
        body = {"schema": route.CONTEXT_DOMAIN,
            "publication_sha256": files.file_pin(route.PUBLICATION_FILE)["sha256"],
            "durable_ack_sha256": files.file_pin(route.ACK_FILE)["sha256"],
            **{key: copy.deepcopy(report[key]) for key in ("cohort_id", "profile_sha256", "input_sha256",
                "target_sha256", "submission_target_sha256", "utility_context_sha256", "collection_source_sha256",
                "cost_epoch_sha256", "resource_profile_sha256", "frozen_base_weights_sha256", "split",
                "opening_id", "game_id", "line_genealogy_id", "transposition_sha256",
                "rules_state_sha256", "ancestor_dataset_publication_sha256", "all_pair_sha256", "known_pair_sha256", "known_pairs")},
            "cost_scope": training.COHORT_COST_SCOPE, "execution_authorized": False}
        context = framed(route.CONTEXT_DOMAIN, body)
        body["context_sha256"] = context
        pool = framed(route.POOL_DOMAIN, [context])
        publication_pin = {key: files.file_pin(route.PUBLICATION_FILE)[key] for key in ("sha256", "bytes")}
        ack_pin = {key: files.file_pin(route.ACK_FILE)[key] for key in ("sha256", "bytes")}
        dataset = framed(route.DATASET_DOMAIN, {"publication": publication_pin, "durable_ack": ack_pin,
            "contexts": [context], "pool_sha256": pool, "source_registration_sha256": self.preparation["registration_sha256"]})
        self.assertEqual(owner.context_ids, (context,))
        self.assertEqual(owner.pool_sha256, pool)
        self.assertEqual(owner.publication_pin, publication_pin)
        self.assertEqual(owner.durable_ack_pin, ack_pin)
        self.assertEqual(owner.sample(0)["source_projection"], body)
        self.assertNotIn("ancestor_datasets", owner.sample(0)["source_projection"])
        self.assertEqual(owner.observation()["dataset_sha256"], dataset)
        self.assertEqual(owner.observation()["rules_state_sha256"], report["rules_state_sha256"])
        self.assertNotEqual(publication_pin["sha256"], framed("not-a-raw-file", files.publication()))

    def test_repeated_sample_preserves_the_same_immutable_capture_and_native_pair_order(self):
        owner = self.build()
        first, second = owner.sample(0), owner.sample(0)
        self.assertIs(first["input"], self.captured)
        self.assertIs(second["input"], self.captured)
        self.assertEqual(first["known_pairs"], [[6, 4], [5, 4]])
        first["known_pairs"].reverse()
        first["source_projection"]["known_pairs"].clear()
        self.assertEqual(second["known_pairs"], [[6, 4], [5, 4]])
        self.assertEqual(owner.sample(0)["known_pairs"], [[6, 4], [5, 4]])
        with self.assertRaises(FrozenInstanceError):
            self.captured.input_sha256 = seal("mutated")
        self.assertFalse(self.captured.observation()["training_target_authorized"])

    def test_historical_context_exposes_only_known_pair_pins_without_a_lease(self):
        files = synthetic_files("known-subset")
        report = report_fixture(files, self.captured, self.preparation)
        owner = self.build(files=files, report=report)
        context = owner.historical_context(owner.context_ids[0])
        self.assertEqual(context["pair_sha256"], report["known_pair_sha256"])
        self.assertEqual(context["unique_known_pairs"], 2)
        self.assertNotEqual(context["pair_sha256"], report["all_pair_sha256"])
        context["source_projection"]["known_pairs"].clear()
        self.assertEqual(owner.historical_context(owner.context_ids[0])["unique_known_pairs"], 2)
        observation = owner.observation()
        for key in ("past_native_authority_reissued", "past_physical_completion_reissued", "execution_authorized"):
            self.assertIs(observation[key], False)
        self.assertEqual(observation["rules_replay_scope"], "original_rust_parent_required")

    def test_validation_and_holdout_contexts_are_not_training_samples(self):
        for split in ("validation", "holdout"):
            files = synthetic_files(split)
            report = report_fixture(files, self.captured, self.preparation, split=split)
            owner = self.build(files=files, report=report)
            self.assertEqual(owner.historical_context(owner.context_ids[0])["unique_known_pairs"], 2)
            with self.subTest(split=split), self.assertRaises(ValueError):
                owner.sample(0)

    def test_only_exact_original_index_and_context_can_be_requested(self):
        owner = self.build()
        for index in (-1, 1, True, 0.0, "0"):
            with self.subTest(index=index), self.assertRaises(ValueError):
                owner.sample(index)
        with self.assertRaises(ValueError):
            owner.historical_context(seal("unregistered-context"))
        with self.assertRaises(TypeError):
            route.HistoricalUtilityRouteDataset()

    def test_body_projection_and_report_shapes_do_not_accept_extra_or_missing_fields(self):
        files = synthetic_files("strict-fields")
        report = report_fixture(files, self.captured, self.preparation)
        self.reader.return_value = files
        for original in ({"input": self.captured, "report": report, "authorized": True}, {"report": report}):
            self.verifier.return_value = original
            with self.assertRaises(ValueError):
                route.HistoricalUtilityRouteDataset.from_registration(object(),
                    preparation=self.preparation, read_budget=object())
        for mutate in (lambda value: value.update(execution_authorized=True),
                       lambda value: value.pop("profile_sha256"),
                       lambda value: value.pop("rules_state_sha256"),
                       lambda value: value.pop("ancestor_datasets")):
            changed = copy.deepcopy(report)
            mutate(changed)
            with self.assertRaises(ValueError):
                self.build(files=files, report=changed)

    def test_report_rejects_changed_source_cost_base_capture_config_and_usage(self):
        files = synthetic_files("source-checks")
        original = report_fixture(files, self.captured, self.preparation)
        changes = [(key, seal("wrong:" + key)) for key in ("cohort_id", "collection_source_sha256",
            "cost_epoch_sha256", "resource_profile_sha256", "frozen_base_weights_sha256", "input_sha256",
            "target_sha256", "submission_target_sha256", "utility_context_sha256")]
        changes += [("usage", "deployment"), ("split", "unknown"),
            ("rules_state_sha256", True), ("rules_state_sha256", "not-a-sha"),
            ("input_reference_json", self.captured.input_json + " ")]
        for key, value in changes:
            changed = copy.deepcopy(original)
            changed[key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.build(files=files, report=changed)
        changed_preparation = copy.deepcopy(self.preparation)
        registered = changed_preparation["registration"]
        registered["frozen_base"]["config"]["max_lines"] = 8
        changed_preparation = training.prepare_cohort_training(changed_preparation["recipe"], registered,
            changed_preparation["parameter_spec"])
        with self.assertRaises(ValueError):
            self.build(files=files, report=original, prepared=changed_preparation)
        with self.assertRaises(ValueError):
            self.build(files=files, report=original, captured=SimpleNamespace(**vars(self.captured)))

    def test_known_pair_duplicates_reverse_directions_masks_and_unsupported_actions_are_rejected(self):
        files = synthetic_files("pairs")
        original = report_fixture(files, self.captured, self.preparation)
        for pairs in ([[6, 4], [6, 4]], [[6, 4], [4, 6]], [None], [[3, 4]], [[4, 4]], [[4, True]]):
            report = copy.deepcopy(original)
            report["known_pairs"] = pairs
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                self.build(files=files, report=report)

    def test_known_pair_raw_seals_must_be_unique_and_belong_to_the_full_publication_inventory(self):
        files = synthetic_files("pair-pins")
        original = report_fixture(files, self.captured, self.preparation)
        for mutate in (lambda value: value["known_pair_sha256"].pop(),
                       lambda value: value["known_pair_sha256"].__setitem__(1, value["known_pair_sha256"][0]),
                       lambda value: value["known_pair_sha256"].__setitem__(1, seal("not-in-original-pairs")),
                       lambda value: value["all_pair_sha256"].pop(),
                       lambda value: value["all_pair_sha256"].__setitem__(1, value["all_pair_sha256"][0])):
            report = copy.deepcopy(original)
            mutate(report)
            with self.assertRaises(ValueError):
                self.build(files=files, report=report)

    def test_original_frozen_pc_descriptors_require_exact_distinct_roles_and_raw_pins(self):
        files = synthetic_files("pc-sources")
        original = report_fixture(files, self.captured, self.preparation)
        for mutate in (lambda sources: sources.pop(),
                       lambda sources: sources[1].update(role=sources[0]["role"]),
                       lambda sources: sources[0].update(role="validator"),
                       lambda sources: sources[0].update(extra="unknown"),
                       lambda sources: sources[0]["artifact"].update(bytes=True),
                       lambda sources: sources[0]["artifact"].pop("source"),
                       lambda sources: sources[0]["artifact"].update(license=""),
                       lambda sources: sources[0]["artifact"].update(extra="unknown"),
                       lambda sources: sources[0].update(encoding_sha256="bad")):
            report = copy.deepcopy(original)
            mutate(report["frozen_pc_sources"])
            with self.assertRaises(ValueError):
                self.build(files=files, report=report)

    def test_execution_sources_match_original_raw_files_and_pc_semantics_independently_of_declared_paths(self):
        files = synthetic_files("execution-source")
        report = report_fixture(files, self.captured, self.preparation)
        original_report = copy.deepcopy(report)
        owner = self.build(files=files, report=report)
        registered = {"current_dataset": {"publication": {"path": "/relocated/publication.json", **owner.publication_pin},
            "durable_ack": {"path": "/relocated/ack.json", **owner.durable_ack_pin}},
            "frozen_nnuec_base": files.publication()["nnuec_weights"],
            "frozen_pc_sources": [{"role": source["role"],
                "artifact": {key: source["artifact"][key] for key in ("path", "sha256", "bytes")},
                "model_semantics_sha256": source["model_semantics_sha256"],
                "model_identity_sha256": source["model_identity_sha256"],
                "encoding_sha256": source["encoding_sha256"]} for source in report["frozen_pc_sources"]]}
        for original, current in zip(report["frozen_pc_sources"], registered["frozen_pc_sources"], strict=True):
            self.assertEqual(set(original["artifact"]), {"path", "sha256", "bytes", "source", "license"})
            self.assertEqual(set(current["artifact"]), {"path", "sha256", "bytes"})
        self.assertIsNone(owner.validate_execution_sources(registered))
        for mutate in (lambda value: value["current_dataset"]["publication"].update(sha256=seal("wrong-publication")),
                       lambda value: value["current_dataset"]["durable_ack"].update(bytes=1),
                       lambda value: value["frozen_nnuec_base"].update(sha256=self.captured.base_weights_sha256),
                       lambda value: value["frozen_pc_sources"][0]["artifact"].update(bytes=1),
                       lambda value: value["frozen_pc_sources"][0].update(model_identity_sha256=seal("wrong-model")),
                       lambda value: value["frozen_pc_sources"][0].update(encoding_sha256=seal("wrong-encoding")),
                       lambda value: value["frozen_pc_sources"][1].update(role=value["frozen_pc_sources"][0]["role"])):
            changed = copy.deepcopy(registered)
            mutate(changed)
            with self.assertRaises(ValueError):
                owner.validate_execution_sources(changed)
        changed = copy.deepcopy(registered)
        changed["frozen_pc_sources"].reverse()
        for source in changed["frozen_pc_sources"]:
            source["artifact"]["path"] = "/relocated/" + source["role"] + ".onnx"
        self.assertIsNone(owner.validate_execution_sources(changed))
        self.assertEqual(report, original_report)

    def test_metadata_changes_invalidate_the_owner_instead_of_resealing_it(self):
        mutations = (lambda owner: owner._report.update(profile_sha256=seal("changed-profile")),
            lambda owner: owner._report.update(rules_state_sha256=seal("changed-rules-state")),
            lambda owner: owner._preparation["registration"].update(cost_epoch_sha256=seal("changed-cost")),
            lambda owner: owner._projection.update(execution_authorized=True),
            lambda owner: setattr(owner, "_pool_sha256", seal("changed-pool")),
            lambda owner: owner._publication_pin.update(bytes=1))
        for mutate in mutations:
            owner = self.build()
            mutate(owner)
            with self.assertRaisesRegex(ValueError, "metadata changed"):
                owner.observation()

    def test_replacing_capture_fields_cannot_preserve_only_its_two_string_seals(self):
        replacements = (
            {"input_bytes": self.captured.input_bytes + b" "},
            {"reference_input": replace(self.captured.reference_input, target_ids=(1, 0))},
            {"config": replace(self.captured.config, max_lines=8)},
            {"target_ids": (1,)},
            {"view_seal_sha256": seal("changed-view")},
            {"target_sha256": seal("changed-target")},
            {"submission_target_sha256": seal("changed-submission")},
        )
        for changed in replacements:
            owner = self.build()
            owner._capture = replace(self.captured, **changed)
            self.assertEqual(owner._capture.input_json, self.captured.input_json)
            self.assertEqual(owner._capture.input_sha256, self.captured.input_sha256)
            with self.subTest(fields=tuple(changed)), self.assertRaises(ValueError):
                owner.sample(0)

    def test_other_well_formed_file_owner_cannot_replace_the_original_metadata_pins(self):
        owner = self.build()
        original = owner.publication_pin
        other = synthetic_files("different-original-files")
        self.assertNotEqual(other.file_pin(route.PUBLICATION_FILE)["sha256"], original["sha256"])
        owner._files = other
        with self.assertRaises(ValueError):
            owner.observation()

    def test_ancestor_declarations_require_sorted_distinct_exact_publication_coverage(self):
        files = synthetic_files("declared-ancestors")
        original = report_fixture(files, self.captured, self.preparation)
        for pins in ([seal("undeclared-owner")], sorted([seal("b"), seal("a")], reverse=True), [seal("same"), seal("same")]):
            report = copy.deepcopy(original)
            report["ancestor_dataset_publication_sha256"] = pins
            with self.assertRaises(ValueError):
                self.build(files=files, report=report)
        owner = self.build()
        fake = SimpleNamespace(observation=lambda: {"publication_sha256": seal("untyped-ancestor")})
        with self.assertRaisesRegex(ValueError, "typed PC/R/cohort"):
            owner.validate_ancestors((fake,))
        with self.assertRaises(ValueError):
            owner.validate_ancestors([fake] * 257)

    def test_typed_pc_and_r_ancestors_scan_every_declared_row_and_preserve_same_split_edges(self):
        pc_pin, r_pin = seal("pc-publication"), seal("r-publication")
        pc = self.typed_ancestor("pc", publication=pc_pin, rows=[
            self.row("pc", label="unrelated"), self.row("pc", label="bridge", opening="current-opening")])
        researcher = self.typed_ancestor("r", publication=r_pin, rows=[self.row("r", label="r-only")])
        files = synthetic_files("mixed")
        report = report_fixture(files, self.captured, self.preparation, ancestors=(pc_pin, r_pin),
            ancestor_descriptors=(ancestor_descriptor(pc), ancestor_descriptor(researcher)))
        owner = self.build(files=files, report=report, ancestors=(pc, researcher))
        observed = owner.validate_ancestors((researcher, pc))
        self.assertEqual(observed["ancestor_publication_sha256"], sorted((pc_pin, r_pin)))
        self.assertEqual(observed["inspected_occurrences"], 3)
        self.assertEqual(observed["ancestor_cross_split_components"], 0)
        self.assertFalse(observed["execution_authorized"])
        self.assertFalse(observed["past_native_authority_reissued"])

    def test_ancestor_direct_or_indirect_cross_split_leakage_rejects_the_current_component(self):
        for kind in ("pc", "r"):
            pin = seal(kind + ":direct-leak")
            ancestor = self.typed_ancestor(kind, publication=pin,
                rows=[self.row(kind, split="validation", opening="current-opening")])
            files = synthetic_files(kind + ":direct")
            report = report_fixture(files, self.captured, self.preparation, ancestors=(pin,),
                ancestor_descriptors=(ancestor_descriptor(ancestor),))
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "crosses global splits"):
                self.build(files=files, report=report, ancestors=(ancestor,))
        state, pc_pin, r_pin = seal("indirect-rules-state"), seal("indirect-pc"), seal("indirect-r")
        pc = self.typed_ancestor("pc", publication=pc_pin,
            rows=[self.row("pc", label="bridge", opening="current-opening", rules_state=state)])
        researcher = self.typed_ancestor("r", publication=r_pin,
            rows=[self.row("r", label="holdout-only", split="holdout", rules_state=state)])
        files = synthetic_files("indirect")
        report = report_fixture(files, self.captured, self.preparation, ancestors=(pc_pin, r_pin),
            ancestor_descriptors=(ancestor_descriptor(pc), ancestor_descriptor(researcher)))
        with self.assertRaisesRegex(ValueError, "crosses global splits"):
            self.build(files=files, report=report, ancestors=(pc, researcher))

    def test_unrelated_ancestor_cross_split_component_is_observed_without_rejecting_current_context(self):
        state, pc_pin, r_pin = seal("unrelated-shared-state"), seal("unrelated-pc"), seal("unrelated-r")
        pc = self.typed_ancestor("pc", publication=pc_pin, rows=[self.row("pc", label="away", rules_state=state)])
        researcher = self.typed_ancestor("r", publication=r_pin,
            rows=[self.row("r", label="far", split="holdout", rules_state=state)])
        files = synthetic_files("unrelated-component")
        report = report_fixture(files, self.captured, self.preparation, ancestors=(pc_pin, r_pin),
            ancestor_descriptors=(ancestor_descriptor(pc), ancestor_descriptor(researcher)))
        owner = self.build(files=files, report=report, ancestors=(pc, researcher))
        self.assertEqual(owner.validate_ancestors((pc, researcher))["ancestor_cross_split_components"], 1)

    def test_typed_utility_ancestor_keeps_native_context_seal_edges_and_cannot_be_repeated_or_self(self):
        ancestor = self.build(label="ancestor")
        pin = ancestor.publication_pin["sha256"]
        files = synthetic_files("child")
        report = report_fixture(files, self.captured, self.preparation, ancestors=(pin,),
            ancestor_descriptors=(ancestor_descriptor(ancestor),), label="different-lineage")
        owner = self.build(files=files, report=report, ancestors=(ancestor,))
        self.assertEqual(owner.validate_ancestors((ancestor,))["inspected_occurrences"], 1)
        with self.assertRaises(ValueError):
            owner.validate_ancestors((ancestor, ancestor))
        with self.assertRaises(ValueError):
            owner.validate_ancestors((owner,))
        report["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "crosses global splits"):
            self.build(files=files, report=report, ancestors=(ancestor,))

    def test_all_four_ancestor_kinds_join_each_original_metadata_seal_and_detach_descriptors(self):
        for kind in ("pc_legacy", "pc_submitted_input_v1", "r_auxiliary", "utility_cohort"):
            if kind == "utility_cohort":
                ancestor = self.build(label="four-seal-utility-ancestor")
            else:
                row_kind = "r" if kind == "r_auxiliary" else "pc"
                ancestor = self.typed_ancestor(kind, publication=seal(kind + ":publication"),
                    rows=[self.row(row_kind, label=kind + "-unrelated")])
            descriptor = ancestor_descriptor(ancestor)
            self.assertEqual(descriptor["kind"], kind)
            self.assertEqual(len(descriptor["metadata"]), 4)
            files = synthetic_files(kind + ":descriptor-child")
            report = report_fixture(files, self.captured, self.preparation,
                ancestors=(descriptor["publication_sha256"],), ancestor_descriptors=(descriptor,))
            owner = self.build(files=files, report=report, ancestors=(ancestor,))
            self.assertEqual(owner.source_ancestor_descriptors(), [descriptor])
            detached = owner.source_ancestor_descriptors()
            detached[0]["metadata"].clear()
            self.assertEqual(owner.source_ancestor_descriptors(), [descriptor])
            self.assertEqual(owner.sample(0)["source_projection"]["ancestor_dataset_publication_sha256"],
                [descriptor["publication_sha256"]])
            self.assertNotIn("ancestor_datasets", owner.sample(0)["source_projection"])
            for metadata_key in descriptor["metadata"]:
                changed = copy.deepcopy(report)
                changed["ancestor_datasets"][0]["metadata"][metadata_key] = seal("different:" + metadata_key)
                with self.subTest(kind=kind, seal=metadata_key), self.assertRaises(ValueError):
                    self.build(files=files, report=changed, ancestors=(ancestor,))
            changed = copy.deepcopy(report)
            changed["ancestor_datasets"][0]["kind"] = (
                "pc_submitted_input_v1" if kind == "pc_legacy" else "pc_legacy")
            with self.subTest(kind=kind, field="kind"), self.assertRaises(ValueError):
                self.build(files=files, report=changed, ancestors=(ancestor,))

    def test_ancestor_descriptors_are_closed_typed_and_cover_the_original_publication_set(self):
        ancestor = self.typed_ancestor("pc_legacy", publication=seal("descriptor-shape-publication"),
            rows=[self.row("pc", label="descriptor-shape-row")])
        descriptor = ancestor_descriptor(ancestor)
        files = synthetic_files("descriptor-shapes")
        original = report_fixture(files, self.captured, self.preparation,
            ancestors=(descriptor["publication_sha256"],), ancestor_descriptors=(descriptor,))
        mutations = (
            lambda report: report["ancestor_datasets"].clear(),
            lambda report: report["ancestor_dataset_publication_sha256"].clear(),
            lambda report: report["ancestor_datasets"].append(copy.deepcopy(report["ancestor_datasets"][0])),
            lambda report: report["ancestor_datasets"][0].update(extra="not-an-original-field"),
            lambda report: report["ancestor_datasets"][0].pop("metadata"),
            lambda report: report["ancestor_datasets"][0].update(kind="pc"),
            lambda report: report["ancestor_datasets"][0].update(publication_sha256=seal("different-publication")),
            lambda report: report["ancestor_datasets"][0]["metadata"].pop("index_file_sha256"),
            lambda report: report["ancestor_datasets"][0]["metadata"].update(extra=seal("extra")),
            lambda report: report["ancestor_datasets"][0]["metadata"].update(manifest_file_sha256=True),
            lambda report: report["ancestor_datasets"][0]["metadata"].update(canonical_index_sha256="not-a-sha"),
        )
        for index, mutate in enumerate(mutations):
            report = copy.deepcopy(original)
            mutate(report)
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.build(files=files, report=report, ancestors=(ancestor,))
        ancestor._codec = object()
        with self.assertRaisesRegex(ValueError, "exact source family"):
            self.build(files=files, report=original, ancestors=(ancestor,))

    def test_exact_rules_state_alone_connects_current_to_pc_r_or_utility_cross_split_ancestors(self):
        current_state = seal("current:rules-state")
        for kind in ("pc", "r"):
            ancestor = self.typed_ancestor(kind, publication=seal(kind + ":rules-only-publication"),
                rows=[self.row(kind, split="validation", label=kind + "-rules-only", rules_state=current_state)])
            descriptor = ancestor_descriptor(ancestor)
            files = synthetic_files(kind + ":rules-only-child")
            report = report_fixture(files, self.captured, self.preparation,
                ancestors=(descriptor["publication_sha256"],), ancestor_descriptors=(descriptor,))
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "crosses global splits"):
                self.build(files=files, report=report, ancestors=(ancestor,))
        different_capture = capture(different_context=True)
        ancestor_files = synthetic_files("utility-rules-only-ancestor")
        ancestor_report = report_fixture(ancestor_files, different_capture, self.preparation,
            split="validation", label="utility-rules-only-ancestor")
        ancestor_report["rules_state_sha256"] = current_state
        ancestor = self.build(files=ancestor_files, report=ancestor_report, captured=different_capture)
        descriptor = ancestor_descriptor(ancestor)
        files = synthetic_files("utility-rules-only-child")
        report = report_fixture(files, self.captured, self.preparation,
            ancestors=(descriptor["publication_sha256"],), ancestor_descriptors=(descriptor,))
        for key in ("input_sha256", "target_sha256", "submission_target_sha256", "utility_context_sha256"):
            self.assertNotEqual(report[key], ancestor_report[key], key)
        for key in ("opening_id", "game_id", "line_genealogy_id", "transposition_sha256"):
            self.assertNotEqual(report[key], ancestor_report[key], key)
        with self.assertRaisesRegex(ValueError, "crosses global splits"):
            self.build(files=files, report=report, ancestors=(ancestor,))


if __name__ == "__main__":
    unittest.main()
