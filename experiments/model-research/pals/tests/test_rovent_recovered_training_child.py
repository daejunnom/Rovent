"""Recovered entry source contracts; no model/optimizer or learner execution."""
import builtins
import copy
import hashlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_rovent_training import NeuralImportsForbidden, membership_fixture, recipe_fixture
from test_rovent_training_bootstrap import recovered_fixture
from test_rovent_training_window_v2 import spec_fixture

_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_recovered_training_child as child
finally:
    sys.meta_path.remove(_guard)


def rules_fixture():
    from rz_pals_model.artifacts import VIRTUAL_RULES_BOUNDS, VIRTUAL_RULES_FEATURES
    from rz_pals_model.config import ModelConfig
    from rz_pals_model.virtual_review import RECORD_FEATURES, QUERY_FEATURES
    profile, base = "rz-pals-rules-virtual-review-v1", "fixture_contract_not_Rules_execution"
    metadata = ["metadata" + str(i) for i in range(16)]
    divergence = ["divergence" + str(i) for i in range(8)]
    semantic_fields = [profile, base, *metadata, *RECORD_FEATURES, *QUERY_FEATURES, *divergence, *VIRTUAL_RULES_FEATURES]
    semantic = hashlib.sha256()
    for field in semantic_fields:
        raw = field.encode("utf-8")
        semantic.update(len(raw).to_bytes(8, "little"))
        semantic.update(raw)
    return {"schema": "rovezero.pals-rules-descriptor.v3", "rules_input_profile": profile,
        "config": ModelConfig.for_profile("virtual_review_v1").to_dict(), "rules_base_semantics": base,
        "metadata_features": metadata, "divergence_features": divergence,
        "record_features": list(RECORD_FEATURES), "query_features": list(QUERY_FEATURES),
        "virtual_review_features": VIRTUAL_RULES_FEATURES, "bounds": VIRTUAL_RULES_BOUNDS,
        "semantic_fields": semantic_fields, "semantic_digest_algorithm": "sha256_u64le_length_prefixed_utf8_fields",
        "rules_input_semantic_sha256": semantic.hexdigest(), "rules_encoder_source_sha256": "7" * 64,
        "learned_input_compatibility": "unverified_declaration_only"}


def metadata_fixture():
    membership = membership_fixture()
    recipe = recipe_fixture(membership)
    spec = spec_fixture(recipe)
    startup = recovered_fixture()
    registration = startup["registration"]
    dataset = registration["current_dataset"]
    dataset.update(recipe_sha256=recipe.sha256, stage_id=spec["stage_id"],
        collection_source_sha256=spec["collection_source_sha256"],
        collection_runtime_library_sha256=spec["collection_runtime_library_sha256"],
        model_semantics_sha256=spec["collection_model_semantics_sha256"],
        model_epoch=spec["collection_model_epoch"], context_epoch=spec["collection_context_epoch"])
    startup["window_id"] = spec["window_id"]
    spec.update(trainer_source_sha256=registration["trainer_source_sha256"],
        trainer_python_executable_sha256=registration["trainer_python_executable"]["sha256"],
        trainer_dependency_set_sha256=registration["trainer_dependency_set_sha256"])
    payloads = {registration["recipe"]["path"]: recipe.to_dict(),
        registration["membership"]["path"]: membership, registration["window_spec"]["path"]: spec,
        registration["rules_profile"]["path"]: rules_fixture()}
    class Reads:
        def json(self, pin, *, maximum):
            return copy.deepcopy(payloads[pin["path"]])
    return startup, spec, payloads, Reads()


class RecoveredChildContractTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_ipc_check_uses_validation_only_then_live_begin_fence_and_end(self):
        from rz_pals_model import rovent_recovered_training_window as window
        calls = []
        owner = SimpleNamespace(completed={"proposer": 0}, dispatched={"proposer": 0},
            pending_microbatches=0, quarantined=False, physical_completion="unknown", cuda_devices=["cuda:0"])
        lease = object()
        authority = SimpleNamespace(
            begin_window=lambda *_args: calls.append("begin") or lease,
            end_window=lambda got, **kwargs: calls.append(("end", got, kwargs["status"], owner.physical_completion)))
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda device: calls.append(("fence", device))))
        startup = {"execution_until_ns": 10, "registration": {"checkpoint_directory": "/fixture/checkpoint"}}
        validation = {"actual_training_executed": False, "optimizer_updates": 0, "admission": {"corpus_sha256": "x"}}
        with patch.object(window, "train_rovent_window", return_value=validation) as validate:
            result = child._execute_ipc_check(startup, owner, object(), {}, authority, torch,
                clock=lambda: 1, cancelled=lambda: False)
        self.assertIs(validate.call_args.kwargs["execute"], False)
        self.assertEqual(calls, ["begin", ("fence", "cuda:0"), ("end", lease, "ipc_check_complete", "confirmed")])
        self.assertEqual(result["optimizer_updates"], 0)
        self.assertFalse(result["actual_training_executed"])
        self.assertEqual(result["scope"], "current_child_dataset_and_ipc_only")

    def test_ipc_check_fence_failure_quarantines_without_end_ack(self):
        from rz_pals_model import rovent_recovered_training_window as window
        owner = SimpleNamespace(completed={}, dispatched={}, pending_microbatches=0, quarantined=False,
            physical_completion="unknown", cuda_devices=["cuda:0"])
        authority = SimpleNamespace(begin_window=lambda *_args: object(),
            end_window=lambda *_args, **_kwargs: self.fail("failed fence must not report successful end"))
        def fail_fence(_device):
            raise RuntimeError("fixture fence failure")
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=fail_fence))
        startup = {"execution_until_ns": 10, "registration": {"checkpoint_directory": "/fixture/checkpoint"}}
        validation = {"actual_training_executed": False, "optimizer_updates": 0, "admission": {}}
        with patch.object(window, "train_rovent_window", return_value=validation):
            with self.assertRaisesRegex(RuntimeError, "fixture fence failure"):
                child._execute_ipc_check(startup, owner, object(), {}, authority, torch,
                    clock=lambda: 1, cancelled=lambda: False)
        self.assertTrue(owner.quarantined)
        self.assertEqual(owner.physical_completion, "unknown")

    def test_startup_read_requires_exact_eof_and_bounded_integer(self):
        self.assertEqual(child._read_startup(io.BytesIO(b"{}"), 2), b"{}")
        for payload, count in ((b"{}x", 2), (b"{", 2), (b"", 0), (b"{}", True)):
            with self.subTest(payload=payload, count=count), self.assertRaisesRegex(ValueError, "startup"):
                child._read_startup(io.BytesIO(payload), count)

    def test_no_execute_fails_before_startup_and_runtime(self):
        with self.assertRaisesRegex(ValueError, "explicitly authorize"):
            child.recovered_rovent_child_main(["--startup-sha256", "1" * 64, "--startup-bytes", "2",
                "--execution-until-ns", "100", "--whole-until-ns", "200"], stdin=io.BytesIO(b"{}"))

    def test_metadata_keeps_collection_ort_and_trainer_identities_distinct(self):
        startup, spec, _payloads, reads = metadata_fixture()
        recipe, membership, selected, _rules, binding = child._metadata_inputs(startup, reads)
        self.assertEqual(selected, spec)
        self.assertEqual(selected["training_model_epoch"], 5)
        self.assertEqual(selected["collection_model_epoch"], 2)
        self.assertNotEqual(selected["collection_runtime_library_sha256"], selected["trainer_python_executable_sha256"])
        self.assertEqual(recipe.sha256, selected["recipe_sha256"])
        self.assertEqual(membership, membership_fixture())
        self.assertFalse(binding["execution_authorized"])
        self.assertFalse(binding["past_native_authority_reissued"])

    def test_old_window_codec_and_runtime_alias_are_not_converted(self):
        startup, spec, payloads, reads = metadata_fixture()
        selected = payloads[startup["registration"]["window_spec"]["path"]]
        selected["schema"] = "rz-rovent-train-window/1"
        with self.assertRaisesRegex(ValueError, "legacy conversion"):
            child._metadata_inputs(startup, reads)
        selected["schema"] = spec["schema"] = "rz-rovent-train-window/2"
        selected["runtime_sha256"] = selected.pop("collection_runtime_library_sha256")
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            child._metadata_inputs(startup, reads)

    def test_metadata_rejects_python_pin_relabelled_as_collection_ort(self):
        startup, spec, _payloads, reads = metadata_fixture()
        startup["registration"]["current_dataset"]["collection_runtime_library_sha256"] = spec["trainer_python_executable_sha256"]
        with self.assertRaisesRegex(ValueError, "identity namespace"):
            child._metadata_inputs(startup, reads)

    def test_missing_imported_torch_endpoint_pin_fails_without_read(self):
        class Reads:
            def verify(self, *_args, **_kwargs):
                raise AssertionError("unregistered endpoint must fail before any read")
        fake = SimpleNamespace(__file__="/explicit/not-a-runtime/torch.py", _C=None)
        with self.assertRaisesRegex(ValueError, "runtime pin missing"):
            child._verify_imported_runtime({"runtime_module_pins": []}, Reads(), fake)

    def test_history_report_copies_recorded_evidence_without_reissuing_authority(self):
        startup = recovered_fixture()
        result = child._history_scope(startup)
        self.assertFalse(result["past_native_authority_reissued"])
        self.assertFalse(result["past_physical_completion_reissued"])
        result["collection_admission"]["execution_authorized"] = True
        self.assertFalse(startup["collection_admission"]["execution_authorized"])

    def test_optional_r_presence_keeps_absent_and_null_distinct_without_reading(self):
        startup = {"registration": {}}
        spec = {}
        self.assertEqual(child._r_auxiliary_presence(startup, spec), ("absent", None))
        self.assertEqual(startup, {"registration": {}})
        self.assertEqual(spec, {})
        startup["registration"]["r_auxiliary_dataset"] = None
        spec["nnuec_r_auxiliary"] = None
        self.assertEqual(child._r_auxiliary_presence(startup, spec), ("null", None))
        self.assertIn("r_auxiliary_dataset", startup["registration"])
        self.assertIn("nnuec_r_auxiliary", spec)

    def test_optional_r_presence_rejects_cross_presence_and_null_enabled_mismatch(self):
        for registration, spec in (({}, {"nnuec_r_auxiliary": None}),
                ({"r_auxiliary_dataset": None}, {}),
                ({"r_auxiliary_dataset": {}}, {"nnuec_r_auxiliary": None}),
                ({"r_auxiliary_dataset": None}, {"nnuec_r_auxiliary": {}}),
                ({"r_auxiliary_dataset": []}, {"nnuec_r_auxiliary": {}})):
            with self.subTest(registration=registration, spec=spec), self.assertRaisesRegex(ValueError, "R auxiliary"):
                child._r_auxiliary_presence({"registration": registration}, spec)
        request = {"fixture": "not yet an admitted corpus"}
        state, returned = child._r_auxiliary_presence({"registration": {"r_auxiliary_dataset": request}},
            {"nnuec_r_auxiliary": {"fixture": "not execution authority"}})
        self.assertEqual(state, "enabled")
        self.assertIs(returned, request)

    def test_disabled_r_intake_performs_no_r_import_or_io(self):
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.endswith(("rovent_historical_r_corpus", "rovent_r_corpus", "rovent_r_auxiliary")):
                raise AssertionError("disabled R path must not import a reader or sampler")
            return original_import(name, *args, **kwargs)
        class Reads:
            def __getattr__(self, _name):
                raise AssertionError("disabled R path must not touch a read budget")
        with patch.object(builtins, "__import__", side_effect=guarded_import):
            for registration, spec in (({}, {}), ({"r_auxiliary_dataset": None}, {"nnuec_r_auxiliary": None})):
                with self.subTest(spec=spec):
                    self.assertIsNone(child._intake_r_auxiliary({"registration": registration}, Reads(),
                        object(), spec, pc_corpus_by_publication=object()))

    def test_enabled_r_intake_uses_a_distinct_typed_reader_and_exact_window_pins(self):
        from rz_pals_model import rovent_historical_r_corpus as historical_r
        request = {"fixture": "mocked file reader only"}
        startup = {"registration": {"r_auxiliary_dataset": request}}
        spec = {"nnuec_r_auxiliary": {"manifest_file_sha256": "a" * 64, "index_file_sha256": "b" * 64}}
        pc_owner, recipe, reads = object(), object(), object()
        corpus = SimpleNamespace(corpus_sha256="a" * 64, index_sha256="b" * 64, index_semantic_sha256="d" * 64)
        r_owner = SimpleNamespace(corpus=corpus, observation=lambda: {"publication_sha256": "e" * 64})
        dependency = {"canonical_index_sha256": "d" * 64, "r_ancestor_index_sha256": []}
        with patch.object(historical_r, "read_historical_r_dependencies", return_value=dependency), \
                patch.object(historical_r.HistoricalRoventRCorpus, "from_registration", return_value=r_owner) as reader:
            result = child._intake_r_auxiliary(startup, reads, recipe, spec, pc_corpus_by_publication={"c" * 64: pc_owner})
            self.assertEqual(result, (r_owner, (), {"e" * 64: corpus}))
            reader.assert_called_once_with(request, recipe=recipe, read_budget=reads, pc_corpora=(pc_owner,), r_ancestors=())
            corpus.index_sha256 = "d" * 64
            with self.assertRaisesRegex(ValueError, "exact window file pins"):
                child._intake_r_auxiliary(startup, reads, recipe, spec, pc_corpus_by_publication={"c" * 64: pc_owner})

    def test_missing_r_ancestor_fails_before_loading_a_current_or_ancestor_row(self):
        from rz_pals_model import rovent_historical_r_corpus as historical_r
        startup = {"registration": {"r_auxiliary_dataset": {}, "ancestor_r_auxiliary_datasets": [{"fixture": "orphan"}]}}
        spec = {"nnuec_r_auxiliary": {}}
        dependency = {"canonical_index_sha256": "a" * 64, "r_ancestor_index_sha256": ["b" * 64]}
        with patch.object(historical_r, "read_historical_r_dependencies", return_value=dependency), \
                patch.object(historical_r.HistoricalRoventRCorpus, "from_registration",
                    side_effect=AssertionError("unknown ancestor must not be substituted with current R")):
            with self.assertRaisesRegex(ValueError, "cyclic, foreign, or incomplete"):
                child._intake_r_auxiliary(startup, object(), object(), spec, pc_corpus_by_publication={})

    def test_r_graph_uses_canonical_dependencies_and_retains_raw_publication_map(self):
        from rz_pals_model import rovent_historical_r_corpus as historical_r
        sources = {name: {"fixture": name} for name in ("a", "b", "current")}
        startup = {"registration": {"r_auxiliary_dataset": sources["current"],
            "ancestor_r_auxiliary_datasets": [sources["b"], sources["a"]]}}
        spec = {"nnuec_r_auxiliary": {"manifest_file_sha256": "d" * 64, "index_file_sha256": "e" * 64}}
        canonical = {"a": "1" * 64, "b": "2" * 64, "current": "3" * 64}
        raw = {"a": "7" * 64, "b": "8" * 64, "current": "9" * 64}
        dependencies = {"a": [], "b": [canonical["a"]], "current": [canonical["a"], canonical["b"]]}
        owners = {}
        for name in sources:
            owners[name] = SimpleNamespace(corpus=SimpleNamespace(index_semantic_sha256=canonical[name],
                corpus_sha256=spec["nnuec_r_auxiliary"]["manifest_file_sha256"],
                index_sha256=spec["nnuec_r_auxiliary"]["index_file_sha256"]),
                observation=lambda name=name: {"publication_sha256": raw[name]})
        def dependency(source, *, read_budget):
            name = source["fixture"]
            return {"canonical_index_sha256": canonical[name], "r_ancestor_index_sha256": dependencies[name]}
        def load(source, **_kwargs):
            return owners[source["fixture"]]
        with patch.object(historical_r, "read_historical_r_dependencies", side_effect=dependency), \
                patch.object(historical_r.HistoricalRoventRCorpus, "from_registration", side_effect=load) as reader:
            current, ancestors, publication_map = child._intake_r_auxiliary(startup, object(), object(), spec,
                pc_corpus_by_publication={})
        self.assertIs(current, owners["current"])
        self.assertEqual(ancestors, (owners["a"], owners["b"]))
        self.assertEqual(publication_map, {raw[name]: owners[name].corpus for name in owners})
        self.assertEqual([call.args[0]["fixture"] for call in reader.call_args_list], ["a", "b", "current"])

    def test_r_observation_join_preserves_data_scope_and_checks_raw_pins_count_and_parameter(self):
        observation = {"publication_sha256": "1" * 64, "manifest_file_sha256": "2" * 64,
            "index_file_sha256": "3" * 64, "nnuec_parameter_sha256": "4" * 64, "record_count": 2,
            "past_native_authority_reissued": False, "past_physical_completion_reissued": False,
            "execution_authorized": False}
        ancestor = {"publication_sha256": "5" * 64}
        report = {"current_publication_sha256": observation["publication_sha256"],
            "ancestor_publication_sha256": [ancestor["publication_sha256"]],
            **{name: observation[name] for name in ("manifest_file_sha256", "index_file_sha256", "nnuec_parameter_sha256", "record_count")}}
        startup = {"r_auxiliary_admission": report}
        result = child._join_r_auxiliary_admission(startup, observation, [ancestor])
        self.assertFalse(result["execution_authorized"])
        self.assertFalse(result["past_native_authority_reissued"])
        self.assertFalse(result["past_physical_completion_reissued"])
        for name, value in (("current_publication_sha256", "6" * 64), ("manifest_file_sha256", "6" * 64),
                ("nnuec_parameter_sha256", "6" * 64), ("record_count", 3), ("record_count", True),
                ("ancestor_publication_sha256", [])):
            changed = copy.deepcopy(startup)
            changed["r_auxiliary_admission"][name] = value
            with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, "pinned admission"):
                child._join_r_auxiliary_admission(changed, observation, [ancestor])

    def test_export_request_has_a_separate_domain_and_direct_new_output_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            startup = {"source_root": str(root / "source"), "registration": {"export_request": "fixture-only-pin",
                "output_root": str(root), "checkpoint_directory": str(root / "checkpoints"), "max_output_bytes": 8192}}
            request = {"schema": "rz-rovent-recovered-export-request/1", "kind": "deployment",
                "directory": str(root / "deployment"), "max_output_bytes": 4096}
            class Reads:
                def json(self, _pin, *, maximum):
                    return copy.deepcopy(request)
            self.assertEqual(child._export_request(startup, Reads()), request)
            request["schema"] = "rz-rovent-original-export-request/1"
            with self.assertRaisesRegex(ValueError, "new recovered"):
                child._export_request(startup, Reads())
            request["schema"] = "rz-rovent-recovered-export-request/1"
            request["directory"] = str(root / "nested" / "deployment")
            with self.assertRaisesRegex(ValueError, "new recovered"):
                child._export_request(startup, Reads())
            self.assertEqual(list(root.iterdir()), [])

    def test_incomplete_historical_graph_fails_before_loading_rows(self):
        from rz_pals_model.rovent_historical_corpus import HistoricalRoventCorpus
        startup = recovered_fixture()
        startup["registration"]["ancestor_datasets"] = [{"fixture": "incomplete"}]
        with patch.object(child, "_publication_metadata", return_value=({"canonical_index_sha256": "a" * 64}, ["b" * 64])), \
                patch.object(HistoricalRoventCorpus, "from_registration", side_effect=AssertionError("no unresolved row loading")):
            with self.assertRaisesRegex(ValueError, "cyclic, foreign, or incomplete"):
                child._intake(startup, object())

    def test_unrelated_registered_ancestor_is_not_silently_added_to_current_lineage(self):
        from rz_pals_model.rovent_historical_corpus import HistoricalRoventCorpus
        startup = recovered_fixture()
        startup["registration"]["ancestor_datasets"] = [{"fixture": "ancestor"}]
        fake_owner = SimpleNamespace(corpus=SimpleNamespace(index_semantic_sha256="a" * 64))
        with patch.object(child, "_publication_metadata", side_effect=[({"canonical_index_sha256": "a" * 64}, []), ({}, [])]), \
                patch.object(HistoricalRoventCorpus, "from_registration", return_value=fake_owner):
            with self.assertRaisesRegex(ValueError, "complete current dataset ancestry"):
                child._intake(startup, object())


if __name__ == "__main__":
    unittest.main()
