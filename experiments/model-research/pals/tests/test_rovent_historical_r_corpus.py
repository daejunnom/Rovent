"""Tiny durable R wire fixtures; no model, Rules or original capture owner."""
import copy
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import unittest


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime", "tensorflow", "jax"}:
            raise AssertionError("neural imports forbidden in historical R publication tests")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_historical_r_corpus as historical
    from rz_pals_model import rovent_historical_corpus as common
    from rz_pals_model import rovent_r_readout as wire
    from rz_pals_model import rovent_r_corpus as corpus
    from rz_pals_model import nnuec_reference as base
    from rz_pals_model import nnuec_v2_reference as v2
    from rz_pals_model.generation_data import digest
    from test_rovent_r_corpus import fixture, config, h, raw, artifact, domain_nul, seal
finally:
    sys.meta_path.remove(_guard)


def source_operands():
    dimensions = config()
    parameters = {name: [0.0] * length for name, length in dimensions.parameter_lengths().items()}
    e, c = dimensions.line_width, dimensions.context_width
    extra = {name: [0.0] * length for name, length in zip(v2.PARAMETER_FIELDS, (4*e, 2*e, 2*e, 2*e, c*e), strict=True)}
    return {"schema": v2.SCHEMA, "semantics": v2.SEMANTICS, "base": {"schema": base.SCHEMA, "semantics": base.SEMANTICS,
        "config": dimensions.to_mapping(), "training": {"kind": "untrained"}, "parameters": parameters, "output_scale": 1.0},
        "parameters": extra}


def file_pin(path, body):
    return {"path": path, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}


def cuda_bundle(runtime_sha):
    nvidia = ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12", "libcufft.so.11", "libcurand.so.10",
        "libnvrtc.so.12", "libnvrtc-builtins.so.12.8", "libnvJitLink.so.12", "libcudnn.so.9", "libcudnn_adv.so.9",
        "libcudnn_cnn.so.9", "libcudnn_ops.so.9", "libcudnn_engines_precompiled.so.9", "libcudnn_engines_runtime_compiled.so.9",
        "libcudnn_graph.so.9", "libcudnn_heuristic.so.9")
    files = [{"role": role, "filename": name, "bytes": 1, "sha256": runtime_sha if role == "core" else h(name)} for role, name in
        (("core", "libonnxruntime.so.1.22.0"), ("providers_shared", "libonnxruntime_providers_shared.so"), ("providers_cuda", "libonnxruntime_providers_cuda.so"))]
    files.extend({"role": "nvidia_dependency", "filename": name, "bytes": 1, "sha256": h(name)} for name in nvidia)
    return {"schema_version": 2, "files": files, "loader_aliases": [{"filename": "libcudnn.so.9.8.0", "target": "libcudnn.so.9"}]}


def write_publication(root, *, empty=False, with_bundle=False, include_bundle_provenance=True):
    recipe, manifest, row, lock = fixture()
    weights_body = raw(source_operands())
    nnuec = lock["manifest"]["engine"]["nnuec"]
    nnuec["weights"] = {**nnuec["weights"], "sha256": hashlib.sha256(weights_body).hexdigest(), "bytes": len(weights_body)}
    parameter = row["bound"]["parameter_sha256"]
    provenance = {"domain": "rz-rovent-nnuec-provenance/1", "weights_raw_sha256": nnuec["weights"]["sha256"],
        "semantic_weights_sha256": parameter, **{key: nnuec[key] for key in ("schema", "semantics", "encoding", "implementation_sha256",
            "configuration_sha256", "context_namespace_policy_sha256", "local_namespace_policy_sha256")},
        "training": {"kind": "untrained"}, "diagnostic": False, "provenance": []}
    provenance_body = raw(provenance)
    nnuec["provenance"] = {**nnuec["provenance"], "sha256": hashlib.sha256(provenance_body).hexdigest(), "bytes": len(provenance_body)}
    bundle = cuda_bundle(manifest["collection"]["runtime_sha256"]) if with_bundle else None
    bundle_body = None if bundle is None else raw(bundle)
    if bundle is not None and include_bundle_provenance:
        lock["manifest"]["provenance"] = [{"path": "original-runtime-bundle.json", "sha256": hashlib.sha256(bundle_body).hexdigest(),
            "bytes": len(bundle_body), "source": "fixture-only", "license": "fixture-only"}]
    lock["canonical_sha256"] = digest("rz-rovent-execution-canonical/1", lock["manifest"])
    descriptor = row["root_descriptor"]
    inp = descriptor["input"]
    inp["startup_lock_sha256"] = lock["canonical_sha256"]
    budget = domain_nul(corpus.BUDGET_DOMAIN, {key: value for key, value in inp.items() if key != "clock"})
    descriptor["root_budget_sha256"] = budget
    descriptor["deadline_seal_sha256"] = domain_nul(corpus.DEADLINE_DOMAIN, {"domain": corpus.DEADLINE_DOMAIN,
        "startup_lock_sha256": lock["canonical_sha256"], "policy": inp["policy"], "authority": inp["authority"],
        "root_budget_sha256": budget, "clock": inp["clock"], "cleanup_max_ms": inp["declared_ceilings"]["cleanup_max_ms"]})
    descriptor_sha = domain_nul(corpus.DESCRIPTOR_DOMAIN, descriptor)
    row.update(root_descriptor_sha256=descriptor_sha, startup_input_lock_sha256=lock["canonical_sha256"])
    row["bound"]["weights_file_sha256"] = nnuec["weights"]["sha256"]
    admission = row["native_admission"]
    admission["native_origin"]["input_lock_sha256"] = lock["canonical_sha256"]
    admission["native_origin"]["runtime_bundle_sha256"] = None if bundle is None else common._runtime_bundle_digest(bundle)
    admission["native_origin_sha256"] = digest(corpus.ORIGIN_DOMAIN, admission["native_origin"])
    admission.update(root_budget_sha256=budget, root_descriptor=copy.deepcopy(descriptor), root_descriptor_sha256=descriptor_sha)
    row["native_admission_sha256"] = digest(corpus.ROOT_DOMAIN, admission)
    if empty:
        row["retirement"].update(attempted=0, selected=0, retained_readouts=0, retained_input_bytes=0,
            ending={"kind": "failed"}, error="source_fixture_no_readouts")
    seal(row)
    rows = [] if empty else [row]
    retirement_sha = row["retirement_sha256"]
    record_shas = sorted(item["record_sha256"] for item in rows)
    terminal = {"schema": historical.RETIREMENT_OBSERVATION_DOMAIN, "native_admission": admission,
        "native_admission_sha256": row["native_admission_sha256"], "retirement": row["retirement"],
        "retirement_sha256": retirement_sha, "readout_record_sha256": record_shas}
    registration_body = {"domain": "rz-nnuec-v2-registration/1", **{key: nnuec[key] for key in ("schema", "semantics", "encoding",
        "implementation_sha256", "configuration_sha256", "context_namespace_policy_sha256", "local_namespace_policy_sha256")},
        "context_namespace_policy": "synthetic-original-C-policy", "local_namespace_policy": "synthetic-original-L-policy",
        "model_identity": {"semantics": nnuec["semantics"], "weights_sha256": parameter, "training": {"kind": "untrained"}}}
    blobs = {"r-corpus-manifest.json": raw(manifest), "input-locks/" + lock["canonical_sha256"] + ".json": raw(lock),
        "nnuec-weights/" + nnuec["weights"]["sha256"] + ".json": weights_body,
        "nnuec-provenance/" + nnuec["provenance"]["sha256"] + ".json": provenance_body,
        "r-retirements/" + retirement_sha + ".json": raw(terminal)}
    bundle_pin = None
    if bundle_body is not None:
        bundle_pin = file_pin("runtime-bundles/" + hashlib.sha256(bundle_body).hexdigest() + ".json", bundle_body)
        blobs[bundle_pin["path"]] = bundle_body
    chunks = []
    if rows:
        body = b"".join(raw(item) for item in rows)
        blobs["r-chunks/000000.jsonl"] = body
        chunks = [{**file_pin("r-chunks/000000.jsonl", body), "records": len(rows)}]
    origins = [{"source_domain": wire.RECORD_DOMAIN, "record_sha256": item["record_sha256"], "input_sha256": item["input"]["semantic_input_sha256"],
        "origin_sha256": item["origin_sha256"], "split": item["split"], "lineage": item["lineage"]} for item in rows]
    index = {"schema": wire.INDEX_DOMAIN, "manifest_sha256": digest(wire.MANIFEST_DOMAIN, manifest),
        "record_count": len(rows), "chunks": chunks, "origins": origins}
    blobs["r-corpus-index.json"] = raw(index)
    lock_pin = file_pin("input-locks/" + lock["canonical_sha256"] + ".json", raw(lock))
    entry = {"native_admission_sha256": row["native_admission_sha256"], "input_lock": lock_pin,
        "canonical_input_lock_sha256": lock["canonical_sha256"], "runtime_bundle_manifest": bundle_pin,
        "nnuec_weights": file_pin("nnuec-weights/" + nnuec["weights"]["sha256"] + ".json", weights_body),
        "nnuec_provenance": file_pin("nnuec-provenance/" + nnuec["provenance"]["sha256"] + ".json", provenance_body),
        "nnuec_parameter_sha256": parameter, "nnuec_registration": registration_body,
        "nnuec_registration_sha256": digest(historical.NNUEC_REGISTRATION_DOMAIN, registration_body)}
    collection = manifest["collection"]
    publication = {"schema": historical.PUBLICATION_DOMAIN, "manifest": file_pin("r-corpus-manifest.json", blobs["r-corpus-manifest.json"]),
        "index": file_pin("r-corpus-index.json", blobs["r-corpus-index.json"]),
        "chunks": [{key: chunk[key] for key in ("path", "sha256", "bytes")} for chunk in chunks],
        "retirements": [{"retirement_sha256": retirement_sha, "file": file_pin("r-retirements/" + retirement_sha + ".json", raw(terminal)),
            "readout_record_sha256": record_shas}], "canonical_manifest_sha256": digest(wire.MANIFEST_DOMAIN, manifest),
        "canonical_index_sha256": digest(wire.INDEX_DOMAIN, index), "source_commit": collection["source_commit"], "source_sha256": collection["source_sha256"],
        "runtime_sha256": collection["runtime_sha256"], "record_count": len(rows), "total_payload_bytes": sum(len(body) for body in blobs.values()),
        "admissions": [entry]}
    for path, body in blobs.items():
        output = root.joinpath(*path.split("/"))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(body)
    publication_body = raw(publication)
    (root / "r-publication.json").write_bytes(publication_body)
    request = {"root": str(root), "publication": {"path": str(root / "r-publication.json"),
        "sha256": hashlib.sha256(publication_body).hexdigest(), "bytes": len(publication_body)},
        "collection_source_commit": collection["source_commit"], "collection_source_sha256": collection["source_sha256"],
        "collection_runtime_library_sha256": collection["runtime_sha256"], "collection_model_semantics_sha256": collection["model_semantics_sha256"],
        "collection_model_epoch": collection["model_epoch"], "collection_context_epoch": collection["context_epoch"],
        "collection_recipe_sha256": recipe.sha256, "collection_stage_id": collection["stage_id"], "nnuec": copy.deepcopy(nnuec),
        "nnuec_parameter_sha256": parameter}
    return recipe, request, publication, rows


def reseal_publication(root, request, publication):
    body = raw(publication)
    (root / "r-publication.json").write_bytes(body)
    request["publication"].update(sha256=hashlib.sha256(body).hexdigest(), bytes=len(body))


def open_publication(root, *, empty=False):
    recipe, request, publication, rows = write_publication(root, empty=empty)
    budget = common.HistoricalReadBudget(64 << 20)
    owner = historical.HistoricalRoventRCorpus.from_registration(request, recipe=recipe, read_budget=budget)
    return owner, budget, request, publication, rows


class HistoricalRCorpusTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_complete_published_files_create_only_historical_data_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            owner, budget, request, publication, rows = open_publication(Path(directory))
            self.assertEqual(len(owner.corpus.rows()), 1)
            sample = owner.corpus.training_sample(owner.corpus.rows()[0])
            self.assertEqual(sample["nnuec_model_sha256"], request["nnuec_parameter_sha256"])
            self.assertNotEqual(sample["nnuec_model_sha256"], sample["nnuec_weights_file_sha256"])
            report = owner.observation()
            self.assertEqual(report["record_count"], 1)
            self.assertEqual(report["retirement_count"], 1)
            self.assertEqual(report["ancestor_publication_sha256"], [])
            self.assertIs(report["execution_authorized"], False)
            self.assertIs(report["nnuec_parameter_identity_recomputed"], False)
            self.assertIs(report["actual_rules_replay_verified"], False)
            self.assertIs(report["original_capture_authority_reissued"], False)
            self.assertGreater(budget.observation()["charged_bytes"], publication["total_payload_bytes"])
            self.assertIs(budget.observation()["unknown"], False)

    def test_failed_empty_retirement_is_preserved_without_fabricating_a_model_input(self):
        with tempfile.TemporaryDirectory() as directory:
            owner, _budget, _request, _publication, _rows = open_publication(Path(directory), empty=True)
            self.assertEqual(len(owner.corpus.rows()), 0)
            report = owner.observation()
            self.assertEqual(report["empty_retirement_count"], 1)
            self.assertEqual(report["record_count"], 0)
            self.assertEqual(owner.corpus.observation()["candidate_wdl_train_unique_input_count"], 0)

    def test_dependency_api_uses_pinned_canonical_indices_without_opening_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _recipe, request, publication, _rows = write_publication(root)
            # Graph metadata does not open source assets or issue a live owner.
            source = publication["admissions"][0]["nnuec_weights"]
            root.joinpath(*source["path"].split("/")).unlink()
            budget = common.HistoricalReadBudget(64 << 20)
            report = historical.read_historical_r_dependencies(request, read_budget=budget)
            self.assertEqual(report["publication_sha256"], request["publication"]["sha256"])
            self.assertEqual(report["canonical_index_sha256"], publication["canonical_index_sha256"])
            self.assertEqual(report["r_ancestor_index_sha256"], [])
            self.assertIs(report["execution_authorized"], False)
            self.assertEqual(budget.observation()["charged_bytes"], request["publication"]["bytes"] + publication["manifest"]["bytes"])

    def test_raw_asset_tamper_is_not_hidden_by_matching_parsed_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe, request, publication, _rows = write_publication(root)
            pin = publication["admissions"][0]["nnuec_weights"]
            path = root.joinpath(*pin["path"].split("/"))
            body = bytearray(path.read_bytes())
            body[-2] = ord(" ")
            path.write_bytes(body)
            budget = common.HistoricalReadBudget(64 << 20)
            with self.assertRaisesRegex(ValueError, "raw byte pin/identity mismatch"):
                historical.HistoricalRoventRCorpus.from_registration(request, recipe=recipe, read_budget=budget)
            self.assertTrue(budget.observation()["unknown"])

    def test_archive_paths_are_not_required_to_equal_original_asset_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            owner, _budget, request, publication, _rows = open_publication(Path(directory))
            self.assertNotEqual(request["nnuec"]["weights"]["path"], publication["admissions"][0]["nnuec_weights"]["path"])
            self.assertEqual(owner.observation()["nnuec_weights_file_sha256"], request["nnuec"]["weights"]["sha256"])

    def test_publication_rejects_duplicate_coverage_and_conflicting_unique_path_pins(self):
        for mutation in ("duplicate_row", "conflicting_path"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                recipe, request, publication, _rows = write_publication(root)
                if mutation == "duplicate_row":
                    duplicate = copy.deepcopy(publication["retirements"][0])
                    duplicate["retirement_sha256"] = h("another retirement")
                    duplicate["file"]["path"] = "r-retirements/" + duplicate["retirement_sha256"] + ".json"
                    publication["retirements"].append(duplicate)
                else:
                    publication["admissions"][0]["nnuec_weights"]["path"] = publication["admissions"][0]["input_lock"]["path"]
                reseal_publication(root, request, publication)
                with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "more than one retirement|source path differs"):
                    historical.HistoricalRoventRCorpus.from_registration(request, recipe=recipe,
                        read_budget=common.HistoricalReadBudget(64 << 20))

    def test_raw_retirement_is_required_even_when_row_retirement_seal_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe, request, publication, _rows = write_publication(root)
            root.joinpath(*publication["retirements"][0]["file"]["path"].split("/")).unlink()
            with self.assertRaises(FileNotFoundError):
                historical.HistoricalRoventRCorpus.from_registration(request, recipe=recipe,
                    read_budget=common.HistoricalReadBudget(64 << 20))

    def test_runtime_library_and_collection_epochs_are_not_inferred_from_trainer(self):
        for field, replacement in (("collection_runtime_library_sha256", h("trainer Python")), ("collection_model_epoch", 3)):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                recipe, request, _publication, _rows = write_publication(root)
                request[field] = replacement
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "registered collection identity"):
                    historical.HistoricalRoventRCorpus.from_registration(request, recipe=recipe,
                        read_budget=common.HistoricalReadBudget(64 << 20))

    def test_parameter_pin_and_model_registration_cannot_use_raw_file_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe, request, _publication, _rows = write_publication(root)
            request["nnuec_parameter_sha256"] = request["nnuec"]["weights"]["sha256"]
            with self.assertRaisesRegex(ValueError, "semantic parameter identity differs"):
                historical.HistoricalRoventRCorpus.from_registration(request, recipe=recipe,
                    read_budget=common.HistoricalReadBudget(64 << 20))

    def test_pure_source_operand_parser_rejects_bool_huge_int_and_nonfinite_without_model(self):
        for number in (True, 10**1000, float("inf"), float("nan")):
            value = source_operands()
            value["base"]["parameters"]["board_bias"][0] = number
            with self.subTest(number=type(number).__name__), self.assertRaisesRegex(ValueError, "finite bounded FP32"):
                historical._weights_metadata(value)
        value = source_operands()
        value["base"]["output_scale"] = 10**1000
        with self.assertRaisesRegex(ValueError, "finite positive original output scale"):
            historical._weights_metadata(value)

    def test_optional_expected_config_cannot_replace_original_collection_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe, request, _publication, _rows = write_publication(root)
            different = base.NnuecConfig.from_mapping({**config().to_mapping(), "hidden_width": 2})
            with self.assertRaisesRegex(ValueError, "source dimensions differ"):
                historical.HistoricalRoventRCorpus.from_registration(request, recipe=recipe, nnuec_config=different,
                    read_budget=common.HistoricalReadBudget(64 << 20))

    def test_source_provenance_declaration_and_publication_metadata_are_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            owner, _budget, _request, _publication, _rows = open_publication(Path(directory))
            owner._registration["nnuec_parameter_sha256"] = h("mutated identity")
            with self.assertRaisesRegex(ValueError, "immutable publication/registration observation changed"):
                owner.observation()

    def test_cuda_bundle_raw_file_pin_and_spec_digest_both_join_original_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe, request, publication, rows = write_publication(root, with_bundle=True)
            owner = historical.HistoricalRoventRCorpus.from_registration(request, recipe=recipe,
                read_budget=common.HistoricalReadBudget(64 << 20))
            self.assertEqual(len(owner.corpus.rows()), 1)
            self.assertNotEqual(publication["admissions"][0]["runtime_bundle_manifest"]["sha256"],
                rows[0]["native_admission"]["native_origin"]["runtime_bundle_sha256"])

    def test_cuda_bundle_without_original_raw_artifact_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe, request, _publication, _rows = write_publication(root, with_bundle=True, include_bundle_provenance=False)
            with self.assertRaisesRegex(ValueError, "bundle raw file pin absent from original lock provenance"):
                historical.HistoricalRoventRCorpus.from_registration(request, recipe=recipe,
                    read_budget=common.HistoricalReadBudget(64 << 20))

    def test_missing_shared_budget_is_rejected_before_opening_a_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe, request, _publication, _rows = write_publication(root)
            (root / "r-publication.json").unlink()
            with self.assertRaisesRegex(ValueError, "shared bounded read accounting required"):
                historical.HistoricalRoventRCorpus.from_registration(request, recipe=recipe, read_budget=None)

    def test_strict_publication_json_rejects_duplicate_and_nonfinite_fields_with_correct_file_pin(self):
        for body in (b'{"schema":"x","schema":"y"}', b'{"schema":NaN}', b'{"schema":-Infinity}'):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                recipe, request, _publication, _rows = write_publication(root)
                (root / "r-publication.json").write_bytes(body)
                request["publication"].update(sha256=hashlib.sha256(body).hexdigest(), bytes=len(body))
                with self.subTest(body=body), self.assertRaisesRegex(ValueError, "duplicate JSON field|nonfinite corpus JSON"):
                    historical.HistoricalRoventRCorpus.from_registration(request, recipe=recipe,
                        read_budget=common.HistoricalReadBudget(64 << 20))


if __name__ == "__main__":
    unittest.main()
