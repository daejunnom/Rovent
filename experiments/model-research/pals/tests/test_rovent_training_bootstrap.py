"""Small stdlib bootstrap fixtures; no process migration or training proof."""
import copy
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("neural runtime import forbidden in bootstrap fixtures")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_training_bootstrap as bootstrap
finally:
    sys.meta_path.remove(_guard)


def pin(path, content=b"fixture-only"):
    return {"path": path, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def fixture():
    """A projection only: no original Rust custody, files or socket exists."""
    source_root = "/tmp/rovent-fixture-output/source"
    package = bootstrap.MODULE_RELATIVE
    sources = [{"relative_path": package + name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        for name, content in (("__init__.py", b"# package fixture\n"),
                              ("rovent_training_child.py", b"# child fixture\n"),
                              ("rovent_training_bootstrap.py", b"# bootstrap fixture\n"))]
    inventory = {entry["relative_path"]: {name: entry[name] for name in ("bytes", "sha256")} for entry in sources}
    policy = {"cpu_quota_millicpus": 2000, "memory_high_bytes": 6 << 30,
        "memory_max_bytes": 12 << 30, "pids_max": 64, "affinity": [0, 2], "device": "cpu"}
    readback = {"cpu_quota_us": 2000000, "cpu_period_us": 1000000, "memory_high_bytes": 6 << 30,
        "memory_max_bytes": 12 << 30, "memory_swap_max_bytes": 0, "pids_max": 64}
    corpus = {"manifest": pin("/tmp/rovent-fixture-corpus/manifest.json", b"manifest"),
        "index": pin("/tmp/rovent-fixture-corpus/index.json", b"index"), "root": "/tmp/rovent-fixture-corpus"}
    registration = {"schema": bootstrap.REGISTRATION_DOMAIN, "execution_id": "fixture-execution",
        "action": "train", "export_request": None,
        "recipe": pin("/tmp/rovent-fixture-input/recipe.json"), "membership": pin("/tmp/rovent-fixture-input/membership.json"),
        "window_spec": pin("/tmp/rovent-fixture-input/window.json"),
        "trainer_source_sha256": bootstrap._digest(bootstrap.SOURCE_DOMAIN, inventory),
        "python_runtime_sha256": "1" * 64, "model_semantics_sha256": "2" * 64,
        "current_corpus": corpus, "ancestor_corpora": [], "checkpoint": None,
        "pc_initial": pin("/tmp/rovent-fixture-input/untrained.pt"),
        "nnuec_initial": pin("/tmp/rovent-fixture-input/nnuec.json"),
        "rules_profile": pin("/tmp/rovent-fixture-input/rules.json"), "checkpoint_mode": "initialize",
        "expected_initial_completed": dict.fromkeys(("proposer", "critic", "validator"), 0),
        "expected_ancestor_completed": dict.fromkeys(("proposer", "critic", "validator"), 0),
        "resources_policy": policy, "require_total_quota": False,
        "output_root": "/tmp/rovent-fixture-output", "control_root": "/tmp/rovent-fixture-control",
        "checkpoint_directory": "/tmp/rovent-fixture-output/checkpoints", "receipt_path": "/tmp/rovent-fixture-output/result.json",
        "max_rpc_messages": 128, "max_rpc_bytes": 1 << 20, "max_read_bytes": 1 << 20, "max_output_bytes": 32 << 20}
    return {"schema": bootstrap.STARTUP_DOMAIN, "execution_id": "fixture-execution", "window_id": "fixture-window",
        "registration_sha256": "3" * 64, "registration": registration,
        "control": {"socket_path": "/tmp/rovent-fixture-control/control.sock", "parent_pid": 17, "parent_start_ticks": 23,
            "control_device": 1, "control_inode": 2},
        "original_started_ns": 100, "execution_until_ns": 10000, "whole_until_ns": 11000,
        "source_root": source_root, "package_root": source_root + "/" + bootstrap.PACKAGE_RELATIVE,
        "source_files": sources, "dependency_directories": [{"path": "/tmp/rovent-fixture-dependencies", "device": 1, "inode": 3}],
        "runtime_module_pins": [pin("/tmp/rovent-fixture-dependencies/runtime.py")],
        "python_runtime": {"version": "3.12.3", "executable": pin("/tmp/rovent-fixture-python")},
        "resources": {"domain": bootstrap.RESOURCE_DOMAIN, "migration": "child_self_only_before_torch",
            "path": "/sys/fs/cgroup/rovent-fixture", "device": 4, "inode": 5, "membership": "/rovent-fixture",
            "policy": copy.deepcopy(policy), "prepared_readback": copy.deepcopy(readback), "readback": copy.deepcopy(readback),
            "available_controllers_snapshot": ["cpu", "memory", "pids"],
            "subtree_controllers_snapshot": ["cpu", "memory", "pids"],
            "scope": "configured_leaf_readback_not_child_membership_or_lifetime_proof"}}


def validate(value, *, raw=None, **changes):
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode() if raw is None else raw
    arguments = {"expected_sha256": hashlib.sha256(raw).hexdigest(), "expected_bytes": len(raw),
        "execution_until_ns": 10000, "whole_until_ns": 11000, "now_ns": 200}
    arguments.update(changes)
    return bootstrap.validate_startup(raw, **arguments)


def recovered_fixture():
    """Explicit new DTO, with collection/exec/Torch pins deliberately distinct."""
    value = fixture()
    old = value["registration"]
    for item in value["source_files"]:
        if item["relative_path"] == bootstrap.MODULE_RELATIVE + "rovent_training_child.py":
            item["relative_path"] = bootstrap.MODULE_RELATIVE + "rovent_recovered_training_child.py"
    source_inventory = {item["relative_path"]: {key: item[key] for key in ("bytes", "sha256")} for item in value["source_files"]}
    endpoints = [{"name": "torch.__file__", "artifact": pin("/tmp/rovent-fixture-dependencies/torch/init.py", b"Torch Python endpoint")},
        {"name": "torch._C.__file__", "artifact": pin("/tmp/rovent-fixture-dependencies/torch/native.so", b"Torch native endpoint")}]
    dataset = {"root": "/tmp/rovent-fixture-history", "publication": pin("/tmp/rovent-fixture-history/publication.json", b"historical dataset publication"),
        "collection_source_commit": "d" * 40, "collection_source_sha256": hashlib.sha256(b"collection native source").hexdigest(),
        "collection_runtime_library_sha256": hashlib.sha256(b"collection ORT library").hexdigest(),
        "model_semantics_sha256": hashlib.sha256(b"shared architecture semantics").hexdigest(), "model_epoch": 2, "context_epoch": 3,
        "recipe_sha256": hashlib.sha256(b"historical canonical recipe").hexdigest(), "stage_id": "common",
        "role_bindings": {"proposer": {"model_semantics_sha256": hashlib.sha256(b"shared architecture semantics").hexdigest(),
            "model_identity_sha256": hashlib.sha256(b"historical proposer model").hexdigest(),
            "weights_sha256": hashlib.sha256(b"historical proposer weights").hexdigest(),
            "encoding_sha256": hashlib.sha256(b"historical encoder").hexdigest()}}}
    registration = {"schema": bootstrap.RECOVERED_REGISTRATION_DOMAIN, "execution_id": old["execution_id"],
        "action": "train", "export_request": None, "recipe": old["recipe"], "membership": old["membership"], "window_spec": old["window_spec"],
        "trainer_source_sha256": bootstrap._digest(bootstrap.RECOVERED_SOURCE_DOMAIN, source_inventory), "trainer_python_version": "3.12.3",
        "trainer_python_executable": pin("/tmp/rovent-fixture-python", b"actual new trainer Python image"),
        "dependency_directories": [entry["path"] for entry in value["dependency_directories"]],
        "trainer_dependency_set_sha256": bootstrap.recovered_dependency_set_digest(endpoints), "runtime_endpoints": endpoints,
        "current_dataset": dataset, "ancestor_datasets": [], "initialization_source": {"kind": "initialize",
            "pc_checkpoint": pin("/tmp/rovent-fixture-input/initial-pc.pt", b"initial P C tensors"),
            "nnuec_checkpoint": pin("/tmp/rovent-fixture-input/initial-nnuec.json", b"initial NNUEC tensors")},
        "rules_profile": old["rules_profile"], "expected_initial_completed": old["expected_initial_completed"],
        "expected_ancestor_completed": old["expected_ancestor_completed"], "resources_policy": old["resources_policy"],
        "require_total_quota": old["require_total_quota"], "output_root": old["output_root"], "control_root": old["control_root"],
        "checkpoint_directory": old["checkpoint_directory"], "receipt_path": old["receipt_path"],
        "max_rpc_messages": old["max_rpc_messages"], "max_rpc_bytes": old["max_rpc_bytes"],
        "max_read_bytes": old["max_read_bytes"], "max_child_read_bytes": old["max_read_bytes"] // 2,
        "max_output_bytes": old["max_output_bytes"], "max_artifact_files": 4096}
    value.update(schema=bootstrap.RECOVERED_STARTUP_DOMAIN, registration=registration,
        runtime_module_pins=[copy.deepcopy(endpoint["artifact"]) for endpoint in endpoints],
        python_runtime={"version": registration["trainer_python_version"], "executable": registration["trainer_python_executable"]},
        collection_admission={"schema": "rz-rovent-recovered-dataset-set/1", "current_publication_sha256": dataset["publication"]["sha256"],
            "ancestor_publication_sha256": [], "past_native_authority_reissued": False,
            "past_physical_completion_reissued": False, "historical_evidence_only": True, "execution_authorized": False},
        checkpoint_admission=None, child_read_credit={"reserved_bytes": registration["max_child_read_bytes"],
            "scope": "charged_in_parent_shared_budget_no_refund"},
        artifact_watch_profile={"schema": "rz-rovent-training-artifact-watch/1",
            "maximum_total_bytes": registration["max_output_bytes"] - min(registration["max_output_bytes"] // 8, 64 << 20),
            "maximum_file_bytes": 2 << 30, "maximum_files": registration["max_artifact_files"],
            "scope": "new_trainer_owned_output_tree_only"})
    return value


def recovered_resume(value):
    value = copy.deepcopy(value)
    publication = pin("/tmp/rovent-fixture-prior-checkpoint/publication.json", b"historical tensor publication")
    value["registration"]["initialization_source"] = {"kind": "resume", "checkpoint": {
        "root": "/tmp/rovent-fixture-prior-checkpoint", "publication": publication,
        "reclamation": pin("/tmp/rovent-fixture-prior-checkpoint/closure.json", b"historical closure bytes")}}
    value["checkpoint_admission"] = {"schema": "rz-rovent-historical-checkpoint-metadata-admission/1",
        "publication_sha256": publication["sha256"], "tensor_sha256": hashlib.sha256(b"prior tensor").hexdigest(),
        "update_chain_sha256": hashlib.sha256(b"prior original update receipts").hexdigest(),
        "snapshot_completed_updates": dict.fromkeys(("proposer", "critic", "validator"), 0),
        "post_snapshot_optimizer_updates": dict.fromkeys(("proposer", "critic", "validator"), 0),
        "recorded_tail_cost": {"status": "unknown", "reason": "fixture historical scope only", "evidence": None},
        "recorded_closure_classification": "unresolved", "historical_window_schema": "rz-rovent-train-window/2",
        "historical_window_identity_scope": "explicit_v2_recorded_pins",
        "historical_metadata_integrity_only": True, "tensor_state_validated": False,
        "past_native_authority_reissued": False, "past_physical_completion_reissued": False, "execution_authorized": False}
    return value


def recovered_r_auxiliary_fixture():
    """Separate R raw file/parameter identities; no P/C record conversion."""
    value = recovered_fixture()
    registration = value["registration"]
    pc = registration["current_dataset"]
    seal = lambda label: hashlib.sha256(label.encode()).hexdigest()
    original = lambda label: {"path": "original-source/" + label + ".json", "sha256": seal(label + " original bytes"),
        "bytes": 17, "source": "fixture-original-declaration", "license": "MIT"}
    request = {"root": "/tmp/rovent-fixture-r-history",
        "publication": pin("/tmp/rovent-fixture-r-history/r-publication.json", b"separate original R publication"),
        "collection_source_commit": pc["collection_source_commit"], "collection_source_sha256": pc["collection_source_sha256"],
        "collection_runtime_library_sha256": pc["collection_runtime_library_sha256"],
        "collection_model_semantics_sha256": pc["model_semantics_sha256"], "collection_model_epoch": pc["model_epoch"],
        "collection_context_epoch": pc["context_epoch"], "collection_recipe_sha256": pc["recipe_sha256"],
        "collection_stage_id": pc["stage_id"], "nnuec_parameter_sha256": seal("parsed R NNUEC parameters"),
        "nnuec": {"schema": "rz-nnuec-fp32-v2",
            "semantics": "board800-exactb192-causal-line-kind4-directedpair-originfirst-exactc256-fp32-v2",
            "encoding": "rz-nnuec-fp32-input/2", "implementation_sha256": seal("original implementation"),
            "configuration_sha256": seal("original config"), "weights": original("weights"), "provenance": original("provenance"),
            "diagnostic": False, "context_namespace_policy_sha256": seal("original context namespace"),
            "local_namespace_policy_sha256": seal("original local namespace"), "search_conditions_sha256": seal("original search")}}
    registration["r_auxiliary_dataset"] = request
    value["r_auxiliary_admission"] = {"schema": "rz-rovent-recovered-r-dataset-set/1",
        "current_publication_sha256": request["publication"]["sha256"], "ancestor_publication_sha256": [],
        "manifest_file_sha256": seal("raw R manifest bytes"), "index_file_sha256": seal("raw R index bytes"),
        "nnuec_parameter_sha256": request["nnuec_parameter_sha256"], "record_count": 3,
        "past_native_authority_reissued": False, "past_physical_completion_reissued": False,
        "historical_evidence_only": True, "execution_authorized": False}
    return value


def validate_recovered(value, **changes):
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    arguments = {"expected_sha256": hashlib.sha256(raw).hexdigest(), "expected_bytes": len(raw),
        "execution_until_ns": 10000, "whole_until_ns": 11000, "now_ns": 200}
    arguments.update(changes)
    return bootstrap.validate_recovered_startup(raw, **arguments)


class RoventRecoveredBootstrapProjectionTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_registration_fixture_matches_the_independently_listed_rust_dto_fields(self):
        # Deliberately independent of the Python validator's expected tuple.
        # Keep this list aligned to RoventRecoveredTrainingRegistration so a
        # newly required wire field cannot disappear from both local fixtures
        # and parsing tests unnoticed.
        expected = {"schema", "execution_id", "action", "export_request", "recipe", "membership", "window_spec",
            "trainer_source_sha256", "trainer_python_executable", "trainer_python_version", "dependency_directories",
            "trainer_dependency_set_sha256", "runtime_endpoints", "current_dataset", "ancestor_datasets", "initialization_source",
            "rules_profile", "expected_initial_completed", "expected_ancestor_completed", "resources_policy", "require_total_quota",
            "output_root", "control_root", "checkpoint_directory", "receipt_path", "max_rpc_messages", "max_rpc_bytes",
            "max_read_bytes", "max_child_read_bytes", "max_output_bytes", "max_artifact_files"}
        value = recovered_fixture()
        self.assertEqual(set(value["registration"]), expected)
        self.assertEqual(validate_recovered(value), value)

    def test_registered_dependency_paths_are_bounded_unique_and_join_the_pinned_startup_roster(self):
        for paths in ([], ["relative"], ["/tmp/dependency-one"] * 2,
                ["/tmp/dependency-" + str(index) for index in range(9)]):
            value = recovered_fixture()
            value["registration"]["dependency_directories"] = paths
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                validate_recovered(value)
        missing = recovered_fixture()
        missing["registration"].pop("dependency_directories")
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            validate_recovered(missing)
        value = recovered_fixture()
        value["dependency_directories"][0]["path"] = "/tmp/foreign-fixture-dependencies"
        with self.assertRaisesRegex(ValueError, "identity roster"):
            validate_recovered(value)
        value = recovered_fixture()
        value["registration"]["dependency_directories"].append("/tmp/second-fixture-dependencies")
        value["dependency_directories"].append({"path": "/tmp/second-fixture-dependencies", "device": 1, "inode": 4})
        self.assertEqual(validate_recovered(value), value)
        value["dependency_directories"].reverse()
        with self.assertRaisesRegex(ValueError, "identity roster"):
            validate_recovered(value)

    def test_r_auxiliary_absent_and_null_presence_are_preserved_without_startup_defaults(self):
        absent = recovered_fixture()
        checked_absent = validate_recovered(absent)
        self.assertNotIn("r_auxiliary_dataset", checked_absent["registration"])
        self.assertNotIn("ancestor_r_auxiliary_datasets", checked_absent["registration"])
        self.assertNotIn("r_auxiliary_admission", checked_absent)
        explicit = recovered_fixture()
        explicit["registration"].update(r_auxiliary_dataset=None, ancestor_r_auxiliary_datasets=None)
        checked = validate_recovered(explicit)
        self.assertIsNone(checked["registration"]["r_auxiliary_dataset"])
        self.assertIsNone(checked["registration"]["ancestor_r_auxiliary_datasets"])
        self.assertNotEqual(json.dumps(checked_absent, sort_keys=True), json.dumps(checked, sort_keys=True))
        explicit["r_auxiliary_admission"] = None
        with self.assertRaisesRegex(ValueError, "disabled R auxiliary"):
            validate_recovered(explicit)

    def test_r_auxiliary_enabled_source_keeps_raw_weights_separate_from_parameters_and_pc(self):
        value = recovered_r_auxiliary_fixture()
        checked = validate_recovered(value)
        r = checked["registration"]["r_auxiliary_dataset"]
        self.assertNotEqual(r["nnuec"]["weights"]["sha256"], r["nnuec_parameter_sha256"])
        self.assertNotEqual(r["publication"]["sha256"], checked["registration"]["current_dataset"]["publication"]["sha256"])
        self.assertFalse(checked["r_auxiliary_admission"]["execution_authorized"])
        self.assertEqual(checked["r_auxiliary_admission"]["nnuec_parameter_sha256"], r["nnuec_parameter_sha256"])
        changed = copy.deepcopy(value)
        changed["r_auxiliary_admission"]["nnuec_parameter_sha256"] = r["nnuec"]["weights"]["sha256"]
        with self.assertRaisesRegex(ValueError, "cannot alias P/C data"):
            validate_recovered(changed)

    def test_r_auxiliary_admission_cannot_claim_past_process_or_current_training_authority(self):
        for name in ("past_native_authority_reissued", "past_physical_completion_reissued", "execution_authorized"):
            value = recovered_r_auxiliary_fixture()
            value["r_auxiliary_admission"][name] = True
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "historical authority"):
                validate_recovered(value)
        value = recovered_r_auxiliary_fixture()
        value.pop("r_auxiliary_admission")
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            validate_recovered(value)

    def test_r_auxiliary_ancestor_presence_and_sorted_distinct_publication_roster_are_strict(self):
        disabled = recovered_fixture()
        disabled["registration"]["ancestor_r_auxiliary_datasets"] = []
        with self.assertRaisesRegex(ValueError, "enabled current R dataset"):
            validate_recovered(disabled)
        value = recovered_r_auxiliary_fixture()
        value["registration"]["ancestor_r_auxiliary_datasets"] = []
        self.assertEqual(validate_recovered(value), value)
        source = value["registration"]["r_auxiliary_dataset"]
        value["registration"]["ancestor_r_auxiliary_datasets"] = [copy.deepcopy(source)]
        with self.assertRaisesRegex(ValueError, "distinct from current"):
            validate_recovered(value)

    def test_r_auxiliary_window_join_preserves_presence_and_original_raw_file_pins(self):
        value = recovered_r_auxiliary_fixture()
        registration, admission = value["registration"], value["r_auxiliary_admission"]
        dataset, rsource = registration["current_dataset"], registration["r_auxiliary_dataset"]
        window = {"schema": bootstrap.RECOVERED_WINDOW_DOMAIN, "window_id": value["window_id"],
            "trainer_source_sha256": registration["trainer_source_sha256"],
            "trainer_python_executable_sha256": registration["trainer_python_executable"]["sha256"],
            "trainer_dependency_set_sha256": registration["trainer_dependency_set_sha256"],
            "trainer_dependency_scope": bootstrap.DEPENDENCY_SCOPE,
            "collection_source_sha256": dataset["collection_source_sha256"],
            "collection_runtime_library_sha256": dataset["collection_runtime_library_sha256"],
            "collection_model_semantics_sha256": dataset["model_semantics_sha256"],
            "collection_model_epoch": dataset["model_epoch"], "collection_context_epoch": dataset["context_epoch"],
            "recipe_sha256": dataset["recipe_sha256"], "stage_id": dataset["stage_id"], "device": "cpu",
            "max_output_bytes": 1024, "training_model_semantics_sha256": dataset["model_semantics_sha256"],
            "nnuec_semantics": rsource["nnuec"]["semantics"],
            "nnuec_r_auxiliary": {"schema": "rz-rovent-r-auxiliary-recipe/1",
                "manifest_file_sha256": admission["manifest_file_sha256"], "index_file_sha256": admission["index_file_sha256"],
                "sampling_seed": 31, "rows_per_pc_update": 1, "weight": 0.5, "normalization": "valid_exact_trajectory_rows",
                "optimizer_owner": "exclusive_nnuec", "supervision_source": "r_auxiliary_only_preserve_pc_evidence"}}
        self.assertFalse(bootstrap.validate_recovered_window_binding(value, window)["execution_authorized"])
        no_recipe = copy.deepcopy(window)
        no_recipe.pop("nnuec_r_auxiliary")
        with self.assertRaisesRegex(ValueError, "preserve exact presence"):
            bootstrap.validate_recovered_window_binding(value, no_recipe)
        changed = copy.deepcopy(window)
        changed["nnuec_r_auxiliary"]["manifest_file_sha256"] = rsource["nnuec_parameter_sha256"]
        with self.assertRaisesRegex(ValueError, "raw file pins"):
            bootstrap.validate_recovered_window_binding(value, changed)

    def test_new_projection_has_distinct_data_python_dependency_and_source_namespaces(self):
        value = recovered_fixture()
        with mock.patch.object(os, "open", side_effect=AssertionError("projection performs no file IO")):
            checked = validate_recovered(value)
        self.assertEqual(checked, value)
        registration = checked["registration"]
        self.assertEqual(len({registration["current_dataset"]["collection_source_sha256"],
            registration["current_dataset"]["collection_runtime_library_sha256"], registration["trainer_source_sha256"],
            registration["trainer_python_executable"]["sha256"], registration["trainer_dependency_set_sha256"]}), 5)
        self.assertFalse(checked["collection_admission"]["execution_authorized"])
        self.assertFalse(checked["collection_admission"]["past_native_authority_reissued"])
        self.assertFalse(checked["collection_admission"]["past_physical_completion_reissued"])

    def test_new_and_original_codecs_reject_each_other_instead_of_converting(self):
        with self.assertRaises(ValueError):
            validate(recovered_fixture())
        with self.assertRaises(ValueError):
            validate_recovered(fixture())
        value = recovered_fixture()
        value["registration"]["python_runtime_sha256"] = value["registration"]["trainer_python_executable"]["sha256"]
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            validate_recovered(value)

    def test_endpoint_digest_sorts_semantic_names_but_startup_preserves_original_artifact_order(self):
        value = recovered_fixture()
        endpoints = value["registration"]["runtime_endpoints"]
        self.assertEqual(bootstrap.recovered_dependency_set_digest(endpoints),
            bootstrap.recovered_dependency_set_digest(list(reversed(endpoints))))
        value["runtime_module_pins"].reverse()
        with self.assertRaisesRegex(ValueError, "endpoint order"):
            validate_recovered(value)

    def test_python_image_cannot_be_relabelled_as_collection_ort_or_another_dependency(self):
        value = recovered_fixture()
        # Replace only the startup projection, keeping the registration's
        # independent exact Python executable pin unchanged.
        value["python_runtime"]["executable"] = {**value["python_runtime"]["executable"],
            "sha256": value["registration"]["current_dataset"]["collection_runtime_library_sha256"]}
        with self.assertRaisesRegex(ValueError, "trainer Python image"):
            validate_recovered(value)

    def test_source_inventory_requires_the_new_domain_without_retagging_old_source_hashes(self):
        value = recovered_fixture()
        inventory = {item["relative_path"]: {key: item[key] for key in ("bytes", "sha256")} for item in value["source_files"]}
        value["registration"]["trainer_source_sha256"] = bootstrap._digest(bootstrap.SOURCE_DOMAIN, inventory)
        with self.assertRaisesRegex(ValueError, "separately registered digest domain"):
            validate_recovered(value)

    def test_tagged_initialization_rejects_old_mode_aliases_and_borrowed_checkpoint_admissions(self):
        for changes in ({"kind": "initialize", "checkpoint_mode": "initialize"}, {"kind": "unknown"}):
            value = recovered_fixture()
            value["registration"]["initialization_source"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_recovered(value)
        value = recovered_fixture()
        value["registration"]["expected_initial_completed"]["proposer"] = 1
        with self.assertRaisesRegex(ValueError, "existing stage updates"):
            validate_recovered(value)
        value = recovered_resume(recovered_fixture())
        self.assertEqual(validate_recovered(value), value)
        value["checkpoint_admission"]["tensor_state_validated"] = True
        with self.assertRaisesRegex(ValueError, "cannot reissue"):
            validate_recovered(value)

    def test_historical_admissions_cannot_promote_past_native_or_physical_permission(self):
        for name in ("past_native_authority_reissued", "past_physical_completion_reissued", "execution_authorized"):
            value = recovered_resume(recovered_fixture())
            value["checkpoint_admission"][name] = True
            with self.subTest(checkpoint=name), self.assertRaisesRegex(ValueError, "cannot reissue"):
                validate_recovered(value)
            value = recovered_fixture()
            value["collection_admission"][name] = True
            with self.subTest(collection=name), self.assertRaisesRegex(ValueError, "cannot reissue"):
                validate_recovered(value)

    def test_child_read_credit_is_an_exact_nonrefundable_slice_of_the_parent_budget(self):
        for change in ({"reserved_bytes": 1}, {"scope": "metadata_claims_free_reading"}):
            value = recovered_fixture()
            value["child_read_credit"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_recovered(value)
        value = recovered_fixture()
        value["registration"]["max_child_read_bytes"] = value["registration"]["max_read_bytes"] + 1
        with self.assertRaises(ValueError):
            validate_recovered(value)

    def test_owned_output_file_profile_matches_the_new_registered_parent_watch(self):
        for count in (0, True, 32769):
            value = recovered_fixture()
            value["registration"]["max_artifact_files"] = count
            with self.subTest(count=count), self.assertRaises(ValueError):
                validate_recovered(value)
        for key, bad in (("maximum_total_bytes", 1), ("maximum_files", True),
                         ("maximum_file_bytes", 2**31 - 1), ("scope", "unbounded_global_output")):
            value = recovered_fixture()
            value["artifact_watch_profile"][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_recovered(value)

    def test_new_window_join_uses_only_v2_fields_and_stays_a_non_authoritative_projection(self):
        value = recovered_fixture()
        registration = value["registration"]
        dataset = registration["current_dataset"]
        window = {"schema": bootstrap.RECOVERED_WINDOW_DOMAIN, "window_id": value["window_id"],
            "trainer_source_sha256": registration["trainer_source_sha256"],
            "trainer_python_executable_sha256": registration["trainer_python_executable"]["sha256"],
            "trainer_dependency_set_sha256": registration["trainer_dependency_set_sha256"],
            "trainer_dependency_scope": bootstrap.DEPENDENCY_SCOPE,
            "collection_source_sha256": dataset["collection_source_sha256"],
            "collection_runtime_library_sha256": dataset["collection_runtime_library_sha256"],
            "collection_model_semantics_sha256": dataset["model_semantics_sha256"],
            "collection_model_epoch": dataset["model_epoch"], "collection_context_epoch": dataset["context_epoch"],
            "recipe_sha256": dataset["recipe_sha256"], "stage_id": dataset["stage_id"], "device": "cpu",
            "max_output_bytes": 1024, "training_model_semantics_sha256": dataset["model_semantics_sha256"]}
        checked = bootstrap.validate_recovered_window_binding(value, window)
        self.assertFalse(checked["execution_authorized"])
        self.assertEqual(checked["transitive_loaded_native_mappings"], "unknown")
        window["schema"] = "rz-rovent-train-window/1"
        with self.assertRaisesRegex(ValueError, "only explicit train-window /2"):
            bootstrap.validate_recovered_window_binding(value, window)


class RoventTrainingBootstrapProjectionTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_exact_startup_is_only_a_projection_and_has_no_os_side_effect(self):
        value = fixture()
        with mock.patch.object(bootstrap, "_open_directory", side_effect=AssertionError("no file access")), \
                mock.patch.object(os, "open", side_effect=AssertionError("no file access")):
            result = validate(value)
        self.assertEqual(result, value)
        self.assertIsNot(result, value)
        self.assertEqual(result["registration_sha256"], "3" * 64)
        self.assertNotIn("execution_authorized", result)

    def test_rejects_byte_hash_deadline_and_unknown_field_changes(self):
        value = fixture()
        for changed in ({"expected_sha256": "0" * 64}, {"expected_bytes": 1},
                {"execution_until_ns": 9999}, {"now_ns": 10000}, {"whole_until_ns": 11001}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate(value, **changed)
        extra = copy.deepcopy(value)
        extra["serialized_authority"] = True
        with self.assertRaises(ValueError):
            validate(extra)

    def test_rejects_duplicate_nonfinite_and_invalid_utf8_before_projection(self):
        for raw in (b'{"schema":"one","schema":"two"}', b'{"value":NaN}', b'{"value":1e9999}', b'\xff'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate({}, raw=raw)

    def test_source_domain_inventory_and_current_entry_pins_are_separate_from_native_source(self):
        value = fixture()
        value["registration"]["trainer_source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source inventory"):
            validate(value)
        value = fixture()
        value["source_files"].pop()
        inventory = {item["relative_path"]: {key: item[key] for key in ("bytes", "sha256")} for item in value["source_files"]}
        value["registration"]["trainer_source_sha256"] = bootstrap._digest(bootstrap.SOURCE_DOMAIN, inventory)
        with self.assertRaisesRegex(ValueError, "entry/bootstrap/package"):
            validate(value)

    def test_resources_dependencies_and_checkpoint_modes_fail_closed(self):
        mutations = (
            lambda value: value["resources"]["readback"].update(memory_swap_max_bytes=1),
            lambda value: value["registration"]["resources_policy"].update(affinity=[0, 0]),
            lambda value: value.update(dependency_directories=["/tmp/unpinned-site-packages"]),
            lambda value: value["registration"].update(checkpoint_mode="resume"),
            lambda value: value["registration"].update(max_read_bytes=1),
            lambda value: value["registration"].update(require_total_quota=1),
            lambda value: value["registration"].update(action="export"),
            lambda value: value["registration"].update(action="unknown"),
            lambda value: value["registration"].update(export_request=pin("/tmp/rovent-fixture-input/export.json")),
            lambda value: value["source_files"][0].update(relative_path=bootstrap.MODULE_RELATIVE + "../outside.py"),
        )
        for mutate in mutations:
            value = fixture()
            mutate(value)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                validate(value)

    def test_export_projection_requires_resume_and_a_pin_without_reading_or_issuing_authority(self):
        value = fixture()
        value["registration"].update(action="export", checkpoint_mode="resume",
            checkpoint=pin("/tmp/rovent-fixture-input/completed.pt"), pc_initial=None, nnuec_initial=None,
            export_request=pin("/tmp/rovent-fixture-input/export.json"))
        with mock.patch.object(os, "open", side_effect=AssertionError("export projection must not read a file")):
            result = validate(value)
        self.assertEqual(result["registration"]["action"], "export")
        self.assertNotIn("export_authorized", result)
        self.assertEqual(result["registration"]["export_request"], value["registration"]["export_request"])

    def test_nonisolated_child_rejects_before_file_resource_or_neural_operations(self):
        flags = types.SimpleNamespace(isolated=0, no_site=1, dont_write_bytecode=1)
        with mock.patch.object(bootstrap, "_require_linux_nofollow"), mock.patch.object(sys, "flags", flags), \
                mock.patch.object(bootstrap, "_open_file", side_effect=AssertionError("no file access")), \
                mock.patch.object(bootstrap, "_join_resources", side_effect=AssertionError("no migration")):
            with self.assertRaisesRegex(bootstrap.OriginalBootstrapUnavailable, "-I -S -B"):
                bootstrap.bootstrap_original_child({}, until_ns=10000, clock=lambda: 200)

    def test_shared_external_read_credit_is_not_permission_or_a_refund(self):
        budget = bootstrap.PinnedReadBudget(10, until_ns=1000, clock=lambda: 200)
        self.assertEqual(budget.account_external(6), 6)
        with self.assertRaisesRegex(ValueError, "aggregate read credit"):
            budget.account_external(5)
        self.assertEqual(budget.spent, 6)
        with self.assertRaises(ValueError):
            budget.account_external(True)
        self.assertEqual(budget.account_external(4), 10)
        expired = bootstrap.PinnedReadBudget(10, until_ns=200, clock=lambda: 200)
        with self.assertRaises(TimeoutError):
            expired.account_external(1)

    def test_external_block_callbacks_charge_actual_bytes_and_close_only_known_unused_probe_credit(self):
        budget = bootstrap.PinnedReadBudget(11, until_ns=1000, clock=lambda: 200)
        lease = budget.reserve_external(11)
        self.assertTrue(budget.observation()["unknown"])
        self.assertEqual(budget.charge_external(lease, 4), 4)
        self.assertEqual(budget.charge_external(lease, 6), 10)
        closed = budget.finish_external(lease, expected_bytes=10)
        self.assertEqual(closed, {"charged_bytes": 10, "known_unused_reserved_bytes": 1})
        self.assertEqual(budget.observation(), {"charged_bytes": 10, "unknown": False, "maximum_bytes": 11})
        self.assertEqual(budget.account_external(1), 11)
        with self.assertRaisesRegex(ValueError, "owner or lifetime"):
            budget.charge_external(lease, 0)

    def test_failed_external_callback_keeps_unknown_reserved_credit_and_returned_bytes(self):
        now = [200]
        budget = bootstrap.PinnedReadBudget(11, until_ns=1000, clock=lambda: now[0])
        lease = budget.reserve_external(11)
        now[0] = 1000
        with self.assertRaises(TimeoutError):
            budget.charge_external(lease, 3)
        self.assertEqual(budget.observation(), {"charged_bytes": 3, "unknown": True, "maximum_bytes": 11})
        budget.fail_external(lease)
        with self.assertRaisesRegex(ValueError, "quarantined"):
            budget.reserve_external(1)


@unittest.skipUnless(sys.platform == "linux" and hasattr(os, "O_NOFOLLOW"), "actual nofollow fixture uses Linux")
class RoventTrainingBootstrapPinTests(unittest.TestCase):
    def test_small_json_pin_and_streamed_read_share_exact_aggregate_credit(self):
        with tempfile.TemporaryDirectory(prefix="rovent-bootstrap-fixture-") as temporary:
            path = Path(temporary) / "metadata.json"
            raw = b'{"fixture":"no-model-no-authority"}'
            path.write_bytes(raw)
            declared = pin(str(path), raw)
            budget = bootstrap.PinnedReadBudget(len(raw) * 2, until_ns=1000, clock=lambda: 200)
            self.assertEqual(budget.json(declared), {"fixture": "no-model-no-authority"})
            observed = budget.verify(declared)
            self.assertEqual(observed["sha256"], declared["sha256"])
            self.assertEqual(budget.spent, len(raw) * 2)
            with self.assertRaisesRegex(ValueError, "aggregate read credit"):
                budget.json(declared)

    def test_links_wrong_bytes_and_changed_hash_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="rovent-bootstrap-fixture-") as temporary:
            root = Path(temporary)
            path = root / "metadata.json"
            raw = b'{"fixture":1}'
            path.write_bytes(raw)
            declared = pin(str(path), raw)
            link = root / "linked.json"
            link.symlink_to(path)
            for changed in ({**declared, "path": str(link)}, {**declared, "sha256": "0" * 64},
                    {**declared, "bytes": len(raw) + 1}):
                with self.subTest(changed=changed), self.assertRaises((ValueError, OSError)):
                    bootstrap.read_original_pin(changed, until_ns=1000, clock=lambda: 200)
            hardlink = root / "hardlink.json"
            os.link(path, hardlink)
            with self.assertRaisesRegex(ValueError, "single-link"):
                bootstrap.read_original_pin(declared, until_ns=1000, clock=lambda: 200)

    def test_failed_file_pin_retains_charged_reads_and_quarantines_the_account(self):
        with tempfile.TemporaryDirectory(prefix="rovent-bootstrap-fixture-") as temporary:
            path = Path(temporary) / "metadata.json"
            raw = b'{"fixture":1}'
            path.write_bytes(raw)
            declared = {**pin(str(path), raw), "sha256": "0" * 64}
            budget = bootstrap.PinnedReadBudget(1000, until_ns=1000, clock=lambda: 200)
            with self.assertRaises(ValueError):
                budget.read(declared)
            self.assertEqual(budget.spent, len(raw))
            with self.assertRaisesRegex(ValueError, "quarantined"):
                budget.read(pin(str(path), raw))


@unittest.skipUnless(sys.platform == "linux" and hasattr(os, "O_NOFOLLOW"), "owned directory fixtures use Linux")
class RoventOwnedRuntimeEnvironmentTests(unittest.TestCase):
    def test_only_public_runtime_paths_are_set_under_the_owned_output(self):
        with tempfile.TemporaryDirectory(prefix="rovent-owned-environment-fixture-") as temporary:
            root = Path(temporary) / "output"
            root.mkdir(mode=0o700)
            startup = {"registration": {"output_root": str(root)}}
            with mock.patch.object(tempfile, "tempdir", None), mock.patch.dict(os.environ, {"FIXTURE_MARKER": "unchanged"}, clear=True):
                report = bootstrap._prepare_owned_runtime_environment(startup, until_ns=1000, clock=lambda: 200)
                self.assertEqual(set(report["variables"]), set(bootstrap._RUNTIME_ENVIRONMENT))
                self.assertEqual(set(os.environ), {*bootstrap._RUNTIME_ENVIRONMENT, "FIXTURE_MARKER"})
                self.assertEqual(os.environ["FIXTURE_MARKER"], "unchanged")
                for name, value in report["variables"].items():
                    self.assertEqual(os.environ[name], value)
                    self.assertTrue(Path(value).is_relative_to(root))
                    self.assertEqual(stat.S_IMODE(Path(value).stat().st_mode), 0o700)
                self.assertEqual(len(report["directories"]), 7)
                self.assertFalse(report["global_cache_inspected"])
                self.assertFalse(report["automatic_cleanup"])
            self.assertEqual(set(path.name for path in root.iterdir()), {"tmp", "home", "runtime-cache"})

    def test_existing_nonempty_runtime_directory_is_preserved_and_not_published(self):
        with tempfile.TemporaryDirectory(prefix="rovent-owned-environment-fixture-") as temporary:
            root = Path(temporary) / "output"
            root.mkdir(mode=0o700)
            (root / "tmp").mkdir(mode=0o700)
            retained = root / "tmp" / "retained-fixture.txt"
            retained.write_text("preserve fixture evidence", encoding="utf-8")
            with mock.patch.object(tempfile, "tempdir", None), mock.patch.dict(os.environ, {"FIXTURE_MARKER": "unchanged"}, clear=True):
                with self.assertRaisesRegex(ValueError, "new or empty"):
                    bootstrap._prepare_owned_runtime_environment({"registration": {"output_root": str(root)}},
                        until_ns=1000, clock=lambda: 200)
                self.assertEqual(dict(os.environ), {"FIXTURE_MARKER": "unchanged"})
            self.assertEqual(retained.read_text(encoding="utf-8"), "preserve fixture evidence")
            self.assertEqual(set(path.name for path in root.iterdir()), {"tmp"})

    def test_linked_runtime_directory_is_not_followed_or_cleaned(self):
        with tempfile.TemporaryDirectory(prefix="rovent-owned-environment-fixture-") as temporary:
            root = Path(temporary) / "output"
            root.mkdir(mode=0o700)
            other = Path(temporary) / "other-fixture-owner"
            other.mkdir(mode=0o700)
            (root / "tmp").symlink_to(other, target_is_directory=True)
            with mock.patch.object(tempfile, "tempdir", None), mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaises((ValueError, OSError)):
                    bootstrap._prepare_owned_runtime_environment({"registration": {"output_root": str(root)}},
                        until_ns=1000, clock=lambda: 200)
                self.assertEqual(dict(os.environ), {})
            self.assertTrue((root / "tmp").is_symlink())
            self.assertEqual(list(other.iterdir()), [])

    def test_bad_owner_mode_or_expired_window_does_not_create_a_runtime_tree(self):
        with tempfile.TemporaryDirectory(prefix="rovent-owned-environment-fixture-") as temporary:
            root = Path(temporary) / "output"
            root.mkdir(mode=0o700)
            startup = {"registration": {"output_root": str(root)}}
            with mock.patch.object(tempfile, "tempdir", None), mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(TimeoutError):
                    bootstrap._prepare_owned_runtime_environment(startup, until_ns=200, clock=lambda: 200)
                root.chmod(0o755)
                with self.assertRaisesRegex(ValueError, "0700"):
                    bootstrap._prepare_owned_runtime_environment(startup, until_ns=1000, clock=lambda: 200)
                self.assertEqual(dict(os.environ), {})
            self.assertEqual(list(root.iterdir()), [])

    def test_previously_cached_temporary_path_is_rejected_without_inspecting_it(self):
        with tempfile.TemporaryDirectory(prefix="rovent-owned-environment-fixture-") as temporary:
            root = Path(temporary) / "output"
            root.mkdir(mode=0o700)
            with mock.patch.object(tempfile, "tempdir", "not-inspected-fixture-path"), mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(bootstrap.OriginalBootstrapUnavailable, "cached"):
                    bootstrap._prepare_owned_runtime_environment({"registration": {"output_root": str(root)}},
                        until_ns=1000, clock=lambda: 200)
                self.assertEqual(dict(os.environ), {})
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
