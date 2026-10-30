"""Pure utility startup fixtures; paths and process ids are never accessed."""
import copy
import hashlib
import json
import unittest

from rz_pals_model import rovent_utility_training_bootstrap as bootstrap


def pin(path, marker="11", size=10):
    return {"path": path, "sha256": marker * 32, "bytes": size}


def fixture():
    output, control = "/tmp/utility-output", "/tmp/utility-control"
    endpoints = [{"name": "torch.__file__", "artifact": pin("/opt/python/site-packages/torch/__init__.py", "12")},
                 {"name": "torch._C.__file__", "artifact": pin("/opt/python/site-packages/torch/_C.so", "13")}]
    source_names = ("__init__.py", "rovent_utility_training_child.py", "rovent_utility_training_bootstrap.py",
        "rovent_training_bootstrap.py", "rovent_utility_training_authority.py", "rovent_utility_training_window.py",
        "rovent_utility_training.py", "rovent_utility_training_run.py", "rovent_utility_tensor_training.py", "nnuec_utility_trainable.py")
    prefix = "experiments/model-research/pals/src/rz_pals_model/"
    sources = [{"relative_path": prefix + name, "sha256": "21" * 32, "bytes": 10} for name in source_names]
    inventory = {source["relative_path"]: {"sha256": source["sha256"], "bytes": source["bytes"]} for source in sources}
    source_raw = json.dumps(["rz-rovent-utility-trainer-source-bundle/1", inventory], sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    endpoint_inventory = sorted(({"name": endpoint["name"], "sha256": endpoint["artifact"]["sha256"],
        "bytes": endpoint["artifact"]["bytes"]} for endpoint in endpoints), key=lambda item: item["name"])
    dependency_raw = json.dumps(["rz-rovent-trainer-dependency-set/1", endpoint_inventory], sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    policy = {"cpu_quota_millicpus": 2000, "memory_high_bytes": 6 * 1024**3, "memory_max_bytes": 12 * 1024**3,
        "pids_max": 32, "affinity": [0, 2], "device": "cpu"}
    registration = {"schema": "rz-rovent-utility-training-registration/1", "execution_id": "utility-execution",
        "action": "train", "export_request": None, "preparation": pin("/data/preparation.json", "31"),
        "window_spec": pin("/data/window.json", "32"), "trainer_source_sha256": hashlib.sha256(source_raw).hexdigest(),
        "trainer_python_executable": pin("/opt/python/bin/python3", "33"), "trainer_python_version": "3.12.7",
        "dependency_directories": ["/opt/python/site-packages"], "runtime_endpoints": endpoints,
        "trainer_dependency_set_sha256": hashlib.sha256(dependency_raw).hexdigest(),
        "current_dataset": {"root": "/data/cohort", "publication": pin("/data/cohort/publication.json", "34"),
            "durable_ack": pin("/data/cohort/durable-ack.json", "39")},
        "ancestor_datasets": [], "initialization_source": {"mode": "initialize", "utility": pin("/data/utility.json", "35")},
        "frozen_nnuec_base": pin("/data/base.json", "36"),
        "frozen_pc_sources": [{"role": role, "artifact": pin("/data/" + role + ".pt", marker),
            "model_semantics_sha256": "41" * 32, "model_identity_sha256": "42" * 32, "encoding_sha256": "43" * 32}
            for role, marker in (("proposer", "37"), ("critic", "38"))],
        "resources_policy": policy, "require_total_quota": True, "output_root": output, "control_root": control,
        "checkpoint_directory": output + "/checkpoints", "receipt_path": output + "/receipt.json",
        "max_rpc_messages": 1000, "max_rpc_bytes": 1048576, "max_read_bytes": 8388608,
        "max_child_read_bytes": 4194304, "max_output_bytes": 8388608, "max_artifact_files": 64}
    readback = {"cpu_quota_us": 2000000, "cpu_period_us": 1000000, "memory_high_bytes": 6 * 1024**3,
        "memory_max_bytes": 12 * 1024**3, "memory_swap_max_bytes": 0, "pids_max": 32}
    return {"schema": "rz-rovent-utility-trainer-startup/1", "execution_id": "utility-execution", "window_id": "utility-window",
        "registration_sha256": "51" * 32, "registration": registration,
        "control": {"socket_path": control + "/parent.sock", "parent_pid": 123, "parent_start_ticks": 456,
            "control_device": 7, "control_inode": 8},
        "original_started_ns": 100, "execution_until_ns": 1000000100, "whole_until_ns": 2000000100,
        "source_root": output + "/source", "package_root": output + "/source/experiments/model-research/pals/src",
        "source_files": sources, "dependency_directories": [{"path": "/opt/python/site-packages", "device": 9, "inode": 10}],
        "runtime_module_pins": [endpoint["artifact"] for endpoint in endpoints],
        "python_runtime": {"version": "3.12.7", "executable": registration["trainer_python_executable"]},
        "resources": {"domain": "rz-rovent-training-resource-startup/1", "migration": "child_self_only_before_torch",
            "path": "/sys/fs/cgroup/utility/leaf", "device": 11, "inode": 12, "membership": "/utility/leaf", "policy": policy,
            "prepared_readback": readback, "readback": copy.deepcopy(readback),
            "available_controllers_snapshot": ["cpu", "memory", "pids"], "subtree_controllers_snapshot": ["cpu", "memory", "pids"],
            "scope": "configured_leaf_readback_not_child_membership_or_lifetime_proof"},
        "child_read_credit": {"reserved_bytes": 4194304, "scope": "charged_in_parent_shared_budget_no_refund"},
        "artifact_watch_profile": {"schema": "rz-rovent-training-artifact-watch/1", "maximum_total_bytes": 7340032,
            "maximum_file_bytes": 2 * 1024**3, "maximum_files": 64, "scope": "new_trainer_owned_output_tree_only"}}


def validate(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return bootstrap.validate_startup(raw, expected_sha256=hashlib.sha256(raw).hexdigest(), expected_bytes=len(raw),
        execution_until_ns=1000000100, whole_until_ns=2000000100, now_ns=200)


class UtilityBootstrapTests(unittest.TestCase):
    def test_exact_new_startup_is_a_pure_declaration_and_preserves_original_source_domains(self):
        value = fixture(); before = copy.deepcopy(value)
        self.assertEqual(validate(value), before)
        self.assertEqual(value, before)
        self.assertNotIn("execution_authorized", value)
        self.assertNotEqual(value["registration"]["trainer_python_executable"]["sha256"], value["registration"]["frozen_nnuec_base"]["sha256"])

    def test_legacy_domains_unknown_fields_deadline_and_child_credit_changes_are_rejected(self):
        for mutate in (lambda value: value.update(schema="rz-rovent-recovered-training-startup/1"),
                       lambda value: value.update(execution_authorized=True),
                       lambda value: value.update(execution_until_ns=1000000101),
                       lambda value: value["child_read_credit"].update(reserved_bytes=4194305),
                       lambda value: value["artifact_watch_profile"].update(maximum_files=True),
                       lambda value: value["registration"].update(source_sha256="52" * 32)):
            changed = fixture(); mutate(changed)
            with self.assertRaises(ValueError): validate(changed)

    def test_raw_byte_pin_and_duplicate_json_keys_cannot_be_reconstructed_from_a_projection(self):
        raw = b'{"schema":"one","schema":"two"}'
        with self.assertRaises(ValueError):
            bootstrap.validate_startup(raw, expected_sha256=hashlib.sha256(raw).hexdigest(), expected_bytes=len(raw),
                execution_until_ns=1000000100, whole_until_ns=2000000100, now_ns=200)
        raw = json.dumps(fixture()).encode("utf-8")
        with self.assertRaises(ValueError):
            bootstrap.validate_startup(raw, expected_sha256="00" * 32, expected_bytes=len(raw),
                execution_until_ns=1000000100, whole_until_ns=2000000100, now_ns=200)

    def test_frozen_pc_sources_are_required_without_instantiating_their_models(self):
        for mutate in (lambda value: value["registration"].update(frozen_pc_sources=[]),
                       lambda value: value["registration"]["frozen_pc_sources"][1].update(role="proposer"),
                       lambda value: value["registration"]["frozen_pc_sources"][0]["artifact"].update(path="/data/.env")):
            changed = fixture(); mutate(changed)
            with self.assertRaises(ValueError): validate(changed)

    def test_export_requires_explicit_resumed_checkpoint_publication(self):
        value = fixture(); value["registration"].update(action="export", export_request=pin("/data/export-request.json", "61"))
        with self.assertRaises(ValueError): validate(value)
        value["registration"]["initialization_source"] = {"mode": "resume",
            "checkpoint": {"root": "/data/old-utility", "publication": pin("/data/old-utility/publication.json", "62")}}
        self.assertEqual(validate(value)["registration"]["action"], "export")

    def test_dataset_durable_ack_is_required_without_changing_the_checkpoint_reference(self):
        value = fixture(); value["registration"]["current_dataset"].pop("durable_ack")
        with self.assertRaises(ValueError): validate(value)
        value = fixture(); value["registration"]["initialization_source"] = {"mode": "resume",
            "checkpoint": {"root": "/data/old-utility", "publication": pin("/data/old-utility/publication.json", "62")}}
        self.assertEqual(validate(value)["registration"]["initialization_source"]["mode"], "resume")
        value["registration"]["initialization_source"]["checkpoint"]["durable_ack"] = pin("/data/old-utility/durable-ack.json", "63")
        with self.assertRaises(ValueError): validate(value)

    def test_registration_v2_is_a_separate_explicit_mixed_ancestry_codec(self):
        value = fixture()["registration"]
        old = copy.deepcopy(value)
        value["schema"] = "rz-rovent-utility-training-registration/2"
        ancestor = {"kind": "utility_cohort", "registration": {"root": "/data/ancestor",
            "publication": pin("/data/ancestor/utility-cohort-publication.json", "71"),
            "durable_ack": pin("/data/ancestor/utility-cohort-durable-ack.json", "72")},
            "preparation": pin("/data/ancestor-preparation.json", "73")}
        value["ancestor_datasets"] = [ancestor]
        self.assertEqual(bootstrap.validate_utility_registration_v2(value), value)
        self.assertEqual(bootstrap.validate_registration(old), old)
        with self.assertRaises(ValueError): bootstrap.validate_registration(value)
        with self.assertRaises(ValueError): bootstrap.validate_utility_registration_v2(old)

    def test_v2_ancestor_tags_do_not_infer_or_drop_original_preparation_and_ack(self):
        value = fixture()["registration"]
        value["schema"] = "rz-rovent-utility-training-registration/2"
        ancestor = {"kind": "utility_cohort", "registration": {"root": "/data/ancestor",
            "publication": pin("/data/ancestor/publication.json", "71"),
            "durable_ack": pin("/data/ancestor/ack.json", "72")},
            "preparation": pin("/data/preparation-other.json", "73")}
        for mutate in (lambda item: item.pop("preparation"), lambda item: item.update(kind="unknown"),
                       lambda item: item["registration"].pop("durable_ack"),
                       lambda item: item.update(recipe=pin("/data/recipe.json", "74")),
                       lambda item: item["registration"]["publication"].update(sha256=value["current_dataset"]["publication"]["sha256"])):
            changed = copy.deepcopy(value); changed["ancestor_datasets"] = [copy.deepcopy(ancestor)]
            mutate(changed["ancestor_datasets"][0])
            with self.assertRaises(ValueError): bootstrap.validate_utility_registration_v2(changed)

    def test_v2_requires_a_distinct_direct_json_terminal_without_changing_legacy_metadata(self):
        value = fixture()["registration"]
        value["schema"] = "rz-rovent-utility-training-registration/2"
        self.assertEqual(bootstrap.validate_utility_registration_v2(value), value)
        for path in (value["output_root"] + "/nested/receipt.json", value["output_root"] + "/receipt.pt", value["checkpoint_directory"]):
            changed = copy.deepcopy(value); changed["receipt_path"] = path
            with self.assertRaises(ValueError): bootstrap.validate_utility_registration_v2(changed)
        old = fixture()["registration"]; old["receipt_path"] = old["output_root"] + "/nested/receipt.bin"
        self.assertEqual(bootstrap.validate_registration(old), old)


if __name__ == "__main__":
    unittest.main()
