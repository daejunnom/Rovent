"""Pure provenance/bridge checks; no model, optimizer, ONNX or NN execution."""
import copy
import builtins
import hashlib
import importlib
import importlib.abc
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("neural execution is forbidden in generation export contract tests")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import generation_training as train
    from rz_pals_model import generation_data as data
    from rz_pals_model.config import ModelConfig, MODEL_SCHEMA_V3
finally:
    sys.meta_path.remove(_guard)


def spec():
    return train.generation_spec(generation_id="g1", source_sha256="1" * 64,
        dataset_sha256="2" * 64, split_sha256="3" * 64,
        model_sha256="4" * 64, runtime_sha256="5" * 64)


def source_receipt():
    """Schema-only metadata: no actual games, tensor, update or export evidence."""
    splits = {f"opening-{i}": "train" if i < 48 else "validation" if i < 56 else "holdout"
              for i in range(64)}
    registered = spec()
    registered["split_sha256"] = data.digest("rz-pals-opening-split/1", splits)
    registered_sha = train.spec_digest(registered)
    frozen = "6" * 64

    def admission(phase):
        producer = {"schema": "rz-pals-actual-collection-authority/2", "source_kind": "actual_selfplay",
                    "source_sha256": registered["source_sha256"],
                    "model_sha256": registered["model_sha256"] if phase == "pc" else frozen,
                    "runtime_sha256": registered["runtime_sha256"], "teacher_binary_sha256": "7" * 64,
                    "teacher_profile_sha256": "8" * 64, "opening_split_sha256": registered["split_sha256"],
                    "registered_before_outcomes": True, "actual_game_ids": [f"game-{i}" for i in range(128)]}
        observed = {"proposer_valid_samples": 1024, "critic_valid_samples": 1024} if phase == "pc" else {
            "validator_unique_inputs": 2048, "validator_known_pairs": 128, "validator_new_witness_contexts": 32}
        return {"schema": train.SPEC_DOMAIN, "spec_sha256": registered_sha,
                "dataset_sha256": registered["dataset_sha256"] if phase == "pc" else "9" * 64,
                "frozen_pc_sha256": None if phase == "pc" else frozen, "phase": phase, "coverage": observed,
                "data_format": "rz-pals-generation-chunk-manifest/1", "producer": producer,
                "opening_splits": copy.deepcopy(splits), "model_epoch": 0 if phase == "pc" else 1,
                "context_epoch": 0, "native_chain_verified": True, "actual_training_executed": False,
                "optimizer_updates": 0, "arena_eligible": False,
                "inference_profile": {"mode": "fresh", "batch_size": 1},
                "profile_exclusions": dict.fromkeys(("cross_split_records_excluded", "cross_split_identities",
                    "duplicate_inputs_excluded", "duplicate_semantic_inputs_excluded", "warm_seed_inputs_excluded"), 0)}

    def artifact(kind, include_validator):
        return {"roles": ["proposer", "critic"] + (["validator"] if include_validator else []),
                "validator_present": include_validator,
                "manifest": {"file": kind + "/export.json", "sha256": "a" * 64, "bytes": 1024},
                "graphs": [{"role": role, "file": kind + "/" + file, "sha256": str(i) * 64, "bytes": 4096}
                           for i, (role, file) in enumerate([
                               ("public", "public_memory.onnx"), ("shared_pc", "shared_pc_if.onnx")]
                               + ([("validator", "role_validator.onnx")] if include_validator else []), 1)]}

    ledger = train.GenerationLedger(registered, completed=dict(train.UPDATES), dispatched=dict(train.UPDATES),
        phase="complete", next_role="validator")
    ledger.cost.update(forward_calls=40000, backward_calls=40000, samples=40000, wall_time_ns=100000)
    checkpoint = {"schema": "rz-pals-verified-generation-checkpoint/1", "checkpoint_domain": train.CHECKPOINT_DOMAIN,
        "eligibility": train.ELIGIBILITY, "file": "checkpoint.pt", "sha256": "b" * 64, "bytes": 8192,
        "phase": "complete", "pending_microbatches": 0, "completed_updates": dict(train.UPDATES),
        "dispatched_updates": dict(train.UPDATES), "ledger_sha256": data.digest("rz-pals-generation-ledger-state/1", ledger.state()),
        "cost": copy.deepcopy(ledger.cost), "state_sha256": dict.fromkeys(
            ("model", "optimizers", "schedulers", "rng", "sampler", "parameter_updates", "membership", "bucket"), "c" * 64),
        "config_sha256": data.digest("rz-pals-generation-model-config/1", ModelConfig.for_profile("virtual_review_v1").to_dict()),
        "frozen_pc_sha256": frozen, "frozen_pc_model_epoch": 1,
        "context_epoch": 0, "nonzero_parameter_owners": ["shared", "proposer", "critic", "validator"], "cuda_devices": [0]}
    return {"schema": train.EXPORT_DOMAIN, "source_kind": "actual_selfplay", "eligibility": "evaluation_candidate_pending_gpu_acceptance",
        "spec": registered, "spec_sha256": registered_sha, "generation_id": registered["generation_id"],
        "source_sha256": registered["source_sha256"], "runtime_sha256": registered["runtime_sha256"],
        "checkpoint": checkpoint, "pc_admission": admission("pc"), "v_admission": admission("v"),
        "collection": artifact("collection", True), "deployment": artifact("deployment", False), "numeric_status": "not_run",
        "cuda_status": "not_run", "arena_accepted": False}


def reseal_cost(value, **changes):
    """Reframe only synthetic cost metadata so relation tests reach validation."""
    summary = value["checkpoint"]
    summary["cost"].update(changes)
    ledger = train.GenerationLedger(value["spec"], completed=copy.deepcopy(summary["completed_updates"]),
        dispatched=copy.deepcopy(summary["dispatched_updates"]), phase="complete", next_role="validator",
        cost=copy.deepcopy(summary["cost"]))
    summary["ledger_sha256"] = data.digest("rz-pals-generation-ledger-state/1", ledger.state())


def tiny_ascii_pins(directory, value):
    """Owned ASCII/JSON digest fixtures, not valid checkpoint/ONNX/NN assets."""
    pins = [value["checkpoint"]]
    for kind in ("collection", "deployment"):
        pins.extend(value[kind]["graphs"])
    for pin in pins:
        path = directory / pin["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        contents = f"contract-only ASCII file pin {pin['file']}\n".encode("ascii")
        with path.open("xb") as stream:
            stream.write(contents)
        pin["bytes"], pin["sha256"] = len(contents), hashlib.sha256(contents).hexdigest()
    config = ModelConfig.for_profile("virtual_review_v1")
    for kind in ("collection", "deployment"):
        artifact = value[kind]
        manifest = {"schema": MODEL_SCHEMA_V3, "model_semantics": config.model_semantics,
            "encoding_schema": config.encoding, "model_profile": config.profile, "config": config.to_dict(),
            "checkpoint_sha256": value["checkpoint"]["sha256"], "trained": True, "training_steps": 5000,
            "roles": artifact["roles"], "validator_present": artifact["validator_present"],
            "precision": "fp32", "tf32": False, "numeric_status": "not_run", "cuda_status": "not_run",
            "graphs": [{"role": pin["role"], "file": Path(pin["file"]).name, "sha256": pin["sha256"]}
                       for pin in artifact["graphs"]]}
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
        pin = artifact["manifest"]
        with (directory / pin["file"]).open("xb") as stream:
            stream.write(encoded)
        pin["bytes"], pin["sha256"] = len(encoded), hashlib.sha256(encoded).hexdigest()
        pins.append(pin)
    return pins


class ForbiddenModel:
    """A state-access sentinel, never a tensor or model implementation."""
    def state_dict(self):
        raise AssertionError("invalid provenance reached caller model state")


class SourceOnlyTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)
        self.addCleanup(sys.meta_path.remove, self.guard)
        self.original_import = builtins.__import__
        self.addCleanup(setattr, builtins, "__import__", self.original_import)

        def forbidden_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
                raise AssertionError("neural library import forbidden even when already cached")
            return self.original_import(name, *args, **kwargs)

        builtins.__import__ = forbidden_import

class DeploymentBoundaryTests(SourceOnlyTests):
    def test_incomplete_deployment_refuses_before_model_or_neural_import(self):
        ledger = train.GenerationLedger(spec())
        with self.assertRaises(ValueError):
            train.deployment_state(ForbiddenModel(), ledger, checkpoint_sha256="a" * 64)

    def test_complete_phase_alone_cannot_replace_actual_updates(self):
        ledger = train.GenerationLedger(spec())
        ledger.phase, ledger.next_role = "complete", "validator"
        with self.assertRaises(ValueError):
            train.deployment_state(ForbiddenModel(), ledger, checkpoint_sha256="a" * 64)

    def test_inconsistent_complete_ledger_refuses_before_state_access(self):
        for mutation in (
            lambda x: x.dispatched.update(validator=999),
            lambda x: x.completed.update(validator=999),
            lambda x: x.completed.update(proposer=True),
            lambda x: x.completed.update(critic=2000.0),
            lambda x: x.cost.update(samples=-1),
            lambda x: setattr(x, "pending_microbatches", 1),
        ):
            ledger = train.GenerationLedger(spec(), completed=dict(train.UPDATES),
                dispatched=dict(train.UPDATES), phase="complete", next_role="validator")
            mutation(ledger)
            before = copy.deepcopy(ledger.__dict__)
            with self.subTest(ledger=before):
                with self.assertRaises(ValueError):
                    train.deployment_state(ForbiddenModel(), ledger, checkpoint_sha256="a" * 64)
                self.assertEqual(before, ledger.__dict__)

    def test_invalid_source_spec_cannot_reach_neural_loader(self):
        for field, invalid in (("schema", "rz-pals-nonzero-diagnostic/1"),
                               ("profile", "legacy_summary_v1"),
                               ("source_sha256", "unknown")):
            ledger = train.GenerationLedger(spec(), completed=dict(train.UPDATES),
                dispatched=dict(train.UPDATES), phase="complete", next_role="validator")
            ledger.spec[field] = invalid
            before = copy.deepcopy(ledger.__dict__)
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    train.deployment_state(ForbiddenModel(), ledger, checkpoint_sha256="a" * 64)
                self.assertEqual(before, ledger.__dict__)


class ProvenanceContractTests(SourceOnlyTests):
    def setUp(self):
        super().setUp()
        self.export = importlib.import_module("rz_pals_model.generation_export")

    def validate(self, value, **expected):
        return self.export.validate_actual_generation_provenance(value,
            expected_sha256=self.export.provenance_digest(value), **expected)

    def rejected_mutations(self, mutations):
        for name, mutation in mutations:
            value = source_receipt()
            mutation(value)
            before = copy.deepcopy(value)
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    self.validate(value)
                self.assertEqual(value, before)

    def test_registered_metadata_contract_accepts_without_neural_import(self):
        value = source_receipt()
        before = copy.deepcopy(value)
        self.validate(value, expected_spec_sha256=value["spec_sha256"],
            expected_checkpoint_sha256=value["checkpoint"]["sha256"])
        self.assertEqual(value, before)
        self.assertFalse(value["arena_accepted"])
        self.assertEqual(value["numeric_status"], "not_run")
        self.assertEqual(value["cuda_status"], "not_run")

    def test_rehashed_unknown_or_diagnostic_domains_still_refuse(self):
        self.rejected_mutations([
            ("receipt-domain", lambda v: v.update(schema="rz-pals-nonzero-checkpoint/1")),
            ("receipt-eligibility", lambda v: v.update(eligibility="diagnostic_only_arena_ineligible")),
            ("checkpoint-domain", lambda v: v["checkpoint"].update(checkpoint_domain="rz-pals-nonzero-checkpoint/1")),
            ("checkpoint-schema", lambda v: v["checkpoint"].update(schema="rz-pals-verified-generation-checkpoint/2")),
            ("checkpoint-eligibility", lambda v: v["checkpoint"].update(eligibility="diagnostic_only_arena_ineligible")),
            ("unknown-collection-source", lambda v: v["pc_admission"]["producer"].update(source_kind="unknown")),
            ("diagnostic-v-source", lambda v: v["v_admission"]["producer"].update(source_kind="diagnostic_fixture")),
            ("in-memory-authority", lambda v: v["pc_admission"]["producer"].update(schema="rz-pals-actual-collection-authority/1")),
        ])

    def test_partial_phase_allmasked_and_typed_update_aliases_refuse(self):
        self.rejected_mutations([
            ("p-only", lambda v: v["checkpoint"]["completed_updates"].update(critic=0, validator=0)),
            ("v-not-finished", lambda v: v["checkpoint"]["completed_updates"].update(validator=999)),
            ("phase-in-progress", lambda v: v["checkpoint"].update(phase="v")),
            ("partial-accumulation", lambda v: v["checkpoint"].update(pending_microbatches=1)),
            ("completed-before-dispatch", lambda v: v["checkpoint"]["dispatched_updates"].update(proposer=1999)),
            ("bool-update-alias", lambda v: v["checkpoint"]["completed_updates"].update(proposer=True)),
            ("float-update-alias", lambda v: v["checkpoint"]["completed_updates"].update(critic=2000.0)),
            ("zero-forward-work", lambda v: v["checkpoint"]["cost"].update(forward_calls=0)),
            ("zero-backward-work", lambda v: v["checkpoint"]["cost"].update(backward_calls=0)),
            ("zero-valid-samples", lambda v: v["pc_admission"]["coverage"].update(proposer_valid_samples=0)),
            ("unknown-v-targets", lambda v: v["v_admission"]["coverage"].update(validator_known_pairs=0)),
        ])

    def test_positive_work_below_update_cost_and_bad_relations_refuse(self):
        self.rejected_mutations([
            ("low-forward", lambda v: reseal_cost(v, forward_calls=1, samples=1)),
            ("low-backward", lambda v: reseal_cost(v, backward_calls=1)),
            ("low-samples", lambda v: reseal_cost(v, samples=1)),
            ("more-backward-than-forward", lambda v: reseal_cost(v, backward_calls=40001)),
            ("samples-not-forward", lambda v: reseal_cost(v, samples=40001)),
        ])

    def test_typed_coverage_games_and_split_registration_refuse(self):
        def replace_registered_openings(value):
            for phase in ("pc_admission", "v_admission"):
                splits = value[phase]["opening_splits"]
                value[phase]["opening_splits"] = {"another-" + key: split for key, split in splits.items()}

        self.rejected_mutations([
            ("coverage-bool", lambda v: v["pc_admission"]["coverage"].update(proposer_valid_samples=True)),
            ("coverage-float", lambda v: v["pc_admission"]["coverage"].update(critic_valid_samples=1024.0)),
            ("coverage-negative", lambda v: v["v_admission"]["coverage"].update(validator_known_pairs=-1)),
            ("game-duplicate", lambda v: v["pc_admission"]["producer"]["actual_game_ids"].__setitem__(1, "game-0")),
            ("unknown-split", lambda v: v["pc_admission"]["opening_splits"].update({"opening-0": "unknown"})),
            ("wrong-split-count", lambda v: v["v_admission"]["opening_splits"].update({"opening-0": "holdout"})),
            ("wrong-admission-phase", lambda v: v["pc_admission"].update(phase="v")),
            ("wrong-v-context", lambda v: v["v_admission"].update(context_epoch=1)),
            ("different-teacher-binary", lambda v: v["v_admission"]["producer"].update(teacher_binary_sha256="f" * 64)),
            ("different-teacher-profile", lambda v: v["v_admission"]["producer"].update(teacher_profile_sha256="f" * 64)),
            ("same-shape-different-split", replace_registered_openings),
            ("different-valid-ledger-sha", lambda v: v["checkpoint"].update(ledger_sha256="f" * 64)),
        ])

    def test_hash_source_runtime_and_generation_join_refuses_mismatch(self):
        self.rejected_mutations([
            ("source", lambda v: v.update(source_sha256="f" * 64)),
            ("runtime", lambda v: v.update(runtime_sha256="f" * 64)),
            ("generation-id", lambda v: v.update(generation_id="another-generation")),
            ("spec-seal", lambda v: v.update(spec_sha256="f" * 64)),
            ("pc-source", lambda v: v["pc_admission"]["producer"].update(source_sha256="f" * 64)),
            ("v-runtime", lambda v: v["v_admission"]["producer"].update(runtime_sha256="f" * 64)),
            ("pc-g0-weight", lambda v: v["pc_admission"]["producer"].update(model_sha256="f" * 64)),
            ("v-frozen-weight", lambda v: v["v_admission"]["producer"].update(model_sha256="f" * 64)),
            ("pc-dataset", lambda v: v["pc_admission"].update(dataset_sha256="f" * 64)),
            ("pc-spec", lambda v: v["pc_admission"].update(spec_sha256="f" * 64)),
            ("v-spec", lambda v: v["v_admission"].update(spec_sha256="f" * 64)),
            ("invalid-checkpoint-sha", lambda v: v["checkpoint"].update(sha256="unknown")),
            ("invalid-state-sha", lambda v: v["checkpoint"]["state_sha256"].update(rng="unknown")),
        ])

    def test_native_chain_training_and_future_acceptance_claims_refuse(self):
        self.rejected_mutations([
            ("pc-native-unknown", lambda v: v["pc_admission"].update(native_chain_verified=False)),
            ("v-native-unknown", lambda v: v["v_admission"].update(native_chain_verified=None)),
            ("in-memory-format", lambda v: v["pc_admission"].update(data_format="rz-pals-generation-data/3")),
            ("self-approved-admission", lambda v: v["pc_admission"].update(arena_eligible=True)),
            ("forged-admission-training", lambda v: v["v_admission"].update(actual_training_executed=True)),
            ("forged-admission-updates", lambda v: v["pc_admission"].update(optimizer_updates=5000)),
            ("future-numeric-claim", lambda v: v.update(numeric_status="passed")),
            ("future-cuda-claim", lambda v: v.update(cuda_status="passed")),
            ("future-arena-claim", lambda v: v.update(arena_accepted=True)),
            ("not-preregistered", lambda v: v["pc_admission"]["producer"].update(registered_before_outcomes=False)),
        ])

    def test_first_generation_profile_and_exclusion_types_refuse_aliases(self):
        self.rejected_mutations([
            ("warm-admission", lambda v: v["pc_admission"]["inference_profile"].update(mode="warm")),
            ("batch16-admission", lambda v: v["v_admission"]["inference_profile"].update(batch_size=16)),
            ("bool-batch-alias", lambda v: v["pc_admission"]["inference_profile"].update(batch_size=True)),
            ("warm-spec", lambda v: v["spec"]["collection_inference"].update(mode="warm")),
            ("batch16-spec", lambda v: v["spec"]["collection_inference"].update(batch_size=16)),
            ("bool-exclusion", lambda v: v["pc_admission"]["profile_exclusions"].update(warm_seed_inputs_excluded=True)),
            ("negative-exclusion", lambda v: v["v_admission"]["profile_exclusions"].update(duplicate_inputs_excluded=-1)),
            ("float-exclusion", lambda v: v["v_admission"]["profile_exclusions"].update(duplicate_semantic_inputs_excluded=0.0)),
            ("missing-exclusion", lambda v: v["v_admission"]["profile_exclusions"].pop("warm_seed_inputs_excluded")),
        ])

    def test_freeze_epoch_and_actual_parameter_owner_evidence_refuse(self):
        self.rejected_mutations([
            ("no-freeze", lambda v: v["checkpoint"].update(frozen_pc_sha256=None)),
            ("same-g0-epoch", lambda v: v["checkpoint"].update(frozen_pc_model_epoch=0)),
            ("context-changed", lambda v: v["checkpoint"].update(context_epoch=1)),
            ("v-freeze-join", lambda v: v["v_admission"].update(frozen_pc_sha256="f" * 64)),
            ("v-epoch-join", lambda v: v["v_admission"].update(model_epoch=2)),
            ("wrong-v-device", lambda v: v["checkpoint"].update(cuda_devices=[])),
            ("bool-device-alias", lambda v: v["checkpoint"].update(cuda_devices=[False])),
            ("no-v-updated-owner", lambda v: v["checkpoint"].update(nonzero_parameter_owners=["shared", "proposer", "critic"])),
            ("unknown-owner-alias", lambda v: v["checkpoint"].update(nonzero_parameter_owners=["shared", "proposer", "critic", "module.validator"])),
        ])

    def test_deployment_cannot_contain_validator_or_unsupported_graphs(self):
        self.rejected_mutations([
            ("v-present", lambda v: v["deployment"].update(validator_present=True)),
            ("v-role", lambda v: v["deployment"]["roles"].append("validator")),
            ("v-graph", lambda v: v["deployment"]["graphs"].append(copy.deepcopy(v["collection"]["graphs"][-1]))),
            ("v-graph-alias", lambda v: v["deployment"]["graphs"][1].update(role="module.validator")),
            ("unknown-initializer-bank", lambda v: v["deployment"].update(initializer_banks={"validator": "f" * 64})),
            ("collection-no-v", lambda v: v["collection"].update(validator_present=False)),
            ("collection-missing-v", lambda v: v["collection"]["graphs"].pop()),
            ("duplicate-graph", lambda v: v["deployment"]["graphs"].append(copy.deepcopy(v["deployment"]["graphs"][0]))),
            ("path-traversal", lambda v: v["deployment"]["graphs"][0].update(file="../public_memory.onnx")),
            ("absolute-artifact", lambda v: v["collection"]["manifest"].update(file="C:/secret/export.json")),
        ])

    def test_artifact_byte_type_binary_pin_and_registered_name_refuse(self):
        self.rejected_mutations([
            ("zero-bytes", lambda v: v["deployment"]["manifest"].update(bytes=0)),
            ("negative-bytes", lambda v: v["deployment"]["graphs"][0].update(bytes=-1)),
            ("bool-bytes", lambda v: v["collection"]["graphs"][0].update(bytes=True)),
            ("unknown-manifest-pin", lambda v: v["collection"]["manifest"].update(sha256="unknown")),
            ("unknown-binary-pin", lambda v: v["deployment"]["graphs"][0].update(sha256="unknown")),
            ("wrong-public-filename", lambda v: v["collection"]["graphs"][0].update(file="collection/not_public.onnx")),
        ])

    def test_exact_fields_and_expected_pins_are_checked(self):
        self.rejected_mutations([
            ("unknown-trained-alias", lambda v: v.update(trained=True, training_steps=5000)),
            ("unknown-checkpoint-field", lambda v: v["checkpoint"].update(optimizer_restored=True)),
            ("missing-state-sha", lambda v: v["checkpoint"]["state_sha256"].pop("membership")),
            ("missing-native-proof", lambda v: v["v_admission"].pop("native_chain_verified")),
            ("unknown-graph-field", lambda v: v["deployment"]["graphs"][0].update(binary_sha256="f" * 64)),
        ])
        value = source_receipt()
        before = copy.deepcopy(value)
        for expected in ({"expected_spec_sha256": "f" * 64}, {"expected_checkpoint_sha256": "f" * 64}):
            with self.subTest(expected=expected):
                with self.assertRaises(ValueError):
                    self.validate(value, **expected)
                self.assertEqual(value, before)
        with self.assertRaises(ValueError):
            self.export.validate_actual_generation_provenance(value, expected_sha256="f" * 64)

    def test_export_default_gate_requires_no_file_or_model_access(self):
        kwargs = dict(expected_checkpoint_sha256="b" * 64, model=ForbiddenModel(), optimizers=None,
            spec=spec(), pc_dataset=None, v_dataset=None, frozen_pc_model_epoch=1,
            sampler_validator=None, rules_profile_json="unopened-rules.json", directory="uncreated-export")
        for mode in ({}, {"execute": False}):
            with self.subTest(mode=mode):
                with mock.patch("builtins.open", side_effect=AssertionError("gate opened file")) as builtin_open, \
                     mock.patch.object(Path, "open", side_effect=AssertionError("gate opened path")) as path_open, \
                     mock.patch.object(Path, "read_bytes", side_effect=AssertionError("gate read path")) as read, \
                     mock.patch.object(Path, "mkdir", side_effect=AssertionError("gate created output")) as mkdir:
                    with self.assertRaises(ValueError):
                        self.export.export_actual_generation("unopened-checkpoint.pt", **kwargs, **mode)
                    for operation in (builtin_open, path_open, read, mkdir):
                        operation.assert_not_called()


class BoundedArtifactFileTests(SourceOnlyTests):
    def setUp(self):
        super().setUp()
        self.export = importlib.import_module("rz_pals_model.generation_export")

    def test_small_ascii_and_json_pins_only_validate_bytes(self):
        with tempfile.TemporaryDirectory(prefix="rz-generation-export-contract-") as temporary:
            root, value = Path(temporary), source_receipt()
            tiny_ascii_pins(root, value)
            before = copy.deepcopy(value)
            result = self.export.verify_actual_generation_artifacts(root, value,
                expected_sha256=self.export.provenance_digest(value))
            self.assertEqual(result, before)
            self.assertEqual(value, before)
            self.assertEqual(result["numeric_status"], "not_run")
            self.assertFalse(result["arena_accepted"])

    def test_checkpoint_graph_or_manifest_bytes_changed_after_pin_refuse(self):
        for relative in ("checkpoint.pt", "deployment/shared_pc_if.onnx", "collection/export.json"):
            with self.subTest(relative=relative):
                with tempfile.TemporaryDirectory(prefix="rz-generation-export-contract-") as temporary:
                    root, value = Path(temporary), source_receipt()
                    tiny_ascii_pins(root, value)
                    path = root / relative
                    path.write_bytes(b"X" * path.stat().st_size)
                    before = copy.deepcopy(value)
                    with self.assertRaises(ValueError):
                        self.export.verify_actual_generation_artifacts(root, value,
                            expected_sha256=self.export.provenance_digest(value))
                    self.assertEqual(value, before)

    def test_hardlinked_ascii_artifact_refuses(self):
        with tempfile.TemporaryDirectory(prefix="rz-generation-export-contract-") as temporary:
            root, value = Path(temporary), source_receipt()
            tiny_ascii_pins(root, value)
            try:
                os.link(root / "checkpoint.pt", root / "second-contract-name.txt")
            except OSError as error:
                self.skipTest("test filesystem does not support owned hardlinks: " + type(error).__name__)
            with self.assertRaises(ValueError):
                self.export.verify_actual_generation_artifacts(root, value,
                    expected_sha256=self.export.provenance_digest(value))

    def test_symlinked_ascii_artifact_refuses(self):
        with tempfile.TemporaryDirectory(prefix="rz-generation-export-contract-") as temporary:
            root, value = Path(temporary), source_receipt()
            tiny_ascii_pins(root, value)
            source = root / "deployment/public_memory.onnx"
            substitute = root / "owned-source-contract.txt"
            source.replace(substitute)
            try:
                source.symlink_to(substitute)
            except OSError as error:
                self.skipTest("test filesystem does not support owned symlinks: " + type(error).__name__)
            with self.assertRaises(ValueError):
                self.export.verify_actual_generation_artifacts(root, value,
                    expected_sha256=self.export.provenance_digest(value))

    def test_rehashed_manifest_cannot_forge_training_roles_or_model_pin(self):
        mutations = [
            ("not-trained", lambda m: m.update(trained=False)),
            ("float-step-alias", lambda m: m.update(training_steps=5000.0)),
            ("missing-update", lambda m: m.update(training_steps=4999)),
            ("unknown-schema", lambda m: m.update(schema="unknown")),
            ("wrong-model-semantics", lambda m: m.update(model_semantics="legacy_summary_v1")),
            ("wrong-model-profile", lambda m: m.update(model_profile="legacy_summary_v1")),
            ("wrong-encoding", lambda m: m.update(encoding_schema="unknown")),
            ("wrong-config", lambda m: m["config"].update(width=256)),
            ("wrong-checkpoint", lambda m: m.update(checkpoint_sha256="f" * 64)),
            ("wrong-precision", lambda m: m.update(precision="fp16")),
            ("tf32-enabled", lambda m: m.update(tf32=True)),
            ("deployment-v-role", lambda m: m["roles"].append("validator")),
            ("deployment-v-presence", lambda m: m.update(validator_present=True)),
            ("wrong-graph-sha", lambda m: m["graphs"][0].update(sha256="f" * 64)),
            ("wrong-graph-name", lambda m: m["graphs"][0].update(file="../other.onnx")),
            ("future-numeric", lambda m: m.update(numeric_status="passed")),
            ("future-cuda", lambda m: m.update(cuda_status="passed")),
        ]
        for name, mutation in mutations:
            with self.subTest(case=name):
                with tempfile.TemporaryDirectory(prefix="rz-generation-export-contract-") as temporary:
                    root, value = Path(temporary), source_receipt()
                    tiny_ascii_pins(root, value)
                    pin = value["deployment"]["manifest"]
                    path = root / pin["file"]
                    manifest = json.loads(path.read_text(encoding="ascii"))
                    mutation(manifest)
                    encoded = json.dumps(manifest, sort_keys=True, allow_nan=False).encode("ascii")
                    path.write_bytes(encoded)
                    pin["bytes"], pin["sha256"] = len(encoded), hashlib.sha256(encoded).hexdigest()
                    before = copy.deepcopy(value)
                    with self.assertRaises(ValueError):
                        self.export.verify_actual_generation_artifacts(root, value,
                            expected_sha256=self.export.provenance_digest(value))
                    self.assertEqual(value, before)

    def test_atomic_json_cannot_overwrite_existing_evidence(self):
        with tempfile.TemporaryDirectory(prefix="rz-generation-export-contract-") as temporary:
            path = Path(temporary) / "registered-contract.json"
            self.export._atomic_json(path, {"contract": "first"})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                self.export._atomic_json(path, {"contract": "second"})
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(path.with_name(path.name + ".partial").exists())

    def test_existing_partial_is_preserved_without_registration(self):
        with tempfile.TemporaryDirectory(prefix="rz-generation-export-contract-") as temporary:
            path = Path(temporary) / "registered-contract.json"
            partial = path.with_name(path.name + ".partial")
            partial.write_text("owned incomplete contract", encoding="ascii")
            with self.assertRaises(FileExistsError):
                self.export._atomic_json(path, {"contract": "new"})
            self.assertFalse(path.exists())
            self.assertEqual(partial.read_text(encoding="ascii"), "owned incomplete contract")

    def test_registration_race_preserves_prior_final_and_own_partial(self):
        with tempfile.TemporaryDirectory(prefix="rz-generation-export-contract-") as temporary:
            path = Path(temporary) / "registered-contract.json"
            original_publish = self.export.publish_noreplace

            def concurrent_registration(source, destination):
                Path(destination).write_text("concurrent contract owner", encoding="ascii")
                return original_publish(source, destination)

            with mock.patch.object(self.export, "publish_noreplace", side_effect=concurrent_registration):
                with self.assertRaises(FileExistsError):
                    self.export._atomic_json(path, {"contract": "own pending evidence"})
            self.assertEqual(path.read_text(encoding="ascii"), "concurrent contract owner")
            partial = path.with_name(path.name + ".partial")
            self.assertEqual(json.loads(partial.read_text(encoding="utf-8")), {"contract": "own pending evidence"})


class PinnedSnapshotTests(SourceOnlyTests):
    def setUp(self):
        super().setUp()
        self.export = importlib.import_module("rz_pals_model.generation_export")

    def test_ascii_snapshot_keeps_verified_a_when_source_changes_b_then_a(self):
        with tempfile.TemporaryDirectory(prefix="rz-generation-pinned-contract-") as temporary:
            root = Path(temporary)
            source = root / "contract-only-source.txt"
            original = b"A verified contract-only ASCII snapshot\n"
            substitute = b"B substituted ASCII bytes after verification\n"
            source.write_bytes(original)
            expected = hashlib.sha256(original).hexdigest()
            with self.export._pinned_checkpoint_snapshot(source, expected) as (snapshot, size):
                self.assertEqual(size, len(original))
                source.write_bytes(substitute)
                self.assertEqual(source.read_bytes(), substitute)
                self.assertEqual(snapshot.read(), original)
                source.write_bytes(original)
                snapshot.seek(0)
                self.assertEqual(snapshot.read(), original)
                self.assertFalse(snapshot.closed)
            self.assertTrue(snapshot.closed)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(list(root.glob("rz-generation-pinned-*")), [])

    def test_wrong_checkpoint_snapshot_pin_never_yields_consumer_stream(self):
        with tempfile.TemporaryDirectory(prefix="rz-generation-pinned-contract-") as temporary:
            root = Path(temporary)
            source = root / "contract-only-source.txt"
            source.write_bytes(b"owned ASCII digest mismatch fixture\n")
            consumer_entered = False
            with self.assertRaises(ValueError):
                with self.export._pinned_checkpoint_snapshot(source, "f" * 64):
                    consumer_entered = True
            self.assertFalse(consumer_entered)
            self.assertEqual(list(root.glob("rz-generation-pinned-*")), [])

    def test_growth_block_exceeding_cap_is_not_written_to_snapshot(self):
        with tempfile.TemporaryDirectory(prefix="rz-generation-pinned-contract-") as temporary:
            source = Path(temporary) / "contract-only-growing.txt"
            source.write_bytes(b"1234")
            original_fdopen = os.fdopen
            original_open = builtins.open
            snapshot = io.BytesIO()
            writes = mock.Mock(wraps=snapshot.write)

            class GrowingStream:
                def __init__(self, stream):
                    self.stream, self.read_count = stream, 0

                def __enter__(self):
                    return self

                def __exit__(self, *unused):
                    self.stream.close()

                def fileno(self):
                    return self.stream.fileno()

                def read(self, size):
                    self.read_count += 1
                    if self.read_count == 1:
                        return self.stream.read(4)
                    if self.read_count == 2:
                        with original_open(source, "ab") as writer:
                            writer.write(b"567890")
                    return self.stream.read(size)

            def growing_fdopen(descriptor, mode, *args, **kwargs):
                return GrowingStream(original_fdopen(descriptor, mode, *args, **kwargs))

            sink = mock.Mock()
            sink.write = writes
            with mock.patch.object(self.export.os, "fdopen", side_effect=growing_fdopen):
                with self.assertRaisesRegex(ValueError, "grew past registered output cap"):
                    self.export._hash_file(source, maximum=8, snapshot=sink)
            writes.assert_called_once_with(b"1234")
            self.assertEqual(snapshot.getvalue(), b"1234")
            self.assertEqual(source.read_bytes(), b"1234567890")

    def test_pinned_json_consumes_a_snapshot_without_reopening_source_b(self):
        with tempfile.TemporaryDirectory(prefix="rz-generation-pinned-contract-") as temporary:
            source = Path(temporary) / "contract-only-json.json"
            encoded = b'{"contract":"A"}'
            substitute = b'{"contract":"B"}'
            source.write_bytes(encoded)
            original_hash = self.export._hash_file
            original_open = builtins.open

            def replace_after_verified_hash(path, maximum, *, snapshot=None):
                result = original_hash(path, maximum, snapshot=snapshot)
                with original_open(path, "wb") as writer:
                    writer.write(substitute)
                return result

            with mock.patch.object(self.export, "_hash_file", side_effect=replace_after_verified_hash) as hashed, \
                 mock.patch("builtins.open", side_effect=AssertionError("pinned JSON reopened source")) as opened, \
                 mock.patch.object(Path, "open", side_effect=AssertionError("pinned JSON reopened path")) as path_open, \
                 mock.patch.object(Path, "read_bytes", side_effect=AssertionError("pinned JSON reread path")) as reread:
                result = self.export._pinned_json(source, hashlib.sha256(encoded).hexdigest(), len(encoded))
                self.assertEqual(result, {"contract": "A"})
                self.assertEqual(hashed.call_count, 1)
                for operation in (opened, path_open, reread):
                    operation.assert_not_called()
            self.assertEqual(source.read_bytes(), substitute)

    def test_pinned_json_bad_hash_or_size_is_rejected_before_decoder(self):
        with tempfile.TemporaryDirectory(prefix="rz-generation-pinned-contract-") as temporary:
            source = Path(temporary) / "contract-only-json.json"
            encoded = b'{"contract":"owned"}'
            source.write_bytes(encoded)
            digest = hashlib.sha256(encoded).hexdigest()
            for expected, size in (("f" * 64, len(encoded)), (digest, len(encoded) + 1)):
                with self.subTest(expected=expected, size=size):
                    with mock.patch.object(self.export, "strict_json", side_effect=AssertionError("unverified metadata decoded")) as decoded:
                        with self.assertRaises(ValueError):
                            self.export._pinned_json(source, expected, size)
                        decoded.assert_not_called()

    def test_raw_provenance_file_pin_keeps_semantic_and_binary_pins_distinct(self):
        with tempfile.TemporaryDirectory(prefix="rz-generation-pinned-contract-") as temporary:
            source = Path(temporary) / "actual-generation.json"
            value = source_receipt()
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
            source.write_bytes(encoded)
            expected = hashlib.sha256(encoded).hexdigest()
            before = copy.deepcopy(value)
            result = self.export.read_actual_generation_provenance(source, expected_file_sha256=expected,
                expected_spec_sha256=value["spec_sha256"], expected_checkpoint_sha256=value["checkpoint"]["sha256"])
            self.assertEqual(result, before)
            self.assertEqual(value, before)
            self.assertEqual(result["numeric_status"], "not_run")
            self.assertFalse(result["arena_accepted"])
            # The same JSON value has the same semantic digest but different raw bytes.
            reformatted = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("ascii")
            self.assertNotEqual(hashlib.sha256(reformatted).hexdigest(), expected)
            source.write_bytes(reformatted)
            with mock.patch.object(self.export, "validate_actual_generation_provenance", side_effect=AssertionError("wrong raw pin reached semantic approval")) as validate:
                with self.assertRaises(ValueError):
                    self.export.read_actual_generation_provenance(source, expected_file_sha256=expected)
                validate.assert_not_called()

    def test_externally_pinned_duplicate_json_fields_are_not_accepted(self):
        with tempfile.TemporaryDirectory(prefix="rz-generation-pinned-contract-") as temporary:
            source = Path(temporary) / "actual-generation.json"
            value = source_receipt()
            base = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            encoded = ('{"schema":"unknown",' + base[1:]).encode("ascii")
            source.write_bytes(encoded)
            with self.assertRaises(ValueError):
                self.export.read_actual_generation_provenance(source,
                    expected_file_sha256=hashlib.sha256(encoded).hexdigest())

    def test_raw_provenance_file_link_is_rejected_before_decode(self):
        with tempfile.TemporaryDirectory(prefix="rz-generation-pinned-contract-") as temporary:
            root = Path(temporary)
            owned, linked = root / "owned-contract.json", root / "actual-generation.json"
            encoded = json.dumps(source_receipt(), sort_keys=True, allow_nan=False).encode("ascii")
            owned.write_bytes(encoded)
            try:
                linked.symlink_to(owned)
            except OSError as error:
                self.skipTest("test filesystem does not support owned symlinks: " + type(error).__name__)
            with mock.patch.object(self.export, "strict_json", side_effect=AssertionError("linked metadata was decoded")) as decoded:
                with self.assertRaises((ValueError, OSError)):
                    self.export.read_actual_generation_provenance(linked,
                        expected_file_sha256=hashlib.sha256(encoded).hexdigest())
                decoded.assert_not_called()

    def test_raw_provenance_requires_registered_basename_before_file_hash(self):
        with tempfile.TemporaryDirectory(prefix="rz-generation-pinned-contract-") as temporary:
            source = Path(temporary) / "other-owned-contract.json"
            encoded = json.dumps(source_receipt(), sort_keys=True, allow_nan=False).encode("ascii")
            source.write_bytes(encoded)
            with mock.patch.object(self.export, "_pinned_json", side_effect=AssertionError("unregistered path was read")) as pinned:
                with self.assertRaises(ValueError):
                    self.export.read_actual_generation_provenance(source,
                        expected_file_sha256=hashlib.sha256(encoded).hexdigest())
                pinned.assert_not_called()


if __name__ == "__main__":
    unittest.main()
