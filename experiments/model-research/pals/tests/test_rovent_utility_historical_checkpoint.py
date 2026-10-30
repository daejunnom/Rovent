"""Strict metadata/byte fixtures; no tensor, original process or live lease."""
import copy
import hashlib
import io
import json
import stat
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rz_pals_model import rovent_utility_historical_checkpoint as history
from rz_pals_model import rovent_utility_training as training
from rz_pals_model.rovent_training_bootstrap import PinnedReadBudget
from rz_pals_model.rovent_utility_training_window import make_utility_sampler
from rz_pals_model.generation_data import digest


def expense(value=0):
    return dict.fromkeys(("forward_calls", "backward_calls", "samples", "wall_time_ns",
        "partial_accumulation_ns", "replay_ns", "masked_groups", "output_bytes"), value)


def ref(path, marker, size=10):
    return {"path": path, "sha256": marker * 32, "bytes": size}


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def prepared():
    config = {"board_width": 2, "line_width": 3, "context_width": 4, "local_width": 5,
        "hidden_width": 6, "causal_window": 3, "max_lines": 64, "max_plies": 256}
    membership = {"hidden_width": 2, "input_width": 17, "optimizer_owner": "utility-owner",
        "frozen_components": ["nnuec_base", "p", "c"],
        "wire_to_parameter": {name: "utility." + name for name in training.PARAMETER_FIELDS},
        "parameters": {name: {"checkpoint_path": "parameters." + name, "shape": shape,
            "dtype": "float32", "planned_requires_grad": True} for name, shape in
            zip(training.PARAMETER_FIELDS, ([2, 17], [2], [7, 2], [7]), strict=True)}}
    registration = {"collection_source_sha256": "10" * 32, "trainer_source_sha256": "11" * 32,
        "resource_profile_sha256": "40" * 32, "cost_epoch_sha256": "50" * 32,
        "cost_scope": "cohort_route_with_bounded_materialization_and_physical_pin_zero_excluding_durable_dataset_io_v1",
        "objective": "exclusive_wall_nanoseconds",
        "frozen_base": {"schema": training.BASE_SCHEMA, "semantics": training.BASE_SEMANTICS,
            "weights_sha256": "20" * 32, "config": config},
        "utility": {"schema": training.UTILITY_SCHEMA, "semantics": training.UTILITY_SEMANTICS,
            "initial_weights_sha256": "30" * 32, "base_weights_sha256": "20" * 32,
            "parameter_spec_sha256": training.parameter_spec_sha256(membership, config, "utility-owner")}}
    return training.prepare_cohort_training(training.make_recipe("utility-owner"), registration, membership)


def projection():
    preparation = prepared()
    base, utility = preparation["registration"]["frozen_base"], preparation["registration"]["utility"]
    metadata = {"schema": "rz-nnuec-utility-torch-metadata/1", "model_schema": utility["schema"],
        "model_semantics": utility["semantics"], "base_schema": base["schema"], "base_semantics": base["semantics"],
        "base_weights_sha256": base["weights_sha256"], "frozen_base_mapping_sha256": "21" * 32,
        "frozen_base_config": base["config"], "hidden_width": 2, "input_width": 17, "initial_training": {"kind": "untrained"}}
    sampler = make_utility_sampler(["61" * 32], "62" * 32, 31, run_id="utility-run")
    result = {"schema": "rz-rovent-utility-tensor-projection/1", "payload_state_sha256": "63" * 32,
        "preparation": preparation, "metadata": metadata, "sampler": sampler, "cuda_devices": [],
        "parameter_names": ["utility." + name for name in training.PARAMETER_FIELDS], "completed_updates": 0,
        "dispatched_updates": 0, "ancestor_updates": 0, "pending_microbatches": 0, "cost": expense(),
        "ancestor_cost": expense(), "completion_receipts": [], "ancestor_completion_sha256": [],
        "parent_checkpoint_sha256": None, "source_cost_observation": None,
        "model_state_sha256": "64" * 32, "optimizer_state_sha256": "65" * 32, "scheduler_state_sha256": "66" * 32,
        "rng_state_sha256": "67" * 32, "sampler_state_sha256": history._plain_state_sha256(sampler)}
    result["derived_training_state"] = {"schema": "rz-rovent-utility-derived-training-state/1",
        "initial_source_declaration": {"kind": "untrained"}, "current_tensor_kind": "initial_source_only",
        "completed_updates": 0, "ancestor_updates": 0, "total_updates": 0,
        "completion_chain_sha256": digest("rz-rovent-utility-derived-training-state/1", []),
        "recipe_sha256": preparation["recipe_sha256"], "registration_sha256": preparation["registration_sha256"],
        "frozen_base_weights_sha256": "20" * 32, "frozen_base_mapping_sha256": "21" * 32,
        "live_training_authority_reissued": False}
    return result


def initial():
    value = projection()
    return {"schema": "rz-rovent-utility-initial-ledger-projection/1",
        "state_admission_scope": "actual_original_child_initial_observation_not_independent_torch_recomputation",
        "states": {name: value[name] for name in history.STATE_FIELDS}, "sampler": value["sampler"],
        "completed_updates": 0, "dispatched_updates": 0, "ancestor_updates": 0, "cost": expense(), "ancestor_cost": expense(),
        "completion_receipts": [], "ancestor_completion_sha256": [], "initial_source_declaration": {"kind": "untrained"},
        "frozen_base_weights_sha256": "20" * 32, "frozen_base_mapping_sha256": "21" * 32,
        "source_checkpoint": None, "source_payload_state_sha256": None, "source_snapshot_cost_observation": None,
        "source_snapshot_cost": None, "source_snapshot_ancestor_cost": None, "checkpoint_cost": None,
        "historical_evidence_only": True, "execution_authorized": False}


def boundary():
    return {"schema": "rz-rovent-utility-original-training-closure/1", "execution_id": "utility-execution",
        "window_id": "utility-window", "run_id": "utility-run", "kind": "optimizer_boundary",
        "child_identity": {"boot_id": "old-boot", "pid": 123, "start_ticks": 456,
            "cgroup_path": "/sys/fs/cgroup/old-utility", "cgroup_device": 7, "cgroup_inode": 8},
        "process_receipt": None, "terminal_receipt": None, "boundary_publication": None,
        "child_reap": "unresolved", "physical_completion": "unknown", "historical_evidence_only": True,
        "execution_authorized": False}


def publication():
    result = {"schema": "rz-rovent-utility-checkpoint-publication/1", "execution_id": "utility-execution",
        "window_id": "utility-window", "run_id": "utility-run", "preparation_sha256": "71" * 32,
        "dataset_sha256": "72" * 32, "window_spec_sha256": "73" * 32, "trainer_source_sha256": "74" * 32,
        "trainer_python_executable_sha256": "75" * 32, "trainer_dependency_set_sha256": "76" * 32,
        "completion_receipts": [], "ancestor_completion_receipts": [], "parent_publication": None,
        "historical_evidence_only": True, "execution_authorized": False}
    for index, name in enumerate(("tensor", "retirement", "tensor_projection", "initial_ledger_state",
            "original_preparation", "original_window", "original_registration", "original_rpc_journal", "cost", "closure")):
        result[name] = ref("original/" + name + (".pt" if name == "tensor" else ".jsonl" if name == "original_rpc_journal" else ".json"), format(128 + index, "02x"))
    return result


def parse_publication(value):
    raw = encoded(value)
    return history.parse_utility_checkpoint_publication(raw, publication_sha256=hashlib.sha256(raw).hexdigest(), publication_bytes=len(raw))


class UtilityHistoricalCheckpointTests(unittest.TestCase):
    def test_original_projection_preserves_actual_model_membership_and_has_no_new_lease(self):
        value = projection(); before = copy.deepcopy(value); raw = encoded(value)
        self.assertEqual(history.parse_utility_tensor_projection(raw, expected_sha256=hashlib.sha256(raw).hexdigest(), expected_bytes=len(raw)), before)
        observed = history.project_utility_tensor_scratch(value, observed_state=before)
        self.assertEqual(value, before)
        self.assertFalse(observed["execution_authorized"])
        self.assertFalse(observed["old_run_reclamation_issued"])
        self.assertFalse(observed["past_physical_completion_reissued"])

    def test_projection_changes_to_frozen_source_parameter_order_rng_or_sampler_fail(self):
        original = projection()
        for mutate in (lambda item: item["metadata"].update(base_weights_sha256="90" * 32),
                       lambda item: item["parameter_names"].reverse(),
                       lambda item: item.update(rng_state_sha256="91" * 32),
                       lambda item: item["sampler"].update(seed=32),
                       lambda item: item.update(pending_microbatches=True)):
            altered = copy.deepcopy(original); mutate(altered)
            with self.assertRaises(ValueError): history.project_utility_tensor_scratch(original, observed_state=altered)

    def test_original_raw_sha_bytes_and_duplicate_keys_are_not_reconstructed(self):
        value = projection(); raw = encoded(value)
        with self.assertRaises(ValueError): history.parse_utility_tensor_projection(raw, expected_sha256="00" * 32, expected_bytes=len(raw))
        with self.assertRaises(ValueError): history.parse_utility_tensor_projection(raw, expected_sha256=hashlib.sha256(raw).hexdigest(), expected_bytes=len(raw) + 1)
        duplicate = b'{"schema":"one","schema":"two"}'
        with self.assertRaises(ValueError): history.parse_utility_tensor_projection(duplicate, expected_sha256=hashlib.sha256(duplicate).hexdigest(), expected_bytes=len(duplicate))

    def test_sampler_state_uses_independent_native_typed_framing(self):
        raw = b'dict["str","a"];["str","x"];["str","b"];list1["int",2];;;'
        self.assertEqual(history._plain_state_sha256({"b": [2], "a": "x"}), hashlib.sha256(raw).hexdigest())
        for value in (True, 1.0, -1, None, "한글"):
            with self.assertRaises(ValueError): history._plain_state_sha256(value)

    def test_publication_cost_and_closure_are_original_file_pins_not_inline_projection(self):
        value = publication()
        self.assertEqual(parse_publication(value), value)
        for mutate in (lambda item: item.update(closure=boundary()),
                       lambda item: item.update(cost={"status": "known"}),
                       lambda item: item.update(schema="rz-rovent-corpus-publication/2"),
                       lambda item: item.update(execution_authorized=0),
                       lambda item: item["tensor"].update(path="../outside.pt")):
            changed = publication(); mutate(changed)
            with self.assertRaises(ValueError): parse_publication(changed)

    def test_tensor_and_journal_preserve_two_gibibyte_envelopes(self):
        value = publication(); value["tensor"]["bytes"] = 2 * 1024**3; value["original_rpc_journal"]["bytes"] = 2 * 1024**3
        self.assertEqual(parse_publication(value), value)
        for name in ("tensor", "original_rpc_journal"):
            changed = copy.deepcopy(value); changed[name]["bytes"] += 1
            with self.assertRaises(ValueError): parse_publication(changed)

    def test_current_and_ancestor_full_update_chains_cannot_repeat_a_sha(self):
        value = publication(); value["completion_receipts"] = [ref("current/1.json", "91")]
        value["ancestor_completion_receipts"] = [ref("ancestor/1.json", "91")]
        with self.assertRaises(ValueError): parse_publication(value)

    def test_live_optimizer_boundary_cannot_claim_old_process_closed_or_omit_nullable_field(self):
        value = boundary(); self.assertEqual(history._closure(value), value)
        for mutate in (lambda item: item.update(physical_completion="confirmed"),
                       lambda item: item.update(child_reap="owned_child_reaped"),
                       lambda item: item.update(boundary_publication=ref("boundary.json", "92")),
                       lambda item: item.pop("boundary_publication")):
            changed = boundary(); mutate(changed)
            with self.assertRaises(ValueError): history._closure(changed)

    def test_fresh_original_observation_cannot_invent_prior_cost_or_model_work(self):
        value = initial(); self.assertEqual(history._initial(value), value)
        for mutate in (lambda item: item["cost"].update(samples=1),
                       lambda item: item["ancestor_cost"].update(wall_time_ns=1),
                       lambda item: item.update(completed_updates=True),
                       lambda item: item.update(execution_authorized=True)):
            changed = initial(); mutate(changed)
            with self.assertRaises(ValueError): history._initial(changed)

    def test_original_fork_scope_is_distinct_and_resets_only_current_work(self):
        value = initial()
        value.update(state_admission_scope="original_checkpoint_model_and_chain_with_actual_child_reset_state_observation",
            source_checkpoint=ref("/original/source.pt", "93"), source_payload_state_sha256="94" * 32,
            source_snapshot_cost=expense(), source_snapshot_ancestor_cost=expense(), ancestor_updates=1,
            ancestor_completion_sha256=["95" * 32], ancestor_cost=expense(2),
            checkpoint_cost={"schema": "rz-rovent-utility-checkpoint-cost/1", "ancestor_cost": expense(),
                "snapshot_cost": expense(), "post_snapshot_delta": expense(),
                "closure_tail": {"status": "unknown", "reason": "original_tail_unmeasured", "evidence": None},
                "inherited_unknown_tails": [], "known_lower_bound": expense(), "total_status": "lower_bound_only",
                "scope": "recorded_cost_only_not_old_process_closure_or_new_window_authority", "execution_authorized": False})
        self.assertEqual(history._initial(value), value)
        for mutate in (lambda item: item["cost"].update(output_bytes=1),
                       lambda item: item.update(dispatched_updates=1),
                       lambda item: item.update(state_admission_scope="invented_fork")):
            changed = copy.deepcopy(value); mutate(changed)
            with self.assertRaises(ValueError): history._initial(changed)

    def test_original_window_device_and_fresh_source_fields_are_not_inferred(self):
        value = projection(); window = {"device": "cpu"}; registration = {"initialization_source": {"mode": "initialize"}}
        history._projection_source_join(value, window, registration, None)
        for mutate in (lambda item: item.update(cuda_devices=[0]),
                       lambda item: item.update(parent_checkpoint_sha256="96" * 32),
                       lambda item: item.update(source_cost_observation={"status": "known"})):
            changed = copy.deepcopy(value); mutate(changed)
            with self.assertRaises(ValueError): history._projection_source_join(changed, window, registration, None)

    def test_resume_preserves_old_fork_pointer_but_fork_pins_immediate_raw_tensor(self):
        parent = {"cost": {"fixture": "original-parent-cost"},
            "publication": {"tensor": ref("original/source.pt", "97")},
            "projection": {"parent_checkpoint_sha256": "98" * 32}}
        value = projection(); value["source_cost_observation"] = copy.deepcopy(parent["cost"])
        window = {"device": "cpu"}; registration = {"initialization_source": {"mode": "resume"}}
        value["parent_checkpoint_sha256"] = "98" * 32
        history._projection_source_join(value, window, registration, parent)
        changed = copy.deepcopy(value); changed["parent_checkpoint_sha256"] = "97" * 32
        with self.assertRaises(ValueError): history._projection_source_join(changed, window, registration, parent)
        registration["initialization_source"]["mode"] = "fork"
        history._projection_source_join(changed, window, registration, parent)
        with self.assertRaises(ValueError): history._projection_source_join(value, window, registration, parent)
        changed["source_cost_observation"] = None
        with self.assertRaises(ValueError): history._projection_source_join(changed, window, registration, parent)

    def test_streamed_journal_counts_newline_inside_256_kib_event_limit(self):
        # In-memory byte/FD fixture only. It never creates an archive source,
        # dataset, original process proof, learner acknowledgement or lease.
        maximum = 256 * 1024
        for extra, accepted in ((0, True), (1, False)):
            prefix, suffix = b'{"padding":"', b'"}\n'
            raw = prefix + b'x' * (maximum + extra - len(prefix) - len(suffix)) + suffix
            pin = {"path": "/fixture/journal.jsonl", "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            archive = history._Archive.__new__(history._Archive)
            archive.reads = PinnedReadBudget(len(raw) + 1, until_ns=10, clock=lambda: 1)
            archive.absolute = lambda reference, maximum: pin
            archive._root = lambda: None
            stream = io.BytesIO(raw)
            info = SimpleNamespace(st_dev=1, st_ino=2, st_mode=stat.S_IFREG | 0o600, st_nlink=1,
                st_size=len(raw), st_mtime_ns=3, st_ctime_ns=4)
            with patch.object(history, "_open_file", return_value=99), patch.object(history.os, "fstat", return_value=info), \
                    patch.object(history.os, "read", side_effect=lambda _fd, count: stream.read(count)), patch.object(history.os, "close"):
                if accepted:
                    self.assertEqual(len(list(archive.journal({}))), 1)
                    self.assertFalse(archive.reads.observation()["unknown"])
                else:
                    with self.assertRaises(ValueError): list(archive.journal({}))
                    self.assertTrue(archive.reads.observation()["unknown"])


if __name__ == "__main__":
    unittest.main()
