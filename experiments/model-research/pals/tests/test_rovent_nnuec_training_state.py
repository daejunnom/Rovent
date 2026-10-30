"""Pure lifecycle metadata fixtures; no optimizer/model or actual run evidence."""
import copy
import hashlib
import importlib.abc
import json
import sys
import unittest


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("neural imports forbidden in NNUEC metadata contract tests")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_nnuec_training_state as lifecycle
finally:
    sys.meta_path.remove(_guard)


def h(value):
    return hashlib.sha256(value.encode()).hexdigest()


def canonical(domain, value):
    return hashlib.sha256(json.dumps([domain, value], sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode()).hexdigest()


HEADS = ("policy", "wdl", "divergence", "task", "nnuec_wdl")


def journal(events=()):
    """Original-shaped synthetic JSON, never an original learner capability."""
    phase = lambda name: {"corpus_sha256": h(name + "corpus"), "nnuec_model_sha256": h("source-model"),
        "model_epoch": 1, "context_epoch": 1, "samplers": {}}
    value = {"schema": "rz-rovent-original-learner-journal/1", "recipe_sha256": h("recipe"), "stage_id": "common",
        "profile_sha256": h("profile"), "phase_inputs": {"pc": phase("pc"), "v": phase("v")},
        "updates": [], "head_counts": dict.fromkeys(HEADS, 0), "frozen_pc_sha256": h("frozen-pc"), "resume_cost_receipts": []}
    roles = dict.fromkeys(("proposer", "critic", "validator"), 0)
    for sequence, (role, nnuec_rows) in enumerate(events, 1):
        roles[role] += 1
        selected_phase = "v" if role == "validator" else "pc"
        heads = dict.fromkeys(HEADS, 0)
        heads["task" if role == "validator" else "policy"] = 2 if role == "validator" else 8
        heads["nnuec_wdl"] = nnuec_rows
        event = {"schema": "rz-rovent-original-learner-update/1", "window_id": "window-" + str(sequence),
            "stage_id": value["stage_id"], "role": role, "completed_role_updates": roles[role],
            "profile_sha256": value["profile_sha256"], "corpus_sha256": value["phase_inputs"][selected_phase]["corpus_sha256"],
            "phase": selected_phase, "sample_origins": [h("origin-" + str(index)) for index in range(8)],
            "sample_inputs": [h("input-" + str(index)) for index in range(8)], "head_counts": heads,
            "actual_model_state_sha256": h("actual-state-" + str(sequence)),
            "original_completion": {"sha256": h("synthetic-receipt-" + str(sequence)), "bytes": 128}}
        value["updates"].append(event)
        for head, count in heads.items():
            value["head_counts"][head] += count
    return value


def derive(*, counts=None, sampler=None, source=None, ancestor=None):
    return lifecycle.derive_nnuec_tensor_training_state(
        source_declaration={"kind": "untrained"} if source is None else source,
        parameter_updates={"nnuec.board_weights": 0, "nnuec.output_weights": 0} if counts is None else counts,
        samplers={"seed": 23} if sampler is None else sampler, ancestor=ancestor)


class NnuecTensorTrainingStateTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_pristine_source_and_inputs_remain_separate_and_are_copied(self):
        source, counts, sampler = {"kind": "untrained"}, {"nnuec.board_weights": 0}, {"seed": 23}
        state = derive(source=source, counts=counts, sampler=sampler)
        source["kind"], counts["nnuec.board_weights"], sampler["seed"] = "changed", 1, 99
        self.assertEqual(state["source_declaration"], {"kind": "untrained"})
        self.assertEqual(state["source_declaration_scope"], "initial_source_asset_only")
        self.assertEqual(state["current"]["nnuec_parameter_update_counts"], {"nnuec.board_weights": 0})
        self.assertEqual(state["current"]["valid_update_groups"], 0)
        self.assertIsNone(state["current"]["journal_sha256"])
        self.assertTrue(lifecycle.nnuec_is_unmodified(state))
        self.assertEqual(lifecycle.nnuec_known_updates(state), 0)
        self.assertFalse(state["actual_training_authority_verified"])

    def test_only_pc_events_with_nnuec_targets_count_and_inactive_parameters_may_lag(self):
        sampler = journal((("proposer", 8), ("critic", 0), ("validator", 0)))
        original = copy.deepcopy(sampler)
        state = derive(counts={"nnuec.board_weights": 1, "nnuec.pair_parent": 0}, sampler=sampler)
        self.assertEqual(state["current"]["valid_update_groups"], 1)
        self.assertEqual(state["current"]["journal_sha256"], canonical("rz-rovent-original-learner-journal/1", sampler))
        self.assertEqual(state["classification"], "modified_requires_original_authority")
        self.assertEqual(lifecycle.nnuec_known_updates(state), 1)
        self.assertFalse(lifecycle.nnuec_is_unmodified(state))
        self.assertFalse(state["actual_training_authority_verified"])
        self.assertEqual(sampler, original)

    def test_parameter_maximum_equals_valid_groups_without_requiring_equal_minimum(self):
        state = derive(counts={"nnuec.board_weights": 2, "nnuec.output_weights": 2, "nnuec.pair_projection": 0},
                       sampler=journal((("proposer", 8), ("critic", 8))))
        self.assertEqual(state["current"]["valid_update_groups"], 2)
        self.assertEqual(state["known_updates"], 2)
        self.assertEqual(state["current"]["nnuec_parameter_update_counts"]["nnuec.pair_projection"], 0)

    def test_step_drift_is_rejected_at_the_journal_group_comparison(self):
        for count in (0, 2):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "valid journal update groups differ from maximum parameter steps"):
                    derive(counts={"nnuec.board_weights": count}, sampler=journal((("proposer", 8),)))

    def test_synthetic_updates_remain_unknown_and_fork_cannot_relabel_them_untrained(self):
        state = derive(counts={"nnuec.board_weights": 3})
        self.assertEqual(state["classification"], "modified_unverified")
        self.assertIsNone(state["current"]["valid_update_groups"])
        self.assertIsNone(lifecycle.nnuec_known_updates(state))
        self.assertFalse(lifecycle.nnuec_is_unmodified(state))
        parent = lifecycle.fork_nnuec_ancestor(state, h("checkpoint"))
        self.assertEqual(parent["parent_state_sha256"], canonical(lifecycle.STATE_DOMAIN, state))
        self.assertNotIn("current", parent)
        self.assertNotIn("ancestor", parent)
        self.assertNotIn("source_declaration", parent)
        child = derive(ancestor=parent)
        self.assertEqual(child["classification"], "modified_unverified")
        self.assertIsNone(child["known_updates"])
        self.assertFalse(lifecycle.nnuec_is_unmodified(child))

    def test_fork_resets_current_counts_but_retains_known_parent_history(self):
        parent = derive(counts={"nnuec.board_weights": 1}, sampler=journal((("proposer", 8),)))
        compact = lifecycle.fork_nnuec_ancestor(parent, h("checkpoint"))
        child = derive(ancestor=compact)
        self.assertEqual(child["current"]["valid_update_groups"], 0)
        self.assertEqual(child["known_updates"], 1)
        self.assertEqual(child["classification"], "modified_requires_original_authority")
        self.assertFalse(lifecycle.nnuec_is_unmodified(child))
        next_child = derive(counts={"nnuec.output_weights": 1}, sampler=journal((("critic", 8),)), ancestor=compact)
        self.assertEqual(next_child["known_updates"], 2)

    def test_learned_source_declaration_is_not_added_to_tracked_actual_updates(self):
        source = {"kind": "learned", "run_id": "initial-source", "steps": 999, "dataset_sha256": h("dataset")}
        state = derive(source=source)
        self.assertEqual(state["source_declaration"], source)
        self.assertTrue(lifecycle.nnuec_is_unmodified(state))
        self.assertEqual(state["known_updates"], 0)
        self.assertFalse(state["actual_training_authority_verified"])

    def test_fork_binds_the_exact_initial_source_declaration(self):
        compact = lifecycle.fork_nnuec_ancestor(derive(), h("checkpoint"))
        different = {"kind": "learned", "run_id": "replacement", "steps": 1, "dataset_sha256": h("dataset")}
        with self.assertRaisesRegex(ValueError, "cannot change its immutable initial source declaration"):
            derive(source=different, ancestor=compact)

    def test_compact_ancestor_rejects_recursive_state_and_authority_claims(self):
        compact = lifecycle.fork_nnuec_ancestor(derive(), h("checkpoint"))
        cases = (("ancestor", {}, "fields mismatch"),
                 ("actual_training_authority_verified", True, "cannot claim actual training authority"),
                 ("parent_checkpoint_sha256", "A" * 64, "lowercase NNUEC SHA-256"),
                 ("known_updates", True, "zero tracked updates"))
        for field, value, message in cases:
            with self.subTest(field=field):
                malformed = {**compact, field: value}
                with self.assertRaisesRegex(ValueError, message):
                    lifecycle.validate_nnuec_ancestor(malformed)

    def test_journal_receipt_and_role_errors_do_not_fall_back_to_unverified_success(self):
        cases = ((lambda value: value["updates"][0]["original_completion"].update(bytes=0), "nonempty bounded"),
                 (lambda value: value["updates"][0]["original_completion"].update(sha256="A" * 64), "lowercase NNUEC SHA-256"),
                 (lambda value: value["updates"][0]["head_counts"].update(nnuec_wdl=True), "bounded NNUEC unsigned integer"),
                 (lambda value: value["updates"][0].update(role="validator"), "event role/phase/order/source mismatch"))
        for mutate, message in cases:
            with self.subTest(message=message):
                sampler = journal((("proposer", 8),))
                mutate(sampler)
                with self.assertRaisesRegex(ValueError, message):
                    derive(counts={"nnuec.board_weights": 1}, sampler=sampler)

    def test_aggregate_overflow_and_classification_tampering_are_rejected(self):
        compact = lifecycle.fork_nnuec_ancestor(
            derive(counts={"nnuec.board_weights": 1}, sampler=journal((("proposer", 8),))), h("checkpoint"))
        compact["known_updates"] = 2**64 - 1
        with self.assertRaisesRegex(ValueError, "tracked update total exceeds uint64"):
            derive(counts={"nnuec.board_weights": 1}, sampler=journal((("proposer", 8),)), ancestor=compact)
        forged = derive(counts={"nnuec.board_weights": 1})
        forged.update(classification="unmodified_from_source", known_updates=0)
        with self.assertRaisesRegex(ValueError, "classification disagrees with current/ancestor history"):
            lifecycle.nnuec_is_unmodified(forged)

    def test_namespace_types_and_json_cycles_are_rejected_without_neural_execution(self):
        cases = (({"proposer.weight": 1}, "explicit nnuec parameter namespace"),
                 ({"nnuec.weight": True}, "bounded NNUEC unsigned integer"),
                 ({"nnuec.weight": -1}, "bounded NNUEC unsigned integer"))
        for counts, message in cases:
            with self.subTest(counts=counts), self.assertRaisesRegex(ValueError, message):
                derive(counts=counts)
        cyclic = {}
        cyclic["self"] = cyclic
        with self.assertRaisesRegex(ValueError, "container/cycle budget"):
            derive(sampler=cyclic)


if __name__ == "__main__":
    unittest.main()
