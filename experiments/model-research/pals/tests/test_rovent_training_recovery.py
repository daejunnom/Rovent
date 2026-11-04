"""Pure codec operands; no files, tensor, old authority or trainer are run."""
import copy
import sys
import unittest

from test_rovent_training import NeuralImportsForbidden

_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_training_recovery as recovery
finally:
    sys.meta_path.remove(_guard)


def counters(value=0):
    return dict.fromkeys(recovery.COST_FIELDS, value)


def pin(path, digit="1"):
    return {"path": path, "sha256": digit * 64, "bytes": 11}


def envelope():
    return {"schema": recovery.RECOVERED_CHECKPOINT_DOMAIN,
        "base": {"schema": recovery.BASE_DOMAIN, "artifact": pin("tensors/base.pt", "1")},
        "checkpoint_publication": pin("checkpoint/publication.json", "2"),
        "dataset_publication": pin("dataset/publication.json", "3"),
        "ancestor_dataset_publications": [pin("dataset-parent/publication.json", "4")],
        "update_chain": pin("training/update-chain.json", "5"),
        "reclamation_evidence": pin("reclamation/old-run-closed.json", "6"),
        "cost_state": recovery.initial_recovered_cost(counters(10), {"status": "unknown", "reason": "tail_not_recovered", "evidence": None}),
        "source_enrollment_scope": "historical_exact_pins_new_window_authority_required",
        "past_native_authority_reissued": False, "execution_authorized": False}


class DurableRecoveryMetadataTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_unknown_tail_is_preserved_when_new_known_window_cost_is_added(self):
        initial = recovery.initial_recovered_cost(counters(10), {"status": "unknown", "reason": "unmeasured_after_snapshot", "evidence": None})
        before = copy.deepcopy(initial)
        advanced = recovery.append_new_window_cost(initial, execution_id="new-execution", window_id="new-window", delta=counters(3), evidence=pin("new/receipt.json"))
        self.assertEqual(initial, before)
        self.assertEqual(advanced["known_lower_bound"], counters(13))
        self.assertEqual(advanced["total_status"], "lower_bound_only")
        self.assertEqual(advanced["historical_tail_cost"], initial["historical_tail_cost"])
        self.assertFalse(advanced["execution_authorized"])

    def test_known_tail_requires_evidence_and_is_not_a_second_snapshot(self):
        value = recovery.initial_recovered_cost(counters(10), {"status": "known", "delta": counters(2), "evidence": pin("old/end.json")})
        self.assertEqual(value["known_lower_bound"], counters(12))
        self.assertEqual(value["total_status"], "recorded_scope_known")
        with self.assertRaisesRegex(ValueError, "artifact reference"):
            recovery.initial_recovered_cost(counters(), {"status": "known", "delta": counters(), "evidence": None})

    def test_duplicate_window_or_unknown_as_zero_relabel_is_rejected(self):
        value = recovery.append_new_window_cost(envelope()["cost_state"], execution_id="execution", window_id="one", delta=counters(1), evidence=pin("one.json"))
        with self.assertRaisesRegex(ValueError, "charged twice"):
            recovery.append_new_window_cost(value, execution_id="execution", window_id="one", delta=counters(1), evidence=pin("two.json"))
        changed = copy.deepcopy(value)
        changed["total_status"] = "recorded_scope_known"
        with self.assertRaisesRegex(ValueError, "uncertainty"):
            recovery.validate_recovered_cost(changed)

    def test_bool_negative_and_overflow_cannot_be_cost_counts(self):
        for bad in (True, -1, 2**64):
            current = counters()
            current["samples"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                recovery.initial_recovered_cost(current, {"status": "unknown", "reason": "unmeasured", "evidence": None})
        value = recovery.initial_recovered_cost(counters(2**64 - 1), {"status": "unknown", "reason": "unmeasured", "evidence": None})
        with self.assertRaises(ValueError):
            recovery.append_new_window_cost(value, execution_id="execution", window_id="overflow", delta=counters(1), evidence=pin("overflow.json"))

    def test_cost_delta_identity_includes_execution_without_erasing_unknown_tail(self):
        first = recovery.append_new_window_cost(envelope()["cost_state"], execution_id="execution-a",
            window_id="window-one", delta=counters(2), evidence=pin("a/receipt.json"))
        second = recovery.append_new_window_cost(first, execution_id="execution-b",
            window_id="window-one", delta=counters(3), evidence=pin("b/receipt.json"))
        self.assertEqual(second["known_lower_bound"], counters(15))
        self.assertEqual(second["total_status"], "lower_bound_only")
        self.assertEqual(len(recovery.validate_recovered_cost(second)["new_window_deltas"]), 2)

    def test_new_known_snapshot_cannot_clear_inherited_unknown_publication_tail(self):
        inherited = [{"checkpoint_publication_sha256": "a" * 64,
            "execution_id": "old-execution", "window_id": "old-window",
            "tail": {"status": "unknown", "reason": "old_tail_not_observed", "evidence": None}}]
        state = recovery.initial_recovered_cost(counters(20),
            {"status": "known", "delta": counters(2), "evidence": pin("latest/end.json")},
            inherited_unknown_tails=inherited)
        self.assertEqual(state["known_lower_bound"], counters(22))
        self.assertEqual(state["total_status"], "lower_bound_only")
        advanced = recovery.append_new_window_cost(state, execution_id="new-execution", window_id="new-window",
            delta=counters(3), evidence=pin("latest/next.json"))
        self.assertEqual(advanced["known_lower_bound"], counters(25))
        self.assertEqual(advanced["inherited_unknown_tails"], inherited)
        self.assertEqual(advanced["total_status"], "lower_bound_only")
        self.assertFalse(advanced["past_native_authority_reissued"])

    def test_inherited_unknowns_reject_duplicate_publication_relabel_and_known_cost(self):
        row = {"checkpoint_publication_sha256": "a" * 64,
            "execution_id": "old-execution", "window_id": "old-window",
            "tail": {"status": "unknown", "reason": "old_tail_not_observed", "evidence": None}}
        variants = [[row, copy.deepcopy(row)], [row, {**row, "window_id": "different-window"}],
            [{**row, "tail": {"status": "known", "delta": counters(0), "evidence": pin("old/end.json")}}]]
        for inherited in variants:
            with self.subTest(inherited=inherited), self.assertRaises(ValueError):
                recovery.initial_recovered_cost(counters(),
                    {"status": "unknown", "reason": "selected_tail_unknown", "evidence": None},
                    inherited_unknown_tails=inherited)

    def test_recovered_codec_preserves_base_pin_and_does_not_issue_live_authority(self):
        value = envelope()
        before = copy.deepcopy(value)
        checked = recovery.validate_recovered_checkpoint(value)
        self.assertEqual(value, before)
        self.assertEqual(checked["base"], value["base"])
        self.assertFalse(checked["past_native_authority_reissued"])
        self.assertFalse(checked["execution_authorized"])
        changed = copy.deepcopy(value)
        changed["base"]["artifact"]["sha256"] = "a" * 64
        self.assertNotEqual(recovery.recovered_checkpoint_digest(value), recovery.recovered_checkpoint_digest(changed))

    def test_live_authority_flags_or_legacy_auto_conversion_are_rejected(self):
        for key in ("past_native_authority_reissued", "execution_authorized"):
            value = envelope()
            value[key] = True
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "restored live authority"):
                recovery.validate_recovered_checkpoint(value)
        value = envelope()
        value["base"]["schema"] = "rz-pals-nonzero-diagnostic-checkpoint/1"
        with self.assertRaisesRegex(ValueError, "automatic conversion"):
            recovery.validate_recovered_checkpoint(value)

    def test_publication_refs_are_relative_and_aliases_cannot_replace_a_chain(self):
        for path in ("/old/absolute.json", "C:/old.json", "../escape.json", "x\\y.json", "x//y.json", "x\x7fy.json", "x\x85y.json"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "relative"):
                recovery.artifact_reference(pin(path))
        value = envelope()
        value["update_chain"]["path"] = value["dataset_publication"]["path"]
        with self.assertRaisesRegex(ValueError, "alias"):
            recovery.validate_recovered_checkpoint(value)


if __name__ == "__main__":
    unittest.main()
