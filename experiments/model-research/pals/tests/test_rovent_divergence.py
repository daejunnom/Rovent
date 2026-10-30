"""Attack-slot target shape only; never actual Repair/native evidence."""
import copy
import hashlib
import importlib.abc
import sys
import unittest


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("divergence contracts cannot import neural runtimes")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_divergence as divergence
    from rz_pals_model import rovent_submitted_input as submitted
    from rz_pals_model import generation_data as legacy
    from test_rovent_submitted_input import fixture as input_fixture
finally:
    sys.meta_path.remove(_guard)


def h(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fixture(*, visible=False):
    snapshot = input_fixture(selected_counter=visible)
    snapshot["question_id"] = h("original submitted question")
    label = {"schema": divergence.TARGET_DOMAIN, "observed_sequence": snapshot["capture_sequence"] + 10,
        "question_id": snapshot["question_id"], "view_sha256": snapshot["view_sha256"],
        "raw_evidence_sha256": h("original raw witness"), "source": divergence.SOURCE,
        "slot": 0, "attack_ply": 1, "counter_question_id": snapshot["question_id"] if visible else h("future counter question"),
        "counter_native_view_sha256": h("distinct original native counter view"), "counter_local_index": 0,
        "counter_revision": 23, "repair_question_id": h("actual repair question"), "status": "supported_after_repair"}
    return snapshot, label


class SubmittedDivergenceTests(unittest.TestCase):
    def setUp(self):
        sys.meta_path.insert(0, _guard)

    def tearDown(self):
        sys.meta_path.remove(_guard)

    def test_future_target_never_modifies_the_past_input_or_inserts_a_counter(self):
        snapshot, label = fixture()
        before, seal = copy.deepcopy(snapshot), submitted.seal_input(snapshot)
        self.assertEqual(divergence.divergence_target(snapshot, label), 1.0)
        self.assertEqual(divergence.divergence_target(snapshot, {**label, "status": "refuted_by_repair"}), 0.0)
        self.assertEqual(snapshot, before)
        self.assertEqual(submitted.seal_input(snapshot), seal)
        self.assertEqual(snapshot["selected_view"]["records"], [])
        self.assertIsNone(snapshot["attack_mapping"][0]["selected_counter_local_index"])

    def test_visible_counter_requires_its_sealed_question_index_and_revision(self):
        snapshot, label = fixture(visible=True)
        self.assertEqual(divergence.divergence_target(snapshot, label), 1.0)
        for change in ({"counter_question_id": h("foreign view")}, {"counter_local_index": 1}, {"counter_revision": 24}):
            with self.subTest(change=tuple(change)), self.assertRaisesRegex(ValueError, "visible source counter"):
                divergence.divergence_target(snapshot, {**label, **change})

    def test_target_slot_and_attack_position_are_the_original_head_candidate(self):
        snapshot, label = fixture()
        for change in ({"slot": 1}, {"slot": True}, {"attack_ply": 2}, {"attack_ply": -1}):
            with self.subTest(change=tuple(change)), self.assertRaises(ValueError):
                divergence.divergence_target(snapshot, {**label, **change})

    def test_source_question_view_and_collection_sequence_are_exact(self):
        snapshot, label = fixture()
        for change in ({"question_id": h("other question")}, {"view_sha256": h("other view")},
                       {"observed_sequence": snapshot["capture_sequence"] - 1}, {"observed_sequence": True},
                       {"source": "teacher_cp"}, {"schema": "old-or-unregistered"}):
            with self.subTest(change=tuple(change)), self.assertRaises(ValueError):
                divergence.divergence_target(snapshot, {**label, **change})

    def test_counter_namespace_and_repair_question_require_original_sha_seals(self):
        snapshot, label = fixture()
        for name in ("raw_evidence_sha256", "counter_question_id", "counter_native_view_sha256", "repair_question_id"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                divergence.divergence_target(snapshot, {**label, name: "unsealed-public-trace"})
        for change in ({"counter_local_index": 64}, {"counter_revision": 0}, {"counter_revision": 2**64}):
            with self.subTest(change=tuple(change)), self.assertRaises(ValueError):
                divergence.divergence_target(snapshot, {**label, **change})

    def test_unknown_disputed_and_teacher_agreement_never_become_known_targets(self):
        snapshot, label = fixture()
        for status in ("unknown", "disputed", "teacher_agreement", "defer"):
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "cannot become a target"):
                divergence.divergence_target(snapshot, {**label, "status": status})

    def test_legacy_target_and_new_target_fields_are_not_interchangeable(self):
        snapshot, label = fixture(visible=True)
        with self.assertRaises(ValueError):
            legacy.divergence_target(snapshot, label)
        bad = {**label, "counter_record_id": "manufactured-visible-id"}
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            divergence.divergence_target(snapshot, bad)


if __name__ == "__main__":
    unittest.main()
