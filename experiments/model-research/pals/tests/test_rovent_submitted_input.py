"""Submitted attack-slot wire contracts; no Rules or neural execution."""
import copy
import hashlib
import importlib.abc
import json
import struct
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("neural imports forbidden in submitted-input wire tests")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import generation_data as legacy
    from rz_pals_model import rovent_submitted_input as submitted
    from rz_pals_model.virtual_review import input_digest
    from test_generation_contract import row as legacy_fixture
finally:
    sys.meta_path.remove(_guard)


def components(move):
    return [move & 63, (move >> 6) & 63, move >> 12]


def fixture(*, role="critic", proposal_length=3, selected_counter=False):
    """Synthetic shape only; actual eight features and Rules belong to Rust."""
    value = legacy_fixture(role)["input"]
    del value["divergence_record_ids"]
    value.update(schema="rz-rovent-submitted-input/1", attack_mapping=[], action_support_mask=[False] * 7)
    view, payload = value["selected_view"], value["encoded_model_input"]
    if role == "critic":
        proposal = [64 if index % 2 == 0 else 128 for index in range(proposal_length)]
        view["query"].update(purpose="divergence_policy", proposal=proposal)
        payload["query"].update(purpose="divergence_policy", proposal=[components(move) for move in proposal])
        value["attack_mapping"] = [{"slot": 0, "attack_ply": 1,
            "selected_counter_local_index": None, "selected_counter_revision": None}]
        payload["divergences"] = [[1 / 256.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    if selected_counter:
        counter = [192, 256]
        view["records"] = [{"record_id": "existing-counter", "kind": "counter", "anchor": 0,
            "moves": counter, "parent": None, "supersedes": None, "observed_sequence": value["capture_sequence"] - 1,
            "perspective": -1, "completed_iteration": True, "rules_terminal": False}]
        payload["records"] = [{"kind": "counterexample", "anchor": 0, "line": [components(move) for move in counter],
            "parent": None, "supersedes": None, "perspective": -1, "completed_iteration": True, "rules_terminal": False}]
        view["query"].update(counter_present=True, counter=counter)
        payload["query"].update(counter_present=True, counter=[components(move) for move in counter])
        value["attack_mapping"][0].update(selected_counter_local_index=0, selected_counter_revision=23)
    view["length_bucket"] = next(bucket for bucket in (8, 16, 32, 64, 128, 256) if proposal_length <= bucket)
    value["view_sha256"] = legacy.view_digest(view)
    return value


def reseal_view(value):
    value["view_sha256"] = legacy.view_digest(value["selected_view"])


class SubmittedInputTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_fresh_attack_slot_does_not_require_or_invent_a_future_counter(self):
        value = fixture()
        before = copy.deepcopy(value)
        self.assertIs(submitted.validate_input(value), value)
        submitted.seal_input(value)
        self.assertEqual(value, before)
        self.assertEqual(value["selected_view"]["records"], [])
        self.assertEqual(value["selected_view"]["query"]["counter"], [])
        self.assertIsNone(value["attack_mapping"][0]["selected_counter_local_index"])

    def test_old_and_new_input_codecs_reject_each_others_fields_without_conversion(self):
        old = legacy_fixture("critic")["input"]
        before = copy.deepcopy(old)
        legacy.seal_input(old)
        with self.assertRaisesRegex(ValueError, "Rovent submitted input fields mismatch"):
            submitted.validate_input(old)
        self.assertEqual(old, before)
        value = fixture()
        with self.assertRaisesRegex(ValueError, "generation input fields mismatch"):
            legacy.validate_input(value)

    def test_input_seal_uses_new_domain_and_independent_fp32_projection(self):
        value = fixture()
        value["encoded_model_input"]["anchors"][0]["metadata"][0] = 1e-7
        projected = copy.deepcopy(value)
        payload = projected["encoded_model_input"]
        for anchor in payload["anchors"]:
            anchor["metadata"] = [struct.pack(">f", float(number)).hex() for number in anchor["metadata"]]
        payload["divergences"] = [[struct.pack(">f", float(number)).hex() for number in row] for row in payload["divergences"]]
        raw = json.dumps(["rz-rovent-submitted-input/1", projected], sort_keys=True,
            ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        before, numeric_digest = copy.deepcopy(value), input_digest(value["encoded_model_input"])
        self.assertEqual(submitted.seal_input(value), hashlib.sha256(raw).hexdigest())
        self.assertEqual(value, before)
        self.assertEqual(input_digest(value["encoded_model_input"]), numeric_digest)

    def test_proposer_without_attack_rows_retains_the_original_model_payload(self):
        value = fixture(role="proposer")
        payload = copy.deepcopy(value["encoded_model_input"])
        submitted.validate_input(value)
        self.assertEqual(value["attack_mapping"], [])
        self.assertEqual(value["encoded_model_input"], payload)

    def test_validator_and_v_action_support_are_rejected_in_new_rovent_domain(self):
        value = fixture(role="validator")
        with self.assertRaisesRegex(ValueError, "requires P/C role"):
            submitted.validate_input(value)
        value = fixture()
        value["action_support_mask"][0] = True
        with self.assertRaisesRegex(ValueError, "entirely false"):
            submitted.validate_input(value)

    def test_attack_mapping_is_a_closed_parallel_slot_order(self):
        for mutation in ("missing", "order", "extra", "bool"):
            with self.subTest(mutation=mutation):
                value = fixture()
                if mutation == "missing":
                    value["attack_mapping"] = []
                elif mutation == "order":
                    value["attack_mapping"][0]["slot"] = 1
                elif mutation == "extra":
                    value["attack_mapping"][0]["future_counter"] = "not-an-input"
                else:
                    value["attack_mapping"][0]["slot"] = False
                with self.assertRaisesRegex(ValueError, "slot/feature ordering|0-based order|attack slot fields mismatch|bounded unsigned"):
                    submitted.validate_input(value)

    def test_attack_ply_uses_proposal_only_and_stays_strictly_inside_it(self):
        for ply in (0, 3, 256, -1, True):
            with self.subTest(ply=ply):
                value = fixture()
                value["attack_mapping"][0]["attack_ply"] = ply
                with self.assertRaisesRegex(ValueError, "interior proposal-relative ply|bounded unsigned"):
                    submitted.validate_input(value)

    def test_last_interior_ply_255_is_supported_without_a_new_length_abi(self):
        value = fixture(proposal_length=256)
        value["attack_mapping"][0]["attack_ply"] = 255
        value["encoded_model_input"]["divergences"][0][0] = 255 / 256.0
        submitted.validate_input(value)
        self.assertEqual(value["selected_view"]["length_bucket"], 256)

    def test_duplicate_attack_plies_are_rejected_even_when_slots_are_distinct(self):
        value = fixture()
        value["attack_mapping"].append({**value["attack_mapping"][0], "slot": 1})
        value["encoded_model_input"]["divergences"].append(list(value["encoded_model_input"]["divergences"][0]))
        with self.assertRaisesRegex(ValueError, "unique interior proposal-relative ply"):
            submitted.validate_input(value)

    def test_first_feature_compares_fp32_bits_without_recomputing_rules_features(self):
        value = fixture()
        value["encoded_model_input"]["divergences"][0][0] = (1 / 256.0) + 1e-12
        value["encoded_model_input"]["divergences"][0][2:] = [13.0, -7.0, 0.25, 0.0, 1.5, -2.0]
        submitted.validate_input(value)
        value["encoded_model_input"]["divergences"][0][0] = 2 / 256.0
        with self.assertRaisesRegex(ValueError, "first FP32 feature"):
            submitted.validate_input(value)

    def test_nonfinite_or_unrepresentable_payload_never_receives_a_seal(self):
        value = fixture()
        value["encoded_model_input"]["divergences"][0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "divergence eight-field capacity/finite"):
            submitted.seal_input(value)
        value = fixture()
        value["encoded_model_input"]["anchors"][0]["metadata"][0] = 1e100
        with self.assertRaisesRegex(ValueError, "anchor metadata requires 16 finite Rules fields"):
            submitted.seal_input(value)

    def test_existing_selected_counter_is_bound_by_local_index_and_positive_revision(self):
        value = fixture(selected_counter=True)
        before = copy.deepcopy(value)
        submitted.validate_input(value)
        self.assertEqual(value, before)
        self.assertEqual(value["attack_mapping"][0]["selected_counter_revision"], 23)

    def test_counter_index_revision_are_a_null_pair_with_exact_integer_types(self):
        for index, revision in ((0, None), (None, 23), (0, 0), (False, 23), (0, True), (0, 2**64)):
            with self.subTest(index=index, revision=revision):
                value = fixture(selected_counter=True)
                value["attack_mapping"][0].update(selected_counter_local_index=index, selected_counter_revision=revision)
                with self.assertRaisesRegex(ValueError, "null pair|positive|bounded unsigned"):
                    submitted.validate_input(value)

    def test_selected_counter_mapping_requires_existing_kind_and_exact_query_counter(self):
        for mutation in ("absent", "kind", "counter_missing", "line"):
            with self.subTest(mutation=mutation):
                value = fixture(selected_counter=True)
                if mutation == "absent":
                    value["attack_mapping"][0]["selected_counter_local_index"] = 1
                elif mutation == "kind":
                    value["selected_view"]["records"][0]["kind"] = "proposal"
                    value["encoded_model_input"]["records"][0]["kind"] = "proposal"
                elif mutation == "counter_missing":
                    value["selected_view"]["query"].update(counter_present=False, counter=[])
                    value["encoded_model_input"]["query"].update(counter_present=False, counter=[])
                else:
                    value["selected_view"]["query"]["counter"] = [320]
                    value["encoded_model_input"]["query"]["counter"] = [components(320)]
                reseal_view(value)
                with self.assertRaisesRegex(ValueError, "counter must already exist in the exact selected view/query"):
                    submitted.validate_input(value)

    def test_future_counter_cannot_be_inserted_into_the_original_submitted_view(self):
        value = fixture(selected_counter=True)
        value["selected_view"]["records"][0]["observed_sequence"] = value["capture_sequence"] + 1
        reseal_view(value)
        with self.assertRaisesRegex(ValueError, "future observation leaked"):
            submitted.validate_input(value)

    def test_original_query_encoder_and_legal_candidate_joins_are_still_enforced(self):
        for mutation, error in (("query", "encoded query changed"), ("candidate", "candidate allocation mismatch")):
            with self.subTest(mutation=mutation):
                value = fixture()
                if mutation == "query":
                    value["encoded_model_input"]["query"]["proposal"] = []
                else:
                    value["encoded_model_input"]["candidates"].reverse()
                with self.assertRaisesRegex(ValueError, error):
                    submitted.validate_input(value)

    def test_bad_input_or_role_is_rejected_before_tensor_module_import_or_model_access(self):
        class ModelMustNotBeUsed:
            def parameters(self):
                raise AssertionError("invalid submitted input reached model")

        with patch.dict(sys.modules, {"rz_pals_model.model": None}):
            value = fixture()
            value["attack_mapping"] = []
            with self.assertRaisesRegex(ValueError, "slot/feature ordering"):
                submitted.submitted_collate_and_forward(ModelMustNotBeUsed(), "critic", value)
            with self.assertRaisesRegex(ValueError, "model role differs"):
                submitted.submitted_collate_and_forward(ModelMustNotBeUsed(), "proposer", fixture())

    def test_collator_port_passes_only_unchanged_encoded_payload(self):
        """Test doubles observe the call boundary; this is not model execution."""
        value, calls = fixture(), []
        payload = value["encoded_model_input"]
        batch = SimpleNamespace(
            validate=lambda config: calls.append(("validate", config)),
            public_args=lambda config: ("public-input",),
            role_args=lambda memory, config: (memory, "role-input"))
        module = ModuleType("rz_pals_model.model")

        class InputPort:
            @staticmethod
            def from_payload(actual, device):
                calls.append(("payload", actual is payload, copy.deepcopy(actual), device))
                return batch

        module.VirtualReviewTensorInput = InputPort

        class ModelPort:
            config = "registered-config"

            def parameters(self):
                yield SimpleNamespace(device="fixture-device")

            def public_encoder(self, *args):
                calls.append(("public", args))
                return "public-memory"

            def role_graph(self, role):
                calls.append(("role", role))
                return lambda *args: ("port-output", args)

        before = copy.deepcopy(value)
        with patch.dict(sys.modules, {"rz_pals_model.model": module}):
            actual_batch, outputs = submitted.submitted_collate_and_forward(ModelPort(), "critic", value)
        self.assertIs(actual_batch, batch)
        self.assertEqual(outputs, ("port-output", ("public-memory", "role-input")))
        self.assertEqual(calls[0], ("payload", True, payload, "fixture-device"))
        self.assertEqual(value, before)


if __name__ == "__main__":
    unittest.main()
