"""Synthetic historical metadata only; no model/native/physical evidence."""
import copy
import hashlib
import importlib.abc
import json
import sys
import unittest
from unittest import mock


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("neural imports forbidden in historical checkpoint contract tests")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_historical_checkpoint as historical
finally:
    sys.meta_path.remove(_guard)


def h(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fixture():
    """Original-shaped plain JSON cannot issue original learner authority."""
    cost = {"forward_calls": 24, "backward_calls": 24, "samples": 24,
            "wall_time_ns": 1000, "partial_accumulation_ns": 0, "replay_ns": 0,
            "masked_groups": 0, "output_bytes": 4096}
    checkpoint = {"schema": historical.CHECKPOINT_DOMAIN,
                  "checkpoint": {"path": "/old/not-present/checkpoint.pt",
                                 "sha256": h("synthetic-checkpoint-bytes"), "bytes": 4096},
                  "completed_updates": {"proposer": 2, "critic": 1, "validator": 0},
                  **{field: h(field) for field in historical.STATE_HASH_FIELDS},
                  "cost": cost, "pending_microbatches": 0}
    receipt = {"schema": historical.RECEIPT_DOMAIN, "execution_id": "execution-old-1",
               "window_id": "window-old-1", "checkpoint": checkpoint}
    observed = {key: copy.deepcopy(value) for key, value in checkpoint.items()
                if key not in {"schema", "checkpoint"}}
    recovered = {**checkpoint["checkpoint"], "path": r"Z:\relocated\checkpoint.pt"}
    return receipt, observed, recovered


def encoded(receipt):
    raw = (json.dumps(receipt, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")) + "\n").encode("utf-8")
    return raw, {"receipt_sha256": hashlib.sha256(raw).hexdigest(), "receipt_bytes": len(raw)}


def project(receipt, observed, recovered):
    raw, pins = encoded(receipt)
    return historical.project_historical_checkpoint(raw, **pins,
        recovered_checkpoint_pin=recovered, observed_state=observed)


class HistoricalCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_recovered_retirement_uses_its_own_explicit_inner_codec_and_false_authority(self):
        receipt, observed, recovered = fixture()
        receipt.update(schema=historical.RECOVERED_RECEIPT_DOMAIN,
            source_kind="historical_exact_pins", past_native_authority_reissued=False)
        receipt["checkpoint"]["schema"] = historical.RECOVERED_CHECKPOINT_DOMAIN
        self.assertFalse(project(receipt, observed, recovered)["past_native_authority_reissued"])
        for name, value in (("past_native_authority_reissued", True), ("source_kind", "live_native")):
            changed = copy.deepcopy(receipt)
            changed[name] = value
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "reissue historical native"):
                project(changed, observed, recovered)
        receipt["checkpoint"]["schema"] = historical.CHECKPOINT_DOMAIN
        with self.assertRaisesRegex(ValueError, "checkpoint receipt domain"):
            project(receipt, observed, recovered)

    def test_verified_relocation_preserves_paths_without_opening_either(self):
        receipt, observed, recovered = fixture()
        raw, pins = encoded(receipt)
        with mock.patch("builtins.open", side_effect=AssertionError("historical paths must not be opened")) as opened:
            parsed = historical.parse_historical_checkpoint_receipt(raw, **pins)
            result = historical.project_historical_checkpoint(raw, **pins,
                recovered_checkpoint_pin=recovered, observed_state=observed)
        opened.assert_not_called()
        self.assertEqual(parsed, receipt)
        self.assertEqual(result["schema"], historical.PROJECTION_DOMAIN)
        self.assertEqual(result["historical_checkpoint_pin"], receipt["checkpoint"]["checkpoint"])
        self.assertEqual(result["recovered_checkpoint_pin"], recovered)
        self.assertNotEqual(result["historical_checkpoint_pin"]["path"], recovered["path"])
        self.assertEqual(result["receipt_pin"], {"sha256": pins["receipt_sha256"], "bytes": len(raw)})
        self.assertEqual(result["historical_execution_id"], "execution-old-1")
        self.assertEqual(result["historical_window_id"], "window-old-1")
        self.assertEqual(result["completed_updates"], observed["completed_updates"])
        self.assertEqual(result["snapshot_cost"], observed["cost"])
        for field in historical.STATE_HASH_FIELDS:
            self.assertEqual(result[field], observed[field])
        self.assertEqual(result["status"], "historical_integrity_only")
        self.assertIs(result["past_native_authority_reissued"], False)
        self.assertIs(result["execution_authorized"], False)
        self.assertEqual(result["physical_completion_scope"], "checkpoint_does_not_prove_old_run_closure")
        self.assertNotIn("trailing_cost", result)
        self.assertNotIn("total_run_cost", result)

    def test_original_receipt_byte_pin_precedes_decoding(self):
        receipt, _, _ = fixture()
        raw, pins = encoded(receipt)
        changed = raw.replace(b"window-old-1", b"window-old-2")
        self.assertEqual(len(changed), len(raw))
        with self.assertRaisesRegex(ValueError, "receipt raw byte pin mismatch"):
            historical.parse_historical_checkpoint_receipt(changed, **pins)
        with self.assertRaisesRegex(ValueError, "receipt raw byte pin mismatch"):
            historical.parse_historical_checkpoint_receipt(raw, **{**pins, "receipt_bytes": len(raw) + 1})

    def test_recorded_path_tamper_cannot_keep_old_receipt_pin(self):
        receipt, _, _ = fixture()
        raw, pins = encoded(receipt)
        changed = raw.replace(b"/old/not-present/checkpoint.pt", b"/new/not-present/checkpoint.pt")
        self.assertEqual(len(changed), len(raw))
        with self.assertRaisesRegex(ValueError, "receipt raw byte pin mismatch"):
            historical.parse_historical_checkpoint_receipt(changed, **pins)

    def test_other_checkpoint_hash_or_length_cannot_be_relocated_as_original(self):
        receipt, observed, recovered = fixture()
        for field, value in (("sha256", h("other-checkpoint")), ("bytes", 4097)):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "differs from recovered exact hash/bytes"):
                    project(receipt, observed, {**recovered, field: value})

    def test_counts_require_real_unsigned_integers(self):
        for source_side in (False, True):
            for value in (True, -1, 2**64):
                receipt, observed, recovered = fixture()
                target = receipt["checkpoint"] if source_side else observed
                target["completed_updates"]["proposer"] = value
                with self.subTest(source_side=source_side, value=value):
                    with self.assertRaisesRegex(ValueError, "unsigned integer required"):
                        project(receipt, observed, recovered)

    def test_snapshot_cost_rejects_bool_and_unsigned_overflow(self):
        for source_side in (False, True):
            for value in (True, 2**64):
                receipt, observed, recovered = fixture()
                target = receipt["checkpoint"] if source_side else observed
                target["cost"]["forward_calls"] = value
                with self.subTest(source_side=source_side, value=value):
                    with self.assertRaisesRegex(ValueError, "unsigned integer required"):
                        project(receipt, observed, recovered)

    def test_checkpoint_bytes_reject_bool_and_unsigned_overflow(self):
        for source_side in (False, True):
            for value in (True, 2**64):
                receipt, observed, recovered = fixture()
                target = receipt["checkpoint"]["checkpoint"] if source_side else recovered
                target["bytes"] = value
                with self.subTest(source_side=source_side, value=value):
                    with self.assertRaisesRegex(ValueError, "unsigned integer required"):
                        project(receipt, observed, recovered)

    def test_completed_boundary_rejects_false_and_partial_accumulation(self):
        for source_side in (False, True):
            for value in (False, 1):
                receipt, observed, recovered = fixture()
                (receipt["checkpoint"] if source_side else observed)["pending_microbatches"] = value
                with self.subTest(source_side=source_side, value=value):
                    with self.assertRaisesRegex(ValueError, "needs a completed optimizer boundary"):
                        project(receipt, observed, recovered)

    def test_each_scratch_state_hash_must_match_original(self):
        for field in historical.STATE_HASH_FIELDS:
            receipt, observed, recovered = fixture()
            observed[field] = h("different-" + field)
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field + " differs from recovered tensor scratch"):
                    project(receipt, observed, recovered)

    def test_optimizer_counts_and_snapshot_cost_are_exact_independent_joins(self):
        receipt, observed, recovered = fixture()
        observed["completed_updates"]["critic"] += 1
        with self.assertRaisesRegex(ValueError, "completed_updates differ from recovered tensor scratch"):
            project(receipt, observed, recovered)
        receipt, observed, recovered = fixture()
        observed["cost"]["replay_ns"] = 1
        with self.assertRaisesRegex(ValueError, "snapshot cost differs from recovered tensor scratch"):
            project(receipt, observed, recovered)

    def test_unknown_source_authority_fields_cannot_modernize_old_capability(self):
        for position in ("outer", "checkpoint", "observed"):
            receipt, observed, recovered = fixture()
            target = receipt if position == "outer" else receipt["checkpoint"] if position == "checkpoint" else observed
            target["original_native_lineage_verified"] = True
            description = {"outer": "original checkpoint retirement receipt",
                           "checkpoint": "historical trainer checkpoint",
                           "observed": "observed recovered tensor state"}[position]
            with self.subTest(position=position):
                with self.assertRaisesRegex(ValueError, description + " fields mismatch"):
                    project(receipt, observed, recovered)

    def test_projection_and_parser_results_do_not_alias_caller_data(self):
        receipt, observed, recovered = fixture()
        original = copy.deepcopy((receipt, observed, recovered))
        result = project(receipt, observed, recovered)
        result["completed_updates"]["proposer"] = 999
        result["snapshot_cost"]["forward_calls"] = 999
        result["recovered_checkpoint_pin"]["path"] = "/changed/result-only.pt"
        result["historical_checkpoint_pin"]["path"] = "/changed/history-result-only.pt"
        raw, pins = encoded(receipt)
        parsed = historical.parse_historical_checkpoint_receipt(raw, **pins)
        parsed["checkpoint"]["cost"]["samples"] = 999
        self.assertEqual((receipt, observed, recovered), original)
        self.assertEqual(historical.parse_historical_checkpoint_receipt(raw, **pins), receipt)

    def test_duplicate_json_fields_and_nonfinite_tokens_have_specific_rejections(self):
        receipt, _, _ = fixture()
        raw, _ = encoded(receipt)
        variants = (
            (raw.replace(b'"window_id":"window-old-1"',
                         b'"window_id":"window-old-1","window_id":"window-old-2"'),
             "duplicate historical checkpoint JSON field"),
            (raw.replace(b'"forward_calls":24', b'"forward_calls":24,"forward_calls":25'),
             "duplicate historical checkpoint JSON field"),
            (raw.replace(b'"pending_microbatches":0', b'"pending_microbatches":NaN'),
             "nonfinite historical checkpoint JSON rejected"),
            (raw.replace(b'"forward_calls":24', b'"forward_calls":Infinity'),
             "nonfinite historical checkpoint JSON rejected"),
            (raw.replace(b'"forward_calls":24', b'"forward_calls":-Infinity'),
             "nonfinite historical checkpoint JSON rejected"),
        )
        for changed, boundary in variants:
            with self.subTest(boundary=boundary):
                with self.assertRaisesRegex(ValueError, boundary):
                    historical.parse_historical_checkpoint_receipt(changed,
                        receipt_sha256=hashlib.sha256(changed).hexdigest(), receipt_bytes=len(changed))

    def test_bytes_and_scalar_pins_are_bounded_before_schema_use(self):
        receipt, _, _ = fixture()
        raw, pins = encoded(receipt)
        with self.assertRaisesRegex(ValueError, "immutable historical checkpoint receipt bytes required"):
            historical.parse_historical_checkpoint_receipt(bytearray(raw), **pins)
        with self.assertRaisesRegex(ValueError, "unsigned integer required"):
            historical.parse_historical_checkpoint_receipt(raw, **{**pins, "receipt_bytes": True})
        with self.assertRaisesRegex(ValueError, "unsigned integer required"):
            historical.parse_historical_checkpoint_receipt(raw, **{**pins, "receipt_bytes": 1024 * 1024 + 1})
        with self.assertRaisesRegex(ValueError, "lowercase historical checkpoint SHA-256 required"):
            historical.parse_historical_checkpoint_receipt(raw, **{**pins, "receipt_sha256": "A" * 64})


if __name__ == "__main__":
    unittest.main()
