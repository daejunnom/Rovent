"""Fresh scalar initializer checks; no Collection/runtime/training acceptance."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rz_pals_model import nnuec_initialization as initializer
from rz_pals_model import nnuec_reference as base
from rz_pals_model import nnuec_v2_reference as relation


class NnuecInitializationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).absolute()
        self.config = {"board_width": 2, "line_width": 3, "context_width": 2,
            "local_width": 2, "hidden_width": 2, "causal_window": 3,
            "max_lines": 4, "max_plies": 8}
        self.config_path = self.root / "explicit-config.json"
        self.config_path.write_bytes(json.dumps(self.config, sort_keys=True).encode("ascii"))

    def invoke(self, output="fresh", **changes):
        args = {"config_sha256": hashlib.sha256(self.config_path.read_bytes()).hexdigest(),
            "seed": 23, "parameter_scale": 0.03125, "output_scale": 1.0,
            "source_commit": "a" * 40, "output_directory": self.root / output,
            "maximum_wall_seconds": 60, "maximum_output_bytes": initializer.MAX_OUTPUT_BYTES,
            "execute": True}
        args.update(changes)
        return initializer.initialize_nnuec_v2(self.config_path, **args)

    def test_splitmix_integer_recurrence_matches_independent_known_seed_zero_words(self):
        source = initializer._SplitMix64(0)
        self.assertEqual(source.next(), 0xE220A8397B1DCDAF)
        self.assertEqual(source.next(), 0x6E789E6AA1B965F4)
        self.assertRaises(ValueError, initializer._SplitMix64, True)
        self.assertRaises(ValueError, initializer._SplitMix64, 1 << 64)

    def test_fresh_v2_checkpoint_preserves_exact_shape_untrained_declaration_and_both_hash_domains(self):
        # Even an accidental independent reference forward is forbidden in this
        # initializer. Parameter construction/parser/hash operations are the
        # only actual numerical work of this source-level generator check.
        with patch.object(base, "evaluate", side_effect=AssertionError("forward forbidden")), \
                patch.object(relation, "evaluate", side_effect=AssertionError("forward forbidden")):
            result = self.invoke()
        root = self.root / "fresh"
        raw = (root / initializer.WEIGHTS_FILENAME).read_bytes()
        checkpoint = relation.NnuecV2Checkpoint.from_mapping(json.loads(raw))
        receipt = json.loads((root / initializer.RECEIPT_FILENAME).read_bytes())
        self.assertEqual(checkpoint.base.config.to_mapping(), self.config)
        self.assertEqual(dict(checkpoint.base.training), {"kind": "untrained"})
        self.assertEqual(result["weights"]["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(result["semantic_weights_sha256"], checkpoint.weights_sha256)
        self.assertNotEqual(result["weights"]["sha256"], result["semantic_weights_sha256"])
        self.assertEqual(len(receipt["parameter_inventory"]), len(base.PARAMETER_FIELDS) + len(relation.PARAMETER_FIELDS))
        self.assertEqual(receipt["algorithm"], initializer.ALGORITHM)
        self.assertEqual(receipt["claims"]["neural_forwards"], 0)
        self.assertEqual(receipt["claims"]["optimizer_updates"], 0)
        self.assertFalse(receipt["claims"]["arena_eligible"])
        self.assertFalse(receipt["claims"]["collection_origin_granted"])
        self.assertEqual((root / initializer.CONFIG_FILENAME).read_bytes(), self.config_path.read_bytes())

    def test_same_explicit_seed_config_and_scale_reproduce_weights_without_using_global_rng(self):
        first = self.invoke("first")
        repeated = self.invoke("repeated")
        other = self.invoke("other", seed=29)
        self.assertEqual(first["weights"], repeated["weights"])
        self.assertEqual(first["semantic_weights_sha256"], repeated["semantic_weights_sha256"])
        self.assertNotEqual(first["semantic_weights_sha256"], other["semantic_weights_sha256"])

    def test_existing_complete_or_partial_directory_is_never_relabelled_or_overwritten(self):
        self.invoke()
        original = (self.root / "fresh" / initializer.RECEIPT_FILENAME).read_bytes()
        with self.assertRaises(FileExistsError):
            self.invoke(seed=29)
        self.assertEqual((self.root / "fresh" / initializer.RECEIPT_FILENAME).read_bytes(), original)
        partial = self.root / "partial"
        partial.mkdir()
        (partial / "nnuec.json.partial").write_bytes(b"original partial evidence")
        with self.assertRaises(FileExistsError):
            self.invoke("partial")
        self.assertEqual((partial / "nnuec.json.partial").read_bytes(), b"original partial evidence")

    def test_duplicate_unknown_or_unpinned_configuration_rejected_before_creating_output(self):
        for raw in (b'{"board_width":2,"board_width":3}',
                    json.dumps({**self.config, "training": {"kind": "learned"}}).encode("ascii")):
            self.config_path.write_bytes(raw)
            with self.assertRaises(ValueError):
                self.invoke()
            self.assertFalse((self.root / "fresh").exists())
        self.config_path.write_bytes(json.dumps(self.config).encode("ascii"))
        with self.assertRaises(ValueError):
            self.invoke(config_sha256="0" * 64)
        self.assertFalse((self.root / "fresh").exists())

    def test_explicit_execution_valid_scales_and_conservative_output_credit_are_required(self):
        for change in ({"execute": False}, {"parameter_scale": 0.0}, {"parameter_scale": True},
                       {"parameter_scale": float("nan")}, {"output_scale": float("inf")},
                       {"output_scale": 10001.0}, {"maximum_output_bytes": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.invoke(**change)
            self.assertFalse((self.root / "fresh").exists())

    def test_expiry_has_no_success_receipt_and_never_renews_original_deadline(self):
        times = iter((0.0, 61.0))
        with self.assertRaises(TimeoutError):
            self.invoke(clock=lambda: next(times))
        self.assertFalse((self.root / "fresh" / initializer.RECEIPT_FILENAME).exists())


if __name__ == "__main__":
    unittest.main()
