"""Passive cumulative R seal metadata only; no model/optimizer execution."""
import copy
import hashlib
import importlib.abc
import sys
import unittest


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("neural imports forbidden in R ancestry metadata tests")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_r_ancestry as ancestry
finally:
    sys.meta_path.remove(_guard)


def seal(name):
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def record(inputs=(), origins=()):
    return {"schema": "rz-rovent-r-auxiliary-ancestry/1",
        "input_sha256": sorted(seal(name) for name in inputs),
        "origin_sha256": sorted(seal(name) for name in origins), "execution_authorized": False}


def auxiliary(*, retained_inputs=(), retained_origins=(), ancestor_inputs=(), ancestor_origins=()):
    """Projection-only fixture; not a valid complete source-owned journal."""
    return {"schema": "rz-rovent-r-auxiliary-journal/1",
        "retained_input_sha256": sorted(seal(name) for name in retained_inputs),
        "retained_origin_sha256": sorted(seal(name) for name in retained_origins),
        "ancestor_input_sha256": sorted(seal(name) for name in ancestor_inputs),
        "ancestor_origin_sha256": sorted(seal(name) for name in ancestor_origins)}


class RAncestryTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_exact_object_retains_original_seals_without_mutation(self):
        original = record(("a", "b"), ("origin-a", "origin-b"))
        before = copy.deepcopy(original)
        self.assertIs(ancestry.validate_ancestry(original), original)
        self.assertEqual(original, before)

    def test_absent_optional_fields_and_empty_chain_return_passive_empty_record(self):
        self.assertEqual(ancestry.ancestry_from_journal({"unrelated": "preserved by its owner"}), record())
        self.assertEqual(ancestry.merge_ancestry([]), record())

    def test_shape_domain_authority_and_count_field_are_closed(self):
        for change in ("schema", "authority", "count", "missing"):
            with self.subTest(change=change):
                value = record()
                if change == "schema":
                    value["schema"] = "rz-rovent-r-auxiliary-ancestry/2"
                elif change == "authority":
                    value["execution_authorized"] = 0
                elif change == "count":
                    value["nnuec_updates"] = 0
                else:
                    del value["origin_sha256"]
                with self.assertRaisesRegex(ValueError, "fields mismatch|cannot issue execution authority"):
                    ancestry.validate_ancestry(value)

    def test_seals_are_lowercase_sorted_unique_lists(self):
        unordered = sorted((seal("a"), seal("b")), reverse=True)
        for seals in ((seal("a"),), [seal("a"), seal("a")], ["A" * 64], [False], unordered):
            with self.subTest(seals=seals):
                value = record()
                value["input_sha256"] = seals
                with self.assertRaisesRegex(ValueError, "seal list|lowercase SHA-256|sorted and unique"):
                    ancestry.validate_ancestry(value)

    def test_retained_and_ancestor_sets_are_projected_once(self):
        journal = {"r_auxiliary": auxiliary(retained_inputs=("b", "c"), retained_origins=("ob", "oc"),
            ancestor_inputs=("a", "b"), ancestor_origins=("oa", "ob"))}
        before = copy.deepcopy(journal)
        self.assertEqual(ancestry.ancestry_from_journal(journal), record(("a", "b", "c"), ("oa", "ob", "oc")))
        self.assertEqual(journal, before)

    def test_enabled_disabled_enabled_chain_preserves_original_r_sets(self):
        a = ancestry.ancestry_from_journal({"r_auxiliary": auxiliary(retained_inputs=("a",), retained_origins=("oa",))})
        disabled_b = {"r_auxiliary_ancestry": copy.deepcopy(a), "updates": [{"role": "proposer"}]}
        b = ancestry.ancestry_from_journal(disabled_b)
        enabled_c = {"r_auxiliary": auxiliary(retained_inputs=("c",), retained_origins=("oc",),
            ancestor_inputs=("a",), ancestor_origins=("oa",)), "r_auxiliary_ancestry": b}
        self.assertEqual(ancestry.ancestry_from_journal(enabled_c), record(("a", "c"), ("oa", "oc")))
        self.assertNotIn("r_auxiliary", disabled_b)

    def test_equal_and_contained_cumulative_metadata_do_not_duplicate_sets(self):
        small, large = record(("a",), ("oa",)), record(("a", "b"), ("oa", "ob"))
        for chain in ([small, large, small], [large, small, large], [large, large]):
            self.assertEqual(ancestry.merge_ancestry(chain), large)

    def test_incomparable_forks_cannot_be_silently_unioned(self):
        with self.assertRaisesRegex(ValueError, "not a nested cumulative"):
            ancestry.merge_ancestry([record(("a",), ("oa",)), record(("b",), ("ob",))])

    def test_fork_rejection_does_not_depend_on_a_larger_record_argument_order(self):
        a, b, larger = record(("a",), ("oa",)), record(("b",), ("ob",)), record(("a", "b"), ("oa", "ob"))
        for values in ([a, b, larger], [larger, a, b], [b, larger, a]):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "not a nested cumulative"):
                    ancestry.merge_ancestry(values)

    def test_cross_direction_input_origin_containment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not a nested cumulative"):
            ancestry.merge_ancestry([record(("a",), ("oa", "ob")), record(("a", "b"), ("oa",))])

    def test_independent_journal_counts_are_never_read_or_exposed(self):
        class CountMustRemainWithOwner:
            def __int__(self):
                raise AssertionError("ancestry must not consume NNUEC counts")

            def __add__(self, other):
                raise AssertionError("ancestry must not sum NNUEC counts")

        sidecar = auxiliary(retained_inputs=("a",), retained_origins=("oa",))
        sidecar.update(current_nnuec_updates=CountMustRemainWithOwner(), ancestor_nnuec_updates=CountMustRemainWithOwner())
        result = ancestry.ancestry_from_journal({"r_auxiliary": sidecar})
        self.assertEqual(result, record(("a",), ("oa",)))
        self.assertNotIn("nnuec_updates", result)

    def test_null_or_partial_source_sidecars_do_not_erase_old_lineage(self):
        for journal in ({"r_auxiliary": None}, {"r_auxiliary_ancestry": None},
            {"r_auxiliary": {"schema": "rz-rovent-r-auxiliary-journal/1"}},
            {"r_auxiliary": auxiliary(), "r_auxiliary_ancestry": {}}):
            with self.subTest(journal=journal):
                with self.assertRaisesRegex(ValueError, "journal seal projection|ancestry fields mismatch"):
                    ancestry.ancestry_from_journal(journal)

    def test_returned_projection_does_not_alias_source_lists(self):
        value = record(("a",), ("oa",))
        merged = ancestry.merge_ancestry([value])
        merged["input_sha256"].clear()
        self.assertEqual(value, record(("a",), ("oa",)))

    def test_set_and_chain_limits_are_explicit(self):
        value = record()
        value["input_sha256"] = ["0" * 64] * 262145
        with self.assertRaisesRegex(ValueError, "bounded R ancestry seal list"):
            ancestry.validate_ancestry(value)
        with self.assertRaisesRegex(ValueError, "bounded explicit R ancestry chain"):
            ancestry.merge_ancestry([record()] * 257)
        with self.assertRaisesRegex(ValueError, "checkpoint journal mapping"):
            ancestry.ancestry_from_journal([])


if __name__ == "__main__":
    unittest.main()
