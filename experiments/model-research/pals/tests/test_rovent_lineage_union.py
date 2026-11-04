"""Projection-only PC/R leakage fixtures; no source reader or actual target."""
import copy
from dataclasses import FrozenInstanceError
import hashlib
import importlib.abc
import sys
from types import SimpleNamespace
import unittest


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("neural imports forbidden in PC/R projection tests")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_lineage_union as union
    from rz_pals_model import rovent_generation as legacy
    from rz_pals_model.generation_data import digest
    from rz_pals_model import rovent_corpus_codec as codec
    from test_rovent_generation import recipe_value
finally:
    sys.meta_path.remove(_guard)


def h(value):
    return hashlib.sha256(value.encode()).hexdigest()


def recipe():
    return legacy.SeriesRecipe(recipe_value())


def entry(registered, domain, suffix, *, split="train", stage="common", input_sha256=None, ancestors=()):
    """Synthetic source provenance: never a PC input, native owner or target."""
    opening = next(name for name, part in registered.to_dict()["opening_splits"].items() if part == split)
    return {"source_domain": domain, "origin_sha256": h(domain + "origin" + suffix),
        "projection": {"input_sha256": h(domain + "input" + suffix) if input_sha256 is None else input_sha256,
            "record_sha256": h(domain + "record" + suffix), "producer_sha256": h(domain + "producer" + stage),
            "stage_id": stage, "round_id": registered.stage(stage).round_id, "teacher_actor_id": None,
            "opening_id": opening, "game_id": "game-" + suffix, "line_genealogy_id": "line-" + suffix,
            "rules_state_sha256": h("rules-" + suffix), "transposition_sha256": h("transposition-" + suffix)},
        "ancestor_origins": sorted((copy.deepcopy(ref) for ref in ancestors),
                                   key=lambda ref: (ref["source_domain"], ref["origin_sha256"]))}


def ref(value):
    return {key: value[key] for key in ("source_domain", "origin_sha256")}


class LineageUnionTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_repeated_actual_input_preserves_occurrences_without_inflating_unique_inputs(self):
        registered = recipe()
        first = entry(registered, union.R_SOURCE, "first")
        second = entry(registered, union.R_SOURCE, "second", input_sha256=first["projection"]["input_sha256"])
        owner = union.LineageUnion.from_projections(registered, [first, second])
        self.assertEqual(len(owner.all_occurrences()), 2)
        self.assertEqual(len(owner.rows("train", optimization=True)), 2)
        report = owner.observation().to_mapping()
        self.assertEqual(report["occurrence_count"], 2)
        self.assertEqual(report["unique_input_count"], 1)
        self.assertEqual(next(item for item in report["sources"] if item["source_domain"] == union.R_SOURCE),
                         {"source_domain": union.R_SOURCE, "occurrence_count": 2, "unique_input_count": 1})
        self.assertNotIn("valid_target_count", report)
        self.assertIs(report["target_validity_verified"], False)
        self.assertIs(report["execution_authorized"], False)
        self.assertIs(report["original_source_authority_verified"], False)
        self.assertNotIn("role", owner.rows("train")[0].projection())

    def test_new_union_canonical_is_explicit_and_legacy_duplicate_rule_is_unchanged(self):
        registered = recipe()
        first = entry(registered, union.PC_SOURCE, "first")
        second = entry(registered, union.PC_SOURCE, "second", input_sha256=first["projection"]["input_sha256"])
        owner = union.LineageUnion.from_projections(registered, [first, second])
        self.assertEqual(owner.sha256, digest(union.UNION_DOMAIN,
            {"recipe_sha256": registered.sha256, "occurrences": [first, second]}))
        legacy_rows = [first["projection"], second["projection"]]
        with self.assertRaisesRegex(ValueError, "duplicate original row index"):
            legacy.GlobalLineageIndex(registered, legacy_rows)
        old = legacy.GlobalLineageIndex(registered, legacy_rows[:1])
        self.assertEqual(old.sha256, digest(legacy.INDEX_DOMAIN,
                         {"recipe_sha256": registered.sha256, "records": legacy_rows[:1]}))

    def test_pc_r_cross_split_transposition_excludes_both_and_whole_opening_component(self):
        registered = recipe()
        train = entry(registered, union.PC_SOURCE, "pc")
        other_train = entry(registered, union.R_SOURCE, "same-opening")
        holdout = entry(registered, union.R_SOURCE, "r-holdout", split="holdout")
        holdout["projection"]["transposition_sha256"] = train["projection"]["transposition_sha256"]
        owner = union.LineageUnion.from_projections(registered, [train, other_train, holdout])
        self.assertEqual(owner.rows("train", optimization=True), ())
        self.assertEqual(owner.rows("holdout"), ())
        self.assertEqual(len(owner.all_occurrences()), 3)
        self.assertEqual(owner.observation().excluded_occurrence_count, 3)
        with self.assertRaisesRegex(ValueError, "global holdout/cross-split contamination"):
            owner.require_training_origins("common", [ref(train)])

    def test_ancestor_origin_edge_propagates_holdout_across_sources(self):
        registered = recipe()
        parent = entry(registered, union.PC_SOURCE, "parent")
        child = entry(registered, union.R_SOURCE, "child", split="validation", stage="sf-ft", ancestors=(ref(parent),))
        owner = union.LineageUnion.from_projections(registered, [parent, child])
        self.assertEqual(owner.rows("train", optimization=True), ())
        self.assertEqual(owner.rows("validation"), ())
        self.assertEqual(owner.observation().cross_split_component_count, 1)

    def test_duplicate_record_and_origin_are_rejected_but_input_equality_is_not(self):
        registered = recipe()
        first = entry(registered, union.R_SOURCE, "first")
        for field in ("record", "origin"):
            second = entry(registered, union.R_SOURCE, "second")
            if field == "record":
                second["projection"]["record_sha256"] = first["projection"]["record_sha256"]
            else:
                second["origin_sha256"] = first["origin_sha256"]
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "duplicate PC/R source origin or source record occurrence"):
                    union.LineageUnion.from_projections(registered, [first, second])

    def test_missing_future_sibling_and_cyclic_ancestor_edges_have_distinct_rejections(self):
        registered = recipe()
        missing = entry(registered, union.R_SOURCE, "missing", ancestors=(
            {"source_domain": union.PC_SOURCE, "origin_sha256": h("absent origin")},))
        with self.assertRaisesRegex(ValueError, "ancestor origin missing from full source union"):
            union.LineageUnion.from_projections(registered, [missing])
        future = entry(registered, union.R_SOURCE, "future", stage="sf-ft")
        earlier = entry(registered, union.PC_SOURCE, "earlier", ancestors=(ref(future),))
        with self.assertRaisesRegex(ValueError, "future or sibling stage"):
            union.LineageUnion.from_projections(registered, [earlier, future])
        sibling_stage = next(stage.stage_id for stage in registered.stages
                             if stage.parent_stage_id == "common" and stage.stage_id != "sf-ft")
        sibling = entry(registered, union.PC_SOURCE, "sibling", stage=sibling_stage)
        future["ancestor_origins"] = [ref(sibling)]
        with self.assertRaisesRegex(ValueError, "future or sibling stage"):
            union.LineageUnion.from_projections(registered, [future, sibling])
        first = entry(registered, union.PC_SOURCE, "first")
        second = entry(registered, union.R_SOURCE, "second", ancestors=(ref(first),))
        first["ancestor_origins"] = [ref(second)]
        with self.assertRaisesRegex(ValueError, "occurrence ancestor cycle rejected"):
            union.LineageUnion.from_projections(registered, [first, second])

    def test_inherited_training_references_keep_source_domains_and_checkpoint_requirement(self):
        registered = recipe()
        parent = entry(registered, union.PC_SOURCE, "parent")
        retained = entry(registered, union.R_SOURCE, "retained", stage="sf-ft", ancestors=(ref(parent),))
        owner = union.LineageUnion.from_projections(registered, [parent, retained])
        references = sorted((ref(parent), ref(retained)), key=lambda item: (item["source_domain"], item["origin_sha256"]))
        audit = owner.require_training_origins("sf-ft", references)
        self.assertEqual(audit.unique_input_count, 2)
        self.assertTrue(audit.same_stage_original_checkpoint_authority_required)
        self.assertIs(audit.original_checkpoint_authority_verified, False)
        self.assertIs(audit.execution_authorized, False)
        with self.assertRaisesRegex(ValueError, "future or sibling stage"):
            owner.require_training_origins("common", [ref(retained)])

    def test_projection_inputs_results_and_observation_are_immutable_copies(self):
        registered = recipe()
        original = entry(registered, union.R_SOURCE, "immutable")
        owner = union.LineageUnion.from_projections(registered, [original])
        original["projection"]["game_id"] = "changed-caller-data"
        result = owner.rows("train")[0]
        changed = result.projection()
        changed["game_id"] = "changed-returned-copy"
        self.assertEqual(result.projection()["game_id"], "game-immutable")
        with self.assertRaises(FrozenInstanceError):
            result.source = union.SourceOrigin(union.PC_SOURCE, h("foreign source"))
        with self.assertRaises(FrozenInstanceError):
            owner.observation().execution_authorized = True
        self.assertEqual(owner.observation().occurrence_count, 1)

    def test_internal_exclusion_state_cannot_change_without_integrity_failure(self):
        registered = recipe()
        owner = union.LineageUnion.from_projections(registered, [entry(registered, union.PC_SOURCE, "immutable")])
        owner._excluded = frozenset({0})
        with self.assertRaisesRegex(ValueError, "immutable PC/R occurrence projection seal changed"):
            owner.rows("train", optimization=True)

    def test_strict_fields_domains_and_metadata_limits_fail_before_sampling(self):
        registered = recipe()
        row = entry(registered, union.R_SOURCE, "strict")
        unknown = copy.deepcopy(row)
        unknown["execution_authorized"] = True
        with self.assertRaisesRegex(ValueError, "PC/R occurrence metadata fields mismatch"):
            union.LineageUnion.from_projections(registered, [unknown])
        unknown = copy.deepcopy(row)
        unknown["projection"]["role"] = "proposer"
        with self.assertRaisesRegex(ValueError, "Rovent original row index fields mismatch"):
            union.LineageUnion.from_projections(registered, [unknown])
        unknown = copy.deepcopy(row)
        unknown["source_domain"] = "legacy-diagnostic"
        with self.assertRaisesRegex(ValueError, "explicit supported PC/R source domain required"):
            union.LineageUnion.from_projections(registered, [unknown])
        with self.assertRaisesRegex(ValueError, "lineage metadata byte/edge limit exceeded"):
            union.LineageUnion.from_projections(registered, [row], max_metadata_bytes=128)
        with self.assertRaisesRegex(ValueError, "invalid bounded unsigned integer"):
            union.LineageUnion.from_projections(registered, [row], max_occurrences=True)

    def test_validation_and_holdout_rows_cannot_be_requested_for_optimization(self):
        registered = recipe()
        owner = union.LineageUnion.from_projections(registered, [entry(registered, union.R_SOURCE, "valid", split="validation")])
        self.assertEqual(len(owner.rows("validation", source_domain=union.R_SOURCE)), 1)
        with self.assertRaisesRegex(ValueError, "validation/global holdout cannot enter optimization"):
            owner.rows("validation", optimization=True)

    def test_old_pc_r_only_union_keeps_original_domains_seal_and_source_counts(self):
        registered = recipe()
        rows = [entry(registered, union.PC_SOURCE, "old"), entry(registered, union.R_SOURCE, "r")]
        owner = union.LineageUnion.from_projections(registered, rows)
        self.assertEqual(owner.sha256, digest("rz-rovent-pc-r-lineage-union/1",
            {"recipe_sha256": registered.sha256, "occurrences": rows}))
        self.assertEqual(owner.observation().to_mapping()["sources"], [
            {"source_domain": "rz-rovent-training-record/1", "occurrence_count": 1, "unique_input_count": 1},
            {"source_domain": "rz-rovent-r-readout-record/1", "occurrence_count": 1, "unique_input_count": 1}])

    def test_submitted_pc_domain_is_preserved_without_relabelling_repeated_inputs(self):
        registered = recipe()
        first = entry(registered, union.PC_SOURCE, "old")
        second = entry(registered, union.PC_SOURCE_V2, "submitted", input_sha256=first["projection"]["input_sha256"])
        owner = union.LineageUnion.from_projections(registered, [first, second])
        self.assertEqual([row.source.source_domain for row in owner.all_occurrences()],
                         ["rz-rovent-training-record/1", "rz-rovent-training-record/2"])
        self.assertEqual(len(owner.rows("train", optimization=True, source_domain=union.PC_SOURCE_V2)), 1)
        self.assertEqual(owner.sha256, digest(union.UNION_DOMAIN,
            {"recipe_sha256": registered.sha256, "occurrences": [first, second]}))
        self.assertEqual(owner.observation().occurrence_count, 2)
        self.assertNotIn("valid_target_count", owner.observation().to_mapping())

    def test_mixed_pc_families_and_r_share_cross_split_exclusion_and_retained_rejection(self):
        registered = recipe()
        old = entry(registered, union.PC_SOURCE, "old")
        submitted = entry(registered, union.PC_SOURCE_V2, "new", input_sha256=old["projection"]["input_sha256"])
        holdout = entry(registered, union.R_SOURCE, "holdout", split="holdout")
        holdout["projection"]["transposition_sha256"] = submitted["projection"]["transposition_sha256"]
        owner = union.LineageUnion.from_projections(registered, [old, submitted, holdout])
        self.assertEqual(owner.rows("train", optimization=True), ())
        self.assertEqual(owner.observation().excluded_occurrence_count, 3)
        self.assertEqual(len(owner.all_occurrences()), 3)
        with self.assertRaisesRegex(ValueError, "global holdout/cross-split contamination"):
            owner.require_training_origins("common", [ref(submitted)])

    def test_submitted_ancestor_edge_retains_its_own_domain_in_mixed_chain(self):
        registered = recipe()
        old = entry(registered, union.PC_SOURCE, "old")
        submitted = entry(registered, union.PC_SOURCE_V2, "new", stage="sf-ft", ancestors=(ref(old),))
        current = entry(registered, union.R_SOURCE, "r", stage="sf-ft", ancestors=(ref(submitted),))
        owner = union.LineageUnion.from_projections(registered, [old, submitted, current])
        audit = owner.require_training_origins("sf-ft", sorted((ref(old), ref(submitted), ref(current)),
            key=lambda value: (value["source_domain"], value["origin_sha256"])))
        self.assertEqual({value.source_domain for value in audit.origins}, set((*union.PC_SOURCES, union.R_SOURCE)))
        wrong = copy.deepcopy(current)
        wrong["ancestor_origins"][0]["source_domain"] = union.PC_SOURCE
        with self.assertRaisesRegex(ValueError, "ancestor origin missing from full source union"):
            union.LineageUnion.from_projections(registered, [old, submitted, wrong])

    def test_owner_codec_must_explicitly_match_original_manifest_family(self):
        old = SimpleNamespace(_manifest={"schema": codec.LEGACY_CODEC.manifest_domain})
        submitted = SimpleNamespace(_manifest={"schema": codec.SUBMITTED_CODEC.manifest_domain}, _codec=codec.SUBMITTED_CODEC)
        self.assertEqual(union.pc_source_domain(old), union.PC_SOURCE)
        self.assertEqual(union.pc_source_domain(submitted), union.PC_SOURCE_V2)
        for owner in (SimpleNamespace(_manifest=submitted._manifest),
                      SimpleNamespace(_manifest=old._manifest, _codec=codec.SUBMITTED_CODEC)):
            with self.assertRaisesRegex(ValueError, "owner codec differs from its manifest family"):
                union.pc_source_domain(owner)


if __name__ == "__main__":
    unittest.main()
