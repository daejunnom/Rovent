"""Synthetic closure metadata only; no file, build, native or target proof."""
import copy
import hashlib
import importlib.abc
import json
import sys
import unittest
from unittest.mock import patch


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime", "tensorflow", "jax"}:
            raise AssertionError("neural imports forbidden in closed source-review fixtures")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import native_source_review as review
    from native_source_review_fixture import FeedbackObservationFixtureTests, IdleArchiveFixtureTests
finally:
    sys.meta_path.remove(_guard)


_ENGINE = "crates/rz-search/src/pals/engine.rs"
_NATIVE = "crates/rz-arena/src/pals_collect/native.rs"
_HELPER = "crates/rz-search/src/pals/engine/semantic_state.rs"
_COMPONENTS = (
    _ENGINE, _NATIVE, _HELPER,
    "crates/rz-search/src/pals/engine/researcher_admission.rs",
    "crates/rz-search/src/pals/engine/researcher_profile.rs",
    "crates/rz-search/src/pals/engine/review_lifecycle.rs",
    "crates/rz-search/src/pals/engine/cpu_lifecycle.rs",
    "crates/rz-search/src/pals/engine/followup_execution.rs",
    "crates/rz-search/src/pals/engine/archive_lifecycle.rs",
    "crates/rz-arena/src/pals_collect.rs",
    "crates/rz-arena/src/pals_collect/full_line.rs",
    "crates/rz-arena/src/pals_collect/divergence.rs",
    "crates/rz-arena/src/pals_collect/continuation.rs",
    "crates/rz-arena/src/pals_collect/producer.rs",
)


def h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def closure(pins):
    # Independent expected framing; no current source file is read or enrolled.
    return hashlib.sha256(json.dumps(["rz-pals-native-reviewed-transition-closure/1", pins],
        sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def fixture(lane="default_disabled"):
    components = {path: {"bytes": 11 + index, "sha256": h("synthetic component " + path)}
                  for index, path in enumerate(_COMPONENTS)}
    pair = {path: copy.deepcopy(components[path]) for path in (_ENGINE, _NATIVE)}
    profile = {"schema": "rz-pals-native-reviewed-transition-profile/1", "review_id": "synthetic-" + lane,
        "lane": lane, "transition_sources": copy.deepcopy(pair), "component_pins": components,
        "compiled_search_implementation_sha256": h("synthetic compiled search closure"),
        "closure_sha256": closure(components)}
    manifest = {"source_commit": "a" * 40,
        "files": [{"path": path, **copy.deepcopy(pin)} for path, pin in components.items()]}
    source = {"implementation_sha256": h("binary checked only by the original caller"),
        "native": {"search_implementation_sha256": profile["compiled_search_implementation_sha256"]}}
    return profile, pair, manifest, source


def call(pair, manifest, source, lane="default_disabled"):
    return review.review_transition_sources(pair=pair, manifest=manifest, source=source, lane=lane)


class NativeSourceReviewTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_empty_registry_does_not_enroll_even_a_well_formed_current_description(self):
        _profile, pair, manifest, source = fixture()
        with patch.object(review, "REVIEWED_CLOSURE_PROFILES", ()):
            with self.assertRaisesRegex(review.UnsupportedTransitionSource, "unreviewed_transition_source_closure"):
                call(pair, manifest, source)

    def test_reviewed_whole_closure_returns_only_detached_historical_metadata(self):
        profile, pair, manifest, source = fixture()
        extras = ["unreviewed/z.txt", "unreviewed/a.txt"]
        manifest["files"].extend({"path": path, "bytes": 3, "sha256": h(path)} for path in extras)
        before = copy.deepcopy((profile, pair, manifest, source))
        with patch.object(review, "REVIEWED_CLOSURE_PROFILES", (profile,)):
            proof = call(pair, manifest, source)
            self.assertEqual(set(proof), {"schema", "review_id", "lane", "transition_sources", "component_pins",
                "compiled_search_implementation_sha256", "closure_sha256", "unreviewed_manifest_files",
                "historical_metadata_only", "execution_authorized", "original_native_authority_reissued",
                "rules_proof_authorized", "training_target_authorized"})
            self.assertEqual(proof["schema"], "rz-pals-native-transition-source-review-proof/1")
            self.assertEqual(proof["review_id"], "synthetic-default_disabled")
            self.assertEqual(proof["transition_sources"], pair)
            self.assertEqual(proof["component_pins"], profile["component_pins"])
            self.assertEqual(proof["closure_sha256"], closure(profile["component_pins"]))
            self.assertEqual(proof["unreviewed_manifest_files"], sorted(extras))
            self.assertIs(proof["historical_metadata_only"], True)
            for name in ("execution_authorized", "original_native_authority_reissued", "rules_proof_authorized", "training_target_authorized"):
                self.assertIs(proof[name], False)
            proof["component_pins"][_HELPER]["sha256"] = "0" * 64
            proof["transition_sources"][_ENGINE]["bytes"] = 999
            self.assertEqual(call(pair, manifest, source)["component_pins"], profile["component_pins"])
        self.assertEqual((profile, pair, manifest, source), before)

    def test_one_helper_change_cannot_hide_under_the_same_engine_native_pair(self):
        profile, pair, manifest, source = fixture()
        next(entry for entry in manifest["files"] if entry["path"] == _HELPER)["sha256"] = h("different helper")
        with patch.object(review, "REVIEWED_CLOSURE_PROFILES", (profile,)):
            with self.assertRaisesRegex(ValueError, "helper closure differs"):
                call(pair, manifest, source)

    def test_missing_manifest_component_and_duplicate_path_are_contradictions(self):
        profile, pair, manifest, source = fixture()
        missing = copy.deepcopy(manifest)
        missing["files"] = [entry for entry in missing["files"] if entry["path"] != _HELPER]
        duplicate = copy.deepcopy(manifest)
        duplicate["files"].append(copy.deepcopy(duplicate["files"][0]))
        with patch.object(review, "REVIEWED_CLOSURE_PROFILES", (profile,)):
            with self.assertRaisesRegex(ValueError, "helper closure differs"):
                call(pair, missing, source)
            with self.assertRaisesRegex(ValueError, "duplicate original transition source manifest path"):
                call(pair, duplicate, source)

    def test_profile_cannot_omit_required_helpers_or_replace_them_with_a_missing_capture_file(self):
        profile, pair, manifest, source = fixture()
        profile["component_pins"].pop("crates/rz-arena/src/pals_collect/full_line.rs")
        profile["component_pins"]["crates/rz-arena/src/pals_collect/capture.rs"] = {"bytes": 1, "sha256": h("not an existing required source")}
        profile["closure_sha256"] = closure(profile["component_pins"])
        with patch.object(review, "REVIEWED_CLOSURE_PROFILES", (profile,)):
            with self.assertRaisesRegex(ValueError, "omits a required source helper"):
                call(pair, manifest, source)

    def test_manifest_paths_are_relative_normalized_and_never_read_from_disk(self):
        profile, pair, manifest, source = fixture()
        for path in ("/absolute.rs", "../escape.rs", "x/../source.rs", "./source.rs", "x//source.rs",
                     "x\\source.rs", "C:/source.rs", "source.rs/", "source\x7f.rs", "source\x85.rs"):
            bad = copy.deepcopy(manifest)
            bad["files"].append({"path": path, "bytes": 1, "sha256": h("bad alias")})
            with self.subTest(path=path), patch.object(review, "REVIEWED_CLOSURE_PROFILES", (profile,)):
                with self.assertRaisesRegex(ValueError, "normalized relative transition source path"):
                    call(pair, bad, source)
        with patch.object(review, "REVIEWED_CLOSURE_PROFILES", (profile,)), patch("builtins.open",
                side_effect=AssertionError("source review must not read current files")):
            self.assertEqual(call(pair, manifest, source)["review_id"], profile["review_id"])

    def test_unreviewed_transition_pair_or_old_current_hybrid_is_not_automatically_enrolled(self):
        old, pair, manifest, source = fixture()
        current = copy.deepcopy(old)
        current["review_id"] = "synthetic-current"
        for path in (_ENGINE, _NATIVE):
            current["component_pins"][path] = {"bytes": 99, "sha256": h("current " + path)}
            current["transition_sources"][path] = copy.deepcopy(current["component_pins"][path])
        current["closure_sha256"] = closure(current["component_pins"])
        for changed in (_ENGINE, _NATIVE):
            hybrid, observed = copy.deepcopy(pair), copy.deepcopy(manifest)
            hybrid[changed] = copy.deepcopy(current["transition_sources"][changed])
            next(entry for entry in observed["files"] if entry["path"] == changed).update(hybrid[changed])
            with self.subTest(changed=changed), patch.object(review, "REVIEWED_CLOSURE_PROFILES", (old, current)):
                with self.assertRaisesRegex(review.UnsupportedTransitionSource, "unreviewed_transition_source_closure"):
                    call(hybrid, observed, source)

    def test_compiled_fact_is_required_and_must_match_the_literal_whole_scope(self):
        profile, pair, manifest, source = fixture()
        missing = copy.deepcopy(source)
        missing["native"].pop("search_implementation_sha256")
        changed = copy.deepcopy(source)
        changed["native"]["search_implementation_sha256"] = h("unreviewed compiled helper scope")
        malformed = copy.deepcopy(source)
        malformed["native"]["search_implementation_sha256"] = True
        with patch.object(review, "REVIEWED_CLOSURE_PROFILES", (profile,)):
            with self.assertRaisesRegex(review.UnsupportedTransitionSource, "compiled_search_closure_fact_missing"):
                call(pair, manifest, missing)
            with self.assertRaisesRegex(ValueError, "compiled search closure differs"):
                call(pair, manifest, changed)
            with self.assertRaisesRegex(ValueError, "invalid SHA-256"):
                call(pair, manifest, malformed)

    def test_default_and_single_reply_lanes_need_distinct_explicit_reviews(self):
        default, pair, manifest, source = fixture()
        single = copy.deepcopy(default)
        single.update(review_id="synthetic-single-reply", lane="legacy_single_reply")
        with patch.object(review, "REVIEWED_CLOSURE_PROFILES", (default,)):
            with self.assertRaisesRegex(review.UnsupportedTransitionSource, "unreviewed_transition_source_closure"):
                call(pair, manifest, source, "legacy_single_reply")
        with patch.object(review, "REVIEWED_CLOSURE_PROFILES", (default, single)):
            self.assertEqual(call(pair, manifest, source)["review_id"], default["review_id"])
            self.assertEqual(call(pair, manifest, source, "legacy_single_reply")["review_id"], single["review_id"])
            with self.assertRaisesRegex(review.UnsupportedTransitionSource, "unsupported_transition_review_lane"):
                call(pair, manifest, source, "rovent_raw3")

    def test_duplicate_same_pair_and_lane_is_ambiguous_even_with_a_different_review_id(self):
        first, pair, manifest, source = fixture()
        second = copy.deepcopy(first)
        second["review_id"] = "another-label-cannot-disambiguate"
        with patch.object(review, "REVIEWED_CLOSURE_PROFILES", (first, second)):
            with self.assertRaisesRegex(ValueError, "ambiguous duplicate literal transition source review"):
                call(pair, manifest, source)

    def test_original_manifest_pin_types_and_exact_fields_fail_before_review_selection(self):
        profile, pair, manifest, source = fixture()
        for change in ({"bytes": True}, {"bytes": -1}, {"bytes": 64 * 1024 * 1024 + 1},
                       {"sha256": "AB" * 32}, {"authority": True}):
            bad = copy.deepcopy(manifest)
            next(entry for entry in bad["files"] if entry["path"] == _HELPER).update(change)
            with self.subTest(change=change), patch.object(review, "REVIEWED_CLOSURE_PROFILES", (profile,)):
                with self.assertRaises(ValueError):
                    call(pair, bad, source)

    def test_literal_profile_roster_and_closure_seal_are_not_caller_registration_options(self):
        profile, pair, manifest, source = fixture()
        for change in ({"closure_sha256": h("unreviewed closure")}, {"actual_execution": True}):
            malformed = {**profile, **change}
            with self.subTest(change=change), patch.object(review, "REVIEWED_CLOSURE_PROFILES", (malformed,)):
                with self.assertRaises(ValueError):
                    call(pair, manifest, source)
        with patch.object(review, "REVIEWED_CLOSURE_PROFILES", [profile]):
            with self.assertRaisesRegex(ValueError, "closed literal transition review registry"):
                call(pair, manifest, source)
        with patch.object(review, "REVIEWED_CLOSURE_PROFILES", (profile,)):
            with self.assertRaises(TypeError):
                review.review_transition_sources(pair=pair, manifest=manifest, source=source,
                    lane="default_disabled", register_current_files=True)

    def test_transition_pair_and_manifest_need_the_same_original_bytes_sha_pin(self):
        profile, pair, manifest, source = fixture()
        next(entry for entry in manifest["files"] if entry["path"] == _NATIVE)["bytes"] += 1
        with patch.object(review, "REVIEWED_CLOSURE_PROFILES", (profile,)):
            with self.assertRaisesRegex(ValueError, "source pair differs from its manifest pin"):
                call(pair, manifest, source)


if __name__ == "__main__":
    unittest.main()
