"""Synthetic external Collection wire joins, never a live process or Rules proof.

The fixture preserves a submitted P input and its conditional Collection teacher
receipt. Its researcher profile is a separate original-lock pin. No executable,
profile file, runtime, checkpoint or publication is opened by these tests.
"""
import copy
import hashlib
import importlib.abc
import json
import sys
import unittest


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("external root wire tests cannot import neural runtimes")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_external_root as external
    from rz_pals_model import rovent_corpus as corpus
    from rz_pals_model import rovent_historical_corpus as historical
    from test_rovent_collection_teacher import (
        fixture as legacy_fixture, submitted_fixture, seal_descriptor,
        validate_binding, validate_head, validate_row,
    )
finally:
    sys.meta_path.remove(_guard)


ROOT = "rz-rovent-collection-root-admission/2"
SOURCE = "rz-rovent-collection-profiled-external-source/1"
PROFILE = "rz-rovent-external-engine-profile/1"
PROFILE_CANONICAL = "rz-rovent-external-engine-profile-canonical/1"
CAPABILITY = "rz-rovent-external-capabilities/1"
SEMANTICS = ("uci-native-cp-or-signed-mate-moves;root-side-to-move;reported-bounds;"
    "selectivity-unknown;no-cp-wdl-calibration;no-rules-proof")
FAMILIES = {
    "stockfish19_embedded_nnue": {
        "name": "Stockfish 19", "version": "19", "repository": "https://github.com/official-stockfish/Stockfish",
        "commit": "edb0d9db6731067ec50ce619ff372b463bc4dd5d", "license": "GPL-3.0-or-later",
        "caps": [True, True, True, True, False, False],
    },
    "reckless090_embedded_nnue": {
        "name": "Reckless 0.9.0", "version": "0.9.0", "repository": "https://github.com/codedeliveryservice/Reckless",
        "commit": "0e92358f5acd66e5ac77b1bf558202e47c515435", "license": "AGPL-3.0-only",
        "caps": [False, False, False, True, False, False],
    },
}


def h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def wire(domain, body):
    """Independent sorted serde-compatible metadata framing for this fixture."""
    raw = json.dumps([domain, body], sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def source_aggregate(commit, components):
    body = b"rz-rovent-source-aggregate/1\0" + commit.encode("ascii") + len(components).to_bytes(4, "little")
    for name, identity in sorted(components.items()):
        encoded = name.encode("utf-8")
        body += len(encoded).to_bytes(4, "little") + encoded + identity.encode("ascii")
    return hashlib.sha256(body).hexdigest()


def profile_fixture(engine="stockfish19_embedded_nnue"):
    """Values are independently copied from the two closed Rust family DTOs."""
    family = FAMILIES[engine]
    source = family["repository"] + "/tree/" + family["commit"]
    binary = {"path": "/opt/synthetic-collection/" + engine, "sha256": h(engine + " synthetic binary"),
        "bytes": 16, "source": source, "license": family["license"]}
    options = {"Threads": "1", "Hash": "256", "MultiPV": "1", "UCI_Chess960": "false"}
    if engine == "stockfish19_embedded_nnue":
        options.update({"Move Overhead": "10", "Skill Level": "20", "UCI_LimitStrength": "false",
            "Ponder": "false", "NumaPolicy": "none", "SyzygyPath": "", "SyzygyProbeLimit": "0", "nodestime": "0"})
    else:
        options.update({"MoveOverhead": "10", "Minimal": "false"})
    profile = {"schema": PROFILE, "engine": engine, "purpose": "research_checker", "program": binary["path"],
        "working_directory": "/", "arguments": [], "identity": {
            "adapter_semantics": SEMANTICS, "binary_sha256": binary["sha256"],
            "launch_arguments_sha256": hashlib.sha256(b"rz-external-uci-arguments/1\0" + bytes(8)).hexdigest(),
            "declared_name": family["name"], "declared_version": family["version"], "declared_source": source,
            "declared_license": family["license"], "options": options, "assets": [],
            "model_metadata": {"weights_sha256": None, "training": "unknown", "declared_rights": None, "precision": None}},
        "expected_uci_name": family["name"], "max_depth": 32, "max_prefix_plies": 256,
        "handshake_timeout_ms": 1000, "max_task_wall_time_ms": 2000, "stop_grace_ms": 500,
        "shutdown_grace_ms": 500, "max_output_bytes": 1048576, "max_line_bytes": 4096}
    # A file hash, sorted semantic hash and flat capability hash are three
    # different facts. Pretty file bytes deliberately differ from canonical.
    raw = (json.dumps(profile, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
    capability = [CAPABILITY, engine, "research_checker", family["commit"], family["name"], SEMANTICS, family["caps"]]
    expected = {"source_commit": family["commit"], "requested_threads": 1, "requested_hash_mib": 256,
        "canonical_sha256": wire(PROFILE_CANONICAL, profile),
        "capability_sha256": hashlib.sha256(json.dumps(capability, ensure_ascii=False, allow_nan=False,
            separators=(",", ":")).encode("utf-8")).hexdigest()}
    pin = {"domain": PROFILE, "profile": {"path": "/opt/synthetic-collection/" + engine + ".json",
        "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "source": source, "license": family["license"]},
        "purpose": "research_checker", "binary": copy.deepcopy(binary), **expected}
    return profile, binary, pin, expected


def reseal(manifest, record):
    """Seal new fixture bytes only; never call the production root hash path."""
    native = record["native"]
    retired = native["raw_retirement"]
    root = retired["root_admission"]
    root["native_origin_sha256"] = wire("rz-rovent-collection-native-origin/1", root["native_origin"])
    if "external_source" in root:
        root["external_source_sha256"] = wire(SOURCE, root["external_source"])
    retired["root_admission_sha256"] = wire(root["schema"], root)
    domain = "rz-rovent-raw-record/3"
    native["raw_submission_sha256"] = wire(domain, native["raw_submission"])
    native["raw_completion"]["submitted_sha256"] = native["raw_submission_sha256"]
    native["raw_completion_sha256"] = wire(domain, native["raw_completion"])
    retired.update(submitted_sha256=native["raw_submission_sha256"], completed_sha256=native["raw_completion_sha256"])
    native["raw_retirement_sha256"] = wire(domain, retired)
    for head, source in record["head_sources"].items():
        source["raw_receipt_sha256"] = wire("rz-rovent-head-source/1", source["raw_receipt"])
        if record["targets"][head] is not None:
            record["targets"][head]["raw_evidence_sha256"] = source["raw_receipt_sha256"]
    origin = {key: manifest[key] for key in ("source_sha256", "recipe_sha256", "stage_id", "round", "round_id", "student", "learner")}
    origin.update(record["lineage"])
    origin.update(input_sha256=record["input_sha256"], **{key: record["input"][key] for key in
        ("game_id", "opening_id", "line_genealogy_id", "transposition_sha256")})
    record["origin_sha256"] = wire("rz-rovent-training-origin/1", origin)
    projected = {**record, "record_sha256": "", "input": {"sealed_input_sha256": record["input_sha256"]}}
    record["record_sha256"] = wire("rz-rovent-training-record/2", projected)


def fixture(engine="stockfish19_embedded_nnue"):
    manifest, record = submitted_fixture()
    profile, binary, profile_pin, expected = profile_fixture(engine)
    components = {"synthetic-external-collection": h("independently compiled Collection source fixture")}
    source_sha = source_aggregate(manifest["source_commit"], components)
    manifest["source_sha256"] = record["native"]["source_sha256"] = source_sha
    execution = {"domain": "rz-rovent-execution/1", "contract_revision": "0.1", "run_id": "synthetic-external-root-only",
        "purpose": "collection", "engine": {"source_commit": manifest["source_commit"], "source_components": components,
            "researcher": {"provider": "external", "profile": profile_pin}},
        # These metadata-only declarations do not claim admission by the full
        # Rust manifest/resource/model validator or a new live InputLock.
        "resources": {}, "coordinator": {}, "wall_time_max_ms": 1000, "cleanup_max_ms": 500,
        "output_bytes_max": 1048576, "agent_renewable": True, "provenance": []}
    lock = {"domain": "rz-rovent-input-lock/1", "manifest": execution,
        "canonical_sha256": wire("rz-rovent-execution-canonical/1", execution)}
    native = record["native"]
    root = native["raw_retirement"]["root_admission"]
    root["schema"] = ROOT
    origin = root["native_origin"]
    origin.update(mode="HybridPcr", input_lock_sha256=lock["canonical_sha256"], source_components=components, source_sha256=source_sha)
    descriptor = root["root_descriptor"]
    descriptor["input"]["startup_lock_sha256"] = lock["canonical_sha256"]
    descriptor_sha = seal_descriptor(descriptor)
    root.update(root_descriptor_sha256=descriptor_sha, root_budget_sha256=descriptor["root_budget_sha256"])
    for name in ("raw_submission", "raw_completion", "raw_retirement"):
        native[name]["source_sha256"] = source_sha
    binding = record["head_sources"]["policy"]["raw_receipt"]["binding"]
    binding.update(root_descriptor=copy.deepcopy(descriptor), root_descriptor_sha256=descriptor_sha)
    binding["input_scope"].update(source_sha256=source_sha, input_lock=lock["canonical_sha256"],
        root_admission=descriptor_sha, root_budget=descriptor["root_budget_sha256"])
    root["external_source"] = {"schema": SOURCE, "root_source_sequence": 0, "worker_owner": 0, "clock_epoch": root["deadline_clock"],
        "authority": {key: root[key] for key in ("process_epoch", "game_generation", "root_generation", "implementation_sha256")},
        "engine_profile": engine, "purpose": "research_checker", "process": {"pid": 321, "process_group": 321, "proc_start_ticks": 777},
        "profile": profile, "profile_file_sha256": profile_pin["profile"]["sha256"], "profile_file_bytes": profile_pin["profile"]["bytes"],
        "profile_canonical_sha256": expected["canonical_sha256"], "profile_capability_sha256": expected["capability_sha256"],
        "binary": binary, "startup_input_lock_sha256": lock["canonical_sha256"],
        "observed_uci": {"name": profile["expected_uci_name"], "author": None},
        "options_application_observed": None, "model_loading_observed": None}
    reseal(manifest, record)
    asset_registration = {"collection_source_commit": manifest["source_commit"], "collection_source_sha256": source_sha,
        "collection_runtime_library_sha256": manifest["runtime_sha256"], "role_bindings": {"proposer": {
            "model_semantics_sha256": manifest["model_semantics_sha256"], "model_identity_sha256": native["model_sha256"],
            "weights_sha256": origin["weights_sha256"], "encoding_sha256": native["encoding_sha256"]}}}
    return manifest, record, lock, asset_registration, expected


def root_of(record):
    return record["native"]["raw_retirement"]["root_admission"]


class ExternalCollectionRootTests(unittest.TestCase):
    def setUp(self):
        sys.meta_path.insert(0, _guard)

    def tearDown(self):
        sys.meta_path.remove(_guard)

    def test_two_official_profiles_use_independent_file_semantic_and_flat_capability_hashes(self):
        for engine in FAMILIES:
            with self.subTest(engine=engine):
                profile, binary, pin, expected = profile_fixture(engine)
                before = copy.deepcopy(profile)
                self.assertEqual(external.validate_external_profile(profile), expected)
                self.assertEqual(profile, before)
                self.assertEqual(len(profile), 16)
                self.assertEqual(len(pin), 9)
                self.assertNotEqual(pin["profile"]["sha256"], expected["canonical_sha256"])
                self.assertNotEqual(expected["capability_sha256"], expected["canonical_sha256"])
                self.assertEqual(pin["binary"], binary)

    def test_raw3_external_root_is_consumed_by_native_teacher_and_historical_asset_joins(self):
        for engine in FAMILIES:
            with self.subTest(engine=engine):
                manifest, record, lock, registration, expected = fixture(engine)
                before = copy.deepcopy((manifest, record, lock, registration))
                self.assertIs(corpus._native(manifest, record), record["native"])
                validate_binding(manifest, record, submitted=True)
                validate_head(manifest, record)
                self.assertIsNotNone(validate_row(manifest, record)["targets"]["policy"])
                root = root_of(record)
                self.assertEqual(external.validate_external_source(root), expected)
                self.assertIs(external.join_locked_external_source(root, lock), root["external_source"])
                original = historical._input_lock(lock, lock["canonical_sha256"])
                admission_sha = record["native"]["raw_retirement"]["root_admission_sha256"]
                self.assertEqual(historical._join_submitted_record(registration, manifest, record,
                    {admission_sha: original}), admission_sha)
                self.assertEqual(len(root), 21)
                self.assertEqual(len(root["external_source"]), 18)
                self.assertEqual(root["external_source"]["root_source_sequence"], 0)
                self.assertEqual(root["external_source"]["worker_owner"], 0)
                self.assertEqual((manifest, record, lock, registration), before)
                self.assertIsNone(record["targets"]["wdl"])

    def test_original_root1_raw2_and_raw3_bodies_and_seals_are_unchanged(self):
        for factory in (legacy_fixture, submitted_fixture):
            with self.subTest(factory=factory.__name__):
                manifest, record = factory()
                before = copy.deepcopy(record)
                root = root_of(record)
                self.assertEqual(root["schema"], "rz-rovent-collection-root-admission/1")
                self.assertEqual(record["native"]["raw_retirement"]["root_admission_sha256"], wire(root["schema"], root))
                self.assertNotIn("external_source", root)
                corpus._native(manifest, record)
                validate_binding(manifest, record, submitted=factory is submitted_fixture)
                self.assertEqual(record, before)

    def test_root2_is_explicit_raw3_only_and_its_descriptor_pair_is_required(self):
        for mutation, error in (("raw2", "requires explicit submitted input and native raw /3"),
                ("missing_descriptor", "external Collection root /2 fields mismatch"),
                ("extra", "external Collection root /2 fields mismatch"),
                ("root_domain", "external Collection root /2 original seal mismatch")):
            with self.subTest(mutation=mutation):
                manifest, record, *_ = fixture()
                root = root_of(record)
                if mutation == "raw2":
                    record["native"]["raw_completion"]["schema"] = "rz-rovent-native-owner-event/2"
                elif mutation == "missing_descriptor":
                    del root["root_descriptor"]
                elif mutation == "extra":
                    root["readyok_proves_options"] = True
                else:
                    root["schema"] = "rz-rovent-collection-root-admission/1"
                reseal(manifest, record)
                with self.assertRaisesRegex(ValueError, error):
                    external.validate_external_root(manifest, record, record["native"]["raw_retirement"])

    def test_root2_preserves_pcr_mode_instead_of_accepting_a_pc_only_origin(self):
        for mode in ("HybridPc", "CpuPc"):
            with self.subTest(mode=mode):
                manifest, record, *_ = fixture()
                root_of(record)["native_origin"]["mode"] = mode
                reseal(manifest, record)
                with self.assertRaisesRegex(ValueError, "requires an original P/C/R mode"):
                    corpus._native(manifest, record)

    def test_rehashed_source_still_rejects_crossed_scope_process_and_observed_fact_claims(self):
        cases = (
            ("domain", lambda s: s.update(schema="rz-rovent-collection-profiled-external-source/0"), "source domain"),
            ("extra", lambda s: s.update(readyok=True), "external Collection root source fields mismatch"),
            ("purpose", lambda s: s.update(purpose="teacher"), "source domain"),
            ("clock", lambda s: s.update(clock_epoch=2), "worker clock"),
            ("authority", lambda s: s["authority"].update(root_generation=2), "root authority"),
            ("authority_type", lambda s: s["authority"].update(process_epoch=True), "root authority"),
            ("pid", lambda s: s["process"].update(pid=0), "positive external root observation"),
            ("group", lambda s: s["process"].update(process_group=0), "positive external root observation"),
            ("start", lambda s: s["process"].update(proc_start_ticks=0), "positive external root observation"),
            ("engine", lambda s: s.update(engine_profile="reckless090_embedded_nnue"), "profile/capability/startup"),
            ("canonical", lambda s: s.update(profile_canonical_sha256=s["profile_file_sha256"]), "profile/capability/startup"),
            ("capability", lambda s: s.update(profile_capability_sha256=s["profile_canonical_sha256"]), "profile/capability/startup"),
            ("startup", lambda s: s.update(startup_input_lock_sha256=h("other lock")), "profile/capability/startup"),
            ("binary", lambda s: s["binary"].update(sha256=h("other executable")), "binary pin"),
            ("license", lambda s: s["binary"].update(license="MIT"), "binary pin"),
            ("name", lambda s: s["observed_uci"].update(name="another engine"), "actual UCI name"),
            ("options", lambda s: s.update(options_application_observed=True), "observations must remain unknown"),
            ("model", lambda s: s.update(model_loading_observed=False), "observations must remain unknown"),
        )
        for name, mutate, error in cases:
            with self.subTest(name=name):
                manifest, record, *_ = fixture()
                mutate(root_of(record)["external_source"])
                reseal(manifest, record)
                with self.assertRaisesRegex(ValueError, error):
                    external.validate_external_source(root_of(record))

    def test_observed_author_preserves_optional_empty_tab_and_utf8_byte_bound(self):
        for author in (None, "", "authors\twith tab", "é" * 128):
            with self.subTest(author=author):
                manifest, record, *_ = fixture()
                root_of(record)["external_source"]["observed_uci"]["author"] = author
                reseal(manifest, record)
                before = copy.deepcopy(record)
                corpus._native(manifest, record)
                self.assertEqual(record, before)
        for author in ("author\nsecond line", "a" * 257, "é" * 128 + "a"):
            with self.subTest(rejected=author):
                manifest, record, *_ = fixture()
                root_of(record)["external_source"]["observed_uci"]["author"] = author
                reseal(manifest, record)
                with self.assertRaisesRegex(ValueError, "bounded external root metadata text"):
                    external.validate_external_source(root_of(record))

    def test_closed_profile_rejects_changed_options_sources_assets_and_resource_bounds(self):
        cases = (
            ("family", lambda p: p.update(engine="other"), "embedded research-checker profile"),
            ("purpose", lambda p: p.update(purpose="teacher"), "embedded research-checker profile"),
            ("arguments", lambda p: p.update(arguments=["--arbitrary"]), "embedded research-checker profile"),
            ("relative", lambda p: p.update(program="bin/stockfish"), "normalized absolute Linux path"),
            ("dotdot", lambda p: p.update(program="/bin/../stockfish"), "normalized absolute Linux path"),
            ("threads", lambda p: p["identity"]["options"].update(Threads="0"), "registered bounds"),
            ("thread_spelling", lambda p: p["identity"]["options"].update(Threads="01"), "registered bounds"),
            ("thread_type", lambda p: p["identity"]["options"].update(Threads=1), "bounded teacher option declaration required"),
            ("hash", lambda p: p["identity"]["options"].update(Hash="262145"), "registered bounds"),
            ("multipv", lambda p: p["identity"]["options"].update(MultiPV="5"), "registered bounds"),
            ("ponder", lambda p: p["identity"]["options"].update(Ponder="true"), "required option profile"),
            ("extra_option", lambda p: p["identity"]["options"].update(Unexpected="false"), "required option profile"),
            ("source", lambda p: p["identity"].update(declared_source="https://example.invalid/source"), "source/name/license/assets"),
            ("license", lambda p: p["identity"].update(declared_license="MIT"), "source/name/license/assets"),
            ("tab_declaration", lambda p: p["identity"].update(declared_name="Stockfish\t19"), "text|source/name/license/assets"),
            ("asset", lambda p: p["identity"]["assets"].append({"purpose": "separate_nnue", "sha256": h("extra")}),
                "source/name/license/assets"),
            ("precision", lambda p: p["identity"]["model_metadata"].update(precision="FP32"), "source/name/license/assets"),
            ("stop", lambda p: p.update(max_task_wall_time_ms=p["stop_grace_ms"]), "original stop reserve"),
            ("line", lambda p: p.update(max_output_bytes=1, max_line_bytes=2), "line cap exceeds"),
        )
        for name, mutate, error in cases:
            with self.subTest(name=name):
                profile, *_ = profile_fixture()
                mutate(profile)
                with self.assertRaisesRegex(ValueError, error):
                    external.validate_external_profile(profile)

    def test_original_lock_researcher_profile_pins_cannot_be_replaced_with_semantic_aliases(self):
        cases = (
            ("provider", lambda r: r.update(provider="own"), "original external researcher lock"),
            ("purpose", lambda r: r["profile"].update(purpose="teacher"), "source differs from original lock"),
            ("raw_profile", lambda r: r["profile"]["profile"].update(sha256=r["profile"]["canonical_sha256"]), "source differs from original lock"),
            ("raw_bytes", lambda r: r["profile"]["profile"].update(bytes=1), "source differs from original lock"),
            ("binary_path", lambda r: r["profile"]["binary"].update(path="/other/binary"), "source differs from original lock"),
            ("commit", lambda r: r["profile"].update(source_commit="f" * 40), "source differs from original lock"),
            ("threads", lambda r: r["profile"].update(requested_threads=2), "source differs from original lock"),
            ("hash", lambda r: r["profile"].update(requested_hash_mib=512), "source differs from original lock"),
            ("capability", lambda r: r["profile"].update(capability_sha256=r["profile"]["canonical_sha256"]), "source differs from original lock"),
        )
        for name, mutate, error in cases:
            with self.subTest(name=name):
                _manifest, record, lock, *_ = fixture()
                mutate(lock["manifest"]["engine"]["researcher"])
                # The pure join checks the registered fields even when handed
                # an outer lock already read by its owner; it cannot reseal it.
                with self.assertRaisesRegex(ValueError, error):
                    external.join_locked_external_source(root_of(record), lock)
        _manifest, record, lock, *_ = fixture()
        lock["manifest"]["run_id"] = "changed-without-original-lock-seal"
        with self.assertRaisesRegex(ValueError, "input lock canonical metadata seal mismatch"):
            historical._input_lock(lock, root_of(record)["native_origin"]["input_lock_sha256"])

    def test_historical_asset_join_keeps_original_publication_lock_and_role_pins(self):
        for mutation, error in (("no_lock", "lacks its published original input lock"),
                ("source", "lock/source component aggregate mismatch"),
                ("runtime", "source/runtime library domain mismatch"),
                ("weights", "role model/weights/encoding")):
            with self.subTest(mutation=mutation):
                manifest, record, lock, registration, *_ = fixture()
                admission_sha = record["native"]["raw_retirement"]["root_admission_sha256"]
                admissions = {admission_sha: lock}
                if mutation == "no_lock":
                    admissions.clear()
                elif mutation == "source":
                    registration["collection_source_sha256"] = h("other source aggregate")
                elif mutation == "runtime":
                    registration["collection_runtime_library_sha256"] = h("another runtime file")
                else:
                    registration["role_bindings"]["proposer"]["weights_sha256"] = h("other weights")
                with self.assertRaisesRegex(ValueError, error):
                    historical._join_submitted_record(registration, manifest, record, admissions)


if __name__ == "__main__":
    unittest.main()
