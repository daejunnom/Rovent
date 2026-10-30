"""Opt-in native recheck authority, separate from the Disabled slot witness.

This source unit checks independently pinned caller/build/source facts, an
already checked current ordinary Repair, the actual full Repair call chain,
prepaid Reply attribution, endpoint store provenance and registered Rules replay.
The closed source profile table contains literal pairs enrolled only after
independent source review. No current file hash is automatically enrolled.

The whole-witness factory also binds the actual engine descriptor and restricted
publication trace. A policy marker or the helpers in this file alone grants none
of those authorities. This module performs no model, process, Rules, training
or optimizer operations.
Existing imports can transitively import Torch; that is not a model execution.
"""
import copy
import struct

from . import native_divergence as native
from . import repair_context as repair
from . import training
from .native_slot_repair import _finite_bits
from .semantic_verifier import CheckedSemanticInput, byte_pin, canonical, digest, _fields, _int, _parse, _sha


ANCHOR_SCHEMA = "rz-pals-native-recheck-repair-anchor/1"
WITNESS_SCHEMA = "rz-pals-native-recheck-witness/1"
SCOPE = "conditional_same_repaired_line_observation_only"
BUILD_SCHEMA = "rz-pals-root-collector-binary-build-registration/1"
POLICY_REGISTRATION_SCHEMA = "rz-pals-native-refinement-registration/1"
TRACE_DOMAIN = "rz-pals-native-post-repair-recheck/1"
TRACE_ARTIFACT = "native-recheck-traces.jsonl"
MAX_POLICY_REGISTRATION_BYTES = 32 * 1024
MAX_BYTES = native.MAX_BYTES
_FACTORY = object()
_SOURCE_PATHS = ("crates/rz-search/src/pals/engine.rs", "crates/rz-arena/src/pals_collect/native.rs")
_SOURCE_RAW_NAMES = ("build_registration", "source_manifest", "engine_source", "native_source")
# Root supplies literal profiles only AFTER final source freeze and review.
# Each entry is (review_name, {exact source path: {bytes, sha256}}). There is no
# callback or public enrollment API, and no comparison against live source files.
_REVIEWED_OPT_IN_SOURCE_PROFILES = (
    ("opt_in_observer_0302e49b_native_0b056c4f", {
        "crates/rz-search/src/pals/engine.rs": {
            "bytes": 343988, "sha256": "0302e49b6a9784641ca90aba17490d404b793098f38836b7b3cf42a12ee131d1"},
        "crates/rz-arena/src/pals_collect/native.rs": {
            "bytes": 153791, "sha256": "0b056c4f2aada6380b1345a067e4d2c139118a8636ad7dc6f1a67a0dc0c55144"},
    }),
    ("opt_in_search_return_0302e49b_native_9653ed5e", {
        "crates/rz-search/src/pals/engine.rs": {
            "bytes": 343988, "sha256": "0302e49b6a9784641ca90aba17490d404b793098f38836b7b3cf42a12ee131d1"},
        "crates/rz-arena/src/pals_collect/native.rs": {
            "bytes": 193797, "sha256": "9653ed5ef33eced92b400aef40a7ad018a2668505d6a14568ae9d80e69dc65df"},
    }),
    ("opt_in_search_return_clock_lint_2c2bfb1c", {
        "crates/rz-search/src/pals/engine.rs": {
            "bytes": 343988, "sha256": "0302e49b6a9784641ca90aba17490d404b793098f38836b7b3cf42a12ee131d1"},
        "crates/rz-arena/src/pals_collect/native.rs": {
            "bytes": 193840, "sha256": "2c2bfb1c8717a68429d315fa576268ba95b9b6106caed57400b885f656d04ae7"},
    }),
    # Module/comment-only engine successor; native does not call standalone
    # replay, so this profile admits no replay causal, witness or utility scope.
    ("opt_in_replay_module_0a0a4800_native_2c2bfb1c", {
        "crates/rz-search/src/pals/engine.rs": {
            "bytes": 344097, "sha256": "0a0a4800e940700721ec856ab86c8d8dbb581f6291fd78976ce1f918ea5a40c3"},
        "crates/rz-arena/src/pals_collect/native.rs": {
            "bytes": 193840, "sha256": "2c2bfb1c8717a68429d315fa576268ba95b9b6106caed57400b885f656d04ae7"},
    }),
    # Reviewed old single-Reply branch only. The collector now explicitly
    # refuses actual C continuation at registration and observer boundaries.
    # This profile does not admit that new lane or its multiple Reply calls.
    ("opt_in_single_reply_guarded_fab843bc_native_e25fc205", {
        "crates/rz-search/src/pals/engine.rs": {
            "bytes": 354492, "sha256": "fab843bc71b7e4b3584a437b7b4d89515be8b7ae12a2ba36ade83086c74c6618"},
        "crates/rz-arena/src/pals_collect/native.rs": {
            "bytes": 195372, "sha256": "e25fc205f895825607a15292a5c23ed6ccfe425f4c6134201a52fe68ca13df1c"},
    }),
    # Reviewed legacy single-Reply branch only; the new observer has a
    # separate policy, registration and artifact, rejected by this consumer.
    ("opt_in_legacy_single_reply_with_multi_observer_b813828e", {
        "crates/rz-search/src/pals/engine.rs": {'bytes': 354492, 'sha256': 'fab843bc71b7e4b3584a437b7b4d89515be8b7ae12a2ba36ade83086c74c6618'},
        "crates/rz-arena/src/pals_collect/native.rs": {'bytes': 212711, 'sha256': 'b813828e9e469feb0769a35bb721f8fbfdf4c7dace9da92907372a757b822ba6'},
    }),
    # Old single-Reply only; the separate v2 continuation observation is refused.
    ("opt_in_single_reply_continuation_selection_v2_eea0654a", {
        "crates/rz-search/src/pals/engine.rs": {'bytes': 355881, 'sha256': '32e393b4ad83bfd33a1b2205cd110cdc0cf007909e7e9c7f617819ccf5c4e93c'},
        "crates/rz-arena/src/pals_collect/native.rs": {'bytes': 220199, 'sha256': 'eea0654a5c11d6e0d69b5c5df71363a8e2d8f4fb7ad02cf5d766fb4b06ec456f'},
    }),
    # Reviewed same-line single-Reply successor only. The actual selected
    # getters now derive their own closed tuple; the legacy tuple is identical.
    # Continuation remains a separate registration/artifact rejected below.
    ("opt_in_single_reply_selected_identity_8886c0e0", {
        "crates/rz-search/src/pals/engine.rs": {
            "bytes": 355881, "sha256": "32e393b4ad83bfd33a1b2205cd110cdc0cf007909e7e9c7f617819ccf5c4e93c"},
        "crates/rz-arena/src/pals_collect/native.rs": {
            "bytes": 221690, "sha256": "8886c0e01d0c753fa0a1b5256e36f2230d41f5e561fca8ac7ae8f895b9e0813c"},
    }),
    # Reviewed legacy same-line branch only. Native changed only continuation
    # revision tests, and the engine is unchanged. The separately changed C
    # module/trace remains excluded by this consumer's policy and artifact.
    ("opt_in_single_reply_continuation_revision_fixture_a286a240", {
        "crates/rz-search/src/pals/engine.rs": {
            "bytes": 355881, "sha256": "32e393b4ad83bfd33a1b2205cd110cdc0cf007909e7e9c7f617819ccf5c4e93c"},
        "crates/rz-arena/src/pals_collect/native.rs": {
            "bytes": 224380, "sha256": "a286a2403317be7f5f058b65488e7c9722d2b7f4b9bf683ea95e5c9f51a7aaf8"},
    }),
    # Independently reviewed frozen followup core, old same-line single-Reply
    # constructor and observer only. Archive is off in that collector path;
    # new V4/frozen/queue policies cannot use its legacy registration. The exact
    # policy, raw CP completion and Rules gates below remain authoritative.
    # No ModelWdl, retired-task, utility or actual execution scope is enrolled.
    ("opt_in_single_reply_followup_core_b86af1f0_native_1ba6d1e5", {
        "crates/rz-search/src/pals/engine.rs": {
            "bytes": 374235, "sha256": "b86af1f0e55f55add0d4cca277772f8d1c8edf3b3d93d147629c4ab6f881103a"},
        "crates/rz-arena/src/pals_collect/native.rs": {
            "bytes": 233158, "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9"},
    }),
    # Independently reviewed lint-only successor, with the same closed legacy
    # single-Reply scope. PalsError identity conversions/Result wrappers change
    # no observation, policy or authority; prior source pairs remain retained.
    ("opt_in_single_reply_followup_core_lint_4729c359_native_1ba6d1e5", {
        "crates/rz-search/src/pals/engine.rs": {
            "bytes": 374216, "sha256": "4729c3595731d82e7252dd4aec316bf051092449d95f4fe8a4a83017038ee018"},
        "crates/rz-arena/src/pals_collect/native.rs": {
            "bytes": 233158, "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9"},
    }),
    # Reviewed controlled-allocation successor in the same archive-off legacy
    # single-Reply scope. Accepted Repair, original controls, Rules replay and
    # completed OwnCPU comparison remain required by this collector/consumer.
    # Raw retention, archive/V4 selection and new endpoints gain no authority.
    ("opt_in_single_reply_followup_controlled_core_752d328a_native_1ba6d1e5", {
        "crates/rz-search/src/pals/engine.rs": {
            "bytes": 382810, "sha256": "752d328ae53dc1b4c06a1f3b7cbbd311834dbb59961e598c4752d8fcb9222cf4"},
        "crates/rz-arena/src/pals_collect/native.rs": {
            "bytes": 233158, "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9"},
    }),
    # Re-reviewed stopped-request/raw-prefix successor, legacy SingleReply only.
    # Rules-only stopped results and a legal local prefix cannot manufacture an
    # accepted Repair record/revision/endpoint or a completed OwnCPU comparison.
    # Typed deadlines and original policy/value gates grant no V4/ModelWdl scope.
    ("opt_in_single_reply_followup_controlled_fallback_4770a553_native_1ba6d1e5", {
        "crates/rz-search/src/pals/engine.rs": {
            "bytes": 391894, "sha256": "4770a55361a994210a2238240ed4d2dea1a24d7a8b85524cc7c2db31d19f5e19"},
        "crates/rz-arena/src/pals_collect/native.rs": {
            "bytes": 233158, "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9"},
    }),
    # Reviewed frozen live-binding successor, archive-off SingleReply only.
    # Original controls, exact accepted Repair and Rules-replayed endpoint
    # provenance remain required. External CP and ModelWdl are separate value
    # namespaces; neither raw prefixes nor retired tasks gain completed OwnCPU
    # authority. This literal pair certifies no build, execution or target.
    ("opt_in_single_reply_followup_live_binding_004db31f_native_1ba6d1e5", {
        "crates/rz-search/src/pals/engine.rs": {
            "bytes": 396086, "sha256": "004db31f9be5133bd1925b373e6df60e97da3d50979a7eb5a031b1dad4ca397b"},
        "crates/rz-arena/src/pals_collect/native.rs": {
            "bytes": 233158, "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9"},
    }),
    # Reviewed observed-progress successor in the same archive-off SingleReply
    # scope. Old constructors still select ReplaceOnFresh; held-parent scheduling
    # requires the separate IterativeFrozenModelWdlV2/PausedStack constructor.
    # The move-only progress wrapper preserves the legacy publication boundary,
    # and the collector bytes are unchanged. This exact pair grants no held-parent,
    # V4/ModelWdl, physical execution, cost, utility or training authority.
    ("opt_in_single_reply_followup_observed_progress_4b272ed8_native_1ba6d1e5", {
        "crates/rz-search/src/pals/engine.rs": {
            "bytes": 402637, "sha256": "4b272ed825d98f51ace408b2798093e88487f6b29f94e38b9d2a753f8fc263d1"},
        "crates/rz-arena/src/pals_collect/native.rs": {
            "bytes": 233158, "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9"},
    }),
    # Reviewed ordered-scope successor, still archive-off legacy SingleReply.
    # ReplaceOnFresh and public FIFO retain their previous behavior. The new
    # nonrecursive priority drain requires both ResumeAfterAcceptedRepairC and
    # IterativeFrozenModelWdlV2, selected by the separate held-parent-v2
    # constructor. The unchanged collector never selects it here. This exact
    # pair grants no held-parent execution, V4/ModelWdl, cost or training proof.
    ("opt_in_single_reply_followup_ordered_scope_70d4bae5_native_1ba6d1e5", {
        "crates/rz-search/src/pals/engine.rs": {
            "bytes": 402843, "sha256": "70d4bae5ad4e19e86137e47cc66922182c5232cbe35b9b30d42bf43ec6850171"},
        "crates/rz-arena/src/pals_collect/native.rs": {
            "bytes": 233158, "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9"},
    }),
    # Independently reviewed optional-review successor, legacy SingleReply only.
    # Old constructors leave review_index/review_policy unset, so full records,
    # accepted Repair, original controls and completed OwnCPU/Rules gates remain.
    # The Fresh-pair fallback keeps the original endpoint call order; the native
    # Err/Review classification cannot issue a completed or successful result.
    # This pair admits no selected-view/adaptive, archive, V4/ModelWdl or actual
    # continuation scope, and certifies no build, execution, utility or training.
    ("opt_in_single_reply_followup_optional_review_ce40b6f9_native_02af59b9", {
        "crates/rz-search/src/pals/engine.rs": {
            "bytes": 414372, "sha256": "ce40b6f9ff8d13a41da8c9482b48528c1437896e9571c9630790c4b11b262726"},
        "crates/rz-arena/src/pals_collect/native.rs": {
            "bytes": 233208, "sha256": "02af59b9edbc350e6e1a204eb6543033bb4a859729eb2ed79c8b470cc6823a01"},
    }),
    # Reviewed compiled-source inventory-only successor, same legacy SingleReply
    # policy. Only engine's implementation-digest source tuple changes; collector
    # and runtime control bodies do not. No adaptive consumption, selected view,
    # build, execution, value, utility or training authority is admitted here.
    ("opt_in_single_reply_followup_compiled_consumption_0ba7c9e4_native_02af59b9", {
        "crates/rz-search/src/pals/engine.rs": {
            "bytes": 414554, "sha256": "0ba7c9e41b106f35516b8b234872a839f5b2f4227153ed79aa7ff1c503d12837"},
        "crates/rz-arena/src/pals_collect/native.rs": {
            "bytes": 233208, "sha256": "02af59b9edbc350e6e1a204eb6543033bb4a859729eb2ed79c8b470cc6823a01"},
    }),
)
_POLICY = {
    "version": "pals-post-repair-recheck/1",
    "policy": "same_repaired_line_once_v1",
    "search_identity": "pals-restricted-refinement-post-repair-recheck/1",
    "conditions_sha256": list(bytes.fromhex("bea44b7e9ab59f32dcffb1b4c597fd36a4b75037803aeeacd6c18785ce838166")),
}
_POLICY_ALIASES = ("post_repair_recheck", "post_repair_recheck_policy", "post_repair_recheck_conditions",
                   "post_repair_recheck_search_version", "refinement_policy", "refinement_conditions", "refinement_search_version")
_DENIED_AUTHORITIES = {
    "wdl_admitted": False,
    "rules_proof_admitted": False,
    "strategic_repair_validity_admitted": False,
    "counterexample_validity_admitted": False,
    "ordinary_policy_target_admitted": False,
    "divergence_ranking_admitted": False,
    "actual_training_executed": False,
    "product_authority": False,
}


class UnsupportedNativeRecheck(ValueError):
    """Retain raw evidence; this implementation has no reviewed admission path."""


def _pin(raw, expected):
    _fields(expected, ("bytes", "sha256"))
    _int(expected["bytes"], 0, MAX_BYTES)
    _sha(expected["sha256"])
    if type(raw) is not bytes or byte_pin(raw) != expected:
        raise ValueError("native recheck actual byte pin mismatch")


def _aggregate(raws):
    if any(type(raw) is not bytes for raw in raws) or sum(map(len, raws)) > MAX_BYTES:
        raise ValueError("native recheck aggregate immutable byte budget")


def _original_receipt_asset(receipt_bytes, expected_receipt_pin, name, raw, expected_asset_pin):
    """Additional raw bytes must already be in the ORIGINAL receipt inventory.

    The complete receipt bytes/hash are retained. This never removes artifacts,
    re-seals a receipt, or retroactively appends a file to a past collection.
    NativeCollectionAuthority still checks the parent and physical owner.
    """
    _aggregate((receipt_bytes, raw))
    _pin(receipt_bytes, expected_receipt_pin)
    receipt = _parse(receipt_bytes)
    if receipt.get("version") != "rz-pals-own-collector/1" or type(receipt.get("artifacts")) is not dict:
        raise ValueError("native recheck original collector receipt required")
    if type(name) is not str or name not in receipt["artifacts"]:
        raise UnsupportedNativeRecheck("native_recheck_asset_absent_from_original_receipt")
    _pin(raw, receipt["artifacts"][name])
    _pin(raw, expected_asset_pin)
    return byte_pin(raw)


def _policy_identity(value):
    """Four exact wire fields; the array uses u8 values, never bool aliases."""
    _fields(value, ("version", "policy", "search_identity", "conditions_sha256"))
    actual = value["conditions_sha256"]
    if type(actual) is not list or len(actual) != 32:
        raise ValueError("native recheck conditions digest requires exactly 32 bytes")
    for number in actual:
        _int(number, 0, 255)
    if canonical(value) != canonical(_POLICY):
        raise UnsupportedNativeRecheck("unsupported_native_recheck_policy_identity")
    return copy.deepcopy(value)


def _original_build_sources(raws, source):
    """Actual caller build/manifest binding BEFORE a closed review lookup.

    This small check deliberately does not copy the current dataset/producer
    registry loader. NativeCollectionAuthority and Repair verification remain
    necessary below. Caller metadata is conditional assurance, not a self seal
    proving that a compiler or child really ran.
    """
    _fields(raws, _SOURCE_RAW_NAMES)
    _aggregate(tuple(raws.values()))
    if type(source) is not dict:
        raise ValueError("native recheck original checked-source object required")
    implementation = _sha(source.get("implementation_sha256"))
    build, manifest = _parse(raws["build_registration"]), _parse(raws["source_manifest"])
    if build.get("schema") != BUILD_SCHEMA:
        raise UnsupportedNativeRecheck("unsupported_native_recheck_build_schema")
    if (type(build.get("actual_build_exit_code")) is not int or build["actual_build_exit_code"] != 0
            or build.get("source_verified_before_and_after_build") is not True
            or build.get("source_manifest") != byte_pin(raws["source_manifest"])):
        raise ValueError("native recheck requires independent successful original build observation")
    commit = build.get("source_commit")
    if (type(commit) is not str or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit)
            or manifest.get("source_commit") != commit):
        raise ValueError("native recheck original build/source commit binding")
    binary = build.get("binary")
    if (type(binary) is not dict or type(binary.get("bytes")) is not int
            or not 0 < binary["bytes"] <= 512 * 1024 * 1024
            or _sha(binary.get("sha256")) != implementation):
        raise ValueError("native recheck actual registered collector binary binding")
    artifact = build.get("compiler_artifact")
    if (type(artifact) is not dict or artifact.get("reason") != "compiler-artifact"
            or type(artifact.get("target")) is not dict or artifact["target"].get("name") != "pals_collect"
            or type(artifact.get("features")) is not list or "pals-collection-onnx" not in artifact["features"]
            or type(artifact.get("fresh")) is not bool):
        raise ValueError("native recheck actual compiler artifact observation required")
    files = manifest.get("files")
    if type(files) is not list or not 1 <= len(files) <= 4096:
        raise ValueError("native recheck bounded original source manifest required")
    by_path = {}
    for item in files:
        _fields(item, ("path", "bytes", "sha256"))
        path = item["path"]
        if type(path) is not str or not path or path in by_path:
            raise ValueError("native recheck original manifest paths must be unique")
        _int(item["bytes"], 0, 64 * 1024 * 1024)
        _sha(item["sha256"])
        by_path[path] = {name: item[name] for name in ("bytes", "sha256")}
    pair = {}
    for path, name in zip(_SOURCE_PATHS, ("engine_source", "native_source")):
        actual = byte_pin(raws[name])
        if by_path.get(path) != actual:
            raise ValueError("native recheck source bytes absent from original registered build manifest")
        pair[path] = actual
    return pair, {
        "source_commit": commit, "collector_binary_sha256": binary["sha256"],
        "build_registration": byte_pin(raws["build_registration"]),
        "source_manifest": byte_pin(raws["source_manifest"]),
        "assurance": "independently_pinned_caller_build_observation;not_self_reported_execution_certification",
    }


def _reviewed_source_pair(actual):
    _fields(actual, _SOURCE_PATHS)
    for review, pair in _REVIEWED_OPT_IN_SOURCE_PROFILES:
        if actual == pair:
            return review, copy.deepcopy(pair)
    raise UnsupportedNativeRecheck("unreviewed_opt_in_collector_transition_source_pair")


def _registered_policy(raw, source):
    """Independent raw registration plus the actual constructor description.

    This wire matches the native owner's separate registration. Final source
    profiles still require independent review before enrollment. A matching
    declaration remains startup selection only.
    """
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_POLICY_REGISTRATION_BYTES:
        raise ValueError("native recheck bounded nonempty policy registration required")
    registration = _fields(_parse(raw, MAX_POLICY_REGISTRATION_BYTES), ("version", "base_registry_canonical_sha256", "collector_binary_sha256", "search_policy"))
    if registration["version"] != POLICY_REGISTRATION_SCHEMA:
        raise UnsupportedNativeRecheck("unsupported_native_recheck_registration_schema")
    _sha(registration["base_registry_canonical_sha256"])
    _sha(registration["collector_binary_sha256"])
    declared = _policy_identity(registration["search_policy"])
    native_source = source.get("native")
    if type(native_source) is not dict or type(native_source.get("independent_registry")) is not dict:
        raise ValueError("native recheck original constructor/base registry facts required")
    registry = native_source["independent_registry"]
    if (registry.get("version") != "rz-pals-native-collection-registry/1"
            or byte_pin(canonical(registry))["sha256"] != registration["base_registry_canonical_sha256"]
            or registration["collector_binary_sha256"] != source.get("implementation_sha256")
            or registry.get("collector_binary_sha256") != registration["collector_binary_sha256"]
            or canonical(native_source.get("refinement_registration")) != canonical(registration)
            or native_source.get("refinement_registration_sha256") != byte_pin(raw)["sha256"]):
        raise ValueError("native recheck original policy registration/base registry/binary binding")
    actual = _policy_identity(native_source.get("pals_search_policy"))
    if canonical(actual) != canonical(declared) or native_source.get("search_version") != actual["search_identity"]:
        raise ValueError("native recheck selected constructor policy differs from independent registration")
    # The actual constructor writes the selected identity at native.* ONLY.
    # Equal declarations elsewhere still do not acquire an admitted meaning.
    for owner in (source, native_source.get("search_configuration", {}), registry):
        if type(owner) is not dict:
            raise ValueError("native recheck policy owner must be an object")
        if any(name in owner for name in ("pals_search_policy", "search_version", "refinement_registration", "refinement_registration_sha256")):
            raise UnsupportedNativeRecheck("unreviewed_native_recheck_policy_owner_location")
    for owner in (source, native_source, native_source.get("search_configuration", {}), registry):
        if any(name in owner for name in _POLICY_ALIASES):
            raise UnsupportedNativeRecheck("unreviewed_native_recheck_policy_alias")
    return {"registration": byte_pin(raw), "search_policy": actual,
            "assurance": "independently_pinned_caller_registration;checked_constructor_selection_only"}


def _source_review(raws, source, policy_raw):
    pair, observation = _original_build_sources(raws, source)
    policy = _registered_policy(policy_raw, source)
    closure = None
    try:
        review, detached = _reviewed_source_pair(pair)
    except UnsupportedNativeRecheck as historical_error:
        from .native_source_review import review_transition_sources, UnsupportedTransitionSource
        try:
            closure = review_transition_sources(pair=pair, manifest=_parse(raws["source_manifest"]),
                source=source, lane="legacy_single_reply")
        except UnsupportedTransitionSource as error:
            if str(error) == "unreviewed_transition_source_closure":
                raise historical_error
            raise UnsupportedNativeRecheck(str(error)) from error
        review, detached = closure["review_id"], copy.deepcopy(pair)
    result = {"build_observation": observation, "policy_registration": policy,
            "source_review_profile": review, "reviewed_transition_sources": detached}
    if closure is not None:
        result["reviewed_transition_closure"] = closure
    return result


def _recheck_identity(value):
    _fields(value, ("game_generation", "search_generation", "root", "root_revision", "repair_record_revision", "repaired_line"))
    for key in ("game_generation", "search_generation", "root_revision", "repair_record_revision", "repaired_line"):
        _int(value[key], 0, 2**64 - 1)
    _fields(value["root"], ("slot", "generation"))
    _int(value["root"]["slot"], 0, 2**64 - 1)
    _int(value["root"]["generation"], 0, 2**64 - 1)
    return copy.deepcopy(value)


def _optional_jsonl_rows(raw):
    """An empty optional journal has no observations, never completion authority.

    Nonempty journals keep the existing exact newline/JSONL parser. Required
    prepared input and producer journals do not use this optional boundary.
    """
    if type(raw) is bytes and raw == b"":
        return []
    return native._rows(raw)


def _sealed_trace_rows(raw, *, expected_pin, identity, descriptor_sha256):
    """Whole raw file seals and one exact attempt's order, not its validity.

    Native clock is microseconds since capture start. Engine deadline ticks are
    nanoseconds from the engine clock origin; they cannot be compared directly.
    Final endpoint/parent-chain checks are required by the whole factory.
    """
    _pin(raw, expected_pin)
    selected_identity = _recheck_identity(identity)
    _sha(descriptor_sha256)
    selected, groups = [], {}
    for _, row in _optional_jsonl_rows(raw):
        _fields(row, ("domain", "stage", "game_id", "identity", "descriptor_sha256", "payload_sha256", "observer_elapsed_us", "data"))
        if row["domain"] != TRACE_DOMAIN or row["stage"] not in ("prepared", "reply_bound", "finished"):
            raise ValueError("native recheck trace domain/stage")
        _recheck_identity(row["identity"])
        _sha(row["descriptor_sha256"])
        _sha(row["payload_sha256"])
        _int(row["observer_elapsed_us"], 0, 2**64 - 1)
        if (type(row["game_id"]) is not str or not row["game_id"] or len(row["game_id"].encode()) > 256
                or type(row["data"]) is not dict):
            raise ValueError("native recheck trace bounded game/data object")
        expected = byte_pin(canonical([TRACE_DOMAIN, row["stage"], row["identity"], row["descriptor_sha256"], row["data"]]))["sha256"]
        if row["payload_sha256"] != expected:
            raise ValueError("native recheck trace payload seal mismatch")
        if row["stage"] == "prepared":
            before = byte_pin(canonical([TRACE_DOMAIN, "prepared-descriptor", row["identity"], row["data"]]))["sha256"]
            if row["descriptor_sha256"] != before:
                raise ValueError("native recheck immutable prepared descriptor seal mismatch")
        key = canonical(row["identity"])
        groups.setdefault(key, []).append(row)
        if key == canonical(selected_identity):
            if row["descriptor_sha256"] != descriptor_sha256:
                raise ValueError("native recheck selected descriptor identity drift")
            selected.append(row)
    for group in groups.values():
        stages = [row["stage"] for row in group]
        if len(stages) != len(set(stages)):
            raise ValueError("native recheck duplicate attempt stage")
        # A finished-only row, if supplied, cannot acquire a before-result
        # descriptor or become a known witness. The current native writer keeps
        # unreserved rejection in its failure envelope instead of inventing one.
        if stages not in (["prepared"], ["prepared", "reply_bound"], ["prepared", "finished"],
                          ["prepared", "reply_bound", "finished"], ["finished"]):
            raise ValueError("native recheck attempt stage chronology")
        if (len({row["game_id"] for row in group}) != 1 or len({row["descriptor_sha256"] for row in group}) != 1
                or [row["observer_elapsed_us"] for row in group] != sorted(row["observer_elapsed_us"] for row in group)):
            raise ValueError("native recheck attempt game/seal/clock drift")
    return copy.deepcopy(selected)


def _logical_context(value, *, identity, purpose, prefix):
    """Typed engine-local identity; never a native dispatch or Rules replay."""
    _fields(value, ("game_generation", "search_generation", "situation", "state", "focus", "purpose", "prefix",
                    "focus_sha256", "prefix_sha256", "proposal_sha256", "refutation_sha256", "divergence_sha256",
                    "public_revision", "situation_revision"))
    for name in ("game_generation", "search_generation", "state", "focus", "public_revision", "situation_revision"):
        _int(value[name], 0, 2**64 - 1)
    _fields(value["situation"], ("slot", "generation"))
    for number in value["situation"].values():
        _int(number, 0, 2**64 - 1)
    for name in ("focus_sha256", "prefix_sha256", "proposal_sha256", "divergence_sha256"):
        _sha(value[name])
    if value["refutation_sha256"] is not None:
        _sha(value["refutation_sha256"])
    repair._moves(value["prefix"])
    if (value["game_generation"] != identity["game_generation"]
            or value["search_generation"] != identity["search_generation"]
            or value["purpose"] != purpose or value["prefix"] != prefix):
        raise ValueError("native recheck actual engine context differs from the bound query")
    return value


def _prepared_input_rows(initial_repair, *, input_sha256, native_request, kind, anchors):
    """Retrieve the ORIGINAL ordinary input, sidecar, lineage and journal.

    This helper does not copy the registry/current selector. The exact Repair
    capability rechecks those gates, and the actual strict-loader binding must
    name this request and all three immutable row bytes. The producer API does
    not return a raw journal SHA, so no such observation is manufactured here.
    """
    initial_repair.verify()
    _sha(input_sha256)
    native._request(native_request)
    _fields(anchors, ("input_row_sha256", "sidecar_row_sha256", "sidecar_sha256", "canonical_tensor_sha256", "lineage_row_sha256"))
    for value in anchors.values():
        _sha(value)
    parents, artifacts = initial_repair._parents, dict(initial_repair._artifacts)
    matches = [(index, row) for index, row in enumerate(parents.records)
               if index in parents.current_view.current_indices and row["input"]["sha256"] == input_sha256]
    if len(matches) != 1:
        raise ValueError("native recheck requires one strict current ordinary input")
    index, row = matches[0]
    snapshot = row["input"]["snapshot"]
    required_role = "proposer" if kind == "Repair" else "critic"
    if kind not in ("Repair", "Reply") or snapshot["role"] != required_role or snapshot["source"]["kind"] != "own_pals":
        raise ValueError("native recheck actual ordinary role differs from its call")
    actual_rows = []
    for name, key, pin_key in (("inputs.jsonl", "sha256", "input_row_sha256"),
                               ("native-inputs.jsonl", "input_sha256", "sidecar_row_sha256"),
                               ("input-lineage.jsonl", "input_sha256", "lineage_row_sha256")):
        raw, parsed = repair._one(native._rows(artifacts[name]), lambda value: value.get(key) == input_sha256, name)
        if byte_pin(raw)["sha256"] != anchors[pin_key]:
            raise ValueError("native recheck original captured row differs from descriptor byte anchor")
        actual_rows.append((raw, parsed))
    (input_raw, frozen), (sidecar_raw, sidecar), (lineage_raw, lineage) = actual_rows
    _fields(lineage, repair._LINEAGE)
    native._request([lineage["process_epoch"], lineage["request_sequence"]])
    if (frozen != row["input"] or [lineage["process_epoch"], lineage["request_sequence"]] != native_request
            or lineage["game_id"] != snapshot["game_id"] or lineage["native_query_kind"] != kind
            or lineage["actual_played_history"] != snapshot["actual_history"] or lineage["divergence_plies"] != []
            or lineage["training_admission"] != "ordinary_role" or lineage["counterfactual_wdl"] != "masked"
            or lineage["actual_outcome_eligible"] is not False):
        raise ValueError("native recheck actual request and ordinary lineage mismatch")
    admitted = [entry for entry in parents.frozen_admission["inputs"]
                if entry["binding"]["input_sha256"] == input_sha256]
    if len(admitted) != 1:
        raise ValueError("native recheck ordinary input lacks a unique actual producer binding")
    admitted = admitted[0]
    _, journal = repair._one(native._rows(artifacts["producer-prepared.jsonl"]),
                            lambda value: type(value.get("prepared")) is dict and value["prepared"].get("input_sha256") == input_sha256,
                            "native recheck prepared journal")
    _fields(journal, ("prepared", "sha256"))
    prepared = journal["prepared"]
    if (journal["sha256"] != training._sorted_canonical(training.PREPARED_PRODUCER_DOMAIN, prepared)
            or journal["sha256"] != admitted["prepared_evidence_sha256"] or prepared["learning_input"] is not True
            or prepared["native_request"] != native_request
            or any(prepared[name] != byte_pin(raw) for name, raw in (("input_json", input_raw),
                          ("tensor_sidecar_json", sidecar_raw), ("lineage_json", lineage_raw)))):
        raise ValueError("native recheck missing exact independently recovered prepared journal")
    policy = admitted["producer"]["encoding_policy"]
    encoded = training.encoded_from_sidecar(sidecar, row, expected_encoder_source_sha256=policy["encoder_source_sha256"],
                                            expected_model_epoch=policy["native_model_epoch"]["sha256"])
    if (encoded != parents.encodings[input_sha256] or sidecar["sha256"] != anchors["sidecar_sha256"]
            or sidecar["canonical_tensor_sha256"] != anchors["canonical_tensor_sha256"]):
        raise ValueError("native recheck actual sealed tensor or encoding differs from the prepared call")
    tensor = _parse(sidecar["tensor_json"].encode(), MAX_BYTES)
    sources = _optional_jsonl_rows(artifacts["public-record-sources.jsonl"])
    source = _parse(initial_repair._raws[1])[1]
    if len(tensor["records"]) != len(snapshot["public_records"]):
        raise ValueError("native recheck public token/source roster mismatch")
    for token, selected in zip(tensor["records"], snapshot["public_records"]):
        _, public = native._pinned_public_row(sources, selected["observation_sha256"])
        _fields(public, ("domain", "game_id", "record_index", "revision", "origin_state_id", "origin_state_id_is_advisory", "origin_rules_state_sha256",
                         "origin_rules_identity_observation", "kind", "line", "value", "completed_depth", "scope", "white_score_perspective", "critical", "source_cpu_profile_sha256"))
        for name in ("record_index", "revision", "origin_state_id", "completed_depth"):
            _int(public[name], 0, 2**64 - 1)
        repair._moves(public["line"])
        if (public["domain"] != "rz-pals-native-public-source/1" or public["game_id"] != snapshot["game_id"]
                or public["record_index"] != token["record_id"] or public["revision"] != token["revision"] or public["critical"] is not token["critical"]
                or type(public["white_score_perspective"]) is not bool or public["source_cpu_profile_sha256"] != source["cpu_profile_sha256"]
                or public["origin_state_id_is_advisory"] is not True or public["origin_rules_state_sha256"] is not None
                or public["origin_rules_identity_observation"] != "unknown"):
            raise ValueError("native recheck selected public source attribution mismatch")
    if kind == "Reply":
        # The existing first-Repair helper encodes 0/2 for Propose/Repair; the
        # actual Reply uses 1. Keep its immutable captured remaining-clock value.
        query = tensor["query"]
        if type(query) is not list or len(query) != 16 or tensor["divergence_features"] != []:
            raise ValueError("native recheck actual ordinary Reply query required")
        prefix, proposal, counter = lineage["virtual_prefix"], lineage["proposal"], lineage["counterexample"]
        repair._moves(prefix)
        repair._moves(proposal)
        if counter is None:
            raise ValueError("native recheck Reply lacks the actual refutation")
        repair._moves(counter)
        expected = [1, len(prefix)/256, len(proposal)/256, 1, len(counter)/256, len(snapshot["legal_moves"])/256,
                    snapshot["input_revision"], query[7], int(snapshot["white_to_move"]), 0, 0, 0, 0, 0, 0, 1]
        if struct.unpack("<f",repair._fp32(query[7]))[0] < 0:
            raise ValueError("native recheck negative captured remaining clock")
        for offset,movement in ((9,prefix[-1] if prefix else None),(12,proposal[0] if proposal else None)):
            if movement is not None:
                start,end,promotion = training.move_components(movement)
                expected[offset:offset+3] = [start/63,end/63,promotion/4]
        if any(repair._fp32(actual) != repair._fp32(want) for actual,want in zip(query,expected)):
            raise ValueError("native recheck Reply query differs from actual sealed lineage")
    return index, row, lineage, journal, tensor, admitted["producer"]


def _raw_reply_events(*, events_bytes, output_bytes, game_id, input_sha256, native_request, candidate_count, expected_pins):
    """Check raw attribution only; returns no checked witness or target.

    Missing/partial/rejected work is unresolved. Contradictory identity, late or
    duplicate consumption is malformed. A physical error is retained without
    treating its logits or the absence of logits as a zero-valued result.
    """
    _fields(expected_pins, ("events", "outputs"))
    _aggregate((events_bytes, output_bytes))
    _pin(events_bytes, expected_pins["events"])
    _pin(output_bytes, expected_pins["outputs"])
    _sha(input_sha256)
    native._request(native_request)
    _int(candidate_count, 1, 256)
    if type(game_id) is not str or not game_id or len(game_id.encode()) > 256:
        raise ValueError("native recheck bounded game identity required")
    events, outputs = [], []
    for _, event in _optional_jsonl_rows(events_bytes):
        _fields(event, ("domain", "game_id", "process_epoch", "request_sequence", "input_sha256", "stage", "observer_elapsed_us", "detail"))
        native._request([event["process_epoch"], event["request_sequence"]])
        _sha(event["input_sha256"])
        _int(event["observer_elapsed_us"], 0, 2**64 - 1)
        if event["domain"] != "rz-pals-native-call-event/1":
            raise ValueError("native recheck raw event domain")
        request = [event["process_epoch"], event["request_sequence"]]
        if request == native_request:
            if event["game_id"] != game_id or event["input_sha256"] != input_sha256:
                raise ValueError("native recheck Reply event identity contradicts request")
            events.append(event)
        elif event["input_sha256"] == input_sha256:
            raise ValueError("native recheck Reply input attributed to a different request")
    for _, output in _optional_jsonl_rows(output_bytes):
        _fields(output, ("domain", "process_epoch", "request_sequence", "input_sha256", "physical_completion_confirmed", "success", "raw"))
        native._request([output["process_epoch"], output["request_sequence"]])
        _sha(output["input_sha256"])
        if output["domain"] != "rz-pals-native-physical-raw/1":
            raise ValueError("native recheck physical raw domain")
        request = [output["process_epoch"], output["request_sequence"]]
        if request == native_request:
            if output["input_sha256"] != input_sha256:
                raise ValueError("native recheck physical output identity contradicts request")
            outputs.append(output)
        elif output["input_sha256"] == input_sha256:
            raise ValueError("native recheck physical input attributed to a different request")
    if len(outputs) > 1:
        raise ValueError("native recheck duplicate physical Reply output")
    stages = [event["stage"] for event in events]
    if len(stages) != len(set(stages)):
        raise ValueError("native recheck duplicate Reply stage")
    valid = ([], ["prepared"], ["prepared", "logically_rejected"],
             ["prepared", "physically_completed"],
             ["prepared", "physically_completed", "logically_rejected"],
             ["prepared", "physically_completed", "delivered"],
             ["prepared", "physically_completed", "delivered", "logically_rejected"],
             ["prepared", "physically_completed", "delivered", "search_consumed"])
    if stages not in valid:
        raise ValueError("native recheck late/double/misordered Reply consumption")
    elapsed = [event["observer_elapsed_us"] for event in events]
    if elapsed != sorted(elapsed):
        raise ValueError("native recheck Reply chronology regressed")
    for event in events:
        detail, stage = event["detail"], event["stage"]
        if stage == "prepared":
            _fields(detail, ("prepared_before_submit", "native_query_kind", "producer_metadata_admitted"))
            if (type(detail) is not dict or detail.get("prepared_before_submit") is not True
                    or detail.get("native_query_kind") != "Reply" or type(detail.get("producer_metadata_admitted")) is not bool):
                raise ValueError("native recheck actual prepaid Reply preparation required")
        elif stage == "physically_completed":
            _fields(detail, ("success", "logical_acceptance_inferred"))
            if (type(detail) is not dict or type(detail.get("success")) is not bool
                    or detail.get("logical_acceptance_inferred") is not False):
                raise ValueError("native recheck physical completion cannot infer consumption")
        elif stage == "delivered":
            if canonical(detail) != canonical({"search_consumed": False}):
                raise ValueError("native recheck delivery cannot infer consumption")
        elif stage == "search_consumed":
            if canonical(detail) != canonical({"search_consumed": True}):
                raise ValueError("native recheck actual Reply consumption required")
        else:
            _fields(detail, ("reason", "search_consumed"))
            if (detail["search_consumed"] is not False or type(detail["reason"]) is not str
                    or not 1 <= len(detail["reason"].encode()) <= 256):
                raise ValueError("native recheck bounded explicit rejection reason required")
    physical = "physically_completed" in stages
    if physical and events[0]["detail"]["producer_metadata_admitted"] is not True:
        raise ValueError("native recheck rejected producer metadata reached physical execution")
    if physical != bool(outputs):
        return {"status": "unresolved", "reason": "missing_physical_reply_observation", "scope": "raw_reply_attribution_only", **_DENIED_AUTHORITIES}
    if outputs:
        output = outputs[0]
        if type(output["success"]) is not bool or type(output["physical_completion_confirmed"]) is not bool:
            raise ValueError("native recheck physical completion/success must be actual booleans")
        observed_success = events[stages.index("physically_completed")]["detail"]["success"]
        if output["success"] is not observed_success:
            raise ValueError("native recheck physical event/output success mismatch")
        if (not output["success"] or not output["physical_completion_confirmed"]):
            if "delivered" in stages or "search_consumed" in stages:
                raise ValueError("native recheck failed/unknown physical result was delivered or consumed")
            return {"status": "unresolved", "reason": "failed_or_unknown_physical_reply", "scope": "raw_reply_attribution_only", **_DENIED_AUTHORITIES}
        heads = _fields(output["raw"], ("representation", "candidate_logits_bits", "wdl_logits_bits", "divergence_logits_bits",
                                        "task_logits_bits", "private_latent_bits", "prediction_is_future_label"))
        if (heads["representation"] != "f32_ieee754_bits" or heads["prediction_is_future_label"] is not False
                or heads["task_logits_bits"] is not None):
            raise ValueError("native recheck Reply raw predictions cannot become labels")
        # This helper is Reply-only: the actual Critic payload requires Some
        # divergence logits even for an empty-D ordinary query, serialized [].
        # Proposer Repair's None head remains checked by repair._ordinary.
        if type(heads["divergence_logits_bits"]) is not list or heads["divergence_logits_bits"]:
            raise ValueError("native recheck Critic Reply requires the exact empty divergence head")
        _finite_bits(heads["candidate_logits_bits"], candidate_count)
        _finite_bits(heads["wdl_logits_bits"], 3)
        _finite_bits(heads["private_latent_bits"], 6144)
    if stages != valid[-1]:
        return {"status": "unresolved", "reason": "reply_not_search_consumed", "scope": "raw_reply_attribution_only", **_DENIED_AUTHORITIES}
    if events[0]["detail"]["producer_metadata_admitted"] is not True:
        raise ValueError("native recheck rejected producer metadata cannot reach consumption")
    return {"status": "consumed_raw_observation", "scope": "raw_reply_attribution_only",
            "native_request": copy.deepcopy(native_request), "input_sha256": input_sha256,
            "event_elapsed_us": elapsed, **_DENIED_AUTHORITIES}


def _ranked_policy_moves(raw, candidates):
    """Match finite Rust f32 total_cmp order, including distinct signed zero.

    This only checks which actual policy output was consumed. It grants no
    validity to a policy, score or proposed continuation.
    """
    _finite_bits(raw["candidate_logits_bits"], len(candidates))
    repair._moves(candidates, minimum=1, maximum=256)
    if len(set(candidates)) != len(candidates):
        raise ValueError("native recheck duplicated actual Rules candidates")
    def order(bits):
        return (~bits & 0xffffffff) if bits & 0x80000000 else bits ^ 0x80000000
    return [candidates[index] for index in sorted(range(len(candidates)), key=lambda index: (-order(raw["candidate_logits_bits"][index]), index))]


def _parent_chain(initial_repair, data, identity):
    """Every accepted Repair suffix call, not merely the last Repair ID."""
    body = _fields(data["parent_repair_chain"], ("initial_prefix", "initial_prefix_len", "full_repaired_line", "calls", "journal_observation"))
    causal = initial_repair.context()
    repaired, refutation = data["repaired"], data["refutation"]
    repair._moves(repaired, minimum=3)
    repair._moves(refutation, minimum=3)
    start = _int(body["initial_prefix_len"], 2, len(repaired) - 1)
    if (body["initial_prefix"] != causal["virtual_prefix"] or body["initial_prefix"] != repaired[:start]
            or len(body["initial_prefix"]) != start or body["full_repaired_line"] != repaired
            or refutation != causal["counterexample"] or repaired[:start] != refutation[:start]
            or body["journal_observation"] != "independent-exact-producer-journal-entry-required; raw-entry-SHA-not-returned-by-producer-API"
            or type(body["calls"]) is not list or len(body["calls"]) != len(repaired) - start):
        raise ValueError("native recheck entire actual Repair suffix and initial causal prefix required")
    root = initial_repair._parents.records[initial_repair._root_index]["input"]["snapshot"]
    artifacts, previous, producer, observed = dict(initial_repair._artifacts), None, None, []
    for ply, call in enumerate(body["calls"], start):
        _fields(call, ("process_epoch", "request_sequence", "input_sha256", "input_row_sha256", "sidecar_row_sha256", "sidecar_sha256",
                       "canonical_tensor_sha256", "lineage_row_sha256", "prefix_len", "prefix", "chosen_move", "position_rules_state_sha256",
                       "proposal", "counterexample", "logical_context_sha256", "logical_context", "producer_metadata_admitted", "selection_observation"))
        request = [call["process_epoch"], call["request_sequence"]]
        native._request(request)
        _int(call["chosen_move"], 0, 65535)
        _int(call["prefix_len"], 0, 64)
        for name in ("prefix", "proposal", "counterexample"):
            repair._moves(call[name])
        _sha(call["position_rules_state_sha256"])
        context = _logical_context(call["logical_context"], identity=identity, purpose="RepairPolicy", prefix=repaired[:ply])
        if (call["prefix_len"] != ply or call["prefix"] != repaired[:ply] or call["chosen_move"] != repaired[ply]
                or call["proposal"] != causal["proposal"] or call["counterexample"] != refutation
                or call["logical_context_sha256"] != byte_pin(canonical(context))["sha256"]
                or context["public_revision"] + 1 != identity["repair_record_revision"]
                or call["producer_metadata_admitted"] is not True
                or call["selection_observation"] != "actual-raw-policy-ranked-first-and-accepted-context-matches-repaired-ply"
                or previous is not None and (request[0] != previous[0] or request[1] <= previous[1])):
            raise ValueError("native recheck accepted Repair call identity/order/context mismatch")
        index, row, lineage, _, tensor, actual_producer = _prepared_input_rows(initial_repair, input_sha256=call["input_sha256"],
            native_request=request, kind="Repair", anchors={name: call[name] for name in
                ("input_row_sha256", "sidecar_row_sha256", "sidecar_sha256", "canonical_tensor_sha256", "lineage_row_sha256")})
        snapshot = row["input"]["snapshot"]
        if (any(canonical(snapshot[name]) != canonical(root[name]) for name in ("game_id", "opening_id", "line_genealogy_id", "actual_history", "source", "frozen_epoch", "encoding_sha256"))
                or snapshot["input_revision"] != context["public_revision"] or snapshot["rules_state_sha256"] != call["position_rules_state_sha256"]
                or lineage["virtual_prefix"] != call["prefix"] or lineage["proposal"] != call["proposal"] or lineage["counterexample"] != refutation
                or producer is not None and producer != actual_producer):
            raise ValueError("native recheck Repair chain changed actual root/history/producer or prepared lineage")
        if ply == start and (row["input"]["sha256"] != causal["repair_input_sha256"] or request != causal["repair_native_request"]):
            raise ValueError("native recheck first Repair differs from the exact causal capability")
        repair._ordinary(initial_repair._parents, index, artifacts, _parse(initial_repair._raws[1])[1], "Repair")
        _, physical = repair._one(native._rows(artifacts["native-raw-outputs.jsonl"]),
                                  lambda value: value.get("input_sha256") == row["input"]["sha256"]
                                  and [value.get("process_epoch"), value.get("request_sequence")] == request, "parent Repair physical output")
        if _ranked_policy_moves(physical["raw"], snapshot["legal_moves"])[0] != call["chosen_move"]:
            raise ValueError("native recheck accepted Repair chosen move differs from actual policy rank")
        previous, producer = request, actual_producer
        observed.append({"input_sha256":row["input"]["sha256"], "native_request":request, "prefix_len":ply, "chosen_move":call["chosen_move"]})
    return observed, previous, producer


class CheckedNativeRecheckRepairAnchor:
    """Factory-only current Repair plus opt-in source authority; no target."""
    __slots__ = ("_repair", "_raws", "_pins", "_identity")

    def __init__(self, token=None, *, initial_repair=None, raws=None, pins=None):
        if token is not _FACTORY:
            raise ValueError("unchecked native recheck Repair anchor refused")
        for name, value in (("_repair", initial_repair), ("_raws", tuple(sorted(raws.items()))),
                            ("_pins", canonical(pins)), ("_identity", digest(ANCHOR_SCHEMA, pins))):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError("checked native recheck Repair anchor is immutable")

    def verify(self):
        pins = _parse(self._pins)
        if digest(ANCHOR_SCHEMA, pins) != self._identity:
            raise ValueError("native recheck Repair anchor identity changed")
        _validate_anchor(self._repair, dict(self._raws), pins)
        return self

    @property
    def sha256(self):
        self.verify()
        return self._identity

    def admission(self):
        self.verify()
        return copy.deepcopy(_validate_anchor(self._repair, dict(self._raws), _parse(self._pins)))


def _validate_anchor(initial_repair, raws, pins):
    if type(initial_repair) is not repair.CheckedRepairContext:
        raise ValueError("exact checked current ordinary Repair causal capability required")
    _fields(raws, (*_SOURCE_RAW_NAMES, "policy_registration"))
    _fields(pins, (*_SOURCE_RAW_NAMES, "policy_registration", "initial_repair_sha256"))
    _aggregate((*raws.values(), *initial_repair._raws, *(raw for _, raw in initial_repair._artifacts)))
    for name, raw in raws.items():
        _pin(raw, pins[name])
    # The original checked-source bytes belong to the exact Repair capability;
    # no sanitized/marker-stripped copy is passed to any legacy authority gate.
    source_wire = _parse(initial_repair._raws[1])
    if (type(source_wire) is not list or len(source_wire) != 2
            or source_wire[0] != training.CHECKED_SOURCE_DOMAIN or type(source_wire[1]) is not dict):
        raise ValueError("native recheck actual original checked-source bytes required")
    source = source_wire[1]
    source_observation = _source_review({name: raws[name] for name in _SOURCE_RAW_NAMES}, source, raws["policy_registration"])
    # This path becomes reachable only after literal opt-in profiles have been
    # independently fixed. verify() reuses the actual strict parent/current/
    # producer authority and original Rules checks.
    initial_repair.verify()
    if initial_repair.sha256 != _sha(pins["initial_repair_sha256"]):
        raise ValueError("native recheck initial Repair capability pin mismatch")
    return {"repair_context": initial_repair.context(), "source_observation": source_observation,
            "scope": "current_ordinary_repair_opt_in_source_anchor_only", **_DENIED_AUTHORITIES}


def admit_native_recheck_repair_anchor(*, initial_repair, policy_registration_bytes,
                                      build_registration_bytes, source_manifest_bytes,
                                      engine_source_bytes, native_source_bytes, expected_pins):
    """No admission until final opt-in source and registration wire are reviewed.

    The original Repair's actual raw receipt and producer/current pins remain
    unchanged. Whole endpoint/replay/publication admission is a separate factory;
    this anchor cannot be collated or consumed by a loss.
    """
    raws = {"policy_registration": policy_registration_bytes, "build_registration": build_registration_bytes,
            "source_manifest": source_manifest_bytes, "engine_source": engine_source_bytes, "native_source": native_source_bytes}
    pins = _parse(canonical(expected_pins))
    _validate_anchor(initial_repair, raws, pins)
    return CheckedNativeRecheckRepairAnchor(_FACTORY, initial_repair=initial_repair, raws=raws, pins=pins)


_OBSERVATION_FIELDS = ("state", "line", "source", "epoch", "value_identity", "checker_identity", "checker_work", "external_report_debug",
                       "model_value_identity", "model_value_input", "cpu_condition", "cpu_pv", "scope", "score", "budget", "kind", "supersedes", "execution")
_CLOCK_FIELDS = ("engine_deadline_tick", "engine_deadline_tick_unit", "engine_deadline_tick_origin", "observer_elapsed_origin", "observer_elapsed_unit",
                 "cancelled_at_observer", "deadline_expired_at_observer", "limits")
_DISPOSITIONS = ("PreparedRejected", "Interrupted", "NoAlternativeResponse", "IncompleteCounterline", "IncomparableEvidence",
                 "MissingCompletedCounterValue", "CounterNotLower", "ConditionalRefutationPublished")


def _clock_controls(data):
    _int(data["engine_deadline_tick"], 0, 2**64 - 1)
    if (data["engine_deadline_tick_unit"] != "nanoseconds" or data["engine_deadline_tick_origin"] != "engine_monotonic_clock_origin"
            or data["observer_elapsed_origin"] != "native_capture_start" or data["observer_elapsed_unit"] != "microseconds"
            or type(data["cancelled_at_observer"]) is not bool or type(data["deadline_expired_at_observer"]) is not bool):
        raise ValueError("native recheck distinct original engine/observer clock scopes required")
    _fields(data["limits"], ("max_rounds", "max_cpu_nodes", "cpu_depth"))
    _int(data["limits"]["max_rounds"], 1, 2**64 - 1)
    _int(data["limits"]["max_cpu_nodes"], 1, 2**64 - 1)
    _int(data["limits"]["cpu_depth"], 1, 65535)


def _own_value_namespace(prepared, source):
    checker = _fields(prepared["checker_identity"], ("kind", "identity"))
    if checker["kind"] != "owned":
        raise UnsupportedNativeRecheck("external_checker_is_not_the_owned_recheck_lane")
    value = _fields(checker["identity"], ("semantics", "weights_sha256", "training"))
    semantics = value["semantics"]
    if (type(semantics) is not str or not semantics.strip() or len(semantics.encode()) > 256
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in semantics)):
        raise ValueError("native recheck bounded actual CPU value semantics")
    if value["weights_sha256"] is not None or value["training"] != {"kind":"bootstrap"}:
        raise UnsupportedNativeRecheck("first_native_recheck_gate_supports_owned_weightless_bootstrap_only")
    cpu = source.get("native", {}).get("cpu_task_source")
    if type(cpu) is not dict or type(cpu.get("configuration")) is not dict:
        raise ValueError("native recheck independently registered actual CPU configuration required")
    config = cpu["configuration"]
    if (config.get("value_identity") != value or config.get("value_precision") != "integer_cp"
            or cpu.get("implementation_sha256") != source["implementation_sha256"]
            or cpu.get("cpu_profile_sha256") != source["cpu_profile_sha256"]
            or byte_pin(canonical(config))["sha256"] != source["cpu_profile_sha256"]):
        raise ValueError("native recheck actual startup value/configuration/binary namespace mismatch")
    condition = prepared["cpu_condition"]
    if type(condition) is not str or not condition or len(condition.encode()) > 2048 or any(ord(char) < 32 for char in condition):
        raise ValueError("native recheck bounded exact CPU conditions required")
    return value


def _endpoint(endpoint, *, prepared, value_identity, root_white, plies):
    """Stored task/evidence checks; no missing CpuReport fields are invented."""
    _fields(endpoint, ("state", "situation", "rules_state_sha256", "board_fen", "evidence"))
    _int(endpoint["state"], 0, 2**64 - 1)
    _fields(endpoint["situation"], ("slot", "generation"))
    for value in endpoint["situation"].values():
        _int(value, 0, 2**64 - 1)
    _sha(endpoint["rules_state_sha256"])
    if type(endpoint["board_fen"]) is not str or not 1 <= len(endpoint["board_fen"].encode()) <= 256:
        raise ValueError("native recheck bounded endpoint FEN required")
    evidence = endpoint["evidence"]
    if type(evidence) is not dict:
        raise ValueError("native recheck typed endpoint evidence required")
    kind = evidence.get("kind")
    if kind == "Unobserved":
        _fields(evidence, ("kind",))
        return {"status":"unresolved", "reason":"endpoint_provenance_unobserved"}
    if kind == "InvalidProvenance":
        _fields(evidence, ("kind", "error"))
        raise ValueError("native recheck endpoint explicitly reports invalid store provenance")
    expected_white = root_white if plies % 2 == 0 else not root_white
    if kind == "RulesTerminal":
        _fields(evidence, ("kind", "reason", "value", "white_perspective"))
        _int(evidence["value"], -2**31, 2**31 - 1)
        if (type(evidence["reason"]) is not str or not 1 <= len(evidence["reason"].encode()) <= 128
                or type(evidence["white_perspective"]) is not bool or evidence["white_perspective"] is not expected_white):
            raise ValueError("native recheck endpoint terminal perspective/provenance")
        return {"status":"unresolved", "reason":"rules_terminal_not_cp_comparison", "raw_terminal":copy.deepcopy(evidence)}
    if kind != "OwnCpu":
        raise UnsupportedNativeRecheck("unsupported_native_recheck_endpoint_evidence")
    _fields(evidence, ("kind", "observation_id", "execution_id", "admitted_scope", "observation", "task", "task_consumer_observation"))
    for name in ("observation_id", "execution_id"):
        _int(evidence[name], 0, 2**64 - 1)
    if evidence["task_consumer_observation"] != "private-consumers-not-observed; engine-borrowed-completed-provenance-only":
        raise ValueError("native recheck private TaskConsumer observation scope changed")
    observation = _fields(evidence["observation"], _OBSERVATION_FIELDS)
    task = _fields(evidence["task"], ("status", "resumed_from", "key"))
    key = _fields(task["key"], ("state", "line", "question", "root_moves", "model", "epoch", "value_identity", "checker_identity",
                               "cpu_condition", "profile", "condition", "input_revision", "requested_depth", "node_budget"))
    for owner, names in ((observation, ("state", "source", "epoch", "budget")), (key, ("state", "model", "epoch", "profile", "condition", "input_revision", "node_budget"))):
        for name in names:
            _int(owner[name], 0, 2**64 - 1)
    _int(key["requested_depth"], 1, 65535)
    for name in ("cpu_pv", "supersedes", "execution"):
        if observation[name] is not None:
            _int(observation[name], 0, 2**64 - 1)
    if task["resumed_from"] is not None:
        _int(task["resumed_from"], 0, 2**64 - 1)
    scope = _fields(observation["scope"], ("kind", "depth", "profile", "condition"))
    _int(scope["depth"], 0, 65535)
    _int(scope["profile"], 0, 2**64 - 1)
    _int(scope["condition"], 0, 2**64 - 1)
    if (scope["kind"] != "DepthLimited" or observation["state"] != endpoint["state"] or key["state"] != endpoint["state"]
            or observation["line"] is not None or key["line"] is not None or observation["execution"] != evidence["execution_id"]
            or observation["kind"] != "CpuAnalysis" or observation["value_identity"] != value_identity or key["value_identity"] != value_identity
            or observation["checker_identity"] is not None or key["checker_identity"] is not None
            or observation["cpu_condition"] != prepared["cpu_condition"] or key["cpu_condition"] != prepared["cpu_condition"]
            or key["question"] != "AnalyzePosition" or key["root_moves"] != [] or key["input_revision"] != 0
            or observation["epoch"] != key["epoch"] or key["model"] != 0 or key["epoch"] != 0
            or observation["budget"] > key["node_budget"] or key["node_budget"] > prepared["limits"]["max_cpu_nodes"]
            or scope["profile"] != key["profile"] or scope["condition"] != key["condition"]
            or key["requested_depth"] != prepared["limits"]["cpu_depth"]
            or any(observation[name] is not None for name in ("checker_work", "external_report_debug", "model_value_identity", "model_value_input"))):
        raise ValueError("native recheck stored CPU execution/key/value namespace contradiction")
    status = task["status"]
    if type(status) is not dict or status.get("kind") not in ("Completed", "Paused", "Failed", "InFlight", "CancellationRequested"):
        raise ValueError("native recheck typed actual task status required")
    if status["kind"] == "Completed":
        _fields(status, ("kind", "observation_id"))
        _int(status["observation_id"], 0, 2**64 - 1)
        if (status["observation_id"] != evidence["observation_id"] or evidence["admitted_scope"] != "CompletedIteration"
                or scope["depth"] < key["requested_depth"]):
            raise ValueError("native recheck completed endpoint does not bind its actual observation/depth")
    elif status["kind"] == "Paused":
        _fields(status, ("kind", "checkpoint", "evidence"))
        _int(status["checkpoint"], 0, 2**64 - 1)
        if status["evidence"] is not None:
            _int(status["evidence"], 0, 2**64 - 1)
        if status["evidence"] != evidence["observation_id"]:
            raise ValueError("native recheck partial endpoint observation mismatch")
    else:
        _fields(status, ("kind",))
    score = observation["score"]
    if type(score) is not dict:
        raise ValueError("native recheck actual raw CPU score required")
    if evidence["admitted_scope"] == "CompletedIteration":
        _fields(score, ("kind", "value", "white_perspective", "bound"))
        _int(score["value"], -2**31, 2**31 - 1)
        if score["kind"] != "Cpu" or score["bound"] != "ExactWithinSearch" or type(score["white_perspective"]) is not bool or score["white_perspective"] is not expected_white:
            raise ValueError("native recheck completed endpoint raw score/perspective/bound mismatch")
    elif evidence["admitted_scope"] == "FrontierOnly":
        _fields(score, ("kind", "value_f32_bits", "white_perspective"))
        _finite_bits([score["value_f32_bits"]], 1)
        if score["kind"] != "Estimate" or type(score["white_perspective"]) is not bool or score["white_perspective"] is not expected_white:
            raise ValueError("native recheck partial frontier score/perspective mismatch")
        return {"status":"unresolved", "reason":"frontier_only_endpoint"}
    else:
        raise ValueError("native recheck endpoint scope is not stored Own CPU work")
    if status["kind"] != "Completed":
        return {"status":"unresolved", "reason":"endpoint_task_not_completed"}
    if abs(score["value"]) > 20000:
        return {"status":"unresolved", "reason":"endpoint_mate_or_non_frontier_score_band"}
    return {"status":"completed_cp", "raw_score":score["value"], "root_score":score["value"] if plies % 2 == 0 else -score["value"],
            "depth":scope["depth"], "profile":scope["profile"], "condition":scope["condition"], "node_budget":key["node_budget"],
            "value_identity":copy.deepcopy(value_identity), "observation_id":evidence["observation_id"], "execution_id":evidence["execution_id"]}


def _rules_line(check, initial_repair, *, prefix, claim, endpoint, expected_sha256):
    if type(check) is not CheckedSemanticInput:
        raise ValueError("exact registered Rust Rules semantic capability required for the whole recheck line")
    if check.verify_parent(initial_repair._parents) != initial_repair._root_index or check.sha256 != _sha(expected_sha256):
        raise ValueError("native recheck whole-line Rules capability changed parent/current identity")
    common, receipt = check.common_query(), check.rules_receipt()
    if common["question"] != "continuation_challenge" or common["prefix"] != prefix or common["root_moves"] or common["claimed_line"] != claim:
        raise ValueError("native recheck Rules query does not replay the exact original prefix/suffix")
    final = receipt["claimed_line"]["final_state"]
    if (final is None or final["rules_state_sha256"] != endpoint["rules_state_sha256"] or final["board_fen"] != endpoint["board_fen"]):
        raise ValueError("native recheck actual Rules full-history endpoint differs from the trace")
    return receipt


def _engine_summary_id(value):
    """Exact Rust stable_id metadata codec, not a source or execution proof."""
    result = 0xcbf29ce484222325
    for byte in value.encode("utf-8"):
        result = ((result ^ byte) * 0x100000001b3) & (2**64 - 1)
    return result


def _publication_shape(publication, *, prepared, identity, root_white, disposition, public_revision, export_manifest_sha256):
    """Actual restricted Estimate publication, not completed CPU/Rules proof."""
    if publication is None:
        return
    if disposition != "ConditionalRefutationPublished":
        raise ValueError("native recheck publication contradicts its actual disposition")
    _fields(publication, ("observation_id", "observation", "conclusion"))
    _int(publication["observation_id"], 0, 2**64 - 1)
    observation = _fields(publication["observation"], _OBSERVATION_FIELDS)
    for name in ("state", "line", "source", "epoch", "budget"):
        _int(observation[name], 0, 2**64 - 1)
    if observation["supersedes"] is not None:
        _int(observation["supersedes"], 0, 2**64 - 1)
    conclusion = _fields(publication["conclusion"], ("status", "evidence", "revision"))
    for name in ("evidence", "revision"):
        _int(conclusion[name], 0, 2**64 - 1)
    summary = _fields(observation["score"], ("kind", "value_f32_bits", "white_perspective"))
    _finite_bits([summary["value_f32_bits"]], 1)
    scope = _fields(observation["scope"], ("kind", "model", "encoding", "input"))
    for name in ("model", "encoding", "input"):
        _int(scope[name], 0, 2**64 - 1)
    if (observation["state"] != prepared["root_state"] or observation["line"] != identity["repaired_line"] or observation["kind"] != "Refutation"
            or summary["kind"] != "Estimate" or type(summary["white_perspective"]) is not bool or summary["white_perspective"] is not root_white
            or scope["kind"] != "Model" or scope["input"] < public_revision or observation["epoch"] != 0 or observation["budget"] != 0
            or observation["source"] != _engine_summary_id(_POLICY["search_identity"])
            or scope["encoding"] != _engine_summary_id("pals-restricted-summary-v1")
            # The checked cold CPU native constructor derives model.identity()
            # from the loaded export manifest, not from the checkpoint digest.
            or scope["model"] != _engine_summary_id("pals-onnx-pc-fp32-" + _sha(export_manifest_sha256))
            or any(observation[name] is not None for name in ("execution", "value_identity", "checker_identity", "checker_work", "external_report_debug",
                      "model_value_identity", "model_value_input", "cpu_condition", "cpu_pv"))
            or conclusion["status"] != "Refuted" or conclusion["evidence"] != publication["observation_id"]
            or conclusion["revision"] != identity["root_revision"] + 1):
        raise ValueError("native recheck actual publication is not the exact restricted estimate/conclusion")


def _whole_trace(anchor, trace_bytes, checks, pins):
    initial = anchor._repair
    root = initial._parents.records[initial._root_index]["input"]["snapshot"]
    artifacts = dict(initial._artifacts)
    _original_receipt_asset(artifacts["receipt.json"], initial._parents.frozen_admission["receipt"], TRACE_ARTIFACT, trace_bytes, pins["trace"])
    rows = _sealed_trace_rows(trace_bytes, expected_pin=pins["trace"], identity=pins["identity"], descriptor_sha256=pins["descriptor_sha256"])
    result = {"schema":WITNESS_SCHEMA, "scope":SCOPE, "anchor_sha256":pins["anchor_sha256"], "descriptor_sha256":pins["descriptor_sha256"],
              "identity":copy.deepcopy(pins["identity"]), "trace":byte_pin(trace_bytes),
              "assurance":"independently_pinned_caller_build_collection_and_Rules_observations;closed_reviewed_transition;no_private_TaskConsumer_or_CPU_stack_observation",
              "status":"unresolved", "reason":"missing_prepared_or_finished_trace", "publication_observed":False,
              # The engine rechecks stop AFTER a successful finished callback.
              # That later search closure is not in this three-stage artifact.
              "final_search_envelope_observed":False, "search_completion_admitted":False, **_DENIED_AUTHORITIES}
    if not rows or rows[0]["stage"] != "prepared":
        return result
    if any(row["game_id"] != root["game_id"] for row in rows):
        raise ValueError("native recheck descriptor belongs to a different actual game")
    prepared = _fields(rows[0]["data"], ("engine_observer_version", "checked_source_sha256", "refinement_registration_sha256", "root_state",
        "root_rules_state_sha256", "anchor_state", "anchor_situation", "anchor_rules_state_sha256", "anchor_ply", "anticipated_reply_context",
        "anticipated_reply_context_sha256", "accepted_repair_record", "repaired", "refutation", "parent_repair_chain", "repaired_endpoint",
        *_CLOCK_FIELDS, "checker_identity", "cpu_condition", "prepared_before_reply_submit", "trace_persistence", "assurance"))
    identity = _recheck_identity(pins["identity"])
    source = _parse(initial._raws[1])[1]
    for name in ("root_state", "anchor_state", "anchor_ply"):
        _int(prepared[name], 0, 2**64 - 1)
    for name in ("checked_source_sha256", "refinement_registration_sha256", "root_rules_state_sha256", "anchor_rules_state_sha256", "anticipated_reply_context_sha256"):
        _sha(prepared[name])
    _fields(prepared["anchor_situation"], ("slot", "generation"))
    for value in prepared["anchor_situation"].values():
        _int(value, 0, 2**64 - 1)
    _clock_controls(prepared)
    if (prepared["engine_observer_version"] != "pals-post-repair-recheck-observer/1"
            or prepared["checked_source_sha256"] != byte_pin(canonical([training.CHECKED_SOURCE_DOMAIN,source]))["sha256"]
            or prepared["refinement_registration_sha256"] != byte_pin(dict(anchor._raws)["policy_registration"])["sha256"]
            or prepared["root_rules_state_sha256"] != root["rules_state_sha256"]
            or prepared["prepared_before_reply_submit"] is not True
            or prepared["trace_persistence"] != "seal-before-submit; prepaid-buffer-drain-after-search"
            or prepared["assurance"] != "conditional-search-observer; no-whole-game-proof; no-training-target; task-private-consumers-not-observed"):
        raise ValueError("native recheck actual source/policy/root or before-submit descriptor binding")
    parent_chain, last_request, producer = _parent_chain(initial, prepared, identity)
    line, ply = prepared["repaired"], prepared["anchor_ply"]
    if not 0 < ply < len(line) or ply % 2 != 1:
        raise ValueError("native recheck Reply must be at an actual opponent ply inside the repaired line")
    # The engine chooses the first opponent anchor following a changed OWN
    # decision; mere shared line length or a matching final Repair ID is weaker.
    changed = next((index for index in range(len(initial.context()["virtual_prefix"]), len(line))
                    if index % 2 == 0 and (index >= len(prepared["refutation"]) or line[index] != prepared["refutation"][index])), None)
    if changed is None or ply != changed + 1:
        raise ValueError("native recheck anchor is not after the actual changed own Repair decision")
    record = _fields(prepared["accepted_repair_record"], ("revision", "origin_state", "kind", "line", "value", "completed_depth", "score_scope",
                                                        "cpu_observation", "white_perspective", "critical"))
    for name in ("revision", "origin_state", "completed_depth"):
        _int(record[name], 0, 2**64 - 1)
    if record["value"] is not None:
        _int(record["value"], -2**31, 2**31 - 1)
    if record["cpu_observation"] is not None:
        _int(record["cpu_observation"], 0, 2**64 - 1)
    repair._moves(record["line"])
    if (record["revision"] != identity["repair_record_revision"] or record["origin_state"] != prepared["root_state"]
            or record["kind"] != "Repair" or record["line"] != line or type(record["white_perspective"]) is not bool
            or record["white_perspective"] is not root["white_to_move"] or type(record["critical"]) is not bool
            or record["score_scope"] not in (None, "FrontierOnly", "CompletedIteration", "RulesTerminal")):
        raise ValueError("native recheck descriptor does not bind the actual accepted full Repair record")
    anticipated = _logical_context(prepared["anticipated_reply_context"], identity=identity, purpose="ReplyPolicy", prefix=line[:ply])
    if (anticipated["state"] != prepared["anchor_state"] or anticipated["situation"] != prepared["anchor_situation"]
            or prepared["anticipated_reply_context_sha256"] != byte_pin(canonical(anticipated))["sha256"]):
        raise ValueError("native recheck anticipated Reply context/state seal mismatch")
    value_identity = _own_value_namespace(prepared, source)
    repaired_evidence = _endpoint(prepared["repaired_endpoint"], prepared=prepared, value_identity=value_identity, root_white=root["white_to_move"], plies=len(line))
    result.update({"parent_repair_chain":parent_chain, "accepted_repair_record":copy.deepcopy(record), "repaired_endpoint":repaired_evidence,
                   "original_limits":copy.deepcopy(prepared["limits"]), "engine_deadline_tick":prepared["engine_deadline_tick"],
                   "trace_persistence":prepared["trace_persistence"], "cpu_completion_scope":"borrowed_store_TaskRecord_and_Observation;CpuReport_completion_and_private_consumers_not_observed"})
    by_stage = {row["stage"]:row for row in rows}
    bound = by_stage.get("reply_bound")
    reply_snapshot, reply_tensor, reply_request = None, None, None
    if bound is not None:
        binding = _fields(bound["data"], ("prepared_payload_sha256", "process_epoch", "request_sequence", "input_sha256", "input_row_sha256", "sidecar_row_sha256",
            "sidecar_sha256", "canonical_tensor_sha256", "lineage_row_sha256", "logical_context", "producer_metadata_admitted", "bound_before_submit",
            "physical_completion_observed", "delivery_observed", "search_consumption_observed"))
        reply_request = [binding["process_epoch"],binding["request_sequence"]]
        native._request(reply_request)
        if (binding["prepared_payload_sha256"] != rows[0]["payload_sha256"] or binding["logical_context"] != anticipated
                or binding["bound_before_submit"] is not True or binding["producer_metadata_admitted"] is not True
                or any(binding[name] is not False for name in ("physical_completion_observed", "delivery_observed", "search_consumption_observed"))
                or reply_request[0] != last_request[0] or reply_request[1] <= last_request[1]):
            raise ValueError("native recheck actual Reply bound row drifted from the prepaid descriptor")
        _, reply, lineage, journal, reply_tensor, reply_producer = _prepared_input_rows(initial, input_sha256=binding["input_sha256"],
            native_request=reply_request, kind="Reply", anchors={name:binding[name] for name in
                ("input_row_sha256", "sidecar_row_sha256", "sidecar_sha256", "canonical_tensor_sha256", "lineage_row_sha256")})
        reply_snapshot = reply["input"]["snapshot"]
        if (reply_producer != producer or lineage["virtual_prefix"] != line[:ply] or lineage["proposal"] != line or lineage["counterexample"] != prepared["refutation"]
                or reply_snapshot["rules_state_sha256"] != prepared["anchor_rules_state_sha256"] or reply_snapshot["input_revision"] != anticipated["public_revision"]
                or reply_snapshot["white_to_move"] is root["white_to_move"]
                or any(canonical(reply_snapshot[name]) != canonical(root[name]) for name in ("game_id", "opening_id", "line_genealogy_id", "actual_history", "source", "frozen_epoch", "encoding_sha256"))):
            raise ValueError("native recheck actual Reply input changed root/producer/history/prefix")
        raw_reply = _raw_reply_events(events_bytes=artifacts["native-events.jsonl"], output_bytes=artifacts["native-raw-outputs.jsonl"], game_id=root["game_id"],
            input_sha256=binding["input_sha256"], native_request=reply_request, candidate_count=len(reply_snapshot["legal_moves"]),
            expected_pins={"events":byte_pin(artifacts["native-events.jsonl"]),"outputs":byte_pin(artifacts["native-raw-outputs.jsonl"])})
        result["reply_observation"] = raw_reply
        result["reply_prepared_evidence_sha256"] = journal["sha256"]
    finished = by_stage.get("finished")
    if finished is None:
        return result
    final = _fields(finished["data"], ("prepared_payload_sha256", "bound_payload_sha256", "prepared_accepted", "reply_call_attempted", "reply_accepted", "reply",
        "reply_context", "selected_response", "counterline", "full_suffix_replayed", "repaired_endpoint", "counter_endpoint", "comparable", "publication",
        "disposition", "original_error", *_CLOCK_FIELDS, "assurance"))
    _clock_controls(final)
    for name in ("prepared_accepted", "reply_call_attempted", "reply_accepted", "full_suffix_replayed", "comparable"):
        if type(final[name]) is not bool:
            raise ValueError("native recheck actual engine completion flags must be booleans")
    if (final["prepared_payload_sha256"] != rows[0]["payload_sha256"] or final["bound_payload_sha256"] != (None if bound is None else bound["payload_sha256"])
            or final["prepared_accepted"] is not True or final["reply_context"] != anticipated
            or any(final[name] != prepared[name] for name in ("engine_deadline_tick", "limits")) or final["disposition"] not in _DISPOSITIONS
            or final["assurance"] != "conditional-search-observer; marker-is-not-refutation; no-training-target; physical-and-delivery-events-required"):
        raise ValueError("native recheck finished descriptor/original condition identity mismatch")
    if final["original_error"] is not None and (type(final["original_error"]) is not str or not 1 <= len(final["original_error"].encode()) <= 4096):
        raise ValueError("native recheck original operation error must be bounded and retained")
    if final["reply"] is not None:
        actual_reply = _fields(final["reply"], ("process_epoch", "request_sequence", "input_sha256", "physical", "delivered", "search_consumed", "rejected", "physical_unknown"))
        native._request([actual_reply["process_epoch"],actual_reply["request_sequence"]])
        for name in ("physical", "delivered", "search_consumed", "rejected", "physical_unknown"):
            if type(actual_reply[name]) is not bool:
                raise ValueError("native recheck actual Reply lifecycle flags must be booleans")
        if (bound is None or [actual_reply["process_epoch"],actual_reply["request_sequence"]] != reply_request
                or actual_reply["input_sha256"] != bound["data"]["input_sha256"]):
            raise ValueError("native recheck finished Reply RequestId/input mismatch")
        consumed = result["reply_observation"]["status"] == "consumed_raw_observation"
        if (actual_reply["search_consumed"] is not consumed or final["reply_accepted"] is not consumed
                or consumed and (not actual_reply["physical"] or not actual_reply["delivered"] or actual_reply["rejected"] or actual_reply["physical_unknown"])
                or final["reply_accepted"] and final["reply_call_attempted"] is not True):
            raise ValueError("native recheck engine accepted Reply contradicts actual physical/delivery/consumption")
    elif bound is not None or final["reply_accepted"]:
        raise ValueError("native recheck finished missing actual bound Reply")
    repair._moves(final["counterline"])
    if final["selected_response"] is not None:
        _int(final["selected_response"], 0, 65535)
        if (reply_snapshot is None or final["selected_response"] not in reply_snapshot["legal_moves"] or final["selected_response"] == line[ply]
                or final["counterline"][:ply] != line[:ply] or len(final["counterline"]) <= ply or final["counterline"][ply] != final["selected_response"]):
            raise ValueError("native recheck selected alternative is not the exact actual Reply/root prefix")
    if final["full_suffix_replayed"] and (final["counterline"][:ply] != line[:ply] or len(final["counterline"]) != len(line)
                                          or final["counterline"][ply+1:] != line[ply+1:] or final["selected_response"] is None):
        raise ValueError("native recheck full counterline is not the same repaired suffix")
    after_repaired = _endpoint(final["repaired_endpoint"], prepared=prepared, value_identity=value_identity, root_white=root["white_to_move"], plies=len(line))
    if final["repaired_endpoint"] != prepared["repaired_endpoint"]:
        raise ValueError("native recheck immutable repaired endpoint changed after Reply")
    counter_evidence = None if final["counter_endpoint"] is None else _endpoint(final["counter_endpoint"], prepared=prepared, value_identity=value_identity,
                                                                               root_white=root["white_to_move"], plies=len(final["counterline"]))
    result.update({"disposition":final["disposition"], "original_error":final["original_error"], "finished_cancelled":final["cancelled_at_observer"],
                   "finished_deadline_expired":final["deadline_expired_at_observer"], "counter_endpoint":counter_evidence,
                   "publication_observed":final["publication"] is not None, "publication":copy.deepcopy(final["publication"])})
    _publication_shape(final["publication"], prepared=prepared, identity=identity, root_white=root["white_to_move"],
                       disposition=final["disposition"], public_revision=anticipated["public_revision"],
                       export_manifest_sha256=source["native"]["independent_registry"]["export_manifest_sha256"])
    if final["disposition"] == "ConditionalRefutationPublished" and (final["publication"] is None or final["full_suffix_replayed"] is not True or final["comparable"] is not True):
        raise ValueError("native recheck published disposition lacks actual full conditional publication")
    # All raw identity/lifecycle contradictions above are checked BEFORE any
    # unresolved branch. A stop is not silently transformed into a good result.
    if (prepared["cancelled_at_observer"] or prepared["deadline_expired_at_observer"] or final["cancelled_at_observer"]
            or final["deadline_expired_at_observer"] or final["original_error"] is not None):
        result["reason"] = "cancelled_expired_or_failed_search_preserved"
        return result
    if (final["disposition"] not in ("CounterNotLower", "ConditionalRefutationPublished") or final["reply_accepted"] is not True
            or final["full_suffix_replayed"] is not True or final["comparable"] is not True or counter_evidence is None
            or after_repaired["status"] != "completed_cp" or counter_evidence["status"] != "completed_cp"):
        result["reason"] = "partial_missing_terminal_or_incomparable_endpoint_observation"
        return result
    if any(after_repaired[name] != counter_evidence[name] for name in ("depth", "profile", "condition", "value_identity")):
        raise ValueError("native recheck declared comparability differs from actual completed CPU namespace/depth")
    if any(check is None for check in checks):
        result["reason"] = "full_line_registered_Rules_capability_missing"
        return result
    repaired_rules = _rules_line(checks[0], initial, prefix=[], claim=line, endpoint=prepared["repaired_endpoint"], expected_sha256=pins["repaired_rules_sha256"])
    reply_rules = _rules_line(checks[1], initial, prefix=line[:ply], claim=final["counterline"][ply:], endpoint=final["counter_endpoint"], expected_sha256=pins["reply_rules_sha256"])
    if (reply_rules["target"]["rules_state_sha256"] != prepared["anchor_rules_state_sha256"]
            or any(reply_rules["target"][name] != reply_snapshot[name] for name in ("rules_state_sha256", "rules_history_sha256", "board_fen", "legal_moves"))):
        raise ValueError("native recheck full known-history Reply prefix differs from actual prepared input")
    lower = counter_evidence["root_score"] < after_repaired["root_score"]
    if (final["disposition"] == "ConditionalRefutationPublished") is not lower:
        raise ValueError("native recheck actual conditional conclusion contradicts endpoint parity comparison")
    publication = final["publication"]
    if lower:
        summary_bits = struct.unpack("<I",struct.pack("<f",counter_evidence["root_score"]))[0]
        if publication["observation"]["score"]["value_f32_bits"] != summary_bits:
            raise ValueError("native recheck actual publication does not bind the repaired line/comparison/conclusion")
    elif publication is not None:
        raise ValueError("native recheck not-lower observation cannot claim a refutation publication")
    result.update({"status":"conditional_cp_observation", "reason":None,
                   "raw_repaired_cp":after_repaired["raw_score"], "raw_counter_cp":counter_evidence["raw_score"],
                   "root_repaired_cp":after_repaired["root_score"], "root_counter_cp":counter_evidence["root_score"],
                   "conditional_counter_lower":lower, "repaired_rules_sha256":checks[0].sha256, "reply_rules_sha256":checks[1].sha256})
    return result


class CheckedNativeRecheckWitness:
    """Immutable conditional observation; no target/collation/loss interface."""
    __slots__ = ("_anchor", "_trace", "_checks", "_pins", "_identity")

    def __init__(self, token=None, *, anchor=None, trace=None, checks=(), pins=None):
        if token is not _FACTORY:
            raise ValueError("unchecked native recheck witness refused")
        for name, value in (("_anchor",anchor), ("_trace",trace), ("_checks",tuple(checks)), ("_pins",canonical(pins)), ("_identity",digest(WITNESS_SCHEMA,pins))):
            object.__setattr__(self,name,value)

    def __setattr__(self,name,value):
        raise AttributeError("checked native recheck witness is immutable")

    def verify(self):
        pins = _parse(self._pins)
        if digest(WITNESS_SCHEMA,pins) != self._identity:
            raise ValueError("native recheck witness identity changed")
        _validate_witness(self._anchor,self._trace,self._checks,pins)
        return self

    @property
    def sha256(self):
        self.verify()
        return self._identity

    def observation(self):
        self.verify()
        return copy.deepcopy(_validate_witness(self._anchor,self._trace,self._checks,_parse(self._pins)))


def _validate_witness(anchor, trace_bytes, checks, pins):
    if type(anchor) is not CheckedNativeRecheckRepairAnchor:
        raise ValueError("exact factory-checked native opt-in Repair anchor required")
    _fields(pins, ("anchor_sha256", "trace", "identity", "descriptor_sha256", "repaired_rules_sha256", "reply_rules_sha256"))
    _recheck_identity(pins["identity"])
    _sha(pins["descriptor_sha256"])
    if len(checks) != 2:
        raise ValueError("two explicit whole repaired/Reply Rules capability slots required")
    for check, name in zip(checks,("repaired_rules_sha256","reply_rules_sha256")):
        if check is None:
            if pins[name] is not None:
                raise ValueError("missing native recheck Rules capability cannot carry a checked identity")
        elif type(check) is not CheckedSemanticInput or check.sha256 != _sha(pins[name]):
            raise ValueError("native recheck exact Rules capability/independent pin mismatch")
    anchor.verify()
    if anchor.sha256 != _sha(pins["anchor_sha256"]):
        raise ValueError("native recheck whole witness changed its exact Repair anchor")
    semantic_raws = tuple(raw for check in (*anchor._repair._checks,*checks) if check is not None for _,raw in check._raws)
    _aggregate((trace_bytes, *dict(anchor._raws).values(), *anchor._repair._raws, *(raw for _,raw in anchor._repair._artifacts), *semantic_raws))
    return _whole_trace(anchor,trace_bytes,checks,pins)


def admit_native_recheck_witness(*, anchor, trace_bytes, repaired_rules_check=None, reply_rules_check=None, expected_pins):
    """Conditional same-line CP observation from original receipt-backed bytes.

    Missing/full partial work remains unresolved. A completed result additionally
    needs the exact actual Repair call chain, prepaid Reply journal/physical/
    delivered/consumed facts, stored Own CPU provenance, and independently pinned
    registered Rust Rules replay capabilities. Neither a marker nor `comparable`
    alone substitutes for that evidence. No WDL/proof/strategic target is made.
    """
    pins = _parse(canonical(expected_pins))
    checks = (repaired_rules_check,reply_rules_check)
    _validate_witness(anchor,trace_bytes,checks,pins)
    return CheckedNativeRecheckWitness(_FACTORY,anchor=anchor,trace=trace_bytes,checks=checks,pins=pins)
