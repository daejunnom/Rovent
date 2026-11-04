"""Conditional, unique prepared-lineage witness for one native C D slot.

The retained wire has no D->Reply->Repair->publication causal IDs. This module
does not invent them. It accepts one uniquely observed sequential window of a
reviewed collector/search transition, conditional on independently pinned
caller build/collection observations and registered Rules-only capabilities.
Reply consumption does not prove that Reply's policy selected the response:
the reviewed engine can repair an independently discovered CPU response.

Auxiliary D ownership, ordinary current-row ownership and public raw-source
ownership remain separate. A complete trace can establish not_examined;
incomplete/unsupported evidence cannot. All policy/WDL/task/divergence/ranking,
strategic validity, training and product authorities remain false. No Rules
implementation, callback matching, model forward, loss or process runs here.

In the usual reviewed flow a published Repair is first selected by the next
D query. That exclusive next-D boundary remains not_examined here. Conditional
known covers only an unusually narrow branch/role-limit window with an actual
ordinary selected-public observation before that boundary. The synthetic
known fixture does not prove that path executed in a real collector.
"""
import copy
import math
import struct

from . import native_divergence as native
from . import repair_context as repair
from . import training
from .semantic_verifier import CheckedSemanticInput, byte_pin, canonical, digest, _fields, _int, _parse, _sha

SCHEMA = "rz-pals-native-slot-repair-witness/1"
SCOPE = "conditional_unique_prepared_lineage"
BUILD_SCHEMA = "rz-pals-root-collector-binary-build-registration/1"
MAX_BYTES = native.MAX_BYTES
_FACTORY = object()
_SOURCE_PATHS = ("crates/rz-search/src/pals/engine.rs", "crates/rz-arena/src/pals_collect/native.rs")
# A closed static review profile, not an assertion about any caller's process.
# The actual collector build manifest must separately bind this pair to its
# registered binary. Other source pairs require a new independent review.
_REVIEWED_SOURCES = (
    {"bytes": 260542, "sha256": "4469f9d8721264c2053380fd8900ef00b23602d42220adb99b8199cef8b62afb"},
    {"bytes": 73365, "sha256": "564d2b29d873df1c7b0424bfcee0b52a8f314cbfd88fa68ef35f4fda99a600d5"},
)
# Independently reviewed default-only successor. The unchanged collector calls
# PalsEngine::new -> Disabled; the new opt-in constructor is NOT admitted.
# Literal pairs only: neither a current file read nor one reviewed component
# can register an otherwise unknown engine/collector combination.
_REVIEWED_DISABLED_RECHECK_SOURCES = (
    {"bytes": 297307, "sha256": "9c0de95911cf54365abec3818f089c20287d80160ea05eee8acbd326f7b49d2c"},
    {"bytes": 73365, "sha256": "564d2b29d873df1c7b0424bfcee0b52a8f314cbfd88fa68ef35f4fda99a600d5"},
)
# Source-only review of the observer-enabled successor's DEFAULT branch.
# The opt-in registration, marker and aliases remain rejected below; this entry
# does not enroll selected recheck collections or claim a caller actually ran.
_REVIEWED_DISABLED_OBSERVER_SOURCES = (
    {"bytes": 343988, "sha256": "0302e49b6a9784641ca90aba17490d404b793098f38836b7b3cf42a12ee131d1"},
    {"bytes": 153861, "sha256": "098005ae44fe5c5a9b0a2181d8cde32a40c3eeaaf0e733e2d92ec6859fd424e5"},
)
# Independently re-reviewed let-chain-only successor of that DEFAULT path.
# Older bytes remain readable; selected recheck collections remain excluded.
_REVIEWED_DISABLED_OBSERVER_LINT_SOURCES = (
    {"bytes": 343988, "sha256": "0302e49b6a9784641ca90aba17490d404b793098f38836b7b3cf42a12ee131d1"},
    {"bytes": 153791, "sha256": "0b056c4f2aada6380b1345a067e4d2c139118a8636ad7dc6f1a67a0dc0c55144"},
)
# Independently reviewed search-return observer successor's DEFAULT branch.
# Its additive raw artifact grants no selected action, whole cost or utility.
# Historical pairs and the Disabled-only policy checks below remain intact.
_REVIEWED_DISABLED_SEARCH_RETURN_SOURCES = (
    {"bytes": 343988, "sha256": "0302e49b6a9784641ca90aba17490d404b793098f38836b7b3cf42a12ee131d1"},
    {"bytes": 193797, "sha256": "9653ed5ef33eced92b400aef40a7ad018a2668505d6a14568ae9d80e69dc65df"},
)
# Independently reviewed clock-fixture and equivalent let-chain successor.
# Keep the original search-return pair as a historical profile.
_REVIEWED_DISABLED_SEARCH_RETURN_CLOCK_LINT_SOURCES = (
    {"bytes": 343988, "sha256": "0302e49b6a9784641ca90aba17490d404b793098f38836b7b3cf42a12ee131d1"},
    {"bytes": 193840, "sha256": "2c2bfb1c8717a68429d315fa576268ba95b9b6106caed57400b885f656d04ae7"},
)
# Reviewed engine module/comment-only successor; its original body is preserved.
# The native collector does not call standalone replay. This profile grants no
# replay causal, witness or utility authority.
_REVIEWED_DISABLED_REPLAY_MODULE_SOURCES = (
    {"bytes": 344097, "sha256": "0a0a4800e940700721ec856ab86c8d8dbb581f6291fd78976ce1f918ea5a40c3"},
    {"bytes": 193840, "sha256": "2c2bfb1c8717a68429d315fa576268ba95b9b6106caed57400b885f656d04ae7"},
)
# Reviewed DEFAULT-only successor. The new C continuation is opt-in; the
# collector retains Disabled, and the policy checks below still exclude every
# selected lane. Literal source-pair review does not certify a native execution.
_REVIEWED_DISABLED_C_CONTINUATION_OWNER_SOURCES = (
    {"bytes": 354492, "sha256": "fab843bc71b7e4b3584a437b7b4d89515be8b7ae12a2ba36ade83086c74c6618"},
    {"bytes": 195372, "sha256": "e25fc205f895825607a15292a5c23ed6ccfe425f4c6134201a52fe68ca13df1c"},
)
# Reviewed DEFAULT-only branch of the separate multi-Reply observer.
# No selected lane, new trace, utility or execution authority is granted here.
_REVIEWED_DISABLED_NATIVE_CONTINUATION_OBSERVER_SOURCES = (
    {'bytes': 354492, 'sha256': 'fab843bc71b7e4b3584a437b7b4d89515be8b7ae12a2ba36ade83086c74c6618'},
    {'bytes': 212711, 'sha256': 'b813828e9e469feb0769a35bb721f8fbfdf4c7dace9da92907372a757b822ba6'},
)
# Reviewed DEFAULT branch only; v2 selection observations do not grant
# multi-Reply execution, endpoint, utility or training authority.
_REVIEWED_DISABLED_CONTINUATION_SELECTION_V2_SOURCES = (
    {'bytes': 355881, 'sha256': '32e393b4ad83bfd33a1b2205cd110cdc0cf007909e7e9c7f617819ccf5c4e93c'},
    {'bytes': 220199, 'sha256': 'eea0654a5c11d6e0d69b5c5df71363a8e2d8f4fb7ad02cf5d766fb4b06ec456f'},
)
# Reviewed DEFAULT-only successor: the collector now derives a selected return
# tuple from the live policy, while Disabled still skips that selected block.
# The legacy prepared-lineage transition is unchanged. This literal pair grants
# no opt-in continuation, execution, cost, utility or training authority.
_REVIEWED_DISABLED_SELECTED_IDENTITY_SOURCES = (
    {"bytes": 355881, "sha256": "32e393b4ad83bfd33a1b2205cd110cdc0cf007909e7e9c7f617819ccf5c4e93c"},
    {"bytes": 221690, "sha256": "8886c0e01d0c753fa0a1b5256e36f2230d41f5e561fca8ac7ae8f895b9e0813c"},
)
# Independently reviewed DEFAULT-only successor. Native's new tests cover the
# separate continuation revision guard; its Disabled production path and the
# engine bytes are unchanged. This pair enrolls no selected continuation lane.
_REVIEWED_DISABLED_CONTINUATION_REVISION_FIXTURE_SOURCES = (
    {"bytes": 355881, "sha256": "32e393b4ad83bfd33a1b2205cd110cdc0cf007909e7e9c7f617819ccf5c4e93c"},
    {"bytes": 224380, "sha256": "a286a2403317be7f5f058b65488e7c9722d2b7f4b9bf683ea95e5c9f51a7aaf8"},
)
# Independently reviewed frozen followup-core pair, legacy Disabled branch only.
# The old loader still calls the old constructor with archive off and legacy
# search identity. V4 resolver/frozen/queue selection is separate and rejected
# by the unchanged policy gate below. Typed retired CPU work grants no completed
# authority. This literal pair does not certify a build, execution or target.
_REVIEWED_DISABLED_FOLLOWUP_CORE_SOURCES = (
    {"bytes": 374235, "sha256": "b86af1f0e55f55add0d4cca277772f8d1c8edf3b3d93d147629c4ab6f881103a"},
    {"bytes": 233158, "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9"},
)
# Reviewed lint successor of the same Disabled branch: two identity PalsError
# conversions and an identical Result wrapper were removed. Historical pairs
# remain separate; no new policy, task, value or execution scope is granted.
_REVIEWED_DISABLED_FOLLOWUP_CORE_LINT_SOURCES = (
    {"bytes": 374216, "sha256": "4729c3595731d82e7252dd4aec316bf051092449d95f4fe8a4a83017038ee018"},
    {"bytes": 233158, "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9"},
)
# Reviewed controlled-allocation successor, still the archive-off legacy
# Disabled collector path. The controls keep raw facts separate from logical
# publication; Rules, typed tasks and the policy gate below retain authority.
# This whole literal pair does not admit archive/V4 policies or target labels.
_REVIEWED_DISABLED_FOLLOWUP_CONTROLLED_CORE_SOURCES = (
    {"bytes": 382810, "sha256": "752d328ae53dc1b4c06a1f3b7cbbd311834dbb59961e598c4752d8fcb9222cf4"},
    {"bytes": 233158, "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9"},
)
# Re-reviewed stopped-request/raw-prefix successor in the same Disabled lane.
# A stopped request reads only current Rules; a returned legal prefix remains
# raw when checked connection fails. Typed deadlines, publication controls and
# the closed policy gate still exclude archive/V4/ModelWdl/target authority.
_REVIEWED_DISABLED_FOLLOWUP_CONTROLLED_FALLBACK_SOURCES = (
    {"bytes": 391894, "sha256": "4770a55361a994210a2238240ed4d2dea1a24d7a8b85524cc7c2db31d19f5e19"},
    {"bytes": 233158, "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9"},
)
# Reviewed frozen live-binding successor, still the archive-off Disabled lane.
# Original controls and current Rules bind acceptance; a stopped request or
# legal raw prefix grants no publication, completed task or value authority.
# Compiled source digests do not certify a build, execution or target. The
# unchanged closed gate excludes archive/V4/ModelWdl and selected recheck lanes.
_REVIEWED_DISABLED_FOLLOWUP_LIVE_BINDING_SOURCES = (
    {"bytes": 396086, "sha256": "004db31f9be5133bd1925b373e6df60e97da3d50979a7eb5a031b1dad4ca397b"},
    {"bytes": 233158, "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9"},
)
# Reviewed observed-progress successor, archive-off legacy Disabled scope only.
# Old constructors select ReplaceOnFresh and the default root-preparation branch
# is a no-op. Held-parent requires the separate IterativeFrozenModelWdlV2/
# PausedStack constructor; this legacy collector never selects it. The move-only
# progress wrapper and unchanged collector retain the existing controls and Rules
# authority. This exact pair grants no held-parent, V4/ModelWdl, execution, cost,
# utility or training authority; the closed policy and raw-build gates still apply.
_REVIEWED_DISABLED_FOLLOWUP_OBSERVED_PROGRESS_SOURCES = (
    {"bytes": 402637, "sha256": "4b272ed825d98f51ace408b2798093e88487f6b29f94e38b9d2a753f8fc263d1"},
    {"bytes": 233158, "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9"},
)
# Reviewed ordered-scope successor, archive-off legacy Disabled scope only.
# ReplaceOnFresh/public FIFO stay unchanged. Ordered nonrecursive Repair/C
# requires ResumeAfterAcceptedRepairC plus IterativeFrozenModelWdlV2 in the
# separate held-parent-v2 constructor; this collector does not select it.
# Raw caller build/policy checks still precede this closed pair lookup. No
# held-parent execution, V4/ModelWdl, utility, cost or training is admitted.
_REVIEWED_DISABLED_FOLLOWUP_ORDERED_SCOPE_SOURCES = (
    {"bytes": 402843, "sha256": "70d4bae5ad4e19e86137e47cc66922182c5232cbe35b9b30d42bf43ec6850171"},
    {"bytes": 233158, "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9"},
)
# Independently reviewed optional-review successor, legacy Disabled scope only.
# Old constructors keep review_index/review_policy unset: input selection,
# publication and Fresh-pair hooks preserve the original full-record path.
# The collector adds only an Err/Review classification; no success is issued.
# Selected views, adaptive ports, archive and V4/ModelWdl remain unadmitted.
# Original build/source checks and the closed policy gate retain authority;
# this whole literal pair grants no execution, cost, utility or training proof.
_REVIEWED_DISABLED_FOLLOWUP_OPTIONAL_REVIEW_SOURCES = (
    {"bytes": 414372, "sha256": "ce40b6f9ff8d13a41da8c9482b48528c1437896e9571c9630790c4b11b262726"},
    {"bytes": 233208, "sha256": "02af59b9edbc350e6e1a204eb6543033bb4a859729eb2ed79c8b470cc6823a01"},
)
# Reviewed compiled-source inventory-only successor of the same Disabled lane.
# Engine adds only the consumption_observation source tuple to its implementation
# digest; collector bytes and all legacy control paths are unchanged. This pair
# grants no adaptive consumption, selected context, execution or target authority.
_REVIEWED_DISABLED_FOLLOWUP_COMPILED_CONSUMPTION_SOURCES = (
    {"bytes": 414554, "sha256": "0ba7c9e41b106f35516b8b234872a839f5b2f4227153ed79aa7ff1c503d12837"},
    {"bytes": 233208, "sha256": "02af59b9edbc350e6e1a204eb6543033bb4a859729eb2ed79c8b470cc6823a01"},
)
_SOURCE_REVIEW_PROFILES = (
    ("legacy_prepared_lineage_4469f9d", _REVIEWED_SOURCES),
    ("legacy_disabled_recheck_9c0de959", _REVIEWED_DISABLED_RECHECK_SOURCES),
    ("legacy_disabled_observer_0302e49b", _REVIEWED_DISABLED_OBSERVER_SOURCES),
    ("legacy_disabled_observer_lint_0b056c4f", _REVIEWED_DISABLED_OBSERVER_LINT_SOURCES),
    ("legacy_disabled_search_return_9653ed5e", _REVIEWED_DISABLED_SEARCH_RETURN_SOURCES),
    ("legacy_disabled_search_return_clock_lint_2c2bfb1c", _REVIEWED_DISABLED_SEARCH_RETURN_CLOCK_LINT_SOURCES),
    ("legacy_disabled_replay_module_0a0a4800_native_2c2bfb1c", _REVIEWED_DISABLED_REPLAY_MODULE_SOURCES),
    ("legacy_disabled_c_continuation_owner_fab843bc_native_e25fc205", _REVIEWED_DISABLED_C_CONTINUATION_OWNER_SOURCES),
    ("legacy_disabled_multi_reply_observer_b813828e", _REVIEWED_DISABLED_NATIVE_CONTINUATION_OBSERVER_SOURCES),
    ("legacy_disabled_continuation_selection_v2_eea0654a", _REVIEWED_DISABLED_CONTINUATION_SELECTION_V2_SOURCES),
    ("legacy_disabled_selected_identity_8886c0e0", _REVIEWED_DISABLED_SELECTED_IDENTITY_SOURCES),
    ("legacy_disabled_continuation_revision_fixture_a286a240", _REVIEWED_DISABLED_CONTINUATION_REVISION_FIXTURE_SOURCES),
    ("legacy_disabled_followup_core_b86af1f0_native_1ba6d1e5", _REVIEWED_DISABLED_FOLLOWUP_CORE_SOURCES),
    ("legacy_disabled_followup_core_lint_4729c359_native_1ba6d1e5", _REVIEWED_DISABLED_FOLLOWUP_CORE_LINT_SOURCES),
    ("legacy_disabled_followup_controlled_core_752d328a_native_1ba6d1e5", _REVIEWED_DISABLED_FOLLOWUP_CONTROLLED_CORE_SOURCES),
    ("legacy_disabled_followup_controlled_fallback_4770a553_native_1ba6d1e5", _REVIEWED_DISABLED_FOLLOWUP_CONTROLLED_FALLBACK_SOURCES),
    ("legacy_disabled_followup_live_binding_004db31f_native_1ba6d1e5", _REVIEWED_DISABLED_FOLLOWUP_LIVE_BINDING_SOURCES),
    ("legacy_disabled_followup_observed_progress_4b272ed8_native_1ba6d1e5", _REVIEWED_DISABLED_FOLLOWUP_OBSERVED_PROGRESS_SOURCES),
    ("legacy_disabled_followup_ordered_scope_70d4bae5_native_1ba6d1e5", _REVIEWED_DISABLED_FOLLOWUP_ORDERED_SCOPE_SOURCES),
    ("legacy_disabled_followup_optional_review_ce40b6f9_native_02af59b9", _REVIEWED_DISABLED_FOLLOWUP_OPTIONAL_REVIEW_SOURCES),
    ("legacy_disabled_followup_compiled_consumption_0ba7c9e4_native_02af59b9", _REVIEWED_DISABLED_FOLLOWUP_COMPILED_CONSUMPTION_SOURCES),
)
_DISABLED_SEARCH_VERSION = "pals-restricted-refinement/0.1"
_OPT_IN_LANE_FIELDS = ("pals_search_policy", "refinement_registration", "refinement_registration_sha256", "post_repair_recheck", "post_repair_recheck_policy", "post_repair_recheck_conditions",
    "post_repair_recheck_search_version", "refinement_policy", "refinement_conditions", "refinement_search_version")
_RAW_NAMES = ("work_summary", "build_registration", "source_manifest", "engine_source", "native_source")
_PIN_NAMES = (*_RAW_NAMES, "divergence_sha256", "slot", "initial_repair_sha256", "reply_rules_sha256", "line_rules_sha256")
_ROOT_FIELDS = ("game_id", "opening_id", "line_genealogy_id", "actual_history", "source", "frozen_epoch", "encoding_sha256")
_RULE_FIELDS = ("rules_state_sha256", "rules_history_sha256", "board_fen", "legal_moves")


class _Unsupported(Exception):
    """Known limitation, distinct from contradicted or malformed evidence."""


def _row_map(raw, identity):
    result = {}
    for wire, value in native._rows(raw):
        key = value.get(identity)
        _sha(key)
        if key in result:
            raise ValueError("duplicate native slot trace identity")
        result[key] = (wire, value)
    return result


def _pin(raw, expected):
    _fields(expected, ("bytes", "sha256"))
    _int(expected["bytes"], 0, MAX_BYTES)
    _sha(expected["sha256"])
    if type(raw) is not bytes or byte_pin(raw) != expected:
        raise ValueError("native slot immutable byte pin mismatch")


def _reviewed_source_pair(actual):
    """Closed pair lookup AFTER raw manifest/build checks; returns no aliases."""
    _fields(actual, _SOURCE_PATHS)
    for _, pair in _SOURCE_REVIEW_PROFILES:
        reviewed = dict(zip(_SOURCE_PATHS, pair))
        if actual == reviewed:
            return copy.deepcopy(reviewed)
    raise _Unsupported("unreviewed_collector_transition_source_pair")


def _source_review(raws, source):
    """Verify actual caller build/binary/manifest bytes before static matching."""
    build, manifest = _parse(raws["build_registration"]), _parse(raws["source_manifest"])
    if build.get("schema") != BUILD_SCHEMA:
        raise _Unsupported("unreviewed_collector_build_schema")
    if (build.get("actual_build_exit_code") != 0 or type(build.get("actual_build_exit_code")) is not int
            or build.get("source_verified_before_and_after_build") is not True
            or build.get("source_manifest") != byte_pin(raws["source_manifest"])):
        raise ValueError("independent successful pre/post collector build observation required")
    commit = build.get("source_commit")
    if (type(commit) is not str or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit)
            or manifest.get("source_commit") != commit):
        raise ValueError("collector build/source commit binding")
    binary = build.get("binary")
    if type(binary) is not dict or type(binary.get("bytes")) is not int or not 0 < binary["bytes"] <= 512 * 1024 * 1024:
        raise ValueError("bounded registered collector binary observation required")
    if _sha(binary.get("sha256")) != source["implementation_sha256"]:
        raise ValueError("collector build binary differs from actual registered native source")
    artifact = build.get("compiler_artifact")
    if (type(artifact) is not dict or artifact.get("reason") != "compiler-artifact"
            or artifact.get("target", {}).get("name") != "pals_collect"
            or "pals-collection-onnx" not in artifact.get("features", [])
            or type(artifact.get("fresh")) is not bool):
        raise ValueError("actual compiler artifact observation required; fresh reuse is not an execution proof")
    files = manifest.get("files")
    if type(files) is not list or not 1 <= len(files) <= 4096:
        raise ValueError("bounded original collector source manifest required")
    by_path = {}
    for entry in files:
        _fields(entry, ("path", "bytes", "sha256"))
        path = entry["path"]
        if type(path) is not str or not path or path in by_path:
            raise ValueError("unique collector manifest paths required")
        _int(entry["bytes"], 0, 64 * 1024 * 1024)
        _sha(entry["sha256"])
        by_path[path] = {key: entry[key] for key in ("bytes", "sha256")}
    actual_pair = {}
    for path, name in zip(_SOURCE_PATHS, ("engine_source", "native_source")):
        actual = byte_pin(raws[name])
        if by_path.get(path) != actual:
            raise ValueError("actual transition source bytes absent from original registered build manifest")
        actual_pair[path] = actual
    closure = None
    try:
        reviewed_pair = _reviewed_source_pair(actual_pair)
    except _Unsupported as historical_error:
        # Historical monolithic profiles keep their original bytes and lookup.
        # A split implementation must additionally match an independently
        # reviewed, literal helper closure; a manifest roster cannot enroll it.
        from .native_source_review import review_transition_sources, UnsupportedTransitionSource
        try:
            closure = review_transition_sources(pair=actual_pair, manifest=manifest,
                source=source, lane="default_disabled")
        except UnsupportedTransitionSource as error:
            if str(error) == "unreviewed_transition_source_closure":
                raise historical_error
            raise _Unsupported(str(error)) from error
        reviewed_pair = copy.deepcopy(actual_pair)
    native_source = source.get("native")
    if type(native_source) is not dict:
        raise ValueError("actual native collector source description required")
    policy_owners = (source, native_source, native_source.get("search_configuration", {}), native_source.get("independent_registry", {}))
    for owner in policy_owners:
        if type(owner) is not dict:
            raise ValueError("native collector policy description must be an object")
        if any(key in owner for key in _OPT_IN_LANE_FIELDS):
            raise _Unsupported("unreviewed_collector_refinement_policy")
    # Historical minimal descriptions omit this declaration. The reviewed
    # constructor/source pair still establishes only the Disabled lane; a
    # present declaration in ANY policy owner must agree. The CPU task source
    # is a separate namespace, outside these four collector policy owners.
    if any(owner.get("search_version", _DISABLED_SEARCH_VERSION) != _DISABLED_SEARCH_VERSION
           for owner in policy_owners):
        raise _Unsupported("unreviewed_collector_refinement_policy")
    result = {"source_commit": commit, "collector_binary_sha256": binary["sha256"],
            "build_registration": byte_pin(raws["build_registration"]), "source_manifest": byte_pin(raws["source_manifest"]),
            "reviewed_transition_sources": reviewed_pair,
            "assurance": "independently_pinned_caller_build_observation;static_source_pair_review;not_self_reported_execution_certification"}
    if closure is not None:
        result["reviewed_transition_closure"] = closure
    return result


def _source_profile(raws, source):
    """Historical private seam; the actual reviewed pair remains authoritative."""
    return _source_review(raws, source)


def _finite_bits(values, count):
    if type(values) is not list or len(values) != count:
        raise ValueError("native slot physical raw head shape")
    for bits in values:
        _int(bits, 0, 2**32 - 1)
        if not math.isfinite(struct.unpack("<f", struct.pack("<I", bits))[0]):
            raise ValueError("native slot physical raw head nonfinite")


def _trace(artifacts, work_raw, parents, source):
    """No caller-selected subtrace: compare the entire retained call roster."""
    inputs = _row_map(artifacts["inputs.jsonl"], "sha256")
    auxiliary = _row_map(artifacts["native-divergence-inputs.jsonl"], "sha256")
    if set(inputs) & set(auxiliary):
        raise ValueError("auxiliary D cannot become an ordinary current input")
    all_inputs = inputs | auxiliary
    sidecars = _row_map(artifacts["native-inputs.jsonl"], "input_sha256")
    aux_sidecars = _row_map(artifacts["native-divergence-sidecars.jsonl"], "input_sha256")
    if set(sidecars) & set(aux_sidecars):
        raise ValueError("duplicate ordinary/auxiliary sidecar ownership")
    sidecars |= aux_sidecars
    lineage = _row_map(artifacts["input-lineage.jsonl"], "input_sha256")
    journals = []
    for wire, journal in native._rows(artifacts["producer-prepared.jsonl"]):
        _fields(journal, ("prepared", "sha256"))
        prepared = journal["prepared"]
        if type(prepared) is not dict or prepared.get("native_request") is None:
            raise _Unsupported("mixed_or_missing_native_prepared_trace")
        identity = _sha(prepared.get("input_sha256"))
        native._request(prepared["native_request"])
        if journal["sha256"] != training._sorted_canonical(training.PREPARED_PRODUCER_DOMAIN, prepared):
            raise ValueError("native slot prepared seal mismatch")
        journals.append((wire, journal, identity))
    identities = [entry[2] for entry in journals]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate native slot prepared input")
    if not journals or set(identities) != set(all_inputs) or set(identities) != set(sidecars) or set(identities) != set(lineage):
        raise _Unsupported("missing_complete_prepared_input_sidecar_lineage_trace")
    events = native._rows(artifacts["native-events.jsonl"])
    outputs = native._rows(artifacts["native-raw-outputs.jsonl"])
    by_events, by_output = {}, {}
    for _, event in events:
        _fields(event, ("domain", "game_id", "process_epoch", "request_sequence", "input_sha256", "stage", "observer_elapsed_us", "detail"))
        native._request([event["process_epoch"], event["request_sequence"]])
        _sha(event["input_sha256"])
        _int(event["observer_elapsed_us"], 0, 2**64 - 1)
        if event["domain"] != "rz-pals-native-call-event/1":
            raise ValueError("native slot event domain")
        by_events.setdefault(event["input_sha256"], []).append(event)
    for _, output in outputs:
        _fields(output, ("domain", "process_epoch", "request_sequence", "input_sha256", "physical_completion_confirmed", "success", "raw"))
        # JSON booleans compare equal to integers in Python; compare only
        # after the existing typed unsigned request validator admits both.
        native._request([output["process_epoch"], output["request_sequence"]])
        identity = _sha(output["input_sha256"])
        if identity in by_output:
            raise ValueError("duplicate native slot physical output")
        by_output[identity] = output
    if set(by_events) != set(identities) or set(by_output) != set(identities):
        raise _Unsupported("missing_complete_physical_event_output_trace")
    trace, blocks, groups = [], [], []
    previous = None
    for _, journal, identity in journals:
        input_raw, frozen = all_inputs[identity]
        sidecar_raw, sidecar = sidecars[identity]
        lineage_raw, meaning = lineage[identity]
        snapshot = frozen["snapshot"]
        kind = meaning.get("native_query_kind")
        if kind not in ("Propose", "Reply", "Repair", "Divergence"):
            raise _Unsupported("unreviewed_native_call_kind")
        _fields(meaning, (*repair._LINEAGE, *(("native_divergence_context_sha256",) if "native_divergence_context_sha256" in meaning else ())))
        for key in ("actual_played_history", "virtual_prefix", "proposal", "divergence_plies"):
            if type(meaning[key]) is not list:
                raise ValueError("native slot ordered lineage lists")
        repair._moves(meaning["actual_played_history"], maximum=65536)
        repair._moves(meaning["virtual_prefix"])
        repair._moves(meaning["proposal"])
        if meaning["counterexample"] is not None:
            repair._moves(meaning["counterexample"])
        if (training.seal_snapshot(snapshot) != identity or snapshot["source"] != source["source"]
                or snapshot["frozen_epoch"] != source["frozen_epoch"] or snapshot["encoding_sha256"] != source["encoding_sha256"]
                or meaning["game_id"] != snapshot["game_id"] or meaning["actual_played_history"] != snapshot["actual_history"]
                or meaning["counterfactual_wdl"] != "masked"
                or meaning["actual_outcome_eligible"] is not (kind == "Propose" and not meaning["virtual_prefix"])):
            raise ValueError("native slot exact owned source/history/lineage")
        prepared = journal["prepared"]
        request = prepared["native_request"]
        if (request != [meaning["process_epoch"], meaning["request_sequence"]]
                or prepared.get("game_id") != snapshot["game_id"] or prepared.get("capture_sequence") != snapshot["capture_sequence"]
                or prepared.get("learning_input") is not (kind != "Divergence")
                or any(prepared.get(name) != byte_pin(raw) for name, raw in
                       (("input_json", input_raw), ("tensor_sidecar_json", sidecar_raw), ("lineage_json", lineage_raw)))):
            raise ValueError("native slot prepared bytes/request/ownership mismatch")
        tensor = _parse(sidecar["tensor_json"].encode(), MAX_BYTES)
        if kind == "Divergence":
            native._sidecar(sidecar, frozen, source)
            if identity not in auxiliary:
                raise ValueError("D requires separately owned false-learning auxiliary bytes")
        else:
            if identity not in inputs:
                raise _Unsupported("nonordinary_role_input_not_admitted")
            # Historical calls remain raw trace facts under their original
            # frozen producer admission. They do not acquire current labels.
            # Known witness endpoints below separately require current_indices.
            current = [i for i in parents.current_view.current_indices if parents.records[i]["input"]["sha256"] == identity]
            matches = [i for i, row in enumerate(parents.records) if row["input"]["sha256"] == identity]
            if not matches or len(current) > 1:
                raise _Unsupported("missing_or_ambiguous_frozen_ordinary_role_binding")
            index = current[0] if current else matches[0]
            row = parents.records[index]
            expected_role = "critic" if kind == "Reply" else "proposer"
            admitted = next((item for item in parents.frozen_admission["inputs"] if item["binding"]["input_sha256"] == identity), None)
            if (row["input"] != frozen or snapshot["role"] != expected_role or admitted is None
                    or admitted["prepared_evidence_sha256"] != journal["sha256"]
                    or admitted["producer"]["producer_id"] != prepared.get("producer_id")
                    or admitted["producer"]["registration_sha256"] != prepared.get("registration_sha256")):
                raise ValueError("actual strict ordinary current role/journal binding required")
            policy = admitted["producer"]["encoding_policy"]
            encoded = training.encoded_from_sidecar(sidecar, row, expected_encoder_source_sha256=policy["encoder_source_sha256"],
                                                    expected_model_epoch=policy["native_model_epoch"]["sha256"])
            if encoded != parents.encodings[identity]:
                raise ValueError("native slot ordinary encoding changed")
            # Preserve legacy query semantics. Reply uses purpose 1; the older
            # P-only helper intentionally does not admit Critic Reply.
            p_tensor = copy.deepcopy(tensor)
            if kind == "Reply":
                if repair._fp32(tensor["query"][0]) != repair._fp32(1):
                    raise ValueError("native Reply purpose must be captured value one")
                p_tensor["query"][0] = 0
            repair._native_query(p_tensor, snapshot, meaning, kind)
        call_events = by_events[identity]
        stages = [event["stage"] for event in call_events]
        if stages not in (["prepared", "physically_completed", "delivered", "search_consumed"],
                          ["prepared", "physically_completed", "delivered", "logically_rejected"]):
            if (len(stages) != len(set(stages)) or "search_consumed" in stages or "logically_rejected" in stages
                    or stages != ["prepared", "physically_completed", "delivered"][:len(stages)]):
                raise ValueError("late/double/misordered native call consumption observation")
            raise _Unsupported("missing_terminal_native_call_observation")
        for event, detail in zip(call_events[:3], ({"prepared_before_submit": True, "native_query_kind": kind, "producer_metadata_admitted": True},
                                                  {"success": True, "logical_acceptance_inferred": False}, {"search_consumed": False})):
            if (event["game_id"] != snapshot["game_id"] or [event["process_epoch"], event["request_sequence"]] != request
                    or canonical(event["detail"]) != canonical(detail)):
                raise ValueError("native slot exact physical/delivered event request attribution")
        last = call_events[-1]
        if (last["game_id"] != snapshot["game_id"] or [last["process_epoch"], last["request_sequence"]] != request
                or (stages[-1] == "search_consumed" and last["detail"] != {"search_consumed": True})):
            raise ValueError("native slot search consumption attribution")
        if stages[-1] == "logically_rejected":
            _fields(last["detail"], ("reason", "search_consumed"))
            if (last["detail"]["search_consumed"] is not False or type(last["detail"]["reason"]) is not str
                    or not 1 <= len(last["detail"]["reason"].encode()) <= 256):
                raise ValueError("native slot explicit bounded rejection observation")
        elapsed = [event["observer_elapsed_us"] for event in call_events]
        if elapsed != sorted(elapsed):
            raise ValueError("native slot call event chronology")
        output = by_output[identity]
        if (output["domain"] != "rz-pals-native-physical-raw/1" or output["physical_completion_confirmed"] is not True
                or output["success"] is not True or [output["process_epoch"], output["request_sequence"]] != request):
            raise ValueError("known actual native physical output/request required")
        heads = _fields(output["raw"], ("representation", "candidate_logits_bits", "wdl_logits_bits", "divergence_logits_bits",
                                        "task_logits_bits", "private_latent_bits", "prediction_is_future_label"))
        if heads["representation"] != "f32_ieee754_bits" or heads["prediction_is_future_label"] is not False or heads["task_logits_bits"] is not None:
            raise ValueError("native raw predictions cannot become labels")
        _finite_bits(heads["candidate_logits_bits"], len(tensor["candidates"]))
        _finite_bits(heads["wdl_logits_bits"], 3)
        _finite_bits(heads["private_latent_bits"], 6144)
        if kind == "Divergence":
            _finite_bits(heads["divergence_logits_bits"], len(meaning["divergence_plies"]))
        elif heads["divergence_logits_bits"] is not None:
            raise ValueError("ordinary native call has a D-only output")
        new_group = kind == "Propose" and not meaning["virtual_prefix"]
        if new_group:
            groups.append([])
        if not groups:
            raise _Unsupported("missing_initial_root_search_boundary")
        if groups[-1]:
            root_prepared = groups[-1][0]["journal"]["prepared"]
            if any(prepared.get(key) != root_prepared.get(key) for key in
                   ("game_id", "producer_id", "registration_sha256", "roster_sha256")):
                raise ValueError("one source-backed root window requires the same registered producer journal")
        if previous is not None:
            if (request[0] != previous["request"][0] or request[1] <= previous["request"][1]
                    or snapshot["capture_sequence"] <= previous["snapshot"]["capture_sequence"]):
                raise ValueError("native slot epoch/request/capture ordering")
            if not new_group and previous["elapsed"][-1] > elapsed[0]:
                raise ValueError("native slot sequential source transition chronology")
            if not new_group and snapshot["input_revision"] < previous["snapshot"]["input_revision"]:
                raise ValueError("native slot same-root input revision regressed")
        call = {"identity": identity, "kind": kind, "snapshot": snapshot, "lineage": meaning,
                "journal": journal, "tensor": tensor, "request": request, "elapsed": elapsed,
                "consumed": stages[-1] == "search_consumed", "index": len(trace)}
        trace.append(call)
        groups[-1].append(call)
        previous = call
        blocks.extend(call_events)
    if [value for _, value in events] != blocks or [value["input_sha256"] for _, value in outputs] != identities:
        raise ValueError("native slot full retained event/output file order mismatch")
    summaries = native._rows(work_raw)
    if len(summaries) != len(groups):
        raise _Unsupported("missing_complete_search_work_windows")
    for (_, summary), group in zip(summaries, groups):
        _fields(summary, ("domain", "cpu_nodes", "cpu_tasks_requested", "cpu_reports_returned", "cpu_work_observation_incomplete",
                          "cpu_task_configuration_sha256", "search_result", "role_calls", "search_consumed_role_outputs"))
        if summary["domain"] != "rz-pals-native-search-work/1" or summary["cpu_task_configuration_sha256"] != source["cpu_profile_sha256"]:
            raise ValueError("native slot search work/source attribution")
        for key in ("cpu_nodes", "cpu_tasks_requested", "cpu_reports_returned", "role_calls", "search_consumed_role_outputs"):
            _int(summary[key], 0, 2**64 - 1)
        if summary["cpu_work_observation_incomplete"] is not False or summary["search_result"] is not None:
            raise _Unsupported("incomplete_or_failed_search_window")
        if summary["role_calls"] != len(group) or summary["search_consumed_role_outputs"] != sum(call["consumed"] for call in group):
            raise _Unsupported("search_work_does_not_close_retained_call_trace")
    finish = _parse(artifacts["receipt.json"])["native_finish"]["receipt"]
    expected_counts = {"physically_completed_role_calls": len(trace), "completed_role_inputs": len(trace),
                       "delivered_role_inputs": len(trace), "search_consumed_role_inputs": sum(call["consumed"] for call in trace),
                       "failed_physical_role_calls": 0, "invalid_role_outputs": 0}
    if any(key not in finish for key in (*expected_counts, "process_epoch", "request_high_water")):
        raise _Unsupported("missing_full_collection_native_call_counters")
    for key, wanted in expected_counts.items():
        _int(finish[key], 0, 2**64 - 1)
        if finish[key] != wanted:
            raise _Unsupported("collection_counters_do_not_close_retained_call_trace")
    _int(finish["process_epoch"], 0, 2**64 - 1)
    _int(finish["request_high_water"], 0, 2**64 - 1)
    if finish["process_epoch"] != trace[0]["request"][0] or finish["request_high_water"] < trace[-1]["request"][1]:
        raise ValueError("native slot collection epoch/high-water attribution")
    # A complete selected-record trace is mandatory even for not_examined.
    # It is not enough to validate only the D's or chosen Repair's records.
    _publications(trace, artifacts, source)
    return trace, groups


def _known_history(receipt):
    descriptors = [receipt["root"], receipt["target"]]
    final = receipt["claimed_line"]["final_state"]
    if final is not None:
        descriptors.append(final)
    if any(item["history_completeness"] != "complete" or item["history_origin"] != "start_position"
           or item["repetition_history_complete"] is not True for item in descriptors):
        raise _Unsupported("unknown_full_rules_history")


def _rules(check, parents, root_index, pin):
    if type(check) is not CheckedSemanticInput:
        raise _Unsupported("missing_registered_rules_capability")
    if check.verify_parent(parents) != root_index or check.sha256 != _sha(pin):
        raise ValueError("native slot Rules capability belongs to another strict current root")
    receipt, common = check.rules_receipt(), check.common_query()
    _known_history(receipt)
    return common, receipt


def _publications(trace, artifacts, source):
    public_raw = artifacts["public-record-sources.jsonl"]
    rows = native._rows(public_raw) if public_raw else []
    found, unique, projection_changed = {}, {}, False
    for raw, value in rows:
        _fields(value, ("domain", "game_id", "record_index", "revision", "origin_state_id", "origin_state_id_is_advisory",
                        "origin_rules_state_sha256", "origin_rules_identity_observation", "kind", "line", "value",
                        "completed_depth", "scope", "white_score_perspective", "critical", "source_cpu_profile_sha256"))
        _int(value["record_index"], 1, 2**64 - 1)
        _int(value["revision"], 0, 2**64 - 1)
        repair._moves(value["line"])
        if (value["domain"] != "rz-pals-native-public-source/1" or value["source_cpu_profile_sha256"] != source["cpu_profile_sha256"]
                or value["origin_state_id_is_advisory"] is not True or value["origin_rules_state_sha256"] is not None
                or value["origin_rules_identity_observation"] != "unknown"):
            raise ValueError("native slot selected public raw source provenance")
        identity = byte_pin(raw)["sha256"]
        if identity in found:
            if found[identity][0] != raw:
                raise ValueError("same native public SHA has different lexical bytes")
            # The observer clears raw_sources per search. Exact source rows
            # can legitimately repeat in a retained multi-search collection.
            # Collapse only identical SHA AND original lexical bytes.
            continue
        key = (value["game_id"], value["revision"])
        if key in unique:
            previous = found[unique[key]][1]
            changes = {field for field in value if canonical(value[field]) != canonical(previous[field])}
            if changes and changes <= {"record_index", "critical"}:
                # Stable record identity is not fully projection-aware in this
                # narrow adapter. Reindexing/critical promotion can be normal;
                # it is unsupported rather than asserted to be contradictory.
                projection_changed = True
            else:
                raise ValueError("contradictory native public source identity or distinct lexical bytes")
        found[identity] = (raw, value, [])
        unique.setdefault(key, identity)
    # Inspect all raw variants before reporting the narrow unsupported case:
    # a projection change cannot mask a later immutable-field contradiction.
    if projection_changed:
        raise _Unsupported("unsupported_public_projection_change")
    for call in trace:
        observations = call["snapshot"]["public_records"]
        tokens = call["tensor"]["records"]
        if len(observations) != len(tokens):
            raise ValueError("native slot selected record/tensor ordering")
        for observation, token in zip(observations, tokens):
            identity = observation["observation_sha256"]
            if identity not in found:
                raise ValueError("missing actual selected public raw bytes")
            _, public, users = found[identity]
            if (public["game_id"] != call["snapshot"]["game_id"] or public["record_index"] != token["record_id"]
                    or public["revision"] != token["revision"] or public["revision"] != observation["situation_revision"]
                    or public["revision"] > call["snapshot"]["input_revision"] or public["critical"] is not token["critical"]):
                raise ValueError("native slot exact selected-public/tensor identity")
            users.append(call)
    if any(not users for _, _, users in found.values()):
        raise ValueError("unselected public raw source cannot establish publication")
    return found


def _chain(divergence, slot, initial, reply_check, line_check, trace, groups, artifacts, source, pins):
    frozen, context, _, admission = divergence._views()
    parents, root_index = divergence._parents, divergence._index
    site = context["divergence_sites"][slot]
    d = next((call for call in trace if call["identity"] == frozen["sha256"]), None)
    if d is None or d["request"] != context["native_request"] or d["kind"] != "Divergence":
        raise ValueError("exact D auxiliary/context absent from full trace")
    if not d["consumed"] or admission["native_search_consumed"] is not True:
        raise ValueError("unconsumed D cannot own a Reply/Repair slot witness")
    group = next(group for group in groups if d in group)
    root = group[0]
    root_identity = parents.records[root_index]["input"]["sha256"]
    if root["identity"] != root_identity or any(canonical(root["snapshot"][key]) != canonical(d["snapshot"][key]) for key in native._SAME_ROOT):
        raise ValueError("D window must start at its exact strict current root")
    stop = next((call["index"] for call in group if call["index"] > d["index"] and call["kind"] == "Divergence"), group[-1]["index"] + 1)
    window = [call for call in group if d["index"] < call["index"] < stop]
    if any(any(canonical(call["snapshot"][key]) != canonical(d["snapshot"][key]) for key in _ROOT_FIELDS) for call in window):
        raise ValueError("different root/history/source inside native D window")
    proposal, ply = context["proposal_move16"], site["divergence_ply"]
    if any(call["kind"] not in ("Reply", "Repair") or call["lineage"]["proposal"] != proposal for call in window):
        raise ValueError("native D window contradicts the reviewed fixed-proposal Reply/Repair transition")
    if window:
        first_reply = window[0]
        selected_sites = [entry for entry in context["divergence_sites"]
                          if first_reply["lineage"]["virtual_prefix"] == proposal[:entry["divergence_ply"]]]
        if (first_reply["kind"] != "Reply" or first_reply["lineage"]["counterexample"] is not None
                or len(selected_sites) != 1):
            raise ValueError("first native Reply is not one exact ordered D site in this window")
    prefix = proposal[:ply]
    replies = [call for call in window if call["kind"] == "Reply" and call["lineage"]["proposal"] == proposal
               and call["lineage"]["virtual_prefix"] == prefix and call["lineage"]["counterexample"] is None]
    base = {"status": "not_examined", "reason": "slot_reply_not_observed_in_exact_D_window", "slot": slot,
            "divergence_ply": ply, "window_start_native_request": d["request"],
            "window_end_exclusive_trace_index": stop, "window_native_requests": [call["request"] for call in window]}
    if len(replies) > 1:
        raise ValueError("ambiguous duplicate Reply for same D slot/window")
    if not replies:
        return base
    reply = replies[0]
    reply_current = [i for i in parents.current_view.current_indices if parents.records[i]["input"]["sha256"] == reply["identity"]]
    if len(reply_current) != 1:
        raise _Unsupported("slot_reply_has_no_unique_current_ordinary_binding")
    repairs = [call for call in window if call["kind"] == "Repair" and call["lineage"]["proposal"] == proposal
               and len(call["lineage"]["virtual_prefix"]) == ply + 1 and call["lineage"]["virtual_prefix"][:ply] == prefix]
    if len(repairs) > 1:
        # CPU and C response branches are both possible. Caller choice cannot
        # collapse them into one slot's unique actual repaired-line witness.
        raise ValueError("ambiguous multiple initial Repair branches for one D slot/window")
    if not reply["consumed"]:
        if repairs:
            raise ValueError("Repair contradicts an unconsumed Reply in the reviewed transition")
        return {**base, "reason": "slot_reply_not_search_consumed"}
    if not repairs:
        return {**base, "reason": "initial_repair_not_observed"}
    first = repairs[0]
    if first["index"] <= reply["index"] or first["snapshot"]["input_revision"] < reply["snapshot"]["input_revision"]:
        raise ValueError("native Reply must precede initial Repair in same revision window")
    if not first["consumed"]:
        return {**base, "reason": "initial_repair_not_search_consumed"}
    if type(initial) is not repair.CheckedRepairContext:
        raise _Unsupported("missing_checked_initial_repair_context")
    initial.verify()
    if (initial._parents is not parents or initial._root_index != root_index or initial.sha256 != _sha(pins["initial_repair_sha256"])
            or parents.records[initial._repair_index]["input"]["sha256"] != first["identity"]):
        raise ValueError("initial Repair capability is not the unique D-window Repair")
    for name, raw in initial._artifacts:
        if artifacts.get(name) != raw:
            raise ValueError("initial Repair and D must retain the same complete raw collection bytes")
    if initial._raws[:4] != divergence._raws[:4]:
        raise ValueError("initial Repair/D registered source/launch authority differs")
    r_context = initial.context()
    if (r_context["divergence_ply"] != ply or r_context["proposal"] != proposal
            or r_context["virtual_prefix"] != first["lineage"]["virtual_prefix"]):
        raise ValueError("initial Repair capability disagrees with ordered D slot/prefix")
    common, rules = _rules(reply_check, parents, root_index, pins["reply_rules_sha256"])
    if (common["question"] != "unrestricted_recheck" or common["prefix"] != prefix or common["root_moves"] or common["claimed_line"]
            or rules["claimed_line"]["status"] != "no_claim"
            or any(rules["target"][key] != reply["snapshot"][key] for key in _RULE_FIELDS)
            or rules["target"]["rules_state_sha256"] != site["prefix_rules_state_sha256"]
            or rules["target"]["rules_history_sha256"] != site["prefix_rules_history_sha256"]
            or rules["target"]["side_to_move"] != ("white" if reply["snapshot"]["white_to_move"] else "black")
            or tuple(rules["target"]["board64_piece_codes"]) != parents.encodings[reply["identity"]].board):
        raise ValueError("actual registered Rules prefix does not reach the exact Reply/D slot")
    for check in initial._checks:
        _known_history(check.rules_receipt())
    repair_prefix = first["lineage"]["virtual_prefix"]
    followers = [call for call in window if call["index"] >= first["index"] and call["kind"] == "Repair"
                 and call["lineage"]["proposal"] == proposal and call["lineage"]["virtual_prefix"][:len(repair_prefix)] == repair_prefix]
    for previous, following in zip(followers, followers[1:]):
        if (not following["consumed"] or following["lineage"]["counterexample"] != first["lineage"]["counterexample"]
                or following["index"] != previous["index"] + 1
                or len(following["lineage"]["virtual_prefix"]) != len(previous["lineage"]["virtual_prefix"]) + 1
                or following["lineage"]["virtual_prefix"][:-1] != previous["lineage"]["virtual_prefix"]
                or following["snapshot"]["input_revision"] != first["snapshot"]["input_revision"]):
            raise ValueError("ambiguous/nonsequential P Repair continuation in D window")
    selected = _publications(trace, artifacts, source)
    last = followers[-1]
    candidates = []
    for raw, public, users in selected.values():
        if (public["kind"] != "Repair" or public["game_id"] != first["snapshot"]["game_id"]
                or public["line"][:len(repair_prefix)] != repair_prefix or public["revision"] <= last["snapshot"]["input_revision"]):
            continue
        first_user = users[0]
        if first_user["index"] <= last["index"]:
            raise ValueError("public Repair predates the alleged prepared continuation")
        if first_user["index"] >= stop:
            continue  # Another D/round cannot fill this slot's missing evidence.
        if first_user["kind"] == "Divergence":
            raise _Unsupported("auxiliary_publication_not_an_ordinary_current_observation")
        candidates.append((raw, public, first_user))
    if len(candidates) > 1:
        raise ValueError("ambiguous multiple public repaired lines in one D slot/window")
    if not candidates:
        return {**base, "reason": "full_repaired_line_publication_not_observed"}
    public_raw, public, observer = candidates[0]
    publication_current = [i for i in parents.current_view.current_indices if parents.records[i]["input"]["sha256"] == observer["identity"]]
    if len(publication_current) != 1:
        raise _Unsupported("publication_has_no_unique_current_ordinary_binding")
    if not observer["consumed"]:
        return {**base, "reason": "publication_observer_not_search_consumed"}
    line = public["line"]
    if len(line) != len(last["lineage"]["virtual_prefix"]) + 1 or line[:-1] != last["lineage"]["virtual_prefix"]:
        raise ValueError("published Repair must be the complete observed P continuation")
    line_common, line_rules = _rules(line_check, parents, root_index, pins["line_rules_sha256"])
    claim = line_rules["claimed_line"]
    if (line_common["question"] != "continuation_challenge" or line_common["prefix"] or line_common["root_moves"]
            or line_common["claimed_line"] != line or claim["status"] != "legal_continuation"
            or claim["legality_verified"] is not True or claim["claim_truth"] != "unknown"):
        raise ValueError("actual full root-anchored repaired line Rules legality required")
    return {**base, "status": "known", "reason": "unique_observed_prepared_lineage",
            "reply_input_sha256": reply["identity"], "reply_native_request": reply["request"],
            "initial_repair_input_sha256": first["identity"], "initial_repair_native_request": first["request"],
            "repair_context_sha256": initial.sha256, "reply_rules_sha256": reply_check.sha256,
            "repaired_line_rules_sha256": line_check.sha256, "repaired_line": line,
            "repair_continuation_native_requests": [call["request"] for call in followers],
            "publication_input_sha256": observer["identity"], "publication_native_request": observer["request"],
            "publication_prepared_sha256": observer["journal"]["sha256"], "public_source": byte_pin(public_raw),
            "public_record_index": public["record_index"], "public_record_revision": public["revision"],
            "endpoint_rules_state_sha256": claim["final_state"]["rules_state_sha256"],
            "endpoint_rules_history_sha256": claim["final_state"]["rules_history_sha256"]}


def _validate(divergence, slot, initial, reply_check, line_check, raws, pins):
    if type(divergence) is not native.CheckedNativeDivergence:
        raise ValueError("exact checked native D capability required; descriptor-only admission refused")
    divergence.verify()
    _fields(raws, _RAW_NAMES)
    _fields(pins, _PIN_NAMES)
    if any(type(value) is not bytes for value in raws.values()) or sum(map(len, raws.values())) > MAX_BYTES:
        raise ValueError("bounded immutable native slot extra assets required")
    for name in _RAW_NAMES:
        _pin(raws[name], pins[name])
    if divergence.sha256 != _sha(pins["divergence_sha256"]) or type(slot) is not int or pins["slot"] != slot:
        raise ValueError("native slot independent D/ordered-slot identity mismatch")
    context = divergence.context()
    _int(slot, 0, len(context["divergence_sites"]) - 1)
    _int(pins["slot"], 0, len(context["divergence_sites"]) - 1)
    for check, name, kind in ((initial, "initial_repair_sha256", repair.CheckedRepairContext),
                              (reply_check, "reply_rules_sha256", CheckedSemanticInput), (line_check, "line_rules_sha256", CheckedSemanticInput)):
        if check is None:
            if pins[name] is not None:
                raise ValueError("missing native slot capability has a positive caller pin")
        elif type(check) is not kind or check.sha256 != _sha(pins[name]):
            raise ValueError("native slot capability type or independent pin mismatch")
    artifacts = dict(divergence._artifacts)
    receipt = _parse(artifacts["receipt.json"])
    work_pin = receipt["artifacts"].get("native-work-summary.jsonl")
    body = {"schema": SCHEMA, "scope": SCOPE, "provenance_mode": "derived_from_existing_prepared_bytes",
            "direct_causal_ids_present": False, "before_dispatch_witness_claimed": False,
            "source_reachability_scope": "ordinary_selected_publication_before_next_D_only;usual_next_D_publication_not_examined;narrow_branch_role_limit_known_path_not_proved_by_synthetic_fixture",
            "divergence_sha256": divergence.sha256, "auxiliary_input_sha256": context["input_sha256"],
            "divergence_context_sha256": context["sha256"], "native_request": context["native_request"],
            "captured_input_revision": context["captured_input_revision"], "slot": slot,
            "challenged_line_sha256": context["challenged_line_sha256"], "proposal": context["proposal_move16"],
            "site": context["divergence_sites"][slot], "current_view_sha256": divergence._parents.current_view.sha256,
            "frozen_admission_sha256": divergence._parents._frozen_admission_identity,
            "collection_receipt_sha256": byte_pin(artifacts["receipt.json"])["sha256"],
            "extra_assets": {name: byte_pin(value) for name, value in raws.items()}}
    source = _parse(divergence._raws[1])[1]
    try:
        profile = _source_review(raws, source)
        body["transition_profile"] = profile
        if work_pin is None:
            raise _Unsupported("missing_original_work_summary_receipt_pin")
        _pin(raws["work_summary"], work_pin)
        trace, groups = _trace(artifacts, raws["work_summary"], divergence._parents, source)
        body.update(_chain(divergence, slot, initial, reply_check, line_check, trace, groups, artifacts, source, pins))
    except _Unsupported as limitation:
        body.update(status="unsupported", reason=str(limitation))
    admission = {"schema": SCHEMA, "scope": SCOPE, "status": body["status"], "reason": body["reason"],
                 "context_sha256": digest(SCHEMA, body), "divergence_sha256": divergence.sha256, "slot": slot,
                 "auxiliary_has_current_label": False, "direct_causal_identity_admitted": False,
                 "reply_policy_selected_response_admitted": False, "all_target_masks_false": True,
                 "strategic_repair_validity_admitted": False, "counterexample_validity_admitted": False,
                 "divergence_ranking_admitted": False, "whole_line_ordinal_admitted": False,
                 "policy_target_created": False, "wdl_target_created": False, "task_target_created": False,
                 "divergence_target_created": False, "training_target_created": False,
                 "loss_executed": False, "actual_training_executed": False, "product_verifier_enabled": False,
                 "scope_limit": "one_exact_D_ordered_slot;no_cross_D_round_fill;future_two_slot_repaired_line_comparison_requires_separate_minimize_criterion"}
    return body, admission


class CheckedNativeSlotRepair:
    """Immutable raw-backed observation; only status known is a slot witness.

    Reload requires the original strict-parent/capability factories and every
    raw asset/pin. This type deliberately has no descriptor/JSON deserializer.
    Python is not a hostile sandbox; independently registered caller evidence
    remains an external trust boundary, as in the underlying capabilities.
    """
    __slots__ = ("_divergence", "_slot", "_initial", "_reply", "_line", "_raws", "_pins", "_identity")

    def __init__(self, token=None, *, divergence=None, slot=None, initial=None, reply=None, line=None, raws=None, pins=None):
        if token is not _FACTORY:
            raise ValueError("unchecked native slot witness constructor refused")
        for name, value in (("_divergence", divergence), ("_slot", slot), ("_initial", initial), ("_reply", reply),
                            ("_line", line), ("_raws", tuple(sorted(raws.items()))), ("_pins", canonical(pins)),
                            ("_identity", digest(SCHEMA, pins))):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError("native slot witness is immutable")

    def _views(self):
        pins = _parse(self._pins)
        if digest(SCHEMA, pins) != self._identity:
            raise ValueError("native slot witness identity changed")
        return _validate(self._divergence, self._slot, self._initial, self._reply, self._line, dict(self._raws), pins)

    def verify(self):
        self._views()
        return self

    @property
    def sha256(self):
        self.verify()
        return self._identity

    @property
    def status(self):
        return self._views()[0]["status"]

    def context(self):
        return copy.deepcopy(self._views()[0])

    def admission(self):
        return copy.deepcopy(self._views()[1])

    def require_witness(self):
        if self.status != "known":
            raise ValueError("native slot witness is " + self.status)
        return self


def admit_native_slot_repair(*, divergence, slot, work_summary_bytes, build_registration_bytes, source_manifest_bytes,
                             engine_source_bytes, native_source_bytes, expected_pins, initial_repair=None,
                             reply_prefix_rules=None, repaired_line_rules=None):
    """Inspect one slot; no caller-selected matching indices or callbacks.

    Missing optional capabilities cannot produce known. A closed complete
    window can still establish not_examined without fabricating absent rows.
    Unsupported profile/trace/Rules coverage is retained separately; pin,
    identity, chronological or uniqueness contradictions raise ValueError.
    """
    raws = dict(zip(_RAW_NAMES, (work_summary_bytes, build_registration_bytes, source_manifest_bytes, engine_source_bytes, native_source_bytes)))
    pins = _parse(canonical(expected_pins))
    _validate(divergence, slot, initial_repair, reply_prefix_rules, repaired_line_rules, raws, pins)
    return CheckedNativeSlotRepair(_FACTORY, divergence=divergence, slot=slot, initial=initial_repair,
                                   reply=reply_prefix_rules, line=repaired_line_rules, raws=raws, pins=pins)
