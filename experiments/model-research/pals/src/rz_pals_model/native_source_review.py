"""Closed historical transition reviews, independent of file I/O or a runtime.

The legacy caller first validates its original build, binary, manifest byte pin
and lane policy. This helper adds only a Root-reviewed source closure join. A
matching manifest and compiled digest cannot reissue a native capability, prove
an actual execution, or admit a Rules/training target. Nothing is enrolled by
hashing current files, caller options or the process environment.
"""
from __future__ import annotations

import copy
import unicodedata

from .generation_data import digest, fields, identifier, sha, uint


PROFILE_SCHEMA = "rz-pals-native-reviewed-transition-profile/1"
CLOSURE_DOMAIN = "rz-pals-native-reviewed-transition-closure/1"
PROOF_SCHEMA = "rz-pals-native-transition-source-review-proof/1"
_LANES = ("default_disabled", "legacy_single_reply")
_TRANSITION_PATHS = (
    "crates/rz-search/src/pals/engine.rs",
    "crates/rz-arena/src/pals_collect/native.rs",
)
_REQUIRED_COMPONENTS = (
    *_TRANSITION_PATHS,
    "crates/rz-search/src/pals/engine/semantic_state.rs",
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
_PROFILE_FIELDS = ("schema", "review_id", "lane", "transition_sources", "component_pins",
                   "compiled_search_implementation_sha256", "closure_sha256")
_MAX_FILES = 4096
_MAX_FILE_BYTES = 64 * 1024 * 1024

# Root alone adds literal profiles after a frozen-source review. Empty means
# unsupported, including for a well-formed source pair seen in a current tree.
# The table is intentionally not populated or changed by this module.
REVIEWED_CLOSURE_PROFILES = ()


class UnsupportedTransitionSource(ValueError):
    """A valid observation lacks this explicit reviewed closure or fact."""


def _path(value):
    if (type(value) is not str or not value or len(value.encode("utf-8")) > 4096
            or "\\" in value or ":" in value or any(part in ("", ".", "..") for part in value.split("/"))
            or any(unicodedata.category(char) == "Cc" for char in value)):
        raise ValueError("normalized relative transition source path required")
    return value


def _pin(value):
    fields(value, ("bytes", "sha256"), "transition source byte pin")
    return {"bytes": uint(value["bytes"], _MAX_FILE_BYTES), "sha256": sha(value["sha256"])}


def _pair(value):
    fields(value, _TRANSITION_PATHS, "closed transition source pair")
    return {path: _pin(value[path]) for path in _TRANSITION_PATHS}


def _component_map(value):
    if type(value) is not dict or not 1 <= len(value) <= _MAX_FILES:
        raise ValueError("bounded explicit transition component map required")
    return {path: _pin(value[path]) for path in sorted(_path(key) for key in value)}


def _profile(value):
    fields(value, _PROFILE_FIELDS, "reviewed transition source profile")
    if value["schema"] != PROFILE_SCHEMA or value["lane"] not in _LANES:
        raise ValueError("closed transition review profile schema/lane required")
    identifier(value["review_id"])
    pair, components = _pair(value["transition_sources"]), _component_map(value["component_pins"])
    if not set(_REQUIRED_COMPONENTS) <= set(components):
        raise ValueError("reviewed transition profile omits a required source helper")
    if any(components[path] != pair[path] for path in _TRANSITION_PATHS):
        raise ValueError("reviewed transition profile changed its original source pair")
    sha(value["compiled_search_implementation_sha256"])
    if digest(CLOSURE_DOMAIN, components) != sha(value["closure_sha256"]):
        raise ValueError("reviewed transition closure semantic seal mismatch")
    return value, pair, components


def _manifest_files(manifest):
    if type(manifest) is not dict:
        raise ValueError("original transition source manifest object required")
    files = manifest.get("files")
    if type(files) is not list or not 1 <= len(files) <= _MAX_FILES:
        raise ValueError("bounded original transition manifest files required")
    result = {}
    for entry in files:
        fields(entry, ("path", "bytes", "sha256"), "original transition manifest file")
        path = _path(entry["path"])
        if path in result:
            raise ValueError("duplicate original transition source manifest path")
        result[path] = _pin({name: entry[name] for name in ("bytes", "sha256")})
    return {path: result[path] for path in sorted(result)}


def review_transition_sources(*, pair, manifest, source, lane):
    """Match one literal reviewed lane/closure; return metadata only.

    Original byte hashes of the manifest/build/source descriptions and the
    actual binary are mandatory caller checks, not claims made by this proof.
    Extra manifest entries remain explicitly outside the reviewed authority.
    """
    if type(lane) is not str or lane not in _LANES:
        raise UnsupportedTransitionSource("unsupported_transition_review_lane")
    actual_pair, actual_files = _pair(pair), _manifest_files(manifest)
    if any(actual_files.get(path) != pin for path, pin in actual_pair.items()):
        raise ValueError("original transition source pair differs from its manifest pin")
    if type(REVIEWED_CLOSURE_PROFILES) is not tuple or len(REVIEWED_CLOSURE_PROFILES) > 256:
        raise ValueError("closed literal transition review registry required")
    matches = []
    for registered in REVIEWED_CLOSURE_PROFILES:
        profile, registered_pair, components = _profile(registered)
        if profile["lane"] == lane and registered_pair == actual_pair:
            matches.append((profile, components))
    if not matches:
        raise UnsupportedTransitionSource("unreviewed_transition_source_closure")
    if len(matches) != 1:
        raise ValueError("ambiguous duplicate literal transition source review")
    profile, components = matches[0]
    if any(actual_files.get(path) != pin for path, pin in components.items()):
        raise ValueError("original transition helper closure differs from reviewed manifest pins")
    if type(source) is not dict:
        raise ValueError("original checked transition source object required")
    native = source.get("native")
    if type(native) is not dict or "search_implementation_sha256" not in native:
        raise UnsupportedTransitionSource("original_compiled_search_closure_fact_missing")
    compiled = sha(native["search_implementation_sha256"])
    if compiled != profile["compiled_search_implementation_sha256"]:
        raise ValueError("original compiled search closure differs from reviewed source scope")
    return {"schema": PROOF_SCHEMA, "review_id": profile["review_id"], "lane": lane,
        "transition_sources": copy.deepcopy(actual_pair), "component_pins": copy.deepcopy(components),
        "compiled_search_implementation_sha256": compiled, "closure_sha256": profile["closure_sha256"],
        "unreviewed_manifest_files": [path for path in actual_files if path not in components],
        "historical_metadata_only": True, "execution_authorized": False,
        "original_native_authority_reissued": False, "rules_proof_authorized": False, "training_target_authorized": False}
