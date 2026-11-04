"""One literal source review for NEW synthetic banks, never product enrollment.

The fourteen byte pins were reviewed at the fixed source revision below.
Reading the checkout only confirms those literal pins; it cannot register a
successor. The compiled fact is deliberately synthetic caller metadata. No
compiler, native process, Rules replay, training or GPU execution is certified.
Historical receipts and the production review registry remain unchanged.

The separate SYNTHETIC_CUSTODY helpers below serve ordinary byte-custody
unit tests without reading a checkout. They are not reviewed Rust sources,
independent-reference acceptance, product enrollment, or execution evidence.
"""
import copy
from contextlib import contextmanager
import hashlib
from unittest.mock import patch

from rz_pals_model import native_source_review as reviewed


REVIEWED_SOURCE_COMMIT = "6ca5946bceb9394d27575b8c3dc7df2ed9a48cb2"
REVIEWED_COMPONENT_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 422725, "sha256": "a86c61bc22dea02d505bc6526d482acc7dc197433b46229d6a3deed5d605e5ed"},
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 234273, "sha256": "83d54753084c62ecc14dc1f6c7b33e7f93f0d1cd4e2a5e5ce5b27a8a86746cbb"},
    "crates/rz-search/src/pals/engine/semantic_state.rs": {
        "bytes": 34003, "sha256": "aecd3584f37b4e4ea1105d6f60a977719e040cad0adfa832801af1089c22812c"},
    "crates/rz-search/src/pals/engine/researcher_admission.rs": {
        "bytes": 5957, "sha256": "5a1838c122e0e7eddd13b501f73cc94d9ea668333c70c1047111e9c137d3fdcd"},
    "crates/rz-search/src/pals/engine/researcher_profile.rs": {
        "bytes": 5443, "sha256": "118239c55485dcc88f4d82e29f60e68f47d3c4ba1e95b717368f41fab99ee810"},
    "crates/rz-search/src/pals/engine/review_lifecycle.rs": {
        "bytes": 37179, "sha256": "4a2059282d54f92ef05fda27d08f6383c9f6ef95d2f944b62c889217d3a8dc72"},
    "crates/rz-search/src/pals/engine/cpu_lifecycle.rs": {
        "bytes": 41911, "sha256": "2ad824a55e1f9d4b4215bf3f56663827955ac4f6efec37e5a695f0c54866fe0f"},
    "crates/rz-search/src/pals/engine/followup_execution.rs": {
        "bytes": 42122, "sha256": "72e85351261631f7bf57e95a53f95d8c8a6a079e3a320f27efcbefec2e1f938a"},
    "crates/rz-search/src/pals/engine/archive_lifecycle.rs": {
        "bytes": 154741, "sha256": "39120ddc5b9173d019e47f2f4757eda8a33676d2961b1120cd5395a1d9adb69c"},
    "crates/rz-arena/src/pals_collect.rs": {
        "bytes": 182486, "sha256": "608ed3c18d4a25e03e686c060a3c332a85443f8370406bfd9ed8f5e963b9e2e6"},
    "crates/rz-arena/src/pals_collect/full_line.rs": {
        "bytes": 18672, "sha256": "6f79dd8e05add10c258ebe873fd3db2642e39727e616ebf4e256d0c7f3b4ce30"},
    "crates/rz-arena/src/pals_collect/divergence.rs": {
        "bytes": 4980, "sha256": "f2916c4128767ffc0fe06407bc39024e2b6c2ec2e1eb952faeef998d7d7b1efd"},
    "crates/rz-arena/src/pals_collect/continuation.rs": {
        "bytes": 29927, "sha256": "22ce51044c1a4d3b61b3e62f9eae23198a004611e30ef14b20825ea5206f3963"},
    "crates/rz-arena/src/pals_collect/producer.rs": {
        "bytes": 21824, "sha256": "d72fad02feee033ce223f8d8608bd4847e6adc954232b82dd0c3abfd2065a304"},
}
REVIEWED_TRANSITION_PINS = {path: copy.deepcopy(REVIEWED_COMPONENT_PINS[path]) for path in (
    "crates/rz-search/src/pals/engine.rs", "crates/rz-arena/src/pals_collect/native.rs")}
REVIEWED_CLOSURE_SHA256 = "5a33f14ad40b1f8cec9bf5482abaa5944fbd8836c00c5d7abd808222f0d14903"
# SHA-256 of the fixed text "synthetic independently declared compiled closure
# fact\\n". This names no actual Rust compiler artifact or product binary.
SYNTHETIC_COMPILED_FACT = "f0cb4112737e170c89be3ba59c7da9dcaf893b15829f749238c5a8f5e5decb8c"


def fixture_profile(lane):
    if lane not in ("default_disabled", "legacy_single_reply"):
        raise ValueError("explicit synthetic review lane required")
    return {"schema": "rz-pals-native-reviewed-transition-profile/1",
        "review_id": "synthetic-reviewed-6ca5946-fourteen-source-" + lane,
        "lane": lane, "transition_sources": copy.deepcopy(REVIEWED_TRANSITION_PINS),
        "component_pins": copy.deepcopy(REVIEWED_COMPONENT_PINS),
        "compiled_search_implementation_sha256": SYNTHETIC_COMPILED_FACT,
        "closure_sha256": REVIEWED_CLOSURE_SHA256}


@contextmanager
def fixture_review_scope(lane):
    """Only this test owns the temporary registry; no retained enrollment."""
    profile = fixture_profile(lane)
    with patch.object(reviewed, "REVIEWED_CLOSURE_PROFILES", (profile,)):
        yield profile


def read_reviewed_fixture_sources(repo):
    """Observe literal reviewed bytes; never derive a new allowed source pin."""
    sources = {}
    for path, expected in REVIEWED_COMPONENT_PINS.items():
        with (repo / path).open("rb") as stream:
            actual = stream.read(expected["bytes"] + 1)
        if {"bytes": len(actual), "sha256": hashlib.sha256(actual).hexdigest()} != expected:
            raise AssertionError("reviewed Rust fixture source pin changed: " + path)
        sources[path] = actual
    return sources


def fixture_manifest():
    # Keep the transition pair first for byte-binding mutation cases. Ordering
    # does not replace the consumer's complete fourteen-path map validation.
    return {"source_commit": REVIEWED_SOURCE_COMMIT,
        "files": [{"path": path, **copy.deepcopy(pin)} for path, pin in REVIEWED_COMPONENT_PINS.items()]}


# Separate reviewed synthetic successor. The original 6ca5946 constants and
# fixture APIs above remain unchanged. The reviewed fixed Git/LF source
# diff: engine.rs adds only the four-line intervention.rs include_str entry to
# compiled_search_implementation_sha256; all other thirteen reviewed files are
# byte-identical. This is a source-inventory change, not an observed execution
# or permission to use either legacy lane for Rovent intervention work.
SUCCESSOR_REVIEWED_SOURCE_COMMIT = "932d7ae7006803dbb655249f7fa10e13483bdc12"
SUCCESSOR_REVIEWED_COMPONENT_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 422913, "sha256": "2729ef10909bdf21182d26b2edc4555f5cad7c17f55ac322111e7d867d8cbf53"},
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 234273, "sha256": "83d54753084c62ecc14dc1f6c7b33e7f93f0d1cd4e2a5e5ce5b27a8a86746cbb"},
    "crates/rz-search/src/pals/engine/semantic_state.rs": {
        "bytes": 34003, "sha256": "aecd3584f37b4e4ea1105d6f60a977719e040cad0adfa832801af1089c22812c"},
    "crates/rz-search/src/pals/engine/researcher_admission.rs": {
        "bytes": 5957, "sha256": "5a1838c122e0e7eddd13b501f73cc94d9ea668333c70c1047111e9c137d3fdcd"},
    "crates/rz-search/src/pals/engine/researcher_profile.rs": {
        "bytes": 5443, "sha256": "118239c55485dcc88f4d82e29f60e68f47d3c4ba1e95b717368f41fab99ee810"},
    "crates/rz-search/src/pals/engine/review_lifecycle.rs": {
        "bytes": 37179, "sha256": "4a2059282d54f92ef05fda27d08f6383c9f6ef95d2f944b62c889217d3a8dc72"},
    "crates/rz-search/src/pals/engine/cpu_lifecycle.rs": {
        "bytes": 41911, "sha256": "2ad824a55e1f9d4b4215bf3f56663827955ac4f6efec37e5a695f0c54866fe0f"},
    "crates/rz-search/src/pals/engine/followup_execution.rs": {
        "bytes": 42122, "sha256": "72e85351261631f7bf57e95a53f95d8c8a6a079e3a320f27efcbefec2e1f938a"},
    "crates/rz-search/src/pals/engine/archive_lifecycle.rs": {
        "bytes": 154741, "sha256": "39120ddc5b9173d019e47f2f4757eda8a33676d2961b1120cd5395a1d9adb69c"},
    "crates/rz-arena/src/pals_collect.rs": {
        "bytes": 182486, "sha256": "608ed3c18d4a25e03e686c060a3c332a85443f8370406bfd9ed8f5e963b9e2e6"},
    "crates/rz-arena/src/pals_collect/full_line.rs": {
        "bytes": 18672, "sha256": "6f79dd8e05add10c258ebe873fd3db2642e39727e616ebf4e256d0c7f3b4ce30"},
    "crates/rz-arena/src/pals_collect/divergence.rs": {
        "bytes": 4980, "sha256": "f2916c4128767ffc0fe06407bc39024e2b6c2ec2e1eb952faeef998d7d7b1efd"},
    "crates/rz-arena/src/pals_collect/continuation.rs": {
        "bytes": 29927, "sha256": "22ce51044c1a4d3b61b3e62f9eae23198a004611e30ef14b20825ea5206f3963"},
    "crates/rz-arena/src/pals_collect/producer.rs": {
        "bytes": 21824, "sha256": "d72fad02feee033ce223f8d8608bd4847e6adc954232b82dd0c3abfd2065a304"},
}
SUCCESSOR_REVIEWED_TRANSITION_PINS = {path: copy.deepcopy(SUCCESSOR_REVIEWED_COMPONENT_PINS[path]) for path in (
    "crates/rz-search/src/pals/engine.rs", "crates/rz-arena/src/pals_collect/native.rs")}
SUCCESSOR_REVIEWED_CLOSURE_SHA256 = "1af475f9fb1f82557ef5ad0d02a405f55118946be5063c2c78ec2d4c6b90c61c"
# SHA-256 of this independent fixture label, not an actual compiled artifact:
# "synthetic independently declared compiled closure fact for reviewed successor 932d7ae\n".
SUCCESSOR_SYNTHETIC_COMPILED_FACT = "f566cc430f184c29fdefbdce86fbd1663c63e89f05af61616f7de4fcd09789d3"


def successor_fixture_profile(lane):
    if lane not in ("default_disabled", "legacy_single_reply"):
        raise ValueError("explicit synthetic successor review lane required")
    return {"schema": "rz-pals-native-reviewed-transition-profile/1",
        "review_id": "synthetic-reviewed-932d7ae-fourteen-source-" + lane,
        "lane": lane, "transition_sources": copy.deepcopy(SUCCESSOR_REVIEWED_TRANSITION_PINS),
        "component_pins": copy.deepcopy(SUCCESSOR_REVIEWED_COMPONENT_PINS),
        "compiled_search_implementation_sha256": SUCCESSOR_SYNTHETIC_COMPILED_FACT,
        "closure_sha256": SUCCESSOR_REVIEWED_CLOSURE_SHA256}


@contextmanager
def successor_fixture_review_scope(lane):
    """Explicit test-local successor only; restore the original registry."""
    profile = successor_fixture_profile(lane)
    with patch.object(reviewed, "REVIEWED_CLOSURE_PROFILES", (profile,)):
        yield profile


def read_successor_fixture_sources(repo):
    """Compare raw bytes to fixed reviewed LF pins; never normalize or enroll."""
    sources = {}
    for path, expected in SUCCESSOR_REVIEWED_COMPONENT_PINS.items():
        with (repo / path).open("rb") as stream:
            actual = stream.read(expected["bytes"] + 1)
        if {"bytes": len(actual), "sha256": hashlib.sha256(actual).hexdigest()} != expected:
            raise AssertionError("reviewed successor Rust fixture source pin changed: " + path)
        sources[path] = actual
    return sources


def successor_fixture_manifest():
    return {"source_commit": SUCCESSOR_REVIEWED_SOURCE_COMMIT,
        "files": [{"path": path, **copy.deepcopy(pin)} for path, pin in SUCCESSOR_REVIEWED_COMPONENT_PINS.items()]}


# Separate PC-shared synthetic fixture, fixed to reviewed Git/LF bytes. Against
# 932d7ae, the fourteen-path scope changes only engine.rs: Root added the eight
# compiled-source inventory lines for shared_nnuec.rs and nnuec_v2_shared_readout.rs.
# The other thirteen files are unchanged. This table certifies no helper run,
# shared readout semantics, compiled binary, native retirement or target. Those
# remain separate actual checks; neither earlier fixture bank is rewritten.
PC_SHARED_REVIEWED_SOURCE_COMMIT = "47b602087de810dbe77c1489a7459202de3dfc9f"
PC_SHARED_REVIEWED_COMPONENT_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 423219, "sha256": "6f89838c7e95ff067bac86432c36ee221cf29acc334fe16d0f4f2b96c2d825d7"},
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 234273, "sha256": "83d54753084c62ecc14dc1f6c7b33e7f93f0d1cd4e2a5e5ce5b27a8a86746cbb"},
    "crates/rz-search/src/pals/engine/semantic_state.rs": {
        "bytes": 34003, "sha256": "aecd3584f37b4e4ea1105d6f60a977719e040cad0adfa832801af1089c22812c"},
    "crates/rz-search/src/pals/engine/researcher_admission.rs": {
        "bytes": 5957, "sha256": "5a1838c122e0e7eddd13b501f73cc94d9ea668333c70c1047111e9c137d3fdcd"},
    "crates/rz-search/src/pals/engine/researcher_profile.rs": {
        "bytes": 5443, "sha256": "118239c55485dcc88f4d82e29f60e68f47d3c4ba1e95b717368f41fab99ee810"},
    "crates/rz-search/src/pals/engine/review_lifecycle.rs": {
        "bytes": 37179, "sha256": "4a2059282d54f92ef05fda27d08f6383c9f6ef95d2f944b62c889217d3a8dc72"},
    "crates/rz-search/src/pals/engine/cpu_lifecycle.rs": {
        "bytes": 41911, "sha256": "2ad824a55e1f9d4b4215bf3f56663827955ac4f6efec37e5a695f0c54866fe0f"},
    "crates/rz-search/src/pals/engine/followup_execution.rs": {
        "bytes": 42122, "sha256": "72e85351261631f7bf57e95a53f95d8c8a6a079e3a320f27efcbefec2e1f938a"},
    "crates/rz-search/src/pals/engine/archive_lifecycle.rs": {
        "bytes": 154741, "sha256": "39120ddc5b9173d019e47f2f4757eda8a33676d2961b1120cd5395a1d9adb69c"},
    "crates/rz-arena/src/pals_collect.rs": {
        "bytes": 182486, "sha256": "608ed3c18d4a25e03e686c060a3c332a85443f8370406bfd9ed8f5e963b9e2e6"},
    "crates/rz-arena/src/pals_collect/full_line.rs": {
        "bytes": 18672, "sha256": "6f79dd8e05add10c258ebe873fd3db2642e39727e616ebf4e256d0c7f3b4ce30"},
    "crates/rz-arena/src/pals_collect/divergence.rs": {
        "bytes": 4980, "sha256": "f2916c4128767ffc0fe06407bc39024e2b6c2ec2e1eb952faeef998d7d7b1efd"},
    "crates/rz-arena/src/pals_collect/continuation.rs": {
        "bytes": 29927, "sha256": "22ce51044c1a4d3b61b3e62f9eae23198a004611e30ef14b20825ea5206f3963"},
    "crates/rz-arena/src/pals_collect/producer.rs": {
        "bytes": 21824, "sha256": "d72fad02feee033ce223f8d8608bd4847e6adc954232b82dd0c3abfd2065a304"},
}
PC_SHARED_REVIEWED_TRANSITION_PINS = {path: copy.deepcopy(PC_SHARED_REVIEWED_COMPONENT_PINS[path]) for path in (
    "crates/rz-search/src/pals/engine.rs", "crates/rz-arena/src/pals_collect/native.rs")}
PC_SHARED_REVIEWED_CLOSURE_SHA256 = "8ec6461fc6ee6cf7f5ab8d9a87f80e9b001286395edf89fd906cef2b78df5c7a"
# SHA-256 of this independent fixture label, not an actual compiled artifact:
# "synthetic independently declared compiled closure fact for reviewed PC shared source 47b6020\n".
PC_SHARED_SYNTHETIC_COMPILED_FACT = "1221b3e57973765549d2e489d995161c116622b24b02780adf2fb56cc3444b97"


def pc_shared_fixture_profile(lane):
    if lane not in ("default_disabled", "legacy_single_reply"):
        raise ValueError("explicit synthetic PC-shared review lane required")
    return {"schema": "rz-pals-native-reviewed-transition-profile/1",
        "review_id": "synthetic-reviewed-47b6020-pc-shared-fourteen-source-" + lane,
        "lane": lane, "transition_sources": copy.deepcopy(PC_SHARED_REVIEWED_TRANSITION_PINS),
        "component_pins": copy.deepcopy(PC_SHARED_REVIEWED_COMPONENT_PINS),
        "compiled_search_implementation_sha256": PC_SHARED_SYNTHETIC_COMPILED_FACT,
        "closure_sha256": PC_SHARED_REVIEWED_CLOSURE_SHA256}


@contextmanager
def pc_shared_fixture_review_scope(lane):
    """Explicit test-local PC-shared bank only; restore the original registry."""
    profile = pc_shared_fixture_profile(lane)
    with patch.object(reviewed, "REVIEWED_CLOSURE_PROFILES", (profile,)):
        yield profile


def read_pc_shared_fixture_sources(repo):
    """Compare raw bytes to the reviewed PC-shared LF pins without normalization."""
    sources = {}
    for path, expected in PC_SHARED_REVIEWED_COMPONENT_PINS.items():
        with (repo / path).open("rb") as stream:
            actual = stream.read(expected["bytes"] + 1)
        if {"bytes": len(actual), "sha256": hashlib.sha256(actual).hexdigest()} != expected:
            raise AssertionError("reviewed PC-shared Rust fixture source pin changed: " + path)
        sources[path] = actual
    return sources


def pc_shared_fixture_manifest():
    return {"source_commit": PC_SHARED_REVIEWED_SOURCE_COMMIT,
        "files": [{"path": path, **copy.deepcopy(pin)} for path, pin in PC_SHARED_REVIEWED_COMPONENT_PINS.items()]}


# Separate reviewed bundle-capacity successor, pinned to original Git/LF
# bytes. Against 47b6020, this fourteen-path scope changes only engine.rs:
# the synchronous capacity match also handles BundleRecordCapacity. No other
# source in this scope changed. Earlier three banks remain byte-for-byte
# declarations of their original reviews; this bank enrolls no product source.
BUNDLE_CAPACITY_REVIEWED_SOURCE_COMMIT = "f1b82eae0f08241b53e885ed2e6e6d573221d54a"
BUNDLE_CAPACITY_REVIEWED_COMPONENT_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 423275, "sha256": "7f1322ba591aef3c5f1f566c30f4f8184b9a736c935136fa4fed2c96273635e4"},
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 234273, "sha256": "83d54753084c62ecc14dc1f6c7b33e7f93f0d1cd4e2a5e5ce5b27a8a86746cbb"},
    "crates/rz-search/src/pals/engine/semantic_state.rs": {
        "bytes": 34003, "sha256": "aecd3584f37b4e4ea1105d6f60a977719e040cad0adfa832801af1089c22812c"},
    "crates/rz-search/src/pals/engine/researcher_admission.rs": {
        "bytes": 5957, "sha256": "5a1838c122e0e7eddd13b501f73cc94d9ea668333c70c1047111e9c137d3fdcd"},
    "crates/rz-search/src/pals/engine/researcher_profile.rs": {
        "bytes": 5443, "sha256": "118239c55485dcc88f4d82e29f60e68f47d3c4ba1e95b717368f41fab99ee810"},
    "crates/rz-search/src/pals/engine/review_lifecycle.rs": {
        "bytes": 37179, "sha256": "4a2059282d54f92ef05fda27d08f6383c9f6ef95d2f944b62c889217d3a8dc72"},
    "crates/rz-search/src/pals/engine/cpu_lifecycle.rs": {
        "bytes": 41911, "sha256": "2ad824a55e1f9d4b4215bf3f56663827955ac4f6efec37e5a695f0c54866fe0f"},
    "crates/rz-search/src/pals/engine/followup_execution.rs": {
        "bytes": 42122, "sha256": "72e85351261631f7bf57e95a53f95d8c8a6a079e3a320f27efcbefec2e1f938a"},
    "crates/rz-search/src/pals/engine/archive_lifecycle.rs": {
        "bytes": 154741, "sha256": "39120ddc5b9173d019e47f2f4757eda8a33676d2961b1120cd5395a1d9adb69c"},
    "crates/rz-arena/src/pals_collect.rs": {
        "bytes": 182486, "sha256": "608ed3c18d4a25e03e686c060a3c332a85443f8370406bfd9ed8f5e963b9e2e6"},
    "crates/rz-arena/src/pals_collect/full_line.rs": {
        "bytes": 18672, "sha256": "6f79dd8e05add10c258ebe873fd3db2642e39727e616ebf4e256d0c7f3b4ce30"},
    "crates/rz-arena/src/pals_collect/divergence.rs": {
        "bytes": 4980, "sha256": "f2916c4128767ffc0fe06407bc39024e2b6c2ec2e1eb952faeef998d7d7b1efd"},
    "crates/rz-arena/src/pals_collect/continuation.rs": {
        "bytes": 29927, "sha256": "22ce51044c1a4d3b61b3e62f9eae23198a004611e30ef14b20825ea5206f3963"},
    "crates/rz-arena/src/pals_collect/producer.rs": {
        "bytes": 21824, "sha256": "d72fad02feee033ce223f8d8608bd4847e6adc954232b82dd0c3abfd2065a304"},
}
BUNDLE_CAPACITY_REVIEWED_TRANSITION_PINS = {path: copy.deepcopy(BUNDLE_CAPACITY_REVIEWED_COMPONENT_PINS[path]) for path in (
    "crates/rz-search/src/pals/engine.rs", "crates/rz-arena/src/pals_collect/native.rs")}
BUNDLE_CAPACITY_REVIEWED_CLOSURE_SHA256 = "16ec5581bf1e1966e9ca4f3a1e09a48ef81d97adc5f8c217280e1e82a1f7e704"
# SHA-256 of an independent synthetic label, never an actual compiler artifact:
# "synthetic independently declared compiled closure fact for reviewed bundle capacity source f1b82ea\n".
BUNDLE_CAPACITY_SYNTHETIC_COMPILED_FACT = "2133baa4c24d61bfca00b1ff874e8ee3da39ec2a57620667ffd6e415a7286c84"


def bundle_capacity_fixture_profile(lane):
    if lane not in ("default_disabled", "legacy_single_reply"):
        raise ValueError("explicit synthetic bundle-capacity review lane required")
    return {"schema": "rz-pals-native-reviewed-transition-profile/1",
        "review_id": "synthetic-reviewed-f1b82ea-bundle-capacity-fourteen-source-" + lane,
        "lane": lane, "transition_sources": copy.deepcopy(BUNDLE_CAPACITY_REVIEWED_TRANSITION_PINS),
        "component_pins": copy.deepcopy(BUNDLE_CAPACITY_REVIEWED_COMPONENT_PINS),
        "compiled_search_implementation_sha256": BUNDLE_CAPACITY_SYNTHETIC_COMPILED_FACT,
        "closure_sha256": BUNDLE_CAPACITY_REVIEWED_CLOSURE_SHA256}


@contextmanager
def bundle_capacity_fixture_review_scope(lane):
    """One fixed synthetic successor only; never change the product registry."""
    profile = bundle_capacity_fixture_profile(lane)
    with patch.object(reviewed, "REVIEWED_CLOSURE_PROFILES", (profile,)):
        yield profile


def read_bundle_capacity_fixture_sources(repo):
    """Compare exact reviewed LF bytes without normalization or enrollment."""
    sources = {}
    for path, expected in BUNDLE_CAPACITY_REVIEWED_COMPONENT_PINS.items():
        with (repo / path).open("rb") as stream:
            actual = stream.read(expected["bytes"] + 1)
        if {"bytes": len(actual), "sha256": hashlib.sha256(actual).hexdigest()} != expected:
            raise AssertionError("reviewed bundle-capacity Rust fixture source pin changed: " + path)
        sources[path] = actual
    return sources


def bundle_capacity_fixture_manifest():
    return {"source_commit": BUNDLE_CAPACITY_REVIEWED_SOURCE_COMMIT,
        "files": [{"path": path, **copy.deepcopy(pin)} for path, pin in BUNDLE_CAPACITY_REVIEWED_COMPONENT_PINS.items()]}

# Separate reviewed feedback-observation successor, fixed to original Git/LF
# bytes. Against f1b82ea, this fourteen-path scope changes only engine.rs:
# four include_str inventory lines name feedback_observation.rs. Other thirteen
# reviewed files are byte-identical. Earlier four banks and product registry
# remain unchanged. This synthetic fixture certifies no compiled artifact,
# observation semantics, lifecycle/physical completion, model or training run.
FEEDBACK_OBSERVATION_REVIEWED_SOURCE_COMMIT = "102b339618c55b399a3b2bb29c97c7b3ef5dbc76"
FEEDBACK_OBSERVATION_REVIEWED_COMPONENT_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 423497,
        "sha256": "2d25a6609561d042aa7cb4927c337e29f9202575bb37c839512f29a8b95cdac4"
    },
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 234273,
        "sha256": "83d54753084c62ecc14dc1f6c7b33e7f93f0d1cd4e2a5e5ce5b27a8a86746cbb"
    },
    "crates/rz-search/src/pals/engine/semantic_state.rs": {
        "bytes": 34003,
        "sha256": "aecd3584f37b4e4ea1105d6f60a977719e040cad0adfa832801af1089c22812c"
    },
    "crates/rz-search/src/pals/engine/researcher_admission.rs": {
        "bytes": 5957,
        "sha256": "5a1838c122e0e7eddd13b501f73cc94d9ea668333c70c1047111e9c137d3fdcd"
    },
    "crates/rz-search/src/pals/engine/researcher_profile.rs": {
        "bytes": 5443,
        "sha256": "118239c55485dcc88f4d82e29f60e68f47d3c4ba1e95b717368f41fab99ee810"
    },
    "crates/rz-search/src/pals/engine/review_lifecycle.rs": {
        "bytes": 37179,
        "sha256": "4a2059282d54f92ef05fda27d08f6383c9f6ef95d2f944b62c889217d3a8dc72"
    },
    "crates/rz-search/src/pals/engine/cpu_lifecycle.rs": {
        "bytes": 41911,
        "sha256": "2ad824a55e1f9d4b4215bf3f56663827955ac4f6efec37e5a695f0c54866fe0f"
    },
    "crates/rz-search/src/pals/engine/followup_execution.rs": {
        "bytes": 42122,
        "sha256": "72e85351261631f7bf57e95a53f95d8c8a6a079e3a320f27efcbefec2e1f938a"
    },
    "crates/rz-search/src/pals/engine/archive_lifecycle.rs": {
        "bytes": 154741,
        "sha256": "39120ddc5b9173d019e47f2f4757eda8a33676d2961b1120cd5395a1d9adb69c"
    },
    "crates/rz-arena/src/pals_collect.rs": {
        "bytes": 182486,
        "sha256": "608ed3c18d4a25e03e686c060a3c332a85443f8370406bfd9ed8f5e963b9e2e6"
    },
    "crates/rz-arena/src/pals_collect/full_line.rs": {
        "bytes": 18672,
        "sha256": "6f79dd8e05add10c258ebe873fd3db2642e39727e616ebf4e256d0c7f3b4ce30"
    },
    "crates/rz-arena/src/pals_collect/divergence.rs": {
        "bytes": 4980,
        "sha256": "f2916c4128767ffc0fe06407bc39024e2b6c2ec2e1eb952faeef998d7d7b1efd"
    },
    "crates/rz-arena/src/pals_collect/continuation.rs": {
        "bytes": 29927,
        "sha256": "22ce51044c1a4d3b61b3e62f9eae23198a004611e30ef14b20825ea5206f3963"
    },
    "crates/rz-arena/src/pals_collect/producer.rs": {
        "bytes": 21824,
        "sha256": "d72fad02feee033ce223f8d8608bd4847e6adc954232b82dd0c3abfd2065a304"
    }
}
FEEDBACK_OBSERVATION_REVIEWED_TRANSITION_PINS = {
    path: copy.deepcopy(FEEDBACK_OBSERVATION_REVIEWED_COMPONENT_PINS[path]) for path in (
        "crates/rz-search/src/pals/engine.rs", "crates/rz-arena/src/pals_collect/native.rs")}
FEEDBACK_OBSERVATION_REVIEWED_CLOSURE_SHA256 = "ff081ae138aa486e54f0966751d47909781f75055c869f5cc1fc4eaee5ce7acb"
# SHA-256 of an independent synthetic label, never an actual compiler artifact:
# "synthetic independently declared compiled closure fact for reviewed feedback observation source 102b339\n".
FEEDBACK_OBSERVATION_SYNTHETIC_COMPILED_FACT = "6d3e272975feee584ead05845efffd6c9e5e5ed4dc2a2e69ac868514fea98664"


def feedback_observation_fixture_profile(lane):
    if lane not in ("default_disabled", "legacy_single_reply"):
        raise ValueError("explicit synthetic feedback-observation review lane required")
    return {"schema": "rz-pals-native-reviewed-transition-profile/1",
        "review_id": "synthetic-reviewed-102b339-feedback-observation-fourteen-source-" + lane,
        "lane": lane, "transition_sources": copy.deepcopy(FEEDBACK_OBSERVATION_REVIEWED_TRANSITION_PINS),
        "component_pins": copy.deepcopy(FEEDBACK_OBSERVATION_REVIEWED_COMPONENT_PINS),
        "compiled_search_implementation_sha256": FEEDBACK_OBSERVATION_SYNTHETIC_COMPILED_FACT,
        "closure_sha256": FEEDBACK_OBSERVATION_REVIEWED_CLOSURE_SHA256}


@contextmanager
def feedback_observation_fixture_review_scope(lane):
    """One fixed synthetic successor only; never change the product registry."""
    profile = feedback_observation_fixture_profile(lane)
    with patch.object(reviewed, "REVIEWED_CLOSURE_PROFILES", (profile,)):
        yield profile


def read_feedback_observation_fixture_sources(repo):
    """Compare exact reviewed LF bytes without normalization or enrollment."""
    sources = {}
    for path, expected in FEEDBACK_OBSERVATION_REVIEWED_COMPONENT_PINS.items():
        with (repo / path).open("rb") as stream:
            actual = stream.read(expected["bytes"] + 1)
        if {"bytes": len(actual), "sha256": hashlib.sha256(actual).hexdigest()} != expected:
            raise AssertionError("reviewed feedback-observation Rust fixture source pin changed: " + path)
        sources[path] = actual
    return sources


def feedback_observation_fixture_manifest():
    return {"source_commit": FEEDBACK_OBSERVATION_REVIEWED_SOURCE_COMMIT,
        "files": [{"path": path, **copy.deepcopy(value)}
                  for path, value in FEEDBACK_OBSERVATION_REVIEWED_COMPONENT_PINS.items()]}


import unittest as _fixture_unittest


class FeedbackObservationFixtureTests(_fixture_unittest.TestCase):

    def test_feedback_observation_bank_preserves_four_prior_banks_and_requires_its_own_fact(self):
        import native_source_review_fixture as banks
        from rz_pals_model import native_source_review as reviewed
        factories = (banks.fixture_profile, banks.successor_fixture_profile,
                     banks.pc_shared_fixture_profile, banks.bundle_capacity_fixture_profile)
        prior = [factory("default_disabled") for factory in factories]
        current = banks.feedback_observation_fixture_profile("default_disabled")
        engine = 'crates/rz-search/src/pals/engine.rs'
        self.assertEqual(banks.feedback_observation_fixture_manifest()["source_commit"],
                         "102b339618c55b399a3b2bb29c97c7b3ef5dbc76")
        self.assertEqual(current["component_pins"][engine], {'bytes': 423497, 'sha256': '2d25a6609561d042aa7cb4927c337e29f9202575bb37c839512f29a8b95cdac4'})
        self.assertEqual(len(current["component_pins"]), 14)
        self.assertEqual([path for path in prior[-1]["component_pins"]
            if prior[-1]["component_pins"][path] != current["component_pins"][path]], [engine])
        self.assertEqual(len({profile["review_id"] for profile in [*prior, current]}), 5)
        before = copy.deepcopy(reviewed.REVIEWED_CLOSURE_PROFILES)
        for lane in ("default_disabled", "legacy_single_reply"):
            with banks.feedback_observation_fixture_review_scope(lane) as profile:
                manifest = banks.feedback_observation_fixture_manifest()
                proof = reviewed.review_transition_sources(pair=profile["transition_sources"], manifest=manifest,
                    source={"native": {"search_implementation_sha256": banks.FEEDBACK_OBSERVATION_SYNTHETIC_COMPILED_FACT}},
                    lane=lane)
                self.assertEqual(proof["closure_sha256"], banks.FEEDBACK_OBSERVATION_REVIEWED_CLOSURE_SHA256)
                self.assertIs(proof["historical_metadata_only"], True)
                for name in ("execution_authorized", "original_native_authority_reissued",
                             "rules_proof_authorized", "training_target_authorized"):
                    self.assertIs(proof[name], False)
                for old in prior:
                    with self.assertRaisesRegex(ValueError, "compiled search closure differs"):
                        reviewed.review_transition_sources(pair=profile["transition_sources"], manifest=manifest,
                            source={"native": {"search_implementation_sha256": old["compiled_search_implementation_sha256"]}},
                            lane=lane)
            self.assertEqual(reviewed.REVIEWED_CLOSURE_PROFILES, before)
        self.assertEqual([factory("default_disabled") for factory in factories], prior)


# Independent static unit bytes. Keep all historical literal banks intact.
# The paths model a complete logical closure; their content is deliberately not
# Rust implementation text, and this ID does not name a Git commit.
_SYNTHETIC_CUSTODY_PATHS = (*FEEDBACK_OBSERVATION_REVIEWED_COMPONENT_PINS,
    "crates/rz-search/src/pals/engine/idle_archive.rs",
    "crates/rz-search/src/pals/engine/adaptive_port/archive_worker.rs",
    "crates/rz-search/src/pals/engine/preparing_root.rs",
    "crates/rz-search/src/pals/review_context.rs",
    "crates/rz-search/src/pals/engine/adaptive_root.rs",
    "crates/rz-search/src/pals/store/archive.rs",
)
_SYNTHETIC_CUSTODY_BYTES = {
    path: ("synthetic byte-custody unit input only; not Rust source: " + path + "\n").encode("utf-8")
    for path in _SYNTHETIC_CUSTODY_PATHS}
SYNTHETIC_CUSTODY_COMPONENT_PINS = {
    path: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    for path, raw in _SYNTHETIC_CUSTODY_BYTES.items()}
SYNTHETIC_CUSTODY_TRANSITION_PINS = {
    path: copy.deepcopy(SYNTHETIC_CUSTODY_COMPONENT_PINS[path]) for path in _SYNTHETIC_CUSTODY_PATHS[:2]}
SYNTHETIC_CUSTODY_SOURCE_ID = hashlib.sha256(b"synthetic-byte-custody-unit-source-id-v1").hexdigest()[:40]
SYNTHETIC_CUSTODY_COMPILED_FACT = hashlib.sha256(b"synthetic-byte-custody-unit-compiled-fact-v1").hexdigest()


def synthetic_custody_fixture_sources():
    """Static unit bytes only; never observe, hash, or enroll a live checkout."""
    return dict(_SYNTHETIC_CUSTODY_BYTES)


def synthetic_custody_fixture_manifest():
    return {"source_commit": SYNTHETIC_CUSTODY_SOURCE_ID,
        "files": [{"path": path, **copy.deepcopy(pin)}
                  for path, pin in SYNTHETIC_CUSTODY_COMPONENT_PINS.items()]}


def synthetic_custody_fixture_profile(lane):
    if lane not in ("default_disabled", "legacy_single_reply"):
        raise ValueError("explicit synthetic custody unit lane required")
    from rz_pals_model.generation_data import digest
    return {"schema": "rz-pals-native-reviewed-transition-profile/1",
        "review_id": "synthetic-byte-custody-unit-v1-" + lane, "lane": lane,
        "transition_sources": copy.deepcopy(SYNTHETIC_CUSTODY_TRANSITION_PINS),
        "component_pins": copy.deepcopy(SYNTHETIC_CUSTODY_COMPONENT_PINS),
        "compiled_search_implementation_sha256": SYNTHETIC_CUSTODY_COMPILED_FACT,
        "closure_sha256": digest("rz-pals-native-reviewed-transition-closure/1", SYNTHETIC_CUSTODY_COMPONENT_PINS)}


SYNTHETIC_CUSTODY_CLOSURE_SHA256 = synthetic_custody_fixture_profile("default_disabled")["closure_sha256"]


@contextmanager
def synthetic_custody_fixture_review_scope(lane):
    """Scoped unit-only registry; never a persistent reviewed-source enrollment."""
    profile = synthetic_custody_fixture_profile(lane)
    with patch.object(reviewed, "REVIEWED_CLOSURE_PROFILES", (profile,)):
        yield profile


# A distinct test-local idle-archive successor at the explicit product source
# commit below. All five prior banks above are retained byte-for-byte. The
# twenty-file scope retains the prior fourteen and explicitly binds six
# idle/archive dependencies; archive_lifecycle.rs was already in the fourteen.
# Its compiled fact is a synthetic label, never an actual binary, archive I/O,
# native retirement, selected-view consumption, Rules or training authority.
IDLE_ARCHIVE_REVIEWED_SOURCE_COMMIT = 'afceff25bd3168fe2891f9cb37bb791875f6bf88'
IDLE_ARCHIVE_REVIEWED_COMPONENT_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 423651,
        "sha256": "a5025b774b090f81a0ed8a49d0d2bee5f4981089363041abe0cd2476b525cc31"
    },
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 234273,
        "sha256": "83d54753084c62ecc14dc1f6c7b33e7f93f0d1cd4e2a5e5ce5b27a8a86746cbb"
    },
    "crates/rz-search/src/pals/engine/semantic_state.rs": {
        "bytes": 34003,
        "sha256": "aecd3584f37b4e4ea1105d6f60a977719e040cad0adfa832801af1089c22812c"
    },
    "crates/rz-search/src/pals/engine/researcher_admission.rs": {
        "bytes": 5957,
        "sha256": "5a1838c122e0e7eddd13b501f73cc94d9ea668333c70c1047111e9c137d3fdcd"
    },
    "crates/rz-search/src/pals/engine/researcher_profile.rs": {
        "bytes": 5443,
        "sha256": "118239c55485dcc88f4d82e29f60e68f47d3c4ba1e95b717368f41fab99ee810"
    },
    "crates/rz-search/src/pals/engine/review_lifecycle.rs": {
        "bytes": 37196,
        "sha256": "b669925a221980da56a91b691468243bb4c72a72141df5a4cf9a55c21eebff44"
    },
    "crates/rz-search/src/pals/engine/cpu_lifecycle.rs": {
        "bytes": 41911,
        "sha256": "2ad824a55e1f9d4b4215bf3f56663827955ac4f6efec37e5a695f0c54866fe0f"
    },
    "crates/rz-search/src/pals/engine/followup_execution.rs": {
        "bytes": 42122,
        "sha256": "72e85351261631f7bf57e95a53f95d8c8a6a079e3a320f27efcbefec2e1f938a"
    },
    "crates/rz-search/src/pals/engine/archive_lifecycle.rs": {
        "bytes": 174025,
        "sha256": "c56b00f70900365b7ada4156cd77e325f4ea5ae6c01769e8b8b55c20d7e68332"
    },
    "crates/rz-arena/src/pals_collect.rs": {
        "bytes": 182486,
        "sha256": "608ed3c18d4a25e03e686c060a3c332a85443f8370406bfd9ed8f5e963b9e2e6"
    },
    "crates/rz-arena/src/pals_collect/full_line.rs": {
        "bytes": 18672,
        "sha256": "6f79dd8e05add10c258ebe873fd3db2642e39727e616ebf4e256d0c7f3b4ce30"
    },
    "crates/rz-arena/src/pals_collect/divergence.rs": {
        "bytes": 4980,
        "sha256": "f2916c4128767ffc0fe06407bc39024e2b6c2ec2e1eb952faeef998d7d7b1efd"
    },
    "crates/rz-arena/src/pals_collect/continuation.rs": {
        "bytes": 29927,
        "sha256": "22ce51044c1a4d3b61b3e62f9eae23198a004611e30ef14b20825ea5206f3963"
    },
    "crates/rz-arena/src/pals_collect/producer.rs": {
        "bytes": 21824,
        "sha256": "d72fad02feee033ce223f8d8608bd4847e6adc954232b82dd0c3abfd2065a304"
    },
    "crates/rz-search/src/pals/engine/idle_archive.rs": {
        "bytes": 61470,
        "sha256": "d7fc45d196adf239209a74af6eb2d554cc210fe24a7f2f9a1a6fedcc7140fb08"
    },
    "crates/rz-search/src/pals/engine/adaptive_port/archive_worker.rs": {
        "bytes": 121261,
        "sha256": "d5da6826be44d9e08994150488250ad328b1a0eb28e0713a9b7b4b3861455e71"
    },
    "crates/rz-search/src/pals/engine/preparing_root.rs": {
        "bytes": 13206,
        "sha256": "98d34d3ffd9f7f823215422d252c8b7b48f790161f24b182d606302e2ef41e31"
    },
    "crates/rz-search/src/pals/review_context.rs": {
        "bytes": 82880,
        "sha256": "482e1851bf4d135886c7383929fe087fbf9aadde717374440d409489fb92898f"
    },
    "crates/rz-search/src/pals/engine/adaptive_root.rs": {
        "bytes": 29813,
        "sha256": "6ccfeda0e113f42167c7c694cf2324bf4557adb3d98c233d2a83bcc22a59c8c8"
    },
    "crates/rz-search/src/pals/store/archive.rs": {
        "bytes": 179458,
        "sha256": "3ad811186f8783a8ec9ac5468471009c7ac026478c0cb8f0862efe9e61057b1a"
    }
}
IDLE_ARCHIVE_REVIEWED_TRANSITION_PINS = {
    path: copy.deepcopy(IDLE_ARCHIVE_REVIEWED_COMPONENT_PINS[path]) for path in (
        "crates/rz-search/src/pals/engine.rs", "crates/rz-arena/src/pals_collect/native.rs")}
IDLE_ARCHIVE_REVIEWED_CLOSURE_SHA256 = '72e34e8533ae82330acd9f62ab9628a70ce3b7227df573b26fe162aa2ef7f4dd'
# SHA-256 of b"synthetic independently declared compiled closure fact for reviewed engine idle source afceff2\n"; no actual compiled proof.
IDLE_ARCHIVE_SYNTHETIC_COMPILED_FACT = '431ab7c9ab24953666ab4424d73f07481b7429475ed3fdf9834d017d307cc018'


def idle_archive_fixture_profile(lane):
    if lane not in ("default_disabled", "legacy_single_reply"):
        raise ValueError("explicit synthetic idle-archive review lane required")
    return {"schema": "rz-pals-native-reviewed-transition-profile/1",
        "review_id": "synthetic-reviewed-afceff2-engine-idle-twenty-source-" + lane,
        "lane": lane, "transition_sources": copy.deepcopy(IDLE_ARCHIVE_REVIEWED_TRANSITION_PINS),
        "component_pins": copy.deepcopy(IDLE_ARCHIVE_REVIEWED_COMPONENT_PINS),
        "compiled_search_implementation_sha256": IDLE_ARCHIVE_SYNTHETIC_COMPILED_FACT,
        "closure_sha256": IDLE_ARCHIVE_REVIEWED_CLOSURE_SHA256}


@contextmanager
def idle_archive_fixture_review_scope(lane):
    """Only this synthetic review is temporary; production enrollment is empty."""
    profile = idle_archive_fixture_profile(lane)
    with patch.object(reviewed, "REVIEWED_CLOSURE_PROFILES", (profile,)):
        yield profile


def read_idle_archive_fixture_sources(repo):
    """Match exact reviewed LF source bytes; never normalize or enroll files."""
    sources = {}
    for path, expected in IDLE_ARCHIVE_REVIEWED_COMPONENT_PINS.items():
        with (repo / path).open("rb") as stream:
            actual = stream.read(expected["bytes"] + 1)
        if {"bytes": len(actual), "sha256": hashlib.sha256(actual).hexdigest()} != expected:
            raise AssertionError("reviewed idle-archive Rust fixture source pin changed: " + path)
        sources[path] = actual
    return sources


def idle_archive_fixture_manifest():
    return {"source_commit": IDLE_ARCHIVE_REVIEWED_SOURCE_COMMIT,
        "files": [{"path": path, **copy.deepcopy(value)}
                  for path, value in IDLE_ARCHIVE_REVIEWED_COMPONENT_PINS.items()]}


class IdleArchiveFixtureTests(_fixture_unittest.TestCase):
    def test_idle_archive_bank_preserves_five_prior_banks_and_requires_its_own_fact(self):
        import native_source_review_fixture as banks
        from rz_pals_model import native_source_review as reviewed
        factories = (banks.fixture_profile, banks.successor_fixture_profile,
                     banks.pc_shared_fixture_profile, banks.bundle_capacity_fixture_profile,
                     banks.feedback_observation_fixture_profile)
        prior = [factory("default_disabled") for factory in factories]
        old = copy.deepcopy(prior)
        expected_extra = {
            "crates/rz-search/src/pals/engine/idle_archive.rs",
            "crates/rz-search/src/pals/engine/adaptive_port/archive_worker.rs",
            "crates/rz-search/src/pals/engine/preparing_root.rs",
            "crates/rz-search/src/pals/review_context.rs",
            "crates/rz-search/src/pals/engine/adaptive_root.rs",
            "crates/rz-search/src/pals/store/archive.rs"}
        current = banks.idle_archive_fixture_profile("default_disabled")
        self.assertEqual(banks.idle_archive_fixture_manifest()["source_commit"], 'afceff25bd3168fe2891f9cb37bb791875f6bf88')
        self.assertEqual(set(current["component_pins"]),
                         set(prior[-1]["component_pins"]) | expected_extra)
        self.assertEqual(len(current["component_pins"]), 20)
        self.assertEqual([path for path in prior[-1]["component_pins"]
            if prior[-1]["component_pins"][path] != current["component_pins"][path]],
            ["crates/rz-search/src/pals/engine.rs",
             "crates/rz-search/src/pals/engine/review_lifecycle.rs",
             "crates/rz-search/src/pals/engine/archive_lifecycle.rs"])
        self.assertEqual(len({profile["review_id"] for profile in [*prior, current]}), 6)
        before = copy.deepcopy(reviewed.REVIEWED_CLOSURE_PROFILES)
        for lane in ("default_disabled", "legacy_single_reply"):
            with banks.idle_archive_fixture_review_scope(lane) as profile:
                manifest = banks.idle_archive_fixture_manifest()
                proof = reviewed.review_transition_sources(pair=profile["transition_sources"], manifest=manifest,
                    source={"native": {"search_implementation_sha256": banks.IDLE_ARCHIVE_SYNTHETIC_COMPILED_FACT}},
                    lane=lane)
                self.assertEqual(proof["closure_sha256"], banks.IDLE_ARCHIVE_REVIEWED_CLOSURE_SHA256)
                self.assertIs(proof["historical_metadata_only"], True)
                for name in ("execution_authorized", "original_native_authority_reissued",
                             "rules_proof_authorized", "training_target_authorized"):
                    self.assertIs(proof[name], False)
                for historical in prior:
                    with self.assertRaisesRegex(ValueError, "compiled search closure differs"):
                        reviewed.review_transition_sources(pair=profile["transition_sources"], manifest=manifest,
                            source={"native": {"search_implementation_sha256": historical["compiled_search_implementation_sha256"]}},
                            lane=lane)
            self.assertEqual(reviewed.REVIEWED_CLOSURE_PROFILES, before)
        self.assertEqual([factory("default_disabled") for factory in factories], old)

    def test_idle_archive_actual_dependency_pins_cannot_be_omitted_or_substituted(self):
        import native_source_review_fixture as banks
        from rz_pals_model import native_source_review as reviewed
        dependencies = (
            "crates/rz-search/src/pals/engine/idle_archive.rs",
            "crates/rz-search/src/pals/engine/archive_lifecycle.rs",
            "crates/rz-search/src/pals/engine/adaptive_port/archive_worker.rs",
            "crates/rz-search/src/pals/engine/preparing_root.rs",
            "crates/rz-search/src/pals/review_context.rs",
            "crates/rz-search/src/pals/engine/adaptive_root.rs",
            "crates/rz-search/src/pals/store/archive.rs")
        before = copy.deepcopy(reviewed.REVIEWED_CLOSURE_PROFILES)
        for lane in ("default_disabled", "legacy_single_reply"):
            with banks.idle_archive_fixture_review_scope(lane) as profile:
                source = {"native": {"search_implementation_sha256": banks.IDLE_ARCHIVE_SYNTHETIC_COMPILED_FACT}}
                for path in dependencies:
                    for omit in (False, True):
                        manifest = banks.idle_archive_fixture_manifest()
                        if omit:
                            manifest["files"] = [row for row in manifest["files"] if row["path"] != path]
                        else:
                            next(row for row in manifest["files"] if row["path"] == path)["sha256"] = "f" * 64
                        with self.subTest(lane=lane, path=path, omit=omit), self.assertRaisesRegex(
                                ValueError, "helper closure differs"):
                            reviewed.review_transition_sources(pair=profile["transition_sources"],
                                manifest=manifest, source=source, lane=lane)
                manifest = banks.idle_archive_fixture_manifest()
                outside = "crates/rz-arena/src/rovent_training_process/utility/runtime.rs"
                manifest["files"].append({"path": outside, "bytes": 1, "sha256": "e" * 64})
                proof = reviewed.review_transition_sources(pair=profile["transition_sources"],
                    manifest=manifest, source=source, lane=lane)
                self.assertEqual(proof["unreviewed_manifest_files"], [outside])
                self.assertIs(proof["execution_authorized"], False)
            self.assertEqual(reviewed.REVIEWED_CLOSURE_PROFILES, before)
