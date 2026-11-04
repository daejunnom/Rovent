//! Compile-time observations of the /3 arena caller's explicitly listed files.
//!
//! The file roster is intentionally narrower than a proven transitive source
//! closure. This record does not amend the child /3 lock or the V5 inventory,
//! and does not attest invocation, a running binary, inference, teacher
//! retirement, GPU completion, publication or training eligibility.
use super::pc_lock_v3::PreparedPcCollectionInputsV3;
use super::{ArenaError, RoventPublicationFile, check, invalid};
use cap_std::fs::Dir;
use rz_experiments::{RoventInputLockV3, pals_data_v3::generation_digest};
use serde::Serialize;
use sha2::{Digest as _, Sha256};
use std::collections::BTreeMap;

pub(crate) const CALLER_SOURCE_V3_DOMAIN: &str = "rz-rovent-collection-caller-source/1";
pub(crate) const CALLER_RECEIPT_V3_DOMAIN: &str = "rz-rovent-collection-caller-source-receipt/1";

// All paths are logical repository paths and all byte slices are embedded by
// this crate at compile time. Update this explicitly when the actual product
// path changes; this list does not claim automatic dependency discovery.
macro_rules! arena_source {
    ($relative:literal) => {
        (
            concat!("crates/rz-arena/", $relative),
            include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/", $relative)).as_slice(),
        )
    };
}

const ARENA_CALLER_FILES: &[(&str, &[u8])] = &[
    arena_source!("Cargo.toml"),
    arena_source!("src/lib.rs"),
    arena_source!("src/bin/rovent-collect.rs"),
    arena_source!("src/pals_collect_v3.rs"),
    arena_source!("src/pals_collect_v3/capture.rs"),
    arena_source!("src/rovent_collect.rs"),
    arena_source!("src/rovent_collect/launch.rs"),
    arena_source!("src/rovent_collect/launch/pc_lock_v3.rs"),
    arena_source!("src/rovent_collect/launch/pc_recovery_v3.rs"),
    arena_source!("src/rovent_collect/launch/pc_cli_managed_v3.rs"),
    arena_source!("src/rovent_collect/launch/pc_managed_prepare_v3.rs"),
    arena_source!("src/rovent_collect/launch/pc_managed_entry_v3.rs"),
    arena_source!("src/rovent_collect/launch/long_game_profile_v3.rs"),
    arena_source!("src/rovent_collect/long_game_profile_v3.rs"),
    arena_source!("src/rovent_collect/launch/managed_output_fd_v2.rs"),
    arena_source!("src/rovent_collect/launch/caller_source_v3.rs"),
    arena_source!("src/rovent_collect/native_capture.rs"),
    arena_source!("src/rovent_collect/capture_router_v3.rs"),
    arena_source!("src/rovent_collect/speculative_route_v3.rs"),
    arena_source!("src/rovent_collect/speculative_route_v3/target_gate.rs"),
    arena_source!("src/rovent_collect/observer_lock_v3.rs"),
    arena_source!("src/rovent_collect/observer.rs"),
    arena_source!("src/rovent_collect/policy_row_pipeline_v3.rs"),
    arena_source!("src/rovent_collect/pc_hooks_lock_v3.rs"),
    arena_source!("src/rovent_collect/pc_worker_v3.rs"),
    arena_source!("src/rovent_collect/own_hooks_lock_v3.rs"),
    arena_source!("src/rovent_collect/own_worker_v3.rs"),
    arena_source!("src/rovent_collect/producer_v2.rs"),
    arena_source!("src/rovent_collect/producer_v3.rs"),
    arena_source!("src/rovent_collect/teacher.rs"),
    arena_source!("src/rovent_collect/teacher/collection.rs"),
    arena_source!("src/rovent_collect/teacher/v3_session.rs"),
    arena_source!("src/rovent_collect/teacher/v3_dispatch.rs"),
    arena_source!("src/rovent_collect/teacher/v3_pump.rs"),
    arena_source!("src/rovent_collect/teacher/v3_policy.rs"),
    arena_source!("src/rovent_collect/teacher/v3_policy_consumer.rs"),
    arena_source!("src/rovent_collect/teacher/policy_prepare_v3.rs"),
    arena_source!("src/rovent_collect/submitted_lock_v3.rs"),
    arena_source!("src/rovent_collect/submitted_v3.rs"),
    arena_source!("src/rovent_collect/publication.rs"),
    arena_source!("src/rovent_collect/publication_lock_v3.rs"),
    arena_source!("src/rovent_collect/writer_lock_v3.rs"),
    arena_source!("src/rovent_resource_scope.rs"),
    arena_source!("src/rovent_resource_scope/launch.rs"),
    arena_source!("src/rovent_resource_scope/direct_control.rs"),
    arena_source!("src/rovent_training_process.rs"),
    arena_source!("src/rovent_training_process/resources.rs"),
    arena_source!("src/rovent_training_process/resources/native_tree.rs"),
];

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct CallerSourceFileV3 {
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct CallerBuildObservationV3 {
    pub(crate) target_os: &'static str,
    pub(crate) target_arch: &'static str,
    pub(crate) feature_observation_scope: &'static str,
    pub(crate) pals_collection_onnx: bool,
    pub(crate) native_cuda: bool,
    pub(crate) experimental_teacher_exact_observation: bool,
    pub(crate) compiler_version: Option<&'static str>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CallerBinaryObservationV3 {
    /// No original executable-file owner was inspected by this observation.
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct CallerSourceInventoryV3 {
    pub(crate) domain: &'static str,
    pub(crate) scope: &'static str,
    pub(crate) transitive_source_closure: &'static str,
    pub(crate) missing_or_unverified: Vec<&'static str>,
    pub(crate) files: BTreeMap<&'static str, CallerSourceFileV3>,
    pub(crate) source_sha256: String,
}

/// Hash the explicit source roster with its own domain. The independent child
/// source inventory and its existing lock remain unchanged.
pub(crate) fn compiled_caller_source_inventory_v3() -> Result<CallerSourceInventoryV3, ArenaError> {
    let mut files = BTreeMap::new();
    for &(path, bytes) in ARENA_CALLER_FILES {
        check(!bytes.is_empty(), "empty V3 caller source entry")?;
        if files
            .insert(
                path,
                CallerSourceFileV3 {
                    bytes: bytes.len() as u64,
                    sha256: format!("{:x}", Sha256::digest(bytes)),
                },
            )
            .is_some()
        {
            return Err(invalid("duplicate V3 caller source path"));
        }
    }
    let source_sha256 = generation_digest(CALLER_SOURCE_V3_DOMAIN, &files)?;
    Ok(CallerSourceInventoryV3 {
        domain: CALLER_SOURCE_V3_DOMAIN,
        scope: "explicit embedded arena caller files only",
        transitive_source_closure: "unknown",
        missing_or_unverified: vec![
            "complete transitive caller dependency closure",
            "actual caller binary and linked dependency bytes",
            "child UCI, Rust toolchain and system runtime identities",
            "production hook, entry and source receipt wiring",
        ],
        files,
        source_sha256,
    })
}

/// Diagnostic source-only record. The caller must separately retain the
/// original input owner and native/teacher/physical completion evidence.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct CallerSourceReceiptV3 {
    pub(crate) schema: &'static str,
    pub(crate) evidence_scope: &'static str,
    pub(crate) input_boundary_scope: &'static str,
    pub(crate) source: CallerSourceInventoryV3,
    pub(crate) build: CallerBuildObservationV3,
    pub(crate) caller_binary: CallerBinaryObservationV3,
    pub(crate) original_plan: RoventPublicationFile,
    pub(crate) original_manifest: RoventPublicationFile,
    pub(crate) input_lock_canonical_sha256: String,
    /// A regular lock descriptor is checked; this path does not hash its raw
    /// bytes or independently attest its association with the typed lock.
    pub(crate) input_lock_raw_bytes: &'static str,
}

impl CallerSourceReceiptV3 {
    /// Use in the original resource admission callback, before any worker
    /// starts. Validation is source/inputs only and grants no launch authority.
    pub(crate) fn from_boundary(
        inputs: &PreparedPcCollectionInputsV3,
        lock: &RoventInputLockV3,
        directory: &Dir,
        original_lock_file: &std::fs::File,
    ) -> Result<Self, ArenaError> {
        inputs.validate_original_boundary(lock, directory, original_lock_file)?;
        Self::from_inputs_and_lock(inputs, lock)
    }

    /// Build the diagnostic after the original callback has validated this
    /// exact owner and lock. Hashing occurs only when this method is called.
    pub(super) fn from_validated_boundary(
        boundary: &super::pc_lock_v3::ValidatedCallerBoundaryV3,
    ) -> Result<Self, ArenaError> {
        Self::from_inputs_and_lock(boundary.inputs(), boundary.lock())
    }

    fn from_inputs_and_lock(
        inputs: &PreparedPcCollectionInputsV3,
        lock: &RoventInputLockV3,
    ) -> Result<Self, ArenaError> {
        Ok(Self {
            schema: CALLER_RECEIPT_V3_DOMAIN,
            evidence_scope: "embedded caller source and declared input pins; execution unknown",
            input_boundary_scope: "validated typed lock and same open directory; lock file is regular",
            source: compiled_caller_source_inventory_v3()?,
            build: CallerBuildObservationV3 {
                target_os: std::env::consts::OS,
                target_arch: std::env::consts::ARCH,
                feature_observation_scope: "only the three named crate features; full feature closure unknown",
                pals_collection_onnx: cfg!(feature = "pals-collection-onnx"),
                native_cuda: cfg!(feature = "native-cuda"),
                experimental_teacher_exact_observation: cfg!(
                    feature = "experimental-teacher-exact-observation"
                ),
                compiler_version: None,
            },
            caller_binary: CallerBinaryObservationV3::Unknown,
            original_plan: inputs.original_plan_pin().clone(),
            original_manifest: inputs.original_manifest_pin().clone(),
            input_lock_canonical_sha256: lock.canonical_sha256.clone(),
            input_lock_raw_bytes: "unknown",
        })
    }
}
