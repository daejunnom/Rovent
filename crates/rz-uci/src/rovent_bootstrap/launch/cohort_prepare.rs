//! Read-only expected cohort metadata before any Ready/root exists.
//!
//! Original descriptor validation is reused, but no native runtime, worker,
//! Loaded NNUEC source, clock authority or trial capability is constructed.
use super::*;
use crate::pals_adaptive::UtilityCohortSourcePins;
use crate::rovent_bootstrap::{configuration_digest_from_metadata, nnuec_source};
use rz_contracts::{Digest, rovent::ExecutionMode};
use rz_eval::asset::parse_sha256;
use rz_eval::pals_onnx::virtual_review::virtual_review_backend_digest_from_metadata;
use rz_experiments::{RoventBackend, RoventPurpose};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub const EXPECTED_UTILITY_COHORT_PREPARATION: &str =
    "rz-rovent-utility-cohort-source-preparation/1";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedUtilityCohortPreparation {
    pub schema: String,
    /// The original lock is an externally pinned input. This preparation file
    /// and the future cohort profile MUST NOT be put back into that lock.
    pub input_lock: ArtifactRef,
    pub native_assets: ArtifactRef,
    pub teacher_sources: Vec<crate::rovent_bootstrap::RoventTeacherSourceSelection>,
    pub maximum_wall_ms: u64,
    /// Admission ceiling for the successful immutable-source payload plan.
    /// This neither meters nor enforces actual filesystem or OS I/O bytes.
    pub maximum_reserved_payload_read_bytes: u64,
}
impl ExpectedUtilityCohortPreparation {
    pub fn validate(&self) -> Result<(), RoventLaunchError> {
        if self.schema != EXPECTED_UTILITY_COHORT_PREPARATION
            || !(1..=300_000).contains(&self.maximum_wall_ms)
            || !(1..=(18_u64 << 30)).contains(&self.maximum_reserved_payload_read_bytes)
            || self.teacher_sources.is_empty()
            || self.teacher_sources.len() > 16
            || self.input_lock.path == self.native_assets.path
        {
            return Err(error(
                "CohortPreparation",
                "explicit bounded CPU source preparation required",
            ));
        }
        for (pin, maximum) in [
            (&self.input_lock, MAX_LOCK_BYTES),
            (&self.native_assets, MAX_SELECTION_BYTES),
        ] {
            pin.validate()
                .map_err(|failure| error("CohortPreparation", failure))?;
            private_path(&pin.path)?;
            if pin.bytes > maximum {
                return Err(error(
                    "CohortPreparation",
                    "source preparation input byte bound",
                ));
            }
        }
        // Use the same original Collection source bounds before any file is
        // opened. A later original-profile comparison checks exact coverage.
        for (index, teacher) in self.teacher_sources.iter().enumerate() {
            if teacher.index != index || teacher.assets.len() > 32 {
                return Err(error(
                    "CohortPreparation",
                    "original teacher selection order or asset bound",
                ));
            }
            private_path(&teacher.working_directory_snapshot_path)?;
            for asset in &teacher.assets {
                asset
                    .artifact
                    .validate()
                    .map_err(|failure| error("CohortPreparation", failure))?;
                private_path(&asset.artifact.path)?;
            }
        }
        let minimum_read = self
            .input_lock
            .bytes
            .checked_add(self.native_assets.bytes)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| {
                error(
                    "CohortPreparation",
                    "initial source read reservation overflow",
                )
            })?;
        if minimum_read > self.maximum_reserved_payload_read_bytes {
            return Err(error(
                "CohortPreparation",
                "initial declared payload reservation exceeds its ceiling",
            ));
        }
        Ok(())
    }
}

/// Verify original metadata and predict source identities without loading an
/// executable library or issuing a native source/Ready/root/trial authority.
/// A returned prediction is still checked against `from_actual` at each trial.
pub fn prepare_expected_utility_cohort_source_pins(
    request: &ExpectedUtilityCohortPreparation,
    cancellation: Arc<AtomicBool>,
) -> Result<Value, RoventLaunchError> {
    request.validate()?;
    let started = Instant::now();
    let until = started
        .checked_add(Duration::from_millis(request.maximum_wall_ms))
        .ok_or_else(|| error("CohortPreparation", "source preparation deadline overflow"))?;
    let control = RoventStartupControl {
        until,
        cleanup_until: until,
        cancellation,
    };
    control
        .check()
        .map_err(|failure| error("CohortPreparation", failure))?;
    let cwd = std::env::current_dir().map_err(|failure| error("Cwd", failure))?;
    let root = Dir::open_ambient_dir(&cwd, cap_std::ambient_authority())
        .map_err(|failure| error("Cwd", failure))?;
    let lock_file = request
        .input_lock
        .open_verified_at(&root, MAX_LOCK_BYTES)
        .map_err(|failure| error("InputLock", failure))?;
    let raw_lock = read_bounded(&lock_file, MAX_LOCK_BYTES)?;
    let lock = Arc::new(
        RoventInputLock::from_json(
            std::str::from_utf8(&raw_lock).map_err(|_| error("InputLock", "source lock UTF8"))?,
        )
        .map_err(|failure| error("InputLock", failure))?,
    );
    control
        .check()
        .map_err(|failure| error("CohortPreparation", failure))?;
    let pc = &lock.manifest.engine.pc;
    if lock.manifest.purpose != RoventPurpose::Collection
        || lock.manifest.engine.mode != ExecutionMode::CpuPcr
        || lock.manifest.engine.nnuec.is_none()
        || !matches!(
            lock.manifest.engine.researcher.as_ref(),
            Some(rz_experiments::RoventResearcher::Own { .. })
        )
        || pc.backend != RoventBackend::OrtCpu
        || pc.device != RoventDevice::Cpu
        || pc.runtime_artifacts.len() != 1
        || lock.manifest.wall_time_max_ms < request.maximum_wall_ms
        || !lock.manifest.provenance.contains(&request.native_assets)
    {
        return Err(error(
            "CohortPreparation",
            "original CPU P-C-R/NNUEC Collection lock and finite budget required",
        ));
    }
    let selection_file = request
        .native_assets
        .open_verified_at(&root, MAX_SELECTION_BYTES)
        .map_err(|failure| error("Assets", failure))?;
    let selection = SelectionWire::parse(&read_bounded(&selection_file, MAX_SELECTION_BYTES)?)?;
    if selection.cuda_control_inventory.is_some() || selection.runtime_bundle_manifest.is_some() {
        return Err(error(
            "CohortPreparation",
            "CPU preparation cannot inherit CUDA inventory or bundle",
        ));
    }
    // The exact selection is needed for the per-call payload plan. Admit that
    // plan before opening any large binary, graph, checkpoint or runtime file.
    let read_reservation = source_read_reservation(&lock, request, &selection)?;
    // This is a real current mapped-binary/source comparison, not runtime or
    // model execution. It cannot turn a metadata result into a Ready event.
    let binary = lock
        .manifest
        .engine
        .binary
        .open_verified_at(&root, 128 << 20)
        .map_err(|failure| error("Binary", failure))?;
    let boundary = verify_actual_process_boundary(
        &lock,
        &lock.manifest.engine.binary,
        &binary,
        &cwd,
        &root,
        &control,
    )
    .map_err(|failure| error("ProcessBoundary", failure))?;
    let export_pin = selection.export_manifest.clone();
    let preload = Arc::new(
        RoventCollectionPreloadSources::open(
            Arc::clone(&lock),
            root.try_clone().map_err(|failure| error("Cwd", failure))?,
            &cwd,
            request.teacher_sources.clone(),
            &control,
        )
        .map_err(|failure| error("CollectionSources", failure))?,
    );
    let sources = RoventPinnedNativeSources::open_collection(
        Arc::clone(&lock),
        root.try_clone().map_err(|failure| error("Cwd", failure))?,
        cwd.clone(),
        selection,
        &control,
        preload,
    )
    .map_err(|failure| error("Assets", failure))?;
    let configuration = sources.configuration();
    let native = configuration.fields().native;
    if native.loading_profile.is_some() {
        return Err(error(
            "CohortPreparation",
            "CPU source preparation cannot inherit a loading profile",
        ));
    }
    let onnx = native
        .onnx(&RoventDevice::Cpu, false)
        .map_err(|failure| error("Configuration", failure))?;
    let runtime = parse_sha256(&pc.runtime_artifacts[0].sha256)
        .map_err(|failure| error("Runtime", failure))?;
    let config_digest = configuration_digest_from_metadata(onnx, None, runtime, None, None)
        .map_err(|failure| error("Configuration", failure))?;
    if parse_sha256(&pc.backend_configuration_sha256)
        .map_err(|failure| error("Configuration", failure))?
        != config_digest
    {
        return Err(error(
            "Configuration",
            "expected native configuration differs from original lock",
        ));
    }
    let backend =
        virtual_review_backend_digest_from_metadata(runtime, None, None, onnx, None, false);
    let registration = nnuec_source::verify_metadata_registration(&sources, &control)
        .map_err(|failure| error("NnuecSource", failure))?;
    let expected = UtilityCohortSourcePins::from_registered_metadata(
        &lock,
        &registration,
        &export_pin,
        Digest(backend),
    )
    .map_err(|failure| error("CohortPreparation", failure))?;
    sources
        .validate_lock()
        .map_err(|failure| error("Assets", failure))?;
    control
        .check()
        .map_err(|failure| error("CohortPreparation", failure))?;
    let result = json!({"schema":"rz-rovent-expected-utility-cohort-source-pins/1",
        "expected_source_pins":expected,"expected_backend_sha256":super::super::hex(backend),
        "expected_native_configuration_sha256":super::super::hex(config_digest),
        "nnuec_registration":registration,"input_lock":request.input_lock,"native_assets":request.native_assets,
        "maximum_wall_ms":request.maximum_wall_ms,
        "maximum_reserved_payload_read_bytes":request.maximum_reserved_payload_read_bytes,
        "expected_payload_read_bytes":read_reservation.expected_payload_bytes,
        "expected_payload_read_plan":read_reservation.passes,
        "read_reservation_scope":"declared-immutable-successful-path-payload-only/1",
        "actual_io_hard_limit_enforced":false,"observed_read_bytes":null,
        "failure_path_io_bytes":"unknown","os_readahead_and_cache_io_bytes":"unknown",
        "metadata_only":true,"source_binary_checked":true,
        "runtime_loaded":false,"provider_ready_observed":false,"nnuec_forward_executed":false,
        "native_ready_issued":false,"root_authority_issued":false,"trial_admission_issued":false,
        "cache_ack_issued":false,"physical_completion_observed":false,"training_executed":false,
        "profile_lock_back_reference_permitted":false});
    drop(boundary);
    Ok(result)
}

#[derive(Serialize)]
struct RegisteredPayloadReadPass {
    stage: &'static str,
    artifact: ArtifactRef,
    passes: u64,
    declared_payload_bytes: u64,
}

#[derive(Default)]
struct RegisteredPayloadReadReservation {
    // Used for conflict rejection only. Accounting never deduplicates paths:
    // each caller occurrence below appends its own pass and charges its bytes.
    declarations_by_path: BTreeMap<String, ArtifactRef>,
    expected_payload_bytes: u64,
    passes: Vec<RegisteredPayloadReadPass>,
}
impl RegisteredPayloadReadReservation {
    fn add(
        &mut self,
        stage: &'static str,
        artifact: &ArtifactRef,
        passes: u64,
    ) -> Result<(), RoventLaunchError> {
        artifact
            .validate()
            .map_err(|failure| error("CohortPreparation", failure))?;
        if self
            .declarations_by_path
            .get(&artifact.path)
            .is_some_and(|original| original != artifact)
        {
            return Err(error(
                "CohortPreparation",
                "conflicting original source file pin",
            ));
        }
        let bytes = artifact
            .bytes
            .checked_mul(passes)
            .filter(|_| passes > 0)
            .ok_or_else(|| error("CohortPreparation", "payload read reservation overflow"))?;
        let total = self
            .expected_payload_bytes
            .checked_add(bytes)
            .ok_or_else(|| error("CohortPreparation", "payload read reservation overflow"))?;
        self.declarations_by_path
            .insert(artifact.path.clone(), artifact.clone());
        self.passes.push(RegisteredPayloadReadPass {
            stage,
            artifact: artifact.clone(),
            passes,
            declared_payload_bytes: bytes,
        });
        self.expected_payload_bytes = total;
        Ok(())
    }
}

/// Exact declared payload plan for this helper's immutable, successful path.
/// It is not an actual-I/O meter or an upper bound on failure-path I/O, kernel
/// read-ahead, page faults or physical disk traffic. Rejected mutable files may
/// consume different bytes; those observations remain unknown. The original
/// readers retain their own per-file size bounds and deadline checks.
fn source_read_reservation(
    lock: &RoventInputLock,
    request: &ExpectedUtilityCohortPreparation,
    selection: &RoventNativeAssetSelection,
) -> Result<RegisteredPayloadReadReservation, RoventLaunchError> {
    let mut reads = RegisteredPayloadReadReservation::default();
    // ArtifactRef::open_verified_at hashes one file payload; read_bounded then
    // reads it once. These are the only reads before this plan is admitted.
    reads.add(
        "input_lock.open_verified_at+read_bounded",
        &request.input_lock,
        2,
    )?;
    reads.add(
        "native_assets.open_verified_at+read_bounded",
        &request.native_assets,
        2,
    )?;
    // Original binary open, verify_binary_descriptor, and /proc/self/exe
    // observation each read the same declared payload on the successful path.
    reads.add("binary.open_verified_at", &lock.manifest.engine.binary, 1)?;
    reads.add(
        "binary.verify_binary_descriptor",
        &lock.manifest.engine.binary,
        1,
    )?;
    reads.add(
        "binary.mapped_executable_observation",
        &lock.manifest.engine.binary,
        1,
    )?;
    // RoventCollectionPreloadSources::open performs one profile open/hash and
    // one read_blob; binary and each separately selected asset are opened once.
    // Iterate occurrences, including repeated artifacts in separate slots.
    for teacher in &lock.manifest.engine.teachers {
        reads.add("teacher.profile.open_blob+read_blob", &teacher.profile, 2)?;
        reads.add("teacher.binary.open_blob", &teacher.binary, 1)?;
    }
    for teacher in &request.teacher_sources {
        for asset in &teacher.assets {
            reads.add("teacher.asset.open_blob", &asset.artifact, 1)?;
        }
    }
    // open_selected opens each selected artifact exactly once. Configuration
    // has an initial read_blob, then validate_lock reads configuration/export/
    // rules once at the end of open_selected and once at the end of this helper.
    reads.add(
        "native.configuration.open+initial_read+two_validations",
        &selection.startup_configuration,
        4,
    )?;
    reads.add(
        "native.export.open+two_validations",
        &selection.export_manifest,
        3,
    )?;
    reads.add(
        "native.rules.open+two_validations",
        &selection.rules_descriptor,
        3,
    )?;
    for graph in &selection.graphs {
        reads.add("native.graph.open_blob", &graph.artifact, 1)?;
    }
    // Shared P/C startup opens the single proposer checkpoint/provenance. The
    // original topology rejects a different critic; it does not open it again.
    reads.add(
        "native.pc_checkpoint.open_blob",
        &lock.manifest.engine.pc.proposer.weights,
        1,
    )?;
    reads.add(
        "native.pc_provenance.open_blob",
        &lock.manifest.engine.pc.proposer.provenance,
        1,
    )?;
    for runtime in &lock.manifest.engine.pc.runtime_artifacts {
        reads.add("native.runtime.open_blob", runtime, 1)?;
    }
    // PinnedNnuecArtifacts::open hashes each artifact once; the source metadata
    // verifier calls its read() once. Further SHA/registration work is over RAM.
    if let Some(nnuec) = &lock.manifest.engine.nnuec {
        reads.add(
            "native.nnuec_weights.open_blob+metadata_read",
            &nnuec.weights,
            2,
        )?;
        reads.add(
            "native.nnuec_provenance.open_blob+metadata_read",
            &nnuec.provenance,
            2,
        )?;
    }
    if reads.expected_payload_bytes > request.maximum_reserved_payload_read_bytes {
        return Err(error(
            "CohortPreparation",
            "declared payload reservation exceeds its explicit ceiling",
        ));
    }
    Ok(reads)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request() -> ExpectedUtilityCohortPreparation {
        ExpectedUtilityCohortPreparation {
            schema: EXPECTED_UTILITY_COHORT_PREPARATION.into(),
            input_lock: ArtifactRef {
                path: "inputs/lock.json".into(),
                sha256: "a".repeat(64),
                bytes: 16,
                source: "https://example.org/metadata-only-fixture".into(),
                license: "MIT".into(),
            },
            native_assets: ArtifactRef {
                path: "inputs/assets.json".into(),
                sha256: "b".repeat(64),
                bytes: 16,
                source: "https://example.org/metadata-only-fixture".into(),
                license: "MIT".into(),
            },
            teacher_sources: vec![crate::rovent_bootstrap::RoventTeacherSourceSelection {
                index: 0,
                working_directory_snapshot_path: "inputs/teacher".into(),
                assets: vec![],
            }],
            maximum_wall_ms: 1_000,
            maximum_reserved_payload_read_bytes: 1 << 20,
        }
    }
    #[test]
    fn preparation_requires_finite_explicit_inputs_without_opening_them() {
        let good = request();
        assert!(good.validate().is_ok());
        let mut bad = good.clone();
        bad.maximum_wall_ms = 0;
        assert!(bad.validate().is_err());
        let mut bad = good.clone();
        bad.native_assets = bad.input_lock.clone();
        assert!(bad.validate().is_err());
        let mut bad = good.clone();
        bad.input_lock.path = "../lock.json".into();
        assert!(bad.validate().is_err());
        let mut bad = good.clone();
        bad.teacher_sources.clear();
        assert!(bad.validate().is_err());
        let mut bad = good.clone();
        bad.teacher_sources[0].index = 1;
        assert!(bad.validate().is_err());
        let mut bad = good.clone();
        bad.teacher_sources[0].working_directory_snapshot_path = "../teacher".into();
        assert!(bad.validate().is_err());
        let mut bad = good;
        bad.maximum_reserved_payload_read_bytes = 1;
        assert!(bad.validate().is_err());
    }
    #[test]
    fn metadata_preparation_cancellation_precedes_file_access() {
        let result = prepare_expected_utility_cohort_source_pins(
            &request(),
            Arc::new(AtomicBool::new(true)),
        );
        assert!(result.is_err());
    }
    #[test]
    fn payload_reservation_charges_repeated_references_without_deduplication() {
        let pin = request().input_lock;
        let mut reads = RegisteredPayloadReadReservation::default();
        reads.add("teacher.slot0", &pin, 2).unwrap();
        reads.add("teacher.slot1", &pin, 2).unwrap();
        reads.add("native.checkpoint", &pin, 1).unwrap();
        assert_eq!(reads.expected_payload_bytes, 5 * pin.bytes);
        assert_eq!(reads.passes.len(), 3);
        assert_eq!(reads.declarations_by_path.len(), 1);
        assert_eq!(reads.passes[0].passes, 2);
        assert_eq!(reads.passes[2].declared_payload_bytes, pin.bytes);
        let mut conflicting = pin;
        conflicting.sha256 = "c".repeat(64);
        assert!(reads.add("conflicting", &conflicting, 1).is_err());
        assert_eq!(reads.expected_payload_bytes, 5 * conflicting.bytes);
        assert_eq!(reads.passes.len(), 3);
    }
    #[test]
    fn payload_reservation_rejects_zero_passes_and_arithmetic_overflow() {
        let mut pin = request().input_lock;
        let mut reads = RegisteredPayloadReadReservation::default();
        assert!(reads.add("empty", &pin, 0).is_err());
        pin.bytes = u64::MAX;
        assert!(reads.add("overflow", &pin, 2).is_err());
        assert_eq!(reads.expected_payload_bytes, 0);
        assert!(reads.passes.is_empty());
    }
}
