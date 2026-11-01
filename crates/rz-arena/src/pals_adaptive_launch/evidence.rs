//! Parent-owned input observations and producer/process evidence joins.
use super::*;
use std::fmt;

#[derive(Clone, Debug, Serialize)]
pub struct PalsAdaptiveAdmissionV5 {
    pub storage: PalsOwnedStorageAuditV5,
    pub source_verification: PalsSourceVerificationV5,
    pub numerical_by_endpoint: BTreeMap<String, PalsNumericalAdmissionV5>,
    pub artifact_domain_by_endpoint: BTreeMap<String, PalsArtifactDomainAdmissionV5>,
}
#[derive(Clone, Debug)]
pub(super) struct PalsAdaptiveInputAdmission {
    pub source_verification: PalsSourceVerificationV5,
    pub numerical_by_endpoint: BTreeMap<String, PalsNumericalAdmissionV5>,
    pub artifact_domain_by_endpoint: BTreeMap<String, PalsArtifactDomainAdmissionV5>,
}
#[derive(Clone, Debug, Serialize)]
pub struct PalsAdaptiveSessionAuditV5 {
    pub endpoint_id: String,
    pub process_id: u32,
    pub process_run_id: String,
    pub startup_sha256: String,
    pub termination_sha256: String,
    pub endpoint: PalsEndpointReceiptV5,
    pub parent_verified_source: PalsSourceVerificationV5,
    pub parent_process_exit_observed: bool,
}
/// Both failure variants retain the private source/model pins. A pre-spawn
/// validation failure cannot be converted into an empty successful pair.
#[allow(
    clippy::large_enum_variant,
    reason = "Admission failure retains the original launch owner inline without a new allocation or changing source/model pin custody"
)]
pub enum PalsAdaptivePairFailureV5 {
    Admission {
        cause: ArenaError,
        owner: NativeLaunchOwner<LockedPalsArenaLaunchV5>,
    },
    Execution(Box<crate::NativePairFailure<LockedPalsArenaLaunchV5>>),
}
impl fmt::Debug for PalsAdaptivePairFailureV5 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission { cause, owner } => f
                .debug_struct("V5AdmissionFailure")
                .field("cause", cause)
                .field("owner", owner)
                .finish(),
            Self::Execution(e) => e.fmt(f),
        }
    }
}
impl fmt::Display for PalsAdaptivePairFailureV5 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission { cause, .. } => write!(
                f,
                "V5 pre-spawn validation failed; original owner retained: {cause}"
            ),
            Self::Execution(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for PalsAdaptivePairFailureV5 {}

#[cfg(target_os = "linux")]
fn verify_file(file: &std::fs::File, artifact: &ArtifactRef) -> Result<[u8; 32], ArenaError> {
    use std::os::unix::fs::FileExt;
    let mut h = Sha256::new();
    let mut n = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let got = file
            .read_at(&mut buffer, n)
            .map_err(|_| invalid("input pin read failed"))?;
        if got == 0 {
            break;
        }
        n = n
            .checked_add(got as u64)
            .ok_or_else(|| invalid("input byte count overflow"))?;
        require(
            n <= artifact.bytes,
            "input pin grew beyond registered budget",
        )?;
        h.update(&buffer[..got]);
    }
    let hash: [u8; 32] = h.finalize().into();
    require(
        n == artifact.bytes && hex(hash) == artifact.sha256,
        "actual input pin differs from registration",
    )?;
    Ok(hash)
}
#[cfg(target_os = "linux")]
fn read_pin(
    owner: &NativeLaunchOwner<LockedPalsArenaLaunchV5>,
    artifact: &ArtifactRef,
    limit: u64,
) -> Result<Vec<u8>, ArenaError> {
    require(
        artifact.bytes <= limit,
        "metadata input exceeds admission bound",
    )?;
    let pin = crate::native_launch::linux::pin(&owner.snapshot.pins, artifact)?;
    use std::os::unix::fs::FileExt;
    let mut out = vec![0u8; artifact.bytes as usize];
    pin.file
        .read_exact_at(&mut out, 0)
        .map_err(|_| invalid("metadata descriptor read failed"))?;
    require(
        out.len() as u64 == artifact.bytes
            && format!("{:x}", Sha256::digest(&out)) == artifact.sha256,
        "actual metadata input differs",
    )?;
    Ok(out)
}

#[cfg(target_os = "linux")]
fn validate_prepared_pals_launch(
    owner: &NativeLaunchOwner<LockedPalsArenaLaunchV5>,
) -> Result<(storage::PalsStorageAdmission, PalsAdaptiveInputAdmission), ArenaError> {
    use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::{OpenOptions, OpenOptionsExt};
    use std::io::Write;
    owner.spec.input.validate()?;
    let storage = owner.spec.storage_admission.lock()
        .map_err(|_| invalid("storage admission state poisoned"))?
        .clone().ok_or_else(|| invalid("owned storage admission missing"))?;
    require(owner.spec.storage_lease.is_some(), "original storage lease missing")?;
    match (&storage, owner.spec.storage_policy) {
        (storage::PalsStorageAdmission::Strict(a), PalsOwnedStoragePolicy::StrictV5) => require(
            a.reserved_attempt_bytes == owner.spec.input.budget.max_artifact_bytes
                && a.no_follow_byte_scan_completed && a.external_serialization_observed,
            "V5 strict storage admission differs from the attempt reservation",
        )?,
        (storage::PalsStorageAdmission::Advisory(a), PalsOwnedStoragePolicy::AdvisoryV1) => require(
            a.external_serialization_observed, "advisory storage lease missing",
        )?,
        _ => return Err(invalid("storage policy and admission domain differ")),
    }
    let lockbytes = read_pin(owner, &owner.spec.input.semantic_lock_artifact, 2 << 20)?;
    owner.spec.validate_additional_manifest(&lockbytes)?;
    let mut actual = BTreeMap::new();
    for pin in &owner.snapshot.pins {
        let sha = verify_file(&pin.file, &pin.artifact)?;
        actual.insert(pin.artifact.path.clone(), sha);
    }
    let mut components = BTreeMap::new();
    for (name, files) in &owner.spec.input.source_files {
        let mut rows = Vec::with_capacity(files.len());
        for f in files {
            rows.push((
                f.logical_source_path.clone(),
                f.artifact.bytes,
                *actual
                    .get(&f.artifact.path)
                    .ok_or_else(|| invalid("actual source pin missing"))?,
            ));
        }
        components.insert(name.clone(), hex(source_component_digest_v5(&rows)?));
    }
    require(
        components == owner.spec.input.semantic_lock.manifest.source_pins,
        "actual component source content differs",
    )?;
    let mut numerical_by_endpoint = BTreeMap::new();
    let mut artifact_domain_by_endpoint = BTreeMap::new();
    for (i, e) in owner.spec.input.endpoints.iter().enumerate() {
        let bytes = read_pin(owner, &e.startup_config, 2 << 20)?;
        require(
            bytes
                == e.configuration
                    .canonical_bytes(&owner.spec.input.semantic_lock)?,
            "actual startup bytes differ from canonical registered config",
        )?;
        let manifest = e
            .configuration
            .asset(PalsStartupAssetPurposeV5::ExportManifest)?;
        let export = read_pin(owner, &manifest.artifact, 256 << 10)?;
        let rules = e
            .configuration
            .asset(PalsStartupAssetPurposeV5::RulesDescriptor)?;
        let rules_bytes = read_pin(owner, &rules.artifact, 64 << 10)?;
        validate_export_rules_content_v5(&e.configuration, &export, &rules_bytes)?;
        let mut domain = super::artifact_domains::validate_pals_artifact_domain_v5(
            &e.configuration,
            &export,
            |artifact, limit| read_pin(owner, artifact, limit as u64),
        )?;
        for (name, expected_sha) in &domain.initializer_source_pins {
            let expected_path = format!("experiments/model-research/pals/src/rz_pals_model/{name}");
            let mut joined = false;
            for files in owner.spec.input.source_files.values() {
                for file in files {
                    if file.logical_source_path == expected_path {
                        require(
                            hex(*actual
                                .get(&file.artifact.path)
                                .ok_or_else(|| invalid("initializer source actual FD absent"))?)
                                == *expected_sha,
                            "initializer source capture differs from actual registered source bytes",
                        )?;
                        joined = true;
                    }
                }
            }
            require(
                joined,
                "initializer source file absent from exact logical source inventory",
            )?;
        }
        domain.registered_artifact_bytes_verified_by_launch_owner = true;
        let numeric = e
            .configuration
            .asset(PalsStartupAssetPurposeV5::NumericalAcceptance)?;
        let numeric_bytes = read_pin(owner, &numeric.artifact, 256 << 10)?;
        let observed = super::numerical::validate_pals_numerical_acceptance_v5(
            &e.configuration,
            &numeric_bytes,
            |artifact, limit| read_pin(owner, artifact, limit as u64),
        )?;
        numerical_by_endpoint.insert(e.configuration.endpoint.id.clone(), observed);
        domain.numerical_acceptance_verified = true;
        artifact_domain_by_endpoint.insert(e.configuration.endpoint.id.clone(), domain);
        require(
            actual.contains_key(
                &owner.spec.input.semantic_lock.manifest.engines[i]
                    .binary
                    .path,
            ),
            "endpoint binary actual hash missing",
        )?;
    }
    let source_filename = match owner.spec.storage_policy {
        PalsOwnedStoragePolicy::StrictV5 => "source-verification.v5.json",
        PalsOwnedStoragePolicy::AdvisoryV1 => "source-verification.storage-advisory-v1.json",
    };
    let mut source = PalsSourceVerificationV5 {
        components,
        binary_sha256_by_endpoint: owner
            .spec
            .input
            .semantic_lock
            .manifest
            .engines
            .iter()
            .map(|e| (e.id.clone(), e.binary.sha256.clone()))
            .collect(),
        content_verified: true,
        evidence: ArtifactRef {
            path: format!(
                "{}/{}",
                owner.snapshot.output_directory, source_filename
            ),
            sha256: "0".repeat(64),
            bytes: 0,
            source: "https://github.com/daejunnom/RoveZero".into(),
            license: "MIT".into(),
        },
    };
    let (source_domain, storage_json) = match &storage {
        storage::PalsStorageAdmission::Strict(a) => ("rz-pals-source-verification-v5/1", serde_json::to_value(a)),
        storage::PalsStorageAdmission::Advisory(a) => ("rz-pals-source-verification-storage-advisory-v1/1", serde_json::to_value(a)),
    };
    let storage_json = storage_json.map_err(|_| invalid("storage admission serialization failed"))?;
    let mut payload = serde_json::json!({"domain":source_domain,"launch_sha256":owner.spec.input_sha256(),"lock_sha256":owner.spec.input.semantic_lock.canonical_sha256,"components":source.components,"binary_sha256_by_endpoint":source.binary_sha256_by_endpoint,"content_verified":true,"inputs_verified":actual.len(),"storage":storage_json,"independent_numerical":numerical_by_endpoint,"artifact_domains":artifact_domain_by_endpoint});
    if owner.spec.storage_policy == PalsOwnedStoragePolicy::AdvisoryV1 {
        payload["base_v5_launch_sha256"] = owner.spec.sha256.clone().into();
        payload["policy_domain"] = "rz-pals-preparation-storage-advisory-v1/1".into();
    }
    let bytes = serde_json::to_vec(&payload)
        .map_err(|_| invalid("source admission evidence encode failed"))?;
    require(
        bytes.len() <= 128 << 10,
        "source admission receipt exceeds fixed metadata reservation",
    )?;
    let mut opt = OpenOptions::new();
    opt.write(true)
        .create_new(true)
        .follow(FollowSymlinks::No)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .mode(0o600);
    let mut out = owner
        .snapshot
        .directory
        .open_with(source_filename, &opt)
        .map_err(|_| invalid("source admission evidence already exists or cannot be created"))?;
    out.write_all(&bytes)
        .and_then(|()| out.sync_all())
        .map_err(|_| invalid("source admission evidence save failed"))?;
    source.evidence.sha256 = format!("{:x}", Sha256::digest(&bytes));
    source.evidence.bytes = bytes.len() as u64;
    source.evidence.validate()?;
    let result = PalsAdaptiveInputAdmission {
        source_verification: source,
        numerical_by_endpoint,
        artifact_domain_by_endpoint,
    };
    *owner.spec.admission.lock().map_err(|_| invalid("admission state poisoned"))? = Some(result.clone());
    Ok((storage, result))
}

#[cfg(target_os = "linux")]
pub fn validate_prepared_pals_launch_v5(
    owner: &NativeLaunchOwner<LockedPalsArenaLaunchV5>,
) -> Result<PalsAdaptiveAdmissionV5, ArenaError> {
    require(owner.spec.storage_policy == PalsOwnedStoragePolicy::StrictV5, "V5 admission requires strict storage")?;
    let (storage, input) = validate_prepared_pals_launch(owner)?;
    let storage::PalsStorageAdmission::Strict(storage) = storage else {
        return Err(invalid("V5 strict storage admission missing"));
    };
    Ok(PalsAdaptiveAdmissionV5 { storage, source_verification: input.source_verification,
        numerical_by_endpoint: input.numerical_by_endpoint,
        artifact_domain_by_endpoint: input.artifact_domain_by_endpoint })
}

#[cfg(target_os = "linux")]
pub(super) fn validate_prepared_pals_launch_advisory(
    owner: &NativeLaunchOwner<LockedPalsArenaLaunchV5>,
) -> Result<(), ArenaError> {
    require(owner.spec.storage_policy == PalsOwnedStoragePolicy::AdvisoryV1, "advisory storage policy missing")?;
    let (storage, _input) = validate_prepared_pals_launch(owner)?;
    let storage::PalsStorageAdmission::Advisory(audit) = storage else {
        return Err(invalid("advisory storage admission missing"));
    };
    // The mandatory source-verification artifact already records the policy,
    // audit and execution identity. Do not create a second optional sidecar.
    let _ = audit;
    Ok(())
}

#[cfg(not(target_os = "linux"))]
pub(super) fn validate_prepared_pals_launch_advisory(
    _owner: &NativeLaunchOwner<LockedPalsArenaLaunchV5>,
) -> Result<(), ArenaError> {
    Err(invalid("advisory input FD validation requires Linux"))
}

#[cfg(not(target_os = "linux"))]
pub fn validate_prepared_pals_launch_v5(
    _owner: &NativeLaunchOwner<LockedPalsArenaLaunchV5>,
) -> Result<PalsAdaptiveAdmissionV5, ArenaError> {
    Err(invalid("V5 input FD validation requires Linux"))
}

#[cfg(target_os = "linux")]
fn decode_records(
    lock: &LockedPalsArenaLaunchV5,
    role: NativeEngineRole,
    startup: &[u8],
    termination: &[u8],
    session: &str,
) -> Result<PalsAdaptiveSessionAuditV5, ArenaError> {
    require(
        startup.len() <= 256 << 10 && termination.len() <= 256 << 10,
        "V5 provider metadata exceeds bounded reservation",
    )?;
    let s: PalsAdaptiveStartupRecordV5 = crate::decode_json(
        std::str::from_utf8(startup).map_err(|_| invalid("startup record not UTF8"))?,
    )?;
    let t: PalsAdaptiveTerminationRecordV5 = crate::decode_json(
        std::str::from_utf8(termination).map_err(|_| invalid("termination record not UTF8"))?,
    )?;
    let declared = &lock.input.semantic_lock.manifest.engines[role_index(role)];
    let cfg = &lock.endpoint(role).startup_config;
    require(
        s.schema_version == 5
            && t.schema_version == 5
            && s.domain == PALS_NATIVE_STARTUP_V5_DOMAIN
            && t.domain == PALS_NATIVE_TERMINATION_V5_DOMAIN
            && s.lock_sha256 == lock.input.semantic_lock.canonical_sha256
            && t.lock_sha256 == s.lock_sha256
            && s.startup_config_sha256 == cfg.sha256
            && t.startup_config_sha256 == s.startup_config_sha256
            && s.endpoint_id == declared.id
            && t.endpoint_id == s.endpoint_id
            && s.process_id > 0
            && s.process_id <= i32::MAX as u32
            && t.process_id == s.process_id
            && s.process_run_id == session
            && t.process_run_id == session
            && session.starts_with("native-process-")
            && session.len() <= 128
            && s.source_commit == declared.source_commit
            && s.component_source_pins == lock.input.semantic_lock.manifest.source_pins,
        "actual V5 process/lock/config/source identity differs",
    )?;
    require(
        t.admission_closed
            && t.question_authority_revoked
            && t.cpu_workers_joined
            && t.gpu_physical_drain_known
            && t.archive_closed
            && !t.training_executed,
        "V5 shutdown did not preserve admission/revoke/CPU/GPU/archive ordering or attempted training",
    )?;
    require(
        s.observation.endpoint_id == declared.id
            && t.observation.endpoint_id == declared.id
            && s.observation.runtime_owner == t.observation.runtime_owner,
        "actual endpoint/owner changed between ready and shutdown",
    )?;
    let admission = lock
        .admission
        .lock()
        .map_err(|_| invalid("admission ledger poisoned"))?
        .clone()
        .ok_or_else(|| invalid("parent actual byte verification missing"))?;
    let mut probe = PalsRunReceiptV5::pending(&lock.input.semantic_lock)?;
    probe.source_verification = PalsObservedV3::Observed {
        value: admission.source_verification.clone(),
        method: "arena-parent read-only FD content verification".into(),
    };
    probe.endpoints = vec![t.observation.clone()];
    probe.validate_against(&lock.input.semantic_lock)?;
    let PalsObservedV3::Observed { value: c, .. } = &t.observation.coordinator else {
        return Err(invalid("actual coordinator termination unavailable"));
    };
    require(
        c.owner_joined
            && c.archive_closed
            && c.cpu_outstanding_reserved == 0
            && c.tickets_queued == 0
            && c.physical_in_flight == 0
            && c.physical_completion_unknown == 0
            && c.tickets_quarantined == 0
            && c.tickets_released == c.tickets_submitted
            && matches!(
                c.physical_state,
                PalsPhysicalStateV3::Completed | PalsPhysicalStateV3::NotRequired
            ),
        "physical completion unknown, live CPU reservation or unreleased worker cannot count as normal termination",
    )?;
    let numerical = admission
        .numerical_by_endpoint
        .get(&declared.id)
        .ok_or_else(|| invalid("parent independent numerical input/output comparison absent"))?;
    let mut observation = t.observation;
    // A child cannot promote its own passed flag. This observation is the
    // independent pre-child raw reader's result over retained registered FDs.
    observation.native_numerical_check = PalsObservedV3::Observed {
        value: PalsNativeNumericalCheckV5 {
            weights_sha256: declared.model.weights.sha256.clone(),
            semantic_sha256: declared.model.semantic_sha256.clone(),
            encoding_sha256: declared.model.encoding_sha256.clone(),
            runtime_implementation_sha256: declared.runtime.implementation_sha256.clone(),
            source_commit: declared.source_commit.clone(),
            passed: true,
            logits_absolute_tolerance: "1e-4".into(),
            logits_relative_tolerance: "1e-3".into(),
            policy_wdl_absolute_tolerance: "1e-4".into(),
            evidence: numerical.acceptance.clone(),
        },
        method: "arena-parent actual registered tensor bytes and independent reference/native raw comparison".into(),
    };
    Ok(PalsAdaptiveSessionAuditV5 {
        endpoint_id: declared.id.clone(),
        process_id: s.process_id,
        process_run_id: s.process_run_id,
        startup_sha256: format!("{:x}", Sha256::digest(startup)),
        termination_sha256: format!("{:x}", Sha256::digest(termination)),
        endpoint: observation,
        parent_verified_source: admission.source_verification,
        parent_process_exit_observed: false,
    })
}

impl NativeProviderDeclaration for LockedPalsArenaLaunchV5 {
    fn original_wall_deadline(&self) -> Result<Option<std::time::Instant>, ArenaError> {
        normalized_receipt::pals_original_wall_deadline_v5(self).map(Some)
    }
    type Audit = PalsAdaptiveSessionAuditV5;
    fn scope(&self) -> &'static str {
        match self.storage_policy {
            PalsOwnedStoragePolicy::StrictV5 => "pals_v5_adaptive_exact_source_native_lifecycle_rules_clock",
            PalsOwnedStoragePolicy::AdvisoryV1 => "pals_v5_semantics_storage_advisory_v1_native_lifecycle_rules_clock",
        }
    }
    fn receipt_filename(&self) -> &'static str {
        match self.storage_policy {
            PalsOwnedStoragePolicy::StrictV5 => "pals-arena-receipt.v5.json",
            PalsOwnedStoragePolicy::AdvisoryV1 => "pals-arena-receipt.storage-advisory-v1.json",
        }
    }
    fn receipt_version(&self) -> u32 {
        match self.storage_policy { PalsOwnedStoragePolicy::StrictV5 => 5, PalsOwnedStoragePolicy::AdvisoryV1 => 6 }
    }
    fn startup_filename(&self) -> &'static str {
        "pals-native-startup.v5.json"
    }
    fn termination_filename(&self) -> &'static str {
        "pals-native-termination.v5.json"
    }
    fn expected_provider_sessions(&self) -> usize {
        4
    }
    fn uses_provider_records(&self, _role: NativeEngineRole) -> Result<bool, ArenaError> {
        Ok(true)
    }
    fn provider_export_artifact(
        &self,
        role: NativeEngineRole,
    ) -> Result<Option<&ArtifactRef>, ArenaError> {
        Ok(Some(
            &self
                .endpoint(role)
                .configuration
                .asset(PalsStartupAssetPurposeV5::ExportManifest)?
                .artifact,
        ))
    }
    #[cfg(target_os = "linux")]
    fn validate_runtime_admission(&self) -> Result<(), ArenaError> {
        let a = self
            .admission
            .lock()
            .map_err(|_| invalid("admission ledger poisoned"))?;
        require(
            a.as_ref().is_some_and(|a| {
                a.source_verification.content_verified
                    && a.source_verification.components
                        == self.input.semantic_lock.manifest.source_pins
                    && a.numerical_by_endpoint.len() == 2
                    && a.artifact_domain_by_endpoint.len() == 2
                    && self.input.endpoints.iter().all(|e| {
                        a.artifact_domain_by_endpoint
                            .get(&e.configuration.endpoint.id)
                            .is_some_and(|d| {
                                d.producer_metadata_verified
                                    && d.registered_artifact_bytes_verified_by_launch_owner
                                    && d.numerical_acceptance_verified
                                    && !d.arena_accepted
                                    && d.checkpoint_sha256
                                        == e.configuration.endpoint.model.weights.sha256
                                    && d.artifact_domain
                                        == e.configuration.endpoint.model.artifact_domain
                            })
                    })
                    && self.input.endpoints.iter().all(|e| {
                        a.numerical_by_endpoint
                            .get(&e.configuration.endpoint.id)
                            .is_some_and(|n| {
                                n.compared_cases > 0
                                    && n.compared_cases.checked_mul(2)
                                        == Some(n.completed_native_nn_inputs)
                                    && n.input_tensor_bytes_compared > 0
                                    && n.batch == 1
                                    && n.invocation == PalsNumericalInvocationV5::Fresh
                            })
                    })
            }),
            "actual source/config/model/storage admission must finish before any child launch",
        )
    }
    #[cfg(target_os = "linux")]
    fn validate_records(
        &self,
        r: NativeEngineRole,
        s: &[u8],
        t: &[u8],
        session: &str,
    ) -> Result<(Self::Audit, u32), ArenaError> {
        let a = decode_records(self, r, s, t, session)?;
        let pid = a.process_id;
        Ok((a, pid))
    }
    #[cfg(target_os = "linux")]
    fn validate_records_with_context(
        &self,
        r: NativeEngineRole,
        s: &[u8],
        t: &[u8],
        session: &str,
        _inputs: &crate::NativeProviderInputContext<'_>,
        stdout: &[u8],
    ) -> Result<(Self::Audit, u32), ArenaError> {
        let mut a = decode_records(self, r, s, t, session)?;
        // The pinned runner trace contains all four game PIDs. This specific
        // process must occur exactly once and every game exit must be zero.
        // The common native supervisor separately proves the group is gone.
        crate::native_exit::validate_endpoint_exit_trace(stdout, &[a.process_id], 3)?;
        a.endpoint.process_tree_gone=PalsObservedV3::Observed{value:true,method:"parent pinned-runner four-game process exit trace and native supervisor group cleanup".into()};
        a.parent_process_exit_observed = true;
        let pid = a.process_id;
        Ok((a, pid))
    }
    #[cfg(target_os = "linux")]
    fn validate_external_preflight(
        &self,
        role: NativeEngineRole,
        preflight: &crate::ExternalUciPreflight,
        directory: &cap_std::fs::Dir,
        _inputs: &crate::NativeProviderInputContext<'_>,
    ) -> Result<Vec<(String, Vec<u8>)>, ArenaError> {
        use cap_fs_ext::DirExt;
        let config = &self.endpoint(role).startup_config;
        let endpoint = &self.input.semantic_lock.manifest.engines[role_index(role)];
        require(
            preflight.engine_id == endpoint.id && preflight.stop_and_legal_bestmove_observed,
            "V5 identification/readiness endpoint differs",
        )?;
        let mut evidence = Vec::new();
        let ids = [
            preflight.identification_process.pid,
            preflight.readiness_process.pid,
        ];
        require(
            ids[0] > 0 && ids[1] > 0 && ids[0] != ids[1],
            "metadata probes reused process identity",
        )?;
        for (index, pid) in ids.into_iter().enumerate() {
            let name = format!("preflight-process-{pid}");
            let dir = directory
                .open_dir_nofollow(&name)
                .map_err(|_| invalid("actual V5 metadata probe directory unavailable"))?;
            let bytes = crate::native_runner::linux::read_file(
                &dir,
                "pals-metadata-preflight.v5.json",
                64 << 10,
            )?;
            let value: PalsAdaptivePreflightRecordV5 = crate::decode_json(
                std::str::from_utf8(&bytes).map_err(|_| invalid("metadata probe not UTF8"))?,
            )?;
            require(
                value.schema_version == 5
                    && value.domain == PALS_PREFLIGHT_V5_DOMAIN
                    && value.lock_sha256 == self.input.semantic_lock.canonical_sha256
                    && value.startup_config_sha256 == config.sha256
                    && value.endpoint_id == endpoint.id
                    && value.process_id == pid
                    && value.process_run_id == name
                    && value.model_sessions_created == 0
                    && value.role_requests_submitted == 0
                    && value.cpu_jobs_submitted == 0
                    && value.helper_processes_created == 0
                    && !value.training_executed
                    && (index == 0 || value.rules_only_stop_witness),
                "metadata probe attempted model/worker/helper/training work or identity is stale",
            )?;
            evidence.push((format!("{name}/pals-metadata-preflight.v5.json"), bytes));
        }
        // No game session exists yet in this role-owned runtime namespace.
        let mut entries = 0usize;
        for entry in directory
            .entries()
            .map_err(|_| invalid("preflight directory scan failed"))?
        {
            entries += 1;
            require(
                entries <= 32,
                "metadata preflight runtime entry budget exceeded",
            )?;
            let name = entry
                .map_err(|_| invalid("preflight entry unavailable"))?
                .file_name();
            let name = name
                .to_str()
                .ok_or_else(|| invalid("preflight entry not UTF8"))?;
            require(
                !name.starts_with("native-process-"),
                "preflight created actual game-session evidence",
            )?;
        }
        Ok(evidence)
    }
    #[cfg(target_os = "linux")]
    fn verify_session_evidence(
        &self,
        role: NativeEngineRole,
        pid: u32,
        startup: &[u8],
        directory: &cap_std::fs::Dir,
    ) -> Result<Vec<(String, Vec<u8>)>, ArenaError> {
        use cap_fs_ext::DirExt;
        let s: PalsAdaptiveStartupRecordV5 = crate::decode_json(
            std::str::from_utf8(startup).map_err(|_| invalid("session startup not UTF8"))?,
        )?;
        let dir = directory
            .open_dir_nofollow(&s.process_run_id)
            .map_err(|_| invalid("actual provider evidence directory unavailable"))?;
        let bytes =
            crate::native_runner::linux::read_file(&dir, self.termination_filename(), 256 << 10)?;
        let t: PalsAdaptiveTerminationRecordV5 = crate::decode_json(
            std::str::from_utf8(&bytes).map_err(|_| invalid("actual termination not UTF8"))?,
        )?;
        require(
            t.process_id == pid && t.process_run_id == s.process_run_id,
            "provider evidence has a different process owner",
        )?;
        let mut refs = Vec::new();
        if let PalsObservedV3::Observed { value, .. } = &t.observation.model_verification {
            refs.push(&value.evidence);
        }
        if let PalsObservedV3::Observed { value, .. } = &t.observation.resource_readback {
            refs.push(&value.evidence);
        }
        if let PalsObservedV3::Observed { value, .. } = &t.observation.coordinator {
            refs.push(&value.evidence);
        }
        // Numerical acceptance is independently read from the owner's retained
        // input FDs before children; child-created numerical files confer no authority.
        let admission = self
            .admission
            .lock()
            .map_err(|_| invalid("parent admission ledger poisoned"))?;
        let prefix = admission
            .as_ref()
            .ok_or_else(|| invalid("parent source evidence absent"))?
            .source_verification
            .evidence
            .path
            .strip_suffix(match self.storage_policy {
                PalsOwnedStoragePolicy::StrictV5 => "/source-verification.v5.json",
                PalsOwnedStoragePolicy::AdvisoryV1 => "/source-verification.storage-advisory-v1.json",
            })
            .ok_or_else(|| invalid("parent source artifact topology differs"))?;
        let endpoint = &self.endpoint(role).configuration;
        let expected = format!(
            "{prefix}/{}/{}/",
            endpoint.runtime_directory, s.process_run_id
        );
        let mut names = BTreeSet::new();
        let mut result = Vec::new();
        for artifact in refs {
            artifact.validate()?;
            let name = artifact.path.strip_prefix(&expected).ok_or_else(|| {
                invalid("provider evidence does not belong to its exact process session")
            })?;
            require(
                !name.contains('/')
                    && name.ends_with(".json")
                    && pals_private_snapshot_path_v5(&format!("inputs/{name}")),
                "provider evidence basename escapes owned session",
            )?;
            require(
                names.insert(name),
                "one process evidence artifact reused for different observation classes",
            )?;
            let bytes = crate::native_runner::linux::read_file(&dir, name, 128 << 10)?;
            require(
                bytes.len() as u64 == artifact.bytes
                    && format!("{:x}", Sha256::digest(&bytes)) == artifact.sha256,
                "actual provider evidence content differs from observation",
            )?;
            if let PalsObservedV3::Observed { value, .. } = &t.observation.model_verification
                && artifact == &value.evidence
            {
                validate_pals_model_verification_evidence_v5(
                    &bytes,
                    &t.endpoint_id,
                    pid,
                    &t.process_run_id,
                    value,
                )?;
            }
            if let PalsObservedV3::Observed { value, .. } = &t.observation.resource_readback
                && artifact == &value.evidence
            {
                validate_pals_resource_readback_evidence_v5(
                    &bytes,
                    &t.endpoint_id,
                    pid,
                    &t.process_run_id,
                    value,
                )?;
            }
            if let PalsObservedV3::Observed { value, .. } = &t.observation.coordinator
                && artifact == &value.evidence
            {
                validate_pals_coordinator_evidence_v5(
                    &bytes,
                    &t.endpoint_id,
                    pid,
                    &t.process_run_id,
                    value,
                )?;
            }
            result.push((format!("{}/{name}", s.process_run_id), bytes));
        }
        Ok(result)
    }
    #[cfg(target_os = "linux")]
    fn validate_conversion_manifest(
        &self,
        r: NativeEngineRole,
        _startup: &[u8],
        bytes: &[u8],
        batch: bool,
    ) -> Result<(), ArenaError> {
        require(!batch, "adaptive V5 first profile has no batch policy")?;
        validate_export_content_v5(&self.endpoint(r).configuration, bytes)
    }
    fn validate_clock_trace(
        &self,
        stdout: &[u8],
        pgn: &crate::PairPgnAudit,
    ) -> Result<Option<crate::NativePilotClockAudit>, ArenaError> {
        let p = self.input.semantic_lock.manifest.pair.as_ref().unwrap();
        crate::validate_pilot_clock_trace_with_ply_limit(
            stdout,
            pgn,
            &self.opening,
            NativeGameClockV3 {
                base_ms: p.base_ms,
                increment_ms: p.increment_ms,
            },
            p.max_plies,
        )
        .map(Some)
    }
    fn audit_failure_trace(
        &self,
        stdout: &[u8],
        pair: &crate::PairSpec,
    ) -> Result<Option<crate::NativePilotFailureAudit>, ArenaError> {
        crate::audit_pilot_startup_failure_trace(stdout, pair)
    }
    #[cfg(target_os = "linux")]
    fn validate_process_exit_trace(&self, stdout: &[u8], pids: &[u32]) -> Result<(), ArenaError> {
        crate::native_exit::validate_endpoint_exit_trace(stdout, pids, 0).map(|_| ())
    }
    #[cfg(target_os = "linux")]
    fn validate_provider_session_set(
        &self,
        sessions: &[Self::Audit],
        preflight: &[crate::ExternalUciPreflight],
        _runtime_root: &cap_std::fs::Dir,
        _inputs: &crate::NativeProviderInputContext<'_>,
        stdout: &[u8],
        artifacts: &[ArtifactRef],
    ) -> Result<(), ArenaError> {
        self.validate_runtime_admission()?;
        require(
            sessions.len() == 4 && preflight.len() == 2,
            "paired V5 run requires two independent probes and four actual game sessions",
        )?;
        let mut ids = BTreeSet::new();
        let mut owners = BTreeSet::new();
        let mut per_endpoint = BTreeMap::<String, usize>::new();
        for session in sessions {
            require(
                session.parent_process_exit_observed
                    && ids.insert(session.process_id)
                    && owners.insert(session.endpoint.runtime_owner.clone()),
                "actual game process/owner reused or parent closure not observed",
            )?;
            let endpoint = self
                .input
                .semantic_lock
                .manifest
                .engines
                .iter()
                .find(|d| d.id == session.endpoint_id)
                .ok_or_else(|| invalid("session belongs to another endpoint"))?;
            *per_endpoint.entry(endpoint.id.clone()).or_default() += 1;
            // This is endpoint acceptance only. No fabricated games or whole
            // run receipt is constructed to validate a partial session.
            validate_pals_endpoint_observation_v5(&session.endpoint, endpoint, true, false, false)?;
            require(
                matches!(
                    &session.endpoint.actual_tf32,
                    PalsObservedV3::Observed { value: false, .. }
                ),
                "actual TF32 off must be observed, not inferred from startup",
            )?;
            let PalsObservedV3::Observed { value: n, .. } =
                &session.endpoint.native_numerical_check
            else {
                return Err(invalid("parent independent numerical admission missing"));
            };
            let ledger = self
                .admission
                .lock()
                .map_err(|_| invalid("parent admission ledger poisoned"))?;
            let a = ledger
                .as_ref()
                .ok_or_else(|| invalid("parent admission state absent"))?;
            let actual = a
                .numerical_by_endpoint
                .get(&endpoint.id)
                .ok_or_else(|| invalid("parent numerical endpoint identity missing"))?;
            require(
                n.passed
                    && n.evidence == actual.acceptance
                    && session.parent_verified_source == a.source_verification,
                "session numerical/source proof not bound to independently verified parent inputs",
            )?;
            {
                let PalsObservedV3::Observed { value: model, .. } =
                    &session.endpoint.model_verification
                else {
                    return Err(invalid("actual model verification absent"));
                };
                require(
                    artifacts.iter().any(|f| {
                        f.path == model.evidence.path
                            && f.sha256 == model.evidence.sha256
                            && f.bytes == model.evidence.bytes
                    }),
                    "actual model observation envelope was not preserved by parent",
                )?;
            }
            let PalsObservedV3::Observed {
                value: resources, ..
            } = &session.endpoint.resource_readback
            else {
                return Err(invalid("actual resource readback absent"));
            };
            require(
                artifacts.iter().any(|f| {
                    f.path == resources.evidence.path
                        && f.sha256 == resources.evidence.sha256
                        && f.bytes == resources.evidence.bytes
                }),
                "actual resource observation envelope was not preserved by parent",
            )?;
            let PalsObservedV3::Observed { value: c, .. } = &session.endpoint.coordinator else {
                return Err(invalid("actual coordinator lifecycle absent"));
            };
            require(
                c.nn_inputs_completed > 0 && c.search_evaluations_consumed > 0,
                "Rules-only probe or mock path cannot count as actual native game work",
            )?;
            require(
                artifacts.iter().any(|f| {
                    f.path == c.evidence.path
                        && f.sha256 == c.evidence.sha256
                        && f.bytes == c.evidence.bytes
                }),
                "actual coordinator observation envelope was not preserved by parent",
            )?;
        }
        require(
            self.input
                .semantic_lock
                .manifest
                .engines
                .iter()
                .all(|e| per_endpoint.get(&e.id) == Some(&2)),
            "paired run lacks two fresh game processes for each endpoint",
        )?;
        require(
            preflight
                .iter()
                .map(|p| &p.engine_id)
                .collect::<BTreeSet<_>>()
                == self
                    .input
                    .semantic_lock
                    .manifest
                    .engines
                    .iter()
                    .map(|e| &e.id)
                    .collect(),
            "actual preflight endpoint set differs",
        )?;
        crate::native_exit::validate_endpoint_exit_trace(
            stdout,
            &ids.into_iter().collect::<Vec<_>>(),
            0,
        )?;
        Ok(())
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom, Write};
    #[test]
    fn positional_hash_preserves_descriptor_cursor_on_success_and_failure() {
        let path = std::env::temp_dir().join(format!(
            "rz-v5-input-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let content = b"a bounded private fixture";
        let mut writer = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        writer.write_all(content).unwrap();
        drop(writer);
        let mut file = std::fs::File::open(&path).unwrap();
        file.seek(SeekFrom::Start(7)).unwrap();
        let mut a = ArtifactRef {
            path: "inputs/fixture.bin".into(),
            sha256: format!("{:x}", Sha256::digest(content)),
            bytes: content.len() as u64,
            source: "https://example.org/authored-boundary-fixture".into(),
            license: "MIT".into(),
        };
        assert!(verify_file(&file, &a).is_ok());
        assert_eq!(file.stream_position().unwrap(), 7);
        a.sha256 = "0".repeat(64);
        assert!(verify_file(&file, &a).is_err());
        assert_eq!(file.stream_position().unwrap(), 7);
        drop(file);
        std::fs::remove_file(path).unwrap();
    }
}
