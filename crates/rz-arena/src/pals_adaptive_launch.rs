//! V5 paired execution over the existing exclusive native launch owner.
//! Provider records are V5; immutable V1--V4 records and validators are untouched.
//! This module does not train, create a cloud runtime or silently retry a pair.
use crate::{
    ArenaError, NativeEngineView, NativeLaunchDeclaration, NativeLaunchOwner, NativePairView,
    NativeProviderDeclaration,
};
use rz_experiments::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    path::{Path, PathBuf},
};

mod artifact_domains;
mod evaluation_series;
#[cfg(feature = "pals-collection-onnx")]
mod evaluation_templates;
mod evidence;
mod export_rules;
mod normalized_receipt;
mod numerical;
mod storage;
mod system_v6;
pub use artifact_domains::*;
pub use evaluation_series::*;
#[cfg(feature = "pals-collection-onnx")]
pub use evaluation_templates::*;
pub use evidence::*;
pub use export_rules::*;
pub use normalized_receipt::*;
pub use numerical::*;
pub use numerical::{pals_numerical_input_tensors_v5, pals_numerical_tensor_digest_v5};
pub use storage::*;
#[cfg(target_os = "linux")]
pub(crate) use system_v6::PalsV6RoleSockets;
#[cfg(target_os = "linux")]
pub(crate) use system_v6::verify_original_proxy_snapshots;
pub use system_v6::*;
#[cfg(target_os = "linux")]
pub(crate) use system_v6::{CancelSourceObservation, compiled_cancel_source_observation};
#[cfg(target_os = "linux")]
pub(crate) use system_v6::{DirectNoGameBinding, DirectSmokeBinding};
#[cfg(target_os = "linux")]
pub(crate) use system_v6::{
    LockedNoGameSmokePlanV1, NO_GAME_SMOKE_PLAN_DOMAIN, NoGameExecutionStartV1,
    NoGameSmokeBudgetV1, NoGameSmokeEndpointV1, NoGameSmokePlanV1,
};
#[cfg(target_os = "linux")]
pub(crate) use system_v6::{
    NoGamePreparationFailure, NoGamePreparationStage, NoGameRoleSocketsV1,
    ProxyEntrySmokeSnapshotOwnerV1, prepare_proxy_entry_smoke_snapshot_v1,
};
#[cfg(target_os = "linux")]
pub(crate) use system_v6::{
    NoGameResourceBindFailure, NoGameResourceLimitsV1, NoGameResourceObservationV1,
    NoGameResourceOwnerV1, ResourceComparisonV1, ResourceLimitReadbackV1,
};
#[cfg(target_os = "linux")]
pub(crate) use system_v6::{ProxyEntryFunctionalSmokeOwnerV1, SMOKE_RECEIPT_FILE};
#[cfg(target_os = "linux")]
pub(crate) use system_v6::{
    SmokeCancelProgress, SmokeControlProgress, SmokeDeadlineControl, SmokeGrantObservation,
    validate_smoke_request_ids,
};

pub const PALS_ARENA_V5_DOMAIN: &str = "rz-pals-arena-launch-v5/1";
pub const PALS_STAGING_BYTES_V5: u64 = 18 << 30;

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsSourceFileV5 {
    pub logical_source_path: String,
    pub artifact: ArtifactRef,
    pub snapshot_path: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsEndpointLaunchV5 {
    pub startup_config: ArtifactRef,
    pub startup_snapshot_path: String,
    pub configuration: PalsAdaptiveStartupConfigV5,
    pub expected_uci_name: String,
    pub version: String,
    pub handshake_max_ms: u64,
    /// A pinned public launcher clears ambient variables. Its actual application
    /// is recorded independently by the common native process supervisor.
    pub environment: Option<EngineEnvironmentV2>,
    pub cuda_bundle: Option<CudaBundleBindingV1>,
}
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsArenaLaunchV5 {
    pub domain: String,
    pub semantic_lock: PalsInputLockV5,
    pub semantic_lock_artifact: ArtifactRef,
    pub lock_snapshot_path: String,
    pub runner: ToolIdentity,
    pub opening_artifact: ArtifactRef,
    pub endpoints: [PalsEndpointLaunchV5; 2],
    pub source_files: BTreeMap<String, Vec<PalsSourceFileV5>>,
    pub budget: NativeResourceBudgetV1,
    pub storage: PalsOwnedStoragePlanV5,
}
#[derive(Clone, Debug)]
pub struct LockedPalsArenaLaunchV5 {
    input: PalsArenaLaunchV5,
    sha256: String,
    opening: OpeningSpec,
    endpoint_views: [ExternalUciEndpointV2; 2],
    admission: std::sync::Arc<std::sync::Mutex<Option<evidence::PalsAdaptiveInputAdmission>>>,
    storage_admission: std::sync::Arc<std::sync::Mutex<Option<storage::PalsStorageAdmission>>>,
    storage_lease: Option<std::sync::Arc<PalsOwnedStorageLeaseV5>>,
    storage_policy: PalsOwnedStoragePolicy,
    execution_sha256: Option<String>,
    timing: std::sync::Arc<std::sync::Mutex<Option<normalized_receipt::PalsAdaptiveRunTimingV5>>>,
}
fn invalid(s: &str) -> ArenaError {
    ArenaError::Integrity(s.into())
}
fn require(b: bool, s: &str) -> Result<(), ArenaError> {
    if b { Ok(()) } else { Err(invalid(s)) }
}
fn role_index(role: NativeEngineRole) -> usize {
    match role {
        NativeEngineRole::Baseline => 0,
        NativeEngineRole::Candidate => 1,
    }
}
fn raw_sha(s: &str) -> Result<[u8; 32], ArenaError> {
    require(
        s.len() == 64
            && s.bytes()
                .all(|b| b.is_ascii_digit() || matches!(b, b'a'..=b'f')),
        "noncanonical SHA256",
    )?;
    let mut out = [0; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte =
            u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).map_err(|_| invalid("invalid SHA256"))?;
    }
    Ok(out)
}
fn hex(s: [u8; 32]) -> String {
    s.iter().map(|b| format!("{b:02x}")).collect()
}

impl PalsArenaLaunchV5 {
    pub fn from_json(s: &str) -> Result<Self, ArenaError> {
        let x: Self = crate::decode_json(s)?;
        x.validate()?;
        Ok(x)
    }
    pub fn validate(&self) -> Result<(), ArenaError> {
        self.semantic_lock.verify()?;
        require(
            self.domain == PALS_ARENA_V5_DOMAIN,
            "V5 launch domain differs",
        )?;
        let m = &self.semantic_lock.manifest;
        require(
            matches!(
                m.purpose,
                PalsRunPurposeV5::CollectionPair | PalsRunPurposeV5::EvaluationPair
            ) && m.engines.len() == 2
                && !m.training_authorized,
            "V5 pair launcher cannot train or dispatch a non-pair purpose",
        )?;
        self.semantic_lock_artifact.validate()?;
        self.opening_artifact.validate()?;
        self.runner.binary.validate()?;
        require(
            pals_private_snapshot_path_v5(&self.lock_snapshot_path),
            "unsafe lock snapshot path",
        )?;
        require(
            self.runner.source_url == crate::FASTCHESS_SOURCE_URL
                && self.runner.source_commit == crate::FASTCHESS_SOURCE_COMMIT
                && self.runner.dirty
                && self.runner.dirty_patch.as_ref().is_some_and(|p| {
                    p.sha256 == crate::pals_launch::PALS_CLOCK_REAP_PATCH_SHA256
                        && p.bytes == crate::pals_launch::PALS_CLOCK_REAP_PATCH_BYTES
                })
                && crate::pals_launch::registered_reap_runner(
                    &self.runner.version,
                    &self.runner.binary.sha256,
                    self.runner.binary.bytes,
                ),
            "V5 whole-clock execution requires the registered source and clock/reap-status runner",
        )?;
        let pair = m.pair.as_ref().ok_or_else(|| invalid("V5 pair missing"))?;
        let opening = opening(pair)?;
        crate::validate_opening_artifact_for_spec(
            &opening,
            pair.max_plies,
            &self.opening_artifact,
        )?;
        let mut snapshot_names = BTreeMap::new();
        let mut artifact_names = BTreeMap::new();
        let mut bind = |a: &ArtifactRef, p: &str| -> Result<(), ArenaError> {
            require(
                pals_private_snapshot_path_v5(p),
                "unsafe private input topology",
            )?;
            if let Some(prior) = snapshot_names.insert(p.to_owned(), a.clone()) {
                require(prior == *a, "different inputs alias a private snapshot")?;
            }
            if let Some(prior) = artifact_names.insert(a.path.clone(), p.to_owned()) {
                require(
                    prior == p,
                    "one artifact has contradictory private snapshot names",
                )?;
            }
            Ok(())
        };
        bind(&self.semantic_lock_artifact, &self.lock_snapshot_path)?;
        for (i, e) in self.endpoints.iter().enumerate() {
            e.startup_config.validate()?;
            e.configuration.validate_against(&self.semantic_lock)?;
            require(
                e.configuration.endpoint == m.engines[i]
                    && e.configuration.runtime_directory
                        == if i == 0 {
                            "baseline-runtime"
                        } else {
                            "candidate-runtime"
                        }
                    && !e.expected_uci_name.is_empty()
                    && e.expected_uci_name.len() <= 512
                    && (1..=180_000).contains(&e.handshake_max_ms),
                "endpoint index/runtime/readiness identity differs",
            )?;
            let config = e.configuration.canonical_bytes(&self.semantic_lock)?;
            require(
                e.startup_config.bytes == config.len() as u64
                    && e.startup_config.sha256 == format!("{:x}", Sha256::digest(&config)),
                "startup config artifact must pin exact canonical bytes",
            )?;
            bind(&e.startup_config, &e.startup_snapshot_path)?;
            for a in &e.configuration.assets {
                bind(&a.artifact, &a.snapshot_path)?;
            }
            if let Some(env) = &e.environment {
                env.validate()?;
            }
            let cuda = m.engines[i].model.backend == PalsModelBackendV3::OrtCuda;
            require(
                cuda == e.cuda_bundle.is_some(),
                "CUDA declaration cannot be omitted or injected into a CPU endpoint",
            )?;
            if let Some(bundle) = &e.cuda_bundle {
                require(
                    e.configuration
                        .assets
                        .iter()
                        .any(|a| a.artifact == bundle.manifest),
                    "CUDA manifest absent from private asset registration",
                )?;
                for library in &bundle.files {
                    require(
                        e.configuration
                            .assets
                            .iter()
                            .any(|a| a.artifact == library.artifact),
                        "CUDA library absent from private registration",
                    )?;
                }
            }
        }
        require(
            self.source_files.keys().collect::<BTreeSet<_>>()
                == m.source_pins.keys().collect::<BTreeSet<_>>(),
            "actual source component set differs from semantic pins",
        )?;
        for (component, files) in &self.source_files {
            require(
                !files.is_empty() && files.len() <= 128,
                "source component is empty or too large",
            )?;
            let mut tuples = Vec::with_capacity(files.len());
            for f in files {
                f.artifact.validate()?;
                bind(&f.artifact, &f.snapshot_path)?;
                tuples.push((
                    f.logical_source_path.clone(),
                    f.artifact.bytes,
                    raw_sha(&f.artifact.sha256)?,
                ));
            }
            require(
                hex(source_component_digest_v5(&tuples)?) == m.source_pins[component],
                "source component list/content pin differs",
            )?;
        }
        self.storage.validate()?;
        let unique = self.unique_input_bytes()?;
        let b = self.budget;
        require(
            unique <= b.max_input_bytes
                && b.max_output_bytes <= m.window.output_bytes_max
                && m.window.cleanup_max_ms >= 2
                && b.max_child_processes >= 5
                && b.max_child_processes <= 16
                && (16..=4096).contains(&b.max_runtime_files)
                && (1..=8).contains(&b.max_runtime_depth)
                && b.address_space_per_process_bytes > 0,
            "input/output/process/tree budget is insufficient or unbounded",
        )?;
        require(
            b.max_artifact_bytes
                == b.max_input_bytes
                    .checked_add(b.max_output_bytes)
                    .and_then(|n| n.checked_add(b.max_runtime_bytes))
                    .ok_or_else(|| invalid("artifact reservation overflow"))?
                && b.max_artifact_bytes <= PALS_STAGING_BYTES_V5,
            "aggregate artifact reservation exceeds owned 18GiB staging ceiling",
        )?;
        crate::native_launch::pair_stream_cap(b.max_output_bytes, 2 << 20)?;
        Ok(())
    }
    pub fn declared_artifacts(&self) -> Vec<&ArtifactRef> {
        let mut a = vec![
            &self.runner.binary,
            &self.opening_artifact,
            &self.semantic_lock_artifact,
        ];
        if let Some(p) = &self.runner.dirty_patch {
            a.push(p);
        }
        a.extend(self.semantic_lock.manifest.declared_artifacts());
        for e in &self.endpoints {
            a.push(&e.startup_config);
            a.extend(e.configuration.assets.iter().map(|a| &a.artifact));
            if let Some(env) = &e.environment {
                a.push(&env.launcher);
            }
        }
        a.extend(self.source_files.values().flatten().map(|f| &f.artifact));
        a
    }
    pub fn unique_input_bytes(&self) -> Result<u64, ManifestError> {
        let mut unique = BTreeMap::new();
        for a in self.declared_artifacts() {
            a.validate()?;
            if let Some(prior) = unique.insert(&a.path, a)
                && prior != a
            {
                return Err(ManifestError::Integrity(
                    "logical source path has contradictory asset pins".into(),
                ));
            }
        }
        unique
            .values()
            .try_fold(0u64, |n, a| n.checked_add(a.bytes))
            .ok_or_else(|| ManifestError::Integrity("unique input bytes overflow".into()))
    }
    pub fn lock(&self) -> Result<LockedPalsArenaLaunchV5, ArenaError> {
        self.validate()?;
        let m = &self.semantic_lock.manifest;
        let p = m.pair.as_ref().unwrap();
        let views = std::array::from_fn(|i| ExternalUciEndpointV2 {
            id: m.engines[i].id.clone(),
            role: if i == 0 {
                NativeEngineRole::Baseline
            } else {
                NativeEngineRole::Candidate
            },
            family: "pals-adaptive".into(),
            version: self.endpoints[i].version.clone(),
            expected_uci_name: self.endpoints[i].expected_uci_name.clone(),
            binary: m.engines[i].binary.clone(),
            source: Some(ExternalSourceV2 {
                url: m.engines[i].binary.source.clone(),
                commit: m.engines[i].source_commit.clone(),
                license: m.engines[i].binary.license.clone(),
            }),
            arguments: vec![],
            assets: self.endpoints[i]
                .configuration
                .assets
                .iter()
                .map(|a| a.artifact.clone())
                .collect(),
            requested_options: BTreeMap::new(),
            environment: self.endpoints[i].environment.clone(),
            handshake_timeout_ms: self.endpoints[i].handshake_max_ms,
        });
        Ok(LockedPalsArenaLaunchV5 {
            input: self.clone(),
            sha256: crate::canonical_sha256(&(PALS_ARENA_V5_DOMAIN, self))?,
            opening: opening(p)?,
            endpoint_views: views,
            admission: std::sync::Arc::new(std::sync::Mutex::new(None)),
            storage_admission: std::sync::Arc::new(std::sync::Mutex::new(None)),
            storage_policy: PalsOwnedStoragePolicy::StrictV5,
            execution_sha256: None,
            storage_lease: None,
            timing: std::sync::Arc::new(std::sync::Mutex::new(None)),
        })
    }
}
fn opening(p: &PalsPairPlanV5) -> Result<OpeningSpec, ArenaError> {
    let words: Vec<_> = p.position_command.split_whitespace().collect();
    require(
        words.first() == Some(&"position"),
        "opening lacks position command",
    )?;
    let (initial, fen, history, next) = match words.get(1) {
        Some(&"startpos") => (
            InitialPosition::Startpos,
            None,
            HistoryCompleteness::Complete,
            2,
        ),
        Some(&"fen") if words.len() >= 8 => (
            InitialPosition::Fen,
            Some(words[2..8].join(" ")),
            HistoryCompleteness::UnknownPrefix,
            8,
        ),
        _ => return Err(invalid("opening has unsupported base state")),
    };
    let moves = if words.len() == next {
        vec![]
    } else {
        require(
            words.get(next) == Some(&"moves"),
            "opening move delimiter differs",
        )?;
        words[next + 1..].iter().map(|s| s.to_string()).collect()
    };
    let o = OpeningSpec {
        id: p.opening_id.clone(),
        initial,
        fen,
        moves,
        history,
        history_origin: "registered V5 position command; FEN prefix remains unknown".into(),
    };
    crate::opening_pgn_for_spec(&o, p.max_plies)?;
    Ok(o)
}
impl LockedPalsArenaLaunchV5 {
    pub fn input(&self) -> &PalsArenaLaunchV5 {
        &self.input
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    fn endpoint(&self, role: NativeEngineRole) -> &PalsEndpointLaunchV5 {
        &self.input.endpoints[role_index(role)]
    }
    fn path_for(&self, a: &ArtifactRef) -> Option<String> {
        if *a == self.input.semantic_lock_artifact {
            return Some(self.input.lock_snapshot_path.clone());
        }
        for e in &self.input.endpoints {
            if *a == e.startup_config {
                return Some(e.startup_snapshot_path.clone());
            }
            if let Some(x) = e.configuration.assets.iter().find(|x| x.artifact == *a) {
                return Some(x.snapshot_path.clone());
            }
        }
        self.input
            .source_files
            .values()
            .flatten()
            .find(|f| f.artifact == *a)
            .map(|f| f.snapshot_path.clone())
    }
    fn args(
        &self,
        role: NativeEngineRole,
        root: &Path,
        preflight: bool,
    ) -> Result<Vec<OsString>, ArenaError> {
        require(
            root.is_absolute(),
            "native owner runtime root must be absolute",
        )?;
        let cwd = root
            .parent()
            .ok_or_else(|| invalid("owner runtime root has no private cwd"))?;
        let e = self.endpoint(role);
        let arg = |prefix: &str, p: PathBuf| -> Result<OsString, ArenaError> {
            let s = p
                .to_str()
                .ok_or_else(|| invalid("startup path is not UTF8"))?;
            require(
                !s.chars().any(char::is_control),
                "startup path contains control data",
            )?;
            Ok(format!("{prefix}{s}").into())
        };
        let mut args = vec![
            arg("--startup-config=", cwd.join(&e.startup_snapshot_path))?,
            arg("--startup-lock=", cwd.join(&self.input.lock_snapshot_path))?,
            format!("--startup-config-sha256={}", e.startup_config.sha256).into(),
        ];
        if preflight {
            args.push("--preflight=true".into());
        }
        Ok(args)
    }
}
impl crate::native_launch::sealed::Sealed for LockedPalsArenaLaunchV5 {}
impl NativeLaunchDeclaration for LockedPalsArenaLaunchV5 {
    fn input_sha256(&self) -> &str {
        self.execution_sha256.as_deref().unwrap_or(&self.sha256)
    }
    fn preparation_policy_envelope(&self) -> Option<crate::NativePreparationPolicyEnvelope> {
        self.execution_sha256.as_ref().map(|execution_sha256| {
            crate::NativePreparationPolicyEnvelope {
                domain: "rz-pals-preparation-storage-advisory-v1/1".into(),
                policy: "advisory".into(),
                base_launch_sha256: self.sha256.clone(),
                execution_sha256: execution_sha256.clone(),
            }
        })
    }
    fn view(&self) -> NativePairView<'_> {
        let m = &self.input.semantic_lock.manifest;
        let p = m.pair.as_ref().unwrap();
        let order = p.white_order.each_ref().map(|id| {
            if *id == m.engines[0].id {
                NativeEngineRole::Baseline
            } else {
                NativeEngineRole::Candidate
            }
        });
        NativePairView {
            pair_id: &m.run_id,
            white_order: order,
            opening: &self.opening,
            opening_artifact: &self.input.opening_artifact,
            runner: &self.input.runner,
            clock: NativePairClock::Game(NativeGameClockV3 {
                base_ms: p.base_ms,
                increment_ms: p.increment_ms,
            }),
            max_plies: p.max_plies,
            timeouts: NativeTimeoutsV1 {
                startup_ms: self
                    .input
                    .endpoints
                    .iter()
                    .map(|e| e.handshake_max_ms)
                    .max()
                    .unwrap(),
                handshake_ms: self
                    .input
                    .endpoints
                    .iter()
                    .map(|e| e.handshake_max_ms)
                    .max()
                    .unwrap(),
                runtime_ms: m.window.wall_time_max_ms,
                drain_ms: m.window.cleanup_max_ms,
                shutdown_ms: m.window.cleanup_max_ms / 2,
            },
            budget: self.input.budget,
        }
    }
    fn declared_inputs(&self) -> Vec<&ArtifactRef> {
        self.input.declared_artifacts()
    }
    fn unique_bytes(&self) -> Result<u64, ManifestError> {
        self.input.unique_input_bytes()
    }
    fn engine_view(&self, role: NativeEngineRole) -> Result<NativeEngineView<'_>, ManifestError> {
        let e = &self.endpoint_views[role_index(role)];
        Ok(NativeEngineView {
            engine_id: &e.id,
            artifacts: &[],
            cuda_bundle: self.endpoint(role).cuda_bundle.as_ref(),
            search: None,
            batch_experiment: None,
            external: Some(e),
            environment: e.environment.as_ref(),
        })
    }
    fn provider_name(&self) -> &'static str {
        "PALS-V5"
    }
    fn pair_metadata_cap(&self) -> u64 {
        2 << 20
    }
    fn snapshot_relative_path(&self, a: &ArtifactRef) -> Option<String> {
        self.path_for(a)
            .map(|p| p.trim_start_matches("inputs/").to_owned())
    }
    fn runtime_arguments(
        &self,
        r: NativeEngineRole,
        p: &Path,
    ) -> Result<Vec<OsString>, ArenaError> {
        self.args(r, p, false)
    }
    fn preflight_arguments(
        &self,
        r: NativeEngineRole,
        p: &Path,
    ) -> Result<Vec<OsString>, ArenaError> {
        self.args(r, p, true)
    }
    fn seed(&self) -> u64 {
        self.input
            .semantic_lock
            .manifest
            .pair
            .as_ref()
            .unwrap()
            .seed
    }
    fn validate_execution(&self) -> Result<(), ArenaError> {
        self.input.validate()
    }
    fn advise_drop_input_cache(&self) -> bool {
        true
    }
    fn additional_manifest(&self) -> Option<&ArtifactRef> {
        Some(&self.input.semantic_lock_artifact)
    }
    fn validate_additional_manifest(&self, bytes: &[u8]) -> Result<(), ArenaError> {
        let s = std::str::from_utf8(bytes).map_err(|_| invalid("lock bytes not UTF8"))?;
        require(
            PalsInputLockV5::from_json(s)? == self.input.semantic_lock,
            "actual private lock differs",
        )
    }
    fn amend_execution_limitations(&self, l: &mut Vec<String>) {
        l.push("V5 preserves exact semantic lock and separate source-content/model/resource/physical observations; readyok is a barrier, never option or resource readback. No training, retry, Elo or model promotion is performed by this pair executor.".into());
        match self.storage_policy {
            PalsOwnedStoragePolicy::StrictV5 => l.push("V5 18GiB owned staging/cache admission is independently observed; private Unix read-only copies do not grant a same-UID sandbox or hard kernel disk quota. Actual V-free/PCV mode is accepted only from model content and native receipts.".into()),
            PalsOwnedStoragePolicy::AdvisoryV1 => l.push("Storage policy advisory-v1: output-root usage is not measured by this pair and an independently operated global monitor is required for the 18GiB warning and its status is unknown to this pair; the original marker lease, per-attempt budgets and V5 child physical-drain receipt checks remain required; an independent physical lifetime guard is not established by this profile.".into()),
        }
    }
}

/// Admit owned storage and create the existing read-only FD snapshots. The
/// run entry separately verifies all source/config/model bytes before a child.
/// No engine, ORT session or worker starts during this preparation.
pub fn prepare_pals_pair_launch_v5(
    lock: &LockedPalsArenaLaunchV5,
    source_root: &Path,
    output_root: &Path,
    attempt: &str,
) -> Result<NativeLaunchOwner<LockedPalsArenaLaunchV5>, Box<crate::NativePreparationFailure>> {
    prepare_pals_pair_launch_with_storage_policy(
        lock,
        source_root,
        output_root,
        attempt,
        PalsOwnedStoragePolicy::StrictV5,
    )
}

fn execution_identity(
    lock: &LockedPalsArenaLaunchV5,
    policy: PalsOwnedStoragePolicy,
) -> Option<String> {
    match policy {
        PalsOwnedStoragePolicy::StrictV5 => None,
        PalsOwnedStoragePolicy::AdvisoryV1 => Some(format!(
            "{:x}",
            Sha256::digest(
                format!("rz-pals-storage-policy-advisory-v1/1:{}", lock.sha256).as_bytes()
            )
        )),
    }
}

fn preparation_policy_envelope(
    lock: &LockedPalsArenaLaunchV5,
    policy: PalsOwnedStoragePolicy,
) -> Option<crate::NativePreparationPolicyEnvelope> {
    execution_identity(lock, policy).map(|execution_sha256| {
        crate::NativePreparationPolicyEnvelope {
            domain: "rz-pals-preparation-storage-advisory-v1/1".into(),
            policy: "advisory".into(),
            base_launch_sha256: lock.sha256.clone(),
            execution_sha256,
        }
    })
}

pub fn prepare_pals_pair_launch_with_storage_policy(
    lock: &LockedPalsArenaLaunchV5,
    source_root: &Path,
    output_root: &Path,
    attempt: &str,
    policy: PalsOwnedStoragePolicy,
) -> Result<NativeLaunchOwner<LockedPalsArenaLaunchV5>, Box<crate::NativePreparationFailure>> {
    // The common preparer preserves failure evidence and the caller's owner.
    // Storage preflight has a separate API so failure cannot fabricate a child.
    let admission = match policy {
        PalsOwnedStoragePolicy::StrictV5 => acquire_pals_owned_storage_v5(
            &lock.input.storage,
            output_root,
            lock.input.budget.max_artifact_bytes,
        )
        .map(|(audit, lease)| (storage::PalsStorageAdmission::Strict(audit), lease)),
        PalsOwnedStoragePolicy::AdvisoryV1 => {
            #[cfg(target_os = "linux")]
            {
                storage::acquire_pals_owned_storage_advisory(&lock.input.storage, output_root)
                    .map(|(audit, lease)| (storage::PalsStorageAdmission::Advisory(audit), lease))
            }
            #[cfg(not(target_os = "linux"))]
            {
                Err(invalid("advisory paired native ownership requires Linux"))
            }
        }
    };
    let (storage, lease) = match admission {
        Ok(a) => a,
        Err(cause) => {
            let receipt=crate::NativePreparationReceipt{receipt_version:5,policy_envelope:preparation_policy_envelope(lock, policy),execution_ready:false,input_sha256:execution_identity(lock, policy).unwrap_or_else(|| lock.sha256.clone()),output_directory:"not-created".into(),attempt_created:false,attempted_snapshot_relative_paths:vec![],completed_snapshots:vec![],child_spawned:false,writers_closed:true,input_pins_required:false,original_error:cause.to_string(),subsequent_file_owner:"caller-owned source/staging remains untouched; no copy, child or retry was admitted".into(),automatic_retry:false};
            return Err(Box::new(crate::NativePreparationFailure {
                cause,
                receipt,
                receipt_artifact: None,
                persistence_error: None,
            }));
        }
    };
    // Give each attempt its own observed state. Reusing a semantic registration
    // cannot transfer an earlier attempt's byte/provenance admission proof.
    let mut attempt_lock = lock.clone();
    attempt_lock.admission = std::sync::Arc::new(std::sync::Mutex::new(None));
    attempt_lock.storage_admission = std::sync::Arc::new(std::sync::Mutex::new(Some(storage)));
    attempt_lock.storage_lease = Some(std::sync::Arc::new(lease));
    attempt_lock.storage_policy = policy;
    attempt_lock.execution_sha256 = execution_identity(lock, policy);
    attempt_lock.timing = std::sync::Arc::new(std::sync::Mutex::new(None));
    crate::native_launch::prepare_native_launch_for(
        &attempt_lock,
        source_root,
        output_root,
        attempt,
    )
}
/// Continue serial execution using the preceding normally closed pair's exact
/// owned storage flock. The original archive keeps its pins and evidence; only
/// the common execution lease was released by `into_closed_archive`.
pub fn prepare_pals_pair_launch_after_closed_v5(
    lock: &LockedPalsArenaLaunchV5,
    previous: &crate::ClosedNativePairArchive<LockedPalsArenaLaunchV5>,
    source_root: &Path,
    output_root: &Path,
    attempt: &str,
) -> Result<NativeLaunchOwner<LockedPalsArenaLaunchV5>, Box<crate::NativePreparationFailure>> {
    prepare_pals_pair_launch_after_closed_with_storage_policy(
        lock,
        previous,
        source_root,
        output_root,
        attempt,
        PalsOwnedStoragePolicy::StrictV5,
    )
}

pub fn prepare_pals_pair_launch_after_closed_with_storage_policy(
    lock: &LockedPalsArenaLaunchV5,
    previous: &crate::ClosedNativePairArchive<LockedPalsArenaLaunchV5>,
    source_root: &Path,
    output_root: &Path,
    attempt: &str,
    policy: PalsOwnedStoragePolicy,
) -> Result<NativeLaunchOwner<LockedPalsArenaLaunchV5>, Box<crate::NativePreparationFailure>> {
    #[cfg(target_os = "linux")]
    let admission = (|| {
        lock.input.validate()?;
        require(
            previous.output_root() == Some(output_root)
                && previous.spec().input.storage == lock.input.storage
                && previous.spec().storage_policy == policy,
            "serial storage continuation requires the same original root, owner and profile",
        )?;
        let guard = previous
            .spec()
            .storage_lease
            .as_ref()
            .ok_or_else(|| invalid("closed original pair lacks a live owned storage guard"))?
            .clone();
        let audit = match policy {
            PalsOwnedStoragePolicy::StrictV5 => {
                storage::PalsStorageAdmission::Strict(storage::readmit_pals_owned_storage_v5(
                    &lock.input.storage,
                    output_root,
                    lock.input.budget.max_artifact_bytes,
                    &guard,
                )?)
            }
            PalsOwnedStoragePolicy::AdvisoryV1 => storage::PalsStorageAdmission::Advisory(
                storage::readmit_pals_owned_storage_advisory(
                    &lock.input.storage,
                    output_root,
                    &guard,
                )?,
            ),
        };
        Ok::<_, ArenaError>((audit, guard))
    })();
    #[cfg(not(target_os = "linux"))]
    let admission: Result<
        (
            storage::PalsStorageAdmission,
            std::sync::Arc<PalsOwnedStorageLeaseV5>,
        ),
        ArenaError,
    > = {
        let _ = previous;
        Err(invalid("serial native storage continuation requires Linux"))
    };
    let (audit, guard) = admission.map_err(|cause| {
        let receipt = crate::NativePreparationReceipt {
            receipt_version: 5, policy_envelope: preparation_policy_envelope(lock, policy), execution_ready: false, input_sha256: execution_identity(lock, policy).unwrap_or_else(|| lock.sha256.clone()),
            output_directory: "not-created".into(), attempt_created: false,
            attempted_snapshot_relative_paths: vec![], completed_snapshots: vec![],
            child_spawned: false, writers_closed: true, input_pins_required: false,
            original_error: cause.to_string(),
            subsequent_file_owner: "original closed archive and live storage guard remain owned; no copy, child or retry admitted".into(),
            automatic_retry: false,
        };
        Box::new(crate::NativePreparationFailure { cause, receipt, receipt_artifact: None, persistence_error: None })
    })?;
    let mut next = lock.clone();
    next.admission = std::sync::Arc::new(std::sync::Mutex::new(None));
    next.storage_admission = std::sync::Arc::new(std::sync::Mutex::new(Some(audit)));
    next.storage_lease = Some(guard);
    next.storage_policy = policy;
    next.execution_sha256 = execution_identity(lock, policy);
    next.timing = std::sync::Arc::new(std::sync::Mutex::new(None));
    crate::native_launch::prepare_native_launch_for(&next, source_root, output_root, attempt)
}

/// Exactly one immutable two-game pair. Failures preserve the owner/process
/// evidence in NativePairFailure, including physical completion unknown.
pub fn run_pals_pair_v5(
    owner: NativeLaunchOwner<LockedPalsArenaLaunchV5>,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<crate::NativePairOutput<LockedPalsArenaLaunchV5>, Box<PalsAdaptivePairFailureV5>> {
    if owner.spec.storage_policy != PalsOwnedStoragePolicy::StrictV5 {
        return Err(Box::new(PalsAdaptivePairFailureV5::Admission {
            cause: invalid("V5 pair entry requires strict storage policy"),
            owner,
        }));
    }
    run_pals_pair_with_storage_policy(owner, cancel)
}

/// Execute a prepared pair using its explicitly selected storage policy.
/// Policy-specific receipts are distinct; physical lifetime rules are unchanged.
pub fn run_pals_pair_with_storage_policy(
    owner: NativeLaunchOwner<LockedPalsArenaLaunchV5>,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<crate::NativePairOutput<LockedPalsArenaLaunchV5>, Box<PalsAdaptivePairFailureV5>> {
    if let Err(cause) = normalized_receipt::begin_pals_run_timing_v5(&owner.spec) {
        return Err(Box::new(PalsAdaptivePairFailureV5::Admission {
            cause,
            owner,
        }));
    }
    let clock = owner.spec.timing.clone();
    let admission = match owner.spec.storage_policy {
        PalsOwnedStoragePolicy::StrictV5 => validate_prepared_pals_launch_v5(&owner).map(|_| ()),
        PalsOwnedStoragePolicy::AdvisoryV1 => {
            evidence::validate_prepared_pals_launch_advisory(&owner)
        }
    };
    if let Err(cause) = admission {
        normalized_receipt::finish_pals_run_timing_v5(&clock);
        return Err(Box::new(PalsAdaptivePairFailureV5::Admission {
            cause,
            owner,
        }));
    }
    let result = crate::native_runner::run_native_pair_for(owner, cancel)
        .map_err(|e| Box::new(PalsAdaptivePairFailureV5::Execution(e)));
    normalized_receipt::finish_pals_run_timing_v5(&clock);
    result
}

#[cfg(test)]
mod tests;
