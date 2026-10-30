//! Rovent declarations, locks and execution observations. Legacy manifests are
//! neither converted nor used as this format's canonical representation.
//!
//! Validation checks persisted relationships only. A launcher must separately
//! verify pinned bytes, assemble capabilities and publish owner observations.
//! A valid declaration is not permission to run, actual execution or acceptance.

use crate::{ArtifactRef, ManifestError, decode_json, digest};
use rz_contracts::rovent::{ExecutionMode, PcCompute, ROVENT_SEARCH_REVISION, SearchGoal};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU16;

mod v2;
pub use v2::*;

pub const ROVENT_EXECUTION_DOMAIN: &str = "rz-rovent-execution/1";
pub const ROVENT_LOCK_DOMAIN: &str = "rz-rovent-input-lock/1";
pub const ROVENT_RECEIPT_DOMAIN: &str = "rz-rovent-execution-receipt/1";
pub const ROVENT_CANONICAL_DOMAIN: &str = "rz-rovent-execution-canonical/1";
pub const ROVENT_EXTERNAL_PROFILE_DOMAIN: &str = "rz-rovent-external-engine-profile/1";
pub const ROVENT_POSITION_VISIT_UNIT: &str = "rovent-position-visits/1";
pub const ROVENT_SOURCE_AGGREGATE_DOMAIN: &str = "rz-rovent-source-aggregate/1";

/// Identifies the declared exact source set; it does not attest that a binary
/// was built from it. Callers retain the original component map and separately
/// compare it with the compiled and pinned sources.
///
/// Canonical bytes are domain UTF-8, NUL, the 40 lowercase ASCII commit bytes,
/// component count as u32 LE, then each BTreeMap entry in key order: UTF-8 key
/// byte length as u32 LE, key bytes and 64 lowercase ASCII SHA-256 bytes. This
/// is a new Rovent domain, not the legacy V5 component digest algorithm.
pub fn rovent_source_aggregate_sha256(
    source_commit: &str,
    source_components: &BTreeMap<String, String>,
) -> Result<String, ManifestError> {
    require(
        hex(source_commit, 40)
            && (1..=64).contains(&source_components.len())
            && source_components.iter().all(|(k, v)| text(k) && sha(v)),
        "invalid exact source/component identity",
    )?;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(ROVENT_SOURCE_AGGREGATE_DOMAIN.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(source_commit.as_bytes());
    bytes.extend_from_slice(&(source_components.len() as u32).to_le_bytes());
    for (name, component) in source_components {
        bytes.extend_from_slice(&(name.len() as u32).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(component.as_bytes());
    }
    Ok(digest(&bytes))
}

#[derive(Deserialize, Serialize)]
#[serde(remote = "ExecutionMode", rename_all = "snake_case")]
enum ExecutionModeWire {
    HybridPcr,
    HybridPc,
    CpuPcr,
    CpuPc,
}

macro_rules! choices {
    ($name:ident { $($variant:ident),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq, Ord, PartialOrd)]
        #[serde(rename_all = "snake_case")]
        pub enum $name { $($variant),+ }
    };
}
choices!(RoventPurpose {
    Validation,
    Collection,
    Deployment
});
choices!(RoventBackend {
    RustCpu,
    OrtCpu,
    OrtCuda
});
choices!(RoventActor {
    Manager,
    PcCaller,
    OrtIntra,
    OrtInter,
    OwnR,
    ExternalR,
    Teacher,
    RuntimeHelper
});
choices!(RoventExternalPurpose {
    Teacher,
    ResearchChecker,
    ArenaOpponent
});
choices!(RoventPhysicalState {
    Complete,
    InFlight,
    QuarantinedUnknown
});
choices!(RoventFailure {
    Cancelled,
    Expired,
    SourceIdentity,
    ModelIdentity,
    ModeViolation,
    ResourceAdmission,
    Allocation,
    Cuda,
    PhysicalCompletionUnknown,
    ProcessCleanup,
    Infrastructure,
    OutputQuota
});

/// This is an observation, with its method, never a defaulted declaration.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum RoventObserved<T> {
    Unknown,
    Observed { value: T, method: String },
}
impl<T> RoventObserved<T> {
    fn value(&self) -> Option<&T> {
        match self {
            Self::Unknown => None,
            Self::Observed { value, .. } => Some(value),
        }
    }
    fn validate(&self) -> Result<(), ManifestError> {
        require(
            match self {
                Self::Unknown => true,
                Self::Observed { method, .. } => text(method),
            },
            "observations require their bounded measurement method",
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventModelPin {
    pub semantics: String,
    pub encoding: String,
    pub policy_head: String,
    pub value_head: String,
    pub implementation_sha256: String,
    pub weights: ArtifactRef,
    /// Provenance is pinned independently of model bytes; diagnostic eligibility
    /// must not be flipped by changing the execution purpose.
    pub provenance: ArtifactRef,
    pub diagnostic: bool,
}
impl RoventModelPin {
    fn validate(&self) -> Result<(), ManifestError> {
        require(
            [
                &self.semantics,
                &self.encoding,
                &self.policy_head,
                &self.value_head,
            ]
            .into_iter()
            .all(|v| text(v))
                && sha(&self.implementation_sha256),
            "invalid model semantics/encoding/head/implementation pin",
        )?;
        self.weights.validate()?;
        self.provenance.validate()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "device", rename_all = "snake_case", deny_unknown_fields)]
pub enum RoventDevice {
    Cpu,
    Cuda {
        ordinal: u32,
        registered_device: String,
    },
}
impl RoventDevice {
    fn compute(&self) -> PcCompute {
        match self {
            Self::Cpu => PcCompute::Cpu,
            Self::Cuda { .. } => PcCompute::Accelerator,
        }
    }
    fn validate(&self) -> Result<(), ManifestError> {
        require(
            match self {
                Self::Cpu => true,
                Self::Cuda {
                    ordinal,
                    registered_device,
                } => *ordinal < 64 && text(registered_device),
            },
            "invalid explicit P/C device",
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventPcConfiguration {
    pub proposer: RoventModelPin,
    pub critic: RoventModelPin,
    pub validator: Option<RoventModelPin>,
    pub backend: RoventBackend,
    pub device: RoventDevice,
    pub backend_configuration_sha256: String,
    pub runtime_artifacts: Vec<ArtifactRef>,
    pub fp32: bool,
    pub tf32: bool,
    pub batch_width: u32,
    pub ort_intra_threads: Option<u32>,
    pub ort_inter_threads: Option<u32>,
    pub ort_spinning: Option<bool>,
}
impl RoventPcConfiguration {
    fn validate(&self) -> Result<(), ManifestError> {
        self.proposer.validate()?;
        self.critic.validate()?;
        if let Some(v) = &self.validator {
            v.validate()?;
        }
        self.device.validate()?;
        require(
            sha(&self.backend_configuration_sha256)
                && self.fp32
                && !self.tf32
                && self.batch_width == 1
                && self.runtime_artifacts.len() <= 32,
            "invalid explicit FP32/B1 backend configuration",
        )?;
        for artifact in &self.runtime_artifacts {
            artifact.validate()?;
        }
        match self.backend {
            RoventBackend::RustCpu => require(
                self.device == RoventDevice::Cpu
                    && self.ort_intra_threads.is_none()
                    && self.ort_inter_threads.is_none()
                    && self.ort_spinning.is_none(),
                "Rust CPU must not declare an ORT session",
            ),
            RoventBackend::OrtCpu | RoventBackend::OrtCuda => require(
                (self.backend == RoventBackend::OrtCpu) == (self.device == RoventDevice::Cpu)
                    && self
                        .ort_intra_threads
                        .is_some_and(|v| (1..=256).contains(&v))
                    && self
                        .ort_inter_threads
                        .is_some_and(|v| (1..=256).contains(&v))
                    && self.ort_spinning.is_some()
                    && !self.runtime_artifacts.is_empty(),
                "ORT provider/device/thread declarations are incomplete or contradictory",
            ),
        }
    }
}

/// Pins the independent UCI profile parser/capability validator. Engine-specific
/// option, asset and `go` restrictions remain the external profile owner's job.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventExternalProfilePin {
    pub domain: String,
    pub profile: ArtifactRef,
    pub canonical_sha256: String,
    pub capability_sha256: String,
    pub purpose: RoventExternalPurpose,
    pub binary: ArtifactRef,
    pub source_commit: String,
    pub requested_threads: u32,
    pub requested_hash_mib: u32,
}
impl RoventExternalProfilePin {
    fn validate(&self) -> Result<(), ManifestError> {
        self.profile.validate()?;
        self.binary.validate()?;
        require(
            self.domain == ROVENT_EXTERNAL_PROFILE_DOMAIN
                && sha(&self.canonical_sha256)
                && sha(&self.capability_sha256)
                && hex(&self.source_commit, 40)
                && (1..=256).contains(&self.requested_threads)
                && (1..=1_048_576).contains(&self.requested_hash_mib),
            "invalid external profile/capability/binary/source/resource pins",
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "provider", rename_all = "snake_case", deny_unknown_fields)]
pub enum RoventResearcher {
    Own {
        implementation_sha256: String,
        search_configuration_sha256: String,
    },
    External {
        profile: Box<RoventExternalProfilePin>,
    },
}
impl RoventResearcher {
    fn validate(&self) -> Result<(), ManifestError> {
        match self {
            Self::Own {
                implementation_sha256,
                search_configuration_sha256,
            } => require(
                sha(implementation_sha256) && sha(search_configuration_sha256),
                "invalid own R pins",
            ),
            Self::External { profile } => {
                profile.validate()?;
                require(
                    profile.purpose == RoventExternalPurpose::ResearchChecker,
                    "R must use a research-checker profile, not a teacher/opponent",
                )
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventNnuecPin {
    pub schema: String,
    pub semantics: String,
    pub encoding: String,
    pub implementation_sha256: String,
    /// Includes explicit dimensions, FP32 arithmetic and all loader bounds.
    /// The NNUEC loader validates the pinned configuration's actual contents.
    pub configuration_sha256: String,
    pub weights: ArtifactRef,
    pub provenance: ArtifactRef,
    pub diagnostic: bool,
    pub context_namespace_policy_sha256: String,
    pub local_namespace_policy_sha256: String,
    pub search_conditions_sha256: String,
}
impl RoventNnuecPin {
    fn validate(&self) -> Result<(), ManifestError> {
        self.weights.validate()?;
        self.provenance.validate()?;
        require(
            text(&self.schema)
                && text(&self.semantics)
                && text(&self.encoding)
                && sha(&self.implementation_sha256)
                && sha(&self.configuration_sha256)
                && sha(&self.context_namespace_policy_sha256)
                && sha(&self.local_namespace_policy_sha256)
                && sha(&self.search_conditions_sha256),
            "NNUEC needs explicit model and C/L/search namespace policies",
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventThreadPool {
    /// Responsibilities sharing one physical pool occur together here once.
    pub actors: BTreeSet<RoventActor>,
    pub max_threads: u32,
    /// Simultaneous planned CPU demand, not the number of kernel threads.
    pub cpu_charge_millicpus: u32,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventResourcePlan {
    pub cpu_quota_millicpus: u32,
    pub pools: Vec<RoventThreadPool>,
    pub memory_high_bytes: u64,
    pub memory_max_bytes: u64,
    pub gpu_devices: u32,
    pub physical_nn_concurrency: u32,
    pub topology_policy: String,
}
impl RoventResourcePlan {
    pub fn canonical_sha256(&self) -> Result<String, ManifestError> {
        Ok(digest(&canonical(&("rz-rovent-resource-plan/1", self))?))
    }
    fn validate(&self, engine: &RoventEngineDeclaration) -> Result<(), ManifestError> {
        require(
            (1..=256_000).contains(&self.cpu_quota_millicpus)
                && (1..=32).contains(&self.pools.len())
                && self.memory_high_bytes > 0
                && self.memory_high_bytes <= self.memory_max_bytes
                && self.memory_max_bytes <= 1_u64 << 50
                && self.physical_nn_concurrency == 1
                && text(&self.topology_policy),
            "invalid bounded aggregate resource policy",
        )?;
        let mut actors = BTreeSet::new();
        let mut total = 0_u64;
        for pool in &self.pools {
            require(
                !pool.actors.is_empty()
                    && (1..=1024).contains(&pool.max_threads)
                    && pool.cpu_charge_millicpus > 0,
                "resource pool needs explicit physical threads and nonzero CPU charge",
            )?;
            for actor in &pool.actors {
                require(
                    actors.insert(*actor),
                    "actor occurs in multiple physical resource pools",
                )?;
            }
            total += u64::from(pool.cpu_charge_millicpus);
        }
        require(
            total <= u64::from(self.cpu_quota_millicpus)
                && actors.contains(&RoventActor::Manager)
                && actors.contains(&RoventActor::PcCaller),
            "all concurrent actors must fit the overall CPU quota",
        )?;
        let own_r = matches!(engine.researcher, Some(RoventResearcher::Own { .. }));
        let external_r = matches!(engine.researcher, Some(RoventResearcher::External { .. }));
        require(
            actors.contains(&RoventActor::OwnR) == own_r
                && actors.contains(&RoventActor::ExternalR) == external_r
                && actors.contains(&RoventActor::Teacher) != engine.teachers.is_empty()
                && self.gpu_devices
                    == u32::from(engine.pc.device.compute() == PcCompute::Accelerator),
            "resource actors/devices do not match the selected roles",
        )?;
        if let Some(n) = engine.pc.ort_intra_threads {
            // A one-thread ORT call executes on the registered caller pool. A
            // larger pool must be accounted separately, without a hidden pool.
            require(
                actors.contains(&RoventActor::OrtIntra) == (n > 1),
                "ORT intra pool is missing or double-counted",
            )?;
            if n > 1 {
                self.require_pool(RoventActor::OrtIntra, n - 1)?;
            }
        } else {
            require(
                !actors.contains(&RoventActor::OrtIntra),
                "Rust backend cannot own ORT intra workers",
            )?;
        }
        if let Some(n) = engine.pc.ort_inter_threads {
            require(
                actors.contains(&RoventActor::OrtInter) == (n > 1),
                "ORT inter pool is missing or double-counted",
            )?;
            if n > 1 {
                self.require_pool(RoventActor::OrtInter, n - 1)?;
            }
        } else {
            require(
                !actors.contains(&RoventActor::OrtInter),
                "Rust backend cannot own ORT inter workers",
            )?;
        }
        if let Some(RoventResearcher::External { profile }) = &engine.researcher {
            self.require_pool(RoventActor::ExternalR, profile.requested_threads)?;
        }
        if !engine.teachers.is_empty() {
            let threads: u32 = engine.teachers.iter().map(|p| p.requested_threads).sum();
            self.require_pool(RoventActor::Teacher, threads)?;
        }
        Ok(())
    }
    fn require_pool(&self, actor: RoventActor, threads: u32) -> Result<(), ManifestError> {
        require(
            self.pools
                .iter()
                .any(|p| p.actors.contains(&actor) && p.max_threads >= threads),
            "declared physical pool omits the actor's requested worker threads",
        )
    }
    fn max_threads(&self) -> u64 {
        self.pools.iter().map(|p| u64::from(p.max_threads)).sum()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventSearchDeclaration {
    pub node_unit: String,
    /// Original `go nodes`; not the per-role or native-R grant limit.
    pub requested_nodes: Option<u64>,
    pub depth_plies: Option<NonZeroU16>,
    pub mate_moves: Option<NonZeroU16>,
    pub registered_max_chess_plies: NonZeroU16,
    /// None is unrestricted. Some is a nonempty, ordered, Rules-sealed root set.
    /// Syntax is checked here; Rules must independently seal actual legality.
    pub searchmoves: Option<Vec<String>>,
    pub search_configuration_sha256: String,
    pub root_state_and_history_sha256: String,
    pub rules_profile_sha256: String,
}
impl RoventSearchDeclaration {
    pub fn goal(&self) -> SearchGoal {
        SearchGoal {
            depth_plies: self.depth_plies,
            mate_moves: self.mate_moves,
        }
    }
    fn validate(&self) -> Result<(), ManifestError> {
        self.goal()
            .validate(self.registered_max_chess_plies)
            .map_err(|e| ManifestError::Integrity(format!("Rovent goal: {e}")))?;
        require(
            self.node_unit == ROVENT_POSITION_VISIT_UNIT
                && self.requested_nodes.is_none_or(|n| n > 0)
                && sha(&self.search_configuration_sha256)
                && sha(&self.root_state_and_history_sha256)
                && sha(&self.rules_profile_sha256),
            "missing exact search/root/Rules declaration",
        )?;
        if let Some(moves) = &self.searchmoves {
            require(
                !moves.is_empty()
                    && moves.len() <= 218
                    && moves.iter().all(|m| move_syntax(m))
                    && moves.iter().collect::<BTreeSet<_>>().len() == moves.len(),
                "explicit root move list must be bounded, nonempty and unique",
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventEngineDeclaration {
    pub source_commit: String,
    pub source_components: BTreeMap<String, String>,
    pub binary: ArtifactRef,
    #[serde(with = "ExecutionModeWire")]
    pub mode: ExecutionMode,
    pub pc: RoventPcConfiguration,
    pub researcher: Option<RoventResearcher>,
    pub teachers: Vec<RoventExternalProfilePin>,
    pub nnuec: Option<RoventNnuecPin>,
    pub search: RoventSearchDeclaration,
}
impl RoventEngineDeclaration {
    fn validate(&self, purpose: RoventPurpose) -> Result<(), ManifestError> {
        require(
            hex(&self.source_commit, 40)
                && (1..=64).contains(&self.source_components.len())
                && self
                    .source_components
                    .iter()
                    .all(|(k, v)| text(k) && sha(v)),
            "invalid exact source/component identity",
        )?;
        self.binary.validate()?;
        self.pc.validate()?;
        self.mode
            .validate_assembly(self.pc.device.compute(), self.researcher.is_some())
            .map_err(|e| ManifestError::Integrity(format!("Rovent assembly: {e}")))?;
        if let Some(r) = &self.researcher {
            r.validate()?;
        }
        require(
            self.teachers.len() <= 2
                && (self.teachers.is_empty()
                    || (purpose == RoventPurpose::Collection && self.mode.has_native_r())),
            "teachers are restricted to explicit P-C-R collection",
        )?;
        let mut pins = BTreeSet::new();
        if let Some(RoventResearcher::External { profile }) = &self.researcher {
            pins.insert(&profile.canonical_sha256);
        }
        for teacher in &self.teachers {
            teacher.validate()?;
            require(
                teacher.purpose == RoventExternalPurpose::Teacher
                    && pins.insert(&teacher.canonical_sha256),
                "invalid or duplicate teacher/research purpose/profile",
            )?;
        }
        if let Some(nnuec) = &self.nnuec {
            nnuec.validate()?;
        }
        self.search.validate()?;
        if purpose == RoventPurpose::Deployment {
            require(
                self.pc.validator.is_none()
                    && self.teachers.is_empty()
                    && !self.pc.proposer.diagnostic
                    && !self.pc.critic.diagnostic
                    && self.nnuec.as_ref().is_none_or(|n| !n.diagnostic),
                "deployment must be V/teacher-free and reject diagnostic model assets",
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventExecutionManifest {
    pub domain: String,
    pub contract_revision: String,
    pub run_id: String,
    pub purpose: RoventPurpose,
    pub engine: RoventEngineDeclaration,
    pub resources: RoventResourcePlan,
    pub coordinator: RoventCoordinatorDeclaration,
    pub wall_time_max_ms: u64,
    pub cleanup_max_ms: u64,
    pub output_bytes_max: u64,
    pub agent_renewable: bool,
    pub provenance: Vec<ArtifactRef>,
}
impl RoventExecutionManifest {
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let manifest: Self = decode_json(input)?;
        manifest.validate()?;
        Ok(manifest)
    }
    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            self.domain == ROVENT_EXECUTION_DOMAIN
                && self.contract_revision == ROVENT_SEARCH_REVISION
                && text(&self.run_id),
            "Rovent domain/revision/run mismatch; legacy input is not converted",
        )?;
        self.engine.validate(self.purpose)?;
        self.resources.validate(&self.engine)?;
        self.coordinator.validate()?;
        require(
            !self.engine.mode.has_native_r() || self.coordinator.cpu_credit_limit > 0,
            "P-C-R needs a nonzero native CPU credit limit",
        )?;
        require(
            (1..=86_400_000).contains(&self.wall_time_max_ms)
                && (1..=30_000).contains(&self.cleanup_max_ms)
                && (1..=64_u64 << 30).contains(&self.output_bytes_max)
                && self.agent_renewable
                && (1..=32).contains(&self.provenance.len()),
            "execution needs a finite renewable window, output quota and provenance",
        )?;
        for provenance in &self.provenance {
            provenance.validate()?;
        }
        Ok(())
    }
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        self.validate()?;
        canonical(&(ROVENT_CANONICAL_DOMAIN, self))
    }
    pub fn model_configuration_sha256(&self) -> Result<String, ManifestError> {
        self.validate()?;
        Ok(digest(&canonical(&(
            "rz-rovent-model-configuration/1",
            &self.engine.pc,
            &self.engine.nnuec,
        ))?))
    }
    pub fn lock(self) -> Result<RoventInputLock, ManifestError> {
        let canonical_sha256 = digest(&self.canonical_bytes()?);
        Ok(RoventInputLock {
            domain: ROVENT_LOCK_DOMAIN.into(),
            manifest: self,
            canonical_sha256,
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventCoordinatorDeclaration {
    pub policy_sha256: String,
    pub questions_max: u32,
    pub ancestor_depth_max: u32,
    pub waiting_jobs_max: u32,
    pub role_tickets_max: u32,
    pub cpu_credit_limit: u64,
    pub original_root_budget_sha256: String,
    pub original_deadline_seal_sha256: String,
}
impl RoventCoordinatorDeclaration {
    fn validate(&self) -> Result<(), ManifestError> {
        require(
            sha(&self.policy_sha256)
                && sha(&self.original_root_budget_sha256)
                && sha(&self.original_deadline_seal_sha256)
                && (1..=65_536).contains(&self.questions_max)
                && (1..=1024).contains(&self.ancestor_depth_max)
                && self.ancestor_depth_max <= self.questions_max
                && (1..=1024).contains(&self.waiting_jobs_max)
                && (1..=1024).contains(&self.role_tickets_max),
            "invalid bounded reciprocal coordinator/root credit declaration",
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventInputLock {
    pub domain: String,
    pub manifest: RoventExecutionManifest,
    pub canonical_sha256: String,
}
impl RoventInputLock {
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let lock: Self = decode_json(input)?;
        lock.validate()?;
        Ok(lock)
    }
    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            self.domain == ROVENT_LOCK_DOMAIN
                && sha(&self.canonical_sha256)
                && digest(&self.manifest.canonical_bytes()?) == self.canonical_sha256,
            "Rovent lock domain/canonical digest mismatch",
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventResourceObservation {
    pub limits_applied: RoventObserved<bool>,
    pub process_tree_quota_millicpus: RoventObserved<u32>,
    pub memory_high_bytes: RoventObserved<u64>,
    pub memory_max_bytes: RoventObserved<u64>,
    pub inventory_complete: RoventObserved<bool>,
    pub kernel_threads: RoventObserved<u64>,
    /// Owner verifies complete quota ancestry, per-thread affinity/topology and
    /// ORT settings before sealing the full declared plan. A partial probe must
    /// leave this unknown rather than hashing the declaration back to itself.
    pub verified_plan_readback_sha256: RoventObserved<String>,
    pub host_peak_bytes: RoventObserved<u64>,
    pub vram_peak_bytes: RoventObserved<u64>,
    pub evidence: ArtifactRef,
}
impl RoventResourceObservation {
    fn validate(&self, plan: &RoventResourcePlan) -> Result<bool, ManifestError> {
        self.limits_applied.validate()?;
        self.process_tree_quota_millicpus.validate()?;
        self.memory_high_bytes.validate()?;
        self.memory_max_bytes.validate()?;
        self.inventory_complete.validate()?;
        self.kernel_threads.validate()?;
        self.verified_plan_readback_sha256.validate()?;
        self.host_peak_bytes.validate()?;
        self.vram_peak_bytes.validate()?;
        self.evidence.validate()?;
        require(
            self.process_tree_quota_millicpus
                .value()
                .is_none_or(|v| *v > 0)
                && self.memory_high_bytes.value().is_none_or(|v| *v > 0)
                && self.memory_max_bytes.value().is_none_or(|v| *v > 0)
                && self.kernel_threads.value().is_none_or(|v| *v > 0)
                && self
                    .verified_plan_readback_sha256
                    .value()
                    .is_none_or(|v| sha(v))
                && self.host_peak_bytes.value().is_none_or(|v| *v > 0),
            "invalid resource observation",
        )?;
        let plan_digest = plan.canonical_sha256()?;
        Ok(self.limits_applied.value() == Some(&true)
            && self.process_tree_quota_millicpus.value() == Some(&plan.cpu_quota_millicpus)
            && self.memory_high_bytes.value() == Some(&plan.memory_high_bytes)
            && self.memory_max_bytes.value() == Some(&plan.memory_max_bytes)
            && self.inventory_complete.value() == Some(&true)
            && self
                .kernel_threads
                .value()
                .is_some_and(|n| *n <= plan.max_threads())
            && self.verified_plan_readback_sha256.value() == Some(&plan_digest)
            && self.host_peak_bytes.value().is_some())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventExecutedCalls {
    pub r_calls: u64,
    /// Native PVS owner total; qsearch is a diagnostic subset, not added again.
    pub native_nodes: u64,
    pub qsearch_nodes: u64,
    /// Unknown stays unknown for kernels without a separate SEE counter.
    pub see_calls: Option<u64>,
    pub external_teacher_queries: u64,
    pub external_research_queries: u64,
    pub nnuec_readouts: u64,
    pub validator_models_loaded: u64,
    pub validator_models_generated: u64,
    pub validator_calls: u64,
    pub completed_nn_inputs: u64,
    pub search_consumed_evaluations: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventNodeObservation {
    pub unit: String,
    pub requested_nodes: Option<u64>,
    pub root_entries: u64,
    pub checked_transitions: u64,
    /// The native owner's actual nodes, including repeated position visits.
    /// External reported nodes are raw observations outside this native sum.
    pub native_nodes: u64,
    pub total_node_visits: u64,
}
impl RoventNodeObservation {
    fn validate(&self, search: &RoventSearchDeclaration) -> Result<(), ManifestError> {
        require(
            self.unit == ROVENT_POSITION_VISIT_UNIT
                && self.requested_nodes == search.requested_nodes
                && self.root_entries <= 1
                && sum(&[
                    self.root_entries,
                    self.checked_transitions,
                    self.native_nodes,
                ]) == Some(self.total_node_visits)
                && search
                    .requested_nodes
                    .is_none_or(|limit| self.total_node_visits <= limit),
            "position-visit unit/original quota/counters overflow or disagree",
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventLifecycleObservation {
    pub submitted: u64,
    pub terminal_before_dispatch: u64,
    pub cache_completed: u64,
    pub dispatched: u64,
    pub queued: u64,
    pub physically_completed: u64,
    pub in_flight: u64,
    pub completion_unknown: u64,
    pub released: u64,
    pub nn_inputs_completed: u64,
    pub nn_inputs_accepted: u64,
    pub cache_evaluations_accepted: u64,
    pub search_evaluations_consumed: u64,
    pub cpu_spent: u64,
    pub cpu_outstanding_reserved: u64,
    pub root_budget_sha256: String,
    pub original_deadline_seal_sha256: String,
    pub authority_anchor_view_and_owner_verified: bool,
    pub archive_closed: bool,
    pub evidence: ArtifactRef,
}
impl RoventLifecycleObservation {
    fn validate(&self, declaration: &RoventCoordinatorDeclaration) -> Result<(), ManifestError> {
        self.evidence.validate()?;
        require(
            sum(&[
                self.terminal_before_dispatch,
                self.cache_completed,
                self.dispatched,
                self.queued,
            ]) == Some(self.submitted)
                && sum(&[
                    self.physically_completed,
                    self.in_flight,
                    self.completion_unknown,
                ]) == Some(self.dispatched)
                && sum(&[
                    self.terminal_before_dispatch,
                    self.cache_completed,
                    self.physically_completed,
                ])
                .is_some_and(|n| self.released <= n)
                && self.nn_inputs_completed <= self.physically_completed
                && self.nn_inputs_accepted <= self.nn_inputs_completed
                && self.cache_evaluations_accepted <= self.cache_completed
                && sum(&[self.nn_inputs_accepted, self.cache_evaluations_accepted])
                    .is_some_and(|n| self.search_evaluations_consumed <= n)
                && sum(&[self.cpu_spent, self.cpu_outstanding_reserved])
                    .is_some_and(|n| n <= declaration.cpu_credit_limit)
                && self.root_budget_sha256 == declaration.original_root_budget_sha256
                && self.original_deadline_seal_sha256 == declaration.original_deadline_seal_sha256,
            "ticket/NN/consumption/credit/root accounting is inconsistent",
        )
    }
    fn closed(&self) -> bool {
        self.queued == 0
            && self.in_flight == 0
            && self.completion_unknown == 0
            && self.released == self.submitted
            && self.cpu_outstanding_reserved == 0
            && self.archive_closed
            && self.authority_anchor_view_and_owner_verified
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "proof", rename_all = "snake_case", deny_unknown_fields)]
pub enum RoventMateObservation {
    ForcedMateWithin {
        root_wins: bool,
        upper_plies: u16,
        restricted_loss: bool,
        evidence: ArtifactRef,
    },
    ExactMateDistance {
        root_wins: bool,
        plies: u16,
        restricted_loss: bool,
        evidence: ArtifactRef,
    },
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventGoalObservation {
    /// Selective obligations and real boundary evaluations, not CPU local depth.
    pub completed_chess_plies: Option<u16>,
    /// A Rules terminal can satisfy the nominal target before reaching it.
    /// This flag never changes `completed_chess_plies` to the requested depth.
    pub depth_target_satisfied: bool,
    pub rules_terminal_closed_early: bool,
    pub open_required_obligations: u64,
    pub completed_boundary_evaluations: u64,
    pub mate: Option<RoventMateObservation>,
    pub emitted_score_mate: Option<i32>,
    pub bestmove_emitted: bool,
    pub bestmove_output_authorized: bool,
    pub evidence: ArtifactRef,
}
impl RoventGoalObservation {
    fn validate(&self, search: &RoventSearchDeclaration) -> Result<(), ManifestError> {
        self.evidence.validate()?;
        require(
            !self.bestmove_emitted || self.bestmove_output_authorized,
            "computed ponder/infinite result does not authorize bestmove output",
        )?;
        require(
            self.depth_target_satisfied == self.completed_chess_plies.is_some()
                && (!self.rules_terminal_closed_early || self.depth_target_satisfied),
            "depth satisfaction, completed ply and early-terminal observations disagree",
        )?;
        if let Some(depth) = self.completed_chess_plies {
            require(
                search.depth_plies.is_some_and(|d| {
                    depth <= d.get() && self.rules_terminal_closed_early == (depth < d.get())
                }) && self.open_required_obligations == 0
                    && self.completed_boundary_evaluations > 0,
                "completed depth lacks its registered selective obligations/boundaries",
            )?;
        }
        let mut exact_score = None;
        if let Some(mate) = &self.mate {
            let (wins, plies, restricted_loss, evidence, exact) = match mate {
                RoventMateObservation::ForcedMateWithin {
                    root_wins,
                    upper_plies,
                    restricted_loss,
                    evidence,
                } => (*root_wins, *upper_plies, *restricted_loss, evidence, false),
                RoventMateObservation::ExactMateDistance {
                    root_wins,
                    plies,
                    restricted_loss,
                    evidence,
                } => (*root_wins, *plies, *restricted_loss, evidence, true),
            };
            evidence.validate()?;
            require(
                plies <= search.registered_max_chess_plies.get()
                    && (plies != 0 || !wins)
                    && restricted_loss == (!wins && search.searchmoves.is_some())
                    && (plies == 0 || (plies % 2 == u16::from(wins))),
                "mate distance/parity or restricted-loss provenance is invalid",
            )?;
            if let Some(horizon) = search
                .goal()
                .mate_horizon(wins)
                .map_err(|e| ManifestError::Integrity(format!("Rovent goal: {e}")))?
            {
                require(
                    plies <= horizon.0,
                    "mate proof exceeds the declared mate objective",
                )?;
            }
            if exact {
                let moves = (i32::from(plies) + 1) / 2;
                exact_score = Some(if wins { moves } else { -moves });
            }
        }
        require(
            self.emitted_score_mate.is_none() || self.emitted_score_mate == exact_score,
            "only exact Rules mate distance can authorize score mate",
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventExecutionReceipt {
    pub domain: String,
    pub run_id: String,
    pub input_lock_sha256: String,
    pub functional_accepted: bool,
    /// This is evidence eligibility, not a performance improvement assertion.
    pub controlled_comparison_eligible: bool,
    pub source_content_verified: RoventObserved<bool>,
    pub source_and_binary_evidence: RoventObserved<ArtifactRef>,
    pub model_content_verified: RoventObserved<bool>,
    pub model_configuration_sha256: RoventObserved<String>,
    pub model_content_evidence: RoventObserved<ArtifactRef>,
    pub pc_device: RoventObserved<RoventDevice>,
    pub backend_configuration_sha256: RoventObserved<String>,
    pub calls: RoventObserved<RoventExecutedCalls>,
    pub nodes: RoventObserved<RoventNodeObservation>,
    pub lifecycle: RoventObserved<RoventLifecycleObservation>,
    pub goal: RoventObserved<RoventGoalObservation>,
    pub root_moves_rules_seal: RoventObserved<String>,
    pub external_profiles_verified: RoventObserved<Vec<String>>,
    pub physical_state: RoventObserved<RoventPhysicalState>,
    pub worker_joined_and_processes_closed: RoventObserved<bool>,
    pub resources: RoventResourceObservation,
    pub failures: BTreeSet<RoventFailure>,
    pub full_gc_requested: bool,
    pub memory_contention: bool,
    pub wall_time_ms: u64,
    pub cleanup_ms: u64,
    pub output_bytes: u64,
    pub execution_evidence: ArtifactRef,
}
impl RoventExecutionReceipt {
    pub fn from_json(input: &str, lock: &RoventInputLock) -> Result<Self, ManifestError> {
        let receipt: Self = decode_json(input)?;
        receipt.validate(lock)?;
        Ok(receipt)
    }
    pub fn validate(&self, lock: &RoventInputLock) -> Result<(), ManifestError> {
        lock.validate()?;
        let m = &lock.manifest;
        require(
            self.domain == ROVENT_RECEIPT_DOMAIN
                && self.run_id == m.run_id
                && self.input_lock_sha256 == lock.canonical_sha256,
            "receipt is not bound to this Rovent input lock",
        )?;
        self.source_content_verified.validate()?;
        self.source_and_binary_evidence.validate()?;
        self.model_content_verified.validate()?;
        self.model_configuration_sha256.validate()?;
        self.model_content_evidence.validate()?;
        self.pc_device.validate()?;
        self.backend_configuration_sha256.validate()?;
        self.calls.validate()?;
        self.nodes.validate()?;
        self.lifecycle.validate()?;
        self.goal.validate()?;
        self.root_moves_rules_seal.validate()?;
        self.external_profiles_verified.validate()?;
        self.physical_state.validate()?;
        self.worker_joined_and_processes_closed.validate()?;
        self.execution_evidence.validate()?;
        let resources_match = self.resources.validate(&m.resources)?;
        if let Some(e) = self.source_and_binary_evidence.value() {
            e.validate()?;
        }
        if let Some(e) = self.model_content_evidence.value() {
            e.validate()?;
        }
        if let Some(d) = self.model_configuration_sha256.value() {
            require(sha(d), "invalid observed model configuration digest")?;
            self.actual(
                d == &m.model_configuration_sha256()?,
                RoventFailure::ModelIdentity,
                "observed model configuration differs without a preserved identity failure",
            )?;
        }
        if let Some(d) = self.pc_device.value() {
            d.validate()?;
            self.actual(
                d == &m.engine.pc.device,
                RoventFailure::ModeViolation,
                "actual P/C device differs without a preserved mode failure",
            )?;
        }
        if let Some(d) = self.backend_configuration_sha256.value() {
            require(sha(d), "invalid observed backend digest")?;
            self.actual(
                d == &m.engine.pc.backend_configuration_sha256,
                RoventFailure::ModelIdentity,
                "actual backend differs without a preserved identity failure",
            )?;
        }
        if let Some(v) = self.source_content_verified.value() {
            self.actual(
                *v,
                RoventFailure::SourceIdentity,
                "failed source content verification needs a preserved identity failure",
            )?;
        }
        if let Some(v) = self.model_content_verified.value() {
            self.actual(
                *v,
                RoventFailure::ModelIdentity,
                "failed model content verification needs a preserved identity failure",
            )?;
        }
        if let Some(seal) = self.root_moves_rules_seal.value() {
            require(sha(seal), "invalid Rules root move seal")?;
        }
        if let Some(pins) = self.external_profiles_verified.value() {
            require(
                pins.len() <= 3
                    && pins.iter().all(|p| sha(p))
                    && pins.iter().collect::<BTreeSet<_>>().len() == pins.len(),
                "invalid verified external profile list",
            )?;
        }
        if let Some(calls) = self.calls.value() {
            self.actual(
                m.engine.mode.has_native_r()
                    || (calls.r_calls == 0
                        && calls.native_nodes == 0
                        && calls.qsearch_nodes == 0
                        && calls.see_calls == Some(0)
                        && calls.external_teacher_queries == 0
                        && calls.external_research_queries == 0),
                RoventFailure::ModeViolation,
                "P-C run contains forbidden R/PVS/qsearch/SEE/teacher execution",
            )?;
            self.actual(
                !m.engine.teachers.is_empty() || calls.external_teacher_queries == 0,
                RoventFailure::ModeViolation,
                "undeclared external teacher was queried",
            )?;
            self.actual(
                m.engine.nnuec.is_some() || calls.nnuec_readouts == 0,
                RoventFailure::ModelIdentity,
                "undeclared NNUEC weights were evaluated",
            )?;
            match &m.engine.researcher {
                Some(RoventResearcher::External { .. }) => self.actual(
                    calls.native_nodes == 0
                        && calls.qsearch_nodes == 0
                        && calls.see_calls == Some(0),
                    RoventFailure::ModeViolation,
                    "external R selection contains undeclared own R search work",
                )?,
                _ => self.actual(
                    calls.external_research_queries == 0,
                    RoventFailure::ModeViolation,
                    "external R query was issued without its declared profile",
                )?,
            }
            self.actual(
                m.purpose != RoventPurpose::Deployment
                    || (calls.validator_models_loaded == 0
                        && calls.validator_models_generated == 0
                        && calls.validator_calls == 0
                        && calls.external_teacher_queries == 0),
                RoventFailure::ModeViolation,
                "deployment contains V loading/calls or teacher execution",
            )?;
            self.actual(
                m.engine.pc.validator.is_some()
                    || (calls.validator_models_loaded == 0
                        && calls.validator_models_generated == 0
                        && calls.validator_calls == 0),
                RoventFailure::ModeViolation,
                "undeclared V model was loaded or called",
            )?;
        }
        if let Some(nodes) = self.nodes.value() {
            nodes.validate(&m.engine.search)?;
            if let Some(calls) = self.calls.value() {
                require(
                    nodes.native_nodes == calls.native_nodes,
                    "position-visit and native kernel counters disagree",
                )?;
            }
            self.actual(
                m.engine.mode.has_native_r() || nodes.native_nodes == 0,
                RoventFailure::ModeViolation,
                "P-C position visits contain native R work",
            )?;
        }
        if let Some(lifecycle) = self.lifecycle.value() {
            lifecycle.validate(&m.coordinator)?;
            require(
                lifecycle.completion_unknown == 0
                    || self
                        .failures
                        .contains(&RoventFailure::PhysicalCompletionUnknown),
                "unknown ticket completion requires a preserved failure",
            )?;
            if let Some(calls) = self.calls.value() {
                require(
                    calls.completed_nn_inputs == lifecycle.nn_inputs_completed
                        && calls.search_consumed_evaluations
                            == lifecycle.search_evaluations_consumed,
                    "call and ticket consumption observations disagree",
                )?;
            }
        }
        if let Some(goal) = self.goal.value() {
            goal.validate(&m.engine.search)?;
        }
        let within = self.wall_time_ms <= m.wall_time_max_ms
            && self.cleanup_ms <= m.cleanup_max_ms
            && self.output_bytes <= m.output_bytes_max;
        require(
            self.wall_time_ms <= m.wall_time_max_ms
                || self.failures.contains(&RoventFailure::Expired),
            "expired execution requires an explicit failure",
        )?;
        require(
            self.cleanup_ms <= m.cleanup_max_ms
                || self.failures.contains(&RoventFailure::ProcessCleanup),
            "cleanup overshoot requires an explicit failure",
        )?;
        require(
            self.output_bytes <= m.output_bytes_max
                || self.failures.contains(&RoventFailure::OutputQuota),
            "output quota overshoot requires an explicit failure",
        )?;
        require(
            self.physical_state.value() != Some(&RoventPhysicalState::QuarantinedUnknown)
                || self
                    .failures
                    .contains(&RoventFailure::PhysicalCompletionUnknown),
            "quarantined completion must preserve its failure",
        )?;
        let mut expected_external = m
            .engine
            .teachers
            .iter()
            .map(|p| p.canonical_sha256.clone())
            .collect::<Vec<_>>();
        if let Some(RoventResearcher::External { profile }) = &m.engine.researcher {
            expected_external.push(profile.canonical_sha256.clone());
        }
        expected_external.sort();
        expected_external.dedup();
        let external_matches = self.external_profiles_verified.value().is_some_and(|pins| {
            let mut actual = pins.clone();
            actual.sort();
            actual == expected_external
        });
        require(
            !self.functional_accepted
                || (self.failures.is_empty()
                    && within
                    && self.source_content_verified.value() == Some(&true)
                    && self.source_and_binary_evidence.value().is_some()
                    && self.model_content_verified.value() == Some(&true)
                    && self.model_content_evidence.value().is_some()
                    && self.model_configuration_sha256.value()
                        == Some(&m.model_configuration_sha256()?)
                    && self.pc_device.value() == Some(&m.engine.pc.device)
                    && self.backend_configuration_sha256.value()
                        == Some(&m.engine.pc.backend_configuration_sha256)
                    && self.calls.value().is_some()
                    && self
                        .nodes
                        .value()
                        .is_some_and(|nodes| nodes.root_entries == 1)
                    && self.root_moves_rules_seal.value().is_some()
                    && self
                        .lifecycle
                        .value()
                        .is_some_and(RoventLifecycleObservation::closed)
                    && self.goal.value().is_some()
                    && external_matches
                    && self.physical_state.value() == Some(&RoventPhysicalState::Complete)
                    && self.worker_joined_and_processes_closed.value() == Some(&true)),
            "accepted execution needs exact source/device/backend/root/call/owner evidence",
        )?;
        require(
            !self.controlled_comparison_eligible
                || (self.functional_accepted
                    && resources_match
                    && !self.full_gc_requested
                    && !self.memory_contention),
            "unknown/mismatched resources or GC/contention cannot support controlled comparison",
        )
    }
    pub fn canonical_bytes(&self, lock: &RoventInputLock) -> Result<Vec<u8>, ManifestError> {
        self.validate(lock)?;
        canonical(&(ROVENT_RECEIPT_DOMAIN, self))
    }
    fn actual(
        &self,
        matches: bool,
        failure: RoventFailure,
        reason: &str,
    ) -> Result<(), ManifestError> {
        require(
            matches || (!self.functional_accepted && self.failures.contains(&failure)),
            reason,
        )
    }
}

fn canonical<T: Serialize>(value: &T) -> Result<Vec<u8>, ManifestError> {
    // Round-trip to sorted maps makes canonical object order independent of
    // serde's struct field order and rejects encoding beyond the shared cap.
    let value = serde_json::to_value(value).map_err(|e| ManifestError::Integrity(e.to_string()))?;
    let bytes = serde_json::to_vec(&value).map_err(|e| ManifestError::Integrity(e.to_string()))?;
    require(
        bytes.len() <= crate::MAX_MANIFEST_BYTES,
        "canonical Rovent input exceeds 4 MiB",
    )?;
    Ok(bytes)
}
fn require(ok: bool, reason: &str) -> Result<(), ManifestError> {
    if ok {
        Ok(())
    } else {
        Err(ManifestError::Integrity(format!("Rovent: {reason}")))
    }
}
fn text(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 4096
        && value.trim() == value
        && !value.chars().any(char::is_control)
        && !["unknown", "latest", "placeholder", "todo", "tbd", "unset"]
            .contains(&value.to_ascii_lowercase().as_str())
}
fn hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        && value.bytes().any(|b| b != b'0')
}
fn sha(value: &str) -> bool {
    hex(value, 64)
}
fn sum(values: &[u64]) -> Option<u64> {
    values.iter().try_fold(0_u64, |a, b| a.checked_add(*b))
}
fn move_syntax(m: &str) -> bool {
    let b = m.as_bytes();
    (b.len() == 4 || b.len() == 5)
        && (b'a'..=b'h').contains(&b[0])
        && (b'1'..=b'8').contains(&b[1])
        && (b'a'..=b'h').contains(&b[2])
        && (b'1'..=b'8').contains(&b[3])
        && b[..2] != b[2..4]
        && (b.len() == 4 || b"qrbn".contains(&b[4]))
}

#[cfg(test)]
mod tests;
