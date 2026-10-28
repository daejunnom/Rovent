//! Strict Rovent-only startup values. Legacy V5 DTOs are neither decoded nor
//! converted here. This declaration owns constructor values, not observations.

pub mod v2;
pub mod v3;
pub mod v4;

use super::{fail, hex};
use crate::pals_adaptive::{
    AdaptiveExecutionLimits, RoventPeerFollowupPolicy, RoventReciprocalPolicy,
};
use crate::pals_native::ticketed::TicketServiceLimits;
use rz_contracts::Digest;
use rz_contracts::rovent::{ExecutionMode, SearchRecipient};
use rz_eval::error::{BackendError, FailureKind as K};
use rz_eval::onnx::{NativeLoadingProfile, Provider};
use rz_eval::pals_onnx::{PalsCuDnnConvSearch, PalsOnnxConfig};
use rz_experiments::{ArtifactRef, RoventDevice, RoventInputLock, RoventResearcher};
use rz_runtime::adaptive::{AdaptiveController, ControllerConfig, CoordinatorPolicy, LogicalCpu};
use rz_search::cpu::{CpuConfig, CpuOrderingPolicy, CpuProfile};
use rz_search::external_cpu::{ExternalGoConstraint, ExternalSlicePolicy};
use rz_search::pals::engine::{PalsSemanticConfig, PcSemanticProfile, PostRepairRecheckPolicy};
use rz_search::pals::review_context::{ReviewAnchorMode, ReviewIndexLimits, ReviewSelectionPolicy};
use rz_search::pals::rovent_frontier::FrontierLimits;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::{num::NonZeroU16, time::Duration};

pub const ROVENT_STARTUP_CONFIGURATION_DOMAIN: &str = "rz-rovent-startup-configuration/1";
pub const ROVENT_SEARCH_CONFIGURATION_DOMAIN: &str = "rz-rovent-search-configuration/1";
pub const ROVENT_NNUEC_CONDITIONS_DOMAIN: &str = "rz-rovent-nnuec-search-conditions/1";
pub const ROVENT_STARTUP_CONFIGURATION_MAX_BYTES: usize = 256 << 10;

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventStartupConfiguration {
    pub domain: String,
    pub semantic: SemanticConfiguration,
    pub review: ReviewConfiguration,
    pub index: IndexConfiguration,
    pub frontier: FrontierConfiguration,
    pub native: NativeConfiguration,
    pub execution: ExecutionConfiguration,
}

/// Exact version dispatch over original bytes. No variant upgrades or supplies
/// policy defaults for another version's immutable wire contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StartupConfigurationDocument {
    V1(RoventStartupConfiguration),
    V2(v2::RoventStartupConfigurationV2),
    V3(v3::RoventStartupConfigurationV3),
    V4(v4::RoventStartupConfigurationV4),
}
impl StartupConfigurationDocument {
    pub fn from_json(bytes: &[u8]) -> Result<Self, BackendError> {
        if bytes.is_empty() || bytes.len() > ROVENT_STARTUP_CONFIGURATION_MAX_BYTES {
            return Err(invalid("startup configuration byte budget"));
        }
        #[derive(Deserialize)]
        struct DomainProbe {
            domain: String,
        }
        let domain: DomainProbe = serde_json::from_slice(bytes)
            .map_err(|_| invalid("strict startup configuration version JSON"))?;
        match domain.domain.as_str() {
            ROVENT_STARTUP_CONFIGURATION_DOMAIN => {
                RoventStartupConfiguration::from_json(bytes).map(Self::V1)
            }
            v2::ROVENT_STARTUP_CONFIGURATION_V2_DOMAIN => {
                v2::RoventStartupConfigurationV2::from_json(bytes).map(Self::V2)
            }
            v3::ROVENT_STARTUP_CONFIGURATION_V3_DOMAIN => {
                v3::RoventStartupConfigurationV3::from_json(bytes).map(Self::V3)
            }
            v4::ROVENT_STARTUP_CONFIGURATION_V4_DOMAIN => {
                v4::RoventStartupConfigurationV4::from_json(bytes).map(Self::V4)
            }
            _ => Err(invalid("startup configuration version domain")),
        }
    }
    pub fn fields(&self) -> ConfigurationFields<'_> {
        match self {
            Self::V1(config) => config.fields(),
            Self::V2(config) => config.fields(),
            Self::V3(config) => config.fields(),
            Self::V4(config) => config.fields(),
        }
    }
    pub fn validate(&self) -> Result<(), BackendError> {
        match self {
            Self::V1(config) => config.validate(),
            Self::V2(config) => config.validate(),
            Self::V3(config) => config.validate(),
            Self::V4(config) => config.validate(),
        }
    }
    pub fn admit_against(&self, lock: &RoventInputLock) -> Result<(), BackendError> {
        match self {
            Self::V1(config) => config.admit_against(lock),
            Self::V2(config) => config.admit_against(lock),
            Self::V3(config) => config.admit_against(lock),
            // /4 requires a distinct policy-aware lock and receipt. A /1 lock
            // cannot attest the new policy even if common fields match.
            Self::V4(_) => Err(invalid(
                "startup configuration /4 requires policy-aware lock",
            )),
        }
    }
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, BackendError> {
        match self {
            Self::V1(config) => config.canonical_bytes(),
            Self::V2(config) => config.canonical_bytes(),
            Self::V3(config) => config.canonical_bytes(),
            Self::V4(config) => config.canonical_bytes(),
        }
    }
    pub fn configuration_digest(&self) -> Result<[u8; 32], BackendError> {
        match self {
            Self::V1(config) => config.configuration_digest(),
            Self::V2(config) => config.configuration_digest(),
            Self::V3(config) => config.configuration_digest(),
            Self::V4(config) => config.configuration_digest(),
        }
    }
    pub fn search_configuration_digest(&self) -> Result<[u8; 32], BackendError> {
        match self {
            Self::V1(config) => config.search_configuration_digest(),
            Self::V2(config) => config.search_configuration_digest(),
            Self::V3(config) => config.search_configuration_digest(),
            Self::V4(config) => config.search_configuration_digest(),
        }
    }
    pub fn nnuec_conditions_digest(&self) -> Result<Option<[u8; 32]>, BackendError> {
        match self {
            Self::V1(config) => config.nnuec_search_conditions_digest(),
            Self::V2(config) => config.nnuec_search_conditions_digest(),
            Self::V3(config) => config.nnuec_search_conditions_digest(),
            Self::V4(config) => config.nnuec_search_conditions_digest(),
        }
    }
    pub fn require_supported_native_capabilities(&self) -> Result<(), BackendError> {
        match self {
            Self::V1(config) => config.require_supported_native_capabilities(),
            Self::V2(config) => config.require_supported_native_capabilities(),
            Self::V3(config) => config.require_supported_native_capabilities(),
            Self::V4(config) => config.require_supported_native_capabilities(),
        }
    }
    /// The historical versions declare the historical own-R policy. /4
    /// explicitly selects it; this is a declaration, not an owner readback.
    pub fn own_r_root_tie_policy(
        &self,
    ) -> Result<Option<rz_search::cpu::CpuRootTiePolicy>, BackendError> {
        self.validate()?;
        match self {
            Self::V4(config) => config.own_r_root_tie_policy(),
            Self::V1(_) | Self::V2(_) | Self::V3(_) => Ok(match self.fields().execution {
                ExecutionConfiguration::OwnResearcher { .. } => {
                    Some(rz_search::cpu::CpuRootTiePolicy::LegacyFirstSearched)
                }
                _ => None,
            }),
        }
    }
    /// Legacy PC and PCR keep their separately established policy choices.
    /// /2 and /3 supply a common explicit selection for both assemblies.
    pub fn post_repair_policy_for(&self, mode: ExecutionMode) -> PostRepairRecheckPolicy {
        match self {
            Self::V1(_) => match mode {
                ExecutionMode::CpuPc | ExecutionMode::HybridPc => PostRepairRecheckPolicy::Disabled,
                ExecutionMode::CpuPcr | ExecutionMode::HybridPcr => {
                    PostRepairRecheckPolicy::IterativeFrozenModelWdlV2
                }
            },
            Self::V2(config) => config.post_repair_policy(),
            Self::V3(config) => config.post_repair_policy(),
            Self::V4(config) => config.post_repair_policy(),
        }
    }
    /// Legacy external declarations carry no explicit query selection. They
    /// remain decodable metadata and cannot silently acquire Combined work.
    pub fn external_question_limit(&self) -> Result<Option<ExternalGoConstraint>, BackendError> {
        match self {
            Self::V3(config) => config.external_question_limit(),
            Self::V4(config) => config.external_question_limit(),
            Self::V1(_) | Self::V2(_) => {
                self.validate()?;
                if matches!(
                    self.fields().execution,
                    ExecutionConfiguration::ExternalResearcher { .. }
                ) {
                    return Err(fail(
                        K::UnsupportedModel,
                        "UnsupportedCapability: external researcher requires explicit startup configuration /3 question_limit",
                    ));
                }
                Ok(None)
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SemanticConfiguration {
    pub beam_width: usize,
    /// Legacy generation chunks remain at most 16 plies.
    pub line_plies: usize,
    pub max_nodes: usize,
    pub max_records: usize,
    pub max_role_calls: u64,
    /// Original-root prefixes are a separate Rovent capability, at most 256.
    pub maximum_root_prefix_plies: u16,
}
impl SemanticConfiguration {
    pub fn actual(&self) -> Result<PcSemanticProfile, BackendError> {
        let actual = PcSemanticProfile {
            semantic: PalsSemanticConfig {
                beam_width: self.beam_width,
                line_plies: self.line_plies,
                max_nodes: self.max_nodes,
                max_records: self.max_records,
                max_role_calls: self.max_role_calls,
            },
            maximum_root_prefix_plies: self.maximum_root_prefix_plies,
        };
        actual.validate().map_err(|_| invalid("semantic limits"))?;
        Ok(actual)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AnchorMode {
    CurrentRootOnly,
    AnchoredCompact,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReviewConfiguration {
    pub anchor_mode: AnchorMode,
    pub token_budget: u16,
    pub max_records: u16,
    pub max_anchors: u8,
}
impl ReviewConfiguration {
    pub fn actual(&self) -> Result<ReviewSelectionPolicy, BackendError> {
        let actual = ReviewSelectionPolicy {
            mode: match self.anchor_mode {
                AnchorMode::CurrentRootOnly => ReviewAnchorMode::CurrentRootOnly,
                AnchorMode::AnchoredCompact => ReviewAnchorMode::AnchoredCompact,
            },
            token_budget: self.token_budget,
            max_records: self.max_records,
            max_anchors: self.max_anchors,
            // A required bundle is live per-question authority, never startup.
            required_bundle: None,
        };
        actual.validate().map_err(|_| invalid("review selection"))?;
        Ok(actual)
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct IndexConfiguration {
    pub active_bundles: usize,
    pub records_per_bundle: usize,
}
impl IndexConfiguration {
    pub fn actual(&self) -> Result<ReviewIndexLimits, BackendError> {
        if !(1..=256).contains(&self.active_bundles) || !(1..=64).contains(&self.records_per_bundle)
        {
            return Err(invalid("review index limits"));
        }
        Ok(ReviewIndexLimits {
            active_bundles: self.active_bundles,
            records_per_bundle: self.records_per_bundle,
        })
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FrontierConfiguration {
    pub maximum_goal_plies: u16,
    pub maximum_frames: usize,
    pub maximum_paths: usize,
    pub maximum_mate_nodes: usize,
    pub maximum_native_receipts: usize,
}
impl FrontierConfiguration {
    pub fn actual(&self) -> Result<FrontierLimits, BackendError> {
        if !(1..=256).contains(&self.maximum_goal_plies)
            || !(1..=1024).contains(&self.maximum_frames)
            || !(1..=65_536).contains(&self.maximum_paths)
            || !(1..=65_536).contains(&self.maximum_mate_nodes)
            || !(1..=65_536).contains(&self.maximum_native_receipts)
        {
            return Err(invalid("frontier limits"));
        }
        Ok(FrontierLimits {
            maximum_goal_plies: NonZeroU16::new(self.maximum_goal_plies).unwrap(),
            maximum_frames: self.maximum_frames,
            maximum_paths: self.maximum_paths,
            maximum_mate_nodes: self.maximum_mate_nodes,
            maximum_native_receipts: self.maximum_native_receipts,
        })
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum LoadingProfile {
    EagerCuda12Cudnn9V1,
    CuDnnShimLazyV1,
}
impl LoadingProfile {
    pub fn actual(self) -> NativeLoadingProfile {
        match self {
            Self::EagerCuda12Cudnn9V1 => NativeLoadingProfile::EagerCuda12Cudnn9V1,
            Self::CuDnnShimLazyV1 => NativeLoadingProfile::CuDnnShimLazyV1,
        }
    }
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CuDnnSearch {
    LegacyImplicitExhaustive,
    ExperimentalDefault,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativeConfiguration {
    pub cache_public_memory: bool,
    pub device_public_memory: bool,
    pub cudnn_conv_search: CuDnnSearch,
    pub loading_profile: Option<LoadingProfile>,
    pub cuda_arena_bytes: Option<u64>,
    pub drain_ms: u64,
    /// These are admission ceilings, not measured process/VRAM peaks.
    pub prepared_host_bytes_per_ticket: u64,
    pub output_host_bytes_per_ticket: u64,
    pub command_capacity: usize,
}
impl NativeConfiguration {
    pub fn tickets(&self, token_budget: u16) -> Result<TicketServiceLimits, BackendError> {
        if self.prepared_host_bytes_per_ticket == 0
            || self.prepared_host_bytes_per_ticket > (1 << 30)
            || self.output_host_bytes_per_ticket == 0
            || self.output_host_bytes_per_ticket > (1 << 30)
            || !(1..=8).contains(&self.command_capacity)
            || !(1..=30_000).contains(&self.drain_ms)
        {
            return Err(invalid("native admission limits"));
        }
        Ok(TicketServiceLimits {
            review_token_budget: token_budget,
            prepared_host_bytes_per_ticket: self.prepared_host_bytes_per_ticket,
            output_host_bytes_per_ticket: self.output_host_bytes_per_ticket,
        })
    }
    pub fn onnx(
        &self,
        device: &RoventDevice,
        explicit_control: bool,
    ) -> Result<PalsOnnxConfig, BackendError> {
        let provider = match (device, self.cuda_arena_bytes) {
            (RoventDevice::Cpu, None) => Provider::Cpu,
            (RoventDevice::Cuda { ordinal, .. }, Some(bytes))
                if bytes != 0 && bytes <= (6_u64 << 30) =>
            {
                Provider::Cuda {
                    device_id: i32::try_from(*ordinal).map_err(|_| invalid("CUDA ordinal"))?,
                    arena_bytes: usize::try_from(bytes)
                        .map_err(|_| invalid("CUDA arena address space"))?,
                }
            }
            _ => return Err(invalid("explicit CPU/CUDA arena selection")),
        };
        let actual = PalsOnnxConfig {
            provider,
            intra_threads: 1,
            cache_public_memory: self.cache_public_memory,
            device_public_memory: self.device_public_memory,
            cudnn_conv_search: match self.cudnn_conv_search {
                CuDnnSearch::LegacyImplicitExhaustive => {
                    PalsCuDnnConvSearch::LegacyImplicitExhaustive
                }
                CuDnnSearch::ExperimentalDefault => PalsCuDnnConvSearch::ExperimentalDefault,
            },
        };
        actual.validate_execution_selection(
            self.loading_profile.map(LoadingProfile::actual),
            explicit_control,
        )?;
        Ok(actual)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum EventOrder {
    Synchronous,
    Fixed,
    Adaptive,
}
impl EventOrder {
    pub fn actual(self) -> CoordinatorPolicy {
        match self {
            Self::Synchronous => CoordinatorPolicy::Synchronous,
            Self::Fixed => CoordinatorPolicy::Fixed,
            Self::Adaptive => CoordinatorPolicy::Adaptive,
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct QueueConfiguration {
    pub waiting_jobs_max: usize,
    pub worker_commands_max: usize,
    pub role_tickets_max: usize,
    pub paused_bytes_per_worker_max: usize,
}
impl QueueConfiguration {
    pub fn actual(&self) -> Result<AdaptiveExecutionLimits, BackendError> {
        let actual = AdaptiveExecutionLimits {
            waiting_jobs_max: self.waiting_jobs_max,
            worker_commands_max: self.worker_commands_max,
            role_tickets_max: self.role_tickets_max,
            paused_bytes_per_worker_max: self.paused_bytes_per_worker_max,
        };
        actual
            .validate()
            .map_err(|_| invalid("queue/paused ceilings"))?;
        Ok(actual)
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ControllerConfiguration {
    pub fixed_node_credit: u64,
    pub fixed_soft_lease_us: u64,
    pub initial_workers: u8,
    pub max_workers: u8,
    pub consecutive_observations: u8,
    pub worker_cooldown_ms: u64,
}
impl ControllerConfiguration {
    pub fn actual(&self, policy: EventOrder) -> Result<ControllerConfig, BackendError> {
        let actual = ControllerConfig {
            policy: policy.actual(),
            fixed_node_credit: self.fixed_node_credit,
            fixed_soft_lease: Duration::from_micros(self.fixed_soft_lease_us),
            initial_workers: self.initial_workers,
            max_workers: self.max_workers,
            consecutive_observations: self.consecutive_observations,
            worker_cooldown: Duration::from_millis(self.worker_cooldown_ms),
        };
        AdaptiveController::new(actual).map_err(|_| invalid("CPU controller"))?;
        Ok(actual)
    }
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CpuProfileSelection {
    PlanAssisted,
    Independent,
    RelaxedSelectivity,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum OrderingSelection {
    LegacyMvvLvaV1,
    LegalSeeV1,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CpuAffinity {
    pub group: u16,
    pub index: u16,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CpuPlanConfiguration {
    pub profile: CpuProfileSelection,
    pub tt_entries: usize,
    pub max_depth: u16,
    pub quiescence_ply: u16,
    pub ordering: OrderingSelection,
    pub paused_bytes_limit: usize,
    pub preserve_paused_side_scope: bool,
    pub affinity: Option<Vec<CpuAffinity>>,
}
impl CpuPlanConfiguration {
    pub fn actual(&self) -> Result<CpuConfig, BackendError> {
        let actual = CpuConfig {
            profile: match self.profile {
                CpuProfileSelection::PlanAssisted => CpuProfile::PlanAssisted,
                CpuProfileSelection::Independent => CpuProfile::Independent,
                CpuProfileSelection::RelaxedSelectivity => CpuProfile::RelaxedSelectivity,
            },
            tt_entries: self.tt_entries,
            max_depth: self.max_depth,
            quiescence_ply: self.quiescence_ply,
        };
        actual
            .tt_allocation_bytes()
            .map_err(|_| invalid("native TT allocation bound"))?;
        if !(1..=64).contains(&self.max_depth)
            || self.quiescence_ply > 32
            || !(1..=rz_search::cpu::CPU_PAUSED_STACK_MAX_BYTES).contains(&self.paused_bytes_limit)
            || self.affinity.as_ref().is_some_and(|cpus| {
                cpus.is_empty()
                    || cpus.len() > 1024
                    || cpus
                        .iter()
                        .enumerate()
                        .any(|(i, cpu)| cpus[..i].contains(cpu))
            })
        {
            return Err(invalid("native CPU plan"));
        }
        Ok(actual)
    }
    pub fn ordering(&self) -> CpuOrderingPolicy {
        match self.ordering {
            OrderingSelection::LegacyMvvLvaV1 => CpuOrderingPolicy::LegacyMvvLvaV1,
            OrderingSelection::LegalSeeV1 => CpuOrderingPolicy::LegalSeeV1,
        }
    }
    pub fn affinity(&self) -> Option<Vec<LogicalCpu>> {
        self.affinity.as_ref().map(|cpus| {
            cpus.iter()
                .map(|cpu| LogicalCpu {
                    group: cpu.group,
                    index: cpu.index,
                })
                .collect()
        })
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "value", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResearchValueConfiguration {
    BaselineBoard,
    NnuecV2 {
        context_namespace_policy: String,
        local_namespace_policy: String,
    },
}
/// Shared P/C readout does not grant an R searcher, stack or node ledger.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PcNnuecReadoutConfiguration {
    pub context_namespace_policy: String,
    pub local_namespace_policy: String,
}
/// Declaration-only assembly selection. A selected variant does not establish
/// a loaded NNUEC owner, preparation actor or a performed shared readout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeNnuecSelection {
    Absent,
    SharedPc,
    OwnResearcher,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Recipient {
    Proposer,
    Critic,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReciprocalConfiguration {
    pub recipient: Recipient,
    pub max_questions_per_job: u16,
    pub node_interval: u64,
    pub first_question_sequence: u64,
}
impl ReciprocalConfiguration {
    pub fn actual(
        &self,
        mode: rz_contracts::rovent::ExecutionMode,
        root_budget: Digest,
    ) -> Result<RoventReciprocalPolicy, BackendError> {
        if !(1..=64).contains(&self.max_questions_per_job)
            || self.node_interval == 0
            || self.first_question_sequence == 0
        {
            return Err(invalid("reciprocal question limits"));
        }
        Ok(RoventReciprocalPolicy {
            mode,
            root_budget,
            recipient: match self.recipient {
                Recipient::Proposer => SearchRecipient::Proposer,
                Recipient::Critic => SearchRecipient::Critic,
            },
            max_questions_per_job: self.max_questions_per_job,
            node_interval: self.node_interval,
            first_question_sequence: self.first_question_sequence,
        })
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PeerConfiguration {
    pub pc_peer: bool,
    pub researcher: bool,
    pub max_sources_per_root: u16,
}
impl PeerConfiguration {
    pub fn actual(&self) -> Result<RoventPeerFollowupPolicy, BackendError> {
        if !(self.pc_peer || self.researcher) || !(1..=64).contains(&self.max_sources_per_root) {
            return Err(invalid("peer source limits"));
        }
        Ok(RoventPeerFollowupPolicy {
            pc_peer: self.pc_peer,
            researcher: self.researcher,
            max_sources_per_root: self.max_sources_per_root,
        })
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ExternalSliceConfiguration {
    pub quantum_ms: u64,
    pub stop_drain_reserve_ms: u64,
    pub requested_reported_nodes: u64,
    pub multipv: u16,
}
impl ExternalSliceConfiguration {
    pub fn actual(&self) -> Result<ExternalSlicePolicy, BackendError> {
        let actual = ExternalSlicePolicy {
            quantum: Duration::from_millis(self.quantum_ms),
            stop_drain_reserve: Duration::from_millis(self.stop_drain_reserve_ms),
            requested_reported_nodes: self.requested_reported_nodes,
            multipv: self.multipv,
        };
        actual
            .validate(256)
            .map_err(|_| invalid("external bounded slice"))?;
        Ok(actual)
    }
}

/// R-free configuration cannot deserialize an R plan, CPU ledger or controller.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "assembly", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExecutionConfiguration {
    RolesOnly {
        event_order: EventOrder,
        role_tickets_max: usize,
        shared_nnuec_readout: Option<PcNnuecReadoutConfiguration>,
    },
    OwnResearcher {
        event_order: EventOrder,
        queues: QueueConfiguration,
        max_rounds: u64,
        max_native_nodes: u64,
        cpu_depth: u16,
        cpu_nodes_per_task: u64,
        cpu_plan: CpuPlanConfiguration,
        value: ResearchValueConfiguration,
        controller: ControllerConfiguration,
        reciprocal: ReciprocalConfiguration,
        peers: PeerConfiguration,
    },
    ExternalResearcher {
        event_order: EventOrder,
        queues: QueueConfiguration,
        max_rounds: u64,
        max_reported_nodes: u64,
        cpu_depth: u16,
        cpu_nodes_per_task: u64,
        profile: ArtifactRef,
        slice: ExternalSliceConfiguration,
        peers: PeerConfiguration,
    },
}
impl ExecutionConfiguration {
    pub fn role_tickets_max(&self) -> usize {
        match self {
            Self::RolesOnly {
                role_tickets_max, ..
            } => *role_tickets_max,
            Self::OwnResearcher { queues, .. } | Self::ExternalResearcher { queues, .. } => {
                queues.role_tickets_max
            }
        }
    }
    pub fn event_order(&self) -> EventOrder {
        match self {
            Self::RolesOnly { event_order, .. }
            | Self::OwnResearcher { event_order, .. }
            | Self::ExternalResearcher { event_order, .. } => *event_order,
        }
    }
}

impl RoventStartupConfiguration {
    pub fn fields(&self) -> ConfigurationFields<'_> {
        ConfigurationFields {
            semantic: &self.semantic,
            review: &self.review,
            index: &self.index,
            frontier: &self.frontier,
            native: &self.native,
            execution: &self.execution,
        }
    }
    pub fn from_json(bytes: &[u8]) -> Result<Self, BackendError> {
        if bytes.is_empty() || bytes.len() > ROVENT_STARTUP_CONFIGURATION_MAX_BYTES {
            return Err(invalid("startup configuration byte budget"));
        }
        let value: Self = serde_json::from_slice(bytes)
            .map_err(|_| invalid("strict startup configuration JSON"))?;
        value.validate()?;
        Ok(value)
    }
    pub fn validate(&self) -> Result<(), BackendError> {
        if self.domain != ROVENT_STARTUP_CONFIGURATION_DOMAIN {
            return Err(invalid("startup configuration domain"));
        }
        self.fields().validate()
    }
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, BackendError> {
        self.validate()?;
        canonical(self)
    }
    pub fn configuration_digest(&self) -> Result<[u8; 32], BackendError> {
        framed_digest(
            ROVENT_STARTUP_CONFIGURATION_DOMAIN,
            &self.canonical_bytes()?,
        )
    }
    /// Identifies actual constructor values consumed by every mode. It does
    /// not certify that a declared mode, model or work item actually executed.
    pub fn search_configuration_digest(&self) -> Result<[u8; 32], BackendError> {
        self.validate()?;
        framed_digest(
            ROVENT_SEARCH_CONFIGURATION_DOMAIN,
            &canonical(&(
                &self.semantic,
                &self.review,
                &self.index,
                &self.frontier,
                &self.execution,
            ))?,
        )
    }
    /// Immutable NNUEC plan/policy only. Dynamic view/Bound seals stay with
    /// the actual preparation actor and must never be filled from this hash.
    pub fn nnuec_search_conditions_digest(&self) -> Result<Option<[u8; 32]>, BackendError> {
        self.validate()?;
        self.fields().nnuec_conditions_digest()
    }
    pub fn admit_against(&self, lock: &RoventInputLock) -> Result<(), BackendError> {
        self.validate()?;
        self.fields().admit_against(
            lock,
            || self.configuration_digest(),
            || self.search_configuration_digest(),
            || self.nnuec_search_conditions_digest(),
        )
    }
    /// First native product path is NNUEC-free P/C. A registered shared
    /// readout remains a valid R-free selection but needs its typed actor
    /// consumer; it is not reclassified as an R topology mismatch.
    pub fn require_supported_native_capabilities(&self) -> Result<(), BackendError> {
        self.validate()?;
        self.fields().require_supported_native_capabilities()
    }
}

/// Read-only common constructor declarations. This borrows metadata and
/// cannot issue root, worker, model or execution authority.
#[derive(Clone, Copy, Debug)]
pub struct ConfigurationFields<'a> {
    pub semantic: &'a SemanticConfiguration,
    pub review: &'a ReviewConfiguration,
    pub index: &'a IndexConfiguration,
    pub frontier: &'a FrontierConfiguration,
    pub native: &'a NativeConfiguration,
    pub execution: &'a ExecutionConfiguration,
}
impl ConfigurationFields<'_> {
    pub fn nnuec_selection(&self) -> NativeNnuecSelection {
        match self.execution {
            ExecutionConfiguration::RolesOnly {
                shared_nnuec_readout: Some(_),
                ..
            } => NativeNnuecSelection::SharedPc,
            ExecutionConfiguration::OwnResearcher {
                value: ResearchValueConfiguration::NnuecV2 { .. },
                ..
            } => NativeNnuecSelection::OwnResearcher,
            _ => NativeNnuecSelection::Absent,
        }
    }
    fn validate(&self) -> Result<(), BackendError> {
        self.semantic.actual()?;
        self.review.actual()?;
        self.index.actual()?;
        self.frontier.actual()?;
        self.native.tickets(self.review.token_budget)?;
        if self.frontier.maximum_goal_plies > self.semantic.maximum_root_prefix_plies
            || !(1..=16).contains(&self.execution.role_tickets_max())
        {
            return Err(invalid("shared context/role limits"));
        }
        match self.execution {
            ExecutionConfiguration::RolesOnly {
                shared_nnuec_readout,
                ..
            } => {
                if shared_nnuec_readout.as_ref().is_some_and(|readout| {
                    readout.context_namespace_policy.is_empty()
                        || readout.context_namespace_policy.len() > 256
                        || readout.local_namespace_policy.is_empty()
                        || readout.local_namespace_policy.len() > 256
                }) {
                    return Err(invalid("P/C shared NNUEC namespace policies"));
                }
            }
            ExecutionConfiguration::OwnResearcher {
                event_order,
                queues,
                max_rounds,
                max_native_nodes,
                cpu_depth,
                cpu_nodes_per_task,
                cpu_plan,
                value,
                controller,
                reciprocal,
                peers,
            } => {
                queues.actual()?;
                cpu_plan.actual()?;
                controller.actual(*event_order)?;
                peers.actual()?;
                reciprocal.actual(rz_contracts::rovent::ExecutionMode::CpuPcr, Digest([1; 32]))?;
                if *max_rounds == 0
                    || *max_native_nodes == 0
                    || *cpu_depth == 0
                    || *cpu_depth > cpu_plan.max_depth
                    || !(1..=10_000_000).contains(cpu_nodes_per_task)
                    || self.native.command_capacity != queues.worker_commands_max
                    || cpu_plan.paused_bytes_limit > queues.paused_bytes_per_worker_max
                {
                    return Err(invalid("own R constructor limits"));
                }
                if let ResearchValueConfiguration::NnuecV2 {
                    context_namespace_policy,
                    local_namespace_policy,
                } = value
                {
                    if context_namespace_policy.is_empty()
                        || context_namespace_policy.len() > 256
                        || local_namespace_policy.is_empty()
                        || local_namespace_policy.len() > 256
                    {
                        return Err(invalid("NNUEC namespace policies"));
                    }
                }
            }
            ExecutionConfiguration::ExternalResearcher {
                queues,
                max_rounds,
                max_reported_nodes,
                cpu_depth,
                cpu_nodes_per_task,
                profile,
                slice,
                peers,
                ..
            } => {
                queues.actual()?;
                profile
                    .validate()
                    .map_err(|_| invalid("external profile artifact"))?;
                slice.actual()?;
                peers.actual()?;
                if *max_rounds == 0
                    || *max_reported_nodes == 0
                    || *max_reported_nodes > 65_536
                    || !(1..=64).contains(cpu_depth)
                    || !(1..=10_000_000).contains(cpu_nodes_per_task)
                    || self.native.command_capacity != queues.worker_commands_max
                {
                    return Err(invalid("external R constructor limits"));
                }
            }
        }
        Ok(())
    }
    fn nnuec_conditions_digest(&self) -> Result<Option<[u8; 32]>, BackendError> {
        match self.execution {
            ExecutionConfiguration::RolesOnly {
                shared_nnuec_readout: Some(readout),
                ..
            } => framed_digest(
                "rz-rovent-pc-nnuec-readout-conditions/1",
                &canonical(readout)?,
            )
            .map(Some),
            ExecutionConfiguration::OwnResearcher {
                cpu_plan,
                value:
                    ResearchValueConfiguration::NnuecV2 {
                        context_namespace_policy,
                        local_namespace_policy,
                    },
                ..
            } => framed_digest(
                ROVENT_NNUEC_CONDITIONS_DOMAIN,
                &canonical(&(cpu_plan, context_namespace_policy, local_namespace_policy))?,
            )
            .map(Some),
            _ => Ok(None),
        }
    }
    fn admit_against(
        &self,
        lock: &RoventInputLock,
        configuration_digest: impl Fn() -> Result<[u8; 32], BackendError>,
        search_configuration_digest: impl Fn() -> Result<[u8; 32], BackendError>,
        nnuec_conditions_digest: impl Fn() -> Result<Option<[u8; 32]>, BackendError>,
    ) -> Result<(), BackendError> {
        lock.validate().map_err(|_| invalid("Rovent input lock"))?;
        let engine = &lock.manifest.engine;
        let declaration = &lock.manifest.coordinator;
        if declaration.policy_sha256 != hex(configuration_digest()?)
            || engine.search.search_configuration_sha256 != hex(search_configuration_digest()?)
            || engine.search.registered_max_chess_plies.get() != self.frontier.maximum_goal_plies
            || declaration.role_tickets_max as usize != self.execution.role_tickets_max()
            || self.frontier.maximum_frames > declaration.questions_max as usize
            || self.native.drain_ms > lock.manifest.cleanup_max_ms
        {
            return Err(invalid("locked startup configuration differs"));
        }
        self.native.onnx(
            &engine.pc.device,
            matches!(engine.pc.device, RoventDevice::Cuda { .. }),
        )?;
        match (self.execution, &engine.researcher) {
            (
                ExecutionConfiguration::RolesOnly {
                    shared_nnuec_readout,
                    ..
                },
                None,
            ) if !engine.mode.has_native_r() => {
                match (
                    shared_nnuec_readout,
                    &engine.nnuec,
                    nnuec_conditions_digest()?,
                ) {
                    (None, None, None) => {}
                    (Some(readout), Some(pin), Some(conditions))
                        if pin.search_conditions_sha256 == hex(conditions)
                            && pin.context_namespace_policy_sha256
                                == hex(framed_digest(
                                    "rz-rovent-nnuec-context-policy/1",
                                    &canonical(&readout.context_namespace_policy)?,
                                )?)
                            && pin.local_namespace_policy_sha256
                                == hex(framed_digest(
                                    "rz-rovent-nnuec-local-policy/1",
                                    &canonical(&readout.local_namespace_policy)?,
                                )?) => {}
                    _ => {
                        return Err(invalid(
                            "P/C readout configuration differs from pinned NNUEC",
                        ));
                    }
                }
            }
            (
                ExecutionConfiguration::OwnResearcher {
                    value,
                    queues,
                    max_native_nodes,
                    ..
                },
                Some(RoventResearcher::Own {
                    search_configuration_sha256,
                    ..
                }),
            ) if engine.mode.has_native_r() => {
                if *max_native_nodes != declaration.cpu_credit_limit
                    || queues.waiting_jobs_max != declaration.waiting_jobs_max as usize
                {
                    return Err(invalid("locked original CPU credit/queue limits differ"));
                }
                if *search_configuration_sha256 != hex(search_configuration_digest()?) {
                    return Err(invalid("locked own R configuration differs"));
                }
                match (value, &engine.nnuec, nnuec_conditions_digest()?) {
                    (ResearchValueConfiguration::BaselineBoard, None, None) => {}
                    (
                        ResearchValueConfiguration::NnuecV2 {
                            context_namespace_policy,
                            local_namespace_policy,
                        },
                        Some(pin),
                        Some(conditions),
                    ) if pin.search_conditions_sha256 == hex(conditions)
                        && pin.context_namespace_policy_sha256
                            == hex(framed_digest(
                                "rz-rovent-nnuec-context-policy/1",
                                &canonical(context_namespace_policy)?,
                            )?)
                        && pin.local_namespace_policy_sha256
                            == hex(framed_digest(
                                "rz-rovent-nnuec-local-policy/1",
                                &canonical(local_namespace_policy)?,
                            )?) => {}
                    _ => {
                        return Err(invalid(
                            "NNUEC value/conditions differ from actual selection",
                        ));
                    }
                }
            }
            (
                ExecutionConfiguration::ExternalResearcher {
                    profile, queues, ..
                },
                Some(RoventResearcher::External { profile: pin }),
            ) if engine.mode.has_native_r()
                && engine.nnuec.is_none()
                && *profile == pin.profile =>
            {
                if queues.waiting_jobs_max != declaration.waiting_jobs_max as usize {
                    return Err(invalid("locked original external queue limit differs"));
                }
            }
            _ => return Err(invalid("mode/researcher startup assembly differs")),
        }
        Ok(())
    }
    fn require_supported_native_capabilities(&self) -> Result<(), BackendError> {
        // SharedPc is assembled only by the typed loaded-owner constructor.
        // This declaration check cannot replace that consumer or its source join.
        self.validate()
    }
}

fn canonical(value: &impl Serialize) -> Result<Vec<u8>, BackendError> {
    fn ordered(value: Value) -> Value {
        match value {
            Value::Object(map) => Value::Object(
                map.into_iter()
                    .map(|(k, v)| (k, ordered(v)))
                    .collect::<std::collections::BTreeMap<_, _>>()
                    .into_iter()
                    .collect(),
            ),
            Value::Array(items) => Value::Array(items.into_iter().map(ordered).collect()),
            other => other,
        }
    }
    let value = serde_json::to_value(value).map_err(|_| invalid("startup serialization"))?;
    serde_json::to_vec(&ordered(value)).map_err(|_| invalid("startup canonical JSON"))
}
fn framed_digest(domain: &str, bytes: &[u8]) -> Result<[u8; 32], BackendError> {
    let mut hash = Sha256::new();
    hash.update(domain.as_bytes());
    hash.update([0]);
    hash.update(bytes);
    Ok(hash.finalize().into())
}
fn invalid(detail: &'static str) -> BackendError {
    fail(K::InvalidInput, detail)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pc() -> RoventStartupConfiguration {
        RoventStartupConfiguration {
            domain: ROVENT_STARTUP_CONFIGURATION_DOMAIN.into(),
            semantic: SemanticConfiguration {
                beam_width: 4,
                line_plies: 4,
                max_nodes: 4096,
                max_records: 64,
                max_role_calls: 4096,
                maximum_root_prefix_plies: 256,
            },
            review: ReviewConfiguration {
                anchor_mode: AnchorMode::AnchoredCompact,
                token_budget: 512,
                max_records: 64,
                max_anchors: 4,
            },
            index: IndexConfiguration {
                active_bundles: 256,
                records_per_bundle: 64,
            },
            frontier: FrontierConfiguration {
                maximum_goal_plies: 256,
                maximum_frames: 64,
                maximum_paths: 4096,
                maximum_mate_nodes: 4096,
                maximum_native_receipts: 64,
            },
            native: NativeConfiguration {
                cache_public_memory: true,
                device_public_memory: false,
                cudnn_conv_search: CuDnnSearch::LegacyImplicitExhaustive,
                loading_profile: None,
                cuda_arena_bytes: None,
                drain_ms: 30_000,
                prepared_host_bytes_per_ticket: 1 << 20,
                output_host_bytes_per_ticket: 1 << 20,
                command_capacity: 8,
            },
            execution: ExecutionConfiguration::RolesOnly {
                event_order: EventOrder::Fixed,
                role_tickets_max: 16,
                shared_nnuec_readout: None,
            },
        }
    }
    #[test]
    fn version_dispatch_preserves_original_v1_bytes_hashes_and_mode_policy() {
        let original = pc();
        let bytes = original.canonical_bytes().unwrap();
        let selected = StartupConfigurationDocument::from_json(&bytes).unwrap();
        assert!(matches!(selected, StartupConfigurationDocument::V1(_)));
        assert_eq!(selected.canonical_bytes().unwrap(), bytes);
        assert_eq!(
            selected.configuration_digest().unwrap(),
            original.configuration_digest().unwrap()
        );
        assert_eq!(
            selected.search_configuration_digest().unwrap(),
            original.search_configuration_digest().unwrap()
        );
        assert_eq!(
            selected.nnuec_conditions_digest().unwrap(),
            original.nnuec_search_conditions_digest().unwrap()
        );
        for mode in [ExecutionMode::CpuPc, ExecutionMode::HybridPc] {
            assert_eq!(
                selected.post_repair_policy_for(mode),
                PostRepairRecheckPolicy::Disabled
            );
        }
        for mode in [ExecutionMode::CpuPcr, ExecutionMode::HybridPcr] {
            assert_eq!(
                selected.post_repair_policy_for(mode),
                PostRepairRecheckPolicy::IterativeFrozenModelWdlV2
            );
        }
        let wire: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(wire.get("post_repair").is_none());
    }

    #[test]
    fn version_dispatch_selects_exact_decoder_without_policy_default_or_fallback() {
        let legacy = serde_json::to_value(pc()).unwrap();
        // Explicit /2 test input; no production conversion API exists.
        for (policy, expected) in [
            ("disabled", PostRepairRecheckPolicy::Disabled),
            (
                "iterative_frozen_model_wdl_v2",
                PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
            ),
        ] {
            let mut wire = legacy.clone();
            wire["domain"] = serde_json::json!(v2::ROVENT_STARTUP_CONFIGURATION_V2_DOMAIN);
            wire["post_repair"] = serde_json::json!(policy);
            let bytes = serde_json::to_vec(&wire).unwrap();
            let selected = StartupConfigurationDocument::from_json(&bytes).unwrap();
            assert!(matches!(selected, StartupConfigurationDocument::V2(_)));
            for mode in [
                ExecutionMode::CpuPc,
                ExecutionMode::HybridPc,
                ExecutionMode::CpuPcr,
                ExecutionMode::HybridPcr,
            ] {
                assert_eq!(selected.post_repair_policy_for(mode), expected);
            }
            assert!(RoventStartupConfiguration::from_json(&bytes).is_err());
        }
        let mut missing = legacy.clone();
        missing["domain"] = serde_json::json!(v2::ROVENT_STARTUP_CONFIGURATION_V2_DOMAIN);
        assert!(
            StartupConfigurationDocument::from_json(&serde_json::to_vec(&missing).unwrap())
                .is_err()
        );
        let mut wrong_version_shape = legacy.clone();
        wrong_version_shape["post_repair"] = serde_json::json!("disabled");
        assert!(
            StartupConfigurationDocument::from_json(
                &serde_json::to_vec(&wrong_version_shape).unwrap()
            )
            .is_err()
        );
        let mut unknown = legacy;
        unknown["domain"] = serde_json::json!("rz-rovent-startup-configuration/3");
        assert!(
            StartupConfigurationDocument::from_json(&serde_json::to_vec(&unknown).unwrap())
                .is_err()
        );
    }
    #[test]
    fn strict_pc_decoder_rejects_r_fields_unknown_fields_and_legacy_domains() {
        let original = pc();
        let bytes = original.canonical_bytes().unwrap();
        assert_eq!(
            RoventStartupConfiguration::from_json(&bytes).unwrap(),
            original
        );
        let mut value = serde_json::to_value(&original).unwrap();
        value["execution"]["cpu_plan"] = serde_json::json!({"tt_entries":8192});
        assert!(
            RoventStartupConfiguration::from_json(&serde_json::to_vec(&value).unwrap()).is_err()
        );
        value = serde_json::to_value(&original).unwrap();
        value["semantic"]["unregistered_limit"] = Value::from(1);
        assert!(
            RoventStartupConfiguration::from_json(&serde_json::to_vec(&value).unwrap()).is_err()
        );
        value = serde_json::to_value(&original).unwrap();
        value["domain"] = Value::from("rz-pals-adaptive-configuration-v5/1");
        assert!(
            RoventStartupConfiguration::from_json(&serde_json::to_vec(&value).unwrap()).is_err()
        );
    }
    #[test]
    fn startup_config_keeps_generation_chunks_separate_from_root_prefix_capacity() {
        let mut config = pc();
        assert_eq!(
            config.semantic.actual().unwrap().maximum_root_prefix_plies,
            256
        );
        config.semantic.line_plies = 17;
        assert!(config.validate().is_err());
        config.semantic.line_plies = 4;
        config.semantic.maximum_root_prefix_plies = 257;
        assert!(config.validate().is_err());
        config.semantic.maximum_root_prefix_plies = 128;
        assert!(config.validate().is_err()); // registered goal cap remains 256
    }
    #[test]
    fn canonical_json_sorts_nested_keys_and_preserves_array_order() {
        let a = serde_json::json!({"z":{"b":1,"a":2},"a":[{"z":3,"a":4},5]});
        assert_eq!(
            String::from_utf8(canonical(&a).unwrap()).unwrap(),
            r#"{"a":[{"a":4,"z":3},5],"z":{"a":2,"b":1}}"#
        );
        let mut bytes = pc().canonical_bytes().unwrap();
        bytes.extend_from_slice(b" {}");
        assert!(RoventStartupConfiguration::from_json(&bytes).is_err());
    }
    #[test]
    fn separate_hashes_track_consumed_search_values_without_inventing_nnuec() {
        let mut config = pc();
        let full = config.configuration_digest().unwrap();
        let search = config.search_configuration_digest().unwrap();
        assert_eq!(config.nnuec_search_conditions_digest().unwrap(), None);
        config.native.output_host_bytes_per_ticket += 1;
        assert_ne!(config.configuration_digest().unwrap(), full);
        assert_eq!(config.search_configuration_digest().unwrap(), search);
        config.semantic.beam_width = 3;
        assert_ne!(config.search_configuration_digest().unwrap(), search);
        let mut bad = pc();
        bad.native.command_capacity = 0;
        assert!(bad.validate().is_err());
        assert!(
            RoventStartupConfiguration::from_json(&vec![
                b' ';
                ROVENT_STARTUP_CONFIGURATION_MAX_BYTES + 1
            ])
            .is_err()
        );
    }

    #[test]
    fn pc_shared_readout_is_a_separate_typed_capability_without_r_configuration() {
        let mut config = pc();
        assert_eq!(
            config.fields().nnuec_selection(),
            NativeNnuecSelection::Absent
        );
        config.execution = ExecutionConfiguration::RolesOnly {
            event_order: EventOrder::Fixed,
            role_tickets_max: 16,
            shared_nnuec_readout: Some(PcNnuecReadoutConfiguration {
                context_namespace_policy: "actual-context-policy-v2".into(),
                local_namespace_policy: "actual-local-policy-v2".into(),
            }),
        };
        assert!(config.validate().is_ok());
        assert!(config.nnuec_search_conditions_digest().unwrap().is_some());
        let bytes = config.canonical_bytes().unwrap();
        let decoded = RoventStartupConfiguration::from_json(&bytes).unwrap();
        assert_eq!(decoded, config);
        assert_eq!(
            decoded.fields().nnuec_selection(),
            NativeNnuecSelection::SharedPc
        );
        assert!(decoded.require_supported_native_capabilities().is_ok());
        let mut invalid = decoded;
        if let ExecutionConfiguration::RolesOnly {
            shared_nnuec_readout: Some(readout),
            ..
        } = &mut invalid.execution
        {
            readout.context_namespace_policy.clear();
        }
        assert!(invalid.require_supported_native_capabilities().is_err());
    }
}
