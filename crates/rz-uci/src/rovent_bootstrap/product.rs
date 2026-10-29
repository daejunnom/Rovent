//! Concrete native product assembly. These constructors reuse the actual
//! selected-view ticket service and private CPU engines; they never substitute
//! the synchronous RoleModel compatibility driver for native inference.

use super::configuration::{ExecutionConfiguration, ResearchValueConfiguration};
use super::external_session::{
    RoventExternalProductStartupRejection, RoventExternalProductStartupRejectionV2,
    external_researcher_session, external_researcher_session_v2,
};
use super::launch::{LaunchRetirementEvidence, LaunchRetirementRegistrationCustody};
use super::nnuec_session::{
    RoventNnuecCollectionHooks, RoventNnuecSessionDriver, RoventNnuecSessionRejection,
    RoventNnuecSessionV2Rejection,
};
use super::nnuec_source::LoadedRoventNnuecSource;
use super::pc_session::{RoventPcCollectionHooks, RoventPcSessionDriver};
use super::process_boundary::VerifiedOwnedRoventProcessBoundaryV2;
use super::{
    PreparedRoventNative, PreparedRoventNativeV2, RoventNativeBackend, RoventPinnedExternalRSource,
    RoventPinnedNativeSources,
};
use crate::EngineIdentity;
use crate::engine::{OwnerRegistry, ProcessClock};
use crate::pals_adaptive::{
    AdaptiveCheckerMetadata, AdaptiveSessionAdmissionRejection, AdaptiveSessionCollectionHooks,
    AdaptiveSessionDriver, AdaptiveSessionOptions, AdaptiveSessionSources, PersistentCpuOwner,
    RoventPcrSessionProfile,
};
use crate::rovent_root_admission::RoventRootAdmissionPolicy;
use crate::search_driver::{
    DriverPhysicalCompletion, PalsCheckerRegistration, SearchKind, SearchSessionContext,
    SearchSessionDriver, SearchSessionFailure, SearchSessionReport,
};
use rz_contracts::pals_adaptive::CpuWorkerId;
use rz_contracts::{ContractError, Digest};
use rz_position::{BoardMove, Position};
use rz_runtime::contracts::ContractClock;
use rz_search::cpu::{CpuEngine, CpuResumePolicy};
use rz_search::cpu_value::BootstrapCpuValue;
use rz_search::cpu_worker::{
    CpuWorkerProfile, cpu_evaluation_namespace_digest, cpu_search_conditions_digest,
};
use rz_search::external_cpu::{CpuEndpointSelection, ExternalSlicePolicy};
use rz_search::pals::engine::adaptive_port::{
    AdaptivePortModel, AdaptivePortQuestion, AdaptiveRulesOwnerIssuer,
};
use rz_search::pals::engine::pc_port::PcPortQuestion;
use rz_search::pals::engine::{PalsConfig, PalsEngine, ResearcherSemanticProfile, ResolverPolicy};
use sha2::{Digest as _, Sha256};
use std::{
    num::NonZeroU16,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

// Shared assembly over the original pinned V1/V2 declarations. The session
// constructors below still own their version-specific admission and custody.
macro_rules! build_own_board_pcr {
    ($pins:expr, $prepared:expr, $clock:expr, $identity:expr) => {{
        let pins = &$pins;
        let prepared = &$prepared;
        let clock = &$clock;
        let identity = $identity;
        (|| -> Result<_, SearchSessionFailure> {
            let configuration = pins.configuration();
            let ExecutionConfiguration::OwnResearcher {
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
            } = configuration.fields().execution
            else {
                return Err(failure(
                    "RoventPcrStartup",
                    "an actual own-R plan is required",
                ));
            };
            if !matches!(value, ResearchValueConfiguration::BaselineBoard)
                || pins.lock().manifest.engine.nnuec.is_some()
                || !prepared.snapshot().mode.has_native_r()
                || prepared.snapshot().process_epoch != clock.epoch().0
            {
                return Err(failure(
                    "RoventPcrValue",
                    "baseline board and NNUEC sources are distinct",
                ));
            }
            if cpu_plan.affinity.is_some() {
                return Err(failure(
                    "RoventPcrAffinityUnsupported",
                    "this engine-retaining session does not apply a per-worker affinity request",
                ));
            }
            let semantic = configuration.fields().semantic;
            let namespace = prepared
                .snapshot()
                .model_namespace()
                .map_err(|error| SearchSessionFailure::debug("RoventPcrNamespace", &error))?;
            let model = AdaptivePortModel::new_with_namespace(
                prepared.snapshot().role_model_identity(),
                Some(
                    prepared
                        .snapshot()
                        .model_value_identity()
                        .map_err(|error| SearchSessionFailure::debug("RoventPcrValue", &error))?,
                ),
                namespace.clone(),
            )
            .map_err(|error| SearchSessionFailure::debug("RoventPcrModel", &error))?;
            let control = controller
                .actual(*event_order)
                .map_err(|error| SearchSessionFailure::debug("RoventPcrController", &error))?;
            let root_tie_policy = configuration
                .own_r_root_tie_policy()
                .map_err(|error| SearchSessionFailure::debug("RoventPcrTiePolicy", &error))?
                .ok_or_else(|| failure("RoventPcrTiePolicy", "own-R policy is required"))?;
            let mut cpu = Vec::new();
            cpu.try_reserve_exact(usize::from(control.max_workers))
                .map_err(|error| SearchSessionFailure::debug("RoventPcrCpuCapacity", &error))?;
            for slot in 0..u16::from(control.max_workers) {
                let evaluator = Arc::new(BootstrapCpuValue::default());
                let actual = cpu_plan
                    .actual()
                    .map_err(|error| SearchSessionFailure::debug("RoventPcrCpuPlan", &error))?;
                let mut engine = if cpu_plan.preserve_paused_side_scope {
                    CpuEngine::with_paused_side_scope_and_tie_policy(
                        actual,
                        evaluator,
                        cpu_plan.ordering(),
                        root_tie_policy,
                    )
                } else {
                    CpuEngine::with_evaluator_ordering_resume_and_tie_policy(
                        actual,
                        evaluator,
                        cpu_plan.ordering(),
                        CpuResumePolicy::PausedStack,
                        root_tie_policy,
                    )
                }
                .map_err(|error| SearchSessionFailure::debug("RoventPcrCpu", &error))?;
                engine
                    .set_paused_stack_max_bytes(cpu_plan.paused_bytes_limit)
                    .map_err(|error| {
                        SearchSessionFailure::debug("RoventPcrPausedCapacity", &error)
                    })?;
                let profile = CpuWorkerProfile {
                    owner: CpuWorkerId(slot + 1),
                    max_depth: engine.config().max_depth,
                    evaluation_namespace: cpu_evaluation_namespace_digest(&engine).map_err(
                        |error| SearchSessionFailure::debug("RoventPcrCpuNamespace", &error),
                    )?,
                    search_conditions: cpu_search_conditions_digest(&engine),
                };
                cpu.push(PersistentCpuOwner { engine, profile });
            }
            let primary = cpu.first().ok_or_else(|| {
                failure("RoventPcrCpu", "controller has no actual researcher owner")
            })?;
            let metadata = AdaptiveCheckerMetadata::capture_owned(&primary.engine)
                .map_err(|error| SearchSessionFailure::debug("RoventPcrMetadata", &error))?;
            let mirror = metadata.clone().into_mirror();
            let resolver = ResolverPolicy::ModelWdlRestricted;
            let recheck = configuration.post_repair_policy_for(prepared.snapshot().mode);
            let registration =
                PalsCheckerRegistration::capture_with_resolver(&mirror, &model, resolver)?;
            let engine = PalsEngine::new_with_boxed_checker_and_rovent_profile(
                ResearcherSemanticProfile {
                    config: PalsConfig {
                        beam_width: semantic.beam_width,
                        line_plies: semantic.line_plies,
                        max_nodes: semantic.max_nodes,
                        max_records: semantic.max_records,
                        max_role_calls: semantic.max_role_calls,
                        cpu_nodes_per_task: *cpu_nodes_per_task,
                    },
                    maximum_root_prefix_plies: semantic.maximum_root_prefix_plies,
                },
                model,
                Box::new(mirror),
                resolver,
                recheck,
            )
            .map_err(|error| SearchSessionFailure::debug("RoventPcrSemantic", &error))?
            .with_selected_review_context(
                configuration
                    .fields()
                    .review
                    .actual()
                    .map_err(|error| SearchSessionFailure::debug("RoventPcrReview", &error))?,
                configuration
                    .fields()
                    .index
                    .actual()
                    .map_err(|error| SearchSessionFailure::debug("RoventPcrIndex", &error))?,
            )
            .map_err(|error| SearchSessionFailure::debug("RoventPcrReview", &error))?;
            if engine.post_repair_recheck_policy() != recheck {
                return Err(failure(
                    "RoventPcrRecheckPolicy",
                    "actual engine policy differs from its registered configuration",
                ));
            }
            let root_budget = Digest(
                rz_eval::asset::parse_sha256(
                    &pins.lock().manifest.coordinator.original_root_budget_sha256,
                )
                .map_err(|error| SearchSessionFailure::debug("RoventPcrBudget", &error))?,
            );
            let mut hash = Sha256::new();
            hash.update(b"rz-rovent-native-product/1\0");
            hash.update(include_bytes!("product.rs"));
            hash.update(super::compiled_bootstrap_source_digest());
            hash.update(
                configuration.configuration_digest().map_err(|error| {
                    SearchSessionFailure::debug("RoventPcrConfiguration", &error)
                })?,
            );
            let options = AdaptiveSessionOptions {
                implementation: Digest(hash.finalize().into()),
                identity,
                max_rounds: *max_rounds,
                max_cpu_nodes: *max_native_nodes,
                cpu_depth: *cpu_depth,
                token_budget: configuration.fields().review.token_budget,
                policy: event_order.actual(),
                endpoints: CpuEndpointSelection::deployment(),
                controller: control,
                device_digest: prepared.snapshot().backend,
                cpu_device_digest: Digest(Sha256::digest(b"native-own-board-host-cpu").into()),
                external_policy: ExternalSlicePolicy::default(),
            };
            let profile = RoventPcrSessionProfile {
                mode: prepared.snapshot().mode,
                maximum_goal_plies: NonZeroU16::new(
                    configuration.fields().frontier.maximum_goal_plies,
                )
                .ok_or_else(|| failure("RoventPcrGoal", "zero goal capability"))?,
                input_lock: prepared.snapshot().input_lock,
                root_budget,
                selective_policy: Digest(configuration.search_configuration_digest().map_err(
                    |error| SearchSessionFailure::debug("RoventPcrConfiguration", &error),
                )?),
                model_namespace: Arc::new(namespace),
                frontier_limits: configuration
                    .fields()
                    .frontier
                    .actual()
                    .map_err(|error| SearchSessionFailure::debug("RoventPcrFrontier", &error))?,
                reciprocal: reciprocal
                    .actual(prepared.snapshot().mode, root_budget)
                    .map_err(|error| SearchSessionFailure::debug("RoventPcrReciprocal", &error))?,
                peers: peers
                    .actual()
                    .map_err(|error| SearchSessionFailure::debug("RoventPcrPeers", &error))?,
            };
            Ok((
                engine,
                cpu,
                metadata,
                registration,
                options,
                queues
                    .actual()
                    .map_err(|error| SearchSessionFailure::debug("RoventPcrQueues", &error))?,
                profile,
            ))
        })()
    }};
}

type PcrDriver = AdaptiveSessionDriver<RoventNativeBackend<AdaptivePortQuestion>>;
mod diagnostics;
pub(super) mod diagnostics_v2;
mod storage;
pub use diagnostics::{
    ROVENT_LIFECYCLE_DIAGNOSTIC_DOMAIN, RoventLifecycleDiagnostics,
    RoventProductLifecycleObservation,
};
pub use storage::RoventProductStorageStartupRejection;
pub(crate) use storage::SharedRoventStorageRegistration;
// A product's single Arc owns exactly one selected driver for the whole
// session, including physical shutdown and retained-source custody.
#[allow(clippy::large_enum_variant)]
enum ActualProductDriver {
    Pc(RoventPcSessionDriver),
    PcNnuec(RoventPcSessionDriver),
    Pcr(PcrDriver),
    Nnuec(RoventNnuecSessionDriver),
}

/// Keeps the same original retained descriptors until actual whole-session
/// shutdown. Unknown drop quarantines both physical owners and their sources.
pub struct RoventNativeProductDriver {
    inner: Arc<ActualProductDriver>,
    sources: Arc<RoventPinnedNativeSources>,
    process_clock: ProcessClock,
    physically_closed: Arc<AtomicBool>,
    retirement_sources: Option<Arc<LaunchRetirementRegistrationCustody>>,
    lifecycle_diagnostics: Option<RoventLifecycleDiagnostics>,
    storage_registration: Option<storage::RetainedStorageRegistration>,
}

/// Read-only observation of the same product's existing finish boundary.
/// Only finish_session's actual Confirmed result can set the private state.
pub(crate) struct RoventProductPhysicalClose {
    closed: Arc<AtomicBool>,
}
impl RoventProductPhysicalClose {
    pub(crate) fn is_confirmed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }
}
// One startup rejection retains the actual prepared/rejected owners and source
// pins. Variant size must not change their existing transfer or drop boundary.
#[allow(clippy::large_enum_variant)]
pub enum RoventProductStartupOwners {
    Pc(PreparedRoventNative<PcPortQuestion>),
    PcNnuec {
        prepared: PreparedRoventNative<PcPortQuestion>,
        loaded: LoadedRoventNnuecSource,
    },
    Pcr(PreparedRoventNative<AdaptivePortQuestion>),
    PcrRejected(AdaptiveSessionAdmissionRejection<RoventNativeBackend<AdaptivePortQuestion>>),
    PcrDriver(Box<PcrDriver>),
    NnuecRejected(Box<RoventNnuecSessionRejection>),
    ExternalRejected {
        rejected: Box<RoventExternalProductStartupRejection>,
        registration: Arc<LaunchRetirementRegistrationCustody>,
    },
    ExternalEvidenceMismatch {
        prepared: PreparedRoventNative<AdaptivePortQuestion>,
        external: RoventPinnedExternalRSource,
        evidence: LaunchRetirementEvidence,
    },
}
pub struct RoventProductStartupRejection {
    pub failure: SearchSessionFailure,
    pub owners: RoventProductStartupOwners,
    /// Same pins even when the rejected physical factory is retained.
    pub sources: Arc<RoventPinnedNativeSources>,
}

impl RoventNativeProductDriver {
    /// The external source stays dormant until an actual live root admission.
    /// It never creates a native CPU checker or substitutes an own-R engine.
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    pub fn external_pcr(
        prepared: PreparedRoventNative<AdaptivePortQuestion>,
        external: RoventPinnedExternalRSource,
        retirement_evidence: LaunchRetirementEvidence,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
        hooks: Option<AdaptiveSessionCollectionHooks<RoventNativeBackend<AdaptivePortQuestion>>>,
    ) -> Result<Self, RoventProductStartupRejection> {
        let sources = Arc::clone(prepared.sources());
        let process_clock = clock.clone();
        if retirement_evidence.input_lock_digest() != sources.lock_digest() {
            return Err(RoventProductStartupRejection {
                failure: failure(
                    "RoventRetirementSource",
                    "retirement registration source lock differs",
                ),
                owners: RoventProductStartupOwners::ExternalEvidenceMismatch {
                    prepared,
                    external,
                    evidence: retirement_evidence,
                },
                sources,
            });
        }
        let (retirement_sink, registration) = retirement_evidence.into_parts();
        let driver = external_researcher_session(
            prepared,
            external,
            retirement_sink,
            clock,
            owners,
            identity,
            root_policy,
            hooks,
        )
        .map_err(|rejected| RoventProductStartupRejection {
            failure: rejected.failure.clone(),
            owners: RoventProductStartupOwners::ExternalRejected {
                rejected: Box::new(rejected),
                registration: Arc::clone(&registration),
            },
            sources: Arc::clone(&sources),
        })?;
        Ok(Self {
            inner: Arc::new(ActualProductDriver::Pcr(driver)),
            sources,
            physically_closed: Arc::new(AtomicBool::new(false)),
            retirement_sources: Some(registration),
            lifecycle_diagnostics: None,
            process_clock,
            storage_registration: None,
        })
    }

    #[allow(clippy::result_large_err)]
    pub fn own_nnuec_pcr(
        prepared: PreparedRoventNative<AdaptivePortQuestion>,
        loaded: LoadedRoventNnuecSource,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
    ) -> Result<Self, RoventProductStartupRejection> {
        let sources = Arc::clone(prepared.sources());
        let process_clock = clock.clone();
        let driver =
            RoventNnuecSessionDriver::new(prepared, loaded, clock, owners, identity, root_policy)
                .map_err(|rejected| RoventProductStartupRejection {
                failure: rejected.failure.clone(),
                owners: RoventProductStartupOwners::NnuecRejected(Box::new(rejected)),
                sources: Arc::clone(&sources),
            })?;
        Ok(Self {
            inner: Arc::new(ActualProductDriver::Nnuec(driver)),
            sources,
            physically_closed: Arc::new(AtomicBool::new(false)),
            retirement_sources: None,
            lifecycle_diagnostics: None,
            process_clock,
            storage_registration: None,
        })
    }
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    pub fn own_nnuec_pcr_with_collection_hooks(
        prepared: PreparedRoventNative<AdaptivePortQuestion>,
        loaded: LoadedRoventNnuecSource,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
        hooks: RoventNnuecCollectionHooks,
    ) -> Result<Self, RoventProductStartupRejection> {
        let sources = Arc::clone(prepared.sources());
        let process_clock = clock.clone();
        let driver = RoventNnuecSessionDriver::new_with_collection_hooks(
            prepared,
            loaded,
            clock,
            owners,
            identity,
            root_policy,
            Some(hooks),
        )
        .map_err(|rejected| RoventProductStartupRejection {
            failure: rejected.failure.clone(),
            owners: RoventProductStartupOwners::NnuecRejected(Box::new(rejected)),
            sources: Arc::clone(&sources),
        })?;
        Ok(Self {
            inner: Arc::new(ActualProductDriver::Nnuec(driver)),
            sources,
            physically_closed: Arc::new(AtomicBool::new(false)),
            retirement_sources: None,
            lifecycle_diagnostics: None,
            process_clock,
            storage_registration: None,
        })
    }

    #[allow(clippy::result_large_err)]
    pub fn pc(
        prepared: PreparedRoventNative<PcPortQuestion>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
    ) -> Result<Self, RoventProductStartupRejection> {
        let sources = Arc::clone(prepared.sources());
        let process_clock = clock.clone();
        let driver = RoventPcSessionDriver::new(prepared, clock, owners, identity, root_policy)
            .map_err(|rejected| RoventProductStartupRejection {
                failure: rejected.failure,
                owners: RoventProductStartupOwners::Pc(rejected.prepared),
                sources: Arc::clone(&sources),
            })?;
        Ok(Self {
            inner: Arc::new(ActualProductDriver::Pc(driver)),
            sources,
            physically_closed: Arc::new(AtomicBool::new(false)),
            retirement_sources: None,
            lifecycle_diagnostics: None,
            process_clock,
            storage_registration: None,
        })
    }

    #[allow(clippy::result_large_err)]
    pub fn pc_with_collection_hooks(
        prepared: PreparedRoventNative<PcPortQuestion>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
        hooks: RoventPcCollectionHooks,
    ) -> Result<Self, RoventProductStartupRejection> {
        let sources = Arc::clone(prepared.sources());
        let process_clock = clock.clone();
        let driver = RoventPcSessionDriver::new_with_collection_hooks(
            prepared,
            clock,
            owners,
            identity,
            root_policy,
            hooks,
        )
        .map_err(|rejected| RoventProductStartupRejection {
            failure: rejected.failure,
            owners: RoventProductStartupOwners::Pc(rejected.prepared),
            sources: Arc::clone(&sources),
        })?;
        Ok(Self {
            inner: Arc::new(ActualProductDriver::Pc(driver)),
            sources,
            physically_closed: Arc::new(AtomicBool::new(false)),
            retirement_sources: None,
            lifecycle_diagnostics: None,
            process_clock,
            storage_registration: None,
        })
    }

    /// Shared readout keeps the original loaded NNUEC source in the PC-only
    /// driver. It does not construct an R checker, CPU ledger or research TT.
    #[allow(clippy::result_large_err)]
    pub fn pc_with_shared_nnuec(
        prepared: PreparedRoventNative<PcPortQuestion>,
        loaded: LoadedRoventNnuecSource,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
    ) -> Result<Self, RoventProductStartupRejection> {
        let sources = Arc::clone(prepared.sources());
        let process_clock = clock.clone();
        let driver = RoventPcSessionDriver::from_prepared_with_shared_nnuec(
            prepared,
            loaded,
            clock,
            owners,
            identity,
            root_policy,
        )
        .map_err(|rejected| RoventProductStartupRejection {
            failure: rejected.failure,
            owners: RoventProductStartupOwners::PcNnuec {
                prepared: rejected.prepared,
                loaded: rejected.loaded,
            },
            sources: Arc::clone(&sources),
        })?;
        Ok(Self {
            inner: Arc::new(ActualProductDriver::PcNnuec(driver)),
            sources,
            physically_closed: Arc::new(AtomicBool::new(false)),
            retirement_sources: None,
            lifecycle_diagnostics: None,
            process_clock,
            storage_registration: None,
        })
    }

    #[allow(clippy::result_large_err)]
    pub fn pc_with_shared_nnuec_and_collection_hooks(
        prepared: PreparedRoventNative<PcPortQuestion>,
        loaded: LoadedRoventNnuecSource,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
        hooks: RoventPcCollectionHooks,
    ) -> Result<Self, RoventProductStartupRejection> {
        let sources = Arc::clone(prepared.sources());
        let process_clock = clock.clone();
        let driver = RoventPcSessionDriver::from_prepared_with_shared_nnuec_and_collection_hooks(
            prepared,
            loaded,
            clock,
            owners,
            identity,
            root_policy,
            hooks,
        )
        .map_err(|rejected| RoventProductStartupRejection {
            failure: rejected.failure,
            owners: RoventProductStartupOwners::PcNnuec {
                prepared: rejected.prepared,
                loaded: rejected.loaded,
            },
            sources: Arc::clone(&sources),
        })?;
        Ok(Self {
            inner: Arc::new(ActualProductDriver::PcNnuec(driver)),
            sources,
            physically_closed: Arc::new(AtomicBool::new(false)),
            retirement_sources: None,
            lifecycle_diagnostics: None,
            process_clock,
            storage_registration: None,
        })
    }

    /// Actual existing material/PST integer board evaluator. This is a distinct
    /// M01 baseline, never a placeholder described as NNUEC.
    #[allow(clippy::result_large_err)]
    pub fn own_board_pcr(
        prepared: PreparedRoventNative<AdaptivePortQuestion>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
    ) -> Result<Self, RoventProductStartupRejection> {
        Self::admit_own_board_pcr(prepared, clock, owners, identity, root_policy, None)
    }

    /// Collection preserves the actual loaded origin and reuses the same
    /// exclusive root owner. Upper callers retain independent teacher owners.
    #[allow(clippy::result_large_err)]
    pub fn own_board_pcr_with_collection_hooks(
        prepared: PreparedRoventNative<AdaptivePortQuestion>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
        hooks: AdaptiveSessionCollectionHooks<RoventNativeBackend<AdaptivePortQuestion>>,
    ) -> Result<Self, RoventProductStartupRejection> {
        Self::admit_own_board_pcr(prepared, clock, owners, identity, root_policy, Some(hooks))
    }

    #[allow(clippy::result_large_err)]
    fn admit_own_board_pcr(
        prepared: PreparedRoventNative<AdaptivePortQuestion>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
        collection_hooks: Option<
            AdaptiveSessionCollectionHooks<RoventNativeBackend<AdaptivePortQuestion>>,
        >,
    ) -> Result<Self, RoventProductStartupRejection> {
        let pins = Arc::clone(prepared.sources());
        let process_clock = clock.clone();
        let collection = match (pins.lock().manifest.purpose, collection_hooks) {
            (rz_experiments::RoventPurpose::Collection, Some(hooks))
                if hooks.root_admission.is_some() =>
            {
                match prepared.collection_identity() {
                    Ok(origin) => Some((origin, hooks)),
                    Err(error) => {
                        return Err(RoventProductStartupRejection {
                            failure: SearchSessionFailure::debug("RoventCollectionOrigin", &error),
                            owners: RoventProductStartupOwners::Pcr(prepared),
                            sources: pins,
                        });
                    }
                }
            }
            (rz_experiments::RoventPurpose::Collection, _) | (_, Some(_)) => {
                return Err(RoventProductStartupRejection {
                    failure: failure(
                        "RoventCollectionHooks",
                        "collection purpose and actual root observer must be selected together",
                    ),
                    owners: RoventProductStartupOwners::Pcr(prepared),
                    sources: pins,
                });
            }
            (_, None) => None,
        };

        let build = build_own_board_pcr!(pins, prepared, clock, identity);
        let (engine, cpu, metadata, registration, options, limits, profile) = match build {
            Ok(built) => built,
            Err(failure) => {
                return Err(RoventProductStartupRejection {
                    failure,
                    owners: RoventProductStartupOwners::Pcr(prepared),
                    sources: pins,
                });
            }
        };
        let (native, _, _) = prepared.into_parts();
        let sources = AdaptiveSessionSources {
            native,
            engine,
            cpu,
            external: None,
            checker_metadata: metadata,
        };
        let issuer: AdaptiveRulesOwnerIssuer = Arc::new(move || owners.allocate());
        let mut driver = match collection {
            Some((origin, hooks)) => {
                AdaptiveSessionDriver::new_rovent_pcr_with_resource_limits_and_collection_hooks(
                    options,
                    limits,
                    sources,
                    clock,
                    issuer,
                    registration,
                    profile,
                    origin,
                    hooks,
                )
            }
            None => AdaptiveSessionDriver::new_rovent_pcr_with_resource_limits(
                options,
                limits,
                sources,
                clock,
                issuer,
                registration,
                profile,
            ),
        }
        .map_err(|rejected| RoventProductStartupRejection {
            failure: rejected.failure.clone(),
            owners: RoventProductStartupOwners::PcrRejected(rejected),
            sources: Arc::clone(&pins),
        })?;
        if let Err(failure) =
            driver.register_rovent_root_policy(Arc::clone(pins.lock_owner()), root_policy)
        {
            return Err(RoventProductStartupRejection {
                failure,
                owners: RoventProductStartupOwners::PcrDriver(Box::new(driver)),
                sources: pins,
            });
        }
        Ok(Self {
            inner: Arc::new(ActualProductDriver::Pcr(driver)),
            sources: pins,
            physically_closed: Arc::new(AtomicBool::new(false)),
            retirement_sources: None,
            lifecycle_diagnostics: None,
            process_clock,
            storage_registration: None,
        })
    }
    pub(crate) fn physical_close_observation(&self) -> RoventProductPhysicalClose {
        RoventProductPhysicalClose {
            closed: Arc::clone(&self.physically_closed),
        }
    }

    /// Opt-in operational JSONL observations only. The handle cannot close or
    /// recreate any owner. Diagnostic I/O never replaces the physical cleanup.
    pub fn enable_lifecycle_stderr_diagnostics(&mut self) -> RoventLifecycleDiagnostics {
        let first_enable = self.lifecycle_diagnostics.is_none();
        let implementation = self.actual().implementation();
        let observation = self
            .lifecycle_diagnostics
            .get_or_insert_with(|| {
                RoventLifecycleDiagnostics::new_from_sources(
                    self.sources.lock_digest(),
                    implementation,
                    self.sources.lock().manifest.engine.mode,
                    Arc::clone(&self.physically_closed),
                )
            })
            .clone();
        #[cfg(feature = "pals-adaptive-diagnostics")]
        if first_enable && let ActualProductDriver::Nnuec(driver) = self.inner.as_ref() {
            driver.enable_nnuec_diagnostics();
        }
        // Existing explicit opt-in only, and only the externally profiled R
        // assembly. The diagnostic never constructs R/native work or a permit.
        #[cfg(feature = "pals-adaptive-diagnostics")]
        if first_enable
            && matches!(
                self.sources.lock().manifest.engine.researcher.as_ref(),
                Some(rz_experiments::RoventResearcher::External { .. })
            )
        {
            let installed = match self.inner.as_ref() {
                ActualProductDriver::Pcr(driver) => {
                    crate::pals_adaptive::ContinuationDiagnostic::new(
                        crate::pals_adaptive::ContinuationDiagnosticOptions {
                            pending_snapshots_max: 0,
                            resolution_records_max:
                                crate::pals_adaptive::CONTINUATION_RESOLUTION_RECORDS_MAX,
                            role_consumption: true,
                        },
                    )
                    .and_then(|diagnostic| {
                        driver.install_continuation_diagnostic(diagnostic.clone())?;
                        Ok(diagnostic)
                    })
                }
                _ => {
                    Err(crate::pals_adaptive::ContinuationDiagnosticError::AdmissionAlreadyStarted)
                }
            };
            observation.install_profiled_feedback(installed);
        }
        #[cfg(not(feature = "pals-adaptive-diagnostics"))]
        let _ = first_enable;
        observation
    }

    /// Add source-only initial-stop evidence to the already selected lifecycle
    /// stream. This does not enable lifecycle output or issue a live permit.
    pub(crate) fn enable_initial_stop_source_stderr_diagnostics(&self) -> Result<(), &'static str> {
        let diagnostics = self
            .lifecycle_diagnostics
            .as_ref()
            .ok_or("initial-stop source diagnostics require lifecycle diagnostics")?;
        diagnostics.enable_initial_stop_source();
        Ok(())
    }

    fn actual(&self) -> &dyn SearchSessionDriver {
        match self.inner.as_ref() {
            ActualProductDriver::Pc(driver) | ActualProductDriver::PcNnuec(driver) => driver,
            ActualProductDriver::Pcr(driver) => driver,
            ActualProductDriver::Nnuec(driver) => driver,
        }
    }
}
impl SearchSessionDriver for RoventNativeProductDriver {
    fn supports_ponder_clock(&self) -> bool {
        self.actual().supports_ponder_clock()
    }
    fn node_budget_semantics(&self) -> crate::bridge::NodeBudgetSemantics {
        self.actual().node_budget_semantics()
    }
    fn validate_go(
        &self,
        position: &Position,
        go: &crate::parser::GoLimits,
    ) -> Result<(), ContractError> {
        self.actual().validate_go(position, go)
    }
    fn kind(&self) -> SearchKind {
        SearchKind::Rovent
    }
    fn identity(&self) -> EngineIdentity {
        self.actual().identity()
    }
    fn implementation(&self) -> Digest {
        self.actual().implementation()
    }
    fn search_goal_capability(&self) -> Option<NonZeroU16> {
        self.actual().search_goal_capability()
    }
    fn reset_game(&self) -> Result<(), ContractError> {
        self.actual().reset_game()
    }
    fn finish_session(
        &self,
        until: Instant,
    ) -> Result<Option<DriverPhysicalCompletion>, SearchSessionFailure> {
        let result = self.actual().finish_session(until);
        if matches!(result, Ok(Some(DriverPhysicalCompletion::Confirmed))) {
            self.physically_closed.store(true, Ordering::Release);
        }
        if let Some(diagnostics) = &self.lifecycle_diagnostics {
            diagnostics.observe_finish(self, &result);
            diagnostics.observe_storage(self, None);
        }
        result
    }
    fn accept_uci_output(
        &self,
        authority: rz_contracts::pals::SearchAuthority,
        bestmove: &str,
        from_report: bool,
    ) -> Result<(), ContractError> {
        self.actual()
            .accept_uci_output(authority, bestmove, from_report)
    }
    fn run(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        let result = self.actual().run(position, context, progress);
        if let Some(diagnostics) = &self.lifecycle_diagnostics {
            diagnostics.observe_root(self, context, result.as_ref());
            diagnostics.observe_storage(self, Some(context.authority));
            #[cfg(feature = "pals-adaptive-diagnostics")]
            {
                diagnostics.observe_profiled_feedback(self, context);
                diagnostics.observe_nnuec_owner(self, context);
            }
        }
        result
    }
}
impl Drop for RoventNativeProductDriver {
    fn drop(&mut self) {
        if !self.physically_closed.load(Ordering::Acquire) {
            std::mem::forget(Arc::clone(&self.inner));
            std::mem::forget(Arc::clone(&self.sources));
            if let Some(registration) = &self.retirement_sources {
                std::mem::forget(Arc::clone(registration));
            }
        }
    }
}

/// PC-only /2 product owner. Collection, storage, NNUEC and R assembly are
/// separate capabilities and are intentionally absent from this constructor.
/// The original V2 pinned sources remain held with the same session until its
/// physical finish; an unknown finish quarantines both owners.
pub struct RoventNativeProductDriverV2 {
    inner: Arc<RoventPcSessionDriver>,
    sources: Arc<RoventPinnedNativeSources<rz_experiments::RoventInputLockV2>>,
    process: Arc<VerifiedOwnedRoventProcessBoundaryV2>,
    physically_closed: Arc<AtomicBool>,
}

pub struct RoventProductV2StartupRejection {
    pub failure: SearchSessionFailure,
    pub prepared: PreparedRoventNativeV2<PcPortQuestion>,
    pub process: Arc<VerifiedOwnedRoventProcessBoundaryV2>,
    pub loaded: Option<LoadedRoventNnuecSource<rz_experiments::RoventInputLockV2>>,
}

impl RoventNativeProductDriverV2 {
    #[allow(clippy::result_large_err)]
    pub fn pc(
        prepared: PreparedRoventNativeV2<PcPortQuestion>,
        process: Arc<VerifiedOwnedRoventProcessBoundaryV2>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
    ) -> Result<Self, RoventProductV2StartupRejection> {
        let sources = Arc::clone(prepared.sources());
        if !Arc::ptr_eq(sources.lock_owner(), process.input_lock())
            || !process.matches_startup_control(prepared.startup_control())
            || sources.private_root_path() != process.private_cwd()
        {
            let mut cause = failure(
                "RoventPcV2ProcessOwner",
                "native source and observed process do not share the original V2 lock, control and root",
            );
            cause.physical_completion = DriverPhysicalCompletion::Unknown;
            return Err(RoventProductV2StartupRejection {
                failure: cause,
                prepared,
                process,
                loaded: None,
            });
        }
        match sources.same_original_root(process.original_root()) {
            Ok(true) => {}
            Ok(false) => {
                let mut cause = failure(
                    "RoventPcV2ProcessOwner",
                    "native source root differs from the process root",
                );
                cause.physical_completion = DriverPhysicalCompletion::Unknown;
                return Err(RoventProductV2StartupRejection {
                    failure: cause,
                    prepared,
                    process,
                    loaded: None,
                });
            }
            Err(error) => {
                let mut cause = SearchSessionFailure::debug("RoventPcV2ProcessRoot", &error);
                cause.physical_completion = DriverPhysicalCompletion::Unknown;
                return Err(RoventProductV2StartupRejection {
                    failure: cause,
                    prepared,
                    process,
                    loaded: None,
                });
            }
        }
        let driver = RoventPcSessionDriver::new_v2(
            prepared,
            Arc::clone(&process),
            clock,
            owners,
            identity,
            root_policy,
        )
        .map_err(|rejected| {
            let mut cause = rejected.failure;
            cause.physical_completion = DriverPhysicalCompletion::Unknown;
            RoventProductV2StartupRejection {
                failure: cause,
                prepared: rejected.prepared,
                process: rejected.process,
                loaded: rejected.loaded,
            }
        })?;
        Ok(Self {
            inner: Arc::new(driver),
            sources,
            process,
            physically_closed: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Uses the original V2 loaded model owner, never a copied V1 lock or
    /// freshly loaded weights. Actor creation and shutdown stay in the session.
    #[allow(clippy::result_large_err, clippy::too_many_arguments)]
    pub fn pc_with_shared_nnuec(
        prepared: PreparedRoventNativeV2<PcPortQuestion>,
        loaded: LoadedRoventNnuecSource<rz_experiments::RoventInputLockV2>,
        process: Arc<VerifiedOwnedRoventProcessBoundaryV2>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
    ) -> Result<Self, RoventProductV2StartupRejection> {
        let sources = Arc::clone(prepared.sources());
        let driver = RoventPcSessionDriver::new_v2_with_shared_nnuec(
            prepared,
            loaded,
            Arc::clone(&process),
            clock,
            owners,
            identity,
            root_policy,
        )
        .map_err(|rejected| {
            let mut failure = rejected.failure;
            failure.physical_completion = DriverPhysicalCompletion::Unknown;
            RoventProductV2StartupRejection {
                failure,
                prepared: rejected.prepared,
                process: rejected.process,
                loaded: rejected.loaded,
            }
        })?;
        Ok(Self {
            inner: Arc::new(driver),
            sources,
            process,
            physically_closed: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Read-only handle to the actual whole-session physical finish.
    pub(crate) fn physical_close_observation(&self) -> RoventProductPhysicalClose {
        RoventProductPhysicalClose {
            closed: Arc::clone(&self.physically_closed),
        }
    }
}

impl SearchSessionDriver for RoventNativeProductDriverV2 {
    fn supports_ponder_clock(&self) -> bool {
        self.inner.supports_ponder_clock()
    }
    fn node_budget_semantics(&self) -> crate::bridge::NodeBudgetSemantics {
        self.inner.node_budget_semantics()
    }
    fn validate_go(
        &self,
        position: &Position,
        go: &crate::parser::GoLimits,
    ) -> Result<(), ContractError> {
        self.inner.validate_go(position, go)
    }
    fn kind(&self) -> SearchKind {
        SearchKind::Rovent
    }
    fn identity(&self) -> EngineIdentity {
        self.inner.identity()
    }
    fn implementation(&self) -> Digest {
        self.inner.implementation()
    }
    fn search_goal_capability(&self) -> Option<NonZeroU16> {
        self.inner.search_goal_capability()
    }
    fn reset_game(&self) -> Result<(), ContractError> {
        self.inner.reset_game()
    }
    fn finish_session(
        &self,
        until: Instant,
    ) -> Result<Option<DriverPhysicalCompletion>, SearchSessionFailure> {
        let result = self.inner.finish_session(until);
        if matches!(result, Ok(Some(DriverPhysicalCompletion::Confirmed))) {
            self.physically_closed.store(true, Ordering::Release);
        }
        result
    }
    fn accept_uci_output(
        &self,
        authority: rz_contracts::pals::SearchAuthority,
        bestmove: &str,
        from_report: bool,
    ) -> Result<(), ContractError> {
        self.inner
            .accept_uci_output(authority, bestmove, from_report)
    }
    fn run(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        self.inner.run(position, context, progress)
    }
}

impl Drop for RoventNativeProductDriverV2 {
    fn drop(&mut self) {
        if !self.physically_closed.load(Ordering::Acquire) {
            std::mem::forget(Arc::clone(&self.inner));
            std::mem::forget(Arc::clone(&self.sources));
            std::mem::forget(Arc::clone(&self.process));
        }
    }
}

/// V2 own-R product retains the actual prepared worker, original input lock,
/// process boundary and CPU owners through the same session physical finish.
/// Collection, NNUEC, storage and diagnostic registrations require their own
/// V2 admission paths and are not silently enabled here.
pub struct RoventOwnPcrProductDriverV2 {
    inner: Arc<PcrDriver>,
    sources: Arc<RoventPinnedNativeSources<rz_experiments::RoventInputLockV2>>,
    process: Arc<VerifiedOwnedRoventProcessBoundaryV2>,
    physically_closed: Arc<AtomicBool>,
}

#[allow(clippy::large_enum_variant)]
pub enum RoventOwnPcrV2RejectedOwners {
    Prepared(PreparedRoventNativeV2<AdaptivePortQuestion>),
    Session(AdaptiveSessionAdmissionRejection<RoventNativeBackend<AdaptivePortQuestion>>),
    Driver(Box<PcrDriver>),
}

pub struct RoventOwnPcrV2StartupRejection {
    pub failure: SearchSessionFailure,
    pub owners: RoventOwnPcrV2RejectedOwners,
    pub sources: Arc<RoventPinnedNativeSources<rz_experiments::RoventInputLockV2>>,
    pub process: Arc<VerifiedOwnedRoventProcessBoundaryV2>,
}

impl RoventOwnPcrProductDriverV2 {
    #[allow(clippy::result_large_err)]
    pub fn own_board_pcr(
        prepared: PreparedRoventNativeV2<AdaptivePortQuestion>,
        process: Arc<VerifiedOwnedRoventProcessBoundaryV2>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
    ) -> Result<Self, RoventOwnPcrV2StartupRejection> {
        let pins = Arc::clone(prepared.sources());
        let same_owner = Arc::ptr_eq(pins.lock_owner(), process.input_lock())
            && process.matches_startup_control(prepared.startup_control())
            && pins.private_root_path() == process.private_cwd();
        let owner_check = if same_owner {
            match pins.same_original_root(process.original_root()) {
                Ok(true) => Ok(()),
                Ok(false) => Err(failure(
                    "RoventPcrV2ProcessOwner",
                    "native and process root differ",
                )),
                Err(error) => Err(SearchSessionFailure::debug(
                    "RoventPcrV2ProcessRoot",
                    &error,
                )),
            }
        } else {
            Err(failure(
                "RoventPcrV2ProcessOwner",
                "native source and process must share the original lock, control and root",
            ))
        };
        if let Err(mut failure) = owner_check {
            failure.physical_completion = DriverPhysicalCompletion::Unknown;
            return Err(RoventOwnPcrV2StartupRejection {
                failure,
                owners: RoventOwnPcrV2RejectedOwners::Prepared(prepared),
                sources: pins,
                process,
            });
        }
        if pins.lock().manifest.purpose == rz_experiments::RoventPurpose::Collection {
            let mut failure = failure(
                "RoventPcrV2CollectionUnsupported",
                "V2 collection requires an actual collection origin and root hooks",
            );
            failure.physical_completion = DriverPhysicalCompletion::Unknown;
            return Err(RoventOwnPcrV2StartupRejection {
                failure,
                owners: RoventOwnPcrV2RejectedOwners::Prepared(prepared),
                sources: pins,
                process,
            });
        }
        let admission_check = (|| -> Result<(), SearchSessionFailure> {
            pins.configuration()
                .require_supported_native_capabilities()
                .map_err(|error| SearchSessionFailure::debug("RoventPcrV2Capability", &error))?;
            if pins.collection_preload_sources().is_some()
                || identity.name.is_empty()
                || identity.author.is_empty()
            {
                return Err(failure(
                    "RoventPcrV2Startup",
                    "unsupported collection inputs or empty engine identity",
                ));
            }
            let ready = prepared.service.ready_clock().ok_or_else(|| {
                failure("RoventPcrV2Startup", "actual native Ready clock is absent")
            })?;
            let original_clock = clock.runtime_clock();
            if prepared.service.is_quarantined()
                || prepared.service.native_acceptance_verifier().is_none()
                || ready.domain() != original_clock.domain()
                || ready.origin() != original_clock.origin()
                || prepared.snapshot().input_lock != pins.lock_digest()
            {
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
            Ok(())
        })();
        if let Err(mut failure) = admission_check {
            failure.physical_completion = DriverPhysicalCompletion::Unknown;
            return Err(RoventOwnPcrV2StartupRejection {
                failure,
                owners: RoventOwnPcrV2RejectedOwners::Prepared(prepared),
                sources: pins,
                process,
            });
        }
        let built = build_own_board_pcr!(pins, prepared, clock, identity);
        let (engine, cpu, metadata, registration, options, limits, profile) = match built {
            Ok(built) => built,
            Err(mut failure) => {
                failure.physical_completion = DriverPhysicalCompletion::Unknown;
                return Err(RoventOwnPcrV2StartupRejection {
                    failure,
                    owners: RoventOwnPcrV2RejectedOwners::Prepared(prepared),
                    sources: pins,
                    process,
                });
            }
        };
        if let Err(mut failure) = validate_v2_own_worker_identities(
            &pins.lock().manifest.own_r_root_tie,
            cpu.iter().map(|owner| &owner.engine),
        ) {
            failure.physical_completion = DriverPhysicalCompletion::Unknown;
            return Err(RoventOwnPcrV2StartupRejection {
                failure,
                owners: RoventOwnPcrV2RejectedOwners::Prepared(prepared),
                sources: pins,
                process,
            });
        }
        let (native, _, _) = prepared.into_parts();
        let sources = AdaptiveSessionSources {
            native,
            engine,
            cpu,
            external: None,
            checker_metadata: metadata,
        };
        let issuer: AdaptiveRulesOwnerIssuer = Arc::new(move || owners.allocate());
        let mut driver = AdaptiveSessionDriver::new_rovent_pcr_with_resource_limits(
            options,
            limits,
            sources,
            clock,
            issuer,
            registration,
            profile,
        )
        .map_err(|rejected| {
            let mut failure = rejected.failure.clone();
            failure.physical_completion = DriverPhysicalCompletion::Unknown;
            RoventOwnPcrV2StartupRejection {
                failure,
                owners: RoventOwnPcrV2RejectedOwners::Session(rejected),
                sources: Arc::clone(&pins),
                process: Arc::clone(&process),
            }
        })?;
        if let Err(mut failure) =
            driver.register_rovent_root_policy_v2(Arc::clone(pins.lock_owner()), root_policy)
        {
            failure.physical_completion = DriverPhysicalCompletion::Unknown;
            return Err(RoventOwnPcrV2StartupRejection {
                failure,
                owners: RoventOwnPcrV2RejectedOwners::Driver(Box::new(driver)),
                sources: pins,
                process,
            });
        }
        Ok(Self {
            inner: Arc::new(driver),
            sources: pins,
            process,
            physically_closed: Arc::new(AtomicBool::new(false)),
        })
    }

    pub(crate) fn physical_close_observation(&self) -> RoventProductPhysicalClose {
        RoventProductPhysicalClose {
            closed: Arc::clone(&self.physically_closed),
        }
    }
}

impl SearchSessionDriver for RoventOwnPcrProductDriverV2 {
    fn supports_ponder_clock(&self) -> bool {
        self.inner.supports_ponder_clock()
    }
    fn node_budget_semantics(&self) -> crate::bridge::NodeBudgetSemantics {
        self.inner.node_budget_semantics()
    }
    fn validate_go(
        &self,
        position: &Position,
        go: &crate::parser::GoLimits,
    ) -> Result<(), ContractError> {
        self.inner.validate_go(position, go)
    }
    fn kind(&self) -> SearchKind {
        SearchKind::Rovent
    }
    fn identity(&self) -> EngineIdentity {
        self.inner.identity()
    }
    fn implementation(&self) -> Digest {
        self.inner.implementation()
    }
    fn search_goal_capability(&self) -> Option<NonZeroU16> {
        self.inner.search_goal_capability()
    }
    fn reset_game(&self) -> Result<(), ContractError> {
        self.inner.reset_game()
    }
    fn finish_session(
        &self,
        until: Instant,
    ) -> Result<Option<DriverPhysicalCompletion>, SearchSessionFailure> {
        let result = self.inner.finish_session(until);
        if matches!(result, Ok(Some(DriverPhysicalCompletion::Confirmed))) {
            self.physically_closed.store(true, Ordering::Release);
        }
        result
    }
    fn accept_uci_output(
        &self,
        authority: rz_contracts::pals::SearchAuthority,
        bestmove: &str,
        from_report: bool,
    ) -> Result<(), ContractError> {
        self.inner
            .accept_uci_output(authority, bestmove, from_report)
    }
    fn run(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        self.inner.run(position, context, progress)
    }
}

impl Drop for RoventOwnPcrProductDriverV2 {
    fn drop(&mut self) {
        if !self.physically_closed.load(Ordering::Acquire) {
            std::mem::forget(Arc::clone(&self.inner));
            std::mem::forget(Arc::clone(&self.sources));
            std::mem::forget(Arc::clone(&self.process));
        }
    }
}

/// V2 external R keeps the original native/OS process and retirement owners
/// through the same physical finish as its P/C and external researcher session.
pub struct RoventExternalPcrProductDriverV2 {
    inner: Arc<PcrDriver>,
    sources: Arc<RoventPinnedNativeSources<rz_experiments::RoventInputLockV2>>,
    process: Arc<VerifiedOwnedRoventProcessBoundaryV2>,
    retirement_sources: Arc<LaunchRetirementRegistrationCustody>,
    physically_closed: Arc<AtomicBool>,
}

#[allow(clippy::large_enum_variant)]
pub enum RoventExternalPcrV2RejectedOwners {
    Preflight {
        prepared: PreparedRoventNativeV2<AdaptivePortQuestion>,
        external: RoventPinnedExternalRSource<rz_experiments::RoventInputLockV2>,
        evidence: LaunchRetirementEvidence,
    },
    Session {
        rejected: Box<RoventExternalProductStartupRejectionV2>,
        registration: Arc<LaunchRetirementRegistrationCustody>,
    },
}

pub struct RoventExternalPcrV2StartupRejection {
    pub failure: SearchSessionFailure,
    pub owners: RoventExternalPcrV2RejectedOwners,
    pub sources: Arc<RoventPinnedNativeSources<rz_experiments::RoventInputLockV2>>,
    pub process: Arc<VerifiedOwnedRoventProcessBoundaryV2>,
}

impl RoventExternalPcrProductDriverV2 {
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    pub fn external_pcr(
        prepared: PreparedRoventNativeV2<AdaptivePortQuestion>,
        external: RoventPinnedExternalRSource<rz_experiments::RoventInputLockV2>,
        retirement_evidence: LaunchRetirementEvidence,
        process: Arc<VerifiedOwnedRoventProcessBoundaryV2>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
    ) -> Result<Self, RoventExternalPcrV2StartupRejection> {
        let sources = Arc::clone(prepared.sources());
        let checked = (|| -> Result<(), SearchSessionFailure> {
            if !Arc::ptr_eq(sources.lock_owner(), process.input_lock())
                || !process.matches_startup_control(prepared.startup_control())
                || sources.private_root_path() != process.private_cwd()
                || !Arc::ptr_eq(&sources, external.original_sources())
                || retirement_evidence.input_lock_digest() != sources.lock_digest()
                || !retirement_evidence
                    .original_lock_owner_v2()
                    .is_some_and(|owner| Arc::ptr_eq(owner, sources.lock_owner()))
                || !retirement_evidence
                    .original_process_v2()
                    .is_some_and(|original| Arc::ptr_eq(original, &process))
            {
                return Err(failure(
                    "RoventExternalPcrV2Owner",
                    "native, external, retirement and process owners differ",
                ));
            }
            match sources.same_original_root(process.original_root()) {
                Ok(true) => Ok(()),
                Ok(false) => Err(failure(
                    "RoventExternalPcrV2Owner",
                    "native and process root differ",
                )),
                Err(error) => Err(SearchSessionFailure::debug(
                    "RoventExternalPcrV2Root",
                    &error,
                )),
            }
        })();
        if let Err(mut failure) = checked {
            failure.physical_completion = DriverPhysicalCompletion::Unknown;
            return Err(RoventExternalPcrV2StartupRejection {
                failure,
                owners: RoventExternalPcrV2RejectedOwners::Preflight {
                    prepared,
                    external,
                    evidence: retirement_evidence,
                },
                sources,
                process,
            });
        }
        let (retirement_sink, registration) = retirement_evidence.into_parts();
        let driver = external_researcher_session_v2(
            prepared,
            external,
            retirement_sink,
            clock,
            owners,
            identity,
            root_policy,
        )
        .map_err(|rejected| {
            let mut failure = rejected.failure.clone();
            failure.physical_completion = DriverPhysicalCompletion::Unknown;
            RoventExternalPcrV2StartupRejection {
                failure,
                owners: RoventExternalPcrV2RejectedOwners::Session {
                    rejected: Box::new(rejected),
                    registration: Arc::clone(&registration),
                },
                sources: Arc::clone(&sources),
                process: Arc::clone(&process),
            }
        })?;
        Ok(Self {
            inner: Arc::new(driver),
            sources,
            process,
            retirement_sources: registration,
            physically_closed: Arc::new(AtomicBool::new(false)),
        })
    }

    pub(crate) fn physical_close_observation(&self) -> RoventProductPhysicalClose {
        RoventProductPhysicalClose {
            closed: Arc::clone(&self.physically_closed),
        }
    }
}

impl SearchSessionDriver for RoventExternalPcrProductDriverV2 {
    fn supports_ponder_clock(&self) -> bool {
        self.inner.supports_ponder_clock()
    }
    fn node_budget_semantics(&self) -> crate::bridge::NodeBudgetSemantics {
        self.inner.node_budget_semantics()
    }
    fn validate_go(
        &self,
        position: &Position,
        go: &crate::parser::GoLimits,
    ) -> Result<(), ContractError> {
        self.inner.validate_go(position, go)
    }
    fn kind(&self) -> SearchKind {
        SearchKind::Rovent
    }
    fn identity(&self) -> EngineIdentity {
        self.inner.identity()
    }
    fn implementation(&self) -> Digest {
        self.inner.implementation()
    }
    fn search_goal_capability(&self) -> Option<NonZeroU16> {
        self.inner.search_goal_capability()
    }
    fn reset_game(&self) -> Result<(), ContractError> {
        self.inner.reset_game()
    }
    fn finish_session(
        &self,
        until: Instant,
    ) -> Result<Option<DriverPhysicalCompletion>, SearchSessionFailure> {
        let result = self.inner.finish_session(until);
        if matches!(result, Ok(Some(DriverPhysicalCompletion::Confirmed))) {
            self.physically_closed.store(true, Ordering::Release);
        }
        result
    }
    fn accept_uci_output(
        &self,
        authority: rz_contracts::pals::SearchAuthority,
        bestmove: &str,
        from_report: bool,
    ) -> Result<(), ContractError> {
        self.inner
            .accept_uci_output(authority, bestmove, from_report)
    }
    fn run(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        self.inner.run(position, context, progress)
    }
}

impl Drop for RoventExternalPcrProductDriverV2 {
    fn drop(&mut self) {
        if !self.physically_closed.load(Ordering::Acquire) {
            std::mem::forget(Arc::clone(&self.inner));
            std::mem::forget(Arc::clone(&self.sources));
            std::mem::forget(Arc::clone(&self.process));
            std::mem::forget(Arc::clone(&self.retirement_sources));
        }
    }
}

// Validate each actual CPU worker, not just the startup document or primary
// mirror, before the prepared native service is transferred to the session.
/// V2 own-R NNUEC product preserves the actual session, model and original
/// process owner until its real session physical shutdown is confirmed.
pub struct RoventOwnNnuecProductDriverV2 {
    inner: Arc<RoventNnuecSessionDriver>,
    sources: Arc<RoventPinnedNativeSources<rz_experiments::RoventInputLockV2>>,
    process: Arc<VerifiedOwnedRoventProcessBoundaryV2>,
    physically_closed: Arc<AtomicBool>,
}
pub struct RoventOwnNnuecV2StartupRejection {
    pub failure: SearchSessionFailure,
    pub rejected: RoventNnuecSessionV2Rejection,
}
impl RoventOwnNnuecProductDriverV2 {
    #[allow(clippy::result_large_err, clippy::too_many_arguments)]
    pub fn own_nnuec_pcr(
        prepared: PreparedRoventNativeV2<AdaptivePortQuestion>,
        loaded: LoadedRoventNnuecSource<rz_experiments::RoventInputLockV2>,
        process: Arc<VerifiedOwnedRoventProcessBoundaryV2>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
    ) -> Result<Self, RoventOwnNnuecV2StartupRejection> {
        let sources = Arc::clone(prepared.sources());
        let driver = RoventNnuecSessionDriver::new_v2(
            prepared,
            loaded,
            Arc::clone(&process),
            clock,
            owners,
            identity,
            root_policy,
        )
        .map_err(|rejected| {
            let mut failure = rejected.failure.clone();
            failure.physical_completion = DriverPhysicalCompletion::Unknown;
            RoventOwnNnuecV2StartupRejection { failure, rejected }
        })?;
        Ok(Self {
            inner: Arc::new(driver),
            sources,
            process,
            physically_closed: Arc::new(AtomicBool::new(false)),
        })
    }
    pub(crate) fn physical_close_observation(&self) -> RoventProductPhysicalClose {
        RoventProductPhysicalClose {
            closed: Arc::clone(&self.physically_closed),
        }
    }
}

impl SearchSessionDriver for RoventOwnNnuecProductDriverV2 {
    fn supports_ponder_clock(&self) -> bool {
        self.inner.supports_ponder_clock()
    }
    fn node_budget_semantics(&self) -> crate::bridge::NodeBudgetSemantics {
        self.inner.node_budget_semantics()
    }
    fn validate_go(
        &self,
        position: &Position,
        go: &crate::parser::GoLimits,
    ) -> Result<(), ContractError> {
        self.inner.validate_go(position, go)
    }
    fn kind(&self) -> SearchKind {
        SearchKind::Rovent
    }
    fn identity(&self) -> EngineIdentity {
        self.inner.identity()
    }
    fn pals_checker_registration(&self) -> Option<PalsCheckerRegistration> {
        self.inner.pals_checker_registration()
    }
    fn checker_started(&self) -> Option<bool> {
        self.inner.checker_started()
    }
    fn implementation(&self) -> Digest {
        self.inner.implementation()
    }
    fn search_goal_capability(&self) -> Option<NonZeroU16> {
        self.inner.search_goal_capability()
    }
    fn reset_game(&self) -> Result<(), ContractError> {
        self.inner.reset_game()
    }
    fn finish_session(
        &self,
        until: Instant,
    ) -> Result<Option<DriverPhysicalCompletion>, SearchSessionFailure> {
        let result = self.inner.finish_session(until);
        if matches!(result, Ok(Some(DriverPhysicalCompletion::Confirmed))) {
            self.physically_closed.store(true, Ordering::Release);
        }
        result
    }
    fn accept_uci_output(
        &self,
        authority: rz_contracts::pals::SearchAuthority,
        bestmove: &str,
        from_report: bool,
    ) -> Result<(), ContractError> {
        self.inner
            .accept_uci_output(authority, bestmove, from_report)
    }
    fn run(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        self.inner.run(position, context, progress)
    }
}

impl Drop for RoventOwnNnuecProductDriverV2 {
    fn drop(&mut self) {
        if !self.physically_closed.load(Ordering::Acquire) {
            std::mem::forget(Arc::clone(&self.inner));
            std::mem::forget(Arc::clone(&self.sources));
            std::mem::forget(Arc::clone(&self.process));
        }
    }
}

fn validate_v2_own_worker_identities<'a>(
    declaration: &rz_experiments::RoventRootTieDeclarationV2,
    workers: impl IntoIterator<Item = &'a CpuEngine>,
) -> Result<(), SearchSessionFailure> {
    let rz_experiments::RoventRootTieDeclarationV2::Own {
        expected_search_identity,
        expected_worker_conditions_sha256,
        ..
    } = declaration
    else {
        return Err(failure(
            "RoventPcrV2WorkerIdentity",
            "own-R worker declaration is absent",
        ));
    };
    let expected = rz_eval::asset::parse_sha256(expected_worker_conditions_sha256)
        .map(Digest)
        .map_err(|error| SearchSessionFailure::debug("RoventPcrV2WorkerIdentity", &error))?;
    let mut count = 0usize;
    for worker in workers {
        count += 1;
        if worker.search_identity() != expected_search_identity
            || cpu_search_conditions_digest(worker) != expected
        {
            return Err(failure(
                "RoventPcrV2WorkerIdentity",
                "actual worker identity or search conditions differ from the original lock",
            ));
        }
    }
    if count == 0 {
        return Err(failure(
            "RoventPcrV2WorkerIdentity",
            "no actual own-R worker",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod v2_own_worker_identity_tests {
    use super::*;
    use rz_experiments::{RoventRootTieDeclarationV2, RoventRootTiePolicyV2};

    #[test]
    fn validates_every_actual_worker_against_the_original_declaration() {
        let a = CpuEngine::new(Default::default()).unwrap();
        let b = CpuEngine::new(Default::default()).unwrap();
        let digest = cpu_search_conditions_digest(&a);
        let hex = digest.0.iter().map(|byte| format!("{byte:02x}")).collect();
        let valid = RoventRootTieDeclarationV2::Own {
            policy: RoventRootTiePolicyV2::LegacyFirstSearched,
            expected_search_identity: a.search_identity().to_owned(),
            expected_worker_conditions_sha256: hex,
        };
        assert!(validate_v2_own_worker_identities(&valid, [&a, &b]).is_ok());
        let changed = CpuEngine::new(rz_search::cpu::CpuConfig {
            max_depth: 15,
            ..Default::default()
        })
        .unwrap();
        assert!(validate_v2_own_worker_identities(&valid, [&a, &changed]).is_err());
        assert!(validate_v2_own_worker_identities(&valid, std::iter::empty()).is_err());
        let mut bad_identity = valid.clone();
        if let RoventRootTieDeclarationV2::Own {
            expected_search_identity,
            ..
        } = &mut bad_identity
        {
            expected_search_identity.push_str("-different");
        }
        assert!(validate_v2_own_worker_identities(&bad_identity, [&a]).is_err());
        let mut bad_conditions = valid;
        if let RoventRootTieDeclarationV2::Own {
            expected_worker_conditions_sha256,
            ..
        } = &mut bad_conditions
        {
            *expected_worker_conditions_sha256 = "00".repeat(32);
        }
        assert!(validate_v2_own_worker_identities(&bad_conditions, [&a, &b]).is_err());
    }
}

fn failure(code: &'static str, detail: &str) -> SearchSessionFailure {
    SearchSessionFailure {
        physical_completion: DriverPhysicalCompletion::Confirmed,
        code,
        detail: detail.into(),
    }
}
