//! V6 CPU/search preparation against the original startup owner.
//!
//! This module prepares in-process own CPU engines only. It neither launches
//! workers nor admits a model, external engine, OS resource plan or GPU work.
//! V6 source observations remain separate from these declarations.

use super::super::{
    AdaptiveBootstrapError, BootstrapResourceIntentV5, boundary, failure,
    require_original_startup_window,
};
use crate::engine::ProcessClock;
use crate::pals_adaptive::{AdaptiveExecutionLimits, PersistentCpuOwner};
use crate::pals_adaptive_startup::{
    PalsSearchSettingsV6, PinnedPalsSystemStartupV6, parse_pals_search_settings_v6,
};
use rz_search::cpu::{CpuEngine, CpuResumePolicy};
use rz_search::cpu_value::BootstrapCpuValue;
use rz_search::cpu_worker::{
    CpuWorkerProfile, cpu_evaluation_namespace_digest, cpu_search_conditions_digest,
};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

/// Exact owned CPU engines and declared source observations. The original V6
/// source and its control stay alive through this value; no V5 lock or source
/// comparison is fabricated. Actual OS affinity/quota and worker execution
/// require separate readback after this preparation.
#[must_use]
pub(super) struct PalsSystemCpuPreparationV6 {
    pub(super) original_source: Arc<PinnedPalsSystemStartupV6>,
    pub(super) clock: ProcessClock,
    pub(super) search: PalsSearchSettingsV6,
    pub(super) intent: Arc<BootstrapResourceIntentV5>,
    pub(super) limits: AdaptiveExecutionLimits,
    pub(super) own_workers: Vec<PersistentCpuOwner>,
    pub(super) original_until: Instant,
    pub(super) original_cancelled: Arc<AtomicBool>,
}

/// `clock` must be the same process clock later handed to the coordinator;
/// matching the epoch alone does not prove a matching monotonic origin. The
/// returned engines have not been moved to workers or run searches.
pub(super) fn prepare_cpu_v6(
    owner: Arc<PinnedPalsSystemStartupV6>,
    clock: &ProcessClock,
) -> Result<PalsSystemCpuPreparationV6, AdaptiveBootstrapError> {
    let (original_until, original_cancelled) = owner.original_startup_control();
    require_original_startup_window(original_until, &original_cancelled)?;
    owner
        .verify_source_unchanged()
        .map_err(|error| failure("SystemV6CpuSource", &error))?;
    let config = &owner.lock().config;
    if clock.epoch().0 != config.process_generation {
        return Err(boundary("SystemV6CpuProcessEpoch"));
    }
    let search = parse_pals_search_settings_v6(config, original_until)
        .map_err(|error| failure("SystemV6CpuSearchSettings", &error))?;
    if search.settings.resume != CpuResumePolicy::PausedStack {
        return Err(boundary("CooperativeCpuRequiresDeclaredPausedStack"));
    }
    let endpoint = &config.endpoint;
    let intent = Arc::new(
        BootstrapResourceIntentV5::prepare(
            &endpoint.resources,
            &endpoint.coordinator.slices,
            endpoint.cpu.external.is_some(),
            clock.epoch(),
            original_until,
            Arc::clone(&original_cancelled),
            Instant::now(),
        )
        .map_err(|error| failure("SystemV6CpuResourceIntent", &error))?,
    );
    let limits = intent.execution_limits();
    let mut own_workers = Vec::new();
    own_workers
        .try_reserve_exact(intent.own_worker_count())
        .map_err(|error| failure("SystemV6CpuOwnerAllocation", &error))?;
    for index in 0..intent.own_worker_count() {
        require_original_startup_window(original_until, &original_cancelled)?;
        let evaluator = Arc::new(BootstrapCpuValue::default());
        let settings = &search.settings;
        let mut engine = if settings.paused_side_scope {
            CpuEngine::with_paused_side_scope(
                settings.own_cpu.clone(),
                evaluator,
                settings.ordering,
            )
        } else {
            CpuEngine::with_evaluator_ordering_and_resume_policy(
                settings.own_cpu.clone(),
                evaluator,
                settings.ordering,
                settings.resume,
            )
        }
        .map_err(|error| failure("SystemV6CpuOwner", &error))?;
        engine
            .set_paused_stack_max_bytes(limits.paused_bytes_per_worker_max)
            .map_err(|error| failure("SystemV6PausedOwnerCapacity", &error))?;
        let profile = CpuWorkerProfile {
            owner: intent
                .own_worker_owner(index, clock.epoch(), Instant::now())
                .map_err(|error| failure("SystemV6CpuOwnerIntent", &error))?,
            max_depth: engine.config().max_depth,
            evaluation_namespace: cpu_evaluation_namespace_digest(&engine)
                .map_err(|error| failure("SystemV6CpuValueNamespace", &error))?,
            search_conditions: cpu_search_conditions_digest(&engine)
                .map_err(|error| failure("SystemV6CpuSearchConditions", &error))?,
        };
        own_workers.push(PersistentCpuOwner { engine, profile });
    }
    owner
        .verify_source_unchanged()
        .map_err(|error| failure("SystemV6CpuSource", &error))?;
    require_original_startup_window(original_until, &original_cancelled)?;
    Ok(PalsSystemCpuPreparationV6 {
        original_source: owner,
        clock: clock.clone(),
        search,
        intent,
        limits,
        own_workers,
        original_until,
        original_cancelled,
    })
}
