//! Explicit external-only entry point into the common ticket/Store lifecycle.
//! No internal CPU worker, ledger or adaptive native controller is created.
use super::*;

pub struct ProfiledExternalManagerAdmissionRejection<Q, B, D, L = rz_experiments::RoventInputLock>
where
    B: TicketNativeBackend<ReviewView, Q>,
    L: profiled_external_source::ExternalManagerLock,
{
    pub error: AdaptiveManagerError,
    pub sources: AdaptiveManagerSources<Q, B, D>,
    pub external: ProfiledExternalManagerSource<L>,
    pub continuation: Option<AdaptiveSequenceContinuation>,
}

/// The public external facade does not expose native-only bare ledger APIs.
/// UCI's private SourceKind may move the whole manager into its common owner.
pub struct ProfiledExternalExecutionManager<Q, B, D>
where
    B: TicketNativeBackend<ReviewView, Q>,
{
    inner: AdaptiveExecutionManager<Q, B, D>,
}
pub struct ProfiledExternalPersistentRejection<Q, B, D>
where
    B: TicketNativeBackend<ReviewView, Q>,
{
    pub error: AdaptiveManagerError,
    pub manager: ProfiledExternalExecutionManager<Q, B, D>,
}

impl<Q, B, D> ProfiledExternalExecutionManager<Q, B, D>
where
    Q: Clone + Send + Sync + 'static,
    B: TicketNativeBackend<ReviewView, Q>,
    B::Output: ProposerRankingOutput,
    D: AdaptiveSearchDriver<Q>,
{
    pub(crate) fn into_session_manager(self) -> AdaptiveExecutionManager<Q, B, D> {
        self.inner
    }
    pub fn phase(&self) -> ManagerPhase {
        self.inner.phase()
    }
    pub fn counters(&self) -> AdaptiveManagerCounters {
        self.inner.counters()
    }
    pub fn driver(&self) -> &D {
        self.inner.driver()
    }
    pub fn driver_mut(&mut self) -> &mut D {
        self.inner.driver_mut()
    }
    pub fn failures(&self) -> impl Iterator<Item = &AdaptiveManagerError> {
        self.inner.failures()
    }
    pub fn register_cleanup_deadline(
        &mut self,
        until: Instant,
    ) -> Result<(), AdaptiveManagerError> {
        self.inner.register_cleanup_deadline(until)
    }
    pub fn register_effective_clock(
        &mut self,
        handle: crate::rovent_ponder::RoventPonderClockHandle,
    ) -> Result<(), AdaptiveManagerError> {
        self.inner.register_effective_clock(handle)
    }
    pub fn can_admit_work(&self) -> bool {
        self.inner.can_admit_work()
    }
    pub fn prepare_root_ranking(
        &mut self,
        input: FrameInput<ReviewView, Q>,
    ) -> Result<PreparationId, AdaptiveManagerError> {
        self.inner.prepare_root_ranking(input)
    }
    pub fn begin_root_quiesce(&mut self, until: Instant) -> Result<(), AdaptiveManagerError> {
        self.inner.begin_root_quiesce(until)
    }
    pub fn begin_shutdown(&mut self, until: Instant) -> Result<(), AdaptiveManagerError> {
        self.inner.begin_shutdown(until)
    }
    pub fn tick(&mut self) -> Result<bool, AdaptiveManagerError> {
        self.inner.tick()
    }
    pub fn registered_cleanup_deadline(&self) -> Option<Instant> {
        self.inner.registered_cleanup_deadline()
    }
    pub fn profiled_shutdown_receipt(
        &self,
    ) -> Result<ProfiledExternalShutdownReceipt, AdaptiveManagerError> {
        self.inner.profiled_shutdown_receipt()
    }
    pub fn run_profiled_root_until(
        &mut self,
        until: Instant,
        canceled: &AtomicBool,
    ) -> Result<ProfiledExternalShutdownReceipt, AdaptiveManagerError> {
        self.inner
            .run_profiled_root_until_observed(until, canceled, &mut |_| Ok(()))
    }
    pub fn run_profiled_until(
        &mut self,
        until: Instant,
        canceled: &AtomicBool,
    ) -> Result<ProfiledExternalShutdownReceipt, AdaptiveManagerError> {
        self.inner.run_profiled_until(until, canceled)
    }
    #[allow(clippy::result_large_err)]
    pub fn into_profiled_persistent_parts(
        self,
    ) -> Result<
        ProfiledExternalPersistentParts<Q, B, D>,
        ProfiledExternalPersistentRejection<Q, B, D>,
    > {
        self.inner
            .into_profiled_persistent_parts()
            .map_err(|rejected| ProfiledExternalPersistentRejection {
                error: rejected.error,
                manager: Self {
                    inner: rejected.manager,
                },
            })
    }
    #[allow(clippy::result_large_err)]
    pub fn into_profiled_process_parts(
        self,
    ) -> Result<
        ProfiledExternalPersistentParts<Q, B, D>,
        ProfiledExternalPersistentRejection<Q, B, D>,
    > {
        self.inner
            .into_profiled_process_parts()
            .map_err(|rejected| ProfiledExternalPersistentRejection {
                error: rejected.error,
                manager: Self {
                    inner: rejected.manager,
                },
            })
    }

    /// Return the original V2 source only after the same physical root fence.
    #[allow(clippy::result_large_err)]
    pub fn into_profiled_v2_persistent_parts(
        self,
    ) -> Result<
        ProfiledExternalPersistentParts<Q, B, D, rz_experiments::RoventInputLockV2>,
        ProfiledExternalPersistentRejection<Q, B, D>,
    > {
        self.inner
            .into_profiled_v2_persistent_parts()
            .map_err(|rejected| ProfiledExternalPersistentRejection {
                error: rejected.error,
                manager: Self {
                    inner: rejected.manager,
                },
            })
    }

    /// Return the original V2 source only after the process-close proof.
    #[allow(clippy::result_large_err)]
    pub fn into_profiled_v2_process_parts(
        self,
    ) -> Result<
        ProfiledExternalPersistentParts<Q, B, D, rz_experiments::RoventInputLockV2>,
        ProfiledExternalPersistentRejection<Q, B, D>,
    > {
        self.inner
            .into_profiled_v2_process_parts()
            .map_err(|rejected| ProfiledExternalPersistentRejection {
                error: rejected.error,
                manager: Self {
                    inner: rejected.manager,
                },
            })
    }
}

impl<Q, B, D> AdaptiveExecutionManager<Q, B, D>
where
    Q: Clone + Send + Sync + 'static,
    B: TicketNativeBackend<ReviewView, Q>,
    B::Output: ProposerRankingOutput,
    D: AdaptiveSearchDriver<Q>,
{
    pub(super) fn begin_profiled_close(
        &mut self,
        until: Instant,
        mode: AdaptiveCloseMode,
    ) -> Result<(), AdaptiveManagerError> {
        let ResearcherResources::ProfiledExternal(source) = &mut self.researcher else {
            return Ok(());
        };
        match mode {
            AdaptiveCloseMode::RootQuiesce if !source.root_drained() => source
                .begin_root_quiesce(until)
                .map_err(AdaptiveManagerError::ProfiledExternal)?,
            AdaptiveCloseMode::RootQuiesce => {}
            AdaptiveCloseMode::ProcessShutdown => source.request_shutdown(until),
        }
        Ok(())
    }
    pub(super) fn poll_profiled_close(&mut self) -> Result<bool, AdaptiveManagerError> {
        let ResearcherResources::ProfiledExternal(source) = &mut self.researcher else {
            return Ok(false);
        };
        match self.close_mode {
            AdaptiveCloseMode::RootQuiesce if !source.root_drained() => {
                match source.try_root_fence() {
                    Some(Ok(_)) => Ok(true),
                    Some(Err(error)) => {
                        self.phase = ManagerPhase::Quarantined;
                        Err(AdaptiveManagerError::ProfiledExternal(error))
                    }
                    None => Ok(false),
                }
            }
            AdaptiveCloseMode::RootQuiesce => Ok(false),
            AdaptiveCloseMode::ProcessShutdown if source.shutdown_snapshot().is_none() => {
                match source.try_join() {
                    Some(Ok(_)) => Ok(true),
                    Some(Err(error)) => {
                        self.phase = ManagerPhase::Quarantined;
                        Err(AdaptiveManagerError::ProfiledExternal(error))
                    }
                    None => Ok(false),
                }
            }
            AdaptiveCloseMode::ProcessShutdown => Ok(false),
        }
    }

    /// Reported per-question limits are not a strict root-visit allowance.
    /// There is no aggregate-root exhaustion witness in this interface yet.
    pub fn profiled_reported_node_limit_reached(&self) -> Option<bool> {
        None
    }

    pub fn profiled_shutdown_receipt(
        &self,
    ) -> Result<ProfiledExternalShutdownReceipt, AdaptiveManagerError> {
        let source = self.researcher.profiled()?;
        let external_process_joined = source.shutdown_snapshot().is_some_and(|result| {
            result.as_ref().is_ok_and(|receipt| {
                receipt.cleanup_complete
                    && receipt.exit_observed
                    && receipt.stdout_drained
                    && receipt.stderr_drained
                    && !receipt.quarantined
                    && !receipt.ownership_lost
            })
        });
        Ok(ProfiledExternalShutdownReceipt {
            authority: self.registration.authority,
            source: source.current_source().clone(),
            native_root_fence: self.native_root_fence,
            external_root_fence: source.root_fence().cloned(),
            process_close_proof: source
                .process_close_proof()
                .map_err(AdaptiveManagerError::ProfiledExternal)?,
            native_process_joined: self.native_joined,
            native_physical_shutdown_confirmed: self.native_finished == Some(true),
            external_process_joined,
            auxiliary_shutdown_confirmed: self.driver.auxiliary_shutdown_confirmed()
                && self.goal_prepared.is_none()
                && !self.goal_continuation_pending,
            pending_tickets: self.registry.occupancy(),
            retained_outputs: self.roles.len()
                + self.accepted_roles.len()
                + self.retained_native_completions.len()
                + self.retained_native_permits.len()
                + self.retained_native_receipts.len()
                + self.deferred.len()
                + self.native.occupancy()
                + usize::from(self.profiled_binding.is_some())
                + usize::from(self.profiled_orphan.is_some())
                + usize::from(source.pending_dispatch().is_some()),
            quarantined: self.phase == ManagerPhase::Quarantined
                || self.native.is_quarantined()
                || source.is_quarantined()
                || !self.own.is_empty()
                || self.external.is_some()
                || self.foreign.is_some()
                || self.nnuec.is_some()
                || self.profiled_orphan.is_some()
                || !self.native_frontier_events.is_empty(),
            request_sequence_high_water: self.registry.next_request_sequence().saturating_sub(1),
            counters: self.counters,
            reported_work: self.driver.profiled_external_work(),
            retained_semantic_host_bytes: self.driver.profiled_retained_host_bytes(),
        })
    }
    pub fn run_profiled_root_until_observed(
        &mut self,
        until: Instant,
        canceled: &AtomicBool,
        observer: &mut dyn FnMut(&mut Self) -> Result<(), AdaptiveManagerError>,
    ) -> Result<ProfiledExternalShutdownReceipt, AdaptiveManagerError> {
        self.researcher.profiled()?;
        if self.shutdown_until.is_some() || self.phase != ManagerPhase::Running {
            return Err(AdaptiveManagerError::Invalid(
                "profiled root quiesce run admission",
            ));
        }
        self.close_mode = AdaptiveCloseMode::RootQuiesce;
        self.run_owner_until_observed(until, canceled, observer)?;
        self.profiled_shutdown_receipt()
    }
    pub fn run_profiled_until(
        &mut self,
        until: Instant,
        canceled: &AtomicBool,
    ) -> Result<ProfiledExternalShutdownReceipt, AdaptiveManagerError> {
        self.researcher.profiled()?;
        self.run_owner_until_observed(until, canceled, &mut |_| Ok(()))?;
        self.profiled_shutdown_receipt()
    }
    #[allow(clippy::result_large_err)]
    pub fn into_profiled_persistent_parts(
        self,
    ) -> Result<ProfiledExternalPersistentParts<Q, B, D>, AdaptivePersistentRejection<Q, B, D>>
    {
        self.into_profiled_settled_parts(AdaptiveCloseMode::RootQuiesce)
    }
    #[allow(clippy::result_large_err)]
    pub fn into_profiled_process_parts(
        self,
    ) -> Result<ProfiledExternalPersistentParts<Q, B, D>, AdaptivePersistentRejection<Q, B, D>>
    {
        self.into_profiled_settled_parts(AdaptiveCloseMode::ProcessShutdown)
    }
    #[allow(clippy::result_large_err)]
    fn into_profiled_settled_parts(
        self,
        mode: AdaptiveCloseMode,
    ) -> Result<ProfiledExternalPersistentParts<Q, B, D>, AdaptivePersistentRejection<Q, B, D>>
    {
        if !matches!(&self.researcher, ResearcherResources::ProfiledExternal(source) if matches!(source.as_ref(), researcher_resources::ProfiledExternalSource::V1(_)))
        {
            return Err(AdaptivePersistentRejection {
                error: AdaptiveManagerError::Invalid("profiled owner is not V1"),
                manager: self,
            });
        }
        let receipt = match self.profiled_shutdown_receipt() {
            Ok(receipt)
                if self.phase == ManagerPhase::Finished
                    && self.close_mode == mode
                    && match mode {
                        AdaptiveCloseMode::RootQuiesce => receipt.permits_root_reuse(),
                        AdaptiveCloseMode::ProcessShutdown => receipt.permits_archive_close(),
                    } =>
            {
                receipt
            }
            Ok(_) => {
                return Err(AdaptivePersistentRejection {
                    error: AdaptiveManagerError::PhysicalUnknown,
                    manager: self,
                });
            }
            Err(error) => {
                return Err(AdaptivePersistentRejection {
                    error,
                    manager: self,
                });
            }
        };
        let continuation = AdaptiveSequenceContinuation {
            authority: self.registration.authority,
            next_request: self.registry.next_request_sequence(),
            next_job: self.next_job,
            next_execution: self.next_execution,
            next_control: self.next_control,
        };
        let ResearcherResources::ProfiledExternal(external) = self.researcher else {
            unreachable!("receipt checked exact profiled source kind")
        };
        let researcher_resources::ProfiledExternalSource::V1(external) = *external else {
            unreachable!("version checked before extraction")
        };
        Ok(ProfiledExternalPersistentParts {
            native: self.native,
            external,
            driver: self.driver,
            continuation,
            receipt,
        })
    }

    #[allow(clippy::result_large_err)]
    pub fn into_profiled_v2_persistent_parts(
        self,
    ) -> Result<
        ProfiledExternalPersistentParts<Q, B, D, rz_experiments::RoventInputLockV2>,
        AdaptivePersistentRejection<Q, B, D>,
    > {
        self.into_profiled_v2_settled_parts(AdaptiveCloseMode::RootQuiesce)
    }
    #[allow(clippy::result_large_err)]
    pub fn into_profiled_v2_process_parts(
        self,
    ) -> Result<
        ProfiledExternalPersistentParts<Q, B, D, rz_experiments::RoventInputLockV2>,
        AdaptivePersistentRejection<Q, B, D>,
    > {
        self.into_profiled_v2_settled_parts(AdaptiveCloseMode::ProcessShutdown)
    }
    #[allow(clippy::result_large_err)]
    fn into_profiled_v2_settled_parts(
        self,
        mode: AdaptiveCloseMode,
    ) -> Result<
        ProfiledExternalPersistentParts<Q, B, D, rz_experiments::RoventInputLockV2>,
        AdaptivePersistentRejection<Q, B, D>,
    > {
        if !matches!(&self.researcher, ResearcherResources::ProfiledExternal(source) if matches!(source.as_ref(), researcher_resources::ProfiledExternalSource::V2(_)))
        {
            return Err(AdaptivePersistentRejection {
                error: AdaptiveManagerError::Invalid("profiled owner is not V2"),
                manager: self,
            });
        }
        let receipt = match self.profiled_shutdown_receipt() {
            Ok(receipt)
                if self.phase == ManagerPhase::Finished
                    && self.close_mode == mode
                    && match mode {
                        AdaptiveCloseMode::RootQuiesce => receipt.permits_root_reuse(),
                        AdaptiveCloseMode::ProcessShutdown => receipt.permits_archive_close(),
                    } =>
            {
                receipt
            }
            Ok(_) => {
                return Err(AdaptivePersistentRejection {
                    error: AdaptiveManagerError::PhysicalUnknown,
                    manager: self,
                });
            }
            Err(error) => {
                return Err(AdaptivePersistentRejection {
                    error,
                    manager: self,
                });
            }
        };
        let continuation = AdaptiveSequenceContinuation {
            authority: self.registration.authority,
            next_request: self.registry.next_request_sequence(),
            next_job: self.next_job,
            next_execution: self.next_execution,
            next_control: self.next_control,
        };
        let ResearcherResources::ProfiledExternal(external) = self.researcher else {
            unreachable!("receipt checked exact profiled source kind")
        };
        let researcher_resources::ProfiledExternalSource::V2(external) = *external else {
            unreachable!("version checked before extraction")
        };
        Ok(ProfiledExternalPersistentParts {
            native: self.native,
            external,
            driver: self.driver,
            continuation,
            receipt,
        })
    }

    #[allow(clippy::result_large_err)]
    pub fn new_profiled_external_retaining(
        registration: AdaptiveManagerRegistration,
        clock: ContractSystemClock,
        limits: AdaptiveExecutionLimits,
        sources: AdaptiveManagerSources<Q, B, D>,
        external: ProfiledExternalManagerSource,
        continuation: Option<AdaptiveSequenceContinuation>,
    ) -> Result<
        ProfiledExternalExecutionManager<Q, B, D>,
        ProfiledExternalManagerAdmissionRejection<Q, B, D>,
    > {
        Self::new_profiled_external_any_retaining(
            registration,
            clock,
            limits,
            sources,
            researcher_resources::ProfiledExternalSource::V1(external),
            continuation,
        )
        .map_err(|(error, sources, external, continuation)| {
            let researcher_resources::ProfiledExternalSource::V1(external) = external else {
                unreachable!("original V1 source retained")
            };
            ProfiledExternalManagerAdmissionRejection {
                error,
                sources,
                external,
                continuation,
            }
        })
    }

    #[allow(clippy::result_large_err)]
    pub fn new_profiled_external_v2_retaining(
        registration: AdaptiveManagerRegistration,
        clock: ContractSystemClock,
        limits: AdaptiveExecutionLimits,
        sources: AdaptiveManagerSources<Q, B, D>,
        external: ProfiledExternalManagerSource<rz_experiments::RoventInputLockV2>,
        continuation: Option<AdaptiveSequenceContinuation>,
    ) -> Result<
        ProfiledExternalExecutionManager<Q, B, D>,
        ProfiledExternalManagerAdmissionRejection<Q, B, D, rz_experiments::RoventInputLockV2>,
    > {
        Self::new_profiled_external_any_retaining(
            registration,
            clock,
            limits,
            sources,
            researcher_resources::ProfiledExternalSource::V2(external),
            continuation,
        )
        .map_err(|(error, sources, external, continuation)| {
            let researcher_resources::ProfiledExternalSource::V2(external) = external else {
                unreachable!("original V2 source retained")
            };
            ProfiledExternalManagerAdmissionRejection {
                error,
                sources,
                external,
                continuation,
            }
        })
    }

    #[allow(clippy::result_large_err, clippy::type_complexity)]
    fn new_profiled_external_any_retaining(
        registration: AdaptiveManagerRegistration,
        clock: ContractSystemClock,
        limits: AdaptiveExecutionLimits,
        sources: AdaptiveManagerSources<Q, B, D>,
        external: researcher_resources::ProfiledExternalSource,
        continuation: Option<AdaptiveSequenceContinuation>,
    ) -> Result<
        ProfiledExternalExecutionManager<Q, B, D>,
        (
            AdaptiveManagerError,
            AdaptiveManagerSources<Q, B, D>,
            researcher_resources::ProfiledExternalSource,
            Option<AdaptiveSequenceContinuation>,
        ),
    > {
        let prepared = (|| -> Result<_, AdaptiveManagerError> {
            limits.validate()?;
            registration
                .endpoints
                .validate()
                .map_err(|_| AdaptiveManagerError::Invalid("profiled external endpoint policy"))?;
            let source = external.current_source();
            let endpoint = if registration.endpoints.execution_use == CpuExecutionUse::Collection {
                registration.endpoints.cpu_t
            } else {
                registration.endpoints.cpu_r
            };
            if !sources.own_workers.is_empty()
                || sources.external.is_some()
                || registration.native_cpu_node_limit != 0
                || endpoint != CpuEndpoint::ExternalUci
                || registration.authority != external.root_authority()
                || source.authority() != registration.authority
                || clock.domain() != external.root_runtime_clock().domain()
                || clock.origin() != external.root_runtime_clock().origin()
                || registration.hard_deadline != external.root_original_context().control.deadline
                || ![128, 256, 512].contains(&registration.token_budget)
                || sources.native.pending_capacity() != limits.role_tickets_max
                || sources.native.command_capacity() != limits.worker_commands_max
                || external.pending_dispatch().is_some()
                || external.is_quarantined()
            {
                return Err(AdaptiveManagerError::Invalid(
                    "profiled external actual source/root/resource registration",
                ));
            }
            source.validate_current().map_err(|error| {
                AdaptiveManagerError::Semantic(format!(
                    "profiled external original source: {error:?}"
                ))
            })?;
            if let Some(previous) = &continuation {
                if registration.authority.epoch != previous.authority.epoch
                    || registration.authority.implementation != previous.authority.implementation
                    || registration.authority.root.0 <= previous.authority.root.0
                    || registration.authority.game.0 < previous.authority.game.0
                    || registration.first_request_sequence != previous.next_request
                {
                    return Err(AdaptiveManagerError::Invalid(
                        "profiled external original continuation",
                    ));
                }
            }
            let role_call_limit = sources.driver.root_role_call_limit()?;
            if role_call_limit == Some(0) {
                return Err(AdaptiveManagerError::Invalid(
                    "profiled root role call limit",
                ));
            }
            let cpu_clock = InstantClock::new(clock.domain(), clock.origin());
            let deadline = cpu_clock
                .deadline_at(registration.hard_deadline)
                .map_err(|_| AdaptiveManagerError::Deadline)?;
            if deadline != external.root_hard_deadline() {
                return Err(AdaptiveManagerError::Invalid(
                    "profiled original hard deadline",
                ));
            }
            let cleanup_boundary = Some(CleanupBoundary::admit(
                registration.hard_deadline,
                external.root_original_context().shutdown_deadline,
            )?);
            // Fallible root replacement retains these same native/external
            // owners on every error; it never issues a replacement source.
            sources.native.replace_root(registration.authority)?;
            let native_ready = match sources.native.ready_clock() {
                Some(ready)
                    if ready.domain() == clock.domain() && ready.origin() == clock.origin() =>
                {
                    true
                }
                Some(_) => {
                    return Err(AdaptiveManagerError::Invalid(
                        "profiled native original clock",
                    ));
                }
                None => false,
            };
            let mut registry = RoleTicketRegistry::new_with_capacity(
                registration.authority,
                registration.token_budget,
                registration.policy,
                registration.first_request_sequence,
                limits.role_tickets_max,
            )?;
            let native_verifier_registered =
                if let Some(verifier) = sources.native.native_acceptance_verifier() {
                    registry.register_native_acceptance_verifier(verifier)?;
                    true
                } else {
                    false
                };
            let machine = RefinementMachine::new(
                registration.authority,
                deadline,
                registration.token_budget,
            )?;
            let ready = ReadySelector::new(limits.waiting_jobs_max)
                .map_err(|_| AdaptiveManagerError::Capacity("profiled ready selector"))?;
            let costs = CostEstimator::new(4096)
                .map_err(|_| AdaptiveManagerError::Capacity("profiled cost estimator"))?;
            Ok((
                cpu_clock,
                deadline,
                native_ready,
                registry,
                native_verifier_registered,
                machine,
                ready,
                costs,
                role_call_limit,
                cleanup_boundary,
            ))
        })();
        let (
            cpu_clock,
            deadline,
            native_ready,
            registry,
            native_verifier_registered,
            machine,
            ready,
            costs,
            role_call_limit,
            cleanup_boundary,
        ) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                return Err((error, sources, external, continuation));
            }
        };
        let (next_job, next_execution, next_control) =
            continuation.as_ref().map_or((1, 1, 1), |value| {
                (value.next_job, value.next_execution, value.next_control)
            });
        Ok(ProfiledExternalExecutionManager {
            inner: Self::assemble(
                registration,
                clock,
                limits,
                sources,
                admitted::ManagerAssembly {
                    cpu_clock,
                    deadline,
                    native_ready,
                    registry,
                    native_verifier_registered,
                    machine,
                    ready,
                    costs,
                    researcher: ResearcherResources::ProfiledExternal(Box::new(external)),
                    role_call_limit,
                    next_job,
                    next_execution,
                    next_control,
                    cleanup_boundary,
                },
            ),
        })
    }
}
