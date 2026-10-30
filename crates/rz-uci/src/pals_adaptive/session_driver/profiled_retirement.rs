//! Durable external evidence leaves Search only through the existing bounded
//! archive worker. The root manager has already returned its actual fences;
//! this session poll performs no filesystem work and grants no search right.

use super::*;
use rz_search::pals::engine::adaptive_port::archive_worker::PersistProfiledRetirementWork;

impl<B> AdaptiveSessionDriver<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    /// A failed evidence transfer never skips physical native/pipe shutdown.
    /// Even after those owners join, its original Engine/payload stays retained
    /// and cannot be mistaken for successful archive or process closure.
    pub(super) fn finish_retained_profiled_io(
        &self,
        state: &mut SessionState<B>,
        retained: SessionState<B>,
        until: Instant,
    ) -> Result<SessionState<B>, SearchSessionFailure> {
        match retained {
            SessionState::ProfiledRetirementPending {
                mut owners,
                mut worker,
                transfer,
                authority,
            } => {
                let physical = self.finish_idle_physical_owners(&mut owners, until);
                transfer.cancel();
                worker.request_shutdown();
                let retained = match worker.try_receive(&transfer) {
                    Ok(Some(completed)) => match completed
                        .install_with_profiled_retirement(authority, transfer.nonce())
                    {
                        Err(quarantine) => SessionState::ArchiveQuarantined {
                            owners,
                            worker,
                            quarantine: Box::new(quarantine),
                        },
                        Ok((owner, _receipt, payload)) => {
                            // Cancellation normally makes install quarantine.
                            // If that contract ever changes, keep the complete
                            // owner/payload rather than silently dropping it.
                            match legacy_archive_engine(owner) {
                                Ok(engine) => owners.engine = Some(engine),
                                Err(owner) => owners.unexpected_archive_owner = Some(owner),
                            }
                            SessionState::ProfiledRetirementRetained {
                                owners,
                                worker,
                                payload: Box::new(payload),
                            }
                        }
                    },
                    _ => SessionState::ProfiledRetirementPending {
                        owners,
                        worker,
                        transfer,
                        authority,
                    },
                };
                *state = retained;
                Err(physical
                    .err()
                    .unwrap_or_else(SearchSessionFailure::physical_completion_unknown))
            }
            SessionState::ProfiledRetirementRetained {
                mut owners,
                mut worker,
                payload,
            } => {
                let physical = self.finish_idle_physical_owners(&mut owners, until);
                worker.request_shutdown();
                *state = SessionState::ProfiledRetirementRetained {
                    owners,
                    worker,
                    payload,
                };
                Err(physical
                    .err()
                    .unwrap_or_else(SearchSessionFailure::physical_completion_unknown))
            }
            SessionState::ArchiveQuarantined {
                mut owners,
                mut worker,
                quarantine,
            } if quarantine.retains_profiled_retirement() => {
                let physical = self.finish_idle_physical_owners(&mut owners, until);
                worker.request_shutdown();
                *state = SessionState::ArchiveQuarantined {
                    owners,
                    worker,
                    quarantine,
                };
                Err(physical
                    .err()
                    .unwrap_or_else(SearchSessionFailure::physical_completion_unknown))
            }
            SessionState::ArchiveJoinedPending { mut owners, worker }
                if owners.researcher_source.is_profiled_external() =>
            {
                if let Err(error) = self.finish_idle_physical_owners(&mut owners, until) {
                    *state = SessionState::ArchiveJoinedPending { owners, worker };
                    return Err(error);
                }
                join_archive_lifecycle(state, owners, worker, until).map(SessionState::Idle)
            }
            other => Ok(other),
        }
    }

    /// Pinless summaries only. Original packet/view/Store owners remain in the
    /// one pending payload until the worker's opaque commit has been checked.
    pub fn profiled_retirement_commits(&self) -> Result<Vec<serde_json::Value>, ContractError> {
        self.profiled_retirement_commits
            .lock()
            .map(|commits| commits.iter().cloned().collect())
            .map_err(|_| poisoned("profiled retirement commit observations"))
    }

    pub(super) fn persist_profiled_retirement(
        &self,
        state: &mut SessionState<B>,
        mut owners: Box<IdleOwners<B>>,
        original_until: Instant,
    ) -> Result<Box<IdleOwners<B>>, SearchSessionFailure> {
        let Some(retirement) = owners.profiled_retirement.as_ref() else {
            return Ok(owners);
        };
        // The source keeps the original private UCI Start admission across the
        // manager handoff. Archival cleanup cannot replace that root's clock
        // with a startup timeout or a newly registered execution window.
        let original_cleanup =
            match &owners.researcher_source {
                SessionResearcherSource::ProfiledExternal(source) => source
                    .retained_root_admission()
                    .filter(|root| {
                        source
                            .retained_root_source()
                            .is_some_and(|actual| actual.same_source(retirement.source()))
                            && root.authority() == retirement.authority()
                            && self.rovent.as_ref().is_some_and(|profile| {
                                root.startup_lock_digest() == profile.input_lock
                            })
                    })
                    .map(|root| root.original_context().shutdown_deadline),
                SessionResearcherSource::ProfiledExternalV2(source) => source
                    .retained_root_admission_v2()
                    .filter(|root| {
                        source
                            .retained_root_source()
                            .is_some_and(|actual| actual.same_source(retirement.source()))
                            && root.authority() == retirement.authority()
                            && self.rovent.as_ref().is_some_and(|profile| {
                                root.startup_lock_digest() == profile.input_lock
                            })
                    })
                    .map(|root| root.original_context().shutdown_deadline),
                SessionResearcherSource::Legacy => None,
            };
        let Some(original_cleanup) = original_cleanup else {
            *state = SessionState::Idle(owners);
            return Err(failure(
                "RoventRetirementRoot",
                "original UCI root admission differs",
            ));
        };
        let Some(sealed_cleanup) = retirement
            .original_cleanup_deadline()
            .filter(|deadline| *deadline == original_cleanup)
        else {
            *state = SessionState::Idle(owners);
            return Err(failure(
                "RoventRetirementDeadline",
                "retirement has no matching original root cleanup seal",
            ));
        };
        let original_until = original_until.min(sealed_cleanup);
        let mut sink_slot = match self.profiled_retirement_sink.lock() {
            Ok(slot) => slot,
            Err(_) => {
                *state = SessionState::Idle(owners);
                return Err(failure("RoventRetirementSink", "sink owner mutex poisoned"));
            }
        };
        let Some(sink) = sink_slot.as_ref() else {
            *state = SessionState::Idle(owners);
            return Err(failure(
                "RoventRetirementSink",
                "original sink owner unavailable",
            ));
        };
        if sink.admission_closed() {
            *state = SessionState::Idle(owners);
            return Err(failure(
                "RoventRetirementSink",
                "original sink admission is closed",
            ));
        }
        let now = Instant::now();
        let Some(persist_until) = now.checked_add(sink.max_persist_duration()) else {
            *state = SessionState::Idle(owners);
            return Err(failure(
                "RoventRetirementDeadline",
                "registered window overflow",
            ));
        };
        let until = original_until.min(persist_until);
        if now >= until {
            *state = SessionState::Idle(owners);
            return Err(failure(
                "RoventRetirementDeadline",
                "original cleanup expired",
            ));
        }
        let max_io_bytes = sink.max_io_bytes_per_root();
        let authority = retirement.authority();
        let engine = owners.engine.as_ref().expect("same idle semantic owner");
        // adaptive_archive_authority prepares a *later* root namespace. This
        // evidence belongs to the just-retired root and must never manufacture
        // that next-root authority to enter the I/O worker.
        if engine.last_adaptive_authority() != Some(authority) {
            *state = SessionState::Idle(owners);
            return Err(failure(
                "RoventRetirementAuthority",
                "retirement is not the actual last registered root",
            ));
        }
        if let Err(error) = retirement.validate_owner(engine.stores()) {
            *state = SessionState::Idle(owners);
            return Err(SearchSessionFailure::debug("RoventRetirementStore", &error));
        }
        let signal = owners.native.signal();
        let mut worker = match ArchiveWorker::start_with_notifier(move || signal.notify()) {
            Ok(worker) => worker,
            Err(error) => {
                *state = SessionState::Idle(owners);
                return Err(SearchSessionFailure::debug(
                    "RoventRetirementWorker",
                    &error,
                ));
            }
        };
        let engine = owners.engine.take().expect("same idle semantic owner");
        let retirement = owners
            .profiled_retirement
            .take()
            .expect("checked pending slot");
        let sink = sink_slot.take().expect("checked original sink slot");
        let transfer = match worker.submit(
            ArchiveOwner::Engine(*engine),
            authority,
            ArchiveWork {
                operation: ArchiveOperation::PersistProfiledRetirement {
                    payload: PersistProfiledRetirementWork::new(retirement, sink),
                },
                deadline: until,
                max_io_bytes,
            },
        ) {
            Ok(transfer) => transfer,
            Err(rejected) => {
                let failure =
                    SearchSessionFailure::debug("RoventRetirementAdmission", &rejected.error);
                let unexpected = match legacy_archive_engine(rejected.owner) {
                    Ok(engine) => {
                        owners.engine = Some(engine);
                        false
                    }
                    Err(owner) => {
                        owners.unexpected_archive_owner = Some(owner);
                        true
                    }
                };
                let ArchiveOperation::PersistProfiledRetirement { payload } =
                    rejected.work.operation
                else {
                    unreachable!("the original submitted operation is returned unchanged")
                };
                if unexpected || payload.commit().is_some() {
                    worker.request_shutdown();
                    *state = SessionState::ProfiledRetirementRetained {
                        owners,
                        worker,
                        payload: Box::new(payload),
                    };
                    return Err(SearchSessionFailure::physical_completion_unknown());
                }
                let (retirement, sink, _) = payload.into_parts();
                owners.profiled_retirement = Some(retirement);
                *sink_slot = Some(sink);
                owners = match join_archive_lifecycle(state, owners, worker, original_until) {
                    Ok(owners) => owners,
                    Err(_) => return Err(failure.unknown_if_unclosed()),
                };
                *state = SessionState::Idle(owners);
                return Err(failure);
            }
        };
        loop {
            match worker.try_receive(&transfer) {
                Ok(Some(completed)) => {
                    match completed.install_with_profiled_retirement(authority, transfer.nonce()) {
                        Ok((owner, receipt, payload)) => {
                            let is_engine = matches!(owner, ArchiveOwner::Engine(_));
                            match legacy_archive_engine(owner) {
                                Ok(engine) => owners.engine = Some(engine),
                                Err(owner) => owners.unexpected_archive_owner = Some(owner),
                            }
                            let valid = is_engine
                                && payload.commit().is_some_and(|commit| {
                                    commit
                                        .matches_retirement(
                                            owners
                                                .engine
                                                .as_ref()
                                                .expect("returned owner")
                                                .stores(),
                                            payload.retirement(),
                                        )
                                        .is_ok()
                                });
                            if !valid {
                                worker.request_shutdown();
                                *state = SessionState::ProfiledRetirementRetained {
                                    owners,
                                    worker,
                                    payload: Box::new(payload),
                                };
                                return Err(failure(
                                    "RoventRetirementAcknowledgement",
                                    "returned commit does not match the original Store and retirement",
                                ).unknown_if_unclosed());
                            }
                            // Observation methods consume an already verified
                            // in-process proof. They do no filesystem readback.
                            let observation =
                                payload.commit().expect("checked commit").observation();
                            let observations = self.profiled_retirement_commits.lock();
                            let receipts = self.archive_receipts.lock();
                            let (Ok(mut observations), Ok(mut receipts)) = (observations, receipts)
                            else {
                                worker.request_shutdown();
                                *state = SessionState::ProfiledRetirementRetained {
                                    owners,
                                    worker,
                                    payload: Box::new(payload),
                                };
                                return Err(failure(
                                    "RoventRetirementObservation",
                                    "verified commit observations could not be retained",
                                )
                                .unknown_if_unclosed());
                            };
                            if observations.len() == 64 {
                                observations.pop_front();
                            }
                            if receipts.len() == 64 {
                                receipts.pop_front();
                            }
                            observations.push_back(observation);
                            receipts.push_back(receipt);
                            let (retirement, sink, commit) = payload.into_parts();
                            *sink_slot = Some(sink);
                            // These pins are released only after same-sink,
                            // same-Store and original retirement acknowledgement.
                            drop(retirement);
                            drop(commit);
                            return join_archive_lifecycle(state, owners, worker, original_until);
                        }
                        Err(quarantine) => {
                            let error = SearchSessionFailure::debug(
                                "RoventRetirementPersistence",
                                &quarantine.reason,
                            )
                            .unknown_if_unclosed();
                            worker.request_shutdown();
                            *state = SessionState::ArchiveQuarantined {
                                owners,
                                worker,
                                quarantine: Box::new(quarantine),
                            };
                            return Err(error);
                        }
                    }
                }
                Ok(None) if Instant::now() < until => {
                    let version = owners.native.signal().version();
                    owners
                        .native
                        .signal()
                        .wait_changed(version, Duration::from_millis(1));
                }
                completion => {
                    let error = match completion {
                        Err(error) => {
                            SearchSessionFailure::debug("RoventRetirementCompletion", &error)
                                .unknown_if_unclosed()
                        }
                        Ok(None) => failure(
                            "RoventRetirementDeadline",
                            "original persistence cleanup expired",
                        )
                        .unknown_if_unclosed(),
                        Ok(Some(_)) => unreachable!("completed transfer handled above"),
                    };
                    transfer.cancel();
                    worker.request_shutdown();
                    *state = SessionState::ProfiledRetirementPending {
                        owners,
                        worker,
                        transfer,
                        authority,
                    };
                    return Err(error);
                }
            }
        }
    }
}
