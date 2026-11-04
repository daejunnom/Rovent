//! Common-node accounting of the original native worker receipts. No worker,
//! model, Store or controller is created by these optional manager hooks.
use super::*;

// There is at most one original unsettled packet per actual worker while the
// semantic owner is away; new Job/Resume admission is then blocked. Retaining
// this packet does not enlarge the manager's 64/8/16 Job/command/role ceilings.
#[allow(clippy::large_enum_variant)]
pub(super) enum NativeFrontierEvent {
    Settled {
        reservation: NativeReservation,
        report: CpuSliceReport,
        accepted: SettledCpuSlice,
    },
    Released {
        reservation: NativeReservation,
        accepted: ReleasedCpuReservation,
    },
    Unknown {
        reservation: NativeReservation,
        recorded: bool,
    },
}
impl NativeFrontierEvent {
    fn grant(&self) -> SliceGrant {
        match self {
            Self::Settled { reservation, .. }
            | Self::Released { reservation, .. }
            | Self::Unknown { reservation, .. } => reservation.grant(),
        }
    }
}

/// Separate opt-in receipt. Legacy V5 shutdown bytes are not extended or
/// relabelled as Rovent common-node evidence.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RoventNativeAccountingObservation {
    pub pending_receipts: usize,
    pub active_reservations: usize,
    pub unknown_reservations: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RoventReviewLeaseObservation {
    pub last: Option<AdaptiveReviewLeaseAdmission>,
    pub declared_role_ticket_ceiling: usize,
    pub effective_role_ticket_ceiling: usize,
    pub blocked_preparations: u64,
    pub terminal_capacity: bool,
}
impl RoventReviewLeaseObservation {
    pub(super) fn new(declared: usize) -> Self {
        Self {
            declared_role_ticket_ceiling: declared,
            effective_role_ticket_ceiling: declared,
            ..Self::default()
        }
    }
    pub(super) fn record_blocked(
        &mut self,
        current: usize,
        capacity: usize,
        required_new: usize,
        replacement_reserve: usize,
    ) {
        self.last = Some(AdaptiveReviewLeaseAdmission {
            current,
            capacity,
            required_new_leases: required_new,
            frontier_replacement_reserve: replacement_reserve,
            frontier_owner_present: true,
            may_prepare: false,
        });
        self.effective_role_ticket_ceiling = self.declared_role_ticket_ceiling.min(
            capacity
                .saturating_sub(1)
                .saturating_sub(replacement_reserve),
        );
        self.blocked_preparations = self.blocked_preparations.saturating_add(1);
    }
}

impl<Q, B, D> AdaptiveExecutionManager<Q, B, D>
where
    Q: Clone + Send + Sync + 'static,
    B: TicketNativeBackend<ReviewView, Q>,
    B::Output: ProposerRankingOutput,
    D: AdaptiveSearchDriver<Q>,
{
    pub fn rovent_review_leases(
        &self,
    ) -> Result<RoventReviewLeaseObservation, AdaptiveManagerError> {
        let mut observation = self.review_leases;
        if let Some(actual) = self.driver.review_lease_admission()? {
            observation.last = Some(actual);
            if actual.frontier_owner_present {
                observation.effective_role_ticket_ceiling = self.role_pending_limit();
            }
        }
        Ok(observation)
    }

    pub(super) fn review_lease_return_can_progress(&self) -> bool {
        !self.driver.semantic_owner_available()
            || self.roles.iter().any(|role| !role.service_completed)
            || !self.acknowledgements.is_empty()
            || self.foreign.is_some()
            || self.own.iter().any(|slot| {
                slot.binding.as_ref().is_some_and(|held| {
                    held.reciprocal.is_none()
                        && !held.resume_pending
                        && held.pending_slice.is_none()
                        && !slot.joined
                })
            })
    }

    pub fn rovent_native_accounting(&self) -> RoventNativeAccountingObservation {
        RoventNativeAccountingObservation {
            pending_receipts: self.native_frontier_events.len(),
            active_reservations: self
                .own
                .iter()
                .filter(|slot| {
                    slot.binding
                        .as_ref()
                        .is_some_and(|held| held.native_frontier.is_some())
                })
                .count(),
            unknown_reservations: self
                .native_frontier_events
                .iter()
                .filter(|event| matches!(event, NativeFrontierEvent::Unknown { .. }))
                .count(),
        }
    }

    pub(super) fn pending_frontier_for(&self, job: CpuJobId) -> bool {
        self.native_frontier_events
            .iter()
            .any(|event| event.grant().job == job)
    }

    /// A semantic replay may wait only on actual, still-owned native credit.
    /// A reservation snapshot alone is not a future settlement guarantee: its
    /// original worker/binding must still be present and its usage known.
    pub(super) fn semantic_replay_can_wait(
        &self,
        error: &AdaptiveManagerError,
    ) -> Result<bool, AdaptiveManagerError> {
        if !matches!(
            error,
            AdaptiveManagerError::Frontier(FrontierError::NodesReserved)
        ) || self.researcher_source_kind() != AdaptiveResearcherSourceKind::Native
            || self.phase != ManagerPhase::Running
            || !self.quarantined_cpu_receipts.is_empty()
            || self
                .native_frontier_events
                .iter()
                .any(|event| matches!(event, NativeFrontierEvent::Unknown { .. }))
        {
            return Ok(false);
        }
        let actual = self.native_ledger()?.registration_observation()?;
        if actual.reserved_native_nodes == 0 || actual.quarantined_jobs != 0 {
            return Ok(false);
        }
        for slot in &self.own {
            let Some(binding) = slot.binding.as_ref() else {
                continue;
            };
            if slot.joined
                || slot.panicked
                || slot.worker.is_none()
                || binding.native_frontier.is_none()
                || binding.resume_pending
            {
                continue;
            }
            let (_, reserved, unknown) = self.native_ledger()?.usage(binding.envelope.id)?;
            if reserved != 0 && !unknown {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(super) fn native_admission_result(
        &mut self,
        result: Result<(), AdaptiveManagerError>,
    ) -> Result<bool, AdaptiveManagerError> {
        match result {
            Ok(()) => Ok(true),
            Err(AdaptiveManagerError::AdmissionClosed) => Ok(false),
            Err(AdaptiveManagerError::Frontier(FrontierError::NodesReserved)) => Ok(false),
            Err(AdaptiveManagerError::Frontier(FrontierError::NodeBudget)) => {
                // Common visits include checked Rules transitions and are not
                // relabelled as exhaustion of the native-only CPU ledger.
                self.begin_root_quiesce(self.cleanup_until()?)?;
                Ok(true)
            }
            Err(error) => Err(error),
        }
    }

    pub(super) fn retain_frontier_settlement(
        &mut self,
        reservation: NativeReservation,
        report: CpuSliceReport,
        accepted: SettledCpuSlice,
    ) {
        // While the semantic owner is away no new Job/Resume can pass its
        // reserve callback. There is at most one packet per real worker.
        self.native_frontier_events
            .push_back(NativeFrontierEvent::Settled {
                reservation,
                report,
                accepted,
            });
    }

    pub(super) fn retain_frontier_release(
        &mut self,
        reservation: NativeReservation,
        accepted: ReleasedCpuReservation,
    ) {
        self.native_frontier_events
            .push_back(NativeFrontierEvent::Released {
                reservation,
                accepted,
            });
    }

    pub(super) fn retain_frontier_unknown(&mut self, reservation: NativeReservation) {
        self.native_frontier_events
            .push_back(NativeFrontierEvent::Unknown {
                reservation,
                recorded: false,
            });
    }

    pub(super) fn release_frontier_before_enqueue(
        &mut self,
        reservation: Option<NativeReservation>,
        rejected: &CpuAdmissionRejection,
    ) -> Result<(), AdaptiveManagerError> {
        let accepted = match self
            .native_ledger_mut()?
            .release_unadmitted_attested(rejected)
        {
            Ok(accepted) => accepted,
            Err(error) => {
                if let Some(reservation) = reservation {
                    self.retain_frontier_unknown(reservation);
                }
                return Err(error.into());
            }
        };
        if let Some(reservation) = reservation {
            self.retain_frontier_release(reservation, accepted);
            self.flush_frontier_native()?;
        }
        Ok(())
    }

    pub(super) fn quarantine_frontier_slot(&mut self, slot: usize) {
        if let Some(reservation) = self.own[slot]
            .binding
            .as_mut()
            .and_then(|held| held.native_frontier.take())
        {
            self.retain_frontier_unknown(reservation);
        }
    }

    pub(super) fn flush_frontier_native(&mut self) -> Result<bool, AdaptiveManagerError> {
        let mut changed = false;
        let mut index = 0;
        while self.driver.semantic_owner_available() && index < self.native_frontier_events.len() {
            let mut event = self
                .native_frontier_events
                .remove(index)
                .expect("bounded receipt index");
            let result = match &mut event {
                NativeFrontierEvent::Settled {
                    reservation,
                    report,
                    accepted,
                } => self
                    .driver
                    .frontier_native_settled(reservation, report, accepted)
                    .map(|_| ()),
                NativeFrontierEvent::Released {
                    reservation,
                    accepted,
                } => self
                    .driver
                    .frontier_native_released(reservation, accepted)
                    .map(|_| ()),
                NativeFrontierEvent::Unknown {
                    reservation,
                    recorded,
                } => {
                    if *recorded {
                        self.native_frontier_events.insert(index, event);
                        index += 1;
                        continue;
                    }
                    self.driver.frontier_native_unknown(reservation).map(|()| {
                        *recorded = true;
                    })
                }
            };
            if let Err(error) = result {
                self.native_frontier_events.insert(index, event);
                return Err(error);
            }
            changed = true;
            if matches!(event, NativeFrontierEvent::Unknown { .. }) {
                // Keep the original reservation as quarantine custody even
                // after its controller has recorded unknown usage.
                self.native_frontier_events.insert(index, event);
                index += 1;
            }
        }
        Ok(changed)
    }
}
