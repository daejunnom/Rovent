//! Original /3 actor and physical teacher worker custody.
//! A worker join alone is not proof of OS process retirement or a label.
use super::actor_binding_v3::BoundCollectionTeacherActorV3;
use super::quarantine_v3::{
    self, ParkedTeacherV3, TeacherRecoveryCustodianV3, TeacherRecoveryKeyV3,
    TeacherRecoveryReservationV3,
};
use super::*;
use crate::pals_adaptive::{RoventCollectionRootAdmissionV3, RoventTeacherScopeCapabilityV3};
use rz_search::external_cpu::process_scope::CollectionProcessMemberWitness;

/// The original actor retains source descriptors and the exact /3 admission.
pub struct RoventCollectionTeacherPhysicalOwnerV3 {
    actor: Option<BoundCollectionTeacherActorV3>,
    worker: CollectionTeacherWorker,
    member_witness: CollectionProcessMemberWitness,
    verified_no_child: bool,
}

impl RoventCollectionTeacherPhysicalOwnerV3 {
    pub fn admission(&self) -> &RoventCollectionRootAdmissionV3 {
        self.actor
            .as_ref()
            .expect("retained /3 teacher actor")
            .admission()
    }
    pub fn worker(&self) -> &CollectionTeacherWorker {
        &self.worker
    }
    pub fn member_witness(&self) -> &CollectionProcessMemberWitness {
        &self.member_witness
    }
    fn actor(&self) -> &BoundCollectionTeacherActorV3 {
        self.actor.as_ref().expect("retained /3 teacher actor")
    }
    fn submit(&mut self, job: TeacherJob) -> Result<TeacherDispatch, CheckerError> {
        let root = self.admission().root_admission();
        let input = job.input_scope;
        if !root
            .original_context()
            .accepts()
            .map_err(|_| invalid("teacher /3 root unavailable"))?
            || input.authority != root.authority()
            || input.root_admission
                != root
                    .descriptor()
                    .digest()
                    .map_err(|_| invalid("teacher /3 root digest"))?
            || input.input_lock != root.startup_lock_digest()
            || input.root_budget != self.admission().root_budget()
            || input.frozen_actor != self.actor().actor_digest()
            || job.original_deadline.clock != root.hard_deadline().clock
            || job.original_deadline.at > root.hard_deadline().at
        {
            return Err(invalid("teacher /3 job differs from original live root"));
        }
        self.actor()
            .source()
            .prepared
            .require_query(job.constraint, job.root_moves.is_some())?;
        self.worker.submit(job)
    }
    fn validate_completion(
        &self,
        completion: &TeacherCompletion,
        dispatch: &TeacherDispatch,
    ) -> Result<(), CheckerError> {
        let root = self.admission().root_admission();
        if !root
            .original_context()
            .accepts()
            .map_err(|_| invalid("teacher /3 root unavailable"))?
            || !completion.matches_submission(dispatch)
            || dispatch.input_scope().input_lock != root.startup_lock_digest()
            || dispatch.input_scope().root_budget != self.admission().root_budget()
            || dispatch.input_scope().authority != root.authority()
            || dispatch.input_scope().root_admission
                != root
                    .descriptor()
                    .digest()
                    .map_err(|_| invalid("teacher /3 root digest"))?
            || dispatch.input_scope().frozen_actor != self.actor().actor_digest()
        {
            return Err(invalid(
                "teacher /3 completion lacks original private dispatch",
            ));
        }
        if let Ok(outcome) = completion.result() {
            let report = &outcome.report;
            let declared = self.actor().source().prepared.profile();
            let attempt = completion
                .attempt()
                .ok_or(invalid("teacher /3 attempt absent"))?;
            let external = attempt
                .external
                .as_ref()
                .ok_or(invalid("teacher /3 physical receipt absent"))?;
            if report.profile != declared.engine
                || report.purpose != ExternalEnginePurpose::Teacher
                || report.raw.identity != declared.identity
                || report.raw.observed_uci.name != declared.expected_uci_name
                || report.requested != dispatch.constraint()
                || completion.issued_constraint() != Some(report.issued)
                || completion.issued_go() != Some(report.issued_go.as_str())
                || completion.go_issue() != rz_search::external_cpu::ExternalGoIssue::Issued
                || !report.readiness_barrier_completed
                || report.raw.request_id == 0
                || external.request_id != report.raw.request_id
                || external.partial_report.as_ref() != Some(&report.raw)
                || attempt.elapsed != report.raw.elapsed
                || external.process.process_identity != Some(dispatch.process_identity())
                || !reusable(external.process)
            {
                return Err(invalid(
                    "teacher /3 raw/private physical provenance mismatch",
                ));
            }
        }
        Ok(())
    }
}

impl Drop for RoventCollectionTeacherPhysicalOwnerV3 {
    fn drop(&mut self) {
        let closed = self.worker.process_after_join().is_some_and(|p| {
            p.exit_observed
                && p.stdout_drained
                && p.stderr_drained
                && p.cleanup_complete
                && !p.quarantined
                && !p.ownership_lost
        });
        if !closed && !self.verified_no_child {
            if let Some(actor) = self.actor.take() {
                std::mem::forget(actor);
            }
        }
    }
}

#[derive(Debug)]
pub enum BoundTeacherSpawnCauseV3 {
    Root(rz_contracts::ContractError),
    Scope(crate::pals_adaptive::AdaptiveCollectionError),
    Source(rz_eval::error::BackendError),
    Io(std::io::Error),
    Checker(CheckerError),
}
pub enum BoundTeacherSpawnFailureV3 {
    BeforeSpawn {
        cause: BoundTeacherSpawnCauseV3,
        actor: BoundCollectionTeacherActorV3,
    },
    AfterSpawn {
        cause: BoundTeacherSpawnCauseV3,
        lifecycle: RoventCollectionTeacherLifecycleV3,
        recovery_key: TeacherRecoveryKeyV3,
    },
}

/// Start only under the actual V3 scope guard and pre-reserved recovery slot.
pub fn spawn_bound_collection_teacher_v3(
    actor: BoundCollectionTeacherActorV3,
    permit: &CollectionProcessPermit,
    scope: &RoventTeacherScopeCapabilityV3,
    startup_deadline: Instant,
    cleanup_ceiling: Instant,
    custodian: &mut TeacherRecoveryCustodianV3,
) -> Result<(RoventCollectionTeacherLifecycleV3, TeacherRecoveryKeyV3), BoundTeacherSpawnFailureV3>
{
    let _scope_guard = match scope.begin_spawn(actor.admission(), permit) {
        Ok(value) => value,
        Err(error) => {
            return Err(BoundTeacherSpawnFailureV3::BeforeSpawn {
                cause: BoundTeacherSpawnCauseV3::Scope(error),
                actor,
            });
        }
    };
    let (reservation, recovery_key) = match TeacherRecoveryReservationV3::reserve() {
        Ok(value) => value,
        Err(error) => {
            return Err(BoundTeacherSpawnFailureV3::BeforeSpawn {
                cause: BoundTeacherSpawnCauseV3::Checker(error),
                actor,
            });
        }
    };
    if let Err(error) = custodian.register(&recovery_key) {
        return Err(BoundTeacherSpawnFailureV3::BeforeSpawn {
            cause: BoundTeacherSpawnCauseV3::Checker(error),
            actor,
        });
    }
    let original_cleanup = actor
        .admission()
        .root_admission()
        .original_context()
        .shutdown_deadline;
    let created = (|| -> Result<CollectionTeacherWorker, BoundTeacherSpawnCauseV3> {
        reservation
            .validate_before_spawn()
            .map_err(BoundTeacherSpawnCauseV3::Checker)?;
        let admission = actor.admission();
        let root = admission.root_admission();
        let control = actor.root_control();
        let context = root.original_context();
        let source = actor.source();
        let lock = root
            .owner()
            .v3()
            .ok_or(BoundTeacherSpawnCauseV3::Checker(invalid(
                "teacher /3 original lock absent",
            )))?;
        if !Arc::ptr_eq(&control.cancellation, &context.control.cancellation)
            || !source
                .prepared
                .original_lock_owner_v3()
                .is_some_and(|owner| Arc::ptr_eq(owner, lock))
            || source.index != actor.source_index()
            || control.cleanup_until > context.shutdown_deadline
            || startup_deadline > control.until
            || cleanup_ceiling > control.cleanup_until
            || actor.actor_digest() == Digest([0; 32])
        {
            return Err(BoundTeacherSpawnCauseV3::Checker(invalid(
                "teacher /3 original source, root or deadline mismatch",
            )));
        }
        control.check().map_err(BoundTeacherSpawnCauseV3::Source)?;
        if !context
            .try_accepts()
            .map_err(BoundTeacherSpawnCauseV3::Root)?
        {
            return Err(BoundTeacherSpawnCauseV3::Checker(invalid(
                "teacher /3 root revoked",
            )));
        }
        let root_digest = root
            .descriptor()
            .digest()
            .map_err(|_| BoundTeacherSpawnCauseV3::Checker(invalid("teacher /3 root digest")))?;
        let registration = TeacherWorkerRegistration {
            collection_owner: actor.collection_owner(),
            authority: root.authority(),
            original_deadline: root.hard_deadline(),
            input_lock: root.startup_lock_digest(),
            root_admission: root_digest,
            root_budget: admission.root_budget(),
            frozen_actor: actor.actor_digest(),
        };
        let executable = source
            .executable
            .try_clone()
            .map_err(BoundTeacherSpawnCauseV3::Io)?;
        let directory = source
            .working_directory
            .try_clone()
            .map_err(BoundTeacherSpawnCauseV3::Io)?;
        let pin = match source.sealed_executable().cloned() {
            Some(sealed) => ExternalLaunchPin::from_verified_sealed(sealed, directory),
            None => ExternalLaunchPin::from_owned_descriptors(executable, directory),
        }
        .map_err(BoundTeacherSpawnCauseV3::Checker)?;
        let config = source
            .prepared
            .configuration()
            .map_err(BoundTeacherSpawnCauseV3::Checker)?;
        let clock = ExternalWorkerClock {
            domain: root.runtime_clock().domain(),
            origin: root.runtime_clock().origin(),
        };
        let hard = clock
            .origin
            .checked_add(std::time::Duration::from_nanos(root.hard_deadline().at.0))
            .ok_or(BoundTeacherSpawnCauseV3::Checker(invalid(
                "teacher /3 hard deadline overflow",
            )))?;
        let ceiling = cleanup_ceiling.min(hard.checked_add(config.shutdown_grace).ok_or(
            BoundTeacherSpawnCauseV3::Checker(invalid("teacher /3 cleanup ceiling overflow")),
        )?);
        if !context
            .try_accepts()
            .map_err(BoundTeacherSpawnCauseV3::Root)?
        {
            return Err(BoundTeacherSpawnCauseV3::Checker(invalid(
                "teacher /3 root revoked before spawn",
            )));
        }
        reservation
            .validate_before_spawn()
            .map_err(BoundTeacherSpawnCauseV3::Checker)?;
        CollectionTeacherWorker::spawn_in_process_scope(
            config,
            source.prepared.profile().engine,
            registration,
            clock,
            startup_deadline,
            ceiling,
            pin,
            permit.clone(),
        )
        .map_err(BoundTeacherSpawnCauseV3::Checker)
    })();
    let worker = match created {
        Ok(value) => value,
        Err(cause) => return Err(BoundTeacherSpawnFailureV3::BeforeSpawn { cause, actor }),
    };
    // No fallible work occurs between physical spawn and the owning lifecycle.
    let owner = RoventCollectionTeacherPhysicalOwnerV3 {
        actor: Some(actor),
        worker,
        member_witness: permit.member_witness(),
        verified_no_child: false,
    };
    let mut lifecycle =
        RoventCollectionTeacherLifecycleV3::new(owner, reservation, original_cleanup);
    match lifecycle
        .owner()
        .admission()
        .root_admission()
        .original_context()
        .try_accepts()
    {
        Ok(true) => Ok((lifecycle, recovery_key)),
        Ok(false) => {
            lifecycle.request_shutdown(cleanup_ceiling);
            Err(BoundTeacherSpawnFailureV3::AfterSpawn {
                cause: BoundTeacherSpawnCauseV3::Checker(invalid(
                    "teacher /3 root revoked during startup",
                )),
                lifecycle,
                recovery_key,
            })
        }
        Err(error) => {
            lifecycle.root_observation_error = Some(error);
            if error.code != rz_contracts::ErrorCode::ResourceExhausted {
                lifecycle.request_shutdown(cleanup_ceiling);
            }
            Err(BoundTeacherSpawnFailureV3::AfterSpawn {
                cause: BoundTeacherSpawnCauseV3::Root(error),
                lifecycle,
                recovery_key,
            })
        }
    }
}

/// A joined worker can still have an unknown physical child state.
pub struct RoventCollectionTeacherLifecycleV3 {
    owner: Option<RoventCollectionTeacherPhysicalOwnerV3>,
    reservation: Option<TeacherRecoveryReservationV3>,
    shutdown_requested: bool,
    root_observation_error: Option<rz_contracts::ContractError>,
    diagnostics: RoventTeacherDiagnosticsDispositionV2,
    cleanup_deadline: Instant,
    registry_poisoned: bool,
    terminal_ledger: TeacherTerminalLedgerV2,
}
impl RoventCollectionTeacherLifecycleV3 {
    fn new(
        owner: RoventCollectionTeacherPhysicalOwnerV3,
        reservation: TeacherRecoveryReservationV3,
        cleanup_deadline: Instant,
    ) -> Self {
        Self {
            owner: Some(owner),
            reservation: Some(reservation),
            shutdown_requested: false,
            root_observation_error: None,
            diagnostics: RoventTeacherDiagnosticsDispositionV2::NotObserved,
            cleanup_deadline,
            registry_poisoned: false,
            terminal_ledger: TeacherTerminalLedgerV2::default(),
        }
    }
    pub(super) fn from_parked(
        reservation: TeacherRecoveryReservationV3,
        p: ParkedTeacherV3,
    ) -> Self {
        Self {
            owner: Some(p.owner),
            reservation: Some(reservation),
            shutdown_requested: p.shutdown_requested,
            root_observation_error: p.root_observation_error,
            diagnostics: p.diagnostics,
            cleanup_deadline: p.cleanup_deadline,
            registry_poisoned: p.registry_poisoned,
            terminal_ledger: p.terminal_ledger,
        }
    }
    pub fn owner(&self) -> &RoventCollectionTeacherPhysicalOwnerV3 {
        self.owner.as_ref().expect("retained /3 teacher")
    }
    pub fn observe_root(&mut self) -> Result<bool, rz_contracts::ContractError> {
        if self.shutdown_requested {
            return Ok(false);
        }
        let context = self.owner().admission().root_admission().original_context();
        let until = context.shutdown_deadline;
        match context.try_accepts() {
            Ok(true) => Ok(true),
            Ok(false) => {
                self.request_shutdown(until);
                Ok(false)
            }
            Err(error) => {
                self.root_observation_error = Some(error);
                if error.code != rz_contracts::ErrorCode::ResourceExhausted {
                    self.request_shutdown(until);
                }
                Err(error)
            }
        }
    }
    pub fn request_shutdown(&mut self, until: Instant) {
        self.shutdown_requested = true;
        self.owner().worker.request_shutdown(until);
    }
    pub fn submit(&mut self, job: TeacherJob) -> Result<TeacherDispatch, CheckerError> {
        if self.terminal_ledger.pending.is_some() || self.terminal_ledger.unmatched.is_some() {
            return Err(invalid("teacher /3 previous terminal unresolved"));
        }
        let next = self
            .terminal_ledger
            .submitted
            .checked_add(1)
            .ok_or_else(|| invalid("teacher /3 dispatch accounting overflow"))?;
        if !self
            .observe_root()
            .map_err(|_| invalid("teacher /3 root observation unavailable"))?
        {
            return Err(invalid("teacher /3 admission closed"));
        }
        let result = self
            .owner
            .as_mut()
            .expect("retained /3 teacher")
            .submit(job);
        if let Ok(dispatch) = &result {
            self.terminal_ledger.pending = Some(dispatch.clone());
            self.terminal_ledger.submitted = next;
        }
        if result.is_err() {
            let _ = self.observe_root();
        }
        result
    }
    pub fn take_completion(&mut self) -> Option<TeacherCompletion> {
        if self.terminal_ledger.unmatched.is_some() {
            return None;
        }
        let completion = self.owner().worker.take_completion()?;
        if self
            .terminal_ledger
            .pending
            .as_ref()
            .is_some_and(|d| completion.matches_dispatch_identity(d))
        {
            self.terminal_ledger.pending = None;
            self.terminal_ledger.delivered += 1;
            Some(completion)
        } else {
            self.terminal_ledger.unmatched = Some(completion);
            None
        }
    }
    pub fn unmatched_completion(&self) -> Option<&TeacherCompletion> {
        self.terminal_ledger.unmatched.as_ref()
    }
    pub fn validate_completion(
        &mut self,
        c: &TeacherCompletion,
        d: &TeacherDispatch,
    ) -> Result<(), CheckerError> {
        if !self
            .observe_root()
            .map_err(|_| invalid("teacher /3 root observation unavailable"))?
        {
            return Err(invalid("teacher /3 admission closed"));
        }
        let result = self.owner().validate_completion(c, d);
        if result.is_err() {
            let _ = self.observe_root();
        }
        result
    }
    pub fn take_diagnostics_after_join(&mut self) -> Option<TeacherDiagnostics> {
        if self.diagnostics != RoventTeacherDiagnosticsDispositionV2::NotObserved {
            return None;
        }
        let worker = &mut self.owner.as_mut().expect("retained /3 teacher").worker;
        let _join_snapshot = worker.shutdown_snapshot()?;
        let value = worker.take_diagnostics_after_join();
        self.diagnostics = if value.is_some() {
            RoventTeacherDiagnosticsDispositionV2::Recovered
        } else {
            RoventTeacherDiagnosticsDispositionV2::UnavailableOrAlreadyTaken
        };
        value
    }
    pub fn poll(&mut self) -> RoventTeacherRetirementObservationV2 {
        if !self.shutdown_requested {
            let _ = self.observe_root();
        }
        let owner = self.owner.as_mut().expect("retained /3 teacher");
        let joined = owner
            .worker
            .try_join()
            .or_else(|| owner.worker.shutdown_snapshot());
        let process = owner.worker.process_after_join();
        let occupancy = owner.worker.occupancy();
        let physical = if !self.shutdown_requested && joined.is_none() {
            RoventTeacherPhysicalRetirementV2::NotRequested
        } else if joined.is_none() {
            RoventTeacherPhysicalRetirementV2::Pending
        } else if let Some(p) = process.filter(|p| {
            p.process_identity.is_some()
                && p.exit_observed
                && p.stdout_drained
                && p.stderr_drained
                && p.cleanup_complete
                && !p.quarantined
                && !p.ownership_lost
        }) {
            RoventTeacherPhysicalRetirementV2::Confirmed(p)
        } else {
            RoventTeacherPhysicalRetirementV2::Unknown(process)
        };
        let terminal_accounting = match self.terminal_ledger.observation() {
            RoventTeacherTerminalAccountingV2::Pending { .. }
                if joined.is_some() && occupancy.reliable_queued == 0 =>
            {
                RoventTeacherTerminalAccountingV2::Unknown
            }
            value => value,
        };
        RoventTeacherRetirementObservationV2 {
            root_observation_error: self.root_observation_error,
            worker_join: joined,
            physical,
            commands: occupancy.commands,
            reliable_queued: occupancy.reliable_queued,
            reliable_reserved: occupancy.reliable_reserved,
            diagnostics: self.diagnostics,
            terminal_accounting,
            registry_poisoned: self.registry_poisoned,
        }
    }
    /// Return original custody only after physical closure and terminal drain.
    pub fn try_finish(
        mut self,
    ) -> Result<
        (
            RoventCollectionTeacherPhysicalOwnerV3,
            RoventTeacherRetirementObservationV2,
        ),
        Self,
    > {
        let mut observation = self.poll();
        if !matches!(
            &observation.physical,
            RoventTeacherPhysicalRetirementV2::Confirmed(_)
        ) || observation.worker_join.is_none()
            || observation.commands != 0
            || observation.reliable_queued != 0
            || observation.reliable_reserved != 0
            || observation.diagnostics == RoventTeacherDiagnosticsDispositionV2::NotObserved
            || !matches!(
                observation.terminal_accounting,
                RoventTeacherTerminalAccountingV2::Delivered { .. }
            )
        {
            return Err(self);
        }
        let poisoned = match self.reservation.as_mut().and_then(|r| r.release().ok()) {
            Some(value) => value,
            None => return Err(self),
        };
        observation.registry_poisoned |= poisoned;
        Ok((self.owner.take().expect("retained /3 teacher"), observation))
    }
}
impl Drop for RoventCollectionTeacherLifecycleV3 {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.take() {
            owner.worker.request_shutdown(self.cleanup_deadline);
            let parked = ParkedTeacherV3 {
                owner,
                shutdown_requested: true,
                root_observation_error: self.root_observation_error,
                diagnostics: self.diagnostics,
                cleanup_deadline: self.cleanup_deadline,
                registry_poisoned: self.registry_poisoned,
                terminal_ledger: std::mem::take(&mut self.terminal_ledger),
            };
            match self.reservation.as_mut() {
                Some(reservation) => {
                    if let Err(parked) = reservation.park(parked) {
                        quarantine_v3::mark_custody_violation();
                        std::mem::forget(parked);
                    }
                }
                None => {
                    quarantine_v3::mark_custody_violation();
                    std::mem::forget(parked);
                }
            }
        }
    }
}

/// Confirmed physical retirement of the original worker and source owner.
pub struct RoventTeacherRetirementTokenV3 {
    owner: RoventCollectionTeacherPhysicalOwnerV3,
    observation: RoventTeacherRetirementObservationV2,
}

/// Worker join and terminal delivery only; physical completion may be unknown.
pub struct RoventTeacherWorkerTerminalTokenV3 {
    lifecycle: RoventCollectionTeacherLifecycleV3,
    observation: RoventTeacherRetirementObservationV2,
}

/// Original owner released only after same-issuer evidence proves no OS child.
pub struct RoventTeacherNoChildRetirementTokenV3 {
    owner: RoventCollectionTeacherPhysicalOwnerV3,
    observation: RoventTeacherRetirementObservationV2,
}

impl RoventCollectionTeacherLifecycleV3 {
    /// A worker join and drained terminals alone do not establish child exit.
    pub fn try_finish_worker(mut self) -> Result<RoventTeacherWorkerTerminalTokenV3, Self> {
        let observation = self.poll();
        if observation.worker_join.is_none()
            || !self
                .owner()
                .worker
                .scope_join_witness()
                .is_some_and(|joined| self.owner().member_witness.same_joined_member(joined))
            || observation.commands != 0
            || observation.reliable_queued != 0
            || observation.reliable_reserved != 0
            || observation.diagnostics == RoventTeacherDiagnosticsDispositionV2::NotObserved
            || !matches!(
                observation.terminal_accounting,
                RoventTeacherTerminalAccountingV2::Delivered { .. }
            )
        {
            return Err(self);
        }
        Ok(RoventTeacherWorkerTerminalTokenV3 {
            lifecycle: self,
            observation,
        })
    }
}

impl RoventTeacherRetirementTokenV3 {
    pub fn owner(&self) -> &RoventCollectionTeacherPhysicalOwnerV3 {
        &self.owner
    }
    pub fn observation(&self) -> &RoventTeacherRetirementObservationV2 {
        &self.observation
    }
    pub fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV3) -> bool {
        self.owner.admission().same_original_admission(admission)
    }
}

impl RoventTeacherNoChildRetirementTokenV3 {
    pub fn owner(&self) -> &RoventCollectionTeacherPhysicalOwnerV3 {
        &self.owner
    }
    pub fn observation(&self) -> &RoventTeacherRetirementObservationV2 {
        &self.observation
    }
    pub fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV3) -> bool {
        self.owner.admission().same_original_admission(admission)
    }
}

impl RoventTeacherWorkerTerminalTokenV3 {
    pub fn member_witness(&self) -> &CollectionProcessMemberWitness {
        self.lifecycle.owner().member_witness()
    }
    pub fn registration(&self) -> TeacherWorkerRegistration {
        self.lifecycle.owner().worker.registration()
    }
    pub(super) fn has_original_join_witness(&self) -> bool {
        self.lifecycle
            .owner()
            .worker
            .scope_join_witness()
            .is_some_and(|joined| self.member_witness().same_joined_member(joined))
    }
    pub fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV3) -> bool {
        self.lifecycle
            .owner()
            .admission()
            .same_original_admission(admission)
    }
    pub fn owner(&self) -> &RoventCollectionTeacherPhysicalOwnerV3 {
        self.lifecycle.owner()
    }
    pub fn observation(&self) -> &RoventTeacherRetirementObservationV2 {
        &self.observation
    }

    /// An unresolved result returns the same token and original custody.
    pub fn try_into_physical_retirement(self) -> Result<RoventTeacherRetirementTokenV3, Self> {
        let Self {
            lifecycle,
            observation,
        } = self;
        match lifecycle.try_finish() {
            Ok((owner, actual)) => Ok(RoventTeacherRetirementTokenV3 {
                owner,
                observation: actual,
            }),
            Err(lifecycle) => Err(Self {
                lifecycle,
                observation,
            }),
        }
    }

    /// Release only with the same scope's joined and no-child observations.
    pub fn try_into_no_child_retirement(
        mut self,
        physical: &CollectionScopePhysicalOutcome,
        workers: &CollectionScopeWorkerOutcome,
    ) -> Result<RoventTeacherNoChildRetirementTokenV3, Self> {
        let witness = self.member_witness();
        if !physical.same_issuer_as_worker(workers)
            || !self.has_original_join_witness()
            || !matches!(
                witness.physical_state_in(physical),
                Some(
                    CollectionMemberPhysicalState::IssuedNoChild
                        | CollectionMemberPhysicalState::AcquiredNoChild
                )
            )
            || witness.state_in(workers) != Some(CollectionMemberWorkerState::Joined)
        {
            return Err(self);
        }
        let poisoned = match self
            .lifecycle
            .reservation
            .as_mut()
            .and_then(|r| r.release().ok())
        {
            Some(value) => value,
            None => return Err(self),
        };
        self.observation.registry_poisoned |= poisoned;
        let mut owner = self
            .lifecycle
            .owner
            .take()
            .expect("retained /3 teacher owner");
        owner.verified_no_child = true;
        Ok(RoventTeacherNoChildRetirementTokenV3 {
            owner,
            observation: self.observation,
        })
    }
}
