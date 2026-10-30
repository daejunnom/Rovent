//! Unique manager custody of the original profiled external R owner.
//!
//! The bootstrap custody retains the verified source descriptors. It is never
//! replaced with a bare worker, a legacy ExternalJob or a metadata-only owner.
//! The parent must register this module only with these same feature gates.
#![cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]

use crate::rovent_bootstrap::RoventExternalRSourceCustody;
use crate::rovent_external_launch::{RoventExternalOwner, VerifiedRoventExternalStartup};
use crate::rovent_root_admission::RoventRootAdmission;
use crate::rovent_root_admission::source_seal::RoventRootAdmissionV2;
use crate::search_driver::SearchSessionContext;
use rz_contracts::Digest;
use rz_contracts::pals::SearchAuthority;
use rz_contracts::pals_adaptive::CpuWorkerId;
use rz_eval::error::FailureKind;
use rz_experiments::{RoventExternalPurpose, RoventInputLock, RoventInputLockV2};
use rz_runtime::CompletionSignal;
use rz_runtime::adaptive::ControlMailbox;
use rz_runtime::contracts::ContractClock;
use rz_runtime::contracts::ContractSystemClock;
use rz_search::cpu_checker::{CheckerError, CheckerShutdown};
use rz_search::external_cpu::worker::ExternalWorkerStartup;
use rz_search::external_cpu::{
    ExternalCpuConfig, ExternalEnginePurpose, ExternalGoConstraint, ProfiledExternalCompletion,
    ProfiledExternalCurrentRootFence, ProfiledExternalDiagnostics, ProfiledExternalDispatch,
    ProfiledExternalJob, ProfiledExternalProcessCloseProof, ProfiledExternalProgress,
    ProfiledExternalRootSource, ProfiledExternalScope,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Version-specific admission keeps the original typed lock and root owners.
/// This trait describes only supported manifest versions, not new authorities.
mod sealed {
    pub trait Sealed {}
    impl Sealed for super::RoventInputLock {}
    impl Sealed for super::RoventInputLockV2 {}
}
pub trait ExternalManagerLock: sealed::Sealed + Send + Sync + 'static {
    type Root: ExternalManagerRoot;
    type Custody;
    fn purpose(custody: &Self::Custody) -> RoventExternalPurpose;
    fn digest(custody: &Self::Custody) -> Digest;
    fn policy(custody: &Self::Custody) -> Result<(ExternalGoConstraint, Duration), CheckerError>;
    fn matches_owner(custody: &Self::Custody, root: &Arc<Self::Root>) -> bool;
    fn admit(owner: &mut RoventExternalOwner, root: Arc<Self::Root>) -> Result<(), CheckerError>;
    fn is_current(owner: &RoventExternalOwner, root: &Arc<Self::Root>) -> bool;
}

pub trait ExternalManagerRoot: Send + Sync + 'static {
    fn startup_lock_digest(&self) -> Digest;
    fn authority(&self) -> SearchAuthority;
    fn original_context(&self) -> &SearchSessionContext;
    fn runtime_clock(&self) -> &ContractSystemClock;
}

macro_rules! impl_manager_root {
    ($root:ty) => {
        impl ExternalManagerRoot for $root {
            fn startup_lock_digest(&self) -> Digest {
                self.startup_lock_digest()
            }
            fn authority(&self) -> SearchAuthority {
                self.authority()
            }
            fn original_context(&self) -> &SearchSessionContext {
                self.original_context()
            }
            fn runtime_clock(&self) -> &ContractSystemClock {
                self.runtime_clock()
            }
        }
    };
}
impl_manager_root!(RoventRootAdmission);
impl_manager_root!(RoventRootAdmissionV2);

macro_rules! impl_manager_lock {
    ($lock:ty, $root:ty, $matches:expr, $admit:ident, $current:ident) => {
        impl ExternalManagerLock for $lock {
            type Root = $root;
            type Custody = RoventExternalRSourceCustody<$lock>;
            fn purpose(c: &Self::Custody) -> RoventExternalPurpose {
                c.source_purpose()
            }
            fn digest(c: &Self::Custody) -> Digest {
                c.input_lock_digest()
            }
            fn policy(c: &Self::Custody) -> Result<(ExternalGoConstraint, Duration), CheckerError> {
                c.question_policy().map_err(|error| match error.kind {
                    FailureKind::UnsupportedModel => CheckerError::Unsupported(error.detail),
                    _ => invalid(error.detail),
                })
            }
            fn matches_owner(c: &Self::Custody, r: &Arc<Self::Root>) -> bool {
                ($matches)(c, r)
            }
            fn admit(o: &mut RoventExternalOwner, r: Arc<Self::Root>) -> Result<(), CheckerError> {
                o.$admit(r)
            }
            fn is_current(o: &RoventExternalOwner, r: &Arc<Self::Root>) -> bool {
                o.$current().is_some_and(|actual| Arc::ptr_eq(actual, r))
            }
        }
    };
}
impl_manager_lock!(
    RoventInputLock,
    RoventRootAdmission,
    |_c: &RoventExternalRSourceCustody<RoventInputLock>, _r: &Arc<RoventRootAdmission>| true,
    admit_actual_root,
    actual_root_admission
);
impl_manager_lock!(
    RoventInputLockV2,
    RoventRootAdmissionV2,
    |c: &RoventExternalRSourceCustody<RoventInputLockV2>, r: &Arc<RoventRootAdmissionV2>| {
        r.owner()
            .v2()
            .is_some_and(|owner| Arc::ptr_eq(c.original_lock_owner(), owner))
    },
    admit_actual_root_v2,
    actual_root_admission_v2
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Running,
    RootDraining,
    RootDrained,
    ChangingRoot,
    ProcessShutdown,
    ProcessClosed,
    Quarantined,
}

struct Pending {
    dispatch: ProfiledExternalDispatch,
    control: ControlMailbox<ProfiledExternalScope>,
}

/// Observation of an explicit process-finish command, not a root work grant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProfiledExternalProcessFinishBasis {
    /// Running, transitioning or uncertain roots retain their original ceiling.
    OriginalRootCleanup,
    /// An actual current-root fence closed all accepted work before this command.
    SettledCurrentRoot,
}

#[derive(Clone, Copy, Debug)]
pub struct ProfiledExternalProcessFinishObservation {
    pub authority: SearchAuthority,
    pub first_requested_until: Instant,
    pub latest_requested_until: Instant,
    pub applied_until: Instant,
    pub original_root_cleanup_until: Instant,
    pub basis: ProfiledExternalProcessFinishBasis,
    pub settled_root_fence_sequence: Option<u64>,
}

/// Both success and failure keep the actual terminal and original dispatch.
/// `validation` is a provenance check, never a success/teacher/value verdict.
pub struct ProfiledExternalManagerCompletion {
    dispatch: Option<ProfiledExternalDispatch>,
    completion: ProfiledExternalCompletion,
    validation: Result<(), CheckerError>,
}

impl ProfiledExternalManagerCompletion {
    pub fn dispatch(&self) -> Option<&ProfiledExternalDispatch> {
        self.dispatch.as_ref()
    }

    pub fn completion(&self) -> &ProfiledExternalCompletion {
        &self.completion
    }

    pub fn validation(&self) -> &Result<(), CheckerError> {
        &self.validation
    }

    pub fn into_parts(
        self,
    ) -> (
        Option<ProfiledExternalDispatch>,
        ProfiledExternalCompletion,
        Result<(), CheckerError>,
    ) {
        (self.dispatch, self.completion, self.validation)
    }
}

/// A source rejection returns the same live owner and all descriptor custody.
/// The caller must retain/clean it; an admission error is not a joined process.
pub type ProfiledExternalSourceRejection<L = RoventInputLock> = (
    RoventExternalOwner,
    <L as ExternalManagerLock>::Custody,
    CheckerError,
);

/// One manager is the only mutable owner. The existing worker's internal
/// bounded mailbox is reused; no per-tick outer mutex or worker clone is added.
pub struct ProfiledExternalManagerSource<L: ExternalManagerLock = RoventInputLock> {
    owner: RoventExternalOwner,
    custody: L::Custody,
    root: Arc<L::Root>,
    startup: VerifiedRoventExternalStartup,
    configuration: ExternalCpuConfig,
    question_constraint: ExternalGoConstraint,
    soft_quantum: Duration,
    source: ProfiledExternalRootSource,
    pending: Option<Pending>,
    next_root: Option<Arc<L::Root>>,
    root_fence: Option<ProfiledExternalCurrentRootFence>,
    joined: Option<Result<CheckerShutdown, CheckerError>>,
    process_finish: Option<ProfiledExternalProcessFinishObservation>,
    phase: Phase,
}

impl<L: ExternalManagerLock> ProfiledExternalManagerSource<L> {
    /// No process is started and no path is reopened. Only the same original
    /// root capability can be retained when it was already admitted upstream.
    #[allow(clippy::result_large_err)]
    pub fn from_ready(
        mut owner: RoventExternalOwner,
        custody: L::Custody,
        root: Arc<L::Root>,
    ) -> Result<Self, ProfiledExternalSourceRejection<L>> {
        let checked = (|| {
            if L::purpose(&custody) != RoventExternalPurpose::ResearchChecker
                || L::digest(&custody) != root.startup_lock_digest()
                || !L::matches_owner(&custody, &root)
                || owner.launch().profile().purpose != ExternalEnginePurpose::ResearchChecker
            {
                return Err(invalid("profiled R original source purpose/lock differs"));
            }
            // This policy comes only from the same original /3 startup source.
            // Legacy sources have no implicit limit, and profile capabilities
            // never manufacture or narrow the registered requested constraint.
            let (question_constraint, soft_quantum) = L::policy(&custody)?;
            owner.launch().require_query(question_constraint, false)?;
            let startup = owner
                .verified_startup()?
                .ok_or(invalid("profiled R actual startup is pending"))?;
            if startup.input_lock_sha256() != hex(root.startup_lock_digest())
                || startup.canonical_sha256() != owner.launch().canonical_sha256()
                || startup.capability_sha256() != owner.launch().capability_sha256()
            {
                return Err(invalid("profiled R actual startup/source pins differ"));
            }
            let configuration = owner.launch().configuration()?;
            L::admit(&mut owner, Arc::clone(&root))?;
            let source = owner
                .worker()
                .current_root_source()?
                .ok_or(invalid("profiled R current root source is pending"))?;
            validate_source::<L>(&owner, &root, &startup, &configuration, &source)?;
            Ok((
                startup,
                configuration,
                question_constraint,
                soft_quantum,
                source,
            ))
        })();
        match checked {
            Ok((startup, configuration, question_constraint, soft_quantum, source)) => Ok(Self {
                owner,
                custody,
                root,
                startup,
                configuration,
                question_constraint,
                soft_quantum,
                source,
                pending: None,
                next_root: None,
                root_fence: None,
                joined: None,
                process_finish: None,
                phase: Phase::Running,
            }),
            Err(error) => Err((owner, custody, error)),
        }
    }

    pub fn owner(&self) -> &RoventExternalOwner {
        &self.owner
    }

    pub fn root_admission(&self) -> &Arc<L::Root> {
        &self.root
    }

    pub fn startup(&self) -> &VerifiedRoventExternalStartup {
        &self.startup
    }

    pub fn startup_report(&self) -> Option<ExternalWorkerStartup> {
        self.owner.worker().startup_report()
    }

    pub fn configuration(&self) -> &ExternalCpuConfig {
        &self.configuration
    }

    pub fn question_constraint(&self) -> ExternalGoConstraint {
        self.question_constraint
    }

    pub fn soft_quantum(&self) -> Duration {
        self.soft_quantum
    }

    pub fn worker_owner(&self) -> CpuWorkerId {
        self.source.worker_owner()
    }

    pub fn current_source(&self) -> &ProfiledExternalRootSource {
        &self.source
    }

    pub fn signal(&self) -> CompletionSignal {
        self.owner.worker().signal()
    }

    pub fn pending_dispatch(&self) -> Option<&ProfiledExternalDispatch> {
        self.pending.as_ref().map(|pending| &pending.dispatch)
    }

    pub fn is_quarantined(&self) -> bool {
        self.phase == Phase::Quarantined
    }

    pub fn root_drained(&self) -> bool {
        self.phase == Phase::RootDrained && self.root_fence.is_some() && self.pending.is_none()
    }

    /// Preserve the exact caller-selected single/combined constraint. Profile
    /// capability and original root/deadline checks remain in the actual owner.
    /// Foreign reported nodes never become a native reservation here.
    pub fn submit(
        &mut self,
        job: ProfiledExternalJob,
    ) -> Result<ProfiledExternalDispatch, CheckerError> {
        if self.phase != Phase::Running || self.pending.is_some() || self.next_root.is_some() {
            return Err(invalid("profiled R source is closed or has pending work"));
        }
        if job.constraint != self.question_constraint || job.soft_quantum != self.soft_quantum {
            return Err(invalid(
                "profiled R requested question policy differs from original source",
            ));
        }
        self.source.validate_current()?;
        if job.owner != self.source.worker_owner()
            || job.question.id.authority != self.root.authority()
            || job.control.scope() != &job.scope()
        {
            return Err(invalid("profiled R dispatch owner/root/control differs"));
        }
        let control = job.control.clone();
        let dispatch = self.owner.submit_actual_root(job)?;
        self.pending = Some(Pending {
            dispatch: dispatch.clone(),
            control,
        });
        Ok(dispatch)
    }

    /// Failed, canceled, late and invalid terminals remain available verbatim.
    /// A mismatched terminal cannot release another question's pending control.
    pub fn take_completion(&mut self) -> Option<ProfiledExternalManagerCompletion> {
        let completion = self.owner.worker().take_completion()?;
        let matches = self.pending.as_ref().is_some_and(|pending| {
            pending.dispatch.scope() == completion.scope()
                && pending.dispatch.question() == completion.question()
        });
        let dispatch = if matches {
            self.pending.take().map(|pending| pending.dispatch)
        } else {
            None
        };
        let validation = match &dispatch {
            Some(dispatch) => self.owner.validate_actual_completion(&completion, dispatch),
            None => Err(invalid(
                "profiled R terminal has no original pending dispatch",
            )),
        };
        if validation.is_err() {
            self.phase = Phase::Quarantined;
        }
        Some(ProfiledExternalManagerCompletion {
            dispatch,
            completion,
            validation,
        })
    }

    /// Observation only. The manager must retain the original progress scope;
    /// progress does not settle a terminal, charge native nodes or grant reuse.
    pub fn take_progress(&self) -> Option<(u64, ProfiledExternalProgress)> {
        // A coalesced observation may be read after its reliable terminal. It
        // remains raw scoped progress, never grounds to revoke a later Job or
        // to resurrect the already consumed old completion.
        self.owner.worker().take_progress()
    }

    pub fn request_cancel(&self, sequence: u64) -> Result<(), CheckerError> {
        self.pending
            .as_ref()
            .ok_or(invalid("profiled R has no pending cancellation target"))?
            .control
            .request_cancel(sequence)
            .map_err(|_| invalid("profiled R cancellation sequence rejected"))
    }

    pub fn request_yield(&self, sequence: u64) -> Result<(), CheckerError> {
        self.pending
            .as_ref()
            .ok_or(invalid("profiled R has no pending yield target"))?
            .control
            .request_yield(sequence)
            .map_err(|_| invalid("profiled R yield sequence rejected"))
    }

    pub fn request_publish_progress(&self, sequence: u64) -> Result<(), CheckerError> {
        self.pending
            .as_ref()
            .ok_or(invalid("profiled R has no pending progress target"))?
            .control
            .request_publish_progress(sequence)
            .map_err(|_| invalid("profiled R progress sequence rejected"))
    }

    /// This is the actual worker's current-root stop/drain/ready operation.
    /// No successor authority or successful-terminal timestamp fabricates it.
    pub fn begin_root_quiesce(&mut self, until: Instant) -> Result<(), CheckerError> {
        if !matches!(self.phase, Phase::Running | Phase::RootDraining) {
            return Err(invalid("profiled R root quiesce phase differs"));
        }
        self.phase = Phase::RootDraining;
        self.owner
            .worker()
            .request_root_quiesce(self.root.authority(), until.min(self.cleanup_until()))
    }

    pub fn try_root_fence(
        &mut self,
    ) -> Option<Result<ProfiledExternalCurrentRootFence, CheckerError>> {
        if self.phase == Phase::RootDrained {
            return self.root_fence.as_ref().cloned().map(Ok);
        }
        if self.phase != Phase::RootDraining {
            return None;
        }
        let result = self.owner.worker().root_quiesce_result()?;
        let checked = result.and_then(|fence| {
            if self.pending.is_some()
                || !fence.permits_reuse()
                || fence.authority() != self.root.authority()
                || fence.process_identity() != self.startup.process_identity()
                || !fence.source().same_source(&self.source)
            {
                return Err(invalid(
                    "profiled R current root fence differs from retained source",
                ));
            }
            Ok(fence)
        });
        match &checked {
            Ok(fence) => {
                self.root_fence = Some(fence.clone());
                self.phase = Phase::RootDrained;
            }
            Err(_) => self.phase = Phase::Quarantined,
        }
        Some(checked)
    }

    /// Preserve an already issued root fence as historical evidence after a
    /// confirmed process close. Its presence does not permit a closed worker
    /// to serve another root; live reuse is separately gated by root_drained.
    pub fn root_fence(&self) -> Option<&ProfiledExternalCurrentRootFence> {
        matches!(self.phase, Phase::RootDrained | Phase::ProcessClosed)
            .then_some(self.root_fence.as_ref())
            .flatten()
    }

    /// Only a real later root may request this existing next-root/reset fence.
    /// The retained original admission is installed after that fence completes.
    pub fn request_root_change(
        &mut self,
        next: Arc<L::Root>,
        until: Instant,
    ) -> Result<(), CheckerError> {
        if self.phase != Phase::RootDrained || self.pending.is_some() || self.next_root.is_some() {
            return Err(invalid(
                "profiled R next root requires the original drained root",
            ));
        }
        if next.startup_lock_digest() != L::digest(&self.custody)
            || !L::matches_owner(&self.custody, &next)
            || next.authority().epoch != self.root.authority().epoch
            || next.authority().implementation != self.root.authority().implementation
        {
            return Err(invalid("profiled R next admission source differs"));
        }
        self.owner.worker().request_root_change(
            next.authority(),
            until.min(next.original_context().shutdown_deadline),
        )?;
        self.next_root = Some(next);
        self.phase = Phase::ChangingRoot;
        Ok(())
    }

    pub fn try_admit_root(&mut self) -> Option<Result<(), CheckerError>> {
        if self.phase != Phase::ChangingRoot {
            return None;
        }
        let fence = self.owner.root_change_result()?;
        let result = fence.and_then(|fence| {
            let next = self
                .next_root
                .as_ref()
                .ok_or(invalid("profiled R next admission is missing"))?;
            if !fence.permits_reuse()
                || fence.previous() != self.root.authority()
                || fence.next() != next.authority()
                || fence.process_identity() != Some(self.startup.process_identity())
            {
                return Err(invalid("profiled R next root fence differs"));
            }
            L::admit(&mut self.owner, Arc::clone(next))?;
            let source = self
                .owner
                .worker()
                .current_root_source()?
                .ok_or(invalid("profiled R next current source is pending"))?;
            validate_source::<L>(
                &self.owner,
                next,
                &self.startup,
                &self.configuration,
                &source,
            )?;
            self.source = source;
            self.root = Arc::clone(next);
            self.next_root = None;
            self.root_fence = None;
            self.phase = Phase::Running;
            Ok(())
        });
        if result.is_err() {
            self.phase = Phase::Quarantined;
        }
        Some(result)
    }

    pub fn request_shutdown(&mut self, until: Instant) {
        if self.joined.is_some() {
            return;
        }
        let observation = match self.process_finish {
            Some(previous) => ProfiledExternalProcessFinishObservation {
                latest_requested_until: until,
                // A repeated command can shorten the registered process finish,
                // never restart a fresh grace after its earlier fence expired.
                applied_until: until.min(previous.applied_until),
                ..previous
            },
            None => {
                let settled = self.phase == Phase::RootDrained
                    && self.pending.is_none()
                    && self.next_root.is_none()
                    && self.root_fence.as_ref().is_some_and(|fence| {
                        fence.permits_reuse()
                            && fence.authority() == self.root.authority()
                            && fence.process_identity() == self.startup.process_identity()
                            && fence.source().same_source(&self.source)
                            && self
                                .owner
                                .actual_root_quiesce_result()
                                .is_some_and(|result| {
                                    result.is_ok_and(|actual| {
                                        actual.permits_reuse()
                                            && actual.sequence() == fence.sequence()
                                            && actual.source().same_source(fence.source())
                                    })
                                })
                    });
                let root_cleanup = self.cleanup_until();
                // A settled root's original work grant has ended. Finishing
                // its retained idle process is a separate bounded command;
                // only this branch may use the caller's new finish deadline.
                ProfiledExternalProcessFinishObservation {
                    authority: self.root.authority(),
                    first_requested_until: until,
                    latest_requested_until: until,
                    applied_until: if settled {
                        until
                    } else {
                        until.min(root_cleanup)
                    },
                    original_root_cleanup_until: root_cleanup,
                    basis: if settled {
                        ProfiledExternalProcessFinishBasis::SettledCurrentRoot
                    } else {
                        ProfiledExternalProcessFinishBasis::OriginalRootCleanup
                    },
                    settled_root_fence_sequence: settled
                        .then(|| self.root_fence.as_ref().map(|fence| fence.sequence()))
                        .flatten(),
                }
            }
        };
        self.owner
            .worker()
            .request_shutdown(observation.applied_until);
        self.process_finish = Some(observation);
        self.phase = Phase::ProcessShutdown;
    }

    pub fn process_finish_observation(&self) -> Option<ProfiledExternalProcessFinishObservation> {
        self.process_finish
    }

    /// Thread join and actual process/pipe cleanup remain distinct evidence.
    /// An error is retained and never converted to an empty shutdown receipt.
    pub fn try_join(&mut self) -> Option<Result<CheckerShutdown, CheckerError>> {
        let result = self.owner.try_join()?;
        self.phase = if result.as_ref().is_ok_and(|receipt| {
            receipt.cleanup_complete
                && receipt.exit_observed
                && receipt.stdout_drained
                && receipt.stderr_drained
                && !receipt.quarantined
                && !receipt.ownership_lost
                && receipt.process_identity == Some(self.startup.process_identity())
        }) {
            Phase::ProcessClosed
        } else {
            Phase::Quarantined
        };
        self.joined = Some(result.clone());
        Some(result)
    }

    pub fn shutdown_snapshot(&self) -> Option<&Result<CheckerShutdown, CheckerError>> {
        self.joined.as_ref()
    }

    /// The original worker may certify semantic closure after actual join and
    /// terminal drain. This proof grants neither readiness nor root reuse and
    /// is distinct from a historical current-root fence. Preserve low-level
    /// join/cleanup errors even when this wrapper remains quarantined.
    pub fn process_close_proof(
        &self,
    ) -> Result<Option<ProfiledExternalProcessCloseProof>, CheckerError> {
        let proof = self.owner.worker().process_close_proof()?;
        Ok(if self.phase == Phase::ProcessClosed {
            proof
        } else {
            None
        })
    }

    pub fn take_diagnostics_after_join(&mut self) -> Option<ProfiledExternalDiagnostics> {
        self.joined.as_ref()?;
        self.owner.try_take_diagnostics_after_join()
    }

    /// Transfer the same opaque owner and FD custody only after a worker-issued
    /// current-root fence or confirmed physical join. Failure returns Self.
    #[allow(clippy::result_large_err)]
    pub fn into_parts(self) -> Result<(RoventExternalOwner, L::Custody), Self> {
        let transferable = self.phase == Phase::ProcessClosed
            || self.phase == Phase::RootDrained && self.next_root.is_none();
        if self.pending.is_some() || !transferable {
            return Err(self);
        }
        Ok((self.owner, self.custody))
    }

    fn cleanup_until(&self) -> Instant {
        self.root.original_context().shutdown_deadline
    }
}

fn validate_source<L: ExternalManagerLock>(
    owner: &RoventExternalOwner,
    root: &Arc<L::Root>,
    startup: &VerifiedRoventExternalStartup,
    configuration: &ExternalCpuConfig,
    source: &ProfiledExternalRootSource,
) -> Result<(), CheckerError> {
    source.validate_current()?;
    let clock = source.clock();
    let declaration = owner.launch().profile();
    if source.authority() != root.authority()
        || source.process_identity() != startup.process_identity()
        || source.profile() != declaration.engine
        || source.purpose() != ExternalEnginePurpose::ResearchChecker
        || clock.domain != root.runtime_clock().domain()
        || clock.origin != root.runtime_clock().origin()
        || !owner.worker().configuration_matches(
            configuration,
            declaration.engine,
            ExternalEnginePurpose::ResearchChecker,
            source.worker_owner(),
        )
        || !L::is_current(owner, root)
    {
        return Err(invalid(
            "profiled R current source/clock/profile/admission differs",
        ));
    }
    Ok(())
}

fn hex(digest: rz_contracts::Digest) -> String {
    digest.0.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn invalid(detail: &'static str) -> CheckerError {
    CheckerError::Invalid(detail)
}
