//! Live `/3` root custody for a registered system-runtime input lock.
//!
//! Preparation uses the same Rules, original go, cancellation, clock and
//! deadline core as older versions. The lock, descriptor and allocation
//! clock stay in their own `/3` domains throughout their lifetimes.

use super::*;

impl<'a> SourceDeclarations<'a> {
    fn from_v3(lock: &'a rz_experiments::RoventInputLockV3) -> Self {
        let m = &lock.manifest;
        Self {
            canonical_sha256: &lock.canonical_sha256,
            search: &m.engine.search,
            coordinator: &m.coordinator,
            resources: &m.resources,
            wall_time_max_ms: m.wall_time_max_ms,
            cleanup_max_ms: m.cleanup_max_ms,
            output_bytes_max: m.output_bytes_max,
        }
    }
}

/// Validated metadata and the exact original owner, before the live gate.
#[derive(Debug)]
pub struct PreparedSourceV3 {
    owner: rz_experiments::RoventOriginalInputLock,
    core: PreparedRootCore,
    descriptor: rz_experiments::RoventRootDescriptorV3,
    allocation_clock: rz_experiments::RoventRootAllocationClockV3,
    rejected: Option<InitialStopReason>,
}

/// A live root capability issued by the original owner under its gate.
#[derive(Debug)]
pub struct RoventRootAdmissionV3 {
    owner: rz_experiments::RoventOriginalInputLock,
    core: PreparedRootCore,
    descriptor: rz_experiments::RoventRootDescriptorV3,
    allocation_clock: rz_experiments::RoventRootAllocationClockV3,
}

/// A stopped source has no live context, clock or worker authority.
#[derive(Debug)]
pub struct StoppedInitialV3 {
    owner: rz_experiments::RoventOriginalInputLock,
    startup_lock: Digest,
    state_and_history: Digest,
    rules_profile: Digest,
    descriptor: rz_experiments::RoventRootDescriptorV3,
    allocation_clock: rz_experiments::RoventRootAllocationClockV3,
    allowed_moves: Arc<[BoardMove]>,
    fallback: Option<BoardMove>,
    stop_reason: InitialStopReason,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum InitialAdmissionOutcomeV3 {
    Admitted(RoventRootAdmissionV3),
    StoppedInitial(StoppedInitialV3),
}

impl PreparedSourceV3 {
    pub fn prepare(
        owner: &rz_experiments::RoventOriginalInputLock,
        policy: RoventRootAdmissionPolicy,
        position: &Position,
        context: &SearchSessionContext,
        clock: &ProcessClock,
    ) -> Result<Self, RoventRootAdmissionError> {
        owner.validate().map_err(|_| {
            fail(
                RoventRootAdmissionFailure::InvalidLock,
                "original startup input lock",
            )
        })?;
        let lock = owner.v3().ok_or_else(|| {
            fail(
                RoventRootAdmissionFailure::InvalidLock,
                "original /3 startup input lock required",
            )
        })?;
        let declarations = SourceDeclarations::from_v3(lock);
        let core = prepare_root_core(&declarations, policy, position, context, clock)?;
        let descriptor =
            rz_experiments::RoventRootDescriptorV3::seal(core.descriptor_input.clone(), lock)
                .map_err(codec_error)?;
        let digest = descriptor.digest().map_err(codec_error)?;
        let allocation_clock = rz_experiments::RoventRootAllocationClockV3::seal(
            core.allocation_input(digest, &declarations),
            &descriptor,
            lock,
        )
        .map_err(codec_error)?;
        Ok(Self {
            owner: owner.clone(),
            core,
            descriptor,
            allocation_clock,
            rejected: None,
        })
    }

    pub fn owner(&self) -> &rz_experiments::RoventOriginalInputLock {
        &self.owner
    }
    pub fn descriptor(&self) -> &rz_experiments::RoventRootDescriptorV3 {
        &self.descriptor
    }
    pub fn allocation_clock(&self) -> &rz_experiments::RoventRootAllocationClockV3 {
        &self.allocation_clock
    }
    pub fn startup_lock_digest(&self) -> Digest {
        self.core.startup_lock
    }

    #[allow(clippy::result_large_err)]
    pub fn try_into_live(self) -> Result<RoventRootAdmissionV3, (Self, RoventRootAdmissionError)> {
        if let Some(reason) = self.rejected {
            return Err((self, stopped_error(reason)));
        }
        let context = self.core.context.clone();
        let clock = self.core.runtime_clock.clone();
        let hard = self.core.hard;
        let mut source = Some(self);
        match with_live_gate(&context, &clock, hard, || {
            let source = source.take().expect("one original /3 source");
            RoventRootAdmissionV3 {
                owner: source.owner,
                core: source.core,
                descriptor: source.descriptor,
                allocation_clock: source.allocation_clock,
            }
        }) {
            Ok(admitted) => Ok(admitted),
            Err((reason, error)) => {
                let mut source = source.expect("unissued original /3 source");
                source.rejected = reason;
                Err((source, error))
            }
        }
    }

    pub fn into_initial(self) -> Result<InitialAdmissionOutcomeV3, RoventRootAdmissionError> {
        match self.try_into_live() {
            Ok(admitted) => Ok(InitialAdmissionOutcomeV3::Admitted(admitted)),
            Err((source, error)) => match source.rejected {
                Some(stop_reason) => Ok(InitialAdmissionOutcomeV3::StoppedInitial(
                    StoppedInitialV3 {
                        owner: source.owner,
                        startup_lock: source.core.startup_lock,
                        state_and_history: source.core.state_and_history,
                        rules_profile: source.core.rules_profile,
                        descriptor: source.descriptor,
                        allocation_clock: source.allocation_clock,
                        allowed_moves: source.core.allowed_moves,
                        fallback: source.core.fallback,
                        stop_reason,
                    },
                )),
                None => Err(error),
            },
        }
    }
}

impl RoventRootAdmissionV3 {
    pub(crate) fn claim_teacher_scope_once(&self) -> bool {
        self.core
            .context
            .original_start
            .as_ref()
            .is_some_and(|start| start.claim_teacher_scope_once())
    }

    pub fn owner(&self) -> &rz_experiments::RoventOriginalInputLock {
        &self.owner
    }
    pub fn descriptor(&self) -> &rz_experiments::RoventRootDescriptorV3 {
        &self.descriptor
    }
    pub fn allocation_clock(&self) -> &rz_experiments::RoventRootAllocationClockV3 {
        &self.allocation_clock
    }
    pub fn startup_lock_digest(&self) -> Digest {
        self.core.startup_lock
    }
    pub fn policy(&self) -> RoventRootAdmissionPolicy {
        self.core.policy
    }
    pub fn state_and_history_digest(&self) -> Digest {
        self.core.state_and_history
    }
    pub fn rules_profile_digest(&self) -> Digest {
        self.core.rules_profile
    }
    pub fn authority(&self) -> rz_contracts::pals::SearchAuthority {
        self.core.context.authority
    }
    pub fn original_go(&self) -> &crate::parser::GoLimits {
        self.core
            .context
            .original_go
            .as_deref()
            .expect("admitted original go")
    }
    pub fn requested_nodes(&self) -> Option<u64> {
        self.core.context.requested_nodes
    }
    pub fn goal(&self) -> rz_contracts::rovent::SearchGoal {
        self.core.context.goal
    }
    pub fn declared_ceilings(&self) -> &RoventRootDeclaredCeilings {
        &self.descriptor.input.declared_ceilings
    }
    pub fn original_context(&self) -> &SearchSessionContext {
        &self.core.context
    }
    pub fn root_snapshot(&self) -> &PositionSnapshot {
        &self.core.snapshot
    }
    pub fn allowed_root_moves(&self) -> &Arc<[BoardMove]> {
        &self.core.allowed_moves
    }
    pub fn root_moves(&self) -> Option<&Arc<[BoardMove]>> {
        self.core.root_moves.as_ref()
    }
    pub fn runtime_clock(&self) -> &ContractSystemClock {
        &self.core.runtime_clock
    }
    pub fn hard_deadline(&self) -> Deadline {
        self.core.hard
    }
    pub fn soft_deadline(&self) -> Deadline {
        self.core.soft
    }
    pub fn admission_deadline(&self) -> Deadline {
        self.core.admission
    }
    pub fn cleanup_deadline(&self) -> Deadline {
        self.core.cleanup
    }
}

impl StoppedInitialV3 {
    pub fn owner(&self) -> &rz_experiments::RoventOriginalInputLock {
        &self.owner
    }
    pub fn startup_lock_digest(&self) -> Digest {
        self.startup_lock
    }
    pub fn state_and_history_digest(&self) -> Digest {
        self.state_and_history
    }
    pub fn rules_profile_digest(&self) -> Digest {
        self.rules_profile
    }
    pub fn descriptor(&self) -> &rz_experiments::RoventRootDescriptorV3 {
        &self.descriptor
    }
    pub fn allocation_clock(&self) -> &rz_experiments::RoventRootAllocationClockV3 {
        &self.allocation_clock
    }
    pub fn allowed_root_moves(&self) -> &Arc<[BoardMove]> {
        &self.allowed_moves
    }
    pub fn fallback(&self) -> Option<BoardMove> {
        self.fallback
    }
    pub fn stop_reason(&self) -> InitialStopReason {
        self.stop_reason
    }
}
