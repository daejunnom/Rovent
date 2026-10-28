//! 실제 per-go root를 원 외부 process fence 뒤에만 등록한다.
//! 원 startup lock과 실제 go의 budget을 재라벨하지 않으며, 공개 JSON은
//! live root cap이나 worker dispatch/completion을 구성하지 못한다.
use super::*;
use crate::rovent_root_admission::RoventRootAdmission;
use crate::rovent_root_admission::source_seal::RoventRootAdmissionV2;
use crate::search_driver::SearchSessionContext;
use rz_contracts::Deadline;
use rz_position::BoardMove;
use rz_runtime::contracts::ContractClock;
use rz_runtime::contracts::ContractSystemClock;
use std::sync::Arc;

enum OriginalExternalAdmission {
    V1(Arc<RoventRootAdmission>),
    V2(Arc<RoventRootAdmissionV2>),
}
impl OriginalExternalAdmission {
    fn same_owner(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::V1(a), Self::V1(b)) => Arc::ptr_eq(a, b),
            (Self::V2(a), Self::V2(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
    fn startup_lock_digest(&self) -> Digest {
        match self {
            Self::V1(a) => a.startup_lock_digest(),
            Self::V2(a) => a.startup_lock_digest(),
        }
    }
    fn original_context(&self) -> &SearchSessionContext {
        match self {
            Self::V1(a) => a.original_context(),
            Self::V2(a) => a.original_context(),
        }
    }
    fn runtime_clock(&self) -> &ContractSystemClock {
        match self {
            Self::V1(a) => a.runtime_clock(),
            Self::V2(a) => a.runtime_clock(),
        }
    }
    fn authority(&self) -> SearchAuthority {
        match self {
            Self::V1(a) => a.authority(),
            Self::V2(a) => a.authority(),
        }
    }
    fn requested_nodes(&self) -> Option<u64> {
        match self {
            Self::V1(a) => a.requested_nodes(),
            Self::V2(a) => a.requested_nodes(),
        }
    }
    fn hard_deadline(&self) -> Deadline {
        match self {
            Self::V1(a) => a.hard_deadline(),
            Self::V2(a) => a.hard_deadline(),
        }
    }
    fn allowed_root_moves(&self) -> &[BoardMove] {
        match self {
            Self::V1(a) => a.allowed_root_moves(),
            Self::V2(a) => a.allowed_root_moves(),
        }
    }
    fn budget(&self) -> Result<Digest, CheckerError> {
        match self {
            Self::V1(a) => Ok(a.root_budget_digest()),
            Self::V2(a) => a
                .descriptor()
                .root_budget_digest()
                .map_err(|_| invalid("actual /2 root budget is invalid")),
        }
    }
}

pub(super) struct ActualExternalRoot {
    admission: OriginalExternalAdmission,
    budget: Digest,
    fence_sequence: u64,
}

impl RoventExternalOwner {
    /// 최초 실제 root는 spawn 때의 원 authority와 같은 경우만 직접 붙인다.
    /// 이미 exact legacy job을 제출했거나 root를 바꾼 경우, 이 물리 owner가
    /// old terminal을 회수하고 ready/reset을 끝낸 원 fence가 필수다.
    pub fn admit_actual_root(
        &mut self,
        admission: Arc<RoventRootAdmission>,
    ) -> Result<(), CheckerError> {
        if self.launch.original_lock_owner_v2().is_some() {
            return Err(invalid("external /2 owner requires a /2 root"));
        }
        self.admit_original_root(OriginalExternalAdmission::V1(admission))
    }

    /// A /2 root must retain the same original lock allocation used to admit
    /// the actual external launch; equal digest bytes alone are insufficient.
    pub fn admit_actual_root_v2(
        &mut self,
        admission: Arc<RoventRootAdmissionV2>,
    ) -> Result<(), CheckerError> {
        let original = self
            .launch
            .original_lock_owner_v2()
            .ok_or(invalid("external /2 original lock owner is missing"))?;
        let root_owner = admission
            .owner()
            .v2()
            .ok_or(invalid("actual /2 root lock owner is missing"))?;
        if !Arc::ptr_eq(original, root_owner) {
            return Err(invalid("actual /2 root belongs to another lock owner"));
        }
        self.admit_original_root(OriginalExternalAdmission::V2(admission))
    }

    fn admit_original_root(
        &mut self,
        admission: OriginalExternalAdmission,
    ) -> Result<(), CheckerError> {
        let expected_lock = self
            .launch
            .input_lock_sha256
            .as_deref()
            .ok_or(invalid("external original startup lock is missing"))?;
        if digest(expected_lock)? != admission.startup_lock_digest()
            || !admission
                .original_context()
                .accepts()
                .map_err(|_| invalid("actual external root authority unavailable"))?
            || !self.worker.clock_matches(ExternalWorkerClock {
                domain: admission.runtime_clock().domain(),
                origin: admission.runtime_clock().origin(),
            })
        {
            return Err(invalid(
                "actual external root differs from original startup/clock/authority",
            ));
        }
        let startup = self
            .verified_startup()?
            .ok_or(invalid("actual external startup is still pending"))?;
        if let Some(active) = &self.active_root {
            if active.admission.same_owner(&admission) {
                return Ok(());
            }
            if active.admission.authority() == admission.authority() {
                return Err(invalid(
                    "same external root cannot replace its original live admission",
                ));
            }
        }
        let previous = self
            .active_root
            .as_ref()
            .map_or(self.initial_root, |root| root.admission.authority());
        let first = self.active_root.is_none()
            && !self.legacy_dispatched
            && admission.authority() == self.initial_root;
        let fence_sequence = if first {
            0
        } else {
            // Read the result from this retained physical owner. A caller may
            // not pass a fence copied from another external process.
            let fence = self
                .worker
                .root_change_result()
                .ok_or(invalid("actual external root change fence is pending"))??;
            if fence.previous() != previous
                || fence.next() != admission.authority()
                || !fence.permits_reuse()
                || fence.process_identity() != Some(startup.process_identity())
                || fence.sequence()
                    <= self
                        .active_root
                        .as_ref()
                        .map_or(0, |root| root.fence_sequence)
                || fence.new_game_reset() != (previous.game != admission.authority().game)
            {
                return Err(invalid(
                    "actual external root fence does not join this process/previous/next root",
                ));
            }
            fence.sequence()
        };
        let budget = admission.budget()?;
        self.active_root = Some(ActualExternalRoot {
            admission,
            budget,
            fence_sequence,
        });
        Ok(())
    }

    pub fn actual_root_admission(&self) -> Option<&Arc<RoventRootAdmission>> {
        match &self.active_root.as_ref()?.admission {
            OriginalExternalAdmission::V1(admission) => Some(admission),
            OriginalExternalAdmission::V2(_) => None,
        }
    }

    pub fn actual_root_admission_v2(&self) -> Option<&Arc<RoventRootAdmissionV2>> {
        match &self.active_root.as_ref()?.admission {
            OriginalExternalAdmission::V1(_) => None,
            OriginalExternalAdmission::V2(admission) => Some(admission),
        }
    }

    pub fn submit_actual_root(
        &mut self,
        job: ProfiledExternalJob,
    ) -> Result<ProfiledExternalDispatch, CheckerError> {
        let root = self
            .active_root
            .as_ref()
            .ok_or(invalid("actual external root was not admitted"))?;
        if !root
            .admission
            .original_context()
            .accepts()
            .map_err(|_| invalid("actual external root authority unavailable"))?
            || root.admission.requested_nodes() == Some(0)
            || Some(job.mode) != self.launch.mode
            || job.question.id.authority != root.admission.authority()
            || job.question.root_budget != root.budget
            || job.question.original_deadline.clock != root.admission.hard_deadline().clock
            || job.question.original_deadline.at > root.admission.hard_deadline().at
        {
            return Err(invalid(
                "actual external job differs from its original go/budget/deadline",
            ));
        }
        if let Some(first) = job.question.original_first_move {
            let first = rz_position::BoardMove::try_from(
                first
                    .decode()
                    .map_err(|_| invalid("external first move encoding"))?,
            )
            .map_err(|_| invalid("external first move Rules mapping"))?;
            if !root.admission.allowed_root_moves().contains(&first) {
                return Err(invalid(
                    "actual external first move is outside original searchmoves",
                ));
            }
        }
        self.launch
            .require_query(job.constraint, job.root_moves.is_some())?;
        self.worker.submit(job)
    }

    /// Original failed/expired terminals remain inspectable for cleanup. This
    /// source check does not make them accepted teacher labels or Rules proofs.
    pub fn validate_actual_completion(
        &self,
        completion: &ProfiledExternalCompletion,
        dispatch: &ProfiledExternalDispatch,
    ) -> Result<(), CheckerError> {
        let root = self
            .active_root
            .as_ref()
            .ok_or(invalid("actual external root was not admitted"))?;
        let original = dispatch.question();
        if original.id.authority != root.admission.authority()
            || original.root_budget != root.budget
            || original.original_deadline.clock != root.admission.hard_deadline().clock
            || original.original_deadline.at > root.admission.hard_deadline().at
        {
            return Err(invalid(
                "external completion is from a different original go/root scope",
            ));
        }
        self.validate_completion_for_budget(completion, dispatch, root.budget)
    }
}
