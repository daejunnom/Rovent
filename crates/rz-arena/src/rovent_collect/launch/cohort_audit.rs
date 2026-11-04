//! Cohort native lifecycle의 bounded 숫자 관측이다. 원 학습 입력은 실제
//! UtilityTrialMaterializedSource가 보존하며 이 관측은 권한을 발급하지 않는다.
use super::*;
use rz_search::pals::coordinator::RoleFailure;
use rz_uci::pals_native::ticketed::{
    NativeTicketQuestion, TicketExecutionEvidence, TicketPreparation,
    VirtualReviewCollectionObserver, VirtualReviewPhysicalInvocationEvidence,
    VirtualReviewPreparedTicket,
};

#[derive(Default)]
pub(super) struct NativeAudit {
    submitted: u64,
    completed: u64,
    retired: u64,
    active: BTreeMap<(u64, u64), RoleTicketEnvelope>,
}
pub(super) struct NativeAuditObserver(pub Arc<Mutex<NativeAudit>>);
impl VirtualReviewCollectionObserver for NativeAuditObserver {
    fn submitted(
        &mut self,
        ticket: &RoleTicketEnvelope,
        _: &TicketPreparation<ReviewView, NativeTicketQuestion>,
        prepared: &VirtualReviewPreparedTicket,
    ) -> Result<(), RoleFailure> {
        let mut state = self
            .0
            .try_lock()
            .map_err(|_| RoleFailure::Backend("cohort native audit unavailable".into()))?;
        let key = (ticket.id.epoch.0, ticket.id.sequence);
        if state.active.len() >= 64
            || state.submitted >= 8192
            || prepared.representation != ticket.representation
            || state.active.contains_key(&key)
        {
            return Err(RoleFailure::Backend(
                "cohort native audit bound/representation/duplicate".into(),
            ));
        }
        state.active.insert(key, *ticket);
        state.submitted += 1;
        Ok(())
    }
    fn completed(
        &mut self,
        ticket: &RoleTicketEnvelope,
        _: PhysicalRoleCompletion,
        _: Option<&RoleOutput>,
        _: Option<TicketExecutionEvidence>,
        _: Option<VirtualReviewPhysicalInvocationEvidence>,
    ) -> Result<(), RoleFailure> {
        let mut state = self
            .0
            .try_lock()
            .map_err(|_| RoleFailure::Backend("cohort native audit unavailable".into()))?;
        if state.active.get(&(ticket.id.epoch.0, ticket.id.sequence)) != Some(ticket)
            || state.completed >= state.submitted
        {
            return Err(RoleFailure::Backend(
                "cohort native completion has no original submission".into(),
            ));
        }
        state.completed += 1;
        Ok(())
    }
    fn retired(&mut self, ticket: &RoleTicketEnvelope, _: bool) -> Result<(), RoleFailure> {
        let mut state = self
            .0
            .try_lock()
            .map_err(|_| RoleFailure::Backend("cohort native audit unavailable".into()))?;
        if state.active.get(&(ticket.id.epoch.0, ticket.id.sequence)) != Some(ticket) {
            return Err(RoleFailure::Backend(
                "cohort native retirement has no original submission".into(),
            ));
        }
        state
            .active
            .remove(&(ticket.id.epoch.0, ticket.id.sequence));
        state.retired += 1;
        Ok(())
    }
}
impl NativeAudit {
    pub(super) fn validate_closed(&self) -> Result<(), ArenaError> {
        check(
            self.active.is_empty() && self.retired == self.submitted,
            "cohort original native submissions have not all retired",
        )
    }
}
