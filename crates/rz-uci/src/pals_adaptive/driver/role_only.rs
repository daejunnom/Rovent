//! Thin NoResearcher owner. The semantic port selects exact questions; the
//! role-only manager and native worker own submission, inference and fences.
use super::*;
use crate::pals_adaptive::{RoleOnlySearchDriver, RoleOnlyShutdownReceipt};
use rz_search::pals::coordinator::AcceptedRoleStep;
use rz_search::pals::engine::NoResearcher;
use rz_search::pals::engine::pc_port::{PcPortQuestion, PcPortResult, PcSemanticPort};

pub struct PcPortDriver<M: RoleModel> {
    port: PcSemanticPort<M>,
    original_root_budget: Option<Digest>,
    original_rules: Arc<ContractState>,
}
impl<M: RoleModel> PcPortDriver<M> {
    pub fn new(port: PcSemanticPort<M>) -> Self {
        let original_root_budget = port.original_root_budget();
        let original_rules = port.root_rules();
        Self {
            port,
            original_root_budget,
            original_rules,
        }
    }
    pub fn port(&self) -> &PcSemanticPort<M> {
        &self.port
    }
    pub fn port_mut(&mut self) -> &mut PcSemanticPort<M> {
        &mut self.port
    }
    pub fn result(&self) -> PcPortResult {
        self.port.finish_materialized()
    }

    #[allow(clippy::result_large_err)]
    pub fn into_port(
        self,
        receipt: &RoleOnlyShutdownReceipt,
    ) -> Result<PcSemanticPort<M>, (Self, AdaptiveManagerError)> {
        if !receipt.permits_semantic_closure()
            || (self.port.requires_shared_nnuec_readout()
                && !receipt
                    .shared_nnuec_closure()
                    .is_some_and(|closure| self.port.shared_nnuec_closure_matches(closure)))
            || (!self.port.requires_shared_nnuec_readout()
                && receipt.shared_nnuec_closure().is_some())
            || receipt.authority != self.port.authority()
            || receipt.mode != self.port.mode()
            || self.original_root_budget != Some(receipt.root_budget)
            || !Arc::ptr_eq(&self.original_rules, &self.port.root_rules())
            || self
                .port
                .original_root_budget()
                .is_some_and(|actual| actual != receipt.root_budget)
        {
            return Err((self, AdaptiveManagerError::PhysicalUnknown));
        }
        Ok(self.port)
    }
}

impl<M: RoleModel> RoleOnlySearchDriver<PcPortQuestion> for PcPortDriver<M> {
    fn researcher(&self) -> &NoResearcher {
        self.port.researcher()
    }
    fn validate_registration(
        &self,
        authority: SearchAuthority,
        mode: rz_contracts::rovent::ExecutionMode,
        hard_deadline: Instant,
        root_budget: Digest,
    ) -> Result<(), AdaptiveManagerError> {
        if authority != self.port.authority()
            || mode != self.port.mode()
            || hard_deadline != self.port.hard_deadline()
            || self.original_root_budget != Some(root_budget)
            || !Arc::ptr_eq(&self.original_rules, &self.port.root_rules())
            || self.port.original_root_budget() != Some(root_budget)
        {
            return Err(AdaptiveManagerError::Invalid(
                "P/C exact semantic registration",
            ));
        }
        Ok(())
    }
    fn next_role(
        &mut self,
    ) -> Result<Option<FrameInput<ReviewView, PcPortQuestion>>, AdaptiveManagerError> {
        self.port.next_role().map_err(semantic)
    }
    fn register_shared_nnuec_readout(
        &mut self,
        model: Arc<rz_search::cpu_value::nnuec::v2::Model>,
    ) -> Result<(), AdaptiveManagerError> {
        self.port
            .register_shared_nnuec_readout(model)
            .map_err(|error| AdaptiveManagerError::Semantic(error.to_string()))
    }
    fn requires_shared_nnuec_readout(&self) -> bool {
        self.port.requires_shared_nnuec_readout()
    }
    fn prepare_shared_nnuec_readout(
        &mut self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        model: &Arc<rz_search::cpu_value::nnuec::v2::Model>,
        canceled: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<rz_search::pals::engine::pc_port::PcNnuecReadoutRequest, AdaptiveManagerError> {
        self.port
            .prepare_shared_nnuec_readout(input, model, canceled)
            .map_err(|error| AdaptiveManagerError::Semantic(error.to_string()))
    }
    fn consume_shared_nnuec_readout(
        &mut self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        seal: &rz_search::pals::engine::pc_port::PcNnuecReadoutSeal,
        model: &Arc<rz_search::cpu_value::nnuec::v2::Model>,
        completion: &rz_search::cpu_value::nnuec::v2::review::Completion,
    ) -> Result<rz_search::pals::engine::pc_port::PcNnuecReadoutObservation, AdaptiveManagerError>
    {
        self.port
            .consume_shared_nnuec_readout(input, seal, model, completion)
            .map_err(|error| AdaptiveManagerError::Semantic(error.to_string()))
    }
    fn shared_nnuec_readout_consumed(
        &self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
    ) -> Result<bool, AdaptiveManagerError> {
        self.port
            .shared_nnuec_readout_consumed(input)
            .map_err(|error| AdaptiveManagerError::Semantic(error.to_string()))
    }
    #[allow(clippy::type_complexity)]
    fn close_shared_nnuec_readouts(
        &self,
        actor: &rz_search::cpu_value::nnuec::v2::review::PreparationActor,
        acknowledgement: rz_search::cpu_value::nnuec::v2::retirement::NnuecCacheReleaseAck,
        terminals: Vec<rz_search::pals::engine::pc_port::PcNnuecReadoutTerminal>,
    ) -> Result<
        rz_search::pals::engine::pc_port::PcSharedReadoutRootClosure,
        (
            AdaptiveManagerError,
            rz_search::cpu_value::nnuec::v2::retirement::NnuecCacheReleaseAck,
            Vec<rz_search::pals::engine::pc_port::PcNnuecReadoutTerminal>,
        ),
    > {
        self.port
            .close_shared_nnuec_readouts(actor, acknowledgement, terminals)
            .map_err(|(error, acknowledgement, terminals)| {
                (
                    AdaptiveManagerError::Semantic(error.to_string()),
                    acknowledgement,
                    terminals,
                )
            })
    }
    fn shared_nnuec_closure_matches(
        &self,
        closure: &rz_search::pals::engine::pc_port::PcSharedReadoutRootClosure,
    ) -> bool {
        self.port.shared_nnuec_closure_matches(closure)
    }
    fn role_submitted(
        &mut self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(), AdaptiveManagerError> {
        self.port.role_submitted(input, ticket).map_err(semantic)
    }
    fn validate_role(
        &self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
    ) -> Result<(), AdaptiveManagerError> {
        self.port
            .validate_role(input, ticket, output)
            .map_err(semantic)
    }
    fn consume_accepted(
        &mut self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: Option<TicketExecutionEvidence>,
    ) -> Result<(), AdaptiveManagerError> {
        let physical = physical.map(|evidence| AdaptivePortRoleEvidence {
            execution: evidence.execution,
            completed_nn_inputs: evidence.completed_nn_inputs,
            measurement_complete: evidence.measurement_complete,
            physical_fresh: evidence.physical_fresh,
        });
        self.port
            .consume_accepted(input, accepted, output, physical)
            .map_err(semantic)
    }
    fn role_discarded(
        &mut self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        ticket: Option<&RoleTicketEnvelope>,
    ) -> Result<(), AdaptiveManagerError> {
        self.port.role_discarded(input, ticket).map_err(semantic)
    }
    fn role_consumption_recorded(
        &self,
        ticket: &RoleTicketEnvelope,
        input: &FrameInput<ReviewView, PcPortQuestion>,
    ) -> Result<bool, AdaptiveManagerError> {
        self.port
            .role_consumption_recorded(ticket, input)
            .map_err(semantic)
    }
    fn is_complete(&self) -> bool {
        self.port.is_complete()
    }
    fn close_role_call_budget(
        &mut self,
        admitted: u64,
        limit: u64,
    ) -> Result<(), AdaptiveManagerError> {
        self.port
            .close_role_call_budget(admitted, limit)
            .map_err(semantic)
    }
    fn quiesce_satisfied_goal(&mut self) -> Result<(), AdaptiveManagerError> {
        self.port
            .quiesce_satisfied_goal()
            .map_err(AdaptiveManagerError::Frontier)
    }
    fn revoke(&mut self) -> Result<(), AdaptiveManagerError> {
        self.port.revoke();
        Ok(())
    }
    fn review_lease_admission(&self) -> Result<AdaptiveReviewLeaseAdmission, AdaptiveManagerError> {
        self.port.review_lease_admission().map_err(semantic)
    }
    fn validate_collection_admission(
        &self,
        admission: &crate::pals_adaptive::RoventCollectionRootAdmission,
    ) -> Result<(), AdaptiveManagerError> {
        let frontier = self
            .port
            .rovent_frontier()
            .ok_or(AdaptiveManagerError::Invalid(
                "P/C collection has no actual attached frontier",
            ))?;
        let registered = frontier.registration();
        if !admission.validates_registration(
            registered.authority,
            registered.mode,
            registered.root_budget,
            registered.original_deadline,
        ) || !Arc::ptr_eq(admission.rules(), &self.original_rules)
            || !Arc::ptr_eq(admission.root_moves(), &registered.root_moves)
            || admission.selective_policy() != registered.selective_policy
        {
            return Err(AdaptiveManagerError::Invalid(
                "P/C collection is outside its actual frontier owner",
            ));
        }
        Ok(())
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn validate_collection_admission_v2(
        &self,
        admission: &crate::pals_adaptive::RoventCollectionRootAdmissionV2,
    ) -> Result<(), AdaptiveManagerError> {
        let frontier = self
            .port
            .rovent_frontier()
            .ok_or(AdaptiveManagerError::Invalid(
                "P/C V2 collection has no actual attached frontier",
            ))?;
        let registered = frontier.registration();
        let root = admission.root_admission();
        if root.authority() != registered.authority
            || root.hard_deadline() != registered.original_deadline
            || admission.root_budget() != registered.root_budget
            || admission
                .native_origin()
                .sources()
                .lock()
                .manifest
                .engine
                .mode
                != registered.mode
            || !Arc::ptr_eq(admission.rules(), &self.original_rules)
            || !Arc::ptr_eq(admission.root_moves(), &registered.root_moves)
            || admission.selective_policy() != registered.selective_policy
        {
            return Err(AdaptiveManagerError::Invalid(
                "P/C V2 collection is outside its actual frontier owner",
            ));
        }
        Ok(())
    }
}

#[cfg(feature = "onnx-cpu")]
impl crate::pals_adaptive::NativeQuestionProjection for PcPortQuestion {
    fn project_native_question(
        &self,
    ) -> Result<
        crate::pals_native::ticketed::NativeTicketQuestion,
        rz_search::pals::coordinator::RoleFailure,
    > {
        Ok(crate::pals_native::ticketed::NativeTicketQuestion {
            position: self.position().clone(),
            context: self.context().clone(),
            deadline: self.deadline(),
            legacy_full_records: None,
        })
    }
    fn owned_host_bytes(&self) -> Option<u64> {
        PcPortQuestion::owned_host_bytes(self)
    }
}
