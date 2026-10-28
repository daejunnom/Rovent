//! 실제 profiled external R 결과를 원 Search·archive owner에 연결한다.
//!
//! 원 packet Arc, opaque source와 reservation을 함께 보존한다. legacy Combined
//! Job·native credit·Rules proof·기존 V action witness로 변환하지 않는다.

use super::*;
use rz_contracts::{ExecutionId, rovent::ReciprocalQuestionId};
use rz_search::cpu_checker::CheckerError;
use rz_search::external_cpu::{
    ExternalGoConstraint, ProfiledExternalCompletion, ProfiledExternalCurrentRootFence,
    ProfiledExternalDispatch, ProfiledExternalProcessCloseProof, ProfiledExternalRootSource,
};
use rz_search::pals::engine::adaptive_port::{
    AdaptiveProfiledExternalAdmission, AdaptiveProfiledExternalCommit,
    AdaptiveProfiledExternalPreparation, ProfiledExternalRootRetirement,
};
use rz_search::pals::rovent_frontier::{FrontierExternalReservation, FrontierExternalWork};
use rz_search::pals::store::ProfiledForeignAppendPermit;

#[derive(Default)]
pub(super) struct ProfiledDriverState {
    pub(super) source: Option<ProfiledExternalRootSource>,
    pub(super) transfer_question: Option<Digest>,
    pub(super) permit: Option<(Digest, ProfiledForeignAppendPermit)>,
    pub(super) committed: HashMap<Digest, RetainedProfiledCommit>,
    retirement: Option<ProfiledExternalRootRetirement>,
    retirement_boundary: Option<ProfiledRetirementBoundary>,
}

enum ProfiledRetirementBoundary {
    Root(ProfiledExternalCurrentRootFence),
    Process(ProfiledExternalProcessCloseProof),
}

pub(super) struct RetainedProfiledCommit {
    work: CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    source: ProfiledExternalRootSource,
    reservation: FrontierExternalReservation,
    completion: Arc<ProfiledExternalCompletion>,
    requested_consumption: bool,
    committed: AdaptiveProfiledExternalCommit,
}

impl RetainedProfiledCommit {
    fn matches(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        source: &ProfiledExternalRootSource,
        reservation: FrontierExternalReservation,
        completion: &Arc<ProfiledExternalCompletion>,
    ) -> bool {
        self.source.same_source(source)
            && self.reservation == reservation
            && Arc::ptr_eq(&self.completion, completion)
            && self.work.frame == work.frame
            && self.work.original_first_move == work.original_first_move
            && self.work.original_deadline == work.original_deadline
            && self.work.input.stage == work.input.stage
            && self.work.input.question_digest == work.input.question_digest
            && self.work.input.public_context == work.input.public_context
            && self.work.input.view_seal == work.input.view_seal
            && Arc::ptr_eq(&self.work.input.view, &work.input.view)
            && Arc::ptr_eq(&self.work.input.question, &work.input.question)
    }

    // 실제 Core 반환의 읽기 결과다. 새 Store 소비·source·packet 권한은 발급하지 않는다.
    fn observed(&self) -> AdaptiveProfiledExternalCommit {
        AdaptiveProfiledExternalCommit {
            observation: self.committed.observation,
            consumed: self.committed.consumed,
        }
    }
}

fn profile_error(error: PalsError) -> AdaptiveManagerError {
    match error {
        PalsError::Checker(error) => AdaptiveManagerError::ProfiledExternal(error),
        PalsError::Capacity => {
            AdaptiveManagerError::Capacity("profiled external retained questions")
        }
        other => semantic(other),
    }
}

fn invalid(message: &'static str) -> AdaptiveManagerError {
    AdaptiveManagerError::ProfiledExternal(CheckerError::Invalid(message))
}

impl<M: RoleModel + 'static> AdaptivePortDriver<M> {
    pub fn attach_rovent_external_frontier(
        &mut self,
        registration: FrontierRegistration,
        limits: FrontierLimits,
        actual: AdaptiveFrontierStartup<'_>,
        source: &ProfiledExternalRootSource,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), AdaptiveManagerError> {
        if self.profiled.source.is_some()
            || self.profiled.retirement.is_some()
            || self.profiled.transfer_question.is_some()
            || self.profiled.permit.is_some()
            || !self.profiled.committed.is_empty()
        {
            return Err(invalid("profiled frontier already owns source custody"));
        }
        self.hot()?
            .attach_rovent_external_frontier(registration, limits, actual, source, clock, now)
            .map_err(AdaptiveManagerError::Frontier)?;
        self.profiled.source = Some(source.clone());
        Ok(())
    }

    /// 실제 부착된 source의 원 cleanup ceiling을 보존한다. 탐색·현재 마감이나
    /// admission을 갱신하지 않으며, 첫 질문 전 등록 조건은 Core가 검증한다.
    pub fn register_profiled_external_cleanup_deadline(
        &mut self,
        source: &ProfiledExternalRootSource,
        original: Instant,
    ) -> Result<(), AdaptiveManagerError> {
        self.profile_source(source)?;
        self.hot()?
            .register_profiled_external_cleanup_deadline(source, original)
            .map_err(profile_error)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn profile_prepare(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        config: &ExternalCpuConfig,
        source: &ProfiledExternalRootSource,
        id: ReciprocalQuestionId,
        execution: ExecutionId,
        constraint: ExternalGoConstraint,
        quantum: Duration,
        signal: CompletionSignal,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<AdaptiveProfiledExternalPreparation, AdaptiveManagerError> {
        self.profile_source(source)?;
        if self.closing {
            return Err(invalid("profiled root admission is closed"));
        }
        self.hot()?
            .prepare_profiled_external(
                work, config, source, id, execution, constraint, quantum, signal, clock, now,
            )
            .map_err(profile_error)
    }

    pub(super) fn profile_admitted(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        source: &ProfiledExternalRootSource,
        dispatch: &ProfiledExternalDispatch,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<AdaptiveProfiledExternalAdmission, AdaptiveManagerError> {
        self.profile_source(source)?;
        self.hot()?
            .profiled_external_admitted(work, source, dispatch, clock, now)
            .map_err(profile_error)
    }

    fn profile_source(
        &self,
        source: &ProfiledExternalRootSource,
    ) -> Result<(), AdaptiveManagerError> {
        if source.authority() != self.authority
            || !self
                .profiled
                .source
                .as_ref()
                .is_some_and(|original| original.same_source(source))
        {
            return Err(invalid(
                "profiled packet belongs to another original root source",
            ));
        }
        Ok(())
    }

    fn profile_raw(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        source: &ProfiledExternalRootSource,
        reservation: FrontierExternalReservation,
        completion: &Arc<ProfiledExternalCompletion>,
        consume_current: bool,
    ) -> Result<AdaptiveProfiledExternalCommit, AdaptiveManagerError> {
        self.profile_source(source)?;
        let key = work.input.question_digest;
        if let Some(original) = self.profiled.committed.get(&key) {
            if !original.matches(work, source, reservation, completion)
                || (consume_current && !original.requested_consumption)
            {
                return Err(invalid(
                    "profiled commit retry replaced its original packet or mode",
                ));
            }
            return Ok(original.observed());
        }
        if self.profiled.committed.len() >= MAX_RETAINED_QUESTIONS {
            return Err(AdaptiveManagerError::Capacity(
                "profiled committed questions",
            ));
        }
        // 확인 전에는 move-only permit을 꺼내지 않는다. owner가 archive에 이동한
        // 동안의 AuxiliaryPending이 원 permit을 떨어뜨리면 안 된다.
        self.hot()?;
        let permit = match self.profiled.permit.take() {
            Some((question, permit)) if question == key => Some(permit),
            Some(other) => {
                self.profiled.permit = Some(other);
                return Err(AdaptiveManagerError::AuxiliaryPending);
            }
            None => None,
        };
        let result = match permit {
            Some(permit) => self.hot()?.commit_profiled_external_permitted(
                work,
                source,
                reservation,
                completion,
                consume_current,
                permit,
            ),
            None => self.hot()?.commit_profiled_external(
                work,
                source,
                reservation,
                completion,
                consume_current,
            ),
        };
        match result {
            Ok(committed) => {
                let observed = AdaptiveProfiledExternalCommit {
                    observation: committed.observation,
                    consumed: committed.consumed,
                };
                self.profiled.committed.insert(
                    key,
                    RetainedProfiledCommit {
                        work: work.clone(),
                        source: source.clone(),
                        reservation,
                        completion: Arc::clone(completion),
                        requested_consumption: consume_current,
                        committed,
                    },
                );
                Ok(observed)
            }
            Err(PalsError::Store(StoreError::ArchiveBudgetRequired)) => {
                let publication = self
                    .hot()?
                    .prepare_profiled_external_publication(work, source, reservation, completion)
                    .map_err(profile_error)?;
                self.transfer(
                    ArchiveOperation::PrepareProfiledForeign {
                        observation: publication.observation,
                        issued: publication.issued,
                    },
                    Some(key),
                )?;
                Err(AdaptiveManagerError::AuxiliaryPending)
            }
            Err(error) => Err(profile_error(error)),
        }
    }

    pub(super) fn profile_commit(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        source: &ProfiledExternalRootSource,
        reservation: FrontierExternalReservation,
        completion: &Arc<ProfiledExternalCompletion>,
    ) -> Result<CheckedStep<AdaptivePortQuestion>, AdaptiveManagerError> {
        let committed = self.profile_raw(work, source, reservation, completion, true)?;
        if !committed.consumed {
            return Err(invalid(
                "profiled raw terminal was not accepted for this live question",
            ));
        }
        // V-free 프로필이다. 기존 native/legacy V action hook으로 비용 witness를
        // 발급하지 않으며 actual profiled packet timing은 원 manager가 보존한다.
        let step = self
            .hot()?
            .after_profiled_external(work, &committed)
            .map_err(profile_error)?;
        let materialized = self.hot()?.materialize_step(step).map_err(profile_error)?;
        self.profiled.committed.remove(&work.input.question_digest);
        Ok(checked(materialized))
    }

    pub(super) fn profile_observe(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        source: &ProfiledExternalRootSource,
        reservation: FrontierExternalReservation,
        completion: &Arc<ProfiledExternalCompletion>,
    ) -> Result<(), AdaptiveManagerError> {
        // raw 보존만 수행한다. consume=false를 refinement나 목표 깊이의 완료로
        // 바꾸지 않으며, stale/late packet의 오류와 원 Arc는 Core에 계속 남는다.
        self.profile_raw(work, source, reservation, completion, false)?;
        Ok(())
    }

    pub(super) fn profile_consume_reciprocal(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        source: &ProfiledExternalRootSource,
        reservation: FrontierExternalReservation,
        completion: &Arc<ProfiledExternalCompletion>,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::CheckedProfiledExternalResponse,
        AdaptiveManagerError,
    > {
        let committed = self.profile_raw(work, source, reservation, completion, true)?;
        // The semantic owner binds this actual packet to the accepted parent's
        // original peer question. No ordinary refinement transition runs here.
        let response = self
            .hot()?
            .after_profiled_reciprocal_cpu(work, source, reservation, completion, &committed)
            .map_err(profile_error)?;
        self.profiled.committed.remove(&work.input.question_digest);
        Ok(response)
    }

    pub(super) fn profile_reject(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<(), AdaptiveManagerError> {
        self.hot()?
            .reject_profiled_external(work)
            .map_err(profile_error)
    }

    pub(super) fn profile_work(&self) -> Option<FrontierExternalWork> {
        self.port
            .as_deref()
            .and_then(|port| port.profiled_external_work())
    }

    pub(super) fn profile_retained_host_bytes(&self) -> Option<u64> {
        // private packet/config allocator의 실측·용량 getter가 없으면 전체를 0으로
        // 계산하지 않는다. 공유 backing의 물리 절약량도 Arc 개수로 추정하지 않는다.
        if !self.profiled.committed.is_empty()
            || self.profiled.permit.is_some()
            || self.profiled.transfer_question.is_some()
            || self.transfer.is_some()
        {
            return None;
        }
        let live = self
            .port
            .as_deref()?
            .profiled_external_retained_host_bytes()?;
        let retired = self
            .profiled
            .retirement
            .as_ref()
            .map_or(Some(0), |value| value.owned_host_bytes())?;
        live.checked_add(retired)?
            .checked_add(u64::try_from(std::mem::size_of::<ProfiledDriverState>()).ok()?)
    }

    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(super) fn profile_validate(
        &self,
        source: &ProfiledExternalRootSource,
        root: &Arc<crate::rovent_root_admission::RoventRootAdmission>,
    ) -> Result<(), AdaptiveManagerError> {
        use rz_runtime::contracts::ContractClock;
        self.profile_source(source)?;
        if !root
            .original_context()
            .accepts()
            .map_err(|_| invalid("profiled root revoked"))?
            || root.authority() != self.authority
            || source.clock().domain != root.runtime_clock().domain()
            || source.clock().origin != root.runtime_clock().origin()
            || self.hard_deadline != root.original_context().control.deadline
        {
            return Err(invalid(
                "profiled manager registration differs from original root clock",
            ));
        }
        let port = self
            .port
            .as_deref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?;
        let frontier = port
            .rovent_frontier()
            .ok_or(invalid("profiled frontier absent"))?;
        let registered = frontier.registration();
        if !Arc::ptr_eq(&registered.root_rules, &self.rules)
            || !self
                .rules
                .rules()
                .snapshot()
                .same_state(root.root_snapshot())
            || registered.original_deadline != root.hard_deadline()
            || registered.root_budget != root.root_budget_digest()
            || registered.goal != root.goal()
            || registered.nodes != root.requested_nodes()
            || registered.root_moves.allowed().as_ref() != root.allowed_root_moves().as_ref()
            || registered.native_ledger.is_some()
        {
            return Err(invalid(
                "profiled original Rules, root domain or budget differs",
            ));
        }
        port.validate_profiled_external_registration(
            source,
            registered.mode,
            root.root_budget_digest(),
        )
        .map_err(profile_error)
    }

    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(super) fn profile_validate_v2(
        &self,
        source: &ProfiledExternalRootSource,
        root: &Arc<crate::rovent_root_admission::source_seal::RoventRootAdmissionV2>,
    ) -> Result<(), AdaptiveManagerError> {
        use rz_runtime::contracts::ContractClock;
        self.profile_source(source)?;
        if !root
            .original_context()
            .accepts()
            .map_err(|_| invalid("profiled V2 root revoked"))?
            || root.authority() != self.authority
            || source.clock().domain != root.runtime_clock().domain()
            || source.clock().origin != root.runtime_clock().origin()
            || self.hard_deadline != root.original_context().control.deadline
        {
            return Err(invalid(
                "profiled V2 manager registration differs from original root clock",
            ));
        }
        let budget = root
            .descriptor()
            .root_budget_digest()
            .map_err(|_| invalid("profiled V2 original root budget invalid"))?;
        let port = self
            .port
            .as_deref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?;
        let frontier = port
            .rovent_frontier()
            .ok_or(invalid("profiled frontier absent"))?;
        let registered = frontier.registration();
        if !Arc::ptr_eq(&registered.root_rules, &self.rules)
            || !self
                .rules
                .rules()
                .snapshot()
                .same_state(root.root_snapshot())
            || registered.original_deadline != root.hard_deadline()
            || registered.root_budget != budget
            || registered.goal != root.goal()
            || registered.nodes != root.requested_nodes()
            || registered.root_moves.allowed().as_ref() != root.allowed_root_moves().as_ref()
            || registered.native_ledger.is_some()
        {
            return Err(invalid(
                "profiled V2 original Rules, root domain or budget differs",
            ));
        }
        port.validate_profiled_external_registration(source, registered.mode, budget)
            .map_err(profile_error)
    }

    pub(super) fn profile_release(
        &mut self,
        source: &ProfiledExternalRootSource,
        fence: &ProfiledExternalCurrentRootFence,
    ) -> Result<(), AdaptiveManagerError> {
        self.profile_source(source)?;
        if self.transfer.is_some()
            || self.profiled.transfer_question.is_some()
            || self.profiled.permit.is_some()
        {
            return Err(AdaptiveManagerError::AuxiliaryPending);
        }
        if self.profiled.retirement.is_some() {
            return if self
                .profiled
                .retirement_boundary
                .as_ref()
                .is_some_and(|boundary| {
                    matches!(boundary, ProfiledRetirementBoundary::Root(original) if
                        original.source().same_source(source)
                            && original.source().same_source(fence.source())
                            && original.authority() == fence.authority()
                            && original.sequence() == fence.sequence()
                            && original.process_identity() == fence.process_identity()
                            && fence.permits_reuse()
                    )
                }) {
                Ok(())
            } else {
                Err(invalid(
                    "profiled root retirement retry replaced its actual fence",
                ))
            };
        }
        let retirement = self
            .hot()?
            .release_profiled_external_root(source, fence)
            .map_err(profile_error)?;
        self.profiled.retirement = Some(retirement);
        self.profiled.retirement_boundary = Some(ProfiledRetirementBoundary::Root(fence.clone()));
        self.profiled.committed.clear();
        Ok(())
    }

    fn profile_release_closed(
        &mut self,
        source: &ProfiledExternalRootSource,
        proof: &ProfiledExternalProcessCloseProof,
    ) -> Result<(), AdaptiveManagerError> {
        self.profile_source(source)?;
        if self.transfer.is_some()
            || self.profiled.transfer_question.is_some()
            || self.profiled.permit.is_some()
        {
            return Err(AdaptiveManagerError::AuxiliaryPending);
        }
        if !proof.source().same_source(source)
            || proof.authority() != self.authority
            || !proof.permits_semantic_closure()
        {
            return Err(invalid(
                "profiled closed process proof belongs to another owner",
            ));
        }
        if self.profiled.retirement.is_some() {
            return if self
                .profiled
                .retirement_boundary
                .as_ref()
                .is_some_and(|boundary| {
                    matches!(boundary, ProfiledRetirementBoundary::Process(original) if
                        original.source().same_source(proof.source())
                            && original.shutdown_receipt() == proof.shutdown_receipt()
                            && original.accepted_jobs() == proof.accepted_jobs()
                            && original.terminal_controls() == proof.terminal_controls()
                            && original.drained_terminals() == proof.drained_terminals()
                            && original.permits_semantic_closure()
                    )
                }) {
                Ok(())
            } else {
                Err(invalid(
                    "profiled retirement retry replaced its closed process proof",
                ))
            };
        }
        let retirement = self
            .hot()?
            .release_profiled_external_closed_root(source, proof)
            .map_err(profile_error)?;
        self.profiled.retirement = Some(retirement);
        self.profiled.retirement_boundary =
            Some(ProfiledRetirementBoundary::Process(proof.clone()));
        self.profiled.committed.clear();
        Ok(())
    }

    /// 원 archive owner와 actual native/external fence를 모두 확인한 뒤에만
    /// 포트와 미게시 원 packet의 move-only retirement를 함께 돌려준다.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    #[allow(clippy::result_large_err)]
    pub fn into_profiled_port(
        mut self,
        receipt: &crate::pals_adaptive::ProfiledExternalShutdownReceipt,
    ) -> Result<(AdaptiveEnginePort<M>, ProfiledExternalRootRetirement), (Self, AdaptiveManagerError)>
    {
        if !receipt.permits_semantic_closure()
            || receipt.authority() != self.authority
            || !self.archive_joined
            || self.transfer.is_some()
            || self.foreign_permit.is_some()
            || self.foreign_transfer_question.is_some()
            || self.profiled.transfer_question.is_some()
            || self.profiled.permit.is_some()
            || self.quarantine.is_some()
            || self.unexpected_archive_owner.is_some()
        {
            return Err((self, AdaptiveManagerError::PhysicalUnknown));
        }
        // 닫힌 process를 재사용 가능한 root로 표시하지 않는다. 이미 발급된 원
        // current fence 또는 실제 join 전용 proof 중 그 시점의 원 증거만 소비한다.
        let release = if let Some(proof) = receipt.process_close_proof() {
            self.profile_release_closed(receipt.source(), proof)
        } else if let Some(fence) = receipt.external_root_fence() {
            self.profile_release(receipt.source(), fence)
        } else {
            Err(invalid(
                "profiled semantic closure lacks actual root or process-close proof",
            ))
        };
        if let Err(error) = release {
            return Err((self, error));
        }
        let Some(port) = self.port.take() else {
            return Err((self, AdaptiveManagerError::PhysicalUnknown));
        };
        let retirement = self
            .profiled
            .retirement
            .take()
            .expect("checked original retirement");
        Ok((*port, retirement))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_profiled_capability_remains_typed_unsupported() {
        assert!(matches!(
            profile_error(PalsError::Checker(CheckerError::Unsupported(
                "no reciprocal resume"
            ))),
            AdaptiveManagerError::ProfiledExternal(CheckerError::Unsupported(
                "no reciprocal resume"
            ))
        ));
    }

    #[test]
    fn actual_profiled_error_keeps_original_stage_and_code() {
        assert!(matches!(
            profile_error(PalsError::Checker(CheckerError::External {
                stage: "root_quiesce",
                code: "physical_completion_unknown",
            })),
            AdaptiveManagerError::ProfiledExternal(CheckerError::External {
                stage: "root_quiesce",
                code: "physical_completion_unknown",
            })
        ));
    }
}
