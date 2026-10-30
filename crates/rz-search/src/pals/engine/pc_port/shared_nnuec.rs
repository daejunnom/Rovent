//! R-free shared NNUEC readouts over original P/C questions.
//! This owner never creates a researcher, CPU worker, TT or node credit.

use super::*;
use crate::cpu_value::CpuValueIdentity;
use crate::cpu_value::nnuec::v2::Model;
use crate::cpu_value::nnuec::v2::retirement::NnuecCacheReleaseAck;
use crate::cpu_value::nnuec::v2::review::{
    Completion, NnuecPreparationTerminalAck, PreparationActor, PreparationError,
    PreparationObservation, PreparationStatus, Provenance, Request, RequestIdentity,
};
use serde::Serialize;
use std::sync::{Mutex, Weak, atomic::AtomicBool};

const MAX_SHARED_REQUESTS: usize = 4096;

#[derive(Debug)]
pub enum PcNnuecReadoutError {
    Semantic(PalsError),
    Preparation(PreparationError),
    Invalid(&'static str),
}
impl std::fmt::Display for PcNnuecReadoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "P/C shared NNUEC readout: {self:?}")
    }
}
impl std::error::Error for PcNnuecReadoutError {}
impl From<PalsError> for PcNnuecReadoutError {
    fn from(error: PalsError) -> Self {
        Self::Semantic(error)
    }
}
impl From<PreparationError> for PcNnuecReadoutError {
    fn from(error: PreparationError) -> Self {
        Self::Preparation(error)
    }
}

pub(super) struct SharedNnuecState {
    model: Arc<Model>,
    requests: Mutex<Vec<SharedRequest>>,
}
struct SharedRequest {
    request: RequestIdentity,
    question: Weak<PcPortQuestion>,
    view: Weak<ReviewView>,
    digest: ContractDigest,
    consumed: bool,
}

pub struct PcNnuecReadoutRequest {
    request: Request,
    seal: PcNnuecReadoutSeal,
}
impl PcNnuecReadoutRequest {
    pub fn into_parts(self) -> (Request, PcNnuecReadoutSeal) {
        (self.request, self.seal)
    }
}

struct ReadoutOrigin {
    owner: Weak<()>,
    question: Weak<PcPortQuestion>,
    view: Weak<ReviewView>,
    model: Weak<Model>,
    request: RequestIdentity,
    provenance: Provenance,
    authority: SearchAuthority,
    root_state: StateId,
    root_budget: Option<ContractDigest>,
    deadline: Deadline,
    question_digest: ContractDigest,
    endpoint: PositionSnapshot,
}

/// Move-only original Request lineage; it owns no View/Store pin.
pub struct PcNnuecReadoutSeal {
    origin: ReadoutOrigin,
}
impl PcNnuecReadoutSeal {
    fn matches_completion(&self, completion: &Completion) -> bool {
        let origin = &self.origin;
        completion.provenance == origin.provenance
            && completion.terminal_ack.provenance() == &origin.provenance
            && completion.terminal_ack.matches_request(&origin.request)
            && completion
                .terminal_ack
                .matches_original_weak(&origin.model, &origin.view)
            && completion.terminal_ack.allows_c_only_cleanup()
            && match &completion.result {
                Ok(bound) => {
                    completion.terminal_ack.status() == PreparationStatus::Completed
                        && bound.matches_request(&origin.request)
                        && bound.shared_readout().is_some_and(|readout| {
                            readout.matches_request(&origin.request)
                                && readout.endpoint().same_state(&origin.endpoint)
                        })
                }
                Err(_) => completion.terminal_ack.status() != PreparationStatus::Completed,
            }
    }
    /// Retiring an actual terminal never grants semantic consumption. Late or
    /// cancelled results still require their original physical/cache closure.
    #[allow(clippy::result_large_err)]
    pub fn retire(
        self,
        completion: Completion,
    ) -> Result<PcNnuecReadoutTerminal, (PcNnuecReadoutError, Self, Completion)> {
        if !self.matches_completion(&completion) {
            return Err((
                PcNnuecReadoutError::Invalid("foreign shared terminal"),
                self,
                completion,
            ));
        }
        let Completion {
            result,
            terminal_ack,
            observation,
            ..
        } = completion;
        drop(result);
        Ok(PcNnuecReadoutTerminal {
            origin: self.origin,
            terminal: Some(terminal_ack),
            observation: Some(observation),
        })
    }
    /// Owning the same non-Clone Request proves it was never moved into the
    /// actor. No fabricated worker terminal or zero-time service is issued.
    #[allow(clippy::result_large_err)]
    pub fn retire_unsubmitted(
        self,
        request: Request,
    ) -> Result<PcNnuecReadoutTerminal, (PcNnuecReadoutError, Self, Request)> {
        if !request.identity().same_request(&self.origin.request)
            || request.provenance() != &self.origin.provenance
            || !request.has_shared_readout()
        {
            return Err((
                PcNnuecReadoutError::Invalid("foreign unsubmitted Request"),
                self,
                request,
            ));
        }
        drop(request);
        Ok(PcNnuecReadoutTerminal {
            origin: self.origin,
            terminal: None,
            observation: None,
        })
    }
}

pub struct PcNnuecReadoutTerminal {
    origin: ReadoutOrigin,
    terminal: Option<NnuecPreparationTerminalAck>,
    observation: Option<PreparationObservation>,
}
impl std::fmt::Debug for PcNnuecReadoutTerminal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PcNnuecReadoutTerminal")
            .field("authority", &self.origin.authority)
            .field("question", &self.origin.question_digest)
            .field("submitted", &self.terminal.is_some())
            .finish_non_exhaustive()
    }
}
impl PcNnuecReadoutTerminal {
    pub fn submitted(&self) -> bool {
        self.terminal.is_some()
    }
    pub fn observation(&self) -> Option<PreparationObservation> {
        self.observation
    }
}

/// Small detached diagnostic record; no model WDL/proof/retention authority.
#[derive(Clone, Debug, Serialize)]
pub struct PcNnuecReadoutObservation {
    pub domain: &'static str,
    pub process_epoch: u64,
    pub game_generation: u64,
    pub root_generation: u64,
    pub question_sha256: String,
    pub view_sha256: String,
    pub input_sha256: String,
    pub context_sha256: String,
    pub model: CpuValueIdentity,
    pub value: f32,
    pub wdl: [f32; 3],
    pub raw_outputs: Vec<f32>,
    pub r_search_calls: u64,
    pub policy_authority: bool,
}

/// The port's complete issued Request roster plus the same actor's real cache
/// ACK. This does not replace the native physical fence or final actor join.
pub struct PcSharedReadoutRootClosure {
    owner: Weak<()>,
    model: Weak<Model>,
    authority: SearchAuthority,
    root_state: StateId,
    root_budget: Option<ContractDigest>,
    deadline: Deadline,
    terminals: Vec<PcNnuecReadoutTerminal>,
    // Snapshot of the original issued roster's actual semantic acceptance,
    // not inferred from submission or a physically completed preparation.
    semantic_consumed_requests: usize,
    _release: NnuecCacheReleaseAck,
}
impl std::fmt::Debug for PcSharedReadoutRootClosure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PcSharedReadoutRootClosure")
            .field("authority", &self.authority)
            .field("requests", &self.terminals.len())
            .finish_non_exhaustive()
    }
}
impl PcSharedReadoutRootClosure {
    pub fn issued_requests(&self) -> usize {
        self.terminals.len()
    }
    pub fn submitted_requests(&self) -> usize {
        self.terminals
            .iter()
            .filter(|terminal| terminal.submitted())
            .count()
    }
    /// Counts only original actor terminal observations in this bounded root
    /// roster. Completed, Cancelled, Failed and Expired remain distinct. A
    /// never-submitted Request has no observation and contributes to none.
    /// These counts are neither native NN inputs nor search/node authority.
    pub fn terminal_status_count(&self, status: PreparationStatus) -> usize {
        self.terminals
            .iter()
            .filter(|terminal| {
                terminal
                    .observation()
                    .is_some_and(|observation| observation.status() == status)
            })
            .count()
    }
    /// Actual successful same-question semantic consumptions at closure.
    /// A completed readout cancelled before consumption is not included.
    pub fn semantic_consumed_requests(&self) -> usize {
        self.semantic_consumed_requests
    }
}

impl<M: RoleModel> PcSemanticPort<M> {
    pub fn register_shared_nnuec_readout(
        &mut self,
        model: Arc<Model>,
    ) -> Result<(), PcNnuecReadoutError> {
        if self.shared_nnuec.is_some()
            || self.next_question != 1
            || !self.pending.is_empty()
            || self.mode.has_native_r()
            || self.counters.submitted_role_calls != 0
        {
            return Err(PcNnuecReadoutError::Invalid(
                "shared readout registration requires untouched R-free root",
            ));
        }
        self.shared_nnuec = Some(SharedNnuecState {
            model,
            requests: Mutex::new(Vec::new()),
        });
        Ok(())
    }
    pub fn requires_shared_nnuec_readout(&self) -> bool {
        self.shared_nnuec.is_some()
    }

    pub fn prepare_shared_nnuec_readout(
        &self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        model: &Arc<Model>,
        cancel: Arc<AtomicBool>,
    ) -> Result<PcNnuecReadoutRequest, PcNnuecReadoutError> {
        self.check_input(input)?;
        let state = self
            .shared_nnuec
            .as_ref()
            .ok_or(PcNnuecReadoutError::Invalid(
                "shared readout is not registered",
            ))?;
        if !Arc::ptr_eq(&state.model, model) {
            return Err(PcNnuecReadoutError::Invalid("foreign shared model owner"));
        }
        let mut requests = state
            .requests
            .lock()
            .map_err(|_| PcNnuecReadoutError::Invalid("shared Request roster poisoned"))?;
        if requests.len() >= MAX_SHARED_REQUESTS
            || requests.iter().any(|request| {
                request.digest == input.question_digest
                    || Weak::ptr_eq(&request.question, &Arc::downgrade(&input.question))
            })
        {
            return Err(PcNnuecReadoutError::Invalid(
                "shared Request duplicate/capacity",
            ));
        }
        requests
            .try_reserve(1)
            .map_err(|_| PcNnuecReadoutError::Invalid("shared Request allocation"))?;
        let request = Request::admit(
            model,
            Arc::clone(&input.view),
            &self.owner.semantic.stores,
            self.root_state_id(),
            self.authority.root.0,
            self.hard_deadline,
            cancel,
        )?;
        let request = match request.with_shared_readout(&input.question.position.snapshot()) {
            Ok(request) => request,
            Err((error, _original)) => return Err(error.into()),
        };
        let original = request.identity();
        let seal = PcNnuecReadoutSeal {
            origin: ReadoutOrigin {
                owner: Arc::downgrade(&self.identity),
                question: Arc::downgrade(&input.question),
                view: Arc::downgrade(&input.view),
                model: Arc::downgrade(model),
                request: original.clone(),
                provenance: request.provenance().clone(),
                authority: self.authority,
                root_state: self.root_state_id(),
                root_budget: self.original_root_budget(),
                deadline: self.deadline,
                question_digest: input.question_digest,
                endpoint: input.question.position.snapshot(),
            },
        };
        requests.push(SharedRequest {
            request: original,
            question: Arc::downgrade(&input.question),
            view: Arc::downgrade(&input.view),
            digest: input.question_digest,
            consumed: false,
        });
        Ok(PcNnuecReadoutRequest { request, seal })
    }

    pub fn consume_shared_nnuec_readout(
        &mut self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        seal: &PcNnuecReadoutSeal,
        model: &Arc<Model>,
        completion: &Completion,
    ) -> Result<PcNnuecReadoutObservation, PcNnuecReadoutError> {
        self.check_input(input)?;
        let state = self
            .shared_nnuec
            .as_ref()
            .ok_or(PcNnuecReadoutError::Invalid(
                "shared readout is not registered",
            ))?;
        let source = &seal.origin;
        if !Arc::ptr_eq(&state.model, model)
            || !Weak::ptr_eq(&source.model, &Arc::downgrade(model))
            || !self.shared_origin_matches(source)
            || !Weak::ptr_eq(&source.question, &Arc::downgrade(&input.question))
            || !Weak::ptr_eq(&source.view, &Arc::downgrade(&input.view))
            || source.question_digest != input.question_digest
            || !seal.matches_completion(completion)
        {
            return Err(PcNnuecReadoutError::Invalid(
                "shared readout source mismatch",
            ));
        }
        let bound = completion
            .result
            .as_ref()
            .map_err(|error| PcNnuecReadoutError::Preparation(error.clone()))?;
        if !bound.validates(
            &self.owner.semantic.stores,
            self.root_state_id(),
            self.authority.root.0,
            input.view.seal_digest(),
        ) || !Arc::ptr_eq(bound.view(), &input.view)
        {
            return Err(PcNnuecReadoutError::Invalid(
                "shared readout expired or different View",
            ));
        }
        let readout = bound.shared_readout().ok_or(PcNnuecReadoutError::Invalid(
            "actor did not execute a shared readout",
        ))?;
        let mut requests = state
            .requests
            .lock()
            .map_err(|_| PcNnuecReadoutError::Invalid("shared Request roster poisoned"))?;
        let original = requests
            .iter_mut()
            .find(|row| row.request.same_request(&source.request))
            .ok_or(PcNnuecReadoutError::Invalid(
                "shared Request not issued here",
            ))?;
        if original.consumed {
            return Err(PcNnuecReadoutError::Invalid(
                "duplicate shared readout consumption",
            ));
        }
        original.consumed = true;
        let numeric = &readout.output().numeric;
        Ok(PcNnuecReadoutObservation {
            domain: "rz-rovent-pc-shared-nnuec-observation/1",
            process_epoch: self.authority.epoch.0,
            game_generation: self.authority.game.0,
            root_generation: self.authority.root.0,
            question_sha256: hex(&input.question_digest.0),
            view_sha256: hex(&input.view.seal_digest()),
            input_sha256: readout.input_sha256().into(),
            context_sha256: readout.output().context_sha256.clone(),
            model: model.identity().clone(),
            value: numeric.value,
            wdl: numeric.wdl,
            raw_outputs: numeric.raw_outputs.to_vec(),
            r_search_calls: 0,
            policy_authority: false,
        })
    }
    pub fn shared_nnuec_readout_consumed(
        &self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
    ) -> Result<bool, PcNnuecReadoutError> {
        self.check_input(input)?;
        let Some(state) = &self.shared_nnuec else {
            return Ok(false);
        };
        let requests = state
            .requests
            .lock()
            .map_err(|_| PcNnuecReadoutError::Invalid("shared Request roster poisoned"))?;
        Ok(requests.iter().any(|row| {
            row.consumed
                && row.digest == input.question_digest
                && Weak::ptr_eq(&row.question, &Arc::downgrade(&input.question))
                && Weak::ptr_eq(&row.view, &Arc::downgrade(&input.view))
        }))
    }
    fn shared_origin_matches(&self, source: &ReadoutOrigin) -> bool {
        Weak::ptr_eq(&source.owner, &Arc::downgrade(&self.identity))
            && source.authority == self.authority
            && source.root_state == self.root_state_id()
            && source.root_budget == self.original_root_budget()
            && source.deadline == self.deadline
    }
    #[allow(clippy::result_large_err)]
    pub fn close_shared_nnuec_readouts(
        &self,
        actor: &PreparationActor,
        release: NnuecCacheReleaseAck,
        terminals: Vec<PcNnuecReadoutTerminal>,
    ) -> Result<
        PcSharedReadoutRootClosure,
        (
            PcNnuecReadoutError,
            NnuecCacheReleaseAck,
            Vec<PcNnuecReadoutTerminal>,
        ),
    > {
        let checked = (|| {
            let state = self
                .shared_nnuec
                .as_ref()
                .ok_or(PcNnuecReadoutError::Invalid(
                    "shared readout is not registered",
                ))?;
            let requests = state
                .requests
                .lock()
                .map_err(|_| PcNnuecReadoutError::Invalid("shared Request roster poisoned"))?;
            if !actor.verifies_shared_cache_release(&state.model, &release)
                || terminals.len() != requests.len()
                || terminals.iter().any(|terminal| {
                    !self.shared_origin_matches(&terminal.origin)
                        || !Weak::ptr_eq(&terminal.origin.model, &Arc::downgrade(&state.model))
                        || terminal
                            .terminal
                            .as_ref()
                            .is_some_and(|ack| !ack.matches_cache_release(&release))
                })
                || requests.iter().any(|request| {
                    terminals
                        .iter()
                        .filter(|terminal| terminal.origin.request.same_request(&request.request))
                        .count()
                        != 1
                })
            {
                return Err(PcNnuecReadoutError::Invalid(
                    "original shared Request/cache release roster differs",
                ));
            }
            Ok((
                Arc::downgrade(&state.model),
                requests.iter().filter(|request| request.consumed).count(),
            ))
        })();
        match checked {
            Err(error) => Err((error, release, terminals)),
            Ok((model, semantic_consumed_requests)) => Ok(PcSharedReadoutRootClosure {
                owner: Arc::downgrade(&self.identity),
                model,
                authority: self.authority,
                root_state: self.root_state_id(),
                root_budget: self.original_root_budget(),
                deadline: self.deadline,
                terminals,
                semantic_consumed_requests,
                _release: release,
            }),
        }
    }
    pub fn shared_nnuec_closure_matches(&self, closure: &PcSharedReadoutRootClosure) -> bool {
        let Some(state) = &self.shared_nnuec else {
            return false;
        };
        let Ok(requests) = state.requests.lock() else {
            return false;
        };
        Weak::ptr_eq(&closure.owner, &Arc::downgrade(&self.identity))
            && Weak::ptr_eq(&closure.model, &Arc::downgrade(&state.model))
            && closure.authority == self.authority
            && closure.root_state == self.root_state_id()
            && closure.root_budget == self.original_root_budget()
            && closure.deadline == self.deadline
            && closure.terminals.len() == requests.len()
            && requests.iter().all(|request| {
                closure
                    .terminals
                    .iter()
                    .filter(|terminal| terminal.origin.request.same_request(&request.request))
                    .count()
                    == 1
            })
    }
}
fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_value::nnuec::v2::review::shared_readout_fixture_model;
    use rz_contracts::{
        ClockDomain, GameGeneration, MonotonicTick, OwnerId, ProcessEpoch, RootGeneration,
    };

    type Port = PcSemanticPort<LegalOrderRoleMock>;
    type Input = FrameInput<ReviewView, PcPortQuestion>;

    fn port() -> Port {
        let mut owner = PcSemanticOwner::new_selected(
            PalsSemanticConfig {
                beam_width: 1,
                line_plies: 2,
                // The unchanged semantic owner reserves its minimum root/
                // transition space even when this fixture never invokes R.
                max_nodes: 257,
                max_records: 64,
                max_role_calls: 32,
            },
            LegalOrderRoleMock,
            ReviewSelectionPolicy::default(),
            ReviewIndexLimits::default(),
        )
        .unwrap();
        let authority = SearchAuthority {
            epoch: ProcessEpoch(23),
            game: GameGeneration(0),
            root: RootGeneration(7),
            implementation: ContractDigest([7; 32]),
        };
        let position = Position::startpos();
        let registration = owner.register_adaptive_root(&position, authority).unwrap();
        let next = Arc::new(std::sync::atomic::AtomicU64::new(1));
        let issuer: adaptive_port::AdaptiveRulesOwnerIssuer =
            Arc::new(move || Ok(OwnerId(next.fetch_add(1, Ordering::Relaxed))));
        Port::new_registered_retaining(
            owner,
            position,
            authority,
            Deadline {
                clock: ClockDomain(authority.epoch),
                at: MonotonicTick(20_000_000_000),
            },
            Instant::now() + Duration::from_secs(20),
            ExecutionMode::CpuPc,
            issuer,
            registration,
            None,
        )
        .map_err(|(_, error)| error)
        .unwrap()
    }
    fn completion(actor: &PreparationActor, request: Request) -> Completion {
        actor
            .try_submit(request)
            .unwrap_or_else(|_| panic!("same actor Request"));
        let until = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(completion) = actor.try_complete().unwrap() {
                return completion;
            }
            assert!(Instant::now() < until, "actual preparation terminal");
            std::thread::yield_now();
        }
    }
    fn release(actor: &PreparationActor) -> NnuecCacheReleaseAck {
        let ticket = actor.invalidate_context_cache().unwrap();
        let until = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(ack) = actor.try_context_cache_release(&ticket).unwrap() {
                return ack;
            }
            assert!(Instant::now() < until, "actual cache release ACK");
            std::thread::yield_now();
        }
    }
    fn prepare(
        port: &Port,
        input: &Input,
        model: &Arc<Model>,
        cancel: Arc<AtomicBool>,
    ) -> (Request, PcNnuecReadoutSeal) {
        port.prepare_shared_nnuec_readout(input, model, cancel)
            .unwrap_or_else(|_| panic!("original P/C question Request"))
            .into_parts()
    }

    #[test]
    fn explicit_shared_readout_consumes_only_original_view_and_never_calls_r() {
        let model = shared_readout_fixture_model();
        let mut port = port();
        assert!(!port.requires_shared_nnuec_readout());
        port.register_shared_nnuec_readout(Arc::clone(&model))
            .unwrap();
        let input = port.next_role().unwrap().unwrap();
        let counters = port.counters;
        assert!(!port.shared_nnuec_readout_consumed(&input).unwrap());
        let actor = PreparationActor::spawn(Arc::clone(&model)).unwrap();
        let (request, seal) = prepare(&port, &input, &model, Arc::new(AtomicBool::new(false)));
        let completion = completion(&actor, request);
        let mut foreign_port = self::port();
        foreign_port
            .register_shared_nnuec_readout(Arc::clone(&model))
            .unwrap();
        let foreign_input = foreign_port.next_role().unwrap().unwrap();
        assert!(
            foreign_port
                .consume_shared_nnuec_readout(&foreign_input, &seal, &model, &completion)
                .is_err()
        );
        let mut different_view = input.clone();
        different_view.view = Arc::clone(&foreign_input.view);
        different_view.view_seal = foreign_input.view_seal;
        assert!(
            port.consume_shared_nnuec_readout(&different_view, &seal, &model, &completion)
                .is_err()
        );
        let equal_bytes_foreign_model = shared_readout_fixture_model();
        assert_eq!(model.identity(), equal_bytes_foreign_model.identity());
        assert!(
            port.consume_shared_nnuec_readout(
                &input,
                &seal,
                &equal_bytes_foreign_model,
                &completion
            )
            .is_err()
        );
        let observation = port
            .consume_shared_nnuec_readout(&input, &seal, &model, &completion)
            .unwrap();
        assert_eq!(observation.r_search_calls, 0);
        assert!(!observation.policy_authority);
        assert!(observation.value.is_finite());
        assert_eq!(port.counters, counters);
        assert_eq!(std::mem::size_of_val(port.researcher()), 0);
        assert!(port.shared_nnuec_readout_consumed(&input).unwrap());
        assert!(
            port.consume_shared_nnuec_readout(&input, &seal, &model, &completion)
                .is_err()
        );
        let terminal = seal
            .retire(completion)
            .unwrap_or_else(|_| panic!("original terminal custody"));
        assert!(terminal.submitted());
        let closure = port
            .close_shared_nnuec_readouts(&actor, release(&actor), vec![terminal])
            .unwrap_or_else(|_| panic!("original root/cache closure"));
        assert_eq!(closure.issued_requests(), 1);
        assert_eq!(closure.submitted_requests(), 1);
        assert_eq!(
            closure.terminal_status_count(PreparationStatus::Completed),
            1
        );
        assert_eq!(
            closure.terminal_status_count(PreparationStatus::Cancelled),
            0
        );
        assert_eq!(closure.terminal_status_count(PreparationStatus::Failed), 0);
        assert_eq!(closure.terminal_status_count(PreparationStatus::Expired), 0);
        assert_eq!(closure.semantic_consumed_requests(), 1);
        assert!(port.shared_nnuec_closure_matches(&closure));
        assert!(!foreign_port.shared_nnuec_closure_matches(&closure));
        actor.shutdown().unwrap();
    }

    #[test]
    fn late_cancelled_readout_cannot_be_consumed_but_original_terminal_closes() {
        let model = shared_readout_fixture_model();
        let mut port = port();
        port.register_shared_nnuec_readout(Arc::clone(&model))
            .unwrap();
        let input = port.next_role().unwrap().unwrap();
        let actor = PreparationActor::spawn(Arc::clone(&model)).unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let (request, seal) = prepare(&port, &input, &model, Arc::clone(&cancel));
        let completion = completion(&actor, request);
        cancel.store(true, Ordering::Release);
        assert!(
            port.consume_shared_nnuec_readout(&input, &seal, &model, &completion)
                .is_err()
        );
        assert!(!port.shared_nnuec_readout_consumed(&input).unwrap());
        port.revoke();
        let terminal = seal
            .retire(completion)
            .unwrap_or_else(|_| panic!("late original terminal still retires"));
        let closure = port
            .close_shared_nnuec_readouts(&actor, release(&actor), vec![terminal])
            .unwrap_or_else(|_| panic!("cancelled root physical closure"));
        assert!(port.shared_nnuec_closure_matches(&closure));
        // Physical completion before a late cancellation remains Completed;
        // the failed semantic consumption must not become a consumed readout.
        assert_eq!(closure.submitted_requests(), 1);
        assert_eq!(
            closure.terminal_status_count(PreparationStatus::Completed),
            1
        );
        assert_eq!(
            closure.terminal_status_count(PreparationStatus::Cancelled),
            0
        );
        assert_eq!(closure.semantic_consumed_requests(), 0);
        assert_eq!(port.counters.consumed_role_outputs, 0);
        assert_eq!(port.counters.completed_nn_inputs, Some(0));

        // A different original Request cancelled before actor execution has
        // an actual Cancelled terminal, rather than the late-cancelled status
        // above. The same actor is reused after its previous cache ACK.
        let mut cancelled_port = self::port();
        cancelled_port
            .register_shared_nnuec_readout(Arc::clone(&model))
            .unwrap();
        let cancelled_input = cancelled_port.next_role().unwrap().unwrap();
        let cancelled_control = Arc::new(AtomicBool::new(false));
        let (cancelled_request, cancelled_seal) = prepare(
            &cancelled_port,
            &cancelled_input,
            &model,
            Arc::clone(&cancelled_control),
        );
        cancelled_control.store(true, Ordering::Release);
        let cancelled_completion = self::completion(&actor, cancelled_request);
        let cancelled_terminal = cancelled_seal
            .retire(cancelled_completion)
            .unwrap_or_else(|_| panic!("original cancelled actor terminal"));
        let cancelled_closure = cancelled_port
            .close_shared_nnuec_readouts(&actor, release(&actor), vec![cancelled_terminal])
            .unwrap_or_else(|_| panic!("original cancelled actor closure"));
        assert_eq!(cancelled_closure.submitted_requests(), 1);
        assert_eq!(
            cancelled_closure.terminal_status_count(PreparationStatus::Completed),
            0
        );
        assert_eq!(
            cancelled_closure.terminal_status_count(PreparationStatus::Cancelled),
            1
        );
        assert_eq!(
            cancelled_closure.terminal_status_count(PreparationStatus::Failed),
            0
        );
        assert_eq!(
            cancelled_closure.terminal_status_count(PreparationStatus::Expired),
            0
        );
        assert_eq!(cancelled_closure.semantic_consumed_requests(), 0);
        actor.shutdown().unwrap();
    }

    #[test]
    fn unsubmitted_cancel_retains_original_request_and_foreign_actor_cannot_close() {
        let model = shared_readout_fixture_model();
        let mut port = port();
        port.register_shared_nnuec_readout(Arc::clone(&model))
            .unwrap();
        let input = port.next_role().unwrap().unwrap();
        let actor = PreparationActor::spawn(Arc::clone(&model)).unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let (request, seal) = prepare(&port, &input, &model, Arc::clone(&cancel));
        let mut other_port = self::port();
        other_port
            .register_shared_nnuec_readout(Arc::clone(&model))
            .unwrap();
        let other_input = other_port.next_role().unwrap().unwrap();
        let (other_request, other_seal) = prepare(
            &other_port,
            &other_input,
            &model,
            Arc::new(AtomicBool::new(false)),
        );
        let (_, seal, other_request) = seal.retire_unsubmitted(other_request).err().unwrap();
        let _other_terminal = other_seal
            .retire_unsubmitted(other_request)
            .unwrap_or_else(|_| panic!("foreign Request preserved"));
        cancel.store(true, Ordering::Release);
        let terminal = seal
            .retire_unsubmitted(request)
            .unwrap_or_else(|_| panic!("original never-submitted Request"));
        assert!(!terminal.submitted());
        assert!(terminal.observation().is_none());
        let other_model = shared_readout_fixture_model();
        let foreign_actor = PreparationActor::spawn(other_model).unwrap();
        let (_, foreign_release, terminals) = port
            .close_shared_nnuec_readouts(&foreign_actor, release(&foreign_actor), vec![terminal])
            .err()
            .unwrap();
        drop(foreign_release);
        let closure = port
            .close_shared_nnuec_readouts(&actor, release(&actor), terminals)
            .unwrap_or_else(|_| panic!("original actor close without fabricated terminal"));
        assert_eq!(closure.submitted_requests(), 0);
        assert_eq!(closure.issued_requests(), 1);
        for status in [
            PreparationStatus::Completed,
            PreparationStatus::Cancelled,
            PreparationStatus::Failed,
            PreparationStatus::Expired,
        ] {
            assert_eq!(closure.terminal_status_count(status), 0);
        }
        assert_eq!(closure.semantic_consumed_requests(), 0);
        assert!(port.shared_nnuec_closure_matches(&closure));
        actor.shutdown().unwrap();
        foreign_actor.shutdown().unwrap();
    }
}
