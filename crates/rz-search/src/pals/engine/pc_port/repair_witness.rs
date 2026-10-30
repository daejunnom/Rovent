//! 실제 P/C 수락 이후에만 발급되는 조건부 Repair 증거다.
//!
//! 공개 trace, 수치 WDL, FreshPairSeal을 이 타입으로 역변환하는 입구는 없다.
//! 원 selected view와 latent는 보관하지 않으며, 작은 수열과 원 수락 출처만
//! 보존한다. 같은 frozen 모델의 비교는 Rules 승패나 전체 방어 증명이 아니다.
use super::*;
use crate::pals::coordinator::FreshPairSeal;
use crate::pals::value::{compare_model_wdl, flip_model_wdl};

const MAX_WITNESS_PLIES: usize = 256;

/// 원 counter에 대한 동일 frozen 모델의 조건부 의견이다.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PcDivergenceResolution {
    Supported,
    Refuted,
    Unknown,
    Disputed,
}

/// 제출당시 C 입력에 실제 존재했던 counter와 실제 수락을 묶는다.
/// private 생성자는 원 Store/view 검사를 통과한 수락만 받는다.
pub struct PcDivergenceSource {
    accepted: AcceptedRoleOrigin,
    source: Arc<PcQuestionSource>,
    counter_local_index: u16,
    counter_revision: u64,
    divergence_index: u16,
    divergence_ply: usize,
    first: BoardMove,
    proposal: Vec<BoardMove>,
    counter: Vec<BoardMove>,
    perspective: Color,
    attack_ranked_source: Option<Arc<PcAttackRankedSource>>,
    visible_source_digest: ContractDigest,
    digest: ContractDigest,
}

/// 원 P Repair 또는 C 재검토 tail의 실제 Store publication이다.
pub struct PcRepairAcceptedLine {
    accepted: AcceptedRoleOrigin,
    source: Arc<PcQuestionSource>,
    line: LineId,
    observation: ObservationId,
    revision: u64,
    kind: RecordKind,
    moves: Vec<BoardMove>,
    digest: ContractDigest,
}

/// 원 native/registry에서 이미 수락한 Fresh 결과의 불변 관측이다.
/// Clone은 과거 증거의 보존만 하며 NN 실행·수락·소비 credit을 발급하지 않는다.
#[derive(Clone)]
pub struct PcAcceptedFreshEndpoint {
    accepted: AcceptedRoleOrigin,
    source: Arc<PcQuestionSource>,
    line: Vec<BoardMove>,
    pair: FreshPairSeal,
    model: ModelValueIdentity,
    perspective: Color,
    raw_wdl: [f32; 3],
    root_wdl: [f32; 3],
    physical: AdaptivePortRoleEvidence,
    digest: ContractDigest,
}

/// 실제 source→Repair→C→frozen pair의 연결이다. 미완료 분기는 Unknown이며
/// source가 없던 과거 입력에 미래 counter를 삽입하지 않는다.
pub struct PcDivergenceWitness {
    source: Arc<PcDivergenceSource>,
    repair: Option<Arc<PcRepairAcceptedLine>>,
    recheck: Option<Arc<PcRepairAcceptedLine>>,
    repaired_endpoint: Option<PcAcceptedFreshEndpoint>,
    counter_endpoint: Option<PcAcceptedFreshEndpoint>,
    resolution: PcDivergenceResolution,
    event_sequence: u64,
    reason: &'static str,
    digest: ContractDigest,
}

fn invalid() -> PalsError {
    RoleError::InvalidOutput.into()
}

pub(super) fn accepted_origin(
    input: &FrameInput<ReviewView, PcPortQuestion>,
    accepted: &AcceptedRoleStep,
    output: &RoleOutput,
    stores: &PalsStores,
) -> Result<AcceptedRoleOrigin, PalsError> {
    PcPortQuestion::validate_input_source(input, stores)?;
    let question = &input.question;
    let ticket = accepted.ticket();
    if question.frame == 0
        || Instant::now() >= question.deadline
        || ticket.authority != question.source.authority
        || ticket.question != input.question_digest
        || ticket.view != input.view_seal
        || ticket.original_deadline != question.source.original_deadline
        || ticket.role != role_for_purpose(question.context.purpose)?
        || ticket.representation.role != ticket.role
        || ticket.representation.record_revision != question.context.public_revision
        || !accepted.validates_output(output)
    {
        return Err(invalid());
    }
    accepted
        .origin(question.frontier_frame(), question.reciprocal_question())
        .map_err(|_| invalid())
}

fn same_owner(left: &PcQuestionSource, right: &PcQuestionSource) -> bool {
    Arc::ptr_eq(&left.owner, &right.owner)
        && left.authority == right.authority
        && left.original_deadline == right.original_deadline
        && left.root_budget.is_some()
        && left.root_budget == right.root_budget
        && left.mode == right.mode
        && left.view.root_state == right.view.root_state
}

fn same_model(left: &RoleTicketEnvelope, right: &RoleTicketEnvelope) -> bool {
    left.representation.model == right.representation.model
        && left.representation.encoding == right.representation.encoding
        && left.representation.precision == right.representation.precision
        && left.representation.frozen_epoch == right.representation.frozen_epoch
}

fn hash_line(hash: &mut Sha256, line: &[BoardMove]) -> Result<(), PalsError> {
    if line.len() > MAX_WITNESS_PLIES {
        return Err(PalsError::Capacity);
    }
    hash.update((line.len() as u64).to_le_bytes());
    for movement in line {
        hash.update(Move16::pack(*movement)?.bits().to_le_bytes());
    }
    Ok(())
}

fn inline_bytes<T>() -> Option<u64> {
    u64::try_from(std::mem::size_of::<T>()).ok()
}
fn move_capacity(line: &Vec<BoardMove>) -> Option<u64> {
    u64::try_from(
        line.capacity()
            .checked_mul(std::mem::size_of::<BoardMove>())?,
    )
    .ok()
}

impl PcDivergenceSource {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn capture(
        input: &FrameInput<ReviewView, PcPortQuestion>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        counter_revision: u64,
        counter_local_index: u16,
        first: BoardMove,
        stores: &PalsStores,
    ) -> Result<Self, PalsError> {
        let origin = accepted_origin(input, accepted, output, stores)?;
        let query = input.view.query();
        let index = usize::from(counter_local_index);
        let record = input.view.records().get(index).ok_or_else(invalid)?;
        let metadata = input
            .view
            .records_metadata()
            .get(index)
            .ok_or_else(invalid)?;
        let anchor = input
            .view
            .anchors()
            .get(usize::from(metadata.anchor))
            .ok_or_else(invalid)?;
        let counter = query.counterexample.as_ref().ok_or_else(invalid)?;
        let RolePayload::Counterexample {
            divergence_logits, ..
        } = &output.payload
        else {
            return Err(invalid());
        };
        if query.purpose != RoleQueryPurpose::DivergencePolicy
            || !query.prefix.is_empty()
            || record.kind != RecordKind::Counterexample
            || counter_revision == 0
            || metadata.source_revision != counter_revision
            || record.revision != counter_revision
            || metadata.source_line_offset != 0
            || metadata.original_first_move != Some(first)
            || record.line != *counter
            || !anchor
                .snapshot
                .same_state(&input.question.position.snapshot())
            || query.proposal.first() != Some(&first)
            || counter.first() != Some(&first)
            || counter.len() > MAX_WITNESS_PLIES
            || query.proposal.len() > MAX_WITNESS_PLIES
            || divergence_logits.len() != query.divergences.len()
            || divergence_logits.is_empty()
            || divergence_logits.iter().any(|value| !value.is_finite())
        {
            return Err(invalid());
        }
        let divergence_index = divergence_logits
            .iter()
            .enumerate()
            .max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a)))
            .map(|(index, _)| index)
            .ok_or_else(invalid)?;
        let divergence_ply = *query
            .divergences
            .get(divergence_index)
            .ok_or_else(invalid)?;
        if divergence_ply == 0
            || divergence_ply >= counter.len()
            || divergence_ply >= query.proposal.len()
        {
            return Err(invalid());
        }
        let divergence_index = u16::try_from(divergence_index).map_err(|_| PalsError::Capacity)?;
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-pc-divergence-source/1\0");
        hash.update(origin.seal_digest().0);
        hash.update(input.question.source.root_budget.ok_or_else(invalid)?.0);
        hash.update(counter_local_index.to_le_bytes());
        hash.update(counter_revision.to_le_bytes());
        hash.update(divergence_index.to_le_bytes());
        hash.update((divergence_ply as u64).to_le_bytes());
        hash_line(&mut hash, &query.proposal)?;
        hash_line(&mut hash, counter)?;
        let digest = ContractDigest(hash.finalize().into());
        Ok(Self {
            accepted: origin,
            source: Arc::clone(&input.question.source),
            counter_local_index,
            counter_revision,
            divergence_index,
            divergence_ply,
            first,
            proposal: query.proposal.clone(),
            counter: counter.clone(),
            perspective: input.question.position.side_to_move(),
            attack_ranked_source: None,
            visible_source_digest: digest,
            digest,
        })
    }
    /// 후속 Counter의 원 증거와, Counter가 아직 없었던 실제 입력을 연결한다.
    /// 원 visible source seal은 보존하고 연결된 증거에만 별도 domain을 쓴다.
    pub(super) fn with_attack_ranked_source(
        mut self,
        attack: Option<Arc<PcAttackRankedSource>>,
    ) -> Result<Self, PalsError> {
        if self.attack_ranked_source.is_some() {
            return Err(invalid());
        }
        if let Some(attack) = attack {
            let original = attack.accepted().ticket();
            let visible = self.accepted.ticket();
            if !same_owner(attack.source(), &self.source)
                || !same_model(original, visible)
                || original.role != Role::Critic
                || original.id.epoch != visible.id.epoch
                || original.id.sequence >= visible.id.sequence
                || attack.source().sequence >= self.source.sequence
                || !attack.source().position.same_state(&self.source.position)
                || attack.root_budget() != self.source.root_budget.ok_or_else(invalid)?
                || attack.divergence_ply() != self.divergence_ply
                || attack.original_first_move() != self.first
                || attack.proposal() != self.proposal
                || attack.perspective() != self.perspective
            {
                return Err(invalid());
            }
            let mut hash = Sha256::new();
            hash.update(b"rz-rovent-pc-divergence-source-with-attack/1\0");
            hash.update(self.visible_source_digest.0);
            hash.update(attack.digest().0);
            hash.update(self.source.sequence.to_le_bytes());
            hash.update(attack.source_sequence().to_le_bytes());
            self.digest = ContractDigest(hash.finalize().into());
            self.attack_ranked_source = Some(attack);
        }
        Ok(self)
    }
    pub fn attack_ranked_source(&self) -> Option<&PcAttackRankedSource> {
        self.attack_ranked_source.as_deref()
    }
    /// The original visible-counter source seal, without an optional causal link.
    pub fn visible_source_digest(&self) -> ContractDigest {
        self.visible_source_digest
    }
    /// 질문 owner가 실제 발급한 순번이다. frame/RequestId/collector 시계와 다르다.
    pub fn source_sequence(&self) -> u64 {
        self.source.sequence
    }
    pub fn accepted(&self) -> &AcceptedRoleOrigin {
        &self.accepted
    }
    pub fn counter_local_index(&self) -> u16 {
        self.counter_local_index
    }
    pub fn counter_revision(&self) -> u64 {
        self.counter_revision
    }
    pub fn divergence_index(&self) -> u16 {
        self.divergence_index
    }
    pub fn divergence_ply(&self) -> usize {
        self.divergence_ply
    }
    pub fn first_move(&self) -> BoardMove {
        self.first
    }
    pub fn proposal(&self) -> &[BoardMove] {
        &self.proposal
    }
    pub fn counter(&self) -> &[BoardMove] {
        &self.counter
    }
    pub fn perspective(&self) -> Color {
        self.perspective
    }
    pub fn digest(&self) -> ContractDigest {
        self.digest
    }
    pub fn owned_host_bytes(&self) -> Option<u64> {
        let bytes = inline_bytes::<Self>()?
            .checked_add(move_capacity(&self.proposal)?)?
            .checked_add(move_capacity(&self.counter)?)?
            .checked_add(self.source.owned_host_bytes()?)?;
        match &self.attack_ranked_source {
            Some(attack) => bytes.checked_add(attack.owned_host_bytes()?),
            None => Some(bytes),
        }
    }
}

impl PcRepairAcceptedLine {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn capture(
        input: &FrameInput<ReviewView, PcPortQuestion>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        line: LineId,
        observation: ObservationId,
        revision: u64,
        kind: RecordKind,
        stores: &PalsStores,
    ) -> Result<Self, PalsError> {
        let origin = accepted_origin(input, accepted, output, stores)?;
        let purpose = input.question.context.purpose;
        let expected_kind = match purpose {
            RoleQueryPurpose::RepairPolicy => (RecordKind::Repair, ObservationKind::Repair),
            RoleQueryPurpose::ReplyPolicy => {
                (RecordKind::Counterexample, ObservationKind::Refutation)
            }
            _ => return Err(invalid()),
        };
        let raw_line = stores.lines.get(line)?;
        let raw_observation = stores.observations.get(observation)?;
        let moves = stores.lines.moves(line)?;
        let prefix = &input.question.context.prefix;
        let last = moves.last().copied().ok_or_else(invalid)?;
        if kind != expected_kind.0
            || raw_observation.kind != expected_kind.1
            || raw_observation.state != input.view.root_state()
            || raw_observation.line != Some(line)
            || raw_line.start_state != input.view.root_state()
            || revision <= input.question.context.public_revision
            || moves.len() != prefix.len().checked_add(1).ok_or(PalsError::Capacity)?
            || !moves.starts_with(prefix)
            || !input.view.query().legal.contains(&last)
            || moves.len() > MAX_WITNESS_PLIES
        {
            return Err(invalid());
        }
        PcPortQuestion::validate_accepted_frontier_child(input, accepted, output, last, stores)?;
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-pc-accepted-repair-line/1\0");
        hash.update(origin.seal_digest().0);
        hash.update((line.0 as u64).to_le_bytes());
        hash.update((observation.0 as u64).to_le_bytes());
        hash.update(revision.to_le_bytes());
        hash.update([kind as u8]);
        hash_line(&mut hash, &moves)?;
        Ok(Self {
            accepted: origin,
            source: Arc::clone(&input.question.source),
            line,
            observation,
            revision,
            kind,
            moves,
            digest: ContractDigest(hash.finalize().into()),
        })
    }
    pub fn accepted(&self) -> &AcceptedRoleOrigin {
        &self.accepted
    }
    pub fn line(&self) -> LineId {
        self.line
    }
    pub fn observation(&self) -> ObservationId {
        self.observation
    }
    pub fn revision(&self) -> u64 {
        self.revision
    }
    pub fn kind(&self) -> RecordKind {
        self.kind
    }
    pub fn moves(&self) -> &[BoardMove] {
        &self.moves
    }
    pub fn digest(&self) -> ContractDigest {
        self.digest
    }
    pub fn owned_host_bytes(&self) -> Option<u64> {
        inline_bytes::<Self>()?
            .checked_add(move_capacity(&self.moves)?)?
            .checked_add(self.source.owned_host_bytes()?)
    }
}

impl PcAcceptedFreshEndpoint {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn capture(
        input: &FrameInput<ReviewView, PcPortQuestion>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: AdaptivePortRoleEvidence,
        model: &ModelValueIdentity,
        perspective: Color,
        stores: &PalsStores,
    ) -> Result<Self, PalsError> {
        let origin = accepted_origin(input, accepted, output, stores)?;
        let RolePayload::Proposal { wdl, .. } = &output.payload else {
            return Err(invalid());
        };
        let execution = physical.execution.ok_or_else(invalid)?;
        if input.question.context.purpose != RoleQueryPurpose::ValueFresh
            || accepted.ticket().role != Role::Proposer
            || physical.physical_fresh != Some(true)
            || !physical.measurement_complete
            || physical.completed_nn_inputs.is_none_or(|count| count == 0)
            || execution.epoch != accepted.ticket().authority.epoch
            || input.view.query().prefix.is_empty()
            || input.view.query().prefix.len() > MAX_WITNESS_PLIES
        {
            return Err(invalid());
        }
        let raw_wdl = wdl.probabilities();
        let value = ModelValueOutput {
            identity: model.clone(),
            input_sha256: output.key.input.0,
            state: input.question.position.position_identity(),
            perspective: input.question.position.side_to_move(),
            wdl: raw_wdl,
        };
        value.validate(&input.question.position, model)?;
        let root_wdl = if value.perspective == perspective {
            raw_wdl
        } else {
            flip_model_wdl(raw_wdl)?
        };
        let pair =
            FreshPairSeal::from_ticket(accepted.ticket(), input.public_context, perspective.into());
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-pc-accepted-fresh-endpoint/1\0");
        hash.update(origin.seal_digest().0);
        hash.update(pair.public_context.0);
        hash.update(execution.epoch.0.to_le_bytes());
        hash.update(execution.sequence.to_le_bytes());
        hash.update(
            physical
                .completed_nn_inputs
                .ok_or_else(invalid)?
                .to_le_bytes(),
        );
        hash.update([perspective as u8]);
        for value in raw_wdl {
            hash.update(value.to_bits().to_le_bytes());
        }
        hash_line(&mut hash, &input.view.query().prefix)?;
        Ok(Self {
            accepted: origin,
            source: Arc::clone(&input.question.source),
            line: input.view.query().prefix.clone(),
            pair,
            model: model.clone(),
            perspective,
            raw_wdl,
            root_wdl,
            physical,
            digest: ContractDigest(hash.finalize().into()),
        })
    }
    pub fn accepted(&self) -> &AcceptedRoleOrigin {
        &self.accepted
    }
    pub fn line(&self) -> &[BoardMove] {
        &self.line
    }
    pub fn pair(&self) -> &FreshPairSeal {
        &self.pair
    }
    pub fn model(&self) -> &ModelValueIdentity {
        &self.model
    }
    pub fn perspective(&self) -> Color {
        self.perspective
    }
    pub fn raw_wdl(&self) -> [f32; 3] {
        self.raw_wdl
    }
    pub fn root_wdl(&self) -> [f32; 3] {
        self.root_wdl
    }
    pub fn physical(&self) -> AdaptivePortRoleEvidence {
        self.physical
    }
    pub fn digest(&self) -> ContractDigest {
        self.digest
    }
    pub fn owned_host_bytes(&self) -> Option<u64> {
        let mut bytes = inline_bytes::<Self>()?
            .checked_add(move_capacity(&self.line)?)?
            .checked_add(self.source.owned_host_bytes()?)?;
        for text in [
            &self.model.semantics,
            &self.model.model,
            &self.model.encoding,
            &self.model.precision,
        ] {
            bytes = bytes.checked_add(u64::try_from(text.capacity()).ok()?)?;
        }
        Some(bytes)
    }
}

impl PcDivergenceWitness {
    /// 실제로 이미 존재한 부분 증거를 보존한다. 미완료 사실은 known label이나
    /// 새로운 acceptance가 되지 않으며, 없는 endpoint를 합성하지 않는다.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn unresolved(
        source: Arc<PcDivergenceSource>,
        repair: Option<Arc<PcRepairAcceptedLine>>,
        recheck: Option<Arc<PcRepairAcceptedLine>>,
        repaired_endpoint: Option<PcAcceptedFreshEndpoint>,
        counter_endpoint: Option<PcAcceptedFreshEndpoint>,
        event_sequence: u64,
        reason: &'static str,
    ) -> Self {
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-pc-divergence-unresolved/1\0");
        hash.update(source.digest.0);
        for digest in [
            repair.as_ref().map(|v| v.digest),
            recheck.as_ref().map(|v| v.digest),
            repaired_endpoint.as_ref().map(|v| v.digest),
            counter_endpoint.as_ref().map(|v| v.digest),
        ] {
            hash.update([u8::from(digest.is_some())]);
            if let Some(digest) = digest {
                hash.update(digest.0);
            }
        }
        hash.update(event_sequence.to_le_bytes());
        hash.update((reason.len() as u64).to_le_bytes());
        hash.update(reason.as_bytes());
        Self {
            source,
            repair,
            recheck,
            repaired_endpoint,
            counter_endpoint,
            resolution: PcDivergenceResolution::Unknown,
            event_sequence,
            reason,
            digest: ContractDigest(hash.finalize().into()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_frozen_pair(
        source: Arc<PcDivergenceSource>,
        repair: Arc<PcRepairAcceptedLine>,
        recheck: Arc<PcRepairAcceptedLine>,
        repaired_endpoint: PcAcceptedFreshEndpoint,
        counter_endpoint: PcAcceptedFreshEndpoint,
        event_sequence: u64,
    ) -> Result<Self, PalsError> {
        let ticket = source.accepted.ticket();
        let repair_ticket = repair.accepted.ticket();
        let recheck_ticket = recheck.accepted.ticket();
        repaired_endpoint
            .pair
            .validate_pair(&counter_endpoint.pair)
            .map_err(|_| invalid())?;
        let attack = source.divergence_ply;
        let fixed = attack.checked_add(1).ok_or(PalsError::Capacity)?;
        if event_sequence == 0
            || repair.kind != RecordKind::Repair
            || recheck.kind != RecordKind::Counterexample
            || !same_owner(&source.source, &repair.source)
            || !same_owner(&source.source, &recheck.source)
            || !same_owner(&source.source, &repaired_endpoint.source)
            || !same_owner(&source.source, &counter_endpoint.source)
            || !same_model(ticket, repair_ticket)
            || !same_model(ticket, recheck_ticket)
            || !same_model(ticket, repaired_endpoint.accepted.ticket())
            || ticket.id.sequence >= repair_ticket.id.sequence
            || repair_ticket.id.sequence >= recheck_ticket.id.sequence
            || recheck_ticket.id.sequence >= repaired_endpoint.accepted.ticket().id.sequence
            || recheck_ticket.id.sequence >= counter_endpoint.accepted.ticket().id.sequence
            || repaired_endpoint.accepted.ticket().id == counter_endpoint.accepted.ticket().id
            || repaired_endpoint.model != counter_endpoint.model
            || repaired_endpoint.perspective != source.perspective
            || counter_endpoint.perspective != source.perspective
            || repaired_endpoint.line != repair.moves
            || counter_endpoint.line != recheck.moves
            || repair.moves.len() != recheck.moves.len()
            || repair.moves.first() != Some(&source.first)
            || recheck.moves.first() != Some(&source.first)
            || !repair
                .moves
                .starts_with(source.counter.get(..fixed).ok_or_else(invalid)?)
        {
            return Err(invalid());
        }
        // 실제 cycle은 원 공격 뒤 P 응답을 바꾸고, 그 이후 상대 차례에서
        // 독립 C 응답을 수락해야 한다. 결과를 보며 임의 suffix를 붙일 수 없다.
        let changed = repair
            .moves
            .iter()
            .zip(&source.counter)
            .position(|(a, b)| a != b)
            .ok_or_else(invalid)?;
        let c_changed = recheck
            .moves
            .iter()
            .zip(&repair.moves)
            .position(|(a, b)| a != b)
            .ok_or_else(invalid)?;
        if changed < fixed || changed % 2 != 0 || c_changed <= changed || c_changed % 2 != 1 {
            return Err(invalid());
        }
        let endpoint_value = |endpoint: &PcAcceptedFreshEndpoint| ModelValueOutput {
            identity: endpoint.model.clone(),
            input_sha256: endpoint.accepted.ticket().representation.input.0,
            state: endpoint.source.position.position_identity(),
            perspective: source.perspective,
            wdl: endpoint.root_wdl,
        };
        let resolution = match compare_model_wdl(
            &repaired_endpoint.model,
            &endpoint_value(&repaired_endpoint),
            &endpoint_value(&counter_endpoint),
        )? {
            std::cmp::Ordering::Greater => PcDivergenceResolution::Refuted,
            std::cmp::Ordering::Less => PcDivergenceResolution::Supported,
            std::cmp::Ordering::Equal => PcDivergenceResolution::Unknown,
        };
        let reason = if resolution == PcDivergenceResolution::Unknown {
            "equal frozen conditional expectations"
        } else {
            "actual Repair and C continuation; frozen model conditional comparison"
        };
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-pc-divergence-witness/1\0");
        for digest in [
            source.digest,
            repair.digest,
            recheck.digest,
            repaired_endpoint.digest,
            counter_endpoint.digest,
        ] {
            hash.update(digest.0);
        }
        hash.update(event_sequence.to_le_bytes());
        hash.update([resolution as u8]);
        Ok(Self {
            source,
            repair: Some(repair),
            recheck: Some(recheck),
            repaired_endpoint: Some(repaired_endpoint),
            counter_endpoint: Some(counter_endpoint),
            resolution,
            event_sequence,
            reason,
            digest: ContractDigest(hash.finalize().into()),
        })
    }
    pub fn source(&self) -> &PcDivergenceSource {
        &self.source
    }
    pub fn repair(&self) -> Option<&PcRepairAcceptedLine> {
        self.repair.as_deref()
    }
    pub fn recheck(&self) -> Option<&PcRepairAcceptedLine> {
        self.recheck.as_deref()
    }
    pub fn repaired_endpoint(&self) -> Option<&PcAcceptedFreshEndpoint> {
        self.repaired_endpoint.as_ref()
    }
    pub fn counter_endpoint(&self) -> Option<&PcAcceptedFreshEndpoint> {
        self.counter_endpoint.as_ref()
    }
    pub fn resolution(&self) -> PcDivergenceResolution {
        self.resolution
    }
    pub fn event_sequence(&self) -> u64 {
        self.event_sequence
    }
    pub fn reason(&self) -> &'static str {
        self.reason
    }
    pub fn digest(&self) -> ContractDigest {
        self.digest
    }
    /// known도 같은 frozen 모델의 의견이며 실제 WDL/Rules 승패 정답은 아니다.
    pub fn target_known(&self) -> bool {
        matches!(
            self.resolution,
            PcDivergenceResolution::Supported | PcDivergenceResolution::Refuted
        )
    }
    /// 공유 Arc의 원 allocation은 보수적으로 중복 계수한다. 원 view나 latent를
    /// 보유하지 않으며 이 값은 실제 allocator peak 관측을 대신하지 않는다.
    pub fn owned_host_bytes(&self) -> Option<u64> {
        let mut bytes = inline_bytes::<Self>()?.checked_add(self.source.owned_host_bytes()?)?;
        for fact in [&self.repair, &self.recheck].into_iter().flatten() {
            bytes = bytes.checked_add(fact.owned_host_bytes()?)?;
        }
        for endpoint in [&self.repaired_endpoint, &self.counter_endpoint]
            .into_iter()
            .flatten()
        {
            bytes = bytes.checked_add(endpoint.owned_host_bytes()?)?;
        }
        Some(bytes)
    }
}
