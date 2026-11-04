//! Historical origin of an actually accepted initial C attack ranking.
//! The input had no counter. A later visible counter is never inserted into it.
use super::*;

pub struct PcAttackRankedSource {
    accepted: AcceptedRoleOrigin,
    source: Arc<PcQuestionSource>,
    divergence_index: u16,
    divergence_ply: usize,
    first: BoardMove,
    proposal: Vec<BoardMove>,
    perspective: Color,
    root_budget: ContractDigest,
    digest: ContractDigest,
}
impl PcAttackRankedSource {
    pub(super) fn capture(
        input: &FrameInput<ReviewView, PcPortQuestion>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        selected_slot: usize,
        stores: &PalsStores,
    ) -> Result<Self, PalsError> {
        let origin = super::repair_witness::accepted_origin(input, accepted, output, stores)?;
        let query = input.view.query();
        let RolePayload::Counterexample {
            divergence_logits, ..
        } = &output.payload
        else {
            return Err(RoleError::InvalidOutput.into());
        };
        let selected = divergence_logits
            .iter()
            .enumerate()
            .max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a)))
            .map(|(index, _)| index)
            .ok_or(RoleError::InvalidOutput)?;
        let first = input.question.first.ok_or(RoleError::InvalidOutput)?;
        let root_budget = input
            .question
            .source
            .root_budget
            .ok_or(RoleError::InvalidOutput)?;
        let ply = *query
            .divergences
            .get(selected_slot)
            .ok_or(RoleError::InvalidOutput)?;
        if query.purpose != RoleQueryPurpose::DivergencePolicy
            || !query.prefix.is_empty()
            || query.counterexample.is_some()
            || query.proposal.first() != Some(&first)
            || query.proposal.len() > 256
            || divergence_logits.len() != query.divergences.len()
            || divergence_logits.iter().any(|value| !value.is_finite())
            || selected_slot != selected
            || ply == 0
            || ply >= query.proposal.len()
            || !input
                .question
                .repair
                .as_ref()
                .is_some_and(|step| step.is_initial_attack_ranking())
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let divergence_index = u16::try_from(selected_slot).map_err(|_| PalsError::Capacity)?;
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-pc-attack-ranked-source/1\0");
        hash.update(origin.seal_digest().0);
        hash.update(root_budget.0);
        hash.update(divergence_index.to_le_bytes());
        hash.update((ply as u64).to_le_bytes());
        hash.update(Move16::pack(first)?.bits().to_le_bytes());
        hash.update((query.proposal.len() as u64).to_le_bytes());
        for movement in &query.proposal {
            hash.update(Move16::pack(*movement)?.bits().to_le_bytes());
        }
        Ok(Self {
            accepted: origin,
            source: Arc::clone(&input.question.source),
            divergence_index,
            divergence_ply: ply,
            first,
            proposal: query.proposal.clone(),
            perspective: input.question.position.side_to_move(),
            root_budget,
            digest: ContractDigest(hash.finalize().into()),
        })
    }
    pub fn accepted(&self) -> &AcceptedRoleOrigin {
        &self.accepted
    }
    pub fn question_digest(&self) -> ContractDigest {
        self.accepted.ticket().question
    }
    pub fn view_seal(&self) -> ReviewViewSeal {
        self.accepted.ticket().view
    }
    /// Port question issuance order; different from branch FrameId and from
    /// native request or collector event sequences.
    pub fn source_sequence(&self) -> u64 {
        self.source.sequence
    }
    pub fn divergence_index(&self) -> u16 {
        self.divergence_index
    }
    pub fn divergence_ply(&self) -> usize {
        self.divergence_ply
    }
    pub fn original_first_move(&self) -> BoardMove {
        self.first
    }
    pub fn proposal(&self) -> &[BoardMove] {
        &self.proposal
    }
    pub fn perspective(&self) -> Color {
        self.perspective
    }
    pub fn root_budget(&self) -> ContractDigest {
        self.root_budget
    }
    pub fn digest(&self) -> ContractDigest {
        self.digest
    }
    pub(super) fn source(&self) -> &PcQuestionSource {
        &self.source
    }
    /// Conservative retained owner bytes; no selected View, latent or GPU data.
    pub fn owned_host_bytes(&self) -> Option<u64> {
        u64::try_from(std::mem::size_of::<Self>())
            .ok()?
            .checked_add(
                u64::try_from(
                    self.proposal
                        .capacity()
                        .checked_mul(std::mem::size_of::<BoardMove>())?,
                )
                .ok()?,
            )?
            .checked_add(self.source.owned_host_bytes()?)
    }
}
