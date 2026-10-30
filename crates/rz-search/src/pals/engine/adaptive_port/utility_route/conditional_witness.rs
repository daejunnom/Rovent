//! Historical bodies of the actual action owner's conditional witnesses.
//! Raw bytes cannot construct this capability or mint a semantic outcome.
use super::*;
use crate::pals::engine::adaptive_port::utility_conditional_basis::{
    UtilityConditionalWitnessBasis, UtilityConditionalWitnessBasisUnavailable,
};
use crate::pals::engine::adaptive_port::validator_port::{
    BoundedObservationPreimage, ObservationSink,
};

mod historical;
pub(in crate::pals::engine::adaptive_port) use historical::{
    HistoricalWitnessObservation, HistoricalWitnessRawScore,
    parse_validated_utility_witness_preimage,
};
pub use historical::{
    UtilityWitnessPreimageError, UtilityWitnessPreimageExpectation,
    validate_utility_witness_preimage,
};

/// Payload limit only; it creates no new role, CPU or Store execution credit.
pub const MAX_UTILITY_WITNESS_BODY_BYTES: usize = 16 * 1024 * 1024;

pub struct UtilityConditionalWitnessBody {
    observation: ObservationId,
    kind: ActionWitnessKind,
    identity: ContractDigest,
    source_question: ContractDigest,
    source_frame: FrameId,
    raw_evidence: ContractDigest,
    preimage: Vec<u8>,
    basis: Option<UtilityConditionalWitnessBasis>,
    basis_unavailable_reason: Option<UtilityConditionalWitnessBasisUnavailable>,
}
impl UtilityConditionalWitnessBody {
    pub fn observation(&self) -> ObservationId {
        self.observation
    }
    pub fn kind(&self) -> ActionWitnessKind {
        self.kind
    }
    pub fn identity(&self) -> ContractDigest {
        self.identity
    }
    pub fn source_question(&self) -> ContractDigest {
        self.source_question
    }
    pub fn source_frame(&self) -> FrameId {
        self.source_frame
    }
    pub fn raw_evidence(&self) -> ContractDigest {
        self.raw_evidence
    }
    pub fn preimage(&self) -> &[u8] {
        &self.preimage
    }
    pub fn basis(&self) -> Option<&UtilityConditionalWitnessBasis> {
        self.basis.as_ref()
    }
    pub fn basis_unavailable_reason(&self) -> Option<UtilityConditionalWitnessBasisUnavailable> {
        self.basis_unavailable_reason
    }
}

/// Move-only bounded bytes, retaining no selected View, Bound or CPU stack.
pub struct UtilityConditionalWitnessBodies {
    action: UtilityRouteActionId,
    owner: StoreReadOwner,
    generation: u64,
    root: StateId,
    entries: Vec<UtilityConditionalWitnessBody>,
}
impl UtilityConditionalWitnessBodies {
    pub fn action(&self) -> &UtilityRouteActionId {
        &self.action
    }
    pub fn entries(&self) -> &[UtilityConditionalWitnessBody] {
        &self.entries
    }
    pub fn matches_store(&self, stores: &PalsStores) -> bool {
        self.owner.matches(stores)
            && self.generation == stores.generation()
            && self.generation == self.action.authority.root.0
            && stores.root().is_some_and(|root| {
                stores
                    .situations
                    .get(root)
                    .is_ok_and(|s| s.state == self.root)
            })
    }
    pub fn matches_outcome(&self, outcome: &UtilityRouteSemanticOutcome) -> bool {
        self.action == *outcome.id()
            && self.entries.len() == outcome.witnesses().len()
            && self
                .entries
                .iter()
                .zip(outcome.witnesses())
                .all(|(body, witness)| {
                    body.observation == witness.observation()
                        && body.kind == witness.kind()
                        && body.identity == witness.identity()
                        && body.source_question == witness.source_question()
                        && body.source_frame == witness.source_frame()
                        && body.raw_evidence == witness.raw_evidence()
                })
    }
    /// Known owned backing, without claiming RSS or physical sharing.
    pub fn owned_host_bytes(&self) -> Option<u64> {
        let mut bytes = std::mem::size_of::<Self>().checked_add(
            self.entries
                .capacity()
                .checked_mul(std::mem::size_of::<UtilityConditionalWitnessBody>())?,
        )?;
        for entry in &self.entries {
            bytes = bytes.checked_add(entry.preimage.capacity())?;
            if let Some(basis) = &entry.basis {
                bytes = bytes.checked_add(usize::try_from(basis.owned_host_bytes()?).ok()?)?;
            }
        }
        u64::try_from(bytes).ok()
    }
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// Recover historical data from the same concluded Store facts. Deadline
    /// or cancellation is not renewed or interpreted as new admission here.
    pub fn freeze_utility_witness_bodies(
        &self,
        outcome: &UtilityRouteSemanticOutcome,
    ) -> Result<UtilityConditionalWitnessBodies, PalsError> {
        let action = &outcome.action;
        let preparation = &action.preparation;
        if action.failed
            || !preparation.store.matches(&self.engine.semantic.stores)
            || preparation.id.authority != self.authority
            || preparation.generation != self.engine.semantic.stores.generation()
            || preparation.generation != self.authority.root.0
            || preparation.root != self.root_state_id()
            || !self.engine.semantic.stores.root().is_some_and(|root| {
                self.engine
                    .semantic
                    .stores
                    .situations
                    .get(root)
                    .is_ok_and(|situation| situation.state == preparation.root)
            })
            || action.committed.is_empty()
            || action.witnesses.len() > MAX_ACTION_WITNESSES
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let maximum = usize::try_from(
            preparation
                .budget
                .original_store_byte_limit
                .min(MAX_UTILITY_WITNESS_BODY_BYTES as u64),
        )
        .map_err(|_| PalsError::Capacity)?;
        let mut remaining = maximum;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(action.witnesses.len())
            .map_err(|_| PalsError::Capacity)?;
        let mut identities = HashSet::new();
        identities
            .try_reserve(action.witnesses.len())
            .map_err(|_| PalsError::Capacity)?;
        for witness in &action.witnesses {
            if witness.source_frame.authority != self.authority
                || !action.witnessed.contains(&witness.identity)
                || preparation.baseline.contains(&witness.identity)
                || !identities.insert(witness.identity)
                || !action
                    .committed_sources
                    .values()
                    .any(|source| *source == (witness.source_frame, witness.source_question))
                || self.validator_committed_witness_kind(witness.observation)? != Some(witness.kind)
                || self.conditional_witness_line(witness.observation)
                    != Some(witness.line.as_slice())
                || witness.line.first() != Some(&preparation.first)
                || self.validator_witness_identity(witness.kind, &witness.line)? != witness.identity
            {
                return Err(RoleError::InvalidOutput.into());
            }
            let mut body = BoundedObservationPreimage::new(remaining);
            body.write(b"rz-rovent-utility-actual-conditional-witness/1\0");
            body.write(witness.source_question.0);
            body.write(witness.source_frame.sequence.to_le_bytes());
            body.write(witness.identity.0);
            self.validator_write_observation_into(&mut body, witness.observation, 0)?;
            let preimage = body.into_bytes()?;
            if ContractDigest(Sha256::digest(&preimage).into()) != witness.raw_evidence {
                return Err(RoleError::InvalidOutput.into());
            }
            remaining = remaining
                .checked_sub(preimage.len())
                .ok_or(PalsError::Capacity)?;
            // Basis bytes use the same declared payload allowance as the raw
            // preimage. A missing source keeps the row and an explicit reason.
            let (basis, basis_unavailable_reason) = match self
                .freeze_utility_conditional_basis(outcome, witness, remaining, &preimage)
            {
                Ok(basis) => {
                    remaining = remaining
                        .checked_sub(basis.original_bytes().len())
                        .ok_or(PalsError::Capacity)?;
                    (Some(basis), None)
                }
                Err(reason) => (None, Some(reason)),
            };
            entries.push(UtilityConditionalWitnessBody {
                observation: witness.observation,
                kind: witness.kind,
                identity: witness.identity,
                source_question: witness.source_question,
                source_frame: witness.source_frame,
                raw_evidence: witness.raw_evidence,
                preimage,
                basis,
                basis_unavailable_reason,
            });
        }
        Ok(UtilityConditionalWitnessBodies {
            action: preparation.id.clone(),
            owner: self.engine.semantic.stores.read_owner(),
            generation: preparation.generation,
            root: preparation.root,
            entries,
        })
    }
}
