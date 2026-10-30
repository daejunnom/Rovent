//! Move-only historical Repair custody. No report contains a selected View,
//! latent, live action permission, worker or NN completion authority.
use super::repair_cycle::{CycleFacts, PcRepairPhase};
use super::*;
use crate::pals::coordinator::AcceptedPeerSourceSeal;

/// Latest bounded recipe of an incomplete cycle, copied without its frozen
/// endpoint View. This is historical progress, never a resumable action.
pub struct PcRepairProgress {
    pub(super) cycle: u64,
    pub(super) parent_cycle: Option<u64>,
    pub(super) phase: PcRepairPhase,
    pub(super) first: BoardMove,
    pub(super) model: ModelValueIdentity,
    pub(super) root_source: AcceptedPeerSourceSeal,
    pub(super) prefix: Vec<BoardMove>,
    pub(super) proposal: Vec<BoardMove>,
    pub(super) counter: Option<Vec<BoardMove>>,
    pub(super) repaired: Option<Vec<BoardMove>>,
}
impl PcRepairProgress {
    pub fn cycle(&self) -> u64 {
        self.cycle
    }
    pub fn parent_cycle(&self) -> Option<u64> {
        self.parent_cycle
    }
    pub fn phase(&self) -> PcRepairPhase {
        self.phase
    }
    pub fn original_first_move(&self) -> BoardMove {
        self.first
    }
    pub fn model(&self) -> &ModelValueIdentity {
        &self.model
    }
    pub fn root_source(&self) -> &AcceptedRoleOrigin {
        self.root_source.origin()
    }
    pub fn root_source_view_digest(&self) -> ContractDigest {
        self.root_source.source_view_digest()
    }
    pub fn prefix(&self) -> &[BoardMove] {
        &self.prefix
    }
    pub fn proposal(&self) -> &[BoardMove] {
        &self.proposal
    }
    pub fn counter(&self) -> Option<&[BoardMove]> {
        self.counter.as_deref()
    }
    pub fn repaired(&self) -> Option<&[BoardMove]> {
        self.repaired.as_deref()
    }
    pub fn owned_host_bytes(&self) -> Option<u64> {
        inline::<Self>()?
            .checked_add(line_bytes(&self.prefix)?)?
            .checked_add(line_bytes(&self.proposal)?)?
            .checked_add(optional_line_bytes(&self.counter)?)?
            .checked_add(optional_line_bytes(&self.repaired)?)?
            .checked_add(model_bytes(&self.model)?)
    }
}

/// Historical root data moves once into the caller's receipt, or follows the
/// SAME semantic owner when the caller has not explicitly recovered it yet.
/// Receiving this report does not certify a successful physical shutdown.
pub struct PcRepairReport {
    authority: SearchAuthority,
    root_budget: ContractDigest,
    root_state: StateId,
    root_snapshot: PositionSnapshot,
    completion: PcPortCompletion,
    counters: PcPortCounters,
    cycles: PcRepairCycles,
}
impl PcRepairReport {
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn root_budget(&self) -> ContractDigest {
        self.root_budget
    }
    pub fn root_state(&self) -> StateId {
        self.root_state
    }
    pub fn root_snapshot(&self) -> &PositionSnapshot {
        &self.root_snapshot
    }
    pub fn completion(&self) -> PcPortCompletion {
        self.completion
    }
    pub fn counters(&self) -> PcPortCounters {
        self.counters
    }
    pub fn policy(&self) -> PostRepairRecheckPolicy {
        self.cycles.policy
    }
    pub fn witnesses(&self) -> impl Iterator<Item = &PcDivergenceWitness> {
        self.cycles.witnesses.iter()
    }
    pub fn incomplete_cycle_count(&self) -> usize {
        self.cycles.active.len()
    }
    pub fn incomplete_progress(&self) -> impl Iterator<Item = &PcRepairProgress> {
        self.cycles.active.values().map(|facts| &facts.progress)
    }
    pub fn incomplete_stopped_reason(&self, cycle: u64) -> Option<&'static str> {
        self.cycles
            .active
            .get(&cycle)
            .and_then(|facts| facts.stopped_reason)
    }
    pub fn incomplete_last_accepted(&self, cycle: u64) -> Option<&AcceptedRoleOrigin> {
        self.cycles
            .active
            .get(&cycle)
            .and_then(|facts| facts.last_accepted.as_ref())
            .map(AcceptedPeerSourceSeal::origin)
    }
    pub fn incomplete_last_source_view_digest(&self, cycle: u64) -> Option<ContractDigest> {
        self.cycles
            .active
            .get(&cycle)
            .and_then(|facts| facts.last_accepted.as_ref())
            .map(AcceptedPeerSourceSeal::source_view_digest)
    }
    pub fn incomplete_resolution(&self, cycle: u64) -> Option<PcDivergenceResolution> {
        self.cycles
            .active
            .contains_key(&cycle)
            .then_some(PcDivergenceResolution::Unknown)
    }
    pub fn incomplete_attack_ranked_source(&self, cycle: u64) -> Option<&PcAttackRankedSource> {
        self.cycles
            .active
            .get(&cycle)
            .and_then(|facts| facts.attack_source.as_deref())
    }
    pub fn incomplete_target(&self, cycle: u64) -> Option<&PcDivergenceSource> {
        self.cycles
            .active
            .get(&cycle)
            .and_then(|facts| facts.target.as_deref())
    }
    pub fn incomplete_repair(&self, cycle: u64) -> Option<&PcRepairAcceptedLine> {
        self.cycles
            .active
            .get(&cycle)
            .and_then(|facts| facts.repair.as_deref())
    }
    pub fn incomplete_recheck(&self, cycle: u64) -> Option<&PcRepairAcceptedLine> {
        self.cycles
            .active
            .get(&cycle)
            .and_then(|facts| facts.recheck.as_deref())
    }
    pub fn incomplete_repaired_endpoint(&self, cycle: u64) -> Option<&PcAcceptedFreshEndpoint> {
        self.cycles
            .active
            .get(&cycle)
            .and_then(|facts| facts.repair_endpoint.as_ref())
    }
    pub fn incomplete_counter_endpoint(&self, cycle: u64) -> Option<&PcAcceptedFreshEndpoint> {
        self.cycles
            .active
            .get(&cycle)
            .and_then(|facts| facts.counter_endpoint.as_ref())
    }
    /// Conservative retained ownership, not allocator peak or VRAM. Shared Arc
    /// facts may be counted more than once; View/latent memory is absent here.
    pub fn owned_host_bytes(&self) -> Option<u64> {
        let mut bytes = inline::<Self>()?
            .checked_add(u64::try_from(self.root_snapshot.retained_history_bytes()?).ok()?)?
            .checked_add(map_storage::<BoardMove, u8>(
                self.cycles.per_first.capacity(),
            )?)?
            .checked_add(map_storage::<u64, CycleFacts>(
                self.cycles.active.capacity(),
            )?)?
            .checked_add(
                u64::try_from(
                    self.cycles
                        .witnesses
                        .capacity()
                        .checked_mul(std::mem::size_of::<PcDivergenceWitness>())?,
                )
                .ok()?,
            )?;
        for facts in self.cycles.active.values() {
            bytes = bytes.checked_add(facts_bytes(facts)?)?;
        }
        for witness in &self.cycles.witnesses {
            bytes = bytes.checked_add(witness.owned_host_bytes()?)?;
        }
        Some(bytes)
    }
}

impl<M: RoleModel> PcSemanticPort<M> {
    /// Admission and all prepared/pending inputs must be retired before moving
    /// this report. Never-submitted recipes are closed with their raw progress retained. Actual success remains the native owner's separate receipt.
    pub fn take_repair_report(&mut self) -> Result<Option<PcRepairReport>, PalsError> {
        if self.completion == PcPortCompletion::Active || !self.pending.is_empty() {
            return Err(StoreError::InvalidConditions(
                "Repair report requires retired logical admission",
            )
            .into());
        }
        // These recipes were never submitted. Their last actual progress is
        // already in the report facts, and closed admission cannot run them.
        self.ready.clear();
        Ok(self.detach_repair_report())
    }
    pub(super) fn detach_repair_report(&mut self) -> Option<PcRepairReport> {
        let cycles = self.repair_cycles.take()?;
        let root_budget = cycles.root_budget;
        Some(PcRepairReport {
            authority: self.authority,
            root_budget,
            root_state: self.owner.semantic.nodes[self.root].state,
            root_snapshot: self.owner.semantic.nodes[self.root].position.snapshot(),
            completion: self.completion,
            counters: self.counters,
            cycles,
        })
    }
    pub fn repair_retained_host_bytes(&self) -> Option<u64> {
        let Some(cycles) = &self.repair_cycles else {
            return Some(0);
        };
        let mut bytes = inline::<PcRepairCycles>()?
            .checked_add(map_storage::<BoardMove, u8>(cycles.per_first.capacity())?)?
            .checked_add(map_storage::<u64, CycleFacts>(cycles.active.capacity())?)?
            .checked_add(
                u64::try_from(
                    cycles
                        .witnesses
                        .capacity()
                        .checked_mul(std::mem::size_of::<PcDivergenceWitness>())?,
                )
                .ok()?,
            )?;
        for facts in cycles.active.values() {
            bytes = bytes.checked_add(facts_bytes(facts)?)?;
        }
        for witness in &cycles.witnesses {
            bytes = bytes.checked_add(witness.owned_host_bytes()?)?;
        }
        for plan in &self.ready {
            if let Some(step) = &plan.repair {
                bytes = bytes.checked_add(step.owned_host_bytes()?)?;
            }
        }
        Some(bytes)
    }
}
impl<M: RoleModel> PcSemanticOwner<M> {
    pub fn retired_repair_report(&self) -> Option<&PcRepairReport> {
        self.semantic.retired_pc_repair.as_ref()
    }
    /// Explicit historical recovery. It never revives the retired root.
    pub fn take_retired_repair_report(&mut self) -> Option<PcRepairReport> {
        self.semantic.retired_pc_repair.take()
    }
    pub fn retired_repair_host_bytes(&self) -> Option<u64> {
        self.semantic
            .retired_pc_repair
            .as_ref()
            .map_or(Some(0), PcRepairReport::owned_host_bytes)
    }
}
fn inline<T>() -> Option<u64> {
    u64::try_from(std::mem::size_of::<T>()).ok()
}
fn map_storage<K, V>(capacity: usize) -> Option<u64> {
    if capacity == 0 {
        return Some(0);
    }
    // Include spare buckets and control bytes conservatively. This is a
    // declared retained upper bound, not an allocator measurement.
    u64::try_from(
        capacity
            .checked_mul(2)?
            .checked_add(16)?
            .checked_mul(std::mem::size_of::<(K, V)>().checked_add(1)?)?,
    )
    .ok()
}
pub(super) fn line_bytes(line: &Vec<BoardMove>) -> Option<u64> {
    u64::try_from(
        line.capacity()
            .checked_mul(std::mem::size_of::<BoardMove>())?,
    )
    .ok()
}
pub(super) fn optional_line_bytes(line: &Option<Vec<BoardMove>>) -> Option<u64> {
    line.as_ref().map_or(Some(0), line_bytes)
}
pub(super) fn model_bytes(model: &ModelValueIdentity) -> Option<u64> {
    u64::try_from(
        model
            .semantics
            .capacity()
            .checked_add(model.model.capacity())?
            .checked_add(model.encoding.capacity())?
            .checked_add(model.precision.capacity())?,
    )
    .ok()
}
fn facts_bytes(facts: &CycleFacts) -> Option<u64> {
    let mut bytes = inline::<CycleFacts>()?.checked_add(facts.progress.owned_host_bytes()?)?;
    if let Some(recipe) = &facts.followup {
        bytes = bytes.checked_add(recipe.owned_host_bytes()?)?;
    }
    if let Some(source) = &facts.attack_source {
        bytes = bytes.checked_add(source.owned_host_bytes()?)?;
    }
    if let Some(source) = &facts.target {
        bytes = bytes.checked_add(source.owned_host_bytes()?)?;
    }
    for line in [&facts.repair, &facts.recheck].into_iter().flatten() {
        bytes = bytes.checked_add(line.owned_host_bytes()?)?;
    }
    for endpoint in [&facts.repair_endpoint, &facts.counter_endpoint]
        .into_iter()
        .flatten()
    {
        bytes = bytes.checked_add(endpoint.owned_host_bytes()?)?;
    }
    Some(bytes)
}
