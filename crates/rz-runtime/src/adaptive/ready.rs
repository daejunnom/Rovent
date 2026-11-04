use crate::RuntimeFault;
use std::time::Duration;

pub const MAX_READY_JOBS: usize = 64;

/// Unknown is retained as a fact, never interpreted as zero service time.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReadyServiceCost {
    Known(Duration),
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReadyJob<J, C, L> {
    pub job: J,
    pub candidate: C,
    pub locality: L,
    pub hard_deadline: Duration,
    pub enqueued_at: Duration,
    pub dependencies_ready: bool,
    /// Bounded preference within one deadline; deficit eventually overrides it.
    pub dependency_priority: u8,
    pub service_cost: ReadyServiceCost,
}

#[derive(Debug)]
pub struct ReadyRejected<J, C, L> {
    pub reason: RuntimeFault,
    pub job: ReadyJob<J, C, L>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReadySelection<J, C, L> {
    pub job: ReadyJob<J, C, L>,
    pub selection_sequence: u64,
    /// Dispatch fairness credit, not NN/CPU node, elapsed-time or utility credit.
    pub candidate_deficit_after: i64,
    pub starvation_age_before: u64,
}

#[derive(Debug)]
struct Pending<J, C, L> {
    descriptor: ReadyJob<J, C, L>,
    admission: u64,
}

#[derive(Debug)]
struct Candidate<C> {
    id: C,
    deficit: i64,
    last_selected: u64,
}

struct SelectionPlan {
    pending_index: usize,
    candidate_index: usize,
    eligible_candidates: [bool; MAX_READY_JOBS],
    next_sequence: u64,
    deficit_after: i64,
    starvation_age_before: u64,
}

/// Bounded EDF/dependency preference with per-candidate deficit and aging.
/// No search data is encoded, no prepared input is copied and no task migrates
/// here. Only never-started Jobs are eligible for idle-worker assignment; owner-
/// bound paused continuations use their original worker's separate path.
///
/// Deadline priority takes precedence. A dependency preference is capped at
/// four dispatch credits. Among repeatedly eligible candidates sharing a
/// root's original hard deadline, deficit eventually overrides even the highest
/// dependency preference and locality. No scheduler can promise service to a
/// later-deadline tier under unlimited earlier-deadline arrivals; admission and
/// the original deadline remain the coordinator's responsibility.
#[derive(Debug)]
pub struct ReadySelector<J, C, L> {
    pending: Vec<Pending<J, C, L>>,
    candidates: Vec<Candidate<C>>,
    capacity: usize,
    admission: u64,
    selection: u64,
}
impl<J: Eq, C: Clone + Eq, L: Eq> ReadySelector<J, C, L> {
    pub fn new(capacity: usize) -> Result<Self, RuntimeFault> {
        if capacity == 0 || capacity > MAX_READY_JOBS {
            return Err(RuntimeFault::InvalidLimits);
        }
        let mut pending = Vec::new();
        let mut candidates = Vec::new();
        pending
            .try_reserve_exact(capacity)
            .map_err(|_| RuntimeFault::ResourceLimit)?;
        candidates
            .try_reserve_exact(MAX_READY_JOBS)
            .map_err(|_| RuntimeFault::ResourceLimit)?;
        Ok(Self {
            pending,
            candidates,
            capacity,
            admission: 0,
            selection: 0,
        })
    }
    pub fn len(&self) -> usize {
        self.pending.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }
    pub fn admit(&mut self, job: ReadyJob<J, C, L>) -> Result<(), ReadyRejected<J, C, L>> {
        let new_candidate = !self
            .candidates
            .iter()
            .any(|entry| entry.id == job.candidate);
        let reason = if self
            .pending
            .iter()
            .any(|entry| entry.descriptor.job == job.job)
        {
            Some(RuntimeFault::DuplicateRequest)
        } else if self.pending.len() == self.capacity {
            Some(RuntimeFault::QueueFull)
        } else if new_candidate && self.candidates.len() == MAX_READY_JOBS {
            Some(RuntimeFault::ResourceLimit)
        } else if job.enqueued_at >= job.hard_deadline {
            Some(RuntimeFault::Expired)
        } else if self.admission == u64::MAX {
            Some(RuntimeFault::ResourceOverflow)
        } else {
            None
        };
        if let Some(reason) = reason {
            return Err(ReadyRejected { reason, job });
        }
        self.admission += 1;
        if new_candidate {
            self.candidates.push(Candidate {
                id: job.candidate.clone(),
                deficit: 0,
                last_selected: self.selection,
            });
        }
        self.pending.push(Pending {
            descriptor: job,
            admission: self.admission,
        });
        Ok(())
    }
    pub fn set_dependencies_ready(&mut self, job: &J, ready: bool) -> Result<(), RuntimeFault> {
        let entry = self
            .pending
            .iter_mut()
            .find(|entry| &entry.descriptor.job == job)
            .ok_or(RuntimeFault::Stale)?;
        entry.descriptor.dependencies_ready = ready;
        Ok(())
    }
    /// Explicit removal never reports an unknown task as successfully canceled.
    pub fn remove(&mut self, job: &J) -> Result<ReadyJob<J, C, L>, RuntimeFault> {
        let position = self
            .pending
            .iter()
            .position(|entry| &entry.descriptor.job == job)
            .ok_or(RuntimeFault::Stale)?;
        Ok(self.pending.swap_remove(position).descriptor)
    }
    /// The owner calls this only after proving the candidate has no in-flight
    /// work/token. The selector can verify only its own pending references.
    pub fn retire_candidate(&mut self, candidate: &C) -> Result<(), RuntimeFault> {
        if self
            .pending
            .iter()
            .any(|entry| &entry.descriptor.candidate == candidate)
        {
            return Err(RuntimeFault::InvalidCompletion);
        }
        let position = self
            .candidates
            .iter()
            .position(|entry| &entry.id == candidate)
            .ok_or(RuntimeFault::Stale)?;
        self.candidates.swap_remove(position);
        Ok(())
    }
    /// Returns only a descriptor. A SliceGrant is still needed before work can
    /// run. Scans are bounded by 64 and queries do not clone/sort snapshots.
    pub fn select(
        &mut self,
        now: Duration,
        preferred_locality: Option<&L>,
    ) -> Result<Option<ReadySelection<J, C, L>>, RuntimeFault> {
        let Some(plan) = self.selection_plan(now, preferred_locality, |_| true)? else {
            return Ok(None);
        };
        for (index, candidate) in self.candidates.iter_mut().enumerate() {
            if plan.eligible_candidates[index] {
                candidate.deficit += 1;
            }
        }
        self.candidates[plan.candidate_index].deficit = plan.deficit_after;
        self.candidates[plan.candidate_index].last_selected = plan.next_sequence;
        self.selection = plan.next_sequence;
        Ok(Some(ReadySelection {
            job: self.pending.swap_remove(plan.pending_index).descriptor,
            selection_sequence: plan.next_sequence,
            candidate_deficit_after: plan.deficit_after,
            starvation_age_before: plan.starvation_age_before,
        }))
    }
    /// Inspect the next matching eligible descriptor without consuming it or
    /// advancing any admission, selection, deficit or starvation state. The
    /// predicate is evaluated once per currently eligible job, before choosing
    /// its deadline tier; an earlier nonmatching job cannot hide a later match.
    /// This observation is neither a reservation nor permission to dispatch.
    pub fn peek_matching(
        &self,
        now: Duration,
        preferred_locality: Option<&L>,
        predicate: impl FnMut(&ReadyJob<J, C, L>) -> bool,
    ) -> Result<Option<&ReadyJob<J, C, L>>, RuntimeFault> {
        Ok(self
            .selection_plan(now, preferred_locality, predicate)?
            .map(|plan| &self.pending[plan.pending_index].descriptor))
    }
    fn selection_plan(
        &self,
        now: Duration,
        preferred_locality: Option<&L>,
        mut predicate: impl FnMut(&ReadyJob<J, C, L>) -> bool,
    ) -> Result<Option<SelectionPlan>, RuntimeFault> {
        let mut matching = [false; MAX_READY_JOBS];
        let mut deadline = None;
        for (index, entry) in self.pending.iter().enumerate() {
            let job = &entry.descriptor;
            if job.dependencies_ready
                && job.enqueued_at <= now
                && job.hard_deadline > now
                && predicate(job)
            {
                matching[index] = true;
                deadline = Some(deadline.map_or(job.hard_deadline, |old: Duration| {
                    old.min(job.hard_deadline)
                }));
            }
        }
        let Some(deadline) = deadline else {
            return Ok(None);
        };
        let next_sequence = self
            .selection
            .checked_add(1)
            .ok_or(RuntimeFault::ResourceOverflow)?;
        let mut eligible_candidates = [false; MAX_READY_JOBS];
        for (index, entry) in self.pending.iter().enumerate() {
            if matching[index] && eligible(&entry.descriptor, now, deadline) {
                let index = self
                    .candidates
                    .iter()
                    .position(|candidate| candidate.id == entry.descriptor.candidate)
                    .expect("admitted job retains candidate registration");
                eligible_candidates[index] = true;
            }
        }
        let count = eligible_candidates
            .iter()
            .filter(|eligible| **eligible)
            .count() as i64;
        // Compute all overflow checks before mutating fairness state.
        for (index, candidate) in self.candidates.iter().enumerate() {
            if eligible_candidates[index] {
                candidate
                    .deficit
                    .checked_add(5)
                    .ok_or(RuntimeFault::ResourceOverflow)?;
            }
        }
        let mut selected: Option<(usize, usize)> = None;
        for (index, entry) in self.pending.iter().enumerate() {
            if !matching[index] || !eligible(&entry.descriptor, now, deadline) {
                continue;
            }
            let candidate_index = self
                .candidates
                .iter()
                .position(|candidate| candidate.id == entry.descriptor.candidate)
                .expect("admitted job retains candidate registration");
            if selected.is_none_or(|(previous, previous_candidate)| {
                let candidate = &self.candidates[candidate_index];
                let old_candidate = &self.candidates[previous_candidate];
                let old = &self.pending[previous];
                let score =
                    candidate.deficit + i64::from(entry.descriptor.dependency_priority.min(4));
                let old_score =
                    old_candidate.deficit + i64::from(old.descriptor.dependency_priority.min(4));
                score
                    .cmp(&old_score)
                    .then_with(|| old_candidate.last_selected.cmp(&candidate.last_selected))
                    .then_with(|| {
                        old.descriptor
                            .enqueued_at
                            .cmp(&entry.descriptor.enqueued_at)
                    })
                    .then_with(|| {
                        let local = preferred_locality
                            .is_some_and(|locality| locality == &entry.descriptor.locality);
                        let old_local = preferred_locality
                            .is_some_and(|locality| locality == &old.descriptor.locality);
                        local.cmp(&old_local)
                    })
                    .then_with(|| old.admission.cmp(&entry.admission))
                    .is_gt()
            }) {
                selected = Some((index, candidate_index));
            }
        }
        let (index, candidate_index) = selected.expect("deadline contains an eligible job");
        let deficit_after = self.candidates[candidate_index]
            .deficit
            .checked_add(1)
            .and_then(|deficit| deficit.checked_sub(count))
            .ok_or(RuntimeFault::ResourceOverflow)?;
        Ok(Some(SelectionPlan {
            pending_index: index,
            candidate_index,
            eligible_candidates,
            next_sequence,
            deficit_after,
            starvation_age_before: next_sequence - self.candidates[candidate_index].last_selected,
        }))
    }
}

fn eligible<J, C, L>(job: &ReadyJob<J, C, L>, now: Duration, deadline: Duration) -> bool {
    job.dependencies_ready
        && job.enqueued_at <= now
        && job.hard_deadline > now
        && job.hard_deadline == deadline
}

#[cfg(test)]
mod tests {
    use super::*;
    fn job(id: u64, candidate: u8) -> ReadyJob<u64, u8, u8> {
        ReadyJob {
            job: id,
            candidate,
            locality: candidate,
            hard_deadline: Duration::from_secs(10),
            enqueued_at: Duration::ZERO,
            dependencies_ready: true,
            dependency_priority: 0,
            service_cost: ReadyServiceCost::Unknown,
        }
    }
    #[test]
    fn dependency_and_original_deadline_precede_fairness_and_locality() {
        let mut selector = ReadySelector::new(8).unwrap();
        selector.admit(job(1, 1)).unwrap();
        let mut earlier = job(2, 2);
        earlier.hard_deadline = Duration::from_secs(2);
        selector.admit(earlier).unwrap();
        let mut blocked = job(3, 3);
        blocked.hard_deadline = Duration::from_secs(1);
        blocked.dependencies_ready = false;
        selector.admit(blocked).unwrap();
        assert_eq!(
            selector
                .select(Duration::ZERO, Some(&1))
                .unwrap()
                .unwrap()
                .job
                .job,
            2
        );
        selector.set_dependencies_ready(&3, true).unwrap();
        assert_eq!(
            selector
                .select(Duration::ZERO, Some(&1))
                .unwrap()
                .unwrap()
                .job
                .job,
            3
        );
        let mut dependency = job(4, 4);
        dependency.dependency_priority = 1;
        selector.admit(dependency).unwrap();
        assert_eq!(
            selector
                .select(Duration::ZERO, Some(&1))
                .unwrap()
                .unwrap()
                .job
                .job,
            4
        );
        assert_eq!(
            selector.select(Duration::from_secs(11), None).unwrap(),
            None
        );
    }
    #[test]
    fn repeated_local_preference_does_not_starve_eligible_candidates() {
        let mut selector = ReadySelector::new(8).unwrap();
        for candidate in 1..=4 {
            selector
                .admit(job(u64::from(candidate), candidate))
                .unwrap();
        }
        let mut counts = [0; 4];
        for id in 10..110 {
            let selection = selector.select(Duration::ZERO, Some(&1)).unwrap().unwrap();
            assert_eq!(selection.job.service_cost, ReadyServiceCost::Unknown);
            assert!(selection.starvation_age_before <= 4);
            counts[usize::from(selection.job.candidate - 1)] += 1;
            selector.admit(job(id, selection.job.candidate)).unwrap();
        }
        assert_eq!(counts, [25; 4]);
    }
    #[test]
    fn repeated_high_dependency_priority_cannot_starve_other_candidate() {
        let mut selector = ReadySelector::new(2).unwrap();
        selector.admit(job(1, 1)).unwrap();
        let mut urgent = job(2, 2);
        urgent.dependency_priority = u8::MAX;
        selector.admit(urgent).unwrap();
        let mut counts = [0; 2];
        for id in 10..110 {
            let selection = selector.select(Duration::ZERO, Some(&2)).unwrap().unwrap();
            assert!(selection.starvation_age_before <= 4);
            counts[usize::from(selection.job.candidate - 1)] += 1;
            let mut next = job(id, selection.job.candidate);
            if next.candidate == 2 {
                next.dependency_priority = u8::MAX;
            }
            selector.admit(next).unwrap();
        }
        assert!((48..=52).contains(&counts[0]));
        assert!((48..=52).contains(&counts[1]));
    }
    #[test]
    fn bounded_admission_and_candidate_retirement_do_not_reassign_ids() {
        let mut selector = ReadySelector::new(1).unwrap();
        selector.admit(job(1, 1)).unwrap();
        assert_eq!(
            selector.admit(job(1, 2)).unwrap_err().reason,
            RuntimeFault::DuplicateRequest
        );
        assert_eq!(
            selector.admit(job(2, 2)).unwrap_err().reason,
            RuntimeFault::QueueFull
        );
        assert_eq!(
            selector.retire_candidate(&1),
            Err(RuntimeFault::InvalidCompletion)
        );
        let descriptor = selector.remove(&1).unwrap();
        assert_eq!(descriptor.job, 1);
        selector.retire_candidate(&1).unwrap();
        assert_eq!(selector.candidate_count(), 0);
        assert_eq!(selector.remove(&1), Err(RuntimeFault::Stale));
    }

    #[test]
    fn repeated_peeks_preserve_queue_and_all_fairness_state() {
        let mut observed = ReadySelector::new(4).unwrap();
        let mut control = ReadySelector::new(4).unwrap();
        for candidate in 1..=4 {
            let mut descriptor = job(u64::from(candidate), candidate);
            descriptor.dependency_priority = candidate;
            observed.admit(descriptor.clone()).unwrap();
            control.admit(descriptor).unwrap();
        }
        for id in 10..50 {
            let pending_before: Vec<_> = observed
                .pending
                .iter()
                .map(|entry| (entry.descriptor.clone(), entry.admission))
                .collect();
            let fairness_before: Vec<_> = observed
                .candidates
                .iter()
                .map(|entry| (entry.id, entry.deficit, entry.last_selected))
                .collect();
            let counters_before = (observed.admission, observed.selection, observed.capacity);
            let first = observed
                .peek_matching(Duration::ZERO, Some(&1), |_| true)
                .unwrap()
                .cloned();
            for _ in 0..3 {
                assert_eq!(
                    observed
                        .peek_matching(Duration::ZERO, Some(&1), |_| true)
                        .unwrap(),
                    first.as_ref()
                );
                observed
                    .peek_matching(Duration::ZERO, None, |job| job.candidate % 2 == 0)
                    .unwrap();
            }
            assert_eq!(
                observed
                    .pending
                    .iter()
                    .map(|entry| (entry.descriptor.clone(), entry.admission))
                    .collect::<Vec<_>>(),
                pending_before
            );
            assert_eq!(
                observed
                    .candidates
                    .iter()
                    .map(|entry| (entry.id, entry.deficit, entry.last_selected))
                    .collect::<Vec<_>>(),
                fairness_before
            );
            assert_eq!(
                (observed.admission, observed.selection, observed.capacity),
                counters_before
            );
            let selected = observed.select(Duration::ZERO, Some(&1)).unwrap().unwrap();
            assert_eq!(Some(&selected.job), first.as_ref());
            assert_eq!(
                Some(selected.clone()),
                control.select(Duration::ZERO, Some(&1)).unwrap()
            );
            let mut next = job(id, selected.job.candidate);
            next.dependency_priority = selected.job.dependency_priority;
            observed.admit(next.clone()).unwrap();
            control.admit(next).unwrap();
        }
    }

    #[test]
    fn matching_role_is_filtered_before_earliest_deadline() {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        enum Role {
            Proposer,
            Critic,
            Researcher,
        }
        let mut selector = ReadySelector::new(4).unwrap();
        for (id, role, deadline) in [
            (1, Role::Proposer, 1),
            (2, Role::Researcher, 2),
            (3, Role::Critic, 4),
            (4, Role::Critic, 5),
        ] {
            selector
                .admit(ReadyJob {
                    job: (id, role),
                    candidate: id,
                    locality: id,
                    hard_deadline: Duration::from_secs(deadline),
                    enqueued_at: Duration::ZERO,
                    dependencies_ready: true,
                    dependency_priority: 0,
                    service_cost: ReadyServiceCost::Unknown,
                })
                .unwrap();
        }
        let mut predicate_calls = 0;
        assert_eq!(
            selector
                .peek_matching(Duration::ZERO, Some(&4), |job| {
                    predicate_calls += 1;
                    job.job.1 == Role::Critic
                })
                .unwrap()
                .unwrap()
                .job,
            (3, Role::Critic)
        );
        assert_eq!(predicate_calls, 4);
        assert_eq!(
            selector
                .select(Duration::ZERO, Some(&4))
                .unwrap()
                .unwrap()
                .job
                .job,
            (1, Role::Proposer)
        );
    }

    #[test]
    fn peeks_keep_expired_blocked_and_future_jobs_in_the_bounded_queue() {
        let mut selector = ReadySelector::new(3).unwrap();
        let mut expired = job(1, 1);
        expired.hard_deadline = Duration::from_secs(1);
        selector.admit(expired).unwrap();
        let mut blocked = job(2, 2);
        blocked.dependencies_ready = false;
        selector.admit(blocked).unwrap();
        let mut future = job(3, 3);
        future.enqueued_at = Duration::from_secs(2);
        selector.admit(future).unwrap();
        let now = Duration::from_secs(1);
        assert_eq!(selector.peek_matching(now, None, |_| true).unwrap(), None);
        assert_eq!(selector.len(), 3);
        assert_eq!(selector.candidate_count(), 3);
        assert_eq!(selector.admission, 3);
        assert_eq!(selector.selection, 0);
        assert_eq!(
            selector.admit(job(4, 4)).unwrap_err().reason,
            RuntimeFault::QueueFull
        );
        selector.set_dependencies_ready(&2, true).unwrap();
        assert_eq!(
            selector
                .peek_matching(now, None, |job| job.job == 3)
                .unwrap(),
            None
        );
        assert_eq!(
            selector
                .peek_matching(now, None, |_| true)
                .unwrap()
                .unwrap()
                .job,
            2
        );
        assert_eq!(selector.select(now, None).unwrap().unwrap().job.job, 2);
        assert_eq!(selector.len(), 2);
        assert_eq!(
            selector
                .peek_matching(Duration::from_secs(2), None, |_| true)
                .unwrap()
                .unwrap()
                .job,
            3
        );
        assert_eq!(selector.len(), 2);
    }

    #[test]
    fn peek_overflow_checks_do_not_mutate_selection_state() {
        let mut selector = ReadySelector::new(1).unwrap();
        selector.admit(job(1, 1)).unwrap();
        selector.selection = u64::MAX;
        assert_eq!(
            selector.peek_matching(Duration::ZERO, None, |_| true),
            Err(RuntimeFault::ResourceOverflow)
        );
        assert_eq!(
            selector.select(Duration::ZERO, None),
            Err(RuntimeFault::ResourceOverflow)
        );
        assert_eq!(selector.len(), 1);
        assert_eq!(selector.selection, u64::MAX);
        assert_eq!(selector.candidates[0].deficit, 0);
        assert_eq!(
            selector
                .peek_matching(Duration::ZERO, None, |_| false)
                .unwrap(),
            None
        );
        selector.selection = 0;
        selector.candidates[0].deficit = i64::MAX - 4;
        assert_eq!(
            selector.peek_matching(Duration::ZERO, None, |_| true),
            Err(RuntimeFault::ResourceOverflow)
        );
        assert_eq!(selector.len(), 1);
        assert_eq!(selector.selection, 0);
        assert_eq!(selector.admission, 1);
        assert_eq!(selector.candidates[0].deficit, i64::MAX - 4);
        assert_eq!(selector.candidates[0].last_selected, 0);
    }
}
