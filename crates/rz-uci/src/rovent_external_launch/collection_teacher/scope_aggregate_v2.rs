//! Original roster, actor terminal and process evidence are combined here.
//! This receipt never certifies teacher semantic success.
use super::*;
use crate::pals_adaptive::{RoventTeacherScopePhysicalAckV2, RoventTeacherScopeWorkerAckV2};

pub enum RoventTeacherRetiredActorV2 {
    Physical(RoventTeacherRetirementTokenV2),
    NoChild(RoventTeacherNoChildRetirementTokenV2),
}
impl RoventTeacherRetiredActorV2 {
    fn owner(&self) -> &RoventCollectionTeacherOwnerV2 {
        match self {
            Self::Physical(token) => token.owner(),
            Self::NoChild(token) => token.owner(),
        }
    }
    fn matches_physical_state(&self, state: &CollectionMemberPhysicalState) -> bool {
        matches!(
            (self, state),
            (
                Self::Physical(_),
                CollectionMemberPhysicalState::PhysicallyConfirmed
            ) | (
                Self::NoChild(_),
                CollectionMemberPhysicalState::IssuedNoChild
                    | CollectionMemberPhysicalState::AcquiredNoChild
            )
        )
    }
}

pub struct RoventTeacherScopeAggregateAckV2 {
    physical: RoventTeacherScopePhysicalAckV2,
    workers: Option<RoventTeacherScopeWorkerAckV2>,
    retired: Vec<RoventTeacherRetiredActorV2>,
}

/// Every input and any partially retired owner remains with the caller.
pub struct RoventTeacherScopeAggregateFailureV2 {
    pub reason: &'static str,
    pub physical: RoventTeacherScopePhysicalAckV2,
    pub workers: Option<RoventTeacherScopeWorkerAckV2>,
    pub remaining: Vec<RoventTeacherWorkerTerminalTokenV2>,
    pub retired: Vec<RoventTeacherRetiredActorV2>,
}
impl RoventTeacherScopeAggregateFailureV2 {
    /// Retry with the same retained original and partially converted evidence.
    pub fn retry(self) -> Result<RoventTeacherScopeAggregateAckV2, Self> {
        RoventTeacherScopeAggregateAckV2::attempt(
            self.physical,
            self.workers,
            self.remaining,
            self.retired,
        )
    }
}

impl RoventTeacherScopeAggregateAckV2 {
    pub fn matches_admission(
        &self,
        admission: &crate::pals_adaptive::RoventCollectionRootAdmissionV2,
    ) -> bool {
        self.physical.matches_admission(admission)
    }
    pub fn physical(&self) -> &RoventTeacherScopePhysicalAckV2 {
        &self.physical
    }
    pub fn workers(&self) -> Option<&RoventTeacherScopeWorkerAckV2> {
        self.workers.as_ref()
    }
    pub fn retired(&self) -> &[RoventTeacherRetiredActorV2] {
        &self.retired
    }

    pub fn try_build(
        physical: RoventTeacherScopePhysicalAckV2,
        workers: Option<RoventTeacherScopeWorkerAckV2>,
        remaining: Vec<RoventTeacherWorkerTerminalTokenV2>,
    ) -> Result<Self, RoventTeacherScopeAggregateFailureV2> {
        Self::attempt(physical, workers, remaining, Vec::new())
    }
    fn attempt(
        physical: RoventTeacherScopePhysicalAckV2,
        workers: Option<RoventTeacherScopeWorkerAckV2>,
        mut remaining: Vec<RoventTeacherWorkerTerminalTokenV2>,
        mut retired: Vec<RoventTeacherRetiredActorV2>,
    ) -> Result<Self, RoventTeacherScopeAggregateFailureV2> {
        let reject = |reason, physical, workers, remaining, retired| {
            Err(RoventTeacherScopeAggregateFailureV2 {
                reason,
                physical,
                workers,
                remaining,
                retired,
            })
        };
        if physical.is_sealed_no_scope() {
            return if workers.is_none() && remaining.is_empty() && retired.is_empty() {
                Ok(Self {
                    physical,
                    workers,
                    retired,
                })
            } else {
                reject(
                    "no-scope has worker evidence",
                    physical,
                    workers,
                    remaining,
                    retired,
                )
            };
        }
        let valid = (|| {
            let p = physical.scope_outcome()?;
            let w_ack = workers.as_ref()?;
            if !w_ack.matches_admission(physical.original_admission()) {
                return None;
            }
            let w = w_ack.outcome();
            if !p.same_issuer_as_worker(w) || p.members().len() != w.members().len() {
                return None;
            }
            for (index, ((p_reg, p_state), (w_reg, w_state))) in
                p.members().iter().zip(w.members()).enumerate()
            {
                if p_reg != w_reg {
                    return None;
                }
                let matching: Vec<_> = remaining
                    .iter()
                    .filter(|token| w.matches_member_at(index, token.member_witness()))
                    .collect();
                let matching_retired: Vec<_> = retired
                    .iter()
                    .filter(|token| w.matches_member_at(index, &token.owner().member_witness))
                    .collect();
                match w_state {
                    CollectionMemberWorkerState::NoThread => {
                        if !matching.is_empty()
                            || !matching_retired.is_empty()
                            || !matches!(
                                p_state,
                                CollectionMemberPhysicalState::NeverIssuedNoChild
                                    | CollectionMemberPhysicalState::IssuedNoChild
                                    | CollectionMemberPhysicalState::AcquiredNoChild
                            )
                        {
                            return None;
                        }
                    }
                    CollectionMemberWorkerState::Joined => {
                        if matching.len() + matching_retired.len() != 1
                            || !matches!(
                                p_state,
                                CollectionMemberPhysicalState::PhysicallyConfirmed
                                    | CollectionMemberPhysicalState::IssuedNoChild
                                    | CollectionMemberPhysicalState::AcquiredNoChild
                            )
                        {
                            return None;
                        }
                        if let Some(token) = matching.first() {
                            if !token.matches_admission(physical.original_admission())
                                || token.registration() != *w_reg
                                || !token.has_original_join_witness()
                            {
                                return None;
                            }
                        }
                        if let Some(token) = matching_retired.first() {
                            let owner = token.owner();
                            if !token.matches_physical_state(p_state)
                                || !owner
                                    .admission()
                                    .same_original_admission(physical.original_admission())
                                || owner.worker.registration() != *w_reg
                                || !owner.worker.scope_join_witness().is_some_and(|joined| {
                                    owner.member_witness.same_joined_member(joined)
                                })
                            {
                                return None;
                            }
                        }
                    }
                    _ => return None,
                }
            }
            if remaining.len() + retired.len()
                != w.members()
                    .iter()
                    .filter(|(_, state)| *state == CollectionMemberWorkerState::Joined)
                    .count()
            {
                return None;
            }
            Some(())
        })()
        .is_some();
        if !valid {
            return reject(
                "scope roster or terminal evidence mismatch",
                physical,
                workers,
                remaining,
                retired,
            );
        }
        if retired.try_reserve(remaining.len()).is_err() {
            return reject(
                "retired actor storage allocation failed",
                physical,
                workers,
                remaining,
                retired,
            );
        }
        while let Some(token) = remaining.pop() {
            let p = physical.scope_outcome().expect("validated scope");
            let w = workers.as_ref().expect("validated workers").outcome();
            let state = token.member_witness().physical_state_in(p);
            let converted = match state {
                Some(CollectionMemberPhysicalState::PhysicallyConfirmed) => token
                    .try_into_physical_retirement()
                    .map(RoventTeacherRetiredActorV2::Physical),
                Some(
                    CollectionMemberPhysicalState::IssuedNoChild
                    | CollectionMemberPhysicalState::AcquiredNoChild,
                ) => token
                    .try_into_no_child_retirement(p, w)
                    .map(RoventTeacherRetiredActorV2::NoChild),
                _ => Err(token),
            };
            match converted {
                Ok(token) => retired.push(token),
                Err(token) => {
                    remaining.push(token);
                    return reject(
                        "actor custody retirement incomplete",
                        physical,
                        workers,
                        remaining,
                        retired,
                    );
                }
            }
        }
        Ok(Self {
            physical,
            workers,
            retired,
        })
    }
}
