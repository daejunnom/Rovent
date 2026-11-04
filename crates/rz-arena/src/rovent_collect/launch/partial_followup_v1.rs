//! Dependency-scoped continuation for a verified partial collection.
//!
//! The caller supplies independently observed process and resource states.
//! This policy does not establish a process fence, GPU completion, dataset
//! authority, or permission to reuse an in-flight session.
use super::partial_collection_v1::PartialCollectionReader;

/// Exit 0 retains its existing strict-success meaning. Exit 2 is reserved by
/// the opt-in partial CLI for a separately verified incomplete receipt.
pub const PARTIAL_COLLECTION_EXIT_CODE: i32 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProducerExit {
    Exited(i32),
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PriorResourceState {
    PhysicalCompletionObserved,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessTreeState {
    GoneObserved,
    Unknown,
}

/// `Isolated` must come from the caller's actual resource admission, not the
/// partial receipt or the absence of an observed error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FollowupResources {
    Isolated,
    ReusesPrior,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FollowupDependency {
    Independent,
    PartialInspection,
    StrictCorpus,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FollowupDecision {
    Run,
    BlockedSharedResource,
    BlockedPartialEvidence,
    BlockedStrictCorpus,
}

/// A reader proves only the partial domain, bytes and row seals. Its data is
/// accepted solely for inspection and cannot authorize model training.
pub fn decide_partial_followup(
    dependency: FollowupDependency,
    resources: FollowupResources,
    physical: PriorResourceState,
    producer_exit: ProducerExit,
    process_tree: ProcessTreeState,
    partial: Option<&PartialCollectionReader>,
) -> FollowupDecision {
    if resources == FollowupResources::ReusesPrior
        && (physical != PriorResourceState::PhysicalCompletionObserved
            || process_tree != ProcessTreeState::GoneObserved)
    {
        return FollowupDecision::BlockedSharedResource;
    }
    match dependency {
        FollowupDependency::Independent => FollowupDecision::Run,
        FollowupDependency::StrictCorpus => FollowupDecision::BlockedStrictCorpus,
        FollowupDependency::PartialInspection => {
            if process_tree == ProcessTreeState::GoneObserved
                && producer_exit == ProducerExit::Exited(PARTIAL_COLLECTION_EXIT_CODE)
                && partial.is_some_and(|reader| {
                    reader.chunk_count() > 0
                        && reader
                            .verify_failure_evidence_current()
                            .is_ok_and(|valid| valid)
                })
            {
                FollowupDecision::Run
            } else {
                FollowupDecision::BlockedPartialEvidence
            }
        }
    }
}

/// Execute exactly one task only when its own dependency and resource
/// admission allow it. A previous failure never becomes successful here.
pub fn run_partial_followup<T>(
    dependency: FollowupDependency,
    resources: FollowupResources,
    physical: PriorResourceState,
    producer_exit: ProducerExit,
    process_tree: ProcessTreeState,
    partial: Option<&PartialCollectionReader>,
    task: impl FnOnce() -> T,
) -> Result<T, FollowupDecision> {
    let decision = decide_partial_followup(
        dependency,
        resources,
        physical,
        producer_exit,
        process_tree,
        partial,
    );
    if decision == FollowupDecision::Run {
        Ok(task())
    } else {
        Err(decision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_isolated_work_survives_failed_or_unknown_producer() {
        let decision = decide_partial_followup(
            FollowupDependency::Independent,
            FollowupResources::Isolated,
            PriorResourceState::Unknown,
            ProducerExit::Unknown,
            ProcessTreeState::Unknown,
            None,
        );
        assert_eq!(decision, FollowupDecision::Run);
        assert_eq!(
            run_partial_followup(
                FollowupDependency::Independent,
                FollowupResources::Isolated,
                PriorResourceState::Unknown,
                ProducerExit::Unknown,
                ProcessTreeState::Unknown,
                None,
                || 17,
            ),
            Ok(17)
        );
    }

    #[test]
    fn unknown_physical_state_blocks_shared_resources_and_strict_claims() {
        let decision = decide_partial_followup(
            FollowupDependency::Independent,
            FollowupResources::ReusesPrior,
            PriorResourceState::Unknown,
            ProducerExit::Exited(PARTIAL_COLLECTION_EXIT_CODE),
            ProcessTreeState::GoneObserved,
            None,
        );
        assert_eq!(decision, FollowupDecision::BlockedSharedResource);
        assert_eq!(
            decide_partial_followup(
                FollowupDependency::Independent,
                FollowupResources::ReusesPrior,
                PriorResourceState::PhysicalCompletionObserved,
                ProducerExit::Exited(PARTIAL_COLLECTION_EXIT_CODE),
                ProcessTreeState::Unknown,
                None,
            ),
            FollowupDecision::BlockedSharedResource
        );
        assert!(
            run_partial_followup(
                FollowupDependency::Independent,
                FollowupResources::ReusesPrior,
                PriorResourceState::Unknown,
                ProducerExit::Exited(PARTIAL_COLLECTION_EXIT_CODE),
                ProcessTreeState::GoneObserved,
                None,
                || panic!("blocked task ran"),
            )
            .is_err()
        );
        assert_eq!(
            decide_partial_followup(
                FollowupDependency::StrictCorpus,
                FollowupResources::Isolated,
                PriorResourceState::Unknown,
                ProducerExit::Exited(PARTIAL_COLLECTION_EXIT_CODE),
                ProcessTreeState::GoneObserved,
                None,
            ),
            FollowupDecision::BlockedStrictCorpus
        );
        assert_eq!(
            decide_partial_followup(
                FollowupDependency::PartialInspection,
                FollowupResources::Isolated,
                PriorResourceState::Unknown,
                ProducerExit::Exited(PARTIAL_COLLECTION_EXIT_CODE),
                ProcessTreeState::GoneObserved,
                None,
            ),
            FollowupDecision::BlockedPartialEvidence
        );
    }
}
