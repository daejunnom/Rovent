//! Caller-owned admission for the guarded PALS V6 path.
use super::*;

/// A panic does not establish shutdown. A missing source slot marks custody
/// unknown, including the unlikely event of an unwind after transfer.
#[must_use]
pub(crate) enum PalsV6Admission<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    Admitted(AdaptiveSessionDriver<B>),
    Rejected(SearchSessionFailure),
    Panicked {
        payload: Box<dyn std::any::Any + Send>,
        source_retained: bool,
    },
}

impl<B> AdaptiveSessionDriver<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    /// The caller owns `sources` throughout validation and retains it after
    /// rejection or an unwind before commit. Do not retry a panicked attempt.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn admit_pals_system_v6_in_slot(
        options: AdaptiveSessionOptions,
        limits: AdaptiveExecutionLimits,
        sources: &mut Option<AdaptiveSessionSources<B>>,
        clock: ProcessClock,
        issuer: AdaptiveRulesOwnerIssuer,
        registration: PalsCheckerRegistration,
        dispatch_observer: Option<AdaptiveCollectionDispatchObserver>,
        closure_observer: Option<AdaptiveRootClosureObserver<B>>,
        collection_hooks: AdaptiveSessionCollectionHooks<B>,
    ) -> PalsV6Admission<B> {
        let mut researcher = Some(SessionResearcherSource::Legacy);
        let attempt = catch_unwind(AssertUnwindSafe(|| {
            Self::admit_with_researcher_source_in_slots(
                options,
                limits,
                sources,
                clock,
                issuer,
                registration,
                dispatch_observer,
                closure_observer,
                collection_hooks,
                true,
                &mut researcher,
            )
        }));
        match attempt {
            Ok(Ok(driver)) => PalsV6Admission::Admitted(driver),
            Ok(Err(error)) => PalsV6Admission::Rejected(error),
            Err(payload) => PalsV6Admission::Panicked {
                payload,
                source_retained: sources.is_some(),
            },
        }
    }
}
