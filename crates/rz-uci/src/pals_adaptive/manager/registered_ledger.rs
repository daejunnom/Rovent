//! Move the preparing root's actual ledger into its ready native manager.
//! Rejection returns that same owner and every original process source.
use super::*;
use rz_search::cpu_worker::CpuLedgerRegistrationObservation;

pub struct AdaptiveRegisteredLedgerSources<Q, B, D>
where
    B: TicketNativeBackend<ReviewView, Q>,
{
    pub sources: AdaptiveManagerSources<Q, B, D>,
    pub ledger: CpuCreditLedger,
    pub root_budget: Digest,
    pub cleanup_deadline: Instant,
}

pub struct AdaptiveRegisteredLedgerAdmissionRejection<Q, B, D>
where
    B: TicketNativeBackend<ReviewView, Q>,
{
    pub error: AdaptiveManagerError,
    pub original: AdaptiveRegisteredLedgerSources<Q, B, D>,
}

pub(super) struct RegisteredLedger {
    pub ledger: CpuCreditLedger,
    pub root_budget: Digest,
    pub cleanup_deadline: Instant,
}

pub(super) struct SourceAdmissionRejection<Q, B, D>
where
    B: TicketNativeBackend<ReviewView, Q>,
{
    pub error: AdaptiveManagerError,
    pub sources: AdaptiveManagerSources<Q, B, D>,
    pub original: Option<RegisteredLedger>,
}

pub(super) fn validate_registered_ledger<Q, D: AdaptiveSearchDriver<Q>>(
    registration: &AdaptiveManagerRegistration,
    clock: &ContractSystemClock,
    driver: &D,
    original: &RegisteredLedger,
) -> Result<(CpuLedgerRegistrationObservation, CleanupBoundary), AdaptiveManagerError> {
    let observed = original.ledger.registration_observation()?;
    if observed.authority != registration.authority
        || observed.clock_domain != clock.domain()
        || observed.clock_origin != clock.origin()
        || observed.native_node_limit != registration.native_cpu_node_limit
        || observed.active_jobs != 0
        || observed.outstanding_executions != 0
        || observed.reserved_native_nodes != 0
        || observed.quarantined_jobs != 0
        || observed.highest_job_sequence.checked_add(1).is_none()
        || observed.highest_execution_sequence.checked_add(1).is_none()
    {
        return Err(AdaptiveManagerError::Invalid(
            "original registered native ledger scope/known usage",
        ));
    }
    let cleanup = CleanupBoundary::admit(registration.hard_deadline, original.cleanup_deadline)?;
    driver.validate_registered_native_ledger(
        &original.ledger.identity(),
        &observed,
        original.root_budget,
        registration.hard_deadline,
    )?;
    Ok((observed, cleanup))
}

impl<Q, B, D> AdaptiveExecutionManager<Q, B, D>
where
    Q: Clone + Send + Sync + 'static,
    B: TicketNativeBackend<ReviewView, Q>,
    B::Output: ProposerRankingOutput,
    D: AdaptiveSearchDriver<Q>,
{
    #[allow(clippy::result_large_err)]
    pub fn new_retaining_with_registered_ledger(
        registration: AdaptiveManagerRegistration,
        clock: ContractSystemClock,
        limits: AdaptiveExecutionLimits,
        original: AdaptiveRegisteredLedgerSources<Q, B, D>,
    ) -> Result<Self, AdaptiveRegisteredLedgerAdmissionRejection<Q, B, D>> {
        let AdaptiveRegisteredLedgerSources {
            sources,
            ledger,
            root_budget,
            cleanup_deadline,
        } = original;
        match Self::admit_sources_with_ledger(
            registration,
            clock,
            limits,
            sources,
            true,
            Some(RegisteredLedger {
                ledger,
                root_budget,
                cleanup_deadline,
            }),
        ) {
            Ok(manager) => Ok(manager),
            Err(rejected) => {
                let original = rejected
                    .original
                    .expect("registered admission retains its original linear ledger");
                Err(AdaptiveRegisteredLedgerAdmissionRejection {
                    error: rejected.error,
                    original: AdaptiveRegisteredLedgerSources {
                        sources: rejected.sources,
                        ledger: original.ledger,
                        root_budget: original.root_budget,
                        cleanup_deadline: original.cleanup_deadline,
                    },
                })
            }
        }
    }
}
