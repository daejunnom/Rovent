//! Small readback for moving the original ledger into its actual manager.
//! These scalar observations neither construct a ledger nor attest work.
use super::*;
use rz_contracts::ClockDomain;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuLedgerRegistrationObservation {
    pub authority: SearchAuthority,
    pub clock_domain: ClockDomain,
    pub clock_origin: Instant,
    pub native_node_limit: u64,
    pub spent_native_nodes: u64,
    pub reserved_native_nodes: u64,
    pub active_jobs: usize,
    pub outstanding_executions: usize,
    pub quarantined_jobs: usize,
    pub highest_job_sequence: u64,
    pub highest_execution_sequence: u64,
}

impl CpuCreditLedger {
    /// A caller must still own and move this same non-Clone ledger and match
    /// its opaque identity against the original attached root controller.
    pub fn registration_observation(
        &self,
    ) -> Result<CpuLedgerRegistrationObservation, CpuWorkerError> {
        let (spent_native_nodes, reserved_native_nodes) = self.global_usage()?;
        if spent_native_nodes
            .checked_add(reserved_native_nodes)
            .is_none_or(|used| used > self.global_limit)
        {
            return Err(CpuWorkerError::Invalid(
                "registered native ledger exceeds original credit",
            ));
        }
        Ok(CpuLedgerRegistrationObservation {
            authority: self.authority,
            clock_domain: self.clock.domain(),
            clock_origin: self.clock.origin(),
            native_node_limit: self.global_limit,
            spent_native_nodes,
            reserved_native_nodes,
            active_jobs: self
                .jobs
                .values()
                .filter(|account| !account.terminal)
                .count(),
            outstanding_executions: self
                .jobs
                .values()
                .filter(|account| account.reservation.is_some())
                .count(),
            quarantined_jobs: self
                .jobs
                .values()
                .filter(|account| account.quarantine)
                .count(),
            highest_job_sequence: self.highest_job_sequence,
            highest_execution_sequence: self.highest_execution_sequence,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::{GameGeneration, ProcessEpoch, RootGeneration};

    #[test]
    fn original_ledger_move_keeps_identity_clock_credit_and_high_water() {
        let authority = SearchAuthority {
            epoch: ProcessEpoch(83),
            game: GameGeneration(1),
            root: RootGeneration(2),
            implementation: Digest([83; 32]),
        };
        let origin = Instant::now();
        let mut ledger = CpuCreditLedger::new(
            authority,
            4096,
            InstantClock::new(ClockDomain(authority.epoch), origin),
        )
        .unwrap();
        // These private states are authored solely to verify the readback;
        // they are not settlement receipts or worker execution evidence.
        ledger.global_spent = 31;
        ledger.highest_job_sequence = 7;
        ledger.highest_execution_sequence = 9;
        let identity = ledger.identity();
        let observed = ledger.registration_observation().unwrap();
        let moved = ledger;
        assert_eq!(moved.identity(), identity);
        assert_eq!(moved.registration_observation().unwrap(), observed);
        assert_eq!(observed.clock_origin, origin);
        assert_eq!(observed.native_node_limit, 4096);
        assert_eq!(observed.spent_native_nodes, 31);
        assert_eq!(observed.highest_job_sequence, 7);
        assert_eq!(observed.highest_execution_sequence, 9);
        assert_eq!(observed.active_jobs, 0);
        assert_eq!(observed.outstanding_executions, 0);
    }
}
