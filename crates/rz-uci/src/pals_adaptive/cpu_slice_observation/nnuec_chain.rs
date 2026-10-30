//! Original accepted-C/child and acknowledged-NNUEC/Job scalar observations.
//! No model, view, Bound, token, output tensor or release authority is retained.
use super::*;
use rz_contracts::pals_adaptive::RoleTicketEnvelope;
use rz_contracts::rovent::QuestionEnvelope;

#[derive(Clone, Copy, Debug)]
pub struct AcceptedCriticChildObservation {
    pub source_frame: FrameId,
    pub ticket: RoleTicketEnvelope,
    pub output: Digest,
    pub origin_seal: Digest,
    pub question: QuestionEnvelope,
    pub child: FrameId,
    pub joined: bool,
    pub actual_child_view: Option<ReviewViewSeal>,
    pub actual_child_public_context: Option<Digest>,
    /// Original Observation::Model.input, not a later record scan.
    pub published_record_revision: Option<u64>,
    /// Index in the actual admitted/joined child's immutable selected view.
    pub selected_record_index: Option<u16>,
    pub selected_record_count: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
pub struct NnuecJobBindingObservation {
    pub frame: FrameId,
    pub question: Digest,
    pub view: ReviewViewSeal,
    pub full_view: Digest,
    pub job: CpuJobEnvelope,
    pub first_grant: SliceGrant,
    pub installation_sequence: u64,
    pub installation_succeeded: bool,
    pub binding: Digest,
    pub installed_owner: Option<rz_contracts::pals_adaptive::CpuWorkerId>,
    pub actual_stack_owner_id: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum NnuecChainEvent {
    AcceptedCriticChild(AcceptedCriticChildObservation),
    /// Successful original begin enqueue AND exact machine frame binding.
    /// This is not an actual worker invocation, readout or resume ACK.
    NnuecJobBound(NnuecJobBindingObservation),
}
impl NnuecChainEvent {
    fn authority(self) -> SearchAuthority {
        match self {
            Self::AcceptedCriticChild(value) => value.ticket.authority,
            Self::NnuecJobBound(value) => value.job.id.authority,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NnuecChainObservation {
    pub sequence: u64,
    pub event: NnuecChainEvent,
}

impl AdaptiveCpuSliceDiagnostic {
    pub(crate) fn record_nnuec_chain(&self, event: NnuecChainEvent) {
        // try_lock and bounded preallocation keep this observation from adding
        // a blocking manager wait, a retry, a product failure or new work.
        let Ok(mut state) = self.inner.state.try_lock() else {
            self.drop_nnuec_chain();
            return;
        };
        if state.authority != Some(event.authority())
            || state.records.len().saturating_add(state.nnuec_chain.len()) >= self.inner.records_max
        {
            self.drop_nnuec_chain();
            return;
        }
        let sequence = state.nnuec_chain.len() as u64 + 1;
        state
            .nnuec_chain
            .push(NnuecChainObservation { sequence, event });
    }
    fn drop_nnuec_chain(&self) {
        let _ = self.inner.nnuec_chain_dropped.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |value| Some(value.saturating_add(1)),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_chain_records_do_not_change_the_original_slice_failure() {
        let journal =
            AdaptiveCpuSliceDiagnostic::new(AdaptiveCpuSliceDiagnosticOptions { records_max: 1 })
                .unwrap();
        // A missing original event cannot be replaced by an authored success.
        journal.drop_nnuec_chain();
        let snapshot = journal.snapshot().unwrap();
        assert!(snapshot.nnuec_chain.is_empty());
        assert_eq!(snapshot.nnuec_chain_dropped, 1);
        assert!(snapshot.authority.is_none());
        assert!(snapshot.unavailable.is_none());
    }
}
