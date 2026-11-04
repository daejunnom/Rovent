//! Actual local refinement responsibilities. The semantic owner retains the
//! original Frontier/Store obligations; this manager only forwards its real
//! machine work and physically retired Fresh evidence.
use super::*;
use rz_search::pals::rovent_frontier::FrontierFrameKind;

pub(super) fn is_fresh_endpoint(stage: RefinementStage) -> bool {
    matches!(
        stage,
        RefinementStage::ProposalValueFresh
            | RefinementStage::CounterValueFresh
            | RefinementStage::RepairValueFresh
            | RefinementStage::FreshRepairedEndpoint
            | RefinementStage::FreshCounterEndpoint
            | RefinementStage::RecheckRepairedValueFresh
    )
}

impl<Q, B, D> AdaptiveExecutionManager<Q, B, D>
where
    Q: Clone + Send + Sync + 'static,
    B: TicketNativeBackend<ReviewView, Q>,
    B::Output: ProposerRankingOutput,
    D: AdaptiveSearchDriver<Q>,
{
    pub(super) fn rovent_candidate_ready(
        &mut self,
        frame: FrameId,
    ) -> Result<(), AdaptiveManagerError> {
        if self.driver.rovent_root_budget()?.is_none() {
            return Ok(());
        }
        let now = self.check_running()?;
        let work = self.machine.ready_work(frame, self.clock.domain(), now)?;
        self.driver.rovent_frame_admitted(
            &work,
            FrontierFrameKind::Candidate,
            self.clock.domain(),
            now,
        )?;
        Ok(())
    }

    /// Only the transition that actually selected a counter/repair line can
    /// register that lane. Continuing its sequence does not register another
    /// obligation or replace the original source question by a later view.
    pub(super) fn rovent_lane_after_resolve(
        &mut self,
        original: &CoordinatorWork<ReviewView, Q>,
    ) -> Result<(), AdaptiveManagerError> {
        if self.driver.rovent_root_budget()?.is_none() {
            return Ok(());
        }
        let Some(frame) = self.machine.frame(original.frame) else {
            return Ok(());
        };
        let selected_stage = frame.input.stage;
        let queued_recheck = frame.parent.is_some() && frame.parent == frame.supersedes;
        let kind = match (original.input.stage, selected_stage) {
            (
                RefinementStage::CriticReply,
                RefinementStage::CounterContinuation | RefinementStage::CounterValidation,
            ) => Some(FrontierFrameKind::Counter),
            (
                RefinementStage::Repair,
                RefinementStage::RepairValidation | RefinementStage::RepairValueFresh,
            ) => Some(FrontierFrameKind::Repair),
            (RefinementStage::CriticRecheck, RefinementStage::RecheckCounterValidation)
                if queued_recheck =>
            {
                Some(FrontierFrameKind::Counter)
            }
            _ => None,
        };
        let selected_lane = matches!(
            selected_stage,
            RefinementStage::CounterContinuation
                | RefinementStage::CounterValidation
                | RefinementStage::RepairValidation
                | RefinementStage::RepairValueFresh
        );
        if kind.is_none() && !selected_lane {
            return Ok(());
        }
        let now = self.check_running()?;
        let selected = self
            .machine
            .ready_work(original.frame, self.clock.domain(), now)?;
        if selected_lane
            && self
                .driver
                .rovent_lane_selected(original, &selected, self.clock.domain(), now)?
        {
            return Ok(());
        }
        if let Some(kind) = kind {
            self.driver
                .rovent_frame_admitted(&selected, kind, self.clock.domain(), now)?;
        }
        Ok(())
    }

    pub(super) fn rovent_seal_root_round(&mut self) -> Result<(), AdaptiveManagerError> {
        let Some(budget) = self.driver.rovent_root_budget()? else {
            return Ok(());
        };
        let now = self.check_running()?;
        self.driver
            .rovent_root_round_sealed(budget, self.clock.domain(), now)?;
        Ok(())
    }
}
