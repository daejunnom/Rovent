//! Common physical process closure also applies while a Store/evidence owner
//! remains in the archive worker. Durable I/O failure cannot skip GPU/pipe join.

use super::*;

impl<B> AdaptiveSessionDriver<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    pub(super) fn finish_idle_physical_owners(
        &self,
        owners: &mut IdleOwners<B>,
        until: Instant,
    ) -> Result<(), SearchSessionFailure> {
        owners.native.begin_shutdown(until);
        if let Some(external) = &owners.external {
            external.worker.request_shutdown(until);
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        match &mut owners.researcher_source {
            SessionResearcherSource::ProfiledExternal(source) => source.begin_process_finish(until),
            SessionResearcherSource::ProfiledExternalV2(source) => {
                source.begin_process_finish(until)
            }
            SessionResearcherSource::Legacy => {}
        }
        loop {
            let version = owners.native.signal().version();
            while let Some(event) = owners.native.poll() {
                match event {
                    TicketNativeEvent::Finished {
                        physical_shutdown_confirmed,
                    } => {
                        owners.native_finished = Some(physical_shutdown_confirmed);
                    }
                    TicketNativeEvent::Ready { .. } | TicketNativeEvent::RootQuiesced(_) => {}
                    _ => return Err(SearchSessionFailure::physical_completion_unknown()),
                }
            }
            if !owners.native_joined {
                owners.native_joined = owners
                    .native
                    .try_join()
                    .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
            }
            if owners.external_finished.is_none() {
                if let Some(external) = &mut owners.external {
                    if let Some(result) = external.worker.try_join() {
                        owners.external_finished =
                            Some(result.map_err(|_| {
                                SearchSessionFailure::physical_completion_unknown()
                            })?);
                    }
                }
            }
            let profiled_complete = {
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                {
                    match &mut owners.researcher_source {
                        SessionResearcherSource::Legacy => true,
                        SessionResearcherSource::ProfiledExternal(source) => {
                            let complete = source
                                .try_process_finish()
                                .map_err(SearchSessionFailure::unknown_if_unclosed)?;
                            owners.external_finished = source.shutdown_receipt();
                            complete
                        }
                        SessionResearcherSource::ProfiledExternalV2(source) => {
                            let complete = source
                                .try_process_finish()
                                .map_err(SearchSessionFailure::unknown_if_unclosed)?;
                            owners.external_finished = source.shutdown_receipt();
                            complete
                        }
                    }
                }
                #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
                {
                    true
                }
            };
            let external_complete = profiled_complete
                && (owners.external.is_none()
                    || owners.external_finished.is_some_and(|receipt| {
                        receipt.cleanup_complete && !receipt.quarantined && !receipt.ownership_lost
                    }));
            if owners.native_joined && owners.native_finished == Some(true) && external_complete {
                return Ok(());
            }
            if Instant::now() >= until || owners.native_finished == Some(false) {
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
            owners
                .native
                .signal()
                .wait_changed(version, Duration::from_millis(1));
        }
    }
}
