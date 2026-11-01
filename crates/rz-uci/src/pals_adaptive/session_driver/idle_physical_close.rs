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
            SessionResearcherSource::ProfiledExternalV3(source) => {
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
                        if owners.native_finished.is_some() {
                            return Err(SearchSessionFailure::physical_completion_unknown());
                        }
                        owners.native_finished = Some(physical_shutdown_confirmed);
                        owners.native_finish_snapshot = owners.native.finish_report_snapshot();
                        owners.native_finish_report = owners.native.take_finish_report();
                    }
                    TicketNativeEvent::Ready { .. } | TicketNativeEvent::RootQuiesced(_) => {}
                    _ => return Err(SearchSessionFailure::physical_completion_unknown()),
                }
            }
            // A process-closed manager may have consumed Finished before it
            // transferred this service to IdleOwners. The sticky snapshot is
            // still the same producer observation, never a new completion.
            if owners.native_finished.is_some() && owners.native_finish_snapshot.is_none() {
                owners.native_finish_snapshot = owners.native.finish_report_snapshot();
            }
            let requires = owners.native.requires_finish_report();
            let snapshot = owners.native_finish_snapshot.as_ref();
            let transferred = owners.native_finish_report.as_ref();
            let mismatch = owners.native_finished.is_some() && (
                (requires && snapshot.is_none())
                || (snapshot.is_none() && transferred.is_some())
                || (!requires && snapshot.is_some())
                || snapshot.zip(transferred).is_some_and(|(s, t)| s != t)
                || snapshot.is_some_and(|report|
                    owners.native_finished != Some(report.physical_shutdown_confirmed))
            );
            if mismatch {
                return Err(SearchSessionFailure::debug(
                    "NativeShutdownMismatch",
                    &(owners.native_finished, snapshot, transferred),
                ).unknown_if_unclosed());
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
                        SessionResearcherSource::ProfiledExternalV3(source) => {
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
                if let Some(report) = &owners.native_finish_snapshot {
                    if report.physical_failure.is_some() || report.semantic_failure.is_some() {
                        return Err(SearchSessionFailure::debug("NativeShutdown", report));
                    }
                }
                return Ok(());
            }
            if Instant::now() >= until || owners.native_finished == Some(false) {
                return Err(match &owners.native_finish_snapshot {
                    Some(report) if report.physical_failure.is_some() || report.semantic_failure.is_some() => {
                        SearchSessionFailure::debug("NativeShutdown", report).unknown_if_unclosed()
                    }
                    _ => SearchSessionFailure::physical_completion_unknown(),
                });
            }
            owners
                .native
                .signal()
                .wait_changed(version, Duration::from_millis(1));
        }
    }
}
