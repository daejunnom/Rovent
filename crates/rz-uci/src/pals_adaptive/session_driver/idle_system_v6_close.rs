//! Borrowed close for a quiescent PALS system-CUDA session.
//!
//! This is an internal V6 building block. It does not close the process,
//! certify a GPU-wide fence, or turn a root/manager state into Idle.
use super::*;

/// The original driver remains borrowed by the caller on every result.
pub(crate) enum PalsV6IdleFinish {
    Complete(AdaptiveSessionProcessReceipt),
    Unavailable(SearchSessionFailure),
    Panicked(Box<dyn std::any::Any + Send>),
}

impl<B> AdaptiveSessionDriver<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    /// Finish only an already-idle PALS owner in its original mutex slot.
    /// Manager, active storage and profiled-retirement transitions are not
    /// emulated; their original owner remains untouched and is reported
    /// unknown. The caller must retain this driver and panic payload.
    pub(crate) fn finish_pals_system_v6_idle_borrowed(&self, until: Instant) -> PalsV6IdleFinish {
        if self.rovent.is_some() {
            return PalsV6IdleFinish::Unavailable(
                SearchSessionFailure::physical_completion_unknown(),
            );
        }
        if Instant::now() >= until {
            return PalsV6IdleFinish::Unavailable(
                SearchSessionFailure::physical_completion_unknown(),
            );
        }
        #[cfg(feature = "onnx-cpu")]
        if self.rovent_collection_hooks_v2.is_some() {
            return PalsV6IdleFinish::Unavailable(
                SearchSessionFailure::physical_completion_unknown(),
            );
        }
        #[cfg(feature = "onnx-cpu")]
        match self.storage.try_lock() {
            Ok(storage) if storage.is_none() => {}
            _ => {
                return PalsV6IdleFinish::Unavailable(
                    SearchSessionFailure::physical_completion_unknown(),
                );
            }
        }
        let mut slot = match self.state.try_lock() {
            Ok(slot) => slot,
            Err(_) => {
                return PalsV6IdleFinish::Unavailable(
                    SearchSessionFailure::physical_completion_unknown(),
                );
            }
        };
        if let SessionState::Closed(receipt) = &*slot {
            return if receipt.permits_process_close() {
                PalsV6IdleFinish::Complete(receipt.clone())
            } else {
                PalsV6IdleFinish::Unavailable(SearchSessionFailure::physical_completion_unknown())
            };
        }
        let SessionState::Idle(owners) = &mut *slot else {
            return PalsV6IdleFinish::Unavailable(
                SearchSessionFailure::physical_completion_unknown(),
            );
        };
        if owners.unexpected_archive_owner.is_some() {
            return PalsV6IdleFinish::Unavailable(
                SearchSessionFailure::physical_completion_unknown(),
            );
        }
        if owners.v6_finish_attempted {
            return PalsV6IdleFinish::Unavailable(
                SearchSessionFailure::physical_completion_unknown(),
            );
        }
        #[cfg(feature = "onnx-cpu")]
        if owners.profiled_retirement.is_some()
            || !matches!(&owners.researcher_source, SessionResearcherSource::Legacy)
        {
            return PalsV6IdleFinish::Unavailable(
                SearchSessionFailure::physical_completion_unknown(),
            );
        }
        owners.v6_finish_attempted = true;
        self.closed.store(true, Ordering::Release);
        let result = catch_unwind(AssertUnwindSafe(|| {
            self.finish_idle_physical_owners(owners, until)?;
            let engine = owners
                .engine
                .as_mut()
                .ok_or_else(SearchSessionFailure::physical_completion_unknown)?;
            let last_authority = engine.last_adaptive_authority();
            let archive = engine.finish_followup_owners(until).map_err(|error| {
                SearchSessionFailure::debug("PalsV6ArchiveFinish", &error).unknown_if_unclosed()
            })?;
            let archive_closed = archive
                .archive
                .as_ref()
                .is_ok_and(|item| item.as_ref().is_none_or(|value| value.admission_closed));
            let latest_root = self
                .latest_root
                .lock()
                .map_err(|_| SearchSessionFailure::physical_completion_unknown())?
                .clone();
            #[cfg(feature = "onnx-cpu")]
            let latest_profiled_root = self
                .latest_profiled_root
                .lock()
                .map_err(|_| SearchSessionFailure::physical_completion_unknown())?
                .clone();
            let native_session_release =
                if owners.native_joined && owners.native_finished == Some(true) {
                    owners.native_session_release.or_else(|| {
                        owners
                            .native_finish_report
                            .as_ref()
                            .and_then(|report| {
                                crate::pals_adaptive::NativeSessionReleaseObservation::from_report(
                                    report, true,
                                )
                            })
                            .or_else(|| {
                                owners.native_finish_snapshot.as_ref().and_then(|report| {
                                crate::pals_adaptive::NativeSessionReleaseObservation::from_report(
                                    report, false,
                                )
                            })
                            })
                    })
                } else {
                    None
                };
            let receipt = AdaptiveSessionProcessReceipt {
                epoch: self.clock.epoch(),
                last_authority,
                native_owner_joined: owners.native_joined,
                native_physical_shutdown_confirmed: owners.native_finished == Some(true),
                native_session_release,
                external_shutdown: owners.external_finished,
                store_archive_closed: archive_closed,
                latest_root,
                #[cfg(feature = "onnx-cpu")]
                latest_profiled_root,
            };
            if !receipt.permits_process_close() {
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
            Ok(receipt)
        }));
        match result {
            Ok(Ok(receipt)) => PalsV6IdleFinish::Complete(receipt),
            Ok(Err(error)) => PalsV6IdleFinish::Unavailable(error),
            Err(payload) => PalsV6IdleFinish::Panicked(payload),
        }
    }
}
