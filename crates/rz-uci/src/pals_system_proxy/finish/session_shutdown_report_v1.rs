//! A separate, session-scoped report from the original child lifecycle.
//! It is a received child report, not an independently observed GPU fence.
//! Existing session-release v1 frames and meanings are unchanged.

use super::*;
use crate::pals_adaptive_bootstrap::OriginalLifecycleSessionEvidence;

const DOMAIN: &str = "rz-pals-original-session-shutdown-report/1";
const ACK_DOMAIN: &str = "rz-pals-original-session-shutdown-report-ack/1";
const SCOPE: &str = "reported_original_session_only_not_device_fence";

/// Every positive field is a child report. A zero synchronous Run count is
/// valid and does not attest a GPU Run, placement, or another session.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReportedSessionShutdownV1 {
    domain: String,
    process_id: u32,
    process_generation: u64,
    startup_lock_sha256: String,
    native_owner_joined_reported: bool,
    native_shutdown_confirmed_reported: bool,
    finish_report_source: String,
    session_release_source: String,
    completed_synchronous_runs: u64,
    scope: String,
}

impl ReportedSessionShutdownV1 {
    /// These are child-reported session facts, not a GPU fence.
    pub(crate) fn process_id(&self) -> u32 {
        self.process_id
    }
    pub(crate) fn process_generation(&self) -> u64 {
        self.process_generation
    }
    pub(crate) fn startup_lock_sha256(&self) -> &str {
        &self.startup_lock_sha256
    }
    pub(crate) fn completed_synchronous_runs(&self) -> u64 {
        self.completed_synchronous_runs
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReportedSessionShutdownAckV1 {
    domain: String,
    process_id: u32,
    process_generation: u64,
    startup_lock_sha256: String,
    report_sha256: String,
    scope: String,
}

fn report_from_original(
    original: &OriginalLifecycleSessionEvidence<'_>,
) -> io::Result<ReportedSessionShutdownV1> {
    // Reuse the v1 producer's runtime, live-parent, receipt and original-report
    // checks, while requiring the stronger lifecycle-owned borrowed evidence.
    let original_event =
        event_from_original_process(original.source(), original.parent(), original.receipt())?;
    Ok(ReportedSessionShutdownV1 {
        domain: DOMAIN.into(),
        process_id: original_event.process_id,
        process_generation: original_event.process_generation,
        startup_lock_sha256: original_event.startup_lock_sha256,
        native_owner_joined_reported: true,
        native_shutdown_confirmed_reported: true,
        finish_report_source: "original_report".into(),
        session_release_source: "original_report".into(),
        completed_synchronous_runs: original_event.completed_synchronous_runs,
        scope: SCOPE.into(),
    })
}

fn report_matches(
    report: &ReportedSessionShutdownV1,
    pid: u32,
    generation: u64,
    lock_sha256: &str,
) -> bool {
    report.domain == DOMAIN
        && report.process_id == pid
        && report.process_generation == generation
        && report.startup_lock_sha256 == lock_sha256
        && report.native_owner_joined_reported
        && report.native_shutdown_confirmed_reported
        && report.finish_report_source == "original_report"
        && report.session_release_source == "original_report"
        && report.scope == SCOPE
}

fn acknowledgement(
    report: &ReportedSessionShutdownV1,
    bytes: &[u8],
) -> ReportedSessionShutdownAckV1 {
    ReportedSessionShutdownAckV1 {
        domain: ACK_DOMAIN.into(),
        process_id: report.process_id,
        process_generation: report.process_generation,
        startup_lock_sha256: report.startup_lock_sha256.clone(),
        report_sha256: format!("{:x}", Sha256::digest(bytes)),
        scope: "report_received_not_physical_fence".into(),
    }
}

fn ack_matches(
    ack: &ReportedSessionShutdownAckV1,
    report: &ReportedSessionShutdownV1,
    bytes: &[u8],
) -> bool {
    let expected = acknowledgement(report, bytes);
    ack.domain == expected.domain
        && ack.process_id == expected.process_id
        && ack.process_generation == expected.process_generation
        && ack.startup_lock_sha256 == expected.startup_lock_sha256
        && ack.report_sha256 == expected.report_sha256
        && ack.scope == expected.scope
}

/// The caller retains the same original stream on every outcome. The clone is
/// only a temporary handle to the same socket for existing authentication;
/// the caller must ensure no competing reader/writer on that description.
/// Authentication can change shared blocking mode and socket timeouts, so
/// retaining the FD does not imply that its transport state is unchanged.
/// Authentication errors may include partial handshake I/O: the retained
/// stream is not a fresh connection and the event offset of zero does not
/// describe the handshake. The caller must not retry it as a new exchange.
pub(crate) fn send_report(
    original: &OriginalLifecycleSessionEvidence<'_>,
    stream: &mut UnixStream,
    until: Instant,
    cancelled: &AtomicBool,
) -> Result<ReportedSessionShutdownAckV1, PalsReleaseSendFailureV6> {
    let failure =
        |stage, error: io::Error, event_bytes: Vec<u8>, written, acknowledgement_bytes| {
            PalsReleaseSendFailureV6 {
                reason: error.to_string(),
                stage,
                io_error: Some(error),
                event_bytes,
                event_frame_bytes_written: written,
                acknowledgement_bytes,
            }
        };
    let report = report_from_original(original).map_err(|error| {
        failure(
            PalsReleaseSendFailureStageV6::Prepare,
            error,
            Vec::new(),
            0,
            Vec::new(),
        )
    })?;
    let bytes = serde_json::to_vec(&report).map_err(|error| {
        failure(
            PalsReleaseSendFailureStageV6::Prepare,
            io::Error::other(error),
            Vec::new(),
            0,
            Vec::new(),
        )
    })?;
    if bytes.len() >= MAX_FINISH_FRAME {
        return Err(failure(
            PalsReleaseSendFailureStageV6::Prepare,
            io::Error::other("session shutdown report exceeds bound"),
            bytes,
            0,
            Vec::new(),
        ));
    }
    let cloned = stream.try_clone().map_err(|error| {
        failure(
            PalsReleaseSendFailureStageV6::Authenticate,
            error,
            bytes.clone(),
            0,
            Vec::new(),
        )
    })?;
    let mut control = original
        .parent()
        .authenticate_separate_control_with_cancel(cloned, until, cancelled)
        .map_err(|error| {
            failure(
                PalsReleaseSendFailureStageV6::Authenticate,
                error,
                bytes.clone(),
                0,
                Vec::new(),
            )
        })?;
    let mut frame = bytes.clone();
    frame.push(b'\n');
    if let Err((written, error)) = write_frame(&mut control, &frame, until, cancelled) {
        return Err(failure(
            PalsReleaseSendFailureStageV6::SendFrame,
            error,
            bytes,
            written,
            Vec::new(),
        ));
    }
    let ack_bytes = match read_frame_detailed(&mut control, until, cancelled) {
        Ok(bytes) => bytes,
        Err((partial, error)) => {
            return Err(failure(
                PalsReleaseSendFailureStageV6::ReadAcknowledgement,
                error,
                bytes,
                frame.len(),
                partial,
            ));
        }
    };
    let ack: ReportedSessionShutdownAckV1 =
        serde_json::from_slice(&ack_bytes).map_err(|error| {
            failure(
                PalsReleaseSendFailureStageV6::ValidateAcknowledgement,
                io::Error::other(error),
                bytes.clone(),
                frame.len(),
                ack_bytes.clone(),
            )
        })?;
    if cancelled.load(Ordering::Acquire) || Instant::now() >= until {
        let kind = if cancelled.load(Ordering::Acquire) {
            io::ErrorKind::Interrupted
        } else {
            io::ErrorKind::TimedOut
        };
        return Err(failure(
            PalsReleaseSendFailureStageV6::ValidateAcknowledgement,
            io::Error::new(
                kind,
                "session shutdown acknowledgement expired or cancelled",
            ),
            bytes,
            frame.len(),
            ack_bytes,
        ));
    }
    if !ack_matches(&ack, &report, &bytes) {
        return Err(failure(
            PalsReleaseSendFailureStageV6::ValidateAcknowledgement,
            io::Error::other("session shutdown report acknowledgement mismatch"),
            bytes,
            frame.len(),
            ack_bytes,
        ));
    }
    Ok(ack)
}

/// The same child and original authenticated finish stream remain with their
/// owner. Even the Reported variant is not a physical release capability.
pub(in crate::pals_system_proxy) enum ObservedSessionShutdownV1 {
    Reported {
        bytes: Vec<u8>,
        report: ReportedSessionShutdownV1,
        acknowledgement: Result<(), PalsReleaseAckWriteFailureV6>,
    },
    Invalid {
        bytes: Vec<u8>,
        reason: String,
    },
    Incomplete {
        bytes: Vec<u8>,
        reason: String,
    },
}

pub(in crate::pals_system_proxy) fn observe_report(
    owner: &mut DirectPalsChildV6,
    stream: &mut UnixStream,
    until: Instant,
    cancelled: &AtomicBool,
) -> io::Result<ObservedSessionShutdownV1> {
    if !owner.image_after_auth_verified
        || !matches!(
            owner.image_grant_observation(),
            crate::pals_system_proxy::PalsImageGrantObservationV6::Acknowledged
        )
    {
        return Err(io::Error::other(
            "original child image admission unavailable",
        ));
    }
    require_original_child_peer(owner, stream)?;
    let bytes = match read_frame(stream, until, cancelled) {
        Ok(bytes) => bytes,
        Err((bytes, reason)) => return Ok(ObservedSessionShutdownV1::Incomplete { bytes, reason }),
    };
    if let Err(error) = require_original_child_peer(owner, stream) {
        return Ok(ObservedSessionShutdownV1::Incomplete {
            bytes,
            reason: error.to_string(),
        });
    }
    let report: ReportedSessionShutdownV1 = match serde_json::from_slice(&bytes) {
        Ok(report) => report,
        Err(error) => {
            return Ok(ObservedSessionShutdownV1::Invalid {
                bytes,
                reason: error.to_string(),
            });
        }
    };
    if !report_matches(
        &report,
        owner.child.id(),
        owner.lock.config.process_generation,
        &owner.lock.canonical_sha256,
    ) {
        return Ok(ObservedSessionShutdownV1::Invalid {
            bytes,
            reason: "reported session shutdown differs from original child".into(),
        });
    }
    let acknowledgement = match serde_json::to_vec(&acknowledgement(&report, &bytes)) {
        Ok(mut ack) if ack.len() < MAX_FINISH_FRAME => {
            ack.push(b'\n');
            write_frame(stream, &ack, until, cancelled).map_err(|(bytes_written, error)| {
                PalsReleaseAckWriteFailureV6 {
                    bytes: ack,
                    bytes_written,
                    reason: error.to_string(),
                    io_error: Some(error),
                }
            })
        }
        Ok(ack) => Err(PalsReleaseAckWriteFailureV6 {
            bytes: ack,
            bytes_written: 0,
            reason: "session shutdown acknowledgement exceeds bound".into(),
            io_error: None,
        }),
        Err(error) => Err(PalsReleaseAckWriteFailureV6 {
            bytes: Vec::new(),
            bytes_written: 0,
            reason: error.to_string(),
            io_error: Some(io::Error::other(error)),
        }),
    };
    Ok(ObservedSessionShutdownV1::Reported {
        bytes,
        report,
        acknowledgement,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_and_ack_bind_the_child_scope_without_claiming_a_device_fence() {
        let pid = std::process::id();
        let lock = format!("{:x}", Sha256::digest(b"synthetic session test lock"));
        let mut report = ReportedSessionShutdownV1 {
            domain: DOMAIN.into(),
            process_id: pid,
            process_generation: 7,
            startup_lock_sha256: lock.clone(),
            native_owner_joined_reported: true,
            native_shutdown_confirmed_reported: true,
            finish_report_source: "original_report".into(),
            session_release_source: "original_report".into(),
            completed_synchronous_runs: 0,
            scope: SCOPE.into(),
        };
        let bytes = serde_json::to_vec(&report).unwrap();
        let ack = acknowledgement(&report, &bytes);
        assert!(report_matches(&report, pid, 7, &lock));
        assert!(ack_matches(&ack, &report, &bytes));
        assert!(!ack_matches(&ack, &report, b"different report bytes"));
        assert!(!report_matches(&report, pid, 8, &lock));
        report.native_shutdown_confirmed_reported = false;
        assert!(!report_matches(&report, pid, 7, &lock));
    }
}
