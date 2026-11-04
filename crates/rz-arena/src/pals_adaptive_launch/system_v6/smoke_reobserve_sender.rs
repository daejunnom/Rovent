//! Read-only bounded observation on the same completed NoGame Run stream.
//! This neither issues a new Run/cleanup grant nor proves physical completion.

use super::*;
use rz_contracts::pals::proxy_reobserve_v1::{
    FRAME_HEADER_BYTES, ProxyReobserveRequestV1, ProxyReobserveResponseV1, frame_length,
};

pub(super) struct ReobserveAttempt {
    request: ProxyReobserveRequestV1,
    until: Instant,
    bytes: Vec<u8>,
    written: usize,
    received: Vec<u8>,
    response: Option<ProxyReobserveResponseV1>,
    first_timeout: Option<Arc<ArenaError>>,
    timed_out: bool,
    terminal_failure: Option<Arc<ArenaError>>,
    io_error: Option<io::Error>,
}

/// Original response and transport bytes, including any earlier wait timeout.
pub(crate) struct SmokeReobserveObservation {
    pub request: ProxyReobserveRequestV1,
    pub response: ProxyReobserveResponseV1,
    pub request_bytes: Vec<u8>,
    pub written: usize,
    pub response_bytes: Vec<u8>,
    pub first_timeout: Option<Arc<ArenaError>>,
}

pub(crate) enum SmokeReobserveProgress {
    Pending { written: usize, received: usize },
    Observed,
    TimedOut,
    Failed,
}

impl SmokeDeadlineControl {
    /// Begin one observation on the original Run stream. `until` only bounds
    /// observation; no original execution or cleanup clock is modified.
    pub(crate) fn begin_no_game_reobserve(
        &mut self,
        custody: &SpawnCustody,
        until: Instant,
    ) -> Result<(), ArenaError> {
        if !matches!(&self.binding, SmokeBinding::NoGame { .. }) || self.reobserve.is_some() {
            return Err(failure("smoke_reobserve_unavailable"));
        }
        let remaining = until
            .checked_duration_since(Instant::now())
            .ok_or_else(|| failure("smoke_reobserve_deadline"))?;
        let budget =
            u64::try_from(remaining.as_millis()).map_err(|_| failure("smoke_reobserve_budget"))?;
        if budget == 0 {
            return Err(failure("smoke_reobserve_budget"));
        }
        let sequence = self
            .reobserve_sequence
            .checked_add(1)
            .ok_or_else(|| failure("smoke_reobserve_sequence"))?;
        let request = ProxyReobserveRequestV1 {
            scope: self.scopes[0].clone(),
            sequence,
            observation_budget_ms: budget,
        };
        let bytes = request.encode().map_err(failure)?;
        let (pid, identity) = self
            .child
            .ok_or_else(|| failure("smoke_reobserve_child_unavailable"))?;
        let peer = self
            .retained_grants
            .iter()
            .find(|peer| {
                peer.response.is_some()
                    && peer.written == PROXY_ENTRY_SMOKE_GRANT_BYTES
                    && ProxyEntrySmokeRequest::decode(&peer.request)
                        .is_ok_and(|value| value.scope == request.scope)
            })
            .ok_or_else(|| failure("smoke_reobserve_original_run_unavailable"))?;
        smoke_cancel_sender::verify_child_and_peer(
            custody,
            &peer.stream,
            pid,
            identity,
            &self.proxy_pin,
            until,
        )?;
        self.reobserve_sequence = sequence;
        self.reobserve = Some(ReobserveAttempt {
            request,
            until,
            bytes,
            written: 0,
            received: Vec::new(),
            response: None,
            first_timeout: None,
            timed_out: false,
            terminal_failure: None,
            io_error: None,
        });
        Ok(())
    }

    /// Resume waiting for the same already issued request after a local
    /// observation timeout. It never resends a fully written request or
    /// grants the proxy additional execution or cleanup time.
    pub(crate) fn resume_no_game_reobserve(&mut self, until: Instant) -> Result<(), ArenaError> {
        let attempt = self
            .reobserve
            .as_mut()
            .ok_or_else(|| failure("smoke_reobserve_missing"))?;
        if !attempt.timed_out
            || attempt.terminal_failure.is_some()
            || attempt.response.is_some()
            || Instant::now() >= until
        {
            return Err(failure("smoke_reobserve_resume_invalid"));
        }
        attempt.until = until;
        attempt.timed_out = false;
        Ok(())
    }

    pub(crate) fn poll_no_game_reobserve(
        &mut self,
        custody: &SpawnCustody,
    ) -> SmokeReobserveProgress {
        let Some(attempt) = self.reobserve.as_ref() else {
            return SmokeReobserveProgress::Failed;
        };
        if attempt.response.is_some() {
            return SmokeReobserveProgress::Observed;
        }
        if attempt.terminal_failure.is_some() {
            return SmokeReobserveProgress::Failed;
        }
        if attempt.timed_out {
            return SmokeReobserveProgress::TimedOut;
        }
        if Instant::now() >= attempt.until {
            let attempt = self.reobserve.as_mut().expect("attempt retained");
            attempt
                .first_timeout
                .get_or_insert_with(|| Arc::new(failure("smoke_reobserve_deadline")));
            attempt.timed_out = true;
            return SmokeReobserveProgress::TimedOut;
        }
        match self.poll_no_game_reobserve_inner(custody) {
            Ok(SmokeReobserveProgress::TimedOut) => {
                let attempt = self.reobserve.as_mut().expect("attempt retained");
                attempt
                    .first_timeout
                    .get_or_insert_with(|| Arc::new(failure("smoke_reobserve_deadline")));
                attempt.timed_out = true;
                SmokeReobserveProgress::TimedOut
            }
            Ok(progress) => progress,
            Err(error) => {
                let attempt = self.reobserve.as_mut().expect("attempt retained");
                attempt
                    .terminal_failure
                    .get_or_insert_with(|| Arc::new(error));
                SmokeReobserveProgress::Failed
            }
        }
    }

    fn poll_no_game_reobserve_inner(
        &mut self,
        custody: &SpawnCustody,
    ) -> Result<SmokeReobserveProgress, ArenaError> {
        let attempt = self
            .reobserve
            .as_mut()
            .ok_or_else(|| failure("smoke_reobserve_missing"))?;
        let (pid, identity) = self
            .child
            .ok_or_else(|| failure("smoke_reobserve_child_unavailable"))?;
        let peer = self
            .retained_grants
            .iter_mut()
            .find(|peer| {
                peer.response.is_some()
                    && peer.written == PROXY_ENTRY_SMOKE_GRANT_BYTES
                    && ProxyEntrySmokeRequest::decode(&peer.request)
                        .is_ok_and(|value| value.scope == attempt.request.scope)
            })
            .ok_or_else(|| failure("smoke_reobserve_original_run_unavailable"))?;
        smoke_cancel_sender::verify_child_and_peer(
            custody,
            &peer.stream,
            pid,
            identity,
            &self.proxy_pin,
            attempt.until,
        )?;
        if Instant::now() >= attempt.until {
            return Ok(SmokeReobserveProgress::TimedOut);
        }
        if attempt.written < attempt.bytes.len() {
            match peer.stream.write(&attempt.bytes[attempt.written..]) {
                Ok(0) => return Err(failure("smoke_reobserve_write_zero")),
                Ok(count) => attempt.written += count,
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                    ) => {}
                Err(error) => {
                    attempt.io_error = Some(error);
                    return Err(failure("smoke_reobserve_write"));
                }
            }
            if attempt.written < attempt.bytes.len() {
                return Ok(SmokeReobserveProgress::Pending {
                    written: attempt.written,
                    received: attempt.received.len(),
                });
            }
        }
        loop {
            if Instant::now() >= attempt.until {
                return Ok(SmokeReobserveProgress::TimedOut);
            }
            if attempt.received.len() >= FRAME_HEADER_BYTES {
                let target =
                    frame_length(&attempt.received[..FRAME_HEADER_BYTES]).map_err(failure)?;
                if attempt.received.len() == target {
                    break;
                }
            }
            let mut byte = [0u8; 1];
            match peer.stream.read(&mut byte) {
                Ok(0) => return Err(failure("smoke_reobserve_eof")),
                Ok(_) => attempt.received.push(byte[0]),
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                    ) =>
                {
                    return Ok(SmokeReobserveProgress::Pending {
                        written: attempt.written,
                        received: attempt.received.len(),
                    });
                }
                Err(error) => {
                    attempt.io_error = Some(error);
                    return Err(failure("smoke_reobserve_read"));
                }
            }
        }
        let response = ProxyReobserveResponseV1::decode(&attempt.received).map_err(failure)?;
        response.validate_for(&attempt.request).map_err(failure)?;
        smoke_cancel_sender::verify_child_and_peer(
            custody,
            &peer.stream,
            pid,
            identity,
            &self.proxy_pin,
            attempt.until,
        )?;
        if Instant::now() >= attempt.until {
            return Ok(SmokeReobserveProgress::TimedOut);
        }
        attempt.response = Some(response);
        Ok(SmokeReobserveProgress::Observed)
    }

    pub(crate) fn take_no_game_reobservation(&mut self) -> Option<SmokeReobserveObservation> {
        if self.reobserve.as_ref()?.response.is_none() {
            return None;
        }
        let attempt = self.reobserve.take()?;
        Some(SmokeReobserveObservation {
            request: attempt.request,
            response: attempt.response?,
            request_bytes: attempt.bytes,
            written: attempt.written,
            response_bytes: attempt.received,
            first_timeout: attempt.first_timeout,
        })
    }
    pub(crate) fn no_game_reobserve_failure(&self) -> Option<&ArenaError> {
        let attempt = self.reobserve.as_ref()?;
        attempt
            .terminal_failure
            .as_deref()
            .or(attempt.first_timeout.as_deref())
    }
    pub(crate) fn no_game_reobserve_first_timeout(&self) -> Option<&ArenaError> {
        self.reobserve.as_ref()?.first_timeout.as_deref()
    }
    pub(crate) fn no_game_reobserve_io_error(&self) -> Option<&io::Error> {
        self.reobserve.as_ref()?.io_error.as_ref()
    }
    pub(crate) fn no_game_reobserve_bytes(&self) -> Option<(&[u8], usize, &[u8])> {
        let attempt = self.reobserve.as_ref()?;
        Some((&attempt.bytes, attempt.written, &attempt.received))
    }
}
