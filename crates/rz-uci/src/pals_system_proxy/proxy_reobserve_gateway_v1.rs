//! Observation-only forwarding on the original NoGame Run grant stream.
//! A response is a child report, not a GPU fence or execution authority.

use super::{
    proxy_entry_smoke_channel_v1::OriginalDirectSmokeGrant,
    runner_v6::{CompletedChildReobserveV1, PalsProxySupervisorV6},
};
use rz_contracts::pals::{
    child_reobserve_v1::ChildReobserveStatusV1,
    proxy_reobserve_v1::{
        FRAME_HEADER_BYTES, FRAME_LIMIT, ProxyReobserveRequestV1, ProxyReobserveResponseV1,
        ProxyReobserveStatusV1, frame_length,
    },
};
use std::{
    io::{self, Read, Write},
    time::{Duration, Instant},
};

/// Owns partial arena frames until their exact exchange is consumed. It never
/// owns a replacement socket or changes the original Run/cleanup windows.
pub(super) struct ProxyReobserveGatewayV1 {
    request_frame: Vec<u8>,
    request_len: Option<usize>,
    request: Option<ProxyReobserveRequestV1>,
    sequence: u64,
    response_frame: Option<Vec<u8>>,
    response_written: usize,
    child_exchange: Option<CompletedChildReobserveV1>,
    child_active: bool,
    child_error: Option<io::Error>,
    completed: Option<CompletedProxyReobserveV1>,
    failure: Option<io::Error>,
}

/// Exact arena frames and child exchange from one complete forwarding pass.
pub(super) struct CompletedProxyReobserveV1 {
    pub request_frame: Vec<u8>,
    pub response_frame: Vec<u8>,
    pub child_exchange: Option<CompletedChildReobserveV1>,
    pub response_sent_after_window: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ProxyReobserveGatewayProgressV1 {
    Pending,
    ResponseSent,
    AwaitingEvidenceRecovery,
}

impl ProxyReobserveGatewayV1 {
    pub(super) fn new() -> Self {
        Self {
            request_frame: Vec::new(),
            request_len: None,
            request: None,
            sequence: 0,
            response_frame: None,
            response_written: 0,
            child_exchange: None,
            child_active: false,
            child_error: None,
            completed: None,
            failure: None,
        }
    }

    pub(super) fn first_failure(&self) -> Option<&io::Error> {
        self.failure.as_ref()
    }

    pub(super) fn child_error(&self) -> Option<&io::Error> {
        self.child_error.as_ref()
    }

    pub(super) fn partial_frames(&self) -> (&[u8], Option<&[u8]>, usize) {
        (
            &self.request_frame,
            self.response_frame.as_deref(),
            self.response_written,
        )
    }

    /// The caller receives the original completed child exchange before
    /// another arena request is admitted; it decides how to persist it.
    pub(super) fn take_completed(&mut self) -> Option<CompletedProxyReobserveV1> {
        self.completed.take()
    }

    pub(super) fn poll(
        &mut self,
        run: &mut OriginalDirectSmokeGrant,
        child: &mut PalsProxySupervisorV6,
        until: Instant,
    ) -> io::Result<ProxyReobserveGatewayProgressV1> {
        if let Some(error) = &self.failure {
            return Err(io::Error::new(error.kind(), error.to_string()));
        }
        if Instant::now() >= until {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "proxy observation window expired",
            ));
        }
        if self.completed.is_some() {
            return Ok(ProxyReobserveGatewayProgressV1::AwaitingEvidenceRecovery);
        }
        let result = self.poll_inner(run, child, until);
        if let Err(error) = result {
            if error.kind() == io::ErrorKind::TimedOut {
                return Err(error);
            }
            self.failure = Some(error);
            let failure = self.failure.as_ref().expect("failure retained");
            return Err(io::Error::new(failure.kind(), failure.to_string()));
        }
        result
    }

    fn poll_inner(
        &mut self,
        run: &mut OriginalDirectSmokeGrant,
        child: &mut PalsProxySupervisorV6,
        until: Instant,
    ) -> io::Result<ProxyReobserveGatewayProgressV1> {
        if let Some(response) = &self.response_frame {
            let stream = run.reobserve_stream()?;
            match stream.write(&response[self.response_written..]) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "proxy observation response closed",
                    ));
                }
                Ok(count) => self.response_written += count,
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                    ) =>
                {
                    return Ok(ProxyReobserveGatewayProgressV1::Pending);
                }
                Err(error) => return Err(error),
            }
            if self.response_written == response.len() {
                self.completed = Some(CompletedProxyReobserveV1 {
                    request_frame: std::mem::take(&mut self.request_frame),
                    response_frame: self.response_frame.take().expect("response retained"),
                    child_exchange: self.child_exchange.take(),
                    response_sent_after_window: Instant::now() >= until,
                });
                self.response_written = 0;
                self.request_len = None;
                self.request = None;
                return Ok(ProxyReobserveGatewayProgressV1::ResponseSent);
            }
            return Ok(ProxyReobserveGatewayProgressV1::Pending);
        }

        if self.request.is_none() {
            let expected = self.request_len.unwrap_or(FRAME_HEADER_BYTES);
            if self.request_frame.len() < expected {
                let mut buffer = [0; FRAME_LIMIT];
                let want = expected - self.request_frame.len();
                let stream = run.reobserve_stream()?;
                match stream.read(&mut buffer[..want]) {
                    Ok(0) => {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "original arena observation stream closed",
                        ));
                    }
                    Ok(count) => self.request_frame.extend_from_slice(&buffer[..count]),
                    Err(error)
                        if matches!(
                            error.kind(),
                            io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                        ) =>
                    {
                        return Ok(ProxyReobserveGatewayProgressV1::Pending);
                    }
                    Err(error) => return Err(error),
                }
            }
            if self.request_len.is_none() && self.request_frame.len() == FRAME_HEADER_BYTES {
                self.request_len =
                    Some(frame_length(&self.request_frame).map_err(io::Error::other)?);
            }
            if self.request_len != Some(self.request_frame.len()) {
                return Ok(ProxyReobserveGatewayProgressV1::Pending);
            }
            let request =
                ProxyReobserveRequestV1::decode(&self.request_frame).map_err(io::Error::other)?;
            if request.scope != *run.scope()
                || self.sequence.checked_add(1) != Some(request.sequence)
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "arena observation scope or sequence differs",
                ));
            }
            let now = Instant::now();
            let remaining = until
                .checked_duration_since(now)
                .filter(|v| !v.is_zero())
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::TimedOut, "proxy observation window expired")
                })?;
            let child_until = now
                .checked_add(Duration::from_millis(request.observation_budget_ms).min(remaining))
                .ok_or_else(|| io::Error::other("observation deadline overflow"))?;
            if self.child_error.is_none() {
                let result = if self.child_active {
                    child.resume_child_reobserve(child_until)
                } else {
                    child.begin_child_reobserve(child_until, request.observation_budget_ms)
                };
                match result {
                    Ok(()) => self.child_active = true,
                    Err(error) if error.kind() == io::ErrorKind::TimedOut => return Err(error),
                    Err(error) => self.child_error = Some(error),
                }
            }
            self.sequence = request.sequence;
            self.request = Some(request);
        }

        if self.child_error.is_some() {
            let request = self.request.as_ref().expect("request retained");
            self.response_frame = Some(
                ProxyReobserveResponseV1 {
                    scope: request.scope.clone(),
                    sequence: request.sequence,
                    status: ProxyReobserveStatusV1::Unknown,
                    original_cleanup_elapsed: None,
                    error: Some("original child observation failed; evidence retained".into()),
                }
                .encode()
                .map_err(io::Error::other)?,
            );
            return Ok(ProxyReobserveGatewayProgressV1::Pending);
        }

        match child.poll_child_reobserve() {
            Ok(false) => Ok(ProxyReobserveGatewayProgressV1::Pending),
            Err(error) if error.kind() == io::ErrorKind::TimedOut => {
                let request = self.request.as_ref().expect("request retained");
                self.response_frame = Some(
                    ProxyReobserveResponseV1 {
                        scope: request.scope.clone(),
                        sequence: request.sequence,
                        status: ProxyReobserveStatusV1::Pending,
                        original_cleanup_elapsed: None,
                        error: None,
                    }
                    .encode()
                    .map_err(io::Error::other)?,
                );
                Ok(ProxyReobserveGatewayProgressV1::Pending)
            }
            Err(error) => Err(error),
            Ok(true) => {
                let request = self.request.as_ref().expect("request retained");
                let response = if let Some(exchange) = child.take_successful_child_reobserve() {
                    self.child_active = false;
                    let status = match exchange.response.status {
                        ChildReobserveStatusV1::Pending => ProxyReobserveStatusV1::Pending,
                        ChildReobserveStatusV1::Unknown => ProxyReobserveStatusV1::Unknown,
                        ChildReobserveStatusV1::BeforeNativeLate => {
                            ProxyReobserveStatusV1::BeforeNativeLate
                        }
                        ChildReobserveStatusV1::NativeFinishedLate => {
                            ProxyReobserveStatusV1::NativeFinishedLate
                        }
                    };
                    let response = ProxyReobserveResponseV1 {
                        scope: request.scope.clone(),
                        sequence: request.sequence,
                        status,
                        original_cleanup_elapsed: Some(exchange.response.original_cleanup_elapsed),
                        error: exchange.response.original_error.clone(),
                    };
                    self.child_exchange = Some(exchange);
                    response
                } else {
                    self.child_active = false;
                    self.child_error = Some(io::Error::other(
                        "original child observation failed; evidence retained",
                    ));
                    ProxyReobserveResponseV1 {
                        scope: request.scope.clone(),
                        sequence: request.sequence,
                        status: ProxyReobserveStatusV1::Unknown,
                        original_cleanup_elapsed: None,
                        error: Some(
                            "original child observation failed; partial evidence retained".into(),
                        ),
                    }
                };
                self.response_frame = Some(response.encode().map_err(io::Error::other)?);
                Ok(ProxyReobserveGatewayProgressV1::Pending)
            }
        }
    }
}
