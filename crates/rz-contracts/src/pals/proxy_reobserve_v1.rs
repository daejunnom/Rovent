//! Read-only observation on the retained, authenticated NoGame Run stream.
//! This separate wire does not renew execution authority or prove a fence.

use super::proxy_entry_smoke_grant_v1::{
    PROXY_ENTRY_SMOKE_REQUEST_BYTES, ProxyEntrySmokePhase, ProxyEntrySmokeRequest,
    ProxyEntrySmokeScope,
};

pub const FRAME_LIMIT: usize = 4096;
pub const FRAME_HEADER_BYTES: usize = 12;
const MAGIC: [u8; 8] = *b"RZPERV1\0";
const REQUEST_KIND: u8 = 1;
const RESPONSE_KIND: u8 = 2;
const SCOPE_END: usize = 1 + PROXY_ENTRY_SMOKE_REQUEST_BYTES;
const REQUEST_BODY: usize = SCOPE_END + 16;
const RESPONSE_FIXED: usize = SCOPE_END + 8 + 1 + 1 + 2;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProxyReobserveRequestV1 {
    pub scope: ProxyEntrySmokeScope,
    pub sequence: u64,
    /// Bounds observation waiting only; it grants no execution time.
    pub observation_budget_ms: u64,
}

impl ProxyReobserveRequestV1 {
    pub fn validate(&self) -> Result<(), &'static str> {
        self.scope.validate()?;
        if self.scope.phase != ProxyEntrySmokePhase::Run
            || self.sequence == 0
            || self.observation_budget_ms == 0
        {
            return Err("proxy_reobserve_request_invalid");
        }
        Ok(())
    }
    pub fn encode(&self) -> Result<Vec<u8>, &'static str> {
        self.validate()?;
        let mut body = Vec::with_capacity(REQUEST_BODY);
        body.push(REQUEST_KIND);
        body.extend_from_slice(
            &ProxyEntrySmokeRequest {
                scope: self.scope.clone(),
            }
            .encode()?,
        );
        body.extend_from_slice(&self.sequence.to_be_bytes());
        body.extend_from_slice(&self.observation_budget_ms.to_be_bytes());
        encode_frame(&body)
    }
    pub fn decode(frame: &[u8]) -> Result<Self, &'static str> {
        let body = decode_frame(frame)?;
        if body.len() != REQUEST_BODY || body[0] != REQUEST_KIND {
            return Err("proxy_reobserve_request_frame_invalid");
        }
        let request = Self {
            scope: ProxyEntrySmokeRequest::decode(&body[1..SCOPE_END])?.scope,
            sequence: u64_at(body, SCOPE_END)?,
            observation_budget_ms: u64_at(body, SCOPE_END + 8)?,
        };
        request.validate()?;
        Ok(request)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProxyReobserveStatusV1 {
    Pending,
    Unknown,
    BeforeNativeLate,
    NativeFinishedLate,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProxyReobserveResponseV1 {
    pub scope: ProxyEntrySmokeScope,
    pub sequence: u64,
    pub status: ProxyReobserveStatusV1,
    pub original_cleanup_elapsed: Option<bool>,
    pub error: Option<String>,
}

impl ProxyReobserveResponseV1 {
    pub fn validate_for(&self, request: &ProxyReobserveRequestV1) -> Result<(), &'static str> {
        request.validate()?;
        if self.scope != request.scope || self.sequence != request.sequence {
            return Err("proxy_reobserve_response_scope_mismatch");
        }
        Ok(())
    }
    pub fn encode(&self) -> Result<Vec<u8>, &'static str> {
        self.scope.validate()?;
        if self.scope.phase != ProxyEntrySmokePhase::Run || self.sequence == 0 {
            return Err("proxy_reobserve_response_invalid");
        }
        let error = self.error.as_deref().map(str::as_bytes);
        let error_len = match error {
            Some(value) => {
                u16::try_from(value.len()).map_err(|_| "proxy_reobserve_error_too_long")?
            }
            None => u16::MAX,
        };
        if error_len == u16::MAX && error.is_some() {
            return Err("proxy_reobserve_error_too_long");
        }
        let mut body = Vec::with_capacity(RESPONSE_FIXED + error.map_or(0, <[u8]>::len));
        body.push(RESPONSE_KIND);
        body.extend_from_slice(
            &ProxyEntrySmokeRequest {
                scope: self.scope.clone(),
            }
            .encode()?,
        );
        body.extend_from_slice(&self.sequence.to_be_bytes());
        body.push(match self.status {
            ProxyReobserveStatusV1::Pending => 1,
            ProxyReobserveStatusV1::Unknown => 2,
            ProxyReobserveStatusV1::BeforeNativeLate => 3,
            ProxyReobserveStatusV1::NativeFinishedLate => 4,
        });
        body.push(match self.original_cleanup_elapsed {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        });
        body.extend_from_slice(&error_len.to_be_bytes());
        if let Some(error) = error {
            body.extend_from_slice(error);
        }
        encode_frame(&body)
    }
    pub fn decode(frame: &[u8]) -> Result<Self, &'static str> {
        let body = decode_frame(frame)?;
        if body.len() < RESPONSE_FIXED || body[0] != RESPONSE_KIND {
            return Err("proxy_reobserve_response_frame_invalid");
        }
        let scope = ProxyEntrySmokeRequest::decode(&body[1..SCOPE_END])?.scope;
        let sequence = u64_at(body, SCOPE_END)?;
        if scope.phase != ProxyEntrySmokePhase::Run || sequence == 0 {
            return Err("proxy_reobserve_response_invalid");
        }
        let status = match body[SCOPE_END + 8] {
            1 => ProxyReobserveStatusV1::Pending,
            2 => ProxyReobserveStatusV1::Unknown,
            3 => ProxyReobserveStatusV1::BeforeNativeLate,
            4 => ProxyReobserveStatusV1::NativeFinishedLate,
            _ => return Err("proxy_reobserve_status_invalid"),
        };
        let original_cleanup_elapsed = match body[SCOPE_END + 9] {
            0 => None,
            1 => Some(false),
            2 => Some(true),
            _ => return Err("proxy_reobserve_elapsed_invalid"),
        };
        let len = u16::from_be_bytes(
            body[SCOPE_END + 10..RESPONSE_FIXED]
                .try_into()
                .map_err(|_| "proxy_reobserve_length_invalid")?,
        );
        let error = if len == u16::MAX {
            if body.len() != RESPONSE_FIXED {
                return Err("proxy_reobserve_length_invalid");
            }
            None
        } else {
            if body.len() != RESPONSE_FIXED + usize::from(len) {
                return Err("proxy_reobserve_length_invalid");
            }
            Some(
                std::str::from_utf8(&body[RESPONSE_FIXED..])
                    .map_err(|_| "proxy_reobserve_error_utf8")?
                    .to_owned(),
            )
        };
        Ok(Self {
            scope,
            sequence,
            status,
            original_cleanup_elapsed,
            error,
        })
    }
}

/// Inspect the fixed header before allocating a bounded frame buffer.
pub fn frame_length(header: &[u8]) -> Result<usize, &'static str> {
    if header.len() != FRAME_HEADER_BYTES || header[..8] != MAGIC {
        return Err("proxy_reobserve_header_invalid");
    }
    let body = u32::from_be_bytes(
        header[8..12]
            .try_into()
            .map_err(|_| "proxy_reobserve_length_invalid")?,
    ) as usize;
    let total = FRAME_HEADER_BYTES
        .checked_add(body)
        .ok_or("proxy_reobserve_length_invalid")?;
    if total > FRAME_LIMIT || body == 0 {
        return Err("proxy_reobserve_length_invalid");
    }
    Ok(total)
}

fn encode_frame(body: &[u8]) -> Result<Vec<u8>, &'static str> {
    let total = FRAME_HEADER_BYTES
        .checked_add(body.len())
        .ok_or("proxy_reobserve_length_invalid")?;
    if total > FRAME_LIMIT || body.is_empty() {
        return Err("proxy_reobserve_length_invalid");
    }
    let len = u32::try_from(body.len()).map_err(|_| "proxy_reobserve_length_invalid")?;
    let mut frame = Vec::with_capacity(total);
    frame.extend_from_slice(&MAGIC);
    frame.extend_from_slice(&len.to_be_bytes());
    frame.extend_from_slice(body);
    Ok(frame)
}
fn decode_frame(frame: &[u8]) -> Result<&[u8], &'static str> {
    if frame.len() < FRAME_HEADER_BYTES
        || frame_length(&frame[..FRAME_HEADER_BYTES])? != frame.len()
    {
        return Err("proxy_reobserve_length_invalid");
    }
    Ok(&frame[FRAME_HEADER_BYTES..])
}
fn u64_at(bytes: &[u8], offset: usize) -> Result<u64, &'static str> {
    Ok(u64::from_be_bytes(
        bytes
            .get(offset..offset + 8)
            .ok_or("proxy_reobserve_length_invalid")?
            .try_into()
            .map_err(|_| "proxy_reobserve_length_invalid")?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pals::{
        deadline_grant_v1::DeadlineRole,
        proxy_entry_smoke_grant_v1::{ProxyEntrySmokeGrant, ProxyEntrySmokePolicy},
    };
    #[test]
    fn reobserve_codec_binds_original_run_and_rejects_grant_wire() {
        let request = ProxyReobserveRequestV1 {
            scope: ProxyEntrySmokeScope {
                policy: ProxyEntrySmokePolicy::ProxyEntryFunctionalSmokeV1,
                owner_id: [1; 16],
                owner_epoch: 1,
                role: DeadlineRole::Baseline,
                generation: 1,
                request_id: 1,
                lock_sha256: [2; 32],
                phase: ProxyEntrySmokePhase::Run,
            },
            sequence: 1,
            observation_budget_ms: 1,
        };
        let frame = request.encode().unwrap();
        assert_eq!(ProxyReobserveRequestV1::decode(&frame).unwrap(), request);
        assert!(ProxyEntrySmokeRequest::decode(&frame).is_err());
        let old_request = ProxyEntrySmokeRequest {
            scope: request.scope.clone(),
        }
        .encode()
        .unwrap();
        assert!(ProxyReobserveRequestV1::decode(&old_request).is_err());
        let old_grant = ProxyEntrySmokeGrant {
            scope: request.scope.clone(),
            remaining_ns: 1,
        }
        .encode()
        .unwrap();
        assert!(ProxyReobserveRequestV1::decode(&old_grant).is_err());
        assert!(ProxyEntrySmokeGrant::decode(&frame).is_err());
        let mut response = ProxyReobserveResponseV1 {
            scope: request.scope.clone(),
            sequence: request.sequence,
            status: ProxyReobserveStatusV1::Unknown,
            original_cleanup_elapsed: None,
            error: None,
        };
        assert_eq!(
            ProxyReobserveResponseV1::decode(&response.encode().unwrap()).unwrap(),
            response
        );
        assert!(response.validate_for(&request).is_ok());
        response.error = Some("원 오류".into());
        response.original_cleanup_elapsed = Some(false);
        assert_eq!(
            ProxyReobserveResponseV1::decode(&response.encode().unwrap()).unwrap(),
            response
        );
        response.error = Some("x".repeat(FRAME_LIMIT));
        assert!(response.encode().is_err());
        response.sequence += 1;
        assert!(response.validate_for(&request).is_err());
        let mut wrong = request.clone();
        wrong.scope.phase = ProxyEntrySmokePhase::Cleanup;
        assert!(wrong.encode().is_err());
        wrong = request;
        wrong.observation_budget_ms = 0;
        assert!(wrong.encode().is_err());
    }
}
