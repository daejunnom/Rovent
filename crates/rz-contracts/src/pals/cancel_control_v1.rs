//! Separate bounded run-cancellation wire. This is not peer authentication,
//! physical completion, cleanup acknowledgement, or a change to PALS 0.1.

use super::deadline_grant_v1::{DeadlinePhase, DeadlineRequest, DeadlineScope, REQUEST_BYTES};

const MAGIC: [u8; 8] = *b"RZPCC1\0\0";
const HEADER_BYTES: usize = 18;
pub const CANCEL_REQUEST_BYTES: usize = HEADER_BYTES + REQUEST_BYTES;
pub const CANCEL_ACK_BYTES: usize = CANCEL_REQUEST_BYTES + 1;

/// Exactly one action exists in V1: request cancellation of the run. The
/// scope is the already admitted original run grant, including request_id.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CancelRequest {
    pub scope: DeadlineScope,
    pub sequence: u64,
}

/// An ACK confirms only that this cancellation was accepted or was an exact
/// duplicate. It says nothing about process exit, drain or physical completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CancelAckDisposition {
    Accepted,
    Duplicate,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CancelAck {
    pub scope: DeadlineScope,
    pub sequence: u64,
    pub disposition: CancelAckDisposition,
}

impl CancelRequest {
    pub fn encode(&self) -> Result<[u8; CANCEL_REQUEST_BYTES], &'static str> {
        if self.sequence == 0 || self.scope.phase != DeadlinePhase::Run {
            return Err("cancel_scope_invalid");
        }
        let inner = DeadlineRequest {
            scope: self.scope.clone(),
        }
        .encode()?;
        let mut out = [0; CANCEL_REQUEST_BYTES];
        out[..8].copy_from_slice(&MAGIC);
        out[8] = 1; // request
        out[9] = 1; // cancel run only
        out[10..18].copy_from_slice(&self.sequence.to_be_bytes());
        out[HEADER_BYTES..].copy_from_slice(&inner);
        Ok(out)
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() != CANCEL_REQUEST_BYTES
            || bytes[..8] != MAGIC
            || bytes[8] != 1
            || bytes[9] != 1
        {
            return Err("cancel_request_invalid");
        }
        let sequence = u64::from_be_bytes(
            bytes[10..18]
                .try_into()
                .map_err(|_| "cancel_request_invalid")?,
        );
        let scope = DeadlineRequest::decode(&bytes[HEADER_BYTES..])?.scope;
        let request = Self { scope, sequence };
        request.encode()?;
        Ok(request)
    }
}

impl CancelAck {
    pub fn encode(&self) -> Result<[u8; CANCEL_ACK_BYTES], &'static str> {
        let mut out = [0; CANCEL_ACK_BYTES];
        out[..CANCEL_REQUEST_BYTES].copy_from_slice(
            &CancelRequest {
                scope: self.scope.clone(),
                sequence: self.sequence,
            }
            .encode()?,
        );
        out[8] = 2; // acknowledgement; distinct from a request
        out[CANCEL_REQUEST_BYTES] = match self.disposition {
            CancelAckDisposition::Accepted => 1,
            CancelAckDisposition::Duplicate => 2,
        };
        Ok(out)
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() != CANCEL_ACK_BYTES || bytes[..8] != MAGIC || bytes[8] != 2 {
            return Err("cancel_ack_invalid");
        }
        let disposition = match bytes[CANCEL_REQUEST_BYTES] {
            1 => CancelAckDisposition::Accepted,
            2 => CancelAckDisposition::Duplicate,
            _ => return Err("cancel_ack_invalid"),
        };
        let mut request = [0; CANCEL_REQUEST_BYTES];
        request.copy_from_slice(&bytes[..CANCEL_REQUEST_BYTES]);
        request[8] = 1;
        let request = CancelRequest::decode(&request)?;
        Ok(Self {
            scope: request.scope,
            sequence: request.sequence,
            disposition,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::deadline_grant_v1::DeadlineRole;
    use super::*;
    fn scope() -> DeadlineScope {
        DeadlineScope {
            owner_id: [1; 16],
            owner_epoch: 2,
            role: DeadlineRole::Candidate,
            game: 3,
            generation: 4,
            request_id: 5,
            lock_sha256: [6; 32],
            phase: DeadlinePhase::Run,
        }
    }
    #[test]
    fn roundtrip_and_strict_action_scope_and_frame_lengths() {
        let request = CancelRequest {
            scope: scope(),
            sequence: 1,
        };
        let encoded = request.encode().unwrap();
        assert_eq!(CancelRequest::decode(&encoded).unwrap(), request);
        assert!(CancelRequest::decode(&encoded[..encoded.len() - 1]).is_err());
        let mut invalid = encoded;
        invalid[9] = 2;
        assert!(CancelRequest::decode(&invalid).is_err());
        let ack = CancelAck {
            scope: scope(),
            sequence: 1,
            disposition: CancelAckDisposition::Accepted,
        };
        assert_eq!(CancelAck::decode(&ack.encode().unwrap()).unwrap(), ack);
        assert!(CancelRequest::decode(&ack.encode().unwrap()).is_err());
        let mut non_run = request;
        non_run.scope.phase = DeadlinePhase::Cleanup;
        assert!(non_run.encode().is_err());
        non_run.scope.phase = DeadlinePhase::Run;
        non_run.sequence = 0;
        assert!(non_run.encode().is_err());
    }
}
