//! Bounded cancellation for a NoGame proxy-entry functional smoke.
//!
//! This is separate from both the original NoGame grant and game-scoped
//! cancellation. A valid frame does not authenticate its sender, prove
//! listener provenance, acknowledge cleanup, or establish physical GPU
//! completion. Sequence ordering and peer authentication belong to the
//! owner of the original connection.

use super::proxy_entry_smoke_grant_v1::{
    ProxyEntrySmokePhase, ProxyEntrySmokeRequest, ProxyEntrySmokeScope,
    PROXY_ENTRY_SMOKE_REQUEST_BYTES,
};

pub const SMOKE_CANCEL_DOMAIN: &str = "rz-pals-proxy-entry-smoke-cancel/1";
const MAGIC: [u8; 8] = *b"RZPSC1\0\0";
const HEADER_BYTES: usize = 18;
pub const SMOKE_CANCEL_REQUEST_BYTES: usize = HEADER_BYTES + PROXY_ENTRY_SMOKE_REQUEST_BYTES;
pub const SMOKE_CANCEL_ACK_BYTES: usize = SMOKE_CANCEL_REQUEST_BYTES + 1;

/// A single request to cancel the already admitted original NoGame run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SmokeCancelRequestV1 {
    pub scope: ProxyEntrySmokeScope,
    pub sequence: u64,
}

/// An acknowledgement is about request acceptance only.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SmokeCancelDispositionV1 {
    Accepted,
    Duplicate,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SmokeCancelAckV1 {
    pub scope: ProxyEntrySmokeScope,
    pub sequence: u64,
    pub disposition: SmokeCancelDispositionV1,
}

impl SmokeCancelRequestV1 {
    pub fn encode(&self) -> Result<[u8; SMOKE_CANCEL_REQUEST_BYTES], &'static str> {
        if self.sequence == 0 || self.scope.phase != ProxyEntrySmokePhase::Run {
            return Err("smoke_cancel_scope_invalid");
        }
        let scope = ProxyEntrySmokeRequest {
            scope: self.scope.clone(),
        }
        .encode()?;
        let mut bytes = [0; SMOKE_CANCEL_REQUEST_BYTES];
        bytes[..8].copy_from_slice(&MAGIC);
        bytes[8] = 1; // request
        bytes[9] = 1; // cancel run only
        bytes[10..HEADER_BYTES].copy_from_slice(&self.sequence.to_be_bytes());
        bytes[HEADER_BYTES..].copy_from_slice(&scope);
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() != SMOKE_CANCEL_REQUEST_BYTES
            || bytes[..8] != MAGIC
            || bytes[8] != 1
            || bytes[9] != 1
        {
            return Err("smoke_cancel_request_invalid");
        }
        let sequence = u64::from_be_bytes(
            bytes[10..HEADER_BYTES]
                .try_into()
                .map_err(|_| "smoke_cancel_request_invalid")?,
        );
        let scope = ProxyEntrySmokeRequest::decode(&bytes[HEADER_BYTES..])?.scope;
        let request = Self { scope, sequence };
        request.encode()?;
        Ok(request)
    }
}

impl SmokeCancelAckV1 {
    pub fn encode(&self) -> Result<[u8; SMOKE_CANCEL_ACK_BYTES], &'static str> {
        let mut bytes = [0; SMOKE_CANCEL_ACK_BYTES];
        bytes[..SMOKE_CANCEL_REQUEST_BYTES].copy_from_slice(
            &SmokeCancelRequestV1 {
                scope: self.scope.clone(),
                sequence: self.sequence,
            }
            .encode()?,
        );
        bytes[8] = 2; // acknowledgement, not another request
        bytes[SMOKE_CANCEL_REQUEST_BYTES] = match self.disposition {
            SmokeCancelDispositionV1::Accepted => 1,
            SmokeCancelDispositionV1::Duplicate => 2,
        };
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() != SMOKE_CANCEL_ACK_BYTES || bytes[..8] != MAGIC || bytes[8] != 2 {
            return Err("smoke_cancel_ack_invalid");
        }
        let disposition = match bytes[SMOKE_CANCEL_REQUEST_BYTES] {
            1 => SmokeCancelDispositionV1::Accepted,
            2 => SmokeCancelDispositionV1::Duplicate,
            _ => return Err("smoke_cancel_ack_invalid"),
        };
        let mut request = [0; SMOKE_CANCEL_REQUEST_BYTES];
        request.copy_from_slice(&bytes[..SMOKE_CANCEL_REQUEST_BYTES]);
        request[8] = 1;
        let request = SmokeCancelRequestV1::decode(&request)?;
        Ok(Self {
            scope: request.scope,
            sequence: request.sequence,
            disposition,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pals::{
        cancel_control_v1::{CancelRequest, CANCEL_REQUEST_BYTES},
        deadline_grant_v1::{DeadlinePhase, DeadlineRole, DeadlineScope},
        proxy_entry_smoke_grant_v1::ProxyEntrySmokePolicy,
    };

    fn scope(role: DeadlineRole) -> ProxyEntrySmokeScope {
        ProxyEntrySmokeScope {
            policy: ProxyEntrySmokePolicy::ProxyEntryFunctionalSmokeV1,
            owner_id: [1; 16],
            owner_epoch: 2,
            role,
            generation: 3,
            request_id: 4,
            lock_sha256: [5; 32],
            phase: ProxyEntrySmokePhase::Run,
        }
    }

    #[test]
    fn roundtrip_preserves_original_scope_and_distinguishes_acknowledgements() {
        for role in [DeadlineRole::Baseline, DeadlineRole::Candidate] {
            let request = SmokeCancelRequestV1 {
                scope: scope(role),
                sequence: 1,
            };
            let frame = request.encode().unwrap();
            assert_eq!(SmokeCancelRequestV1::decode(&frame).unwrap(), request);
            for disposition in [
                SmokeCancelDispositionV1::Accepted,
                SmokeCancelDispositionV1::Duplicate,
            ] {
                let ack = SmokeCancelAckV1 {
                    scope: request.scope.clone(),
                    sequence: request.sequence,
                    disposition,
                };
                let encoded = ack.encode().unwrap();
                assert_eq!(SmokeCancelAckV1::decode(&encoded).unwrap(), ack);
                assert!(SmokeCancelRequestV1::decode(&encoded).is_err());
                assert!(SmokeCancelAckV1::decode(&frame).is_err());
            }
        }
    }

    #[test]
    fn invalid_action_scope_sequence_and_lengths_are_rejected() {
        let mut request = SmokeCancelRequestV1 {
            scope: scope(DeadlineRole::Baseline),
            sequence: 1,
        };
        let frame = request.encode().unwrap();
        assert!(SmokeCancelRequestV1::decode(&frame[..frame.len() - 1]).is_err());
        let mut long = frame.to_vec();
        long.push(0);
        assert!(SmokeCancelRequestV1::decode(&long).is_err());
        for index in [0, 8, 9] {
            let mut bad = frame;
            bad[index] = 0xff;
            assert!(SmokeCancelRequestV1::decode(&bad).is_err());
        }
        let mut ack = SmokeCancelAckV1 {
            scope: request.scope.clone(),
            sequence: 1,
            disposition: SmokeCancelDispositionV1::Accepted,
        }
        .encode()
        .unwrap();
        ack[SMOKE_CANCEL_REQUEST_BYTES] = 0;
        assert!(SmokeCancelAckV1::decode(&ack).is_err());
        request.sequence = 0;
        assert!(request.encode().is_err());
        request.sequence = 1;
        request.scope.phase = ProxyEntrySmokePhase::Cleanup;
        assert!(request.encode().is_err());
        request.scope.phase = ProxyEntrySmokePhase::Run;
        request.scope.owner_id = [0; 16];
        assert!(request.encode().is_err());
    }

    #[test]
    fn game_cancel_and_nogame_cancel_cannot_be_mistaken_for_each_other() {
        let game = CancelRequest {
            scope: DeadlineScope {
                owner_id: [1; 16],
                owner_epoch: 2,
                role: DeadlineRole::Baseline,
                game: 1,
                generation: 3,
                request_id: 4,
                lock_sha256: [5; 32],
                phase: DeadlinePhase::Run,
            },
            sequence: 1,
        }
        .encode()
        .unwrap();
        let no_game = SmokeCancelRequestV1 {
            scope: scope(DeadlineRole::Baseline),
            sequence: 1,
        }
        .encode()
        .unwrap();
        assert_eq!(game.len(), CANCEL_REQUEST_BYTES);
        assert!(SmokeCancelRequestV1::decode(&game).is_err());
        assert!(CancelRequest::decode(&no_game).is_err());
    }
}
