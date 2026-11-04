//! Fixed NoGame grants for the separate proxy-entry functional smoke policy.
//!
//! This wire is not a game `DeadlineScope`, exact-spawn observation, peer
//! authentication, execution attestation, cancellation or physical completion.
//! No dummy game number or cross-process Instant is encoded.

use super::deadline_grant_v1::DeadlineRole;

pub const PROXY_ENTRY_SMOKE_POLICY_DOMAIN: &str = "rz-pals-proxy-entry-functional-smoke/1";
const MAGIC: [u8; 8] = *b"RZPES1\0\0";
pub const PROXY_ENTRY_SMOKE_REQUEST_BYTES: usize = 84;
pub const PROXY_ENTRY_SMOKE_GRANT_BYTES: usize = PROXY_ENTRY_SMOKE_REQUEST_BYTES + 8;
const CHALLENGE_MAGIC: [u8; 8] = *b"RZPCH1\0\0";
pub const PROXY_ENTRY_SMOKE_CHALLENGE_BYTES: usize = 9 + 2 * PROXY_ENTRY_SMOKE_REQUEST_BYTES;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProxyEntrySmokePolicy {
    ProxyEntryFunctionalSmokeV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProxyEntrySmokePhase {
    Run,
    Cleanup,
}

/// The actual original owner establishes every field; these values cannot
/// authenticate a peer or authorize selecting a role by themselves.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProxyEntrySmokeScope {
    pub policy: ProxyEntrySmokePolicy,
    pub owner_id: [u8; 16],
    pub owner_epoch: u64,
    pub role: DeadlineRole,
    pub generation: u64,
    pub request_id: u64,
    pub lock_sha256: [u8; 32],
    pub phase: ProxyEntrySmokePhase,
}

impl ProxyEntrySmokeScope {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.owner_id == [0; 16]
            || self.owner_epoch == 0
            || self.generation == 0
            || self.request_id == 0
        {
            return Err("proxy_smoke_scope_invalid");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProxyEntrySmokeRequest {
    pub scope: ProxyEntrySmokeScope,
}

/// Two original NoGame scopes selected by the owner; the frame itself is not
/// proof of listener origin, peer identity or execution authority.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProxyEntrySmokeChallenge {
    pub run: ProxyEntrySmokeScope,
    pub cleanup: ProxyEntrySmokeScope,
}

impl ProxyEntrySmokeChallenge {
    pub fn validate(&self) -> Result<(), &'static str> {
        self.run.validate()?;
        self.cleanup.validate()?;
        let left = &self.run;
        let right = &self.cleanup;
        if left.phase != ProxyEntrySmokePhase::Run
            || right.phase != ProxyEntrySmokePhase::Cleanup
            || left.policy != right.policy
            || left.owner_id != right.owner_id
            || left.owner_epoch != right.owner_epoch
            || left.role != right.role
            || left.generation != right.generation
            || left.lock_sha256 != right.lock_sha256
            || left.request_id == right.request_id
        {
            return Err("proxy_smoke_challenge_scope_mismatch");
        }
        Ok(())
    }
    pub fn encode(&self) -> Result<[u8; PROXY_ENTRY_SMOKE_CHALLENGE_BYTES], &'static str> {
        self.validate()?;
        let run = ProxyEntrySmokeRequest {
            scope: self.run.clone(),
        }
        .encode()?;
        let cleanup = ProxyEntrySmokeRequest {
            scope: self.cleanup.clone(),
        }
        .encode()?;
        let mut bytes = [0; PROXY_ENTRY_SMOKE_CHALLENGE_BYTES];
        bytes[..8].copy_from_slice(&CHALLENGE_MAGIC);
        bytes[8] = 1;
        bytes[9..9 + PROXY_ENTRY_SMOKE_REQUEST_BYTES].copy_from_slice(&run);
        bytes[9 + PROXY_ENTRY_SMOKE_REQUEST_BYTES..].copy_from_slice(&cleanup);
        Ok(bytes)
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() != PROXY_ENTRY_SMOKE_CHALLENGE_BYTES
            || bytes[..8] != CHALLENGE_MAGIC
            || bytes[8] != 1
        {
            return Err("proxy_smoke_challenge_header_invalid");
        }
        let run =
            ProxyEntrySmokeRequest::decode(&bytes[9..9 + PROXY_ENTRY_SMOKE_REQUEST_BYTES])?.scope;
        let cleanup =
            ProxyEntrySmokeRequest::decode(&bytes[9 + PROXY_ENTRY_SMOKE_REQUEST_BYTES..])?.scope;
        let challenge = Self { run, cleanup };
        challenge.validate()?;
        Ok(challenge)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProxyEntrySmokeGrant {
    pub scope: ProxyEntrySmokeScope,
    /// Owner-measured remaining duration immediately before this reply, not an
    /// Instant or a receiver-side measurement. Zero grants no authority.
    pub remaining_ns: u64,
}

fn encode_scope(
    scope: &ProxyEntrySmokeScope,
    kind: u8,
) -> Result<[u8; PROXY_ENTRY_SMOKE_REQUEST_BYTES], &'static str> {
    scope.validate()?;
    let mut bytes = [0; PROXY_ENTRY_SMOKE_REQUEST_BYTES];
    bytes[..8].copy_from_slice(&MAGIC);
    bytes[8] = kind;
    bytes[9] = match scope.policy {
        ProxyEntrySmokePolicy::ProxyEntryFunctionalSmokeV1 => 1,
    };
    bytes[10] = match scope.role {
        DeadlineRole::Baseline => 1,
        DeadlineRole::Candidate => 2,
    };
    bytes[11] = match scope.phase {
        ProxyEntrySmokePhase::Run => 1,
        ProxyEntrySmokePhase::Cleanup => 2,
    };
    bytes[12..28].copy_from_slice(&scope.owner_id);
    bytes[28..36].copy_from_slice(&scope.owner_epoch.to_be_bytes());
    bytes[36..44].copy_from_slice(&scope.generation.to_be_bytes());
    bytes[44..52].copy_from_slice(&scope.request_id.to_be_bytes());
    bytes[52..84].copy_from_slice(&scope.lock_sha256);
    Ok(bytes)
}

fn decode_scope(bytes: &[u8], kind: u8) -> Result<ProxyEntrySmokeScope, &'static str> {
    if bytes.len() < PROXY_ENTRY_SMOKE_REQUEST_BYTES || bytes[..8] != MAGIC || bytes[8] != kind {
        return Err("proxy_smoke_header_invalid");
    }
    let policy = match bytes[9] {
        1 => ProxyEntrySmokePolicy::ProxyEntryFunctionalSmokeV1,
        _ => return Err("proxy_smoke_policy_invalid"),
    };
    let role = match bytes[10] {
        1 => DeadlineRole::Baseline,
        2 => DeadlineRole::Candidate,
        _ => return Err("proxy_smoke_role_invalid"),
    };
    let phase = match bytes[11] {
        1 => ProxyEntrySmokePhase::Run,
        2 => ProxyEntrySmokePhase::Cleanup,
        _ => return Err("proxy_smoke_phase_invalid"),
    };
    let scope = ProxyEntrySmokeScope {
        policy,
        owner_id: bytes[12..28]
            .try_into()
            .map_err(|_| "proxy_smoke_length_invalid")?,
        owner_epoch: u64::from_be_bytes(
            bytes[28..36]
                .try_into()
                .map_err(|_| "proxy_smoke_length_invalid")?,
        ),
        role,
        generation: u64::from_be_bytes(
            bytes[36..44]
                .try_into()
                .map_err(|_| "proxy_smoke_length_invalid")?,
        ),
        request_id: u64::from_be_bytes(
            bytes[44..52]
                .try_into()
                .map_err(|_| "proxy_smoke_length_invalid")?,
        ),
        lock_sha256: bytes[52..84]
            .try_into()
            .map_err(|_| "proxy_smoke_length_invalid")?,
        phase,
    };
    scope.validate()?;
    Ok(scope)
}

impl ProxyEntrySmokeRequest {
    pub fn encode(&self) -> Result<[u8; PROXY_ENTRY_SMOKE_REQUEST_BYTES], &'static str> {
        encode_scope(&self.scope, 1)
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() != PROXY_ENTRY_SMOKE_REQUEST_BYTES {
            return Err("proxy_smoke_length_invalid");
        }
        Ok(Self {
            scope: decode_scope(bytes, 1)?,
        })
    }
}

impl ProxyEntrySmokeGrant {
    pub fn encode(&self) -> Result<[u8; PROXY_ENTRY_SMOKE_GRANT_BYTES], &'static str> {
        if self.remaining_ns == 0 {
            return Err("proxy_smoke_deadline_expired");
        }
        let mut bytes = [0; PROXY_ENTRY_SMOKE_GRANT_BYTES];
        bytes[..PROXY_ENTRY_SMOKE_REQUEST_BYTES].copy_from_slice(&encode_scope(&self.scope, 2)?);
        bytes[PROXY_ENTRY_SMOKE_REQUEST_BYTES..].copy_from_slice(&self.remaining_ns.to_be_bytes());
        Ok(bytes)
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() != PROXY_ENTRY_SMOKE_GRANT_BYTES {
            return Err("proxy_smoke_length_invalid");
        }
        let scope = decode_scope(bytes, 2)?;
        let remaining_ns = u64::from_be_bytes(
            bytes[PROXY_ENTRY_SMOKE_REQUEST_BYTES..]
                .try_into()
                .map_err(|_| "proxy_smoke_length_invalid")?,
        );
        if remaining_ns == 0 {
            return Err("proxy_smoke_deadline_expired");
        }
        Ok(Self {
            scope,
            remaining_ns,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::deadline_grant_v1::{DeadlinePhase, DeadlineRequest, DeadlineScope};
    use super::*;
    fn scope(role: DeadlineRole, phase: ProxyEntrySmokePhase) -> ProxyEntrySmokeScope {
        ProxyEntrySmokeScope {
            policy: ProxyEntrySmokePolicy::ProxyEntryFunctionalSmokeV1,
            owner_id: [1; 16],
            owner_epoch: 2,
            role,
            generation: 3,
            request_id: 4,
            lock_sha256: [5; 32],
            phase,
        }
    }
    #[test]
    fn no_game_scope_roundtrips_for_both_roles_and_phases() {
        for role in [DeadlineRole::Baseline, DeadlineRole::Candidate] {
            for phase in [ProxyEntrySmokePhase::Run, ProxyEntrySmokePhase::Cleanup] {
                let scope = scope(role, phase);
                let request = ProxyEntrySmokeRequest {
                    scope: scope.clone(),
                };
                let grant = ProxyEntrySmokeGrant {
                    scope,
                    remaining_ns: 42,
                };
                assert_eq!(
                    ProxyEntrySmokeRequest::decode(&request.encode().unwrap()).unwrap(),
                    request
                );
                assert_eq!(
                    ProxyEntrySmokeGrant::decode(&grant.encode().unwrap()).unwrap(),
                    grant
                );
                assert!(ProxyEntrySmokeGrant::decode(&request.encode().unwrap()).is_err());
                assert!(ProxyEntrySmokeRequest::decode(&grant.encode().unwrap()).is_err());
            }
        }
    }
    #[test]
    fn malformed_policy_phase_length_and_zero_ids_are_rejected() {
        let request = ProxyEntrySmokeRequest {
            scope: scope(DeadlineRole::Baseline, ProxyEntrySmokePhase::Run),
        };
        let encoded = request.encode().unwrap();
        assert!(ProxyEntrySmokeRequest::decode(&encoded[..encoded.len() - 1]).is_err());
        for index in [8, 9, 10, 11] {
            let mut bad = encoded;
            bad[index] = 0;
            assert!(ProxyEntrySmokeRequest::decode(&bad).is_err());
        }
        let mut bad_scope = request.scope;
        bad_scope.owner_id = [0; 16];
        assert!(
            ProxyEntrySmokeRequest { scope: bad_scope }
                .encode()
                .is_err()
        );
        let grant = ProxyEntrySmokeGrant {
            scope: scope(DeadlineRole::Baseline, ProxyEntrySmokePhase::Run),
            remaining_ns: 0,
        };
        assert!(grant.encode().is_err());
    }
    #[test]
    fn challenge_requires_same_original_nogame_scope_and_distinct_requests() {
        let run = scope(DeadlineRole::Baseline, ProxyEntrySmokePhase::Run);
        let mut cleanup = scope(DeadlineRole::Baseline, ProxyEntrySmokePhase::Cleanup);
        cleanup.request_id += 1;
        let challenge = ProxyEntrySmokeChallenge { run, cleanup };
        let encoded = challenge.encode().unwrap();
        assert_eq!(
            ProxyEntrySmokeChallenge::decode(&encoded).unwrap(),
            challenge
        );
        assert!(ProxyEntrySmokeChallenge::decode(&encoded[..encoded.len() - 1]).is_err());
        let mut different = challenge.clone();
        different.cleanup.role = DeadlineRole::Candidate;
        assert!(different.encode().is_err());
        different.cleanup.role = different.run.role;
        different.cleanup.request_id = different.run.request_id;
        assert!(different.encode().is_err());
    }
    #[test]
    fn game_deadline_wire_is_not_a_smoke_wire() {
        let legacy = DeadlineRequest {
            scope: DeadlineScope {
                owner_id: [1; 16],
                owner_epoch: 2,
                role: DeadlineRole::Baseline,
                game: 0,
                generation: 3,
                request_id: 4,
                lock_sha256: [5; 32],
                phase: DeadlinePhase::Run,
            },
        };
        assert!(ProxyEntrySmokeRequest::decode(&legacy.encode().unwrap()).is_err());
    }
}
