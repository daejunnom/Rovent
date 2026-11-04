//! Bounded, versioned IPC encoding for an arena owner's deadline grant.
//!
//! This is separate from the PALS revision 0.1 and existing canonical formats.
//! Scope identifies an expected conversation; it authenticates neither a peer
//! nor a process. Replay, transport, original-owner custody and GPU completion
//! remain the respective consumers' responsibilities.

const MAGIC: [u8; 8] = *b"RZPDG1\0\0";
pub const REQUEST_BYTES: usize = 87;
pub const GRANT_BYTES: usize = 95;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeadlineRole {
    Baseline,
    Candidate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeadlinePhase {
    Startup,
    Run,
    Cleanup,
}

/// Owner identity and epoch must come from the live owner's admission state.
/// The lock hash is the binary SHA-256 value, not a hash of this message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeadlineScope {
    pub owner_id: [u8; 16],
    pub owner_epoch: u64,
    pub role: DeadlineRole,
    pub game: u32,
    pub generation: u64,
    pub request_id: u64,
    pub lock_sha256: [u8; 32],
    pub phase: DeadlinePhase,
}

impl DeadlineScope {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.owner_id == [0; 16]
            || self.owner_epoch == 0
            || self.generation == 0
            || self.request_id == 0
        {
            return Err("deadline_scope_invalid");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeadlineRequest {
    pub scope: DeadlineScope,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeadlineGrant {
    pub scope: DeadlineScope,
    /// Duration measured by the owner immediately before preparing this reply,
    /// from its current monotonic clock to the owner's original phase deadline.
    /// It is neither a serialized Instant nor the receiver's remaining time.
    pub remaining_ns: u64,
}

fn scope_bytes(scope: &DeadlineScope, kind: u8) -> Result<[u8; REQUEST_BYTES], &'static str> {
    scope.validate()?;
    let mut out = [0; REQUEST_BYTES];
    out[..8].copy_from_slice(&MAGIC);
    out[8] = kind;
    out[9] = match scope.role {
        DeadlineRole::Baseline => 1,
        DeadlineRole::Candidate => 2,
    };
    out[10] = match scope.phase {
        DeadlinePhase::Startup => 1,
        DeadlinePhase::Run => 2,
        DeadlinePhase::Cleanup => 3,
    };
    out[11..27].copy_from_slice(&scope.owner_id);
    out[27..35].copy_from_slice(&scope.owner_epoch.to_be_bytes());
    out[35..39].copy_from_slice(&scope.game.to_be_bytes());
    out[39..47].copy_from_slice(&scope.generation.to_be_bytes());
    out[47..55].copy_from_slice(&scope.request_id.to_be_bytes());
    out[55..87].copy_from_slice(&scope.lock_sha256);
    Ok(out)
}

fn read_scope(bytes: &[u8], kind: u8) -> Result<DeadlineScope, &'static str> {
    if bytes.len() < REQUEST_BYTES || bytes[..8] != MAGIC || bytes[8] != kind {
        return Err("deadline_wire_header_invalid");
    }
    let role = match bytes[9] {
        1 => DeadlineRole::Baseline,
        2 => DeadlineRole::Candidate,
        _ => return Err("deadline_wire_role_invalid"),
    };
    let phase = match bytes[10] {
        1 => DeadlinePhase::Startup,
        2 => DeadlinePhase::Run,
        3 => DeadlinePhase::Cleanup,
        _ => return Err("deadline_wire_phase_invalid"),
    };
    let scope = DeadlineScope {
        owner_id: bytes[11..27]
            .try_into()
            .map_err(|_| "deadline_wire_length_invalid")?,
        owner_epoch: u64::from_be_bytes(
            bytes[27..35]
                .try_into()
                .map_err(|_| "deadline_wire_length_invalid")?,
        ),
        role,
        game: u32::from_be_bytes(
            bytes[35..39]
                .try_into()
                .map_err(|_| "deadline_wire_length_invalid")?,
        ),
        generation: u64::from_be_bytes(
            bytes[39..47]
                .try_into()
                .map_err(|_| "deadline_wire_length_invalid")?,
        ),
        request_id: u64::from_be_bytes(
            bytes[47..55]
                .try_into()
                .map_err(|_| "deadline_wire_length_invalid")?,
        ),
        lock_sha256: bytes[55..87]
            .try_into()
            .map_err(|_| "deadline_wire_length_invalid")?,
        phase,
    };
    scope.validate()?;
    Ok(scope)
}

impl DeadlineRequest {
    pub fn encode(&self) -> Result<[u8; REQUEST_BYTES], &'static str> {
        scope_bytes(&self.scope, 1)
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() != REQUEST_BYTES {
            return Err("deadline_wire_length_invalid");
        }
        Ok(Self {
            scope: read_scope(bytes, 1)?,
        })
    }
}

impl DeadlineGrant {
    pub fn encode(&self) -> Result<[u8; GRANT_BYTES], &'static str> {
        if self.remaining_ns == 0 {
            return Err("deadline_expired");
        }
        let mut out = [0; GRANT_BYTES];
        out[..REQUEST_BYTES].copy_from_slice(&scope_bytes(&self.scope, 2)?);
        out[REQUEST_BYTES..].copy_from_slice(&self.remaining_ns.to_be_bytes());
        Ok(out)
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() != GRANT_BYTES {
            return Err("deadline_wire_length_invalid");
        }
        let scope = read_scope(bytes, 2)?;
        let remaining_ns = u64::from_be_bytes(
            bytes[REQUEST_BYTES..]
                .try_into()
                .map_err(|_| "deadline_wire_length_invalid")?,
        );
        if remaining_ns == 0 {
            return Err("deadline_expired");
        }
        Ok(Self {
            scope,
            remaining_ns,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope() -> DeadlineScope {
        DeadlineScope {
            owner_id: [1; 16],
            owner_epoch: 2,
            role: DeadlineRole::Candidate,
            game: 1,
            generation: 3,
            request_id: 4,
            lock_sha256: [5; 32],
            phase: DeadlinePhase::Run,
        }
    }

    #[test]
    fn request_and_grant_round_trip_without_cross_decoding() {
        let request = DeadlineRequest { scope: scope() };
        let grant = DeadlineGrant {
            scope: scope(),
            remaining_ns: 42,
        };
        assert_eq!(
            DeadlineRequest::decode(&request.encode().unwrap()),
            Ok(request.clone())
        );
        assert_eq!(
            DeadlineGrant::decode(&grant.encode().unwrap()),
            Ok(grant.clone())
        );
        assert!(DeadlineGrant::decode(&request.encode().unwrap()).is_err());
        assert!(DeadlineRequest::decode(&grant.encode().unwrap()).is_err());
    }

    #[test]
    fn invalid_owner_epoch_and_empty_grant_are_rejected() {
        let mut s = scope();
        s.owner_id = [0; 16];
        assert!(DeadlineRequest { scope: s }.encode().is_err());
        let mut s = scope();
        s.owner_epoch = 0;
        assert!(DeadlineRequest { scope: s }.encode().is_err());
        assert!(DeadlineGrant {
            scope: scope(),
            remaining_ns: 0
        }
        .encode()
        .is_err());
    }
}
