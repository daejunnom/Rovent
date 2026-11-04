//! Same-child, observation-only request and response frames.
//!
//! These bytes authenticate no peer and carry no authority to enter native
//! code, extend an original run/cleanup deadline, or release a GPU resource.
//! Sequence ordering and comparison with the original child/lock are the
//! caller's responsibility. The original domain bytes are included verbatim.

pub const CHILD_REOBSERVE_DOMAIN: &str = "rz-pals-child-reobserve/1";
pub const CHILD_REOBSERVE_MAX_ERROR_BYTES: usize = 512;

const KIND: usize = CHILD_REOBSERVE_DOMAIN.len();
const GENERATION: usize = KIND + 1;
const LOCK: usize = GENERATION + 8;
const LOCK_BYTES: usize = 64;
const SEQUENCE: usize = LOCK + LOCK_BYTES;
const BODY: usize = SEQUENCE + 8;
pub const CHILD_REOBSERVE_REQUEST_BYTES: usize = BODY + 8;
const RESPONSE_STATUS: usize = BODY;
const RESPONSE_ELAPSED: usize = RESPONSE_STATUS + 1;
const RESPONSE_ERROR_PRESENT: usize = RESPONSE_ELAPSED + 1;
const RESPONSE_ERROR_LENGTH: usize = RESPONSE_ERROR_PRESENT + 1;
const RESPONSE_ERROR: usize = RESPONSE_ERROR_LENGTH + 2;
pub const CHILD_REOBSERVE_RESPONSE_BYTES: usize = RESPONSE_ERROR + CHILD_REOBSERVE_MAX_ERROR_BYTES;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChildReobserveRequestV1 {
    pub generation: u64,
    pub lock_sha256: String,
    pub sequence: u64,
    /// Duration for this observation only. Zero permits an immediate poll.
    pub observation_budget_ms: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChildReobserveStatusV1 {
    Pending,
    Unknown,
    BeforeNativeLate,
    NativeFinishedLate,
}

/// A status from the same child, not an independently verified physical
/// completion. In particular, neither late status is a device-wide fence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChildReobserveResponseV1 {
    pub generation: u64,
    pub lock_sha256: String,
    pub sequence: u64,
    pub status: ChildReobserveStatusV1,
    pub original_cleanup_elapsed: bool,
    pub original_error: Option<String>,
}

fn validate_scope(generation: u64, lock: &str, sequence: u64) -> Result<(), &'static str> {
    if generation == 0
        || sequence == 0
        || lock.len() != LOCK_BYTES
        || !lock.as_bytes().iter().all(u8::is_ascii_hexdigit)
    {
        return Err("child_reobserve_scope_invalid");
    }
    Ok(())
}

fn encode_scope(bytes: &mut [u8], kind: u8, generation: u64, lock: &str, sequence: u64) {
    bytes[..KIND].copy_from_slice(CHILD_REOBSERVE_DOMAIN.as_bytes());
    bytes[KIND] = kind;
    bytes[GENERATION..LOCK].copy_from_slice(&generation.to_be_bytes());
    bytes[LOCK..SEQUENCE].copy_from_slice(lock.as_bytes());
    bytes[SEQUENCE..BODY].copy_from_slice(&sequence.to_be_bytes());
}

fn decode_scope(bytes: &[u8], kind: u8) -> Result<(u64, String, u64), &'static str> {
    if bytes.len() < BODY
        || &bytes[..KIND] != CHILD_REOBSERVE_DOMAIN.as_bytes()
        || bytes[KIND] != kind
    {
        return Err("child_reobserve_header_invalid");
    }
    let generation = u64::from_be_bytes(
        bytes[GENERATION..LOCK]
            .try_into()
            .map_err(|_| "child_reobserve_length_invalid")?,
    );
    let lock = std::str::from_utf8(&bytes[LOCK..SEQUENCE])
        .map_err(|_| "child_reobserve_scope_invalid")?
        .to_owned();
    let sequence = u64::from_be_bytes(
        bytes[SEQUENCE..BODY]
            .try_into()
            .map_err(|_| "child_reobserve_length_invalid")?,
    );
    validate_scope(generation, &lock, sequence)?;
    Ok((generation, lock, sequence))
}

impl ChildReobserveRequestV1 {
    pub fn encode(&self) -> Result<[u8; CHILD_REOBSERVE_REQUEST_BYTES], &'static str> {
        validate_scope(self.generation, &self.lock_sha256, self.sequence)?;
        let mut bytes = [0; CHILD_REOBSERVE_REQUEST_BYTES];
        // Kind 1 has exactly one meaning: observe. No control action is encoded.
        encode_scope(
            &mut bytes,
            1,
            self.generation,
            &self.lock_sha256,
            self.sequence,
        );
        bytes[BODY..].copy_from_slice(&self.observation_budget_ms.to_be_bytes());
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() != CHILD_REOBSERVE_REQUEST_BYTES {
            return Err("child_reobserve_length_invalid");
        }
        let (generation, lock_sha256, sequence) = decode_scope(bytes, 1)?;
        let observation_budget_ms = u64::from_be_bytes(
            bytes[BODY..]
                .try_into()
                .map_err(|_| "child_reobserve_length_invalid")?,
        );
        Ok(Self {
            generation,
            lock_sha256,
            sequence,
            observation_budget_ms,
        })
    }
}

impl ChildReobserveResponseV1 {
    /// Check only the declared request scope. The actual same-parent peer and
    /// original owner still require independent authentication by the caller.
    pub fn matches_request(&self, request: &ChildReobserveRequestV1) -> bool {
        validate_scope(self.generation, &self.lock_sha256, self.sequence).is_ok()
            && validate_scope(request.generation, &request.lock_sha256, request.sequence).is_ok()
            && self.generation == request.generation
            && self.sequence == request.sequence
            && self.lock_sha256.eq_ignore_ascii_case(&request.lock_sha256)
    }

    pub fn encode(&self) -> Result<[u8; CHILD_REOBSERVE_RESPONSE_BYTES], &'static str> {
        validate_scope(self.generation, &self.lock_sha256, self.sequence)?;
        let error = self.original_error.as_ref().map(|value| value.as_bytes());
        if error.is_some_and(|value| value.len() > CHILD_REOBSERVE_MAX_ERROR_BYTES) {
            return Err("child_reobserve_error_too_long");
        }
        let mut bytes = [0; CHILD_REOBSERVE_RESPONSE_BYTES];
        encode_scope(
            &mut bytes,
            2,
            self.generation,
            &self.lock_sha256,
            self.sequence,
        );
        bytes[RESPONSE_STATUS] = match self.status {
            ChildReobserveStatusV1::Pending => 1,
            ChildReobserveStatusV1::Unknown => 2,
            ChildReobserveStatusV1::BeforeNativeLate => 3,
            ChildReobserveStatusV1::NativeFinishedLate => 4,
        };
        bytes[RESPONSE_ELAPSED] = u8::from(self.original_cleanup_elapsed);
        if let Some(error) = error {
            bytes[RESPONSE_ERROR_PRESENT] = 1;
            bytes[RESPONSE_ERROR_LENGTH..RESPONSE_ERROR]
                .copy_from_slice(&(error.len() as u16).to_be_bytes());
            bytes[RESPONSE_ERROR..RESPONSE_ERROR + error.len()].copy_from_slice(error);
        }
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() != CHILD_REOBSERVE_RESPONSE_BYTES {
            return Err("child_reobserve_length_invalid");
        }
        let (generation, lock_sha256, sequence) = decode_scope(bytes, 2)?;
        let status = match bytes[RESPONSE_STATUS] {
            1 => ChildReobserveStatusV1::Pending,
            2 => ChildReobserveStatusV1::Unknown,
            3 => ChildReobserveStatusV1::BeforeNativeLate,
            4 => ChildReobserveStatusV1::NativeFinishedLate,
            _ => return Err("child_reobserve_status_invalid"),
        };
        let original_cleanup_elapsed = match bytes[RESPONSE_ELAPSED] {
            0 => false,
            1 => true,
            _ => return Err("child_reobserve_elapsed_invalid"),
        };
        let length = u16::from_be_bytes(
            bytes[RESPONSE_ERROR_LENGTH..RESPONSE_ERROR]
                .try_into()
                .map_err(|_| "child_reobserve_length_invalid")?,
        ) as usize;
        if length > CHILD_REOBSERVE_MAX_ERROR_BYTES
            || bytes[RESPONSE_ERROR + length..]
                .iter()
                .any(|byte| *byte != 0)
        {
            return Err("child_reobserve_error_invalid");
        }
        let original_error = match bytes[RESPONSE_ERROR_PRESENT] {
            0 if length == 0 => None,
            1 => Some(
                std::str::from_utf8(&bytes[RESPONSE_ERROR..RESPONSE_ERROR + length])
                    .map_err(|_| "child_reobserve_error_invalid")?
                    .to_owned(),
            ),
            _ => return Err("child_reobserve_error_invalid"),
        };
        Ok(Self {
            generation,
            lock_sha256,
            sequence,
            status,
            original_cleanup_elapsed,
            original_error,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lock() -> String {
        "aB".repeat(32)
    }

    #[test]
    fn roundtrips_scopes_statuses_and_bounded_original_errors() {
        let request = ChildReobserveRequestV1 {
            generation: 7,
            lock_sha256: lock(),
            sequence: 1,
            observation_budget_ms: 0,
        };
        let encoded = request.encode().unwrap();
        assert_eq!(ChildReobserveRequestV1::decode(&encoded).unwrap(), request);
        for status in [
            ChildReobserveStatusV1::Pending,
            ChildReobserveStatusV1::Unknown,
            ChildReobserveStatusV1::BeforeNativeLate,
            ChildReobserveStatusV1::NativeFinishedLate,
        ] {
            for error in [
                None,
                Some(String::new()),
                Some("오류".to_owned()),
                Some("x".repeat(CHILD_REOBSERVE_MAX_ERROR_BYTES)),
            ] {
                let response = ChildReobserveResponseV1 {
                    generation: request.generation,
                    lock_sha256: request.lock_sha256.clone(),
                    sequence: request.sequence,
                    status,
                    original_cleanup_elapsed: true,
                    original_error: error,
                };
                let frame = response.encode().unwrap();
                assert_eq!(ChildReobserveResponseV1::decode(&frame).unwrap(), response);
                assert!(ChildReobserveRequestV1::decode(&frame).is_err());
                assert!(ChildReobserveResponseV1::decode(&encoded).is_err());
            }
        }
    }

    #[test]
    fn rejects_invalid_scope_header_length_status_and_error_padding() {
        let request = ChildReobserveRequestV1 {
            generation: 1,
            lock_sha256: lock(),
            sequence: 1,
            observation_budget_ms: 10,
        };
        let frame = request.encode().unwrap();
        assert!(ChildReobserveRequestV1::decode(&frame[..frame.len() - 1]).is_err());
        let mut wrong = frame;
        wrong[KIND] = 2;
        assert!(ChildReobserveRequestV1::decode(&wrong).is_err());
        let mut wrong = frame;
        wrong[0] ^= 1;
        assert!(ChildReobserveRequestV1::decode(&wrong).is_err());
        let mut invalid = request.clone();
        invalid.generation = 0;
        assert!(invalid.encode().is_err());
        invalid = request.clone();
        invalid.sequence = 0;
        assert!(invalid.encode().is_err());
        invalid = request.clone();
        invalid.lock_sha256.replace_range(..1, "z");
        assert!(invalid.encode().is_err());
        let mut response = ChildReobserveResponseV1 {
            generation: 1,
            lock_sha256: lock(),
            sequence: 1,
            status: ChildReobserveStatusV1::Unknown,
            original_cleanup_elapsed: false,
            original_error: None,
        };
        let frame = response.encode().unwrap();
        for index in [RESPONSE_STATUS, RESPONSE_ELAPSED, RESPONSE_ERROR_PRESENT] {
            let mut bad = frame;
            bad[index] = 0xff;
            assert!(ChildReobserveResponseV1::decode(&bad).is_err());
        }
        let mut bad = frame;
        bad[RESPONSE_ERROR] = 1;
        assert!(ChildReobserveResponseV1::decode(&bad).is_err());
        let mut bad = frame;
        bad[RESPONSE_ERROR_PRESENT] = 1;
        bad[RESPONSE_ERROR_LENGTH + 1] = 1;
        bad[RESPONSE_ERROR] = 0xff;
        assert!(ChildReobserveResponseV1::decode(&bad).is_err());
        response.original_error = Some("x".repeat(CHILD_REOBSERVE_MAX_ERROR_BYTES + 1));
        assert!(response.encode().is_err());
    }

    #[test]
    fn caller_compares_the_original_scope_without_case_sensitive_digest_errors() {
        let request = ChildReobserveRequestV1 {
            generation: 3,
            lock_sha256: lock(),
            sequence: 12,
            observation_budget_ms: 50,
        };
        let mut response = ChildReobserveResponseV1 {
            generation: request.generation,
            lock_sha256: request.lock_sha256.to_ascii_uppercase(),
            sequence: request.sequence,
            status: ChildReobserveStatusV1::Unknown,
            original_cleanup_elapsed: false,
            original_error: None,
        };
        assert!(response.matches_request(&request));
        response.sequence += 1;
        assert!(!response.matches_request(&request));
        response.sequence = request.sequence;
        response.generation += 1;
        assert!(!response.matches_request(&request));
        response.generation = request.generation;
        response.lock_sha256 = "0".repeat(64);
        assert!(!response.matches_request(&request));
    }

    #[test]
    fn existing_grant_wire_is_not_reinterpreted_as_reobservation() {
        use crate::pals::{
            deadline_grant_v1::DeadlineRole,
            proxy_entry_smoke_grant_v1::{
                ProxyEntrySmokePhase, ProxyEntrySmokePolicy, ProxyEntrySmokeRequest,
                ProxyEntrySmokeScope,
            },
        };
        let old = ProxyEntrySmokeRequest {
            scope: ProxyEntrySmokeScope {
                policy: ProxyEntrySmokePolicy::ProxyEntryFunctionalSmokeV1,
                owner_id: [1; 16],
                owner_epoch: 1,
                role: DeadlineRole::Baseline,
                generation: 3,
                request_id: 12,
                lock_sha256: [2; 32],
                phase: ProxyEntrySmokePhase::Run,
            },
        }
        .encode()
        .unwrap();
        let request = ChildReobserveRequestV1 {
            generation: 3,
            lock_sha256: lock(),
            sequence: 12,
            observation_budget_ms: 50,
        }
        .encode()
        .unwrap();
        assert!(ChildReobserveRequestV1::decode(&old).is_err());
        assert!(ProxyEntrySmokeRequest::decode(&request).is_err());
    }
}
