//! Persisted Rovent root descriptors, independent of live search authority.
//!
//! This codec checks declarations, joins and their canonical seals only. It
//! cannot check chess legality or Rules move order, issue a clock/root owner,
//! observe applied resources, or attest neural execution. The actual UCI owner
//! must retain and validate its original Rules, go, clock and revocation state.

use crate::{
    ManifestError, ROVENT_POSITION_VISIT_UNIT, RoventCoordinatorDeclaration, RoventInputLock,
    RoventInputLockV2, RoventResourcePlan, RoventSearchDeclaration,
};
use rz_contracts::{Digest, rovent::SearchGoal};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::{collections::BTreeSet, io::Write, num::NonZeroU16};

pub const ROVENT_ROOT_ADMISSION_DOMAIN: &str = "rz-rovent-root-admission/1";
pub const ROVENT_ROOT_BUDGET_DOMAIN: &str = "rz-rovent-root-budget/1";
pub const ROVENT_ROOT_DEADLINE_DOMAIN: &str = "rz-rovent-root-deadline/1";
pub const ROVENT_ROOT_ADMISSION_V2_DOMAIN: &str = "rz-rovent-root-admission/2";
pub const ROVENT_ROOT_BUDGET_V2_DOMAIN: &str = "rz-rovent-root-budget/2";
pub const ROVENT_ROOT_DEADLINE_V2_DOMAIN: &str = "rz-rovent-root-deadline/2";
const MAX_DESCRIPTOR_BYTES: usize = 64 << 10;
const MAX_RAW_MOVES_BYTES: usize = 32 << 10;
const MAX_ROOT_MOVES: usize = 218;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RoventRootAdmissionPolicy {
    ExactRegistered,
    DynamicGo,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RoventOriginalGoCoverage {
    SemanticContextOnly,
    FullOriginalGo,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventRootAuthority {
    pub epoch: u64,
    pub game: u64,
    pub root: u64,
    pub implementation_sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventRootDeadline {
    pub clock_epoch: u64,
    pub at_ns: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventRootClocks {
    pub hard: RoventRootDeadline,
    pub soft: RoventRootDeadline,
    pub admission: RoventRootDeadline,
    pub cleanup: RoventRootDeadline,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventParsedGo {
    /// Received order and repetitions are preserved; this is not Rules order.
    pub searchmoves: Option<Vec<String>>,
    pub depth_plies: Option<NonZeroU16>,
    pub mate_moves: Option<NonZeroU16>,
    pub movetime_ms: Option<u64>,
    pub white_time_ms: Option<u64>,
    pub black_time_ms: Option<u64>,
    pub white_increment_ms: Option<u64>,
    pub black_increment_ms: Option<u64>,
    pub moves_to_go: Option<u32>,
    pub nodes: Option<u64>,
    pub infinite: bool,
    pub ponder: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventResolvedGo {
    pub coverage: RoventOriginalGoCoverage,
    pub requested_nodes: Option<u64>,
    pub depth_plies: Option<NonZeroU16>,
    pub mate_moves: Option<NonZeroU16>,
    /// Distinct moves in the actual consumer's declared order. This codec does
    /// not prove that order came from Rules.
    pub searchmoves: Option<Vec<String>>,
    pub transport_simulation_cap: u64,
    pub original_parsed_limits: Option<RoventParsedGo>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventRootDeclaredCeilings {
    /// Mode/actor/backend semantics are checked by the original lock validator
    /// during verify_against; a standalone descriptor is not a resource lease.
    pub resource_plan: RoventResourcePlan,
    pub registered_max_chess_plies: NonZeroU16,
    pub cpu_credit_limit: u64,
    pub questions_max: u32,
    /// Matches RoventCoordinatorDeclaration's existing u32 field.
    pub ancestor_depth_max: u32,
    pub waiting_jobs_max: u32,
    pub role_tickets_max: u32,
    pub wall_time_max_ms: u64,
    pub cleanup_max_ms: u64,
    pub output_bytes_max: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventRootDescriptorInput {
    pub policy: RoventRootAdmissionPolicy,
    pub startup_lock_sha256: String,
    pub state_and_history_sha256: String,
    pub rules_profile_sha256: String,
    pub authority: RoventRootAuthority,
    pub node_unit: String,
    pub search_configuration_sha256: String,
    pub coordinator_policy_sha256: String,
    pub original_go: RoventResolvedGo,
    pub allowed_root_moves: Vec<String>,
    pub clock: RoventRootClocks,
    pub declared_ceilings: RoventRootDeclaredCeilings,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventRootDescriptor {
    pub domain: String,
    pub input: RoventRootDescriptorInput,
    pub root_budget_sha256: String,
    pub deadline_seal_sha256: String,
}

impl RoventRootDescriptor {
    pub fn seal(input: RoventRootDescriptorInput) -> Result<Self, ManifestError> {
        let (root_budget_sha256, deadline_seal_sha256) = seals(&input)?;
        let descriptor = Self {
            domain: ROVENT_ROOT_ADMISSION_DOMAIN.into(),
            input,
            root_budget_sha256,
            deadline_seal_sha256,
        };
        // Also bound the complete wrapper, not only its individual seal inputs.
        canonical(&descriptor)?;
        Ok(descriptor)
    }

    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        require(
            input.len() <= MAX_DESCRIPTOR_BYTES,
            "root descriptor input exceeds 64 KiB",
        )?;
        // The shared decoder rejects duplicate keys recursively before deriving
        // this closed DTO; normal serde_json::Value decoding would lose them.
        let descriptor: Self = crate::decode_json(input)?;
        descriptor.validate()?;
        Ok(descriptor)
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            self.domain == ROVENT_ROOT_ADMISSION_DOMAIN
                && sha(&self.root_budget_sha256)
                && sha(&self.deadline_seal_sha256),
            "root descriptor domain or seal syntax differs",
        )?;
        let (budget, deadline) = seals(&self.input)?;
        require(
            budget == self.root_budget_sha256 && deadline == self.deadline_seal_sha256,
            "root descriptor seals differ from their canonical inputs",
        )?;
        canonical(self)?;
        Ok(())
    }

    /// Joins declarations to the original validated lock. Matching metadata
    /// does not establish actual source invocation, live authority or Rules.
    pub fn verify_against(&self, lock: &RoventInputLock) -> Result<(), ManifestError> {
        lock.validate()?;
        self.validate()?;
        let manifest = &lock.manifest;
        verify_declarations(
            &self.input,
            &lock.canonical_sha256,
            &manifest.engine.search,
            &manifest.coordinator,
            &manifest.resources,
            manifest.wall_time_max_ms,
            manifest.cleanup_max_ms,
            manifest.output_bytes_max,
        )
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        self.validate()?;
        canonical(self)
    }

    pub fn digest(&self) -> Result<Digest, ManifestError> {
        Ok(domain_hash(
            ROVENT_ROOT_ADMISSION_DOMAIN,
            &self.canonical_bytes()?,
        ))
    }

    pub fn root_budget_digest(&self) -> Result<Digest, ManifestError> {
        self.validate()?;
        parse_digest(&self.root_budget_sha256)
    }

    pub fn deadline_seal_digest(&self) -> Result<Digest, ManifestError> {
        self.validate()?;
        parse_digest(&self.deadline_seal_sha256)
    }
}

/// A distinct /2 wire domain for declarations joined to a typed /2 lock.
/// It is metadata only and cannot grant a live root or observation authority.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventRootDescriptorV2 {
    pub domain: String,
    pub startup_lock_domain: String,
    pub input: RoventRootDescriptorInput,
    pub root_budget_sha256: String,
    pub deadline_seal_sha256: String,
}

impl RoventRootDescriptorV2 {
    pub fn seal(
        input: RoventRootDescriptorInput,
        lock: &RoventInputLockV2,
    ) -> Result<Self, ManifestError> {
        let (root_budget_sha256, deadline_seal_sha256) = seals_for(
            &input,
            ROVENT_ROOT_BUDGET_V2_DOMAIN,
            ROVENT_ROOT_DEADLINE_V2_DOMAIN,
        )?;
        let value = Self {
            domain: ROVENT_ROOT_ADMISSION_V2_DOMAIN.into(),
            startup_lock_domain: crate::ROVENT_LOCK_V2_DOMAIN.into(),
            input,
            root_budget_sha256,
            deadline_seal_sha256,
        };
        value.verify_against(lock)?;
        canonical(&value)?;
        Ok(value)
    }

    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        require(
            input.len() <= MAX_DESCRIPTOR_BYTES,
            "root descriptor input exceeds 64 KiB",
        )?;
        let value: Self = crate::decode_json(input)?;
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            self.domain == ROVENT_ROOT_ADMISSION_V2_DOMAIN
                && self.startup_lock_domain == crate::ROVENT_LOCK_V2_DOMAIN
                && sha(&self.root_budget_sha256)
                && sha(&self.deadline_seal_sha256),
            "root /2 descriptor domain or seal syntax differs",
        )?;
        let (budget, deadline) = seals_for(
            &self.input,
            ROVENT_ROOT_BUDGET_V2_DOMAIN,
            ROVENT_ROOT_DEADLINE_V2_DOMAIN,
        )?;
        require(
            budget == self.root_budget_sha256 && deadline == self.deadline_seal_sha256,
            "root /2 descriptor seals differ from their canonical inputs",
        )?;
        canonical(self)?;
        Ok(())
    }

    /// Compares declarations only. The caller must separately preserve and
    /// validate the real owner, Rules state, clock and revocation gate.
    pub fn verify_against(&self, lock: &RoventInputLockV2) -> Result<(), ManifestError> {
        lock.validate()?;
        self.validate()?;
        let manifest = &lock.manifest;
        verify_declarations(
            &self.input,
            &lock.canonical_sha256,
            &manifest.engine.search,
            &manifest.coordinator,
            &manifest.resources,
            manifest.wall_time_max_ms,
            manifest.cleanup_max_ms,
            manifest.output_bytes_max,
        )
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        self.validate()?;
        canonical(self)
    }

    pub fn digest(&self) -> Result<Digest, ManifestError> {
        Ok(domain_hash(
            ROVENT_ROOT_ADMISSION_V2_DOMAIN,
            &self.canonical_bytes()?,
        ))
    }

    pub fn root_budget_digest(&self) -> Result<Digest, ManifestError> {
        self.validate()?;
        parse_digest(&self.root_budget_sha256)
    }

    pub fn deadline_seal_digest(&self) -> Result<Digest, ManifestError> {
        self.validate()?;
        parse_digest(&self.deadline_seal_sha256)
    }
}

fn verify_declarations(
    input: &RoventRootDescriptorInput,
    canonical_sha256: &str,
    search: &RoventSearchDeclaration,
    coordinator: &RoventCoordinatorDeclaration,
    resources: &RoventResourcePlan,
    wall_time_max_ms: u64,
    cleanup_max_ms: u64,
    output_bytes_max: u64,
) -> Result<(), ManifestError> {
    let ceilings = &input.declared_ceilings;
    require(
        input.startup_lock_sha256 == canonical_sha256
            && input.node_unit == search.node_unit
            && input.search_configuration_sha256 == search.search_configuration_sha256
            && input.coordinator_policy_sha256 == coordinator.policy_sha256
            && input.rules_profile_sha256 == search.rules_profile_sha256,
        "root descriptor differs from startup source/configuration/Rules declarations",
    )?;
    require(
        ceilings.resource_plan == *resources
            && ceilings.registered_max_chess_plies == search.registered_max_chess_plies
            && ceilings.cpu_credit_limit == coordinator.cpu_credit_limit
            && ceilings.questions_max == coordinator.questions_max
            && ceilings.ancestor_depth_max == coordinator.ancestor_depth_max
            && ceilings.waiting_jobs_max == coordinator.waiting_jobs_max
            && ceilings.role_tickets_max == coordinator.role_tickets_max
            && ceilings.wall_time_max_ms == wall_time_max_ms
            && ceilings.cleanup_max_ms == cleanup_max_ms
            && ceilings.output_bytes_max == output_bytes_max,
        "root descriptor changes an original declared ceiling or resource plan",
    )?;
    if input.policy == RoventRootAdmissionPolicy::ExactRegistered {
        require(
            input.state_and_history_sha256 == search.root_state_and_history_sha256
                && input.original_go.requested_nodes == search.requested_nodes
                && input.original_go.depth_plies == search.depth_plies
                && input.original_go.mate_moves == search.mate_moves
                && same_optional_move_set(
                    input.original_go.searchmoves.as_deref(),
                    search.searchmoves.as_deref(),
                ),
            "root descriptor differs from the exact registered workload",
        )?;
    }
    // DynamicGo intentionally inherits only startup source/assets/runtime,
    // resources and profiles via the exact lock, never its workload label.
    Ok(())
}

fn seals(input: &RoventRootDescriptorInput) -> Result<(String, String), ManifestError> {
    seals_for(
        input,
        ROVENT_ROOT_BUDGET_DOMAIN,
        ROVENT_ROOT_DEADLINE_DOMAIN,
    )
}

fn seals_for(
    input: &RoventRootDescriptorInput,
    budget_domain: &str,
    deadline_domain: &str,
) -> Result<(String, String), ManifestError> {
    validate_input(input)?;
    // Serialize once within the byte bound before constructing a Value. Clock
    // is the only excluded input: raw go and every declared ceiling stay bound.
    let mut stable: Value = serde_json::from_slice(&canonical(input)?)
        .map_err(|error| ManifestError::Parse(error.to_string()))?;
    stable
        .as_object_mut()
        .ok_or_else(|| integrity("root input is not an object"))?
        .remove("clock");
    let budget = hex(domain_hash(budget_domain, &canonical(&stable)?));
    let deadline = json!({
        "domain": deadline_domain,
        "startup_lock_sha256": input.startup_lock_sha256,
        "policy": input.policy,
        "authority": input.authority,
        "root_budget_sha256": budget,
        "clock": input.clock,
        "cleanup_max_ms": input.declared_ceilings.cleanup_max_ms,
    });
    let deadline = hex(domain_hash(deadline_domain, &canonical(&deadline)?));
    Ok((budget, deadline))
}

fn validate_input(input: &RoventRootDescriptorInput) -> Result<(), ManifestError> {
    require(
        [
            &input.startup_lock_sha256,
            &input.state_and_history_sha256,
            &input.rules_profile_sha256,
            &input.authority.implementation_sha256,
            &input.search_configuration_sha256,
            &input.coordinator_policy_sha256,
        ]
        .into_iter()
        .all(|value| sha(value))
            && input.authority.epoch != 0
            && input.node_unit == ROVENT_POSITION_VISIT_UNIT,
        "root descriptor needs exact SHA/domain/authority declarations",
    )?;
    let ceilings = &input.declared_ceilings;
    require(
        (1..=65_536).contains(&ceilings.questions_max)
            && (1..=1024).contains(&ceilings.ancestor_depth_max)
            && ceilings.ancestor_depth_max <= ceilings.questions_max
            && (1..=1024).contains(&ceilings.waiting_jobs_max)
            && (1..=1024).contains(&ceilings.role_tickets_max)
            && (1..=86_400_000).contains(&ceilings.wall_time_max_ms)
            && (1..=30_000).contains(&ceilings.cleanup_max_ms)
            && (1..=64_u64 << 30).contains(&ceilings.output_bytes_max),
        "root descriptor changes the bounded declaration profile",
    )?;
    SearchGoal {
        depth_plies: input.original_go.depth_plies,
        mate_moves: input.original_go.mate_moves,
    }
    .validate(ceilings.registered_max_chess_plies)
    .map_err(|error| integrity(&format!("root descriptor goal: {error}")))?;
    validate_distinct_moves(&input.allowed_root_moves, true)?;
    if let Some(moves) = &input.original_go.searchmoves {
        validate_distinct_moves(moves, false)?;
        require(
            moves == &input.allowed_root_moves,
            "resolved root restriction and allowed move order differ",
        )?;
    }
    match (
        input.original_go.coverage,
        &input.original_go.original_parsed_limits,
    ) {
        (RoventOriginalGoCoverage::SemanticContextOnly, None) => {}
        (RoventOriginalGoCoverage::FullOriginalGo, Some(raw)) => {
            require(
                raw.nodes == input.original_go.requested_nodes
                    && raw.depth_plies == input.original_go.depth_plies
                    && raw.mate_moves == input.original_go.mate_moves,
                "original go nodes or goal differs from the resolved declaration",
            )?;
            if let Some(moves) = &raw.searchmoves {
                let bytes = moves.iter().try_fold(0usize, |sum, movement| {
                    sum.checked_add(movement.len())?.checked_add(3)
                });
                require(
                    !moves.is_empty()
                        && bytes.is_some_and(|bytes| bytes <= MAX_RAW_MOVES_BYTES)
                        && moves.iter().all(|movement| move_syntax(movement)),
                    "original root moves exceed the raw syntax/byte bound",
                )?;
                require(
                    moves.iter().collect::<BTreeSet<_>>().len() <= MAX_ROOT_MOVES,
                    "original root restriction exceeds the distinct move bound",
                )?;
            }
            require(
                same_optional_move_set(
                    raw.searchmoves.as_deref(),
                    input.original_go.searchmoves.as_deref(),
                ),
                "original go root set differs from the resolved declaration",
            )?;
        }
        _ => {
            return Err(integrity(
                "root descriptor original go coverage is inconsistent",
            ));
        }
    }
    let clock = &input.clock;
    require(
        [clock.hard, clock.soft, clock.admission, clock.cleanup]
            .into_iter()
            .all(|deadline| deadline.clock_epoch == input.authority.epoch)
            && clock.soft.at_ns <= clock.admission.at_ns
            && clock.admission.at_ns <= clock.hard.at_ns
            && clock.hard.at_ns <= clock.cleanup.at_ns,
        "root clock domains or original deadline order differ",
    )?;
    let cleanup_max_ns = ceilings
        .cleanup_max_ms
        .checked_mul(1_000_000)
        .ok_or_else(|| integrity("cleanup ceiling overflows nanoseconds"))?;
    require(
        clock
            .cleanup
            .at_ns
            .checked_sub(clock.hard.at_ns)
            .is_some_and(|duration| duration <= cleanup_max_ns),
        "root cleanup exceeds its declared nanosecond ceiling",
    )
}

fn same_optional_move_set(left: Option<&[String]>, right: Option<&[String]>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            left.iter().collect::<BTreeSet<_>>() == right.iter().collect::<BTreeSet<_>>()
        }
        _ => false,
    }
}

fn validate_distinct_moves(moves: &[String], empty_allowed: bool) -> Result<(), ManifestError> {
    require(
        (empty_allowed || !moves.is_empty())
            && moves.len() <= MAX_ROOT_MOVES
            && moves.iter().all(|movement| move_syntax(movement))
            && moves.iter().collect::<BTreeSet<_>>().len() == moves.len(),
        "resolved root moves must be distinct bounded UCI tokens",
    )
}

fn move_syntax(movement: &str) -> bool {
    let bytes = movement.as_bytes();
    matches!(bytes.len(), 4 | 5)
        && (b'a'..=b'h').contains(&bytes[0])
        && (b'1'..=b'8').contains(&bytes[1])
        && (b'a'..=b'h').contains(&bytes[2])
        && (b'1'..=b'8').contains(&bytes[3])
        && bytes[..2] != bytes[2..4]
        && (bytes.len() == 4 || matches!(bytes[4], b'q' | b'r' | b'b' | b'n'))
}

fn sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn parse_digest(value: &str) -> Result<Digest, ManifestError> {
    require(sha(value), "root seal is not lowercase SHA-256")?;
    let mut digest = [0; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let nibble = |byte: u8| {
            if byte <= b'9' {
                byte - b'0'
            } else {
                byte - b'a' + 10
            }
        };
        digest[index] = (nibble(pair[0]) << 4) | nibble(pair[1]);
    }
    Ok(Digest(digest))
}

fn domain_hash(domain: &str, bytes: &[u8]) -> Digest {
    let mut hash = Sha256::new();
    hash.update(domain.as_bytes());
    hash.update([0]);
    hash.update(bytes);
    Digest(hash.finalize().into())
}

fn hex(digest: Digest) -> String {
    digest.0.iter().map(|byte| format!("{byte:02x}")).collect()
}

struct BoundedJson(Vec<u8>);
impl Write for BoundedJson {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self
            .0
            .len()
            .checked_add(bytes.len())
            .is_none_or(|size| size > MAX_DESCRIPTOR_BYTES)
        {
            return Err(std::io::Error::other("root descriptor exceeds 64 KiB"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn bounded_json(value: &impl Serialize) -> Result<Vec<u8>, ManifestError> {
    let mut output = BoundedJson(Vec::new());
    serde_json::to_writer(&mut output, value)
        .map_err(|error| integrity(&format!("bounded root JSON: {error}")))?;
    Ok(output.0)
}

fn sorted(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(key, value)| (key, sorted(value)))
                .collect::<std::collections::BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.into_iter().map(sorted).collect()),
        other => other,
    }
}

fn canonical(value: &impl Serialize) -> Result<Vec<u8>, ManifestError> {
    let value: Value = serde_json::from_slice(&bounded_json(value)?)
        .map_err(|error| ManifestError::Parse(error.to_string()))?;
    bounded_json(&sorted(value))
}

fn require(valid: bool, message: &str) -> Result<(), ManifestError> {
    if valid {
        Ok(())
    } else {
        Err(integrity(message))
    }
}
fn integrity(message: &str) -> ManifestError {
    ManifestError::Integrity(format!("Rovent root descriptor: {message}"))
}

#[cfg(test)]
#[path = "rovent_root_descriptor/tests.rs"]
mod tests;
