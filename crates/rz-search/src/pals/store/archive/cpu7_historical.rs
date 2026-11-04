//! Detached checks for an existing CPU7 view projection.
//!
//! This is deliberately not a /2 reader or a live admission path.  In
//! particular, caller-supplied hashes, this projection and its internal digest
//! cannot prove the original selection owner, its separate durable ACK, a
//! physical CPU completion, or eligibility for training.  Those sources are
//! not wired here, so every well-formed result remains historical `Unknown`.

use super::*;
use crate::pals::engine::RecordKind;
use crate::pals::review_context::{
    CohortCpu7OriginalRecord, MAX_LINE, cohort_cpu7_original_digest, cohort_cpu7_view_binding,
    snapshot_digest,
};
use rz_position::PositionLimits;

const MAX_PROJECTION_BYTES: usize = 16 * 1024 * 1024;
const MAX_PROJECTION_ENTRIES: usize = 65_536;
const MAX_HISTORY_CHARGE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HistoricalRecord {
    record: RoleRecordWire,
    evidence: Option<ObservationId>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HistoricalStore {
    schema: String,
    generation: u64,
    states: Vec<StateWire>,
    lines: Vec<LineWire>,
    observations: Vec<(ObservationId, ObservationWire)>,
    // These collections are bounded and retained as input, but their entire
    // dependency semantics are not replayed by this partial checker.
    situations: Vec<serde_json::Value>,
    executions: Vec<serde_json::Value>,
    dependencies: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HistoricalProjection {
    schema: String,
    bundle_id: u64,
    generation: u64,
    root: StateWire,
    anchor: StateWire,
    selected_len: usize,
    original_record_handle_digest: [u8; 32],
    view_binding: [u8; 32],
    view_digest: [u8; 32],
    view_seal: [u8; 32],
    policy_digest: [u8; 32],
    records: Vec<HistoricalRecord>,
    store: HistoricalStore,
}

/// Only detached observations. The absent original owner, registered /2
/// source pin, separate ACK, selected model input and full dependency replay
/// remain unknown even when every check implemented here succeeds.
#[derive(Debug, Eq, PartialEq)]
pub struct HistoricalCpu7Observation {
    pub selected_records: usize,
    pub original_records: usize,
    pub checked_cpu_observations: usize,
    pub original_record_handle_digest: [u8; 32],
    pub view_binding: [u8; 32],
    /// A byte-level join is not authentication of an original ACK owner.
    pub ack_content_join: &'static str,
    pub original_owner: &'static str,
    pub registration_and_ack: &'static str,
    pub source_record_evidence_semantics: &'static str,
    pub full_dependency_replay: &'static str,
    pub model_input_equivalence: &'static str,
    pub physical_completion: &'static str,
    pub training_authorized: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DiagnosticPin {
    path: String,
    sha256: String,
    bytes: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DiagnosticAuthority {
    process_epoch: u64,
    game_generation: u64,
    root_generation: u64,
    implementation_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DiagnosticAck {
    schema: String,
    cohort_id: String,
    trial_sequence: u64,
    authority: DiagnosticAuthority,
    trial_semantic_sha256: String,
    registration: DiagnosticPin,
    projection: DiagnosticPin,
    scope: String,
    physical_completion_attested: bool,
    training_authorized: bool,
}

/// Explicit caller-provided joins. These values are declarations, not a
/// verified live owner, durable publication, or permission to train.
pub struct Cpu7DiagnosticExpected<'a> {
    pub ack_sha256: [u8; 32],
    pub cohort_id: &'a str,
    pub trial_sequence: u64,
    pub process_epoch: u64,
    pub game_generation: u64,
    pub root_generation: u64,
    pub implementation_sha256: &'a str,
    pub trial_semantic_sha256: &'a str,
    pub registration_path: &'a str,
    pub projection_path: &'a str,
    /// Separate limits for this historical read. They are not a measurement
    /// of RSS and do not replace the original selection's registration.
    pub reader_max_projection_entries: usize,
    pub reader_max_dependency_edges: usize,
    pub reader_max_history_charge_bytes: usize,
}

fn lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn relative_pin_path(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 4096
        && !value.starts_with('/')
        && !value.contains('\\')
        && !value.contains(':')
        && !value.chars().any(char::is_control)
        && value
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
}

fn pin_matches(pin: &DiagnosticPin, path: &str, bytes: &[u8], digest: &str) -> bool {
    relative_pin_path(&pin.path)
        && pin.path == path
        && lowercase_sha256(&pin.sha256)
        && u64::try_from(bytes.len()).ok() == Some(pin.bytes)
        && !bytes.is_empty()
        && digest == pin.sha256
}

/// Join one detached diagnostic ACK to its raw projection and registration.
/// This does not open paths and cannot authenticate those bytes as the
/// original owner. The unimplemented registration semantics, original owner,
/// selection-to-trial link and physical lifetime remain `unknown`.
pub fn inspect_cpu7_diagnostic_ack(
    ack_bytes: &[u8],
    projection_bytes: &[u8],
    registration_bytes: &[u8],
    expected: &Cpu7DiagnosticExpected<'_>,
    check: &mut dyn FnMut() -> Result<(), StoreError>,
) -> Result<HistoricalCpu7Observation, StoreError> {
    const MAX_ACK_BYTES: usize = 64 * 1024;
    const MAX_REGISTRATION_BYTES: usize = 64 * 1024;
    check()?;
    if ack_bytes.is_empty()
        || ack_bytes.len() > MAX_ACK_BYTES
        || registration_bytes.is_empty()
        || registration_bytes.len() > MAX_REGISTRATION_BYTES
        || projection_bytes.is_empty()
        || projection_bytes.len() > MAX_PROJECTION_BYTES
    {
        return Err(StoreError::ArchiveByteBudget);
    }
    let ack_sha256: [u8; 32] = Sha256::digest(ack_bytes).into();
    if ack_sha256 != expected.ack_sha256 {
        return Err(StoreError::ArchiveIntegrity("CPU7 diagnostic ACK hash"));
    }
    let ack: DiagnosticAck = serde_json::from_slice(ack_bytes)
        .map_err(|_| StoreError::ArchiveIntegrity("CPU7 diagnostic ACK JSON"))?;
    check()?;
    let registration_digest = format!("{:x}", Sha256::digest(registration_bytes));
    check()?;
    let projection_digest = format!("{:x}", Sha256::digest(projection_bytes));
    check()?;
    if ack.schema != "rz-rovent-cpu7-diagnostic-projection-ack/1"
        || ack.scope != "selection_diagnostic_only"
        || ack.physical_completion_attested
        || ack.training_authorized
        || !lowercase_sha256(&ack.cohort_id)
        || ack.cohort_id != expected.cohort_id
        || ack.trial_sequence != expected.trial_sequence
        || ack.authority.process_epoch != expected.process_epoch
        || ack.authority.game_generation != expected.game_generation
        || ack.authority.root_generation != expected.root_generation
        || !lowercase_sha256(&ack.authority.implementation_sha256)
        || ack.authority.implementation_sha256 != expected.implementation_sha256
        || !lowercase_sha256(&ack.trial_semantic_sha256)
        || ack.trial_semantic_sha256 != expected.trial_semantic_sha256
        || !pin_matches(
            &ack.registration,
            expected.registration_path,
            registration_bytes,
            &registration_digest,
        )
        || !pin_matches(
            &ack.projection,
            expected.projection_path,
            projection_bytes,
            &projection_digest,
        )
    {
        return Err(StoreError::ArchiveIntegrity("CPU7 diagnostic ACK identity"));
    }
    let mut observation = inspect_cpu7_historical_projection(
        projection_bytes,
        expected.reader_max_projection_entries,
        expected.reader_max_dependency_edges,
        expected.reader_max_history_charge_bytes,
        check,
    )?;
    observation.ack_content_join = "matched_caller_pins_not_authenticated";
    Ok(observation)
}

/// Inspect original bytes, without opening paths or issuing capabilities.
/// The pin is a caller claim, not an authentication source. The future /3
/// consumer must independently join the held registration, publication and
/// ACK before interpreting any of these detached checks.
fn inspect_cpu7_historical_projection(
    bytes: &[u8],
    max_entries: usize,
    max_edges: usize,
    max_history_charge_bytes: usize,
    check: &mut dyn FnMut() -> Result<(), StoreError>,
) -> Result<HistoricalCpu7Observation, StoreError> {
    check()?;
    if bytes.is_empty() || bytes.len() > MAX_PROJECTION_BYTES {
        return Err(StoreError::ArchiveByteBudget);
    }
    if !(1..=MAX_PROJECTION_ENTRIES).contains(&max_entries)
        || max_edges > MAX_PROJECTION_BYTES
        || !(1..=MAX_HISTORY_CHARGE_BYTES).contains(&max_history_charge_bytes)
    {
        return Err(StoreError::InvalidConditions("CPU7 historical read limits"));
    }
    let wire: HistoricalProjection = serde_json::from_slice(bytes)
        .map_err(|_| StoreError::ArchiveIntegrity("CPU7 projection JSON"))?;
    check()?;
    if wire.schema != "rz-pals-cpu7-view-evidence/1"
        || wire.store.schema != "rz-pals-review-store-projection/1"
        || wire.generation != wire.store.generation
        || wire.records.is_empty()
        || wire.records.len() > 64
        || wire.selected_len == 0
        || wire.selected_len > wire.records.len()
    {
        return Err(StoreError::ArchiveIntegrity("CPU7 projection shape"));
    }
    let entries = wire
        .store
        .states
        .len()
        .checked_add(wire.store.lines.len())
        .and_then(|n| n.checked_add(wire.store.observations.len()))
        .and_then(|n| n.checked_add(wire.store.situations.len()))
        .and_then(|n| n.checked_add(wire.store.executions.len()))
        .ok_or(StoreError::Capacity("CPU7 projection entries"))?;
    if entries > max_entries {
        return Err(StoreError::Capacity("CPU7 projection entries"));
    }
    if wire.store.dependencies.len() > max_edges {
        return Err(StoreError::Capacity("CPU7 projection dependency edges"));
    }

    // Bound this reader's declared Rules-history backing before replaying any
    // state. Raw JSON/DTO, one transient Rules replay and allocator overhead
    // remain outside this charge; this is not an RSS or process peak limit.
    let mut history_charge = wire
        .root
        .history_charge()?
        .checked_add(wire.anchor.history_charge()?)
        .ok_or(StoreError::Capacity("CPU7 historical Rules history"))?;
    for state in &wire.store.states {
        check()?;
        history_charge = history_charge
            .checked_add(state.history_charge()?)
            .filter(|charge| *charge <= max_history_charge_bytes)
            .ok_or(StoreError::Capacity("CPU7 historical Rules history"))?;
    }
    if history_charge > max_history_charge_bytes {
        return Err(StoreError::Capacity("CPU7 historical Rules history"));
    }

    // The original writer encodes these from the same state object. The
    // separate snapshot digest does not include its revision, so compare the
    // complete wire before decoding and before making any identity claim.
    let (mut root_wire_found, mut anchor_wire_found) = (false, false);
    for state in &wire.store.states {
        check()?;
        root_wire_found |= state == &wire.root;
        anchor_wire_found |= state == &wire.anchor;
    }
    if !root_wire_found || !anchor_wire_found {
        return Err(StoreError::ArchiveIntegrity(
            "CPU7 original state wire differs",
        ));
    }

    let root_id = wire.root.id;
    let anchor_id = wire.anchor.id;
    let root = wire.root.decode()?;
    let anchor = wire.anchor.decode()?;
    let mut states = BTreeMap::new();
    for state in wire.store.states {
        check()?;
        let id = state.id;
        if states.insert(id, state.decode()?).is_some() {
            return Err(StoreError::ArchiveIntegrity("CPU7 duplicate state"));
        }
    }
    for (id, expected) in [(root_id, &root), (anchor_id, &anchor)] {
        let actual = states
            .get(&id)
            .ok_or(StoreError::ArchiveIntegrity("CPU7 missing state"))?;
        if snapshot_digest(actual).map_err(|_| StoreError::ArchiveIntegrity("CPU7 state digest"))?
            != snapshot_digest(expected)
                .map_err(|_| StoreError::ArchiveIntegrity("CPU7 state digest"))?
        {
            return Err(StoreError::ArchiveIntegrity("CPU7 exact state differs"));
        }
    }
    let mut line_ids = BTreeSet::new();
    for line in wire.store.lines {
        check()?;
        if !line_ids.insert(line.id) {
            return Err(StoreError::ArchiveIntegrity("CPU7 duplicate line"));
        }
        // A Store line's configured ply bound is independent of the Rules
        // trace bound and can exceed it. This partial reader validates the
        // chunk shape without inventing a smaller line-store limit; full
        // dependency and chain replay remain explicitly unknown.
        line.decode(usize::MAX)?;
    }
    let mut observations = BTreeMap::new();
    for (id, observation) in wire.store.observations {
        check()?;
        if observations.insert(id, observation.decode()?).is_some() {
            return Err(StoreError::ArchiveIntegrity("CPU7 duplicate observation"));
        }
    }
    let mut revisions = BTreeSet::new();
    let mut original = Vec::with_capacity(wire.records.len());
    let mut first_move = None;
    for item in wire.records {
        check()?;
        let record = item.record.decode()?;
        let first = record.line.first().copied();
        if record.origin_state != anchor_id
            || record.line.len() > MAX_LINE
            || first_move.is_some_and(|expected| expected != first)
            || [record.parent_revision, record.supersedes_revision]
                .into_iter()
                .flatten()
                .any(|parent| parent >= record.revision || !revisions.contains(&parent))
            || !revisions.insert(record.revision)
        {
            return Err(StoreError::ArchiveIntegrity(
                "CPU7 record anchor or revision",
            ));
        }
        first_move.get_or_insert(first);
        let mut position = Position::from_snapshot(&anchor, PositionLimits::default())
            .map_err(|_| StoreError::ArchiveIntegrity("CPU7 anchor replay"))?;
        for movement in &record.line {
            check()?;
            position
                .make_move(*movement)
                .map_err(|_| StoreError::ArchiveIntegrity("CPU7 illegal line"))?;
        }
        original.push(CohortCpu7OriginalRecord::historical(record, item.evidence));
    }
    let selected = &original[..wire.selected_len];
    if selected[0].record().kind != RecordKind::Proposal {
        return Err(StoreError::ArchiveIntegrity("CPU7 proposal absent"));
    }
    let selected_revisions: BTreeSet<_> =
        selected.iter().map(|item| item.record().revision).collect();
    let mut cpu_ids = BTreeSet::new();
    let mut seventh = None;
    for (index, item) in selected.iter().enumerate() {
        check()?;
        let record = item.record();
        if [record.parent_revision, record.supersedes_revision]
            .into_iter()
            .flatten()
            .any(|id| !selected_revisions.contains(&id))
        {
            return Err(StoreError::ArchiveIntegrity("CPU7 prefix relation"));
        }
        if item
            .evidence()
            .is_some_and(|id| !observations.contains_key(&id))
        {
            return Err(StoreError::ArchiveIntegrity("CPU7 missing evidence"));
        }
        let mut position = Position::from_snapshot(&anchor, PositionLimits::default())
            .map_err(|_| StoreError::ArchiveIntegrity("CPU7 anchor replay"))?;
        let target = if record.kind == RecordKind::CpuVerification {
            let id = record
                .cpu_observation
                .ok_or(StoreError::ArchiveIntegrity("CPU7 CPU ID"))?;
            if !cpu_ids.insert(id) {
                return Err(StoreError::ArchiveIntegrity(
                    "CPU7 duplicate CPU observation",
                ));
            }
            let observation = observations
                .get(&id)
                .ok_or(StoreError::ArchiveIntegrity("CPU7 missing CPU observation"))?;
            if !matches!(
                (observation.kind, observation.scope),
                (
                    ObservationKind::CpuAnalysis,
                    EvidenceScope::DepthLimited { .. }
                ) | (
                    ObservationKind::ExternalCpuAnalysis,
                    EvidenceScope::ProfiledExternalUci { .. } | EvidenceScope::ExternalUci { .. }
                )
            ) {
                return Err(StoreError::ArchiveIntegrity("CPU7 CPU observation scope"));
            }
            if cpu_ids.len() == 7 {
                seventh = Some(index);
            }
            Some(
                states
                    .get(&observation.state)
                    .ok_or(StoreError::ArchiveIntegrity("CPU7 CPU state"))?,
            )
        } else {
            None
        };
        let mut connected = target.is_none_or(|target| position.snapshot().same_state(target));
        for movement in &record.line {
            check()?;
            position
                .make_move(*movement)
                .map_err(|_| StoreError::ArchiveIntegrity("CPU7 illegal line"))?;
            connected |= target.is_some_and(|target| position.snapshot().same_state(target));
        }
        if !connected {
            return Err(StoreError::ArchiveIntegrity("CPU7 CPU line anchor"));
        }
    }
    if cpu_ids.len() != 7 || seventh != Some(wire.selected_len - 1) {
        return Err(StoreError::ArchiveIntegrity("CPU7 selected prefix"));
    }
    check()?;
    let actual = cohort_cpu7_original_digest(
        wire.bundle_id,
        wire.generation,
        root_id,
        &root,
        &anchor,
        &original,
        wire.selected_len,
    )
    .map_err(|_| StoreError::ArchiveIntegrity("CPU7 original digest input"))?;
    if actual != wire.original_record_handle_digest
        || cohort_cpu7_view_binding(actual, wire.policy_digest, wire.view_digest, wire.view_seal)
            != wire.view_binding
    {
        return Err(StoreError::ArchiveIntegrity("CPU7 bundle or view binding"));
    }
    check()?;
    Ok(HistoricalCpu7Observation {
        selected_records: wire.selected_len,
        original_records: original.len(),
        checked_cpu_observations: cpu_ids.len(),
        original_record_handle_digest: actual,
        view_binding: wire.view_binding,
        ack_content_join: "not_checked",
        original_owner: "unknown",
        registration_and_ack: "unknown",
        source_record_evidence_semantics: "unknown",
        full_dependency_replay: "unknown",
        model_input_equivalence: "unknown",
        physical_completion: "unknown",
        training_authorized: false,
    })
}
