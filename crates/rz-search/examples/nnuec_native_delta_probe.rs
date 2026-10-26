//! Independent CPU numeric caller for a Root-supplied checkpoint and request.
//! It uses only public rz-search/rz-position APIs. No original PALS authority,
//! live accepted-C lineage, worker installation ACK, cache-release capability,
//! UCI session closure, training or performance evidence is reconstructed.
//!
//! Input scope is deliberately diagnostic_rules_admission. Ordinary typed
//! Stores/ReviewView/Request/PreparationActor admission constructs this run's
//! real Bound; a matching serialized seal never substitutes an original owner.
//!
//! The original work/cleanup deadlines also cover canonical work and failure
//! cleanup. Native initialize/observe retain their existing Bound guards.
//! Supply explicit checkpoint/request paths, byte lengths and SHA-256 pins,
//! plus finite --work-ms and --cleanup-ms limits. Output is a bounded JSON
//! diagnostic receipt; source provenance and final OS closure stay with the caller.
//! Optional --context-mutations diagnostic_v1 adds separate C arithmetic and
//! failure-atomicity observations for quiet_start. It issues no new Bound/ACK
//! and does not certify current accepted-C lineage or performance.

#![forbid(unsafe_code)]

use rz_position::{BoardMove, Color, Position};
use rz_search::cpu_value::nnuec::NNUEC_MAX_BYTES;
use rz_search::cpu_value::nnuec::v2::cpu::V2CpuValue;
use rz_search::cpu_value::nnuec::v2::review::{
    BoundNnuecReviewContext, PreparationActor, Request as PreparationRequest,
};
use rz_search::cpu_value::nnuec::v2::{self, Model, Output};
use rz_search::cpu_value::{CpuAccumulator, CpuValueEvaluator, CpuValueIdentity};
use rz_search::pals::engine::{RecordKind, RoleQueryPurpose, RoleRecord};
use rz_search::pals::review_context::{
    ReviewContextIndex, ReviewQuerySequences, ReviewSelectionPolicy,
};
use rz_search::pals::store::{PalsStores, StoreLimits};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

const REQUEST_CAP: usize = 256 * 1024;
const OUTPUT_CAP: usize = 2 * 1024 * 1024;
const MAX_CASES: usize = 16;
const MAX_PLIES: usize = 256;
const MAX_HISTORY: usize = 4096;
const MAX_WORK_MS: u64 = 600_000;
const MAX_CLEANUP_MS: u64 = 30_000;

// This is a Rules workload contract, not a model or owner registration.
const SPECIALS: [(&str, &str, &str); 12] = [
    ("quiet_start", "startpos", "e2e4"),
    (
        "castle_white_kingside",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "e1g1",
    ),
    (
        "castle_black_queenside",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "e8c8",
    ),
    (
        "en_passant_white",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",
        "e5d6",
    ),
    (
        "promotion_white_queen",
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "a7a8q",
    ),
    (
        "promotion_white_rook",
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "a7a8r",
    ),
    (
        "promotion_white_bishop",
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "a7a8b",
    ),
    (
        "promotion_white_knight",
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "a7a8n",
    ),
    (
        "promotion_black_queen",
        "4k3/8/8/8/8/8/p7/4K3 b - - 0 1",
        "a2a1q",
    ),
    (
        "promotion_black_rook",
        "4k3/8/8/8/8/8/p7/4K3 b - - 0 1",
        "a2a1r",
    ),
    (
        "promotion_black_bishop",
        "4k3/8/8/8/8/8/p7/4K3 b - - 0 1",
        "a2a1b",
    ),
    (
        "promotion_black_knight",
        "4k3/8/8/8/8/8/p7/4K3 b - - 0 1",
        "a2a1n",
    ),
];

#[derive(Clone, Debug, Serialize)]
struct ProbeError {
    stage: String,
    detail: String,
}
impl ProbeError {
    fn new(stage: &str, detail: impl ToString) -> Self {
        Self {
            stage: stage.into(),
            detail: detail.to_string().chars().take(2048).collect(),
        }
    }
}
type Result<T> = std::result::Result<T, ProbeError>;

fn need(ok: bool, stage: &str, detail: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(ProbeError::new(stage, detail))
    }
}
fn digest_text(value: &str, size: usize) -> bool {
    value.len() == size
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}
fn hex(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn check_work(until: Instant, stage: &str) -> Result<()> {
    need(
        Instant::now() < until,
        stage,
        "original probe work window expired",
    )
}

#[derive(Clone, Serialize)]
struct RawPin {
    path: String,
    bytes: usize,
    sha256: String,
}
struct Arguments {
    checkpoint: RawPin,
    request: RawPin,
    work_ms: u64,
    cleanup_ms: u64,
    context_mutations: bool,
}
fn forbidden_input_name(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    lower.is_empty()
        || lower == ".env"
        || lower.starts_with(".env.")
        || [
            ".ssh",
            ".aws",
            ".azure",
            ".gnupg",
            ".kube",
            ".secrets",
            "secrets",
            "gcloud",
            ".netrc",
            "_netrc",
            "token",
            "tokens",
            "token.json",
            "tokens.json",
            "auth.json",
            "authentication.json",
            "adc.json",
            "authorized_keys",
            "known_hosts",
        ]
        .contains(&lower.as_str())
        || [
            "credential",
            "service-account",
            "service_account",
            "ssh-key",
            "api-key",
            "api_key",
            "apikey",
            "client_secret",
            "client-secret",
            "private_key",
            "private-key",
        ]
        .iter()
        .any(|word| lower.contains(*word))
        || ["id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"]
            .iter()
            .any(|prefix| {
                lower == *prefix
                    || [".", "-", "_"].iter().any(|separator| {
                        let candidate = format!("{prefix}{separator}");
                        lower.starts_with(candidate.as_str())
                    })
            })
        || lower.starts_with("ssh_host_")
        || [".pem", ".key", ".p12", ".pfx", ".p8", ".jks", ".keystore"]
            .iter()
            .any(|ext| lower.ends_with(*ext))
}

fn argument_pin(path: String, size: String, digest: String, maximum: usize) -> Result<RawPin> {
    let p = Path::new(&path);
    need(
        p.is_absolute()
            && !path.chars().any(|c| matches!(c, '\0' | '\r' | '\n'))
            && !p.components().any(|c| matches!(c, Component::ParentDir)),
        "InputPath",
        "explicit absolute input path required",
    )?;
    // Reject known secret names in every ancestor as well as the basename,
    // before any file is opened. This predicate never reads secret content.
    need(
        p.file_name().is_some()
            && p.components().all(|component| match component {
                Component::Normal(name) => name
                    .to_str()
                    .is_some_and(|name| !forbidden_input_name(name)),
                _ => true,
            }),
        "InputPath",
        "secret/key/credential component or ancestor is forbidden",
    )?;
    let bytes = size
        .parse::<usize>()
        .map_err(|e| ProbeError::new("InputPin", e))?;
    need(
        bytes > 0 && bytes <= maximum && digest_text(&digest, 64),
        "InputPin",
        "positive bounded byte count and lowercase SHA256 required",
    )?;
    Ok(RawPin {
        path,
        bytes,
        sha256: digest,
    })
}
fn arguments() -> Result<Arguments> {
    let values: Vec<String> = std::env::args().skip(1).collect();
    need(
        values.len() == 16 || values.len() == 18,
        "CLI",
        "use --checkpoint PATH --checkpoint-bytes N --checkpoint-sha256 SHA --request PATH --request-bytes N --request-sha256 SHA --work-ms N --cleanup-ms N",
    )?;
    // The original eight flags remain required. C arithmetic is opt-in.
    let names = [
        "--checkpoint",
        "--checkpoint-bytes",
        "--checkpoint-sha256",
        "--request",
        "--request-bytes",
        "--request-sha256",
        "--work-ms",
        "--cleanup-ms",
        "--context-mutations",
    ];
    let mut map = std::collections::BTreeMap::new();
    for pair in values.chunks_exact(2) {
        need(
            names.contains(&pair[0].as_str())
                && map.insert(pair[0].as_str(), pair[1].as_str()).is_none(),
            "CLI",
            "unknown or repeated CLI flag",
        )?;
    }
    need(
        names[..8].iter().all(|name| map.contains_key(name)),
        "CLI",
        "all explicit flags are required",
    )?;
    let work_ms = map["--work-ms"]
        .parse()
        .map_err(|e| ProbeError::new("CLI", e))?;
    let cleanup_ms = map["--cleanup-ms"]
        .parse()
        .map_err(|e| ProbeError::new("CLI", e))?;
    need(
        (1..=MAX_WORK_MS).contains(&work_ms) && (1..=MAX_CLEANUP_MS).contains(&cleanup_ms),
        "CLI",
        "finite work/cleanup bounds outside this numeric profile",
    )?;
    let context_mutations = match map.get("--context-mutations") {
        None => false,
        Some(&"diagnostic_v1") => true,
        Some(_) => {
            return Err(ProbeError::new(
                "CLI",
                "--context-mutations requires diagnostic_v1",
            ));
        }
    };
    Ok(Arguments {
        checkpoint: argument_pin(
            map["--checkpoint"].into(),
            map["--checkpoint-bytes"].into(),
            map["--checkpoint-sha256"].into(),
            NNUEC_MAX_BYTES,
        )?,
        request: argument_pin(
            map["--request"].into(),
            map["--request-bytes"].into(),
            map["--request-sha256"].into(),
            REQUEST_CAP,
        )?,
        work_ms,
        cleanup_ms,
        context_mutations,
    })
}
fn read_pinned(pin: &RawPin) -> Result<Vec<u8>> {
    let path = PathBuf::from(&pin.path);
    let named = fs::symlink_metadata(&path).map_err(|e| ProbeError::new("InputRead", e))?;
    need(
        named.is_file() && !named.file_type().is_symlink(),
        "InputRead",
        "named input must be a regular non-symlink file",
    )?;
    let file = File::open(&path).map_err(|e| ProbeError::new("InputRead", e))?;
    let before = file
        .metadata()
        .map_err(|e| ProbeError::new("InputRead", e))?;
    need(
        before.is_file() && before.len() == pin.bytes as u64,
        "InputRead",
        "actual input length differs from supplied pin",
    )?;
    let mut bytes = Vec::new();
    file.take((pin.bytes + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| ProbeError::new("InputRead", e))?;
    // A successful short read under a bytes+1 limit has reached actual EOF.
    need(
        bytes.len() == pin.bytes && hex(Sha256::digest(&bytes)) == pin.sha256,
        "InputRawPin",
        "actual bytes/EOF/SHA256 do not match supplied pin",
    )?;
    Ok(bytes)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PositionInput {
    fen: String,
    moves: Vec<String>,
}
impl PositionInput {
    fn rules(&self) -> Result<Position> {
        need(
            self.fen.len() <= 128 && self.moves.len() <= MAX_HISTORY,
            "RulesInput",
            "Rules root exceeds bounded input",
        )?;
        let mut rules = if self.fen == "startpos" {
            Position::startpos()
        } else {
            Position::from_fen(&self.fen).map_err(|e| ProbeError::new("RulesInput", e))?
        };
        for text in &self.moves {
            rules
                .make_move(movement(text)?)
                .map_err(|e| ProbeError::new("RulesReplay", e))?;
        }
        Ok(rules)
    }
}
fn movement(text: &str) -> Result<BoardMove> {
    need(
        (4..=5).contains(&text.len()),
        "MoveInput",
        "bounded UCI coordinate required",
    )?;
    text.parse().map_err(|e| ProbeError::new("MoveInput", e))
}
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum KindInput {
    Proposal,
    Counterexample,
    Repair,
    CpuVerification,
}
impl KindInput {
    fn kind(&self) -> RecordKind {
        match self {
            Self::Proposal => RecordKind::Proposal,
            Self::Counterexample => RecordKind::Counterexample,
            Self::Repair => RecordKind::Repair,
            Self::CpuVerification => RecordKind::CpuVerification,
        }
    }
}
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum SideInput {
    White,
    Black,
}
impl SideInput {
    fn color(&self) -> Color {
        match self {
            Self::White => Color::White,
            Self::Black => Color::Black,
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DiagnosticRecord {
    revision: u64,
    parent_revision: Option<u64>,
    supersedes_revision: Option<u64>,
    kind: KindInput,
    moves: Vec<String>,
    perspective: SideInput,
    critical: bool,
}
#[derive(Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
enum SealExpectation {
    ObserveDiagnostic,
    MatchBytes { sha256: String },
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CaseInput {
    id: String,
    coverage: String,
    root: PositionInput,
    local_moves: Vec<String>,
    diagnostic_records: Vec<DiagnosticRecord>,
    seal_expectation: SealExpectation,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema: String,
    declared_library_source: String,
    context_lineage: String,
    expected_model_identity: CpuValueIdentity,
    cases: Vec<CaseInput>,
}
impl Input {
    fn validate(&self, model: &Model) -> Result<()> {
        need(
            self.schema == "rovent.nnuec-native-delta-request/1"
                && self.context_lineage == "diagnostic_rules_admission",
            "InputScope",
            "only diagnostic Rules/Stores admission is supported; serialized PALS authority is not supported",
        )?;
        need(
            digest_text(&self.declared_library_source, 40),
            "SourceDeclaration",
            "explicit full40 declaration required; Root verifies the library source outside this tool",
        )?;
        need(
            &self.expected_model_identity == model.identity()
                && model.identity().semantics == v2::SEMANTICS,
            "ModelIdentity",
            "actual V2 model differs from explicit Root-supplied identity",
        )?;
        need(
            (13..=MAX_CASES).contains(&self.cases.len()),
            "CaseRoster",
            "twelve special cases and one to four multi-ply cases required",
        )?;
        let mut ids = BTreeSet::new();
        let mut covered = BTreeSet::new();
        let mut multi = 0;
        for case in &self.cases {
            need(
                !case.id.is_empty()
                    && case.id.len() <= 64
                    && case
                        .id
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
                    && ids.insert(case.id.as_str()),
                "CaseRoster",
                "unique bounded case IDs required",
            )?;
            need(
                !case.local_moves.is_empty()
                    && case.local_moves.len() <= MAX_PLIES
                    && case.local_moves.len() <= model.base_checkpoint().config.max_plies,
                "CaseCapacity",
                "actual model must support every declared local ply; no truncation",
            )?;
            if case.coverage == "multi_ply" {
                need(
                    case.local_moves.len() >= 3,
                    "CaseRoster",
                    "multi-ply case requires a nested undo chain",
                )?;
                multi += 1;
            } else {
                let (_, fen, text) = SPECIALS
                    .iter()
                    .find(|(name, _, _)| *name == case.coverage)
                    .ok_or_else(|| ProbeError::new("CaseRoster", "unknown special coverage"))?;
                need(
                    covered.insert(case.coverage.as_str())
                        && case.root.fen == *fen
                        && case.root.moves.is_empty()
                        && case.local_moves == [text.to_string()],
                    "CaseRoster",
                    "special Rules workload differs from declared suite",
                )?;
            }
            need(
                (2..=64).contains(&case.diagnostic_records.len())
                    && case.diagnostic_records.len() <= model.base_checkpoint().config.max_lines
                    && case
                        .diagnostic_records
                        .iter()
                        .any(|r| r.parent_revision.is_some()),
                "ContextInput",
                "explicit nonempty diagnostic C with a directed relation is required",
            )?;
        }
        need(
            covered.len() == SPECIALS.len() && multi > 0,
            "CaseRoster",
            "incomplete fixed special coverage",
        )?;
        Ok(())
    }
}

#[derive(Serialize)]
struct Comparison {
    phase: String,
    ply: usize,
    native_bits_sha256: String,
    canonical_bits_sha256: String,
    bitwise_equal: bool,
}
#[derive(Serialize)]
struct CaseResult {
    id: String,
    coverage: String,
    status: String,
    phase: String,
    view_seal_sha256: Option<String>,
    view_seal_expectation: String,
    view_seal_bytes_matched: Option<bool>,
    record_scope_sha256: Option<String>,
    numeric_context_sha256: Option<String>,
    context_nonzero: Option<bool>,
    local_changed: bool,
    deltas_applied: usize,
    owned_restores: usize,
    failure_cleanup_restores: usize,
    native_readouts: usize,
    canonical_readouts: usize,
    peak_owned_undos: usize,
    final_rules_root_restored: Option<bool>,
    final_numeric_root_restored: Option<bool>,
    preparation_actor_join: String,
    review_pins_after_join: Option<usize>,
    comparisons: Vec<Comparison>,
    error: Option<ProbeError>,
    cleanup_errors: Vec<ProbeError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_mutation_diagnostic: Option<ContextMutationDiagnostic>,
}
impl CaseResult {
    fn new(case: &CaseInput) -> Self {
        Self {
            id: case.id.clone(),
            coverage: case.coverage.clone(),
            status: "pending".into(),
            phase: "admission".into(),
            view_seal_sha256: None,
            view_seal_expectation: match &case.seal_expectation {
                SealExpectation::ObserveDiagnostic => "observe_diagnostic",
                SealExpectation::MatchBytes { .. } => "match_bytes",
            }
            .into(),
            view_seal_bytes_matched: None,
            record_scope_sha256: None,
            numeric_context_sha256: None,
            context_nonzero: None,
            context_mutation_diagnostic: None,
            local_changed: false,
            deltas_applied: 0,
            owned_restores: 0,
            failure_cleanup_restores: 0,
            native_readouts: 0,
            canonical_readouts: 0,
            peak_owned_undos: 0,
            final_rules_root_restored: None,
            final_numeric_root_restored: None,
            preparation_actor_join: "not_started".into(),
            review_pins_after_join: None,
            comparisons: Vec::new(),
            error: None,
            cleanup_errors: Vec::new(),
        }
    }
}

fn output_bits(output: &Output) -> Result<Vec<u8>> {
    let n = &output.numeric;
    let mut bytes = Vec::new();
    for text in [&n.schema, &n.weights_sha256, &output.context_sha256] {
        bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
        bytes.extend_from_slice(text.as_bytes());
    }
    for values in [
        n.board_accumulators[0].as_slice(),
        n.board_accumulators[1].as_slice(),
        n.context_accumulator.as_slice(),
        n.local_accumulator.as_slice(),
        n.hidden.as_slice(),
        &n.raw_outputs,
        &n.wdl_logits,
        &n.wdl,
        std::slice::from_ref(&n.value),
    ] {
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            need(
                value.is_finite(),
                "NumericFinite",
                "nonfinite actual NNUEC output",
            )?;
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }
    Ok(bytes)
}
fn compare(
    actual: &Output,
    expected: &Output,
    phase: &str,
    ply: usize,
    until: Instant,
    record: &mut CaseResult,
) -> Result<()> {
    check_work(until, "NumericComparisonBefore")?;
    let native = output_bits(actual)?;
    let canonical = output_bits(expected)?;
    let matched = native == canonical;
    let comparison = Comparison {
        phase: phase.into(),
        ply,
        native_bits_sha256: hex(Sha256::digest(&native)),
        canonical_bits_sha256: hex(Sha256::digest(&canonical)),
        bitwise_equal: matched,
    };
    if !matched {
        // Preserve an observed mismatch as failure even if the comparison
        // itself ran late. It grants no successful result acceptance.
        record.comparisons.push(comparison);
        return Err(ProbeError::new(
            "NativeNumericMismatch",
            "native and rebuilt/owned-before output bits differ",
        ));
    }
    check_work(until, "NumericComparisonAcceptance")?;
    record.comparisons.push(comparison);
    Ok(())
}
fn observe(
    evaluator: &V2CpuValue,
    value: &CpuAccumulator,
    rules: &Position,
    record: &mut CaseResult,
) -> Result<Output> {
    let output = evaluator
        .observe(value, rules)
        .map_err(|e| ProbeError::new("NativeReadout", e))?;
    record.native_readouts += 1;
    Ok(output)
}
fn rebuild(
    model: &Model,
    bound: &BoundNnuecReviewContext,
    root: &Position,
    prefix: &[BoardMove],
    rules: &Position,
    until: Instant,
    record: &mut CaseResult,
) -> Result<Output> {
    check_work(until, "CanonicalLocalBefore")?;
    let mut local = model
        .local(root)
        .map_err(|e| ProbeError::new("RebuildLocal", e))?;
    check_work(until, "CanonicalLocalResult")?;
    for movement in prefix {
        check_work(until, "CanonicalAppendBefore")?;
        model
            .append_local(&mut local, *movement)
            .map_err(|e| ProbeError::new("RebuildLocal", e))?;
        check_work(until, "CanonicalAppendResult")?;
    }
    need(
        local.position().snapshot().same_state(&rules.snapshot()),
        "RebuildRules",
        "canonical local replay differs from actual Rules history",
    )?;
    check_work(until, "CanonicalBoardBefore")?;
    let board = model
        .board(rules)
        .map_err(|e| ProbeError::new("RebuildBoard", e))?;
    check_work(until, "CanonicalBoardResult")?;
    check_work(until, "CanonicalForwardBefore")?;
    let output = model
        .forward(&board, bound.context(), &local, rules)
        .map_err(|e| ProbeError::new("RebuildReadout", e))?;
    // Count a forward that actually returned even if its result is too late
    // for comparison. A rejected late result is never appended as a match.
    record.canonical_readouts += 1;
    check_work(until, "CanonicalForwardResult")?;
    Ok(output)
}

// This diagnostic observes immutable C arithmetic only. It never rebinds a
// mutated Context to the admitted View/Bound or issues installation authority.
#[derive(Serialize)]
struct ContextMutationDiagnostic {
    schema: &'static str,
    scope: &'static str,
    status: &'static str,
    direct_forwards_completed: usize,
    original_context_preserved: Option<bool>,
    accepted_c_lineage_verified: bool,
    mutated_bound_created: bool,
    performance_evidence: bool,
    steps: Vec<ContextMutationStep>,
    rejections: Vec<ContextMutationRejection>,
    error: Option<ProbeError>,
}
impl ContextMutationDiagnostic {
    fn new() -> Self {
        Self {
            schema: "rovent.nnuec-context-mutation-diagnostic/1",
            scope: "actual_checkpoint_immutable_context_arithmetic_only",
            status: "running",
            direct_forwards_completed: 0,
            original_context_preserved: None,
            accepted_c_lineage_verified: false,
            mutated_bound_created: false,
            performance_evidence: false,
            steps: Vec::new(),
            rejections: Vec::new(),
            error: None,
        }
    }
}
#[derive(Serialize)]
struct ContextMutationStep {
    operation: String,
    context_sha256: String,
    line_count: usize,
    edge_count: usize,
    accumulator_bits_sha256: String,
    raw_rebuild_accumulator_bits_sha256: String,
    output_bits_sha256: String,
    raw_rebuild_output_bits_sha256: String,
    raw_rebuild_equal: bool,
    stored_contribution_rebuild_equal: bool,
    seal_matches_original: bool,
    numeric_matches_original: bool,
}
#[derive(Serialize)]
struct ContextMutationRejection {
    operation: String,
    observed_error: String,
    original_context_sha256: String,
    original_accumulator_and_output_preserved: Option<bool>,
}
#[derive(Clone)]
struct ContextLineInput {
    id: u64,
    anchor: Position,
    moves: Vec<BoardMove>,
    kind: v2::LineKind,
}
#[derive(Clone)]
struct ContextState {
    context: Arc<v2::Context>,
    lines: Vec<ContextLineInput>,
    edges: Vec<v2::Edge>,
}
#[derive(PartialEq, Eq)]
struct ContextObserved {
    seal: String,
    accumulator: Vec<u32>,
    output: Vec<u8>,
    numeric: Vec<u8>,
    lines: usize,
    edges: usize,
}
fn context_bits_digest(values: &[u32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_le_bytes());
    }
    hex(hash.finalize())
}
fn observe_context(
    model: &Model,
    context: &v2::Context,
    root: &Position,
    until: Instant,
    diagnostic: &mut ContextMutationDiagnostic,
) -> Result<ContextObserved> {
    check_work(until, "ContextObserveBefore")?;
    let board = model
        .board(root)
        .map_err(|e| ProbeError::new("ContextBoard", e))?;
    let local = model
        .local(root)
        .map_err(|e| ProbeError::new("ContextLocal", e))?;
    check_work(until, "ContextForwardBefore")?;
    let output = model
        .forward(&board, context, &local, root)
        .map_err(|e| ProbeError::new("ContextForward", e))?;
    diagnostic.direct_forwards_completed += 1;
    check_work(until, "ContextForwardResult")?;
    // Exclude the order-sensitive seal only from this separately named numeric
    // comparison. The actual output and its actual seal are also retained.
    let mut numeric_only = output.clone();
    numeric_only.context_sha256.clear();
    Ok(ContextObserved {
        seal: context
            .sha256()
            .map_err(|e| ProbeError::new("ContextSeal", e))?,
        accumulator: context.values().iter().map(|v| v.to_bits()).collect(),
        output: output_bits(&output)?,
        numeric: output_bits(&numeric_only)?,
        lines: context.line_count(),
        edges: context.edge_count(),
    })
}
fn verify_context_state(
    model: &Model,
    state: &ContextState,
    root: &Position,
    until: Instant,
    operation: &str,
    baseline: Option<&ContextObserved>,
    diagnostic: &mut ContextMutationDiagnostic,
) -> Result<ContextObserved> {
    check_work(until, "ContextRawRebuildBefore")?;
    let mut rebuilt = model.empty_context();
    for line in &state.lines {
        check_work(until, "ContextRawLineBefore")?;
        let encoded = model
            .encode_line(&line.anchor, &line.moves, line.kind)
            .map_err(|e| ProbeError::new("ContextRawLine", e))?;
        rebuilt = model
            .with_line(&rebuilt, line.id, encoded)
            .map_err(|e| ProbeError::new("ContextRawInsert", e))?;
    }
    for edge in &state.edges {
        check_work(until, "ContextRawEdgeBefore")?;
        rebuilt = model
            .with_edge(&rebuilt, *edge)
            .map_err(|e| ProbeError::new("ContextRawEdge", e))?;
    }
    let stored_rebuild = model
        .rebuild_context(&state.context)
        .map_err(|e| ProbeError::new("ContextStoredContributionRebuild", e))?;
    let actual = observe_context(model, &state.context, root, until, diagnostic)?;
    let expected = observe_context(model, &rebuilt, root, until, diagnostic)?;
    let raw_equal = actual == expected;
    let stored_equal = actual.accumulator
        == stored_rebuild
            .values()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
        && actual.seal
            == stored_rebuild
                .sha256()
                .map_err(|e| ProbeError::new("ContextStoredSeal", e))?;
    let original = baseline.unwrap_or(&actual);
    diagnostic.steps.push(ContextMutationStep {
        operation: operation.into(),
        context_sha256: actual.seal.clone(),
        line_count: actual.lines,
        edge_count: actual.edges,
        accumulator_bits_sha256: context_bits_digest(&actual.accumulator),
        raw_rebuild_accumulator_bits_sha256: context_bits_digest(&expected.accumulator),
        output_bits_sha256: hex(Sha256::digest(&actual.output)),
        raw_rebuild_output_bits_sha256: hex(Sha256::digest(&expected.output)),
        raw_rebuild_equal: raw_equal,
        stored_contribution_rebuild_equal: stored_equal,
        seal_matches_original: actual.seal == original.seal,
        numeric_matches_original: actual.accumulator == original.accumulator
            && actual.numeric == original.numeric,
    });
    need(
        raw_equal && stored_equal,
        "ContextNumericMismatch",
        "actual immutable C differs from raw-input or stored-contribution rebuild",
    )?;
    check_work(until, "ContextComparisonAcceptance")?;
    Ok(actual)
}
fn verify_context_rejection(
    model: &Model,
    state: &ContextState,
    root: &Position,
    until: Instant,
    operation: &str,
    diagnostic: &mut ContextMutationDiagnostic,
    action: impl FnOnce() -> std::result::Result<
        Arc<v2::Context>,
        rz_search::cpu_value::nnuec::NnuecError,
    >,
) -> Result<()> {
    let before = observe_context(model, &state.context, root, until, diagnostic)?;
    let error = match action() {
        Err(error) => error,
        Ok(_) => return Err(ProbeError::new("ContextExpectedRejection", operation)),
    };
    // Preserve the actual rejection even if the following observation expires.
    // An unobserved preservation check remains null, never an invented true.
    diagnostic.rejections.push(ContextMutationRejection {
        operation: operation.into(),
        observed_error: error.to_string().chars().take(2048).collect(),
        original_context_sha256: before.seal.clone(),
        original_accumulator_and_output_preserved: None,
    });
    let after = observe_context(model, &state.context, root, until, diagnostic)?;
    let preserved = before == after;
    diagnostic
        .rejections
        .last_mut()
        .unwrap()
        .original_accumulator_and_output_preserved = Some(preserved);
    need(
        preserved,
        "ContextFailureAtomicity",
        "rejected change modified the original C",
    )
}
fn run_context_mutations(
    model: &Model,
    bound: &BoundNnuecReviewContext,
    view: &rz_search::pals::review_context::ReviewView,
    root: &Position,
    until: Instant,
    diagnostic: &mut ContextMutationDiagnostic,
) -> Result<()> {
    check_work(until, "ContextMutationAdmission")?;
    let local = model
        .local(root)
        .map_err(|e| ProbeError::new("ContextInputLocal", e))?;
    let reference = model
        .reference_input(bound.context(), &local, root)
        .map_err(|e| ProbeError::new("ContextReferenceInput", e))?;
    need(
        reference.context_lines.len() == view.records().len()
            && bound.context_source_snapshots().len() == view.records().len(),
        "ContextSourceRows",
        "same admitted source rows required",
    )?;
    let mut lines = Vec::new();
    for (index, record) in view.records().iter().enumerate() {
        check_work(until, "ContextSourceRow")?;
        need(
            reference.context_lines[index].id == index as u64
                && view.records_metadata()[index].source_line_offset == 0,
            "ContextSourceRows",
            "diagnostic raw encode requires original, unre-based rows",
        )?;
        let original = bound
            .context()
            .line(index as u64)
            .ok_or_else(|| ProbeError::new("ContextSourceRows", "actual context line absent"))?;
        lines.push(ContextLineInput {
            id: index as u64,
            anchor: Position::from_snapshot(
                &bound.context_source_snapshots()[index],
                Default::default(),
            )
            .map_err(|e| ProbeError::new("ContextSourceAnchor", e))?,
            moves: record.line.clone(),
            kind: original.kind(),
        });
    }
    let edges = reference
        .context_edges
        .iter()
        .map(|edge| {
            Ok(v2::Edge {
                source: edge.source,
                target: edge.target,
                kind: v2::EdgeKind::try_from(edge.kind)
                    .map_err(|e| ProbeError::new("ContextEdgeKind", e))?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let original = ContextState {
        context: Arc::clone(bound.context()),
        lines,
        edges,
    };
    let baseline =
        verify_context_state(model, &original, root, until, "original", None, diagnostic)?;
    let edge = *original.edges.last().ok_or_else(|| {
        ProbeError::new("ContextMutationAdmission", "one actual relation required")
    })?;
    let target = original
        .lines
        .iter()
        .position(|line| line.id == edge.target)
        .ok_or_else(|| {
            ProbeError::new("ContextMutationAdmission", "actual relation target absent")
        })?;
    let mut changed = original.clone();
    changed.lines[target].kind = match changed.lines[target].kind {
        v2::LineKind::Repair => v2::LineKind::Counterexample,
        _ => v2::LineKind::Repair,
    };
    let line = &changed.lines[target];
    let replacement = model
        .encode_line(&line.anchor, &line.moves, line.kind)
        .map_err(|e| ProbeError::new("ContextReplacementEncode", e))?;
    changed.context = model
        .with_line(&original.context, edge.target, replacement)
        .map_err(|e| ProbeError::new("ContextReplace", e))?;
    verify_context_state(
        model,
        &changed,
        root,
        until,
        "replace_line_kind",
        Some(&baseline),
        diagnostic,
    )?;
    changed.context = model
        .with_line(
            &changed.context,
            edge.target,
            Arc::clone(original.context.line(edge.target).unwrap()),
        )
        .map_err(|e| ProbeError::new("ContextReplaceUndo", e))?;
    changed.lines = original.lines.clone();
    let undone = verify_context_state(
        model,
        &changed,
        root,
        until,
        "replace_line_undo",
        Some(&baseline),
        diagnostic,
    )?;
    need(
        undone == baseline,
        "ContextReplaceUndo",
        "inverse replacement must restore original C",
    )?;
    let mut removed = original.clone();
    removed.context = model
        .without_edge(&removed.context, edge)
        .map_err(|e| ProbeError::new("ContextRemoveEdge", e))?;
    removed.edges.pop();
    verify_context_state(
        model,
        &removed,
        root,
        until,
        "remove_last_edge",
        Some(&baseline),
        diagnostic,
    )?;
    removed.context = model
        .with_edge(&removed.context, edge)
        .map_err(|e| ProbeError::new("ContextRestoreEdge", e))?;
    removed.edges.push(edge);
    let undone = verify_context_state(
        model,
        &removed,
        root,
        until,
        "restore_last_edge",
        Some(&baseline),
        diagnostic,
    )?;
    need(
        undone == baseline,
        "ContextRestoreEdge",
        "last-edge inverse must restore original order and C",
    )?;
    let mut isolated = original.clone();
    let last =
        isolated.lines.last().cloned().ok_or_else(|| {
            ProbeError::new("ContextMutationAdmission", "actual last line absent")
        })?;
    let incident = original
        .edges
        .iter()
        .copied()
        .filter(|edge| edge.source == last.id || edge.target == last.id)
        .collect::<Vec<_>>();
    for edge in incident.iter().rev() {
        check_work(until, "ContextIsolateEdge")?;
        isolated.context = model
            .without_edge(&isolated.context, *edge)
            .map_err(|e| ProbeError::new("ContextIsolateEdge", e))?;
    }
    isolated
        .edges
        .retain(|edge| edge.source != last.id && edge.target != last.id);
    isolated.context = model
        .without_line(&isolated.context, last.id)
        .map_err(|e| ProbeError::new("ContextRemoveIsolatedLine", e))?;
    isolated.lines.pop();
    verify_context_state(
        model,
        &isolated,
        root,
        until,
        "remove_isolated_last_line",
        Some(&baseline),
        diagnostic,
    )?;
    isolated.context = model
        .with_line(
            &isolated.context,
            last.id,
            Arc::clone(original.context.line(last.id).unwrap()),
        )
        .map_err(|e| ProbeError::new("ContextRestoreIsolatedLine", e))?;
    isolated.lines.push(last);
    for edge in &incident {
        check_work(until, "ContextRestoreIncidentEdge")?;
        isolated.context = model
            .with_edge(&isolated.context, *edge)
            .map_err(|e| ProbeError::new("ContextRestoreIncidentEdge", e))?;
        isolated.edges.push(*edge);
    }
    let restored = verify_context_state(
        model,
        &isolated,
        root,
        until,
        "restore_isolated_line_and_edges",
        Some(&baseline),
        diagnostic,
    )?;
    // Re-added incident edges may have a new insertion order/seal. Numeric
    // equality is required; exact seal restoration is reported, not invented.
    need(
        restored.accumulator == baseline.accumulator && restored.numeric == baseline.numeric,
        "ContextRestoreIsolatedLine",
        "line/edge inverse must restore actual numeric C",
    )?;
    let ids = original
        .lines
        .iter()
        .map(|line| line.id)
        .collect::<Vec<_>>();
    let empty = model
        .without_bundle(&original.context, &ids)
        .map_err(|e| ProbeError::new("ContextRemoveClosedBundle", e))?;
    verify_context_state(
        model,
        &ContextState {
            context: empty,
            lines: Vec::new(),
            edges: Vec::new(),
        },
        root,
        until,
        "remove_closed_bundle",
        Some(&baseline),
        diagnostic,
    )?;
    verify_context_rejection(
        model,
        &original,
        root,
        until,
        "duplicate_edge",
        diagnostic,
        || model.with_edge(&original.context, edge),
    )?;
    verify_context_rejection(
        model,
        &original,
        root,
        until,
        "connected_line_removal",
        diagnostic,
        || model.without_line(&original.context, edge.target),
    )?;
    verify_context_rejection(
        model,
        &original,
        root,
        until,
        "dangling_bundle_removal",
        diagnostic,
        || model.without_bundle(&original.context, &[edge.source]),
    )?;
    verify_context_rejection(
        model,
        &original,
        root,
        until,
        "duplicate_bundle_ids",
        diagnostic,
        || model.without_bundle(&original.context, &[edge.source, edge.source]),
    )?;
    let target_line = &original.lines[target];
    let wrong_move = target_line
        .anchor
        .legal_moves()
        .into_iter()
        .find(|movement| Some(movement) != target_line.moves.first())
        .ok_or_else(|| {
            ProbeError::new(
                "ContextWrongFirstMove",
                "no alternate actual legal first move",
            )
        })?;
    let wrong_first = model
        .encode_line(&target_line.anchor, &[wrong_move], target_line.kind)
        .map_err(|e| ProbeError::new("ContextWrongFirstMove", e))?;
    verify_context_rejection(
        model,
        &original,
        root,
        until,
        "replacement_first_move_mismatch",
        diagnostic,
        || model.with_line(&original.context, edge.target, wrong_first),
    )?;
    let after = observe_context(model, &original.context, root, until, diagnostic)?;
    diagnostic.original_context_preserved = Some(after == baseline);
    need(
        after == baseline,
        "ContextOriginalPreservation",
        "original admitted C changed",
    )?;
    diagnostic.status = "context_mutation_and_failure_atomicity_verified";
    Ok(())
}

fn check_cleanup(
    cleanup_until: Instant,
    phase: &str,
    owned_pending: usize,
    record: &CaseResult,
) -> Result<()> {
    if Instant::now() < cleanup_until {
        return Ok(());
    }
    let not_restored = record
        .deltas_applied
        .checked_sub(record.owned_restores + record.failure_cleanup_restores);
    Err(ProbeError::new(
        "CleanupDeadline",
        format!(
            "original cleanup window expired at {phase}; owned_undos_pending_at_stop={owned_pending}; native_deltas_not_restored={not_restored:?}; no numeric restoration is inferred"
        ),
    ))
}
struct DeltaWindow {
    work_until: Instant,
    cleanup_until: Instant,
}

fn run_deltas(
    evaluator: &V2CpuValue,
    model: &Model,
    bound: &BoundNnuecReviewContext,
    root: &Position,
    movements: &[BoardMove],
    window: DeltaWindow,
    record: &mut CaseResult,
) -> Result<()> {
    let DeltaWindow {
        work_until: until,
        cleanup_until,
    } = window;
    let mut rules = root.clone();
    let mut value = evaluator
        .initialize(&rules)
        .map_err(|e| ProbeError::new("NativeInitialize", e))?;
    let initial = observe(evaluator, &value, &rules, record)?;
    let canonical = rebuild(model, bound, root, &[], &rules, until, record)?;
    compare(&initial, &canonical, "initial", 0, until, record)?;
    let mut pending = Vec::new();
    let result = (|| {
        for (index, movement) in movements.iter().enumerate() {
            record.phase = format!("delta:{}", index + 1);
            check_work(until, "DeltaDeadline")?;
            let before = observe(evaluator, &value, &rules, record)?;
            let undo = rules
                .make_move(*movement)
                .map_err(|e| ProbeError::new("RulesMake", e))?;
            let native = match evaluator.apply_delta(&mut value, undo.delta()) {
                Ok(native) => native,
                Err(error) => {
                    if let Err(cleanup) = rules.unmake(undo) {
                        record
                            .cleanup_errors
                            .push(ProbeError::new("RejectedDeltaRulesUnmake", cleanup));
                    }
                    return Err(ProbeError::new("NativeDelta", error));
                }
            };
            record.deltas_applied += 1;
            pending.push((undo, native, before));
            record.peak_owned_undos = record.peak_owned_undos.max(pending.len());
            let actual = observe(evaluator, &value, &rules, record)?;
            record.local_changed |= actual
                .numeric
                .local_accumulator
                .iter()
                .map(|v| v.to_bits())
                .ne(pending
                    .last()
                    .unwrap()
                    .2
                    .numeric
                    .local_accumulator
                    .iter()
                    .map(|v| v.to_bits()));
            let canonical = rebuild(
                model,
                bound,
                root,
                &movements[..=index],
                &rules,
                until,
                record,
            )?;
            compare(
                &actual,
                &canonical,
                "delta_rebuild",
                index + 1,
                until,
                record,
            )?;
        }
        while !pending.is_empty() {
            // Check the original deadline before consuming the owned pair so
            // the failure path can still restore every pending native delta.
            check_work(until, "RestoreDeadline")?;
            let (undo, native, before) = pending.pop().unwrap();
            record.phase = format!("restore:{}", pending.len());
            rules
                .unmake(undo)
                .map_err(|e| ProbeError::new("RulesUnmake", e))?;
            evaluator
                .restore(&mut value, native, &rules)
                .map_err(|e| ProbeError::new("NativeRestore", e))?;
            record.owned_restores += 1;
            let actual = observe(evaluator, &value, &rules, record)?;
            compare(
                &actual,
                &before,
                "owned_before_restore",
                pending.len(),
                until,
                record,
            )?;
            let canonical = rebuild(
                model,
                bound,
                root,
                &movements[..pending.len()],
                &rules,
                until,
                record,
            )?;
            compare(
                &actual,
                &canonical,
                "restore_rebuild",
                pending.len(),
                until,
                record,
            )?;
        }
        let root_ok = rules.snapshot().same_state(&root.snapshot());
        record.final_rules_root_restored = Some(root_ok);
        need(root_ok, "FinalRules", "actual Rules root was not restored")?;
        let actual = observe(evaluator, &value, &rules, record)?;
        let numeric_ok = output_bits(&actual)? == output_bits(&initial)?;
        record.final_numeric_root_restored = Some(numeric_ok);
        compare(&actual, &initial, "final_root", 0, until, record)?;
        Ok(())
    })();
    // On failure perform only the still-owned cleanup, without new readout,
    // canonical inference, a replacement undo, or a synthetic success.
    if result.is_err() {
        while !pending.is_empty() {
            // Do not take the original owned pair or start another cleanup
            // operation after the same outer cleanup cutoff has expired.
            if let Err(error) =
                check_cleanup(cleanup_until, "before_rules_unmake", pending.len(), record)
            {
                record.cleanup_errors.push(error);
                break;
            }
            let (undo, native, _) = pending.pop().unwrap();
            if let Err(error) = rules.unmake(undo) {
                record
                    .cleanup_errors
                    .push(ProbeError::new("FailureRulesUnmake", error));
                if let Err(deadline) = check_cleanup(
                    cleanup_until,
                    "rules_unmake_error",
                    pending.len() + 1,
                    record,
                ) {
                    record.cleanup_errors.push(deadline);
                }
                break;
            }
            if let Err(error) = check_cleanup(
                cleanup_until,
                "before_native_restore",
                pending.len() + 1,
                record,
            ) {
                record.cleanup_errors.push(error);
                break;
            }
            match evaluator.restore(&mut value, native, &rules) {
                Ok(()) => record.failure_cleanup_restores += 1,
                Err(error) => {
                    record
                        .cleanup_errors
                        .push(ProbeError::new("FailureNativeRestore", error));
                    if let Err(deadline) =
                        check_cleanup(cleanup_until, "native_restore_error", pending.len(), record)
                    {
                        record.cleanup_errors.push(deadline);
                    }
                    break;
                }
            }
            if let Err(error) = check_cleanup(
                cleanup_until,
                "native_restore_returned",
                pending.len(),
                record,
            ) {
                record.cleanup_errors.push(error);
                break;
            }
        }
        // The original error is returned unchanged. Remaining undo owners and
        // numeric state are dropped only with this diagnostic owner; they are
        // never reported as restored or reused. The original counters and any
        // CleanupDeadline preserve partial progress without a new readout.
        record.final_rules_root_restored = Some(rules.snapshot().same_state(&root.snapshot()));
    }
    result
}

fn run_case(
    case: &CaseInput,
    model: &Arc<Model>,
    until: Instant,
    cleanup_until: Instant,
    context_mutations: bool,
) -> CaseResult {
    let mut record = CaseResult::new(case);
    let cancel = Arc::new(AtomicBool::new(false));
    let mut actor: Option<PreparationActor> = None;
    let mut stores = PalsStores::new(StoreLimits::default());
    let result = (|| {
        check_work(until, "CaseAdmission")?;
        let root = case.root.rules()?;
        let situation = stores
            .focus_actual_moves(root.snapshot())
            .map_err(|e| ProbeError::new("StoreRoot", e))?;
        let state = stores
            .situations
            .get(situation)
            .map_err(|e| ProbeError::new("StoreRoot", e))?
            .state;
        let mut index = ReviewContextIndex::default();
        let mut revisions = BTreeSet::new();
        for row in &case.diagnostic_records {
            need(
                revisions.insert(row.revision) && row.moves.len() <= MAX_PLIES,
                "DiagnosticRecord",
                "unique revisions and bounded diagnostic lines required",
            )?;
            let line = row
                .moves
                .iter()
                .map(|text| movement(text))
                .collect::<Result<Vec<_>>>()?;
            let raw = RoleRecord {
                revision: row.revision,
                parent_revision: row.parent_revision,
                supersedes_revision: row.supersedes_revision,
                origin_state: state,
                kind: row.kind.kind(),
                line,
                value: None,
                completed_depth: 0,
                score_scope: None,
                cpu_observation: None,
                perspective: row.perspective.color(),
                critical: row.critical,
            };
            index
                .publish(&stores, &raw, None)
                .map_err(|e| ProbeError::new("ReviewPublish", e))?;
        }
        let view = Arc::new(
            index
                .select(
                    &stores,
                    state,
                    ReviewQuerySequences {
                        purpose: RoleQueryPurpose::ValueFresh,
                        prefix: &[],
                        proposal: &[],
                        counterexample: None,
                        legal: &[],
                        divergences: &[],
                    },
                    ReviewSelectionPolicy::default(),
                )
                .map_err(|e| ProbeError::new("ReviewSelect", e))?,
        );
        let selected: BTreeSet<_> = view
            .records_metadata()
            .iter()
            .map(|row| row.source_revision)
            .collect();
        need(
            selected == revisions && view.records().len() == case.diagnostic_records.len(),
            "ReviewSelection",
            "all declared diagnostic records must survive selection; no silent truncation",
        )?;
        let seal = view.seal_digest();
        let seal_hex = hex(seal);
        record.view_seal_sha256 = Some(seal_hex.clone());
        match &case.seal_expectation {
            SealExpectation::ObserveDiagnostic => {}
            SealExpectation::MatchBytes { sha256 } => {
                let matches = digest_text(sha256, 64) && *sha256 == seal_hex;
                record.view_seal_bytes_matched = Some(matches);
                need(
                    matches,
                    "ViewSealBytes",
                    "supplied seal bytes differ; matching bytes would still not recover original PALS authority",
                )?;
            }
        }
        let request = PreparationRequest::admit(
            model,
            Arc::clone(&view),
            &stores,
            state,
            stores.generation(),
            until,
            Arc::clone(&cancel),
        )
        .map_err(|e| ProbeError::new("PreparationAdmit", e))?;
        actor = Some(
            PreparationActor::spawn(Arc::clone(model))
                .map_err(|e| ProbeError::new("PreparationSpawn", e))?,
        );
        let producer = actor.as_ref().unwrap();
        producer
            .try_submit(request)
            .map_err(|(error, _)| ProbeError::new("PreparationSubmit", error))?;
        record.phase = "preparation".into();
        let completion = loop {
            check_work(until, "PreparationDeadline")?;
            if let Some(completion) = producer
                .try_complete()
                .map_err(|e| ProbeError::new("PreparationPoll", e))?
            {
                break completion;
            }
            std::thread::sleep(Duration::from_millis(1));
        };
        need(
            completion.terminal_ack.matches_original(model, &view)
                && completion.provenance.view_seal == seal,
            "PreparationTerminal",
            "actual original actor/view terminal required",
        )?;
        let bound = Arc::new(
            completion
                .result
                .map_err(|e| ProbeError::new("PreparationResult", e))?,
        );
        need(
            bound.validates(&stores, state, stores.generation(), seal),
            "BoundAdmission",
            "actual public Bound must validate original Stores/root/generation/seal",
        )?;
        record.record_scope_sha256 = Some(hex(bound.provenance().record_scope_digest));
        record.numeric_context_sha256 = Some(bound.numeric_context_sha256().into());
        record.context_nonzero = Some(bound.context().values().iter().any(|v| *v != 0.0));
        // Same public constructor used by the original UCI deferred-value and
        // adaptive-port paths. No worker install ACK or PALS consume is issued.
        if context_mutations && case.coverage == "quiet_start" {
            let mut diagnostic = ContextMutationDiagnostic::new();
            let result = run_context_mutations(model, &bound, &view, &root, until, &mut diagnostic);
            if let Err(error) = &result {
                diagnostic.status = "failed";
                diagnostic.error = Some(error.clone());
            }
            record.context_mutation_diagnostic = Some(diagnostic);
            result?;
        }
        let evaluator =
            V2CpuValue::from_bound(Arc::clone(model), Arc::clone(&bound), &stores, seal)
                .map_err(|e| ProbeError::new("NativeValueInstall", e))?;
        let movements = case
            .local_moves
            .iter()
            .map(|text| movement(text))
            .collect::<Result<Vec<_>>>()?;
        run_deltas(
            &evaluator,
            model,
            &bound,
            &root,
            &movements,
            DeltaWindow {
                work_until: until,
                cleanup_until,
            },
            &mut record,
        )
    })();
    // All evaluator/Bound/View/undo locals from the body are now dropped.
    cancel.store(true, Ordering::Release);
    if let Err(error) = result {
        record.error = Some(error);
    }
    if let Some(mut producer) = actor {
        record.preparation_actor_join = "pending".into();
        producer.request_shutdown();
        loop {
            match producer.try_join() {
                Ok(Some(())) => {
                    record.preparation_actor_join = "joined".into();
                    break;
                }
                Ok(None) if Instant::now() < cleanup_until => {
                    std::thread::sleep(Duration::from_millis(1));
                }
                Ok(None) => {
                    record.preparation_actor_join = "unknown_retained_until_process_exit".into();
                    record.cleanup_errors.push(ProbeError::new(
                        "ActorJoinDeadline",
                        "actual actor join unobserved; no later case is permitted",
                    ));
                    // Preserve an unresolved owner rather than block in Drop or
                    // claim release. Root must close this owned process scope.
                    std::mem::forget(producer);
                    break;
                }
                Err(error) => {
                    record.preparation_actor_join = "failed".into();
                    record
                        .cleanup_errors
                        .push(ProbeError::new("ActorJoin", error));
                    break;
                }
            }
        }
    }
    match stores.review_pin_count() {
        Ok(count) => {
            record.review_pins_after_join = Some(count);
            if count != 0 {
                record.cleanup_errors.push(ProbeError::new(
                    "ReviewPins",
                    "original Stores still retain review pins",
                ));
            }
        }
        Err(error) => record
            .cleanup_errors
            .push(ProbeError::new("ReviewPins", error)),
    }
    if record.error.is_none()
        && record.cleanup_errors.is_empty()
        && record.preparation_actor_join == "joined"
    {
        record.status = "numeric_and_owned_undo_verified".into();
        record.phase = "complete".into();
    } else {
        record.status = "failed_or_cleanup_unknown".into();
    }
    record
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    scope: &'static str,
    status: String,
    declared_library_source: Option<String>,
    library_source_verified_by_probe: bool,
    verified_checkpoint_raw_pin: Option<RawPin>,
    verified_request_raw_pin: Option<RawPin>,
    input_owner_or_fd_name_identity_verified: bool,
    actual_model_identity: Option<CpuValueIdentity>,
    context_lineage: &'static str,
    pals_accepted_c_lineage_verified: bool,
    uci_worker_installation_attested: bool,
    product_authority_issued: bool,
    product_physical_completion_attested: bool,
    training_execution_performed: bool,
    performance_evidence: bool,
    cases: Vec<CaseResult>,
    error: Option<ProbeError>,
}
fn report() -> Report {
    Report {
        schema: "rovent.nnuec-native-delta-result/1",
        scope: "diagnostic_actual_checkpoint_cpu_native_delta_and_owned_undo_only",
        status: "not_started".into(),
        declared_library_source: None,
        library_source_verified_by_probe: false,
        verified_checkpoint_raw_pin: None,
        verified_request_raw_pin: None,
        input_owner_or_fd_name_identity_verified: false,
        actual_model_identity: None,
        context_lineage: "diagnostic_rules_admission",
        pals_accepted_c_lineage_verified: false,
        uci_worker_installation_attested: false,
        product_authority_issued: false,
        product_physical_completion_attested: false,
        training_execution_performed: false,
        performance_evidence: false,
        cases: Vec::new(),
        error: None,
    }
}
fn execute(record: &mut Report) -> Result<()> {
    let args = arguments()?;
    let started = Instant::now();
    let until = started
        .checked_add(Duration::from_millis(args.work_ms))
        .ok_or_else(|| ProbeError::new("Window", "work deadline overflow"))?;
    let cleanup_until = until
        .checked_add(Duration::from_millis(args.cleanup_ms))
        .ok_or_else(|| ProbeError::new("Window", "cleanup deadline overflow"))?;
    let checkpoint = read_pinned(&args.checkpoint)?;
    record.verified_checkpoint_raw_pin = Some(args.checkpoint);
    check_work(until, "CheckpointDeadline")?;
    let model = Arc::new(
        Model::from_reader(checkpoint.as_slice())
            .map_err(|e| ProbeError::new("CheckpointLoad", e))?,
    );
    drop(checkpoint);
    record.actual_model_identity = Some(model.identity().clone());
    let request = read_pinned(&args.request)?;
    record.verified_request_raw_pin = Some(args.request);
    let input: Input =
        serde_json::from_slice(&request).map_err(|e| ProbeError::new("RequestDecode", e))?;
    drop(request);
    input.validate(&model)?;
    record.declared_library_source = Some(input.declared_library_source.clone());
    record.status = "running".into();
    for case in &input.cases {
        let result = run_case(case, &model, until, cleanup_until, args.context_mutations);
        let passed = result.status == "numeric_and_owned_undo_verified";
        record.cases.push(result);
        if !passed {
            return Err(ProbeError::new(
                "CaseStopped",
                "first numeric/admission/cleanup failure retained; later cases were not executed",
            ));
        }
    }
    record.status = "diagnostic_suite_verified".into();
    Ok(())
}
fn main() -> std::process::ExitCode {
    let mut record = report();
    let result = execute(&mut record);
    if let Err(error) = result {
        record.status = "failed".into();
        record.error = Some(error);
    }
    let passed = record.status == "diagnostic_suite_verified";
    let encoded = match serde_json::to_vec(&record) {
        Ok(encoded) if encoded.len() < OUTPUT_CAP => encoded,
        Ok(_) => {
            eprintln!("NNUEC native delta probe output exceeded explicit cap; no success");
            return std::process::ExitCode::FAILURE;
        }
        Err(error) => {
            eprintln!("NNUEC native delta probe output failed: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let mut stdout = std::io::stdout().lock();
    if stdout
        .write_all(&encoded)
        .and_then(|_| stdout.write_all(b"\n"))
        .is_err()
    {
        return std::process::ExitCode::FAILURE;
    }
    if passed {
        std::process::ExitCode::SUCCESS
    } else {
        std::process::ExitCode::FAILURE
    }
}
