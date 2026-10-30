//! One physical V3 owner for selected immutable review tensors.
//!
//! The public graph is role-neutral, while one scalar-routed P/C graph owns
//! its private experts. Deployment never loads the optional V graph. All L
//! buckets use these same sessions. Failed CUDA Run retains every input,
//! public pin and session; a logical ticket cancellation grants no release.
use super::*;
use crate::pals_model::virtual_review::{
    self as model, PalsVirtualReviewInput, PreparedVirtualReviewTensors, ReviewExecutionIdentity,
};
use sha2::{Digest as _, Sha256};
#[cfg(feature = "experimental-io-binding")]
mod binding;
mod legacy_manifest;
mod manifest;
#[cfg(feature = "native-validation-diagnostics")]
pub mod native_validation_diagnostics;
mod owned_sources;
pub use legacy_manifest::{
    validate_pals_legacy_export_manifest, LegacyPalsExportManifestSummary, LegacyPalsGraphPin,
};
pub use owned_sources::{VirtualReviewOwnedSources, VirtualReviewProfileDirectory};
mod tickets;
use rz_contracts::pals_adaptive::RoleTicketEnvelope;
pub use tickets::VirtualReviewCompletion;

pub const PUBLIC_LAYOUT: &str = "virtual_review_public_v1";
pub const SHARED_LAYOUT: &str = "virtual_review_shared_pc_if_v1";
pub const WARM_LAYOUT: &str = "virtual_review_shared_pc_if_approx_warm_v1";

/// Best-effort host phase evidence only. This owns no native resource and never
/// grants a provider fence, numerical result, memory measurement or retry.
/// The explicit environment switch is read once per backend load, not per Run.
struct VirtualReviewPhaseTrace {
    origin: Option<std::time::Instant>,
    provider: &'static str,
    emitted: u8,
    first_run_graphs: u8,
}
impl VirtualReviewPhaseTrace {
    fn new(provider: Provider) -> Self {
        let enabled = std::env::var_os("RZ_PALS_VIRTUAL_REVIEW_PHASE_TRACE")
            .is_some_and(|value| value == "1");
        Self {
            origin: enabled.then(std::time::Instant::now),
            provider: match provider {
                Provider::Cpu => "cpu",
                Provider::Cuda { .. } => "cuda",
            },
            emitted: 0,
            first_run_graphs: 0,
        }
    }
    fn role(role: &str) -> &'static str {
        match role {
            "public" => "public",
            "shared_pc" => "shared_pc",
            "shared_warm" => "shared_warm",
            "validator" => "validator",
            _ => "other",
        }
    }
    fn record(&mut self, phase: &'static str, role: &'static str, serialized_bytes: Option<usize>) {
        use std::io::Write;
        let Some(origin) = self.origin else {
            return;
        };
        if self.emitted >= 32 {
            return;
        }
        self.emitted += 1;
        // Fixed identifiers and scalar metadata only. No asset path, model,
        // tensor, error payload or new output file is retained by this channel.
        #[derive(Serialize)]
        struct Record {
            schema: &'static str,
            ordinal: u8,
            phase: &'static str,
            graph_role: &'static str,
            configured_backend_provider: &'static str,
            elapsed_ns: u128,
            serialized_bytes: Option<usize>,
            physical_completion: &'static str,
            performance_evidence_eligible: bool,
        }
        let record = Record {
            schema: "rz-pals-virtual-review-host-phases/1",
            ordinal: self.emitted,
            phase,
            graph_role: role,
            configured_backend_provider: self.provider,
            elapsed_ns: origin.elapsed().as_nanos(),
            serialized_bytes,
            physical_completion: "unknown",
            performance_evidence_eligible: false,
        };
        // This distinct stderr prefix cannot become the stdout numeric status.
        // A missing/partial line is missing evidence, never a backend failure.
        let mut stderr = std::io::stderr().lock();
        if write!(stderr, "RZ_PALS_VIRTUAL_REVIEW_PHASE_JSON=").is_ok()
            && serde_json::to_writer(&mut stderr, &record).is_ok()
        {
            let _ = writeln!(stderr);
        }
        let _ = stderr.flush();
    }
    fn first_run_entered(&mut self, role: &'static str) -> bool {
        if self.origin.is_none() {
            return false;
        }
        let bit = match role {
            "public" => 1,
            "shared_pc" => 2,
            "validator" => 4,
            "shared_warm" => 8,
            _ => return false,
        };
        if self.first_run_graphs & bit != 0 {
            return false;
        }
        self.first_run_graphs |= bit;
        self.record("first_run_entered", role, None);
        true
    }
    fn first_run_returned(&mut self, first: bool, role: &'static str, ok: bool) {
        if first {
            // An API return is not a physical GPU completion/fence claim.
            self.record(
                if ok {
                    "first_run_returned_ok"
                } else {
                    "first_run_returned_error"
                },
                role,
                None,
            );
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VirtualReviewExecutionMode {
    Deployment,
    Collection,
}
/// Explicit artifact admission. No automatic Fresh/Warm conversion is allowed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VirtualReviewArtifactKind {
    Fresh,
    ApproxWarm,
}
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct VirtualReviewGraphPin {
    pub role: String,
    pub file: String,
    /// Manifest declaration only; graph bytes have not been read by preflight.
    pub sha256: [u8; 32],
    pub opset: u32,
}
/// Pure validation summary of actual bounded manifest bytes. This is not a
/// session/provider/numerical acceptance or proof of graph/checkpoint contents.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct VirtualReviewExportManifestSummary {
    manifest_sha256: [u8; 32],
    manifest_bytes: usize,
    pub artifact_kind: VirtualReviewArtifactKind,
    pub config: PalsModelConfig,
    pub model_semantics: String,
    pub encoding_schema: String,
    pub model_profile: PalsModelProfile,
    pub checkpoint_sha256: [u8; 32],
    pub roles: Vec<String>,
    pub validator_present: bool,
    pub layout: String,
    pub graphs: Vec<VirtualReviewGraphPin>,
    pub trained: bool,
    pub training_steps: u64,
    pub rules_input_profile: String,
    pub rules_input_semantic_sha256: [u8; 32],
    pub rules_encoder_source_sha256: [u8; 32],
    /// Manifest declaration of actual descriptor bytes; startup independently
    /// hashes its held Rules descriptor and joins that pin before any child.
    pub rules_profile_descriptor_sha256: [u8; 32],
}
impl VirtualReviewExportManifestSummary {
    pub fn manifest_sha256(&self) -> [u8; 32] {
        self.manifest_sha256
    }
    pub fn manifest_bytes(&self) -> usize {
        self.manifest_bytes
    }
}
/// Fresh-only pre-child admission. No files, native libraries, sessions or
/// device execution are accessed; callers provide the actual manifest bytes.
pub fn validate_virtual_review_export_manifest(
    bytes: &[u8],
    mode: VirtualReviewExecutionMode,
) -> Result<VirtualReviewExportManifestSummary, BackendError> {
    validate_virtual_review_export_manifest_kind(bytes, mode, VirtualReviewArtifactKind::Fresh)
}
/// Separate explicit Warm admission reuses the registered strict parser.
pub fn validate_virtual_review_export_manifest_kind(
    bytes: &[u8],
    mode: VirtualReviewExecutionMode,
    kind: VirtualReviewArtifactKind,
) -> Result<VirtualReviewExportManifestSummary, BackendError> {
    let value = match kind {
        VirtualReviewArtifactKind::Fresh => manifest::Manifest::parse(bytes)?,
        VirtualReviewArtifactKind::ApproxWarm => manifest::Manifest::parse_warm(bytes)?,
    };
    if mode == VirtualReviewExecutionMode::Deployment && value.validator_present {
        return Err(fail(
            K::UnsupportedModel,
            S::Admission,
            "V3 Deployment preflight rejects a V-containing manifest",
        ));
    }
    let checkpoint_sha256 = value.epoch()?;
    let rules_input_semantic_sha256 = parse_sha256(&value.rules_input_semantic_sha256)?;
    let rules_encoder_source_sha256 = parse_sha256(&value.rules_encoder_source_sha256)?;
    let rules_profile_descriptor_sha256 = parse_sha256(&value.rules_profile_descriptor_sha256)?;
    let graphs = value
        .graphs
        .into_iter()
        .map(|graph| {
            Ok(VirtualReviewGraphPin {
                role: graph.role,
                file: graph.file,
                sha256: parse_sha256(&graph.sha256)?,
                opset: graph.opset,
            })
        })
        .collect::<Result<Vec<_>, BackendError>>()?;
    Ok(VirtualReviewExportManifestSummary {
        manifest_sha256: asset::sha256(bytes),
        manifest_bytes: bytes.len(),
        artifact_kind: kind,
        config: value.config,
        model_semantics: value.model_semantics,
        encoding_schema: value.encoding_schema,
        model_profile: value.model_profile,
        checkpoint_sha256,
        roles: value.roles,
        validator_present: value.validator_present,
        layout: value.layout,
        graphs,
        trained: value.trained,
        training_steps: value.training_steps,
        rules_input_profile: value.rules_input_profile,
        rules_input_semantic_sha256,
        rules_encoder_source_sha256,
        rules_profile_descriptor_sha256,
    })
}
struct ActiveWarm {
    initial_latent: Tensor<f32>,
    warm_start: Tensor<bool>,
}

struct Active {
    host_bytes: u64,
    role: Tensor<bool>,
    anchor_board: Tensor<i64>,
    anchor_metadata: Tensor<f32>,
    anchor_mask: Tensor<bool>,
    records: Tensor<f32>,
    record_anchor: Tensor<i64>,
    record_mask: Tensor<bool>,
    record_line_tokens: Tensor<i64>,
    record_line_mask: Tensor<bool>,
    record_relations: Tensor<i64>,
    query: Tensor<f32>,
    query_anchor: Tensor<i64>,
    query_line_tokens: Tensor<i64>,
    query_line_mask: Tensor<bool>,
    candidates: Tensor<i64>,
    candidate_mask: Tensor<bool>,
    divergences: Tensor<f32>,
    divergence_mask: Tensor<bool>,
    warm: Option<ActiveWarm>,
}
impl Active {
    fn new(
        value: PreparedVirtualReviewTensors,
        role: PalsRole,
        warm: Option<(Vec<f32>, bool)>,
    ) -> Result<Self, BackendError> {
        let (a, r, c, d, l) = (
            value.anchors,
            value.records_count,
            value.candidates_count,
            value.divergences_count,
            value.length_bucket,
        );
        let role_values = vec![role == PalsRole::Critic];
        let query_values = value.query.to_vec();
        let query_anchors = value.query_anchor.to_vec();
        // Prepared arrays become owned Vec backing in OrtValues. Count actual
        // capacities before the move, rather than assuming allocation == len.
        let mut host_bytes = value.owned_bytes() as u64 - (8 * 4 + 3 * 8)
            + role_values.capacity() as u64
            + query_values.capacity() as u64 * 4
            + query_anchors.capacity() as u64 * 8;
        let error = |e| {
            native(
                CauseCode::TensorCreate,
                "cannot own selected-review tensor",
                e,
            )
        };
        let warm = warm
            .map(|(latent, mode)| {
                if latent.len() != 16 * 384 || latent.iter().any(|v| !v.is_finite()) {
                    return Err(fail(
                        K::InvalidInput,
                        S::Admission,
                        "V3 Warm input seed must be finite and correctly shaped",
                    ));
                }
                let mode = vec![mode];
                host_bytes += latent.capacity() as u64 * 4 + mode.capacity() as u64;
                Ok(ActiveWarm {
                    initial_latent: Tensor::from_array(([1, 16, 384], latent)).map_err(error)?,
                    warm_start: Tensor::from_array(([] as [usize; 0], mode)).map_err(error)?,
                })
            })
            .transpose()?;
        Ok(Self {
            host_bytes,
            warm,
            role: Tensor::from_array(([] as [usize; 0], role_values)).map_err(error)?,
            anchor_board: Tensor::from_array(([1, a, 64], value.anchor_board)).map_err(error)?,
            anchor_metadata: Tensor::from_array(([1, a, 16], value.anchor_metadata))
                .map_err(error)?,
            anchor_mask: Tensor::from_array(([1, a], value.anchor_mask)).map_err(error)?,
            records: Tensor::from_array(([1, r, 8], value.records)).map_err(error)?,
            record_anchor: Tensor::from_array(([1, r], value.record_anchor)).map_err(error)?,
            record_mask: Tensor::from_array(([1, r], value.record_mask)).map_err(error)?,
            record_line_tokens: Tensor::from_array(([1, r, l, 3], value.record_line_tokens))
                .map_err(error)?,
            record_line_mask: Tensor::from_array(([1, r, l], value.record_line_mask))
                .map_err(error)?,
            record_relations: Tensor::from_array(([1, r, 2], value.record_relations))
                .map_err(error)?,
            query: Tensor::from_array(([1, 8], query_values)).map_err(error)?,
            query_anchor: Tensor::from_array(([1, 3], query_anchors)).map_err(error)?,
            query_line_tokens: Tensor::from_array(([1, 3, l, 3], value.query_line_tokens))
                .map_err(error)?,
            query_line_mask: Tensor::from_array(([1, 3, l], value.query_line_mask))
                .map_err(error)?,
            candidates: Tensor::from_array(([1, c, 3], value.candidates)).map_err(error)?,
            candidate_mask: Tensor::from_array(([1, c], value.candidate_mask)).map_err(error)?,
            divergences: Tensor::from_array(([1, d, 8], value.divergence_features))
                .map_err(error)?,
            divergence_mask: Tensor::from_array(([1, d], value.divergence_mask)).map_err(error)?,
        })
    }
}

struct PublicMemory {
    key: Tensor<f32>,
    value: Tensor<f32>,
    mask: Tensor<bool>,
    anchors: Tensor<f32>,
}
const MAX_PUBLIC_BYTES: u64 = ((66 + model::MAX_RECORDS) * 2 * 64 * 4 * 2
    + (66 + model::MAX_RECORDS)
    + model::MAX_ANCHORS * 384 * 4) as u64
    + std::mem::size_of::<PublicMemory>() as u64;

/// NN-zero snapshot of known Rust-owned capacity. Native prepack, allocator
/// workspace, residency and peak VRAM remain unobserved, not declared zero.
#[derive(Clone, Debug)]
pub struct VirtualReviewOwnerSnapshot {
    pub game_generation: u64,
    pub active_prepared_host_bytes: u64,
    /// Fixed CPU output Vec capacities. Native binding metadata is unmeasured.
    pub bound_output_host_backing_bytes: u64,
    pub public_bank: MemoryBankSnapshot,
    pub native_sessions: usize,
    pub physical_completion_unknown: bool,
    pub accepted_seed_host_backing_bytes: u64,
    pub native_resident_parameter_bytes: Option<u64>,
    pub vram_peak_bytes: Option<u64>,
}

#[allow(clippy::large_enum_variant)]
pub enum VirtualReviewNativeCommand {
    Evaluate(VirtualReviewInvocation),
    EvaluateTicket {
        ticket: RoleTicketEnvelope,
        invocation: VirtualReviewInvocation,
    },
    #[cfg(feature = "native-validation-diagnostics")]
    EvaluateTicketForValidation {
        ticket: RoleTicketEnvelope,
        execution: rz_contracts::ExecutionId,
        invocation: VirtualReviewInvocation,
    },
    AcceptTicket {
        ticket: RoleTicketEnvelope,
        invocation: VirtualReviewInvocation,
        completion: VirtualReviewCompletion,
    },
    DiscardTicket(RoleTicketEnvelope),
    /// End root authority after physical idle without tearing down sessions.
    QuiesceRoot,
    ResetTo(u64),
    NewGame,
    SnapshotStats,
    SnapshotOwner,
    VerifyRuntime,
    VerifyCudaPlacement,
    VerifyVirtualCudaPlacement,
}
/// The immutable view's exact history comes from Rules/ticket authority. It is
/// deliberately outside numerical model features and must not be synthesized
/// from FEN. The backend validates its own model/backend portion at admission.
#[derive(Clone, Debug)]
pub struct VirtualReviewInvocation {
    pub input: PalsVirtualReviewInput,
    pub identity: ReviewExecutionIdentity,
    /// Exact Rules/history at the question endpoint (anchor + query prefix),
    /// independently captured by the bridge. identity.rules_history remains
    /// the anchor-zero root and must not be replaced by this endpoint.
    pub request_history: [u8; 32],
}

/// Deltas captured inside this single physical owner around one ticket Run.
/// The role-graph counters exclude the public-memory graph. A cache lookup,
/// successful output validation and an accepted search result are distinct.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VirtualReviewTicketOwnerAccounting {
    public_cache_hits: u64,
    public_cache_misses: u64,
    public_nn_runs_attempted: u64,
    public_nn_runs_completed: u64,
    public_nn_runs_failed_known: u64,
    role_nn_runs_attempted: u64,
    role_nn_runs_completed: u64,
    role_nn_runs_failed_known: u64,
    validator_nn_runs_attempted: u64,
    validator_nn_runs_completed: u64,
    validator_nn_runs_failed_known: u64,
}
impl VirtualReviewTicketOwnerAccounting {
    // There is deliberately no public constructor from caller-provided counts.
    fn between(
        before: &PalsBackendStats,
        after: &PalsBackendStats,
        role: PalsRole,
        physical_completion_known: bool,
    ) -> Option<Self> {
        let completion = PalsNNInputEvidence::between(before, after, physical_completion_known);
        if !completion.complete {
            return None;
        }
        let hits = after
            .public_cache_hits
            .checked_sub(before.public_cache_hits)?;
        let misses = after
            .public_cache_misses
            .checked_sub(before.public_cache_misses)?;
        let public_attempted = after
            .public_nn_runs_attempted
            .checked_sub(before.public_nn_runs_attempted)?;
        let public_completed = after
            .public_nn_runs_completed
            .checked_sub(before.public_nn_runs_completed)?;
        let public_failed = after
            .public_nn_runs_failed_known
            .checked_sub(before.public_nn_runs_failed_known)?;
        let attempted = after
            .role_nn_runs_attempted
            .checked_sub(before.role_nn_runs_attempted)?;
        let completed = after
            .role_nn_runs_completed
            .checked_sub(before.role_nn_runs_completed)?;
        let failed = after
            .role_nn_runs_failed_known
            .checked_sub(before.role_nn_runs_failed_known)?;
        // One EvaluateTicket command is B1. Counter mixing from another
        // invocation cannot be reported as this original ticket's observation.
        if hits.checked_add(misses)? > 1
            || public_attempted > 1
            || public_completed.checked_add(public_failed)? > public_attempted
            || attempted > 1
            || completed.checked_add(failed)? > attempted
        {
            return None;
        }
        let is_validator = role == PalsRole::Validator;
        Some(Self {
            public_cache_hits: hits,
            public_cache_misses: misses,
            public_nn_runs_attempted: public_attempted,
            public_nn_runs_completed: public_completed,
            public_nn_runs_failed_known: public_failed,
            role_nn_runs_attempted: attempted,
            role_nn_runs_completed: completed,
            role_nn_runs_failed_known: failed,
            validator_nn_runs_attempted: if is_validator { attempted } else { 0 },
            validator_nn_runs_completed: if is_validator { completed } else { 0 },
            validator_nn_runs_failed_known: if is_validator { failed } else { 0 },
        })
    }
    pub fn public_cache_hits(&self) -> u64 {
        self.public_cache_hits
    }
    pub fn public_cache_misses(&self) -> u64 {
        self.public_cache_misses
    }
    pub fn public_nn_runs_attempted(&self) -> u64 {
        self.public_nn_runs_attempted
    }
    pub fn public_nn_runs_completed(&self) -> u64 {
        self.public_nn_runs_completed
    }
    pub fn public_nn_runs_failed_known(&self) -> u64 {
        self.public_nn_runs_failed_known
    }
    pub fn role_nn_runs_attempted(&self) -> u64 {
        self.role_nn_runs_attempted
    }
    pub fn role_nn_runs_completed(&self) -> u64 {
        self.role_nn_runs_completed
    }
    pub fn role_nn_runs_failed_known(&self) -> u64 {
        self.role_nn_runs_failed_known
    }
    pub fn validator_nn_runs_attempted(&self) -> u64 {
        self.validator_nn_runs_attempted
    }
    pub fn validator_nn_runs_completed(&self) -> u64 {
        self.validator_nn_runs_completed
    }
    pub fn validator_nn_runs_failed_known(&self) -> u64 {
        self.validator_nn_runs_failed_known
    }
}

// Completion owns the original output/error and its single-owner accounting.
// Keep them in the same bounded worker message instead of adding a separate
// allocation on every physical completion solely to equalize variant sizes.
#[allow(clippy::large_enum_variant)]
pub enum VirtualReviewNativeResult {
    Evaluation(PalsRawOutput),
    TicketEvaluationWithEvidence {
        outcome: Result<VirtualReviewCompletion, BackendError>,
        actual_completed_nn_inputs_delta: Option<u64>,
        complete: bool,
        owner_accounting: Option<VirtualReviewTicketOwnerAccounting>,
    },
    TicketAccepted,
    TicketDiscarded,
    RootQuiesced {
        game_generation: u64,
    },
    ResetTo {
        game_generation: u64,
    },
    NewGame,
    Stats(PalsBackendStats),
    Owner(VirtualReviewOwnerSnapshot),
    RuntimeVerified,
    CudaPlacementVerified(Box<PalsCudaPlacementWitness>),
    VirtualCudaPlacementVerified(Box<VirtualReviewCudaPlacementWitness>),
}

/// V3 evidence is owned separately from the legacy P/C-only witness domain.
struct VirtualControlAudit {
    policy: PalsCudaControlPolicy,
    profile_root: std::path::PathBuf,
    // Present only on the new descriptor-capability loader. Its procfd parent
    // is retained for all session/profile I/O and physical-unknown quarantine.
    profile_directory: Option<VirtualReviewProfileDirectory>,
    initialization: Arc<[PalsGraphPlacement]>,
    proposer_seen: bool,
    critic_seen: bool,
    validator_seen: bool,
    warm_seed_seen: bool,
    witness: Option<VirtualReviewCudaPlacementWitness>,
    rejected: Option<BackendError>,
}

pub struct PalsVirtualReviewOnnxBackend {
    public: Option<Session>,
    shared_pc: Option<Session>,
    validator: Option<Session>,
    applied_session_settings: Arc<[PalsAppliedSessionSettings]>,
    runtime: Option<OrtRuntime>,
    config: PalsOnnxConfig,
    model_config: PalsModelConfig,
    epoch: [u8; 32],
    manifest_digest: [u8; 32],
    trained: bool,
    rules_semantics: [u8; 32],
    backend_digest: [u8; 32],
    memory: Option<MemoryBank<PublicMemory>>,
    active_memory: Option<MemoryPin<PublicMemory>>,
    active: Option<Active>,
    #[cfg(feature = "experimental-io-binding")]
    active_binding: Option<binding::Binding>,
    stats: PalsBackendStats,
    generation: u64,
    mapping: RefCell<CudaMappingAudit>,
    audit: Option<VirtualControlAudit>,
    quarantine: Option<BackendError>,
    residency: PalsSessionResidency,
    mode: VirtualReviewExecutionMode,
    warm: bool,
    seeds: tickets::TicketSeeds,
    // Keep descriptor/snapshot authority with the same physical session owner.
    // A CUDA unknown-completion path must retain this Arc, not just tensors.
    sources: Option<Arc<dyn VirtualReviewOwnedSources>>,
    initialization: Arc<[PalsGraphPlacement]>,
    phase_trace: VirtualReviewPhaseTrace,
    #[cfg(feature = "native-validation-diagnostics")]
    validation_trace: Option<native_validation_diagnostics::GpuLifecycleTrace>,
}
impl PalsVirtualReviewOnnxBackend {
    pub fn load(
        path: &Path,
        expected_sha256: &str,
        runtime: OrtRuntime,
        config: PalsOnnxConfig,
    ) -> Result<Self, BackendError> {
        Self::load_inner(
            path,
            expected_sha256,
            runtime,
            config,
            None,
            VirtualReviewExecutionMode::Deployment,
            false,
        )
    }
    pub fn load_collection(
        path: &Path,
        expected_sha256: &str,
        runtime: OrtRuntime,
        config: PalsOnnxConfig,
    ) -> Result<Self, BackendError> {
        Self::load_inner(
            path,
            expected_sha256,
            runtime,
            config,
            None,
            VirtualReviewExecutionMode::Collection,
            false,
        )
    }
    /// Explicit ApproxWarm graph domain. Unregistered caller latent input is
    /// never accepted. A missing accepted seed dispatches declared Fresh mode.
    pub fn load_approx_warm(
        path: &Path,
        expected_sha256: &str,
        runtime: OrtRuntime,
        config: PalsOnnxConfig,
    ) -> Result<Self, BackendError> {
        Self::load_inner(
            path,
            expected_sha256,
            runtime,
            config,
            None,
            VirtualReviewExecutionMode::Deployment,
            true,
        )
    }
    pub fn load_with_cuda_control_policy(
        path: &Path,
        expected_sha256: &str,
        runtime: OrtRuntime,
        config: PalsOnnxConfig,
        policy: PalsCudaControlPolicy,
        profile_root: &Path,
    ) -> Result<Self, BackendError> {
        if !matches!(config.provider, Provider::Cuda { .. }) || config.device_public_memory {
            return Err(fail(
                K::BackendUnavailable,
                S::Admission,
                "selected-review control requires explicit CUDA and host public memory",
            ));
        }
        Self::load_inner(
            path,
            expected_sha256,
            runtime,
            config,
            Some((policy, cuda_control::owned_profile_root(profile_root)?)),
            VirtualReviewExecutionMode::Deployment,
            false,
        )
    }
    pub fn load_collection_with_cuda_control_policy(
        path: &Path,
        expected_sha256: &str,
        runtime: OrtRuntime,
        config: PalsOnnxConfig,
        policy: PalsCudaControlPolicy,
        profile_root: &Path,
    ) -> Result<Self, BackendError> {
        if !matches!(config.provider, Provider::Cuda { .. }) || config.device_public_memory {
            return Err(fail(
                K::BackendUnavailable,
                S::Admission,
                "V3 collection control requires explicit CUDA/host public memory",
            ));
        }
        Self::load_inner(
            path,
            expected_sha256,
            runtime,
            config,
            Some((policy, cuda_control::owned_profile_root(profile_root)?)),
            VirtualReviewExecutionMode::Collection,
            false,
        )
    }
    /// Explicit V3 ApproxWarm inventory. A legacy Fresh P/C inventory does not
    /// authorize a Warm graph, its scalar dispatch, or its accepted-seed branch.
    pub fn load_warm_with_cuda_control_policy(
        path: &Path,
        expected_sha256: &str,
        runtime: OrtRuntime,
        config: PalsOnnxConfig,
        policy: PalsCudaControlPolicy,
        profile_root: &Path,
    ) -> Result<Self, BackendError> {
        if !matches!(config.provider, Provider::Cuda { .. }) || config.device_public_memory {
            return Err(fail(
                K::BackendUnavailable,
                S::Admission,
                "V3 Warm control requires explicit CUDA/host public memory",
            ));
        }
        Self::load_inner(
            path,
            expected_sha256,
            runtime,
            config,
            Some((policy, cuda_control::owned_profile_root(profile_root)?)),
            VirtualReviewExecutionMode::Deployment,
            true,
        )
    }
    fn load_inner(
        path: &Path,
        expected_sha256: &str,
        runtime: OrtRuntime,
        config: PalsOnnxConfig,
        audit: Option<(PalsCudaControlPolicy, std::path::PathBuf)>,
        mode: VirtualReviewExecutionMode,
        warm: bool,
    ) -> Result<Self, BackendError> {
        Self::load_sources_inner(
            Arc::new(owned_sources::PathSources::new(path)),
            expected_sha256,
            runtime,
            config,
            audit,
            mode,
            warm,
            None,
        )
    }
    /// Consume actual manifest/graph bytes from a bootstrap-owned read-only
    /// descriptor inventory. Snapshot filenames may differ; the unchanged
    /// original manifest role/file/digest is passed to the source owner. Every
    /// returned byte buffer is independently checked before ORT admission.
    #[allow(clippy::too_many_arguments)]
    pub fn load_owned_sources(
        sources: Arc<dyn VirtualReviewOwnedSources>,
        expected_manifest_sha256: &str,
        runtime: OrtRuntime,
        config: PalsOnnxConfig,
        mode: VirtualReviewExecutionMode,
        kind: VirtualReviewArtifactKind,
        control: Option<(PalsCudaControlPolicy, std::path::PathBuf)>,
    ) -> Result<Self, BackendError> {
        if kind == VirtualReviewArtifactKind::ApproxWarm
            && mode != VirtualReviewExecutionMode::Deployment
        {
            return Err(fail(
                K::UnsupportedModel,
                S::Admission,
                "registered V3 ApproxWarm is a PC-only Deployment artifact; Collection is unsupported",
            ));
        }
        if control.is_some()
            && (!matches!(config.provider, Provider::Cuda { .. }) || config.device_public_memory)
        {
            return Err(fail(
                K::BackendUnavailable,
                S::Admission,
                "owned V3 control requires explicit CUDA and host public memory",
            ));
        }
        config.validate_execution_selection(runtime.native_loading_profile(), control.is_some())?;
        let audit = control
            .map(|(policy, path)| {
                Ok::<_, BackendError>((policy, cuda_control::owned_profile_root(&path)?))
            })
            .transpose()?;
        Self::load_sources_inner(
            sources,
            expected_manifest_sha256,
            runtime,
            config,
            audit,
            mode,
            kind == VirtualReviewArtifactKind::ApproxWarm,
            None,
        )
    }
    /// Descriptor-capability route for Linux startup owners. This never calls
    /// the legacy pathname canonicalize/create-dir helper. Ordinary path APIs
    /// remain separate; a held directory owner is required for CUDA profiling.
    #[allow(clippy::too_many_arguments)]
    pub fn load_owned_sources_with_profile(
        sources: Arc<dyn VirtualReviewOwnedSources>,
        expected_manifest_sha256: &str,
        runtime: OrtRuntime,
        config: PalsOnnxConfig,
        mode: VirtualReviewExecutionMode,
        kind: VirtualReviewArtifactKind,
        control: Option<(PalsCudaControlPolicy, VirtualReviewProfileDirectory)>,
    ) -> Result<Self, BackendError> {
        if kind == VirtualReviewArtifactKind::ApproxWarm
            && mode != VirtualReviewExecutionMode::Deployment
        {
            return Err(fail(
                K::UnsupportedModel,
                S::Admission,
                "registered V3 ApproxWarm is a PC-only Deployment artifact; Collection is unsupported",
            ));
        }
        if control.is_some()
            && (!matches!(config.provider, Provider::Cuda { .. }) || config.device_public_memory)
        {
            return Err(fail(
                K::BackendUnavailable,
                S::Admission,
                "owned V3 profile capability requires explicit CUDA and host public memory",
            ));
        }
        config.validate_execution_selection(runtime.native_loading_profile(), control.is_some())?;
        let (audit, profile_directory) = match control {
            Some((policy, directory)) => {
                directory.validate_live()?;
                let root = directory.path().to_owned();
                (Some((policy, root)), Some(directory))
            }
            None => (None, None),
        };
        Self::load_sources_inner(
            sources,
            expected_manifest_sha256,
            runtime,
            config,
            audit,
            mode,
            kind == VirtualReviewArtifactKind::ApproxWarm,
            profile_directory,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn load_sources_inner(
        sources: Arc<dyn VirtualReviewOwnedSources>,
        expected_sha256: &str,
        runtime: OrtRuntime,
        config: PalsOnnxConfig,
        audit: Option<(PalsCudaControlPolicy, std::path::PathBuf)>,
        mode: VirtualReviewExecutionMode,
        warm: bool,
        profile_directory: Option<VirtualReviewProfileDirectory>,
    ) -> Result<Self, BackendError> {
        let mut phase_trace = VirtualReviewPhaseTrace::new(config.provider);
        phase_trace.record("backend_load_entered", "other", None);
        config.validate_execution_selection(runtime.native_loading_profile(), audit.is_some())?;
        #[cfg(not(feature = "experimental-io-binding"))]
        if warm && matches!(config.provider, Provider::Cuda { .. }) {
            return Err(fail(
                K::BackendUnavailable,
                S::Admission,
                "V3 CUDA ApproxWarm requires compiled fixed I/O binding/fence support",
            ));
        }
        if config.device_public_memory {
            return Err(fail(
                K::BackendUnavailable,
                S::Admission,
                "legacy fixed packing cannot select V3 dynamic review ABI",
            ));
        }
        let manifest_digest = parse_sha256(expected_sha256)?;
        let bytes = owned_sources::read_manifest_checked(
            sources.as_ref(),
            manifest_digest,
            manifest::MAX_MANIFEST_BYTES_V3,
        )?;
        // read_manifest_checked has independently hashed these exact bytes.
        // The verified expected value now equals that actual digest; avoid a
        // second pass over the same immutable serialization buffer.
        let manifest = if warm {
            manifest::Manifest::parse_warm(&bytes)?
        } else {
            manifest::Manifest::parse(&bytes)?
        };
        if mode == VirtualReviewExecutionMode::Deployment && manifest.validator_present {
            return Err(fail(
                K::UnsupportedModel,
                S::Admission,
                "Deployment rejects a V-containing manifest before any V graph read/session load",
            ));
        }
        if let Some((policy, _)) = &audit {
            let layout = policy.virtual_review_layout().ok_or_else(|| {
                fail(
                    K::UnsupportedModel,
                    S::Admission,
                    "a legacy inventory cannot authorize V3 selected-review graphs",
                )
            })?;
            let expected_roles: &[&str] = if warm {
                &["public", "shared_warm"]
            } else if mode == VirtualReviewExecutionMode::Collection {
                &["public", "shared_pc", "validator"]
            } else {
                &["public", "shared_pc"]
            };
            if layout.artifact_kind() != if warm { "approx_warm" } else { "fresh" }
                || layout.layout_name() != manifest.layout
                || layout.graph_roles() != expected_roles
                || manifest
                    .graphs
                    .iter()
                    .map(|g| g.role.as_str())
                    .collect::<Vec<_>>()
                    != expected_roles
            {
                return Err(fail(
                    K::IdentityMismatch,
                    S::Admission,
                    "V3 control inventory/artifact/mode/graph roles differ before session creation",
                ));
            }
            // All pins, including V, are checked before reading any graph bytes.
            for graph in &manifest.graphs {
                policy.validate_pins(manifest_digest, &graph.legacy_pin_descriptor())?;
            }
            policy.validate_virtual_manifest_weight_bindings(&bytes)?;
        }
        let seeds = tickets::TicketSeeds::new()?;
        let epoch = manifest.epoch()?;
        let rules_semantics = parse_sha256(&manifest.rules_input_semantic_sha256)?;
        match (config.provider, runtime.bundle_digest()) {
            (Provider::Cpu, Some(_)) | (Provider::Cuda { .. }, None) => {
                return Err(fail(
                    K::IdentityMismatch,
                    S::Backend,
                    "selected-review provider and native bundle differ",
                ));
            }
            (Provider::Cuda { .. }, Some(_)) => runtime.verify_cuda_dependencies()?,
            _ => {}
        }
        let mut sessions = BTreeMap::new();
        let mut applied_session_settings = Vec::with_capacity(manifest.graphs.len());
        let mut placements = Vec::new();
        let mut identities = Vec::new();
        let mut total = 0_u64;
        for graph in &manifest.graphs {
            // Explicit V3 inventory covers every registered graph independently.
            let graph_audit = audit.as_ref();
            if let Some((policy, _)) = graph_audit {
                policy.validate_pins(manifest_digest, &graph.legacy_pin_descriptor())?;
            }
            let (data, digest) = owned_sources::read_graph_checked(
                sources.as_ref(),
                &graph.role,
                &graph.file,
                parse_sha256(&graph.sha256)?,
                MAX_GRAPH_BYTES,
            )?;
            total = total
                .checked_add(data.len() as u64)
                .filter(|n| *n <= MAX_MODEL_BYTES)
                .ok_or_else(|| {
                    fail(
                        K::ResourceExhausted,
                        S::Asset,
                        "V3 total serialized model budget exceeded",
                    )
                })?;
            identities.push(PalsGraphIdentity {
                role: graph.role.clone(),
                sha256: digest,
                serialized_bytes: data.len() as u64,
            });
            let role = VirtualReviewPhaseTrace::role(&graph.role);
            let serialized_bytes = data.len();
            phase_trace.record("session_load_entered", role, Some(serialized_bytes));
            let loaded = load_session(
                data,
                config,
                &graph.role,
                graph_audit.map(|(p, r)| (p, r.as_path())),
            );
            // The unchanged load_session owns the Vec and drops it on every
            // normal Ok/Err return (explicitly just after commit_from_memory).
            // This is an after-return observation, not the exact release time,
            // nor evidence that ORT native/prepacked/allocator storage is freed.
            phase_trace.record(
                "serialized_vec_released_after_loader_return",
                role,
                Some(serialized_bytes),
            );
            let (session, placement, applied) = match loaded {
                Ok(value) => {
                    phase_trace.record("session_load_returned_ok", role, None);
                    value
                }
                Err(error) => {
                    phase_trace.record("session_load_returned_error", role, None);
                    return Err(error);
                }
            };
            if let Err(error) = manifest.validate_session(&session, graph) {
                phase_trace.record("session_interface_returned_error", role, None);
                return Err(error);
            }
            // Actual Session settings and the existing initialization/interface
            // gates returned successfully. No kernel/placement/fence is inferred.
            phase_trace.record("session_settings_and_interface_validated", role, None);
            applied_session_settings.push(applied);
            if let Some(placement) = placement {
                placements.push(placement);
            }
            sessions.insert(graph.role.clone(), session);
        }
        let session_count = sessions.len();
        let initialization: Arc<[PalsGraphPlacement]> = placements.into();
        let backend_digest = execution_backend_digest(
            &runtime,
            config,
            audit.as_ref().map(|(p, _)| p.inventory),
            warm,
        );
        Ok(Self {
            public: sessions.remove("public"),
            shared_pc: sessions.remove(if warm { "shared_warm" } else { "shared_pc" }),
            validator: sessions.remove("validator"),
            applied_session_settings: applied_session_settings.into(),
            runtime: Some(runtime),
            config,
            model_config: manifest.config,
            epoch,
            manifest_digest,
            trained: manifest.trained,
            rules_semantics,
            backend_digest,
            memory: Some(MemoryBank::new(1, MAX_PUBLIC_BYTES).map_err(public_bank_error)?),
            active_memory: None,
            active: None,
            #[cfg(feature = "experimental-io-binding")]
            active_binding: None,
            stats: PalsBackendStats::default(),
            generation: 0,
            mapping: RefCell::default(),
            initialization: initialization.clone(),
            audit: audit.map(|(policy, profile_root)| VirtualControlAudit {
                policy,
                profile_root,
                profile_directory,
                initialization,
                proposer_seen: false,
                critic_seen: false,
                validator_seen: false,
                warm_seed_seen: false,
                witness: None,
                rejected: None,
            }),
            quarantine: None,
            residency: PalsSessionResidency {
                graphs: identities,
                native_sessions: session_count,
                layout: if warm { WARM_LAYOUT } else { SHARED_LAYOUT }.into(),
                reader_initializer_bank: None,
                role_reader_weights_shared: None,
                native_resident_parameter_bytes: None,
                vram_peak_bytes: None,
            },
            mode,
            warm,
            seeds,
            sources: Some(sources),
            phase_trace,
            #[cfg(feature = "native-validation-diagnostics")]
            validation_trace: None,
        })
    }
    /// Explicit diagnostic admission only; cannot change normal model/runtime policy.
    #[cfg(feature = "native-validation-diagnostics")]
    pub fn enable_gpu_lifecycle_validation(
        &mut self,
        trace: native_validation_diagnostics::GpuLifecycleTrace,
    ) -> Result<(), BackendError> {
        if self.validation_trace.is_some()
            || self.warm
            || self.validator_present()
            || self.config.cache_public_memory
            || !matches!(self.config.provider, Provider::Cuda { .. })
            || !self.virtual_cuda_placement_required()
            || self.stats.admitted_role_requests != 0
            || self.stats.completed_nn_inputs != 0
        {
            return Err(BackendError::new(crate::error::FailureKind::InvalidInput,
                crate::error::FailureStage::Admission,
                "unused V-free Fresh CUDA owner with public cache off required for explicit validation"));
        }
        trace.bind_actual_cuda_owner(
            self.model_epoch(),
            self.backend_digest(),
            self.manifest_digest(),
            self.runtime_binary_digest(),
        );
        self.validation_trace = Some(trace);
        Ok(())
    }
    #[cfg(feature = "native-validation-diagnostics")]
    pub fn gpu_lifecycle_validation_trace(
        &self,
    ) -> Option<native_validation_diagnostics::GpuLifecycleTrace> {
        self.validation_trace.clone()
    }
    pub fn model_epoch(&self) -> [u8; 32] {
        self.epoch
    }
    pub fn manifest_digest(&self) -> [u8; 32] {
        self.manifest_digest
    }
    pub fn backend_digest(&self) -> [u8; 32] {
        self.backend_digest
    }
    pub fn model_config(&self) -> &PalsModelConfig {
        &self.model_config
    }
    pub fn trained(&self) -> bool {
        self.trained
    }
    pub fn config(&self) -> PalsOnnxConfig {
        self.config
    }
    /// 성공한 원본 loader의 setter/EP 등록/commit 기록이며 OS/Run 관측과 다르다.
    pub fn applied_session_settings(&self) -> Arc<[PalsAppliedSessionSettings]> {
        Arc::clone(&self.applied_session_settings)
    }
    pub fn rules_input_semantic_digest(&self) -> [u8; 32] {
        self.rules_semantics
    }
    pub fn runtime_binary_digest(&self) -> [u8; 32] {
        self.runtime
            .as_ref()
            .expect("runtime owned")
            .binary_digest()
    }
    pub fn runtime_bundle_digest(&self) -> Option<[u8; 32]> {
        self.runtime
            .as_ref()
            .expect("runtime owned")
            .bundle_digest()
    }
    /// Initialization placement only. No Run/profile kernels, physical
    /// completion, numerical equivalence or normal-exit authority is issued.
    pub fn startup_placement(&self) -> Vec<PalsGraphPlacement> {
        self.initialization.to_vec()
    }
    pub fn runtime_build_info(&self) -> &str {
        self.runtime.as_ref().expect("runtime owned").build_info()
    }
    pub fn native_loading_profile(&self) -> Option<NativeLoadingProfile> {
        self.runtime
            .as_ref()
            .expect("runtime owned")
            .native_loading_profile()
    }
    pub fn native_loading_profile_digest(&self) -> Option<[u8; 32]> {
        self.runtime
            .as_ref()
            .expect("runtime owned")
            .native_loading_profile_digest()
    }
    pub fn execution_mode(&self) -> VirtualReviewExecutionMode {
        self.mode
    }
    pub fn validator_present(&self) -> bool {
        self.validator.is_some()
    }
    pub fn approximate_warm_capable(&self) -> bool {
        self.warm
    }
    /// Derived from the actual registered policy, never a caller assertion.
    pub fn virtual_cuda_placement_required(&self) -> bool {
        self.audit
            .as_ref()
            .is_some_and(|audit| audit.policy.virtual_review_layout().is_some())
    }
    pub fn game_generation(&self) -> u64 {
        self.generation
    }
    pub fn residency(&self) -> &PalsSessionResidency {
        &self.residency
    }
    pub fn quarantine_cause(&self) -> Option<&BackendError> {
        self.quarantine.as_ref()
    }
    fn idle(&self) -> Result<(), BackendError> {
        if let Some(cause) = &self.quarantine {
            return Err(cause.clone());
        }
        if self.active.is_some() || self.active_memory.is_some() {
            return Err(fail(
                K::BackendFailure,
                S::Backend,
                "selected-review owner must be physically idle",
            ));
        }
        #[cfg(feature = "experimental-io-binding")]
        if self.active_binding.is_some() {
            return Err(fail(
                K::BackendFailure,
                S::Backend,
                "V3 binding owner must be physically idle",
            ));
        }
        Ok(())
    }
    fn public_key(
        &self,
        input: &PreparedVirtualReviewTensors,
        identity: &ReviewExecutionIdentity,
    ) -> MemoryKey {
        let mut hash = Sha256::new();
        hash.update(b"rz-pals-virtual-review-public-pre-rel/1");
        hash.update(public_input_digest(input));
        hash.update(self.rules_semantics);
        hash.update(identity.rules_history);
        hash.update((identity.anchor_history_seals.len() as u64).to_le_bytes());
        for history in &identity.anchor_history_seals {
            hash.update(history);
        }
        hash.update(self.backend_digest);
        MemoryKey::PublicPage(PublicPageKey {
            kind: PublicPageKind::WholeInput,
            content: Digest(hash.finalize().into()),
            model: Digest(self.epoch),
            encoding: Digest(asset::sha256(model::ENCODING.as_bytes())),
            precision: PrecisionProfile::Fp32,
            frozen_epoch: 0,
            game_generation: self.generation,
        })
    }
    pub fn new_game(&mut self) -> Result<(), BackendError> {
        let next = self.generation.checked_add(1).ok_or_else(|| {
            fail(
                K::ResourceExhausted,
                S::Admission,
                "V3 game generation exhausted",
            )
        })?;
        self.reset_to(next)
    }
    pub fn reset_to(&mut self, next: u64) -> Result<(), BackendError> {
        self.idle()?;
        if next <= self.generation {
            return Err(fail(
                K::IdentityMismatch,
                S::Admission,
                "V3 reset generation must increase",
            ));
        }
        self.memory
            .as_mut()
            .expect("bank owned")
            .clear()
            .map_err(public_bank_error)?;
        self.generation = next;
        self.seeds.clear();
        self.stats.new_game_resets = self.stats.new_game_resets.saturating_add(1);
        Ok(())
    }
    /// Root stop/drain is distinct from game reset and process shutdown.
    /// This runs on the same exclusive physical owner after the preceding
    /// invocation's known completion. Sessions, profile collectors, runtime
    /// pins, cumulative counters and ticket high-water remain unchanged.
    pub fn quiesce_root(&mut self) -> Result<(), BackendError> {
        self.idle()?;
        self.seeds.clear();
        self.memory
            .as_mut()
            .expect("bank owned")
            .clear()
            .map_err(public_bank_error)?;
        Ok(())
    }
    pub fn snapshot_stats(&self) -> Result<PalsBackendStats, BackendError> {
        self.idle()?;
        let mut stats = self.stats.clone();
        stats.live_public_cache_entries = self.memory.as_ref().expect("bank owned").len() as u64;
        stats.validate()?;
        Ok(stats)
    }
    pub fn owner_snapshot(&self) -> VirtualReviewOwnerSnapshot {
        VirtualReviewOwnerSnapshot {
            game_generation: self.generation,
            active_prepared_host_bytes: self.active.as_ref().map_or(0, |a| a.host_bytes),
            bound_output_host_backing_bytes: {
                #[cfg(feature = "experimental-io-binding")]
                {
                    self.active_binding
                        .as_ref()
                        .map_or(0, |binding| binding.host_backing_bytes())
                }
                #[cfg(not(feature = "experimental-io-binding"))]
                {
                    0
                }
            },
            public_bank: self.memory.as_ref().expect("bank owned").snapshot(),
            native_sessions: self.residency.native_sessions,
            physical_completion_unknown: self.quarantine.is_some(),
            accepted_seed_host_backing_bytes: self.seeds.owned_bytes(),
            native_resident_parameter_bytes: None,
            vram_peak_bytes: None,
        }
    }
    pub fn verify_runtime(&self) -> Result<(), BackendError> {
        self.idle()?;
        if matches!(self.config.provider, Provider::Cuda { .. }) {
            self.mapping.borrow_mut().final_audit(|| {
                self.runtime
                    .as_ref()
                    .expect("runtime owned")
                    .verify_cuda_runtime_mappings()
            })?;
        }
        Ok(())
    }
    pub fn verify_cuda_placement(&mut self) -> Result<PalsCudaPlacementWitness, BackendError> {
        // This compatibility method never re-labels V3 evidence as the legacy
        // P/C-only witness. Explicit V3 callers use the separate typed command.
        self.idle()?;
        Err(fail(
            K::UnsupportedModel,
            S::Backend,
            "V3 placement requires VerifyVirtualCudaPlacement and a V3 inventory",
        ))
    }
    pub fn verify_virtual_cuda_placement(
        &mut self,
    ) -> Result<VirtualReviewCudaPlacementWitness, BackendError> {
        self.idle()?;
        self.verify_runtime()?;
        let audit = self.audit.as_mut().ok_or_else(|| {
            fail(
                K::BackendUnavailable,
                S::Backend,
                "V3 has no explicit selected-review CUDA control policy",
            )
        })?;
        if let Some(cause) = &audit.rejected {
            return Err(cause.clone());
        }
        if let Some(witness) = &audit.witness {
            return Ok(witness.clone());
        }
        // Missing role execution is a typed lack of evidence; keep profiles
        // open so a later actual role may supply evidence. Never warm up an NN
        // merely to manufacture an acceptance witness.
        if !audit.proposer_seen
            || !audit.critic_seen
            || (self.validator.is_some() && !audit.validator_seen)
            || (self.warm && !audit.warm_seed_seen)
        {
            return Err(fail(
                K::BackendUnavailable,
                S::Backend,
                "V3 placement lacks validated execution for every registered private role",
            ));
        }
        let checked = (|| {
            let finish = |session: &mut Session, role: &str| {
                if let Some(directory) = &audit.profile_directory {
                    audit
                        .policy
                        .finish_virtual_profile_owned(session, directory, role)
                } else {
                    audit
                        .policy
                        .finish_virtual_profile(session, &audit.profile_root, role)
                }
            };
            let public = finish(self.public.as_mut().expect("public"), "public")?;
            let private_role = if self.warm {
                "shared_warm"
            } else {
                "shared_pc"
            };
            let pc = finish(self.shared_pc.as_mut().expect("P/C"), private_role)?;
            let mut profiles =
                BTreeMap::from([("public".into(), public), (private_role.into(), pc)]);
            if let Some(validator) = &mut self.validator {
                profiles.insert("validator".into(), finish(validator, "validator")?);
            }
            let runtime = self.runtime.as_ref().expect("runtime owned");
            audit.policy.verify_virtual_profiles(
                &profiles,
                audit.initialization.to_vec(),
                runtime.binary_digest(),
                runtime.bundle_digest().ok_or_else(|| {
                    fail(K::IdentityMismatch, S::Backend, "V3 CUDA bundle missing")
                })?,
            )
        })();
        match checked {
            Ok(w) => {
                audit.witness = Some(w.clone());
                Ok(w)
            }
            Err(e) => {
                audit.rejected = Some(e.clone());
                Err(e)
            }
        }
    }
    pub fn validate_identity(
        &self,
        identity: &ReviewExecutionIdentity,
    ) -> Result<(), BackendError> {
        if identity.model_epoch != self.epoch
            || identity.backend != self.backend_digest
            || !identity.fp32
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Admission,
                "V3 execution identity differs from loaded model/backend/precision",
            ));
        }
        Ok(())
    }
    pub fn run(
        &mut self,
        input: &PalsVirtualReviewInput,
        identity: &ReviewExecutionIdentity,
    ) -> Result<PalsRawOutput, BackendError> {
        self.run_selected(input, identity, None)
    }
    pub fn run_for_ticket(
        &mut self,
        ticket: &RoleTicketEnvelope,
        invocation: &VirtualReviewInvocation,
    ) -> Result<VirtualReviewCompletion, BackendError> {
        self.idle()?;
        self.validate_identity(&invocation.identity)?;
        if ticket.authority.game.0 != self.generation {
            return Err(fail(
                K::IdentityMismatch,
                S::Admission,
                "V3 ticket game differs from this physical owner's registered generation",
            ));
        }
        self.seeds.admit(ticket, invocation)?;
        let seed = if self.warm {
            self.seeds.seed(ticket, invocation)?
        } else {
            None
        };
        let seed_digest = seed.as_ref().map(|seed| tickets::latent_digest(seed));
        let before = self.stats.completed_nn_inputs;
        let output = self.run_selected(&invocation.input, &invocation.identity, seed)?;
        let delta = self
            .stats
            .completed_nn_inputs
            .checked_sub(before)
            .ok_or_else(|| {
                fail(
                    K::BackendFailure,
                    S::Backend,
                    "V3 native completed input counter regressed",
                )
            })?;
        self.seeds
            .completed(ticket, invocation, output, delta, seed_digest)
    }
    /// Call only from the original ticket's manager acceptance callback. The
    /// completion token is issued by this owner after an actual native fence;
    /// changed output, different query, duplicate ack or newer root is rejected.
    pub fn accept_completed_ticket(
        &mut self,
        ticket: &RoleTicketEnvelope,
        invocation: &VirtualReviewInvocation,
        completion: &VirtualReviewCompletion,
    ) -> Result<(), BackendError> {
        self.idle()?;
        self.validate_identity(&invocation.identity)?;
        self.seeds.accept(ticket, invocation, completion, self.warm)
    }
    pub fn discard_completed_ticket(
        &mut self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(), BackendError> {
        self.idle()?;
        self.seeds.discard(ticket)
    }
    fn run_selected(
        &mut self,
        input: &PalsVirtualReviewInput,
        identity: &ReviewExecutionIdentity,
        seed: Option<Vec<f32>>,
    ) -> Result<PalsRawOutput, BackendError> {
        self.idle()?;
        self.mapping.borrow().allow_run()?;
        self.validate_identity(identity)?;
        if identity.anchor_history_seals.len() != input.anchors.len()
            || identity.anchor_history_seals.first() != Some(&identity.rules_history)
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Admission,
                "V3 exact histories must align with ordered anchors and current root",
            ));
        }
        if input.role == PalsRole::Validator && self.validator.is_none() {
            return Err(fail(
                K::UnsupportedModel,
                S::Admission,
                "deployment has no validator graph",
            ));
        }
        let prepared = input.prepare().map_err(model_input)?;
        let used_accepted_seed = self.warm && seed.is_some();
        let key = self.public_key(&prepared, identity);
        let warm = if self.warm {
            Some(match seed {
                Some(seed) => (seed, true),
                None => (vec![0.; 16 * 384], false),
            })
        } else {
            None
        };
        self.active = Some(Active::new(prepared, input.role, warm)?);
        self.stats.admitted_role_requests = self.stats.admitted_role_requests.saturating_add(1);
        let before = self.stats.completed_nn_inputs;
        let result = self.run_active(input, key);
        if self.quarantine.is_none() {
            #[cfg(feature = "experimental-io-binding")]
            {
                self.active_binding = None;
            }
            self.active_memory = None;
            self.active = None;
        }
        if self.quarantine.is_none() && matches!(self.config.provider, Provider::Cuda { .. }) {
            self.mapping
                .borrow_mut()
                .after_run(self.stats.completed_nn_inputs > before, || {
                    self.runtime
                        .as_ref()
                        .expect("runtime owned")
                        .verify_cuda_runtime_mappings()
                })?;
        }
        if result.is_ok() {
            if let Some(audit) = &mut self.audit {
                audit.proposer_seen |= input.role == PalsRole::Proposer;
                audit.critic_seen |= input.role == PalsRole::Critic;
                audit.validator_seen |= input.role == PalsRole::Validator;
                audit.warm_seed_seen |= used_accepted_seed;
            }
        }
        result
    }
    fn run_active(
        &mut self,
        input: &PalsVirtualReviewInput,
        key: MemoryKey,
    ) -> Result<PalsRawOutput, BackendError> {
        self.active_memory = if self.config.cache_public_memory {
            self.memory.as_mut().expect("bank").acquire(key)
        } else {
            None
        };
        if self.active_memory.is_some() {
            self.stats.public_cache_hits = self.stats.public_cache_hits.saturating_add(1);
        } else {
            // Disabled caching still uses a bounded immutable owner for the
            // physical invocation; clear only while no native pin exists.
            if !self.config.cache_public_memory {
                self.memory
                    .as_mut()
                    .expect("bank")
                    .clear()
                    .map_err(public_bank_error)?;
            }
            self.stats.public_cache_misses = self.stats.public_cache_misses.saturating_add(1);
            self.stats.public_nn_runs_attempted =
                self.stats.public_nn_runs_attempted.saturating_add(1);
            let a = self.active.as_ref().expect("input owned");
            #[cfg(feature = "native-validation-diagnostics")]
            let validation_run = self
                .validation_trace
                .as_ref()
                .and_then(|trace| trace.enter_ort(native_validation_diagnostics::RunGraph::Public));
            let values = ort::inputs!["anchor_board"=>&a.anchor_board,"anchor_metadata"=>&a.anchor_metadata,"anchor_mask"=>&a.anchor_mask,"records"=>&a.records,"record_anchor"=>&a.record_anchor,"record_line_tokens"=>&a.record_line_tokens,"record_line_mask"=>&a.record_line_mask,"record_mask"=>&a.record_mask];
            let first_phase_run = self.phase_trace.first_run_entered("public");
            let result = self.public.as_mut().expect("public").run(values);
            self.phase_trace
                .first_run_returned(first_phase_run, "public", result.is_ok());
            #[cfg(feature = "native-validation-diagnostics")]
            if let Some(span) = validation_run {
                span.returned(result.is_ok());
            }
            let outputs = match result {
                Ok(o) => o,
                Err(e) => {
                    let error = native(CauseCode::OrtRun, "V3 public Run failed", e);
                    if matches!(self.config.provider, Provider::Cuda { .. }) {
                        self.quarantine = Some(error.clone());
                    } else {
                        self.stats.public_nn_runs_failed_known =
                            self.stats.public_nn_runs_failed_known.saturating_add(1);
                    }
                    return Err(error);
                }
            };
            self.stats.public_nn_runs_completed =
                self.stats.public_nn_runs_completed.saturating_add(1);
            self.stats.completed_nn_inputs = self.stats.completed_nn_inputs.saturating_add(1);
            let tokens = 66 + input.records.len().max(1);
            let anchors = input.anchors.len();
            let keys = extract_float(&outputs, "memory_key", &[1, 2, tokens as i64, 64])?;
            let values = extract_float(&outputs, "memory_value", &[1, 2, tokens as i64, 64])?;
            let contexts = extract_float(&outputs, "anchor_context", &[1, anchors as i64, 384])?;
            let (shape, mask) = outputs["memory_mask"]
                .try_extract_tensor::<bool>()
                .map_err(|e| {
                    native(
                        CauseCode::PolicyExtract,
                        "V3 public mask extraction failed",
                        e,
                    )
                })?;
            if shape.as_ref() != [1, tokens as i64]
                || mask[..66].iter().any(|v| !*v)
                || mask[66..]
                    .iter()
                    .copied()
                    .ne((0..input.records.len().max(1)).map(|i| i < input.records.len()))
            {
                return Err(fail(
                    K::NumericalFailure,
                    S::Output,
                    "V3 public mask disagrees with selected records",
                ));
            }
            let mask = mask.to_vec();
            let bytes = (keys.capacity() + values.capacity() + contexts.capacity()) as u64 * 4
                + mask.capacity() as u64
                + std::mem::size_of::<PublicMemory>() as u64;
            let error = |e| {
                native(
                    CauseCode::TensorCreate,
                    "V3 public output ownership failed",
                    e,
                )
            };
            let memory = PublicMemory {
                key: Tensor::from_array(([1, 2, tokens, 64], keys)).map_err(error)?,
                value: Tensor::from_array(([1, 2, tokens, 64], values)).map_err(error)?,
                mask: Tensor::from_array(([1, tokens], mask)).map_err(error)?,
                anchors: Tensor::from_array(([1, anchors, 384], contexts)).map_err(error)?,
            };
            drop(outputs);
            self.active_memory = Some(
                self.memory
                    .as_mut()
                    .expect("bank")
                    .insert(key, memory, bytes)
                    .map_err(public_bank_error)?,
            );
            self.stats.validated_public_outputs =
                self.stats.validated_public_outputs.saturating_add(1);
        }
        #[cfg(feature = "experimental-io-binding")]
        if self.warm && matches!(self.config.provider, Provider::Cuda { .. }) {
            // CUDA Warm uses its separate I/O binding path, without this direct
            // Session.run trace. Missing phase rows imply neither no Run nor failure.
            return self.run_bound_warm(input);
        }
        let a = self.active.as_ref().expect("input owned");
        let m = self.active_memory.as_ref().expect("public pinned");
        let mut values = ort::inputs!["memory_key"=>&m.key,"memory_value"=>&m.value,"memory_mask"=>&m.mask,"candidates"=>&a.candidates,"candidate_mask"=>&a.candidate_mask,"divergence_features"=>&a.divergences,"divergence_mask"=>&a.divergence_mask,"query"=>&a.query,"anchor_context"=>&m.anchors,"query_anchor"=>&a.query_anchor,"query_line_tokens"=>&a.query_line_tokens,"query_line_mask"=>&a.query_line_mask,"record_relations"=>&a.record_relations];
        if input.role != PalsRole::Validator {
            values.extend(ort::inputs!["role_is_critic"=>&a.role]);
        }
        if let Some(warm) = &a.warm {
            values.extend(
                ort::inputs!["initial_latent"=>&warm.initial_latent,"warm_start"=>&warm.warm_start],
            );
        }
        self.stats.role_nn_runs_attempted = self.stats.role_nn_runs_attempted.saturating_add(1);
        #[cfg(feature = "native-validation-diagnostics")]
        let validation_run = self
            .validation_trace
            .as_ref()
            .and_then(|trace| trace.enter_ort(native_validation_diagnostics::RunGraph::Private));
        let phase_role = if input.role == PalsRole::Validator {
            "validator"
        } else if self.warm {
            "shared_warm"
        } else {
            "shared_pc"
        };
        let first_phase_run = self.phase_trace.first_run_entered(phase_role);
        let session = if input.role == PalsRole::Validator {
            self.validator.as_mut().expect("V admitted")
        } else {
            self.shared_pc.as_mut().expect("P/C")
        };
        let result = session.run(values);
        self.phase_trace
            .first_run_returned(first_phase_run, phase_role, result.is_ok());
        #[cfg(feature = "native-validation-diagnostics")]
        if let Some(span) = validation_run {
            span.returned(result.is_ok());
        }
        let outputs = match result {
            Ok(o) => o,
            Err(e) => {
                let error = native(CauseCode::OrtRun, "V3 private Run failed", e);
                if matches!(self.config.provider, Provider::Cuda { .. }) {
                    self.quarantine = Some(error.clone());
                } else {
                    self.stats.role_nn_runs_failed_known =
                        self.stats.role_nn_runs_failed_known.saturating_add(1);
                }
                return Err(error);
            }
        };
        self.stats.role_nn_runs_completed = self.stats.role_nn_runs_completed.saturating_add(1);
        self.stats.completed_nn_inputs = self.stats.completed_nn_inputs.saturating_add(1);
        let output = extract_role_output(&outputs, input)?;
        self.stats.validated_role_outputs = self.stats.validated_role_outputs.saturating_add(1);
        Ok(output)
    }
    #[cfg(feature = "experimental-io-binding")]
    fn run_bound_warm(
        &mut self,
        input: &PalsVirtualReviewInput,
    ) -> Result<PalsRawOutput, BackendError> {
        if input.role == PalsRole::Validator
            || self.active.as_ref().and_then(|a| a.warm.as_ref()).is_none()
        {
            return Err(fail(
                K::UnsupportedModel,
                S::Admission,
                "V3 Warm binding requires admitted P/C seed/mode owner",
            ));
        }
        // Construction binds only owned CPU output aliases. Install the owner
        // before the first input bind can enqueue a CUDA transfer.
        self.active_binding = Some(
            binding::Binding::new(
                self.shared_pc.as_ref().expect("P/C"),
                input.candidates.len().max(1),
                input.divergences.len().max(1),
                input.role == PalsRole::Critic,
            )
            .map_err(|e| {
                native(
                    CauseCode::TensorCreate,
                    "V3 fixed output binding allocation failed",
                    e,
                )
            })?,
        );
        self.stats.role_nn_runs_attempted = self.stats.role_nn_runs_attempted.saturating_add(1);
        let result = self
            .active_binding
            .as_mut()
            .expect("binding installed")
            .run(
                self.shared_pc.as_mut().expect("P/C"),
                self.active.as_ref().expect("input owned"),
                self.active_memory.as_ref().expect("public pinned"),
            );
        if let Err(e) = result {
            let error = native(CauseCode::OrtRun, "V3 Warm binding Run/fence failed", e);
            self.quarantine = Some(error.clone());
            return Err(error);
        }
        self.stats.role_nn_runs_completed = self.stats.role_nn_runs_completed.saturating_add(1);
        self.stats.completed_nn_inputs = self.stats.completed_nn_inputs.saturating_add(1);
        let output = self
            .active_binding
            .as_ref()
            .expect("binding retained after fence")
            .extract(input)?;
        self.stats.validated_role_outputs = self.stats.validated_role_outputs.saturating_add(1);
        Ok(output)
    }
    pub fn into_worker(
        mut self,
    ) -> Result<
        SingleWorker<VirtualReviewNativeCommand, Result<VirtualReviewNativeResult, BackendError>>,
        BackendError,
    > {
        SingleWorker::spawn_with_outcome(move |command| {
            let result = match command {
                VirtualReviewNativeCommand::Evaluate(invocation) => self
                    .run(&invocation.input, &invocation.identity)
                    .map(VirtualReviewNativeResult::Evaluation),
                VirtualReviewNativeCommand::EvaluateTicket { ticket, invocation } => {
                    let before = self.stats.clone();
                    let outcome = self.run_for_ticket(ticket, invocation);
                    let evidence = PalsNNInputEvidence::between(
                        &before,
                        &self.stats,
                        self.quarantine.is_none(),
                    );
                    let owner_accounting = VirtualReviewTicketOwnerAccounting::between(
                        &before,
                        &self.stats,
                        invocation.input.role,
                        self.quarantine.is_none(),
                    );
                    Ok(VirtualReviewNativeResult::TicketEvaluationWithEvidence {
                        outcome,
                        actual_completed_nn_inputs_delta: evidence.actual_completed_nn_inputs_delta,
                        complete: evidence.complete,
                        owner_accounting,
                    })
                }
                #[cfg(feature = "native-validation-diagnostics")]
                VirtualReviewNativeCommand::EvaluateTicketForValidation {
                    ticket,
                    execution,
                    invocation,
                } => {
                    let _validation_ticket = self
                        .validation_trace
                        .as_ref()
                        .map(|trace| trace.enter_ticket(*ticket, *execution));
                    let before = self.stats.clone();
                    let outcome = self.run_for_ticket(ticket, invocation);
                    let evidence = PalsNNInputEvidence::between(
                        &before,
                        &self.stats,
                        self.quarantine.is_none(),
                    );
                    let owner_accounting = VirtualReviewTicketOwnerAccounting::between(
                        &before,
                        &self.stats,
                        invocation.input.role,
                        self.quarantine.is_none(),
                    );
                    Ok(VirtualReviewNativeResult::TicketEvaluationWithEvidence {
                        outcome,
                        actual_completed_nn_inputs_delta: evidence.actual_completed_nn_inputs_delta,
                        complete: evidence.complete,
                        owner_accounting,
                    })
                }
                VirtualReviewNativeCommand::AcceptTicket {
                    ticket,
                    invocation,
                    completion,
                } => self
                    .accept_completed_ticket(ticket, invocation, completion)
                    .map(|()| VirtualReviewNativeResult::TicketAccepted),
                VirtualReviewNativeCommand::DiscardTicket(ticket) => self
                    .discard_completed_ticket(ticket)
                    .map(|()| VirtualReviewNativeResult::TicketDiscarded),
                VirtualReviewNativeCommand::QuiesceRoot => {
                    self.quiesce_root()
                        .map(|()| VirtualReviewNativeResult::RootQuiesced {
                            game_generation: self.generation,
                        })
                }
                VirtualReviewNativeCommand::ResetTo(generation) => {
                    self.reset_to(*generation)
                        .map(|()| VirtualReviewNativeResult::ResetTo {
                            game_generation: *generation,
                        })
                }
                VirtualReviewNativeCommand::NewGame => {
                    self.new_game().map(|()| VirtualReviewNativeResult::NewGame)
                }
                VirtualReviewNativeCommand::SnapshotStats => {
                    self.snapshot_stats().map(VirtualReviewNativeResult::Stats)
                }
                VirtualReviewNativeCommand::SnapshotOwner => self
                    .idle()
                    .map(|()| VirtualReviewNativeResult::Owner(self.owner_snapshot())),
                VirtualReviewNativeCommand::VerifyRuntime => self
                    .verify_runtime()
                    .map(|()| VirtualReviewNativeResult::RuntimeVerified),
                VirtualReviewNativeCommand::VerifyCudaPlacement => self
                    .verify_cuda_placement()
                    .map(|w| VirtualReviewNativeResult::CudaPlacementVerified(Box::new(w))),
                VirtualReviewNativeCommand::VerifyVirtualCudaPlacement => self
                    .verify_virtual_cuda_placement()
                    .map(|w| VirtualReviewNativeResult::VirtualCudaPlacementVerified(Box::new(w))),
            };
            match &self.quarantine {
                Some(cause) => PhysicalRun::Quarantined(cause.clone()),
                None => PhysicalRun::Complete(result),
            }
        })
    }
}
fn execution_backend_digest(
    runtime: &OrtRuntime,
    config: PalsOnnxConfig,
    control: Option<[u8; 32]>,
    warm: bool,
) -> [u8; 32] {
    virtual_review_backend_digest_from_metadata(
        runtime.binary_digest(),
        runtime.native_loading_profile_digest(),
        runtime.bundle_digest(),
        config,
        control,
        warm,
    )
}

/// Predict the unchanged backend identity from explicitly registered inputs.
/// This shared byte codec never loads ORT, checks a provider, creates a session
/// or grants execution authority. The actual backend passes its independently
/// observed runtime identities through the same codec above.
pub fn virtual_review_backend_digest_from_metadata(
    runtime_binary: [u8; 32],
    native_loading_profile: Option<[u8; 32]>,
    runtime_bundle: Option<[u8; 32]>,
    config: PalsOnnxConfig,
    control: Option<[u8; 32]>,
    warm: bool,
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"rz-pals-virtual-review-actual-backend-fp32/1");
    h.update(if warm {
        WARM_LAYOUT.as_bytes()
    } else {
        SHARED_LAYOUT.as_bytes()
    });
    h.update(runtime_binary);
    if let Some(profile) = native_loading_profile {
        h.update([1]);
        h.update(profile);
    } else {
        h.update([0]);
    }
    if let Some(inventory) = control {
        h.update([1]);
        h.update(inventory);
    } else {
        h.update([0]);
    }
    h.update([match config.cudnn_conv_search {
        PalsCuDnnConvSearch::LegacyImplicitExhaustive => 0,
        PalsCuDnnConvSearch::ExperimentalDefault => 1,
        PalsCuDnnConvSearch::ExperimentalDefaultDeviceWarm => 2,
    }]);
    match runtime_bundle {
        Some(d) => {
            h.update([1]);
            h.update(d);
        }
        None => h.update([0]),
    };
    match config.provider {
        Provider::Cpu => h.update([0]),
        Provider::Cuda {
            device_id,
            arena_bytes,
        } => {
            h.update([1]);
            h.update(device_id.to_le_bytes());
            h.update(arena_bytes.to_le_bytes());
        }
    };
    h.update((config.intra_threads as u64).to_le_bytes());
    h.update(b"inter1;spinningoff;tf32off;B1");
    h.finalize().into()
}
impl Drop for PalsVirtualReviewOnnxBackend {
    fn drop(&mut self) {
        if self.quarantine.is_some() {
            #[cfg(feature = "experimental-io-binding")]
            if let Some(binding) = self.active_binding.take() {
                std::mem::forget(binding);
            }
            for session in [
                self.public.take(),
                self.shared_pc.take(),
                self.validator.take(),
            ]
            .into_iter()
            .flatten()
            {
                std::mem::forget(session);
            }
            if let Some(a) = self.active.take() {
                std::mem::forget(a);
            }
            if let Some(m) = self.active_memory.take() {
                std::mem::forget(m);
            }
            if let Some(bank) = self.memory.take() {
                std::mem::forget(bank);
            }
            if let Some(runtime) = self.runtime.take() {
                std::mem::forget(runtime);
            }
            if let Some(sources) = self.sources.take() {
                std::mem::forget(sources);
            }
            if let Some(directory) = self
                .audit
                .as_mut()
                .and_then(|audit| audit.profile_directory.take())
            {
                std::mem::forget(directory);
            }
        } else {
            #[cfg(feature = "experimental-io-binding")]
            {
                self.active_binding.take();
            }
            self.active_memory.take();
            self.active.take();
            self.memory.take();
            self.validator.take();
            self.shared_pc.take();
            self.public.take();
            self.runtime.take();
            self.sources.take();
            if let Some(audit) = self.audit.as_mut() {
                audit.profile_directory.take();
            }
        }
    }
}

fn public_input_digest(value: &PreparedVirtualReviewTensors) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"rz-pals-review-public-typed/1");
    for n in [value.anchors, value.records_count, value.length_bucket] {
        h.update((n as u64).to_le_bytes());
    }
    for values in [
        &value.anchor_board,
        &value.record_anchor,
        &value.record_line_tokens,
    ] {
        for v in values {
            h.update(v.to_le_bytes());
        }
    }
    for values in [&value.anchor_metadata, &value.records] {
        for v in values {
            h.update(v.to_le_bytes());
        }
    }
    for values in [
        &value.anchor_mask,
        &value.record_mask,
        &value.record_line_mask,
    ] {
        for v in values {
            h.update([u8::from(*v)]);
        }
    }
    h.finalize().into()
}
fn extract_float(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
    expected: &[i64],
) -> Result<Vec<f32>, BackendError> {
    let (shape, value) = outputs[name]
        .try_extract_tensor::<f32>()
        .map_err(|e| native(CauseCode::PolicyExtract, "V3 output extraction failed", e))?;
    if shape.as_ref() != expected || value.iter().any(|v| !v.is_finite()) {
        return Err(fail(
            K::NumericalFailure,
            S::Output,
            "V3 output shape/finite check failed",
        ));
    }
    Ok(value.to_vec())
}
fn extract_role_output(
    outputs: &ort::session::SessionOutputs<'_>,
    input: &PalsVirtualReviewInput,
) -> Result<PalsRawOutput, BackendError> {
    let c = input.candidates.len().max(1);
    let d = input.divergences.len().max(1);
    let mut candidate_logits = extract_float(outputs, "candidate_logits", &[1, c as i64])?;
    candidate_logits.truncate(input.candidates.len());
    let wdl_logits = extract_float(outputs, "wdl_logits", &[1, 3])?
        .try_into()
        .map_err(|_| fail(K::NumericalFailure, S::Output, "V3 WDL shape"))?;
    let private_latent = extract_float(outputs, "private_latent", &[1, 16, 384])?;
    let (divergence_logits, task_logits) = if input.role == PalsRole::Validator {
        (
            None,
            Some(
                extract_float(outputs, "task_logits", &[1, 7])?
                    .try_into()
                    .map_err(|_| fail(K::NumericalFailure, S::Output, "V3 task shape"))?,
            ),
        )
    } else {
        let (shape, tag) = outputs["is_critic"]
            .try_extract_tensor::<bool>()
            .map_err(|e| native(CauseCode::PolicyExtract, "V3 role tag extraction", e))?;
        validate_role_tag(shape.as_ref(), tag, input.role == PalsRole::Critic)?;
        let mut divergence = extract_float(outputs, "divergence_logits", &[1, d as i64])?;
        divergence.truncate(input.divergences.len());
        (
            if input.role == PalsRole::Critic {
                Some(divergence)
            } else {
                None
            },
            None,
        )
    };
    Ok(PalsRawOutput {
        candidate_logits,
        wdl_logits,
        divergence_logits,
        task_logits,
        private_latent,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pals_model::PalsCandidateToken;
    use model::{
        ReviewAnchorInput, ReviewQueryInput, ReviewQueryPurpose, ReviewRecordInput,
        ReviewRecordKind, INPUT_SCHEMA,
    };
    #[test]
    fn expected_backend_codec_preserves_original_cpu_bytes_and_distinguishes_inputs() {
        let config = PalsOnnxConfig::cpu();
        let runtime = [7; 32];
        // Original byte framing, independent of the production shared helper.
        let mut original = Sha256::new();
        original.update(b"rz-pals-virtual-review-actual-backend-fp32/1");
        original.update(SHARED_LAYOUT.as_bytes());
        original.update(runtime);
        original.update([0, 0, 0, 0, 0]); // loading/control/cuDNN/bundle/CPU
        original.update((config.intra_threads as u64).to_le_bytes());
        original.update(b"inter1;spinningoff;tf32off;B1");
        let expected: [u8; 32] = original.finalize().into();
        assert_eq!(
            virtual_review_backend_digest_from_metadata(runtime, None, None, config, None, false),
            expected
        );
        for changed in [
            virtual_review_backend_digest_from_metadata([8; 32], None, None, config, None, false),
            virtual_review_backend_digest_from_metadata(
                runtime,
                Some([1; 32]),
                None,
                config,
                None,
                false,
            ),
            virtual_review_backend_digest_from_metadata(
                runtime,
                None,
                Some([1; 32]),
                config,
                None,
                false,
            ),
            virtual_review_backend_digest_from_metadata(
                runtime,
                None,
                None,
                config,
                Some([1; 32]),
                false,
            ),
            virtual_review_backend_digest_from_metadata(runtime, None, None, config, None, true),
        ] {
            assert_ne!(changed, expected);
        }
    }
    fn accounting_stats() -> PalsBackendStats {
        PalsBackendStats {
            admitted_role_requests: 0,
            public_cache_hits: 0,
            public_cache_misses: 0,
            public_nn_runs_attempted: 0,
            public_nn_runs_completed: 0,
            public_nn_runs_failed_known: 0,
            role_nn_runs_attempted: 0,
            role_nn_runs_completed: 0,
            role_nn_runs_failed_known: 0,
            completed_nn_inputs: 0,
            validated_public_outputs: 0,
            validated_role_outputs: 0,
            new_game_resets: 0,
            live_public_cache_entries: 0,
        }
    }
    #[test]
    fn ticket_accounting_zero_nn_does_not_invent_cache_hits_or_validator_calls() {
        let before = accounting_stats();
        let after = before.clone();
        let actual =
            VirtualReviewTicketOwnerAccounting::between(&before, &after, PalsRole::Validator, true)
                .unwrap();
        assert_eq!(actual.public_cache_hits(), 0);
        assert_eq!(actual.public_cache_misses(), 0);
        assert_eq!(actual.validator_nn_runs_attempted(), 0);
        assert_eq!(actual.validator_nn_runs_completed(), 0);
    }
    #[test]
    fn ticket_accounting_separates_public_cache_and_actual_validator_role_graph() {
        let before = accounting_stats();
        let mut hit = before.clone();
        hit.admitted_role_requests = 1;
        hit.public_cache_hits = 1;
        hit.role_nn_runs_attempted = 1;
        hit.role_nn_runs_completed = 1;
        hit.completed_nn_inputs = 1;
        let v =
            VirtualReviewTicketOwnerAccounting::between(&before, &hit, PalsRole::Validator, true)
                .unwrap();
        assert_eq!(v.public_cache_hits(), 1);
        assert_eq!(v.public_cache_misses(), 0);
        assert_eq!(v.public_nn_runs_attempted(), 0);
        assert_eq!(v.public_nn_runs_completed(), 0);
        assert_eq!(v.validator_nn_runs_attempted(), 1);
        assert_eq!(v.validator_nn_runs_completed(), 1);
        let pc =
            VirtualReviewTicketOwnerAccounting::between(&before, &hit, PalsRole::Proposer, true)
                .unwrap();
        assert_eq!(pc.role_nn_runs_completed(), 1);
        assert_eq!(pc.validator_nn_runs_completed(), 0);
        let mut miss = hit;
        miss.public_cache_hits = 0;
        miss.public_cache_misses = 1;
        miss.public_nn_runs_attempted = 1;
        miss.public_nn_runs_completed = 1;
        miss.completed_nn_inputs = 2;
        let v =
            VirtualReviewTicketOwnerAccounting::between(&before, &miss, PalsRole::Validator, true)
                .unwrap();
        assert_eq!(v.public_cache_misses(), 1);
        assert_eq!(v.public_nn_runs_attempted(), 1);
        assert_eq!(v.public_nn_runs_completed(), 1);
        assert_eq!(v.public_nn_runs_failed_known(), 0);
        assert_eq!(v.role_nn_runs_completed(), 1);
        assert_eq!(v.validator_nn_runs_completed(), 1);
    }
    #[test]
    fn ticket_accounting_known_role_failure_retains_attempt_without_completed_input() {
        let before = accounting_stats();
        let mut failed = before.clone();
        failed.admitted_role_requests = 1;
        failed.public_cache_hits = 1;
        failed.role_nn_runs_attempted = 1;
        failed.role_nn_runs_failed_known = 1;
        let actual = VirtualReviewTicketOwnerAccounting::between(
            &before,
            &failed,
            PalsRole::Validator,
            true,
        )
        .unwrap();
        assert_eq!(actual.validator_nn_runs_attempted(), 1);
        assert_eq!(actual.validator_nn_runs_failed_known(), 1);
        assert_eq!(actual.validator_nn_runs_completed(), 0);
        assert!(VirtualReviewTicketOwnerAccounting::between(
            &before,
            &failed,
            PalsRole::Validator,
            false,
        )
        .is_none());
    }
    #[test]
    fn ticket_accounting_public_only_failure_never_counts_a_validator_role_call() {
        let before = accounting_stats();
        let mut failed = before.clone();
        failed.admitted_role_requests = 1;
        failed.public_cache_misses = 1;
        failed.public_nn_runs_attempted = 1;
        failed.public_nn_runs_failed_known = 1;
        let actual = VirtualReviewTicketOwnerAccounting::between(
            &before,
            &failed,
            PalsRole::Validator,
            true,
        )
        .unwrap();
        assert_eq!(actual.public_cache_misses(), 1);
        assert_eq!(actual.public_nn_runs_attempted(), 1);
        assert_eq!(actual.public_nn_runs_failed_known(), 1);
        assert_eq!(actual.public_nn_runs_completed(), 0);
        assert_eq!(actual.role_nn_runs_attempted(), 0);
        assert_eq!(actual.validator_nn_runs_attempted(), 0);
        assert_eq!(actual.validator_nn_runs_completed(), 0);
        assert_eq!(actual.validator_nn_runs_failed_known(), 0);
        assert!(VirtualReviewTicketOwnerAccounting::between(
            &before,
            &failed,
            PalsRole::Validator,
            false,
        )
        .is_none());
    }
    #[test]
    fn ticket_accounting_rejects_decreasing_saturated_inconsistent_and_mixed_counters() {
        let before = accounting_stats();
        let mut changed = before.clone();
        changed.admitted_role_requests = 1;
        changed.role_nn_runs_completed = 1;
        assert!(VirtualReviewTicketOwnerAccounting::between(
            &before,
            &changed,
            PalsRole::Validator,
            true,
        )
        .is_none());
        changed = before.clone();
        changed.admitted_role_requests = u64::MAX;
        assert!(VirtualReviewTicketOwnerAccounting::between(
            &before,
            &changed,
            PalsRole::Validator,
            true,
        )
        .is_none());
        changed = before.clone();
        changed.admitted_role_requests = 1;
        assert!(VirtualReviewTicketOwnerAccounting::between(
            &changed,
            &before,
            PalsRole::Validator,
            true,
        )
        .is_none());
        changed.admitted_role_requests = 2;
        changed.public_cache_hits = 2;
        assert!(VirtualReviewTicketOwnerAccounting::between(
            &before,
            &changed,
            PalsRole::Validator,
            true,
        )
        .is_none());
    }
    fn input() -> PalsVirtualReviewInput {
        let movement = PalsCandidateToken {
            from: 12,
            to: 28,
            promotion: 0,
        };
        PalsVirtualReviewInput {
            schema: INPUT_SCHEMA.into(),
            role: PalsRole::Proposer,
            anchors: vec![ReviewAnchorInput {
                board: vec![0; 64],
                metadata: [0.; 16],
            }],
            records: vec![ReviewRecordInput {
                kind: ReviewRecordKind::Proposal,
                perspective: 1,
                completed_iteration: false,
                rules_terminal: false,
                line: vec![movement],
                anchor: 0,
                parent: None,
                supersedes: None,
            }],
            query: ReviewQueryInput {
                purpose: ReviewQueryPurpose::ProposePolicy,
                perspective: 1,
                prefix: vec![],
                proposal: vec![movement],
                counter: vec![],
                counter_present: false,
                anchors: [0; 3],
            },
            candidates: vec![movement],
            divergences: vec![],
            token_budget: 512,
        }
    }
    #[test]
    fn public_projection_does_not_cache_private_or_relation_mixing() {
        let source = input();
        let key = public_input_digest(&source.prepare().unwrap());
        let mut query = source.clone();
        query.query.prefix.push(PalsCandidateToken {
            from: 52,
            to: 36,
            promotion: 0,
        });
        query.role = PalsRole::Critic;
        query.divergences = vec![[1.; 8]];
        assert_eq!(key, public_input_digest(&query.prepare().unwrap()));
        query = source.clone();
        query.records[0].line.push(PalsCandidateToken {
            from: 52,
            to: 36,
            promotion: 0,
        });
        assert_ne!(key, public_input_digest(&query.prepare().unwrap()));
        query = source;
        query.anchors[0].metadata[0] = 1.;
        assert_ne!(key, public_input_digest(&query.prepare().unwrap()));
    }
    #[test]
    fn owner_budget_covers_largest_public_payload() {
        let actual = ((66 + 64) * 2 * 64 * 4 * 2 + (66 + 64) + 4 * 384 * 4) as u64
            + std::mem::size_of::<PublicMemory>() as u64;
        assert_eq!(MAX_PUBLIC_BYTES, actual);
    }
}
