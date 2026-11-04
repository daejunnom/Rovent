//! V3 P/C Deployment numeric capture through the original typed ticket service.
//! This finite harness accepts numbers, not production manager semantic work.
//! V5 Warm content admission remains unavailable. Numerical acceptance is issued
//! only by a separate reader using these exact accepted seed bits.
use rz_contracts::pals::{Role, SearchAuthority};
use rz_contracts::pals_adaptive::{ReviewViewSeal, RoleTicketEnvelope};
use rz_contracts::{
    Deadline, Digest, GameGeneration, MonotonicTick, OwnerId, ProcessEpoch, RootGeneration,
    StateIdentity, StateRevision,
};
use rz_eval::asset;
use rz_eval::onnx::{NativeLoadingProfile, OrtRuntime, Provider};
use rz_eval::pals_model::virtual_review as model;
use rz_eval::pals_model::{PalsRawOutput, PalsRole};
use rz_eval::pals_onnx::virtual_review::{
    PalsVirtualReviewOnnxBackend, VirtualReviewArtifactKind, VirtualReviewExecutionMode,
    validate_virtual_review_export_manifest_kind,
};
use rz_eval::pals_onnx::{
    PalsCuDnnConvSearch, PalsCudaControlPolicy, PalsOnnxConfig, VirtualReviewCudaPlacementWitness,
};
use rz_eval::runtime_pin::{CudaRuntimeBundleSpec, RuntimeBundleFileRole, RuntimeCache};
use rz_position::{BoardMove, Position};
use rz_runtime::Clock;
use rz_runtime::adaptive::CoordinatorPolicy;
use rz_runtime::contracts::{ContractClock, ContractSystemClock};
use rz_search::pals::coordinator::{
    PhysicalRoleCompletion, RoleFailure, RoleSubmission, RoleTicketRegistry,
};
use rz_search::pals::engine::{
    RoleLogicalContext, RoleQueryPurpose, role_divergence_sha256, role_line_sha256,
};
use rz_search::pals::review_context::{
    ReviewAnchorMode, ReviewContextIndex, ReviewQuerySequences, ReviewSelectionPolicy, ReviewView,
};
use rz_search::pals::store::{PalsStores, SituationId, StoreLimits};
use rz_uci::pals_native::ticketed::{
    NativeTicketQuestion, NativeVirtualReviewTicketBackend, TicketExecutionEvidence,
    TicketNativeEvent, TicketPreparation, TicketServiceLimits, VirtualReviewCollectionObserver,
    VirtualReviewPhysicalInvocationEvidence, VirtualReviewPreparedTicket,
    VirtualReviewRoleTicketService, VirtualReviewTicketOutput,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
#[cfg(target_os = "linux")]
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Registry = RoleTicketRegistry<ReviewView, Arc<VirtualReviewTicketOutput>>;
const DOMAIN: &str = "rz-pals-v3-actual-seed-numeric/1";
const SCOPE: &str = "typed_service_seed_acceptance_not_manager_semantic_consumption";
const LIMIT: usize = 4 * 1024 * 1024;
const TOTAL: usize = 32 * 1024 * 1024;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    plan: PathBuf,
    plan_sha256: String,
    export_manifest: PathBuf,
    export_manifest_sha256: String,
    checkpoint_sha256: String,
    runtime_library: PathBuf,
    runtime_library_sha256: String,
    runtime_cache: PathBuf,
    output_root: PathBuf,
    source_components: BTreeMap<String, String>,
    provider: String,
    cudnn_conv_search: String,
    cuda_bundle: Option<PathBuf>,
    cuda_bundle_sha256: Option<String>,
    cuda_inventory: Option<PathBuf>,
    cuda_inventory_sha256: Option<String>,
    gpu_device_id: i32,
    gpu_arena_bytes: usize,
    process_epoch: u64,
    deadline_seconds: u64,
    cleanup_seconds: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    domain: String,
    initial_fen: String,
    game_moves: Vec<String>,
    groups: Vec<Group>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Group {
    id: String,
    role: PalsRole,
    line_bucket: u16,
    prefix: Vec<String>,
    proposal: Vec<String>,
}
fn checked<T, E: std::fmt::Debug>(value: std::result::Result<T, E>) -> Result<T> {
    value.map_err(|error| format!("{error:?}").into())
}
fn hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn sha(bytes: &[u8]) -> String {
    hex(Sha256::digest(bytes).into())
}
fn valid_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn decode_json<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    // Value maps alone silently overwrite duplicate keys. This visitor validates
    // every object before the deny_unknown_fields structs are constructed.
    struct Unique;
    impl<'de> Deserialize<'de> for Unique {
        fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
            struct Visitor;
            impl<'de> serde::de::Visitor<'de> for Visitor {
                type Value = Unique;
                fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    f.write_str("JSON with unique object keys")
                }
                fn visit_map<A: serde::de::MapAccess<'de>>(
                    self,
                    mut a: A,
                ) -> std::result::Result<Unique, A::Error> {
                    let mut keys = BTreeSet::new();
                    while let Some(key) = a.next_key::<String>()? {
                        if !keys.insert(key) {
                            return Err(serde::de::Error::custom("duplicate key"));
                        }
                        a.next_value::<Unique>()?;
                    }
                    Ok(Unique)
                }
                fn visit_seq<A: serde::de::SeqAccess<'de>>(
                    self,
                    mut a: A,
                ) -> std::result::Result<Unique, A::Error> {
                    while a.next_element::<Unique>()?.is_some() {}
                    Ok(Unique)
                }
                fn visit_bool<E: serde::de::Error>(
                    self,
                    _: bool,
                ) -> std::result::Result<Unique, E> {
                    Ok(Unique)
                }
                fn visit_i64<E: serde::de::Error>(self, _: i64) -> std::result::Result<Unique, E> {
                    Ok(Unique)
                }
                fn visit_u64<E: serde::de::Error>(self, _: u64) -> std::result::Result<Unique, E> {
                    Ok(Unique)
                }
                fn visit_f64<E: serde::de::Error>(self, _: f64) -> std::result::Result<Unique, E> {
                    Ok(Unique)
                }
                fn visit_str<E: serde::de::Error>(self, _: &str) -> std::result::Result<Unique, E> {
                    Ok(Unique)
                }
                fn visit_unit<E: serde::de::Error>(self) -> std::result::Result<Unique, E> {
                    Ok(Unique)
                }
            }
            d.deserialize_any(Visitor)
        }
    }
    serde_json::from_slice::<Unique>(bytes)?;
    Ok(serde_json::from_slice(bytes)?)
}

struct PlacementObserver(Arc<Mutex<Option<Value>>>);
impl VirtualReviewCollectionObserver for PlacementObserver {
    fn submitted(
        &mut self,
        _: &RoleTicketEnvelope,
        _: &TicketPreparation<ReviewView, NativeTicketQuestion>,
        _: &VirtualReviewPreparedTicket,
    ) -> std::result::Result<(), RoleFailure> {
        Ok(())
    }
    fn completed(
        &mut self,
        _: &RoleTicketEnvelope,
        _: PhysicalRoleCompletion,
        _: Option<&rz_contracts::pals::RoleOutput>,
        _: Option<TicketExecutionEvidence>,
        _: Option<VirtualReviewPhysicalInvocationEvidence>,
    ) -> std::result::Result<(), RoleFailure> {
        Ok(())
    }
    fn retired(&mut self, _: &RoleTicketEnvelope, _: bool) -> std::result::Result<(), RoleFailure> {
        Ok(())
    }
    fn runtime_audited(
        &mut self,
        witness: &VirtualReviewCudaPlacementWitness,
    ) -> std::result::Result<(), RoleFailure> {
        // Observation failure does not alter the original physical finish result.
        // The independent capture reader separately requires this exact witness.
        if let Ok(bytes) = serde_json::to_vec(witness)
            && bytes.len() <= 64 * 1024
            && let Ok(value) = serde_json::from_slice(&bytes)
            && let Ok(mut slot) = self.0.lock()
        {
            *slot = Some(value);
        }
        Ok(())
    }
}

/// All receipts use a retained directory capability and create_new publication.
/// Path replacement is rejected again before/after every write and after join.
struct Output {
    path: PathBuf,
    original: File,
    dir: cap_std::fs::Dir,
    bytes: usize,
}
#[cfg(target_os = "linux")]
fn outside_git(path: &Path) -> Result<PathBuf> {
    if !path.is_absolute() {
        return Err("absolute paths required".into());
    }
    for ancestor in path.ancestors() {
        if fs::symlink_metadata(ancestor)?.file_type().is_symlink()
            || ancestor.join(".git").exists()
        {
            return Err("output must be outside Git without linked ancestors".into());
        }
    }
    Ok(path.canonicalize()?)
}
impl Output {
    fn create(path: &Path) -> Result<Self> {
        #[cfg(not(target_os = "linux"))]
        {
            let _ = path;
            Err("retained output-directory identity currently requires Linux".into())
        }
        #[cfg(target_os = "linux")]
        {
            let parent = outside_git(path.parent().ok_or("output parent absent")?)?;
            let root = parent.join(path.file_name().ok_or("output name absent")?);
            if root != path {
                return Err("canonical new output root required".into());
            }
            fs::create_dir(&root)?;
            let original = File::open(&root)?;
            let dir = cap_std::fs::Dir::from_std_file(original.try_clone()?);
            let owner = Self {
                path: root,
                original,
                dir,
                bytes: 0,
            };
            owner.verify()?;
            Ok(owner)
        }
    }
    fn verify(&self) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::fs::MetadataExt;
            let held = self.original.metadata()?;
            let named = fs::symlink_metadata(&self.path)?;
            if !held.is_dir()
                || !named.is_dir()
                || named.file_type().is_symlink()
                || (held.dev(), held.ino(), held.uid(), held.gid())
                    != (named.dev(), named.ino(), named.uid(), named.gid())
                || outside_git(&self.path)? != self.path
            {
                return Err("original output directory identity changed".into());
            }
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = &self.original;
            Err("retained directory identity unavailable".into())
        }
    }
    fn write(&mut self, name: &str, value: &impl Serialize) -> Result<Value> {
        self.verify()?;
        let bytes = serde_json::to_vec(value)?;
        if bytes.len() > LIMIT
            || self
                .bytes
                .checked_add(bytes.len())
                .is_none_or(|v| v > TOTAL)
        {
            return Err("bounded numeric output exhausted".into());
        }
        let mut options = cap_std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        let mut file = self.dir.open_with(name, &options)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        self.bytes += bytes.len();
        self.verify()?;
        Ok(json!({"file":name,"bytes":bytes.len(),"sha256":sha(&bytes)}))
    }
}
fn seed_bits(raw: &PalsRawOutput) -> Result<(Vec<u32>, String)> {
    if raw.private_latent.len() != 6144
        || raw.private_latent.iter().any(|v| !v.is_finite())
        || !raw.private_latent.iter().any(|v| *v != 0.0)
    {
        return Err("full finite nonzero 16x384 latent required".into());
    }
    let mut hash = Sha256::new();
    hash.update(b"rz-pals-review-accepted-final-latent/1");
    hash.update(6144u64.to_le_bytes());
    for v in &raw.private_latent {
        hash.update(v.to_le_bytes());
    }
    Ok((
        raw.private_latent.iter().map(|v| v.to_bits()).collect(),
        hex(hash.finalize().into()),
    ))
}
fn ticket_json(t: &RoleTicketEnvelope) -> Value {
    json!({"request_sequence":t.id.sequence,"process_epoch":t.id.epoch.0,
        "game":t.authority.game.0,"root":t.authority.root.0,"authority_implementation":hex(t.authority.implementation.0),
        "question":hex(t.question.0),"role":format!("{:?}",t.role),
        "deadline_ns":t.original_deadline.at.0,"deadline_epoch":t.original_deadline.clock.0.0,
        "view_sequence":t.view.view_sequence,"view_digest":hex(t.view.view_digest.0),
        "context_policy":hex(t.view.context_policy.0),"actual_tokens":t.view.actual_tokens,
        "selected_records":t.view.selected_records,"anchors":t.view.anchors,"line_bucket":t.view.length_bucket,
        "root_state":{"owner":t.view.root_state.owner.0,"revision":t.view.root_state.revision.0,"semantic":hex(t.view.root_state.semantic.0)},
        "representation":{"input":hex(t.representation.input.0),"history":hex(t.representation.history.0),
        "candidates":hex(t.representation.candidates.0),"mask_positions":hex(t.representation.mask_positions.0),
        "model":hex(t.representation.model.0),"encoding":hex(t.representation.encoding.0),
        "precision":"fp32","frozen_epoch":t.representation.frozen_epoch,"record_revision":t.representation.record_revision}})
}
fn next(
    service: &VirtualReviewRoleTicketService,
    until: Instant,
) -> Result<TicketNativeEvent<VirtualReviewTicketOutput>> {
    loop {
        if let Some(event) = service.poll() {
            return Ok(event);
        }
        if Instant::now() >= until {
            return Err("original finite window expired".into());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}
fn unexpected(event: &TicketNativeEvent<VirtualReviewTicketOutput>) -> String {
    match event {
        TicketNativeEvent::StartupFailed(failure) => {
            format!("original startup failure: {failure:?}")
        }
        TicketNativeEvent::PreparationFailed {
            preparation,
            failure,
        } => format!(
            "original preparation {} failure: {failure:?}",
            preparation.0
        ),
        TicketNativeEvent::Rejected { preparation, error } => format!(
            "original preparation {} rejection: {error:?}",
            preparation.0
        ),
        TicketNativeEvent::Retired {
            ticket,
            accepted,
            result,
            ..
        } => format!(
            "original ticket {} retirement accepted={accepted}: {result:?}",
            ticket.id.sequence
        ),
        TicketNativeEvent::Completed { completion, .. } => format!(
            "original ticket {} physical={:?} failure={:?}",
            completion.ticket.id.sequence,
            completion.physical,
            completion.result.as_ref().err()
        ),
        _ => "unexpected original ticket event ordering".to_owned(),
    }
}
fn moves(values: &[String]) -> Result<Vec<BoardMove>> {
    if values.len() > 256 {
        return Err("line exceeds 256 plies".into());
    }
    values.iter().map(|v| Ok(BoardMove::from_uci(v)?)).collect()
}
fn root_position(plan: &Plan) -> Result<Position> {
    let mut root = Position::from_fen(&plan.initial_fen)?;
    for movement in &plan.game_moves {
        root.make_move(BoardMove::from_uci(movement)?)?;
    }
    Ok(root)
}
#[cfg(feature = "native-validation-diagnostics")]
#[path = "virtual_review_warm_numeric_capture/gpu_inflight_validation.rs"]
pub mod gpu_inflight_validation;

fn validate_plan(plan: &Plan) -> Result<()> {
    validate_plan_domain(plan, DOMAIN)
}
fn validate_plan_domain(plan: &Plan, domain: &str) -> Result<()> {
    if plan.domain != domain || plan.game_moves.len() > 4096 || plan.groups.len() != 12 {
        return Err(
            "separate bounded V3 Warm plan with exactly twelve P/C bucket groups required".into(),
        );
    }
    let mut ids = BTreeSet::new();
    let mut coverage = BTreeSet::new();
    let root = root_position(plan)?;
    for g in &plan.groups {
        if g.id.is_empty()
            || g.id.len() > 64
            || !g.id.bytes().all(|v| v.is_ascii_alphanumeric() || v == b'-')
            || !ids.insert(&g.id)
            || g.role == PalsRole::Validator
            || !model::LENGTH_BUCKETS.contains(&(g.line_bucket as usize))
            || !coverage.insert((format!("{:?}", g.role), g.line_bucket))
        {
            return Err("duplicate/invalid id, role or L coverage".into());
        }
        let prefix = moves(&g.prefix)?;
        let proposal = moves(&g.proposal)?;
        for line in [&prefix, &proposal] {
            let mut replay = root.clone();
            for movement in line {
                replay.make_move(*movement)?;
            }
        }
        if model::length_bucket(prefix.len().max(proposal.len()))? != g.line_bucket as usize {
            return Err("registered physical L differs from Rules query lengths".into());
        }
    }
    Ok(())
}

fn load(config: &Config, profiles: &Path) -> Result<PalsVirtualReviewOnnxBackend> {
    load_kind(config, profiles, VirtualReviewArtifactKind::ApproxWarm)
}
fn load_kind(
    config: &Config,
    profiles: &Path,
    kind: VirtualReviewArtifactKind,
) -> Result<PalsVirtualReviewOnnxBackend> {
    load_kind_from_sources(config, profiles, kind, None, None)
}

fn load_kind_from_sources(
    config: &Config,
    profiles: &Path,
    kind: VirtualReviewArtifactKind,
    owned_sources: Option<Arc<dyn rz_eval::pals_onnx::virtual_review::VirtualReviewOwnedSources>>,
    source_metadata: Option<(Vec<u8>, Vec<u8>)>,
) -> Result<PalsVirtualReviewOnnxBackend> {
    let mut selected = PalsOnnxConfig::cpu();
    selected.intra_threads = 1;
    selected.cache_public_memory = kind == VirtualReviewArtifactKind::ApproxWarm;
    selected.cudnn_conv_search = match config.cudnn_conv_search.as_str() {
        "legacy_implicit_exhaustive" => PalsCuDnnConvSearch::LegacyImplicitExhaustive,
        "experimental_default" => PalsCuDnnConvSearch::ExperimentalDefault,
        _ => return Err("unsupported explicit convolution-search selection".into()),
    };
    let cache = RuntimeCache::open(&config.runtime_cache)?;
    let (pin, policy) = match config.provider.as_str() {
        "cpu" => {
            if config.cuda_bundle.is_some()
                || config.cuda_bundle_sha256.is_some()
                || config.cuda_inventory.is_some()
                || config.cuda_inventory_sha256.is_some()
                || selected.cudnn_conv_search != PalsCuDnnConvSearch::LegacyImplicitExhaustive
            {
                return Err("CPU cannot carry CUDA assets/options".into());
            }
            (
                cache.library(&config.runtime_library, &config.runtime_library_sha256)?,
                None,
            )
        }
        "cuda-control" | "cuda-control-shim" => {
            if std::env::var("NVIDIA_TF32_OVERRIDE").ok().as_deref() != Some("0")
                || config.gpu_device_id != 0
                || !(1..=(6usize << 30)).contains(&config.gpu_arena_bytes)
            {
                return Err("explicit CUDA0 FP32/TF32-off finite arena required".into());
            }
            let bundle = config.cuda_bundle.as_ref().ok_or("CUDA bundle absent")?;
            let bytes = if let Some((bundle_bytes, _)) = source_metadata.as_ref() {
                bundle_bytes.clone()
            } else {
                asset::read_bounded(bundle, 64 * 1024)?
            };
            if Some(sha(&bytes)) != config.cuda_bundle_sha256 {
                return Err("CUDA bundle byte pin differs".into());
            }
            let spec = CudaRuntimeBundleSpec::from_json(std::str::from_utf8(&bytes)?)?;
            let core = spec
                .files
                .iter()
                .find(|file| file.role == RuntimeBundleFileRole::Core)
                .ok_or("CUDA core absent")?;
            if core.sha256 != config.runtime_library_sha256
                || config
                    .runtime_library
                    .file_name()
                    .and_then(|name| name.to_str())
                    != Some(core.filename.as_str())
            {
                return Err("original core filename/byte pin differs".into());
            }
            selected.provider = Provider::Cuda {
                device_id: config.gpu_device_id,
                arena_bytes: config.gpu_arena_bytes,
            };
            let policy = if let Some((_, inventory_bytes)) = source_metadata.as_ref() {
                PalsCudaControlPolicy::from_virtual_review_inventory_bytes(
                    inventory_bytes,
                    config
                        .cuda_inventory_sha256
                        .as_deref()
                        .ok_or("Warm inventory pin absent")?,
                )?
            } else {
                PalsCudaControlPolicy::from_virtual_review_inventory(
                    config
                        .cuda_inventory
                        .as_ref()
                        .ok_or("Warm V3 inventory absent")?,
                    config
                        .cuda_inventory_sha256
                        .as_deref()
                        .ok_or("Warm inventory pin absent")?,
                )?
            };
            (
                cache.cuda_bundle(
                    config
                        .runtime_library
                        .parent()
                        .ok_or("runtime parent absent")?,
                    &spec,
                )?,
                Some(policy),
            )
        }
        _ => return Err("explicit cpu/cuda-control/cuda-control-shim provider required".into()),
    };
    let runtime = if config.provider == "cuda-control-shim" {
        OrtRuntime::load_with_cuda_loading_profile(&pin, NativeLoadingProfile::CuDnnShimLazyV1)?
    } else {
        OrtRuntime::load(&pin)?
    };
    if hex(runtime.binary_digest()) != config.runtime_library_sha256 {
        return Err("actual ORT pin differs".into());
    }
    selected.validate_execution_selection(runtime.native_loading_profile(), policy.is_some())?;
    let backend = if let Some(sources) = owned_sources {
        PalsVirtualReviewOnnxBackend::load_owned_sources(
            sources,
            &config.export_manifest_sha256,
            runtime,
            selected,
            VirtualReviewExecutionMode::Deployment,
            kind,
            policy.map(|policy| (policy, profiles.to_path_buf())),
        )?
    } else {
        match (policy, kind) {
            (Some(policy), VirtualReviewArtifactKind::Fresh) => {
                PalsVirtualReviewOnnxBackend::load_with_cuda_control_policy(
                    &config.export_manifest,
                    &config.export_manifest_sha256,
                    runtime,
                    selected,
                    policy,
                    profiles,
                )?
            }
            (None, VirtualReviewArtifactKind::Fresh) => PalsVirtualReviewOnnxBackend::load(
                &config.export_manifest,
                &config.export_manifest_sha256,
                runtime,
                selected,
            )?,
            (Some(policy), VirtualReviewArtifactKind::ApproxWarm) => {
                PalsVirtualReviewOnnxBackend::load_warm_with_cuda_control_policy(
                    &config.export_manifest,
                    &config.export_manifest_sha256,
                    runtime,
                    selected,
                    policy,
                    profiles,
                )?
            }
            (None, VirtualReviewArtifactKind::ApproxWarm) => {
                PalsVirtualReviewOnnxBackend::load_approx_warm(
                    &config.export_manifest,
                    &config.export_manifest_sha256,
                    runtime,
                    selected,
                )?
            }
        }
    };
    if backend.validator_present()
        || backend.approximate_warm_capable() != (kind == VirtualReviewArtifactKind::ApproxWarm)
        || hex(backend.model_epoch()) != config.checkpoint_sha256
    {
        return Err("actual V-free Warm owner/checkpoint differs".into());
    }
    Ok(backend)
}

#[allow(
    clippy::too_many_arguments,
    reason = "retain the original Rules/view/clock owners at the numerical-only preparation boundary"
)]
fn question(
    root: &Position,
    stores: &PalsStores,
    situation: SituationId,
    index: &mut ReviewContextIndex,
    group: &Group,
    authority: SearchAuthority,
    clock: &ContractSystemClock,
    until: Instant,
) -> Result<TicketPreparation<ReviewView, NativeTicketQuestion>> {
    let value = stores.situations.get(situation)?;
    let prefix = moves(&group.prefix)?;
    let proposal = moves(&group.proposal)?;
    let mut position = root.clone();
    for movement in &prefix {
        position.make_move(*movement)?;
    }
    let legal = position.legal_moves();
    if legal.is_empty() {
        return Err("nonterminal numeric question required".into());
    }
    let purpose = if group.role == PalsRole::Proposer {
        RoleQueryPurpose::ProposePolicy
    } else {
        RoleQueryPurpose::ReplyPolicy
    };
    let view = Arc::new(index.select(
        stores,
        value.state,
        ReviewQuerySequences {
            purpose,
            prefix: &prefix,
            proposal: &proposal,
            counterexample: None,
            legal: &legal,
            divergences: &[],
        },
        ReviewSelectionPolicy {
            mode: ReviewAnchorMode::AnchoredCompact,
            max_anchors: 4,
            ..ReviewSelectionPolicy::default()
        },
    )?);
    if view.length_bucket() != group.line_bucket {
        return Err("actual Rules view L differs from registered group".into());
    }
    let semantic = rz_position::contracts::rules_snapshot_semantic_digest(&root.snapshot());
    let seal = ReviewViewSeal {
        authority,
        view_sequence: view.sequence(),
        context_policy: Digest(view.policy_digest()),
        view_digest: Digest(view.seal_digest()),
        root_state: StateIdentity {
            owner: OwnerId(1),
            revision: StateRevision(root.snapshot().revision()),
            semantic,
        },
        actual_tokens: view.token_count(),
        selected_records: view.records().len() as u16,
        anchors: view.anchors().len() as u8,
        length_bucket: view.length_bucket(),
    };
    let context = RoleLogicalContext {
        game_generation: authority.game.0,
        search_generation: stores.generation(),
        situation,
        state: value.state,
        focus: value.focus,
        purpose,
        prefix: prefix.clone(),
        focus_sha256: role_line_sha256(b"focus", &[])?,
        prefix_sha256: role_line_sha256(b"prefix", &prefix)?,
        proposal_sha256: role_line_sha256(b"proposal", &proposal)?,
        refutation_sha256: None,
        divergence_sha256: role_divergence_sha256(&[])?,
        public_revision: 0,
        situation_revision: 0,
    };
    let duration = until
        .checked_duration_since(clock.origin())
        .ok_or("deadline precedes clock origin")?;
    let deadline = Deadline {
        clock: clock.domain(),
        at: MonotonicTick(u64::try_from(duration.as_nanos())?),
    };
    let query_digest = sha(&serde_json::to_vec(
        &json!({"domain":DOMAIN,"view":hex(view.seal_digest()),
        "purpose":format!("{purpose:?}"),"history":hex(rz_uci::pals_native::pals_history_digest(&position)?) }),
    )?);
    let question = Digest(Sha256::digest(query_digest.as_bytes()).into());
    Ok(TicketPreparation {
        question,
        seal,
        role: if group.role == PalsRole::Proposer {
            Role::Proposer
        } else {
            Role::Critic
        },
        deadline,
        view,
        query: NativeTicketQuestion {
            position,
            context,
            deadline: until,
            legacy_full_records: None,
        },
    })
}

fn run_cases(
    service: &mut VirtualReviewRoleTicketService,
    config: &Config,
    plan: &Plan,
    owner: &mut Output,
    clock: &ContractSystemClock,
    metadata: &Value,
    until: Instant,
) -> Result<Vec<Value>> {
    let root = root_position(plan)?;
    let mut stores = PalsStores::new(StoreLimits::default());
    let situation = stores.focus_actual_moves(root.snapshot())?;
    let mut index = ReviewContextIndex::default();
    let authority = SearchAuthority {
        epoch: clock.domain().0,
        game: GameGeneration(1),
        root: RootGeneration(1),
        implementation: Digest(
            Sha256::digest(include_bytes!("virtual_review_warm_numeric_capture.rs")).into(),
        ),
    };
    let mut registry = checked(Registry::new_with_capacity(
        authority,
        512,
        CoordinatorPolicy::Fixed,
        1,
        1,
    ))?;
    checked(
        registry.register_native_acceptance_verifier(
            service
                .native_acceptance_verifier()
                .ok_or("owner verifier absent")?,
        ),
    )?;
    let epoch = parse_digest(
        metadata["model_epoch"]
            .as_str()
            .ok_or("loaded epoch absent")?,
    )?;
    let backend = parse_digest(
        metadata["backend"]
            .as_str()
            .ok_or("loaded backend absent")?,
    )?;
    let mut artifacts = Vec::new();
    for group in &plan.groups {
        let mut accepted: Option<Value> = None;
        for step in 0..2 {
            let q = question(
                &root, &stores, situation, &mut index, group, authority, clock, until,
            )?;
            let view = Arc::clone(&q.view);
            let (input, identity) =
                rz_uci::pals_native::virtual_encoding::prepare_selected_virtual_review_input(
                    &q.query.position,
                    &q.query.context,
                    &view,
                    epoch,
                    backend,
                )?;
            let expected_input = input.fresh_key(&identity)?.0;
            let expected_history = rz_uci::pals_native::pals_history_digest(&q.query.position)?;
            let id = service
                .prepare(q)
                .map_err(|(e, _)| format!("prepare rejected: {e:?}"))?;
            let seal = match next(service, until)? {
                TicketNativeEvent::Prepared(seal) if seal.preparation == id => seal,
                event => return Err(unexpected(&event).into()),
            };
            if seal.representation.input.0 != expected_input
                || seal.representation.history.0 != expected_history
                || seal.representation.model.0 != identity.model_epoch
                || seal.view.view_digest.0 != view.seal_digest()
            {
                return Err("original prepared bytes/query/model/view differ".into());
            }
            let ticket = checked(registry.admit(
                RoleSubmission {
                    question: seal.question,
                    view: seal.view,
                    representation: seal.representation,
                    role: seal.role,
                    original_deadline: seal.original_deadline,
                },
                view,
                clock.domain(),
                clock.now(),
            ))?;
            if registry
                .take_dispatch()
                .is_none_or(|dispatch| dispatch.ticket != ticket)
            {
                return Err("original dispatch absent".into());
            }
            checked(service.execute(id, ticket))?;
            let mut started = false;
            let (completion, evidence, observation_failures) = loop {
                match next(service, until)? {
                    TicketNativeEvent::Started(t) if t == ticket => {
                        checked(registry.physical_started(&t))?;
                        started = true;
                    }
                    TicketNativeEvent::OwnerRunStarted { ticket: t, .. } if t == ticket => {}
                    TicketNativeEvent::Completed {
                        completion,
                        evidence,
                        observation_failures,
                        ..
                    } if completion.ticket == ticket && started => {
                        break (completion, evidence, observation_failures);
                    }
                    event => return Err(unexpected(&event).into()),
                }
            };
            if completion.physical != PhysicalRoleCompletion::Confirmed {
                if let Err(rejected) = registry.complete(completion) {
                    std::mem::forget(rejected);
                }
                std::mem::forget(registry);
                return Err("physical completion unknown; original owners retained".into());
            }
            if !observation_failures.is_empty() {
                let error = format!("native observation failure: {:?}", observation_failures);
                std::mem::forget(registry);
                return Err(error.into());
            }
            if ticket.role == Role::Proposer {
                registry
                    .complete_proposer_ranking(completion)
                    .map_err(|e| format!("completion rejected: {:?}", e.error))?;
            } else {
                registry
                    .complete(completion)
                    .map_err(|e| format!("completion rejected: {:?}", e.error))?;
            }
            let ready = registry
                .take_ready(clock.domain(), clock.now())
                .ok_or("original delivery absent")?;
            if !ready.logically_eligible {
                checked(service.discard(id, ticket))?;
                return Err("late/stale original delivery".into());
            }
            let output = checked(ready.result)?;
            if output.output.id != ticket.id
                || output.output.authority != ticket.authority
                || output.output.key != ticket.representation
            {
                return Err("original output identity differs".into());
            }
            let invocation = output.physical_invocation();
            let raw = output.raw_output();
            let (bits, digest) = seed_bits(raw)?;
            if invocation.physical_fresh != (step == 0)
                || invocation.accepted_seed_digest.map(|v| hex(v.0))
                    != accepted
                        .as_ref()
                        .map(|v| v["latent_sha256"].as_str().unwrap_or("").to_owned())
            {
                return Err(
                    "actual Warm mode/accepted seed differs from original retired predecessor"
                        .into(),
                );
            }
            let canonical_invocation = match invocation.accepted_seed_digest {
                Some(seed) => input.approximate_warm_key(&identity, seed.0)?.0,
                None => input.fresh_key(&identity)?.0,
            };
            if invocation.invocation_key.0 != canonical_invocation {
                return Err("physical invocation differs from canonical Fresh/Warm key".into());
            }
            let evidence = evidence.ok_or("actual NN evidence absent")?;
            let execution = evidence
                .execution
                .ok_or("original physical execution identity absent")?;
            if !evidence.measurement_complete
                || evidence.completed_nn_inputs.is_none()
                || evidence.physical_fresh != Some(step == 0)
                || execution.epoch != ticket.authority.epoch
                || execution.sequence == 0
            {
                return Err("actual NN evidence unavailable".into());
            }
            let accounting = evidence.accounting.ok_or("actual owner counters absent")?;
            if accounting.role_nn_runs_completed() != 1
                || accounting.role_nn_runs_attempted() != 1
                || accounting.role_nn_runs_failed_known() != 0
                || accounting.public_nn_runs_failed_known() != 0
                || accounting.validator_nn_runs_attempted() != 0
                || accounting.validator_nn_runs_completed() != 0
                || accounting.validator_nn_runs_failed_known() != 0
                || evidence.completed_nn_inputs
                    != Some(
                        accounting.public_nn_runs_completed() + accounting.role_nn_runs_completed(),
                    )
            {
                return Err("original P/C-only NN accounting differs".into());
            }
            let name = format!("{}-{step}", group.id);
            let raw_pin = owner.write(&format!("{name}.raw.json"), &json!({
                "domain":DOMAIN,"scope":SCOPE,"case":name,"line_bucket":group.line_bucket,
                "bindings":{"checkpoint_sha256":config.checkpoint_sha256,"export_manifest_sha256":config.export_manifest_sha256,
                    "runtime_library_sha256":config.runtime_library_sha256,"source_components":config.source_components},
                "ticket":ticket_json(&ticket),"input":input,"identity":{"model_epoch":hex(identity.model_epoch),
                    "backend":hex(identity.backend),"rules_history":hex(identity.rules_history),
                    "anchor_history_seals":identity.anchor_history_seals.iter().copied().map(hex).collect::<Vec<_>>(),"fp32":identity.fp32},
                "physical_completion_known":true,"physical_fresh":invocation.physical_fresh,"invocation_key":hex(invocation.invocation_key.0),
                "actual_seed":accepted,"accepted_seed_digest":invocation.accepted_seed_digest.map(|v|hex(v.0)),
                "raw_output":raw,"final_latent_bits":bits,"final_latent_sha256":digest,
                "completed_nn_inputs":evidence.completed_nn_inputs,"owner_accounting":{
                    "public_cache_hits":accounting.public_cache_hits(),"public_cache_misses":accounting.public_cache_misses(),
                    "public_nn_runs_attempted":accounting.public_nn_runs_attempted(),"public_nn_runs_completed":accounting.public_nn_runs_completed(),
                    "public_nn_runs_failed_known":accounting.public_nn_runs_failed_known(),
                    "role_nn_runs_attempted":accounting.role_nn_runs_attempted(),"role_nn_runs_completed":accounting.role_nn_runs_completed(),
                    "role_nn_runs_failed_known":accounting.role_nn_runs_failed_known(),
                    "validator_nn_runs_attempted":accounting.validator_nn_runs_attempted(),"validator_nn_runs_completed":accounting.validator_nn_runs_completed(),
                    "validator_nn_runs_failed_known":accounting.validator_nn_runs_failed_known()},
                "execution":{"epoch":execution.epoch.0,"sequence":execution.sequence},"numerical_acceptance":false,
            }))?;
            if ticket.role == Role::Proposer {
                let permit = checked(registry.issue_native_acceptance(
                    &ticket,
                    clock.domain(),
                    clock.now(),
                ))?;
                service
                    .accept_with_permit(id, ticket, permit)
                    .map_err(|(e, _)| format!("ack rejected: {e:?}"))?;
            } else {
                checked(service.accept(id, ticket))?;
            }
            match next(service, until)? {
                TicketNativeEvent::Retired {
                    ticket: t,
                    accepted: true,
                    result: Ok(()),
                    native_receipt,
                    rejected_permit: None,
                    observation_failures,
                } if t == ticket => {
                    if !observation_failures.is_empty() {
                        let error = format!(
                            "native retirement observation failure: {:?}",
                            observation_failures
                        );
                        // Preserve the exact receipt and registry until the
                        // process is gone; do not promote it to logical accept.
                        if let Some(receipt) = native_receipt {
                            std::mem::forget(receipt);
                        }
                        std::mem::forget(registry);
                        return Err(error.into());
                    }
                    if let Some(receipt) = native_receipt {
                        registry
                            .confirm_native_acceptance(receipt, clock.domain(), clock.now())
                            .map_err(|e| format!("original receipt rejected: {:?}", e.error))?;
                    } else if ticket.role == Role::Proposer {
                        return Err("original Proposer owner receipt absent".into());
                    }
                }
                event => return Err(unexpected(&event).into()),
            }
            checked(registry.accept(&ticket, clock.domain(), clock.now()))?;
            let retirement = owner.write(&format!("{name}.accepted.json"), &json!({"domain":DOMAIN,"scope":SCOPE,
                "ticket":ticket_json(&ticket),"raw":raw_pin,"native_retired_accepted":true,"registry_logical_acceptance":true,
                "production_manager_semantic_consumption":false}))?;
            accepted = Some(
                json!({"case":name,"ticket":ticket_json(&ticket),"retirement":retirement,
                "raw":raw_pin,"latent_bits":bits,"latent_sha256":digest}),
            );
            artifacts.push(json!({"raw":raw_pin,"retirement":retirement}));
            // A second ack must be refused by the original service, never replayed
            // to the backend callback. This adds no NN and no replacement token.
            checked(service.accept(id, ticket))?;
            match next(service, until)? {
                TicketNativeEvent::Rejected { preparation, error } if preparation == id => {
                    if error
                        != rz_uci::pals_native::ticketed::TicketServiceError::UnknownPreparation
                    {
                        return Err(
                            format!("unexpected original duplicate rejection: {error:?}").into(),
                        );
                    }
                    artifacts.push(owner.write(&format!("{name}.duplicate-rejected.json"), &json!({
                        "domain":DOMAIN,"ticket":ticket_json(&ticket),"error":format!("{error:?}"),"nn_execution_claim":false}))?);
                }
                event => return Err(unexpected(&event).into()),
            }
        }
    }
    // Cancel a prepared, never executed question through the original mailbox.
    let q = question(
        &root,
        &stores,
        situation,
        &mut index,
        &plan.groups[0],
        authority,
        clock,
        until,
    )?;
    let canceled = service
        .prepare(q)
        .map_err(|(e, _)| format!("cancel preparation: {e:?}"))?;
    match next(service, until)? {
        TicketNativeEvent::Prepared(seal) if seal.preparation == canceled => {}
        event => return Err(unexpected(&event).into()),
    }
    checked(service.cancel(canceled))?;
    match next(service, until)? {
        TicketNativeEvent::PreparationFailed {
            preparation,
            failure: RoleFailure::Canceled,
        } if preparation == canceled => {
            artifacts.push(owner.write("canceled-before-execute.json",&json!({"domain":DOMAIN,
                "preparation":canceled.0,"original_failure":"Canceled","native_execute_submitted":false}))?);
        }
        event => return Err(unexpected(&event).into()),
    }
    checked(service.begin_root_quiesce(until))?;
    match next(service, until)? {
        TicketNativeEvent::RootQuiesced(fence)
            if fence.authority() == authority && fence.permits_reuse() => {}
        event => return Err(unexpected(&event).into()),
    }
    Ok(artifacts)
}
fn parse_digest(s: &str) -> Result<[u8; 32]> {
    if !valid_sha(s) {
        return Err("lowercase SHA256 required".into());
    }
    let mut result = [0; 32];
    for (i, value) in result.iter_mut().enumerate() {
        *value = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)?;
    }
    Ok(result)
}
fn execute(config: Config) -> Result<()> {
    let started = Instant::now();
    if !(1..=3600).contains(&config.deadline_seconds)
        || !(1..=30).contains(&config.cleanup_seconds)
        || config.process_epoch == 0
    {
        return Err("finite original execution/cleanup and nonzero epoch required".into());
    }
    for path in [
        &config.plan,
        &config.export_manifest,
        &config.runtime_library,
        &config.runtime_cache,
        &config.output_root,
    ]
    .into_iter()
    .chain(config.cuda_bundle.as_ref())
    .chain(config.cuda_inventory.as_ref())
    {
        if !path.is_absolute() {
            return Err("absolute asset paths required".into());
        }
    }
    for pin in [
        &config.plan_sha256,
        &config.export_manifest_sha256,
        &config.checkpoint_sha256,
        &config.runtime_library_sha256,
    ] {
        parse_digest(pin)?;
    }
    if config.source_components != rz_experiments::compiled_source_components_v5()? {
        return Err("exact compiled source components differ".into());
    }
    let bytes = asset::read_bounded(&config.plan, LIMIT)?;
    if sha(&bytes) != config.plan_sha256 {
        return Err("actual plan SHA differs".into());
    }
    let plan: Plan = decode_json(&bytes)?;
    validate_plan(&plan)?;
    let manifest = asset::read_bounded(&config.export_manifest, 256 * 1024)?;
    let summary = validate_virtual_review_export_manifest_kind(
        &manifest,
        VirtualReviewExecutionMode::Deployment,
        VirtualReviewArtifactKind::ApproxWarm,
    )?;
    if hex(summary.manifest_sha256()) != config.export_manifest_sha256
        || hex(summary.checkpoint_sha256) != config.checkpoint_sha256
    {
        return Err("actual separate Warm export/checkpoint differs".into());
    }
    let until = started + Duration::from_secs(config.deadline_seconds);
    let mut owner = Output::create(&config.output_root)?;
    owner.write("registration.json", &json!({"domain":DOMAIN,"scope":SCOPE,"plan_sha256":config.plan_sha256,
        "capture_source_sha256":sha(include_bytes!("virtual_review_warm_numeric_capture.rs")),
        "source_components":config.source_components,"production_v5_warm_admission":"NOT_READY","numerical_acceptance":false}))?;
    let loaded = Arc::new(Mutex::new(None));
    let publish = Arc::clone(&loaded);
    let placement = Arc::new(Mutex::new(None));
    let observe_placement = Arc::clone(&placement);
    let factory_config = config.clone();
    let profiles = owner.path.join("profiles");
    let mut service = checked(VirtualReviewRoleTicketService::spawn_with_capacity(
        move || {
            let build = || -> Result<NativeVirtualReviewTicketBackend> {
                if Instant::now() >= until {
                    return Err("expired before native load".into());
                }
                let backend = load(&factory_config, &profiles)?;
                if Instant::now() >= until {
                    return Err("expired after native load".into());
                }
                let metadata = json!({"model_epoch":hex(backend.model_epoch()),"backend":hex(backend.backend_digest()),
                "validator_present":backend.validator_present(),"approximate_warm_capable":backend.approximate_warm_capable()});
                let clock = ContractSystemClock::with_origin(
                    ProcessEpoch(factory_config.process_epoch),
                    started,
                );
                let mut result = checked(NativeVirtualReviewTicketBackend::new(
                    backend,
                    clock,
                    Duration::from_secs(factory_config.cleanup_seconds),
                ))?;
                if factory_config.provider != "cpu" {
                    checked(result.enable_accepted_warm_profile_window())?;
                }
                checked(
                    result.set_collection_observer(Box::new(PlacementObserver(Arc::clone(
                        &observe_placement,
                    )))),
                )?;
                *publish.lock().map_err(|_| "loaded metadata poisoned")? = Some(metadata);
                Ok(result)
            };
            build().map_err(|error| RoleFailure::Backend(error.to_string()))
        },
        TicketServiceLimits::default(),
        1,
        1,
    ))?;
    let work = (|| -> Result<Vec<Value>> {
        let clock = match next(&service, until)? {
            TicketNativeEvent::Ready { clock } => clock,
            TicketNativeEvent::StartupFailed(failure) => {
                return Err(format!("original startup: {failure:?}").into());
            }
            _ => return Err("original Ready absent".into()),
        };
        let metadata = loaded
            .lock()
            .map_err(|_| "loaded metadata poisoned")?
            .clone()
            .ok_or("actual loaded identity absent")?;
        owner.write("loaded-owner.json", &metadata)?;
        run_cases(
            &mut service,
            &config,
            &plan,
            &mut owner,
            &clock,
            &metadata,
            until,
        )
    })();
    let cleanup = Instant::now() + Duration::from_secs(config.cleanup_seconds);
    service.begin_shutdown(cleanup);
    let mut physical_shutdown = false;
    let mut joined = false;
    let mut join_error = None;
    while Instant::now() < cleanup {
        while let Some(event) = service.poll() {
            if let TicketNativeEvent::Finished {
                physical_shutdown_confirmed,
            } = event
            {
                physical_shutdown = physical_shutdown_confirmed;
            }
        }
        match service.try_join() {
            Ok(true) => {
                joined = true;
                break;
            }
            Ok(false) => {}
            Err(error) => {
                join_error = Some(format!("{error:?}"));
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    // Finished can be published between the poll and successful thread join.
    while let Some(event) = service.poll() {
        if let TicketNativeEvent::Finished {
            physical_shutdown_confirmed,
        } = event
        {
            physical_shutdown = physical_shutdown_confirmed;
        }
    }
    let placement = placement.lock().ok().and_then(|value| value.clone());
    let success = work.is_ok()
        && physical_shutdown
        && joined
        && !service.is_quarantined()
        && (config.provider == "cpu" || placement.is_some());
    owner.write("capture-receipt.json", &json!({"domain":DOMAIN,"scope":SCOPE,
        "status":if success {"captured_actual_seed_numbers"} else {"capture_failed_preserve_originals"},
        "error":work.as_ref().err().map(ToString::to_string),"artifacts":work.as_ref().ok(),
        "provider":config.provider,"plan_sha256":config.plan_sha256,"actual_virtual_cuda_placement":placement,
        "physical_shutdown_confirmed":physical_shutdown,"native_owner_joined":joined,"quarantined":service.is_quarantined(),
        "join_error":join_error,
        "numerical_acceptance":false,"production_manager_semantic_consumption":false,
        "production_v5_warm_admission":"NOT_READY","performance_evidence_eligible":false,
        "query_coverage":"P_ProposePolicy_C_ReplyPolicy_one_Rules_anchor_empty_records_no_divergence_targets",
        "process_exit_evidence":"requires_external_process_owner_exit_observation"}))?;
    if !success {
        return Err("capture/physical closure failed; originals preserved".into());
    }
    println!(
        "RZ_V3_WARM_CAPTURE={{\"status\":\"captured\",\"numerical_acceptance\":false,\"production_v5_warm_admission\":\"NOT_READY\"}}"
    );
    Ok(())
}
fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args == ["--source-pins"] {
        println!(
            "{}",
            json!({"source_components":rz_experiments::compiled_source_components_v5()?,
            "capture_source_sha256":sha(include_bytes!("virtual_review_warm_numeric_capture.rs"))})
        );
        return Ok(());
    }
    if args.len() != 1 || !Path::new(&args[0]).is_absolute() {
        return Err(
            "usage: virtual_review_warm_numeric_capture --source-pins | ABSOLUTE_CONFIG.json"
                .into(),
        );
    }
    execute(decode_json(&asset::read_bounded(
        Path::new(&args[0]),
        64 * 1024,
    )?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_keys_match_independent_byte_framing_and_separate_fresh_from_warm() {
        let movement = rz_eval::pals_model::PalsCandidateToken {
            from: 12,
            to: 28,
            promotion: 0,
        };
        let mut metadata = [0.0; 16];
        metadata[0] = -0.0;
        let input = model::PalsVirtualReviewInput {
            schema: model::INPUT_SCHEMA.to_owned(),
            role: PalsRole::Proposer,
            anchors: vec![model::ReviewAnchorInput {
                board: vec![0; 64],
                metadata,
            }],
            records: vec![],
            query: model::ReviewQueryInput {
                purpose: model::ReviewQueryPurpose::ProposePolicy,
                perspective: -1,
                prefix: vec![movement],
                proposal: vec![],
                counter: vec![],
                counter_present: false,
                anchors: [0; 3],
            },
            candidates: vec![movement],
            divergences: vec![],
            token_budget: 512,
        };
        let identity = model::ReviewExecutionIdentity {
            model_epoch: [1; 32],
            rules_history: [3; 32],
            backend: [2; 32],
            anchor_history_seals: vec![[3; 32]],
            fp32: true,
        };
        let seed = parse_digest("c6c71f043f19cd8d880510cb2416396e8e59a83070608e1a9e5a59ffaba5c159")
            .unwrap();
        let fresh = input.fresh_key(&identity).unwrap().0;
        let warm = input.approximate_warm_key(&identity, seed).unwrap().0;
        // Independent Python stdlib vector includes f32 -0.0 and i8 -1 framing.
        assert_eq!(
            hex(fresh),
            "b038382273247dd0eec43bb7b86d5a57335bdbedac6ee7c696b3b0a2bb9cf278"
        );
        assert_eq!(
            hex(warm),
            "f8a64059ec1d4391366b17d90c28e2dcdc1a5f5146ec9c5bfe26c08add7782b6"
        );
        assert_ne!(fresh, warm);
        assert_ne!(
            warm,
            input.approximate_warm_key(&identity, [0; 32]).unwrap().0
        );
    }

    #[test]
    fn original_rules_views_cover_both_roles_and_six_buckets_without_a_backend() {
        let cycle = ["g1f3", "g8f6", "f3g1", "f6g8"];
        let mut groups = Vec::new();
        for role in [PalsRole::Proposer, PalsRole::Critic] {
            for length in model::LENGTH_BUCKETS {
                groups.push(Group {
                    id: format!("{role:?}-{length}"),
                    role,
                    line_bucket: length as u16,
                    prefix: Vec::new(),
                    proposal: (0..length).map(|i| cycle[i % 4].to_owned()).collect(),
                });
            }
        }
        let plan = Plan {
            domain: DOMAIN.to_owned(),
            initial_fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_owned(),
            game_moves: Vec::new(),
            groups,
        };
        validate_plan(&plan).unwrap();
        let root = root_position(&plan).unwrap();
        let mut stores = PalsStores::new(StoreLimits::default());
        let situation = stores.focus_actual_moves(root.snapshot()).unwrap();
        let mut index = ReviewContextIndex::default();
        let clock = ContractSystemClock::new(ProcessEpoch(7));
        let authority = SearchAuthority {
            epoch: ProcessEpoch(7),
            game: GameGeneration(1),
            root: RootGeneration(1),
            implementation: Digest([4; 32]),
        };
        for group in &plan.groups {
            let q = question(
                &root,
                &stores,
                situation,
                &mut index,
                group,
                authority,
                &clock,
                Instant::now() + Duration::from_secs(30),
            )
            .unwrap();
            let (input, _) =
                rz_uci::pals_native::virtual_encoding::prepare_selected_virtual_review_input(
                    &q.query.position,
                    &q.query.context,
                    &q.view,
                    [1; 32],
                    [2; 32],
                )
                .unwrap();
            assert_eq!(input.role, group.role);
            assert_eq!(
                input.validate().unwrap().length_bucket,
                group.line_bucket as usize
            );
        }
        let mut invalid = plan;
        invalid.groups[0].role = PalsRole::Validator;
        assert!(validate_plan(&invalid).is_err());
    }

    #[test]
    fn duplicate_nested_json_never_overwrites_the_registered_value() {
        assert!(decode_json::<Value>(br#"{"x":{"seed":1,"seed":2}}"#).is_err());
        assert!(decode_json::<Config>(br#"{"plan":"unregistered"}"#).is_err());
    }

    #[test]
    fn full_seed_observation_preserves_bits_and_rejects_nonfinite_or_partial_latent() {
        let mut raw = PalsRawOutput {
            candidate_logits: vec![0.0],
            wdl_logits: [0.0; 3],
            divergence_logits: None,
            task_logits: None,
            private_latent: vec![1.0; 6144],
        };
        raw.private_latent[0] = -0.0;
        let (bits, digest) = seed_bits(&raw).unwrap();
        assert_eq!(bits[0], 0x80000000);
        // Independently framed using Python stdlib SHA256 + struct.pack.
        assert_eq!(
            digest,
            "c6c71f043f19cd8d880510cb2416396e8e59a83070608e1a9e5a59ffaba5c159"
        );
        raw.private_latent[0] = f32::NAN;
        assert!(seed_bits(&raw).is_err());
        raw.private_latent.pop();
        assert!(seed_bits(&raw).is_err());
    }
}
