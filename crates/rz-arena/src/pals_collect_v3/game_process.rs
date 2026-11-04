//! A fresh, source-pinned whole-game child and its original supervisor owner.
//! Session epochs remain native session scopes. Only a verified new OS child,
//! one actual standalone game, raw readback and original child/group join issue
//! this module's private process-restart receipt. No JSON flag grants authority.

use super::*;
#[cfg(target_os = "linux")]
use crate::process::{OriginalProcessInput, OriginalProcessWindow};
use crate::process::{OriginalProcessOutput, OwnedArtifactTreeWatch, ProcessLimits};
use rz_eval::error::{BackendError, FailureKind, FailureStage};
#[cfg(target_os = "linux")]
use rz_eval::pals_model::virtual_review::{PalsVirtualReviewInput, ReviewExecutionIdentity};
use rz_eval::pals_onnx::virtual_review::PalsVirtualReviewOnnxBackend;
#[cfg(target_os = "linux")]
use rz_eval::pals_onnx::virtual_review::{
    VirtualReviewExecutionMode, validate_virtual_review_export_manifest,
};
#[cfg(target_os = "linux")]
use rz_experiments::PalsStartupAssetPurposeV5;
use rz_experiments::{ArtifactRef, PalsAdaptiveStartupConfigV5};
use rz_search::time::TimeBudgetConfig;
use rz_uci::pals_adaptive_startup::{PalsStartupOptionsV5, PinnedPalsStartupV5};
use rz_uci::pals_native::startup_virtual::NativeVirtualReviewIntakeV5;
#[cfg(target_os = "linux")]
use sha2::{Digest as _, Sha256};
#[cfg(target_os = "linux")]
use std::ffi::OsString;
use std::fs::File;
#[cfg(target_os = "linux")]
use std::path::Component;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const GENERATION_GAME_CHILD_ARGUMENT_V3: &str = "--generation-single-game-child-v3";
pub const GENERATION_GAME_CHILD_OUTPUT_V3: &str = "generation-game-output.json";
pub const GENERATION_GAME_REQUEST_DOMAIN_V3: &str = "rz-pals-whole-game-child-input/1";
const REQUEST_DOMAIN: &str = GENERATION_GAME_REQUEST_DOMAIN_V3;
#[cfg(target_os = "linux")]
const OUTPUT_DOMAIN: &str = "rz-pals-whole-game-child-output/1";
#[cfg(target_os = "linux")]
const RECEIPT_DOMAIN: &str = "rz-pals-whole-game-parent-observation/1";
#[cfg(target_os = "linux")]
const MAX_CONTROL_BYTES: u64 = 2 * 1024 * 1024;
#[cfg(target_os = "linux")]
const MAX_BINARY_BYTES: u64 = 512 * 1024 * 1024;
#[cfg(target_os = "linux")]
pub(super) const MAX_OUTPUT_BYTES: u64 = 4 * 1024 * 1024;
#[cfg(target_os = "linux")]
const MAX_GAME_WINDOW_NS: u64 = 3_600_000_000_000;
#[cfg(target_os = "linux")]
const MAX_GAME_CLEANUP_NS: u64 = 30_000_000_000;
static CHILD_GAME_BEGUN_V3: AtomicBool = AtomicBool::new(false);
static NEXT_PROCESS_LAUNCH_V3: AtomicU64 = AtomicU64::new(1);

#[cfg(target_os = "linux")]
fn monotonic_ns() -> Result<u64, ArenaError> {
    let now = nix::time::clock_gettime(nix::time::ClockId::CLOCK_MONOTONIC)
        .map_err(|_| invalid("whole-game original monotonic clock unavailable"))?;
    let seconds =
        u64::try_from(now.tv_sec()).map_err(|_| invalid("whole-game monotonic seconds"))?;
    let nanos =
        u64::try_from(now.tv_nsec()).map_err(|_| invalid("whole-game monotonic nanoseconds"))?;
    seconds
        .checked_mul(1_000_000_000)
        .and_then(|s| s.checked_add(nanos))
        .ok_or_else(|| invalid("whole-game monotonic clock overflow"))
}

/// Thin child entry converts the parent's same-host Linux monotonic absolute
/// ticks to conservative local Instants. Capture the Instant before sampling
/// the clock so this never renews either original parent ceiling.
pub fn generation_game_child_window_v3(
    execution_monotonic_ns: u64,
    whole_monotonic_ns: u64,
) -> Result<(Instant, Instant), ArenaError> {
    #[cfg(target_os = "linux")]
    {
        let entry = Instant::now();
        let now = monotonic_ns()?;
        let execution = execution_monotonic_ns
            .checked_sub(now)
            .ok_or_else(|| invalid("whole-game original child execution ceiling expired"))?;
        let whole = whole_monotonic_ns
            .checked_sub(now)
            .ok_or_else(|| invalid("whole-game original child closure ceiling expired"))?;
        check(
            execution > 0
                && execution <= MAX_GAME_WINDOW_NS
                && execution < whole
                && whole - execution <= MAX_GAME_CLEANUP_NS,
            "whole-game original child window bounds",
        )?;
        let deadline = |nanos| {
            entry
                .checked_add(Duration::from_nanos(nanos))
                .ok_or_else(|| invalid("whole-game original child Instant overflow"))
        };
        Ok((deadline(execution)?, deadline(whole)?))
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (execution_monotonic_ns, whole_monotonic_ns);
        Err(invalid("whole-game original process window requires Linux"))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationGameStartupControlV3 {
    pub config_file: String,
    pub lock_file: String,
}

/// Exact clock configuration, without silently selecting native defaults.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationGameClockConfigV3 {
    pub default_moves_to_go: u32,
    pub increment_share_basis_points: u16,
    pub hard_budget_multiplier: u32,
    pub output_margin_ns: u64,
    pub drain_margin_ns: u64,
}

/// This whole-game phase binds the producer header to checkpoint file bytes and
/// the actual runtime library bytes. Other collection phases retain their own
/// producer model/runtime domains; this is not a named tensor-content proof.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationGameProducerPhaseV3 {
    NativeSelfplayCheckpointAndRuntimeBinary,
}
impl GenerationGameClockConfigV3 {
    pub fn from_actual(config: TimeBudgetConfig) -> Result<Self, ArenaError> {
        config.validate().map_err(|e| invalid(e.to_string()))?;
        Ok(Self {
            default_moves_to_go: config.default_moves_to_go,
            increment_share_basis_points: config.increment_share_basis_points,
            hard_budget_multiplier: config.hard_budget_multiplier,
            output_margin_ns: u64::try_from(config.output_margin.as_nanos())
                .map_err(|_| invalid("whole-game output margin overflow"))?,
            drain_margin_ns: u64::try_from(config.drain_margin.as_nanos())
                .map_err(|_| invalid("whole-game drain margin overflow"))?,
        })
    }
    fn actual(&self) -> Result<TimeBudgetConfig, ArenaError> {
        let config = TimeBudgetConfig {
            default_moves_to_go: self.default_moves_to_go,
            increment_share_basis_points: self.increment_share_basis_points,
            hard_budget_multiplier: self.hard_budget_multiplier,
            output_margin: Duration::from_nanos(self.output_margin_ns),
            drain_margin: Duration::from_nanos(self.drain_margin_ns),
        };
        config.validate().map_err(|e| invalid(e.to_string()))?;
        Ok(config)
    }
}

/// Execution declaration only. Deserialization creates no successful child,
/// native completion or generation authority. The parent retains these bytes.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationWholeGameRequestV3 {
    pub schema: String,
    pub plan: SelfplayCollectionPlanV3,
    pub game_ordinal: u16,
    pub producer: GenerationProducerAuthorityV3,
    pub producer_phase: GenerationGameProducerPhaseV3,
    pub model_epoch: u64,
    pub context_epoch: u64,
    pub controls: [GenerationGameStartupControlV3; 2],
    pub startup_configs: [PalsAdaptiveStartupConfigV5; 2],
    pub startup_config_sha256: [String; 2],
    pub archive_directory: String,
    pub clock: GenerationGameClockConfigV3,
}
impl GenerationWholeGameRequestV3 {
    pub fn validate(&self) -> Result<(), ArenaError> {
        self.plan.validate()?;
        self.producer.validate(&self.plan.opening_splits())?;
        self.clock.actual()?;
        check(
            self.schema == REQUEST_DOMAIN
                && self.game_ordinal < self.plan.games
                && self.model_epoch > 0
                && self.context_epoch > 0
                && self.producer.input_seals.is_empty()
                && self.producer.actual_game_ids.is_empty()
                && relative_name(&self.archive_directory)
                && self.archive_directory.split('/').count() == 1
                && self.startup_config_sha256.iter().all(|s| generation_sha(s)),
            "whole-game exact registered request and fresh producer",
        )?;
        for control in &self.controls {
            check(
                relative_name(&control.config_file) && relative_name(&control.lock_file),
                "whole-game relative startup controls",
            )?;
        }
        let a = &self.startup_configs[0];
        let b = &self.startup_configs[1];
        for config in &self.startup_configs {
            check(
                !config.assets.is_empty()
                    && config.assets.len() <= 96
                    && !config.source_components.is_empty()
                    && config.source_components.len() <= 64,
                "whole-game startup declaration bounds",
            )?;
            config.endpoint.binary.validate()?;
        }
        check(
            a.endpoint.mode == rz_experiments::PalsEngineModeV5::Collection
                && b.endpoint.mode == rz_experiments::PalsEngineModeV5::Collection
                && !a.training_executed
                && !b.training_executed
                && a.source_components == b.source_components
                && a.endpoint.binary.sha256 == b.endpoint.binary.sha256
                && a.endpoint.binary.bytes == b.endpoint.binary.bytes,
            "whole-game binary/source collection identity",
        )?;
        rz_experiments::verify_compiled_source_components_v5(&a.source_components)?;
        let inventory = rz_experiments::compiled_source_inventory_v5()?;
        check(
            inventory.iter().any(|(component, files)| {
                a.source_components.contains_key(component)
                    && files.iter().any(|(path, _, _)| {
                        path == "crates/rz-arena/src/pals_collect_v3/game_process.rs"
                    })
            }),
            "whole-game producer file absent from registered compiled sources",
        )
    }
    pub fn original_bytes(&self) -> Result<Vec<u8>, ArenaError> {
        self.validate()?;
        let mut value = serde_json::to_value(self).map_err(|e| invalid(e.to_string()))?;
        value.sort_all_objects();
        let bytes = serde_json::to_vec(&value).map_err(|e| invalid(e.to_string()))?;
        check(
            !bytes.is_empty() && bytes.len() <= crate::MAX_ORIGINAL_INPUT_BYTES,
            "whole-game original input byte bound",
        )?;
        Ok(bytes)
    }
    pub fn assignment(&self) -> Result<GenerationGameAssignmentV3, ArenaError> {
        self.plan
            .games()?
            .into_iter()
            .nth(usize::from(self.game_ordinal))
            .ok_or_else(|| invalid("whole-game preassigned ordinal absent"))
    }
    fn run_pins_sha256(&self) -> Result<String, ArenaError> {
        generation_digest("rz-pals-whole-game-run-pins/1", &serde_json::json!({
            "plan":self.plan,"producer":self.producer,"producer_phase":self.producer_phase,"model_epoch":self.model_epoch,
            "context_epoch":self.context_epoch,"clock":self.clock,
            "startup_configs":self.startup_configs
        })).map_err(Into::into)
    }
}

fn relative_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 512
        && value.split('/').count() <= 8
        && value.split('/').all(|part| {
            !part.is_empty()
                && part.len() <= 128
                && !part.starts_with('.')
                && part
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        })
}

/// Parent-owned actual executable, private cwd and native startup pins. Caller
/// supplies the original absolute window; this launch never renews it.
pub struct GenerationGameProcessLaunchV3 {
    pub binary: File,
    pub binary_pin: ArtifactRef,
    /// Original absolute no-follow name of the same executable descriptor.
    /// The logical ArtifactRef path alone cannot revalidate named custody.
    pub binary_path: PathBuf,
    pub cwd: File,
    pub cwd_path: PathBuf,
    pub pinned: [Arc<PinnedPalsStartupV5>; 2],
    pub request: GenerationWholeGameRequestV3,
    pub started: Instant,
    pub execution_deadline: Instant,
    pub whole_deadline: Instant,
    pub limits: ProcessLimits,
    /// The existing tree supervisor observes the complete private cwd, including
    /// staged inputs. Its declared total therefore includes those static bytes.
    pub watch: OwnedArtifactTreeWatch,
    pub raw_limits: GenerationGameRawLimitsV3,
}

#[derive(Clone, Copy, Debug)]
pub struct GenerationGameRawLimitsV3 {
    pub max_records: usize,
    pub max_evidence: usize,
    pub max_retained_bytes: u64,
    pub max_read_bytes: u64,
}
impl GenerationGameRawLimitsV3 {
    #[cfg(target_os = "linux")]
    fn validate(&self) -> Result<(), ArenaError> {
        check(
            self.max_records > 0
                && self.max_records <= GENERATION_MAX_RECORDS
                && self.max_evidence > 0
                && self.max_evidence <= 1_048_576
                && self.max_retained_bytes > 0
                && self.max_retained_bytes <= 512 * 1024 * 1024
                && self.max_read_bytes > 0
                && self.max_read_bytes <= 4 * GENERATION_OUTPUT_CAP,
            "whole-game bounded raw read and retention declaration",
        )
    }
}

#[cfg(target_os = "linux")]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GameFilePin {
    file: String,
    sha256: String,
    bytes: u64,
}
#[cfg(target_os = "linux")]
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChildOutput {
    schema: String,
    pid: u32,
    original_input_sha256: String,
    game_ordinal: u16,
    game_id: String,
    opening_id: String,
    engine_a_white: bool,
    source_components: BTreeMap<String, String>,
    startup_config_sha256: [String; 2],
    session_epochs: [u64; 2],
    native_readbacks: [NativeLoadedReadback; 2],
    engine_names: [String; 2],
    termination: CollectionPgnTerminationV3,
    rules_terminated: bool,
    recorded_inputs: usize,
    pgn_receipt_sha256: String,
    decisions: Vec<GenerationGameDecisionTraceV3>,
    manifest: GameFilePin,
}

/// Private read-only observations issued by the actual loaded backend before
/// its first Run. The source-pinned child transports these observations; parent
/// independently joins manifest/runtime descriptors and actual native receipts.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeLoadedReadback {
    model_epoch_sha256: String,
    manifest_sha256: String,
    backend_sha256: String,
    runtime_binary_sha256: String,
    runtime_bundle_sha256: Option<String>,
    native_loading_profile_sha256: Option<String>,
}
impl NativeLoadedReadback {
    fn actual(backend: &PalsVirtualReviewOnnxBackend) -> Self {
        Self {
            model_epoch_sha256: hex(&backend.model_epoch()),
            manifest_sha256: hex(&backend.manifest_digest()),
            backend_sha256: hex(&backend.backend_digest()),
            runtime_binary_sha256: hex(&backend.runtime_binary_digest()),
            runtime_bundle_sha256: backend.runtime_bundle_digest().map(|s| hex(&s)),
            native_loading_profile_sha256: backend.native_loading_profile_digest().map(|s| hex(&s)),
        }
    }
}

/// This capability exists only in the child which reopens exact startup
/// controls and hashes its actual mapped executable. It may run one game once.
pub struct GenerationGameChildAuthorityV3 {
    request: GenerationWholeGameRequestV3,
    // Preserve canonical input custody in the cross-platform capability type;
    // only the Linux loaded-image producer can seal or construct this owner.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    original_bytes: Vec<u8>,
    pinned: [Arc<PinnedPalsStartupV5>; 2],
    cwd: PathBuf,
    _binary: File,
    execution_deadline: Instant,
    whole_deadline: Instant,
}
impl GenerationGameChildAuthorityV3 {
    pub fn request(&self) -> &GenerationWholeGameRequestV3 {
        &self.request
    }
}

/// Explicit child entry. Construction alone does not execute native inference.
pub fn begin_generation_game_child_v3(
    bytes: Vec<u8>,
    execution_deadline: Instant,
    whole_deadline: Instant,
) -> Result<GenerationGameChildAuthorityV3, ArenaError> {
    let until = execution_deadline;
    check(
        Instant::now() < until
            && execution_deadline < whole_deadline
            && whole_deadline.duration_since(execution_deadline) <= Duration::from_secs(30)
            && !bytes.is_empty()
            && bytes.len() <= crate::MAX_ORIGINAL_INPUT_BYTES,
        "whole-game child input bound",
    )?;
    let request: GenerationWholeGameRequestV3 = serde_json::from_slice(&bytes)
        .map_err(|e| invalid(format!("whole-game child input: {e}")))?;
    request.validate()?;
    check(
        request.original_bytes()? == bytes,
        "whole-game original canonical input bytes",
    )?;
    check(
        CHILD_GAME_BEGUN_V3
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok(),
        "whole-game child already admitted a game",
    )?;
    let cwd = std::env::current_dir().map_err(|_| invalid("whole-game current cwd unavailable"))?;
    let open = |index: usize| -> Result<Arc<PinnedPalsStartupV5>, ArenaError> {
        let control = &request.controls[index];
        let pinned = PinnedPalsStartupV5::open(
            PalsStartupOptionsV5 {
                config_path: cwd.join(&control.config_file),
                lock_path: cwd.join(&control.lock_file),
                config_sha256: request.startup_config_sha256[index].clone(),
                metadata_preflight: false,
            },
            &cwd,
        )?;
        check(
            pinned.config == request.startup_configs[index],
            "whole-game child startup readback differs",
        )?;
        Ok(Arc::new(pinned))
    };
    let pinned = [open(0)?, open(1)?];
    check(
        Instant::now() < until,
        "whole-game child startup pins exceeded original deadline",
    )?;
    #[cfg(target_os = "linux")]
    let binary = {
        let file = File::open("/proc/self/exe")
            .map_err(|_| invalid("whole-game mapped executable unavailable"))?;
        let pin = &request.startup_configs[0].endpoint.binary;
        let mut credit = ReadCredit {
            remaining: pin.bytes,
            until,
            cancel: None,
        };
        verify_held(
            &file,
            &GameFilePin {
                file: "mapped-child".into(),
                sha256: pin.sha256.clone(),
                bytes: pin.bytes,
            },
            false,
            &mut credit,
        )?;
        file
    };
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (until, &pinned);
        Err(invalid(
            "whole-game original loaded-image producer requires Linux",
        ))
    }
    #[cfg(target_os = "linux")]
    {
        Ok(GenerationGameChildAuthorityV3 {
            request,
            original_bytes: bytes,
            pinned,
            cwd,
            _binary: binary,
            execution_deadline,
            whole_deadline,
        })
    }
}

/// Executes one actual in-process native game inside this new OS child, joins
/// its two native owners and writer, then seals an output from the opaque archive.
/// The caller preserves returned bytes as stdout; parent authority still requires
/// its own held-file readback and original supervisor completion.
pub fn run_generation_game_child_v3(
    authority: GenerationGameChildAuthorityV3,
    execution_deadline: Instant,
    whole_deadline: Instant,
) -> Result<Vec<u8>, ArenaError> {
    run_generation_game_child_with_cancellation_v3(
        authority,
        execution_deadline,
        whole_deadline,
        Arc::new(AtomicBool::new(false)),
    )
}

/// The explicit game-worker shares its original SIGINT/SIGTERM flag with
/// startup, native intake, the actual game loop and each exact root control.
/// Cancellation keeps the existing drain/partial archive/unknown semantics.
pub fn run_generation_game_child_with_cancellation_v3(
    authority: GenerationGameChildAuthorityV3,
    execution_deadline: Instant,
    whole_deadline: Instant,
    cancellation: Arc<AtomicBool>,
) -> Result<Vec<u8>, ArenaError> {
    check(
        Instant::now() < execution_deadline
            && execution_deadline < whole_deadline
            && execution_deadline <= authority.execution_deadline
            && whole_deadline <= authority.whole_deadline
            && !cancellation.load(Ordering::Acquire),
        "whole-game child original local execution/closure window/cancellation",
    )?;
    let readbacks: [Arc<Mutex<Option<NativeLoadedReadback>>>; 2] =
        std::array::from_fn(|_| Arc::new(Mutex::new(None)));
    let intakes: [NativeVirtualReviewIntakeV5; 2] = readbacks.clone().map(|slot| {
        let cancellation = Arc::clone(&cancellation);
        Box::new(
            move |backend: &PalsVirtualReviewOnnxBackend, until: Instant| {
                if Instant::now() >= until || cancellation.load(Ordering::Acquire) {
                    return Err(BackendError::new(
                        FailureKind::InvalidInput,
                        FailureStage::Admission,
                        "whole-game intake original deadline/cancellation",
                    ));
                }
                let mut state = slot.try_lock().map_err(|_| {
                    BackendError::new(
                        FailureKind::BackendFailure,
                        FailureStage::Admission,
                        "whole-game intake owner busy",
                    )
                })?;
                if state.is_some() {
                    return Err(BackendError::new(
                        FailureKind::IdentityMismatch,
                        FailureStage::Admission,
                        "whole-game duplicate native intake",
                    ));
                }
                *state = Some(NativeLoadedReadback::actual(backend));
                Ok(())
            },
        ) as NativeVirtualReviewIntakeV5
    });
    let mut factory = ActualGenerationSelfplayFactoryV3::new(
        authority.pinned.clone(),
        authority.request.clock.actual()?,
    )?
    .with_native_intakes(intakes)
    .with_cancellation(Arc::clone(&cancellation));
    let mut collector = GenerationSelfplayCollectorV3::new(
        authority.request.plan.clone(),
        authority.request.producer.clone(),
        authority.request.model_epoch,
        authority.request.context_epoch,
        authority.cwd.join(&authority.request.archive_directory),
        whole_deadline,
    )?;
    let archive = collector.run_standalone_game_child_with_cancellation(
        usize::from(authority.request.game_ordinal),
        &mut factory,
        execution_deadline,
        whole_deadline,
        Some(&cancellation),
    )?;
    check(
        !cancellation.load(Ordering::Acquire),
        "canceled whole-game child cannot issue completed output authority",
    )?;
    let [readback_a, readback_b] = readbacks.map(|slot| {
        let mut state = slot
            .try_lock()
            .map_err(|_| invalid("whole-game actual native intake owner busy"))?;
        state
            .take()
            .ok_or_else(|| invalid("whole-game actual native intake unobserved"))
    });
    let output = seal_actual_child_output(
        &authority,
        &archive,
        [readback_a?, readback_b?],
        whole_deadline,
    )?;
    check(
        !cancellation.load(Ordering::Acquire),
        "canceled whole-game child cannot publish completed sealed output",
    )?;
    Ok(output)
}

fn seal_actual_child_output(
    authority: &GenerationGameChildAuthorityV3,
    archive: &GenerationStandaloneGameArchiveV3,
    native_readbacks: [NativeLoadedReadback; 2],
    until: Instant,
) -> Result<Vec<u8>, ArenaError> {
    let epochs: Vec<_> = archive
        .actual_close()
        .process_epochs()
        .map(|e| e.0)
        .collect();
    let assignment = authority.request.assignment()?;
    let report = archive.report();
    check(
        epochs.len() == 2
            && epochs[0] != 0
            && epochs[1] != 0
            && epochs[0] != epochs[1]
            && report.game_id == assignment.game_id
            && report.opening_id == assignment.opening_id
            && report.engine_a_white == assignment.engine_a_white
            && archive.actual_trajectory().game_id() == assignment.game_id
            && archive.directory() == authority.cwd.join(&authority.request.archive_directory),
        "whole-game actual standalone archive/close scope",
    )?;
    archive.manifest().validate()?;
    #[cfg(target_os = "linux")]
    {
        let dir = open_directory(&authority.cwd)?;
        let name = format!("{}/manifest.json", authority.request.archive_directory);
        let file = open_relative(&dir, &name)?;
        let size = file
            .metadata()
            .map_err(|_| invalid("whole-game manifest metadata"))?
            .len();
        check(
            size > 0 && size <= MAX_OUTPUT_BYTES,
            "whole-game manifest control bound",
        )?;
        let bytes = read_unpinned_control(&file, size, until)?;
        let recovered: GenerationChunkManifestV3 = serde_json::from_slice(&bytes)
            .map_err(|e| invalid(format!("whole-game manifest readback: {e}")))?;
        check(
            serde_json::to_value(recovered).map_err(|e| invalid(e.to_string()))?
                == serde_json::to_value(archive.manifest()).map_err(|e| invalid(e.to_string()))?,
            "whole-game joined archive readback differs",
        )?;
        let output = ChildOutput {
            schema: OUTPUT_DOMAIN.into(),
            pid: std::process::id(),
            original_input_sha256: format!("{:x}", Sha256::digest(&authority.original_bytes)),
            game_ordinal: authority.request.game_ordinal,
            game_id: report.game_id.clone(),
            opening_id: report.opening_id.clone(),
            engine_a_white: report.engine_a_white,
            source_components: authority.request.startup_configs[0]
                .source_components
                .clone(),
            startup_config_sha256: authority.request.startup_config_sha256.clone(),
            session_epochs: [epochs[0], epochs[1]],
            native_readbacks,
            engine_names: archive.engine_names().clone(),
            termination: report.termination,
            rules_terminated: report.rules_terminated,
            recorded_inputs: report.recorded_inputs,
            pgn_receipt_sha256: report.pgn_receipt_sha256.clone(),
            decisions: archive.decision_trace().to_vec(),
            manifest: GameFilePin {
                file: name,
                sha256: format!("{:x}", Sha256::digest(&bytes)),
                bytes: size,
            },
        };
        let mut value = serde_json::to_value(output).map_err(|e| invalid(e.to_string()))?;
        value.sort_all_objects();
        let mut bytes = serde_json::to_vec(&value).map_err(|e| invalid(e.to_string()))?;
        bytes.push(b'\n');
        check(
            bytes.len() as u64 <= MAX_OUTPUT_BYTES && Instant::now() < until,
            "whole-game child output bound/original deadline",
        )?;
        use std::io::Write;
        let mut options = cap_std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        let mut output_file = dir
            .open_with(GENERATION_GAME_CHILD_OUTPUT_V3, &options)
            .map_err(|_| invalid("whole-game exclusive output publication"))?;
        output_file
            .write_all(&bytes)
            .and_then(|_| output_file.sync_all())
            .map_err(|_| invalid("whole-game output durable publication"))?;
        drop(output_file);
        Ok(bytes)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (until, &native_readbacks);
        Err(invalid(
            "whole-game child output requires Linux pinned files",
        ))
    }
}

struct GameProcessOwner {
    launch: GenerationGameProcessLaunchV3,
    launch_sequence: u64,
    process: Option<OriginalProcessOutput>,
    #[cfg(target_os = "linux")]
    held: Vec<HeldGameArtifact>,
}

/// A name and content pin belong to the same original read-only descriptor.
/// The descriptor is retained even after the decoded record/evidence maps are
/// released; reopening a matching hash at a different inode grants no custody.
#[cfg(target_os = "linux")]
struct HeldGameArtifact {
    file: File,
    pin: GameFilePin,
}
/// Every failure retains original process and native/input/output file scopes.
/// Dropping unresolved physical cleanup quarantines the owner without a retry.
pub struct GenerationGameProcessFailureV3 {
    pub error: ArenaError,
    owner: Option<Box<GameProcessOwner>>,
}
impl GenerationGameProcessFailureV3 {
    pub fn process(&self) -> Option<&OriginalProcessOutput> {
        self.owner.as_ref().and_then(|o| o.process.as_ref())
    }
    pub fn request(&self) -> Option<&GenerationWholeGameRequestV3> {
        self.owner.as_ref().map(|o| &o.launch.request)
    }
}
impl std::fmt::Debug for GenerationGameProcessFailureV3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerationGameProcessFailureV3")
            .field("error", &self.error)
            .field("process", &self.process().map(|p| &p.process().receipt))
            .finish_non_exhaustive()
    }
}
impl Drop for GenerationGameProcessFailureV3 {
    fn drop(&mut self) {
        if self.process().is_some_and(|p| {
            p.process().pending_child.is_some()
                || p.process().receipt.group_cleanup == crate::CleanupStatus::Unverified
        }) && let Some(owner) = self.owner.take()
        {
            let _quarantine = Box::leak(owner);
        }
    }
}

/// Successful OS game closure plus verified original raw artifacts. This cannot
/// be deserialized or minted from stdout booleans. It is a single-game receipt,
/// never a completed 128-game archive or an optimizer/training acceptance.
pub struct VerifiedGenerationGameProcessV3 {
    owner: GameProcessOwner,
    receipt: serde_json::Value,
    manifest: GenerationChunkManifestV3,
    records: BTreeMap<String, GenerationRecordV3>,
    evidence: BTreeMap<String, serde_json::Value>,
    trajectory: Option<GenerationTrajectoryAuthorityV3>,
    summary: VerifiedGenerationGameSummaryV3,
}

/// Small independently replayed result and PGN retained after raw row parsing.
/// It is issued only by the original process/readback path, never deserialized
/// into an actual game result by a caller or an aggregate manifest parser.
pub struct VerifiedGenerationGameSummaryV3 {
    pgn_evidence: serde_json::Value,
    pgn_receipt_sha256: String,
    result: CollectionPgnResultV3,
    termination: CollectionPgnTerminationV3,
    rules_terminated: bool,
    engine_names: [String; 2],
    decision_count: usize,
}
impl VerifiedGenerationGameSummaryV3 {
    pub fn pgn_evidence(&self) -> &serde_json::Value {
        &self.pgn_evidence
    }
    pub fn pgn_receipt_sha256(&self) -> &str {
        &self.pgn_receipt_sha256
    }
    pub fn result(&self) -> CollectionPgnResultV3 {
        self.result
    }
    pub fn termination(&self) -> CollectionPgnTerminationV3 {
        self.termination
    }
    pub fn rules_terminated(&self) -> bool {
        self.rules_terminated
    }
    pub fn engine_names(&self) -> &[String; 2] {
        &self.engine_names
    }
    pub fn decision_count(&self) -> usize {
        self.decision_count
    }
}
impl VerifiedGenerationGameProcessV3 {
    pub fn receipt(&self) -> &serde_json::Value {
        &self.receipt
    }
    pub fn process(&self) -> &OriginalProcessOutput {
        self.owner
            .process
            .as_ref()
            .expect("verified original process")
    }
    pub fn request(&self) -> &GenerationWholeGameRequestV3 {
        &self.owner.launch.request
    }
    pub fn manifest(&self) -> &GenerationChunkManifestV3 {
        &self.manifest
    }
    pub fn records(&self) -> &BTreeMap<String, GenerationRecordV3> {
        &self.records
    }
    pub fn evidence(&self) -> &BTreeMap<String, serde_json::Value> {
        &self.evidence
    }
    pub fn trajectory(&self) -> Option<&GenerationTrajectoryAuthorityV3> {
        self.trajectory.as_ref()
    }
    pub fn cwd_path(&self) -> &Path {
        &self.owner.launch.cwd_path
    }
    pub fn summary(&self) -> &VerifiedGenerationGameSummaryV3 {
        &self.summary
    }

    /// Consume the verified current game and release its potentially large raw
    /// maps. This cannot fail halfway and drop the original authority: all
    /// remaining descriptor, supervisor and provenance scopes move together.
    /// The on-disk raw files are not removed or replaced by this operation.
    pub fn into_archive(self) -> VerifiedGenerationGameArchiveV3 {
        let record_seals = self
            .records
            .into_keys()
            .map(|seal| verified_sha_bytes(&seal))
            .collect();
        let evidence_seals = self
            .evidence
            .into_keys()
            .map(|seal| verified_sha_bytes(&seal))
            .collect();
        VerifiedGenerationGameArchiveV3 {
            owner: self.owner,
            receipt: self.receipt,
            manifest: self.manifest,
            trajectory: self.trajectory,
            summary: self.summary,
            record_seals,
            evidence_seals,
        }
    }
}

/// Compact custody of one verified actual game. Large decoded records and
/// evidence are absent, while every original file and OS completion scope is
/// still held. A generation aggregate may retain 128 of these; it must not
/// substitute JSON/readback hashes for this opaque original authority.
pub struct VerifiedGenerationGameArchiveV3 {
    owner: GameProcessOwner,
    receipt: serde_json::Value,
    manifest: GenerationChunkManifestV3,
    trajectory: Option<GenerationTrajectoryAuthorityV3>,
    summary: VerifiedGenerationGameSummaryV3,
    record_seals: Vec<[u8; 32]>,
    evidence_seals: Vec<[u8; 32]>,
}
impl VerifiedGenerationGameArchiveV3 {
    pub fn receipt(&self) -> &serde_json::Value {
        &self.receipt
    }
    pub fn process(&self) -> &OriginalProcessOutput {
        self.owner
            .process
            .as_ref()
            .expect("verified original process")
    }
    pub fn request(&self) -> &GenerationWholeGameRequestV3 {
        &self.owner.launch.request
    }
    pub fn manifest(&self) -> &GenerationChunkManifestV3 {
        &self.manifest
    }
    pub fn trajectory(&self) -> Option<&GenerationTrajectoryAuthorityV3> {
        self.trajectory.as_ref()
    }
    pub fn cwd_path(&self) -> &Path {
        &self.owner.launch.cwd_path
    }
    pub fn summary(&self) -> &VerifiedGenerationGameSummaryV3 {
        &self.summary
    }
    pub fn record_seals(&self) -> &[[u8; 32]] {
        &self.record_seals
    }
    pub fn evidence_seals(&self) -> &[[u8; 32]] {
        &self.evidence_seals
    }

    /// Reissue one exact pre-move root from this actual game's retained output.
    /// The opaque saved root owns an Arc to this original process/archive; no
    /// FEN, PGN, standalone receipt or caller boolean can create that authority.
    /// The separate bound covers only read/replay, never a new child execution.
    pub fn saved_action_root(
        self: &Arc<Self>,
        decision_index: usize,
        until: Instant,
        cancel: Option<&AtomicBool>,
        max_read_bytes: u64,
    ) -> Result<(GenerationSavedActionRootV3, u64), ArenaError> {
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (decision_index, until, cancel, max_read_bytes);
            Err(invalid(
                "saved actual archive roots require Linux pinned transport",
            ))
        }
        #[cfg(target_os = "linux")]
        {
            check(
                decision_index < self.summary.decision_count
                    && max_read_bytes > 0
                    && max_read_bytes <= MAX_OUTPUT_BYTES,
                "saved actual root bounded decision/read credit",
            )?;
            let cwd = open_directory(self.cwd_path())?;
            same_directory(&self.owner.launch.cwd, &cwd)?;
            let mut matches = self
                .owner
                .held
                .iter()
                .filter(|held| held.pin.file == GENERATION_GAME_CHILD_OUTPUT_V3);
            let held = matches
                .next()
                .ok_or_else(|| invalid("saved root original output absent"))?;
            check(
                matches.next().is_none(),
                "saved root original output ambiguous",
            )?;
            same_named_artifact(&cwd, &held.file, &held.pin)?;
            let mut credit = ReadCredit {
                remaining: max_read_bytes,
                until,
                cancel,
            };
            let bytes = verify_held(&held.file, &held.pin, true, &mut credit)?;
            check(
                self.receipt["original_output_sha256"] == held.pin.sha256
                    && bytes == self.process().process().stdout,
                "saved root readback differs from original actual child output",
            )?;
            let output: ChildOutput = serde_json::from_slice(&bytes)
                .map_err(|e| invalid(format!("saved root original output decode: {e}")))?;
            let request = self.request();
            let assignment = request.assignment()?;
            check(
                output.schema == OUTPUT_DOMAIN
                    && output.game_id == assignment.game_id
                    && output.opening_id == assignment.opening_id
                    && output.engine_a_white == assignment.engine_a_white
                    && output.decisions.len() == self.summary.decision_count
                    && output.pgn_receipt_sha256 == self.summary.pgn_receipt_sha256
                    && output.session_epochs
                        == serde_json::from_value::<[u64; 2]>(
                            self.receipt["session_epochs"].clone(),
                        )
                        .map_err(|_| invalid("saved root original session epochs"))?,
                "saved root original decision/PGN/session identity",
            )?;
            let opening = request
                .plan
                .openings
                .iter()
                .find(|opening| opening.id == assignment.opening_id)
                .ok_or_else(|| invalid("saved root original opening absent"))?;
            let mut position = match &opening.initial_fen {
                Some(fen) => Position::from_fen(fen).map_err(|e| invalid(e.to_string()))?,
                None => Position::startpos(),
            };
            for movement in &opening.moves {
                credit.take(0)?;
                position
                    .make_move(BoardMove::from_uci(movement).map_err(|e| invalid(e.to_string()))?)
                    .map_err(|e| invalid(e.to_string()))?;
            }
            let (white, black) = if output.engine_a_white {
                (&output.engine_names[0], &output.engine_names[1])
            } else {
                (&output.engine_names[1], &output.engine_names[0])
            };
            let mut pgn = CollectionGamePgnV3::begin(
                output.game_id.clone(),
                white.clone(),
                black.clone(),
                &position,
            )?;
            for (index, decision) in output.decisions.iter().enumerate().take(decision_index + 1) {
                credit.take(0)?;
                let packed: Move16 = serde_json::from_value(serde_json::json!(decision.movement))
                    .map_err(|e| invalid(e.to_string()))?;
                let movement = packed.unpack().map_err(|e| invalid(e.to_string()))?;
                // Includes the selected boundary: time forfeits and illegal
                // last moves cannot issue a successful accepted action root.
                pgn.push_actual_move(
                    &position,
                    movement,
                    Duration::from_nanos(decision.elapsed_ns),
                )?;
                if index == decision_index {
                    let engine_a =
                        (position.side_to_move() == Color::White) == output.engine_a_white;
                    let epoch = output.session_epochs[usize::from(!engine_a)];
                    return Ok((
                        GenerationSavedActionRootV3::from_verified_archive(
                            Arc::clone(self),
                            position,
                            epoch,
                        )?,
                        max_read_bytes - credit.remaining,
                    ));
                }
                position
                    .make_move(movement)
                    .map_err(|e| invalid(e.to_string()))?;
            }
            Err(invalid("saved root selected decision absent"))
        }
    }

    /// A separately bounded publication readback, after the original child has
    /// already been verified and joined. The new deadline controls only this
    /// read-only operation and never restarts or extends the child's window.
    /// Failure leaves this owner and the raw files intact for diagnosis.
    pub fn reverify_owned_artifacts(
        &self,
        until: Instant,
        cancel: Option<&AtomicBool>,
        max_read_bytes: u64,
    ) -> Result<u64, ArenaError> {
        reverify_owned_artifacts(&self.owner, until, cancel, max_read_bytes)
    }
}

// Private callers can supply only a seal which already passed the canonical
// 64-character lowercase SHA validator while verifying actual raw rows.
fn verified_sha_bytes(value: &str) -> [u8; 32] {
    debug_assert!(generation_sha(value));
    let mut result = [0; 32];
    for (pair, byte) in value.as_bytes().chunks_exact(2).zip(&mut result) {
        let nibble = |b: u8| (b.wrapping_add(9 * (b >> 6))) & 0xf;
        *byte = (nibble(pair[0]) << 4) | nibble(pair[1]);
    }
    result
}

/// One original OS child per preassigned game, strictly serial and non-retrying.
/// A numeric PID may be reused after the previous original child is joined.
/// Freshness belongs to each private supervisor invocation and launch sequence.
#[derive(Default)]
pub struct GenerationGameProcessSeriesV3 {
    next_ordinal: u16,
    run_pins_sha256: Option<String>,
    halted: bool,
}
impl GenerationGameProcessSeriesV3 {
    pub fn completed_games(&self) -> u16 {
        self.next_ordinal
    }
    pub fn run_next(
        &mut self,
        launch: GenerationGameProcessLaunchV3,
        cancel: Option<&AtomicBool>,
    ) -> Result<VerifiedGenerationGameProcessV3, Box<GenerationGameProcessFailureV3>> {
        let mut owner = Box::new(GameProcessOwner {
            launch,
            launch_sequence: 0,
            process: None,
            #[cfg(target_os = "linux")]
            held: vec![],
        });
        let result = (|| {
            owner.launch.request.validate()?;
            check(
                !self.halted
                    && self.next_ordinal < 128
                    && owner.launch.request.game_ordinal == self.next_ordinal
                    && self.run_pins_sha256.as_ref().is_none_or(|pins| {
                        owner
                            .launch
                            .request
                            .run_pins_sha256()
                            .is_ok_and(|actual| pins == &actual)
                    }),
                "whole-game original serial assignment or halted owner",
            )?;
            let pins = owner.launch.request.run_pins_sha256()?;
            self.run_pins_sha256.get_or_insert(pins);
            owner.launch_sequence = NEXT_PROCESS_LAUNCH_V3
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |sequence| {
                    sequence.checked_add(1)
                })
                .map_err(|_| invalid("whole-game actual process launch identity exhausted"))?;
            run_parent(&mut owner, cancel)
        })();
        match result {
            Ok(verified) => {
                self.next_ordinal += 1;
                Ok(VerifiedGenerationGameProcessV3 {
                    owner: *owner,
                    receipt: verified.receipt,
                    manifest: verified.manifest,
                    records: verified.records,
                    evidence: verified.evidence,
                    trajectory: verified.trajectory,
                    summary: verified.summary,
                })
            }
            Err(error) => {
                self.halted = true;
                Err(Box::new(GenerationGameProcessFailureV3 {
                    error,
                    owner: Some(owner),
                }))
            }
        }
    }
}

struct VerifiedRaw {
    receipt: serde_json::Value,
    manifest: GenerationChunkManifestV3,
    records: BTreeMap<String, GenerationRecordV3>,
    evidence: BTreeMap<String, serde_json::Value>,
    trajectory: Option<GenerationTrajectoryAuthorityV3>,
    summary: VerifiedGenerationGameSummaryV3,
}

#[cfg(not(target_os = "linux"))]
fn run_parent(
    _owner: &mut GameProcessOwner,
    _cancel: Option<&AtomicBool>,
) -> Result<VerifiedRaw, ArenaError> {
    Err(invalid(
        "whole-game actual original process transport requires Linux",
    ))
}

#[cfg(target_os = "linux")]
fn run_parent(
    owner: &mut GameProcessOwner,
    cancel: Option<&AtomicBool>,
) -> Result<VerifiedRaw, ArenaError> {
    let launch = &owner.launch;
    check(
        launch
            .execution_deadline
            .checked_duration_since(launch.started)
            .is_some_and(|d| d.as_nanos() <= u128::from(MAX_GAME_WINDOW_NS))
            && launch
                .whole_deadline
                .checked_duration_since(launch.execution_deadline)
                .is_some_and(|d| d.as_nanos() <= u128::from(MAX_GAME_CLEANUP_NS)),
        "whole-game original execution and cleanup policy bounds",
    )?;
    launch.request.validate()?;
    launch.binary_pin.validate()?;
    launch.raw_limits.validate()?;
    check(
        launch.binary_pin.bytes <= MAX_BINARY_BYTES
            && launch.binary_pin.sha256 == launch.request.startup_configs[0].endpoint.binary.sha256
            && launch.binary_pin.bytes == launch.request.startup_configs[0].endpoint.binary.bytes,
        "whole-game held executable differs from both endpoint pins",
    )?;
    let mut credit = ReadCredit {
        remaining: launch.raw_limits.max_read_bytes,
        until: launch.execution_deadline,
        cancel,
    };
    let binary_pin = GameFilePin {
        file: "held-runner".into(),
        sha256: launch.binary_pin.sha256.clone(),
        bytes: launch.binary_pin.bytes,
    };
    verify_held(&launch.binary, &binary_pin, false, &mut credit)?;
    same_named_binary(launch)?;
    let cwd = open_directory(&launch.cwd_path)?;
    same_directory(&launch.cwd, &cwd)?;
    for (index, pinned) in launch.pinned.iter().enumerate() {
        pinned.config.validate_against(&pinned.lock)?;
        check(
            !pinned.metadata_preflight
                && pinned.private_cwd() == launch.cwd_path
                && pinned.config == launch.request.startup_configs[index]
                && pinned.config_sha256 == launch.request.startup_config_sha256[index],
            "whole-game retained native startup owner differs",
        )?;
        for name in [
            &launch.request.controls[index].config_file,
            &launch.request.controls[index].lock_file,
        ] {
            let file = open_relative(&cwd, name)?;
            let size = file
                .metadata()
                .map_err(|_| invalid("whole-game control metadata"))?
                .len();
            check(
                size > 0 && size <= MAX_CONTROL_BYTES,
                "whole-game control file bound",
            )?;
            credit.take(
                size.checked_mul(2)
                    .ok_or_else(|| invalid("whole-game control read credit overflow"))?,
            )?;
            let bytes = read_unpinned_control(&file, size, launch.execution_deadline)?;
            if name == &launch.request.controls[index].config_file {
                check(
                    format!("{:x}", Sha256::digest(&bytes)) == pinned.config_sha256,
                    "whole-game held startup config SHA",
                )?;
            } else {
                let lock = rz_experiments::PalsInputLockV5::from_json(
                    std::str::from_utf8(&bytes)
                        .map_err(|_| invalid("whole-game startup lock UTF8"))?,
                )?;
                check(
                    lock.canonical_sha256 == pinned.lock.canonical_sha256,
                    "whole-game held startup lock readback",
                )?;
            }
            owner.held.push(HeldGameArtifact {
                file,
                pin: GameFilePin {
                    file: name.clone(),
                    sha256: format!("{:x}", Sha256::digest(&bytes)),
                    bytes: size,
                },
            });
        }
    }
    check(
        cwd.symlink_metadata(GENERATION_GAME_CHILD_OUTPUT_V3)
            .is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound)
            && cwd
                .symlink_metadata(&launch.request.archive_directory)
                .is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound),
        "whole-game output/archive must be fresh absent names",
    )?;
    let input = launch.request.original_bytes()?;
    let input_sha256 = format!("{:x}", Sha256::digest(&input));
    let window = OriginalProcessWindow::new(
        launch.started,
        launch.execution_deadline,
        launch.whole_deadline,
    )?;
    let mut mapped_credit = ReadCredit {
        remaining: launch.binary_pin.bytes,
        until: launch.execution_deadline,
        cancel,
    };
    let mut verify_loaded = |file: &File| {
        verify_held(file, &binary_pin, false, &mut mapped_credit)
            .map(|_| ())
            .map_err(|_| ())
    };
    let tick = monotonic_ns()?;
    let sampled = Instant::now();
    let absolute = |deadline: Instant| -> Result<u64, ArenaError> {
        let remaining = deadline
            .checked_duration_since(sampled)
            .ok_or_else(|| invalid("whole-game original deadline expired before launch"))?;
        tick.checked_add(
            u64::try_from(remaining.as_nanos())
                .map_err(|_| invalid("whole-game original deadline tick overflow"))?,
        )
        .ok_or_else(|| invalid("whole-game original absolute clock overflow"))
    };
    let args = [
        OsString::from(GENERATION_GAME_CHILD_ARGUMENT_V3),
        OsString::from(absolute(launch.execution_deadline)?.to_string()),
        OsString::from(absolute(launch.whole_deadline)?.to_string()),
    ];
    let process = crate::process::supervise_verified_input_in_original_window(
        &launch.binary,
        &args,
        &launch.cwd,
        launch.limits,
        cancel,
        &launch.watch,
        OriginalProcessInput {
            bytes: &input,
            window,
        },
        &mut verify_loaded,
    )?;
    owner.process = Some(process);
    let process = owner
        .process
        .as_ref()
        .expect("original child output retained");
    check(
        process.transport_complete(),
        "whole-game original child/input/pipe/group/reap incomplete",
    )?;
    credit.until = launch.whole_deadline;
    let loaded = process.checked_loaded_image()?;
    let timing = process.checked_timing()?;
    verify_held(&launch.binary, &binary_pin, false, &mut credit)?;
    same_named_binary(launch)?;
    same_directory(&launch.cwd, &cwd)?;
    let output_file = open_relative(&cwd, GENERATION_GAME_CHILD_OUTPUT_V3)?;
    let stdout = &process.process().stdout;
    check(
        !stdout.is_empty() && stdout.len() as u64 <= MAX_OUTPUT_BYTES,
        "whole-game bounded stdout output",
    )?;
    let output_pin = GameFilePin {
        file: GENERATION_GAME_CHILD_OUTPUT_V3.into(),
        sha256: format!("{:x}", Sha256::digest(stdout)),
        bytes: stdout.len() as u64,
    };
    let bytes = verify_held(&output_file, &output_pin, true, &mut credit)?;
    owner.held.push(HeldGameArtifact {
        file: output_file,
        pin: output_pin.clone(),
    });
    let output: ChildOutput = serde_json::from_slice(&bytes)
        .map_err(|e| invalid(format!("whole-game output readback: {e}")))?;
    let assignment = launch.request.assignment()?;
    check(
        output.schema == OUTPUT_DOMAIN
            && output.pid == process.process().receipt.pid
            && output.original_input_sha256 == input_sha256
            && output.game_ordinal == launch.request.game_ordinal
            && output.game_id == assignment.game_id
            && output.opening_id == assignment.opening_id
            && output.engine_a_white == assignment.engine_a_white
            && output.source_components == launch.request.startup_configs[0].source_components
            && output.startup_config_sha256 == launch.request.startup_config_sha256
            && output.session_epochs[0] != 0
            && output.session_epochs[1] != 0
            && output.session_epochs[0] < output.session_epochs[1]
            && output.decisions.len() <= usize::from(launch.request.plan.max_plies)
            && output.manifest.file
                == format!("{}/manifest.json", launch.request.archive_directory)
            && output.manifest.bytes <= MAX_OUTPUT_BYTES,
        "whole-game exact child PID/input/source/assignment/owner readback",
    )?;
    for (pinned, observed) in launch.pinned.iter().zip(&output.native_readbacks) {
        credit.take(0)?;
        let manifest = pinned.asset(PalsStartupAssetPurposeV5::ExportManifest)?;
        check(
            manifest.declaration.artifact.bytes <= MAX_CONTROL_BYTES,
            "whole-game actual export manifest byte bound",
        )?;
        let actual = verify_held(
            &manifest.file,
            &GameFilePin {
                file: manifest.declaration.snapshot_path.clone(),
                sha256: manifest.declaration.artifact.sha256.clone(),
                bytes: manifest.declaration.artifact.bytes,
            },
            true,
            &mut credit,
        )?;
        let parsed = validate_virtual_review_export_manifest(
            &actual,
            VirtualReviewExecutionMode::Collection,
        )
        .map_err(|_| invalid("whole-game actual export manifest semantics"))?;
        let checkpoint = pinned.asset(PalsStartupAssetPurposeV5::Weights)?;
        let runtime = pinned.asset(PalsStartupAssetPurposeV5::RuntimeLibrary)?;
        for asset in [checkpoint, runtime] {
            verify_held(
                &asset.file,
                &GameFilePin {
                    file: asset.declaration.snapshot_path.clone(),
                    sha256: asset.declaration.artifact.sha256.clone(),
                    bytes: asset.declaration.artifact.bytes,
                },
                false,
                &mut credit,
            )?;
        }
        check(
            observed.model_epoch_sha256 == hex(&parsed.checkpoint_sha256)
                && observed.model_epoch_sha256 == checkpoint.declaration.artifact.sha256
                && observed.model_epoch_sha256 == launch.request.producer.model_sha256
                && observed.manifest_sha256 == hex(&parsed.manifest_sha256())
                && observed.runtime_binary_sha256 == runtime.declaration.artifact.sha256
                && observed.runtime_binary_sha256 == launch.request.producer.runtime_sha256
                && generation_sha(&observed.backend_sha256)
                && observed
                    .runtime_bundle_sha256
                    .as_ref()
                    .is_none_or(|s| generation_sha(s))
                && observed
                    .native_loading_profile_sha256
                    .as_ref()
                    .is_none_or(|s| generation_sha(s)),
            "whole-game actual loaded checkpoint/manifest/runtime and scoped producer phase",
        )?;
    }
    let manifest_file = open_relative(&cwd, &output.manifest.file)?;
    let manifest_bytes = verify_held(&manifest_file, &output.manifest, true, &mut credit)?;
    owner.held.push(HeldGameArtifact {
        file: manifest_file,
        pin: output.manifest.clone(),
    });
    let manifest: GenerationChunkManifestV3 = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| invalid(format!("whole-game manifest decode: {e}")))?;
    manifest.validate()?;
    let (records, evidence, trajectory, summary) = verify_raw(
        &launch.request,
        &output,
        &manifest,
        &cwd,
        &mut owner.held,
        &mut credit,
        launch.raw_limits,
    )?;
    let receipt = serde_json::json!({"schema":RECEIPT_DOMAIN,"game_ordinal":output.game_ordinal,
        "launch_sequence":owner.launch_sequence,"run_pins_sha256":launch.request.run_pins_sha256()?,
        "game_id":output.game_id,"opening_id":output.opening_id,"engine_a_white":output.engine_a_white,
        "process":process.process().receipt,"original_transport":process.observation(),
        "original_input_sha256":input_sha256,"original_output_sha256":output_pin.sha256,
        "binary_sha256":launch.binary_pin.sha256,"source_components":output.source_components,
        "startup_config_sha256":output.startup_config_sha256,"session_epochs":output.session_epochs,
        "producer_phase":launch.request.producer_phase,"native_readbacks":output.native_readbacks,
        "manifest":output.manifest,"pgn_receipt_sha256":output.pgn_receipt_sha256,
        "loaded_image":{"pid":loaded.process_id(),"device":loaded.file_device(),"inode":loaded.file_inode(),
            "observed_ns":loaded.observed_ns(),"first_input_written_ns":loaded.first_input_written_ns()},
        "original_timing":{"entry_ns":timing.supervisor_entry_ns(),"launch_ns":timing.launch_started_ns(),
            "exit_ns":timing.exit_observed_ns(),"finished_ns":timing.supervisor_finished_ns()},
        "assurance_scope":"fresh_owned_OS_child_single_actual_game_raw_readback_group_reap; no_training_or_128game_completion"});
    check(
        Instant::now() < launch.whole_deadline,
        "whole-game parent raw verification exceeded original deadline",
    )?;
    Ok(VerifiedRaw {
        receipt,
        manifest,
        records,
        evidence,
        trajectory,
        summary,
    })
}

#[cfg(target_os = "linux")]
struct ReadCredit<'a> {
    remaining: u64,
    until: Instant,
    cancel: Option<&'a AtomicBool>,
}
#[cfg(target_os = "linux")]
impl ReadCredit<'_> {
    fn take(&mut self, bytes: u64) -> Result<(), ArenaError> {
        check(
            Instant::now() < self.until && !self.cancel.is_some_and(|c| c.load(Ordering::Acquire)),
            "whole-game original read deadline/cancellation",
        )?;
        self.remaining = self
            .remaining
            .checked_sub(bytes)
            .ok_or_else(|| invalid("whole-game pinned read credit exhausted"))?;
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn open_directory(path: &Path) -> Result<cap_std::fs::Dir, ArenaError> {
    use cap_fs_ext::DirExt;
    use std::os::unix::fs::OpenOptionsExt;
    check(
        path.is_absolute()
            && path.components().count() <= 64
            && path
                .components()
                .all(|c| matches!(c, Component::RootDir | Component::Normal(_))),
        "whole-game absolute no-follow private cwd",
    )?;
    let root = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open("/")
        .map_err(|_| invalid("whole-game root directory pin"))?;
    let mut dir = cap_std::fs::Dir::from_std_file(root);
    for component in path.components() {
        if let Component::Normal(name) = component {
            dir = dir
                .open_dir_nofollow(name)
                .map_err(|_| invalid("whole-game private directory pin"))?;
        }
    }
    Ok(dir)
}
#[cfg(target_os = "linux")]
fn same_directory(file: &File, dir: &cap_std::fs::Dir) -> Result<(), ArenaError> {
    use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::OpenOptionsExt;
    use std::os::unix::fs::MetadataExt;
    let a = file
        .metadata()
        .map_err(|_| invalid("whole-game held cwd metadata"))?;
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .read(true)
        .follow(FollowSymlinks::No)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC);
    let b = dir
        .open_with(".", &options)
        .map_err(|_| invalid("whole-game named cwd pin"))?
        .into_std()
        .metadata()
        .map_err(|_| invalid("whole-game named cwd metadata"))?;
    check(
        a.is_dir()
            && b.is_dir()
            && a.dev() == b.dev()
            && a.ino() == b.ino()
            && a.mode() & 0o077 == 0
            && a.uid() == b.uid(),
        "whole-game private cwd inode/permissions",
    )
}
#[cfg(target_os = "linux")]
fn open_relative(dir: &cap_std::fs::Dir, name: &str) -> Result<File, ArenaError> {
    use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::OpenOptionsExt;
    check(relative_name(name), "whole-game bounded relative held path")?;
    let mut parent = dir
        .try_clone()
        .map_err(|_| invalid("whole-game directory clone"))?;
    let parts: Vec<_> = name.split('/').collect();
    for part in &parts[..parts.len() - 1] {
        parent = parent
            .open_dir_nofollow(part)
            .map_err(|_| invalid("whole-game artifact parent pin"))?;
    }
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .read(true)
        .follow(FollowSymlinks::No)
        .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC);
    parent
        .open_with(parts[parts.len() - 1], &options)
        .map(|f| f.into_std())
        .map_err(|_| invalid("whole-game held regular file open"))
}

#[cfg(target_os = "linux")]
fn same_regular_file(held: &File, named: &File) -> Result<(), ArenaError> {
    use std::os::unix::fs::MetadataExt;
    let held = held
        .metadata()
        .map_err(|_| invalid("whole-game original held artifact metadata"))?;
    let named = named
        .metadata()
        .map_err(|_| invalid("whole-game named artifact metadata"))?;
    check(
        held.is_file()
            && named.is_file()
            && held.nlink() == 1
            && named.nlink() == 1
            && held.dev() == named.dev()
            && held.ino() == named.ino()
            && held.len() == named.len()
            && held.uid() == named.uid(),
        "whole-game named artifact differs from retained original inode",
    )
}

#[cfg(target_os = "linux")]
fn same_named_artifact(
    cwd: &cap_std::fs::Dir,
    file: &File,
    pin: &GameFilePin,
) -> Result<(), ArenaError> {
    let named = open_relative(cwd, &pin.file)?;
    same_regular_file(file, &named)
}

#[cfg(target_os = "linux")]
fn same_named_binary(launch: &GenerationGameProcessLaunchV3) -> Result<(), ArenaError> {
    let parent = launch
        .binary_path
        .parent()
        .ok_or_else(|| invalid("whole-game original binary parent absent"))?;
    let name = launch
        .binary_path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| invalid("whole-game original binary filename invalid"))?;
    let parent = open_directory(parent)?;
    let named = open_relative(&parent, name)?;
    same_regular_file(&launch.binary, &named)
}

#[cfg(target_os = "linux")]
fn reverify_owned_artifacts(
    owner: &GameProcessOwner,
    until: Instant,
    cancel: Option<&AtomicBool>,
    max_read_bytes: u64,
) -> Result<u64, ArenaError> {
    let launch = &owner.launch;
    // A series may have more remaining credit than one original game. Clamp
    // this read-only pass to both ceilings and charge only its actual reads.
    let max_read_bytes = max_read_bytes.min(launch.raw_limits.max_read_bytes);
    check(
        max_read_bytes > 0,
        "whole-game publication read credit exhausted",
    )?;
    let mut credit = ReadCredit {
        remaining: max_read_bytes,
        until,
        cancel,
    };
    credit.take(0)?;
    let process = owner
        .process
        .as_ref()
        .ok_or_else(|| invalid("whole-game publication lacks original process owner"))?;
    check(
        process.transport_complete(),
        "whole-game publication requires previously joined original transport",
    )?;
    // New publication work does not touch or replace any original deadline.
    let cwd = open_directory(&launch.cwd_path)?;
    same_directory(&launch.cwd, &cwd)?;

    // Two endpoints can declare the same private name. Check every retained
    // descriptor against that name, then hash its bytes once for this pass.
    let mut unique: BTreeMap<&str, (&File, GameFilePin)> = BTreeMap::new();
    for artifact in &owner.held {
        credit.take(0)?;
        same_named_artifact(&cwd, &artifact.file, &artifact.pin)?;
        match unique.get(artifact.pin.file.as_str()) {
            Some((prior, pin)) => {
                check(
                    pin.sha256 == artifact.pin.sha256 && pin.bytes == artifact.pin.bytes,
                    "whole-game conflicting original raw path pins",
                )?;
                same_regular_file(prior, &artifact.file)?;
            }
            None => {
                unique.insert(
                    artifact.pin.file.as_str(),
                    (&artifact.file, artifact.pin.clone()),
                );
            }
        }
    }
    for pinned in &launch.pinned {
        credit.take(0)?;
        check(
            pinned.private_cwd() == launch.cwd_path,
            "whole-game publication startup cwd differs from original",
        )?;
        for asset in pinned.assets() {
            credit.take(0)?;
            let name = asset.declaration.snapshot_path.as_str();
            let pin = GameFilePin {
                file: name.to_owned(),
                sha256: asset.declaration.artifact.sha256.clone(),
                bytes: asset.declaration.artifact.bytes,
            };
            same_named_artifact(&cwd, &asset.file, &pin)?;
            match unique.get(name) {
                Some((prior, prior_pin)) => {
                    check(
                        prior_pin.sha256 == pin.sha256 && prior_pin.bytes == pin.bytes,
                        "whole-game conflicting original startup path pins",
                    )?;
                    same_regular_file(prior, &asset.file)?;
                }
                None => {
                    unique.insert(name, (&asset.file, pin));
                }
            }
        }
    }
    let owned_bytes = unique.values().try_fold(0u64, |total, (_, pin)| {
        total
            .checked_add(pin.bytes)
            .ok_or_else(|| invalid("whole-game publication owned artifact bytes overflow"))
    })?;
    check(
        owned_bytes <= GENERATION_OUTPUT_CAP && owned_bytes <= launch.watch.max_total_bytes,
        "whole-game publication retained artifacts exceed original staging/output policy",
    )?;
    same_named_binary(launch)?;
    verify_held(
        &launch.binary,
        &GameFilePin {
            file: launch.binary_pin.path.clone(),
            sha256: launch.binary_pin.sha256.clone(),
            bytes: launch.binary_pin.bytes,
        },
        false,
        &mut credit,
    )?;
    same_named_binary(launch)?;
    for (file, pin) in unique.values() {
        same_named_artifact(&cwd, file, pin)?;
        verify_held(file, pin, false, &mut credit)?;
        same_named_artifact(&cwd, file, pin)?;
    }
    same_directory(&launch.cwd, &open_directory(&launch.cwd_path)?)?;
    credit.take(0)?;
    Ok(max_read_bytes - credit.remaining)
}

#[cfg(not(target_os = "linux"))]
fn reverify_owned_artifacts(
    _owner: &GameProcessOwner,
    _until: Instant,
    _cancel: Option<&AtomicBool>,
    _max_read_bytes: u64,
) -> Result<u64, ArenaError> {
    Err(invalid(
        "whole-game original archive custody requires Linux pinned files",
    ))
}

#[cfg(target_os = "linux")]
fn verify_held(
    file: &File,
    pin: &GameFilePin,
    retain: bool,
    credit: &mut ReadCredit<'_>,
) -> Result<Vec<u8>, ArenaError> {
    use std::os::unix::fs::{FileExt, MetadataExt};
    let before = file
        .metadata()
        .map_err(|_| invalid("whole-game file metadata"))?;
    check(
        before.is_file()
            && before.nlink() == 1
            && before.len() == pin.bytes
            && pin.bytes > 0
            && generation_sha(&pin.sha256),
        "whole-game exact single-link regular held file",
    )?;
    let mut bytes = if retain {
        Vec::with_capacity(
            usize::try_from(pin.bytes).map_err(|_| invalid("whole-game read allocation"))?,
        )
    } else {
        vec![]
    };
    let mut hash = Sha256::new();
    let mut offset = 0;
    let mut block = [0u8; 64 * 1024];
    while offset < pin.bytes {
        let expected = (pin.bytes - offset).min(block.len() as u64) as usize;
        credit.take(expected as u64)?;
        let read = file
            .read_at(&mut block[..expected], offset)
            .map_err(|_| invalid("whole-game pinned read"))?;
        check(read > 0, "whole-game held file truncated")?;
        hash.update(&block[..read]);
        if retain {
            bytes.extend_from_slice(&block[..read]);
        }
        offset += read as u64;
    }
    let after = file
        .metadata()
        .map_err(|_| invalid("whole-game readback metadata"))?;
    check(
        before.dev() == after.dev()
            && before.ino() == after.ino()
            && before.len() == after.len()
            && before.mtime() == after.mtime()
            && before.mtime_nsec() == after.mtime_nsec()
            && before.ctime() == after.ctime()
            && before.ctime_nsec() == after.ctime_nsec()
            && format!("{:x}", hash.finalize()) == pin.sha256,
        "whole-game pinned content changed",
    )?;
    credit.take(0)?;
    Ok(bytes)
}
#[cfg(target_os = "linux")]
fn read_unpinned_control(file: &File, bytes: u64, until: Instant) -> Result<Vec<u8>, ArenaError> {
    use std::os::unix::fs::{FileExt, MetadataExt};
    let meta = file
        .metadata()
        .map_err(|_| invalid("whole-game control metadata"))?;
    check(
        bytes > 0
            && bytes <= MAX_OUTPUT_BYTES
            && meta.is_file()
            && meta.nlink() == 1
            && meta.len() == bytes,
        "whole-game raw control regular file bound",
    )?;
    let mut result = vec![0; bytes as usize];
    let mut offset = 0usize;
    while offset < result.len() {
        check(
            Instant::now() < until,
            "whole-game original control read deadline",
        )?;
        let end = (offset + 64 * 1024).min(result.len());
        let read = file
            .read_at(&mut result[offset..end], offset as u64)
            .map_err(|_| invalid("whole-game raw control read"))?;
        check(read > 0, "whole-game control file truncated")?;
        offset += read;
    }
    let pin = GameFilePin {
        file: "held-control".into(),
        sha256: format!("{:x}", Sha256::digest(&result)),
        bytes,
    };
    let mut credit = ReadCredit {
        remaining: bytes,
        until,
        cancel: None,
    };
    verify_held(file, &pin, false, &mut credit)?;
    Ok(result)
}

#[cfg(target_os = "linux")]
type VerifiedGameRaw = (
    BTreeMap<String, GenerationRecordV3>,
    BTreeMap<String, serde_json::Value>,
    Option<GenerationTrajectoryAuthorityV3>,
    VerifiedGenerationGameSummaryV3,
);

#[cfg(target_os = "linux")]
fn verify_raw(
    request: &GenerationWholeGameRequestV3,
    output: &ChildOutput,
    manifest: &GenerationChunkManifestV3,
    cwd: &cap_std::fs::Dir,
    held: &mut Vec<HeldGameArtifact>,
    credit: &mut ReadCredit<'_>,
    limits: GenerationGameRawLimitsV3,
) -> Result<VerifiedGameRaw, ArenaError> {
    let mut records = BTreeMap::new();
    let mut evidence = BTreeMap::new();
    let mut inputs = BTreeSet::new();
    let mut trajectories = vec![];
    let mut retained_bytes = 0u64;
    check(
        manifest.total_bytes <= request.plan.maximum_output_bytes
            && manifest.opening_splits == request.plan.opening_splits(),
        "whole-game archive/split pin",
    )?;
    for pin in &manifest.chunks {
        credit.take(0)?;
        let owned_pin = GameFilePin {
            file: format!("{}/{}", request.archive_directory, pin.path),
            sha256: pin.sha256.clone(),
            bytes: pin.bytes,
        };
        let file = open_relative(cwd, &owned_pin.file)?;
        let bytes = verify_held(&file, &owned_pin, true, credit)?;
        held.push(HeldGameArtifact {
            file,
            pin: owned_pin,
        });
        check(bytes.ends_with(b"\n"), "whole-game JSONL trailing boundary")?;
        let rows: Vec<_> = bytes[..bytes.len() - 1].split(|b| *b == b'\n').collect();
        check(
            rows.len() == pin.rows as usize,
            "whole-game raw chunk row accounting",
        )?;
        for bytes in rows {
            credit.take(0)?;
            check(
                !bytes.is_empty() && bytes.len() <= GENERATION_MAX_ROW_BYTES,
                "whole-game raw row bound",
            )?;
            retained_bytes = retained_bytes
                .checked_add(bytes.len() as u64)
                .ok_or_else(|| invalid("whole-game retained bytes overflow"))?;
            check(
                retained_bytes <= limits.max_retained_bytes,
                "whole-game declared raw retention bound",
            )?;
            let row: serde_json::Value = serde_json::from_slice(bytes)
                .map_err(|e| invalid(format!("whole-game raw row: {e}")))?;
            let mut canonical = row.clone();
            canonical.sort_all_objects();
            check(
                serde_json::to_vec(&canonical).map_err(|e| invalid(e.to_string()))? == bytes,
                "whole-game canonical raw row bytes",
            )?;
            match pin.kind {
                GenerationChunkKindV3::Inputs => {
                    check(
                        row.as_object().is_some_and(|r| r.len() == 1),
                        "whole-game input row exact schema",
                    )?;
                    let seal = row["input_sha256"]
                        .as_str()
                        .filter(|s| generation_sha(s))
                        .ok_or_else(|| invalid("whole-game input seal"))?;
                    check(
                        inputs.len() < limits.max_records && inputs.insert(seal.to_owned()),
                        "whole-game duplicate/raw input bound",
                    )?;
                }
                GenerationChunkKindV3::Records => {
                    let record: GenerationRecordV3 = serde_json::from_value(row)
                        .map_err(|e| invalid(format!("whole-game raw record: {e}")))?;
                    record.validate()?;
                    check(
                        records.len() < limits.max_records
                            && record.input.game_id == output.game_id
                            && record.input.opening_id == output.opening_id
                            && record.input.model_epoch == request.model_epoch
                            && record.input.context_epoch == request.context_epoch
                            && records
                                .insert(record.input_sha256.clone(), record)
                                .is_none(),
                        "whole-game exact raw record owner/uniqueness",
                    )?;
                }
                GenerationChunkKindV3::Evidence => {
                    check(
                        row.as_object().is_some_and(|r| r.len() == 2),
                        "whole-game evidence row exact schema",
                    )?;
                    let sha = row["sha256"]
                        .as_str()
                        .ok_or_else(|| invalid("whole-game evidence digest"))?;
                    let raw = row["receipt"].clone();
                    let schema = raw["schema"]
                        .as_str()
                        .ok_or_else(|| invalid("whole-game evidence domain"))?;
                    check(
                        generation_digest(schema, &raw)? == sha
                            && evidence.len() < limits.max_evidence
                            && evidence.insert(sha.to_owned(), raw).is_none(),
                        "whole-game duplicate/raw evidence seal or bound",
                    )?;
                }
                GenerationChunkKindV3::Trajectories => {
                    check(
                        row.as_object().is_some_and(|r| r.len() == 2),
                        "whole-game trajectory row exact schema",
                    )?;
                    let trajectory: GenerationTrajectoryAuthorityV3 =
                        serde_json::from_value(row["trajectory"].clone())
                            .map_err(|e| invalid(format!("whole-game trajectory decode: {e}")))?;
                    check(
                        row["sha256"] == trajectory.sha256()? && trajectories.is_empty(),
                        "whole-game trajectory seal/uniqueness",
                    )?;
                    trajectories.push(trajectory);
                }
            }
        }
    }
    check(
        inputs == records.keys().cloned().collect()
            && records.len() == output.recorded_inputs
            && manifest.total_records == records.len() as u64,
        "whole-game actual record/seal exact membership",
    )?;
    let mut header = manifest.producer.clone();
    let fields = header
        .as_object_mut()
        .ok_or_else(|| invalid("whole-game inline producer"))?;
    fields.insert(
        "schema".into(),
        serde_json::json!("rz-pals-actual-collection-authority/1"),
    );
    fields.insert("input_seals".into(), serde_json::json!(inputs));
    let actual: GenerationProducerAuthorityV3 = serde_json::from_value(header)
        .map_err(|e| invalid(format!("whole-game recovered producer: {e}")))?;
    actual.validate(&request.plan.opening_splits())?;
    let mut expected = request.producer.clone();
    expected.input_seals = records.keys().cloned().collect();
    expected.actual_game_ids = vec![output.game_id.clone()];
    check(
        actual == expected,
        "whole-game original producer/source/model/runtime/teacher/inputs pin",
    )?;
    let mut evidence_by_input: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (sha, raw) in &evidence {
        credit.take(0)?;
        if let Some(input) = raw["input_sha256"].as_str() {
            let group = evidence_by_input.entry(input).or_default();
            check(
                group.len() < 4096,
                "whole-game per-input raw evidence work bound",
            )?;
            group.push(sha);
        }
    }
    let (trajectory, summary) =
        verify_replay(request, output, &evidence, trajectories.pop(), credit)?;
    let trajectory_sha = trajectory.as_ref().map(|t| t.sha256()).transpose()?;
    for record in records.values() {
        credit.take(0)?;
        let subset = record_evidence(record, &evidence, &evidence_by_input, credit)?;
        validate_generation_head_evidence(record, &subset)?;
        credit.take(0)?;
        verify_native_record(
            record,
            &subset,
            output.session_epochs,
            &output.native_readbacks,
        )?;
        credit.take(0)?;
        if let Some(wdl) = &record.future_targets.wdl {
            let t = trajectory
                .as_ref()
                .ok_or_else(|| invalid("whole-game WDL lacks actual Rules termination"))?;
            check(
                wdl.game_id == t.game_id
                    && wdl.ending == t.ending
                    && wdl.outcome == t.outcome
                    && Some(&wdl.trajectory_membership_sha256) == trajectory_sha.as_ref()
                    && wdl.rules_result_sha256 == t.result_evidence_sha256
                    && t.anchors.contains(&wdl.anchor),
                "whole-game WDL exact replay membership/outcome",
            )?;
        }
    }
    Ok((records, evidence, trajectory, summary))
}

/// Each raw receipt is indexed once. Preserve every matching binding/group so
/// the existing validators still see duplicate evidence, then join only their
/// explicit receipt references. Model/history/backend hashes are comparison
/// values and are not evidence-map edges.
#[cfg(target_os = "linux")]
fn record_evidence(
    record: &GenerationRecordV3,
    evidence: &BTreeMap<String, serde_json::Value>,
    by_input: &BTreeMap<&str, Vec<&str>>,
    credit: &mut ReadCredit<'_>,
) -> Result<BTreeMap<String, serde_json::Value>, ArenaError> {
    fn references(value: &serde_json::Value, pending: &mut Vec<String>) {
        match value {
            serde_json::Value::Object(fields) => {
                for (key, value) in fields {
                    if matches!(
                        key.as_str(),
                        "raw_evidence_sha256"
                            | "causal_receipt_sha256"
                            | "teacher_submission_sha256"
                            | "teacher_completion_sha256"
                            | "action_authority_sha256"
                    ) {
                        if let Some(sha) = value.as_str() {
                            pending.push(sha.to_owned());
                        }
                    } else {
                        references(value, pending);
                    }
                }
            }
            serde_json::Value::Array(values) => {
                for value in values {
                    references(value, pending);
                }
            }
            _ => {}
        }
    }
    let mut pending: Vec<String> = by_input
        .get(record.input_sha256.as_str())
        .into_iter()
        .flatten()
        .map(|sha| (*sha).to_owned())
        .collect();
    references(
        &serde_json::to_value(&record.future_targets).map_err(|e| invalid(e.to_string()))?,
        &mut pending,
    );
    let mut subset = BTreeMap::new();
    let mut bytes = 0u64;
    while let Some(sha) = pending.pop() {
        credit.take(0)?;
        if subset.contains_key(&sha) {
            continue;
        }
        check(
            subset.len() < 4096 && generation_sha(&sha),
            "whole-game per-input evidence reference work bound",
        )?;
        let raw = evidence
            .get(&sha)
            .ok_or_else(|| invalid("whole-game unrecovered input receipt reference"))?;
        let size = serde_json::to_vec(raw)
            .map_err(|e| invalid(e.to_string()))?
            .len() as u64;
        bytes = bytes
            .checked_add(size)
            .ok_or_else(|| invalid("whole-game evidence subset bytes overflow"))?;
        check(
            bytes <= 8 * 1024 * 1024,
            "whole-game per-input evidence working-byte bound",
        )?;
        references(raw, &mut pending);
        subset.insert(sha, raw.clone());
    }
    Ok(subset)
}

#[cfg(target_os = "linux")]
fn verify_replay(
    request: &GenerationWholeGameRequestV3,
    output: &ChildOutput,
    evidence: &BTreeMap<String, serde_json::Value>,
    stored: Option<GenerationTrajectoryAuthorityV3>,
    credit: &mut ReadCredit<'_>,
) -> Result<
    (
        Option<GenerationTrajectoryAuthorityV3>,
        VerifiedGenerationGameSummaryV3,
    ),
    ArenaError,
> {
    credit.take(0)?;
    let opening = request
        .plan
        .openings
        .iter()
        .find(|o| o.id == output.opening_id)
        .ok_or_else(|| invalid("whole-game assigned opening absent"))?;
    let mut initial = match &opening.initial_fen {
        Some(fen) => Position::from_fen(fen).map_err(|e| invalid(e.to_string()))?,
        None => Position::startpos(),
    };
    for movement in &opening.moves {
        credit.take(0)?;
        initial
            .make_move(BoardMove::from_uci(movement).map_err(|e| invalid(e.to_string()))?)
            .map_err(|e| invalid(e.to_string()))?;
    }
    let (white, black) = if output.engine_a_white {
        (&output.engine_names[0], &output.engine_names[1])
    } else {
        (&output.engine_names[1], &output.engine_names[0])
    };
    let mut pgn = CollectionGamePgnV3::begin(
        output.game_id.clone(),
        white.clone(),
        black.clone(),
        &initial,
    )?;
    let mut replay = ActualGameTrajectoryV3::begin(output.game_id.clone(), initial)?;
    for (index, decision) in output.decisions.iter().enumerate() {
        credit.take(0)?;
        let packed: Move16 = serde_json::from_value(serde_json::json!(decision.movement))
            .map_err(|e| invalid(e.to_string()))?;
        let movement = packed.unpack().map_err(|e| invalid(e.to_string()))?;
        let before = replay.position().clone();
        match pgn.push_actual_move(&before, movement, Duration::from_nanos(decision.elapsed_ns)) {
            Ok(()) => replay.play_pals_move(movement)?,
            Err(error) => check(
                index + 1 == output.decisions.len()
                    && pgn.observed_failure() == Some(output.termination),
                &format!(
                    "whole-game replay rejected decision without exact terminal failure: {error}"
                ),
            )?,
        }
    }
    let computed = match output.termination {
        CollectionPgnTerminationV3::RulesAutomatic => Some(replay.terminate_rules()?),
        CollectionPgnTerminationV3::RulesClaim(reason) => Some(replay.claim_rules_draw(reason)?),
        CollectionPgnTerminationV3::PlyLimit => {
            check(
                output.decisions.len() == usize::from(request.plan.max_plies),
                "whole-game actual ply-limit trace",
            )?;
            None
        }
        CollectionPgnTerminationV3::TimeForfeit { .. }
        | CollectionPgnTerminationV3::IllegalMove { .. } => {
            check(
                pgn.observed_failure() == Some(output.termination),
                "whole-game actual forfeit trace",
            )?;
            None
        }
        _ => {
            return Err(invalid(
                "whole-game infrastructure/crash/cancel cannot issue completed child authority",
            ));
        }
    };
    credit.take(0)?;
    let result = if let Some((trajectory, raw)) = &computed {
        check(
            stored.as_ref() == Some(trajectory)
                && evidence.get(&trajectory.result_evidence_sha256) == Some(raw),
            "whole-game independently replayed Rules trajectory/result differs",
        )?;
        match trajectory.outcome {
            GenerationOutcomeV3::WhiteWin => CollectionPgnResultV3::WhiteWin,
            GenerationOutcomeV3::BlackWin => CollectionPgnResultV3::BlackWin,
            GenerationOutcomeV3::Draw => CollectionPgnResultV3::Draw,
        }
    } else {
        check(
            stored.is_none(),
            "whole-game unfinished/forfeit cannot carry Rules trajectory",
        )?;
        match output.termination {
            CollectionPgnTerminationV3::TimeForfeit { loser }
            | CollectionPgnTerminationV3::IllegalMove { loser } => {
                if loser == Color::White {
                    CollectionPgnResultV3::BlackWin
                } else {
                    CollectionPgnResultV3::WhiteWin
                }
            }
            _ => CollectionPgnResultV3::Incomplete,
        }
    };
    let clock = pgn.clock_audit()?;
    let text = pgn.finish(result, output.termination)?;
    let raw = serde_json::json!({"schema":"rz-pals-generation-game-pgn-evidence/1","game_id":output.game_id,
        "opening_id":output.opening_id,"engine_a_white":output.engine_a_white,"pgn":text,"clock_audit":clock});
    check(
        output.rules_terminated == computed.is_some()
            && generation_digest("rz-pals-generation-game-pgn-evidence/1", &raw)?
                == output.pgn_receipt_sha256
            && evidence.get(&output.pgn_receipt_sha256) == Some(&raw),
        "whole-game exact Rules/clock/SAN PGN readback differs",
    )?;
    credit.take(0)?;
    Ok((
        computed.map(|(trajectory, _)| trajectory),
        VerifiedGenerationGameSummaryV3 {
            pgn_evidence: raw,
            pgn_receipt_sha256: output.pgn_receipt_sha256.clone(),
            result,
            termination: output.termination,
            rules_terminated: output.rules_terminated,
            engine_names: output.engine_names.clone(),
            decision_count: output.decisions.len(),
        },
    ))
}

#[cfg(target_os = "linux")]
fn verify_native_record(
    record: &GenerationRecordV3,
    evidence: &BTreeMap<String, serde_json::Value>,
    epochs: [u64; 2],
    loaded: &[NativeLoadedReadback; 2],
) -> Result<(), ArenaError> {
    let mut submissions = evidence.iter().filter(|(_, raw)| {
        raw["schema"] == "rz-pals-generation-submission-evidence/1"
            && raw["input_sha256"] == record.input_sha256
    });
    let (submitted_sha, submitted) = submissions
        .next()
        .ok_or_else(|| invalid("whole-game missing original native submission"))?;
    check(
        submissions.next().is_none(),
        "whole-game duplicate native input submission",
    )?;
    let epoch = submitted["ticket"]["epoch"]
        .as_u64()
        .ok_or_else(|| invalid("whole-game native ticket epoch"))?;
    let native_owner = epochs
        .iter()
        .position(|candidate| *candidate == epoch)
        .and_then(|index| loaded.get(index))
        .ok_or_else(|| invalid("whole-game native epoch absent from original A/B closure"))?;
    let native_input = decode_native_input(&record.input)?;
    let model = submitted["native_model_epoch_sha256"]
        .as_str()
        .ok_or_else(|| invalid("whole-game native model epoch absent"))?;
    let backend = submitted["native_backend_sha256"]
        .as_str()
        .ok_or_else(|| invalid("whole-game native backend absent"))?;
    let histories = submitted["native_anchor_history_sha256"]
        .as_array()
        .ok_or_else(|| invalid("whole-game original native anchor histories absent"))?
        .iter()
        .map(|s| {
            s.as_str()
                .ok_or_else(|| invalid("whole-game native history seal type"))
                .and_then(parse_native_digest)
        })
        .collect::<Result<Vec<_>, _>>()?;
    check(
        histories.len() == native_input.anchors.len() && !histories.is_empty(),
        "whole-game original native anchor history membership",
    )?;
    let identity = ReviewExecutionIdentity {
        model_epoch: parse_native_digest(model)?,
        backend: parse_native_digest(backend)?,
        rules_history: histories[0],
        anchor_history_seals: histories,
        fp32: true,
    };
    let native_digest = native_input
        .prepare()
        .map_err(|e| invalid(e.to_string()))?
        .input_digest(native_input.role);
    let fresh = native_input
        .fresh_key(&identity)
        .map_err(|e| invalid(e.to_string()))?;
    check(
        epochs.contains(&epoch)
            && native_owner.model_epoch_sha256 == model
            && native_owner.backend_sha256 == backend
            && submitted["native_input_sha256"] == hex(&native_digest)
            && submitted["ticket"]["representation_input_sha256"] == hex(&fresh.0)
            && submitted["ticket"]["representation_model_sha256"] == model
            && submitted["ticket"]["representation_encoding_sha256"]
                == hex(&Sha256::digest(
                    rz_eval::pals_model::virtual_review::ENCODING.as_bytes(),
                ))
            && submitted["native_endpoint_history_sha256"]
                == submitted["ticket"]["representation_history_sha256"]
            && record.input.worker_id == format!("native-owner-{epoch}")
            && submitted["game_id"] == record.input.game_id
            && submitted["opening_id"] == record.input.opening_id
            && submitted["generation_view_sha256"] == record.input.view_sha256
            && submitted["query_anchor"]
                == serde_json::to_value(&record.input.rules_anchor)
                    .map_err(|e| invalid(e.to_string()))?
            && submitted["capture_sequence"].as_u64() == Some(record.input.capture_sequence),
        "whole-game original native owner/input binding",
    )?;
    let mut completions = evidence.iter().filter(|(_, raw)| {
        raw["schema"] == "rz-pals-generation-native-completion-evidence/1"
            && raw["input_sha256"] == record.input_sha256
            && raw["submission_sha256"] == *submitted_sha
    });
    let (completed_sha, completed) = completions
        .next()
        .ok_or_else(|| invalid("whole-game missing actual native completion"))?;
    check(
        completions.next().is_none()
            && completed["ticket"] == submitted["ticket"]
            && completed["physical_completion"] == "confirmed"
            && completed["admissible_actual_nn"] == true
            && completed["completed_nn_inputs"]
                .as_u64()
                .is_some_and(|n| n > 0)
            && completed["measurement_complete"] == true
            && completed["execution"]["epoch"].as_u64() == Some(epoch)
            && completed["execution"]["sequence"]
                .as_u64()
                .is_some_and(|n| n > 0),
        "whole-game native physical execution/owner evidence",
    )?;
    let mut retirements = evidence.values().filter(|raw| {
        raw["schema"] == "rz-pals-generation-native-retirement-evidence/1"
            && raw["input_sha256"] == record.input_sha256
            && raw["submission_sha256"] == *submitted_sha
            && raw["completion_sha256"] == *completed_sha
    });
    let retired = retirements
        .next()
        .ok_or_else(|| invalid("whole-game missing actual native retirement"))?;
    check(
        retirements.next().is_none()
            && retired["ticket"] == submitted["ticket"]
            && retired["accepted"] == true,
        "whole-game exact accepted physical native retirement",
    )
}

#[cfg(target_os = "linux")]
fn parse_native_digest(sha: &str) -> Result<[u8; 32], ArenaError> {
    rz_eval::asset::parse_sha256(sha).map_err(|_| invalid("whole-game native digest syntax"))
}

/// Reverse only the public array-token transport ABI, then use the model's
/// actual typed validation/preparation/digest. Roundtrip rejects a permissive
/// projection or a second interpretation of the retained JSON input.
#[cfg(target_os = "linux")]
fn decode_native_input(input: &GenerationInputV3) -> Result<PalsVirtualReviewInput, ArenaError> {
    input.validate()?;
    fn tokens(value: &mut serde_json::Value) -> Result<(), ArenaError> {
        let values = value
            .as_array_mut()
            .ok_or_else(|| invalid("whole-game native token line type"))?;
        for value in values {
            let token: [u8; 3] = serde_json::from_value(value.clone())
                .map_err(|_| invalid("whole-game exact three-component native token"))?;
            *value = serde_json::json!({"from":token[0],"to":token[1],"promotion":token[2]});
        }
        Ok(())
    }
    let mut model = input.encoded_model_input.clone();
    for record in model["records"]
        .as_array_mut()
        .ok_or_else(|| invalid("whole-game native records type"))?
    {
        tokens(&mut record["line"])?;
    }
    for field in ["prefix", "proposal", "counter"] {
        tokens(&mut model["query"][field])?;
    }
    tokens(&mut model["candidates"])?;
    let model: PalsVirtualReviewInput = serde_json::from_value(model)
        .map_err(|e| invalid(format!("whole-game typed native input: {e}")))?;
    model.validate().map_err(|e| invalid(e.to_string()))?;
    check(
        virtual_review_json_v3(&model)? == input.encoded_model_input,
        "whole-game original native array ABI roundtrip",
    )?;
    Ok(model)
}

#[cfg(test)]
mod archive_pin_tests {
    use super::*;

    #[test]
    fn compact_seal_keeps_all_bytes_and_canonical_order() {
        let expected = std::array::from_fn::<_, 32, _>(|i| (i as u8).wrapping_mul(7));
        assert_eq!(verified_sha_bytes(&hex(&expected)), expected);
        assert_eq!(verified_sha_bytes(&"0".repeat(64)), [0; 32]);
        assert_eq!(verified_sha_bytes(&"f".repeat(64)), [255; 32]);
    }

    #[cfg(target_os = "linux")]
    struct OwnedTestDirectory(PathBuf);
    #[cfg(target_os = "linux")]
    impl OwnedTestDirectory {
        fn new() -> Self {
            use std::os::unix::fs::DirBuilderExt;
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "rz-game-archive-pin-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let mut builder = std::fs::DirBuilder::new();
            builder.mode(0o700).create(&path).unwrap();
            Self(path)
        }
    }
    #[cfg(target_os = "linux")]
    impl Drop for OwnedTestDirectory {
        fn drop(&mut self) {
            // These fixed files were created only by the two bounded tests.
            // Never recurse or remove an unverified parent directory.
            for name in ["artifact.jsonl", "retained.jsonl"] {
                let _ = std::fs::remove_file(self.0.join(name));
            }
            let _ = std::fs::remove_dir(&self.0);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn same_hash_at_replaced_name_does_not_replace_original_custody() {
        let owned = OwnedTestDirectory::new();
        std::fs::write(owned.0.join("artifact.jsonl"), b"actual-row\n").unwrap();
        let dir = open_directory(&owned.0).unwrap();
        let held = open_relative(&dir, "artifact.jsonl").unwrap();
        let pin = GameFilePin {
            file: "artifact.jsonl".into(),
            sha256: format!("{:x}", Sha256::digest(b"actual-row\n")),
            bytes: 11,
        };
        same_named_artifact(&dir, &held, &pin).unwrap();
        std::fs::rename(
            owned.0.join("artifact.jsonl"),
            owned.0.join("retained.jsonl"),
        )
        .unwrap();
        std::fs::write(owned.0.join("artifact.jsonl"), b"actual-row\n").unwrap();
        assert!(same_named_artifact(&dir, &held, &pin).is_err());
        let mut credit = ReadCredit {
            remaining: 11,
            until: Instant::now() + Duration::from_secs(1),
            cancel: None,
        };
        // The original raw bytes remain available for diagnosing the failure.
        verify_held(&held, &pin, false, &mut credit).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn publication_read_checks_credit_cancel_and_content_without_cursor_changes() {
        use std::io::{Seek, SeekFrom};
        let owned = OwnedTestDirectory::new();
        std::fs::write(owned.0.join("artifact.jsonl"), b"actual-row\n").unwrap();
        let dir = open_directory(&owned.0).unwrap();
        let mut held = open_relative(&dir, "artifact.jsonl").unwrap();
        held.seek(SeekFrom::Start(3)).unwrap();
        let pin = GameFilePin {
            file: "artifact.jsonl".into(),
            sha256: format!("{:x}", Sha256::digest(b"actual-row\n")),
            bytes: 11,
        };
        let deadline = Instant::now() + Duration::from_secs(1);
        let mut insufficient = ReadCredit {
            remaining: 10,
            until: deadline,
            cancel: None,
        };
        assert!(verify_held(&held, &pin, false, &mut insufficient).is_err());
        let cancelled = AtomicBool::new(true);
        let mut credit = ReadCredit {
            remaining: 11,
            until: deadline,
            cancel: Some(&cancelled),
        };
        assert!(verify_held(&held, &pin, false, &mut credit).is_err());
        cancelled.store(false, Ordering::Release);
        verify_held(&held, &pin, false, &mut credit).unwrap();
        assert_eq!(credit.remaining, 0);
        assert_eq!(held.stream_position().unwrap(), 3);
        std::fs::write(owned.0.join("artifact.jsonl"), b"changedrow\n").unwrap();
        credit.remaining = 11;
        assert!(verify_held(&held, &pin, false, &mut credit).is_err());
    }
}
