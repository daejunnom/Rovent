//! Rovent native startup, independent of the legacy V5 process/receipt domain.
//!
//! This lower constructor creates only the selected P/C model ticket owner.
//! A PCR or typed R-free product owner must subsequently consume it; loading a
//! model is not a complete engine, first inference, resource or numeric proof.

pub mod configuration;
pub mod external_session;
pub mod launch;
pub mod nnuec_preparing;
pub mod nnuec_session;
pub use nnuec_session::{
    RoventNnuecCollectionAdmissionFailureV2, RoventNnuecCollectionAdmissionPortV2,
    RoventNnuecCollectionHandoffV2, RoventNnuecCollectionHooksV2,
    RoventNnuecCollectionObserverRejectionV2,
};
pub mod nnuec_source;
pub mod pc_session;
mod pinned;
use pinned::NativeInputLock;
pub mod process_boundary;
pub mod product;
pub mod source_inventory_v3;
pub(crate) mod storage_lifecycle;

pub use pinned::{
    ExternalNativeInputLock, RoventCollectionPreloadSources, RoventCollectionTeacherSourceCustody,
    RoventCollectionTeacherSources, RoventExternalRAssetSelection, RoventExternalRSourceCustody,
    RoventExternalRSourceSelection, RoventPinnedExternalRSource, RoventTeacherAssetSelection,
    RoventTeacherSourceSelection,
};
pub use pinned::{
    RoventNativeAssetSelection, RoventNativeGraphAsset, RoventPinnedNativeSources,
    RoventPinnedStartupV2,
};

use crate::pals_adaptive::{NativeQuestionProjection, ProjectedNativeBackend};
use crate::pals_native::ticketed::{
    NativeVirtualReviewTicketBackend, TicketNativeBackend, TicketNativeEvent, TicketServiceLimits,
    TicketedNativeService, VirtualReviewCollectionObserver,
};
use rz_contracts::rovent::ExecutionMode;
use rz_contracts::{Digest, PrecisionProfile};
use rz_eval::asset::parse_sha256;
use rz_eval::error::{BackendError, FailureKind as K, FailureStage as S};
use rz_eval::onnx::{NativeLoadingProfile, OrtRuntime, Provider};
use rz_eval::pals_model::virtual_review;
#[cfg(target_os = "linux")]
use rz_eval::pals_onnx::virtual_review::VirtualReviewStartupCustody;
use rz_eval::pals_onnx::virtual_review::{
    PalsVirtualReviewOnnxBackend, VirtualReviewArtifactKind, VirtualReviewExecutionMode,
    VirtualReviewOwnedSources, VirtualReviewProfileDirectory,
};
use rz_eval::pals_onnx::{
    PalsAppliedSessionSettings, PalsCudaControlPolicy, PalsOnnxConfig, PalsSessionResidency,
};
use rz_eval::runtime_pin::RuntimeLibraryPin;
#[cfg(target_os = "linux")]
use rz_experiments::NativeRuntimeSourceV6;
use rz_experiments::{
    RoventBackend, RoventCoordinatorDeclaration, RoventDevice, RoventEngineDeclaration,
    RoventInputLock, RoventInputLockV2, RoventInputLockV3, RoventPcConfiguration, RoventPurpose,
    RoventResearcher, SystemLibraryObservationV6,
};
use rz_runtime::contracts::{ContractClock, ContractSystemClock};
use rz_search::pals::coordinator::RoleFailure;
use rz_search::pals::engine::adaptive_port::{AdaptiveModelNamespace, AdaptivePortQuestion};
use rz_search::pals::review_context::ReviewView;
use rz_search::pals::value::ModelValueIdentity;
use sha2::{Digest as _, Sha256};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::time::{Duration, Instant};

pub const ROVENT_NATIVE_STARTUP_DOMAIN: &str = "rz-rovent-native-startup/1";
pub const ROVENT_NATIVE_CONFIGURATION_DOMAIN: &str = "rz-rovent-native-configuration/1";
pub const ROVENT_NATIVE_POLICY_HEAD: &str = "legal-candidates/1";
pub const ROVENT_NATIVE_VALUE_HEAD: &str = "side-to-move-wdl/1";

pub type RoventNativeBackend<Q = AdaptivePortQuestion> =
    ProjectedNativeBackend<Q, NativeVirtualReviewTicketBackend>;
pub type RoventNativeService<Q = AdaptivePortQuestion> =
    TicketedNativeService<ReviewView, Q, RoventNativeBackend<Q>>;

/// Topology selection is a declaration. It does not construct a dummy checker,
/// worker, CPU ledger or PausedStack for an R-free mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventStartupTopology {
    RolesOnly,
    OwnResearcher,
    ExternalResearcher,
}

/// Original bounded startup control. Clones retain the same cancel/deadline;
/// waiting for the physical constructor must never renew either one.
#[derive(Clone)]
pub struct RoventStartupControl {
    pub until: Instant,
    pub cleanup_until: Instant,
    pub cancellation: Arc<AtomicBool>,
}
impl RoventStartupControl {
    pub fn check(&self) -> Result<(), BackendError> {
        if self.cancellation.load(Ordering::Acquire) {
            Err(fail(K::BackendUnavailable, "Rovent startup was canceled"))
        } else if Instant::now() >= self.until {
            Err(fail(K::BackendUnavailable, "Rovent startup expired"))
        } else {
            Ok(())
        }
    }
}

// Optional constructor diagnostics only. No request, root, Ready, provider or
// physical-completion authority can be constructed from these observations.
#[derive(Clone, Copy)]
#[repr(usize)]
pub(crate) enum NativeStartupDiagnosticStage {
    ResourceAdmission,
    SourcePins,
    CudaProfileDirectory,
    RuntimePin,
    NnuecSource,
    NativePreparation,
    NativeFactory,
    CudaInventory,
    OrtRuntime,
    ModelSessions,
    NativeReady,
    ProductAssembly,
    UciServe,
    FactoryDrain,
    ProductDrain,
}
const NATIVE_STARTUP_DIAGNOSTIC_STAGE_COUNT: usize = 15;
const NATIVE_STARTUP_DIAGNOSTIC_STAGE_NAMES: [&str; NATIVE_STARTUP_DIAGNOSTIC_STAGE_COUNT] = [
    "resource_admission",
    "source_pins",
    "cuda_profile_directory",
    "runtime_pin",
    "nnuec_source",
    "native_preparation",
    "native_factory",
    "cuda_inventory",
    "ort_runtime",
    "model_sessions",
    "native_ready",
    "product_assembly",
    "uci_serve",
    "factory_drain",
    "product_drain",
];
const MAX_NATIVE_STARTUP_DIAGNOSTIC_EMISSIONS: u8 = 40;
#[derive(Clone, Copy)]
pub(crate) enum NativeStartupDiagnosticPhase {
    Entry,
    Exit,
    Error,
}
impl NativeStartupDiagnosticPhase {
    fn name(self) -> &'static str {
        match self {
            Self::Entry => "entry",
            Self::Exit => "exit",
            Self::Error => "error",
        }
    }
}
struct NativeStartupDiagnosticState {
    origin: Instant,
    emitted: std::sync::atomic::AtomicU8,
}
#[derive(Clone, Default)]
pub(crate) struct NativeStartupDiagnostic {
    state: Option<Arc<NativeStartupDiagnosticState>>,
}
impl NativeStartupDiagnostic {
    pub(crate) fn new(enabled: bool, origin: Instant) -> Self {
        Self {
            state: enabled.then(|| {
                Arc::new(NativeStartupDiagnosticState {
                    origin,
                    emitted: std::sync::atomic::AtomicU8::new(0),
                })
            }),
        }
    }
    // Reserved slots, not an acknowledgement that an OS write completed.
    fn reserve(&self) -> Option<u8> {
        self.state
            .as_ref()?
            .emitted
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                if n < MAX_NATIVE_STARTUP_DIAGNOSTIC_EMISSIONS {
                    Some(n + 1)
                } else {
                    None
                }
            })
            .ok()
            .map(|n| n + 1)
    }
    pub(crate) fn emit(
        &self,
        stage: NativeStartupDiagnosticStage,
        phase: NativeStartupDiagnosticPhase,
    ) {
        let Some(sequence) = self.reserve() else {
            return;
        };
        let state = self
            .state
            .as_ref()
            .expect("diagnostic reservation retains its state");
        let elapsed_ns = u64::try_from(state.origin.elapsed().as_nanos()).unwrap_or(u64::MAX);
        use std::io::Write;
        // Static closed tokens plus sequence/time only. No original error text,
        // paths, environment, model/tensor values or diagnostic objects.
        // A bounded row count is not a hard OS-I/O deadline: retain the caller's
        // original external supervisor and all existing startup/cleanup limits.
        let _ = writeln!(
            std::io::stderr().lock(),
            "rovent-startup-diagnostic/1 seq={sequence} stage={} phase={} elapsed_ns={elapsed_ns}",
            NATIVE_STARTUP_DIAGNOSTIC_STAGE_NAMES[stage as usize],
            phase.name()
        );
    }
    pub(crate) fn trace<T, E>(
        &self,
        stage: NativeStartupDiagnosticStage,
        f: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        if self.state.is_none() {
            return f();
        }
        self.emit(stage, NativeStartupDiagnosticPhase::Entry);
        let result = f();
        self.emit(
            stage,
            if result.is_ok() {
                NativeStartupDiagnosticPhase::Exit
            } else {
                NativeStartupDiagnosticPhase::Error
            },
        );
        result
    }
}
#[cfg(test)]
mod native_startup_diagnostic_tests {
    use super::*;
    #[test]
    fn diagnostic_default_is_off_and_cloned_budget_never_renews() {
        let off = NativeStartupDiagnostic::default();
        assert!(off.reserve().is_none());
        let on = NativeStartupDiagnostic::new(true, Instant::now());
        let same = on.clone();
        for n in 1..=MAX_NATIVE_STARTUP_DIAGNOSTIC_EMISSIONS {
            assert_eq!(
                if n % 2 == 0 {
                    same.reserve()
                } else {
                    on.reserve()
                },
                Some(n)
            );
        }
        assert!(on.reserve().is_none());
        assert!(same.reserve().is_none());
        // An exhausted diagnostic budget must neither skip work nor replace
        // the original error owner. Disabled diagnostics preserve it too.
        let calls = std::cell::Cell::new(0);
        let original = Arc::new(());
        for diagnostic in [&on, &off] {
            let rejected = diagnostic
                .trace(NativeStartupDiagnosticStage::NativeFactory, || {
                    calls.set(calls.get() + 1);
                    Err::<(), _>(Arc::clone(&original))
                })
                .unwrap_err();
            assert!(Arc::ptr_eq(&rejected, &original));
        }
        assert_eq!(calls.get(), 2);
        assert_eq!(
            NATIVE_STARTUP_DIAGNOSTIC_STAGE_NAMES.len(),
            NATIVE_STARTUP_DIAGNOSTIC_STAGE_COUNT
        );
    }
}

/// An actual verified runtime capability; it is consumed on the ticket owner.
/// CUDA control inventory is read from the same retained asset selection.
pub struct RoventNativeResources {
    pub runtime: RuntimeLibraryPin,
    pub configuration: PalsOnnxConfig,
    pub loading_profile: Option<NativeLoadingProfile>,
    pub cuda_profile: Option<VirtualReviewProfileDirectory>,
    pub drain_limit: Duration,
    pub tickets: TicketServiceLimits,
    pub command_capacity: usize,
}

/// System CUDA is selected by the retained V3 lock and is loaded on the
/// native owner thread.  Unlike the legacy input it contains no CUDA bundle or
/// per-run copy; the host installation remains the runtime owner.
pub struct RoventSystemNativeResourcesV3 {
    pub configuration: PalsOnnxConfig,
    pub cuda_profile: Option<VirtualReviewProfileDirectory>,
    pub drain_limit: Duration,
    pub tickets: TicketServiceLimits,
    pub command_capacity: usize,
}

/// Native resources selected by the original V3 runtime lock. CPU retains its
/// original ORT pin; system CUDA retains the system-runtime registration.
pub enum RoventNativeResourcesV3 {
    Cpu(RoventNativeResources),
    SystemCuda(RoventSystemNativeResourcesV3),
}

impl From<RoventSystemNativeResourcesV3> for RoventNativeResourcesV3 {
    fn from(resources: RoventSystemNativeResourcesV3) -> Self {
        Self::SystemCuda(resources)
    }
}

/// Session setup and actual mapping observations from the original worker.
/// This is not an inference result or proof of GPU physical completion.
#[derive(Clone, Debug)]
pub struct RoventSystemReadyObservationV3 {
    pub runtime_identity: Digest,
    pub libraries: Vec<SystemLibraryObservationV6>,
    /// None denotes unavailable mapping observation, not an empty list.
    pub unregistered_native_mappings: Option<Vec<String>>,
    /// True only after every actual model session accepted CUDA registration,
    /// reported the provider available, and completed session construction.
    /// This does not attest a GPU inference or its physical completion.
    pub provider_ready: bool,
    /// Original successful session-builder observations, not requested
    /// configuration or model execution evidence.
    pub session_settings: Arc<[PalsAppliedSessionSettings]>,
}

/// Observations from the original successful constructor, not first Run.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct RoventNativeStartupSnapshot {
    pub domain: &'static str,
    pub input_lock: Digest,
    pub process_epoch: u64,
    pub mode: ExecutionMode,
    pub topology: RoventStartupTopology,
    pub weights: Digest,
    pub export_manifest: Digest,
    pub backend: Digest,
    pub runtime_library: Digest,
    pub runtime_bundle: Option<Digest>,
    pub native_configuration: Digest,
    /// Populated only by the original V3 system-CUDA factory after sessions
    /// and mapping checks complete. Physical completion remains separate.
    pub system_runtime: Option<RoventSystemReadyObservationV3>,
    pub startup_configuration: Digest,
    pub search_configuration: Digest,
    pub validator_loaded: bool,
    pub physical_workers_created: usize,
    pub session_residency: PalsSessionResidency,
    /// The factory cannot authorize a numeric/GPU acceptance or an engine.
    pub neural_service_executed: bool,
}
impl RoventNativeStartupSnapshot {
    pub fn role_model_identity(&self) -> String {
        format!(
            "pals-native-virtual-v3:{}:{}:{}",
            hex(self.export_manifest.0),
            hex(self.weights.0),
            hex(self.backend.0)
        )
    }
    pub fn model_value_identity(&self) -> Result<ModelValueIdentity, BackendError> {
        let identity = ModelValueIdentity {
            semantics: rz_search::pals::value::MODEL_WDL_VALUE_SEMANTICS.into(),
            model: self.role_model_identity(),
            encoding: format!(
                "{}:{}",
                virtual_review::ENCODING,
                hex(virtual_review::registered_encoding_sha256())
            ),
            precision: "fp32_tf32_off".into(),
            model_epoch: self.weights.0,
        };
        identity
            .validate()
            .map_err(|_| fail(K::IdentityMismatch, "invalid loaded model namespace"))?;
        Ok(identity)
    }
    pub fn model_namespace(&self) -> Result<AdaptiveModelNamespace, BackendError> {
        AdaptiveModelNamespace::new(
            self.model_value_identity()?,
            self.weights,
            Digest(Sha256::digest(virtual_review::ENCODING.as_bytes()).into()),
            PrecisionProfile::Fp32,
            1,
        )
        .map_err(|_| fail(K::IdentityMismatch, "invalid loaded ticket namespace"))
    }
}

/// Physical sources are returned on failure. Dropping a failed startup is not
/// a positive finish/join receipt; a process owner must drain or quarantine it.
pub struct RoventNativeStartupRejection<Q: NativeQuestionProjection = AdaptivePortQuestion> {
    pub error: BackendError,
    pub native: Option<Box<RoventNativeService<Q>>>,
}

/// V3-only rejection that preserves original inputs when the factory could
/// not start. This is absence of entry for this attempt, not a statement
/// about earlier activity on the device or a physical GPU completion fence.
#[cfg(target_os = "linux")]
pub enum RoventNativeStartupRejectionV3<Q: NativeQuestionProjection = AdaptivePortQuestion> {
    Started(RoventNativeStartupRejection<Q>),
    Unstarted {
        error: BackendError,
        owner: RoventUnstartedNativeCustodyV3,
    },
}

#[cfg(target_os = "linux")]
impl<Q: NativeQuestionProjection> std::fmt::Debug for RoventNativeStartupRejectionV3<Q> {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Started(rejection) => output.debug_tuple("Started").field(rejection).finish(),
            Self::Unstarted { error, .. } => output
                .debug_struct("Unstarted")
                .field("error", error)
                .field("original_owner_retained", &true)
                .finish(),
        }
    }
}

/// Exact owner supplied to an unstarted V3 factory. It can hold a collection
/// observer as well as source/runtime capabilities; no replacement is made.
#[cfg(target_os = "linux")]
pub struct RoventUnstartedNativeCustodyV3 {
    original: Box<dyn std::any::Any + Send>,
}

#[cfg(target_os = "linux")]
pub struct RoventPreAdmissionCustodyV3 {
    sources: Arc<RoventPinnedNativeSources<RoventInputLockV3>>,
    resources: RoventNativeResourcesV3,
    observer: Option<Box<dyn VirtualReviewCollectionObserver>>,
}

#[cfg(target_os = "linux")]
impl RoventPreAdmissionCustodyV3 {
    pub fn into_parts(
        self,
    ) -> (
        Arc<RoventPinnedNativeSources<RoventInputLockV3>>,
        RoventNativeResourcesV3,
        Option<Box<dyn VirtualReviewCollectionObserver>>,
    ) {
        (self.sources, self.resources, self.observer)
    }
}

#[cfg(target_os = "linux")]
impl RoventUnstartedNativeCustodyV3 {
    pub fn downcast<T: std::any::Any + Send>(self) -> Result<Box<T>, Self> {
        self.original
            .downcast()
            .map_err(|original| Self { original })
    }
}

#[cfg(target_os = "linux")]
impl<Q: NativeQuestionProjection> From<RoventNativeStartupRejection<Q>>
    for RoventNativeStartupRejectionV3<Q>
{
    fn from(rejection: RoventNativeStartupRejection<Q>) -> Self {
        Self::Started(rejection)
    }
}

#[cfg(target_os = "linux")]
impl<Q: NativeQuestionProjection> RoventNativeStartupRejectionV3<Q> {
    /// Compatibility for the historical V3 constructor. New launch owners
    /// must use the retaining constructor instead of discarding this owner.
    fn into_legacy(self) -> RoventNativeStartupRejection<Q> {
        match self {
            Self::Started(rejection) => rejection,
            Self::Unstarted { error, owner: _ } => RoventNativeStartupRejection {
                error,
                native: None,
            },
        }
    }
}
impl<Q: NativeQuestionProjection> std::fmt::Debug for RoventNativeStartupRejection<Q> {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("RoventNativeStartupRejection")
            .field("error", &self.error)
            .field("retained_owner", &self.native.is_some())
            .finish()
    }
}
pub struct PreparedRoventNative<Q: NativeQuestionProjection = AdaptivePortQuestion> {
    service: RoventNativeService<Q>,
    snapshot: RoventNativeStartupSnapshot,
    /// Retains the exact declaration and descriptor capabilities for startup.
    sources: Arc<RoventPinnedNativeSources>,
    /// The same successful Ready service owns one lifetime across every origin.
    product_lifetime: Arc<crate::pals_adaptive::RoventNativeProductLifetime>,
}
impl<Q: NativeQuestionProjection> PreparedRoventNative<Q> {
    /// 원 성공 load와 descriptor owner를 보존한다. 실제 root 수집은 별도로
    /// Frontier 부착과 native submit/complete/retire를 모두 요구한다.
    pub fn collection_identity(
        &self,
    ) -> Result<crate::pals_adaptive::RoventNativeCollectionOrigin, BackendError> {
        crate::pals_adaptive::RoventNativeCollectionOrigin::from_loaded_with_lifetime(
            &self.snapshot,
            Arc::clone(&self.sources),
            Arc::clone(&self.product_lifetime),
        )
        .map_err(|_| {
            fail(
                K::IdentityMismatch,
                "loaded source is not eligible for Rovent collection origin",
            )
        })
    }
    pub fn snapshot(&self) -> &RoventNativeStartupSnapshot {
        &self.snapshot
    }
    pub fn sources(&self) -> &Arc<RoventPinnedNativeSources> {
        &self.sources
    }
    pub fn into_parts(
        self,
    ) -> (
        RoventNativeService<Q>,
        RoventNativeStartupSnapshot,
        Arc<RoventPinnedNativeSources>,
    ) {
        (self.service, self.snapshot, self.sources)
    }
}

/// Actually load the V-free Fresh P/C export once, off the manager thread, and
/// create its original bounded ticket service. It neither creates R nor routes
/// a native selected-view model through the synchronous RoleModel driver.
pub fn prepare_native<Q: NativeQuestionProjection>(
    sources: Arc<RoventPinnedNativeSources>,
    resources: RoventNativeResources,
    clock: ContractSystemClock,
    control: RoventStartupControl,
) -> Result<PreparedRoventNative<Q>, RoventNativeStartupRejection<Q>> {
    prepare_native_with_optional_observer(
        sources,
        resources,
        clock,
        control,
        None,
        NativeStartupDiagnostic::default(),
    )
}

/// 원 native owner가 입력을 준비하는 시점에만 수집기를 호출한다. observer는
/// Ready 이전에 설치되며 기존 호출자의 기본 경로·모델·물리 worker 수는 같다.
pub fn prepare_native_with_collection_observer<Q: NativeQuestionProjection>(
    sources: Arc<RoventPinnedNativeSources>,
    resources: RoventNativeResources,
    clock: ContractSystemClock,
    control: RoventStartupControl,
    observer: Box<dyn VirtualReviewCollectionObserver>,
) -> Result<PreparedRoventNative<Q>, RoventNativeStartupRejection<Q>> {
    prepare_native_with_optional_observer(
        sources,
        resources,
        clock,
        control,
        Some(observer),
        NativeStartupDiagnostic::default(),
    )
}

fn prepare_native_with_optional_observer<Q: NativeQuestionProjection>(
    sources: Arc<RoventPinnedNativeSources>,
    resources: RoventNativeResources,
    clock: ContractSystemClock,
    control: RoventStartupControl,
    observer: Option<Box<dyn VirtualReviewCollectionObserver>>,
    diagnostic: NativeStartupDiagnostic,
) -> Result<PreparedRoventNative<Q>, RoventNativeStartupRejection<Q>> {
    let (native, snapshot) = spawn_native_owner(
        Arc::clone(&sources),
        resources,
        clock,
        &control,
        observer,
        &diagnostic,
    )?;
    let product_lifetime = match crate::pals_adaptive::RoventNativeProductLifetime::from_ready(
        &snapshot, &sources, &native,
    ) {
        Ok(lifetime) => lifetime,
        Err(error) => {
            return Err(rejected_native(
                fail(
                    K::IdentityMismatch,
                    "original Ready native lifetime was rejected",
                )
                .with_diagnostic("RoventNativeLifetime", &error.to_string()),
                Some(native),
                control.cleanup_until,
            ));
        }
    };
    diagnostic.emit(
        NativeStartupDiagnosticStage::NativeReady,
        NativeStartupDiagnosticPhase::Exit,
    );
    Ok(PreparedRoventNative {
        service: *native,
        snapshot,
        sources,
        product_lifetime,
    })
}

/// Retains the exact /2 sources and the same physical ticket-service owner.
/// Collection provenance additionally requires original preloaded sources.
pub struct PreparedRoventNativeV2<Q: NativeQuestionProjection = AdaptivePortQuestion> {
    service: RoventNativeService<Q>,
    snapshot: RoventNativeStartupSnapshot,
    sources: Arc<RoventPinnedNativeSources<RoventInputLockV2>>,
    _verifier: rz_search::pals::coordinator::NativeAcceptanceVerifier,
    control: RoventStartupControl,
}
/// Borrowed witness that can only be constructed beside the prepared owner.
/// Its fields are private; unrelated native sources cannot be mixed into it.
pub(crate) struct RoventCollectionReadyOwnerV2<'a, Q: NativeQuestionProjection> {
    prepared: &'a PreparedRoventNativeV2<Q>,
}
impl<Q: NativeQuestionProjection> RoventCollectionReadyOwnerV2<'_, Q> {
    pub(crate) fn snapshot(&self) -> &RoventNativeStartupSnapshot {
        &self.prepared.snapshot
    }
    pub(crate) fn sources(&self) -> &Arc<RoventPinnedNativeSources<RoventInputLockV2>> {
        &self.prepared.sources
    }
    pub(crate) fn service(&self) -> &RoventNativeService<Q> {
        &self.prepared.service
    }
}
impl<Q: NativeQuestionProjection> PreparedRoventNativeV2<Q> {
    /// Issue Collection origin only from this Ready service and its exact /2
    /// source owner. Frontier/root and teacher authority remain separate.
    pub fn collection_identity(
        &self,
    ) -> Result<crate::pals_adaptive::RoventNativeCollectionOriginV2, BackendError> {
        crate::pals_adaptive::RoventNativeCollectionOriginV2::from_loaded_ready(
            RoventCollectionReadyOwnerV2 { prepared: self },
        )
        .map_err(|_| {
            fail(
                K::IdentityMismatch,
                "loaded V2 source is not eligible for Rovent collection origin",
            )
        })
    }
    pub fn snapshot(&self) -> &RoventNativeStartupSnapshot {
        &self.snapshot
    }
    pub fn sources(&self) -> &Arc<RoventPinnedNativeSources<RoventInputLockV2>> {
        &self.sources
    }
    pub(crate) fn startup_control(&self) -> &RoventStartupControl {
        &self.control
    }
    pub fn into_parts(
        self,
    ) -> (
        RoventNativeService<Q>,
        RoventNativeStartupSnapshot,
        Arc<RoventPinnedNativeSources<RoventInputLockV2>>,
    ) {
        (self.service, self.snapshot, self.sources)
    }
}

/// Load only the selected V2 V-free P/C source. Real Ready and worker-issued
/// acceptance verifier are required; this does not grant a root or receipt.
pub fn prepare_native_v2<Q: NativeQuestionProjection>(
    sources: Arc<RoventPinnedNativeSources<RoventInputLockV2>>,
    resources: RoventNativeResources,
    clock: ContractSystemClock,
    control: RoventStartupControl,
) -> Result<PreparedRoventNativeV2<Q>, RoventNativeStartupRejection<Q>> {
    prepare_native_v2_inner(sources, resources, clock, control, None)
}

/// Prepare the original /2 Collection native worker with its actual input
/// observer. This only establishes the Ready owner: it does not create root,
/// teacher, publication, or physical-completion authority.
pub fn prepare_native_v2_collection<Q: NativeQuestionProjection>(
    sources: Arc<RoventPinnedNativeSources<RoventInputLockV2>>,
    resources: RoventNativeResources,
    clock: ContractSystemClock,
    control: RoventStartupControl,
    observer: Box<dyn VirtualReviewCollectionObserver>,
) -> Result<PreparedRoventNativeV2<Q>, RoventNativeStartupRejection<Q>> {
    if sources.lock().manifest.purpose != rz_experiments::RoventPurpose::Collection
        || sources.collection_preload_sources().is_none()
    {
        return Err(RoventNativeStartupRejection {
            error: fail(
                K::IdentityMismatch,
                "V2 Collection needs its original preload",
            ),
            native: None,
        });
    }
    prepare_native_v2_inner(sources, resources, clock, control, Some(observer))
}

fn prepare_native_v2_inner<Q: NativeQuestionProjection>(
    sources: Arc<RoventPinnedNativeSources<RoventInputLockV2>>,
    resources: RoventNativeResources,
    clock: ContractSystemClock,
    control: RoventStartupControl,
    observer: Option<Box<dyn VirtualReviewCollectionObserver>>,
) -> Result<PreparedRoventNativeV2<Q>, RoventNativeStartupRejection<Q>> {
    if !sources.matches_startup_control(&control) {
        return Err(RoventNativeStartupRejection {
            error: fail(
                K::IdentityMismatch,
                "V2 source and native startup control differ",
            ),
            native: None,
        });
    }
    let diagnostic = NativeStartupDiagnostic::default();
    let (native, snapshot) = spawn_native_owner(
        Arc::clone(&sources),
        resources,
        clock.clone(),
        &control,
        observer,
        &diagnostic,
    )?;
    let ready_clock_matches = native
        .ready_clock()
        .is_some_and(|ready| ready.domain() == clock.domain() && ready.origin() == clock.origin());
    let verifier = match native.native_acceptance_verifier() {
        Some(verifier)
            if snapshot.input_lock == sources.lock_digest()
                && snapshot.process_epoch == clock.domain().0.0
                && ready_clock_matches
                && snapshot.model_namespace().is_ok()
                && snapshot.physical_workers_created == 1
                && !snapshot.neural_service_executed =>
        {
            verifier
        }
        _ => {
            return Err(rejected_native(
                fail(
                    K::IdentityMismatch,
                    "original V2 Ready service/source differs",
                ),
                Some(native),
                control.cleanup_until,
            ));
        }
    };
    diagnostic.emit(
        NativeStartupDiagnosticStage::NativeReady,
        NativeStartupDiagnosticPhase::Exit,
    );
    Ok(PreparedRoventNativeV2 {
        service: *native,
        snapshot,
        sources,
        _verifier: verifier,
        control,
    })
}

/// The original V3 source and its single Ready physical worker.
/// Collection authority and physical-completion receipts are not implied.
pub struct PreparedRoventNativeV3<Q: NativeQuestionProjection = AdaptivePortQuestion> {
    service: RoventNativeService<Q>,
    snapshot: RoventNativeStartupSnapshot,
    sources: Arc<RoventPinnedNativeSources<RoventInputLockV3>>,
    _verifier: rz_search::pals::coordinator::NativeAcceptanceVerifier,
    control: RoventStartupControl,
}

impl<Q: NativeQuestionProjection> PreparedRoventNativeV3<Q> {
    /// Borrow the original worker; this issues no new authority or ownership.
    pub(crate) fn collection_ready_service(&self) -> &RoventNativeService<Q> {
        &self.service
    }

    pub fn snapshot(&self) -> &RoventNativeStartupSnapshot {
        &self.snapshot
    }
    pub fn sources(&self) -> &Arc<RoventPinnedNativeSources<RoventInputLockV3>> {
        &self.sources
    }
    pub(crate) fn startup_control(&self) -> &RoventStartupControl {
        &self.control
    }
    pub fn into_parts(
        self,
    ) -> (
        RoventNativeService<Q>,
        RoventNativeStartupSnapshot,
        Arc<RoventPinnedNativeSources<RoventInputLockV3>>,
    ) {
        (self.service, self.snapshot, self.sources)
    }
}

/// Prepare the admitted CPU or system-CUDA configuration on one physical worker.
/// A Ready result is not an inference, GPU fence, or Collection receipt.
#[cfg(target_os = "linux")]
pub fn prepare_native_v3<Q: NativeQuestionProjection>(
    sources: Arc<RoventPinnedNativeSources<RoventInputLockV3>>,
    resources: impl Into<RoventNativeResourcesV3>,
    clock: ContractSystemClock,
    control: RoventStartupControl,
) -> Result<PreparedRoventNativeV3<Q>, RoventNativeStartupRejection<Q>> {
    prepare_native_v3_retaining(sources, resources, clock, control)
        .map_err(RoventNativeStartupRejectionV3::into_legacy)
}

/// V3 process owners use this route to retain an unstarted factory owner.
#[cfg(target_os = "linux")]
pub fn prepare_native_v3_retaining<Q: NativeQuestionProjection>(
    sources: Arc<RoventPinnedNativeSources<RoventInputLockV3>>,
    resources: impl Into<RoventNativeResourcesV3>,
    clock: ContractSystemClock,
    control: RoventStartupControl,
) -> Result<PreparedRoventNativeV3<Q>, RoventNativeStartupRejectionV3<Q>> {
    prepare_native_v3_inner(
        sources,
        resources.into(),
        clock,
        control,
        None,
        None,
        V3ObserverIntent::None,
    )
}

/// Collection captures training input. The system numeric observer is an
/// independent, opt-in diagnostic and must not grant Collection authority.
#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum V3ObserverIntent {
    None,
    Collection,
    SystemNumeric,
}

/// A diagnostic observer currently constructed only at the explicit
/// system-CUDA validation opt-in branch. The private field limits its
/// construction to this module and its descendants.
#[cfg(target_os = "linux")]
pub(crate) struct V3SystemNumericObserver(Box<dyn VirtualReviewCollectionObserver>);

#[cfg(target_os = "linux")]
fn permits_v3_observer(
    purpose: RoventPurpose,
    has_collection_preload: bool,
    observer_present: bool,
    intent: V3ObserverIntent,
    system_cuda: bool,
    numeric_opt_in: bool,
) -> bool {
    match (purpose, has_collection_preload, observer_present, intent) {
        (RoventPurpose::Collection, true, true, V3ObserverIntent::Collection) => true,
        (
            RoventPurpose::Validation | RoventPurpose::Deployment,
            false,
            false,
            V3ObserverIntent::None,
        ) => true,
        (RoventPurpose::Validation, false, true, V3ObserverIntent::SystemNumeric) => {
            system_cuda && numeric_opt_in
        }
        _ => false,
    }
}

/// The parent connection is created only by the controlled V3 launch after
/// the original endpoint has accepted and authenticated its actual parent.
#[cfg(target_os = "linux")]
pub(crate) fn prepare_native_v3_retaining_with_parent<Q: NativeQuestionProjection>(
    sources: Arc<RoventPinnedNativeSources<RoventInputLockV3>>,
    resources: impl Into<RoventNativeResourcesV3>,
    clock: ContractSystemClock,
    control: RoventStartupControl,
    parent: Option<Arc<launch::ConnectedParent>>,
    observer: Option<V3SystemNumericObserver>,
) -> Result<PreparedRoventNativeV3<Q>, RoventNativeStartupRejectionV3<Q>> {
    let intent = if observer.is_some() {
        V3ObserverIntent::SystemNumeric
    } else {
        V3ObserverIntent::None
    };
    let observer = observer.map(|numeric| numeric.0);
    prepare_native_v3_inner(
        sources,
        resources.into(),
        clock,
        control,
        observer,
        parent,
        intent,
    )
}

/// Prepare the single original /3 Collection worker with its input observer.
/// The caller must retain custody of that observer and recover it only after
/// physical completion, worker join and any teacher settlement are verified.
/// Ready does not establish root, teacher, publication or completion authority.
#[cfg(target_os = "linux")]
pub fn prepare_native_v3_collection<Q: NativeQuestionProjection>(
    sources: Arc<RoventPinnedNativeSources<RoventInputLockV3>>,
    resources: impl Into<RoventNativeResourcesV3>,
    clock: ContractSystemClock,
    control: RoventStartupControl,
    observer: Box<dyn VirtualReviewCollectionObserver>,
) -> Result<PreparedRoventNativeV3<Q>, RoventNativeStartupRejection<Q>> {
    prepare_native_v3_collection_retaining(sources, resources.into(), clock, control, observer)
        .map_err(RoventNativeStartupRejectionV3::into_legacy)
}

#[cfg(target_os = "linux")]
pub fn prepare_native_v3_collection_retaining<Q: NativeQuestionProjection>(
    sources: Arc<RoventPinnedNativeSources<RoventInputLockV3>>,
    resources: RoventNativeResourcesV3,
    clock: ContractSystemClock,
    control: RoventStartupControl,
    observer: Box<dyn VirtualReviewCollectionObserver>,
) -> Result<PreparedRoventNativeV3<Q>, RoventNativeStartupRejectionV3<Q>> {
    prepare_native_v3_inner(
        sources,
        resources,
        clock,
        control,
        Some(observer),
        None,
        V3ObserverIntent::Collection,
    )
}

#[cfg(target_os = "linux")]
fn prepare_native_v3_inner<Q: NativeQuestionProjection>(
    sources: Arc<RoventPinnedNativeSources<RoventInputLockV3>>,
    resources: RoventNativeResourcesV3,
    clock: ContractSystemClock,
    control: RoventStartupControl,
    observer: Option<Box<dyn VirtualReviewCollectionObserver>>,
    parent: Option<Arc<launch::ConnectedParent>>,
    observer_intent: V3ObserverIntent,
) -> Result<PreparedRoventNativeV3<Q>, RoventNativeStartupRejectionV3<Q>> {
    let admission = || -> Result<(), BackendError> {
        control.check()?;
        sources.validate_lock()?;
        if !sources.admits_original_startup_control(&control)
            || clock.domain().0.0 == 0
            || control.cleanup_until < control.until
            || control
                .cleanup_until
                .saturating_duration_since(control.until)
                > Duration::from_millis(sources.lock().cleanup_max_ms())
        {
            return Err(fail(
                K::IdentityMismatch,
                "V3 startup owner or deadline differs",
            ));
        }
        let system_cuda = matches!(
            &sources.lock().manifest.system_runtime_lock.runtime,
            NativeRuntimeSourceV6::SystemCuda(_)
        );
        // SystemNumeric can only be constructed at the opt-in branch in the
        // launch, so admission does not reread mutable process environment.
        let numeric_opt_in = cfg!(feature = "native-validation-diagnostics");
        if !permits_v3_observer(
            sources.lock().manifest.purpose,
            sources.collection_preload_sources().is_some(),
            observer.is_some(),
            observer_intent,
            system_cuda,
            numeric_opt_in,
        ) {
            return Err(fail(
                K::IdentityMismatch,
                "V3 observer intent differs from purpose, preload or diagnostic configuration",
            ));
        }
        if observer_intent == V3ObserverIntent::Collection {
            sources.verify_collection_v3_origin()?;
        }
        match (
            &sources.lock().manifest.system_runtime_lock.runtime,
            &resources,
        ) {
            (NativeRuntimeSourceV6::Cpu, RoventNativeResourcesV3::Cpu(cpu)) => {
                validate_native_resources(&sources, cpu)
            }
            (NativeRuntimeSourceV6::SystemCuda(_), RoventNativeResourcesV3::SystemCuda(cuda)) => {
                validate_system_native_resources_v3(&sources, cuda)
            }
            _ => Err(fail(
                K::IdentityMismatch,
                "V3 native resources differ from runtime lock",
            )),
        }
    };
    if let Err(error) = admission() {
        return Err(RoventNativeStartupRejectionV3::Unstarted {
            error,
            owner: RoventUnstartedNativeCustodyV3 {
                original: Box::new(RoventPreAdmissionCustodyV3 {
                    sources,
                    resources,
                    observer,
                }),
            },
        });
    }
    let pending_capacity = sources.lock().role_tickets_max() as usize;
    let (limits, commands, expects_system_cuda) = match &resources {
        RoventNativeResourcesV3::Cpu(cpu) => (cpu.tickets, cpu.command_capacity, false),
        RoventNativeResourcesV3::SystemCuda(cuda) => (cuda.tickets, cuda.command_capacity, true),
    };
    let factory_sources = Arc::clone(&sources);
    let factory_clock = clock.clone();
    let factory_control = control.clone();
    let diagnostic = NativeStartupDiagnostic::default();
    let factory_diagnostic = diagnostic.clone();
    let (native, snapshot) = match resources {
        RoventNativeResourcesV3::Cpu(cpu) => spawn_native_factory(
            pending_capacity,
            limits,
            commands,
            clock.clone(),
            &control,
            &diagnostic,
            move || {
                load_native(
                    factory_sources,
                    cpu,
                    factory_clock,
                    &factory_control,
                    observer,
                    &factory_diagnostic,
                )
            },
        )?,
        RoventNativeResourcesV3::SystemCuda(cuda) => spawn_system_native_factory_v3(
            factory_sources,
            cuda,
            factory_clock,
            factory_control,
            observer,
            parent,
            &control,
            &diagnostic,
        )?,
    };
    let ready_clock_matches = native
        .ready_clock()
        .is_some_and(|ready| ready.domain() == clock.domain() && ready.origin() == clock.origin());
    let verifier = match native.native_acceptance_verifier() {
        Some(verifier)
            if snapshot.input_lock == sources.lock_digest()
                && snapshot.process_epoch == clock.domain().0.0
                && ready_clock_matches
                && snapshot.model_namespace().is_ok()
                && snapshot.system_runtime.is_some() == expects_system_cuda
                && snapshot.physical_workers_created == 1
                && !snapshot.neural_service_executed =>
        {
            verifier
        }
        _ => {
            return Err(rejected_native(
                fail(
                    K::IdentityMismatch,
                    "original V3 Ready service/source differs",
                ),
                Some(native),
                control.cleanup_until,
            )
            .into());
        }
    };
    Ok(PreparedRoventNativeV3 {
        service: *native,
        snapshot,
        sources,
        _verifier: verifier,
        control,
    })
}

fn spawn_native_owner<Q: NativeQuestionProjection, L: NativeInputLock>(
    sources: Arc<RoventPinnedNativeSources<L>>,
    resources: RoventNativeResources,
    clock: ContractSystemClock,
    control: &RoventStartupControl,
    observer: Option<Box<dyn VirtualReviewCollectionObserver>>,
    diagnostic: &NativeStartupDiagnostic,
) -> Result<
    (Box<RoventNativeService<Q>>, RoventNativeStartupSnapshot),
    RoventNativeStartupRejection<Q>,
> {
    let before = || -> Result<(), BackendError> {
        control.check()?;
        sources.validate_lock()?;
        validate_native_resources(&sources, &resources)?;
        if sources.collection_preload_sources().is_some() && observer.is_none() {
            return Err(fail(
                K::UnsupportedModel,
                "explicit Collection startup requires its original input observer",
            ));
        }
        if clock.domain().0.0 == 0
            || control.cleanup_until < control.until
            || control
                .cleanup_until
                .saturating_duration_since(control.until)
                > Duration::from_millis(sources.lock().cleanup_max_ms())
        {
            return Err(fail(
                K::InvalidInput,
                "Rovent native startup requires its original epoch",
            ));
        }
        Ok(())
    };
    if let Err(error) = before() {
        return Err(RoventNativeStartupRejection {
            error,
            native: None,
        });
    }
    let pending_capacity = sources.lock().role_tickets_max() as usize;
    let limits = resources.tickets;
    let commands = resources.command_capacity;
    let factory_sources = Arc::clone(&sources);
    let factory_control = control.clone();
    let factory_clock = clock.clone();
    let load_diagnostic = diagnostic.clone();
    spawn_native_factory(
        pending_capacity,
        limits,
        commands,
        clock,
        control,
        diagnostic,
        move || {
            load_native(
                factory_sources,
                resources,
                factory_clock,
                &factory_control,
                observer,
                &load_diagnostic,
            )
        },
    )
}

/// Shared ticket owner and Ready/drain protocol for retained and system
/// runtimes. The supplied factory always executes on the physical owner.
fn spawn_native_factory<Q, F>(
    pending_capacity: usize,
    limits: TicketServiceLimits,
    commands: usize,
    clock: ContractSystemClock,
    control: &RoventStartupControl,
    diagnostic: &NativeStartupDiagnostic,
    factory: F,
) -> Result<
    (Box<RoventNativeService<Q>>, RoventNativeStartupSnapshot),
    RoventNativeStartupRejection<Q>,
>
where
    Q: NativeQuestionProjection,
    F: FnOnce() -> Result<
            (
                NativeVirtualReviewTicketBackend,
                RoventNativeStartupSnapshot,
            ),
            BackendError,
        > + Send
        + 'static,
{
    let factory_diagnostic = diagnostic.clone();
    let (sender, receiver) = mpsc::sync_channel(1);
    let native = TicketedNativeService::spawn_with_capacity(
        move || {
            let loaded =
                factory_diagnostic.trace(NativeStartupDiagnosticStage::NativeFactory, || factory());
            let observation = loaded
                .as_ref()
                .map(|(_, snapshot)| snapshot.clone())
                .map_err(Clone::clone);
            let _ = sender.send(observation);
            loaded
                .map(|(backend, _)| ProjectedNativeBackend::new(backend))
                .map_err(|error| RoleFailure::Backend(error.to_string()))
        },
        limits,
        pending_capacity,
        commands,
    )
    .map_err(|_| RoventNativeStartupRejection {
        error: fail(K::ResourceExhausted, "Rovent native owner spawn failed"),
        native: None,
    })?;
    wait_original_native_factory(native, receiver, clock, control, diagnostic)
}

fn wait_original_native_factory<Q: NativeQuestionProjection>(
    native: RoventNativeService<Q>,
    receiver: mpsc::Receiver<Result<RoventNativeStartupSnapshot, BackendError>>,
    clock: ContractSystemClock,
    control: &RoventStartupControl,
    diagnostic: &NativeStartupDiagnostic,
) -> Result<
    (Box<RoventNativeService<Q>>, RoventNativeStartupSnapshot),
    RoventNativeStartupRejection<Q>,
> {
    let mut native = Some(Box::new(native));
    diagnostic.emit(
        NativeStartupDiagnosticStage::NativeReady,
        NativeStartupDiagnosticPhase::Entry,
    );
    let snapshot = loop {
        if let Err(error) = control.check() {
            return Err(rejected_native(error, native, control.cleanup_until));
        }
        let wait = control
            .until
            .saturating_duration_since(Instant::now())
            .min(Duration::from_millis(10));
        match receiver.recv_timeout(wait) {
            Ok(Ok(snapshot)) => break snapshot,
            Ok(Err(error)) => return Err(rejected_native(error, native, control.cleanup_until)),
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(rejected_native(
                    fail(K::BackendFailure, "Rovent native factory channel closed"),
                    native,
                    control.cleanup_until,
                ));
            }
        }
    };
    // Factory observation can arrive just before the service publishes Ready.
    // Consume only that startup event; no preparation/Run can exist yet.
    loop {
        if let Err(error) = control.check() {
            return Err(rejected_native(error, native, control.cleanup_until));
        }
        let owner = native.as_ref().expect("startup retains native owner");
        match owner.poll() {
            Some(TicketNativeEvent::Ready { clock: ready })
                if ready.domain() == clock.domain() && ready.origin() == clock.origin() =>
            {
                break;
            }
            Some(_) => {
                return Err(rejected_native(
                    fail(
                        K::IdentityMismatch,
                        "unexpected native event before Rovent admission",
                    ),
                    native,
                    control.cleanup_until,
                ));
            }
            None => {
                let signal = owner.signal();
                let version = signal.version();
                signal.wait_changed(version, Duration::from_millis(1));
            }
        }
    }
    if let Err(error) = control.check() {
        return Err(rejected_native(error, native, control.cleanup_until));
    }
    Ok((
        native
            .take()
            .expect("successful startup retains native owner"),
        snapshot,
    ))
}

/// Same original source, runtime and sessions remain in the factory owner on
/// every failure. This owner is neither a GPU reservation nor a fence.
#[cfg(target_os = "linux")]
pub struct RoventSystemFactoryCustodyV3 {
    sources: Arc<RoventPinnedNativeSources<RoventInputLockV3>>,
    review: VirtualReviewStartupCustody,
    system_owner: Option<rz_eval::onnx::SystemRuntime>,
    rejected_runtime: Option<OrtRuntime>,
    startup: Option<crate::pals_adaptive_bootstrap::runtime_assets::system_cuda::SystemCudaStartup>,
    ticket: Option<NativeVirtualReviewTicketBackend>,
    observer: Option<Box<dyn VirtualReviewCollectionObserver>>,
    physical_reservation: Option<rz_eval::worker::PhysicalRecoveryReservation>,
    physical_key: Option<rz_eval::worker::PhysicalRecoveryKey>,
}

#[cfg(target_os = "linux")]
impl RoventSystemFactoryCustodyV3 {
    pub fn sources(&self) -> &Arc<RoventPinnedNativeSources<RoventInputLockV3>> {
        &self.sources
    }
    /// Only an explicitly returned, unstarted owner may hand its observer back.
    /// A caller must not treat this as evidence about unrelated GPU work.
    pub fn take_unstarted_observer(&mut self) -> Option<Box<dyn VirtualReviewCollectionObserver>> {
        self.observer.take()
    }
}

#[cfg(target_os = "linux")]
impl crate::pals_native::ticketed::FactoryStartupRecovery for RoventSystemFactoryCustodyV3 {
    fn reobserve(
        &mut self,
        until: Instant,
    ) -> Result<crate::pals_native::ticketed::TicketNativeRecoveryProbe, RoleFailure> {
        match self.ticket.as_mut() {
            Some(ticket) => ticket.reobserve_quarantined(until),
            None => Ok(crate::pals_native::ticketed::TicketNativeRecoveryProbe::Unknown),
        }
    }

    fn release(
        &mut self,
        proof: &rz_eval::worker::PhysicalRecoveryConfirmation,
        until: Instant,
    ) -> Result<crate::pals_native::ticketed::TicketNativeRelease, RoleFailure> {
        match self.ticket.as_mut() {
            Some(ticket) => ticket.release_quarantined(proof, until),
            None => Ok(crate::pals_native::ticketed::TicketNativeRelease::Unknown),
        }
    }
}

#[cfg(target_os = "linux")]
fn spawn_system_native_factory_v3<Q: NativeQuestionProjection>(
    sources: Arc<RoventPinnedNativeSources<RoventInputLockV3>>,
    resources: RoventSystemNativeResourcesV3,
    clock: ContractSystemClock,
    factory_control: RoventStartupControl,
    observer: Option<Box<dyn VirtualReviewCollectionObserver>>,
    parent: Option<Arc<launch::ConnectedParent>>,
    control: &RoventStartupControl,
    diagnostic: &NativeStartupDiagnostic,
) -> Result<
    (Box<RoventNativeService<Q>>, RoventNativeStartupSnapshot),
    RoventNativeStartupRejectionV3<Q>,
> {
    let pending = sources.lock().role_tickets_max() as usize;
    let limits = resources.tickets;
    let commands = resources.command_capacity;
    let owner = RoventSystemFactoryCustodyV3 {
        sources: Arc::clone(&sources),
        review: VirtualReviewStartupCustody::new(
            Arc::clone(&sources) as Arc<dyn VirtualReviewOwnedSources>,
            resources.cuda_profile.clone(),
        ),
        system_owner: None,
        rejected_runtime: None,
        startup: None,
        ticket: None,
        observer,
        physical_reservation: None,
        physical_key: None,
    };
    let (sender, receiver) = mpsc::sync_channel(1);
    let factory_clock = clock.clone();
    let factory_diagnostic = diagnostic.clone();
    let native = TicketedNativeService::spawn_with_startup_custody(
        owner,
        move |factory| {
            let loaded = (|| {
                // Both process-local recovery slots exist before the first
                // native entry. They are not a cross-process device lease.
                let (reservation, key) = rz_eval::worker::PhysicalRecoveryReservation::reserve()?;
                factory
                    .bind_inner(&key, factory_clock.domain().0.0)
                    .map_err(|_| {
                        fail(
                            K::ResourceExhausted,
                            "cannot bind original physical recovery owner",
                        )
                    })?;
                let (owner, entry) = factory.parts();
                owner.physical_reservation = Some(reservation);
                owner.physical_key = Some(key);
                load_native_system_v3(
                    sources,
                    resources,
                    factory_clock,
                    &factory_control,
                    &factory_diagnostic,
                    parent,
                    owner,
                    entry,
                )
            })();
            // Keep the original ticket and observer in the same factory
            // custody while preparing and publishing the observation. A
            // failure or panic here must not unwind-drop the ticket owner.
            let observation = loaded.as_ref().cloned().map_err(Clone::clone);
            let _ = sender.send(observation);
            match loaded {
                Ok(_) => factory
                    .parts()
                    .0
                    .ticket
                    .take()
                    .map(ProjectedNativeBackend::new)
                    .ok_or(RoleFailure::InvalidOutput),
                Err(error) => Err(RoleFailure::Backend(error.to_string())),
            }
        },
        limits,
        pending,
        commands,
    )
    .map_err(|failure| RoventNativeStartupRejectionV3::Unstarted {
        error: fail(K::ResourceExhausted, "Rovent native owner spawn failed")
            .with_diagnostic("TicketOwnerSpawn", &format!("{:?}", failure.error)),
        owner: RoventUnstartedNativeCustodyV3 {
            // Keep the entire original rejection: its owner, optional
            // recovery key and any not-yet-joined original thread handle.
            original: Box::new(failure),
        },
    })?;
    wait_original_native_factory(native, receiver, clock, control, diagnostic).map_err(Into::into)
}

fn rejected_native<Q: NativeQuestionProjection>(
    error: BackendError,
    native: Option<Box<RoventNativeService<Q>>>,
    cleanup_until: Instant,
) -> RoventNativeStartupRejection<Q> {
    if let Some(owner) = &native {
        owner.begin_shutdown(cleanup_until);
    }
    RoventNativeStartupRejection { error, native }
}

/// Configuration identity for system CUDA, separate from legacy bundle and
/// control-inventory identity. The registration uses actual observed hashes,
/// not a built-in digest of a particular CUDA library distribution.
pub fn system_configuration_digest_v3(
    configuration: PalsOnnxConfig,
    runtime_binary: [u8; 32],
    system_identity: [u8; 32],
) -> Result<[u8; 32], BackendError> {
    let common =
        configuration_digest_from_metadata(configuration, None, runtime_binary, None, None)?;
    let mut hash = Sha256::new();
    hash.update(b"rz-rovent-native-system-configuration/1");
    hash.update(common);
    hash.update(system_identity);
    Ok(hash.finalize().into())
}

/// An explicit control-graph inventory changes the CUDA placement policy and
/// therefore has a separate identity. Preserve the historical strict digest.
#[cfg(target_os = "linux")]
pub fn system_configuration_digest_v3_with_inventory(
    configuration: PalsOnnxConfig,
    runtime_binary: [u8; 32],
    system_identity: [u8; 32],
    inventory: Option<[u8; 32]>,
) -> Result<[u8; 32], BackendError> {
    let strict = system_configuration_digest_v3(configuration, runtime_binary, system_identity)?;
    match inventory {
        None => Ok(strict),
        Some(inventory) => {
            let mut hash = Sha256::new();
            hash.update(b"rz-rovent-native-system-inventory-configuration/1");
            hash.update(strict);
            hash.update(inventory);
            Ok(hash.finalize().into())
        }
    }
}

#[cfg(target_os = "linux")]
fn validate_system_native_resources_v3(
    sources: &RoventPinnedNativeSources<RoventInputLockV3>,
    resources: &RoventSystemNativeResourcesV3,
) -> Result<(), BackendError> {
    let lock = sources.lock();
    let spec = match &lock.manifest.system_runtime_lock.runtime {
        NativeRuntimeSourceV6::SystemCuda(spec) => spec,
        NativeRuntimeSourceV6::Cpu => {
            return Err(fail(
                K::IdentityMismatch,
                "V3 system CUDA cannot select CPU",
            ));
        }
    };
    let startup = sources.configuration();
    let pc = &lock.engine().pc;
    let expected = startup.fields().native.onnx_system(&pc.device, spec)?;
    let tickets = startup
        .fields()
        .native
        .tickets(startup.fields().review.token_budget)?;
    let matches_provider = match (expected.provider, resources.configuration.provider) {
        (
            Provider::Cuda {
                device_id: a,
                arena_bytes: b,
            },
            Provider::Cuda {
                device_id: c,
                arena_bytes: d,
            },
        ) => a == c && b == d,
        _ => false,
    };
    if !matches_provider
        || expected.intra_threads != resources.configuration.intra_threads
        || expected.cache_public_memory != resources.configuration.cache_public_memory
        || expected.device_public_memory != resources.configuration.device_public_memory
        || expected.cudnn_conv_search != resources.configuration.cudnn_conv_search
        || startup.fields().native.command_capacity != resources.command_capacity
        || Duration::from_millis(startup.fields().native.drain_ms) != resources.drain_limit
        || tickets.review_token_budget != resources.tickets.review_token_budget
        || tickets.prepared_host_bytes_per_ticket
            != resources.tickets.prepared_host_bytes_per_ticket
        || tickets.output_host_bytes_per_ticket != resources.tickets.output_host_bytes_per_ticket
        || resources.drain_limit.is_zero()
        || resources.drain_limit > Duration::from_millis(lock.cleanup_max_ms())
        || resources.configuration.intra_threads != 1
        || pc.ort_intra_threads != Some(1)
        || pc.ort_inter_threads != Some(1)
        || pc.ort_spinning != Some(false)
        || resources.configuration.device_public_memory
        || !(1..=8).contains(&resources.command_capacity)
        || lock.role_tickets_max() > 16
        || sources.cuda_inventory_digest().is_some() != resources.cuda_profile.is_some()
    {
        return Err(fail(
            K::IdentityMismatch,
            "V3 system native resources differ from the lock",
        ));
    }
    let core = spec
        .libraries
        .iter()
        .find(|item| item.role == "ort-core")
        .ok_or_else(|| fail(K::IdentityMismatch, "V3 system ORT core was not registered"))?;
    let actual = system_configuration_digest_v3_with_inventory(
        resources.configuration,
        parse_sha256(&core.sha256)?,
        parse_sha256(&spec.system_identity_sha256)?,
        sources.cuda_inventory_digest(),
    )?;
    if parse_sha256(&pc.backend_configuration_sha256)? != actual {
        return Err(fail(
            K::IdentityMismatch,
            "V3 system native configuration digest differs",
        ));
    }
    resources
        .configuration
        .validate_system_execution_selection()
}

/// Require a live connection to the original parent immediately before native
/// system-CUDA entry. This does not prove GPU identity, device exclusivity,
/// future parent liveness or physical completion.
#[cfg(target_os = "linux")]
fn require_system_cuda_supervisor_capability(
    parent: Option<&launch::ConnectedParent>,
) -> Result<(), BackendError> {
    let parent = parent.ok_or_else(|| {
        fail(
            K::BackendUnavailable,
            "authenticated system CUDA supervisor capability unavailable",
        )
    })?;
    parent.ensure_live().map_err(|_| {
        fail(
            K::BackendUnavailable,
            "original parent connection is no longer live",
        )
    })
}

#[cfg(target_os = "linux")]
fn enter_supervised_system_cuda<T>(
    parent: Option<&launch::ConnectedParent>,
    native: impl FnOnce() -> Result<T, BackendError>,
) -> Result<T, BackendError> {
    require_system_cuda_supervisor_capability(parent)?;
    native()
}

#[cfg(all(test, target_os = "linux"))]
mod system_cuda_supervisor_entry_tests {
    use super::*;

    #[test]
    fn control_inventory_has_distinct_identity_and_preserves_strict_digest() {
        let config = PalsOnnxConfig::cpu();
        let strict = system_configuration_digest_v3(config, [1; 32], [2; 32]).unwrap();
        assert_eq!(
            strict,
            system_configuration_digest_v3_with_inventory(config, [1; 32], [2; 32], None).unwrap()
        );
        let a =
            system_configuration_digest_v3_with_inventory(config, [1; 32], [2; 32], Some([3; 32]))
                .unwrap();
        let b =
            system_configuration_digest_v3_with_inventory(config, [1; 32], [2; 32], Some([4; 32]))
                .unwrap();
        assert_ne!(strict, a);
        assert_ne!(a, b);
    }

    #[test]
    fn missing_supervisor_rejects_before_native_callback() {
        let called = std::cell::Cell::new(false);
        let result = enter_supervised_system_cuda(None, || {
            called.set(true);
            Ok(())
        });
        assert_eq!(result.unwrap_err().kind, K::BackendUnavailable);
        assert!(!called.get());
    }
}

#[cfg(target_os = "linux")]
fn load_native_system_v3(
    sources: Arc<RoventPinnedNativeSources<RoventInputLockV3>>,
    resources: RoventSystemNativeResourcesV3,
    clock: ContractSystemClock,
    control: &RoventStartupControl,
    diagnostic: &NativeStartupDiagnostic,
    parent: Option<Arc<launch::ConnectedParent>>,
    custody: &mut RoventSystemFactoryCustodyV3,
    entry: &mut crate::pals_native::ticketed::FactoryNativeEntry,
) -> Result<RoventNativeStartupSnapshot, BackendError> {
    control.check()?;
    sources.validate_lock()?;
    validate_system_native_resources_v3(&sources, &resources)?;
    let policy = diagnostic.trace(NativeStartupDiagnosticStage::CudaInventory, || {
        sources
            .cuda_inventory_bytes()?
            .map(|(raw, expected)| {
                PalsCudaControlPolicy::from_virtual_review_inventory_bytes(&raw, &hex(expected))
            })
            .transpose()
    })?;
    if let Some(parent) = parent.as_ref() {
        let ordinal = match &sources.lock().manifest.engine.pc.device {
            rz_experiments::RoventDevice::Cuda { ordinal, .. } => i32::try_from(*ordinal)
                .map_err(|_| fail(K::IdentityMismatch, "CUDA ordinal is out of range"))?,
            _ => {
                return Err(fail(
                    K::IdentityMismatch,
                    "system CUDA device declaration missing",
                ));
            }
        };
        if !parent.matches(
            clock.domain().0.0,
            &sources.lock().canonical_sha256,
            ordinal,
        ) {
            return Err(fail(
                K::IdentityMismatch,
                "parent connection differs from original V3 launch",
            ));
        }
    }
    let startup = diagnostic.trace(NativeStartupDiagnosticStage::OrtRuntime, || {
        sources.load_system_cuda_with(control, |system| {
            // The pinned original lives in this factory even if the native
            // initializer fails, times out or unwinds without a Session.
            custody.system_owner = Some(system.clone());
            enter_supervised_system_cuda(parent.as_deref(), || {
                let runtime = OrtRuntime::load_system_cuda_with_entry(system, || {
                    entry.mark_native_entered()
                })?;
                // Capture the same runtime before later cancellation/identity
                // checks can reject a successfully returned initialization.
                if let Err((error, rejected)) = custody.review.install_runtime(runtime.clone()) {
                    custody.rejected_runtime = Some(rejected);
                    return Err(error);
                }
                Ok(runtime)
            })
        })
    })?;
    custody.startup = Some(startup);
    control.check()?;
    let runtime = custody
        .startup
        .as_ref()
        .expect("original startup owned")
        .runtime();
    let identity = runtime.cuda_runtime_identity().ok_or_else(|| {
        fail(
            K::IdentityMismatch,
            "system CUDA owner identity unavailable",
        )
    })?;
    let config_digest = system_configuration_digest_v3_with_inventory(
        resources.configuration,
        runtime.binary_digest(),
        identity,
        sources.cuda_inventory_digest(),
    )?;
    let collection = sources.lock().manifest.purpose == RoventPurpose::Collection;
    let execution_mode = if collection {
        VirtualReviewExecutionMode::Collection
    } else {
        VirtualReviewExecutionMode::Deployment
    };
    diagnostic.trace(NativeStartupDiagnosticStage::ModelSessions, || {
        PalsVirtualReviewOnnxBackend::load_owned_sources_with_profile_into(
            &mut custody.review,
            &sources.export_digest_text(),
            resources.configuration,
            execution_mode,
            VirtualReviewArtifactKind::Fresh,
            policy,
            &mut || entry.mark_native_entered(),
        )
    })?;
    control.check()?;
    let backend = custody
        .review
        .backend()
        .ok_or_else(|| fail(K::BackendFailure, "original V3 startup backend missing"))?;
    if backend.validator_present() != collection
        || backend.approximate_warm_capable()
        || backend.model_epoch() != sources.weights_digest()
        || backend.rules_input_semantic_digest()
            != crate::pals_native::virtual_encoding::pals_virtual_rules_semantic_digest()
    {
        return Err(fail(
            K::IdentityMismatch,
            "loaded V3 role set or model namespace differs",
        ));
    }
    let session_settings = backend.applied_session_settings();
    let residency = backend.residency();
    if session_settings.is_empty()
        || session_settings.len() != residency.native_sessions
        || session_settings.len() != residency.graphs.len()
        || session_settings.iter().any(|session| {
            let selected_provider = match (session.provider(), resources.configuration.provider) {
                (
                    Provider::Cuda {
                        device_id: actual,
                        arena_bytes: actual_bytes,
                    },
                    Provider::Cuda {
                        device_id: expected,
                        arena_bytes: expected_bytes,
                    },
                ) => actual == expected && actual_bytes == expected_bytes,
                _ => false,
            };
            !selected_provider
                || session.cuda_provider_available() != Some(true)
                || residency
                    .graphs
                    .iter()
                    .filter(|graph| graph.role == session.role())
                    .count()
                    != 1
                || session_settings
                    .iter()
                    .filter(|other| other.role() == session.role())
                    .count()
                    != 1
        })
    {
        return Err(fail(
            K::IdentityMismatch,
            "actual V3 model sessions did not all confirm the selected CUDA provider",
        ));
    }
    let mapping = custody
        .startup
        .as_ref()
        .expect("original startup owned")
        .observe_ready_mappings()?;
    control.check()?;
    let snapshot = RoventNativeStartupSnapshot {
        domain: "rz-rovent-native-system-startup/1",
        input_lock: sources.lock_digest(),
        process_epoch: clock.domain().0.0,
        mode: sources.lock().engine().mode,
        topology: sources.topology(),
        weights: Digest(backend.model_epoch()),
        export_manifest: Digest(backend.manifest_digest()),
        backend: Digest(backend.backend_digest()),
        runtime_library: Digest(backend.runtime_binary_digest()),
        runtime_bundle: None,
        native_configuration: Digest(config_digest),
        system_runtime: Some(RoventSystemReadyObservationV3 {
            runtime_identity: Digest(identity),
            libraries: mapping.libraries,
            unregistered_native_mappings: mapping.unregistered_native_mappings,
            provider_ready: true,
            session_settings,
        }),
        startup_configuration: Digest(sources.configuration().configuration_digest()?),
        search_configuration: Digest(sources.configuration().search_configuration_digest()?),
        validator_loaded: backend.validator_present(),
        physical_workers_created: 1,
        session_residency: backend.residency().clone(),
        neural_service_executed: false,
    };
    let ticket = NativeVirtualReviewTicketBackend::new_with_collection_observer_retained(
        custody.review.backend_slot_mut(),
        custody.physical_reservation.as_mut().ok_or_else(|| {
            fail(
                K::ResourceExhausted,
                "original physical recovery reservation missing",
            )
        })?,
        clock,
        resources.drain_limit,
        &mut custody.observer,
    )
    .map_err(|_| {
        fail(
            K::BackendFailure,
            "V3 system ticket backend creation failed",
        )
    })?;
    custody.ticket = Some(ticket);
    Ok(snapshot)
}

fn load_native<L: NativeInputLock>(
    sources: Arc<RoventPinnedNativeSources<L>>,
    resources: RoventNativeResources,
    clock: ContractSystemClock,
    control: &RoventStartupControl,
    observer: Option<Box<dyn VirtualReviewCollectionObserver>>,
    diagnostic: &NativeStartupDiagnostic,
) -> Result<
    (
        NativeVirtualReviewTicketBackend,
        RoventNativeStartupSnapshot,
    ),
    BackendError,
> {
    control.check()?;
    validate_native_resources(&sources, &resources)?;
    let config_digest = configuration_digest(&resources, sources.cuda_inventory_digest())?;
    let profile = diagnostic.trace(NativeStartupDiagnosticStage::CudaInventory, || {
        match (sources.cuda_inventory_bytes()?, resources.cuda_profile) {
            (Some((raw, expected)), Some(directory)) => Ok(Some((
                PalsCudaControlPolicy::from_virtual_review_inventory_bytes(&raw, &hex(expected))?,
                directory,
            ))),
            (None, None) => Ok(None),
            _ => Err(fail(
                K::IdentityMismatch,
                "CUDA control and directory ownership differ",
            )),
        }
    })?;
    let runtime = diagnostic.trace(NativeStartupDiagnosticStage::OrtRuntime, || match resources
        .loading_profile
    {
        Some(profile) => OrtRuntime::load_with_cuda_loading_profile(&resources.runtime, profile),
        None => OrtRuntime::load(&resources.runtime),
    })?;
    control.check()?;
    // V1/V2 retain Deployment; an original V3 Collection lock explicitly
    // requests its validator and Collection inputs through the same loader.
    let execution_mode = sources.lock().virtual_review_mode();
    let backend = diagnostic.trace(NativeStartupDiagnosticStage::ModelSessions, || {
        PalsVirtualReviewOnnxBackend::load_owned_sources_with_profile(
            sources.clone() as Arc<dyn VirtualReviewOwnedSources>,
            &sources.export_digest_text(),
            runtime,
            resources.configuration,
            execution_mode,
            VirtualReviewArtifactKind::Fresh,
            profile,
        )
    })?;
    control.check()?;
    if backend.validator_present() != (execution_mode == VirtualReviewExecutionMode::Collection)
        || backend.approximate_warm_capable()
        || backend.model_epoch() != sources.weights_digest()
        || backend.rules_input_semantic_digest()
            != crate::pals_native::virtual_encoding::pals_virtual_rules_semantic_digest()
    {
        return Err(fail(
            K::IdentityMismatch,
            "loaded role set or model namespace differs from admitted export",
        ));
    }
    let snapshot = RoventNativeStartupSnapshot {
        domain: ROVENT_NATIVE_STARTUP_DOMAIN,
        input_lock: sources.lock_digest(),
        process_epoch: clock.domain().0.0,
        mode: sources.lock().engine().mode,
        topology: sources.topology(),
        weights: Digest(backend.model_epoch()),
        export_manifest: Digest(backend.manifest_digest()),
        backend: Digest(backend.backend_digest()),
        runtime_library: Digest(backend.runtime_binary_digest()),
        runtime_bundle: backend.runtime_bundle_digest().map(Digest),
        native_configuration: Digest(config_digest),
        system_runtime: None,
        startup_configuration: Digest(sources.configuration().configuration_digest()?),
        search_configuration: Digest(sources.configuration().search_configuration_digest()?),
        validator_loaded: backend.validator_present(),
        physical_workers_created: 1,
        session_residency: backend.residency().clone(),
        neural_service_executed: false,
    };
    let backend = NativeVirtualReviewTicketBackend::new_with_collection_observer(
        backend,
        clock,
        resources.drain_limit,
        observer,
    )
    .map_err(|_| fail(K::BackendFailure, "Rovent ticket backend creation failed"))?;
    // This constructor has created the inner physical worker. Always transfer
    // it to the outer service, even if admission expires now; Ready admission
    // will reject it while retaining the real finish/join owner for cleanup.
    Ok((backend, snapshot))
}

fn validate_native_resources<L: NativeInputLock>(
    sources: &RoventPinnedNativeSources<L>,
    resources: &RoventNativeResources,
) -> Result<(), BackendError> {
    let lock = sources.lock();
    validate_native_resource_fields(
        &lock.engine().pc,
        lock.cleanup_max_ms(),
        lock.role_tickets_max(),
        sources.configuration(),
        sources.cuda_inventory_digest(),
        resources,
        || sources.validate_runtime(&resources.runtime),
    )
}

/// Shared resource validation for the original typed lock's borrowed fields.
/// Runtime custody still belongs to the actual source owner supplied by the
/// caller; this function does not construct or attest that owner.
fn validate_native_resource_fields(
    pc: &RoventPcConfiguration,
    cleanup_max_ms: u64,
    role_tickets_max: u32,
    startup: &configuration::StartupConfigurationDocument,
    cuda_inventory: Option<[u8; 32]>,
    resources: &RoventNativeResources,
    validate_runtime: impl FnOnce() -> Result<(), BackendError>,
) -> Result<(), BackendError> {
    let expected = startup
        .fields()
        .native
        .onnx(&pc.device, resources.cuda_profile.is_some())?;
    let tickets = startup
        .fields()
        .native
        .tickets(startup.fields().review.token_budget)?;
    let provider_matches = match (expected.provider, resources.configuration.provider) {
        (Provider::Cpu, Provider::Cpu) => true,
        (
            Provider::Cuda {
                device_id: a,
                arena_bytes: b,
            },
            Provider::Cuda {
                device_id: c,
                arena_bytes: d,
            },
        ) => a == c && b == d,
        _ => false,
    };
    if !provider_matches
        || expected.intra_threads != resources.configuration.intra_threads
        || expected.cache_public_memory != resources.configuration.cache_public_memory
        || expected.device_public_memory != resources.configuration.device_public_memory
        || expected.cudnn_conv_search != resources.configuration.cudnn_conv_search
        || startup
            .fields()
            .native
            .loading_profile
            .map(configuration::LoadingProfile::actual)
            != resources.loading_profile
        || startup.fields().native.command_capacity != resources.command_capacity
        || Duration::from_millis(startup.fields().native.drain_ms) != resources.drain_limit
        || tickets.review_token_budget != resources.tickets.review_token_budget
        || tickets.prepared_host_bytes_per_ticket
            != resources.tickets.prepared_host_bytes_per_ticket
        || tickets.output_host_bytes_per_ticket != resources.tickets.output_host_bytes_per_ticket
    {
        return Err(fail(
            K::IdentityMismatch,
            "actual native constructor values differ from pinned startup configuration",
        ));
    }
    if resources.drain_limit.is_zero()
        || resources.drain_limit > Duration::from_millis(cleanup_max_ms)
        || resources.configuration.intra_threads != 1
        || pc.ort_intra_threads != Some(1)
        || pc.ort_inter_threads != Some(1)
        || pc.ort_spinning != Some(false)
        || resources.configuration.device_public_memory
        || !(1..=8).contains(&resources.command_capacity)
        || role_tickets_max > 16
    {
        return Err(fail(
            K::UnsupportedModel,
            "unsupported explicit Rovent native owner resource profile",
        ));
    }
    match (&pc.device, pc.backend, resources.configuration.provider) {
        (RoventDevice::Cpu, RoventBackend::OrtCpu, Provider::Cpu)
            if resources.loading_profile.is_none()
                && resources.cuda_profile.is_none()
                && cuda_inventory.is_none()
                && resources.runtime.bundle_digest().is_none() => {}
        (
            RoventDevice::Cuda { ordinal, .. },
            RoventBackend::OrtCuda,
            Provider::Cuda { device_id, .. },
        ) if i32::try_from(*ordinal).ok() == Some(device_id)
            && resources.loading_profile.is_some()
            && resources.cuda_profile.is_some()
            && cuda_inventory.is_some()
            && resources.runtime.bundle_digest().is_some() => {}
        _ => {
            return Err(fail(
                K::IdentityMismatch,
                "requested native provider/device differs from actual owner",
            ));
        }
    }
    validate_runtime()?;
    let actual = configuration_digest(resources, cuda_inventory)?;
    if parse_sha256(&pc.backend_configuration_sha256)? != actual {
        return Err(fail(
            K::IdentityMismatch,
            "native configuration differs from the Rovent lock",
        ));
    }
    resources
        .configuration
        .validate_execution_selection(resources.loading_profile, resources.cuda_profile.is_some())
}

/// Register this exact configuration before taking the lock. Paths and actual
/// memory/placement observations are deliberately outside this semantic hash.
pub fn configuration_digest(
    resources: &RoventNativeResources,
    control_inventory: Option<[u8; 32]>,
) -> Result<[u8; 32], BackendError> {
    configuration_digest_from_metadata(
        resources.configuration,
        resources.loading_profile,
        resources.runtime.binary_digest(),
        resources.runtime.bundle_digest(),
        control_inventory,
    )
}

/// Shared declaration codec. Unlike `configuration_digest`, this receives no
/// runtime owner and therefore cannot attest loading, a provider or readiness.
pub fn configuration_digest_from_metadata(
    configuration: PalsOnnxConfig,
    loading_profile: Option<NativeLoadingProfile>,
    runtime_binary: [u8; 32],
    runtime_bundle: Option<[u8; 32]>,
    control_inventory: Option<[u8; 32]>,
) -> Result<[u8; 32], BackendError> {
    let mut hash = Sha256::new();
    hash.update(ROVENT_NATIVE_CONFIGURATION_DOMAIN.as_bytes());
    match configuration.provider {
        Provider::Cpu => hash.update([0]),
        Provider::Cuda {
            device_id,
            arena_bytes,
        } => {
            hash.update([1]);
            hash.update(device_id.to_le_bytes());
            hash.update((arena_bytes as u64).to_le_bytes());
        }
    }
    hash.update((configuration.intra_threads as u64).to_le_bytes());
    hash.update([
        u8::from(configuration.cache_public_memory),
        u8::from(configuration.device_public_memory),
    ]);
    let cudnn = format!("{:?}", configuration.cudnn_conv_search);
    hash.update((cudnn.len() as u64).to_le_bytes());
    hash.update(cudnn.as_bytes());
    match loading_profile {
        None => hash.update([0]),
        Some(profile) => {
            hash.update([1]);
            hash.update((profile.identifier().len() as u64).to_le_bytes());
            hash.update(profile.identifier().as_bytes());
        }
    }
    hash.update(runtime_binary);
    for digest in [runtime_bundle, control_inventory] {
        match digest {
            None => hash.update([0]),
            Some(value) => {
                hash.update([1]);
                hash.update(value);
            }
        }
    }
    Ok(hash.finalize().into())
}

pub(crate) fn admitted_topology(
    lock: &RoventInputLock,
) -> Result<RoventStartupTopology, BackendError> {
    admitted_native_topology(lock, None)
}
pub(crate) fn admitted_collection_topology(
    lock: &RoventInputLock,
    original: &RoventCollectionPreloadSources,
) -> Result<RoventStartupTopology, BackendError> {
    admitted_native_topology(lock, Some(original))
}
fn admitted_native_topology(
    lock: &RoventInputLock,
    collection: Option<&RoventCollectionPreloadSources>,
) -> Result<RoventStartupTopology, BackendError> {
    lock.validate()
        .map_err(|_| fail(K::IdentityMismatch, "Rovent lock validation failed"))?;
    let engine = &lock.manifest.engine;
    let explicit_purpose = match collection {
        None => {
            matches!(
                lock.manifest.purpose,
                RoventPurpose::Validation | RoventPurpose::Deployment
            ) && engine.teachers.is_empty()
        }
        Some(original) => {
            lock.manifest.purpose == RoventPurpose::Collection
                && original.matches_input_lock(lock)
                && original.teacher_count() == engine.teachers.len()
        }
    };
    admitted_native_topology_fields(engine, &lock.manifest.coordinator, explicit_purpose)
}

/// Admit a /2 declaration using its original typed lock. Collection is not
/// admitted here: it requires a separately retained original teacher owner.
pub(crate) fn admitted_topology_v2(
    lock: &RoventInputLockV2,
) -> Result<RoventStartupTopology, BackendError> {
    lock.validate()
        .map_err(|_| fail(K::IdentityMismatch, "Rovent lock validation failed"))?;
    let engine = &lock.manifest.engine;
    let explicit_purpose = matches!(
        lock.manifest.purpose,
        RoventPurpose::Validation | RoventPurpose::Deployment
    ) && engine.teachers.is_empty();
    admitted_native_topology_fields(engine, &lock.manifest.coordinator, explicit_purpose)
}

/// Collection is a distinct admission: the original V2 lock must explicitly
/// declare teachers, and P/C still uses the same native capability checks.
pub(crate) fn admitted_collection_topology_v2(
    lock: &RoventInputLockV2,
) -> Result<RoventStartupTopology, BackendError> {
    lock.validate()
        .map_err(|_| fail(K::IdentityMismatch, "Rovent lock validation failed"))?;
    let engine = &lock.manifest.engine;
    let explicit =
        lock.manifest.purpose == RoventPurpose::Collection && !engine.teachers.is_empty();
    admitted_native_topology_fields(engine, &lock.manifest.coordinator, explicit)
}

fn admitted_native_topology_fields(
    engine: &RoventEngineDeclaration,
    coordinator: &RoventCoordinatorDeclaration,
    explicit_purpose: bool,
) -> Result<RoventStartupTopology, BackendError> {
    admitted_native_topology_fields_v3(engine, coordinator.view(), explicit_purpose, false)
}

/// V3 Collection uses the same model/export for all three role graphs.
/// Historical V1/V2 admission keeps the original validator-free condition.
fn admitted_native_topology_fields_v3<S>(
    engine: &RoventEngineDeclaration<S>,
    coordinator: rz_experiments::RoventCoordinatorView<'_>,
    explicit_purpose: bool,
    collection: bool,
) -> Result<RoventStartupTopology, BackendError> {
    if !explicit_purpose
        || if collection {
            !engine.pc.validator.as_ref().is_some_and(|validator| {
                validator.semantics == engine.pc.proposer.semantics
                    && validator.encoding == engine.pc.proposer.encoding
                    && validator.weights.sha256 == engine.pc.proposer.weights.sha256
                    && validator.weights.bytes == engine.pc.proposer.weights.bytes
            })
        } else {
            engine.pc.validator.is_some()
        }
        || !matches!(
            engine.pc.backend,
            RoventBackend::OrtCpu | RoventBackend::OrtCuda
        )
        || engine.pc.ort_intra_threads != Some(1)
        || engine.pc.ort_inter_threads != Some(1)
        || engine.pc.ort_spinning != Some(false)
        || coordinator.role_tickets_max > 16
    {
        return Err(fail(
            K::UnsupportedModel,
            if collection {
                "V3 Collection requires compatible semantics and shared P/C/V weights"
            } else {
                "this native constructor requires explicit V-free P/C use"
            },
        ));
    }
    if engine.pc.proposer != engine.pc.critic
        || engine.pc.proposer.semantics != virtual_review::MODEL_SEMANTICS
        || engine.pc.proposer.encoding != virtual_review::ENCODING
        || engine.pc.proposer.policy_head != ROVENT_NATIVE_POLICY_HEAD
        || engine.pc.proposer.value_head != ROVENT_NATIVE_VALUE_HEAD
        || engine.pc.proposer.diagnostic
    {
        return Err(fail(
            K::UnsupportedModel,
            "unsupported shared P/C model semantics or provenance",
        ));
    }
    match (&engine.researcher, engine.mode.has_native_r()) {
        (None, false) => Ok(RoventStartupTopology::RolesOnly),
        (Some(RoventResearcher::Own { .. }), true) => Ok(RoventStartupTopology::OwnResearcher),
        (Some(RoventResearcher::External { .. }), true) => {
            Ok(RoventStartupTopology::ExternalResearcher)
        }
        _ => Err(fail(
            K::IdentityMismatch,
            "Rovent mode/researcher topology differs",
        )),
    }
}

pub fn compiled_bootstrap_source_digest() -> [u8; 32] {
    bootstrap_source_digest(BOOTSTRAP_SOURCES)
}

fn bootstrap_source_digest(sources: &[&str]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"rz-rovent-native-bootstrap-source/1\0");
    for source in sources {
        hash.update((source.len() as u64).to_le_bytes());
        hash.update(source.as_bytes());
    }
    hash.finalize().into()
}

const BOOTSTRAP_SOURCES: &[&str] = &[
    include_str!("rovent_bootstrap.rs"),
    include_str!("rovent_bootstrap/pinned.rs"),
    include_str!("rovent_bootstrap/configuration.rs"),
    include_str!("rovent_bootstrap/configuration/v2.rs"),
    include_str!("rovent_bootstrap/configuration/v3.rs"),
    include_str!("rovent_bootstrap/configuration/v4.rs"),
    include_str!("rovent_bootstrap/pinned/external_r.rs"),
    include_str!("rovent_bootstrap/external_session.rs"),
    include_str!("rovent_bootstrap/pc_session.rs"),
    include_str!("rovent_bootstrap/product.rs"),
    include_str!("rovent_bootstrap/product/diagnostics.rs"),
    include_str!("rovent_bootstrap/product/diagnostics_v2.rs"),
    include_str!("rovent_bootstrap/product/diagnostics_v2_adaptive.rs"),
    include_str!("rovent_bootstrap/product/diagnostics_v2_nnuec.rs"),
    include_str!("rovent_bootstrap/product/diagnostics_v2_pc.rs"),
    include_str!("rovent_bootstrap/product/diagnostics/nnuec.rs"),
    include_str!("rovent_bootstrap/product/diagnostics/nnuec_cpu_slices.rs"),
    include_str!("rovent_bootstrap/product/diagnostics/nnuec_cpu_slices/chain.rs"),
    include_str!("rovent_bootstrap/product/storage.rs"),
    include_str!("rovent_bootstrap/storage_lifecycle.rs"),
    include_str!("rovent_bootstrap/storage_lifecycle/observation.rs"),
    include_str!("rovent_bootstrap/launch.rs"),
    include_str!("rovent_bootstrap/launch/selected_view_observation.rs"),
    include_str!("rovent_bootstrap/launch/collection.rs"),
    include_str!("rovent_bootstrap/launch/collection_v2_observer.rs"),
    include_str!("rovent_bootstrap/launch/evidence.rs"),
    include_str!("rovent_bootstrap/launch/storage.rs"),
    include_str!("rovent_bootstrap/launch/v2_external.rs"),
    include_str!("rovent_bootstrap/launch/v2_rejections.rs"),
    include_str!("rovent_bootstrap/nnuec_source.rs"),
    include_str!("rovent_bootstrap/nnuec_preparing.rs"),
    include_str!("rovent_bootstrap/nnuec_session.rs"),
    include_str!("rovent_bootstrap/nnuec_session/root_run.rs"),
    include_str!("rovent_bootstrap/nnuec_session/diagnostics.rs"),
    include_str!("rovent_bootstrap/nnuec_session/diagnostics/cpu_slices.rs"),
    include_str!("rovent_bootstrap/nnuec_session/retirement.rs"),
    include_str!("rovent_bootstrap/nnuec_session/initial_cleanup.rs"),
    include_str!("rovent_bootstrap/pinned/nnuec.rs"),
    include_str!("rovent_bootstrap/pinned/collection.rs"),
    include_str!("pals_adaptive_bootstrap/runtime_assets.rs"),
];

#[cfg(test)]
#[test]
fn bootstrap_source_identity_covers_selected_view_observer() {
    let source = include_str!("rovent_bootstrap/launch/selected_view_observation.rs");
    assert_eq!(
        BOOTSTRAP_SOURCES.iter().filter(|s| **s == source).count(),
        1
    );
    let without: Vec<_> = BOOTSTRAP_SOURCES
        .iter()
        .copied()
        .filter(|s| *s != source)
        .collect();
    assert_ne!(
        compiled_bootstrap_source_digest(),
        bootstrap_source_digest(&without)
    );
}
pub(crate) fn fail(kind: K, message: &'static str) -> BackendError {
    BackendError::new(kind, S::Admission, message)
}
pub(crate) fn hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests;
