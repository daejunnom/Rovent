//! 원 loaded native source와 실제 Frontier 부착을 함께 보존하는 수집 권한.
//! 이 파일은 wire DTO가 아니며 JSON/복사한 receipt에서 권한을 복원하지 않는다.
use super::*;
use rz_contracts::{Deadline, Digest};
use rz_position::contracts::ContractState;
use rz_search::pals::engine::adaptive_port::AdaptiveModelNamespace;
use rz_search::pals::rovent_engine::RootMoveDomain;
use rz_search::pals::rovent_frontier::RoventFrontier;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
use serde_json::{Value, json};
use std::sync::Arc;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
use std::sync::OnceLock;

pub const ROVENT_COLLECTION_NATIVE_ORIGIN: &str = "rz-rovent-collection-native-origin/1";
pub const ROVENT_COLLECTION_ROOT_ADMISSION: &str = "rz-rovent-collection-root-admission/1";
pub const ROVENT_COLLECTION_EXTERNAL_ROOT_ADMISSION: &str = "rz-rovent-collection-root-admission/2";
pub const ROVENT_COLLECTION_PROFILED_EXTERNAL_SOURCE: &str =
    "rz-rovent-collection-profiled-external-source/1";
/// These domains describe a /2 input-lock origin and attached root. The older
/// collection-root-admission/2 domain is already used by V1 external R.
pub const ROVENT_COLLECTION_NATIVE_ORIGIN_LOCK_V2: &str =
    "rz-rovent-collection-native-origin-lock-v2/1";
pub const ROVENT_COLLECTION_ROOT_ADMISSION_LOCK_V2: &str =
    "rz-rovent-collection-root-admission-lock-v2/1";
/// Distinct observed domains; serialized output does not create authority.
pub const ROVENT_COLLECTION_NATIVE_ORIGIN_LOCK_V3: &str =
    "rz-rovent-collection-native-origin-lock-v3/1";
pub const ROVENT_COLLECTION_ROOT_ADMISSION_LOCK_V3: &str =
    "rz-rovent-collection-root-admission-lock-v3/1";

fn reject(message: impl Into<String>) -> AdaptiveCollectionError {
    AdaptiveCollectionError::new(
        AdaptiveCollectionStage::Accepted,
        "RoventRootAdmission",
        message,
    )
}
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
fn text_sha(value: Digest) -> String {
    value.0.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
fn mode_wire_v2(value: rz_contracts::rovent::ExecutionMode) -> &'static str {
    use rz_contracts::rovent::ExecutionMode;
    match value {
        ExecutionMode::HybridPcr => "hybrid_pcr",
        ExecutionMode::HybridPc => "hybrid_pc",
        ExecutionMode::CpuPcr => "cpu_pcr",
        ExecutionMode::CpuPc => "cpu_pc",
    }
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
fn topology_wire_v2(value: crate::rovent_bootstrap::RoventStartupTopology) -> &'static str {
    use crate::rovent_bootstrap::RoventStartupTopology;
    match value {
        RoventStartupTopology::RolesOnly => "roles_only",
        RoventStartupTopology::OwnResearcher => "own_researcher",
        RoventStartupTopology::ExternalResearcher => "external_researcher",
    }
}

/// A /2 Collection origin is issued only for the original Ready native service
/// and the source owner that retained its /2 lock and Collection descriptors.
/// This is a live owner, not a reconstructed /1 origin or a receipt.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
#[derive(Clone)]
pub struct RoventNativeCollectionOriginV2 {
    original: Arc<LoadedOriginV2>,
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
struct VersionedLoadedOrigin<L, R> {
    sources: Arc<crate::rovent_bootstrap::RoventPinnedNativeSources<L>>,
    preload: Arc<crate::rovent_bootstrap::RoventCollectionPreloadSources>,
    process_epoch: u64,
    mode: rz_contracts::rovent::ExecutionMode,
    input_lock: Digest,
    namespace: AdaptiveModelNamespace,
    clock: rz_runtime::contracts::ContractSystemClock,
    _original_verifier: rz_search::pals::coordinator::NativeAcceptanceVerifier,
    // The actual successful load and Ready observation, not a reconstructed
    // snapshot from persisted declarations. It alone does not prove a run.
    snapshot: crate::rovent_bootstrap::RoventNativeStartupSnapshot,
    raw: Value,
    raw_sha256: String,
    first_root: OnceLock<Arc<R>>,
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
type LoadedOriginV2 = VersionedLoadedOrigin<
    rz_experiments::RoventInputLockV2,
    crate::rovent_root_admission::source_seal::RoventRootAdmissionV2,
>;

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu", target_os = "linux"))]
type LoadedOriginV3 = VersionedLoadedOrigin<
    rz_experiments::RoventInputLockV3,
    crate::rovent_root_admission::source_seal::RoventRootAdmissionV3,
>;

/// Shared checks bind a Collection origin to the real Ready service. A ready
/// event never proves inference, teacher completion or a physical GPU fence.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
#[allow(clippy::too_many_arguments)]
fn collection_ready_identity<Q: crate::pals_adaptive::NativeQuestionProjection>(
    service: &crate::rovent_bootstrap::RoventNativeService<Q>,
    snapshot: &crate::rovent_bootstrap::RoventNativeStartupSnapshot,
    engine: &rz_experiments::RoventEngineDeclaration,
    purpose: rz_experiments::RoventPurpose,
    input_lock: Digest,
    weights: [u8; 32],
    export_digest: &str,
    validator_required: bool,
    version: &str,
) -> Result<
    (
        rz_runtime::contracts::ContractSystemClock,
        rz_search::pals::coordinator::NativeAcceptanceVerifier,
        AdaptiveModelNamespace,
    ),
    AdaptiveCollectionError,
> {
    use rz_runtime::contracts::ContractClock;
    let clock = service.ready_clock().ok_or_else(|| {
        reject(format!(
            "original {version} native service has not reached Ready"
        ))
    })?;
    let verifier = service.native_acceptance_verifier().ok_or_else(|| {
        reject(format!(
            "original {version} native service has no worker verifier"
        ))
    })?;
    let pc = &engine.pc;
    if service.is_quarantined()
        || purpose != rz_experiments::RoventPurpose::Collection
        || pc.proposer.diagnostic
        || pc.critic.diagnostic
        || pc.validator.is_some() != validator_required
        || snapshot.validator_loaded != validator_required
        || snapshot.process_epoch == 0
        || clock.domain().0.0 != snapshot.process_epoch
        || snapshot.input_lock != input_lock
        || snapshot.mode != engine.mode
        || snapshot.weights.0 != weights
        || text_sha(snapshot.export_manifest) != export_digest
        || snapshot.physical_workers_created != 1
        || snapshot.neural_service_executed
    {
        return Err(reject(format!(
            "original {version} Ready, Collection source or snapshot differs"
        )));
    }
    let namespace = snapshot
        .model_namespace()
        .map_err(|error| reject(error.to_string()))?;
    Ok((clock, verifier, namespace))
}

// Both lock revisions must attach the actual same loaded owner and Frontier.
// The versions retain distinct typed locks and descriptor domains; no serialized
// capability is converted or reconstructed by this shared check.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
macro_rules! check_collection_frontier {
    ($loaded:expr, $preload:expr, $lock:expr, $frontier:expr, $root:expr, $budget:expr, $policy:expr, $version:expr) => {{
        use rz_runtime::contracts::ContractClock;
        let registration = $frontier.registration();
        let root_state = registration.root_rules.snapshot().identity();
        if !Arc::ptr_eq($preload, &$loaded.preload)
            || !Arc::ptr_eq($lock, $loaded.sources.lock_owner())
            || registration.mode.has_native_r() != registration.native_ledger.is_some()
            || $frontier.external_source().is_some()
            || registration.mode != $loaded.mode
            || registration.authority.epoch.0 != $loaded.process_epoch
            || registration.root_budget != $budget
            || registration.selective_policy != $policy
            || registration.model_namespace.as_ref() != &$loaded.namespace
            || root_state != registration.root_moves.root_identity()
            || !$root
                .original_context()
                .accepts()
                .map_err(|error| reject(error.to_string()))?
            || $root.startup_lock_digest() != $loaded.input_lock
            || $root.authority() != registration.authority
            || $root
                .descriptor()
                .root_budget_digest()
                .map_err(|error| reject(error.to_string()))?
                != $budget
            || $root.hard_deadline() != registration.original_deadline
            || $root.goal() != registration.goal
            || $root.requested_nodes() != registration.nodes
            || $root.runtime_clock().domain() != $loaded.clock.domain()
            || $root.runtime_clock().origin() != $loaded.clock.origin()
            || $root.state_and_history_digest() != root_state.semantic
            || $root.root_snapshot().revision() != root_state.revision.0
            || !$root
                .root_snapshot()
                .same_state(registration.root_rules.rules().snapshot())
            || $root.allowed_root_moves().as_ref() != registration.root_moves.allowed().as_ref()
        {
            return Err(reject(format!(
                "actual attached Frontier differs from original {} Collection scope",
                $version
            )));
        }
        $root
            .descriptor()
            .verify_against($lock)
            .map_err(|error| reject(error.to_string()))?;
        $root
            .allocation_clock()
            .verify_against($root.descriptor(), $lock)
            .map_err(|error| reject(error.to_string()))?;
        (registration, root_state)
    }};
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
macro_rules! collection_root_observation {
    ($schema:expr, $original:expr, $registration:expr, $root:expr, $state:expr, $budget:expr, $policy:expr, $descriptor:expr, $allocation:expr) => {
        json!({
            "schema": $schema,
            "native_origin": $original.raw_receipt(),
            "native_origin_sha256": $original.raw_sha256(),
            "process_epoch": $registration.authority.epoch.0,
            "game_generation": $registration.authority.game.0,
            "root_generation": $registration.authority.root.0,
            "implementation_sha256": text_sha($registration.authority.implementation),
            "deadline_clock": $registration.original_deadline.clock.0.0,
            "deadline_tick": $registration.original_deadline.at.0,
            "root_budget_sha256": text_sha($budget),
            "selective_policy_sha256": text_sha($policy),
            "root_state": {"owner": $state.owner.0,
                "revision": $state.revision.0, "semantic": text_sha($state.semantic)},
            "store_root_state": $registration.root_state.0,
            "root_moves": $registration.root_moves.allowed().iter()
                .map(|movement| movement.to_string()).collect::<Vec<_>>(),
            "depth_plies": $registration.goal.depth_plies.map(|value| value.get()),
            "mate_moves": $registration.goal.mate_moves.map(|value| value.get()),
            "requested_nodes": $registration.nodes,
            "root_descriptor": $root.descriptor(),
            "root_descriptor_sha256": text_sha($descriptor),
            "allocation_clock": $root.allocation_clock(),
            "allocation_clock_sha256": text_sha($allocation),
        })
    };
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl RoventNativeCollectionOriginV2 {
    /// Compare the original in-process origin owner, not copied metadata.
    /// This identity check grants no root, dispatch or completion authority.
    pub fn same_original_origin(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.original, &other.original)
    }
    /// Compare the actual loaded verifier and clock, not a reconstructed
    /// model or lock digest. Call before any collection hook can spawn work.
    pub(crate) fn matches_native_owner(
        &self,
        verifier: &rz_search::pals::coordinator::NativeAcceptanceVerifier,
        clock: &rz_runtime::contracts::ContractSystemClock,
    ) -> bool {
        use rz_runtime::contracts::ContractClock;
        self.original._original_verifier.same_owner(verifier)
            && self.original.clock.domain() == clock.domain()
            && self.original.clock.origin() == clock.origin()
    }

    /// Called from the actual Ready owner, never from serialized observations.
    pub(crate) fn from_loaded_ready<Q: crate::pals_adaptive::NativeQuestionProjection>(
        owner: crate::rovent_bootstrap::RoventCollectionReadyOwnerV2<'_, Q>,
    ) -> Result<Self, AdaptiveCollectionError> {
        let snapshot = owner.snapshot();
        let sources = Arc::clone(owner.sources());
        let service = owner.service();
        let preload = Arc::clone(
            sources
                .verify_collection_v2_origin()
                .map_err(|error| reject(error.to_string()))?,
        );
        let lock = sources.lock();
        let pc = &lock.manifest.engine.pc;
        let (clock, verifier, namespace) = collection_ready_identity(
            service,
            snapshot,
            &lock.manifest.engine,
            lock.manifest.purpose,
            sources.lock_digest(),
            sources.weights_digest(),
            &sources.export_digest_text(),
            false,
            "/2",
        )?;
        let (representation_model, representation_encoding, representation_precision, frozen_epoch) =
            namespace.representation_identity();
        let representation_precision = match representation_precision {
            rz_contracts::PrecisionProfile::Fp32 => "fp32",
            _ => return Err(reject("unsupported /2 collection representation precision")),
        };
        let source_sha256 = rz_experiments::rovent_source_aggregate_sha256(
            &lock.manifest.engine.source_commit,
            &lock.manifest.engine.source_components,
        )
        .map_err(|error| reject(error.to_string()))?;
        // Separate the startup observation from lock declarations. In
        // particular, a source declaration is not executable attestation and
        // a Ready event is not a completed neural request or GPU fence.
        let raw = json!({
            "schema": ROVENT_COLLECTION_NATIVE_ORIGIN_LOCK_V2,
            "input_lock_domain": rz_experiments::ROVENT_LOCK_V2_DOMAIN,
            "input_lock_sha256": text_sha(snapshot.input_lock),
            "declaration": {
                "purpose": "collection",
                "mode": mode_wire_v2(lock.manifest.engine.mode),
                "source_commit": lock.manifest.engine.source_commit,
                "source_components": lock.manifest.engine.source_components,
                "source_sha256_domain": rz_experiments::ROVENT_SOURCE_AGGREGATE_DOMAIN,
                "source_sha256": source_sha256,
                "model_semantics": pc.proposer.semantics,
                "model_semantics_sha256": text_sha(Digest(rz_eval::pals_model::PalsModelProfile::VirtualReviewV1.registered_semantics_sha256())),
                "encoding": pc.proposer.encoding,
                "encoding_sha256": text_sha(Digest({use sha2::Digest as _; sha2::Sha256::digest(rz_eval::pals_model::virtual_review::ENCODING.as_bytes()).into()})),
                "encoding_registry_sha256": text_sha(Digest(rz_eval::pals_model::virtual_review::registered_encoding_sha256())),
                "model_implementation_sha256": pc.proposer.implementation_sha256,
                "teacher_profiles": lock.manifest.engine.teachers,
            },
            "startup_observation": {
                "domain": snapshot.domain,
                "process_epoch": snapshot.process_epoch,
                "mode": mode_wire_v2(snapshot.mode),
                "topology": topology_wire_v2(snapshot.topology),
                "weights_sha256": text_sha(snapshot.weights),
                "export_manifest_sha256": text_sha(snapshot.export_manifest),
                "backend_sha256": text_sha(snapshot.backend),
                "runtime_library_sha256": text_sha(snapshot.runtime_library),
                "runtime_bundle_codec_sha256": snapshot.runtime_bundle.map(text_sha),
                "native_configuration_sha256": text_sha(snapshot.native_configuration),
                "startup_configuration_sha256": text_sha(snapshot.startup_configuration),
                "search_configuration_sha256": text_sha(snapshot.search_configuration),
                "validator_loaded": snapshot.validator_loaded,
                "physical_workers_created": snapshot.physical_workers_created,
                "neural_service_executed": snapshot.neural_service_executed,
            },
            "registered_namespace": {
                "value_semantics": namespace.value_identity().semantics,
                "value_model": namespace.value_identity().model,
                "value_encoding": namespace.value_identity().encoding,
                "value_precision": namespace.value_identity().precision,
                "representation_model_domain": "loaded_checkpoint_sha256",
                "representation_model_sha256": text_sha(representation_model),
                "representation_encoding_sha256": text_sha(representation_encoding),
                "representation_precision": representation_precision,
                "frozen_epoch": frozen_epoch,
            },
            "process_boundary_observation": "unknown",
            "neural_completion": "unknown",
            "physical_completion": "unknown",
        });
        let raw_sha256 = rz_experiments::pals_data_v3::generation_digest(
            ROVENT_COLLECTION_NATIVE_ORIGIN_LOCK_V2,
            &raw,
        )
        .map_err(|error| reject(error.to_string()))?;
        Ok(Self {
            original: Arc::new(LoadedOriginV2 {
                sources,
                preload,
                process_epoch: snapshot.process_epoch,
                mode: snapshot.mode,
                input_lock: snapshot.input_lock,
                namespace,
                clock,
                _original_verifier: verifier,
                snapshot: snapshot.clone(),
                raw,
                raw_sha256,
                first_root: OnceLock::new(),
            }),
        })
    }

    pub fn sources(
        &self,
    ) -> &Arc<crate::rovent_bootstrap::RoventPinnedNativeSources<rz_experiments::RoventInputLockV2>>
    {
        &self.original.sources
    }

    pub fn collection_preload(
        &self,
    ) -> &Arc<crate::rovent_bootstrap::RoventCollectionPreloadSources> {
        &self.original.preload
    }

    /// Original Ready observation. It is not a liveness, inference, process
    /// boundary or physical-completion attestation.
    pub fn startup_snapshot(&self) -> &crate::rovent_bootstrap::RoventNativeStartupSnapshot {
        &self.original.snapshot
    }

    /// Historical projection only; serialized metadata cannot reconstruct
    /// this origin or create a verifier, worker or live root capability.
    pub fn raw_receipt(&self) -> &Value {
        &self.original.raw
    }
    pub fn raw_sha256(&self) -> &str {
        &self.original.raw_sha256
    }

    /// Original /2 lock metadata, not the byte identity of its input file.
    pub fn original_input_lock(&self) -> &rz_experiments::RoventInputLockV2 {
        self.original.sources.lock()
    }

    /// Original retained bundle metadata bytes joined to the codec digest
    /// reported by this same Ready load. The artifact SHA is a separate file
    /// byte pin; neither digest attests execution or physical completion.
    pub fn runtime_bundle_publication_bytes(
        &self,
    ) -> Result<Option<(rz_experiments::ArtifactRef, Vec<u8>)>, AdaptiveCollectionError> {
        let original = self
            .original
            .sources
            .runtime_bundle_publication_bytes()
            .map_err(|error| reject(error.to_string()))?;
        match (&original, self.original.snapshot.runtime_bundle) {
            (Some((_, bytes)), Some(expected)) => {
                let text = std::str::from_utf8(bytes)
                    .map_err(|_| reject("original /2 bundle metadata is not UTF-8"))?;
                let spec = rz_eval::runtime_pin::CudaRuntimeBundleSpec::from_json(text)
                    .map_err(|error| reject(error.to_string()))?;
                let actual = Digest(spec.digest().map_err(|error| reject(error.to_string()))?);
                if actual != expected {
                    return Err(reject("original /2 bundle differs from Ready codec digest"));
                }
            }
            (None, None) => {}
            _ => return Err(reject("original /2 bundle metadata/Ready presence differs")),
        }
        Ok(original)
    }
}

/// A /3 origin borrows the actual Ready worker and retains its original source
/// and teacher preload. It is neither a /2 conversion nor completion proof.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu", target_os = "linux"))]
#[derive(Clone)]
pub struct RoventNativeCollectionOriginV3 {
    original: Arc<LoadedOriginV3>,
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu", target_os = "linux"))]
impl RoventNativeCollectionOriginV3 {
    pub(crate) fn from_loaded_ready<Q: crate::pals_adaptive::NativeQuestionProjection>(
        prepared: &crate::rovent_bootstrap::PreparedRoventNativeV3<Q>,
    ) -> Result<Self, AdaptiveCollectionError> {
        let snapshot = prepared.snapshot();
        let sources = Arc::clone(prepared.sources());
        let preload = Arc::clone(
            sources
                .verify_collection_v3_origin()
                .map_err(|error| reject(error.to_string()))?,
        );
        let lock = sources.lock();
        let pc = &lock.manifest.engine.pc;
        let (clock, verifier, namespace) = collection_ready_identity(
            prepared.collection_ready_service(),
            snapshot,
            &lock.manifest.engine,
            lock.manifest.purpose,
            sources.lock_digest(),
            sources.weights_digest(),
            &sources.export_digest_text(),
            pc.validator.is_some(),
            "/3",
        )?;
        let system = snapshot
            .system_runtime
            .as_ref()
            .filter(|observation| observation.provider_ready)
            .ok_or_else(|| reject("original /3 system CUDA provider is not Ready"))?;
        let (representation_model, representation_encoding, representation_precision, frozen_epoch) =
            namespace.representation_identity();
        let representation_precision = match representation_precision {
            rz_contracts::PrecisionProfile::Fp32 => "fp32",
            _ => return Err(reject("unsupported /3 collection representation precision")),
        };
        let raw = json!({
            "schema": ROVENT_COLLECTION_NATIVE_ORIGIN_LOCK_V3,
            "input_lock_domain": rz_experiments::ROVENT_LOCK_V3_DOMAIN,
            "input_lock_sha256": text_sha(snapshot.input_lock),
            "declaration": {
                "purpose": "collection",
                "mode": mode_wire_v2(lock.manifest.engine.mode),
                "source_commit": lock.manifest.engine.source_commit,
                "source_components": lock.manifest.engine.source_components,
                "model_semantics": pc.proposer.semantics,
                "encoding": pc.proposer.encoding,
                "validator_model": pc.validator,
                "teacher_profiles": lock.manifest.engine.teachers,
            },
            "startup_observation": {
                "domain": snapshot.domain,
                "process_epoch": snapshot.process_epoch,
                "mode": mode_wire_v2(snapshot.mode),
                "topology": topology_wire_v2(snapshot.topology),
                "weights_sha256": text_sha(snapshot.weights),
                "export_manifest_sha256": text_sha(snapshot.export_manifest),
                "backend_sha256": text_sha(snapshot.backend),
                "runtime_library_sha256": text_sha(snapshot.runtime_library),
                "system_runtime_identity_sha256": text_sha(system.runtime_identity),
                "system_libraries": system.libraries,
                "unregistered_native_mappings": system.unregistered_native_mappings,
                "provider_ready": system.provider_ready,
                "native_configuration_sha256": text_sha(snapshot.native_configuration),
                "startup_configuration_sha256": text_sha(snapshot.startup_configuration),
                "search_configuration_sha256": text_sha(snapshot.search_configuration),
                "validator_loaded": snapshot.validator_loaded,
                "physical_workers_created": snapshot.physical_workers_created,
                "neural_service_executed": snapshot.neural_service_executed,
            },
            "registered_namespace": {
                "value_semantics": namespace.value_identity().semantics,
                "value_model": namespace.value_identity().model,
                "value_encoding": namespace.value_identity().encoding,
                "value_precision": namespace.value_identity().precision,
                "representation_model_domain": "loaded_checkpoint_sha256",
                "representation_model_sha256": text_sha(representation_model),
                "representation_encoding_sha256": text_sha(representation_encoding),
                "representation_precision": representation_precision,
                "frozen_epoch": frozen_epoch,
            },
            "process_boundary_observation": "unknown",
            "neural_completion": "unknown",
            "physical_completion": "unknown",
        });
        let raw_sha256 = rz_experiments::pals_data_v3::generation_digest(
            ROVENT_COLLECTION_NATIVE_ORIGIN_LOCK_V3,
            &raw,
        )
        .map_err(|error| reject(error.to_string()))?;
        Ok(Self {
            original: Arc::new(LoadedOriginV3 {
                sources,
                preload,
                process_epoch: snapshot.process_epoch,
                mode: snapshot.mode,
                input_lock: snapshot.input_lock,
                namespace,
                clock,
                _original_verifier: verifier,
                snapshot: snapshot.clone(),
                raw,
                raw_sha256,
                first_root: OnceLock::new(),
            }),
        })
    }

    pub fn same_original_origin(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.original, &other.original)
    }
    pub(crate) fn matches_native_owner(
        &self,
        verifier: &rz_search::pals::coordinator::NativeAcceptanceVerifier,
        clock: &rz_runtime::contracts::ContractSystemClock,
    ) -> bool {
        use rz_runtime::contracts::ContractClock;
        self.original._original_verifier.same_owner(verifier)
            && self.original.clock.domain() == clock.domain()
            && self.original.clock.origin() == clock.origin()
    }
    pub fn sources(
        &self,
    ) -> &Arc<crate::rovent_bootstrap::RoventPinnedNativeSources<rz_experiments::RoventInputLockV3>>
    {
        &self.original.sources
    }
    pub fn collection_preload(
        &self,
    ) -> &Arc<crate::rovent_bootstrap::RoventCollectionPreloadSources> {
        &self.original.preload
    }
    pub fn startup_snapshot(&self) -> &crate::rovent_bootstrap::RoventNativeStartupSnapshot {
        &self.original.snapshot
    }
    pub fn raw_receipt(&self) -> &Value {
        &self.original.raw
    }
    pub fn raw_sha256(&self) -> &str {
        &self.original.raw_sha256
    }
    pub fn original_input_lock(&self) -> &rz_experiments::RoventInputLockV3 {
        self.original.sources.lock()
    }
}

/// A /2 root capability issued only at the successful real Frontier attach.
/// It retains original owners; it is not a physical completion or label proof.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
#[derive(Clone)]
pub struct RoventCollectionRootAdmissionV2 {
    original: RoventNativeCollectionOriginV2,
    root_admission: Arc<crate::rovent_root_admission::source_seal::RoventRootAdmissionV2>,
    rules: Arc<ContractState>,
    root_moves: Arc<RootMoveDomain>,
    frontier_root_path: rz_search::pals::rovent_frontier::FrontierPath,
    native_ledger: Option<rz_search::cpu_worker::CpuLedgerIdentity>,
    root_state: rz_contracts::StateIdentity,
    root_budget: Digest,
    selective_policy: Digest,
    raw: Arc<Value>,
    raw_sha256: Arc<str>,
}

/// A nonblocking ticket check distinguishes temporary lock contention from
/// a rejected, expired, malformed, or poisoned original authority.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventTicketValidationKindV2 {
    Busy,
    Revoked,
    Poisoned,
    Invalid,
    Expired,
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub struct RoventTicketValidationErrorV2 {
    pub kind: RoventTicketValidationKindV2,
    pub error: AdaptiveCollectionError,
    pub source: Option<rz_contracts::ContractError>,
}

// Ticket error classes describe lock/authority state only; they do not hold
// or convert a V2 root. The same semantics apply to V3.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub type RoventTicketValidationKindV3 = RoventTicketValidationKindV2;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub type RoventTicketValidationErrorV3 = RoventTicketValidationErrorV2;

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
fn try_validate_collection_ticket(
    context: &crate::search_driver::SearchSessionContext,
    authority: rz_contracts::pals::SearchAuthority,
    deadline: Deadline,
    root_state: rz_contracts::StateIdentity,
    ticket: &RoleTicketEnvelope,
    validator_allowed: bool,
    mismatch_message: &'static str,
) -> Result<(), RoventTicketValidationErrorV2> {
    let fail = |kind, message| RoventTicketValidationErrorV2 {
        kind,
        error: reject(message),
        source: None,
    };
    let current = context.current.try_lock().map_err(|error| {
        let (kind, code, message) = match error {
            std::sync::TryLockError::WouldBlock => (
                RoventTicketValidationKindV2::Busy,
                rz_contracts::ErrorCode::ResourceExhausted,
                "UCI search authority is busy",
            ),
            std::sync::TryLockError::Poisoned(_) => (
                RoventTicketValidationKindV2::Poisoned,
                rz_contracts::ErrorCode::BackendFailure,
                "UCI search authority poisoned",
            ),
        };
        let source =
            rz_contracts::ContractError::new(code, rz_contracts::Stage::Admission, message);
        RoventTicketValidationErrorV2 {
            kind,
            error: reject(source.to_string()),
            source: Some(source),
        }
    })?;
    if *current != crate::contracts::SessionScope::Search(context.authority)
        || context.cancellation.is_canceled()
        || context
            .control
            .cancellation
            .load(std::sync::atomic::Ordering::Acquire)
    {
        return Err(fail(
            RoventTicketValidationKindV2::Revoked,
            "native ticket original authority revoked",
        ));
    }
    if std::time::Instant::now() >= context.control.deadline {
        return Err(fail(
            RoventTicketValidationKindV2::Expired,
            "native ticket original authority expired",
        ));
    }
    if ticket.authority != authority
        || ticket.original_deadline != deadline
        || ticket.view.root_state != root_state
        || (ticket.role == Role::Validator && !validator_allowed)
    {
        return Err(fail(
            RoventTicketValidationKindV2::Invalid,
            mismatch_message,
        ));
    }
    Ok(())
}

/// Identity-only late-result checks intentionally do not require a live root.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
fn validate_collection_discard<V, Q, P>(
    context: &crate::pals_native::ticketed::TicketOwnerDispatchContext<'_, V, Q, P>,
    verifier: &rz_search::pals::coordinator::NativeAcceptanceVerifier,
    authority: rz_contracts::pals::SearchAuthority,
    deadline: Deadline,
    root_state: rz_contracts::StateIdentity,
    validator_allowed: bool,
    mismatch_message: &'static str,
) -> Result<(), AdaptiveCollectionError> {
    let ticket = context.ticket();
    if !context.same_owner(verifier)
        || ticket.authority != authority
        || ticket.original_deadline != deadline
        || ticket.view.root_state != root_state
        || (ticket.role == Role::Validator && !validator_allowed)
    {
        return Err(reject(mismatch_message));
    }
    Ok(())
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
fn validate_collection_ranking(
    ranking: &ProposerRankingPublished,
    rules: &ContractState,
    validate_ticket: impl FnOnce(&RoleTicketEnvelope) -> Result<(), AdaptiveCollectionError>,
    mismatch_message: &'static str,
) -> Result<(), AdaptiveCollectionError> {
    validate_ticket(&ranking.source().ticket)?;
    if ranking.source().ticket.role != Role::Proposer || !std::ptr::eq(ranking.rules(), rules) {
        return Err(reject(mismatch_message));
    }
    Ok(())
}

// Use the exact same claim sequence for both domains: check the original
// live root and scope issuer, consume the actual Start's one-shot claim, and
// recheck revocation while returning the original scope on every failure.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
macro_rules! claim_collection_teacher_scope {
    ($admission:ident, $scope:ident, $failure:ident, $capability:ident) => {{
        match $admission.root_admission.original_context().accepts() {
            Ok(true) => {}
            Ok(false) => {
                return Err($failure::BeforeClaim {
                    error: reject("teacher scope root is revoked"),
                    scope: $scope,
                });
            }
            Err(error) => {
                return Err($failure::BeforeClaim {
                    error: reject(error.to_string()),
                    scope: $scope,
                });
            }
        }
        let root_digest = match $admission.root_admission.descriptor().digest() {
            Ok(digest) => digest,
            Err(error) => {
                return Err($failure::BeforeClaim {
                    error: reject(error.to_string()),
                    scope: $scope,
                });
            }
        };
        if !$scope.is_unissued_for_root(
            $admission.root_admission.authority(),
            $admission.root_admission.hard_deadline(),
            $admission.root_admission.startup_lock_digest(),
            root_digest,
            $admission.root_budget,
        ) {
            return Err($failure::BeforeClaim {
                error: reject("teacher scope is shared, used, closed, or belongs to another root"),
                scope: $scope,
            });
        }
        if !$admission.root_admission.claim_teacher_scope_once() {
            return Err($failure::BeforeClaim {
                error: reject("teacher scope already claimed for this original root"),
                scope: $scope,
            });
        }
        let capability = $capability {
            admission: $admission.clone(),
            core: TeacherScopeCore::new($scope),
        };
        match $admission.root_admission.original_context().accepts() {
            Ok(true) => {}
            Ok(false) => {
                return Err($failure::AfterClaim {
                    error: reject("teacher scope root revoked during claim"),
                    capability,
                });
            }
            Err(error) => {
                return Err($failure::AfterClaim {
                    error: reject(error.to_string()),
                    capability,
                });
            }
        }
        Ok(capability)
    }};
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl RoventCollectionRootAdmissionV2 {
    /// The original native root's clock domain, already checked against the
    /// attached source clock at admission. This is identity only and does not
    /// itself authorize or advance a request.
    pub fn clock_domain(&self) -> rz_contracts::ClockDomain {
        self.root_admission.runtime_clock().domain()
    }

    /// Consume the original one-shot claim without creating a teacher scope.
    /// No compliant teacher scope was issued for this Start; this says nothing
    /// about teacher success or any unrelated OS process.
    pub fn seal_unclaimed_teacher_scope(
        &self,
    ) -> Result<RoventTeacherScopePhysicalAckV2, AdaptiveCollectionError> {
        if !self.root_admission.claim_teacher_scope_once() {
            return Err(reject("teacher scope was already claimed or sealed"));
        }
        Ok(RoventTeacherScopePhysicalAckV2 {
            admission: self.clone(),
            evidence: TeacherScopePhysicalEvidenceV2::NoScope,
        })
    }

    /// Claim this exact live root's sole teacher process scope. A claim is
    /// never reissued, including after failed or physically unknown work.
    /// Do all reversible roster/source preflight before calling this method.
    pub fn claim_teacher_scope(
        &self,
        scope: rz_search::external_cpu::process_scope::CollectionProcessScope,
    ) -> Result<RoventTeacherScopeCapabilityV2, RoventTeacherScopeClaimFailureV2> {
        claim_collection_teacher_scope!(
            self,
            scope,
            RoventTeacherScopeClaimFailureV2,
            RoventTeacherScopeCapabilityV2
        )
    }
    /// Call immediately after the real port has attached the Frontier and
    /// checked its Rules, Store, namespace, restriction and ledger owners.
    pub(crate) fn from_attached_frontier(
        original: &RoventNativeCollectionOriginV2,
        frontier: &RoventFrontier,
        root_admission: Arc<crate::rovent_root_admission::source_seal::RoventRootAdmissionV2>,
        root_budget: Digest,
        selective_policy: Digest,
    ) -> Result<Self, AdaptiveCollectionError> {
        let loaded = &original.original;
        let preload = loaded
            .sources
            .verify_collection_v2_origin()
            .map_err(|error| reject(error.to_string()))?;
        let lock = root_admission
            .owner()
            .v2()
            .ok_or_else(|| reject("Collection /2 root has no original /2 lock"))?;
        let (registration, root_state) = check_collection_frontier!(
            loaded,
            preload,
            lock,
            frontier,
            root_admission,
            root_budget,
            selective_policy,
            "/2"
        );
        let descriptor_digest = root_admission
            .descriptor()
            .digest()
            .map_err(|error| reject(error.to_string()))?;
        let allocation_digest = root_admission
            .allocation_clock()
            .digest()
            .map_err(|error| reject(error.to_string()))?;
        let raw = collection_root_observation!(
            ROVENT_COLLECTION_ROOT_ADMISSION_LOCK_V2,
            original,
            registration,
            root_admission,
            root_state,
            root_budget,
            selective_policy,
            descriptor_digest,
            allocation_digest
        );
        let raw_sha256 = rz_experiments::pals_data_v3::generation_digest(
            ROVENT_COLLECTION_ROOT_ADMISSION_LOCK_V2,
            &raw,
        )
        .map_err(|error| reject(error.to_string()))?;
        if !root_admission
            .original_context()
            .accepts()
            .map_err(|error| reject(error.to_string()))?
        {
            return Err(reject("original /2 root revoked during attach"));
        }
        let _ = loaded.first_root.set(Arc::clone(&root_admission));
        Ok(Self {
            original: original.clone(),
            root_admission,
            rules: Arc::clone(&registration.root_rules),
            root_moves: Arc::clone(&registration.root_moves),
            frontier_root_path: frontier.root_path(),
            native_ledger: registration.native_ledger.clone(),
            root_state,
            root_budget,
            selective_policy,
            raw: Arc::new(raw),
            raw_sha256: raw_sha256.into(),
        })
    }

    pub fn root_admission(
        &self,
    ) -> &Arc<crate::rovent_root_admission::source_seal::RoventRootAdmissionV2> {
        &self.root_admission
    }

    pub fn native_origin(&self) -> &RoventNativeCollectionOriginV2 {
        &self.original
    }

    pub fn rules(&self) -> &Arc<ContractState> {
        &self.rules
    }
    pub fn root_moves(&self) -> &Arc<RootMoveDomain> {
        &self.root_moves
    }
    /// Compare this issued admission or one of its clones. A separately
    /// issued origin is not equivalent merely because it uses the same worker.
    /// A match does not establish current liveness or physical completion.
    pub fn same_original_admission(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.original.original, &other.original.original)
            && Arc::ptr_eq(&self.root_admission, &other.root_admission)
            && Arc::ptr_eq(&self.rules, &other.rules)
            && Arc::ptr_eq(&self.root_moves, &other.root_moves)
            && self.frontier_root_path == other.frontier_root_path
            && self.native_ledger == other.native_ledger
    }
    /// Match only the same attached in-process frontier and ledger; this
    /// comparison neither recreates them nor certifies later liveness.
    pub(crate) fn validate_frontier(
        &self,
        frontier: &RoventFrontier,
    ) -> Result<(), AdaptiveCollectionError> {
        if frontier.root_path() != self.frontier_root_path
            || frontier.registration().native_ledger != self.native_ledger
        {
            return Err(reject("frontier or ledger differs from original /2 attach"));
        }
        Ok(())
    }
    pub fn root_budget(&self) -> Digest {
        self.root_budget
    }
    pub fn selective_policy(&self) -> Digest {
        self.selective_policy
    }

    /// Metadata captured at actual attach; this cannot recreate a live root.
    pub fn raw_receipt(&self) -> &Value {
        &self.raw
    }
    pub fn raw_sha256(&self) -> &str {
        &self.raw_sha256
    }

    /// Match the manager's verifier to the original Ready worker. Matching a
    /// worker is not proof of current liveness, output acceptance, or completion.
    pub(crate) fn validate_native_owner(
        &self,
        verifier: &rz_search::pals::coordinator::NativeAcceptanceVerifier,
    ) -> Result<(), AdaptiveCollectionError> {
        if !self
            .root_admission
            .original_context()
            .accepts()
            .map_err(|error| reject(error.to_string()))?
        {
            return Err(reject("original /2 root no longer accepts"));
        }
        self.validate_native_identity(verifier)
    }

    /// Identity only: physical events may arrive after root revocation.
    pub(super) fn validate_native_identity(
        &self,
        verifier: &rz_search::pals::coordinator::NativeAcceptanceVerifier,
    ) -> Result<(), AdaptiveCollectionError> {
        if !self
            .original
            .original
            ._original_verifier
            .same_owner(verifier)
        {
            return Err(reject(
                "native verifier is not the original /2 worker verifier",
            ));
        }
        Ok(())
    }

    /// Check the real service-issued dispatch against this root and its
    /// original worker. This is not native completion or output acceptance.
    pub fn validate_native_dispatch_owner<V, Q, P>(
        &self,
        context: &crate::pals_native::ticketed::TicketOwnerDispatchContext<'_, V, Q, P>,
    ) -> Result<(), AdaptiveCollectionError> {
        self.validate_ticket(context.ticket())?;
        if !context.same_owner(&self.original.original._original_verifier) {
            return Err(reject("dispatch does not belong to the original /2 worker"));
        }
        Ok(())
    }

    /// Check only the original worker and immutable ticket identity when
    /// discarding a late capture. This intentionally does not assert that the
    /// root is live and grants no acceptance, target, or physical-completion
    /// authority. It must not be used to admit a new request.
    pub fn validate_native_discard_owner<V, Q, P>(
        &self,
        context: &crate::pals_native::ticketed::TicketOwnerDispatchContext<'_, V, Q, P>,
    ) -> Result<(), AdaptiveCollectionError> {
        validate_collection_discard(
            context,
            &self.original.original._original_verifier,
            self.root_admission.authority(),
            self.root_admission.hard_deadline(),
            self.root_state,
            false,
            "discard ticket is outside the original /2 worker or root",
        )
    }

    pub fn validate_ticket(
        &self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(), AdaptiveCollectionError> {
        if !self
            .root_admission
            .original_context()
            .accepts()
            .map_err(|error| reject(error.to_string()))?
            || ticket.authority != self.root_admission.authority()
            || ticket.original_deadline != self.root_admission.hard_deadline()
            || ticket.view.root_state != self.root_state
            || ticket.role == Role::Validator
        {
            return Err(reject(
                "native ticket is outside the original admitted /2 Collection root",
            ));
        }
        Ok(())
    }

    /// Nonblocking equivalent of `validate_ticket`: the same original root,
    /// cancellation, control deadline and ticket checks remain mandatory.
    /// Busy means no claim was made and may be retried while the root lives.
    pub fn try_validate_ticket(
        &self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(), RoventTicketValidationErrorV2> {
        try_validate_collection_ticket(
            self.root_admission.original_context(),
            self.root_admission.authority(),
            self.root_admission.hard_deadline(),
            self.root_state,
            ticket,
            false,
            "native ticket is outside the original admitted /2 Collection root",
        )
    }

    /// Validate against the namespace of the same original Ready worker and
    /// the still-live /2 root, without reconstructing a /1 capability.
    pub fn validate_output(
        &self,
        ticket: &RoleTicketEnvelope,
        output: &rz_contracts::pals::RoleOutput,
    ) -> Result<(), AdaptiveCollectionError> {
        self.validate_ticket(ticket)?;
        let namespace = &self.original.original.namespace;
        namespace
            .verify(ticket, output, Some(namespace.value_identity()))
            .map_err(|error| reject(error.to_string()))
    }

    /// A ranking belongs to this root only if it uses the original Rules
    /// owner. This is a check, not a publication or teacher authority.
    pub(super) fn validate_ranking(
        &self,
        ranking: &ProposerRankingPublished,
    ) -> Result<(), AdaptiveCollectionError> {
        validate_collection_ranking(
            ranking,
            self.rules.as_ref(),
            |ticket| self.validate_ticket(ticket),
            "initial P publication is not from the original admitted /2 Rules owner",
        )
    }
}

/// V3 root authority can only be issued from the actual attached Frontier and
/// retains the original /3 lock and Ready worker. It does not grant teacher
/// scope or imply physical completion.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu", target_os = "linux"))]
#[derive(Clone)]
pub struct RoventCollectionRootAdmissionV3 {
    original: RoventNativeCollectionOriginV3,
    root_admission: Arc<crate::rovent_root_admission::source_seal::RoventRootAdmissionV3>,
    rules: Arc<ContractState>,
    root_moves: Arc<RootMoveDomain>,
    frontier_root_path: rz_search::pals::rovent_frontier::FrontierPath,
    native_ledger: Option<rz_search::cpu_worker::CpuLedgerIdentity>,
    root_state: rz_contracts::StateIdentity,
    root_budget: Digest,
    selective_policy: Digest,
    raw: Arc<Value>,
    raw_sha256: Arc<str>,
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu", target_os = "linux"))]
impl RoventCollectionRootAdmissionV3 {
    pub fn clock_domain(&self) -> rz_contracts::ClockDomain {
        use rz_runtime::contracts::ContractClock;
        self.root_admission.runtime_clock().domain()
    }
    /// Seal only the original Start's one-shot claim; this is no-scope
    /// evidence and says nothing about unrelated processes or success.
    pub fn seal_unclaimed_teacher_scope(
        &self,
    ) -> Result<RoventTeacherScopePhysicalAckV3, AdaptiveCollectionError> {
        if !self.root_admission.claim_teacher_scope_once() {
            return Err(reject("teacher scope was already claimed or sealed"));
        }
        Ok(RoventTeacherScopePhysicalAckV3 {
            admission: self.clone(),
            evidence: TeacherScopePhysicalEvidenceV2::NoScope,
        })
    }
    pub fn claim_teacher_scope(
        &self,
        scope: rz_search::external_cpu::process_scope::CollectionProcessScope,
    ) -> Result<RoventTeacherScopeCapabilityV3, RoventTeacherScopeClaimFailureV3> {
        claim_collection_teacher_scope!(
            self,
            scope,
            RoventTeacherScopeClaimFailureV3,
            RoventTeacherScopeCapabilityV3
        )
    }
    pub(crate) fn from_attached_frontier(
        original: &RoventNativeCollectionOriginV3,
        frontier: &RoventFrontier,
        root_admission: Arc<crate::rovent_root_admission::source_seal::RoventRootAdmissionV3>,
        root_budget: Digest,
        selective_policy: Digest,
    ) -> Result<Self, AdaptiveCollectionError> {
        let loaded = &original.original;
        let preload = loaded
            .sources
            .verify_collection_v3_origin()
            .map_err(|error| reject(error.to_string()))?;
        let lock = root_admission
            .owner()
            .v3()
            .ok_or_else(|| reject("Collection /3 root has no original /3 lock"))?;
        let (registration, root_state) = check_collection_frontier!(
            loaded,
            preload,
            lock,
            frontier,
            root_admission,
            root_budget,
            selective_policy,
            "/3"
        );
        let descriptor_digest = root_admission
            .descriptor()
            .digest()
            .map_err(|error| reject(error.to_string()))?;
        let allocation_digest = root_admission
            .allocation_clock()
            .digest()
            .map_err(|error| reject(error.to_string()))?;
        let raw = collection_root_observation!(
            ROVENT_COLLECTION_ROOT_ADMISSION_LOCK_V3,
            original,
            registration,
            root_admission,
            root_state,
            root_budget,
            selective_policy,
            descriptor_digest,
            allocation_digest
        );
        let raw_sha256 = rz_experiments::pals_data_v3::generation_digest(
            ROVENT_COLLECTION_ROOT_ADMISSION_LOCK_V3,
            &raw,
        )
        .map_err(|error| reject(error.to_string()))?;
        if !root_admission
            .original_context()
            .accepts()
            .map_err(|error| reject(error.to_string()))?
        {
            return Err(reject("original /3 root revoked during attach"));
        }
        let _ = loaded.first_root.set(Arc::clone(&root_admission));
        Ok(Self {
            original: original.clone(),
            root_admission,
            rules: Arc::clone(&registration.root_rules),
            root_moves: Arc::clone(&registration.root_moves),
            frontier_root_path: frontier.root_path(),
            native_ledger: registration.native_ledger.clone(),
            root_state,
            root_budget,
            selective_policy,
            raw: Arc::new(raw),
            raw_sha256: raw_sha256.into(),
        })
    }

    pub fn root_admission(
        &self,
    ) -> &Arc<crate::rovent_root_admission::source_seal::RoventRootAdmissionV3> {
        &self.root_admission
    }
    pub fn native_origin(&self) -> &RoventNativeCollectionOriginV3 {
        &self.original
    }
    pub fn rules(&self) -> &Arc<ContractState> {
        &self.rules
    }
    pub fn root_moves(&self) -> &Arc<RootMoveDomain> {
        &self.root_moves
    }
    pub fn root_budget(&self) -> Digest {
        self.root_budget
    }
    pub fn selective_policy(&self) -> Digest {
        self.selective_policy
    }
    pub fn raw_receipt(&self) -> &Value {
        &self.raw
    }
    pub fn raw_sha256(&self) -> &str {
        &self.raw_sha256
    }
    pub fn same_original_admission(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.original.original, &other.original.original)
            && Arc::ptr_eq(&self.root_admission, &other.root_admission)
            && Arc::ptr_eq(&self.rules, &other.rules)
            && Arc::ptr_eq(&self.root_moves, &other.root_moves)
            && self.frontier_root_path == other.frontier_root_path
            && self.native_ledger == other.native_ledger
    }
    pub(crate) fn validate_frontier(
        &self,
        frontier: &RoventFrontier,
    ) -> Result<(), AdaptiveCollectionError> {
        if frontier.root_path() != self.frontier_root_path
            || frontier.registration().native_ledger != self.native_ledger
        {
            return Err(reject("frontier or ledger differs from original /3 attach"));
        }
        Ok(())
    }
    /// This validates the original Ready worker as well as current root
    /// liveness. It does not establish physical completion.
    pub(crate) fn validate_native_owner(
        &self,
        verifier: &rz_search::pals::coordinator::NativeAcceptanceVerifier,
    ) -> Result<(), AdaptiveCollectionError> {
        if !self
            .root_admission
            .original_context()
            .accepts()
            .map_err(|error| reject(error.to_string()))?
        {
            return Err(reject("original /3 root no longer accepts"));
        }
        self.validate_native_identity(verifier)
    }
    /// Identity only: physical events may arrive after root revocation.
    pub(super) fn validate_native_identity(
        &self,
        verifier: &rz_search::pals::coordinator::NativeAcceptanceVerifier,
    ) -> Result<(), AdaptiveCollectionError> {
        if !self
            .original
            .original
            ._original_verifier
            .same_owner(verifier)
        {
            return Err(reject(
                "native verifier differs from original /3 Ready worker",
            ));
        }
        Ok(())
    }
    pub fn validate_native_dispatch_owner<V, Q, P>(
        &self,
        context: &crate::pals_native::ticketed::TicketOwnerDispatchContext<'_, V, Q, P>,
    ) -> Result<(), AdaptiveCollectionError> {
        self.validate_ticket(context.ticket())?;
        if !context.same_owner(&self.original.original._original_verifier) {
            return Err(reject("dispatch does not belong to the original /3 worker"));
        }
        Ok(())
    }
    /// Check original identity only; root revocation cannot prevent discarding
    /// a late result, and this check does not grant new request authority.
    pub fn validate_native_discard_owner<V, Q, P>(
        &self,
        context: &crate::pals_native::ticketed::TicketOwnerDispatchContext<'_, V, Q, P>,
    ) -> Result<(), AdaptiveCollectionError> {
        validate_collection_discard(
            context,
            &self.original.original._original_verifier,
            self.root_admission.authority(),
            self.root_admission.hard_deadline(),
            self.root_state,
            self.original.original.snapshot.validator_loaded,
            "discard ticket is outside the original /3 worker or root",
        )
    }
    pub fn validate_ticket(
        &self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(), AdaptiveCollectionError> {
        if !self
            .root_admission
            .original_context()
            .accepts()
            .map_err(|error| reject(error.to_string()))?
            || ticket.authority != self.root_admission.authority()
            || ticket.original_deadline != self.root_admission.hard_deadline()
            || ticket.view.root_state != self.root_state
            || (ticket.role == Role::Validator && !self.original.original.snapshot.validator_loaded)
        {
            return Err(reject(
                "native ticket is outside the original admitted /3 Collection root",
            ));
        }
        Ok(())
    }
    /// Busy and poisoned authority are explicit; this never blocks a manager.
    pub fn try_validate_ticket(
        &self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(), RoventTicketValidationErrorV3> {
        try_validate_collection_ticket(
            self.root_admission.original_context(),
            self.root_admission.authority(),
            self.root_admission.hard_deadline(),
            self.root_state,
            ticket,
            self.original.original.snapshot.validator_loaded,
            "native ticket is outside the original admitted /3 Collection root",
        )
    }
    pub fn validate_output(
        &self,
        ticket: &RoleTicketEnvelope,
        output: &rz_contracts::pals::RoleOutput,
    ) -> Result<(), AdaptiveCollectionError> {
        self.validate_ticket(ticket)?;
        let namespace = &self.original.original.namespace;
        namespace
            .verify(ticket, output, Some(namespace.value_identity()))
            .map_err(|error| reject(error.to_string()))
    }
    pub(super) fn validate_ranking(
        &self,
        ranking: &ProposerRankingPublished,
    ) -> Result<(), AdaptiveCollectionError> {
        validate_collection_ranking(
            ranking,
            self.rules.as_ref(),
            |ticket| self.validate_ticket(ticket),
            "initial P publication is not from the original admitted /3 Rules owner",
        )
    }
}

/// Non-cloneable authority for one original root's teacher scope. Closing an
/// old capability only touches its own root even when a later generation has
/// already started. It is not a physical-retirement or publication proof.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub enum RoventTeacherScopeClaimFailureV2 {
    BeforeClaim {
        error: AdaptiveCollectionError,
        scope: rz_search::external_cpu::process_scope::CollectionProcessScope,
    },
    /// The one-shot claim was consumed. The original scope must be retained
    /// through close and physical outcome; it cannot be reissued.
    AfterClaim {
        error: AdaptiveCollectionError,
        capability: RoventTeacherScopeCapabilityV2,
    },
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
enum TeacherScopePhysicalEvidenceV2 {
    NoScope,
    Scope(rz_search::external_cpu::process_scope::CollectionScopePhysicalOutcome),
}

/// Non-cloneable acknowledgement bound to the actual original admission. It
/// represents physical no-scope/no-child or confirmed exit only; it does not
/// certify teacher success, consumption, model output or publication.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub struct RoventTeacherScopePhysicalAckV2 {
    admission: RoventCollectionRootAdmissionV2,
    evidence: TeacherScopePhysicalEvidenceV2,
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl RoventTeacherScopePhysicalAckV2 {
    pub(crate) fn original_admission(&self) -> &RoventCollectionRootAdmissionV2 {
        &self.admission
    }
    pub fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV2) -> bool {
        self.admission.same_original_admission(admission)
    }

    /// None is the sealed no-scope outcome, not a process-exit report.
    pub fn scope_outcome(
        &self,
    ) -> Option<&rz_search::external_cpu::process_scope::CollectionScopePhysicalOutcome> {
        match &self.evidence {
            TeacherScopePhysicalEvidenceV2::NoScope => None,
            TeacherScopePhysicalEvidenceV2::Scope(outcome) => Some(outcome),
        }
    }

    /// A sealed no-scope Start cannot have a compliant scoped teacher worker.
    pub fn is_sealed_no_scope(&self) -> bool {
        matches!(&self.evidence, TeacherScopePhysicalEvidenceV2::NoScope)
    }
}

/// One original scope's joined/no-thread worker roster. This says nothing
/// about OS child completion, terminal delivery or teacher semantic success.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub struct RoventTeacherScopeWorkerAckV2 {
    admission: RoventCollectionRootAdmissionV2,
    outcome: rz_search::external_cpu::process_scope::CollectionScopeWorkerOutcome,
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl RoventTeacherScopeWorkerAckV2 {
    pub(crate) fn original_admission(&self) -> &RoventCollectionRootAdmissionV2 {
        &self.admission
    }
    pub fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV2) -> bool {
        self.admission.same_original_admission(admission)
    }

    pub fn outcome(&self) -> &rz_search::external_cpu::process_scope::CollectionScopeWorkerOutcome {
        &self.outcome
    }
}

// Shared physical scope lifetime. Versioned wrappers retain their original
// typed admission separately; this core cannot create a root or an ACK.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
struct TeacherScopeCore {
    scope: std::sync::Mutex<Option<rz_search::external_cpu::process_scope::CollectionProcessScope>>,
    physical_issued: std::sync::atomic::AtomicBool,
    worker_issued: std::sync::atomic::AtomicBool,
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl TeacherScopeCore {
    fn new(scope: rz_search::external_cpu::process_scope::CollectionProcessScope) -> Self {
        Self {
            scope: std::sync::Mutex::new(Some(scope)),
            physical_issued: std::sync::atomic::AtomicBool::new(false),
            worker_issued: std::sync::atomic::AtomicBool::new(false),
        }
    }
    fn close(&self) -> Result<(), AdaptiveCollectionError> {
        let slot = self.scope.try_lock().map_err(|error| match error {
            std::sync::TryLockError::WouldBlock => AdaptiveCollectionError::new(
                AdaptiveCollectionStage::Settled,
                "TeacherScopeBusy",
                "teacher scope close is temporarily busy",
            ),
            std::sync::TryLockError::Poisoned(_) => AdaptiveCollectionError::new(
                AdaptiveCollectionStage::Settled,
                "TeacherScopePoisoned",
                "teacher scope close is poisoned",
            ),
        })?;
        if let Some(scope) = slot.as_ref() {
            scope.close_admission();
        }
        Ok(())
    }
    fn try_physical_retirement(
        &self,
    ) -> Result<
        Option<rz_search::external_cpu::process_scope::CollectionScopePhysicalOutcome>,
        AdaptiveCollectionError,
    > {
        let mut slot = self.scope.try_lock().map_err(|error| match error {
            std::sync::TryLockError::WouldBlock => AdaptiveCollectionError::new(
                AdaptiveCollectionStage::Settled,
                "TeacherScopeBusy",
                "teacher scope retirement is temporarily busy",
            ),
            std::sync::TryLockError::Poisoned(_) => AdaptiveCollectionError::new(
                AdaptiveCollectionStage::Settled,
                "TeacherScopePoisoned",
                "teacher scope retirement is poisoned",
            ),
        })?;
        let scope = slot
            .as_ref()
            .ok_or_else(|| reject("teacher scope retirement already issued"))?;
        let Some(outcome) = scope
            .try_physical_retirement()
            .map_err(|error| reject(error.to_string()))?
        else {
            return Ok(None);
        };
        self.physical_issued
            .store(true, std::sync::atomic::Ordering::Release);
        if self
            .worker_issued
            .load(std::sync::atomic::Ordering::Acquire)
        {
            drop(slot.take());
        }
        Ok(Some(outcome))
    }
    fn try_worker_retirement(
        &self,
    ) -> Result<
        Option<rz_search::external_cpu::process_scope::CollectionScopeWorkerOutcome>,
        AdaptiveCollectionError,
    > {
        let mut slot = self.scope.try_lock().map_err(|error| match error {
            std::sync::TryLockError::WouldBlock => AdaptiveCollectionError::new(
                AdaptiveCollectionStage::Settled,
                "TeacherScopeBusy",
                "teacher worker retirement is temporarily busy",
            ),
            std::sync::TryLockError::Poisoned(_) => AdaptiveCollectionError::new(
                AdaptiveCollectionStage::Settled,
                "TeacherScopePoisoned",
                "teacher worker retirement is poisoned",
            ),
        })?;
        let scope = slot
            .as_ref()
            .ok_or_else(|| reject("teacher scope retirement already issued"))?;
        let Some(outcome) = scope
            .try_worker_retirement()
            .map_err(|error| reject(error.to_string()))?
        else {
            return Ok(None);
        };
        self.worker_issued
            .store(true, std::sync::atomic::Ordering::Release);
        if self
            .physical_issued
            .load(std::sync::atomic::Ordering::Acquire)
        {
            drop(slot.take());
        }
        Ok(Some(outcome))
    }
    fn permit(
        &self,
        manifest: Digest,
        registration: rz_search::external_cpu::collection_teacher::TeacherWorkerRegistration,
    ) -> Result<
        rz_search::external_cpu::process_scope::CollectionProcessPermit,
        rz_search::cpu_checker::CheckerError,
    > {
        let slot = self.scope.try_lock().map_err(|_| {
            rz_search::cpu_checker::CheckerError::Invalid("teacher scope permit unavailable")
        })?;
        let scope = slot
            .as_ref()
            .ok_or(rz_search::cpu_checker::CheckerError::Invalid(
                "teacher scope closed",
            ))?;
        scope.permit(manifest, registration)
    }
    fn begin_spawn<'a>(
        &'a self,
        same_admission: bool,
        permit: &rz_search::external_cpu::process_scope::CollectionProcessPermit,
        try_accepts: impl FnOnce() -> Result<bool, AdaptiveCollectionError>,
    ) -> Result<
        std::sync::MutexGuard<
            'a,
            Option<rz_search::external_cpu::process_scope::CollectionProcessScope>,
        >,
        AdaptiveCollectionError,
    > {
        let slot = self
            .scope
            .try_lock()
            .map_err(|_| reject("teacher scope spawn unavailable"))?;
        let scope = slot
            .as_ref()
            .ok_or_else(|| reject("teacher scope closed"))?;
        if !same_admission || !permit.belongs_to(scope) || !try_accepts()? {
            return Err(reject(
                "teacher scope is closed, stale, or has a different root",
            ));
        }
        Ok(slot)
    }
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub struct RoventTeacherScopeCapabilityV2 {
    admission: RoventCollectionRootAdmissionV2,
    core: TeacherScopeCore,
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl RoventTeacherScopeCapabilityV2 {
    pub fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV2) -> bool {
        self.admission.same_original_admission(admission)
    }
    /// Close only this original scope; Busy/poison never attest completion.
    pub fn close(&self) -> Result<(), AdaptiveCollectionError> {
        self.core.close()
    }
    /// Return an ACK only after the same scope confirms physical retirement.
    pub fn try_physical_retirement(
        &self,
    ) -> Result<Option<RoventTeacherScopePhysicalAckV2>, AdaptiveCollectionError> {
        Ok(self
            .core
            .try_physical_retirement()?
            .map(|outcome| RoventTeacherScopePhysicalAckV2 {
                admission: self.admission.clone(),
                evidence: TeacherScopePhysicalEvidenceV2::Scope(outcome),
            }))
    }
    /// Worker join/no-thread evidence is distinct from physical retirement.
    pub fn try_worker_retirement(
        &self,
    ) -> Result<Option<RoventTeacherScopeWorkerAckV2>, AdaptiveCollectionError> {
        Ok(self
            .core
            .try_worker_retirement()?
            .map(|outcome| RoventTeacherScopeWorkerAckV2 {
                admission: self.admission.clone(),
                outcome,
            }))
    }
    pub fn permit(
        &self,
        manifest: Digest,
        registration: rz_search::external_cpu::collection_teacher::TeacherWorkerRegistration,
    ) -> Result<
        rz_search::external_cpu::process_scope::CollectionProcessPermit,
        rz_search::cpu_checker::CheckerError,
    > {
        self.core.permit(manifest, registration)
    }
    pub(crate) fn begin_spawn<'a>(
        &'a self,
        admission: &RoventCollectionRootAdmissionV2,
        permit: &rz_search::external_cpu::process_scope::CollectionProcessPermit,
    ) -> Result<
        std::sync::MutexGuard<
            'a,
            Option<rz_search::external_cpu::process_scope::CollectionProcessScope>,
        >,
        AdaptiveCollectionError,
    > {
        self.core
            .begin_spawn(self.matches_admission(admission), permit, || {
                self.admission
                    .root_admission
                    .original_context()
                    .try_accepts()
                    .map_err(|e| reject(e.to_string()))
            })
    }
}

/// /3 failures retain either the original unclaimed scope or the original
/// claimed capability. No /2 root or serialized receipt can replace them.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu", target_os = "linux"))]
pub enum RoventTeacherScopeClaimFailureV3 {
    BeforeClaim {
        error: AdaptiveCollectionError,
        scope: rz_search::external_cpu::process_scope::CollectionProcessScope,
    },
    AfterClaim {
        error: AdaptiveCollectionError,
        capability: RoventTeacherScopeCapabilityV3,
    },
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu", target_os = "linux"))]
pub struct RoventTeacherScopePhysicalAckV3 {
    admission: RoventCollectionRootAdmissionV3,
    evidence: TeacherScopePhysicalEvidenceV2,
}
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu", target_os = "linux"))]
impl RoventTeacherScopePhysicalAckV3 {
    pub(crate) fn original_admission(&self) -> &RoventCollectionRootAdmissionV3 {
        &self.admission
    }
    pub fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV3) -> bool {
        self.admission.same_original_admission(admission)
    }
    pub fn scope_outcome(
        &self,
    ) -> Option<&rz_search::external_cpu::process_scope::CollectionScopePhysicalOutcome> {
        match &self.evidence {
            TeacherScopePhysicalEvidenceV2::NoScope => None,
            TeacherScopePhysicalEvidenceV2::Scope(outcome) => Some(outcome),
        }
    }
    pub fn is_sealed_no_scope(&self) -> bool {
        matches!(self.evidence, TeacherScopePhysicalEvidenceV2::NoScope)
    }
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu", target_os = "linux"))]
pub struct RoventTeacherScopeWorkerAckV3 {
    admission: RoventCollectionRootAdmissionV3,
    outcome: rz_search::external_cpu::process_scope::CollectionScopeWorkerOutcome,
}
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu", target_os = "linux"))]
impl RoventTeacherScopeWorkerAckV3 {
    pub(crate) fn original_admission(&self) -> &RoventCollectionRootAdmissionV3 {
        &self.admission
    }
    pub fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV3) -> bool {
        self.admission.same_original_admission(admission)
    }
    pub fn outcome(&self) -> &rz_search::external_cpu::process_scope::CollectionScopeWorkerOutcome {
        &self.outcome
    }
}

/// This retains one actual /3 root and the same shared scope core as /2.
/// Its acknowledgements attest neither teacher semantics nor publication.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu", target_os = "linux"))]
pub struct RoventTeacherScopeCapabilityV3 {
    admission: RoventCollectionRootAdmissionV3,
    core: TeacherScopeCore,
}
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu", target_os = "linux"))]
impl RoventTeacherScopeCapabilityV3 {
    pub fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV3) -> bool {
        self.admission.same_original_admission(admission)
    }
    pub fn close(&self) -> Result<(), AdaptiveCollectionError> {
        self.core.close()
    }
    pub fn try_physical_retirement(
        &self,
    ) -> Result<Option<RoventTeacherScopePhysicalAckV3>, AdaptiveCollectionError> {
        Ok(self
            .core
            .try_physical_retirement()?
            .map(|outcome| RoventTeacherScopePhysicalAckV3 {
                admission: self.admission.clone(),
                evidence: TeacherScopePhysicalEvidenceV2::Scope(outcome),
            }))
    }
    pub fn try_worker_retirement(
        &self,
    ) -> Result<Option<RoventTeacherScopeWorkerAckV3>, AdaptiveCollectionError> {
        Ok(self
            .core
            .try_worker_retirement()?
            .map(|outcome| RoventTeacherScopeWorkerAckV3 {
                admission: self.admission.clone(),
                outcome,
            }))
    }
    pub fn permit(
        &self,
        manifest: Digest,
        registration: rz_search::external_cpu::collection_teacher::TeacherWorkerRegistration,
    ) -> Result<
        rz_search::external_cpu::process_scope::CollectionProcessPermit,
        rz_search::cpu_checker::CheckerError,
    > {
        self.core.permit(manifest, registration)
    }
    pub(crate) fn begin_spawn<'a>(
        &'a self,
        admission: &RoventCollectionRootAdmissionV3,
        permit: &rz_search::external_cpu::process_scope::CollectionProcessPermit,
    ) -> Result<
        std::sync::MutexGuard<
            'a,
            Option<rz_search::external_cpu::process_scope::CollectionProcessScope>,
        >,
        AdaptiveCollectionError,
    > {
        self.core
            .begin_spawn(self.matches_admission(admission), permit, || {
                self.admission
                    .root_admission
                    .original_context()
                    .try_accepts()
                    .map_err(|e| reject(e.to_string()))
            })
    }
}

struct LoadedOrigin {
    // Retains the original pinned source descriptors without copying weights.
    _source_custody: Arc<dyn Send + Sync>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    sources: Arc<crate::rovent_bootstrap::RoventPinnedNativeSources>,
    process_epoch: u64,
    mode: rz_contracts::rovent::ExecutionMode,
    input_lock: Digest,
    namespace: AdaptiveModelNamespace,
    product_lifetime: Option<Arc<RoventNativeProductLifetime>>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    raw: Value,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    raw_sha256: String,
}

/// Current product provenance only. No JSON, public constructor or copied
/// process/source hash can manufacture a new native-service lifetime.
/// The actual PreparedNative issuer is a separate registration boundary.
pub struct RoventNativeProductLifetime {
    _private: (),
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    first_root: OnceLock<Arc<crate::rovent_root_admission::RoventRootAdmission>>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    sources: std::sync::Weak<crate::rovent_bootstrap::RoventPinnedNativeSources>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    clock: rz_runtime::contracts::ContractSystemClock,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    namespace: AdaptiveModelNamespace,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    _original_verifier: rz_search::pals::coordinator::NativeAcceptanceVerifier,
}
impl RoventNativeProductLifetime {
    /// Only the successful bootstrap calls this after consuming the actual
    /// Ready event. The worker-issued verifier and ready clock are required;
    /// snapshot metadata alone cannot create this lifetime.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(crate) fn from_ready<Q: crate::pals_adaptive::NativeQuestionProjection>(
        snapshot: &crate::rovent_bootstrap::RoventNativeStartupSnapshot,
        sources: &Arc<crate::rovent_bootstrap::RoventPinnedNativeSources>,
        actual_service: &crate::rovent_bootstrap::RoventNativeService<Q>,
    ) -> Result<Arc<Self>, AdaptiveCollectionError> {
        use rz_runtime::contracts::ContractClock;
        let clock = actual_service
            .ready_clock()
            .ok_or_else(|| reject("original native service has not reached Ready"))?;
        let verifier = actual_service
            .native_acceptance_verifier()
            .ok_or_else(|| reject("original native service has no worker-issued verifier"))?;
        if clock.domain().0.0 != snapshot.process_epoch
            || snapshot.input_lock != sources.lock_digest()
            || snapshot.physical_workers_created != 1
            || snapshot.neural_service_executed
        {
            return Err(reject("original Ready service/snapshot/source differs"));
        }
        let namespace = snapshot
            .model_namespace()
            .map_err(|error| reject(error.to_string()))?;
        Ok(Arc::new(Self {
            _private: (),
            first_root: OnceLock::new(),
            sources: Arc::downgrade(sources),
            clock,
            namespace,
            _original_verifier: verifier,
        }))
    }
    pub fn same_product(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

/// 발급은 PreparedRoventNative의 원 성공 owner에서만 가능하다.
/// public constructor·Deserialize·wire에서의 복원은 제공하지 않는다.
#[derive(Clone)]
pub struct RoventNativeCollectionOrigin {
    original: Arc<LoadedOrigin>,
}
impl RoventNativeCollectionOrigin {
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(crate) fn from_loaded_with_lifetime(
        snapshot: &crate::rovent_bootstrap::RoventNativeStartupSnapshot,
        sources: Arc<crate::rovent_bootstrap::RoventPinnedNativeSources>,
        lifetime: Arc<RoventNativeProductLifetime>,
    ) -> Result<Self, AdaptiveCollectionError> {
        use rz_runtime::contracts::ContractClock;
        if !lifetime.sources.ptr_eq(&Arc::downgrade(&sources))
            || lifetime.clock.domain().0.0 != snapshot.process_epoch
            || lifetime.namespace
                != snapshot
                    .model_namespace()
                    .map_err(|error| reject(error.to_string()))?
        {
            return Err(reject(
                "original native lifetime differs from loaded source",
            ));
        }
        let mut origin = Self::from_loaded(snapshot, sources)?;
        Arc::get_mut(&mut origin.original)
            .expect("new original loaded owner")
            .product_lifetime = Some(lifetime);
        Ok(origin)
    }
    pub fn product_lifetime(&self) -> Option<&Arc<RoventNativeProductLifetime>> {
        self.original.product_lifetime.as_ref()
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(crate) fn from_loaded(
        snapshot: &crate::rovent_bootstrap::RoventNativeStartupSnapshot,
        sources: Arc<crate::rovent_bootstrap::RoventPinnedNativeSources>,
    ) -> Result<Self, AdaptiveCollectionError> {
        let lock = sources.lock();
        let manifest = &lock.manifest;
        let pc = &manifest.engine.pc;
        if manifest.purpose != rz_experiments::RoventPurpose::Collection
            || pc.proposer.diagnostic
            || pc.critic.diagnostic
            || pc.validator.is_some()
            || snapshot.validator_loaded
            || snapshot.process_epoch == 0
            || snapshot.input_lock != sources.lock_digest()
            || snapshot.mode != manifest.engine.mode
            || snapshot.weights.0 != sources.weights_digest()
            || text_sha(snapshot.export_manifest) != sources.export_digest_text()
            || snapshot.neural_service_executed
        {
            return Err(reject(
                "loaded source is not the original nondiagnostic Rovent collection startup",
            ));
        }
        let namespace = snapshot
            .model_namespace()
            .map_err(|error| reject(error.to_string()))?;
        let source_sha256 = rz_experiments::rovent_source_aggregate_sha256(
            &manifest.engine.source_commit,
            &manifest.engine.source_components,
        )
        .map_err(|error| reject(error.to_string()))?;
        let raw = json!({
            "schema":ROVENT_COLLECTION_NATIVE_ORIGIN,
            "purpose":"collection",
            "process_epoch":snapshot.process_epoch,
            "mode":format!("{:?}",snapshot.mode),
            "input_lock_sha256":text_sha(snapshot.input_lock),
            "source_commit":manifest.engine.source_commit,
            "source_components":manifest.engine.source_components,
            "source_sha256":source_sha256,
            "weights_sha256":text_sha(snapshot.weights),
            "export_manifest_sha256":text_sha(snapshot.export_manifest),
            "backend_sha256":text_sha(snapshot.backend),
            "runtime_library_sha256":text_sha(snapshot.runtime_library),
            "runtime_bundle_sha256":snapshot.runtime_bundle.map(text_sha),
            "native_configuration_sha256":text_sha(snapshot.native_configuration),
            "model_semantics":pc.proposer.semantics,
            "model_semantics_sha256":text_sha(Digest(rz_eval::pals_model::PalsModelProfile::VirtualReviewV1.registered_semantics_sha256())),
            "encoding":pc.proposer.encoding,
            "encoding_sha256":text_sha(Digest({use sha2::Digest as _; sha2::Sha256::digest(rz_eval::pals_model::virtual_review::ENCODING.as_bytes()).into()})),
            "encoding_registry_sha256":text_sha(Digest(rz_eval::pals_model::virtual_review::registered_encoding_sha256())),
            "model_implementation_sha256":pc.proposer.implementation_sha256,
            "representation_model_domain":"loaded_checkpoint_sha256",
            "teacher_profiles":manifest.engine.teachers,
            "frozen_epoch":1,
            "runtime_sha256_domain":"loaded_runtime_library_file_sha256",
            "runtime_sha256":text_sha(snapshot.runtime_library),
        });
        let raw_sha256 =
            rz_experiments::pals_data_v3::generation_digest(ROVENT_COLLECTION_NATIVE_ORIGIN, &raw)
                .map_err(|error| reject(error.to_string()))?;
        let process_epoch = snapshot.process_epoch;
        let mode = snapshot.mode;
        let input_lock = snapshot.input_lock;
        Ok(Self {
            original: Arc::new(LoadedOrigin {
                _source_custody: Arc::clone(&sources) as Arc<dyn Send + Sync>,
                sources,
                process_epoch,
                mode,
                input_lock,
                namespace,
                product_lifetime: None,
                raw,
                raw_sha256,
            }),
        })
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn raw_receipt(&self) -> &Value {
        &self.original.raw
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn raw_sha256(&self) -> &str {
        &self.original.raw_sha256
    }
    /// Original immutable parsed lock, retained by the actual loaded source.
    /// Serializing this snapshot preserves meaning, not original whitespace.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn original_input_lock(&self) -> &rz_experiments::RoventInputLock {
        self.original.sources.lock()
    }
    /// Same retained source allocation, not matching declarations or hashes.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn matches_loaded_nnuec(
        &self,
        source: &crate::rovent_bootstrap::nnuec_source::LoadedRoventNnuecSource,
    ) -> bool {
        source.matches_sources(&self.original.sources)
            && source.input_lock_digest() == self.original.input_lock
    }
    /// Original metadata bytes only; neither this read nor historical replay
    /// grants runtime, native, physical or training execution authority.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn runtime_bundle_publication_bytes(
        &self,
    ) -> Result<Option<(rz_experiments::ArtifactRef, Vec<u8>)>, AdaptiveCollectionError> {
        let original = self
            .original
            .sources
            .runtime_bundle_publication_bytes()
            .map_err(|error| reject(error.to_string()))?;
        match &original {
            Some((_, bytes)) => {
                let text = std::str::from_utf8(bytes)
                    .map_err(|_| reject("original bundle metadata is not UTF-8"))?;
                let spec = rz_eval::runtime_pin::CudaRuntimeBundleSpec::from_json(text)
                    .map_err(|error| reject(error.to_string()))?;
                let digest = text_sha(Digest(
                    spec.digest().map_err(|error| reject(error.to_string()))?,
                ));
                if self.original.raw["runtime_bundle_sha256"] != digest {
                    return Err(reject(
                        "original bundle file bytes do not join loaded binary-codec digest",
                    ));
                }
            }
            None if !self.original.raw["runtime_bundle_sha256"].is_null() => {
                return Err(reject(
                    "loaded CUDA bundle lacks original retained metadata bytes",
                ));
            }
            None => {}
        }
        Ok(original)
    }
    pub(crate) fn verify_profile(
        &self,
        mode: rz_contracts::rovent::ExecutionMode,
        input_lock: Digest,
        namespace: &AdaptiveModelNamespace,
    ) -> Result<(), AdaptiveCollectionError> {
        if self.original.mode != mode
            || self.original.input_lock != input_lock
            || &self.original.namespace != namespace
        {
            return Err(reject(
                "collection profile differs from the original loaded source identity",
            ));
        }
        Ok(())
    }
}

/// 원 attach 성공 제어흐름에서 발급하는 root-scoped 권한이다. 원 Rules와
/// 제한된 root move domain을 Arc로 보존하므로 복사한 digest는 이를 대신하지 못한다.
#[derive(Clone)]
pub struct RoventCollectionRootAdmission {
    original: RoventNativeCollectionOrigin,
    authority: SearchAuthority,
    deadline: Deadline,
    rules: Arc<ContractState>,
    root_moves: Arc<RootMoveDomain>,
    root_state: rz_contracts::StateIdentity,
    root_budget: Digest,
    selective_policy: Digest,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    root_admission: Arc<crate::rovent_root_admission::RoventRootAdmission>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    external_source: Option<crate::rovent_external_launch::RoventAdmittedExternalRootSource>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    raw: Arc<Value>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    raw_sha256: Arc<str>,
}
impl RoventCollectionRootAdmission {
    /// Absent current lifetime evidence stays unknown. This does not infer
    /// fresh cache state from a first-looking authority or an epoch number.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn is_first_product_root(&self) -> bool {
        self.original
            .original
            .product_lifetime
            .as_ref()
            .is_some_and(|lifetime| {
                lifetime
                    .first_root
                    .get()
                    .is_some_and(|root| Arc::ptr_eq(root, &self.root_admission))
            })
    }
    /// Call only immediately after the actual port's attach_rovent_frontier
    /// has checked independent Rules/Store/namespace/root-domain/ledger owners.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(crate) fn from_attached_frontier(
        original: &RoventNativeCollectionOrigin,
        frontier: &RoventFrontier,
        input_lock: Digest,
        root_budget: Digest,
        selective_policy: Digest,
        root_admission: Arc<crate::rovent_root_admission::RoventRootAdmission>,
    ) -> Result<Self, AdaptiveCollectionError> {
        let registration = frontier.registration();
        if registration.mode.has_native_r() != registration.native_ledger.is_some()
            || frontier.external_source().is_some()
        {
            return Err(reject(
                "own collection frontier requires the original native ledger assembly",
            ));
        }
        Self::from_attached_scope(
            original,
            frontier,
            input_lock,
            root_budget,
            selective_policy,
            root_admission,
            None,
        )
    }
    /// 실제 owner가 같은 lock/live RootAdmission과 결합한 외부 source만 받는다.
    /// 외부 reported work는 native/common node credit으로 바꾸지 않는다.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_attached_external_frontier(
        original: &RoventNativeCollectionOrigin,
        frontier: &RoventFrontier,
        input_lock: Digest,
        root_budget: Digest,
        selective_policy: Digest,
        root_admission: Arc<crate::rovent_root_admission::RoventRootAdmission>,
        source: &crate::rovent_external_launch::RoventAdmittedExternalRootSource,
    ) -> Result<Self, AdaptiveCollectionError> {
        let registration = frontier.registration();
        source
            .validate_current()
            .map_err(|error| reject(error.to_string()))?;
        let actual = source.worker_source();
        let raw = source.collection_evidence();
        let Some(rz_experiments::RoventResearcher::External { profile }) =
            &original.original_input_lock().manifest.engine.researcher
        else {
            return Err(reject(
                "external collection root has no original external researcher pin",
            ));
        };
        if !registration.mode.has_native_r()
            || registration.native_ledger.is_some()
            || !Arc::ptr_eq(source.root_admission(), &root_admission)
            || !frontier
                .external_source()
                .is_some_and(|registered| registered.same_source(actual))
            || actual.authority() != registration.authority
            || actual.clock().domain != root_admission.runtime_clock().domain()
            || actual.clock().origin != root_admission.runtime_clock().origin()
            || raw["schema"] != ROVENT_COLLECTION_PROFILED_EXTERNAL_SOURCE
            || raw["startup_input_lock_sha256"] != text_sha(input_lock)
            || raw["profile_file_sha256"] != profile.profile.sha256
            || raw["profile_file_bytes"].as_u64() != Some(profile.profile.bytes)
            || raw["profile_canonical_sha256"] != profile.canonical_sha256
            || raw["profile_capability_sha256"] != profile.capability_sha256
            || raw["binary"]
                != serde_json::to_value(&profile.binary)
                    .map_err(|error| reject(error.to_string()))?
            || raw["purpose"] != "research_checker"
        {
            return Err(reject(
                "external collection original worker/source/root/lock differs",
            ));
        }
        Self::from_attached_scope(
            original,
            frontier,
            input_lock,
            root_budget,
            selective_policy,
            root_admission,
            Some(source),
        )
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    #[allow(clippy::too_many_arguments)]
    fn from_attached_scope(
        original: &RoventNativeCollectionOrigin,
        frontier: &RoventFrontier,
        input_lock: Digest,
        root_budget: Digest,
        selective_policy: Digest,
        root_admission: Arc<crate::rovent_root_admission::RoventRootAdmission>,
        external: Option<&crate::rovent_external_launch::RoventAdmittedExternalRootSource>,
    ) -> Result<Self, AdaptiveCollectionError> {
        let registration = frontier.registration();
        let root_state = registration.root_rules.snapshot().identity();
        let loaded = &original.original;
        if input_lock != loaded.input_lock
            || registration.mode != loaded.mode
            || registration.authority.epoch.0 != loaded.process_epoch
            || registration.root_budget != root_budget
            || registration.selective_policy != selective_policy
            || registration.model_namespace.as_ref() != &loaded.namespace
            || registration.root_rules.snapshot().identity()
                != registration.root_moves.root_identity()
            || !root_admission
                .original_context()
                .accepts()
                .map_err(|error| reject(error.to_string()))?
            || root_admission.startup_lock_digest() != input_lock
            || root_admission.authority() != registration.authority
            || root_admission.root_budget_digest() != root_budget
            || root_admission.hard_deadline() != registration.original_deadline
            || root_admission.goal() != registration.goal
            || root_admission.requested_nodes() != registration.nodes
            || root_admission.state_and_history_digest() != root_state.semantic
            || root_admission.root_snapshot().revision() != root_state.revision.0
            || !root_admission
                .root_snapshot()
                .same_state(registration.root_rules.rules().snapshot())
            || root_admission.allowed_root_moves().as_ref()
                != registration.root_moves.allowed().as_ref()
        {
            return Err(reject(
                "actual attached frontier differs from original loaded collection scope",
            ));
        }
        root_admission
            .descriptor()
            .verify_against(original.original_input_lock())
            .map_err(|error| reject(error.to_string()))?;
        let domain = if external.is_some() {
            ROVENT_COLLECTION_EXTERNAL_ROOT_ADMISSION
        } else {
            ROVENT_COLLECTION_ROOT_ADMISSION
        };
        let mut raw = json!({
            "schema":domain,
            "native_origin":original.raw_receipt(),
            "native_origin_sha256":original.raw_sha256(),
            "process_epoch":registration.authority.epoch.0,
            "game_generation":registration.authority.game.0,
            "root_generation":registration.authority.root.0,
            "implementation_sha256":text_sha(registration.authority.implementation),
            "deadline_clock":registration.original_deadline.clock.0.0,
            "deadline_tick":registration.original_deadline.at.0,
            "root_budget_sha256":text_sha(root_budget),
            "selective_policy_sha256":text_sha(selective_policy),
            "root_state":{"owner":root_state.owner.0,
                "revision":root_state.revision.0,"semantic":text_sha(root_state.semantic)},
            "store_root_state":registration.root_state.0,
            "root_moves":registration.root_moves.allowed().iter().map(|movement|movement.to_string()).collect::<Vec<_>>(),
            "depth_plies":registration.goal.depth_plies.map(|value|value.get()),
            "mate_moves":registration.goal.mate_moves.map(|value|value.get()),
            "requested_nodes":registration.nodes,
            "root_descriptor":root_admission.descriptor(),
            "root_descriptor_sha256":text_sha(root_admission.digest()),
        });
        if let Some(source) = external {
            raw["external_source"] = source.collection_evidence().clone();
            raw["external_source_sha256"] = json!(
                rz_experiments::pals_data_v3::generation_digest(
                    ROVENT_COLLECTION_PROFILED_EXTERNAL_SOURCE,
                    source.collection_evidence()
                )
                .map_err(|error| reject(error.to_string()))?
            );
        }
        let raw_sha256 = rz_experiments::pals_data_v3::generation_digest(domain, &raw)
            .map_err(|error| reject(error.to_string()))?;
        if let Some(lifetime) = &loaded.product_lifetime {
            // Only actual successful attach can observe this root. A second
            // root leaves the original first Arc unchanged and is not fresh.
            let _ = lifetime.first_root.set(Arc::clone(&root_admission));
        }
        Ok(Self {
            original: original.clone(),
            authority: registration.authority,
            deadline: registration.original_deadline,
            rules: Arc::clone(&registration.root_rules),
            root_moves: Arc::clone(&registration.root_moves),
            root_state,
            root_budget,
            selective_policy,
            root_admission,
            external_source: external.cloned(),
            raw: Arc::new(raw),
            raw_sha256: raw_sha256.into(),
        })
    }
    #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
    pub(crate) fn from_attached_frontier(
        _: &RoventNativeCollectionOrigin,
        _: &RoventFrontier,
        _: Digest,
        _: Digest,
        _: Digest,
    ) -> Result<Self, AdaptiveCollectionError> {
        Err(AdaptiveCollectionError::new(
            AdaptiveCollectionStage::Accepted,
            "RoventCollectionUnsupported",
            "actual native collection requires pals-adaptive and onnx-cpu",
        ))
    }
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn validates_registration(
        &self,
        authority: SearchAuthority,
        mode: rz_contracts::rovent::ExecutionMode,
        root_budget: Digest,
        deadline: Deadline,
    ) -> bool {
        self.authority == authority
            && self.original.original.mode == mode
            && self.root_budget == root_budget
            && self.deadline == deadline
    }
    pub fn selective_policy(&self) -> Digest {
        self.selective_policy
    }
    pub fn rules(&self) -> &Arc<ContractState> {
        &self.rules
    }
    pub fn root_moves(&self) -> &Arc<RootMoveDomain> {
        &self.root_moves
    }
    pub fn native_origin(&self) -> &RoventNativeCollectionOrigin {
        &self.original
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn root_admission(&self) -> &Arc<crate::rovent_root_admission::RoventRootAdmission> {
        &self.root_admission
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn external_source(
        &self,
    ) -> Option<&crate::rovent_external_launch::RoventAdmittedExternalRootSource> {
        self.external_source.as_ref()
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn raw_receipt(&self) -> &Value {
        &self.raw
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn raw_sha256(&self) -> &str {
        &self.raw_sha256
    }
    pub fn validate_ticket(
        &self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(), AdaptiveCollectionError> {
        if ticket.authority != self.authority
            || ticket.original_deadline != self.deadline
            || ticket.view.root_state != self.root_state
            || ticket.role == Role::Validator
        {
            return Err(reject(
                "native ticket is outside the original admitted Rovent P/C root",
            ));
        }
        Ok(())
    }
    pub fn validate_output(
        &self,
        ticket: &RoleTicketEnvelope,
        output: &rz_contracts::pals::RoleOutput,
    ) -> Result<(), AdaptiveCollectionError> {
        self.validate_ticket(ticket)?;
        let namespace = &self.original.original.namespace;
        namespace
            .verify(ticket, output, Some(namespace.value_identity()))
            .map_err(|error| reject(error.to_string()))
    }
    pub(super) fn validate_ranking(
        &self,
        ranking: &ProposerRankingPublished,
    ) -> Result<(), AdaptiveCollectionError> {
        self.validate_ticket(&ranking.source().ticket)?;
        if ranking.source().ticket.role != Role::Proposer
            || !std::ptr::eq(ranking.rules(), self.rules.as_ref())
        {
            return Err(reject(
                "initial P publication is not from the original admitted Rules owner",
            ));
        }
        Ok(())
    }
}
