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
struct LoadedOriginV2 {
    sources:
        Arc<crate::rovent_bootstrap::RoventPinnedNativeSources<rz_experiments::RoventInputLockV2>>,
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
    first_root: OnceLock<Arc<crate::rovent_root_admission::source_seal::RoventRootAdmissionV2>>,
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl RoventNativeCollectionOriginV2 {
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
        use rz_runtime::contracts::ContractClock;
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
        let clock = service
            .ready_clock()
            .ok_or_else(|| reject("original /2 native service has not reached Ready"))?;
        let verifier = service
            .native_acceptance_verifier()
            .ok_or_else(|| reject("original /2 native service has no worker verifier"))?;
        if service.is_quarantined()
            || lock.manifest.purpose != rz_experiments::RoventPurpose::Collection
            || pc.proposer.diagnostic
            || pc.critic.diagnostic
            || pc.validator.is_some()
            || snapshot.validator_loaded
            || snapshot.process_epoch == 0
            || clock.domain().0.0 != snapshot.process_epoch
            || snapshot.input_lock != sources.lock_digest()
            || snapshot.mode != lock.manifest.engine.mode
            || snapshot.weights.0 != sources.weights_digest()
            || text_sha(snapshot.export_manifest) != sources.export_digest_text()
            || snapshot.physical_workers_created != 1
            || snapshot.neural_service_executed
        {
            return Err(reject(
                "original /2 Ready, Collection source or snapshot differs",
            ));
        }
        let namespace = snapshot
            .model_namespace()
            .map_err(|error| reject(error.to_string()))?;
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

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl RoventCollectionRootAdmissionV2 {
    /// Claim this exact live root's sole teacher process scope. A claim is
    /// never reissued, including after failed or physically unknown work.
    /// Do all reversible roster/source preflight before calling this method.
    pub fn claim_teacher_scope(&self, scope: rz_search::external_cpu::process_scope::CollectionProcessScope)
        -> Result<RoventTeacherScopeCapabilityV2, (AdaptiveCollectionError, rz_search::external_cpu::process_scope::CollectionProcessScope)> {
        match self.root_admission.original_context().accepts() {
            Ok(true) => {}
            Ok(false) => return Err((reject("teacher scope root is revoked"), scope)),
            Err(error) => return Err((reject(error.to_string()), scope)),
        }
        let root_digest = match self.root_admission.descriptor().digest() {
            Ok(digest) => digest,
            Err(error) => return Err((reject(error.to_string()), scope)),
        };
        if !scope.is_unissued_for_root(
            self.root_admission.authority(), self.root_admission.hard_deadline(),
            self.root_admission.startup_lock_digest(), root_digest, self.root_budget,
        ) {
            return Err((reject("teacher scope is shared, used, closed, or belongs to another root"), scope));
        }
        if !self.root_admission.claim_teacher_scope_once() {
            return Err((reject("teacher scope already claimed for this original root"), scope));
        }
        match self.root_admission.original_context().accepts() {
            Ok(true) => {}
            Ok(false) => return Err((reject("teacher scope root revoked during claim"), scope)),
            Err(error) => return Err((reject(error.to_string()), scope)),
        }
        Ok(RoventTeacherScopeCapabilityV2 {
            admission: self.clone(),
            scope: std::sync::Mutex::new(Some(scope)),
        })
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
        use rz_runtime::contracts::ContractClock;
        let loaded = &original.original;
        let preload = loaded
            .sources
            .verify_collection_v2_origin()
            .map_err(|error| reject(error.to_string()))?;
        let lock = root_admission
            .owner()
            .v2()
            .ok_or_else(|| reject("Collection /2 root has no original /2 lock"))?;
        let registration = frontier.registration();
        let root_state = registration.root_rules.snapshot().identity();
        if !Arc::ptr_eq(preload, &loaded.preload)
            || !Arc::ptr_eq(lock, loaded.sources.lock_owner())
            || registration.mode.has_native_r() != registration.native_ledger.is_some()
            || frontier.external_source().is_some()
            || registration.mode != loaded.mode
            || registration.authority.epoch.0 != loaded.process_epoch
            || registration.root_budget != root_budget
            || registration.selective_policy != selective_policy
            || registration.model_namespace.as_ref() != &loaded.namespace
            || root_state != registration.root_moves.root_identity()
            || !root_admission
                .original_context()
                .accepts()
                .map_err(|error| reject(error.to_string()))?
            || root_admission.startup_lock_digest() != loaded.input_lock
            || root_admission.authority() != registration.authority
            || root_admission
                .descriptor()
                .root_budget_digest()
                .map_err(|error| reject(error.to_string()))?
                != root_budget
            || root_admission.hard_deadline() != registration.original_deadline
            || root_admission.goal() != registration.goal
            || root_admission.requested_nodes() != registration.nodes
            || root_admission.runtime_clock().domain() != loaded.clock.domain()
            || root_admission.runtime_clock().origin() != loaded.clock.origin()
            || root_admission.state_and_history_digest() != root_state.semantic
            || root_admission.root_snapshot().revision() != root_state.revision.0
            || !root_admission
                .root_snapshot()
                .same_state(registration.root_rules.rules().snapshot())
            || root_admission.allowed_root_moves().as_ref()
                != registration.root_moves.allowed().as_ref()
        {
            return Err(reject(
                "actual attached Frontier differs from original /2 Collection scope",
            ));
        }
        root_admission
            .descriptor()
            .verify_against(lock)
            .map_err(|error| reject(error.to_string()))?;
        root_admission
            .allocation_clock()
            .verify_against(root_admission.descriptor(), lock)
            .map_err(|error| reject(error.to_string()))?;
        let descriptor_digest = root_admission
            .descriptor()
            .digest()
            .map_err(|error| reject(error.to_string()))?;
        let allocation_digest = root_admission
            .allocation_clock()
            .digest()
            .map_err(|error| reject(error.to_string()))?;
        let raw = json!({
            "schema": ROVENT_COLLECTION_ROOT_ADMISSION_LOCK_V2,
            "native_origin": original.raw_receipt(),
            "native_origin_sha256": original.raw_sha256(),
            "process_epoch": registration.authority.epoch.0,
            "game_generation": registration.authority.game.0,
            "root_generation": registration.authority.root.0,
            "implementation_sha256": text_sha(registration.authority.implementation),
            "deadline_clock": registration.original_deadline.clock.0.0,
            "deadline_tick": registration.original_deadline.at.0,
            "root_budget_sha256": text_sha(root_budget),
            "selective_policy_sha256": text_sha(selective_policy),
            "root_state": {"owner": root_state.owner.0,
                "revision": root_state.revision.0, "semantic": text_sha(root_state.semantic)},
            "store_root_state": registration.root_state.0,
            "root_moves": registration.root_moves.allowed().iter()
                .map(|movement| movement.to_string()).collect::<Vec<_>>(),
            "depth_plies": registration.goal.depth_plies.map(|value| value.get()),
            "mate_moves": registration.goal.mate_moves.map(|value| value.get()),
            "requested_nodes": registration.nodes,
            "root_descriptor": root_admission.descriptor(),
            "root_descriptor_sha256": text_sha(descriptor_digest),
            "allocation_clock": root_admission.allocation_clock(),
            "allocation_clock_sha256": text_sha(allocation_digest),
        });
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
        if !self.original.original._original_verifier.same_owner(verifier) {
            return Err(reject("native verifier is not the original /2 worker verifier"));
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
        self.validate_ticket(&ranking.source().ticket)?;
        if ranking.source().ticket.role != Role::Proposer
            || !std::ptr::eq(ranking.rules(), self.rules.as_ref())
        {
            return Err(reject(
                "initial P publication is not from the original admitted /2 Rules owner",
            ));
        }
        Ok(())
    }
}

/// Non-cloneable authority for one original root's teacher scope. Closing an
/// old capability only touches its own root even when a later generation has
/// already started. It is not a physical-retirement or publication proof.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub struct RoventTeacherScopeCapabilityV2 {
    admission: RoventCollectionRootAdmissionV2,
    scope: std::sync::Mutex<Option<rz_search::external_cpu::process_scope::CollectionProcessScope>>,
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl RoventTeacherScopeCapabilityV2 {
    pub fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV2) -> bool {
        self.admission.same_original_admission(admission)
    }

    /// Call outside manager/root locks. Contention or poisoning is not a
    /// closure acknowledgement and leaves the capability with its owner.
    pub fn close(&self) -> Result<(), AdaptiveCollectionError> {
        let mut slot = self.scope.try_lock().map_err(|_| reject("teacher scope close unavailable"))?;
        if let Some(scope) = slot.take() {
            scope.close_admission();
            // Outstanding physical permits retain the original issuer Arc.
            // Releasing the issuer here does not attest their completion.
            drop(scope);
        }
        Ok(())
    }

    pub fn permit(
        &self,
        manifest: Digest,
        registration: rz_search::external_cpu::collection_teacher::TeacherWorkerRegistration,
    ) -> Result<rz_search::external_cpu::process_scope::CollectionProcessPermit, rz_search::cpu_checker::CheckerError> {
        let slot = self.scope.try_lock().map_err(|_| rz_search::cpu_checker::CheckerError::Invalid("teacher scope permit unavailable"))?;
        let scope = slot.as_ref().ok_or(rz_search::cpu_checker::CheckerError::Invalid("teacher scope closed"))?;
        scope.permit(manifest, registration)
    }

    pub(crate) fn begin_spawn<'a>(
        &'a self,
        admission: &RoventCollectionRootAdmissionV2,
        permit: &rz_search::external_cpu::process_scope::CollectionProcessPermit,
    ) -> Result<std::sync::MutexGuard<'a, Option<rz_search::external_cpu::process_scope::CollectionProcessScope>>, AdaptiveCollectionError> {
        let slot = self.scope.try_lock().map_err(|_| reject("teacher scope spawn unavailable"))?;
        let scope = slot.as_ref().ok_or_else(|| reject("teacher scope closed"))?;
        if !self.matches_admission(admission) || !permit.belongs_to(scope)
            || !self.admission.root_admission.original_context().try_accepts().map_err(|e| reject(e.to_string()))?
        {
            return Err(reject("teacher scope is closed, stale, or has a different root"));
        }
        Ok(slot)
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
