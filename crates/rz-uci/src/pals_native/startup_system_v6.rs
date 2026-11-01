//! Source-owned PALS system-CUDA model input boundary.
//!
//! Admission checks the actual retained descriptors and model declarations;
//! it does not load ORT, prove numerical equivalence or authorize a game.
use crate::pals_adaptive_startup::{
    PalsSystemSourceObservationV6, PalsSystemSourceReadFailureV6, PinnedPalsSystemStartupV6,
};
use crate::pals_native::startup_virtual::graph_purpose;
use crate::pals_native::virtual_encoding::{
    VirtualRulesDescriptorSummary, observe_pals_virtual_rules_descriptor_bytes,
    validate_pals_virtual_rules_descriptor_bytes,
};
use rz_eval::asset::parse_sha256;
use rz_eval::error::{BackendError, CauseCode, FailureKind, FailureStage};
use rz_eval::pals_model::{PalsModelProfile, virtual_review};
use rz_eval::pals_onnx::virtual_review::{
    VirtualReviewArtifactKind, VirtualReviewExecutionMode, VirtualReviewOwnedSources,
    validate_virtual_review_export_manifest_kind,
};
use rz_experiments::{
    PalsArtifactDomainV5, PalsEngineModeV5, PalsStartupAssetPurposeV5 as Purpose,
    PalsSystemSourcePolicyV6,
};
use std::{
    path::{Component, Path},
    sync::Arc,
};

fn error(detail: &'static str) -> BackendError {
    BackendError::new(
        FailureKind::IdentityMismatch,
        FailureStage::Admission,
        detail,
    )
}
fn transport(_: rz_experiments::ManifestError) -> BackendError {
    error("retained PALS system model asset failed verification")
}
fn transport_source(source: PalsSystemSourceReadFailureV6) -> BackendError {
    let (kind, code) = match &source {
        PalsSystemSourceReadFailureV6::Cancelled => {
            (FailureKind::BackendUnavailable, "SourceCancelled")
        }
        PalsSystemSourceReadFailureV6::Expired => {
            (FailureKind::BackendUnavailable, "SourceDeadlineExpired")
        }
        PalsSystemSourceReadFailureV6::Io(_) => (FailureKind::Io, "SourceIo"),
        PalsSystemSourceReadFailureV6::Resource => {
            (FailureKind::ResourceExhausted, "SourceResource")
        }
        PalsSystemSourceReadFailureV6::Identity | PalsSystemSourceReadFailureV6::Owner(_) => {
            (FailureKind::IdentityMismatch, "SourceOwner")
        }
    };
    BackendError::new(
        kind,
        FailureStage::Admission,
        "PALS system source admission failed",
    )
    .with_external_cause(CauseCode::ModelLoad, &source)
    .with_diagnostic(code, &source.to_string())
}
fn hex(digest: [u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}
fn validate_export_graph_name(file: &str) -> Result<(), BackendError> {
    if !matches!(
        Path::new(file).components().collect::<Vec<_>>().as_slice(),
        [Component::Normal(_)]
    ) || Path::new(file).file_name().and_then(|name| name.to_str()) != Some(file)
    {
        return Err(error("PALS system export graph name is not a basename"));
    }
    Ok(())
}

/// Source and numeric observations are separate from functional admission.
/// The consumer must retain these findings rather than silently treating
/// observational source drift as verified model compatibility.
#[must_use]
pub(crate) struct PalsSystemVirtualObservationV6 {
    pub rules: VirtualRulesDescriptorSummary,
    pub declared_model_source: String,
    pub declared_component_source: Option<String>,
    pub compiled_model_source: Option<String>,
    pub model_source_evidence: PalsSystemSourceObservationV6,
    pub numerical_acceptance: Option<[u8; 32]>,
}

/// Retains the real startup owner, same startup deadline and cancellation for
/// every source read. The backend must retain this Arc for its full lifetime.
pub(crate) struct PalsSystemVirtualSourcesV6 {
    owner: Arc<PinnedPalsSystemStartupV6>,
}
impl PalsSystemVirtualSourcesV6 {
    pub(crate) fn new(owner: Arc<PinnedPalsSystemStartupV6>) -> Self {
        Self { owner }
    }

    /// Inspect the retained model inputs first, then capture the system
    /// runtime with the same original source owner and its deadline. This
    /// does not create a PALS worker or authorize native game execution.
    ///
    /// The caller-supplied boundary must retain the original native owner on
    /// ORT entry, errors, cancellation and unwind; the returned observations
    /// do not prove a provider session or physical completion.
    #[allow(dead_code)]
    pub(crate) fn inspect_and_load_system_cuda_with(
        &self,
        load: impl FnOnce(
            &rz_eval::onnx::SystemRuntime,
        ) -> Result<rz_eval::onnx::OrtRuntime, BackendError>,
    ) -> Result<
        (
            crate::pals_adaptive_bootstrap::runtime_assets::system_cuda::SystemCudaStartup,
            PalsSystemVirtualObservationV6,
        ),
        BackendError,
    > {
        let observation = self.inspect()?;
        let startup = self.owner.load_system_cuda_with(load)?;
        Ok((startup, observation))
    }

    fn read(&self, purpose: Purpose, maximum: usize) -> Result<Vec<u8>, BackendError> {
        self.owner
            .read_asset_exact(purpose, maximum)
            .map_err(transport_source)
    }
    /// Validate model/export/Rules declarations against the original held
    /// bytes. Provenance, learned compatibility and independent numerical
    /// acceptance remain separate evidence and are not granted here.
    pub(crate) fn inspect(&self) -> Result<PalsSystemVirtualObservationV6, BackendError> {
        self.owner
            .verify_source_unchanged()
            .map_err(transport_source)?;
        let config = &self.owner.lock().config;
        let model = &config.endpoint.model;
        if model.profile != virtual_review::PROFILE
            || model.semantic_sha256
                != hex(PalsModelProfile::VirtualReviewV1.registered_semantics_sha256())
            || model.encoding_sha256 != hex(virtual_review::registered_encoding_sha256())
        {
            return Err(error("PALS system model or encoding declaration differs"));
        }
        let compiled_model_source = self
            .owner
            .compiled_component("model_native")
            .map(str::to_owned);
        let declared_component_source = config.source_components.get("model_native").cloned();
        let model_source_evidence = match (&compiled_model_source, &declared_component_source) {
            (Some(actual), Some(declared))
                if actual == declared && actual == &model.implementation_sha256 =>
            {
                PalsSystemSourceObservationV6::Verified
            }
            (None, _) | (_, None) => PalsSystemSourceObservationV6::Missing,
            _ => PalsSystemSourceObservationV6::Mismatch,
        };
        if config.source_policy == PalsSystemSourcePolicyV6::Strict
            && model_source_evidence != PalsSystemSourceObservationV6::Verified
        {
            return Err(error("strict PALS system model source is not verified"));
        }
        let mode = match config.endpoint.mode {
            PalsEngineModeV5::Deployment => VirtualReviewExecutionMode::Deployment,
            PalsEngineModeV5::Collection => VirtualReviewExecutionMode::Collection,
        };
        let kind = if config.native_options.approximate_warm {
            VirtualReviewArtifactKind::ApproxWarm
        } else {
            VirtualReviewArtifactKind::Fresh
        };
        let manifest_raw = self.read(Purpose::ExportManifest, 256 * 1024)?;
        let manifest = validate_virtual_review_export_manifest_kind(&manifest_raw, mode, kind)?;
        let rules_raw = self.read(Purpose::RulesDescriptor, 64 * 1024)?;
        let rules = match config.source_policy {
            PalsSystemSourcePolicyV6::Strict => {
                validate_pals_virtual_rules_descriptor_bytes(&rules_raw)?
            }
            PalsSystemSourcePolicyV6::Observe => {
                observe_pals_virtual_rules_descriptor_bytes(&rules_raw)?
            }
        };
        if manifest.checkpoint_sha256 != parse_sha256(&model.weights.sha256)?
            || manifest.model_profile != PalsModelProfile::VirtualReviewV1
            || manifest.validator_present != (config.endpoint.mode == PalsEngineModeV5::Collection)
            || manifest.config != rules.config
            || manifest.rules_input_semantic_sha256 != rules.semantic_sha256
            || manifest.rules_encoder_source_sha256 != rules.encoder_source_sha256
            || manifest.rules_profile_descriptor_sha256 != rules.descriptor_sha256()
        {
            return Err(error(
                "PALS system export, weights or Rules descriptor differs",
            ));
        }
        match model.artifact_domain {
            PalsArtifactDomainV5::Untrained if manifest.trained || manifest.training_steps != 0 => {
                return Err(error("untrained declaration contains trained export"));
            }
            PalsArtifactDomainV5::RealGeneration
                if !manifest.trained || manifest.training_steps == 0 =>
            {
                return Err(error("generation export does not match registered recipe"));
            }
            PalsArtifactDomainV5::FrozenPcCollection => {
                return Err(error("V6 frozen collection content admission unsupported"));
            }
            _ => {}
        }
        for graph in &manifest.graphs {
            validate_export_graph_name(&graph.file)?;
            let asset = self
                .owner
                .asset(graph_purpose(&graph.role)?)
                .map_err(transport)?;
            if parse_sha256(&asset.declaration().artifact.sha256)? != graph.sha256 {
                return Err(error("PALS system graph declaration differs from export"));
            }
        }
        self.owner
            .verify_source_unchanged()
            .map_err(transport_source)?;
        Ok(PalsSystemVirtualObservationV6 {
            rules,
            declared_model_source: model.implementation_sha256.clone(),
            declared_component_source,
            compiled_model_source,
            model_source_evidence,
            numerical_acceptance: None,
        })
    }
}
impl VirtualReviewOwnedSources for PalsSystemVirtualSourcesV6 {
    fn read_manifest(&self, maximum: usize) -> Result<Vec<u8>, BackendError> {
        self.read(Purpose::ExportManifest, maximum)
    }
    fn read_graph(
        &self,
        role: &str,
        registered_file: &str,
        expected: [u8; 32],
        maximum: usize,
    ) -> Result<Vec<u8>, BackendError> {
        let asset = self.owner.asset(graph_purpose(role)?).map_err(transport)?;
        validate_export_graph_name(registered_file)?;
        if parse_sha256(&asset.declaration().artifact.sha256)? != expected {
            return Err(error("PALS system graph differs from original export"));
        }
        self.read(graph_purpose(role)?, maximum)
    }
}

/// Original V6 system-runtime factory owner. This is intended to live inside
/// `TicketedNativeService::spawn_with_startup_custody`, so startup errors and
/// unwinds park this very owner instead of dropping possibly live native state.
/// It is not itself a worker, launch supervisor, or GPU completion witness.
#[cfg(feature = "onnx-cuda")]
#[allow(dead_code)]
pub(crate) struct PalsSystemFactoryCustodyV6 {
    sources: Arc<PalsSystemVirtualSourcesV6>,
    review: rz_eval::pals_onnx::virtual_review::VirtualReviewStartupCustody,
    system_owner: Option<rz_eval::onnx::SystemRuntime>,
    rejected_runtime: Option<rz_eval::onnx::OrtRuntime>,
    startup: Option<crate::pals_adaptive_bootstrap::runtime_assets::system_cuda::SystemCudaStartup>,
    observation: Option<PalsSystemVirtualObservationV6>,
    ready_mappings:
        Option<crate::pals_adaptive_bootstrap::runtime_assets::system_cuda::SystemReadyMappings>,
    ticket: Option<crate::pals_native::ticketed::NativeVirtualReviewTicketBackend>,
    observer: Option<Box<dyn crate::pals_native::ticketed::VirtualReviewCollectionObserver>>,
    parent: Arc<crate::rovent_bootstrap::launch::ConnectedParent>,
    physical_reservation: Option<rz_eval::worker::PhysicalRecoveryReservation>,
    physical_key: Option<rz_eval::worker::PhysicalRecoveryKey>,
}

#[cfg(feature = "onnx-cuda")]
impl PalsSystemFactoryCustodyV6 {
    pub(crate) fn new(
        owner: Arc<PinnedPalsSystemStartupV6>,
        observer: Option<Box<dyn crate::pals_native::ticketed::VirtualReviewCollectionObserver>>,
        parent: Arc<crate::rovent_bootstrap::launch::ConnectedParent>,
    ) -> Self {
        use rz_eval::pals_onnx::virtual_review::VirtualReviewStartupCustody;

        let sources = Arc::new(PalsSystemVirtualSourcesV6::new(owner));
        let review = VirtualReviewStartupCustody::new(
            Arc::clone(&sources) as Arc<dyn VirtualReviewOwnedSources>,
            None,
        );
        Self {
            sources,
            review,
            system_owner: None,
            rejected_runtime: None,
            startup: None,
            observation: None,
            ready_mappings: None,
            ticket: None,
            observer,
            parent,
            physical_reservation: None,
            physical_key: None,
        }
    }

    /// Reserve the original recovery slot and authenticate the original live
    /// parent before entering ORT. The caller must keep the same factory worker
    /// and supervisor after every error; it must never treat this return as a
    /// provider, session, or physical-completion proof.
    pub(crate) fn initialize_runtime(
        factory: &mut crate::pals_native::ticketed::FactoryStartupCustody<Self>,
        parent: &Arc<crate::rovent_bootstrap::launch::ConnectedParent>,
    ) -> Result<(), BackendError> {
        use rz_eval::error::{FailureKind, FailureStage};
        use rz_experiments::NativeRuntimeSourceV6;

        let fail = |kind, detail| BackendError::new(kind, FailureStage::Admission, detail);
        let (epoch, digest, ordinal) = {
            let (owner, entry) = factory.parts();
            if entry.entered()
                || owner.system_owner.is_some()
                || owner.startup.is_some()
                || owner.physical_reservation.is_some()
                || !Arc::ptr_eq(&owner.parent, parent)
            {
                return Err(fail(
                    FailureKind::InvalidInput,
                    "PALS system factory already entered",
                ));
            }
            let lock = owner.sources.owner.lock();
            let NativeRuntimeSourceV6::SystemCuda(runtime) =
                &lock.config.system_runtime_lock.runtime
            else {
                return Err(fail(
                    FailureKind::IdentityMismatch,
                    "PALS system factory requires CUDA",
                ));
            };
            (
                lock.config.process_generation,
                lock.canonical_sha256.clone(),
                runtime.cuda_device_id,
            )
        };
        if !parent.matches(epoch, &digest, ordinal) || parent.ensure_live().is_err() {
            return Err(fail(
                FailureKind::IdentityMismatch,
                "PALS original system supervisor differs or is unavailable",
            ));
        }
        let (reservation, key) = rz_eval::worker::PhysicalRecoveryReservation::reserve()?;
        factory.bind_inner(&key, epoch).map_err(|_| {
            fail(
                FailureKind::ResourceExhausted,
                "PALS original physical owner could not be bound",
            )
        })?;
        let (owner, entry) = factory.parts();
        owner.physical_reservation = Some(reservation);
        owner.physical_key = Some(key);
        let sources = Arc::clone(&owner.sources);
        let (startup, observation) = sources.inspect_and_load_system_cuda_with(|system| {
            owner.system_owner = Some(system.clone());
            parent.ensure_live().map_err(|_| {
                fail(
                    FailureKind::BackendUnavailable,
                    "PALS original system supervisor is no longer live",
                )
            })?;
            let runtime = rz_eval::onnx::OrtRuntime::load_system_cuda_with_entry(system, || {
                entry.mark_native_entered()
            })?;
            if let Err((error, rejected)) = owner.review.install_runtime(runtime.clone()) {
                owner.rejected_runtime = Some(rejected);
                return Err(error);
            }
            Ok(runtime)
        })?;
        owner.startup = Some(startup);
        owner.observation = Some(observation);
        Ok(())
    }

    /// Load sessions on the original factory thread. The owner retains the
    /// already-created sessions on every error or unwind. Session readiness is
    /// not GPU completion, a model acceptance, or arena launch authority.
    pub(crate) fn initialize_sessions(
        factory: &mut crate::pals_native::ticketed::FactoryStartupCustody<Self>,
        parent: &Arc<crate::rovent_bootstrap::launch::ConnectedParent>,
    ) -> Result<(), BackendError> {
        use rz_eval::onnx::Provider;
        use rz_eval::pals_onnx::virtual_review::PalsVirtualReviewOnnxBackend;
        use rz_eval::pals_onnx::{PalsCuDnnConvSearch, PalsOnnxConfig};
        use rz_experiments::NativeRuntimeSourceV6;

        let fail = |kind, detail| BackendError::new(kind, FailureStage::Admission, detail);
        let (owner, entry) = factory.parts();
        if !entry.entered()
            || owner.startup.is_none()
            || owner.observation.is_none()
            || owner.review.backend().is_some()
            || !Arc::ptr_eq(&owner.parent, parent)
        {
            return Err(fail(
                FailureKind::InvalidInput,
                "PALS system runtime is not ready for sessions",
            ));
        }
        let lock = owner.sources.owner.lock();
        let NativeRuntimeSourceV6::SystemCuda(runtime) = &lock.config.system_runtime_lock.runtime
        else {
            return Err(fail(
                FailureKind::IdentityMismatch,
                "PALS system factory requires CUDA",
            ));
        };
        if !parent.matches(
            lock.config.process_generation,
            &lock.canonical_sha256,
            runtime.cuda_device_id,
        ) || parent.ensure_live().is_err()
        {
            return Err(fail(
                FailureKind::IdentityMismatch,
                "PALS original system supervisor differs or is unavailable",
            ));
        }
        owner
            .sources
            .owner
            .verify_source_unchanged()
            .map_err(transport_source)?;
        let options = &lock.config.native_options;
        let cudnn_conv_search = match options.cudnn_conv_search.as_str() {
            "legacy_implicit_exhaustive" => PalsCuDnnConvSearch::LegacyImplicitExhaustive,
            "experimental_default" => PalsCuDnnConvSearch::ExperimentalDefault,
            _ => {
                return Err(fail(
                    FailureKind::InvalidInput,
                    "PALS system cuDNN option is unsupported",
                ));
            }
        };
        let arena_bytes = usize::try_from(runtime.cuda_arena_bytes).map_err(|_| {
            fail(
                FailureKind::ResourceExhausted,
                "PALS system CUDA arena exceeds host address space",
            )
        })?;
        let configuration = PalsOnnxConfig {
            provider: Provider::Cuda {
                device_id: runtime.cuda_device_id,
                arena_bytes,
            },
            intra_threads: 1,
            cache_public_memory: options.cache_public_memory,
            device_public_memory: options.device_public_memory,
            cudnn_conv_search,
        };
        configuration.validate_system_execution_selection()?;
        let mode = match lock.config.endpoint.mode {
            PalsEngineModeV5::Deployment => VirtualReviewExecutionMode::Deployment,
            PalsEngineModeV5::Collection => VirtualReviewExecutionMode::Collection,
        };
        let kind = if options.approximate_warm {
            VirtualReviewArtifactKind::ApproxWarm
        } else {
            VirtualReviewArtifactKind::Fresh
        };
        let manifest_sha = &owner
            .sources
            .owner
            .asset(Purpose::ExportManifest)
            .map_err(transport)?
            .declaration()
            .artifact
            .sha256;
        PalsVirtualReviewOnnxBackend::load_owned_sources_with_profile_into(
            &mut owner.review,
            manifest_sha,
            configuration,
            mode,
            kind,
            None,
            &mut || entry.mark_native_entered(),
        )?;
        owner
            .sources
            .owner
            .verify_source_unchanged()
            .map_err(transport_source)?;
        let backend = owner.review.backend().ok_or_else(|| {
            fail(
                FailureKind::BackendFailure,
                "PALS original session owner is unavailable",
            )
        })?;
        if backend.validator_present() != (mode == VirtualReviewExecutionMode::Collection)
            || backend.approximate_warm_capable() != options.approximate_warm
            || backend.model_epoch() != parse_sha256(&lock.config.endpoint.model.weights.sha256)?
        {
            return Err(fail(
                FailureKind::IdentityMismatch,
                "PALS system sessions differ from declared model or mode",
            ));
        }
        let sessions = backend.applied_session_settings();
        let residency = backend.residency();
        if sessions.is_empty() || sessions.len() != residency.native_sessions || sessions.len() != residency.graphs.len()
            || sessions.iter().any(|session| {
                let provider_matches = matches!(session.provider(), Provider::Cuda { device_id, arena_bytes: bytes }
                    if device_id == runtime.cuda_device_id && bytes == arena_bytes);
                !provider_matches || session.cuda_provider_available() != Some(true)
                    || residency.graphs.iter().filter(|graph| graph.role == session.role()).count() != 1
                    || sessions.iter().filter(|other| other.role() == session.role()).count() != 1
            })
        {
            return Err(fail(FailureKind::IdentityMismatch, "PALS system sessions did not confirm selected CUDA provider"));
        }
        owner.ready_mappings = Some(
            owner
                .startup
                .as_ref()
                .expect("checked startup")
                .observe_ready_mappings()?,
        );
        Ok(())
    }

    /// Hand the exact retained backend to its physical worker, while keeping
    /// the ticket in factory custody until the caller publishes it. A failure
    /// must leave this same custody on the original factory thread.
    pub(crate) fn prepare_ticket(
        factory: &mut crate::pals_native::ticketed::FactoryStartupCustody<Self>,
        clock: rz_runtime::contracts::ContractSystemClock,
    ) -> Result<(), rz_search::pals::coordinator::RoleFailure> {
        use crate::pals_native::ticketed::NativeVirtualReviewTicketBackend;
        use rz_runtime::contracts::ContractClock;
        use rz_search::pals::coordinator::RoleFailure;
        use std::time::Duration;

        let (owner, entry) = factory.parts();
        let lock = owner.sources.owner.lock();
        if !entry.entered()
            || owner.startup.is_none()
            || owner.ready_mappings.is_none()
            || owner.ticket.is_some()
            || clock.domain().0.0 != lock.config.process_generation
        {
            return Err(RoleFailure::InvalidOutput);
        }
        let reservation = owner
            .physical_reservation
            .as_mut()
            .ok_or(RoleFailure::InvalidOutput)?;
        let ticket = NativeVirtualReviewTicketBackend::new_with_collection_observer_retained(
            owner.review.backend_slot_mut(),
            reservation,
            clock,
            Duration::from_millis(lock.config.native_options.drain_ms),
            &mut owner.observer,
        )?;
        owner.ticket = Some(ticket);
        Ok(())
    }

    /// Transfer only the ticket already prepared by this same factory. The
    /// spawn closure must return it directly and preserve custody on failure.
    pub(crate) fn take_prepared_ticket(
        &mut self,
    ) -> Option<crate::pals_native::ticketed::NativeVirtualReviewTicketBackend> {
        self.ticket.take()
    }

    /// Return the same factory's startup observation. It remains a child
    /// report until independently bound by the parent, and deliberately says
    /// physical completion is unknown regardless of provider readiness.
    pub(crate) fn startup_runtime_report(
        &self,
    ) -> Result<rz_experiments::SystemRuntimeReceiptV6, BackendError> {
        use rz_experiments::{PhysicalCompletionV6, ProviderReadinessV6, SystemRuntimeReceiptV6};
        let fail = |detail| {
            BackendError::new(
                FailureKind::BackendUnavailable,
                FailureStage::Backend,
                detail,
            )
        };
        let mappings = self
            .ready_mappings
            .as_ref()
            .ok_or_else(|| fail("PALS system mappings not observed"))?;
        let startup = self
            .startup
            .as_ref()
            .ok_or_else(|| fail("PALS original system runtime unavailable"))?;
        let runtime = startup.runtime();
        let identity = runtime
            .cuda_runtime_identity()
            .ok_or_else(|| fail("PALS system runtime identity unavailable"))?;
        let lock = self.sources.owner.lock();
        let receipt = SystemRuntimeReceiptV6 {
            domain: rz_experiments::SYSTEM_RUNTIME_RECEIPT_V6_DOMAIN.into(),
            lock_sha256: lock.config.system_runtime_lock.canonical_sha256.clone(),
            process_id: std::process::id(),
            process_generation: lock.config.process_generation,
            observed_identity_sha256: Some(hex(identity)),
            observed_libraries: mappings.libraries.clone(),
            unregistered_native_mappings: mappings.unregistered_native_mappings.clone(),
            provider_ready: ProviderReadinessV6::Ready,
            physical_completion: PhysicalCompletionV6::Unknown,
        };
        receipt
            .validate_against(
                &lock.config.system_runtime_lock,
                std::process::id(),
                lock.config.process_generation,
            )
            .map_err(transport)?;
        Ok(receipt)
    }
}

#[cfg(feature = "onnx-cuda")]
impl crate::pals_native::ticketed::FactoryStartupRecovery for PalsSystemFactoryCustodyV6 {
    fn reobserve(
        &mut self,
        until: std::time::Instant,
    ) -> Result<
        crate::pals_native::ticketed::TicketNativeRecoveryProbe,
        rz_search::pals::coordinator::RoleFailure,
    > {
        match self.ticket.as_mut() {
            Some(ticket) => ticket.reobserve_quarantined(until),
            None => Ok(crate::pals_native::ticketed::TicketNativeRecoveryProbe::Unknown),
        }
    }

    fn release(
        &mut self,
        proof: &rz_eval::worker::PhysicalRecoveryConfirmation,
        until: std::time::Instant,
    ) -> Result<
        crate::pals_native::ticketed::TicketNativeRelease,
        rz_search::pals::coordinator::RoleFailure,
    > {
        match self.ticket.as_mut() {
            Some(ticket) => ticket.release_quarantined(proof, until),
            None => Ok(crate::pals_native::ticketed::TicketNativeRelease::Unknown),
        }
    }
}
