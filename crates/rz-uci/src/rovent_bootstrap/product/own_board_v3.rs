//! V3 owner for the existing material/PST board researcher.
//!
//! The native service, original process boundary and CPU workers remain in
//! caller-owned slots through assembly.  Root budgets are bound by the actual
//! V3 `go` admission; no historical lock or placeholder budget is created.

use super::*;
use rz_experiments::{RoventInputLockV3, RoventPurpose};

/// Startup custody is installed in the caller's original scope before the
/// native worker is started.  A missing slot never implies physical release.
pub(crate) struct RoventOwnBoardAssemblyV3 {
    prepared: Option<PreparedRoventNativeV3<AdaptivePortQuestion>>,
    sources: Option<AdaptiveSessionSources<RoventNativeBackend<AdaptivePortQuestion>>>,
    session: Option<PcrDriver>,
    process: Arc<VerifiedOwnedRoventProcessBoundaryV3>,
    product: Option<Arc<RoventOwnBoardProductDriverV3>>,
    recovery_key: Option<crate::pals_native::ticketed::TicketNativeRecoveryKey>,
}

impl RoventOwnBoardAssemblyV3 {
    pub(crate) fn new(process: Arc<VerifiedOwnedRoventProcessBoundaryV3>) -> Self {
        Self {
            prepared: None,
            sources: None,
            session: None,
            process,
            product: None,
            recovery_key: None,
        }
    }

    #[allow(clippy::result_large_err)]
    pub(crate) fn install_prepared(
        &mut self,
        prepared: PreparedRoventNativeV3<AdaptivePortQuestion>,
    ) -> Result<(), PreparedRoventNativeV3<AdaptivePortQuestion>> {
        if self.prepared.is_some()
            || self.sources.is_some()
            || self.session.is_some()
            || self.product.is_some()
        {
            return Err(prepared);
        }
        self.recovery_key = prepared.service.recovery_key();
        self.prepared = Some(prepared);
        Ok(())
    }

    /// Only borrows the original native owner for bounded recovery.  This is
    /// neither a completion proof nor permission to create a replacement.
    pub(crate) fn prepared_service(
        &mut self,
    ) -> Option<&mut super::super::RoventNativeService<AdaptivePortQuestion>> {
        if let Some(prepared) = &mut self.prepared {
            return Some(&mut prepared.service);
        }
        self.sources.as_mut().map(|sources| &mut sources.native)
    }

    pub(crate) fn system_runtime_observation(
        &self,
    ) -> Option<super::super::RoventSystemReadyObservationV3> {
        self.prepared
            .as_ref()
            .and_then(|prepared| prepared.snapshot().system_runtime.clone())
    }

    pub(crate) fn recovery_key(
        &self,
    ) -> Option<crate::pals_native::ticketed::TicketNativeRecoveryKey> {
        self.recovery_key.clone()
    }

    /// If root-policy admission failed after the original native service moved
    /// into a session, only this same session can finish its physical owners.
    pub(crate) fn session(&self) -> Option<&PcrDriver> {
        self.session.as_ref()
    }

    pub(crate) fn assemble(
        &mut self,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
    ) -> Result<Arc<RoventOwnBoardProductDriverV3>, SearchSessionFailure> {
        if self.sources.is_some() || self.session.is_some() || self.product.is_some() {
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        let prepared = self
            .prepared
            .as_ref()
            .ok_or_else(SearchSessionFailure::physical_completion_unknown)?;
        let pins = Arc::clone(prepared.sources());
        if !Arc::ptr_eq(pins.lock_owner(), self.process.input_lock())
            || !self
                .process
                .matches_startup_control(prepared.startup_control())
            || pins.private_root_path() != self.process.private_cwd()
            || !pins
                .same_original_root(self.process.original_root())
                .map_err(|error| SearchSessionFailure::debug("RoventOwnBoardV3Root", &error))?
        {
            return Err(failure(
                "RoventOwnBoardV3Custody",
                "original /3 source, process and control differ",
            ));
        }
        if !matches!(
            pins.lock().manifest.purpose,
            RoventPurpose::Validation | RoventPurpose::Deployment
        ) || pins.collection_preload_sources().is_some()
        {
            return Err(failure(
                "RoventOwnBoardV3CollectionUnsupported",
                "V3 Collection requires its own original hook owner",
            ));
        }
        pins.configuration()
            .require_supported_native_capabilities()
            .map_err(|error| SearchSessionFailure::debug("RoventOwnBoardV3Capability", &error))?;
        if !super::super::pc_session::runtime_matches_v3(pins.lock(), prepared.snapshot())? {
            return Err(failure(
                "RoventOwnBoardV3Runtime",
                "actual native runtime differs from the original /3 lock",
            ));
        }
        if identity.name.is_empty() || identity.author.is_empty() {
            return Err(failure(
                "RoventOwnBoardV3Identity",
                "engine identity is empty",
            ));
        }
        let ready = prepared
            .service
            .ready_clock()
            .ok_or_else(|| failure("RoventOwnBoardV3Ready", "native Ready clock is absent"))?;
        let expected = clock.runtime_clock();
        if prepared.service.is_quarantined()
            || prepared.service.native_acceptance_verifier().is_none()
            || ready.domain() != expected.domain()
            || ready.origin() != expected.origin()
            || prepared.snapshot().input_lock != pins.lock_digest()
        {
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        let (engine, cpu, metadata, registration, options, limits, profile) =
            build_own_board_pcr!(v3, pins, prepared, clock, identity)?;
        validate_v2_own_worker_identities(
            &pins.lock().manifest.own_r_root_tie,
            cpu.iter().map(|owner| &owner.engine),
        )?;

        // No fallible operation occurs between taking the original service and
        // placing it into the caller's still-live custody slot.
        let (native, _, _) = self
            .prepared
            .take()
            .expect("validated prepared owner")
            .into_parts();
        self.sources = Some(AdaptiveSessionSources {
            native,
            engine,
            cpu,
            external: None,
            checker_metadata: metadata,
        });
        let issuer: AdaptiveRulesOwnerIssuer = Arc::new(move || owners.allocate());
        PcrDriver::admit_rovent_own_v3_in_slots(
            options,
            limits,
            &mut self.sources,
            &mut self.session,
            clock.clone(),
            issuer,
            registration,
            profile,
            Arc::clone(pins.lock_owner()),
            root_policy,
        )?;
        let session = self
            .session
            .take()
            .ok_or_else(SearchSessionFailure::physical_completion_unknown)?;
        let product = Arc::new(RoventOwnBoardProductDriverV3 {
            inner: Arc::new(session),
            sources: pins,
            process: Arc::clone(&self.process),
            physically_closed: Arc::new(AtomicBool::new(false)),
            process_clock: clock,
            storage_registration: std::sync::Mutex::new(None),
        });
        self.product = Some(Arc::clone(&product));
        Ok(product)
    }
}

/// Actual own CPU and native owners stay together until the same session
/// reports physical completion.  This is not a device-wide GPU fence.
pub(crate) struct RoventOwnBoardProductDriverV3 {
    pub(super) inner: Arc<PcrDriver>,
    pub(super) sources: Arc<RoventPinnedNativeSources<RoventInputLockV3>>,
    pub(super) process: Arc<VerifiedOwnedRoventProcessBoundaryV3>,
    pub(super) physically_closed: Arc<AtomicBool>,
    pub(super) process_clock: ProcessClock,
    pub(super) storage_registration: std::sync::Mutex<Option<storage::RetainedStorageRegistration>>,
}

impl RoventOwnBoardProductDriverV3 {
    pub(crate) fn physical_close_observation(&self) -> RoventProductPhysicalClose {
        RoventProductPhysicalClose {
            closed: Arc::clone(&self.physically_closed),
        }
    }

    /// Reports only the release observation emitted by this exact session.
    pub(crate) fn session_release_observation(
        &self,
    ) -> Option<crate::pals_adaptive::NativeSessionReleaseObservation> {
        self.inner.closed_native_session_release().ok().flatten()
    }
}

impl SearchSessionDriver for RoventOwnBoardProductDriverV3 {
    fn supports_ponder_clock(&self) -> bool {
        self.inner.supports_ponder_clock()
    }
    fn node_budget_semantics(&self) -> crate::bridge::NodeBudgetSemantics {
        self.inner.node_budget_semantics()
    }
    fn validate_go(
        &self,
        position: &Position,
        go: &crate::parser::GoLimits,
    ) -> Result<(), ContractError> {
        self.inner.validate_go(position, go)
    }
    fn kind(&self) -> SearchKind {
        SearchKind::Rovent
    }
    fn identity(&self) -> EngineIdentity {
        self.inner.identity()
    }
    fn implementation(&self) -> Digest {
        self.inner.implementation()
    }
    fn search_goal_capability(&self) -> Option<NonZeroU16> {
        self.inner.search_goal_capability()
    }
    fn reset_game(&self) -> Result<(), ContractError> {
        self.inner.reset_game()
    }
    fn finish_session(
        &self,
        until: Instant,
    ) -> Result<Option<DriverPhysicalCompletion>, SearchSessionFailure> {
        let result = self.inner.finish_session(until);
        if matches!(result, Ok(Some(DriverPhysicalCompletion::Confirmed))) {
            self.physically_closed.store(true, Ordering::Release);
        }
        result
    }
    fn accept_uci_output(
        &self,
        authority: rz_contracts::pals::SearchAuthority,
        bestmove: &str,
        from_report: bool,
    ) -> Result<(), ContractError> {
        self.inner
            .accept_uci_output(authority, bestmove, from_report)
    }
    fn run(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        self.inner.run(position, context, progress)
    }
}

impl Drop for RoventOwnBoardProductDriverV3 {
    fn drop(&mut self) {
        if !self.physically_closed.load(Ordering::Acquire) {
            std::mem::forget(Arc::clone(&self.inner));
            std::mem::forget(Arc::clone(&self.sources));
            std::mem::forget(Arc::clone(&self.process));
        }
    }
}
