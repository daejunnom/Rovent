//! Immutable checker registration for a Search owner whose actual CPU execution
//! lives on private workers. This mirror owns no engine, TT, token or process.
//! Metadata never certifies physical startup, analysis or shutdown.

use rz_contracts::pals::SearchAuthority;
use rz_contracts::pals_adaptive::CpuWorkerId;
use rz_contracts::{Digest, ProcessEpoch};
use rz_position::{BoardMove, Position};
use rz_search::cpu::{CpuEngine, CpuLimits, CpuResumePolicy, CpuResumeToken, CpuSearcher};
use rz_search::cpu_checker::{
    CheckerAttempt, CheckerCapabilities, CheckerError, CheckerIdentity, CheckerReport,
    CheckerShutdown, CpuChecker, ExternalProcessIdentity, OwnedCheckerDescriptor,
};
use rz_search::external_cpu::worker::{
    ExternalCpuWorker, ExternalInitialGameBinding, ExternalNewGameFence,
    external_evaluation_namespace,
};
use rz_search::external_cpu::{ExternalCpuConfig, ExternalUciCpuChecker};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[cfg(feature = "pals-adaptive")]
mod profiled_external;
#[cfg(feature = "pals-adaptive")]
use profiled_external::ProfiledExternalMetadataBinding;

const MAX_CONDITIONS_BYTES: usize = 8192;
const METADATA_ONLY: &str = "adaptive metadata mirror has no physical checker owner";

/// A declaration snapshot. Construction validates consistency, not the actual
/// execution or provenance of the represented owner. Capture before moving the
/// real owner to its worker; never reconstruct its TT from this descriptor.
#[derive(Clone, Debug)]
pub struct AdaptiveCheckerMetadata {
    identity: CheckerIdentity,
    conditions: String,
    capabilities: CheckerCapabilities,
    owned_descriptor: Option<OwnedCheckerDescriptor>,
    resume_policy: Option<CpuResumePolicy>,
    // Minted only by capture_owned from the concrete engine's actual owner
    // snapshot. Namespace equality alone cannot identify a live CPU owner.
    actual_owner_id: Option<u64>,
    reset: Arc<Mutex<ResetState>>,
}

#[derive(Debug, Default)]
struct ResetState {
    sequence: u64,
    pending: Option<MetadataResetArm>,
    external: Option<ExternalOwnerBinding>,
    #[cfg(feature = "pals-adaptive")]
    profiled: Option<ProfiledExternalMetadataBinding>,
}

#[derive(Clone, Copy, Debug)]
#[allow(
    clippy::large_enum_variant,
    reason = "the bounded reset arm keeps its original Copy fence inline without a reset-path allocation"
)]
enum MetadataResetArm {
    Owned {
        deadline: Instant,
    },
    InitialExternal {
        deadline: Instant,
    },
    External {
        fence: ExternalNewGameFence,
    },
    #[cfg(feature = "pals-adaptive")]
    ProfiledExternal {
        deadline: Instant,
    },
}

#[derive(Debug)]
struct ExternalOwnerBinding {
    owner: CpuWorkerId,
    epoch: ProcessEpoch,
    implementation: Digest,
    configuration: ExternalCpuConfig,
    process_identity: ExternalProcessIdentity,
    last_reset_high_water: u64,
    last_authority: Option<SearchAuthority>,
}

impl AdaptiveCheckerMetadata {
    pub fn new(
        identity: CheckerIdentity,
        conditions: String,
        capabilities: CheckerCapabilities,
        owned_descriptor: Option<OwnedCheckerDescriptor>,
        resume_policy: Option<CpuResumePolicy>,
    ) -> Result<Self, CheckerError> {
        identity.validate()?;
        if conditions.trim().is_empty()
            || conditions.len() > MAX_CONDITIONS_BYTES
            || conditions.capacity() > MAX_CONDITIONS_BYTES
            || conditions.chars().any(char::is_control)
            || capabilities.max_depth == 0
            || capabilities.max_depth > 64
            || capabilities.max_root_moves > 256
            || capabilities.max_prefix_plies > 4096
        {
            return Err(CheckerError::Invalid(
                "adaptive checker metadata exceeds bounds",
            ));
        }
        match &identity {
            CheckerIdentity::Owned(_) => {
                let descriptor = owned_descriptor.as_ref().ok_or(CheckerError::Invalid(
                    "owned adaptive metadata requires actual captured descriptor",
                ))?;
                if resume_policy.is_none()
                    || descriptor.config.max_depth == 0
                    || descriptor.config.max_depth > 64
                    || descriptor.config.quiescence_ply > 32
                    || descriptor.config.tt_entries > 1_048_576
                    || descriptor.search_identity.trim().is_empty()
                    || descriptor.search_identity.len() > 256
                    || descriptor.search_identity.chars().any(char::is_control)
                    || descriptor.search_conditions != conditions
                    || descriptor.search_conditions.capacity() > MAX_CONDITIONS_BYTES
                    || descriptor.capabilities.max_prefix_plies > 64
                    || descriptor.capabilities.max_root_moves > 256
                    || descriptor.config.max_depth != descriptor.capabilities.max_depth
                    || checker_capabilities(descriptor.capabilities) != capabilities
                {
                    return Err(CheckerError::Invalid(
                        "owned adaptive metadata differs from descriptor",
                    ));
                }
            }
            CheckerIdentity::ExternalUci(_) => {
                if owned_descriptor.is_some()
                    || resume_policy.is_some()
                    || capabilities.resume
                    || capabilities.selective_search.is_some()
                {
                    return Err(CheckerError::Invalid(
                        "external metadata cannot claim own CPU semantics",
                    ));
                }
            }
        }
        Ok(Self {
            identity,
            conditions,
            capabilities,
            owned_descriptor,
            resume_policy,
            actual_owner_id: None,
            reset: Arc::new(Mutex::new(ResetState::default())),
        })
    }

    /// Read only the existing checker's registered declarations. Its startup,
    /// attempt, process and paused-token evidence is deliberately not copied.
    pub fn capture(checker: &dyn CpuChecker) -> Result<Self, CheckerError> {
        checker.validate_namespace()?;
        Self::new(
            checker.identity().clone(),
            checker.conditions().to_owned(),
            checker.capabilities(),
            checker.owned_descriptor(),
            checker.owned_resume_policy(),
        )
    }

    /// Capture from the actual engine before ownership transfer, without
    /// constructing OwnedCpuChecker or allocating a second TT/engine.
    pub fn capture_owned(engine: &CpuEngine) -> Result<Self, CheckerError> {
        let conditions = engine.search_conditions();
        let native_capabilities = CpuSearcher::capabilities(engine);
        let descriptor = OwnedCheckerDescriptor {
            config: engine.config().clone(),
            search_identity: engine.search_identity(),
            search_conditions: conditions.clone(),
            capabilities: native_capabilities,
        };
        let mut metadata = Self::new(
            CheckerIdentity::Owned(engine.value_identity().clone()),
            conditions,
            checker_capabilities(native_capabilities),
            Some(descriptor),
            Some(engine.resume_policy()),
        )?;
        metadata.actual_owner_id = engine
            .paused_stack_snapshot()
            .filter(|snapshot| snapshot.complete && snapshot.owner_id != 0)
            .map(|snapshot| snapshot.owner_id);
        Ok(metadata)
    }

    pub fn identity(&self) -> &CheckerIdentity {
        &self.identity
    }
    pub fn conditions(&self) -> &str {
        &self.conditions
    }
    pub fn capabilities(&self) -> CheckerCapabilities {
        self.capabilities
    }
    pub fn owned_descriptor(&self) -> Option<OwnedCheckerDescriptor> {
        self.owned_descriptor.clone()
    }
    pub fn resume_policy(&self) -> Option<CpuResumePolicy> {
        self.resume_policy
    }

    pub fn into_mirror(self) -> AdaptiveCheckerMetadataMirror {
        AdaptiveCheckerMetadataMirror { metadata: self }
    }

    /// Off-manager lifecycle operation on the caller's actual, idle primary
    /// engine. Only this successful reset publishes one metadata-consumption
    /// opportunity; there is no boolean or caller-mintable reset permit.
    /// Additional workers must be independently reset by their owners.
    pub fn reset_owned_and_arm(
        &self,
        engine: &mut CpuEngine,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<(), CheckerError> {
        let mut reset = self
            .reset
            .lock()
            .map_err(|_| CheckerError::Invalid("metadata reset owner poisoned"))?;
        if reset.pending.is_some() {
            return Err(CheckerError::Invalid("metadata reset already armed"));
        }
        let sequence = reset
            .sequence
            .checked_add(1)
            .ok_or(CheckerError::Invalid("metadata reset sequence exhausted"))?;
        check_controls(deadline, cancel)?;
        self.matches_owned(engine)?;
        engine.clear();
        self.matches_owned(engine)?;
        let snapshot = engine
            .paused_stack_snapshot()
            .ok_or(CheckerError::Unsupported(
                "actual CPU reset owner readback unavailable",
            ))?;
        if snapshot.tokens_retained != 0 || snapshot.bytes_current != 0 {
            return Err(CheckerError::Invalid(
                "actual CPU reset still retains paused work",
            ));
        }
        check_controls(deadline, cancel)?;
        reset.sequence = sequence;
        reset.pending = Some(MetadataResetArm::Owned { deadline });
        Ok(())
    }

    /// Exact actual primary-owner check for startup admission. Additional own
    /// workers share the namespace, not this primary's opaque frame owner ID.
    pub fn validate_owned_engine(&self, engine: &CpuEngine) -> Result<(), CheckerError> {
        self.matches_owned(engine)
    }

    /// Declaration consistency only; this never arms a game reset or transfers
    /// a paused token to another same-namespace worker.
    pub fn validate_owned_namespace(&self, engine: &CpuEngine) -> Result<(), CheckerError> {
        let actual = Self::capture_owned(engine)?;
        self.same_owned_namespace(&actual)
    }

    /// Bind expectations only after the existing concrete external worker's
    /// actual successful startup. Its retained PID/group/start-time identity is
    /// fixed now, not adopted from a later worker's same-namespace fence. Epoch
    /// and implementation are subsequently checked against actual reset proof.
    /// This does not spawn, reset or issue any UCI command.
    pub fn bind_external_owner(
        &self,
        worker: &ExternalCpuWorker,
        config: &ExternalCpuConfig,
        owner: CpuWorkerId,
        epoch: ProcessEpoch,
        implementation: Digest,
    ) -> Result<(), CheckerError> {
        self.matches_external_config(config)?;
        if !worker.configuration_matches(config, owner) {
            return Err(CheckerError::Invalid(
                "external metadata actual owner/config mismatch",
            ));
        }
        let startup = worker.startup_report().ok_or(CheckerError::Unsupported(
            "actual external startup not observed before metadata binding",
        ))?;
        startup.result?;
        let process = startup
            .attempt
            .and_then(|attempt| attempt.external)
            .ok_or(CheckerError::Invalid(
                "actual external startup process evidence missing",
            ))?
            .process;
        let process_identity = live_external_process(process)?;
        let mut reset = self
            .reset
            .lock()
            .map_err(|_| CheckerError::Invalid("metadata reset owner poisoned"))?;
        #[cfg(feature = "pals-adaptive")]
        if reset.profiled.is_some() {
            return Err(CheckerError::Invalid(
                "external metadata profiled owner already bound",
            ));
        }
        if reset.pending.is_some() || reset.external.is_some() {
            return Err(CheckerError::Invalid(
                "external metadata owner already bound or armed",
            ));
        }
        reset.external = Some(ExternalOwnerBinding {
            owner,
            epoch,
            implementation,
            configuration: config.clone(),
            process_identity,
            last_reset_high_water: 0,
            last_authority: None,
        });
        Ok(())
    }

    /// The first semantic game may adopt a physically pristine startup owner.
    /// This token certifies that binding only, never a reset or old-root fence.
    pub fn arm_external_initial_game(
        &self,
        worker: &ExternalCpuWorker,
        config: &ExternalCpuConfig,
        proof: ExternalInitialGameBinding,
    ) -> Result<(), CheckerError> {
        self.matches_external_config(config)?;
        if !worker.configuration_matches(config, proof.owner())
            || proof.source() != external_evaluation_namespace(&config.identity)?
        {
            return Err(CheckerError::Invalid(
                "external initial game actual owner differs",
            ));
        }
        let mut reset = self
            .reset
            .lock()
            .map_err(|_| CheckerError::Invalid("metadata reset owner poisoned"))?;
        if reset.pending.is_some() {
            return Err(CheckerError::Invalid("metadata reset already armed"));
        }
        let sequence = reset
            .sequence
            .checked_add(1)
            .ok_or(CheckerError::Invalid("metadata reset sequence exhausted"))?;
        let binding = reset.external.as_mut().ok_or(CheckerError::Unsupported(
            "actual external metadata owner was not bound",
        ))?;
        let next = proof.authority();
        if binding.owner != proof.owner()
            || binding.configuration != *config
            || binding.epoch != next.epoch
            || binding.implementation != next.implementation
            || binding.process_identity != proof.process_identity()
            || binding.last_authority.is_some()
            || binding.last_reset_high_water != 0
        {
            return Err(CheckerError::Invalid(
                "external initial game binding already used or foreign",
            ));
        }
        let deadline = worker.consume_initial_game_binding(proof)?;
        binding.last_authority = Some(next);
        reset.sequence = sequence;
        reset.pending = Some(MetadataResetArm::InitialExternal { deadline });
        Ok(())
    }

    /// An actual ucinewgame/ready fence grants one metadata lifecycle delivery.
    /// Its source, retained owner/process, epoch, implementation and high-water
    /// are checked; a copied old proof cannot grant another game's reset.
    pub fn arm_external_new_game(
        &self,
        config: &ExternalCpuConfig,
        proof: &ExternalNewGameFence,
    ) -> Result<(), CheckerError> {
        self.matches_external_config(config)?;
        let next = proof.authority();
        let previous = proof.previous_authority();
        let process = proof.process_identity().ok_or(CheckerError::Invalid(
            "external newgame actual process missing",
        ))?;
        let source = external_evaluation_namespace(&config.identity)?;
        if !proof.permits_reuse()
            || proof.source() != source
            || process.pid == 0
            || process.process_group == 0
            || process.proc_start_ticks == 0
            || next.epoch != previous.epoch
            || next.implementation != previous.implementation
            || next.game.0 <= previous.game.0
            || next.root.0 <= previous.root.0
        {
            return Err(CheckerError::Invalid(
                "external newgame opaque fence mismatch",
            ));
        }
        let mut reset = self
            .reset
            .lock()
            .map_err(|_| CheckerError::Invalid("metadata reset owner poisoned"))?;
        if reset.pending.is_some() {
            return Err(CheckerError::Invalid("metadata reset already armed"));
        }
        let sequence = reset
            .sequence
            .checked_add(1)
            .ok_or(CheckerError::Invalid("metadata reset sequence exhausted"))?;
        let binding = reset.external.as_mut().ok_or(CheckerError::Unsupported(
            "actual external metadata owner was not bound",
        ))?;
        if binding.configuration != *config
            || binding.owner != proof.owner()
            || binding.epoch != next.epoch
            || binding.implementation != next.implementation
            || binding.process_identity != process
            || proof.game_reset_high_water() <= binding.last_reset_high_water
            || binding.last_authority.is_some_and(|old| {
                previous.epoch != old.epoch
                    || previous.implementation != old.implementation
                    || previous.game.0 < old.game.0
                    || previous.root.0 < old.root.0
            })
        {
            return Err(CheckerError::Invalid(
                "external newgame retained owner/high-water mismatch",
            ));
        }
        binding.last_reset_high_water = proof.game_reset_high_water();
        binding.last_authority = Some(next);
        reset.sequence = sequence;
        reset.pending = Some(MetadataResetArm::External { fence: *proof });
        Ok(())
    }

    /// Discard an unconsumed arm after a lifecycle failure, using the same
    /// retained physical worker and its actual current fence. The sequence is
    /// an abandonment observation, not a new reset permit. The old high-water
    /// remains consumed: retry requires another actual new-game producer fence.
    pub fn abandon_external_reset_arm(
        &self,
        worker: &ExternalCpuWorker,
        config: &ExternalCpuConfig,
        proof: &ExternalNewGameFence,
    ) -> Result<u64, CheckerError> {
        self.matches_external_config(config)?;
        if !proof.permits_reuse()
            || !worker
                .new_game_fence()
                .is_some_and(|result| result.is_ok_and(|observed| observed == *proof))
        {
            return Err(CheckerError::Invalid(
                "external reset abandonment actual fence differs",
            ));
        }
        let mut reset = self
            .reset
            .lock()
            .map_err(|_| CheckerError::Invalid("metadata reset owner poisoned"))?;
        let binding = reset.external.as_ref().ok_or(CheckerError::Unsupported(
            "actual external metadata owner was not bound",
        ))?;
        if !worker.configuration_matches(config, binding.owner)
            || binding.configuration != *config
            || binding.owner != proof.owner()
            || binding.epoch != proof.authority().epoch
            || binding.implementation != proof.authority().implementation
            || Some(binding.process_identity) != proof.process_identity()
            || proof.source() != external_evaluation_namespace(&config.identity)?
            || binding.last_reset_high_water != proof.game_reset_high_water()
            || binding.last_authority != Some(proof.authority())
            || !matches!(reset.pending,
                Some(MetadataResetArm::External { fence }) if fence == *proof)
        {
            return Err(CheckerError::Invalid(
                "external reset abandonment retained owner or arm differs",
            ));
        }
        reset.pending = None;
        Ok(reset.sequence)
    }

    fn matches_external_config(&self, config: &ExternalCpuConfig) -> Result<(), CheckerError> {
        config.validate()?;
        // Reuse the existing declaration producer. create() is explicitly
        // unstarted: it does not open files, start a process or allocate a TT.
        let declared = ExternalUciCpuChecker::create(config.clone())?;
        if self.identity != *declared.identity()
            || self.conditions != declared.conditions()
            || self.capabilities != declared.capabilities()
            || self.owned_descriptor.is_some()
            || self.resume_policy.is_some()
        {
            return Err(CheckerError::Invalid(
                "external metadata configuration differs",
            ));
        }
        Ok(())
    }

    /// A lifecycle failure before mirror.new_game (for example model preflight)
    /// leaves its reset unconsumed. The same actual owner may explicitly revoke
    /// that arm, including after expiry. The returned sequence is an abandonment
    /// observation, never permission to reset another game; a retry must clear
    /// the actual owner again through reset_owned_and_arm.
    pub fn abandon_owned_reset_arm(&self, engine: &CpuEngine) -> Result<u64, CheckerError> {
        self.matches_owned(engine)?;
        let mut reset = self
            .reset
            .lock()
            .map_err(|_| CheckerError::Invalid("metadata reset owner poisoned"))?;
        if !matches!(reset.pending, Some(MetadataResetArm::Owned { .. })) {
            return Err(CheckerError::Invalid(
                "owned metadata reset arm already consumed or absent",
            ));
        }
        reset.pending = None;
        Ok(reset.sequence)
    }

    fn matches_owned(&self, engine: &CpuEngine) -> Result<(), CheckerError> {
        let owner = self.actual_owner_id.ok_or(CheckerError::Unsupported(
            "declaration metadata has no actual PausedStack owner binding",
        ))?;
        let snapshot = engine
            .paused_stack_snapshot()
            .ok_or(CheckerError::Unsupported(
                "actual CPU owner readback unavailable",
            ))?;
        if !snapshot.complete || snapshot.owner_id != owner {
            return Err(CheckerError::Invalid(
                "metadata reset actual CPU owner differs",
            ));
        }
        let actual = Self::capture_owned(engine)?;
        self.same_owned_namespace(&actual)
    }

    fn same_owned_namespace(&self, actual: &Self) -> Result<(), CheckerError> {
        if self.identity != actual.identity
            || self.conditions != actual.conditions
            || self.capabilities != actual.capabilities
            || self.resume_policy != actual.resume_policy
            || !same_descriptor(
                self.owned_descriptor.as_ref(),
                actual.owned_descriptor.as_ref(),
            )
        {
            return Err(CheckerError::Invalid(
                "metadata reset does not belong to actual CPU owner",
            ));
        }
        Ok(())
    }
}

fn checker_capabilities(native: rz_search::cpu::CpuCapabilities) -> CheckerCapabilities {
    CheckerCapabilities {
        max_depth: native.max_depth,
        max_prefix_plies: native.max_prefix_plies,
        max_root_moves: native.max_root_moves,
        root_moves: native.root_moves,
        divergence: native.divergence,
        resume: native.completed_iteration_resume,
        selective_search: Some(native.selective_reductions),
    }
}

fn same_descriptor(
    left: Option<&OwnedCheckerDescriptor>,
    right: Option<&OwnedCheckerDescriptor>,
) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => {
            left.config.profile == right.config.profile
                && left.config.tt_entries == right.config.tt_entries
                && left.config.max_depth == right.config.max_depth
                && left.config.quiescence_ply == right.config.quiescence_ply
                && left.search_identity == right.search_identity
                && left.search_conditions == right.search_conditions
                && left.capabilities == right.capabilities
        }
        (None, None) => true,
        _ => false,
    }
}

fn check_controls(deadline: Instant, cancel: &AtomicBool) -> Result<(), CheckerError> {
    if cancel.load(Ordering::Acquire) || Instant::now() >= deadline {
        return Err(CheckerError::Invalid(
            "metadata lifecycle canceled or deadline expired",
        ));
    }
    Ok(())
}

fn live_external_process(
    process: CheckerShutdown,
) -> Result<ExternalProcessIdentity, CheckerError> {
    let identity = process.process_identity.ok_or(CheckerError::Invalid(
        "actual external startup process identity missing",
    ))?;
    if identity.pid == 0
        || identity.process_group == 0
        || identity.proc_start_ticks == 0
        || process.quit_sent
        || process.exit_observed
        || process.cleanup_complete
        || process.quarantined
        || process.ownership_lost
    {
        return Err(CheckerError::Invalid(
            "actual external startup process is not reusable",
        ));
    }
    Ok(identity)
}

/// The PalsEngine's declaration mirror. It never returns a CPU report, native
/// paused-token proof, UCI handshake, or physical shutdown receipt.
pub struct AdaptiveCheckerMetadataMirror {
    metadata: AdaptiveCheckerMetadata,
}

impl CpuChecker for AdaptiveCheckerMetadataMirror {
    fn identity(&self) -> &CheckerIdentity {
        self.metadata.identity()
    }
    fn conditions(&self) -> &str {
        self.metadata.conditions()
    }
    fn capabilities(&self) -> CheckerCapabilities {
        self.metadata.capabilities()
    }
    fn owned_descriptor(&self) -> Option<OwnedCheckerDescriptor> {
        self.metadata.owned_descriptor()
    }
    fn owned_resume_policy(&self) -> Option<CpuResumePolicy> {
        self.metadata.resume_policy()
    }
    fn registered_worker_stack_owner_id(&self) -> Option<u64> {
        self.metadata.actual_owner_id
    }
    fn start(&mut self, _: Instant, _: &AtomicBool) -> Result<(), CheckerError> {
        Err(CheckerError::Unsupported(METADATA_ONLY))
    }
    fn preflight_task(&self, _: CpuLimits) -> Result<(), CheckerError> {
        Err(CheckerError::Unsupported(METADATA_ONLY))
    }
    fn last_attempt(&self) -> Option<&CheckerAttempt> {
        None
    }
    fn reset_attempt(&mut self) { /* No physical attempt ledger is owned here. */
    }
    fn analyze(
        &mut self,
        _: &Position,
        _: CpuLimits,
        _: &AtomicBool,
    ) -> Result<CheckerReport, CheckerError> {
        Err(CheckerError::Unsupported(METADATA_ONLY))
    }
    fn analyze_root_moves(
        &mut self,
        _: &Position,
        _: &[BoardMove],
        _: CpuLimits,
        _: &AtomicBool,
    ) -> Result<CheckerReport, CheckerError> {
        Err(CheckerError::Unsupported(METADATA_ONLY))
    }
    fn analyze_divergence(
        &mut self,
        _: &Position,
        _: &[BoardMove],
        _: CpuLimits,
        _: &AtomicBool,
    ) -> Result<CheckerReport, CheckerError> {
        Err(CheckerError::Unsupported(METADATA_ONLY))
    }
    fn resume(
        &mut self,
        _: &Position,
        _: &CpuResumeToken,
        _: CpuLimits,
        _: &AtomicBool,
    ) -> Result<CheckerReport, CheckerError> {
        Err(CheckerError::Unsupported(METADATA_ONLY))
    }
    fn new_game(&mut self, deadline: Instant, cancel: &AtomicBool) -> Result<(), CheckerError> {
        let mut reset = self
            .metadata
            .reset
            .lock()
            .map_err(|_| CheckerError::Invalid("metadata reset owner poisoned"))?;
        // Consume even on canceled/expired delivery, preventing a late retry
        // from treating an old reset as permission for another game.
        let arm = reset.pending.take().ok_or(CheckerError::Unsupported(
            "actual checker new-game reset was not armed",
        ))?;
        let effective_deadline = match arm {
            MetadataResetArm::Owned { deadline: owner } => deadline.min(owner),
            MetadataResetArm::InitialExternal { deadline: owner } => deadline.min(owner),
            MetadataResetArm::External { .. } => deadline,
            #[cfg(feature = "pals-adaptive")]
            MetadataResetArm::ProfiledExternal { deadline: owner } => deadline.min(owner),
        };
        check_controls(effective_deadline, cancel)
    }
    fn shutdown(&mut self, _: Instant) -> Result<CheckerShutdown, CheckerError> {
        Err(CheckerError::Unsupported(METADATA_ONLY))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_search::cpu::{CpuCapabilities, CpuConfig, CpuProfile};
    use rz_search::cpu_checker::{
        ExternalCheckerIdentity, ExternalModelMetadata, ExternalTrainingKnowledge,
    };
    use rz_search::cpu_value::{CpuTrainingState, CpuValueIdentity};
    use std::collections::BTreeMap;
    use std::time::Duration;

    fn declared_owned() -> (
        CheckerIdentity,
        String,
        CheckerCapabilities,
        OwnedCheckerDescriptor,
    ) {
        let conditions = "declared fixture conditions; no CPU engine constructed".to_owned();
        let descriptor = OwnedCheckerDescriptor {
            config: CpuConfig {
                profile: CpuProfile::Independent,
                tt_entries: 1_048_576,
                max_depth: 8,
                quiescence_ply: 8,
            },
            search_identity: "metadata-fixture-search-v1",
            search_conditions: conditions.clone(),
            capabilities: CpuCapabilities {
                max_depth: 8,
                max_prefix_plies: 64,
                max_root_moves: 256,
                root_moves: true,
                divergence: true,
                completed_iteration_resume: true,
                selective_reductions: false,
            },
        };
        let identity = CheckerIdentity::Owned(CpuValueIdentity {
            semantics: "metadata-bootstrap-fixture".into(),
            weights_sha256: None,
            training: CpuTrainingState::Bootstrap,
        });
        (
            identity,
            conditions,
            checker_capabilities(descriptor.capabilities),
            descriptor,
        )
    }
    fn metadata() -> AdaptiveCheckerMetadata {
        let (identity, conditions, caps, descriptor) = declared_owned();
        AdaptiveCheckerMetadata::new(
            identity,
            conditions,
            caps,
            Some(descriptor),
            Some(CpuResumePolicy::PausedStack),
        )
        .unwrap()
    }

    #[test]
    fn declaration_mirror_preserves_exact_namespace_without_allocating_cpu() {
        let metadata = metadata();
        let expected = metadata.clone();
        let checker = metadata.into_mirror();
        checker.validate_namespace().unwrap();
        assert_eq!(checker.identity(), expected.identity());
        assert_eq!(checker.conditions(), expected.conditions());
        assert_eq!(checker.capabilities(), expected.capabilities());
        assert_eq!(checker.owned_resume_policy(), expected.resume_policy());
        assert!(same_descriptor(
            checker.owned_descriptor().as_ref(),
            expected.owned_descriptor().as_ref()
        ));
        assert!(checker.startup_uci().is_none());
        assert!(checker.last_attempt().is_none());
        assert!(checker.paused_stack_snapshot().is_none());
        assert!(!checker.supports_paused_stack_discard());
        // A descriptor constructor can preserve namespace declarations, but
        // cannot mint the concrete engine owner's reset authority.
        assert!(expected.actual_owner_id.is_none());
    }

    #[test]
    fn metadata_only_lifecycle_does_not_fabricate_owner_evidence() {
        let mut checker = metadata().into_mirror();
        let deadline = Instant::now() + Duration::from_secs(1);
        let cancel = AtomicBool::new(false);
        assert!(matches!(
            checker.start(deadline, &cancel),
            Err(CheckerError::Unsupported(_))
        ));
        assert!(matches!(
            checker.new_game(deadline, &cancel),
            Err(CheckerError::Unsupported(_))
        ));
        assert!(matches!(
            checker.shutdown(deadline),
            Err(CheckerError::Unsupported(_))
        ));
        let position = Position::startpos();
        let limits = CpuLimits {
            max_depth: 1,
            max_nodes: 1,
            deadline: Some(deadline),
        };
        assert!(matches!(
            checker.preflight_task(limits),
            Err(CheckerError::Unsupported(_))
        ));
        assert!(matches!(
            checker.analyze(&position, limits, &cancel),
            Err(CheckerError::Unsupported(_))
        ));
        assert!(matches!(
            checker.analyze_root_moves(&position, &[], limits, &cancel),
            Err(CheckerError::Unsupported(_))
        ));
        assert!(matches!(
            checker.analyze_divergence(&position, &[], limits, &cancel),
            Err(CheckerError::Unsupported(_))
        ));
        checker.reset_attempt();
        assert!(checker.last_attempt().is_none());
    }

    #[test]
    fn external_declaration_preserves_unknowns_and_rejects_own_semantics() {
        let identity = CheckerIdentity::ExternalUci(ExternalCheckerIdentity {
            adapter_semantics: "metadata-external-uci-fixture".into(),
            binary_sha256: "a".repeat(64),
            launch_arguments_sha256: "b".repeat(64),
            declared_name: "fixture".into(),
            declared_version: "1".into(),
            declared_source: "test declaration only".into(),
            declared_license: "MIT".into(),
            options: BTreeMap::new(),
            assets: Vec::new(),
            model_metadata: ExternalModelMetadata {
                weights_sha256: None,
                training: ExternalTrainingKnowledge::Unknown,
                declared_rights: None,
                precision: None,
            },
        });
        let caps = CheckerCapabilities {
            max_depth: 8,
            max_prefix_plies: 256,
            max_root_moves: 256,
            root_moves: true,
            divergence: true,
            resume: false,
            selective_search: None,
        };
        let metadata = AdaptiveCheckerMetadata::new(
            identity.clone(),
            "external fixture".into(),
            caps,
            None,
            None,
        )
        .unwrap();
        let checker = metadata.into_mirror();
        assert_eq!(checker.identity(), &identity);
        assert!(checker.startup_uci().is_none());
        assert!(checker.owned_descriptor().is_none());
        assert!(checker.owned_resume_policy().is_none());
        assert!(
            AdaptiveCheckerMetadata::new(
                identity.clone(),
                "external fixture".into(),
                CheckerCapabilities {
                    resume: true,
                    ..caps
                },
                None,
                None
            )
            .is_err()
        );
        let (_, _, _, descriptor) = declared_owned();
        assert!(
            AdaptiveCheckerMetadata::new(
                identity,
                "external fixture".into(),
                caps,
                Some(descriptor),
                Some(CpuResumePolicy::PausedStack)
            )
            .is_err()
        );
    }

    #[test]
    fn descriptor_capability_or_conditions_drift_is_rejected() {
        let (identity, conditions, mut caps, descriptor) = declared_owned();
        caps.resume = false;
        assert!(
            AdaptiveCheckerMetadata::new(
                identity.clone(),
                conditions.clone(),
                caps,
                Some(descriptor.clone()),
                Some(CpuResumePolicy::PausedStack)
            )
            .is_err()
        );
        let caps = checker_capabilities(descriptor.capabilities);
        assert!(
            AdaptiveCheckerMetadata::new(
                identity.clone(),
                "different".into(),
                caps,
                Some(descriptor.clone()),
                Some(CpuResumePolicy::PausedStack)
            )
            .is_err()
        );
        assert!(
            AdaptiveCheckerMetadata::new(identity, conditions, caps, Some(descriptor), None)
                .is_err()
        );
    }

    #[test]
    fn oversized_retained_conditions_are_rejected_before_mirror_creation() {
        let (identity, _, caps, mut descriptor) = declared_owned();
        let mut conditions = String::with_capacity(MAX_CONDITIONS_BYTES + 1);
        conditions.push_str("small content, excessive retained allocation");
        descriptor.search_conditions = conditions.clone();
        assert!(
            AdaptiveCheckerMetadata::new(
                identity,
                conditions,
                caps,
                Some(descriptor),
                Some(CpuResumePolicy::PausedStack)
            )
            .is_err()
        );
    }

    #[test]
    fn startup_process_validation_requires_full_live_identity() {
        // This checks a private admission predicate only. It never binds a
        // worker or creates a mirror lifecycle arm from synthetic observations.
        let observed = CheckerShutdown {
            process_identity: Some(ExternalProcessIdentity {
                pid: 17,
                process_group: 17,
                proc_start_ticks: 41,
            }),
            ..CheckerShutdown::default()
        };
        assert_eq!(live_external_process(observed).unwrap().pid, 17);
        for rejected in [
            CheckerShutdown::default(),
            CheckerShutdown {
                process_identity: Some(ExternalProcessIdentity {
                    pid: 0,
                    ..observed.process_identity.unwrap()
                }),
                ..observed
            },
            CheckerShutdown {
                process_identity: Some(ExternalProcessIdentity {
                    process_group: 0,
                    ..observed.process_identity.unwrap()
                }),
                ..observed
            },
            CheckerShutdown {
                process_identity: Some(ExternalProcessIdentity {
                    proc_start_ticks: 0,
                    ..observed.process_identity.unwrap()
                }),
                ..observed
            },
            CheckerShutdown {
                quit_sent: true,
                ..observed
            },
            CheckerShutdown {
                exit_observed: true,
                ..observed
            },
            CheckerShutdown {
                cleanup_complete: true,
                ..observed
            },
            CheckerShutdown {
                quarantined: true,
                ..observed
            },
            CheckerShutdown {
                ownership_lost: true,
                ..observed
            },
        ] {
            assert!(matches!(
                live_external_process(rejected),
                Err(CheckerError::Invalid(_))
            ));
        }
    }
}
