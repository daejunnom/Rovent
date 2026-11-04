//! Opt-in, isolated test-ELF teacher worker; never an official teacher source.
//!
//! The wrapper keeps native dispatch and completion capabilities private.
//! Its observations cannot enter the collection producer or training path.
use super::*;
use serde::{Deserialize, Serialize};

pub const FIXTURE_SOURCE_DOMAIN: &str = "rz-rovent-test-teacher-identity/1";
const FIXTURE_UCI_NAME: &str = "RZ external checker fixture";
const MAX_RETAINED: usize = 64;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureWorkerSource {
    pub schema: String,
    pub executable_sha256: String,
    pub executable_bytes: u64,
    pub fixture_scenario: String,
    pub scope: String,
    pub official_engine_identity: bool,
    pub training_eligible: bool,
}

impl FixtureWorkerSource {
    fn validate(
        &self,
        config: &ExternalCpuConfig,
        pin: &ExternalLaunchPin,
    ) -> Result<(), CheckerError> {
        if self.schema != FIXTURE_SOURCE_DOMAIN
            || self.scope != "test_elf_only"
            || self.official_engine_identity
            || self.training_eligible
            || self.fixture_scenario.is_empty()
            || self.fixture_scenario.len() > 128
            || self.fixture_scenario.chars().any(char::is_control)
            || self.executable_bytes == 0
            || self.executable_bytes > 256 * 1024 * 1024
            || self.executable_sha256.len() != 64
            || !self
                .executable_sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || self.executable_sha256 != config.identity.binary_sha256
            || config.identity.declared_source != FIXTURE_SOURCE_DOMAIN
            || config.identity.declared_name != FIXTURE_UCI_NAME
            || config.expected_uci_name.as_deref() != Some(FIXTURE_UCI_NAME)
        {
            return Err(CheckerError::Invalid(
                "test teacher fixture identity mismatch",
            ));
        }
        let file = pin
            .executable
            .metadata()
            .map_err(|_| CheckerError::Invalid("test fixture executable metadata"))?;
        if !file.is_file() || file.len() != self.executable_bytes {
            return Err(CheckerError::Invalid(
                "test fixture executable size differs",
            ));
        }
        Ok(())
    }
}

fn create_fixture_checker(
    config: ExternalCpuConfig,
    internal_profile: ExternalEngineProfile,
) -> Result<ExternalUciCpuChecker, CheckerError> {
    let mut checker = ExternalUciCpuChecker::create(config)?;
    // The private checker needs its existing policy parser; this enum is an
    // implementation parameter only and is never exported as an identity.
    checker.engine_profile = Some((internal_profile, ExternalEnginePurpose::Teacher));
    Ok(checker)
}

/// No raw dispatch, completion, original worker or official pin escapes this
/// wrapper. Bounded originals remain in its custody until the caller joins it.
pub struct FixtureTeacherWorker {
    worker: CollectionTeacherWorker,
    pending: Option<TeacherDispatch>,
    retained: Vec<(TeacherDispatch, TeacherCompletion)>,
    clean_join_observed: bool,
    fixture_owner: Arc<()>,
}

#[derive(Debug)]
pub struct FixtureTeacherObservation {
    request_sequence: u64,
    conditional_policy_target_known: Option<bool>,
    scope: TeacherInputScope,
    original_position: rz_position::PositionSnapshot,
    restricted: bool,
    fixture_owner: Arc<()>,
}

impl FixtureTeacherObservation {
    pub fn request_sequence(&self) -> u64 {
        self.request_sequence
    }
    pub fn conditional_policy_target_known(&self) -> Option<bool> {
        self.conditional_policy_target_known
    }
    pub fn input_scope(&self) -> TeacherInputScope {
        self.scope
    }
    pub fn has_root_restriction(&self) -> bool {
        self.restricted
    }
    pub fn matches_position(&self, position: &rz_position::PositionSnapshot) -> bool {
        self.original_position.same_state(position)
    }
    pub fn same_fixture_worker(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.fixture_owner, &other.fixture_owner)
    }
    /// Fixture workers do not own the original manifest-registered teacher.
    /// A matching caller-provided digest alone cannot establish that fact.
    pub fn original_registered_actor_proven(&self) -> bool {
        false
    }
    pub fn training_eligible(&self) -> bool {
        false
    }
    pub fn official_engine_identity(&self) -> bool {
        false
    }
    pub fn physical_completion(&self) -> &'static str {
        "not_attested"
    }
}

impl FixtureTeacherWorker {
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        source: &FixtureWorkerSource,
        config: ExternalCpuConfig,
        registration: TeacherWorkerRegistration,
        clock: ExternalWorkerClock,
        startup_deadline: Instant,
        cleanup_ceiling: Instant,
        pin: ExternalLaunchPin,
    ) -> Result<Self, CheckerError> {
        source.validate(&config, &pin)?;
        let worker = CollectionTeacherWorker::spawn_checked(
            config,
            ExternalEngineProfile::Reckless090EmbeddedNnue,
            registration,
            clock,
            startup_deadline,
            cleanup_ceiling,
            pin,
            None,
            create_fixture_checker,
        )?;
        Ok(Self {
            worker,
            pending: None,
            retained: Vec::new(),
            clean_join_observed: false,
            fixture_owner: Arc::new(()),
        })
    }

    /// Startup observation remains a fixture result, never an official UCI pin.
    pub fn startup_ready(&self) -> Option<Result<(), CheckerError>> {
        self.worker
            .startup_report()
            .map(|startup| startup.result.map(|_| ()))
    }

    pub fn submit(&mut self, job: TeacherJob) -> Result<(), CheckerError> {
        if self.pending.is_some() || self.retained.len() >= MAX_RETAINED {
            return Err(CheckerError::Invalid(
                "test teacher pending/custody capacity",
            ));
        }
        self.retained
            .try_reserve(1)
            .map_err(|_| CheckerError::Invalid("test teacher custody allocation"))?;
        self.pending = Some(self.worker.submit(job)?);
        Ok(())
    }

    pub fn poll_observation(&mut self) -> Result<Option<FixtureTeacherObservation>, CheckerError> {
        if self.pending.is_none() {
            return Ok(None);
        }
        let Some(completion) = self.worker.take_completion() else {
            return Ok(None);
        };
        let dispatch = self.pending.take().expect("pending dispatch checked");
        let same = completion.matches_submission(&dispatch);
        let observation = FixtureTeacherObservation {
            request_sequence: dispatch.id().sequence,
            conditional_policy_target_known: completion
                .result()
                .as_ref()
                .ok()
                .map(|result| result.conditional_policy_target_known),
            scope: dispatch.input_scope(),
            original_position: dispatch.original_position().clone(),
            restricted: dispatch.original_root_moves().is_some(),
            fixture_owner: Arc::clone(&self.fixture_owner),
        };
        self.retained.push((dispatch, completion));
        if !same {
            return Err(CheckerError::Invalid(
                "test teacher completion identity differs",
            ));
        }
        Ok(Some(observation))
    }

    pub fn request_shutdown(&self, deadline: Instant) {
        self.worker.request_shutdown(deadline);
    }

    /// Compare this exact fixture worker's clean process join with the
    /// observation owner. This does not attest any independent native fence.
    pub fn clean_join_matches(&self, observation: &FixtureTeacherObservation) -> bool {
        self.clean_join_observed
            && self.pending.is_none()
            && Arc::ptr_eq(&self.fixture_owner, &observation.fixture_owner)
    }

    /// Join is a worker-thread observation only; do not infer a GPU fence.
    pub fn poll_join(&mut self) -> Option<Result<(), CheckerError>> {
        self.worker.try_join().map(|result| {
            result.and_then(|shutdown| {
                if !shutdown.cleanup_complete || shutdown.quarantined || shutdown.ownership_lost {
                    return Err(CheckerError::Invalid(
                        "test teacher process cleanup unknown",
                    ));
                }
                self.clean_join_observed = true;
                Ok(())
            })
        })
    }
}

impl Drop for FixtureTeacherWorker {
    fn drop(&mut self) {
        if !self.clean_join_observed {
            if let Some(pending) = self.pending.take() {
                std::mem::forget(pending);
            }
            std::mem::forget(std::mem::take(&mut self.retained));
        }
        // The underlying worker requests its bounded shutdown in its Drop.
    }
}
