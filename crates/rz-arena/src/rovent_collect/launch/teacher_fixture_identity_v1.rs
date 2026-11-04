//! Test-only teacher identity declaration, separated from official profiles.
//!
//! A declaration is not a process owner, a source pin for Stockfish/Reckless,
//! a root admission or a training artifact. No adapter from this type to the
//! real teacher preload/actor exists; those paths keep their original checks.
use serde::{Deserialize, Serialize};

pub const TEACHER_FIXTURE_IDENTITY_DOMAIN: &str = "rz-rovent-test-teacher-identity/1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureEvidenceScope {
    TestElfOnly,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherFixtureIdentityV1 {
    pub schema: String,
    pub executable_sha256: String,
    pub executable_bytes: u64,
    pub fixture_scenario: String,
    pub scope: FixtureEvidenceScope,
    pub official_engine_identity: bool,
    pub training_eligible: bool,
}

impl TeacherFixtureIdentityV1 {
    /// Only the feature-gated fixture worker accepts this translation. It is
    /// deliberately not an adapter to product preload, actor, or teacher pins.
    #[cfg(all(target_os = "linux", feature = "test-fixtures"))]
    pub fn worker_source(
        &self,
    ) -> Result<
        rz_search::external_cpu::collection_teacher::fixture_worker::FixtureWorkerSource,
        &'static str,
    > {
        self.validate()?;
        Ok(
            rz_search::external_cpu::collection_teacher::fixture_worker::FixtureWorkerSource {
                schema: self.schema.clone(),
                executable_sha256: self.executable_sha256.clone(),
                executable_bytes: self.executable_bytes,
                fixture_scenario: self.fixture_scenario.clone(),
                scope: "test_elf_only".into(),
                official_engine_identity: false,
                training_eligible: false,
            },
        )
    }

    /// Checks metadata only; actual opened descriptor, byte hash and custody
    /// remain the responsibility of a future feature-gated fixture adapter.
    pub fn from_json(input: &str) -> Result<Self, &'static str> {
        if input.len() > 16 * 1024 {
            return Err("fixture identity metadata exceeds bound");
        }
        let identity: Self =
            rz_experiments::decode_json(input).map_err(|_| "fixture identity JSON is invalid")?;
        identity.validate()?;
        Ok(identity)
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.schema != TEACHER_FIXTURE_IDENTITY_DOMAIN
            || self.executable_sha256.len() != 64
            || !self
                .executable_sha256
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
            || self.executable_bytes == 0
            || self.executable_bytes > 256 * 1024 * 1024
            || self.fixture_scenario.is_empty()
            || self.fixture_scenario.len() > 128
            || self.fixture_scenario.chars().any(char::is_control)
            || self.scope != FixtureEvidenceScope::TestElfOnly
            || self.official_engine_identity
            || self.training_eligible
        {
            return Err("fixture identity domain, bounds or scope invalid");
        }
        Ok(())
    }
}

/// Original producer rows and fixture observations stay in this test-only
/// custody. No API converts them into teacher labels or strict publications.
#[cfg(all(
    target_os = "linux",
    feature = "test-fixtures",
    feature = "pals-collection-onnx"
))]
pub struct FixtureProducerAdapter {
    rows: Vec<(
        super::super::RoventRetiredLockV2Record,
        rz_search::external_cpu::collection_teacher::fixture_worker::FixtureTeacherObservation,
    )>,
}

#[cfg(all(
    target_os = "linux",
    feature = "test-fixtures",
    feature = "pals-collection-onnx"
))]
pub struct FixtureAttachFailure {
    pub error: String,
    pub row: super::super::RoventRetiredLockV2Record,
    pub observation:
        rz_search::external_cpu::collection_teacher::fixture_worker::FixtureTeacherObservation,
    pub panic: Option<Box<dyn std::any::Any + Send>>,
}

#[cfg(all(
    target_os = "linux",
    feature = "test-fixtures",
    feature = "pals-collection-onnx"
))]
impl FixtureProducerAdapter {
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }

    /// The original producer performs the owner check; Rules replay and all
    /// shared root/input fields must match this worker-issued one-shot view.
    /// Errors and panics return both original objects without altering custody.
    pub fn attach(
        &mut self,
        producer: &super::super::RoventCorpusProducerLockV2,
        row: super::super::RoventRetiredLockV2Record,
        observation: rz_search::external_cpu::collection_teacher::fixture_worker::FixtureTeacherObservation,
    ) -> Result<(), Box<FixtureAttachFailure>> {
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
                if self.rows.len() >= 2 {
                    return Err("fixture row capacity".into());
                }
                producer
                    .validate_teacher_row(&row)
                    .map_err(|e| e.to_string())?;
                let input = &row.record.0.input;
                let scope = observation.input_scope();
                let root = row.admission.root_admission();
                let parse = |s: &str| {
                    rz_eval::asset::parse_sha256(s)
                        .map(rz_contracts::Digest)
                        .map_err(|e| e.to_string())
                };
                let seal = row.submitted_input_seal();
                let actor = seal
                    .teacher_actor_id
                    .as_deref()
                    .ok_or_else(|| "original teacher actor missing".to_owned())?;
                if row.record.0.lineage.teacher_actor_id.as_deref() != Some(actor) {
                    return Err("original teacher actor differs from lineage".into());
                }
                let pin = row
                    .manifest_owner
                    .registration
                    .teachers
                    .get(actor)
                    .ok_or_else(|| "original registered teacher pin missing".to_owned())?;
                // Same domain and payload as the original frozen actor constructor;
                // the digest does not itself prove the fixture worker owns it.
                let actor_digest = rz_experiments::pals_data_v3::generation_digest(
                    "rz-rovent-frozen-teacher-actor/1",
                    &serde_json::json!({"actor_id":actor,"teacher_pin":pin}),
                )
                .map_err(|e| e.to_string())?;
                if scope.frozen_actor != parse(&actor_digest)? {
                    return Err("original frozen teacher actor digest differs".into());
                }
                if !observation.original_registered_actor_proven() {
                    return Err(
                        "fixture worker does not own the original registered teacher actor".into(),
                    );
                }
                let replay = row
                    .original_teacher_query_position()
                    .map_err(|e| e.to_string())?;
                let root_digest = root.descriptor().digest().map_err(|e| e.to_string())?;
                if observation.conditional_policy_target_known().is_none()
                    || observation.training_eligible()
                    || observation.official_engine_identity()
                    || observation.has_root_restriction()
                    || scope.authority != row.ticket.authority
                    || scope.role != row.ticket.role
                    || scope.input_sha256 != parse(&row.record.0.input_sha256)?
                    || scope.view_sha256 != parse(&input.view_sha256)?
                    || scope.source_sha256 != parse(&row.manifest_owner.registration.source_sha256)?
                    || scope.rules_semantic != parse(&input.rules_anchor.rules_state_sha256)?
                    || scope.input_lock != root.startup_lock_digest()
                    || scope.root_admission != root_digest
                    || scope.root_budget != row.admission.root_budget()
                    || !observation.matches_position(&replay.snapshot())
                {
                    return Err("fixture observation differs from original root/input/Rules".into());
                }
                if let Some((first_row, first_observation)) = self.rows.first() {
                    if !first_row.admission.same_original_admission(&row.admission)
                        || !first_observation.same_fixture_worker(&observation)
                        || first_observation.request_sequence() == observation.request_sequence()
                        || first_row.record.0.native.retired_sequence
                            == row.record.0.native.retired_sequence
                    {
                        return Err(
                            "fixture observations differ in original root/worker or duplicate row"
                                .into(),
                        );
                    }
                }
                self.rows
                    .try_reserve(1)
                    .map_err(|_| "fixture row custody allocation".to_owned())?;
                Ok(())
            }));
        match result {
            Ok(Ok(())) => {
                self.rows.push((row, observation));
                Ok(())
            }
            Ok(Err(error)) => Err(Box::new(FixtureAttachFailure {
                error,
                row,
                observation,
                panic: None,
            })),
            Err(panic) => Err(Box::new(FixtureAttachFailure {
                error: "fixture row check panicked".into(),
                row,
                observation,
                panic: Some(panic),
            })),
        }
    }

    /// Test-only observation, not a producer seal or completed native fence.
    pub fn mixed_exact_and_unknown(&self) -> bool {
        self.rows.len() == 2
            && self
                .rows
                .iter()
                .any(|(_, o)| o.conditional_policy_target_known() == Some(true))
            && self
                .rows
                .iter()
                .any(|(_, o)| o.conditional_policy_target_known() == Some(false))
    }

    /// Release only after the same real collection run confirms its original
    /// fenced/settled owner and the same fixture worker confirms clean join.
    /// Every check happens before mutating custody; failure retains all rows.
    pub fn release_after_observed_fences(
        &mut self,
        run: &super::RoventPcCollectionRunV2,
        worker: &rz_search::external_cpu::collection_teacher::fixture_worker::FixtureTeacherWorker,
    ) -> Result<(), String> {
        let owner = run.closed_owner().map_err(|e| e.to_string())?;
        for (row, observation) in &self.rows {
            if !worker.clean_join_matches(observation) {
                return Err(
                    "fixture worker clean join is absent or belongs to another owner".into(),
                );
            }
            owner
                .with_retired_producer(row, |producer| producer.validate_teacher_row(row))
                .map_err(|e| e.to_string())?;
        }
        self.rows.clear();
        Ok(())
    }
}

#[cfg(all(
    target_os = "linux",
    feature = "test-fixtures",
    feature = "pals-collection-onnx"
))]
impl Drop for FixtureProducerAdapter {
    fn drop(&mut self) {
        // The test adapter cannot prove device/owner retirement. Preserve its
        // exact originals if its caller has not established another owner.
        std::mem::forget(std::mem::take(&mut self.rows));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_experiments::rovent_corpus_lock_v2::publication::{
        ROVENT_LOCK_V2_PUBLICATION_DOMAIN, RoventLockV2InputLockBytesOrigin,
        RoventLockV2Publication, RoventLockV2PublicationAdmission, RoventLockV2PublicationFile,
    };
    use sha2::{Digest as _, Sha256};

    fn fixture() -> TeacherFixtureIdentityV1 {
        TeacherFixtureIdentityV1 {
            schema: TEACHER_FIXTURE_IDENTITY_DOMAIN.into(),
            executable_sha256: "a".repeat(64),
            executable_bytes: 128,
            fixture_scenario: "test-elf-scenario".into(),
            scope: FixtureEvidenceScope::TestElfOnly,
            official_engine_identity: false,
            training_eligible: false,
        }
    }

    #[test]
    fn fixture_declaration_cannot_be_parsed_as_official_profile() {
        let source = serde_json::to_vec(&fixture()).unwrap();
        let digest = format!("{:x}", Sha256::digest(&source));
        assert!(TeacherFixtureIdentityV1::from_json(std::str::from_utf8(&source).unwrap()).is_ok());
        assert!(
            rz_uci::pals_checker_profile::rovent::RoventExternalProfile::from_verified_bytes(
                &source, &digest
            )
            .is_err()
        );
        let mut forged = fixture();
        forged.official_engine_identity = true;
        assert!(forged.validate().is_err());
        forged.official_engine_identity = false;
        forged.training_eligible = true;
        assert!(forged.validate().is_err());
    }

    #[test]
    fn strict_publication_reader_rejects_fixture_domain() {
        let file = |path: &str, byte: char, bytes: u64| RoventLockV2PublicationFile {
            path: path.into(),
            sha256: byte.to_string().repeat(64),
            bytes,
        };
        let strict = RoventLockV2Publication {
            schema: ROVENT_LOCK_V2_PUBLICATION_DOMAIN.into(),
            manifest: file("corpus-manifest.json", 'a', 10),
            index: file("corpus-index.json", 'b', 20),
            chunks: vec![file("chunks/0.jsonl", 'c', 30)],
            canonical_manifest_sha256: "a".repeat(64),
            canonical_index_sha256: "b".repeat(64),
            source_commit: "f".repeat(40),
            source_sha256: "e".repeat(64),
            runtime_sha256: "d".repeat(64),
            record_count: 2,
            total_payload_bytes: 150,
            native_admission_sha256: vec!["a".repeat(64), "b".repeat(64)],
            admissions: ["a", "b"]
                .into_iter()
                .map(|id| RoventLockV2PublicationAdmission {
                    native_admission_sha256: id.repeat(64),
                    input_lock: file("input-locks/lock.json", 'd', 40),
                    input_lock_bytes_origin:
                        RoventLockV2InputLockBytesOrigin::ValidatedSemanticSnapshot,
                    canonical_input_lock_sha256: "e".repeat(64),
                    runtime_bundle_manifest: Some(file("runtime-bundles/runtime.json", 'f', 50)),
                })
                .collect(),
        };
        // The control proves that the same complete DTO is accepted before
        // changing only the domain; neither case proves real native custody.
        assert!(
            RoventLockV2Publication::from_json(&serde_json::to_string(&strict).unwrap()).is_ok()
        );
        let mut different_domain = strict;
        different_domain.schema = TEACHER_FIXTURE_IDENTITY_DOMAIN.into();
        assert!(
            RoventLockV2Publication::from_json(&serde_json::to_string(&different_domain).unwrap())
                .is_err()
        );
    }
}
