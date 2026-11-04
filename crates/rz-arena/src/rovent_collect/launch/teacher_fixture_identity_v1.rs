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
