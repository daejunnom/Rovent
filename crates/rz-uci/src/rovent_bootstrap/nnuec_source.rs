//! Actual NNUEC V2 source ownership, separate from native P/C readiness.
//!
//! Only the original retained-source owner supplies bytes. A validated training
//! declaration is not proof of training, and this loader grants no arena or
//! numerical acceptance. Raw artifact hashes and semantic parameter hashes are
//! deliberately different identities and are checked independently.

use super::pinned::NativeInputLock;
use super::{RoventPinnedNativeSources, RoventStartupControl, fail, hex};
use rz_contracts::Digest;
use rz_eval::asset::parse_sha256;
use rz_eval::error::{BackendError, CauseCode, FailureKind as K};
use rz_experiments::{ArtifactRef, RoventInputLock, RoventNnuecPin, decode_json};
use rz_search::cpu_value::nnuec::v2::Model;
use rz_search::cpu_value::nnuec::v2::registration::NnuecV2Registration;
use rz_search::cpu_value::{CpuTrainingState, CpuValueIdentity};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::{collections::BTreeSet, sync::Arc};

pub const NNUEC_PROVENANCE_DOMAIN: &str = "rz-rovent-nnuec-provenance/1";
pub const NNUEC_WEIGHTS_MAX_BYTES: usize = 32 << 20;
pub const NNUEC_PROVENANCE_MAX_BYTES: usize = 256 << 10;
const MAX_PROVENANCE_ARTIFACTS: usize = 32;

/// Construction is closed over the original retained bytes, actual loaded
/// model, compiled registration and exact startup lock. Keeping the source Arc
/// also keeps its original directory and file descriptors alive.
/// Clone은 같은 source/model Arc를 공유하며 재로딩이나 새 권한을 발급하지 않는다.
pub struct LoadedRoventNnuecSource<L = RoventInputLock> {
    sources: Arc<RoventPinnedNativeSources<L>>,
    model: Arc<Model>,
    registration: NnuecV2Registration,
    weights_artifact: ArtifactRef,
    provenance_artifact: ArtifactRef,
}

impl<L> Clone for LoadedRoventNnuecSource<L> {
    fn clone(&self) -> Self {
        Self {
            sources: Arc::clone(&self.sources),
            model: Arc::clone(&self.model),
            registration: self.registration.clone(),
            weights_artifact: self.weights_artifact.clone(),
            provenance_artifact: self.provenance_artifact.clone(),
        }
    }
}

impl<L: NativeInputLock> LoadedRoventNnuecSource<L> {
    pub fn matches_sources(&self, sources: &Arc<RoventPinnedNativeSources<L>>) -> bool {
        Arc::ptr_eq(&self.sources, sources)
    }
    pub fn load(
        sources: Arc<RoventPinnedNativeSources<L>>,
        control: &RoventStartupControl,
    ) -> Result<Self, BackendError> {
        let verified = verify_source_metadata(&sources, control)?;
        Ok(Self {
            sources,
            model: verified.model,
            registration: verified.registration,
            weights_artifact: verified.weights_artifact,
            provenance_artifact: verified.provenance_artifact,
        })
    }

    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }
    /// Equality of public metadata cannot substitute another model owner.
    pub fn matches_model(&self, candidate: &Arc<Model>) -> bool {
        Arc::ptr_eq(&self.model, candidate)
    }
    pub fn registration(&self) -> &NnuecV2Registration {
        &self.registration
    }
    pub fn weights_artifact(&self) -> &ArtifactRef {
        &self.weights_artifact
    }
    pub fn provenance_artifact(&self) -> &ArtifactRef {
        &self.provenance_artifact
    }
    pub fn input_lock_digest(&self) -> Digest {
        self.sources.lock_digest()
    }
    pub fn semantic_weights_identity(&self) -> &CpuValueIdentity {
        self.model.identity()
    }
    /// The publication owner calls this outside the search manager. It reads
    /// the same original file descriptors and does not load another model or
    /// recreate any past root, Bound, worker or native execution capability.
    pub fn publication_source_bytes(
        &self,
        control: &RoventStartupControl,
    ) -> Result<(ArtifactRef, Vec<u8>, ArtifactRef, Vec<u8>), BackendError> {
        control.check()?;
        let (weights, weights_bytes, provenance, provenance_bytes) = self
            .sources
            .nnuec_artifact_bytes()?
            .ok_or_else(|| mismatch("original NNUEC publication sources absent"))?;
        if weights != self.weights_artifact || provenance != self.provenance_artifact {
            return Err(mismatch(
                "NNUEC publication source declarations differ from loaded owner",
            ));
        }
        verify_raw_artifact(&weights, &weights_bytes, NNUEC_WEIGHTS_MAX_BYTES, control)?;
        verify_raw_artifact(
            &provenance,
            &provenance_bytes,
            NNUEC_PROVENANCE_MAX_BYTES,
            control,
        )?;
        control.check()?;
        Ok((weights, weights_bytes, provenance, provenance_bytes))
    }
}

/// Read-only metadata verification reuses the actual loader's original byte,
/// provenance and compiled-registration checks. The temporary parsed model is
/// dropped here; no Loaded source, Bound, worker or execution cap is created.
pub(crate) fn verify_metadata_registration(
    sources: &RoventPinnedNativeSources,
    control: &RoventStartupControl,
) -> Result<NnuecV2Registration, BackendError> {
    Ok(verify_source_metadata(sources, control)?.registration)
}

struct VerifiedNnuecMetadata {
    model: Arc<Model>,
    registration: NnuecV2Registration,
    weights_artifact: ArtifactRef,
    provenance_artifact: ArtifactRef,
}

fn verify_source_metadata<L: NativeInputLock>(
    sources: &RoventPinnedNativeSources<L>,
    control: &RoventStartupControl,
) -> Result<VerifiedNnuecMetadata, BackendError> {
    control.check()?;
    if !sources.admits_original_startup_control(control) {
        return Err(mismatch(
            "original NNUEC startup control differs from native sources",
        ));
    }
    sources.lock().validate_original().map_err(|error| {
        invalid("original NNUEC input lock").with_external_cause(CauseCode::ProfileParse, &error)
    })?;
    sources
        .lock()
        .admit_configuration(sources.configuration())?;
    let pin = sources.lock().engine().nnuec.as_ref().ok_or_else(|| {
        fail(
            K::UnsupportedModel,
            "original startup does not declare NNUEC",
        )
    })?;
    if !(1..=NNUEC_WEIGHTS_MAX_BYTES as u64).contains(&pin.weights.bytes)
        || !(1..=NNUEC_PROVENANCE_MAX_BYTES as u64).contains(&pin.provenance.bytes)
    {
        return Err(fail(
            K::ResourceExhausted,
            "declared NNUEC source byte budget",
        ));
    }
    let (weights_artifact, weights, provenance_artifact, provenance) = sources
        .nnuec_artifact_bytes()?
        .ok_or_else(|| mismatch("original NNUEC descriptors are absent"))?;
    control.check()?;
    if weights_artifact != pin.weights || provenance_artifact != pin.provenance {
        return Err(mismatch(
            "original NNUEC retained descriptors differ from lock",
        ));
    }
    verify_raw_artifact(
        &weights_artifact,
        &weights,
        NNUEC_WEIGHTS_MAX_BYTES,
        control,
    )?;
    verify_raw_artifact(
        &provenance_artifact,
        &provenance,
        NNUEC_PROVENANCE_MAX_BYTES,
        control,
    )?;
    let body = decode_provenance(&provenance)?;
    control.check()?;
    // This existing bounded CPU parser is synchronous, not preemptible.
    // Checks on both sides reject late admission without claiming that
    // parsing itself must return by `until` or renewing that boundary.
    let model = Arc::new(Model::from_reader(weights.as_slice()).map_err(|error| {
        invalid("actual NNUEC V2 model load failed")
            .with_external_cause(CauseCode::ModelLoad, &error)
    })?);
    control.check()?;
    let registration = model.registration().map_err(|error| {
        invalid("actual NNUEC V2 registration failed")
            .with_external_cause(CauseCode::ModelLoad, &error)
    })?;
    verify_declarations(
        pin,
        sources.lock().provenance(),
        sources.configuration().nnuec_conditions_digest()?,
        &body,
        &model,
        &registration,
    )?;
    control.check()?;
    Ok(VerifiedNnuecMetadata {
        model,
        registration,
        weights_artifact,
        provenance_artifact,
    })
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct NnuecProvenance {
    domain: String,
    weights_raw_sha256: String,
    semantic_weights_sha256: String,
    schema: String,
    semantics: String,
    encoding: String,
    implementation_sha256: String,
    configuration_sha256: String,
    context_namespace_policy_sha256: String,
    local_namespace_policy_sha256: String,
    #[serde(deserialize_with = "deserialize_strict_training")]
    training: CpuTrainingState,
    diagnostic: bool,
    provenance: Vec<ArtifactRef>,
}

/// Provenance has a strict field contract of its own. Serde's internally
/// tagged *unit* variants may ignore payload fields, even when their enum
/// carries deny_unknown_fields. Empty struct variants enforce the same valid
/// wire shape without changing legacy CpuTrainingState/model parsers.
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum StrictTrainingDeclaration {
    Bootstrap {},
    Untrained {},
    Learned {
        run_id: String,
        steps: u64,
        dataset_sha256: String,
    },
}

fn deserialize_strict_training<'de, D>(deserializer: D) -> Result<CpuTrainingState, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(
        match StrictTrainingDeclaration::deserialize(deserializer)? {
            StrictTrainingDeclaration::Bootstrap {} => CpuTrainingState::Bootstrap,
            StrictTrainingDeclaration::Untrained {} => CpuTrainingState::Untrained,
            StrictTrainingDeclaration::Learned {
                run_id,
                steps,
                dataset_sha256,
            } => CpuTrainingState::Learned {
                run_id,
                steps,
                dataset_sha256,
            },
        },
    )
}

fn decode_provenance(bytes: &[u8]) -> Result<NnuecProvenance, BackendError> {
    if bytes.is_empty() || bytes.len() > NNUEC_PROVENANCE_MAX_BYTES {
        return Err(fail(K::ResourceExhausted, "NNUEC provenance byte budget"));
    }
    let text = std::str::from_utf8(bytes).map_err(|_| invalid("NNUEC provenance UTF-8"))?;
    // The shared decoder rejects duplicate keys at every nesting level. Serde
    // then rejects fields outside this version, including training/artifacts.
    let body: NnuecProvenance = decode_json(text).map_err(|error| {
        invalid("strict NNUEC provenance JSON").with_external_cause(CauseCode::ProfileParse, &error)
    })?;
    validate_provenance(&body)?;
    Ok(body)
}

fn validate_provenance(body: &NnuecProvenance) -> Result<(), BackendError> {
    if body.domain != NNUEC_PROVENANCE_DOMAIN
        || [&body.schema, &body.semantics, &body.encoding]
            .into_iter()
            .any(|tag| tag.is_empty() || tag.len() > 256 || tag.chars().any(char::is_control))
    {
        return Err(invalid("NNUEC provenance domain/model tags"));
    }
    for digest in [
        &body.weights_raw_sha256,
        &body.semantic_weights_sha256,
        &body.implementation_sha256,
        &body.configuration_sha256,
        &body.context_namespace_policy_sha256,
        &body.local_namespace_policy_sha256,
    ] {
        parse_sha256(digest)?;
    }
    CpuValueIdentity {
        semantics: body.semantics.clone(),
        weights_sha256: Some(body.semantic_weights_sha256.clone()),
        training: body.training.clone(),
    }
    .validate()
    .map_err(|_| invalid("NNUEC provenance training/semantic identity"))?;
    if body.provenance.len() > MAX_PROVENANCE_ARTIFACTS {
        return Err(fail(K::ResourceExhausted, "NNUEC source provenance count"));
    }
    let mut paths = BTreeSet::new();
    let mut contents = BTreeSet::new();
    for artifact in &body.provenance {
        artifact.validate().map_err(|error| {
            invalid("NNUEC original source provenance artifact")
                .with_external_cause(CauseCode::ProfileParse, &error)
        })?;
        if !paths.insert(artifact.path.as_str())
            || !contents.insert((artifact.sha256.as_str(), artifact.bytes))
        {
            return Err(invalid("duplicate NNUEC source provenance"));
        }
    }
    Ok(())
}

fn verify_declarations(
    pin: &RoventNnuecPin,
    source_provenance: &[ArtifactRef],
    configured_conditions: Option<[u8; 32]>,
    body: &NnuecProvenance,
    model: &Model,
    registration: &NnuecV2Registration,
) -> Result<(), BackendError> {
    validate_provenance(body)?;
    let conditions = configured_conditions
        .ok_or_else(|| mismatch("actual NNUEC search conditions are absent"))?;
    if pin.search_conditions_sha256 != hex(conditions)
        || pin.schema != registration.schema()
        || pin.semantics != registration.semantics()
        || pin.encoding != registration.encoding()
        || pin.implementation_sha256 != registration.implementation_sha256()
        || pin.configuration_sha256 != registration.configuration_sha256()
        || pin.context_namespace_policy_sha256 != registration.context_namespace_policy_sha256()
        || pin.local_namespace_policy_sha256 != registration.local_namespace_policy_sha256()
    {
        return Err(mismatch(
            "actual NNUEC compiled registration differs from lock",
        ));
    }
    if registration.model_identity() != model.identity()
        || model.identity().weights_sha256.as_deref() != Some(body.semantic_weights_sha256.as_str())
        || body.weights_raw_sha256 != pin.weights.sha256
        || body.schema != registration.schema()
        || body.semantics != registration.semantics()
        || body.encoding != registration.encoding()
        || body.implementation_sha256 != registration.implementation_sha256()
        || body.configuration_sha256 != registration.configuration_sha256()
        || body.context_namespace_policy_sha256 != registration.context_namespace_policy_sha256()
        || body.local_namespace_policy_sha256 != registration.local_namespace_policy_sha256()
        || body.training != model.identity().training
        || body.diagnostic != pin.diagnostic
    {
        return Err(mismatch(
            "actual NNUEC provenance differs from loaded model or lock",
        ));
    }
    if body
        .provenance
        .iter()
        .any(|artifact| !source_provenance.contains(artifact))
    {
        return Err(mismatch(
            "NNUEC source provenance is absent from original input lock",
        ));
    }
    Ok(())
}

fn verify_raw_artifact(
    artifact: &ArtifactRef,
    bytes: &[u8],
    max_bytes: usize,
    control: &RoventStartupControl,
) -> Result<(), BackendError> {
    control.check()?;
    artifact.validate().map_err(|error| {
        invalid("NNUEC retained artifact declaration")
            .with_external_cause(CauseCode::ProfileParse, &error)
    })?;
    if bytes.is_empty() || bytes.len() > max_bytes || artifact.bytes > max_bytes as u64 {
        return Err(fail(
            K::ResourceExhausted,
            "NNUEC retained artifact byte budget",
        ));
    }
    if u64::try_from(bytes.len()).ok() != Some(artifact.bytes) {
        return Err(mismatch("NNUEC retained artifact byte count"));
    }
    let mut hash = Sha256::new();
    for chunk in bytes.chunks(64 << 10) {
        control.check()?;
        hash.update(chunk);
    }
    let actual: [u8; 32] = hash.finalize().into();
    if parse_sha256(&artifact.sha256)? != actual {
        return Err(mismatch("NNUEC retained artifact raw SHA-256"));
    }
    control.check()
}

fn invalid(detail: &'static str) -> BackendError {
    fail(K::InvalidInput, detail)
}
fn mismatch(detail: &'static str) -> BackendError {
    fail(K::IdentityMismatch, detail)
}

#[cfg(test)]
mod tests;
