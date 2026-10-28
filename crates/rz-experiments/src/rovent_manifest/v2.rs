//! Additive policy-aware Rovent declarations. These records validate declared
//! relationships only: they do not open artifacts, issue runtime authority,
//! or attest a full execution, physical completion, or product acceptance.
//!
//! The /1 declaration, lock, receipt and their canonical bytes are unchanged.
//! A /2 consumer must retain its typed /2 lock throughout admission; this
//! module deliberately provides no conversion into a /1 lock or receipt.

use super::*;
use std::num::NonZeroU64;

pub const ROVENT_EXECUTION_V2_DOMAIN: &str = "rz-rovent-execution/2";
pub const ROVENT_LOCK_V2_DOMAIN: &str = "rz-rovent-input-lock/2";
pub const ROVENT_CANONICAL_V2_DOMAIN: &str = "rz-rovent-execution-canonical/2";
pub const ROVENT_RECEIPT_V2_DOMAIN: &str = "rz-rovent-execution-receipt/2";

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RoventRootTiePolicyV2 {
    LegacyFirstSearched,
    RulesOrderConfirmed,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RoventOwnRNotApplicableV2 {
    PcOnly,
    ExternalResearcher,
}

/// The expected worker conditions digest is the actual CPU engine's
/// `rz-cpu-search-conditions/1` identity. It is distinct from the NNUEC plan
/// digest in `RoventNnuecPin::search_conditions_sha256`.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RoventRootTieDeclarationV2 {
    NotApplicable {
        reason: RoventOwnRNotApplicableV2,
    },
    Own {
        policy: RoventRootTiePolicyV2,
        expected_search_identity: String,
        expected_worker_conditions_sha256: String,
    },
}

/// Full /2 declaration. Shared /1-shaped fields are checked using the /1
/// *declaration* validator only; no /1 lock or accepted receipt is produced.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventExecutionManifestV2 {
    pub domain: String,
    pub contract_revision: String,
    pub run_id: String,
    pub purpose: RoventPurpose,
    pub engine: RoventEngineDeclaration,
    pub resources: RoventResourcePlan,
    pub coordinator: RoventCoordinatorDeclaration,
    pub wall_time_max_ms: u64,
    pub cleanup_max_ms: u64,
    pub output_bytes_max: u64,
    pub agent_renewable: bool,
    pub provenance: Vec<ArtifactRef>,
    pub startup_document: ArtifactRef,
    pub startup_configuration_sha256: String,
    pub startup_search_configuration_sha256: String,
    pub own_r_root_tie: RoventRootTieDeclarationV2,
}

impl RoventExecutionManifestV2 {
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let manifest: Self = decode_json(input)?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            self.domain == ROVENT_EXECUTION_V2_DOMAIN,
            "Rovent /2 execution domain mismatch",
        )?;
        // This temporary value is private and validates shared declaration
        // fields only. It is never locked, returned, or used as admission.
        RoventExecutionManifest {
            domain: ROVENT_EXECUTION_DOMAIN.into(),
            contract_revision: self.contract_revision.clone(),
            run_id: self.run_id.clone(),
            purpose: self.purpose,
            engine: self.engine.clone(),
            resources: self.resources.clone(),
            coordinator: self.coordinator.clone(),
            wall_time_max_ms: self.wall_time_max_ms,
            cleanup_max_ms: self.cleanup_max_ms,
            output_bytes_max: self.output_bytes_max,
            agent_renewable: self.agent_renewable,
            provenance: self.provenance.clone(),
        }
        .validate()?;
        self.startup_document.validate()?;
        require(
            sha(&self.startup_configuration_sha256)
                && sha(&self.startup_search_configuration_sha256)
                && self.engine.search.search_configuration_sha256
                    == self.startup_search_configuration_sha256,
            "startup configuration/search declaration mismatch",
        )?;
        match (&self.engine.researcher, &self.own_r_root_tie) {
            (
                None,
                RoventRootTieDeclarationV2::NotApplicable {
                    reason: RoventOwnRNotApplicableV2::PcOnly,
                },
            ) if !self.engine.mode.has_native_r() => Ok(()),
            (
                Some(RoventResearcher::External { .. }),
                RoventRootTieDeclarationV2::NotApplicable {
                    reason: RoventOwnRNotApplicableV2::ExternalResearcher,
                },
            ) if self.engine.mode.has_native_r() => Ok(()),
            (
                Some(RoventResearcher::Own { .. }),
                RoventRootTieDeclarationV2::Own {
                    expected_search_identity,
                    expected_worker_conditions_sha256,
                    ..
                },
            ) if self.engine.mode.has_native_r() => require(
                text(expected_search_identity) && sha(expected_worker_conditions_sha256),
                "own R needs bounded expected worker identity and conditions digest",
            ),
            _ => Err(ManifestError::Integrity(
                "root tie declaration does not match actual R assembly".into(),
            )),
        }
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        self.validate()?;
        canonical(&(ROVENT_CANONICAL_V2_DOMAIN, self))
    }

    pub fn lock(self) -> Result<RoventInputLockV2, ManifestError> {
        let canonical_sha256 = digest(&self.canonical_bytes()?);
        Ok(RoventInputLockV2 {
            domain: ROVENT_LOCK_V2_DOMAIN.into(),
            manifest: self,
            canonical_sha256,
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventInputLockV2 {
    pub domain: String,
    pub manifest: RoventExecutionManifestV2,
    pub canonical_sha256: String,
}

impl RoventInputLockV2 {
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let lock: Self = decode_json(input)?;
        lock.validate()?;
        Ok(lock)
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            self.domain == ROVENT_LOCK_V2_DOMAIN
                && sha(&self.canonical_sha256)
                && digest(&self.manifest.canonical_bytes()?) == self.canonical_sha256,
            "Rovent /2 lock domain/canonical digest mismatch",
        )
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RoventOwnRNotIssuedReasonV2 {
    NoRWorkRequested,
    NnuecPreMaterializationStop,
}

/// This is the NNUEC plan/namespace domain, not the CPU worker's conditions
/// domain. Board-only own R explicitly records that this plan is inapplicable.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum RoventNnuecPlanObservationV2 {
    Unknown,
    NotApplicable,
    Observed {
        conditions_sha256: String,
        evidence: ArtifactRef,
    },
}

/// Explicit work observations: absent instrumentation is never inferred to be
/// a zero, and external-R work is not counted as own-R work.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum RoventOwnRActivityV2 {
    Unknown,
    NotApplicable {
        evidence: ArtifactRef,
    },
    NotIssued {
        evidence: ArtifactRef,
    },
    Issued {
        jobs: NonZeroU64,
        /// This record covers one owner and its sealed job subset only. A
        /// multi-owner execution needs a separate complete owner-set join.
        owner_id: String,
        root_authority_sha256: String,
        job_set_sha256: String,
        evidence: ArtifactRef,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum RoventRootTieObservationV2 {
    Unknown,
    NotApplicable {
        reason: RoventOwnRNotApplicableV2,
        evidence: ArtifactRef,
    },
    NotIssued {
        reason: RoventOwnRNotIssuedReasonV2,
        evidence: ArtifactRef,
    },
    Observed {
        policy: RoventRootTiePolicyV2,
        search_identity: String,
        worker_conditions_sha256: String,
        nnuec_plan: RoventNnuecPlanObservationV2,
        root_state_and_history_sha256: String,
        root_authority_sha256: String,
        owner_id: String,
        ready_evidence: ArtifactRef,
        /// Owner retirement is separate from ready; absence is not completion.
        retirement_evidence: Option<ArtifactRef>,
    },
}

/// A structurally consistent record is not a verified artifact or an accepted
/// execution. Product producers/consumers must verify original owner custody,
/// artifact bytes, resource/process closure and GPU fences separately.
/// In particular, `job_set_sha256` is a declared seal: this comparison does
/// not read the job-set evidence or prove membership. A pre-materialization
/// stop also needs a later join to the actual startup and owner witness; the
/// mere presence of an NNUEC pin is not proof that own R was selected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventPolicyRecordConsistencyV2 {
    Unknown,
    ConsistentUnattested,
    Mismatch,
}

/// Policy portion of a /2 execution record. It intentionally has no
/// `functional_accepted` or /1 projection: this alone cannot accept a run.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventExecutionReceiptV2 {
    pub domain: String,
    pub run_id: String,
    pub input_lock_sha256: String,
    pub own_r_activity: RoventOwnRActivityV2,
    pub root_tie_observation: RoventRootTieObservationV2,
    pub execution_evidence: ArtifactRef,
}

impl RoventExecutionReceiptV2 {
    /// Parsing returns the policy verdict explicitly. Even a consistent
    /// verdict remains unattested and cannot accept an execution by itself.
    pub fn parse_and_assess(
        input: &str,
        lock: &RoventInputLockV2,
    ) -> Result<(Self, RoventPolicyRecordConsistencyV2), ManifestError> {
        let receipt: Self = decode_json(input)?;
        let verdict = receipt.policy_consistency(lock)?;
        Ok((receipt, verdict))
    }

    pub fn policy_consistency(
        &self,
        lock: &RoventInputLockV2,
    ) -> Result<RoventPolicyRecordConsistencyV2, ManifestError> {
        lock.validate()?;
        require(
            self.domain == ROVENT_RECEIPT_V2_DOMAIN
                && self.run_id == lock.manifest.run_id
                && self.input_lock_sha256 == lock.canonical_sha256,
            "Rovent /2 receipt is not bound to this /2 lock",
        )?;
        self.execution_evidence.validate()?;
        match &self.own_r_activity {
            RoventOwnRActivityV2::Unknown => {}
            RoventOwnRActivityV2::NotApplicable { evidence }
            | RoventOwnRActivityV2::NotIssued { evidence } => evidence.validate()?,
            RoventOwnRActivityV2::Issued {
                owner_id,
                root_authority_sha256,
                job_set_sha256,
                evidence,
                ..
            } => {
                evidence.validate()?;
                require(
                    text(owner_id) && sha(root_authority_sha256) && sha(job_set_sha256),
                    "invalid own R owner/job authority",
                )?;
            }
        }
        match &self.root_tie_observation {
            RoventRootTieObservationV2::Unknown => {}
            RoventRootTieObservationV2::NotApplicable { evidence, .. }
            | RoventRootTieObservationV2::NotIssued { evidence, .. } => evidence.validate()?,
            RoventRootTieObservationV2::Observed {
                search_identity,
                worker_conditions_sha256,
                nnuec_plan,
                root_state_and_history_sha256,
                root_authority_sha256,
                owner_id,
                ready_evidence,
                retirement_evidence,
                ..
            } => {
                ready_evidence.validate()?;
                if let Some(retirement) = retirement_evidence {
                    retirement.validate()?;
                }
                if let RoventNnuecPlanObservationV2::Observed {
                    conditions_sha256,
                    evidence,
                } = nnuec_plan
                {
                    evidence.validate()?;
                    require(sha(conditions_sha256), "invalid observed NNUEC plan digest")?;
                }
                require(
                    text(search_identity)
                        && sha(worker_conditions_sha256)
                        && sha(root_state_and_history_sha256)
                        && sha(root_authority_sha256)
                        && text(owner_id),
                    "invalid observed own R identity",
                )?;
            }
        }
        if matches!(
            (&lock.manifest.own_r_root_tie, &self.own_r_activity),
            (
                RoventRootTieDeclarationV2::NotApplicable { .. },
                RoventOwnRActivityV2::NotIssued { .. } | RoventOwnRActivityV2::Issued { .. }
            ) | (
                RoventRootTieDeclarationV2::Own { .. },
                RoventOwnRActivityV2::NotApplicable { .. }
            )
        ) {
            return Ok(RoventPolicyRecordConsistencyV2::Mismatch);
        }
        let result = match (&lock.manifest.own_r_root_tie, &self.root_tie_observation) {
            (_, RoventRootTieObservationV2::Unknown) => RoventPolicyRecordConsistencyV2::Unknown,
            (
                RoventRootTieDeclarationV2::NotApplicable { reason: expected },
                RoventRootTieObservationV2::NotApplicable { reason, .. },
            ) => {
                if reason != expected {
                    RoventPolicyRecordConsistencyV2::Mismatch
                } else if matches!(self.own_r_activity, RoventOwnRActivityV2::Unknown) {
                    RoventPolicyRecordConsistencyV2::Unknown
                } else if matches!(
                    self.own_r_activity,
                    RoventOwnRActivityV2::NotApplicable { .. }
                ) {
                    RoventPolicyRecordConsistencyV2::ConsistentUnattested
                } else {
                    RoventPolicyRecordConsistencyV2::Mismatch
                }
            }
            (
                RoventRootTieDeclarationV2::Own { .. },
                RoventRootTieObservationV2::NotIssued { reason, .. },
            ) => {
                let valid_reason = *reason
                    != RoventOwnRNotIssuedReasonV2::NnuecPreMaterializationStop
                    || lock.manifest.engine.nnuec.is_some();
                if !valid_reason {
                    RoventPolicyRecordConsistencyV2::Mismatch
                } else if matches!(self.own_r_activity, RoventOwnRActivityV2::Unknown) {
                    RoventPolicyRecordConsistencyV2::Unknown
                } else if matches!(self.own_r_activity, RoventOwnRActivityV2::NotIssued { .. }) {
                    RoventPolicyRecordConsistencyV2::ConsistentUnattested
                } else {
                    RoventPolicyRecordConsistencyV2::Mismatch
                }
            }
            (
                RoventRootTieDeclarationV2::Own {
                    policy: expected_policy,
                    expected_search_identity,
                    expected_worker_conditions_sha256,
                },
                RoventRootTieObservationV2::Observed {
                    policy,
                    search_identity,
                    worker_conditions_sha256,
                    nnuec_plan,
                    root_state_and_history_sha256,
                    root_authority_sha256,
                    owner_id,
                    ..
                },
            ) => {
                let plan = match (&lock.manifest.engine.nnuec, nnuec_plan) {
                    (_, RoventNnuecPlanObservationV2::Unknown) => {
                        RoventPolicyRecordConsistencyV2::Unknown
                    }
                    (None, RoventNnuecPlanObservationV2::NotApplicable) => {
                        RoventPolicyRecordConsistencyV2::ConsistentUnattested
                    }
                    (
                        Some(pin),
                        RoventNnuecPlanObservationV2::Observed {
                            conditions_sha256, ..
                        },
                    ) if pin.search_conditions_sha256 == *conditions_sha256 => {
                        RoventPolicyRecordConsistencyV2::ConsistentUnattested
                    }
                    _ => RoventPolicyRecordConsistencyV2::Mismatch,
                };
                if policy != expected_policy
                    || search_identity != expected_search_identity
                    || worker_conditions_sha256 != expected_worker_conditions_sha256
                    || root_state_and_history_sha256
                        != &lock.manifest.engine.search.root_state_and_history_sha256
                    || plan == RoventPolicyRecordConsistencyV2::Mismatch
                {
                    RoventPolicyRecordConsistencyV2::Mismatch
                } else if matches!(&self.own_r_activity, RoventOwnRActivityV2::Issued { owner_id: issued_owner, root_authority_sha256: issued_authority, .. } if issued_owner != owner_id || issued_authority != root_authority_sha256)
                {
                    RoventPolicyRecordConsistencyV2::Mismatch
                } else if matches!(self.own_r_activity, RoventOwnRActivityV2::Unknown)
                    || plan == RoventPolicyRecordConsistencyV2::Unknown
                {
                    RoventPolicyRecordConsistencyV2::Unknown
                } else {
                    RoventPolicyRecordConsistencyV2::ConsistentUnattested
                }
            }
            // Preserve well-formed but mismatching evidence as a mismatch,
            // never as a successful execution or an inferred zero.
            _ => RoventPolicyRecordConsistencyV2::Mismatch,
        };
        Ok(result)
    }

    pub fn canonical_bytes(&self, lock: &RoventInputLockV2) -> Result<Vec<u8>, ManifestError> {
        self.policy_consistency(lock)?;
        canonical(&(ROVENT_RECEIPT_V2_DOMAIN, self))
    }
}
