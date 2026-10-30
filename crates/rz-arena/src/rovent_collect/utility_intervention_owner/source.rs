//! 원 첫 등록 root와 trial별 실제 자원 관측의 bounded 자료 projection.
//! 공개 body는 역사 자료이며 현재 Product/Root/Loaded 권한을 발급하지 않는다.
use super::*;
use rz_experiments::ArtifactRef;
use rz_experiments::rovent_corpus_v2::RoventCorpusManifestV2;
use rz_uci::pals_adaptive::{UtilityCohortSourcePins, UtilityInterventionCohortOwner};
use rz_uci::rovent_bootstrap::nnuec_source::LoadedRoventNnuecSource;

pub const ROVENT_UTILITY_INTERVENTION_COLLECTION_SOURCE: &str =
    "rz-rovent-utility-cohort-collection-source/2";

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventUtilityInterventionCollectionSource {
    pub schema: String,
    pub source_pins: UtilityCohortSourcePins,
    pub nnuec_registration: Value,
    pub nnuec_weights_artifact: ArtifactRef,
    pub nnuec_provenance_artifact: ArtifactRef,
    pub native_origin: Value,
    /// 첫 trial이 미자료화여도 원 cohort ID preimage를 버리지 않는다.
    pub registration_root: Value,
    pub collection_manifest: RoventCorpusManifestV2,
    pub collection_metadata: RoventCollectionMetadata,
    pub rules_root: RoventRulesPositionReplay,
    pub transposition_sha256: String,
    pub frozen_pc_sources: Vec<RoventHistoricalUtilityFrozenPcSource>,
    pub ancestor_datasets: Vec<RoventUtilityCohortAncestorSource>,
    /// 각 fresh Product의 독립 resource custody 관측. 시계 영역을 합치지 않는다.
    pub resource_observation: Vec<Value>,
}

impl RoventUtilityInterventionCollectionSource {
    pub fn capture_original(
        owner: &UtilityInterventionCohortOwner,
        registration_root: &RoventCollectionRootAdmission,
        source: &LoadedRoventNnuecSource,
        manifest: &RoventCorpusManifestV2,
        metadata: &RoventCollectionMetadata,
        ancestor_datasets: Vec<RoventUtilityCohortAncestorSource>,
        resource_observation: Vec<Value>,
    ) -> Result<Self, ArenaError> {
        manifest.validate()?;
        let origin = registration_root.native_origin();
        let actual = registration_root.root_admission();
        let lock = origin.original_input_lock();
        check(
            origin.matches_loaded_nnuec(source)
                && source.input_lock_digest() == actual.startup_lock_digest()
                && owner.source_pins().input_lock_sha256 == hex(source.input_lock_digest())
                && owner.source_pins().source_sha256 == manifest.source_sha256
                && resource_observation.len() <= 3
                && ancestor_datasets.len() <= 256,
            "utility cohort original source/root/manifest/observation coverage",
        )?;
        let mut ancestors = BTreeSet::new();
        for descriptor in &ancestor_datasets {
            validate_intervention_ancestor(descriptor)?;
            check(
                ancestors.insert(descriptor.publication_sha256.clone()),
                "utility cohort duplicate original ancestor",
            )?;
        }
        let raw = origin.raw_receipt();
        let string = |key: &str| -> Result<String, ArenaError> {
            raw[key]
                .as_str()
                .filter(|value| generation_sha(value))
                .map(str::to_string)
                .ok_or_else(|| invalid("utility cohort original native namespace missing"))
        };
        let mut frozen_pc_sources = Vec::with_capacity(2);
        for (role, model) in [
            ("proposer", &lock.manifest.engine.pc.proposer),
            ("critic", &lock.manifest.engine.pc.critic),
        ] {
            check(
                !model.diagnostic && model.weights.sha256 == string("weights_sha256")?,
                "utility cohort frozen role differs from actual loaded model",
            )?;
            frozen_pc_sources.push(RoventHistoricalUtilityFrozenPcSource {
                role: role.into(),
                artifact: model.weights.clone(),
                model_semantics_sha256: string("model_semantics_sha256")?,
                model_identity_sha256: string("weights_sha256")?,
                encoding_sha256: string("encoding_sha256")?,
            });
        }
        let position = Position::from_snapshot(actual.root_snapshot(), Default::default())
            .map_err(|error| invalid(error.to_string()))?;
        let value = Self {
            schema: ROVENT_UTILITY_INTERVENTION_COLLECTION_SOURCE.into(),
            source_pins: owner.source_pins().clone(),
            nnuec_registration: serde_json::to_value(source.registration())
                .map_err(|error| invalid(error.to_string()))?,
            nnuec_weights_artifact: source.weights_artifact().clone(),
            nnuec_provenance_artifact: source.provenance_artifact().clone(),
            native_origin: raw.clone(),
            registration_root: registration_root.raw_receipt().clone(),
            collection_manifest: manifest.clone(),
            collection_metadata: metadata.clone(),
            rules_root: RoventRulesPositionReplay::capture(&position)?,
            transposition_sha256: crate::pals_collect_v3::generation_transposition_sha256(
                &position,
            ),
            frozen_pc_sources,
            ancestor_datasets,
            resource_observation,
        };
        check(
            serde_json::to_vec(&value)
                .map_err(|error| invalid(error.to_string()))?
                .len()
                <= 1 << 20,
            "utility cohort source projection exceeds 1 MiB",
        )?;
        Ok(value)
    }
}
