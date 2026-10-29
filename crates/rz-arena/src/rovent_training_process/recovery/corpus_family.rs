//! Closed PC corpus-family dispatch. Original typed documents and their
//! canonical seals survive admission; common roster metadata has no codec,
//! schema replacement, serialization or independent digest authority.
use super::*;
use crate::rovent_collect::RoventCorpusPublicationV2;
use rz_experiments::rovent_corpus::{RoventCorpusChunk, RoventTeacherPin};
use rz_experiments::rovent_corpus_v2::{
    RoventCorpusIndexV2, RoventCorpusManifestV2, RoventTrainingRecordV2,
};
use rz_experiments::rovent_submitted_input::ROVENT_PUBLICATION_V2_DOMAIN;

#[derive(Clone, Debug, PartialEq)]
pub(super) enum OriginalPublication {
    Legacy(RoventCorpusPublication),
    Submitted(RoventCorpusPublicationV2),
}
#[derive(Clone, Debug, PartialEq)]
pub(super) enum OriginalManifest {
    Legacy(RoventCorpusManifest),
    Submitted(RoventCorpusManifestV2),
}
#[derive(Clone, Debug, PartialEq)]
pub(super) enum OriginalIndex {
    Legacy(RoventCorpusIndex),
    Submitted(RoventCorpusIndexV2),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct HistoricalPublicationMetadata {
    pub(super) manifest: RoventPublicationFile,
    pub(super) index: RoventPublicationFile,
    pub(super) chunks: Vec<RoventPublicationFile>,
    pub(super) canonical_manifest_sha256: String,
    pub(super) canonical_index_sha256: String,
    pub(super) source_commit: String,
    pub(super) source_sha256: String,
    pub(super) runtime_sha256: String,
    pub(super) record_count: u64,
    pub(super) total_payload_bytes: u64,
    pub(super) native_admission_sha256: Vec<String>,
    pub(super) admissions: Vec<crate::rovent_collect::RoventPublicationAdmission>,
}
#[derive(Clone, Debug, PartialEq)]
pub(super) struct HistoricalCollectionMetadata {
    pub(super) corpus_id: String,
    pub(super) source_commit: String,
    pub(super) source_sha256: String,
    pub(super) runtime_sha256: String,
    pub(super) model_semantics_sha256: String,
    pub(super) model_epoch: u64,
    pub(super) context_epoch: u64,
    pub(super) recipe_sha256: String,
    pub(super) stage_id: String,
    pub(super) round: u64,
    pub(super) round_id: String,
    pub(super) student: String,
    pub(super) learner: String,
    pub(super) one_learner: bool,
    pub(super) inherited_input_sha256: Vec<String>,
    pub(super) inherited_origin_sha256: Vec<String>,
    pub(super) teachers: BTreeMap<String, RoventTeacherPin>,
    pub(super) opening_splits: BTreeMap<String, GenerationSplitV3>,
    pub(super) ancestor_index_sha256: Vec<String>,
}
impl HistoricalCollectionMetadata {
    pub(super) fn legacy(m: &RoventCorpusManifest) -> Self {
        Self {
            corpus_id: m.corpus_id.clone(),
            source_commit: m.source_commit.clone(),
            source_sha256: m.source_sha256.clone(),
            runtime_sha256: m.runtime_sha256.clone(),
            model_semantics_sha256: m.model_semantics_sha256.clone(),
            model_epoch: m.model_epoch,
            context_epoch: m.context_epoch,
            recipe_sha256: m.recipe_sha256.clone(),
            stage_id: m.stage_id.clone(),
            round: m.round,
            round_id: m.round_id.clone(),
            student: m.student.clone(),
            learner: m.learner.clone(),
            one_learner: m.one_learner,
            inherited_input_sha256: m.inherited_input_sha256.clone(),
            inherited_origin_sha256: m.inherited_origin_sha256.clone(),
            teachers: m.teachers.clone(),
            opening_splits: m.opening_splits.clone(),
            ancestor_index_sha256: m.ancestor_index_sha256.clone(),
        }
    }
    pub(super) fn submitted(m: &RoventCorpusManifestV2) -> Self {
        // E's explicitly named projection shares registration validation only.
        // Its temporary schema is never retained, encoded or hashed here.
        Self::legacy(&m.common_registration_projection())
    }
}
#[derive(Clone, Debug, PartialEq)]
pub(super) struct HistoricalIndexMetadata {
    pub(super) record_count: u64,
    pub(super) chunks: Vec<RoventCorpusChunk>,
    pub(super) origins: Vec<RoventCorpusOrigin>,
    canonical_sha256: String,
}
impl HistoricalIndexMetadata {
    /// Projection of an already validated original index; never a new codec
    /// or a replacement for that index's domain-specific canonical digest.
    pub(super) fn from_validated_parts(
        record_count: u64,
        chunks: Vec<RoventCorpusChunk>,
        origins: Vec<RoventCorpusOrigin>,
        canonical_sha256: String,
    ) -> Self {
        Self { record_count, chunks, origins, canonical_sha256 }
    }
    /// Compatibility query of the already checked original family seal. This
    /// common metadata has no serializer and never computes a legacy digest.
    pub(super) fn sha256(&self) -> Result<String, rz_experiments::ManifestError> {
        Ok(self.canonical_sha256.clone())
    }
}
impl OriginalPublication {
    pub(super) fn decode(raw: &[u8]) -> Result<Self, ArenaError> {
        let value: Value = decode(raw)?;
        match value["schema"].as_str() {
            Some("rz-rovent-corpus-publication/1") => {
                let p: RoventCorpusPublication =
                    serde_json::from_value(value).map_err(|e| invalid(e.to_string()))?;
                p.validate()?;
                Ok(Self::Legacy(p))
            }
            Some(ROVENT_PUBLICATION_V2_DOMAIN) => {
                let p: RoventCorpusPublicationV2 =
                    serde_json::from_value(value).map_err(|e| invalid(e.to_string()))?;
                p.validate()?;
                Ok(Self::Submitted(p))
            }
            _ => Err(invalid("unsupported original PC publication family")),
        }
    }
    pub(super) fn family(&self) -> RoventPcRecordFamily {
        match self {
            Self::Legacy(_) => RoventPcRecordFamily::LegacyRecordV1,
            Self::Submitted(_) => RoventPcRecordFamily::SubmittedInputV1,
        }
    }
    pub(super) fn metadata(&self) -> HistoricalPublicationMetadata {
        macro_rules! common {
            ($p:expr) => {{
                let p = $p;
                HistoricalPublicationMetadata {
                    manifest: p.manifest.clone(),
                    index: p.index.clone(),
                    chunks: p.chunks.clone(),
                    canonical_manifest_sha256: p.canonical_manifest_sha256.clone(),
                    canonical_index_sha256: p.canonical_index_sha256.clone(),
                    source_commit: p.source_commit.clone(),
                    source_sha256: p.source_sha256.clone(),
                    runtime_sha256: p.runtime_sha256.clone(),
                    record_count: p.record_count,
                    total_payload_bytes: p.total_payload_bytes,
                    native_admission_sha256: p.native_admission_sha256.clone(),
                    admissions: p.admissions.clone(),
                }
            }};
        }
        match self {
            Self::Legacy(p) => common!(p),
            Self::Submitted(p) => common!(p),
        }
    }
    fn json(&self) -> Result<Value, ArenaError> {
        match self {
            Self::Legacy(p) => serde_json::to_value(p),
            Self::Submitted(p) => serde_json::to_value(p),
        }
        .map_err(|e| invalid(e.to_string()))
    }
}
impl OriginalManifest {
    pub(super) fn decode(raw: &[u8], family: RoventPcRecordFamily) -> Result<Self, ArenaError> {
        match family {
            RoventPcRecordFamily::LegacyRecordV1 => {
                let m: RoventCorpusManifest = decode(raw)?;
                m.validate()?;
                Ok(Self::Legacy(m))
            }
            RoventPcRecordFamily::SubmittedInputV1 => {
                let m: RoventCorpusManifestV2 = decode(raw)?;
                m.validate()?;
                Ok(Self::Submitted(m))
            }
        }
    }
    pub(super) fn sha256(&self) -> Result<String, ArenaError> {
        match self {
            Self::Legacy(m) => m.sha256(),
            Self::Submitted(m) => m.sha256(),
        }
        .map_err(|e| invalid(e.to_string()))
    }
    pub(super) fn metadata(&self) -> HistoricalCollectionMetadata {
        match self {
            Self::Legacy(m) => HistoricalCollectionMetadata::legacy(m),
            Self::Submitted(m) => HistoricalCollectionMetadata::submitted(m),
        }
    }
    fn json(&self) -> Result<Value, ArenaError> {
        match self {
            Self::Legacy(m) => serde_json::to_value(m),
            Self::Submitted(m) => serde_json::to_value(m),
        }
        .map_err(|e| invalid(e.to_string()))
    }
}
impl OriginalIndex {
    pub(super) fn decode(raw: &[u8], manifest: &OriginalManifest) -> Result<Self, ArenaError> {
        match manifest {
            OriginalManifest::Legacy(m) => {
                let i: RoventCorpusIndex = decode(raw)?;
                i.validate()?;
                check(
                    i.manifest_sha256 == m.sha256()?,
                    "legacy original index manifest seal",
                )?;
                Ok(Self::Legacy(i))
            }
            OriginalManifest::Submitted(m) => {
                let i: RoventCorpusIndexV2 = decode(raw)?;
                i.validate(m)?;
                Ok(Self::Submitted(i))
            }
        }
    }
    pub(super) fn sha256(&self) -> Result<String, ArenaError> {
        match self {
            Self::Legacy(i) => i.sha256(),
            Self::Submitted(i) => i.sha256(),
        }
        .map_err(|e| invalid(e.to_string()))
    }
    pub(super) fn metadata(&self) -> Result<HistoricalIndexMetadata, ArenaError> {
        let canonical_sha256 = self.sha256()?;
        let (record_count, chunks, origins) = match self {
            Self::Legacy(i) => (i.record_count, i.chunks.clone(), i.origins.clone()),
            Self::Submitted(i) => (i.record_count, i.chunks.clone(), i.origins.clone()),
        };
        Ok(HistoricalIndexMetadata {
            record_count,
            chunks,
            origins,
            canonical_sha256,
        })
    }
    fn json(&self) -> Result<Value, ArenaError> {
        match self {
            Self::Legacy(i) => serde_json::to_value(i),
            Self::Submitted(i) => serde_json::to_value(i),
        }
        .map_err(|e| invalid(e.to_string()))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct OriginalPcCorpus {
    pub(super) publication: OriginalPublication,
    pub(super) manifest: OriginalManifest,
    pub(super) index: OriginalIndex,
}
impl OriginalPcCorpus {
    pub(super) fn json(&self) -> Result<Value, ArenaError> {
        Ok(
            json!({"publication":self.publication.json()?,"manifest":self.manifest.json()?,"index":self.index.json()?}),
        )
    }
}

/// Strict source probe, not an opaque dataset or execution admission. Each
/// original family is decoded before its own canonical index key is returned.
pub(super) fn probe_registered_pc_dataset(
    registration: &RoventHistoricalDatasetRegistration,
    budget: &RoventRecoveryReadBudget,
) -> Result<(RoventPcRecordFamily, String, Vec<String>), ArenaError> {
    let (_, raw) = budget.read(&registration.publication, MAX_METADATA_BYTES, true)?;
    let publication = OriginalPublication::decode(&raw)?;
    let family = publication.family();
    let p = publication.metadata();
    let (_, raw) = budget.read(
        &archive_pin(&registration.root, &p.manifest)?,
        MAX_METADATA_BYTES,
        true,
    )?;
    let manifest = OriginalManifest::decode(&raw, family)?;
    check(
        manifest.sha256()? == p.canonical_manifest_sha256,
        "original PC probe manifest canonical seal mismatch",
    )?;
    let (_, raw) = budget.read(
        &archive_pin(&registration.root, &p.index)?,
        MAX_METADATA_BYTES,
        true,
    )?;
    let index = OriginalIndex::decode(&raw, &manifest)?;
    check(
        index.sha256()? == p.canonical_index_sha256,
        "original PC probe index canonical seal mismatch",
    )?;
    Ok((
        family,
        p.canonical_index_sha256,
        manifest.metadata().ancestor_index_sha256,
    ))
}

/// Small original lineage projection, built only after the correct typed row
/// and its Rules/native evidence pass admission. It contains no input tensor.
#[derive(Clone, Debug)]
pub(super) struct HistoricalPcLineageRow {
    pub(super) family: RoventPcRecordFamily,
    pub(super) origin: RoventCorpusOrigin,
    pub(super) rules_state_sha256: String,
}
pub(super) enum OriginalPcRow {
    Legacy(rz_experiments::rovent_corpus::RoventTrainingRecord),
    Submitted(RoventTrainingRecordV2),
}
impl OriginalPcRow {
    pub(super) fn decode(line: &str, manifest: &OriginalManifest) -> Result<Self, ArenaError> {
        match manifest {
            OriginalManifest::Legacy(m) => {
                let r: rz_experiments::rovent_corpus::RoventTrainingRecord =
                    rz_experiments::decode_json(line)?;
                r.validate_rules_replay_v2(m)?;
                Ok(Self::Legacy(r))
            }
            OriginalManifest::Submitted(m) => {
                let r: RoventTrainingRecordV2 = rz_experiments::decode_json(line)?;
                r.validate(m)?;
                Ok(Self::Submitted(r))
            }
        }
    }
    pub(super) fn role(&self) -> GenerationRoleV3 {
        match self {
            Self::Legacy(r) => r.role,
            Self::Submitted(r) => r.role,
        }
    }
    pub(super) fn native(&self) -> &rz_experiments::rovent_corpus::RoventNativeLifecycle {
        match self {
            Self::Legacy(r) => &r.native,
            Self::Submitted(r) => &r.native,
        }
    }
    pub(super) fn authority(
        &self,
        manifest: &OriginalManifest,
    ) -> Result<NativeRowAuthority, ArenaError> {
        match (self, manifest) {
            (Self::Legacy(r), OriginalManifest::Legacy(_)) => native_row_authority_legacy(r),
            (Self::Submitted(r), OriginalManifest::Submitted(m)) => native_row_authority_v2(r, m),
            _ => Err(invalid("original PC row/manifest family mismatch")),
        }
    }
    pub(super) fn lineage(&self, authority: &NativeRowAuthority) -> HistoricalPcLineageRow {
        HistoricalPcLineageRow {
            family: authority.family,
            origin: authority.origin.clone(),
            rules_state_sha256: match self {
                Self::Legacy(r) => r.input.rules_anchor.rules_state_sha256.clone(),
                Self::Submitted(r) => r.input.rules_anchor.rules_state_sha256.clone(),
            },
        }
    }
}

/// Mixed-family ancestry audit consumes original canonical index seals. An
/// ancestor's common roster is never rewritten into a legacy index to hash it.
pub(super) fn validate_original_ancestry(
    manifest: &HistoricalCollectionMetadata,
    index: &HistoricalIndexMetadata,
    ancestors: &[Arc<VerifiedRoventDataset>],
) -> Result<(), ArenaError> {
    let metadata = ancestors
        .iter()
        .map(|a| (a.index_sha256(), &a.manifest, &a.index))
        .collect::<Vec<_>>();
    audit_original_ancestry(manifest, index, &metadata)
}
/// Pure parent-admission audit. It does not modify either closed codec or
/// rewrite a historical inherited-training declaration to make it pass.
pub(super) fn audit_original_ancestry(
    manifest: &HistoricalCollectionMetadata,
    index: &HistoricalIndexMetadata,
    ancestors: &[(
        &str,
        &HistoricalCollectionMetadata,
        &HistoricalIndexMetadata,
    )],
) -> Result<(), ArenaError> {
    let actual = ancestors
        .iter()
        .map(|(sha, _, _)| (*sha).to_owned())
        .collect::<BTreeSet<_>>();
    check(
        actual.len() == ancestors.len()
            && actual.iter().cloned().collect::<Vec<_>>() == manifest.ancestor_index_sha256,
        "missing/extra/duplicate mixed-family original PC ancestor index seals",
    )?;
    for (_, ancestor, _) in ancestors {
        check(
            ancestor
                .ancestor_index_sha256
                .iter()
                .all(|sha| actual.contains(sha)),
            "mixed-family original PC ancestry omits a transitive ancestor index",
        )?;
    }
    let mut known_inputs = BTreeSet::new();
    let mut known_origins = BTreeSet::new();
    let mut namespaces = BTreeMap::new();
    let mut origin_rows: BTreeMap<&str, &RoventCorpusOrigin> = BTreeMap::new();
    for (_, _, a) in ancestors {
        for o in &a.origins {
            known_inputs.insert(o.input_sha256.as_str());
            known_origins.insert(o.origin_sha256.as_str());
            if let Some(previous) = origin_rows.insert(o.origin_sha256.as_str(), o) {
                check(
                    previous == o,
                    "same original inherited origin has conflicting PC metadata",
                )?;
            }
        }
    }
    for i in ancestors
        .iter()
        .map(|(_, _, i)| *i)
        .chain(std::iter::once(index))
    {
        for o in &i.origins {
            for (kind, value) in [
                ("game", &o.game_id),
                ("opening", &o.opening_id),
                ("genealogy", &o.line_genealogy_id),
                ("state", &o.transposition_sha256),
                ("input", &o.input_sha256),
                ("origin", &o.origin_sha256),
            ] {
                if let Some(previous) = namespaces.insert((kind, value.as_str()), o.split) {
                    check(
                        previous == o.split,
                        "mixed-family original PC ancestor/holdout leakage",
                    )?;
                }
            }
        }
    }
    check(
        manifest
            .inherited_input_sha256
            .iter()
            .all(|s| known_inputs.contains(s.as_str()))
            && manifest
                .inherited_origin_sha256
                .iter()
                .all(|s| known_origins.contains(s.as_str()))
            && index.origins.iter().all(|o| {
                o.ancestor_origin_sha256
                    .iter()
                    .all(|s| known_origins.contains(s.as_str()))
            }),
        "unresolved mixed-family original inherited input/origin",
    )?;
    for declaration in ancestors
        .iter()
        .map(|(_, m, _)| *m)
        .chain(std::iter::once(manifest))
    {
        let mut inputs = BTreeSet::new();
        for origin in &declaration.inherited_origin_sha256 {
            let row = origin_rows
                .get(origin.as_str())
                .ok_or_else(|| invalid("declared original inherited training origin is missing"))?;
            check(
                row.split == GenerationSplitV3::Train,
                "declared original inherited training origin belongs to validation/holdout",
            )?;
            inputs.insert(row.input_sha256.clone());
        }
        check(
            inputs == declaration.inherited_input_sha256.iter().cloned().collect(),
            "declared original inherited train origin/input sets do not exactly join",
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn manifest() -> RoventCorpusManifest {
        RoventCorpusManifest {
            schema: rz_experiments::rovent_corpus::CORPUS_DOMAIN.into(),
            corpus_id: "corpus".into(),
            source_commit: "a".repeat(40),
            source_sha256: "a".repeat(64),
            runtime_sha256: "b".repeat(64),
            model_semantics_sha256: "c".repeat(64),
            model_epoch: 1,
            context_epoch: 1,
            recipe_sha256: "d".repeat(64),
            stage_id: "stage".into(),
            round: 0,
            round_id: "round".into(),
            student: "student".into(),
            learner: "learner".into(),
            one_learner: true,
            inherited_input_sha256: vec![],
            inherited_origin_sha256: vec![],
            teachers: BTreeMap::new(),
            opening_splits: BTreeMap::from([("opening".into(), GenerationSplitV3::Train)]),
            ancestor_index_sha256: vec![],
        }
    }
    fn origin(split: GenerationSplitV3) -> RoventCorpusOrigin {
        RoventCorpusOrigin {
            record_sha256: "1".repeat(64),
            input_sha256: "2".repeat(64),
            origin_sha256: "3".repeat(64),
            split,
            game_id: "game".into(),
            opening_id: "opening".into(),
            line_genealogy_id: "line".into(),
            transposition_sha256: "4".repeat(64),
            actor_id: "student".into(),
            teacher_actor_id: None,
            ancestor_origin_sha256: vec![],
        }
    }
    fn empty_index(sha: &str) -> HistoricalIndexMetadata {
        HistoricalIndexMetadata {
            record_count: 0,
            chunks: vec![],
            origins: vec![],
            canonical_sha256: sha.into(),
        }
    }
    #[test]
    fn original_legacy_metadata_bytes_and_digest_remain_in_legacy_domain() {
        let original = manifest();
        let raw = serde_json::to_vec(&original).unwrap();
        let parsed = OriginalManifest::decode(&raw, RoventPcRecordFamily::LegacyRecordV1).unwrap();
        assert_eq!(parsed.sha256().unwrap(), original.sha256().unwrap());
        assert_eq!(
            serde_json::to_vec(&parsed.json().unwrap()).unwrap(),
            serde_json::to_vec(&serde_json::to_value(&original).unwrap()).unwrap()
        );
        assert!(OriginalManifest::decode(&raw, RoventPcRecordFamily::SubmittedInputV1).is_err());
    }
    #[test]
    fn mixed_family_index_seals_and_record_domains_are_not_projected_into_legacy() {
        let original = manifest();
        let mut new = serde_json::to_value(&original).unwrap();
        new["schema"] = json!(rz_experiments::rovent_submitted_input::ROVENT_CORPUS_V2_DOMAIN);
        let new = OriginalManifest::decode(
            &serde_json::to_vec(&new).unwrap(),
            RoventPcRecordFamily::SubmittedInputV1,
        )
        .unwrap();
        let old = OriginalManifest::Legacy(original.clone());
        assert_ne!(old.sha256().unwrap(), new.sha256().unwrap());
        let old_index = OriginalIndex::Legacy(RoventCorpusIndex {
            schema: rz_experiments::rovent_corpus::INDEX_DOMAIN.into(),
            manifest_sha256: old.sha256().unwrap(),
            record_count: 0,
            chunks: vec![],
            origins: vec![],
        });
        let new_index = OriginalIndex::Submitted(RoventCorpusIndexV2 {
            schema: rz_experiments::rovent_submitted_input::ROVENT_INDEX_V2_DOMAIN.into(),
            manifest_sha256: new.sha256().unwrap(),
            record_count: 0,
            chunks: vec![],
            origins: vec![],
        });
        assert_ne!(old_index.sha256().unwrap(), new_index.sha256().unwrap());
        assert_ne!(
            RoventPcRecordFamily::LegacyRecordV1.record_domain(),
            RoventPcRecordFamily::SubmittedInputV1.record_domain()
        );
        let old_meta = old.metadata();
        let new_meta = new.metadata();
        let old_i = old_index.metadata().unwrap();
        let new_i = new_index.metadata().unwrap();
        let mut child = HistoricalCollectionMetadata::legacy(&original);
        child.ancestor_index_sha256 = vec![
            old_i.canonical_sha256.clone(),
            new_i.canonical_sha256.clone(),
        ];
        child.ancestor_index_sha256.sort();
        audit_original_ancestry(
            &child,
            &empty_index(&"e".repeat(64)),
            &[
                (&old_i.canonical_sha256, &old_meta, &old_i),
                (&new_i.canonical_sha256, &new_meta, &new_i),
            ],
        )
        .unwrap();
    }
    #[test]
    fn missing_transitive_original_index_is_rejected_in_parent_admission() {
        let mut ancestor = HistoricalCollectionMetadata::legacy(&manifest());
        ancestor.ancestor_index_sha256 = vec!["a".repeat(64)];
        let mut child = HistoricalCollectionMetadata::legacy(&manifest());
        child.ancestor_index_sha256 = vec!["b".repeat(64)];
        assert!(
            audit_original_ancestry(
                &child,
                &empty_index(&"c".repeat(64)),
                &[(&"b".repeat(64), &ancestor, &empty_index(&"b".repeat(64)))]
            )
            .is_err()
        );
    }
    #[test]
    fn inherited_training_requires_exact_train_origin_to_input_projection() {
        let ancestor = HistoricalCollectionMetadata::legacy(&manifest());
        let mut index = empty_index(&"a".repeat(64));
        index.origins = vec![origin(GenerationSplitV3::Train)];
        index.record_count = 1;
        let mut child = HistoricalCollectionMetadata::legacy(&manifest());
        child.ancestor_index_sha256 = vec!["a".repeat(64)];
        child.inherited_origin_sha256 = vec!["3".repeat(64)];
        child.inherited_input_sha256 = vec!["2".repeat(64)];
        let child_index = empty_index(&"b".repeat(64));
        audit_original_ancestry(
            &child,
            &child_index,
            &[(&"a".repeat(64), &ancestor, &index)],
        )
        .unwrap();
        child.inherited_input_sha256 = vec!["9".repeat(64)];
        assert!(
            audit_original_ancestry(
                &child,
                &child_index,
                &[(&"a".repeat(64), &ancestor, &index)]
            )
            .is_err()
        );
        child.inherited_input_sha256 = vec!["2".repeat(64)];
        index.origins[0].split = GenerationSplitV3::Holdout;
        assert!(
            audit_original_ancestry(
                &child,
                &child_index,
                &[(&"a".repeat(64), &ancestor, &index)]
            )
            .is_err()
        );
    }
}
