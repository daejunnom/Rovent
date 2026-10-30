//! 실제 root별 R observer와 원 append writer의 custody를 연결한다.
//! 원 caps는 publication/FD 검증 전까지 보존하며 오류를 성공 ACK로 바꾸지 않는다.
use super::*;
use crate::rovent_training_process::recovery::{
    RoventHistoricalRAuxiliaryDatasetRegistration, VerifiedRoventRAuxiliaryDataset,
};
use rz_experiments::rovent_r_readout::RoventRReadoutManifest;
use rz_search::cpu_value::nnuec::v2::capture::NnuecCaptureLimits;
use rz_search::pals::engine::adaptive_port::AdaptivePortQuestion;
use rz_uci::pals_adaptive::{AdaptiveCpuSemanticObserverBox, AdaptiveCpuSemanticRetirement};
use rz_uci::rovent_bootstrap::nnuec_source::LoadedRoventNnuecSource;
use std::sync::atomic::AtomicU64;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRCollectionAncestor {
    pub registration: RoventHistoricalRAuxiliaryDatasetRegistration,
    pub parent_indices: Vec<u16>,
    pub pc_parent_indices: Vec<u16>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRCollectionPlan {
    pub manifest: RoventPublicationFile,
    pub recipe: RoventPublicationFile,
    pub maximum_captures: u16,
    pub maximum_witnesses: u16,
    pub maximum_readouts: u16,
    pub every_nth_score: u32,
    pub maximum_input_bytes: u32,
    pub ancestor_datasets: Vec<RoventRCollectionAncestor>,
    pub current_ancestor_indices: Vec<u16>,
}
pub(super) struct RStream {
    manifest: RoventRReadoutManifest,
    recipe: Value,
    limits: RoventRObserverLimits,
    sequence: Arc<AtomicU64>,
    pc: Vec<Arc<VerifiedRoventDataset>>,
    ancestors: Vec<Arc<VerifiedRoventRAuxiliaryDataset>>,
    active: Option<RootRStream>,
    closed: Vec<RootRStream>,
    unmatched: Vec<AdaptiveCpuSemanticRetirement<AdaptivePortQuestion>>,
}
struct RootRStream {
    admission: RoventCollectionRootAdmission,
    source: Arc<LoadedRoventNnuecSource>,
    path: PathBuf,
    directory: Dir,
    receiver: Option<RoventRCollectionReceiver>,
    closure: Option<RoventRCollectionClosure>,
    rejected: Option<RoventRCollectionCloseRejected>,
    recovering: Option<AdaptiveCpuSemanticRetirement<AdaptivePortQuestion>>,
    collector: RoventRReadoutCollector,
    captured: Vec<CapturedRoventRRetirement>,
    writer: Option<RoventRCorpusWriter>,
    finished: Option<FinishedRoventRCorpus>,
    packet: Option<RoventRPublicationPacket>,
    publication: Option<RoventRCorpusPublication>,
    dataset: Option<Arc<VerifiedRoventRAuxiliaryDataset>>,
    archive_acknowledged: bool,
    semantic: Option<RoventRSemanticObservations>,
    semantic_file: Option<RoventPublicationFile>,
    unsubmitted: Option<RoventRUnsubmittedObservations>,
    unsubmitted_file: Option<RoventPublicationFile>,
}
fn select_r(
    indices: &[u16],
    owners: &[Arc<VerifiedRoventRAuxiliaryDataset>],
) -> Result<Vec<Arc<VerifiedRoventRAuxiliaryDataset>>, ArenaError> {
    let mut seen = BTreeSet::new();
    indices
        .iter()
        .map(|index| {
            check(seen.insert(*index), "duplicate R ancestor occurrence")?;
            owners
                .get(usize::from(*index))
                .cloned()
                .ok_or_else(|| invalid("R ancestor is not yet verified"))
        })
        .collect()
}
impl RStream {
    pub(super) fn ancestor_source_descriptors(
        &self,
    ) -> Result<Vec<RoventUtilityCohortAncestorSource>, ArenaError> {
        let mut descriptors = Vec::new();
        for owner in &self.pc {
            descriptors.push(RoventUtilityCohortAncestorSource {
                kind: match owner.family() {
                    crate::rovent_training_process::RoventPcRecordFamily::LegacyRecordV1 => {
                        "pc_legacy"
                    }
                    crate::rovent_training_process::RoventPcRecordFamily::SubmittedInputV1 => {
                        "pc_submitted_input_v1"
                    }
                }
                .into(),
                publication_sha256: owner.registration().publication.sha256.clone(),
                metadata: json!({"manifest_file_sha256":owner.manifest_file_sha256(),
                    "index_file_sha256":owner.index_file_sha256(),
                    "canonical_manifest_sha256":owner.manifest_sha256(),
                    "canonical_index_sha256":owner.index_sha256()}),
            });
        }
        for owner in &self.ancestors {
            let observation = owner.observation();
            descriptors.push(RoventUtilityCohortAncestorSource {
                kind: "r_auxiliary".into(),
                publication_sha256: owner.publication_pin().sha256.clone(),
                metadata: json!({"manifest_file_sha256":owner.manifest_file_sha256(),
                    "index_file_sha256":owner.index_file_sha256(),
                    "canonical_manifest_sha256":observation["canonical_manifest_sha256"],
                    "canonical_index_sha256":observation["canonical_index_sha256"]}),
            });
        }
        for descriptor in &descriptors {
            descriptor.validate()?;
        }
        Ok(descriptors)
    }
    pub(super) fn prepare(
        directory: &Dir,
        plan: RoventRCollectionPlan,
        caller: &RoventCollectionLaunchPlan,
        pc: Vec<Arc<VerifiedRoventDataset>>,
    ) -> Result<Self, ArenaError> {
        check(
            plan.ancestor_datasets.len() <= 256 && plan.current_ancestor_indices.len() <= 256,
            "bounded R ancestry",
        )?;
        let manifest: RoventRReadoutManifest = rz_experiments::decode_json(
            std::str::from_utf8(&read_pin(directory, &plan.manifest, 16 << 20)?)
                .map_err(|_| invalid("R manifest UTF8"))?,
        )?;
        manifest.validate()?;
        let recipe: Value = rz_experiments::decode_json(
            std::str::from_utf8(&read_pin(directory, &plan.recipe, 16 << 20)?)
                .map_err(|_| invalid("R recipe UTF8"))?,
        )?;
        let capture = NnuecCaptureLimits::new(
            plan.maximum_readouts,
            plan.every_nth_score,
            plan.maximum_input_bytes as usize,
        )
        .map_err(|error| invalid(format!("{error:?}")))?;
        let limits = RoventRObserverLimits::new(
            plan.maximum_captures.into(),
            plan.maximum_witnesses.into(),
            capture,
        )?;
        let mut ancestors = Vec::new();
        let budget = historical_budget(caller)?;
        for declaration in &plan.ancestor_datasets {
            let parents = select_r(&declaration.parent_indices, &ancestors)?;
            let pc_parents = select_ancestors(&declaration.pc_parent_indices, &pc)?;
            ancestors.push(Arc::new(VerifiedRoventRAuxiliaryDataset::open(
                declaration.registration.clone(),
                &recipe,
                pc_parents,
                parents,
                &budget,
            )?));
        }
        let ancestors = select_r(&plan.current_ancestor_indices, &ancestors)?;
        Ok(Self {
            manifest,
            recipe,
            limits,
            sequence: Arc::new(AtomicU64::new(1)),
            pc,
            ancestors,
            active: None,
            closed: Vec::new(),
            unmatched: Vec::new(),
        })
    }
    pub(super) fn admit(
        &mut self,
        admission: RoventCollectionRootAdmission,
        source: Arc<LoadedRoventNnuecSource>,
        caller: &RoventCollectionLaunchPlan,
        output: &Dir,
    ) -> Result<AdaptiveCpuSemanticObserverBox<AdaptivePortQuestion>, ArenaError> {
        check(
            self.active.is_none() && self.closed.len() < usize::from(caller.maximum_roots),
            "R root collection already active/capped",
        )?;
        let (observer, receiver) = RoventRCollectionObserver::new(
            admission.clone(),
            Arc::clone(&source),
            self.limits,
            Arc::clone(&self.sequence),
        )?;
        let name = admission.raw_sha256();
        let parent = output.open_dir_nofollow("r-roots").map_err(io)?;
        parent.create_dir(name).map_err(io)?;
        let directory = parent.open_dir_nofollow(name).map_err(io)?;
        let path = caller.output_directory.join("r-roots").join(name);
        self.active = Some(RootRStream {
            admission,
            source,
            path,
            directory,
            receiver: Some(receiver),
            closure: None,
            rejected: None,
            recovering: None,
            collector: RoventRReadoutCollector::new(self.manifest.clone())?,
            captured: vec![],
            writer: Some(RoventRCorpusWriter::new(&self.manifest)?),
            finished: None,
            packet: None,
            publication: None,
            dataset: None,
            archive_acknowledged: false,
            semantic: None,
            semantic_file: None,
            unsubmitted: None,
            unsubmitted_file: None,
        });
        Ok(Box::new(observer))
    }
    pub(super) fn close(
        &mut self,
        retirement: AdaptiveCpuSemanticRetirement<AdaptivePortQuestion>,
        caller: &RoventCollectionLaunchPlan,
    ) -> Result<(), ArenaError> {
        if self.active.is_none() {
            self.unmatched.push(retirement);
            return Err(invalid(
                "R receiver missing after actual fence; original retirement retained",
            ));
        }
        let root = self.active.as_mut().expect("original active R receiver");
        root.recovering = Some(retirement);
        let receiver = root
            .receiver
            .take()
            .ok_or_else(|| invalid("R receiver already recovered"))?;
        match receiver.finish_retaining(root.recovering.take().expect("stored original retirement"))
        {
            Ok(closure) => root.closure = Some(closure),
            Err(rejected) => {
                let error = invalid(rejected.error.to_string());
                root.rejected = Some(rejected);
                return Err(error);
            }
        }
        let closure = root.closure.as_ref().expect("retained original closure");
        let context = RoventRCollectionContext {
            game_id: caller.metadata.game_id.clone(),
            opening_id: caller.metadata.opening_id.clone(),
            line_genealogy_id: caller.metadata.line_genealogy_id.clone(),
            actor_id: caller.metadata.lineage.actor_id.clone(),
            teacher_actor_id: caller.metadata.lineage.teacher_actor_id.clone(),
            ancestor_origin_sha256: caller.metadata.lineage.ancestor_origin_sha256.clone(),
            pc_genealogy: None,
        };
        let until = Instant::now()
            .checked_add(Duration::from_millis(caller.publication_timeout_ms))
            .ok_or_else(|| invalid("R publication deadline overflow"))?;
        let control = RoventStartupControl {
            until,
            cleanup_until: until,
            cancellation: Arc::new(AtomicBool::new(false)),
        };
        let mut published_file_sha = None;
        for original in closure.captures() {
            // terminal 부재 자체는 미시작 근거가 아니다. 아래 원 manager
            // unsubmitted cap 전수 대조가 성공한 경우에만 별도로 ACK한다.
            let Some(retirement) = original.retirement() else {
                continue;
            };
            let captured = root.collector.capture_retirement(
                &root.admission,
                Arc::clone(&root.source),
                retirement,
                &context,
            )?;
            // 원 Rules/입력 계산만 재사용하며 과거 입력의 forward는 실행하지 않는다.
            for row in captured.rows() {
                verify_historical_r_readout(
                    row.record(),
                    &self.manifest,
                    root.admission.native_origin().original_input_lock(),
                    &root.source.model().base_checkpoint().config,
                )?;
            }
            root.captured.push(captured);
            let latest = root
                .captured
                .last()
                .expect("original captured retirement retained");
            let directory = &root.directory;
            let packet = &mut root.packet;
            root.writer
                .as_mut()
                .ok_or_else(|| invalid("R writer absent"))?
                .append_retirement(latest, &control, |value| {
                    put_packet(directory, packet, value, until)
                })?;
        }
        if !root.captured.is_empty() {
            let writer = root
                .writer
                .take()
                .ok_or_else(|| invalid("R writer missing before finish"))?;
            match writer.finish_retaining(|value| {
                put_packet(&root.directory, &mut root.packet, value, until)
            }) {
                Ok(finished) => root.finished = Some(finished),
                Err(rejected) => {
                    let rejected = *rejected;
                    root.writer = Some(rejected.writer);
                    return Err(rejected.error);
                }
            }
            let finished = root.finished.as_ref().expect("original finished R owner");
            let cancel = AtomicBool::new(false);
            root.publication = Some(finished.publish_atomic(&root.directory, until, &cancel)?);
            finished.publish_allocation_supplement(&root.directory, until, &cancel)?;
            let publication = root.publication.as_ref().expect("published R commit");
            let raw =
                serde_json::to_vec(publication).map_err(|error| invalid(error.to_string()))?;
            let source_pin = root
                .admission
                .native_origin()
                .original_input_lock()
                .manifest
                .engine
                .nnuec
                .clone()
                .ok_or_else(|| invalid("original R NNUEC pin missing"))?;
            let manifest = &self.manifest.collection;
            root.dataset = Some(Arc::new(VerifiedRoventRAuxiliaryDataset::open(
                RoventHistoricalRAuxiliaryDatasetRegistration {
                    root: root.path.clone(),
                    publication: RoventTrainingFilePin {
                        path: root.path.join("r-publication.json"),
                        sha256: hex(Digest(Sha256::digest(&raw).into())),
                        bytes: raw.len() as u64,
                    },
                    collection_source_commit: manifest.source_commit.clone(),
                    collection_source_sha256: manifest.source_sha256.clone(),
                    collection_runtime_library_sha256: manifest.runtime_sha256.clone(),
                    collection_model_semantics_sha256: manifest.model_semantics_sha256.clone(),
                    collection_model_epoch: manifest.model_epoch,
                    collection_context_epoch: manifest.context_epoch,
                    collection_recipe_sha256: manifest.recipe_sha256.clone(),
                    collection_stage_id: manifest.stage_id.clone(),
                    nnuec: source_pin,
                    nnuec_parameter_sha256: root
                        .source
                        .semantic_weights_identity()
                        .weights_sha256
                        .clone()
                        .ok_or_else(|| invalid("original R semantic parameter identity absent"))?,
                },
                &self.recipe,
                self.pc.clone(),
                self.ancestors.clone(),
                &historical_budget(caller)?,
            )?));
            let publication_sha = hex(Digest(Sha256::digest(&raw).into()));
            root.semantic = Some(capture_rovent_r_semantic_observations(
                closure,
                &root.captured,
                publication,
                &publication_sha,
            )?);
            let semantic = root
                .semantic
                .as_ref()
                .expect("original semantic observations retained");
            let bytes = serde_json::to_vec(semantic).map_err(|error| invalid(error.to_string()))?;
            root.semantic_file = Some(publish_semantic(
                &root.directory,
                &bytes,
                publication,
                &publication_sha,
                until,
            )?);
            published_file_sha = Some(publication_sha);
        }
        root.unsubmitted = capture_rovent_r_unsubmitted_observations(
            closure,
            &root.captured,
            published_file_sha.as_deref(),
        )?;
        if let Some(observation) = &root.unsubmitted {
            let bytes =
                serde_json::to_vec(observation).map_err(|error| invalid(error.to_string()))?;
            root.unsubmitted_file = Some(publish_sidecar(
                &root.directory,
                &bytes,
                "r-unsubmitted-observations",
                ROVENT_R_UNSUBMITTED_MAX_BYTES,
                until,
                |readback| {
                    RoventRUnsubmittedObservations::from_json_historical(
                        readback,
                        root.admission.raw_sha256(),
                        root.publication.as_ref(),
                        published_file_sha.as_deref(),
                    )?;
                    Ok(())
                },
            )?);
        }
        check(
            closure.original_retirement().failures().is_empty()
                && closure.original_retirement().unrecorded_witness_attempts() == 0,
            "R semantic delivery failed; original closure retained",
        )?;
        check(
            closure.witnesses().is_empty() || root.semantic_file.is_some(),
            "original R semantic observations require a durable supplement ACK",
        )?;
        root.archive_acknowledged = true;
        root.closure = None; // 원 reservation/Bound/View pins는 실제 archive/FD ACK 뒤 해제한다.
        root.captured.clear();
        let root = self.active.take().expect("acknowledged same R root");
        self.closed.push(root);
        Ok(())
    }
    pub(super) fn datasets(&self) -> Vec<Arc<VerifiedRoventRAuxiliaryDataset>> {
        self.closed
            .iter()
            .filter_map(|root| root.dataset.clone())
            .collect()
    }
    pub(super) fn sidecars(&self) -> Vec<RoventTrainingFilePin> {
        self.closed
            .iter()
            .flat_map(|root| {
                root.semantic_file
                    .iter()
                    .chain(root.unsubmitted_file.iter())
                    .map(move |pin| RoventTrainingFilePin {
                        path: root.path.join(&pin.path),
                        sha256: pin.sha256.clone(),
                        bytes: pin.bytes,
                    })
            })
            .collect()
    }
    pub(super) fn fully_closed(&self) -> bool {
        self.active.is_none()
            && self.unmatched.is_empty()
            && self.closed.iter().all(|root| root.archive_acknowledged)
    }
    pub(super) fn for_current_pc(
        &self,
        pc: Arc<VerifiedRoventDataset>,
        caller: &RoventCollectionLaunchPlan,
    ) -> Result<Vec<Arc<VerifiedRoventRAuxiliaryDataset>>, ArenaError> {
        let budget = historical_budget(caller)?;
        self.datasets()
            .iter()
            .map(|r| r.for_current_pc(Arc::clone(&pc), &self.recipe, &budget))
            .collect()
    }
}
pub(super) fn publish_semantic(
    directory: &Dir,
    raw: &[u8],
    publication: &RoventRCorpusPublication,
    publication_sha: &str,
    until: Instant,
) -> Result<RoventPublicationFile, ArenaError> {
    publish_sidecar(
        directory,
        raw,
        "r-semantic-observations",
        ROVENT_R_SEMANTIC_MAX_BYTES,
        until,
        |readback| {
            RoventRSemanticObservations::from_json_historical(
                readback,
                publication,
                publication_sha,
            )?;
            Ok(())
        },
    )
}
pub(super) fn publish_sidecar(
    directory: &Dir,
    raw: &[u8],
    stem: &str,
    maximum: usize,
    until: Instant,
    validate: impl FnOnce(&str) -> Result<(), ArenaError>,
) -> Result<RoventPublicationFile, ArenaError> {
    check(
        raw.len() <= maximum && Instant::now() < until,
        "R semantic file cap/deadline",
    )?;
    let pending = format!("{stem}.pending.json");
    let name = format!("{stem}.json");
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .write(true)
        .create_new(true)
        .follow(FollowSymlinks::No);
    let mut file = directory.open_with(&pending, &options).map_err(io)?;
    for bytes in raw.chunks(64 * 1024) {
        check(Instant::now() < until, "R semantic write deadline")?;
        file.write_all(bytes).map_err(io)?;
    }
    file.sync_all().map_err(io)?;
    let file_pin = RoventPublicationFile {
        path: pending.clone(),
        sha256: hex(Digest(Sha256::digest(raw).into())),
        bytes: raw.len() as u64,
    };
    let readback = read_pin(directory, &file_pin, maximum as u64)?;
    validate(std::str::from_utf8(&readback).map_err(|_| invalid("R sidecar file UTF8"))?)?;
    let root = crate::pals_collect_v3::training_process::directory_file(directory)?;
    root.sync_all().map_err(io)?;
    check(Instant::now() < until, "R semantic commit deadline")?;
    directory
        .hard_link(&pending, directory, &name)
        .map_err(io)?;
    directory.remove_file(&pending).map_err(io)?;
    root.sync_all().map_err(io)?;
    Ok(RoventPublicationFile {
        path: name,
        ..file_pin
    })
}
pub(super) fn put_packet(
    directory: &Dir,
    pending: &mut Option<RoventRPublicationPacket>,
    packet: RoventRPublicationPacket,
    until: Instant,
) -> Result<(), ArenaError> {
    check(pending.is_none(), "original failed R packet retained")?;
    *pending = Some(packet);
    let value = pending.as_ref().expect("stored original R packet");
    let pin = &value.declaration;
    pin.validate()?;
    check(
        valid_relative(&pin.path)
            && Instant::now() < until
            && pin.bytes == value.bytes.len() as u64
            && pin.sha256 == hex(Digest(Sha256::digest(&value.bytes).into())),
        "original R packet bytes/path/deadline",
    )?;
    let (parent, leaf) = pin
        .path
        .rsplit_once('/')
        .ok_or_else(|| invalid("R packet directory required"))?;
    check(
        !parent.contains('/'),
        "R packet parent must be a single registered directory",
    )?;
    match directory.create_dir(parent) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(io(error)),
    }
    let parent = directory.open_dir_nofollow(parent).map_err(io)?;
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .write(true)
        .create_new(true)
        .follow(FollowSymlinks::No);
    let mut file = parent.open_with(leaf, &options).map_err(io)?;
    for part in value.bytes.chunks(64 * 1024) {
        check(Instant::now() < until, "R packet write deadline")?;
        file.write_all(part).map_err(io)?;
    }
    file.sync_all().map_err(io)?;
    *pending = None;
    Ok(())
}
