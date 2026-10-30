//! 선택형 cohort는 trial마다 원 전체 bootstrap을 다시 실행한다.
//! 이전 root-local/R 게시 경로를 action 비용 구간으로 재사용하지 않는다.
use super::*;
use rz_experiments::rovent_r_readout::RoventRReadoutManifest;
use rz_experiments::rovent_utility_cohort::RoventUtilityComparisonCohort;
use rz_search::cpu_value::nnuec::v2::capture::NnuecCaptureLimits;
use rz_search::pals::engine::adaptive_port::AdaptivePortQuestion;
use rz_uci::pals_adaptive::{
    AdaptiveCpuSemanticRetirement, UtilityCohortReadyDisposition, UtilityCohortReadyObserver,
    UtilityCohortReadyRejection, UtilityCohortReadySnapshot, UtilityComparisonCohortOwner,
    UtilityTrialPhysicalRetirement,
};
use rz_uci::rovent_bootstrap::nnuec_source::LoadedRoventNnuecSource;
use std::sync::atomic::AtomicU64;

pub const ROVENT_UTILITY_COHORT_LAUNCH: &str = "rz-rovent-utility-cohort-launch/1";

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventUtilityCohortLaunchPlan {
    pub schema: String,
    /// 원 수집 계보/teacher source/R capture 한도와 별도 output root.
    pub collection_plan: RoventPublicationFile,
    pub cohort_profile: RoventPublicationFile,
    /// 각 새 Product의 명시 fresh cgroup/evidence 경로. 자동 이름 추정은 없다.
    pub trial_resource_plans: Vec<RoventPublicationFile>,
    pub process_epoch_base: u64,
    pub target_record: u16,
    pub position: RoventRulesPositionReplay,
    pub go_movetime_ms: u64,
}

pub struct RoventUtilityCohortLaunchReceipt {
    pub publication: RoventPublicationFile,
    pub durable_ack: RoventPublicationFile,
    pub output_directory: PathBuf,
    pub trials: u64,
    pub pairs: u64,
    pub historical: RoventHistoricalUtilityCohortEvidence,
    original: Arc<Mutex<CohortCaller>>,
}
impl RoventUtilityCohortLaunchReceipt {
    pub fn collection_observations(&self) -> Result<Value, ArenaError> {
        self.original
            .lock()
            .map_err(|_| invalid("cohort owner poisoned"))?
            .collection_observations()
    }
    pub fn resource_observations(&self) -> Result<Vec<Value>, ArenaError> {
        let owner = self
            .original
            .lock()
            .map_err(|_| invalid("cohort owner poisoned"))?;
        owner.resources()
    }
    pub fn finite_diagnostics(&self) -> Result<Vec<Value>, ArenaError> {
        self.original
            .lock()
            .map_err(|_| invalid("cohort owner poisoned"))?
            .finite_diagnostics()
    }
}
pub struct RoventUtilityCohortLaunchFailure {
    pub detail: String,
    original: Option<Arc<Mutex<CohortCaller>>>,
}
impl std::fmt::Debug for RoventUtilityCohortLaunchFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoventUtilityCohortLaunchFailure")
            .field("detail", &self.detail)
            .field("original_retained", &self.original.is_some())
            .finish()
    }
}
impl std::fmt::Display for RoventUtilityCohortLaunchFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.detail.fmt(f)
    }
}
impl std::error::Error for RoventUtilityCohortLaunchFailure {}
impl RoventUtilityCohortLaunchFailure {
    pub fn collection_observations(&self) -> Result<Option<Value>, ArenaError> {
        self.original
            .as_ref()
            .map(|owner| {
                owner
                    .lock()
                    .map_err(|_| invalid("cohort owner poisoned"))?
                    .collection_observations()
            })
            .transpose()
    }
    pub fn finite_diagnostics(&self) -> Result<Vec<Value>, ArenaError> {
        match &self.original {
            Some(owner) => owner
                .lock()
                .map_err(|_| invalid("cohort owner poisoned"))?
                .finite_diagnostics(),
            None => Ok(Vec::new()),
        }
    }
}

struct CohortCaller {
    profile: RoventUtilityComparisonCohort,
    plan: RoventCollectionLaunchPlan,
    manifest: RoventCorpusManifestV2,
    r_manifest: RoventRReadoutManifest,
    ancestry: r_stream::RStream,
    profile_bytes: Vec<u8>,
    limits: RoventRObserverLimits,
    sequence: Arc<AtomicU64>,
    directory: Dir,
    owner: Option<UtilityComparisonCohortOwner>,
    registration_root: Option<RoventCollectionRootAdmission>,
    writer: RoventUtilityCohortTrialWriter,
    trials: Vec<Arc<Mutex<TrialCaller>>>,
    source: Option<RoventUtilityCohortCollectionSource>,
    publication: Option<RoventUtilityCohortPublication>,
    durable_ack: Option<RoventUtilityCohortDurableAck>,
    payloads: BTreeMap<String, Vec<u8>>,
    pairs: Vec<rz_uci::pals_adaptive::UtilityCohortPair>,
    pair_rejected: Option<rz_uci::pals_adaptive::UtilityCohortPairRejection>,
    held_payload: Option<std::fs::File>,
    failure: Option<String>,
}
struct TrialCaller {
    resource: resources::CollectionResourceCustody,
    admission: Option<RoventCollectionRootAdmission>,
    source: Option<Arc<LoadedRoventNnuecSource>>,
    observer: Option<Box<dyn UtilityCohortReadyObserver<AdaptivePortQuestion>>>,
    receiver: Option<RoventRCollectionReceiver>,
    recovering: Option<AdaptiveCpuSemanticRetirement<AdaptivePortQuestion>>,
    closure: Option<RoventRCollectionClosure>,
    close_rejected: Option<RoventRCollectionCloseRejected>,
    collector: RoventRReadoutCollector,
    captured: Vec<CapturedRoventRRetirement>,
    materialized: Option<RoventRCohortMaterialization>,
    materialization_rejected: Option<Box<RoventRCohortMaterializationRejected>>,
    authorization_rejected: Option<Box<RoventRCohortAuthorizationRejected>>,
    ready: Option<RoventRCohortPublicationReady>,
    original_final: Option<Arc<UtilityTrialPhysicalRetirement>>,
    packet: Option<RoventRPublicationPacket>,
    r_publication: Option<RoventRCorpusPublication>,
    r_archive_io: Option<Value>,
    r_archive_error: Option<ArenaError>,
    directory: Dir,
    finished_resource_error: Option<String>,
    native_audit: Arc<Mutex<super::cohort_audit::NativeAudit>>,
    finite: Arc<Mutex<super::cohort_driver::FiniteTrialObservation>>,
}

struct ReadyObserver {
    shared: std::sync::Weak<Mutex<CohortCaller>>,
    loaded: Arc<LoadedRoventNnuecSource>,
}
impl UtilityCohortReadyObserver<AdaptivePortQuestion> for ReadyObserver {
    fn ready(
        &mut self,
        snapshot: UtilityCohortReadySnapshot<AdaptivePortQuestion>,
    ) -> UtilityCohortReadyDisposition<AdaptivePortQuestion> {
        let Some(owner) = self.shared.upgrade() else {
            return UtilityCohortReadyDisposition::Rejected(UtilityCohortReadyRejection {
                snapshot,
                error: AdaptiveManagerError::Invalid("cohort original owner unavailable"),
            });
        };
        let mut state = match owner.try_lock() {
            Ok(state) => state,
            Err(_) => {
                return UtilityCohortReadyDisposition::Rejected(UtilityCohortReadyRejection {
                    snapshot,
                    error: AdaptiveManagerError::Invalid("cohort owner unavailable"),
                });
            }
        };
        if state.owner.is_none() {
            match UtilityComparisonCohortOwner::register(
                state.profile.clone(),
                &snapshot,
                Arc::clone(&self.loaded),
            ) {
                Ok(owner) => {
                    state.registration_root = Some(snapshot.admission().clone());
                    state.owner = Some(owner);
                }
                Err(error) => {
                    return UtilityCohortReadyDisposition::Rejected(UtilityCohortReadyRejection {
                        snapshot,
                        error,
                    });
                }
            }
        }
        let sequence = state.writer.original_trials().count();
        let owner = state.owner.as_mut().expect("registered actual cohort");
        let Some(recipient) = owner.planned_recipients().get(sequence).copied() else {
            return UtilityCohortReadyDisposition::ContinuePrelude;
        };
        if snapshot.selected_offer_index(recipient).is_none() {
            return UtilityCohortReadyDisposition::ContinuePrelude;
        }
        match owner.admit_trial(snapshot, recipient, Arc::clone(&self.loaded)) {
            Ok(trial) => UtilityCohortReadyDisposition::Trial(trial),
            Err(rejected) => UtilityCohortReadyDisposition::Rejected(UtilityCohortReadyRejection {
                snapshot: rejected.snapshot,
                error: rejected.error,
            }),
        }
    }
}

impl CohortCaller {
    fn collection_observations(&self) -> Result<Value, ArenaError> {
        let r = self
            .trials
            .iter()
            .map(|trial| {
                trial
                    .lock()
                    .map(|trial| trial.r_archive_io.clone())
                    .map_err(|_| invalid("cohort R collection observation poisoned"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(
            json!({"r_archives":r,"checkpoint_acknowledgements":self.writer.acknowledgement_observations().collect::<Vec<_>>(),
            "final_publication":self.durable_ack.as_ref().map(|ack|&ack.collection_io)}),
        )
    }
    fn finite_diagnostics(&self) -> Result<Vec<Value>, ArenaError> {
        self.trials.iter().map(|trial| {
            let trial=trial.lock().map_err(|_|invalid("cohort trial diagnostics poisoned"))?;
            let original=trial.finite.lock().map_err(|_|invalid("cohort finite diagnostics poisoned"))?;
            Ok(json!({"protocol_utf8_prefix":original.protocol_utf8_prefix(),
                "diagnostics_utf8_prefix":original.diagnostics_utf8_prefix(),
                "protocol_bytes":original.protocol_bytes(),"diagnostics_bytes":original.diagnostics_bytes(),
                "protocol_truncated":original.protocol_truncated(),"diagnostics_truncated":original.diagnostics_truncated(),
                "bestmove_count":original.bestmove_count,"maximum_wall_reached":original.maximum_wall_reached,
                "watchdog_joined":original.watchdog_joined,"engine_error_retained":original.engine_error.is_some(),
                "engine_panic_retained":original.engine_panic.is_some()}))
        }).collect()
    }
    fn resources(&self) -> Result<Vec<Value>, ArenaError> {
        self.trials
            .iter()
            .map(|trial| {
                trial
                    .lock()
                    .map(|trial| trial.resource.observation())
                    .map_err(|_| invalid("cohort resource owner poisoned"))
            })
            .collect()
    }
    fn has_next_trial(&self) -> bool {
        self.owner.as_ref().is_none_or(|owner| {
            self.writer.original_trials().count() < owner.planned_recipients().len()
        })
    }
}

fn trial_hook(
    trial: &Arc<Mutex<TrialCaller>>,
    action: impl FnOnce(&mut TrialCaller) -> Result<(), ArenaError>,
) -> Result<(), AdaptiveManagerError> {
    let mut trial = trial
        .lock()
        .map_err(|_| AdaptiveManagerError::Invalid("cohort trial custody poisoned"))?;
    action(&mut trial).map_err(manager_error)
}

impl TrialCaller {
    fn close_memory(
        &mut self,
        retirement: AdaptiveCpuSemanticRetirement<AdaptivePortQuestion>,
        plan: &RoventCollectionLaunchPlan,
        manifest: &RoventRReadoutManifest,
    ) -> Result<(), ArenaError> {
        self.recovering = Some(retirement);
        let receiver = self
            .receiver
            .take()
            .ok_or_else(|| invalid("cohort original R receiver missing"))?;
        match receiver.finish_retaining(self.recovering.take().expect("stored actual retirement")) {
            Ok(closure) => self.closure = Some(closure),
            Err(rejected) => {
                let error = invalid(rejected.error.to_string());
                self.close_rejected = Some(rejected);
                return Err(error);
            }
        }
        let closure = self.closure.as_ref().expect("original R closure");
        let admission = self
            .admission
            .as_ref()
            .ok_or_else(|| invalid("cohort actual root absent"))?;
        let source = self
            .source
            .as_ref()
            .ok_or_else(|| invalid("cohort actual loaded source absent"))?;
        let context = RoventRCollectionContext {
            game_id: plan.metadata.game_id.clone(),
            opening_id: plan.metadata.opening_id.clone(),
            line_genealogy_id: plan.metadata.line_genealogy_id.clone(),
            actor_id: plan.metadata.lineage.actor_id.clone(),
            teacher_actor_id: plan.metadata.lineage.teacher_actor_id.clone(),
            ancestor_origin_sha256: plan.metadata.lineage.ancestor_origin_sha256.clone(),
            pc_genealogy: None,
        };
        for capture in closure.captures() {
            if let Some(terminal) = capture.retirement() {
                self.captured.push(self.collector.capture_retirement(
                    admission,
                    Arc::clone(source),
                    terminal,
                    &context,
                )?);
            }
        }
        let cleanup = admission.root_admission().cleanup_deadline();
        let until = admission
            .root_admission()
            .runtime_clock()
            .origin()
            .checked_add(Duration::from_nanos(cleanup.at.0))
            .ok_or_else(|| invalid("cohort cleanup overflow"))?;
        let control = RoventStartupControl {
            until,
            cleanup_until: until,
            cancellation: Arc::new(AtomicBool::new(false)),
        };
        match materialize_rovent_r_cohort_retaining(
            self.closure.take().expect("stored actual closure"),
            std::mem::take(&mut self.captured),
            manifest,
            &control,
            ROVENT_R_COHORT_SERIALIZED_MAX_BYTES,
        ) {
            Ok(materialized) => self.materialized = Some(materialized),
            Err(rejected) => {
                let error = invalid(rejected.error.to_string());
                self.materialization_rejected = Some(rejected);
                return Err(error);
            }
        }
        Ok(())
    }
    fn authorize_final(
        &mut self,
        actual: &Arc<UtilityTrialPhysicalRetirement>,
    ) -> Result<(), ArenaError> {
        check(
            self.original_final.is_none(),
            "cohort original final attempt is already retained",
        )?;
        self.original_final = Some(Arc::clone(actual));
        let source = self
            .source
            .as_ref()
            .ok_or_else(|| invalid("cohort final has no actual loader"))?;
        let original = self
            .admission
            .as_ref()
            .ok_or_else(|| invalid("cohort final has no original root"))?;
        check(
            actual.matches_root_admission(original.root_admission()),
            "cohort final root identity differs",
        )?;
        let materialized = self
            .materialized
            .take()
            .ok_or_else(|| invalid("cohort R memory packing incomplete"))?;
        match materialized.authorize_after_retirement_retaining(actual, source) {
            Ok(ready) => self.ready = Some(ready),
            Err(rejected) => {
                let error = invalid(rejected.error.to_string());
                self.authorization_rejected = Some(rejected);
                return Err(error);
            }
        }
        Ok(())
    }
    fn publish_r(&mut self, timeout_ms: u64) -> Result<(), ArenaError> {
        check(
            self.r_archive_io.is_none(),
            "cohort original R archive attempt is already retained",
        )?;
        let root = self
            .admission
            .as_ref()
            .ok_or_else(|| invalid("cohort R original IO clock absent"))?
            .root_admission()
            .clone();
        let start = root
            .runtime_clock()
            .try_now()
            .map_err(|error| invalid(format!("{error:?}")))?
            .0;
        let result = self.publish_r_inner(timeout_ms);
        let end = root
            .runtime_clock()
            .try_now()
            .map_err(|error| invalid(format!("{error:?}")));
        if let Err(error) = result {
            self.r_archive_error = Some(error);
        }
        match end {
            Ok(end) => {
                self.r_archive_io = Some(json!({"clock_epoch":root.authority().epoch.0,
                    "started_tick_ns":start,"finished_tick_ns":end.0,
                    "elapsed_ns":end.0.checked_sub(start),
                    "scope":"post_final_r_archive_io_and_validation",
                    "file_byte_accounting":"unknown","error_retained":self.r_archive_error.is_some()}));
                check(end.0 >= start, "cohort R IO original clock reversed")?;
            }
            Err(error) => {
                self.r_archive_io = Some(json!({"clock_epoch":root.authority().epoch.0,
                    "started_tick_ns":start,"finished_tick_ns":null,"elapsed_ns":null,
                    "scope":"post_final_r_archive_io_and_validation",
                    "file_byte_accounting":"unknown","clock_error":error.to_string(),"error_retained":self.r_archive_error.is_some()}));
                return Err(error);
            }
        }
        match &self.r_archive_error {
            Some(error) => Err(invalid(error.to_string())),
            None => Ok(()),
        }
    }
    fn publish_r_inner(&mut self, timeout_ms: u64) -> Result<(), ArenaError> {
        let until = Instant::now()
            .checked_add(Duration::from_millis(timeout_ms))
            .ok_or_else(|| invalid("cohort R durable deadline overflow"))?;
        let cancel = AtomicBool::new(false);
        let ready = self
            .ready
            .as_ref()
            .ok_or_else(|| invalid("cohort original R final authorization absent"))?;
        for packet in ready.packets() {
            r_stream::put_packet(
                &self.directory,
                &mut self.packet,
                RoventRPublicationPacket {
                    declaration: packet.declaration.clone(),
                    bytes: packet.bytes.clone(),
                },
                until,
            )?;
        }
        if let Some(finished) = ready.finished() {
            let publication = finished.publish_atomic(&self.directory, until, &cancel)?;
            check(
                Some(&publication) == ready.publication()
                    && Some(
                        serde_json::to_vec(&publication)
                            .map_err(|error| invalid(error.to_string()))?
                            .as_slice(),
                    ) == ready.publication_bytes(),
                "cohort R publication differs from original planned bytes",
            )?;
            self.r_publication = Some(publication);
            finished.publish_allocation_supplement(&self.directory, until, &cancel)?;
        }
        let semantic = ready.semantic();
        if let (Some(publication), Some(bytes), Some(sha)) = (
            self.r_publication.as_ref(),
            semantic.semantic_bytes(),
            semantic.publication_sha256(),
        ) {
            r_stream::publish_semantic(&self.directory, bytes, publication, sha, until)?;
        }
        if let Some(bytes) = semantic.unsubmitted_bytes() {
            let root = self.admission.as_ref().expect("authorized same root");
            r_stream::publish_sidecar(
                &self.directory,
                bytes,
                "r-unsubmitted-observations",
                ROVENT_R_UNSUBMITTED_MAX_BYTES,
                until,
                |raw| {
                    RoventRUnsubmittedObservations::from_json_historical(
                        raw,
                        root.raw_sha256(),
                        self.r_publication.as_ref(),
                        semantic.publication_sha256(),
                    )?;
                    Ok(())
                },
            )?;
        }
        Ok(())
    }
}

/// 별도 opt-in plan만 받으며 기존 Collection launch/1 해석은 변경하지 않는다.
pub fn run_rovent_utility_cohort_from_cli(
    pin: &RoventPublicationFile,
    arguments: Vec<String>,
) -> Result<RoventUtilityCohortLaunchReceipt, RoventUtilityCohortLaunchFailure> {
    let prepared = (|| -> Result<_, ArenaError> {
        let input = Dir::open_ambient_dir(
            std::env::current_dir().map_err(io)?,
            cap_std::ambient_authority(),
        )
        .map_err(io)?;
        let raw = read_pin(&input, pin, 256 << 10)?;
        let value: Value = rz_experiments::decode_json(
            std::str::from_utf8(&raw).map_err(|_| invalid("cohort launch UTF8"))?,
        )?;
        check(value.is_object(), "cohort launch object required")?;
        let plan: RoventUtilityCohortLaunchPlan =
            serde_json::from_value(value).map_err(|error| invalid(error.to_string()))?;
        Ok((input, plan))
    })()
    .map_err(|error| RoventUtilityCohortLaunchFailure {
        detail: error.to_string(),
        original: None,
    })?;
    run_rovent_utility_cohort(arguments, prepared.1, &prepared.0)
}

pub fn run_rovent_utility_cohort(
    arguments: Vec<String>,
    request: RoventUtilityCohortLaunchPlan,
    input: &Dir,
) -> Result<RoventUtilityCohortLaunchReceipt, RoventUtilityCohortLaunchFailure> {
    let prepared = (|| -> Result<_, ArenaError> {
        check(
            request.schema == ROVENT_UTILITY_COHORT_LAUNCH
                && request.process_epoch_base > 0
                && request.target_record < 64
                && (1..=86_400_000).contains(&request.go_movetime_ms),
            "cohort explicit finite launch",
        )?;
        let collection_raw = read_pin(input, &request.collection_plan, 256 << 10)?;
        let plan: RoventCollectionLaunchPlan = rz_experiments::decode_json(
            std::str::from_utf8(&collection_raw)
                .map_err(|_| invalid("cohort collection plan UTF8"))?,
        )?;
        let profile_bytes = read_pin(input, &request.cohort_profile, 64 << 10)?;
        let profile = RoventUtilityComparisonCohort::from_json(
            std::str::from_utf8(&profile_bytes).map_err(|_| invalid("cohort profile UTF8"))?,
        )?;
        let profile_window = profile.total_window_ns()?;
        check(
            request.trial_resource_plans.len() == profile.input.requested_recipients.len()
                && request
                    .process_epoch_base
                    .checked_add(request.trial_resource_plans.len() as u64 - 1)
                    .is_some()
                && request
                    .go_movetime_ms
                    .checked_mul(1_000_000)
                    .is_some_and(|n| n >= profile_window - profile.input.cleanup_ns),
            "cohort trial resources/epochs/original go window",
        )?;
        let mut resource_pins = BTreeSet::new();
        let mut leaves = BTreeSet::new();
        let mut evidence_directories = BTreeSet::new();
        let mut resource_conditions = None;
        for pin in &request.trial_resource_plans {
            pin.validate()?;
            check(
                resource_pins.insert(&pin.path),
                "cohort trial resource plan must be distinct",
            )?;
            let bytes=read_pin(input,pin,64<<10)?;
            let resource:crate::rovent_resource_scope::launch::RoventNativeResourceLaunchPlan=
                rz_experiments::decode_json(std::str::from_utf8(&bytes).map_err(|_|invalid("cohort resource plan UTF8"))?)?;
            resource.validate()?;
            let conditions=json!({"delegation":resource.delegated_cgroup_path,"options":resource.options,"profile":resource.profile});
            check(resource_conditions.as_ref().is_none_or(|original|original==&conditions)
                &&leaves.insert((resource.delegated_cgroup_path.clone(),resource.leaf_name.clone()))
                &&evidence_directories.insert(resource.evidence_directory.clone()),
                "cohort trials require the same declared resource conditions and distinct fresh leaf/evidence paths")?;
            resource_conditions.get_or_insert(conditions);
        }
        let manifest: RoventCorpusManifestV2 = rz_experiments::decode_json(
            std::str::from_utf8(&read_pin(input, &plan.manifest, 16 << 20)?)
                .map_err(|_| invalid("cohort manifest UTF8"))?,
        )?;
        plan.validate(&manifest)?;
        check(
            generation_identifier(&plan.metadata.game_id)
                && generation_identifier(&plan.metadata.opening_id)
                && generation_identifier(&plan.metadata.line_genealogy_id)
                && manifest
                    .opening_splits
                    .contains_key(&plan.metadata.opening_id),
            "cohort original collection metadata/opening is not registered",
        )?;
        let r = plan
            .r_readout
            .as_ref()
            .ok_or_else(|| invalid("cohort requires original NNUEC R capture plan"))?;
        let r_manifest: RoventRReadoutManifest = rz_experiments::decode_json(
            std::str::from_utf8(&read_pin(input, &r.manifest, 16 << 20)?)
                .map_err(|_| invalid("cohort R manifest UTF8"))?,
        )?;
        r_manifest.validate()?;
        let limits = RoventRObserverLimits::new(
            r.maximum_captures.into(),
            r.maximum_witnesses.into(),
            NnuecCaptureLimits::new(
                r.maximum_readouts,
                r.every_nth_score,
                r.maximum_input_bytes as usize,
            )
            .map_err(|error| invalid(format!("{error:?}")))?,
        )?;
        let ancestry = r_stream::RStream::prepare(input, r.clone(), &plan, open_ancestors(&plan)?)?;
        request.position.restore()?;
        let parent = plan
            .output_directory
            .parent()
            .ok_or_else(|| invalid("cohort output parent"))?;
        let parent = Dir::open_ambient_dir(parent, cap_std::ambient_authority()).map_err(io)?;
        let leaf = plan
            .output_directory
            .file_name()
            .ok_or_else(|| invalid("cohort output leaf"))?;
        parent.create_dir(leaf).map_err(io)?;
        let directory = parent.open_dir_nofollow(leaf).map_err(io)?;
        let writer = RoventUtilityCohortTrialWriter::new(directory.try_clone().map_err(io)?);
        Ok(CohortCaller {
            profile,
            profile_bytes,
            plan,
            manifest,
            r_manifest,
            limits,
            ancestry,
            sequence: Arc::new(AtomicU64::new(1)),
            directory,
            owner: None,
            registration_root: None,
            writer,
            trials: Vec::new(),
            source: None,
            publication: None,
            durable_ack: None,
            payloads: BTreeMap::new(),
            pairs: Vec::new(),
            pair_rejected: None,
            held_payload: None,
            failure: None,
        })
    })()
    .map_err(|error| RoventUtilityCohortLaunchFailure {
        detail: error.to_string(),
        original: None,
    })?;
    let shared = Arc::new(Mutex::new(prepared));
    let result = (|| -> Result<_, ArenaError> {
        let position_command = position_command(&request.position)?;
        let go_command = format!("go movetime {}", request.go_movetime_ms);
        for (ordinal, resource_pin) in request.trial_resource_plans.iter().enumerate() {
            if !shared
                .lock()
                .map_err(|_| invalid("cohort owner poisoned"))?
                .has_next_trial()
            {
                break;
            }
            let trial = {
                let mut owner = shared
                    .lock()
                    .map_err(|_| invalid("cohort owner poisoned"))?;
                let name = format!("r-trial-{ordinal:06}");
                owner.directory.create_dir(&name).map_err(io)?;
                let directory = owner.directory.open_dir_nofollow(&name).map_err(io)?;
                let trial = Arc::new(Mutex::new(TrialCaller {
                    resource: resources::CollectionResourceCustody::prepare(input, resource_pin)?,
                    admission: None,
                    source: None,
                    observer: None,
                    receiver: None,
                    recovering: None,
                    closure: None,
                    close_rejected: None,
                    collector: RoventRReadoutCollector::new(owner.r_manifest.clone())?,
                    captured: Vec::new(),
                    materialized: None,
                    materialization_rejected: None,
                    authorization_rejected: None,
                    ready: None,
                    original_final: None,
                    packet: None,
                    r_publication: None,
                    r_archive_io: None,
                    r_archive_error: None,
                    directory,
                    finished_resource_error: None,
                    native_audit: Arc::new(Mutex::new(super::cohort_audit::NativeAudit::default())),
                    finite: Arc::new(Mutex::new(
                        super::cohort_driver::FiniteTrialObservation::default(),
                    )),
                }));
                owner.trials.push(Arc::clone(&trial));
                trial
            };
            run_trial(
                &shared,
                &trial,
                &request,
                ordinal,
                &arguments,
                &position_command,
                &go_command,
            )?;
        }
        let mut owner = shared
            .lock()
            .map_err(|_| invalid("cohort owner poisoned"))?;
        check(
            !owner.has_next_trial(),
            "cohort finite prelude ended without an available completed trial",
        )?;
        let (publication, durable_ack, historical) = owner.publish_final()?;
        check(
            owner.trials.iter().all(|trial| {
                trial
                    .lock()
                    .is_ok_and(|trial| trial.finished_resource_error.is_none())
            }),
            "cohort 자료는 게시했으나 엄격 자원 인수는 HOLD 상태입니다",
        )?;
        Ok(RoventUtilityCohortLaunchReceipt {
            publication,
            durable_ack,
            output_directory: owner.plan.output_directory.clone(),
            trials: owner.writer.original_trials().count() as u64,
            pairs: owner.pairs.len() as u64,
            historical,
            original: Arc::clone(&shared),
        })
    })();
    result.map_err(|error| {
        if let Ok(mut owner) = shared.lock() {
            owner.failure.get_or_insert_with(|| error.to_string());
        }
        RoventUtilityCohortLaunchFailure {
            detail: error.to_string(),
            original: Some(shared),
        }
    })
}

pub(super) fn position_command(replay: &RoventRulesPositionReplay) -> Result<String, ArenaError> {
    replay.restore()?;
    let mut command = match replay.origin {
        RoventReplayOrigin::StartPosition => "position startpos".to_string(),
        RoventReplayOrigin::Fen => format!("position fen {}", replay.start_fen),
    };
    if !replay.moves.is_empty() {
        command.push_str(" moves");
        for bits in &replay.moves {
            let mv = rz_contracts::pals::Move16::try_from_bits(*bits)
                .and_then(|m| m.decode())
                .map_err(|error| invalid(error.to_string()))?;
            let mv = BoardMove::try_from(mv).map_err(|error| invalid(error.to_string()))?;
            command.push(' ');
            command.push_str(&mv.to_string());
        }
    }
    check(
        command.len() <= 1 << 20 && !command.contains(['\r', '\n', '\0']),
        "cohort position command bound",
    )?;
    Ok(command)
}

// The assembly closure must return this trial's original prepared Product,
// Store and cleanup owners on startup rejection. It has one active startup;
// adding an error allocation here would introduce another failure path while
// the original typed custody and physical-drain obligations must be retained.
#[allow(clippy::result_large_err)]
fn run_trial(
    shared: &Arc<Mutex<CohortCaller>>,
    trial: &Arc<Mutex<TrialCaller>>,
    request: &RoventUtilityCohortLaunchPlan,
    ordinal: usize,
    arguments: &[String],
    position_command: &str,
    go_command: &str,
) -> Result<(), ArenaError> {
    let mut arguments = arguments.to_vec();
    let epoch = request
        .process_epoch_base
        .checked_add(ordinal as u64)
        .ok_or_else(|| invalid("cohort process epoch overflow"))?;
    let old: Vec<_> = arguments
        .iter()
        .filter(|value| value.starts_with("--process-epoch="))
        .collect();
    check(
        old.len() == 1
            && old[0]
                .strip_prefix("--process-epoch=")
                .and_then(|value| value.parse::<u64>().ok())
                == Some(request.process_epoch_base),
        "cohort original base process epoch",
    )?;
    arguments.retain(|value| !value.starts_with("--process-epoch="));
    arguments.push(format!("--process-epoch={epoch}"));
    let resource = Arc::clone(trial);
    let assemble = Arc::clone(shared);
    let assembled_trial = Arc::clone(trial);
    let (teachers, profile, manifest, plan, limits, sequence, position) = {
        let state = shared
            .lock()
            .map_err(|_| invalid("cohort owner poisoned"))?;
        (
            state.plan.teacher_sources.clone(),
            state.profile.clone(),
            state.r_manifest.clone(),
            state.plan.clone(),
            state.limits,
            Arc::clone(&state.sequence),
            request.position.restore()?.snapshot(),
        )
    };
    let expected_profile = profile.clone();
    let target_record = request.target_record;
    let resource_pins = request.trial_resource_plans.clone();
    let audit = Arc::clone(
        &trial
            .lock()
            .map_err(|_| invalid("cohort trial owner poisoned"))?
            .native_audit,
    );
    let observer = Box::new(super::cohort_audit::NativeAuditObserver(audit));
    let finite = Arc::clone(
        &trial
            .lock()
            .map_err(|_| invalid("cohort finite custody poisoned"))?
            .finite,
    );
    let command_position = position_command.to_string();
    let command_go = go_command.to_string();
    let wall = Duration::from_millis(request.go_movetime_ms)
        .checked_add(Duration::from_nanos(profile.input.cleanup_ns))
        .and_then(|n| n.checked_add(Duration::from_millis(plan.publication_timeout_ms)))
        .and_then(|n| n.checked_add(Duration::from_secs(5)))
        .ok_or_else(|| invalid("cohort finite driver wall overflow"))?;
    let result = rz_uci::rovent_bootstrap::launch::run_collection_with_driver(
        arguments,
        teachers,
        observer,
        move |lock| {
            if lock.manifest.engine.nnuec.is_none()
                || !matches!(
                    &lock.manifest.engine.researcher,
                    Some(rz_experiments::RoventResearcher::Own { .. })
                )
                || lock.manifest.engine.mode != rz_contracts::rovent::ExecutionMode::CpuPcr
                || lock.manifest.resources.canonical_sha256().ok().as_deref()
                    != Some(expected_profile.input.resource_plan_sha256.as_str())
                || lock.canonical_sha256 != expected_profile.input.input_lock_sha256
            {
                return Err(RoventLaunchError {
                    stage: "CohortSource",
                    detail: "cohort requires the original CPU P-C-R/NNUEC source and plan".into(),
                });
            }
            // profile는 원 lock SHA를 참조하는 별도 사전 고정 선언이다.
            // 그 raw SHA를 다시 같은 lock에 넣어 자기 hash 순환을 만들지 않는다.
            // profile FD/정확 raw SHA는 outer launch plan에서 이미 검사했다.
            for pin in &resource_pins {
                if !lock.manifest.provenance.iter().any(|artifact| {
                    artifact.path == pin.path
                        && artifact.sha256 == pin.sha256
                        && artifact.bytes == pin.bytes
                }) {
                    return Err(RoventLaunchError {
                        stage: "CohortSource",
                        detail: "cohort resource input lacks original lock provenance".into(),
                    });
                }
            }
            resource
                .lock()
                .map_err(|_| RoventLaunchError {
                    stage: "CohortResource",
                    detail: "cohort resource owner poisoned".into(),
                })?
                .resource
                .admit(lock)
                .map_err(|error| RoventLaunchError {
                    stage: "CohortResource",
                    detail: error.to_string(),
                })
        },
        move |prepared| {
            let RoventCollectionPreparedLaunch {
                native,
                clock,
                owners,
                identity,
                root_policy,
                ..
            } = prepared;
            let loaded = match &native {
                RoventCollectionPreparedNative::OwnNnuecPcr { loaded, .. } => {
                    Some(Arc::new((**loaded).clone()))
                }
                _ => None,
            };
            let admitted = Arc::clone(&assembled_trial);
            let closing = Arc::clone(&assembled_trial);
            let final_trial = Arc::clone(&assembled_trial);
            let ready_owner = Arc::clone(&assemble);
            let final_owner = Arc::clone(&assemble);
            let final_timeout = plan.publication_timeout_ms;
            let hooks = AdaptiveSessionCollectionHooks {
                root_admission: Some(Arc::new(move |manager| {
                    let cap = manager.collection_root_admission().cloned().ok_or(
                        AdaptiveManagerError::Invalid("cohort actual Collection admission absent"),
                    )?;
                    let source = loaded.as_ref().ok_or(AdaptiveManagerError::Invalid(
                        "cohort actual NNUEC source absent",
                    ))?;
                    let observer = {
                        let mut trial = admitted.lock().map_err(|_| {
                            AdaptiveManagerError::Invalid("cohort trial owner poisoned")
                        })?;
                        if trial.admission.is_some()
                            || !position.same_state(cap.root_admission().root_snapshot())
                            || !cap.is_first_product_root()
                        {
                            return Err(AdaptiveManagerError::Invalid(
                                "cohort root is repeated, differs from original Rules or has no fresh Product",
                            ));
                        }
                        trial.admission = Some(cap.clone());
                        trial.source = Some(Arc::clone(source));
                        let (observer, receiver) = RoventRCollectionObserver::new(
                            cap,
                            Arc::clone(source),
                            limits,
                            Arc::clone(&sequence),
                        )
                        .map_err(manager_error)?;
                        trial.receiver = Some(receiver);
                        observer
                    };
                    manager.set_cpu_semantic_observer(Box::new(observer))?;
                    let observer = Box::new(ReadyObserver {
                        shared: Arc::downgrade(&ready_owner),
                        loaded: Arc::clone(source),
                    });
                    match manager.set_utility_cohort_ready_observer(
                        target_record,
                        &profile,
                        observer,
                    ) {
                        Ok(()) => Ok(()),
                        Err((error, observer)) => {
                            if let Ok(mut trial) = admitted.lock() {
                                trial.observer = Some(observer);
                            }
                            Err(error)
                        }
                    }
                })),
                root_closure: Some(Arc::new(move |manager, _| {
                    trial_hook(&closing, |trial| {
                        check(
                            trial.observer.is_none()
                                && trial.receiver.is_some()
                                && trial.materialized.is_none()
                                && trial.materialization_rejected.is_none(),
                            "cohort original root closure is already retained",
                        )?;
                        // 원 manager의 cap을 꺼내기 전에 destination custody를
                        // 잠근다. 잠금 실패가 move-only retirement를 drop하지 않는다.
                        trial.observer = manager
                            .take_utility_cohort_ready_observer()
                            .map_err(|error| invalid(format!("{error:?}")))?;
                        let retirement = manager
                            .take_cpu_semantic_retirement()
                            .map_err(|error| invalid(format!("{error:?}")))?;
                        let retirement = retirement
                            .ok_or_else(|| invalid("cohort actual R semantic retirement absent"))?;
                        trial.close_memory(retirement, &plan, &manifest)
                    })?;
                    manager.materialize_utility_cohort_trial()?;
                    Ok(())
                })),
                utility_trial_retired: Some(Arc::new(move |actual| {
                    trial_hook(&final_trial, |trial| {
                        trial.authorize_final(actual)?;
                        trial.publish_r(final_timeout)
                    })?;
                    let mut state = final_owner.lock().map_err(|_| {
                        AdaptiveManagerError::Invalid("cohort durable owner poisoned")
                    })?;
                    let admission = final_trial
                        .lock()
                        .map_err(|_| {
                            AdaptiveManagerError::Invalid("cohort original trial poisoned")
                        })?
                        .admission
                        .clone()
                        .ok_or(AdaptiveManagerError::Invalid(
                            "cohort original root missing",
                        ))?;
                    let until = Instant::now()
                        .checked_add(Duration::from_millis(final_timeout))
                        .ok_or(AdaptiveManagerError::Invalid(
                            "cohort durable deadline overflow",
                        ))?;
                    let CohortCaller { owner, writer, .. } = &mut *state;
                    let owner = owner
                        .as_mut()
                        .ok_or(AdaptiveManagerError::Invalid("cohort actual owner missing"))?;
                    writer
                        .publish_trial_retaining(
                            actual,
                            owner,
                            &admission,
                            until,
                            &AtomicBool::new(false),
                        )
                        .map_err(manager_error)
                })),
                utility_route_completed: None,
                utility_intervention_trial_retired: None,
                external_teacher: None,
                cooperative: None,
            };
            match native {
                RoventCollectionPreparedNative::OwnNnuecPcr { prepared, loaded } => {
                    RoventNativeProductDriver::own_nnuec_pcr_with_collection_hooks(
                        *prepared,
                        *loaded,
                        clock,
                        owners,
                        identity,
                        root_policy,
                        hooks,
                    )
                }
                RoventCollectionPreparedNative::PcNnuec { prepared, loaded } => {
                    Err(reject_pc_nnuec_cohort(*prepared, *loaded))
                }
                // Pre-model source admission rejects these variants. Their
                // ordinary constructors still retain the original prepared
                // owner if a future configuration route violates that check.
                RoventCollectionPreparedNative::Pc(prepared) => {
                    RoventNativeProductDriver::pc_with_collection_hooks(
                        *prepared,
                        clock,
                        owners,
                        identity,
                        root_policy,
                        Default::default(),
                    )
                }
                RoventCollectionPreparedNative::OwnBoardPcr(prepared) => {
                    RoventNativeProductDriver::own_board_pcr_with_collection_hooks(
                        *prepared,
                        clock,
                        owners,
                        identity,
                        root_policy,
                        Default::default(),
                    )
                }
                RoventCollectionPreparedNative::ExternalPcr {
                    prepared,
                    external,
                    retirement_evidence,
                } => RoventNativeProductDriver::external_pcr(
                    *prepared,
                    *external,
                    *retirement_evidence,
                    clock,
                    owners,
                    identity,
                    root_policy,
                    Some(Default::default()),
                ),
            }
        },
        move |product, clock, owners, settings| {
            super::cohort_driver::run_finite_collection_trial(
                product,
                clock,
                owners,
                settings,
                command_position,
                command_go,
                wall,
                finite,
            )
        },
    );
    let mut trial = trial
        .lock()
        .map_err(|_| invalid("cohort trial final custody poisoned"))?;
    if let Err(error) = trial.resource.finish_live() {
        trial.finished_resource_error = Some(error.to_string());
    }
    result.map_err(|error| invalid(error.to_string()))?;
    trial
        .native_audit
        .lock()
        .map_err(|_| invalid("cohort native audit poisoned"))?
        .validate_closed()?;
    check(
        trial.original_final.is_some(),
        "cohort no actual trial final retirement",
    )
}

impl CohortCaller {
    fn publish_final(
        &mut self,
    ) -> Result<
        (
            RoventPublicationFile,
            RoventPublicationFile,
            RoventHistoricalUtilityCohortEvidence,
        ),
        ArenaError,
    > {
        let trials: Vec<_> = self.writer.original_trials().cloned().collect();
        check(
            !trials.is_empty()
                && trials.len() == self.trials.len()
                && self.writer.failure().is_none(),
            "cohort complete original trial/ACK coverage required",
        )?;
        let registration = self
            .registration_root
            .as_ref()
            .ok_or_else(|| invalid("cohort original registration root missing"))?;
        let loaded = self
            .trials
            .first()
            .expect("original trial0")
            .lock()
            .map_err(|_| invalid("cohort trial poisoned"))?
            .source
            .clone()
            .ok_or_else(|| invalid("cohort original source missing"))?;
        let owner = self
            .owner
            .as_ref()
            .ok_or_else(|| invalid("cohort original owner absent"))?;
        self.source = Some(RoventUtilityCohortCollectionSource::capture_original(
            owner,
            registration,
            &loaded,
            &self.manifest,
            &self.plan.metadata,
            self.ancestry.ancestor_source_descriptors()?,
            self.resources()?,
        )?);
        let final_root = self
            .trials
            .last()
            .expect("original last trial")
            .lock()
            .map_err(|_| invalid("cohort trial poisoned"))?
            .admission
            .clone()
            .ok_or_else(|| invalid("cohort original final root absent"))?;
        let clock = final_root.root_admission().runtime_clock();
        let started_tick_ns = clock
            .try_now()
            .map_err(|error| invalid(format!("{error:?}")))?
            .0;
        let until = Instant::now()
            .checked_add(Duration::from_millis(self.plan.publication_timeout_ms))
            .ok_or_else(|| invalid("cohort final publication deadline overflow"))?;
        let control = RoventStartupControl {
            until,
            cleanup_until: until,
            cancellation: Arc::new(AtomicBool::new(false)),
        };
        let (_, weights, _, provenance) = loaded
            .publication_source_bytes(&control)
            .map_err(|error| invalid(error.to_string()))?;
        let mut insert = |path: &str, raw: Vec<u8>| -> Result<(), ArenaError> {
            let total = self
                .payloads
                .values()
                .try_fold(raw.len(), |sum, bytes| sum.checked_add(bytes.len()))
                .ok_or_else(|| invalid("cohort original payload accounting overflow"))?;
            check(
                total as u64 <= ROVENT_UTILITY_COHORT_PAYLOAD_MAX
                    && !self.payloads.contains_key(path),
                "cohort bounded unique original payloads",
            )?;
            self.payloads.insert(path.into(), raw);
            Ok(())
        };
        insert("cohort-profile.json", self.profile_bytes.clone())?;
        insert(
            "frozen-context.json",
            serde_json::to_vec(
                &owner
                    .frozen_decision()
                    .map_err(|error| invalid(format!("{error:?}")))?,
            )
            .map_err(|error| invalid(error.to_string()))?,
        )?;
        insert(
            "source.json",
            serde_json::to_vec(self.source.as_ref().expect("original source projection"))
                .map_err(|error| invalid(error.to_string()))?,
        )?;
        insert(
            "input-lock.json",
            serde_json::to_vec(registration.native_origin().original_input_lock())
                .map_err(|error| invalid(error.to_string()))?,
        )?;
        insert("nnuec-weights.json", weights)?;
        insert("nnuec-provenance.json", provenance)?;
        for (index, trial) in trials.iter().enumerate() {
            insert(
                &format!("trials/{index:06}.json"),
                trial.original_bytes().to_vec(),
            )?;
            if let Some(materialized) = trial.materialized() {
                insert(
                    &format!("materializations/{index:06}.json"),
                    materialized.original_bytes().to_vec(),
                )?;
            }
        }
        for left in 0..trials.len() {
            for right in left + 1..trials.len() {
                match rz_uci::pals_adaptive::UtilityCohortPair::compare(
                    Arc::clone(&trials[left]),
                    Arc::clone(&trials[right]),
                ) {
                    Ok(pair) => self.pairs.push(pair),
                    Err(rejected) => {
                        let error = invalid(format!("{:?}", rejected.error));
                        self.pair_rejected = Some(rejected);
                        return Err(error);
                    }
                }
                let index = self.pairs.len() - 1;
                self.payloads.insert(
                    format!("pairs/{index:06}.json"),
                    self.pairs[index].original_bytes().to_vec(),
                );
            }
        }
        let pin = |path: &str| -> Result<RoventPublicationFile, ArenaError> {
            let raw = self
                .payloads
                .get(path)
                .ok_or_else(|| invalid("cohort actual payload missing"))?;
            Ok(RoventPublicationFile {
                path: path.into(),
                sha256: hex(Digest(Sha256::digest(raw).into())),
                bytes: raw.len() as u64,
            })
        };
        let pair_files = (0..self.pairs.len())
            .map(|index| pin(&format!("pairs/{index:06}.json")))
            .collect::<Result<Vec<_>, _>>()?;
        self.publication = Some(RoventUtilityCohortPublication {
            schema: ROVENT_UTILITY_COHORT_PUBLICATION.into(),
            cohort_id: hex(owner.cohort_id()),
            cost_scope: ROVENT_UTILITY_COHORT_COST_SCOPE.into(),
            cohort_profile: pin("cohort-profile.json")?,
            frozen_context: pin("frozen-context.json")?,
            source: pin("source.json")?,
            input_lock: pin("input-lock.json")?,
            nnuec_weights: pin("nnuec-weights.json")?,
            nnuec_provenance: pin("nnuec-provenance.json")?,
            materializations: self.writer.materialization_files().cloned().collect(),
            trials: self.writer.trial_files().cloned().collect(),
            pair_count: pair_files.len() as u64,
            pairs: pair_files,
            trial_count: trials.len() as u64,
            total_payload_bytes: self
                .payloads
                .values()
                .try_fold(0u64, |sum, bytes| sum.checked_add(bytes.len() as u64))
                .ok_or_else(|| invalid("cohort aggregate payload overflow"))?,
        });
        let publication = self
            .publication
            .as_ref()
            .expect("original publication metadata");
        publication.validate()?;
        let historical = verify_historical_utility_cohort(publication, &self.payloads);
        let mut written = 0u64;
        let mut directories = 0u64;
        for declaration in publication.payloads() {
            let raw = &self.payloads[&declaration.path];
            if declaration.path.starts_with("trials/")
                || declaration.path.starts_with("materializations/")
            {
                let readback = read_pin(
                    &self.directory,
                    declaration,
                    ROVENT_UTILITY_COHORT_MATERIALIZATION_MAX,
                )?;
                check(
                    readback == *raw,
                    "cohort original durable checkpoint changed",
                )?;
                let mut options = cap_std::fs::OpenOptions::new();
                options.read(true).follow(FollowSymlinks::No);
                self.directory
                    .open_with(&declaration.path, &options)
                    .map_err(io)?
                    .sync_all()
                    .map_err(io)?;
            } else {
                put_final_raw(
                    &self.directory,
                    &declaration.path,
                    raw,
                    &mut self.held_payload,
                    until,
                )?;
                written = written
                    .checked_add(declaration.bytes)
                    .ok_or_else(|| invalid("cohort collection write overflow"))?;
                directories += 1 + u64::from(declaration.path.contains('/'));
            }
        }
        let publication_raw = publication.to_json_bytes()?;
        let publication_file = put_final_raw(
            &self.directory,
            ROVENT_UTILITY_COHORT_PUBLICATION_FILE,
            &publication_raw,
            &mut self.held_payload,
            until,
        )?;
        written = written
            .checked_add(publication_file.bytes)
            .ok_or_else(|| invalid("cohort publication write overflow"))?;
        directories += 1;
        let finished_tick_ns = clock
            .try_now()
            .map_err(|error| invalid(format!("{error:?}")))?
            .0;
        self.durable_ack = Some(RoventUtilityCohortDurableAck {
            schema: ROVENT_UTILITY_COHORT_DURABLE_ACK.into(),
            cohort_id: publication.cohort_id.clone(),
            cost_scope: publication.cost_scope.clone(),
            publication: publication_file.clone(),
            trial_acknowledgements: self
                .writer
                .acknowledgement_observations()
                .cloned()
                .collect(),
            collection_io: RoventUtilityCohortCollectionIo {
                clock_epoch: final_root.authority().epoch.0,
                started_tick_ns,
                finished_tick_ns,
                elapsed_ns: finished_tick_ns
                    .checked_sub(started_tick_ns)
                    .ok_or_else(|| invalid("cohort final I/O clock reversed"))?,
                written_bytes: written,
                readback_bytes: publication.total_payload_bytes + publication_file.bytes,
                synced_files: publication.payloads().count() as u64 + 1,
                synced_directories: directories,
            },
        });
        let ack = self
            .durable_ack
            .as_ref()
            .expect("original durable ACK metadata");
        ack.validate_historical(publication, &publication_file)?;
        let ack_raw = serde_json::to_vec(ack).map_err(|error| invalid(error.to_string()))?;
        check(
            ack_raw.len() <= ROVENT_UTILITY_COHORT_METADATA_MAX,
            "cohort durable ACK metadata cap",
        )?;
        let ack_file = put_final_raw(
            &self.directory,
            ROVENT_UTILITY_COHORT_ACK_FILE,
            &ack_raw,
            &mut self.held_payload,
            until,
        )?;
        // put_final_raw already compared every original byte through its
        // retained FD. Decode that same buffer without another unaccounted
        // publication read. This never creates a current trial ACK.
        RoventUtilityCohortPublication::from_json_historical(
            std::str::from_utf8(&publication_raw)
                .map_err(|_| invalid("cohort publication readback UTF8"))?,
        )?;
        RoventUtilityCohortDurableAck::from_json_historical(
            std::str::from_utf8(&ack_raw).map_err(|_| invalid("cohort ACK readback UTF8"))?,
            publication,
            &publication_file,
        )?;
        ack.validate_original_trial_io(publication, &self.payloads)?;
        Ok((publication_file, ack_file, historical?))
    }
}

pub(super) fn put_final_raw(
    root: &Dir,
    path: &str,
    bytes: &[u8],
    retained: &mut Option<std::fs::File>,
    until: Instant,
) -> Result<RoventPublicationFile, ArenaError> {
    use std::io::{Seek, SeekFrom};
    check(
        valid_relative(path)
            && bytes.len() as u64 <= ROVENT_UTILITY_COHORT_MATERIALIZATION_MAX
            && retained.is_none(),
        "cohort final original payload path/cap/custody",
    )?;
    let (directory, name) = match path.rsplit_once('/') {
        Some((parent, name)) => {
            check(
                !parent.contains('/'),
                "cohort final payload one registered parent",
            )?;
            match root.create_dir(parent) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(io(error)),
            }
            (root.open_dir_nofollow(parent).map_err(io)?, name)
        }
        None => (root.try_clone().map_err(io)?, path),
    };
    let pending = format!("{name}.pending");
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .read(true)
        .write(true)
        .create_new(true)
        .follow(FollowSymlinks::No);
    *retained = Some(
        directory
            .open_with(&pending, &options)
            .map_err(io)?
            .into_std(),
    );
    let file = retained.as_mut().expect("original publication FD retained");
    for part in bytes.chunks(64 << 10) {
        check(Instant::now() < until, "cohort publication write expired")?;
        file.write_all(part).map_err(io)?;
    }
    file.sync_all().map_err(io)?;
    file.seek(SeekFrom::Start(0)).map_err(io)?;
    let mut buffer = [0u8; 8192];
    for part in bytes.chunks(buffer.len()) {
        check(
            Instant::now() < until,
            "cohort publication readback expired",
        )?;
        file.read_exact(&mut buffer[..part.len()]).map_err(io)?;
        check(
            buffer[..part.len()] == *part,
            "cohort original publication readback differs",
        )?;
    }
    check(
        file.read(&mut buffer[..1]).map_err(io)? == 0,
        "cohort publication grew",
    )?;
    directory
        .hard_link(&pending, &directory, name)
        .map_err(io)?;
    directory.remove_file(&pending).map_err(io)?;
    crate::pals_collect_v3::training_process::directory_file(&directory)?
        .sync_all()
        .map_err(io)?;
    if path.contains('/') {
        crate::pals_collect_v3::training_process::directory_file(root)?
            .sync_all()
            .map_err(io)?;
    }
    *retained = None;
    Ok(RoventPublicationFile {
        path: path.into(),
        sha256: hex(Digest(Sha256::digest(bytes).into())),
        bytes: bytes.len() as u64,
    })
}
