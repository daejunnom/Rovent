//! 명시 Collection 진입점의 실제 caller다. 원 native observer와 root cap을
//! 제품에 설치하고 독립 CPU_T를 outer callback에서만 구동한다. 공개 계획은
//! 입력 선언이며, 원 source/퇴역/수락 없이 게시 owner를 만들 수 없다.
use super::*;
use crate::rovent_training_process::{
    RoventTrainingFilePin,
    recovery::{
        RoventHistoricalDatasetRegistration, RoventHistoricalRoleBinding, RoventRecoveryReadBudget,
        VerifiedRoventDataset,
    },
};
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::Dir;
use rz_experiments::rovent_corpus_v2::RoventCorpusManifestV2;
use rz_search::external_cpu::ExternalGoConstraint;
use rz_search::pals::engine::pc_port::PcDivergenceWitness;
use rz_uci::pals_adaptive::{
    AdaptiveCollectionError, AdaptiveCollectionStage, AdaptiveManagerError,
    AdaptiveSessionCollectionHooks,
};
use rz_uci::rovent_bootstrap::launch::{
    RoventCollectionPreparedLaunch, RoventCollectionPreparedNative, RoventLaunchError,
    run_collection_with_resource_admission,
};
use rz_uci::rovent_bootstrap::pc_session::RoventPcCollectionHooks;
use rz_uci::rovent_bootstrap::product::RoventNativeProductDriver;
use rz_uci::rovent_bootstrap::{
    RoventCollectionPreloadSources, RoventStartupControl, RoventTeacherSourceSelection,
};
use serde::{Deserialize, Serialize};
use std::{
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex, atomic::AtomicBool},
    time::{Duration, Instant},
};

pub const ROVENT_COLLECTION_LAUNCH_PLAN: &str = "rz-rovent-collection-launch/1";
mod cohort;
mod cohort_audit;
mod cohort_driver;
mod cohort_prepare;
mod intervention;
mod r_stream;
mod resources;
mod utility_stream;
pub use cohort::{
    RoventUtilityCohortLaunchFailure, RoventUtilityCohortLaunchPlan,
    RoventUtilityCohortLaunchReceipt, run_rovent_utility_cohort,
    run_rovent_utility_cohort_from_cli,
};
pub use cohort_prepare::prepare_rovent_utility_cohort_pins_from_cli;
pub use intervention::{
    RoventUtilityInterventionLaunchFailure, RoventUtilityInterventionLaunchPlan,
    RoventUtilityInterventionLaunchReceipt, run_rovent_utility_intervention,
    run_rovent_utility_intervention_from_cli,
};
pub use r_stream::{RoventRCollectionAncestor, RoventRCollectionPlan};
pub const ROVENT_COLLECTION_USAGE: &str = "rz-rovent-collect --collection-plan=inputs/<plan.json> --collection-plan-sha256=<64 lowercase hex> --collection-plan-bytes=<positive bytes> <original Rovent source/config/FD arguments>";

/// Cohort profiles require actual own-R NNUEC authority. A PC-only source
/// remains a distinct rejected product, with both original owners retained
/// until the launcher's existing physical drain confirms their release.
fn reject_pc_nnuec_cohort(
    prepared: rz_uci::rovent_bootstrap::PreparedRoventNative<
        rz_search::pals::engine::pc_port::PcPortQuestion,
    >,
    loaded: rz_uci::rovent_bootstrap::nnuec_source::LoadedRoventNnuecSource,
) -> rz_uci::rovent_bootstrap::product::RoventProductStartupRejection {
    let sources = Arc::clone(prepared.sources());
    rz_uci::rovent_bootstrap::product::RoventProductStartupRejection {
        failure: rz_uci::search_driver::SearchSessionFailure {
            physical_completion: rz_uci::search_driver::DriverPhysicalCompletion::Unknown,
            code: "UtilityCohortUnsupportedPcNnuec",
            detail: "utility cohort requires the original own-R P-C-R NNUEC source; PC-only shared NNUEC is unsupported".into(),
        },
        owners: rz_uci::rovent_bootstrap::product::RoventProductStartupOwners::PcNnuec {
            prepared,
            loaded,
        },
        sources,
    }
}

pub enum RoventCollectionCliParse {
    Help,
    PrepareCohortPins {
        plan: RoventPublicationFile,
    },
    Cohort {
        plan: RoventPublicationFile,
        arguments: Vec<String>,
    },
    Intervention {
        plan: RoventPublicationFile,
        arguments: Vec<String>,
    },
    Run {
        plan: RoventPublicationFile,
        arguments: Vec<String>,
    },
}
/// 새 caller 인수만 분리한다. 원 UCI source/config/resource 인수의 의미와
/// 순서는 유지하며 같은 pin의 중복 또는 빠진 필드는 거부한다.
pub fn parse_rovent_collection_cli(
    arguments: Vec<String>,
) -> Result<RoventCollectionCliParse, ArenaError> {
    if arguments == ["--help"] {
        return Ok(RoventCollectionCliParse::Help);
    }
    if arguments
        .iter()
        .any(|argument| argument.starts_with("--utility-intervention-plan"))
    {
        return intervention::parse_intervention_cli(arguments);
    }
    if arguments
        .iter()
        .any(|argument| argument.starts_with("--prepare-utility-cohort-pins"))
    {
        return cohort_prepare::parse_preparation_cli(arguments);
    }
    if arguments
        .iter()
        .any(|argument| argument.starts_with("--utility-cohort-plan"))
    {
        return parse_cohort_cli(arguments);
    }
    let mut plan = None;
    let mut sha256 = None;
    let mut bytes = None;
    let mut original = Vec::new();
    for argument in arguments {
        if let Some(value) = argument.strip_prefix("--collection-plan=") {
            check(
                plan.replace(value.to_string()).is_none(),
                "duplicate Collection plan",
            )?;
        } else if let Some(value) = argument.strip_prefix("--collection-plan-sha256=") {
            check(
                sha256.replace(value.to_string()).is_none(),
                "duplicate Collection plan SHA",
            )?;
        } else if let Some(value) = argument.strip_prefix("--collection-plan-bytes=") {
            let value = value
                .parse::<u64>()
                .map_err(|_| invalid("Collection plan bytes"))?;
            check(
                bytes.replace(value).is_none(),
                "duplicate Collection plan bytes",
            )?;
        } else {
            original.push(argument);
        }
    }
    let plan = RoventPublicationFile {
        path: plan.ok_or_else(|| invalid("explicit Collection plan required"))?,
        sha256: sha256.ok_or_else(|| invalid("explicit Collection plan SHA required"))?,
        bytes: bytes.ok_or_else(|| invalid("explicit Collection plan bytes required"))?,
    };
    plan.validate()?;
    check(
        plan.path.starts_with("inputs/") && valid_relative(&plan.path) && plan.bytes <= 256 << 10,
        "Collection plan must be a bounded original private input",
    )?;
    Ok(RoventCollectionCliParse::Run {
        plan,
        arguments: original,
    })
}

fn parse_cohort_cli(arguments: Vec<String>) -> Result<RoventCollectionCliParse, ArenaError> {
    let mut path = None;
    let mut sha256 = None;
    let mut bytes = None;
    let mut original = Vec::new();
    for argument in arguments {
        if let Some(value) = argument.strip_prefix("--utility-cohort-plan=") {
            check(
                path.replace(value.to_string()).is_none(),
                "duplicate cohort plan",
            )?;
        } else if let Some(value) = argument.strip_prefix("--utility-cohort-plan-sha256=") {
            check(
                sha256.replace(value.to_string()).is_none(),
                "duplicate cohort plan SHA",
            )?;
        } else if let Some(value) = argument.strip_prefix("--utility-cohort-plan-bytes=") {
            check(
                bytes
                    .replace(
                        value
                            .parse::<u64>()
                            .map_err(|_| invalid("cohort plan bytes"))?,
                    )
                    .is_none(),
                "duplicate cohort plan bytes",
            )?;
        } else {
            check(
                !argument.starts_with("--collection-plan"),
                "Collection/cohort plan families cannot be mixed",
            )?;
            original.push(argument);
        }
    }
    let plan = RoventPublicationFile {
        path: path.ok_or_else(|| invalid("explicit cohort plan required"))?,
        sha256: sha256.ok_or_else(|| invalid("explicit cohort plan SHA required"))?,
        bytes: bytes.ok_or_else(|| invalid("explicit cohort plan bytes required"))?,
    };
    plan.validate()?;
    check(
        plan.path.starts_with("inputs/") && valid_relative(&plan.path) && plan.bytes <= 256 << 10,
        "bounded original cohort plan required",
    )?;
    Ok(RoventCollectionCliParse::Cohort {
        plan,
        arguments: original,
    })
}

pub fn run_rovent_collection_from_cli(
    plan: &RoventPublicationFile,
    arguments: Vec<String>,
) -> Result<RoventCollectionLaunchReceipt, RoventCollectionLaunchFailure> {
    let read = || -> Result<_, ArenaError> {
        let directory = Dir::open_ambient_dir(
            std::env::current_dir().map_err(io)?,
            cap_std::ambient_authority(),
        )
        .map_err(io)?;
        let raw = read_pin(&directory, plan, 256 << 10)?;
        let request = rz_experiments::decode_json(
            std::str::from_utf8(&raw).map_err(|_| invalid("Collection plan UTF8"))?,
        )?;
        Ok((directory, request))
    };
    let (directory, request) = read().map_err(|error| RoventCollectionLaunchFailure {
        detail: error.to_string(),
        original: None,
    })?;
    run_rovent_collection(arguments, request, &directory)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventCollectionTeacherSlot {
    pub actor_id: String,
    pub source_index: usize,
    pub collection_owner: u32,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoventCollectionAncestorFamily {
    LegacyRaw2,
    SubmittedInputV1,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventCollectionAncestor {
    pub family: RoventCollectionAncestorFamily,
    pub registration: RoventHistoricalDatasetRegistration,
    /// 앞에서 실제 검증한 owner의 인덱스다. 미래 owner를 참조할 수 없다.
    pub parent_indices: Vec<u16>,
}

/// 원 source 선택과 유한 caller 예산이다. 이 선언으로 실행/target 권한을
/// 만들지 않으며 실제 loader, RootAdmission, physical receipt를 추가로 요구한다.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventCollectionLaunchPlan {
    pub schema: String,
    pub manifest: RoventPublicationFile,
    /// 원 input lock provenance로 봉인한 OS 자원 계획이다.
    pub resource_plan: RoventPublicationFile,
    /// own NNUEC에만 선택하는 별도 role-less 원 R 자료 경로다.
    pub r_readout: Option<RoventRCollectionPlan>,
    pub metadata: RoventCollectionMetadata,
    pub teacher_sources: Vec<RoventTeacherSourceSelection>,
    pub teacher_slots: Vec<RoventCollectionTeacherSlot>,
    pub teacher_constraint: ExternalGoConstraint,
    pub teacher_soft_quantum_ns: u64,
    pub maximum_roots: u16,
    pub maximum_records: u64,
    pub maximum_pending_c: u16,
    /// 새로 생성할 명시 managed archive 디렉터리. 기존 경로는 덮어쓰지 않는다.
    pub output_directory: PathBuf,
    pub publication_timeout_ms: u64,
    pub historical_role_bindings: BTreeMap<String, RoventHistoricalRoleBinding>,
    pub ancestor_datasets: Vec<RoventCollectionAncestor>,
    pub current_ancestor_indices: Vec<u16>,
    pub historical_maximum_read_bytes: u64,
    pub historical_validation_timeout_ms: u64,
}

fn io(error: std::io::Error) -> ArenaError {
    ArenaError::Io(error.to_string())
}
fn manager_error(error: ArenaError) -> AdaptiveManagerError {
    AdaptiveCollectionError::new(
        AdaptiveCollectionStage::Settled,
        "RoventCollectionCaller",
        error.to_string(),
    )
    .into()
}
fn valid_relative(path: &str) -> bool {
    !path.is_empty()
        && !path.contains(['\\', ':'])
        && !Path::new(path).is_absolute()
        && path
            .split('/')
            .all(|p| !p.is_empty() && p != "." && p != ".." && !p.starts_with('.'))
}
fn read_pin(
    directory: &Dir,
    pin: &RoventPublicationFile,
    maximum: u64,
) -> Result<Vec<u8>, ArenaError> {
    pin.validate()?;
    check(
        valid_relative(&pin.path) && pin.bytes <= maximum,
        "Collection input file path/size",
    )?;
    let mut options = cap_std::fs::OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    let mut file = directory.open_with(&pin.path, &options).map_err(io)?;
    let metadata = file.metadata().map_err(io)?;
    check(
        metadata.is_file() && metadata.len() == pin.bytes,
        "Collection input file type/bytes",
    )?;
    let mut bytes = Vec::new();
    Read::take(&mut file, pin.bytes + 1)
        .read_to_end(&mut bytes)
        .map_err(io)?;
    check(
        bytes.len() as u64 == pin.bytes && hex(Digest(Sha256::digest(&bytes).into())) == pin.sha256,
        "Collection original input file SHA",
    )?;
    Ok(bytes)
}

impl RoventCollectionLaunchPlan {
    pub fn validate(&self, manifest: &RoventCorpusManifestV2) -> Result<(), ArenaError> {
        manifest.validate()?;
        self.manifest.validate()?;
        check(
            self.schema == ROVENT_COLLECTION_LAUNCH_PLAN
                && (1..=64).contains(&self.maximum_roots)
                && (1..=8192).contains(&self.maximum_records)
                && (1..=16).contains(&self.maximum_pending_c)
                && (1..=30_000).contains(&self.publication_timeout_ms)
                && (1..=60_000).contains(&self.historical_validation_timeout_ms)
                && self.historical_maximum_read_bytes > 0
                && self.historical_maximum_read_bytes <= 1 << 34
                && !self.historical_role_bindings.is_empty()
                && self.historical_role_bindings.len() <= 2
                && self.ancestor_datasets.len() <= 256
                && self.current_ancestor_indices.len() <= 256
                && self.teacher_soft_quantum_ns > 0
                && self.teacher_soft_quantum_ns <= 1_000_000_000
                && self.teacher_sources.len() == manifest.teachers.len()
                && self.teacher_slots.len() == manifest.teachers.len()
                && !self.teacher_slots.is_empty()
                && self.teacher_slots.len() <= 16
                && self.output_directory.is_absolute()
                && self.output_directory.components().all(|part| {
                    matches!(
                        part,
                        Component::RootDir | Component::Prefix(_) | Component::Normal(_)
                    )
                }),
            "finite Collection plan/teacher coverage/output path",
        )?;
        let mut actors = BTreeSet::new();
        let mut slots = BTreeSet::new();
        let mut owners = BTreeSet::new();
        for binding in &self.teacher_slots {
            check(
                manifest.teachers.contains_key(&binding.actor_id)
                    && actors.insert(binding.actor_id.clone())
                    && slots.insert(binding.source_index)
                    && binding.collection_owner > 0
                    && owners.insert(binding.collection_owner)
                    && binding.source_index < self.teacher_sources.len()
                    && self.teacher_sources[binding.source_index].index == binding.source_index,
                "Collection exact actor/source slot/worker owner coverage",
            )?;
        }
        match self.teacher_constraint {
            ExternalGoConstraint::Depth { depth } => check(depth > 0, "teacher depth")?,
            ExternalGoConstraint::Nodes { reported_nodes } => {
                check(reported_nodes > 0, "teacher nodes")?
            }
            ExternalGoConstraint::Combined {
                depth,
                reported_nodes,
            } => check(depth > 0 && reported_nodes > 0, "teacher combined limit")?,
            ExternalGoConstraint::MoveTime { milliseconds } => {
                check(milliseconds > 0, "teacher time")?
            }
        }
        Ok(())
    }
}

fn historical_budget(
    plan: &RoventCollectionLaunchPlan,
) -> Result<RoventRecoveryReadBudget, ArenaError> {
    let until = Instant::now()
        .checked_add(Duration::from_millis(plan.historical_validation_timeout_ms))
        .ok_or_else(|| invalid("Collection historical deadline overflow"))?;
    RoventRecoveryReadBudget::new(
        plan.historical_maximum_read_bytes,
        until,
        Arc::new(AtomicBool::new(false)),
    )
}
fn select_ancestors(
    indices: &[u16],
    owners: &[Arc<VerifiedRoventDataset>],
) -> Result<Vec<Arc<VerifiedRoventDataset>>, ArenaError> {
    let mut seen = BTreeSet::new();
    indices
        .iter()
        .map(|index| {
            check(
                seen.insert(*index),
                "Collection duplicate ancestor occurrence",
            )?;
            owners
                .get(usize::from(*index))
                .cloned()
                .ok_or_else(|| invalid("Collection ancestor is not yet verified"))
        })
        .collect()
}
fn open_ancestors(
    plan: &RoventCollectionLaunchPlan,
) -> Result<Vec<Arc<VerifiedRoventDataset>>, ArenaError> {
    let budget = historical_budget(plan)?;
    let mut owners = Vec::new();
    for ancestor in &plan.ancestor_datasets {
        let parents = select_ancestors(&ancestor.parent_indices, &owners)?;
        let owner = match ancestor.family {
            RoventCollectionAncestorFamily::LegacyRaw2 => {
                VerifiedRoventDataset::open(ancestor.registration.clone(), parents, &budget)?
            }
            RoventCollectionAncestorFamily::SubmittedInputV1 => {
                VerifiedRoventDataset::open_submitted_input_v1(
                    ancestor.registration.clone(),
                    parents,
                    &budget,
                )?
            }
        };
        owners.push(Arc::new(owner));
    }
    select_ancestors(&plan.current_ancestor_indices, &owners)
}

struct RootCaller {
    admission: RoventCollectionRootAdmission,
    pump: teacher::RoventCollectionTeacherPump,
    pending_c: Vec<RoventRetiredRecordV2>,
    shutdown: Vec<teacher::RoventCollectionTeacherShutdown>,
    shutdown_requested: bool,
}
struct CollectionCaller {
    plan: RoventCollectionLaunchPlan,
    resource: resources::CollectionResourceCustody,
    r: Option<r_stream::RStream>,
    utility: utility_stream::UtilityRootLocalStream,
    unmatched_r_retirement: Option<
        rz_uci::pals_adaptive::AdaptiveCpuSemanticRetirement<
            rz_search::pals::engine::adaptive_port::AdaptivePortQuestion,
        >,
    >,
    manifest: RoventCorpusManifestV2,
    owner: RoventCollectionOwnerV2,
    writer: Option<RoventCorpusWriterV2>,
    finished: Option<FinishedRoventCorpusV2>,
    directory: Dir,
    preload: Option<Arc<RoventCollectionPreloadSources>>,
    cancellation: Arc<AtomicBool>,
    pending_admission: Option<RoventCollectionRootAdmission>,
    active: Option<RootCaller>,
    // 물리 종료 뒤에도 원 오류/완료 근거를 caller가 버리지 않는다.
    closed: Vec<RootCaller>,
    teacher_failures: Vec<teacher::RoventCollectionTeacherFailure>,
    publishing: Option<PublishedRoventRecordV2>,
    retiring: Option<RoventRetiredRecordV2>,
    chunk: Option<RoventChunkPacket>,
    roots: u16,
    records: u64,
    failure: Option<String>,
}

impl CollectionCaller {
    fn utility_completed(
        &mut self,
        actual: &Arc<rz_uci::pals_adaptive::CompletedUtilityRouteAction>,
        source: Arc<rz_uci::rovent_bootstrap::nnuec_source::LoadedRoventNnuecSource>,
    ) -> Result<(), ArenaError> {
        let admission = self
            .closed
            .iter()
            .find(|root| root.admission.authority() == actual.id().authority())
            .map(|root| root.admission.clone())
            .ok_or_else(|| invalid("utility completed action has no original closed root"))?;
        self.utility.completed(
            actual,
            admission,
            source,
            &self.directory,
            Duration::from_millis(self.plan.publication_timeout_ms),
            &self.cancellation,
        )
    }
    fn admit(&mut self, admission: RoventCollectionRootAdmission) -> Result<(), ArenaError> {
        check(
            self.failure.is_none()
                && self.pending_admission.is_none()
                && self.active.is_none()
                && self.roots < self.plan.maximum_roots,
            "Collection root already pending/closed/capped",
        )?;
        self.roots += 1;
        self.pending_admission = Some(admission);
        Ok(())
    }
    fn activate(&mut self) -> Result<(), ArenaError> {
        // Keep the original admission in the caller until session creation
        // succeeds. Roster/scope validation can fail, and a failure must not
        // silently discard the only caller-owned root evidence.
        let Some(admission) = self.pending_admission.as_ref().cloned() else {
            return Ok(());
        };
        let roster = self
            .plan
            .teacher_slots
            .iter()
            .map(|binding| (binding.actor_id.clone(), binding.collection_owner))
            .collect::<Vec<_>>();
        let session =
            teacher::RoventCollectionTeacherSession::new_submitted_input_v1_with_process_scope(
                &self.manifest,
                admission.clone(),
                &roster,
            )?;
        self.pending_admission = None;
        self.active = Some(RootCaller {
            admission,
            pump: teacher::RoventCollectionTeacherPump::new_submitted_input_v1(
                self.owner.clone(),
                session,
            ),
            pending_c: Vec::new(),
            shutdown: Vec::new(),
            shutdown_requested: false,
        });
        let active = self
            .active
            .as_mut()
            .ok_or_else(|| invalid("Collection root disappeared"))?;
        let root = active.admission.root_admission();
        let instant = |deadline: rz_contracts::Deadline| {
            root.runtime_clock()
                .origin()
                .checked_add(Duration::from_nanos(deadline.at.0))
                .ok_or_else(|| invalid("Collection deadline overflow"))
        };
        let until = instant(root.admission_deadline())?;
        let cleanup_until = instant(root.cleanup_deadline())?;
        let control = RoventStartupControl {
            until,
            cleanup_until,
            cancellation: Arc::clone(&self.cancellation),
        };
        let preload = self
            .preload
            .as_ref()
            .ok_or_else(|| invalid("Collection original preload missing"))?;
        for binding in &self.plan.teacher_slots {
            let sources = preload
                .teacher_sources(binding.source_index, &control)
                .map_err(|error| invalid(error.to_string()))?;
            active.pump.session_mut().add_teacher_sources(
                &self.manifest.common_registration_projection(),
                &binding.actor_id,
                sources,
                binding.collection_owner,
                until,
                cleanup_until,
                &self.cancellation,
            )?;
        }
        Ok(())
    }
    fn poll(&mut self) -> Result<(), ArenaError> {
        self.activate()?;
        if let Some(error) = self.owner.failure()? {
            return Err(invalid(error.to_string()));
        }
        if let Some(active) = &mut self.active {
            active.pump.poll_submitted_input_v1(
                &self.manifest,
                self.plan.teacher_constraint,
                Duration::from_nanos(self.plan.teacher_soft_quantum_ns),
            )?;
        }
        self.drain_ready()
    }
    fn drain_ready(&mut self) -> Result<(), ArenaError> {
        check(
            self.retiring.is_none() && self.publishing.is_none() && self.chunk.is_none(),
            "Collection failed publication custody must be recovered before next row",
        )?;
        while let Some(active) = &mut self.active {
            let Some(divergence) = active.pump.next_ready_submitted_input_is_divergence() else {
                break;
            };
            if divergence && active.pending_c.len() >= usize::from(self.plan.maximum_pending_c) {
                return Err(invalid(
                    "Collection pending C custody full; original outboxes retained",
                ));
            }
            let Some(row) = active.pump.take_ready_submitted_input_v1() else {
                break;
            };
            // 실제 witness는 root fence 뒤에만 완성될 수 있다. 모든 C divergence
            // 입력을 원 native ticket과 함께 유한 custody에 보존한다.
            if row.input().selected_view.query.purpose == "divergence_policy" {
                active.pending_c.push(row);
            } else {
                self.publish_row(row)?;
            }
        }
        if let Some(active) = &mut self.active {
            while self.teacher_failures.len() < 1024 {
                let Some(error) = active.pump.take_failure() else {
                    break;
                };
                self.teacher_failures.push(error);
            }
        }
        Ok(())
    }
    fn publish_row(&mut self, row: RoventRetiredRecordV2) -> Result<(), ArenaError> {
        check(
            self.retiring.is_none(),
            "Collection original retired row already retained",
        )?;
        self.retiring = Some(row);
        check(
            self.records < self.plan.maximum_records && self.publishing.is_none(),
            "Collection publication record cap/original pending row",
        )?;
        let row = self
            .retiring
            .as_mut()
            .ok_or_else(|| invalid("Collection original retired row lost"))?;
        self.owner
            .with_producer(|producer| producer.seal_retired(row))?;
        let row = self
            .retiring
            .take()
            .ok_or_else(|| invalid("Collection sealed retired row lost"))?;
        // 원 seal 검사가 성공한 row만 원 private Published 타입으로 이동한다.
        self.publishing = Some(PublishedRoventRecordV2 {
            record: row.record,
            manifest_sha256: row.manifest_sha256,
            admission: row.admission,
        });
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| invalid("Collection writer already finished"))?;
        writer.append_original(
            self.publishing
                .as_ref()
                .ok_or_else(|| invalid("Collection publication disappeared"))?,
        )?;
        self.publishing = None;
        self.records += 1;
        self.flush()
    }
    fn flush(&mut self) -> Result<(), ArenaError> {
        if self.chunk.is_none() {
            self.chunk = self
                .writer
                .as_mut()
                .ok_or_else(|| invalid("Collection writer closed"))?
                .flush()?;
        }
        let Some(packet) = &self.chunk else {
            return Ok(());
        };
        let declaration = &packet.declaration;
        check(
            valid_relative(&declaration.path)
                && packet.bytes.len() as u64 == declaration.bytes
                && hex(Digest(Sha256::digest(&packet.bytes).into())) == declaration.sha256,
            "Collection original chunk packet changed",
        )?;
        let mut options = cap_std::fs::OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No);
        let mut file = self
            .directory
            .open_with(&declaration.path, &options)
            .map_err(io)?;
        file.write_all(&packet.bytes)
            .and_then(|_| file.sync_all())
            .map_err(io)?;
        self.chunk = None;
        Ok(())
    }
    fn close_root<'a>(
        &mut self,
        witnesses: impl Iterator<Item = &'a PcDivergenceWitness>,
    ) -> Result<(), ArenaError> {
        // 원 native fence 후에도 실제 teacher 종료를 별도로 확인한다. 원 root의
        // cleanup ceiling을 늘리지 않으며 timeout은 unknown 실패로 남긴다.
        let mut first_error = self.activate().err();
        let active = self
            .active
            .as_mut()
            .ok_or_else(|| invalid("Collection closure without actual root"))?;
        let root = active.admission.root_admission();
        let until = root
            .runtime_clock()
            .origin()
            .checked_add(Duration::from_nanos(root.cleanup_deadline().at.0))
            .ok_or_else(|| invalid("Collection cleanup overflow"))?;
        if let Err(error) = active.pump.close_input_admission() {
            first_error.get_or_insert(error);
        }
        active.pump.request_shutdown(until);
        active.shutdown_requested = true;
        loop {
            if let Some(active) = &mut self.active {
                if let Err(error) = active.pump.poll_submitted_input_v1(
                    &self.manifest,
                    self.plan.teacher_constraint,
                    Duration::from_nanos(self.plan.teacher_soft_quantum_ns),
                ) {
                    first_error.get_or_insert(error);
                }
                if let Err(error) = active.pump.close_input_admission() {
                    first_error.get_or_insert(error);
                }
                active.shutdown = active.pump.poll_shutdown();
            }
            if let Err(error) = self.drain_ready() {
                first_error.get_or_insert(error);
            }
            let active = self
                .active
                .as_ref()
                .ok_or_else(|| invalid("Collection root custody lost"))?;
            if (active.shutdown.len() == self.plan.teacher_slots.len() || first_error.is_some())
                && active.shutdown.iter().all(|state| {
                    state.worker_joined
                        && state.process.is_some_and(|p| {
                            p.exit_observed
                                && p.stdout_drained
                                && p.stderr_drained
                                && p.cleanup_complete
                                && !p.quarantined
                                && !p.ownership_lost
                        })
                })
                && (active.pump.pending_rows() == 0 || first_error.is_some())
                && (self.owner.pending_retired_count()? == 0 || first_error.is_some())
            {
                break;
            }
            check(
                Instant::now() < until,
                "Collection teacher physical shutdown unknown; custody retained",
            )?;
            std::thread::sleep(Duration::from_millis(1));
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        if self
            .active
            .as_ref()
            .is_some_and(|active| active.shutdown.iter().any(|state| state.error.is_some()))
        {
            return Err(invalid(
                "Collection teacher shutdown reported an original failure; custody retained",
            ));
        }
        // 원 report를 먼저 빼앗지 않고 SAME cycle owner의 witness를 빌린다.
        for witness in witnesses {
            let original = witness.source().accepted().ticket();
            if let Some(active) = &mut self.active {
                let visible = active
                    .pending_c
                    .iter()
                    .position(|row| row.ticket() == original)
                    .ok_or_else(|| {
                        invalid("actual visible C retired row missing; witness not relabeled")
                    })?;
                if let Some(attack) = witness.source().attack_ranked_source() {
                    let initial = active
                        .pending_c
                        .iter()
                        .position(|row| row.ticket() == attack.accepted().ticket())
                        .ok_or_else(|| invalid("actual initial attack retired row missing"))?;
                    check(
                        initial != visible,
                        "initial and future visible ticket must differ",
                    )?;
                    let (initial, visible) = if initial < visible {
                        let (left, right) = active.pending_c.split_at_mut(visible);
                        (&mut left[initial], &right[0])
                    } else {
                        let (left, right) = active.pending_c.split_at_mut(initial);
                        (&mut right[0], &left[visible])
                    };
                    self.owner.with_producer(|producer| {
                        producer.attach_pc_divergence_from_visible(initial, witness, visible)
                    })?;
                } else {
                    self.owner.with_producer(|producer| {
                        producer.attach_pc_divergence(&mut active.pending_c[visible], witness)
                    })?;
                }
            }
        }
        loop {
            let row = self
                .active
                .as_mut()
                .and_then(|active| active.pending_c.pop());
            let Some(row) = row else {
                break;
            };
            self.publish_row(row)?;
        }
        let active = self
            .active
            .take()
            .ok_or_else(|| invalid("Collection closed root lost"))?;
        self.closed.push(active);
        Ok(())
    }
}

/// 성공 결과는 실제 원 writer가 게시한 파일 pin이다. 학습이나 현재 native
/// 권한을 발급하지 않는다. 오류는 원 caller custody를 함께 반환한다.
pub struct RoventCollectionLaunchReceipt {
    pub publication: RoventCorpusPublicationV2,
    pub publication_file: RoventPublicationFile,
    pub output_directory: PathBuf,
    pub roots: u16,
    pub records: u64,
    pub teacher_failure_count: usize,
    pub historical_dataset: Arc<VerifiedRoventDataset>,
    pub r_datasets:
        Vec<Arc<crate::rovent_training_process::recovery::VerifiedRoventRAuxiliaryDataset>>,
    pub r_sidecars: Vec<RoventTrainingFilePin>,
    pub utility_root_local_files: Vec<RoventTrainingFilePin>,
    original: Arc<Mutex<CollectionCaller>>,
}
impl RoventCollectionLaunchReceipt {
    /// 같은 resource scope와 원 FD가 성공 뒤에도 보존되어 있다.
    pub fn resource_observation(&self) -> Result<Value, ArenaError> {
        Ok(self
            .original
            .lock()
            .map_err(|_| invalid("Collection resource custody poisoned"))?
            .resource
            .observation())
    }
}
pub struct RoventCollectionLaunchFailure {
    pub detail: String,
    original: Option<Arc<Mutex<CollectionCaller>>>,
}
impl std::fmt::Debug for RoventCollectionLaunchFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoventCollectionLaunchFailure")
            .field("detail", &self.detail)
            .field("original_caller_retained", &self.original.is_some())
            .finish()
    }
}
impl std::fmt::Display for RoventCollectionLaunchFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.detail.fmt(f)
    }
}
impl std::error::Error for RoventCollectionLaunchFailure {}

fn hook(
    shared: &Arc<Mutex<CollectionCaller>>,
    action: impl FnOnce(&mut CollectionCaller) -> Result<(), ArenaError>,
) -> Result<(), AdaptiveManagerError> {
    let mut state = shared
        .lock()
        .map_err(|_| AdaptiveManagerError::Invalid("Collection caller poisoned"))?;
    match action(&mut state) {
        Ok(()) => Ok(()),
        Err(error) => {
            state.failure.get_or_insert_with(|| error.to_string());
            Err(manager_error(error))
        }
    }
}

/// 원 root/native hook과 별도 teacher caller를 실제 제품 launch에 연결한다.
/// arguments는 기존 UCI source/config/FD 경계를 그대로 통과한다.
// The native assembly closure returns the original prepared Product/Store and
// cleanup owners in RoventProductStartupRejection. Keeping that bounded single
// startup custody inline avoids allocating while handling a failed startup;
// this does not replace failure with a metadata-only error or release its pins.
#[allow(clippy::result_large_err)]
pub fn run_rovent_collection(
    arguments: Vec<String>,
    plan: RoventCollectionLaunchPlan,
    input_directory: &Dir,
) -> Result<RoventCollectionLaunchReceipt, RoventCollectionLaunchFailure> {
    let prepare = || -> Result<_, ArenaError> {
        let raw = read_pin(input_directory, &plan.manifest, 16 << 20)?;
        let manifest: RoventCorpusManifestV2 = rz_experiments::decode_json(
            std::str::from_utf8(&raw).map_err(|_| invalid("Collection manifest UTF8"))?,
        )?;
        plan.validate(&manifest)?;
        let parent = plan
            .output_directory
            .parent()
            .ok_or_else(|| invalid("Collection output parent"))?;
        let leaf = plan
            .output_directory
            .file_name()
            .ok_or_else(|| invalid("Collection output leaf"))?;
        let original_parent =
            Dir::open_ambient_dir(parent, cap_std::ambient_authority()).map_err(io)?;
        original_parent.create_dir(leaf).map_err(io)?;
        let directory = original_parent.open_dir_nofollow(leaf).map_err(io)?;
        directory.create_dir("chunks").map_err(io)?;
        if plan.r_readout.is_some() {
            directory.create_dir("r-roots").map_err(io)?;
        }
        let (owner, observer) =
            RoventCollectionOwnerV2::new(manifest.clone(), plan.metadata.clone())?;
        let writer = RoventCorpusWriterV2::new(&manifest)?;
        Ok((manifest, directory, owner, observer, writer))
    };
    let (manifest, directory, owner, observer, writer) =
        prepare().map_err(|error| RoventCollectionLaunchFailure {
            detail: error.to_string(),
            original: None,
        })?;
    let ancestors = open_ancestors(&plan).map_err(|error| RoventCollectionLaunchFailure {
        detail: error.to_string(),
        original: None,
    })?;
    let r = plan
        .r_readout
        .clone()
        .map(|request| {
            r_stream::RStream::prepare(input_directory, request, &plan, ancestors.clone())
        })
        .transpose()
        .map_err(|error| RoventCollectionLaunchFailure {
            detail: error.to_string(),
            original: None,
        })?;
    let shared = Arc::new(Mutex::new(CollectionCaller {
        plan: plan.clone(),
        r,
        utility: utility_stream::UtilityRootLocalStream::default(),
        unmatched_r_retirement: None,
        resource: resources::CollectionResourceCustody::prepare(
            input_directory,
            &plan.resource_plan,
        )
        .map_err(|error| RoventCollectionLaunchFailure {
            detail: error.to_string(),
            original: None,
        })?,
        manifest,
        owner,
        writer: Some(writer),
        finished: None,
        directory,
        preload: None,
        cancellation: Arc::new(AtomicBool::new(false)),
        pending_admission: None,
        active: None,
        closed: Vec::new(),
        teacher_failures: Vec::new(),
        publishing: None,
        retiring: None,
        chunk: None,
        roots: 0,
        records: 0,
        failure: None,
    }));
    let original = Arc::clone(&shared);
    let resource = Arc::clone(&shared);
    let result = run_collection_with_resource_admission(
        arguments,
        plan.teacher_sources.clone(),
        Box::new(observer),
        move |lock| {
            let mut state = resource.lock().map_err(|_| RoventLaunchError {
                stage: "resource_admission",
                detail: "Collection resource custody poisoned".into(),
            })?;
            state
                .resource
                .admit(lock)
                .map_err(|error| RoventLaunchError {
                    stage: "resource_admission",
                    detail: error.to_string(),
                })?;
            if state.r.is_some()
                && (lock.manifest.engine.nnuec.is_none()
                    || !lock.manifest.engine.mode.has_native_r()
                    || !matches!(
                        &lock.manifest.engine.researcher,
                        Some(rz_experiments::RoventResearcher::Own { .. })
                    ))
            {
                return Err(RoventLaunchError {
                    stage: "r_capture_source",
                    detail: "R capture requires the original own NNUEC source".into(),
                });
            }
            Ok(())
        },
        move |prepared| {
            let RoventCollectionPreparedLaunch {
                native,
                preload,
                control,
                clock,
                owners,
                identity,
                root_policy,
            } = prepared;
            // 생성자가 호출되기 전에는 원 native owner를 잘못된 오류 값으로 버리지 않는다.
            if let Ok(mut state) = original.lock() {
                state.preload = Some(preload);
                state.cancellation = Arc::clone(&control.cancellation);
            }
            let admission = Arc::clone(&original);
            let closure = Arc::clone(&original);
            let cooperative = Arc::clone(&original);
            let poll: Arc<dyn Fn() -> Result<(), AdaptiveManagerError> + Send + Sync> =
                Arc::new(move || hook(&cooperative, |state| state.poll()));
            let r_source = match &native {
                RoventCollectionPreparedNative::OwnNnuecPcr { loaded, .. } => {
                    Some(Arc::new((**loaded).clone()))
                }
                _ => None,
            };
            let utility_caller = Arc::clone(&original);
            let utility_route_completed = r_source.clone().map(|source| {
                Arc::new(
                    move |actual: &Arc<rz_uci::pals_adaptive::CompletedUtilityRouteAction>| {
                        hook(&utility_caller, |state| {
                            state.utility_completed(actual, Arc::clone(&source))
                        })
                    },
                )
                    as Arc<
                        dyn Fn(
                                &Arc<rz_uci::pals_adaptive::CompletedUtilityRouteAction>,
                            ) -> Result<(), AdaptiveManagerError>
                            + Send
                            + Sync,
                    >
            });
            match native {
                native @ (RoventCollectionPreparedNative::Pc(_)
                | RoventCollectionPreparedNative::PcNnuec { .. }) => {
                    let hooks = RoventPcCollectionHooks {
                        root_admission: Some(Arc::new(move |manager, cap| {
                            hook(&admission, |state| {
                                manager
                                    .install_collection_observer(
                                        cap.clone(),
                                        state.owner.observer(),
                                    )
                                    .map_err(|error| invalid(format!("{error:?}")))?;
                                state.admit(cap)
                            })
                        })),
                        root_closure: Some(Arc::new(move |manager| {
                            hook(&closure, |state| {
                                state.close_root(manager.driver().port().divergence_witnesses())
                            })
                        })),
                        cooperative: Some(poll),
                    };
                    match native {
                        RoventCollectionPreparedNative::Pc(prepared) => {
                            RoventNativeProductDriver::pc_with_collection_hooks(
                                *prepared,
                                clock,
                                owners,
                                identity,
                                root_policy,
                                hooks,
                            )
                        }
                        RoventCollectionPreparedNative::PcNnuec { prepared, loaded } => {
                            RoventNativeProductDriver::pc_with_shared_nnuec_and_collection_hooks(
                                *prepared,
                                *loaded,
                                clock,
                                owners,
                                identity,
                                root_policy,
                                hooks,
                            )
                        }
                        RoventCollectionPreparedNative::OwnBoardPcr(_)
                        | RoventCollectionPreparedNative::OwnNnuecPcr { .. }
                        | RoventCollectionPreparedNative::ExternalPcr { .. } => {
                            unreachable!("separate exact PC variants")
                        }
                    }
                }
                native => {
                    let hooks = AdaptiveSessionCollectionHooks {
                        root_admission: Some(Arc::new(move |manager| {
                            hook(&admission, |state| {
                                manager
                                    .set_collection_observer(state.owner.observer())
                                    .map_err(|error| invalid(format!("{error:?}")))?;
                                let cap = state
                                    .owner
                                    .current_admission()?
                                    .ok_or_else(|| invalid("Collection actual cap missing"))?;
                                if let Some(stream) = &mut state.r {
                                    let source = r_source.as_ref().ok_or_else(|| {
                                        invalid("original NNUEC R source missing")
                                    })?;
                                    let observer = stream.admit(
                                        cap.clone(),
                                        Arc::clone(source),
                                        &state.plan,
                                        &state.directory,
                                    )?;
                                    manager
                                        .set_cpu_semantic_observer(observer)
                                        .map_err(|error| invalid(format!("{error:?}")))?;
                                }
                                state.admit(cap)
                            })
                        })),
                        root_closure: Some(Arc::new(move |manager, _completion| {
                            hook(&closure, |state| {
                                let retirement = manager
                                    .take_cpu_semantic_retirement()
                                    .map_err(|error| invalid(format!("{error:?}")))?;
                                let teacher = state.close_root(std::iter::empty());
                                let r = match retirement {
                                    Some(retirement) => match &mut state.r {
                                        Some(stream) => stream.close(retirement, &state.plan),
                                        None => {
                                            state.unmatched_r_retirement = Some(retirement);
                                            Err(invalid("unmatched actual R retirement retained"))
                                        }
                                    },
                                    None => Ok(()),
                                };
                                teacher?;
                                r
                            })
                        })),
                        external_teacher: None,
                        utility_route_completed,
                        utility_trial_retired: None,
                        utility_intervention_trial_retired: None,
                        cooperative: Some(poll),
                    };
                    match native {
                        RoventCollectionPreparedNative::OwnBoardPcr(prepared) => {
                            RoventNativeProductDriver::own_board_pcr_with_collection_hooks(
                                *prepared,
                                clock,
                                owners,
                                identity,
                                root_policy,
                                hooks,
                            )
                        }
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
                            Some(hooks),
                        ),
                        RoventCollectionPreparedNative::Pc(_)
                        | RoventCollectionPreparedNative::PcNnuec { .. } => {
                            unreachable!("separate exact PC variants")
                        }
                    }
                }
            }
        },
    );
    // 실패에도 원 OS scope를 보존하고 가능한 현재 관측을 남긴다. 실제 종료
    // 후 전체 비교 자료가 부족하면 시작 성공을 취소하지 않고 여기서 HOLD한다.
    let resource_result = shared
        .lock()
        .map_err(|_| invalid("Collection resource custody poisoned"))
        .and_then(|mut state| state.resource.finish_live());
    let finish = || -> Result<RoventCollectionLaunchReceipt, ArenaError> {
        result.map_err(|error| invalid(error.to_string()))?;
        let mut state = shared
            .lock()
            .map_err(|_| invalid("Collection final caller poisoned"))?;
        check(
            state.failure.is_none()
                && state.active.is_none()
                && state.pending_admission.is_none()
                && state.publishing.is_none()
                && state.retiring.is_none()
                && state.teacher_failures.is_empty()
                && state.unmatched_r_retirement.is_none()
                && state.r.as_ref().is_none_or(|stream| stream.fully_closed())
                && state.owner.pending_retired_count()? == 0
                && state.records > 0
                && state.roots as usize == state.closed.len(),
            "Collection session not fully fenced; original caller retained",
        )?;
        state.flush()?;
        if state.finished.is_none() {
            let writer = state
                .writer
                .take()
                .ok_or_else(|| invalid("Collection writer already finalized"))?;
            match writer.finish_publication_retaining() {
                Ok(finished) => state.finished = Some(finished),
                Err(rejected) => {
                    let rejected = *rejected;
                    state.writer = Some(rejected.writer);
                    return Err(rejected.failure);
                }
            }
        }
        let until = Instant::now()
            .checked_add(Duration::from_millis(state.plan.publication_timeout_ms))
            .ok_or_else(|| invalid("Collection publication deadline overflow"))?;
        // 실행 취소와 별도로 이미 끝난 원 데이터의 내구성 보존만 수행한다.
        let archive_cancel = AtomicBool::new(false);
        let finished = state
            .finished
            .as_ref()
            .ok_or_else(|| invalid("Collection original finished custody lost"))?;
        let publication = finished.publish_atomic(&state.directory, until, &archive_cancel)?;
        finished.publish_allocation_supplement(&state.directory, until, &archive_cancel)?;
        let raw = serde_json::to_vec(&publication).map_err(|error| invalid(error.to_string()))?;
        let publication_file = RoventPublicationFile {
            path: publication::ROVENT_PUBLICATION_FILE.into(),
            sha256: hex(Digest(Sha256::digest(&raw).into())),
            bytes: raw.len() as u64,
        };
        let manifest = &state.manifest;
        let registration = RoventHistoricalDatasetRegistration {
            root: state.plan.output_directory.clone(),
            publication: RoventTrainingFilePin {
                path: state.plan.output_directory.join(&publication_file.path),
                sha256: publication_file.sha256.clone(),
                bytes: publication_file.bytes,
            },
            collection_source_commit: manifest.source_commit.clone(),
            collection_source_sha256: manifest.source_sha256.clone(),
            collection_runtime_library_sha256: manifest.runtime_sha256.clone(),
            model_semantics_sha256: manifest.model_semantics_sha256.clone(),
            model_epoch: manifest.model_epoch,
            context_epoch: manifest.context_epoch,
            recipe_sha256: manifest.recipe_sha256.clone(),
            stage_id: manifest.stage_id.clone(),
            role_bindings: state.plan.historical_role_bindings.clone(),
        };
        let budget = historical_budget(&state.plan)?;
        let historical_dataset = Arc::new(VerifiedRoventDataset::open_submitted_input_v1(
            registration,
            ancestors,
            &budget,
        )?);
        let r_datasets = state
            .r
            .as_ref()
            .map(|stream| stream.for_current_pc(Arc::clone(&historical_dataset), &state.plan))
            .transpose()?
            .unwrap_or_default();
        // 종료 후 엄격 비교 HOLD도 이미 퇴역한 원 자료의 내구 게시와 전체
        // FD 검증을 건너뛰지 않는다. 같은 scope의 원 오류는 caller에 남는다.
        resource_result?;
        Ok(RoventCollectionLaunchReceipt {
            publication,
            publication_file,
            output_directory: state.plan.output_directory.clone(),
            roots: state.roots,
            records: state.records,
            teacher_failure_count: state.teacher_failures.len(),
            historical_dataset,
            r_datasets,
            r_sidecars: state
                .r
                .as_ref()
                .map(|stream| stream.sidecars())
                .unwrap_or_default(),
            utility_root_local_files: state.utility.files(&state.plan.output_directory),
            original: Arc::clone(&shared),
        })
    };
    finish().map_err(|error| RoventCollectionLaunchFailure {
        detail: error.to_string(),
        original: Some(shared),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn collection_cli_keeps_original_source_arguments_and_exact_plan_pin() {
        let parsed = parse_rovent_collection_cli(vec![
            "--collection-plan=inputs/plan.json".into(),
            format!("--collection-plan-sha256={}", "a".repeat(64)),
            "--collection-plan-bytes=31".into(),
            "--input-lock=inputs/lock.json".into(),
            "--root-policy=dynamic-go".into(),
        ])
        .unwrap();
        let RoventCollectionCliParse::Run { plan, arguments } = parsed else {
            panic!("run")
        };
        assert_eq!(plan.bytes, 31);
        assert_eq!(
            arguments,
            ["--input-lock=inputs/lock.json", "--root-policy=dynamic-go"]
        );
    }
    #[test]
    fn collection_cli_rejects_missing_duplicate_and_escaping_plan_without_reading() {
        assert!(
            parse_rovent_collection_cli(vec!["--collection-plan=inputs/plan.json".into()]).is_err()
        );
        let make = |path: &str| {
            vec![
                format!("--collection-plan={path}"),
                format!("--collection-plan-sha256={}", "a".repeat(64)),
                "--collection-plan-bytes=1".into(),
            ]
        };
        assert!(parse_rovent_collection_cli(make("inputs/../other.json")).is_err());
        let mut repeated = make("inputs/plan.json");
        repeated.push("--collection-plan-bytes=2".into());
        assert!(parse_rovent_collection_cli(repeated).is_err());
    }
    #[test]
    fn cohort_cli_has_an_explicit_family_and_preserves_original_bootstrap_options() {
        let args = vec![
            "--utility-cohort-plan=inputs/cohort.json".into(),
            format!("--utility-cohort-plan-sha256={}", "b".repeat(64)),
            "--utility-cohort-plan-bytes=91".into(),
            "--process-epoch=700".into(),
            "--input-lock=inputs/lock.json".into(),
        ];
        let RoventCollectionCliParse::Cohort { plan, arguments } =
            parse_rovent_collection_cli(args.clone()).unwrap()
        else {
            panic!("explicit cohort")
        };
        assert_eq!(plan.bytes, 91);
        assert_eq!(
            arguments,
            ["--process-epoch=700", "--input-lock=inputs/lock.json"]
        );
        let mut mixed = args;
        mixed.push("--collection-plan=inputs/old.json".into());
        assert!(parse_rovent_collection_cli(mixed).is_err());
    }
}
