//! 실제 go cap 뒤의 독립 CPU_T session caller다. native preload와 다르며
//! P-C 검색 manager에서 교사 pipe I/O를 실행하지 않는다.
use super::collection::RoventCollectionTeacherBinding;
use super::*;
use rz_contracts::Deadline;
use rz_search::external_cpu::collection_teacher::{
    TeacherCompletion, TeacherDispatch, TeacherJob, TeacherJobId, TeacherWorkerRegistration,
};
use rz_search::external_cpu::process_scope::CollectionProcessScope;
use rz_search::external_cpu::worker::ExternalWorkerClock;
use rz_uci::rovent_external_launch::RoventCollectionTeacherOwner;
use std::{
    fs::File,
    sync::Arc,
    time::{Duration, Instant},
};

struct ActorOwner {
    registered: RoventRegisteredTeacher,
    owner: RoventCollectionTeacherOwner,
    next_sequence: u64,
    original_source_custody: Option<rz_uci::rovent_bootstrap::RoventCollectionTeacherSourceCustody>,
    original_slot: Option<usize>,
}
struct Pending {
    actor: String,
    binding: RoventCollectionTeacherBinding,
    dispatch: TeacherDispatch,
}
/// actual cap은 from_attached_frontier에서 발급된 원 opaque 값이다. 입력 JSON이나
/// startup 관측만으로 이 session을 재구성하는 경로는 없다.
pub struct RoventCollectionTeacherSession {
    manifest_sha256: String,
    registration_metadata_sha256: String,
    admission: RoventCollectionRootAdmission,
    actors: BTreeMap<String, ActorOwner>,
    pending: BTreeMap<(u32, u64), Pending>,
    shutdown_requested: bool,
    quarantined: Vec<(TeacherCompletion, String)>,
    process_scope: Option<CollectionProcessScope>,
}
pub struct RoventCollectionTeacherCompleted {
    actor: String,
    binding: RoventCollectionTeacherBinding,
    dispatch: TeacherDispatch,
    completion: TeacherCompletion,
}
impl RoventCollectionTeacherCompleted {
    pub fn completion(&self) -> &TeacherCompletion {
        &self.completion
    }
    pub fn input_sha256(&self) -> &str {
        self.binding.input_sha256()
    }
    pub fn actor_id(&self) -> &str {
        &self.actor
    }
}
impl RoventRetiredRecord {
    pub fn original_teacher_query_position(&self) -> Result<Position, ArenaError> {
        let replay: rules_replay::RoventSubmittedRulesReplay =
            serde_json::from_value(self.record.native.raw_submission["rules_replay"].clone())
                .map_err(|e| invalid(e.to_string()))?;
        replay.original_query_position(&self.record.input)
    }
}
#[derive(Clone, Debug)]
pub struct RoventCollectionTeacherShutdown {
    pub actor_id: String,
    pub worker_joined: bool,
    pub process: Option<rz_search::cpu_checker::CheckerShutdown>,
    pub error: Option<String>,
}
impl RoventCollectionTeacherSession {
    /// 새 closed family의 실제 corpus seal은 /2로 유지하고 공통 actor 등록
    /// metadata 검사만 공유한다. 공통 projection 문서를 파일로 게시하지 않는다.
    pub fn new_submitted_input_v1(
        manifest: &rz_experiments::rovent_corpus_v2::RoventCorpusManifestV2,
        admission: RoventCollectionRootAdmission,
    ) -> Result<Self, ArenaError> {
        manifest.validate()?;
        let mut session = Self::new(&manifest.common_registration_projection(), admission)?;
        session.manifest_sha256 = manifest.sha256()?;
        Ok(session)
    }

    /// Register the exact bounded teacher roster for this opaque admitted root.
    /// This only permits the declared idle processes; go/order/resource limits
    /// and all original teacher startup evidence remain unchanged.
    pub fn new_submitted_input_v1_with_process_scope(
        manifest: &rz_experiments::rovent_corpus_v2::RoventCorpusManifestV2,
        admission: RoventCollectionRootAdmission,
        roster: &[(String, u32)],
    ) -> Result<Self, ArenaError> {
        let mut session = Self::new_submitted_input_v1(manifest, admission)?;
        check(
            !roster.is_empty()
                && roster.len() <= 16
                && roster.len() == manifest.teachers.len()
                && roster
                    .iter()
                    .map(|(actor, _)| actor)
                    .collect::<BTreeSet<_>>()
                    .len()
                    == roster.len()
                && roster
                    .iter()
                    .all(|(actor, owner)| *owner != 0 && manifest.teachers.contains_key(actor)),
            "Collection process roster differs from frozen manifest",
        )?;
        let root = session.admission.root_admission();
        let registrations = roster
            .iter()
            .map(|(actor_id, collection_owner)| {
                let registered = RoventRegisteredTeacher {
                    actor_id: actor_id.clone(),
                    pin: manifest.teachers[actor_id].clone(),
                };
                Ok(process_registration(
                    root,
                    *collection_owner,
                    registered.frozen_actor_digest()?,
                ))
            })
            .collect::<Result<Vec<_>, ArenaError>>()?;
        session.process_scope = Some(
            CollectionProcessScope::new(
                process_scope_manifest_digest(&session.manifest_sha256)?,
                registrations,
            )
            .map_err(|error| invalid(error.to_string()))?,
        );
        Ok(session)
    }
    pub fn new(
        manifest: &RoventCorpusManifest,
        admission: RoventCollectionRootAdmission,
    ) -> Result<Self, ArenaError> {
        manifest.validate()?;
        check(
            manifest.sha256()?.len() == 64
                && manifest.source_sha256
                    == admission.native_origin().raw_receipt()["source_sha256"],
            "Collection teacher session original manifest/source cap",
        )?;
        Ok(Self {
            manifest_sha256: manifest.sha256()?,
            registration_metadata_sha256: manifest.sha256()?,
            admission,
            actors: BTreeMap::new(),
            pending: BTreeMap::new(),
            shutdown_requested: false,
            quarantined: vec![],
            process_scope: None,
        })
    }
    /// Caller는 preload가 보존한 원 source FD를 이동한다. 실제 worker가 동일 FD
    /// bytes를 다시 검증하고 startup을 관측할 때까지 target은 발급하지 않는다.
    #[allow(clippy::too_many_arguments)]
    pub fn add_pinned_teacher(
        &mut self,
        manifest: &RoventCorpusManifest,
        actor_id: &str,
        launch: PreparedRoventExternalLaunch,
        original_profile_bytes: &[u8],
        executable: File,
        cwd: File,
        collection_owner: u32,
        startup_deadline: Instant,
        cleanup_ceiling: Instant,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<(), ArenaError> {
        check(
            !self.shutdown_requested
                && manifest.sha256()? == self.registration_metadata_sha256
                && !self.actors.contains_key(actor_id)
                && !self
                    .actors
                    .values()
                    .any(|a| a.owner.worker().registration().collection_owner == collection_owner),
            "Collection teacher duplicate/closed session",
        )?;
        let registered = RoventRegisteredTeacher::from_prepared(
            manifest,
            actor_id,
            &launch,
            original_profile_bytes,
        )?;
        let root = Arc::clone(self.admission.root_admission());
        let clock = ExternalWorkerClock {
            domain: root.hard_deadline().clock,
            origin: root.runtime_clock().origin(),
        };
        // 두 원 소유자의 정리 상한 중 더 이른 값을 사용한다. root의 여유를
        // 교사 profile의 shutdown_grace 확장으로 해석하지 않는다.
        let profile = launch
            .configuration()
            .map_err(|error| invalid(error.to_string()))?;
        let profile_cleanup = clock
            .origin
            .checked_add(Duration::from_nanos(root.hard_deadline().at.0))
            .and_then(|hard| hard.checked_add(profile.shutdown_grace))
            .ok_or_else(|| invalid("Collection teacher original cleanup ceiling overflow"))?;
        let root_cleanup = clock
            .origin
            .checked_add(Duration::from_nanos(root.cleanup_deadline().at.0))
            .ok_or_else(|| invalid("Collection root original cleanup ceiling overflow"))?;
        let cleanup_ceiling = cleanup_ceiling.min(root_cleanup).min(profile_cleanup);
        let frozen_actor = registered.frozen_actor_digest()?;
        let permit = self
            .process_scope
            .as_ref()
            .map(|scope| {
                scope
                    .permit(
                        process_scope_manifest_digest(&self.manifest_sha256)?,
                        process_registration(root.as_ref(), collection_owner, frozen_actor),
                    )
                    .map_err(|error| invalid(error.to_string()))
            })
            .transpose()?;
        let owner = match permit {
            Some(permit) => launch.spawn_collection_teacher_pinned_in_process_scope(
                executable,
                cwd,
                collection_owner,
                frozen_actor,
                root,
                clock,
                startup_deadline,
                cleanup_ceiling,
                cancel,
                permit,
            ),
            None => launch.spawn_collection_teacher_pinned(
                executable,
                cwd,
                collection_owner,
                frozen_actor,
                root,
                clock,
                startup_deadline,
                cleanup_ceiling,
                cancel,
            ),
        }
        .map_err(|e| invalid(e.to_string()))?;
        self.actors.insert(
            actor_id.into(),
            ActorOwner {
                registered,
                owner,
                next_sequence: 1,
                original_source_custody: None,
                original_slot: None,
            },
        );
        Ok(())
    }
    /// Product caller는 preload가 보존한 전체 source bundle을 이동한다. 원
    /// source Arc와 별도 asset FD는 worker의 최종 physical shutdown까지 보존한다.
    #[allow(clippy::too_many_arguments)]
    pub fn add_teacher_sources(
        &mut self,
        manifest: &RoventCorpusManifest,
        actor_id: &str,
        sources: rz_uci::rovent_bootstrap::RoventCollectionTeacherSources,
        collection_owner: u32,
        startup_deadline: Instant,
        cleanup_ceiling: Instant,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<(), ArenaError> {
        check(
            sources.original_input_lock_digest()
                == self.admission.root_admission().startup_lock_digest(),
            "Collection preload source belongs to another original input lock",
        )?;
        let (index, prepared, profile, executable, cwd, custody) = sources.into_parts();
        check(
            !self
                .actors
                .values()
                .any(|owner| owner.original_slot == Some(index)),
            "duplicate original Collection teacher slot",
        )?;
        self.add_pinned_teacher(
            manifest,
            actor_id,
            prepared,
            &profile,
            executable,
            cwd,
            collection_owner,
            startup_deadline,
            cleanup_ceiling,
            cancel,
        )?;
        let actor = self
            .actors
            .get_mut(actor_id)
            .ok_or_else(|| invalid("Collection teacher source owner missing after spawn"))?;
        actor.original_source_custody = Some(custody);
        actor.original_slot = Some(index);
        Ok(())
    }
    /// 모든 원 frozen teacher의 실제 ready를 확인한다. Pending과 실패를 성공으로
    /// 바꾸지 않으며, options applied/NNUE loaded의 독립 관측도 주장하지 않는다.
    pub fn startup_ready(&self, manifest: &RoventCorpusManifest) -> Result<bool, ArenaError> {
        check(
            manifest.sha256()? == self.registration_metadata_sha256
                && self.actors.keys().eq(manifest.teachers.keys()),
            "Collection teacher actual owner coverage differs from frozen manifest",
        )?;
        for actor in self.actors.values() {
            if actor
                .owner
                .verified_startup()
                .map_err(|e| invalid(e.to_string()))?
                .is_none()
            {
                return Ok(false);
            }
        }
        Ok(true)
    }
    /// Root manager는 이 호출을 하지 않는다. 별도 Collection caller가 원 native
    /// submitted seal을 넘기며 teacher 예산·원 deadline을 따로 보존한다.
    #[allow(clippy::too_many_arguments)]
    pub fn submit(
        &mut self,
        producer: &RoventCorpusProducer,
        input: &RoventSubmittedInputSeal,
        actor_id: &str,
        position: Position,
        root_moves: Option<Vec<BoardMove>>,
        constraint: rz_search::external_cpu::ExternalGoConstraint,
        original_deadline: Deadline,
        soft_quantum: Duration,
    ) -> Result<TeacherJobId, ArenaError> {
        check(
            !self.shutdown_requested && input.manifest_sha256 == self.manifest_sha256,
            "Collection teacher closed/wrong input",
        )?;
        let actor = self
            .actors
            .get_mut(actor_id)
            .ok_or_else(|| invalid("Collection teacher actor not installed"))?;
        let id = TeacherJobId {
            collection_owner: actor.owner.worker().registration().collection_owner,
            sequence: actor.next_sequence,
        };
        let next = actor
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| invalid("Collection teacher sequence exhausted"))?;
        let scope = producer.collection_teacher_scope(input, &actor.registered, &actor.owner)?;
        let job = TeacherJob {
            id,
            input_scope: scope,
            position,
            root_moves,
            constraint,
            original_deadline,
            soft_quantum,
            control: actor.owner.worker().control_mailbox(id),
        };
        let binding =
            producer.bind_collection_teacher_input(input, &actor.registered, &actor.owner, &job)?;
        let dispatch = actor
            .owner
            .submit(job)
            .map_err(|e| invalid(e.to_string()))?;
        actor.next_sequence = next;
        self.pending.insert(
            (id.collection_owner, id.sequence),
            Pending {
                actor: actor_id.into(),
                binding,
                dispatch,
            },
        );
        Ok(id)
    }
    pub fn submit_retired(
        &mut self,
        producer: &RoventCorpusProducer,
        row: &RoventRetiredRecord,
        constraint: rz_search::external_cpu::ExternalGoConstraint,
        soft_quantum: Duration,
    ) -> Result<TeacherJobId, ArenaError> {
        check(
            !row.record.input.legal_moves.is_empty(),
            "teacher policy has no supported legal candidates",
        )?;
        let actor = row
            .record
            .lineage
            .teacher_actor_id
            .as_deref()
            .ok_or_else(|| invalid("row has no frozen teacher actor"))?;
        self.submit(
            producer,
            &row.submitted_input_seal(),
            actor,
            row.original_teacher_query_position()?,
            None,
            constraint,
            row.ticket.original_deadline,
            soft_quantum,
        )
    }
    /// 원 private completion은 input의 NN 퇴역보다 먼저 도착해도 여기 보존한다.
    /// 별도 completed owner는 미래 target을 과거 native input에 삽입하지 않는다.
    pub fn take_completed(
        &mut self,
        actor_id: &str,
    ) -> Result<Option<RoventCollectionTeacherCompleted>, ArenaError> {
        check(
            self.quarantined.len() < 16,
            "Collection completion quarantine full",
        )?;
        let actor = self
            .actors
            .get(actor_id)
            .ok_or_else(|| invalid("Collection teacher actor absent"))?;
        let Some(completion) = actor.owner.worker().take_completion() else {
            return Ok(None);
        };
        let id = completion.id();
        let key = (id.collection_owner, id.sequence);
        let Some(pending) = self.pending.get(&key) else {
            self.shutdown_requested = true;
            self.quarantined
                .push((completion, "Collection teacher orphan completion".into()));
            return Err(invalid("Collection teacher orphan completion quarantined"));
        };
        // Physical validation must precede target application, but an Err
        // completion is still returned as its original failure for evidence.
        if pending.actor != actor_id {
            self.shutdown_requested = true;
            self.quarantined
                .push((completion, "Collection teacher actor mismatch".into()));
            return Err(invalid("Collection teacher actor mismatch quarantined"));
        }
        if let Err(error) = actor
            .owner
            .validate_completion(&completion, &pending.dispatch)
        {
            self.shutdown_requested = true;
            self.quarantined.push((completion, error.to_string()));
            return Err(invalid("Collection teacher physical mismatch quarantined"));
        }
        let pending = self
            .pending
            .remove(&key)
            .ok_or_else(|| invalid("Collection teacher pending disappeared"))?;
        Ok(Some(RoventCollectionTeacherCompleted {
            actor: pending.actor,
            binding: pending.binding,
            dispatch: pending.dispatch,
            completion,
        }))
    }
    pub fn attach_policy(
        &self,
        producer: &mut RoventCorpusProducer,
        row: &mut RoventRetiredRecord,
        completed: &RoventCollectionTeacherCompleted,
    ) -> Result<(), ArenaError> {
        let actor = self
            .actors
            .get(&completed.actor)
            .ok_or_else(|| invalid("Collection teacher owner retired before head attachment"))?;
        check(
            row.manifest_sha256 == self.manifest_sha256,
            "Collection teacher retired row manifest",
        )?;
        producer.attach_collection_teacher_policy(
            row,
            completed.binding.clone(),
            &actor.owner,
            &completed.dispatch,
            &completed.completion,
        )
    }
    pub fn request_shutdown(&mut self, until: Instant) {
        self.shutdown_requested = true;
        // Drop the initial issuer reference even while closed root evidence is
        // retained. Running/quarantined owners retain their own opaque leases.
        if let Some(scope) = self.process_scope.take() {
            scope.close_admission();
        }
        for actor in self.actors.values() {
            actor.owner.worker().request_shutdown(until);
        }
    }
    pub fn attach_policy_submitted_input_v1(
        &self,
        producer: &mut RoventCorpusProducerV2,
        row: &mut RoventRetiredRecordV2,
        completed: &RoventCollectionTeacherCompleted,
    ) -> Result<(), ArenaError> {
        let actor = self
            .actors
            .get(&completed.actor)
            .ok_or_else(|| invalid("Collection teacher owner retired before head attachment"))?;
        check(
            row.manifest_sha256 == self.manifest_sha256,
            "raw3 Collection teacher row manifest",
        )?;
        producer.attach_collection_teacher_policy(
            row,
            completed.binding.clone(),
            &actor.owner,
            &completed.dispatch,
            &completed.completion,
        )
    }
    pub fn submit_retired_submitted_input_v1(
        &mut self,
        producer: &RoventCorpusProducerV2,
        row: &RoventRetiredRecordV2,
        constraint: rz_search::external_cpu::ExternalGoConstraint,
        soft_quantum: Duration,
    ) -> Result<TeacherJobId, ArenaError> {
        check(
            !row.record.input.legal_moves.is_empty(),
            "raw3 teacher policy has no legal candidates",
        )?;
        let actor = row
            .record
            .lineage
            .teacher_actor_id
            .as_deref()
            .ok_or_else(|| invalid("raw3 row has no frozen teacher actor"))?;
        let replay: RoventSubmittedRulesReplay =
            serde_json::from_value(row.record.native.raw_submission["rules_replay"].clone())
                .map_err(|error| invalid(error.to_string()))?;
        let position =
            replay.original_query_position(&row.record.input.common_validation_projection())?;
        self.submit(
            producer.original_teacher_producer(),
            &row.submitted_input_seal(),
            actor,
            position,
            None,
            constraint,
            row.ticket.original_deadline,
            soft_quantum,
        )
    }
    pub fn poll_shutdown(&mut self) -> Vec<RoventCollectionTeacherShutdown> {
        self.actors
            .iter_mut()
            .map(|(id, actor)| {
                let result = actor
                    .owner
                    .try_join()
                    .or_else(|| actor.owner.worker().shutdown_snapshot());
                RoventCollectionTeacherShutdown {
                    actor_id: id.clone(),
                    worker_joined: result.is_some(),
                    process: actor.owner.worker().process_after_join(),
                    error: result.and_then(Result::err).map(|e| e.to_string()),
                }
            })
            .collect()
    }
    pub fn take_diagnostics_after_join(
        &mut self,
        actor_id: &str,
    ) -> Option<rz_search::external_cpu::collection_teacher::TeacherDiagnostics> {
        self.actors
            .get_mut(actor_id)?
            .owner
            .take_diagnostics_after_join()
    }
    pub fn quarantined_completions(&self) -> impl Iterator<Item = (&TeacherCompletion, &str)> {
        self.quarantined.iter().map(|(c, why)| (c, why.as_str()))
    }
}

fn process_registration(
    root: &rz_uci::rovent_root_admission::RoventRootAdmission,
    collection_owner: u32,
    frozen_actor: Digest,
) -> TeacherWorkerRegistration {
    TeacherWorkerRegistration {
        collection_owner,
        authority: root.authority(),
        original_deadline: root.hard_deadline(),
        input_lock: root.startup_lock_digest(),
        root_admission: root.digest(),
        root_budget: root.root_budget_digest(),
        frozen_actor,
    }
}

fn process_scope_manifest_digest(value: &str) -> Result<Digest, ArenaError> {
    check(
        generation_sha(value),
        "Collection process manifest SHA syntax",
    )?;
    let mut bytes = [0; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let digit = |byte: u8| {
            if byte <= b'9' {
                byte - b'0'
            } else {
                byte - b'a' + 10
            }
        };
        bytes[index] = digit(pair[0]) * 16 + digit(pair[1]);
    }
    Ok(Digest(bytes))
}
