//! native 퇴역 outbox와 독립 CPU_T session 사이의 유한 caller pump다.
//! manager 안에서 호출하지 않으며 실제 pipe 처리는 teacher worker가 담당한다.
use super::{session::*, *};
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

enum PumpRow {
    Legacy(RoventRetiredRecord),
    SubmittedV1(RoventRetiredRecordV2),
}
impl PumpRow {
    fn input_sha256(&self) -> &str {
        match self {
            Self::Legacy(row) => row.input_sha256(),
            Self::SubmittedV1(row) => row.input_sha256(),
        }
    }
    fn actor(&self) -> Option<&str> {
        match self {
            Self::Legacy(row) => row.record.lineage.teacher_actor_id.as_deref(),
            Self::SubmittedV1(row) => row.record.lineage.teacher_actor_id.as_deref(),
        }
    }
    fn has_candidates(&self) -> bool {
        match self {
            Self::Legacy(row) => !row.record.input.legal_moves.is_empty(),
            Self::SubmittedV1(row) => !row.record.input.legal_moves.is_empty(),
        }
    }
}
enum PumpOwner {
    Legacy(RoventCollectionOwner),
    SubmittedV1(RoventCollectionOwnerV2),
}
impl PumpOwner {
    fn take_retired(&self) -> Result<Option<PumpRow>, ArenaError> {
        match self {
            Self::Legacy(owner) => owner.take_retired().map(|r| r.map(PumpRow::Legacy)),
            Self::SubmittedV1(owner) => owner.take_retired().map(|r| r.map(PumpRow::SubmittedV1)),
        }
    }
    fn attach(
        &self,
        session: &RoventCollectionTeacherSession,
        row: &mut PumpRow,
        completed: &RoventCollectionTeacherCompleted,
    ) -> Result<(), ArenaError> {
        match (self, row) {
            (Self::Legacy(owner), PumpRow::Legacy(row)) => {
                owner.with_producer(|p| session.attach_policy(p, row, completed))
            }
            (Self::SubmittedV1(owner), PumpRow::SubmittedV1(row)) => {
                owner.with_producer(|p| session.attach_policy_submitted_input_v1(p, row, completed))
            }
            _ => Err(invalid("Collection pump refuses mixed corpus family")),
        }
    }
    fn submit(
        &self,
        session: &mut RoventCollectionTeacherSession,
        row: &PumpRow,
        constraint: rz_search::external_cpu::ExternalGoConstraint,
        quantum: Duration,
    ) -> Result<rz_search::external_cpu::collection_teacher::TeacherJobId, ArenaError> {
        match (self, row) {
            (Self::Legacy(owner), PumpRow::Legacy(row)) => {
                owner.with_producer(|p| session.submit_retired(p, row, constraint, quantum))
            }
            (Self::SubmittedV1(owner), PumpRow::SubmittedV1(row)) => owner.with_producer(|p| {
                session.submit_retired_submitted_input_v1(p, row, constraint, quantum)
            }),
            _ => Err(invalid("Collection pump refuses mixed corpus family")),
        }
    }
}

struct WaitingRow {
    row: PumpRow,
    issued: bool,
}
pub struct RoventCollectionTeacherFailure {
    pub input_sha256: String,
    pub detail: String,
    original_completion: Option<RoventCollectionTeacherCompleted>,
}
impl RoventCollectionTeacherFailure {
    pub fn completion(
        &self,
    ) -> Option<&rz_search::external_cpu::collection_teacher::TeacherCompletion> {
        self.original_completion.as_ref().map(|v| v.completion())
    }
}
pub struct RoventCollectionTeacherPump {
    owner: PumpOwner,
    session: RoventCollectionTeacherSession,
    waiting: BTreeMap<String, WaitingRow>,
    ready: VecDeque<PumpRow>,
    failures: VecDeque<RoventCollectionTeacherFailure>,
    quarantined_rows: VecDeque<PumpRow>,
    closed: bool,
}
impl RoventCollectionTeacherPump {
    pub fn new_submitted_input_v1(
        owner: RoventCollectionOwnerV2,
        session: RoventCollectionTeacherSession,
    ) -> Self {
        Self {
            owner: PumpOwner::SubmittedV1(owner),
            session,
            waiting: BTreeMap::new(),
            ready: VecDeque::new(),
            failures: VecDeque::new(),
            quarantined_rows: VecDeque::new(),
            closed: false,
        }
    }
    pub fn poll_submitted_input_v1(
        &mut self,
        manifest: &rz_experiments::rovent_corpus_v2::RoventCorpusManifestV2,
        constraint: rz_search::external_cpu::ExternalGoConstraint,
        quantum: Duration,
    ) -> Result<usize, ArenaError> {
        check(
            matches!(self.owner, PumpOwner::SubmittedV1(_)),
            "submitted-input poll refuses legacy owner",
        )?;
        manifest.validate()?;
        self.poll(
            &manifest.common_registration_projection(),
            constraint,
            quantum,
        )
    }
    pub fn take_ready_submitted_input_v1(&mut self) -> Option<RoventRetiredRecordV2> {
        if !matches!(self.ready.front(), Some(PumpRow::SubmittedV1(_))) {
            return None;
        }
        self.ready.pop_front().map(|row| match row {
            PumpRow::SubmittedV1(row) => row,
            _ => unreachable!("checked submitted-input"),
        })
    }
    #[cfg(target_os = "linux")]
    pub(crate) fn next_ready_submitted_input_is_divergence(&self) -> Option<bool> {
        self.ready.front().and_then(|row| match row {
            PumpRow::SubmittedV1(row) => {
                Some(row.input().selected_view.query.purpose == "divergence_policy")
            }
            _ => None,
        })
    }
    pub fn take_quarantined_submitted_input_v1(&mut self) -> Option<RoventRetiredRecordV2> {
        if !matches!(self.quarantined_rows.front(), Some(PumpRow::SubmittedV1(_))) {
            return None;
        }
        self.quarantined_rows.pop_front().map(|row| match row {
            PumpRow::SubmittedV1(row) => row,
            _ => unreachable!("checked submitted-input"),
        })
    }
    pub fn new(owner: RoventCollectionOwner, session: RoventCollectionTeacherSession) -> Self {
        Self {
            owner: PumpOwner::Legacy(owner),
            session,
            waiting: BTreeMap::new(),
            ready: VecDeque::new(),
            failures: VecDeque::new(),
            quarantined_rows: VecDeque::new(),
            closed: false,
        }
    }
    pub fn session(&self) -> &RoventCollectionTeacherSession {
        &self.session
    }
    pub fn session_mut(&mut self) -> &mut RoventCollectionTeacherSession {
        &mut self.session
    }
    /// native root 종료 전 별도 caller가 반복 호출한다. 한 poll은 최대 16개
    /// 원 row와 actor당 한 terminal만 처리하고 실제 teacher 완료를 기다리지 않는다.
    pub fn poll(
        &mut self,
        manifest: &RoventCorpusManifest,
        constraint: rz_search::external_cpu::ExternalGoConstraint,
        quantum: Duration,
    ) -> Result<usize, ArenaError> {
        check(
            self.failures.len() < 16 && self.ready.len() < 16 && self.quarantined_rows.len() < 16,
            "Collection caller outbox full; drain original evidence",
        )?;
        let mut changed = 0;
        while self.waiting.len() + self.ready.len() + self.quarantined_rows.len() < 16 {
            let Some(row) = self.owner.take_retired()? else {
                break;
            };
            let sha = row.input_sha256().to_owned();
            if self.waiting.contains_key(&sha) {
                self.quarantined_rows.push_back(row);
                self.closed = true;
                return Err(invalid(
                    "duplicate original teacher input occurrence retained",
                ));
            }
            if row.actor().is_none() || !row.has_candidates() {
                self.ready.push_back(row);
            } else {
                self.waiting.insert(sha, WaitingRow { row, issued: false });
            }
            changed += 1;
        }
        for actor in manifest.teachers.keys() {
            if self.failures.len() >= 16 || self.ready.len() >= 16 {
                break;
            }
            if let Some(completed) = self.session.take_completed(actor)? {
                let sha = completed.input_sha256().to_owned();
                let Some(mut waiting) = self.waiting.remove(&sha) else {
                    self.failures.push_back(RoventCollectionTeacherFailure {
                        input_sha256: sha,
                        detail: "orphan Collection completion".into(),
                        original_completion: Some(completed),
                    });
                    return Err(invalid("orphan Collection completion retained"));
                };
                let result = self
                    .owner
                    .attach(&self.session, &mut waiting.row, &completed);
                if let Err(error) = result {
                    self.failures.push_back(RoventCollectionTeacherFailure {
                        input_sha256: sha,
                        detail: error.to_string(),
                        original_completion: Some(completed),
                    });
                }
                // 실패/부분 의견은 policy=None이다. 원 row 및 원 완료는 각각
                // 보존하며 실패를 성공 label로 바꾸지 않는다.
                self.ready.push_back(waiting.row);
                changed += 1;
            }
        }
        if self.closed {
            return Ok(changed);
        }
        if !self.session.startup_ready(manifest)? {
            return Ok(changed);
        }
        for actor in manifest.teachers.keys() {
            if self.failures.len() >= 16 || self.ready.len() >= 16 {
                break;
            }
            let has_pending = self
                .waiting
                .values()
                .any(|v| v.issued && v.row.actor() == Some(actor.as_str()));
            if has_pending {
                continue;
            }
            let next = self
                .waiting
                .iter()
                .find(|(_, v)| !v.issued && v.row.actor() == Some(actor.as_str()))
                .map(|(sha, _)| sha.clone());
            let Some(sha) = next else {
                continue;
            };
            let waiting = self
                .waiting
                .get_mut(&sha)
                .ok_or_else(|| invalid("Collection waiting row disappeared"))?;
            match self
                .owner
                .submit(&mut self.session, &waiting.row, constraint, quantum)
            {
                Ok(_) => {
                    waiting.issued = true;
                    changed += 1;
                }
                Err(error) => {
                    // 원 deadline이 종료한 입력을 새 go로 재태그하지 않는다.
                    self.failures.push_back(RoventCollectionTeacherFailure {
                        input_sha256: sha.clone(),
                        detail: error.to_string(),
                        original_completion: None,
                    });
                    let waiting = self
                        .waiting
                        .remove(&sha)
                        .ok_or_else(|| invalid("Collection failed row disappeared"))?;
                    self.ready.push_back(waiting.row);
                    changed += 1;
                }
            }
        }
        Ok(changed)
    }
    pub fn take_ready(&mut self) -> Option<RoventRetiredRecord> {
        if !matches!(self.ready.front(), Some(PumpRow::Legacy(_))) {
            return None;
        }
        self.ready.pop_front().map(|row| match row {
            PumpRow::Legacy(row) => row,
            _ => unreachable!("checked legacy"),
        })
    }
    pub fn take_failure(&mut self) -> Option<RoventCollectionTeacherFailure> {
        self.failures.pop_front()
    }
    pub fn take_quarantined_row(&mut self) -> Option<RoventRetiredRecord> {
        if !matches!(self.quarantined_rows.front(), Some(PumpRow::Legacy(_))) {
            return None;
        }
        self.quarantined_rows.pop_front().map(|row| match row {
            PumpRow::Legacy(row) => row,
            _ => unreachable!("checked legacy"),
        })
    }
    pub fn request_shutdown(&mut self, until: Instant) {
        self.closed = true;
        self.session.request_shutdown(until);
    }
    /// 실제 root fence 뒤 더 이상 교사 질문을 제출하지 않는다. 아직 제출하지
    /// 않은 row는 unknown으로 그대로 돌려주며, 이미 제출한 질문의 terminal은
    /// poll이 원 completion을 회수할 때까지 waiting custody에 남는다.
    pub fn close_input_admission(&mut self) -> Result<(), ArenaError> {
        self.closed = true;
        let unissued = self
            .waiting
            .iter()
            .filter(|(_, row)| !row.issued)
            .map(|(sha, _)| sha.clone())
            .collect::<Vec<_>>();
        check(
            self.ready.len() + unissued.len() <= 16,
            "Collection shutdown ready outbox full; original rows retained",
        )?;
        for sha in unissued {
            if let Some(waiting) = self.waiting.remove(&sha) {
                self.ready.push_back(waiting.row);
            }
        }
        Ok(())
    }
    pub fn poll_shutdown(&mut self) -> Vec<RoventCollectionTeacherShutdown> {
        self.session.poll_shutdown()
    }
    pub fn pending_rows(&self) -> usize {
        self.waiting.len()
    }
}
