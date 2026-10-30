//! 실제 원 root·worker·process의 정지와 reset 관측만 발급한다.
//! 공개 report, caller bool, 다음 root 추정으로 증거를 만들지 않는다.

use super::*;
use std::sync::Weak;

/// worker가 정상·오류·panic 어느 경로로 끝나도 옛 source의 admission을 닫는다.
/// 이것은 physical cleanup 성공을 발급하지 않으며 남은 terminal 예약도 보존한다.
pub(super) struct WorkerAdmissionGuard {
    pub(super) shutdown: Arc<AtomicBool>,
    pub(super) mailbox: Mailbox,
}

impl Drop for WorkerAdmissionGuard {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.mailbox.close();
        self.mailbox.signal().notify();
    }
}

#[derive(Clone)]
pub struct ProfiledExternalRootSource {
    roots: Weak<Mutex<RootState>>,
    shutdown: Weak<AtomicBool>,
    authority: SearchAuthority,
    source_sequence: u64,
    owner: CpuWorkerId,
    clock: ExternalWorkerClock,
    profile: ExternalEngineProfile,
    purpose: ExternalEnginePurpose,
    process: ExternalProcessIdentity,
}

pub type ExternalRootSource = ProfiledExternalRootSource;

impl std::fmt::Debug for ProfiledExternalRootSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProfiledExternalRootSource")
            .field("authority", &self.authority)
            .field("source_sequence", &self.source_sequence)
            .field("owner", &self.owner)
            .field("clock", &self.clock)
            .field("profile", &self.profile)
            .field("purpose", &self.purpose)
            .field("process", &self.process)
            .finish_non_exhaustive()
    }
}

impl ProfiledExternalRootSource {
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    /// Historical generation metadata only; this does not construct a source
    /// capability or grant current admission, root reuse or execution credit.
    pub fn source_sequence(&self) -> u64 {
        self.source_sequence
    }
    pub fn worker_owner(&self) -> CpuWorkerId {
        self.owner
    }
    pub fn owner(&self) -> CpuWorkerId {
        self.owner
    }
    pub fn process_identity(&self) -> ExternalProcessIdentity {
        self.process
    }
    pub fn clock(&self) -> ExternalWorkerClock {
        self.clock
    }
    pub fn profile(&self) -> ExternalEngineProfile {
        self.profile
    }
    pub fn purpose(&self) -> ExternalEnginePurpose {
        self.purpose
    }
    /// 원 출처의 동일성이다. 취소·퇴역 후 cleanup에서도 비교할 수 있으며 현재
    /// admission이 열려 있다는 의미는 별도 validate_current로만 검사한다.
    pub fn same_source(&self, other: &Self) -> bool {
        self.same_worker(other)
            && self.authority == other.authority
            && self.source_sequence == other.source_sequence
    }
    /// 다음 root의 source를 같은 실제 worker에 재결합할 때의 opaque 비교다.
    /// authority·source sequence만 제외하며, 공개 PID나 owner 숫자의 일치만으로
    /// 다른 worker를 채택하지 않는다. 현재 admission의 liveness와는 별도다.
    pub fn same_worker(&self, other: &Self) -> bool {
        self.roots.ptr_eq(&other.roots)
            && self.shutdown.ptr_eq(&other.shutdown)
            && self.owner == other.owner
            && same_clock(self.clock, other.clock)
            && self.profile == other.profile
            && self.purpose == other.purpose
            && self.process == other.process
    }
    pub fn validate_current(&self) -> Result<(), CheckerError> {
        let shutdown = self
            .shutdown
            .upgrade()
            .ok_or_else(|| error("profiled_source", "worker_owner_dropped"))?;
        if shutdown.load(Ordering::Acquire) {
            return Err(error("profiled_source", "worker_closed"));
        }
        let roots = self
            .roots
            .upgrade()
            .ok_or_else(|| error("profiled_source", "root_owner_dropped"))?;
        let root = roots
            .lock()
            .map_err(|_| error("profiled_source", "state_poisoned"))?;
        if root.authority != self.authority
            || root.source_sequence != self.source_sequence
            || root.process != Some(self.process)
            || root.is_revoked()
            || root.lifecycle_failure.is_some()
            || root.initial_reset.is_none()
        {
            return Err(error("profiled_source", "root_revoked_or_replaced"));
        }
        Ok(())
    }
}

/// 실제 ucinewgame 명령과 그 뒤 readyok를 관측한 동일 owner만 발급한다.
/// 초기 proof는 실제 reset이며, pristine startup을 reset으로 재표시하지 않는다.
#[derive(Clone, Debug)]
pub struct ProfiledExternalResetProof {
    source: ProfiledExternalRootSource,
    previous: Option<SearchAuthority>,
    sequence: u64,
}

impl ProfiledExternalResetProof {
    pub fn authority(&self) -> SearchAuthority {
        self.source.authority()
    }
    pub fn previous_authority(&self) -> Option<SearchAuthority> {
        self.previous
    }
    pub fn worker_owner(&self) -> CpuWorkerId {
        self.source.worker_owner()
    }
    pub fn owner(&self) -> CpuWorkerId {
        self.source.owner()
    }
    pub fn process_identity(&self) -> ExternalProcessIdentity {
        self.source.process_identity()
    }
    pub fn clock(&self) -> ExternalWorkerClock {
        self.source.clock()
    }
    pub fn profile(&self) -> ExternalEngineProfile {
        self.source.profile()
    }
    pub fn purpose(&self) -> ExternalEnginePurpose {
        self.source.purpose()
    }
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    pub fn is_initial(&self) -> bool {
        self.previous.is_none()
    }
    pub fn matches_source(&self, source: &ProfiledExternalRootSource) -> bool {
        self.source.same_source(source)
    }
}

/// 원 authority의 admission을 닫고 실제 terminal과 제어를 모두 회수한 뒤의
/// ready barrier다. 다음 root·quit·process exit·Rules 결과를 증명하지 않는다.
#[derive(Clone, Debug)]
pub struct ProfiledExternalCurrentRootFence {
    source: ProfiledExternalRootSource,
    sequence: u64,
    execution_high_water: u64,
    native_request_high_water: u64,
    accepted_jobs: u64,
    terminal_controls: u64,
    drained_terminals: u64,
    process: CheckerShutdown,
}

impl ProfiledExternalCurrentRootFence {
    pub(crate) fn shutdown_receipt(&self) -> CheckerShutdown {
        self.process
    }
    pub fn source(&self) -> &ProfiledExternalRootSource {
        &self.source
    }
    pub fn authority(&self) -> SearchAuthority {
        self.source.authority()
    }
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    pub fn process_identity(&self) -> ExternalProcessIdentity {
        self.source.process_identity()
    }
    pub fn execution_high_water(&self) -> u64 {
        self.execution_high_water
    }
    pub fn native_request_high_water(&self) -> u64 {
        self.native_request_high_water
    }
    pub fn accepted_jobs(&self) -> u64 {
        self.accepted_jobs
    }
    pub fn terminal_controls(&self) -> u64 {
        self.terminal_controls
    }
    pub fn drained_terminals(&self) -> u64 {
        self.drained_terminals
    }
    pub fn permits_reuse(&self) -> bool {
        reusable_process(self.process)
            && self.process.process_identity == Some(self.source.process_identity())
            && self.accepted_jobs == self.terminal_controls
            && self.accepted_jobs == self.drained_terminals
    }
}

/// 같은 실제 worker의 join·process exit·pipe drain을 확인한 close 전용 증거다.
/// ready barrier나 root 재사용을 증명하지 않으며, 공개 shutdown DTO로 생성할 수 없다.
#[derive(Clone, Debug)]
pub struct ProfiledExternalProcessCloseProof {
    source: ProfiledExternalRootSource,
    process: CheckerShutdown,
    accepted_jobs: u64,
    terminal_controls: u64,
    drained_terminals: u64,
}

impl ProfiledExternalProcessCloseProof {
    pub fn source(&self) -> &ProfiledExternalRootSource {
        &self.source
    }
    pub fn authority(&self) -> SearchAuthority {
        self.source.authority()
    }
    pub fn process_identity(&self) -> ExternalProcessIdentity {
        self.source.process_identity()
    }
    pub fn worker_owner(&self) -> CpuWorkerId {
        self.source.worker_owner()
    }
    pub fn clock(&self) -> ExternalWorkerClock {
        self.source.clock()
    }
    pub fn accepted_jobs(&self) -> u64 {
        self.accepted_jobs
    }
    pub fn terminal_controls(&self) -> u64 {
        self.terminal_controls
    }
    pub fn drained_terminals(&self) -> u64 {
        self.drained_terminals
    }
    pub fn shutdown_receipt(&self) -> CheckerShutdown {
        self.process
    }
    pub fn permits_semantic_closure(&self) -> bool {
        completed_process_close(self.process, self.source.process_identity())
            && self.accepted_jobs == self.terminal_controls
            && self.accepted_jobs == self.drained_terminals
    }
}

fn completed_process_close(process: CheckerShutdown, original: ExternalProcessIdentity) -> bool {
    process.process_identity == Some(original)
        && process.exit_observed
        && process.stdout_drained
        && process.stderr_drained
        && process.cleanup_complete
        && !process.quarantined
        && !process.ownership_lost
}

#[derive(Clone, Copy, Debug)]
pub(super) struct QuiesceRequest {
    authority: SearchAuthority,
    sequence: u64,
    until: Instant,
}

impl RootState {
    pub(super) fn new(authority: SearchAuthority) -> Self {
        Self {
            authority,
            transition: None,
            result: None,
            fence_sequence: 0,
            execution_high_water: 0,
            last_question: None,
            source_sequence: 0,
            process: None,
            accepted_jobs: 0,
            terminal_controls: 0,
            drained_terminals: 0,
            lifecycle_failure: None,
            quiesce: None,
            quiesce_result: None,
            initial_reset: None,
            current_reset: None,
            reset_sequence: 0,
        }
    }
    pub(super) fn is_revoked(&self) -> bool {
        self.transition.is_some() || self.quiesce.is_some()
    }
    pub(super) fn physical_deadline(&self) -> Option<Instant> {
        match (
            self.transition.map(|request| request.until),
            self.quiesce.map(|request| request.until),
        ) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }
    pub(super) fn terminals_drained(&self) -> bool {
        self.lifecycle_failure.is_none()
            && self.accepted_jobs == self.terminal_controls
            && self.accepted_jobs == self.drained_terminals
    }
}

fn same_clock(a: ExternalWorkerClock, b: ExternalWorkerClock) -> bool {
    a.domain == b.domain && a.origin == b.origin
}

#[allow(clippy::too_many_arguments)]
fn source_for(
    roots: &Arc<Mutex<RootState>>,
    shutdown: &Arc<AtomicBool>,
    root: &RootState,
    owner: CpuWorkerId,
    clock: ExternalWorkerClock,
    profile: ExternalEngineProfile,
    purpose: ExternalEnginePurpose,
) -> Result<ProfiledExternalRootSource, CheckerError> {
    let process = root
        .process
        .ok_or_else(|| error("profiled_source", "actual_process_missing"))?;
    Ok(ProfiledExternalRootSource {
        roots: Arc::downgrade(roots),
        shutdown: Arc::downgrade(shutdown),
        authority: root.authority,
        source_sequence: root.source_sequence,
        owner,
        clock,
        profile,
        purpose,
        process,
    })
}

impl ProfiledExternalWorker {
    /// take_completion으로 원 terminal을 모두 회수하고 실제 try_join이 성공한
    /// 뒤에만 close 증거를 돌려준다. root quiesce·ready를 대신 발급하지 않는다.
    pub fn process_close_proof(
        &self,
    ) -> Result<Option<ProfiledExternalProcessCloseProof>, CheckerError> {
        let Some(joined) = self.joined.as_ref() else {
            return Ok(None);
        };
        let process = *joined.as_ref().map_err(Clone::clone)?;
        if self.thread.is_some() || !self.shutdown.load(Ordering::Acquire) {
            return Err(error(
                "profiled_close",
                "worker_not_joined_or_admission_open",
            ));
        }
        let occupancy = self.mailbox.occupancy();
        if !occupancy.closed
            || occupancy.commands != 0
            || occupancy.reliable_queued != 0
            || occupancy.reliable_reserved != 0
        {
            return Ok(None);
        }
        let root = self
            .roots
            .lock()
            .map_err(|_| error("profiled_close", "state_poisoned"))?;
        if !root.terminals_drained() {
            if root.lifecycle_failure.is_some()
                || root.terminal_controls > root.accepted_jobs
                || root.drained_terminals > root.terminal_controls
            {
                return Err(error(
                    "profiled_close",
                    "original_terminal_accounting_inconsistent",
                ));
            }
            // reliable dequeue와 원 counter 갱신 사이를 실제 failure로 오인하지 않는다.
            return Ok(None);
        }
        let source = source_for(
            &self.roots,
            &self.shutdown,
            &root,
            self.owner,
            self.clock,
            self.profile,
            self.purpose,
        )?;
        if !completed_process_close(process, source.process_identity()) {
            return Err(error("profiled_close", "original_process_cleanup_unknown"));
        }
        Ok(Some(ProfiledExternalProcessCloseProof {
            source,
            process,
            accepted_jobs: root.accepted_jobs,
            terminal_controls: root.terminal_controls,
            drained_terminals: root.drained_terminals,
        }))
    }

    pub fn current_root_source(&self) -> Result<Option<ProfiledExternalRootSource>, CheckerError> {
        let Some(startup) = self.startup_report() else {
            return Ok(None);
        };
        startup.result?;
        if self.thread.as_ref().is_none_or(JoinHandle::is_finished) {
            return Err(error("profiled_source", "physical_owner_unavailable"));
        }
        let root = self
            .roots
            .lock()
            .map_err(|_| error("profiled_source", "state_poisoned"))?;
        let source = source_for(
            &self.roots,
            &self.shutdown,
            &root,
            self.owner,
            self.clock,
            self.profile,
            self.purpose,
        )?;
        drop(root);
        source.validate_current()?;
        Ok(Some(source))
    }

    pub fn initial_reset_proof(&self) -> Result<Option<ProfiledExternalResetProof>, CheckerError> {
        let Some(startup) = self.startup_report() else {
            return Ok(None);
        };
        startup.result?;
        Ok(self
            .roots
            .lock()
            .map_err(|_| error("profiled_reset", "state_poisoned"))?
            .initial_reset
            .clone())
    }

    pub fn current_reset_proof(&self) -> Option<ProfiledExternalResetProof> {
        let root = self.roots.lock().unwrap_or_else(|p| p.into_inner());
        root.current_reset
            .as_ref()
            .filter(|proof| {
                proof.authority() == root.authority
                    && proof.source.source_sequence == root.source_sequence
            })
            .cloned()
    }

    /// 별도 lifecycle mailbox로 admission을 즉시 닫는다. 원 terminal은 caller가
    /// 계속 회수해야 하며, pending0 이전에 ready fence를 발급하지 않는다.
    pub fn request_root_quiesce(
        &self,
        authority: SearchAuthority,
        until: Instant,
    ) -> Result<(), CheckerError> {
        if self.shutdown.load(Ordering::Acquire)
            || self.thread.as_ref().is_none_or(JoinHandle::is_finished)
            || Instant::now() >= until
        {
            return Err(error("profiled_quiesce", "closed_or_expired"));
        }
        let mut root = self
            .roots
            .lock()
            .map_err(|_| error("profiled_quiesce", "state_poisoned"))?;
        if root.authority != authority
            || root.transition.is_some()
            || root.lifecycle_failure.is_some()
        {
            return Err(error("profiled_quiesce", "wrong_or_transitioning_root"));
        }
        if let Some(old) = root.quiesce {
            if old.authority != authority {
                return Err(error("profiled_quiesce", "different_quiesce_pending"));
            }
            if root.quiesce_result.is_none() {
                root.quiesce = Some(QuiesceRequest {
                    until: old.until.min(until),
                    ..old
                });
            }
        } else {
            root.fence_sequence = root
                .fence_sequence
                .checked_add(1)
                .ok_or(CheckerError::Invalid("profiled root sequence overflow"))?;
            root.quiesce = Some(QuiesceRequest {
                authority,
                sequence: root.fence_sequence,
                until,
            });
            root.quiesce_result = None;
        }
        drop(root);
        self.mailbox.signal().notify();
        Ok(())
    }

    pub fn root_quiesce_result(
        &self,
    ) -> Option<Result<ProfiledExternalCurrentRootFence, CheckerError>> {
        self.roots
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .quiesce_result
            .clone()
    }
}

/// start·ucinewgame·ready가 모두 같은 물리 owner에서 성공한 뒤에 호출한다.
#[allow(clippy::too_many_arguments)]
pub(super) fn register_initial_reset(
    checker: &mut ExternalUciCpuChecker,
    roots: &Arc<Mutex<RootState>>,
    shutdown: &Arc<AtomicBool>,
    owner: CpuWorkerId,
    clock: ExternalWorkerClock,
    profile: ExternalEngineProfile,
    purpose: ExternalEnginePurpose,
) -> Result<(), CheckerError> {
    let process = checker.owner()?.evidence();
    if !reusable_process(process) {
        return Err(error("profiled_reset", "initial_owner_not_reusable"));
    }
    let mut root = roots
        .lock()
        .map_err(|_| error("profiled_reset", "state_poisoned"))?;
    if root.process.is_some() || root.initial_reset.is_some() || root.accepted_jobs != 0 {
        return Err(error(
            "profiled_reset",
            "duplicate_or_nonpristine_initial_reset",
        ));
    }
    root.process = process.process_identity;
    root.reset_sequence = 1;
    let proof = ProfiledExternalResetProof {
        source: source_for(roots, shutdown, &root, owner, clock, profile, purpose)?,
        previous: None,
        sequence: root.reset_sequence,
    };
    root.initial_reset = Some(proof.clone());
    root.current_reset = Some(proof);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn register_transition_reset(
    roots: &Arc<Mutex<RootState>>,
    shutdown: &Arc<AtomicBool>,
    root: &mut RootState,
    previous: SearchAuthority,
    owner: CpuWorkerId,
    clock: ExternalWorkerClock,
    profile: ExternalEngineProfile,
    purpose: ExternalEnginePurpose,
) -> Result<(), CheckerError> {
    root.reset_sequence = root
        .reset_sequence
        .checked_add(1)
        .ok_or(CheckerError::Invalid("profiled reset sequence overflow"))?;
    root.current_reset = Some(ProfiledExternalResetProof {
        source: source_for(roots, shutdown, root, owner, clock, profile, purpose)?,
        previous: Some(previous),
        sequence: root.reset_sequence,
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn poll_quiesce(
    checker: &mut ExternalUciCpuChecker,
    roots: &Arc<Mutex<RootState>>,
    shutdown: &Arc<AtomicBool>,
    shutdown_deadline: &Mutex<Option<Instant>>,
    mailbox: &Mailbox,
    owner: CpuWorkerId,
    clock: ExternalWorkerClock,
    profile: ExternalEngineProfile,
    purpose: ExternalEnginePurpose,
) -> bool {
    let request = {
        let root = roots.lock().unwrap_or_else(|p| p.into_inner());
        root.quiesce_result
            .is_none()
            .then_some(root.quiesce)
            .flatten()
    };
    let Some(request) = request else {
        return false;
    };
    if Instant::now() >= request.until {
        roots
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .quiesce_result = Some(Err(error(
            "profiled_quiesce",
            "deadline_before_terminal_drain",
        )));
        mailbox.signal().notify();
        return true;
    }
    let occupancy = mailbox.occupancy();
    if occupancy.commands != 0 || occupancy.reliable_queued != 0 || occupancy.reliable_reserved != 0
    {
        return false;
    }
    {
        let root = roots.lock().unwrap_or_else(|p| p.into_inner());
        // take_completion은 mailbox에서 제거한 뒤 이 카운터를 갱신한다. 그 짧은
        // 회수 구간을 실패나 완료로 재표시하지 않고 다음 notification을 기다린다.
        if !root.terminals_drained() {
            return false;
        }
    }
    mailbox.take_progress();
    let read_fence = || physical_fence(shutdown_deadline, roots);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        checker.quiesce_root(request.until, shutdown, &read_fence)
    }))
    .unwrap_or_else(|_| {
        checker.failed = true;
        Err(error("profiled_quiesce", "barrier_panicked"))
    });
    let mut root = roots.lock().unwrap_or_else(|p| p.into_inner());
    let result = result.and_then(|process| {
        if root.authority != request.authority
            || root.quiesce.is_none_or(|pending| {
                pending.sequence != request.sequence
                    || pending.authority != request.authority
                    || Instant::now() >= pending.until
            })
            || root.transition.is_some()
            || !root.terminals_drained()
            || process.process_identity != root.process
            || !reusable_process(process)
        {
            return Err(error(
                "profiled_quiesce",
                "late_or_incomplete_current_root_fence",
            ));
        }
        Ok(ProfiledExternalCurrentRootFence {
            source: source_for(roots, shutdown, &root, owner, clock, profile, purpose)?,
            sequence: request.sequence,
            execution_high_water: root.execution_high_water,
            native_request_high_water: checker.request_id,
            accepted_jobs: root.accepted_jobs,
            terminal_controls: root.terminal_controls,
            drained_terminals: root.drained_terminals,
            process,
        })
    });
    root.quiesce_result = Some(result);
    drop(root);
    mailbox.signal().notify();
    true
}

#[cfg(test)]
mod tests;
