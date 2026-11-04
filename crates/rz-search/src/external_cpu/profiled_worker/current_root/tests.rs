//! private 상태의 불변식 검사다. 실제 ELF·pipe·ready 관측의 성공을 주장하지 않는다.
use super::*;
use rz_contracts::{ClockDomain, GameGeneration, ProcessEpoch, RootGeneration};

fn authority() -> SearchAuthority {
    SearchAuthority {
        epoch: ProcessEpoch(7),
        game: GameGeneration(1),
        root: RootGeneration(1),
        implementation: Digest([9; 32]),
    }
}

fn state() -> (
    Arc<Mutex<RootState>>,
    Arc<AtomicBool>,
    ProfiledExternalRootSource,
) {
    let roots = Arc::new(Mutex::new(RootState::new(authority())));
    let shutdown = Arc::new(AtomicBool::new(false));
    let mut root = roots.lock().unwrap();
    root.process = Some(ExternalProcessIdentity {
        pid: 31,
        process_group: 31,
        proc_start_ticks: 1001,
    });
    let source = source_for(
        &roots,
        &shutdown,
        &root,
        CpuWorkerId(1),
        ExternalWorkerClock {
            domain: ClockDomain(authority().epoch),
            origin: Instant::now(),
        },
        ExternalEngineProfile::Stockfish19EmbeddedNnue,
        ExternalEnginePurpose::ResearchChecker,
    )
    .unwrap();
    drop(root);
    (roots, shutdown, source)
}

/// 테스트 내부의 작성 상태이며 실제 reset 발급 함수를 대체하지 않는다.
fn authored_initial_reset(root: &mut RootState, source: &ProfiledExternalRootSource) {
    let proof = ProfiledExternalResetProof {
        source: source.clone(),
        previous: None,
        sequence: 1,
    };
    root.initial_reset = Some(proof.clone());
    root.current_reset = Some(proof);
    root.reset_sequence = 1;
}

#[test]
fn pure_source_requires_recorded_initial_reset_and_current_owner() {
    let (roots, _shutdown, source) = state();
    assert!(source.validate_current().is_err());
    authored_initial_reset(&mut roots.lock().unwrap(), &source);
    assert!(source.validate_current().is_ok());
    assert_eq!(source.authority(), authority());
    assert_eq!(source.worker_owner(), CpuWorkerId(1));
    assert_eq!(source.owner(), source.worker_owner());
    roots
        .lock()
        .unwrap()
        .process
        .as_mut()
        .unwrap()
        .proc_start_ticks += 1;
    assert!(source.validate_current().is_err());
}

#[test]
fn pure_source_identity_remains_comparable_after_current_root_revocation() {
    let (roots, _shutdown, source) = state();
    authored_initial_reset(&mut roots.lock().unwrap(), &source);
    let same = source.clone();
    assert!(source.same_source(&same));
    roots.lock().unwrap().quiesce = Some(QuiesceRequest {
        authority: authority(),
        sequence: 1,
        until: Instant::now() + Duration::from_secs(1),
    });
    assert!(source.validate_current().is_err());
    assert!(source.same_source(&same));
}

#[test]
fn pure_same_numeric_process_and_root_cannot_join_another_worker() {
    let (_roots, _shutdown, source) = state();
    let (_other_roots, _other_shutdown, mut other) = state();
    other.clock = source.clock;
    assert_eq!(source.authority(), other.authority());
    assert_eq!(source.process_identity(), other.process_identity());
    assert!(!source.same_source(&other));
    assert!(!source.same_worker(&other));
}

#[test]
fn pure_source_rejects_replacement_accounting_failure_and_worker_exit() {
    let (roots, shutdown, source) = state();
    authored_initial_reset(&mut roots.lock().unwrap(), &source);
    roots.lock().unwrap().source_sequence += 1;
    assert!(source.validate_current().is_err());
    roots.lock().unwrap().source_sequence -= 1;
    roots.lock().unwrap().lifecycle_failure = Some(error("fixture", "unknown_terminal"));
    assert!(source.validate_current().is_err());
    roots.lock().unwrap().lifecycle_failure = None;
    shutdown.store(true, Ordering::Release);
    assert!(source.validate_current().is_err());
    shutdown.store(false, Ordering::Release);
    drop(roots);
    assert!(source.validate_current().is_err());
}

#[test]
fn pure_terminal_reclamation_requires_all_three_original_counts() {
    let mut root = RootState::new(authority());
    assert!(root.terminals_drained());
    root.accepted_jobs = 1;
    assert!(!root.terminals_drained());
    root.terminal_controls = 1;
    assert!(!root.terminals_drained());
    root.drained_terminals = 1;
    assert!(root.terminals_drained());
    root.lifecycle_failure = Some(error("fixture", "unobserved_cost"));
    assert!(!root.terminals_drained());
}

#[test]
fn pure_current_fence_rejects_unknown_terminals_or_changed_process() {
    let (_roots, _shutdown, source) = state();
    let mut fence = ProfiledExternalCurrentRootFence {
        process: CheckerShutdown {
            process_identity: Some(source.process_identity()),
            ..Default::default()
        },
        source,
        sequence: 1,
        execution_high_water: 3,
        native_request_high_water: 2,
        accepted_jobs: 3,
        terminal_controls: 3,
        drained_terminals: 2,
    };
    assert!(!fence.permits_reuse());
    fence.drained_terminals = 3;
    assert!(fence.permits_reuse());
    fence.process.quarantined = true;
    assert!(!fence.permits_reuse());
    fence.process.quarantined = false;
    fence
        .process
        .process_identity
        .as_mut()
        .unwrap()
        .proc_start_ticks += 1;
    assert!(!fence.permits_reuse());
}

#[test]
fn pure_reset_proof_never_follows_new_root_or_changes_initial_classification() {
    let (roots, _shutdown, source) = state();
    authored_initial_reset(&mut roots.lock().unwrap(), &source);
    let proof = roots.lock().unwrap().initial_reset.clone().unwrap();
    assert!(proof.is_initial());
    assert_eq!(proof.previous_authority(), None);
    assert!(proof.matches_source(&source));
    let mut next = source.clone();
    next.authority.root.0 += 1;
    next.source_sequence += 1;
    assert!(!proof.matches_source(&next));
    assert!(source.same_worker(&next));
    let reset = ProfiledExternalResetProof {
        source: next.clone(),
        previous: Some(source.authority()),
        sequence: 2,
    };
    assert!(!reset.is_initial());
    assert_eq!(reset.previous_authority(), Some(authority()));
    assert!(reset.matches_source(&next));
    assert!(!reset.matches_source(&source));
}

#[test]
fn pure_worker_exit_guard_closes_admission_without_physical_completion_claim() {
    let shutdown = Arc::new(AtomicBool::new(false));
    let mailbox = Mailbox::new(1, 2, CompletionSignal::default()).unwrap();
    {
        let _guard = WorkerAdmissionGuard {
            shutdown: Arc::clone(&shutdown),
            mailbox: mailbox.clone(),
        };
        assert!(!shutdown.load(Ordering::Acquire));
    }
    assert!(shutdown.load(Ordering::Acquire));
    assert!(mailbox.occupancy().closed);
}

#[test]
fn pure_process_close_requires_exit_both_drains_and_original_process() {
    let (_roots, _shutdown, source) = state();
    let mut process = CheckerShutdown {
        process_identity: Some(source.process_identity()),
        exit_observed: true,
        stdout_drained: true,
        stderr_drained: true,
        cleanup_complete: true,
        ..Default::default()
    };
    assert!(completed_process_close(process, source.process_identity()));
    for missing in 0..4 {
        let mut incomplete = process;
        match missing {
            0 => incomplete.exit_observed = false,
            1 => incomplete.stdout_drained = false,
            2 => incomplete.stderr_drained = false,
            _ => incomplete.cleanup_complete = false,
        }
        assert!(!completed_process_close(
            incomplete,
            source.process_identity()
        ));
    }
    process.process_identity.as_mut().unwrap().proc_start_ticks += 1;
    assert!(!completed_process_close(process, source.process_identity()));
}

#[test]
fn pure_closed_process_does_not_replace_terminal_reclamation_with_exit() {
    let (_roots, _shutdown, source) = state();
    let mut proof = ProfiledExternalProcessCloseProof {
        process: CheckerShutdown {
            process_identity: Some(source.process_identity()),
            exit_observed: true,
            stdout_drained: true,
            stderr_drained: true,
            cleanup_complete: true,
            ..Default::default()
        },
        source,
        accepted_jobs: 2,
        terminal_controls: 2,
        drained_terminals: 1,
    };
    assert!(!proof.permits_semantic_closure());
    proof.drained_terminals = 2;
    assert!(proof.permits_semantic_closure());
    proof.process.ownership_lost = true;
    assert!(!proof.permits_semantic_closure());
    proof.process.ownership_lost = false;
    proof.process.quarantined = true;
    assert!(!proof.permits_semantic_closure());
}
