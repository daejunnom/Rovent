//! Authored contract receipts test custody and semantic control only. These
//! tests do not execute a neural model or establish native/GPU acceptance.
use super::*;
use crate::pals::coordinator::{
    NativeAcceptanceIssuer, PhysicalRoleCompletion, RoleCompletion, RoleSubmission,
    RoleTicketRegistry,
};
use rz_contracts::pals::RepresentationKey;
use rz_contracts::{
    ClockDomain, GameGeneration, MonotonicTick, OwnerId, PrecisionProfile, ProcessEpoch,
    RootGeneration,
};
use rz_runtime::adaptive::CoordinatorPolicy;

type Port = PcSemanticPort<LegalOrderRoleMock>;
type Input = FrameInput<ReviewView, PcPortQuestion>;
type IdleArchiveOwner = super::super::idle_archive::IdleSemanticArchiveOwner<LegalOrderRoleMock>;
type IdleArchiveResult = Result<
    (
        IdleArchiveOwner,
        adaptive_port::archive_worker::ArchiveWorkReceipt,
    ),
    adaptive_port::archive_worker::ArchiveQuarantine<LegalOrderRoleMock>,
>;

struct IdleArchiveScratch(std::path::PathBuf, bool);
impl Drop for IdleArchiveScratch {
    fn drop(&mut self) {
        if self.1
            && self.0.parent() == Some(std::env::temp_dir().as_path())
            && self
                .0
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("rz-idle-archive-"))
            && std::fs::symlink_metadata(&self.0)
                .is_ok_and(|meta| meta.is_dir() && !meta.file_type().is_symlink())
        {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}
fn idle_archive_scratch() -> IdleArchiveScratch {
    static SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let root = std::env::temp_dir().join(format!(
        "rz-idle-archive-{}-{}",
        std::process::id(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&root).unwrap();
    IdleArchiveScratch(root, false)
}
fn idle_archive_limits() -> crate::pals::store::ArchiveRuntimeLimits {
    crate::pals::store::ArchiveRuntimeLimits {
        game_bytes_max: 1024 * 1024,
        global_bytes_max: 4 * 1024 * 1024,
        index_entries_max: 256,
        index_bytes_max: 512 * 1024,
        load_bytes_max: 512 * 1024,
        load_deadline_max_ms: 20_000,
        record_payload_bytes_max: 512 * 1024,
        max_load_pins: 16,
    }
}
fn idle_archive_segments(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut segments = Vec::new();
    for game in std::fs::read_dir(root).unwrap() {
        let game = game.unwrap();
        if !game.file_type().unwrap().is_dir()
            || !game.file_name().to_string_lossy().starts_with("game-")
        {
            continue;
        }
        for file in std::fs::read_dir(game.path()).unwrap() {
            let path = file.unwrap().path();
            if path.extension().is_some_and(|extension| extension == "arc") {
                segments.push(path);
            }
        }
    }
    segments.sort();
    segments
}

fn unused_setup_pc(reset: bool, deadline: Instant) -> (IdleArchiveOwner, SearchAuthority) {
    let mut owner = Box::new(
        PcSemanticOwner::new_selected(
            PalsSemanticConfig {
                beam_width: 1,
                line_plies: 2,
                max_nodes: 257,
                max_records: 128,
                max_role_calls: 128,
            },
            LegalOrderRoleMock,
            ReviewSelectionPolicy::default(),
            ReviewIndexLimits::default(),
        )
        .unwrap(),
    );
    let scope = SearchAuthority {
        game: GameGeneration(1),
        ..port_authority_for_idle_test()
    };
    if reset {
        owner
            .try_adaptive_new_game(scope, deadline, &AtomicBool::new(false))
            .unwrap();
    }
    (
        owner
            .into_idle_archive()
            .unwrap_or_else(|_| panic!("original unused PC")),
        scope,
    )
}
fn unused_setup_payload(
    root: &std::path::Path,
    limits: crate::pals::store::ArchiveRuntimeLimits,
    deadline: Instant,
    cancellation: Arc<AtomicBool>,
) -> super::super::idle_archive::UnusedIdleArchiveSetup {
    use super::super::idle_archive::{
        UnusedIdleArchiveSetup, UnusedIdleArchiveSetupControl, UnusedIdleArchiveSetupIdentity,
    };
    let held = cap_std::fs::Dir::open_ambient_dir(root, cap_std::ambient_authority()).unwrap();
    UnusedIdleArchiveSetup::new(
        crate::pals::store::ArchiveConfig::new(
            root.to_path_buf(),
            std::env::current_dir().unwrap(),
        ),
        crate::pals::store::ArchiveRootCapability::from_directory(root.to_path_buf(), held)
            .unwrap(),
        limits,
        UnusedIdleArchiveSetupIdentity {
            profile: ContractDigest([51; 32]),
            input_lock: ContractDigest([52; 32]),
            registration: ContractDigest([53; 32]),
        },
        UnusedIdleArchiveSetupControl {
            deadline,
            cancellation,
            max_io_bytes: 16 * 1024 * 1024,
        },
    )
}
fn unused_setup_completion<M: RoleModel + 'static>(
    worker: &mut adaptive_port::archive_worker::ArchiveWorker<M>,
    transfer: &adaptive_port::archive_worker::ArchiveTransfer,
    deadline: Instant,
) -> adaptive_port::archive_worker::UnusedIdleArchiveSetupCompletion<M> {
    loop {
        if let Some(completion) = worker.try_receive_unused_idle_setup(transfer).unwrap() {
            return completion;
        }
        assert!(
            Instant::now() < deadline,
            "bounded original setup completion"
        );
        std::thread::yield_now();
    }
}

#[test]
fn idle_unused_setup_preserves_first_go_readiness_then_registers_the_original_root() {
    use adaptive_port::archive_worker::{ArchiveOperation, ArchiveWorker};
    let mut scratch = idle_archive_scratch();
    let deadline = Instant::now() + Duration::from_secs(20);
    let (idle, scope) = unused_setup_pc(true, deadline);
    let store = idle.stores().read_owner();
    let registry = idle.semantic().adaptive_registry_identity();
    let setup = unused_setup_payload(
        &scratch.0,
        idle_archive_limits(),
        deadline,
        Arc::new(AtomicBool::new(false)),
    );
    let identity = setup.identity();
    let mut worker = ArchiveWorker::start().unwrap();
    let transfer = worker
        .submit_unused_idle_setup(idle, scope, setup)
        .unwrap_or_else(|rejected| panic!("setup admission: {}", rejected.error));
    let completion = unused_setup_completion(&mut worker, &transfer, deadline);
    let (idle, receipt) = completion
        .install(scope, transfer.nonce(), identity)
        .unwrap_or_else(|rejected| panic!("setup install: {:?}", rejected.failure()));
    assert!(receipt.configuration_complete);
    assert_eq!(receipt.remaining_io_bytes, Some(16 * 1024 * 1024));
    assert_eq!(
        receipt.archive_owner.unwrap().runtime_limits,
        Some(idle_archive_limits())
    );
    assert!(receipt.archive_owner.unwrap().global_scan_complete);
    assert!(store.matches(idle.stores()));
    assert!(registry.ptr_eq(&idle.semantic().adaptive_registry_identity()));
    assert_eq!(idle.stores().generation(), 0);
    assert!(idle.stores().root().is_none());
    assert_eq!(idle.semantic().last_adaptive_authority(), None);
    assert_eq!(idle.semantic().adaptive_new_game_scope(), Some(scope));
    assert_eq!(
        idle.stores().hot_stats(),
        crate::pals::store::ArchiveStats::default()
    );
    let duplicate = worker
        .submit_unused_idle_setup(
            idle,
            scope,
            unused_setup_payload(
                &scratch.0,
                idle_archive_limits(),
                deadline,
                Arc::new(AtomicBool::new(false)),
            ),
        )
        .err()
        .expect("setup is one-shot");
    assert!(store.matches(duplicate.owner.stores()));
    let mut owner = duplicate
        .owner
        .into_pc()
        .unwrap_or_else(|_| panic!("same PC kind"));
    let registration = owner
        .register_adaptive_root(&Position::startpos(), scope)
        .unwrap();
    assert_eq!(owner.stores().generation(), scope.root.0);
    assert_eq!(owner.semantic.adaptive_new_game_scope(), None);
    assert_eq!(owner.last_adaptive_authority(), Some(scope));
    owner.abandon_unconsumed_adaptive_root(scope).unwrap();
    drop(registration);
    let idle = owner
        .into_idle_archive()
        .unwrap_or_else(|_| panic!("original unconsumed root"));
    let authority = idle
        .adaptive_archive_authority(SearchAuthority {
            root: RootGeneration(scope.root.0 + 1),
            ..scope
        })
        .unwrap();
    let (_idle, _) = idle_archive_work(
        &mut worker,
        idle,
        authority,
        ArchiveOperation::FinishIdleGame,
        deadline,
    )
    .unwrap_or_else(|rejected| panic!("original checkpoint: {:?}", rejected.reason));
    worker.finish().unwrap();
    scratch.1 = true;
}

#[test]
fn idle_unused_setup_rejects_missing_reset_wrong_scope_live_pin_and_active_root() {
    use adaptive_port::archive_worker::ArchiveWorker;
    let mut scratch = idle_archive_scratch();
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut worker = ArchiveWorker::start().unwrap();
    for case in 0..4 {
        let (mut idle, mut scope) = unused_setup_pc(case != 0, deadline);
        let pin = if case == 2 {
            Some(
                idle.stores()
                    .pin_review(crate::pals::store::StorePins::default())
                    .unwrap(),
            )
        } else {
            None
        };
        if case == 1 {
            scope.root.0 += 1;
        }
        if case == 3 {
            let mut owner = idle.into_pc().unwrap_or_else(|_| panic!("unused PC kind"));
            let _registration = owner
                .register_adaptive_root(&Position::startpos(), scope)
                .unwrap();
            idle = owner
                .into_idle_archive()
                .unwrap_or_else(|_| panic!("logically idle unconsumed registration"));
        }
        let store = idle.stores().read_owner();
        let rejected = worker
            .submit_unused_idle_setup(
                idle,
                scope,
                unused_setup_payload(
                    &scratch.0,
                    idle_archive_limits(),
                    deadline,
                    Arc::new(AtomicBool::new(false)),
                ),
            )
            .err()
            .expect("non-unused source rejected before I/O");
        assert!(store.matches(rejected.owner.stores()));
        assert!(!rejected.owner.archive_enabled());
        assert!(rejected.owner.setup_snapshot().is_none());
        drop(pin);
    }
    assert!(idle_archive_segments(&scratch.0).is_empty());
    worker.finish().unwrap();
    scratch.1 = true;
}

#[test]
fn idle_unused_setup_partial_selector_scan_and_panic_keep_the_original_owner() {
    use adaptive_port::archive_worker::{
        ArchiveQuarantineReason, ArchiveWorker, UnusedIdleArchiveSetupShutdownParts,
    };
    for case in 0..3 {
        let mut scratch = idle_archive_scratch();
        let deadline = Instant::now() + Duration::from_secs(20);
        let (mut idle, scope) = unused_setup_pc(true, deadline);
        if case == 2 {
            idle.panic_setup_after_enable_for_test();
        }
        if case == 1 {
            std::fs::write(
                scratch.0.join("foreign-not-an-archive"),
                b"original scan rejection",
            )
            .unwrap();
        }
        let store = idle.stores().read_owner();
        let limits = if case == 0 {
            crate::pals::store::ArchiveRuntimeLimits {
                record_payload_bytes_max: 0,
                ..idle_archive_limits()
            }
        } else {
            idle_archive_limits()
        };
        let setup = unused_setup_payload(
            &scratch.0,
            limits,
            deadline,
            Arc::new(AtomicBool::new(false)),
        );
        let identity = setup.identity();
        let mut worker = ArchiveWorker::start().unwrap();
        let transfer = worker
            .submit_unused_idle_setup(idle, scope, setup)
            .unwrap_or_else(|rejected| panic!("unused setup admission: {}", rejected.error));
        // A later caller nonce error must not replace the owner's first error.
        let rejected = unused_setup_completion(&mut worker, &transfer, deadline)
            .install(scope, transfer.nonce() + 1, identity)
            .err()
            .expect("actual selected startup stage must fail");
        let UnusedIdleArchiveSetupShutdownParts {
            owner: idle,
            setup,
            receipt,
            result,
        } = rejected.into_owner_and_setup_for_shutdown();
        match case {
            0 => assert!(matches!(
                result,
                Err(ArchiveQuarantineReason::Operation(PalsError::Store(
                    StoreError::InvalidConditions("archive runtime selectors")
                )))
            )),
            1 => assert!(matches!(
                result,
                Err(ArchiveQuarantineReason::Operation(PalsError::Store(
                    StoreError::ArchiveIntegrity("archive root is not exclusively managed")
                )))
            )),
            2 => assert!(matches!(result, Err(ArchiveQuarantineReason::Panic))),
            _ => unreachable!("closed authored failure cases"),
        }
        assert!(store.matches(idle.stores()));
        assert!(
            idle.archive_enabled(),
            "original partial enable remains owned"
        );
        assert_eq!(setup.identity(), identity);
        assert_eq!(setup.runtime_limits(), limits);
        assert!(!receipt.configuration_complete);
        let pending = idle.setup_snapshot().unwrap();
        assert_eq!(pending.original_deadline, deadline);
        assert_eq!(
            pending.remaining_io_bytes,
            if case == 2 {
                None
            } else {
                Some(16 * 1024 * 1024)
            }
        );
        assert_eq!(pending.completion_unknown, case == 2);
        if case == 1 {
            assert_eq!(receipt.archive_owner.unwrap().runtime_limits, Some(limits));
        }
        assert!(
            idle.into_pc().is_err(),
            "partial/unknown setup cannot become a search owner"
        );
        worker.finish().unwrap();
        scratch.1 = true;
    }
}

#[test]
fn idle_unused_setup_wrong_poll_install_and_busy_cancel_preserve_original_custody() {
    use adaptive_port::archive_worker::{
        ArchiveQuarantineReason, ArchiveWorker, ArchiveWorkerError,
        UnusedIdleArchiveSetupShutdownParts,
    };
    let mut scratch = idle_archive_scratch();
    let deadline = Instant::now() + Duration::from_secs(20);
    let (idle, scope) = unused_setup_pc(true, deadline);
    let store = idle.stores().read_owner();
    let cancellation = Arc::new(AtomicBool::new(false));
    let setup = unused_setup_payload(
        &scratch.0,
        idle_archive_limits(),
        deadline,
        Arc::clone(&cancellation),
    );
    let identity = setup.identity();
    let mut worker = ArchiveWorker::start().unwrap();
    let mut other = ArchiveWorker::<LegalOrderRoleMock>::start().unwrap();
    let transfer = worker
        .submit_unused_idle_setup(idle, scope, setup)
        .unwrap_or_else(|rejected| panic!("setup admission: {}", rejected.error));
    assert!(matches!(
        worker.try_receive(&transfer),
        Err(ArchiveWorkerError::StaleTransfer)
    ));
    assert!(matches!(
        other.try_receive_unused_idle_setup(&transfer),
        Err(ArchiveWorkerError::StaleTransfer)
    ));
    let (second, _) = unused_setup_pc(true, deadline);
    let second_store = second.stores().read_owner();
    let busy = worker
        .submit_unused_idle_setup(
            second,
            scope,
            unused_setup_payload(
                &scratch.0,
                idle_archive_limits(),
                deadline,
                Arc::new(AtomicBool::new(false)),
            ),
        )
        .err()
        .expect("one worker slot retains second original owner");
    assert!(second_store.matches(busy.owner.stores()));
    assert!(busy.owner.setup_snapshot().is_none());
    let completion = unused_setup_completion(&mut worker, &transfer, deadline);
    let rejected = completion
        .install(scope, transfer.nonce() + 1, identity)
        .err()
        .expect("wrong setup nonce retains completed custody");
    let UnusedIdleArchiveSetupShutdownParts {
        owner: idle,
        setup,
        result,
        ..
    } = rejected.into_owner_and_setup_for_shutdown();
    assert!(matches!(
        result,
        Err(ArchiveQuarantineReason::InvalidReturn(
            "unused setup identity/nonce"
        ))
    ));
    assert!(store.matches(idle.stores()));
    assert!(Arc::ptr_eq(&setup.control.cancellation, &cancellation));
    assert!(idle.into_pc().is_err());
    let (idle, _) = unused_setup_pc(true, deadline);
    let cancellation = Arc::new(AtomicBool::new(false));
    let setup = unused_setup_payload(
        &scratch.0,
        idle_archive_limits(),
        deadline,
        Arc::clone(&cancellation),
    );
    let identity = setup.identity();
    let store = idle.stores().read_owner();
    let transfer = worker
        .submit_unused_idle_setup(idle, scope, setup)
        .unwrap_or_else(|rejected| panic!("next original setup: {}", rejected.error));
    // The actual completed setup is still tied to its first go's cancellation.
    // A late cancellation forbids installation, even after the worker returned.
    let completion = unused_setup_completion(&mut worker, &transfer, deadline);
    transfer.cancel();
    let rejected = completion
        .install(scope, transfer.nonce(), identity)
        .err()
        .expect("late original cancel is not success");
    let UnusedIdleArchiveSetupShutdownParts {
        owner: idle,
        setup,
        result,
        ..
    } = rejected.into_owner_and_setup_for_shutdown();
    assert!(matches!(result, Err(ArchiveQuarantineReason::Canceled)));
    assert!(store.matches(idle.stores()));
    assert!(Arc::ptr_eq(&setup.control.cancellation, &cancellation));
    assert!(cancellation.load(Ordering::Acquire));
    assert!(idle.into_pc().is_err());
    worker.finish().unwrap();
    other.finish().unwrap();
    scratch.1 = true;
}

#[test]
fn idle_unused_researcher_setup_keeps_the_deferred_source_and_unconsumed_readiness() {
    use adaptive_port::archive_worker::ArchiveWorker;
    let mut scratch = idle_archive_scratch();
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut owner = Box::new(super::super::preparing_root::tests::owner());
    let model = Arc::clone(owner.source().model());
    let scope = SearchAuthority {
        game: GameGeneration(1),
        ..port_authority_for_idle_test()
    };
    owner
        .try_adaptive_new_game(scope, deadline, &AtomicBool::new(false))
        .unwrap();
    let store = owner.stores().read_owner();
    let idle = owner
        .into_idle_archive()
        .unwrap_or_else(|_| panic!("unused original preparing owner"));
    let setup = unused_setup_payload(
        &scratch.0,
        idle_archive_limits(),
        deadline,
        Arc::new(AtomicBool::new(false)),
    );
    let identity = setup.identity();
    let mut worker = ArchiveWorker::start().unwrap();
    let transfer = worker
        .submit_unused_idle_setup(idle, scope, setup)
        .unwrap_or_else(|rejected| panic!("preparing setup: {}", rejected.error));
    let completion = unused_setup_completion(&mut worker, &transfer, deadline);
    let (idle, receipt) = completion
        .install(scope, transfer.nonce(), identity)
        .unwrap_or_else(|rejected| panic!("preparing setup return: {:?}", rejected.failure()));
    assert_eq!(
        receipt.owner_kind,
        super::super::idle_archive::IdleArchiveOwnerKind::Researcher
    );
    assert_eq!(idle.semantic().adaptive_new_game_scope(), Some(scope));
    let owner = idle
        .into_researcher()
        .unwrap_or_else(|_| panic!("same preparing Box"));
    assert!(store.matches(owner.stores()));
    assert!(Arc::ptr_eq(owner.source().model(), &model));
    assert!(owner.last_authority().is_none());
    assert!(owner.stores().root().is_none());
    assert_eq!(owner.stores().generation(), 0);
    assert_eq!(
        owner.stores().hot_stats(),
        crate::pals::store::ArchiveStats::default()
    );
    let mut idle = owner
        .into_idle_archive()
        .unwrap_or_else(|_| panic!("same configured idle"));
    assert!(idle.close_archive_owner().unwrap().unwrap().complete);
    worker.finish().unwrap();
    scratch.1 = true;
}

fn port(beam: usize, line_plies: usize) -> Port {
    port_with_observation_limit(beam, line_plies, None)
}
fn port_with_observation_limit(
    beam: usize,
    line_plies: usize,
    observations: Option<usize>,
) -> Port {
    port_with_archive(beam, line_plies, observations, None)
}
fn port_with_archive(
    beam: usize,
    line_plies: usize,
    observations: Option<usize>,
    archive: Option<(&std::path::Path, crate::pals::store::ArchiveRuntimeLimits)>,
) -> Port {
    let config = PalsSemanticConfig {
        beam_width: beam,
        line_plies,
        max_nodes: 257,
        max_records: 128,
        max_role_calls: 128,
    };
    let mut owner = PcSemanticOwner::new_selected(
        config,
        LegalOrderRoleMock,
        ReviewSelectionPolicy::default(),
        ReviewIndexLimits::default(),
    )
    .unwrap();
    if let Some(observations) = observations {
        owner.semantic.stores = PalsStores::new(StoreLimits {
            observations,
            ..StoreLimits::default()
        });
    }
    if let Some((root, limits)) = archive {
        let mut idle = Box::new(owner)
            .into_idle_archive()
            .unwrap_or_else(|_| panic!("unused original PC owner"));
        let held = cap_std::fs::Dir::open_ambient_dir(root, cap_std::ambient_authority()).unwrap();
        idle.enable_archive_with_capability(
            crate::pals::store::ArchiveConfig::new(
                root.to_path_buf(),
                std::env::current_dir().unwrap(),
            ),
            crate::pals::store::ArchiveRootCapability::from_directory(root.to_path_buf(), held)
                .unwrap(),
        )
        .unwrap();
        idle.set_archive_runtime_limits(limits).unwrap();
        owner = *idle
            .into_pc()
            .unwrap_or_else(|_| panic!("same unused PC owner"));
    }
    let authority = SearchAuthority {
        epoch: ProcessEpoch(23),
        game: GameGeneration(0),
        root: RootGeneration(7),
        implementation: ContractDigest([7; 32]),
    };
    registered_port(owner, Position::startpos(), authority)
}
fn registered_port(
    mut owner: PcSemanticOwner<LegalOrderRoleMock>,
    position: Position,
    authority: SearchAuthority,
) -> Port {
    let registration = owner.register_adaptive_root(&position, authority).unwrap();
    let next = Arc::new(std::sync::atomic::AtomicU64::new(1));
    let issuer: adaptive_port::AdaptiveRulesOwnerIssuer =
        Arc::new(move || Ok(OwnerId(next.fetch_add(1, Ordering::Relaxed))));
    let roots = [
        BoardMove::from_uci("e2e4").unwrap(),
        BoardMove::from_uci("d2d4").unwrap(),
    ];
    Port::new_registered_retaining(
        owner,
        position,
        authority,
        Deadline {
            clock: ClockDomain(authority.epoch),
            at: MonotonicTick(20_000_000_000),
        },
        Instant::now() + Duration::from_secs(20),
        ExecutionMode::CpuPc,
        issuer,
        registration,
        Some(&roots),
    )
    .map_err(|(_, error)| error)
    .unwrap()
}

fn accepted<M: RoleModel>(
    port: &mut PcSemanticPort<M>,
    input: &Input,
    request: u64,
) -> (AcceptedRoleStep, RoleOutput) {
    let mut registry = RoleTicketRegistry::<ReviewView, RoleOutput>::new(
        port.authority,
        512,
        CoordinatorPolicy::Fixed,
        request,
    )
    .unwrap();
    let (issuer, verifier) = NativeAcceptanceIssuer::new_pair();
    registry
        .register_native_acceptance_verifier(verifier)
        .unwrap();
    let role = role_for_purpose(input.question.context.purpose).unwrap();
    let representation = RepresentationKey {
        input: ContractDigest([request as u8; 32]),
        history: ContractDigest([2; 32]),
        candidates: ContractDigest([3; 32]),
        mask_positions: ContractDigest([4; 32]),
        model: ContractDigest([5; 32]),
        encoding: ContractDigest([6; 32]),
        precision: PrecisionProfile::Fp32,
        frozen_epoch: 1,
        record_revision: input.question.context.public_revision,
        role,
    };
    let ticket = registry
        .admit(
            RoleSubmission {
                question: input.question_digest,
                view: input.view_seal,
                representation,
                role,
                original_deadline: port.deadline,
            },
            Arc::clone(&input.view),
            port.deadline.clock,
            MonotonicTick(0),
        )
        .unwrap();
    // Each authored receipt starts at its distinct process request sequence.
    registry.take_dispatch().unwrap();
    port.role_submitted(input, &ticket).unwrap();
    let logits: Vec<_> = input
        .view
        .query()
        .legal
        .iter()
        .map(|movement| match movement.to_string().as_str() {
            "e2e4" => 10.0,
            "d2d4" => 9.0,
            _ => 0.0,
        })
        .collect();
    let wdl = rz_contracts::Wdl::try_new(0.25, 0.5, 0.25, 0.0).unwrap();
    let output = RoleOutput {
        id: ticket.id,
        authority: ticket.authority,
        key: representation,
        payload: if role == Role::Critic {
            RolePayload::Counterexample {
                candidate_logits: logits,
                divergence_logits: Vec::new(),
                wdl,
            }
        } else {
            RolePayload::Proposal {
                candidate_logits: logits,
                wdl,
            }
        },
        private_latent: vec![0.0; 16 * 384],
    };
    registry
        .complete_reciprocal_role(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(output.clone()),
        })
        .unwrap();
    registry
        .take_ready(port.deadline.clock, MonotonicTick(0))
        .unwrap();
    let permit = registry
        .issue_reciprocal_native_acceptance(&ticket, port.deadline.clock, MonotonicTick(0))
        .unwrap();
    let accepted = registry
        .accept_reciprocal_native(
            issuer.issue_success(permit),
            port.deadline.clock,
            MonotonicTick(0),
        )
        .unwrap();
    (accepted, output)
}

fn evidence<M: RoleModel>(
    port: &PcSemanticPort<M>,
    input: &Input,
    sequence: u64,
) -> AdaptivePortRoleEvidence {
    AdaptivePortRoleEvidence {
        execution: Some(rz_contracts::ExecutionId {
            epoch: port.authority.epoch,
            sequence,
        }),
        completed_nn_inputs: Some(1),
        measurement_complete: true,
        physical_fresh: Some(input.question.context.purpose == RoleQueryPurpose::ValueFresh),
    }
}

#[test]
fn two_pending_pc_questions_retain_their_original_views() {
    let mut port = port(2, 3);
    assert_eq!(std::mem::size_of_val(port.researcher()), 0);
    let root = port.next_role().unwrap().unwrap();
    let (proof, output) = accepted(&mut port, &root, 1);
    port.consume_accepted(&root, &proof, &output, Some(evidence(&port, &root, 1)))
        .unwrap();
    let first = port.next_role().unwrap().unwrap();
    let second = port.next_role().unwrap().unwrap();
    assert_eq!(
        first.question.context.purpose,
        RoleQueryPurpose::ReplyPolicy
    );
    assert_eq!(
        second.question.context.purpose,
        RoleQueryPurpose::ReplyPolicy
    );
    assert_ne!(
        first.question.context.prefix,
        second.question.context.prefix
    );
    assert_ne!(first.view_seal, second.view_seal);
    assert!(port.pending.contains_key(&first.question_digest));
    assert!(port.pending.contains_key(&second.question_digest));
    assert!(first.view.validates(port.owner.stores()));
    assert!(second.view.validates(port.owner.stores()));
    // Publishing a newer source does not relabel the older submitted view.
    let old_scope = first.view.record_scope_digest();
    assert_eq!(old_scope, first.view.record_scope_digest());
    assert!(
        port.owner
            .semantic
            .nodes
            .iter()
            .all(|node| node.evidence.is_none() && node.resume.is_none())
    );
}

#[test]
fn idle_archive_worker_keeps_actual_pc_projection_and_restores_original_relations() {
    use crate::pals::engine::adaptive_port::archive_worker::{
        ArchiveOperation, ArchiveOwner, ArchiveWork, ArchiveWorker,
    };
    use crate::pals::review_context::ReviewQuerySequences;
    use crate::pals::store::{ArchiveConfig, ArchiveRuntimeLimits};

    let record_identity = |record: &RoleRecord| {
        (
            record.revision,
            record.parent_revision,
            record.supersedes_revision,
            record.origin_state,
            record.kind,
            record.line.clone(),
            record.value,
            record.completed_depth,
            record.score_scope,
            record.cpu_observation,
            record.perspective,
            record.critical,
        )
    };

    let mut scratch = idle_archive_scratch();
    // Select the actual runtime profile before the original root starts. Keep
    // this fixture's original ArchiveConfig quotas and 16MiB load/I/O bound.
    let config = ArchiveConfig::new(scratch.0.clone(), std::env::current_dir().unwrap());
    let limits = ArchiveRuntimeLimits {
        game_bytes_max: config.game_bytes,
        global_bytes_max: config.global_bytes,
        record_payload_bytes_max: config.record_bytes,
        load_bytes_max: 16 * 1024 * 1024,
        ..idle_archive_limits()
    };
    let mut owner =
        idle_pc_from_actual_counter(port_with_archive(1, 2, None, Some((&scratch.0, limits))));
    assert_eq!(std::mem::size_of_val(owner.researcher()), 0);
    assert_eq!(owner.stores().review_pin_count().unwrap(), 0);
    let original_owner = owner.stores().read_owner();
    let original = Position::startpos();
    let state = owner.semantic.records[0].origin_state;
    let records = owner.records().to_vec();
    assert_eq!(records[1].parent_revision, Some(records[0].revision));
    assert_eq!(records[1].kind, RecordKind::Counterexample);
    let bundle = owner
        .semantic
        .review_index
        .as_ref()
        .unwrap()
        .bundle_for_revision(records[1].revision)
        .unwrap();
    assert_eq!(
        owner
            .semantic
            .review_index
            .as_ref()
            .unwrap()
            .bundle(bundle)
            .unwrap()
            .protected_consumptions(),
        2
    );
    // Exact, unrelated Rules states provide real hot pressure outside the next
    // root graph. Current selected evidence remains independently protected.
    for fullmove in 100..310 {
        let position =
            Position::from_fen(&format!("8/7k/8/8/8/8/K7/8 w - - 0 {fullmove}")).unwrap();
        owner.intern_hot(position).unwrap();
    }
    let nodes_before = owner.semantic.nodes.len();
    let query = || ReviewQuerySequences {
        purpose: RoleQueryPurpose::ValueFresh,
        prefix: &[],
        proposal: &[],
        counterexample: None,
        legal: &[],
        divergences: &[],
    };
    let policy = owner.review_policy().unwrap();
    let view = owner
        .semantic
        .review_index
        .as_mut()
        .unwrap()
        .select(&owner.semantic.stores, state, query(), policy)
        .unwrap();
    let digest = view.digest();
    let metadata = view.records_metadata().to_vec();
    let input_records = view.records().to_vec();
    drop(view);
    let idle = owner
        .into_idle_archive()
        .unwrap_or_else(|_| panic!("quiescent PC owner"));
    let next = SearchAuthority {
        root: RootGeneration(9),
        ..port_authority_for_idle_test()
    };
    let authority = idle.adaptive_archive_authority(next).unwrap();
    let mut position = original.clone();
    position.make_move("e2e4".parse().unwrap()).unwrap();
    let mut worker = ArchiveWorker::start().unwrap();
    let (idle, missing) = idle_archive_work(
        &mut worker,
        idle,
        authority,
        ArchiveOperation::RestoreRootIfArchived {
            position: original.clone(),
        },
        Instant::now() + Duration::from_secs(20),
    )
    .unwrap_or_else(|_| panic!("normal exact cold lookup miss"));
    assert!(missing.loaded_root.is_none());
    assert!(original_owner.matches(idle.stores()));
    assert_eq!(idle.semantic().records.len(), records.len());
    let work = ArchiveWork {
        operation: ArchiveOperation::PrepareRoot { position },
        deadline: Instant::now() + Duration::from_secs(20),
        max_io_bytes: 16 * 1024 * 1024,
    };
    let transfer = worker
        .submit(ArchiveOwner::Idle(idle), authority, work)
        .unwrap_or_else(|_| panic!("idle archive admitted"));
    let until = Instant::now() + Duration::from_secs(20);
    let completion = loop {
        if let Some(completion) = worker.try_receive(&transfer).unwrap() {
            break completion;
        }
        assert!(Instant::now() < until, "bounded real archive completion");
        std::thread::yield_now();
    };
    let (ArchiveOwner::Idle(idle), receipt) = completion
        .install(authority, transfer.nonce())
        .unwrap_or_else(|_| panic!("same idle owner return"))
    else {
        panic!("wrong typed owner");
    };
    assert!(receipt.io_bytes.is_some_and(|bytes| bytes > 0));
    assert!(original_owner.matches(idle.stores()));
    let mut owner = idle
        .into_pc()
        .unwrap_or_else(|_| panic!("original PC type"));
    assert!(owner.semantic.nodes.len() < nodes_before);
    assert!(owner.records().is_empty());
    assert_eq!(owner.semantic.revision, records[1].revision);
    assert_eq!(
        owner
            .semantic
            .review_index
            .as_ref()
            .unwrap()
            .bundle(bundle)
            .unwrap()
            .protected_consumptions(),
        2
    );
    let view = owner
        .semantic
        .review_index
        .as_mut()
        .unwrap()
        .select(&owner.semantic.stores, state, query(), policy)
        .unwrap();
    assert_eq!(view.digest(), digest);
    assert_eq!(view.records_metadata(), metadata);
    assert_eq!(
        view.records()
            .iter()
            .map(record_identity)
            .collect::<Vec<_>>(),
        input_records
            .iter()
            .map(record_identity)
            .collect::<Vec<_>>()
    );
    drop(view);
    let idle = owner
        .into_idle_archive()
        .unwrap_or_else(|_| panic!("same returned PC owner"));
    let authority = idle.adaptive_archive_authority(next).unwrap();
    let transfer = worker
        .submit(
            ArchiveOwner::Idle(idle),
            authority,
            ArchiveWork {
                operation: ArchiveOperation::RestoreRootIfArchived {
                    position: original.clone(),
                },
                deadline: Instant::now() + Duration::from_secs(20),
                max_io_bytes: 16 * 1024 * 1024,
            },
        )
        .unwrap_or_else(|_| panic!("original root restoration admitted"));
    let until = Instant::now() + Duration::from_secs(20);
    let completion = loop {
        if let Some(completion) = worker.try_receive(&transfer).unwrap() {
            break completion;
        }
        assert!(Instant::now() < until, "bounded original root restoration");
        std::thread::yield_now();
    };
    let (ArchiveOwner::Idle(idle), receipt) = completion
        .install(authority, transfer.nonce())
        .unwrap_or_else(|_| panic!("same restored owner"))
    else {
        panic!("wrong restored type");
    };
    assert!(receipt.loaded_root.is_some());
    assert!(original_owner.matches(idle.stores()));
    let owner = idle
        .into_pc()
        .unwrap_or_else(|_| panic!("same restored PC owner"));
    assert_eq!(
        owner
            .records()
            .iter()
            .map(record_identity)
            .collect::<Vec<_>>(),
        records.iter().map(record_identity).collect::<Vec<_>>()
    );
    assert_eq!(owner.stores().review_pin_count().unwrap(), 0);
    assert_eq!(
        owner
            .semantic
            .review_index
            .as_ref()
            .unwrap()
            .bundle(bundle)
            .unwrap()
            .protected_consumptions(),
        2
    );
    let idle = owner
        .into_idle_archive()
        .unwrap_or_else(|_| panic!("restored idle"));
    let authority = idle.adaptive_archive_authority(next).unwrap();
    let (idle, receipt) = idle_archive_work(
        &mut worker,
        idle,
        authority,
        ArchiveOperation::FinishIdleGame,
        Instant::now() + Duration::from_secs(20),
    )
    .unwrap_or_else(|rejected| {
        panic!(
            "original final checkpoint and close: reason={:?}; receipt={:?}",
            rejected.reason, rejected.receipt
        )
    });
    assert!(receipt.idle_checkpoint.is_some());
    assert!(
        idle.archive_owner_snapshot()
            .unwrap()
            .unwrap()
            .admission_closed
    );
    worker.finish().unwrap();
    scratch.1 = true;
}

#[allow(clippy::result_large_err)]
fn idle_archive_work(
    worker: &mut adaptive_port::archive_worker::ArchiveWorker<LegalOrderRoleMock>,
    idle: IdleArchiveOwner,
    authority: SearchAuthority,
    operation: adaptive_port::archive_worker::ArchiveOperation,
    deadline: Instant,
) -> IdleArchiveResult {
    use adaptive_port::archive_worker::{ArchiveOwner, ArchiveWork};
    let transfer = worker
        .submit(
            ArchiveOwner::Idle(idle),
            authority,
            ArchiveWork {
                operation,
                deadline,
                max_io_bytes: 16 * 1024 * 1024,
            },
        )
        .unwrap_or_else(|rejected| panic!("original idle command admitted: {:?}", rejected.error));
    let completion = loop {
        if let Some(completion) = worker.try_receive(&transfer).unwrap() {
            break completion;
        }
        assert!(Instant::now() < deadline, "bounded original idle command");
        std::thread::yield_now();
    };
    completion
        .install(authority, transfer.nonce())
        .map(|(owner, receipt)| {
            let ArchiveOwner::Idle(owner) = owner else {
                panic!("original idle owner type");
            };
            (owner, receipt)
        })
}

fn prepare_idle_archive_for_transition(
    worker: &mut adaptive_port::archive_worker::ArchiveWorker<LegalOrderRoleMock>,
    idle: IdleArchiveOwner,
    authority: SearchAuthority,
    deadline: Instant,
) -> IdleArchiveOwner {
    let store = idle.stores().read_owner();
    let hot = idle.stores().hot_stats();
    let revisions: Vec<_> = idle
        .semantic()
        .records
        .iter()
        .map(|record| record.revision)
        .collect();
    // The selected owner needs its real bounded usage scan before NewGame
    // admission. No hot pressure exists here, so this original worker command
    // must preserve every source fact while establishing actual completeness.
    let (idle, receipt) = idle_archive_work(
        worker,
        idle,
        authority,
        adaptive_port::archive_worker::ArchiveOperation::PrepareRoot {
            position: Position::startpos(),
        },
        deadline,
    )
    .unwrap_or_else(|rejected| {
        panic!(
            "original bounded archive startup scan: reason={:?}; receipt={:?}",
            rejected.reason, rejected.receipt
        )
    });
    assert!(store.matches(idle.stores()));
    assert_eq!(idle.stores().hot_stats(), hot);
    assert_eq!(
        idle.semantic()
            .records
            .iter()
            .map(|record| record.revision)
            .collect::<Vec<_>>(),
        revisions
    );
    assert_eq!(receipt.io_bytes, Some(0));
    assert!(receipt.idle_checkpoint.is_none());
    let measured = idle.archive_owner_snapshot().unwrap().unwrap();
    assert!(
        measured.global_scan_complete
            && measured.global_scan_after_last_commit
            && measured.global_managed_bytes == Some(0)
            && measured.complete
            && !measured.admission_closed
    );
    idle
}

fn port_authority_for_idle_test() -> SearchAuthority {
    SearchAuthority {
        epoch: ProcessEpoch(23),
        game: GameGeneration(0),
        root: RootGeneration(7),
        implementation: ContractDigest([7; 32]),
    }
}

fn idle_pc_after_actual_counter() -> Box<PcSemanticOwner<LegalOrderRoleMock>> {
    // Reuse the actual registry/accepted-head producer: no fabricated public
    // record, researcher, TT, R worker, or neural execution is needed.
    idle_pc_from_actual_counter(port(1, 2))
}
fn idle_pc_from_actual_counter(mut port: Port) -> Box<PcSemanticOwner<LegalOrderRoleMock>> {
    for request in 1..=8 {
        let input = port.next_role().unwrap().expect("real PC question");
        let (accepted, output) = accepted(&mut port, &input, request);
        port.consume_accepted(
            &input,
            &accepted,
            &output,
            Some(evidence(&port, &input, request)),
        )
        .unwrap();
        if port.owner.records().len() == 2 {
            break;
        }
    }
    assert_eq!(port.owner.records().len(), 2);
    port.revoke();
    assert!(port.take_repair_report().unwrap().is_none());
    Box::new(port.into_owner())
}

#[test]
fn idle_archive_rejections_keep_original_pc_owner_and_do_not_enable_storage() {
    use crate::pals::engine::adaptive_port::archive_worker::{
        ArchiveOperation, ArchiveOwner, ArchiveWork, ArchiveWorker,
    };
    let owner = idle_pc_after_actual_counter();
    let store = owner.stores().read_owner();
    let mut idle = owner
        .into_idle_archive()
        .unwrap_or_else(|_| panic!("original idle PC"));
    let authority = port_authority_for_idle_test();
    let mut worker = ArchiveWorker::start().unwrap();
    for case in 0..4 {
        let operation = match case {
            1 => ArchiveOperation::NewGame {
                next_authority: SearchAuthority {
                    game: GameGeneration(1),
                    root: RootGeneration(9),
                    ..authority
                },
            },
            2 => ArchiveOperation::ReclaimHot {
                incoming_pins: Default::default(),
            },
            _ => ArchiveOperation::PrepareRoot {
                position: Position::startpos(),
            },
        };
        let submitted = if case == 0 {
            SearchAuthority {
                root: RootGeneration(8),
                ..authority
            }
        } else {
            authority
        };
        let rejected = worker
            .submit(
                ArchiveOwner::Idle(idle),
                submitted,
                ArchiveWork {
                    operation,
                    deadline: Instant::now() + Duration::from_secs(20),
                    max_io_bytes: 16 * 1024 * 1024,
                },
            )
            .err()
            .expect("exact idle rejection");
        match case {
            0 => assert!(matches!(
                rejected.error,
                PalsError::Store(StoreError::StaleConsumer)
            )),
            2 => assert!(matches!(
                rejected.error,
                PalsError::Store(StoreError::InvalidConditions(_))
            )),
            _ => assert!(matches!(
                rejected.error,
                PalsError::Store(StoreError::ArchiveDisabled)
            )),
        }
        let ArchiveOwner::Idle(returned) = rejected.owner else {
            panic!("original idle type lost");
        };
        assert!(store.matches(returned.stores()));
        assert_eq!(returned.semantic().records.len(), 2);
        assert!(!returned.archive_enabled());
        assert_eq!(returned.stores().review_pin_count().unwrap(), 0);
        idle = returned;
    }
    worker.finish().unwrap();
}

#[test]
fn idle_newgame_checkpoints_actual_pc_and_orphan_raw_before_unique_ledger_transfer() {
    use crate::pals::store::{TaskAdmission, TaskConsumer, TaskKey, TaskQuestion, TaskStatus};
    use adaptive_port::archive_worker::{ArchiveOperation, ArchiveWorker};
    let mut scratch = idle_archive_scratch();
    let mut owner = idle_pc_from_actual_counter(port_with_archive(
        1,
        2,
        None,
        Some((&scratch.0, idle_archive_limits())),
    ));
    let old_store = owner.stores().read_owner();
    let records = owner.records().to_vec();
    let state = records[0].origin_state;
    let orphan = owner
        .semantic
        .stores
        .append_observation(Observation {
            state,
            line: None,
            source: 9,
            epoch: 0,
            scope: EvidenceScope::Model {
                model: 1,
                encoding: 2,
                input: 0,
            },
            score: RawScore::Unknown,
            value_identity: None,
            checker_identity: None,
            checker_work: None,
            external_report: None,
            model_value_identity: None,
            model_value_input: None,
            cpu_condition: None,
            cpu_pv: None,
            budget: 0,
            kind: ObservationKind::Proposal,
            supersedes: None,
            execution: None,
        })
        .unwrap();
    // This authored model task is cancelled and failed without a neural call,
    // accepted role step or consumption. It is raw evidence, never a dummy R.
    let situation = owner.stores().root().unwrap();
    let generation = owner.stores().generation();
    let TaskAdmission::Start(execution) = owner
        .semantic
        .stores
        .request_task(
            TaskKey {
                state,
                line: None,
                question: TaskQuestion::ModelProposal,
                root_moves: vec![],
                model: 1,
                epoch: 0,
                value_identity: None,
                checker_identity: None,
                cpu_condition: None,
                profile: 1,
                condition: 1,
                input_revision: 0,
                requested_depth: 0,
                node_budget: 0,
            },
            TaskConsumer {
                id: 81,
                situation,
                revision: 0,
                generation,
                deadline_tick: 100,
            },
            0,
        )
        .unwrap()
    else {
        panic!("original authored raw model task");
    };
    owner
        .semantic
        .stores
        .tasks
        .cancel_consumer(execution, 81)
        .unwrap();
    owner.semantic.stores.tasks.fail(execution).unwrap();
    assert_eq!(
        owner.stores().tasks.get(execution).unwrap().status,
        TaskStatus::Failed
    );
    let raw_task = serde_json::to_value(owner.stores().tasks.get(execution).unwrap()).unwrap();
    let hot = owner.stores().hot_stats();
    let before = owner.stores().archive_owner_snapshot().unwrap().unwrap();
    let idle = owner
        .into_idle_archive()
        .unwrap_or_else(|_| panic!("actual idle original PC"));
    let next = SearchAuthority {
        game: GameGeneration(1),
        root: RootGeneration(9),
        ..port_authority_for_idle_test()
    };
    let authority = idle.adaptive_archive_authority(next).unwrap();
    let mut worker = ArchiveWorker::start().unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    let idle = prepare_idle_archive_for_transition(&mut worker, idle, authority, deadline);
    let (idle, receipt) = idle_archive_work(
        &mut worker,
        idle,
        authority,
        ArchiveOperation::NewGame {
            next_authority: next,
        },
        deadline,
    )
    .unwrap_or_else(|rejected| {
        panic!(
            "durable final game checkpoint and unique replacement: reason={:?}; receipt={:?}",
            rejected.reason, rejected.receipt
        )
    });
    let checkpoint = receipt.idle_checkpoint.unwrap();
    assert_eq!(checkpoint.hot, hot);
    assert_eq!(checkpoint.records, 2);
    assert!(checkpoint.committed_bytes > 0);
    assert_eq!(receipt.next_authority, Some(next));
    assert!(receipt.loaded_root.is_none());
    assert!(receipt.io_bytes.is_some_and(|bytes| bytes > 0));
    assert!(!old_store.matches(idle.stores()));
    let closed = idle.semantic().archive_state.previous_closed_owner.unwrap();
    let actual = idle.archive_owner_snapshot().unwrap().unwrap();
    assert_eq!(closed.owner_id, before.owner_id);
    assert!(closed.admission_closed && closed.complete);
    assert_ne!(actual.owner_id, closed.owner_id);
    assert_eq!(actual.root_sha256, closed.root_sha256);
    assert_eq!(actual.runtime_limits, closed.runtime_limits);
    assert_eq!(
        actual.session_write_ledger_id,
        closed.session_write_ledger_id
    );
    assert_eq!(
        actual.session_write_bytes_max,
        closed.session_write_bytes_max
    );
    assert_eq!(
        actual.session_write_bytes_consumed,
        closed.session_write_bytes_consumed
    );
    assert!(
        actual
            .session_write_bytes_consumed
            .is_some_and(|bytes| bytes > 0)
    );

    // Inspect only this fixture's actual committed segment. The unchanged codec
    // preserves original task/unknown data and P/C parent IDs across old close.
    let segments = idle_archive_segments(&scratch.0);
    assert_eq!(segments.len(), 1);
    let bytes = std::fs::read(&segments[0]).unwrap();
    assert_eq!(bytes.len() as u64, checkpoint.committed_bytes);
    let wire: serde_json::Value = serde_json::from_slice(&bytes[56..]).unwrap();
    let tasks = wire["tasks"].as_array().unwrap();
    let task = tasks
        .iter()
        .find(|row| row[0] == serde_json::json!(execution))
        .unwrap();
    assert_eq!(task[1], raw_task);
    assert_eq!(task[1]["consumers"][0]["cancelled"], true);
    assert_eq!(task[1]["consumers"][0]["consumed"], false);
    let raw = wire["observations"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row[0] == serde_json::json!(orphan))
        .unwrap();
    assert_eq!(raw[1]["score"], "Unknown");
    assert!(raw[1]["execution"].is_null());
    assert_eq!(
        wire["engine_records"][1]["parent_revision"],
        serde_json::json!(records[0].revision)
    );

    let owner = idle
        .into_pc()
        .unwrap_or_else(|_| panic!("same PC kind after newgame"));
    assert!(owner.records().is_empty());
    assert_eq!(std::mem::size_of_val(owner.researcher()), 0);
    let owner = idle_pc_from_actual_counter(registered_port(*owner, Position::startpos(), next));
    assert_eq!(owner.records().len(), 2);
    let mut idle = owner
        .into_idle_archive()
        .unwrap_or_else(|_| panic!("second original game idle"));
    assert!(matches!(
        idle.close_archive_owner(),
        Err(PalsError::Store(StoreError::InvalidConditions(_)))
    ));
    let following = SearchAuthority {
        root: RootGeneration(11),
        ..next
    };
    let authority = idle.adaptive_archive_authority(following).unwrap();
    let (idle, finish) = idle_archive_work(
        &mut worker,
        idle,
        authority,
        ArchiveOperation::FinishIdleGame,
        Instant::now() + Duration::from_secs(20),
    )
    .unwrap_or_else(|_| panic!("second game final checkpoint and close"));
    assert_eq!(finish.idle_checkpoint.unwrap().records, 2);
    assert!(finish.next_authority.is_none());
    assert!(finish.loaded_root.is_none());
    let idle = idle
        .into_pc()
        .err()
        .expect("finished archive owner cannot resume search");
    let terminal = idle.transition_snapshot().unwrap().unwrap();
    assert!(terminal.completed && !terminal.completion_unknown);
    assert!(terminal.original_closed.unwrap().admission_closed);
    assert_eq!(std::fs::read(&segments[0]).unwrap(), bytes);
    worker.finish().unwrap();
    scratch.1 = true;
}

#[test]
fn idle_newgame_cancel_at_actual_close_keeps_one_checkpoint_and_unspent_original_bound() {
    use adaptive_port::archive_worker::{
        ArchiveOperation, ArchiveOwner, ArchiveWork, ArchiveWorker,
    };
    let mut scratch = idle_archive_scratch();
    let owner = idle_pc_from_actual_counter(port_with_archive(
        1,
        2,
        None,
        Some((&scratch.0, idle_archive_limits())),
    ));
    let old = owner.stores().read_owner();
    let idle = owner
        .into_idle_archive()
        .unwrap_or_else(|_| panic!("original PC"));
    let next = SearchAuthority {
        game: GameGeneration(1),
        root: RootGeneration(9),
        ..port_authority_for_idle_test()
    };
    let authority = idle.adaptive_archive_authority(next).unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut worker = ArchiveWorker::start().unwrap();
    let mut idle = prepare_idle_archive_for_transition(&mut worker, idle, authority, deadline);
    idle.cancel_after_close_for_test();
    let rejected = idle_archive_work(
        &mut worker,
        idle,
        authority,
        ArchiveOperation::NewGame {
            next_authority: next,
        },
        deadline,
    )
    .err()
    .expect("real close precedes scheduled cancellation");
    let first_checkpoint = Arc::clone(rejected.receipt.idle_checkpoint.as_ref().unwrap());
    let ArchiveOwner::Idle(mut idle) = rejected
        .into_owner_for_shutdown()
        .unwrap_or_else(|_| panic!("same closed transition retained"))
    else {
        panic!("idle kind");
    };
    assert!(old.matches(idle.stores()));
    let pending = idle.transition_snapshot().unwrap().unwrap();
    assert!(!pending.completed && !pending.completion_unknown);
    assert!(pending.original_closed.unwrap().admission_closed);
    assert!(pending.replacement.is_some());
    assert!(pending.remaining_io_bytes < 16 * 1024 * 1024);
    assert_eq!(pending.original_deadline, deadline);
    idle = idle
        .into_pc()
        .err()
        .expect("closed transition cannot become a searchable PC owner");
    let extended = worker
        .submit(
            ArchiveOwner::Idle(idle),
            authority,
            ArchiveWork {
                operation: ArchiveOperation::NewGame {
                    next_authority: next,
                },
                deadline: deadline + Duration::from_secs(1),
                max_io_bytes: 16 * 1024 * 1024,
            },
        )
        .err()
        .expect("new command cannot extend original transition");
    assert!(matches!(
        extended.error,
        PalsError::Store(StoreError::StaleConsumer)
    ));
    let ArchiveOwner::Idle(idle) = extended.owner else {
        panic!("same retained owner");
    };
    let (idle, retried) = idle_archive_work(
        &mut worker,
        idle,
        authority,
        ArchiveOperation::NewGame {
            next_authority: next,
        },
        deadline,
    )
    .unwrap_or_else(|rejected| {
        panic!(
            "same original transition resumes without another checkpoint: reason={:?}; receipt={:?}",
            rejected.reason, rejected.receipt
        )
    });
    assert_eq!(retried.idle_checkpoint, Some(Arc::clone(&first_checkpoint)));
    assert_eq!(retried.io_bytes, Some(0));
    assert_eq!(idle.transition_remaining_io(), pending.remaining_io_bytes);
    assert_ne!(
        idle.archive_owner_snapshot().unwrap().unwrap().owner_id,
        first_checkpoint.owner.owner_id
    );
    // The replacement is empty: no remaining hot data can be lost at close.
    let mut idle = idle;
    idle.close_archive_owner().unwrap();
    worker.finish().unwrap();
    scratch.1 = true;
}

#[test]
fn idle_final_checkpoint_quota_failure_keeps_actual_hot_owner_and_pending_game() {
    use adaptive_port::archive_worker::{ArchiveOperation, ArchiveOwner, ArchiveWorker};
    let mut scratch = idle_archive_scratch();
    let limits = crate::pals::store::ArchiveRuntimeLimits {
        record_payload_bytes_max: 1,
        ..idle_archive_limits()
    };
    let owner =
        idle_pc_from_actual_counter(port_with_archive(1, 2, None, Some((&scratch.0, limits))));
    let store = owner.stores().read_owner();
    let hot = owner.stores().hot_stats();
    let revisions: Vec<_> = owner
        .records()
        .iter()
        .map(|record| record.revision)
        .collect();
    let idle = owner
        .into_idle_archive()
        .unwrap_or_else(|_| panic!("original PC before quota"));
    let next = SearchAuthority {
        game: GameGeneration(1),
        root: RootGeneration(9),
        ..port_authority_for_idle_test()
    };
    let authority = idle.adaptive_archive_authority(next).unwrap();
    let mut worker = ArchiveWorker::start().unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    let idle = prepare_idle_archive_for_transition(&mut worker, idle, authority, deadline);
    let rejected = idle_archive_work(
        &mut worker,
        idle,
        authority,
        ArchiveOperation::NewGame {
            next_authority: next,
        },
        deadline,
    )
    .err()
    .expect("actual final metadata exceeds selected segment quota");
    assert!(matches!(
        rejected.reason,
        adaptive_port::archive_worker::ArchiveQuarantineReason::Operation(PalsError::Store(
            StoreError::ArchiveQuota("archive segment bytes")
        ))
    ));
    assert!(rejected.receipt.idle_checkpoint.is_none());
    let ArchiveOwner::Idle(idle) = rejected
        .into_owner_for_shutdown()
        .unwrap_or_else(|_| panic!("failed checkpoint keeps source"))
    else {
        panic!("idle kind");
    };
    assert!(store.matches(idle.stores()));
    assert_eq!(idle.stores().hot_stats(), hot);
    assert_eq!(
        idle.semantic()
            .records
            .iter()
            .map(|record| record.revision)
            .collect::<Vec<_>>(),
        revisions
    );
    let pending = idle.transition_snapshot().unwrap().unwrap();
    assert!(
        pending.checkpoint.is_none()
            && pending.replacement.is_none()
            && pending.original_closed.is_none()
    );
    assert!(
        !idle
            .archive_owner_snapshot()
            .unwrap()
            .unwrap()
            .admission_closed
    );
    assert!(idle.into_pc().is_err());
    worker.finish().unwrap();
    // Serialization failed before any commit/file owner; only this bounded
    // authored fixture is torn down after verifying its retained source.
    scratch.1 = true;
}

#[test]
fn idle_optional_restore_keeps_missing_metadata_and_corrupt_segments_as_errors() {
    use crate::pals::store::{ArchiveIoBudget, StorePins};
    use adaptive_port::archive_worker::{
        ArchiveOperation, ArchiveOwner, ArchiveQuarantineReason, ArchiveWorker,
    };
    for corrupt in [false, true] {
        let mut scratch = idle_archive_scratch();
        let mut owner = idle_pc_from_actual_counter(port_with_archive(
            1,
            2,
            None,
            Some((&scratch.0, idle_archive_limits())),
        ));
        let state = owner.records()[0].origin_state;
        let store = owner.stores().read_owner();
        let hot = owner.stores().hot_stats();
        let revision = owner.semantic.revision;
        let mut budget = ArchiveIoBudget {
            deadline: Instant::now() + Duration::from_secs(20),
            max_bytes: 16 * 1024 * 1024,
        };
        if corrupt {
            owner
                .semantic
                .checkpoint_idle_archive_game(&mut budget)
                .unwrap();
            let segments = idle_archive_segments(&scratch.0);
            assert_eq!(segments.len(), 1);
            let mut bytes = std::fs::read(&segments[0]).unwrap();
            // Mutate this actual fixture segment after its checked commit.
            // The original checksum/owner guard must fail, never return None.
            bytes[56] ^= 1;
            std::fs::write(&segments[0], bytes).unwrap();
        } else {
            // A real cold state may exist without an archived engine root.
            // Preserve that specific missing-metadata cause rather than erase it.
            owner
                .semantic
                .stores
                .archive_engine_records_with_pins_and_budget(
                    &[],
                    &[],
                    StorePins {
                        states: std::collections::BTreeSet::from([state]),
                        ..Default::default()
                    },
                    &mut budget,
                )
                .unwrap();
        }
        let idle = owner
            .into_idle_archive()
            .unwrap_or_else(|_| panic!("same original PC"));
        let next = SearchAuthority {
            root: RootGeneration(9),
            ..port_authority_for_idle_test()
        };
        let authority = idle.adaptive_archive_authority(next).unwrap();
        let mut worker = ArchiveWorker::start().unwrap();
        let rejected = idle_archive_work(
            &mut worker,
            idle,
            authority,
            ArchiveOperation::RestoreRootIfArchived {
                position: Position::startpos(),
            },
            Instant::now() + Duration::from_secs(20),
        )
        .err()
        .expect("original lookup error is not a miss");
        if corrupt {
            assert!(matches!(
                rejected.reason,
                ArchiveQuarantineReason::Operation(PalsError::Store(StoreError::ArchiveIntegrity(
                    _
                )))
            ));
        } else {
            assert!(matches!(
                rejected.reason,
                ArchiveQuarantineReason::Operation(PalsError::Store(StoreError::ColdRecord(
                    "archived engine metadata"
                )))
            ));
        }
        assert!(rejected.receipt.loaded_root.is_none());
        let ArchiveOwner::Idle(idle) = rejected
            .into_owner_for_shutdown()
            .unwrap_or_else(|_| panic!("original failed source retained"))
        else {
            panic!("idle kind");
        };
        assert!(store.matches(idle.stores()));
        assert_eq!(idle.stores().hot_stats(), hot);
        assert_eq!(idle.semantic().revision, revision);
        assert_eq!(idle.semantic().records.len(), 2);
        worker.finish().unwrap();
        scratch.1 = true;
    }
}

#[test]
fn forged_question_and_late_acceptance_do_not_consume_original_view() {
    let mut port = port(1, 2);
    let input = port.next_role().unwrap().unwrap();
    let (proof, output) = accepted(&mut port, &input, 1);
    let mut forged = input.clone();
    Arc::make_mut(&mut forged.question).context.public_revision += 1;
    assert!(
        port.validate_role(&forged, proof.ticket(), &output)
            .is_err()
    );
    assert_eq!(port.counters.consumed_role_outputs, 0);
    port.revoke();
    assert!(
        port.consume_accepted(&input, &proof, &output, None)
            .is_err()
    );
    assert_eq!(port.counters.consumed_role_outputs, 0);
    port.role_discarded(&input, Some(proof.ticket())).unwrap();
    assert!(port.pending.is_empty());
}

#[test]
fn exact_accepted_retry_is_not_a_second_semantic_consumption() {
    let mut port = port(1, 2);
    let input = port.next_role().unwrap().unwrap();
    let (proof, output) = accepted(&mut port, &input, 1);
    port.consume_accepted(&input, &proof, &output, None)
        .unwrap();
    let before = port.counters;
    let records = port.owner.records().len();
    port.consume_accepted(&input, &proof, &output, None)
        .unwrap();
    assert_eq!(before, port.counters);
    assert_eq!(records, port.owner.records().len());
    let mut changed = output.clone();
    if let RolePayload::Proposal {
        candidate_logits, ..
    } = &mut changed.payload
    {
        candidate_logits[0] += 1.0;
    }
    assert!(
        port.consume_accepted(&input, &proof, &changed, None)
            .is_err()
    );
    assert_eq!(before, port.counters);
}

#[test]
fn role_admission_cap_does_not_revoke_an_already_submitted_question() {
    let mut port = port(2, 2);
    port.owner.config.max_role_calls = 2;
    port.owner.profile.semantic.max_role_calls = 2;
    let root = port.next_role().unwrap().unwrap();
    let (root_proof, root_output) = accepted(&mut port, &root, 1);
    port.consume_accepted(&root, &root_proof, &root_output, None)
        .unwrap();
    let child = port.next_role().unwrap().unwrap();
    let (child_proof, child_output) = accepted(&mut port, &child, 2);
    assert!(port.next_role().unwrap().is_none());
    assert_eq!(port.completion, PcPortCompletion::Active);
    port.consume_accepted(&child, &child_proof, &child_output, None)
        .unwrap();
    assert_eq!(port.counters.consumed_role_outputs, 2);
    assert!(port.next_role().unwrap().is_none());
    assert_eq!(port.completion, PcPortCompletion::RoleBudget);
}

#[test]
fn store_publication_failure_preserves_actual_head_consumption_once() {
    let mut port = port_with_observation_limit(1, 2, Some(0));
    let root = port.next_role().unwrap().unwrap();
    let (proof, output) = accepted(&mut port, &root, 1);
    assert!(port.consume_accepted(&root, &proof, &output, None).is_err());
    assert_eq!(port.counters.consumed_role_outputs, 1);
    assert_eq!(port.counters.accepted_proposer, 1);
    assert!(
        port.role_consumption_recorded(proof.ticket(), &root)
            .unwrap()
    );
    assert_eq!(port.completion, PcPortCompletion::Failed);
    assert!(!port.owner.semantic.nodes[port.root].edges.is_empty());
    assert!(port.pending[&root.question_digest].failure.is_some());
    let before = port.counters;
    assert!(port.consume_accepted(&root, &proof, &output, None).is_err());
    assert_eq!(port.counters, before);
    assert!(
        port.role_consumption_recorded(proof.ticket(), &root)
            .unwrap()
    );
}

#[test]
fn discarded_live_lane_cannot_be_reported_as_selective_completion() {
    let mut port = port(1, 2);
    let root = port.next_role().unwrap().unwrap();
    port.role_discarded(&root, None).unwrap();
    assert_eq!(
        port.finish_materialized().completion,
        PcPortCompletion::Failed
    );
    assert!(port.is_complete());
}

#[path = "goal_tests.rs"]
mod goals;
