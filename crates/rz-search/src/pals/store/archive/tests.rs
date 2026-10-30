use super::*;
use std::time::Duration;

/// A source fixture, never an observed engine execution. It exercises the
/// production issued-report boundary instead of manufacturing opaque permits.
fn foreign_worker_fixture(raw: &Observation) -> crate::external_cpu::worker::ExternalWorkerReport {
    use crate::external_cpu::worker::{ExternalWorkerReport, ExternalWorkerScope};
    use crate::external_cpu::{ExternalCooperativeReport, ExternalGoIssue};
    use rz_contracts::pals::SearchAuthority;
    use rz_contracts::pals_adaptive::{
        CpuJobId, CpuWorkerId, ExternalCpuJobEnvelope, ExternalSliceGrant,
    };
    use rz_contracts::{
        ClockDomain, Deadline, Digest, GameGeneration, MonotonicTick, OwnerId, ProcessEpoch,
        RootGeneration, StateIdentity, StateRevision,
    };
    let epoch = ProcessEpoch(1);
    let id = CpuJobId {
        authority: SearchAuthority {
            epoch,
            game: GameGeneration(1),
            root: RootGeneration(1),
            implementation: Digest([1; 32]),
        },
        sequence: 1,
    };
    let execution = rz_contracts::ExecutionId::new(epoch, 2);
    let owner = CpuWorkerId(0);
    let job = ExternalCpuJobEnvelope {
        id,
        state_identity: StateIdentity {
            owner: OwnerId(1),
            revision: StateRevision(0),
            semantic: Digest([2; 32]),
        },
        question: Digest([3; 32]),
        root_restriction: Digest([4; 32]),
        evaluation_namespace: Digest([5; 32]),
        search_conditions: Digest([6; 32]),
        requested_reported_node_hint: 128,
        hard_deadline: Deadline {
            clock: ClockDomain(epoch),
            at: MonotonicTick(10_000_000_000),
        },
    };
    ExternalWorkerReport {
        job,
        grant: ExternalSliceGrant {
            job: id,
            execution,
            owner,
            requested_reported_node_hint: 128,
            soft_quantum: Duration::from_millis(100),
            control_sequence: 1,
        },
        scope: ExternalWorkerScope {
            job: id,
            execution,
            owner,
        },
        result: Ok(ExternalCooperativeReport {
            report: (**raw.external_report.as_ref().unwrap()).clone(),
            opinions: Vec::new(),
            stop_control: None,
            strict_native_nodes: None,
            reported_node_request: 128,
            requested_multipv: 1,
            option_application_observed: None,
            reported_node_overshoot: false,
            readiness_barrier_completed: true,
            policy_target_known: false,
            go_issue: ExternalGoIssue::Issued,
        }),
        attempt: None,
        controls: rz_runtime::adaptive::ControlSnapshot {
            terminal: true,
            ..Default::default()
        },
        exact_resume_supported: false,
        go_issue: ExternalGoIssue::Issued,
    }
}

fn permit_fixture(
    store: &mut PalsStores,
    raw: Observation,
    allowance: &mut ArchiveIoBudget,
) -> Result<ForeignAppendPermit, StoreError> {
    let worker = foreign_worker_fixture(&raw);
    let issued = ForeignGoIssuedProof::checked_worker_report(&worker, &raw, allowance.deadline)?;
    store.prepare_foreign_append_permit(raw, issued, allowance)
}

fn live_foreign_fixture(store: &mut PalsStores) -> Observation {
    let position = Position::startpos();
    let root = store.focus_actual_moves(position.snapshot()).unwrap();
    let state = store.situations.get(root).unwrap().state;
    let TaskAdmission::Start(execution) = store
        .request_task(
            foreign_key(state),
            TaskConsumer {
                id: 71,
                situation: root,
                revision: 0,
                generation: store.generation(),
                deadline_tick: 100,
            },
            0,
        )
        .unwrap()
    else {
        panic!("fixture foreign task must start")
    };
    let pv = store.append_cpu_pv(&position, &[mv("e2e4")]).unwrap();
    foreign_observation(state, execution, pv, mv("e2e4"), Color::White, 1)
}

#[test]
fn foreign_worker_permit_scan_does_not_publish_and_hot_commit_does_not_touch_archive() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let raw = live_foreign_fixture(&mut store);
    let before_sequence = store.observations.next_observation_sequence();
    let before_hot = store.hot_stats();
    let mut allowance = budget();
    let permit = permit_fixture(&mut store, raw.clone(), &mut allowance).unwrap();
    assert_eq!(store.hot_stats(), before_hot);
    assert_eq!(
        store.observations.next_observation_sequence(),
        before_sequence
    );
    assert_eq!(permit.request_id(), 1);
    assert_eq!(permit.observation(), &raw);
    // Native worker execution and semantic Store execution remain distinct.
    assert_eq!(permit.native_execution().sequence, 2);
    let archive_before_append = owner_snapshot(&store);
    let remaining = allowance;
    let appended = store.hot_append_foreign_permitted(permit).unwrap();
    assert_eq!(appended.0, before_sequence);
    assert_eq!(store.observations.get(appended).unwrap(), &raw);
    assert_eq!(owner_snapshot(&store), archive_before_append);
    assert_eq!(allowance.deadline, remaining.deadline);
    assert_eq!(allowance.max_bytes, remaining.max_bytes);
    assert_eq!(
        store.tasks.get(raw.execution.unwrap()).unwrap().status,
        TaskStatus::InFlight
    );
}

#[test]
fn foreign_permit_rejects_not_issued_unknown_and_replaced_exact_payload() {
    use crate::external_cpu::ExternalGoIssue;
    let root = TestRoot::new();
    let mut store = setup(&root);
    let raw = live_foreign_fixture(&mut store);
    let mut worker = foreign_worker_fixture(&raw);
    for go in [ExternalGoIssue::NotIssued, ExternalGoIssue::Unknown] {
        worker.go_issue = go;
        assert!(matches!(
            ForeignGoIssuedProof::checked_worker_report(&worker, &raw, budget().deadline),
            Err(StoreError::InvalidEvidence(
                "foreign go was not exactly issued"
            ))
        ));
    }
    worker.go_issue = ExternalGoIssue::Issued;
    let mut allowance = budget();
    let issued =
        ForeignGoIssuedProof::checked_worker_report(&worker, &raw, allowance.deadline).unwrap();
    let mut changed = raw;
    changed.external_report.as_mut().unwrap().request_id += 1;
    assert!(matches!(
        store.prepare_foreign_append_permit(changed, issued, &mut allowance),
        Err(StoreError::InvalidEvidence(
            "foreign append permit payload differs"
        ))
    ));
    assert_eq!(store.observations.len(), 0);
}

#[test]
fn foreign_permit_rejects_new_hot_fact_or_replacement_owner_before_append() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let raw = live_foreign_fixture(&mut store);
    let permit = permit_fixture(&mut store, raw.clone(), &mut budget()).unwrap();
    store.append_observation(observation(raw.state)).unwrap();
    assert_eq!(
        store.hot_append_foreign_permitted(permit),
        Err(StoreError::ForeignPermitStale)
    );
    let permit = permit_fixture(&mut store, raw, &mut budget()).unwrap();
    let replacement_root = TestRoot::new();
    let mut replacement = setup(&replacement_root);
    assert_eq!(
        replacement.hot_append_foreign_permitted(permit),
        Err(StoreError::ForeignPermitStale)
    );
}

#[test]
fn foreign_permit_rejects_new_cold_generation_before_append() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let raw = live_foreign_fixture(&mut store);
    let permit = permit_fixture(&mut store, raw, &mut budget()).unwrap();
    let manager = store.archive.as_mut().unwrap();
    // A committed empty fixture segment still changes the archive namespace.
    manager
        .commit(&Bundle::new(0), &mut budget(), ArchiveStats::default())
        .unwrap();
    assert_eq!(
        store.hot_append_foreign_permitted(permit),
        Err(StoreError::ForeignPermitStale)
    );
    assert_eq!(store.observations.len(), 0);
}

#[test]
fn foreign_permit_hot_capacity_failure_requires_new_bounded_worker_scan() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let raw = live_foreign_fixture(&mut store);
    let mut allowance = budget();
    let permit = permit_fixture(&mut store, raw.clone(), &mut allowance).unwrap();
    let first_nonce = permit.nonce();
    store.observations.observations.fail_next_reservations(1);
    let archive_before = owner_snapshot(&store);
    assert_eq!(
        store.hot_append_foreign_permitted(permit),
        Err(StoreError::Capacity("injected hot allocation"))
    );
    assert_eq!(owner_snapshot(&store), archive_before);
    assert_eq!(store.observations.len(), 0);
    // The consumed permission cannot be reused. A second explicit preparation
    // gives a new nonce and performs its own bounded uniqueness read scope.
    let second = permit_fixture(&mut store, raw, &mut allowance).unwrap();
    assert_ne!(first_nonce, second.nonce());
    store.hot_append_foreign_permitted(second).unwrap();
}

#[test]
fn foreign_permit_cold_scan_preserves_global_request_uniqueness_and_debits_reads() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let raw = live_foreign_fixture(&mut store);
    let old_execution = raw.execution.unwrap();
    let mut allowance = budget();
    let permit = permit_fixture(&mut store, raw, &mut allowance).unwrap();
    let old_observation = store.hot_append_foreign_permitted(permit).unwrap();
    store.complete_task(old_execution, old_observation).unwrap();
    let mut position = Position::startpos();
    position.make_move(mv("e2e4")).unwrap();
    let current = store.focus_actual_moves(position.snapshot()).unwrap();
    store
        .archive_inactive_with_budget(&mut allowance)
        .unwrap()
        .unwrap();
    let state = store.situations.get(current).unwrap().state;
    let TaskAdmission::Start(execution) = store
        .request_task(
            foreign_key(state),
            TaskConsumer {
                id: 72,
                situation: current,
                revision: 0,
                generation: store.generation(),
                deadline_tick: 100,
            },
            0,
        )
        .unwrap()
    else {
        panic!("new-root fixture task must start")
    };
    let pv = store.append_cpu_pv(&position, &[mv("e7e5")]).unwrap();
    let duplicate = foreign_observation(state, execution, pv, mv("e7e5"), Color::Black, 1);
    let before = store.hot_stats();
    let before_sequence = store.observations.next_observation_sequence();
    let remaining = allowance.max_bytes;
    let cold_bytes = fs::metadata(store.archive.as_ref().unwrap().path(0))
        .unwrap()
        .len();
    assert!(matches!(
        permit_fixture(&mut store, duplicate, &mut allowance),
        Err(StoreError::InvalidEvidence(
            "foreign physical request belongs to another archived task"
        ))
    ));
    assert_eq!(remaining - allowance.max_bytes, cold_bytes);
    assert_eq!(store.hot_stats(), before);
    assert_eq!(
        store.observations.next_observation_sequence(),
        before_sequence
    );
}

#[test]
fn foreign_permit_rejects_root_generation_and_same_high_water_fact_owner_replacement() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let raw = live_foreign_fixture(&mut store);
    let permit = permit_fixture(&mut store, raw.clone(), &mut budget()).unwrap();
    store
        .hot_focus_actual_moves(Position::startpos().snapshot())
        .unwrap();
    assert_eq!(
        store.hot_append_foreign_permitted(permit),
        Err(StoreError::ForeignPermitStale)
    );
    let permit = permit_fixture(&mut store, raw, &mut budget()).unwrap();
    let old_high_water = store.observations.next_observation_sequence();
    store.observations = ObservationStore::new(limits().observations);
    assert_eq!(
        store.observations.next_observation_sequence(),
        old_high_water
    );
    assert_eq!(
        store.hot_append_foreign_permitted(permit),
        Err(StoreError::ForeignPermitStale)
    );
}

struct TestRoot(PathBuf);
impl TestRoot {
    fn new() -> Self {
        let n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("rz-pals-store-tests-{}-{n}", std::process::id()));
        fs::create_dir(&root).unwrap();
        Self(root)
    }
    fn config(&self) -> ArchiveConfig {
        ArchiveConfig::new(self.0.join("archive"), std::env::current_dir().unwrap())
    }
}
impl Drop for TestRoot {
    fn drop(&mut self) {
        // This unique test-owned root is outside the checkout. Refuse links or
        // paths outside the exact temporary parent before recursive cleanup.
        if self.0.parent() == Some(std::env::temp_dir().as_path())
            && self
                .0
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("rz-pals-store-tests-")
            && no_link(&self.0).is_ok()
        {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}
fn limits() -> StoreLimits {
    StoreLimits {
        states: 4,
        situations: 4,
        line_chunks: 16,
        observations: 8,
        dependency_edges: 16,
        executions: 4,
        consumers: 8,
        conclusions_per_situation: 8,
        root_moves_per_task: 16,
        retained_history_bytes: 64 * 1024,
        ..StoreLimits::default()
    }
}
fn budget() -> ArchiveIoBudget {
    ArchiveIoBudget {
        deadline: Instant::now() + Duration::from_secs(10),
        max_bytes: 8 * 1024 * 1024,
    }
}
fn setup(root: &TestRoot) -> PalsStores {
    let mut store = PalsStores::new(limits());
    store.enable_archive(root.config()).unwrap();
    store
}
fn runtime_limits(write_bytes: u64) -> ArchiveRuntimeLimits {
    ArchiveRuntimeLimits {
        game_bytes_max: write_bytes,
        global_bytes_max: GLOBAL_ARCHIVE_BYTES,
        index_entries_max: 16,
        index_bytes_max: 1024,
        load_bytes_max: write_bytes.min(ARCHIVE_RECORD_PAYLOAD_BYTES),
        load_deadline_max_ms: 10_000,
        record_payload_bytes_max: write_bytes.min(ARCHIVE_RECORD_PAYLOAD_BYTES),
        max_load_pins: 4,
    }
}
fn managed_setup(root: &TestRoot, selected: ArchiveRuntimeLimits) -> PalsStores {
    let mut store = setup(root);
    store.set_archive_runtime_limits(selected).unwrap();
    store
}
fn owner_snapshot(store: &PalsStores) -> ArchiveOwnerSnapshot {
    store.archive_owner_snapshot().unwrap().unwrap()
}

fn held_setup(root: &TestRoot) -> (PalsStores, ArchiveRootCapability) {
    let config = root.config();
    fs::create_dir(&config.root).unwrap();
    let held = Dir::open_ambient_dir(&config.root, cap_std::ambient_authority()).unwrap();
    let capability = ArchiveRootCapability::from_directory(config.root.clone(), held).unwrap();
    let mut store = PalsStores::new(limits());
    store
        .enable_archive_with_capability(config, capability.clone())
        .unwrap();
    (store, capability)
}

#[test]
fn held_archive_capability_preserves_commit_readback_and_session_domain() {
    let root = TestRoot::new();
    let (mut first, capability) = held_setup(&root);
    let bundle = Bundle::new(0);
    let extent = HEADER + serde_json::to_vec(&bundle).unwrap().len() as u64;
    let selected = runtime_limits(2 * extent);
    first.set_archive_runtime_limits(selected).unwrap();
    let receipt = first
        .archive
        .as_mut()
        .unwrap()
        .commit(&bundle, &mut budget(), ArchiveStats::default())
        .unwrap();
    let manager = first.archive.as_ref().unwrap();
    assert!(manager.directory_capability.is_some());
    assert_eq!(
        manager
            .read(0, Some(receipt.checksum), &mut budget())
            .unwrap()
            .1,
        receipt.checksum
    );
    assert_eq!(manager.usage(budget()).unwrap(), (extent, extent));
    let first_game = manager.directory_identity.unwrap();
    first.close_archive_owner().unwrap();
    let mut next = PalsStores::new(limits());
    next.enable_archive_with_capability(root.config(), capability)
        .unwrap();
    next.set_archive_runtime_limits(selected).unwrap();
    next.inherit_archive_write_budget_from(&first).unwrap();
    let next_manager = next.archive.as_ref().unwrap();
    assert_ne!(next_manager.directory_identity.unwrap(), first_game);
    assert_eq!(next_manager.usage(budget()).unwrap(), (0, extent));
    assert_eq!(
        owner_snapshot(&next).session_write_bytes_consumed,
        Some(extent)
    );
}

#[test]
fn held_archive_capability_rejects_unrelated_directory_and_path_domain() {
    let root = TestRoot::new();
    let config = root.config();
    fs::create_dir(&config.root).unwrap();
    let unrelated = root.0.join("unrelated");
    fs::create_dir(&unrelated).unwrap();
    let wrong = Dir::open_ambient_dir(&unrelated, cap_std::ambient_authority()).unwrap();
    assert!(matches!(
        ArchiveRootCapability::from_directory(config.root, wrong),
        Err(StoreError::ArchiveIntegrity(
            "held archive root was renamed or replaced"
        )),
    ));
}

#[cfg(unix)]
#[test]
fn held_archive_rejects_normal_parent_replacement_before_owner_open() {
    let root = TestRoot::new();
    let runtime = root.0.join("runtime");
    let archive = runtime.join("archive");
    fs::create_dir_all(&archive).unwrap();
    let held = Dir::open_ambient_dir(&archive, cap_std::ambient_authority()).unwrap();
    let capability = ArchiveRootCapability::from_directory(archive.clone(), held).unwrap();
    fs::rename(&runtime, root.0.join("retained-runtime")).unwrap();
    fs::create_dir_all(&archive).unwrap();
    let mut store = PalsStores::new(limits());
    assert!(matches!(
        store.enable_archive_with_capability(
            ArchiveConfig::new(archive.clone(), std::env::current_dir().unwrap()),
            capability,
        ),
        Err(StoreError::ArchiveIntegrity(
            "held archive root was renamed or replaced"
        )),
    ));
    assert!(!store.archive_enabled());
    assert_eq!(fs::read_dir(archive).unwrap().count(), 0);
    assert_eq!(
        fs::read_dir(root.0.join("retained-runtime/archive"))
            .unwrap()
            .count(),
        0
    );
}

#[cfg(unix)]
#[test]
fn held_archive_rejects_normal_root_replacement_after_owner_open_without_redirecting_io() {
    let root = TestRoot::new();
    let (mut store, _) = held_setup(&root);
    let config = root.config();
    let original_game = store
        .archive
        .as_ref()
        .unwrap()
        .directory
        .file_name()
        .unwrap()
        .to_owned();
    fs::rename(&config.root, root.0.join("retained-archive")).unwrap();
    fs::create_dir(&config.root).unwrap();
    let manager = store.archive.as_mut().unwrap();
    assert!(matches!(
        manager.commit(&Bundle::new(0), &mut budget(), ArchiveStats::default()),
        Err(StoreError::ArchiveIntegrity(
            "held archive root was renamed or replaced"
        )),
    ));
    assert_eq!(manager.snapshot().committed_chunks, 0);
    assert_eq!(manager.snapshot().io_failures, 1);
    assert_eq!(fs::read_dir(&config.root).unwrap().count(), 0);
    assert_eq!(
        fs::read_dir(root.0.join("retained-archive").join(original_game))
            .unwrap()
            .count(),
        0
    );
}

#[cfg(unix)]
#[test]
fn held_archive_rejects_game_directory_replacement() {
    let root = TestRoot::new();
    let (store, _) = held_setup(&root);
    let manager = store.archive.as_ref().unwrap();
    fs::rename(
        &manager.directory,
        manager.config.root.join("retained-game"),
    )
    .unwrap();
    fs::create_dir(&manager.directory).unwrap();
    assert!(matches!(
        manager.usage(budget()),
        Err(StoreError::ArchiveIntegrity(
            "held archive game was renamed or replaced"
        )),
    ));
    assert_eq!(fs::read_dir(&manager.directory).unwrap().count(), 0);
}

#[cfg(unix)]
#[test]
fn held_quota_lock_drop_retains_replaced_lock() {
    let root = TestRoot::new();
    let (store, capability) = held_setup(&root);
    let manager = store.archive.as_ref().unwrap();
    let lock = QuotaLock::take(&manager.config.root, Some(&capability)).unwrap();
    let named = manager.config.root.join(".quota.lock");
    fs::rename(&named, manager.config.root.join("retained-lock")).unwrap();
    fs::write(&named, b"replacement").unwrap();
    drop(lock);
    assert_eq!(fs::read(named).unwrap(), b"replacement");
}

#[cfg(target_os = "linux")]
#[test]
fn held_archive_fifo_replacement_is_rejected_without_blocking_open_or_lock_drop() {
    use nix::sys::stat::Mode;
    use nix::unistd::mkfifo;
    let root = TestRoot::new();
    let (store, capability) = held_setup(&root);
    let manager = store.archive.as_ref().unwrap();
    mkfifo(&manager.path(0), Mode::S_IRUSR | Mode::S_IWUSR).unwrap();
    assert!(matches!(
        manager.read(0, None, &mut budget()),
        Err(StoreError::ArchiveIntegrity("archive record type")),
    ));
    let lock = QuotaLock::take(&manager.config.root, Some(&capability)).unwrap();
    let named = manager.config.root.join(".quota.lock");
    fs::rename(&named, manager.config.root.join("retained-lock")).unwrap();
    mkfifo(&named, Mode::S_IRUSR | Mode::S_IWUSR).unwrap();
    drop(lock);
    assert!(!fs::metadata(named).unwrap().is_file());
}

#[test]
fn session_write_cap_is_exact_across_live_and_closed_game_owners() {
    for close_first in [false, true] {
        let root = TestRoot::new();
        let bundle = Bundle::new(0);
        let extent = HEADER + serde_json::to_vec(&bundle).unwrap().len() as u64;
        let selected = runtime_limits(2 * extent);
        let mut first = managed_setup(&root, selected);
        let receipt = first
            .archive
            .as_mut()
            .unwrap()
            .commit(&bundle, &mut budget(), ArchiveStats::default())
            .unwrap();
        let handle = receipt.state_handle(StateId(0));
        let first_snapshot = owner_snapshot(&first);
        assert_eq!(first_snapshot.archive_write_bytes_total, Some(extent));
        assert_eq!(first_snapshot.session_write_bytes_consumed, Some(extent));
        assert_eq!(first_snapshot.write_bytes_remaining, Some(extent));
        if close_first {
            let closed = first.close_archive_owner().unwrap().unwrap();
            assert_eq!(closed.lifecycle, ArchiveOwnerLifecycle::Closed);
            assert!(closed.cleanup_complete && closed.admission_closed);
            assert!(!first.archive_enabled());
            assert!(handle.owner.upgrade().is_none());
            assert_eq!(
                first.archive_inactive(budget()).unwrap_err(),
                StoreError::ArchiveDisabled
            );
        }

        // The domain is canonical, even when the requested path is spelled
        // differently. Handle identity is still the new manager's own Arc.
        let mut next = PalsStores::new(limits());
        let mut config = root.config();
        config.root = config.root.join(".");
        next.enable_archive(config).unwrap();
        next.set_archive_runtime_limits(selected).unwrap();
        next.inherit_archive_write_budget_from(&first).unwrap();
        let inherited = owner_snapshot(&next);
        assert_ne!(inherited.owner_id, first_snapshot.owner_id);
        assert_eq!(inherited.generation, 0);
        assert_eq!(inherited.archive_write_bytes_total, Some(0));
        assert_eq!(
            inherited.session_write_ledger_id,
            first_snapshot.session_write_ledger_id
        );
        assert_eq!(inherited.session_write_bytes_max, Some(2 * extent));
        assert_eq!(inherited.session_write_bytes_consumed, Some(extent));
        assert_eq!(inherited.write_bytes_remaining, Some(extent));
        assert_eq!(
            next.archive.as_ref().unwrap().verify_owner(&handle),
            Err(StoreError::InvalidHandle("cold owner or generation"))
        );
        next.archive
            .as_mut()
            .unwrap()
            .commit(&bundle, &mut budget(), ArchiveStats::default())
            .unwrap();
        let exhausted = owner_snapshot(&next);
        assert_eq!(exhausted.session_write_bytes_consumed, Some(2 * extent));
        assert_eq!(exhausted.archive_write_bytes_total, Some(extent));
        assert_eq!(exhausted.write_bytes_reserved, 0);
        assert_eq!(exhausted.write_bytes_remaining, Some(0));

        let mut third = managed_setup(&root, selected);
        third.inherit_archive_write_budget_from(&next).unwrap();
        assert_eq!(
            third
                .archive
                .as_mut()
                .unwrap()
                .commit(&bundle, &mut budget(), ArchiveStats::default())
                .unwrap_err(),
            StoreError::ArchiveQuota("archive lifetime write bytes")
        );
        let rejected = owner_snapshot(&third);
        assert_eq!(rejected.generation, 0);
        assert_eq!(rejected.archive_write_bytes_total, Some(0));
        assert_eq!(rejected.session_write_bytes_consumed, Some(2 * extent));
        assert_eq!(rejected.write_bytes_remaining, Some(0));
        assert!(
            fs::read_dir(&third.archive.as_ref().unwrap().directory)
                .unwrap()
                .next()
                .is_none()
        );
    }
}

#[test]
fn write_budget_inheritance_rejects_domain_use_and_duplicate_without_mutation() {
    let root = TestRoot::new();
    let other_root = TestRoot::new();
    let selected = runtime_limits(1024 * 1024);
    let source = managed_setup(&root, selected);
    let source_before = owner_snapshot(&source);
    for case in 0..12 {
        let mut changed = selected;
        match case {
            1 => changed.game_bytes_max += 1,
            2 => changed.global_bytes_max -= 1,
            3 => changed.index_entries_max += 1,
            4 => changed.index_bytes_max += 1,
            5 => changed.load_bytes_max -= 1,
            6 => changed.load_deadline_max_ms += 1,
            7 => changed.record_payload_bytes_max -= 1,
            8 => changed.max_load_pins += 1,
            _ => {}
        }
        let mut next = managed_setup(if case == 0 { &other_root } else { &root }, changed);
        match case {
            9 => {
                next.focus_actual_moves(Position::startpos().snapshot())
                    .unwrap();
            }
            10 => {
                next.measure_archive_usage_with_budget(&mut budget())
                    .unwrap();
            }
            11 => {
                next.inherit_archive_write_budget_from(&source).unwrap();
            }
            _ => {}
        }
        let before = owner_snapshot(&next);
        let hot_before = next.hot_stats();
        assert!(
            matches!(
                next.inherit_archive_write_budget_from(&source),
                Err(StoreError::InvalidConditions(_))
            ),
            "case {case}"
        );
        assert_eq!(owner_snapshot(&next), before, "case {case}");
        assert_eq!(next.hot_stats(), hot_before, "case {case}");
        assert_eq!(owner_snapshot(&source), source_before, "case {case}");
    }

    let mut legacy = setup(&root);
    let legacy_before = owner_snapshot(&legacy);
    assert!(legacy.inherit_archive_write_budget_from(&source).is_err());
    assert_eq!(owner_snapshot(&legacy), legacy_before);
    let mut selected_next = managed_setup(&root, selected);
    let before = owner_snapshot(&selected_next);
    assert!(
        selected_next
            .inherit_archive_write_budget_from(&legacy)
            .is_err()
    );
    assert_eq!(owner_snapshot(&selected_next), before);
    let legacy_source = setup(&root);
    legacy
        .inherit_archive_write_budget_from(&legacy_source)
        .unwrap();
    legacy
        .inherit_archive_write_budget_from(&legacy_source)
        .unwrap();
    assert_eq!(owner_snapshot(&legacy), legacy_before);
}

#[test]
fn partial_write_and_failed_publication_debit_actual_bytes_and_retain_known_files() {
    for fault in [
        TestCommitFault::PartialPayload(17),
        TestCommitFault::ExpirePostCommitScan,
    ] {
        let root = TestRoot::new();
        let selected = runtime_limits(1024 * 1024);
        let mut store = managed_setup(&root, selected);
        let mut position = Position::startpos();
        store.focus_actual_moves(position.snapshot()).unwrap();
        position.make_move(mv("e2e4")).unwrap();
        store.focus_actual_moves(position.snapshot()).unwrap();
        let hot_before = store.hot_stats();
        let directory = store.archive.as_ref().unwrap().directory.clone();
        store.archive.as_mut().unwrap().test_commit_fault = Some(fault);
        let error = store.archive_inactive(budget()).unwrap_err();
        let partial = matches!(fault, TestCommitFault::PartialPayload(_));
        if partial {
            assert!(matches!(
                error,
                StoreError::ArchiveIo {
                    stage: "injected archive partial write",
                    ..
                }
            ));
        } else {
            assert_eq!(error, StoreError::ArchiveDeadline);
        }
        let path = directory.join(if partial {
            "seg-00000000000000000000.pending"
        } else {
            "seg-00000000000000000000.arc"
        });
        let actual = fs::metadata(&path).unwrap().len();
        let snapshot = owner_snapshot(&store);
        assert_eq!(snapshot.pending_commit_bytes, Some(actual));
        assert!(snapshot.pending_files_retained);
        assert!(!snapshot.pending_buffers_retained);
        assert!(snapshot.admission_closed);
        assert_eq!(snapshot.lifecycle, ArchiveOwnerLifecycle::Failed);
        assert!(!snapshot.complete);
        assert_eq!(snapshot.archive_write_bytes_total, Some(actual));
        assert_eq!(snapshot.session_write_bytes_consumed, Some(actual));
        assert_eq!(
            snapshot.write_bytes_remaining,
            Some(selected.game_bytes_max - actual)
        );
        assert_eq!(snapshot.write_bytes_reserved, 0);
        assert_eq!(
            store.archive_write_ledger.as_ref().unwrap().facts(),
            Some((actual, 0))
        );
        assert_eq!(store.hot_stats(), hot_before);
        assert_eq!(snapshot.ram_reclaim_events, 0);
        if partial {
            assert_eq!(actual, HEADER + 17);
            assert_eq!(snapshot.committed_chunks, 0);
            assert_eq!(snapshot.integrity_verified_chunks, 0);
        } else {
            assert_eq!(snapshot.committed_chunks, 1);
            assert_eq!(snapshot.integrity_verified_chunks, 1);
        }
        assert_eq!(
            store.close_archive_owner().unwrap_err(),
            StoreError::InvalidConditions("archive close retains pending ownership")
        );
        let rejected_close = owner_snapshot(&store);
        assert!(!rejected_close.cleanup_complete);
        assert!(rejected_close.pending_files_retained);
        assert_eq!(rejected_close.pending_commit_bytes, Some(actual));
        assert_eq!(fs::metadata(&path).unwrap().len(), actual);

        let mut next = managed_setup(&root, selected);
        next.inherit_archive_write_budget_from(&store).unwrap();
        let inherited = owner_snapshot(&next);
        assert_eq!(
            inherited.session_write_ledger_id,
            snapshot.session_write_ledger_id
        );
        assert_eq!(inherited.session_write_bytes_consumed, Some(actual));
        assert_eq!(inherited.archive_write_bytes_total, Some(0));
    }
}

#[test]
fn prewrite_create_failure_releases_unwritten_reservation_and_keeps_zero_actual_debit() {
    let root = TestRoot::new();
    let selected = runtime_limits(1024 * 1024);
    let mut store = managed_setup(&root, selected);
    let directory = store.archive.as_ref().unwrap().directory.clone();
    fs::remove_dir(&directory).unwrap();
    assert!(matches!(
        store.archive.as_mut().unwrap().commit(
            &Bundle::new(0),
            &mut budget(),
            ArchiveStats::default()
        ),
        Err(StoreError::ArchiveIo {
            stage: "archive temporary create",
            ..
        })
    ));
    let snapshot = owner_snapshot(&store);
    assert_eq!(snapshot.session_write_bytes_consumed, Some(0));
    assert_eq!(snapshot.archive_write_bytes_total, Some(0));
    assert_eq!(
        snapshot.write_bytes_remaining,
        Some(selected.game_bytes_max)
    );
    assert_eq!(snapshot.write_bytes_reserved, 0);
    assert_eq!(snapshot.pending_commit_bytes, Some(0));
    assert!(!snapshot.pending_files_retained);
    assert_eq!(
        store.archive_write_ledger.as_ref().unwrap().facts(),
        Some((0, 0))
    );
}

#[test]
fn closed_owner_retains_only_ledger_through_deadline_cancel_and_repeated_cleanup() {
    let root = TestRoot::new();
    let selected = runtime_limits(1024 * 1024);
    let mut previous = managed_setup(&root, selected);
    let receipt = previous
        .archive
        .as_mut()
        .unwrap()
        .commit(&Bundle::new(0), &mut budget(), ArchiveStats::default())
        .unwrap();
    let handle = receipt.state_handle(StateId(0));
    let path = previous.archive.as_ref().unwrap().path(receipt.generation);
    let closed = previous.close_archive_owner().unwrap().unwrap();
    for failure in [StoreError::ArchiveDeadline, StoreError::ArchiveCanceled] {
        let mut staged = managed_setup(&root, selected);
        staged.inherit_archive_write_budget_from(&previous).unwrap();
        let mut allowance = budget();
        if failure == StoreError::ArchiveDeadline {
            allowance.deadline = Instant::now();
        }
        let result: Result<(), StoreError> =
            staged.with_archive_allocation(StorePins::default(), &mut allowance, |_| {
                Err(failure.clone())
            });
        assert_eq!(result.unwrap_err(), failure);
        let snapshot = owner_snapshot(&staged);
        assert_eq!(
            snapshot.session_write_ledger_id,
            closed.session_write_ledger_id
        );
        assert_eq!(
            snapshot.session_write_bytes_consumed,
            closed.session_write_bytes_consumed
        );
        assert_eq!(snapshot.write_bytes_remaining, closed.write_bytes_remaining);
        assert_eq!(snapshot.archive_write_bytes_total, Some(0));
        assert_eq!(snapshot.generation, 0);
        drop(staged);
        assert!(previous.close_archive_owner().unwrap().is_none());
        assert!(!previous.archive_enabled());
        assert!(previous.archive_owner_snapshot().unwrap().is_none());
        assert!(handle.owner.upgrade().is_none());
        assert_eq!(
            previous.enable_archive(root.config()),
            Err(StoreError::InvalidConditions(
                "archive session write authority requires a replacement store"
            ))
        );
        assert_eq!(
            previous.archive_inactive(budget()).unwrap_err(),
            StoreError::ArchiveDisabled
        );
        assert_eq!(fs::metadata(&path).unwrap().len(), receipt.committed_bytes);
    }
    let mut retry = managed_setup(&root, selected);
    retry.inherit_archive_write_budget_from(&previous).unwrap();
    let snapshot = owner_snapshot(&retry);
    assert_eq!(
        snapshot.session_write_ledger_id,
        closed.session_write_ledger_id
    );
    assert_eq!(
        snapshot.session_write_bytes_consumed,
        closed.session_write_bytes_consumed
    );
    assert_eq!(snapshot.write_bytes_remaining, closed.write_bytes_remaining);
}
fn mv(text: &str) -> BoardMove {
    text.parse().unwrap()
}
fn identity() -> CpuValueIdentity {
    CpuValueIdentity {
        semantics: crate::cpu::BOOTSTRAP_SCORE_VERSION.into(),
        weights_sha256: None,
        training: crate::cpu_value::CpuTrainingState::Bootstrap,
    }
}
fn observation(state: StateId) -> Observation {
    Observation {
        state,
        line: None,
        source: 1,
        epoch: 1,
        value_identity: Some(identity()),
        checker_identity: None,
        checker_work: None,
        external_report: None,
        model_value_identity: None,
        model_value_input: None,
        cpu_condition: Some("archive-fixture-cpu-conditions-v1".into()),
        cpu_pv: None,
        scope: EvidenceScope::DepthLimited {
            depth: 4,
            profile: 1,
            condition: 1,
        },
        score: RawScore::Cpu {
            value: 123,
            perspective: Color::White,
            bound: BoundKind::ExactWithinSearch,
        },
        budget: 128,
        kind: ObservationKind::CpuAnalysis,
        supersedes: None,
        execution: None,
    }
}
fn key(state: StateId) -> TaskKey {
    TaskKey {
        state,
        line: None,
        question: TaskQuestion::AnalyzePosition,
        root_moves: vec![],
        model: 1,
        epoch: 1,
        value_identity: Some(identity()),
        checker_identity: None,
        cpu_condition: Some("archive-fixture-cpu-conditions-v1".into()),
        profile: 1,
        condition: 1,
        input_revision: 0,
        requested_depth: 4,
        node_budget: 128,
    }
}

#[test]
fn multi_move_small_store_preserves_exact_history_lines_and_evidence() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    let original = position.snapshot();
    let old_situation = store.focus_actual_moves(original.clone()).unwrap();
    let old_state = store.situations.get(old_situation).unwrap().state;
    let old_line = store
        .append_cpu_pv(&position, &[mv("e2e4"), mv("e7e5"), mv("g1f3")])
        .unwrap();
    let mut raw = observation(old_state);
    raw.cpu_pv = Some(old_line);
    let old_observation = store.append_observation(raw.clone()).unwrap();
    store
        .dependencies
        .add(old_observation, old_situation)
        .unwrap();
    position.make_move(mv("g1f3")).unwrap();
    store.focus_actual_moves(position.snapshot()).unwrap();
    let receipt = store.archive_inactive(budget()).unwrap().unwrap();
    assert_eq!(
        store.states.get(old_state).unwrap_err(),
        StoreError::ColdRecord("state")
    );
    let replacement = store.root().unwrap();
    assert!(store.situations.get(old_situation).is_err());
    let old_revision = store.situations.get(replacement).unwrap().revision;
    for movement in [
        "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8", "e2e4", "a7a6", "e4e5", "d7d5",
        "e5d6",
    ] {
        position.make_move(mv(movement)).unwrap();
        store.focus_actual_moves(position.snapshot()).unwrap();
        store.archive_inactive(budget()).unwrap();
        assert!(store.states.len() <= limits().states);
        assert!(store.lines.len() <= limits().line_chunks);
        assert!(store.situations.len() <= limits().situations);
        assert_eq!(store.states.snapshots.index.len(), store.states.len());
    }
    let handle = store
        .lookup_cold_state(&original, budget())
        .unwrap()
        .unwrap();
    assert_eq!(handle.original_id(), old_state);
    let loaded = store.pin_load(&handle, budget()).unwrap();
    assert!(store.states.get(old_state).unwrap().same_state(&original));
    assert_eq!(
        store.states.get(old_state).unwrap().known_history_fens(),
        original.known_history_fens()
    );
    let evidence = store
        .pin_load(&receipt.observation_handle(old_observation), budget())
        .unwrap();
    assert_eq!(*store.observations.get(old_observation).unwrap(), raw);
    assert_eq!(
        store.lines.moves(old_line).unwrap(),
        vec![mv("e2e4"), mv("e7e5"), mv("g1f3")]
    );
    assert_eq!(
        store
            .dependencies
            .invalidate(old_observation, &mut store.situations)
            .unwrap(),
        0
    );
    if let Ok(replacement) = store.situations.get(replacement) {
        assert_eq!(replacement.revision, old_revision);
    }
    store.release_loaded(evidence.pin).unwrap();
    store.release_loaded(loaded.pin).unwrap();
}

#[test]
fn roots_frontiers_paused_tasks_and_dependency_closure_are_pinned() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    let root_id = store.focus_actual_moves(position.snapshot()).unwrap();
    position.make_move(mv("e2e4")).unwrap();
    let paused = store.insert_situation(position.snapshot()).unwrap();
    let paused_state = store.situations.get(paused).unwrap().state;
    let TaskAdmission::Start(execution) = store
        .request_task(
            key(paused_state),
            TaskConsumer {
                id: 7,
                situation: paused,
                revision: 0,
                generation: store.generation(),
                deadline_tick: 100,
            },
            0,
        )
        .unwrap()
    else {
        panic!("new task")
    };
    store.pause_task(execution, 19, None).unwrap();
    position.make_move(mv("e7e5")).unwrap();
    let frontier = store.insert_situation(position.snapshot()).unwrap();
    let frontier_state = store.situations.get(frontier).unwrap().state;
    let evidence = store
        .append_observation(observation(frontier_state))
        .unwrap();
    store.dependencies.add(evidence, frontier).unwrap();
    position.make_move(mv("g1f3")).unwrap();
    let inactive = store.insert_situation(position.snapshot()).unwrap();
    let inactive_state = store.situations.get(inactive).unwrap().state;
    let mut pins = StorePins::default();
    pins.situations.insert(frontier);
    store.set_archive_pins(pins).unwrap();
    store.archive_inactive(budget()).unwrap().unwrap();
    assert!(store.situations.get(root_id).is_ok());
    assert!(store.situations.get(paused).is_ok());
    assert!(store.situations.get(frontier).is_ok());
    assert!(store.observations.get(evidence).is_ok());
    assert!(store.states.get(inactive_state).is_err());
    assert!(matches!(
        store.tasks.get(execution).unwrap().status,
        TaskStatus::Paused { checkpoint: 19, .. }
    ));
    assert_eq!(
        store.dependencies.affected(evidence).collect::<Vec<_>>(),
        vec![frontier]
    );
    assert!(matches!(
        store.archive_inactive(budget()),
        Err(StoreError::PinSaturated(_))
    ));
}

#[test]
fn byte_deadline_checksum_and_owner_failures_do_not_publish_hot_records() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    let old = store.focus_actual_moves(position.snapshot()).unwrap();
    let state = store.situations.get(old).unwrap().state;
    position.make_move(mv("e2e4")).unwrap();
    store.focus_actual_moves(position.snapshot()).unwrap();
    let receipt = store.archive_inactive(budget()).unwrap().unwrap();
    let handle = receipt.state_handle(state);
    let before = store.hot_stats();
    assert_eq!(
        store
            .pin_load(
                &handle,
                ArchiveIoBudget {
                    deadline: Instant::now(),
                    max_bytes: u64::MAX
                }
            )
            .unwrap_err(),
        StoreError::ArchiveDeadline
    );
    assert_eq!(
        store
            .pin_load(
                &handle,
                ArchiveIoBudget {
                    max_bytes: 4,
                    ..budget()
                }
            )
            .unwrap_err(),
        StoreError::ArchiveByteBudget
    );
    let other_root = TestRoot::new();
    let mut other = setup(&other_root);
    assert!(matches!(
        other.pin_load(&handle, budget()),
        Err(StoreError::InvalidHandle(_))
    ));
    assert_eq!(store.hot_stats(), before);
    let file = store.archive.as_ref().unwrap().path(receipt.generation);
    use std::io::{Seek, SeekFrom};
    let mut output = OpenOptions::new().write(true).open(file).unwrap();
    output.seek(SeekFrom::End(-1)).unwrap();
    output.write_all(b"!").unwrap();
    output.sync_all().unwrap();
    assert_eq!(
        store.pin_load(&handle, budget()).unwrap_err(),
        StoreError::ArchiveIntegrity("archive checksum")
    );
    assert_eq!(store.hot_stats(), before);
}

#[test]
fn quota_and_io_failure_keep_the_original_ram_and_allow_explicit_recovery() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    store.focus_actual_moves(position.snapshot()).unwrap();
    position.make_move(mv("e2e4")).unwrap();
    store.focus_actual_moves(position.snapshot()).unwrap();
    let before = store.hot_stats();
    store.archive.as_mut().unwrap().config.game_bytes = 80;
    assert_eq!(
        store.archive_inactive(budget()).unwrap_err(),
        StoreError::ArchiveQuota("game archive")
    );
    assert_eq!(store.hot_stats(), before);
    store.archive.as_mut().unwrap().config.game_bytes = GAME_ARCHIVE_BYTES;
    let directory = store.archive.as_ref().unwrap().directory.clone();
    fs::remove_dir(&directory).unwrap();
    assert!(matches!(
        store.archive_inactive(budget()),
        Err(StoreError::ArchiveIo { .. })
    ));
    assert_eq!(store.hot_stats(), before);
    fs::create_dir(&directory).unwrap();
    let receipt = store.archive_inactive(budget()).unwrap().unwrap();
    assert!(receipt.committed_bytes > 0);
    assert!(receipt.after.states < receipt.before.states);
    position.make_move(mv("e7e5")).unwrap();
    store.focus_actual_moves(position.snapshot()).unwrap();
    let before = store.hot_stats();
    store.archive.as_mut().unwrap().config.global_bytes = receipt.committed_bytes + 80;
    assert_eq!(
        store.archive_inactive(budget()).unwrap_err(),
        StoreError::ArchiveQuota("global archive")
    );
    assert_eq!(store.hot_stats(), before);
}

#[test]
fn automatic_pressure_and_one_allocation_retry_preserve_serial_identity() {
    for failures in [1, 2] {
        let root = TestRoot::new();
        let mut store = setup(&root);
        let mut position = Position::startpos();
        let old = store.focus_actual_moves(position.snapshot()).unwrap();
        let old_state = store.situations.get(old).unwrap().state;
        position.make_move(mv("e2e4")).unwrap();
        store.focus_actual_moves(position.snapshot()).unwrap();
        store.set_archive_io_budget(budget()).unwrap();
        store.states.snapshots.fail_next_reservations(failures);
        position.make_move(mv("e7e5")).unwrap();
        let result = store.insert_situation(position.snapshot());
        assert_eq!(store.archive.as_ref().unwrap().next_generation, 1);
        assert!(store.states.get(old_state).is_err());
        if failures == 1 {
            let new = result.unwrap();
            let state = store.situations.get(new).unwrap().state;
            assert!(state.0 > old_state.0);
        } else {
            assert_eq!(
                result.unwrap_err(),
                StoreError::Capacity("injected hot allocation")
            );
        }
    }
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    for text in ["e2e4", "e7e5", "g1f3"] {
        store.focus_actual_moves(position.snapshot()).unwrap();
        position.make_move(mv(text)).unwrap();
    }
    store.focus_actual_moves(position.snapshot()).unwrap();
    assert_eq!(store.states.len(), 4);
    let receipt = store.reclaim_if_needed(budget()).unwrap().unwrap();
    assert_eq!(receipt.after.states, 1);
}

#[test]
fn stale_hot_pool_owner_and_situation_slot_cannot_alias_cold_evidence() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    let old = store.focus_actual_moves(position.snapshot()).unwrap();
    let old_state = store.situations.get(old).unwrap().state;
    position.make_move(mv("e2e4")).unwrap();
    store.focus_actual_moves(position.snapshot()).unwrap();
    let receipt = store.archive_inactive(budget()).unwrap().unwrap();
    let loaded = store
        .pin_load(&receipt.situation_handle(old), budget())
        .unwrap();
    let new = loaded.situations[0].1;
    assert_ne!(old, new);
    assert!(store.situations.get(old).is_err());
    assert_eq!(store.situations.get(new).unwrap().state, old_state);
    store.release_loaded(loaded.pin).unwrap();
    store.states = StateStore::new(limits().states, limits().retained_history_bytes);
    store
        .states
        .insert(Position::startpos().snapshot())
        .unwrap();
    let before = store.hot_stats();
    assert_eq!(
        store
            .pin_load(&receipt.state_handle(old_state), budget())
            .unwrap_err(),
        StoreError::InvalidHandle("archive hot-store owner changed")
    );
    assert_eq!(store.hot_stats(), before);
}

#[test]
fn engine_archive_keeps_roles_relationships_and_ordered_node_payloads() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    let situation = store.focus_actual_moves(position.snapshot()).unwrap();
    let state = store.situations.get(situation).unwrap().state;
    let opaque_cpu = store.append_observation(observation(state)).unwrap();
    let record = crate::pals::engine::RoleRecord {
        revision: 7,
        parent_revision: Some(3),
        supersedes_revision: Some(2),
        origin_state: state,
        kind: crate::pals::engine::RecordKind::Repair,
        line: vec![mv("e2e4"), mv("e7e5")],
        value: Some(19),
        completed_depth: 4,
        score_scope: Some(crate::cpu::CpuScoreScope::CompletedIteration),
        cpu_observation: None,
        perspective: Color::White,
        critical: true,
    };
    let node = EngineArchiveNode {
        state,
        situation,
        edges: vec![(Move16::pack(mv("e2e4")).unwrap(), None)],
        metadata: b"node-v1:visits=9".to_vec(),
    };
    let receipt = store
        .archive_engine_records_with_pins(
            &[record],
            &[node],
            StorePins {
                observations: BTreeSet::from([opaque_cpu]),
                ..StorePins::default()
            },
            budget(),
        )
        .unwrap();
    assert_eq!(receipt.records, 1);
    position.make_move(mv("e2e4")).unwrap();
    store.focus_actual_moves(position.snapshot()).unwrap();
    store.archive_inactive(budget()).unwrap();
    let loaded = store.load_engine_archive(&receipt, budget()).unwrap();
    assert_eq!(
        *store.observations.get(opaque_cpu).unwrap(),
        observation(state)
    );
    let record = &loaded.engine_records[0];
    assert_eq!(record.revision, 7);
    assert_eq!(record.parent_revision, Some(3));
    assert_eq!(record.supersedes_revision, Some(2));
    assert_eq!(record.value, Some(19));
    assert_eq!(record.line, vec![mv("e2e4"), mv("e7e5")]);
    assert!(record.critical);
    assert_eq!(loaded.engine_nodes[0].metadata, b"node-v1:visits=9");
    assert_eq!(
        loaded.engine_nodes[0].edges[0].0.unpack().unwrap(),
        mv("e2e4")
    );
    assert!(store.situations.get(situation).is_err());
    store.release_loaded(loaded.pin).unwrap();
}

#[test]
fn retired_pauses_preserve_partial_evidence_and_never_resume_old_checkpoints() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    let old_snapshot = position.snapshot();
    let old = store.focus_actual_moves(old_snapshot.clone()).unwrap();
    let state = store.situations.get(old).unwrap().state;
    let TaskAdmission::Start(execution) = store
        .request_task(
            key(state),
            TaskConsumer {
                id: 21,
                situation: old,
                revision: 0,
                generation: store.generation(),
                deadline_tick: 100,
            },
            0,
        )
        .unwrap()
    else {
        panic!("new task")
    };
    let mut partial = observation(state);
    partial.execution = Some(execution);
    partial.scope = EvidenceScope::DepthLimited {
        depth: 2,
        profile: 1,
        condition: 1,
    };
    let observation = store.append_observation(partial.clone()).unwrap();
    store.pause_task(execution, 11, Some(observation)).unwrap();
    assert_eq!(
        store.tasks.paused_executions_for_state(state).unwrap(),
        vec![execution]
    );
    assert_eq!(
        store
            .tasks
            .remap_paused_checkpoints(&BTreeMap::from([(state, 7)]))
            .unwrap(),
        1
    );
    assert!(matches!(
        store.tasks.get(execution).unwrap().status,
        TaskStatus::Paused { checkpoint: 7, .. }
    ));
    assert_eq!(
        store.tasks.retire_paused_except(&BTreeSet::new()).unwrap(),
        1
    );
    assert_eq!(store.tasks.active_states().count(), 0);
    assert_eq!(
        store
            .tasks
            .remap_paused_checkpoints(&BTreeMap::from([(state, 0)]))
            .unwrap(),
        0
    );
    position.make_move(mv("e2e4")).unwrap();
    store.focus_actual_moves(position.snapshot()).unwrap();
    let receipt = store.archive_inactive(budget()).unwrap().unwrap();
    assert!(store.tasks.get(execution).is_err());
    let loaded = store
        .pin_load(&receipt.execution_handle(execution), budget())
        .unwrap();
    assert_eq!(
        store.tasks.get(execution).unwrap().status,
        TaskStatus::RetiredPaused {
            checkpoint: 7,
            evidence: Some(observation)
        }
    );
    assert_eq!(*store.observations.get(observation).unwrap(), partial);
    let new = store.focus_actual_moves(old_snapshot).unwrap();
    let admission = store
        .request_task(
            key(state),
            TaskConsumer {
                id: 22,
                situation: new,
                revision: 0,
                generation: store.generation(),
                deadline_tick: 100,
            },
            0,
        )
        .unwrap();
    assert!(matches!(admission,TaskAdmission::Start(new_execution) if new_execution.0>execution.0));
    assert!(store.complete_task(execution, observation).is_err());
    store.release_loaded(loaded.pin).unwrap();
}

fn model_observation(state: StateId, perspective: Color, wdl: [f32; 3]) -> Observation {
    let mut o = observation(state);
    o.value_identity = None;
    o.cpu_condition = None;
    o.model_value_identity = Some(crate::pals::value::ModelValueIdentity {
        semantics: crate::pals::value::MODEL_WDL_VALUE_SEMANTICS.into(),
        model: "actual-fixture-model".into(),
        encoding: "actual-fixture-encoding".into(),
        precision: "fp32".into(),
        model_epoch: [7; 32],
    });
    o.model_value_input = Some([state.0 as u8; 32]);
    o.scope = EvidenceScope::Model {
        model: 7,
        encoding: 8,
        input: 9,
    };
    o.kind = ObservationKind::Proposal;
    o.score = RawScore::Wdl {
        win: wdl[0],
        draw: wdl[1],
        loss: wdl[2],
        perspective,
    };
    o
}

fn context_model_observation(
    state: StateId,
    perspective: Color,
    wdl: [f32; 3],
    context_revision: u64,
) -> Observation {
    let mut o = model_observation(state, perspective, wdl);
    o.score = RawScore::ContextWdl {
        win: wdl[0],
        draw: wdl[1],
        loss: wdl[2],
        perspective,
        context_revision,
    };
    o
}

#[test]
fn conditional_model_evidence_stays_distinct_and_archives_both_actual_inputs() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    let old = store.focus_actual_moves(position.snapshot()).unwrap();
    let state = store.situations.get(old).unwrap().state;
    let line = store
        .append_line(
            store.situations.get(old).unwrap().focus,
            &[mv("e2e4"), mv("e7e5")],
        )
        .unwrap();
    position.make_move(mv("e2e4")).unwrap();
    let repaired_state = store.states.insert(position.snapshot()).unwrap();
    let repaired_raw = context_model_observation(repaired_state, Color::Black, [0.1, 0.2, 0.7], 43);
    let repaired = store.append_observation(repaired_raw.clone()).unwrap();
    position.make_move(mv("e7e5")).unwrap();
    let counter_state = store.states.insert(position.snapshot()).unwrap();
    let counter_raw = context_model_observation(counter_state, Color::White, [0.2, 0.3, 0.5], 43);
    let counter = store.append_observation(counter_raw.clone()).unwrap();
    let mut conditional = model_observation(state, Color::White, [0.2, 0.3, 0.5]);
    conditional.line = Some(line);
    conditional.kind = ObservationKind::Refutation;
    conditional.model_value_identity = None;
    conditional.model_value_input = None;
    conditional.score = RawScore::ConditionalWdl {
        expectation: 0.2 - 0.5,
        perspective: Color::White,
        repaired,
        counter,
        context_revision: 43,
    };
    let before = store.hot_stats();
    let mut invalid = conditional.clone();
    invalid.model_value_identity = counter_raw.model_value_identity.clone();
    invalid.model_value_input = counter_raw.model_value_input;
    assert!(store.append_observation(invalid).is_err());
    let mut invalid = conditional.clone();
    invalid.line = None;
    assert!(store.append_observation(invalid).is_err());
    let mut invalid = conditional.clone();
    invalid.score = RawScore::ConditionalWdl {
        expectation: 0.3,
        perspective: Color::White,
        repaired,
        counter,
        context_revision: 43,
    };
    assert!(store.append_observation(invalid).is_err());
    let mut invalid = conditional.clone();
    invalid.score = RawScore::ConditionalWdl {
        expectation: 0.5 - 0.2,
        perspective: Color::Black,
        repaired,
        counter,
        context_revision: 43,
    };
    assert!(store.append_observation(invalid).is_err());
    assert_eq!(store.hot_stats(), before);
    let mut malformed_raw = counter_raw.clone();
    malformed_raw.model_value_input = None;
    assert!(validate_conditional_wdl(&conditional, &repaired_raw, &malformed_raw).is_err());
    let another_context =
        context_model_observation(counter_state, Color::White, [0.2, 0.3, 0.5], 44);
    assert!(validate_conditional_wdl(&conditional, &repaired_raw, &another_context).is_err());
    let legacy_raw = model_observation(counter_state, Color::White, [0.2, 0.3, 0.5]);
    assert!(validate_conditional_wdl(&conditional, &repaired_raw, &legacy_raw).is_err());
    let mut another_revision = conditional.clone();
    if let RawScore::ConditionalWdl {
        context_revision, ..
    } = &mut another_revision.score
    {
        *context_revision = 44;
    }
    assert!(validate_conditional_wdl(&another_revision, &repaired_raw, &counter_raw).is_err());
    let mut mismatched_raw = counter_raw.clone();
    mismatched_raw
        .model_value_identity
        .as_mut()
        .unwrap()
        .model_epoch = [8; 32];
    assert!(validate_conditional_wdl(&conditional, &repaired_raw, &mismatched_raw).is_err());
    let mut malformed_raw = counter_raw.clone();
    malformed_raw.score = RawScore::Wdl {
        win: f32::NAN,
        draw: 0.3,
        loss: 0.5,
        perspective: Color::White,
    };
    assert!(validate_conditional_wdl(&conditional, &repaired_raw, &malformed_raw).is_err());
    assert!(validate_conditional_wdl(&conditional, &counter_raw, &counter_raw).is_err());
    assert!(validate_conditional_wdl(&conditional, &repaired_raw, &conditional).is_err());

    let derived = store.append_observation(conditional.clone()).unwrap();
    store.dependencies.add(derived, old).unwrap();
    let closure = store
        .expand_pins(StorePins {
            situations: BTreeSet::from([old]),
            ..StorePins::default()
        })
        .unwrap();
    assert!(
        closure
            .observations
            .is_superset(&BTreeSet::from([derived, repaired, counter]))
    );
    assert!(
        closure
            .states
            .is_superset(&BTreeSet::from([state, repaired_state, counter_state]))
    );
    position.make_move(mv("g1f3")).unwrap();
    store.focus_actual_moves(position.snapshot()).unwrap();
    let receipt = store.archive_inactive(budget()).unwrap().unwrap();
    assert!(store.observations.get(derived).is_err());
    let loaded = store
        .pin_load(&receipt.observation_handle(derived), budget())
        .unwrap();
    assert_eq!(*store.observations.get(repaired).unwrap(), repaired_raw);
    assert_eq!(*store.observations.get(counter).unwrap(), counter_raw);
    let restored = store.observations.get(derived).unwrap();
    assert_eq!(*restored, conditional);
    assert!(restored.model_value_identity.is_none());
    assert!(restored.model_value_input.is_none());
    assert!(matches!(
        restored.score,
        RawScore::ConditionalWdl {
            context_revision: 43,
            ..
        }
    ));
    store.release_loaded(loaded.pin).unwrap();
}

#[test]
fn only_fresh_committed_actual_root_receipt_unlocks_a_full_hot_store() {
    let root = TestRoot::new();
    let mut small = limits();
    small.states = 1;
    small.situations = 1;
    let mut store = PalsStores::new(small);
    store.enable_archive(root.config()).unwrap();
    let mut position = Position::startpos();
    let old = store.focus_actual_moves(position.snapshot()).unwrap();
    let old_state = store.situations.get(old).unwrap().state;
    let old_generation = store.generation();
    position.make_move(mv("e2e4")).unwrap();
    assert!(matches!(
        store.focus_actual_moves(position.snapshot()),
        Err(StoreError::Capacity(_))
    ));
    assert_eq!(store.root(), Some(old));
    let unrelated = store.archive_engine_records(&[], &[], budget()).unwrap();
    assert!(store.retire_actual_root_for_archive(&unrelated).is_err());
    assert_eq!(store.generation(), old_generation);
    let receipt = store
        .archive_engine_records(
            &[],
            &[EngineArchiveNode {
                state: old_state,
                situation: old,
                edges: vec![],
                metadata: b"root-v1".to_vec(),
            }],
            budget(),
        )
        .unwrap();
    let lookup = store
        .lookup_engine_archive(old_state, budget())
        .unwrap()
        .unwrap();
    assert!(store.retire_actual_root_for_archive(&lookup).is_err());
    assert_eq!(store.root(), Some(old));
    assert_eq!(
        store.retire_actual_root_for_archive(&receipt).unwrap(),
        Some(old)
    );
    assert_eq!(store.root(), None);
    assert_eq!(store.generation(), old_generation + 1);
    let reclaimed = store.archive_inactive(budget()).unwrap().unwrap();
    assert_eq!(reclaimed.after.states, 0);
    let new = store.focus_actual_moves(position.snapshot()).unwrap();
    assert_ne!(old, new);
    assert!(store.situations.get(old).is_err());
    assert!(store.states.get(old_state).is_err());
    assert!(store.retire_actual_root_for_archive(&receipt).is_err());
}

fn foreign_identity() -> crate::cpu_checker::ExternalCheckerIdentity {
    use crate::cpu_checker::{ExternalModelMetadata, ExternalTrainingKnowledge};
    crate::cpu_checker::ExternalCheckerIdentity {
        adapter_semantics: "fixture-external-archive-raw/1".into(),
        binary_sha256: "a".repeat(64),
        launch_arguments_sha256: "b".repeat(64),
        declared_name: "fixture".into(),
        declared_version: "1".into(),
        declared_source: "test-source".into(),
        declared_license: "MIT".into(),
        options: BTreeMap::new(),
        assets: vec![],
        model_metadata: ExternalModelMetadata {
            weights_sha256: None,
            training: ExternalTrainingKnowledge::Unknown,
            declared_rights: None,
            precision: None,
        },
    }
}
fn foreign_key(state: StateId) -> TaskKey {
    let mut task = key(state);
    task.value_identity = None;
    task.checker_identity = Some(CheckerIdentity::ExternalUci(foreign_identity()));
    task.cpu_condition = Some("archive-fixture-external-conditions".into());
    task
}
fn foreign_observation(
    state: StateId,
    execution: ExecutionId,
    pv: LineId,
    movement: BoardMove,
    perspective: Color,
    request_id: u64,
) -> Observation {
    let report = ExternalCheckerReport {
        identity: foreign_identity(),
        observed_uci: crate::cpu_checker::ExternalUciIdentity {
            name: "fixture".into(),
            author: None,
        },
        request_id,
        best_move: Some(movement),
        pv: vec![movement],
        score: ExternalRawScore::MateMoves(19),
        bound: ExternalBound::Lower,
        wdl_per_mille: Some([900, 50, 50]),
        perspective,
        requested_depth: 4,
        reported_depth: Some(1),
        seldepth: Some(3),
        root_restricted: false,
        completion: ExternalCompletion::BestMove,
        work: CheckerWork {
            nodes: Some(20),
            qnodes: None,
            tt_hits: None,
        },
        elapsed: Duration::from_millis(1),
    };
    let mut o = observation(state);
    o.source = external_source_id(&report.identity.adapter_semantics);
    o.value_identity = None;
    o.checker_identity = Some(CheckerIdentity::ExternalUci(report.identity.clone()));
    o.checker_work = Some(report.work);
    o.cpu_condition = Some("archive-fixture-external-conditions".into());
    o.cpu_pv = Some(pv);
    o.scope = EvidenceScope::ExternalUci {
        requested_depth: report.requested_depth,
        reported_depth: report.reported_depth,
        seldepth: report.seldepth,
        bound: report.bound,
    };
    o.score = RawScore::ExternalUci {
        value: report.score,
        bound: report.bound,
        perspective,
        wdl_per_mille: report.wdl_per_mille,
    };
    o.budget = 20;
    o.kind = ObservationKind::ExternalCpuAnalysis;
    o.execution = Some(execution);
    o.external_report = Some(Box::new(report));
    o
}

fn live_profiled_fixture(
    store: &mut PalsStores,
    requested: ProfiledExternalConstraint,
) -> Observation {
    let position = Position::startpos();
    let root = store.focus_actual_moves(position.snapshot()).unwrap();
    let state = store.situations.get(root).unwrap().state;
    let mut key = foreign_key(state);
    key.question = TaskQuestion::ProfiledExternal { requested };
    key.requested_depth = requested.requested_depth().unwrap_or(0);
    key.node_budget = requested.requested_nodes().unwrap_or(0);
    let TaskAdmission::Start(execution) = store
        .request_task(
            key,
            TaskConsumer {
                id: 71,
                situation: root,
                revision: 0,
                generation: store.generation(),
                deadline_tick: 100,
            },
            0,
        )
        .unwrap()
    else {
        panic!("profiled archive task")
    };
    let pv = store.append_cpu_pv(&position, &[mv("e2e4")]).unwrap();
    let mut raw = foreign_observation(state, execution, pv, mv("e2e4"), Color::White, 1);
    let report = raw.external_report.as_mut().unwrap();
    report.requested_depth = requested.requested_depth().unwrap_or(0);
    raw.scope = EvidenceScope::ProfiledExternalUci {
        requested,
        issued: requested,
        reported_depth: report.reported_depth,
        seldepth: report.seldepth,
        bound: report.bound,
    };
    raw
}

#[test]
fn retirement_writers_preserve_existing_wire_bytes_and_unpublished_source_values() {
    let mut store = PalsStores::new(limits());
    let raw = live_profiled_fixture(
        &mut store,
        ProfiledExternalConstraint::Nodes { reported_nodes: 16 },
    );
    let source = store.states.get(raw.state).unwrap().clone();
    let pv = raw.cpu_pv.unwrap();
    let before = store.hot_stats();
    let mut actual = Vec::new();
    store
        .write_retirement_state(raw.state, &mut actual as &mut dyn Write)
        .unwrap();
    assert_eq!(
        actual,
        serde_json::to_vec(&StateWire::encode(raw.state, &source).unwrap()).unwrap()
    );
    actual.clear();
    store.write_retirement_line(pv, &mut actual).unwrap();
    assert_eq!(
        actual,
        serde_json::to_vec(&LineWire::encode(pv, store.lines.get(pv).unwrap())).unwrap()
    );
    actual.clear();
    store
        .write_retirement_observation_value(&raw, &mut actual)
        .unwrap();
    let expected = serde_json::to_vec(&ObservationWire::encode(&raw).unwrap()).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(store.hot_stats(), before);
    assert_eq!(store.observations.len(), 0);
    let report = raw.external_report.as_deref().unwrap();
    actual.clear();
    store
        .write_retirement_external_report(report, &mut actual)
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&actual).unwrap(),
        serde_json::from_slice::<serde_json::Value>(&expected).unwrap()["external"]
    );
    let observation = store.append_observation(raw.clone()).unwrap();
    actual.clear();
    store
        .write_retirement_observation(observation, &mut actual)
        .unwrap();
    assert_eq!(actual, expected);
    let record = crate::pals::engine::RoleRecord {
        revision: 7,
        parent_revision: Some(3),
        supersedes_revision: Some(2),
        origin_state: raw.state,
        kind: crate::pals::engine::RecordKind::Counterexample,
        line: vec![mv("e2e4")],
        value: None,
        completed_depth: 0,
        score_scope: None,
        cpu_observation: Some(observation),
        perspective: Color::White,
        critical: true,
    };
    actual.clear();
    store
        .write_retirement_role_record(&record, &mut actual)
        .unwrap();
    assert_eq!(
        actual,
        serde_json::to_vec(&RoleRecordWire::encode(&record).unwrap()).unwrap()
    );
    // An original source snapshot is not replaced by today's hot state/board.
    let absent = StateId(usize::MAX);
    actual.clear();
    assert!(store.write_retirement_state(absent, &mut actual).is_err());
    assert!(actual.is_empty());
    store
        .write_retirement_snapshot(absent, &source, &mut actual)
        .unwrap();
    assert_eq!(
        actual,
        serde_json::to_vec(&StateWire::encode(absent, &source).unwrap()).unwrap()
    );
    assert!(store.states.get(absent).is_err());
    assert_eq!(store.observations.len(), 1);
}

#[test]
fn retirement_writer_preserves_the_bounded_sink_io_failure() {
    struct FailedWriter;
    impl Write for FailedWriter {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "authored bounded writer failure",
            ))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut store = PalsStores::new(limits());
    let raw = live_profiled_fixture(
        &mut store,
        ProfiledExternalConstraint::Nodes { reported_nodes: 16 },
    );
    let before = store.hot_stats();
    assert_eq!(
        store.write_retirement_observation_value(&raw, &mut FailedWriter),
        Err(StoreError::ArchiveIo {
            stage: "retirement record serialization",
            kind: std::io::ErrorKind::BrokenPipe
        })
    );
    assert_eq!(store.hot_stats(), before);
    assert_eq!(store.observations.len(), 0);
}

#[test]
fn retirement_closure_keeps_original_consumers_and_checks_unique_pin_bound() {
    let mut store = PalsStores::new(limits());
    let position = Position::startpos();
    let root = store.focus_actual_moves(position.snapshot()).unwrap();
    let state = store.situations.get(root).unwrap().state;
    let focus = store.append_cpu_pv(&position, &[mv("e2e4")]).unwrap();
    let other = store.situations.insert(state, focus).unwrap();
    let original = TaskConsumer {
        id: 81,
        situation: root,
        revision: 0,
        generation: store.generation(),
        deadline_tick: 100,
    };
    let TaskAdmission::Start(execution) = store.request_task(key(state), original, 0).unwrap()
    else {
        panic!("retirement closure task")
    };
    assert!(
        matches!(store.request_task(key(state), TaskConsumer { id: 82, situation: other, ..original }, 0).unwrap(), TaskAdmission::Join(id) if id == execution)
    );
    store.tasks.cancel_consumer(execution, original.id).unwrap();
    store.tasks.fail(execution).unwrap();
    let evidence = store.append_observation(observation(state)).unwrap();
    store.dependencies.add(evidence, other).unwrap();
    let seed = StorePins {
        executions: BTreeSet::from([execution]),
        ..StorePins::default()
    };
    let original_seed = seed.clone();
    let before = store.hot_stats();
    let closure = store.retirement_dependency_closure(&seed, 32).unwrap();
    assert_eq!(closure.situations, BTreeSet::from([root, other]));
    assert!(closure.lines.contains(&focus));
    assert!(closure.states.contains(&state));
    assert!(closure.observations.contains(&evidence));
    assert_eq!(
        store.retirement_dependency_edges().collect::<Vec<_>>(),
        vec![(evidence, &BTreeSet::from([other]))]
    );
    assert_eq!(
        store
            .retirement_dependency_closure(&seed, closure.count())
            .unwrap(),
        closure
    );
    assert_eq!(
        store.retirement_dependency_closure(&seed, closure.count() - 1),
        Err(StoreError::Capacity("retirement dependency closure"))
    );
    assert_eq!(
        store.retirement_dependency_closure(&seed, 0),
        Err(StoreError::Capacity("retirement dependency closure"))
    );
    // Existing archive selection semantics do not start retaining historical
    // consumers merely because the retirement reader requires their records.
    let legacy = store.expand_pins(seed.clone()).unwrap();
    assert!(legacy.situations.is_empty());
    assert!(!legacy.observations.contains(&evidence));
    assert_eq!(seed, original_seed);
    assert_eq!(store.hot_stats(), before);
}

#[test]
fn profiled_archive_roundtrip_retains_missing_limits_raw_units_and_question() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let requested = ProfiledExternalConstraint::MoveTime { milliseconds: 100 };
    let raw = live_profiled_fixture(&mut store, requested);
    let execution = raw.execution.unwrap();
    let wire = ObservationWire::encode(&raw).unwrap();
    let json = serde_json::to_vec(&wire).unwrap();
    assert_eq!(
        serde_json::from_slice::<ObservationWire>(&json)
            .unwrap()
            .decode()
            .unwrap(),
        raw
    );
    let task_json = serde_json::to_vec(store.tasks.get(execution).unwrap()).unwrap();
    let decoded: TaskRecord = serde_json::from_slice(&task_json).unwrap();
    assert_eq!(decoded.key, store.tasks.get(execution).unwrap().key);
    let observation = store
        .append_observation_checked_with_archive_budget(raw.clone(), &mut budget())
        .unwrap();
    assert!(store.complete_task(execution, observation).is_err());
    store.tasks.fail(execution).unwrap();
    let mut next = Position::startpos();
    next.make_move(mv("e2e4")).unwrap();
    store.focus_actual_moves(next.snapshot()).unwrap();
    let receipt = store.archive_inactive(budget()).unwrap().unwrap();
    assert!(store.observations.get(observation).is_err());
    let loaded = store
        .pin_load(&receipt.observation_handle(observation), budget())
        .unwrap();
    assert_eq!(store.observations.get(observation).unwrap(), &raw);
    assert_eq!(
        store.tasks.get(execution).unwrap().key.question,
        TaskQuestion::ProfiledExternal { requested }
    );
    assert_eq!(store.tasks.get(execution).unwrap().key.requested_depth, 0);
    assert_eq!(store.tasks.get(execution).unwrap().key.node_budget, 0);
    assert_eq!(
        store.tasks.get(execution).unwrap().status,
        TaskStatus::Failed
    );
    drop(loaded);
}

#[test]
fn controlled_retirement_closure_interrupts_clone_and_parent_walk_without_store_mutation() {
    let mut store = PalsStores::new(StoreLimits {
        line_chunks: 64,
        line_plies: 64,
        ..limits()
    });
    let root = store
        .focus_actual_moves(Position::startpos().snapshot())
        .unwrap();
    let mut line = store.situations.get(root).unwrap().focus;
    let mut all_lines = BTreeSet::from([line]);
    // A source-authored coordinate-chunk graph exercises parent traversal.
    // This fixture issues no checked CPU PV, Rules proof or execution receipt.
    for _ in 0..24 {
        line = store.lines.append(line, &[mv("e2e4")]).unwrap();
        all_lines.insert(line);
    }
    for seed in [
        StorePins {
            lines: BTreeSet::from([line]),
            ..StorePins::default()
        },
        StorePins {
            lines: all_lines,
            ..StorePins::default()
        },
    ] {
        let original = seed.clone();
        let before = store.hot_stats();
        let expected = store.retirement_dependency_closure(&seed, 64).unwrap();
        let mut total_checks = 0usize;
        let completed = store
            .retirement_dependency_closure_controlled(&seed, 64, &mut || {
                total_checks += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(completed, expected);
        assert!(total_checks > 64);
        for (stop_at, error) in [
            (1, StoreError::ArchiveCanceled),
            (3, StoreError::ArchiveDeadline),
            (27, StoreError::ArchiveCanceled),
            (total_checks / 2, StoreError::ArchiveDeadline),
            (total_checks, StoreError::ArchiveCanceled),
        ] {
            let mut checks = 0usize;
            assert_eq!(
                store.retirement_dependency_closure_controlled(&seed, 64, &mut || {
                    checks += 1;
                    if checks == stop_at {
                        Err(error.clone())
                    } else {
                        Ok(())
                    }
                }),
                Err(error.clone())
            );
            assert_eq!(checks, stop_at);
            assert_eq!(seed, original);
            assert_eq!(store.hot_stats(), before);
        }
    }
}

#[test]
fn profiled_cold_load_rejects_an_individually_valid_but_replaced_constraint() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let raw = live_profiled_fixture(
        &mut store,
        ProfiledExternalConstraint::MoveTime { milliseconds: 100 },
    );
    let execution = raw.execution.unwrap();
    let observation = store
        .append_observation_checked_with_archive_budget(raw, &mut budget())
        .unwrap();
    store.tasks.fail(execution).unwrap();
    // Corruption fixture bypasses the public writer; the committed segment has
    // a valid checksum, so its original task/observation relation must be checked.
    let key = &mut store.tasks.tasks.get_mut(execution.0).unwrap().key;
    key.question = TaskQuestion::ProfiledExternal {
        requested: ProfiledExternalConstraint::Nodes { reported_nodes: 16 },
    };
    key.node_budget = 16;
    let mut next = Position::startpos();
    next.make_move(mv("e2e4")).unwrap();
    store.focus_actual_moves(next.snapshot()).unwrap();
    let receipt = store.archive_inactive(budget()).unwrap().unwrap();
    let before = store.hot_stats();
    assert_eq!(
        store
            .pin_load(&receipt.observation_handle(observation), budget())
            .unwrap_err(),
        StoreError::ArchiveIntegrity("profiled observation task constraint")
    );
    assert_eq!(store.hot_stats(), before);
    assert!(store.observations.get(observation).is_err());
}

#[test]
fn foreign_request_uniqueness_scan_shares_explicit_budget_and_fails_closed() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    let old = store.focus_actual_moves(position.snapshot()).unwrap();
    let state = store.situations.get(old).unwrap().state;
    let TaskAdmission::Start(execution) = store
        .request_task(
            foreign_key(state),
            TaskConsumer {
                id: 71,
                situation: old,
                revision: 0,
                generation: store.generation(),
                deadline_tick: 100,
            },
            0,
        )
        .unwrap()
    else {
        panic!("new foreign task")
    };
    let pv = store.append_cpu_pv(&position, &[mv("e2e4")]).unwrap();
    let raw = foreign_observation(state, execution, pv, mv("e2e4"), Color::White, 1);
    let before = store.hot_stats();
    assert_eq!(
        store.append_observation(raw.clone()).unwrap_err(),
        StoreError::ArchiveBudgetRequired
    );
    assert_eq!(store.hot_stats(), before);
    let mut shared = budget();
    let accepted = store
        .append_observation_checked_with_archive_budget(raw.clone(), &mut shared)
        .unwrap();
    store.complete_task(execution, accepted).unwrap();
    position.make_move(mv("e2e4")).unwrap();
    let current = store.focus_actual_moves(position.snapshot()).unwrap();
    let receipt = store
        .archive_inactive_with_budget(&mut shared)
        .unwrap()
        .unwrap();
    assert!(store.observations.get(accepted).is_err());
    let current_state = store.situations.get(current).unwrap().state;
    let TaskAdmission::Start(other) = store
        .request_task(
            foreign_key(current_state),
            TaskConsumer {
                id: 72,
                situation: current,
                revision: 0,
                generation: store.generation(),
                deadline_tick: 100,
            },
            0,
        )
        .unwrap()
    else {
        panic!("another foreign task")
    };
    let pv = store.append_cpu_pv(&position, &[mv("e7e5")]).unwrap();
    let duplicate = foreign_observation(current_state, other, pv, mv("e7e5"), Color::Black, 1);
    let before = store.hot_stats();
    let prior_bytes = shared.max_bytes;
    assert_eq!(
        store
            .append_observation_checked_with_archive_budget(duplicate.clone(), &mut shared)
            .unwrap_err(),
        StoreError::InvalidEvidence("foreign physical request belongs to another archived task")
    );
    assert_eq!(prior_bytes - shared.max_bytes, receipt.committed_bytes);
    assert_eq!(store.hot_stats(), before);
    let unique = foreign_observation(current_state, other, pv, mv("e7e5"), Color::Black, 2);
    let mut small = ArchiveIoBudget {
        max_bytes: receipt.committed_bytes - 1,
        ..budget()
    };
    assert_eq!(
        store
            .append_observation_checked_with_archive_budget(unique.clone(), &mut small)
            .unwrap_err(),
        StoreError::ArchiveByteBudget
    );
    assert_eq!(store.hot_stats(), before);
    let mut expired = ArchiveIoBudget {
        deadline: Instant::now(),
        ..budget()
    };
    assert_eq!(
        store
            .append_observation_checked_with_archive_budget(unique.clone(), &mut expired)
            .unwrap_err(),
        StoreError::ArchiveDeadline
    );
    assert_eq!(store.hot_stats(), before);
    let prior_bytes = shared.max_bytes;
    let new_id = store
        .append_observation_checked_with_archive_budget(unique.clone(), &mut shared)
        .unwrap();
    assert_eq!(prior_bytes - shared.max_bytes, receipt.committed_bytes);
    assert_eq!(*store.observations.get(new_id).unwrap(), unique);

    let file = store.archive.as_ref().unwrap().path(receipt.generation);
    use std::io::{Seek, SeekFrom};
    let mut output = OpenOptions::new().write(true).open(&file).unwrap();
    output.seek(SeekFrom::End(-1)).unwrap();
    output.write_all(b"!").unwrap();
    output.sync_all().unwrap();
    let third = foreign_observation(current_state, other, pv, mv("e7e5"), Color::Black, 3);
    let before = store.hot_stats();
    let prior_bytes = shared.max_bytes;
    assert_eq!(
        store
            .append_observation_checked_with_archive_budget(third, &mut shared)
            .unwrap_err(),
        StoreError::ArchiveIntegrity("archive checksum")
    );
    assert_eq!(prior_bytes - shared.max_bytes, receipt.committed_bytes);
    assert_eq!(store.hot_stats(), before);
    drop(output);
    fs::remove_file(file).unwrap();
    let directory = store.archive.as_ref().unwrap().directory.clone();
    fs::remove_dir(&directory).unwrap();
    let third = foreign_observation(current_state, other, pv, mv("e7e5"), Color::Black, 3);
    assert!(matches!(
        store.append_observation_checked_with_archive_budget(third, &mut shared),
        Err(StoreError::ArchiveIo { .. })
    ));
    assert_eq!(store.hot_stats(), before);
}

#[test]
fn bounded_encode_buffer_reserves_from_length_before_crossing_capacity() {
    let mut buffer = BoundedBuffer {
        bytes: Vec::new(),
        maximum: 9000,
        deadline: budget().deadline,
        error: None,
    };
    buffer.write_all(&[1; 8191]).unwrap();
    buffer.write_all(&[2; 2]).unwrap();
    buffer.write_all(&[3; 807]).unwrap();
    assert_eq!(buffer.bytes.len(), 9000);
    assert!(buffer.bytes.capacity() <= 9000);
    assert!(buffer.write_all(&[4]).is_err());
    assert_eq!(buffer.error, Some(StoreError::ArchiveByteBudget));
    assert_eq!(buffer.bytes.len(), 9000);
}

#[test]
fn conditional_repair_preserves_fresh_context_pair_and_active_refutation_link() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    let old = store.focus_actual_moves(position.snapshot()).unwrap();
    let state = store.situations.get(old).unwrap().state;
    let focus = store.situations.get(old).unwrap().focus;
    let refuted_line = store.append_line(focus, &[mv("e2e4"), mv("e7e5")]).unwrap();
    let repaired_line = store.append_line(focus, &[mv("e2e4"), mv("c7c5")]).unwrap();
    position.make_move(mv("e2e4")).unwrap();
    let next_actual = position.clone();
    let proposal_state = store.states.insert(position.snapshot()).unwrap();
    let proposal_raw = context_model_observation(proposal_state, Color::Black, [0.1, 0.2, 0.7], 43);
    let proposal = store.append_observation(proposal_raw).unwrap();
    position.make_move(mv("e7e5")).unwrap();
    let counter_state = store.states.insert(position.snapshot()).unwrap();
    let counter = store
        .append_observation(context_model_observation(
            counter_state,
            Color::White,
            [0.2, 0.3, 0.5],
            43,
        ))
        .unwrap();
    let mut refutation = model_observation(state, Color::White, [0.2, 0.3, 0.5]);
    refutation.model_value_identity = None;
    refutation.model_value_input = None;
    refutation.kind = ObservationKind::Refutation;
    refutation.line = Some(refuted_line);
    refutation.score = RawScore::ConditionalWdl {
        expectation: 0.2 - 0.5,
        perspective: Color::White,
        repaired: proposal,
        counter,
        context_revision: 43,
    };
    let refutation_id = store.append_observation(refutation.clone()).unwrap();
    store
        .refute_continuation(old, refuted_line, refutation_id)
        .unwrap();
    let before_raw = context_model_observation(counter_state, Color::White, [0.2, 0.3, 0.5], 44);
    let before = store.append_observation(before_raw.clone()).unwrap();
    let mut repaired_position = next_actual.clone();
    repaired_position.make_move(mv("c7c5")).unwrap();
    let after_state = store.states.insert(repaired_position.snapshot()).unwrap();
    let after_raw = context_model_observation(after_state, Color::White, [0.4, 0.3, 0.3], 44);
    let after = store.append_observation(after_raw.clone()).unwrap();
    let mut repair = refutation.clone();
    repair.kind = ObservationKind::Repair;
    repair.line = Some(repaired_line);
    repair.supersedes = Some(refutation_id);
    repair.score = RawScore::ConditionalRepairWdl {
        expectation: 0.4 - 0.3,
        perspective: Color::White,
        before,
        after,
        context_revision: 44,
    };
    let hot = store.hot_stats();
    let mut invalid = repair.clone();
    invalid.supersedes = None;
    assert!(store.append_observation(invalid).is_err());
    let mut invalid = repair.clone();
    invalid.supersedes = Some(before);
    assert!(store.append_observation(invalid).is_err());
    let mut invalid = repair.clone();
    invalid.model_value_identity = before_raw.model_value_identity.clone();
    invalid.model_value_input = before_raw.model_value_input;
    assert!(store.append_observation(invalid).is_err());
    let mut invalid = repair.clone();
    invalid.line = None;
    assert!(store.append_observation(invalid).is_err());
    let mut invalid = repair.clone();
    invalid.kind = ObservationKind::Refutation;
    assert!(store.append_observation(invalid).is_err());
    for score in [
        RawScore::ConditionalRepairWdl {
            expectation: 0.2 - 0.5,
            perspective: Color::White,
            before,
            after: before,
            context_revision: 44,
        },
        RawScore::ConditionalRepairWdl {
            expectation: 0.2 - 0.5,
            perspective: Color::White,
            before: after,
            after: before,
            context_revision: 44,
        },
        RawScore::ConditionalRepairWdl {
            expectation: 0.5,
            perspective: Color::White,
            before,
            after,
            context_revision: 44,
        },
        RawScore::ConditionalRepairWdl {
            expectation: 0.4 - 0.3,
            perspective: Color::White,
            before,
            after,
            context_revision: 45,
        },
    ] {
        let mut invalid = repair.clone();
        invalid.score = score;
        assert!(store.append_observation(invalid).is_err());
    }
    let different_context =
        context_model_observation(after_state, Color::White, [0.4, 0.3, 0.3], 45);
    assert!(
        validate_conditional_repair_wdl(&repair, &before_raw, &different_context, &refutation)
            .is_err()
    );
    let mut wrong_model = after_raw.clone();
    wrong_model
        .model_value_identity
        .as_mut()
        .unwrap()
        .model_epoch = [8; 32];
    assert!(
        validate_conditional_repair_wdl(&repair, &before_raw, &wrong_model, &refutation).is_err()
    );
    let legacy_raw = model_observation(counter_state, Color::White, [0.2, 0.3, 0.5]);
    assert!(
        validate_conditional_repair_wdl(&repair, &legacy_raw, &after_raw, &refutation).is_err()
    );
    assert_eq!(store.hot_stats(), hot);
    let repaired_observation = store.append_observation(repair.clone()).unwrap();
    store
        .repair(old, refuted_line, repaired_line, repaired_observation)
        .unwrap();
    let closure = store
        .expand_pins(StorePins {
            observations: BTreeSet::from([repaired_observation]),
            ..StorePins::default()
        })
        .unwrap();
    assert!(closure.observations.is_superset(&BTreeSet::from([
        before,
        after,
        repaired_observation,
        refutation_id,
        proposal,
        counter,
    ])));
    store.focus_actual_moves(next_actual.snapshot()).unwrap();
    let receipt = store.archive_inactive(budget()).unwrap().unwrap();
    assert!(store.observations.get(repaired_observation).is_err());
    let loaded = store
        .pin_load(&receipt.situation_handle(old), budget())
        .unwrap();
    let restored_situation = loaded
        .situations
        .iter()
        .find(|(prior, _)| *prior == old)
        .unwrap()
        .1;
    assert_ne!(restored_situation, old);
    assert_eq!(*store.observations.get(before).unwrap(), before_raw);
    assert_eq!(*store.observations.get(after).unwrap(), after_raw);
    assert_eq!(
        *store.observations.get(repaired_observation).unwrap(),
        repair
    );
    assert_eq!(
        store
            .situations
            .get(restored_situation)
            .unwrap()
            .conclusions
            .get(refuted_line)
            .unwrap()
            .status,
        ContinuationStatus::RepairedBy(repaired_line)
    );
    assert_eq!(
        store
            .situations
            .get(restored_situation)
            .unwrap()
            .conclusions
            .get(repaired_line)
            .unwrap()
            .evidence,
        Some(repaired_observation)
    );
    store.release_loaded(loaded.pin).unwrap();
}

#[test]
fn legacy_wdl_wire_stays_unchanged_while_context_and_repair_are_distinct() {
    let legacy = model_observation(StateId(0), Color::White, [0.4, 0.3, 0.3]);
    let wire = ObservationWire::encode(&legacy).unwrap();
    let json = serde_json::to_value(&wire).unwrap();
    assert_eq!(
        json["score"],
        serde_json::json!({"Wdl": [[0.4_f32.to_bits(), 0.3_f32.to_bits(), 0.3_f32.to_bits()], 0]})
    );
    let decoded = serde_json::from_value::<ObservationWire>(json)
        .unwrap()
        .decode()
        .unwrap();
    assert_eq!(decoded, legacy);
    let context = context_model_observation(StateId(0), Color::White, [0.4, 0.3, 0.3], u64::MAX);
    let json = serde_json::to_value(ObservationWire::encode(&context).unwrap()).unwrap();
    assert!(json["score"].get("ContextWdl").is_some());
    assert_eq!(
        serde_json::from_value::<ObservationWire>(json)
            .unwrap()
            .decode()
            .unwrap(),
        context
    );
    let mut repair = legacy.clone();
    repair.kind = ObservationKind::Repair;
    repair.line = Some(LineId(7));
    repair.supersedes = Some(ObservationId(8));
    repair.model_value_identity = None;
    repair.model_value_input = None;
    repair.score = RawScore::ConditionalRepairWdl {
        expectation: 0.4 - 0.3,
        perspective: Color::White,
        before: ObservationId(3),
        after: ObservationId(4),
        context_revision: 44,
    };
    let wire = ObservationWire::encode(&repair).unwrap();
    let mut pins = StorePins::default();
    wire.add_pins(&mut pins);
    assert_eq!(
        pins.observations,
        BTreeSet::from([ObservationId(3), ObservationId(4), ObservationId(8)])
    );
    let json = serde_json::to_value(wire).unwrap();
    assert!(json["score"].get("ConditionalRepairWdl").is_some());
    assert_eq!(
        serde_json::from_value::<ObservationWire>(json)
            .unwrap()
            .decode()
            .unwrap(),
        repair
    );
}

#[test]
fn cold_load_revalidates_raw_context_revision_before_publishing_evidence() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    let old = store.focus_actual_moves(position.snapshot()).unwrap();
    let state = store.situations.get(old).unwrap().state;
    let line = store
        .append_line(store.situations.get(old).unwrap().focus, &[mv("e2e4")])
        .unwrap();
    let repaired = store
        .append_observation(context_model_observation(
            state,
            Color::White,
            [0.8, 0.1, 0.1],
            43,
        ))
        .unwrap();
    let counter = store
        .append_observation(context_model_observation(
            state,
            Color::White,
            [0.2, 0.3, 0.5],
            43,
        ))
        .unwrap();
    let mut conditional = model_observation(state, Color::White, [0.2, 0.3, 0.5]);
    conditional.line = Some(line);
    conditional.kind = ObservationKind::Refutation;
    conditional.model_value_identity = None;
    conditional.model_value_input = None;
    conditional.score = RawScore::ConditionalWdl {
        expectation: 0.2 - 0.5,
        perspective: Color::White,
        repaired,
        counter,
        context_revision: 43,
    };
    let derived = store.append_observation(conditional).unwrap();
    // Bypass the public writer only in this corruption fixture. The segment
    // receives a valid checksum; load must independently verify its relation.
    let raw = store.observations.observations.get_mut(counter.0).unwrap();
    if let RawScore::ContextWdl {
        context_revision, ..
    } = &mut raw.score
    {
        *context_revision = 44;
    }
    position.make_move(mv("e2e4")).unwrap();
    store.focus_actual_moves(position.snapshot()).unwrap();
    let receipt = store.archive_inactive(budget()).unwrap().unwrap();
    let before = store.hot_stats();
    assert_eq!(
        store
            .pin_load(&receipt.observation_handle(derived), budget())
            .unwrap_err(),
        StoreError::InvalidEvidence(
            "conditional evidence needs actual raw WDL under its context revision"
        )
    );
    assert_eq!(store.hot_stats(), before);
    assert!(store.observations.get(derived).is_err());
    assert!(store.states.get(state).is_err());
}

#[test]
fn node_lookup_ignores_newer_record_only_segment_without_changing_generic_lookup() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let old = store
        .focus_actual_moves(Position::startpos().snapshot())
        .unwrap();
    let state = store.situations.get(old).unwrap().state;
    let nodes = store
        .archive_engine_records(
            &[],
            &[EngineArchiveNode {
                state,
                situation: old,
                edges: vec![],
                metadata: b"node-v1".to_vec(),
            }],
            budget(),
        )
        .unwrap();
    let record = crate::pals::engine::RoleRecord {
        revision: 8,
        parent_revision: None,
        supersedes_revision: None,
        origin_state: state,
        kind: crate::pals::engine::RecordKind::Proposal,
        line: vec![],
        value: None,
        completed_depth: 0,
        score_scope: None,
        cpu_observation: None,
        perspective: Color::White,
        critical: false,
    };
    let records = store
        .archive_engine_records(&[record], &[], budget())
        .unwrap();
    let generic = store
        .lookup_engine_archive(state, budget())
        .unwrap()
        .unwrap();
    assert_eq!(generic.archive.generation, records.archive.generation);
    assert_eq!(generic.nodes, 0);
    let mut shared = budget();
    let initial = shared.max_bytes;
    let with_nodes = store
        .lookup_engine_node_archive_with_budget(state, &mut shared)
        .unwrap()
        .unwrap();
    assert_eq!(with_nodes.archive.generation, nodes.archive.generation);
    assert_eq!(with_nodes.nodes, 1);
    assert_eq!(
        initial - shared.max_bytes,
        nodes.archive.committed_bytes + records.archive.committed_bytes
    );
    assert_eq!(with_nodes.archive.io_bytes, initial - shared.max_bytes);
}

#[test]
fn scoped_terminal_admission_keeps_uninstalled_situation_and_pending_raw_ids_hot() {
    let root = TestRoot::new();
    let mut small = limits();
    small.observations = 5;
    let mut store = PalsStores::new(small);
    store.enable_archive(root.config()).unwrap();
    let actual = store
        .focus_actual_moves(Position::startpos().snapshot())
        .unwrap();
    let actual_state = store.situations.get(actual).unwrap().state;
    let mate = Position::from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1").unwrap();
    let mut shared = budget();
    let original = shared;
    let pending = store
        .with_archive_allocation(StorePins::default(), &mut shared, |stores| {
            stores.insert_situation(mate.snapshot())
        })
        .unwrap();
    assert_eq!(shared.max_bytes, original.max_bytes);
    let pending_state = store.situations.get(pending).unwrap().state;
    let pending_line = store
        .append_line(store.situations.get(actual).unwrap().focus, &[mv("e2e4")])
        .unwrap();
    let mut raw = observation(actual_state);
    raw.line = Some(pending_line);
    let pending_raw = store.append_observation(raw).unwrap();
    let historical = (0..3)
        .map(|_| store.append_observation(observation(actual_state)).unwrap())
        .collect::<Vec<_>>();
    assert!(store.archive_pressure());
    let mut pins = StorePins::default();
    pins.states.insert(pending_state);
    pins.situations.insert(pending);
    pins.lines.insert(pending_line);
    pins.observations.insert(pending_raw);
    let terminal = store
        .with_archive_allocation(pins, &mut shared, |stores| {
            stores.append_rules_terminal_controlled(&mate, 0, 0, || Ok(()))
        })
        .unwrap();
    assert_eq!(store.situations.get(pending).unwrap().state, pending_state);
    assert!(store.lines.get(pending_line).is_ok());
    assert!(store.observations.get(pending_raw).is_ok());
    assert_eq!(
        store.observations.get(terminal).unwrap().state,
        pending_state
    );
    assert_eq!(
        store.observations.get(terminal).unwrap().score,
        RawScore::Terminal {
            winner: Some(Color::White)
        }
    );
    for id in historical {
        assert_eq!(
            store.observations.get(id),
            Err(StoreError::ColdRecord("observation"))
        );
    }
    let manager = store.archive.as_ref().unwrap();
    assert_eq!(manager.next_generation, 1);
    let bytes = fs::metadata(manager.path(0)).unwrap().len();
    assert_eq!(original.max_bytes - shared.max_bytes, 2 * bytes);
    assert_eq!(shared.deadline, original.deadline);
    assert!(!manager.allocation_active);
    assert!(manager.auto_budget.is_none());
    assert_eq!(manager.pins, StorePins::default());
}

#[test]
fn scoped_allocation_has_one_capacity_retry_and_returns_debits_on_both_outcomes() {
    for failures in [1, 2] {
        let root = TestRoot::new();
        let mut store = setup(&root);
        let mut position = Position::startpos();
        let old = store.focus_actual_moves(position.snapshot()).unwrap();
        let old_state = store.situations.get(old).unwrap().state;
        position.make_move(mv("e2e4")).unwrap();
        let actual = store.focus_actual_moves(position.snapshot()).unwrap();
        let mut old_pins = StorePins::default();
        old_pins.situations.insert(actual);
        store.set_archive_pins(old_pins.clone()).unwrap();
        store.states.snapshots.fail_next_reservations(failures);
        position.make_move(mv("e7e5")).unwrap();
        let mut shared = budget();
        let original = shared;
        let result = store.with_archive_allocation(StorePins::default(), &mut shared, |stores| {
            stores.insert_situation_controlled(position.snapshot(), || Ok(()))
        });
        let manager = store.archive.as_ref().unwrap();
        assert_eq!(manager.next_generation, 1);
        assert_eq!(
            original.max_bytes - shared.max_bytes,
            2 * fs::metadata(manager.path(0)).unwrap().len()
        );
        assert_eq!(shared.deadline, original.deadline);
        assert_eq!(manager.pins, old_pins);
        assert!(!manager.allocation_active);
        assert!(manager.auto_budget.is_none());
        assert!(store.states.get(old_state).is_err());
        assert!(store.situations.get(actual).is_ok());
        if failures == 1 {
            let fresh = result.unwrap();
            assert!(store.situations.get(fresh).unwrap().state.0 > old_state.0);
            assert_ne!(fresh, old);
        } else {
            assert_eq!(result, Err(StoreError::Capacity("injected hot allocation")));
        }
    }
}

#[test]
fn scoped_allowance_rejects_second_budget_and_restores_pins_after_error_and_unwind() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let actual = store
        .focus_actual_moves(Position::startpos().snapshot())
        .unwrap();
    let state = store.situations.get(actual).unwrap().state;
    let receipt = store.archive_engine_records(&[], &[], budget()).unwrap();
    let mut old_pins = StorePins::default();
    old_pins.states.insert(state);
    store.set_archive_pins(old_pins.clone()).unwrap();
    let mut extra = StorePins::default();
    extra.states.insert(state);
    extra.situations.insert(actual);
    let mut shared = budget();
    let original = shared;
    let conflict = StoreError::InvalidConditions("archive allocation allowance is active");
    let result: Result<(), StoreError> =
        store.with_archive_allocation(extra, &mut shared, |stores| {
            let pins = &stores.archive.as_ref().unwrap().pins;
            assert_eq!(pins.states, old_pins.states);
            assert!(pins.situations.contains(&actual));
            let mut other = budget();
            let other_original = other;
            assert_eq!(
                stores.with_archive_allocation(StorePins::default(), &mut other, |_| Ok(())),
                Err(conflict.clone())
            );
            assert_eq!(stores.set_archive_io_budget(other), Err(conflict.clone()));
            assert_eq!(
                stores.set_archive_pins(StorePins::default()),
                Err(conflict.clone())
            );
            assert_eq!(
                stores
                    .append_observation_checked_with_archive_budget(observation(state), &mut other),
                Err(conflict.clone())
            );
            assert!(matches!(
                stores.archive_inactive_with_budget(&mut other),
                Err(StoreError::InvalidConditions(_))
            ));
            assert!(matches!(
                stores.lookup_cold_state_id_with_budget(state, &mut other),
                Err(StoreError::InvalidConditions(_))
            ));
            assert!(matches!(
                stores.archive_engine_records_with_pins_and_budget(
                    &[],
                    &[],
                    StorePins::default(),
                    &mut other
                ),
                Err(StoreError::InvalidConditions(_))
            ));
            assert!(matches!(
                stores.read_engine_archive(&receipt, other),
                Err(StoreError::InvalidConditions(_))
            ));
            assert!(matches!(
                stores.pin_load(&receipt.archive.state_handle(state), other),
                Err(StoreError::InvalidConditions(_))
            ));
            assert_eq!(other.max_bytes, other_original.max_bytes);
            Err(StoreError::InvalidConditions("fixture operation failed"))
        });
    assert_eq!(
        result,
        Err(StoreError::InvalidConditions("fixture operation failed"))
    );
    assert_eq!(shared.max_bytes, original.max_bytes);
    assert_eq!(store.archive.as_ref().unwrap().pins, old_pins);
    assert!(store.archive.as_ref().unwrap().auto_budget.is_none());
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        store.with_archive_allocation::<()>(StorePins::default(), &mut shared, |_| {
            panic!("fixture unwind")
        })
    }));
    assert!(unwind.is_err());
    let manager = store.archive.as_ref().unwrap();
    assert_eq!(manager.pins, old_pins);
    assert!(!manager.allocation_active);
    assert!(manager.auto_budget.is_none());
    let mut invalid = StorePins::default();
    invalid.states.insert(StateId(usize::MAX));
    assert_eq!(
        store.with_archive_allocation(invalid, &mut shared, |_| -> Result<(), StoreError> {
            panic!("invalid pin must fail before operation")
        }),
        Err(StoreError::InvalidHandle("state"))
    );
    let manager = store.archive.as_ref().unwrap();
    assert_eq!(manager.pins, old_pins);
    assert!(!manager.allocation_active);
    assert!(manager.auto_budget.is_none());
    assert_eq!(shared.max_bytes, original.max_bytes);
    assert_eq!(shared.deadline, original.deadline);
    store
        .with_archive_allocation(StorePins::default(), &mut shared, |_| Ok(()))
        .unwrap();
    store.set_archive_io_budget(original).unwrap();
    assert_eq!(
        store.with_archive_allocation(StorePins::default(), &mut shared, |_| Ok(())),
        Err(conflict)
    );
    assert_eq!(
        store
            .archive
            .as_ref()
            .unwrap()
            .auto_budget
            .unwrap()
            .max_bytes,
        original.max_bytes
    );
}

#[test]
fn scoped_quota_io_bytes_and_expired_deadline_failures_preserve_hot_rows() {
    for failure in ["quota", "io", "bytes", "deadline"] {
        let root = TestRoot::new();
        let mut store = setup(&root);
        let mut position = Position::startpos();
        store.focus_actual_moves(position.snapshot()).unwrap();
        position.make_move(mv("e2e4")).unwrap();
        store.focus_actual_moves(position.snapshot()).unwrap();
        position.make_move(mv("e7e5")).unwrap();
        let before = store.hot_stats();
        store.states.snapshots.fail_next_reservations(1);
        let directory = store.archive.as_ref().unwrap().directory.clone();
        let mut shared = budget();
        match failure {
            "quota" => store.archive.as_mut().unwrap().config.game_bytes = 80,
            "io" => fs::remove_dir(&directory).unwrap(),
            "bytes" => shared.max_bytes = 0,
            "deadline" => shared.deadline = Instant::now() - Duration::from_millis(1),
            _ => unreachable!(),
        }
        let original = shared;
        let error = store
            .with_archive_allocation(StorePins::default(), &mut shared, |stores| {
                stores.insert_situation(position.snapshot())
            })
            .unwrap_err();
        match failure {
            "quota" => assert_eq!(error, StoreError::ArchiveQuota("game archive")),
            "io" => assert!(matches!(error, StoreError::ArchiveIo { .. })),
            "bytes" => assert_eq!(error, StoreError::ArchiveByteBudget),
            "deadline" => assert_eq!(error, StoreError::ArchiveDeadline),
            _ => unreachable!(),
        }
        assert_eq!(store.hot_stats(), before);
        assert_eq!(shared.max_bytes, original.max_bytes);
        assert_eq!(shared.deadline, original.deadline);
        let manager = store.archive.as_ref().unwrap();
        assert!(!manager.allocation_active);
        assert!(manager.auto_budget.is_none());
        assert_eq!(manager.pins, StorePins::default());
        store.archive.as_mut().unwrap().config.game_bytes = GAME_ARCHIVE_BYTES;
        if failure == "io" {
            fs::create_dir(&directory).unwrap();
        }
        let mut recovered = budget();
        store
            .with_archive_allocation(StorePins::default(), &mut recovered, |stores| {
                stores.insert_situation(position.snapshot())
            })
            .unwrap();
    }
}

#[test]
fn scoped_dependency_refutation_and_repair_retry_do_not_publish_partial_revisions() {
    for operation in ["dependency", "refutation", "repair"] {
        for failures in [1, 2] {
            let root = TestRoot::new();
            let mut store = setup(&root);
            let mut position = Position::startpos();
            let old = store.focus_actual_moves(position.snapshot()).unwrap();
            let old_state = store.situations.get(old).unwrap().state;
            let old_raw = store.append_observation(observation(old_state)).unwrap();
            store.add_dependency(old_raw, old).unwrap();
            position.make_move(mv("e2e4")).unwrap();
            let actual = store.focus_actual_moves(position.snapshot()).unwrap();
            let current = store.situations.get(actual).unwrap().clone();
            let old_line = store.append_line(current.focus, &[mv("e7e5")]).unwrap();
            let repaired_line = store.append_line(current.focus, &[mv("c7c5")]).unwrap();
            let mut raw = observation(current.state);
            raw.line = Some(old_line);
            raw.kind = ObservationKind::Refutation;
            let refutation = store.append_observation(raw).unwrap();
            let evidence = if operation == "repair" {
                store
                    .refute_continuation(actual, old_line, refutation)
                    .unwrap();
                let mut raw = observation(current.state);
                raw.line = Some(repaired_line);
                raw.kind = ObservationKind::Repair;
                raw.supersedes = Some(refutation);
                store.append_observation(raw).unwrap()
            } else {
                refutation
            };
            let revision = store.situations.get(actual).unwrap().revision;
            store.dependencies.fail_allocations = failures;
            let mut shared = budget();
            let original = shared;
            let result =
                store.with_archive_allocation(StorePins::default(), &mut shared, |stores| {
                    match operation {
                        "dependency" => {
                            stores.add_dependency_controlled(evidence, actual, || Ok(()))
                        }
                        "refutation" => stores.refute_continuation_controlled(
                            actual,
                            old_line,
                            evidence,
                            || Ok(()),
                        ),
                        "repair" => stores.repair_controlled(
                            actual,
                            old_line,
                            repaired_line,
                            evidence,
                            || Ok(()),
                        ),
                        _ => unreachable!(),
                    }
                });
            assert_eq!(store.archive.as_ref().unwrap().next_generation, 1);
            assert!(shared.max_bytes < original.max_bytes);
            assert!(store.states.get(old_state).is_err());
            assert!(store.observations.get(evidence).is_ok());
            let current = store.situations.get(actual).unwrap();
            if failures == 2 {
                assert_eq!(
                    result,
                    Err(StoreError::Capacity("injected dependency allocation"))
                );
                assert_eq!(current.revision, revision);
                assert_eq!(store.dependencies.affected(evidence).count(), 0);
                if operation == "repair" {
                    assert_eq!(
                        current.conclusions.get(old_line).unwrap().status,
                        ContinuationStatus::Refuted
                    );
                    assert!(current.conclusions.get(repaired_line).is_none());
                } else {
                    assert!(current.conclusions.get(old_line).is_none());
                }
            } else {
                result.unwrap();
                assert_eq!(
                    store.dependencies.affected(evidence).collect::<Vec<_>>(),
                    vec![actual]
                );
                assert_eq!(
                    current.revision,
                    revision + u64::from(operation != "dependency")
                );
                if operation == "repair" {
                    assert_eq!(
                        current.conclusions.get(old_line).unwrap().status,
                        ContinuationStatus::RepairedBy(repaired_line)
                    );
                    assert_eq!(
                        current.conclusions.get(repaired_line).unwrap().status,
                        ContinuationStatus::Supported
                    );
                }
            }
            assert!(store.archive.as_ref().unwrap().auto_budget.is_none());
        }
    }
}

#[test]
fn scoped_controls_stop_after_commit_before_initial_allocation_or_capacity_retry() {
    for forced in [false, true] {
        for cause in [StoreError::ArchiveCanceled, StoreError::ArchiveDeadline] {
            let root = TestRoot::new();
            let mut store = setup(&root);
            let mut position = Position::startpos();
            let old = store.focus_actual_moves(position.snapshot()).unwrap();
            let old_state = store.situations.get(old).unwrap().state;
            for movement in if forced {
                vec!["e2e4"]
            } else {
                vec!["e2e4", "e7e5", "g1f3"]
            } {
                position.make_move(mv(movement)).unwrap();
                store.focus_actual_moves(position.snapshot()).unwrap();
            }
            let actual = store.root().unwrap();
            if forced {
                store.states.snapshots.fail_next_reservations(1);
            }
            position
                .make_move(mv(if forced { "e7e5" } else { "b8c6" }))
                .unwrap();
            let serial_before = store.states.snapshots.next_id();
            let mut shared = budget();
            let original = shared;
            let mut calls = 0;
            let result =
                store.with_archive_allocation(StorePins::default(), &mut shared, |stores| {
                    stores.insert_situation_controlled(position.snapshot(), || {
                        calls += 1;
                        if calls == if forced { 4 } else { 2 } {
                            Err(cause.clone())
                        } else {
                            Ok(())
                        }
                    })
                });
            assert_eq!(result, Err(cause));
            assert_eq!(store.states.snapshots.next_id(), serial_before);
            assert!(store.states.find(&position.snapshot()).is_none());
            assert!(store.situations.get(actual).is_ok());
            assert!(store.states.get(old_state).is_err());
            let manager = store.archive.as_ref().unwrap();
            assert_eq!(manager.next_generation, 1);
            assert_eq!(
                original.max_bytes - shared.max_bytes,
                2 * fs::metadata(manager.path(0)).unwrap().len()
            );
            assert_eq!(shared.deadline, original.deadline);
            assert!(!manager.allocation_active);
            assert!(manager.auto_budget.is_none());
        }
    }
}

#[test]
fn scoped_pin_saturation_preserves_last_valid_rows_and_does_not_extend_budget() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let actual = store
        .focus_actual_moves(Position::startpos().snapshot())
        .unwrap();
    let state = store.situations.get(actual).unwrap().state;
    let mut pins = StorePins::default();
    pins.situations.insert(actual);
    for _ in 0..limits().observations {
        pins.observations
            .insert(store.append_observation(observation(state)).unwrap());
    }
    let before = store.hot_stats();
    let mut shared = budget();
    let original = shared;
    let result = store.with_archive_allocation(pins.clone(), &mut shared, |stores| {
        stores.append_observation(observation(state))
    });
    assert_eq!(
        result,
        Err(StoreError::PinSaturated("all hot records pinned"))
    );
    assert_eq!(store.hot_stats(), before);
    for id in pins.observations {
        assert!(store.observations.get(id).is_ok());
    }
    assert_eq!(shared.max_bytes, original.max_bytes);
    assert_eq!(shared.deadline, original.deadline);
    assert_eq!(store.archive.as_ref().unwrap().next_generation, 0);
    assert!(store.archive.as_ref().unwrap().auto_budget.is_none());
}

#[test]
fn final_control_failure_keeps_raw_evidence_without_publishing_refutation_or_repair() {
    for repairing in [false, true] {
        let root = TestRoot::new();
        let mut store = setup(&root);
        let actual = store
            .focus_actual_moves(Position::startpos().snapshot())
            .unwrap();
        let current = store.situations.get(actual).unwrap().clone();
        let refuted = store.append_line(current.focus, &[mv("e2e4")]).unwrap();
        let repaired = store.append_line(current.focus, &[mv("d2d4")]).unwrap();
        let mut raw = observation(current.state);
        raw.line = Some(refuted);
        raw.kind = ObservationKind::Refutation;
        let refutation = store.append_observation(raw).unwrap();
        let evidence = if repairing {
            store
                .refute_continuation(actual, refuted, refutation)
                .unwrap();
            let mut raw = observation(current.state);
            raw.line = Some(repaired);
            raw.kind = ObservationKind::Repair;
            raw.supersedes = Some(refutation);
            store.append_observation(raw).unwrap()
        } else {
            refutation
        };
        let retained = store.observations.get(evidence).unwrap().clone();
        let revision = store.situations.get(actual).unwrap().revision;
        let mut calls = 0;
        let mut shared = budget();
        let original = shared;
        let result = store.with_archive_allocation(StorePins::default(), &mut shared, |stores| {
            let controls = || {
                calls += 1;
                if calls == 3 {
                    Err(StoreError::ArchiveCanceled)
                } else {
                    Ok(())
                }
            };
            if repairing {
                stores.repair_controlled(actual, refuted, repaired, evidence, controls)
            } else {
                stores.refute_continuation_controlled(actual, refuted, evidence, controls)
            }
        });
        assert_eq!(result, Err(StoreError::ArchiveCanceled));
        assert_eq!(store.situations.get(actual).unwrap().revision, revision);
        assert_eq!(store.observations.get(evidence).unwrap(), &retained);
        assert_eq!(
            store.dependencies.affected(evidence).collect::<Vec<_>>(),
            vec![actual]
        );
        let conclusions = &store.situations.get(actual).unwrap().conclusions;
        assert!(conclusions.get(repaired).is_none());
        if repairing {
            assert_eq!(
                conclusions.get(refuted).unwrap().status,
                ContinuationStatus::Refuted
            );
        } else {
            assert!(conclusions.get(refuted).is_none());
        }
        assert_eq!(shared.max_bytes, original.max_bytes);
        assert!(store.archive.as_ref().unwrap().auto_budget.is_none());
    }
}

#[test]
fn unreturned_task_admission_retires_only_new_reservations_and_keeps_physical_facts() {
    for admission_kind in ["start", "join", "reuse", "resume"] {
        let root = TestRoot::new();
        let mut store = setup(&root);
        let actual = store
            .focus_actual_moves(Position::startpos().snapshot())
            .unwrap();
        let state = store.situations.get(actual).unwrap().state;
        let consumer = TaskConsumer {
            id: 1,
            situation: actual,
            revision: 0,
            generation: store.generation(),
            deadline_tick: 100,
        };
        let previous = if admission_kind != "start" {
            let TaskAdmission::Start(execution) =
                store.request_task(key(state), consumer, 0).unwrap()
            else {
                panic!("start")
            };
            Some(execution)
        } else {
            None
        };
        let mut completed = None;
        if admission_kind == "reuse" {
            let execution = previous.unwrap();
            let mut raw = observation(state);
            raw.execution = Some(execution);
            let evidence = store.append_observation(raw).unwrap();
            store.complete_task(execution, evidence).unwrap();
            completed = Some(evidence);
        } else if admission_kind == "resume" {
            store.pause_task(previous.unwrap(), 7, None).unwrap();
        }
        let mut calls = 0;
        let mut shared = budget();
        let result = store.with_archive_allocation(StorePins::default(), &mut shared, |stores| {
            stores.request_task_controlled(
                key(state),
                TaskConsumer { id: 2, ..consumer },
                0,
                || {
                    calls += 1;
                    if calls == 3 {
                        Err(StoreError::ArchiveCanceled)
                    } else {
                        Ok(())
                    }
                },
            )
        });
        assert_eq!(result, Err(StoreError::ArchiveCanceled));
        let issued = *store.tasks.latest.get(&key(state)).unwrap();
        let task = store.tasks.get(issued).unwrap();
        assert!(
            task.consumers
                .iter()
                .find(|record| record.consumer.id == 2)
                .unwrap()
                .cancelled
        );
        match admission_kind {
            "start" => {
                assert_eq!(task.status, TaskStatus::Failed);
                assert!(store.tasks.active_states().next().is_none());
            }
            "resume" => {
                assert_eq!(task.status, TaskStatus::Failed);
                assert_eq!(task.resumed_from, previous);
                assert!(matches!(
                    store.tasks.get(previous.unwrap()).unwrap().status,
                    TaskStatus::Paused { checkpoint: 7, .. }
                ));
            }
            "join" => {
                assert_eq!(issued, previous.unwrap());
                assert_eq!(task.status, TaskStatus::InFlight);
                assert!(!task.consumers[0].cancelled);
            }
            "reuse" => {
                assert_eq!(issued, previous.unwrap());
                assert_eq!(task.status, TaskStatus::Completed(completed.unwrap()));
                assert!(store.observations.get(completed.unwrap()).is_ok());
            }
            _ => unreachable!(),
        }
        assert!(store.archive.as_ref().unwrap().auto_budget.is_none());
    }
}

#[test]
fn controlled_foreign_scan_and_pressure_commit_return_shared_debits_on_cancel() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    let old = store.focus_actual_moves(position.snapshot()).unwrap();
    let state = store.situations.get(old).unwrap().state;
    let consumer = TaskConsumer {
        id: 71,
        situation: old,
        revision: 0,
        generation: store.generation(),
        deadline_tick: 100,
    };
    let TaskAdmission::Start(execution) =
        store.request_task(foreign_key(state), consumer, 0).unwrap()
    else {
        panic!("start")
    };
    let pv = store.append_cpu_pv(&position, &[mv("e2e4")]).unwrap();
    let raw = foreign_observation(state, execution, pv, mv("e2e4"), Color::White, 1);
    let accepted = store
        .append_observation_checked_with_archive_budget(raw, &mut budget())
        .unwrap();
    store.complete_task(execution, accepted).unwrap();
    position.make_move(mv("e2e4")).unwrap();
    let actual = store.focus_actual_moves(position.snapshot()).unwrap();
    store.archive_inactive(budget()).unwrap().unwrap();
    let state = store.situations.get(actual).unwrap().state;
    let TaskAdmission::Start(execution) = store
        .request_task(
            foreign_key(state),
            TaskConsumer {
                id: 72,
                situation: actual,
                generation: store.generation(),
                ..consumer
            },
            0,
        )
        .unwrap()
    else {
        panic!("new start")
    };
    let pv = store.append_cpu_pv(&position, &[mv("e7e5")]).unwrap();
    let raw = foreign_observation(state, execution, pv, mv("e7e5"), Color::Black, 2);
    for _ in 0..7 {
        store.append_observation(observation(state)).unwrap();
    }
    let serial_before = store.observations.observations.next_id();
    let mut calls = 0;
    let mut shared = budget();
    let original = shared;
    let result =
        store.append_observation_checked_with_archive_budget_controlled(raw, &mut shared, || {
            calls += 1;
            if calls == 2 {
                Err(StoreError::ArchiveCanceled)
            } else {
                Ok(())
            }
        });
    assert_eq!(result, Err(StoreError::ArchiveCanceled));
    assert_eq!(store.observations.observations.next_id(), serial_before);
    assert_eq!(
        store.tasks.get(execution).unwrap().status,
        TaskStatus::InFlight
    );
    assert!(store.lines.get(pv).is_ok());
    assert!(store.states.get(state).is_ok());
    let manager = store.archive.as_ref().unwrap();
    assert_eq!(manager.next_generation, 2);
    let read = fs::metadata(manager.path(0)).unwrap().len();
    let committed = fs::metadata(manager.path(1)).unwrap().len();
    assert_eq!(original.max_bytes - shared.max_bytes, read + 2 * committed);
    assert_eq!(shared.deadline, original.deadline);
    assert!(manager.auto_budget.is_none());
}

#[test]
fn checked_first_foreign_append_keeps_original_budget_and_never_reclaims_or_retries() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    let old = store.focus_actual_moves(position.snapshot()).unwrap();
    let state = store.situations.get(old).unwrap().state;
    let consumer = TaskConsumer {
        id: 71,
        situation: old,
        revision: 0,
        generation: store.generation(),
        deadline_tick: 100,
    };
    let TaskAdmission::Start(execution) =
        store.request_task(foreign_key(state), consumer, 0).unwrap()
    else {
        panic!("start")
    };
    let pv = store.append_cpu_pv(&position, &[mv("e2e4")]).unwrap();
    let raw = foreign_observation(state, execution, pv, mv("e2e4"), Color::White, 1);
    let accepted = store
        .append_observation_checked_with_archive_budget(raw, &mut budget())
        .unwrap();
    store.complete_task(execution, accepted).unwrap();
    position.make_move(mv("e2e4")).unwrap();
    let actual = store.focus_actual_moves(position.snapshot()).unwrap();
    store.archive_inactive(budget()).unwrap().unwrap();
    let state = store.situations.get(actual).unwrap().state;
    let TaskAdmission::Start(execution) = store
        .request_task(
            foreign_key(state),
            TaskConsumer {
                id: 72,
                situation: actual,
                generation: store.generation(),
                ..consumer
            },
            0,
        )
        .unwrap()
    else {
        panic!("new start")
    };
    let pv = store.append_cpu_pv(&position, &[mv("e7e5")]).unwrap();
    store.tasks.cancel_consumer(execution, 72).unwrap();
    let retained = (0..7)
        .map(|_| store.append_observation(observation(state)).unwrap())
        .collect::<Vec<_>>();
    assert!(store.archive_pressure());
    let before = store.hot_stats();
    let serial_before = store.observations.observations.next_id();
    let mut shared = budget();
    let original = shared;
    let duplicate = foreign_observation(state, execution, pv, mv("e7e5"), Color::Black, 1);
    assert_eq!(
        store.append_observation_checked_first_with_archive_budget(duplicate, &mut shared),
        Err(StoreError::InvalidEvidence(
            "foreign physical request belongs to another archived task"
        ))
    );
    let cold_bytes = fs::metadata(store.archive.as_ref().unwrap().path(0))
        .unwrap()
        .len();
    assert_eq!(original.max_bytes - shared.max_bytes, cold_bytes);
    let raw = foreign_observation(state, execution, pv, mv("e7e5"), Color::Black, 2);
    let mut expired = ArchiveIoBudget {
        deadline: Instant::now() - Duration::from_millis(1),
        ..shared
    };
    assert_eq!(
        store.append_observation_checked_first_with_archive_budget(raw.clone(), &mut expired),
        Err(StoreError::ArchiveDeadline)
    );
    assert_eq!(expired.max_bytes, shared.max_bytes);
    let mut bytes = ArchiveIoBudget {
        max_bytes: 1,
        ..shared
    };
    assert_eq!(
        store.append_observation_checked_first_with_archive_budget(raw.clone(), &mut bytes),
        Err(StoreError::ArchiveByteBudget)
    );
    assert_eq!(bytes.max_bytes, 1);
    store.observations.observations.fail_next_reservations(2);
    for _ in 0..2 {
        let remaining = shared.max_bytes;
        assert_eq!(
            store.append_observation_checked_first_with_archive_budget(raw.clone(), &mut shared),
            Err(StoreError::Capacity("injected hot allocation"))
        );
        assert_eq!(remaining - shared.max_bytes, cold_bytes);
        assert_eq!(store.hot_stats(), before);
        assert_eq!(store.observations.observations.next_id(), serial_before);
        assert_eq!(store.archive.as_ref().unwrap().next_generation, 1);
    }
    let accepted = store
        .append_observation_checked_first_with_archive_budget(raw, &mut shared)
        .unwrap();
    assert_eq!(accepted.0, serial_before);
    assert_eq!(store.observations.len(), before.observations + 1);
    for id in retained {
        assert!(store.observations.get(id).is_ok());
    }
    assert_eq!(
        store.tasks.get(execution).unwrap().status,
        TaskStatus::CancellationRequested
    );
    let third = foreign_observation(state, execution, pv, mv("e7e5"), Color::Black, 3);
    assert_eq!(
        store.append_observation_checked_first_with_archive_budget(third, &mut shared),
        Err(StoreError::Capacity("observations"))
    );
    assert_eq!(store.archive.as_ref().unwrap().next_generation, 1);
    assert_eq!(shared.deadline, original.deadline);
    assert!(store.archive.as_ref().unwrap().auto_budget.is_none());
}
