//! Exercises the real idle archive worker; no neural execution or GPU fence.
use super::*;
use adaptive_port::archive_worker::{ArchiveOperation, ArchiveWorker};

#[test]
fn registered_root_then_new_game_closes_empty_original_owner_on_worker() {
    let mut scratch = idle_archive_scratch();
    let owner = idle_pc_from_actual_counter(port_with_archive(
        1, 2, None, Some((&scratch.0, idle_archive_limits())),
    ));
    let old_store = owner.stores().read_owner();
    let idle = owner.into_idle_archive().unwrap_or_else(|_| panic!("original idle PC"));
    let prior = port_authority_for_idle_test();
    assert_eq!(idle.last_registered_scope(), Some(prior));
    let next = SearchAuthority { game: GameGeneration(1), root: RootGeneration(9), ..prior };
    let authority = idle.adaptive_archive_authority(next).unwrap();
    let mut worker = ArchiveWorker::start().unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    let idle = prepare_idle_archive_for_transition(&mut worker, idle, authority, deadline);
    let (idle, transition) = idle_archive_work(
        &mut worker, idle, authority, ArchiveOperation::NewGame { next_authority: next }, deadline,
    ).unwrap_or_else(|error| panic!("original new game: {:?}", error.reason));
    assert_eq!(transition.next_authority, Some(next));
    assert_eq!(idle.last_registered_scope(), Some(prior));
    assert!(!old_store.matches(idle.stores()));
    let actual = idle.archive_owner_snapshot().unwrap().unwrap();
    assert!(!actual.admission_closed && actual.complete);

    let wrong = SearchAuthority { root: RootGeneration(next.root.0 + 1), ..next };
    let rejected = match worker.submit_empty_idle_close(idle, wrong, deadline) {
        Ok(_) => panic!("wrong ready scope must not be admitted"),
        Err(rejected) => rejected,
    };
    let idle = rejected.owner;
    assert_eq!(idle.last_registered_scope(), Some(prior));
    assert_eq!(idle.archive_owner_snapshot().unwrap(), Some(actual));
    let transfer = worker.submit_empty_idle_close(idle, next, deadline)
        .unwrap_or_else(|rejected| panic!("real empty close: {:?}", rejected.error));
    let completion = loop {
        if let Some(completion) = worker.try_receive_empty_idle_close(&transfer).unwrap() {
            break completion;
        }
        assert!(Instant::now() < deadline, "bounded worker completion");
        std::thread::yield_now();
    };
    let (idle, receipt) = completion.install(next, transfer.nonce(), actual.owner_id)
        .unwrap_or_else(|failure| panic!("original close rejected: {:?}", failure.failure()));
    assert_eq!(idle.last_registered_scope(), Some(prior));
    let closed = receipt.archive_owner.expect("actual owner close receipt");
    assert_eq!(closed.owner_id, actual.owner_id);
    assert!(closed.admission_closed && closed.complete);
    assert_eq!(idle.archive_owner_snapshot().unwrap(), Some(closed));
    worker.request_shutdown();
    let joined = loop {
        if let Some(joined) = worker.try_join().unwrap() { break joined; }
        assert!(Instant::now() < deadline, "bounded worker join");
        std::thread::yield_now();
    };
    assert!(joined.worker_joined && joined.owners_returned);
    drop(idle);
    scratch.1 = true;
}
