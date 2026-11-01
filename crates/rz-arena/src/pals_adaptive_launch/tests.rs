use super::*;

fn pair(command: &str) -> PalsPairPlanV5 {
    PalsPairPlanV5 {
        opening_id: "fixed-opening".into(),
        position_command: command.into(),
        white_order: ["g0".into(), "g1".into()],
        base_ms: 120000,
        increment_ms: 1000,
        max_plies: 256,
        seed: 1,
        concurrent_games: 1,
        restart_processes_each_game: true,
        ponder: false,
        score_adjudication: false,
    }
}
fn asset(bytes: &[u8]) -> ArtifactRef {
    ArtifactRef {
        path: "pals-owned-storage.v5.json".into(),
        sha256: format!("{:x}", Sha256::digest(bytes)),
        bytes: bytes.len() as u64,
        source: "https://example.org/authored-storage-fixture".into(),
        license: "MIT".into(),
    }
}
#[test]
fn opening_preserves_literal_history_and_fen_unknown_prefix() {
    let start = opening(&pair("position startpos moves e2e4 e7e5")).unwrap();
    assert_eq!(start.initial, InitialPosition::Startpos);
    assert_eq!(start.moves, ["e2e4", "e7e5"]);
    assert_eq!(start.history, HistoryCompleteness::Complete);
    let fen = opening(&pair(
        "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves e2e4",
    ))
    .unwrap();
    assert_eq!(fen.initial, InitialPosition::Fen);
    assert_eq!(fen.history, HistoryCompleteness::UnknownPrefix);
    assert!(fen.fen.is_some());
    assert_eq!(fen.moves, ["e2e4"]);
}
#[test]
fn opening_rejects_illegal_or_terminal_input_without_creating_children() {
    assert!(opening(&pair("position startpos moves e2e5")).is_err());
    assert!(opening(&pair("position startpos garbage e2e4")).is_err());
    assert!(opening(&pair("position fen 7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")).is_err());
}
#[test]
fn storage_plan_never_turns_retained_evidence_into_automatic_eviction() {
    let marker=br#"{"domain":"rz-pals-owned-storage-v5/1","owner_id":"fixture-owner","purpose":"staging_and_cache"}"#;
    let mut p = PalsOwnedStoragePlanV5 {
        owner_id: "fixture-owner".into(),
        marker: asset(marker),
        byte_limit: PALS_STAGING_BYTES_V5,
        external_serialization_required: true,
        automatic_eviction: false,
    };
    assert!(p.validate().is_ok());
    p.automatic_eviction = true;
    assert!(p.validate().is_err());
    p.automatic_eviction = false;
    p.byte_limit = 32 << 30;
    assert!(p.validate().is_err());
}
#[cfg(target_os = "linux")]
#[test]
fn exclusive_storage_lease_covers_reservation_and_failed_owner_lifetime() {
    use std::io::Write;
    let root = std::env::temp_dir().join(format!(
        "rz-v5-owner-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    let marker=br#"{"domain":"rz-pals-owned-storage-v5/1","owner_id":"fixture-owner","purpose":"staging_and_cache"}"#;
    std::fs::File::create(root.join("pals-owned-storage.v5.json"))
        .unwrap()
        .write_all(marker)
        .unwrap();
    let p = PalsOwnedStoragePlanV5 {
        owner_id: "fixture-owner".into(),
        marker: asset(marker),
        byte_limit: PALS_STAGING_BYTES_V5,
        external_serialization_required: true,
        automatic_eviction: false,
    };
    let (a, lease) = acquire_pals_owned_storage_v5(&p, &root, 1024).unwrap();
    assert!(a.external_serialization_observed);
    assert!(acquire_pals_owned_storage_v5(&p, &root, 1024).is_err());
    drop(lease);
    assert!(acquire_pals_owned_storage_v5(&p, &root, PALS_STAGING_BYTES_V5).is_err());
    let (_, lease) = acquire_pals_owned_storage_v5(&p, &root, 1024).unwrap();
    drop(lease);
    // Only this test's two explicitly created entries are removed, without a
    // recursive path calculation or deletion of another task's managed root.
    std::fs::remove_file(root.join("pals-owned-storage.v5.json")).unwrap();
    std::fs::remove_dir(&root).unwrap();
}

/// A historical sparse logical file must not prevent a new advisory run from
/// acquiring its genuine marker lease. Strict V5 retains its registered cap.
#[cfg(target_os = "linux")]
#[test]
fn advisory_storage_preserves_owner_when_historical_bytes_exceed_v5_limit() {
    use std::io::Write;
    use std::path::PathBuf;
    struct OwnedFixture(PathBuf);
    impl Drop for OwnedFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(self.0.join("retained-sparse"));
            let _ = std::fs::remove_file(self.0.join("pals-owned-storage.v5.json"));
            let _ = std::fs::remove_dir(&self.0);
        }
    }
    let root = OwnedFixture(std::env::temp_dir().join(format!(
        "rz-advisory-owner-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
    )));
    std::fs::create_dir(&root.0).unwrap();
    let marker = br#"{"domain":"rz-pals-owned-storage-v5/1","owner_id":"fixture-owner","purpose":"staging_and_cache"}"#;
    std::fs::File::create(root.0.join("pals-owned-storage.v5.json"))
        .unwrap()
        .write_all(marker)
        .unwrap();
    let retained = std::fs::File::create(root.0.join("retained-sparse")).unwrap();
    retained.set_len(PALS_STAGING_BYTES_V5 + 1).unwrap();
    drop(retained);
    let p = PalsOwnedStoragePlanV5 {
        owner_id: "fixture-owner".into(),
        marker: asset(marker),
        byte_limit: PALS_STAGING_BYTES_V5,
        external_serialization_required: true,
        automatic_eviction: false,
    };
    assert!(acquire_pals_owned_storage_v5(&p, &root.0, 1024).is_err());
    let (audit, lease) = storage::acquire_pals_owned_storage_advisory(&p, &root.0).unwrap();
    assert_eq!(audit.observation_status, "not_measured");
    assert!(audit.measured_root_bytes.is_none() && audit.threshold_exceeded.is_none());
    assert!(audit.external_serialization_observed);
    assert!(storage::acquire_pals_owned_storage_advisory(&p, &root.0).is_err());
    let continued = storage::readmit_pals_owned_storage_advisory(&p, &root.0, &lease).unwrap();
    assert_eq!(continued.observation_status, "not_measured");
    assert!(continued.measured_root_bytes.is_none() && continued.threshold_exceeded.is_none());
    assert_eq!(
        std::fs::metadata(root.0.join("retained-sparse"))
            .unwrap()
            .len(),
        PALS_STAGING_BYTES_V5 + 1
    );
    drop(lease);
}
