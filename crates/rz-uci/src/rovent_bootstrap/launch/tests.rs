use super::*;

#[test]
fn launch_retains_the_original_lock_version_without_projecting_it() {
    use rz_contracts::rovent::ExecutionMode;
    use rz_experiments::{
        ROVENT_EXECUTION_V2_DOMAIN, RoventExecutionManifest, RoventExecutionManifestV2,
        RoventOwnRNotApplicableV2, RoventRootTieDeclarationV2,
    };

    let wire = crate::rovent_bootstrap::tests::wire(ExecutionMode::CpuPc);
    let base = RoventExecutionManifest::from_json(&wire.to_string()).unwrap();
    let v1 = base.clone().lock().unwrap();
    let v1_bytes = serde_json::to_vec(&v1).unwrap();
    let LaunchInputLock::V1(original) = parse_launch_lock(&v1_bytes).unwrap() else {
        panic!("original V1 lock must remain V1")
    };
    assert_eq!(*original, v1);

    let v2 = RoventExecutionManifestV2 {
        domain: ROVENT_EXECUTION_V2_DOMAIN.into(),
        contract_revision: base.contract_revision,
        run_id: base.run_id,
        purpose: base.purpose,
        startup_document: base.engine.binary.clone(),
        startup_search_configuration_sha256: base.engine.search.search_configuration_sha256.clone(),
        startup_configuration_sha256: base.coordinator.policy_sha256.clone(),
        engine: base.engine,
        resources: base.resources,
        coordinator: base.coordinator,
        wall_time_max_ms: base.wall_time_max_ms,
        cleanup_max_ms: base.cleanup_max_ms,
        output_bytes_max: base.output_bytes_max,
        agent_renewable: base.agent_renewable,
        provenance: base.provenance,
        own_r_root_tie: RoventRootTieDeclarationV2::NotApplicable {
            reason: RoventOwnRNotApplicableV2::PcOnly,
        },
    }
    .lock()
    .unwrap();
    let LaunchInputLock::V2(original) =
        parse_launch_lock(&serde_json::to_vec(&v2).unwrap()).unwrap()
    else {
        panic!("original V2 lock must remain V2")
    };
    assert_eq!(*original, v2);

    let mut invalid = v2;
    invalid.canonical_sha256 = "0".repeat(64);
    let invalid_bytes = serde_json::to_vec(&invalid).unwrap();
    let legacy_failure =
        RoventInputLock::from_json(std::str::from_utf8(&invalid_bytes).unwrap()).unwrap_err();
    let failure = parse_launch_lock(&invalid_bytes).unwrap_err();
    assert_eq!(failure.stage, "InputLock");
    assert_eq!(failure.detail, error("InputLock", legacy_failure).detail);
    assert_eq!(parse_launch_lock(b"\xff").unwrap_err().stage, "InputLock");
}

fn arguments() -> Vec<String> {
    [
        "--input-lock=inputs/rovent-input-lock.json",
        "--native-assets=inputs/rovent-native-assets.json",
        "--process-epoch=23",
        "--startup-ms=60000",
        "--root-policy=dynamic-go",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

#[test]
fn product_arguments_require_one_explicit_closed_selection_and_policy() {
    let args = Arguments::parse(&arguments()).unwrap();
    assert_eq!(args.root_policy, RoventRootAdmissionPolicy::DynamicGo);
    assert_eq!(args.process_epoch, 23);
    assert_eq!(args.external_r_sources, None);
    assert!(!args.startup_diagnostics);
    assert!(!args.lifecycle_diagnostics);
    assert!(!args.initial_stop_source_diagnostics);
    assert!(args.storage.is_none());
    let mut lifecycle = arguments();
    lifecycle.push("--lifecycle-diagnostics=stderr-v1".into());
    assert!(Arguments::parse(&lifecycle).unwrap().lifecycle_diagnostics);
    lifecycle.push("--lifecycle-diagnostics=stderr-v1".into());
    assert!(Arguments::parse(&lifecycle).is_err());
    let mut diagnostics = arguments();
    diagnostics.push("--startup-diagnostics=stderr-v1".into());
    assert!(Arguments::parse(&diagnostics).unwrap().startup_diagnostics);
    diagnostics.push("--startup-diagnostics=stderr-v1".into());
    assert!(Arguments::parse(&diagnostics).is_err());
    let mut exact = arguments();
    exact[4] = "--root-policy=exact-registered".into();
    assert_eq!(
        Arguments::parse(&exact).unwrap().root_policy,
        RoventRootAdmissionPolicy::ExactRegistered
    );
    for index in 0..arguments().len() {
        let mut missing = arguments();
        missing.remove(index);
        assert!(Arguments::parse(&missing).is_err());
    }
    for duplicate in arguments() {
        let mut args = arguments();
        args.push(duplicate);
        assert!(Arguments::parse(&args).is_err());
    }
}

#[test]
fn product_arguments_reject_unregistered_paths_and_unbounded_controls() {
    for bad in [
        "--input-lock=../lock.json",
        "--native-assets=/tmp/assets.json",
        "--input-lock=inputs/../lock.json",
        "--native-assets=inputs\\assets.json",
        "--process-epoch=0",
        "--process-epoch=18446744073709551616",
        "--startup-ms=0",
        "--startup-ms=3600001",
        "--startup-ms=+1",
        "--root-policy=legacy-v5",
        "--external-r-sources=../external.json",
        "--retirement-sink=/tmp/registration.json",
        "--evidence-root=relative/evidence",
        "--unknown=true",
        "--startup-diagnostics=true",
        "--startup-diagnostics=stdout-v1",
        "--lifecycle-diagnostics=true",
        "--lifecycle-diagnostics=stdout-v1",
        "--initial-stop-source-diagnostics=true",
        "--initial-stop-source-diagnostics=stdout-v1",
    ] {
        let mut args = arguments();
        let key = bad.split_once('=').unwrap().0;
        args.retain(|argument| argument.split_once('=').unwrap().0 != key);
        args.push(bad.into());
        assert!(Arguments::parse(&args).is_err(), "{bad}");
    }
}

#[test]
fn initial_stop_source_diagnostics_require_the_explicit_lifecycle_stream() {
    let mut selected = arguments();
    selected.push("--initial-stop-source-diagnostics=stderr-v1".into());
    assert!(Arguments::parse(&selected).is_err());
    selected.push("--lifecycle-diagnostics=stderr-v1".into());
    let parsed = Arguments::parse(&selected).unwrap();
    assert!(parsed.initial_stop_source_diagnostics && parsed.lifecycle_diagnostics);
    // Dependency validation is independent of CLI option order.
    selected.swap(5, 6);
    assert!(
        Arguments::parse(&selected)
            .unwrap()
            .initial_stop_source_diagnostics
    );
    selected.push("--initial-stop-source-diagnostics=stderr-v1".into());
    assert!(Arguments::parse(&selected).is_err());
}

#[test]
fn external_source_mapping_is_explicit_closed_and_independent_of_native_assets() {
    let mut args = arguments();
    args.push("--external-r-sources=inputs/external-sources.json".into());
    assert_eq!(
        Arguments::parse(&args)
            .unwrap()
            .external_r_sources
            .as_deref(),
        Some("inputs/external-sources.json")
    );
    args.push("--external-r-sources=inputs/other.json".into());
    assert!(Arguments::parse(&args).is_err());

    let wire = br#"{"domain":"rz-rovent-external-r-sources/1","working_directory_snapshot_path":"inputs/external","assets":[]}"#;
    let selection = ExternalSelectionWire::parse(wire).unwrap();
    assert_eq!(selection.working_directory_snapshot_path, "inputs/external");
    assert!(selection.assets.is_empty());
    for bad in [
        br#"{"domain":"rz-rovent-native-assets/1","working_directory_snapshot_path":"inputs/external","assets":[]}"#.as_slice(),
        br#"{"domain":"rz-rovent-external-r-sources/1","working_directory_snapshot_path":"../external","assets":[]}"#.as_slice(),
        br#"{"domain":"rz-rovent-external-r-sources/1","working_directory_snapshot_path":"inputs/external","assets":[],"external_question_limit":{"kind":"depth","depth":1}}"#.as_slice(),
        br#"{"domain":"rz-rovent-external-r-sources/1","working_directory_snapshot_path":"inputs/external"}"#.as_slice(),
    ] {
        assert!(ExternalSelectionWire::parse(bad).is_err());
    }
}

#[test]
fn retirement_arguments_keep_the_original_registration_and_explicit_absolute_root() {
    let root = std::env::current_dir().unwrap().join("managed-evidence");
    let mut args = arguments();
    args.push("--retirement-sink=inputs/retirement-registration.json".into());
    args.push(format!("--evidence-root={}", root.display()));
    let parsed = Arguments::parse(&args).unwrap();
    assert_eq!(
        parsed.retirement_sink.as_deref(),
        Some("inputs/retirement-registration.json")
    );
    assert_eq!(parsed.evidence_root.as_deref(), Some(root.as_path()));
    for repeated in [
        "--retirement-sink=inputs/other-registration.json".into(),
        format!("--evidence-root={}", root.display()),
    ] {
        let mut repeated_args = args.clone();
        repeated_args.push(repeated);
        assert!(Arguments::parse(&repeated_args).is_err());
    }
    assert!(managed_path(root.join("..").join("elsewhere").to_str().unwrap()).is_err());
}

#[test]
fn native_asset_wire_cannot_admit_a_legacy_or_incomplete_selection() {
    assert!(SelectionWire::parse(br#"{"domain":"rz-pals-startup/5"}"#).is_err());
    assert!(
        SelectionWire::parse(br#"{"domain":"rz-rovent-native-assets/1","graphs":[]}"#).is_err()
    );
    assert!(SelectionWire::parse(br#"{} {}"#).is_err());
}

fn storage_arguments() -> Vec<String> {
    let root = std::env::current_dir().unwrap();
    vec![
        "--storage-profile=inputs/storage-profile.json".into(),
        format!("--storage-profile-sha256={}", "a".repeat(64)),
        "--storage-profile-bytes=1024".into(),
        format!("--archive-root={}", root.join("archive").display()),
        format!(
            "--storage-checkout-root={}",
            root.join("checkout").display()
        ),
        format!(
            "--storage-protected-root={}",
            root.join("protected").display()
        ),
    ]
}

#[test]
fn storage_is_off_by_default_and_requires_all_six_explicit_options() {
    assert!(Arguments::parse(&arguments()).unwrap().storage.is_none());
    let selection = storage_arguments();
    let mut complete = arguments();
    complete.extend(selection.clone());
    let parsed = Arguments::parse(&complete).unwrap();
    let storage = parsed.storage.unwrap();
    assert_eq!(storage.profile, "inputs/storage-profile.json");
    assert_eq!(storage.profile_sha256, "a".repeat(64));
    assert_eq!(storage.profile_bytes, 1024);
    // Enumerate every partial subset, rather than only one missing option.
    for mask in 1_u8..63 {
        let mut partial = arguments();
        partial.extend(
            selection
                .iter()
                .enumerate()
                .filter(|(index, _)| mask & (1 << *index) != 0)
                .map(|(_, value)| value.clone()),
        );
        assert!(Arguments::parse(&partial).is_err(), "subset {mask}");
    }
    for repeated in selection {
        let mut repeated_args = complete.clone();
        repeated_args.push(repeated);
        assert!(Arguments::parse(&repeated_args).is_err());
    }
}

#[test]
fn storage_options_reject_unpinned_inputs_and_any_selected_private_root() {
    let mut complete = arguments();
    complete.extend(storage_arguments());
    for bad in [
        "--storage-profile=../storage.json",
        "--storage-profile=/tmp/storage.json",
        "--storage-profile-sha256=abcd",
        "--storage-profile-sha256=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "--storage-profile-bytes=0",
        "--storage-profile-bytes=18446744073709551616",
        "--archive-root=relative/archive",
        "--storage-checkout-root=relative/checkout",
        "--storage-protected-root=relative/protected",
        "--storage-private-root=/tmp/not-actual-cwd",
        "--private-snapshot-root=/tmp/not-actual-cwd",
    ] {
        let key = bad.split_once('=').unwrap().0;
        let mut changed = complete.clone();
        changed.retain(|argument| argument.split_once('=').unwrap().0 != key);
        changed.push(bad.into());
        assert!(Arguments::parse(&changed).is_err(), "{bad}");
    }
    let mut too_large = complete;
    too_large.retain(|argument| !argument.starts_with("--storage-profile-bytes="));
    too_large.push(format!(
        "--storage-profile-bytes={}",
        rz_experiments::MAX_MANIFEST_BYTES + 1
    ));
    assert!(Arguments::parse(&too_large).is_err());
}
