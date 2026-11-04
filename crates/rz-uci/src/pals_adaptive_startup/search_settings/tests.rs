use super::*;
use std::time::Duration;

// Independent declaration fixture: no engine, checker, TT or worker is created.
fn declarations() -> (BTreeMap<String, String>, BTreeMap<String, String>) {
    let search = [
        ("beam_width", "3"),
        ("line_plies", "7"),
        ("max_nodes", "8192"),
        ("max_records", "64"),
        ("max_role_calls", "123"),
        ("cpu_nodes_per_task", "2345"),
        ("max_rounds", "13"),
        ("max_cpu_nodes", "76543"),
        ("cpu_depth", "5"),
        ("resolver", "pals-cpu-raw-restricted/0.1"),
        ("post_repair_recheck", "disabled"),
    ]
    .into_iter()
    .map(|(key, value)| (key.into(), value.into()))
    .collect();
    let own = [
        ("profile", "cpu-independent-conservative-v1"),
        ("tt_entries", "1024"),
        ("max_depth", "11"),
        ("quiescence_ply", "9"),
        ("ordering", "cpu-ordering-rules-legal-target-exchange-v1"),
        ("resume", "cpu-resume-completed-iteration-v1"),
        ("value_semantics", "bootstrap-material-pst-v1"),
        ("paused_side_scope", "false"),
    ]
    .into_iter()
    .map(|(key, value)| (key.into(), value.into()))
    .collect();
    (search, own)
}

fn parse(
    search: &BTreeMap<String, String>,
    own: &BTreeMap<String, String>,
    checker: PalsCpuSelectionV5,
) -> Result<PalsSearchSettingsV5, PalsSearchSettingsErrorV5> {
    parse_option_maps(search, own, checker, Instant::now())
}

#[test]
fn declarations_preserve_every_budget_policy_and_the_original_deadline() {
    let (search, own) = declarations();
    let deadline = Instant::now() + Duration::from_secs(17);
    let settings = parse_option_maps(&search, &own, PalsCpuSelectionV5::Own, deadline).unwrap();
    assert_eq!(settings.config.beam_width, 3);
    assert_eq!(settings.config.line_plies, 7);
    assert_eq!(settings.config.max_nodes, 8192);
    assert_eq!(settings.config.max_records, 64);
    assert_eq!(settings.config.max_role_calls, 123);
    assert_eq!(settings.config.cpu_nodes_per_task, 2345);
    assert_eq!(settings.limits.deadline, deadline);
    assert_eq!(settings.limits.max_rounds, 13);
    assert_eq!(settings.limits.max_cpu_nodes, 76543);
    assert_eq!(settings.limits.cpu_depth, 5);
    assert_eq!(settings.own_cpu.profile, CpuProfile::Independent);
    assert_eq!(settings.own_cpu.tt_entries, 1024);
    assert_eq!(settings.own_cpu.max_depth, 11);
    assert_eq!(settings.own_cpu.quiescence_ply, 9);
    assert_eq!(settings.ordering, CpuOrderingPolicy::LegalSeeV1);
    assert_eq!(settings.resume, CpuResumePolicy::CompletedIteration);
    assert_eq!(settings.resolver, ResolverPolicy::OwnRawRestricted);
    assert_eq!(
        settings.post_repair_recheck,
        PostRepairRecheckPolicy::Disabled
    );
    assert_eq!(settings.value_semantics, BOOTSTRAP_SCORE_VERSION);
    assert!(!settings.paused_side_scope);
}

#[test]
fn missing_or_unrecognized_fields_never_select_implicit_defaults() {
    let (search, own) = declarations();
    for key in search.keys() {
        let mut incomplete = search.clone();
        incomplete.remove(key);
        assert_eq!(
            parse(&incomplete, &own, PalsCpuSelectionV5::Own).unwrap_err(),
            PalsSearchSettingsErrorV5::Fields {
                component: "search"
            },
            "missing search.{key}"
        );
    }
    for key in own.keys() {
        let mut incomplete = own.clone();
        incomplete.remove(key);
        assert_eq!(
            parse(&search, &incomplete, PalsCpuSelectionV5::Own).unwrap_err(),
            PalsSearchSettingsErrorV5::Fields { component: "own" },
            "missing own.{key}"
        );
    }
    let mut extra_search = search.clone();
    extra_search.insert("simulation_limit".into(), "128".into());
    assert!(matches!(
        parse(&extra_search, &own, PalsCpuSelectionV5::Own),
        Err(PalsSearchSettingsErrorV5::Fields {
            component: "search"
        })
    ));
    let mut extra_own = own.clone();
    extra_own.insert("threads".into(), "1".into());
    assert!(matches!(
        parse(&search, &extra_own, PalsCpuSelectionV5::Own),
        Err(PalsSearchSettingsErrorV5::Fields { component: "own" })
    ));
}

#[test]
fn numeric_options_require_decimal_canonical_form_and_detect_overflow() {
    let (search, own) = declarations();
    let malformed = ["", "01", "+1", "-1", " 1", "1 ", "1.0", "1e0", "true", "１"];
    for key in [
        "beam_width",
        "line_plies",
        "max_nodes",
        "max_records",
        "max_role_calls",
        "cpu_nodes_per_task",
        "max_rounds",
        "max_cpu_nodes",
        "cpu_depth",
    ] {
        for text in malformed {
            let mut changed = search.clone();
            changed.insert(key.into(), text.into());
            assert!(
                parse(&changed, &own, PalsCpuSelectionV5::Own).is_err(),
                "search.{key}={text:?}"
            );
        }
    }
    for key in ["tt_entries", "max_depth", "quiescence_ply"] {
        for text in malformed {
            let mut changed = own.clone();
            changed.insert(key.into(), text.into());
            assert!(
                parse(&search, &changed, PalsCpuSelectionV5::Own).is_err(),
                "own.{key}={text:?}"
            );
        }
    }
    let mut overflow = search.clone();
    overflow.insert("max_cpu_nodes".into(), "18446744073709551616".into());
    assert!(matches!(
        parse(&overflow, &own, PalsCpuSelectionV5::Own),
        Err(PalsSearchSettingsErrorV5::Range {
            component: "search",
            key: "max_cpu_nodes"
        })
    ));
}

#[test]
fn pals_capacity_and_cpu_allocation_bounds_are_checked_without_allocation() {
    let (search, own) = declarations();
    for (key, text) in [
        ("beam_width", "0"),
        ("beam_width", "17"),
        ("line_plies", "0"),
        ("line_plies", "17"),
        ("max_nodes", "256"),
        ("max_nodes", "65537"),
        ("max_records", "3"),
        ("max_records", "129"),
        ("max_role_calls", "0"),
        ("max_role_calls", "1000001"),
        ("cpu_nodes_per_task", "0"),
        ("cpu_nodes_per_task", "10000001"),
    ] {
        let mut changed = search.clone();
        changed.insert(key.into(), text.into());
        assert!(
            matches!(
                parse(&changed, &own, PalsCpuSelectionV5::Own),
                Err(PalsSearchSettingsErrorV5::InvalidPalsConfig)
            ),
            "search.{key}={text}"
        );
    }
    for (key, text) in [
        ("tt_entries", "1048577"),
        ("max_depth", "0"),
        ("max_depth", "65"),
        ("quiescence_ply", "33"),
    ] {
        let mut changed = own.clone();
        changed.insert(key.into(), text.into());
        assert!(
            matches!(
                parse(&search, &changed, PalsCpuSelectionV5::Own),
                Err(PalsSearchSettingsErrorV5::InvalidCpuConfig)
            ),
            "own.{key}={text}"
        );
    }
    let mut no_tt = own.clone();
    no_tt.insert("tt_entries".into(), "0".into());
    no_tt.insert("quiescence_ply".into(), "0".into());
    let parsed = parse(&search, &no_tt, PalsCpuSelectionV5::Own).unwrap();
    assert_eq!(parsed.own_cpu.tt_entries, 0);
    assert_eq!(parsed.own_cpu.quiescence_ply, 0);
    no_tt.insert("tt_entries".into(), "1048576".into());
    no_tt.insert("max_depth".into(), "64".into());
    no_tt.insert("quiescence_ply".into(), "32".into());
    assert!(parse(&search, &no_tt, PalsCpuSelectionV5::Own).is_ok());
}

#[test]
fn limits_are_positive_and_own_depth_does_not_exceed_the_declared_checker() {
    let (search, own) = declarations();
    for (key, text) in [
        ("max_rounds", "0"),
        ("max_cpu_nodes", "0"),
        ("cpu_depth", "0"),
        ("cpu_depth", "65"),
        ("cpu_depth", "65536"),
    ] {
        let mut changed = search.clone();
        changed.insert(key.into(), text.into());
        assert!(
            parse(&changed, &own, PalsCpuSelectionV5::Own).is_err(),
            "search.{key}={text}"
        );
    }
    let mut deeper = search.clone();
    deeper.insert("cpu_depth".into(), "12".into());
    assert!(matches!(
        parse(&deeper, &own, PalsCpuSelectionV5::Own),
        Err(PalsSearchSettingsErrorV5::Range {
            component: "search",
            key: "cpu_depth"
        })
    ));
    deeper.insert("resolver".into(), "pals-model-wdl-restricted/0.1".into());
    assert!(parse(&deeper, &own, PalsCpuSelectionV5::Stockfish19Full).is_ok());
}

#[test]
fn external_checker_never_inherits_own_raw_or_own_only_recheck_policy() {
    let (mut search, own) = declarations();
    assert!(matches!(
        parse(&search, &own, PalsCpuSelectionV5::Stockfish19Full),
        Err(PalsSearchSettingsErrorV5::Unsupported {
            component: "search",
            key: "resolver"
        })
    ));
    assert!(matches!(
        parse(&search, &own, PalsCpuSelectionV5::Disabled),
        Err(PalsSearchSettingsErrorV5::Unsupported {
            component: "search",
            key: "checker"
        })
    ));
    search.insert("resolver".into(), "pals-model-wdl-restricted/0.1".into());
    for (text, expected) in [
        ("disabled", PostRepairRecheckPolicy::Disabled),
        (
            "frozen_model_wdl_v2",
            PostRepairRecheckPolicy::FrozenModelWdlV2,
        ),
        (
            "iterative_frozen_model_wdl_v2",
            PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
        ),
    ] {
        search.insert("post_repair_recheck".into(), text.into());
        let accepted = parse(&search, &own, PalsCpuSelectionV5::Stockfish19Full).unwrap();
        assert_eq!(accepted.resolver, ResolverPolicy::ModelWdlRestricted);
        assert_eq!(accepted.post_repair_recheck, expected);
    }
    for text in [
        "same_repaired_line_once_v1",
        "actual_opponent_continuation_v1",
    ] {
        search.insert("post_repair_recheck".into(), text.into());
        assert!(parse(&search, &own, PalsCpuSelectionV5::Own).is_ok());
        assert!(matches!(
            parse(&search, &own, PalsCpuSelectionV5::Stockfish19Full),
            Err(PalsSearchSettingsErrorV5::Unsupported {
                component: "search",
                key: "post_repair_recheck"
            })
        ));
    }
}

#[test]
fn unsupported_value_namespaces_and_paused_scope_require_explicit_capabilities() {
    let (search, own) = declarations();
    for (key, text) in [
        ("profile", "independent"),
        ("ordering", "see"),
        ("resume", "paused"),
        ("value_semantics", "frozen-cpu-value-v1"),
    ] {
        let mut changed = own.clone();
        changed.insert(key.into(), text.into());
        assert!(matches!(
            parse(&search, &changed, PalsCpuSelectionV5::Own),
            Err(PalsSearchSettingsErrorV5::Unsupported { .. })
        ));
    }
    for key in [
        "frozen_enabled",
        "value_checkpoint",
        "value_weights_sha256",
        "cpu_value_file",
    ] {
        let mut changed = own.clone();
        changed.insert(key.into(), "false".into());
        assert!(matches!(
            parse(&search, &changed, PalsCpuSelectionV5::Own),
            Err(PalsSearchSettingsErrorV5::Unsupported {
                component: "own",
                key: "value_semantics"
            })
        ));
    }
    for text in ["0", "1", "True", "FALSE", " false"] {
        let mut changed = own.clone();
        changed.insert("paused_side_scope".into(), text.into());
        assert!(matches!(
            parse(&search, &changed, PalsCpuSelectionV5::Own),
            Err(PalsSearchSettingsErrorV5::NonCanonical {
                component: "own",
                key: "paused_side_scope"
            })
        ));
    }
    let mut paused = own.clone();
    paused.insert("paused_side_scope".into(), "true".into());
    assert!(matches!(
        parse(&search, &paused, PalsCpuSelectionV5::Own),
        Err(PalsSearchSettingsErrorV5::Unsupported {
            component: "own",
            key: "paused_side_scope"
        })
    ));
    paused.insert("resume".into(), "cpu-resume-paused-stack-v1".into());
    let accepted = parse(&search, &paused, PalsCpuSelectionV5::Own).unwrap();
    assert!(accepted.paused_side_scope);
    assert_eq!(accepted.resume, CpuResumePolicy::PausedStack);
}

#[test]
fn both_component_pins_and_the_declared_inventory_must_match_compiled_sources() {
    let pins = compiled_source_components_v5().unwrap();
    let search = PalsComponentV3 {
        semantic_id: "source-check-only-search".into(),
        implementation_sha256: pins["coordinator"].clone(),
        options: BTreeMap::new(),
    };
    let own = PalsComponentV3 {
        semantic_id: "source-check-only-own".into(),
        implementation_sha256: pins["cpu_search"].clone(),
        options: BTreeMap::new(),
    };
    verify_component_sources(&search, &own, &pins).unwrap();
    let mut wrong_search = search.clone();
    wrong_search.implementation_sha256 = "0".repeat(64);
    assert!(verify_component_sources(&wrong_search, &own, &pins).is_err());
    let mut wrong_own = own.clone();
    wrong_own.implementation_sha256 = "0".repeat(64);
    assert!(verify_component_sources(&search, &wrong_own, &pins).is_err());
    for key in ["coordinator", "cpu_search"] {
        let mut missing = pins.clone();
        missing.remove(key);
        assert!(verify_component_sources(&search, &own, &missing).is_err());
        let mut wrong = pins.clone();
        wrong.insert(key.into(), "f".repeat(64));
        assert!(verify_component_sources(&search, &own, &wrong).is_err());
    }
}

#[test]
fn v6_reuses_search_semantics_and_keeps_source_observation_independent() {
    let (search_options, own_options) = declarations();
    let compiled = compiled_source_components_v5().unwrap();
    let search = PalsComponentV3 {
        semantic_id: "v6-search".into(),
        implementation_sha256: compiled["coordinator"].clone(),
        options: search_options,
    };
    let own = PalsComponentV3 {
        semantic_id: "v6-own".into(),
        implementation_sha256: compiled["cpu_search"].clone(),
        options: own_options,
    };
    let deadline = Instant::now() + Duration::from_secs(17);
    let parse_v6 =
        |policy, declared: &BTreeMap<String, String>, actual: Option<&BTreeMap<String, String>>| {
            parse_v6_components(
                &search,
                &own,
                PalsCpuSelectionV5::Own,
                policy,
                declared,
                actual,
                deadline,
            )
        };
    let strict = parse_v6(PalsSystemSourcePolicyV6::Strict, &compiled, Some(&compiled)).unwrap();
    assert_eq!(strict.settings.limits.deadline, deadline);
    assert_eq!(strict.settings.config.beam_width, 3);
    assert_eq!(strict.settings.own_cpu.max_depth, 11);
    assert_eq!(
        strict.coordinator_source.observation,
        PalsSystemSourceObservationV6::Verified
    );
    assert_eq!(
        strict.own_cpu_source.observation,
        PalsSystemSourceObservationV6::Verified
    );

    let mut missing = compiled.clone();
    missing.remove("coordinator");
    assert!(matches!(
        parse_v6(PalsSystemSourcePolicyV6::Strict, &missing, Some(&compiled)),
        Err(PalsSearchSettingsErrorV6::SourceMismatch {
            component: "coordinator"
        })
    ));
    let observed = parse_v6(PalsSystemSourcePolicyV6::Observe, &missing, Some(&compiled)).unwrap();
    assert_eq!(
        observed.coordinator_source.observation,
        PalsSystemSourceObservationV6::Missing
    );
    assert_eq!(observed.settings.limits.deadline, deadline);

    let unknown = parse_v6(PalsSystemSourcePolicyV6::Observe, &compiled, None).unwrap();
    assert_eq!(
        unknown.coordinator_source.observation,
        PalsSystemSourceObservationV6::Missing
    );
    assert!(
        unknown
            .coordinator_source
            .compiled_component_sha256
            .is_none()
    );
    let mut changed = compiled.clone();
    let mut wrong_cpu = compiled["cpu_search"].clone();
    let replacement = if wrong_cpu.starts_with('0') { "1" } else { "0" };
    wrong_cpu.replace_range(..1, replacement);
    changed.insert("cpu_search".into(), wrong_cpu);
    let mismatch = parse_v6(PalsSystemSourcePolicyV6::Observe, &changed, Some(&compiled)).unwrap();
    assert_eq!(
        mismatch.own_cpu_source.observation,
        PalsSystemSourceObservationV6::Mismatch
    );
    assert!(matches!(
        parse_v6(PalsSystemSourcePolicyV6::Strict, &changed, Some(&compiled)),
        Err(PalsSearchSettingsErrorV6::SourceMismatch {
            component: "cpu_search"
        })
    ));
}
