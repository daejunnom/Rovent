//! Strict bridge from registered V5 startup options to existing search types.
//! It does not alter persisted V5 parsing or legacy CLI defaults. Actual startup
//! calls this before constructing a checker, allocating a TT or creating workers.
use super::PalsSystemSourceObservationV6;
use rz_experiments::{
    PalsAdaptiveStartupConfigV5, PalsComponentV3, PalsCpuSelectionV5, PalsEngineModeV5,
    PalsSystemSourcePolicyV6, PalsSystemStartupConfigV6, compiled_source_components_v5,
};
use rz_search::{
    cpu::{BOOTSTRAP_SCORE_VERSION, CpuConfig, CpuOrderingPolicy, CpuProfile, CpuResumePolicy},
    pals::engine::{PalsConfig, PalsLimits, PostRepairRecheckPolicy, ResolverPolicy},
};
use std::{collections::BTreeMap, fmt, time::Instant};

pub const PALS_SEARCH_OPTION_KEYS_V5: [&str; 11] = [
    "beam_width",
    "line_plies",
    "max_nodes",
    "max_records",
    "max_role_calls",
    "cpu_nodes_per_task",
    "max_rounds",
    "max_cpu_nodes",
    "cpu_depth",
    "resolver",
    "post_repair_recheck",
];
pub const PALS_OWN_CPU_OPTION_KEYS_V5: [&str; 8] = [
    "profile",
    "tt_entries",
    "max_depth",
    "quiescence_ply",
    "ordering",
    "resume",
    "value_semantics",
    "paused_side_scope",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PalsSearchSettingsErrorV5 {
    Fields {
        component: &'static str,
    },
    NonCanonical {
        component: &'static str,
        key: &'static str,
    },
    Range {
        component: &'static str,
        key: &'static str,
    },
    Unsupported {
        component: &'static str,
        key: &'static str,
    },
    InvalidPalsConfig,
    InvalidCpuConfig,
    SourceMismatch {
        component: &'static str,
    },
    SourceRegistry,
}
impl fmt::Display for PalsSearchSettingsErrorV5 {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fields { component } => write!(
                out,
                "V5 {component} options require the exact registered key set"
            ),
            Self::NonCanonical { component, key } => write!(
                out,
                "V5 {component}.{key} is not a canonical decimal or boolean"
            ),
            Self::Range { component, key } => write!(
                out,
                "V5 {component}.{key} exceeds the existing bounded search contract"
            ),
            Self::Unsupported { component, key } => write!(
                out,
                "V5 {component}.{key} selects an unsupported capability or semantic profile"
            ),
            Self::InvalidPalsConfig => {
                out.write_str("V5 PALS settings violate PalsConfig::validate")
            }
            Self::InvalidCpuConfig => {
                out.write_str("V5 own CPU settings violate the existing bounded CPU configuration")
            }
            Self::SourceMismatch { component } => write!(
                out,
                "V5 {component} implementation differs from this executable's common source component"
            ),
            Self::SourceRegistry => {
                out.write_str("V5 compiled source registry could not be verified")
            }
        }
    }
}
impl std::error::Error for PalsSearchSettingsErrorV5 {}

/// These fields are declarations validated against existing types, not evidence
/// that an engine, TT, worker or paused scope has actually been created.
#[derive(Clone, Debug)]
pub struct PalsSearchSettingsV5 {
    pub config: PalsConfig,
    pub limits: PalsLimits,
    pub resolver: ResolverPolicy,
    pub post_repair_recheck: PostRepairRecheckPolicy,
    pub own_cpu: CpuConfig,
    pub ordering: CpuOrderingPolicy,
    pub resume: CpuResumePolicy,
    pub value_semantics: &'static str,
    pub paused_side_scope: bool,
}

/// Comparison of one declared search component with the current binary's
/// registered source. Values are observations, not search execution evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PalsSearchComponentSourceV6 {
    pub component: &'static str,
    pub declared_implementation_sha256: String,
    pub declared_component_sha256: Option<String>,
    pub compiled_component_sha256: Option<String>,
    pub observation: PalsSystemSourceObservationV6,
}

/// Exact V5 search semantics parsed for a V6 caller. Source comparison must
/// remain visible to the actual owner; this value alone grants no GPU, worker,
/// physical-completion, or model numerical acceptance authority.
#[derive(Clone, Debug)]
#[must_use]
pub struct PalsSearchSettingsV6 {
    pub settings: PalsSearchSettingsV5,
    pub coordinator_source: PalsSearchComponentSourceV6,
    pub own_cpu_source: PalsSearchComponentSourceV6,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PalsSearchSettingsErrorV6 {
    Options(PalsSearchSettingsErrorV5),
    SourceRegistry,
    SourceMismatch { component: &'static str },
}
impl fmt::Display for PalsSearchSettingsErrorV6 {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Options(error) => write!(out, "V6 shared search options: {error}"),
            Self::SourceRegistry => out.write_str("V6 compiled search source registry unavailable"),
            Self::SourceMismatch { component } => write!(
                out,
                "V6 {component} source not verified under strict policy"
            ),
        }
    }
}
impl std::error::Error for PalsSearchSettingsErrorV6 {}

/// Parse the same registered search/CPU option semantics and preserve the
/// caller's original deadline. Strict requires both actual component hashes;
/// Observe retains missing or mismatched provenance without granting it trust.
pub fn parse_pals_search_settings_v6(
    config: &PalsSystemStartupConfigV6,
    original_hard_deadline: Instant,
) -> Result<PalsSearchSettingsV6, PalsSearchSettingsErrorV6> {
    let checker = match config.endpoint.mode {
        PalsEngineModeV5::Collection => config.endpoint.cpu.cpu_t,
        PalsEngineModeV5::Deployment => config.endpoint.cpu.cpu_r,
    };
    let compiled = compiled_source_components_v5();
    if config.source_policy == PalsSystemSourcePolicyV6::Strict && compiled.is_err() {
        return Err(PalsSearchSettingsErrorV6::SourceRegistry);
    }
    parse_v6_components(
        &config.endpoint.search,
        &config.endpoint.cpu.own,
        checker,
        config.source_policy,
        &config.source_components,
        compiled.as_ref().ok(),
        original_hard_deadline,
    )
}

fn observe_v6_component(
    name: &'static str,
    component: &PalsComponentV3,
    declared: &BTreeMap<String, String>,
    compiled: Option<&BTreeMap<String, String>>,
) -> PalsSearchComponentSourceV6 {
    let declared_component_sha256 = declared.get(name).cloned();
    let compiled_component_sha256 = compiled.and_then(|map| map.get(name)).cloned();
    let observation = match (&compiled_component_sha256, &declared_component_sha256) {
        (Some(actual), _) if &component.implementation_sha256 != actual => {
            PalsSystemSourceObservationV6::Mismatch
        }
        (Some(actual), Some(pin)) if actual == pin => PalsSystemSourceObservationV6::Verified,
        (Some(_), Some(_)) => PalsSystemSourceObservationV6::Mismatch,
        _ => PalsSystemSourceObservationV6::Missing,
    };
    PalsSearchComponentSourceV6 {
        component: name,
        declared_implementation_sha256: component.implementation_sha256.clone(),
        declared_component_sha256,
        compiled_component_sha256,
        observation,
    }
}

fn parse_v6_components(
    search: &PalsComponentV3,
    own: &PalsComponentV3,
    checker: PalsCpuSelectionV5,
    source_policy: PalsSystemSourcePolicyV6,
    declared: &BTreeMap<String, String>,
    compiled: Option<&BTreeMap<String, String>>,
    deadline: Instant,
) -> Result<PalsSearchSettingsV6, PalsSearchSettingsErrorV6> {
    let coordinator_source = observe_v6_component("coordinator", search, declared, compiled);
    let own_cpu_source = observe_v6_component("cpu_search", own, declared, compiled);
    if source_policy == PalsSystemSourcePolicyV6::Strict {
        for source in [&coordinator_source, &own_cpu_source] {
            if source.observation != PalsSystemSourceObservationV6::Verified {
                return Err(PalsSearchSettingsErrorV6::SourceMismatch {
                    component: source.component,
                });
            }
        }
    }
    let settings = parse_option_maps(&search.options, &own.options, checker, deadline)
        .map_err(PalsSearchSettingsErrorV6::Options)?;
    Ok(PalsSearchSettingsV6 {
        settings,
        coordinator_source,
        own_cpu_source,
    })
}

/// Parse only at the actual runtime entry. The deadline is the caller's original
/// hard deadline; this function never starts a fresh duration or adds credit.
pub fn parse_pals_search_settings_v5(
    config: &PalsAdaptiveStartupConfigV5,
    original_hard_deadline: Instant,
) -> Result<PalsSearchSettingsV5, PalsSearchSettingsErrorV5> {
    verify_pals_search_implementation_v5(config)?;
    let checker = match config.endpoint.mode {
        PalsEngineModeV5::Collection => config.endpoint.cpu.cpu_t,
        PalsEngineModeV5::Deployment => config.endpoint.cpu.cpu_r,
    };
    parse_option_maps(
        &config.endpoint.search.options,
        &config.endpoint.cpu.own.options,
        checker,
        original_hard_deadline,
    )
}

/// The search component denotes the adaptive coordinator implementation and
/// the own component denotes the CPU search/value implementation. Keep these
/// common V5 pins separate from older custom-framed implementation digests.
pub fn verify_pals_search_implementation_v5(
    config: &PalsAdaptiveStartupConfigV5,
) -> Result<(), PalsSearchSettingsErrorV5> {
    verify_component_sources(
        &config.endpoint.search,
        &config.endpoint.cpu.own,
        &config.source_components,
    )
}
fn verify_component_sources(
    search: &PalsComponentV3,
    own: &PalsComponentV3,
    declared: &BTreeMap<String, String>,
) -> Result<(), PalsSearchSettingsErrorV5> {
    let compiled =
        compiled_source_components_v5().map_err(|_| PalsSearchSettingsErrorV5::SourceRegistry)?;
    for (name, component) in [("coordinator", search), ("cpu_search", own)] {
        let actual = compiled
            .get(name)
            .ok_or(PalsSearchSettingsErrorV5::SourceRegistry)?;
        if declared.get(name) != Some(actual) || &component.implementation_sha256 != actual {
            return Err(PalsSearchSettingsErrorV5::SourceMismatch { component: name });
        }
    }
    Ok(())
}

fn exact_keys<const N: usize>(
    options: &BTreeMap<String, String>,
    keys: &[&str; N],
    component: &'static str,
) -> Result<(), PalsSearchSettingsErrorV5> {
    if options.len() != N || keys.iter().any(|key| !options.contains_key(*key)) {
        return Err(PalsSearchSettingsErrorV5::Fields { component });
    }
    Ok(())
}
fn number(
    options: &BTreeMap<String, String>,
    component: &'static str,
    key: &'static str,
) -> Result<u64, PalsSearchSettingsErrorV5> {
    let text = options
        .get(key)
        .ok_or(PalsSearchSettingsErrorV5::Fields { component })?;
    if text.is_empty()
        || text.len() > 20
        || !text.bytes().all(|byte| byte.is_ascii_digit())
        || (text.len() > 1 && text.starts_with('0'))
    {
        return Err(PalsSearchSettingsErrorV5::NonCanonical { component, key });
    }
    text.parse()
        .map_err(|_| PalsSearchSettingsErrorV5::Range { component, key })
}
fn usize_value(
    options: &BTreeMap<String, String>,
    component: &'static str,
    key: &'static str,
) -> Result<usize, PalsSearchSettingsErrorV5> {
    usize::try_from(number(options, component, key)?)
        .map_err(|_| PalsSearchSettingsErrorV5::Range { component, key })
}
fn depth(
    options: &BTreeMap<String, String>,
    component: &'static str,
    key: &'static str,
) -> Result<u16, PalsSearchSettingsErrorV5> {
    u16::try_from(number(options, component, key)?)
        .map_err(|_| PalsSearchSettingsErrorV5::Range { component, key })
}
fn bool_value(
    options: &BTreeMap<String, String>,
    component: &'static str,
    key: &'static str,
) -> Result<bool, PalsSearchSettingsErrorV5> {
    match options.get(key).map(String::as_str) {
        Some("true") => Ok(true),
        Some("false") => Ok(false),
        _ => Err(PalsSearchSettingsErrorV5::NonCanonical { component, key }),
    }
}
fn unsupported(component: &'static str, key: &'static str) -> PalsSearchSettingsErrorV5 {
    PalsSearchSettingsErrorV5::Unsupported { component, key }
}

fn parse_option_maps(
    search: &BTreeMap<String, String>,
    own: &BTreeMap<String, String>,
    checker: PalsCpuSelectionV5,
    deadline: Instant,
) -> Result<PalsSearchSettingsV5, PalsSearchSettingsErrorV5> {
    // Frozen/learned CPU values are a different artifact/semantic profile. An
    // extra option is not interpreted as Bootstrap, even when its value is false.
    if own.keys().any(|key| {
        key.starts_with("frozen_")
            || matches!(
                key.as_str(),
                "value_checkpoint" | "value_weights_sha256" | "cpu_value_file"
            )
    }) {
        return Err(unsupported("own", "value_semantics"));
    }
    exact_keys(search, &PALS_SEARCH_OPTION_KEYS_V5, "search")?;
    exact_keys(own, &PALS_OWN_CPU_OPTION_KEYS_V5, "own")?;
    let config = PalsConfig {
        beam_width: usize_value(search, "search", "beam_width")?,
        line_plies: usize_value(search, "search", "line_plies")?,
        max_nodes: usize_value(search, "search", "max_nodes")?,
        max_records: usize_value(search, "search", "max_records")?,
        max_role_calls: number(search, "search", "max_role_calls")?,
        cpu_nodes_per_task: number(search, "search", "cpu_nodes_per_task")?,
    };
    config
        .validate()
        .map_err(|_| PalsSearchSettingsErrorV5::InvalidPalsConfig)?;
    let profile = match own["profile"].as_str() {
        value if value == CpuProfile::PlanAssisted.identity() => CpuProfile::PlanAssisted,
        value if value == CpuProfile::Independent.identity() => CpuProfile::Independent,
        value if value == CpuProfile::RelaxedSelectivity.identity() => {
            CpuProfile::RelaxedSelectivity
        }
        _ => return Err(unsupported("own", "profile")),
    };
    let own_cpu = CpuConfig {
        profile,
        tt_entries: usize_value(own, "own", "tt_entries")?,
        max_depth: depth(own, "own", "max_depth")?,
        quiescence_ply: depth(own, "own", "quiescence_ply")?,
    };
    // The bounds match CpuEngine::with_evaluator_ordering_and_resume_policy;
    // calling the owner constructor here would allocate the actual private TT.
    own_cpu
        .tt_allocation_bytes()
        .map_err(|_| PalsSearchSettingsErrorV5::InvalidCpuConfig)?;
    if !(1..=64).contains(&own_cpu.max_depth) || own_cpu.quiescence_ply > 32 {
        return Err(PalsSearchSettingsErrorV5::InvalidCpuConfig);
    }
    let ordering = match own["ordering"].as_str() {
        value if value == CpuOrderingPolicy::LegacyMvvLvaV1.identity() => {
            CpuOrderingPolicy::LegacyMvvLvaV1
        }
        value if value == CpuOrderingPolicy::LegalSeeV1.identity() => CpuOrderingPolicy::LegalSeeV1,
        _ => return Err(unsupported("own", "ordering")),
    };
    let resume = match own["resume"].as_str() {
        value if value == CpuResumePolicy::CompletedIteration.identity() => {
            CpuResumePolicy::CompletedIteration
        }
        value if value == CpuResumePolicy::PausedStack.identity() => CpuResumePolicy::PausedStack,
        _ => return Err(unsupported("own", "resume")),
    };
    if own["value_semantics"] != BOOTSTRAP_SCORE_VERSION {
        return Err(unsupported("own", "value_semantics"));
    }
    let paused_side_scope = bool_value(own, "own", "paused_side_scope")?;
    if paused_side_scope && resume != CpuResumePolicy::PausedStack {
        return Err(unsupported("own", "paused_side_scope"));
    }
    let limits = PalsLimits {
        deadline,
        max_rounds: number(search, "search", "max_rounds")?,
        max_cpu_nodes: number(search, "search", "max_cpu_nodes")?,
        cpu_depth: depth(search, "search", "cpu_depth")?,
    };
    for (key, valid) in [
        ("max_rounds", limits.max_rounds > 0),
        ("max_cpu_nodes", limits.max_cpu_nodes > 0),
        (
            "cpu_depth",
            (1..=64).contains(&limits.cpu_depth)
                && (checker != PalsCpuSelectionV5::Own || limits.cpu_depth <= own_cpu.max_depth),
        ),
    ] {
        if !valid {
            return Err(PalsSearchSettingsErrorV5::Range {
                component: "search",
                key,
            });
        }
    }
    let resolver = match search["resolver"].as_str() {
        value if value == ResolverPolicy::OwnRawRestricted.version() => {
            ResolverPolicy::OwnRawRestricted
        }
        value if value == ResolverPolicy::ModelWdlRestricted.version() => {
            ResolverPolicy::ModelWdlRestricted
        }
        _ => return Err(unsupported("search", "resolver")),
    };
    let post_repair_recheck = [
        PostRepairRecheckPolicy::Disabled,
        PostRepairRecheckPolicy::SameRepairedLineOnceV1,
        PostRepairRecheckPolicy::ActualOpponentContinuationV1,
        PostRepairRecheckPolicy::FrozenModelWdlV2,
        PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
    ]
    .into_iter()
    .find(|policy| {
        policy.registration_policy().unwrap_or("disabled") == search["post_repair_recheck"]
    })
    .ok_or_else(|| unsupported("search", "post_repair_recheck"))?;
    if checker == PalsCpuSelectionV5::Disabled {
        return Err(unsupported("search", "checker"));
    }
    if checker == PalsCpuSelectionV5::Stockfish19Full {
        if resolver == ResolverPolicy::OwnRawRestricted {
            return Err(unsupported("search", "resolver"));
        }
        if post_repair_recheck != PostRepairRecheckPolicy::Disabled
            && !post_repair_recheck.uses_frozen_model_wdl()
        {
            return Err(unsupported("search", "post_repair_recheck"));
        }
    }
    Ok(PalsSearchSettingsV5 {
        config,
        limits,
        resolver,
        post_repair_recheck,
        own_cpu,
        ordering,
        resume,
        value_semantics: BOOTSTRAP_SCORE_VERSION,
        paused_side_scope,
    })
}

#[cfg(test)]
mod tests;
