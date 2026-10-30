//! 엔진별 선언과 실제 외부 checker의 지원 범위를 분리한다.
//!
//! 공개 source pin과 option profile은 실행·모델 로딩·권리·옵션 적용의
//! 관측 증거가 아니다. ArenaOpponent 선언을 CPU_T/CPU_R로 재사용하지 않으며,
//! 외부 CP/mate는 기존 raw report 의미를 그대로 유지한다.

use super::{
    CheckerError, EXTERNAL_UCI_SCORE_SEMANTICS, ExternalCheckerIdentity, ExternalCheckerReport,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const RECKLESS090_SOURCE_COMMIT: &str = "0e92358f5acd66e5ac77b1bf558202e47c515435";
const STOCKFISH_SOURCE: &str = "https://github.com/official-stockfish/Stockfish";
const RECKLESS_SOURCE: &str = "https://github.com/codedeliveryservice/Reckless";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalEngineProfile {
    Stockfish19EmbeddedNnue,
    Reckless090EmbeddedNnue,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalEnginePurpose {
    Teacher,
    ResearchChecker,
    ArenaOpponent,
}

/// Source-verified support, not evidence from a running process. No profile
/// here supports separately loaded assets or exact stack/TT ownership transfer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExternalEngineCapabilities {
    pub searchmoves: bool,
    pub combined_depth_nodes_movetime: bool,
    pub ponder: bool,
    pub embedded_nnue: bool,
    pub separate_assets: bool,
    pub exact_stack_resume: bool,
}

/// Requested options. The later UCI handshake independently checks advertised
/// names, types and ranges; `readyok` alone does not attest option application.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExternalEngineResources {
    pub threads: u16,
    pub hash_mib: u32,
    pub multipv: u16,
    pub move_overhead_ms: u16,
}

/// Explicit UCI constraints. A single-limit parser never receives the combined
/// form. Hard wall/stop/drain is enforced separately by the existing owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExternalGoConstraint {
    Combined { depth: u16, reported_nodes: u64 },
    Nodes { reported_nodes: u64 },
    Depth { depth: u16 },
    MoveTime { milliseconds: u64 },
}

impl ExternalGoConstraint {
    pub fn requested_depth(self) -> Option<u16> {
        match self {
            Self::Combined { depth, .. } | Self::Depth { depth } => Some(depth),
            Self::Nodes { .. } | Self::MoveTime { .. } => None,
        }
    }

    pub fn requested_nodes(self) -> Option<u64> {
        match self {
            Self::Combined { reported_nodes, .. } | Self::Nodes { reported_nodes } => {
                Some(reported_nodes)
            }
            Self::Depth { .. } | Self::MoveTime { .. } => None,
        }
    }

    pub(super) fn command(self, remaining_ms: u128) -> (String, Self) {
        match self {
            Self::Combined {
                depth,
                reported_nodes,
            } => (
                format!("go depth {depth} nodes {reported_nodes} movetime {remaining_ms}"),
                self,
            ),
            Self::Nodes { reported_nodes } => (format!("go nodes {reported_nodes}"), self),
            Self::Depth { depth } => (format!("go depth {depth}"), self),
            Self::MoveTime { milliseconds } => {
                let milliseconds = milliseconds.min(remaining_ms.min(u128::from(u64::MAX)) as u64);
                (
                    format!("go movetime {milliseconds}"),
                    Self::MoveTime { milliseconds },
                )
            }
        }
    }
}

/// A new query namespace; this is not a legacy PALS completed-depth report.
/// `raw.requested_depth == 0` means no depth was requested and is deliberately
/// rejected by legacy consumers. Reported depth remains a raw observation.
#[derive(Clone, Debug)]
pub struct ProfiledExternalReport {
    pub profile: ExternalEngineProfile,
    pub purpose: ExternalEnginePurpose,
    pub requested: ExternalGoConstraint,
    pub issued: ExternalGoConstraint,
    pub issued_go: String,
    pub raw: ExternalCheckerReport,
    pub reported_node_overshoot: bool,
    pub strict_native_nodes: Option<u64>,
    pub option_application_observed: Option<bool>,
    pub readiness_barrier_completed: bool,
}

impl Default for ExternalEngineResources {
    fn default() -> Self {
        Self {
            threads: 1,
            hash_mib: 256,
            multipv: 1,
            move_overhead_ms: 10,
        }
    }
}

impl ExternalEngineProfile {
    pub const fn expected_uci_name(self) -> &'static str {
        match self {
            Self::Stockfish19EmbeddedNnue => "Stockfish 19",
            Self::Reckless090EmbeddedNnue => "Reckless 0.9.0",
        }
    }

    pub const fn source_commit(self) -> &'static str {
        match self {
            Self::Stockfish19EmbeddedNnue => super::STOCKFISH19_SOURCE_COMMIT,
            Self::Reckless090EmbeddedNnue => RECKLESS090_SOURCE_COMMIT,
        }
    }

    pub const fn source_repository(self) -> &'static str {
        match self {
            Self::Stockfish19EmbeddedNnue => STOCKFISH_SOURCE,
            Self::Reckless090EmbeddedNnue => RECKLESS_SOURCE,
        }
    }

    pub const fn license(self) -> &'static str {
        match self {
            Self::Stockfish19EmbeddedNnue => "GPL-3.0-or-later",
            Self::Reckless090EmbeddedNnue => "AGPL-3.0-only",
        }
    }

    pub const fn capabilities(self) -> ExternalEngineCapabilities {
        match self {
            Self::Stockfish19EmbeddedNnue => ExternalEngineCapabilities {
                searchmoves: true,
                combined_depth_nodes_movetime: true,
                ponder: true,
                embedded_nnue: true,
                separate_assets: false,
                exact_stack_resume: false,
            },
            // v0.9.0/src/uci.rs::parse_limits returns the first recognized
            // depth/nodes/movetime limit and does not parse searchmoves/ponder.
            // Sending Stockfish's combined go would silently lose constraints.
            Self::Reckless090EmbeddedNnue => ExternalEngineCapabilities {
                searchmoves: false,
                combined_depth_nodes_movetime: false,
                ponder: false,
                embedded_nnue: true,
                separate_assets: false,
                exact_stack_resume: false,
            },
        }
    }

    pub fn require_query(
        self,
        constraint: ExternalGoConstraint,
        root_restricted: bool,
    ) -> Result<(), CheckerError> {
        if root_restricted && !self.capabilities().searchmoves {
            return Err(CheckerError::Unsupported(
                "engine does not support searchmoves",
            ));
        }
        if matches!(constraint, ExternalGoConstraint::Combined { .. })
            && !self.capabilities().combined_depth_nodes_movetime
        {
            return Err(CheckerError::Unsupported(
                "engine does not support combined go constraints",
            ));
        }
        if constraint
            .requested_depth()
            .is_some_and(|depth| depth == 0 || depth > 64)
            || constraint.requested_nodes() == Some(0)
            || matches!(
                constraint,
                ExternalGoConstraint::MoveTime {
                    milliseconds: 0 | 180_001..
                }
            )
        {
            return Err(CheckerError::Invalid(
                "external query constraint exceeds finite bounds",
            ));
        }
        // Reckless v0.9.0 parses all paired limit values as i32. A larger
        // positive node number is skipped and can fall through to Infinite.
        if self == Self::Reckless090EmbeddedNnue
            && constraint
                .requested_nodes()
                .is_some_and(|nodes| nodes > i32::MAX as u64)
        {
            return Err(CheckerError::Invalid(
                "Reckless node limit exceeds its signed parser range",
            ));
        }
        Ok(())
    }

    /// Create engine-specific options; unsupported Stockfish-only names are
    /// never passed to Reckless. Strength limiting and tablebases stay off.
    pub fn options(
        self,
        purpose: ExternalEnginePurpose,
        resources: ExternalEngineResources,
    ) -> Result<BTreeMap<String, String>, CheckerError> {
        validate_resources(purpose, resources)?;
        let mut options = BTreeMap::from([
            ("Threads".into(), resources.threads.to_string()),
            ("Hash".into(), resources.hash_mib.to_string()),
            ("MultiPV".into(), resources.multipv.to_string()),
            ("UCI_Chess960".into(), "false".into()),
        ]);
        match self {
            Self::Stockfish19EmbeddedNnue => {
                for (name, value) in [
                    ("Skill Level", "20"),
                    ("UCI_LimitStrength", "false"),
                    ("Ponder", "false"),
                    ("NumaPolicy", "none"),
                    ("SyzygyPath", ""),
                    ("SyzygyProbeLimit", "0"),
                    ("nodestime", "0"),
                ] {
                    options.insert(name.into(), value.into());
                }
                options.insert(
                    "Move Overhead".into(),
                    resources.move_overhead_ms.to_string(),
                );
            }
            Self::Reckless090EmbeddedNnue => {
                options.insert("Minimal".into(), "false".into());
                options.insert(
                    "MoveOverhead".into(),
                    resources.move_overhead_ms.to_string(),
                );
                // SyzygyPath is compile-feature-dependent and not required.
                // No path is supplied to a fresh process, so no tablebase is
                // loaded. v0.9.0's setter requires a nonempty token; sending
                // an empty value would only produce an Unknown option report.
            }
        }
        Ok(options)
    }

    /// Validate a caller-pinned declaration before startup. No file is opened
    /// here, and actual binary hash/ELF pinning remains the existing owner's job.
    pub fn validate_identity(
        self,
        purpose: ExternalEnginePurpose,
        identity: &ExternalCheckerIdentity,
        expected_uci_name: &str,
    ) -> Result<(), CheckerError> {
        identity.validate()?;
        let version = match self {
            Self::Stockfish19EmbeddedNnue => "19",
            Self::Reckless090EmbeddedNnue => "0.9.0",
        };
        let source = self.source_repository();
        let commit = self.source_commit();
        if identity.adapter_semantics != EXTERNAL_UCI_SCORE_SEMANTICS
            || identity.launch_arguments_sha256 != super::arguments_sha256(&[])
            || identity.declared_name != self.expected_uci_name()
            || expected_uci_name != self.expected_uci_name()
            || identity.declared_version != version
            || identity.declared_license != self.license()
            || ![
                format!("{source}/tree/{commit}"),
                format!("{source}/commit/{commit}"),
            ]
            .contains(&identity.declared_source)
        {
            return Err(CheckerError::Invalid(
                "external engine profile name/version/source/license mismatch",
            ));
        }
        if !identity.assets.is_empty() {
            return Err(CheckerError::Unsupported(
                "embedded engine profile rejects separately loaded assets",
            ));
        }
        if identity.model_metadata.weights_sha256.is_some()
            || identity.model_metadata.precision.is_some()
            || identity.model_metadata.declared_rights.is_some()
        {
            return Err(CheckerError::Invalid(
                "embedded model loading/precision/rights remain unknown",
            ));
        }
        self.validate_options(purpose, &identity.options)
    }

    pub fn validate_options(
        self,
        purpose: ExternalEnginePurpose,
        options: &BTreeMap<String, String>,
    ) -> Result<(), CheckerError> {
        let numeric = |name: &str| {
            options
                .get(name)
                .and_then(|value| value.parse().ok())
                .ok_or(CheckerError::Invalid(
                    "external engine resource option missing",
                ))
        };
        let overhead = match self {
            Self::Stockfish19EmbeddedNnue => "Move Overhead",
            Self::Reckless090EmbeddedNnue => "MoveOverhead",
        };
        let resources = ExternalEngineResources {
            threads: numeric("Threads")?,
            hash_mib: options
                .get("Hash")
                .and_then(|value| value.parse().ok())
                .ok_or(CheckerError::Invalid("external engine Hash option missing"))?,
            multipv: numeric("MultiPV")?,
            move_overhead_ms: numeric(overhead)?,
        };
        let expected = self.options(purpose, resources)?;
        if options != &expected {
            return Err(CheckerError::Invalid(
                "external engine required/optional option profile mismatch",
            ));
        }
        Ok(())
    }
}

fn validate_resources(
    purpose: ExternalEnginePurpose,
    resources: ExternalEngineResources,
) -> Result<(), CheckerError> {
    if !(1..=512).contains(&resources.threads)
        || !(1..=262_144).contains(&resources.hash_mib)
        || resources.move_overhead_ms > 2000
        || resources.multipv == 0
        || resources.multipv > 4
        || (purpose == ExternalEnginePurpose::ArenaOpponent && resources.multipv != 1)
    {
        return Err(CheckerError::Invalid(
            "external engine finite resources/MultiPV purpose mismatch",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_checker::{ExternalModelMetadata, ExternalTrainingKnowledge};

    fn identity(
        profile: ExternalEngineProfile,
        purpose: ExternalEnginePurpose,
    ) -> ExternalCheckerIdentity {
        ExternalCheckerIdentity {
            adapter_semantics: EXTERNAL_UCI_SCORE_SEMANTICS.into(),
            binary_sha256: "a".repeat(64),
            launch_arguments_sha256: super::super::arguments_sha256(&[]),
            declared_name: profile.expected_uci_name().into(),
            declared_version: match profile {
                ExternalEngineProfile::Stockfish19EmbeddedNnue => "19",
                ExternalEngineProfile::Reckless090EmbeddedNnue => "0.9.0",
            }
            .into(),
            declared_source: format!(
                "{}/tree/{}",
                profile.source_repository(),
                profile.source_commit()
            ),
            declared_license: profile.license().into(),
            options: profile
                .options(purpose, ExternalEngineResources::default())
                .unwrap(),
            assets: Vec::new(),
            model_metadata: ExternalModelMetadata {
                weights_sha256: None,
                training: ExternalTrainingKnowledge::Unknown,
                declared_rights: None,
                precision: None,
            },
        }
    }

    #[test]
    fn engine_options_do_not_inherit_stockfish_names_or_optional_tablebase_requirement() {
        let profile = ExternalEngineProfile::Reckless090EmbeddedNnue;
        let purpose = ExternalEnginePurpose::ArenaOpponent;
        let options = profile
            .options(purpose, ExternalEngineResources::default())
            .unwrap();
        assert_eq!(options["MoveOverhead"], "10");
        assert_eq!(options["Minimal"], "false");
        for unsupported in [
            "NumaPolicy",
            "Skill Level",
            "UCI_LimitStrength",
            "Ponder",
            "Move Overhead",
            "SyzygyPath",
        ] {
            assert!(!options.contains_key(unsupported));
        }
        let mut optional = options.clone();
        optional.insert("SyzygyPath".into(), String::new());
        assert!(profile.validate_options(purpose, &optional).is_err());
        optional.insert("SyzygyPath".into(), "/unregistered/tablebase".into());
        assert!(profile.validate_options(purpose, &optional).is_err());
        let mut unsupported = options;
        unsupported.insert("Ponder".into(), "false".into());
        assert!(profile.validate_options(purpose, &unsupported).is_err());
    }

    #[test]
    fn arena_admission_cannot_authorize_query_constraints_the_engine_lacks() {
        let profile = ExternalEngineProfile::Reckless090EmbeddedNnue;
        let identity = identity(profile, ExternalEnginePurpose::ArenaOpponent);
        assert!(
            profile
                .validate_identity(
                    ExternalEnginePurpose::ArenaOpponent,
                    &identity,
                    "Reckless 0.9.0"
                )
                .is_ok()
        );
        assert!(
            profile
                .require_query(
                    ExternalGoConstraint::Nodes {
                        reported_nodes: 4096
                    },
                    false
                )
                .is_ok()
        );
        assert!(
            profile
                .require_query(ExternalGoConstraint::Depth { depth: 4 }, false)
                .is_ok()
        );
        assert_eq!(
            profile.require_query(
                ExternalGoConstraint::Nodes {
                    reported_nodes: 4096
                },
                true
            ),
            Err(CheckerError::Unsupported(
                "engine does not support searchmoves"
            ))
        );
        assert_eq!(
            profile.require_query(
                ExternalGoConstraint::Combined {
                    depth: 4,
                    reported_nodes: 4096
                },
                false
            ),
            Err(CheckerError::Unsupported(
                "engine does not support combined go constraints"
            ))
        );
        assert!(!profile.capabilities().searchmoves);
        assert!(!profile.capabilities().combined_depth_nodes_movetime);
        assert!(!profile.capabilities().exact_stack_resume);
    }

    #[test]
    fn source_pin_is_exact_and_embedded_model_observations_are_not_invented() {
        let profile = ExternalEngineProfile::Stockfish19EmbeddedNnue;
        let purpose = ExternalEnginePurpose::ResearchChecker;
        let original = identity(profile, purpose);
        assert!(
            profile
                .validate_identity(purpose, &original, "Stockfish 19")
                .is_ok()
        );
        let mut changed = original.clone();
        changed.declared_source = format!("https://example.org/{}", profile.source_commit());
        assert!(
            profile
                .validate_identity(purpose, &changed, "Stockfish 19")
                .is_err()
        );
        changed = original.clone();
        changed.model_metadata.precision = Some("FP32".into());
        assert!(
            profile
                .validate_identity(purpose, &changed, "Stockfish 19")
                .is_err()
        );
        changed = original;
        changed
            .options
            .insert("EvalFile".into(), "unregistered.nnue".into());
        assert!(
            profile
                .validate_identity(purpose, &changed, "Stockfish 19")
                .is_err()
        );
    }

    #[test]
    fn purpose_and_resource_limits_are_checked_before_handshake() {
        let profile = ExternalEngineProfile::Stockfish19EmbeddedNnue;
        let four = ExternalEngineResources {
            multipv: 4,
            ..Default::default()
        };
        assert!(
            profile
                .options(ExternalEnginePurpose::Teacher, four)
                .is_ok()
        );
        assert!(
            profile
                .options(ExternalEnginePurpose::ArenaOpponent, four)
                .is_err()
        );
        for resources in [
            ExternalEngineResources {
                threads: 0,
                ..Default::default()
            },
            ExternalEngineResources {
                hash_mib: 0,
                ..Default::default()
            },
            ExternalEngineResources {
                move_overhead_ms: 2001,
                ..Default::default()
            },
        ] {
            assert!(
                profile
                    .options(ExternalEnginePurpose::ResearchChecker, resources)
                    .is_err()
            );
        }
    }

    #[test]
    fn single_limit_wire_does_not_invent_a_depth_or_append_ignored_constraints() {
        let nodes = ExternalGoConstraint::Nodes {
            reported_nodes: 4096,
        };
        assert_eq!(nodes.requested_depth(), None);
        assert_eq!(nodes.command(550), ("go nodes 4096".into(), nodes));
        let depth = ExternalGoConstraint::Depth { depth: 4 };
        assert_eq!(depth.requested_nodes(), None);
        assert_eq!(depth.command(550), ("go depth 4".into(), depth));
        let requested = ExternalGoConstraint::MoveTime { milliseconds: 1000 };
        assert_eq!(
            requested.command(550),
            (
                "go movetime 550".into(),
                ExternalGoConstraint::MoveTime { milliseconds: 550 }
            )
        );
        let reckless = ExternalEngineProfile::Reckless090EmbeddedNnue;
        assert!(
            reckless
                .require_query(
                    ExternalGoConstraint::Nodes {
                        reported_nodes: i32::MAX as u64
                    },
                    false
                )
                .is_ok()
        );
        assert_eq!(
            reckless.require_query(
                ExternalGoConstraint::Nodes {
                    reported_nodes: i32::MAX as u64 + 1
                },
                false
            ),
            Err(CheckerError::Invalid(
                "Reckless node limit exceeds its signed parser range"
            ))
        );
        let combined = ExternalGoConstraint::Combined {
            depth: 4,
            reported_nodes: 4096,
        };
        assert_eq!(
            combined.command(550),
            ("go depth 4 nodes 4096 movetime 550".into(), combined)
        );
    }

    #[test]
    fn profiled_checker_refuses_unsupported_query_before_a_process_is_acquired() {
        use super::super::{ExternalCpuConfig, ExternalUciCpuChecker};
        use crate::{cpu::CpuLimits, cpu_checker::CpuChecker};
        use rz_position::Position;
        use std::sync::atomic::AtomicBool;
        use std::time::{Duration, Instant};
        let profile = ExternalEngineProfile::Reckless090EmbeddedNnue;
        let purpose = ExternalEnginePurpose::ResearchChecker;
        let config = ExternalCpuConfig {
            program: std::env::temp_dir().join("rz-absent-engine-profile-fixture"),
            arguments: Vec::new(),
            working_directory: std::env::temp_dir(),
            identity: identity(profile, purpose),
            expected_uci_name: Some(profile.expected_uci_name().into()),
            max_depth: 64,
            max_prefix_plies: 256,
            handshake_timeout: Duration::from_secs(1),
            max_task_wall_time: Duration::from_secs(2),
            stop_grace: Duration::from_millis(500),
            shutdown_grace: Duration::from_millis(500),
            max_output_bytes: 4096,
            max_line_bytes: 1024,
        };
        let mut checker = ExternalUciCpuChecker::create_profiled(config, profile, purpose).unwrap();
        assert!(checker.startup_uci().is_none());
        assert!(!checker.capabilities().root_moves);
        assert_eq!(
            checker.preflight_task(CpuLimits::default()),
            Err(CheckerError::Unsupported(
                "engine does not support combined go constraints"
            ))
        );
        let position = Position::startpos();
        let roots = position.ordered_legal_moves();
        assert_eq!(
            checker
                .analyze_profiled(
                    &position,
                    Some(&roots.moves()[..1]),
                    ExternalGoConstraint::Nodes {
                        reported_nodes: 4096
                    },
                    Instant::now() + Duration::from_secs(2),
                    &AtomicBool::new(false)
                )
                .unwrap_err(),
            CheckerError::Unsupported("engine does not support searchmoves")
        );
        assert!(checker.last_attempt().is_none());
        assert!(checker.startup_uci().is_none());
    }
}
