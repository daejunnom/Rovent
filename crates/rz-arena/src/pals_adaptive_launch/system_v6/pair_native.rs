//! Pair views over the common native snapshot owner. Native execution remains
//! explicitly unavailable until the V6 child lifetime/receipt consumer is
//! connected to the same original supervisor.

use super::LockedPalsSystemArenaPairV6;
use crate::{ArenaError, NativeEngineView, NativeLaunchDeclaration, NativePairView};
use rz_experiments::{
    ArtifactRef, ManifestError, NativeEngineRole, NativeGameClockV3, NativePairClock,
    NativeRuntimeSourceV6, NativeTimeoutsV1,
};
use std::{ffi::OsString, path::Path};

impl crate::native_launch::sealed::Sealed for LockedPalsSystemArenaPairV6 {}

impl NativeLaunchDeclaration for LockedPalsSystemArenaPairV6 {
    fn input_sha256(&self) -> &str {
        self.sha256()
    }

    fn view(&self) -> NativePairView<'_> {
        let input = self.input();
        let pair = &input.pair;
        let first_id = &input.endpoints[0].startup_lock.config.endpoint.id;
        let white_order = pair.white_order.each_ref().map(|id| {
            if id == first_id {
                NativeEngineRole::Baseline
            } else {
                NativeEngineRole::Candidate
            }
        });
        let handshake = input
            .endpoints
            .iter()
            .map(|endpoint| endpoint.handshake_max_ms)
            .max()
            .unwrap_or(0);
        let window = &input.endpoints[0].startup_lock.config.window;
        NativePairView {
            pair_id: &input.pair_id,
            white_order,
            opening: self.opening(),
            opening_artifact: &input.opening_artifact,
            runner: &input.runner,
            clock: NativePairClock::Game(NativeGameClockV3 {
                base_ms: pair.base_ms,
                increment_ms: pair.increment_ms,
            }),
            max_plies: pair.max_plies,
            timeouts: NativeTimeoutsV1 {
                startup_ms: handshake,
                handshake_ms: handshake,
                runtime_ms: window.wall_time_max_ms,
                drain_ms: window.cleanup_max_ms,
                shutdown_ms: window.cleanup_max_ms / 2,
            },
            budget: input.budget,
        }
    }

    fn declared_inputs(&self) -> Vec<&ArtifactRef> {
        self.input().declared_artifacts()
    }

    fn unique_bytes(&self) -> Result<u64, ManifestError> {
        self.input().unique_input_bytes()
    }

    fn engine_view(&self, role: NativeEngineRole) -> Result<NativeEngineView<'_>, ManifestError> {
        let endpoint = self.endpoint_view(role);
        Ok(NativeEngineView {
            engine_id: &endpoint.id,
            artifacts: &[],
            cuda_bundle: None,
            search: None,
            batch_experiment: None,
            external: Some(endpoint),
            environment: endpoint.environment.as_ref(),
        })
    }

    fn provider_name(&self) -> &'static str {
        "PALS-system-V6"
    }

    fn runtime_source_v6(&self, role: NativeEngineRole) -> Option<&NativeRuntimeSourceV6> {
        Some(
            &self
                .endpoint(role)
                .startup_lock
                .config
                .system_runtime_lock
                .runtime,
        )
    }

    fn pair_metadata_cap(&self) -> u64 {
        2 << 20
    }

    fn snapshot_relative_path(&self, artifact: &ArtifactRef) -> Option<String> {
        LockedPalsSystemArenaPairV6::snapshot_relative_path(self, artifact)
            .and_then(|path| path.strip_prefix("inputs/").map(str::to_owned))
    }

    fn runtime_arguments(
        &self,
        role: NativeEngineRole,
        runtime_root: &Path,
    ) -> Result<Vec<OsString>, ArenaError> {
        if !runtime_root.is_absolute() {
            return Err(ArenaError::Invalid(
                "PALS V6 runtime root must be absolute".into(),
            ));
        }
        let cwd = runtime_root
            .parent()
            .ok_or_else(|| ArenaError::Invalid("PALS V6 private cwd missing".into()))?;
        let endpoint = self.endpoint(role);
        let path = cwd.join(&endpoint.startup_snapshot_path);
        let path = path
            .to_str()
            .filter(|value| !value.chars().any(char::is_control))
            .ok_or_else(|| ArenaError::Invalid("PALS V6 startup path is not valid UTF-8".into()))?;
        Ok(vec![
            format!("--pals-system-startup-lock={path}").into(),
            format!(
                "--pals-system-startup-lock-sha256={}",
                endpoint.startup_lock.canonical_sha256
            )
            .into(),
            format!(
                "--pals-system-startup-ms={}",
                endpoint.startup_lock.config.native_options.startup_ms
            )
            .into(),
        ])
    }

    fn preflight_arguments(
        &self,
        role: NativeEngineRole,
        runtime_root: &Path,
    ) -> Result<Vec<OsString>, ArenaError> {
        let mut arguments = self.runtime_arguments(role, runtime_root)?;
        arguments.push("--preflight=true".into());
        Ok(arguments)
    }

    fn seed(&self) -> u64 {
        self.input().pair.seed
    }

    fn validate_execution(&self) -> Result<(), ArenaError> {
        self.input().validate()?;
        // In particular this must not set the receipt-consumer capability to
        // true before actual PALS session records and supervisor lifetime are
        // bound to the V6 child. This guard runs before snapshots or spawning.
        Err(ArenaError::Invalid(
            "PALS V6 native session owner and receipt consumer are not connected".into(),
        ))
    }
}
