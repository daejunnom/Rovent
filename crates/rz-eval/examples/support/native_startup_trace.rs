//! Diagnostic stage records shared by explicit native startup checkers.
use std::error::Error;
use std::io::Write;
use std::time::Instant;

/// These best-effort records are neither provider fences nor performance data.
pub(super) struct NativeStartupTrace(Option<Instant>);
impl NativeStartupTrace {
    #[allow(
        dead_code,
        reason = "used by the separately compiled numerical checker"
    )]
    pub(super) fn new(cost_mode: bool) -> Result<Self, Box<dyn Error>> {
        let enabled =
            std::env::var_os("RZ_PALS_NUMERIC_STARTUP_TRACE").is_some_and(|value| value == "1");
        if enabled && cost_mode {
            return Err("native startup trace cannot be combined with cost measurement".into());
        }
        Ok(Self(enabled.then(Instant::now)))
    }

    #[allow(
        dead_code,
        reason = "used by the separately compiled explicit loader checker"
    )]
    pub(super) fn explicit() -> Self {
        Self(Some(Instant::now()))
    }

    pub(super) fn stage(&self, stage: &'static str) {
        if let Some(started) = self.0 {
            // An unavailable diagnostic sink must not change the native result.
            let mut stderr = std::io::stderr().lock();
            let _ = writeln!(
                stderr,
                "RZ_PALS_CHECK_STAGE stage={stage} elapsed_ns={} performance_evidence_eligible=false physical_completion=unknown",
                started.elapsed().as_nanos()
            );
            let _ = stderr.flush();
        }
    }
}
