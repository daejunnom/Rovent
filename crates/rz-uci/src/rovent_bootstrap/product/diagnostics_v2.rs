//! Opt-in read-only observations of the original V2 product lifecycle.
//!
//! This wrapper never creates admission authority or changes the driver's
//! result. In particular, a missing row is not a physical completion fence.

use super::*;
use std::fs::File;
use std::io::Write;
use std::sync::Mutex;

#[path = "diagnostics_v2_adaptive.rs"]
mod adaptive;
#[path = "diagnostics_v2_nnuec.rs"]
mod nnuec;
#[path = "diagnostics_v2_pc.rs"]
mod pc;

const DOMAIN: &str = "rz-rovent-v2-product-observation/1";
const MAX_ROOT_ROWS: usize = 128;
const MAX_ROWS: usize = MAX_ROOT_ROWS + 1;
// Linux guarantees PIPE_BUF >= 4096. A single nonblocking pipe write at or
// below this size cannot partially interleave with another pipe writer.
const MAX_ROW_BYTES: usize = 4096;
const MAX_TOTAL_BYTES: usize = MAX_ROWS * MAX_ROW_BYTES;

trait OriginalV2Observation {
    fn project_original_root(
        &self,
        context: &SearchSessionContext,
    ) -> Result<Option<serde_json::Value>, &'static str>;
}

impl OriginalV2Observation for RoventNativeProductDriverV2 {
    fn project_original_root(
        &self,
        context: &SearchSessionContext,
    ) -> Result<Option<serde_json::Value>, &'static str> {
        pc::project_pc(&self.inner, context, &self.process)
    }
}
impl OriginalV2Observation for RoventOwnPcrProductDriverV2 {
    fn project_original_root(
        &self,
        context: &SearchSessionContext,
    ) -> Result<Option<serde_json::Value>, &'static str> {
        adaptive::project_adaptive(&self.inner, context, &self.process)
    }
}
impl OriginalV2Observation for RoventExternalPcrProductDriverV2 {
    fn project_original_root(
        &self,
        context: &SearchSessionContext,
    ) -> Result<Option<serde_json::Value>, &'static str> {
        adaptive::project_adaptive(&self.inner, context, &self.process)
    }
}
impl OriginalV2Observation for RoventOwnNnuecProductDriverV2 {
    fn project_original_root(
        &self,
        context: &SearchSessionContext,
    ) -> Result<Option<serde_json::Value>, &'static str> {
        nnuec::project_nnuec(&self.inner, context, &self.process)
    }
}

#[derive(Default)]
struct State {
    seen: Vec<rz_contracts::pals::SearchAuthority>,
    rows: usize,
    bytes: usize,
    incomplete: bool,
    sink_failed: bool,
    finish_emitted: bool,
}

/// Retains the actual product; observes its real run and finish calls only.
pub(in crate::rovent_bootstrap) struct ObservedV2<T> {
    inner: T,
    input_lock_sha256: String,
    state: Mutex<State>,
    sink: Mutex<Option<ObservationSink>>,
    observation_lost: AtomicBool,
}

struct ObservationSink {
    file: File,
    #[cfg(target_os = "linux")]
    identity: (u64, u64),
}

#[cfg(target_os = "linux")]
fn nonblocking_stderr_pipe() -> Option<ObservationSink> {
    use std::os::unix::fs::{FileTypeExt, MetadataExt, OpenOptionsExt};
    let before = std::fs::metadata("/proc/self/fd/2").ok()?;
    if !before.file_type().is_fifo() {
        return None;
    }
    // Reopen the proc FD rather than changing O_NONBLOCK on the original
    // stderr file description, which the rest of the product may share.
    let file = std::fs::OpenOptions::new()
        .write(true)
        .custom_flags(nix::libc::O_NONBLOCK | nix::libc::O_CLOEXEC)
        .open("/proc/self/fd/2")
        .ok()?;
    let opened = file.metadata().ok()?;
    let after = std::fs::metadata("/proc/self/fd/2").ok()?;
    let identity = (before.dev(), before.ino());
    let flags = nix::fcntl::fcntl(&file, nix::fcntl::FcntlArg::F_GETFL).ok()?;
    if !opened.file_type().is_fifo()
        || !after.file_type().is_fifo()
        || (opened.dev(), opened.ino()) != identity
        || (after.dev(), after.ino()) != identity
        || flags & nix::libc::O_NONBLOCK == 0
    {
        return None;
    }
    Some(ObservationSink { file, identity })
}
#[cfg(not(target_os = "linux"))]
fn nonblocking_stderr_pipe() -> Option<ObservationSink> {
    None
}

#[cfg(target_os = "linux")]
fn same_stderr_pipe(sink: &ObservationSink) -> bool {
    use std::os::unix::fs::{FileTypeExt, MetadataExt};
    std::fs::metadata("/proc/self/fd/2")
        .ok()
        .is_some_and(|meta| meta.file_type().is_fifo() && (meta.dev(), meta.ino()) == sink.identity)
}
#[cfg(not(target_os = "linux"))]
fn same_stderr_pipe(_: &ObservationSink) -> bool {
    false
}

impl<T> ObservedV2<T> {
    pub(in crate::rovent_bootstrap) fn new(inner: T, input_lock_sha256: String) -> Self {
        let sink = nonblocking_stderr_pipe();
        let unavailable = sink.is_none();
        Self {
            inner,
            input_lock_sha256,
            state: Mutex::new(State::default()),
            sink: Mutex::new(sink),
            observation_lost: AtomicBool::new(unavailable),
        }
    }

    fn emit(
        &self,
        row: serde_json::Value,
        authority: Option<rz_contracts::pals::SearchAuthority>,
        incomplete: bool,
    ) {
        let Ok(mut sink) = self.sink.try_lock() else {
            self.observation_lost.store(true, Ordering::Release);
            return;
        };
        let Some(sink) = sink.as_mut() else {
            self.observation_lost.store(true, Ordering::Release);
            return;
        };
        if !same_stderr_pipe(sink) {
            self.observation_lost.store(true, Ordering::Release);
            return;
        }
        self.emit_to(&mut sink.file, row, authority, incomplete);
    }

    fn emit_to(
        &self,
        sink: &mut impl Write,
        mut row: serde_json::Value,
        authority: Option<rz_contracts::pals::SearchAuthority>,
        incomplete: bool,
    ) {
        let Ok(mut state) = self.state.try_lock() else {
            self.observation_lost.store(true, Ordering::Release);
            return;
        };
        if state.sink_failed {
            return;
        }
        state.incomplete |= incomplete;
        if let Some(authority) = authority {
            if state.rows >= MAX_ROOT_ROWS || state.finish_emitted {
                state.incomplete = true;
                self.observation_lost.store(true, Ordering::Release);
                return;
            }
            if state.seen.contains(&authority) {
                state.incomplete = true;
                row["source_status"] = serde_json::json!("duplicate_authority_unknown");
                row["root"] = serde_json::Value::Null;
            } else if state.seen.len() < MAX_ROOT_ROWS {
                state.seen.push(authority);
            } else {
                state.incomplete = true;
            }
        } else if state.finish_emitted {
            state.incomplete = true;
            self.observation_lost.store(true, Ordering::Release);
            return;
        }
        if state.rows >= MAX_ROWS {
            state.incomplete = true;
            self.observation_lost.store(true, Ordering::Release);
            return;
        }
        row["domain"] = serde_json::json!(DOMAIN);
        row["observation_complete_so_far"] =
            serde_json::json!(!state.incomplete && !self.observation_lost.load(Ordering::Acquire));
        let Ok(mut encoded) = serde_json::to_vec(&row) else {
            state.incomplete = true;
            self.observation_lost.store(true, Ordering::Release);
            return;
        };
        encoded.push(b'\n');
        if encoded.len() > MAX_ROW_BYTES
            || state.bytes.saturating_add(encoded.len()) > MAX_TOTAL_BYTES
        {
            state.incomplete = true;
            self.observation_lost.store(true, Ordering::Release);
            return;
        }
        // Single bounded write; do not retry partial writes or EAGAIN because
        // the original search/finish and its cleanup take precedence.
        if !matches!(sink.write(&encoded), Ok(count) if count == encoded.len())
            || sink.flush().is_err()
        {
            state.incomplete = true;
            state.sink_failed = true;
            self.observation_lost.store(true, Ordering::Release);
            return;
        }
        state.rows += 1;
        state.bytes += encoded.len();
        if authority.is_none() {
            state.finish_emitted = true;
        }
    }
}

fn failure_observation(failure: &SearchSessionFailure) -> serde_json::Value {
    let detail: String = failure.detail.chars().take(256).collect();
    let detail_truncated = detail.len() < failure.detail.len();
    let physical = match failure.physical_completion {
        DriverPhysicalCompletion::Confirmed => "confirmed",
        DriverPhysicalCompletion::Unknown => "unknown",
    };
    serde_json::json!({"code":failure.code, "detail":detail,
        "detail_truncated":detail_truncated, "reported_physical_completion":physical})
}

fn authority_observation(authority: rz_contracts::pals::SearchAuthority) -> serde_json::Value {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut implementation = String::with_capacity(64);
    for byte in authority.implementation.0 {
        implementation.push(DIGITS[(byte >> 4) as usize] as char);
        implementation.push(DIGITS[(byte & 15) as usize] as char);
    }
    serde_json::json!({
        "epoch": authority.epoch.0,
        "game": authority.game.0,
        "root": authority.root.0,
        "implementation_sha256": implementation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::SessionScope;
    use crate::search_driver::DriverWork;
    use rz_contracts::{CancelToken, GameGeneration, ProcessEpoch, RootGeneration};
    use rz_search::driver::SearchControl;
    use std::io;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    fn authority(root: u64) -> rz_contracts::pals::SearchAuthority {
        rz_contracts::pals::SearchAuthority {
            epoch: ProcessEpoch(1),
            game: GameGeneration(2),
            root: RootGeneration(root),
            implementation: Digest([3; 32]),
        }
    }

    fn context() -> SearchSessionContext {
        let authority = authority(4);
        let deadline = Instant::now() + Duration::from_secs(5);
        SearchSessionContext {
            authority,
            original_go: None,
            original_time_budget: None,
            original_start: None,
            effective_clock: None,
            root_moves: None,
            goal: rz_contracts::rovent::SearchGoal {
                depth_plies: None,
                mate_moves: None,
            },
            requested_nodes: None,
            control: SearchControl::new(deadline, 1),
            cancellation: CancelToken::new(),
            shutdown_deadline: deadline + Duration::from_secs(1),
            current: Arc::new(Mutex::new(SessionScope::Search(authority))),
        }
    }

    struct MockDriver {
        fail_run: bool,
        runs: AtomicUsize,
        finishes: AtomicUsize,
    }

    impl OriginalV2Observation for MockDriver {
        fn project_original_root(
            &self,
            _: &SearchSessionContext,
        ) -> Result<Option<serde_json::Value>, &'static str> {
            Err("mock_receipt_busy")
        }
    }

    impl SearchSessionDriver for MockDriver {
        fn kind(&self) -> SearchKind {
            SearchKind::Cpu
        }
        fn implementation(&self) -> Digest {
            Digest([3; 32])
        }
        fn identity(&self) -> EngineIdentity {
            EngineIdentity {
                name: "mock".into(),
                author: "test".into(),
            }
        }
        fn run(
            &self,
            _: &Position,
            _: &SearchSessionContext,
            _: &mut dyn FnMut(BoardMove),
        ) -> Result<SearchSessionReport, SearchSessionFailure> {
            self.runs.fetch_add(1, Ordering::Relaxed);
            if self.fail_run {
                Err(SearchSessionFailure {
                    physical_completion: DriverPhysicalCompletion::Unknown,
                    code: "mock_failure",
                    detail: "original failure".into(),
                })
            } else {
                Ok(SearchSessionReport {
                    mate_strategy: None,
                    best_move: None,
                    work: DriverWork::Cpu {
                        nodes: 7,
                        completed_depth: 2,
                    },
                })
            }
        }
        fn finish_session(
            &self,
            _: Instant,
        ) -> Result<Option<DriverPhysicalCompletion>, SearchSessionFailure> {
            self.finishes.fetch_add(1, Ordering::Relaxed);
            Ok(Some(DriverPhysicalCompletion::Confirmed))
        }
    }

    struct FailingSink {
        calls: usize,
    }
    impl Write for FailingSink {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            self.calls += 1;
            Err(io::Error::other("fixture output failure"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn observation_write_failure_is_latched_and_never_retried() {
        let observed = ObservedV2::new((), String::new());
        let mut sink = FailingSink { calls: 0 };
        observed.emit_to(&mut sink, serde_json::json!({"event":"root"}), None, false);
        observed.emit_to(
            &mut sink,
            serde_json::json!({"event":"finish"}),
            None,
            false,
        );
        assert_eq!(sink.calls, 1);
        assert!(observed.observation_lost.load(Ordering::Acquire));
        assert!(observed.state.lock().unwrap().sink_failed);
    }

    #[test]
    fn incomplete_observation_is_not_promoted_by_confirmed_finish() {
        let observed = ObservedV2::new((), String::new());
        let mut sink = Vec::new();
        observed.emit_to(
            &mut sink,
            serde_json::json!({"event":"finish", "physical_completion":"confirmed"}),
            None,
            true,
        );
        let row: serde_json::Value = serde_json::from_slice(&sink).unwrap();
        assert_eq!(row["physical_completion"], "confirmed");
        assert_eq!(row["observation_complete_so_far"], false);
    }

    #[test]
    fn observation_failure_preserves_original_run_and_finish_results() {
        for fail_run in [false, true] {
            let observed = ObservedV2::new(
                MockDriver {
                    fail_run,
                    runs: AtomicUsize::new(0),
                    finishes: AtomicUsize::new(0),
                },
                "lock".into(),
            );
            let result = observed.run(&Position::startpos(), &context(), &mut |_| {});
            if fail_run {
                let failure = result.unwrap_err();
                assert_eq!(failure.code, "mock_failure");
                assert_eq!(failure.detail, "original failure");
                assert_eq!(
                    failure.physical_completion,
                    DriverPhysicalCompletion::Unknown
                );
            } else {
                assert!(matches!(
                    result.unwrap().work,
                    DriverWork::Cpu {
                        nodes: 7,
                        completed_depth: 2
                    }
                ));
            }
            assert_eq!(observed.inner.runs.load(Ordering::Relaxed), 1);
            assert_eq!(
                observed
                    .finish_session(Instant::now() + Duration::from_secs(1))
                    .unwrap(),
                Some(DriverPhysicalCompletion::Confirmed)
            );
            assert_eq!(observed.inner.finishes.load(Ordering::Relaxed), 1);
        }
    }

    #[test]
    fn duplicate_authority_and_busy_observation_remain_incomplete() {
        let observed = ObservedV2::new((), String::new());
        observed.observation_lost.store(false, Ordering::Release);
        let mut sink = Vec::new();
        let root = authority(4);
        observed.emit_to(
            &mut sink,
            serde_json::json!({"event":"root", "root":{"value":1}}),
            Some(root),
            false,
        );
        observed.emit_to(
            &mut sink,
            serde_json::json!({"event":"root", "root":{"value":2}}),
            Some(root),
            false,
        );
        let rows: Vec<serde_json::Value> = sink
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap())
            .collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1]["source_status"], "duplicate_authority_unknown");
        assert!(rows[1]["root"].is_null());
        assert_eq!(rows[1]["observation_complete_so_far"], false);

        let state_guard = observed.state.lock().unwrap();
        observed.emit_to(
            &mut sink,
            serde_json::json!({"event":"root"}),
            Some(authority(5)),
            false,
        );
        drop(state_guard);
        assert!(observed.observation_lost.load(Ordering::Acquire));
        assert_eq!(sink.iter().filter(|byte| **byte == b'\n').count(), 2);
    }
}

impl<T: SearchSessionDriver + OriginalV2Observation> SearchSessionDriver for ObservedV2<T> {
    fn supports_ponder_clock(&self) -> bool {
        self.inner.supports_ponder_clock()
    }
    fn node_budget_semantics(&self) -> crate::bridge::NodeBudgetSemantics {
        self.inner.node_budget_semantics()
    }
    fn validate_go(
        &self,
        position: &Position,
        go: &crate::parser::GoLimits,
    ) -> Result<(), ContractError> {
        self.inner.validate_go(position, go)
    }
    fn search_goal_capability(&self) -> Option<NonZeroU16> {
        self.inner.search_goal_capability()
    }
    fn kind(&self) -> SearchKind {
        self.inner.kind()
    }
    fn implementation(&self) -> Digest {
        self.inner.implementation()
    }
    fn identity(&self) -> EngineIdentity {
        self.inner.identity()
    }
    fn pals_checker_registration(&self) -> Option<PalsCheckerRegistration> {
        self.inner.pals_checker_registration()
    }
    fn checker_started(&self) -> Option<bool> {
        self.inner.checker_started()
    }
    fn work_receipt(
        &self,
    ) -> Result<Option<crate::search_driver::ProcessSearchWorkReceipt>, ContractError> {
        self.inner.work_receipt()
    }
    fn checker_startup_uci(
        &self,
    ) -> Result<Option<rz_search::cpu_checker::ExternalUciIdentity>, SearchSessionFailure> {
        self.inner.checker_startup_uci()
    }
    fn start_checker(
        &self,
        until: Instant,
        cancel: Arc<AtomicBool>,
    ) -> Result<Option<bool>, SearchSessionFailure> {
        self.inner.start_checker(until, cancel)
    }
    fn checker_attempts(
        &self,
    ) -> Result<Option<Vec<rz_search::cpu_checker::CheckerAttempt>>, SearchSessionFailure> {
        self.inner.checker_attempts()
    }
    fn checker_last_attempt(
        &self,
    ) -> Result<Option<rz_search::cpu_checker::CheckerAttempt>, SearchSessionFailure> {
        self.inner.checker_last_attempt()
    }
    fn finish_checker(
        &self,
        until: Instant,
    ) -> Result<Option<rz_search::cpu_checker::CheckerShutdown>, SearchSessionFailure> {
        self.inner.finish_checker(until)
    }
    fn checker_shutdown(
        &self,
    ) -> Result<Option<rz_search::cpu_checker::CheckerShutdown>, SearchSessionFailure> {
        self.inner.checker_shutdown()
    }
    #[cfg(feature = "search-work-receipts")]
    fn followup_lifecycle(
        &self,
    ) -> Result<Option<crate::pals_attestation::followup::PalsFollowupLifecycleV4>, ContractError>
    {
        self.inner.followup_lifecycle()
    }
    #[cfg(feature = "search-work-receipts")]
    fn seal_followup_lifecycle(&self, protocol_retired: bool) -> Result<(), ContractError> {
        self.inner.seal_followup_lifecycle(protocol_retired)
    }
    #[cfg(feature = "search-work-receipts")]
    fn checker_startup_resources(
        &self,
    ) -> Result<
        Option<crate::pals_attestation::checker::resource::ReadyBoundaryResources>,
        SearchSessionFailure,
    > {
        self.inner.checker_startup_resources()
    }
    #[cfg(feature = "search-work-receipts")]
    fn checker_startup_resource_unavailable(
        &self,
    ) -> Result<Option<&'static str>, SearchSessionFailure> {
        self.inner.checker_startup_resource_unavailable()
    }
    fn reset_game(&self) -> Result<(), ContractError> {
        self.inner.reset_game()
    }
    fn accept_uci_output(
        &self,
        authority: rz_contracts::pals::SearchAuthority,
        bestmove: &str,
        from_report: bool,
    ) -> Result<(), ContractError> {
        self.inner
            .accept_uci_output(authority, bestmove, from_report)
    }
    fn run(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        let result = self.inner.run(position, context, progress);
        let observation = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.inner.project_original_root(context)
        }));
        let (status, root) = match observation {
            Ok(Ok(Some(root))) => ("original_receipt_matched", Some(root)),
            Ok(Ok(None)) => ("original_receipt_unavailable", None),
            Ok(Err(issue)) => (issue, None),
            Err(_) => ("observation_panicked", None),
        };
        let row = serde_json::json!({
            "event":"root", "authority":authority_observation(context.authority),
            "input_lock_sha256":self.input_lock_sha256.as_str(),
            "source_status":status, "root":root, "driver_result":if result.is_ok(){"ok"}else{"error"},
            "driver_failure":result.as_ref().err().map(failure_observation),
            "root_physical_completion":"unknown_unless_explicit_in_original_receipt",
        });
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.emit(
                row,
                Some(context.authority),
                status != "original_receipt_matched",
            )
        }))
        .is_err()
        {
            self.observation_lost.store(true, Ordering::Release);
        }
        result
    }
    fn finish_session(
        &self,
        until: Instant,
    ) -> Result<Option<DriverPhysicalCompletion>, SearchSessionFailure> {
        let result = self.inner.finish_session(until);
        let physical = match &result {
            Ok(Some(DriverPhysicalCompletion::Confirmed)) => "confirmed",
            Ok(Some(DriverPhysicalCompletion::Unknown)) => "unknown",
            Ok(None) => "unavailable",
            Err(_) => "error_unknown",
        };
        let row = serde_json::json!({"event":"finish", "input_lock_sha256":self.input_lock_sha256.as_str(), "physical_completion":physical,
            "driver_result":if result.is_ok(){"ok"}else{"error"},
            "driver_failure":result.as_ref().err().map(failure_observation)});
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.emit(row, None, false)))
            .is_err()
        {
            self.observation_lost.store(true, Ordering::Release);
        }
        result
    }
}
