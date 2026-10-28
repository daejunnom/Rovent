//! Authored ELF/pipe fixtures exercise the real profiled worker and semantic
//! owner. The declared protocol profile is test data, not an official engine
//! binary, NNUE/model execution, native-node calibration, or strength evidence.
#![cfg(target_os = "linux")]

use super::*;
use crate::contract_time::{ContractClock, InstantClock};
use crate::cpu_checker::{
    CheckerCapabilities, ExternalCheckerIdentity, ExternalModelMetadata, ExternalTrainingKnowledge,
};
use crate::external_cpu::worker::ExternalWorkerClock;
use crate::external_cpu::{
    EXTERNAL_FIXTURE_LOCK, ExternalEngineProfile, ExternalEngineResources, ExternalUciCpuChecker,
    ProfiledExternalWorker,
};
use crate::pals::coordinator::{FrameId, FrameTransition};
use crate::pals::engine::{PostRepairRecheckPolicy, ResearcherSemanticProfile, ResolverPolicy};
use crate::pals::review_context::{ReviewIndexLimits, ReviewSelectionPolicy};
use crate::pals::rovent_engine::RootMoveDomain;
use crate::pals::rovent_frontier::{FrontierLimits, FrontierRegistration};
use crate::pals::store::{ArchiveConfig, TaskStatus};
use rz_contracts::pals_adaptive::CpuWorkerId;
use rz_contracts::rovent::{ExecutionMode, SearchGoal};
use rz_contracts::{GameGeneration, PrecisionProfile, ProcessEpoch, RootGeneration};
use std::fs::{self, File};
use std::io::Read;
use std::num::NonZeroU16;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::AtomicU64;

type Port = AdaptiveEnginePort<AdaptivePortModel>;
type Work = CoordinatorWork<ReviewView, AdaptivePortQuestion>;

mod peer;

const COMPILE_OUTPUT_CAP: u64 = 16 * 1024 * 1024;
const POLL: Duration = Duration::from_millis(5);

// Keep the fixture protocol small. There is no search/evaluator here: the
// tested positions are startpos after e2e4 and e2e4 e7e5. These two fixed
// protocol replies are legal; no search or evaluator chooses them.
const AUTHORED_UCI: &str = r#"
use std::io::{self, BufRead, Write};
const LONG_PV: bool = false;
fn main() {
    let mut after_reply = false;
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        if line == "uci" {
            println!("id name Reckless 0.9.0");
            println!("id author RoveZero authored protocol fixture");
            println!("option name Threads type spin default 1 min 1 max 512");
            println!("option name Hash type spin default 256 min 1 max 262144");
            println!("option name MultiPV type spin default 1 min 1 max 4");
            println!("option name UCI_Chess960 type check default false");
            println!("option name Minimal type check default false");
            println!("option name MoveOverhead type spin default 10 min 0 max 2000");
            println!("uciok");
        } else if line == "isready" {
            println!("readyok");
        } else if line.starts_with("position ") {
            after_reply = line == "position startpos moves e2e4 e7e5";
        } else if line.starts_with("go ") {
            // The real adapter, not this fixture, owns requested/issued limits.
            let movement = if after_reply { "g1f3" } else { "e7e5" };
            let pv = if LONG_PV {
                if after_reply { "g1f3 b8c6" } else { "e7e5 g1f3" }
            } else { movement };
            println!("info depth 2 seldepth 3 nodes 7 score cp 21 pv {pv}");
            println!("bestmove {movement}");
        } else if line == "quit" {
            break;
        }
        io::stdout().flush().unwrap();
    }
}
"#;

/// A test-owned executable outside the repository. Creation runs only when
/// Root explicitly runs this CPU test. Compilation has two allowed CPUs,
/// a ten-second wall fence, a process group, and bounded output files.
struct AuthoredElf {
    root: PathBuf,
    temporary_base: PathBuf,
    program: PathBuf,
}

impl AuthoredElf {
    fn new(long_pv: bool) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let temporary_base = std::env::temp_dir().canonicalize().unwrap();
        let root = temporary_base.join(format!(
            "rz-profiled-authored-worker-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let fixture = Self {
            program: root.join("authored-uci"),
            root,
            temporary_base,
        };
        let source = fixture.root.join("authored_uci.rs");
        let stdout = fixture.root.join("rustc.stdout");
        let stderr = fixture.root.join("rustc.stderr");
        let compiler_tmp = fixture.root.join("compiler-tmp");
        fs::create_dir(&compiler_tmp).unwrap();
        // The registered Reckless wire profile permits no launch arguments.
        // Compile a separately hashed authored ELF for the negative PV case
        // instead of changing that guarded launch contract.
        let selector = "const LONG_PV: bool = false;";
        assert_eq!(AUTHORED_UCI.matches(selector).count(), 1);
        let authored_source = if long_pv {
            AUTHORED_UCI.replacen(selector, "const LONG_PV: bool = true;", 1)
        } else {
            AUTHORED_UCI.to_owned()
        };
        fs::write(&source, authored_source).unwrap();
        let cpus = first_two_allowed_cpus();
        let compiler = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        let mut command = Command::new("prlimit");
        command
            .arg(format!("--fsize={COMPILE_OUTPUT_CAP}"))
            .args([
                "--cpu=10",
                "--",
                "timeout",
                "--signal=TERM",
                "--kill-after=1s",
                "10s",
                "taskset",
                "-c",
            ])
            .arg(cpus)
            .arg(compiler)
            .arg("--edition=2021")
            .arg("--crate-name=rz_authored_profiled_uci")
            .arg("-Ccodegen-units=2")
            .arg("-Cdebuginfo=0")
            .arg("-Copt-level=0")
            .arg(&source)
            .arg("-o")
            .arg(&fixture.program)
            .current_dir(&fixture.root)
            .env("TMPDIR", &compiler_tmp)
            .stdin(Stdio::null())
            .stdout(Stdio::from(File::create(&stdout).unwrap()))
            .stderr(Stdio::from(File::create(&stderr).unwrap()))
            .process_group(0);
        // OS tools retain the resource fence without a Rust pre-exec/FFI hook.
        // Missing timeout/taskset/rustc is an explicit setup failure captured
        // below; none of the guards silently falls back to an unbounded run.
        let mut child = command
            .spawn()
            .expect("authored ELF setup requires prlimit, timeout, taskset and rustc");
        let group = child.id();
        assert!(group > 0 && group <= i32::MAX as u32);
        let until = Instant::now() + Duration::from_secs(10);
        let status = loop {
            let bytes = owned_output_bytes(&fixture.root);
            if Instant::now() >= until || bytes > COMPILE_OUTPUT_CAP {
                // process_group(0) made the validated positive child PID its
                // own group ID. Pass it as one OS-command argument, not shell
                // source, so only this original group can receive the signal.
                let killed = Command::new("kill")
                    .args(["-KILL", "--"])
                    .arg(format!("-{group}"))
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status();
                if !killed.as_ref().is_ok_and(|status| status.success()) {
                    // The safe direct-child kill is a final reap fallback.
                    // GNU timeout still owns its one-second descendant fence.
                    let _ = child.kill();
                }
                let _ = child.wait();
                assert!(killed.is_ok(), "authored ELF setup requires OS kill");
                panic!("authored ELF compile exceeded ten seconds or 16MiB output");
            }
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            std::thread::sleep(POLL);
        };
        let bytes = owned_output_bytes(&fixture.root);
        assert!(bytes <= COMPILE_OUTPUT_CAP);
        let mut diagnostics = String::new();
        File::open(stderr)
            .unwrap()
            .take(16 * 1024)
            .read_to_string(&mut diagnostics)
            .unwrap();
        assert!(status.success(), "authored ELF compile: {diagnostics}");
        fixture
    }

    fn config(&self) -> ExternalCpuConfig {
        let profile = ExternalEngineProfile::Reckless090EmbeddedNnue;
        let arguments = Vec::new();
        let arguments_sha256 = crate::external_cpu::arguments_sha256(&arguments);
        let mut hash = Sha256::new();
        let mut file = File::open(&self.program).unwrap();
        let mut buffer = [0_u8; 65_536];
        loop {
            let count = file.read(&mut buffer).unwrap();
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
        }
        ExternalCpuConfig {
            program: self.program.clone(),
            arguments,
            working_directory: self.root.clone(),
            identity: ExternalCheckerIdentity {
                adapter_semantics: crate::external_cpu::EXTERNAL_UCI_SCORE_SEMANTICS.into(),
                binary_sha256: format!("{:x}", hash.finalize()),
                launch_arguments_sha256: arguments_sha256,
                // Authored declarations select the guarded wire protocol; the
                // binary hash above is this fixture's, never the official pin.
                declared_name: profile.expected_uci_name().into(),
                declared_version: "0.9.0".into(),
                declared_source: format!(
                    "{}/tree/{}",
                    profile.source_repository(),
                    profile.source_commit()
                ),
                declared_license: profile.license().into(),
                options: profile
                    .options(
                        ExternalEnginePurpose::ResearchChecker,
                        ExternalEngineResources::default(),
                    )
                    .unwrap(),
                assets: Vec::new(),
                model_metadata: ExternalModelMetadata {
                    weights_sha256: None,
                    training: ExternalTrainingKnowledge::Unknown,
                    declared_rights: None,
                    precision: None,
                },
            },
            expected_uci_name: Some(profile.expected_uci_name().into()),
            max_depth: 8,
            max_prefix_plies: 64,
            handshake_timeout: Duration::from_secs(2),
            max_task_wall_time: Duration::from_secs(2),
            stop_grace: Duration::from_millis(500),
            shutdown_grace: Duration::from_secs(1),
            max_output_bytes: 65_536,
            max_line_bytes: 4096,
        }
    }
}

fn owned_output_bytes(root: &Path) -> u64 {
    let mut pending = vec![root.to_path_buf()];
    let mut bytes = 0_u64;
    let mut entries = 0_usize;
    while let Some(directory) = pending.pop() {
        let Ok(children) = fs::read_dir(directory) else {
            continue;
        };
        for child in children.flatten() {
            entries += 1;
            if entries > 256 {
                return COMPILE_OUTPUT_CAP + 1;
            }
            let Ok(metadata) = fs::symlink_metadata(child.path()) else {
                continue;
            };
            if metadata.file_type().is_symlink() {
                return COMPILE_OUTPUT_CAP + 1;
            }
            if metadata.is_dir() {
                pending.push(child.path());
            } else {
                bytes = bytes.saturating_add(metadata.len());
            }
            if bytes > COMPILE_OUTPUT_CAP {
                return bytes;
            }
        }
    }
    bytes
}

impl Drop for AuthoredElf {
    fn drop(&mut self) {
        // Only the exact, non-symlink directory created above is ours. Worker
        // shutdown/join precedes this owner in AuthoredWorkerFixture::drop.
        if self.root.parent() == Some(self.temporary_base.as_path())
            && fs::symlink_metadata(&self.root).is_ok_and(|metadata| metadata.file_type().is_dir())
        {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
}

fn first_two_allowed_cpus() -> String {
    let status = fs::read_to_string("/proc/self/status").unwrap();
    let allowed = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .expect("Linux affinity readback")
        .trim();
    let mut cpus = Vec::new();
    for segment in allowed.split(',') {
        let (first, last) = segment.split_once('-').unwrap_or((segment, segment));
        for cpu in first.parse::<u32>().unwrap()..=last.parse::<u32>().unwrap() {
            cpus.push(cpu.to_string());
            if cpus.len() == 2 {
                return cpus.join(",");
            }
        }
    }
    assert!(!cpus.is_empty());
    cpus.join(",")
}

/// Metadata only: any accidental manager-side CPU execution fails immediately.
struct MetadataChecker {
    identity: CheckerIdentity,
    conditions: String,
    capabilities: CheckerCapabilities,
}
impl CpuChecker for MetadataChecker {
    fn identity(&self) -> &CheckerIdentity {
        &self.identity
    }
    fn conditions(&self) -> &str {
        &self.conditions
    }
    fn capabilities(&self) -> CheckerCapabilities {
        self.capabilities
    }
    fn last_attempt(&self) -> Option<&CheckerAttempt> {
        None
    }
    fn reset_attempt(&mut self) {}
    fn analyze(
        &mut self,
        _: &Position,
        _: CpuLimits,
        _: &AtomicBool,
    ) -> Result<CheckerReport, CheckerError> {
        panic!("semantic owner must not execute the authored external worker")
    }
    fn new_game(&mut self, _: Instant, _: &AtomicBool) -> Result<(), CheckerError> {
        Ok(())
    }
    fn shutdown(&mut self, _: Instant) -> Result<CheckerShutdown, CheckerError> {
        Err(CheckerError::Unsupported(
            "metadata has no physical process",
        ))
    }
}

struct AuthoredWorkerFixture {
    port: Port,
    worker: Option<ProfiledExternalWorker>,
    source: ProfiledExternalRootSource,
    config: ExternalCpuConfig,
    clock: InstantClock,
    elf: AuthoredElf,
}

impl AuthoredWorkerFixture {
    fn new(lifetime: Duration, archive: bool) -> Self {
        Self::with_goal_and_pv(lifetime, archive, 2, false)
    }

    fn with_goal_and_pv(lifetime: Duration, archive: bool, goal_plies: u16, long_pv: bool) -> Self {
        Self::with_root_limits(lifetime, archive, goal_plies, 128, long_pv)
    }

    fn with_root_limits(
        lifetime: Duration,
        archive: bool,
        goal_plies: u16,
        root_nodes: u64,
        long_pv: bool,
    ) -> Self {
        Self::with_root_limits_and_record_limit(
            lifetime, archive, goal_plies, root_nodes, long_pv, 16,
        )
    }

    fn with_root_limits_and_record_limit(
        lifetime: Duration,
        archive: bool,
        goal_plies: u16,
        root_nodes: u64,
        long_pv: bool,
        max_records: usize,
    ) -> Self {
        let goal_plies = NonZeroU16::new(goal_plies).expect("authored original goal is nonzero");
        let elf = AuthoredElf::new(long_pv);
        let config = elf.config();
        let authority = SearchAuthority {
            epoch: ProcessEpoch(73),
            game: GameGeneration(1),
            root: RootGeneration(1),
            implementation: ContractDigest([19; 32]),
        };
        let origin = Instant::now();
        let clock = InstantClock::new(ClockDomain(authority.epoch), origin);
        let deadline = Deadline {
            clock: clock.domain(),
            at: MonotonicTick(u64::try_from(lifetime.as_nanos()).unwrap()),
        };
        let worker = ProfiledExternalWorker::spawn(
            config.clone(),
            ExternalEngineProfile::Reckless090EmbeddedNnue,
            ExternalEnginePurpose::ResearchChecker,
            CpuWorkerId(1),
            ExternalWorkerClock {
                domain: clock.domain(),
                origin,
            },
            authority,
            origin + Duration::from_secs(5),
            None,
        )
        .unwrap();
        let source = poll(&worker, || worker.current_root_source().unwrap());
        let reset = worker.initial_reset_proof().unwrap().unwrap();
        assert!(reset.is_initial() && reset.matches_source(&source));

        let value = LegalOrderRoleMock.value_identity().unwrap().clone();
        let namespace = AdaptiveModelNamespace::new(
            value.clone(),
            ContractDigest([5; 32]),
            ContractDigest([6; 32]),
            PrecisionProfile::Fp32,
            1,
        )
        .unwrap();
        let model = AdaptivePortModel::new_with_namespace(
            value.model.clone(),
            Some(value),
            namespace.clone(),
        )
        .unwrap();
        // Obtain the actual profiled declaration without starting a second
        // process. The metadata mirror preserves its exact identity, condition
        // and capability contract; legacy searchmoves admission stays strict.
        let declared_checker = ExternalUciCpuChecker::create_profiled(
            config.clone(),
            ExternalEngineProfile::Reckless090EmbeddedNnue,
            ExternalEnginePurpose::ResearchChecker,
        )
        .unwrap();
        let declared = declared_checker.profiled_researcher_admission().unwrap();
        let metadata = MetadataChecker {
            identity: declared.identity().clone(),
            conditions: declared.conditions().to_owned(),
            capabilities: declared.capabilities(),
        };
        drop(declared_checker);
        let mut engine = PalsEngine::new_with_boxed_profiled_researcher_and_rovent_profile(
            ResearcherSemanticProfile {
                config: PalsConfig {
                    max_nodes: 257,
                    max_records,
                    beam_width: 1,
                    line_plies: 2,
                    cpu_nodes_per_task: 32,
                    ..PalsConfig::default()
                },
                maximum_root_prefix_plies: 64,
            },
            model,
            Box::new(metadata),
            ResolverPolicy::ModelWdlRestricted,
            PostRepairRecheckPolicy::Disabled,
            &declared,
        )
        .unwrap()
        .with_selected_review_context(
            ReviewSelectionPolicy::default(),
            ReviewIndexLimits::default(),
        )
        .unwrap();
        engine.new_game();
        assert_eq!(engine.semantic.game_generation, Some(authority.game.0));
        let sequence = Arc::new(AtomicU64::new(1));
        let issuer: AdaptiveRulesOwnerIssuer =
            Arc::new(move || Ok(OwnerId(sequence.fetch_add(1, Ordering::Relaxed))));
        let mut port = Port::new(
            engine,
            Position::startpos(),
            authority,
            deadline,
            PalsLimits {
                deadline: origin + lifetime,
                max_rounds: 1,
                max_cpu_nodes: root_nodes,
                cpu_depth: 2,
            },
            issuer,
        )
        .unwrap();
        if archive {
            port.engine
                .semantic
                .stores
                .enable_archive(ArchiveConfig::new(
                    elf.root.join("archive"),
                    PathBuf::from(env!("CARGO_MANIFEST_DIR")),
                ))
                .unwrap();
        }
        let first = BoardMove::from_uci("e2e4").unwrap();
        let domain = Arc::new(RootMoveDomain::new(&port.rules, Some(&[first])).unwrap());
        let budget = ContractDigest([10; 32]);
        port.attach_rovent_external_frontier(
            FrontierRegistration {
                authority,
                mode: ExecutionMode::CpuPcr,
                original_deadline: deadline,
                root_budget: budget,
                goal: SearchGoal {
                    depth_plies: Some(goal_plies),
                    mate_moves: None,
                },
                nodes: Some(root_nodes),
                selective_policy: ContractDigest([11; 32]),
                root_rules: Arc::clone(&port.rules),
                root_moves: Arc::clone(&domain),
                root_state: port.root_state_id(),
                model_namespace: Arc::new(namespace),
                native_ledger: None,
            },
            FrontierLimits {
                maximum_goal_plies: NonZeroU16::new(16).unwrap(),
                maximum_frames: 16,
                maximum_paths: 64,
                maximum_mate_nodes: 64,
                maximum_native_receipts: 16,
            },
            AdaptiveFrontierStartup {
                mode: ExecutionMode::CpuPcr,
                root_budget: budget,
                root_moves: &domain,
                native_ledger: None,
            },
            &source,
            clock.domain(),
            clock.now().unwrap(),
        )
        .unwrap();
        let original_cleanup_deadline = origin + lifetime + Duration::from_secs(2);
        assert!(
            port.register_profiled_external_cleanup_deadline(&source, origin)
                .is_err()
        );
        port.register_profiled_external_cleanup_deadline(&source, original_cleanup_deadline)
            .unwrap();
        assert!(
            port.register_profiled_external_cleanup_deadline(
                &source,
                original_cleanup_deadline + Duration::from_secs(1),
            )
            .is_err()
        );
        assert_eq!(
            port.profiled_external.original_cleanup_deadline,
            Some(original_cleanup_deadline)
        );
        port.prepare_rovent_root_rules(clock.domain(), clock.now().unwrap())
            .unwrap();
        Self {
            port,
            worker: Some(worker),
            source,
            config,
            clock,
            elf,
        }
    }

    fn work(&mut self, sequence: u64) -> Work {
        let first = BoardMove::from_uci("e2e4").unwrap();
        let before = self.port.rovent_frontier().unwrap().snapshot().work;
        // An attached original frontier must still reject an uncharged hot
        // insertion. This fixture uses the same charged Rules replay/adoption
        // path as the product instead of manufacturing an edge behind it.
        assert!(matches!(
            self.port.connect_hot(self.port.root, first),
            Err(PalsError::Rovent(
                crate::pals::rovent_frontier::FrontierError::Invalid(
                    "uncharged hot connection under original frontier"
                )
            ))
        ));
        let already_checked = self
            .port
            .existing_checked_frontier_line(&[first])
            .unwrap()
            .is_some();
        let node = self.port.replay_node(&[first]).unwrap();
        let charged = self.port.rovent_frontier().unwrap().snapshot().work;
        assert_eq!(
            charged.checked_transitions,
            before.checked_transitions + u64::from(!already_checked)
        );
        assert_eq!(charged.native_nodes, before.native_nodes);
        assert_eq!(self.port.replay_node(&[first]).unwrap(), node);
        let reused = self.port.rovent_frontier().unwrap().snapshot().work;
        assert_eq!(reused.checked_transitions, charged.checked_transitions);
        assert_eq!(reused.native_nodes, charged.native_nodes);
        let input = self
            .port
            .input(
                node,
                RefinementStage::CpuValidation,
                Some(first),
                vec![first],
                vec![first],
                None,
                None,
                None,
                false,
                None,
                None,
            )
            .unwrap();
        Work {
            frame: FrameId {
                authority: self.port.authority,
                sequence,
            },
            original_first_move: rz_contracts::pals::Move16::encode(
                rz_contracts::Move::try_from(first).unwrap(),
            ),
            original_deadline: self.port.deadline,
            budget: FrameBudget {
                original_cpu_credit: self.port.limits.max_cpu_nodes,
                role_submissions_remaining: self.port.engine.config.max_role_calls,
                original_store_byte_limit: 8 * 1024 * 1024,
            },
            input,
        }
    }

    fn dispatch(
        &mut self,
        work: &Work,
        sequence: u64,
    ) -> (
        AdaptiveProfiledExternalAdmission,
        Arc<ProfiledExternalCompletion>,
    ) {
        let requested = ExternalGoConstraint::Nodes { reported_nodes: 16 };
        let prepared = self
            .port
            .prepare_profiled_external(
                work,
                &self.config,
                &self.source,
                ReciprocalQuestionId {
                    authority: self.port.authority,
                    sequence,
                },
                rz_contracts::ExecutionId::new(self.port.authority.epoch, sequence),
                requested,
                Duration::from_millis(100),
                CompletionSignal::default(),
                self.clock.domain(),
                self.clock.now().unwrap(),
            )
            .unwrap();
        let AdaptiveProfiledExternalPreparation::Ready(job) = prepared else {
            panic!("unrestricted requested-nodes fixture must be supported")
        };
        assert_eq!(job.constraint, requested);
        let worker = self.worker.as_mut().unwrap();
        let dispatch = worker.submit(job).unwrap();
        let admitted = self
            .port
            .profiled_external_admitted(
                work,
                &self.source,
                &dispatch,
                self.clock.domain(),
                self.clock.now().unwrap(),
            )
            .unwrap();
        assert_eq!(admitted.binding.frame(), work.frame);
        assert_eq!(
            admitted.binding.question().question_digest,
            work.input.question_digest
        );
        let completion = Arc::new(poll(worker, || worker.take_completion()));
        assert!(completion.matches_submission(&dispatch));
        assert_eq!(completion.go_issue(), ExternalGoIssue::Issued);
        let report = &completion.result().as_ref().unwrap().report;
        assert_eq!(report.requested, requested);
        assert_eq!(report.issued, requested);
        assert_eq!(report.raw.requested_depth, 0);
        assert_eq!(report.strict_native_nodes, None);
        (admitted, completion)
    }

    fn quiesce(&self) -> ProfiledExternalCurrentRootFence {
        let worker = self.worker.as_ref().unwrap();
        worker
            .request_root_quiesce(self.port.authority, Instant::now() + Duration::from_secs(2))
            .unwrap();
        let fence = poll(worker, || worker.root_quiesce_result()).unwrap();
        assert!(fence.permits_reuse());
        fence
    }
}

fn poll<T>(worker: &ProfiledExternalWorker, mut read: impl FnMut() -> Option<T>) -> T {
    let until = Instant::now() + Duration::from_secs(5);
    loop {
        let version = worker.signal().version();
        if let Some(value) = read() {
            return value;
        }
        assert!(
            Instant::now() < until,
            "authored worker did not finish bounded stage"
        );
        worker
            .signal()
            .wait_changed(version, Duration::from_millis(20));
    }
}

impl Drop for AuthoredWorkerFixture {
    fn drop(&mut self) {
        if let Some(mut worker) = self.worker.take() {
            let until = Instant::now() + Duration::from_secs(2);
            worker.request_shutdown(until);
            loop {
                let version = worker.signal().version();
                if worker.try_join().is_some() {
                    break;
                }
                if Instant::now() >= until {
                    break;
                }
                worker
                    .signal()
                    .wait_changed(version, Duration::from_millis(10));
            }
        }
        // The ELF's Drop follows this bounded shutdown/join; no shared cache,
        // model, report, source-tree artifact or foreign temp root is removed.
        let _ = &self.elf;
    }
}

#[test]
fn retained_question_capacity_keeps_every_slot_until_root_retirement() {
    // Scalar values exercise the production allocation guard without forging
    // 64 worker submissions, source capabilities, Store tasks or view leases.
    let mut retained = HashMap::<ContractDigest, u8>::new();
    for sequence in 0..PROFILED_QUESTION_CAPACITY {
        reserve_profiled_question_slot(&mut retained).unwrap();
        assert_eq!(retained.len(), sequence);
        let mut key = [0; 32];
        key[..8].copy_from_slice(&(sequence as u64).to_le_bytes());
        retained.insert(ContractDigest(key), sequence as u8);
    }
    let original = retained.clone();
    for _ in 0..2 {
        assert!(matches!(
            reserve_profiled_question_slot(&mut retained),
            Err(PalsError::Capacity)
        ));
        assert_eq!(retained, original);
    }
    assert_eq!(PROFILED_QUESTION_CAPACITY, 64);
}

#[test]
fn foreign_frontier_lineage_is_rejected_before_external_preparation() {
    let _serial = EXTERNAL_FIXTURE_LOCK.lock().unwrap();
    let mut fixture = AuthoredWorkerFixture::new(Duration::from_secs(20), false);
    let mut work = fixture.work(1);
    // Keep the actual board, move and view while changing the context after
    // its producer seal was made. The ordinary work check alone accepts it.
    let context = &mut Arc::make_mut(&mut work.input.question).context;
    context.public_revision = context.public_revision.wrapping_add(1);
    assert!(fixture.port.check_work(&work).is_ok());
    assert!(AdaptivePortQuestion::validate_frontier_source(
        &work, &fixture.port.engine.semantic.stores,
    ).is_err());
    let before = fixture
        .port
        .rovent_frontier()
        .unwrap()
        .external_work()
        .unwrap();
    let entries_before = fixture.port.profiled_external.entries.len();
    let result = fixture.port.prepare_profiled_external(
        &work,
        &fixture.config,
        &fixture.source,
        ReciprocalQuestionId {
            authority: fixture.port.authority,
            sequence: 1,
        },
        rz_contracts::ExecutionId::new(fixture.port.authority.epoch, 1),
        ExternalGoConstraint::Nodes { reported_nodes: 16 },
        Duration::from_millis(100),
        CompletionSignal::default(),
        fixture.clock.domain(),
        fixture.clock.now().unwrap(),
    );
    assert!(result.is_err());
    assert_eq!(fixture.port.profiled_external.entries.len(), entries_before);
    assert_eq!(
        fixture
            .port
            .rovent_frontier()
            .unwrap()
            .external_work()
            .unwrap(),
        before
    );
}

#[test]
fn post_submit_lineage_failure_still_allows_real_worker_terminal_and_fence() {
    let _serial = EXTERNAL_FIXTURE_LOCK.lock().unwrap();
    let mut fixture = AuthoredWorkerFixture::new(Duration::from_secs(20), false);
    let mut work = fixture.work(1);
    let constraint = ExternalGoConstraint::Nodes { reported_nodes: 16 };
    let prepared = fixture
        .port
        .prepare_profiled_external(
            &work,
            &fixture.config,
            &fixture.source,
            ReciprocalQuestionId {
                authority: fixture.port.authority,
                sequence: 1,
            },
            rz_contracts::ExecutionId::new(fixture.port.authority.epoch, 1),
            constraint,
            Duration::from_millis(100),
            CompletionSignal::default(),
            fixture.clock.domain(),
            fixture.clock.now().unwrap(),
        )
        .unwrap();
    let AdaptiveProfiledExternalPreparation::Ready(job) = prepared else {
        panic!("fixture's research query must be supported")
    };
    let dispatch = fixture.worker.as_mut().unwrap().submit(job).unwrap();
    // Fault injection after physical admission: it must not be accepted as
    // the original sealed question, even though the worker now owns the go.
    let context = &mut Arc::make_mut(&mut work.input.question).context;
    context.public_revision = context.public_revision.wrapping_add(1);
    assert!(
        fixture
            .port
            .profiled_external_admitted(
                &work,
                &fixture.source,
                &dispatch,
                fixture.clock.domain(),
                fixture.clock.now().unwrap(),
            )
            .is_err()
    );
    let worker = fixture.worker.as_mut().unwrap();
    let terminal = poll(worker, || worker.take_completion());
    assert!(terminal.matches_submission(&dispatch));
    assert!(terminal.controls().terminal);
    assert_eq!(terminal.go_issue(), ExternalGoIssue::Issued);
    let fence = fixture.quiesce();
    assert!(fence.permits_reuse());
    assert!(fence.source().same_source(&fixture.source));
    assert_eq!(fence.process_identity(), fixture.source.process_identity());
    assert_eq!(fence.authority(), fixture.source.authority());
    let observed = fixture
        .port
        .rovent_frontier()
        .unwrap()
        .external_work()
        .unwrap();
    assert_eq!(observed.dispatched_questions, 0);
    assert_eq!(observed.returned_terminal_packets, 0);
}

#[test]
fn authored_worker_nodes_only_keeps_raw_constraint_and_publication_once() {
    let _serial = EXTERNAL_FIXTURE_LOCK.lock().unwrap();
    let mut fixture = AuthoredWorkerFixture::new(Duration::from_secs(20), false);
    let work = fixture.work(1);
    let (admitted, packet) = fixture.dispatch(&work, 1);
    let before_projection = fixture.port.rovent_frontier().unwrap().snapshot().work;
    let raw_only = fixture
        .port
        .commit_profiled_external(&work, &fixture.source, admitted.reservation, &packet, false)
        .unwrap();
    assert!(!raw_only.consumed);
    assert_eq!(work.input.stage, RefinementStage::CpuValidation);
    let retained_raw = fixture
        .port
        .engine
        .semantic
        .stores
        .observations
        .get(raw_only.observation)
        .unwrap();
    let report = retained_raw.external_report.as_ref().unwrap();
    assert_eq!(report.pv, [BoardMove::from_uci("e7e5").unwrap()]);
    assert_eq!(
        fixture
            .port
            .engine
            .semantic
            .stores
            .lines
            .moves(retained_raw.cpu_pv.expect("live validation has a paid PV"))
            .unwrap(),
        report.pv
    );
    let after_projection = fixture.port.rovent_frontier().unwrap().snapshot().work;
    assert_eq!(
        after_projection.checked_transitions,
        before_projection.checked_transitions + 1
    );
    assert_eq!(
        after_projection.native_nodes,
        before_projection.native_nodes
    );
    let execution = fixture.port.profiled_external.entries[&work.input.question_digest].execution;
    let task = fixture
        .port
        .engine
        .semantic
        .stores
        .tasks
        .get(execution)
        .unwrap();
    assert_eq!(task.status, TaskStatus::Failed);
    assert_eq!(task.key.requested_depth, 0);
    assert_eq!(task.key.node_budget, 16);
    assert!(matches!(
        task.key.question,
        TaskQuestion::ProfiledExternal {
            requested: ProfiledExternalConstraint::Nodes { reported_nodes: 16 }
        }
    ));

    // Consume the original retained packet and publish exactly once under the
    // immutable configured bound. Record exhaustion closes its own root and
    // is checked separately; it is not recoverable by changing this bound.
    let committed = fixture
        .port
        .commit_profiled_external(&work, &fixture.source, admitted.reservation, &packet, true)
        .unwrap();
    assert!(committed.consumed);
    assert_eq!(committed.observation, raw_only.observation);
    assert!(fixture.port.profiled_external.entries[&work.input.question_digest].feedback_published);
    let matching_records = || {
        fixture
            .port
            .engine
            .semantic
            .records
            .iter()
            .filter(|record| {
                record.kind == RecordKind::CpuVerification
                    && record.cpu_observation == Some(committed.observation)
            })
            .count()
    };
    assert_eq!(matching_records(), 1);
    let record = fixture
        .port
        .engine
        .semantic
        .records
        .iter()
        .find(|record| record.cpu_observation == Some(committed.observation))
        .unwrap();
    assert_eq!(record.completed_depth, 0);
    assert!(record.value.is_none() && record.score_scope.is_none());
    let before = fixture.port.rovent_frontier().unwrap().snapshot().work;
    fixture
        .port
        .commit_profiled_external(&work, &fixture.source, admitted.reservation, &packet, true)
        .unwrap();
    assert_eq!(
        fixture
            .port
            .engine
            .semantic
            .records
            .iter()
            .filter(|record| { record.cpu_observation == Some(committed.observation) })
            .count(),
        1
    );
    assert_eq!(fixture.port.counters.external_checker_tasks, 1);
    assert_eq!(fixture.port.counters.external_checker_reports, 1);
    assert_eq!(fixture.port.counters.consumed_external_checker_tasks, 1);
    assert_eq!(fixture.port.counters.cpu_nodes, 0);
    assert_eq!(fixture.port.counters.completed_cpu_tasks, 0);
    let after = fixture.port.rovent_frontier().unwrap().snapshot().work;
    assert_eq!(after.checked_transitions, before.checked_transitions);
    assert_eq!(after.native_nodes, 0);
    let raw = fixture
        .port
        .engine
        .semantic
        .stores
        .observations
        .get(committed.observation)
        .unwrap();
    assert!(matches!(
        raw.scope,
        EvidenceScope::ProfiledExternalUci {
            requested: ProfiledExternalConstraint::Nodes { reported_nodes: 16 },
            issued: ProfiledExternalConstraint::Nodes { reported_nodes: 16 },
            ..
        }
    ));

    // Continuation is derived from this exact raw consumer, not fake depth=1,
    // native credit, an own CPU score or a different question's latest slot.
    let step = fixture
        .port
        .after_profiled_external(&work, &committed)
        .unwrap();
    let materialized = fixture.port.materialize_step(step).unwrap();
    let FrameTransition::Continue(next) = materialized.transition else {
        panic!("expected same-stage Fresh followup")
    };
    assert_eq!(next.stage, RefinementStage::ProposalValueFresh);
    assert_eq!(
        next.question.context.prefix,
        work.input.question.context.prefix
    );
    assert_eq!(
        next.question.context.game_generation,
        work.frame.authority.game.0
    );

    let other_work = fixture.work(2);
    let (other_admission, other_packet) = fixture.dispatch(&other_work, 2);
    assert!(!Arc::ptr_eq(&packet, &other_packet));
    assert!(
        fixture
            .port
            .commit_profiled_external(
                &work,
                &fixture.source,
                admitted.reservation,
                &other_packet,
                true
            )
            .is_err()
    );
    assert_eq!(fixture.port.counters.external_checker_reports, 1);
    fixture
        .port
        .commit_profiled_external(
            &other_work,
            &fixture.source,
            other_admission.reservation,
            &other_packet,
            false,
        )
        .unwrap();
    let fence = fixture.quiesce();
    let retirement = fixture
        .port
        .release_profiled_external_root(&fixture.source, &fence)
        .unwrap();
    assert_eq!(retirement.retained_questions(), 2);
    assert_eq!(
        retirement.original_cleanup_deadline(),
        Some(fixture.port.limits.deadline + Duration::from_secs(2))
    );
}

#[test]
fn authored_worker_record_capacity_keeps_consumed_raw_and_error_through_drain() {
    let _serial = EXTERNAL_FIXTURE_LOCK.lock().unwrap();
    // Register the bound before any question. No transient config reduction,
    // synthetic record, marker reset or post-failure bound increase is used.
    let mut fixture = AuthoredWorkerFixture::with_root_limits_and_record_limit(
        Duration::from_secs(20),
        false,
        2,
        128,
        false,
        4,
    );
    for sequence in 1..=4 {
        let work = fixture.work(sequence);
        let (admitted, packet) = fixture.dispatch(&work, sequence);
        let committed = fixture
            .port
            .commit_profiled_external(&work, &fixture.source, admitted.reservation, &packet, true)
            .unwrap();
        assert!(committed.consumed);
        assert_eq!(
            fixture.port.engine.semantic.records.len(),
            sequence as usize
        );
    }
    let records_before = fixture
        .port
        .engine
        .semantic
        .records
        .iter()
        .map(|record| (record.revision, record.line.clone(), record.cpu_observation))
        .collect::<Vec<_>>();

    let work = fixture.work(5);
    let (admitted, packet) = fixture.dispatch(&work, 5);
    let raw_only = fixture
        .port
        .commit_profiled_external(&work, &fixture.source, admitted.reservation, &packet, false)
        .unwrap();
    assert!(!raw_only.consumed);
    let execution = fixture.port.profiled_external.entries[&work.input.question_digest].execution;
    let reported_before = fixture.port.profiled_external_work().unwrap();
    let checked_before = fixture.port.rovent_frontier().unwrap().snapshot().work;
    assert!(matches!(
        fixture.port.commit_profiled_external(
            &work,
            &fixture.source,
            admitted.reservation,
            &packet,
            true,
        ),
        Err(PalsError::Rovent(
            crate::pals::rovent_frontier::FrontierError::PublicRecordCapacity
        ))
    ));
    let pending = &fixture.port.profiled_external.entries[&work.input.question_digest];
    // The exact raw head was consumed before the later record publication
    // refused admission. Its real use and original packet cannot be erased.
    assert_eq!(pending.published, Some(raw_only.observation));
    assert_eq!(pending.consumed, Some(raw_only.observation));
    assert!(pending.terminal && pending.accounted && !pending.feedback_published);
    assert!(Arc::ptr_eq(pending.completion.as_ref().unwrap(), &packet));
    assert!(matches!(
        pending.publication_error,
        Some(PalsError::Rovent(
            crate::pals::rovent_frontier::FrontierError::PublicRecordCapacity
        ))
    ));
    assert!(
        fixture
            .port
            .rovent_frontier()
            .unwrap()
            .public_record_capacity_refused()
    );
    assert_eq!(
        fixture.port.rovent_frontier().unwrap().snapshot().goal,
        crate::pals::rovent_frontier::FrontierGoalStatus::Stopped(
            crate::pals::rovent_frontier::FrontierStop::Capacity
        ),
    );
    assert!(
        fixture
            .port
            .rovent_frontier()
            .unwrap()
            .snapshot()
            .horizon
            .completed_depth
            .is_none()
    );

    // This is the actual Draining route: retain/re-observe the same raw packet
    // with consume_current=false. It does not authorize a Fresh continuation,
    // clear the earlier publication error, or consume the original head twice.
    let drained = fixture
        .port
        .commit_profiled_external(&work, &fixture.source, admitted.reservation, &packet, false)
        .unwrap();
    assert_eq!(drained.observation, raw_only.observation);
    assert!(!drained.consumed);
    assert_eq!(fixture.port.counters.external_checker_tasks, 5);
    assert_eq!(fixture.port.counters.external_checker_reports, 5);
    assert_eq!(fixture.port.counters.consumed_external_checker_tasks, 5);
    assert_eq!(fixture.port.counters.cpu_nodes, 0);
    assert_eq!(fixture.port.counters.completed_cpu_tasks, 0);
    assert_eq!(
        fixture.port.profiled_external_work().unwrap(),
        reported_before
    );
    let checked_after = fixture.port.rovent_frontier().unwrap().snapshot().work;
    assert_eq!(
        checked_after.checked_transitions,
        checked_before.checked_transitions
    );
    assert_eq!(checked_after.native_nodes, 0);
    assert!(
        fixture
            .port
            .rovent_frontier()
            .unwrap()
            .public_record_capacity_refused()
    );
    assert_eq!(
        fixture
            .port
            .engine
            .semantic
            .records
            .iter()
            .map(|record| { (record.revision, record.line.clone(), record.cpu_observation) })
            .collect::<Vec<_>>(),
        records_before,
    );
    let task = fixture
        .port
        .engine
        .semantic
        .stores
        .tasks
        .get(execution)
        .unwrap();
    assert_eq!(task.status, TaskStatus::Failed);
    assert_eq!(task.key.requested_depth, 0);
    assert_eq!(task.key.node_budget, 16);
    assert!(matches!(
        task.key.question,
        TaskQuestion::ProfiledExternal {
            requested: ProfiledExternalConstraint::Nodes { reported_nodes: 16 }
        }
    ));
    let fence = fixture.quiesce();
    let retirement = fixture
        .port
        .release_profiled_external_root(&fixture.source, &fence)
        .unwrap();
    assert_eq!(retirement.retained_questions(), 5);
    let original = retirement
        .packets()
        .find(|held| held.execution() == execution)
        .unwrap();
    assert!(Arc::ptr_eq(original.completion().unwrap(), &packet));
    assert_eq!(original.observation(), Some(raw_only.observation));
    assert_eq!(
        original.actual_consumed_observation(),
        Some(raw_only.observation)
    );
    assert!(original.search_consumed() && original.terminal() && original.accounted());
    assert!(!original.feedback_published());
    assert!(matches!(
        original.publication_error(),
        Some(PalsError::Rovent(
            crate::pals::rovent_frontier::FrontierError::PublicRecordCapacity
        ))
    ));
    assert_eq!(fixture.port.engine.config.max_records, 4);
}

#[test]
fn authored_worker_complete_pv_beyond_original_allowance_retains_packet_without_publication() {
    let _serial = EXTERNAL_FIXTURE_LOCK.lock().unwrap();
    for (goal, root_nodes, paid_before_failure) in [(2, 128, 0), (3, 3, 1)] {
        let mut fixture = AuthoredWorkerFixture::with_root_limits(
            Duration::from_secs(20),
            false,
            goal,
            root_nodes,
            true,
        );
        let work = fixture.work(1);
        let (admitted, packet) = fixture.dispatch(&work, 1);
        assert_eq!(work.input.stage, RefinementStage::CpuValidation);
        assert_eq!(
            packet.result().as_ref().unwrap().report.raw.pv,
            [
                BoardMove::from_uci("e7e5").unwrap(),
                BoardMove::from_uci("g1f3").unwrap()
            ]
        );
        let before = fixture.port.rovent_frontier().unwrap().snapshot().work;
        let failure = fixture.port.commit_profiled_external(
            &work,
            &fixture.source,
            admitted.reservation,
            &packet,
            false,
        );
        if goal == 2 {
            assert!(matches!(
                failure,
                Err(PalsError::Rovent(
                    crate::pals::rovent_frontier::FrontierError::DepthHorizon
                ))
            ));
        } else {
            assert!(matches!(
                failure,
                Err(PalsError::Rovent(
                    crate::pals::rovent_frontier::FrontierError::NodeBudget
                ))
            ));
        }
        let after = fixture.port.rovent_frontier().unwrap().snapshot().work;
        // Horizon rejection happens before replay. Quota failure keeps the
        // paid first transition; neither case publishes a partial observation.
        assert_eq!(
            after.checked_transitions,
            before.checked_transitions + paid_before_failure
        );
        assert_eq!(after.native_nodes, before.native_nodes);
        let original = &fixture.port.profiled_external.entries[&work.input.question_digest];
        assert!(Arc::ptr_eq(original.completion.as_ref().unwrap(), &packet));
        assert!(original.prepared_observation.is_none());
        assert!(original.published.is_none() && original.consumed.is_none());
        assert!(original.publication_error.is_some());
        assert_eq!(fixture.port.counters.consumed_external_checker_tasks, 0);
        let fence = fixture.quiesce();
        let retirement = fixture
            .port
            .release_profiled_external_root(&fixture.source, &fence)
            .unwrap();
        assert_eq!(retirement.retained_questions(), 1);
        let retained = retirement.packets().next().unwrap();
        assert!(Arc::ptr_eq(retained.completion().unwrap(), &packet));
        assert!(retained.publication_error().is_some());
        assert!(!retained.search_consumed() && retained.observation().is_none());
    }
}

#[test]
fn authored_worker_expired_revoked_archive_packet_retires_before_append() {
    let _serial = EXTERNAL_FIXTURE_LOCK.lock().unwrap();
    let mut fixture = AuthoredWorkerFixture::new(Duration::from_secs(3), true);
    let work = fixture.work(1);
    let (admitted, packet) = fixture.dispatch(&work, 1);
    let fence = fixture.quiesce();
    assert!(fixture.source.validate_current().is_err());
    // Real expiry of the original registered deadline, not a replaced public
    // deadline field or caller-provided completion boolean.
    while Instant::now() < fixture.port.limits.deadline {
        std::thread::sleep(POLL);
    }
    assert!(
        fixture
            .port
            .prepare_profiled_external_publication(
                &work,
                &fixture.source,
                admitted.reservation,
                &packet
            )
            .is_err()
    );
    let original = &fixture.port.profiled_external.entries[&work.input.question_digest];
    let execution = original.execution;
    assert!(original.terminal);
    assert_eq!(
        fixture
            .port
            .engine
            .semantic
            .stores
            .tasks
            .get(execution)
            .unwrap()
            .status,
        TaskStatus::Failed
    );
    assert!(Arc::ptr_eq(original.completion.as_ref().unwrap(), &packet));
    assert!(original.publication_error.is_some());
    assert!(original.published.is_none() && original.consumed.is_none());
    assert_eq!(
        fixture
            .port
            .profiled_external_work()
            .unwrap()
            .outstanding_questions,
        0
    );
    assert_eq!(fixture.port.counters.consumed_external_checker_tasks, 0);
    let reports = fixture.port.counters.external_checker_reports;
    assert!(
        fixture
            .port
            .commit_profiled_external(&work, &fixture.source, admitted.reservation, &packet, true)
            .is_err()
    );
    assert_eq!(fixture.port.counters.external_checker_reports, reports);
    let retirement = fixture
        .port
        .release_profiled_external_root(&fixture.source, &fence)
        .unwrap();
    retirement
        .validate_owner(&fixture.port.engine.semantic.stores)
        .unwrap();
    assert!(
        retirement
            .protected_pins()
            .unwrap()
            .executions
            .contains(&execution)
    );
    assert_eq!(retirement.retained_questions(), 1);
    let retained = retirement.packets().next().unwrap();
    assert!(Arc::ptr_eq(retained.completion().unwrap(), &packet));
    assert!(Arc::ptr_eq(
        &retained.work().input.question,
        &work.input.question
    ));
    assert!(Arc::ptr_eq(&retained.work().input.view, &work.input.view));
    assert!(retained.publication_error().is_some());
    assert!(!retained.search_consumed());
    assert!(retained.observation().is_none());
    assert!(fixture.port.profiled_external.entries.is_empty());
}
