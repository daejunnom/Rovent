//! Same-owner NoGame preparation for the existing child supervisor.
//! This does not authenticate role listeners, spawn a child, or establish
//! physical completion. The executable entry remains guarded.

use super::{
    PalsDirectChildObservationV6, PalsProxyBinaryPinV6, PalsProxyLaunchV6,
    cli_v6::{parse_arguments, sensitive, validate_descriptor},
    finish::PalsSessionReleaseClaimV6,
    proxy_entry_no_game_smoke_v1::{
        ProxyEntryNoGameAdmittedBundleV1, ProxyEntryNoGameSmokeReceiptV1,
    },
    proxy_entry_smoke_cancel_v1::NoGameCancelReceiverV1,
    proxy_entry_smoke_prepare_v1::{OriginalProxyStdioV1, PreparedNoGameGrantOwnersV1},
    relay,
    runner_v6::{
        PalsProxySupervisorV6, ProxyStageV6, ProxySupervisorObservationV6,
        SessionReleaseObservationV6,
    },
    smoke_expected_argv_v1::ParsedSmokeExpectedArgvV1,
};
use std::{
    fs, io,
    os::unix::fs::{MetadataExt, PermissionsExt},
    os::unix::net::UnixStream,
    sync::{Arc, atomic::Ordering},
    time::Instant,
};

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Retains the original NoGame lock and grant streams beside the supervisor.
/// The caller must retain this owner across any unknown physical completion.
#[must_use = "retain the original lock, grants, and child supervisor"]
pub(super) struct NoGamePreparedSupervisorV1 {
    pub argv: ParsedSmokeExpectedArgvV1,
    pub bundle: ProxyEntryNoGameAdmittedBundleV1,
    pub receipt: ProxyEntryNoGameSmokeReceiptV1,
    supervisor: PalsProxySupervisorV6,
}

impl NoGamePreparedSupervisorV1 {
    /// The caller must separately prove this is the original role listener.
    /// Failure returns the same socket; peer identity comes from the held run
    /// grant and the actual cancel stream, never the supplied wire fields.
    pub(super) fn attach_original_cancel(
        &mut self,
        stream: UnixStream,
    ) -> Result<(), (io::Error, UnixStream)> {
        let receiver = NoGameCancelReceiverV1::from_original_bundle(stream, &self.bundle)?;
        self.supervisor
            .attach_no_game_cancel(receiver)
            .map_err(|receiver| {
                (
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "NoGame cancel supervisor rejected owner",
                    ),
                    receiver.into_stream(),
                )
            })
    }

    /// Preparation is not execution authority; the caller must independently
    /// admit the listener and preserve this owner on every outcome.
    pub(super) fn start_once(&mut self) -> io::Result<ProxyStageV6> {
        let window = &self.bundle.window;
        if window.cancelled.load(Ordering::Acquire)
            || window.cleanup_cancelled.load(Ordering::Acquire)
        {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "NoGame owner cancelled",
            ));
        }
        if Instant::now() >= window.startup_until || Instant::now() >= window.run_until {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "NoGame original window expired",
            ));
        }
        self.bundle.lock_admission.verify_held()?;
        Ok(self.supervisor.start_once())
    }

    pub(super) fn poll(&mut self) -> io::Result<ProxySupervisorObservationV6> {
        self.supervisor.poll()
    }

    /// This may wait until the original cleanup deadline and records only the
    /// original child's session claim, not a GPU fence.
    pub(super) fn observe_session_release_once(&mut self) -> SessionReleaseObservationV6 {
        self.supervisor.observe_session_release_once()
    }

    pub(super) fn session_release_claim(&self) -> Option<&PalsSessionReleaseClaimV6> {
        self.supervisor.session_release.as_ref()
    }

    /// Reaping the original child is not evidence of GPU physical completion.
    pub(super) fn observe_child_exit(
        &mut self,
    ) -> io::Result<Option<PalsDirectChildObservationV6>> {
        self.supervisor.observe_child_exit()
    }

    /// A false result leaves the same relay owner and its unjoined threads.
    pub(super) fn join_relay_until(&mut self, until: Instant) -> bool {
        self.supervisor.join_relay_until(until)
    }
}

pub(super) fn startup_arguments(arguments: &[String]) -> Vec<String> {
    const SEPARATE: &[&str] = &[
        "--pals-smoke-policy",
        "--pals-smoke-owner-id",
        "--pals-smoke-owner-epoch",
        "--pals-smoke-role",
        "--pals-smoke-generation",
        "--pals-smoke-lock-sha256",
        "--pals-smoke-plan-sha256",
        "--pals-smoke-proxy-sha256",
        "--pals-arena-grant-socket",
        "--pals-arena-cancel-socket",
    ];
    arguments
        .iter()
        .filter(|argument| {
            argument.as_str() != "--pals-proxy-entry-functional-smoke=true"
                && !argument
                    .split_once('=')
                    .is_some_and(|(name, _)| SEPARATE.contains(&name))
        })
        .cloned()
        .collect()
}

/// Parse and pin before transferring any descriptor. Every error returns the
/// original grant, lock, argv and stdio owners; no child or model is started.
pub(super) fn prepare_supervisor(
    prepared: PreparedNoGameGrantOwnersV1,
    max_line: usize,
) -> Result<NoGamePreparedSupervisorV1, (io::Error, PreparedNoGameGrantOwnersV1)> {
    let result = (|| -> io::Result<PalsProxyLaunchV6> {
        let args = parse_arguments(&startup_arguments(prepared.argv.original_arguments()))?;
        let bundle = &prepared.bundle;
        let admission = &bundle.lock_admission;
        let window = &bundle.window;
        admission.verify_held()?;
        let (root, lock_path, device, inode) = admission.local_paths_and_root_identity()?;
        if args.startup.lock_path != lock_path
            || args.startup.lock_sha256 != admission.lock().canonical_sha256
            || args.startup.startup_ms != admission.lock().config.native_options.startup_ms
            || sensitive(&root)
            || sensitive(&args.child_path)
            || fs::canonicalize(&root)? != root
            || Instant::now() >= window.startup_until
        {
            return Err(invalid("NoGame original startup declaration differs"));
        }
        let meta = fs::symlink_metadata(&root)?;
        if !meta.is_dir()
            || meta.uid() != nix::unistd::geteuid().as_raw()
            || meta.permissions().mode() & 0o077 != 0
            || meta.dev() != device
            || meta.ino() != inode
        {
            return Err(invalid("NoGame original private root differs"));
        }
        relay::fifo(&prepared.stdio.incoming)?;
        relay::fifo(&prepared.stdio.outgoing)?;
        validate_descriptor(&prepared.stdio.incoming, true, false)?;
        validate_descriptor(&prepared.stdio.outgoing, false, true)?;
        validate_descriptor(&prepared.stdio.diagnostics, false, false)?;
        relay::validate_line_bound(max_line)?;
        let binary = Arc::new(PalsProxyBinaryPinV6::open(
            root,
            args.child_path,
            &admission.lock().config.endpoint.binary,
            window.startup_until,
            &window.cancelled,
        )?);
        let binary_root = binary.cwd.metadata()?;
        if binary_root.dev() != device || binary_root.ino() != inode {
            return Err(invalid("NoGame child pin root differs"));
        }
        admission.verify_held()?;
        PalsProxyLaunchV6::new(
            binary,
            lock_path,
            args.startup
                .parent_socket
                .ok_or_else(|| invalid("NoGame child socket missing"))?,
            admission.lock().clone(),
            window.startup_until,
            Arc::clone(&window.cancelled),
        )
    })();
    let launch = match result {
        Ok(launch) => launch,
        Err(error) => return Err((error, prepared)),
    };
    let PreparedNoGameGrantOwnersV1 {
        argv,
        stdio,
        bundle,
        receipt,
    } = prepared;
    let OriginalProxyStdioV1 {
        incoming,
        outgoing,
        diagnostics,
    } = stdio;
    let window = &bundle.window;
    let supervisor = PalsProxySupervisorV6::new(
        launch,
        incoming,
        outgoing,
        diagnostics,
        max_line,
        window.bound_reference,
        window.run_until,
        window.cleanup_until,
        Arc::clone(&window.cancelled),
        Arc::clone(&window.cleanup_cancelled),
    );
    Ok(NoGamePreparedSupervisorV1 {
        argv,
        bundle,
        receipt,
        supervisor,
    })
}
