//! Initial local admission for a NoGame functional smoke.
//!
//! This preserves the incoming declaration and actual stdio/lock owners.
//! It does not authenticate role listeners, obtain grants, execute native
//! code, or establish physical completion.

use super::{
    cli_v6::parse_arguments,
    proxy_entry_smoke_lock_v1::{OriginalSmokeLockAdmission, SmokeLockAdmissionFailure},
    proxy_entry_smoke_prepare_v1::{NoGameProxyPrepareOwnerV1, OriginalProxyStdioV1},
    proxy_entry_smoke_supervisor_v1::startup_arguments,
    proxy_entry_smoke_v1::ProxyEntryFunctionalSmokeV1,
    smoke_expected_argv_v1::ParsedSmokeExpectedArgvV1,
};
use std::{
    io,
    path::Path,
    sync::{Arc, atomic::AtomicBool},
    time::Instant,
};

/// The raw vector remains available even if the initial parser rejects it.
pub(super) enum AdmissionArgvV1 {
    Raw(Vec<String>),
    Parsed(ParsedSmokeExpectedArgvV1),
}

/// Partial lock admission includes its original root or pin, when acquired.
pub(super) enum AdmissionFailureCauseV1 {
    ExpectedArguments(io::Error),
    StartupArguments(io::Error),
    Lock(SmokeLockAdmissionFailure),
    Prepare(io::Error),
}

/// An error never silently discards an already acquired input owner.
/// The caller retains the two cancellation Arcs supplied by reference.
#[must_use = "retain or explicitly settle the original admission owners"]
pub(super) struct AdmissionFailureV1 {
    pub cause: AdmissionFailureCauseV1,
    pub argv: AdmissionArgvV1,
    pub entry: ProxyEntryFunctionalSmokeV1,
    pub stdio: OriginalProxyStdioV1,
    pub lock: Option<OriginalSmokeLockAdmission>,
}

/// Only prepares the original owner. The entry Instant must be the value
/// captured by the caller at process entry; this function does not attest it.
pub(super) fn admit_initial(
    arguments: Vec<String>,
    entry_at: Instant,
    private_cwd: &Path,
    stdio: OriginalProxyStdioV1,
    cancelled: &Arc<AtomicBool>,
    cleanup_cancelled: &Arc<AtomicBool>,
) -> Result<NoGameProxyPrepareOwnerV1, AdmissionFailureV1> {
    let entry = ProxyEntryFunctionalSmokeV1::from_local_entry(entry_at);
    let argv = match ParsedSmokeExpectedArgvV1::from_arguments(arguments) {
        Ok(argv) => argv,
        Err((error, arguments)) => {
            return Err(AdmissionFailureV1 {
                cause: AdmissionFailureCauseV1::ExpectedArguments(error),
                argv: AdmissionArgvV1::Raw(arguments),
                entry,
                stdio,
                lock: None,
            });
        }
    };
    // The NoGame parser retains the original vector. The legacy parser sees
    // only its own arguments; Neither parsed text nor a named path is proof
    // of original listener, process, or execution authority.
    let startup = match parse_arguments(&startup_arguments(argv.original_arguments())) {
        Ok(startup) => startup.startup,
        Err(error) => {
            return Err(AdmissionFailureV1 {
                cause: AdmissionFailureCauseV1::StartupArguments(error),
                argv: AdmissionArgvV1::Parsed(argv),
                entry,
                stdio,
                lock: None,
            });
        }
    };
    let lock = match OriginalSmokeLockAdmission::open(
        &startup.lock_path,
        &startup.lock_sha256,
        startup.startup_ms,
        private_cwd,
        entry_at,
        cancelled,
    ) {
        Ok(lock) => lock,
        Err(failure) => {
            return Err(AdmissionFailureV1 {
                cause: AdmissionFailureCauseV1::Lock(failure),
                argv: AdmissionArgvV1::Parsed(argv),
                entry,
                stdio,
                lock: None,
            });
        }
    };
    match NoGameProxyPrepareOwnerV1::new(argv, entry, lock, stdio, cancelled, cleanup_cancelled) {
        Ok(owner) => Ok(owner),
        Err((error, argv, entry, lock, stdio)) => Err(AdmissionFailureV1 {
            cause: AdmissionFailureCauseV1::Prepare(error),
            argv: AdmissionArgvV1::Parsed(argv),
            entry,
            stdio,
            lock: Some(lock),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::{
        fs::OpenOptions,
        os::fd::{AsRawFd, OwnedFd},
    };

    fn stdio() -> OriginalProxyStdioV1 {
        let open = || -> OwnedFd {
            OpenOptions::new()
                .read(true)
                .write(true)
                .open("/dev/null")
                .unwrap()
                .into()
        };
        OriginalProxyStdioV1 {
            incoming: open(),
            outgoing: open(),
            diagnostics: open(),
        }
    }

    fn fds(stdio: &OriginalProxyStdioV1) -> [i32; 3] {
        [
            stdio.incoming.as_raw_fd(),
            stdio.outgoing.as_raw_fd(),
            stdio.diagnostics.as_raw_fd(),
        ]
    }

    #[test]
    fn rejects_bad_expected_arguments_with_same_raw_argv_and_stdio_owners() {
        let argv = vec!["--pals-smoke-policy=invalid".to_owned()];
        let stdio = stdio();
        let before = fds(&stdio);
        let run = Arc::new(AtomicBool::new(false));
        let cleanup = Arc::new(AtomicBool::new(false));
        let failure = match admit_initial(
            argv.clone(),
            Instant::now(),
            Path::new("/"),
            stdio,
            &run,
            &cleanup,
        ) {
            Err(failure) => failure,
            Ok(_) => panic!("invalid expected fields accepted"),
        };
        assert!(matches!(
            failure.cause,
            AdmissionFailureCauseV1::ExpectedArguments(_)
        ));
        assert!(matches!(failure.argv, AdmissionArgvV1::Raw(value) if value == argv));
        assert_eq!(fds(&failure.stdio), before);
        assert!(failure.lock.is_none());
    }

    #[test]
    fn rejects_bad_startup_arguments_after_preserving_parsed_argv_and_stdio() {
        let digest = format!("{:x}", Sha256::digest(b"synthetic startup declaration"));
        let argv = vec![
            "--pals-smoke-policy=proxy-entry-v1".to_owned(),
            format!("--pals-smoke-owner-id={}", &digest[..32]),
            "--pals-smoke-owner-epoch=1".to_owned(),
            "--pals-smoke-role=baseline".to_owned(),
            "--pals-smoke-generation=1".to_owned(),
            format!("--pals-smoke-lock-sha256={digest}"),
            format!("--pals-smoke-plan-sha256={digest}"),
            format!("--pals-smoke-proxy-sha256={digest}"),
        ];
        let stdio = stdio();
        let before = fds(&stdio);
        let run = Arc::new(AtomicBool::new(false));
        let cleanup = Arc::new(AtomicBool::new(false));
        let failure = match admit_initial(
            argv.clone(),
            Instant::now(),
            Path::new("/"),
            stdio,
            &run,
            &cleanup,
        ) {
            Err(failure) => failure,
            Ok(_) => panic!("invalid startup fields accepted"),
        };
        assert!(matches!(
            failure.cause,
            AdmissionFailureCauseV1::StartupArguments(_)
        ));
        match failure.argv {
            AdmissionArgvV1::Parsed(parsed) => assert_eq!(parsed.original_arguments(), argv),
            AdmissionArgvV1::Raw(_) => panic!("expected NoGame fields were not parsed"),
        }
        assert_eq!(fds(&failure.stdio), before);
        assert!(failure.lock.is_none());
    }

    #[test]
    fn missing_lock_retains_opened_root_argv_and_stdio() {
        let root = std::env::current_dir().unwrap();
        let nonce = format!(
            "{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let lock_path = root.join("inputs").join(format!("missing-{nonce}.lock"));
        let digest = format!("{:x}", Sha256::digest(nonce.as_bytes()));
        let argv = vec![
            "--pals-proxy-entry-functional-smoke=true".to_owned(),
            "--pals-system-proxy=true".to_owned(),
            format!(
                "--pals-system-child-binary={}",
                root.join("unused-engine").display()
            ),
            format!("--pals-system-startup-lock={}", lock_path.display()),
            format!("--pals-system-startup-lock-sha256={digest}"),
            "--pals-system-startup-ms=30000".to_owned(),
            "--pals-system-parent-abstract=synthetic-child".to_owned(),
            "--pals-smoke-policy=proxy-entry-v1".to_owned(),
            format!("--pals-smoke-owner-id={}", &digest[..32]),
            "--pals-smoke-owner-epoch=1".to_owned(),
            "--pals-smoke-role=baseline".to_owned(),
            "--pals-smoke-generation=1".to_owned(),
            format!("--pals-smoke-lock-sha256={digest}"),
            format!("--pals-smoke-plan-sha256={digest}"),
            format!("--pals-smoke-proxy-sha256={digest}"),
        ];
        let stdio = stdio();
        let before = fds(&stdio);
        let run = Arc::new(AtomicBool::new(false));
        let cleanup = Arc::new(AtomicBool::new(false));
        let failure =
            match admit_initial(argv.clone(), Instant::now(), &root, stdio, &run, &cleanup) {
                Err(failure) => failure,
                Ok(_) => panic!("nonexistent lock accepted"),
            };
        match failure.cause {
            AdmissionFailureCauseV1::Lock(partial) => {
                assert!(partial.root.is_some());
                assert!(partial.pin.is_none());
            }
            _ => panic!("expected missing lock admission failure"),
        }
        match failure.argv {
            AdmissionArgvV1::Parsed(parsed) => assert_eq!(parsed.original_arguments(), argv),
            AdmissionArgvV1::Raw(_) => panic!("expected parsed argv"),
        }
        assert_eq!(fds(&failure.stdio), before);
        assert!(failure.lock.is_none());
    }
}
