//! NoGame entry preparation using the same lock, grant and supervisor owners.
//! This does not start a child, admit listener provenance, or establish GPU
//! completion. The production execution guard remains in force.
use super::{
    arena_launch_inputs::{ArenaSocketAddress, ArenaSocketArgs},
    cli_v6::parse_arguments,
    proxy_entry_smoke_admission_v1::{AdmissionFailureV1, admit_initial},
    proxy_entry_smoke_prepare_v1::{
        NoGameProxyPrepareOwnerV1, OriginalProxyStdioV1, PreparedNoGameGrantOwnersV1,
        SmokePreparationStageV1,
    },
    proxy_entry_smoke_supervisor_v1::{
        NoGamePreparedSupervisorV1, prepare_supervisor, startup_arguments,
    },
};
use std::{
    io,
    os::unix::net::UnixStream,
    path::{Path, PathBuf},
    sync::{Arc, atomic::AtomicBool},
    thread,
    time::{Duration, Instant},
};

#[cfg(test)]
#[path = "smoke_entry_tests.rs"]
mod tests;

#[must_use = "retain or explicitly settle the original NoGame entry owners"]
pub(super) enum NoGameEntryFailureV1 {
    Inputs {
        error: io::Error,
        arguments: Vec<String>,
        stdio: OriginalProxyStdioV1,
    },
    Admission(AdmissionFailureV1),
    Grant {
        error: io::Error,
        owner: NoGameProxyPrepareOwnerV1,
        stream: Option<UnixStream>,
    },
    Supervisor {
        error: io::Error,
        owner: PreparedNoGameGrantOwnersV1,
    },
    Cancel {
        error: io::Error,
        owner: NoGamePreparedSupervisorV1,
        stream: Option<UnixStream>,
    },
}

fn path(address: ArenaSocketAddress) -> PathBuf {
    match address {
        ArenaSocketAddress::Path(path) => path,
        ArenaSocketAddress::Abstract(name) => PathBuf::from(format!("abstract:{name}")),
    }
}

fn parse_routes(arguments: &[String], entry: Instant) -> io::Result<(PathBuf, PathBuf, Instant)> {
    let startup = parse_arguments(&startup_arguments(arguments))?.startup;
    let child = startup
        .parent_socket
        .as_deref()
        .and_then(Path::to_str)
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "NoGame child socket missing")
        })?;
    let child = ArenaSocketAddress::parse(child)?;
    let mut arena = ArenaSocketArgs::default();
    for argument in arguments {
        arena.consume(argument)?;
    }
    let routes = arena.finish(&child)?;
    let until = entry
        .checked_add(Duration::from_millis(startup.startup_ms))
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "NoGame startup bound overflow")
        })?;
    Ok((path(routes.grant), path(routes.cancel), until))
}

/// This performs no child spawn. On any failure all acquired original owners
/// are returned, including a socket that failed after connecting.
pub(super) fn prepare_entry(
    arguments: Vec<String>,
    entry: Instant,
    private_cwd: &Path,
    stdio: OriginalProxyStdioV1,
    cancelled: &Arc<AtomicBool>,
    cleanup_cancelled: &Arc<AtomicBool>,
    max_line: usize,
) -> Result<NoGamePreparedSupervisorV1, NoGameEntryFailureV1> {
    let (grant_path, cancel_path, until) = match parse_routes(&arguments, entry) {
        Ok(value) => value,
        Err(error) => {
            return Err(NoGameEntryFailureV1::Inputs {
                error,
                arguments,
                stdio,
            });
        }
    };
    let mut owner = admit_initial(
        arguments,
        entry,
        private_cwd,
        stdio,
        cancelled,
        cleanup_cancelled,
    )
    .map_err(NoGameEntryFailureV1::Admission)?;
    for desired in [
        SmokePreparationStageV1::AwaitCleanupChannel,
        SmokePreparationStageV1::GrantsPrepared,
    ] {
        let stream = match owner.open_original_arena_stream(&grant_path, until) {
            Ok(stream) => stream,
            Err((error, stream)) => {
                return Err(NoGameEntryFailureV1::Grant {
                    error,
                    owner,
                    stream,
                });
            }
        };
        if let Err((error, stream)) = owner.attach_channel(stream, until) {
            return Err(NoGameEntryFailureV1::Grant {
                error,
                owner,
                stream: Some(stream),
            });
        }
        loop {
            match owner.poll() {
                Ok(stage) if stage == desired => break,
                Ok(_) if Instant::now() < until => thread::sleep(Duration::from_millis(1)),
                Ok(_) => {
                    return Err(NoGameEntryFailureV1::Grant {
                        error: io::Error::new(
                            io::ErrorKind::TimedOut,
                            "NoGame grant preparation expired",
                        ),
                        owner,
                        stream: None,
                    });
                }
                Err(error) => {
                    return Err(NoGameEntryFailureV1::Grant {
                        error,
                        owner,
                        stream: None,
                    });
                }
            }
        }
    }
    let prepared = match owner.into_prepared_grants() {
        Ok(prepared) => prepared,
        Err(owner) => {
            return Err(NoGameEntryFailureV1::Grant {
                error: io::Error::new(
                    io::ErrorKind::InvalidData,
                    "NoGame grant preparation incomplete",
                ),
                owner,
                stream: None,
            });
        }
    };
    let mut supervisor = prepare_supervisor(prepared, max_line)
        .map_err(|(error, owner)| NoGameEntryFailureV1::Supervisor { error, owner })?;
    let stream = match supervisor
        .bundle
        .lock_admission
        .open_direct_arena_stream(&cancel_path, until)
    {
        Ok(stream) => stream,
        Err((error, stream)) => {
            return Err(NoGameEntryFailureV1::Cancel {
                error,
                owner: supervisor,
                stream,
            });
        }
    };
    if let Err((error, stream)) = supervisor.attach_original_cancel(stream) {
        return Err(NoGameEntryFailureV1::Cancel {
            error,
            owner: supervisor,
            stream: Some(stream),
        });
    }
    Ok(supervisor)
}
