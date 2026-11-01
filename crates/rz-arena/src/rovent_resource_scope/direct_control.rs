//! Explicit direct-run custody endpoint. Its listener and original endpoint
//! owner are kept together; a socket message is never a physical fence.

use crate::{
    ArenaError,
    rovent_training_process::utility::runtime::ipc_endpoint::{
        OwnedUtilityEndpoint, UtilityListener,
    },
};
use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::Dir;
use nix::{
    fcntl::{OFlag, openat},
    sys::stat::{Mode, fchmod},
};
use std::{
    fs::File,
    io::{self, BufRead, BufReader, Read, Write},
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

fn invalid(detail: impl std::fmt::Display) -> ArenaError {
    ArenaError::Integrity(format!("direct custody endpoint: {detail}"))
}

/// The exact writer directory, endpoint and listener belong to one run. The
/// caller stores this before native entry and keeps it while completion is
/// unknown. Only explicit, proven normal/pre-native closure may call cleanup.
pub(super) struct DirectControl {
    endpoint: OwnedUtilityEndpoint,
    listener: Option<UtilityListener>,
    path: PathBuf,
    cleanup_budget: Duration,
    recovery_evidence: Mutex<RecoveryEvidence>,
}

enum RecoveryEvidence {
    NotStarted,
    Running {
        receiver: std::sync::mpsc::Receiver<Result<(), String>>,
        _worker: std::thread::JoinHandle<()>,
    },
    Finished(Result<(), String>),
}

struct BoundedEvidence(Vec<u8>);
#[derive(serde::Serialize)]
struct RecoveryEvidenceDocument<'a> {
    schema: &'static str,
    observation: &'a serde_json::Value,
    complete_event_export: bool,
    original_owner_retained: bool,
    release_authorized: bool,
}
impl Write for BoundedEvidence {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > super::MAX_EVIDENCE_BYTES.saturating_sub(self.0.len()) {
            return Err(io::Error::other(
                "recovery evidence exceeds configured limit",
            ));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Created only after the same original endpoint has bound successfully.
/// Its private shell cannot be replaced by the UCI control-slot consumer.
pub(super) struct BoundDirectControl {
    original: Arc<Mutex<DirectControl>>,
}

impl BoundDirectControl {
    pub(super) fn from_bound(
        original: Arc<Mutex<DirectControl>>,
    ) -> Result<Self, Arc<Mutex<DirectControl>>> {
        let bound = original
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .listener
            .is_some();
        if bound {
            Ok(Self { original })
        } else {
            Err(original)
        }
    }

    pub(super) fn observe_one(
        &self,
        key: Option<&rz_uci::rovent_bootstrap::launch::RoventV3ScopeKey>,
        until: Instant,
        recover: &mut dyn FnMut(Instant) -> Option<serde_json::Value>,
    ) -> io::Result<()> {
        self.original
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .observe_one(key, until, recover)
    }

    pub(super) fn observe_bound_identity(&self) -> Result<(), ArenaError> {
        self.original
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .observe_bound_identity()
    }
}

impl DirectControl {
    pub(super) fn prepare(
        directory: &Dir,
        route: &Path,
        cleanup_until: Instant,
        cleanup_budget: Duration,
    ) -> Result<Self, ArenaError> {
        let original: File = directory.try_clone().map_err(invalid)?.into_std_file();
        let metadata = original.metadata().map_err(invalid)?;
        if !metadata.is_dir() || metadata.uid() != nix::unistd::geteuid().as_raw() {
            return Err(invalid("exclusive original writer directory unavailable"));
        }
        // cap-std directory capabilities may be O_PATH descriptors, which
        // cannot be passed to fchmod. Reopen "." relative to the original FD
        // and verify it is the same directory before changing its mode.
        let file = File::from(
            openat(
                &original,
                ".",
                OFlag::O_RDONLY | OFlag::O_DIRECTORY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
                Mode::empty(),
            )
            .map_err(invalid)?,
        );
        let reopened = file.metadata().map_err(invalid)?;
        if !reopened.is_dir()
            || reopened.dev() != metadata.dev()
            || reopened.ino() != metadata.ino()
            || reopened.uid() != metadata.uid()
        {
            return Err(invalid(
                "reopened control directory differs from original writer",
            ));
        }
        // This is the exclusive per-process writer leaf, not a shared root.
        // Explicit custody mode requires a private directory for reconnects.
        fchmod(&file, Mode::from_bits_truncate(0o700)).map_err(invalid)?;
        let endpoint_path = route.join("custody.sock");
        let endpoint = OwnedUtilityEndpoint::prepare(file, route, &endpoint_path, cleanup_until)?;
        let path = endpoint.owner_process_socket_path();
        Ok(Self {
            endpoint,
            listener: None,
            path,
            cleanup_budget,
            recovery_evidence: Mutex::new(RecoveryEvidence::NotStarted),
        })
    }

    pub(super) fn bind(&mut self) -> Result<(), ArenaError> {
        let listener = self.endpoint.bind()?;
        self.listener = Some(listener);
        Ok(())
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    pub(super) fn observe_bound_identity(&self) -> Result<(), ArenaError> {
        if self.listener.is_none() {
            return Err(invalid("original listener not retained"));
        }
        self.endpoint.observe_bound_identity()
    }

    pub(super) fn listener(&self) -> Option<&UtilityListener> {
        self.listener.as_ref()
    }

    /// Publish only the bounded, explicitly incomplete observation. The
    /// original typed values and pins remain in the foreground custody owner.
    /// One writer per owner prevents a slow filesystem from blocking control.
    fn recovery_evidence_status(&self, recovered: Option<&serde_json::Value>) -> serde_json::Value {
        let mut state = self
            .recovery_evidence
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let RecoveryEvidence::Running { receiver, .. } = &*state {
            match receiver.try_recv() {
                Ok(result) => *state = RecoveryEvidence::Finished(result),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    *state = RecoveryEvidence::Finished(Err(
                        "evidence writer terminated without result".into(),
                    ))
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => (),
            }
        }
        if matches!(*state, RecoveryEvidence::NotStarted) {
            if let Some(summary) = recovered.filter(|value| {
                value.get("state").and_then(|v| v.as_str())
                    == Some("same_owner_recovered_and_retained")
            }) {
                let directory = self.endpoint.clone_original_directory();
                let mut bytes = BoundedEvidence(Vec::new());
                let encoded = serde_json::to_writer(
                    &mut bytes,
                    &RecoveryEvidenceDocument {
                        schema: "rz-rovent-before-native-recovery-observation/1",
                        observation: summary,
                        complete_event_export: false,
                        original_owner_retained: true,
                        release_authorized: false,
                    },
                );
                match (directory, encoded) {
                    (Ok(directory), Ok(())) => {
                        let bytes = bytes.0;
                        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
                        let worker = std::thread::Builder::new()
                            .name("rovent-recovery-evidence".into())
                            .spawn(move || {
                                let result = (|| -> Result<(), String> {
                                    let mut options = cap_std::fs::OpenOptions::new();
                                    options
                                        .write(true)
                                        .create_new(true)
                                        .follow(FollowSymlinks::No);
                                    let mut file = directory
                                        .open_with(
                                            "before-native-recovery-observation.json",
                                            &options,
                                        )
                                        .map_err(|e| e.to_string())?;
                                    file.write_all(&bytes)
                                        .and_then(|()| file.sync_all())
                                        .map_err(|e| e.to_string())?;
                                    directory
                                        .open(".")
                                        .and_then(|file| file.sync_all())
                                        .map_err(|e| e.to_string())?;
                                    Ok(())
                                })();
                                let _ = sender.send(result);
                            });
                        *state = match worker {
                            Ok(worker) => RecoveryEvidence::Running {
                                receiver,
                                _worker: worker,
                            },
                            Err(error) => RecoveryEvidence::Finished(Err(error.to_string())),
                        };
                    }
                    (Err(error), _) => *state = RecoveryEvidence::Finished(Err(error.to_string())),
                    (_, Err(error)) => *state = RecoveryEvidence::Finished(Err(error.to_string())),
                }
            }
        }
        match &*state {
            RecoveryEvidence::NotStarted => serde_json::json!({"state":"not_started"}),
            RecoveryEvidence::Running { .. } => serde_json::json!({"state":"writing"}),
            RecoveryEvidence::Finished(Ok(())) => {
                serde_json::json!({"state":"observation_synced", "complete_event_export":false})
            }
            RecoveryEvidence::Finished(Err(error)) => {
                serde_json::json!({"state":"failed", "error":error})
            }
        }
    }

    /// Serve one bounded, same-UID observation. A client EOF or malformed
    /// request closes only that connection and never releases the owner.
    pub(super) fn observe_one(
        &self,
        key: Option<&rz_uci::rovent_bootstrap::launch::RoventV3ScopeKey>,
        original_cleanup_until: Instant,
        recover: &mut dyn FnMut(Instant) -> Option<serde_json::Value>,
    ) -> io::Result<()> {
        use nix::sys::socket::{getsockopt, sockopt::PeerCredentials};
        let listener = self
            .listener
            .as_ref()
            .ok_or_else(|| io::Error::other("original control listener unavailable"))?;
        let (mut stream, _) = listener.accept()?;
        let peer = getsockopt(&stream, PeerCredentials).map_err(io::Error::other)?;
        if peer.uid() != nix::unistd::geteuid().as_raw() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "control peer uid differs",
            ));
        }
        stream.set_read_timeout(Some(Duration::from_secs(1)))?;
        stream.set_write_timeout(Some(Duration::from_secs(1)))?;
        let mut frame = Vec::new();
        BufReader::new((&mut stream).take(4097)).read_until(b'\n', &mut frame)?;
        if frame.len() > 4096 || frame.last() != Some(&b'\n') {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bounded control frame required",
            ));
        }
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Request {
            op: String,
        }
        let request: Request = serde_json::from_slice(&frame).map_err(io::Error::other)?;
        let scope = key.and_then(|key| key.observe());
        let session_evidence = key
            .map(|key| match key.terminal_evidence() {
                rz_uci::rovent_bootstrap::launch::V3ScopeTerminalEvidence::BeforeNative => {
                    "before_native"
                }
                rz_uci::rovent_bootstrap::launch::V3ScopeTerminalEvidence::SessionReleased => {
                    "session_released"
                }
                rz_uci::rovent_bootstrap::launch::V3ScopeTerminalEvidence::Unknown => "unknown",
            })
            .unwrap_or("scope_unavailable");
        let mut recovery = None;
        let observation = match request.op.as_str() {
            "status" => None,
            "reobserve" if key.is_some() && Instant::now() < original_cleanup_until => {
                let deadline = Instant::now()
                    .checked_add(Duration::from_secs(1))
                    .map(|until| until.min(original_cleanup_until))
                    .unwrap_or(original_cleanup_until);
                recovery = recover(deadline);
                if Instant::now() < deadline {
                    key.map(|key| format!("{:?}", key.reobserve_native(deadline)))
                } else {
                    Some("reobserve_deadline_expired".into())
                }
            }
            "reobserve" if key.is_none() => Some("original_scope_key_unavailable".into()),
            "reobserve" => Some("original_cleanup_deadline_expired".into()),
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "unsupported control operation",
                ));
            }
        };
        let evidence = self.recovery_evidence_status(recovery.as_ref());
        let response = serde_json::json!({
            "schema":"rz-rovent-direct-custody-status/1",
            "process_id":std::process::id(),
            "physical_completion":"unknown",
            "session_evidence":session_evidence,
            "scope_state":scope.as_ref().map(|value| format!("{:?}", value.state)),
            "original_owner_parked":scope.as_ref().map(|value| value.parked_original),
            "native_observation":observation,
            "before_native_recovery":recovery,
            "recovery_observation_evidence":evidence,
            "socket_identity_verified":self.observe_bound_identity().is_ok(),
            "release_authorized":false,
        });
        serde_json::to_writer(&mut stream, &response).map_err(io::Error::other)?;
        stream.write_all(b"\n")?;
        Ok(())
    }

    pub(super) fn cleanup_confirmed(&mut self) -> Result<(), ArenaError> {
        drop(self.listener.take());
        self.endpoint.cleanup()
    }

    /// Only called after the original session's terminal evidence. This is a
    /// new bounded filesystem-cleanup window, not more native execution time.
    pub(super) fn cleanup_after_session(&mut self) -> Result<(), ArenaError> {
        let until = Instant::now().checked_add(self.cleanup_budget);
        drop(self.listener.take());
        // Overflow is an expired cleanup window, not permission for Drop to
        // retry against the original startup deadline.
        self.endpoint
            .cleanup_after_session(until.unwrap_or_else(Instant::now))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn serve_when_ready(control: &DirectControl) -> io::Result<()> {
        let until = Instant::now() + Duration::from_secs(2);
        loop {
            match control.observe_one(None, Instant::now(), &mut |_| None) {
                Err(error)
                    if error.kind() == io::ErrorKind::WouldBlock && Instant::now() < until =>
                {
                    std::thread::sleep(Duration::from_millis(1))
                }
                result => return result,
            }
        }
    }

    struct Fixture {
        parent: Dir,
        name: String,
        control: Option<DirectControl>,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            if let Some(control) = self.control.as_mut() {
                let writer_active = match &*control
                    .recovery_evidence
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                {
                    RecoveryEvidence::Running { _worker, .. } => !_worker.is_finished(),
                    _ => false,
                };
                if writer_active {
                    return;
                }
                let _ = control.cleanup_confirmed();
            }
            drop(self.control.take());
            let _ = self.parent.remove_file(format!(
                "{}/before-native-recovery-observation.json",
                self.name
            ));
            let _ = self.parent.remove_dir(&self.name);
        }
    }

    #[test]
    fn original_private_writer_endpoint_is_explicitly_cleaned() {
        let parent =
            Dir::open_ambient_dir(std::env::temp_dir(), cap_std::ambient_authority()).unwrap();
        let name = format!(
            "rovent-direct-control-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            "x".repeat(110)
        );
        parent.create_dir(&name).unwrap();
        let mut fixture = Fixture {
            parent,
            name: name.clone(),
            control: None,
        };
        let directory = fixture.parent.open_dir(&name).unwrap();
        let path = std::env::temp_dir().join(&name);
        fixture.control = Some(
            DirectControl::prepare(
                &directory,
                &path,
                Instant::now() + Duration::from_secs(60),
                Duration::from_secs(60),
            )
            .unwrap(),
        );
        let control = fixture.control.as_mut().unwrap();
        control.bind().unwrap();
        assert!(control.path().exists());
        control.listener().unwrap().set_nonblocking(true).unwrap();
        std::thread::scope(|scope| {
            let serving = scope.spawn(|| serve_when_ready(control));
            let client = std::os::unix::net::UnixStream::connect(control.path()).unwrap();
            drop(client);
            assert!(
                serving.join().unwrap().is_err(),
                "one EOF is not a completion"
            );
        });
        std::thread::scope(|scope| {
            let serving = scope.spawn(|| serve_when_ready(control));
            let mut client = std::os::unix::net::UnixStream::connect(control.path()).unwrap();
            client
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            client.write_all(b"{\"op\":\"reobserve\"}\n").unwrap();
            let mut response = String::new();
            BufReader::new(client).read_line(&mut response).unwrap();
            let response: serde_json::Value = serde_json::from_str(&response).unwrap();
            assert_eq!(response["physical_completion"], "unknown");
            assert_eq!(response["session_evidence"], "scope_unavailable");
            assert_eq!(
                response["native_observation"],
                "original_scope_key_unavailable"
            );
            assert_eq!(response["release_authorized"], false);
            serving.join().unwrap().unwrap();
        });
        let summary =
            serde_json::json!({"state":"same_owner_recovered_and_retained", "marker":"first"});
        let until = Instant::now() + Duration::from_secs(5);
        loop {
            let evidence = control.recovery_evidence_status(Some(&summary));
            if evidence["state"] == "observation_synced" {
                break;
            }
            assert_eq!(evidence["state"], "writing", "{evidence}");
            assert!(Instant::now() < until, "evidence writer did not finish");
            std::thread::sleep(Duration::from_millis(5));
        }
        let replacement = serde_json::json!({"state":"same_owner_recovered_and_retained", "marker":"replacement"});
        assert_eq!(
            control.recovery_evidence_status(Some(&replacement))["state"],
            "observation_synced"
        );
        let evidence: serde_json::Value = serde_json::from_slice(
            &std::fs::read(path.join("before-native-recovery-observation.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(evidence["observation"]["marker"], "first");
        assert_eq!(evidence["complete_event_export"], false);
        assert_eq!(evidence["release_authorized"], false);
        control.cleanup_confirmed().unwrap();
        assert!(!path.join("custody.sock").exists());
    }
}
