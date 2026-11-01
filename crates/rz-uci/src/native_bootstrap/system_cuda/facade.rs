//! Minimal library facade for an LC0-specific authenticated supervisor.
//!
//! The caller captures a ticket before publishing any endpoint, accepts a
//! connection from the original parent, and retains the resulting session and
//! process whenever completion is unknown. The legacy CLI guard remains in
//! place until a real parent/listener owner and termination path use this API.

use super::{control::Lc0ControlSupervisor, *};
use std::os::unix::net::UnixListener;

/// Captured before endpoint publication. This contains no loaded ORT runtime.
pub struct Lc0PendingSupervisor {
    pending: Lc0PendingParent,
    lock: SystemRuntimeLockV6,
    config: NativeConfig,
    epoch: u64,
    control_until: Instant,
}

/// Owns the original authenticated parent, startup custody and worker.
/// This is not an OS-process lifetime manager; dropping it or exiting on an
/// unknown physical result is not safe. No resource is copied to a cache.
#[must_use]
pub struct Lc0SupervisedSession {
    supervisor: Lc0ControlSupervisor,
    lock: SystemRuntimeLockV6,
    config: NativeConfig,
    owners: Arc<OwnerRegistry>,
}

impl Lc0PendingSupervisor {
    /// Read the bounded, explicitly pinned V6 lock and capture this process's
    /// actual parent before the caller creates or publishes an endpoint.
    pub fn capture(
        config: &NativeConfig,
        epoch: u64,
        control_until: Instant,
    ) -> Result<Self, NativeBootstrapError> {
        let path = config
            .system_cuda_lock
            .as_ref()
            .ok_or(NativeBootstrapError::Config(
                "LC0 system CUDA lock path is absent",
            ))?;
        let bytes = rz_eval::asset::read_bounded(path, 256 * 1024)?;
        if Some(rz_eval::asset::sha256(&bytes)) != config.system_cuda_lock_sha256 {
            return Err(NativeBootstrapError::Config(
                "LC0 system CUDA lock file digest differs",
            ));
        }
        let text = std::str::from_utf8(&bytes)
            .map_err(|_| NativeBootstrapError::Config("LC0 system CUDA lock is not UTF-8"))?;
        let lock = SystemRuntimeLockV6::from_json(text)
            .map_err(|_| NativeBootstrapError::Config("LC0 system CUDA lock is invalid"))?;
        if control_until <= Instant::now() || config.parallelism != 1 {
            return Err(NativeBootstrapError::Config(
                "LC0 system control window or B1 is invalid",
            ));
        }
        let pending = Lc0PendingParent::capture(epoch, &lock)?;
        Ok(Self {
            pending,
            lock,
            config: config.clone(),
            epoch,
            control_until,
        })
    }

    /// This stream must come from the listener owned by the original process
    /// supervisor. Any error here occurs before ORT/session native entry.
    pub fn authenticate(
        self,
        stream: UnixStream,
        until: Instant,
    ) -> Result<Lc0SupervisedSession, NativeBootstrapError> {
        let parent = self
            .pending
            .authenticate(stream, until.min(self.control_until))
            .map_err(|_| NativeBootstrapError::Config("LC0 original parent admission failed"))?;
        let asset = MaiaAsset::load(
            &self.config.source_weights,
            &self.config.onnx_model,
            &self.config.export_manifest,
        )?;
        let custody =
            Lc0SystemStartupCustody::new(asset, parent, self.epoch, &self.lock, &self.config)?;
        Ok(Lc0SupervisedSession {
            supervisor: Lc0ControlSupervisor::new(
                Lc0SystemSessionOwner::new(custody),
                self.control_until,
            ),
            lock: self.lock,
            config: self.config,
            owners: Arc::new(OwnerRegistry::default()),
        })
    }
}

impl Lc0SupervisedSession {
    /// This only returns the same registry used to construct the factory.
    pub fn owners(&self) -> Arc<OwnerRegistry> {
        Arc::clone(&self.owners)
    }

    /// The caller provides a private profiling prefix already reserved under
    /// its retained output capability. All B1 settings are checked by custody
    /// before native entry; this function retains the owner on failure.
    pub fn prepare_until(
        &mut self,
        until: Instant,
        cancelled: &AtomicBool,
        backend: BackendConfig,
    ) -> Result<(), &'static str> {
        self.supervisor.prepare_until(
            &self.lock,
            until,
            cancelled,
            backend,
            &self.owners,
            &self.config,
        )
    }

    /// Setup, UCI serving and its failures must be inside `serve`. Return from
    /// this function is not proof of process-wide or device-wide completion.
    pub fn serve_until(
        &mut self,
        until: Instant,
        serve: impl FnOnce(Arc<NativeCudaFactory>) -> Result<(), EngineError>,
    ) -> Result<(), NativeBootstrapError> {
        self.supervisor.serve_until(until, serve)
    }

    /// Required after setup failure before any service has begun. A session
    /// that never reached a worker remains explicitly unknown and retained.
    pub fn close_without_service(&mut self) {
        self.supervisor.close_without_service();
    }

    /// The listener belongs to the external process supervisor; its creation,
    /// path and cleanup are outside this facade. This bounded loop processes
    /// one request at a time only after the caller has returned from serving.
    pub fn control_loop(&mut self, listener: &UnixListener) -> io::Result<()> {
        self.supervisor.control_loop(listener)
    }
}
