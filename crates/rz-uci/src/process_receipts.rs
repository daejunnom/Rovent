//! Shared bounded process receipt capabilities, independent of any model/backend.
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::{
    ambient_authority,
    fs::{Dir, OpenOptions},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::{
    fmt,
    fs::File,
    io::{self, Read, Write},
};

pub const MAX_RECEIPT_BYTES: usize = 256 * 1024;
const MAX_EXECUTABLE_BYTES: u64 = 128 * 1024 * 1024;
pub(crate) fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|value| format!("{value:02x}")).collect()
}
pub(crate) fn process_run_id() -> String {
    format!("native-process-{}", std::process::id())
}

pub struct ProcessReceiptError {
    pub stage: &'static str,
    pub io_kind: Option<io::ErrorKind>,
    pub io_error: Option<io::Error>,
}
impl ProcessReceiptError {
    pub(crate) fn boundary(stage: &'static str) -> Self {
        Self {
            stage,
            io_kind: None,
            io_error: None,
        }
    }
    pub(crate) fn io(stage: &'static str, error: io::Error) -> Self {
        Self {
            stage,
            io_kind: Some(error.kind()),
            io_error: Some(error),
        }
    }
}
impl fmt::Debug for ProcessReceiptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProcessReceiptError")
            .field("stage", &self.stage)
            .field("io_kind", &self.io_kind)
            .finish()
    }
}
impl fmt::Display for ProcessReceiptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "process receipt failed: {} ({:?})",
            self.stage, self.io_kind
        )
    }
}
impl std::error::Error for ProcessReceiptError {}
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutableIdentitySourceV1 {
    /// The handle follows Linux's current mapped executable inode, including
    /// when its original pathname was replaced or unlinked after launch.
    LinuxProcSelfExe,
    /// A file observed at the platform's current_exe path. This is not proof of
    /// every byte in the operating system's already mapped process image.
    CurrentExecutableFile,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutableIdentityV1 {
    pub sha256: String,
    pub bytes: u64,
    pub identity_source: ExecutableIdentitySourceV1,
}
impl ExecutableIdentityV1 {
    pub fn observe() -> Result<Self, ProcessReceiptError> {
        Self::observe_with_budget(MAX_EXECUTABLE_BYTES, false, || true)
    }

    /// V3 binds the mapped executable's complete bytes to its original
    /// ArtifactRef and original startup control rather than a global ceiling.
    /// The callback is checked between reads; a single OS read is not preempted.
    pub(crate) fn observe_exact_with_control(
        declared_bytes: u64,
        control_active: impl FnMut() -> bool,
    ) -> Result<Self, ProcessReceiptError> {
        Self::observe_with_budget(declared_bytes, true, control_active)
    }

    fn observe_with_budget(
        max_bytes: u64,
        exact: bool,
        mut control_active: impl FnMut() -> bool,
    ) -> Result<Self, ProcessReceiptError> {
        if max_bytes == 0 {
            return Err(ProcessReceiptError::boundary(
                "executable identity byte budget is empty",
            ));
        }
        #[cfg(target_os = "linux")]
        let (file, source) = (
            File::open("/proc/self/exe"),
            ExecutableIdentitySourceV1::LinuxProcSelfExe,
        );
        #[cfg(not(target_os = "linux"))]
        let (file, source) = (
            std::env::current_exe().and_then(File::open),
            ExecutableIdentitySourceV1::CurrentExecutableFile,
        );
        let mut file =
            file.map_err(|error| ProcessReceiptError::io("observe executable", error))?;
        let mut bytes = 0_u64;
        let mut hash = Sha256::new();
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            if !control_active() {
                return Err(ProcessReceiptError::boundary(
                    "executable identity control expired or cancelled",
                ));
            }
            let count = file
                .read(&mut buffer)
                .map_err(|error| ProcessReceiptError::io("read executable", error))?;
            if count == 0 {
                break;
            }
            bytes = bytes
                .checked_add(count as u64)
                .filter(|count| *count <= max_bytes)
                .ok_or_else(|| {
                    ProcessReceiptError::boundary("executable identity byte budget exceeded")
                })?;
            hash.update(&buffer[..count]);
        }
        if bytes == 0 {
            return Err(ProcessReceiptError::boundary(
                "executable identity is empty",
            ));
        }
        if exact && bytes != max_bytes {
            return Err(ProcessReceiptError::boundary(
                "executable identity differs from declared length",
            ));
        }
        if !control_active() {
            return Err(ProcessReceiptError::boundary(
                "executable identity control expired or cancelled",
            ));
        }
        Ok(Self {
            sha256: hex(&hash.finalize().into()),
            bytes,
            identity_source: source,
        })
    }
}

struct BoundedJson {
    bytes: Vec<u8>,
}
impl Write for BoundedJson {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > MAX_RECEIPT_BYTES.saturating_sub(self.bytes.len()) {
            return Err(io::Error::other("attestation JSON byte budget exceeded"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
pub(crate) fn bounded_json(value: &impl Serialize) -> Result<Vec<u8>, ProcessReceiptError> {
    let mut writer = BoundedJson { bytes: Vec::new() };
    writer
        .bytes
        .try_reserve_exact(MAX_RECEIPT_BYTES)
        .map_err(|_| ProcessReceiptError::boundary("cannot reserve bounded receipt bytes"))?;
    serde_json::to_writer(&mut writer, value)
        .map_err(|_| ProcessReceiptError::boundary("cannot encode bounded public receipt"))?;
    writer
        .write_all(b"\n")
        .map_err(|error| ProcessReceiptError::io("finish bounded receipt encoding", error))?;
    Ok(writer.bytes)
}

pub struct ProcessReceiptWriter {
    _directory: Dir,
    startup: cap_std::fs::File,
    termination: cap_std::fs::File,
    startup_written: bool,
    startup_attempted: bool,
    termination_attempted: bool,
}
impl ProcessReceiptWriter {
    /// Model-specific receipts reuse the same pinned, no-follow capability.
    /// This does not construct an LC0 model configuration for another model.
    pub fn open_named_root(
        root: &std::path::Path,
        startup_name: &'static str,
        termination_name: &'static str,
    ) -> Result<Self, ProcessReceiptError> {
        if !root.is_absolute() {
            return Err(ProcessReceiptError::boundary(
                "attestation output root must be absolute",
            ));
        }
        let parent = root.parent().ok_or_else(|| {
            ProcessReceiptError::boundary(
                "attestation requires a dedicated private output directory",
            )
        })?;
        let name = root.file_name().ok_or_else(|| {
            ProcessReceiptError::boundary("attestation requires a named private output directory")
        })?;
        let parent = Dir::open_ambient_dir(parent, ambient_authority())
            .map_err(|error| ProcessReceiptError::io("pin output parent directory", error))?;
        let directory = parent.open_dir_nofollow(name).map_err(|error| {
            ProcessReceiptError::io(
                "pin private output directory without following symlinks",
                error,
            )
        })?;
        // Fastchess may restart each role for the second game. Each executable
        // instance owns a fresh bounded ASCII slot; PID reuse never overwrites
        // an older run. The launcher validates each slot separately.
        let run_id = process_run_id();
        directory.create_dir(&run_id).map_err(|error| {
            ProcessReceiptError::io("reserve fresh process receipt directory", error)
        })?;
        let process_directory = directory.open_dir_nofollow(&run_id).map_err(|error| {
            ProcessReceiptError::io(
                "pin process receipt directory without following symlinks",
                error,
            )
        })?;
        Self::from_directory_named(process_directory, startup_name, termination_name)
    }
    pub(crate) fn from_directory_named(
        directory: Dir,
        startup_name: &'static str,
        termination_name: &'static str,
    ) -> Result<Self, ProcessReceiptError> {
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No);
        let startup = directory
            .open_with(startup_name, &options)
            .map_err(|error| ProcessReceiptError::io("reserve new startup receipt", error))?;
        let termination = directory
            .open_with(termination_name, &options)
            .map_err(|error| ProcessReceiptError::io("reserve new termination receipt", error))?;
        Ok(Self {
            _directory: directory,
            startup,
            termination,
            startup_written: false,
            startup_attempted: false,
            termination_attempted: false,
        })
    }
    pub fn startup(&mut self, receipt: &impl Serialize) -> Result<(), ProcessReceiptError> {
        self.publish_startup(receipt)
    }
    pub fn publish_startup(&mut self, receipt: &impl Serialize) -> Result<(), ProcessReceiptError> {
        if self.startup_attempted {
            return Err(ProcessReceiptError::boundary(
                "startup receipt publication was already attempted",
            ));
        }
        let bytes = bounded_json(receipt)?;
        self.startup_attempted = true;
        self.startup
            .write_all(&bytes)
            .map_err(|error| ProcessReceiptError::io("write startup receipt", error))?;
        self.startup
            .sync_all()
            .map_err(|error| ProcessReceiptError::io("sync startup receipt", error))?;
        self.startup_written = true;
        Ok(())
    }
    #[cfg(feature = "onnx-cpu")]
    pub(crate) fn publish_auxiliary(
        &self,
        name: &'static str,
        receipt: &impl Serialize,
    ) -> Result<(), ProcessReceiptError> {
        if !self.startup_written {
            return Err(ProcessReceiptError::boundary(
                "auxiliary receipt requires issued startup",
            ));
        }
        let bytes = bounded_json(receipt)?;
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No);
        let mut file = self
            ._directory
            .open_with(name, &options)
            .map_err(|e| ProcessReceiptError::io("reserve auxiliary receipt", e))?;
        file.write_all(&bytes)
            .map_err(|e| ProcessReceiptError::io("write auxiliary receipt", e))?;
        file.sync_all()
            .map_err(|e| ProcessReceiptError::io("sync auxiliary receipt", e))
    }
    /// Stream a bounded experiment journal directly to its capability; no
    /// second full JSON allocation. Partial files are explicitly incomplete.
    #[cfg(feature = "experimental-batch")]
    pub(crate) fn publish_large_auxiliary(
        &self,
        name: &'static str,
        receipt: &impl Serialize,
        cap: u64,
    ) -> Result<(String, u64), ProcessReceiptError> {
        #[cfg(unix)]
        use cap_std::fs::OpenOptionsExt;
        if !self.startup_written {
            return Err(ProcessReceiptError::boundary("journal requires startup"));
        }
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        let mut file = self
            ._directory
            .open_with(name, &options)
            .map_err(|e| ProcessReceiptError::io("create bounded journal", e))?;
        struct Bounded<'a> {
            file: &'a mut cap_std::fs::File,
            hash: Sha256,
            bytes: u64,
            cap: u64,
        }
        impl Write for Bounded<'_> {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if bytes.len() as u64 > self.cap.saturating_sub(self.bytes) {
                    return Err(std::io::Error::other("batch journal byte budget exceeded"));
                }
                let count = self.file.write(bytes)?;
                self.hash.update(&bytes[..count]);
                self.bytes += count as u64;
                Ok(count)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                self.file.flush()
            }
        }
        let mut writer = Bounded {
            file: &mut file,
            hash: Sha256::new(),
            bytes: 0,
            cap,
        };
        serde_json::to_writer(&mut writer, receipt)
            .map_err(|_| ProcessReceiptError::boundary("serialize bounded batch journal"))?;
        let result = (hex(&writer.hash.finalize().into()), writer.bytes);
        file.sync_all()
            .map_err(|e| ProcessReceiptError::io("sync bounded batch journal", e))?;
        Ok(result)
    }
    pub fn termination(&mut self, receipt: &impl Serialize) -> Result<(), ProcessReceiptError> {
        self.publish_termination(receipt)
    }
    pub fn publish_termination(
        &mut self,
        receipt: &impl Serialize,
    ) -> Result<(), ProcessReceiptError> {
        if !self.startup_written || self.termination_attempted {
            return Err(ProcessReceiptError::boundary(
                "termination receipt requires one issued startup receipt",
            ));
        }
        let bytes = bounded_json(receipt)?;
        self.termination_attempted = true;
        self.termination
            .write_all(&bytes)
            .map_err(|error| ProcessReceiptError::io("write termination receipt", error))?;
        self.termination
            .sync_all()
            .map_err(|error| ProcessReceiptError::io("sync termination receipt", error))?;
        Ok(())
    }
}
