//! A diagnostic-only prerequisite, using the unchanged V5 numerical scope.
//! The initializer/export producer is independent of the executable source.
//! No metadata from an old initializer is edited or issued by this module.
use super::{Config, Result};
use cap_fs_ext::DirExt;
use cap_std::fs::Dir;
use rz_experiments::{
    ArtifactRef, ManifestError, PalsArtifactDomainV5, PalsEngineModeV5, PalsModelBackendV3,
    PalsNumericalScopeV5, PalsPrecisionV3, PalsStartupAssetPurposeV5 as Purpose,
    compiled_source_components_v5, validate_pals_numerical_scope_acceptance_v5,
    validate_pals_scope_artifact_domain_v5,
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

const DOMAIN: &str = "rz-explicit-gpu-inflight-content-bound-input/1";
const MAX_BYTES: u64 = 18 << 30;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ContentBoundValidation {
    schema: String,
    execution_source_commit: String,
    initializer_source_commit: String,
    export_producer_source_commit: String,
    artifact_root: PathBuf,
    pub(super) execution: Config,
    numerical_scope: PalsNumericalScopeV5,
}

fn need(value: bool, detail: &str) -> Result<()> {
    if value {
        Ok(())
    } else {
        Err(detail.to_owned().into())
    }
}
fn commit(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}
fn execution_commit(declared: &str) -> Result<()> {
    let registered = env!("RZ_PALS_COMPILED_SOURCE_COMMIT");
    need(
        commit(registered) && commit(declared) && declared == registered,
        "executor commit differs from original build registration",
    )
}
fn deadline(until: Instant) -> Result<()> {
    need(
        Instant::now() < until,
        "original prerequisite deadline reached",
    )
}
fn absolute_plain(path: &Path) -> bool {
    path.is_absolute()
        && path.components().all(|p| match p {
            Component::RootDir => true,
            Component::Normal(name) => name.to_str().is_some_and(|name| {
                let name = name.to_ascii_lowercase();
                !name.starts_with(".env")
                    && !name.starts_with("id_rsa")
                    && !name.starts_with("id_ed25519")
                    && !matches!(name.as_str(), ".ssh" | ".git")
                    && ![
                        "credential",
                        "service-account",
                        "service_account",
                        "api-key",
                        "api_key",
                    ]
                    .iter()
                    .any(|part| name.contains(part))
            }),
            _ => false,
        })
}

fn executor_components(
    execution: &BTreeMap<String, String>,
    scope: &BTreeMap<String, String>,
) -> Result<()> {
    let compiled = compiled_source_components_v5()?;
    need(
        *execution == compiled && *scope == compiled,
        "actual compiled executor components must match both consumers",
    )
}

impl ContentBoundValidation {
    /// Declarations only. The original scope and producer gate still run below.
    fn check_identity(&self) -> Result<()> {
        let scope = &self.numerical_scope;
        need(
            self.schema == DOMAIN
                && commit(&self.execution_source_commit)
                && commit(&self.initializer_source_commit)
                && commit(&self.export_producer_source_commit),
            "closed diagnostic envelope and explicit producer/executor sources required",
        )?;
        execution_commit(&self.execution_source_commit)?;
        need(
            absolute_plain(&self.artifact_root),
            "explicit plain public asset root required",
        )?;
        need(
            scope.source_commit == self.initializer_source_commit,
            "scope retains the actual original initializer source",
        )?;
        need(
            scope.mode == PalsEngineModeV5::Deployment
                && scope.model.artifact_domain == PalsArtifactDomainV5::Untrained
                && scope.model.profile == "virtual_review_v1"
                && scope.model.backend == PalsModelBackendV3::OrtCuda
                && scope.model.precision == PalsPrecisionV3::Fp32
                && !scope.model.tf32,
            "original G0, V-free Deployment, CUDA FP32 and TF32-off required",
        )?;
        executor_components(&self.execution.source_components, &scope.source_components)?;
        scope.validate_pre_acceptance()?;
        let exact = |purpose, path: &Path, digest: &str| -> Result<()> {
            let a = &scope.asset(purpose)?.artifact;
            need(
                self.artifact_root.join(&a.path) == path && a.sha256 == digest,
                "standalone path/hash differs from this exact numerical scope",
            )
        };
        exact(
            Purpose::ExportManifest,
            &self.execution.export_manifest,
            &self.execution.export_manifest_sha256,
        )?;
        exact(
            Purpose::RuntimeLibrary,
            &self.execution.runtime_library,
            &self.execution.runtime_library_sha256,
        )?;
        need(
            scope.model.weights.sha256 == self.execution.checkpoint_sha256,
            "same actual checkpoint bytes required",
        )?;
        let export = &scope.asset(Purpose::ExportManifest)?.artifact;
        need(
            export
                .source
                .ends_with(&format!("/tree/{}", self.export_producer_source_commit)),
            "export reference preserves its declared original producer source",
        )?;
        exact(
            Purpose::CudaManifest,
            self.execution
                .cuda_bundle
                .as_deref()
                .ok_or("CUDA bundle absent")?,
            self.execution
                .cuda_bundle_sha256
                .as_deref()
                .ok_or("CUDA bundle hash absent")?,
        )?;
        exact(
            Purpose::CudaControlInventory,
            self.execution
                .cuda_inventory
                .as_deref()
                .ok_or("CUDA inventory absent")?,
            self.execution
                .cuda_inventory_sha256
                .as_deref()
                .ok_or("CUDA inventory hash absent")?,
        )?;
        scope.asset(Purpose::NumericalAcceptance)?;
        need(
            scope
                .assets
                .iter()
                .any(|a| a.purpose == Purpose::NumericalRawEvidence),
            "actual current-executor independent math evidence absent",
        )
    }

    pub(super) fn prepare(&self, until: Instant) -> Result<PreparedPrerequisite> {
        deadline(until)?;
        self.check_identity()?;
        let assets = HeldAssets::open(&self.artifact_root, &self.numerical_scope, until)?;
        let export = assets.read(
            &self
                .numerical_scope
                .asset(Purpose::ExportManifest)?
                .artifact,
            256 * 1024,
            until,
        )?;
        let producer = validate_pals_scope_artifact_domain_v5(
            &self.numerical_scope,
            &export,
            |artifact, limit| assets.read(artifact, limit, until).map_err(metadata_error),
        )?;
        need(
            producer.initializer_source_commit.as_deref()
                == Some(self.initializer_source_commit.as_str()),
            "original producer metadata did not verify this initializer",
        )?;
        let acceptance = assets.read(
            &self
                .numerical_scope
                .asset(Purpose::NumericalAcceptance)?
                .artifact,
            4 * 1024 * 1024,
            until,
        )?;
        let numerical = validate_pals_numerical_scope_acceptance_v5(
            &self.numerical_scope,
            &acceptance,
            |artifact, limit| assets.read(artifact, limit, until).map_err(metadata_error),
        )?;
        deadline(until)?;
        let observation = json!({"schema":"rz-explicit-gpu-inflight-prerequisite-observation/1",
            "execution_source_commit":self.execution_source_commit,
            "initializer_source_commit":self.initializer_source_commit,
            "export_producer_source_commit":self.export_producer_source_commit,
            "source_components":self.execution.source_components,"producer_metadata":producer,
            "prior_independent_math_admission":numerical,"weights":self.execution.checkpoint_sha256,
            "export":self.execution.export_manifest_sha256,"new_initializer_or_export_executed":false,
            "completed_startup_or_arena_authority":false,"this_invocation_GPU_completion":false,
            "runtime_library_copy_and_pins":"original_RuntimeCache_authority_unchanged"});
        let manifest = self
            .numerical_scope
            .asset(Purpose::ExportManifest)?
            .artifact
            .clone();
        let parent = Path::new(&manifest.path)
            .parent()
            .ok_or("export parent absent")?;
        let mut graphs = BTreeMap::new();
        for (purpose, role) in [
            (Purpose::PublicGraph, "public"),
            (Purpose::SharedPcGraph, "shared_pc"),
        ] {
            let graph = self.numerical_scope.asset(purpose)?.artifact.clone();
            need(
                Path::new(&graph.path).parent() == Some(parent),
                "actual graphs retain their original manifest-relative parent",
            )?;
            let filename = Path::new(&graph.path)
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or("graph name absent")?
                .to_owned();
            graphs.insert(role.to_owned(), (filename, graph));
        }
        Ok(PreparedPrerequisite {
            assets,
            observation,
            manifest,
            graphs,
            read_until: until,
            cuda_manifest: self
                .numerical_scope
                .asset(Purpose::CudaManifest)?
                .artifact
                .clone(),
            cuda_inventory: self
                .numerical_scope
                .asset(Purpose::CudaControlInventory)?
                .artifact
                .clone(),
        })
    }
}

fn metadata_error(error: Box<dyn std::error::Error>) -> ManifestError {
    ManifestError::Artifact(format!("held diagnostic prerequisite: {error}"))
}

pub(super) struct PreparedPrerequisite {
    assets: HeldAssets,
    pub(super) observation: Value,
    manifest: ArtifactRef,
    graphs: BTreeMap<String, (String, ArtifactRef)>,
    read_until: Instant,
    cuda_manifest: ArtifactRef,
    cuda_inventory: ArtifactRef,
}
impl PreparedPrerequisite {
    pub(super) fn recheck(&self, until: Instant) -> Result<()> {
        self.assets.recheck(until)
    }
    pub(super) fn cuda_metadata(&self) -> Result<(Vec<u8>, Vec<u8>)> {
        Ok((
            self.assets
                .read(&self.cuda_manifest, 64 * 1024, self.read_until)?,
            self.assets
                .read(&self.cuda_inventory, 4 * 1024 * 1024, self.read_until)?,
        ))
    }
}

// This exact Arc is passed into the original model owner. Its graph and
// manifest reads use these same retained descriptors, never ambient PathSources.
impl rz_eval::pals_onnx::virtual_review::VirtualReviewOwnedSources for PreparedPrerequisite {
    fn read_manifest(
        &self,
        max_bytes: usize,
    ) -> std::result::Result<Vec<u8>, rz_eval::error::BackendError> {
        self.assets
            .read(&self.manifest, max_bytes, self.read_until)
            .map_err(native_source_error)
    }
    fn read_graph(
        &self,
        role: &str,
        registered_file: &str,
        expected_sha256: [u8; 32],
        max_bytes: usize,
    ) -> std::result::Result<Vec<u8>, rz_eval::error::BackendError> {
        let read = || -> Result<Vec<u8>> {
            let (filename, artifact) = self
                .graphs
                .get(role)
                .ok_or("graph role outside original retained scope")?;
            need(
                filename == registered_file
                    && artifact.sha256
                        == expected_sha256
                            .iter()
                            .map(|b| format!("{b:02x}"))
                            .collect::<String>(),
                "native graph role/file/hash differs from the original held scope",
            )?;
            self.assets.read(artifact, max_bytes, self.read_until)
        };
        read().map_err(native_source_error)
    }
}
fn native_source_error(error: Box<dyn std::error::Error>) -> rz_eval::error::BackendError {
    rz_eval::error::BackendError::new(
        rz_eval::error::FailureKind::InvalidInput,
        rz_eval::error::FailureStage::Asset,
        "held prerequisite source unavailable",
    )
    .with_diagnostic("HeldPrerequisite", &error.to_string())
}

#[cfg(target_os = "linux")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct Identity(u64, u64, u64, u32, u64, u32, u32, i64, i64, i64, i64);
#[cfg(target_os = "linux")]
fn identity(file: &File) -> Result<Identity> {
    use std::os::unix::fs::MetadataExt;
    let m = file.metadata()?;
    need(m.is_file(), "retained prerequisite must be a regular file")?;
    Ok(Identity(
        m.dev(),
        m.ino(),
        m.len(),
        m.mode(),
        m.nlink(),
        m.uid(),
        m.gid(),
        m.mtime(),
        m.mtime_nsec(),
        m.ctime(),
        m.ctime_nsec(),
    ))
}

#[cfg(target_os = "linux")]
struct HeldAsset {
    artifact: ArtifactRef,
    parent: Dir,
    name: String,
    file: File,
    identity: Identity,
}
#[cfg(target_os = "linux")]
impl HeldAsset {
    fn check_name(&self) -> Result<()> {
        use cap_std::fs::MetadataExt;
        let name = self.parent.symlink_metadata(&self.name)?;
        need(
            name.is_file()
                && !name.file_type().is_symlink()
                && name.dev() == self.identity.0
                && name.ino() == self.identity.1
                && name.len() == self.identity.2
                && identity(&self.file)? == self.identity,
            "held prerequisite name or descriptor changed",
        )
    }
    fn stream(&self, maximum: usize, collect: bool, until: Instant) -> Result<Vec<u8>> {
        use std::os::unix::fs::FileExt;
        deadline(until)?;
        self.check_name()?;
        need(
            self.artifact.bytes <= maximum as u64,
            "bounded prerequisite read exceeded",
        )?;
        let mut bytes = Vec::new();
        let mut hash = Sha256::new();
        let mut offset = 0u64;
        let mut buffer = [0u8; 64 * 1024];
        while offset < self.artifact.bytes {
            deadline(until)?;
            let limit = usize::try_from((self.artifact.bytes - offset).min(buffer.len() as u64))?;
            let count = self.file.read_at(&mut buffer[..limit], offset)?;
            need(count != 0, "prerequisite truncated")?;
            offset += count as u64;
            hash.update(&buffer[..count]);
            if collect {
                bytes.extend_from_slice(&buffer[..count]);
            }
        }
        need(
            self.file.read_at(&mut buffer[..1], offset)? == 0
                && format!("{:x}", hash.finalize()) == self.artifact.sha256,
            "held prerequisite hash or EOF differs",
        )?;
        self.check_name()?;
        deadline(until)?;
        Ok(bytes)
    }
}

#[cfg(target_os = "linux")]
struct HeldAssets {
    _root: Dir,
    files: BTreeMap<String, HeldAsset>,
}
#[cfg(target_os = "linux")]
impl HeldAssets {
    fn open(root: &Path, scope: &PalsNumericalScopeV5, until: Instant) -> Result<Self> {
        let mut directory = Dir::open_ambient_dir("/", cap_std::ambient_authority())?;
        for part in root.components() {
            if let Component::Normal(name) = part {
                directory = directory.open_dir_nofollow(name)?;
            }
        }
        let mut files = BTreeMap::new();
        for declared in &scope.assets {
            deadline(until)?;
            declared.artifact.validate()?;
            let mut parent = directory.try_clone()?;
            let parts = declared.artifact.path.split('/').collect::<Vec<_>>();
            let (name, ancestors) = parts.split_last().ok_or("empty actual artifact path")?;
            for part in ancestors {
                parent = parent.open_dir_nofollow(part)?;
            }
            let mut local = declared.artifact.clone();
            local.path = (*name).to_owned();
            let file = local.open_verified_at(&parent, MAX_BYTES)?;
            let stamp = identity(&file)?;
            let held = HeldAsset {
                artifact: declared.artifact.clone(),
                parent,
                name: (*name).to_owned(),
                file,
                identity: stamp,
            };
            held.check_name()?;
            deadline(until)?;
            need(
                files.insert(declared.artifact.path.clone(), held).is_none(),
                "duplicate held asset path",
            )?;
        }
        Ok(Self {
            _root: directory,
            files,
        })
    }
    fn read(&self, artifact: &ArtifactRef, limit: usize, until: Instant) -> Result<Vec<u8>> {
        let file = self
            .files
            .get(&artifact.path)
            .ok_or("metadata reference outside retained actual scope")?;
        need(
            file.artifact == *artifact,
            "same path has different actual metadata",
        )?;
        file.stream(limit, true, until)
    }
    fn recheck(&self, until: Instant) -> Result<()> {
        for file in self.files.values() {
            file.stream(usize::try_from(MAX_BYTES)?, false, until)?;
        }
        Ok(())
    }
}

#[cfg(not(target_os = "linux"))]
struct HeldAssets;
#[cfg(not(target_os = "linux"))]
impl HeldAssets {
    fn open(_: &Path, _: &PalsNumericalScopeV5, _: Instant) -> Result<Self> {
        Err("retained prerequisite requires Linux DirFD/positional reads".into())
    }
    fn read(&self, _: &ArtifactRef, _: usize, _: Instant) -> Result<Vec<u8>> {
        Err("unsupported held metadata reader".into())
    }
    fn recheck(&self, _: Instant) -> Result<()> {
        Err("unsupported held descriptor recheck".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn producer_and_executor_commits_are_independent_and_never_rewritten() {
        assert!(commit("5341d90f250d9630f852897dc3efb8d1c252755f"));
        assert!(commit("e63e8798687e827f28a9ddb8bc8861f483bf1500"));
        assert!(!commit("G"));
        assert!(!commit(&"A".repeat(40)));
        let registered = env!("RZ_PALS_COMPILED_SOURCE_COMMIT");
        assert_eq!(execution_commit(registered).is_ok(), commit(registered));
        let other = if registered == "a".repeat(40) {
            "b".repeat(40)
        } else {
            "a".repeat(40)
        };
        assert!(execution_commit(&other).is_err());
    }
    #[test]
    fn secret_and_nonabsolute_roots_are_rejected_before_any_io() {
        assert!(!absolute_plain(Path::new("relative/assets")));
        assert!(!absolute_plain(Path::new("/root/.ssh/assets")));
        assert!(!absolute_plain(Path::new("/root/credentials/assets")));
        assert!(!absolute_plain(Path::new("/root/.env/assets")));
        assert!(absolute_plain(Path::new("/private/validation-assets")));
    }
    #[test]
    fn expired_prerequisite_does_not_receive_a_new_deadline() {
        assert!(deadline(Instant::now()).is_err());
    }
    #[test]
    fn old_or_incomplete_executor_components_never_bypass_the_compiled_gate() {
        let current = compiled_source_components_v5().expect("compiled public inventory");
        assert!(executor_components(&current, &current).is_ok());
        for key in ["model_native", "nn_runtime"] {
            let mut old = current.clone();
            old.insert(key.to_owned(), "0".repeat(64));
            assert!(executor_components(&old, &current).is_err());
            assert!(executor_components(&current, &old).is_err());
        }
        let mut missing = current.clone();
        missing.remove("model_export");
        assert!(executor_components(&current, &missing).is_err());
    }
}
