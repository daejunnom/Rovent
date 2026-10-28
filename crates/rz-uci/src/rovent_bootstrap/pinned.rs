//! Exact Rovent asset selection over retained, verified descriptor owners.
//! No legacy manifest or V5 success record is synthesized for this model.

use super::configuration::{ROVENT_STARTUP_CONFIGURATION_MAX_BYTES, StartupConfigurationDocument};
use super::{
    RoventStartupControl, RoventStartupTopology, admitted_topology,
    compiled_bootstrap_source_digest, fail, hex,
};
use crate::pals_adaptive_bootstrap::runtime_assets::{
    RuntimeAssetDeclaration, RuntimeAssetKind, RuntimeAssetView, RuntimeBackend,
    admit_runtime_roster, pin_retained_runtime,
};
use crate::pals_native::virtual_encoding::validate_pals_virtual_rules_descriptor_bytes;
use cap_std::fs::Dir;
use rz_contracts::Digest;
use rz_eval::asset::parse_sha256;
use rz_eval::error::{BackendError, FailureKind as K};
use rz_eval::pals_onnx::virtual_review::{
    VirtualReviewArtifactKind, VirtualReviewExecutionMode, VirtualReviewOwnedSources,
    validate_virtual_review_export_manifest_kind,
};
use rz_eval::runtime_pin::{RuntimeBundleFileRole, RuntimeLibraryPin};
use rz_experiments::{ArtifactRef, RoventInputLock, RoventInputLockV2};
use sha2::{Digest as _, Sha256};
use std::{
    collections::BTreeSet,
    fs::File,
    path::{Component, Path, PathBuf},
    sync::Arc,
};
mod collection;
mod external_r;
pub use external_r::{
    RoventExternalRAssetSelection, RoventExternalRSourceCustody, RoventExternalRSourceSelection,
    RoventPinnedExternalRSource,
};
mod nnuec;
pub use collection::{
    RoventCollectionPreloadSources, RoventCollectionTeacherSourceCustody,
    RoventCollectionTeacherSources, RoventTeacherAssetSelection, RoventTeacherSourceSelection,
};
use nnuec::PinnedNnuecArtifacts;

const MAX_GRAPH_BYTES: u64 = 1 << 30;

/// Original CUDA inventory metadata bytes and their declared file SHA-256.
/// Reading these bytes does not create a runtime or physical completion proof.
pub(crate) type CudaInventoryBytesAndSha256 = (Vec<u8>, [u8; 32]);
const MAX_WEIGHTS_BYTES: u64 = 8 << 30;
const MAX_PROVENANCE_BYTES: u64 = 16 << 20;

#[derive(Clone, Debug)]
pub struct RoventNativeGraphAsset {
    pub role: String,
    /// The original export basename, distinct from the private snapshot path.
    pub registered_file: String,
    pub artifact: ArtifactRef,
}
#[derive(Clone, Debug)]
pub struct RoventNativeAssetSelection {
    pub startup_configuration: ArtifactRef,
    pub export_manifest: ArtifactRef,
    pub rules_descriptor: ArtifactRef,
    pub graphs: Vec<RoventNativeGraphAsset>,
    pub cuda_control_inventory: Option<ArtifactRef>,
    /// A bundle manifest is separate from the exact regular-library roster.
    pub runtime_bundle_manifest: Option<ArtifactRef>,
}

struct PinnedBlob {
    declaration: ArtifactRef,
    file: File,
}
struct PinnedGraph {
    role: String,
    registered_file: String,
    blob: PinnedBlob,
}

/// The /2 lock and exact /4 startup file remain owned together. This is a
/// startup admission only: it does not pin model/runtime assets, construct a
/// worker, issue an R job, or attest execution. No /1 lock is synthesized.
pub struct RoventPinnedStartupV2 {
    lock: Arc<RoventInputLockV2>,
    _root: Dir,
    blob: PinnedBlob,
    configuration: StartupConfigurationDocument,
}

impl RoventPinnedStartupV2 {
    pub fn open(
        lock: Arc<RoventInputLockV2>,
        root: Dir,
        selected: ArtifactRef,
        control: &RoventStartupControl,
    ) -> Result<Self, BackendError> {
        control.check()?;
        lock.validate()
            .map_err(|_| fail(K::IdentityMismatch, "invalid Rovent input lock /2"))?;
        if selected != lock.manifest.startup_document {
            return Err(fail(
                K::IdentityMismatch,
                "selected startup differs from /2 lock",
            ));
        }
        let blob = open_blob(
            &root,
            selected,
            ROVENT_STARTUP_CONFIGURATION_MAX_BYTES as u64,
        )?;
        let configuration = StartupConfigurationDocument::from_json(&read_blob(
            &blob,
            ROVENT_STARTUP_CONFIGURATION_MAX_BYTES,
        )?)?;
        configuration.admit_against_v2(&lock)?;
        control.check()?;
        Ok(Self {
            lock,
            _root: root,
            blob,
            configuration,
        })
    }

    pub fn lock(&self) -> &RoventInputLockV2 {
        &self.lock
    }

    pub fn configuration(&self) -> &StartupConfigurationDocument {
        &self.configuration
    }

    pub fn configuration_artifact(&self) -> &ArtifactRef {
        &self.blob.declaration
    }

    /// Re-read the retained descriptor, re-check bytes and the typed lock.
    /// As with the /1 owner, same-UID mutation prevention remains the caller's
    /// responsibility; pathname or metadata checks alone do not freeze bytes.
    pub fn revalidate(&self) -> Result<(), BackendError> {
        if self.blob.declaration != self.lock.manifest.startup_document {
            return Err(fail(
                K::IdentityMismatch,
                "retained startup differs from /2 lock",
            ));
        }
        let current = StartupConfigurationDocument::from_json(&read_blob(
            &self.blob,
            ROVENT_STARTUP_CONFIGURATION_MAX_BYTES,
        )?)?;
        current.admit_against_v2(&self.lock)?;
        if current != self.configuration {
            return Err(fail(K::IdentityMismatch, "retained startup /4 changed"));
        }
        Ok(())
    }
}

// The owner retains the original typed lock. Implementations are sealed so a
// caller cannot substitute a declaration-only object for actual admission.
mod native_lock_sealed {
    pub trait Sealed {}
}

#[doc(hidden)]
pub trait NativeInputLock: native_lock_sealed::Sealed + Send + Sync + 'static {
    fn engine(&self) -> &rz_experiments::RoventEngineDeclaration;
    fn provenance(&self) -> &[ArtifactRef];
    fn canonical_sha256(&self) -> &str;
    fn cleanup_max_ms(&self) -> u64;
    fn role_tickets_max(&self) -> u32;
    fn topology(
        &self,
        collection: Option<&RoventCollectionPreloadSources>,
    ) -> Result<RoventStartupTopology, BackendError>;
    fn admit_configuration(
        &self,
        config: &StartupConfigurationDocument,
    ) -> Result<(), BackendError>;
    fn admit_startup_artifact(&self, artifact: &ArtifactRef) -> Result<(), BackendError>;
}

impl native_lock_sealed::Sealed for RoventInputLock {}
impl NativeInputLock for RoventInputLock {
    fn engine(&self) -> &rz_experiments::RoventEngineDeclaration {
        &self.manifest.engine
    }
    fn provenance(&self) -> &[ArtifactRef] {
        &self.manifest.provenance
    }
    fn canonical_sha256(&self) -> &str {
        &self.canonical_sha256
    }
    fn cleanup_max_ms(&self) -> u64 {
        self.manifest.cleanup_max_ms
    }
    fn role_tickets_max(&self) -> u32 {
        self.manifest.coordinator.role_tickets_max
    }
    fn topology(
        &self,
        collection: Option<&RoventCollectionPreloadSources>,
    ) -> Result<RoventStartupTopology, BackendError> {
        match collection {
            Some(original) => super::admitted_collection_topology(self, original),
            None => admitted_topology(self),
        }
    }
    fn admit_configuration(
        &self,
        config: &StartupConfigurationDocument,
    ) -> Result<(), BackendError> {
        config.admit_against(self)
    }
    fn admit_startup_artifact(&self, artifact: &ArtifactRef) -> Result<(), BackendError> {
        if self.provenance().contains(artifact) {
            Ok(())
        } else {
            Err(fail(
                K::IdentityMismatch,
                "selected native asset is absent or duplicated in the Rovent lock",
            ))
        }
    }
}

impl native_lock_sealed::Sealed for RoventInputLockV2 {}
impl NativeInputLock for RoventInputLockV2 {
    fn engine(&self) -> &rz_experiments::RoventEngineDeclaration {
        &self.manifest.engine
    }
    fn provenance(&self) -> &[ArtifactRef] {
        &self.manifest.provenance
    }
    fn canonical_sha256(&self) -> &str {
        &self.canonical_sha256
    }
    fn cleanup_max_ms(&self) -> u64 {
        self.manifest.cleanup_max_ms
    }
    fn role_tickets_max(&self) -> u32 {
        self.manifest.coordinator.role_tickets_max
    }
    fn topology(
        &self,
        collection: Option<&RoventCollectionPreloadSources>,
    ) -> Result<RoventStartupTopology, BackendError> {
        if collection.is_some() {
            return Err(fail(
                K::UnsupportedModel,
                "V2 Collection preload is not connected",
            ));
        }
        let topology = super::admitted_topology_v2(self)?;
        if topology != RoventStartupTopology::RolesOnly {
            return Err(fail(
                K::UnsupportedModel,
                "V2 researcher source is not connected",
            ));
        }
        Ok(topology)
    }
    fn admit_configuration(
        &self,
        config: &StartupConfigurationDocument,
    ) -> Result<(), BackendError> {
        config.admit_against_v2(self)
    }
    fn admit_startup_artifact(&self, artifact: &ArtifactRef) -> Result<(), BackendError> {
        if *artifact == self.manifest.startup_document {
            Ok(())
        } else {
            Err(fail(
                K::IdentityMismatch,
                "selected startup differs from the V2 lock",
            ))
        }
    }
}

/// Descriptors remain bound to the exact new lock. Path replacement cannot
/// redirect a later read. Participating owners must still prevent mutation of
/// these private files; metadata checks are not a hostile same-UID sandbox.
pub struct RoventPinnedNativeSources<L = RoventInputLock> {
    lock: Arc<L>,
    topology: RoventStartupTopology,
    _root: Dir,
    private_root_path: PathBuf,
    configuration_blob: PinnedBlob,
    configuration: StartupConfigurationDocument,
    export: PinnedBlob,
    rules: PinnedBlob,
    graphs: Vec<PinnedGraph>,
    control: Option<PinnedBlob>,
    _weights: PinnedBlob,
    _provenance: PinnedBlob,
    runtime: Vec<PinnedBlob>,
    runtime_bundle_manifest: Option<PinnedBlob>,
    nnuec: Option<PinnedNnuecArtifacts>,
    collection: Option<Arc<RoventCollectionPreloadSources>>,
}
impl RoventPinnedNativeSources {
    /// Pure byte/descriptor admission. No ORT runtime, model or R worker starts.
    pub fn open(
        lock: Arc<RoventInputLock>,
        root: Dir,
        private_root_path: PathBuf,
        selection: RoventNativeAssetSelection,
        control: &RoventStartupControl,
    ) -> Result<Self, BackendError> {
        Self::open_selected(lock, root, private_root_path, selection, control, None)
    }
    /// Explicit source-only Collection preload. The same full lock and all
    /// original teacher source descriptors are retained; no target is granted.
    pub fn open_collection(
        lock: Arc<RoventInputLock>,
        root: Dir,
        private_root_path: PathBuf,
        selection: RoventNativeAssetSelection,
        control: &RoventStartupControl,
        collection: Arc<RoventCollectionPreloadSources>,
    ) -> Result<Self, BackendError> {
        if !collection.matches_input_lock(&lock) {
            return Err(fail(
                K::IdentityMismatch,
                "Collection preload source lock differs",
            ));
        }
        Self::open_selected(
            lock,
            root,
            private_root_path,
            selection,
            control,
            Some(collection),
        )
    }
}

impl RoventPinnedNativeSources<RoventInputLockV2> {
    /// Pin native inputs against the original V2 lock. V2 Collection and R
    /// owners remain unsupported until their original typed paths are connected.
    pub fn open_v2(
        lock: Arc<RoventInputLockV2>,
        root: Dir,
        private_root_path: PathBuf,
        selection: RoventNativeAssetSelection,
        control: &RoventStartupControl,
    ) -> Result<Self, BackendError> {
        Self::open_selected(lock, root, private_root_path, selection, control, None)
    }
}

impl<L: NativeInputLock> RoventPinnedNativeSources<L> {
    fn open_selected(
        lock: Arc<L>,
        root: Dir,
        private_root_path: PathBuf,
        selection: RoventNativeAssetSelection,
        control: &RoventStartupControl,
        collection: Option<Arc<RoventCollectionPreloadSources>>,
    ) -> Result<Self, BackendError> {
        control.check()?;
        let topology = lock.topology(collection.as_deref())?;
        if !private_root_path.is_absolute() {
            return Err(fail(
                K::InvalidInput,
                "original private root path must be absolute",
            ));
        }
        if selection.graphs.len() != 2 {
            return Err(fail(
                K::UnsupportedModel,
                "Fresh V-free native startup needs exactly two original graphs",
            ));
        }
        let compiled = rz_experiments::compiled_source_components_v5().map_err(|_| {
            fail(
                K::IdentityMismatch,
                "compiled native source inventory unavailable",
            )
        })?;
        let model = &lock.engine().pc.proposer;
        if lock.engine().source_components.get("rovent_bootstrap")
            != Some(&hex(compiled_bootstrap_source_digest()))
            || compiled.get("model_native") != Some(&model.implementation_sha256)
            || lock.engine().source_components.get("model_native")
                != Some(&model.implementation_sha256)
        {
            return Err(fail(
                K::IdentityMismatch,
                "native and bootstrap source pins differ from compiled bytes",
            ));
        }
        let mut selected_paths = BTreeSet::new();
        lock.admit_startup_artifact(&selection.startup_configuration)?;
        selected_paths.insert(selection.startup_configuration.path.clone());
        for artifact in std::iter::once(&selection.export_manifest)
            .chain(std::iter::once(&selection.rules_descriptor))
            .chain(selection.graphs.iter().map(|graph| &graph.artifact))
            .chain(selection.cuda_control_inventory.iter())
            .chain(selection.runtime_bundle_manifest.iter())
        {
            if !lock.provenance().contains(artifact)
                || !selected_paths.insert(artifact.path.clone())
            {
                return Err(fail(
                    K::IdentityMismatch,
                    "selected native asset is absent or duplicated in the Rovent lock",
                ));
            }
        }
        let configuration_blob = open_blob(
            &root,
            selection.startup_configuration,
            ROVENT_STARTUP_CONFIGURATION_MAX_BYTES as u64,
        )?;
        let configuration = StartupConfigurationDocument::from_json(&read_blob(
            &configuration_blob,
            ROVENT_STARTUP_CONFIGURATION_MAX_BYTES,
        )?)?;
        lock.admit_configuration(&configuration)?;
        control.check()?;
        let export = open_blob(&root, selection.export_manifest, 256 << 10)?;
        control.check()?;
        let rules = open_blob(&root, selection.rules_descriptor, 64 << 10)?;
        control.check()?;
        let mut graphs = Vec::new();
        let mut roles = BTreeSet::new();
        for graph in selection.graphs {
            if !matches!(graph.role.as_str(), "public" | "shared_pc")
                || !roles.insert(graph.role.clone())
                || !basename(&graph.registered_file)
            {
                return Err(fail(
                    K::IdentityMismatch,
                    "native graph role/basename is unsupported or duplicated",
                ));
            }
            graphs.push(PinnedGraph {
                role: graph.role,
                registered_file: graph.registered_file,
                blob: open_blob(&root, graph.artifact, MAX_GRAPH_BYTES)?,
            });
            control.check()?;
        }
        let control_blob = selection
            .cuda_control_inventory
            .map(|artifact| open_blob(&root, artifact, 4 << 20))
            .transpose()?;
        let weights = open_blob(&root, model.weights.clone(), MAX_WEIGHTS_BYTES)?;
        control.check()?;
        let provenance = open_blob(&root, model.provenance.clone(), MAX_PROVENANCE_BYTES)?;
        control.check()?;
        let runtime_bundle_manifest = selection
            .runtime_bundle_manifest
            .map(|artifact| {
                if !rz_experiments::pals_private_snapshot_path_v5(&artifact.path) {
                    return Err(fail(
                        K::InvalidInput,
                        "runtime bundle requires the closed private snapshot namespace",
                    ));
                }
                open_blob(&root, artifact, 64 << 10)
            })
            .transpose()?;
        let mut runtime = Vec::new();
        for artifact in &lock.engine().pc.runtime_artifacts {
            if !rz_experiments::pals_private_snapshot_path_v5(&artifact.path) {
                return Err(fail(
                    K::InvalidInput,
                    "runtime library requires the closed private snapshot namespace",
                ));
            }
            if !selected_paths.insert(artifact.path.clone()) {
                return Err(fail(
                    K::IdentityMismatch,
                    "runtime roster duplicates another selected artifact",
                ));
            }
            runtime.push(open_blob(&root, artifact.clone(), 1 << 30)?);
            control.check()?;
        }
        if matches!(lock.engine().pc.device, rz_experiments::RoventDevice::Cpu)
            != runtime_bundle_manifest.is_none()
        {
            return Err(fail(
                K::IdentityMismatch,
                "explicit runtime bundle manifest/provider mismatch",
            ));
        }
        let nnuec = PinnedNnuecArtifacts::open(&root, lock.engine().nnuec.as_ref(), control)?;
        let owner = Self {
            lock,
            topology,
            _root: root,
            private_root_path,
            configuration_blob,
            configuration,
            export,
            rules,
            graphs,
            control: control_blob,
            _weights: weights,
            _provenance: provenance,
            runtime,
            runtime_bundle_manifest,
            nnuec,
            collection,
        };
        owner.validate_lock()?;
        control.check()?;
        Ok(owner)
    }
    pub fn lock(&self) -> &L {
        &self.lock
    }
    /// Borrow the original allocation; equal bytes in a new Arc are not custody.
    pub(crate) fn lock_owner(&self) -> &Arc<L> {
        &self.lock
    }
    pub fn collection_preload_sources(&self) -> Option<&Arc<RoventCollectionPreloadSources>> {
        self.collection.as_ref()
    }
    /// Read only the original retained metadata descriptor. Its ArtifactRef
    /// pins JSON bytes; a CUDA runtime bundle's binary-codec digest is separate.
    /// No library, model, runtime or worker is loaded by this read.
    pub fn runtime_bundle_publication_bytes(
        &self,
    ) -> Result<Option<(ArtifactRef, Vec<u8>)>, BackendError> {
        self.runtime_bundle_manifest
            .as_ref()
            .map(|blob| read_blob(blob, 64 * 1024).map(|bytes| (blob.declaration.clone(), bytes)))
            .transpose()
    }
    pub fn topology(&self) -> RoventStartupTopology {
        self.topology
    }
    pub fn configuration(&self) -> &StartupConfigurationDocument {
        &self.configuration
    }
    pub fn configuration_artifact(&self) -> &ArtifactRef {
        &self.configuration_blob.declaration
    }
    pub fn private_root_path(&self) -> &Path {
        &self.private_root_path
    }
    /// The same generic retained-FD loader serves V5 and Rovent. This owner
    /// issues the roster from its original lock/FDs; callers provide no path
    /// or verified flag that could bypass the root/named-file guards.
    pub fn pin_runtime(&self) -> Result<RuntimeLibraryPin, BackendError> {
        self.validate_lock()?;
        let backend = match self.lock.engine().pc.backend {
            rz_experiments::RoventBackend::OrtCpu => RuntimeBackend::OrtCpu,
            rz_experiments::RoventBackend::OrtCuda => RuntimeBackend::OrtCuda,
            _ => {
                return Err(fail(
                    K::UnsupportedModel,
                    "native runtime requires explicit ORT selection",
                ));
            }
        };
        let kind = |artifact: &ArtifactRef| {
            if backend == RuntimeBackend::OrtCpu
                || Path::new(&artifact.path)
                    .file_name()
                    .and_then(|name| name.to_str())
                    == Some("libonnxruntime.so.1.22.0")
            {
                RuntimeAssetKind::Core
            } else {
                RuntimeAssetKind::CudaDependency
            }
        };
        let mut expected: Vec<_> = self
            .lock
            .engine()
            .pc
            .runtime_artifacts
            .iter()
            .map(|artifact| runtime_declaration(artifact, kind(artifact)))
            .collect();
        let mut held: Vec<_> = self
            .runtime
            .iter()
            .map(|blob| RuntimeAssetView {
                declaration: runtime_declaration(&blob.declaration, kind(&blob.declaration)),
                file: &blob.file,
            })
            .collect();
        if let Some(manifest) = &self.runtime_bundle_manifest {
            let original = self
                .lock
                .provenance()
                .iter()
                .find(|artifact| **artifact == manifest.declaration)
                .ok_or_else(|| {
                    fail(
                        K::IdentityMismatch,
                        "original runtime manifest provenance disappeared",
                    )
                })?;
            expected.push(runtime_declaration(
                original,
                RuntimeAssetKind::BundleManifest,
            ));
            held.push(RuntimeAssetView {
                declaration: runtime_declaration(
                    &manifest.declaration,
                    RuntimeAssetKind::BundleManifest,
                ),
                file: &manifest.file,
            });
        }
        let roster = admit_runtime_roster(backend, &expected, &held)?;
        let root = self
            ._root
            .try_clone()
            .map_err(|_| fail(K::Io, "cannot retain original runtime root capability"))?;
        let pin = pin_retained_runtime(&self.private_root_path, root, roster)?;
        self.validate_runtime(&pin)?;
        Ok(pin)
    }
    pub fn lock_digest(&self) -> Digest {
        // Construction has already validated this exact immutable lock string.
        Digest(parse_sha256(self.lock.canonical_sha256()).expect("admitted lock digest"))
    }
    pub fn export_digest_text(&self) -> String {
        self.export.declaration.sha256.clone()
    }
    pub fn weights_digest(&self) -> [u8; 32] {
        parse_sha256(&self._weights.declaration.sha256).expect("admitted weights digest")
    }
    pub fn cuda_inventory_digest(&self) -> Option<[u8; 32]> {
        self.control
            .as_ref()
            .map(|blob| parse_sha256(&blob.declaration.sha256).expect("admitted inventory digest"))
    }
    pub(crate) fn cuda_inventory_bytes(
        &self,
    ) -> Result<Option<CudaInventoryBytesAndSha256>, BackendError> {
        self.control
            .as_ref()
            .map(|blob| {
                Ok((
                    read_blob(blob, 4 << 20)?,
                    parse_sha256(&blob.declaration.sha256)?,
                ))
            })
            .transpose()
    }
    pub fn validate_lock(&self) -> Result<(), BackendError> {
        let topology = self.lock.topology(self.collection.as_deref())?;
        if topology != self.topology {
            return Err(fail(
                K::IdentityMismatch,
                "native topology changed after pinning",
            ));
        }
        let configuration = StartupConfigurationDocument::from_json(&read_blob(
            &self.configuration_blob,
            ROVENT_STARTUP_CONFIGURATION_MAX_BYTES,
        )?)?;
        self.lock
            .admit_startup_artifact(&self.configuration_blob.declaration)?;
        self.lock.admit_configuration(&configuration)?;
        if configuration != self.configuration {
            return Err(fail(
                K::IdentityMismatch,
                "original pinned startup configuration changed",
            ));
        }
        let raw = read_blob(&self.export, 256 << 10)?;
        let summary = validate_virtual_review_export_manifest_kind(
            &raw,
            VirtualReviewExecutionMode::Deployment,
            VirtualReviewArtifactKind::Fresh,
        )?;
        let rules_raw = read_blob(&self.rules, 64 << 10)?;
        let rules = validate_pals_virtual_rules_descriptor_bytes(&rules_raw)?;
        let model = &self.lock.engine().pc.proposer;
        if summary.checkpoint_sha256 != self.weights_digest()
            || summary.validator_present
            || summary.model_semantics != model.semantics
            || summary.encoding_schema != model.encoding
            || summary.graphs.len() != self.graphs.len()
            || summary.config != rules.config
            || summary.rules_input_semantic_sha256 != rules.semantic_sha256
            || summary.rules_encoder_source_sha256 != rules.encoder_source_sha256
            || summary.rules_profile_descriptor_sha256 != rules.descriptor_sha256()
        {
            return Err(fail(
                K::IdentityMismatch,
                "actual export, checkpoint and exact Rules descriptor disagree",
            ));
        }
        for expected in &summary.graphs {
            let matching = self
                .graphs
                .iter()
                .find(|graph| graph.role == expected.role)
                .ok_or_else(|| {
                    fail(
                        K::IdentityMismatch,
                        "native graph role missing from retained roster",
                    )
                })?;
            if matching.registered_file != expected.file
                || parse_sha256(&matching.blob.declaration.sha256)? != expected.sha256
            {
                return Err(fail(
                    K::IdentityMismatch,
                    "original graph identity differs from retained selected artifact",
                ));
            }
        }
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, pin: &RuntimeLibraryPin) -> Result<(), BackendError> {
        let artifacts = &self.lock.engine().pc.runtime_artifacts;
        match pin.bundle_files() {
            None => {
                if artifacts.len() != 1
                    || parse_sha256(&artifacts[0].sha256)? != pin.binary_digest()
                    || artifacts[0].bytes != pin.storage().bytes
                {
                    return Err(fail(
                        K::IdentityMismatch,
                        "actual CPU runtime pin differs from exact Rovent roster",
                    ));
                }
            }
            Some(files) => {
                if artifacts.len() != files.len() {
                    return Err(fail(
                        K::IdentityMismatch,
                        "CUDA runtime retained roster size differs",
                    ));
                }
                let mut matched = BTreeSet::new();
                for file in files {
                    let matches: Vec<_> = artifacts
                        .iter()
                        .enumerate()
                        .filter(|(_, artifact)| {
                            Path::new(&artifact.path)
                                .file_name()
                                .and_then(|name| name.to_str())
                                == Some(file.filename.as_str())
                                && artifact.sha256 == file.sha256
                                && artifact.bytes == file.bytes
                        })
                        .collect();
                    if matches.len() != 1
                        || !matched.insert(matches[0].0)
                        || (file.role == RuntimeBundleFileRole::Core
                            && parse_sha256(&file.sha256)? != pin.binary_digest())
                    {
                        return Err(fail(
                            K::IdentityMismatch,
                            "actual CUDA runtime file differs from original Rovent pin",
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}
impl<L: NativeInputLock> VirtualReviewOwnedSources for RoventPinnedNativeSources<L> {
    fn read_manifest(&self, maximum: usize) -> Result<Vec<u8>, BackendError> {
        read_blob(&self.export, maximum)
    }
    fn read_graph(
        &self,
        role: &str,
        registered_file: &str,
        expected: [u8; 32],
        maximum: usize,
    ) -> Result<Vec<u8>, BackendError> {
        let graph = self
            .graphs
            .iter()
            .find(|graph| graph.role == role)
            .ok_or_else(|| {
                fail(
                    K::IdentityMismatch,
                    "native owner requested an unpinned graph role",
                )
            })?;
        if graph.registered_file != registered_file
            || parse_sha256(&graph.blob.declaration.sha256)? != expected
        {
            return Err(fail(
                K::IdentityMismatch,
                "native owner requested a different original graph",
            ));
        }
        read_blob(&graph.blob, maximum)
    }
}

fn runtime_declaration(
    artifact: &ArtifactRef,
    kind: RuntimeAssetKind,
) -> RuntimeAssetDeclaration<'_> {
    RuntimeAssetDeclaration {
        artifact,
        snapshot_path: artifact.path.as_str(),
        kind,
    }
}
fn open_blob(
    root: &Dir,
    declaration: ArtifactRef,
    maximum: u64,
) -> Result<PinnedBlob, BackendError> {
    let file = declaration.open_verified_at(root, maximum).map_err(|_| {
        fail(
            K::IdentityMismatch,
            "cannot admit exact retained Rovent artifact",
        )
    })?;
    Ok(PinnedBlob { declaration, file })
}
fn basename(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 256
        && matches!(
            Path::new(name).components().collect::<Vec<_>>().as_slice(),
            [Component::Normal(_)]
        )
}
fn read_blob(blob: &PinnedBlob, maximum: usize) -> Result<Vec<u8>, BackendError> {
    let before = blob
        .file
        .metadata()
        .map_err(|_| fail(K::Io, "retained artifact metadata unavailable"))?;
    let length = usize::try_from(blob.declaration.bytes)
        .map_err(|_| fail(K::ResourceExhausted, "retained artifact size overflow"))?;
    if !before.is_file()
        || length == 0
        || length > maximum
        || before.len() != blob.declaration.bytes
    {
        return Err(fail(
            K::ResourceExhausted,
            "retained artifact exceeds exact bounded read",
        ));
    }
    let mut raw = Vec::new();
    raw.try_reserve_exact(length)
        .map_err(|_| fail(K::ResourceExhausted, "retained artifact allocation failed"))?;
    raw.resize(length, 0);
    #[cfg(windows)]
    let reader =
        rz_native_loader::pinned_reader::independent_read_handle(&blob.file).map_err(|_| {
            fail(
                K::Io,
                "retained artifact independent read handle unavailable",
            )
        })?;
    #[cfg(windows)]
    let reader = &reader;
    #[cfg(not(windows))]
    let reader = &blob.file;
    let mut at = 0;
    while at < length {
        let read = positional_read(reader, &mut raw[at..], at as u64)
            .map_err(|_| fail(K::Io, "retained artifact positional read failed"))?;
        if read == 0 {
            return Err(fail(
                K::IdentityMismatch,
                "retained artifact ended before its declared size",
            ));
        }
        at += read;
    }
    let after = blob
        .file
        .metadata()
        .map_err(|_| fail(K::Io, "retained artifact metadata disappeared"))?;
    if before.len() != after.len()
        || before.modified().ok() != after.modified().ok()
        || <[u8; 32]>::from(Sha256::digest(&raw)) != parse_sha256(&blob.declaration.sha256)?
    {
        return Err(fail(
            K::IdentityMismatch,
            "retained artifact changed during its exact read",
        ));
    }
    Ok(raw)
}
#[cfg(unix)]
fn positional_read(file: &File, out: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(out, offset)
}
#[cfg(windows)]
fn positional_read(file: &File, out: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_read(out, offset)
}
#[cfg(not(any(unix, windows)))]
fn positional_read(_: &File, _: &mut [u8], _: u64) -> std::io::Result<usize> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "positional asset read unavailable",
    ))
}
