//! Explicit /3 system-CUDA child entry. Historical /1 and /2 inputs are never
//! projected into this path. Runtime and process owners remain distinct.

use super::*;
#[cfg(target_os = "linux")]
use crate::rovent_bootstrap::process_boundary::{
    RoventProcessBoundaryInputsV3, SourceEvidencePolicy, VerifiedOwnedRoventProcessBoundaryV3,
    verify_actual_owned_process_boundary_v3_with_policy,
};
#[cfg(target_os = "linux")]
use rz_experiments::NativeRuntimeSourceV6;
use rz_experiments::RoventInputLockV3;

/// Select only original lock entries. No independent mutable index is needed;
/// `open_v3` still opens and verifies every original asset and validates the
/// export/Rules/graph/weights relationship before starting the native owner.
#[cfg(target_os = "linux")]
fn direct_selection(
    lock: &RoventInputLockV3,
    selection: &DirectNativeAssets,
) -> Result<RoventNativeAssetSelection, RoventLaunchError> {
    let original = |path: &str| -> Result<ArtifactRef, RoventLaunchError> {
        let mut matches = lock
            .manifest
            .provenance
            .iter()
            .filter(|item| item.path == path);
        let artifact = matches
            .next()
            .ok_or_else(|| error("Assets", "direct path is absent from original provenance"))?;
        if matches.next().is_some() {
            return Err(error(
                "Assets",
                "direct path is ambiguous in original provenance",
            ));
        }
        Ok(artifact.clone())
    };
    let graph = |role: &str,
                 path: &str,
                 original_name: &Option<String>|
     -> Result<RoventNativeGraphAsset, RoventLaunchError> {
        let registered_file = match original_name {
            Some(name) => name.clone(),
            None => path
                .rsplit('/')
                .next()
                .ok_or_else(|| error("Assets", "graph basename is missing"))?
                .to_owned(),
        };
        Ok(RoventNativeGraphAsset {
            role: role.to_owned(),
            registered_file,
            artifact: original(path)?,
        })
    };
    let mut graphs = vec![
        graph(
            "public",
            &selection.public_graph,
            &selection.public_graph_name,
        )?,
        graph(
            "shared_pc",
            &selection.shared_pc_graph,
            &selection.shared_pc_graph_name,
        )?,
    ];
    match (
        lock.manifest.purpose == rz_experiments::RoventPurpose::Collection,
        &selection.validator_graph,
    ) {
        (true, Some(path)) => {
            graphs.push(graph("validator", path, &selection.validator_graph_name)?)
        }
        (false, None) => {}
        (true, None) => return Err(error("Assets", "V3 Collection requires a validator graph")),
        (false, Some(_)) => {
            return Err(error(
                "Assets",
                "validator graph is restricted to V3 Collection",
            ));
        }
    }
    Ok(RoventNativeAssetSelection {
        startup_configuration: lock.manifest.startup_document.clone(),
        export_manifest: original(&selection.export)?,
        rules_descriptor: original(&selection.rules)?,
        graphs,
        cuda_control_inventory: None,
        runtime_bundle_manifest: None,
    })
}

/// Pin the declared original executable beneath the already-owned root.
/// Hashing and mapped-executable identity are performed by the V3 process
/// boundary under the same original cancellation and startup deadline.
#[cfg(target_os = "linux")]
fn open_original_binary_v3(root: &Dir, artifact: &ArtifactRef) -> Result<File, RoventLaunchError> {
    use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::OpenOptionsExt;
    artifact
        .validate()
        .map_err(|failure| error("Binary", failure))?;
    let parts: Vec<_> = artifact.path.split('/').collect();
    let (filename, parents) = parts
        .split_last()
        .ok_or_else(|| error("Binary", "empty binary path"))?;
    let mut scoped = root
        .try_clone()
        .map_err(|failure| error("Binary", failure))?;
    for parent in parents {
        scoped = scoped
            .open_dir_nofollow(parent)
            .map_err(|failure| error("Binary", failure))?;
    }
    let before = scoped
        .symlink_metadata(filename)
        .map_err(|failure| error("Binary", failure))?;
    if !before.is_file() {
        return Err(error(
            "Binary",
            "original executable must be a regular file",
        ));
    }
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .read(true)
        .follow(FollowSymlinks::No)
        .custom_flags(nix::libc::O_NONBLOCK | nix::libc::O_NOFOLLOW);
    let file = scoped
        .open_with(filename, &options)
        .map_err(|failure| error("Binary", failure))?;
    let metadata = file
        .metadata()
        .map_err(|failure| error("Binary", failure))?;
    if !metadata.is_file() || metadata.len() != artifact.bytes || artifact.bytes == 0 {
        return Err(error(
            "Binary",
            "original executable length or file type differs from declaration",
        ));
    }
    Ok(file.into_std())
}

#[cfg(target_os = "linux")]
struct SystemReceiptWriter {
    directory: Dir,
}

#[cfg(target_os = "linux")]
impl SystemReceiptWriter {
    fn open(root: &Dir, path: &std::path::Path) -> std::io::Result<Self> {
        // Walk only the original private-root capability. The resource owner
        // creates the parent; this process exclusively creates its own leaf.
        let mut dir = root.try_clone()?;
        for part in path.components() {
            let Component::Normal(name) = part else {
                return Err(std::io::Error::other("invalid receipt path"));
            };
            dir = dir.open_dir_nofollow(name)?;
        }
        let leaf = format!("native-process-{}", std::process::id());
        dir.create_dir(&leaf)?;
        Ok(Self {
            directory: dir.open_dir_nofollow(&leaf)?,
        })
    }

    fn write(
        &self,
        name: &str,
        receipt: &rz_experiments::SystemRuntimeReceiptV6,
        lock: &rz_experiments::SystemRuntimeLockV6,
    ) -> Result<(), String> {
        let bytes = receipt
            .canonical_bytes(lock, receipt.process_id, receipt.process_generation)
            .map_err(|failure| failure.to_string())?;
        self.write_bytes(name, &bytes)
    }

    fn write_bytes(&self, name: &str, bytes: &[u8]) -> Result<(), String> {
        use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
        use std::io::Write;
        let mut options = cap_std::fs::OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No);
        let mut file = self
            .directory
            .open_with(name, &options)
            .map_err(|e| e.to_string())?;
        file.write_all(&bytes)
            .and_then(|_| file.sync_all())
            .map_err(|e| e.to_string())
    }
}

#[cfg(target_os = "linux")]
fn write_receipt(
    writer: Option<&SystemReceiptWriter>,
    name: &str,
    lock: &RoventInputLockV3,
    receipt: &rz_experiments::SystemRuntimeReceiptV6,
) {
    if let Some(writer) = writer {
        if let Err(failure) = writer.write(name, receipt, &lock.manifest.system_runtime_lock) {
            // Recording failures do not imply a GPU failure or permit early
            // owner release. A supervisor must mark missing evidence unknown.
            eprintln!("system runtime receipt warning: {failure}");
        }
    }
}

#[cfg(target_os = "linux")]
fn runtime_receipt(
    lock: &RoventInputLockV3,
    generation: u64,
    observation: Option<&super::super::RoventSystemReadyObservationV3>,
) -> rz_experiments::SystemRuntimeReceiptV6 {
    use rz_experiments::{PhysicalCompletionV6, ProviderReadinessV6, SystemRuntimeReceiptV6};
    SystemRuntimeReceiptV6 {
        domain: rz_experiments::SYSTEM_RUNTIME_RECEIPT_V6_DOMAIN.into(),
        lock_sha256: lock.manifest.system_runtime_lock.canonical_sha256.clone(),
        process_id: std::process::id(),
        process_generation: generation,
        observed_identity_sha256: observation
            .map(|value| super::super::hex(value.runtime_identity.0)),
        observed_libraries: observation
            .map(|value| value.libraries.clone())
            .unwrap_or_default(),
        unregistered_native_mappings: observation
            .and_then(|value| value.unregistered_native_mappings.clone()),
        // Provider setup is observed independently of mapping and inference.
        provider_ready: if observation.is_some_and(|value| value.provider_ready) {
            ProviderReadinessV6::Ready
        } else {
            ProviderReadinessV6::Unknown
        },
        physical_completion: PhysicalCompletionV6::Unknown,
    }
}

#[cfg(target_os = "linux")]
struct OriginalV3Inputs {
    lock: Arc<RoventInputLockV3>,
    boundary: Arc<VerifiedOwnedRoventProcessBoundaryV3>,
    sources: Arc<RoventPinnedNativeSources<RoventInputLockV3>>,
    control: RoventStartupControl,
    clock: ProcessClock,
    // A legacy selection index is retained through physical completion;
    // direct lock-backed selection has no separate index descriptor.
    selection_file: Option<File>,
    lock_file: File,
    receipt_writer: Option<SystemReceiptWriter>,
}

#[cfg(target_os = "linux")]
fn admit_original<'a>(
    arguments: &Arguments,
    lock: Arc<RoventInputLockV3>,
    root: Dir,
    cwd: PathBuf,
    lock_file: File,
    resource: Option<ResourceAdmissionV3<'a>>,
) -> Result<OriginalV3Inputs, RoventLaunchError> {
    if arguments.storage.is_some()
        || arguments.startup_diagnostics
        || arguments.lifecycle_diagnostics
        || arguments.lifecycle_diagnostics_v2
        || arguments.initial_stop_source_diagnostics
    {
        return Err(error(
            "InputLockUnsupportedV3",
            "selected legacy storage or diagnostic owner is not connected to /3",
        ));
    }
    if !matches!(
        lock.manifest.purpose,
        rz_experiments::RoventPurpose::Validation | rz_experiments::RoventPurpose::Deployment
    ) {
        return Err(error(
            "InputLockUnsupportedV3",
            "collection requires an original /3 collection owner",
        ));
    }
    let system = match &lock.manifest.system_runtime_lock.runtime {
        NativeRuntimeSourceV6::SystemCuda(system) => system,
        NativeRuntimeSourceV6::Cpu => {
            return Err(error(
                "InputLockUnsupportedV3",
                "this /3 route explicitly selects system CUDA",
            ));
        }
    };
    if arguments.startup_ms > lock.manifest.wall_time_max_ms
        || arguments.startup_ms > system.startup_ms
    {
        return Err(error(
            "StartupWindow",
            "startup exceeds the original /3 or system runtime window",
        ));
    }
    let until = Instant::now()
        .checked_add(Duration::from_millis(arguments.startup_ms))
        .ok_or_else(|| error("StartupWindow", "startup deadline overflow"))?;
    let cleanup_until = until
        .checked_add(Duration::from_millis(lock.manifest.cleanup_max_ms))
        .ok_or_else(|| error("StartupWindow", "cleanup deadline overflow"))?;
    let control = RoventStartupControl {
        until,
        cleanup_until,
        cancellation: Arc::new(AtomicBool::new(false)),
    };
    let clock = ProcessClock::new(ProcessEpoch(arguments.process_epoch));
    control
        .check()
        .map_err(|failure| error("Binary", failure))?;
    let binary = open_original_binary_v3(&root, &lock.manifest.engine.binary)?;
    control
        .check()
        .map_err(|failure| error("Binary", failure))?;
    let source_policy = if arguments.strict_source.unwrap_or(false) {
        SourceEvidencePolicy::StrictReproducibility
    } else {
        SourceEvidencePolicy::FunctionalWithSourceObservation
    };
    let boundary = verify_actual_owned_process_boundary_v3_with_policy(
        RoventProcessBoundaryInputsV3 {
            lock: Arc::clone(&lock),
            original_binary: binary,
            original_root: root,
            private_cwd: cwd.clone(),
            control: control.clone(),
        },
        source_policy,
    )
    .map_err(|rejected| error("ProcessBoundary", rejected.error))?;
    let root = boundary.original_root();
    let view = boundary
        .startup_view()
        .map_err(|failure| error("ProcessBoundary", failure))?;
    if let Some(admit_resource) = resource {
        control
            .check()
            .map_err(|failure| error("ResourceAdmission", failure))?;
        admit_resource(&lock)?;
        control
            .check()
            .map_err(|failure| error("ResourceAdmission", failure))?;
    }
    let receipt_writer = arguments
        .system_runtime_receipt_root
        .as_ref()
        .and_then(|path| match SystemReceiptWriter::open(root, path) {
            Ok(writer) => Some(writer),
            Err(failure) => {
                eprintln!("system runtime receipt warning: {failure}");
                None
            }
        });
    let (selection, selection_file) = if let Some(direct) = arguments.direct_native_assets.as_ref()
    {
        (direct_selection(&lock, direct)?, None)
    } else {
        let selected: Vec<_> = lock
            .manifest
            .provenance
            .iter()
            .filter(|artifact| artifact.path == arguments.native_assets)
            .collect();
        if selected.len() != 1 {
            return Err(error(
                "Assets",
                "selection is not a unique original provenance artifact",
            ));
        }
        let selection_file = selected[0]
            .open_verified_at(root, MAX_SELECTION_BYTES)
            .map_err(|failure| error("Assets", failure))?;
        let selection = SelectionWire::parse_with_graph_count(
            &read_bounded(&selection_file, MAX_SELECTION_BYTES)?,
            if lock.manifest.purpose == rz_experiments::RoventPurpose::Collection {
                3
            } else {
                2
            },
        )?;
        (selection, Some(selection_file))
    };
    let source_root = root.try_clone().map_err(|failure| error("Cwd", failure))?;
    let sources = Arc::new(
        if arguments.strict_source.unwrap_or(false) {
            RoventPinnedNativeSources::<RoventInputLockV3>::open_v3(
                Arc::clone(&lock),
                source_root,
                cwd,
                selection,
                &control,
            )
        } else {
            RoventPinnedNativeSources::<RoventInputLockV3>::open_v3_observational(
                Arc::clone(&lock),
                source_root,
                cwd,
                selection,
                &control,
            )
        }
        .map_err(|failure| error("Assets", failure))?,
    );
    sources
        .configuration()
        .require_supported_native_capabilities()
        .map_err(|failure| error("Capability", failure))?;
    let evidence = boundary.source_evidence();
    if evidence.commit_status != super::super::process_boundary::SourceEvidenceStatus::Verified
        || evidence.component_status
            != super::super::process_boundary::SourceEvidenceStatus::Verified
        || sources.source_evidence() != super::super::pinned::NativeSourceEvidence::Verified
    {
        eprintln!(
            "source provenance unverified: commit={:?} components={:?} native={:?}",
            evidence.commit_status,
            evidence.component_status,
            sources.source_evidence()
        );
    }
    if let Some(writer) = receipt_writer.as_ref() {
        // These fields describe source provenance only. The declared aggregate
        // is never represented as a digest of actual compiled bytes.
        let json = serde_json::json!({
            "domain": "rz-rovent-source-observation-v3/1",
            "process_id": std::process::id(),
            "process_generation": arguments.process_epoch,
            "policy": format!("{:?}", evidence.policy),
            "commit_status": format!("{:?}", evidence.commit_status),
            "component_status": format!("{:?}", evidence.component_status),
            "native_source_status": format!("{:?}", sources.source_evidence()),
            "declared_commit": evidence.declared_commit,
            "compiled_commit": evidence.compiled_commit,
            "declared_components": evidence.declared_components,
            "compiled_components": evidence.compiled_components,
            "declared_aggregate_sha256": evidence.declared_aggregate.map(|value| super::super::hex(value.0)),
        });
        if let Err(failure) =
            writer.write_bytes("source-observation.v3.json", json.to_string().as_bytes())
        {
            eprintln!("source observation warning: {failure}");
        }
    }
    boundary
        .revalidate_startup()
        .map_err(|failure| error("ProcessBoundary", failure))?;
    drop(view);
    Ok(OriginalV3Inputs {
        lock,
        boundary,
        sources,
        control,
        clock,
        selection_file,
        lock_file,
        receipt_writer,
    })
}

#[cfg(target_os = "linux")]
pub(super) fn run_native(
    arguments: Arguments,
    lock: Arc<RoventInputLockV3>,
    root: Dir,
    cwd: PathBuf,
    lock_file: File,
) -> Result<(), RoventLaunchError> {
    run_native_inner(arguments, lock, root, cwd, lock_file, None)
}

#[cfg(target_os = "linux")]
pub(super) fn run_native_with_resource<'a>(
    arguments: Arguments,
    lock: Arc<RoventInputLockV3>,
    root: Dir,
    cwd: PathBuf,
    lock_file: File,
    resource: ResourceAdmissionV3<'a>,
) -> Result<(), RoventLaunchError> {
    run_native_inner(arguments, lock, root, cwd, lock_file, Some(resource))
}

#[cfg(target_os = "linux")]
fn run_native_inner<'a>(
    arguments: Arguments,
    lock: Arc<RoventInputLockV3>,
    root: Dir,
    cwd: PathBuf,
    lock_file: File,
    resource: Option<ResourceAdmissionV3<'a>>,
) -> Result<(), RoventLaunchError> {
    let original = admit_original(&arguments, lock, root, cwd, lock_file, resource)?;
    let OriginalV3Inputs {
        lock,
        boundary,
        sources,
        control,
        clock,
        selection_file,
        lock_file,
        receipt_writer,
    } = original;
    // Each selected product receives its own original V3 owner. Collection
    // still requires its own V3 owner and remains separate.
    let nnuec_selection = match (
        sources.topology(),
        sources.configuration().fields().execution,
        sources.configuration().fields().nnuec_selection(),
    ) {
        (
            super::super::RoventStartupTopology::RolesOnly,
            ExecutionConfiguration::RolesOnly {
                shared_nnuec_readout: None,
                ..
            },
            NativeNnuecSelection::Absent,
        ) => NativeNnuecSelection::Absent,
        (
            super::super::RoventStartupTopology::RolesOnly,
            ExecutionConfiguration::RolesOnly {
                shared_nnuec_readout: Some(_),
                ..
            },
            NativeNnuecSelection::SharedPc,
        ) => NativeNnuecSelection::SharedPc,
        (
            super::super::RoventStartupTopology::OwnResearcher,
            ExecutionConfiguration::OwnResearcher {
                value: ResearchValueConfiguration::NnuecV2 { .. },
                ..
            },
            NativeNnuecSelection::OwnResearcher,
        ) => NativeNnuecSelection::OwnResearcher,
        (
            super::super::RoventStartupTopology::ExternalResearcher,
            ExecutionConfiguration::ExternalResearcher { .. },
            NativeNnuecSelection::Absent,
        ) => NativeNnuecSelection::Absent,
        _ => {
            return Err(error(
                "InputLockUnsupportedV3",
                "selected V3 product owner is not connected",
            ));
        }
    };
    // Choose the physical owner before starting the native worker, so no
    // missing-helper branch can drop an already-started GPU owner.
    // One startup owner is moved exactly once; do not add a heap allocation
    // solely to shrink this short-lived choice.
    #[allow(clippy::large_enum_variant)]
    enum ProductChoice {
        Pc,
        PcShared(LoadedRoventNnuecSource<RoventInputLockV3>),
        OwnPcr(LoadedRoventNnuecSource<RoventInputLockV3>),
        External(
            RoventPinnedExternalRSource<RoventInputLockV3>,
            LaunchRetirementEvidence,
        ),
    }
    let (external_source, external_selection_file, retirement_evidence) = match (
        sources.configuration().fields().execution,
        arguments.external_r_sources.as_deref(),
        arguments.retirement_sink.as_deref(),
        arguments.evidence_root.as_deref(),
    ) {
        (
            ExecutionConfiguration::ExternalResearcher { .. },
            Some(path),
            Some(registration),
            Some(root),
        ) => {
            sources
                .configuration()
                .external_question_limit()
                .map_err(|failure| error("ExternalQuestionPolicy", failure))?
                .ok_or_else(|| {
                    error("ExternalQuestionPolicy", "explicit external query required")
                })?;
            let mut matching = lock
                .manifest
                .provenance
                .iter()
                .filter(|artifact| artifact.path == path);
            let artifact = matching.next().ok_or_else(|| {
                error(
                    "ExternalSources",
                    "selection is absent from original provenance",
                )
            })?;
            if matching.next().is_some() {
                return Err(error(
                    "ExternalSources",
                    "selection is not a unique original provenance artifact",
                ));
            }
            let mapping = artifact
                .open_verified_at(boundary.original_root(), MAX_SELECTION_BYTES)
                .map_err(|failure| error("ExternalSources", failure))?;
            let selected =
                ExternalSelectionWire::parse(&read_bounded(&mapping, MAX_SELECTION_BYTES)?)?;
            let source = RoventPinnedExternalRSource::open(&sources, &selected, &control)
                .map_err(|failure| error("ExternalSources", failure))?;
            let evidence = LaunchRetirementEvidence::open_v3(
                Arc::clone(&boundary),
                registration,
                root,
                &control,
            )
            .map_err(|failure| error("RetirementEvidence", failure))?;
            (Some(source), Some(mapping), Some(evidence))
        }
        (ExecutionConfiguration::ExternalResearcher { .. }, _, _, _) => {
            return Err(error(
                "ExternalSources",
                "external researcher requires --external-r-sources, --retirement-sink and --evidence-root",
            ));
        }
        (_, None, None, None) => (None, None, None),
        _ => {
            return Err(error(
                "ExternalSources",
                "external source and retirement arguments require the external researcher",
            ));
        }
    };
    let product = match (nnuec_selection, external_source, retirement_evidence) {
        (NativeNnuecSelection::Absent, Some(external), Some(evidence)) => {
            ProductChoice::External(external, evidence)
        }
        (NativeNnuecSelection::Absent, None, None) => ProductChoice::Pc,
        (NativeNnuecSelection::SharedPc, None, None) => ProductChoice::PcShared(
            LoadedRoventNnuecSource::load(Arc::clone(&sources), &control)
                .map_err(|failure| error("NnuecSource", failure))?,
        ),
        (NativeNnuecSelection::OwnResearcher, None, None) => ProductChoice::OwnPcr(
            LoadedRoventNnuecSource::load(Arc::clone(&sources), &control)
                .map_err(|failure| error("NnuecSource", failure))?,
        ),
        _ => {
            return Err(error(
                "ExternalSources",
                "incompatible product and external owners",
            ));
        }
    };
    let system = match &lock.manifest.system_runtime_lock.runtime {
        NativeRuntimeSourceV6::SystemCuda(system) => system,
        NativeRuntimeSourceV6::Cpu => {
            return Err(error(
                "InputLockUnsupportedV3",
                "this /3 route explicitly selects system CUDA",
            ));
        }
    };
    let native = sources.configuration().fields().native;
    let resources = super::super::RoventSystemNativeResourcesV3 {
        configuration: native
            .onnx_system(&lock.manifest.engine.pc.device, system)
            .map_err(|failure| error("Configuration", failure))?,
        drain_limit: Duration::from_millis(native.drain_ms),
        tickets: native
            .tickets(sources.configuration().fields().review.token_budget)
            .map_err(|failure| error("Configuration", failure))?,
        command_capacity: native.command_capacity,
    };
    let diagnostic = NativeStartupDiagnostic::default();
    let owners = Arc::new(OwnerRegistry::default());
    let identity = EngineIdentity {
        name: format!("Rovent {:?}", lock.manifest.engine.mode),
        author: "Rovent contributors".into(),
    };
    let (driver, completion, observation): (Arc<dyn SearchSessionDriver>, _, _) = match product {
        ProductChoice::OwnPcr(loaded) => {
            let prepared = match super::super::prepare_native_v3::<AdaptivePortQuestion>(
                Arc::clone(&sources),
                resources,
                clock.runtime_clock(),
                control.clone(),
            ) {
                Ok(prepared) => prepared,
                Err(failure) => {
                    let failure = reject_factory(failure, control.cleanup_until, &diagnostic);
                    let unknown = runtime_receipt(&lock, arguments.process_epoch, None);
                    write_receipt(
                        receipt_writer.as_ref(),
                        "system-runtime-startup.v6.json",
                        &lock,
                        &unknown,
                    );
                    write_receipt(
                        receipt_writer.as_ref(),
                        "system-runtime-termination.v6.json",
                        &lock,
                        &unknown,
                    );
                    if failure.stage == "PhysicalCompletionUnknown" {
                        std::mem::forget((boundary, sources, selection_file, lock_file, loaded));
                    }
                    return Err(failure);
                }
            };
            let observation = prepared.snapshot().system_runtime.clone();
            let startup = runtime_receipt(&lock, arguments.process_epoch, observation.as_ref());
            write_receipt(
                receipt_writer.as_ref(),
                "system-runtime-startup.v6.json",
                &lock,
                &startup,
            );
            let driver = match super::super::product::RoventOwnNnuecProductDriverV3::own_nnuec_pcr(
                prepared,
                loaded,
                Arc::clone(&boundary),
                clock.clone(),
                Arc::clone(&owners),
                identity,
                arguments.root_policy,
            ) {
                Ok(driver) => driver,
                Err(mut failure) => {
                    let detail = error("ProductAssembly", &failure.failure);
                    let unknown =
                        runtime_receipt(&lock, arguments.process_epoch, observation.as_ref());
                    write_receipt(
                        receipt_writer.as_ref(),
                        "system-runtime-termination.v6.json",
                        &lock,
                        &unknown,
                    );
                    if !drain_native(
                        &mut failure.rejected.prepared.service,
                        control.cleanup_until,
                    ) {
                        std::mem::forget((failure, boundary, sources, selection_file, lock_file));
                        return Err(error("PhysicalCompletionUnknown", detail));
                    }
                    return Err(detail);
                }
            };
            let completion = driver.physical_close_observation();
            (Arc::new(driver), completion, observation)
        }
        ProductChoice::External(external, evidence) => {
            let prepared = match super::super::prepare_native_v3::<AdaptivePortQuestion>(
                Arc::clone(&sources),
                resources,
                clock.runtime_clock(),
                control.clone(),
            ) {
                Ok(prepared) => prepared,
                Err(failure) => {
                    let failure = reject_factory(failure, control.cleanup_until, &diagnostic);
                    let unknown = runtime_receipt(&lock, arguments.process_epoch, None);
                    write_receipt(
                        receipt_writer.as_ref(),
                        "system-runtime-startup.v6.json",
                        &lock,
                        &unknown,
                    );
                    write_receipt(
                        receipt_writer.as_ref(),
                        "system-runtime-termination.v6.json",
                        &lock,
                        &unknown,
                    );
                    if failure.stage == "PhysicalCompletionUnknown" {
                        std::mem::forget((
                            boundary,
                            sources,
                            selection_file,
                            external_selection_file,
                            lock_file,
                            external,
                            evidence,
                        ));
                    }
                    return Err(failure);
                }
            };
            let observation = prepared.snapshot().system_runtime.clone();
            let startup = runtime_receipt(&lock, arguments.process_epoch, observation.as_ref());
            write_receipt(
                receipt_writer.as_ref(),
                "system-runtime-startup.v6.json",
                &lock,
                &startup,
            );
            let driver = match super::super::product::RoventExternalPcrProductDriverV3::external_pcr(
                prepared,
                external,
                evidence,
                Arc::clone(&boundary),
                clock.clone(),
                Arc::clone(&owners),
                identity,
                arguments.root_policy,
            ) {
                Ok(driver) => driver,
                Err(mut failure) => {
                    use super::super::external_session::RoventExternalProductStartupOwnersV3;
                    use super::super::product::RoventExternalPcrV3RejectedOwners;
                    let detail = error("ProductAssembly", &failure.failure);
                    let unknown =
                        runtime_receipt(&lock, arguments.process_epoch, observation.as_ref());
                    write_receipt(
                        receipt_writer.as_ref(),
                        "system-runtime-termination.v6.json",
                        &lock,
                        &unknown,
                    );
                    let confirmed = match &mut failure.owners {
                        RoventExternalPcrV3RejectedOwners::Preflight { prepared, .. } => drain_native(&mut prepared.service, control.cleanup_until),
                        RoventExternalPcrV3RejectedOwners::Session { rejected, .. } => match &mut rejected.owners {
                            RoventExternalProductStartupOwnersV3::Prepared { prepared, .. }
                            | RoventExternalProductStartupOwnersV3::SourcePreparation { prepared, .. }
                            | RoventExternalProductStartupOwnersV3::PreparedSource { prepared, .. } => drain_native(&mut prepared.service, control.cleanup_until),
                            RoventExternalProductStartupOwnersV3::Session(rejected) => drain_native(&mut rejected.sources.native, control.cleanup_until),
                            RoventExternalProductStartupOwnersV3::Driver(driver) => driver.finish_session(control.cleanup_until).is_ok_and(|completion| completion == Some(crate::search_driver::DriverPhysicalCompletion::Confirmed)),
                        },
                    };
                    if !confirmed {
                        std::mem::forget((
                            failure,
                            boundary,
                            sources,
                            selection_file,
                            external_selection_file,
                            lock_file,
                        ));
                        return Err(error("PhysicalCompletionUnknown", detail));
                    }
                    return Err(detail);
                }
            };
            let completion = driver.physical_close_observation();
            (Arc::new(driver), completion, observation)
        }
        pc_product @ (ProductChoice::Pc | ProductChoice::PcShared(_)) => {
            let shared_loaded = match pc_product {
                ProductChoice::Pc => None,
                ProductChoice::PcShared(loaded) => Some(loaded),
                ProductChoice::OwnPcr(_) | ProductChoice::External(_, _) => {
                    unreachable!("matched P/C product")
                }
            };
            let prepared = match super::super::prepare_native_v3::<PcPortQuestion>(
                Arc::clone(&sources),
                resources,
                clock.runtime_clock(),
                control.clone(),
            ) {
                Ok(prepared) => prepared,
                Err(failure) => {
                    let failure = reject_factory(failure, control.cleanup_until, &diagnostic);
                    let unknown = runtime_receipt(&lock, arguments.process_epoch, None);
                    write_receipt(
                        receipt_writer.as_ref(),
                        "system-runtime-startup.v6.json",
                        &lock,
                        &unknown,
                    );
                    write_receipt(
                        receipt_writer.as_ref(),
                        "system-runtime-termination.v6.json",
                        &lock,
                        &unknown,
                    );
                    if failure.stage == "PhysicalCompletionUnknown" {
                        std::mem::forget((
                            boundary,
                            sources,
                            selection_file,
                            lock_file,
                            shared_loaded,
                        ));
                    }
                    return Err(failure);
                }
            };
            let observation = prepared.snapshot().system_runtime.clone();
            let startup = runtime_receipt(&lock, arguments.process_epoch, observation.as_ref());
            write_receipt(
                receipt_writer.as_ref(),
                "system-runtime-startup.v6.json",
                &lock,
                &startup,
            );
            let assembled = match shared_loaded {
                Some(loaded) => {
                    super::super::product::RoventNativeProductDriverV3::pc_with_shared_nnuec(
                        prepared,
                        loaded,
                        Arc::clone(&boundary),
                        clock.clone(),
                        Arc::clone(&owners),
                        identity,
                        arguments.root_policy,
                    )
                }
                None => super::super::product::RoventNativeProductDriverV3::pc(
                    prepared,
                    Arc::clone(&boundary),
                    clock.clone(),
                    Arc::clone(&owners),
                    identity,
                    arguments.root_policy,
                ),
            };
            let driver = match assembled {
                Ok(driver) => driver,
                Err(mut failure) => {
                    let detail = error("ProductAssembly", &failure.failure);
                    let unknown =
                        runtime_receipt(&lock, arguments.process_epoch, observation.as_ref());
                    write_receipt(
                        receipt_writer.as_ref(),
                        "system-runtime-termination.v6.json",
                        &lock,
                        &unknown,
                    );
                    if !drain_native(&mut failure.prepared.service, control.cleanup_until) {
                        std::mem::forget((failure, boundary, sources, selection_file, lock_file));
                        return Err(error("PhysicalCompletionUnknown", detail));
                    }
                    return Err(detail);
                }
            };
            let completion = driver.physical_close_observation();
            (Arc::new(driver), completion, observation)
        }
    };
    let mut settings = EngineSettings {
        shutdown_limit: Duration::from_millis(lock.manifest.cleanup_max_ms),
        ..EngineSettings::default()
    };
    settings.search.untimed_limit = Duration::from_millis(lock.manifest.wall_time_max_ms);
    let result = serve_driver(driver, clock, owners, settings)
        .map_err(|failure| error("UciSession", failure));
    let physical_confirmed = completion.is_confirmed();
    let mut termination = runtime_receipt(&lock, arguments.process_epoch, observation.as_ref());
    if physical_confirmed {
        // This is an observation of this product's confirmed finish_session,
        // not a CUDA fence token or an independent supervisor proof.
        termination.physical_completion = rz_experiments::PhysicalCompletionV6::Confirmed {
            completion_id: format!(
                "session_finish_confirmed/1:{}:{}",
                std::process::id(),
                arguments.process_epoch
            ),
        };
    }
    write_receipt(
        receipt_writer.as_ref(),
        "system-runtime-termination.v6.json",
        &lock,
        &termination,
    );
    if !physical_confirmed {
        std::mem::forget((
            boundary,
            sources,
            selection_file,
            external_selection_file,
            lock_file,
        ));
        return Err(error(
            "PhysicalCompletionUnknown",
            match result {
                Ok(()) => "UCI returned before actual physical finish".into(),
                Err(failure) => format!("UCI failed before actual physical finish: {failure}"),
            },
        ));
    }
    result
}

#[cfg(not(target_os = "linux"))]
pub(super) fn run_native(
    _arguments: Arguments,
    _lock: Arc<RoventInputLockV3>,
    _root: Dir,
    _cwd: PathBuf,
    _lock_file: File,
) -> Result<(), RoventLaunchError> {
    Err(error(
        "InputLockUnsupportedV3",
        "system CUDA native owner requires Linux",
    ))
}

#[cfg(not(target_os = "linux"))]
pub(super) fn run_native_with_resource<'a>(
    _arguments: Arguments,
    _lock: Arc<RoventInputLockV3>,
    _root: Dir,
    _cwd: PathBuf,
    _lock_file: File,
    _resource: ResourceAdmissionV3<'a>,
) -> Result<(), RoventLaunchError> {
    Err(error(
        "InputLockUnsupportedV3",
        "system CUDA native owner requires Linux",
    ))
}
