//! First-initialization G0 numerical bootstrap. Pre-init declarations contain no
//! generated artifact references. Only actual prepare output descriptors create
//! model bindings; the final owner is numerical-only, never startup authority.
use super::{
    GenerationTrainingFilePinV3 as Pin, GenerationTrainingPythonRuntimeV3, check, invalid,
};
use crate::{ArenaError, process::OriginalProcessOutput};
#[cfg(target_os = "linux")]
use rz_experiments::{ArtifactRef, PalsCpuSlicesV5, PalsEngineModeV5, PalsModelSpecificationV5};
use rz_experiments::{
    PalsModelBackendV3, PalsNativeStartupOptionsV5, PalsNumericalAdmissionV5, PalsNumericalScopeV5,
    PalsPrecisionV3, PalsResourcePlanV5, PalsStartupAssetV5,
};
#[cfg(any(target_os = "linux", test))]
use rz_experiments::{PalsNumericalBindingsV5, PalsStartupAssetPurposeV5 as Purpose};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::{Path, PathBuf},
    sync::{Arc, atomic::AtomicBool},
};

pub const G0_BOOTSTRAP_LAUNCH_V5: &str = "rz-pals-g0-bootstrap-numerical-launch/2";
pub const G0_PRE_INIT_SCOPE_V5: &str = "rz-pals-g0-pre-init-numerical-scope/2";
pub const G0_CASE_SPEC_V5: &str = "rz-pals-g0-numerical-cases/2";
pub const G0_NATIVE_RECIPE_V5: &str = "rz-pals-g0-native-capture-recipe/2";
const MAX_JSON: u64 = 4 << 20;
const MAX_FILE: u64 = 1 << 30;
const MAX_STAGING: u64 = 18 << 30;

/// Source/licence of newly produced original G0 assets, without a fictitious
/// ArtifactRef or future content hash. Actual references are created afterward.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct G0ArtifactOriginV5 {
    pub source: String,
    pub license: String,
}

/// Closed fresh V3 recipe. All dimensions, roles, seed, FP32, TF32-off and epoch
/// are fixed by this producer (initial model epoch 1, optimizer updates 0);
/// a weights field (optional or otherwise) is absent.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct G0NumericalModelRecipeV5 {
    pub profile: String,
    pub semantic_sha256: String,
    pub encoding_sha256: String,
    pub implementation_sha256: String,
    pub backend: PalsModelBackendV3,
    pub origin: G0ArtifactOriginV5,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct G0NumericalPreInitScopeV5 {
    pub schema: String,
    pub source_commit: String,
    pub source_components: BTreeMap<String, String>,
    pub model: G0NumericalModelRecipeV5,
    pub private_cwd: PathBuf,
    pub native_options: PalsNativeStartupOptionsV5,
    pub resources: PalsResourcePlanV5,
    pub input_assets: Vec<PalsStartupAssetV5>,
    pub input_pins: Vec<super::g0_numerical_process::G0NumericalInputPinV5>,
}

/// Preregistered immutable part of the numerical binding. The four actual
/// producer fields (weights/export/graphs/epoch) cannot appear in this format.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct G0NumericalImmutableBindingsV5 {
    pub model_profile: String,
    pub model_schema: String,
    pub model_semantics: String,
    pub encoding_schema: String,
    pub model_semantic_sha256: String,
    pub encoding_sha256: String,
    pub runtime_library_sha256: String,
    pub cuda_libraries: BTreeSet<String>,
    pub cuda_manifest_sha256: Option<String>,
    pub backend: PalsModelBackendV3,
    pub precision: PalsPrecisionV3,
    pub tf32: bool,
    pub source_components: BTreeMap<String, String>,
}
impl G0NumericalImmutableBindingsV5 {
    /// Resolve only preregistered, already existing runtime/source identities.
    /// It is deliberately not a model specification or startup capability.
    #[cfg(any(target_os = "linux", test))]
    fn from_intent(intent: &G0NumericalPreInitScopeV5) -> Result<Self, ArenaError> {
        let one = |purpose| {
            let mut rows = intent.input_assets.iter().filter(|a| a.purpose == purpose);
            let first = rows
                .next()
                .ok_or_else(|| invalid("G0 immutable input asset absent"))?;
            check(rows.next().is_none(), "G0 immutable input asset ambiguous")?;
            Ok::<_, ArenaError>(first.artifact.sha256.clone())
        };
        let cuda = intent.model.backend == PalsModelBackendV3::OrtCuda;
        Ok(Self {
            model_profile: intent.model.profile.clone(),
            model_schema: rz_experiments::PALS_V5_MODEL_SCHEMA.into(),
            model_semantics: rz_experiments::PALS_V5_MODEL_SEMANTICS.into(),
            encoding_schema: rz_experiments::PALS_V5_ENCODING_SCHEMA.into(),
            model_semantic_sha256: intent.model.semantic_sha256.clone(),
            encoding_sha256: intent.model.encoding_sha256.clone(),
            runtime_library_sha256: one(Purpose::RuntimeLibrary)?,
            cuda_libraries: intent
                .input_assets
                .iter()
                .filter(|a| a.purpose == Purpose::CudaLibrary)
                .map(|a| a.artifact.sha256.clone())
                .collect(),
            cuda_manifest_sha256: if cuda {
                Some(one(Purpose::CudaManifest)?)
            } else {
                None
            },
            backend: intent.model.backend,
            precision: PalsPrecisionV3::Fp32,
            tf32: false,
            source_components: [
                "model_native",
                "model_encoding",
                "nn_runtime",
                "model_reference",
                "model_export",
            ]
            .into_iter()
            .map(|key| {
                intent
                    .source_components
                    .get(key)
                    .map(|value| (key.into(), value.clone()))
                    .ok_or_else(|| invalid("G0 immutable numeric source component absent"))
            })
            .collect::<Result<_, _>>()?,
        })
    }
    #[cfg(any(target_os = "linux", test))]
    pub(super) fn from_actual(b: &PalsNumericalBindingsV5) -> Self {
        Self {
            model_profile: b.model_profile.clone(),
            model_schema: b.model_schema.clone(),
            model_semantics: b.model_semantics.clone(),
            encoding_schema: b.encoding_schema.clone(),
            model_semantic_sha256: b.model_semantic_sha256.clone(),
            encoding_sha256: b.encoding_sha256.clone(),
            runtime_library_sha256: b.runtime_library_sha256.clone(),
            cuda_libraries: b.cuda_libraries.clone(),
            cuda_manifest_sha256: b.cuda_manifest_sha256.clone(),
            backend: b.backend,
            precision: b.precision,
            tf32: b.tf32,
            source_components: b.source_components.clone(),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct G0NumericalCaseSpecV5 {
    pub schema: String,
    pub immutable_bindings: G0NumericalImmutableBindingsV5,
    /// Ordered raw fixture inputs. They certify tensor math, not game legality.
    pub groups: Value,
}
impl G0NumericalCaseSpecV5 {
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            self.schema == G0_CASE_SPEC_V5
                && self.immutable_bindings.precision == PalsPrecisionV3::Fp32
                && !self.immutable_bindings.tf32,
            "G0 sealed case specification domain/precision",
        )?;
        let groups = self
            .groups
            .as_array()
            .ok_or_else(|| invalid("G0 case groups absent"))?;
        check(groups.len() == 18, "G0 bounded sealed groups")?;
        let mut ids = BTreeSet::new();
        let mut roles = BTreeSet::new();
        let mut cases = BTreeSet::new();
        let mut buckets = BTreeSet::new();
        let mut coverage = BTreeSet::new();
        for group in groups {
            let object = group
                .as_object()
                .ok_or_else(|| invalid("G0 group object"))?;
            let id = group["id"].as_str().ok_or_else(|| invalid("G0 group ID"))?;
            let role = group["role"]
                .as_str()
                .ok_or_else(|| invalid("G0 group role"))?;
            check(
                object.len() == 4
                    && object.contains_key("cases")
                    && object.contains_key("line_bucket")
                    && super::training_process::plain_name(id)
                    && ids.insert(id)
                    && matches!(role, "proposer" | "critic" | "validator"),
                "G0 exact group envelope",
            )?;
            roles.insert(role);
            let bucket = group["line_bucket"]
                .as_u64()
                .ok_or_else(|| invalid("G0 integer physical line bucket"))?;
            check(
                [8, 16, 32, 64, 128, 256].contains(&bucket) && buckets.insert((role, bucket)),
                "G0 unique registered role/physical bucket",
            )?;
            let rows = group["cases"]
                .as_array()
                .ok_or_else(|| invalid("G0 case rows absent"))?;
            check(
                !rows.is_empty() && rows.len() <= 10,
                "G0 bounded group cases",
            )?;
            for row in rows {
                check(
                    row.as_object().is_some_and(|o| {
                        o.len() == 2 && o.contains_key("id") && o.contains_key("input")
                    }),
                    "G0 sealed case exact identity/input envelope",
                )?;
                let id = row["id"]
                    .as_str()
                    .ok_or_else(|| invalid("G0 sealed case ID"))?;
                check(
                    super::training_process::plain_name(id)
                        && cases.insert(id)
                        && ids.insert(id)
                        && cases.len() <= 256,
                    "G0 unique bounded case identity",
                )?;
                let value = &row["input"];
                check(
                    value
                        .as_object()
                        .is_some_and(|o| o.len() == 2 && o.contains_key("input"))
                        && value["model_input_kind"] == "virtual_review",
                    "G0 sealed V3 typed input codec",
                )?;
                let input: rz_eval::pals_model::virtual_review::PalsVirtualReviewInput =
                    serde_json::from_value(value["input"].clone())
                        .map_err(|e| invalid(e.to_string()))?;
                let admission = input.validate().map_err(|e| invalid(e.to_string()))?;
                let input_role = match input.role {
                    rz_eval::pals_model::PalsRole::Proposer => "proposer",
                    rz_eval::pals_model::PalsRole::Critic => "critic",
                    rz_eval::pals_model::PalsRole::Validator => "validator",
                };
                check(
                    input_role == role
                        && u64::try_from(admission.length_bucket).ok() == Some(bucket)
                        && !input.candidates.is_empty()
                        && [0.0, 1.0].contains(&input.anchors[0].metadata[0]),
                    "G0 sealed case role/anchor side/bucket differs",
                )?;
                coverage.insert((
                    role,
                    bucket,
                    input.anchors.len(),
                    u8::from(input.anchors[0].metadata[0] == 1.0),
                ));
            }
        }
        check(
            coverage.len() == 3 * 6 * 4 * 2,
            "G0 sealed cases require all role/bucket/anchor/side coverage before initialization",
        )?;
        check(
            roles == BTreeSet::from(["proposer", "critic", "validator"]),
            "G0 case specification P/C/V scope",
        )
    }
    #[cfg(any(target_os = "linux", test))]
    fn materialize(&self, actual: &PalsNumericalBindingsV5) -> Result<Value, ArenaError> {
        self.validate()?;
        check(
            self.immutable_bindings == G0NumericalImmutableBindingsV5::from_actual(actual),
            "G0 immutable source/runtime/model bindings changed after prepare",
        )?;
        let plan = serde_json::json!({"schema_version":5,"domain":"rz-pals-numerical-case-plan-v5/1",
            "invocation":"fresh","batch":1,"bindings":actual,"groups":self.groups});
        super::numerical_process::validate_plan(&plan, actual)?;
        check(
            plan["groups"] == self.groups,
            "G0 ordered case byte seal changed",
        )?;
        Ok(plan)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct G0NativeCaptureRecipeV5 {
    pub schema: String,
    pub runtime_library: Pin,
    pub runtime_cache: PathBuf,
    pub cuda_bundle: Option<Pin>,
    pub native_options: PalsNativeStartupOptionsV5,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct G0NumericalBootstrapRegistrationV5 {
    pub schema: String,
    pub numerical_id: String,
    pub source_commit: String,
    pub pre_init_scope: Pin,
    pub case_spec: Pin,
    pub native_recipe: Pin,
    pub rules_profile: Pin,
    pub native_executable: Pin,
    pub python_runtime: GenerationTrainingPythonRuntimeV3,
    pub dependency_directories: Vec<PathBuf>,
    pub runtime_module_pins: Vec<Pin>,
    pub staging_root: PathBuf,
    pub initialization_directory: PathBuf,
    pub export_directory: PathBuf,
    pub reference_output_root: PathBuf,
    pub native_output_root: PathBuf,
    pub prepare_receipt_path: PathBuf,
    pub reference_receipt_path: PathBuf,
    pub acceptance_path: PathBuf,
    pub parent_receipt_path: PathBuf,
    pub max_wall_ns: u64,
    pub max_output_bytes: u64,
    pub max_read_bytes: u64,
}
impl G0NumericalBootstrapRegistrationV5 {
    pub fn validate(&self) -> Result<(), ArenaError> {
        use super::training_process::{absolute_safe, plain_name};
        check(
            self.schema == G0_BOOTSTRAP_LAUNCH_V5
                && plain_name(&self.numerical_id)
                && self.numerical_id.len() <= 96
                && self.source_commit.len() == 40
                && self
                    .source_commit
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
                && (1..=3_600_000_000_000).contains(&self.max_wall_ns)
                && (1..=MAX_STAGING).contains(&self.max_output_bytes)
                && (1..=4 * MAX_STAGING).contains(&self.max_read_bytes)
                && !self.python_runtime.version.is_empty()
                && self.python_runtime.version.len() <= 128,
            "G0 bootstrap exact finite registration",
        )?;
        for (pin, cap) in [
            (&self.pre_init_scope, 8 << 20),
            (&self.case_spec, MAX_JSON),
            (&self.native_recipe, 64 << 10),
            (&self.rules_profile, MAX_JSON),
            (&self.native_executable, 256 << 20),
            (&self.python_runtime.executable, 256 << 20),
        ] {
            pin.validate(cap)?;
        }
        check(
            (1..=8).contains(&self.dependency_directories.len())
                && (1..=64).contains(&self.runtime_module_pins.len())
                && self.dependency_directories.iter().all(|p| absolute_safe(p))
                && self
                    .dependency_directories
                    .iter()
                    .collect::<BTreeSet<_>>()
                    .len()
                    == self.dependency_directories.len()
                && self
                    .runtime_module_pins
                    .iter()
                    .map(|p| &p.path)
                    .collect::<BTreeSet<_>>()
                    .len()
                    == self.runtime_module_pins.len(),
            "G0 bootstrap bounded distinct dependency namespace",
        )?;
        for pin in &self.runtime_module_pins {
            pin.validate(MAX_FILE)?;
            check(
                self.dependency_directories
                    .iter()
                    .any(|p| pin.path.starts_with(p)),
                "G0 runtime pin outside explicit dependencies",
            )?;
        }
        let directories = [
            &self.initialization_directory,
            &self.export_directory,
            &self.reference_output_root,
            &self.native_output_root,
        ];
        let receipts = [
            &self.prepare_receipt_path,
            &self.reference_receipt_path,
            &self.acceptance_path,
            &self.parent_receipt_path,
        ];
        let paths = directories.into_iter().chain(receipts).collect::<Vec<_>>();
        check(
            absolute_safe(&self.staging_root)
                && paths.iter().all(|p| {
                    absolute_safe(p)
                        && p.starts_with(&self.staging_root)
                        && **p != self.staging_root
                })
                && paths.iter().collect::<BTreeSet<_>>().len() == paths.len()
                && directories
                    .iter()
                    .all(|a| paths.iter().all(|b| *a == *b || !b.starts_with(a)))
                && receipts
                    .iter()
                    .all(|p| p.extension().is_some_and(|s| s == "json")),
            "G0 bootstrap disjoint owned output paths",
        )?;
        for input in [
            &self.pre_init_scope,
            &self.case_spec,
            &self.native_recipe,
            &self.rules_profile,
            &self.native_executable,
            &self.python_runtime.executable,
        ] {
            check(
                !directories.iter().any(|p| input.path.starts_with(p))
                    && !receipts.iter().any(|p| input.path == **p),
                "G0 bootstrap input aliases generated namespace",
            )?;
        }
        Ok(())
    }
}

/// No decoder or public constructor. Full startup config/lock and CPU_T/game
/// authority are deliberately absent; actual startup must run its own guards.
pub struct VerifiedG0BootstrapNumericalV5 {
    scope: PalsNumericalScopeV5,
    admission: PalsNumericalAdmissionV5,
    acceptance_pin: Pin,
    parent_receipt: Value,
    processes: Vec<OriginalProcessOutput>,
    #[cfg(target_os = "linux")]
    _registration: File,
    #[cfg(target_os = "linux")]
    held: super::training_process::LinuxHeld,
}
impl VerifiedG0BootstrapNumericalV5 {
    pub fn scope(&self) -> &PalsNumericalScopeV5 {
        &self.scope
    }
    pub fn admission(&self) -> &PalsNumericalAdmissionV5 {
        &self.admission
    }
    pub fn acceptance_pin(&self) -> &Pin {
        &self.acceptance_pin
    }
    pub fn parent_receipt(&self) -> &Value {
        &self.parent_receipt
    }
    pub fn processes(&self) -> &[OriginalProcessOutput] {
        &self.processes
    }
    /// Internal original-directory join for the metadata-only Deployment seal.
    /// This exposes neither a public constructor nor a serialized capability.
    #[cfg(target_os = "linux")]
    pub(super) fn held_directory_file(&self, path: &Path) -> Option<&File> {
        self.held
            .directories
            .iter()
            .find(|(named, _, _)| named == path)
            .map(|(_, file, _)| file)
    }
    #[cfg(target_os = "linux")]
    pub(super) fn cuda_artifact_tree(
        &self,
    ) -> Option<rz_eval::runtime_pin::PreparedCudaArtifactTree> {
        self.held.cuda_artifact_tree.clone()
    }
    #[cfg(target_os = "linux")]
    pub(super) fn cuda_artifact_tree_observation(&self) -> Option<Value> {
        self.held.cuda_artifact_tree_observation.clone()
    }
    pub fn held_files(&self) -> impl Iterator<Item = (&Pin, &File)> {
        #[cfg(target_os = "linux")]
        {
            self.held
                .checkpoints
                .iter()
                .chain(self.held.inputs.iter())
                .map(|f| (&f.pin, &f.file))
        }
        #[cfg(not(target_os = "linux"))]
        {
            std::iter::empty()
        }
    }
}

pub fn run_g0_bootstrap_numerical_original_v5(
    path: &Path,
    sha256: &str,
    cancel: Arc<AtomicBool>,
) -> Result<VerifiedG0BootstrapNumericalV5, Box<super::g0_numerical_process::G0NumericalFailureV5>>
{
    #[cfg(target_os = "linux")]
    {
        linux::run(path, sha256, &cancel)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (path, sha256, cancel);
        Err(Box::new(
            super::g0_numerical_process::G0NumericalFailureV5 {
                error: invalid("G0 bootstrap original numerical transport requires Linux"),
                processes: Vec::new(),
                child_receipts: Vec::new(),
                preservation_reports: Vec::new(),
                charged_read_bytes: 0,
                read_usage_uncertain: false,
                registration_pin: None,
                _registration: None,
            },
        ))
    }
}

#[cfg(target_os = "linux")]
pub(super) mod linux {
    use super::super::{
        g0_numerical_process::linux as original, numerical_process as numerical,
        training_process as common,
    };
    use super::*;
    use common::{LinuxHeld, ReadAccounting, ReadCredit};
    use original::{OriginalG0Scope, Running};
    use std::{
        sync::atomic::Ordering,
        time::{Duration, Instant},
    };
    const COMPONENTS: [&str; 5] = [
        "model_native",
        "model_encoding",
        "nn_runtime",
        "model_reference",
        "model_export",
    ];
    const INITIALIZERS: [&str; 8] = [
        "__init__.py",
        "config.py",
        "model.py",
        "artifacts.py",
        "controlled_initialization.py",
        "virtual_review.py",
        "onnx_shared.py",
        "initialization_evidence.py",
    ];

    impl G0NumericalBootstrapRegistrationV5 {
        fn execution_scope(&self, options: &PalsNativeStartupOptionsV5) -> OriginalG0Scope {
            OriginalG0Scope {
                numerical_id: self.numerical_id.clone(),
                source_commit: self.source_commit.clone(),
                python_runtime: self.python_runtime.clone(),
                runtime_module_pins: self.runtime_module_pins.clone(),
                native_executable: self.native_executable.clone(),
                staging_root: self.staging_root.clone(),
                initialization_directory: self.initialization_directory.clone(),
                export_directory: self.export_directory.clone(),
                reference_output_root: self.reference_output_root.clone(),
                native_output_root: self.native_output_root.clone(),
                prepare_receipt_path: self.prepare_receipt_path.clone(),
                reference_receipt_path: self.reference_receipt_path.clone(),
                parent_receipt_path: self.parent_receipt_path.clone(),
                cudnn_conv_search: options.cudnn_conv_search.clone(),
                max_wall_ns: self.max_wall_ns,
                max_output_bytes: self.max_output_bytes,
            }
        }
    }
    type Completed = (PalsNumericalScopeV5, PalsNumericalAdmissionV5, Pin, Value);
    pub(super) fn run(
        path: &Path,
        sha256: &str,
        cancel: &AtomicBool,
    ) -> Result<
        VerifiedG0BootstrapNumericalV5,
        Box<super::super::g0_numerical_process::G0NumericalFailureV5>,
    > {
        let started = Instant::now();
        let mut running = Running {
            registration_pin: None,
            registration: None,
            held: None,
            processes: Vec::new(),
            receipts: Vec::new(),
            preservation: Vec::new(),
            account: ReadAccounting::new(),
        };
        match execute(path, sha256, started, cancel, &mut running) {
            Ok((scope, admission, acceptance_pin, parent_receipt)) => {
                Ok(VerifiedG0BootstrapNumericalV5 {
                    scope,
                    admission,
                    acceptance_pin,
                    parent_receipt,
                    processes: running.processes,
                    _registration: running
                        .registration
                        .expect("original bootstrap declaration FD"),
                    held: running.held.expect("actual G0 custody"),
                })
            }
            Err(error) => Err(Box::new(
                super::super::g0_numerical_process::G0NumericalFailureV5 {
                    error,
                    processes: running.processes,
                    child_receipts: running.receipts,
                    preservation_reports: running.preservation,
                    charged_read_bytes: running.account.charged.load(Ordering::Acquire),
                    read_usage_uncertain: running.account.uncertain.load(Ordering::Acquire),
                    registration_pin: running.registration_pin,
                    _registration: running.registration,
                    _held: running.held,
                },
            )),
        }
    }
    fn execute(
        path: &Path,
        sha256: &str,
        started: Instant,
        cancel: &AtomicBool,
        running: &mut Running,
    ) -> Result<Completed, ArenaError> {
        let mut credit = ReadCredit {
            remaining: 4 * MAX_STAGING,
            maximum: 4 * MAX_STAGING,
            until: started + Duration::from_secs(60),
            cancel,
            account: Arc::clone(&running.account),
        };
        running.registration = Some(common::open_file(path)?);
        let file = running
            .registration
            .as_ref()
            .expect("first bootstrap descriptor");
        let pin = Pin {
            path: path.into(),
            sha256: sha256.into(),
            bytes: file
                .metadata()
                .map_err(|_| invalid("G0 bootstrap declaration metadata"))?
                .len(),
        };
        running.registration_pin = Some(pin.clone());
        pin.validate(MAX_JSON)?;
        let r: G0NumericalBootstrapRegistrationV5 =
            common::decode(&common::verify_bytes(file, &pin, true, &mut credit)?)?;
        r.validate()?;
        let until = started
            .checked_add(Duration::from_nanos(r.max_wall_ns))
            .ok_or_else(|| invalid("G0 bootstrap deadline overflow"))?;
        let cleanup_until = until + Duration::from_secs(30);
        credit.remaining = r
            .max_read_bytes
            .checked_sub(running.account.charged.load(Ordering::Acquire))
            .ok_or_else(|| invalid("bootstrap declaration consumed read credit"))?;
        credit.maximum = r.max_read_bytes;
        credit.until = until;
        credit.take(0)?;
        running.held = Some(common::prepare_original_root(
            &r.staging_root,
            r.staging_root
                .join(format!("g0-bootstrap-source-{}", r.numerical_id)),
            &r.prepare_receipt_path,
            &[
                &r.initialization_directory,
                &r.export_directory,
                &r.reference_output_root,
                &r.native_output_root,
            ],
        )?);
        let held = running.held.as_mut().expect("bootstrap directory custody");
        for directory in [
            &r.initialization_directory,
            &r.export_directory,
            &r.reference_output_root,
            &r.native_output_root,
        ] {
            check(
                common::open_directory(directory)?
                    .entries()
                    .map_err(|_| invalid("bootstrap output entries"))?
                    .next()
                    .is_none(),
                "bootstrap requires fresh empty output directories",
            )?;
        }
        for output in [
            &r.reference_receipt_path,
            &r.acceptance_path,
            &r.parent_receipt_path,
        ] {
            check(
                !output.exists(),
                "bootstrap refuses pre-existing terminal output",
            )?;
        }
        common::hold_input(pin, false, held, &mut credit)?;
        let raw = common::hold_input(r.pre_init_scope.clone(), true, held, &mut credit)?;
        let intent: G0NumericalPreInitScopeV5 = common::decode(&raw)?;
        validate_intent(&intent, &r)?;
        common::retain_directory(intent.private_cwd.clone(), held)?;
        for input in &intent.input_pins {
            common::hold_input(input.source.clone(), false, held, &mut credit)?;
        }
        let raw = common::hold_input(r.case_spec.clone(), true, held, &mut credit)?;
        let cases: G0NumericalCaseSpecV5 = common::decode(&raw)?;
        cases.validate()?;
        check(
            cases.immutable_bindings == G0NumericalImmutableBindingsV5::from_intent(&intent)?,
            "G0 preregistered case source/runtime/model scope differs before initializer",
        )?;
        let case_bytes_seal =
            serde_json::to_vec(&cases.groups).map_err(|e| invalid(e.to_string()))?;
        let raw = common::hold_input(r.native_recipe.clone(), true, held, &mut credit)?;
        let native: G0NativeCaptureRecipeV5 = common::decode(&raw)?;
        validate_native_recipe(&native, &intent, &r)?;
        if native.cuda_bundle.is_some() {
            return Err(super::super::g0_cuda_artifact_tree::legacy_cuda_execution_disabled());
        }
        common::retain_directory(native.runtime_cache.clone(), held)?;
        for input in [
            &r.rules_profile,
            &r.native_executable,
            &r.python_runtime.executable,
        ] {
            common::hold_input(input.clone(), false, held, &mut credit)?;
        }
        let dependencies = r
            .dependency_directories
            .iter()
            .map(|p| common::retain_directory(p.clone(), held))
            .collect::<Result<Vec<_>, _>>()?;
        for input in &r.runtime_module_pins {
            common::hold_input(input.clone(), false, held, &mut credit)?;
        }
        super::super::g0_cuda_artifact_tree::prepare(
            held,
            &native.runtime_cache,
            &native.runtime_library.path,
            native.cuda_bundle.as_ref().map(|pin| pin.path.as_path()),
            r.max_output_bytes,
            &mut credit,
        )?;
        super::super::g0_cuda_artifact_tree::ensure_output_credit(
            held,
            r.max_output_bytes,
            36 << 20,
            until,
            cancel,
        )?;
        common::install_sources_for_components(held, &mut credit, &COMPONENTS)?;
        let sources = COMPONENTS
            .into_iter()
            .map(|name| {
                intent
                    .source_components
                    .get(name)
                    .map(|sha| (name.into(), sha.clone()))
                    .ok_or_else(|| invalid("bootstrap exact source component absent"))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let mut initializers = BTreeMap::new();
        for name in INITIALIZERS {
            let expected = held
                .source_root
                .join("experiments/model-research/pals/src/rz_pals_model")
                .join(name);
            let source = held
                .sources
                .iter()
                .find(|f| f.pin.path == expected)
                .ok_or_else(|| invalid("bootstrap compiled initializer source missing"))?;
            initializers.insert(name, source.pin.sha256.clone());
        }
        let initialization_sources = serde_json::json!({"schema":"rz-pals-g0-initialization-sources/1","source_commit":r.source_commit,"pins":initializers});
        let execution = r.execution_scope(&intent.native_options);
        let prepare = original::child(
            &execution,
            "prepare",
            serde_json::json!({"rules_profile":r.rules_profile,"case_plan_template":r.case_spec}),
            &sources,
            &initialization_sources,
            &Value::Array(dependencies.clone()),
            started,
            until,
            cleanup_until,
            cancel,
            running,
            &mut credit,
        )?;
        let outputs = prepare
            .get("outputs")
            .ok_or_else(|| invalid("bootstrap original prepare outputs absent"))?;
        let mut scope = prepared_scope(
            &intent,
            &r,
            outputs,
            running.held.as_mut().expect("G0 output custody"),
            &mut credit,
        )?;
        validate_prepared(
            &scope,
            &intent.private_cwd,
            &r,
            running.held.as_ref().expect("G0 prepared original files"),
            &mut credit,
        )?;
        let bindings = rz_experiments::numerical_scope_bindings_v5(&scope)?;
        let plan = cases.materialize(&bindings)?;
        check(
            serde_json::to_vec(&plan["groups"]).map_err(|e| invalid(e.to_string()))?
                == case_bytes_seal,
            "bootstrap case order/content changed",
        )?;
        let raw = serde_json::to_vec(&plan).map_err(|e| invalid(e.to_string()))?;
        let held = running
            .held
            .as_mut()
            .expect("bootstrap actual plan custody");
        super::super::g0_cuda_artifact_tree::ensure_output_credit(
            held,
            r.max_output_bytes,
            raw.len() as u64,
            until,
            cancel,
        )?;
        let file = common::write_owned_new(
            &r.staging_root
                .join(format!("g0-bootstrap-plan-{}.json", r.numerical_id)),
            &raw,
        )?;
        let plan_pin = file.pin.clone();
        held.checkpoints.push(file);
        let reference = original::child(
            &execution,
            "reference",
            serde_json::json!({"case_plan":plan_pin,"checkpoint":original::pin_value(&outputs["checkpoint"])?,
            "checkpoint_metadata":original::pin_value(&outputs["checkpoint_metadata"])?,"initialization_evidence":original::pin_value(&outputs["initialization_evidence"])?,
            "export_manifest":original::pin_value(&outputs["export_manifest"])?,"export_graphs":original::graph_pin_values(&outputs["export_graphs"])?,
            "cuda_control_inventory":original::pin_value(&outputs["cuda_control_inventory"])?}),
            &sources,
            &initialization_sources,
            &Value::Array(dependencies),
            started,
            until,
            cleanup_until,
            cancel,
            running,
            &mut credit,
        )?;
        let references = original::hold_reference_outputs(
            &execution,
            &reference,
            &plan,
            running.held.as_mut().expect("actual raw custody"),
            &mut credit,
        )?;
        let capture = materialize_native(
            &native,
            &scope,
            &intent.private_cwd,
            &r,
            &plan_pin,
            until,
            cleanup_until,
        )?;
        original::execute_native(
            &execution,
            capture,
            &plan,
            started,
            until,
            cleanup_until,
            cancel,
            running,
            &mut credit,
        )?;
        finalize(
            &r,
            &intent,
            &mut scope,
            &plan,
            &references,
            cleanup_until,
            cancel,
            running,
            &mut credit,
        )
    }

    fn validate_intent(
        i: &G0NumericalPreInitScopeV5,
        r: &G0NumericalBootstrapRegistrationV5,
    ) -> Result<(), ArenaError> {
        check(
            i.schema == G0_PRE_INIT_SCOPE_V5
                && i.source_commit == r.source_commit
                && i.model.profile == "virtual_review_v1"
                && matches!(
                    i.model.backend,
                    PalsModelBackendV3::OrtCpu | PalsModelBackendV3::OrtCuda
                )
                && common::absolute_safe(&i.private_cwd)
                && i.private_cwd.starts_with(&r.staging_root)
                && i.private_cwd != r.staging_root
                && [&r.initialization_directory, &r.export_directory]
                    .into_iter()
                    .all(|p| {
                        p.starts_with(i.private_cwd.join("inputs"))
                            && *p != i.private_cwd.join("inputs")
                    })
                && r.acceptance_path.parent() == Some(i.private_cwd.join("inputs").as_path()),
            "G0 pre-init closed model/owner/source intent",
        )?;
        for sha in [
            &i.model.semantic_sha256,
            &i.model.encoding_sha256,
            &i.model.implementation_sha256,
        ] {
            check(
                sha.len() == 64
                    && sha
                        .bytes()
                        .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
                "G0 model recipe actual source digest",
            )?;
        }
        check(
            i.model.semantic_sha256
                == hex(rz_eval::pals_model::PalsModelProfile::VirtualReviewV1
                    .registered_semantics_sha256())
                && i.model.encoding_sha256
                    == hex(rz_eval::pals_model::virtual_review::registered_encoding_sha256())
                && i.source_components.len() <= 128,
            "bootstrap compiled model/source/tensor registry differs",
        )?;
        let compiled = rz_experiments::compiled_source_components_v5()?;
        check(
            COMPONENTS.into_iter().all(|key| {
                i.source_components
                    .get(key)
                    .is_some_and(|value| compiled.get(key) == Some(value))
            }) && i
                .source_components
                .iter()
                .all(|(key, value)| compiled.get(key) == Some(value)),
            "G0 actual numerical source components differ from compiled registry",
        )?;
        check(
            !i.model.origin.source.is_empty()
                && i.model.origin.source.len() <= 2048
                && !i.model.origin.license.is_empty()
                && i.model.origin.license.len() <= 128,
            "G0 actual produced asset source/licence required",
        )?;
        i.resources.validate(&PalsCpuSlicesV5::default())?;
        check(
            i.resources.cpu_quota_millis == 2000
                && i.resources.gpu_count
                    == u32::from(i.model.backend == PalsModelBackendV3::OrtCuda),
            "bootstrap declared CPU/GPU quota differs",
        )?;
        validate_options(&i.native_options, i.model.backend, r.max_wall_ns)?;
        check(
            !i.input_assets.is_empty()
                && i.input_assets.len() <= 96
                && i.input_pins.len() == i.input_assets.len(),
            "bootstrap exact actual input roster",
        )?;
        let mut names = BTreeSet::new();
        let mut paths = BTreeSet::new();
        for asset in &i.input_assets {
            asset.artifact.validate()?;
            check(
                matches!(
                    asset.purpose,
                    Purpose::RuntimeLibrary
                        | Purpose::RulesDescriptor
                        | Purpose::CudaManifest
                        | Purpose::CudaLibrary
                ) && rz_experiments::pals_private_snapshot_path_v5(&asset.snapshot_path)
                    && names.insert(&asset.snapshot_path)
                    && paths.insert(&asset.artifact.path),
                "bootstrap cannot preregister generated/unused startup assets",
            )?;
            let rows = i
                .input_pins
                .iter()
                .filter(|p| p.snapshot_path == asset.snapshot_path)
                .collect::<Vec<_>>();
            check(rows.len() == 1, "bootstrap input pin missing/duplicate")?;
            let row = rows[0];
            row.source.validate(MAX_STAGING)?;
            check(
                row.source.path == i.private_cwd.join(&asset.snapshot_path)
                    && row.source.sha256 == asset.artifact.sha256
                    && row.source.bytes == asset.artifact.bytes,
                "bootstrap exact source path/hash/size differs",
            )?;
        }
        let count = |p| i.input_assets.iter().filter(|a| a.purpose == p).count();
        check(
            count(Purpose::RulesDescriptor) == 1 && count(Purpose::RuntimeLibrary) == 1,
            "bootstrap requires exact original Rules/runtime",
        )?;
        let rules = i
            .input_assets
            .iter()
            .find(|a| a.purpose == Purpose::RulesDescriptor)
            .expect("checked Rules");
        check(
            i.input_pins
                .iter()
                .any(|p| p.snapshot_path == rules.snapshot_path && p.source == r.rules_profile),
            "bootstrap original Rules pin differs",
        )?;
        let cuda = i.model.backend == PalsModelBackendV3::OrtCuda;
        check(
            if cuda {
                count(Purpose::CudaManifest) == 1 && count(Purpose::CudaLibrary) > 0
            } else {
                count(Purpose::CudaManifest) == 0 && count(Purpose::CudaLibrary) == 0
            },
            "bootstrap cannot inherit another provider asset namespace",
        )
    }
    fn validate_options(
        o: &PalsNativeStartupOptionsV5,
        backend: PalsModelBackendV3,
        max_wall_ns: u64,
    ) -> Result<(), ArenaError> {
        check(
            !o.cache_public_memory
                && !o.approximate_warm
                && !o.device_public_memory
                && o.startup_ms > 0
                && u128::from(o.startup_ms) * 1_000_000 <= u128::from(max_wall_ns)
                && (1..=30_000).contains(&o.drain_ms),
            "bootstrap Fresh/host/fixed original startup/drain policy",
        )?;
        if backend == PalsModelBackendV3::OrtCuda {
            check(
                o.native_loading_profile.as_deref()
                    == Some(rz_experiments::PALS_NATIVE_EAGER_CUDA_PROFILE_V5)
                    && o.cuda_device_id.is_some_and(|n| n >= 0)
                    && o.cuda_arena_bytes
                        .is_some_and(|n| (1..=6u64 << 30).contains(&n))
                    && matches!(
                        o.cudnn_conv_search.as_str(),
                        "legacy_implicit_exhaustive" | "experimental_default"
                    ),
                "bootstrap exact eager CUDA profile/device/arena/cuDNN",
            )
        } else {
            check(
                o.native_loading_profile.is_none()
                    && o.cuda_device_id.is_none()
                    && o.cuda_arena_bytes.is_none()
                    && o.cudnn_conv_search == "legacy_implicit_exhaustive",
                "bootstrap CPU cannot inherit CUDA selections",
            )
        }
    }
    fn validate_native_recipe(
        n: &G0NativeCaptureRecipeV5,
        i: &G0NumericalPreInitScopeV5,
        r: &G0NumericalBootstrapRegistrationV5,
    ) -> Result<(), ArenaError> {
        check(
            n.schema == G0_NATIVE_RECIPE_V5
                && n.native_options == i.native_options
                && common::absolute_safe(&n.runtime_cache)
                && n.runtime_cache.starts_with(&r.staging_root)
                && n.runtime_cache != r.staging_root
                && [
                    &r.initialization_directory,
                    &r.export_directory,
                    &r.reference_output_root,
                    &r.native_output_root,
                ]
                .into_iter()
                .all(|p| !n.runtime_cache.starts_with(p) && !p.starts_with(&n.runtime_cache)),
            "bootstrap native recipe differs from original options/cache namespace",
        )?;
        let pin = |purpose| {
            i.input_assets
                .iter()
                .find(|a| a.purpose == purpose)
                .and_then(|a| {
                    i.input_pins
                        .iter()
                        .find(|p| p.snapshot_path == a.snapshot_path)
                })
                .map(|p| &p.source)
        };
        check(
            pin(Purpose::RuntimeLibrary) == Some(&n.runtime_library),
            "bootstrap native runtime pin differs",
        )?;
        check(
            if i.model.backend == PalsModelBackendV3::OrtCuda {
                n.cuda_bundle.as_ref() == pin(Purpose::CudaManifest)
            } else {
                n.cuda_bundle.is_none()
            },
            "bootstrap actual CUDA bundle pin differs",
        )
    }
    fn hex(bytes: [u8; 32]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }
    pub(in crate::pals_collect_v3) fn asset(
        pin: &Pin,
        cwd: &Path,
        purpose: Purpose,
        origin: &G0ArtifactOriginV5,
    ) -> Result<PalsStartupAssetV5, ArenaError> {
        let relative = pin
            .path
            .strip_prefix(cwd)
            .map_err(|_| invalid("actual bootstrap artifact escaped original cwd"))?
            .to_str()
            .ok_or_else(|| invalid("bootstrap artifact UTF-8"))?
            .replace('\\', "/");
        check(
            rz_experiments::pals_private_snapshot_path_v5(&relative),
            "actual bootstrap private snapshot name",
        )?;
        let a = PalsStartupAssetV5 {
            artifact: ArtifactRef {
                path: format!("g0/{relative}"),
                sha256: pin.sha256.clone(),
                bytes: pin.bytes,
                source: origin.source.clone(),
                license: origin.license.clone(),
            },
            snapshot_path: relative,
            purpose,
        };
        a.artifact.validate()?;
        Ok(a)
    }
    fn prepared_scope(
        i: &G0NumericalPreInitScopeV5,
        r: &G0NumericalBootstrapRegistrationV5,
        o: &Value,
        held: &mut LinuxHeld,
        credit: &mut ReadCredit<'_>,
    ) -> Result<PalsNumericalScopeV5, ArenaError> {
        check(
            o.as_object().is_some_and(|v| v.len() == 6),
            "bootstrap exact actual prepare roster",
        )?;
        let mut assets = i.input_assets.clone();
        for (name, path, purpose, cap) in [
            (
                "checkpoint",
                r.initialization_directory.join("untrained.pt"),
                Purpose::Weights,
                MAX_FILE,
            ),
            (
                "checkpoint_metadata",
                r.initialization_directory.join("checkpoint.json"),
                Purpose::InitializationCheckpointMetadata,
                64 << 10,
            ),
            (
                "initialization_evidence",
                r.initialization_directory
                    .join("g0-initialization-evidence.json"),
                Purpose::InitializationEvidence,
                MAX_JSON,
            ),
            (
                "export_manifest",
                r.export_directory.join("export.json"),
                Purpose::ExportManifest,
                256 << 10,
            ),
        ] {
            let pin = original::hold_observed(&o[name], &path, cap, held, credit)?;
            assets.push(asset(&pin, &i.private_cwd, purpose, &i.model.origin)?);
        }
        let graphs = o["export_graphs"]
            .as_array()
            .ok_or_else(|| invalid("bootstrap actual graphs absent"))?;
        check(graphs.len() == 3, "bootstrap actual Collection graphs")?;
        for ((role, file, purpose), g) in [
            ("public", "public_memory.onnx", Purpose::PublicGraph),
            ("shared_pc", "shared_pc_if.onnx", Purpose::SharedPcGraph),
            ("validator", "role_validator.onnx", Purpose::ValidatorGraph),
        ]
        .into_iter()
        .zip(graphs)
        {
            check(
                g["role"] == role && g["file"] == file,
                "bootstrap original graph role/order",
            )?;
            let observed = serde_json::json!({"path":g["path"],"sha256":g["sha256"],"bytes":g["bytes"],"device":g["device"],"inode":g["inode"]});
            let pin = original::hold_observed(
                &observed,
                &r.export_directory.join(file),
                MAX_FILE,
                held,
                credit,
            )?;
            assets.push(asset(&pin, &i.private_cwd, purpose, &i.model.origin)?);
        }
        let inventory = original::hold_observed(
            &o["cuda_control_inventory"],
            &r.export_directory.join("cuda-control-inventory.json"),
            MAX_JSON,
            held,
            credit,
        )?;
        if i.model.backend == PalsModelBackendV3::OrtCuda {
            assets.push(asset(
                &inventory,
                &i.private_cwd,
                Purpose::CudaControlInventory,
                &i.model.origin,
            )?);
        }
        let weights = assets
            .iter()
            .find(|a| a.purpose == Purpose::Weights)
            .expect("actual held weights")
            .artifact
            .clone();
        let model = PalsModelSpecificationV5 {
            profile: i.model.profile.clone(),
            model_schema: rz_experiments::PALS_V5_MODEL_SCHEMA.into(),
            model_semantics: rz_experiments::PALS_V5_MODEL_SEMANTICS.into(),
            encoding_schema: rz_experiments::PALS_V5_ENCODING_SCHEMA.into(),
            semantic_sha256: i.model.semantic_sha256.clone(),
            encoding_sha256: i.model.encoding_sha256.clone(),
            implementation_sha256: i.model.implementation_sha256.clone(),
            weights,
            artifact_domain: rz_experiments::PalsArtifactDomainV5::Untrained,
            provenance: None,
            checkpoint_domain: None,
            epoch: 1,
            backend: i.model.backend,
            precision: PalsPrecisionV3::Fp32,
            tf32: false,
            exported_roles: BTreeSet::from([
                rz_experiments::PalsRoleV3::Proposer,
                rz_experiments::PalsRoleV3::Critic,
                rz_experiments::PalsRoleV3::Validator,
            ]),
            width: 384,
            latent_count: 16,
            temporal_width: 64,
            candidate_head_width: 128,
            record_feature_axes: 8,
            query_feature_axes: 8,
            max_records: 64,
            max_anchors: 4,
            line_buckets: vec![8, 16, 32, 64, 128, 256],
            one_dynamic_line_graph: true,
            ids_as_numeric_features: false,
            fresh_warm_separate_domains: true,
        };
        let scope = PalsNumericalScopeV5 {
            mode: PalsEngineModeV5::Collection,
            model,
            source_commit: r.source_commit.clone(),
            source_components: i.source_components.clone(),
            assets,
            training_executed: false,
        };
        scope.validate_pre_acceptance()?;
        Ok(scope)
    }
    pub(in crate::pals_collect_v3) fn validate_prepared(
        s: &PalsNumericalScopeV5,
        cwd: &Path,
        r: &G0NumericalBootstrapRegistrationV5,
        held: &LinuxHeld,
        credit: &mut ReadCredit<'_>,
    ) -> Result<(), ArenaError> {
        let mut read =
            |a: &ArtifactRef, n: usize| original::read_assets(&s.assets, cwd, held, a, n, credit);
        let export = read(&s.asset(Purpose::ExportManifest)?.artifact, 256 << 10)?;
        crate::pals_adaptive_launch::validate_scope_export_content_v5(s, &export)?;
        let rules = read(
            &s.asset(Purpose::RulesDescriptor)?.artifact,
            MAX_JSON as usize,
        )?;
        crate::pals_adaptive_launch::validate_scope_export_rules_content_v5(s, &export, &rules)?;
        let domain = rz_experiments::validate_pals_scope_artifact_domain_v5(s, &export, &mut read)?;
        check(
            domain.initializer_source_commit.as_deref() == Some(r.source_commit.as_str()),
            "bootstrap original initializer commit differs",
        )?;
        for (name, sha) in domain.initializer_source_pins {
            check(
                held.sources.iter().any(|f| {
                    f.pin.path
                        == held
                            .source_root
                            .join("experiments/model-research/pals/src/rz_pals_model")
                            .join(&name)
                        && f.pin.sha256 == sha
                }),
                "bootstrap initializer evidence has no original compiled source join",
            )?;
        }
        Ok(())
    }
    pub(in crate::pals_collect_v3) fn materialize_native(
        n: &G0NativeCaptureRecipeV5,
        s: &PalsNumericalScopeV5,
        cwd: &Path,
        r: &G0NumericalBootstrapRegistrationV5,
        plan: &Pin,
        until: Instant,
        cleanup_until: Instant,
    ) -> Result<numerical::NativeCaptureConfig, ArenaError> {
        let path = |p| s.asset(p).map(|a| cwd.join(&a.snapshot_path));
        let cuda = s.model.backend == PalsModelBackendV3::OrtCuda;
        let inventory = if cuda {
            Some(s.asset(Purpose::CudaControlInventory)?)
        } else {
            None
        };
        let capture = numerical::NativeCaptureConfig {
            case_plan: plan.path.clone(),
            case_plan_sha256: plan.sha256.clone(),
            export_manifest: path(Purpose::ExportManifest)?,
            runtime_library: n.runtime_library.path.clone(),
            runtime_cache: n.runtime_cache.clone(),
            output_root: r.native_output_root.clone(),
            output_root_admission: None,
            provider: if cuda { "cuda-control" } else { "cpu" }.into(),
            cuda_bundle: n.cuda_bundle.as_ref().map(|p| p.path.clone()),
            cuda_inventory: inventory.map(|a| cwd.join(&a.snapshot_path)),
            cuda_inventory_sha256: inventory.map(|a| a.artifact.sha256.clone()),
            cudnn_conv_search: Some(n.native_options.cudnn_conv_search.clone()),
            gpu_device_id: n.native_options.cuda_device_id.unwrap_or(0),
            gpu_arena_bytes: usize::try_from(n.native_options.cuda_arena_bytes.unwrap_or(0))
                .map_err(|_| invalid("bootstrap native arena conversion"))?,
            deadline_seconds: until.saturating_duration_since(Instant::now()).as_secs(),
            cleanup_seconds: cleanup_until.duration_since(until).as_secs(),
        };
        check(
            capture.deadline_seconds > 0
                && capture.runtime_library == path(Purpose::RuntimeLibrary)?
                && !n.native_options.approximate_warm
                && !n.native_options.device_public_memory,
            "bootstrap original native runtime/Fresh/provider deadline",
        )?;
        if cuda {
            check(
                capture.cuda_bundle == Some(path(Purpose::CudaManifest)?)
                    && Some(capture.gpu_device_id) == n.native_options.cuda_device_id
                    && u64::try_from(capture.gpu_arena_bytes).ok()
                        == n.native_options.cuda_arena_bytes,
                "bootstrap native CUDA actual bundle/device/arena differs",
            )?;
        } else {
            check(
                capture.cuda_bundle.is_none()
                    && capture.cuda_inventory.is_none()
                    && capture.cuda_inventory_sha256.is_none(),
                "bootstrap CPU inherited CUDA artifacts",
            )?;
        }
        Ok(capture)
    }
    #[allow(
        clippy::too_many_arguments,
        reason = "One original owner retains scopes and bounded physical completion custody"
    )]
    fn finalize(
        r: &G0NumericalBootstrapRegistrationV5,
        i: &G0NumericalPreInitScopeV5,
        s: &mut PalsNumericalScopeV5,
        plan: &Value,
        references: &BTreeMap<String, Pin>,
        cleanup_until: Instant,
        cancel: &AtomicBool,
        running: &mut Running,
        credit: &mut ReadCredit<'_>,
    ) -> Result<Completed, ArenaError> {
        check(
            !cancel.load(Ordering::Acquire),
            "bootstrap numerical admission cancelled",
        )?;
        let preserve_cancel = AtomicBool::new(false);
        let mut cleanup = ReadCredit {
            remaining: credit.remaining,
            maximum: credit.maximum,
            until: cleanup_until,
            cancel: &preserve_cancel,
            account: Arc::clone(&credit.account),
        };
        let result = (|| -> Result<Completed, ArenaError> {
            let credit = &mut cleanup;
            let held = running
                .held
                .as_mut()
                .expect("bootstrap final original custody");
            let mut pairs = Vec::new();
            let mut total = 0u64;
            for group in plan["groups"]
                .as_array()
                .ok_or_else(|| invalid("bootstrap final groups absent"))?
            {
                let id = group["id"]
                    .as_str()
                    .ok_or_else(|| invalid("bootstrap final group ID"))?;
                let reference = references
                    .get(&format!("reference.{id}.json"))
                    .ok_or_else(|| invalid("bootstrap actual reference absent"))?;
                let native = numerical::observe_raw(
                    &r.native_output_root.join(format!("native-{id}.json")),
                    credit,
                )?;
                let mut artifacts = Vec::new();
                for (label, pin) in [("reference", reference), ("native", &native)] {
                    total = total
                        .checked_add(pin.bytes)
                        .filter(|n| *n <= 64 << 20)
                        .ok_or_else(|| invalid("bootstrap total raw payload cap"))?;
                    let file = common::open_file(&pin.path)?;
                    let raw = common::verify_bytes(&file, pin, true, credit)?;
                    super::super::g0_cuda_artifact_tree::ensure_output_credit(
                        held,
                        r.max_output_bytes,
                        raw.len() as u64,
                        cleanup_until,
                        &preserve_cancel,
                    )?;
                    let copy = common::write_owned_new(
                        &i.private_cwd
                            .join(format!("inputs/g0-bootstrap-{label}-{id}.json")),
                        &raw,
                    )?;
                    let a = asset(
                        &copy.pin,
                        &i.private_cwd,
                        Purpose::NumericalRawEvidence,
                        &i.model.origin,
                    )?;
                    artifacts.push(a.artifact.clone());
                    s.assets.push(a);
                    held.checkpoints.push(copy);
                }
                let (file, _) = common::read_observed_output(native, MAX_JSON, credit, false)?;
                held.checkpoints.push(file);
                pairs.push(rz_experiments::PalsNumericalRunPairV5 {
                    reference: artifacts.remove(0),
                    native: artifacts.remove(0),
                });
            }
            let acceptance = rz_experiments::PalsNumericalAcceptanceV5 {
                schema_version: 5,
                domain: rz_experiments::PALS_NUMERICAL_ACCEPTANCE_V5_DOMAIN.into(),
                bindings: rz_experiments::numerical_scope_bindings_v5(s)?,
                invocation: rz_experiments::PalsNumericalInvocationV5::Fresh,
                batch: 1,
                runs: pairs,
            };
            let raw = serde_json::to_vec(&acceptance).map_err(|e| invalid(e.to_string()))?;
            check(
                raw.len() <= 256 << 10,
                "bootstrap bounded actual acceptance",
            )?;
            super::super::g0_cuda_artifact_tree::ensure_output_credit(
                held,
                r.max_output_bytes,
                raw.len() as u64,
                cleanup_until,
                &preserve_cancel,
            )?;
            let acceptance_pin =
                common::write_original_owned_new(&r.acceptance_path, &raw, held, credit)?;
            s.assets.push(asset(
                &acceptance_pin,
                &i.private_cwd,
                Purpose::NumericalAcceptance,
                &i.model.origin,
            )?);
            let mut read = |a: &ArtifactRef, n: usize| {
                original::read_assets(&s.assets, &i.private_cwd, held, a, n, credit)
            };
            let admission =
                rz_experiments::validate_pals_numerical_scope_acceptance_v5(s, &raw, &mut read)?;
            let export = read(&s.asset(Purpose::ExportManifest)?.artifact, 256 << 10)?;
            let domain =
                rz_experiments::validate_pals_scope_artifact_domain_v5(s, &export, &mut read)?;
            check(
                domain.artifact_domain == rz_experiments::PalsArtifactDomainV5::Untrained
                    && !domain.tensor_optimizer_content_verified
                    && !domain.arena_accepted,
                "bootstrap numerical cannot issue startup/trained/game authority",
            )?;
            common::revalidate(held, credit)?;
            let staging_observation = super::super::g0_cuda_artifact_tree::scan_staging(
                held,
                r.max_output_bytes,
                cleanup_until,
                &preserve_cancel,
            )?;
            check(
                running.processes.len() == 3,
                "bootstrap exact original prepare/reference/native process chain",
            )?;
            let mut parent = serde_json::json!({"schema":"rz-pals-original-g0-bootstrap-numerical-parent/2","numerical_id":r.numerical_id,
                "launch_registration":running.registration_pin,"pre_init_scope":r.pre_init_scope,"case_spec":r.case_spec,"native_recipe":r.native_recipe,
                "scope":s,"admission":admission,"artifact_domain":domain,"acceptance":acceptance_pin,"resources_declared":i.resources,"resources_applied":"unknown",
                "initialization_seed":23,"initial_model_namespace_epoch":1,"trained":false,"optimizer_updates_executed":0,"training_authority_issued":false,"collection_authority_issued":false,
                "numerical_only":true,"startup_admission_not_issued":true,"startup_config_or_lock_issued":false,"arena_accepted":false,
                "processes":running.processes.iter().map(|p|serde_json::json!({"receipt":p.process().receipt,"transport":p.observation()})).collect::<Vec<_>>(),"child_receipts":running.receipts,
                "charged_read_bytes":running.account.charged.load(Ordering::Acquire),"charged_read_bytes_scope":"before_parent_receipt_publication",
                "parent_receipt_readback_counted_in_live_owner":true,"read_usage_uncertain":running.account.uncertain.load(Ordering::Acquire),
                "staging_cap_bytes":r.max_output_bytes,"original_deadline_preserved":true,"first_original_registration_descriptor_retained":true,
                "original_g0_initializer_export_process_and_fds_retained":true,"physical_completion":"confirmed_native_join","automatic_retry":false,
                "cuda_actual_plan":"unknown","actual_tf32":"unknown","sole_gpu_lease_observed_here":false});
            if held.cuda_artifact_tree.is_some() {
                parent["managed_staging_observation"] = staging_observation;
            }
            let raw = serde_json::to_vec(&parent).map_err(|e| invalid(e.to_string()))?;
            check(
                raw.len() as u64 <= MAX_JSON,
                "bootstrap bounded parent receipt",
            )?;
            super::super::g0_cuda_artifact_tree::ensure_output_credit(
                held,
                r.max_output_bytes,
                raw.len() as u64,
                cleanup_until,
                &preserve_cancel,
            )?;
            common::write_original_owned_new(&r.parent_receipt_path, &raw, held, credit)?;
            // Preservation used its false cancel solely to retain evidence.
            // Issuance must rejoin the original cancel and same whole deadline
            // after the last write/sync; ledger and remaining credit stay intact.
            common::check_original_g0_issue(credit, cancel)?;
            Ok((s.clone(), admission, acceptance_pin, parent))
        })();
        credit.remaining = cleanup.remaining;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bindings() -> PalsNumericalBindingsV5 {
        PalsNumericalBindingsV5 {
            model_profile: "virtual_review_v1".into(),
            model_schema: rz_experiments::PALS_V5_MODEL_SCHEMA.into(),
            model_semantics: rz_experiments::PALS_V5_MODEL_SEMANTICS.into(),
            encoding_schema: rz_experiments::PALS_V5_ENCODING_SCHEMA.into(),
            model_semantic_sha256: "a".repeat(64),
            encoding_sha256: "b".repeat(64),
            model_epoch: 1,
            weights_sha256: "c".repeat(64),
            export_manifest_sha256: "d".repeat(64),
            graphs: BTreeMap::from([
                ("public".into(), "e".repeat(64)),
                ("shared_pc".into(), "f".repeat(64)),
                ("validator".into(), "0".repeat(64)),
            ]),
            runtime_library_sha256: "1".repeat(64),
            cuda_libraries: BTreeSet::new(),
            cuda_manifest_sha256: None,
            backend: PalsModelBackendV3::OrtCpu,
            precision: PalsPrecisionV3::Fp32,
            tf32: false,
            source_components: [
                "model_native",
                "model_encoding",
                "nn_runtime",
                "model_reference",
                "model_export",
            ]
            .into_iter()
            .map(|key| (key.into(), "2".repeat(64)))
            .collect(),
        }
    }
    // Small typed CPU input metadata only. This fixture never creates files,
    // imports a framework, starts a process, performs NN work or issues authority.
    fn cases() -> G0NumericalCaseSpecV5 {
        use rz_eval::pals_model::{
            PalsCandidateToken, PalsRole,
            virtual_review::{
                PalsVirtualReviewInput, ReviewAnchorInput, ReviewQueryInput, ReviewQueryPurpose,
            },
        };
        let mut groups = Vec::new();
        for (name, role) in [
            ("proposer", PalsRole::Proposer),
            ("critic", PalsRole::Critic),
            ("validator", PalsRole::Validator),
        ] {
            for bucket in [8, 16, 32, 64, 128, 256] {
                let mut rows = Vec::new();
                for anchors in 1..=4 {
                    for white in [false, true] {
                        let mut metadata = [0.; 16];
                        metadata[0] = f32::from(u8::from(white));
                        let input = PalsVirtualReviewInput {
                            schema: rz_eval::pals_model::virtual_review::INPUT_SCHEMA.into(),
                            role,
                            anchors: vec![
                                ReviewAnchorInput {
                                    board: vec![0; 64],
                                    metadata
                                };
                                anchors
                            ],
                            records: Vec::new(),
                            query: ReviewQueryInput {
                                purpose: if role == PalsRole::Validator {
                                    ReviewQueryPurpose::TaskPolicy
                                } else {
                                    ReviewQueryPurpose::ReplyPolicy
                                },
                                perspective: if white { 1 } else { -1 },
                                prefix: Vec::new(),
                                proposal: vec![
                                    PalsCandidateToken {
                                        from: 1,
                                        to: 18,
                                        promotion: 0
                                    };
                                    bucket
                                ],
                                counter: Vec::new(),
                                counter_present: false,
                                anchors: [0, 0, 0],
                            },
                            candidates: vec![PalsCandidateToken {
                                from: 1,
                                to: 18,
                                promotion: 0,
                            }],
                            divergences: Vec::new(),
                            token_budget: 512,
                        };
                        rows.push(serde_json::json!({"id":format!("{name}_L{bucket}_A{anchors}_W{}",u8::from(white)),
                            "input":{"model_input_kind":"virtual_review","input":input}}));
                    }
                }
                groups.push(serde_json::json!({"id":format!("{name}_L{bucket}"),"role":name,"line_bucket":bucket,"cases":rows}));
            }
        }
        G0NumericalCaseSpecV5 {
            schema: G0_CASE_SPEC_V5.into(),
            immutable_bindings: G0NumericalImmutableBindingsV5::from_actual(&bindings()),
            groups: Value::Array(groups),
        }
    }
    #[test]
    fn pre_init_model_rejects_future_optional_weights_and_missing_fields() {
        let valid = serde_json::json!({"profile":"virtual_review_v1","semantic_sha256":"a".repeat(64),"encoding_sha256":"b".repeat(64),
            "implementation_sha256":"c".repeat(64),"backend":"ort_cuda","origin":{"source":"local-source","license":"MIT"}});
        assert!(serde_json::from_value::<G0NumericalModelRecipeV5>(valid.clone()).is_ok());
        for name in [
            "weights",
            "model_epoch",
            "export_manifest_sha256",
            "graphs",
            "numerical_acceptance",
        ] {
            let mut value = valid.clone();
            value[name] = Value::Null;
            assert!(serde_json::from_value::<G0NumericalModelRecipeV5>(value).is_err());
        }
        let mut missing = valid;
        missing
            .as_object_mut()
            .unwrap()
            .remove("implementation_sha256");
        assert!(serde_json::from_value::<G0NumericalModelRecipeV5>(missing).is_err());
    }
    #[test]
    fn sealed_cases_preserve_order_and_accept_only_actual_output_rebinding() {
        let spec = cases();
        spec.validate()
            .expect("typed sealed G0 cases must satisfy the original bounded group/name guards");
        let seal = serde_json::to_vec(&spec.groups).unwrap();
        let mut actual = bindings();
        actual.weights_sha256 = "8".repeat(64);
        actual.export_manifest_sha256 = "9".repeat(64);
        let plan = spec.materialize(&actual).unwrap();
        assert_eq!(serde_json::to_vec(&plan["groups"]).unwrap(), seal);
        assert_eq!(plan["bindings"]["weights_sha256"], actual.weights_sha256);
        for mutation in 0..3 {
            let mut changed = actual.clone();
            match mutation {
                0 => changed.runtime_library_sha256 = "3".repeat(64),
                1 => changed.source_components.clear(),
                _ => changed.encoding_sha256 = "4".repeat(64),
            }
            assert!(spec.materialize(&changed).is_err());
        }
    }
    #[test]
    fn case_spec_refuses_future_refs_partial_coverage_and_typed_scope_mismatch() {
        let spec = cases();
        for name in [
            "weights_sha256",
            "export_manifest_sha256",
            "graphs",
            "model_epoch",
        ] {
            let mut json = serde_json::to_value(&spec).unwrap();
            json["immutable_bindings"][name] = Value::Null;
            assert!(serde_json::from_value::<G0NumericalCaseSpecV5>(json).is_err());
        }
        let mut partial = spec.clone();
        partial.groups.as_array_mut().unwrap().pop();
        assert!(partial.validate().is_err());
        let mut wrong = spec.clone();
        wrong.groups[0]["cases"][0]["input"]["input"]["role"] = serde_json::json!("critic");
        assert!(wrong.validate().is_err());
        let mut bucket = spec;
        bucket.groups[0]["line_bucket"] = serde_json::json!(16);
        assert!(bucket.validate().is_err());
    }
    #[test]
    fn intent_binding_contains_only_existing_input_scope() {
        let artifact = rz_experiments::ArtifactRef {
            path: "runtime/original.so".into(),
            sha256: bindings().runtime_library_sha256,
            bytes: 1,
            source: "metadata-only".into(),
            license: "MIT".into(),
        };
        let intent = G0NumericalPreInitScopeV5 {
            schema: G0_PRE_INIT_SCOPE_V5.into(),
            source_commit: "a".repeat(40),
            source_components: bindings().source_components,
            model: G0NumericalModelRecipeV5 {
                profile: "virtual_review_v1".into(),
                semantic_sha256: "a".repeat(64),
                encoding_sha256: "b".repeat(64),
                implementation_sha256: "c".repeat(64),
                backend: PalsModelBackendV3::OrtCpu,
                origin: G0ArtifactOriginV5 {
                    source: "metadata-only".into(),
                    license: "MIT".into(),
                },
            },
            private_cwd: PathBuf::from("metadata-only"),
            native_options: PalsNativeStartupOptionsV5 {
                native_loading_profile: None,
                cuda_device_id: None,
                cuda_arena_bytes: None,
                cache_public_memory: false,
                device_public_memory: false,
                cudnn_conv_search: "legacy_implicit_exhaustive".into(),
                approximate_warm: false,
                startup_ms: 100,
                drain_ms: 100,
            },
            resources: PalsResourcePlanV5::default(),
            input_assets: vec![PalsStartupAssetV5 {
                artifact,
                snapshot_path: "inputs/runtime.so".into(),
                purpose: Purpose::RuntimeLibrary,
            }],
            input_pins: Vec::new(),
        };
        assert_eq!(
            G0NumericalImmutableBindingsV5::from_intent(&intent).unwrap(),
            G0NumericalImmutableBindingsV5::from_actual(&bindings())
        );
        let mut missing = intent;
        missing.input_assets.clear();
        assert!(G0NumericalImmutableBindingsV5::from_intent(&missing).is_err());
    }
    #[test]
    fn native_recipe_rejects_generated_capture_fields() {
        let valid = serde_json::json!({"schema":G0_NATIVE_RECIPE_V5,
            "runtime_library":{"path":"runtime.so","sha256":"a".repeat(64),"bytes":1},"runtime_cache":"cache","cuda_bundle":null,
            "native_options":{"native_loading_profile":null,"cuda_device_id":null,"cuda_arena_bytes":null,"cache_public_memory":false,
                "device_public_memory":false,"cudnn_conv_search":"legacy_implicit_exhaustive","approximate_warm":false,"startup_ms":1,"drain_ms":1}});
        assert!(serde_json::from_value::<G0NativeCaptureRecipeV5>(valid.clone()).is_ok());
        for name in [
            "export_manifest",
            "case_plan",
            "cuda_inventory_sha256",
            "numerical_acceptance",
        ] {
            let mut value = valid.clone();
            value[name] = serde_json::json!("future");
            assert!(serde_json::from_value::<G0NativeCaptureRecipeV5>(value).is_err());
        }
    }
    #[cfg(target_os = "linux")]
    fn registration() -> G0NumericalBootstrapRegistrationV5 {
        let pin = |name: &str| Pin {
            path: PathBuf::from(format!("/original/{name}")),
            sha256: "a".repeat(64),
            bytes: 1,
        };
        G0NumericalBootstrapRegistrationV5 {
            schema: G0_BOOTSTRAP_LAUNCH_V5.into(),
            numerical_id: "metadata-g0".into(),
            source_commit: "b".repeat(40),
            pre_init_scope: pin("intent.json"),
            case_spec: pin("cases.json"),
            native_recipe: pin("native.json"),
            rules_profile: pin("rules.json"),
            native_executable: pin("native"),
            python_runtime: GenerationTrainingPythonRuntimeV3 {
                version: "3.12.3".into(),
                executable: pin("python"),
            },
            dependency_directories: vec![PathBuf::from("/trusted")],
            runtime_module_pins: vec![Pin {
                path: PathBuf::from("/trusted/torch.py"),
                sha256: "c".repeat(64),
                bytes: 1,
            }],
            staging_root: PathBuf::from("/managed"),
            initialization_directory: PathBuf::from("/managed/private/inputs/init"),
            export_directory: PathBuf::from("/managed/private/inputs/export"),
            reference_output_root: PathBuf::from("/managed/reference"),
            native_output_root: PathBuf::from("/managed/native"),
            prepare_receipt_path: PathBuf::from("/managed/prepare.json"),
            reference_receipt_path: PathBuf::from("/managed/reference.json"),
            acceptance_path: PathBuf::from("/managed/private/inputs/acceptance.json"),
            parent_receipt_path: PathBuf::from("/managed/parent.json"),
            max_wall_ns: 1_000_000_000,
            max_output_bytes: MAX_STAGING,
            max_read_bytes: MAX_STAGING,
        }
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn bootstrap_registration_preserves_original_finite_credit_and_disjoint_paths() {
        let valid = registration();
        assert!(valid.validate().is_ok());
        let mut alias = valid.clone();
        alias.native_output_root = alias.reference_output_root.clone();
        assert!(alias.validate().is_err());
        let mut credit = valid.clone();
        credit.max_read_bytes = 4 * MAX_STAGING + 1;
        assert!(credit.validate().is_err());
        let mut time = valid.clone();
        time.max_wall_ns = 0;
        assert!(time.validate().is_err());
        let mut reused = valid;
        reused.case_spec.path = reused.export_directory.join("cases.json");
        assert!(reused.validate().is_err());
    }
    #[cfg(not(target_os = "linux"))]
    #[test]
    fn bootstrap_platform_refusal_opens_no_inputs() {
        let failure = run_g0_bootstrap_numerical_original_v5(
            Path::new("not-present"),
            "bad",
            Arc::new(AtomicBool::new(false)),
        )
        .err()
        .expect("unsupported");
        assert!(failure.processes.is_empty());
        assert_eq!(failure.charged_read_bytes, 0);
        assert!(failure.registration_pin.is_none());
    }
}
