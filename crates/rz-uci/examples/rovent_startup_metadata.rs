//! Bounded, declaration-only metadata for an original Rovent startup JSON file.
//!
//! Usage: `rovent_startup_metadata --config=PATH.json [--backend-metadata=PATH.json]`
//! Optional: `--position-command=position startpos moves e2e4` (one argv value).
//! Optional: `--session-implementation-root-policy=dynamic-go|exact-registered`
//! declares the existing session digest without constructing it. External R
//! additionally requires `--input-lock-metadata=PATH.json`; the original typed
//! lock and its configuration join are validated without admitting assets.
//! Standalone: `--runtime-bundle-metadata=PATH.json` declares the original bundle
//! JSON byte pin and its existing typed semantic digest, without opening libraries.
//! The existing startup codec owns validation, canonicalization and domain hashes.
//! This example never opens referenced assets or creates a native/runtime owner.

#![forbid(unsafe_code)]

#[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
fn main() -> std::process::ExitCode {
    use std::io::Write as _;
    let _ = std::io::stderr().lock().write_all(
        b"{\"schema\":\"rz-rovent-startup-metadata-error/1\",\"code\":\"unsupported_build_requires_pals_adaptive_and_onnx_cpu\",\"input_read\":false,\"actual_startup_ready\":false,\"native_authority\":false,\"training\":false}\n",
    );
    std::process::ExitCode::from(2)
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
fn main() -> std::process::ExitCode {
    metadata::main()
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
mod metadata {
    use rz_eval::error::BackendError;
    use rz_eval::runtime_pin::CudaRuntimeBundleSpec;
    use rz_experiments::{ManifestError, RoventDevice, RoventInputLock};
    use rz_position::contracts::{profile_digest, rules_snapshot_semantic_digest};
    use rz_position::{BoardMove, Position, PositionError};
    use rz_uci::parser::{self, Command, ParseError, ParserLimits, PositionBase, PositionSpec};
    use rz_uci::rovent_bootstrap::configuration::{
        ExecutionConfiguration, LoadingProfile, ROVENT_STARTUP_CONFIGURATION_DOMAIN,
        ResearchValueConfiguration, StartupConfigurationDocument,
        v2::ROVENT_STARTUP_CONFIGURATION_V2_DOMAIN, v3::ROVENT_STARTUP_CONFIGURATION_V3_DOMAIN,
        v4::ROVENT_STARTUP_CONFIGURATION_V4_DOMAIN,
    };
    use rz_uci::rovent_bootstrap::configuration_digest_from_metadata;
    use rz_uci::rovent_bootstrap::external_session::external_researcher_implementation_digest_from_metadata;
    use rz_uci::rovent_bootstrap::nnuec_session::RoventNnuecSessionDriver;
    use rz_uci::rovent_bootstrap::pc_session::RoventPcSessionDriver;
    use rz_uci::rovent_root_admission::RoventRootAdmissionPolicy;
    use serde::de::{MapAccess, SeqAccess, Visitor};
    use serde::{Deserialize, Deserializer, Serialize};
    use sha2::{Digest as _, Sha256};
    use std::collections::BTreeSet;
    use std::ffi::OsString;
    use std::fmt;
    use std::fs::File;
    use std::io::{self, Read, Write};
    use std::path::PathBuf;
    use std::process::ExitCode;

    const INPUT_MAX_BYTES: usize = 64 << 10;
    const LOCK_INPUT_MAX_BYTES: usize = 2 << 20;
    const OUTPUT_MAX_BYTES: usize = 256 << 10;
    // A failed/partial stdout write may still be followed by this one diagnostic.
    const DIAGNOSTIC_MAX_BYTES: usize = 4 << 10;
    const STDOUT_MAX_BYTES: usize = OUTPUT_MAX_BYTES - DIAGNOSTIC_MAX_BYTES;
    const ERROR_DETAIL_MAX_BYTES: usize = 512;

    #[derive(Debug)]
    enum Error {
        Arguments,
        InputKind,
        InputBudget,
        BackendDeclaration(&'static str),
        RuntimeBundleDeclaration(Box<BackendError>),
        SessionImplementationAssembly,
        SessionImplementation(Box<BackendError>),
        SessionImplementationLock(&'static str),
        InputLock(Box<ManifestError>),
        InputLockBudget,
        PositionSyntax(ParseError),
        PositionCommand,
        PositionRules {
            stage: &'static str,
            zero_based_ply: Option<usize>,
            cause: PositionError,
        },
        Io {
            stage: &'static str,
            cause: io::Error,
        },
        Json(serde_json::Error),
        Configuration(Box<BackendError>),
        CanonicalUtf8(std::str::Utf8Error),
        OutputEncoding(serde_json::Error),
        OutputBudget,
    }
    impl Error {
        fn code(&self) -> &'static str {
            match self {
                Self::Arguments => "invalid_arguments",
                Self::InputKind => "unsupported_input_file",
                Self::InputBudget => "configuration_byte_budget",
                Self::BackendDeclaration(_) => "backend_metadata_declaration_rejected",
                Self::RuntimeBundleDeclaration(_) => "runtime_bundle_declaration_rejected",
                Self::SessionImplementationAssembly => {
                    "session_implementation_assembly_unsupported"
                }
                Self::SessionImplementation(_) => "session_implementation_declaration_rejected",
                Self::SessionImplementationLock(_) => "session_implementation_input_lock_rejected",
                Self::InputLock(_) => "input_lock_metadata_rejected",
                Self::InputLockBudget => "input_lock_byte_budget",
                Self::PositionSyntax(_) => "position_command_syntax_rejected",
                Self::PositionCommand => "position_command_required",
                Self::PositionRules { stage, .. } => stage,
                Self::Io { stage, .. } => stage,
                Self::Json(_) => "invalid_or_duplicate_json",
                Self::Configuration(_) => "startup_configuration_rejected",
                Self::CanonicalUtf8(_) => "canonical_utf8_rejected",
                Self::OutputEncoding(_) => "metadata_encoding_failed",
                Self::OutputBudget => "metadata_output_budget",
            }
        }
    }
    impl fmt::Display for Error {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Arguments => f.write_str("expected --config=PATH.json and optional --backend-metadata=PATH.json / --position-command=POSITION / --session-implementation-root-policy=dynamic-go|exact-registered; external session metadata also requires --input-lock-metadata=PATH.json; or standalone --runtime-bundle-metadata=PATH.json, at most once each"),
                Self::InputKind => f.write_str("input must be a regular non-symlink JSON file"),
                Self::InputBudget => f.write_str("each input must contain 1..=65536 bytes"),
                Self::BackendDeclaration(detail) => f.write_str(detail),
                Self::RuntimeBundleDeclaration(cause) => write!(f, "runtime bundle declaration: {cause}"),
                Self::SessionImplementationAssembly => f.write_str("session implementation metadata supports roles-only PC, own-NNUEC PCR, or external researcher declarations"),
                Self::SessionImplementation(cause) => write!(f, "session implementation declaration: {cause}"),
                Self::SessionImplementationLock(detail) => f.write_str(detail),
                Self::InputLock(cause) => write!(f, "input lock metadata: {cause}"),
                Self::InputLockBudget => f.write_str("input lock metadata requires 1..=2097152 bytes"),
                Self::PositionSyntax(cause) => write!(f, "position command: {cause}"),
                Self::PositionCommand => f.write_str("only one UCI position command is accepted"),
                Self::PositionRules { stage, zero_based_ply, cause } => {
                    write!(f, "{stage}")?;
                    if let Some(ply) = zero_based_ply { write!(f, " at zero-based ply {ply}")?; }
                    write!(f, ": {cause}")
                }
                Self::Io { stage, cause } => write!(f, "{stage}: {cause}"),
                Self::Json(cause) => write!(f, "JSON syntax or duplicate key: {cause}"),
                Self::Configuration(cause) => write!(f, "startup declaration: {cause}"),
                Self::CanonicalUtf8(cause) => write!(f, "canonical JSON UTF-8: {cause}"),
                Self::OutputEncoding(cause) => write!(f, "metadata JSON: {cause}"),
                Self::OutputBudget => f.write_str("metadata exceeds reserved stdout budget"),
            }
        }
    }
    fn io_error(stage: &'static str, cause: io::Error) -> Error {
        Error::Io { stage, cause }
    }
    fn configuration_error(cause: BackendError) -> Error {
        Error::Configuration(Box::new(cause))
    }

    #[derive(Debug, PartialEq, Eq)]
    struct Arguments {
        configuration: PathBuf,
        backend_metadata: Option<PathBuf>,
        position_command: Option<PositionCommand>,
        session_implementation_root_policy: Option<SessionImplementationPolicy>,
        input_lock_metadata: Option<PathBuf>,
    }
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum SessionImplementationPolicy {
        ExactRegistered,
        DynamicGo,
    }
    impl SessionImplementationPolicy {
        fn actual(self) -> RoventRootAdmissionPolicy {
            match self {
                Self::ExactRegistered => RoventRootAdmissionPolicy::ExactRegistered,
                Self::DynamicGo => RoventRootAdmissionPolicy::DynamicGo,
            }
        }
        fn wire_name(self) -> &'static str {
            match self {
                Self::ExactRegistered => "exact_registered",
                Self::DynamicGo => "dynamic_go",
            }
        }
    }
    #[derive(Debug, PartialEq, Eq)]
    enum Invocation {
        Startup(Arguments),
        RuntimeBundleMetadata(PathBuf),
    }
    #[derive(Debug, PartialEq, Eq)]
    struct PositionCommand {
        raw: String,
        position: PositionSpec,
    }
    fn parse_position_command(raw: &str) -> Result<PositionCommand, Error> {
        // Use the ordinary bounded parser: 16KiB, at most 2048 moves and its
        // existing one-line/ASCII/FEN-field syntax. This creates no UCI owner.
        match parser::parse(raw, ParserLimits::default()).map_err(Error::PositionSyntax)? {
            Command::Position(position) => Ok(PositionCommand {
                raw: raw.to_owned(),
                position,
            }),
            _ => Err(Error::PositionCommand),
        }
    }
    fn json_path(text: &str) -> Result<PathBuf, Error> {
        if text.is_empty() {
            return Err(Error::Arguments);
        }
        let path = PathBuf::from(text);
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            return Err(Error::Arguments);
        }
        Ok(path)
    }
    fn parse_arguments(arguments: impl IntoIterator<Item = OsString>) -> Result<Invocation, Error> {
        let mut configuration = None;
        let mut backend_metadata = None;
        let mut position_command = None;
        let mut session_implementation_root_policy = None;
        let mut input_lock_metadata = None;
        let mut runtime_bundle_metadata = None;
        for argument in arguments {
            let text = argument.to_str().ok_or(Error::Arguments)?;
            if let Some(path) = text.strip_prefix("--config=") {
                if configuration.is_some() {
                    return Err(Error::Arguments);
                }
                configuration = Some(json_path(path)?);
            } else if let Some(path) = text.strip_prefix("--backend-metadata=") {
                if backend_metadata.is_some() {
                    return Err(Error::Arguments);
                }
                backend_metadata = Some(json_path(path)?);
            } else if let Some(command) = text.strip_prefix("--position-command=") {
                if position_command.is_some() {
                    return Err(Error::Arguments);
                }
                position_command = Some(parse_position_command(command)?);
            } else if let Some(policy) = text.strip_prefix("--session-implementation-root-policy=")
            {
                if session_implementation_root_policy.is_some() {
                    return Err(Error::Arguments);
                }
                session_implementation_root_policy = Some(match policy {
                    "dynamic-go" => SessionImplementationPolicy::DynamicGo,
                    "exact-registered" => SessionImplementationPolicy::ExactRegistered,
                    _ => return Err(Error::Arguments),
                });
            } else if let Some(path) = text.strip_prefix("--input-lock-metadata=") {
                if input_lock_metadata.is_some() {
                    return Err(Error::Arguments);
                }
                input_lock_metadata = Some(json_path(path)?);
            } else if let Some(path) = text.strip_prefix("--runtime-bundle-metadata=") {
                if runtime_bundle_metadata.is_some() {
                    return Err(Error::Arguments);
                }
                runtime_bundle_metadata = Some(json_path(path)?);
            } else {
                return Err(Error::Arguments);
            }
        }
        if let Some(bundle) = runtime_bundle_metadata {
            return if configuration.is_none()
                && backend_metadata.is_none()
                && position_command.is_none()
                && session_implementation_root_policy.is_none()
                && input_lock_metadata.is_none()
            {
                Ok(Invocation::RuntimeBundleMetadata(bundle))
            } else {
                Err(Error::Arguments)
            };
        }
        if input_lock_metadata.is_some() && session_implementation_root_policy.is_none() {
            return Err(Error::Arguments);
        }
        Ok(Invocation::Startup(Arguments {
            configuration: configuration.ok_or(Error::Arguments)?,
            backend_metadata,
            position_command,
            session_implementation_root_policy,
            input_lock_metadata,
        }))
    }

    fn read_configuration(path: &std::path::Path) -> Result<Vec<u8>, Error> {
        read_bounded_json(path, INPUT_MAX_BYTES)
    }
    fn read_input_lock(path: &std::path::Path) -> Result<Vec<u8>, Error> {
        read_bounded_json(path, LOCK_INPUT_MAX_BYTES).map_err(|error| match error {
            Error::InputBudget => Error::InputLockBudget,
            other => other,
        })
    }
    fn read_bounded_json(path: &std::path::Path, maximum: usize) -> Result<Vec<u8>, Error> {
        let before = std::fs::symlink_metadata(path).map_err(|e| io_error("input_metadata", e))?;
        if !before.file_type().is_file() || before.file_type().is_symlink() {
            return Err(Error::InputKind);
        }
        if before.len() == 0 || before.len() > maximum as u64 {
            return Err(Error::InputBudget);
        }
        let file = File::open(path).map_err(|e| io_error("input_open", e))?;
        if !file
            .metadata()
            .map_err(|e| io_error("opened_input_metadata", e))?
            .is_file()
        {
            return Err(Error::InputKind);
        }
        let mut bytes = Vec::with_capacity(before.len() as usize);
        file.take((maximum + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|e| io_error("input_read", e))?;
        if bytes.is_empty() || bytes.len() > maximum {
            return Err(Error::InputBudget);
        }
        // The pin below identifies these actual read bytes, not an immutable
        // filesystem snapshot or the identity of any referenced native asset.
        Ok(bytes)
    }
    fn check_input_budget(bytes: &[u8]) -> Result<(), Error> {
        if bytes.is_empty() || bytes.len() > INPUT_MAX_BYTES {
            Err(Error::InputBudget)
        } else {
            Ok(())
        }
    }

    // This pass stores only keys of the current object. It adds no startup
    // encoder/schema; the existing typed codec still decides all fields/values.
    // Decoded key comparison also rejects escaped spellings of the same key.
    struct UniqueJson;
    impl<'de> Deserialize<'de> for UniqueJson {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            deserializer.deserialize_any(UniqueVisitor)
        }
    }
    struct UniqueVisitor;
    impl<'de> Visitor<'de> for UniqueVisitor {
        type Value = UniqueJson;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("JSON with unique object keys")
        }
        fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<Self::Value, A::Error> {
            let mut keys = BTreeSet::new();
            while let Some(key) = access.next_key::<String>()? {
                if !keys.insert(key) {
                    return Err(serde::de::Error::custom("duplicate JSON object key"));
                }
                access.next_value::<UniqueJson>()?;
            }
            Ok(UniqueJson)
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut access: A) -> Result<Self::Value, A::Error> {
            while access.next_element::<UniqueJson>()?.is_some() {}
            Ok(UniqueJson)
        }
        fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<Self::Value, E> {
            Ok(UniqueJson)
        }
        fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<Self::Value, E> {
            Ok(UniqueJson)
        }
        fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<Self::Value, E> {
            Ok(UniqueJson)
        }
        fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<Self::Value, E> {
            Ok(UniqueJson)
        }
        fn visit_str<E: serde::de::Error>(self, _: &str) -> Result<Self::Value, E> {
            Ok(UniqueJson)
        }
        fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
            Ok(UniqueJson)
        }
    }

    #[derive(Serialize)]
    struct BytePin {
        bytes: usize,
        sha256: String,
    }
    fn hex(bytes: &[u8]) -> String {
        const DIGITS: &[u8; 16] = b"0123456789abcdef";
        let mut text = String::with_capacity(bytes.len() * 2);
        for &byte in bytes {
            text.push(DIGITS[(byte >> 4) as usize] as char);
            text.push(DIGITS[(byte & 15) as usize] as char);
        }
        text
    }
    fn byte_pin(bytes: &[u8]) -> BytePin {
        BytePin {
            bytes: bytes.len(),
            sha256: hex(&Sha256::digest(bytes)),
        }
    }

    #[derive(Serialize)]
    struct RuntimeBundleMetadata {
        domain: &'static str,
        raw_json_sha256: String,
        bytes: usize,
        semantic_bundle_sha256: String,
        schema_version: u32,
        declared_only: bool,
        runtime_loading_executed: bool,
        sessions_created: u64,
        neural_forwards: u64,
        native_authority: bool,
        training: bool,
    }
    fn prepare_runtime_bundle_metadata(bytes: &[u8]) -> Result<Vec<u8>, Error> {
        check_input_budget(bytes)?;
        serde_json::from_slice::<UniqueJson>(bytes).map_err(Error::Json)?;
        // The typed parser and binary codec own the roster/aliases and digest.
        // This operation opens no declared file and constructs no runtime pin,
        // ORT session, loading profile or native/GPU owner.
        let text = std::str::from_utf8(bytes).map_err(Error::CanonicalUtf8)?;
        let bundle = CudaRuntimeBundleSpec::from_json(text)
            .map_err(|cause| Error::RuntimeBundleDeclaration(Box::new(cause)))?;
        let semantic = bundle
            .digest()
            .map_err(|cause| Error::RuntimeBundleDeclaration(Box::new(cause)))?;
        let raw = byte_pin(bytes);
        let output = RuntimeBundleMetadata {
            domain: "rz-rovent-runtime-bundle-declaration/1",
            raw_json_sha256: raw.sha256,
            bytes: raw.bytes,
            semantic_bundle_sha256: hex(&semantic),
            schema_version: bundle.schema_version,
            declared_only: true,
            runtime_loading_executed: false,
            sessions_created: 0,
            neural_forwards: 0,
            native_authority: false,
            training: false,
        };
        let mut encoded = serde_json::to_vec(&output).map_err(Error::OutputEncoding)?;
        encoded.push(b'\n');
        enforce_stdout_budget(encoded)
    }
    // Required tagged declarations distinguish explicit absence from omission.
    // No default, path scan or runtime hash is substituted for a missing value.
    #[derive(Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
    enum DeclaredDigest {
        Absent {},
        Sha256 { value: String },
    }
    impl DeclaredDigest {
        fn digest(&self) -> Result<Option<[u8; 32]>, Error> {
            match self {
                Self::Absent {} => Ok(None),
                Self::Sha256 { value } => declared_sha256(value).map(Some),
            }
        }
    }
    #[derive(Deserialize)]
    #[serde(tag = "kind", deny_unknown_fields)]
    enum DeclaredLoadingProfile {
        #[serde(rename = "absent")]
        Absent {},
        #[serde(rename = "eager_cuda12_cudnn9_v1")]
        EagerCuda12Cudnn9V1 {},
        #[serde(rename = "cu_dnn_shim_lazy_v1")]
        CuDnnShimLazyV1 {},
    }
    impl DeclaredLoadingProfile {
        fn selected(&self) -> Option<LoadingProfile> {
            match self {
                Self::Absent {} => None,
                Self::EagerCuda12Cudnn9V1 {} => Some(LoadingProfile::EagerCuda12Cudnn9V1),
                Self::CuDnnShimLazyV1 {} => Some(LoadingProfile::CuDnnShimLazyV1),
            }
        }
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct BackendInput {
        domain: String,
        device: RoventDevice,
        runtime_binary_sha256: String,
        runtime_bundle_sha256: DeclaredDigest,
        control_inventory_sha256: DeclaredDigest,
        loading_profile: DeclaredLoadingProfile,
    }
    fn declared_sha256(text: &str) -> Result<[u8; 32], Error> {
        if text.len() != 64
            || !text
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(Error::BackendDeclaration(
                "SHA256 declarations require 64 lowercase hexadecimal characters",
            ));
        }
        let mut bytes = [0; 32];
        for (out, pair) in bytes.iter_mut().zip(text.as_bytes().chunks_exact(2)) {
            let nibble = |b: u8| {
                if b.is_ascii_digit() {
                    b - b'0'
                } else {
                    b - b'a' + 10
                }
            };
            *out = (nibble(pair[0]) << 4) | nibble(pair[1]);
        }
        if bytes == [0; 32] {
            return Err(Error::BackendDeclaration(
                "an all-zero placeholder is not an independently registered runtime identity",
            ));
        }
        Ok(bytes)
    }
    #[derive(Serialize)]
    struct BackendMetadata {
        scope: &'static str,
        source_metadata: BytePin,
        backend_configuration_sha256: String,
        declared_runtime_binary_sha256: String,
        declared_runtime_bundle_sha256: Option<String>,
        declared_control_inventory_sha256: Option<String>,
        actual_runtime_identity_verified: bool,
        actual_provider_verified: bool,
    }
    fn prepare_backend(
        document: &StartupConfigurationDocument,
        raw: &[u8],
    ) -> Result<BackendMetadata, Error> {
        check_input_budget(raw)?;
        serde_json::from_slice::<UniqueJson>(raw).map_err(Error::Json)?;
        let input: BackendInput = serde_json::from_slice(raw).map_err(Error::Json)?;
        if input.domain != "rz-rovent-backend-metadata-input/1" {
            return Err(Error::BackendDeclaration(
                "unsupported backend metadata domain",
            ));
        }
        let native = document.fields().native;
        if input.loading_profile.selected() != native.loading_profile {
            return Err(Error::BackendDeclaration(
                "backend loading profile differs from the original startup document",
            ));
        }
        let binary = declared_sha256(&input.runtime_binary_sha256)?;
        let bundle = input.runtime_bundle_sha256.digest()?;
        let control = input.control_inventory_sha256.digest()?;
        // Presence means a declared explicit control selection only. It does
        // not establish inventory bytes, a loaded provider or GPU capability.
        let configuration = native
            .onnx(&input.device, control.is_some())
            .map_err(configuration_error)?;
        let digest = configuration_digest_from_metadata(
            configuration,
            native.loading_profile.map(LoadingProfile::actual),
            binary,
            bundle,
            control,
        )
        .map_err(configuration_error)?;
        Ok(BackendMetadata {
            scope: "independently_supplied_declarations_only",
            source_metadata: byte_pin(raw),
            backend_configuration_sha256: hex(&digest),
            declared_runtime_binary_sha256: hex(&binary),
            declared_runtime_bundle_sha256: bundle.map(|d| hex(&d)),
            declared_control_inventory_sha256: control.map(|d| hex(&d)),
            actual_runtime_identity_verified: false,
            actual_provider_verified: false,
        })
    }
    #[derive(Serialize)]
    struct Checks {
        unique_json_object_keys: &'static str,
        existing_closed_codec_and_declaration_constraints: &'static str,
        actual_lock_join: &'static str,
        compiled_native_capability_admission: &'static str,
        referenced_asset_identity: &'static str,
        backend_runtime_configuration_digest: &'static str,
        native_loading_provider_execution: &'static str,
        source_file_snapshot_stability: &'static str,
    }
    #[derive(Serialize)]
    struct RootMetadata {
        scope: &'static str,
        position_command: BytePin,
        root_state_and_history_sha256: String,
        rules_profile_sha256: String,
        root_authority: bool,
    }
    fn prepare_root_metadata(command: &PositionCommand) -> Result<RootMetadata, Error> {
        let mut position = match &command.position.base {
            PositionBase::StartPos => Position::startpos(),
            PositionBase::Fen(fen) => {
                Position::from_fen(fen).map_err(|cause| Error::PositionRules {
                    stage: "position_base_rejected",
                    zero_based_ply: None,
                    cause,
                })?
            }
        };
        for (ply, raw_move) in command.position.moves.iter().enumerate() {
            let mv = BoardMove::from_uci(raw_move).map_err(|cause| Error::PositionRules {
                stage: "position_move_syntax_rejected",
                zero_based_ply: Some(ply),
                cause,
            })?;
            position
                .make_move(mv)
                .map_err(|cause| Error::PositionRules {
                    stage: "position_move_rejected",
                    zero_based_ply: Some(ply),
                    cause,
                })?;
        }
        // from_fen retains Rules' UnknownPrefix; no missing past is invented.
        // This is a semantic snapshot observation, not a live root/Start seal.
        Ok(RootMetadata {
            scope: "rules_snapshot_semantic_observation_only",
            position_command: byte_pin(command.raw.as_bytes()),
            root_state_and_history_sha256: hex(&rules_snapshot_semantic_digest(
                &position.snapshot(),
            )
            .0),
            rules_profile_sha256: hex(&profile_digest().0),
            root_authority: false,
        })
    }
    #[derive(Serialize)]
    struct SessionImplementationMetadata {
        scope: &'static str,
        family: &'static str,
        root_policy: &'static str,
        implementation_sha256: String,
        compiled_bootstrap_source_sha256: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        input_lock_canonical_sha256: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        input_lock_json: Option<BytePin>,
        actual_session_constructed: bool,
        root_authority: bool,
    }
    fn prepare_session_implementation(
        document: &StartupConfigurationDocument,
        policy: SessionImplementationPolicy,
    ) -> Result<SessionImplementationMetadata, Error> {
        prepare_session_implementation_with_lock(document, policy, None)
    }
    fn prepare_session_implementation_with_lock(
        document: &StartupConfigurationDocument,
        policy: SessionImplementationPolicy,
        input_lock: Option<(&RoventInputLock, &[u8])>,
    ) -> Result<SessionImplementationMetadata, Error> {
        // The assembly comes from the original closed document. Neither an
        // arbitrary expected SHA nor a caller-selected incompatible family is
        // accepted, and these pure producers issue no live capability.
        let (family, digest) = match document.fields().execution {
            ExecutionConfiguration::RolesOnly { .. } => (
                "pc_native_session",
                RoventPcSessionDriver::implementation_digest_from_metadata(
                    document,
                    policy.actual(),
                ),
            ),
            ExecutionConfiguration::OwnResearcher {
                value: ResearchValueConfiguration::NnuecV2 { .. },
                ..
            } => (
                "own_nnuec_pcr_native_session",
                RoventNnuecSessionDriver::implementation_digest_from_metadata(
                    document,
                    policy.actual(),
                ),
            ),
            ExecutionConfiguration::ExternalResearcher { .. } => {
                let (lock, _) = input_lock.ok_or(Error::SessionImplementationLock(
                    "external session implementation metadata requires an explicit validated input lock",
                ))?;
                (
                    "external_researcher_product_session",
                    external_researcher_implementation_digest_from_metadata(
                        document,
                        lock,
                        policy.actual(),
                    ),
                )
            }
            _ => return Err(Error::SessionImplementationAssembly),
        };
        if input_lock.is_some()
            && !matches!(
                document.fields().execution,
                ExecutionConfiguration::ExternalResearcher { .. }
            )
        {
            return Err(Error::SessionImplementationLock(
                "input lock metadata is supported only for an external researcher session declaration",
            ));
        }
        let digest = digest.map_err(|cause| Error::SessionImplementation(Box::new(cause)))?;
        Ok(SessionImplementationMetadata {
            scope: if input_lock.is_some() {
                "compiled_source_configuration_lock_policy_digest_only"
            } else {
                "compiled_source_configuration_policy_digest_only"
            },
            family,
            root_policy: policy.wire_name(),
            implementation_sha256: hex(&digest.0),
            compiled_bootstrap_source_sha256: hex(
                &rz_uci::rovent_bootstrap::compiled_bootstrap_source_digest(),
            ),
            input_lock_canonical_sha256: input_lock.map(|(lock, _)| lock.canonical_sha256.clone()),
            input_lock_json: input_lock.map(|(_, raw)| byte_pin(raw)),
            actual_session_constructed: false,
            root_authority: false,
        })
    }
    #[derive(Serialize)]
    struct Metadata<'a> {
        schema: &'static str,
        scope: &'static str,
        configuration_document_domain: &'static str,
        source_configuration: BytePin,
        canonical_configuration: BytePin,
        canonical_configuration_json: &'a str,
        configuration_sha256: String,
        search_configuration_sha256: String,
        nnuec_conditions_sha256: Option<String>,
        backend_metadata: Option<BackendMetadata>,
        checks: Checks,
        actual_startup_ready: bool,
        native_authority: bool,
        training: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        root_metadata: Option<RootMetadata>,
        #[serde(skip_serializing_if = "Option::is_none")]
        session_implementation: Option<SessionImplementationMetadata>,
    }

    fn prepare_metadata(bytes: &[u8], backend_raw: Option<&[u8]>) -> Result<Vec<u8>, Error> {
        prepare_metadata_with_position(bytes, backend_raw, None)
    }
    fn prepare_metadata_with_position(
        bytes: &[u8],
        backend_raw: Option<&[u8]>,
        position: Option<&PositionCommand>,
    ) -> Result<Vec<u8>, Error> {
        prepare_metadata_with_options(bytes, backend_raw, position, None)
    }
    fn prepare_metadata_with_options(
        bytes: &[u8],
        backend_raw: Option<&[u8]>,
        position: Option<&PositionCommand>,
        session_policy: Option<SessionImplementationPolicy>,
    ) -> Result<Vec<u8>, Error> {
        prepare_metadata_with_options_and_input_lock(
            bytes,
            backend_raw,
            position,
            session_policy,
            None,
        )
    }
    fn prepare_metadata_with_options_and_input_lock(
        bytes: &[u8],
        backend_raw: Option<&[u8]>,
        position: Option<&PositionCommand>,
        session_policy: Option<SessionImplementationPolicy>,
        input_lock_raw: Option<&[u8]>,
    ) -> Result<Vec<u8>, Error> {
        check_input_budget(bytes)?;
        serde_json::from_slice::<UniqueJson>(bytes).map_err(Error::Json)?;
        let document =
            StartupConfigurationDocument::from_json(bytes).map_err(configuration_error)?;
        document.validate().map_err(configuration_error)?;
        let canonical = document.canonical_bytes().map_err(configuration_error)?;
        let backend_metadata = backend_raw
            .map(|raw| prepare_backend(&document, raw))
            .transpose()?;
        let backend_scope = if backend_metadata.is_some() {
            "declaration_digest_only_actual_identity_pending"
        } else {
            "pending_independent_runtime_metadata"
        };
        let root_metadata = position.map(prepare_root_metadata).transpose()?;
        let session_implementation = if let Some(raw) = input_lock_raw {
            let policy = session_policy.ok_or(Error::SessionImplementationLock(
                "input lock metadata requires an explicit session implementation root policy",
            ))?;
            if !matches!(
                document.fields().execution,
                ExecutionConfiguration::ExternalResearcher { .. }
            ) {
                return Err(Error::SessionImplementationLock(
                    "input lock metadata is supported only for an external researcher session declaration",
                ));
            }
            if raw.is_empty() || raw.len() > LOCK_INPUT_MAX_BYTES {
                return Err(Error::InputLockBudget);
            }
            serde_json::from_slice::<UniqueJson>(raw).map_err(Error::Json)?;
            let lock =
                RoventInputLock::from_json(std::str::from_utf8(raw).map_err(Error::CanonicalUtf8)?)
                    .map_err(|cause| Error::InputLock(Box::new(cause)))?;
            Some(prepare_session_implementation_with_lock(
                &document,
                policy,
                Some((&lock, raw)),
            )?)
        } else {
            session_policy
                .map(|policy| prepare_session_implementation(&document, policy))
                .transpose()?
        };
        let output = Metadata {
            schema: if session_implementation.is_some() {
                "rz-rovent-startup-metadata/3"
            } else if root_metadata.is_some() {
                "rz-rovent-startup-metadata/2"
            } else {
                "rz-rovent-startup-metadata/1"
            },
            scope: if session_implementation.is_some() {
                "startup_and_session_implementation_metadata_only"
            } else if root_metadata.is_some() {
                "startup_and_rules_metadata_only"
            } else {
                "startup_configuration_declarations_only"
            },
            configuration_document_domain: match &document {
                StartupConfigurationDocument::V1(_) => ROVENT_STARTUP_CONFIGURATION_DOMAIN,
                StartupConfigurationDocument::V2(_) => ROVENT_STARTUP_CONFIGURATION_V2_DOMAIN,
                StartupConfigurationDocument::V3(_) => ROVENT_STARTUP_CONFIGURATION_V3_DOMAIN,
                StartupConfigurationDocument::V4(_) => ROVENT_STARTUP_CONFIGURATION_V4_DOMAIN,
            },
            source_configuration: byte_pin(bytes),
            canonical_configuration: byte_pin(&canonical),
            canonical_configuration_json: std::str::from_utf8(&canonical)
                .map_err(Error::CanonicalUtf8)?,
            configuration_sha256: hex(&document
                .configuration_digest()
                .map_err(configuration_error)?),
            search_configuration_sha256: hex(&document
                .search_configuration_digest()
                .map_err(configuration_error)?),
            nnuec_conditions_sha256: document
                .nnuec_conditions_digest()
                .map_err(configuration_error)?
                .map(|d| hex(&d)),
            backend_metadata,
            checks: Checks {
                unique_json_object_keys: "passed",
                existing_closed_codec_and_declaration_constraints: "passed",
                actual_lock_join: if input_lock_raw.is_some() {
                    "declared_configuration_and_input_lock_only_no_runtime_admission"
                } else {
                    "pending_external_inputs"
                },
                compiled_native_capability_admission: "not_performed",
                referenced_asset_identity: "not_performed",
                backend_runtime_configuration_digest: backend_scope,
                native_loading_provider_execution: "not_performed",
                source_file_snapshot_stability: "not_claimed",
            },
            actual_startup_ready: false,
            native_authority: false,
            training: false,
            root_metadata,
            session_implementation,
        };
        let mut encoded = serde_json::to_vec(&output).map_err(Error::OutputEncoding)?;
        encoded.push(b'\n');
        enforce_stdout_budget(encoded)
    }
    fn enforce_stdout_budget(bytes: Vec<u8>) -> Result<Vec<u8>, Error> {
        if bytes.len() > STDOUT_MAX_BYTES {
            Err(Error::OutputBudget)
        } else {
            Ok(bytes)
        }
    }

    #[derive(Serialize)]
    struct Diagnostic<'a> {
        schema: &'static str,
        code: &'static str,
        detail_prefix: &'a str,
        detail_truncated: bool,
        metadata_stdout_complete: bool,
        actual_startup_ready: bool,
        native_authority: bool,
        training: bool,
    }
    fn diagnostic(error: &Error) -> Vec<u8> {
        // Causes remain owned by `error`; only the displayed prefix is bounded.
        let detail = error.to_string();
        let mut end = detail.len().min(ERROR_DETAIL_MAX_BYTES);
        while !detail.is_char_boundary(end) {
            end -= 1;
        }
        let value = Diagnostic {
            schema: "rz-rovent-startup-metadata-error/1",
            code: error.code(),
            detail_prefix: &detail[..end],
            detail_truncated: end != detail.len(),
            metadata_stdout_complete: false,
            actual_startup_ready: false,
            native_authority: false,
            training: false,
        };
        let fallback = b"{\"schema\":\"rz-rovent-startup-metadata-error/1\",\"code\":\"bounded_diagnostic_encoding_failed\",\"actual_startup_ready\":false,\"native_authority\":false,\"training\":false}\n";
        match serde_json::to_vec(&value) {
            Ok(mut bytes) if bytes.len() < DIAGNOSTIC_MAX_BYTES => {
                bytes.push(b'\n');
                bytes
            }
            _ => fallback.to_vec(),
        }
    }
    fn run() -> Result<(), Error> {
        let output = match parse_arguments(std::env::args_os().skip(1))? {
            Invocation::Startup(arguments) => {
                let input = read_configuration(&arguments.configuration)?;
                let backend = arguments
                    .backend_metadata
                    .as_deref()
                    .map(read_configuration)
                    .transpose()?;
                let input_lock = arguments
                    .input_lock_metadata
                    .as_deref()
                    .map(read_input_lock)
                    .transpose()?;
                match arguments.session_implementation_root_policy {
                    Some(policy) => prepare_metadata_with_options_and_input_lock(
                        &input,
                        backend.as_deref(),
                        arguments.position_command.as_ref(),
                        Some(policy),
                        input_lock.as_deref(),
                    )?,
                    None => match arguments.position_command.as_ref() {
                        Some(position) => prepare_metadata_with_position(
                            &input,
                            backend.as_deref(),
                            Some(position),
                        )?,
                        None => prepare_metadata(&input, backend.as_deref())?,
                    },
                }
            }
            Invocation::RuntimeBundleMetadata(path) => {
                prepare_runtime_bundle_metadata(&read_configuration(&path)?)?
            }
        };
        let mut stdout = io::stdout().lock();
        stdout
            .write_all(&output)
            .map_err(|e| io_error("stdout_write", e))?;
        stdout.flush().map_err(|e| io_error("stdout_flush", e))
    }
    pub(super) fn main() -> ExitCode {
        match run() {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                let _ = io::stderr().lock().write_all(&diagnostic(&error));
                ExitCode::from(2)
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        // Explicit synthetic constructor declarations, not a product asset/lock.
        fn fixture(domain: &str, version_fields: &str) -> Vec<u8> {
            r#"{
              "domain":"__DOMAIN__",
              "semantic":{"beam_width":4,"line_plies":4,"max_nodes":4096,"max_records":64,"max_role_calls":4096,"maximum_root_prefix_plies":256},
              "review":{"anchor_mode":"anchored_compact","token_budget":512,"max_records":64,"max_anchors":4},
              "index":{"active_bundles":256,"records_per_bundle":64},
              "frontier":{"maximum_goal_plies":256,"maximum_frames":64,"maximum_paths":4096,"maximum_mate_nodes":4096,"maximum_native_receipts":64},
              "native":{"cache_public_memory":true,"device_public_memory":false,"cudnn_conv_search":"legacy_implicit_exhaustive","loading_profile":null,"cuda_arena_bytes":null,"drain_ms":30000,"prepared_host_bytes_per_ticket":1048576,"output_host_bytes_per_ticket":1048576,"command_capacity":8},
              "execution":{"assembly":"roles_only","event_order":"fixed","role_tickets_max":16,"shared_nnuec_readout":null}__VERSION_FIELDS__
            }"#.replace("__DOMAIN__", domain).replace("__VERSION_FIELDS__", version_fields).into_bytes()
        }
        fn own_nnuec_fixture() -> Vec<u8> {
            let mut value: serde_json::Value =
                serde_json::from_slice(&fixture(ROVENT_STARTUP_CONFIGURATION_DOMAIN, "")).unwrap();
            // Fixed uses the same controller validation; these two declarations
            // match ControllerConfig::default().
            value["execution"] = serde_json::json!({
                "assembly":"own_researcher", "event_order":"fixed",
                "queues":{"waiting_jobs_max":8,"worker_commands_max":8,
                    "role_tickets_max":16,"paused_bytes_per_worker_max":1048576},
                "max_rounds":16,"max_native_nodes":4096,"cpu_depth":2,
                "cpu_nodes_per_task":1024,
                "cpu_plan":{"profile":"independent","tt_entries":4096,
                    "max_depth":8,"quiescence_ply":4,"ordering":"legal_see_v1",
                    "paused_bytes_limit":1048576,"preserve_paused_side_scope":true,
                    "affinity":null},
                "value":{"value":"nnuec_v2","context_namespace_policy":"fixture-context",
                    "local_namespace_policy":"fixture-local"},
                "controller":{"fixed_node_credit":64,"fixed_soft_lease_us":1000,
                    "initial_workers":1,"max_workers":1,"consecutive_observations":3,
                    "worker_cooldown_ms":2000},
                "reciprocal":{"recipient":"critic","max_questions_per_job":1,
                    "node_interval":1,"first_question_sequence":1},
                "peers":{"pc_peer":true,"researcher":true,"max_sources_per_root":1}
            });
            serde_json::to_vec(&value).unwrap()
        }

        fn external_fixture() -> Vec<u8> {
            let mut value: serde_json::Value = serde_json::from_slice(&fixture(
                ROVENT_STARTUP_CONFIGURATION_V3_DOMAIN,
                r#", "post_repair":"disabled", "external":{"kind":"absent"}"#,
            ))
            .unwrap();
            value["execution"] = serde_json::json!({
                "assembly":"external_researcher", "event_order":"fixed",
                "queues":{"waiting_jobs_max":16,"worker_commands_max":8,
                    "role_tickets_max":16,"paused_bytes_per_worker_max":1048576},
                "max_rounds":64,"max_reported_nodes":64,"cpu_depth":8,"cpu_nodes_per_task":128,
                "profile":{"path":"fixtures/external-profile.json","sha256":"a".repeat(64),
                    "bytes":32,"source":"https://example.invalid/profile","license":"fixture"},
                "slice":{"quantum_ms":1000,"stop_drain_reserve_ms":500,
                    "requested_reported_nodes":256,"multipv":1},
                "peers":{"pc_peer":true,"researcher":true,"max_sources_per_root":16}
            });
            value["external"] = serde_json::json!({"kind":"researcher",
                "question_limit":{"kind":"nodes","reported_nodes":32}});
            serde_json::to_vec(&value).unwrap()
        }

        #[test]
        fn external_lock_arguments_are_explicit_and_cannot_mix_standalone_queries() {
            let Invocation::Startup(arguments) = parse_arguments(
                [
                    "--config=config.json",
                    "--input-lock-metadata=lock.json",
                    "--session-implementation-root-policy=dynamic-go",
                ]
                .map(OsString::from),
            )
            .unwrap() else {
                panic!("startup invocation required")
            };
            assert_eq!(
                arguments.input_lock_metadata,
                Some(PathBuf::from("lock.json"))
            );
            for args in [
                vec!["--config=config.json", "--input-lock-metadata=lock.json"],
                vec![
                    "--config=config.json",
                    "--session-implementation-root-policy=dynamic-go",
                    "--input-lock-metadata=lock.json",
                    "--input-lock-metadata=lock.json",
                ],
                vec![
                    "--config=config.json",
                    "--session-implementation-root-policy=dynamic-go",
                    "--input-lock-metadata=.env",
                ],
                vec![
                    "--runtime-bundle-metadata=bundle.json",
                    "--input-lock-metadata=lock.json",
                ],
            ] {
                assert!(parse_arguments(args.into_iter().map(OsString::from)).is_err());
            }
        }

        #[test]
        fn external_lock_decode_refuses_missing_invalid_duplicate_and_over_budget_inputs() {
            let external = external_fixture();
            assert!(prepare_metadata(&external, None).is_ok()); // Original declaration-only path.
            let policy = Some(SessionImplementationPolicy::DynamicGo);
            assert!(matches!(
                prepare_metadata_with_options(&external, None, None, policy),
                Err(Error::SessionImplementationLock(_))
            ));
            assert!(matches!(
                prepare_metadata_with_options_and_input_lock(
                    &external,
                    None,
                    None,
                    None,
                    Some(b"{}")
                ),
                Err(Error::SessionImplementationLock(_))
            ));
            assert!(matches!(
                prepare_metadata_with_options_and_input_lock(
                    &external,
                    None,
                    None,
                    policy,
                    Some(b"{}")
                ),
                Err(Error::InputLock(_))
            ));
            assert!(matches!(
                prepare_metadata_with_options_and_input_lock(
                    &external,
                    None,
                    None,
                    policy,
                    Some(br#"{"domain":null,"domain":null}"#)
                ),
                Err(Error::Json(_))
            ));
            let excessive = vec![b' '; LOCK_INPUT_MAX_BYTES + 1];
            assert!(matches!(
                prepare_metadata_with_options_and_input_lock(
                    &external,
                    None,
                    None,
                    policy,
                    Some(&excessive)
                ),
                Err(Error::InputLockBudget)
            ));
            let pc = fixture(ROVENT_STARTUP_CONFIGURATION_DOMAIN, "");
            assert!(matches!(
                prepare_metadata_with_options_and_input_lock(&pc, None, None, policy, Some(b"{}")),
                Err(Error::SessionImplementationLock(_))
            ));
        }

        #[test]
        fn session_policy_arguments_are_closed_and_cannot_supply_expected_identity() {
            for (literal, policy) in [
                ("dynamic-go", SessionImplementationPolicy::DynamicGo),
                (
                    "exact-registered",
                    SessionImplementationPolicy::ExactRegistered,
                ),
            ] {
                let flag = format!("--session-implementation-root-policy={literal}");
                let Invocation::Startup(arguments) = parse_arguments([
                    OsString::from("--config=a.json"),
                    OsString::from(flag.as_str()),
                ])
                .unwrap() else {
                    panic!("startup selection required")
                };
                assert_eq!(arguments.session_implementation_root_policy, Some(policy));
                assert!(
                    parse_arguments([
                        OsString::from("--config=a.json"),
                        OsString::from(flag.as_str()),
                        OsString::from(flag.as_str()),
                    ])
                    .is_err()
                );
                assert!(
                    parse_arguments([
                        OsString::from("--runtime-bundle-metadata=a.json"),
                        OsString::from(flag),
                    ])
                    .is_err()
                );
            }
            for flag in [
                "--session-implementation-root-policy=",
                "--session-implementation-root-policy=true",
                "--session-implementation-root-policy=dynamic_go",
                "--session-implementation-root-policy=Dynamic-Go",
                "--session-implementation-sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            ] {
                assert!(parse_arguments(["--config=a.json", flag].map(OsString::from),).is_err());
            }
            assert!(
                parse_arguments([OsString::from(
                    "--session-implementation-root-policy=dynamic-go"
                ),])
                .is_err()
            );
        }

        #[test]
        fn explicit_session_metadata_preserves_legacy_shape_and_has_no_live_authority() {
            let position = parse_position_command("position startpos").unwrap();
            for (domain, fields) in [
                (ROVENT_STARTUP_CONFIGURATION_DOMAIN, ""),
                (
                    ROVENT_STARTUP_CONFIGURATION_V2_DOMAIN,
                    r#", "post_repair":"disabled""#,
                ),
                (
                    ROVENT_STARTUP_CONFIGURATION_V3_DOMAIN,
                    r#", "post_repair":"disabled", "external":{"kind":"absent"}"#,
                ),
            ] {
                let raw = fixture(domain, fields);
                let document = StartupConfigurationDocument::from_json(&raw).unwrap();
                let legacy = prepare_metadata(&raw, None).unwrap();
                assert_eq!(
                    serde_json::from_slice::<serde_json::Value>(&legacy).unwrap()["schema"],
                    "rz-rovent-startup-metadata/1"
                );
                assert!(
                    !std::str::from_utf8(&legacy)
                        .unwrap()
                        .contains("session_implementation")
                );
                let old_position =
                    prepare_metadata_with_position(&raw, None, Some(&position)).unwrap();
                assert_eq!(
                    serde_json::from_slice::<serde_json::Value>(&old_position).unwrap()["schema"],
                    "rz-rovent-startup-metadata/2"
                );
                assert!(
                    !std::str::from_utf8(&old_position)
                        .unwrap()
                        .contains("session_implementation")
                );
                for selected_position in [None, Some(&position)] {
                    let raw_output = prepare_metadata_with_options(
                        &raw,
                        None,
                        selected_position,
                        Some(SessionImplementationPolicy::DynamicGo),
                    )
                    .unwrap();
                    let output: serde_json::Value = serde_json::from_slice(&raw_output).unwrap();
                    let session = &output["session_implementation"];
                    assert_eq!(output["schema"], "rz-rovent-startup-metadata/3");
                    assert_eq!(session["family"], "pc_native_session");
                    assert_eq!(session["root_policy"], "dynamic_go");
                    assert_eq!(
                        session["implementation_sha256"],
                        hex(&RoventPcSessionDriver::implementation_digest_from_metadata(
                            &document,
                            RoventRootAdmissionPolicy::DynamicGo,
                        )
                        .unwrap()
                        .0,)
                    );
                    assert_eq!(
                        session["scope"],
                        "compiled_source_configuration_policy_digest_only"
                    );
                    assert_eq!(session["actual_session_constructed"], false);
                    assert_eq!(session["root_authority"], false);
                    for key in ["actual_startup_ready", "native_authority", "training"] {
                        assert_eq!(output[key], false);
                    }
                    assert_eq!(
                        output.get("root_metadata").is_some(),
                        selected_position.is_some()
                    );
                    assert!(raw_output.len() <= STDOUT_MAX_BYTES);
                }
            }
        }

        #[test]
        fn session_metadata_rejects_cross_family_and_unsupported_researcher_assembly() {
            let pc_raw = fixture(ROVENT_STARTUP_CONFIGURATION_DOMAIN, "");
            let pc = StartupConfigurationDocument::from_json(&pc_raw).unwrap();
            let nnuec_raw = own_nnuec_fixture();
            let nnuec = StartupConfigurationDocument::from_json(&nnuec_raw).unwrap();
            let policy = RoventRootAdmissionPolicy::DynamicGo;
            assert!(
                RoventPcSessionDriver::implementation_digest_from_metadata(&nnuec, policy).is_err()
            );
            assert!(
                RoventNnuecSessionDriver::implementation_digest_from_metadata(&pc, policy).is_err()
            );
            let own =
                prepare_session_implementation(&nnuec, SessionImplementationPolicy::DynamicGo)
                    .unwrap();
            assert_eq!(own.family, "own_nnuec_pcr_native_session");
            assert_eq!(
                own.implementation_sha256,
                hex(
                    &RoventNnuecSessionDriver::implementation_digest_from_metadata(&nnuec, policy)
                        .unwrap()
                        .0
                )
            );
            let mut baseline: serde_json::Value = serde_json::from_slice(&nnuec_raw).unwrap();
            baseline["execution"]["value"] = serde_json::json!({"value":"baseline_board"});
            let baseline = serde_json::to_vec(&baseline).unwrap();
            assert!(prepare_metadata(&baseline, None).is_ok());
            assert!(matches!(
                prepare_metadata_with_options(
                    &baseline,
                    None,
                    None,
                    Some(SessionImplementationPolicy::DynamicGo),
                ),
                Err(Error::SessionImplementationAssembly)
            ));
            let mut shared: serde_json::Value = serde_json::from_slice(&pc_raw).unwrap();
            shared["execution"]["shared_nnuec_readout"] = serde_json::json!({
                "context_namespace_policy":"fixture-context",
                "local_namespace_policy":"fixture-local",
            });
            let shared = serde_json::to_vec(&shared).unwrap();
            let shared = StartupConfigurationDocument::from_json(&shared).unwrap();
            let metadata =
                prepare_session_implementation(&shared, SessionImplementationPolicy::DynamicGo)
                    .unwrap();
            assert_eq!(metadata.family, "pc_native_session");
            assert_ne!(
                metadata.implementation_sha256,
                prepare_session_implementation(&pc, SessionImplementationPolicy::DynamicGo)
                    .unwrap()
                    .implementation_sha256
            );
            let mut invalid = pc;
            let StartupConfigurationDocument::V1(configuration) = &mut invalid else {
                panic!("original V1 fixture required")
            };
            configuration.semantic.line_plies = 0;
            assert!(
                RoventPcSessionDriver::implementation_digest_from_metadata(&invalid, policy)
                    .is_err()
            );
        }

        #[test]
        fn exact_arguments_reject_missing_extra_unknown_and_non_json() {
            assert!(parse_arguments(Vec::<OsString>::new()).is_err());
            for args in [
                vec!["--help"],
                vec!["--config="],
                vec!["--config=.env"],
                vec!["--config=a.json", "--other=1"],
                vec!["--config=a.json", "--config=b.json"],
                vec!["--backend-metadata=a.json"],
                vec!["--runtime-bundle-metadata="],
                vec!["--runtime-bundle-metadata=.env"],
                vec!["--runtime-bundle-metadata=a.json", "--other=1"],
                vec![
                    "--runtime-bundle-metadata=a.json",
                    "--runtime-bundle-metadata=b.json",
                ],
                vec!["--runtime-bundle-metadata=a.json", "--config=b.json"],
                vec!["--config=b.json", "--runtime-bundle-metadata=a.json"],
                vec![
                    "--runtime-bundle-metadata=a.json",
                    "--backend-metadata=b.json",
                ],
                vec![
                    "--runtime-bundle-metadata=a.json",
                    "--position-command=position startpos",
                ],
                vec![
                    "--config=a.json",
                    "--backend-metadata=a.json",
                    "--backend-metadata=b.json",
                ],
            ] {
                assert!(parse_arguments(args.into_iter().map(OsString::from)).is_err());
            }
            assert_eq!(
                parse_arguments([OsString::from("--config=선언.json")]).unwrap(),
                Invocation::Startup(Arguments {
                    configuration: PathBuf::from("선언.json"),
                    backend_metadata: None,
                    position_command: None,
                    session_implementation_root_policy: None,
                    input_lock_metadata: None,
                })
            );
            assert_eq!(
                parse_arguments(
                    ["--backend-metadata=b.json", "--config=a.json"].map(OsString::from)
                )
                .unwrap(),
                Invocation::Startup(Arguments {
                    configuration: PathBuf::from("a.json"),
                    backend_metadata: Some(PathBuf::from("b.json")),
                    position_command: None,
                    session_implementation_root_policy: None,
                    input_lock_metadata: None,
                })
            );
            assert_eq!(
                parse_arguments([OsString::from("--runtime-bundle-metadata=선언.json")]).unwrap(),
                Invocation::RuntimeBundleMetadata(PathBuf::from("선언.json"))
            );
        }

        fn runtime_bundle_fixture(version: u32) -> serde_json::Value {
            // Same closed synthetic roster as runtime_assets::tests::Fixture::cuda.
            // Metadata only: no library file is created or opened by these tests.
            let names = [
                ("libonnxruntime.so.1.22.0", "core"),
                ("libonnxruntime_providers_shared.so", "providers_shared"),
                ("libonnxruntime_providers_cuda.so", "providers_cuda"),
                ("libcudart.so.12", "nvidia_dependency"),
                ("libcublas.so.12", "nvidia_dependency"),
                ("libcublasLt.so.12", "nvidia_dependency"),
                ("libcufft.so.11", "nvidia_dependency"),
                ("libcurand.so.10", "nvidia_dependency"),
                ("libnvrtc.so.12", "nvidia_dependency"),
                ("libnvrtc-builtins.so.12.8", "nvidia_dependency"),
                ("libnvJitLink.so.12", "nvidia_dependency"),
                ("libcudnn.so.9", "nvidia_dependency"),
                ("libcudnn_adv.so.9", "nvidia_dependency"),
                ("libcudnn_cnn.so.9", "nvidia_dependency"),
                ("libcudnn_ops.so.9", "nvidia_dependency"),
                ("libcudnn_engines_precompiled.so.9", "nvidia_dependency"),
                (
                    "libcudnn_engines_runtime_compiled.so.9",
                    "nvidia_dependency",
                ),
                ("libcudnn_graph.so.9", "nvidia_dependency"),
                ("libcudnn_heuristic.so.9", "nvidia_dependency"),
            ];
            let files = names
                .into_iter()
                .map(|(filename, role)| {
                    serde_json::json!({"filename": filename, "role": role, "bytes": 7, "sha256": "11".repeat(32)})
                })
                .collect::<Vec<_>>();
            let mut value = serde_json::json!({"schema_version": version, "files": files});
            if version == 2 {
                value["loader_aliases"] = serde_json::json!([
                    {"filename":"libcudnn.so.9.8.0", "target":"libcudnn.so.9"},
                    {"filename":"libcudnn_cnn.so.9.8.0", "target":"libcudnn_cnn.so.9"}
                ]);
            }
            value
        }

        #[test]
        fn runtime_bundle_declaration_keeps_raw_and_typed_semantic_pins_separate() {
            for version in [1, 2] {
                let value = runtime_bundle_fixture(version);
                let raw = serde_json::to_vec(&value).unwrap();
                let output = prepare_runtime_bundle_metadata(&raw).unwrap();
                let observed: serde_json::Value = serde_json::from_slice(&output).unwrap();
                let typed =
                    CudaRuntimeBundleSpec::from_json(std::str::from_utf8(&raw).unwrap()).unwrap();
                let semantic = hex(&typed.digest().unwrap());
                let original = byte_pin(&raw);
                assert_eq!(observed["domain"], "rz-rovent-runtime-bundle-declaration/1");
                assert_eq!(
                    observed["raw_json_sha256"].as_str(),
                    Some(original.sha256.as_str())
                );
                assert_eq!(observed["bytes"].as_u64(), Some(raw.len() as u64));
                assert_eq!(
                    observed["semantic_bundle_sha256"].as_str(),
                    Some(semantic.as_str())
                );
                assert_ne!(original.sha256, semantic);
                assert_eq!(
                    observed["schema_version"].as_u64(),
                    Some(u64::from(version))
                );
                assert_eq!(observed["declared_only"], true);
                for field in ["runtime_loading_executed", "native_authority", "training"] {
                    assert_eq!(observed[field], false);
                }
                assert_eq!(observed["sessions_created"], 0);
                assert_eq!(observed["neural_forwards"], 0);
                assert!(output.len() <= STDOUT_MAX_BYTES);

                let mut reordered = value;
                reordered["files"].as_array_mut().unwrap().reverse();
                if let Some(aliases) = reordered.get_mut("loader_aliases") {
                    aliases.as_array_mut().unwrap().reverse();
                }
                let reordered = serde_json::to_vec_pretty(&reordered).unwrap();
                let other: serde_json::Value =
                    serde_json::from_slice(&prepare_runtime_bundle_metadata(&reordered).unwrap())
                        .unwrap();
                assert_ne!(observed["raw_json_sha256"], other["raw_json_sha256"]);
                assert_eq!(
                    observed["semantic_bundle_sha256"],
                    other["semantic_bundle_sha256"]
                );
            }
        }

        #[test]
        fn runtime_bundle_metadata_rejects_duplicate_keys_bad_rosters_and_bounds() {
            let value = runtime_bundle_fixture(2);
            let raw = serde_json::to_string(&value).unwrap();
            let duplicate = raw.replacen(
                "\"schema_version\":2",
                "\"schema_version\":2,\"schema_version\":2",
                1,
            );
            assert_ne!(raw, duplicate);
            assert!(matches!(
                prepare_runtime_bundle_metadata(duplicate.as_bytes()),
                Err(Error::Json(_))
            ));
            let duplicate = raw.replacen(
                "\"filename\":",
                "\"filename\":\"duplicate\",\"filename\":",
                1,
            );
            assert!(matches!(
                prepare_runtime_bundle_metadata(duplicate.as_bytes()),
                Err(Error::Json(_))
            ));
            let mut invalid = value.clone();
            invalid["files"][0]["filename"] = serde_json::json!("undeclared.so");
            assert!(matches!(
                prepare_runtime_bundle_metadata(&serde_json::to_vec(&invalid).unwrap()),
                Err(Error::RuntimeBundleDeclaration(_))
            ));
            let mut invalid = value;
            invalid["unknown"] = serde_json::json!(true);
            assert!(matches!(
                prepare_runtime_bundle_metadata(&serde_json::to_vec(&invalid).unwrap()),
                Err(Error::RuntimeBundleDeclaration(_))
            ));
            assert!(prepare_runtime_bundle_metadata(format!("{raw} {{}}").as_bytes()).is_err());
            for rejected in [Vec::new(), vec![b' '; INPUT_MAX_BYTES + 1]] {
                assert!(matches!(
                    prepare_runtime_bundle_metadata(&rejected),
                    Err(Error::InputBudget)
                ));
            }
        }
        #[test]
        fn byte_bounds_and_aggregate_output_reservation_are_explicit() {
            assert!(matches!(
                prepare_metadata(&[], None),
                Err(Error::InputBudget)
            ));
            assert!(matches!(
                prepare_metadata(&vec![b' '; INPUT_MAX_BYTES + 1], None),
                Err(Error::InputBudget)
            ));
            assert!(enforce_stdout_budget(vec![0; STDOUT_MAX_BYTES]).is_ok());
            assert!(matches!(
                enforce_stdout_budget(vec![0; STDOUT_MAX_BYTES + 1]),
                Err(Error::OutputBudget)
            ));
            assert_eq!(STDOUT_MAX_BYTES + DIAGNOSTIC_MAX_BYTES, OUTPUT_MAX_BYTES);
        }
        #[test]
        fn duplicate_keys_include_nested_and_escaped_spellings() {
            for raw in [
                br#"{"a":1,"\u0061":2}"#.as_slice(),
                br#"{"outer":[{"x":1,"x":2}]}"#.as_slice(),
            ] {
                assert!(serde_json::from_slice::<UniqueJson>(raw).is_err());
            }
            assert!(
                serde_json::from_slice::<UniqueJson>(br#"{"one":{"x":1},"two":{"x":2}}"#).is_ok()
            );
        }
        #[test]
        fn existing_codec_rejects_unknown_fields_invalid_values_and_extra_json() {
            let raw = String::from_utf8(fixture(ROVENT_STARTUP_CONFIGURATION_DOMAIN, "")).unwrap();
            for rejected in [
                raw.replace("\"max_nodes\":4096", "\"max_nodes\":4096,\"unknown\":1"),
                raw.replace("\"line_plies\":4", "\"line_plies\":0"),
                format!("{raw} {{}}"),
            ] {
                assert!(prepare_metadata(rejected.as_bytes(), None).is_err());
            }
        }
        #[derive(Deserialize)]
        struct Observed {
            configuration_document_domain: String,
            canonical_configuration_json: String,
            configuration_sha256: String,
            search_configuration_sha256: String,
            nnuec_conditions_sha256: Option<String>,
            actual_startup_ready: bool,
            native_authority: bool,
            training: bool,
        }
        #[test]
        fn all_versions_use_existing_digests_without_runtime_authority() {
            for (domain, fields) in [
                (ROVENT_STARTUP_CONFIGURATION_DOMAIN, ""),
                (
                    ROVENT_STARTUP_CONFIGURATION_V2_DOMAIN,
                    r#", "post_repair":"disabled""#,
                ),
                (
                    ROVENT_STARTUP_CONFIGURATION_V3_DOMAIN,
                    r#", "post_repair":"disabled", "external":{"kind":"absent"}"#,
                ),
            ] {
                let raw = fixture(domain, fields);
                let original = StartupConfigurationDocument::from_json(&raw).unwrap();
                let output = prepare_metadata(&raw, None).unwrap();
                let observed: Observed = serde_json::from_slice(&output).unwrap();
                assert_eq!(observed.configuration_document_domain, domain);
                assert_eq!(
                    observed.canonical_configuration_json.as_bytes(),
                    original.canonical_bytes().unwrap()
                );
                assert_eq!(
                    observed.configuration_sha256,
                    hex(&original.configuration_digest().unwrap())
                );
                assert_eq!(
                    observed.search_configuration_sha256,
                    hex(&original.search_configuration_digest().unwrap())
                );
                assert_eq!(
                    observed.nnuec_conditions_sha256,
                    original.nnuec_conditions_digest().unwrap().map(|d| hex(&d))
                );
                assert!(
                    !observed.actual_startup_ready
                        && !observed.native_authority
                        && !observed.training
                );
                assert!(output.len() <= STDOUT_MAX_BYTES);
            }
        }
        fn backend_fixture() -> Vec<u8> {
            // A declaration-only synthetic digest, not an actual native asset.
            format!(r#"{{"domain":"rz-rovent-backend-metadata-input/1","device":{{"device":"cpu"}},"runtime_binary_sha256":"{}","runtime_bundle_sha256":{{"kind":"absent"}},"control_inventory_sha256":{{"kind":"absent"}},"loading_profile":{{"kind":"absent"}}}}"#, "11".repeat(32)).into_bytes()
        }
        #[test]
        fn optional_backend_digest_reuses_native_typed_declaration_codec() {
            let startup = fixture(ROVENT_STARTUP_CONFIGURATION_DOMAIN, "");
            let document = StartupConfigurationDocument::from_json(&startup).unwrap();
            let raw = backend_fixture();
            let output = prepare_backend(&document, &raw).unwrap();
            let native = document.fields().native;
            let existing_config = native.onnx(&RoventDevice::Cpu, false).unwrap();
            let existing_digest =
                configuration_digest_from_metadata(existing_config, None, [0x11; 32], None, None)
                    .unwrap();
            assert_eq!(output.backend_configuration_sha256, hex(&existing_digest));
            assert_eq!(output.source_metadata.sha256, byte_pin(&raw).sha256);
            assert_eq!(output.source_metadata.bytes, raw.len());
            assert_eq!(output.scope, "independently_supplied_declarations_only");
            assert!(!output.actual_runtime_identity_verified && !output.actual_provider_verified);
            assert!(prepare_metadata(&startup, Some(&raw)).is_ok());
        }
        #[test]
        fn position_metadata_observes_actual_rules_history_and_original_command_bytes() {
            let raw = "position startpos moves e2e4 e7e5 g1f3";
            let command = parse_position_command(raw).unwrap();
            let observed = prepare_root_metadata(&command).unwrap();
            let mut expected = Position::startpos();
            for mv in ["e2e4", "e7e5", "g1f3"] {
                expected
                    .make_move(BoardMove::from_uci(mv).unwrap())
                    .unwrap();
            }
            assert_eq!(observed.position_command.bytes, raw.len());
            assert_eq!(
                observed.position_command.sha256,
                byte_pin(raw.as_bytes()).sha256
            );
            assert_eq!(
                observed.root_state_and_history_sha256,
                hex(&rules_snapshot_semantic_digest(&expected.snapshot()).0)
            );
            assert_eq!(observed.rules_profile_sha256, hex(&profile_digest().0));
            assert_eq!(observed.scope, "rules_snapshot_semantic_observation_only");
            assert!(!observed.root_authority);
        }
        #[test]
        fn fen_observation_preserves_unknown_prefix_instead_of_fabricating_start_history() {
            let start = parse_position_command("position startpos").unwrap();
            let fen =
                parse_position_command(&format!("position fen {}", Position::startpos().to_fen()))
                    .unwrap();
            let start = prepare_root_metadata(&start).unwrap();
            let fen = prepare_root_metadata(&fen).unwrap();
            assert_eq!(start.rules_profile_sha256, fen.rules_profile_sha256);
            assert_ne!(
                start.root_state_and_history_sha256,
                fen.root_state_and_history_sha256
            );
            assert!(!fen.root_authority);
        }
        #[test]
        fn position_selection_rejects_other_commands_bounds_duplicate_args_and_illegal_moves() {
            for raw in [
                "uci",
                "go depth 1",
                "position startpos\nuci",
                "position startpos moves e2e4\0",
            ] {
                assert!(parse_position_command(raw).is_err());
            }
            let over_line = format!(
                "position startpos{}",
                " ".repeat(ParserLimits::default().max_line_bytes)
            );
            assert!(matches!(
                parse_position_command(&over_line),
                Err(Error::PositionSyntax(ParseError::LineTooLong))
            ));
            let over_moves = format!(
                "position startpos moves{}",
                " g1f3".repeat(ParserLimits::default().max_moves + 1)
            );
            assert!(parse_position_command(&over_moves).is_err());
            assert!(
                parse_arguments(
                    [
                        "--config=a.json",
                        "--position-command=position startpos",
                        "--position-command=position startpos",
                    ]
                    .map(OsString::from)
                )
                .is_err()
            );
            let illegal = parse_position_command("position startpos moves e2e5").unwrap();
            assert!(matches!(
                prepare_root_metadata(&illegal),
                Err(Error::PositionRules {
                    zero_based_ply: Some(0),
                    cause: PositionError::IllegalMove,
                    ..
                })
            ));
            let fen = parse_position_command("position fen 8/8/8/8/8/8/8/8 w - - 0 1").unwrap();
            assert!(matches!(
                prepare_root_metadata(&fen),
                Err(Error::PositionRules {
                    stage: "position_base_rejected",
                    ..
                })
            ));
        }
        #[derive(Deserialize)]
        struct RootObserved {
            scope: String,
            root_state_and_history_sha256: String,
            rules_profile_sha256: String,
            root_authority: bool,
        }
        #[derive(Deserialize)]
        struct PositionMetadataObserved {
            schema: String,
            root_metadata: RootObserved,
            actual_startup_ready: bool,
            native_authority: bool,
            training: bool,
        }
        #[test]
        fn only_explicit_position_selects_v2_and_legacy_has_no_new_nullable_field() {
            let startup = fixture(ROVENT_STARTUP_CONFIGURATION_DOMAIN, "");
            let legacy = prepare_metadata(&startup, None).unwrap();
            let legacy_text = std::str::from_utf8(&legacy).unwrap();
            assert!(legacy_text.contains("\"schema\":\"rz-rovent-startup-metadata/1\""));
            assert!(!legacy_text.contains("root_metadata"));
            let command = parse_position_command("position startpos").unwrap();
            let output = prepare_metadata_with_position(&startup, None, Some(&command)).unwrap();
            let observed: PositionMetadataObserved = serde_json::from_slice(&output).unwrap();
            assert_eq!(observed.schema, "rz-rovent-startup-metadata/2");
            assert_eq!(
                observed.root_metadata.scope,
                "rules_snapshot_semantic_observation_only"
            );
            assert_eq!(
                observed.root_metadata.root_state_and_history_sha256,
                hex(&rules_snapshot_semantic_digest(&Position::startpos().snapshot()).0)
            );
            assert_eq!(
                observed.root_metadata.rules_profile_sha256,
                hex(&profile_digest().0)
            );
            assert!(
                !observed.root_metadata.root_authority
                    && !observed.actual_startup_ready
                    && !observed.native_authority
                    && !observed.training
            );
            assert!(output.len() <= STDOUT_MAX_BYTES);
        }
        #[test]
        fn backend_requires_explicit_absence_and_matching_profile_and_device() {
            let startup = fixture(ROVENT_STARTUP_CONFIGURATION_DOMAIN, "");
            let document = StartupConfigurationDocument::from_json(&startup).unwrap();
            let raw = String::from_utf8(backend_fixture()).unwrap();
            for rejected in [
                raw.replace("\"runtime_bundle_sha256\":{\"kind\":\"absent\"},", ""),
                raw.replace("\"control_inventory_sha256\":{\"kind\":\"absent\"}", "\"control_inventory_sha256\":null"),
                raw.replace("\"loading_profile\":{\"kind\":\"absent\"}", "\"loading_profile\":{\"kind\":\"cu_dnn_shim_lazy_v1\"}"),
                raw.replace("\"device\":{\"device\":\"cpu\"}", "\"device\":{\"device\":\"cuda\",\"ordinal\":0,\"registered_device\":\"synthetic\"}"),
                raw.replace("\"kind\":\"absent\"", "\"kind\":\"absent\",\"extra\":1"),
                raw.replace("\"domain\":", "\"domain\":\"rz-rovent-backend-metadata-input/1\",\"domain\":"),
                raw.replace("rz-rovent-backend-metadata-input/1", "rz-rovent-backend-metadata-input/2"),
            ] {
                assert!(prepare_backend(&document, rejected.as_bytes()).is_err());
            }
            assert!(matches!(
                prepare_backend(&document, &vec![b' '; INPUT_MAX_BYTES + 1]),
                Err(Error::InputBudget)
            ));
            for rejected in [
                "0".repeat(64),
                "a".repeat(63),
                "A".repeat(64),
                "g".repeat(64),
            ] {
                assert!(declared_sha256(&rejected).is_err());
            }
        }
        #[test]
        fn diagnostics_are_bounded_and_cannot_promote_a_failed_output() {
            let error = Error::Io {
                stage: "stdout_flush",
                cause: io::Error::other("한".repeat(5000)),
            };
            let bytes = diagnostic(&error);
            assert!(bytes.len() <= DIAGNOSTIC_MAX_BYTES);
            let text = std::str::from_utf8(&bytes).unwrap();
            assert!(text.contains("\"detail_truncated\":true"));
            assert!(text.contains("\"metadata_stdout_complete\":false"));
            assert!(text.contains("\"native_authority\":false"));
        }
    }
}
