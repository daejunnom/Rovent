//! Explicit system-runtime registration, distinct from historical V1--V5.
//!
//! A declaration or a preflight identity does not imply a loaded provider,
//! successful inference, or physical completion. This domain neither opens
//! native libraries nor owns a process; those authorities remain with the
//! native loader and execution supervisor.
use crate::{ManifestError, decode_json};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const SYSTEM_RUNTIME_LOCK_V6_DOMAIN: &str = "rz-system-runtime-lock-v6/1";
pub const SYSTEM_RUNTIME_RECEIPT_V6_DOMAIN: &str = "rz-system-runtime-receipt-v6/1";

fn invalid(message: &str) -> ManifestError {
    ManifestError::Integrity(message.into())
}

fn digest_string(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn text(value: &str, max: usize) -> bool {
    !value.is_empty() && value.len() <= max && !value.chars().any(char::is_control)
}

fn explicit_optional<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SystemPlatformV6 {
    Linux,
    Windows,
}

fn absolute_path(platform: SystemPlatformV6, value: &str) -> bool {
    if !text(value, 4096) {
        return false;
    }
    let absolute = match platform {
        SystemPlatformV6::Linux => value.starts_with('/'),
        SystemPlatformV6::Windows => {
            let bytes = value.as_bytes();
            (bytes.len() >= 3
                && bytes[0].is_ascii_alphabetic()
                && bytes[1] == b':'
                && matches!(bytes[2], b'/' | b'\\'))
                || value.starts_with("\\\\")
        }
    };
    absolute
        && !value
            .split(['/', '\\'])
            .any(|part| part == "." || part == "..")
}

/// A requested host library and its independently captured registration.
/// A missing version is explicitly unknown, not an inferred ABI version.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SystemLibrarySpecV6 {
    pub role: String,
    pub requested_path: String,
    pub canonical_path: String,
    #[serde(deserialize_with = "explicit_optional")]
    pub version: Option<String>,
    pub sha256: String,
    pub bytes: u64,
    pub device: u64,
    pub inode: u64,
    pub required_mapping: bool,
}

/// System libraries belong to the environment and are never private input
/// snapshots. File selection and its captured identity are registered once.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SystemRuntimeSpecV6 {
    pub platform: SystemPlatformV6,
    pub libraries: Vec<SystemLibrarySpecV6>,
    /// Identity emitted by the native owner during registration; not a
    /// source-code constant and not evidence of a loaded provider.
    pub system_identity_sha256: String,
    pub cuda_device_id: i32,
    pub cuda_arena_bytes: u64,
    pub startup_ms: u64,
    pub drain_ms: u64,
    pub capture_max_libraries: u32,
    pub capture_max_file_bytes: u64,
    pub capture_max_total_bytes: u64,
    pub capture_timeout_ms: u64,
}

/// The absence of a CUDA bundle is never used to infer CPU or system CUDA.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(
    tag = "kind",
    content = "configuration",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum NativeRuntimeSourceV6 {
    Cpu,
    SystemCuda(SystemRuntimeSpecV6),
}

/// Child-observed identity and mapping are separate from the registration.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SystemLibraryObservationV6 {
    pub role: String,
    pub canonical_path: String,
    pub sha256: String,
    pub bytes: u64,
    pub device: u64,
    pub inode: u64,
    /// `None` means that process-mapping evidence could not be observed.
    /// It is neither an observed absence nor proof that the image was loaded.
    /// Existing JSON booleans remain `Some(true)`/`Some(false)`.
    #[serde(deserialize_with = "explicit_optional")]
    pub mapped: Option<bool>,
}

impl SystemRuntimeSpecV6 {
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.libraries.is_empty()
            || self.libraries.len() > 128
            || !digest_string(&self.system_identity_sha256)
            || self.cuda_device_id < 0
            || self.cuda_arena_bytes == 0
            || self.startup_ms == 0
            || self.drain_ms == 0
            || self.capture_max_libraries == 0
            || self.capture_max_libraries > 128
            || self.libraries.len() > self.capture_max_libraries as usize
            || self.capture_max_file_bytes == 0
            || self.capture_max_total_bytes == 0
            || self.capture_timeout_ms == 0
            || self.capture_timeout_ms > self.startup_ms
        {
            return Err(invalid("invalid system CUDA identity or resource limits"));
        }
        let mut roles = BTreeSet::new();
        let mut files = BTreeSet::new();
        let mut total_bytes = 0u64;
        for library in &self.libraries {
            if !text(&library.role, 128)
                || !library
                    .role
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
                || !roles.insert(library.role.as_str())
                || !absolute_path(self.platform, &library.requested_path)
                || !absolute_path(self.platform, &library.canonical_path)
                || library.version.as_ref().is_some_and(|v| !text(v, 128))
                || !digest_string(&library.sha256)
                || library.bytes == 0
                || library.bytes > self.capture_max_file_bytes
                || library.inode == 0
                || !files.insert((library.device, library.inode))
            {
                return Err(invalid("invalid or duplicate system library identity"));
            }
            total_bytes = total_bytes
                .checked_add(library.bytes)
                .ok_or_else(|| invalid("system library registration size overflow"))?;
        }
        if total_bytes > self.capture_max_total_bytes {
            return Err(invalid(
                "system libraries exceed the registered capture budget",
            ));
        }
        for role in ["ort-core", "ort-provider-cuda"] {
            if !self
                .libraries
                .iter()
                .any(|item| item.role == role && item.required_mapping)
            {
                return Err(invalid(
                    "system CUDA requires explicit ORT core and CUDA provider roles",
                ));
            }
        }
        Ok(())
    }

    /// Validate the captured identity. Libraries may be lazy and not yet
    /// mapped at this point; this does not imply provider readiness.
    pub fn validate_observed(
        &self,
        observed: &[SystemLibraryObservationV6],
    ) -> Result<(), ManifestError> {
        self.validate()?;
        let by_role: BTreeMap<_, _> = observed.iter().map(|v| (v.role.as_str(), v)).collect();
        if by_role.len() != observed.len() || observed.len() != self.libraries.len() {
            return Err(invalid(
                "system library observation roles are incomplete or duplicated",
            ));
        }
        for expected in &self.libraries {
            let actual = by_role
                .get(expected.role.as_str())
                .ok_or_else(|| invalid("system library observation role missing"))?;
            if actual.canonical_path != expected.canonical_path
                || actual.sha256 != expected.sha256
                || actual.bytes != expected.bytes
                || actual.device != expected.device
                || actual.inode != expected.inode
            {
                return Err(invalid(
                    "system library observation differs from registration",
                ));
            }
        }
        Ok(())
    }

    /// Called only after provider setup; optional lazy images may remain absent.
    pub fn validate_ready_observed(
        &self,
        observed: &[SystemLibraryObservationV6],
    ) -> Result<(), ManifestError> {
        self.validate_observed(observed)?;
        if observed.iter().any(|item| item.mapped.is_none()) {
            return Err(invalid("system library mapping observation is unknown"));
        }
        for item in &self.libraries {
            if item.required_mapping
                && !observed
                    .iter()
                    .any(|o| o.role == item.role && o.mapped == Some(true))
            {
                return Err(invalid("required system library was not observed mapped"));
            }
        }
        Ok(())
    }
}

impl NativeRuntimeSourceV6 {
    pub fn validate(&self) -> Result<(), ManifestError> {
        match self {
            Self::Cpu => Ok(()),
            Self::SystemCuda(spec) => spec.validate(),
        }
    }
}

fn canonical<T: Serialize>(value: &T) -> Result<Vec<u8>, ManifestError> {
    let mut value = serde_json::to_value(value).map_err(|error| invalid(&error.to_string()))?;
    value.sort_all_objects();
    serde_json::to_vec(&value).map_err(|error| invalid(&error.to_string()))
}

/// A new registration, not a converted V5 input lock.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SystemRuntimeLockV6 {
    pub domain: String,
    pub runtime: NativeRuntimeSourceV6,
    pub canonical_sha256: String,
}

impl SystemRuntimeLockV6 {
    pub fn new(runtime: NativeRuntimeSourceV6) -> Result<Self, ManifestError> {
        runtime.validate()?;
        let bytes = canonical(&(SYSTEM_RUNTIME_LOCK_V6_DOMAIN, &runtime))?;
        Ok(Self {
            domain: SYSTEM_RUNTIME_LOCK_V6_DOMAIN.into(),
            runtime,
            canonical_sha256: format!("{:x}", Sha256::digest(bytes)),
        })
    }
    pub fn verify(&self) -> Result<(), ManifestError> {
        if self.domain != SYSTEM_RUNTIME_LOCK_V6_DOMAIN
            || self.canonical_sha256 != Self::new(self.runtime.clone())?.canonical_sha256
        {
            return Err(invalid("system runtime lock identity differs"));
        }
        Ok(())
    }
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let value: Self = decode_json(input)?;
        value.verify()?;
        Ok(value)
    }
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        self.verify()?;
        canonical(self)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ProviderReadinessV6 {
    Unknown,
    Ready,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum PhysicalCompletionV6 {
    Unknown,
    Confirmed { completion_id: String },
    Failed { reason: String },
}

/// A child report is a claim until its owner/supervisor and native evidence
/// have been verified separately. Failure and partial reports remain valid.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SystemRuntimeReceiptV6 {
    pub domain: String,
    pub lock_sha256: String,
    pub process_id: u32,
    pub process_generation: u64,
    #[serde(deserialize_with = "explicit_optional")]
    pub observed_identity_sha256: Option<String>,
    pub observed_libraries: Vec<SystemLibraryObservationV6>,
    /// None means the mappings could not be observed; Some(empty) is an
    /// observed empty set, not a substitute for a failed observation.
    #[serde(deserialize_with = "explicit_optional")]
    pub unregistered_native_mappings: Option<Vec<String>>,
    pub provider_ready: ProviderReadinessV6,
    pub physical_completion: PhysicalCompletionV6,
}

impl SystemRuntimeReceiptV6 {
    pub fn from_json(
        input: &str,
        lock: &SystemRuntimeLockV6,
        process_id: u32,
        process_generation: u64,
    ) -> Result<Self, ManifestError> {
        let value: Self = decode_json(input)?;
        value.validate_against(lock, process_id, process_generation)?;
        Ok(value)
    }

    pub fn canonical_bytes(
        &self,
        lock: &SystemRuntimeLockV6,
        process_id: u32,
        process_generation: u64,
    ) -> Result<Vec<u8>, ManifestError> {
        self.validate_against(lock, process_id, process_generation)?;
        canonical(self)
    }

    /// Keep failure reports even if observed identity differs; they never
    /// satisfy validate_success. The supervisor supplies the expected owner.
    pub fn validate_against(
        &self,
        lock: &SystemRuntimeLockV6,
        process_id: u32,
        process_generation: u64,
    ) -> Result<(), ManifestError> {
        lock.verify()?;
        if self.domain != SYSTEM_RUNTIME_RECEIPT_V6_DOMAIN
            || self.lock_sha256 != lock.canonical_sha256
            || self.process_id == 0
            || self.process_id != process_id
            || self.process_generation == 0
            || self.process_generation != process_generation
            || self
                .observed_identity_sha256
                .as_ref()
                .is_some_and(|s| !digest_string(s))
            || self.observed_libraries.len() > 128
            || self
                .unregistered_native_mappings
                .as_ref()
                .is_some_and(|mappings| mappings.len() > 128)
        {
            return Err(invalid(
                "runtime receipt is not bound to the registered child",
            ));
        }
        match &self.physical_completion {
            PhysicalCompletionV6::Confirmed { completion_id } if !text(completion_id, 256) => {
                return Err(invalid("missing physical completion identity"));
            }
            PhysicalCompletionV6::Failed { reason } if !text(reason, 1024) => {
                return Err(invalid("missing physical completion failure"));
            }
            _ => {}
        }
        Ok(())
    }
    /// A structurally valid receipt is not itself proof that its observations
    /// came from the child; the supervisor must authenticate that separately.
    pub fn validate_success(
        &self,
        lock: &SystemRuntimeLockV6,
        process_id: u32,
        process_generation: u64,
    ) -> Result<(), ManifestError> {
        self.validate_against(lock, process_id, process_generation)?;
        if self.provider_ready != ProviderReadinessV6::Ready
            || !matches!(
                self.physical_completion,
                PhysicalCompletionV6::Confirmed { .. }
            )
        {
            return Err(invalid(
                "runtime readiness or physical completion is unconfirmed",
            ));
        }
        match &lock.runtime {
            NativeRuntimeSourceV6::Cpu => {
                if self.observed_identity_sha256.is_some() || !self.observed_libraries.is_empty() {
                    return Err(invalid(
                        "CPU receipt cannot assert system CUDA observations",
                    ));
                }
            }
            NativeRuntimeSourceV6::SystemCuda(spec) => {
                if self
                    .unregistered_native_mappings
                    .as_ref()
                    .is_none_or(|mappings| !mappings.is_empty())
                {
                    return Err(invalid("system CUDA native mappings are unconfirmed"));
                }
                if self.observed_identity_sha256.as_deref() != Some(&spec.system_identity_sha256) {
                    return Err(invalid(
                        "observed system identity differs from registration",
                    ));
                }
                spec.validate_ready_observed(&self.observed_libraries)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registered() -> SystemRuntimeSpecV6 {
        let libraries = ["ort-core", "ort-provider-cuda"]
            .into_iter()
            .enumerate()
            .map(|(index, role)| SystemLibrarySpecV6 {
                role: role.into(),
                requested_path: format!("/usr/lib/{role}.so"),
                canonical_path: format!("/usr/lib/{role}.so.1"),
                version: None,
                sha256: format!("{:064x}", index + 1),
                bytes: (index + 1) as u64,
                device: 5,
                inode: (index + 1) as u64,
                required_mapping: true,
            })
            .collect();
        SystemRuntimeSpecV6 {
            platform: SystemPlatformV6::Linux,
            libraries,
            system_identity_sha256: format!("{:064x}", 42),
            cuda_device_id: 0,
            cuda_arena_bytes: 1024,
            startup_ms: 1000,
            drain_ms: 200,
            capture_max_libraries: 2,
            capture_max_file_bytes: 2,
            capture_max_total_bytes: 3,
            capture_timeout_ms: 100,
        }
    }

    fn observations(spec: &SystemRuntimeSpecV6) -> Vec<SystemLibraryObservationV6> {
        spec.libraries
            .iter()
            .map(|library| SystemLibraryObservationV6 {
                role: library.role.clone(),
                canonical_path: library.canonical_path.clone(),
                sha256: library.sha256.clone(),
                bytes: library.bytes,
                device: library.device,
                inode: library.inode,
                mapped: Some(true),
            })
            .collect()
    }

    #[test]
    fn registration_keeps_cpu_and_system_cuda_distinct_and_locked() {
        let spec = registered();
        let lock = SystemRuntimeLockV6::new(NativeRuntimeSourceV6::SystemCuda(spec)).unwrap();
        let encoded = lock.canonical_bytes().unwrap();
        assert_eq!(
            lock,
            SystemRuntimeLockV6::from_json(std::str::from_utf8(&encoded).unwrap()).unwrap()
        );
        assert_ne!(
            lock.canonical_sha256,
            SystemRuntimeLockV6::new(NativeRuntimeSourceV6::Cpu)
                .unwrap()
                .canonical_sha256
        );
        let mut changed = lock;
        if let NativeRuntimeSourceV6::SystemCuda(spec) = &mut changed.runtime {
            spec.cuda_device_id += 1;
        }
        assert!(changed.verify().is_err());
    }

    #[test]
    fn observation_distinguishes_identity_and_provider_mapping() {
        let spec = registered();
        let mut actual = observations(&spec);
        actual[1].mapped = Some(false);
        assert!(spec.validate_observed(&actual).is_ok());
        assert!(spec.validate_ready_observed(&actual).is_err());
        actual[1].mapped = None;
        assert!(spec.validate_observed(&actual).is_ok());
        assert!(spec.validate_ready_observed(&actual).is_err());
        actual[1].mapped = Some(true);
        assert!(spec.validate_ready_observed(&actual).is_ok());
        actual[0].inode += 10;
        assert!(spec.validate_observed(&actual).is_err());
    }

    #[test]
    fn mapping_wire_keeps_legacy_boolean_and_requires_explicit_unknown() {
        let spec = registered();
        let known = observations(&spec).remove(0);
        let mut wire = serde_json::to_value(&known).unwrap();
        assert_eq!(wire["mapped"], serde_json::Value::Bool(true));
        assert_eq!(
            serde_json::from_value::<SystemLibraryObservationV6>(wire.clone())
                .unwrap()
                .mapped,
            Some(true)
        );
        wire["mapped"] = serde_json::Value::Null;
        assert_eq!(
            serde_json::from_value::<SystemLibraryObservationV6>(wire.clone())
                .unwrap()
                .mapped,
            None
        );
        wire.as_object_mut().unwrap().remove("mapped");
        assert!(serde_json::from_value::<SystemLibraryObservationV6>(wire).is_err());
    }

    #[test]
    fn incomplete_receipt_is_preserved_but_never_success() {
        let spec = registered();
        let lock =
            SystemRuntimeLockV6::new(NativeRuntimeSourceV6::SystemCuda(spec.clone())).unwrap();
        let mut receipt = SystemRuntimeReceiptV6 {
            domain: SYSTEM_RUNTIME_RECEIPT_V6_DOMAIN.into(),
            lock_sha256: lock.canonical_sha256.clone(),
            process_id: 7,
            process_generation: 1,
            observed_identity_sha256: Some(spec.system_identity_sha256.clone()),
            observed_libraries: observations(&spec),
            unregistered_native_mappings: Some(vec![]),
            provider_ready: ProviderReadinessV6::Ready,
            physical_completion: PhysicalCompletionV6::Unknown,
        };
        assert!(receipt.validate_against(&lock, 7, 1).is_ok());
        assert!(receipt.validate_success(&lock, 7, 1).is_err());
        let bytes = receipt.canonical_bytes(&lock, 7, 1).unwrap();
        assert_eq!(
            receipt,
            SystemRuntimeReceiptV6::from_json(std::str::from_utf8(&bytes).unwrap(), &lock, 7, 1)
                .unwrap()
        );
        let mut missing: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        missing
            .as_object_mut()
            .unwrap()
            .remove("unregistered_native_mappings");
        assert!(
            SystemRuntimeReceiptV6::from_json(
                &serde_json::to_string(&missing).unwrap(),
                &lock,
                7,
                1
            )
            .is_err()
        );
        receipt.physical_completion = PhysicalCompletionV6::Confirmed {
            completion_id: "fence".into(),
        };
        assert!(receipt.validate_success(&lock, 7, 1).is_ok());
        assert!(receipt.validate_success(&lock, 8, 1).is_err());
        receipt.unregistered_native_mappings = None;
        assert!(receipt.validate_against(&lock, 7, 1).is_ok());
        assert!(receipt.validate_success(&lock, 7, 1).is_err());
        receipt
            .unregistered_native_mappings
            .get_or_insert_default()
            .push("foreign.so".into());
        assert!(receipt.validate_success(&lock, 7, 1).is_err());
    }

    #[test]
    fn cpu_success_keeps_unobserved_cuda_mappings_unknown() {
        let lock = SystemRuntimeLockV6::new(NativeRuntimeSourceV6::Cpu).unwrap();
        let receipt = SystemRuntimeReceiptV6 {
            domain: SYSTEM_RUNTIME_RECEIPT_V6_DOMAIN.into(),
            lock_sha256: lock.canonical_sha256.clone(),
            process_id: 7,
            process_generation: 1,
            observed_identity_sha256: None,
            observed_libraries: vec![],
            unregistered_native_mappings: None,
            provider_ready: ProviderReadinessV6::Ready,
            physical_completion: PhysicalCompletionV6::Confirmed {
                completion_id: "cpu-worker-joined".into(),
            },
        };
        assert!(receipt.validate_success(&lock, 7, 1).is_ok());
        let bytes = receipt.canonical_bytes(&lock, 7, 1).unwrap();
        let decoded =
            SystemRuntimeReceiptV6::from_json(std::str::from_utf8(&bytes).unwrap(), &lock, 7, 1)
                .unwrap();
        assert_eq!(decoded.unregistered_native_mappings, None);
    }

    #[test]
    fn limits_and_system_paths_are_validated_without_fixed_file_values() {
        let mut spec = registered();
        assert!(spec.validate().is_ok());
        spec.libraries[0].sha256 = format!("{:064x}", 12345);
        assert!(spec.validate().is_ok());
        spec.capture_max_total_bytes = 2;
        assert!(spec.validate().is_err());
        spec.capture_max_total_bytes = 3;
        spec.libraries[0].canonical_path = "/usr/lib/../elsewhere.so".into();
        assert!(spec.validate().is_err());
    }
}
