#![forbid(unsafe_code)]

mod artifact;
mod manifest;
mod manifest_v2;
mod match_resources;
mod native_cuda;
mod native_cuda_batch;
mod native_cuda_pilot;
mod native_cuda_v2;
mod native_launch;
mod pals_compiled_sources_v5;
mod pals_data;
pub mod pals_data_v3;
mod pals_manifest;
mod pals_manifest_v4;
mod pals_manifest_v5;
mod pals_source_pins_v5;
mod pals_startup_v5;
pub mod rovent_corpus;
pub mod rovent_corpus_v2;
mod rovent_manifest;
mod rovent_input_lock_owner;
pub mod rovent_r_readout;
mod rovent_root_allocation;
mod rovent_root_descriptor;
pub mod rovent_storage;
pub mod rovent_submitted_input;
pub mod rovent_utility_cohort;
pub mod rovent_utility_intervention_preparation;
mod validation;

pub use manifest::*;
pub use manifest_v2::*;
pub use match_resources::*;
pub use native_cuda::*;
pub use native_cuda_batch::*;
pub use native_cuda_pilot::*;
pub use native_cuda_v2::*;
pub use native_launch::*;
pub use pals_compiled_sources_v5::*;
pub use pals_data::*;
pub use pals_manifest::*;
pub use pals_manifest_v4::*;
pub use pals_manifest_v5::*;
pub use pals_source_pins_v5::*;
pub use pals_startup_v5::*;
pub use rovent_manifest::*;
pub use rovent_input_lock_owner::*;
pub use rovent_root_allocation::*;
pub use rovent_root_descriptor::*;
pub use rovent_storage::*;

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fmt;

pub const MAX_MANIFEST_BYTES: usize = 4 * 1024 * 1024;

/// Bounded, duplicate-key rejecting JSON decoding for E-owned persisted formats.
/// The target type must separately reject unknown fields and validate semantics.
pub fn decode_json<T: serde::de::DeserializeOwned>(input: &str) -> Result<T, ManifestError> {
    parse(input)
}

#[derive(Debug)]
pub enum ManifestError {
    Parse(String),
    Validation(Vec<Violation>),
    Integrity(String),
    Artifact(String),
}

#[derive(Debug, PartialEq, Eq)]
pub struct Violation {
    pub path: String,
    pub code: &'static str,
    pub message: String,
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(issues) => {
                for (i, issue) in issues.iter().enumerate() {
                    if i > 0 {
                        writeln!(f)?;
                    }
                    write!(f, "{} [{}]: {}", issue.path, issue.code, issue.message)?;
                }
                Ok(())
            }
            Self::Parse(reason) => write!(f, "manifest parse: {reason}"),
            Self::Integrity(reason) => write!(f, "input lock integrity: {reason}"),
            Self::Artifact(reason) => write!(f, "artifact verification: {reason}"),
        }
    }
}

impl std::error::Error for ManifestError {}

pub(crate) fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

// All objects, including arbitrary UCI option maps, reject duplicate keys.
// serde_json's normal Value parser would silently retain the last value.
struct UniqueValue(serde_json::Value);

impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = UniqueValue;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("JSON with unique object keys")
            }
            fn visit_bool<E>(self, v: bool) -> Result<Self::Value, E> {
                Ok(UniqueValue(v.into()))
            }
            fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
                Ok(UniqueValue(v.into()))
            }
            fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E> {
                Ok(UniqueValue(v.into()))
            }
            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(v)
                    .map(|n| UniqueValue(n.into()))
                    .ok_or_else(|| E::custom("non-finite JSON number"))
            }
            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
                Ok(UniqueValue(v.into()))
            }
            fn visit_string<E>(self, v: String) -> Result<Self::Value, E> {
                Ok(UniqueValue(v.into()))
            }
            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(UniqueValue(serde_json::Value::Null))
            }
            fn visit_none<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                self.visit_unit()
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut a: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(v) = a.next_element::<UniqueValue>()? {
                    values.push(v.0);
                }
                Ok(UniqueValue(values.into()))
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut a: A,
            ) -> Result<Self::Value, A::Error> {
                let mut map = serde_json::Map::new();
                while let Some(key) = a.next_key::<String>()? {
                    if map.contains_key(&key) {
                        return Err(serde::de::Error::custom("duplicate JSON object key"));
                    }
                    map.insert(key, a.next_value::<UniqueValue>()?.0);
                }
                Ok(UniqueValue(map.into()))
            }
        }
        d.deserialize_any(Visitor)
    }
}

pub(crate) fn parse<T: serde::de::DeserializeOwned>(input: &str) -> Result<T, ManifestError> {
    if input.len() > MAX_MANIFEST_BYTES {
        return Err(ManifestError::Parse("input exceeds 4 MiB limit".into()));
    }
    let unique: UniqueValue =
        serde_json::from_str(input).map_err(|e| ManifestError::Parse(e.to_string()))?;
    serde_json::from_value(unique.0).map_err(|e| ManifestError::Parse(e.to_string()))
}
