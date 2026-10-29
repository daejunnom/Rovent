//! Actual worker orchestration for the additive PALS adaptive route.
//!
//! Rules/store semantics are supplied by the existing Search owner. This module
//! owns admission, credit and message order; encoding, neural calls, CPU search,
//! external pipe drain and model teardown stay on their private owners.

#[cfg(feature = "pals-adaptive-diagnostics")]
mod continuation_diagnostic;
#[cfg(feature = "pals-adaptive-diagnostics")]
pub use continuation_diagnostic::*;
mod cpu_slice_observation;
pub use cpu_slice_observation::*;
mod manager;
pub use manager::*;
mod driver;
pub use driver::*;
mod validator;
pub use validator::*;
mod utility_route;
pub use utility_route::*;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
mod utility_cohort;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub use utility_cohort::*;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
mod utility_cohort_historical;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub use utility_cohort_historical::*;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
mod utility_intervention;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub use utility_intervention::*;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
mod utility_intervention_historical;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub use utility_intervention_historical::*;
mod native_accounting;
mod session_driver;
pub use native_accounting::AdaptiveNativeAccountingSnapshot;
pub use session_driver::*;
#[cfg(feature = "onnx-cpu")]
mod projection;
#[cfg(feature = "onnx-cpu")]
pub use projection::*;

pub fn adaptive_execution_manager_source_digest() -> [u8; 32] {
    use sha2::{Digest as _, Sha256};
    let mut hash = Sha256::new();
    hash.update(b"rz-pals-adaptive-execution-manager/1\0");
    for (name, bytes) in [
        (
            "pals_adaptive.rs",
            include_bytes!("pals_adaptive.rs").as_slice(),
        ),
        (
            "pals_adaptive/manager.rs",
            include_bytes!("pals_adaptive/manager.rs").as_slice(),
        ),
        (
            "pals_adaptive/manager/reciprocal.rs",
            include_bytes!("pals_adaptive/manager/reciprocal.rs").as_slice(),
        ),
        (
            "pals_adaptive/continuation_diagnostic.rs",
            include_bytes!("pals_adaptive/continuation_diagnostic.rs").as_slice(),
        ),
        (
            "pals_adaptive/cpu_slice_observation.rs",
            include_bytes!("pals_adaptive/cpu_slice_observation.rs").as_slice(),
        ),
        (
            "pals_adaptive/cpu_slice_observation/nnuec_chain.rs",
            include_bytes!("pals_adaptive/cpu_slice_observation/nnuec_chain.rs").as_slice(),
        ),
        (
            "pals_adaptive/projection.rs",
            include_bytes!("pals_adaptive/projection.rs").as_slice(),
        ),
        (
            "pals_adaptive/validator.rs",
            include_bytes!("pals_adaptive/validator.rs").as_slice(),
        ),
        (
            "pals_adaptive/driver.rs",
            include_bytes!("pals_adaptive/driver.rs").as_slice(),
        ),
        (
            "pals_adaptive/session_driver.rs",
            include_bytes!("pals_adaptive/session_driver.rs").as_slice(),
        ),
        (
            "pals_adaptive/session_driver/collection_v2.rs",
            include_bytes!("pals_adaptive/session_driver/collection_v2.rs").as_slice(),
        ),
        (
            "pals_adaptive/native_accounting.rs",
            include_bytes!("pals_adaptive/native_accounting.rs").as_slice(),
        ),
    ] {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    hash.finalize().into()
}
