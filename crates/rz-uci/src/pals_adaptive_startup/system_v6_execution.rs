//! Narrow identity for the V6 child execution integration.
//!
//! This is separate from the historical V6 boundary digest. It does not
//! attest the parent executable, the entire workspace, or GPU completion.
use super::PalsSystemSourceObservationV6;
use sha2::{Digest as _, Sha256};
use std::collections::BTreeMap;

pub const PALS_SYSTEM_V6_EXECUTION_SOURCE_COMPONENT: &str = "pals_system_v6_execution";

pub fn compiled_pals_system_v6_execution_source_digest() -> String {
    let files: &[(&str, &[u8])] = &[
        ("execution_digest", include_bytes!("system_v6_execution.rs")),
        (
            "cpu_prepare",
            include_bytes!("../pals_adaptive_bootstrap/system_v6/cpu_prepare.rs"),
        ),
        (
            "driver_prepare",
            include_bytes!("../pals_adaptive_bootstrap/system_v6/driver_prepare.rs"),
        ),
        (
            "driver_assemble",
            include_bytes!("../pals_adaptive_bootstrap/system_v6/driver_assemble.rs"),
        ),
        (
            "driver_facade",
            include_bytes!("../pals_adaptive_bootstrap/system_v6/driver_facade.rs"),
        ),
        (
            "driver_lifecycle",
            include_bytes!("../pals_adaptive_bootstrap/system_v6/driver_lifecycle.rs"),
        ),
        (
            "pipeline",
            include_bytes!("../pals_adaptive_bootstrap/system_v6/pipeline.rs"),
        ),
        (
            "failed_startup_drain",
            include_bytes!("../pals_adaptive_bootstrap/system_v6/failed_startup_drain.rs"),
        ),
        (
            "before_native_recovery",
            include_bytes!("../pals_adaptive_bootstrap/system_v6/before_native_recovery.rs"),
        ),
        (
            "admission",
            include_bytes!("../pals_adaptive/session_driver/admission_system_v6.rs"),
        ),
        (
            "idle_close",
            include_bytes!("../pals_adaptive/session_driver/idle_system_v6_close.rs"),
        ),
        (
            "external_admission",
            include_bytes!("../pals_adaptive_bootstrap/external/system_v6.rs"),
        ),
        (
            "parent_transport",
            include_bytes!("../rovent_bootstrap/launch/system_parent.rs"),
        ),
        (
            "release_transport",
            include_bytes!("../pals_system_proxy/finish.rs"),
        ),
        (
            "proxy_owner_source",
            include_bytes!("../pals_system_proxy.rs"),
        ),
        (
            "proxy_parent_source",
            include_bytes!("../pals_system_proxy/parent.rs"),
        ),
        (
            "proxy_runner_source",
            include_bytes!("../pals_system_proxy/runner_v6.rs"),
        ),
    ];
    let mut hash = Sha256::new();
    hash.update(b"rz-pals-system-v6-execution-source/1");
    for (name, bytes) in files {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    format!("{:x}", hash.finalize())
}

pub(super) fn execution_source_observation(
    declared: &BTreeMap<String, String>,
    compiled: &str,
) -> PalsSystemSourceObservationV6 {
    match declared.get(PALS_SYSTEM_V6_EXECUTION_SOURCE_COMPONENT) {
        None => PalsSystemSourceObservationV6::Missing,
        Some(expected) if expected == compiled => PalsSystemSourceObservationV6::Verified,
        Some(_) => PalsSystemSourceObservationV6::Mismatch,
    }
}

#[cfg(all(target_os = "linux", feature = "onnx-cuda"))]
pub(super) fn require_execution_source(
    policy: rz_experiments::PalsSystemSourcePolicyV6,
    observation: PalsSystemSourceObservationV6,
) -> std::io::Result<()> {
    if policy == rz_experiments::PalsSystemSourcePolicyV6::Observe
        || observation == PalsSystemSourceObservationV6::Verified
    {
        Ok(())
    } else {
        Err(std::io::Error::other(
            "PALS system V6 execution source is not verified",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_identity_is_compared_to_actual_compiled_content() {
        let compiled = compiled_pals_system_v6_execution_source_digest();
        let mut declared = BTreeMap::new();
        assert_eq!(
            execution_source_observation(&declared, &compiled),
            PalsSystemSourceObservationV6::Missing
        );
        declared.insert(
            PALS_SYSTEM_V6_EXECUTION_SOURCE_COMPONENT.to_owned(),
            compiled.clone(),
        );
        assert_eq!(
            execution_source_observation(&declared, &compiled),
            PalsSystemSourceObservationV6::Verified
        );
        let first = if compiled.starts_with('0') { '1' } else { '0' };
        let different = format!("{first}{}", &compiled[1..]);
        declared.insert(
            PALS_SYSTEM_V6_EXECUTION_SOURCE_COMPONENT.to_owned(),
            different,
        );
        assert_eq!(
            execution_source_observation(&declared, &compiled),
            PalsSystemSourceObservationV6::Mismatch
        );
    }

    #[cfg(all(target_os = "linux", feature = "onnx-cuda"))]
    #[test]
    fn native_admission_requires_the_new_execution_observation() {
        use rz_experiments::PalsSystemSourcePolicyV6::{Observe, Strict};
        assert!(require_execution_source(Strict, PalsSystemSourceObservationV6::Missing).is_err());
        assert!(require_execution_source(Strict, PalsSystemSourceObservationV6::Mismatch).is_err());
        assert!(require_execution_source(Strict, PalsSystemSourceObservationV6::Verified).is_ok());
        assert!(require_execution_source(Observe, PalsSystemSourceObservationV6::Missing).is_ok());
        assert!(require_execution_source(Observe, PalsSystemSourceObservationV6::Mismatch).is_ok());
    }
}
