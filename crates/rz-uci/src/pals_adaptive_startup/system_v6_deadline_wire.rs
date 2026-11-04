//! Narrow identity for the separately selected deadline wire profile.
//!
//! This inventories source compiled into this child, not the arena binary or
//! proof that a grant, model inference, or physical completion occurred.
use super::PalsSystemSourceObservationV6;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const PALS_SYSTEM_V6_DEADLINE_WIRE_SOURCE_COMPONENT: &str = "pals_system_v6_deadline_wire";

/// Use the existing source-row framing so arena's independently pinned source
/// inventory can verify the same component without changing older digests.
pub fn compiled_pals_system_v6_deadline_wire_source_digest() -> String {
    let files: &[(&str, &[u8])] = &[
        (
            "crates/rz-contracts/src/pals/deadline_grant_v1.rs",
            include_bytes!("../../../rz-contracts/src/pals/deadline_grant_v1.rs"),
        ),
        (
            "crates/rz-uci/src/pals_adaptive_startup/system_v6_deadline_wire.rs",
            include_bytes!("system_v6_deadline_wire.rs"),
        ),
        (
            "crates/rz-uci/src/pals_system_proxy/deadline_grant.rs",
            include_bytes!("../pals_system_proxy/deadline_grant.rs"),
        ),
    ];
    let rows: Vec<_> = files
        .iter()
        .map(|(path, bytes)| {
            (
                (*path).to_owned(),
                bytes.len() as u64,
                <[u8; 32]>::from(Sha256::digest(bytes)),
            )
        })
        .collect();
    let digest = rz_experiments::source_component_digest_v5(&rows)
        .expect("compiled deadline wire source inventory is valid");
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(super) fn source_observation(
    declared: &BTreeMap<String, String>,
) -> PalsSystemSourceObservationV6 {
    match declared.get(PALS_SYSTEM_V6_DEADLINE_WIRE_SOURCE_COMPONENT) {
        None => PalsSystemSourceObservationV6::Missing,
        Some(expected) if *expected == compiled_pals_system_v6_deadline_wire_source_digest() => {
            PalsSystemSourceObservationV6::Verified
        }
        Some(_) => PalsSystemSourceObservationV6::Mismatch,
    }
}

/// Invoke only when the new deadline wire profile is actually selected.
/// Observe preserves unknown or mismatch, while Strict rejects either.
pub(super) fn check_selected_profile(
    policy: rz_experiments::PalsSystemSourcePolicyV6,
    declared: &BTreeMap<String, String>,
) -> std::io::Result<PalsSystemSourceObservationV6> {
    let observed = source_observation(declared);
    if policy == rz_experiments::PalsSystemSourcePolicyV6::Strict
        && observed != PalsSystemSourceObservationV6::Verified
    {
        Err(std::io::Error::other(
            "deadline wire source is not verified",
        ))
    } else {
        Ok(observed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_experiments::PalsSystemSourcePolicyV6::{Observe, Strict};

    #[test]
    fn selected_wire_profile_preserves_observe_and_checks_strict() {
        let mut declared = BTreeMap::new();
        assert_eq!(
            check_selected_profile(Observe, &declared).unwrap(),
            PalsSystemSourceObservationV6::Missing
        );
        assert!(check_selected_profile(Strict, &declared).is_err());
        let compiled = compiled_pals_system_v6_deadline_wire_source_digest();
        let mut mismatch = compiled.clone();
        mismatch.replace_range(..1, if compiled.starts_with('0') { "1" } else { "0" });
        declared.insert(
            PALS_SYSTEM_V6_DEADLINE_WIRE_SOURCE_COMPONENT.into(),
            mismatch,
        );
        assert_eq!(
            check_selected_profile(Observe, &declared).unwrap(),
            PalsSystemSourceObservationV6::Mismatch
        );
        assert!(check_selected_profile(Strict, &declared).is_err());
        declared.insert(
            PALS_SYSTEM_V6_DEADLINE_WIRE_SOURCE_COMPONENT.into(),
            compiled,
        );
        assert_eq!(
            check_selected_profile(Strict, &declared).unwrap(),
            PalsSystemSourceObservationV6::Verified
        );
    }
}
