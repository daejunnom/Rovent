//! One parsed, bounded expected context for the NoGame smoke invocation.
//! These argv values are declarations, never peer, plan, executable, source,
//! role-listener or physical-completion authentication.

use super::proxy_entry_smoke_channel_v1::{
    ExpectedSmokeContext, ObservedSmokeChallenge, PendingDirectSmokeChallenge,
    PendingDirectSmokeGrant,
};
use rz_contracts::pals::{
    deadline_grant_v1::DeadlineRole,
    proxy_entry_smoke_grant_v1::{ProxyEntrySmokePhase, ProxyEntrySmokePolicy},
};
use std::io;

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn hex<const N: usize>(text: &str) -> io::Result<[u8; N]> {
    if text.len() != N * 2 {
        return Err(invalid("smoke expected hex length invalid"));
    }
    let mut bytes = [0; N];
    fn nibble(value: u8) -> Option<u8> {
        match value {
            b'0'..=b'9' => Some(value - b'0'),
            b'a'..=b'f' => Some(value - b'a' + 10),
            b'A'..=b'F' => Some(value - b'A' + 10),
            _ => None,
        }
    }
    for (i, pair) in text.as_bytes().chunks_exact(2).enumerate() {
        bytes[i] = (nibble(pair[0]).ok_or_else(|| invalid("smoke expected hex invalid"))? << 4)
            | nibble(pair[1]).ok_or_else(|| invalid("smoke expected hex invalid"))?;
    }
    Ok(bytes)
}

fn set_once<'a>(slot: &mut Option<&'a str>, value: &'a str) -> io::Result<()> {
    if slot.is_some() {
        return Err(invalid("duplicate smoke expected field"));
    }
    *slot = Some(value);
    Ok(())
}

struct ParsedFields {
    expected: ExpectedSmokeContext,
    plan_sha256: [u8; 32],
    proxy_sha256: [u8; 32],
}

fn fields(arguments: &[String]) -> io::Result<ParsedFields> {
    let (mut policy, mut owner, mut epoch, mut role, mut generation, mut lock, mut plan, mut proxy) =
        (None, None, None, None, None, None, None, None);
    for argument in arguments {
        let Some((name, value)) = argument.split_once('=') else {
            continue;
        };
        match name {
            "--pals-smoke-policy" => set_once(&mut policy, value)?,
            "--pals-smoke-owner-id" => set_once(&mut owner, value)?,
            "--pals-smoke-owner-epoch" => set_once(&mut epoch, value)?,
            "--pals-smoke-role" => set_once(&mut role, value)?,
            "--pals-smoke-generation" => set_once(&mut generation, value)?,
            "--pals-smoke-lock-sha256" => set_once(&mut lock, value)?,
            "--pals-smoke-plan-sha256" => set_once(&mut plan, value)?,
            "--pals-smoke-proxy-sha256" => set_once(&mut proxy, value)?,
            _ => {}
        }
    }
    if policy != Some("proxy-entry-v1") {
        return Err(invalid("smoke expected policy missing or unsupported"));
    }
    let owner_id = hex::<16>(owner.ok_or_else(|| invalid("smoke expected owner missing"))?)?;
    let owner_epoch = epoch
        .ok_or_else(|| invalid("smoke expected epoch missing"))?
        .parse::<u64>()
        .map_err(|_| invalid("smoke expected epoch invalid"))?;
    let generation = generation
        .ok_or_else(|| invalid("smoke expected generation missing"))?
        .parse::<u64>()
        .map_err(|_| invalid("smoke expected generation invalid"))?;
    if owner_id == [0; 16] || owner_epoch == 0 || generation == 0 {
        return Err(invalid("smoke expected identity invalid"));
    }
    let role = match role {
        Some("baseline") => DeadlineRole::Baseline,
        Some("candidate") => DeadlineRole::Candidate,
        _ => return Err(invalid("smoke expected role invalid")),
    };
    Ok(ParsedFields {
        expected: ExpectedSmokeContext {
            policy: ProxyEntrySmokePolicy::ProxyEntryFunctionalSmokeV1,
            owner_id,
            owner_epoch,
            role,
            generation,
            lock_sha256: hex::<32>(lock.ok_or_else(|| invalid("smoke expected lock missing"))?)?,
        },
        plan_sha256: hex::<32>(plan.ok_or_else(|| invalid("smoke expected plan missing"))?)?,
        proxy_sha256: hex::<32>(proxy.ok_or_else(|| invalid("smoke expected proxy missing"))?)?,
    })
}

/// The original argv and all expected fields stay together; no Clone is
/// provided. Parsing failure returns the exact argument vector to the caller.
pub(super) struct ParsedSmokeExpectedArgvV1 {
    arguments: Vec<String>,
    expected: ExpectedSmokeContext,
    declared_plan_sha256: [u8; 32],
    declared_proxy_sha256: [u8; 32],
}

impl ParsedSmokeExpectedArgvV1 {
    pub(super) fn from_arguments(arguments: Vec<String>) -> Result<Self, (io::Error, Vec<String>)> {
        match fields(&arguments) {
            Ok(parsed) => Ok(Self {
                arguments,
                expected: parsed.expected,
                declared_plan_sha256: parsed.plan_sha256,
                declared_proxy_sha256: parsed.proxy_sha256,
            }),
            Err(error) => Err((error, arguments)),
        }
    }
    /// Compare declarations with the same pinned local lock; this does not
    /// authenticate argv provenance, the plan or the running proxy binary.
    pub(super) fn matches_local_lock(&self, canonical_sha256: &str, generation: u64) -> bool {
        hex::<32>(canonical_sha256).is_ok_and(|digest| digest == self.expected.lock_sha256)
            && generation == self.expected.generation
    }

    pub(super) fn original_arguments(&self) -> &[String] {
        &self.arguments
    }
    /// Declaration-only digests, not a verified original owner or mapped image.
    pub(super) fn declared_plan_sha256(&self) -> &[u8; 32] {
        &self.declared_plan_sha256
    }
    pub(super) fn declared_proxy_sha256(&self) -> &[u8; 32] {
        &self.declared_proxy_sha256
    }
    /// The challenge itself must already have been read from the retained
    /// original direct-parent stream. The caller still proves this argv was
    /// created by the original plan and independently validates its binary.
    pub(super) fn compare_original_challenge(
        &self,
        challenge: PendingDirectSmokeChallenge,
        phase: ProxyEntrySmokePhase,
        prior: Option<&ObservedSmokeChallenge>,
    ) -> Result<
        (PendingDirectSmokeGrant, ObservedSmokeChallenge),
        (io::Error, PendingDirectSmokeChallenge),
    > {
        challenge.into_grant(&self.expected, phase, prior)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn arguments() -> Vec<String> {
        vec![
            "--pals-smoke-policy=proxy-entry-v1".into(),
            format!("--pals-smoke-owner-id={}", "ab".repeat(16)),
            "--pals-smoke-owner-epoch=2".into(),
            "--pals-smoke-role=baseline".into(),
            "--pals-smoke-generation=3".into(),
            format!("--pals-smoke-lock-sha256={}", "0F".repeat(32)),
            format!("--pals-smoke-plan-sha256={}", "12".repeat(32)),
            format!("--pals-smoke-proxy-sha256={}", "34".repeat(32)),
            "--unrelated=preserved".into(),
        ]
    }
    #[test]
    fn context_keeps_argv_and_accepts_hex_case() {
        let args = arguments();
        let parsed = ParsedSmokeExpectedArgvV1::from_arguments(args.clone())
            .ok()
            .unwrap();
        assert_eq!(parsed.original_arguments(), args);
        assert_eq!(parsed.expected.owner_id, [0xab; 16]);
        assert_eq!(parsed.expected.lock_sha256, [0x0f; 32]);
        assert_eq!(parsed.declared_plan_sha256(), &[0x12; 32]);
        assert_eq!(parsed.declared_proxy_sha256(), &[0x34; 32]);
        assert!(parsed.matches_local_lock(&"0f".repeat(32), 3));
        assert!(!parsed.matches_local_lock(&"10".repeat(32), 3));
        assert!(!parsed.matches_local_lock(&"0f".repeat(32), 4));
    }
    #[test]
    fn duplicates_and_invalid_fields_return_the_original_argv() {
        let mut args = arguments();
        args.push("--pals-smoke-role=candidate".into());
        let (_, returned) = match ParsedSmokeExpectedArgvV1::from_arguments(args.clone()) {
            Ok(_) => panic!("duplicate must fail"),
            Err(error) => error,
        };
        assert_eq!(returned, args);
        let mut args = arguments();
        args[2] = "--pals-smoke-owner-epoch=0".into();
        assert!(ParsedSmokeExpectedArgvV1::from_arguments(args).is_err());
        let mut args = arguments();
        args[5] = "--pals-smoke-lock-sha256=bad".into();
        assert!(ParsedSmokeExpectedArgvV1::from_arguments(args).is_err());
    }
}
