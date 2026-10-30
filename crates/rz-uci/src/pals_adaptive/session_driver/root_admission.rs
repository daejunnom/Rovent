//! Original startup source selection plus independent per-go root authority.

use super::*;

#[cfg(feature = "pals-adaptive")]
pub(super) struct RoventPcrRootPolicy {
    pub(super) lock: Arc<rz_experiments::RoventInputLock>,
    pub(super) policy: crate::rovent_root_admission::RoventRootAdmissionPolicy,
}

#[cfg(feature = "pals-adaptive")]
pub(super) struct RoventPcrRootPolicyV2 {
    pub(super) owner: rz_experiments::RoventOriginalInputLock,
    pub(super) policy: crate::rovent_root_admission::RoventRootAdmissionPolicy,
}

#[cfg(feature = "pals-adaptive")]
fn v2_declares_own_researcher(
    mode: rz_contracts::rovent::ExecutionMode,
    researcher: Option<&rz_experiments::RoventResearcher>,
) -> bool {
    mode.has_native_r()
        && matches!(
            researcher,
            Some(rz_experiments::RoventResearcher::Own { .. })
        )
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
fn v2_declares_external_researcher(
    mode: rz_contracts::rovent::ExecutionMode,
    researcher: Option<&rz_experiments::RoventResearcher>,
) -> bool {
    mode.has_native_r()
        && matches!(researcher,
            Some(rz_experiments::RoventResearcher::External { profile })
                if profile.purpose == rz_experiments::RoventExternalPurpose::ResearchChecker)
}

#[cfg(all(test, feature = "pals-adaptive"))]
mod v2_topology_tests {
    use super::v2_declares_own_researcher;
    use rz_contracts::rovent::ExecutionMode;
    use rz_experiments::{
        ArtifactRef, RoventExternalProfilePin, RoventExternalPurpose, RoventResearcher,
    };

    #[test]
    fn own_r_registration_rejects_external_absent_and_pc_only_declarations() {
        let own = RoventResearcher::Own {
            implementation_sha256: "a".repeat(64),
            search_configuration_sha256: "b".repeat(64),
        };
        let artifact = ArtifactRef {
            path: "fixture/source".into(),
            sha256: "c".repeat(64),
            bytes: 1,
            source: "fixture".into(),
            license: "fixture".into(),
        };
        let external = RoventResearcher::External {
            profile: Box::new(RoventExternalProfilePin {
                domain: "fixture".into(),
                profile: artifact.clone(),
                canonical_sha256: "d".repeat(64),
                capability_sha256: "e".repeat(64),
                purpose: RoventExternalPurpose::ResearchChecker,
                binary: artifact,
                source_commit: "f".repeat(40),
                requested_threads: 1,
                requested_hash_mib: 1,
            }),
        };
        for mode in [ExecutionMode::CpuPcr, ExecutionMode::HybridPcr] {
            assert!(v2_declares_own_researcher(mode, Some(&own)));
            assert!(!v2_declares_own_researcher(mode, Some(&external)));
            assert!(!v2_declares_own_researcher(mode, None));
            #[cfg(feature = "onnx-cpu")]
            {
                assert!(super::v2_declares_external_researcher(
                    mode,
                    Some(&external)
                ));
                assert!(!super::v2_declares_external_researcher(mode, Some(&own)));
                assert!(!super::v2_declares_external_researcher(mode, None));
            }
        }
        for mode in [ExecutionMode::CpuPc, ExecutionMode::HybridPc] {
            assert!(!v2_declares_own_researcher(mode, Some(&own)));
            assert!(!v2_declares_own_researcher(mode, Some(&external)));
            assert!(!v2_declares_own_researcher(mode, None));
            #[cfg(feature = "onnx-cpu")]
            assert!(!super::v2_declares_external_researcher(
                mode,
                Some(&external)
            ));
        }
        #[cfg(feature = "onnx-cpu")]
        {
            let mut teacher = external.clone();
            if let RoventResearcher::External { profile } = &mut teacher {
                profile.purpose = RoventExternalPurpose::Teacher;
            }
            assert!(!super::v2_declares_external_researcher(
                ExecutionMode::CpuPcr,
                Some(&teacher),
            ));
        }
    }
}

pub(super) struct RoventPcrRootAdmission {
    pub(super) budget: Digest,
    #[cfg(feature = "pals-adaptive")]
    pub(super) original: Option<Arc<crate::rovent_root_admission::RoventRootAdmission>>,
    #[cfg(feature = "pals-adaptive")]
    pub(super) original_v2:
        Option<Arc<crate::rovent_root_admission::source_seal::RoventRootAdmissionV2>>,
}
// One original source is consumed for each root admission; keeping its seal
// inline preserves the same final gate and stopped-source ownership.
#[allow(clippy::large_enum_variant)]
pub(super) enum RoventPcrRootSource {
    Legacy(Digest),
    #[cfg(feature = "pals-adaptive")]
    Original(crate::rovent_root_admission::source_seal::SourceSeal),
    #[cfg(feature = "pals-adaptive")]
    OriginalV2(crate::rovent_root_admission::source_seal::PreparedSourceV2),
}

impl<B> AdaptiveSessionDriver<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    /// Pure framing over a lock already validated by the caller. This creates
    /// no session, root policy registration or live admission authority.
    #[cfg(feature = "pals-adaptive")]
    pub(crate) fn rovent_root_policy_implementation_digest(
        implementation: Digest,
        lock: &rz_experiments::RoventInputLock,
        policy: crate::rovent_root_admission::RoventRootAdmissionPolicy,
    ) -> Digest {
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-pcr-startup-root-policy/1\0");
        hash.update(implementation.0);
        hash.update(lock.canonical_sha256.as_bytes());
        hash.update([match policy {
            crate::rovent_root_admission::RoventRootAdmissionPolicy::ExactRegistered => 0,
            crate::rovent_root_admission::RoventRootAdmissionPolicy::DynamicGo => 1,
        }]);
        hash.update(include_bytes!("root_admission.rs"));
        Digest(hash.finalize().into())
    }

    /// Same implementation identity used by the /2 policy at registration.
    #[cfg(feature = "pals-adaptive")]
    pub(crate) fn rovent_root_policy_implementation_digest_v2(
        implementation: Digest,
        lock: &rz_experiments::RoventInputLockV2,
        policy: crate::rovent_root_admission::RoventRootAdmissionPolicy,
    ) -> Digest {
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-pcr-startup-root-policy/2\0");
        hash.update(implementation.0);
        hash.update(lock.canonical_sha256.as_bytes());
        hash.update([match policy {
            crate::rovent_root_admission::RoventRootAdmissionPolicy::ExactRegistered => 0,
            crate::rovent_root_admission::RoventRootAdmissionPolicy::DynamicGo => 1,
        }]);
        hash.update(include_bytes!("root_admission.rs"));
        Digest(hash.finalize().into())
    }

    /// Startup-only opt-in. The original lock's exact workload is unchanged;
    /// DynamicGo is a distinct per-root domain over the same admitted sources.
    #[cfg(feature = "pals-adaptive")]
    pub fn register_rovent_root_policy(
        &mut self,
        lock: Arc<rz_experiments::RoventInputLock>,
        policy: crate::rovent_root_admission::RoventRootAdmissionPolicy,
    ) -> Result<(), SearchSessionFailure> {
        lock.validate()
            .map_err(|error| SearchSessionFailure::debug("RoventStartupLock", &error))?;
        let profile = self
            .rovent
            .as_ref()
            .ok_or_else(|| failure("RoventStartupPolicy", "no actual Rovent PCR profile"))?;
        let profile_lock: String = profile
            .input_lock
            .0
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        if self.rovent_root_policy.is_some()
            || self.rovent_root_policy_v2.is_some()
            || self.closed.load(Ordering::Acquire)
            || profile_lock != lock.canonical_sha256
            || profile.mode != lock.manifest.engine.mode
            || profile.maximum_goal_plies != lock.manifest.engine.search.registered_max_chess_plies
            || self
                .latest_rovent_root
                .lock()
                .map_err(|_| failure("RoventStartupPolicy", "root receipt poisoned"))?
                .is_some()
        {
            return Err(failure(
                "RoventStartupPolicy",
                "startup source/mode/policy differs or was already used",
            ));
        }
        let state = self.state.try_lock().map_err(|_| {
            failure(
                "RoventStartupPolicy",
                "startup root policy requires the original idle owner",
            )
        })?;
        if !matches!(&*state, SessionState::Idle(owners) if owners.continuation.is_none() && owners.native_finished.is_none())
        {
            return Err(failure(
                "RoventStartupPolicy",
                "root policy cannot change after physical work",
            ));
        }
        let SessionState::Idle(owners) = &*state else {
            unreachable!()
        };
        let ready = owners
            .native
            .ready_clock()
            .ok_or_else(SearchSessionFailure::physical_completion_unknown)?;
        if ready.domain() != self.clock.domain()
            || ready.origin() != self.clock.runtime_clock().origin()
            || owners.native.is_quarantined()
        {
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        self.options.implementation = Self::rovent_root_policy_implementation_digest(
            self.options.implementation,
            &lock,
            policy,
        );
        self.rovent_root_policy = Some(RoventPcrRootPolicy { lock, policy });
        Ok(())
    }

    /// Register the actual /2 allocation; no /1 lock is synthesized. This
    /// remains a declaration until the original root crosses its live gate.
    #[cfg(feature = "pals-adaptive")]
    pub fn register_rovent_root_policy_v2(
        &mut self,
        lock: Arc<rz_experiments::RoventInputLockV2>,
        policy: crate::rovent_root_admission::RoventRootAdmissionPolicy,
    ) -> Result<(), SearchSessionFailure> {
        let owner = rz_experiments::RoventOriginalInputLock::from_v2(lock).map_err(|rejected| {
            SearchSessionFailure::debug("RoventStartupLockV2", &rejected.error)
        })?;
        let lock = owner.v2().expect("original /2 lock");
        let profile = self
            .rovent
            .as_ref()
            .ok_or_else(|| failure("RoventStartupPolicy", "no actual Rovent PCR profile"))?;
        let profile_lock: String = profile
            .input_lock
            .0
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        if self.rovent_root_policy.is_some()
            || self.rovent_root_policy_v2.is_some()
            || self.closed.load(Ordering::Acquire)
            || self.rovent_collection_origin.is_some()
            || {
                let own = v2_declares_own_researcher(
                    lock.manifest.engine.mode,
                    lock.manifest.engine.researcher.as_ref(),
                );
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                let external = v2_declares_external_researcher(
                    lock.manifest.engine.mode,
                    lock.manifest.engine.researcher.as_ref(),
                );
                #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
                let external = false;
                !own && !external
            }
            || profile_lock != lock.canonical_sha256
            || profile.mode != lock.manifest.engine.mode
            || profile.maximum_goal_plies != lock.manifest.engine.search.registered_max_chess_plies
            || self
                .latest_rovent_root
                .lock()
                .map_err(|_| failure("RoventStartupPolicy", "root receipt poisoned"))?
                .is_some()
            || self
                .latest_rovent_root_v2
                .lock()
                .map_err(|_| failure("RoventStartupPolicy", "V2 root receipt poisoned"))?
                .is_some()
        {
            return Err(failure(
                "RoventStartupPolicy",
                "startup source/mode/policy differs or was already used",
            ));
        }
        let state = self.state.try_lock().map_err(|_| {
            failure(
                "RoventStartupPolicy",
                "startup root policy requires the original idle owner",
            )
        })?;
        let SessionState::Idle(owners) = &*state else {
            return Err(failure(
                "RoventStartupPolicy",
                "root policy cannot change after physical work",
            ));
        };
        if owners.continuation.is_some() || owners.native_finished.is_some() {
            return Err(failure(
                "RoventStartupPolicy",
                "root policy cannot change after physical work",
            ));
        }
        let own_ready = v2_declares_own_researcher(
            lock.manifest.engine.mode,
            lock.manifest.engine.researcher.as_ref(),
        ) && !owners.researcher_source.is_profiled_external()
            && owners.external.is_none()
            && !owners.cpu.is_empty();
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        let external_ready = v2_declares_external_researcher(
            lock.manifest.engine.mode,
            lock.manifest.engine.researcher.as_ref(),
        ) && owners.external.is_none()
            && owners.cpu.is_empty()
            && matches!(&owners.researcher_source,
                SessionResearcherSource::ProfiledExternalV2(source)
                    if source.same_original_lock(lock)
                        && source.ready_for_root_policy_registration());
        #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
        let external_ready = false;
        if !own_ready && !external_ready {
            return Err(failure(
                "RoventStartupPolicy",
                "V2 researcher requires its original ready native or pinned external owner",
            ));
        }
        let ready = owners
            .native
            .ready_clock()
            .ok_or_else(SearchSessionFailure::physical_completion_unknown)?;
        if ready.domain() != self.clock.domain()
            || ready.origin() != self.clock.runtime_clock().origin()
            || owners.native.is_quarantined()
        {
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        self.options.implementation = Self::rovent_root_policy_implementation_digest_v2(
            self.options.implementation,
            lock,
            policy,
        );
        self.rovent_root_policy_v2 = Some(RoventPcrRootPolicyV2 { owner, policy });
        Ok(())
    }

    pub(super) fn prepare_rovent_go(
        &self,
        position: &Position,
        context: &SearchSessionContext,
    ) -> Result<Option<RoventPcrRootSource>, SearchSessionFailure> {
        #[cfg(not(feature = "pals-adaptive"))]
        let _ = (position, context);
        let Some(profile) = &self.rovent else {
            return Ok(None);
        };
        #[cfg(feature = "pals-adaptive")]
        if let Some(registered) = &self.rovent_root_policy {
            let source = crate::rovent_root_admission::source_seal::SourceSeal::prepare(
                &registered.lock,
                registered.policy,
                position,
                context,
                &self.clock,
            )
            .map_err(|error| SearchSessionFailure::debug("RoventRootAdmission", &error))?;
            return Ok(Some(RoventPcrRootSource::Original(source)));
        }
        #[cfg(feature = "pals-adaptive")]
        if let Some(registered) = &self.rovent_root_policy_v2 {
            let source = crate::rovent_root_admission::source_seal::PreparedSourceV2::prepare(
                &registered.owner,
                registered.policy,
                position,
                context,
                &self.clock,
            )
            .map_err(|error| SearchSessionFailure::debug("RoventRootAdmissionV2", &error))?;
            return Ok(Some(RoventPcrRootSource::OriginalV2(source)));
        }
        Ok(Some(RoventPcrRootSource::Legacy(profile.root_budget)))
    }
}
