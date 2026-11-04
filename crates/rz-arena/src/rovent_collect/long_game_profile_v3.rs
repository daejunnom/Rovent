//! Separate, opt-in long-game collection declaration and custody envelope.
//!
//! This is not registered with the existing `/3` launch plan. A declaration
//! does not itself change admission, release a root slot, establish physical
//! completion, or provide durable publication. The eventual owner must pin
//! this profile independently and enforce every limit at its original scope.

use rz_uci::pals_adaptive::RoventCollectionRootAdmissionV3;
use serde::{Deserialize, Serialize};

pub(crate) const LONG_GAME_PROFILE_DOMAIN: &str =
    "rz-rovent-collection-long-game-profile/1";

/// Root admissions and chess plies are separate quantities: an engine can
/// search a position more than once, and a side need not search every ply.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LongGameProfileV3 {
    pub schema: String,
    pub maximum_game_plies: u32,
    pub maximum_game_root_admissions: u64,
    pub maximum_active_root_slots: u32,
    pub maximum_root_workers: u32,
    pub maximum_simultaneous_teacher_roots: u32,
    /// Upper bound for the original retained evidence owner, not proof that
    /// the selected storage has reserved or durably written this capacity.
    pub maximum_retained_evidence_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LongGameProfileErrorV3 {
    WrongDomain,
    EmptyLimit,
}

impl LongGameProfileV3 {
    /// Checks the domain and required finite, nonzero limits. Zero teachers
    /// explicitly disables teacher roots; the selected collection mode must
    /// separately validate whether this is compatible with its roster.
    pub(crate) fn validate(&self) -> Result<(), LongGameProfileErrorV3> {
        if self.schema != LONG_GAME_PROFILE_DOMAIN {
            return Err(LongGameProfileErrorV3::WrongDomain);
        }
        if self.maximum_game_plies == 0
            || self.maximum_game_root_admissions == 0
            || self.maximum_active_root_slots == 0
            || self.maximum_root_workers == 0
            || self.maximum_retained_evidence_bytes == 0
        {
            return Err(LongGameProfileErrorV3::EmptyLimit);
        }
        Ok(())
    }

    /// A looser registered ceiling is valid: a tighter independent limit still
    /// applies. These values are not a reservation or an ownership proof.
    pub(crate) fn effective_root_worker_limit(&self) -> u64 {
        self.maximum_game_root_admissions
            .min(u64::from(self.maximum_active_root_slots))
            .min(u64::from(self.maximum_root_workers))
    }

    pub(crate) fn effective_teacher_root_limit(&self) -> u64 {
        self.effective_root_worker_limit()
            .min(u64::from(self.maximum_simultaneous_teacher_roots))
    }
}

/// Keeps an original root admission and its original owner together while an
/// audited bridge transfers custody. Merely constructing or retaining this
/// envelope is not an authorization to remove the active root: same-owner
/// native/worker/teacher/route and semantic settlement proofs remain required.
pub(crate) struct PreservedRootOwnerV3<T> {
    admission: RoventCollectionRootAdmissionV3,
    owner: T,
}

impl<T> PreservedRootOwnerV3<T> {
    pub(crate) fn new(admission: RoventCollectionRootAdmissionV3, owner: T) -> Self {
        Self { admission, owner }
    }

    pub(crate) fn matches_original(&self, admission: &RoventCollectionRootAdmissionV3) -> bool {
        self.admission.same_original_admission(admission)
    }

    pub(crate) fn admission(&self) -> &RoventCollectionRootAdmissionV3 {
        &self.admission
    }

    pub(crate) fn owner(&self) -> &T {
        &self.owner
    }

    pub(crate) fn into_parts(self) -> (RoventCollectionRootAdmissionV3, T) {
        (self.admission, self.owner)
    }
}
