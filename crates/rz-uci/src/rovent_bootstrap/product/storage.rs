//! Original registration shared with exactly one admitted session. A setup is
//! issued only against its original opaque go; registration is not readiness.
//! A session using setup must include archive owner return and worker join in
//! its actual Confirmed finish before product custody can be released.

use super::{
    ActualProductDriver, RoventNativeProductDriver, RoventNativeProductDriverV2,
    RoventOwnNnuecProductDriverV2, RoventOwnPcrProductDriverV2,
};
use crate::engine::ProcessClock;
use crate::rovent_bootstrap::launch::storage::{
    RoventStorageAdmissionOwners, RoventStorageCompiledCustody, RoventStorageCompiledCustodyV2,
    RoventStorageOwnedRuntimeRegistration, RoventStorageOwnedRuntimeRegistrationV2,
    RoventStorageRegistrationFailureV2, RoventStorageRuntimeError,
    RoventStorageRuntimeOwnershipFailure, RoventStorageRuntimeOwnershipFailureV2,
    RoventStorageRuntimeRegistrationFailure, RoventStorageSetupRejection,
    own_rovent_storage_runtime_registration, own_rovent_storage_runtime_registration_v2,
    register_rovent_storage_runtime, register_rovent_storage_runtime_v2,
};
use crate::rovent_bootstrap::process_boundary::{
    VerifiedOwnedRoventProcessBoundary, VerifiedOwnedRoventProcessBoundaryV2,
};
use crate::rovent_root_admission::{RoventRootAdmission, source_seal::RoventRootAdmissionV2};
use crate::search_driver::{DriverPhysicalCompletion, SearchSessionDriver, SearchSessionFailure};
use rz_experiments::{ManifestError, RoventInputLock, RoventInputLockV2, RoventStorageProfile};
use rz_runtime::contracts::ContractClock;
use rz_search::pals::engine::idle_archive::{
    UnusedIdleArchiveSetup, UnusedIdleArchiveSetupIdentity,
};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

/// The same move-only registration remains in one mutex. Immutable copies are
/// bounded attribution/profile selections, never runtime or physical proof.
pub(crate) struct SharedRoventStorageRegistration {
    original: Mutex<StorageRegistrationOwner>,
    input_lock: StorageInputOwner,
    profile: RoventStorageProfile,
    identity: UnusedIdleArchiveSetupIdentity,
}
enum StorageRegistrationOwner {
    V1(RoventStorageOwnedRuntimeRegistration),
    V2(RoventStorageOwnedRuntimeRegistrationV2),
}
enum StorageInputOwner {
    V1(Arc<RoventInputLock>),
    V2(Arc<RoventInputLockV2>),
}
impl SharedRoventStorageRegistration {
    pub(crate) fn new(original: RoventStorageOwnedRuntimeRegistration) -> Self {
        Self {
            input_lock: StorageInputOwner::V1(Arc::clone(original.owners().input_lock())),
            profile: original.profile().clone(),
            identity: original.identity(),
            original: Mutex::new(StorageRegistrationOwner::V1(original)),
        }
    }
    pub(crate) fn new_v2(original: RoventStorageOwnedRuntimeRegistrationV2) -> Self {
        Self {
            input_lock: StorageInputOwner::V2(Arc::clone(original.owners().input_lock())),
            profile: original.profile().clone(),
            identity: original.identity(),
            original: Mutex::new(StorageRegistrationOwner::V2(original)),
        }
    }
    pub(crate) fn identity(&self) -> UnusedIdleArchiveSetupIdentity {
        self.identity
    }
    pub(crate) fn profile(&self) -> &RoventStorageProfile {
        &self.profile
    }
    pub(crate) fn input_lock_owner(&self) -> &Arc<RoventInputLock> {
        match &self.input_lock {
            StorageInputOwner::V1(lock) => lock,
            StorageInputOwner::V2(_) => unreachable!("V1-only caller"),
        }
    }
    pub(crate) fn validate_profile(&self) -> Result<(), ManifestError> {
        match &self.input_lock {
            StorageInputOwner::V1(lock) => self.profile.validate_for_manifest(&lock.manifest),
            StorageInputOwner::V2(lock) => self.profile.validate_for_manifest_v2(&lock.manifest),
        }
    }
    pub(crate) fn accepts_v1_admission(&self) -> bool {
        matches!(&self.input_lock, StorageInputOwner::V1(_))
    }
    pub(crate) fn validate_v2_admission(
        &self,
        admission: &RoventRootAdmissionV2,
    ) -> Result<(), RoventStorageRuntimeError> {
        let owner = self.original.try_lock().map_err(|cause| {
            RoventStorageRuntimeError::Namespace(match cause {
                std::sync::TryLockError::WouldBlock => "original storage registration is busy",
                std::sync::TryLockError::Poisoned(_) => "original storage registration is poisoned",
            })
        })?;
        match (&*owner, &self.input_lock) {
            (StorageRegistrationOwner::V2(original), StorageInputOwner::V2(lock))
                if Arc::ptr_eq(original.owners().input_lock(), lock) =>
            {
                original.validate_live_admission(admission)
            }
            _ => Err(RoventStorageRuntimeError::Namespace(
                "V2 admission and original storage owner differ",
            )),
        }
    }
    /// The mutex is held only for the original one-shot split. No archive IO,
    /// await, drain, worker completion or join is performed under this lock.
    pub(crate) fn take_unused_setup(
        &self,
        admission: Arc<RoventRootAdmission>,
    ) -> Result<UnusedIdleArchiveSetup, RoventStorageSetupRejection> {
        match self.original.try_lock() {
            Ok(mut owner) => {
                let StorageRegistrationOwner::V1(original) = &mut *owner else {
                    return Err(RoventStorageSetupRejection {
                        error: RoventStorageRuntimeError::Namespace(
                            "V1 admission with V2 registration",
                        ),
                        admission,
                    });
                };
                if !Arc::ptr_eq(original.owners().input_lock(), self.input_lock_owner()) {
                    return Err(RoventStorageSetupRejection {
                        error: RoventStorageRuntimeError::Namespace(
                            "shared storage attribution no longer names the original owner",
                        ),
                        admission,
                    });
                }
                original.take_unused_idle_setup(admission)
            }
            Err(cause) => Err(RoventStorageSetupRejection {
                error: RoventStorageRuntimeError::Namespace(match cause {
                    std::sync::TryLockError::WouldBlock => "original storage registration is busy",
                    std::sync::TryLockError::Poisoned(_) => {
                        "original storage registration is poisoned"
                    }
                }),
                admission,
            }),
        }
    }
    pub(crate) fn take_unused_setup_v2(
        &self,
        admission: Arc<RoventRootAdmissionV2>,
    ) -> Result<UnusedIdleArchiveSetup, RoventStorageSetupRejection<RoventRootAdmissionV2>> {
        match self.original.try_lock() {
            Ok(mut owner) => {
                let (StorageRegistrationOwner::V2(original), StorageInputOwner::V2(lock)) =
                    (&mut *owner, &self.input_lock)
                else {
                    return Err(RoventStorageSetupRejection {
                        error: RoventStorageRuntimeError::Namespace(
                            "V2 admission with V1 registration",
                        ),
                        admission,
                    });
                };
                if !Arc::ptr_eq(original.owners().input_lock(), lock) {
                    return Err(RoventStorageSetupRejection {
                        error: RoventStorageRuntimeError::Namespace(
                            "shared V2 storage attribution changed",
                        ),
                        admission,
                    });
                }
                original.take_unused_idle_setup(admission)
            }
            Err(cause) => Err(RoventStorageSetupRejection {
                error: RoventStorageRuntimeError::Namespace(match cause {
                    std::sync::TryLockError::WouldBlock => "original storage registration is busy",
                    std::sync::TryLockError::Poisoned(_) => {
                        "original storage registration is poisoned"
                    }
                }),
                admission,
            }),
        }
    }
}

/// Move-only custody. Unknown Drop deliberately retains the original value;
/// neither Drop nor a diagnostic can mark the actual product physically closed.
pub(super) struct RetainedOnUnknown<T> {
    original: Option<T>,
    physically_closed: Arc<AtomicBool>,
}
impl<T> RetainedOnUnknown<T> {
    fn new(original: T, physically_closed: Arc<AtomicBool>) -> Self {
        Self {
            original: Some(original),
            physically_closed,
        }
    }
    fn original(&self) -> &T {
        self.original.as_ref().expect("original storage owner")
    }
}
impl<T> Drop for RetainedOnUnknown<T> {
    fn drop(&mut self) {
        if !self.physically_closed.load(Ordering::Acquire) {
            if let Some(original) = self.original.take() {
                std::mem::forget(original);
            }
        }
    }
}
pub(super) type RetainedStorageRegistration =
    RetainedOnUnknown<Arc<SharedRoventStorageRegistration>>;

// Preserve the exact partial proof/owners from each original API. No borrowed
// failure is decoded into a replacement registration or new FD capability.
#[allow(clippy::large_enum_variant)]
enum RejectedStorageOwners<'boundary, 'process> {
    Pending {
        custody: RoventStorageCompiledCustody<'boundary, 'process>,
        boundary: Arc<VerifiedOwnedRoventProcessBoundary>,
    },
    Registration {
        rejected: RoventStorageRuntimeRegistrationFailure<'boundary, 'process>,
        boundary: Arc<VerifiedOwnedRoventProcessBoundary>,
    },
    Ownership(RoventStorageRuntimeOwnershipFailure<'boundary, 'process>),
    Session {
        registration: RoventStorageOwnedRuntimeRegistration,
        failure: SearchSessionFailure,
    },
}

/// A rejection returns the same original partial custody and owned process
/// boundary. It shares only the real product's physical-close observation.
/// Dropping it before actual native closure quarantines those owners as well.
pub struct RoventProductStorageStartupRejection<'boundary, 'process> {
    pub failure: SearchSessionFailure,
    retained: RetainedOnUnknown<RejectedStorageOwners<'boundary, 'process>>,
}
impl RoventProductStorageStartupRejection<'_, '_> {
    /// Preserve a session-install failure separately from bounded CLI prose.
    pub fn retained_session_failure(&self) -> Option<&SearchSessionFailure> {
        match self.retained.original() {
            RejectedStorageOwners::Session { failure, .. } => Some(failure),
            _ => None,
        }
    }

    pub fn owners(&self) -> &RoventStorageAdmissionOwners {
        match self.retained.original() {
            RejectedStorageOwners::Pending { custody, .. } => custody.owners(),
            RejectedStorageOwners::Registration { rejected, .. } => rejected.custody.owners(),
            RejectedStorageOwners::Ownership(rejected) => rejected.registration.owners(),
            RejectedStorageOwners::Session { registration, .. } => registration.owners(),
        }
    }
    pub fn process_boundary(&self) -> &Arc<VerifiedOwnedRoventProcessBoundary> {
        match self.retained.original() {
            RejectedStorageOwners::Pending { boundary, .. }
            | RejectedStorageOwners::Registration { boundary, .. } => boundary,
            RejectedStorageOwners::Ownership(rejected) => &rejected.candidate,
            RejectedStorageOwners::Session { registration, .. } => registration.process_boundary(),
        }
    }
}

fn same_original_lock<T>(source: &T, storage: &Arc<T>, process: &Arc<T>) -> bool {
    std::ptr::eq(source, storage.as_ref()) && Arc::ptr_eq(storage, process)
}

fn validate_attachment_selection(
    already_installed: bool,
    physically_closed: bool,
    same_lock_owner: bool,
    actual_clock: &ProcessClock,
    requested_clock: &ProcessClock,
) -> Result<(), &'static str> {
    if already_installed {
        return Err("storage runtime registration is already installed");
    }
    if physically_closed {
        return Err("product is already physically closed");
    }
    if !same_lock_owner {
        return Err("product, storage and process do not retain the same original lock object");
    }
    let actual = actual_clock.runtime_clock();
    let requested = requested_clock.runtime_clock();
    if actual.domain() != requested.domain() || actual.origin() != requested.origin() {
        return Err("storage registration clock differs from the original product clock");
    }
    Ok(())
}

impl RoventNativeProductDriver {
    /// Explicit move-only ownership installation, not archive execution. The
    /// same prepared source Arc feeds this product and its actual inner driver.
    /// A caller cannot choose a replacement implementation/clock from a receipt:
    /// register uses only this product's original constructor clock and actual
    /// inner driver's implementation. No Store/worker/setup is started here.
    #[allow(clippy::result_large_err)]
    pub fn attach_storage_runtime_registration<'boundary, 'process>(
        &mut self,
        custody: RoventStorageCompiledCustody<'boundary, 'process>,
        boundary: Arc<VerifiedOwnedRoventProcessBoundary>,
        requested_clock: &ProcessClock,
    ) -> Result<(), RoventProductStorageStartupRejection<'boundary, 'process>> {
        if let Err(reason) = validate_attachment_selection(
            self.storage_registration.is_some(),
            self.physically_closed.load(Ordering::Acquire),
            same_original_lock(
                self.sources.lock(),
                custody.owners().input_lock(),
                boundary.input_lock(),
            ),
            &self.process_clock,
            requested_clock,
        ) {
            return Err(
                self.reject_storage(reason, RejectedStorageOwners::Pending { custody, boundary })
            );
        }
        let registration = match register_rovent_storage_runtime(
            custody,
            self.process_clock.clone(),
            self.actual().implementation(),
        ) {
            Ok(registration) => registration,
            Err(rejected) => {
                return Err(self.reject_storage(
                    rejected.error.to_string(),
                    RejectedStorageOwners::Registration { rejected, boundary },
                ));
            }
        };
        match own_rovent_storage_runtime_registration(registration, boundary) {
            Ok(registration) => match self.install_session_storage(registration) {
                Ok(shared) => {
                    self.storage_registration = Some(RetainedOnUnknown::new(
                        shared,
                        Arc::clone(&self.physically_closed),
                    ));
                    Ok(())
                }
                Err((registration, failure)) => Err(self.reject_storage(
                    format!("{}: {}", failure.code, failure.detail),
                    RejectedStorageOwners::Session {
                        registration,
                        failure,
                    },
                )),
            },
            Err(rejected) => Err(self.reject_storage(
                rejected.error.to_string(),
                RejectedStorageOwners::Ownership(rejected),
            )),
        }
    }

    #[allow(clippy::result_large_err)]
    fn install_session_storage(
        &self,
        registration: RoventStorageOwnedRuntimeRegistration,
    ) -> Result<
        Arc<SharedRoventStorageRegistration>,
        (RoventStorageOwnedRuntimeRegistration, SearchSessionFailure),
    > {
        match self.inner.as_ref() {
            ActualProductDriver::Pc(driver) | ActualProductDriver::PcNnuec(driver) => {
                driver.install_storage_registration(registration)
            }
            ActualProductDriver::Nnuec(driver) => driver.install_storage_registration(registration),
            ActualProductDriver::Pcr(driver) => driver.install_storage_registration(registration),
        }
    }

    fn reject_storage<'boundary, 'process>(
        &self,
        detail: impl AsRef<str>,
        owners: RejectedStorageOwners<'boundary, 'process>,
    ) -> RoventProductStorageStartupRejection<'boundary, 'process> {
        RoventProductStorageStartupRejection {
            failure: SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Unknown,
                code: "RoventStorageRuntimeRegistration",
                detail: detail.as_ref().chars().take(512).collect(),
            },
            retained: RetainedOnUnknown::new(owners, Arc::clone(&self.physically_closed)),
        }
    }
}

#[allow(clippy::large_enum_variant)]
enum RejectedStorageOwnersV2<'boundary, 'process> {
    Pending {
        custody: RoventStorageCompiledCustodyV2<'boundary, 'process>,
        boundary: Arc<VerifiedOwnedRoventProcessBoundaryV2>,
    },
    Registration {
        rejected: RoventStorageRegistrationFailureV2<'boundary, 'process>,
        boundary: Arc<VerifiedOwnedRoventProcessBoundaryV2>,
    },
    Ownership(RoventStorageRuntimeOwnershipFailureV2<'boundary, 'process>),
    Session {
        registration: RoventStorageOwnedRuntimeRegistrationV2,
        failure: SearchSessionFailure,
    },
}

/// Failure retains the original /2 custody and process owner until the real
/// product has physically closed; a rejection is never a fence.
pub struct RoventProductStorageStartupRejectionV2<'boundary, 'process> {
    pub failure: SearchSessionFailure,
    retained: RetainedOnUnknown<RejectedStorageOwnersV2<'boundary, 'process>>,
}
impl RoventProductStorageStartupRejectionV2<'_, '_> {
    pub fn owners(&self) -> &RoventStorageAdmissionOwners<RoventInputLockV2> {
        match self.retained.original() {
            RejectedStorageOwnersV2::Pending { custody, .. } => custody.owners(),
            RejectedStorageOwnersV2::Registration { rejected, .. } => rejected.custody.owners(),
            RejectedStorageOwnersV2::Ownership(rejected) => rejected.registration.owners(),
            RejectedStorageOwnersV2::Session { registration, .. } => registration.owners(),
        }
    }
    pub fn process_boundary(&self) -> &Arc<VerifiedOwnedRoventProcessBoundaryV2> {
        match self.retained.original() {
            RejectedStorageOwnersV2::Pending { boundary, .. }
            | RejectedStorageOwnersV2::Registration { boundary, .. } => boundary,
            RejectedStorageOwnersV2::Ownership(rejected) => &rejected.candidate,
            RejectedStorageOwnersV2::Session { registration, .. } => {
                registration.process_boundary()
            }
        }
    }
    pub fn retained_session_failure(&self) -> Option<&SearchSessionFailure> {
        match self.retained.original() {
            RejectedStorageOwnersV2::Session { failure, .. } => Some(failure),
            _ => None,
        }
    }
}

macro_rules! implement_v2_storage_attachment {
    ($product:ty) => {
        impl $product {
            #[allow(clippy::result_large_err)]
            pub fn attach_storage_runtime_registration_v2<'boundary, 'process>(
                &mut self,
                custody: RoventStorageCompiledCustodyV2<'boundary, 'process>,
                boundary: Arc<VerifiedOwnedRoventProcessBoundaryV2>,
                requested_clock: &ProcessClock,
            ) -> Result<(), RoventProductStorageStartupRejectionV2<'boundary, 'process>> {
                let same_owner = Arc::ptr_eq(&self.process, &boundary)
                    && same_original_lock(
                        self.sources.lock(),
                        custody.owners().input_lock(),
                        boundary.input_lock(),
                    );
                if let Err(reason) = validate_attachment_selection(
                    self.storage_registration.is_some(),
                    self.physically_closed.load(Ordering::Acquire),
                    same_owner,
                    &self.process_clock,
                    requested_clock,
                ) {
                    return Err(self.reject_storage_v2(
                        reason,
                        RejectedStorageOwnersV2::Pending { custody, boundary },
                    ));
                }
                let registration = match register_rovent_storage_runtime_v2(
                    custody,
                    self.process_clock.clone(),
                    self.inner.implementation(),
                ) {
                    Ok(registration) => registration,
                    Err(rejected) => {
                        return Err(self.reject_storage_v2(
                            rejected.error.to_string(),
                            RejectedStorageOwnersV2::Registration { rejected, boundary },
                        ));
                    }
                };
                match own_rovent_storage_runtime_registration_v2(registration, boundary) {
                    Ok(registration) => {
                        match self.inner.install_storage_registration_v2(registration) {
                            Ok(shared) => {
                                self.storage_registration = Some(RetainedOnUnknown::new(
                                    shared,
                                    Arc::clone(&self.physically_closed),
                                ));
                                Ok(())
                            }
                            Err((registration, failure)) => Err(self.reject_storage_v2(
                                format!("{}: {}", failure.code, failure.detail),
                                RejectedStorageOwnersV2::Session {
                                    registration,
                                    failure,
                                },
                            )),
                        }
                    }
                    Err(rejected) => Err(self.reject_storage_v2(
                        rejected.error.to_string(),
                        RejectedStorageOwnersV2::Ownership(rejected),
                    )),
                }
            }
            fn reject_storage_v2<'boundary, 'process>(
                &self,
                detail: impl AsRef<str>,
                owners: RejectedStorageOwnersV2<'boundary, 'process>,
            ) -> RoventProductStorageStartupRejectionV2<'boundary, 'process> {
                RoventProductStorageStartupRejectionV2 {
                    failure: SearchSessionFailure {
                        physical_completion: DriverPhysicalCompletion::Unknown,
                        code: "RoventStorageRuntimeRegistrationV2",
                        detail: detail.as_ref().chars().take(512).collect(),
                    },
                    retained: RetainedOnUnknown::new(owners, Arc::clone(&self.physically_closed)),
                }
            }
        }
    };
}
implement_v2_storage_attachment!(RoventNativeProductDriverV2);
implement_v2_storage_attachment!(RoventOwnPcrProductDriverV2);
implement_v2_storage_attachment!(RoventOwnNnuecProductDriverV2);

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::ProcessEpoch;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn lock_identity_rejects_equal_values_in_another_original_object() {
        // Pointer comparison only, no fake process/storage proof is issued.
        let original = Arc::new(7_u8);
        let alias = Arc::clone(&original);
        let same_bytes = Arc::new(*original);
        assert!(same_original_lock(original.as_ref(), &alias, &original));
        assert!(!same_original_lock(same_bytes.as_ref(), &alias, &original));
        assert!(!same_original_lock(
            original.as_ref(),
            &same_bytes,
            &original
        ));
        assert!(!same_original_lock(original.as_ref(), &alias, &same_bytes));
    }

    #[test]
    fn attach_selection_rejects_wrong_clock_closed_product_and_duplicate_installation() {
        let original = ProcessClock::new(ProcessEpoch(401));
        let same = original.clone();
        let wrong = ProcessClock::new(ProcessEpoch(402));
        assert!(validate_attachment_selection(false, false, true, &original, &same).is_ok());
        for (installed, closed, same_owner, clock) in [
            (true, false, true, &same),
            (false, true, true, &same),
            (false, false, false, &same),
            (false, false, true, &wrong),
        ] {
            assert!(
                validate_attachment_selection(installed, closed, same_owner, &original, clock)
                    .is_err()
            );
        }
        let same_epoch_new_origin = ProcessClock::new(original.epoch());
        if original.runtime_clock().origin() != same_epoch_new_origin.runtime_clock().origin() {
            assert!(
                validate_attachment_selection(
                    false,
                    false,
                    true,
                    &original,
                    &same_epoch_new_origin
                )
                .is_err()
            );
        }
    }

    struct DropProbe(Arc<AtomicUsize>);
    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::AcqRel);
        }
    }

    #[test]
    fn unknown_drop_preserves_original_custody_without_setting_a_completion_flag() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let closed = Arc::new(AtomicBool::new(false));
        let held = RetainedOnUnknown::new(DropProbe(Arc::clone(&dropped)), Arc::clone(&closed));
        drop(held);
        assert_eq!(dropped.load(Ordering::Acquire), 0);
        assert!(!closed.load(Ordering::Acquire));
        assert_eq!(Arc::strong_count(&dropped), 2);
    }

    #[test]
    fn unused_custody_is_released_only_after_the_observed_native_close() {
        // Local holder policy only: this fixture does not issue an actual
        // DriverPhysicalCompletion or storage-worker join receipt.
        let dropped = Arc::new(AtomicUsize::new(0));
        let closed = Arc::new(AtomicBool::new(false));
        let held = RetainedOnUnknown::new(DropProbe(Arc::clone(&dropped)), Arc::clone(&closed));
        closed.store(true, Ordering::Release);
        drop(held);
        assert_eq!(dropped.load(Ordering::Acquire), 1);
        assert_eq!(Arc::strong_count(&dropped), 1);
    }
}
