//! Drain the original /2 product owners on startup rejection.
//!
//! These adapters deliberately retain the complete rejection until native
//! physical shutdown is confirmed. A failed or unknown drain quarantines the
//! original owners instead of manufacturing a replacement completion.

use super::super::external_session::RoventExternalProductStartupOwnersV2;
use super::super::product::{
    RoventExternalPcrV2RejectedOwners, RoventExternalPcrV2StartupRejection,
    RoventOwnNnuecV2StartupRejection,
};
use super::*;

pub(super) fn reject_own_nnuec_product_v2(
    mut failure: RoventOwnNnuecV2StartupRejection,
    until: Instant,
    diagnostic: &NativeStartupDiagnostic,
) -> RoventLaunchError {
    let detail = error("ProductAssembly", &failure.failure);
    diagnostic.emit(
        NativeStartupDiagnosticStage::ProductDrain,
        NativeStartupDiagnosticPhase::Entry,
    );
    let confirmed = drain_native(&mut failure.rejected.prepared.service, until);
    diagnostic.emit(
        NativeStartupDiagnosticStage::ProductDrain,
        if confirmed {
            NativeStartupDiagnosticPhase::Exit
        } else {
            NativeStartupDiagnosticPhase::Error
        },
    );
    if !confirmed {
        let _quarantine = Box::leak(Box::new(failure));
        return error("PhysicalCompletionUnknown", detail);
    }
    detail
}

pub(super) fn reject_external_pcr_product_v2(
    mut failure: RoventExternalPcrV2StartupRejection,
    until: Instant,
    diagnostic: &NativeStartupDiagnostic,
) -> RoventLaunchError {
    let detail = error("ProductAssembly", &failure.failure);
    diagnostic.emit(
        NativeStartupDiagnosticStage::ProductDrain,
        NativeStartupDiagnosticPhase::Entry,
    );
    let confirmed = match &mut failure.owners {
        RoventExternalPcrV2RejectedOwners::Preflight { prepared, .. } => {
            drain_native(&mut prepared.service, until)
        }
        RoventExternalPcrV2RejectedOwners::Session { rejected, .. } => match &mut rejected.owners {
            RoventExternalProductStartupOwnersV2::Prepared { prepared, .. }
            | RoventExternalProductStartupOwnersV2::SourcePreparation { prepared, .. }
            | RoventExternalProductStartupOwnersV2::PreparedSource { prepared, .. } => {
                drain_native(&mut prepared.service, until)
            }
            RoventExternalProductStartupOwnersV2::Session(rejected) => {
                drain_native(&mut rejected.sources.native, until)
            }
            RoventExternalProductStartupOwnersV2::Driver(driver) => {
                driver.finish_session(until).is_ok_and(|completion| {
                    completion == Some(crate::search_driver::DriverPhysicalCompletion::Confirmed)
                })
            }
        },
    };
    diagnostic.emit(
        NativeStartupDiagnosticStage::ProductDrain,
        if confirmed {
            NativeStartupDiagnosticPhase::Exit
        } else {
            NativeStartupDiagnosticPhase::Error
        },
    );
    if !confirmed {
        let _quarantine = Box::leak(Box::new(failure));
        return error("PhysicalCompletionUnknown", detail);
    }
    detail
}
