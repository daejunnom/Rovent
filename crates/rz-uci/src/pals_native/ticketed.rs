//! Additive ticket service. Encoding, native calls, acknowledgement and finish
//! run on its owner thread; the manager only exchanges bounded owned messages.
//! The historical synchronous NativeRoleModel is not the ticket authority.

#[cfg(feature = "onnx-cpu")]
#[path = "ticketed/native_bridge.rs"]
mod native_bridge;
#[path = "ticketed/service.rs"]
mod service;
#[path = "ticketed/recovery.rs"]
mod recovery;
#[cfg(feature = "onnx-cpu")]
#[path = "ticketed/virtual_bridge.rs"]
mod virtual_bridge;
#[cfg(feature = "onnx-cpu")]
pub use native_bridge::*;
pub use service::*;
pub use recovery::{TicketNativeRecoveryKey, TicketNativeRecoveryReservation, TicketRecoveryConfirmation, TicketRecoveryError, TicketRecoveryStatus, TicketUndeliveredEvent};
#[cfg(feature = "onnx-cpu")]
pub use virtual_bridge::*;

pub const TICKETED_NATIVE_SCHEMA: &str = "rz-pals-ticketed-native/1";

pub fn ticketed_native_source_digest() -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(TICKETED_NATIVE_SCHEMA);
    for (name, source) in [
        ("ticketed.rs", include_bytes!("ticketed.rs").as_slice()),
        (
            "ticketed/service.rs",
            include_bytes!("ticketed/service.rs").as_slice(),
        ),
        (
            "ticketed/recovery.rs",
            include_bytes!("ticketed/recovery.rs").as_slice(),
        ),
        (
            "ticketed/service_recovery.rs",
            include_bytes!("ticketed/service_recovery.rs").as_slice(),
        ),
        (
            "ticketed/native_bridge.rs",
            include_bytes!("ticketed/native_bridge.rs").as_slice(),
        ),
        (
            "ticketed/virtual_bridge.rs",
            include_bytes!("ticketed/virtual_bridge.rs").as_slice(),
        ),
    ] {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((source.len() as u64).to_le_bytes());
        hash.update(source);
    }
    hash.finalize().into()
}
