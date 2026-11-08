//! Separate corpus family joined to the original Rovent input-lock /3.
//! Records are evidence metadata and never recreate live native authority.
pub mod cpu_native;
pub mod native;
pub mod publication;
pub mod record;

pub use native::{
    ROVENT_LOCK_V3_NATIVE_RAW_DOMAIN, ROVENT_LOCK_V3_ORIGIN_DOMAIN, ROVENT_LOCK_V3_ROOT_DOMAIN,
    validate_lock_v3_root_metadata, validate_native_lifecycle_lock_v3,
};
