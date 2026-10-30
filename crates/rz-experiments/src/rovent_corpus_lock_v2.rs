//! Distinct corpus family for records scoped to an input-lock /2.
//! Decoding metadata never creates a live execution or publication authority.
pub mod native;
pub mod publication;
pub mod record;

pub use native::{
    ROVENT_LOCK_V2_NATIVE_RAW_DOMAIN, ROVENT_LOCK_V2_ORIGIN_DOMAIN, ROVENT_LOCK_V2_ROOT_DOMAIN,
    validate_lock_v2_root_metadata, validate_native_lifecycle_lock_v2,
};
