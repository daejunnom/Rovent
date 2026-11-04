//! Optional numeric input bytes for independent scalar diagnosis only.
//! No model, Bound, token, owner or runtime capability is retained.
use super::{CURRENT, Sample};
use std::cell::RefCell;
use std::ffi::OsString;
use std::io::{self, Write};

pub const INPUT_BYTES_ENV: &str = "RZ_NNUEC_REFERENCE_INPUT_MAX_BYTES";
/// A diagnostic transport bound, never an admission bound for a model or NN.
pub const MAX_INPUT_BYTES: usize = 256 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputUnavailable {
    InvalidLimit,
    NoSuccessfulScore,
    Busy,
    Allocation,
    Encoding,
    ByteLimit,
}
impl InputUnavailable {
    pub fn code(self) -> &'static str {
        match self {
            Self::InvalidLimit => "invalid_optional_input_limit",
            Self::NoSuccessfulScore => "no_successful_score_observed",
            Self::Busy => "optional_input_scope_busy",
            Self::Allocation => "optional_input_allocation_failed",
            Self::Encoding => "optional_reference_input_encoding_failed",
            Self::ByteLimit => "optional_reference_input_byte_limit",
        }
    }
}

#[derive(Debug)]
pub struct ReferenceInputSnapshot {
    pub maximum_bytes: usize,
    pub scalar: Option<Sample>,
    /// Exact Rust serialization, not a JSON re-encoding of a prior input.
    pub input_json: Option<String>,
    pub unavailable: Option<InputUnavailable>,
}
struct Capture {
    snapshot: ReferenceInputSnapshot,
    attempted: bool,
}
impl Capture {
    fn from_setting(setting: Option<OsString>) -> Option<Self> {
        let setting = setting?;
        let limit = setting.to_str().and_then(|v| v.parse::<usize>().ok());
        let valid = limit.is_some_and(|n| (1..=MAX_INPUT_BYTES).contains(&n));
        Some(Self {
            snapshot: ReferenceInputSnapshot {
                maximum_bytes: limit.unwrap_or(0),
                scalar: None,
                input_json: None,
                unavailable: Some(if valid {
                    InputUnavailable::NoSuccessfulScore
                } else {
                    InputUnavailable::InvalidLimit
                }),
            },
            attempted: !valid,
        })
    }
}
thread_local! {
    static INPUT: RefCell<Option<Capture>> = const { RefCell::new(None) };
}
pub(super) fn begin() {
    INPUT.with(|slot| {
        if let Ok(mut slot) = slot.try_borrow_mut() {
            *slot = Capture::from_setting(std::env::var_os(INPUT_BYTES_ENV));
        }
    });
}
pub(super) fn clear() {
    INPUT.with(|slot| {
        if let Ok(mut slot) = slot.try_borrow_mut() {
            *slot = None;
        }
    });
}
pub(super) fn take() -> Option<ReferenceInputSnapshot> {
    INPUT.with(|slot| {
        slot.try_borrow_mut()
            .ok()
            .and_then(|mut slot| slot.take())
            .map(|value| value.snapshot)
    })
}

/// The builder is never called without explicit opt-in, or after the first
/// successful scalar in this actual Invocation. It must only obtain the
/// already computed invocation's ReferenceInput, never call another forward.
pub(crate) fn capture<T: serde::Serialize, E>(build: impl FnOnce() -> Result<T, E>) {
    let sample = CURRENT.with(|slot| {
        slot.try_borrow().ok().and_then(|current| {
            current.as_ref().and_then(|s| {
                (s.successful_score_returns == 1)
                    .then_some(s.first)
                    .flatten()
            })
        })
    });
    let Some(sample) = sample else {
        return;
    };
    INPUT.with(|slot| {
        let Ok(mut slot) = slot.try_borrow_mut() else {
            return;
        };
        let Some(capture) = slot.as_mut() else {
            return;
        };
        if capture.attempted {
            return;
        }
        capture.attempted = true;
        capture.snapshot.scalar = Some(sample);
        let input = match build() {
            Ok(value) => value,
            Err(_) => {
                capture.snapshot.unavailable = Some(InputUnavailable::Encoding);
                return;
            }
        };
        let mut writer = BoundedBytes::new(capture.snapshot.maximum_bytes);
        if writer.failed == Some(InputUnavailable::Allocation) {
            capture.snapshot.unavailable = writer.failed;
            return;
        }
        if serde_json::to_writer(&mut writer, &input).is_err() {
            capture.snapshot.unavailable =
                Some(writer.failed.unwrap_or(InputUnavailable::Encoding));
            return;
        }
        match String::from_utf8(writer.bytes) {
            Ok(raw) => {
                capture.snapshot.input_json = Some(raw);
                capture.snapshot.unavailable = None;
            }
            Err(_) => capture.snapshot.unavailable = Some(InputUnavailable::Encoding),
        }
    });
}
struct BoundedBytes {
    bytes: Vec<u8>,
    maximum: usize,
    failed: Option<InputUnavailable>,
}
impl BoundedBytes {
    fn new(maximum: usize) -> Self {
        let mut bytes = Vec::new();
        let failed = bytes
            .try_reserve_exact(maximum)
            .err()
            .map(|_| InputUnavailable::Allocation);
        Self {
            bytes,
            maximum,
            failed,
        }
    }
}
impl Write for BoundedBytes {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.failed.is_some() {
            return Err(io::Error::other("optional input writer unavailable"));
        }
        if bytes.len() > self.maximum.saturating_sub(self.bytes.len()) {
            self.failed = Some(InputUnavailable::ByteLimit);
            return Err(io::Error::other("optional input byte limit"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Diagnostic-only capacity checks. No model or original authority is built.
    #[test]
    fn no_setting_and_invalid_limits_do_not_become_numeric_inputs() {
        assert!(Capture::from_setting(None).is_none());
        for setting in ["0", "bad", "262145"] {
            let capture = Capture::from_setting(Some(OsString::from(setting))).unwrap();
            assert_eq!(
                capture.snapshot.unavailable,
                Some(InputUnavailable::InvalidLimit)
            );
            assert!(capture.snapshot.input_json.is_none());
        }
    }
    #[test]
    fn serialized_capacity_rejects_before_exceeding_the_original_buffer() {
        let mut writer = BoundedBytes::new(2);
        writer.write_all(b"ab").unwrap();
        assert!(writer.write_all(b"c").is_err());
        assert_eq!(writer.bytes.as_slice(), b"ab");
        assert_eq!(writer.failed, Some(InputUnavailable::ByteLimit));
    }
}
