//! Fixed-size raw observation on the actual private worker thread.
//! No owner/Bound/model/token is retained. A sample records an already returned
//! numeric result; no new forward, score, hash, clock, allocation or I/O occurs.
use super::review::Provenance;
use rz_contracts::Digest;
use std::cell::{Cell, RefCell};

#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub binding: Digest,
    pub full_view: [u8; 32],
    pub view_seal: [u8; 32],
    pub record_scope: [u8; 32],
    pub generation: u64,
    pub raw_value_bits: u32,
    pub scaled_value_bits: u32,
    pub returned_frontier_score: i32,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct Summary {
    pub successful_score_returns: u64,
    pub first: Option<Sample>,
    pub last: Option<Sample>,
    pub incomplete: bool,
}
thread_local! {
    static CURRENT: RefCell<Option<Summary>> = const { RefCell::new(None) };
    static INCOMPLETE: Cell<bool> = const { Cell::new(false) };
}
/// One explicit diagnostic scope around the original engine invocation.
/// Its Drop always clears the thread-local view, including on unwinding.
pub(crate) struct Invocation {
    owns_scope: bool,
    // The guard owns THIS thread's TLS. Zero-size marker only: no Rc allocation.
    // It must never move/share across workers before finish or Drop.
    _thread_local_owner: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Invocation {
    pub(crate) fn begin() -> Self {
        let owns_scope = CURRENT.with(|slot| {
            let Ok(mut current) = slot.try_borrow_mut() else {
                INCOMPLETE.set(true);
                return false;
            };
            if let Some(current) = current.as_mut() {
                current.incomplete = true;
                return false;
            }
            INCOMPLETE.set(false);
            *current = Some(Summary::default());
            true
        });
        Self {
            owns_scope,
            _thread_local_owner: std::marker::PhantomData,
        }
    }
    pub(crate) fn finish(self) -> Option<Summary> {
        if !self.owns_scope {
            return None;
        }
        CURRENT
            .with(|slot| {
                slot.try_borrow_mut()
                    .ok()
                    .and_then(|mut value| value.take())
            })
            .map(|mut value| {
                value.incomplete |= INCOMPLETE.get();
                value
            })
    }
}
impl Drop for Invocation {
    fn drop(&mut self) {
        if !self.owns_scope {
            return;
        }
        CURRENT.with(|slot| {
            if let Ok(mut value) = slot.try_borrow_mut() {
                *value = None;
            }
        });
        INCOMPLETE.set(false);
    }
}
pub(super) fn record(
    binding: Digest,
    provenance: &Provenance,
    raw_value_bits: u32,
    scaled_value_bits: u32,
    returned_frontier_score: i32,
) {
    CURRENT.with(|slot| {
        let Ok(mut current) = slot.try_borrow_mut() else {
            INCOMPLETE.set(true);
            return;
        };
        let Some(current) = current.as_mut() else {
            return;
        };
        let sample = Sample {
            binding,
            full_view: provenance.full_view_digest,
            view_seal: provenance.view_seal,
            record_scope: provenance.record_scope_digest,
            generation: provenance.generation,
            raw_value_bits,
            scaled_value_bits,
            returned_frontier_score,
        };
        if let Some(count) = current.successful_score_returns.checked_add(1) {
            current.successful_score_returns = count;
        } else {
            current.incomplete = true;
        }
        if current.first.is_none() {
            current.first = Some(sample);
        }
        current.last = Some(sample);
    });
}
#[derive(Clone, Copy, Debug)]
pub struct SliceObservation {
    pub scalar: Option<Summary>,
    pub resume_command: bool,
    pub prior_token_present: bool,
    pub before: Option<crate::cpu::CpuPausedStackSnapshot>,
    pub after: Option<crate::cpu::CpuPausedStackSnapshot>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_value::{CpuTrainingState, CpuValueIdentity};
    use crate::pals::store::StateId;

    /// Authored metadata for observation-ownership tests only. No model,
    /// weights, Rules replay, scalar forward or execution authority is built.
    fn provenance() -> Provenance {
        let model = CpuValueIdentity {
            semantics: "diagnostic-unit-declared-identity-only".into(),
            weights_sha256: Some("01".repeat(32)),
            training: CpuTrainingState::Untrained,
        };
        model.validate().unwrap();
        Provenance {
            root: StateId(1),
            generation: 2,
            sequence: 3,
            full_view_digest: [4; 32],
            view_seal: [5; 32],
            record_scope_digest: [6; 32],
            policy_digest: [7; 32],
            model,
        }
    }
    fn assert_sample(
        value: Sample,
        binding: Digest,
        provenance: &Provenance,
        raw: u32,
        scaled: u32,
        returned: i32,
    ) {
        assert_eq!(value.binding, binding);
        assert_eq!(value.full_view, provenance.full_view_digest);
        assert_eq!(value.view_seal, provenance.view_seal);
        assert_eq!(value.record_scope, provenance.record_scope_digest);
        assert_eq!(value.generation, provenance.generation);
        assert_eq!(value.raw_value_bits, raw);
        assert_eq!(value.scaled_value_bits, scaled);
        assert_eq!(value.returned_frontier_score, returned);
    }
    #[test]
    fn zero_scores_are_zero_count_and_no_sample() {
        let value = Invocation::begin().finish().unwrap();
        assert_eq!(value.successful_score_returns, 0);
        assert!(value.first.is_none());
        assert!(value.last.is_none());
        assert!(!value.incomplete);
        CURRENT.with(|current| assert!(current.borrow().is_none()));
    }
    #[test]
    fn first_last_count_preserve_the_exact_given_bits_and_returned_value() {
        let provenance = provenance();
        let binding = Digest([8; 32]);
        let owner = Invocation::begin();
        record(
            binding,
            &provenance,
            (-0.0f32).to_bits(),
            (-0.0f32).to_bits(),
            0,
        );
        record(binding, &provenance, 1.25f32.to_bits(), 2.5f32.to_bits(), 3);
        record(
            binding,
            &provenance,
            (-1.25f32).to_bits(),
            (-2.5f32).to_bits(),
            -3,
        );
        let value = owner.finish().unwrap();
        assert_eq!(value.successful_score_returns, 3);
        assert!(!value.incomplete);
        assert_sample(
            value.first.unwrap(),
            binding,
            &provenance,
            (-0.0f32).to_bits(),
            (-0.0f32).to_bits(),
            0,
        );
        assert_sample(
            value.last.unwrap(),
            binding,
            &provenance,
            (-1.25f32).to_bits(),
            (-2.5f32).to_bits(),
            -3,
        );
    }
    #[test]
    fn nested_scope_keeps_the_outer_custody_and_marks_it_incomplete() {
        let provenance = provenance();
        let binding = Digest([9; 32]);
        let outer = Invocation::begin();
        record(binding, &provenance, 1.0f32.to_bits(), 1.0f32.to_bits(), 1);
        let inner = Invocation::begin();
        assert!(inner.finish().is_none());
        CURRENT.with(|current| assert!(current.borrow().is_some()));
        record(binding, &provenance, 2.0f32.to_bits(), 2.0f32.to_bits(), 2);
        let value = outer.finish().unwrap();
        assert_eq!(value.successful_score_returns, 2);
        assert!(value.incomplete);
        assert_sample(
            value.first.unwrap(),
            binding,
            &provenance,
            1.0f32.to_bits(),
            1.0f32.to_bits(),
            1,
        );
        assert_sample(
            value.last.unwrap(),
            binding,
            &provenance,
            2.0f32.to_bits(),
            2.0f32.to_bits(),
            2,
        );
    }
    #[test]
    fn unwind_drop_clears_the_original_thread_scope_before_the_next_invocation() {
        let outcome = std::panic::catch_unwind(|| {
            let _owner = Invocation::begin();
            record(
                Digest([10; 32]),
                &provenance(),
                3.0f32.to_bits(),
                3.0f32.to_bits(),
                3,
            );
            panic!("authored diagnostic ownership unwind");
        });
        assert!(outcome.is_err());
        let next = Invocation::begin().finish().unwrap();
        assert_eq!(next.successful_score_returns, 0);
        assert!(next.first.is_none());
        assert!(next.last.is_none());
        assert!(!next.incomplete);
    }
    #[test]
    fn count_overflow_stays_incomplete_without_erasing_the_original_samples() {
        let provenance = provenance();
        let binding = Digest([11; 32]);
        let owner = Invocation::begin();
        record(binding, &provenance, 4.0f32.to_bits(), 4.0f32.to_bits(), 4);
        CURRENT.with(|current| {
            current
                .borrow_mut()
                .as_mut()
                .unwrap()
                .successful_score_returns = u64::MAX;
        });
        record(binding, &provenance, 5.0f32.to_bits(), 5.0f32.to_bits(), 5);
        let value = owner.finish().unwrap();
        assert_eq!(value.successful_score_returns, u64::MAX);
        assert!(value.incomplete);
        assert_sample(
            value.first.unwrap(),
            binding,
            &provenance,
            4.0f32.to_bits(),
            4.0f32.to_bits(),
            4,
        );
        assert_sample(
            value.last.unwrap(),
            binding,
            &provenance,
            5.0f32.to_bits(),
            5.0f32.to_bits(),
            5,
        );
    }
}
