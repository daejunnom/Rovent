//! 규칙, 평가, 탐색과 런타임이 공유하는 작은 값 타입과 실패 계약.
//!
//! ID의 발급과 digest 계산은 실제 소유자의 책임이다. 발급자는 `checked_add`로
//! sequence/generation overflow를 거부하고, 같은 epoch 안에서 발급한 값을
//! 재사용하지 않는다. 이 모듈은 상태 digest나 가상의 소유자를 만들어내지 않는다.

use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// 초안의 minor revision도 자동 호환으로 취급하지 않는다.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SchemaVersion {
    pub major: u16,
    pub minor: u16,
}

pub const CONTRACT_REVISION: SchemaVersion = SchemaVersion { major: 0, minor: 1 };

impl SchemaVersion {
    pub fn validate(self) -> Result<(), ContractError> {
        if self == CONTRACT_REVISION {
            Ok(())
        } else {
            Err(ContractError::new(
                ErrorCode::UnsupportedContract,
                Stage::Contract,
                "unsupported contract revision",
            ))
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProcessEpoch(pub u64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct OwnerId(pub u64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct GameGeneration(pub u64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RootGeneration(pub u64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StateRevision(pub u64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SlotGeneration(pub u64);

/// epoch별 단일 Instant origin 이후의 나노초. checked u64 변환을 사용한다.
/// 서로 다른 ClockDomain의 값을 비교하거나 UTC/프로세스 밖 Instant를 복사하지 않는다.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MonotonicTick(pub u64);

/// 발급자가 실제 semantic 입력으로 계산한 digest. 계산 알고리즘은 manifest에 잠근다.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Digest(pub [u8; 32]);

/// 논리 평가 subscriber의 ID. 재시도에는 새 ID를 발급한다.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RequestId {
    pub epoch: ProcessEpoch,
    pub sequence: u64,
}

impl RequestId {
    pub const fn new(epoch: ProcessEpoch, sequence: u64) -> Self {
        Self { epoch, sequence }
    }
}

/// 탐색 traversal의 ID. 같은 selection을 중복 backup하지 않는다.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SelectionId {
    pub epoch: ProcessEpoch,
    pub sequence: u64,
}

impl SelectionId {
    pub const fn new(epoch: ProcessEpoch, sequence: u64) -> Self {
        Self { epoch, sequence }
    }
}

/// 물리 backend 실행의 ID. 여러 논리 요청이 하나의 실행을 공유할 수 있다.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExecutionId {
    pub epoch: ProcessEpoch,
    pub sequence: u64,
}

impl ExecutionId {
    pub const fn new(epoch: ProcessEpoch, sequence: u64) -> Self {
        Self { epoch, sequence }
    }
}

/// 규칙 판정용 상태 키. 신경망 평가 입력 키와 서로 대체할 수 없다.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RuleKey(pub Digest);

/// 실제 이력·모델 입력을 반영하는 완전 평가 입력 키.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EvalInputKey(pub Digest);

/// 합법 수 배열의 의미와 순서를 식별한다.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LegalOrderIdentity(pub Digest);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelHandle {
    pub owner: OwnerId,
    pub slot: u64,
    pub generation: SlotGeneration,
    pub manifest: Digest,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EncodingHandle {
    pub owner: OwnerId,
    pub slot: u64,
    pub generation: SlotGeneration,
    pub manifest: Digest,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StateIdentity {
    pub owner: OwnerId,
    pub revision: StateRevision,
    pub semantic: Digest,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Color {
    White,
    Black,
}

impl Color {
    pub const fn opposite(self) -> Self {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum HistoryCompleteness {
    Complete,
    UnknownPrefix,
}

/// 0..=63의 표준 체스 칸. 좌표 방향은 encoding manifest에서 별도로 명시한다.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Square(u8);

impl Square {
    pub fn try_new(index: u8) -> Result<Self, ContractError> {
        if index < 64 {
            Ok(Self(index))
        } else {
            Err(ContractError::new(
                ErrorCode::InvalidInput,
                Stage::Contract,
                "square index must be less than 64",
            ))
        }
    }

    pub const fn index(self) -> u8 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Promotion {
    Queen,
    Rook,
    Bishop,
    Knight,
}

/// 기하학적인 수 표현. 이 타입의 생성 성공은 실제 상태에서의 합법성을 뜻하지 않는다.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Move {
    pub from: Square,
    pub to: Square,
    pub promotion: Option<Promotion>,
}

impl Move {
    pub fn new(
        from: Square,
        to: Square,
        promotion: Option<Promotion>,
    ) -> Result<Self, ContractError> {
        if from == to {
            Err(ContractError::new(
                ErrorCode::InvalidInput,
                Stage::Contract,
                "move source and destination must differ",
            ))
        } else {
            Ok(Self {
                from,
                to,
                promotion,
            })
        }
    }
}

/// side-to-move 관점의 W/D/L 확률. 검증은 값을 clip하거나 정규화하지 않는다.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wdl {
    win: f32,
    draw: f32,
    loss: f32,
}

impl Wdl {
    pub fn try_new(win: f32, draw: f32, loss: f32, tolerance: f32) -> Result<Self, ContractError> {
        if !tolerance.is_finite() || !(0.0..=0.01).contains(&tolerance) {
            return Err(ContractError::new(
                ErrorCode::InvalidInput,
                Stage::Contract,
                "WDL tolerance must be finite and between 0 and 0.01",
            ));
        }
        if [win, draw, loss]
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(ContractError::new(
                ErrorCode::NumericalFailure,
                Stage::Output,
                "WDL probabilities must be finite and between 0 and 1",
            ));
        }
        // 검증 자체의 f32 덧셈 반올림을 피하도록 f64에서 합계를 계산한다.
        let sum = f64::from(win) + f64::from(draw) + f64::from(loss);
        if (sum - 1.0).abs() > f64::from(tolerance) {
            return Err(ContractError::new(
                ErrorCode::NumericalFailure,
                Stage::Output,
                "WDL probabilities do not sum to 1 within tolerance",
            ));
        }
        Ok(Self { win, draw, loss })
    }

    pub const fn probabilities(self) -> [f32; 3] {
        [self.win, self.draw, self.loss]
    }

    /// 검증된 값을 f64로 넓힌 뒤 합이 1인 복사본을 명시적으로 만든다.
    /// 원래 head와 관점은 유지한다. try_new의 허용 오차 밖 값은 복구하지 않는다.
    pub fn normalized(self) -> [f64; 3] {
        let values = self.probabilities().map(f64::from);
        let sum = values.iter().sum::<f64>();
        values.map(|value| value / sum)
    }

    pub const fn flipped(self) -> Self {
        Self {
            win: self.loss,
            draw: self.draw,
            loss: self.win,
        }
    }

    pub fn value(self) -> f32 {
        self.win - self.loss
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ClockDomain(pub ProcessEpoch);

/// `now == at`도 만료다. tick은 반드시 같은 단조 시계에서 얻는다.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Deadline {
    pub clock: ClockDomain,
    pub at: MonotonicTick,
}

impl Deadline {
    pub fn accepts(self, clock: ClockDomain, now: MonotonicTick) -> Result<(), ContractError> {
        if self.clock != clock {
            return Err(ContractError::new(
                ErrorCode::InvalidInput,
                Stage::Admission,
                "deadline clock domain mismatch",
            ));
        }
        if now >= self.at {
            return Err(ContractError::new(
                ErrorCode::Expired,
                Stage::Admission,
                "deadline has expired",
            ));
        }
        Ok(())
    }
}

/// clone은 같은 취소 권한을 공유한다. 물리 backend buffer 해제를 보장하지 않는다.
#[derive(Clone)]
pub struct CancelToken {
    canceled: Arc<AtomicBool>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self {
            canceled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn cancel(&self) {
        self.canceled.store(true, Ordering::Release);
    }

    pub fn is_canceled(&self) -> bool {
        self.canceled.load(Ordering::Acquire)
    }

    /// 원래 취소 권한의 동일 소유 여부만 대조한다. 같은 취소 상태나 숫자 ID를
    /// 권한으로 취급하지 않으며 물리 완료·buffer 해제를 입증하지 않는다.
    pub fn shares_cancellation_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.canceled, &other.canceled)
    }
}

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for CancelToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CancelToken")
            .field("canceled", &self.is_canceled())
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PrecisionProfile {
    Fp32,
    Fp16,
    Bf16,
    Quantized(Digest),
}

/// fresh 완전 계산의 유한 단계 예산. warm-start/latent는 초기 계약에 포함하지 않는다.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ComputeBudget {
    pub min_steps: u32,
    pub max_steps: u32,
    pub require_full: bool,
}

impl ComputeBudget {
    pub fn validate(self) -> Result<(), ContractError> {
        if self.min_steps == 0 || self.max_steps == 0 || self.min_steps > self.max_steps {
            return Err(ContractError::new(
                ErrorCode::InvalidInput,
                Stage::Contract,
                "compute budget must have nonzero min_steps <= max_steps",
            ));
        }
        Ok(())
    }
}

/// host/device/pinned의 서로 다른 예약 한도를 보존한다.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ByteBudget {
    pub host: u64,
    pub device: u64,
    pub pinned: u64,
}

impl ByteBudget {
    pub fn checked_add(self, other: Self) -> Result<Self, ContractError> {
        let overflow = || {
            ContractError::new(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "byte budget addition overflow",
            )
        };
        Ok(Self {
            host: self.host.checked_add(other.host).ok_or_else(overflow)?,
            device: self.device.checked_add(other.device).ok_or_else(overflow)?,
            pinned: self.pinned.checked_add(other.pinned).ok_or_else(overflow)?,
        })
    }

    pub const fn fits(self, limit: Self) -> bool {
        self.host <= limit.host && self.device <= limit.device && self.pinned <= limit.pinned
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ErrorCode {
    InvalidInput,
    UnsupportedContract,
    BackendUnavailable,
    ResourceExhausted,
    NumericalFailure,
    IdentityMismatch,
    BackendFailure,
    Canceled,
    Expired,
    Stale,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Stage {
    Contract,
    Admission,
    Backend,
    Output,
    Backup,
}

/// detail에는 비밀·사용자 경로·backend의 무제한 외부 문자열을 담지 않는다.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ContractError {
    pub code: ErrorCode,
    pub stage: Stage,
    pub detail: &'static str,
}

impl ContractError {
    pub const fn new(code: ErrorCode, stage: Stage, detail: &'static str) -> Self {
        Self {
            code,
            stage,
            detail,
        }
    }
}

impl fmt::Display for ContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:?}/{:?}: {}",
            self.stage, self.code, self.detail
        )
    }
}

impl Error for ContractError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wdl_rejects_nonfinite_invalid_ranges_and_normalization() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
            assert_eq!(
                Wdl::try_new(invalid, 0.0, 0.0, 0.001).unwrap_err().code,
                ErrorCode::NumericalFailure,
            );
        }
        for tolerance in [f32::NAN, f32::INFINITY, -0.001, 0.011] {
            assert_eq!(
                Wdl::try_new(1.0, 0.0, 0.0, tolerance).unwrap_err().code,
                ErrorCode::InvalidInput,
            );
        }
        assert!(Wdl::try_new(0.4, 0.3, 0.2, 0.01).is_err());
        let accepted = Wdl::try_new(0.5, 0.25, 0.249, 0.002).unwrap();
        assert_eq!(accepted.probabilities(), [0.5, 0.25, 0.249]);
        assert_eq!(accepted.flipped().probabilities(), [0.249, 0.25, 0.5]);
        assert_eq!(accepted.flipped().flipped(), accepted);
        assert_eq!(accepted.value(), 0.5 - 0.249);
        assert!(Wdl::try_new(1.0, 0.0, 0.0, 0.0).is_ok());
    }

    #[test]
    fn moves_validate_geometry_and_preserve_all_promotions() {
        assert!(Square::try_new(64).is_err());
        assert!(Square::try_new(u8::MAX).is_err());
        let from = Square::try_new(48).unwrap();
        let to = Square::try_new(56).unwrap();
        assert_eq!(to.index(), 56);
        assert!(Move::new(from, from, None).is_err());
        for promotion in [
            Promotion::Queen,
            Promotion::Rook,
            Promotion::Bishop,
            Promotion::Knight,
        ] {
            let movement = Move::new(from, to, Some(promotion)).unwrap();
            assert_eq!(movement.promotion, Some(promotion));
            assert_eq!(movement.from, from);
            assert_eq!(movement.to, to);
        }
    }

    #[test]
    fn deadline_is_strict_and_requires_the_same_clock_domain() {
        let clock = ClockDomain(ProcessEpoch(7));
        let deadline = Deadline {
            clock,
            at: MonotonicTick(10),
        };
        assert!(deadline.accepts(clock, MonotonicTick(9)).is_ok());
        for now in [10, 11] {
            assert_eq!(
                deadline
                    .accepts(clock, MonotonicTick(now))
                    .unwrap_err()
                    .code,
                ErrorCode::Expired,
            );
        }
        assert_eq!(
            deadline
                .accepts(ClockDomain(ProcessEpoch(8)), MonotonicTick(1))
                .unwrap_err()
                .code,
            ErrorCode::InvalidInput,
        );
    }

    #[test]
    fn cancellation_is_shared_and_debug_exposes_only_state() {
        let original = CancelToken::default();
        let subscriber = original.clone();
        let unrelated = CancelToken::new();
        assert!(original.shares_cancellation_with(&subscriber));
        assert!(!original.shares_cancellation_with(&unrelated));
        assert!(!subscriber.is_canceled());
        assert_eq!(format!("{original:?}"), "CancelToken { canceled: false }");
        let worker = std::thread::spawn(move || subscriber.cancel());
        worker.join().unwrap();
        assert!(original.is_canceled());
        unrelated.cancel();
        assert!(!original.shares_cancellation_with(&unrelated));
        original.cancel();
        assert!(original.is_canceled());
        assert_eq!(format!("{original:?}"), "CancelToken { canceled: true }");
        assert!(!CancelToken::new().is_canceled());
    }

    #[test]
    fn byte_budgets_check_each_dimension_and_addition_overflow() {
        let base = ByteBudget {
            host: 1,
            device: 2,
            pinned: 3,
        };
        let total = base.checked_add(base).unwrap();
        assert_eq!(
            total,
            ByteBudget {
                host: 2,
                device: 4,
                pinned: 6,
            },
        );
        assert!(base.fits(total));
        for overflow in [
            ByteBudget {
                host: u64::MAX,
                device: 0,
                pinned: 0,
            },
            ByteBudget {
                host: 0,
                device: u64::MAX,
                pinned: 0,
            },
            ByteBudget {
                host: 0,
                device: 0,
                pinned: u64::MAX,
            },
        ] {
            assert!(!overflow.fits(base));
            assert_eq!(
                overflow.checked_add(base).unwrap_err().code,
                ErrorCode::ResourceExhausted,
            );
        }
    }

    #[test]
    fn revision_is_exact_and_compute_budget_is_finite() {
        assert!(CONTRACT_REVISION.validate().is_ok());
        for revision in [
            SchemaVersion { major: 0, minor: 0 },
            SchemaVersion { major: 0, minor: 2 },
            SchemaVersion { major: 1, minor: 1 },
        ] {
            assert_eq!(
                revision.validate().unwrap_err().code,
                ErrorCode::UnsupportedContract,
            );
        }
        for (min_steps, max_steps) in [(0, 0), (0, 1), (1, 0), (2, 1)] {
            assert!(ComputeBudget {
                min_steps,
                max_steps,
                require_full: true,
            }
            .validate()
            .is_err());
        }
        assert!(ComputeBudget {
            min_steps: 1,
            max_steps: 1,
            require_full: true,
        }
        .validate()
        .is_ok());
    }
}
