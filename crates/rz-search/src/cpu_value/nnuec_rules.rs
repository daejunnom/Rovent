//! Pure Rules-to-numerical-input conversion for historical verification.
//!
//! These helpers do not load weights, encode neural activations, evaluate a
//! model, or issue Bound/worker/readout/target authority. Replaying a saved
//! suffix here is additional historical verification work, not original search
//! work or a performance observation. The caller still checks a particular
//! model's possibly smaller `max_plies` before submitting its input.

use super::{NnuecError, ReferenceLine, ReferenceMove, board_features, color_name};
use rz_position::{BoardMove, Color, Position};

pub const MAX_REFERENCE_LINE_PLIES: usize = 256;

/// The existing B feature order: White perspective, then Black perspective.
/// This numerical representation intentionally does not substitute for exact
/// Rules/history identity; different histories can have equal model features.
pub fn reference_board_features(position: &Position) -> [Vec<usize>; 2] {
    [
        board_features(position, Color::White),
        board_features(position, Color::Black),
    ]
}

/// Keep the anchor's exact known Rules history in a new mutation owner and
/// validate the whole ordered suffix before publishing any numerical input.
/// No prefix is truncated, no missing history is invented, and an error cannot
/// mutate the caller's anchor. Empty rebased suffixes remain valid.
pub fn reference_line_from_rules(
    anchor: &Position,
    moves: &[BoardMove],
) -> Result<ReferenceLine, NnuecError> {
    if moves.len() > MAX_REFERENCE_LINE_PLIES {
        return Err(NnuecError::Capacity("reference line plies"));
    }
    let mut work = anchor.clone();
    for movement in moves {
        work.make_move(*movement)
            .map_err(|error| NnuecError::Rules(error.to_string()))?;
    }
    Ok(ReferenceLine {
        anchor_features: board_features(anchor, anchor.side_to_move()),
        perspective: color_name(anchor.side_to_move()).into(),
        moves: moves
            .iter()
            .copied()
            .map(ReferenceMove::from_board)
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moves(text: &str) -> Vec<BoardMove> {
        text.split_whitespace()
            .map(|value| value.parse().unwrap())
            .collect()
    }

    #[test]
    fn checked_historical_line_keeps_original_features_order_and_history() {
        let mut anchor = Position::startpos();
        for movement in moves("g1f3 g8f6 f3g1 f6g8") {
            anchor.make_move(movement).unwrap();
        }
        let before = anchor.snapshot();
        let line = reference_line_from_rules(&anchor, &moves("e2e4 e7e5")).unwrap();
        assert_eq!(line.perspective, "white");
        assert_eq!(line.anchor_features, reference_board_features(&anchor)[0]);
        assert_eq!(
            line.moves
                .iter()
                .map(|movement| (movement.from_square, movement.to_square, movement.promotion))
                .collect::<Vec<_>>(),
            vec![(12, 28, 0), (52, 36, 0)]
        );
        assert!(anchor.snapshot().same_state(&before));
        assert_eq!(anchor.known_history_len(), 5);
        assert!(anchor.matches_snapshot(&before));
    }

    #[test]
    fn black_anchor_and_all_four_promotion_tokens_use_existing_mapping() {
        for (suffix, expected) in [("q", 1), ("r", 2), ("b", 3), ("n", 4)] {
            let anchor = Position::from_fen("4k3/8/8/8/8/8/1p6/4K3 b - - 0 1").unwrap();
            let line =
                reference_line_from_rules(&anchor, &moves(&format!("b2b1{suffix}"))).unwrap();
            assert_eq!(line.perspective, "black");
            assert_eq!(line.anchor_features, reference_board_features(&anchor)[1]);
            let token = line.moves[0];
            assert_eq!((token.from_square, token.to_square), (9, 1));
            assert_eq!(token.promotion, expected);
        }
    }

    #[test]
    fn invalid_suffix_preserves_original_anchor_and_empty_suffix_is_supported() {
        let anchor = Position::startpos();
        let before = anchor.snapshot();
        assert!(matches!(
            reference_line_from_rules(&anchor, &moves("e2e4 e2e3")),
            Err(NnuecError::Rules(_))
        ));
        assert!(anchor.matches_snapshot(&before));
        let empty = reference_line_from_rules(&anchor, &[]).unwrap();
        assert!(empty.moves.is_empty());
        assert_eq!(empty.anchor_features, reference_board_features(&anchor)[0]);
    }

    #[test]
    fn suffix_capacity_is_rejected_before_rules_replay_without_truncation() {
        let anchor = Position::startpos();
        let too_long = vec!["e2e4".parse().unwrap(); MAX_REFERENCE_LINE_PLIES + 1];
        assert_eq!(
            reference_line_from_rules(&anchor, &too_long).unwrap_err(),
            NnuecError::Capacity("reference line plies")
        );
    }
}
