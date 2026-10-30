use rz_uci::{Command, GoLimits, ParseError, ParserLimits, PositionBase, PositionSpec, parse};
use std::num::NonZeroU16;

fn parse_default(line: &str) -> Result<Command, ParseError> {
    parse(line, ParserLimits::default())
}

#[test]
fn protocol_commands_accept_whitespace_and_one_physical_line_ending() {
    for (line, expected) in [
        (" uci\r\n", Command::Uci),
        ("\tisready\t", Command::IsReady),
        ("stop\n", Command::Stop),
        ("quit", Command::Quit),
        ("ucinewgame", Command::NewGame),
    ] {
        assert_eq!(parse_default(line), Ok(expected), "{line:?}");
    }
}

#[test]
fn unexpected_protocol_arguments_and_empty_input_are_rejected() {
    for line in ["uci x", "isready x", "stop x", "quit x", "ucinewgame x"] {
        assert!(
            matches!(parse_default(line), Err(ParseError::Malformed(_))),
            "{line:?}"
        );
    }
    assert_eq!(parse_default(" \t\r\n"), Err(ParseError::Empty));
}

#[test]
fn multiline_non_ascii_and_control_bytes_are_rejected() {
    for line in [
        "uci\nquit",
        "uci\rquit",
        "uci\0",
        "uci\u{000b}",
        "uci\u{000c}",
        "uci\u{007f}",
        "한글",
    ] {
        assert_eq!(
            parse_default(line),
            Err(ParseError::InvalidText),
            "{line:?}"
        );
    }
}

#[test]
fn unsupported_modes_and_unknown_commands_remain_distinct() {
    assert_eq!(parse_default("ponderhit"), Ok(Command::PonderHit));
    assert_eq!(
        parse_default("go ponder"),
        Ok(Command::Go(GoLimits {
            ponder: true,
            ..GoLimits::default()
        }))
    );
    assert_eq!(
        parse_default("go unsupported 3"),
        Err(ParseError::Unsupported("unsupported".into()))
    );
    assert_eq!(
        parse_default("debug on"),
        Ok(Command::Unknown("debug".into()))
    );
}

#[test]
fn go_depth_and_mate_keep_distinct_chess_units_alone_and_with_other_limits() {
    for (line, depth_plies, mate_moves) in [
        ("go depth 3", NonZeroU16::new(3), None),
        ("go mate 2", None, NonZeroU16::new(2)),
        ("go depth 6 mate 3", NonZeroU16::new(6), NonZeroU16::new(3)),
        ("go depth 65535", NonZeroU16::new(u16::MAX), None),
        ("go mate 32767", None, NonZeroU16::new(32767)),
    ] {
        assert_eq!(
            parse_default(line),
            Ok(Command::Go(GoLimits {
                depth_plies,
                mate_moves,
                ..GoLimits::default()
            })),
            "{line:?}"
        );
    }
    assert_eq!(
        parse_default("go mate 3 movetime 50 nodes 100 depth 2"),
        Ok(Command::Go(GoLimits {
            depth_plies: NonZeroU16::new(2),
            mate_moves: NonZeroU16::new(3),
            movetime_ms: Some(50),
            nodes: Some(100),
            ..GoLimits::default()
        }))
    );
}

#[test]
fn infinite_and_ponder_preserve_finite_objectives_for_separate_output_ownership() {
    for (mode, infinite, ponder) in [("infinite", true, false), ("ponder", false, true)] {
        let expected = GoLimits {
            searchmoves: Some(vec!["e2e4".into()]),
            depth_plies: NonZeroU16::new(6),
            mate_moves: NonZeroU16::new(3),
            nodes: Some(100),
            infinite,
            ponder,
            ..GoLimits::default()
        };
        for line in [
            format!("go {mode} depth 6 mate 3 nodes 100 searchmoves e2e4"),
            format!("go searchmoves e2e4 depth 6 mate 3 nodes 100 {mode}"),
        ] {
            assert_eq!(
                parse_default(&line),
                Ok(Command::Go(expected.clone())),
                "{line:?}"
            );
        }
    }
    assert_eq!(
        parse_default("go infinite nodes 1"),
        Ok(Command::Go(GoLimits {
            infinite: true,
            nodes: Some(1),
            ..GoLimits::default()
        }))
    );
}

#[test]
fn searchmoves_preserves_received_moves_and_accepts_keywords_before_or_after_the_list() {
    let expected = GoLimits {
        searchmoves: Some(vec!["e2e4".into(), "d2d4".into(), "e2e4".into()]),
        depth_plies: NonZeroU16::new(4),
        mate_moves: NonZeroU16::new(2),
        nodes: Some(8),
        ..GoLimits::default()
    };
    for line in [
        "go searchmoves e2e4 d2d4 e2e4 depth 4 mate 2 nodes 8",
        "go depth 4 mate 2 nodes 8 searchmoves e2e4 d2d4 e2e4",
        "go mate 2 searchmoves e2e4 d2d4 e2e4 nodes 8 depth 4",
    ] {
        assert_eq!(
            parse_default(line),
            Ok(Command::Go(expected.clone())),
            "{line:?}"
        );
    }
    for suffix in ["q", "r", "b", "n"] {
        let movement = format!("a7a8{suffix}");
        assert_eq!(
            parse_default(&format!("go searchmoves {movement} depth 1")),
            Ok(Command::Go(GoLimits {
                searchmoves: Some(vec![movement]),
                depth_plies: NonZeroU16::new(1),
                ..GoLimits::default()
            }))
        );
    }
    // Syntax alone does not establish root legality. Even duplicate moves are
    // preserved here for Rules to validate and canonicalize as a single domain.
    assert!(parse_default("go depth 1 searchmoves a1a8 e1g1 e5d6").is_ok());
}

#[test]
fn searchmoves_stops_at_each_supported_go_keyword() {
    for tail in [
        "depth 1",
        "mate 1",
        "movetime 0",
        "wtime 1 btime 2",
        "btime 2 wtime 1",
        "winc 1 wtime 2 btime 3",
        "binc 1 wtime 2 btime 3",
        "movestogo 1 wtime 2 btime 3",
        "nodes 1",
        "infinite",
        "ponder",
    ] {
        let line = format!("go searchmoves e2e4 {tail}");
        let Ok(Command::Go(limits)) = parse_default(&line) else {
            panic!("expected valid go: {line:?}");
        };
        assert_eq!(limits.searchmoves, Some(vec!["e2e4".into()]), "{line:?}");
    }
}

#[test]
fn searchmoves_rejects_empty_malformed_repeated_and_unbounded_lists() {
    for (line, reason) in [
        ("go searchmoves", "searchmoves requires a nonempty list"),
        (
            "go searchmoves depth 1",
            "searchmoves requires a nonempty list",
        ),
        (
            "go depth 1 searchmoves e2e4 searchmoves d2d4",
            "duplicate go field",
        ),
        (
            "go depth 1 searchmoves e2e4 unexpected",
            "searchmoves requires from/to and optional q/r/b/n promotion",
        ),
    ] {
        assert_eq!(
            parse_default(line),
            Err(ParseError::Malformed(reason)),
            "{line:?}"
        );
    }
    for movement in ["0000", "e2e2", "e9e4", "e2e4qz", "a7a8k", "a7a8Q"] {
        assert_eq!(
            parse_default(&format!("go depth 1 searchmoves {movement}")),
            Err(ParseError::Malformed(
                "searchmoves requires from/to and optional q/r/b/n promotion"
            )),
            "{movement:?}"
        );
    }
    assert_eq!(
        parse_default("go searchmoves e2e4"),
        Ok(Command::Go(GoLimits {
            searchmoves: Some(vec!["e2e4".into()]),
            ..GoLimits::default()
        }))
    );
    let bounded = ParserLimits {
        max_legal_moves: 2,
        max_moves: 1,
        ..ParserLimits::default()
    };
    assert!(parse("go depth 1 searchmoves e2e4 d2d4", bounded).is_ok());
    assert_eq!(
        parse("go depth 1 searchmoves e2e4 d2d4 g1f3", bounded),
        Err(ParseError::Limit("searchmoves"))
    );
    let no_moves = ParserLimits {
        max_legal_moves: 0,
        ..ParserLimits::default()
    };
    assert_eq!(
        parse("go depth 1 searchmoves e2e4", no_moves),
        Err(ParseError::Limit("searchmoves"))
    );
}

#[test]
fn depth_and_mate_reject_zero_overflow_bad_decimals_and_duplicate_fields() {
    for value in ["32768", "65535"] {
        assert_eq!(
            parse_default(&format!("go mate {value}")),
            Err(ParseError::Malformed("mate ply horizon overflow")),
            "mate {value}"
        );
    }
    for (keyword, zero_reason, overflow_reason) in [
        ("depth", "depth must be positive", "depth overflow"),
        ("mate", "mate must be positive", "mate overflow"),
    ] {
        assert_eq!(
            parse_default(&format!("go {keyword}")),
            Err(ParseError::Malformed("go field lacks value"))
        );
        assert_eq!(
            parse_default(&format!("go {keyword} 0")),
            Err(ParseError::Malformed(zero_reason))
        );
        assert_eq!(
            parse_default(&format!("go {keyword} 65536")),
            Err(ParseError::Malformed(overflow_reason))
        );
        assert_eq!(
            parse_default(&format!("go {keyword} 1 {keyword} 2")),
            Err(ParseError::Malformed("duplicate go field"))
        );
        for value in ["-1", "+1", "1.5", "NaN", "depth"] {
            assert_eq!(
                parse_default(&format!("go {keyword} {value}")),
                Err(ParseError::Malformed("expected unsigned decimal integer")),
                "{keyword} {value}"
            );
        }
    }
}

#[test]
fn go_parses_each_supported_budget_mode_without_losing_units() {
    assert_eq!(parse_default("go"), Ok(Command::Go(GoLimits::default())));
    assert_eq!(
        parse_default("go movetime 150 nodes 23"),
        Ok(Command::Go(GoLimits {
            movetime_ms: Some(150),
            nodes: Some(23),
            ..GoLimits::default()
        }))
    );
    assert_eq!(
        parse_default("go wtime 900 btime 800 winc 5 binc 7 movestogo 12 nodes 99"),
        Ok(Command::Go(GoLimits {
            white_time_ms: Some(900),
            black_time_ms: Some(800),
            white_increment_ms: Some(5),
            black_increment_ms: Some(7),
            moves_to_go: Some(12),
            nodes: Some(99),
            ..GoLimits::default()
        }))
    );
    assert_eq!(
        parse_default("go infinite"),
        Ok(Command::Go(GoLimits {
            infinite: true,
            ..GoLimits::default()
        }))
    );
    assert_eq!(
        parse_default("go nodes 1"),
        Ok(Command::Go(GoLimits {
            nodes: Some(1),
            ..GoLimits::default()
        }))
    );
    assert!(parse_default("go movetime 0").is_ok());
    assert!(parse_default("go wtime 0 btime 0").is_ok());
}

#[test]
fn go_rejects_conflicts_missing_fields_duplicates_and_overflow() {
    for line in [
        "go movetime",
        "go nodes 0",
        "go movestogo 0 wtime 1 btime 1",
        "go wtime 1",
        "go btime 1",
        "go winc 1",
        "go binc 1",
        "go movestogo 1",
        "go movetime 1 wtime 1 btime 1",
        "go infinite movetime 1",
        "go infinite wtime 1 btime 1",
        "go infinite infinite",
        "go ponder ponder",
        "go ponder infinite",
        "ponderhit extra",
        "go nodes 1 nodes 2",
        "go movetime 1 movetime 2",
        "go wtime 1 btime 1 winc 1 winc 2",
        "go wtime 1 btime 1 movestogo 4294967296",
        "go nodes 18446744073709551616",
        "go movetime -1",
        "go movetime +1",
        "go movetime 1.5",
        "go movetime NaN",
    ] {
        assert!(
            matches!(parse_default(line), Err(ParseError::Malformed(_))),
            "{line:?}"
        );
    }
}

#[test]
fn position_preserves_fen_and_complete_ordered_move_trace() {
    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    assert_eq!(
        parse_default(&format!("position fen {fen} moves e2e4 e7e5 g1f3")),
        Ok(Command::Position(PositionSpec {
            base: PositionBase::Fen(fen.into()),
            moves: vec!["e2e4".into(), "e7e5".into(), "g1f3".into()],
        }))
    );
    for suffix in ["q", "r", "b", "n"] {
        let movement = format!("a7a8{suffix}");
        assert_eq!(
            parse_default(&format!("position startpos moves {movement}")),
            Ok(Command::Position(PositionSpec {
                base: PositionBase::StartPos,
                moves: vec![movement]
            }))
        );
    }
}

#[test]
fn malformed_position_text_is_rejected_before_rules() {
    for line in [
        "position",
        "position startpos moves",
        "position startpos e2e4",
        "position startpos moves e9e4",
        "position startpos moves e2e2",
        "position startpos moves e2e4qz",
        "position startpos moves a7a8k",
        "position startpos moves a7a8Q",
        "position fen 8/8/8/8/8/8/8/8 w - - 0",
        "position fen 8/8/8/8/8/8/8 w - - 0 1",
        "position fen 9/8/8/8/8/8/8/8 w - - 0 1",
        "position fen 7/8/8/8/8/8/8/8 w - - 0 1",
        "position fen 8/8/8/8/8/8/8/8 x - - 0 1",
        "position fen 8/8/8/8/8/8/8/8 w KK - 0 1",
        "position fen 8/8/8/8/8/8/8/8 w - a4 0 1",
        "position fen 8/8/8/8/8/8/8/8 w - - -1 1",
        "position fen 8/8/8/8/8/8/8/8 w - - 0 0",
        "position fen 8/8/8/8/8/8/8/8 w - - 4294967296 1",
        "position fen 8/8/8/8/8/8/8/8 w - - 0 1 extra",
    ] {
        assert!(
            matches!(parse_default(line), Err(ParseError::Malformed(_))),
            "{line:?}"
        );
    }
    // Text validity is insufficient to establish chess semantics; Rules owns that check.
    assert!(parse_default("position fen 8/8/8/8/8/8/8/8 w - - 0 1").is_ok());
}

#[test]
fn option_parser_preserves_multiword_names_values_buttons_and_empty_strings() {
    assert_eq!(
        parse_default("setoption name Clear Hash"),
        Ok(Command::SetOption {
            name: "Clear Hash".into(),
            value: None
        })
    );
    assert_eq!(
        parse_default("setoption name Weights File value one two"),
        Ok(Command::SetOption {
            name: "Weights File".into(),
            value: Some("one two".into())
        })
    );
    assert_eq!(
        parse_default("setoption name Weights File value"),
        Ok(Command::SetOption {
            name: "Weights File".into(),
            value: Some(String::new())
        })
    );
    for line in [
        "setoption",
        "setoption value true",
        "setoption name",
        "setoption name value true",
    ] {
        assert!(
            matches!(parse_default(line), Err(ParseError::Malformed(_))),
            "{line:?}"
        );
    }
}

#[test]
fn option_payloads_preserve_internal_spaces_and_only_trim_syntax_boundaries() {
    for (line, name, value) in [
        (
            "setoption name Use  Cache value true",
            "Use  Cache",
            Some("true"),
        ),
        (
            "setoption name Search Mode value Very  Fast",
            "Search Mode",
            Some("Very  Fast"),
        ),
        (
            "\tsetoption\tname\tWeights  File \tvalue\tmodel  small.onnx\t\r\n",
            "Weights  File",
            Some("model  small.onnx"),
        ),
        (
            "setoption name Weights value first value  second",
            "Weights",
            Some("first value  second"),
        ),
        ("setoption name Clear  Hash \t", "Clear  Hash", None),
        ("setoption name Weights value \t", "Weights", Some("")),
    ] {
        assert_eq!(
            parse_default(line),
            Ok(Command::SetOption {
                name: name.into(),
                value: value.map(str::to_owned),
            }),
            "{line:?}"
        );
    }
}

#[test]
fn option_byte_limits_measure_the_preserved_payload() {
    let bounded = ParserLimits {
        max_option_name_bytes: 3,
        max_option_value_bytes: 3,
        ..ParserLimits::default()
    };
    assert!(parse("setoption name A B value x y", bounded).is_ok());
    assert_eq!(
        parse("setoption name A  B value x y", bounded),
        Err(ParseError::Limit("option name"))
    );
    assert_eq!(
        parse("setoption name A B value x  y", bounded),
        Err(ParseError::Limit("option value"))
    );
}

#[test]
fn configured_external_input_limits_are_enforced_at_the_boundary() {
    let short_line = ParserLimits {
        max_line_bytes: 3,
        ..ParserLimits::default()
    };
    assert_eq!(parse("uci", short_line), Ok(Command::Uci));
    assert_eq!(parse("uci\n", short_line), Err(ParseError::LineTooLong));
    let bounded = ParserLimits {
        max_moves: 2,
        max_option_name_bytes: 4,
        max_option_value_bytes: 3,
        ..ParserLimits::default()
    };
    assert!(parse("position startpos moves e2e4 e7e5", bounded).is_ok());
    assert_eq!(
        parse("position startpos moves e2e4 e7e5 g1f3", bounded),
        Err(ParseError::Limit("move trace"))
    );
    assert!(parse("setoption name Hash value 123", bounded).is_ok());
    assert_eq!(
        parse("setoption name HashX value 123", bounded),
        Err(ParseError::Limit("option name"))
    );
    assert_eq!(
        parse("setoption name Hash value 1234", bounded),
        Err(ParseError::Limit("option value"))
    );
}
