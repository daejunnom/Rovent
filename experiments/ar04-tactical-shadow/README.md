# AR04 tactical shadow: state observation only

This independent, opt-in Rust experiment records one chess position, a proposed
candidate and up to two alternative replies. It does not run a neural network,
make a tactical classification, select a move, produce policy/WDL, or change
the engine. The AR04 manual circuit input definitions and weights are not
registered here; the result always records `research_circuit: not_available`.
Do not substitute guessed coefficients or interpret this as AR04 execution.

After registering the source and using the repository's managed build lifecycle,
an example invocation is:

```text
ar04-tactical-shadow --startpos --move e2e4 --candidate e7e5 --reply g1f3 --reply b1c3
```

For an imported position, use `--fen "<FEN>"` instead of `--startpos`.
The exact supplied history is passed as repeated `--move <UCI>` arguments. A
FEN origin has an unknown earlier prefix; the JSON records the actual Rules
history completeness and repetition evidence separately. A halfmove counter
or a reversible prefix does not prove a complete game history. Candidates and
replies are checked against board-legal moves of their respective actual states.
Terminal positions are explicitly marked unavailable. Draw claims are not
exercised, and unknown repetition history is not treated as complete. The
`ongoing` result is the current Rules classification, not proof that every
possible dead position has been ruled out: dead-position recognition covers
only its documented proven material subset. Fivefold availability and known
repetition count are reported separately.

The emitted legal-move order is the Rules order, and replies remain in supplied
order. Up to 256 history plies and two replies are accepted. Parsing or Rules
failures produce a nonzero exit and no success JSON. Output includes input
origin, supplied and normalized initial and resulting FENs, input moves, root legal moves,
candidate and reply observations. An illegal candidate has no child; an illegal
reply has no resulting FEN. Terminal-state candidate/reply legality is `null`,
not an assertion that a board-legal move is illegal. No source package or private research artifact is
copied, and the experiment does not include the AR04 21-feature encoder.

Independent checking should use `python-chess` in a separate process, as in
`crates/rz-position/tests/oracle.py`, for move sets and FEN transitions,
including promotion, en passant, castling and pin cases. Compare en-passant FEN
conventions explicitly. Add original problem positions and separately verified
defensive roles/variations before making any motif claim. Even that comparison
does not certify a forced mate unless all relevant responses and actual terminal
states are independently checked. To evaluate the real research network, first
obtain and verify its exact revision, rights, input mapping and coefficients;
pin those per experiment, not as the engine's permanent default.
