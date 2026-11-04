//! Typed projections of publicly observable external checker data.
//! These values do not recreate an owner, establish a process or GPU fence,
//! or authorize a training target.

use rz_position::{BoardMove, Color, PieceKind};
use rz_search::cpu_checker::{
    CheckerAttempt, CheckerShutdown, CheckerWork, ExternalAttemptEvidence, ExternalBound,
    ExternalCheckerReport, ExternalCompletion, ExternalProcessIdentity, ExternalRawScore,
};
use serde_json::{Value, json};
use std::time::Duration;

fn duration(value: Duration) -> Value {
    json!({"seconds":value.as_secs(), "nanoseconds":value.subsec_nanos()})
}

fn movement(value: BoardMove) -> Value {
    let promotion = value.promotion.map(|piece| match piece {
        PieceKind::Pawn => "pawn",
        PieceKind::Knight => "knight",
        PieceKind::Bishop => "bishop",
        PieceKind::Rook => "rook",
        PieceKind::Queen => "queen",
        PieceKind::King => "king",
    });
    json!({"from":{"file":value.from.file(),"rank":value.from.rank()},
        "to":{"file":value.to.file(),"rank":value.to.rank()},"promotion":promotion})
}

fn score(value: ExternalRawScore) -> Value {
    match value {
        ExternalRawScore::Centipawns(n) => json!({"kind":"centipawns","value":n}),
        ExternalRawScore::MateMoves(n) => json!({"kind":"mate_moves","value":n}),
        ExternalRawScore::Unknown => json!({"kind":"unknown"}),
    }
}

fn bound(value: ExternalBound) -> &'static str {
    match value {
        ExternalBound::ExactReported => "exact_reported",
        ExternalBound::Lower => "lower",
        ExternalBound::Upper => "upper",
        ExternalBound::Unknown => "unknown",
    }
}

fn completion(value: ExternalCompletion) -> &'static str {
    match value {
        ExternalCompletion::Pending => "pending",
        ExternalCompletion::BestMove => "best_move",
        ExternalCompletion::StoppedDeadline => "stopped_deadline",
        ExternalCompletion::StoppedCanceled => "stopped_canceled",
        ExternalCompletion::StoppedYield => "stopped_yield",
    }
}

fn work(value: CheckerWork) -> Value {
    json!({"nodes":value.nodes,"qnodes":value.qnodes,"tt_hits":value.tt_hits})
}

fn process_identity(value: ExternalProcessIdentity) -> Value {
    json!({"pid":value.pid,"process_group":value.process_group,"proc_start_ticks":value.proc_start_ticks})
}

pub(super) fn external_report(value: &ExternalCheckerReport) -> Value {
    json!({
        "identity":value.identity,
        "observed_uci":{"name":value.observed_uci.name,"author":value.observed_uci.author},
        "request_id":value.request_id,
        "best_move":value.best_move.map(movement),
        "pv":value.pv.iter().copied().map(movement).collect::<Vec<_>>(),
        "score":score(value.score),"bound":bound(value.bound),"wdl_per_mille":value.wdl_per_mille,
        "perspective":match value.perspective { Color::White => "white", Color::Black => "black" },
        "requested_depth":value.requested_depth,"reported_depth":value.reported_depth,
        "seldepth":value.seldepth,"root_restricted":value.root_restricted,
        "completion":completion(value.completion),"work":work(value.work),"elapsed":duration(value.elapsed)
    })
}

pub(super) fn checker_shutdown(value: &CheckerShutdown) -> Value {
    json!({
        "process_identity":value.process_identity.map(process_identity),
        "stop_sent":value.stop_sent,"quit_sent":value.quit_sent,"exit_observed":value.exit_observed,
        "stdout_drained":value.stdout_drained,"stderr_drained":value.stderr_drained,
        "exit_code":value.exit_code,"exit_signal":value.exit_signal,"cleanup_complete":value.cleanup_complete,
        "quarantined":value.quarantined,"ownership_lost":value.ownership_lost,
        "stdout_bytes":value.stdout_bytes,"stderr_bytes":value.stderr_bytes
    })
}

pub(super) fn external_attempt(value: &ExternalAttemptEvidence) -> Value {
    json!({"request_id":value.request_id,
        "partial_report":value.partial_report.as_ref().map(external_report),
        "process":checker_shutdown(&value.process)})
}

pub(super) fn checker_attempt(value: &CheckerAttempt) -> Value {
    json!({"work":work(value.work),"elapsed":duration(value.elapsed),
        "external":value.external.as_ref().map(external_attempt)})
}
