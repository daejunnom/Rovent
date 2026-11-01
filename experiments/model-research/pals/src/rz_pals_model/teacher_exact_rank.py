"""Opt-in same-go exact-rank observation codec; never target authority.

The caller must independently obtain the expected identity and legal candidates
from the original live owners. JSON equality cannot establish owner custody or
physical completion, and this module does not register a training data domain.
"""

DOMAIN = "rz-pals-teacher-exact-rank-observation/1"
_HASHES = ("input_sha256", "view_sha256", "rules_state_sha256", "rules_history_sha256",
           "input_lock_sha256", "evaluation_namespace_sha256", "root_restriction_sha256",
           "teacher_binary_sha256", "teacher_profile_sha256")
_ID_KEYS = frozenset((*_HASHES, "process_id", "process_group", "process_start_ticks",
                      "execution_epoch", "clock_epoch", "original_deadline_ns", "root_generation", "collection_owner", "job_sequence",
                      "request_id", "requested_multipv", "root_moves"))
_OBS_KEYS = frozenset(("schema", "policy", "identity", "observation_sequence",
                      "opinions", "final_bestmove", "final_bestmove_matches_rank1", "go_ns",
                      "bestmove_ns", "ready_ns", "returned_ns",
                      "cancel_sequence", "yield_sequence", "stop_sent"))


def _uint(value, bits):
    if type(value) is not int or not 0 <= value < (1 << bits):
        raise ValueError("exact-rank integer out of range")
    return value


def _move(value):
    _uint(value, 16)
    if value >> 12 > 4 or (value & 63) == ((value >> 6) & 63):
        raise ValueError("exact-rank invalid Move16")
    return value


def _moves(values, *, unique=False):
    if type(values) is not list or not 1 <= len(values) <= 256:
        raise ValueError("exact-rank bounded nonempty moves required")
    for value in values:
        _move(value)
    if unique and len(set(values)) != len(values):
        raise ValueError("exact-rank duplicate moves")
    return values


def _identity(value):
    if type(value) is not dict or set(value) != _ID_KEYS:
        raise ValueError("exact-rank owner identity fields differ")
    for key in _HASHES:
        text = value[key]
        if type(text) is not str or len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
            raise ValueError("exact-rank identity digest invalid")
    for key in ("process_id", "process_group", "collection_owner"):
        _uint(value[key], 32)
    for key in ("process_start_ticks", "execution_epoch", "clock_epoch", "original_deadline_ns", "root_generation", "job_sequence", "request_id"):
        _uint(value[key], 64)
    _uint(value["requested_multipv"], 16)
    if not all(value[key] for key in ("process_id", "process_group", "process_start_ticks", "job_sequence", "request_id")):
        raise ValueError("exact-rank process or job identity missing")
    if value["root_moves"] is not None:
        _moves(value["root_moves"], unique=True)
    return value


def validate_teacher_exact_rank_observation(observation, expected_identity, legal_moves):
    """Validate bytes/joins only; return no row, target, or capability."""
    if type(observation) is not dict or set(observation) != _OBS_KEYS:
        raise ValueError("exact-rank observation fields differ")
    if observation["schema"] != DOMAIN or observation["policy"] != "completed_exact_rank_set":
        raise ValueError("exact-rank observation domain or policy mismatch")
    identity = _identity(observation["identity"])
    if identity != _identity(expected_identity):
        raise ValueError("exact-rank original owner scope differs")
    legal = _moves(legal_moves, unique=True)
    roots = identity["root_moves"]
    if roots is not None and any(move not in legal for move in roots):
        raise ValueError("exact-rank root restriction differs from legal moves")
    for key in ("go_ns", "bestmove_ns", "ready_ns", "returned_ns",
                "cancel_sequence", "yield_sequence", "observation_sequence"):
        _uint(observation[key], 64)
    if (not observation["go_ns"] <= observation["bestmove_ns"] <= observation["ready_ns"]
            <= observation["returned_ns"] < identity["original_deadline_ns"]
            or observation["cancel_sequence"] != 0 or observation["yield_sequence"] != 0
            or observation["stop_sent"] is not False or not observation["observation_sequence"]):
        raise ValueError("exact-rank original go did not finish naturally within deadline")
    count = identity["requested_multipv"]
    opinions = observation["opinions"]
    final = _move(observation["final_bestmove"])
    if (not 1 <= count <= 4 or type(opinions) is not list or len(opinions) != count
            or final not in legal or roots is not None and final not in roots):
        raise ValueError("exact-rank count or final bestmove invalid")
    depth, ranked = None, []
    for index, opinion in enumerate(opinions):
        if type(opinion) is not dict or set(opinion) != {"rank", "depth", "bound", "pv"}:
            raise ValueError("exact-rank opinion fields differ")
        rank = _uint(opinion["rank"], 16)
        current_depth = _uint(opinion["depth"], 16)
        pv = _moves(opinion["pv"])
        if depth is None:
            depth = current_depth
        first = pv[0]
        if (rank != index + 1 or not depth or current_depth != depth or opinion["bound"] != "ExactReported"
                or first in ranked or first not in legal or roots is not None and first not in roots):
            raise ValueError("exact-rank rank/depth/bound/PV invalid")
        ranked.append(first)
    matches = observation["final_bestmove_matches_rank1"]
    if type(matches) is not bool or matches != (ranked[0] == final):
        raise ValueError("exact-rank final bestmove comparison differs")
