"""Historical joins for original P/C Repair witnesses, never live authority.

The arena attaches these bytes only while borrowing the private accepted owner.
This reader checks the preserved causal chain and binary invocation seals. It
cannot reconstruct Store ownership, a native acceptance permit, a physical
completion capability, or a Rules outcome from a conditional model comparison.
"""
from __future__ import annotations

import hashlib
import math
import struct
import unicodedata

from .generation_data import fields, sha, uint, _moves

HEAD_DOMAIN = "rz-rovent-head-source/1"
WITNESS_DOMAIN = "rz-rovent-pc-divergence-witness/1"
ATTACK_DOMAIN = "rz-rovent-pc-attack-ranked-source/1"
ATTACK_SOURCE_DOMAIN = "rz-rovent-pc-divergence-source-with-attack/1"
SOURCE_NAME = "rovent_actual_pc_divergence"
_TICKET_FIELDS = ("process_epoch", "game_generation", "root_generation", "request_sequence", "implementation_sha256",
    "question_digest", "view_sequence", "view_digest", "context_policy", "actual_tokens", "selected_records", "anchors",
    "length_bucket", "root_state", "representation", "deadline_clock", "deadline_tick")
_REPRESENTATION_FIELDS = ("input", "history", "candidates", "mask_positions", "model", "encoding", "precision",
    "frozen_epoch", "record_revision", "role")
_AUTHORITY_FIELDS = ("process_epoch", "game_generation", "root_generation", "implementation_sha256")


def _bytes(value):
    return bytes.fromhex(sha(value))


def _u(value, size=8):
    return uint(value, 2 ** (8 * size) - 1).to_bytes(size, "little")


def _positive(value, maximum=2**64 - 1):
    if uint(value, maximum) == 0:
        raise ValueError("positive original P/C witness identity required")
    return value


def _text(value, maximum):
    if (type(value) is not str or not value or len(value.encode("utf-8")) > maximum
            or any(unicodedata.category(c) == "Cc" for c in value)):
        raise ValueError("bounded original P/C witness text required")
    return value


def _line(value, *, empty=False):
    _moves(value)
    if not empty and not value:
        raise ValueError("nonempty original P/C witness line required")
    return _u(len(value)) + b"".join(_u(move, 2) for move in value)


def _state(value):
    fields(value, ("owner", "revision", "semantic"), "P/C witness exact root state")
    uint(value["owner"])
    uint(value["revision"])
    sha(value["semantic"])
    return value


def _ticket(value):
    fields(value, _TICKET_FIELDS, "P/C witness original frozen ticket")
    for key in ("process_epoch", "request_sequence", "view_sequence"):
        _positive(value[key])
    for key in ("game_generation", "root_generation", "deadline_clock"):
        uint(value[key])
    _positive(value["deadline_tick"])
    uint(value["length_bucket"], 256)
    for key in ("implementation_sha256", "question_digest", "view_digest", "context_policy"):
        sha(value[key])
    if (uint(value["actual_tokens"], 512) > 512 or uint(value["selected_records"], 64) > 64
            or not 1 <= uint(value["anchors"], 4) or value["length_bucket"] not in (8, 16, 32, 64, 128, 256)):
        raise ValueError("P/C witness selected context allocation differs")
    if value["deadline_clock"] != value["process_epoch"]:
        raise ValueError("P/C witness original deadline clock differs from its process")
    _state(value["root_state"])
    rep = fields(value["representation"], _REPRESENTATION_FIELDS, "P/C witness exact frozen representation")
    for key in ("input", "history", "candidates", "mask_positions", "model", "encoding"):
        sha(rep[key])
    uint(rep["frozen_epoch"])
    uint(rep["record_revision"])
    if rep["role"] not in ("proposer", "critic") or rep["precision"] != "Fp32":
        raise ValueError("P/C witness requires its original P/C FP32 representation")
    return value


def _same_scope(left, right):
    # The invocation seal does not hash clock domain, root StateIdentity or all
    # view allocation fields; keep these independent original-owner joins.
    return all(left[key] == right[key] for key in (*_AUTHORITY_FIELDS, "deadline_clock", "deadline_tick", "root_state"))


def _same_model(left, right):
    return all(left["representation"][key] == right["representation"][key]
               for key in ("model", "encoding", "precision", "frozen_epoch"))


def _float_bits(value, count=3):
    if type(value) is not list or len(value) != count:
        raise ValueError("P/C witness original FP32 allocation differs")
    raw = b"".join(_u(bits, 4) for bits in value)
    values = struct.unpack("<" + "f" * count, raw)
    if any(not math.isfinite(number) for number in values):
        raise ValueError("P/C witness original FP32 values must be finite")
    return raw, values


def _wdl(value):
    raw, values = _float_bits(value)
    total = 0.0
    for number in values:
        total = struct.unpack("<f", struct.pack("<f", total + number))[0]
    tolerance = struct.unpack("<f", struct.pack("<f", 1e-4))[0]
    if any(number < 0 or number > 1 for number in values) or abs(_f32_subtract(total, 1.0)) > tolerance:
        raise ValueError("P/C witness WDL probabilities are not normalized")
    return raw, values


def _f32_subtract(left, right):
    # Rust compares f32 W-L, rather than a Python binary64 subtraction.
    return struct.unpack("<f", struct.pack("<f", left - right))[0]


def _total_f32_key(bits):
    # Finite f32::total_cmp orders -0.0 before +0.0. The selected attack slot
    # must follow the original native tie rule, not Python max(float).
    uint(bits, 2**32 - 1)
    return (~bits & 0xffffffff) if bits & 0x80000000 else bits | 0x80000000


def _seal(domain, framed, expected):
    if hashlib.sha256(domain.encode("ascii") + b"\0" + framed).hexdigest() != sha(expected):
        raise ValueError("original P/C native binary invocation seal differs")


def _authority(value):
    fields(value, _AUTHORITY_FIELDS, "P/C witness original authority")
    _positive(value["process_epoch"])
    uint(value["game_generation"])
    uint(value["root_generation"])
    sha(value["implementation_sha256"])
    return value


def _view(value, ticket):
    fields(value, ("authority", "view_sequence", "context_policy", "view_digest", "root_state", "actual_tokens",
                   "selected_records", "anchors", "length_bucket"), "P/C witness peer original view")
    authority = _authority(value["authority"])
    _state(value["root_state"])
    for key in ("view_sequence", "actual_tokens", "selected_records", "anchors", "length_bucket"):
        uint(value[key])
    for key in ("context_policy", "view_digest"):
        sha(value[key])
    if (any(authority[key] != ticket[key] for key in _AUTHORITY_FIELDS)
            or any(value[key] != ticket[key] for key in set(value) - {"authority"})):
        raise ValueError("P/C witness actual peer view differs from its accepted ticket")


def _question_id(value):
    fields(value, ("authority", "sequence"), "P/C witness reciprocal question id")
    _authority(value["authority"])
    _positive(value["sequence"])
    return value


def _peer(value, ticket):
    fields(value, ("id", "parent", "sender", "recipient", "purpose", "anchor", "anchor_root_ply",
        "original_first_move", "prefix_digest", "line_digest", "line_offset", "question_digest", "view",
        "evidence_revision", "root_budget", "original_deadline"), "P/C witness original peer envelope")
    _question_id(value["id"])
    if value["parent"] is not None:
        _question_id(value["parent"])
        if (value["parent"]["authority"] != value["id"]["authority"]
                or value["parent"]["sequence"] >= value["id"]["sequence"]):
            raise ValueError("P/C witness peer parent chronology/scope differs")
    if (value["sender"] not in ("Proposer", "Critic", "Researcher")
            or value["recipient"] != {"proposer": "Proposer", "critic": "Critic"}[ticket["representation"]["role"]]
            or value["sender"] == value["recipient"]
            or value["purpose"] not in ("Propose", "Counter", "Repair", "TacticalReview", "Continue", "FreshValue")):
        raise ValueError("P/C witness actual peer role/purpose differs")
    _state(value["anchor"])
    uint(value["anchor_root_ply"], 65535)
    uint(value["line_offset"], 65535)
    uint(value["evidence_revision"])
    if value["original_first_move"] is not None:
        _moves([value["original_first_move"]], 1)
    for key in ("prefix_digest", "question_digest", "root_budget"):
        sha(value[key])
    if value["line_digest"] is not None:
        sha(value["line_digest"])
    elif value["line_offset"] != 0:
        raise ValueError("P/C witness peer line offset requires its original line descriptor")
    _view(value["view"], ticket)
    deadline = fields(value["original_deadline"], ("clock", "tick"), "P/C witness peer original deadline")
    if (any(value["id"]["authority"][key] != ticket[key] for key in _AUTHORITY_FIELDS)
            or value["question_digest"] != ticket["question_digest"]
            or uint(deadline["clock"]) != ticket["deadline_clock"]
            or uint(deadline["tick"]) != ticket["deadline_tick"]):
        raise ValueError("P/C witness actual peer authority/question/deadline differs")
    return value


def _origin(value):
    fields(value, ("frame", "ticket", "output_digest", "kind", "peer", "seal_sha256"),
           "P/C witness original accepted origin")
    ticket = _ticket(value["ticket"])
    frame = fields(value["frame"], (*_AUTHORITY_FIELDS, "sequence"), "P/C witness original frame")
    _positive(frame["sequence"])
    if any(frame[key] != ticket[key] or type(frame[key]) is not type(ticket[key]) for key in _AUTHORITY_FIELDS):
        raise ValueError("P/C witness original frame authority differs from its ticket")
    framed = _u(frame["sequence"]) + _u(ticket["request_sequence"])
    framed += b"".join(_u(ticket[key]) for key in ("process_epoch", "game_generation", "root_generation"))
    framed += _bytes(ticket["implementation_sha256"]) + _bytes(ticket["question_digest"])
    framed += _bytes(ticket["view_digest"]) + _u(ticket["view_sequence"]) + _bytes(value["output_digest"])
    rep = ticket["representation"]
    framed += b"".join(_bytes(rep[key]) for key in ("input", "history", "candidates", "mask_positions", "model", "encoding"))
    framed += _u(rep["frozen_epoch"]) + _u(rep["record_revision"])
    framed += bytes(({"proposer": 0, "critic": 1}[rep["role"]], 0)) + _u(ticket["deadline_tick"])
    if value["kind"] == "local_accepted" and value["peer"] is None:
        framed += b"\0"
    elif value["kind"] == "peer_question" and value["peer"] is not None:
        peer = _peer(value["peer"], ticket)
        framed += b"\1" + _u(peer["id"]["sequence"]) + _bytes(peer["question_digest"])
    else:
        raise ValueError("P/C witness local source cannot be relabelled as an actual peer question")
    _seal("rz-rovent-accepted-local-origin-seal/1", framed, value["seal_sha256"])
    return ticket


def _check_origin(value, original, *, role, root_budget):
    ticket = _origin(value)
    if (not _same_scope(ticket, original) or not _same_model(ticket, original)
            or ticket["representation"]["role"] != role):
        raise ValueError("P/C witness changed original owner scope/model/private role")
    if value["peer"] is not None and value["peer"]["root_budget"] != root_budget:
        raise ValueError("P/C witness actual peer changed the original root budget")
    return ticket


def _source(record, value):
    fields(value, ("native_source_sha256", "accepted", "root_budget_sha256", "counter_local_index",
        "counter_revision", "divergence_index", "divergence_ply", "first_move", "proposal", "counter", "perspective"),
        "P/C witness original visible counter source")
    snapshot, native = record["input"], record["native"]
    original = native["raw_submission"]["ticket"]
    budget = native["raw_retirement"]["root_admission"]["root_budget_sha256"]
    ticket = _check_origin(value["accepted"], original, role="critic", root_budget=budget)
    if (ticket != original or value["accepted"]["output_digest"] != native["output_digest"]
            or value["root_budget_sha256"] != budget or record["role"] != "critic"):
        raise ValueError("P/C witness source is not the exact original retired critic input/output")
    slot = uint(value["divergence_index"], 65535)
    local = uint(value["counter_local_index"], 63)
    revision = _positive(value["counter_revision"])
    attack = uint(value["divergence_ply"], 255)
    _moves([value["first_move"]], 1)
    proposal, counter = value["proposal"], value["counter"]
    framed_lines = _line(proposal) + _line(counter)
    view, query = snapshot["selected_view"], snapshot["selected_view"]["query"]
    if (slot >= len(snapshot["attack_mapping"]) or local >= len(view["records"])
            or not 0 < attack < min(len(proposal), len(counter))
            or query["purpose"] != "divergence_policy" or query["prefix"] or query["counter_present"] is not True
            or query["proposal"] != proposal or query["counter"] != counter
            or proposal[0] != value["first_move"] or counter[0] != value["first_move"]
            or value["perspective"] != ("white" if snapshot["rules_anchor"]["white_to_move"] else "black")):
        raise ValueError("P/C witness visible source changed its exact submitted question/attack/anchor")
    mapping = snapshot["attack_mapping"][slot]
    if (mapping["attack_ply"] != attack or mapping["slot"] != slot
            or mapping["selected_counter_local_index"] is not None
                and (mapping["selected_counter_local_index"] != local or mapping["selected_counter_revision"] != revision)):
        raise ValueError("P/C witness source differs from the original submitted attack mapping")
    selected = view["records"][local]
    review = native["raw_submission"]["submitted_review"]
    metadata = review["record_metadata"][local]
    for key in ("record_revision", "source_revision", "source_line_offset"):
        uint(metadata[key])
    if metadata["original_first_move"] is not None:
        _moves([metadata["original_first_move"]], 1)
    if (selected["kind"] != "counter" or selected["moves"] != counter
            or view["anchors"][selected["anchor"]] != snapshot["rules_anchor"]
            or metadata["record_revision"] != revision or metadata["source_revision"] != revision
            or metadata["source_line_offset"] != 0 or metadata["original_first_move"] != value["first_move"]):
        raise ValueError("P/C witness counter lacks its original visible revision/Rules anchor")
    bits = native["raw_completion"]["output"]["payload"]["divergence_logits_bits"]
    _float_bits(bits, len(snapshot["attack_mapping"]))
    if not bits or max(range(len(bits)), key=lambda index: (_total_f32_key(bits[index]), -index)) != slot:
        raise ValueError("P/C witness attack slot is not the original finite FP32 selected head")
    framed = _bytes(value["accepted"]["seal_sha256"]) + _bytes(budget) + _u(local, 2) + _u(revision)
    framed += _u(slot, 2) + _u(attack) + framed_lines
    _seal("rz-rovent-pc-divergence-source/1", framed, value["native_source_sha256"])
    return ticket


def _attack_source(record, value, visible, visible_ticket):
    fields(value, ("schema", "native_attack_sha256", "accepted", "source_sequence", "root_budget_sha256", "divergence_index",
        "divergence_ply", "first_move", "proposal", "perspective"), "P/C witness original attack ranking")
    snapshot, native = record["input"], record["native"]
    original = native["raw_submission"]["ticket"]
    budget = native["raw_retirement"]["root_admission"]["root_budget_sha256"]
    ticket = _check_origin(value["accepted"], original, role="critic", root_budget=budget)
    _positive(value["source_sequence"])
    slot, attack = uint(value["divergence_index"], 65535), uint(value["divergence_ply"], 255)
    _moves([value["first_move"]], 1)
    proposal = value["proposal"]
    framed_line = _line(proposal)
    query = snapshot["selected_view"]["query"]
    if (value["schema"] != ATTACK_DOMAIN or ticket != original or record["role"] != "critic"
            or value["accepted"]["output_digest"] != native["output_digest"]
            or value["root_budget_sha256"] != budget or budget != visible["root_budget_sha256"]
            or not _same_scope(ticket, visible_ticket) or not _same_model(ticket, visible_ticket)
            or ticket["request_sequence"] >= visible_ticket["request_sequence"]
            or slot >= len(snapshot["attack_mapping"]) or not 0 < attack < len(proposal)
            or query["purpose"] != "divergence_policy" or query["prefix"]
            or query["counter_present"] is not False or query["counter"]
            or query["proposal"] != proposal or proposal[0] != value["first_move"]
            or proposal != visible["proposal"] or attack != visible["divergence_ply"]
            or visible["counter"][:attack] != proposal[:attack] or visible["counter"][attack] == proposal[attack]
            or value["first_move"] != visible["first_move"] or value["perspective"] != visible["perspective"]
            or value["perspective"] != ("white" if snapshot["rules_anchor"]["white_to_move"] else "black")):
        raise ValueError("P/C witness initial attack changed its original input or future visible chronology/scope")
    mapping = snapshot["attack_mapping"][slot]
    if (mapping["slot"] != slot or mapping["attack_ply"] != attack
            or mapping["selected_counter_local_index"] is not None or mapping["selected_counter_revision"] is not None):
        raise ValueError("P/C witness initial attack cannot contain a future selected counter")
    bits = native["raw_completion"]["output"]["payload"]["divergence_logits_bits"]
    _float_bits(bits, len(snapshot["attack_mapping"]))
    if not bits or max(range(len(bits)), key=lambda index: (_total_f32_key(bits[index]), -index)) != slot:
        raise ValueError("P/C witness initial attack slot is not the original finite FP32 selected head")
    framed = _bytes(value["accepted"]["seal_sha256"]) + _bytes(budget)
    framed += _u(slot, 2) + _u(attack) + _u(value["first_move"], 2) + framed_line
    _seal(ATTACK_DOMAIN, framed, value["native_attack_sha256"])
    return ticket


def _source_context(record, value, manifest):
    if type(value) is not dict or "schema" not in value:
        return _source(record, value), value
    fields(value, ("schema", "native_source_sha256", "visible_source", "visible_source_sequence", "attack_ranked_source", "visible_record_evidence"),
           "P/C witness source with original attack ranking")
    if value["schema"] != ATTACK_SOURCE_DOMAIN or manifest is None:
        raise ValueError("P/C witness original attack source requires its explicit corpus and future invocation evidence")
    from .rovent_native_v3 import validate_native_v3_evidence
    evidence = value["visible_record_evidence"]
    future = validate_native_v3_evidence(manifest, evidence)
    visible = value["visible_source"]
    visible_ticket = _source(future, visible)
    initial, later = record["input"], future["input"]
    if (any(initial[key] != later[key] for key in ("game_id", "opening_id", "model_epoch", "context_epoch",
                                                  "rules_anchor", "actual_history"))
            or initial["capture_sequence"] >= later["capture_sequence"]
            or record["native"]["source_sha256"] != evidence["native"]["source_sha256"]
            or record["native"]["raw_retirement"]["root_admission"] !=
                evidence["native"]["raw_retirement"]["root_admission"]):
        raise ValueError("P/C witness future visible evidence crossed its original root/source/input chronology")
    attack = value["attack_ranked_source"]
    _attack_source(record, attack, visible, visible_ticket)
    if _positive(attack["source_sequence"]) >= _positive(value["visible_source_sequence"]):
        raise ValueError("P/C witness original question-source issuance chronology differs")
    framed = _bytes(visible["native_source_sha256"]) + _bytes(attack["native_attack_sha256"])
    framed += _u(value["visible_source_sequence"]) + _u(attack["source_sequence"])
    _seal(ATTACK_SOURCE_DOMAIN, framed,
          value["native_source_sha256"])
    # Repair/C/Fresh scope and conditional comparison use the visible question.
    # Only the witness source digest gains the original attack causal binding.
    return visible_ticket, {**visible, "native_source_sha256": value["native_source_sha256"]}


def _accepted_line(value, original, budget, kind):
    fields(value, ("native_line_sha256", "accepted", "line_id", "observation_id", "revision", "kind", "moves"),
           "P/C witness actual accepted line")
    role = "proposer" if kind == "repair" else "critic"
    ticket = _check_origin(value["accepted"], original, role=role, root_budget=budget)
    uint(value["line_id"])
    uint(value["observation_id"])
    revision = _positive(value["revision"])
    if value["kind"] != kind or revision <= ticket["representation"]["record_revision"]:
        raise ValueError("P/C witness accepted line kind/publication revision differs")
    framed = _bytes(value["accepted"]["seal_sha256"])
    framed += _u(value["line_id"]) + _u(value["observation_id"]) + _u(revision)
    framed += bytes((2 if kind == "repair" else 1,)) + _line(value["moves"])
    _seal("rz-rovent-pc-accepted-repair-line/1", framed, value["native_line_sha256"])
    return ticket


def _pair(value, ticket, perspective):
    fields(value, ("public_context", "authority", "root_state", "context_policy", "selected_records", "anchors",
        "model", "encoding", "precision", "frozen_epoch", "record_revision", "role", "perspective"),
        "P/C witness exact frozen Fresh pair")
    sha(value["public_context"])
    _authority(value["authority"])
    _state(value["root_state"])
    for key in ("selected_records", "anchors", "frozen_epoch", "record_revision"):
        uint(value[key])
    for key in ("context_policy", "model", "encoding"):
        sha(value[key])
    if (any(value["authority"][key] != ticket[key] for key in _AUTHORITY_FIELDS)
            or any(value[key] != ticket[key] for key in ("root_state", "context_policy", "selected_records", "anchors"))
            or any(value[key] != ticket["representation"][key]
                   for key in ("model", "encoding", "precision", "frozen_epoch", "record_revision", "role"))
            or value["perspective"] != perspective):
        raise ValueError("P/C witness frozen Fresh pair differs from its original accepted endpoint")


def _endpoint(value, original, budget, source, loaded):
    fields(value, ("native_endpoint_sha256", "accepted", "line", "pair", "model", "perspective", "raw_wdl_bits",
        "root_wdl_bits", "physical"), "P/C witness original Fresh endpoint")
    ticket = _check_origin(value["accepted"], original, role="proposer", root_budget=budget)
    _line(value["line"])
    if value["perspective"] != source["perspective"]:
        raise ValueError("P/C witness Fresh endpoint changed the original root perspective")
    _pair(value["pair"], ticket, value["perspective"])
    model = fields(value["model"], ("semantics", "model", "encoding", "precision", "model_epoch_sha256"),
                   "P/C witness original loaded model value identity")
    expected = {
        "semantics": "rz-pals-context-wdl/1;side-to-move;restricted-model-estimate;actual-prepared-input;no-cp-calibration;no-rules-proof",
        "model": "pals-native-virtual-v3:" + ":".join(loaded[key]
            for key in ("export_manifest_sha256", "weights_sha256", "backend_sha256")),
        "encoding": loaded["encoding"] + ":" + loaded["encoding_registry_sha256"],
        "precision": "fp32_tf32_off", "model_epoch_sha256": loaded["weights_sha256"],
    }
    if (model != expected or ticket["representation"]["model"] != loaded["weights_sha256"]
            or ticket["representation"]["encoding"] != loaded["encoding_sha256"]
            or ticket["representation"]["frozen_epoch"] != loaded["frozen_epoch"]):
        raise ValueError("P/C witness conditional value identity differs from the actual loaded source pins")
    raw, _ = _wdl(value["raw_wdl_bits"])
    _wdl(value["root_wdl_bits"])
    # Source is a current-root query, so an odd endpoint prefix flips its
    # side-to-move W/L into the source perspective. No CP/Rules conversion.
    expected_bits = list(reversed(value["raw_wdl_bits"])) if len(value["line"]) % 2 else value["raw_wdl_bits"]
    if value["root_wdl_bits"] != expected_bits:
        raise ValueError("P/C witness Fresh WDL did not preserve its exact root perspective")
    physical = fields(value["physical"], ("execution_epoch", "execution_sequence", "completed_nn_inputs",
        "measurement_complete", "physical_fresh"), "P/C witness original Fresh physical observation")
    if (uint(physical["execution_epoch"]) != ticket["process_epoch"]
            or _positive(physical["execution_sequence"]) == 0 or _positive(physical["completed_nn_inputs"]) == 0
            or physical["measurement_complete"] is not True or physical["physical_fresh"] is not True):
        raise ValueError("P/C witness Fresh endpoint has no original confirmed physical observation")
    framed = _bytes(value["accepted"]["seal_sha256"]) + _bytes(value["pair"]["public_context"])
    framed += _u(physical["execution_epoch"]) + _u(physical["execution_sequence"]) + _u(physical["completed_nn_inputs"])
    framed += bytes((0 if value["perspective"] == "white" else 1,)) + raw + _line(value["line"])
    _seal("rz-rovent-pc-accepted-fresh-endpoint/1", framed, value["native_endpoint_sha256"])
    return ticket


def _witness(record, value, manifest=None):
    fields(value, ("schema", "native_witness_sha256", "native_hash_kind", "event_sequence", "reason", "resolution",
        "target_known", "source", "repair", "recheck", "repaired_endpoint", "counter_endpoint"),
        "P/C witness original causal evidence")
    if (value["schema"] != WITNESS_DOMAIN or value["resolution"] not in ("supported", "refuted", "unknown", "disputed")
            or type(value["target_known"]) is not bool
            or value["target_known"] != (value["resolution"] in ("supported", "refuted"))):
        raise ValueError("P/C witness typed resolution/known mask differs")
    sequence = uint(value["event_sequence"])
    reason = _text(value["reason"], 4096).encode("utf-8")
    original, source = _source_context(record, value["source"], manifest)
    budget = source["root_budget_sha256"]
    facts = ("repair", "recheck", "repaired_endpoint", "counter_endpoint")
    for key, kind in (("repair", "repair"), ("recheck", "counterexample")):
        if value[key] is not None:
            _accepted_line(value[key], original, budget, kind)
    loaded = record["native"]["raw_retirement"]["root_admission"]["native_origin"]
    for key in ("repaired_endpoint", "counter_endpoint"):
        if value[key] is not None:
            _endpoint(value[key], original, budget, source, loaded)
    digests = [None if value[key] is None else value[key]["native_line_sha256" if key in ("repair", "recheck")
        else "native_endpoint_sha256"] for key in facts]
    if value["native_hash_kind"] == "unresolved":
        if value["resolution"] != "unknown" or value["target_known"]:
            raise ValueError("unresolved P/C witness cannot issue a known target")
        framed = _bytes(source["native_source_sha256"])
        framed += b"".join(b"\0" if item is None else b"\1" + _bytes(item) for item in digests)
        framed += _u(sequence) + _u(len(reason)) + reason
        _seal("rz-rovent-pc-divergence-unresolved/1", framed, value["native_witness_sha256"])
        return value
    if (value["native_hash_kind"] != "frozen_pair" or not sequence or any(item is None for item in digests)
            or value["resolution"] == "disputed"):
        raise ValueError("P/C witness frozen comparison requires all original completed facts")
    repair, recheck = value["repair"], value["recheck"]
    left, right = value["repaired_endpoint"], value["counter_endpoint"]
    tickets = [item["accepted"]["ticket"] for item in (repair, recheck, left, right)]
    if (not original["request_sequence"] < tickets[0]["request_sequence"] < tickets[1]["request_sequence"]
            or any(tickets[1]["request_sequence"] >= item["request_sequence"] for item in tickets[2:])
            or tickets[2]["request_sequence"] == tickets[3]["request_sequence"]
            or left["pair"] != right["pair"] or left["model"] != right["model"]
            or left["line"] != repair["moves"] or right["line"] != recheck["moves"]):
        raise ValueError("P/C witness changed actual Repair/C/Fresh chronology or frozen pair")
    fixed = source["divergence_ply"] + 1
    p_line, c_line, counter = repair["moves"], recheck["moves"], source["counter"]
    if (len(p_line) != len(c_line) or p_line[0] != source["first_move"] or c_line[0] != source["first_move"]
            or p_line[:fixed] != counter[:fixed]):
        raise ValueError("P/C witness Repair changed its fixed original first move/attack prefix")
    changed = next((i for i, (a, b) in enumerate(zip(p_line, counter)) if a != b), None)
    c_changed = next((i for i, (a, b) in enumerate(zip(c_line, p_line)) if a != b), None)
    if changed is None or changed < fixed or changed % 2 or c_changed is None or c_changed <= changed or c_changed % 2 != 1:
        raise ValueError("P/C witness requires actual P repair followed by an independent C reply")
    _, lvalues = _wdl(left["root_wdl_bits"])
    _, rvalues = _wdl(right["root_wdl_bits"])
    lvalue, rvalue = _f32_subtract(lvalues[0], lvalues[2]), _f32_subtract(rvalues[0], rvalues[2])
    expected = "refuted" if lvalue > rvalue else "supported" if lvalue < rvalue else "unknown"
    expected_reason = ("equal frozen conditional expectations" if expected == "unknown" else
                       "actual Repair and C continuation; frozen model conditional comparison")
    if value["resolution"] != expected or value["reason"] != expected_reason:
        raise ValueError("P/C witness conditional target differs from its original FP32 frozen comparison")
    framed = _bytes(source["native_source_sha256"]) + b"".join(_bytes(item) for item in digests)
    framed += _u(sequence) + bytes(({"supported": 0, "refuted": 1, "unknown": 2}[expected],))
    _seal(WITNESS_DOMAIN, framed, value["native_witness_sha256"])
    return value


def validate_pc_divergence_head(record, source, label, *, manifest=None):
    """Validate exact persisted evidence; returns no live or Rules capability."""
    from .rovent_divergence import divergence_target
    from .rovent_submitted_input import INPUT_DOMAIN, validate_input
    from .generation_data import digest

    if record.get("schema") != "rz-rovent-training-record/2" or record["input"].get("schema") != INPUT_DOMAIN:
        raise ValueError("actual P/C divergence requires the explicit submitted corpus family")
    validate_input(record["input"])
    fields(source, ("actor_id", "source", "binary_sha256", "profile_sha256", "raw_receipt", "raw_receipt_sha256",
        "observed_sequence", "anchor"), "P/C divergence original head source")
    raw = fields(source["raw_receipt"], ("schema", "kind", "input_sha256", "anchor", "observed_sequence", "witness"),
                 "P/C divergence original raw causal receipt")
    snapshot = record["input"]
    if (source["source"] != SOURCE_NAME or source["actor_id"] != record["lineage"]["actor_id"]
            or source["binary_sha256"] is not None or source["profile_sha256"] is not None
            or raw["schema"] != HEAD_DOMAIN or raw["kind"] != "pc_divergence"
            or raw["input_sha256"] != record["input_sha256"]
            or raw["anchor"] != snapshot["rules_anchor"] or source["anchor"] != raw["anchor"]
            or uint(raw["observed_sequence"]) != uint(source["observed_sequence"])
            or raw["observed_sequence"] < snapshot["capture_sequence"]
            or sha(source["raw_receipt_sha256"]) != digest(HEAD_DOMAIN, raw)):
        raise ValueError("P/C divergence source changed original input/anchor/collector chronology or raw seal")
    witness = _witness(record, raw["witness"], manifest)
    wrapped = witness["source"].get("schema") == ATTACK_SOURCE_DOMAIN
    if wrapped and raw["observed_sequence"] < witness["source"]["visible_record_evidence"]["native"]["retired_sequence"]:
        raise ValueError("P/C divergence collector receipt precedes its future visible retirement")
    if label is None:
        return witness
    divergence_target(snapshot, label)
    counter = witness["source"]["visible_source"] if wrapped else witness["source"]
    attack = witness["source"]["attack_ranked_source"] if wrapped else counter
    mapping = snapshot["attack_mapping"][attack["divergence_index"]]
    if (not witness["target_known"] or not wrapped and mapping["selected_counter_local_index"] is None
            or label["raw_evidence_sha256"] != source["raw_receipt_sha256"]
            or label["observed_sequence"] != raw["observed_sequence"]
            or label["slot"] != attack["divergence_index"] or label["attack_ply"] != attack["divergence_ply"]
            or label["counter_question_id"] != counter["accepted"]["ticket"]["question_digest"]
            or label["counter_native_view_sha256"] != counter["accepted"]["ticket"]["view_digest"]
            or label["counter_local_index"] != counter["counter_local_index"]
            or label["counter_revision"] != counter["counter_revision"]
            or label["repair_question_id"] != witness["repair"]["accepted"]["ticket"]["question_digest"]
            or label["status"] != {"supported": "supported_after_repair", "refuted": "refuted_by_repair"}.get(witness["resolution"])):
        raise ValueError("P/C divergence future target differs from the actual owner witness and original visible slot")
    return witness
