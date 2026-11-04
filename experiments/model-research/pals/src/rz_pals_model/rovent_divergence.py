"""Submitted attack-slot targets; original owner evidence is checked separately.

The slot and proposal-relative ply belong to the sealed past input. A counter
reference belongs to its explicitly named question/native-view namespace and
must never be inserted into that past view. This pure value function does not
prove an actual Repair, fresh comparison, Rules result or execution capability.
"""
from __future__ import annotations

from .generation_data import fields, sha, uint
from .rovent_submitted_input import validate_input

TARGET_DOMAIN = "rz-rovent-divergence-target/1"
SOURCE = "rovent_resolved_pc_divergence"
TARGET_FIELDS = ("schema", "observed_sequence", "question_id", "view_sha256", "raw_evidence_sha256", "source",
    "slot", "attack_ply", "counter_question_id", "counter_native_view_sha256", "counter_local_index",
    "counter_revision", "repair_question_id", "status")


def divergence_target(snapshot, label):
    validate_input(snapshot)
    fields(label, TARGET_FIELDS, "submitted conditional divergence target")
    slot = uint(label["slot"], 65535)
    uint(label["attack_ply"], 65535)
    if (label["schema"] != TARGET_DOMAIN or label["source"] != SOURCE or snapshot["role"] != "critic"
            or uint(label["observed_sequence"]) < snapshot["capture_sequence"]
            or label["question_id"] != snapshot["question_id"] or label["view_sha256"] != snapshot["view_sha256"]
            or slot >= len(snapshot["attack_mapping"])):
        raise ValueError("resolved divergence must join its original submitted attack slot and chronology")
    for name in ("raw_evidence_sha256", "counter_question_id", "counter_native_view_sha256", "repair_question_id"):
        sha(label[name])
    uint(label["counter_local_index"], 63)
    if not uint(label["counter_revision"]):
        raise ValueError("resolved divergence requires its original positive counter revision")
    mapping = snapshot["attack_mapping"][slot]
    if mapping["slot"] != slot or mapping["attack_ply"] != label["attack_ply"]:
        raise ValueError("resolved divergence changed its original attack position")
    if mapping["selected_counter_local_index"] is not None and (
            label["counter_question_id"] != snapshot["question_id"]
            or label["counter_local_index"] != mapping["selected_counter_local_index"]
            or label["counter_revision"] != mapping["selected_counter_revision"]):
        raise ValueError("visible source counter differs from the sealed submitted mapping")
    if label["status"] not in ("supported_after_repair", "refuted_by_repair"):
        raise ValueError("unresolved or disputed divergence cannot become a target")
    return 1.0 if label["status"] == "supported_after_repair" else 0.0
