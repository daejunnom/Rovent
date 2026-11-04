"""Pure protocol/cost fixtures; simulated data never issues a live lease."""
import copy
import unittest

from rz_pals_model import rovent_utility_training_authority as authority


def expense(value):
    return dict.fromkeys(("forward_calls", "backward_calls", "samples", "wall_time_ns",
        "partial_accumulation_ns", "replay_ns", "masked_groups", "output_bytes"), value)


def evidence():
    return {"path": "evidence/final-retirement.json", "sha256": "81" * 32, "bytes": 123}


def cost(*, known=True):
    return {"schema": "rz-rovent-utility-checkpoint-cost/1", "ancestor_cost": expense(2),
        "snapshot_cost": expense(3), "post_snapshot_delta": expense(5),
        "closure_tail": {"status": "known", "delta": expense(7), "evidence": evidence()} if known else
            {"status": "unknown", "reason": "physical_completion_unknown", "evidence": None},
        "inherited_unknown_tails": [], "known_lower_bound": expense(17 if known else 10),
        "total_status": "known" if known else "lower_bound_only",
        "scope": "recorded_cost_only_not_old_process_closure_or_new_window_authority", "execution_authorized": False}


class UtilityAuthorityMetadataTests(unittest.TestCase):
    def test_json_flags_duck_types_and_arbitrary_constructor_tokens_are_not_live_authority(self):
        for candidate in ({"execution_authorized": True, "connected": True}, object(), None):
            with self.assertRaises(authority.OriginalUtilityAuthorityRequired):
                authority.require_original_utility_authority(candidate, allow_unleased=True)
        with self.assertRaises(authority.OriginalUtilityAuthorityRequired):
            authority.ParentSocketUtilityTrainerAuthority(object(), None, {}, "00" * 32, None, None)

    def test_empty_ack_and_original_completion_pin_have_exact_fields_and_types(self):
        self.assertEqual(authority._empty({}), {})
        self.assertEqual(authority._pin({"sha256": "91" * 32, "bytes": 22}), {"sha256": "91" * 32, "bytes": 22})
        self.assertEqual(authority._pin({"sha256": "91" * 32, "bytes": 65537})["bytes"], 65537)
        self.assertEqual(authority._pin({"sha256": "91" * 32, "bytes": 16 * 1024**2})["bytes"], 16 * 1024**2)
        for value in ({"sha256": "91" * 32, "bytes": True}, {"sha256": "91" * 32, "bytes": 0},
                      {"sha256": "91" * 32, "bytes": 16 * 1024**2 + 1},
                      {"sha256": "91" * 32, "bytes": 22, "authorized": True}):
            with self.assertRaises(ValueError): authority._pin(value)
        with self.assertRaises(ValueError): authority._empty({"lease": "fabricated"})

    def test_known_snapshot_post_delta_and_known_tail_are_each_counted_once(self):
        original = cost(); before = copy.deepcopy(original)
        checked = authority.validate_utility_checkpoint_cost(original)
        self.assertEqual(checked["known_lower_bound"], expense(17))
        self.assertEqual(checked["total_status"], "known")
        self.assertFalse(checked["execution_authorized"])
        self.assertEqual(original, before)
        changed = cost(); changed["known_lower_bound"] = expense(22)
        with self.assertRaises(ValueError): authority.validate_utility_checkpoint_cost(changed)

    def test_unknown_tail_preserves_lower_bound_instead_of_inventing_known_zero(self):
        checked = authority.validate_utility_checkpoint_cost(cost(known=False))
        self.assertEqual(checked["known_lower_bound"], expense(10))
        self.assertEqual(checked["total_status"], "lower_bound_only")
        for mutate in (lambda value: value.update(total_status="known"),
                       lambda value: value["closure_tail"].update(delta=expense(0)),
                       lambda value: value["closure_tail"].update(evidence={"path": "../elsewhere", "sha256": "81" * 32, "bytes": 1})):
            changed = cost(known=False); mutate(changed)
            with self.assertRaises(ValueError): authority.validate_utility_checkpoint_cost(changed)

    def test_inherited_unknowns_remain_unknown_and_cannot_be_duplicated_or_relabelled(self):
        value = cost()
        inherited = {"checkpoint_publication_sha256": "82" * 32, "execution_id": "old-utility-execution",
            "window_id": "old-utility-window", "tail": {"status": "unknown", "reason": "lost_tail", "evidence": evidence()}}
        value["inherited_unknown_tails"] = [inherited]
        value["total_status"] = "lower_bound_only"
        self.assertEqual(authority.validate_utility_checkpoint_cost(value)["inherited_unknown_tails"], [inherited])
        changed = copy.deepcopy(value); changed["inherited_unknown_tails"].append(copy.deepcopy(inherited))
        with self.assertRaises(ValueError): authority.validate_utility_checkpoint_cost(changed)
        changed = copy.deepcopy(value); changed["inherited_unknown_tails"][0]["tail"] = cost()["closure_tail"]
        with self.assertRaises(ValueError): authority.validate_utility_checkpoint_cost(changed)
        for field, replacement in (("execution_id", "relabeled-execution"), ("checkpoint_publication_sha256", "83" * 32)):
            changed = copy.deepcopy(value); other = copy.deepcopy(inherited); other[field] = replacement
            changed["inherited_unknown_tails"].append(other)
            with self.assertRaises(ValueError): authority.validate_utility_checkpoint_cost(changed)
        changed = copy.deepcopy(value); other = copy.deepcopy(inherited)
        other.update(checkpoint_publication_sha256="81" * 32, execution_id="another-execution", window_id="another-window")
        changed["inherited_unknown_tails"].append(other)
        with self.assertRaises(ValueError): authority.validate_utility_checkpoint_cost(changed)

    def test_cost_metadata_has_exact_domain_counter_types_and_no_execution_permission(self):
        for mutate in (lambda value: value.update(schema="rz-rovent-training-checkpoint-cost/1"),
                       lambda value: value.update(execution_authorized=0),
                       lambda value: value.update(execution_authorized=True),
                       lambda value: value["snapshot_cost"].update(samples=True),
                       lambda value: value["post_snapshot_delta"].update(output_bytes=-1),
                       lambda value: value["closure_tail"]["evidence"].update(path=".env"),
                       lambda value: value.update(source_authority=True)):
            changed = cost(); mutate(changed)
            with self.assertRaises(ValueError): authority.validate_utility_checkpoint_cost(changed)


if __name__ == "__main__":
    unittest.main()
