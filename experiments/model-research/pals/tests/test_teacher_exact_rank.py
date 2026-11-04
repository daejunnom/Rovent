"""Codec-only tests; these JSON values never create real teacher authority."""
import copy
import unittest

from rz_pals_model import generation_data
from rz_pals_model.teacher_exact_rank import DOMAIN, validate_teacher_exact_rank_observation


def _fixture():
    identity = {name: char * 64 for name, char in (
        ("input_sha256", "a"), ("view_sha256", "b"), ("rules_state_sha256", "c"),
        ("rules_history_sha256", "d"), ("input_lock_sha256", "e"),
        ("evaluation_namespace_sha256", "f"), ("root_restriction_sha256", "1"),
        ("teacher_binary_sha256", "2"), ("teacher_profile_sha256", "3"))}
    identity.update(process_id=7, process_group=7, process_start_ticks=8,
                    execution_epoch=9, clock_epoch=9, original_deadline_ns=23, root_generation=1, collection_owner=1,
                    job_sequence=2, request_id=3, requested_multipv=2, root_moves=[64, 128])
    observation = dict(schema=DOMAIN, policy="completed_exact_rank_set", identity=copy.deepcopy(identity),
                       observation_sequence=5,
                       opinions=[dict(rank=1, depth=8, bound="ExactReported", pv=[64]),
                                 dict(rank=2, depth=8, bound="ExactReported", pv=[128])],
                       final_bestmove=128, final_bestmove_matches_rank1=False,
                       go_ns=10, bestmove_ns=20, ready_ns=21, returned_ns=22,
                       cancel_sequence=0, yield_sequence=0, stop_sent=False)
    return observation, identity, [64, 128]


class TeacherExactRankTests(unittest.TestCase):
    def test_exact_rank_is_opt_in_and_preserves_changed_final_bestmove(self):
        observation, identity, legal = _fixture()
        self.assertIsNone(validate_teacher_exact_rank_observation(observation, identity, legal))
        # The old evidence registry cannot silently adopt this new domain.
        with self.assertRaises(ValueError):
            generation_data.validate_evidence_receipt(generation_data.digest(DOMAIN, observation), observation)

    def test_exact_rank_rejects_changed_scope_bound_and_incomplete_lifecycle(self):
        observation, identity, legal = _fixture()
        mutations = [
            lambda row: row["identity"].__setitem__("request_id", 4),
            lambda row: row["opinions"][0].__setitem__("bound", "Lower"),
            lambda row: row["opinions"][1].__setitem__("depth", 9),
            lambda row: row["opinions"][1].__setitem__("pv", [64]),
            lambda row: row.__setitem__("cancel_sequence", 1),
            lambda row: row.__setitem__("returned_ns", 23),
            lambda row: row.__setitem__("final_bestmove_matches_rank1", True),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                invalid = copy.deepcopy(observation)
                mutate(invalid)
                with self.assertRaises(ValueError):
                    validate_teacher_exact_rank_observation(invalid, identity, legal)

    def test_root_moves_wire_key_is_required_even_when_null(self):
        observation, identity, legal = _fixture()
        observation["identity"]["root_moves"] = None
        identity["root_moves"] = None
        self.assertIsNone(validate_teacher_exact_rank_observation(observation, identity, legal))
        del observation["identity"]["root_moves"]
        del identity["root_moves"]
        with self.assertRaises(ValueError):
            validate_teacher_exact_rank_observation(observation, identity, legal)
