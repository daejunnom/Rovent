"""Pure coverage accounting; no model, ORT, or GPU is initialized."""
import unittest

from rz_pals_model.rovent_system_numeric_reference import _critic_divergence_coverage


class CriticDivergenceCoverageTests(unittest.TestCase):
    def test_empty_critic_head_is_not_coverage(self):
        captures = [{"role": "critic", "input": {"divergences": []}}]
        self.assertEqual(_critic_divergence_coverage(captures, []), {
            "status": "not_observed", "input_count": 0, "compared_count": 0,
        })

    def test_nonempty_head_requires_actual_same_case_comparison(self):
        captures = [
            {"role": "critic", "input": {"divergences": []}},
            {"role": "critic", "input": {"divergences": [[0.0] * 8, [1.0] * 8]}},
        ]
        self.assertEqual(_critic_divergence_coverage(captures, [])['status'], 'incomplete')
        wrong_case = [{"index": 0, "fields": {"divergence_logits": {"count": 2, "passed": True}}}]
        self.assertEqual(_critic_divergence_coverage(captures, wrong_case)['status'], 'incomplete')
        good = [{"index": 1, "fields": {"divergence_logits": {"count": 2, "passed": True}}}]
        self.assertEqual(_critic_divergence_coverage(captures, good), {
            "status": "match", "input_count": 1, "compared_count": 1,
        })
        good[0]["fields"]["divergence_logits"]["passed"] = False
        self.assertEqual(_critic_divergence_coverage(captures, good)['status'], 'mismatch')


if __name__ == "__main__":
    unittest.main()
