"""Detached /2 metadata fixtures; no NN, optimizer or source authority."""
import copy
import hashlib
import unittest

from rz_pals_model import rovent_utility_training as legacy
from rz_pals_model import rovent_utility_intervention_preparation as prepare
from rz_pals_model.rovent_utility_intervention import COST_SCOPE, canonical_bytes
from test_rovent_utility_training import membership, registration


def intervention_source():
    artifact = {"path": "registered/frozen-pc.json", "sha256": "a0" * 32,
        "bytes": 17, "source": "https://example.org/frozen-pc", "license": "MIT"}
    return {"cohort_id": "01" * 32, "profile_sha256": "02" * 32,
        "publication": {"sha256": "03" * 32, "bytes": 101},
        "durable_ack": {"sha256": "04" * 32, "bytes": 102},
        "neutral_input_sha256": "05" * 32, "neutral_target_sha256": "06" * 32,
        "neutral_context_sha256": "07" * 32, "root_rules_sha256": "08" * 32,
        "role_queries": [{"trial_sequence": number, "recipient": route,
            "question_sha256": f"{number + 8:02x}" * 32, "view_sha256": f"{number + 18:02x}" * 32,
            "full_view_sha256": f"{number + 28:02x}" * 32}
            for number, route in enumerate(("route_p", "route_c"), 1)],
        "frozen_pc_sources": [{"role": role, "artifact": copy.deepcopy(artifact),
            "model_semantics_sha256": "31" * 32, "model_identity_sha256": "32" * 32,
            "encoding_sha256": "33" * 32} for role in ("proposer", "critic")],
        "ancestor_datasets": [{"kind": "pc_submitted_input_v1", "publication_sha256": "40" * 32,
            "metadata": {name: f"{number + 65:02x}" * 32 for number, name in enumerate((
                "manifest_file_sha256", "index_file_sha256", "canonical_manifest_sha256", "canonical_index_sha256"))}}],
        "known_pair_sha256": [], "all_pair_sha256": ["50" * 32]}


def preparation_parts():
    spec = membership()
    registered = registration(spec)
    registered["cost_scope"] = COST_SCOPE
    return legacy.make_recipe(spec["optimizer_owner"]), registered, spec, intervention_source()


class InterventionPreparationTests(unittest.TestCase):
    def test_exact_nine_fields_preserve_original_role_sources_without_execution(self):
        parts = preparation_parts()
        before = copy.deepcopy(parts)
        result = prepare.prepare_intervention_metadata(*parts)
        self.assertEqual(set(result), set(prepare.PREPARATION_FIELDS))
        self.assertEqual(result["schema"], prepare.PREPARATION_DOMAIN)
        self.assertEqual(result["intervention_source"], parts[3])
        self.assertFalse(result["execution_authorized"])
        self.assertEqual(result["performed_optimizer_updates"], 0)
        self.assertEqual(prepare.validate_intervention_preparation(result), result)
        self.assertEqual(parts, before)

    def test_sha_excludes_only_itself_and_uses_array_canonical_framing(self):
        result = prepare.prepare_intervention_metadata(*preparation_parts())
        unsigned = {name: item for name, item in result.items() if name != "preparation_sha256"}
        self.assertEqual(result["preparation_sha256"], hashlib.sha256(
            canonical_bytes([prepare.PREPARATION_DOMAIN, unsigned])).hexdigest())
        other = hashlib.sha256(prepare.PREPARATION_DOMAIN.encode("ascii") + b"\0"
            + canonical_bytes(unsigned)).hexdigest()
        self.assertNotEqual(result["preparation_sha256"], other)
        for name in ("profile_sha256", "neutral_target_sha256", "root_rules_sha256"):
            changed = preparation_parts()
            changed[3][name] = "fe" * 32
            self.assertNotEqual(prepare.prepare_intervention_metadata(*changed)["preparation_sha256"],
                result["preparation_sha256"])

    def test_original_preparation_reader_does_not_accept_new_domain(self):
        result = prepare.prepare_intervention_metadata(*preparation_parts())
        with self.assertRaises(ValueError):
            legacy.validate_training_preparation(result)

    def test_source_is_closed_and_original_queries_are_contiguous_unique_and_present(self):
        mutations = (
            lambda s: s.update(execution_authorized=True),
            lambda s: s.update(role_queries=s["role_queries"][:1]),
            lambda s: s["role_queries"][0].update(trial_sequence=True),
            lambda s: s["role_queries"][1].update(trial_sequence=3),
            lambda s: s["role_queries"][1].update(recipient="route_p"),
            lambda s: s["role_queries"][0].update(authority={}),
            lambda s: s["publication"].update(path="publication.json"),
            lambda s: s["publication"].update(bytes=0),
            lambda s: s["publication"].update(bytes=65537),
            lambda s: s.update(known_pair_sha256=["51" * 32]),
            lambda s: s["frozen_pc_sources"].reverse(),
            lambda s: s["ancestor_datasets"][0].update(kind="utility_intervention"),
            lambda s: s["ancestor_datasets"][0]["metadata"].update(extra="ff" * 32),
        )
        for change in mutations:
            source = intervention_source()
            change(source)
            with self.subTest(source=source), self.assertRaises(ValueError):
                prepare.validate_intervention_source(source)

    def test_unknown_pair_metadata_remains_masked_and_three_original_arms_are_supported(self):
        source = intervention_source()
        self.assertEqual(prepare.validate_intervention_source(source)["known_pair_sha256"], [])
        source["role_queries"].append({"trial_sequence": 3, "recipient": "route_r",
            "question_sha256": "60" * 32, "view_sha256": "61" * 32, "full_view_sha256": "62" * 32})
        source["all_pair_sha256"].extend(["63" * 32, "64" * 32])
        source["known_pair_sha256"] = ["64" * 32]
        self.assertEqual(prepare.validate_intervention_source(source), source)

    def test_altered_status_authority_updates_or_extra_fields_cannot_reseal_as_valid(self):
        original = prepare.prepare_intervention_metadata(*preparation_parts())
        for change in (lambda p: p.update(status="ready"),
                lambda p: p.update(execution_authorized=True),
                lambda p: p.update(performed_optimizer_updates=1),
                lambda p: p.update(performed_optimizer_updates=False),
                lambda p: p.update(performed_optimizer_updates=0.0),
                lambda p: p.update(source_verified=True),
                lambda p: p["intervention_source"]["role_queries"][0].update(view_sha256="ef" * 32)):
            changed = copy.deepcopy(original)
            change(changed)
            with self.assertRaises(ValueError):
                prepare.validate_intervention_preparation(changed)


if __name__ == "__main__":
    unittest.main()
