"""No-execution exact source custody and unchanged numerical metadata."""
import copy
from pathlib import Path
import tempfile
import unittest

from rz_pals_model.generation_data import digest
from rz_pals_model.rovent_utility_intervention_dataset import (HistoricalUtilityInterventionDataset,
    DATASET_DOMAIN, POOL_DOMAIN, CONTEXT_DOMAIN)
from rz_pals_model.rovent_utility_intervention_numerical import validate_intervention_numerical_parts
from rz_pals_model.rovent_utility_intervention_preparation import prepare_intervention_from_files, PREPARATION_DOMAIN
from rz_pals_model.rovent_utility_route import HistoricalUtilityRouteDataset
from test_rovent_utility_intervention_payload import intervention_case, write_case
from test_rovent_utility_intervention_files import budget


class InterventionDatasetTests(unittest.TestCase):
    def test_original_dataset_projection_has_separate_source_family_and_no_learner_permission(self):
        recipe, registered, spec, blobs = intervention_case()
        with tempfile.TemporaryDirectory() as directory:
            reference = write_case(Path(directory).resolve(), blobs)
            prepared = prepare_intervention_from_files(reference, recipe, registered, spec, read_budget=budget(4 << 20))
            parts = validate_intervention_numerical_parts(prepared)
            owner = HistoricalUtilityInterventionDataset.from_registration(reference, preparation=prepared,
                read_budget=budget(4 << 20))
            observed = owner.observation()
            self.assertEqual(observed["data_family"], "intervention_v2")
            self.assertFalse(observed["execution_authorized"])
            self.assertFalse(observed["training_target_authorized"])
            projection = owner.historical_context(owner.context_ids[0])["source_projection"]
            self.assertEqual(len(projection), 27)
            unsigned = {k: v for k, v in projection.items() if k != "context_sha256"}
            self.assertEqual(projection["context_sha256"], digest(CONTEXT_DOMAIN, unsigned))
            self.assertEqual(owner.pool_sha256, digest(POOL_DOMAIN, list(owner.context_ids)))
            expected = digest(DATASET_DOMAIN, {"data_family": "intervention_v2", "publication": owner.publication_pin,
                "durable_ack": owner.durable_ack_pin, "contexts": list(owner.context_ids), "pool_sha256": owner.pool_sha256,
                "source_registration_sha256": parts.registration_sha256})
            self.assertEqual(observed["dataset_sha256"], expected)
            with self.assertRaises(ValueError):
                HistoricalUtilityRouteDataset.from_registration(reference, preparation=prepared, read_budget=budget(4 << 20))

    def test_numerical_accessor_preserves_exact_nine_fields_without_old_prep_envelope(self):
        recipe, registered, spec, blobs = intervention_case()
        with tempfile.TemporaryDirectory() as directory:
            reference = write_case(Path(directory).resolve(), blobs)
            original = prepare_intervention_from_files(reference, recipe, registered, spec, read_budget=budget(4 << 20))
            parts = validate_intervention_numerical_parts(original)
            self.assertEqual(parts.preparation, original)
            self.assertEqual(len(parts.preparation), 9)
            self.assertNotIn("recipe_sha256", parts.preparation)
            self.assertNotIn("registration_sha256", parts.preparation)
            self.assertEqual(parts.registration_sha256,
                digest(PREPARATION_DOMAIN, {"kind": "source_registration", "value": original["registration"]}))
            changed = parts.registration
            changed["cost_epoch_sha256"] = "ff" * 32
            self.assertEqual(parts.registration, original["registration"])
            parts._original["registration"]["cost_epoch_sha256"] = "ff" * 32
            with self.assertRaises(ValueError):
                _ = parts.registration_sha256

    def test_original_custody_rejects_changed_preparation_source_and_missing_ancestor(self):
        recipe, registered, spec, blobs = intervention_case()
        with tempfile.TemporaryDirectory() as directory:
            reference = write_case(Path(directory).resolve(), blobs)
            original = prepare_intervention_from_files(reference, recipe, registered, spec, read_budget=budget(4 << 20))
            changed = copy.deepcopy(original)
            changed["intervention_source"]["role_queries"][0]["question_sha256"] = "ff" * 32
            with self.assertRaises(ValueError):
                HistoricalUtilityInterventionDataset.from_registration(reference, preparation=changed,
                    read_budget=budget(4 << 20))


if __name__ == "__main__":
    unittest.main()
