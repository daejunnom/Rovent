"""A dependency observation is partial evidence, never deletion authority."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from managed_storage import write_record
from storage_catalog_dependencies import observe


class CatalogDependencyObservationTests(unittest.TestCase):
    def test_known_overlap_and_legacy_unknown_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "outputs"
            catalog.mkdir()
            target = root / "assets" / "model"
            write_record(catalog / "one.json", {"schema": "rovezero.managed-output.v1",
                         "state": "active", "source": str(root / "source"),
                         "output": str(root / "results"), "inputs": [str(root / "assets")]})
            write_record(catalog / "two.json", {"schema": "rovezero.managed-output.v1",
                         "state": "retained", "source": str(root / "other")})
            result = observe(target, output_catalogs=[catalog])
            self.assertEqual([item["field"] for item in result["matches"]], ["inputs"])
            self.assertEqual(result["matches"][0]["state"], "active")
            self.assertEqual(result["unknown"][0]["reason"], "inputs_not_recorded_or_invalid")
            self.assertFalse(result["deletion_authorized"])
            self.assertEqual(result["input_completeness"], "unknown")

    def test_build_source_overlap_and_unavailable_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slot = root / "build" / "slots" / "slot"
            slot.mkdir(parents=True)
            write_record(slot / "record.json", {"schema": "rovezero.managed-build.v1",
                         "state": "complete", "source": str(root / "assets")})
            result = observe(root / "assets" / "bin", build_catalogs=[root / "build", root / "missing"])
            self.assertEqual(result["matches"][0]["field"], "source")
            self.assertIn("catalog_unavailable_or_invalid", [item["reason"] for item in result["unknown"]])
            self.assertFalse(result["catalogs_locked_together"])


if __name__ == "__main__":
    unittest.main()
