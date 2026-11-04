"""Separated raw identity fixtures; no imports or execution of a neural runtime."""
import copy
import sys
import unittest

from test_rovent_training import NeuralImportsForbidden, membership_fixture, recipe_fixture
from unittest import mock

_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_training_window_v2 as window
    from rz_pals_model.generation_training import OPTIMIZER
finally:
    sys.meta_path.remove(_guard)


def dependency_endpoints_fixture():
    return [{"name": "torch._C.__file__", "sha256": "6" * 64, "bytes": 20},
            {"name": "torch.__file__", "sha256": "7" * 64, "bytes": 30}]


def spec_fixture(recipe):
    registered = recipe.to_dict()
    return {"schema": window.WINDOW_DOMAIN, "window_id": "window-one", "recipe_sha256": recipe.sha256,
        "stage_id": "common", "collection_source_sha256": registered["source_sha256"],
        "collection_runtime_library_sha256": "1" * 64,
        "collection_model_semantics_sha256": registered["model_semantics_sha256"],
        "collection_model_epoch": 2, "collection_context_epoch": 3,
        "trainer_source_sha256": "2" * 64, "trainer_python_executable_sha256": "3" * 64,
        "trainer_dependency_set_sha256": window.trainer_dependency_set_digest(dependency_endpoints_fixture()),
        "trainer_dependency_scope": window.DEPENDENCY_SCOPE,
        "training_model_semantics_sha256": registered["model_semantics_sha256"], "training_model_epoch": 5,
        "nnuec_collection_model_sha256": "5" * 64,
        "nnuec_semantics": "board800-exactb192-causal-line-exactc256-fp32-v1",
        "phase": "pc", "frozen_pc_sha256": None, "device": "cpu", "precision": "fp32", "tf32": False,
        "deterministic": True, "cpu_threads": 2, "interop_threads": 1, "microbatch": 1, "accumulation": 8,
        "optimizer": copy.deepcopy(OPTIMIZER), "sampling_seed": 31, "training_seed": 31,
        "loss_weights": {"policy": 1.0, "wdl": 1.0, "divergence": 1.0, "task": 1.0},
        "nnuec_loss": {"loss": "wdl_cross_entropy", "target": "same_exact_rules_anchor",
            "normalization": "valid_rows", "weight": 0.5},
        "max_updates": 2, "max_groups": 8, "max_samples": 64, "max_wall_ns": 10**9,
        "max_output_bytes": 1024**2, "checkpoint_interval": 2}


def manifest_fixture(spec):
    return {"recipe_sha256": spec["recipe_sha256"], "stage_id": spec["stage_id"],
        "source_sha256": spec["collection_source_sha256"], "runtime_sha256": spec["collection_runtime_library_sha256"],
        "model_semantics_sha256": spec["collection_model_semantics_sha256"],
        "model_epoch": spec["collection_model_epoch"], "context_epoch": spec["collection_context_epoch"]}


class WindowV2IdentityTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)
        self.recipe = recipe_fixture(membership_fixture())
        self.spec = spec_fixture(self.recipe)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_distinct_ort_python_torch_inventory_and_source_pins_are_accepted(self):
        self.assertEqual(window.validate_window_spec(self.recipe, self.spec), self.spec)
        self.assertEqual(len({self.spec[name] for name in ("collection_source_sha256",
            "collection_runtime_library_sha256", "trainer_source_sha256", "trainer_python_executable_sha256",
            "trainer_dependency_set_sha256")}), 5)
        checked = window.validate_collection_binding(self.recipe, self.spec, manifest_fixture(self.spec))
        self.assertFalse(checked["execution_authorized"])

    def test_corpus_join_uses_collection_runtime_and_never_python_executable(self):
        manifest = manifest_fixture(self.spec)
        manifest["runtime_sha256"] = self.spec["trainer_python_executable_sha256"]
        with self.assertRaisesRegex(ValueError, "collection namespace"):
            window.validate_collection_binding(self.recipe, self.spec, manifest)

    def test_source_and_model_semantics_have_explicit_recipe_joins(self):
        for name in ("collection_source_sha256", "training_model_semantics_sha256"):
            changed = copy.deepcopy(self.spec)
            changed[name] = "a" * 64
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "registered recipe"):
                window.validate_window_spec(self.recipe, changed)
        changed = copy.deepcopy(self.spec)
        changed["collection_model_semantics_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "explicitly equal"):
            window.validate_window_spec(self.recipe, changed)

    def test_epoch_projection_uses_collection_epoch_without_relabelling_training_state(self):
        manifest = manifest_fixture(self.spec)
        manifest["model_epoch"] = self.spec["training_model_epoch"]
        with self.assertRaisesRegex(ValueError, "collection namespace"):
            window.validate_collection_binding(self.recipe, self.spec, manifest)
        for name in ("collection_model_epoch", "collection_context_epoch", "training_model_epoch"):
            changed = copy.deepcopy(self.spec)
            changed[name] = True
            with self.subTest(name=name), self.assertRaises(ValueError):
                window.validate_window_spec(self.recipe, changed)

    def test_old_runtime_field_or_domain_is_not_automatically_converted(self):
        changed = copy.deepcopy(self.spec)
        changed["runtime_sha256"] = changed.pop("collection_runtime_library_sha256")
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            window.validate_window_spec(self.recipe, changed)
        changed = copy.deepcopy(self.spec)
        changed["schema"] = "rz-rovent-train-window/1"
        with self.assertRaisesRegex(ValueError, "legacy conversion"):
            window.validate_window_spec(self.recipe, changed)

    def test_declared_dependency_pins_cannot_claim_all_loaded_native_libraries(self):
        changed = copy.deepcopy(self.spec)
        changed["trainer_dependency_scope"] = "all_transitive_libraries_verified"
        with self.assertRaisesRegex(ValueError, "transitive loaded runtime"):
            window.validate_window_spec(self.recipe, changed)

    def test_trainer_binding_uses_separate_source_executable_and_endpoint_files(self):
        endpoints = dependency_endpoints_fixture()
        checked = window.validate_trainer_binding(self.recipe, self.spec,
            source_sha256=self.spec["trainer_source_sha256"],
            python_executable_sha256=self.spec["trainer_python_executable_sha256"], dependency_endpoints=endpoints)
        self.assertEqual(checked["dependency_scope"], "declared_pinned_endpoints_only")
        self.assertEqual(checked["transitive_loaded_native_mappings"], "unknown")
        self.assertFalse(checked["execution_authorized"])
        for pin in ("collection_runtime_library_sha256", "trainer_source_sha256"):
            with self.subTest(pin=pin), self.assertRaisesRegex(ValueError, "executable or dependency"):
                window.validate_trainer_binding(self.recipe, self.spec,
                    source_sha256=self.spec["trainer_source_sha256"],
                    python_executable_sha256=self.spec[pin], dependency_endpoints=endpoints)

    def test_dependency_inventory_requires_sorted_unique_actual_torch_endpoint_names(self):
        endpoints = dependency_endpoints_fixture()
        variants = [list(reversed(endpoints)), [endpoints[0], endpoints[0]],
            [endpoints[0], {**endpoints[1], "name": "unknown.module"}],
            [endpoints[0], {**endpoints[1], "bytes": True}],
            [endpoints[0], {**endpoints[1], "bytes": 0}]]
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                window.trainer_dependency_set_digest(value)

    def test_fixed_profile_includes_distinct_runtime_ownership_and_explicit_loss(self):
        renewed = copy.deepcopy(self.spec)
        renewed.update(window_id="window-two", max_wall_ns=2 * 10**9, max_updates=1)
        self.assertEqual(window.training_profile_digest(renewed), window.training_profile_digest(self.spec))
        same_coefficients = copy.deepcopy(self.spec)
        same_coefficients["loss_weights"]["policy"] = 1
        self.assertEqual(window.training_profile_digest(same_coefficients), window.training_profile_digest(self.spec))
        captured = []
        with mock.patch.object(window, "digest", side_effect=lambda domain, payload: captured.append((domain, payload)) or "0" * 64):
            window.training_profile_digest(self.spec)
        self.assertEqual(captured[0][0], "rz-rovent-fixed-training-profile/2")
        self.assertEqual(captured[0][1]["algorithm"], "coefficient-f64-bits-v1")
        encoded = captured[0][1]["window"]
        self.assertEqual(encoded["optimizer"]["epsilon"], "3e45798ee2308c3a")
        self.assertEqual(encoded["optimizer"]["clip_norm"], "3ff0000000000000")
        self.assertEqual(encoded["nnuec_loss"]["weight"], "3fe0000000000000")
        self.assertEqual(encoded["cpu_threads"], 2)
        for name in ("collection_runtime_library_sha256", "trainer_source_sha256",
                     "trainer_python_executable_sha256", "trainer_dependency_set_sha256"):
            changed = copy.deepcopy(self.spec)
            changed[name] = "a" * 64
            with self.subTest(name=name):
                self.assertNotEqual(window.training_profile_digest(changed), window.training_profile_digest(self.spec))
        for bad in (None, True, 10**1000, float("inf"), float("nan")):
            unresolved = copy.deepcopy(self.spec)
            unresolved["nnuec_loss"]["weight"] = bad
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "explicitly resolved"):
                window.validate_window_spec(self.recipe, unresolved)
        oversized = copy.deepcopy(self.spec)
        oversized["loss_weights"]["policy"] = 10**1000
        with self.assertRaisesRegex(ValueError, "finite nonnegative"):
            window.validate_window_spec(self.recipe, oversized)

    def test_controls_reject_unbounded_work_or_implicit_device_fallback(self):
        for name, value in (("max_wall_ns", 0), ("max_updates", True), ("max_samples", 7),
                            ("device", "cuda:01"), ("tf32", True)):
            changed = copy.deepcopy(self.spec)
            changed[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                window.validate_window_spec(self.recipe, changed)


if __name__ == "__main__":
    unittest.main()
