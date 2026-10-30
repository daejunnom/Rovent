"""New-window codec/read ownership fixtures, never a real learner execution."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
import zipfile

from test_rovent_training import NeuralImportsForbidden, membership_fixture, recipe_fixture
from test_rovent_training_window_v2 import spec_fixture

_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_recovered_training_window as recovered
    from rz_pals_model import rovent_training_window as legacy
    from rz_pals_model.rovent_recovered_training_authority import RecoveredParentSocketTrainerAuthority
    from rz_pals_model.rovent_training_bootstrap import PinnedReadBudget
    from rz_pals_model.generation_export import _hash_file
    from rz_pals_model.rovent_tensor_training import _load_pinned_tensor_file
finally:
    sys.meta_path.remove(_guard)


class RecoveredLearnerConsumerTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)
        self.recipe = recipe_fixture(membership_fixture())
        self.spec = spec_fixture(self.recipe)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_publication_actual_fields_include_and_join_separate_training_epoch(self):
        # Independent Rust DTO field roster: do not manufacture this fixture
        # from the decoder's own _PUBLICATION_FIELDS constant.
        names = ("schema", "execution_id", "window_id", "collection_source_sha256",
            "collection_runtime_library_sha256", "trainer_source_sha256", "trainer_python_executable_sha256",
            "trainer_dependency_set_sha256", "training_model_semantics_sha256", "initialization_source",
            "collection_model_epoch", "collection_context_epoch", "training_model_epoch", "recipe_sha256", "stage_id",
            "training_profile_sha256", "membership_sha256", "current_dataset_publication_sha256",
            "ancestor_dataset_publication_sha256", "tensor", "checkpoint_retirement", "update_receipts", "terminal_receipt",
            "original_recipe", "original_membership", "original_window_spec", "parent_publication", "boundary",
            "recorded_tail_cost", "recorded_closure")
        publication = dict.fromkeys(names)
        for name in ("collection_source_sha256", "collection_runtime_library_sha256", "collection_model_epoch",
                     "collection_context_epoch", "training_model_epoch", "trainer_source_sha256",
                     "trainer_python_executable_sha256", "trainer_dependency_set_sha256", "training_model_semantics_sha256",
                     "window_id", "stage_id", "recipe_sha256"):
            publication[name] = self.spec[name]
        publication.update(schema=recovered.PUBLICATION_DOMAIN,
            training_profile_sha256=recovered.training_profile_digest(self.spec))
        before = copy.deepcopy(publication)
        self.assertIs(recovered.validate_checkpoint_publication_metadata(publication, original_window=self.spec), publication)
        self.assertEqual(publication, before)
        missing = {name: value for name, value in publication.items() if name != "training_model_epoch"}
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            recovered.validate_checkpoint_publication_metadata(missing)
        for epoch in (0, self.spec["collection_model_epoch"]):
            with self.subTest(epoch=epoch), self.assertRaises(ValueError):
                recovered.validate_checkpoint_publication_metadata({**publication, "training_model_epoch": epoch},
                    original_window=self.spec)
        with self.assertRaisesRegex(ValueError, "identity or epoch"):
            recovered.validate_checkpoint_publication_metadata({**publication,
                "trainer_python_executable_sha256": self.spec["collection_runtime_library_sha256"]}, original_window=self.spec)
        zero = {**self.spec, "training_model_epoch": 0}
        zero_publication = {**publication, "training_model_epoch": 0,
            "training_profile_sha256": recovered.training_profile_digest(zero)}
        self.assertEqual(recovered.validate_checkpoint_publication_metadata(zero_publication,
            original_window=zero)["training_model_epoch"], 0)

    def owner(self, spec=None):
        spec = self.spec if spec is None else spec
        admission = {"corpus_sha256": "a" * 64}
        journal = recovered.initial_training_journal(self.recipe, spec, admission)
        return SimpleNamespace(recipe=self.recipe, stage_id=spec["stage_id"],
            completed=dict.fromkeys(legacy.ROLES, 0), inherited_input_sha256=[], inherited_origin_sha256=[],
            samplers=journal), admission

    def test_journal_keeps_collection_and_training_epochs_in_a_separate_domain(self):
        owner, admission = self.owner()
        before = copy.deepcopy(owner.samplers)
        checked = recovered._journal(owner, self.spec, admission)
        self.assertEqual(checked["schema"], "rz-rovent-recovered-learner-journal/1")
        self.assertEqual(checked["phase_inputs"]["pc"]["model_epoch"], 2)
        self.assertEqual(checked["phase_inputs"]["pc"]["training_model_epoch"], 5)
        self.assertEqual(owner.samplers, before)
        owner.samplers["phase_inputs"]["pc"]["training_model_epoch"] = 2
        with self.assertRaisesRegex(ValueError, "collection changed"):
            recovered._journal(owner, self.spec, admission)

    def test_resume_export_preparation_never_opens_or_materializes_checkpoint_bytes(self):
        for function in (recovered.stage_rovent_resume, recovered.stage_rovent_export):
            with self.subTest(function=function.__name__):
                prepared = function("/not-present/checkpoint.pt", "1" * 64, None, self.recipe,
                    membership_fixture(), None, self.spec, max_file_bytes=100)
                self.assertFalse(prepared["actual_tensor_bytes_loaded"])
                self.assertFalse(prepared["caller_state_mutated"])
                self.assertFalse(prepared["execution_authorized"])

    def test_historical_metadata_cannot_construct_current_execution_authority(self):
        with self.assertRaisesRegex(ValueError, "kernel-authenticated"):
            recovered.stage_rovent_resume("/not-present/checkpoint.pt", "1" * 64, None, self.recipe,
                membership_fixture(), None, self.spec, max_file_bytes=100,
                execution_authority={"execution_authorized": True, "historical_integrity_only": True}, execute=True)
        with self.assertRaisesRegex(ValueError, "original file-read budget"):
            RecoveredParentSocketTrainerAuthority.connect_registered({}, raw_startup=b"{}",
                expected_sha256="1" * 64, execute=True)

    def test_explicit_fork_preparation_preserves_parent_and_requires_fresh_child_state(self):
        spec = copy.deepcopy(self.spec)
        target = next(stage for stage in self.recipe.to_dict()["stages"] if stage["parent_stage_id"] == "common")
        spec["stage_id"] = target["stage_id"]
        prepared = recovered.stage_rovent_fork("/not-present/checkpoint.pt", "1" * 64, None,
            self.recipe, membership_fixture(), None, spec, max_file_bytes=100)
        self.assertEqual(prepared["source_stage_id"], "common")
        self.assertTrue(prepared["reset_optimizer_state"])
        self.assertFalse(prepared["execution_authorized"])

    def test_legacy_journal_is_not_implicitly_retagged_into_a_recovered_window(self):
        owner, admission = self.owner()
        owner.samplers["schema"] = legacy.JOURNAL_DOMAIN
        with self.assertRaisesRegex(ValueError, "pinned corpus/profile/epoch"):
            recovered._journal(owner, self.spec, admission)
        old_spec = copy.deepcopy(self.spec)
        old_spec["schema"] = "rz-rovent-train-window/1"
        with self.assertRaisesRegex(ValueError, "legacy conversion"):
            recovered.initial_training_journal(self.recipe, old_spec, admission)

    def test_recovered_rpc_uses_its_own_domains_without_a_real_socket_or_process(self):
        # This is only a wire codec fixture, not a live peer/permission claim.
        instance = object.__new__(RecoveredParentSocketTrainerAuthority)
        response = {"schema": instance.ACK_SCHEMA, "window_id": "new-window", "sequence": 1,
            "operation": "begin_window", "accepted": True, "payload": {}}
        class Stream:
            def settimeout(self, value):
                if not 0 < value <= 30:
                    raise AssertionError("unbounded observation timeout")
            def sendall(self, raw):
                self.sent = json.loads(raw)
            def recv(self, count):
                result, self.remaining = self.remaining[:count], self.remaining[count:]
                return result
        stream = Stream()
        stream.remaining = (json.dumps(response) + "\n").encode()
        instance._socket, instance._clock = stream, lambda: 1
        instance._startup = {"window_id": "new-window", "execution_until_ns": 1000, "whole_until_ns": 2000}
        instance._startup_sha256, instance._sequence, instance._failed = "a" * 64, 0, False
        instance._lock = threading.RLock()
        self.assertEqual(instance._rpc("begin_window", {}), {})
        self.assertEqual(stream.sent["schema"], "rz-rovent-recovered-trainer-rpc/1")
        self.assertNotEqual(stream.sent["schema"], "rz-rovent-original-trainer-rpc/1")

    def test_pinned_hash_charges_actual_blocks_and_preserves_failed_reservation(self):
        with tempfile.TemporaryDirectory(prefix="rz-recovered-hash-") as directory:
            path = Path(directory) / "source.bin"
            path.write_bytes(b"bounded-source")
            budget = PinnedReadBudget(1000, until_ns=100, clock=lambda: 1, strict_growth_credit=True)
            identity, size = _hash_file(path, 1000, read_accounting=budget)
            self.assertEqual(identity, hashlib.sha256(b"bounded-source").hexdigest())
            self.assertEqual(budget.observation(), {"charged_bytes": size, "unknown": False, "maximum_bytes": 1000})
            class BrokenSnapshot:
                def write(self, _block):
                    raise OSError("synthetic owned snapshot write failure")
            failed = PinnedReadBudget(1000, until_ns=100, clock=lambda: 1, strict_growth_credit=True)
            with self.assertRaises(OSError):
                _hash_file(path, 1000, snapshot=BrokenSnapshot(), read_accounting=failed)
            self.assertEqual(failed.observation()["charged_bytes"], size)
            self.assertTrue(failed.observation()["unknown"])

    def test_tensor_loader_reads_a_snapshot_under_new_output_without_importing_torch(self):
        with tempfile.TemporaryDirectory(prefix="rz-recovered-snapshot-") as directory:
            root = Path(directory)
            archive = root / "historical"
            archive.mkdir()
            snapshots = root / "new-output"
            snapshots.mkdir()
            source = archive / "base.pt"
            with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_STORED) as stream:
                stream.writestr("fixture.txt", b"synthetic-not-a-tensor")
            raw = source.read_bytes()
            budget = PinnedReadBudget(10000, until_ns=100, clock=lambda: 1, strict_growth_credit=True)
            class NoTensorDecoder:
                @staticmethod
                def load(stream, *, map_location, weights_only):
                    if map_location != "cpu" or weights_only is not True or stream.read() != raw:
                        raise AssertionError("source pin/snapshot decoder boundary changed")
                    return {"fixture_only": True}
            payload, size = _load_pinned_tensor_file(source, hashlib.sha256(raw).hexdigest(), len(raw), NoTensorDecoder,
                read_accounting=budget, snapshot_directory=snapshots)
            self.assertEqual(payload, {"fixture_only": True})
            self.assertEqual(budget.spent, size)
            self.assertEqual([path.name for path in archive.iterdir()], ["base.pt"])
            self.assertFalse(budget.observation()["unknown"])


if __name__ == "__main__":
    unittest.main()
