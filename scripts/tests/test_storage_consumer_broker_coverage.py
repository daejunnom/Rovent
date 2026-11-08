"""Temporary local registry fixtures; not GPU or child-read evidence."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from managed_storage import StorageError
from storage_consumer_broker_coverage import pin_descriptor_coverage
from storage_consumer_registry import ConsumerRegistry


@unittest.skipUnless(os.name == "posix", "POSIX no-follow fixtures required")
class DescriptorCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = ConsumerRegistry.initialize(self.root / "registry", scope="synthetic-coverage")
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.descriptor = self.inputs / "descriptor.json"
        self.descriptor.write_text("{}", encoding="ascii")

    def test_directory_coverage_and_descriptor_replacement(self):
        lease = self.registry.register([self.inputs])
        with self.registry.guard() as guard:
            with pin_descriptor_coverage(guard, [lease], self.descriptor) as pin:
                pin.verify()
                replacement = self.inputs / "replacement"
                replacement.write_text("{}", encoding="ascii")
                os.replace(replacement, self.descriptor)
                with self.assertRaises(StorageError):
                    pin.verify()

    def test_exact_file_coverage_under_original_guard(self):
        lease = self.registry.register([self.descriptor])
        with self.registry.guard() as guard:
            # A second registry.guard() here would fail on the original lock.
            with pin_descriptor_coverage(guard, [lease], self.descriptor) as pin:
                pin.verify()

    def test_future_path_cannot_prove_existing_descriptor(self):
        self.descriptor.unlink()
        lease = self.registry.register([], future_inputs=[self.descriptor])
        self.descriptor.write_text("{}", encoding="ascii")
        with self.registry.guard() as guard:
            with self.assertRaises(StorageError):
                pin_descriptor_coverage(guard, [lease], self.descriptor)

    def test_replaced_registered_directory_is_rejected(self):
        lease = self.registry.register([self.inputs])
        self.inputs.rename(self.root / "old-inputs")
        self.inputs.mkdir()
        self.descriptor.write_text("{}", encoding="ascii")
        with self.registry.guard() as guard:
            with self.assertRaises(StorageError):
                pin_descriptor_coverage(guard, [lease], self.descriptor)

    def test_different_registry_or_changed_lease_is_rejected(self):
        lease = self.registry.register([self.inputs])
        other = ConsumerRegistry.initialize(self.root / "other-registry", scope="synthetic-other")
        with other.guard() as guard:
            with self.assertRaises(StorageError):
                pin_descriptor_coverage(guard, [lease], self.descriptor)
        original = lease.row
        lease.row = dict(original, pid=original["pid"] + 1)
        with self.registry.guard() as guard:
            with self.assertRaises(StorageError):
                pin_descriptor_coverage(guard, [lease], self.descriptor)

    def test_symlink_nonregular_and_secret_names_are_rejected(self):
        lease = self.registry.register([self.inputs])
        alias = self.inputs / "alias"
        alias.symlink_to(self.descriptor)
        with self.registry.guard() as guard:
            for target in (alias, self.inputs, self.inputs / ".env", self.inputs / "api-key.json"):
                with self.subTest(target=target), self.assertRaises(StorageError):
                    pin_descriptor_coverage(guard, [lease], target)


if __name__ == "__main__":
    unittest.main()
