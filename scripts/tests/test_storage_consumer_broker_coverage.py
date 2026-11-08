"""Temporary local registry fixtures; not GPU or child-read evidence."""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from managed_storage import StorageError
from storage_consumer_broker_coverage import DescriptorCloseUnknown, pin_descriptor_coverage
from storage_consumer_broker_state import BrokerStateError, ConsumerBrokerState
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

    def test_verification_close_unknown_keeps_primary_and_closes_other_fds(self):
        import storage_consumer_broker_coverage as coverage
        lease = self.registry.register([self.inputs])
        with self.registry.guard() as guard:
            pin = pin_descriptor_coverage(guard, [lease], self.descriptor)
            source_fd, target_fd = pin._source_fd, pin._target_fd
            original_identity, original_close = coverage._identity, os.close
            target_identity = (self.descriptor.stat().st_dev, self.descriptor.stat().st_ino)
            close_failures = []

            def fail_original_check(fd, *, directory):
                if not directory and fd != target_fd:
                    raise ValueError("original verification failure")
                return original_identity(fd, directory=directory)

            def uncertain_close(fd):
                info = os.fstat(fd)
                is_temporary_target = ((info.st_dev, info.st_ino) == target_identity
                                       and fd != target_fd)
                original_close(fd)
                if is_temporary_target and not close_failures:
                    close_failures.append(fd)
                    raise OSError("injected close outcome unknown")

            with patch.object(coverage, "_identity", side_effect=fail_original_check), \
                 patch.object(coverage.os, "close", side_effect=uncertain_close):
                with self.assertRaisesRegex(ValueError, "original verification failure") as raised:
                    with pin:
                        pin.verify()
            self.assertTrue(getattr(raised.exception, "descriptor_close_unknown", False))
            self.assertTrue(pin.close_unknown)
            self.assertEqual(len(close_failures), 1)
            for fd in (source_fd, target_fd):
                with self.assertRaises(OSError):
                    os.fstat(fd)
            with self.assertRaises(DescriptorCloseUnknown):
                pin.close()  # Never retry an FD whose close outcome is unknown.

    def test_broker_preserves_lease_and_original_error_after_verify_close_unknown(self):
        import storage_consumer_broker_coverage as coverage
        state = ConsumerBrokerState(self.registry.guard, lambda: None, lambda: None)
        state.begin(1, deadline=time.monotonic() + 10)
        state.commit(1, [self.inputs])
        state.begin(2, deadline=time.monotonic() + 10, descriptor=self.descriptor)
        pin = state._descriptor_pin
        original_close = os.close
        target_identity = (self.descriptor.stat().st_dev, self.descriptor.stat().st_ino)
        close_failures = []

        def uncertain_close(fd):
            info = os.fstat(fd)
            is_temporary_target = ((info.st_dev, info.st_ino) == target_identity
                                   and fd != pin._target_fd)
            original_close(fd)
            if is_temporary_target and not close_failures:
                close_failures.append(fd)
                raise OSError("injected close outcome unknown")

        with patch.object(coverage.os, "close", side_effect=uncertain_close):
            with self.assertRaises(DescriptorCloseUnknown) as raised:
                state.commit(2, [])
        self.assertTrue(getattr(raised.exception, "descriptor_close_unknown", False))
        self.assertTrue(state.descriptor_close_unknown)
        self.assertFalse(state.lock_release_unknown)
        with self.assertRaises(BrokerStateError):
            state.finish(child_never_dispatched=False, tree_gone=True)
        # The original guard was released and the consumer row stays active.
        with self.registry.guard() as guard:
            self.assertTrue(guard.conflicts([self.descriptor])["conflicts"])

    def test_pin_constructor_close_unknown_preserves_broker_lease(self):
        import storage_consumer_broker_coverage as coverage
        state = ConsumerBrokerState(self.registry.guard, lambda: None, lambda: None)
        state.begin(1, deadline=time.monotonic() + 10)
        state.commit(1, [self.inputs])
        original_close = os.close
        identity = (self.descriptor.stat().st_dev, self.descriptor.stat().st_ino)
        close_failures = []

        def uncertain_close(fd):
            info = os.fstat(fd)
            original_close(fd)
            if (info.st_dev, info.st_ino) == identity and not close_failures:
                close_failures.append(fd)
                raise OSError("injected close outcome unknown")

        with patch.object(coverage.os, "close", side_effect=uncertain_close):
            with self.assertRaises(DescriptorCloseUnknown):
                state.begin(2, deadline=time.monotonic() + 10, descriptor=self.descriptor)
        self.assertTrue(state.descriptor_close_unknown)
        with self.assertRaises(BrokerStateError):
            state.finish(child_never_dispatched=False, tree_gone=True)
        with self.registry.guard() as guard:
            self.assertTrue(guard.conflicts([self.descriptor])["conflicts"])

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
