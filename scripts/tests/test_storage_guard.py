"""Pure boot-identity contracts; no real registry, process or recovery calls."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import storage_guard as guard


class WindowsBootIdentityTests(unittest.TestCase):
    def registry(self, value=37, kind=4):
        key = MagicMock()
        key.__enter__.return_value = key
        return SimpleNamespace(KEY_READ=0x20019, KEY_WOW64_64KEY=0x100,
            HKEY_LOCAL_MACHINE=object(), REG_DWORD=4,
            OpenKey=Mock(return_value=key), QueryValueEx=Mock(return_value=(value, kind))), key

    def life(self, boot, *, platform="win32", host="fixture-host"):
        return {"host": host, "boot": boot, "platform": platform, "process_birth": "fixture-process"}

    def test_reads_only_existing_os_counter_in_native_registry_view(self):
        registry, key = self.registry()
        self.assertEqual(guard._windows_boot_identifier(registry), "windows-boot-counter-v1:37")
        registry.OpenKey.assert_called_once_with(registry.HKEY_LOCAL_MACHINE,
            guard.WINDOWS_BOOT_COUNTER_KEY, 0, registry.KEY_READ | registry.KEY_WOW64_64KEY)
        registry.QueryValueEx.assert_called_once_with(key, "BootId")
        key.__exit__.assert_called_once()

    def test_registry_missing_denied_or_malformed_has_no_fallback(self):
        for error in (FileNotFoundError("absent fixture counter"), PermissionError("denied fixture counter")):
            registry, _ = self.registry()
            registry.OpenKey.side_effect = error
            with self.subTest(error=type(error).__name__), self.assertRaises(type(error)):
                guard._windows_boot_identifier(registry)
            registry.QueryValueEx.assert_not_called()
        for value, kind in ((37, 1), ("37", 4), (True, 4), (-1, 4), (1 << 32, 4)):
            with self.subTest(value=value, kind=kind), self.assertRaises(OSError):
                guard._windows_boot_identifier(self.registry(value, kind)[0])

    def test_counter_zero_and_dword_max_are_representable(self):
        for value in (0, 0xffffffff):
            identifier = guard._windows_boot_identifier(self.registry(value)[0])
            self.assertEqual(guard._windows_boot_count(identifier), value)

    def test_only_strictly_increasing_same_domain_counter_proves_reboot(self):
        previous = self.life("windows-boot-counter-v1:37")
        self.assertTrue(guard.reboot_proves_dead(previous, self.life("windows-boot-counter-v1:38")))
        for count in (0, 36, 37):
            with self.subTest(count=count):
                self.assertFalse(guard.reboot_proves_dead(previous,
                    self.life(guard.WINDOWS_BOOT_COUNTER_PREFIX + str(count))))
        self.assertFalse(guard.reboot_proves_dead(self.life("windows-boot-counter-v1:4294967295"),
                                                self.life("windows-boot-counter-v1:0")))

    def test_legacy_and_mixed_domains_never_prove_windows_reboot(self):
        legacy = "23a96d684c30f111a28ed0a68790d294"
        for previous, current in ((legacy, "0" * 32), (legacy, "windows-boot-counter-v1:38"),
                                  ("windows-boot-counter-v1:37", legacy)):
            with self.subTest(previous=previous, current=current):
                self.assertFalse(guard.reboot_proves_dead(self.life(previous), self.life(current)))

    def test_noncanonical_and_unknown_counter_formats_reject(self):
        for value in ("", "-1", "+1", "01", "1.0", " 1", "1 ", "１", "4294967296", "9" * 100):
            malformed = guard.WINDOWS_BOOT_COUNTER_PREFIX + value
            with self.subTest(value=value):
                self.assertIsNone(guard._windows_boot_count(malformed))
                self.assertFalse(guard.reboot_proves_dead(self.life("windows-boot-counter-v1:37"),
                                                         self.life(malformed)))

    def test_host_platform_and_exact_lifetime_shape_still_required(self):
        previous, current = self.life("windows-boot-counter-v1:37"), self.life("windows-boot-counter-v1:38")
        for changed in (None, {}, {**current, "extra": "not-supported"},
                        {**current, "process_birth": ""}, {**current, "host": "foreign"},
                        {**current, "platform": "linux"}):
            with self.subTest(current=changed):
                self.assertFalse(guard.reboot_proves_dead(previous, changed))

    def test_linux_boot_uuid_semantics_are_unchanged(self):
        self.assertFalse(guard.reboot_proves_dead(self.life("fixture-a", platform="linux"),
                                                 self.life("fixture-a", platform="linux")))
        self.assertTrue(guard.reboot_proves_dead(self.life("fixture-a", platform="linux"),
                                                self.life("fixture-b", platform="linux")))


if __name__ == "__main__":
    unittest.main()
