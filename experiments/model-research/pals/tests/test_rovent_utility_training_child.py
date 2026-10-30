"""Stdlib child boundaries; no live process, model, optimizer or lease."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rz_pals_model import rovent_utility_training_child as child


class _MetadataReader:
    """Test-only pinned operand projection; never a production authority."""
    def __init__(self, value):
        self.value = value

    def json(self, _pin, *, maximum):
        if maximum != 65536:
            raise AssertionError("export operand must retain its bounded metadata read")
        return dict(self.value)


class UtilityChildBoundaryTests(unittest.TestCase):
    def test_startup_requires_exact_bounded_original_bytes_and_eof(self):
        self.assertEqual(child._read_startup(io.BytesIO(b"abc"), 3), b"abc")
        for stream, count in ((io.BytesIO(b"abc"), 2), (io.BytesIO(b"a"), 3),
                              (io.BytesIO(b"a"), True), (io.BytesIO(b""), 0),
                              (io.BytesIO(b""), child.MAX_STARTUP + 1)):
            with self.assertRaises(ValueError): child._read_startup(stream, count)

    def test_original_v1_metadata_cannot_enter_the_actual_v2_source_lane(self):
        startup = {"registration": {"schema": "rz-rovent-utility-training-registration/1"}}
        with self.assertRaisesRegex(ValueError, "mixed-ancestry registration/2"):
            child._metadata_inputs(startup, None)

    def test_old_startup_cannot_reach_os_bootstrap_or_socket_before_actual_v2_guard(self):
        startup = {"registration": {"schema": "rz-rovent-utility-training-registration/1"}}
        args = ["--execute", "--startup-sha256", "10" * 32, "--startup-bytes", "1",
                "--execution-until-ns", "100", "--whole-until-ns", "200"]
        with patch("rz_pals_model.rovent_utility_training_bootstrap.validate_startup", return_value=startup), \
                patch("rz_pals_model.rovent_utility_training_bootstrap.bootstrap_utility_child") as bootstrap:
            with self.assertRaisesRegex(ValueError, "mixed-ancestry registration/2"):
                child.original_utility_child_main(args, stdin=io.BytesIO(b"x"), clock=lambda: 1)
            bootstrap.assert_not_called()

    def test_execute_flag_is_required_before_any_startup_or_runtime_work(self):
        args = ["--startup-sha256", "10" * 32, "--startup-bytes", "1",
                "--execution-until-ns", "100", "--whole-until-ns", "200"]
        with self.assertRaisesRegex(ValueError, "explicit execution"):
            child.original_utility_child_main(args, stdin=io.BytesIO(b"x"), clock=lambda: 1)

    def test_export_operand_has_separate_schema_and_new_owned_json_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            startup = {"registration": {"export_request": {}, "output_root": str(root), "max_output_bytes": 1048576}}
            request = {"schema": "rz-rovent-utility-export-request/1", "kind": "deployment",
                       "path": str(root / "utility.json"), "max_output_bytes": 4096}
            self.assertEqual(child._export_request(startup, _MetadataReader(request)), request)
            for change in ({"schema": "rz-rovent-original-export-request/1"}, {"kind": "pcv"},
                           {"path": str(root / "utility.pt")}, {"max_output_bytes": True},
                           {"max_output_bytes": 1048577}, {"execution_authorized": True}):
                altered = dict(request); altered.update(change)
                with self.assertRaises(ValueError): child._export_request(startup, _MetadataReader(altered))


if __name__ == "__main__":
    unittest.main()
