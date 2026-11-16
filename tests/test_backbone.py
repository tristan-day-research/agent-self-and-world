"""Local checks only: no model downloads, pod connections, or R2 requests."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from core import runs
from core.device import resolve_device
from core.storage import sync_to_r2


class BackboneTests(unittest.TestCase):
    def test_cli_smoke_and_unknown_experiment(self):
        from run import main

        with tempfile.TemporaryDirectory() as tmp, patch.object(runs, "PROJECT_ROOT", Path(tmp)):
            self.assertEqual(main(["smoke", "--smoke", "--device", "cpu"]), 0)
            directory, = (Path(tmp) / runs.RESULTS_SUBDIR / "_smoke" / "smoke").iterdir()
            self.assertEqual(json.loads((directory / "status.json").read_text())["state"], "completed")
            self.assertEqual(json.loads((directory / "config.json").read_text())["device"], "cpu")
            self.assertFalse(json.loads((directory / "outputs.json").read_text())["model_loaded"])
            self.assertTrue((directory / "run.log").is_file())
        result = subprocess.run([sys.executable, "run.py", "unknown"], cwd=runs.PROJECT_ROOT,
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("smoke", result.stderr)

    def test_failure_is_recorded_and_logger_closed(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runs, "PROJECT_ROOT", Path(tmp)):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                with runs.run_context("example", {}) as run:
                    raise RuntimeError("test failure")
            status = json.loads((run.directory / "status.json").read_text())
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["error_type"], "RuntimeError")
            self.assertEqual(run.logger.handlers, [])

    def test_device_selection_with_mocked_torch(self):
        torch = Mock()
        torch.cuda.is_available.return_value = False
        torch.backends.mps.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": torch}):
            self.assertEqual(resolve_device(), "mps")
            with self.assertRaises(RuntimeError):
                resolve_device("cuda")
        with patch.dict(sys.modules, {"torch": None}):
            self.assertEqual(resolve_device(), "cpu")

    def test_r2_missing_credentials_and_mocked_upload(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(sync_to_r2("unused", prefix="unused"), 0)
        with patch.dict("os.environ", {"R2_BUCKET": "example"}, clear=True):
            with self.assertRaises(ValueError):
                sync_to_r2("unused", prefix="unused")
        env = {"R2_ENDPOINT": "https://example.invalid", "R2_ACCESS_KEY_ID": "test",
               "R2_SECRET_ACCESS_KEY": "test", "R2_BUCKET": "example"}
        sdk = Mock()
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True), \
                patch.dict(sys.modules, {"boto3": sdk}):
            source = Path(tmp) / "result.json"
            runs.write_json(source, {"example": True})
            self.assertEqual(sync_to_r2(tmp, prefix="project/run"), 1)
            sdk.client.return_value.upload_file.assert_called_once_with(
                str(source), "example", "project/run/result.json")


if __name__ == "__main__":
    unittest.main()
