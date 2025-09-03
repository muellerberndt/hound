"""
Tests for DebugLogger to ensure it writes HTML logs with prompts and responses.
"""

import unittest
import tempfile
import shutil
from pathlib import Path

from analysis.debug_logger import DebugLogger


class TestDebugLogger(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_log_and_finalize(self):
        dl = DebugLogger("test_session", output_dir=self.tmp)
        self.assertTrue(dl.log_file.exists())

        dl.log_interaction(
            system_prompt="You are a test",
            user_prompt="Please do something",
            response={"ok": True},
            schema=None,
            duration=0.12,
            error=None,
            tool_calls=[{"tool_name": "echo", "parameters": {"x": 1}}],
        )

        # File contains prompts and response
        content = dl.log_file.read_text()
        self.assertIn("You are a test", content)
        self.assertIn("Please do something", content)
        self.assertIn("ok", content)

        # Finalize writes footer and returns path
        out = dl.finalize({"total": 1})
        self.assertTrue(Path(out).exists())
        footer = Path(out).read_text()
        self.assertIn("Completed:", footer)
