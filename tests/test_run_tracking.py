"""Unit tests for run tracking functionality."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.token_tracker import TokenTracker, get_token_tracker
from analysis.run_tracker import RunTracker


class TestTokenTracker(unittest.TestCase):
    """Test TokenTracker functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.tracker = TokenTracker()

    def test_track_single_usage(self):
        """Test tracking a single token usage."""
        self.tracker.track_usage(
            provider="openai", model="gpt-4", input_tokens=100, output_tokens=50, profile="agent"
        )

        summary = self.tracker.get_summary()

        self.assertEqual(summary["total_usage"]["input_tokens"], 100)
        self.assertEqual(summary["total_usage"]["output_tokens"], 50)
        self.assertEqual(summary["total_usage"]["total_tokens"], 150)
        self.assertEqual(summary["total_usage"]["call_count"], 1)

        # Check model-specific tracking
        self.assertIn("openai:gpt-4", summary["by_model"])
        model_usage = summary["by_model"]["openai:gpt-4"]
        self.assertEqual(model_usage["input_tokens"], 100)
        self.assertEqual(model_usage["output_tokens"], 50)
        self.assertEqual(model_usage["call_count"], 1)

    def test_track_multiple_usages(self):
        """Test tracking multiple token usages."""
        self.tracker.track_usage("openai", "gpt-4", 100, 50)
        self.tracker.track_usage("openai", "gpt-4", 200, 100)
        self.tracker.track_usage("anthropic", "claude-3", 150, 75)

        summary = self.tracker.get_summary()

        # Total across all calls
        self.assertEqual(summary["total_usage"]["input_tokens"], 450)
        self.assertEqual(summary["total_usage"]["output_tokens"], 225)
        self.assertEqual(summary["total_usage"]["total_tokens"], 675)
        self.assertEqual(summary["total_usage"]["call_count"], 3)

        # Per-model tracking
        self.assertEqual(summary["by_model"]["openai:gpt-4"]["call_count"], 2)
        self.assertEqual(summary["by_model"]["anthropic:claude-3"]["call_count"], 1)

    def test_reset(self):
        """Test resetting the tracker."""
        self.tracker.track_usage("openai", "gpt-4", 100, 50)
        self.tracker.reset()

        summary = self.tracker.get_summary()
        self.assertEqual(summary["total_usage"]["call_count"], 0)
        self.assertEqual(len(summary["by_model"]), 0)


class TestRunTracker(unittest.TestCase):
    """Test RunTracker functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self.output_file = Path(self.temp_file.name)
        self.temp_file.close()
        self.tracker = RunTracker(self.output_file)

    def tearDown(self):
        """Clean up test fixtures."""
        self.output_file.unlink(missing_ok=True)

    def test_initial_state(self):
        """Test initial state of run tracker."""
        with open(self.output_file) as f:
            data = json.load(f)

        self.assertEqual(data["status"], "running")
        self.assertIsNone(data["run_id"])
        self.assertEqual(data["command_args"], [])
        self.assertIsNone(data["end_time"])
        self.assertEqual(data["investigations"], [])
        self.assertEqual(data["errors"], [])

    def test_set_run_info(self):
        """Test setting run information."""
        self.tracker.set_run_info("test_run_123", ["hound", "agent", "test"])

        with open(self.output_file) as f:
            data = json.load(f)

        self.assertEqual(data["run_id"], "test_run_123")
        self.assertEqual(data["command_args"], ["hound", "agent", "test"])

    def test_update_token_usage(self):
        """Test updating token usage."""
        token_summary = {
            "total_usage": {
                "input_tokens": 1000,
                "output_tokens": 500,
                "total_tokens": 1500,
                "call_count": 5,
            },
            "by_model": {
                "openai:gpt-4": {
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "total_tokens": 1500,
                    "call_count": 5,
                }
            },
        }

        self.tracker.update_token_usage(token_summary)

        with open(self.output_file) as f:
            data = json.load(f)

        self.assertEqual(data["token_usage"], token_summary)

    def test_add_investigation(self):
        """Test adding investigation results."""
        investigation = {
            "goal": "Find vulnerabilities",
            "priority": 1,
            "category": "security",
            "iterations_completed": 5,
            "hypotheses": {"total": 3},
        }

        self.tracker.add_investigation(investigation)

        with open(self.output_file) as f:
            data = json.load(f)

        self.assertEqual(len(data["investigations"]), 1)
        self.assertEqual(data["investigations"][0], investigation)

    def test_add_error(self):
        """Test adding error messages."""
        self.tracker.add_error("Test error message")

        with open(self.output_file) as f:
            data = json.load(f)

        self.assertEqual(len(data["errors"]), 1)
        self.assertEqual(data["errors"][0]["error"], "Test error message")
        self.assertIn("timestamp", data["errors"][0])

    def test_finalize(self):
        """Test finalizing the run."""
        time.sleep(0.1)  # Ensure some runtime
        self.tracker.finalize(status="completed")

        with open(self.output_file) as f:
            data = json.load(f)

        self.assertEqual(data["status"], "completed")
        self.assertIsNotNone(data["end_time"])
        self.assertGreater(data["runtime_seconds"], 0)

    def test_runtime_tracking(self):
        """Test that runtime is tracked correctly."""
        time.sleep(0.2)
        self.tracker.add_investigation({"goal": "test"})

        with open(self.output_file) as f:
            data = json.load(f)

        self.assertGreater(data["runtime_seconds"], 0.1)


class TestIntegration(unittest.TestCase):
    """Test integration between token tracker and run tracker."""

    def test_token_tracking_integration(self):
        """Test that token tracking integrates with run tracking."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_file = Path(f.name)

        try:
            # Set up trackers
            token_tracker = TokenTracker()
            run_tracker = RunTracker(output_file)

            # Simulate some token usage
            token_tracker.track_usage("openai", "gpt-4", 100, 50)
            token_tracker.track_usage("openai", "gpt-4", 200, 100)

            # Update run tracker with token usage
            run_tracker.update_token_usage(token_tracker.get_summary())

            # Check that data was saved correctly
            with open(output_file) as f:
                data = json.load(f)

            self.assertIn("token_usage", data)
            self.assertEqual(data["token_usage"]["total_usage"]["input_tokens"], 300)
            self.assertEqual(data["token_usage"]["total_usage"]["output_tokens"], 150)
            self.assertEqual(data["token_usage"]["total_usage"]["call_count"], 2)
        finally:
            output_file.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
