"""
Test that GraphBuilder properly uses model-specific context limits.
"""

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGraphContextLimits(unittest.TestCase):
    """Test that GraphBuilder respects model-specific context limits."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manifest_dir = Path(self.temp_dir) / "manifest"
        self.manifest_dir.mkdir(parents=True)

        # Create test manifest
        manifest_data = {
            "repo_path": str(self.temp_dir),
            "num_files": 100,
            "files": [f"file_{i}.py" for i in range(100)],
        }
        with open(self.manifest_dir / "manifest.json", "w") as f:
            json.dump(manifest_data, f)

        # Create many cards to test context limits
        self.cards = []
        for i in range(100):
            self.cards.append(
                {
                    "id": f"card_{i}",
                    "relpath": f"file_{i}.py",
                    "content": "x" * 10000,  # 10k chars per card = ~1M chars total
                }
            )

        with open(self.manifest_dir / "cards.jsonl", "w") as f:
            for card in self.cards:
                f.write(json.dumps(card) + "\n")

    def tearDown(self):
        """Clean up."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("analysis.graph_builder.LLMClient")
    @patch("analysis.graph_builder.count_tokens")
    def test_graph_model_uses_own_context_limit(self, mock_count_tokens, mock_llm_class):
        """Test that graph model uses its own max_context when configured."""
        from analysis.graph_builder import GraphBuilder

        # Mock token counting (4 chars = 1 token roughly)
        mock_count_tokens.side_effect = lambda text, provider, model: len(text) // 4

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        # Config with graph model having large context
        config = {
            "models": {
                "graph": {
                    "provider": "openai",
                    "model": "gpt-4.1",
                    "max_context": 1000000,  # 1M tokens
                },
                "strategist": {"provider": "openai", "model": "gpt-4"},
            },
            "context": {
                "max_tokens": 256000  # Global limit is smaller
            },
        }

        builder = GraphBuilder(config, debug=True)

        # Test sampling with large context
        sampled = builder._sample_cards(self.cards)

        # With 1M token context, should be able to fit more cards
        # Each card is ~10k chars = ~2.5k tokens
        # 1M context - 30k reserved = 970k available
        # 80% of 970k = 776k target tokens
        # Should fit ~310 cards worth of tokens, but we only have 100
        self.assertEqual(len(sampled), 100, "Should use all cards with 1M context")

    @patch("analysis.graph_builder.LLMClient")
    @patch("analysis.graph_builder.count_tokens")
    def test_falls_back_to_global_context_limit(self, mock_count_tokens, mock_llm_class):
        """Test fallback to global context limit when graph model doesn't specify one."""
        from analysis.graph_builder import GraphBuilder

        # Mock token counting
        mock_count_tokens.side_effect = lambda text, provider, model: len(text) // 4

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        # Config WITHOUT graph model max_context
        config = {
            "models": {
                "graph": {
                    "provider": "openai",
                    "model": "gpt-4",  # No max_context specified
                }
            },
            "context": {
                "max_tokens": 100000  # Small global limit
            },
        }

        builder = GraphBuilder(config, debug=False)

        # Test sampling with smaller global context
        sampled = builder._sample_cards(self.cards)

        # With 100k token context:
        # 100k - 30k reserved = 70k available
        # 80% of 70k = 56k target tokens
        # Each card is ~2.5k tokens
        # Should fit ~22 cards
        self.assertLess(len(sampled), 30, "Should sample fewer cards with smaller context")
        self.assertGreater(len(sampled), 15, "Should sample at least some cards")

    @patch("analysis.graph_builder.LLMClient")
    def test_context_usage_logging(self, mock_llm_class):
        """Test that context usage is properly logged in debug mode."""
        from analysis.graph_builder import GraphBuilder

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        # Config with large context
        config = {
            "models": {"graph": {"provider": "openai", "model": "gpt-4.1", "max_context": 1000000}}
        }

        # Create small dataset that fits in context
        small_cards = [{"id": "card_1", "relpath": "test.py", "content": "def test(): pass"}]

        builder = GraphBuilder(config, debug=True)

        # Capture debug output
        import io
        from contextlib import redirect_stdout

        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            sampled = builder._sample_cards(small_cards)

        output = captured_output.getvalue()

        # Check that it logs the model's max_context
        self.assertIn("1,000,000", output, "Should show 1M token context in debug output")
        self.assertIn("Using ALL", output, "Should indicate using all cards")

    @patch("analysis.graph_builder.LLMClient")
    @patch("analysis.graph_builder.count_tokens")
    def test_discovery_uses_strategist_context(self, mock_count_tokens, mock_llm_class):
        """Test that discovery phase uses strategist model's context, not graph model's."""
        from analysis.graph_builder import GraphBuilder

        # Mock token counting
        mock_count_tokens.side_effect = lambda text, provider, model: len(text) // 4

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        # Config with different context limits
        config = {
            "models": {
                "graph": {
                    "provider": "openai",
                    "model": "gpt-4.1",
                    "max_context": 1000000,  # Graph model has 1M
                },
                "strategist": {
                    "provider": "openai",
                    "model": "gpt-4",
                    # No max_context, will use global
                },
            },
            "context": {
                "max_tokens": 256000  # Global/strategist limit is 256k
            },
        }

        builder = GraphBuilder(config, debug=False)

        # Test discovery sampling
        sampled_discovery = builder._sample_cards_for_discovery(self.cards)

        # Discovery should use 256k context (global), not 1M
        # 256k - 50k reserved = 206k available (more conservative now)
        # 70% of 206k = ~144k target tokens (more conservative)
        # Each card is ~2.5k tokens
        # Should fit ~57 cards
        self.assertLess(len(sampled_discovery), 65, "Discovery should use smaller context")
        self.assertGreater(
            len(sampled_discovery), 50, "Discovery should still sample reasonable amount"
        )

        # Graph building should use 1M context
        sampled_graph = builder._sample_cards(self.cards)
        self.assertEqual(len(sampled_graph), 100, "Graph building should use full 1M context")

    def test_config_structure(self):
        """Test that config properly supports max_context per model."""
        config = {
            "models": {
                "graph": {"provider": "openai", "model": "gpt-4.1", "max_context": 1000000},
                "scout": {"provider": "openai", "model": "gpt-4", "max_context": 128000},
                "strategist": {"provider": "openai", "model": "gpt-5", "max_context": 400000},
            },
            "context": {
                "max_tokens": 256000  # Global default
            },
        }

        # Test accessing model-specific context
        graph_context = config["models"]["graph"].get("max_context")
        self.assertEqual(graph_context, 1000000)

        scout_context = config["models"]["scout"].get("max_context")
        self.assertEqual(scout_context, 128000)

        # Model without max_context should return None
        finalize_context = config["models"].get("finalize", {}).get("max_context")
        self.assertIsNone(finalize_context)


if __name__ == "__main__":
    unittest.main()
