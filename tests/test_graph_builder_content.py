"""
Test that GraphBuilder properly loads full content from cards.
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGraphBuilderContentLoading(unittest.TestCase):
    """Test that GraphBuilder loads full content from cards."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_dir = Path(self.temp_dir) / "test_repo"
        self.repo_dir.mkdir(parents=True)
        self.manifest_dir = Path(self.temp_dir) / "manifest"
        self.output_dir = Path(self.temp_dir) / "graphs"
        self.manifest_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)

        # Create a test source file
        test_file = self.repo_dir / "test.py"
        self.full_content = """def important_function():
    # This is line 2
    # This is line 3
    # This is line 4
    # This is line 5
    # This is line 6
    # This is line 7
    # This is line 8
    # This is line 9
    # This is line 10
    # This is line 11
    # This is line 12
    # This is line 13
    # This is line 14
    # This is line 15
    # This is line 16
    # This is line 17
    # This is line 18
    # This is line 19
    # This is line 20
    return "Complete content"
"""
        test_file.write_text(self.full_content)

        # Create manifest with repo_path
        manifest_data = {
            "repository": str(self.repo_dir),
            "repo_path": str(self.repo_dir),  # Important for content extraction
            "files": ["test.py"],
            "num_files": 1,
            "total_chars": len(self.full_content),
            "stats": {"total_files": 1, "total_chars": len(self.full_content)},
        }
        with open(self.manifest_dir / "manifest.json", "w") as f:
            json.dump(manifest_data, f)

        # Create cards.jsonl with only peek_head and peek_tail (no content field)
        cards_data = [
            {
                "id": "card_0",
                "relpath": "test.py",
                "peek_head": "def important_function():\n    # This is line 2\n    # This is line 3",
                "peek_tail": '    # This is line 19\n    # This is line 20\n    return "Complete content"',
                "char_start": 0,
                "char_end": len(self.full_content),
                "type": "function",
                # Note: NO 'content' field - this simulates the current behavior
            }
        ]
        with open(self.manifest_dir / "cards.jsonl", "w") as f:
            for card in cards_data:
                f.write(json.dumps(card) + "\n")

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("analysis.graph_builder.LLMClient")
    def test_graph_builder_loads_full_content(self, mock_llm_class):
        """Test that GraphBuilder loads full content, not just peek_head/peek_tail."""
        from analysis.graph_builder import GraphBuilder

        # Create mock LLM client
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        # Track what content the LLM sees
        seen_content = []

        def mock_parse(system, user, schema):
            # Capture the user prompt to see what content was provided
            user_data = json.loads(user) if isinstance(user, str) else user
            if "code_samples" in user_data:
                for sample in user_data["code_samples"]:
                    if "content" in sample:
                        seen_content.append(sample["content"])

            # Return appropriate mock response based on schema
            if schema.__name__ == "GraphDiscovery":
                from analysis.graph_builder import GraphDiscovery, GraphSpec

                return GraphDiscovery(
                    graphs_needed=[GraphSpec(name="TestGraph", focus="testing")],
                    suggested_node_types=["function"],
                    suggested_edge_types=["calls"],
                )
            elif schema.__name__ == "GraphUpdate":
                from analysis.graph_builder import GraphUpdate, NodeSpec

                return GraphUpdate(
                    target_graph="TestGraph",
                    new_nodes=[
                        NodeSpec(
                            id="func_important",
                            type="function",
                            label="important_function",
                            refs=["card_0"],
                        )
                    ],
                    new_edges=[],
                    node_updates=[],
                )
            return None

        mock_llm.parse = mock_parse

        # Create builder
        config = {
            "models": {
                "graph": {"provider": "mock", "model": "mock"},
                "strategist": {"provider": "mock", "model": "mock"},
            }
        }
        builder = GraphBuilder(config, debug=True)

        # Run build
        results = builder.build(
            manifest_dir=self.manifest_dir,
            output_dir=self.output_dir,
            max_iterations=1,
            max_graphs=1,
        )

        # Verify that the full content was loaded and passed to LLM
        self.assertTrue(len(seen_content) > 0, "No content was passed to LLM")

        # Check that we got the FULL content, not just the peeks
        for content in seen_content:
            if content:  # Skip empty content
                # The full content should include middle lines that aren't in peek_head or peek_tail
                self.assertIn(
                    "# This is line 10",
                    content,
                    "Middle content missing - only peek_head/tail was used!",
                )
                self.assertIn(
                    "# This is line 15",
                    content,
                    "Middle content missing - only peek_head/tail was used!",
                )

                # Verify it's the complete content
                self.assertEqual(
                    content.strip(),
                    self.full_content.strip(),
                    "Content doesn't match the full file content",
                )

    def test_extract_card_content_directly(self):
        """Test the extract_card_content function directly."""
        from analysis.cards import extract_card_content

        # Test with a card that has no content field
        card = {
            "id": "test_card",
            "relpath": "test.py",
            "peek_head": "def important_function():\n    # This is line 2",
            "peek_tail": '    return "Complete content"',
            "char_start": 0,
            "char_end": len(self.full_content),
        }

        # Extract content
        content = extract_card_content(card, self.repo_dir)

        # Verify we got the full content
        self.assertEqual(
            content, self.full_content, "extract_card_content didn't return full content"
        )

        # Test fallback when no repo_root
        content_fallback = extract_card_content(card, None)
        expected_fallback = card["peek_head"] + "\n" + card["peek_tail"]
        self.assertEqual(
            content_fallback.strip(),
            expected_fallback.strip(),
            "Fallback to peek_head/tail didn't work",
        )

    @patch("llm.client.LLMClient")  # Patch the direct LLMClient used in graph_custom
    @patch("analysis.graph_builder.LLMClient")
    def test_graph_custom_loads_full_content(self, mock_gb_llm_class, mock_llm_class):
        """Test that graph_custom.py also loads full content."""
        from commands.graph_custom import build_custom_graph

        # Create mock LLM for both graph_builder and graph_custom
        mock_llm = MagicMock()
        mock_gb_llm_class.return_value = mock_llm
        mock_llm_class.return_value = mock_llm

        # Track content seen
        seen_in_custom = []

        def mock_parse(system, user, schema):
            # Check if this is the custom graph call with code samples
            if isinstance(user, str):
                try:
                    # Try to find and parse JSON in the user prompt
                    import json
                    import re

                    # Look for JSON array in the prompt
                    json_match = re.search(r"\[[\s\S]*?\]", user)
                    if json_match:
                        samples = json.loads(json_match.group())
                        for sample in samples:
                            if isinstance(sample, dict) and "content" in sample:
                                seen_in_custom.append(sample["content"])
                except:
                    pass

            # Return mock response
            if schema.__name__ == "CustomGraphSpec":
                from commands.graph_custom import CustomGraphSpec

                return CustomGraphSpec(
                    name="CustomTest",
                    focus="custom testing",
                    node_selection="all functions",
                    edge_types=["calls"],
                )
            elif schema.__name__ == "GraphDiscovery":
                from analysis.graph_builder import GraphDiscovery, GraphSpec

                return GraphDiscovery(
                    graphs_needed=[GraphSpec(name="CustomTest", focus="custom testing")],
                    suggested_node_types=["function"],
                    suggested_edge_types=["calls"],
                )
            elif schema.__name__ == "GraphUpdate":
                from analysis.graph_builder import GraphUpdate

                return GraphUpdate(
                    target_graph="CustomTest", new_nodes=[], new_edges=[], node_updates=[]
                )
            return None

        mock_llm.parse = mock_parse

        # Mock the config loading
        with patch("commands.graph_custom.load_config") as mock_config:
            mock_config.return_value = {
                "models": {
                    "agent": {"provider": "mock", "model": "mock"},
                    "graph": {"provider": "mock", "model": "mock"},
                    "strategist": {"provider": "mock", "model": "mock"},
                }
            }

            # Mock the console to avoid output
            with patch("commands.graph_custom.console"):
                # Create graphs directory (required by graph_custom)
                graphs_dir = Path(self.temp_dir) / "graphs"
                graphs_dir.mkdir(exist_ok=True)

                # Run custom graph build
                result_path = build_custom_graph(
                    project_path=Path(self.temp_dir),
                    description="Test custom graph",
                    name="CustomTest",
                    iterations=1,
                    debug=False,
                )

        # Verify that full content was seen
        self.assertTrue(len(seen_in_custom) > 0, "No content passed to custom graph builder")

        for content in seen_in_custom:
            if content:  # Skip empty content
                # Should have middle content, not just peek_head/tail
                self.assertIn(
                    "# This is line 10", content, "Custom graph builder didn't get full content"
                )


if __name__ == "__main__":
    unittest.main()
