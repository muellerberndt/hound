"""
Simplified unit tests for GraphBuilder with mocked LLM responses.
Shows how to test graph generation without API keys.
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


class TestGraphBuilderMocking(unittest.TestCase):
    """Test GraphBuilder with simple LLM mocking."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manifest_dir = Path(self.temp_dir) / "manifest"
        self.output_dir = Path(self.temp_dir) / "graphs"
        self.manifest_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)

        # Create minimal test manifest with required fields
        manifest_data = {
            "repository": str(self.temp_dir),
            "files": ["test.py"],
            "num_files": 1,
            "total_chars": 100,
            "stats": {"total_files": 1, "total_chars": 100},
        }
        with open(self.manifest_dir / "manifest.json", "w") as f:
            json.dump(manifest_data, f)

        # Create cards.jsonl file (required by graph builder)
        cards_data = [
            {"id": "card_0", "path": "test.py", "content": "def test(): pass", "metadata": {}}
        ]
        with open(self.manifest_dir / "cards.jsonl", "w") as f:
            for card in cards_data:
                f.write(json.dumps(card) + "\n")

        # Create bundles directory with one bundle
        bundles_dir = self.manifest_dir / "bundles"
        bundles_dir.mkdir()

        bundle_data = {
            "id": "bundle_0",
            "cards": [{"id": "card_0", "path": "test.py", "content": "def test(): pass"}],
        }
        with open(bundles_dir / "bundle_0.json", "w") as f:
            json.dump(bundle_data, f)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("analysis.graph_builder.LLMClient")
    def test_basic_graph_building_with_mock(self, mock_llm_class):
        """Test basic graph building with mocked LLM responses."""
        from analysis.graph_builder import GraphBuilder

        # Create mock LLM client
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        # Mock the parse method to return test data
        def mock_parse(*args, **kwargs):
            # Get response_model from either args or kwargs
            response_model = kwargs.get("response_model") or kwargs.get("schema")
            if response_model is None and len(args) >= 3:
                response_model = args[2]

            # Return different responses based on the model class name
            if response_model and response_model.__name__ == "GraphDiscovery":
                # Initial discovery - what graphs to build
                from analysis.graph_builder import GraphDiscovery

                return GraphDiscovery(
                    graphs_needed=[{"name": "TestGraph", "focus": "testing"}],
                    suggested_node_types=["test_node"],
                    suggested_edge_types=["test_edge"],
                )
            elif response_model and response_model.__name__ == "GraphUpdate":
                # Graph building iteration
                from analysis.graph_builder import GraphUpdate

                return GraphUpdate(
                    target_graph="TestGraph",
                    new_nodes=[
                        {
                            "id": "node1",
                            "type": "test_node",
                            "label": "Test Node 1",
                            "properties": {},
                            "confidence": 0.9,
                        }
                    ],
                    new_edges=[],
                    node_updates=[],
                )
            return None

        mock_llm.parse.side_effect = mock_parse

        # Create builder with mock config
        config = {"models": {"graph": {"provider": "mock", "model": "mock"}}}
        builder = GraphBuilder(config)

        # Run build with minimal iterations
        results = builder.build(
            manifest_dir=self.manifest_dir,
            output_dir=self.output_dir,
            max_iterations=1,
            max_graphs=1,
        )

        # Verify results
        self.assertIn("graphs", results)
        self.assertEqual(len(results["graphs"]), 1)

        # Verify graph file was created
        graph_files = list(self.output_dir.glob("graph_*.json"))
        self.assertEqual(len(graph_files), 1)

        # Load and check the graph
        with open(graph_files[0]) as f:
            graph_data = json.load(f)

        # The name may be different based on internal logic
        # Just verify the graph was created with expected structure
        self.assertIn("name", graph_data)
        self.assertIn("focus", graph_data)
        self.assertIn("nodes", graph_data)
        self.assertIn("edges", graph_data)

        # Verify our mock was called
        self.assertTrue(mock_llm.parse.called)
        self.assertGreaterEqual(mock_llm.parse.call_count, 2)  # Discovery + at least 1 iteration

    @patch("llm.unified_client.UnifiedLLMClient")
    def test_agent_with_mock_llm(self, mock_llm_class):
        """Test AutonomousAgent with mocked LLM."""
        from analysis.agent_core import AutonomousAgent

        # Create mock LLM
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        # Mock raw method for agent operations
        def mock_raw(system, user, profile=None):
            # Return simple JSON for testing
            return json.dumps(
                {
                    "action": "load_graph",
                    "parameters": {"graph_name": "TestGraph"},
                    "reasoning": "Testing graph loading",
                }
            )

        mock_llm.raw.side_effect = mock_raw

        # Create test graphs directory
        graphs_dir = Path(self.temp_dir) / "graphs"
        graphs_dir.mkdir(exist_ok=True)

        # Create knowledge_graphs.json
        kg_data = {"graphs": {"TestGraph": str(graphs_dir / "test_graph.json")}}
        with open(graphs_dir / "knowledge_graphs.json", "w") as f:
            json.dump(kg_data, f)

        # Create test graph
        test_graph = {"name": "TestGraph", "nodes": [{"id": "n1", "label": "Node 1"}], "edges": []}
        with open(graphs_dir / "test_graph.json", "w") as f:
            json.dump(test_graph, f)

        # Create manifest
        with open(graphs_dir / "manifest.json", "w") as f:
            json.dump({"repo_path": str(self.temp_dir)}, f)

        # Create agent with mocked config
        config = {"models": {"agent": {"provider": "mock", "model": "mock"}}}

        agent = AutonomousAgent(
            graphs_metadata_path=graphs_dir / "knowledge_graphs.json",
            manifest_path=graphs_dir,
            agent_id="test_agent",
            config=config,
            debug=False,
        )

        # Verify agent was created
        self.assertIsNotNone(agent)
        self.assertEqual(agent.agent_id, "test_agent")


class TestEndToEndWithoutAPIs(unittest.TestCase):
    """Test complete pipeline without requiring API keys."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_dir = Path(self.temp_dir) / "test_project"
        self.project_dir.mkdir(parents=True)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch.dict(os.environ, {}, clear=True)  # Clear environment variables
    @patch("llm.openai_provider.OpenAIProvider.__init__", return_value=None)
    @patch("llm.unified_client.UnifiedLLMClient")
    def test_no_api_keys_required(self, mock_client, mock_openai_init):
        """Verify tests work without any API keys in environment."""
        # This test ensures our mocking strategy works even when
        # no API keys are present in the environment

        # Verify no API keys are set
        self.assertNotIn("OPENAI_API_KEY", os.environ)
        self.assertNotIn("ANTHROPIC_API_KEY", os.environ)
        self.assertNotIn("GOOGLE_API_KEY", os.environ)

        # Mock the client to avoid API calls
        mock_llm = MagicMock()
        mock_client.return_value = mock_llm

        # Create a simple config
        config = {"models": {"test": {"provider": "openai", "model": "gpt-4"}}}

        # This should work without API keys due to mocking
        from llm.unified_client import UnifiedLLMClient

        client = UnifiedLLMClient(config, profile="test")

        # Verify the mock was used
        self.assertEqual(client, mock_llm)


if __name__ == "__main__":
    unittest.main()
