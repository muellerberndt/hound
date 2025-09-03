"""
Unit tests for agent_core graph loading functionality.
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.agent_core import AutonomousAgent


class TestAgentGraphLoading(unittest.TestCase):
    """Test graph loading functionality in AutonomousAgent."""

    def create_agent(self, agent_id="test_agent"):
        """Helper to create an agent with proper parameters."""
        # Create manifest.json for the agent
        manifest_file = self.graphs_dir / "manifest.json"
        with open(manifest_file, "w") as f:
            json.dump({"repo_path": str(self.temp_dir)}, f)

        # Mock config with required model profiles
        config = {
            "models": {"agent": {"provider": "mock", "model": "mock-model", "api_key": "test"}}
        }

        # Patch the UnifiedLLMClient to avoid actual API calls
        with patch("llm.unified_client.UnifiedLLMClient") as MockLLM:
            mock_llm = MagicMock()
            MockLLM.return_value = mock_llm

            agent = AutonomousAgent(
                graphs_metadata_path=self.graphs_dir / "knowledge_graphs.json",
                manifest_path=self.graphs_dir,
                agent_id=agent_id,
                config=config,
                debug=False,
            )

            # Set the mocked LLM
            agent.llm = mock_llm

            return agent

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.graphs_dir = Path(self.temp_dir) / "graphs"
        self.graphs_dir.mkdir()

        # Create test graph files
        self.test_graphs = {
            "SystemArchitecture": {
                "name": "SystemArchitecture",
                "internal_name": "SystemArchitecture",
                "nodes": [
                    {"id": "sys1", "label": "System Node 1", "type": "component"},
                    {"id": "sys2", "label": "System Node 2", "type": "component"},
                ],
                "edges": [{"source": "sys1", "target": "sys2", "type": "depends"}],
                "metadata": {"version": "1.0"},
            },
            "AuthorizationRolesActions": {
                "name": "AuthorizationRolesActions",
                "internal_name": "AuthorizationRolesActions",
                "focus": "authorization",
                "nodes": [
                    {"id": "auth1", "label": "Admin Role", "type": "role"},
                    {"id": "auth2", "label": "User Role", "type": "role"},
                    {"id": "auth3", "label": "Transfer Function", "type": "function"},
                ],
                "edges": [
                    {"source": "auth1", "target": "auth3", "type": "can_call"},
                    {"source": "auth2", "target": "auth3", "type": "can_call"},
                ],
                "metadata": {"version": "1.0"},
            },
            "DataFlow": {
                "name": "DataFlow",
                "internal_name": "DataFlow",
                "nodes": [
                    {"id": "data1", "label": "Input", "type": "data"},
                    {"id": "data2", "label": "Process", "type": "function"},
                    {"id": "data3", "label": "Output", "type": "data"},
                ],
                "edges": [
                    {"source": "data1", "target": "data2", "type": "flows_to"},
                    {"source": "data2", "target": "data3", "type": "produces"},
                ],
                "metadata": {"version": "1.0"},
            },
        }

        # Write graph files
        for graph_name, graph_data in self.test_graphs.items():
            graph_file = self.graphs_dir / f"graph_{graph_name}.json"
            with open(graph_file, "w") as f:
                json.dump(graph_data, f, indent=2)

        # Create graphs metadata file in the correct format
        self.graphs_metadata = {
            "graphs": {
                "SystemArchitecture": str(self.graphs_dir / "graph_SystemArchitecture.json"),
                "AuthorizationRolesActions": str(
                    self.graphs_dir / "graph_AuthorizationRolesActions.json"
                ),
                "DataFlow": str(self.graphs_dir / "graph_DataFlow.json"),
            }
        }

        # Create a knowledge_graphs.json file (this is what agent expects)
        knowledge_graphs_file = self.graphs_dir / "knowledge_graphs.json"
        with open(knowledge_graphs_file, "w") as f:
            json.dump(self.graphs_metadata, f, indent=2)

        # Create empty hypotheses file in the parent directory (where agent expects it)
        hyp_file = Path(self.temp_dir) / "hypotheses.json"
        with open(hyp_file, "w") as f:
            json.dump(
                {"version": "1.0", "hypotheses": {}, "metadata": {"total": 0, "confirmed": 0}}, f
            )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_agent_initialization(self):
        """Test that agent initializes with correct graph data."""
        agent = self.create_agent()

        # Check that available graphs are loaded
        self.assertIn("SystemArchitecture", agent.available_graphs)
        self.assertIn("AuthorizationRolesActions", agent.available_graphs)
        self.assertIn("DataFlow", agent.available_graphs)

        # Check system graph is auto-loaded
        self.assertIsNotNone(agent.loaded_data.get("system_graph"))
        self.assertEqual(agent.loaded_data["system_graph"]["name"], "SystemArchitecture")

    def test_load_graph_success(self):
        """Test successful graph loading."""
        agent = self.create_agent()

        # Load a non-system graph
        result = agent._load_graph("AuthorizationRolesActions")

        self.assertEqual(result["status"], "success")
        self.assertIn("Loaded AuthorizationRolesActions", result["summary"])
        # Check that summary contains node and edge counts
        self.assertIn("3 nodes", result["summary"])
        self.assertIn("2 edges", result["summary"])

        # Verify graph is in loaded data
        self.assertIn("AuthorizationRolesActions", agent.loaded_data["graphs"])

    def test_load_graph_not_found(self):
        """Test loading a non-existent graph."""
        agent = self.create_agent()

        result = agent._load_graph("NonExistentGraph")

        self.assertEqual(result["status"], "error")
        self.assertIn("Graph not found", result["error"])
        self.assertIn("Available:", result["error"])

    def test_load_graph_case_insensitive(self):
        """Test case-insensitive graph name matching."""
        agent = self.create_agent()

        # Try loading with different case
        result = agent._load_graph("authorizationrolesactions")

        self.assertEqual(result["status"], "success")
        self.assertIn("AuthorizationRolesActions", result["summary"])

    def test_load_graph_name_cleanup(self):
        """Test that graph names are cleaned up properly."""
        agent = self.create_agent()

        # Test with various artifacts
        test_names = [
            '"AuthorizationRolesActions"',
            "'AuthorizationRolesActions'",
            "AuthorizationRolesActions}",
            "AuthorizationRolesActions\n",
            " AuthorizationRolesActions ",
            "AuthorizationRolesActions)",
        ]

        for test_name in test_names:
            result = agent._load_graph(test_name)
            self.assertEqual(result["status"], "success", f"Failed for: {test_name}")

    def test_reload_system_graph(self):
        """Test that reloading system graph returns info message."""
        agent = self.create_agent()

        # Try to reload the system graph
        result = agent._load_graph("SystemArchitecture")

        self.assertEqual(result["status"], "info")
        self.assertIn("already loaded as the system graph", result["summary"])

    def test_save_and_reload_graph_updates(self):
        """Test saving and reloading graph with updates."""
        agent = self.create_agent()

        # Load a graph
        agent._load_graph("DataFlow")

        # Modify the graph
        agent.loaded_data["graphs"]["DataFlow"]["nodes"].append(
            {"id": "data4", "label": "New Node", "type": "data"}
        )

        # Save updates
        success = agent._save_graph_updates("DataFlow", agent.loaded_data["graphs"]["DataFlow"])
        self.assertTrue(success)

        # Create new agent instance to test persistence
        agent2 = self.create_agent("test_agent2")

        # Load the graph in new agent
        result = agent2._load_graph("DataFlow")
        self.assertEqual(result["status"], "success")
        # Check that summary contains the updated node count
        self.assertIn("4 nodes", result["summary"])  # Should have the new node

    def test_concurrent_graph_access(self):
        """Test that concurrent access to graphs works correctly."""
        import threading

        results = []
        errors = []

        def load_and_modify_graph(agent_id, graph_name):
            """Load a graph and modify it."""
            try:
                agent = self.create_agent(agent_id)

                # Load graph
                result = agent._load_graph(graph_name)
                if result["status"] != "success":
                    errors.append(f"Agent {agent_id}: Failed to load {graph_name}")
                    return

                # Modify graph
                agent.loaded_data["graphs"][graph_name]["nodes"].append(
                    {"id": f"node_{agent_id}", "label": f"Added by {agent_id}", "type": "test"}
                )

                # Save updates
                success = agent._save_graph_updates(
                    graph_name, agent.loaded_data["graphs"][graph_name]
                )
                results.append((agent_id, success))

            except Exception as e:
                errors.append(f"Agent {agent_id}: {str(e)}")

        # Create multiple threads to access the same graph
        threads = []
        for i in range(3):
            t = threading.Thread(target=load_and_modify_graph, args=(f"agent_{i}", "DataFlow"))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Check results
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 3)

        # All saves should succeed (with proper locking)
        for agent_id, success in results:
            self.assertTrue(success, f"Agent {agent_id} failed to save")


class TestAgentNodeOperations(unittest.TestCase):
    """Test node-related operations in AutonomousAgent."""

    def create_agent(self, agent_id="test_agent"):
        """Helper to create an agent with proper parameters."""
        # Create manifest.json for the agent
        manifest_file = self.graphs_dir / "manifest.json"
        with open(manifest_file, "w") as f:
            json.dump({"repo_path": str(self.temp_dir)}, f)

        # Mock config with required model profiles
        config = {
            "models": {"agent": {"provider": "mock", "model": "mock-model", "api_key": "test"}}
        }

        # Patch the UnifiedLLMClient to avoid actual API calls
        with patch("llm.unified_client.UnifiedLLMClient") as MockLLM:
            mock_llm = MagicMock()
            MockLLM.return_value = mock_llm

            agent = AutonomousAgent(
                graphs_metadata_path=self.graphs_dir / "knowledge_graphs.json",
                manifest_path=self.graphs_dir,
                agent_id=agent_id,
                config=config,
                debug=False,
            )

            # Set the mocked LLM
            agent.llm = mock_llm

            return agent

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.graphs_dir = Path(self.temp_dir) / "graphs"
        self.graphs_dir.mkdir()

        # Create a test graph with nodes
        self.test_graph = {
            "name": "TestGraph",
            "internal_name": "TestGraph",
            "nodes": [
                {
                    "id": "node1",
                    "label": "Test Node 1",
                    "type": "function",
                    "source_refs": ["card1"],
                },
                {
                    "id": "node2",
                    "label": "Test Node 2",
                    "type": "component",
                    "source_refs": ["card2"],
                    "observations": ["Initial observation"],
                },
            ],
            "edges": [],
            "metadata": {"version": "1.0"},
        }

        # Create card store
        self.card_store = {
            "card1": {
                "relpath": "src/file1.py",
                "char_start": 0,
                "char_end": 100,
                "peek_head": "def function1():",
            },
            "card2": {
                "relpath": "src/file2.py",
                "char_start": 200,
                "char_end": 300,
                "peek_head": "class Component2:",
            },
        }

        # Write files
        graph_file = self.graphs_dir / "graph_TestGraph.json"
        with open(graph_file, "w") as f:
            json.dump(self.test_graph, f)

        card_file = self.graphs_dir / "card_store.json"
        with open(card_file, "w") as f:
            json.dump(self.card_store, f)

        # Create metadata in the correct format
        metadata = {"graphs": {"TestGraph": str(graph_file)}}

        metadata_file = self.graphs_dir / "knowledge_graphs.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        # Create empty hypotheses file in the parent directory (where agent expects it)
        hyp_file = Path(self.temp_dir) / "hypotheses.json"
        with open(hyp_file, "w") as f:
            json.dump(
                {"version": "1.0", "hypotheses": {}, "metadata": {"total": 0, "confirmed": 0}}, f
            )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_nodes(self):
        """Test loading node data."""
        agent = self.create_agent()

        # Load the test graph first
        agent._load_graph("TestGraph")

        # Load nodes - now requires graph_name parameter
        result = agent._load_nodes(["node1", "node2"], graph_name="TestGraph")

        self.assertEqual(result["status"], "success")
        # Check that summary mentions loading 2 nodes
        self.assertIn("Loaded 2 nodes", result["summary"])
        self.assertIn("node1", agent.loaded_data["nodes"])
        self.assertIn("node2", agent.loaded_data["nodes"])

    def test_update_node(self):
        """Test updating node with observations and assumptions."""
        agent = self.create_agent()

        # TestGraph is auto-loaded as system graph since it's the only one
        # Load the node first - now requires graph_name parameter
        agent._load_nodes(["node2"], graph_name="TestGraph")

        # Update node
        params = {
            "node_id": "node2",
            "observations": ["New observation"],
            "assumptions": ["New assumption"],
        }

        result = agent._update_node(params)

        self.assertEqual(result["status"], "success")
        self.assertIn("1 observations, 1 assumptions", result["summary"])

        # Verify the node was updated in the system graph (since TestGraph is auto-loaded as system)
        updated_graph = agent.loaded_data["system_graph"]["data"]
        node2 = next(n for n in updated_graph["nodes"] if n["id"] == "node2")

        self.assertIn("New observation", node2["observations"])
        self.assertIn("New assumption", node2["assumptions"])
        # Original observation should still be there
        self.assertIn("Initial observation", node2["observations"])


if __name__ == "__main__":
    unittest.main()
