"""
Unit tests for GraphStore and related graph loading functionality.
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

from analysis.concurrent_knowledge import GraphStore, HypothesisStore, ConcurrentFileStore


class TestConcurrentFileStore(unittest.TestCase):
    """Test the base ConcurrentFileStore class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test_store.json"

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_lock_acquisition_and_release(self):
        """Test that locks are properly acquired and released."""

        class TestStore(ConcurrentFileStore):
            def _get_empty_data(self):
                return {"test": "data"}

        store = TestStore(self.test_file, "test_agent")

        # Test acquiring lock
        lock = store._acquire_lock(timeout=1.0)
        self.assertIsNotNone(lock)

        # Lock file should exist
        lock_path = self.test_file.with_suffix(".lock")
        self.assertTrue(lock_path.exists())

        # Release lock
        store._release_lock(lock)

        # Lock file should be cleaned up
        self.assertFalse(lock_path.exists())

    def test_atomic_update(self):
        """Test atomic update operations."""

        class TestStore(ConcurrentFileStore):
            def _get_empty_data(self):
                return {"counter": 0}

        store = TestStore(self.test_file, "test_agent")

        # Define update function
        def increment_counter(data):
            data["counter"] += 1
            return data, data["counter"]

        # Perform atomic update
        result = store.update_atomic(increment_counter)
        self.assertEqual(result, 1)

        # Verify data was saved
        with open(self.test_file) as f:
            data = json.load(f)
        self.assertEqual(data["counter"], 1)

        # Test multiple updates
        for i in range(5):
            result = store.update_atomic(increment_counter)

        with open(self.test_file) as f:
            data = json.load(f)
        self.assertEqual(data["counter"], 6)


class TestGraphStore(unittest.TestCase):
    """Test the GraphStore class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_graph_file = Path(self.temp_dir) / "test_graph.json"

        # Load actual graph data from fixture
        fixture_path = Path(__file__).parent / "fixtures" / "test_graph.json"
        if fixture_path.exists():
            with open(fixture_path) as f:
                self.sample_graph_data = json.load(f)
        else:
            # Fallback sample data if fixture doesn't exist
            self.sample_graph_data = {
                "name": "TestGraph",
                "internal_name": "test_graph",
                "nodes": [
                    {"id": "node1", "label": "Node 1", "type": "function"},
                    {"id": "node2", "label": "Node 2", "type": "function"},
                ],
                "edges": [{"source": "node1", "target": "node2", "type": "calls"}],
                "metadata": {"version": "1.0"},
            }

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_graph(self):
        """Test saving and loading a complete graph."""
        store = GraphStore(self.test_graph_file, "test_agent")

        # Save graph
        success = store.save_graph(self.sample_graph_data)
        self.assertTrue(success)

        # Load graph
        loaded_data = store.load_graph()
        self.assertIsNotNone(loaded_data)

        # Verify key fields
        self.assertEqual(loaded_data.get("name"), self.sample_graph_data.get("name"))
        self.assertEqual(
            len(loaded_data.get("nodes", [])), len(self.sample_graph_data.get("nodes", []))
        )
        self.assertEqual(
            len(loaded_data.get("edges", [])), len(self.sample_graph_data.get("edges", []))
        )

    def test_concurrent_access(self):
        """Test that concurrent access is handled properly."""
        import threading
        import time

        store1 = GraphStore(self.test_graph_file, "agent1")
        store2 = GraphStore(self.test_graph_file, "agent2")

        results = []

        def save_with_delay(store, data, delay):
            """Save data with a simulated delay."""

            def update(existing):
                time.sleep(delay)
                return data, True

            result = store.update_atomic(update)
            results.append(result)

        # Create two different graphs
        graph1 = self.sample_graph_data.copy()
        graph1["name"] = "Graph1"

        graph2 = self.sample_graph_data.copy()
        graph2["name"] = "Graph2"

        # Start concurrent saves
        t1 = threading.Thread(target=save_with_delay, args=(store1, graph1, 0.1))
        t2 = threading.Thread(target=save_with_delay, args=(store2, graph2, 0.1))

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        # Both operations should succeed
        self.assertEqual(len(results), 2)
        self.assertTrue(all(results))

        # The last write should win
        final_data = store1.load_graph()
        self.assertIn(final_data["name"], ["Graph1", "Graph2"])

    def test_update_nodes(self):
        """Test updating specific nodes in a graph."""
        store = GraphStore(self.test_graph_file, "test_agent")

        # Save initial graph
        store.save_graph(self.sample_graph_data)

        # Update specific nodes
        node_updates = [
            {"id": "node1", "observations": ["New observation"]},
            {"id": "node2", "assumptions": ["New assumption"]},
        ]

        success = store.update_nodes(node_updates)
        self.assertTrue(success)

        # Verify updates
        loaded_data = store.load_graph()
        nodes = {n["id"]: n for n in loaded_data.get("nodes", [])}

        if "node1" in nodes:
            self.assertIn("observations", nodes["node1"])
            self.assertEqual(nodes["node1"]["observations"], ["New observation"])

        if "node2" in nodes:
            self.assertIn("assumptions", nodes["node2"])
            self.assertEqual(nodes["node2"]["assumptions"], ["New assumption"])


class TestHypothesisStore(unittest.TestCase):
    """Test the HypothesisStore class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_hyp_file = Path(self.temp_dir) / "test_hypotheses.json"
        self.store = HypothesisStore(self.test_hyp_file, "test_agent")

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_propose_hypothesis(self):
        """Test proposing a new hypothesis."""
        from analysis.concurrent_knowledge import Hypothesis

        hyp = Hypothesis(
            title="Test Vulnerability",
            description="A test vulnerability description",
            vulnerability_type="access_control",
            severity="high",
            confidence=0.7,
            node_refs=["node1", "node2"],
        )

        success, hyp_id = self.store.propose(hyp)
        self.assertTrue(success)
        self.assertTrue(hyp_id.startswith("hyp_"))

        # Verify hypothesis was stored
        data = self.store._load_data()
        self.assertIn(hyp_id, data["hypotheses"])
        self.assertEqual(data["hypotheses"][hyp_id]["title"], "Test Vulnerability")

    def test_duplicate_detection(self):
        """Test that duplicate hypotheses are detected."""
        from analysis.concurrent_knowledge import Hypothesis

        hyp1 = Hypothesis(
            title="Duplicate Test",
            description="This is a test",
            vulnerability_type="access_control",
            severity="high",
            node_refs=["node1"],
        )

        # First proposal should succeed
        success1, hyp_id1 = self.store.propose(hyp1)
        self.assertTrue(success1)

        # Duplicate with same title should fail
        hyp2 = Hypothesis(
            title="Duplicate Test",  # Same title
            description="Different description",
            vulnerability_type="access_control",
            severity="medium",
            node_refs=["node2"],
        )

        success2, hyp_id2 = self.store.propose(hyp2)
        self.assertFalse(success2)
        self.assertIn("Duplicate", hyp_id2)

    def test_add_evidence(self):
        """Test adding evidence to a hypothesis."""
        from analysis.concurrent_knowledge import Hypothesis, Evidence

        # Create hypothesis
        hyp = Hypothesis(
            title="Evidence Test",
            description="Testing evidence",
            vulnerability_type="input_validation",
            severity="medium",
            node_refs=["node1"],
        )

        success, hyp_id = self.store.propose(hyp)
        self.assertTrue(success)

        # Add supporting evidence
        evidence = Evidence(
            description="Found validation bypass",
            type="supports",
            confidence=0.8,
            node_refs=["node1"],
        )

        success = self.store.add_evidence(hyp_id, evidence)
        self.assertTrue(success)

        # Verify evidence was added
        data = self.store._load_data()
        hyp_data = data["hypotheses"][hyp_id]
        self.assertEqual(len(hyp_data["evidence"]), 1)
        self.assertEqual(hyp_data["evidence"][0]["description"], "Found validation bypass")

    def test_adjust_confidence(self):
        """Test adjusting hypothesis confidence."""
        from analysis.concurrent_knowledge import Hypothesis

        hyp = Hypothesis(
            title="Confidence Test",
            description="Testing confidence adjustment",
            vulnerability_type="race_condition",
            severity="low",
            confidence=0.5,
            node_refs=["node1"],
        )

        success, hyp_id = self.store.propose(hyp)
        self.assertTrue(success)

        # Adjust confidence
        success = self.store.adjust_confidence(hyp_id, 0.9, "Found strong evidence")
        self.assertTrue(success)

        # Verify confidence was updated
        data = self.store._load_data()
        self.assertEqual(data["hypotheses"][hyp_id]["confidence"], 0.9)

        # Test auto-rejection at low confidence
        success = self.store.adjust_confidence(hyp_id, 0.05, "Evidence refuted")
        self.assertTrue(success)

        data = self.store._load_data()
        self.assertEqual(data["hypotheses"][hyp_id]["status"], "rejected")

    def test_get_by_node(self):
        """Test retrieving hypotheses by node ID."""
        from analysis.concurrent_knowledge import Hypothesis

        # Create hypotheses for different nodes
        hyp1 = Hypothesis(
            title="Node1 Vulnerability",
            description="A vulnerability in node1",
            vulnerability_type="injection",
            severity="high",
            node_refs=["node1", "node2"],
        )

        hyp2 = Hypothesis(
            title="Node2 Vulnerability",
            description="A vulnerability in node2",
            vulnerability_type="xss",
            severity="medium",
            node_refs=["node2", "node3"],
        )

        hyp3 = Hypothesis(
            title="Node3 Vulnerability",
            description="A vulnerability in node3",
            vulnerability_type="csrf",
            severity="low",
            node_refs=["node3"],
        )

        # Propose all hypotheses
        self.store.propose(hyp1)
        self.store.propose(hyp2)
        self.store.propose(hyp3)

        # Test get_by_node for node2 (should return 2 hypotheses)
        node2_hyps = self.store.get_by_node("node2")
        self.assertEqual(len(node2_hyps), 2)
        titles = [h["title"] for h in node2_hyps]
        self.assertIn("Node1 Vulnerability", titles)
        self.assertIn("Node2 Vulnerability", titles)

        # Test get_by_node for node3 (should return 2 hypotheses)
        node3_hyps = self.store.get_by_node("node3")
        self.assertEqual(len(node3_hyps), 2)

        # Test get_by_node for non-existent node
        node4_hyps = self.store.get_by_node("node4")
        self.assertEqual(len(node4_hyps), 0)

    def test_concurrent_hypothesis_operations(self):
        """Test that concurrent hypothesis operations are handled safely."""
        import threading
        from analysis.concurrent_knowledge import Hypothesis

        results = []
        errors = []

        def propose_hypothesis(store, title, node_id):
            """Propose a hypothesis in a thread."""
            try:
                hyp = Hypothesis(
                    title=title,
                    description=f"Description for {title}",
                    vulnerability_type="test",
                    severity="medium",
                    node_refs=[node_id],
                )
                success, hyp_id = store.propose(hyp)
                results.append((title, success, hyp_id))
            except Exception as e:
                errors.append(str(e))

        # Create multiple threads proposing different hypotheses
        threads = []
        for i in range(5):
            t = threading.Thread(
                target=propose_hypothesis, args=(self.store, f"Concurrent Hyp {i}", f"node{i}")
            )
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Check results
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 5)

        # All should succeed since they have different titles
        for title, success, hyp_id in results:
            self.assertTrue(success, f"Failed to propose: {title}")

        # Verify all hypotheses are in the store
        data = self.store._load_data()
        self.assertEqual(len(data["hypotheses"]), 5)

    def test_hypothesis_status_transitions(self):
        """Test hypothesis status transitions through evidence."""
        from analysis.concurrent_knowledge import Hypothesis, Evidence

        hyp = Hypothesis(
            title="Status Test",
            description="Testing status transitions",
            vulnerability_type="test",
            severity="medium",
            confidence=0.5,
            node_refs=["node1"],
        )

        success, hyp_id = self.store.propose(hyp)
        self.assertTrue(success)

        # Initial status should be "proposed"
        data = self.store._load_data()
        self.assertEqual(data["hypotheses"][hyp_id]["status"], "proposed")

        # Add supporting evidence - should change to "investigating"
        evidence1 = Evidence(
            description="Found initial evidence",
            type="supports",
            confidence=0.6,
            node_refs=["node1"],
        )
        self.store.add_evidence(hyp_id, evidence1)

        data = self.store._load_data()
        self.assertEqual(data["hypotheses"][hyp_id]["status"], "investigating")

        # Add more supporting evidence - should change to "supported"
        for i in range(3):
            evidence = Evidence(
                description=f"Supporting evidence {i + 2}",
                type="supports",
                confidence=0.7,
                node_refs=["node1"],
            )
            self.store.add_evidence(hyp_id, evidence)

        data = self.store._load_data()
        self.assertEqual(data["hypotheses"][hyp_id]["status"], "supported")

        # Add overwhelming refuting evidence - should change to "refuted"
        for i in range(10):
            evidence = Evidence(
                description=f"Refuting evidence {i}",
                type="refutes",
                confidence=0.8,
                node_refs=["node1"],
            )
            self.store.add_evidence(hyp_id, evidence)

        data = self.store._load_data()
        self.assertEqual(data["hypotheses"][hyp_id]["status"], "refuted")


if __name__ == "__main__":
    unittest.main()
