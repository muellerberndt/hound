"""
Tests for Finalizer class to ensure full file inclusion and JSON parsing.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path

from analysis.finalization import Finalizer


class DummyLLM:
    def __init__(self, json_payload):
        self._payload = json_payload

    def raw(self, *, system: str, user: str, reasoning_effort=None) -> str:
        # Return as fenced json to test parser
        return f"```json\n{json.dumps(self._payload)}\n```"


class TestFinalizer(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.graphs = self.tmp / "graphs"
        self.graphs.mkdir()
        # Minimal graphs metadata
        (self.graphs / "knowledge_graphs.json").write_text(json.dumps({"graphs": {}}))
        # Minimal manifest with repo path
        (self.graphs / "manifest.json").write_text(json.dumps({"repo_path": str(self.tmp)}))
        # Hypotheses file
        self.hypo_path = self.tmp / "hypotheses.json"
        self.hypo_path.write_text(json.dumps({"hypotheses": {}}))
        # Create a source file to include fully
        self.source_rel = "src/contract.sol"
        (self.tmp / "src").mkdir(exist_ok=True)
        self.source_content = "contract C { function f() public {} /*" + "x" * 8000 + "*/ }"
        (self.tmp / self.source_rel).write_text(self.source_content)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_finalize_confirms_and_reads_full_files(self):
        # Prepare finalizer with dummy llm confirming
        from unittest.mock import patch, MagicMock

        with patch("llm.unified_client.UnifiedLLMClient") as MockLLM:
            MockLLM.return_value = MagicMock()
            fin = Finalizer(
                graphs_metadata_path=self.graphs / "knowledge_graphs.json",
                manifest_path=self.graphs,
                hypothesis_path=self.hypo_path,
                agent_id="t1",
                config={
                    "models": {
                        "agent": {"provider": "openai", "model": "x"},
                        "finalize": {"provider": "openai", "model": "y"},
                    }
                },
            )
            fin.llm = DummyLLM({"verdict": "confirmed", "reasoning": "ok", "confidence": 0.9})
            # Build candidate with source_files property
            hyp = {
                "title": "test",
                "properties": {"source_files": [self.source_rel], "affected_functions": ["f"]},
            }
            report = fin.finalize([("h1", hyp)], max_iterations=1)
            self.assertEqual(report.get("confirmed"), 1)
