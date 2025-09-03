"""
Tests for ReportGenerator to ensure it builds reports and calls LLM.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import json
from unittest.mock import patch, MagicMock

from analysis.report_generator import ReportGenerator


class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # Create project structure
        (self.tmp / "graphs").mkdir()
        # Simple graph file
        (self.tmp / "graphs" / "graph_SystemArchitecture.json").write_text(
            json.dumps(
                {
                    "name": "SystemArchitecture",
                    "nodes": [{"id": "n1", "label": "Comp", "type": "component"}],
                    "edges": [],
                }
            )
        )
        # Hypotheses
        (self.tmp / "hypotheses.json").write_text(
            json.dumps(
                {
                    "hypotheses": {
                        "h1": {
                            "title": "test",
                            "vulnerability_type": "x",
                            "severity": "low",
                            "confidence": 0.9,
                            "status": "confirmed",
                        }
                    }
                }
            )
        )
        # No agent_runs required

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_generate_html_report(self):
        cfg = {"models": {"reporting": {"provider": "openai", "model": "x"}}}
        with patch("analysis.report_generator.UnifiedLLMClient") as MockLLM:
            mock_llm = MagicMock()
            MockLLM.return_value = mock_llm
            mock_llm.raw.return_value = (
                '{"executive_summary": "Executive Summary Here", '
                '"system_overview": "System Overview Here"}'
            )

            rg = ReportGenerator(self.tmp, cfg)
            html = rg.generate(
                project_name="Proj", project_source="repo", title="Report", auditors=["A"]
            )
            self.assertIn("Executive Summary", html)
            self.assertIn("Proj", html)
            self.assertIn("Findings", html)
            self.assertIn("System Overview", html)
