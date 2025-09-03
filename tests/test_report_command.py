"""
Tests for report CLI command to ensure --show-prompt wiring and generation call.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from click.testing import CliRunner
from commands.report import report as report_cmd


class TestReportCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # minimal project structure
        (self.tmp / "graphs").mkdir(parents=True, exist_ok=True)
        (self.tmp / "graphs" / "graph_SystemArchitecture.json").write_text(
            '{"nodes":[],"edges":[]}'
        )
        (self.tmp / "hypotheses.json").write_text('{"hypotheses": {}}')
        # project manager expects a record; patch ProjectManager.get_project

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_report_cli_invokes_generator(self):
        runner = CliRunner()
        with (
            patch("commands.report.ProjectManager") as PM,
            patch("commands.graph.load_config") as load_cfg,
            patch("commands.report.ReportGenerator") as RG,
        ):
            pm = MagicMock()
            PM.return_value = pm
            pm.get_project.return_value = {"path": str(self.tmp), "source_path": str(self.tmp)}
            load_cfg.return_value = {"models": {"reporting": {"provider": "openai", "model": "x"}}}
            rg = MagicMock()
            RG.return_value = rg
            rg.generate.return_value = "<html>OK</html>"

            result = runner.invoke(report_cmd, ["projname", "--show-prompt", "--format", "html"])
            self.assertEqual(result.exit_code, 0)
            rg.generate.assert_called_once()
