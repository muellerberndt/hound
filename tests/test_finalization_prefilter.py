"""
Tests for finalization pre-filtering robustness and parsing.
"""

import unittest
from typing import Dict

from analysis.finalization_prefilter import pre_filter_hypotheses


class DummyLLM:
    def __init__(self, response_text: str):
        self._text = response_text

    def raw(self, *, system: str, user: str) -> str:
        return self._text


class TestFinalizationPrefilter(unittest.TestCase):
    def setUp(self):
        self.hyps: Dict[str, Dict] = {
            "h1": {
                "title": "owner can pause",
                "vulnerability_type": "access_control",
                "confidence": 0.8,
            },
            "h2": {
                "title": "reentrancy in withdraw",
                "vulnerability_type": "reentrancy",
                "confidence": 0.9,
            },
        }

    def test_prefilter_parses_fenced_json(self):
        llm = DummyLLM(
            """Here is my response:\n```json\n{\n  \"decisions\": [\n    {\"index\": 0, \"decision\": \"FILTER\", \"reason\": \"Requires admin role\"},\n    {\"index\": 1, \"decision\": \"KEEP\"}\n  ]\n}\n```\nThanks."""
        )
        candidates, filtered = pre_filter_hypotheses(self.hyps, threshold=0.5, llm=llm)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(filtered), 1)

    def test_prefilter_direct_json(self):
        llm = DummyLLM(
            '{"decisions": [{"index":0, "decision":"KEEP"}, {"index":1, "decision":"KEEP"}]}'
        )
        candidates, filtered = pre_filter_hypotheses(self.hyps, threshold=0.5, llm=llm)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(len(filtered), 0)
