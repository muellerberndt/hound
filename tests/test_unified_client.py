"""
Tests for UnifiedLLMClient provider selection and raw passthrough.
"""

import unittest
from unittest.mock import patch, MagicMock

from llm.unified_client import UnifiedLLMClient


class DummyProvider:
    provider_name = "dummy"
    supports_thinking = False

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs

    def raw(self, *, system: str, user: str) -> str:
        return f"SYS:{system}|USER:{user}"


class TestUnifiedClient(unittest.TestCase):
    def test_selects_openai_provider(self):
        cfg = {"models": {"reporting": {"provider": "openai", "model": "x"}}}
        with patch("llm.unified_client.OpenAIProvider", DummyProvider):
            uc = UnifiedLLMClient(cfg, profile="reporting")
            self.assertEqual(uc.provider.provider_name, "dummy")
            out = uc.raw(system="S", user="U")
            self.assertIn("SYS:S|USER:U", out)
