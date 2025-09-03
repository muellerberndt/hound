"""
Tests for robust JSON extraction utility.
"""

import unittest

from utils.json_utils import extract_json_object


class TestJsonUtils(unittest.TestCase):
    def test_extract_from_fenced(self):
        text = """
        Here is output:
        ```json
        {"a": 1, "b": 2}
        ```
        Thanks.
        """
        obj = extract_json_object(text)
        self.assertIsInstance(obj, dict)
        self.assertEqual(obj.get("a"), 1)

    def test_extract_balanced(self):
        text = 'Noise before {\n  "k": [1,2,3,],\n} and after'
        obj = extract_json_object(text)
        self.assertIsInstance(obj, dict)
        self.assertEqual(obj.get("k"), [1, 2, 3])
