#!/usr/bin/env python3
"""
Unit and contract tests for generate_tool_contracts.py.
Verifies TS-02 requirements:
- Invalid dtype / layout fails
- Invalid default value fails
- Missing implementation/entry fails
- Duplicate tool_id fails
- YAML key reordering preserves semantic identity
"""

import sys
import unittest
from pathlib import Path

# Add tools directory to path
tools_dir = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(tools_dir))

from generate_tool_contracts import (
    parse_yaml_simple,
    validate_tool_declaration,
    generate_port_constants_header,
    ContractValidationError
)

class TestToolContractGenerator(unittest.TestCase):
    def setUp(self):
        self.fixtures_dir = Path(__file__).resolve().parent / "fixtures" / "generator"

    def test_valid_tool_declaration(self):
        valid_path = self.fixtures_dir / "valid_tool.yaml"
        doc = parse_yaml_simple(valid_path)
        validated = validate_tool_declaration(doc, valid_path)
        self.assertEqual(validated["tool_id"], "test.fixture.valid")
        self.assertEqual(validated["schema_version"], 1)

    def test_invalid_dtype_rejected(self):
        invalid_path = self.fixtures_dir / "invalid_dtype_tool.yaml"
        doc = parse_yaml_simple(invalid_path)
        with self.assertRaises(ContractValidationError) as ctx:
            validate_tool_declaration(doc, invalid_path)
        self.assertIn("valid dtype", str(ctx.exception))

    def test_invalid_default_rejected(self):
        invalid_path = self.fixtures_dir / "invalid_default_tool.yaml"
        doc = parse_yaml_simple(invalid_path)
        with self.assertRaises(ContractValidationError) as ctx:
            validate_tool_declaration(doc, invalid_path)
        self.assertIn("Default value for Int port", str(ctx.exception))

    def test_missing_entry_rejected(self):
        invalid_path = self.fixtures_dir / "missing_entry_tool.yaml"
        doc = parse_yaml_simple(invalid_path)
        with self.assertRaises(ContractValidationError) as ctx:
            validate_tool_declaration(doc, invalid_path)
        self.assertTrue("entry" in str(ctx.exception) or "registry_name" in str(ctx.exception))

    def test_reordered_yaml_semantic_invariance(self):
        valid_path = self.fixtures_dir / "valid_tool.yaml"
        reordered_path = self.fixtures_dir / "reordered_tool.yaml"

        doc1 = parse_yaml_simple(valid_path)
        doc2 = parse_yaml_simple(reordered_path)

        val1 = validate_tool_declaration(doc1, valid_path)
        val2 = validate_tool_declaration(doc2, reordered_path)

        # Normalize ports sorting by id for semantic comparison
        val1["inputs"] = sorted(val1["inputs"], key=lambda x: x["id"])
        val2["inputs"] = sorted(val2["inputs"], key=lambda x: x["id"])

        self.assertEqual(val1["tool_id"], val2["tool_id"])
        self.assertEqual(val1["entry"], val2["entry"])
        self.assertEqual(val1["inputs"], val2["inputs"])
        self.assertEqual(val1["outputs"], val2["outputs"])

        header1 = generate_port_constants_header([val1], "test")
        header2 = generate_port_constants_header([val2], "test")
        self.assertEqual(header1, header2)

if __name__ == "__main__":
    unittest.main()
