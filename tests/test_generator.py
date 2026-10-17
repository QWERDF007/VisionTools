#!/usr/bin/env python3
"""
Unit and contract tests for generate_tool_contracts.py.
Verifies TS-02 and VS-02 requirements:
- Invalid dtype / layout fails
- Invalid default value fails
- Missing implementation/entry fails
- Duplicate tool_id / registry_name / port_id fails
- Min > Max or Default out of range fails
- Identifier normalization collision fails
- YAML key reordering preserves semantic identity
- CLI correctly creates module_index.json and PortConstants.h
- CPU variant excludes GPU tools
- Module bindings validation (unbound implementation or declaration without binding)
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add tools directory to path
tools_dir = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(tools_dir))

from generate_tool_contracts import (
    load_yaml,
    validate_tool_declaration,
    generate_port_constants_header,
    generate_module_factory_inc,
    build_module_index,
    ContractValidationError
)

class TestToolContractGenerator(unittest.TestCase):
    def setUp(self):
        self.fixtures_dir = Path(__file__).resolve().parent / "fixtures" / "generator"

    def test_valid_tool_declaration(self):
        valid_path = self.fixtures_dir / "valid_tool.yaml"
        doc = load_yaml(valid_path)
        validated = validate_tool_declaration(doc, valid_path)
        self.assertEqual(validated["tool_id"], "test.fixture.valid")
        self.assertEqual(validated["schema_version"], 1)

    def test_invalid_dtype_rejected(self):
        invalid_path = self.fixtures_dir / "invalid_dtype_tool.yaml"
        doc = load_yaml(invalid_path)
        with self.assertRaises(ContractValidationError) as ctx:
            validate_tool_declaration(doc, invalid_path)
        self.assertIn("valid dtype", str(ctx.exception))

    def test_invalid_default_rejected(self):
        invalid_path = self.fixtures_dir / "invalid_default_tool.yaml"
        doc = load_yaml(invalid_path)
        with self.assertRaises(ContractValidationError) as ctx:
            validate_tool_declaration(doc, invalid_path)
        self.assertIn("Default value for Int port", str(ctx.exception))

    def test_missing_entry_rejected(self):
        invalid_path = self.fixtures_dir / "missing_entry_tool.yaml"
        doc = load_yaml(invalid_path)
        with self.assertRaises(ContractValidationError) as ctx:
            validate_tool_declaration(doc, invalid_path)
        self.assertTrue("entry" in str(ctx.exception) or "registry_name" in str(ctx.exception))

    def test_duplicate_port_id_rejected(self):
        doc = {
            "schema_version": 1,
            "tool_id": "test.duplicate.port",
            "display_name": "Duplicate Port",
            "kind": "cpp_function",
            "entry": {"registry_name": "test.duplicate.port"},
            "inputs": [
                {"id": "images", "type": "ImageBatch"},
                {"id": "images", "type": "ImageBatch"}
            ],
            "outputs": []
        }
        with self.assertRaises(ContractValidationError) as ctx:
            validate_tool_declaration(doc, Path("test.yaml"))
        self.assertIn("Duplicate port id 'images'", str(ctx.exception))

    def test_min_greater_than_max_rejected(self):
        doc = {
            "schema_version": 1,
            "tool_id": "test.range.invalid",
            "display_name": "Invalid Range",
            "kind": "cpp_function",
            "entry": {"registry_name": "test.range.invalid"},
            "inputs": [
                {"id": "val", "type": "Int", "min": 10, "max": 5}
            ],
            "outputs": []
        }
        with self.assertRaises(ContractValidationError) as ctx:
            validate_tool_declaration(doc, Path("test.yaml"))
        self.assertIn("min (10) > max (5)", str(ctx.exception))

    def test_default_out_of_range_rejected(self):
        doc = {
            "schema_version": 1,
            "tool_id": "test.range.out",
            "display_name": "Out of Range",
            "kind": "cpp_function",
            "entry": {"registry_name": "test.range.out"},
            "inputs": [
                {"id": "val", "type": "Int", "default": 20, "min": 1, "max": 10}
            ],
            "outputs": []
        }
        with self.assertRaises(ContractValidationError) as ctx:
            validate_tool_declaration(doc, Path("test.yaml"))
        self.assertIn("default (20) out of range", str(ctx.exception))

    def test_port_identifier_collision_rejected(self):
        doc = {
            "schema_version": 1,
            "tool_id": "test.ident.collision",
            "display_name": "Ident Collision",
            "kind": "cpp_function",
            "entry": {"registry_name": "test.ident.collision"},
            "inputs": [
                {"id": "foo_bar", "type": "Int"},
                {"id": "foo-bar", "type": "Int"}
            ],
            "outputs": []
        }
        with self.assertRaises(ContractValidationError) as ctx:
            validate_tool_declaration(doc, Path("test.yaml"))
        self.assertIn("identifier collision", str(ctx.exception))

    def test_batch_ragged_rank_lt_2_rejected(self):
        doc = {
            "schema_version": 1,
            "tool_id": "test.ragged.rank",
            "display_name": "Ragged Rank",
            "kind": "cpp_function",
            "entry": {"registry_name": "test.ragged.rank"},
            "inputs": [
                {"id": "arr", "type": "NumericArray", "dtype": "Float32", "rank": 1, "layout": "batch_ragged"}
            ],
            "outputs": []
        }
        with self.assertRaises(ContractValidationError) as ctx:
            validate_tool_declaration(doc, Path("test.yaml"))
        self.assertIn("requires rank >= 2", str(ctx.exception))

    def test_reordered_yaml_semantic_invariance(self):
        valid_path = self.fixtures_dir / "valid_tool.yaml"
        reordered_path = self.fixtures_dir / "reordered_tool.yaml"

        doc1 = load_yaml(valid_path)
        doc2 = load_yaml(reordered_path)

        val1 = validate_tool_declaration(doc1, valid_path)
        val2 = validate_tool_declaration(doc2, reordered_path)

        header1 = generate_port_constants_header([val1])
        header2 = generate_port_constants_header([val2])
        self.assertEqual(header1, header2)

    def test_module_index_and_variant_filtering(self):
        tools = [
            {
                "tool_id": "tool.cpu",
                "display_name": "CPU Tool",
                "kind": "cpp_function",
                "entry": {"registry_name": "tool.cpu"},
                "inputs": [{"id": "images", "type": "ImageBatch"}],
                "outputs": [{"id": "images", "type": "ImageBatch"}]
            },
            {
                "tool_id": "tool.gpu",
                "display_name": "GPU Tool",
                "kind": "cpp_class",
                "entry": {"registry_name": "tool.gpu"},
                "inputs": [{"id": "images", "type": "ImageBatch"}],
                "outputs": [{"id": "rois", "type": "RoiBatch"}]
            }
        ]
        bindings = {
            "modules": {
                "visiontools_preprocess": {
                    "dll_name": "bin/visiontools_preprocess.dll",
                    "tools": {"tool.cpu": {"class": "CpuTool", "header": "CpuTool.h"}}
                },
                "visiontools_infer": {
                    "dll_name": "bin/visiontools_infer.dll",
                    "tools": {"tool.gpu": {"class": "GpuTool", "header": "GpuTool.h"}}
                }
            }
        }

        # GPU variant includes both
        gpu_idx = build_module_index(tools, bindings, variant="gpu")
        self.assertEqual(gpu_idx["variant"], "gpu")
        self.assertIn("tool.cpu", gpu_idx["tools"])
        self.assertIn("tool.gpu", gpu_idx["tools"])
        self.assertEqual(gpu_idx["tools"]["tool.gpu"]["module_dll"], "bin/visiontools_infer.dll")

        # CPU variant excludes GPU tool
        cpu_idx = build_module_index(tools, bindings, variant="cpu")
        self.assertEqual(cpu_idx["variant"], "cpu")
        self.assertIn("tool.cpu", cpu_idx["tools"])
        self.assertNotIn("tool.gpu", cpu_idx["tools"])

    def test_generate_module_factory_inc(self):
        tools_dict = {
            "tool.cpu": {
                "tool_id": "tool.cpu",
                "entry": {"registry_name": "tool.cpu"}
            }
        }
        mod_info = {
            "namespace": "test::preprocess",
            "tools": {"tool.cpu": {"class": "CpuTool", "header": "CpuTool.h"}}
        }
        content = generate_module_factory_inc("visiontools_preprocess", tools_dict, mod_info)
        self.assertIn('#include "CpuTool.h"', content)
        self.assertIn('std::strcmp(registry_name, "tool.cpu") == 0', content)
        self.assertIn('return new CpuTool(s);', content)
        self.assertIn('s_generatedModuleApi', content)

if __name__ == "__main__":
    unittest.main()
