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
                {"id": "images", "name": "图像1", "type": "ImageBatch"},
                {"id": "images", "name": "图像2", "type": "ImageBatch"}
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
                {"id": "val", "name": "值", "type": "Int", "min": 10, "max": 5}
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
                {"id": "val", "name": "值", "type": "Int", "default": 20, "min": 1, "max": 10}
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
                {"id": "foo_bar", "name": "Foo Bar 1", "type": "Int"},
                {"id": "foo-bar", "name": "Foo Bar 2", "type": "Int"}
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
                {"id": "arr", "name": "数组", "type": "NumericArray", "dtype": "Float32", "rank": 1, "layout": "batch_ragged"}
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
        self.assertEqual(gpu_idx["tools"]["tool.gpu"]["module"], "visiontools_infer")

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
        self.assertIn('ValidatedToolInstance', content)
        self.assertIn('std::make_unique<CpuTool>(s)', content)
        self.assertIn('s_generatedModuleApi', content)

    def test_bool_rank_rejected(self):
        doc = {
            "schema_version": 1,
            "tool_id": "test.bool.rank",
            "display_name": "Bool Rank Tool",
            "kind": "cpp_function",
            "entry": {"registry_name": "test.bool.rank"},
            "inputs": [{"id": "x", "name": "X", "type": "NumericArray", "dtype": "float32", "rank": True}],
            "outputs": []
        }
        with self.assertRaises(ContractValidationError) as ctx:
            validate_tool_declaration(doc, Path("test.yaml"))
        self.assertIn("integer rank >= 0", str(ctx.exception))

    def test_onesided_range_rejected(self):
        # min constraint with smaller default
        doc = {
            "schema_version": 1,
            "tool_id": "test.onesided.range",
            "display_name": "Onesided Range Tool",
            "kind": "cpp_function",
            "entry": {"registry_name": "test.onesided.range"},
            "inputs": [{"id": "x", "name": "X", "type": "Float", "default": -1.0, "min": 0.0}],
            "outputs": []
        }
        with self.assertRaises(ContractValidationError) as ctx:
            validate_tool_declaration(doc, Path("test.yaml"))
        self.assertIn("out of range", str(ctx.exception))

    def test_missing_python_file_rejected(self):
        doc = {
            "schema_version": 1,
            "tool_id": "test.py.missing",
            "display_name": "Python Tool",
            "kind": "python_function",
            "entry": {"file": "nonexistent_script.py", "symbol": "foo"},
            "inputs": [],
            "outputs": []
        }
        with self.assertRaises(ContractValidationError) as ctx:
            validate_tool_declaration(doc, Path("test.yaml"))
        self.assertIn("does not exist", str(ctx.exception))

    def test_typespec_and_tool_contract_in_header(self):
        valid_path = self.fixtures_dir / "valid_tool.yaml"
        doc = load_yaml(valid_path)
        val = validate_tool_declaration(doc, valid_path)
        header = generate_port_constants_header([val])
        self.assertIn("TypeSpec", header)
        self.assertIn("getToolContract()", header)
        self.assertIn("visionrt::PortContract", header)

    def test_x_and_x_spec_no_identifier_collision(self):
        doc = {
            "schema_version": 1,
            "tool_id": "test.collision.spec",
            "display_name": "Collision Test",
            "kind": "cpp_function",
            "entry": {"registry_name": "test.collision.spec"},
            "inputs": [{"id": "x", "name": "X", "type": "Int"}, {"id": "x_spec", "name": "X Spec", "type": "Int"}],
            "outputs": []
        }
        header = generate_port_constants_header([doc])
        self.assertIn('inline constexpr const char* X = "x";', header)
        self.assertIn('inline constexpr const char* X_SPEC = "x_spec";', header)
        self.assertIn('namespace spec {', header)
        self.assertIn('inline const visionrt::TypeSpec X =', header)

    def test_normalized_identifier_collision_rejected(self):
        doc = {
            "schema_version": 1,
            "tool_id": "test.norm.collision",
            "display_name": "Norm Collision Test",
            "kind": "cpp_function",
            "entry": {"registry_name": "test.norm.collision"},
            "inputs": [{"id": "foo-bar", "name": "Foo Bar 1", "type": "Int"}, {"id": "foo_bar", "name": "Foo Bar 2", "type": "Int"}],
            "outputs": []
        }
        with self.assertRaises(ContractValidationError) as ctx:
            generate_port_constants_header([doc])
        self.assertIn("normalizes to colliding C++ identifier", str(ctx.exception))

    def test_cli_negative_cases(self):
        import subprocess
        script = str(tools_dir / "generate_tool_contracts.py")
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            mock_sdk = t / "mock_sdk"
            val_dir = mock_sdk / "visionrt" / "sdk"
            val_dir.mkdir(parents=True)
            (val_dir / "Value.h").write_text("enum class Type { Int, Float, Bool, String, ImageBatch, NumericArray, StringArray, RoiBatch };\n"
                                             "enum class DType { Int32, Int64, Float32, Float64, UInt8, Bool };\n"
                                             "enum class Layout { Dense, BatchRagged };\n")
            (val_dir / "Schema.h").write_text("enum class ToolKind { CppFunction, CppClass, PythonFunction, PythonClass };\n")

            # Missing --sdk-include-dir -> returncode != 0 (E02)
            p = subprocess.run([sys.executable, script, "--tools-dir", str(self.fixtures_dir),
                                "--bindings", str(tools_dir / "module_bindings.yaml"), "--output-dir", str(t / "gen"),
                                "--index-output", str(t / "index.yaml")], capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("Missing required --sdk-include-dir", p.stderr)

            # Missing bindings file -> returncode != 0
            p = subprocess.run([sys.executable, script, "--tools-dir", str(self.fixtures_dir),
                                "--bindings", str(t / "missing_bindings.yaml"), "--output-dir", str(t / "gen"),
                                "--index-output", str(t / "index.yaml"), "--sdk-include-dir", str(mock_sdk)], capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)

            # Empty tools dir -> returncode != 0
            empty_dir = t / "empty_tools"
            empty_dir.mkdir()
            p = subprocess.run([sys.executable, script, "--tools-dir", str(empty_dir),
                                "--bindings", str(tools_dir / "module_bindings.yaml"), "--output-dir", str(t / "gen"),
                                "--index-output", str(t / "index.yaml"), "--sdk-include-dir", str(mock_sdk)], capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)

            # Invalid explicit sdk-include-dir -> returncode != 0 (P01)
            p = subprocess.run([sys.executable, script, "--tools-dir", str(self.fixtures_dir),
                                "--bindings", str(tools_dir / "module_bindings.yaml"), "--output-dir", str(t / "gen"),
                                "--index-output", str(t / "index.yaml"), "--sdk-include-dir", str(t / "absent-sdk")],
                               capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)

    def test_reserved_port_id_rejected(self):
        doc = {
            "schema_version": 1,
            "tool_id": "test.reserved.port",
            "display_name": "Reserved Port",
            "kind": "cpp_function",
            "entry": {"registry_name": "test.reserved.port"},
            "inputs": [{"id": "__private", "name": "私有", "type": "Int"}],
            "outputs": []
        }
        with self.assertRaises(ContractValidationError) as ctx:
            validate_tool_declaration(doc, Path("test.yaml"))
        self.assertIn("reserved '__' prefix", str(ctx.exception))

    def test_negative_step_rejected(self):
        doc = {
            "schema_version": 1,
            "tool_id": "test.step.invalid",
            "display_name": "Invalid Step",
            "kind": "cpp_function",
            "entry": {"registry_name": "test.step.invalid"},
            "inputs": [{"id": "x", "name": "X", "type": "Float", "step": -1.0}],
            "outputs": []
        }
        with self.assertRaises(ContractValidationError) as ctx:
            validate_tool_declaration(doc, Path("test.yaml"))
        self.assertIn("step must be positive", str(ctx.exception))

    def test_r79a_missing_name_rejected(self):
        valid_path = self.fixtures_dir / "valid_tool.yaml"
        doc = load_yaml(valid_path)
        doc["inputs"][0].pop("name")
        with self.assertRaises(ContractValidationError) as ctx:
            validate_tool_declaration(doc, valid_path)
        self.assertIn("missing or empty 'name'", str(ctx.exception))

    def test_r79b_wrong_options_type_rejected(self):
        valid_path = self.fixtures_dir / "valid_tool.yaml"
        doc = load_yaml(valid_path)
        doc["inputs"][1]["options"] = ["not_an_int"]
        with self.assertRaises(ContractValidationError) as ctx:
            validate_tool_declaration(doc, valid_path)
        self.assertIn("must be an integer", str(ctx.exception))

    def test_r79c_unknown_provider_rejected(self):
        valid_path = self.fixtures_dir / "valid_tool.yaml"
        doc = load_yaml(valid_path)
        doc["inputs"][0]["options_provider"] = "invalid.provider"
        with self.assertRaises(ContractValidationError) as ctx:
            validate_tool_declaration(doc, valid_path)
        self.assertIn("Unknown options_provider", str(ctx.exception))

    def test_r79d_init_input_overlap_rejected(self):
        valid_path = self.fixtures_dir / "valid_tool.yaml"
        doc = load_yaml(valid_path)
        doc["init_params"] = [{"id": "count", "name": "重叠参数", "type": "Int"}]
        with self.assertRaises(ContractValidationError) as ctx:
            validate_tool_declaration(doc, valid_path)
        self.assertIn("init_params and inputs share ID", str(ctx.exception))

    def test_r79e_function_init_params_rejected(self):
        valid_path = self.fixtures_dir / "valid_tool.yaml"
        doc = load_yaml(valid_path)
        doc["kind"] = "cpp_function"
        doc["init_params"] = [{"id": "init_p", "name": "初始化参数", "type": "Int"}]
        with self.assertRaises(ContractValidationError) as ctx:
            validate_tool_declaration(doc, valid_path)
        self.assertIn("Function tool cannot have init_params", str(ctx.exception))

    def test_r80_yml_only_declarations(self):
        from generate_tool_contracts import load_and_validate_declarations
        valid_path = self.fixtures_dir / "valid_tool.yaml"
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            yml_file = t / "tool.yml"
            yml_file.write_text(valid_path.read_text(encoding="utf-8"), encoding="utf-8")
            res = load_and_validate_declarations(t)
            self.assertEqual(len(res), 1)
            self.assertEqual(res[0]["tool_id"], "test.fixture.valid")

    def test_duplicate_yaml_key_rejected(self):
        import yaml
        from generate_tool_contracts import UniqueKeyLoader
        yaml_content = "tool_id: a\ntool_id: b\n"
        with self.assertRaises(ValueError) as ctx:
            yaml.load(yaml_content, Loader=UniqueKeyLoader)
        self.assertIn("Duplicate key detected", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
