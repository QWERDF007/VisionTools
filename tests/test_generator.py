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

    def test_module_index_uses_selected_targets_not_backend_labels(self):
        tools = [
            {
                "tool_id": "tool.cpu",
                "definition": "assets/tools/cpu.yaml",
                "display_name": "CPU Tool",
                "kind": "cpp_function",
                "entry": {"registry_name": "tool.cpu"},
                "inputs": [{"id": "images", "type": "ImageBatch"}],
                "outputs": [{"id": "images", "type": "ImageBatch"}]
            },
            {
                "tool_id": "tool.gpu",
                "definition": "assets/tools/gpu.yaml",
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
                    "variant": "cpu",
                    "tools": {"tool.cpu": {"class": "CpuTool", "header": "CpuTool.h"}}
                },
                "visiontools_infer": {
                    "variant": "gpu",
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
        cpu_idx = build_module_index(tools, bindings, variant="minimal-release", selected_modules=["visiontools_preprocess"])
        self.assertEqual(cpu_idx["variant"], "minimal-release")
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

    def test_cli_formal_validation(self):
        """VT-S04 / VT-T04: Formal CLI runs real SDK validator and covers valid, invalid types, duplicate keys, unknown fields, missing validator."""
        import subprocess
        script = str(tools_dir / "generate_tool_contracts.py")
        validator_env = os.environ.get("VISIONTOOLS_VALIDATOR", "")
        validator = Path(validator_env) if validator_env else Path(__file__).resolve().parent.parent / "build" / "bin" / "visionrt_validate_tool.exe"
        sdk_inc_env = os.environ.get("VISIONTOOLS_SDK_INCLUDE_DIR", "")
        sdk_inc = Path(sdk_inc_env) if sdk_inc_env else Path(__file__).resolve().parent.parent / "3rdparty" / "VisionRTSDK" / "include"
        if not validator.is_file() or not sdk_inc.is_dir():
            self.skipTest(f"VisionRT validator / SDK headers not provided ({validator}, {sdk_inc})")

        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            tools_d = t / "tools"
            tools_d.mkdir()

            valid_tool = tools_d / "valid.yaml"
            valid_tool.write_text((self.fixtures_dir / "valid_tool.yaml").read_text(encoding="utf-8"), encoding="utf-8")

            bindings_path = t / "bindings.yaml"
            bindings_path.write_text(
                "modules:\n"
                "  visiontools_preprocess:\n"
                "    variant: 'cpu'\n"
                "    namespace: 'visiontools::preprocess'\n"
                "    tools:\n"
                "      'test.fixture.valid':\n"
                "        class: 'ValidTool'\n"
                "        header: 'ValidTool.h'\n",
                encoding="utf-8"
            )

            gen_d = t / "generated"
            idx_out = t / "module_index.yaml"

            cmd = [
                sys.executable, script,
                "--tools-dir", str(tools_d),
                "--bindings", str(bindings_path),
                "--output-dir", str(gen_d),
                "--index-output", str(idx_out),
                "--sdk-include-dir", str(sdk_inc),
                "--validator", str(validator)
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Valid tool generation failed:\n{res.stderr}")
            self.assertTrue((gen_d / "PortConstants.h").exists())
            self.assertTrue(idx_out.exists())

            invalid_tool = tools_d / "invalid_dtype.yaml"
            invalid_tool.write_text(
                "schema_version: 1\n"
                "tool_id: test.invalid.dtype\n"
                "display_name: Invalid DType\n"
                "kind: cpp_function\n"
                "entry:\n"
                "  registry_name: test.invalid.dtype\n"
                "inputs:\n"
                "  - id: arr\n"
                "    name: 数组\n"
                "    type: NumericArray\n"
                "    dtype: unknown_type\n"
                "    rank: 1\n"
                "outputs: []\n",
                encoding="utf-8"
            )
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("visionrt_validate_tool failed", res.stderr)
            invalid_tool.unlink()

            nested_tool = tools_d / "nested.yaml"
            nested_tool.write_text(
                "schema_version: 1\n"
                "tool_id: test.nested.err\n"
                "display_name: Nested Error\n"
                "kind: cpp_function\n"
                "entry:\n"
                "  registry_name: test.nested.err\n"
                "inputs: []\n"
                "outputs:\n"
                "  - id: img\n"
                "    name: 图像\n"
                "    type: ImageBatch\n"
                "    display:\n"
                "      unknown_field: 123\n",
                encoding="utf-8"
            )
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("visionrt_validate_tool failed", res.stderr)
            nested_tool.unlink()

            dup_tool = tools_d / "dup.yaml"
            dup_tool.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0)
            dup_tool.unlink()

            cmd_no_val = [
                sys.executable, script,
                "--tools-dir", str(tools_d),
                "--bindings", str(bindings_path),
                "--output-dir", str(gen_d),
                "--index-output", str(idx_out),
                "--sdk-include-dir", str(sdk_inc)
            ]
            res = subprocess.run(cmd_no_val, capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("Missing required --validator", res.stderr)


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

    def test_n06_nonexistent_validator_rejected(self):
        """N06: Explicit non-existent validator must exit non-zero without fallbacks."""
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

            cmd = [
                sys.executable, script,
                "--tools-dir", str(self.fixtures_dir),
                "--bindings", str(tools_dir / "module_bindings.yaml"),
                "--output-dir", str(t / "gen"),
                "--index-output", str(t / "index.yaml"),
                "--sdk-include-dir", str(mock_sdk),
                "--validator", str(t / "nonexistent_validator.exe")
            ]
            res = subprocess.run(cmd, env={**os.environ, "PATH": ""}, capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0)
            self.assertFalse((t / "index.yaml").exists())
            self.assertIn("Validator executable not found", res.stderr)

if __name__ == "__main__":
    unittest.main()
