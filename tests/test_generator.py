#!/usr/bin/env python3
"""Lightweight generator tests.

Schema semantics belong to the real VisionRTContracts validator. These tests
cover generator-only rules and, when the real validator is available, the
formal CLI path. The formal distribution test never skips a missing validator.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from generate_tool_contracts import (
    ContractValidationError,
    build_module_index,
    generate_module_factory_inc,
    generate_port_constants_header,
    load_and_validate_declarations,
    normalize_declaration,
    validate_generation_rules,
)


def find_validator():
    configured = os.environ.get("VISIONTOOLS_VALIDATOR")
    if configured and Path(configured).is_file():
        return Path(configured)
    discovered = shutil.which("visionrt_validate_tool")
    return Path(discovered) if discovered else None


VALIDATOR = find_validator()


class TestToolContractGenerator(unittest.TestCase):
    def setUp(self):
        self.fixtures_dir = Path(__file__).resolve().parent / "fixtures" / "generator"
        self.validator = VALIDATOR

    def require_validator(self):
        if self.validator is None:
            self.skipTest("lightweight generator unit group: real validator not provided")

    def normalized(self, path):
        self.require_validator()
        return normalize_declaration(self.validator, path)

    def cli(self, tools_dir, bindings=None, output=None, validator="present", extra=None):
        command = [
            sys.executable, str(TOOLS / "generate_tool_contracts.py"),
            "--tools-dir", str(tools_dir),
            "--bindings", str(bindings or TOOLS / "module_bindings.yaml"),
            "--output-dir", str(output or Path(tempfile.mkdtemp()) / "generated"),
            "--index-output", str((output or Path(tempfile.mkdtemp())) / "module_index.yaml"),
            "--variant", "portable-release",
        ]
        if validator == "present":
            command += ["--validator", str(self.validator)]
        elif validator is not None:
            command += ["--validator", str(validator)]
        if extra:
            command += extra
        return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")

    def test_valid_declaration_is_normalized_by_the_real_validator(self):
        document = self.normalized(self.fixtures_dir / "valid_tool.yaml")
        self.assertEqual(document["tool_id"], "test.fixture.valid")
        self.assertEqual(document["schema_version"], 1)
        self.assertIn("__status_code", [port["id"] for port in document["outputs"]])

    def test_missing_entry_is_rejected_by_the_real_validator(self):
        self.require_validator()
        with self.assertRaisesRegex(ContractValidationError, "真实校验器拒绝"):
            normalize_declaration(self.validator, self.fixtures_dir / "missing_entry_tool.yaml")

    def test_duplicate_port_id_is_rejected_by_the_real_validator(self):
        self.require_validator()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.yaml"
            path.write_text(
                "schema_version: 1\n"
                "tool_id: test.duplicate.port\n"
                "display_name: Duplicate Port\n"
                "version: 1.0.0\n"
                "kind: cpp_function\n"
                "entry:\n  registry_name: test.duplicate.port\n"
                "init_params: []\n"
                "inputs:\n"
                "  - {id: images, name: 图像1, type: ImageBatch}\n"
                "  - {id: images, name: 图像2, type: ImageBatch}\n"
                "outputs: []\n",
                encoding="utf-8")
            with self.assertRaisesRegex(ContractValidationError, "真实校验器拒绝"):
                normalize_declaration(self.validator, path)

    def test_reserved_port_id_is_rejected_by_the_real_validator(self):
        self.require_validator()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reserved.yaml"
            path.write_text(
                "schema_version: 1\n"
                "tool_id: test.reserved.port\n"
                "display_name: Reserved Port\n"
                "version: 1.0.0\n"
                "kind: cpp_function\n"
                "entry:\n  registry_name: test.reserved.port\n"
                "init_params: []\n"
                "inputs:\n  - {id: __private, name: 私有, type: Int}\n"
                "outputs: []\n",
                encoding="utf-8")
            with self.assertRaisesRegex(ContractValidationError, "真实校验器拒绝"):
                normalize_declaration(self.validator, path)

    def test_port_identifier_collision_is_a_generation_rule(self):
        document = {
            "schema_version": 1, "tool_id": "test.ident.collision", "display_name": "Ident Collision",
            "version": "1.0.0", "kind": "cpp_function",
            "entry": {"registry_name": "test.ident.collision"},
            "init_params": [], "inputs": [
                {"id": "foo_bar", "name": "Foo Bar 1", "type": "Int"},
                {"id": "foo-bar", "name": "Foo Bar 2", "type": "Int"},
            ], "outputs": [],
        }
        with self.assertRaisesRegex(ContractValidationError, "冲突"):
            validate_generation_rules(document, Path("test.yaml"))

    def test_reordered_yaml_has_identical_normalized_header(self):
        first = self.normalized(self.fixtures_dir / "valid_tool.yaml")
        second = self.normalized(self.fixtures_dir / "reordered_tool.yaml")
        self.assertEqual(generate_port_constants_header([first]), generate_port_constants_header([second]))

    def test_system_outputs_are_not_generated_as_business_contracts(self):
        document = self.normalized(self.fixtures_dir / "valid_tool.yaml")
        header = generate_port_constants_header([document])
        self.assertNotIn("__STATUS_CODE", header)
        self.assertNotIn("__status_code", header)

    def test_generated_header_carries_complete_execution_constraints(self):
        document = self.normalized(ROOT / "assets/tools/builtin/采集/图像采集/图像读取.yaml")
        header = generate_port_constants_header([document])
        self.assertIn('visionrt::Utf8("file")', header)
        self.assertIn("std::vector<visionrt::PortValue>", header)
        self.assertIn("std::optional<visionrt::PortValue>", header)
        self.assertIn("std::int64_t{0}", header)

    def test_typespec_and_tool_contract_in_header(self):
        document = self.normalized(self.fixtures_dir / "valid_tool.yaml")
        header = generate_port_constants_header([document])
        self.assertIn("TypeSpec", header)
        self.assertIn("getToolContract()", header)
        self.assertIn("visionrt::PortContract", header)
        self.assertIn("std::int64_t{1}", header)
        self.assertIn("std::int64_t{100}", header)

    def test_x_and_x_spec_no_identifier_collision(self):
        document = {
            "schema_version": 1, "tool_id": "test.collision.spec", "display_name": "Collision Test",
            "version": "1.0.0", "kind": "cpp_function",
            "entry": {"registry_name": "test.collision.spec"}, "init_params": [],
            "inputs": [{"id": "x", "name": "X", "type": "Int"}, {"id": "x_spec", "name": "X Spec", "type": "Int"}],
            "outputs": [],
        }
        header = generate_port_constants_header([document])
        self.assertIn('inline constexpr const char* X = "x";', header)
        self.assertIn('inline constexpr const char* X_SPEC = "x_spec";', header)
        self.assertIn("namespace spec {", header)
        self.assertIn("inline const visionrt::TypeSpec X =", header)

    def test_normalized_identifier_collision_rejected(self):
        document = {
            "schema_version": 1, "tool_id": "test.norm.collision", "display_name": "Norm Collision Test",
            "version": "1.0.0", "kind": "cpp_function",
            "entry": {"registry_name": "test.norm.collision"}, "init_params": [],
            "inputs": [{"id": "foo-bar", "name": "Foo Bar 1", "type": "Int"},
                       {"id": "foo_bar", "name": "Foo Bar 2", "type": "Int"}],
            "outputs": [],
        }
        with self.assertRaisesRegex(ContractValidationError, "冲突"):
            generate_port_constants_header([document])

    def test_module_index_uses_selected_targets_not_backend_labels(self):
        tools = [
            {
                "tool_id": "tool.cpu", "_source_path": str(self.fixtures_dir / "valid_tool.yaml"),
                "display_name": "CPU Tool", "kind": "cpp_function",
                "entry": {"registry_name": "tool.cpu"}, "init_params": [],
                "inputs": [{"id": "images", "name": "图像", "type": "ImageBatch"}],
                "outputs": [{"id": "images", "name": "图像", "type": "ImageBatch"}],
            },
            {
                "tool_id": "tool.gpu", "_source_path": str(self.fixtures_dir / "valid_tool.yaml"),
                "display_name": "GPU Tool", "kind": "cpp_class",
                "entry": {"registry_name": "tool.gpu"}, "init_params": [],
                "inputs": [{"id": "images", "name": "图像", "type": "ImageBatch"}],
                "outputs": [{"id": "rois", "name": "ROI", "type": "RoiBatch"}],
            },
        ]
        bindings = {
            "modules": {
                "visiontools_preprocess": {
                    "variant": "cpu",
                    "tools": {"tool.cpu": {"class": "CpuTool", "header": "CpuTool.h"}},
                },
                "visiontools_infer": {
                    "variant": "gpu",
                    "tools": {"tool.gpu": {"class": "GpuTool", "header": "GpuTool.h"}},
                },
            }
        }
        gpu_index = build_module_index(tools, bindings, variant="gpu")
        self.assertEqual(gpu_index["variant"], "gpu")
        self.assertIn("tool.cpu", gpu_index["tools"])
        self.assertIn("tool.gpu", gpu_index["tools"])
        self.assertEqual(gpu_index["tools"]["tool.gpu"]["module"], "visiontools_infer")

        cpu_index = build_module_index(tools, bindings, variant="minimal-release",
                                       selected_modules=["visiontools_preprocess"])
        self.assertEqual(cpu_index["variant"], "minimal-release")
        self.assertIn("tool.cpu", cpu_index["tools"])
        self.assertNotIn("tool.gpu", cpu_index["tools"])

    def test_generate_module_factory_inc_uses_one_validated_wrapper(self):
        tools_dict = {"tool.cpu": {"tool_id": "tool.cpu", "entry": {"registry_name": "tool.cpu"}}}
        module = {
            "namespace": "test::preprocess",
            "tools": {"tool.cpu": {"class": "CpuTool", "header": "CpuTool.h"}},
        }
        content = generate_module_factory_inc("visiontools_preprocess", tools_dict, module)
        self.assertIn('#include "CpuTool.h"', content)
        self.assertIn('std::strcmp(registry_name, "tool.cpu") == 0', content)
        self.assertIn("ValidatedToolInstance", content)
        self.assertIn("std::make_unique<CpuTool>(s)", content)
        self.assertIn("s_generatedModuleApi", content)
        self.assertIn("初始化参数不在声明中", content)
        self.assertNotIn("return new CpuTool(s);", content)

    def test_generate_module_factory_rejects_missing_definition(self):
        with self.assertRaisesRegex(ContractValidationError, "未声明"):
            generate_module_factory_inc("visiontools_preprocess", {},
                                        {"namespace": "test", "tools": {"tool.missing": {"class": "Missing"}}})

    def test_missing_python_file_rejected(self):
        document = {
            "schema_version": 1, "tool_id": "test.py.missing", "display_name": "Python Tool",
            "version": "1.0.0", "kind": "python_function",
            "entry": {"file": "nonexistent_script.py", "symbol": "foo"},
            "init_params": [], "inputs": [], "outputs": [],
        }
        with self.assertRaisesRegex(ContractValidationError, "入口脚本不存在"):
            validate_generation_rules(document, Path("test.yaml"))

    def test_r80_yml_only_declarations(self):
        self.require_validator()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "tool.yml"
            target.write_text((self.fixtures_dir / "valid_tool.yaml").read_text(encoding="utf-8"), encoding="utf-8")
            declarations = load_and_validate_declarations(Path(temporary), self.validator)
            self.assertEqual(len(declarations), 1)
            self.assertEqual(declarations[0]["tool_id"], "test.fixture.valid")

    def test_cli_negative_cases(self):
        self.require_validator()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "generated"
            index = output / "module_index.yaml"
            missing_validator = self.cli(self.fixtures_dir, output=output, validator=None)
            self.assertNotEqual(missing_validator.returncode, 0)
            self.assertIn("--validator", missing_validator.stderr)
            self.assertFalse(index.exists())

            missing_bindings = self.cli(self.fixtures_dir, bindings=root / "missing_bindings.yaml",
                                        output=output, validator="present")
            self.assertNotEqual(missing_bindings.returncode, 0)
            self.assertFalse(index.exists())

            empty = root / "empty_tools"
            empty.mkdir()
            empty_result = self.cli(empty, output=output, validator="present")
            self.assertNotEqual(empty_result.returncode, 0)
            self.assertFalse(index.exists())

            nonexistent = self.cli(self.fixtures_dir, output=output, validator=root / "missing-validator")
            self.assertNotEqual(nonexistent.returncode, 0)
            self.assertIn("真实校验器不存在", nonexistent.stderr)
            self.assertFalse(index.exists())

    def test_cli_formal_validation(self):
        self.require_validator()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tools_dir = root / "tools"
            tools_dir.mkdir()
            valid_tool = tools_dir / "valid.yaml"
            valid_tool.write_text((self.fixtures_dir / "valid_tool.yaml").read_text(encoding="utf-8"), encoding="utf-8")
            bindings = root / "bindings.yaml"
            bindings.write_text(
                "modules:\n"
                "  visiontools_preprocess:\n"
                "    namespace: 'visiontools::preprocess'\n"
                "    tools:\n"
                "      'test.fixture.valid':\n"
                "        class: 'ValidTool'\n"
                "        header: 'ValidTool.h'\n",
                encoding="utf-8")
            generated = root / "generated"
            index = root / "module_index.yaml"
            command = [
                sys.executable, str(TOOLS / "generate_tool_contracts.py"),
                "--tools-dir", str(tools_dir), "--bindings", str(bindings),
                "--output-dir", str(generated), "--index-output", str(index),
                "--variant", "portable-release", "--validator", str(self.validator),
            ]
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((generated / "PortConstants.h").exists())
            self.assertTrue(index.exists())

            invalid = tools_dir / "invalid.yaml"
            invalid.write_text(
                "schema_version: 1\n"
                "tool_id: test.invalid.dtype\n"
                "display_name: Invalid DType\n"
                "version: 1.0.0\n"
                "kind: cpp_function\n"
                "entry:\n  registry_name: test.invalid.dtype\n"
                "init_params: []\n"
                "inputs:\n"
                "  - {id: arr, name: 数组, type: NumericArray, dtype: unknown_type, rank: 1}\n"
                "outputs: []\n",
                encoding="utf-8")
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("真实校验器拒绝", result.stderr)
            invalid.unlink()

            unknown = tools_dir / "unknown.yaml"
            unknown.write_text(
                "schema_version: 1\n"
                "tool_id: test.nested.err\n"
                "display_name: Nested Error\n"
                "version: 1.0.0\n"
                "kind: cpp_function\n"
                "entry:\n  registry_name: test.nested.err\n"
                "init_params: []\n"
                "inputs: []\n"
                "outputs:\n"
                "  - id: img\n"
                "    name: 图像\n"
                "    type: ImageBatch\n"
                "    display:\n"
                "      unknown_field: 123\n",
                encoding="utf-8")
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("真实校验器拒绝", result.stderr)
            unknown.unlink()

            duplicate = tools_dir / "duplicate.yaml"
            duplicate.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertNotEqual(result.returncode, 0)
            duplicate.unlink()


if __name__ == "__main__":
    unittest.main()
