#!/usr/bin/env python3
"""Generate tool contracts and the module index from normalized declarations.

The real VisionRTContracts validator is the single semantic entry for tool
declaration parsing and validation. This generator only keeps code-generation
rules: implementation bindings, generated identifier collisions, Python entry
symbols and the generated file set.
"""

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required for tool contract generation. Please install pyyaml.")


class ContractValidationError(Exception):
    pass


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ContractValidationError(f"YAML 存在重复键：{key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_yaml(file_path: Path) -> Dict[str, Any]:
    with Path(file_path).open("r", encoding="utf-8") as stream:
        return yaml.load(stream, Loader=UniqueKeyLoader) or {}


def to_cpp_identifier(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if cleaned and cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned


def cpp_string(value: str) -> str:
    escaped = (value.replace("\\", "\\\\").replace('"', '\\"')
               .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"))
    return f'"{escaped}"'


def system_output(port: Dict[str, Any]) -> bool:
    return port.get("id") in ("__status_code", "__status_message")


def business_outputs(tool: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [port for port in tool.get("outputs", []) if not system_output(port)]


def normalize_declaration(validator: Path, file_path: Path) -> Dict[str, Any]:
    """Run the real validator and consume its canonical YAML output."""
    result = subprocess.run(
        [str(validator), "--emit-normalized-yaml", str(file_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise ContractValidationError(f"[{Path(file_path).name}] 真实校验器拒绝：{detail}")
    try:
        document = yaml.safe_load(result.stdout)
    except yaml.YAMLError as error:
        raise ContractValidationError(f"[{Path(file_path).name}] 规范化输出不是有效 YAML：{error}") from error
    if not isinstance(document, dict):
        raise ContractValidationError(f"[{Path(file_path).name}] 规范化输出必须是映射")
    document["_source_path"] = str(file_path)
    return document


def validate_generation_rules(document: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
    prefix = f"[{Path(file_path).name}] "
    if document.get("schema_version") != 1:
        raise ContractValidationError(f"{prefix}规范化声明的 schema_version 无效")
    tool_id = document.get("tool_id")
    if not tool_id or not isinstance(tool_id, str):
        raise ContractValidationError(f"{prefix}缺少 tool_id")
    if not document.get("display_name"):
        raise ContractValidationError(f"{prefix}缺少 display_name")
    kind = document.get("kind")
    if kind not in ("cpp_function", "cpp_class", "python_function", "python_class"):
        raise ContractValidationError(f"{prefix}未知工具类型：{kind}")
    entry = document.get("entry")
    if not isinstance(entry, dict):
        raise ContractValidationError(f"{prefix}缺少 entry")

    if kind in ("python_function", "python_class"):
        script_name = entry.get("file")
        symbol = entry.get("symbol")
        if not script_name or not symbol:
            raise ContractValidationError(f"{prefix}Python 工具需要 entry.file 与 entry.symbol")
        script = (Path(file_path).parent / script_name).resolve()
        if not script.is_file():
            raise ContractValidationError(f"{prefix}Python 入口脚本不存在：{script}")
        try:
            tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
        except SyntaxError as error:
            raise ContractValidationError(f"{prefix}Python 入口脚本语法错误：{error}") from error
        found = False
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
                found = True
                break
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == symbol:
                        found = True
                        break
        if not found:
            raise ContractValidationError(f"{prefix}Python 入口符号 '{symbol}' 不存在于 '{script_name}'")
    elif not entry.get("registry_name"):
        raise ContractValidationError(f"{prefix}C++ 工具需要 entry.registry_name")

    if tool_id == "visionrt.flow.receive":
        legacy = {"source_flow_name", "source_node_name"}
        used = legacy & {port.get("id") for port in document.get("init_params", []) if isinstance(port, dict)}
        if used:
            raise ContractValidationError(
                f"{prefix}跨流程接收声明仍使用旧字段 {sorted(used)}；请改用 source_flow_id/source_node_seq 稳定身份")

    for group in ("init_params", "inputs", "outputs"):
        ports = document.get(group, [])
        if not isinstance(ports, list):
            raise ContractValidationError(f"{prefix}'{group}' 必须是列表")
        identifiers: Dict[str, str] = {}
        for port in ports:
            if not isinstance(port, dict) or not port.get("id"):
                raise ContractValidationError(f"{prefix}'{group}' 中的端口缺少 id")
            port_id = port["id"]
            identifier = to_cpp_identifier(port_id).upper()
            if identifier in identifiers:
                raise ContractValidationError(
                    f"{prefix}端口 '{port_id}' 规范化后与 '{identifiers[identifier]}' 冲突（{identifier}）")
            identifiers[identifier] = port_id
    return document


def load_and_validate_declarations(tools_dir: Path, validator: Path) -> List[Dict[str, Any]]:
    tools_dir = Path(tools_dir)
    yaml_files = sorted(set(tools_dir.glob("**/*.yaml")) | set(tools_dir.glob("**/*.yml")))
    if not yaml_files:
        raise ContractValidationError(f"未找到工具声明：{tools_dir}")
    seen_tool_ids: Set[str] = set()
    seen_reg_names: Set[str] = set()
    validated: List[Dict[str, Any]] = []
    for file_path in yaml_files:
        document = normalize_declaration(validator, file_path)
        validate_generation_rules(document, file_path)
        tool_id = document["tool_id"]
        if tool_id in seen_tool_ids:
            raise ContractValidationError(f"重复 tool_id '{tool_id}'：{file_path}")
        seen_tool_ids.add(tool_id)
        registry = document.get("entry", {}).get("registry_name")
        if registry:
            if registry in seen_reg_names:
                raise ContractValidationError(f"重复 registry_name '{registry}'：{file_path}")
            seen_reg_names.add(registry)
        validated.append(document)
    return validated


def port_to_typespec_cpp(port: Dict[str, Any]) -> str:
    port_type = port.get("type", "Int")
    type_cpp = f"visionrt::Type::{port_type}"
    dtype = port.get("dtype")
    if dtype:
        normalized = str(dtype).capitalize()
        if normalized.startswith("Uint"):
            normalized = "UInt" + normalized[4:]
        dtype_cpp = f"visionrt::DType::{normalized}"
    else:
        dtype_cpp = "std::nullopt"
    rank = port.get("rank", -1)
    if rank is None:
        rank = -1
    layout = str(port.get("layout", "dense")).lower()
    layout_cpp = "visionrt::Layout::BatchRagged" if layout in ("batch_ragged", "batchragged") else "visionrt::Layout::Dense"
    return f"visionrt::TypeSpec{{ {type_cpp}, {dtype_cpp}, {rank}, {layout_cpp} }}"


def port_value_cpp(value: Any, port_type: str) -> str:
    if port_type == "Int":
        return f"std::int64_t{{{int(value)}}}"
    if port_type == "Float":
        return repr(float(value))
    if port_type == "Bool":
        return "true" if value else "false"
    if port_type == "String":
        return f"visionrt::Utf8({cpp_string(str(value))})"
    raise ContractValidationError(f"无法为类型 {port_type} 生成执行约束值：{value!r}")


def port_contract_cpp(port: Dict[str, Any], output: bool) -> str:
    port_id = port["id"]
    identifier = to_cpp_identifier(port_id).upper()
    prefix = "out" if output else ("init" if port.get("_group") == "init_params" else "in")
    contract_id = f"{prefix}::{identifier}"
    spec = f"{prefix}::spec::{identifier}"
    port_type = port.get("type", "Int")
    required = "true" if output else ("false" if "default" in port else "true")
    options = port.get("options", [])
    if options:
        option_values = ", ".join(port_value_cpp(option, port_type) for option in options)
        options_cpp = f"std::vector<visionrt::PortValue>{{{option_values}}}"
    else:
        options_cpp = "{}"
    minimum = port.get("min")
    maximum = port.get("max")
    minimum_cpp = ("std::nullopt" if minimum is None
                   else f"std::optional<visionrt::PortValue>{{{port_value_cpp(minimum, port_type)}}}")
    maximum_cpp = ("std::nullopt" if maximum is None
                   else f"std::optional<visionrt::PortValue>{{{port_value_cpp(maximum, port_type)}}}")
    return (f"visionrt::PortContract{{{contract_id}, {spec}, {required}, {options_cpp}, "
            f"{minimum_cpp}, {maximum_cpp}}}")


def generate_port_constants_header(tools: List[Dict[str, Any]]) -> str:
    sorted_tools = sorted(tools, key=lambda tool: tool["tool_id"])
    lines = [
        "// Generated by tools/generate_tool_contracts.py. DO NOT EDIT DIRECTLY.",
        "#pragma once",
        "",
        "#include <visionrt/contracts/Contract.h>",
        "#include <visionrt/contracts/Value.h>",
        "#include <optional>",
        "#include <string>",
        "#include <vector>",
        "",
        "namespace visiontools::ports {",
        "",
    ]
    seen_safe_ids: Set[str] = set()
    for tool in sorted_tools:
        if any("type_from" in port for port in business_outputs(tool)):
            continue
        tool_id = tool["tool_id"]
        safe_id = to_cpp_identifier(tool_id)
        if safe_id in seen_safe_ids:
            raise ContractValidationError(f"tool_id '{tool_id}' 生成标识符冲突：'{safe_id}'")
        seen_safe_ids.add(safe_id)

        lines.append(f"namespace {safe_id} {{")
        lines.append(f"inline constexpr const char* TOOL_ID = {cpp_string(tool_id)};")
        registry_name = tool.get("entry", {}).get("registry_name", "")
        if registry_name:
            lines.append(f"inline constexpr const char* REGISTRY_NAME = {cpp_string(registry_name)};")

        for group, key in [("init", "init_params"), ("in", "inputs"), ("out", "outputs")]:
            ports = sorted(tool.get(key, []), key=lambda port: port["id"])
            if group == "out":
                ports = [port for port in ports if not system_output(port)]
            seen_port_identifiers: Dict[str, str] = {}
            for port in ports:
                identifier = to_cpp_identifier(port["id"]).upper()
                if identifier in seen_port_identifiers:
                    raise ContractValidationError(
                        f"端口 '{port['id']}' 规范化后与 '{seen_port_identifiers[identifier]}' 冲突（{identifier}）")
                seen_port_identifiers[identifier] = port["id"]

            lines.append(f"namespace {group} {{")
            for port in ports:
                identifier = to_cpp_identifier(port["id"]).upper()
                lines.append(f"inline constexpr const char* {identifier} = {cpp_string(port['id'])};")
            lines.append("namespace spec {")
            for port in ports:
                identifier = to_cpp_identifier(port["id"]).upper()
                lines.append(f"inline const visionrt::TypeSpec {identifier} = {port_to_typespec_cpp(port)};")
            lines.append("}")
            lines.append("}")

        lines.append("")
        lines.append("inline visionrt::ToolContract getToolContract() {")
        lines.append("    visionrt::ToolContract c;")
        lines.append("    c.toolId = TOOL_ID;")
        if registry_name:
            lines.append("    c.registryName = REGISTRY_NAME;")
        for group, key in [("init", "init_params"), ("in", "inputs"), ("out", "outputs")]:
            ports = sorted(tool.get(key, []), key=lambda port: port["id"])
            if group == "out":
                ports = [port for port in ports if not system_output(port)]
            member = {"init_params": "initParams", "inputs": "inputs", "outputs": "outputs"}[key]
            lines.append(f"    c.{member} = {{")
            for port in ports:
                copied = dict(port)
                copied["_group"] = key
                lines.append(f"        {port_contract_cpp(copied, group == 'out')},")
            lines.append("    };")
        lines.append("    return c;")
        lines.append("}")
        lines.append(f"}} // namespace {safe_id}")
        lines.append("")
    lines.append("} // namespace visiontools::ports")
    return "\n".join(lines) + "\n"


def generate_module_factory_inc(module_name: str, tools_dict: Dict[str, Dict[str, Any]],
                                bindings: Dict[str, Any]) -> str:
    lines = [
        "// Generated by tools/generate_tool_contracts.py. DO NOT EDIT DIRECTLY.",
        "#pragma once",
        "",
        "#include <visionrt/contracts/Module.h>",
        "#include <visionrt/contracts/Tool.h>",
        "#include <visionrt/contracts/Contract.h>",
        '#include "PortConstants.h"',
        "#include <algorithm>",
        "#include <cstring>",
        "#include <memory>",
        "#include <string>",
        "#include <stdexcept>",
        "#include <utility>",
        "",
    ]
    tool_bindings = bindings.get("tools", {})
    included_headers = set()
    for registry, binding in sorted(tool_bindings.items()):
        header = binding.get("header")
        if header and header not in included_headers:
            lines.append(f'#include "{header}"')
            included_headers.add(header)

    namespace = bindings.get("namespace", f"visiontools::{module_name}")
    lines.extend([
        "",
        f"namespace {namespace} {{",
        "",
        "class ValidatedToolInstance : public visionrt::IToolInstance {",
        "public:",
        "    ValidatedToolInstance(std::unique_ptr<visionrt::IToolInstance> inner, visionrt::ToolContract contract)",
        "        : inner_(std::move(inner)), contract_(std::move(contract)) {}",
        "",
        "    visionrt::ToolStatus initialize(const visionrt::Values& initParams) override {",
        "        try {",
        "            for (const auto& [k, v] : initParams) {",
        "                const bool known = std::any_of(contract_.initParams.begin(), contract_.initParams.end(),",
        "                    [&](const auto& p) { return p.id == k; });",
        "                if (!known) {",
        "                    throw visionrt::ContractError(\"初始化参数不在声明中：\" + k);",
        "                }",
        "            }",
        "            visionrt::validateToolValues(contract_.initParams, initParams, false);",
        "        } catch (const visionrt::ContractError& e) {",
        "            return visionrt::ToolStatus{-2, e.what()};",
        "        }",
        "        return inner_->initialize(initParams);",
        "    }",
        "",
        "    visionrt::ToolResult execute(const visionrt::Values& inputs) override {",
        "        try {",
        "            for (const auto& [k, v] : inputs) {",
        "                const bool known = std::any_of(contract_.inputs.begin(), contract_.inputs.end(),",
        "                    [&](const auto& p) { return p.id == k; });",
        "                if (!known) {",
        "                    throw visionrt::ContractError(\"输入端口不在声明中：\" + k);",
        "                }",
        "            }",
        "            visionrt::validateToolValues(contract_.inputs, inputs, false);",
        "        } catch (const visionrt::ContractError& e) {",
        "            visionrt::Values missingOutputs;",
        "            for (const auto& out : contract_.outputs) {",
        "                missingOutputs.emplace(out.id, visionrt::Missing{});",
        "            }",
        "            return visionrt::ToolResult{visionrt::ToolStatus{-2, e.what()}, std::move(missingOutputs)};",
        "        }",
        "        visionrt::ToolResult result = inner_->execute(inputs);",
        "        if (result.status.code != 0) {",
        "            visionrt::Values missingOutputs;",
        "            for (const auto& out : contract_.outputs) {",
        "                missingOutputs.emplace(out.id, visionrt::Missing{});",
        "            }",
        "            result.outputs = std::move(missingOutputs);",
        "            return result;",
        "        }",
        "        try {",
        "            for (const auto& [k, v] : result.outputs) {",
        "                const bool known = std::any_of(contract_.outputs.begin(), contract_.outputs.end(),",
        "                    [&](const auto& p) { return p.id == k; });",
        "                if (!known) {",
        "                    throw visionrt::ContractError(\"输出端口不在声明中：\" + k);",
        "                }",
        "            }",
        "            visionrt::validateToolValues(contract_.outputs, result.outputs, false);",
        "        } catch (const visionrt::ContractError& e) {",
        "            visionrt::Values missingOutputs;",
        "            for (const auto& out : contract_.outputs) {",
        "                missingOutputs.emplace(out.id, visionrt::Missing{});",
        "            }",
        "            return visionrt::ToolResult{visionrt::ToolStatus{-2, std::string(\"输出合同错误：\") + e.what()}, std::move(missingOutputs)};",
        "        }",
        "        return result;",
        "    }",
        "",
        "    visionrt::ToolStatus release() override {",
        "        return inner_->release();",
        "    }",
        "",
        "private:",
        "    std::unique_ptr<visionrt::IToolInstance> inner_;",
        "    visionrt::ToolContract contract_;",
        "};",
        "",
        "inline visionrt::IToolInstance* VT_CALL createToolInstance(",
        "    const char* registry_name,",
        "    const visionrt::ToolServices* services,",
        "    visionrt::ToolStatus* status) noexcept {",
        "    if (!registry_name) {",
        "        if (status) { status->code = -3; status->message = \"registry_name 为空\"; }",
        "        return nullptr;",
        "    }",
        "    try {",
        "        const visionrt::ToolServices s = services ? *services : visionrt::ToolServices{};",
        "",
    ])

    for registry, binding in sorted(tool_bindings.items()):
        class_name = binding["class"]
        tool = tools_dict.get(registry)
        if not tool or not tool.get("tool_id"):
            raise ContractValidationError(f"实现绑定引用了未声明的 registry_name：{registry}")
        safe_id = to_cpp_identifier(tool["tool_id"])
        lines.extend([
            f'        if (std::strcmp(registry_name, {cpp_string(registry)}) == 0) {{',
            "            if (status) { status->code = 0; status->message.clear(); }",
            f"            return new ValidatedToolInstance(std::make_unique<{class_name}>(s), "
            f"visiontools::ports::{safe_id}::getToolContract());",
            "        }",
        ])

    lines.extend([
        "",
        "        if (status) {",
        "            status->code = -3;",
        '            status->message = std::string("未知实现注册名：") + registry_name;',
        "        }",
        "        return nullptr;",
        "    } catch (const std::exception& e) {",
        "        if (status) { status->code = -3; status->message = std::string(\"创建工具实例失败：\") + e.what(); }",
        "        return nullptr;",
        "    } catch (...) {",
        "        if (status) { status->code = -3; status->message = \"创建工具实例失败：未知异常\"; }",
        "        return nullptr;",
        "    }",
        "}",
        "",
        "inline void VT_CALL destroyToolInstance(visionrt::IToolInstance* instance) noexcept {",
        "    delete instance;",
        "}",
        "",
        "inline const visionrt::ModuleApi s_generatedModuleApi = {",
        "    visionrt::MODULE_ABI_MAJOR,",
        "    sizeof(visionrt::ModuleApi),",
        "    &createToolInstance,",
        "    &destroyToolInstance",
        "};",
        "",
        f"}} // namespace {namespace}",
        "",
    ])
    return "\n".join(lines) + "\n"


def build_module_index(tools: List[Dict[str, Any]], bindings_data: Optional[Dict[str, Any]],
                       variant: str = "release", project_root: Optional[Path] = None,
                       selected_modules: Optional[List[str]] = None) -> Dict[str, Any]:
    bindings_data = bindings_data or {}
    module_bindings = bindings_data.get("modules", {})
    selected = set(module_bindings if selected_modules is None else selected_modules)
    if selected - set(module_bindings):
        raise ContractValidationError(f"选择了未声明的模块：{selected - set(module_bindings)}")
    by_registry: Dict[str, str] = {}
    for module_name, module_info in module_bindings.items():
        for registry_name in module_info.get("tools", {}):
            if registry_name in by_registry:
                raise ContractValidationError(f"重复实现绑定：{registry_name}")
            by_registry[registry_name] = module_name
    host_bindings = bindings_data.get("host_bindings", {})
    tools_map: Dict[str, Any] = {}
    referenced: Set[str] = set()
    for tool in tools:
        tool_id = tool["tool_id"]
        registry_name = tool.get("entry", {}).get("registry_name")
        binding = host_bindings.get(tool_id)
        if not binding and any("type_from" in port for port in business_outputs(tool)):
            raise ContractValidationError(f"源派生输出类型需要 host_binding：{tool_id}")
        if binding:
            if not tool["kind"].startswith("cpp") or registry_name in by_registry:
                raise ContractValidationError(f"host_binding 组合无效：{tool_id}")
            entry: Dict[str, Any] = {"host_binding": binding}
        elif tool["kind"].startswith("python"):
            entry = {}
        else:
            module_name = by_registry.get(registry_name)
            if module_name is None:
                raise ContractValidationError(f"缺少实现绑定：{tool_id}")
            if module_name not in selected:
                continue
            entry = {"module": module_name}
            referenced.add(module_name)
        source = tool.get("_source_path") or tool.get("definition")
        if not source:
            raise ContractValidationError(f"缺少声明路径：{tool_id}")
        definition = Path(source).resolve()
        if project_root is not None:
            definition = definition.relative_to(Path(project_root).resolve())
        entry["definition"] = definition.as_posix()
        tools_map[tool_id] = entry
    if set(host_bindings) - {tool["tool_id"] for tool in tools}:
        raise ContractValidationError("host_binding 引用了未声明的 tool_id")
    return {
        "schema_version": 2,
        "variant": variant,
        "modules": {name: {} for name in sorted(referenced)},
        "tools": tools_map,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Tool Contract and Registry Generator")
    parser.add_argument("--tools-dir", default="assets/tools", help="Directory with tool YAMLs")
    parser.add_argument("--bindings", default="tools/module_bindings.yaml", help="Path to module_bindings.yaml")
    parser.add_argument("--output-dir", default="build/generated", help="Directory for generated C++ files")
    parser.add_argument("--index-output", default="module_index.yaml", help="Output path for module index YAML")
    parser.add_argument("--variant", default="release", help="Distribution configuration name")
    parser.add_argument("--modules", nargs="*", help="Module targets included in this build")
    parser.add_argument("--validator", default=None, help="Path to visionrt_validate_tool executable")
    args = parser.parse_args()

    if not args.validator:
        print("[ERROR] 正式构建必须提供 --validator 真实校验器", file=sys.stderr)
        return 1
    validator = Path(args.validator).resolve()
    if not validator.is_file():
        print(f"[ERROR] 真实校验器不存在：{validator}", file=sys.stderr)
        return 1

    tools_dir = Path(args.tools_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    yaml_files = sorted(set(tools_dir.glob("**/*.yaml")) | set(tools_dir.glob("**/*.yml")))
    if not yaml_files:
        print(f"[ERROR] 未找到工具声明：{tools_dir}", file=sys.stderr)
        return 1

    bindings_path = Path(args.bindings)
    if not bindings_path.exists():
        print(f"[ERROR] 模块绑定文件不存在：{bindings_path}", file=sys.stderr)
        return 1

    try:
        bindings_data = load_yaml(bindings_path)
    except Exception as error:
        print(f"[ERROR] 无法读取模块绑定 {bindings_path}：{error}", file=sys.stderr)
        return 1

    validated_tools: List[Dict[str, Any]] = []
    seen_tool_ids: Set[str] = set()
    seen_reg_names: Set[str] = set()
    reg_to_tool: Dict[str, Dict[str, Any]] = {}
    for file_path in yaml_files:
        try:
            document = normalize_declaration(validator, file_path)
            validate_generation_rules(document, file_path)
            tool_id = document["tool_id"]
            if tool_id in seen_tool_ids:
                raise ContractValidationError(f"重复 tool_id '{tool_id}'：{file_path}")
            seen_tool_ids.add(tool_id)
            registry = document.get("entry", {}).get("registry_name")
            if registry:
                if registry in seen_reg_names:
                    raise ContractValidationError(f"重复 registry_name '{registry}'：{file_path}")
                seen_reg_names.add(registry)
                reg_to_tool[registry] = document
            validated_tools.append(document)
        except Exception as error:
            print(f"[ERROR] 工具声明校验失败：{error}", file=sys.stderr)
            return 1

    if not validated_tools:
        print(f"[ERROR] 未找到有效工具声明：{tools_dir}", file=sys.stderr)
        return 1

    bound_regs: Dict[str, str] = {}
    for module_name, module_info in bindings_data.get("modules", {}).items():
        for registry in module_info.get("tools", {}):
            if registry in bound_regs:
                print(f"[ERROR] registry_name '{registry}' 同时绑定到模块 '{bound_regs[registry]}' 与 "
                      f"'{module_name}'", file=sys.stderr)
                return 1
            bound_regs[registry] = module_name
    for registry, module_name in bound_regs.items():
        if registry not in reg_to_tool:
            print(f"[ERROR] 模块 '{module_name}' 绑定了未声明的 registry_name '{registry}'", file=sys.stderr)
            return 1
    host_registries = {tool.get("entry", {}).get("registry_name") for tool in validated_tools
                       if tool["tool_id"] in bindings_data.get("host_bindings", {})}
    host_registries.discard(None)
    for registry in seen_reg_names:
        if registry not in bound_regs and registry not in host_registries:
            print(f"[ERROR] C++ 工具 registry_name '{registry}' 缺少实现绑定：{bindings_path}", file=sys.stderr)
            return 1

    try:
        for module_name, module_info in bindings_data.get("modules", {}).items():
            factory_content = generate_module_factory_inc(module_name, reg_to_tool, module_info)
            (out_dir / f"{module_name}_factory.inc").write_text(factory_content, encoding="utf-8")
        constants_path = out_dir / "PortConstants.h"
        constants_path.write_text(generate_port_constants_header(validated_tools), encoding="utf-8")
        index_path = Path(args.index_output)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_data = build_module_index(validated_tools, bindings_data, variant=args.variant,
                                        project_root=tools_dir.parent.parent, selected_modules=args.modules)
        index_path.write_text(yaml.safe_dump(index_data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    except Exception as error:
        print(f"[ERROR] 生成失败：{error}", file=sys.stderr)
        return 1

    print(f"[SUCCESS] 已规范化 {len(validated_tools)} 个工具声明，生成 {constants_path} 与 {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
