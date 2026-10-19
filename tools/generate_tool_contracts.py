#!/usr/bin/env python3
"""
Tool Contract & Module Registry Generator for VisionTools.
Conforms to shared SDK contract C03 and specs TS-02 / VS-02.

Validates tool YAML declarations against schema v1, ensures single source of truth,
and generates:
  1. C++ Port Constants header (<output-dir>/PortConstants.h) with TypeSpec definitions
  2. C++ Module Factory dispatch header (<output-dir>/<module>_factory.inc)
  3. Module Index JSON (<index-output>) for host runtime discovery.
"""

import argparse
import ast
import json
import math
import re
import shutil
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
            raise ValueError(f"Duplicate key detected in YAML: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping

UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)

def load_yaml(file_path: Path) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=UniqueKeyLoader) or {}

def to_cpp_identifier(name: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if cleaned and cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned

def load_sdk_enums(sdk_include_dir: Optional[Path] = None) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    """
    Derive canonical enums from VisionRT SDK headers (Value.h / Schema.h) to eliminate parallel definitions (SSOT).
    Strictly fails if an explicit sdk_include_dir is invalid or missing headers.
    """
    if sdk_include_dir is not None:
        sdir = Path(sdk_include_dir).resolve()
        val_h = sdir / "visionrt" / "sdk" / "Value.h"
        if not sdir.exists() or not val_h.exists():
            raise ContractValidationError(f"Explicit SDK include directory does not exist or missing Value.h: {sdir}")
        search_dirs = [sdir]
    else:
        search_dirs = [
            Path(__file__).resolve().parent.parent / "3rdparty" / "VisionRTSDK" / "include",
        ]

    for sdir in search_dirs:
        val_h = sdir / "visionrt" / "sdk" / "Value.h"
        schema_h = sdir / "visionrt" / "sdk" / "Schema.h"
        if val_h.exists() and schema_h.exists():
            try:
                content = val_h.read_text(encoding="utf-8")
                m_type = re.search(r'enum\s+class\s+Type\s*\{([^}]+)\}', content)
                types = {x.strip() for x in m_type.group(1).split(',') if x.strip()} if m_type else set()
                m_dtype = re.search(r'enum\s+class\s+DType\s*\{([^}]+)\}', content)
                dtypes = {x.strip() for x in m_dtype.group(1).split(',') if x.strip()} if m_dtype else set()
                dtypes |= {x.lower() for x in dtypes}
                m_layout = re.search(r'enum\s+class\s+Layout\s*\{([^}]+)\}', content)
                layouts = {x.strip() for x in m_layout.group(1).split(',') if x.strip()} if m_layout else set()
                layouts |= {re.sub(r'(?<!^)(?=[A-Z])', '_', x).lower() for x in layouts}

                kinds = {"cpp_function", "cpp_class", "python_function", "python_class"}
                scontent = schema_h.read_text(encoding="utf-8")
                m_kind = re.search(r'enum\s+class\s+ToolKind\s*\{([^}]+)\}', scontent)
                if m_kind:
                    raw_kinds = [x.strip() for x in m_kind.group(1).split(',') if x.strip()]
                    kinds = {re.sub(r'(?<!^)(?=[A-Z])', '_', x).lower() for x in raw_kinds}

                if types and dtypes and layouts:
                    return types, dtypes, layouts, kinds
            except Exception as e:
                raise ContractValidationError(f"Failed to read SDK enums from {sdir}: {e}")

    raise ContractValidationError(
        f"Failed to extract SDK enums from: {[str(p) for p in search_dirs]}. "
        "Please ensure 3rdparty/VisionRTSDK is checked out."
    )

VALID_TYPES, VALID_DTYPES, VALID_LAYOUTS, VALID_KINDS = load_sdk_enums()

def validate_tool_declaration(doc: Dict[str, Any], file_path: Path, tools_dir: Optional[Path] = None) -> Dict[str, Any]:
    prefix = f"[{file_path.name}] "

    # 1. Schema version
    version = doc.get("schema_version")
    if version != 1:
        raise ContractValidationError(f"{prefix}Invalid or missing schema_version (expected 1, got {version})")

    # 2. Tool identity
    tool_id = doc.get("tool_id")
    if not tool_id or not isinstance(tool_id, str):
        raise ContractValidationError(f"{prefix}Missing or invalid 'tool_id'")

    display_name = doc.get("display_name")
    if not display_name or not isinstance(display_name, str):
        raise ContractValidationError(f"{prefix}Missing or invalid 'display_name'")

    kind = doc.get("kind")
    if kind not in VALID_KINDS:
        raise ContractValidationError(f"{prefix}Invalid tool kind: '{kind}'")
    if kind in ("cpp_function", "python_function") and doc.get("init_params"):
        raise ContractValidationError(f"{prefix}Function tool cannot have init_params")

    init_ports = {p.get("id") for p in doc.get("init_params", []) if isinstance(p, dict)}
    input_ports_set = {p.get("id") for p in doc.get("inputs", []) if isinstance(p, dict)}
    overlap = init_ports & input_ports_set
    if overlap:
        raise ContractValidationError(f"{prefix}init_params and inputs share ID: {overlap}")

    # 3. Entry point
    entry = doc.get("entry")
    if not entry or not isinstance(entry, dict):
        raise ContractValidationError(f"{prefix}Missing or invalid 'entry' dictionary")

    if kind.startswith("cpp"):
        reg_name = entry.get("registry_name")
        if not reg_name or not isinstance(reg_name, str):
            raise ContractValidationError(f"{prefix}C++ tool requires 'entry.registry_name'")
    elif kind.startswith("python"):
        script_file = entry.get("file")
        symbol = entry.get("symbol")
        if not script_file or not symbol:
            raise ContractValidationError(f"{prefix}Python tool requires 'entry.file' and 'entry.symbol'")

        # Resolve script path statically (without executing)
        script_path = None
        candidates = [
            file_path.parent / script_file,
            (tools_dir / script_file) if tools_dir else None
        ]
        for c in candidates:
            if c and c.is_file():
                script_path = c
                break
        if not script_path:
            raise ContractValidationError(f"{prefix}Python entry script '{script_file}' does not exist")

        # Statically verify symbol declared in script AST
        try:
            tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
            found_symbol = False
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name == symbol:
                        found_symbol = True
                        break
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == symbol:
                            found_symbol = True
                            break
            if not found_symbol:
                raise ContractValidationError(f"{prefix}Python entry symbol '{symbol}' not found in '{script_file}'")
        except SyntaxError as e:
            raise ContractValidationError(f"{prefix}Python entry script '{script_file}' has syntax error: {e}")

    # 4. Port lists validation & identifier collision detection
    for port_group in ["init_params", "inputs", "outputs"]:
        ports = doc.get(port_group, [])
        if not isinstance(ports, list):
            raise ContractValidationError(f"{prefix}'{port_group}' must be a list")

        port_ids = set()
        port_c_idents = set()
        for p in ports:
            if not isinstance(p, dict):
                raise ContractValidationError(f"{prefix}Port in '{port_group}' must be an object")
            p_id = p.get("id")
            if not p_id or not isinstance(p_id, str):
                raise ContractValidationError(f"{prefix}Port missing 'id' in '{port_group}'")
            if "name" not in p or not isinstance(p["name"], str) or not p["name"].strip():
                raise ContractValidationError(f"{prefix}Port missing or empty 'name' on port: '{p_id}'")
            if p_id in port_ids:
                raise ContractValidationError(f"{prefix}Duplicate port id '{p_id}' in '{port_group}'")
            port_ids.add(p_id)

            if p_id.startswith("__"):
                if not (port_group == "outputs" and p_id in ("__status_code", "__status_message")):
                    raise ContractValidationError(f"{prefix}Port ID uses reserved '__' prefix: '{p_id}' in '{port_group}'")

            c_ident = to_cpp_identifier(p_id).upper()
            if c_ident in port_c_idents:
                raise ContractValidationError(f"{prefix}C++ identifier collision in '{port_group}' for port '{p_id}' ({c_ident})")
            port_c_idents.add(c_ident)

            p_type = p.get("type")
            if p_type not in VALID_TYPES:
                raise ContractValidationError(f"{prefix}Port '{p_id}' has invalid type '{p_type}'")


    return doc


def load_and_validate_declarations(tools_dir: Path, bindings_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Discover and validate all tool declarations in tools_dir.
    Exported for use by package_tools and tests.
    """
    tools_dir = Path(tools_dir)
    yaml_files = sorted(set(tools_dir.glob("**/*.yaml")) | set(tools_dir.glob("**/*.yml")))
    if not yaml_files:
        raise ContractValidationError(f"No tool YAML declarations found in {tools_dir}")

    seen_tool_ids = set()
    seen_reg_names = set()
    validated_tools = []

    for yf in yaml_files:
        doc = load_yaml(yf)
        validated = validate_tool_declaration(doc, yf, tools_dir=tools_dir)
        validated["_source_path"] = str(yf)
        try:
            rel_parts = yf.relative_to(tools_dir).parts
        except ValueError:
            rel_parts = yf.parts
        if any(part in rel_parts for part in ("custom", "示例", "examples")):
            validated["is_example"] = True
        tid = validated["tool_id"]
        if tid in seen_tool_ids:
            raise ContractValidationError(f"Duplicate tool_id '{tid}' in {yf}")
        seen_tool_ids.add(tid)

        reg = validated.get("entry", {}).get("registry_name")
        if reg:
            if reg in seen_reg_names:
                raise ContractValidationError(f"Duplicate registry_name '{reg}' in {yf}")
            seen_reg_names.add(reg)

        validated_tools.append(validated)

    return validated_tools

def port_to_typespec_cpp(p: Dict[str, Any]) -> str:
    ptype = p.get("type", "Int")
    type_cpp = f"visionrt::Type::{ptype}"

    dtype = p.get("dtype")
    if dtype:
        d = dtype.capitalize()
        if d.startswith("Uint"):
            d = "UInt" + d[4:]
        dtype_cpp = f"visionrt::DType::{d}"
    else:
        dtype_cpp = "std::nullopt"

    rank = p.get("rank", -1)
    if rank is None:
        rank = -1
    rank_cpp = str(rank)

    layout = p.get("layout", "dense")
    if layout.lower() in ("batch_ragged", "batchragged"):
        layout_cpp = "visionrt::Layout::BatchRagged"
    else:
        layout_cpp = "visionrt::Layout::Dense"

    return f"visionrt::TypeSpec{{ {type_cpp}, {dtype_cpp}, {rank_cpp}, {layout_cpp} }}"

def generate_port_constants_header(tools: List[Dict[str, Any]]) -> str:
    # Sort deterministically by tool_id to ensure semantic invariance under file ordering
    sorted_tools = sorted(tools, key=lambda t: t["tool_id"])

    lines = [
        "// Generated by tools/generate_tool_contracts.py. DO NOT EDIT DIRECTLY.",
        "#pragma once",
        "",
        "#include <visionrt/sdk/Contract.h>",
        "#include <visionrt/sdk/Value.h>",
        "#include <optional>",
        "#include <string>",
        "#include <vector>",
        "",
        "namespace visiontools::ports {",
        ""
    ]

    seen_safe_ids = set()
    for tool in sorted_tools:
        tool_id = tool["tool_id"]
        safe_id = to_cpp_identifier(tool_id)
        if safe_id in seen_safe_ids:
            raise ContractValidationError(f"C++ identifier collision for tool_id '{tool_id}' -> '{safe_id}'")
        seen_safe_ids.add(safe_id)

        lines.append(f"namespace {safe_id} {{")
        lines.append(f"inline constexpr const char* TOOL_ID = \"{tool_id}\";")
        
        reg_name = tool.get("entry", {}).get("registry_name", "")
        if reg_name:
            lines.append(f"inline constexpr const char* REGISTRY_NAME = \"{reg_name}\";")

        for grp, key in [("init", "init_params"), ("in", "inputs"), ("out", "outputs")]:
            ports = sorted(tool.get(key, []), key=lambda x: x["id"])
            seen_port_idents = {}
            for p in ports:
                pid = p["id"]
                c_name = to_cpp_identifier(pid).upper()
                if c_name in seen_port_idents:
                    raise ContractValidationError(f"Port ID '{pid}' normalizes to colliding C++ identifier '{c_name}' with '{seen_port_idents[c_name]}'")
                seen_port_idents[c_name] = pid

            lines.append(f"namespace {grp} {{")
            for p in ports:
                pid = p["id"]
                c_name = to_cpp_identifier(pid).upper()
                lines.append(f"inline constexpr const char* {c_name} = \"{pid}\";")
            lines.append("namespace spec {")
            for p in ports:
                pid = p["id"]
                c_name = to_cpp_identifier(pid).upper()
                spec_cpp = port_to_typespec_cpp(p)
                lines.append(f"inline const visionrt::TypeSpec {c_name} = {spec_cpp};")
            lines.append("}")
            lines.append("}")

        lines.append("")
        lines.append("inline visionrt::ToolContract getToolContract() {")
        lines.append("    visionrt::ToolContract c;")
        lines.append("    c.toolId = TOOL_ID;")
        if reg_name:
            lines.append("    c.registryName = REGISTRY_NAME;")
        lines.append("    c.initParams = {")
        for p in sorted(tool.get("init_params", []), key=lambda x: x["id"]):
            pid = p["id"]
            c_name = to_cpp_identifier(pid).upper()
            req = "false" if "default" in p and p["default"] is not None else "true"
            lines.append(f"        visionrt::PortContract{{init::{c_name}, init::spec::{c_name}, {req}}},")
        lines.append("    };")
        lines.append("    c.inputs = {")
        for p in sorted(tool.get("inputs", []), key=lambda x: x["id"]):
            pid = p["id"]
            c_name = to_cpp_identifier(pid).upper()
            req = "false" if "default" in p and p["default"] is not None else "true"
            lines.append(f"        visionrt::PortContract{{in::{c_name}, in::spec::{c_name}, {req}}},")
        lines.append("    };")
        lines.append("    c.outputs = {")
        for p in sorted(tool.get("outputs", []), key=lambda x: x["id"]):
            pid = p["id"]
            c_name = to_cpp_identifier(pid).upper()
            lines.append(f"        visionrt::PortContract{{out::{c_name}, out::spec::{c_name}, true}},")
        lines.append("    };")
        lines.append("    return c;")
        lines.append("}")

        lines.append(f"}} // namespace {safe_id}")
        lines.append("")

    lines.append("} // namespace visiontools::ports")
    return "\n".join(lines) + "\n"

def generate_module_factory_inc(module_name: str, tools_dict: Dict[str, Dict[str, Any]], bindings: Dict[str, Any]) -> str:
    """
    Generate C++ factory dispatch code for a module.
    tools_dict: mapping of registry_name to validated tool dict
    bindings: module entry from module_bindings.yaml
    """
    lines = [
        "// Generated by tools/generate_tool_contracts.py. DO NOT EDIT DIRECTLY.",
        "#pragma once",
        "",
        "#include <visionrt/sdk/Module.h>",
        "#include <visionrt/sdk/Tool.h>",
        "#include <visionrt/sdk/Contract.h>",
        "#include \"PortConstants.h\"",
        "#include <cstring>",
        "#include <memory>",
        "#include <string>",
        "#include <stdexcept>",
        "#include <utility>",
        ""
    ]

    tool_bindings = bindings.get("tools", {})
    # Include headers
    included_headers = set()
    for reg, b in sorted(tool_bindings.items()):
        hdr = b.get("header")
        if hdr and hdr not in included_headers:
            lines.append(f'#include "{hdr}"')
            included_headers.add(hdr)

    ns = bindings.get("namespace", f"visiontools::{module_name}")

    lines.extend([
        "",
        f"namespace {ns} {{",
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
        "                    throw visionrt::ContractError(\"Extra unexpected init parameter: \" + k);",
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
        "                    throw visionrt::ContractError(\"Extra unexpected input port: \" + k);",
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
        "                    throw visionrt::ContractError(\"Extra unexpected output port: \" + k);",
        "                }",
        "            }",
        "            visionrt::validateToolValues(contract_.outputs, result.outputs, false);",
        "        } catch (const visionrt::ContractError& e) {",
        "            visionrt::Values missingOutputs;",
        "            for (const auto& out : contract_.outputs) {",
        "                missingOutputs.emplace(out.id, visionrt::Missing{});",
        "            }",
        "            return visionrt::ToolResult{visionrt::ToolStatus{-2, std::string(\"Output contract error: \") + e.what()}, std::move(missingOutputs)};",
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
        "        if (status) { status->code = -3; status->message = \"registry_name is null\"; }",
        "        return nullptr;",
        "    }",
        "    try {",
        "        const visionrt::ToolServices s = services ? *services : visionrt::ToolServices{};",
        ""
    ])

    for reg, b in sorted(tool_bindings.items()):
        cls = b["class"]
        tool_decl = tools_dict.get(reg, {})
        tid = tool_decl.get("tool_id", "")
        safe_tid = to_cpp_identifier(tid) if tid else ""
        if safe_tid:
            lines.extend([
                f'        if (std::strcmp(registry_name, "{reg}") == 0) {{',
                "            if (status) { status->code = 0; status->message.clear(); }",
                f"            return new ValidatedToolInstance(std::make_unique<{cls}>(s), visiontools::ports::{safe_tid}::getToolContract());",
                "        }"
            ])
        else:
            lines.extend([
                f'        if (std::strcmp(registry_name, "{reg}") == 0) {{',
                "            if (status) { status->code = 0; status->message.clear(); }",
                f"            return new {cls}(s);",
                "        }"
            ])

    lines.extend([
        "",
        "        if (status) {",
        "            status->code = -3;",
        '            status->message = std::string("Unknown registry_name: ") + registry_name;',
        "        }",
        "        return nullptr;",
        "    } catch (const std::exception& e) {",
        "        if (status) { status->code = -3; status->message = e.what(); }",
        "        return nullptr;",
        "    } catch (...) {",
        "        if (status) { status->code = -3; status->message = \"Unknown exception in createToolInstance\"; }",
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
        f"}} // namespace {ns}",
        ""
    ])

    return "\n".join(lines) + "\n"

def build_module_index(tools: List[Dict[str, Any]], bindings_data: Optional[Dict[str, Any]], variant: str = "gpu", project_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Generate build/generated/module_index.yaml metadata.
    Lightweight structure: variant, modules, tools (YS-01 / YS-02).
    """
    modules = {}
    tools_map = {}

    if bindings_data and "modules" in bindings_data:
        for mod_name, mod_info in bindings_data["modules"].items():
            mod_variant = mod_info.get("variant", "cpu")
            if variant == "cpu" and mod_variant == "gpu":
                continue
            modules[mod_name] = {}
            for reg in mod_info.get("tools", {}):
                matching_tool = next((t for t in tools if t.get("entry", {}).get("registry_name") == reg or t.get("tool_id") == reg), None)
                if matching_tool and not matching_tool.get("is_example"):
                    src_path = matching_tool.get("_source_path") or matching_tool.get("definition")
                    if src_path:
                        p = Path(src_path)
                        if project_root:
                            try:
                                rel_def = p.relative_to(project_root)
                            except ValueError:
                                rel_def = p
                        else:
                            parts = p.parts
                            if "assets" in parts:
                                idx = parts.index("assets")
                                rel_def = Path(*parts[idx:])
                            else:
                                rel_def = p
                        def_str = str(rel_def).replace("\\", "/")
                    else:
                        def_str = f"assets/tools/{reg}.yaml"

                    tool_id = matching_tool.get("tool_id") or reg
                    tools_map[tool_id] = {
                        "module": mod_name,
                        "definition": def_str
                    }

    return {
        "schema_version": 2,
        "variant": variant,
        "modules": modules,
        "tools": tools_map
    }

def main():
    parser = argparse.ArgumentParser(description="Tool Contract and Registry Generator")
    parser.add_argument("--tools-dir", default="assets/tools", help="Directory with tool YAMLs")
    parser.add_argument("--bindings", default="tools/module_bindings.yaml", help="Path to module_bindings.yaml")
    parser.add_argument("--output-dir", default="build/generated", help="Directory for generated C++ files")
    parser.add_argument("--index-output", default="module_index.yaml", help="Output path for module index YAML")
    parser.add_argument("--variant", default="gpu", choices=["gpu", "cpu"], help="Distribution variant")
    parser.add_argument("--sdk-include-dir", default=None, help="VisionRT SDK include directory")
    parser.add_argument("--validator", default=None, help="Path to visionrt_validate_tool executable")
    args = parser.parse_args()

    if not args.sdk_include_dir:
        print("[ERROR] Missing required --sdk-include-dir argument in formal build", file=sys.stderr)
        return 1

    try:
        global VALID_TYPES, VALID_DTYPES, VALID_LAYOUTS, VALID_KINDS
        VALID_TYPES, VALID_DTYPES, VALID_LAYOUTS, VALID_KINDS = load_sdk_enums(Path(args.sdk_include_dir))
    except Exception as e:
        print(f"[ERROR] Invalid SDK include directory: {e}", file=sys.stderr)
        return 1

    tools_dir = Path(args.tools_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    yaml_files = sorted(set(tools_dir.glob("**/*.yaml")) | set(tools_dir.glob("**/*.yml")))
    if not yaml_files:
        print(f"[ERROR] No YAML tool declarations found in {tools_dir}", file=sys.stderr)
        return 1

    # Load bindings: strictly require bindings file
    bindings_path = Path(args.bindings)
    if not bindings_path.exists():
        print(f"[ERROR] Module bindings file not found: {bindings_path}", file=sys.stderr)
        return 1

    # Run SDK validator: required input (BS-02, F82, N06)
    if not args.validator:
        print("[ERROR] Missing required --validator executable", file=sys.stderr)
        return 1
    validator_bin = Path(args.validator)
    if not validator_bin.is_file():
        print(f"[ERROR] Validator executable not found: {validator_bin}", file=sys.stderr)
        return 1

    for yf in yaml_files:
        res = subprocess.run([str(validator_bin), str(yf)], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[ERROR] visionrt_validate_tool failed for {yf}:\n{res.stderr or res.stdout}", file=sys.stderr)
            return 1

    try:
        bindings_data = load_yaml(bindings_path)
    except Exception as e:
        print(f"[ERROR] Failed to load module bindings from {bindings_path}: {e}", file=sys.stderr)
        return 1

    seen_tool_ids = set()
    seen_reg_names = set()
    validated_tools = []
    reg_to_tool = {}

    for yf in yaml_files:
        try:
            doc = load_yaml(yf)
            validated = validate_tool_declaration(doc, yf, tools_dir=tools_dir)
            validated["_source_path"] = str(yf)
            try:
                rel_parts = yf.relative_to(tools_dir).parts
            except ValueError:
                rel_parts = yf.parts
            if any(part in rel_parts for part in ("custom", "示例", "examples")):
                validated["is_example"] = True
            tid = validated["tool_id"]
            if tid in seen_tool_ids:
                raise ContractValidationError(f"Duplicate tool_id '{tid}' in {yf}")
            seen_tool_ids.add(tid)

            reg = validated.get("entry", {}).get("registry_name")
            if reg:
                if reg in seen_reg_names:
                    raise ContractValidationError(f"Duplicate registry_name '{reg}' in {yf}")
                seen_reg_names.add(reg)
                reg_to_tool[reg] = validated

            validated_tools.append(validated)
        except Exception as e:
            print(f"[ERROR] Contract validation failed: {e}", file=sys.stderr)
            return 1

    if not validated_tools:
        print(f"[ERROR] No valid tool declarations found in {tools_dir}", file=sys.stderr)
        return 1

    # Validate module bindings: ensure each registry is bound in at most one module
    bound_regs = {}
    for mod_name, mod_info in bindings_data.get("modules", {}).items():
        for reg in mod_info.get("tools", {}):
            if reg in bound_regs:
                print(f"[ERROR] Duplicate binding for registry_name '{reg}' across modules '{bound_regs[reg]}' and '{mod_name}'", file=sys.stderr)
                return 1
            bound_regs[reg] = mod_name

    # Validate that all C++ bound tools exist in declared YAMLs
    for reg, mod_name in bound_regs.items():
        if reg not in reg_to_tool:
            print(f"[ERROR] Binding declared for unknown tool registry_name '{reg}' in module '{mod_name}'", file=sys.stderr)
            return 1

    # Validate that all C++ declared tools have bindings
    for reg in seen_reg_names:
        if reg not in bound_regs:
            print(f"[ERROR] C++ tool with registry_name '{reg}' has no implementation binding in {bindings_path}", file=sys.stderr)
            return 1

    # Generate module factory headers
    for mod_name, mod_info in bindings_data.get("modules", {}).items():
        factory_content = generate_module_factory_inc(mod_name, reg_to_tool, mod_info)
        factory_file = out_dir / f"{mod_name}_factory.inc"
        with open(factory_file, "w", encoding="utf-8") as f:
            f.write(factory_content)

    # Generate constants header with TypeSpec contracts
    constants_content = generate_port_constants_header(validated_tools)
    constants_path = out_dir / "PortConstants.h"
    with open(constants_path, "w", encoding="utf-8") as f:
        f.write(constants_content)

    # Generate module_index.yaml (YS-01)
    index_path = Path(args.index_output)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_data = build_module_index(validated_tools, bindings_data, variant=args.variant, project_root=tools_dir.parent.parent)
    with open(index_path, "w", encoding="utf-8") as f:
        yaml.dump(index_data, f, sort_keys=False, allow_unicode=True)

    print(f"[SUCCESS] Validated {len(validated_tools)} tool declarations, generated {constants_path} and {index_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
