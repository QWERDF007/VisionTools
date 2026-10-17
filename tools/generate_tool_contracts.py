#!/usr/bin/env python3
"""
Tool Contract & Module Registry Generator for VisionTools.
Implements shared SDK contract C03 and spec TS-02.

Validates tool YAML declarations against schema v1, ensures single source of truth,
and generates:
  1. C++ Port Constants header (<Module>PortConstants.h)
  2. C++ Contract & Factory Dispatch (<Module>Bindings.h / .cpp)
  3. Module Index JSON (module_index.json) for host runtime discovery.
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Supported types per shared SDK contract C02.2 / C04
VALID_TYPES = {
    "Int", "Float", "Bool", "String",
    "ImageBatch", "NumericArray", "StringArray", "RoiBatch"
}

VALID_DTYPES = {
    "uint8", "int32", "int64", "float32", "float64", "bool",
    "UInt8", "Int32", "Int64", "Float32", "Float64", "Bool"
}

VALID_LAYOUTS = {"dense", "batch_ragged", "Dense", "BatchRagged"}
VALID_KINDS = {"cpp_function", "cpp_class", "python_function", "python_class"}

class ContractValidationError(Exception):
    pass

def parse_yaml_simple(file_path: Path) -> Dict[str, Any]:
    """Parse YAML file using PyYAML if available, else a robust fallback parser."""
    try:
        import yaml
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        pass

    # Basic fallback YAML parser for schema v1 files
    result = {}
    current_list_key = None
    current_list_item = None

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line_str = line.rstrip()
        stripped = line_str.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line_str) - len(line_str.lstrip())
        
        # Check list item start
        if stripped.startswith("- "):
            item_content = stripped[2:].strip()
            if ":" in item_content:
                k, v = [x.strip() for x in item_content.split(":", 1)]
                current_list_item = {k: _parse_value(v)}
                if current_list_key:
                    result.setdefault(current_list_key, []).append(current_list_item)
            else:
                if current_list_key:
                    result.setdefault(current_list_key, []).append(_parse_value(item_content))
            continue

        if stripped.startswith("-") and len(stripped) == 1:
            current_list_item = {}
            if current_list_key:
                result.setdefault(current_list_key, []).append(current_list_item)
            continue

        if ":" in stripped:
            key, val = [x.strip() for x in stripped.split(":", 1)]
            if indent == 0:
                current_list_key = None
                current_list_item = None
                if not val:
                    current_list_key = key
                    result[key] = []
                else:
                    result[key] = _parse_value(val)
            elif current_list_item is not None:
                current_list_item[key] = _parse_value(val)
            elif indent > 0:
                pass

    return result

def _parse_value(v: str) -> Any:
    if v == "" or v == "''" or v == '""':
        return ""
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.lower() == "none" or v.lower() == "null":
        return None
    if v == "[]":
        return []
    if v == "{}":
        return {}
    try:
        if "." in v or "e" in v.lower():
            return float(v)
        return int(v)
    except ValueError:
        return v.strip("'\"")

def validate_tool_declaration(doc: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
    """Validate tool YAML against C03 specification rules."""
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

    # 4. Port lists validation
    for port_group in ["init_params", "inputs", "outputs"]:
        ports = doc.get(port_group, [])
        if not isinstance(ports, list):
            raise ContractValidationError(f"{prefix}'{port_group}' must be a list")

        port_ids = set()
        for p in ports:
            if not isinstance(p, dict):
                raise ContractValidationError(f"{prefix}Port in '{port_group}' must be an object")
            p_id = p.get("id")
            if not p_id or not isinstance(p_id, str):
                raise ContractValidationError(f"{prefix}Port missing 'id' in '{port_group}'")
            if p_id in port_ids:
                raise ContractValidationError(f"{prefix}Duplicate port id '{p_id}' in '{port_group}'")
            port_ids.add(p_id)

            p_type = p.get("type")
            if p_type not in VALID_TYPES:
                raise ContractValidationError(f"{prefix}Port '{p_id}' has invalid type '{p_type}'")

            # NumericArray validation
            if p_type == "NumericArray":
                dtype = p.get("dtype")
                if not dtype or dtype not in VALID_DTYPES:
                    raise ContractValidationError(f"{prefix}Port '{p_id}' of NumericArray requires valid dtype (got '{dtype}')")
                rank = p.get("rank")
                if rank is None or not isinstance(rank, int) or rank < 0:
                    raise ContractValidationError(f"{prefix}Port '{p_id}' of NumericArray requires integer rank >= 0")
                layout = p.get("layout", "dense")
                if layout not in VALID_LAYOUTS:
                    raise ContractValidationError(f"{prefix}Port '{p_id}' has invalid layout '{layout}'")
                if layout.lower() == "batch_ragged" and rank < 2:
                    raise ContractValidationError(f"{prefix}Port '{p_id}' with batch_ragged requires rank >= 2")

            # StringArray validation
            if p_type == "StringArray":
                if "dtype" in p and p["dtype"] is not None:
                    raise ContractValidationError(f"{prefix}StringArray port '{p_id}' must not declare dtype")
                rank = p.get("rank")
                if rank is None or not isinstance(rank, int) or rank < 0:
                    raise ContractValidationError(f"{prefix}Port '{p_id}' of StringArray requires integer rank >= 0")
                layout = p.get("layout", "dense")
                if layout not in VALID_LAYOUTS:
                    raise ContractValidationError(f"{prefix}Port '{p_id}' has invalid layout '{layout}'")

            # Default value validation
            if "default" in p and p["default"] is not None:
                default_val = p["default"]
                if p_type == "Int":
                    if not isinstance(default_val, int) or isinstance(default_val, bool):
                        raise ContractValidationError(f"{prefix}Default value for Int port '{p_id}' must be an integer (got {default_val})")
                elif p_type == "Float":
                    if not isinstance(default_val, (int, float)) or isinstance(default_val, bool):
                        raise ContractValidationError(f"{prefix}Default value for Float port '{p_id}' must be a number")
                    if math.isnan(default_val) or math.isinf(default_val):
                        raise ContractValidationError(f"{prefix}Default value for Float port '{p_id}' must be finite")
                elif p_type == "Bool":
                    if not isinstance(default_val, bool):
                        raise ContractValidationError(f"{prefix}Default value for Bool port '{p_id}' must be boolean")
                elif p_type == "String":
                    if not isinstance(default_val, str):
                        raise ContractValidationError(f"{prefix}Default value for String port '{p_id}' must be string")

            # Range validation
            if "min" in p and "max" in p:
                min_val = p["min"]
                max_val = p["max"]
                if min_val > max_val:
                    raise ContractValidationError(f"{prefix}Port '{p_id}' min ({min_val}) > max ({max_val})")
                if "default" in p and p["default"] is not None:
                    def_val = p["default"]
                    if isinstance(def_val, (int, float)) and not (min_val <= def_val <= max_val):
                        raise ContractValidationError(f"{prefix}Port '{p_id}' default ({def_val}) out of range [{min_val}, {max_val}]")

    return doc

def to_cpp_identifier(name: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if cleaned and cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned

def generate_port_constants_header(tools: List[Dict[str, Any]], module_name: str) -> str:
    lines = [
        "// Generated by tools/generate_tool_contracts.py. DO NOT EDIT DIRECTLY.",
        "#pragma once",
        "",
        "namespace visiontools::ports {",
        ""
    ]

    for tool in tools:
        tool_id = tool["tool_id"]
        safe_id = to_cpp_identifier(tool_id)
        lines.append(f"namespace {safe_id} {{")
        lines.append(f"inline constexpr const char* TOOL_ID = \"{tool_id}\";")
        
        reg_name = tool.get("entry", {}).get("registry_name", "")
        if reg_name:
            lines.append(f"inline constexpr const char* REGISTRY_NAME = \"{reg_name}\";")

        lines.append("namespace init {")
        for p in tool.get("init_params", []):
            pid = p["id"]
            const_name = to_cpp_identifier(pid).upper()
            lines.append(f"inline constexpr const char* {const_name} = \"{pid}\";")
        lines.append("}")

        lines.append("namespace in {")
        for p in tool.get("inputs", []):
            pid = p["id"]
            const_name = to_cpp_identifier(pid).upper()
            lines.append(f"inline constexpr const char* {const_name} = \"{pid}\";")
        lines.append("}")

        lines.append("namespace out {")
        for p in tool.get("outputs", []):
            pid = p["id"]
            const_name = to_cpp_identifier(pid).upper()
            lines.append(f"inline constexpr const char* {const_name} = \"{pid}\";")
        lines.append("}")

        lines.append("}") # namespace safe_id
        lines.append("")

    lines.append("} // namespace visiontools::ports")
    return "\n".join(lines) + "\n"

def generate_module_bindings(tools: List[Dict[str, Any]], module_name: str, class_bindings: Dict[str, str]) -> Tuple[str, str]:
    """
    Generate ModuleApi bindings header and implementation.
    class_bindings maps registry_name to C++ class name, e.g.
    'visionrt.image.read' -> 'visiontools::preprocess::ReadImageTool'
    """
    header_guard = f"VISIONTOOLS_{module_name.upper()}_BINDINGS_H"
    header_lines = [
        "// Generated by tools/generate_tool_contracts.py. DO NOT EDIT DIRECTLY.",
        f"#ifndef {header_guard}",
        f"#define {header_guard}",
        "",
        "#include <visionrt/sdk/Module.h>",
        "#include <visionrt/sdk/Contract.h>",
        "",
        f"namespace visiontools::{module_name} {{",
        "",
        "const visionrt::ModuleApi* getModuleApi() noexcept;",
        "",
        f"}} // namespace visiontools::{module_name}",
        "",
        f"#endif // {header_guard}",
        ""
    ]

    source_lines = [
        "// Generated by tools/generate_tool_contracts.py. DO NOT EDIT DIRECTLY.",
        f"#include \"{module_name}_bindings.h\"",
        "#include <cstring>",
        "#include <stdexcept>",
        ""
    ]

    # Include tool headers declared in class bindings
    for reg_name, class_info in class_bindings.items():
        if "header" in class_info:
            source_lines.append(f"#include \"{class_info['header']}\"")

    source_lines.extend([
        "",
        f"namespace visiontools::{module_name} {{",
        "",
        "static visionrt::IToolInstance* VT_CALL createToolInstance(",
        "    const char* registry_name,",
        "    const visionrt::ToolServices* services,",
        "    visionrt::ToolStatus* status) noexcept {",
        "    if (!registry_name) {",
        "        if (status) { status->code = -3; status->message = \"registry_name is null\"; }",
        "        return nullptr;",
        "    }",
        "    try {"
    ])

    for reg_name, class_info in class_bindings.items():
        class_name = class_info["class"]
        source_lines.extend([
            f"        if (std::strcmp(registry_name, \"{reg_name}\") == 0) {{",
            f"            if (status) {{ status->code = 0; status->message.clear(); }}",
            f"            return new {class_name}(*services);",
            "        }"
        ])

    source_lines.extend([
        "        if (status) {",
        "            status->code = -3;",
        "            status->message = std::string(\"Unknown registry_name: \") + registry_name;",
        "        }",
        "        return nullptr;",
        "    } catch (const std::exception& e) {",
        "        if (status) { status->code = -3; status->message = e.what(); }",
        "        return nullptr;",
        "    } catch (...) {",
        "        if (status) { status->code = -3; status->message = \"Unknown exception in create\"; }",
        "        return nullptr;",
        "    }",
        "}",
        "",
        "static void VT_CALL destroyToolInstance(visionrt::IToolInstance* instance) noexcept {",
        "    delete instance;",
        "}",
        "",
        "static const visionrt::ModuleApi s_api = {",
        "    visionrt::MODULE_ABI_MAJOR,",
        "    sizeof(visionrt::ModuleApi),",
        "    &createToolInstance,",
        "    &destroyToolInstance",
        "};",
        "",
        "const visionrt::ModuleApi* getModuleApi() noexcept {",
        "    return &s_api;",
        "}",
        "",
        f"}} // namespace visiontools::{module_name}",
        "",
        "// Public DLL export conforming to shared SDK contract C02.1",
        "extern \"C\" VT_EXPORT const visionrt::ModuleApi* VT_CALL visiontoolsModuleApi() noexcept {",
        f"    return visiontools::{module_name}::getModuleApi();",
        "}",
        ""
    ])

    return "\n".join(header_lines), "\n".join(source_lines)

def build_module_index(tools: List[Dict[str, Any]], module_map: Dict[str, str]) -> Dict[str, Any]:
    """Build module_index.json metadata for host loader."""
    entries = {}
    for tool in tools:
        tid = tool["tool_id"]
        reg = tool.get("entry", {}).get("registry_name", tid)
        mod_name = module_map.get(tid, "")
        dll_name = f"{mod_name}.dll" if mod_name else ""
        
        entries[reg] = {
            "tool_id": tid,
            "display_name": tool.get("display_name", ""),
            "version": tool.get("version", "1.0.0"),
            "kind": tool.get("kind", "cpp_function"),
            "module_dll": dll_name,
            "inputs": [p["id"] for p in tool.get("inputs", [])],
            "outputs": [p["id"] for p in tool.get("outputs", [])]
        }

    return {
        "version": "1.0.0",
        "tools": entries
    }

def main():
    parser = argparse.ArgumentParser(description="Tool Contract and Registry Generator")
    parser.add_argument("--tools-dir", default="assets/tools/builtin", help="Directory with tool YAMLs")
    parser.add_argument("--output-dir", default="src/generated", help="Directory for generated C++ files")
    parser.add_argument("--index-output", default="module_index.json", help="Output path for module index JSON")
    args = parser.parse_args()

    tools_dir = Path(args.tools_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    yaml_files = sorted(tools_dir.glob("**/*.yaml"))
    if not yaml_files:
        print(f"[WARN] No YAML files found in {tools_dir}")
        return 0

    seen_tool_ids = set()
    validated_tools = []

    for yf in yaml_files:
        try:
            doc = parse_yaml_simple(yf)
            validated = validate_tool_declaration(doc, yf)
            tid = validated["tool_id"]
            if tid in seen_tool_ids:
                raise ContractValidationError(f"Duplicate tool_id '{tid}' in {yf}")
            seen_tool_ids.add(tid)
            validated_tools.append(validated)
        except Exception as e:
            print(f"[ERROR] Contract validation failed: {e}", file=sys.stderr)
            return 1

    # Generate constants header
    constants_content = generate_port_constants_header(validated_tools, "all")
    constants_path = out_dir / "PortConstants.h"
    with open(constants_path, "w", encoding="utf-8") as f:
        f.write(constants_content)

    print(f"[SUCCESS] Validated {len(validated_tools)} tool declarations and generated {constants_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
