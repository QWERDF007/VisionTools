#!/usr/bin/env python3
"""
Tool Contract & Module Registry Generator for VisionTools.
Conforms to shared SDK contract C03 and specs TS-02 / VS-02.

Validates tool YAML declarations against schema v1, ensures single source of truth,
and generates:
  1. C++ Port Constants header (<output-dir>/PortConstants.h)
  2. C++ Module Factory dispatch header (<output-dir>/<module>_factory.inc)
  3. Module Index JSON (<index-output>) for host runtime discovery.
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required for tool contract generation. Please install pyyaml.")

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

def load_yaml(file_path: Path) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def to_cpp_identifier(name: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if cleaned and cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned

def validate_tool_declaration(doc: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
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
            if p_id in port_ids:
                raise ContractValidationError(f"{prefix}Duplicate port id '{p_id}' in '{port_group}'")
            port_ids.add(p_id)

            c_ident = to_cpp_identifier(p_id).upper()
            if c_ident in port_c_idents:
                raise ContractValidationError(f"{prefix}C++ identifier collision in '{port_group}' for port '{p_id}' ({c_ident})")
            port_c_idents.add(c_ident)

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
                        raise ContractValidationError(f"{prefix}Default value for Float port '{p_id}' must be a number (got {default_val})")
                    if math.isnan(default_val) or math.isinf(default_val):
                        raise ContractValidationError(f"{prefix}Default value for Float port '{p_id}' must be finite")
                elif p_type == "Bool":
                    if not isinstance(default_val, bool):
                        raise ContractValidationError(f"{prefix}Default value for Bool port '{p_id}' must be boolean (got {default_val})")
                elif p_type == "String":
                    if not isinstance(default_val, str):
                        raise ContractValidationError(f"{prefix}Default value for String port '{p_id}' must be string (got {default_val})")

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

def generate_port_constants_header(tools: List[Dict[str, Any]]) -> str:
    # Sort deterministically by tool_id to ensure semantic invariance under file ordering
    sorted_tools = sorted(tools, key=lambda t: t["tool_id"])

    lines = [
        "// Generated by tools/generate_tool_contracts.py. DO NOT EDIT DIRECTLY.",
        "#pragma once",
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

        lines.append("namespace init {")
        for p in sorted(tool.get("init_params", []), key=lambda x: x["id"]):
            pid = p["id"]
            const_name = to_cpp_identifier(pid).upper()
            lines.append(f"inline constexpr const char* {const_name} = \"{pid}\";")
        lines.append("}")

        lines.append("namespace in {")
        for p in sorted(tool.get("inputs", []), key=lambda x: x["id"]):
            pid = p["id"]
            const_name = to_cpp_identifier(pid).upper()
            lines.append(f"inline constexpr const char* {const_name} = \"{pid}\";")
        lines.append("}")

        lines.append("namespace out {")
        for p in sorted(tool.get("outputs", []), key=lambda x: x["id"]):
            pid = p["id"]
            const_name = to_cpp_identifier(pid).upper()
            lines.append(f"inline constexpr const char* {const_name} = \"{pid}\";")
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
        "#include <cstring>",
        "#include <string>",
        "#include <stdexcept>",
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

def build_module_index(tools: List[Dict[str, Any]], bindings_data: Optional[Dict[str, Any]], variant: str = "gpu") -> Dict[str, Any]:
    """
    Generate module_index.json metadata for host loader.
    Only includes tools matching the specified variant (e.g. CPU excludes GPU tools).
    """
    entries = {}
    reg_to_module = {}

    if bindings_data and "modules" in bindings_data:
        for mod_name, mod_info in bindings_data["modules"].items():
            if variant == "cpu" and "infer" in mod_name:
                continue
            dll_path = mod_info.get("dll_name", f"bin/{mod_name}.dll")
            for reg in mod_info.get("tools", {}):
                reg_to_module[reg] = dll_path

    for tool in sorted(tools, key=lambda t: t["tool_id"]):
        tid = tool["tool_id"]
        reg = tool.get("entry", {}).get("registry_name", tid)
        kind = tool.get("kind", "cpp_function")

        if kind.startswith("cpp"):
            if reg not in reg_to_module:
                # Excluded by variant or not a C++ bound tool
                continue
            dll = reg_to_module[reg]
        else:
            dll = None

        entries[reg] = {
            "tool_id": tid,
            "display_name": tool.get("display_name", ""),
            "version": tool.get("version", "1.0.0"),
            "kind": kind,
            "module_dll": dll,
            "entry": tool.get("entry", {}),
            "inputs": [p["id"] for p in tool.get("inputs", [])],
            "outputs": [p["id"] for p in tool.get("outputs", [])]
        }

    return {
        "version": "1.0.0",
        "variant": variant,
        "tools": entries
    }

def main():
    parser = argparse.ArgumentParser(description="Tool Contract and Registry Generator")
    parser.add_argument("--tools-dir", default="assets/tools", help="Directory with tool YAMLs")
    parser.add_argument("--bindings", default="tools/module_bindings.yaml", help="Path to module_bindings.yaml")
    parser.add_argument("--output-dir", default="src/generated", help="Directory for generated C++ files")
    parser.add_argument("--index-output", default="module_index.json", help="Output path for module index JSON")
    parser.add_argument("--variant", default="gpu", choices=["gpu", "cpu"], help="Distribution variant")
    args = parser.parse_args()

    tools_dir = Path(args.tools_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    yaml_files = sorted(tools_dir.glob("**/*.yaml"))
    if not yaml_files:
        print(f"[WARN] No YAML files found in {tools_dir}")
        return 0

    seen_tool_ids = set()
    seen_reg_names = set()
    validated_tools = []
    reg_to_tool = {}

    for yf in yaml_files:
        try:
            doc = load_yaml(yf)
            validated = validate_tool_declaration(doc, yf)
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

    # Load bindings if provided
    bindings_path = Path(args.bindings)
    bindings_data = None
    if bindings_path.exists():
        bindings_data = load_yaml(bindings_path)
        # Validate that all C++ bound tools exist in declared YAMLs
        bound_regs = set()
        for mod_name, mod_info in bindings_data.get("modules", {}).items():
            for reg in mod_info.get("tools", {}):
                if reg not in reg_to_tool:
                    print(f"[ERROR] Binding declared for unknown tool registry_name '{reg}' in module '{mod_name}'", file=sys.stderr)
                    return 1
                bound_regs.add(reg)

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

    # Generate constants header
    constants_content = generate_port_constants_header(validated_tools)
    constants_path = out_dir / "PortConstants.h"
    with open(constants_path, "w", encoding="utf-8") as f:
        f.write(constants_content)

    # Generate module_index.json
    index_path = Path(args.index_output)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_data = build_module_index(validated_tools, bindings_data, variant=args.variant)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Validated {len(validated_tools)} tool declarations, generated {constants_path} and {index_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
