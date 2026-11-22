"""Packaging view of the public C++ enums and the Python SDK export."""
from enum import IntEnum
from pathlib import Path
import sys
import re

CONTRACTS = Path(__file__).resolve().parents[1] / "3rdparty/VisionRTContracts"
sys.path.insert(0, str(CONTRACTS / "tools"))
from export_enums import read_enums, python_source

HEADERS = [CONTRACTS / "include/visionrt/contracts" / name
           for name in ("ProtocolEnums.h", "TypeSpec.h", "ROI.h")]
DEFINITIONS = read_enums(HEADERS)
globals().update({name: IntEnum(name, dict(values)) for name, values in DEFINITIONS})
SDK_SOURCE = python_source(DEFINITIONS).encode("utf-8")


# Wire spellings belong to Contracts, independently of Python enum member names.
_TOOL_KIND_NAMES = {wire: ToolKind[member] for member, wire in re.findall(
    r'VRT_DECLARATION_NAME\(ToolKind, (\w+), "(\w+)"\)',
    (CONTRACTS / "include/visionrt/contracts/DeclarationNames.def").read_text(encoding="utf-8"))}

def declared_kind(implementation):
    return _TOOL_KIND_NAMES.get(implementation.get("kind"))
