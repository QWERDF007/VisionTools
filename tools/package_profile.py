"""Target profiles for on-demand packaging and the dependency selection they imply.

The package must be explainable: every shipped file belongs to a component, is required by a
named entry point and, for TensorRT architecture resources, targets a declared SM. This module is
the single place where that decision is made, so the CMake install rule, the packaging script and
the delivery report cannot disagree.

Unknown TensorRT builder layouts are refused instead of being copied wholesale: a release that
silently ships every architecture is exactly what the profile exists to prevent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


class ProfileError(ValueError):
    """Raised when a profile, an SM selection or a dependency layout cannot be explained."""


# A TensorRT builder resource that carries its architecture in the file name, for example
# `nvinfer_builder_resource_10.14.1_sm89.dll` or `nvinfer_builder_resource_sm120.dll`.
_SM_IN_NAME = re.compile(r"(?:^|[_.-])sm[_.-]?(\d{2,3})(?:[_.-]|$)", re.IGNORECASE)

_TRT_CORE = ("nvinfer_", "nvinfer_plugin", "nvonnxparser", "nvinfer_lean", "nvinfer_dispatch")
_CUDA_RUNTIME = ("cudart", "cublas", "cufft", "curand", "cudnn", "nvrtc")
# The inference stack's own libraries: they are only reachable through the infer module, so a
# profile without inference must not ship them either.
_INFERENCE = ("inferrt_",)


@dataclass(frozen=True)
class Profile:
    name: str
    gpu_architectures: tuple[str, ...]
    infer_enabled: bool
    engine_build: bool
    runtime: str
    hardware_forward_compatibility: bool
    description: str = ""

    @property
    def sm_numbers(self) -> tuple[int, ...]:
        return tuple(int(architecture[2:]) for architecture in self.gpu_architectures)

    def as_report(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "gpu_architectures": list(self.gpu_architectures),
            "infer_enabled": self.infer_enabled,
            "engine_build": self.engine_build,
            "runtime": self.runtime,
            "hardware_forward_compatibility": self.hardware_forward_compatibility,
        }


@dataclass
class Selection:
    """Files that ship, files that do not, and why."""

    profile: Profile
    kept: list[dict] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def report(self) -> dict:
        components: dict[str, dict] = {}
        for entry in self.kept:
            component = entry["component"]
            bucket = components.setdefault(component, {"files": 0, "bytes": 0})
            bucket["files"] += 1
            size = entry.get("size_bytes")
            bucket["bytes"] += int(size) if isinstance(size, int) and size >= 0 else 0
        return {
            "schema_version": 1,
            "profile": self.profile.as_report(),
            "components": components,
            "files": self.kept,
            "excluded": self.dropped,
            "errors": self.errors,
        }


def load_profiles(path: Path) -> dict[str, Profile]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ProfileError(f"profile 文件需要 schema_version 1：{path}")
    raw_profiles = document.get("profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise ProfileError(f"profile 文件缺少 profiles：{path}")
    profiles: dict[str, Profile] = {}
    for name, raw in raw_profiles.items():
        if not isinstance(raw, dict):
            raise ProfileError(f"profile {name} 必须是对象")
        architectures = raw.get("gpu_architectures", [])
        if not isinstance(architectures, list) or any(not isinstance(item, str) for item in architectures):
            raise ProfileError(f"profile {name} 的 gpu_architectures 必须是字符串数组")
        for architecture in architectures:
            if not re.fullmatch(r"sm\d{2,3}", architecture):
                raise ProfileError(f"profile {name} 的架构名称无效：{architecture}")
        profiles[name] = Profile(
            name=name,
            gpu_architectures=tuple(architectures),
            infer_enabled=bool(raw.get("infer_enabled", False)),
            engine_build=bool(raw.get("engine_build", False)),
            runtime=str(raw.get("runtime", "none")),
            hardware_forward_compatibility=bool(raw.get("hardware_forward_compatibility", False)),
            description=str(raw.get("description", "")),
        )
    return profiles


def resolve_profile(profiles: dict[str, Profile], name: str, sms: Sequence[str] | None = None,
                    engine_build: bool | None = None) -> Profile:
    if name not in profiles:
        raise ProfileError(f"未知 profile：{name}（可用：{', '.join(sorted(profiles))}）")
    profile = profiles[name]
    if name != "gpu-custom":
        if sms:
            raise ProfileError(f"profile {name} 已固定架构 {list(profile.gpu_architectures)}，不接受 --sm")
        return profile

    if not sms:
        raise ProfileError("gpu-custom 需要显式 --sm，例如 --sm 89,86")
    architectures: list[str] = []
    for value in sms:
        cleaned = value.strip().lower()
        cleaned = cleaned[2:] if cleaned.startswith("sm") else cleaned
        if not cleaned.isdigit():
            raise ProfileError(f"SM 取值无效：{value}")
        architecture = f"sm{int(cleaned)}"
        if architecture not in architectures:
            architectures.append(architecture)
    return Profile(
        name=profile.name,
        gpu_architectures=tuple(sorted(architectures, key=lambda item: int(item[2:]))),
        infer_enabled=True,
        engine_build=profile.engine_build if engine_build is None else bool(engine_build),
        runtime=profile.runtime,
        hardware_forward_compatibility=profile.hardware_forward_compatibility,
        description=profile.description,
    )


def _entry_name(entry: dict) -> str:
    destination = str(entry.get("destination") or entry.get("relative_path") or "")
    return Path(destination.replace("\\", "/")).name


def _component_of(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("nvinfer_builder_resource") and "_ptx" in lowered:
        return "tensorrt-ptx-resource"
    if lowered.startswith("nvinfer_builder_resource"):
        return "tensorrt-builder-resource"
    if lowered.startswith(_TRT_CORE):
        return "tensorrt"
    if lowered.startswith(_INFERENCE):
        return "inference-runtime"
    if lowered.startswith(_CUDA_RUNTIME):
        return "cuda-runtime"
    if lowered.startswith("opencv"):
        return "opencv"
    if lowered.startswith("qt6"):
        return "qt"
    if lowered.startswith("python") or lowered.endswith(".pyd"):
        return "python"
    if lowered.startswith("visiontools_"):
        return "tools-module"
    return "shared"


def _reason_for(component: str, profile: Profile, sm: int | None = None) -> str:
    if component == "tensorrt-builder-resource":
        return f"TensorRT 分区 builder 资源，目标 {f'sm{sm}' if sm is not None else '未知架构'}"
    if component == "tensorrt":
        return "TensorRT 运行时（PE 导入闭包）"
    if component == "cuda-runtime":
        return "CUDA 运行库（PE 导入闭包）"
    if component == "inference-runtime":
        return "推理运行时（infer 模块的导入闭包）"
    if component == "tools-module":
        return "工具模块入口"
    if component == "opencv":
        return "OpenCV 实际链接目标"
    if component == "qt":
        return "Qt 插件与库（windeployqt/实际 QML 模块）"
    if component == "python":
        return "Python 运行时与该工具所需扩展"
    return "公共运行时依赖"


def select_dependencies(entries: Iterable[dict], profile: Profile) -> Selection:
    """Decides every file's fate from the profile alone.

    Each entry needs at least a `source` and a `destination`; unknown extra keys are preserved so the
    manifest can carry `required_by` from the build.
    """

    selection = Selection(profile=profile)
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("source") or not entry.get("destination"):
            raise ProfileError("运行时文件条目缺少 source 或 destination")
        name = _entry_name(entry)
        component = _component_of(name)
        source = str(entry["source"])
        sm: int | None = None

        if component == "tensorrt-ptx-resource":
            # PTX is a compatibility resource, not an architecture partition: it ships only when the
            # profile explicitly promises forward compatibility.
            if not profile.hardware_forward_compatibility:
                selection.dropped.append({"source": source, "destination": entry["destination"],
                                          "component": component,
                                          "reason": "该 profile 不保留 PTX 前向兼容资源"})
                continue
            selection.kept.append({
                "source": source, "destination": entry["destination"], "component": component,
                "reason": "PTX 硬件前向兼容资源（profile 显式启用）",
                "required_by": entry.get("required_by") if isinstance(entry.get("required_by"), list) else ["declared-runtime"],
                "target_sm": ["shared"],
            })
            continue

        if component == "tensorrt-builder-resource" and not profile.engine_build:
            # A runtime-only profile never ships build resources: the UI must not promise an
            # engine build that the package cannot perform.
            selection.dropped.append({"source": source, "destination": entry["destination"],
                                      "component": component, "reason": "该 profile 不提供现场构建能力"})
            continue

        if component == "tensorrt-builder-resource":
            match = _SM_IN_NAME.search(name)
            if match:
                sm = int(match.group(1))
            elif not profile.infer_enabled:
                selection.dropped.append({"source": source, "destination": entry["destination"],
                                          "component": component, "reason": "该 profile 不包含推理能力"})
                continue
            else:
                # An unknown layout is refused, never copied: shipping every architecture is what
                # the profile exists to prevent.
                selection.errors.append(
                    f"无法从文件名判断 TensorRT builder 资源的目标架构：{name}；未知布局拒绝打包")
                continue

        if not profile.infer_enabled and component in ("tensorrt", "cuda-runtime", "tensorrt-builder-resource",
                                                       "inference-runtime"):
            selection.dropped.append({"source": source, "destination": entry["destination"],
                                      "component": component, "reason": "该 profile 不包含 GPU 推理"})
            continue

        if component == "tensorrt-builder-resource" and sm is not None and sm not in profile.sm_numbers:
            selection.dropped.append({"source": source, "destination": entry["destination"],
                                      "component": component,
                                      "reason": f"sm{sm} 不在目标架构 {list(profile.gpu_architectures)} 内"})
            continue

        required_by = entry.get("required_by")
        if not isinstance(required_by, list) or not required_by:
            required_by = ["declared-runtime"]
        kept = dict(entry)
        kept.update({
            "component": component,
            "reason": _reason_for(component, profile, sm),
            "required_by": required_by,
            "target_sm": [f"sm{sm}"] if sm is not None else ["shared"],
        })
        selection.kept.append(kept)

    # A build profile that promises on-site engine building must ship a builder resource for every
    # architecture it targets: a package that cannot build the engine it advertises is incomplete,
    # and until now an empty resource set passed the filter without a word.
    if selection.profile.engine_build and selection.profile.infer_enabled:
        shipped = {sm for item in selection.kept
                   if item["component"] == "tensorrt-builder-resource" for sm in item["target_sm"]}
        for architecture in selection.profile.gpu_architectures:
            if architecture not in shipped:
                selection.errors.append(
                    f"profile {selection.profile.name} 缺少 {architecture} 的 TensorRT builder 资源；"
                    "无法在现场构建引擎")
    return selection


def apply_to_manifest(metadata: dict, selection: Selection) -> dict:
    """Writes the selected files back into the build manifest, keeping every other key."""

    updated = dict(metadata)
    updated["runtime_files"] = selection.kept
    updated["package_profile"] = selection.profile.as_report()
    updated["dependencies_report"] = {
        "excluded": selection.dropped,
        "errors": selection.errors,
    }
    return updated
