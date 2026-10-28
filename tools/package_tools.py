"""Collect the configured build into a validated tools ZIP and its derived index.

The package group owns three targets: the install directory, the ZIP and the
external index. All three are pre-checked before anything is written, collected
into same-volume staging paths, validated with the real host validator, and then
committed with a recoverable backup/rename transaction. A failed run never
deletes a previous usable package.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import zipfile

import json

import package_profile
from package_payload import CopyPlan, PackagingError, ToolPayload, load_json, relative_path, resolve_inside, save_json


MARKER_NAME = ".visiontools-package.json"
MARKER_KIND = "visiontools-tools-package"
MARKER_VERSION = 1
SOURCE_DIR_NAMES = ("assets", "python", "src", "tools", "tests", "3rdparty", "docs", "cmake", "probes")
def _remove_path(path: Path) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except FileNotFoundError:
        pass


def _rename(source: Path, target: Path) -> None:
    source = Path(source)
    target = Path(target)
    if source.is_dir() and not source.is_symlink():
        source.rename(target)
    else:
        os.replace(source, target)


@dataclass(frozen=True)
class PackageTargets:
    group: str
    install_dir: Path
    zip_output: Path
    index_output: Path


def resolve_targets(install_dir: Path, project_root: Path, zip_output: Path | None = None) -> PackageTargets:
    install = Path(install_dir).resolve()
    group = install.name
    if not group or group in (".", "..") or group.startswith(".") or group.endswith((".zip", ".module_index.json")):
        raise PackagingError(f"安装目录必须是专用发行组目录：{install}")
    expected_zip = install.parent / f"{group}.zip"
    if zip_output is None:
        archive = expected_zip
    else:
        archive = Path(zip_output).resolve()
        if archive.parent != install.parent or archive.name != expected_zip.name:
            raise PackagingError(f"ZIP 必须与安装目录属于同一发行组：{expected_zip}")
    return PackageTargets(group, install, archive, install.parent / f"{group}.module_index.json")


def _source_roots(project_root: Path, build_dir: Path) -> list[Path]:
    roots = [build_dir, project_root / ".git"]
    roots.extend(project_root / name for name in SOURCE_DIR_NAMES)
    return roots


def _reject_source_location(target: Path, project_root: Path, build_dir: Path, label: str) -> None:
    if target == project_root or project_root.is_relative_to(target):
        raise PackagingError(f"{label}不能是工程根目录或其上级目录：{target}")
    for root in _source_roots(project_root, build_dir):
        if target.is_relative_to(root):
            raise PackagingError(f"{label}不能位于源码或构建目录内：{target}")


def _marker_data(install_dir: Path):
    marker = install_dir / MARKER_NAME
    if not marker.is_file():
        return None
    try:
        data = load_json(marker)
    except (OSError, json.JSONDecodeError, PackagingError) as error:
        raise PackagingError(f"包标记无法读取：{marker}：{error}") from error
    if not isinstance(data, dict) or data.get("kind") != MARKER_KIND or data.get("version") != MARKER_VERSION:
        raise PackagingError(f"包标记无效：{marker}")
    return data


def _validate_install_target(targets: PackageTargets, project_root: Path, build_dir: Path) -> bool:
    install = targets.install_dir
    _reject_source_location(install, project_root, build_dir, "安装目录")
    if install.is_symlink():
        raise PackagingError(f"安装目录不能是符号链接：{install}")
    if install.is_relative_to(project_root):
        if install == project_root / "dist":
            raise PackagingError("不能直接使用 dist 根目录；请选择 dist 下的专用子目录")
        if install.parent != project_root / "dist":
            raise PackagingError(f"仓内输出只允许 dist 下的专用子目录：{install}")
    elif not install.parent.is_dir():
        raise PackagingError(f"仓外输出的父目录不存在：{install.parent}")

    if install.exists() and not install.is_dir():
        raise PackagingError(f"安装目录路径已被文件占用：{install}")

    marker = _marker_data(install) if install.is_dir() else None
    claimed = marker is not None
    if install.is_dir() and any(install.iterdir()) and not claimed:
        raise PackagingError(f"目标目录已存在且不是本工具包：{install}；请选择新目录或迁移该目录")
    if marker is not None and marker.get("package") != targets.group:
        raise PackagingError(f"包标记属于其他发行组：{install}（标记为 {marker.get('package')}）")
    return claimed


def _validate_artifact_target(target: Path, label: str, targets: PackageTargets, install_claimed: bool,
                              project_root: Path, build_dir: Path) -> None:
    _reject_source_location(target, project_root, build_dir, label)
    if target.is_symlink():
        raise PackagingError(f"{label}不能是符号链接：{target}")
    if target.exists() and target.is_dir():
        raise PackagingError(f"{label}不能是目录：{target}")
    if target.is_relative_to(project_root) and target.parent != project_root / "dist":
        raise PackagingError(f"仓内{label}只允许与安装目录同处 dist：{target}")
    if target.exists() and not install_claimed:
        raise PackagingError(f"存在同名旧产物但安装目录未标记，无法确认归属：{target}；请先移除或选择新发行组")


def _validate_targets(targets: PackageTargets, project_root: Path, build_dir: Path) -> None:
    claimed = _validate_install_target(targets, project_root, build_dir)
    _validate_artifact_target(targets.zip_output, "ZIP", targets, claimed, project_root, build_dir)
    _validate_artifact_target(targets.index_output, "外置索引", targets, claimed, project_root, build_dir)


def _load_build(build_dir: Path, config: str, project_root: Path, validator: Path | None = None,
                profile: package_profile.Profile | None = None) -> tuple[dict, dict, Path, dict[str, ToolPayload]]:
    try:
        metadata = load_json(build_dir / config / "resolved_dependencies.json")
        catalog = load_json(project_root / "tools/package_catalog.json")
    except (OSError, ValueError) as error:
        raise PackagingError(f"无法读取构建元数据或发行目录表：{error}") from error
    if not isinstance(metadata, dict) or not isinstance(catalog, dict):
        raise PackagingError("构建元数据与发行目录表必须是对象")
    # The profile decides which installed runtime files ship. It is applied to the real CMake
    # install output below, not to the (possibly empty) manifest list, so the report and the files
    # on disk come from one set.
    if profile is not None:
        metadata["__profile"] = profile
    if metadata.get("config") != config or not isinstance(metadata.get("variant"), str) or not metadata["variant"]:
        raise PackagingError("构建元数据配置不一致或缺少 variant")
    if catalog.get("schema_version") != 1 or set(catalog) != {"schema_version", "modules", "tools"}:
        raise PackagingError("发行目录表需要 schema_version 1、modules 与 tools")
    known_modules = catalog["modules"]
    if not isinstance(known_modules, list) or any(not isinstance(name, str) or not name for name in known_modules):
        raise PackagingError("发行目录表 modules 必须是模块名称数组")
    if len(set(known_modules)) != len(known_modules):
        raise PackagingError("发行目录表模块重复")
    modules = metadata.get("modules")
    if not isinstance(modules, dict) or set(modules) - set(known_modules):
        raise PackagingError("构建包含未在发行目录表声明的模块")
    if not isinstance(catalog["tools"], list):
        raise PackagingError("发行目录表 tools 必须是数组")
    index = {"schema_version": 1, "variant": metadata["variant"], "modules": {name: {} for name in modules}, "tools": {}}
    identities = set()
    payloads: dict[str, ToolPayload] = {}
    for entry in catalog["tools"]:
        if not isinstance(entry, dict) or set(entry) - {"definition", "module"} or "definition" not in entry:
            raise PackagingError("发行目录表工具条目只允许 definition 与 module")
        payload = ToolPayload.read(project_root, entry["definition"])
        source, declaration = payload.source, payload.document
        tool_id = declaration.get("tool_id")
        if not isinstance(tool_id, str) or not tool_id or tool_id in identities:
            raise PackagingError(f"工具声明 tool_id 为空或重复：{source}")
        identities.add(tool_id)
        module = entry.get("module")
        kind = declaration.get("kind")
        if kind == "cpp":
            if module not in known_modules:
                raise PackagingError(f"C++ 工具缺少有效模块归属：{tool_id}")
            if module not in modules:
                continue
        elif kind in ("host", "python_function", "python_class"):
            if module is not None:
                raise PackagingError(f"非 C++ 工具不能关联原生模块：{tool_id}")
        else:
            raise PackagingError(f"工具 kind 无效：{tool_id}")
        index["tools"][tool_id] = dict(entry)
        payloads[tool_id] = payload
    referenced = {entry["module"] for entry in index["tools"].values() if "module" in entry}
    if set(modules) != referenced:
        raise PackagingError("构建模块与声明归属不一致")
    validator_value = validator or metadata.get("validator", "")
    if not validator_value:
        raise PackagingError("构建元数据缺少宿主校验器路径")
    validator_path = Path(validator_value).resolve()
    if not validator_path.is_file():
        raise PackagingError(f"宿主校验器不存在：{validator_path}")
    return metadata, index, validator_path, payloads

def _precheck_payload(metadata: dict, payloads: dict[str, ToolPayload]) -> None:
    for module_name, module in metadata.get("modules", {}).items():
        if not isinstance(module, dict) or not module.get("source"):
            raise PackagingError(f"模块 {module_name} 缺少 source")
        source = Path(module["source"]).resolve()
        if not source.is_file():
            raise PackagingError(f"模块不存在：{source}")
        relative_path(f"bin/{source.name}", "模块目标")

    for payload in payloads.values():
        payload.validate()

    runtime_install = metadata.get("runtime_install")
    if runtime_install is not None:
        if not isinstance(runtime_install, dict) or not runtime_install.get("cmake") or not runtime_install.get("component"):
            raise PackagingError("运行时安装配置无效")
    for runtime in metadata.get("runtime_files", []):
        if not isinstance(runtime, dict) or not runtime.get("source") or not runtime.get("destination"):
            raise PackagingError("运行时文件配置无效")
        if not Path(runtime["source"]).resolve().is_file():
            raise PackagingError(f"运行时文件不存在：{runtime['source']}")
        relative_path(runtime["destination"], "运行时文件目标")


def _run_runtime_install(staging_dir: Path, build_dir: Path, config: str, runtime_install: dict) -> list[str]:
    command = [
        str(runtime_install["cmake"]), "--install", str(build_dir), "--config", config,
        "--component", str(runtime_install["component"]), "--prefix", str(staging_dir),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError as error:
        raise PackagingError(f"无法运行运行时安装命令：{error}") from error
    if result.returncode:
        raise PackagingError(f"运行时安装失败（返回码 {result.returncode}）：{result.stderr or result.stdout}")
    manifest = build_dir / f"install_manifest_{runtime_install['component']}.txt"
    if not manifest.is_file():
        raise PackagingError(f"运行时安装清单不存在：{manifest}")
    # The staged directory is the truth: only files that really landed can be selected or reported.
    installed_files = [
        path.relative_to(staging_dir).as_posix()
        for path in sorted(staging_dir.rglob("*"))
        if path.is_file()
    ]
    if not installed_files:
        raise PackagingError("运行时安装没有产生任何文件")
    return installed_files


def _select_runtime(entries: list[dict], profile) -> tuple[list[dict], list[dict]]:
    """Applies the one profile policy to the installed set and returns kept/dropped."""
    if profile is None:
        return [dict(entry, component="shared", reason="构建声明运行时文件",
                     required_by=["declared-runtime"], target_sm=["shared"]) for entry in entries], []
    selection = package_profile.select_dependencies(entries, profile)
    if selection.errors:
        raise PackagingError("；".join(selection.errors))
    return selection.kept, selection.dropped


def _dependency_report(staging_dir: Path, profile, kept: list[dict], dropped: list[dict]) -> dict:
    described = {entry["destination"]: entry for entry in kept}
    files = []
    components: dict[str, dict] = {}
    for path in sorted(staging_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(staging_dir).as_posix()
        entry = described.get(relative) or {
            "destination": relative,
            "component": "tools-payload",
            "reason": "工具声明、Python 代码或索引",
            "required_by": ["package"],
            "target_sm": ["shared"],
        }
        size = path.stat().st_size
        files.append({
            "destination": relative,
            "component": entry["component"],
            "reason": entry["reason"],
            "required_by": entry["required_by"],
            "target_sm": entry["target_sm"],
            "size_bytes": size,
        })
        bucket = components.setdefault(entry["component"], {"files": 0, "bytes": 0})
        bucket["files"] += 1
        bucket["bytes"] += size
    return {
        "schema_version": 1,
        "profile": profile.as_report() if profile is not None else None,
        "components": components,
        "files": files,
        "excluded": dropped,
        "errors": [],
    }


def _collect_package(metadata: dict, index: dict, payloads: dict[str, ToolPayload], staging_dir: Path,
                     build_dir: Path, config: str) -> dict:
    plan = CopyPlan()
    profile = metadata.get("__profile")
    final_index = {
        "schema_version": 1,
        "variant": index["variant"],
        "modules": {},
        "tools": {identifier: dict(entry) for identifier, entry in index["tools"].items()},
        "runtime_files": [],
    }
    module_files = {}
    runtime_install = metadata.get("runtime_install")
    # A profile without inference ships neither the infer module nor the tools that need it.
    infer_allowed = profile is None or profile.infer_enabled
    for module_name, module in metadata.get("modules", {}).items():
        if not infer_allowed and "infer" in module_name.lower():
            continue
        source = Path(module["source"]).resolve()
        relative = f"bin/{source.name}"
        module_files[module_name] = relative
        final_index["modules"][module_name] = {"library": relative}
        if not runtime_install:
            plan.add(source, Path(relative))

    for tool_id in list(final_index["tools"]):
        entry = final_index["tools"][tool_id]
        module = entry.get("module")
        if module is not None and module not in final_index["modules"]:
            del final_index["tools"][tool_id]
            continue
        payload = payloads[tool_id]
        payload.add_to(plan)
        entry["definition"] = payload.destination.as_posix()

    runtime_entries = []
    runtime_stage = None
    if runtime_install:
        # CMake installs into an isolated directory first. Nothing lands in the package until the
        # single profile policy has decided the final set.
        runtime_stage = staging_dir.parent / f".runtime-{uuid.uuid4().hex}"
        installed_files = _run_runtime_install(runtime_stage, build_dir, config, runtime_install)
        runtime_entries = [
            {"source": str((runtime_stage / relative).resolve()), "destination": relative}
            for relative in installed_files
        ]
        kept, dropped = _select_runtime(runtime_entries, profile)
        for entry in kept:
            plan.add(Path(entry["source"]), Path(entry["destination"]))
    else:
        runtime_entries = [
            {"source": str(Path(runtime["source"]).resolve()), "destination": runtime["destination"]}
            for runtime in metadata.get("runtime_files", [])
        ]
        kept, dropped = _select_runtime(runtime_entries, profile)
        for entry in kept:
            plan.add(Path(entry["source"]), Path(entry["destination"]))

    try:
        plan.copy_to(staging_dir)
    finally:
        if runtime_stage is not None:
            _remove_path(runtime_stage)

    for module_name, relative in module_files.items():
        if not (staging_dir / relative).is_file():
            raise PackagingError(f"模块未进入安装目录：{module_name} -> {relative}")
    module_libraries = set(module_files.values())
    final_index["runtime_files"] = sorted(
        entry["destination"] for entry in kept if entry["destination"] not in module_libraries
    )
    final_index["modules"] = [{"id": name, **module} for name, module in sorted(final_index["modules"].items())]
    final_index["tools"] = [{"tool_id": identifier, **entry} for identifier, entry in sorted(final_index["tools"].items())]
    if profile is not None:
        save_json(_dependency_report(staging_dir, profile, kept, dropped), staging_dir / "dependencies.json")
    return final_index


def _write_zip(staging_dir: Path, zip_path: Path) -> None:
    _remove_path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for source in sorted(staging_dir.rglob("*")):
            if source.is_file() and source.name != MARKER_NAME:
                bundle.write(source, source.relative_to(staging_dir).as_posix())


def _verify_zip(zip_path: Path, staging_dir: Path, index_bytes: bytes) -> None:
    try:
        with zipfile.ZipFile(zip_path) as bundle:
            names = bundle.namelist()
            if len(names) != len(set(names)):
                raise PackagingError(f"ZIP 包含重复条目：{zip_path}")
            if "module_index.json" not in names:
                raise PackagingError(f"ZIP 缺少 module_index.json：{zip_path}")
            if bundle.read("module_index.json") != index_bytes:
                raise PackagingError(f"ZIP 内索引与安装索引不一致：{zip_path}")
    except zipfile.BadZipFile as error:
        raise PackagingError(f"ZIP 无法读取：{zip_path}：{error}") from error
    expected = {
        path.relative_to(staging_dir).as_posix()
        for path in staging_dir.rglob("*")
        if path.is_file() and path.name != MARKER_NAME
    }
    if set(names) != expected:
        raise PackagingError(f"ZIP 内容与安装目录不一致：{zip_path}")


def _write_marker(staging_dir: Path, group: str) -> None:
    save_json({"kind": MARKER_KIND, "version": MARKER_VERSION, "package": group}, staging_dir / MARKER_NAME)


def _commit(outputs: list[tuple[Path, Path, str]], token: str) -> None:
    backups: list[tuple[Path, Path]] = []
    committed: list[Path] = []
    try:
        for staged, target, label in outputs:
            if target.exists() or target.is_symlink():
                backup = target.with_name(f".{target.name}.backup-{token}")
                _remove_path(backup)
                _rename(target, backup)
                backups.append((target, backup))
            _rename(staged, target)
            committed.append(target)
    except Exception as error:
        for target in reversed(committed):
            try:
                _remove_path(target)
            except OSError:
                pass
        for target, backup in reversed(backups):
            if backup.exists() or backup.is_symlink():
                try:
                    _rename(backup, target)
                except OSError:
                    pass
        raise PackagingError(f"打包提交失败，已恢复原产物：{error}") from error
    for _, backup in backups:
        try:
            _remove_path(backup)
        except OSError as error:
            print(f"[警告] 无法删除旧产物备份 {backup}：{error}", file=sys.stderr)


def verify_package(install_dir: Path, validator: Path) -> bool:
    validator = Path(validator).resolve()
    if not validator.is_file():
        raise PackagingError(f"宿主校验器不存在：{validator}")
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join([str(Path(install_dir).resolve() / "bin"), *os.get_exec_path(environment)])
    result = subprocess.run(
        [str(validator), "--package", str(Path(install_dir).resolve())],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=environment,
    )
    if result.returncode:
        print(f"宿主包校验失败（返回码 {result.returncode}）：{result.stderr or result.stdout}", file=sys.stderr)
        return False
    root = Path(install_dir).resolve()
    index = load_json(root / "module_index.json")
    for tool in index["tools"]:
        load_json(root / tool["definition"])
    return True


def package_tools(
    build_dir: Path,
    install_dir: Path,
    config: str = "Release",
    project_root: Path | None = None,
    validator: Path | None = None,
    zip_output: Path | None = None,
    profile: package_profile.Profile | None = None,
) -> Path:
    build_dir = Path(build_dir).resolve()
    project_root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    targets = resolve_targets(install_dir, project_root, zip_output)
    metadata, index, build_validator, payloads = _load_build(build_dir, config, project_root, validator, profile)
    _validate_targets(targets, project_root, build_dir)
    _precheck_payload(metadata, payloads)

    token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staging_dir = targets.install_dir.parent / f".{targets.install_dir.name}.staging-{token}"
    staging_zip = targets.zip_output.parent / f".{targets.zip_output.name}.staging-{token}"
    staging_index = targets.index_output.parent / f".{targets.index_output.name}.staging-{token}"

    try:
        targets.install_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir.mkdir()
        final_index = _collect_package(metadata, index, payloads, staging_dir, build_dir, config)
        save_json(final_index, staging_dir / "module_index.json")

        # 构建元数据只有一个来源：构建结束目标写出的 <build-dir>/build_info.json。多配置目录
        # （<build-dir>/<config>/）里的同名文件属于旧流程，会掩盖真实构建时刻，必须忽略。
        build_metadata_path = build_dir / "build_info.json"
        if build_metadata_path.is_file():
            try:
                build_metadata = json.loads(build_metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise PackagingError(f"构建元数据不可读：{error}") from error
            epoch = build_metadata.get("built_at_epoch") if isinstance(build_metadata, dict) else None
            if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= 0:
                raise PackagingError(
                    f"{build_metadata_path} 缺少有效的 built_at_epoch，旧格式不得冒充构建时间"
                )
            shutil.copy2(build_metadata_path, staging_dir / "build_info.json")

        _write_marker(staging_dir, targets.group)
        if not verify_package(staging_dir, build_validator):
            raise PackagingError("宿主包校验失败")
        try:
            _write_zip(staging_dir, staging_zip)
        except OSError as error:
            raise PackagingError(f"写入 ZIP 失败：{error}") from error
        index_bytes = (staging_dir / "module_index.json").read_bytes()
        staging_index.write_bytes(index_bytes)
        _verify_zip(staging_zip, staging_dir, index_bytes)
        if staging_index.read_bytes() != index_bytes:
            raise PackagingError("外置索引与安装索引不一致")
        _commit([
            (staging_dir, targets.install_dir, "安装目录"),
            (staging_zip, targets.zip_output, "ZIP"),
            (staging_index, targets.index_output, "外置索引"),
        ], token)
    except PackagingError:
        raise
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        raise PackagingError(f"打包失败：{error}") from error
    finally:
        _remove_path(staging_dir)
        _remove_path(staging_zip)
        _remove_path(staging_index)

    print(f"已打包 {len(final_index['tools'])} 个工具：{targets.zip_output}")
    return targets.zip_output


def verify_runtime(install_dir: Path, consumer_bin: Path) -> bool:
    install_dir = Path(install_dir).resolve()
    index = load_json(install_dir / "module_index.json")
    module = next((item for item in index["modules"] if item["id"] == "visiontools_preprocess"), None)
    if module is None:
        raise PackagingError("原生运行验证需要 CPU 工具模块")
    library = resolve_inside(install_dir, module["library"], "模块库")
    environment = os.environ.copy()
    if os.name == "nt":
        system_root = Path(os.environ["SystemRoot"])
        environment["PATH"] = os.pathsep.join(map(str, [install_dir / "bin", system_root / "System32", system_root]))
    else:
        environment["LD_LIBRARY_PATH"] = os.pathsep.join([str(install_dir / "bin"), environment.get("LD_LIBRARY_PATH", "")])
    result = subprocess.run([str(Path(consumer_bin).resolve()), str(library)], env=environment)
    return result.returncode == 0

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--install-dir", type=Path, default=Path("dist/VisionTools"))
    parser.add_argument("--config", default="Release")
    parser.add_argument("--zip-output", type=Path)
    parser.add_argument("--validator", type=Path)
    parser.add_argument("--profile", help="打包目标：cpu / rtx40-runtime / rtx40-build / gpu-custom")
    parser.add_argument("--sm", help="gpu-custom 的目标架构，逗号分隔，例如 89,86")
    parser.add_argument("--engine-build", action="store_true", help="gpu-custom 是否保留现场构建能力")
    parser.add_argument("--profiles", type=Path, default=Path(__file__).with_name("package_profiles.json"))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-runtime", action="store_true")
    parser.add_argument("--consumer", type=Path,
                        default=Path("build/tests/test_installation_consumer.exe") if os.name == "nt"
                        else Path("build/tests/test_installation_consumer"))
    arguments = parser.parse_args()
    try:
        profile = None
        if arguments.profile:
            profiles = package_profile.load_profiles(arguments.profiles)
            sms = [item for item in arguments.sm.split(",") if item.strip()] if arguments.sm else None
            profile = package_profile.resolve_profile(profiles, arguments.profile, sms,
                                                       arguments.engine_build or None)
        if arguments.verify_only:
            metadata = load_json(arguments.build_dir / arguments.config / "resolved_dependencies.json")
            validator = arguments.validator or Path(metadata["validator"])
            if not verify_package(arguments.install_dir, validator):
                return 1
        else:
            package_tools(arguments.build_dir, arguments.install_dir, arguments.config,
                          validator=arguments.validator, zip_output=arguments.zip_output, profile=profile)
        if arguments.verify_runtime and not verify_runtime(arguments.install_dir, arguments.consumer):
            return 1
        return 0
    except (OSError, RuntimeError, KeyError, ValueError, json.JSONDecodeError) as error:
        print(f"打包失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
