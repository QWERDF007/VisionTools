"""Collect the configured build into a validated tools ZIP and its derived index.

The package group owns three targets: the install directory, the ZIP and the
external index. All three are pre-checked before anything is written, collected
into same-volume staging paths, validated with the real SDK validator, and then
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

import yaml


MARKER_NAME = ".visiontools-package.yaml"
MARKER_KIND = "visiontools-tools-package"
MARKER_VERSION = 1
SOURCE_DIR_NAMES = ("assets", "src", "tools", "tests", "3rdparty", "docs", "cmake", "probes")
CODE_SUFFIXES = (".py", ".pyd", ".so")
CACHE_PARTS = {"__pycache__"}
CACHE_SUFFIXES = {".pyc"}


class PackagingError(RuntimeError):
    """An actionable packaging failure with a Chinese message."""


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise PackagingError(f"YAML 存在重复键：{key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_yaml(path: Path):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.load(stream, Loader=UniqueKeyLoader)


def save_yaml(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _relative_path(relative, label: str) -> Path:
    path = Path(relative)
    if not path.parts or path.is_absolute() or path.drive or ".." in path.parts:
        raise PackagingError(f"{label}必须位于包根目录内：{relative}")
    return path


def resolve_inside(root: Path, relative, label: str = "包内引用") -> Path:
    reference = _relative_path(relative, label)
    resolved = (Path(root) / reference).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise PackagingError(f"{label}逃逸根目录：{relative}")
    return resolved


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
    if not group or group in (".", "..") or group.startswith(".") or group.endswith((".zip", ".module_index.yaml")):
        raise PackagingError(f"安装目录必须是专用发行组目录：{install}")
    expected_zip = install.parent / f"{group}.zip"
    if zip_output is None:
        archive = expected_zip
    else:
        archive = Path(zip_output).resolve()
        if archive.parent != install.parent or archive.name != expected_zip.name:
            raise PackagingError(f"ZIP 必须与安装目录属于同一发行组：{expected_zip}")
    return PackageTargets(group, install, archive, install.parent / f"{group}.module_index.yaml")


class CopyPlan:
    """Path-preserving copy plan that deduplicates shared package roots."""

    def __init__(self) -> None:
        self._items: dict[str, Path] = {}

    def add(self, source: Path, destination: Path) -> None:
        source = Path(source).resolve()
        relative = _relative_path(destination, "包内目标")
        key = relative.as_posix()
        existing = self._items.get(key)
        if existing is None:
            self._items[key] = source
            return
        if existing == source:
            return
        try:
            same_bytes = existing.read_bytes() == source.read_bytes()
        except OSError as error:
            raise PackagingError(f"无法比较包内目标冲突：{key}：{error}") from error
        if not same_bytes:
            raise PackagingError(f"包内目标路径冲突：{key} 来自不同来源 {existing} 与 {source}")

    def items(self) -> list[tuple[Path, Path]]:
        return [(Path(key), source) for key, source in sorted(self._items.items())]

    def copy_to(self, root: Path) -> None:
        for relative, source in self.items():
            target = resolve_inside(root, relative, "包内目标")
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(source, target)
            except OSError as error:
                raise PackagingError(f"复制包文件失败：{source} -> {target}：{error}") from error


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
        data = load_yaml(marker)
    except (OSError, yaml.YAMLError, PackagingError) as error:
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


def _load_build(build_dir: Path, config: str, validator: Path | None = None) -> tuple[dict, dict, Path]:
    try:
        metadata = load_yaml(build_dir / config / "resolved_dependencies.yaml")
        index = load_yaml(build_dir / "generated/module_index.yaml")
    except (OSError, yaml.YAMLError, PackagingError) as error:
        raise PackagingError(f"无法读取构建元数据或索引：{error}") from error
    if not isinstance(metadata, dict) or not isinstance(index, dict):
        raise PackagingError("构建元数据与索引必须是映射")
    if metadata.get("config") != config or metadata.get("variant") != index.get("variant"):
        raise PackagingError("构建元数据与索引配置不一致")
    if not index.get("variant") or index.get("schema_version") != 2:
        raise PackagingError("构建索引格式无效（需要 schema_version 2 与 variant）")
    if set(metadata.get("modules", {})) != set(index.get("modules", {})):
        raise PackagingError("构建模块与索引模块不一致")
    validator_value = validator or metadata.get("validator", "")
    if not validator_value:
        raise PackagingError("构建元数据缺少 SDK 校验器路径")
    validator_path = Path(validator_value).resolve()
    if not validator_path.is_file():
        raise PackagingError(f"SDK 校验器不存在：{validator_path}")
    return metadata, index, validator_path


def _validate_package_root(package_root: Path, project_root: Path) -> None:
    if not package_root.is_relative_to(project_root):
        raise PackagingError(f"Python 代码包根必须位于工程内：{package_root}")
    for root in [project_root, *(project_root / name for name in SOURCE_DIR_NAMES)]:
        if root == package_root or root.is_relative_to(package_root):
            raise PackagingError(f"Python 代码包根过大，包含工程源码：{package_root}")


def _precheck_payload(metadata: dict, index: dict, project_root: Path) -> None:
    for module_name, module in metadata.get("modules", {}).items():
        if not isinstance(module, dict) or not module.get("source"):
            raise PackagingError(f"模块 {module_name} 缺少 source")
        source = Path(module["source"]).resolve()
        if not source.is_file():
            raise PackagingError(f"模块不存在：{source}")
        _relative_path(f"bin/{source.name}", "模块目标")

    tools = index.get("tools", {})
    if not isinstance(tools, dict):
        raise PackagingError("索引 tools 必须是映射")
    for tool_id, entry in tools.items():
        if not isinstance(entry, dict) or not entry.get("definition"):
            raise PackagingError(f"工具 {tool_id} 缺少 definition")
        definition_relative = _relative_path(entry["definition"], "工具声明")
        source = resolve_inside(project_root, definition_relative, "工具声明")
        if not source.is_file():
            raise PackagingError(f"工具声明不存在：{source}")
        declaration = load_yaml(source)
        if not isinstance(declaration, dict):
            raise PackagingError(f"工具声明必须是映射：{source}")
        if declaration.get("tool_id") != tool_id:
            raise PackagingError(f"工具声明 tool_id 不一致：{source}（索引 {tool_id}）")
        kind = declaration.get("kind", "")
        entry_data = declaration.get("entry")
        if not isinstance(entry_data, dict):
            raise PackagingError(f"工具 {tool_id} 缺少 entry")
        if kind in ("python_function", "python_class"):
            script = resolve_inside(source.parent, entry_data.get("file", ""), "Python 入口脚本")
            if not script.is_file():
                raise PackagingError(f"Python 入口脚本不存在：{script}")
            _validate_package_root(script.parent, project_root)
        resources = declaration.get("resources", [])
        if not isinstance(resources, list):
            raise PackagingError(f"工具 {tool_id} 的 resources 必须是列表")
        for resource in resources:
            resolved = resolve_inside(source.parent, resource, "资源")
            if not resolved.exists():
                raise PackagingError(f"资源不存在：{resolved}")
            if not resolved.is_relative_to(project_root):
                raise PackagingError(f"资源必须位于工程根目录内：{resolved}")

    runtime_install = metadata.get("runtime_install")
    if runtime_install is not None:
        if not isinstance(runtime_install, dict) or not runtime_install.get("cmake") or not runtime_install.get("component"):
            raise PackagingError("运行时安装配置无效")
    for runtime in metadata.get("runtime_files", []):
        if not isinstance(runtime, dict) or not runtime.get("source") or not runtime.get("destination"):
            raise PackagingError("运行时文件配置无效")
        if not Path(runtime["source"]).resolve().is_file():
            raise PackagingError(f"运行时文件不存在：{runtime['source']}")
        _relative_path(runtime["destination"], "运行时文件目标")


def _collect_python_package(plan: CopyPlan, script: Path, project_root: Path) -> None:
    package_root = script.parent
    _validate_package_root(package_root, project_root)
    for path in sorted(package_root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in CACHE_PARTS for part in path.parts) or path.suffix in CACHE_SUFFIXES:
            continue
        if path.suffix not in CODE_SUFFIXES and path != script:
            continue
        plan.add(path, path.relative_to(project_root))


def _collect_resources(plan: CopyPlan, declaration_path: Path, declaration: dict, project_root: Path) -> None:
    for resource in declaration.get("resources", []):
        source = resolve_inside(declaration_path.parent, resource, "资源")
        if source.is_dir():
            for path in sorted(source.rglob("*")):
                if not path.is_file():
                    continue
                if any(part in CACHE_PARTS for part in path.parts) or path.suffix in CACHE_SUFFIXES:
                    continue
                plan.add(path, path.relative_to(project_root))
        elif source.is_file():
            plan.add(source, source.relative_to(project_root))
        else:
            raise PackagingError(f"资源不存在：{source}")


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
    installed_files = []
    for installed in manifest.read_text(encoding="utf-8").splitlines():
        if not installed.strip():
            continue
        relative = Path(os.path.relpath(Path(installed).resolve(), staging_dir.resolve()))
        if relative.parts and relative.parts[0] == "..":
            raise PackagingError(f"运行时安装清单文件逃逸包根：{installed}")
        installed_files.append(relative.as_posix())
    return installed_files


def _collect_package(metadata: dict, index: dict, staging_dir: Path, project_root: Path, build_dir: Path,
                     config: str) -> dict:
    plan = CopyPlan()
    final_index = {
        "schema_version": 2,
        "variant": index["variant"],
        "modules": {},
        "tools": dict(index.get("tools", {})),
        "runtime_files": [],
    }
    module_files = {}
    runtime_install = metadata.get("runtime_install")
    for module_name, module in metadata.get("modules", {}).items():
        source = Path(module["source"]).resolve()
        relative = f"bin/{source.name}"
        module_files[module_name] = relative
        final_index["modules"][module_name] = {"library": relative}
        if not runtime_install:
            plan.add(source, Path(relative))

    for tool_id, entry in final_index["tools"].items():
        source = resolve_inside(project_root, entry["definition"], "工具声明")
        declaration = load_yaml(source)
        plan.add(source, Path(entry["definition"]))
        if declaration.get("kind") in ("python_function", "python_class"):
            script = resolve_inside(source.parent, declaration["entry"]["file"], "Python 入口脚本")
            _collect_python_package(plan, script, project_root)
        _collect_resources(plan, source, declaration, project_root)

    for runtime in metadata.get("runtime_files", []):
        source = Path(runtime["source"]).resolve()
        plan.add(source, Path(runtime["destination"]))

    plan.copy_to(staging_dir)

    installed_files = []
    if runtime_install:
        installed_files = _run_runtime_install(staging_dir, build_dir, config, runtime_install)
    for module_name, relative in module_files.items():
        if not (staging_dir / relative).is_file():
            raise PackagingError(f"模块未进入安装目录：{module_name} -> {relative}")
    module_libraries = set(module_files.values())
    final_index["runtime_files"] = sorted({path for path in installed_files if path not in module_libraries})
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
            if "module_index.yaml" not in names:
                raise PackagingError(f"ZIP 缺少 module_index.yaml：{zip_path}")
            if bundle.read("module_index.yaml") != index_bytes:
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
    save_yaml({"kind": MARKER_KIND, "version": MARKER_VERSION, "package": group}, staging_dir / MARKER_NAME)


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
        raise PackagingError(f"SDK 校验器不存在：{validator}")
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join([str(Path(install_dir).resolve() / "bin"), *os.get_exec_path(environment)])
    result = subprocess.run(
        [str(validator), "--package", str(Path(install_dir).resolve())],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=environment,
    )
    if result.returncode:
        print(f"SDK 包校验失败（返回码 {result.returncode}）：{result.stderr or result.stdout}", file=sys.stderr)
    return result.returncode == 0


def package_tools(
    build_dir: Path,
    install_dir: Path,
    config: str = "Release",
    project_root: Path | None = None,
    validator: Path | None = None,
    zip_output: Path | None = None,
) -> Path:
    build_dir = Path(build_dir).resolve()
    project_root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    targets = resolve_targets(install_dir, project_root, zip_output)
    metadata, index, build_validator = _load_build(build_dir, config, validator)
    _validate_targets(targets, project_root, build_dir)
    _precheck_payload(metadata, index, project_root)

    token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staging_dir = targets.install_dir.parent / f".{targets.install_dir.name}.staging-{token}"
    staging_zip = targets.zip_output.parent / f".{targets.zip_output.name}.staging-{token}"
    staging_index = targets.index_output.parent / f".{targets.index_output.name}.staging-{token}"

    try:
        targets.install_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir.mkdir()
        final_index = _collect_package(metadata, index, staging_dir, project_root, build_dir, config)
        save_yaml(final_index, staging_dir / "module_index.yaml")
        _write_marker(staging_dir, targets.group)
        if not verify_package(staging_dir, build_validator):
            raise PackagingError("SDK 包校验失败")
        try:
            _write_zip(staging_dir, staging_zip)
        except OSError as error:
            raise PackagingError(f"写入 ZIP 失败：{error}") from error
        index_bytes = (staging_dir / "module_index.yaml").read_bytes()
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
    except (OSError, KeyError, ValueError, yaml.YAMLError) as error:
        raise PackagingError(f"打包失败：{error}") from error
    finally:
        _remove_path(staging_dir)
        _remove_path(staging_zip)
        _remove_path(staging_index)

    print(f"已打包 {len(final_index['tools'])} 个工具：{targets.zip_output}")
    return targets.zip_output


def verify_runtime(install_dir: Path, consumer_bin: Path) -> bool:
    install_dir = Path(install_dir).resolve()
    system_root = Path(os.environ.get("SystemRoot", "C:/Windows"))
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join(map(str, [install_dir / "bin", system_root / "System32", system_root]))
    result = subprocess.run([str(Path(consumer_bin).resolve()), str(install_dir)], env=environment)
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--install-dir", type=Path, default=Path("dist/VisionTools"))
    parser.add_argument("--config", default="Release")
    parser.add_argument("--zip-output", type=Path)
    parser.add_argument("--validator", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-runtime", action="store_true")
    parser.add_argument("--consumer", type=Path,
                        default=Path("build/tests/test_installation_consumer.exe") if os.name == "nt"
                        else Path("build/tests/test_installation_consumer"))
    arguments = parser.parse_args()
    try:
        if arguments.verify_only:
            metadata = load_yaml(arguments.build_dir / arguments.config / "resolved_dependencies.yaml")
            validator = arguments.validator or Path(metadata["validator"])
            if not verify_package(arguments.install_dir, validator):
                return 1
        else:
            package_tools(arguments.build_dir, arguments.install_dir, arguments.config,
                          validator=arguments.validator, zip_output=arguments.zip_output)
        if arguments.verify_runtime and not verify_runtime(arguments.install_dir, arguments.consumer):
            return 1
        return 0
    except (OSError, RuntimeError, KeyError, ValueError, yaml.YAMLError) as error:
        print(f"打包失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
