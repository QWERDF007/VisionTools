"""Resolve authored declarations and code packages into one conflict-checked payload."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Iterator


CODE_SUFFIXES = {".py", ".pyd", ".so"}
CACHE_PARTS = {"__pycache__"}
CACHE_SUFFIXES = {".pyc"}


class PackagingError(RuntimeError):
    """An actionable packaging failure with a Chinese message."""


def _unique_members(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"重复成员：{key}")
        result[key] = value
    return result


def _reject_nonfinite(token: str):
    raise ValueError(f"非有限 JSON 数值：{token}")


def load_json(path: Path):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            return json.load(stream, object_pairs_hook=_unique_members, parse_constant=_reject_nonfinite)
    except ValueError as error:
        raise PackagingError(f"JSON 文件无效：{path}：{error}") from error


def json_bytes(data) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(data))


def relative_path(relative, label: str, *, allow_root: bool = False) -> Path:
    if not isinstance(relative, (str, Path)):
        raise PackagingError(f"{label}必须是包内相对路径")
    path = Path(relative)
    if not str(relative) or (not path.parts and not allow_root) or path.is_absolute() or path.drive or ".." in path.parts:
        raise PackagingError(f"{label}必须位于包根目录内：{relative}")
    return path


def resolve_inside(root: Path, relative, label: str = "包内引用", *, allow_root: bool = False) -> Path:
    reference = relative_path(relative, label, allow_root=allow_root)
    resolved = (Path(root) / reference).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise PackagingError(f"{label}逃逸根目录：{relative}")
    return resolved


class CopyPlan:
    """One payload write path for authored files and derived JSON bytes."""

    def __init__(self) -> None:
        self._items: dict[str, tuple[Path, Path | bytes]] = {}

    def add(self, source: Path | bytes, destination: Path) -> None:
        source = source if isinstance(source, bytes) else Path(source).resolve()
        relative = relative_path(destination, "包内目标")
        key = os.path.normcase(relative.as_posix())
        existing = self._items.get(key)
        if existing is None:
            self._items[key] = (relative, source)
            return
        previous = existing[1]
        if previous == source:
            return
        try:
            first = previous if isinstance(previous, bytes) else previous.read_bytes()
            second = source if isinstance(source, bytes) else source.read_bytes()
        except OSError as error:
            raise PackagingError(f"无法比较包内目标冲突：{key}：{error}") from error
        if first != second:
            raise PackagingError(f"包内目标路径冲突：{key} 来自不同来源")

    def items(self) -> list[tuple[Path, Path | bytes]]:
        return [entry for _, entry in sorted(self._items.items())]

    def copy_to(self, root: Path) -> None:
        for relative, source in self.items():
            target = resolve_inside(root, relative, "包内目标")
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                if isinstance(source, bytes):
                    target.write_bytes(source)
                else:
                    shutil.copy2(source, target)
            except OSError as error:
                raise PackagingError(f"复制包文件失败：{relative}：{error}") from error


def _files(directory: Path, boundary: Path, code_only: bool, ancestors: frozenset[Path] = frozenset()) -> Iterator[Path]:
    canonical = directory.resolve()
    if not canonical.is_relative_to(boundary):
        raise PackagingError(f"目录逃逸所选包根目录：{directory}")
    if canonical in ancestors:
        raise PackagingError(f"包内目录存在循环引用：{directory}")
    for path in sorted(directory.iterdir()):
        if path.name in CACHE_PARTS or path.suffix in CACHE_SUFFIXES:
            continue
        if path.is_dir():
            yield from _files(path, boundary, code_only, ancestors | {canonical})
        elif not code_only or path.suffix in CODE_SUFFIXES:
            if not path.resolve().is_relative_to(boundary):
                raise PackagingError(f"文件逃逸所选包根目录：{path}")
            if not path.is_file():
                raise PackagingError(f"包内文件不存在：{path}")
            yield path


@dataclass(frozen=True)
class ToolPayload:
    project_root: Path
    definition: Path
    source: Path
    document: dict
    origin: str

    @classmethod
    def read(cls, project_root: Path, definition) -> ToolPayload:
        relative = relative_path(definition, "工具声明")
        if len(relative.parts) < 4 or relative.parts[:2] != ("assets", "tools") or relative.parts[2] not in ("builtin", "custom"):
            raise PackagingError(f"工具声明必须位于 assets/tools/builtin 或 custom：{relative}")
        origin = relative.parts[2]
        source = resolve_inside(project_root, relative, "工具声明")
        declaration_root = resolve_inside(project_root, Path("assets/tools") / origin, "声明根")
        if not source.is_relative_to(declaration_root) or source.suffix != ".json":
            raise PackagingError(f"工具声明必须是所选来源根内的 JSON：{relative}")
        document = load_json(source)
        if not isinstance(document, dict):
            raise PackagingError(f"工具声明必须是对象：{source}")
        return cls(project_root, relative, source, document, origin)

    @property
    def destination(self) -> Path:
        return Path(*self.definition.parts[1:])

    @property
    def is_python(self) -> bool:
        return self.document.get("kind") in ("python_function", "python_class")

    @property
    def python_package(self) -> tuple[Path, Path]:
        entry = self.document.get("entry")
        if not isinstance(entry, dict):
            raise PackagingError(f"工具缺少 Python entry：{self.source}")
        code_root = resolve_inside(self.project_root, Path("python") / self.origin, "Python 代码根")
        package = relative_path(entry.get("package"), "Python 代码包", allow_root=True)
        source = resolve_inside(code_root, package, "Python 代码包", allow_root=True)
        return source, Path("python") / self.origin / package

    def validate(self) -> None:
        if self.is_python:
            package_root, _ = self.python_package
            file = relative_path(self.document["entry"].get("file"), "Python 入口脚本")
            if file.suffix != ".py" or any(not part.isidentifier() for part in file.with_suffix("").parts):
                raise PackagingError(f"Python 入口脚本的模块路径必须是标识符：{file}")
            script = resolve_inside(package_root, file, "Python 入口脚本")
            if not script.is_file():
                raise PackagingError(f"Python 入口脚本不存在：{script}")
        self.resources()

    def resources(self) -> list[tuple[Path, Path, Path]]:
        resources = self.document.get("resources", [])
        if not isinstance(resources, list):
            raise PackagingError(f"工具的 resources 必须是列表：{self.source}")
        if self.is_python:
            base, destination = self.python_package
            boundary = base
        else:
            base, destination = self.source.parent, self.destination.parent
            boundary = base
        result = []
        for resource in resources:
            if not isinstance(resource, str) or not resource:
                raise PackagingError("资源必须是非空相对路径")
            relative = Path(resource)
            if relative.is_absolute() or relative.drive:
                raise PackagingError(f"资源必须位于所选包根目录内：{resource}")
            logical = Path(os.path.normpath(base / relative))
            if not logical.resolve().is_relative_to(boundary):
                raise PackagingError(f"资源必须位于所选包根目录内：{resource}")
            if not logical.exists():
                raise PackagingError(f"资源不存在：{logical}")
            target = destination / logical.relative_to(base)
            result.append((logical, target, boundary))
        return result

    def add_to(self, plan: CopyPlan) -> None:
        if self.is_python:
            source, destination = self.python_package
            for path in _files(source, source, True):
                plan.add(path, destination / path.relative_to(source))
            # The author SDK has one source file in the repository; the packager is the only writer
            # that puts it inside a tool package, so every shipped tool imports the same code and
            # no example carries a hand-maintained copy.
            sdk = self.project_root / "python/author_sdk/visiontools_sdk"
            if not sdk.is_dir():
                raise PackagingError(f"作者 SDK 缺失：{sdk}")
            for path in _files(sdk, sdk, True):
                plan.add(path, destination / "visiontools_sdk" / path.relative_to(sdk))
        resources = self.resources()
        for source, destination, boundary in resources:
            if source.is_dir():
                for path in _files(source, boundary, False):
                    plan.add(path, destination / path.relative_to(source))
            else:
                plan.add(source, destination)
        document = dict(self.document)
        if resources and not self.is_python:
            document["resources"] = [Path(os.path.relpath(target, self.destination.parent)).as_posix()
                                     for _, target, _ in resources]
        plan.add(json_bytes(document), self.destination)
