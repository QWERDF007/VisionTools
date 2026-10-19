"""Collect the configured build into a validated tools ZIP and its derived index."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate key in YAML: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_yaml(path: Path):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.load(stream, Loader=UniqueKeyLoader)


def save_yaml(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def resolve_inside(root: Path, relative: str) -> Path:
    reference = Path(relative)
    if reference.is_absolute() or reference.drive or ".." in reference.parts:
        raise RuntimeError(f"Package reference must stay inside its root: {relative}")
    resolved = (root / reference).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise RuntimeError(f"Package reference escapes its root: {relative}")
    return resolved


def verify_package(install_dir: Path, validator: Path) -> bool:
    validator = Path(validator).resolve()
    if not validator.is_file():
        raise RuntimeError(f"SDK validator not found: {validator}")
    result = subprocess.run(
        [str(validator), "--package", str(Path(install_dir).resolve())],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode:
        print(result.stderr or result.stdout, file=sys.stderr)
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
    install_dir = Path(install_dir).resolve()
    project_root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    protected = [project_root, build_dir, *(project_root / name for name in ("assets", "src", "tools", "tests", "3rdparty"))]
    if install_dir.parent == install_dir or any(
        protected_path == install_dir or protected_path.is_relative_to(install_dir)
        for protected_path in protected
    ) or any(install_dir.is_relative_to(path) for path in protected[2:]):
        raise RuntimeError(f"Refusing to replace source or build directory: {install_dir}")

    metadata = load_yaml(build_dir / config / "resolved_dependencies.yaml")
    index = load_yaml(build_dir / "generated/module_index.yaml")
    if not isinstance(metadata, dict) or not isinstance(index, dict):
        raise RuntimeError("Build metadata and index must be mappings")
    if metadata.get("config") != config or metadata.get("variant") != index.get("variant"):
        raise RuntimeError("Build metadata and index configuration mismatch")
    if not index.get("variant") or index.get("schema_version") != 2:
        raise RuntimeError("Invalid build index format")
    if set(metadata.get("modules", {})) != set(index.get("modules", {})):
        raise RuntimeError("Build modules and indexed modules do not match")
    validator = Path(validator or metadata.get("validator", ""))
    if not validator.is_file():
        raise RuntimeError(f"SDK validator is required: {validator}")
    zip_output = Path(zip_output).resolve() if zip_output else install_dir.with_suffix(".zip")
    if zip_output.is_relative_to(install_dir):
        raise RuntimeError("ZIP output must be outside the collection directory")

    if install_dir.exists():
        shutil.rmtree(install_dir)
    install_dir.mkdir(parents=True)
    final_index = {
        "schema_version": 2,
        "variant": index["variant"],
        "modules": {},
        "tools": index.get("tools", {}),
        "runtime_files": [],
    }
    for module_name, module in metadata["modules"].items():
        source = Path(module["source"])
        if not source.is_file():
            raise RuntimeError(f"Missing built module: {source}")
        relative = f"bin/{source.name}"
        target = resolve_inside(install_dir, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        final_index["modules"][module_name] = {"library": relative}

    for tool_id, entry in final_index["tools"].items():
        source = resolve_inside(project_root, entry["definition"])
        declaration = load_yaml(source)
        if declaration.get("tool_id") != tool_id:
            raise RuntimeError(f"Indexed tool_id does not match declaration: {tool_id}")
        target = resolve_inside(install_dir, entry["definition"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if declaration.get("kind") in ("python_function", "python_class"):
            for adjacent in sorted(source.parent.rglob("*")):
                if not adjacent.is_file() or "__pycache__" in adjacent.parts or adjacent.suffix not in (".py", ".pyd", ".so"):
                    continue
                relative = adjacent.relative_to(project_root).as_posix()
                checked = resolve_inside(project_root, relative)
                destination = resolve_inside(install_dir, relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(checked, destination)
        for resource in declaration.get("resources", []):
            source_resource = resolve_inside(source.parent, resource)
            relative = source_resource.relative_to(project_root).as_posix()
            destination = resolve_inside(install_dir, relative)
            if source_resource.is_dir():
                for resource_file in sorted(source_resource.rglob("*")):
                    if not resource_file.is_file() or "__pycache__" in resource_file.parts or resource_file.suffix == ".pyc":
                        continue
                    checked = resolve_inside(project_root, resource_file.relative_to(project_root).as_posix())
                    target_resource = resolve_inside(install_dir, checked.relative_to(project_root).as_posix())
                    target_resource.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(checked, target_resource)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_resource, destination)

    for runtime in metadata.get("runtime_files", []):
        source = Path(runtime["source"])
        if not source.is_file():
            raise RuntimeError(f"Missing runtime dependency: {source}")
        target = resolve_inside(install_dir, runtime["destination"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        final_index["runtime_files"].append(target.relative_to(install_dir).as_posix())
    final_index["runtime_files"] = sorted(set(final_index["runtime_files"]))
    save_yaml(final_index, install_dir / "module_index.yaml")
    if not verify_package(install_dir, validator):
        raise RuntimeError("SDK package validation failed")

    zip_output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for source in sorted(install_dir.rglob("*")):
            if source.is_file():
                bundle.write(source, source.relative_to(install_dir).as_posix())
    external_index = zip_output.parent / "module_index.yaml"
    if external_index != install_dir / "module_index.yaml":
        shutil.copy2(install_dir / "module_index.yaml", external_index)
    print(f"Packaged {len(final_index['tools'])} tools: {zip_output}")
    return zip_output


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
    parser.add_argument("--consumer", type=Path, default=Path("build/tests/test_installation_consumer.exe"))
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
        print(f"Packaging failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
