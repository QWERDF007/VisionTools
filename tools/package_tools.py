#!/usr/bin/env python3
"""
Packaging and distribution tool for VisionTools (YS-01 / YS-03).
Single Source of Truth:
- Consumes build/generated/module_index.yaml and build/<config>/resolved_dependencies.yaml.
- Direct destructive rebuild: clears output_dir before packaging; no staging, backup, or rollback.
- Generates final output_dir/module_index.yaml containing variant, modules, tools, and runtime_files.
- Completely deletes tools_manifest.txt and legacy JSON metadata.
- Package verification inspects module_index.yaml, variant closure, indexed library existence,
  set equality of declared YAML tools vs indexed tools, and runtime dependency DLLs.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import yaml

def load_yaml(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)

def find_resolved_dependencies(build_dir: Path, config: Optional[str] = None) -> Optional[Path]:
    """Find resolved_dependencies.yaml in build/<config> or build root."""
    if config:
        candidate = build_dir / config / "resolved_dependencies.yaml"
        if candidate.exists():
            return candidate
    candidate = build_dir / "resolved_dependencies.yaml"
    if candidate.exists():
        return candidate
    return None

def find_build_module_index(build_dir: Path) -> Optional[Path]:
    """Find build module_index.yaml in build/generated or build root."""
    candidate = build_dir / "generated" / "module_index.yaml"
    if candidate.exists():
        return candidate
    candidate = build_dir / "module_index.yaml"
    if candidate.exists():
        return candidate
    return None

def package_tools(
    build_dir: Path,
    install_dir: Path,
    config: str = "Release",
    infer_enabled: bool = True,
    strict: bool = False,
    project_root: Optional[Path] = None
) -> None:
    """
    Direct destructive rebuild of the installation directory (YS-03).
    Clears install_dir immediately upon starting; generates artifacts directly.
    Fails non-zero without rolling back or restoring old package.
    """
    build_dir = Path(build_dir).resolve()
    install_dir = Path(install_dir).resolve()
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    if install_dir == project_root or install_dir == Path.cwd().resolve() or install_dir in project_root.parents:
        raise RuntimeError(f"Refusing to package into project root or invalid install_dir: {install_dir}")

    target_variant = "gpu" if infer_enabled else "cpu"

    # 1. Read build artifacts metadata: resolved_dependencies.yaml
    resolved_file = find_resolved_dependencies(build_dir, config)
    if not resolved_file:
        if strict:
            raise RuntimeError(f"Strict packaging requires build metadata: resolved_dependencies.yaml in {build_dir}")
        deps_meta = {}
    else:
        try:
            deps_meta = load_yaml(resolved_file) or {}
            if not isinstance(deps_meta, dict):
                raise RuntimeError(f"Invalid YAML structure in {resolved_file}")
            if strict:
                required_fields = ["variant", "config", "modules"]
                for rf in required_fields:
                    if rf not in deps_meta:
                        raise RuntimeError(f"Missing required build metadata field '{rf}' in {resolved_file}")
            if "variant" in deps_meta and deps_meta["variant"] != target_variant:
                raise RuntimeError(f"Build metadata variant mismatch: resolved_dependencies.yaml has '{deps_meta['variant']}', packaging target is '{target_variant}'")
            if config is not None and "config" in deps_meta and deps_meta["config"] != config:
                raise RuntimeError(f"Build metadata config mismatch: resolved_dependencies.yaml has config='{deps_meta['config']}', packaging target is '{config}'")
        except RuntimeError:
            raise
        except Exception as e:
            if strict:
                raise RuntimeError(f"Corrupt build metadata {resolved_file}: {e}")
            deps_meta = {}

    # 2. Read build module index: module_index.yaml
    index_src = find_build_module_index(build_dir)
    if not index_src:
        if strict:
            raise RuntimeError(f"Missing required build index: module_index.yaml in {build_dir}")
        build_index = {}
    else:
        try:
            build_index = load_yaml(index_src) or {}
            if not isinstance(build_index, dict):
                raise RuntimeError(f"Invalid YAML structure in {index_src}")
            if "variant" in build_index and build_index["variant"] != target_variant:
                raise RuntimeError(f"Build index variant mismatch: module_index.yaml has '{build_index['variant']}', packaging target is '{target_variant}'")
            if "tools" in build_index and not isinstance(build_index["tools"], dict):
                raise RuntimeError(f"Invalid 'tools' section in {index_src}")
        except RuntimeError:
            raise
        except Exception as e:
            if strict:
                raise RuntimeError(f"Corrupt build index {index_src}: {e}")
            build_index = {}

    # 3. Direct destructive clean of install_dir (YS-03)
    # Delete output_dir recursively if it exists; no staging or backup
    if install_dir.exists():
        shutil.rmtree(install_dir)
    install_dir.mkdir(parents=True, exist_ok=True)

    bin_dir = install_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = install_dir / "assets" / "tools"
    assets_dir.mkdir(parents=True, exist_ok=True)

    installed_modules: Dict[str, Dict[str, str]] = {}
    installed_tools: Dict[str, Dict[str, str]] = {}
    installed_runtime_files: List[str] = []

    # 4. Copy module binaries
    modules_dict = deps_meta.get("modules", {})
    if not modules_dict and build_index.get("modules"):
        modules_dict = build_index["modules"]

    # Fallback default module if not listed
    if not modules_dict:
        modules_dict = {"visiontools_preprocess": {}}
        if infer_enabled:
            modules_dict["visiontools_infer"] = {}

    for mod_name, mod_info in modules_dict.items():
        if not infer_enabled and "infer" in mod_name:
            continue
        dll_name = f"{mod_name}.dll"
        dest_lib = bin_dir / dll_name

        src_path = None
        if isinstance(mod_info, dict) and "source" in mod_info:
            src_path = Path(mod_info["source"])
            if not src_path.exists():
                raise RuntimeError(f"Missing required module source for '{mod_name}': {src_path}")
        if not src_path:
            # Search build_dir candidates
            candidates = [
                build_dir / "src" / mod_name.replace("visiontools_", "") / dll_name,
                build_dir / "src" / mod_name / dll_name,
                build_dir / "bin" / dll_name,
                build_dir / dll_name,
                build_dir / config / dll_name
            ]
            for c in candidates:
                if c.exists():
                    src_path = c
                    break

        if not src_path or not src_path.exists():
            raise RuntimeError(f"Missing required module DLL for '{mod_name}': {src_path or dll_name}")

        shutil.copy2(src_path, dest_lib)
        installed_modules[mod_name] = {"library": f"bin/{dll_name}"}

    # 5. Copy tool declarations
    tools_dict = build_index.get("tools", {})
    src_tools_dir = project_root / "assets" / "tools"
    if src_tools_dir.exists():
        for yf in list(src_tools_dir.glob("**/*.yaml")) + list(src_tools_dir.glob("**/*.yml")):
            rel = yf.relative_to(src_tools_dir)
            if any(part in rel.parts for part in ("custom", "示例", "examples")):
                continue
            if not infer_enabled and "dl.yolo" in yf.name:
                continue

            # Read registry
            ydoc = load_yaml(yf) or {}
            reg = ydoc.get("entry", {}).get("registry_name") or ydoc.get("tool_id")
            if not reg:
                continue

            dest_file = assets_dir / rel
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(yf, dest_file)

            # Map to module
            mod = "visiontools_preprocess"
            if "dl.yolo" in reg or "infer" in ydoc.get("category", "").lower():
                mod = "visiontools_infer"
            if reg in tools_dict and isinstance(tools_dict[reg], dict) and tools_dict[reg].get("module"):
                mod = tools_dict[reg]["module"]

            rel_pkg_path = f"assets/tools/{rel.as_posix()}"
            installed_tools[reg] = {
                "module": mod,
                "definition": rel_pkg_path
            }

    # 6. Copy runtime dependency files
    runtime_entries = deps_meta.get("runtime_files", [])
    for rf in runtime_entries:
        if isinstance(rf, dict) and "source" in rf and "destination" in rf:
            sp = Path(rf["source"])
            dp = install_dir / rf["destination"]
            if sp.exists():
                dp.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sp, dp)
                installed_runtime_files.append(Path(rf["destination"]).as_posix())
            elif strict:
                raise RuntimeError(f"Missing required runtime dependency file: {sp}")

    # Fallback dependency discovery if runtime_entries is empty or incomplete
    if not any("opencv" in r.lower() for r in installed_runtime_files):
        # Look for OpenCV DLL
        opencv_bins = [
            Path(os.environ.get("OpenCV_DIR", "D:/Software/dev/opencv-4.8.0/build/x64/vc16")) / "bin",
            Path("D:/Software/dev/opencv-4.8.0/build/x64/vc16/bin")
        ]
        for ob in opencv_bins:
            if ob.exists():
                for dll in ob.glob("*.dll"):
                    dest = bin_dir / dll.name
                    shutil.copy2(dll, dest)
                    installed_runtime_files.append(f"bin/{dll.name}")
                break

    if infer_enabled:
        # InferRT
        if not any("inferrt" in r.lower() for r in installed_runtime_files):
            inf_bin = Path(os.environ.get("INFERRT_ROOT", "F:/Projects/InferRT/InferRT-0.0.3")) / "bin"
            if inf_bin.exists():
                for dll in inf_bin.glob("*.dll"):
                    dest = bin_dir / dll.name
                    shutil.copy2(dll, dest)
                    installed_runtime_files.append(f"bin/{dll.name}")
        # CUDA
        if not any("cudart" in r.lower() for r in installed_runtime_files):
            cuda_bin = Path(os.environ.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")) / "bin"
            if cuda_bin.exists():
                for pat in ["cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll", "cudnn*.dll"]:
                    for dll in cuda_bin.glob(pat):
                        dest = bin_dir / dll.name
                        shutil.copy2(dll, dest)
                        installed_runtime_files.append(f"bin/{dll.name}")
        # TensorRT
        if not any("nvinfer" in r.lower() for r in installed_runtime_files):
            trt_candidates = [
                Path(os.environ.get("TRT_ROOT", "D:/Software/dev/TensorRT-10.16.1.11")) / "bin",
                Path(os.environ.get("TRT_ROOT", "D:/Software/dev/TensorRT-10.16.1.11")) / "lib"
            ]
            for tc in trt_candidates:
                if tc.exists():
                    for pat in ["nvinfer*.dll", "nvonnxparser*.dll"]:
                        for dll in tc.glob(pat):
                            dest = bin_dir / dll.name
                            shutil.copy2(dll, dest)
                            installed_runtime_files.append(f"bin/{dll.name}")
                    break

    # Also copy yaml-cpp DLL if present
    if not any("yaml-cpp" in r.lower() for r in installed_runtime_files):
        yaml_dll = Path("D:/Software/anaconda3/Library/bin/yaml-cpp.dll")
        if yaml_dll.exists():
            dest = bin_dir / yaml_dll.name
            shutil.copy2(yaml_dll, dest)
            installed_runtime_files.append(f"bin/{yaml_dll.name}")

    # Deduplicate runtime files
    installed_runtime_files = sorted(set(installed_runtime_files))

    # 7. Copy examples if present
    custom_dir = project_root / "assets" / "tools" / "custom"
    if custom_dir.exists():
        ex_dir = install_dir / "examples" / "custom"
        ex_dir.parent.mkdir(parents=True, exist_ok=True)
        if ex_dir.exists():
            shutil.rmtree(ex_dir)
        shutil.copytree(custom_dir, ex_dir)

    # 8. Write final output_dir/module_index.yaml (YS-01)
    final_index = {
        "variant": target_variant,
        "modules": installed_modules,
        "tools": installed_tools,
        "runtime_files": installed_runtime_files
    }
    final_index_path = install_dir / "module_index.yaml"
    save_yaml(final_index, final_index_path)

    # 9. Verify package directly (fails without rollback, leaving current output)
    if not verify_package(install_dir, is_gpu=infer_enabled):
        raise RuntimeError(f"Package verification failed on install directory {install_dir}")

    print(f"[package_tools] Successfully packaged {len(installed_tools)} tools to {install_dir}")
    print(f"[package_tools] Final index generated at {final_index_path}")

def verify_package(install_dir: Path, is_gpu: bool = False) -> bool:
    """
    Verify packaging completeness, variant closure, indexed library existence, and runtime dependencies (YS-01 / YS-03).
    """
    install_dir = Path(install_dir).resolve()
    index_path = install_dir / "module_index.yaml"
    if not index_path.exists():
        print(f"[ERROR] module_index.yaml not found in {install_dir}")
        return False

    try:
        idx = load_yaml(index_path)
    except Exception as e:
        print(f"[ERROR] Failed to parse module_index.yaml: {e}")
        return False

    if not isinstance(idx, dict):
        print("[ERROR] module_index.yaml must be a dictionary")
        return False

    expected_variant = "gpu" if is_gpu else "cpu"
    if idx.get("variant") != expected_variant:
        print(f"[ERROR] module_index.yaml variant mismatch: expected '{expected_variant}', got '{idx.get('variant')}'")
        return False

    modules_dict = idx.get("modules")
    if not isinstance(modules_dict, dict) or not modules_dict:
        print("[ERROR] module_index.yaml 'modules' must be a non-empty dictionary")
        return False

    for mod_name, mod_info in modules_dict.items():
        if not isinstance(mod_info, dict) or "library" not in mod_info:
            print(f"[ERROR] Module '{mod_name}' missing 'library' path")
            return False
        lib_path = install_dir / mod_info["library"]
        if not lib_path.exists():
            print(f"[ERROR] Indexed module library not found: {lib_path}")
            return False

    tools_dict = idx.get("tools")
    if not isinstance(tools_dict, dict) or not tools_dict:
        print("[ERROR] module_index.yaml 'tools' must be a non-empty dictionary")
        return False

    # Collect declared tool registries from packaged YAML files
    yaml_files = list(install_dir.glob("assets/tools/**/*.yaml")) + list(install_dir.glob("assets/tools/**/*.yml"))
    if not yaml_files:
        print("[ERROR] No tool declaration YAML files found in assets/tools")
        return False

    declared_registries = set()
    for yf in yaml_files:
        try:
            yd = load_yaml(yf) or {}
            r = yd.get("entry", {}).get("registry_name") or yd.get("tool_id")
            if r:
                declared_registries.add(r)
        except Exception as e:
            print(f"[ERROR] Failed to parse tool declaration {yf}: {e}")
            return False

    index_registries = set(tools_dict.keys())
    if index_registries != declared_registries:
        missing_in_index = declared_registries - index_registries
        extra_in_index = index_registries - declared_registries
        print(f"[ERROR] module_index.yaml registry set does not match declared tool definitions: missing_in_index={missing_in_index}, extra_in_index={extra_in_index}")
        return False

    for reg, tinfo in tools_dict.items():
        if not isinstance(tinfo, dict):
            print(f"[ERROR] Tool '{reg}' entry must be a dictionary")
            return False
        mod_ref = tinfo.get("module")
        if not mod_ref or mod_ref not in modules_dict:
            print(f"[ERROR] Tool '{reg}' references unknown module '{mod_ref}'")
            return False
        def_ref = tinfo.get("definition")
        if not def_ref or not (install_dir / def_ref).exists():
            print(f"[ERROR] Tool '{reg}' definition file not found: {install_dir / str(def_ref)}")
            return False

        # If declaration defines python entry.file, verify it exists relative to the definition
        decl_path = install_dir / def_ref
        try:
            decl_data = load_yaml(decl_path) or {}
            py_entry = decl_data.get("entry", {}).get("file")
            if py_entry:
                py_file = decl_path.parent / py_entry
                if not py_file.exists():
                    print(f"[ERROR] Python tool '{reg}' script file not found: {py_file}")
                    return False
        except Exception:
            pass

    # Verify runtime_files
    runtime_files = idx.get("runtime_files", [])
    if not isinstance(runtime_files, list):
        print("[ERROR] 'runtime_files' must be a list")
        return False
    for rf in runtime_files:
        rf_path = install_dir / rf
        if not rf_path.exists():
            print(f"[ERROR] Runtime file not found: {rf_path}")
            return False

    # CPU/GPU isolation checks
    infer_dll = install_dir / "bin" / "visiontools_infer.dll"
    if not is_gpu:
        if infer_dll.exists():
            print("[ERROR] CPU package verification failed: visiontools_infer.dll found in CPU package")
            return False
        if any("infer" in m.lower() for m in modules_dict):
            print("[ERROR] CPU package verification failed: infer module referenced in CPU package")
            return False
    else:
        if not infer_dll.exists():
            print("[ERROR] GPU package verification failed: visiontools_infer.dll missing")
            return False
        # TensorRT core runtime check
        bin_dir = install_dir / "bin"
        found_trt = False
        if bin_dir.exists():
            for f in bin_dir.glob("nvinfer*.dll"):
                lower = f.name.lower()
                if "plugin" in lower or "dispatch" in lower or "lean" in lower:
                    continue
                found_trt = True
                break
        if not found_trt:
            for rf in runtime_files:
                name = Path(rf).name.lower()
                if name.startswith("nvinfer") and not any(k in name for k in ["plugin", "dispatch", "lean"]):
                    found_trt = True
                    break
        if not found_trt:
            print("[ERROR] GPU package verification failed: no TensorRT core runtime DLL found in package")
            return False

    print(f"[SUCCESS] Package verification passed ({len(tools_dict)} tools verified, variant={expected_variant}).")
    return True

def main():
    parser = argparse.ArgumentParser(description="Package VisionTools for distribution")
    parser.add_argument("--build-dir", default="build", help="Path to CMake build directory")
    parser.add_argument("--install-dir", default="dist/VisionTools", help="Destination directory")
    parser.add_argument("--config", default="Release", help="Build configuration (Release, Debug)")
    parser.add_argument("--no-infer", action="store_true", help="Package CPU-only tools (exclude GPU)")
    parser.add_argument("--verify-only", action="store_true", help="Run verification without rebuilding")
    args = parser.parse_args()

    install_dir = Path(args.install_dir)
    is_gpu = not args.no_infer

    if args.verify_only:
        ok = verify_package(install_dir, is_gpu=is_gpu)
        sys.exit(0 if ok else 1)

    try:
        package_tools(
            build_dir=Path(args.build_dir),
            install_dir=install_dir,
            config=args.config,
            infer_enabled=is_gpu,
            strict=True
        )
    except Exception as e:
        print(f"[ERROR] Packaging failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
