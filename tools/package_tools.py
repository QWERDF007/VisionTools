#!/usr/bin/env python3
"""
Packaging and distribution tool for VisionTools.
Implements spec TS-07 and ticket TL-06.

Produces clean installation tree containing:
  - Domain DLLs (bin/visiontools_*.dll)
  - Tool YAML declarations (assets/tools/builtin/, assets/tools/custom/)
  - Module index (module_index.json)
  - Backend runtime dependencies (OpenCV DLLs, InferRT/CUDA DLLs if GPU enabled)
  - Plain relative file manifest (tools_manifest.txt)
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Set

def collect_files(source_dir: Path, target_dir: Path, patterns: List[str]) -> List[Path]:
    copied = []
    target_dir.mkdir(parents=True, exist_ok=True)
    for pat in patterns:
        for f in source_dir.glob(pat):
            if f.is_file():
                dest = target_dir / f.name
                shutil.copy2(f, dest)
                copied.append(dest)
    return copied

def package_tools(build_dir: Path, install_dir: Path, config: str = "Release", infer_enabled: bool = True):
    install_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = install_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    manifest_files = []

    # 1. Copy Domain DLLs from build directory
    dll_candidates = [
        build_dir / "src" / "preprocess" / "visiontools_preprocess.dll",
        build_dir / "src" / "preprocess" / config / "visiontools_preprocess.dll",
        build_dir / "bin" / "visiontools_preprocess.dll",
    ]
    preprocess_dll = next((p for p in dll_candidates if p.exists()), None)
    if not preprocess_dll:
        raise RuntimeError(f"Could not find visiontools_preprocess.dll in {build_dir}")

    shutil.copy2(preprocess_dll, bin_dir / "visiontools_preprocess.dll")
    manifest_files.append(Path("bin") / "visiontools_preprocess.dll")

    if infer_enabled:
        infer_candidates = [
            build_dir / "src" / "infer" / "visiontools_infer.dll",
            build_dir / "src" / "infer" / config / "visiontools_infer.dll",
            build_dir / "bin" / "visiontools_infer.dll",
        ]
        infer_dll = next((p for p in infer_candidates if p.exists()), None)
        if infer_dll:
            shutil.copy2(infer_dll, bin_dir / "visiontools_infer.dll")
            manifest_files.append(Path("bin") / "visiontools_infer.dll")

    # 2. Copy assets/tools
    src_assets = Path("assets/tools")
    dst_assets = install_dir / "assets" / "tools"
    if src_assets.exists():
        if dst_assets.exists():
            shutil.rmtree(dst_assets)
        shutil.copytree(src_assets, dst_assets)
        for root, _, files in os.walk(dst_assets):
            for file in files:
                rel = Path(root).relative_to(install_dir) / file
                manifest_files.append(rel)

    # 3. Generate and copy module_index.json
    index_src = Path("module_index.json")
    if not index_src.exists():
        # Fallback generated index
        index_data = {
            "version": "1.0.0",
            "variant": "gpu" if infer_enabled else "cpu",
            "modules": {
                "visiontools_preprocess": "bin/visiontools_preprocess.dll",
                "visiontools_infer": "bin/visiontools_infer.dll" if infer_enabled else None
            }
        }
        with open(install_dir / "module_index.json", "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
    else:
        shutil.copy2(index_src, install_dir / "module_index.json")
    manifest_files.append(Path("module_index.json"))

    # 4. Copy runtime dependency DLLs
    opencv_bin = Path(os.environ.get("OpenCV_DIR", "D:/Software/dev/opencv-4.8.0/build/x64/vc16")) / "bin"
    if not opencv_bin.exists():
        opencv_bin = Path("D:/Software/dev/opencv-4.8.0/build/x64/vc16/bin")

    if opencv_bin.exists():
        for dll in opencv_bin.glob("*.dll"):
            dest = bin_dir / dll.name
            shutil.copy2(dll, dest)
            manifest_files.append(Path("bin") / dll.name)

    if infer_enabled:
        inferrt_root = Path(os.environ.get("INFERRT_ROOT", "F:/Projects/InferRT/InferRT-0.0.3"))
        inferrt_bin = inferrt_root / "bin"
        if inferrt_bin.exists():
            for dll in inferrt_bin.glob("*.dll"):
                dest = bin_dir / dll.name
                shutil.copy2(dll, dest)
                manifest_files.append(Path("bin") / dll.name)

        cuda_bin = Path(os.environ.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")) / "bin"
        if cuda_bin.exists():
            for pat in ["cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll", "cudnn*.dll"]:
                for dll in cuda_bin.glob(pat):
                    dest = bin_dir / dll.name
                    shutil.copy2(dll, dest)
                    manifest_files.append(Path("bin") / dll.name)

    # 5. Write plain relative tools_manifest.txt
    manifest_path = install_dir / "tools_manifest.txt"
    manifest_lines = [
        f"# VisionTools distribution manifest",
        f"# variant: {'gpu' if infer_enabled else 'cpu'}",
        ""
    ]
    for rel_path in sorted(manifest_files):
        manifest_lines.append(str(rel_path).replace("\\", "/"))

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines) + "\n")

    print(f"[package_tools] Successfully packaged {len(manifest_files)} items to {install_dir}")
    print(f"[package_tools] Manifest generated at {manifest_path}")

def verify_package(install_dir: Path, is_gpu: bool = False) -> bool:
    manifest_path = install_dir / "tools_manifest.txt"
    if not manifest_path.exists():
        print(f"[ERROR] tools_manifest.txt not found in {install_dir}")
        return False

    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    missing = []
    for rel_path in lines:
        target = install_dir / rel_path
        if not target.exists():
            missing.append(rel_path)

    if missing:
        print(f"[ERROR] Package verification failed. Missing {len(missing)} files: {missing}")
        return False

    print(f"[SUCCESS] Package verification passed ({len(lines)} files verified).")
    return True

def main():
    parser = argparse.ArgumentParser(description="Package VisionTools distribution tree")
    parser.add_argument("--build-dir", default="build", help="Build directory")
    parser.add_argument("--install-dir", default="dist/VisionTools", help="Target installation directory")
    parser.add_argument("--config", default="Release", help="Build configuration (Release/Debug)")
    parser.add_argument("--no-infer", action="store_true", help="Package CPU-only tools")
    parser.add_argument("--verify", action="store_true", help="Verify existing package")
    args = parser.parse_args()

    install_dir = Path(args.install_dir)
    if args.verify:
        ok = verify_package(install_dir, is_gpu=not args.no_infer)
        sys.exit(0 if ok else 1)

    package_tools(Path(args.build_dir), install_dir, config=args.config, infer_enabled=not args.no_infer)

if __name__ == "__main__":
    main()
