#!/usr/bin/env python3
"""
Link or copy runtime dependencies (OpenCV, InferRT, CUDA) into the build directory
so executables and contract tests run cleanly without modifying system PATH.
"""

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

def resolve_dependencies(build_bin_dir: Path, infer_enabled: bool = True):
    build_bin_dir.mkdir(parents=True, exist_ok=True)
    
    # Defaults aligned with dependencies.yaml
    opencv_bin = Path(os.environ.get("OpenCV_DIR", "D:/Software/dev/opencv-4.8.0/build/x64/vc16")) / "bin"
    if not opencv_bin.exists():
        opencv_bin = Path("D:/Software/dev/opencv-4.8.0/build/x64/vc16/bin")

    inferrt_root = Path(os.environ.get("INFERRT_ROOT", "F:/Projects/InferRT/InferRT-0.0.3"))
    inferrt_bin = inferrt_root / "bin"

    cuda_bin = Path(os.environ.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")) / "bin"

    copied = 0
    # Copy OpenCV DLLs
    if opencv_bin.exists():
        for dll in opencv_bin.glob("*.dll"):
            target = build_bin_dir / dll.name
            if not target.exists() or target.stat().st_mtime < dll.stat().st_mtime:
                shutil.copy2(dll, target)
                copied += 1

    # Copy InferRT and CUDA DLLs if enabled
    if infer_enabled:
        if inferrt_bin.exists():
            for dll in inferrt_bin.glob("*.dll"):
                target = build_bin_dir / dll.name
                if not target.exists() or target.stat().st_mtime < dll.stat().st_mtime:
                    shutil.copy2(dll, target)
                    copied += 1

        if cuda_bin.exists():
            for pattern in ["cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll", "cudnn*.dll"]:
                for dll in cuda_bin.glob(pattern):
                    target = build_bin_dir / dll.name
                    if not target.exists() or target.stat().st_mtime < dll.stat().st_mtime:
                        shutil.copy2(dll, target)
                        copied += 1

    print(f"[link_dependencies] Synchronized {copied} runtime DLLs to {build_bin_dir}")

def main():
    parser = argparse.ArgumentParser(description="Synchronize runtime dependencies")
    parser.add_argument("--bin-dir", default="build/bin", help="Target build binary directory")
    parser.add_argument("--no-infer", action="store_true", help="Skip InferRT / CUDA dependencies")
    args = parser.parse_args()

    resolve_dependencies(Path(args.bin_dir), infer_enabled=not args.no_infer)

if __name__ == "__main__":
    main()
