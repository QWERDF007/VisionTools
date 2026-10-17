#!/usr/bin/env python3
"""
Unit tests for packaging and verification logic (VT-05 / VS-06).
Tests cover:
  - Missing infer DLL in GPU mode (strict raises, verify fails)
  - Missing dependency directories (OpenCV, TensorRT in GPU mode)
  - GPU -> CPU repackaging in identical directory leaves no residual GPU artifacts
  - CPU package excludes GPU tools and DLLs
  - Paths with spaces
  - Verification failures on tampered/incomplete packages
"""

import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import package_tools as pkg

class TestPackageTools(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.t = pathlib.Path(self.temp_dir.name)
        self.build_dir = self.t / "build"
        self.build_dir.mkdir(parents=True)
        self.install_dir = self.t / "dist" / "VisionTools"

        # Mock preprocess DLL
        prep_dir = self.build_dir / "src" / "preprocess"
        prep_dir.mkdir(parents=True)
        (prep_dir / "visiontools_preprocess.dll").write_bytes(b"MOCK_PREPROCESS_DLL")

        # Mock dependencies
        self.mock_deps_dir = self.t / "deps"
        self.mock_opencv = self.mock_deps_dir / "opencv" / "bin"
        self.mock_opencv.mkdir(parents=True)
        (self.mock_opencv / "opencv_world480.dll").write_bytes(b"MOCK_OPENCV")

        self.mock_inferrt = self.mock_deps_dir / "inferrt" / "bin"
        self.mock_inferrt.mkdir(parents=True)
        (self.mock_inferrt / "inferrt_core.dll").write_bytes(b"MOCK_INFERRT")

        self.mock_cuda = self.mock_deps_dir / "cuda" / "bin"
        self.mock_cuda.mkdir(parents=True)
        (self.mock_cuda / "cudart64_110.dll").write_bytes(b"MOCK_CUDA")

        self.mock_trt = self.mock_deps_dir / "trt" / "bin"
        self.mock_trt.mkdir(parents=True)
        (self.mock_trt / "nvinfer_10.dll").write_bytes(b"MOCK_TRT")

        # Write resolved_dependencies.json
        self._write_resolved_deps(is_gpu=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_resolved_deps(self, is_gpu=True):
        deps = {
            "version": "0.1.0",
            "build_infer": is_gpu,
            "generator_variant": "gpu" if is_gpu else "cpu",
            "opencv_dir": str(self.mock_opencv.parent).replace("\\", "/"),
            "visionrt_sdk_dir": "C:/dev/visionrt-sdk",
        }
        if is_gpu:
            deps["inferrt_root"] = str(self.mock_inferrt.parent).replace("\\", "/")
            deps["cuda_root"] = str(self.mock_cuda.parent).replace("\\", "/")
            deps["tensorrt_root"] = str(self.mock_trt.parent).replace("\\", "/")

        with open(self.build_dir / "resolved_dependencies.json", "w", encoding="utf-8") as f:
            json.dump(deps, f, indent=2)

    def test_missing_preprocess_dll_raises(self):
        empty_build = self.t / "empty_build"
        empty_build.mkdir()
        with self.assertRaises(RuntimeError):
            pkg.package_tools(empty_build, self.install_dir, infer_enabled=False, strict=True)

    def test_missing_infer_dll_gpu_strict_raises(self):
        # build_dir does not have visiontools_infer.dll
        with self.assertRaises(RuntimeError) as ctx:
            pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=True, strict=True)
        self.assertIn("visiontools_infer.dll", str(ctx.exception))

    def test_missing_infer_dll_gpu_verify_fails(self):
        # Packaging without strict should not copy infer DLL, but verify MUST fail
        pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=True, strict=False)
        self.assertFalse((self.install_dir / "bin" / "visiontools_infer.dll").exists())
        self.assertFalse(pkg.verify_package(self.install_dir, is_gpu=True))

    def test_valid_gpu_package(self):
        infer_dir = self.build_dir / "src" / "infer"
        infer_dir.mkdir(parents=True)
        (infer_dir / "visiontools_infer.dll").write_bytes(b"MOCK_INFER_DLL")

        pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=True, strict=True)
        self.assertTrue((self.install_dir / "bin" / "visiontools_preprocess.dll").exists())
        self.assertTrue((self.install_dir / "bin" / "visiontools_infer.dll").exists())
        self.assertTrue(list(self.install_dir.rglob("visionrt.dl.yolo.yaml")))
        self.assertTrue(pkg.verify_package(self.install_dir, is_gpu=True))

    def test_valid_cpu_package(self):
        pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)
        self.assertTrue((self.install_dir / "bin" / "visiontools_preprocess.dll").exists())
        self.assertFalse((self.install_dir / "bin" / "visiontools_infer.dll").exists())
        self.assertFalse(bool(list(self.install_dir.rglob("visionrt.dl.yolo.yaml"))))
        self.assertTrue(pkg.verify_package(self.install_dir, is_gpu=False))

    def test_gpu_then_cpu_same_directory_no_leak(self):
        # 1. Package GPU
        infer_dir = self.build_dir / "src" / "infer"
        infer_dir.mkdir(parents=True)
        (infer_dir / "visiontools_infer.dll").write_bytes(b"MOCK_INFER_DLL")
        pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=True, strict=True)
        self.assertTrue((self.install_dir / "bin" / "visiontools_infer.dll").exists())
        self.assertTrue(bool(list(self.install_dir.rglob("visionrt.dl.yolo.yaml"))))

        # 2. Package CPU into the EXACT SAME directory
        self._write_resolved_deps(is_gpu=False)
        pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)

        # 3. Assert all GPU artifacts are gone
        self.assertFalse((self.install_dir / "bin" / "visiontools_infer.dll").exists())
        self.assertFalse(bool(list(self.install_dir.rglob("visionrt.dl.yolo.yaml"))))
        self.assertFalse((self.install_dir / "bin" / "inferrt_core.dll").exists())
        self.assertFalse((self.install_dir / "bin" / "cudart64_110.dll").exists())
        self.assertFalse((self.install_dir / "bin" / "nvinfer_10.dll").exists())
        self.assertTrue(pkg.verify_package(self.install_dir, is_gpu=False))

    def test_verify_fails_if_cpu_package_contains_gpu_artifacts(self):
        pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)
        # Poison CPU package with infer DLL
        (self.install_dir / "bin" / "visiontools_infer.dll").write_bytes(b"LEAKED_DLL")
        self.assertFalse(pkg.verify_package(self.install_dir, is_gpu=False))

    def test_verify_fails_if_manifest_file_missing(self):
        pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)
        # Delete an asset from package
        prep_yaml = next(self.install_dir.rglob("visionrt.image.crop.yaml"))
        prep_yaml.unlink()
        self.assertFalse(pkg.verify_package(self.install_dir, is_gpu=False))

    def test_path_with_spaces(self):
        spaced_dir = self.t / "dist with spaces" / "Vision Tools"
        pkg.package_tools(self.build_dir, spaced_dir, infer_enabled=False, strict=True)
        self.assertTrue((spaced_dir / "bin" / "visiontools_preprocess.dll").exists())
        self.assertTrue(pkg.verify_package(spaced_dir, is_gpu=False))

if __name__ == "__main__":
    unittest.main()
