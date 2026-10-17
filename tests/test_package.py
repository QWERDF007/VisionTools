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

        # Write resolved_dependencies.json and module_index.json
        self._write_resolved_deps(is_gpu=False)
        self._write_module_index(is_gpu=False)

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

    def _write_module_index(self, is_gpu=True):
        tools = {
            "visionrt.image.crop": {
                "kind": "cpp_function",
                "module_dll": "bin/visiontools_preprocess.dll"
            }
        }
        if is_gpu:
            tools["visionrt.dl.yolo"] = {
                "kind": "cpp_function",
                "module_dll": "bin/visiontools_infer.dll"
            }
        idx = {
            "variant": "gpu" if is_gpu else "cpu",
            "tools": tools
        }
        with open(self.build_dir / "module_index.json", "w", encoding="utf-8") as f:
            json.dump(idx, f, indent=2)

    def test_missing_preprocess_dll_raises(self):
        empty_build = self.t / "empty_build"
        empty_build.mkdir()
        with self.assertRaises(RuntimeError):
            pkg.package_tools(empty_build, self.install_dir, infer_enabled=False, strict=True)

    def test_missing_infer_dll_gpu_strict_raises(self):
        # build_dir does not have visiontools_infer.dll
        self._write_resolved_deps(is_gpu=True)
        self._write_module_index(is_gpu=True)
        with self.assertRaises(RuntimeError) as ctx:
            pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=True, strict=True)
        self.assertIn("visiontools_infer.dll", str(ctx.exception))

    def test_missing_infer_dll_gpu_verify_fails(self):
        # Packaging without strict should not copy infer DLL, but stage verification MUST fail
        self._write_resolved_deps(is_gpu=True)
        self._write_module_index(is_gpu=True)
        with self.assertRaises(RuntimeError):
            pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=True, strict=False)
        self.assertFalse((self.install_dir / "bin" / "visiontools_infer.dll").exists())
        self.assertFalse(pkg.verify_package(self.install_dir, is_gpu=True))

    def test_valid_gpu_package(self):
        infer_dir = self.build_dir / "src" / "infer"
        infer_dir.mkdir(parents=True)
        (infer_dir / "visiontools_infer.dll").write_bytes(b"MOCK_INFER_DLL")
        self._write_resolved_deps(is_gpu=True)
        self._write_module_index(is_gpu=True)

        pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=True, strict=True)
        self.assertTrue((self.install_dir / "bin" / "visiontools_preprocess.dll").exists())
        self.assertTrue((self.install_dir / "bin" / "visiontools_infer.dll").exists())
        self.assertTrue(list(self.install_dir.rglob("visionrt.dl.yolo.yaml")))
        self.assertTrue(pkg.verify_package(self.install_dir, is_gpu=True))

    def test_valid_cpu_package(self):
        self._write_resolved_deps(is_gpu=False)
        self._write_module_index(is_gpu=False)
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
        self._write_resolved_deps(is_gpu=True)
        self._write_module_index(is_gpu=True)
        pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=True, strict=True)
        self.assertTrue((self.install_dir / "bin" / "visiontools_infer.dll").exists())
        self.assertTrue(bool(list(self.install_dir.rglob("visionrt.dl.yolo.yaml"))))

        # 2. Package CPU into the EXACT SAME directory
        self._write_resolved_deps(is_gpu=False)
        self._write_module_index(is_gpu=False)
        pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)

        # 3. Assert all GPU artifacts are gone
        self.assertFalse((self.install_dir / "bin" / "visiontools_infer.dll").exists())
        self.assertFalse(bool(list(self.install_dir.rglob("visionrt.dl.yolo.yaml"))))
        self.assertFalse((self.install_dir / "bin" / "inferrt_core.dll").exists())
        self.assertFalse((self.install_dir / "bin" / "cudart64_110.dll").exists())
        self.assertFalse((self.install_dir / "bin" / "nvinfer_10.dll").exists())
        self.assertTrue(pkg.verify_package(self.install_dir, is_gpu=False))

    def test_verify_fails_if_cpu_package_contains_gpu_artifacts(self):
        self._write_resolved_deps(is_gpu=False)
        self._write_module_index(is_gpu=False)
        pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)
        # Poison CPU package with infer DLL
        (self.install_dir / "bin" / "visiontools_infer.dll").write_bytes(b"LEAKED_DLL")
        self.assertFalse(pkg.verify_package(self.install_dir, is_gpu=False))

    def test_verify_fails_if_manifest_file_missing(self):
        self._write_resolved_deps(is_gpu=False)
        self._write_module_index(is_gpu=False)
        pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)
        # Delete an asset from package
        prep_yaml = next(self.install_dir.rglob("visionrt.image.crop.yaml"))
        prep_yaml.unlink()
        self.assertFalse(pkg.verify_package(self.install_dir, is_gpu=False))

    def test_path_with_spaces(self):
        self._write_resolved_deps(is_gpu=False)
        self._write_module_index(is_gpu=False)
        spaced_dir = self.t / "dist with spaces" / "Vision Tools"
        pkg.package_tools(self.build_dir, spaced_dir, infer_enabled=False, strict=True)
        self.assertTrue((spaced_dir / "bin" / "visiontools_preprocess.dll").exists())
        self.assertTrue(pkg.verify_package(spaced_dir, is_gpu=False))

    def test_missing_build_index_raises(self):
        (self.build_dir / "module_index.json").unlink()
        self._write_resolved_deps(is_gpu=False)
        with self.assertRaises(RuntimeError) as ctx:
            pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)
        self.assertIn("Missing required build index", str(ctx.exception))

    def test_corrupt_build_index_raises(self):
        (self.build_dir / "module_index.json").write_text("{corrupt json")
        self._write_resolved_deps(is_gpu=False)
        with self.assertRaises(RuntimeError) as ctx:
            pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)
        self.assertIn("Corrupt build index", str(ctx.exception))

    def test_wrong_variant_build_index_raises(self):
        self._write_module_index(is_gpu=True)
        self._write_resolved_deps(is_gpu=False)
        with self.assertRaises(RuntimeError) as ctx:
            pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)
        self.assertIn("Build index variant mismatch", str(ctx.exception))

    def test_missing_resolved_deps_strict_raises(self):
        (self.build_dir / "resolved_dependencies.json").unlink()
        with self.assertRaises(RuntimeError) as ctx:
            pkg.load_dependencies(self.build_dir, ROOT, strict=True)
        self.assertIn("missing required build metadata", str(ctx.exception))

    def test_verify_fails_empty_tools(self):
        self._write_resolved_deps(is_gpu=False)
        self._write_module_index(is_gpu=False)
        pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)
        (self.install_dir / "module_index.json").write_text(json.dumps({"variant": "cpu", "tools": {}}))
        self.assertFalse(pkg.verify_package(self.install_dir, is_gpu=False))

    def test_verify_fails_bad_tool_entry(self):
        self._write_resolved_deps(is_gpu=False)
        self._write_module_index(is_gpu=False)
        pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)
        (self.install_dir / "module_index.json").write_text(json.dumps({"variant": "cpu", "tools": {"bad": 123}}))
        self.assertFalse(pkg.verify_package(self.install_dir, is_gpu=False))

    def test_verify_fails_escaping_module_dll(self):
        self._write_resolved_deps(is_gpu=False)
        self._write_module_index(is_gpu=False)
        pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)
        (self.install_dir / "module_index.json").write_text(json.dumps({
            "variant": "cpu",
            "tools": {"tool": {"kind": "cpp_function", "module_dll": "../escape.dll"}}
        }))
        self.assertFalse(pkg.verify_package(self.install_dir, is_gpu=False))

    def test_verify_fails_missing_tensorrt_gpu(self):
        infer_dir = self.build_dir / "src" / "infer"
        infer_dir.mkdir(parents=True)
        (infer_dir / "visiontools_infer.dll").write_bytes(b"MOCK_INFER_DLL")
        self._write_resolved_deps(is_gpu=True)
        self._write_module_index(is_gpu=True)
        pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=True, strict=True)
        # Delete nvinfer DLL
        (self.install_dir / "bin" / "nvinfer_10.dll").unlink()
        self.assertFalse(pkg.verify_package(self.install_dir, is_gpu=True))

    def test_package_preserves_previous_on_failure(self):
        # Create an existing install dir with a marker file
        self.install_dir.mkdir(parents=True)
        marker = self.install_dir / "previous_release.txt"
        marker.write_text("old release")

        # Set module_index to point to a non-existent DLL
        (self.build_dir / "module_index.json").write_text(json.dumps({
            "variant": "cpu",
            "tools": {"fake": {"kind": "cpp_function", "module_dll": "bin/missing.dll"}}
        }))
        self._write_resolved_deps(is_gpu=False)

        with self.assertRaises(RuntimeError):
            pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)

        # Previous release MUST be preserved!
        self.assertTrue(marker.exists())
        self.assertEqual(marker.read_text(), "old release")

    def test_package_rollback_on_rename_failure(self):
        self.install_dir.mkdir(parents=True)
        marker = self.install_dir / "previous_release.txt"
        marker.write_text("old release")
        self._write_resolved_deps(is_gpu=False)
        self._write_module_index(is_gpu=False)

        # Inject error when moving staging_dir to install_dir
        orig_rename = pathlib.Path.rename
        def failing_rename(inst, target, *args, **kwargs):
            if ".tmp_pkg" in str(inst) or "staging" in str(inst):
                raise OSError("Simulated rename error")
            return orig_rename(inst, target, *args, **kwargs)

        import unittest.mock as mock
        with mock.patch.object(pathlib.Path, "rename", failing_rename):
            with mock.patch("shutil.move", side_effect=OSError("Simulated move error")):
                with self.assertRaises(RuntimeError):
                    pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)

        # Marker MUST still exist after rollback
        self.assertTrue(marker.exists())
        self.assertEqual(marker.read_text(), "old release")

if __name__ == "__main__":
    unittest.main()
