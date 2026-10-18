#!/usr/bin/env python3
"""
Unit tests for packaging and verification logic (YS-01 / YS-03).
Tests cover:
  - Direct destructive rebuild: clears pre-existing marker file upon rebuild
  - Packaging failure exits non-zero and leaves no backup or staging directory
  - Re-run after fixing input produces complete, verified package
  - Missing infer DLL in GPU mode (strict raises, verify fails)
  - GPU -> CPU repackaging in identical directory leaves no residual GPU artifacts
  - CPU package excludes GPU tools and DLLs
  - Paths with spaces
  - YAML index / resolved_dependencies validation
  - Verification failures on tampered/incomplete packages
"""

import os
import pathlib
import shutil
import sys
import tempfile
import unittest
import yaml

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
        (prep_dir / "visiontools_preprocess.dll").write_bytes(b"MOCK_PREPROCESS_DLL\x00visiontoolsModuleApi\x00")

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

        # Write resolved_dependencies.yaml and module_index.yaml
        self._write_resolved_deps(is_gpu=False)
        self._write_module_index(is_gpu=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_resolved_deps(self, is_gpu=True, config="Release"):
        prep_dll = self.build_dir / "src" / "preprocess" / "visiontools_preprocess.dll"
        modules = {
            "visiontools_preprocess": {
                "source": str(prep_dll).replace("\\", "/")
            }
        }
        runtime_files = [
            {
                "source": str(self.mock_opencv / "opencv_world480.dll").replace("\\", "/"),
                "destination": "bin/opencv_world480.dll"
            }
        ]
        if is_gpu:
            infer_dll = self.build_dir / "src" / "infer" / "visiontools_infer.dll"
            modules["visiontools_infer"] = {
                "source": str(infer_dll).replace("\\", "/")
            }
            runtime_files.extend([
                {
                    "source": str(self.mock_inferrt / "inferrt_core.dll").replace("\\", "/"),
                    "destination": "bin/inferrt_core.dll"
                },
                {
                    "source": str(self.mock_cuda / "cudart64_110.dll").replace("\\", "/"),
                    "destination": "bin/cudart64_110.dll"
                },
                {
                    "source": str(self.mock_trt / "nvinfer_10.dll").replace("\\", "/"),
                    "destination": "bin/nvinfer_10.dll"
                }
            ])

        deps = {
            "variant": "gpu" if is_gpu else "cpu",
            "config": config,
            "modules": modules,
            "runtime_files": runtime_files
        }
        cfg_dir = self.build_dir / config
        cfg_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg_dir / "resolved_dependencies.yaml", "w", encoding="utf-8") as f:
            yaml.dump(deps, f, sort_keys=False)

    def _write_module_index(self, is_gpu=True):
        tools = {
            "visiontools.preprocess.adaptive_binary": {
                "module": "visiontools_preprocess",
                "definition": "assets/tools/builtin/预处理/图像处理/visiontools.preprocess.adaptive_binary.yaml"
            },
            "visionrt.image.crop": {
                "module": "visiontools_preprocess",
                "definition": "assets/tools/builtin/预处理/图像处理/visionrt.image.crop.yaml"
            },
            "visionrt.detection.filter": {
                "module": "visiontools_preprocess",
                "definition": "assets/tools/builtin/处理/检测/visionrt.detection.filter.yaml"
            },
            "visionrt.roi.projection": {
                "module": "visiontools_preprocess",
                "definition": "assets/tools/builtin/预处理/ROI/visionrt.roi.projection.yaml"
            },
            "visionrt.image.preprocess": {
                "module": "visiontools_preprocess",
                "definition": "assets/tools/builtin/预处理/图像处理/visionrt.image.preprocess.yaml"
            },
            "visionrt.image.read": {
                "module": "visiontools_preprocess",
                "definition": "assets/tools/builtin/采集/图像采集/visionrt.image.read.yaml"
            }
        }
        modules = {
            "visiontools_preprocess": {}
        }
        if is_gpu:
            tools["visionrt.dl.yolo"] = {
                "module": "visiontools_infer",
                "definition": "assets/tools/builtin/深度学习/目标检测/visionrt.dl.yolo.yaml"
            }
            modules["visiontools_infer"] = {}

        idx = {
            "variant": "gpu" if is_gpu else "cpu",
            "modules": modules,
            "tools": tools
        }
        gen_dir = self.build_dir / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        with open(gen_dir / "module_index.yaml", "w", encoding="utf-8") as f:
            yaml.dump(idx, f, sort_keys=False)

    def test_missing_preprocess_dll_raises(self):
        empty_build = self.t / "empty_build"
        empty_build.mkdir()
        with self.assertRaises(RuntimeError):
            pkg.package_tools(empty_build, self.install_dir, infer_enabled=False, strict=True)

    def test_missing_infer_dll_gpu_strict_raises(self):
        self._write_resolved_deps(is_gpu=True)
        self._write_module_index(is_gpu=True)
        with self.assertRaises(RuntimeError) as ctx:
            pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=True, strict=True)
        self.assertIn("visiontools_infer.dll", str(ctx.exception))

    def test_missing_infer_dll_gpu_verify_fails(self):
        self._write_resolved_deps(is_gpu=True)
        self._write_module_index(is_gpu=True)
        with self.assertRaises(RuntimeError):
            pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=True, strict=False)
        self.assertFalse((self.install_dir / "bin" / "visiontools_infer.dll").exists())
        self.assertFalse(pkg.verify_package(self.install_dir, is_gpu=True))

    def test_valid_gpu_package(self):
        infer_dir = self.build_dir / "src" / "infer"
        infer_dir.mkdir(parents=True)
        (infer_dir / "visiontools_infer.dll").write_bytes(b"MOCK_INFER_DLL\x00visiontoolsModuleApi\x00")
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
        (infer_dir / "visiontools_infer.dll").write_bytes(b"MOCK_INFER_DLL\x00visiontoolsModuleApi\x00")
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

    def test_verify_fails_if_yaml_declaration_missing(self):
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
        (self.build_dir / "generated" / "module_index.yaml").unlink()
        self._write_resolved_deps(is_gpu=False)
        with self.assertRaises(RuntimeError) as ctx:
            pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)
        self.assertIn("Missing required build index", str(ctx.exception))

    def test_corrupt_build_index_raises(self):
        (self.build_dir / "generated" / "module_index.yaml").write_text("{corrupt yaml: [")
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
        (self.build_dir / "Release" / "resolved_dependencies.yaml").unlink()
        with self.assertRaises(RuntimeError) as ctx:
            pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)
        self.assertIn("Strict packaging requires build metadata", str(ctx.exception))

    def test_verify_fails_empty_tools(self):
        self._write_resolved_deps(is_gpu=False)
        self._write_module_index(is_gpu=False)
        pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)
        with open(self.install_dir / "module_index.yaml", "w", encoding="utf-8") as f:
            yaml.dump({"variant": "cpu", "modules": {"visiontools_preprocess": {"library": "bin/visiontools_preprocess.dll"}}, "tools": {}}, f)
        self.assertFalse(pkg.verify_package(self.install_dir, is_gpu=False))

    def test_verify_fails_bad_tool_entry(self):
        self._write_resolved_deps(is_gpu=False)
        self._write_module_index(is_gpu=False)
        pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)
        with open(self.install_dir / "module_index.yaml", "w", encoding="utf-8") as f:
            yaml.dump({"variant": "cpu", "modules": {"visiontools_preprocess": {"library": "bin/visiontools_preprocess.dll"}}, "tools": {"bad": 123}}, f)
        self.assertFalse(pkg.verify_package(self.install_dir, is_gpu=False))

    def test_verify_fails_escaping_module_dll(self):
        self._write_resolved_deps(is_gpu=False)
        self._write_module_index(is_gpu=False)
        pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)
        with open(self.install_dir / "module_index.yaml", "w", encoding="utf-8") as f:
            yaml.dump({
                "variant": "cpu",
                "modules": {"visiontools_preprocess": {"library": "../escape.dll"}},
                "tools": {"tool": {"module": "visiontools_preprocess", "definition": "assets/tools/foo.yaml"}}
            }, f)
        self.assertFalse(pkg.verify_package(self.install_dir, is_gpu=False))

    def test_verify_fails_missing_tensorrt_gpu(self):
        infer_dir = self.build_dir / "src" / "infer"
        infer_dir.mkdir(parents=True)
        (infer_dir / "visiontools_infer.dll").write_bytes(b"MOCK_INFER_DLL\x00visiontoolsModuleApi\x00")
        self._write_resolved_deps(is_gpu=True)
        self._write_module_index(is_gpu=True)
        pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=True, strict=True)
        # Delete nvinfer DLL
        (self.install_dir / "bin" / "nvinfer_10.dll").unlink()
        self.assertFalse(pkg.verify_package(self.install_dir, is_gpu=True))

    def test_direct_destructive_rebuild_clears_old_marker(self):
        """YS-03: Pre-existing marker in install_dir is removed during rebuild."""
        self.install_dir.mkdir(parents=True)
        marker = self.install_dir / "previous_release.txt"
        marker.write_text("old release")

        self._write_resolved_deps(is_gpu=False)
        self._write_module_index(is_gpu=False)
        pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)

        self.assertFalse(marker.exists())
        self.assertTrue((self.install_dir / "bin" / "visiontools_preprocess.dll").exists())
        self.assertTrue(pkg.verify_package(self.install_dir, is_gpu=False))

    def test_failure_exits_nonzero_no_backup_or_staging(self):
        """YS-03: Failure clears target, exits nonzero, leaves no backup or .tmp_pkg staging."""
        self.install_dir.mkdir(parents=True)
        marker = self.install_dir / "previous_release.txt"
        marker.write_text("old release")

        # Inject failure: non-existent module source
        deps = {
            "variant": "cpu",
            "config": "Release",
            "modules": {
                "visiontools_preprocess": {
                    "source": str(self.build_dir / "nonexistent.dll").replace("\\", "/")
                }
            },
            "runtime_files": []
        }
        cfg_dir = self.build_dir / "Release"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg_dir / "resolved_dependencies.yaml", "w", encoding="utf-8") as f:
            yaml.dump(deps, f)

        # Packaging must fail
        with self.assertRaises(RuntimeError):
            pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)

        # Destructive clean happened upfront, marker is gone
        self.assertFalse(marker.exists())

        # No backup or .tmp_pkg directory was created
        parent = self.install_dir.parent
        staging_dirs = [d for d in parent.iterdir() if ".tmp_pkg" in d.name or "backup" in d.name]
        self.assertEqual(staging_dirs, [])

    def test_fix_and_rerun_succeeds(self):
        """YS-03: After a failed packaging attempt, fixing inputs and rerunning succeeds."""
        # 1. Trigger failure
        deps = {
            "variant": "cpu",
            "config": "Release",
            "modules": {
                "visiontools_preprocess": {
                    "source": str(self.build_dir / "nonexistent.dll").replace("\\", "/")
                }
            },
            "runtime_files": []
        }
        cfg_dir = self.build_dir / "Release"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg_dir / "resolved_dependencies.yaml", "w", encoding="utf-8") as f:
            yaml.dump(deps, f)
        with self.assertRaises(RuntimeError):
            pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)

        # 2. Fix inputs
        self._write_resolved_deps(is_gpu=False)
        self._write_module_index(is_gpu=False)

        # 3. Rerun succeeds
        pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)
        self.assertTrue(pkg.verify_package(self.install_dir, is_gpu=False))

    def test_root_resolved_dependencies_rejected(self):
        """N05: Packaging fails if resolved_dependencies.yaml is only at root and not in config dir."""
        rel_file = self.build_dir / "Release" / "resolved_dependencies.yaml"
        root_file = self.build_dir / "resolved_dependencies.yaml"
        rel_file.rename(root_file)
        with self.assertRaises(RuntimeError) as ctx:
            pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)
        self.assertIn("Strict packaging requires build metadata", str(ctx.exception))

    def test_missing_required_dependency_in_manifest_fails_even_if_in_env(self):
        """VT-S01 / VT-T01: Missing required dependency in manifest fails packaging/verification even if available in env."""
        self._write_resolved_deps(is_gpu=False)
        cfg_file = self.build_dir / "Release" / "resolved_dependencies.yaml"
        deps = pkg.load_yaml(cfg_file)
        deps["runtime_files"] = []  # Omit opencv
        pkg.save_yaml(deps, cfg_file)
        self._write_module_index(is_gpu=False)

        env_backup = os.environ.get("OPENCV_DIR")
        os.environ["OPENCV_DIR"] = str(self.mock_opencv)
        try:
            with self.assertRaises(RuntimeError):
                pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)
            self.assertFalse(pkg.verify_package(self.install_dir, is_gpu=False))
        finally:
            if env_backup is not None:
                os.environ["OPENCV_DIR"] = env_backup
            else:
                os.environ.pop("OPENCV_DIR", None)

    def test_non_yolo_gpu_module_variant_filtering(self):
        """VT-S01 / VT-T01: Non-YOLO GPU module fixture is filtered by module_bindings variant without string heuristics."""
        custom_bindings = {
            "modules": {
                "visiontools_preprocess": {
                    "variant": "cpu",
                    "tools": {
                        "visionrt.image.crop": {"class": "Crop"}
                    }
                },
                "visiontools_segmentation": {
                    "variant": "gpu",
                    "tools": {
                        "visiontools.sam.segment": {"class": "Sam"}
                    }
                }
            }
        }
        tools_dict = {
            "visionrt.image.crop": {
                "module": "visiontools_preprocess",
                "definition": "assets/tools/crop.yaml"
            }
        }
        idx = {
            "variant": "cpu",
            "modules": {"visiontools_preprocess": {"library": "bin/visiontools_preprocess.dll"}},
            "tools": tools_dict,
            "runtime_files": ["bin/opencv_world480.dll"]
        }
        (self.install_dir / "bin").mkdir(parents=True, exist_ok=True)
        pkg.save_yaml(idx, self.install_dir / "module_index.yaml")
        (self.install_dir / "bin" / "visiontools_preprocess.dll").write_bytes(b"PREPROCESS")
        (self.install_dir / "bin" / "opencv_world480.dll").write_bytes(b"OPENCV")
        (self.install_dir / "assets" / "tools").mkdir(parents=True, exist_ok=True)
        (self.install_dir / "assets" / "tools" / "crop.yaml").write_text("entry:\n  registry_name: visionrt.image.crop\n")

        self.assertTrue(pkg.verify_package(self.install_dir, is_gpu=False))

        (self.install_dir / "bin" / "visiontools_segmentation.dll").write_bytes(b"GPU_SEG")
        bindings_file = self.install_dir.parent.parent / "tools" / "module_bindings.yaml"
        bindings_file.parent.mkdir(parents=True, exist_ok=True)
        existing_bindings = bindings_file.read_text(encoding="utf-8") if bindings_file.exists() else None
        try:
            pkg.save_yaml(custom_bindings, bindings_file)
            self.assertFalse(pkg.verify_package(self.install_dir, is_gpu=False))
        finally:
            if existing_bindings is not None:
                bindings_file.write_text(existing_bindings, encoding="utf-8")
            elif bindings_file.exists():
                bindings_file.unlink()

    def test_omitted_tool_in_build_index_rejected(self):
        """N01: Build index missing an active declared tool is rejected."""
        self._write_resolved_deps(is_gpu=False)
        self._write_module_index(is_gpu=False)
        idx_file = self.build_dir / "generated" / "module_index.yaml"
        idx = pkg.load_yaml(idx_file)
        idx["tools"].pop("visionrt.image.crop")
        pkg.save_yaml(idx, idx_file)
        with self.assertRaises(RuntimeError) as ctx:
            pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)
        self.assertIn("Exact tool set mismatch", str(ctx.exception))

    def test_missing_variant_in_build_index_rejected(self):
        """N02: Build index missing variant is rejected."""
        self._write_resolved_deps(is_gpu=False)
        self._write_module_index(is_gpu=False)
        idx_file = self.build_dir / "generated" / "module_index.yaml"
        idx = pkg.load_yaml(idx_file)
        idx.pop("variant")
        pkg.save_yaml(idx, idx_file)
        with self.assertRaises(RuntimeError) as ctx:
            pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=False, strict=True)
        self.assertIn("Missing required 'variant'", str(ctx.exception))

    def test_tensorrt_plugin_rejected(self):
        """N04: TensorRT plugin DLL cannot replace core nvinfer DLL in GPU verify."""
        infer_dir = self.build_dir / "src" / "infer"
        infer_dir.mkdir(parents=True, exist_ok=True)
        (infer_dir / "visiontools_infer.dll").write_bytes(b"MOCK_INFER_DLL\x00visiontoolsModuleApi\x00")
        self._write_resolved_deps(is_gpu=True)
        self._write_module_index(is_gpu=True)
        pkg.package_tools(self.build_dir, self.install_dir, infer_enabled=True, strict=True)
        # Rename core to plugin
        (self.install_dir / "bin" / "nvinfer_10.dll").rename(self.install_dir / "bin" / "nvinfer_plugin_10.dll")
        idx_file = self.install_dir / "module_index.yaml"
        idx = pkg.load_yaml(idx_file)
        idx["runtime_files"] = [f.replace("nvinfer_10.dll", "nvinfer_plugin_10.dll") for f in idx["runtime_files"]]
        pkg.save_yaml(idx, idx_file)
        self.assertFalse(pkg.verify_package(self.install_dir, is_gpu=True))

if __name__ == "__main__":
    unittest.main()
