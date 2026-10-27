"""Profile selection contract: what ships, why, and what is refused."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import package_profile  # noqa: E402


PROFILES = Path(__file__).resolve().parents[1] / "tools/package_profiles.json"


def entry(name: str, **extra) -> dict:
    payload = {"source": f"C:/sdk/{name}", "destination": f"bin/{name}"}
    payload.update(extra)
    return payload


class ProfileLoadTests(unittest.TestCase):
    def test_shipped_profiles_describe_the_four_targets(self):
        profiles = package_profile.load_profiles(PROFILES)
        self.assertEqual({"cpu", "rtx40-runtime", "rtx40-build", "gpu-custom"}, set(profiles))
        self.assertEqual(("sm89",), profiles["rtx40-build"].gpu_architectures)
        self.assertTrue(profiles["rtx40-build"].engine_build)
        self.assertFalse(profiles["rtx40-runtime"].engine_build)
        self.assertFalse(profiles["cpu"].infer_enabled)

    def test_invalid_architecture_is_refused(self):
        document = {"schema_version": 1, "profiles": {"x": {"gpu_architectures": ["8.9"]}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(package_profile.ProfileError):
                package_profile.load_profiles(path)

    def test_custom_profile_needs_an_explicit_sm_set(self):
        profiles = package_profile.load_profiles(PROFILES)
        with self.assertRaises(package_profile.ProfileError):
            package_profile.resolve_profile(profiles, "gpu-custom")
        custom = package_profile.resolve_profile(profiles, "gpu-custom", ["86", "sm89", "89"])
        self.assertEqual(("sm86", "sm89"), custom.gpu_architectures)
        self.assertTrue(custom.infer_enabled)

    def test_fixed_profile_refuses_a_contradicting_sm(self):
        profiles = package_profile.load_profiles(PROFILES)
        with self.assertRaises(package_profile.ProfileError):
            package_profile.resolve_profile(profiles, "rtx40-build", ["86"])


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.profiles = package_profile.load_profiles(PROFILES)

    def names(self, selection):
        return sorted(Path(item["destination"]).name for item in selection.kept)

    def test_build_profile_keeps_only_its_own_builder_partition(self):
        selection = package_profile.select_dependencies(
            [
                entry("nvinfer_10.dll"),
                entry("nvinfer_builder_resource_10.14.1_sm89.dll"),
                entry("nvinfer_builder_resource_10.14.1_sm86.dll"),
                entry("nvinfer_builder_resource_10.14.1_sm120.dll"),
                entry("opencv_world480.dll"),
                entry("Qt6Core.dll"),
            ],
            self.profiles["rtx40-build"],
        )
        self.assertFalse(selection.errors)
        self.assertEqual(
            ["Qt6Core.dll", "nvinfer_10.dll", "nvinfer_builder_resource_10.14.1_sm89.dll", "opencv_world480.dll"],
            self.names(selection),
        )
        excluded = {Path(item["destination"]).name for item in selection.dropped}
        self.assertEqual({"nvinfer_builder_resource_10.14.1_sm86.dll",
                          "nvinfer_builder_resource_10.14.1_sm120.dll"}, excluded)
        for item in selection.dropped:
            self.assertIn("sm", item["reason"])

    def test_runtime_profile_ships_no_builder_partition(self):
        selection = package_profile.select_dependencies(
            [entry("nvinfer_10.dll"), entry("nvinfer_builder_resource_sm89.dll")],
            self.profiles["rtx40-runtime"],
        )
        self.assertFalse(selection.errors)
        self.assertEqual([], [item for item in selection.kept
                              if item["component"] == "tensorrt-builder-resource"])

    def test_cpu_profile_drops_every_gpu_runtime(self):
        selection = package_profile.select_dependencies(
            [entry("nvinfer_10.dll"), entry("cudart64_110.dll"), entry("opencv_core480.dll")],
            self.profiles["cpu"],
        )
        self.assertEqual(["opencv_core480.dll"], self.names(selection))
        self.assertEqual(2, len(selection.dropped))

    def test_unknown_builder_layout_is_an_error_not_a_copy(self):
        selection = package_profile.select_dependencies(
            [entry("nvinfer_builder_resource.dll")], self.profiles["rtx40-build"]
        )
        self.assertEqual(1, len(selection.errors))
        self.assertIn("未知布局", selection.errors[0])
        self.assertEqual([], selection.kept)

    def test_real_partitioned_sdk_layout_selects_one_architecture(self):
        # The installed TensorRT 10 layout names its partitions `_sm89_`, `_sm120_`, `_ptx_`.
        selection = package_profile.select_dependencies(
            [
                entry("nvinfer_builder_resource_sm75_10.dll"),
                entry("nvinfer_builder_resource_sm89_10.dll"),
                entry("nvinfer_builder_resource_sm120_10.dll"),
                entry("nvinfer_builder_resource_ptx_10.dll"),
            ],
            self.profiles["rtx40-build"],
        )
        self.assertFalse(selection.errors)
        self.assertEqual(["nvinfer_builder_resource_sm89_10.dll"], self.names(selection))

    def test_ptx_ships_only_for_a_forward_compatible_profile(self):
        plain = package_profile.select_dependencies(
            [entry("nvinfer_builder_resource_ptx_10.dll")], self.profiles["rtx40-build"]
        )
        self.assertEqual([], plain.kept)
        self.assertIn("PTX", plain.dropped[0]["reason"])

        compatible = package_profile.Profile(
            name="rtx40-build", gpu_architectures=("sm89",), infer_enabled=True, engine_build=True,
            runtime="full", hardware_forward_compatibility=True,
        )
        kept = package_profile.select_dependencies([entry("nvinfer_builder_resource_ptx_10.dll")], compatible)
        self.assertEqual(["nvinfer_builder_resource_ptx_10.dll"], self.names(kept))

    def test_every_kept_file_is_explainable(self):
        selection = package_profile.select_dependencies(
            [entry("nvinfer_builder_resource_sm89.dll", required_by=["visiontools_infer"])],
            self.profiles["rtx40-build"],
        )
        item = selection.kept[0]
        self.assertEqual("tensorrt-builder-resource", item["component"])
        self.assertEqual(["visiontools_infer"], item["required_by"])
        self.assertEqual(["sm89"], item["target_sm"])
        self.assertTrue(item["reason"])

        shared = package_profile.select_dependencies([entry("Qt6Gui.dll")], self.profiles["rtx40-build"]).kept[0]
        self.assertEqual(["shared"], shared["target_sm"])
        self.assertEqual(["declared-runtime"], shared["required_by"])

    def test_report_counts_files_and_bytes_per_component(self):
        selection = package_profile.select_dependencies(
            [
                entry("nvinfer_builder_resource_sm89.dll", size_bytes=100),
                entry("opencv_world480.dll", size_bytes=50),
                entry("opencv_core480.dll", size_bytes=25),
            ],
            self.profiles["rtx40-build"],
        )
        report = selection.report()
        self.assertEqual({"files": 1, "bytes": 100}, report["components"]["tensorrt-builder-resource"])
        self.assertEqual({"files": 2, "bytes": 75}, report["components"]["opencv"])
        self.assertEqual(report["profile"]["name"], "rtx40-build")

    def test_manifest_keeps_unrelated_keys_and_records_the_profile(self):
        metadata = {
            "variant": "VisionTools-0.1.0",
            "config": "Release",
            "runtime_files": [entry("nvinfer_builder_resource_sm86.dll"), entry("opencv_world480.dll")],
        }
        selection = package_profile.select_dependencies(metadata["runtime_files"], self.profiles["rtx40-build"])
        updated = package_profile.apply_to_manifest(metadata, selection)
        self.assertEqual("VisionTools-0.1.0", updated["variant"])
        self.assertEqual(["opencv_world480.dll"],
                         [Path(item["destination"]).name for item in updated["runtime_files"]])
        self.assertEqual("rtx40-build", updated["package_profile"]["name"])
        self.assertEqual(1, len(updated["dependencies_report"]["excluded"]))


if __name__ == "__main__":
    unittest.main()
