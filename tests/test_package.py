import os
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import package_tools as package


class TestPackageTools(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        self.build = self.project / "build"
        self.output = self.project / "dist/工具 包"
        self.validator = Path(os.environ.get("VISIONTOOLS_VALIDATOR", ROOT / "build/3rdparty/VisionRTSDK/visionrt_validate_tool.exe"))
        self.assertTrue(self.validator.is_file(), "Build the SDK validator before package tests")
        self.declaration = self.project / "assets/tools/custom/Math/Arithmetic/tool.yaml"
        self.declaration.parent.mkdir(parents=True)
        self.document = {
            "schema_version": 1, "tool_id": "custom.math", "display_name": "Math",
            "category": "Mathematics", "subcategory": "Arithmetic", "version": "1.0.0",
            "kind": "python_function", "entry": {"file": "tool.py", "symbol": "run"},
            "resources": ["bias.dat"], "init_params": [], "inputs": [],
            "outputs": [{"id": "value", "name": "Value", "type": "Int"}],
        }
        (self.declaration.parent / "tool.py").write_text(
            'from .helper import calculate\ndef run(inputs): return {"value": calculate()}\n', encoding="utf-8")
        (self.declaration.parent / "helper.py").write_text(
            'from pathlib import Path\ndef calculate(): return int(Path(__file__).with_name("bias.dat").read_text())\n', encoding="utf-8")
        (self.declaration.parent / "bias.dat").write_text("42", encoding="utf-8")
        self.index = {
            "schema_version": 2, "variant": "portable-release", "modules": {},
            "tools": {"custom.math": {"definition": self.declaration.relative_to(self.project).as_posix()}},
        }
        self.manifest = {"variant": "portable-release", "config": "Release", "modules": {}, "runtime_files": []}

    def publish(self):
        package.save_yaml(self.document, self.declaration)
        package.save_yaml(self.index, self.build / "generated/module_index.yaml")
        package.save_yaml(self.manifest, self.build / "Release/resolved_dependencies.yaml")
        return package.package_tools(self.build, self.output, project_root=self.project, validator=self.validator)

    def test_zip_and_external_metadata_have_one_source(self):
        archive = self.publish()
        with zipfile.ZipFile(archive) as bundle:
            self.assertEqual(bundle.read("module_index.yaml"), (archive.parent / "module_index.yaml").read_bytes())
            index = yaml.safe_load(bundle.read("module_index.yaml"))
            self.assertEqual(index["variant"], "portable-release")
            self.assertEqual(index["modules"], {})
            self.assertEqual(set(index["tools"]), {"custom.math"})
        self.assertTrue(package.verify_package(self.output, self.validator))

    def test_backend_label_does_not_require_vendor_modules(self):
        self.index["variant"] = self.manifest["variant"] = "gpu"
        archive = self.publish()
        self.assertTrue(archive.is_file())
        self.assertFalse((self.output / "bin").exists())

    def test_python_relative_imports_and_declared_data_are_collected(self):
        archive = self.publish()
        with zipfile.ZipFile(archive) as bundle:
            prefix = self.declaration.parent.relative_to(self.project).as_posix()
            for name in ("tool.py", "helper.py", "bias.dat"):
                self.assertIn(f"{prefix}/{name}", bundle.namelist())

    def test_native_tool_can_ship_declared_data_without_python(self):
        self.document["kind"] = "cpp_function"
        self.document["entry"] = {"registry_name": "AdaptiveBinaryTool"}
        self.index["modules"] = {"visiontools_preprocess": {}}
        self.index["tools"]["custom.math"]["module"] = "visiontools_preprocess"
        self.manifest["modules"] = {"visiontools_preprocess": {"source": str(ROOT / "build/src/preprocess/visiontools_preprocess.dll")}}
        self.publish()
        self.assertEqual((self.output / self.declaration.parent.relative_to(self.project) / "bias.dat").read_text(), "42")

    def test_platform_only_package_has_no_fake_module(self):
        self.document["kind"] = "cpp_class"
        self.document["entry"] = {"registry_name": "platform.math"}
        self.index["tools"]["custom.math"]["host_binding"] = "visionrt.receiver"
        archive = self.publish()
        with zipfile.ZipFile(archive) as bundle:
            self.assertFalse(any(name.startswith("bin/") for name in bundle.namelist()))

    def test_caches_examples_and_unrelated_data_are_not_published(self):
        cache = self.declaration.parent / "__pycache__/tool.cpython-39.pyc"
        cache.parent.mkdir()
        cache.write_bytes(b"obsolete")
        (self.declaration.parent / "unrelated.log").write_text("development log")
        examples = self.project / "examples/private.dat"
        examples.parent.mkdir()
        examples.write_bytes(b"not a runtime resource")
        archive = self.publish()
        with zipfile.ZipFile(archive) as bundle:
            self.assertFalse(any("__pycache__" in name or name.endswith((".pyc", ".log")) or name.startswith("examples/") for name in bundle.namelist()))

    def test_runtime_files_are_manifest_driven(self):
        dependency = self.project / "runtime/accelerator.dat"
        dependency.parent.mkdir()
        dependency.write_bytes(b"runtime data")
        self.manifest["runtime_files"] = [{"source": str(dependency), "destination": "runtime/accelerator.dat"}]
        self.publish()
        self.assertEqual((self.output / "runtime/accelerator.dat").read_bytes(), b"runtime data")

    def test_republish_removes_obsolete_files(self):
        self.publish()
        (self.output / "obsolete.dll").write_bytes(b"old")
        self.publish()
        self.assertFalse((self.output / "obsolete.dll").exists())

    def test_configuration_mismatch_is_rejected(self):
        self.manifest["variant"] = "different-build"
        with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
            self.publish()

    def test_module_manifest_mismatch_is_rejected(self):
        self.index["modules"] = {"missing": {}}
        with self.assertRaisesRegex(RuntimeError, "modules.*do not match"):
            self.publish()

    def test_tool_identity_mismatch_is_rejected(self):
        self.document["tool_id"] = "different.identity"
        with self.assertRaisesRegex(RuntimeError, "tool_id"):
            self.publish()

    def test_missing_script_is_rejected_by_sdk(self):
        self.document["entry"]["file"] = "missing.py"
        with self.assertRaisesRegex(RuntimeError, "SDK package validation failed"):
            self.publish()

    def test_python_cannot_reference_a_native_module(self):
        self.index["tools"]["custom.math"]["module"] = "missing"
        with self.assertRaisesRegex(RuntimeError, "SDK package validation failed"):
            self.publish()

    def test_missing_resource_is_rejected(self):
        self.document["resources"] = ["missing.bin"]
        with self.assertRaises(OSError):
            self.publish()

    def test_resource_cannot_escape_its_declaration(self):
        self.document["resources"] = ["../outside.dat"]
        with self.assertRaisesRegex(RuntimeError, "inside its root"):
            self.publish()

    def test_runtime_destination_cannot_escape_collection(self):
        source = self.project / "runtime.dat"
        source.write_bytes(b"runtime")
        self.manifest["runtime_files"] = [{"source": str(source), "destination": "../outside.dat"}]
        with self.assertRaisesRegex(RuntimeError, "inside its root"):
            self.publish()
        self.assertFalse((self.output.parent / "outside.dat").exists())

    def test_package_cannot_replace_source_directory(self):
        with self.assertRaisesRegex(RuntimeError, "Refusing"):
            package.package_tools(self.build, self.project / "assets", project_root=self.project, validator=self.validator)
        self.assertTrue(self.declaration.parent.is_dir())

    def test_verifier_requires_the_real_sdk_validator(self):
        with self.assertRaisesRegex(RuntimeError, "validator not found"):
            package.verify_package(self.output, self.project / "missing-validator.exe")


if __name__ == "__main__":
    unittest.main()
