import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import package_tools as package


def validator_path() -> Path:
    configured = os.environ.get("VISIONTOOLS_VALIDATOR")
    if not configured:
        raise AssertionError("VISIONTOOLS_VALIDATOR must point at the real host validator")
    path = Path(configured)
    if not path.is_file():
        raise AssertionError(f"VISIONTOOLS_VALIDATOR is not a file: {path}")
    return path


class PackageFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        self.build = self.project / "build"
        self.output = self.project / "dist/工具 包"
        self.validator = validator_path()
        self.declaration = self.project / "assets/tools/custom/Math/Arithmetic/tool.json"
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
            'from pathlib import Path\ndef calculate(): return int(Path(__file__).with_name("bias.dat").read_text())\n',
            encoding="utf-8")
        (self.declaration.parent / "bias.dat").write_text("42", encoding="utf-8")
        self.index = {
            "schema_version": 1, "modules": [],
            "tools": [{"definition": self.declaration.relative_to(self.project).as_posix()}],
        }
        self.manifest = {"variant": "portable-release", "config": "Release", "modules": {}, "runtime_files": []}

    @property
    def targets(self):
        return package.resolve_targets(self.output, self.project)

    def publish(self, install_dir=None, zip_output=None):
        package.save_json(self.document, self.declaration)
        package.save_json(self.index, self.project / "tools/package_catalog.json")
        package.save_json(self.manifest, self.build / "Release/resolved_dependencies.json")
        return package.package_tools(self.build, install_dir or self.output, project_root=self.project,
                                     validator=self.validator, zip_output=zip_output)

    def state(self):
        targets = self.targets
        files = {}
        if targets.install_dir.is_dir():
            for path in sorted(targets.install_dir.rglob("*")):
                if path.is_file():
                    files[path.relative_to(targets.install_dir).as_posix()] = path.read_bytes()
        return {
            "install": files,
            "zip": targets.zip_output.read_bytes() if targets.zip_output.is_file() else None,
            "index": targets.index_output.read_bytes() if targets.index_output.is_file() else None,
        }

    def assert_state(self, expected):
        self.assertEqual(self.state(), expected)

    def staging_entries(self):
        return [path for path in self.output.parent.iterdir()
                if ".staging-" in path.name or ".backup-" in path.name]


class TestPackageTransaction(PackageFixture):
    def test_author_json_lint_preserves_the_previous_package(self):
        self.publish()
        previous = self.state()
        source = self.declaration.read_text(encoding="utf-8")
        invalid = (
            source.replace('"tool_id": "custom.math"', '"tool_id": "ignored", "tool_id": "custom.math"'),
            source.replace('"type": "Int"', '"type": "Float", "type": "Int"'),
            source.replace('"type": "Int"', '"type": "Float", "t\\u0079pe": "Int"'),
            source.replace('"schema_version": 1', '"schema_version": NaN'),
        )
        for document in invalid:
            with self.subTest(document=document):
                self.assertNotEqual(document, source)
                self.declaration.write_text(document, encoding="utf-8")
                with self.assertRaisesRegex(package.PackagingError, r"tool\.json.*(重复成员|非有限)"):
                    package.package_tools(self.build, self.output, project_root=self.project,
                                          validator=self.validator)
                self.assert_state(previous)
                self.assertEqual(self.staging_entries(), [])

    def test_verification_lints_packaged_declarations_and_index(self):
        self.publish()
        declaration = self.output / self.declaration.relative_to(self.project)
        index = self.output / "module_index.json"
        command = [sys.executable, str(ROOT / "tools/package_tools.py"), "--verify-only",
                   "--build-dir", str(self.build), "--install-dir", str(self.output),
                   "--validator", str(self.validator)]
        for path, member in ((declaration, "kind"), (index, "variant")):
            original = path.read_text(encoding="utf-8")
            path.write_text('{"' + member + '": "ignored", ' + original[1:], encoding="utf-8")
            with self.subTest(path=path):
                result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("重复成员", result.stderr)
                self.assertIn(path.name, result.stderr)
            path.write_text(original, encoding="utf-8")
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_zip_and_external_index_are_one_source(self):
        archive = self.publish()
        targets = self.targets
        with zipfile.ZipFile(archive) as bundle:
            self.assertEqual(bundle.read("module_index.json"), targets.index_output.read_bytes())
            index = json.loads(bundle.read("module_index.json"))
            self.assertEqual(index["variant"], "portable-release")
            self.assertEqual(index["modules"], [])
            self.assertEqual({tool["tool_id"] for tool in index["tools"]}, {"custom.math"})
        self.assertTrue(package.verify_package(targets.install_dir, self.validator))

    def test_install_directory_is_marked_and_marker_is_not_packaged(self):
        archive = self.publish()
        marker = self.output / package.MARKER_NAME
        self.assertTrue(marker.is_file())
        self.assertEqual(json.loads(marker.read_text(encoding="utf-8"))["package"], "工具 包")
        with zipfile.ZipFile(archive) as bundle:
            self.assertNotIn(package.MARKER_NAME, bundle.namelist())

    def test_backend_label_does_not_require_vendor_modules(self):
        self.manifest["variant"] = "gpu"
        archive = self.publish()
        self.assertTrue(archive.is_file())
        self.assertFalse((self.output / "bin").exists())

    def test_python_code_package_root_and_declared_data_are_collected(self):
        archive = self.publish()
        with zipfile.ZipFile(archive) as bundle:
            prefix = self.declaration.parent.relative_to(self.project).as_posix()
            for name in ("tool.py", "helper.py", "bias.dat"):
                self.assertIn(f"{prefix}/{name}", bundle.namelist())

    def test_native_tool_can_ship_declared_data_without_python(self):
        self.document["kind"] = "cpp"
        self.document.pop("entry")
        self.index["modules"] = ["visiontools_preprocess"]
        self.index["tools"][0]["module"] = "visiontools_preprocess"
        module = self.build / "src/preprocess/vst_image.dll"
        module.parent.mkdir(parents=True)
        module.write_bytes(b"module")
        self.manifest["modules"] = {"visiontools_preprocess": {"source": str(module)}}
        self.publish()
        self.assertEqual((self.output / self.declaration.parent.relative_to(self.project) / "bias.dat").read_text(), "42")

    def test_platform_only_package_has_no_fake_module(self):
        self.document["kind"] = "host"
        self.document["entry"] = {"binding": "visionrt.receiver"}
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
            self.assertFalse(any("__pycache__" in name or name.endswith((".pyc", ".log")) or name.startswith("examples/")
                                 for name in bundle.namelist()))

    def test_runtime_files_are_manifest_driven(self):
        dependency = self.project / "runtime/accelerator.dat"
        dependency.parent.mkdir()
        dependency.write_bytes(b"runtime data")
        self.manifest["runtime_files"] = [{"source": str(dependency), "destination": "runtime/accelerator.dat"}]
        self.publish()
        self.assertEqual((self.output / "runtime/accelerator.dat").read_bytes(), b"runtime data")

    @unittest.skipUnless(os.name == "nt", "Windows DLL search path contract")
    def test_validator_uses_collected_runtime_without_developer_path(self):
        runtime = next(
            candidate for directory in os.get_exec_path()
            for candidate in Path(directory).glob("opencv_world*.dll")
            if candidate.stem[-1].isdigit()
        )
        self.manifest["runtime_files"] = [{"source": str(runtime), "destination": f"bin/{runtime.name}"}]
        system_root = Path(os.environ["SystemRoot"])
        with patch.dict(os.environ, {"PATH": os.pathsep.join(map(str, [system_root / "System32", system_root]))}):
            archive = self.publish()
        self.assertTrue(archive.is_file())

    def test_republish_removes_obsolete_files(self):
        self.publish()
        (self.output / "obsolete.dll").write_bytes(b"old")
        self.publish()
        self.assertFalse((self.output / "obsolete.dll").exists())
        self.assertEqual(self.staging_entries(), [])

    def test_configuration_mismatch_is_rejected(self):
        self.manifest["config"] = "Debug"
        with self.assertRaisesRegex(RuntimeError, "配置不一致"):
            self.publish()

    def test_module_manifest_mismatch_is_rejected(self):
        self.manifest["modules"] = {"missing": {"source": "missing.dll"}}
        with self.assertRaisesRegex(RuntimeError, "模块"):
            self.publish()

    def test_duplicate_tool_identity_is_rejected(self):
        second = self.declaration.parent / "second.json"
        package.save_json(self.document, second)
        self.index["tools"].append({"definition": second.relative_to(self.project).as_posix()})
        with self.assertRaisesRegex(RuntimeError, "tool_id"):
            self.publish()

    def test_missing_script_is_rejected(self):
        self.document["entry"]["file"] = "missing.py"
        with self.assertRaisesRegex(RuntimeError, "入口脚本"):
            self.publish()

    def test_python_cannot_reference_a_native_module(self):
        self.index["tools"][0]["module"] = "missing"
        with self.assertRaisesRegex(RuntimeError, "模块"):
            self.publish()

    def test_missing_resource_is_rejected(self):
        self.document["resources"] = ["missing.bin"]
        with self.assertRaisesRegex(RuntimeError, "资源不存在"):
            self.publish()

    def test_resource_cannot_escape_the_project(self):
        self.document["resources"] = [os.path.relpath(self.project.parent / "outside.dat", self.declaration.parent)]
        with self.assertRaisesRegex(RuntimeError, "根目录内"):
            self.publish()

    def test_shared_resource_can_use_a_parent_within_the_package(self):
        resource = self.declaration.parent.parent / "shared.dat"
        resource.write_text("shared", encoding="utf-8")
        self.document["resources"] = ["../shared.dat"]
        self.publish()
        self.assertEqual((self.output / resource.relative_to(self.project)).read_text(), "shared")

    def test_runtime_destination_cannot_escape_collection(self):
        source = self.project / "runtime.dat"
        source.write_bytes(b"runtime")
        self.manifest["runtime_files"] = [{"source": str(source), "destination": "../outside.dat"}]
        with self.assertRaisesRegex(RuntimeError, "根目录内"):
            self.publish()
        self.assertFalse((self.output.parent / "outside.dat").exists())

    def test_verifier_requires_the_real_host_validator(self):
        with self.assertRaisesRegex(RuntimeError, "宿主校验器不存在"):
            package.verify_package(self.output, self.project / "missing-validator.exe")


class TestTargetProtection(PackageFixture):
    def test_protected_install_targets_are_rejected_without_writes(self):
        sentinel = self.project / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        for name in ("docs", "cmake", "build", "src", "assets", "tools", "tests", "3rdparty", "dist"):
            target = self.project / name
            target.mkdir(parents=True, exist_ok=True)
            marker = target / "sentinel.txt"
            marker.write_text(name, encoding="utf-8")
        for target in (self.project, self.project / "docs", self.project / "cmake", self.project / "build",
                       self.project / "src", self.project / "assets", self.project / "tools",
                       self.project / "tests", self.project / "3rdparty", self.project / "dist"):
            with self.subTest(target=target):
                with self.assertRaises(RuntimeError):
                    self.publish(install_dir=target)
                self.assertTrue(sentinel.is_file())
                self.assertEqual(self.staging_entries(), [])
        self.assertEqual((self.project / "docs/sentinel.txt").read_text(), "docs")
        self.assertEqual((self.project / "dist/sentinel.txt").read_text(), "dist")

    def test_internal_output_must_be_a_dedicated_dist_child(self):
        nested = self.project / "dist/group/nested"
        with self.assertRaisesRegex(RuntimeError, "dist"):
            self.publish(install_dir=nested)
        self.assertFalse(nested.exists())

    def test_symlink_to_source_is_rejected(self):
        link = self.project / "dist/source-link"
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(self.project / "docs", link, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"symlinks are unavailable: {error}")
        with self.assertRaises(RuntimeError):
            self.publish(install_dir=link)
        self.assertFalse((self.project / "docs/module_index.json").exists())

    def test_existing_unmarked_directory_is_rejected(self):
        unrelated = self.project / "dist/Unrelated"
        unrelated.mkdir(parents=True)
        (unrelated / "user.txt").write_text("user", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "不是本工具包"):
            self.publish(install_dir=unrelated)
        self.assertEqual((unrelated / "user.txt").read_text(), "user")

    def test_existing_unmarked_zip_is_rejected(self):
        self.publish()
        shutil.rmtree(self.output)
        self.targets.zip_output.write_bytes(b"other package")
        with self.assertRaisesRegex(RuntimeError, "旧产物"):
            self.publish()
        self.assertEqual(self.targets.zip_output.read_bytes(), b"other package")

    def test_external_new_and_marked_directory_can_be_published(self):
        external = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, external, True)
        target = external / "VisionTools"
        self.publish(install_dir=target)
        self.assertTrue((target / package.MARKER_NAME).is_file())
        self.publish(install_dir=target)
        self.assertTrue((target / "module_index.json").is_file())

    def test_external_unrelated_directory_is_rejected(self):
        external = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, external, True)
        target = external / "user-data"
        target.mkdir()
        (target / "private.txt").write_text("private", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "不是本工具包"):
            self.publish(install_dir=target)
        self.assertEqual((target / "private.txt").read_text(), "private")

    def test_zip_must_belong_to_the_same_package_group(self):
        self.publish()
        state = self.state()
        with self.assertRaisesRegex(RuntimeError, "同一发行组"):
            self.publish(zip_output=self.project / "dist/other.zip")
        self.assert_state(state)

    def test_two_packages_share_a_parent_without_conflict(self):
        other = self.project / "dist/Other"
        self.publish()
        first = self.state()
        self.publish(install_dir=other)
        self.assert_state(first)
        other_targets = package.resolve_targets(other, self.project)
        self.assertTrue(other_targets.zip_output.is_file())
        self.assertTrue(other_targets.index_output.is_file())


class TestFailureRollback(PackageFixture):
    def setUp(self):
        super().setUp()
        self.publish()
        self.baseline = self.state()

    def assert_previous_package_usable(self):
        self.assert_state(self.baseline)
        self.assertEqual(self.staging_entries(), [])
        self.assertTrue((self.output / package.MARKER_NAME).is_file())
        self.assertTrue(package.verify_package(self.output, self.validator))

    def test_missing_module_preserves_previous_package(self):
        self.manifest["modules"] = {"visiontools_preprocess": {"source": str(self.build / "missing.dll")}}
        self.index["modules"] = ["visiontools_preprocess"]
        self.index["tools"][0]["module"] = "visiontools_preprocess"
        with self.assertRaisesRegex(RuntimeError, "模块"):
            self.publish()
        self.assert_previous_package_usable()

    def test_validator_failure_preserves_previous_package(self):
        with patch.object(package, "verify_package", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "宿主包校验失败"):
                self.publish()
        self.assert_previous_package_usable()

    def test_copy_failure_preserves_previous_package(self):
        original = package.shutil.copy2

        def failing_copy(source, destination, *args, **kwargs):
            if str(source).endswith("bias.dat"):
                raise OSError("copy injected failure")
            return original(source, destination, *args, **kwargs)

        with patch.object(package.shutil, "copy2", side_effect=failing_copy):
            with self.assertRaisesRegex(RuntimeError, "复制"):
                self.publish()
        self.assert_previous_package_usable()

    def test_zip_failure_preserves_previous_package(self):
        with patch.object(package, "_write_zip", side_effect=OSError("zip injected failure")):
            with self.assertRaisesRegex(RuntimeError, "ZIP"):
                self.publish()
        self.assert_previous_package_usable()

    def test_commit_rename_failure_rolls_back_every_target(self):
        original = package._rename

        def failing_rename(source, target):
            if ".staging-" in Path(source).name and Path(target).name.endswith(".zip"):
                raise OSError("rename injected failure")
            return original(source, target)

        with patch.object(package, "_rename", side_effect=failing_rename):
            with self.assertRaisesRegex(RuntimeError, "提交"):
                self.publish()
        self.assert_previous_package_usable()

    def test_install_failure_preserves_previous_package(self):
        self.manifest["runtime_install"] = {"cmake": sys.executable, "component": "VisionToolsRuntime"}
        with patch.object(package.subprocess, "run") as run:
            run.return_value = type("Result", (), {"returncode": 1, "stderr": "install failed", "stdout": ""})()
            with self.assertRaisesRegex(RuntimeError, "安装"):
                self.publish()
        self.assert_previous_package_usable()

    def test_marker_is_not_created_for_a_failed_new_package(self):
        fresh = self.project / "dist/Fresh"
        self.manifest["modules"] = {"visiontools_preprocess": {"source": str(self.build / "missing.dll")}}
        self.index["modules"] = ["visiontools_preprocess"]
        self.index["tools"][0]["module"] = "visiontools_preprocess"
        with self.assertRaises(RuntimeError):
            self.publish(install_dir=fresh)
        self.assertFalse(fresh.exists())


class TestPythonPackageBoundary(PackageFixture):
    def test_code_package_root_collects_nested_modules_only(self):
        package_root = self.declaration.parent
        (package_root / "sub").mkdir()
        (package_root / "sub/util.py").write_text("value = 1\n", encoding="utf-8")
        (package_root / "sub/data.bin").write_bytes(b"data")
        (package_root / "notes.md").write_text("notes", encoding="utf-8")
        sibling = self.project / "assets/tools/custom/Math/Other"
        sibling.mkdir()
        (sibling / "experiment.py").write_text("print('not shipped')\n", encoding="utf-8")
        archive = self.publish()
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            self.assertIn(f"{package_root.relative_to(self.project).as_posix()}/sub/util.py", names)
            self.assertFalse(any("experiment.py" in name for name in names))
            self.assertFalse(any(name.endswith("notes.md") for name in names))
            self.assertFalse(any(name.endswith("sub/data.bin") for name in names))

    def test_declared_resource_directory_is_collected_without_caches(self):
        data = self.declaration.parent / "data"
        (data / "nested").mkdir(parents=True)
        (data / "nested/value.txt").write_text("value", encoding="utf-8")
        (data / "__pycache__").mkdir()
        (data / "__pycache__/value.pyc").write_bytes(b"cache")
        self.document["resources"] = ["bias.dat", "data"]
        archive = self.publish()
        with zipfile.ZipFile(archive) as bundle:
            prefix = self.declaration.parent.relative_to(self.project).as_posix()
            self.assertIn(f"{prefix}/data/nested/value.txt", bundle.namelist())
            self.assertFalse(any("__pycache__" in name or name.endswith(".pyc") for name in bundle.namelist()))

    def test_shared_package_root_is_deduplicated(self):
        second = self.declaration.parent / "second.json"
        second_document = dict(self.document)
        second_document["tool_id"] = "custom.math.second"
        second_document["display_name"] = "Math Second"
        second_document["entry"] = {"file": "tool.py", "symbol": "run"}
        package.save_json(second_document, second)
        self.index["tools"].append({"definition": second.relative_to(self.project).as_posix()})
        archive = self.publish()
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            self.assertEqual(len(names), len(set(names)), "shared package files must be copied once")

    def test_same_destination_different_bytes_is_a_conflict(self):
        first = self.project / "first.py"
        second = self.project / "second.py"
        first.write_text("one", encoding="utf-8")
        second.write_text("two", encoding="utf-8")
        plan = package.CopyPlan()
        plan.add(first, Path("shared/module.py"))
        with self.assertRaisesRegex(RuntimeError, "冲突"):
            plan.add(second, Path("shared/module.py"))

    def test_same_destination_same_bytes_is_deduplicated(self):
        source = self.project / "same.py"
        source.write_text("same", encoding="utf-8")
        plan = package.CopyPlan()
        plan.add(source, Path("shared/module.py"))
        plan.add(source, Path("shared/module.py"))
        self.assertEqual(len(plan.items()), 1)


if __name__ == "__main__":
    unittest.main()
