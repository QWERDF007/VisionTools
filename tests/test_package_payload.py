import json
from pathlib import Path
import shutil
import unittest
import zipfile

from package_support import PackageFixture, package


class TestPackagePayload(PackageFixture):
    def test_package_can_select_the_origin_root(self):
        self.document["entry"]["package"] = "."
        self.document["entry"]["file"] = "examples/math/tool.py"
        self.document["resources"] = ["examples/math/bias.dat"]
        archive = self.publish()
        with zipfile.ZipFile(archive) as bundle:
            self.assertIn("python/custom/examples/math/tool.py", bundle.namelist())
            self.assertNotIn("assets/", bundle.namelist())

    def test_same_package_name_is_scoped_to_its_origin(self):
        declaration = self.project / "assets/tools/builtin/math/tool.json"
        document = dict(self.document, tool_id="builtin.math", resources=[])
        package.save_json(document, declaration)
        self.index["tools"].append({"definition": declaration.relative_to(self.project).as_posix()})
        code = self.project / "python/builtin/examples/math/tool.py"
        code.parent.mkdir(parents=True)
        code.write_text('def run(inputs): return {"value": 43}\n', encoding="utf-8")
        archive = self.publish()
        with zipfile.ZipFile(archive) as bundle:
            self.assertIn(b'43', bundle.read("python/builtin/examples/math/tool.py"))
            self.assertIn(b'calculate()', bundle.read("python/custom/examples/math/tool.py"))
            document = json.loads(bundle.read("tools/builtin/math/tool.json"))
            self.assertNotIn("origin", document)
            self.assertEqual(document["entry"]["package"], "examples/math")

    def test_missing_custom_script_never_falls_back_to_builtin(self):
        code = self.project / "python/builtin/examples/math/tool.py"
        code.parent.mkdir(parents=True)
        shutil.copy2(self.code / "tool.py", code)
        (self.code / "tool.py").unlink()
        with self.assertRaisesRegex(package.PackagingError, "入口脚本不存在"):
            self.publish()
        self.assertFalse(self.output.exists())

    def test_entry_directory_link_cannot_escape_selected_package(self):
        outside = self.project / "python/custom/other"
        self.link_directory(outside, self.code / "entry")
        (outside / "tool.py").write_text("raise RuntimeError('not imported')", encoding="utf-8")
        self.document["entry"]["file"] = "entry/tool.py"
        with self.assertRaisesRegex(package.PackagingError, "逃逸"):
            self.publish()

    def test_helper_directory_link_cannot_escape_selected_package(self):
        outside = self.project / "python/custom/other"
        self.link_directory(outside, self.code / "helpers")
        (outside / "helper.py").write_text("raise RuntimeError('not imported')", encoding="utf-8")
        with self.assertRaisesRegex(package.PackagingError, "逃逸"):
            self.publish()

    def test_resource_directory_link_cannot_escape_selected_package(self):
        outside = self.project / "python/custom/other"
        self.link_directory(outside, self.code / "data")
        (outside / "bias.dat").write_text("99", encoding="utf-8")
        self.document["resources"] = ["data/bias.dat"]
        with self.assertRaisesRegex(package.PackagingError, "包根目录"):
            self.publish()

    def test_internal_directory_alias_preserves_module_names(self):
        helpers = self.code / "helpers"
        self.link_directory(helpers, self.code / "alias")
        (helpers / "helper.py").write_text("value = 42", encoding="utf-8")
        archive = self.publish()
        with zipfile.ZipFile(archive) as bundle:
            self.assertEqual(bundle.read("python/custom/examples/math/alias/helper.py"), b"value = 42")
            self.assertEqual(bundle.read("python/custom/examples/math/helpers/helper.py"), b"value = 42")

    def test_directory_link_cycle_is_rejected(self):
        self.link_directory(self.code, self.code / "cycle")
        with self.assertRaisesRegex(package.PackagingError, "循环"):
            self.publish()

    def test_native_python_helpers_are_collected_without_import(self):
        (self.code / "native.cp312-win_amd64.pyd").write_bytes(b"test-pyd")
        (self.code / "native.cpython-312-x86_64-linux-gnu.so").write_bytes(b"test-so")
        archive = self.publish()
        with zipfile.ZipFile(archive) as bundle:
            self.assertEqual(bundle.read("python/custom/examples/math/native.cp312-win_amd64.pyd"), b"test-pyd")
            self.assertEqual(bundle.read("python/custom/examples/math/native.cpython-312-x86_64-linux-gnu.so"), b"test-so")

    def test_declarations_must_be_in_an_authored_origin_root(self):
        declaration = self.project / "elsewhere/tool.json"
        package.save_json(self.document, declaration)
        self.index["tools"] = [{"definition": "elsewhere/tool.json"}]
        with self.assertRaisesRegex(package.PackagingError, "assets/tools"):
            self.publish()

    def test_derived_json_uses_the_same_destination_conflict_check(self):
        source = self.project / "source.json"
        source.write_bytes(b"{\"value\":42}")
        plan = package.CopyPlan()
        plan.add(source, Path("tools/custom/value.json"))
        plan.add(b'{"value":42}', Path("tools/custom/value.json"))
        self.assertEqual(len(plan.items()), 1)
        with self.assertRaisesRegex(package.PackagingError, "冲突"):
            plan.add(b'{"value":43}', Path("tools/custom/value.json"))


if __name__ == "__main__":
    unittest.main()
