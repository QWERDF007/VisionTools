import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import package_tools
import generate_tool_contracts


class DistributionContract(unittest.TestCase):
    def test_normal_generation_indexes_python_and_platform_declarations(self):
        validator = Path(os.environ.get("VISIONTOOLS_VALIDATOR", ROOT / "build/3rdparty/VisionRTSDK/visionrt_validate_tool.exe"))
        with tempfile.TemporaryDirectory() as temporary:
            generated = Path(temporary)
            result = subprocess.run([
                sys.executable, str(ROOT / "tools/generate_tool_contracts.py"),
                "--tools-dir", str(ROOT / "assets/tools"),
                "--bindings", str(ROOT / "tools/module_bindings.yaml"),
                "--output-dir", str(generated), "--index-output", str(generated / "module_index.yaml"),
                "--variant", "portable-release", "--modules", "visiontools_preprocess",
                "--sdk-include-dir", str(ROOT / "3rdparty/VisionRTSDK/include"),
                "--validator", str(validator),
            ], capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(result.returncode, 0, result.stderr)
            index = yaml.safe_load((generated / "module_index.yaml").read_text(encoding="utf-8"))
            self.assertIn("custom.image.mean", index["tools"])
            self.assertNotIn("module", index["tools"]["custom.image.mean"])
            self.assertEqual(index["tools"]["visionrt.flow.receive.int"]["host_binding"], "visionrt.receiver")
            self.assertEqual(set(index["modules"]), {"visiontools_preprocess"})

    def test_python_package_uses_actual_manifest_without_backend_requirements(self):
        validator = Path(os.environ.get("VISIONTOOLS_VALIDATOR", ROOT / "build/3rdparty/VisionRTSDK/visionrt_validate_tool.exe"))
        self.assertTrue(validator.is_file(), "Build the SDK validator before this contract")
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            definition = project / "assets/tools/custom/math/arithmetic/add.yaml"
            definition.parent.mkdir(parents=True)
            document = {
                "schema_version": 1, "tool_id": "test.add", "display_name": "Add",
                "version": "1.0.0", "kind": "python_function",
                "entry": {"file": "add.py", "symbol": "run"},
                "init_params": [], "inputs": [],
                "outputs": [{"id": "value", "name": "Value", "type": "Int"}],
                "resources": ["bias.dat"],
            }
            definition.write_text(yaml.safe_dump(document), encoding="utf-8")
            (definition.parent / "add.py").write_text(
                'from .helper import value\nfrom pathlib import Path\n'
                'def run(inputs): return {"__status_code": 0, "__status_message": "", '
                '"value": value + int(Path(__file__).with_name("bias.dat").read_text())}\n',
                encoding="utf-8")
            (definition.parent / "helper.py").write_text("value = 37\n", encoding="utf-8")
            (definition.parent / "bias.dat").write_text("5", encoding="utf-8")
            build = project / "build"
            (build / "generated").mkdir(parents=True)
            (build / "Release").mkdir()
            document["_source_path"] = str(definition)
            index = generate_tool_contracts.build_module_index(
                [document], {"modules": {}}, variant="portable-release", project_root=project, selected_modules=[])
            (build / "generated/module_index.yaml").write_text(yaml.safe_dump(index), encoding="utf-8")
            manifest = {"variant": "portable-release", "config": "Release", "modules": {}, "runtime_files": []}
            (build / "Release/resolved_dependencies.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
            output = project / "dist/tools"
            archive = package_tools.package_tools(build, output, project_root=project, validator=validator)
            with zipfile.ZipFile(archive) as bundle:
                self.assertEqual(bundle.read("module_index.yaml"), (archive.parent / "module_index.yaml").read_bytes())
                installed = yaml.safe_load(bundle.read("module_index.yaml"))
                self.assertEqual(installed["variant"], "portable-release")
                self.assertEqual(installed["modules"], {})
                self.assertEqual(set(installed["tools"]), {"test.add"})
            result = subprocess.run([str(validator), "--package", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            consumer = os.environ.get("VISIONRT_PYTHON_CONSUMER")
            if consumer:
                deploy_script = Path(os.environ["VISIONRT_DEPLOY_SCRIPT"])
                shutil.rmtree(project / "assets")
                installation = project / "installed"
                deployed = subprocess.run([
                    sys.executable, "-B", "-c",
                    "import runpy,sys; from pathlib import Path; api=runpy.run_path(sys.argv[1]); "
                    "api['deploy_tools_package'](Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))",
                    str(deploy_script), str(installation), str(archive), str(validator),
                ], capture_output=True, text=True, encoding="utf-8", errors="replace")
                self.assertEqual(deployed.returncode, 0, deployed.stderr)
                executed = subprocess.run([consumer, "--package", str(installation / "tools-package")],
                                          capture_output=True, text=True, encoding="utf-8", errors="replace")
                self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)
                self.assertIn("Pure Python ZIP result: 42", executed.stdout)


if __name__ == "__main__":
    unittest.main()
