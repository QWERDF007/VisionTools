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


def require_validator() -> Path:
    configured = os.environ.get("VISIONTOOLS_VALIDATOR")
    if not configured:
        raise AssertionError("VISIONTOOLS_VALIDATOR must point at the real SDK validator")
    path = Path(configured)
    if not path.is_file():
        raise AssertionError(f"VISIONTOOLS_VALIDATOR is not a file: {path}")
    return path


class DistributionContract(unittest.TestCase):
    def test_source_derived_output_requires_host_binding(self):
        declaration = yaml.safe_load((ROOT / "assets/tools/builtin/流程控制/跨流程接收.yaml").read_text(encoding="utf-8"))
        declaration["_source_path"] = "receive.yaml"
        bindings = {"modules": {"test": {"tools": {declaration["entry"]["registry_name"]: {}}}}}
        with self.assertRaisesRegex(generate_tool_contracts.ContractValidationError, "host_binding"):
            generate_tool_contracts.build_module_index([declaration], bindings)

    def test_declarations_use_tool_names_and_one_receiver(self):
        declarations = list((ROOT / "assets/tools").rglob("*.yaml"))
        receivers = []
        for path in declarations:
            declaration = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(path.stem, declaration["display_name"])
            if declaration["tool_id"].startswith("visionrt.flow.receive"):
                receivers.append(declaration)
        self.assertEqual([item["tool_id"] for item in receivers], ["visionrt.flow.receive"])
        self.assertIn("type_from", receivers[0]["outputs"][0])

    def test_receiver_uses_stable_source_identity(self):
        declaration = yaml.safe_load((ROOT / "assets/tools/builtin/流程控制/跨流程接收.yaml").read_text(encoding="utf-8"))
        parameters = {parameter["id"]: parameter for parameter in declaration["init_params"]}
        self.assertEqual(list(parameters), ["source_flow_id", "source_node_seq", "source_port_id", "timeout_ms"])
        for parameter_id in ("source_flow_id", "source_node_seq", "source_port_id"):
            self.assertEqual(parameters[parameter_id]["type"], "String")
            self.assertTrue(parameters[parameter_id]["name"])
        self.assertNotIn("source_flow_name", parameters)
        self.assertNotIn("source_node_name", parameters)
        self.assertEqual(declaration["outputs"][0]["type_from"],
                         {"flow": "source_flow_id", "node": "source_node_seq", "port": "source_port_id"})

    def test_receiver_declaration_passes_real_validator(self):
        validator = require_validator()
        result = subprocess.run([str(validator), str(ROOT / "assets/tools/builtin/流程控制/跨流程接收.yaml")],
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_old_receiver_fields_are_rejected_by_generator(self):
        validator = require_validator()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "old_receiver.yaml"
            path.write_text(
                "schema_version: 1\n"
                "tool_id: visionrt.flow.receive\n"
                "display_name: 跨流程接收\n"
                "version: 1.0.0\n"
                "kind: cpp_class\n"
                "entry:\n  registry_name: visionrt.flow.receive\n"
                "init_params:\n"
                "  - {id: source_flow_name, name: 源流程名称, type: String, default: ''}\n"
                "  - {id: source_node_name, name: 源节点名称, type: String, default: ''}\n"
                "  - {id: source_port_id, name: 源端口ID, type: String, default: ''}\n"
                "inputs: []\n"
                "outputs:\n"
                "  - id: value\n"
                "    name: 接收数据\n"
                "    type_from: {flow: source_flow_name, node: source_node_name, port: source_port_id}\n",
                encoding="utf-8")
            document = generate_tool_contracts.normalize_declaration(validator, path)
            with self.assertRaisesRegex(generate_tool_contracts.ContractValidationError, "source_flow_id"):
                generate_tool_contracts.validate_generation_rules(document, path)

    def test_normal_generation_indexes_python_and_platform_declarations(self):
        validator = require_validator()
        with tempfile.TemporaryDirectory() as temporary:
            generated = Path(temporary)
            result = subprocess.run([
                sys.executable, str(ROOT / "tools/generate_tool_contracts.py"),
                "--tools-dir", str(ROOT / "assets/tools"),
                "--bindings", str(ROOT / "tools/module_bindings.yaml"),
                "--output-dir", str(generated), "--index-output", str(generated / "module_index.yaml"),
                "--variant", "portable-release", "--modules", "visiontools_preprocess",
                "--validator", str(validator),
            ], capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(result.returncode, 0, result.stderr)
            index = yaml.safe_load((generated / "module_index.yaml").read_text(encoding="utf-8"))
            self.assertIn("custom.image.mean", index["tools"])
            self.assertNotIn("module", index["tools"]["custom.image.mean"])
            self.assertEqual(index["tools"]["visionrt.flow.receive"]["host_binding"], "visionrt.receiver")
            self.assertEqual(set(index["modules"]), {"visiontools_preprocess"})

    def test_python_package_uses_actual_manifest_without_backend_requirements(self):
        validator = require_validator()
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
                'def run(inputs): return {"__status_code": 0, "__status_message": "运行成功", '
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
            external_index = archive.parent / f"{archive.stem}.module_index.yaml"
            with zipfile.ZipFile(archive) as bundle:
                self.assertEqual(bundle.read("module_index.yaml"), external_index.read_bytes())
                installed = yaml.safe_load(bundle.read("module_index.yaml"))
                self.assertEqual(installed["variant"], "portable-release")
                self.assertEqual(installed["modules"], {})
                self.assertEqual(set(installed["tools"]), {"test.add"})
            result = subprocess.run([str(validator), "--package", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with tempfile.TemporaryDirectory() as extracted:
                with zipfile.ZipFile(archive) as bundle:
                    bundle.extractall(extracted)
                package_root = Path(extracted) / "assets/tools/custom/math/arithmetic"
                (package_root / "__init__.py").write_text("", encoding="utf-8")
                probe = subprocess.run([
                    sys.executable, "-B", "-c",
                    "import importlib.util, pathlib, sys\n"
                    "root = pathlib.Path(sys.argv[1])\n"
                    "spec = importlib.util.spec_from_file_location('add_pkg', root / '__init__.py',"
                    " submodule_search_locations=[str(root)])\n"
                    "pkg = importlib.util.module_from_spec(spec); sys.modules['add_pkg'] = pkg; spec.loader.exec_module(pkg)\n"
                    "spec = importlib.util.spec_from_file_location('add_pkg.add', root / 'add.py')\n"
                    "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
                    "print(module.run({})['value'])\n",
                    str(package_root),
                ], capture_output=True, text=True, encoding="utf-8", errors="replace")
                self.assertEqual(probe.returncode, 0, probe.stderr)
                self.assertEqual(probe.stdout.strip(), "42")
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
