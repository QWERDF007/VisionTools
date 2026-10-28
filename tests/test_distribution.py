import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import package_tools


def require_validator() -> Path:
    configured = os.environ.get("VISIONTOOLS_VALIDATOR")
    if not configured:
        raise AssertionError("VISIONTOOLS_VALIDATOR must point at the real host validator")
    path = Path(configured)
    if not path.is_file():
        raise AssertionError(f"VISIONTOOLS_VALIDATOR is not a file: {path}")
    return path


class DistributionContract(unittest.TestCase):
    def test_all_author_declarations_pass_the_host_json_reader(self):
        validator = require_validator()
        declarations = sorted((ROOT / "assets/tools").rglob("*.json"))
        catalog = package_tools.load_json(ROOT / "tools/package_catalog.json")
        self.assertEqual({path.relative_to(ROOT).as_posix() for path in declarations},
                         {entry["definition"] for entry in catalog["tools"]})
        self.assertTrue(all(path.suffix == ".json" for path in (ROOT / "assets").rglob("*") if path.is_file()))
        for path in declarations:
            with self.subTest(path=path):
                package_tools.load_json(path)
                result = subprocess.run([str(validator), "--emit-normalized-json", str(path)],
                                        capture_output=True, text=True, encoding="utf-8", errors="replace")
                self.assertEqual(result.returncode, 0, result.stderr)
                document = json.loads(result.stdout)
                self.assertNotIn("registry_name", document.get("entry", {}))
                self.assertTrue(all(not port["id"].startswith("__") for port in document["outputs"]))

    def test_receiver_uses_stable_identity_and_host_binding(self):
        declaration = json.loads((ROOT / "assets/tools/builtin/流程控制/跨流程接收.json").read_text(encoding="utf-8"))
        self.assertEqual(declaration["kind"], "host")
        self.assertEqual(declaration["entry"], {"binding": "visionrt.receiver"})
        parameters = {item["id"]: item for item in declaration["init_params"]}
        self.assertEqual(list(parameters), ["source_flow_id", "source_node_seq", "source_port_id", "timeout_ms"])
        for identifier in ("source_flow_id", "source_node_seq", "source_port_id"):
            self.assertEqual(parameters[identifier]["type"], "String")
        self.assertEqual(declaration["outputs"][0]["type_from"],
                         {"flow": "source_flow_id", "node": "source_node_seq", "port": "source_port_id"})

    def test_python_package_uses_actual_manifest_without_backend_requirements(self):
        validator = require_validator()
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            # A project that ships Python tools carries the one author SDK the packager injects.
            sdk = project / "python/author_sdk/visiontools_sdk"
            sdk.mkdir(parents=True)
            (sdk / "__init__.py").write_text("", encoding="utf-8")
            definition = project / "assets/tools/custom/math/arithmetic/add.json"
            definition.parent.mkdir(parents=True)
            document = {
                "schema_version": 4, "tool_id": "test.add", "display_name": "Add",
                "version": "1.0.0", "kind": "python_function",
                "entry": {"package": "math/arithmetic", "file": "add.py", "symbol": "run"},
                "init_params": [], "inputs": [],
                "outputs": [{"id": "value", "name": "Value", "type": "Int"}],
                "resources": ["bias.dat"],
            }
            definition.write_text(json.dumps(document), encoding="utf-8")
            code = project / "python/custom/math/arithmetic"
            code.mkdir(parents=True)
            (code / "add.py").write_text(
                'from .helper import value\nfrom pathlib import Path\n'
                'def run(inputs):\n'
                '    return {"status": {"code": 0, "message": "运行成功", "detail": ""},\n'
                '            "outputs": {"value": value + int(Path(__file__).with_name("bias.dat").read_text())}}\n',
                encoding="utf-8")
            (code / "helper.py").write_text("value = 37\n", encoding="utf-8")
            (code / "bias.dat").write_text("5", encoding="utf-8")
            build = project / "build"
            build.mkdir()
            (build / "Release").mkdir()
            catalog = {"schema_version": 1, "modules": [],
                       "tools": [{"definition": definition.relative_to(project).as_posix()}]}
            package_tools.save_json(catalog, project / "tools/package_catalog.json")
            manifest = {"variant": "portable-release", "config": "Release", "modules": {}, "runtime_files": []}
            (build / "Release/resolved_dependencies.json").write_text(json.dumps(manifest), encoding="utf-8")
            output = project / "dist/tools"
            archive = package_tools.package_tools(build, output, project_root=project, validator=validator)
            external_index = archive.parent / f"{archive.stem}.module_index.json"
            with zipfile.ZipFile(archive) as bundle:
                self.assertEqual(bundle.read("module_index.json"), external_index.read_bytes())
                installed = json.loads(bundle.read("module_index.json"))
                self.assertEqual(installed["variant"], "portable-release")
                self.assertEqual(installed["modules"], [])
                self.assertEqual({tool["tool_id"] for tool in installed["tools"]}, {"test.add"})
            result = subprocess.run([str(validator), "--package", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for source_root in (project / "assets", project / "python"):
                self.assertTrue(source_root.resolve().is_relative_to(project))
                shutil.rmtree(source_root)
            with tempfile.TemporaryDirectory() as extracted:
                with zipfile.ZipFile(archive) as bundle:
                    bundle.extractall(extracted)
                package_root = Path(extracted) / "python/custom/math/arithmetic"
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
                    "print(module.run({})['outputs']['value'])\n",
                    str(package_root),
                ], capture_output=True, text=True, encoding="utf-8", errors="replace")
                self.assertEqual(probe.returncode, 0, probe.stderr)
                self.assertEqual(probe.stdout.strip(), "42")
            consumer = os.environ.get("VISIONRT_PYTHON_CONSUMER")
            if consumer:
                deploy_script = Path(os.environ["VISIONRT_DEPLOY_SCRIPT"])
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
