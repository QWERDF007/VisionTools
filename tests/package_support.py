import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


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
            "schema_version": 2, "tool_id": "custom.math", "display_name": "Math",
            "category": "Mathematics", "subcategory": "Arithmetic", "version": "1.0.0",
            "kind": "python_function", "entry": {"package": "examples/math", "file": "tool.py", "symbol": "run"},
            "resources": ["bias.dat"], "init_params": [], "inputs": [],
            "outputs": [{"id": "value", "name": "Value", "type": "Int"}],
        }
        self.code = self.project / "python/custom/examples/math"
        self.code.mkdir(parents=True)
        (self.code / "tool.py").write_text(
            'from .helper import calculate\ndef run(inputs): return {"value": calculate()}\n', encoding="utf-8")
        (self.code / "helper.py").write_text(
            'from pathlib import Path\ndef calculate(): return int(Path(__file__).with_name("bias.dat").read_text())\n',
            encoding="utf-8")
        (self.code / "bias.dat").write_text("42", encoding="utf-8")
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

    def link_directory(self, target, link):
        target.mkdir(parents=True, exist_ok=True)
        link.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
                           capture_output=True, check=True)
            self.addCleanup(os.rmdir, link)
        else:
            link.symlink_to(target, target_is_directory=True)
            self.addCleanup(link.unlink)
