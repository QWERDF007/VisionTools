"""Packaging adapter for the shared native Git workspace policy."""

import json
import os
from pathlib import Path
import subprocess
import tempfile


def check_git_workspace(repo_dir, git_cmd="git", *, cmake_cmd=None):
    script = Path(__file__).resolve().parents[1] / "3rdparty/VisionRTContracts/cmake/GitWorkspace.cmake"
    with tempfile.TemporaryDirectory(prefix="git-workspace-") as temporary:
        output = Path(temporary) / "workspace.json"
        try:
            subprocess.run([
                cmake_cmd or os.environ.get("CMAKE_COMMAND", "cmake"),
                f"-DREPO_DIR={Path(repo_dir).resolve().as_posix()}",
                f"-DGIT_EXECUTABLE={git_cmd}", f"-DOUTPUT={output.as_posix()}",
                "-P", str(script),
            ], check=True, capture_output=True)
            result = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, subprocess.SubprocessError, ValueError):
            return "unknown", "unknown", "unknown"
    return result["commit"], result["branch"], result["dirty"]
