#!/usr/bin/env python3
"""Check git workspace status using stable NUL-delimited porcelain interface (S47-02).

Outputs:
COMMIT=<40-char SHA or unknown>
BRANCH=<branch name or unknown>
DIRTY=<false|true|unknown>
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Tuple, Union


def is_whitelisted(path_str: str) -> bool:
    norm = path_str.replace("\\", "/").strip()
    if not norm:
        return True
    if norm == "WORKLOG.md" or norm.endswith("/WORKLOG.md"):
        return True
    if re.match(r"^docs/R[0-9].*\.md$", norm):
        return True
    if re.match(r"^docs/VisionRT_.*\.zip$", norm):
        return True
    return False


def check_git_workspace(
    repo_dir: Union[str, Path],
    git_cmd: str = "git",
) -> Tuple[str, str, Union[bool, str]]:
    repo_path = Path(repo_dir).resolve()
    if not repo_path.is_dir():
        return "unknown", "unknown", "unknown"

    # 1. Query commit
    try:
        res_commit = subprocess.run(
            [git_cmd, "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown", "unknown", "unknown"

    if res_commit.returncode != 0 or not res_commit.stdout.strip():
        return "unknown", "unknown", "unknown"
    commit = res_commit.stdout.strip()

    # 2. Query branch
    try:
        res_branch = subprocess.run(
            [git_cmd, "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        branch = res_branch.stdout.strip() if res_branch.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        branch = "unknown"

    # 3. Query status using NUL-delimited porcelain interface (-z)
    try:
        res_status = subprocess.run(
            [git_cmd, "status", "--porcelain", "-z", "-uall"],
            cwd=repo_path,
            capture_output=True,
        )
    except (OSError, subprocess.SubprocessError):
        return commit, branch, "unknown"

    if res_status.returncode != 0:
        return commit, branch, "unknown"

    raw = res_status.stdout
    if not raw:
        return commit, branch, False

    tokens = raw.split(b"\x00")
    dirty = False
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token:
            i += 1
            continue
        status_prefix = token[:3]
        path1 = token[3:]
        # Check if status code represents rename or copy
        is_rename = status_prefix.startswith((b"R", b"C")) or (
            len(status_prefix) >= 2 and status_prefix[1:2] in (b"R", b"C")
        )
        paths = [path1]
        if is_rename and i + 1 < len(tokens):
            i += 1
            paths.append(tokens[i])
        i += 1

        for p_bytes in paths:
            p_str = p_bytes.decode("utf-8", errors="replace").strip()
            if not is_whitelisted(p_str):
                dirty = True
                break
        if dirty:
            break

    return commit, branch, dirty


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: check_git_workspace.py <repo_dir> [git_cmd]", file=sys.stderr)
        return 1

    repo = sys.argv[1]
    git = sys.argv[2] if len(sys.argv) > 2 else "git"
    commit, branch, dirty = check_git_workspace(repo, git)

    dirty_str = "unknown" if dirty == "unknown" else ("true" if dirty else "false")
    print(f"COMMIT={commit}")
    print(f"BRANCH={branch}")
    print(f"DIRTY={dirty_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
