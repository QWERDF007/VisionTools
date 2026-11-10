#!/usr/bin/env python3
"""Runs the distribution tests the way a release has to run them.

The suite contains one test that only the real host can answer: it deploys the
archive and executes the packaged tool through VisionRT's bridge. Without the
three paths below that test skips, and a green run would then say nothing about
the thing most likely to break. This wrapper supplies the paths and sets
``VISIONTOOLS_REQUIRE_HOST_EXECUTION=1`` so a missing piece is a failure, not a
skip.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def default(path: str) -> Path:
    return REPO_ROOT.parent / "VisionRT" / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--consumer", type=Path, default=default("build/bin/python_tools_execution_test.exe"),
                        help="VisionRT 的 --package 消费者")
    parser.add_argument("--deploy-script", type=Path, default=default("tools/package_app.py"),
                        help="把 ZIP 部署成 tools-package 的脚本")
    parser.add_argument("--validator", type=Path, default=default("build/bin/visionrt_validate_tool.exe"),
                        help="工具声明校验器")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="运行测试的解释器")
    parser.add_argument("tests", nargs="*", help="传给 unittest 的用例名（默认整个发行套件）")
    args = parser.parse_args(argv)

    missing = [str(path) for path in (args.consumer, args.deploy_script, args.validator) if not path.is_file()]
    if missing:
        print("release check cannot run; the host execution inputs are missing:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        print("build VisionRT first, or pass --consumer/--deploy-script/--validator.", file=sys.stderr)
        return 2

    environment = os.environ.copy()
    environment["VISIONTOOLS_REQUIRE_HOST_EXECUTION"] = "1"
    environment["VISIONRT_PYTHON_CONSUMER"] = str(args.consumer)
    environment["VISIONRT_DEPLOY_SCRIPT"] = str(args.deploy_script)
    environment["VISIONTOOLS_VALIDATOR"] = str(args.validator)

    command = [str(args.python), "-B", "-m", "unittest", "-v"]
    command.extend(args.tests or ["tests.test_distribution"])
    print("host execution is required for this run:", file=sys.stderr)
    print(f"  consumer      {args.consumer}", file=sys.stderr)
    print(f"  deploy script {args.deploy_script}", file=sys.stderr)
    print(f"  validator     {args.validator}", file=sys.stderr)
    completed = subprocess.run(command, cwd=REPO_ROOT, env=environment)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
