"""Status envelope shared by the Python tools.

execute() answers ``{"status": {...}, "outputs": {...}}``; initialize() and release() answer a
bare status. Codes are the tool's own positive business codes: the host turns the status into the
two status output ports and never reads a status key out of ``outputs``.
"""

from __future__ import annotations

# Codes every Python tool can report for a port the host did not deliver.
INPUT_MISSING = 1001
INPUT_TYPE_MISMATCH = 1002
INPUT_OUT_OF_RANGE = 1003
OPERATOR_FAILED = 1005


def success(message: str = "运行成功") -> dict:
    return {"code": 0, "message": message, "detail": ""}


def failure(code: int, message: str, detail: str = "") -> dict:
    return {"code": int(code), "message": message, "detail": detail}


def executed(status: dict, outputs: dict) -> dict:
    res = dict(outputs)
    res["__status_code"] = int(status.get("code", 0))
    res["__status_message"] = str(status.get("message", ""))
    return {"outputs": res, "detail": str(status.get("detail", ""))}
