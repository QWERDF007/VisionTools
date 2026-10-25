"""Condition tool: one Bool formula producing two mutually exclusive outputs."""

from __future__ import annotations

from ..common.engine import EngineError, Program


class BranchTool:
    def __init__(self):
        self._program: Program | None = None

    def initialize(self, params):
        try:
            self._program = Program.from_tables(params.get("variables", []), params.get("program", []),
                                                condition=True)
        except EngineError as error:
            self._program = None
            return {"__status_code": 1, "__status_message": str(error)}
        except Exception:  # noqa: BLE001
            self._program = None
            return {"__status_code": 1, "__status_message": "条件初始化失败"}
        return {"__status_code": 0, "__status_message": "运行成功"}

    def execute(self, inputs):
        if self._program is None:
            return self._failure("条件未初始化")
        try:
            values = self._program.evaluate(dict(inputs))
        except EngineError as error:
            return self._failure(str(error))
        except Exception as error:  # noqa: BLE001
            return self._failure(f"条件求值失败：{error}")
        result = self._program.results[0]
        value = values.get(result.port_id)
        if type(value) is not bool:
            return self._failure("条件结果必须是布尔值")
        return {"__status_code": 0, "__status_message": "运行成功", "on_true": value, "on_false": not value}

    def _failure(self, message: str):
        return {"__status_code": 1, "__status_message": message, "on_true": None, "on_false": None}

    def release(self):
        self._program = None
        return {"__status_code": 0, "__status_message": ""}
