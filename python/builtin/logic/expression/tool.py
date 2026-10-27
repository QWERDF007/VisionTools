"""Multi-result expression tool: evaluates the compiled program table."""

from __future__ import annotations

import sys

from ..common.engine import EngineError, Program


class ExpressionTool:
    def __init__(self):
        self._program: Program | None = None

    def initialize(self, params):
        try:
            self._program = Program.from_tables(params.get("variables", []), params.get("program", []))
        except EngineError as error:
            self._program = None
            return {"__status_code": 1, "__status_message": str(error)}
        except Exception as error:  # noqa: BLE001 - surface the real cause once
            self._program = None
            print(f"expression initialize failed: {error}", file=sys.stderr)
            return {"__status_code": 1, "__status_message": "表达式初始化失败"}
        if not self._program.results:
            return {"__status_code": 1, "__status_message": "表达式表为空"}
        return {"__status_code": 0, "__status_message": "运行成功"}

    def execute(self, inputs):
        if self._program is None:
            return {"__status_code": 1, "__status_message": "表达式未初始化"}
        try:
            values = self._program.evaluate(dict(inputs))
        except EngineError as error:
            return self._failure(str(error))
        except Exception as error:  # noqa: BLE001
            return self._failure(f"表达式求值失败：{error}")
        result = {"__status_code": 0, "__status_message": "运行成功"}
        for port in self._program.exposed_ports():
            result[port] = values.get(port)
        return result

    def _failure(self, message: str):
        result = {"__status_code": 1, "__status_message": message}
        if self._program is not None:
            for port in self._program.exposed_ports():
                result[port] = None
        return result

    def release(self):
        self._program = None
        return {"__status_code": 0, "__status_message": ""}
