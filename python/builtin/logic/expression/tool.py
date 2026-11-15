"""Multi-result expression tool: evaluates the compiled program table."""

from __future__ import annotations

from ..visiontools_sdk import status as lifecycle_status, result as tool_result
from ..common.engine import EngineError, Program
from .errors import EVALUATION_FAILED, INITIALIZE_FAILED, NOT_INITIALIZED, PROGRAM_INVALID

VARIABLES, PROGRAM = 0, 1


class ExpressionTool:
    def __init__(self):
        self._program: Program | None = None

    def initialize(self, params):
        try:
            self._program = Program.from_tables(params.get(VARIABLES, []), params.get(PROGRAM, []))
        except EngineError as error:
            self._program = None
            return lifecycle_status(PROGRAM_INVALID, str(error))
        except Exception as error:  # noqa: BLE001 - the detail carries the technical cause
            self._program = None
            return lifecycle_status(INITIALIZE_FAILED, "表达式初始化失败", str(error))
        if not self._program.results:
            self._program = None
            return lifecycle_status(PROGRAM_INVALID, "表达式表为空")
        return lifecycle_status(message="运行成功")

    def execute(self, inputs):
        if self._program is None:
            return self._failure(lifecycle_status(NOT_INITIALIZED, "表达式未初始化"))
        try:
            values = self._program.evaluate(dict(inputs))
        except EngineError as error:
            return self._failure(lifecycle_status(EVALUATION_FAILED, str(error)))
        except Exception as error:  # noqa: BLE001
            return self._failure(lifecycle_status(EVALUATION_FAILED, "表达式求值失败", str(error)))
        outputs = {}
        for port in self._program.exposed_ports():
            outputs[port] = values.get(port)
        return tool_result(outputs, message="运行成功")

    def _failure(self, failure: dict):
        outputs = {}
        if self._program is not None:
            for port in self._program.exposed_ports():
                outputs[port] = None
        return tool_result(outputs, failure["code"], failure["message"], failure["detail"])

    def release(self):
        self._program = None
        return lifecycle_status(message="运行成功")
