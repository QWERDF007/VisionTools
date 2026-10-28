"""Multi-result expression tool: evaluates the compiled program table."""

from __future__ import annotations

from ..common import status
from ..common.engine import EngineError, Program
from .errors import EVALUATION_FAILED, INITIALIZE_FAILED, NOT_INITIALIZED, PROGRAM_INVALID


class ExpressionTool:
    def __init__(self):
        self._program: Program | None = None

    def initialize(self, params):
        try:
            self._program = Program.from_tables(params.get("variables", []), params.get("program", []))
        except EngineError as error:
            self._program = None
            return status.failure(PROGRAM_INVALID, str(error))
        except Exception as error:  # noqa: BLE001 - the detail carries the technical cause
            self._program = None
            return status.failure(INITIALIZE_FAILED, "表达式初始化失败", str(error))
        if not self._program.results:
            self._program = None
            return status.failure(PROGRAM_INVALID, "表达式表为空")
        return status.success()

    def execute(self, inputs):
        if self._program is None:
            return self._failure(status.failure(NOT_INITIALIZED, "表达式未初始化"))
        try:
            values = self._program.evaluate(dict(inputs))
        except EngineError as error:
            return self._failure(status.failure(EVALUATION_FAILED, str(error)))
        except Exception as error:  # noqa: BLE001
            return self._failure(status.failure(EVALUATION_FAILED, "表达式求值失败", str(error)))
        outputs = {}
        for port in self._program.exposed_ports():
            outputs[port] = values.get(port)
        return status.executed(status.success(), outputs)

    def _failure(self, failure: dict):
        outputs = {}
        if self._program is not None:
            for port in self._program.exposed_ports():
                outputs[port] = None
        return status.executed(failure, outputs)

    def release(self):
        self._program = None
        return status.success()
