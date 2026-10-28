"""Condition tool: one Bool result from a formula or an upstream Bool, with mutually exclusive branches."""

from __future__ import annotations

from ..common import status
from ..common.engine import EngineError, Program
from .errors import (
    CONDITION_NOT_BOOLEAN,
    EVALUATION_FAILED,
    INITIALIZE_FAILED,
    MODE_INVALID,
    NOT_INITIALIZED,
)

FORMULA = "formula"
UPSTREAM = "upstream"


class BranchTool:
    def __init__(self):
        self._program: Program | None = None
        self._mode: str | None = None

    def initialize(self, params):
        mode = params.get("condition_mode", FORMULA)
        if mode not in (FORMULA, UPSTREAM):
            self._program = None
            self._mode = None
            return status.failure(MODE_INVALID, f"未知的条件来源模式：{mode}")
        self._mode = mode
        if mode == UPSTREAM:
            # The upstream mode reads one Bool; no AST is compiled or evaluated.
            self._program = None
            return status.success()
        try:
            self._program = Program.from_tables(params.get("variables", []), params.get("program", []),
                                                condition=True)
        except EngineError as error:
            self._program = None
            return status.failure(MODE_INVALID, str(error))
        except Exception as error:  # noqa: BLE001
            self._program = None
            return status.failure(INITIALIZE_FAILED, "条件初始化失败", str(error))
        return status.success()

    def execute(self, inputs):
        if self._mode == UPSTREAM:
            value = inputs.get("condition")
            if type(value) is not bool:
                return self._failure(status.failure(CONDITION_NOT_BOOLEAN, "上层条件必须是布尔值"))
            return self._branch(value)
        if self._mode != FORMULA or self._program is None:
            return self._failure(status.failure(NOT_INITIALIZED, "条件未初始化"))
        try:
            values = self._program.evaluate(dict(inputs))
        except EngineError as error:
            return self._failure(status.failure(EVALUATION_FAILED, str(error)))
        except Exception as error:  # noqa: BLE001
            return self._failure(status.failure(EVALUATION_FAILED, "条件求值失败", str(error)))
        result = self._program.results[0]
        value = values.get(result.port_id)
        if type(value) is not bool:
            return self._failure(status.failure(CONDITION_NOT_BOOLEAN, "条件结果必须是布尔值"))
        return self._branch(value)

    def _branch(self, value: bool):
        outputs = {
            "on_true": value,
            "on_false": not value,
            "on_true_status": 0 if value else 1,
            "on_false_status": 1 if value else 0,
        }
        return status.executed(status.success(), outputs)

    def _failure(self, failure: dict):
        outputs = {
            "on_true": None,
            "on_false": None,
            "on_true_status": 1,
            "on_false_status": 1,
        }
        return status.executed(failure, outputs)

    def release(self):
        self._program = None
        self._mode = None
        return status.success()
