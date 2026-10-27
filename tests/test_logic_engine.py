"""Contract for the shared restricted expression engine and both Python tools.

The engine is imported from the tool package root so this test exercises exactly
the code that ships inside the tools ZIP.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "python" / "builtin"
sys.path.insert(0, str(PACKAGE_ROOT))

from logic.branch.tool import BranchTool  # noqa: E402
from logic.common.engine import EngineError, Program  # noqa: E402
from logic.expression.tool import ExpressionTool  # noqa: E402


def tables(variables, program):
    return ([
        [port_id, symbol, type_name, dtype, str(rank), "true" if allow_missing else "false"]
        for port_id, symbol, type_name, dtype, rank, allow_missing in variables
    ], [
        [port_id, symbol, expression, type_name, dtype, str(rank), "true" if exposed else "false"]
        for port_id, symbol, expression, type_name, dtype, rank, exposed in program
    ])


class EngineSemantics(unittest.TestCase):
    def program(self, variables, program, **kwargs):
        variable_rows, program_rows = tables(variables, program)
        return Program.from_tables(variable_rows, program_rows, **kwargs)

    def test_true_and_floor_division(self):
        compiled = self.program([], [
            ("p1", "half", "5 / 2", "Float", "", -1, True),
            ("p2", "whole", "5 // 2", "Int", "", -1, True),
        ])
        values = compiled.evaluate({})
        self.assertEqual(values["p1"], 2.5)
        self.assertEqual(values["p2"], 2)

    def test_slicing_and_lists(self):
        compiled = self.program([("p1", "data", "Array", "int32", 1, False)], [
            ("p2", "window", "data[1:3]", "Array", "int32", 1, True),
            ("p3", "literal", "[1, 2, 3]", "Array", "int32", 1, True),
        ])
        values = compiled.evaluate({"p1": [10, 20, 30, 40]})
        self.assertEqual(values["p2"], [20, 30])
        self.assertEqual(values["p3"], [1, 2, 3])

    def test_vector_helpers_and_empty_sets(self):
        compiled = self.program([("p1", "radii", "Array", "float64", 1, False)], [
            ("p2", "ok", "all_nonempty(v_gt(radii, 5.0))", "Bool", "", -1, True),
            ("p3", "empty", "is_empty(radii)", "Bool", "", -1, True),
        ])
        self.assertTrue(compiled.evaluate({"p1": [6.0, 7.0]})["p2"])
        empty = compiled.evaluate({"p1": []})
        self.assertFalse(empty["p2"])
        self.assertTrue(empty["p3"])

    def test_missing_state_is_explicit(self):
        compiled = self.program([("p1", "maybe", "Float", "", -1, True)], [
            ("p2", "absent", "is_missing(maybe)", "Bool", "", -1, True),
        ])
        self.assertTrue(compiled.evaluate({"p1": None})["p2"])
        arithmetic = self.program([("p1", "maybe", "Float", "", -1, True)], [
            ("p2", "value", "maybe + 1", "Float", "", -1, True),
        ])
        with self.assertRaises(EngineError):
            arithmetic.evaluate({"p1": None})

    def test_division_by_zero_reports_error(self):
        compiled = self.program([], [("p1", "bad", "1 / 0", "Float", "", -1, True)])
        with self.assertRaises(EngineError) as context:
            compiled.evaluate({})
        self.assertIn("除数为零", str(context.exception))

    def test_attribute_and_method_calls_are_rejected(self):
        compiled = self.program([("p1", "text", "String", "", -1, False)], [
            ("p2", "upper", "text.upper()", "String", "", -1, True),
        ])
        with self.assertRaises(EngineError):
            compiled.evaluate({"p1": "abc"})

    def test_forward_reference_rejected(self):
        variable_rows, program_rows = tables([], [
            ("p1", "first", "later + 1", "Int", "", -1, True),
            ("p2", "later", "2", "Int", "", -1, True),
        ])
        with self.assertRaises(EngineError):
            Program.from_tables(variable_rows, program_rows)

    def test_strict_output_types(self):
        compiled = self.program([], [("p1", "count", "2.5", "Int", "", -1, True)])
        with self.assertRaises(EngineError):
            compiled.evaluate({})
        promoted = self.program([], [("p1", "value", "2", "Float", "", -1, True)])
        self.assertEqual(promoted.evaluate({})["p1"], 2.0)

    def test_expression_length_limit(self):
        long_expression = "1" + " + 1" * 2000
        with self.assertRaises(EngineError):
            self.program([], [("p1", "value", long_expression, "Int", "", -1, True)])

    def test_cumulative_element_budget(self):
        rows = [[0] * 100 for _ in range(1000)]  # 100000 leaves is the exact boundary
        compiled = self.program([("p1", "big", "Array", "int32", 2, False)], [
            ("p2", "total", "len(big)", "Int", "", -1, True),
        ])
        self.assertEqual(compiled.evaluate({"p1": rows})["p2"], 1000)
        overflow = [[0] * 101 for _ in range(1000)]  # 101000 leaves exceeds it
        with self.assertRaises(EngineError) as context:
            compiled.evaluate({"p1": overflow})
        self.assertIn("累计元素上限", str(context.exception))

    def test_cumulative_budget_spans_results(self):
        rows = [[0] * 100 for _ in range(500)]
        compiled = self.program([("p1", "data", "Array", "int32", 2, False)], [
            ("p2", "first", "data", "Array", "int32", 2, True),
            ("p3", "second", "data", "Array", "int32", 2, True),
        ])
        with self.assertRaises(EngineError) as context:
            compiled.evaluate({"p1": rows})
        self.assertIn("累计元素上限", str(context.exception))

    def test_concat_checks_size_before_allocating(self):
        rows = [0] * 60000
        compiled = self.program([("p1", "data", "Array", "int32", 1, False)], [
            ("p2", "joined", "concat(data, data)", "Array", "int32", 1, True),
        ])
        with self.assertRaises(EngineError) as context:
            compiled.evaluate({"p1": rows})
        self.assertIn("超出元素数量上限", str(context.exception))

    def test_float32_representable_range(self):
        compiled = self.program([], [("p1", "value", "[1e100]", "Array", "float32", 1, True)])
        with self.assertRaises(EngineError) as context:
            compiled.evaluate({})
        self.assertIn("float32 可表示范围", str(context.exception))
        accepted = self.program([], [("p1", "value", "[1e30, -2.5]", "Array", "float32", 1, True)])
        self.assertEqual(accepted.evaluate({})["p1"], [1e30, -2.5])


class ToolContracts(unittest.TestCase):
    def test_expression_tool_returns_exposed_ports_only(self):
        tool = ExpressionTool()
        variable_rows, program_rows = tables([("p1", "radii", "Array", "float64", 1, False)], [
            ("p2", "count", "len(radii)", "Int", "", -1, True),
            ("p3", "hidden", "sum(radii)", "Float", "", -1, False),
        ])
        status = tool.initialize({"variables": variable_rows, "program": program_rows})
        self.assertEqual(status["__status_code"], 0)
        result = tool.execute({"p1": [1.0, 2.0]})
        self.assertEqual(result["__status_code"], 0)
        self.assertEqual(result["p2"], 2)
        self.assertNotIn("p3", result)
        failure = tool.execute({"p1": None})
        self.assertNotEqual(failure["__status_code"], 0)
        self.assertIsNone(failure["p2"])

    def test_branch_tool_is_mutually_exclusive(self):
        tool = BranchTool()
        variable_rows, program_rows = tables([("p1", "count", "Int", "", -1, False)], [
            ("p2", "ok", "count > 3", "Bool", "", -1, False),
        ])
        self.assertEqual(tool.initialize({"variables": variable_rows, "program": program_rows})["__status_code"], 0)
        yes = tool.execute({"p1": 5})
        self.assertTrue(yes["on_true"])
        self.assertFalse(yes["on_false"])
        no = tool.execute({"p1": 1})
        self.assertFalse(no["on_true"])
        self.assertTrue(no["on_false"])

    def test_branch_accepts_only_true_bool(self):
        tool = BranchTool()
        variable_rows, program_rows = tables([("p1", "count", "Int", "", -1, False)], [
            ("p2", "ok", "count", "Bool", "", -1, False),
        ])
        status = tool.initialize({"variables": variable_rows, "program": program_rows})
        self.assertEqual(status["__status_code"], 0)
        result = tool.execute({"p1": 1})
        self.assertNotEqual(result["__status_code"], 0)
        self.assertIsNone(result["on_true"])
        self.assertIsNone(result["on_false"])

    def test_branch_requires_single_hidden_result(self):
        tool = BranchTool()
        variable_rows, program_rows = tables([], [("p1", "ok", "True", "Bool", "", -1, True)])
        status = tool.initialize({"variables": variable_rows, "program": program_rows})
        self.assertNotEqual(status["__status_code"], 0)


if __name__ == "__main__":
    unittest.main()
