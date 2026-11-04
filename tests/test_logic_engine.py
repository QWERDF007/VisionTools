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

try:  # The host interpreter ships NumPy; the engine itself stays importable without it.
    import numpy as np
except ImportError:  # pragma: no cover - only a NumPy-free interpreter takes this branch
    np = None


def _port(port_id):
    """Tests spell ports as p1/p2; the wire carries the canonical decimal digits."""
    number = int(port_id[1:]) if isinstance(port_id, str) and port_id.startswith("p") else int(port_id)
    return str(number)


def tables(variables, program):
    return ([
        [_port(port_id), symbol, type_name, dtype, str(rank), "true" if allow_missing else "false"]
        for port_id, symbol, type_name, dtype, rank, allow_missing in variables
    ], [
        [_port(port_id), symbol, expression, type_name, dtype, str(rank), "true" if exposed else "false"]
        for port_id, symbol, expression, type_name, dtype, rank, exposed in program
    ])


def compile_program(variables, program, **kwargs):
    variable_rows, program_rows = tables(variables, program)
    return Program.from_tables(variable_rows, program_rows, **kwargs)


class EngineSemantics(unittest.TestCase):
    def program(self, variables, program, **kwargs):
        return compile_program(variables, program, **kwargs)

    def test_true_and_floor_division(self):
        compiled = self.program([], [
            ("p1", "half", "5 / 2", "Float", "", -1, True),
            ("p2", "whole", "5 // 2", "Int", "", -1, True),
        ])
        values = compiled.evaluate({})
        self.assertEqual(values[1], 2.5)
        self.assertEqual(values[2], 2)

    def test_slicing_and_lists(self):
        compiled = self.program([("p1", "data", "Array", "int32", 1, False)], [
            ("p2", "window", "data[1:3]", "Array", "int32", 1, True),
            ("p3", "literal", "[1, 2, 3]", "Array", "int32", 1, True),
        ])
        values = compiled.evaluate({1: [10, 20, 30, 40]})
        self.assertEqual(values[2], [20, 30])
        self.assertEqual(values[3], [1, 2, 3])

    def test_vector_helpers_and_empty_sets(self):
        compiled = self.program([("p1", "radii", "Array", "float64", 1, False)], [
            ("p2", "ok", "all_nonempty(v_gt(radii, 5.0))", "Bool", "", -1, True),
            ("p3", "empty", "is_empty(radii)", "Bool", "", -1, True),
        ])
        self.assertTrue(compiled.evaluate({1: [6.0, 7.0]})[2])
        empty = compiled.evaluate({1: []})
        self.assertFalse(empty[2])
        self.assertTrue(empty[3])

    def test_missing_state_is_explicit(self):
        compiled = self.program([("p1", "maybe", "Float", "", -1, True)], [
            ("p2", "absent", "is_missing(maybe)", "Bool", "", -1, True),
        ])
        self.assertTrue(compiled.evaluate({1: None})[2])
        arithmetic = self.program([("p1", "maybe", "Float", "", -1, True)], [
            ("p2", "value", "maybe + 1", "Float", "", -1, True),
        ])
        with self.assertRaises(EngineError):
            arithmetic.evaluate({1: None})

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
            compiled.evaluate({1: "abc"})

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
        self.assertEqual(promoted.evaluate({})[1], 2.0)

    def test_expression_length_limit(self):
        long_expression = "1" + " + 1" * 2000
        with self.assertRaises(EngineError):
            self.program([], [("p1", "value", long_expression, "Int", "", -1, True)])

    def test_cumulative_element_budget(self):
        rows = [[0] * 100 for _ in range(1000)]  # 100000 leaves is the exact boundary
        compiled = self.program([("p1", "big", "Array", "int32", 2, False)], [
            ("p2", "total", "len(big)", "Int", "", -1, True),
        ])
        self.assertEqual(compiled.evaluate({1: rows})[2], 1000)
        overflow = [[0] * 101 for _ in range(1000)]  # 101000 leaves exceeds it
        with self.assertRaises(EngineError) as context:
            compiled.evaluate({1: overflow})
        self.assertIn("累计元素上限", str(context.exception))

    def test_cumulative_budget_spans_results(self):
        rows = [[0] * 100 for _ in range(500)]
        compiled = self.program([("p1", "data", "Array", "int32", 2, False)], [
            ("p2", "first", "data", "Array", "int32", 2, True),
            ("p3", "second", "data", "Array", "int32", 2, True),
        ])
        with self.assertRaises(EngineError) as context:
            compiled.evaluate({1: rows})
        self.assertIn("累计元素上限", str(context.exception))

    def test_concat_checks_size_before_allocating(self):
        rows = [0] * 60000
        compiled = self.program([("p1", "data", "Array", "int32", 1, False)], [
            ("p2", "joined", "concat(data, data)", "Array", "int32", 1, True),
        ])
        with self.assertRaises(EngineError) as context:
            compiled.evaluate({1: rows})
        self.assertIn("超出元素数量上限", str(context.exception))

    def test_float32_representable_range(self):
        compiled = self.program([], [("p1", "value", "[1e100]", "Array", "float32", 1, True)])
        with self.assertRaises(EngineError) as context:
            compiled.evaluate({})
        self.assertIn("float32 可表示范围", str(context.exception))
        accepted = self.program([], [("p1", "value", "[1e30, -2.5]", "Array", "float32", 1, True)])
        self.assertEqual(accepted.evaluate({})[1], [1e30, -2.5])


@unittest.skipIf(np is None, "需要 NumPy")
class NumpyInputs(unittest.TestCase):
    """A numeric host Array crosses into Python as an ndarray, so inputs must reach expressions as lists."""

    def test_list_and_ndarray_inputs_agree(self):
        program = compile_program([("p1", "values", "Array", "float64", 1, False)], [
            ("p2", "average", "mean(values)", "Float", "", -1, True),
            ("p3", "count", "len(values)", "Int", "", -1, True),
        ])
        as_list = program.evaluate({1: [2.0, 4.0]})
        as_numpy = program.evaluate({1: np.array([2.0, 4.0], dtype=np.float64)})
        self.assertEqual(as_list, as_numpy)
        self.assertEqual(as_numpy[2], 3.0)

    def test_rank_two_int32_input_is_normalised(self):
        program = compile_program([("p1", "grid", "Array", "int32", 2, False)], [
            ("p2", "rows", "len(grid)", "Int", "", -1, True),
            ("p3", "first", "mean(grid[0])", "Float", "", -1, True),
        ])
        values = program.evaluate({1: np.array([[1, 2], [3, 4]], dtype=np.int32)})
        self.assertEqual(values[2], 2)
        self.assertEqual(values[3], 1.5)

    def test_branch_tool_accepts_numpy_array(self):
        tool = BranchTool()
        variable_rows, program_rows = tables([("p1", "values", "Array", "float64", 1, False)], [
            ("p2", "ok", "mean(values) > 2", "Bool", "", -1, False),
        ])
        self.assertEqual(tool.initialize({0: variable_rows, 1: program_rows})["code"], 0)
        yes = tool.execute({1: np.array([2.0, 4.0], dtype=np.float64)})
        self.assertEqual(yes["outputs"][0], 0)
        self.assertTrue(yes["outputs"][2])
        self.assertFalse(yes["outputs"][3])

    def test_declared_dtype_and_rank_are_enforced(self):
        program = compile_program([("p1", "values", "Array", "float64", 1, False)], [
            ("p2", "count", "len(values)", "Int", "", -1, True),
        ])
        with self.assertRaises(EngineError) as dtype_error:
            program.evaluate({1: np.array([1.0], dtype=np.float32)})
        self.assertIn("元素类型", str(dtype_error.exception))
        with self.assertRaises(EngineError) as rank_error:
            program.evaluate({1: np.array([[1.0]], dtype=np.float64)})
        self.assertIn("维度", str(rank_error.exception))

    def test_budget_is_charged_before_conversion(self):
        program = compile_program([
            ("p1", "first", "Array", "int32", 1, False),
            ("p2", "second", "Array", "int32", 1, False),
        ], [("p3", "count", "len(first) + len(second)", "Int", "", -1, True)])
        with self.assertRaises(EngineError) as context:
            program.evaluate({1: np.zeros(60000, dtype=np.int32),
                              2: np.zeros(60000, dtype=np.int32)})
        self.assertIn("累计元素上限", str(context.exception))

    def test_zero_leaf_shape_still_bounds_containers(self):
        program = compile_program([("p1", "grid", "Array", "int32", 2, False)], [
            ("p2", "rows", "len(grid)", "Int", "", -1, True),
        ])
        with self.assertRaises(EngineError) as context:
            program.evaluate({1: np.zeros((200000, 0), dtype=np.int32)})
        self.assertIn("容器数量", str(context.exception))

    def test_non_finite_and_string_arrays_are_rejected(self):
        numeric = compile_program([("p1", "values", "Array", "float64", 1, False)], [
            ("p2", "count", "len(values)", "Int", "", -1, True),
        ])
        with self.assertRaises(EngineError) as context:
            numeric.evaluate({1: np.array([np.nan], dtype=np.float64)})
        self.assertIn("有限数值", str(context.exception))
        strings = compile_program([("p1", "labels", "Array", "string", 1, False)], [
            ("p2", "count", "len(labels)", "Int", "", -1, True),
        ])
        with self.assertRaises(EngineError) as string_error:
            strings.evaluate({1: np.array(["a"], dtype="U1")})
        self.assertIn("NumPy 字符串数组", str(string_error.exception))


class ToolContracts(unittest.TestCase):
    def test_expression_tool_returns_exposed_ports_only(self):
        tool = ExpressionTool()
        variable_rows, program_rows = tables([("p1", "radii", "Array", "float64", 1, False)], [
            ("p2", "count", "len(radii)", "Int", "", -1, True),
            ("p3", "hidden", "sum(radii)", "Float", "", -1, False),
        ])
        status = tool.initialize({0: variable_rows, 1: program_rows})
        self.assertEqual(status["code"], 0)
        result = tool.execute({1: [1.0, 2.0]})
        self.assertEqual(result["outputs"][0], 0)
        self.assertEqual(result["outputs"][2], 2)
        self.assertNotIn(3, result["outputs"])
        failure = tool.execute({1: None})
        self.assertNotEqual(failure["outputs"][0], 0)
        self.assertIsNone(failure["outputs"][2])

    def test_branch_tool_is_mutually_exclusive(self):
        tool = BranchTool()
        variable_rows, program_rows = tables([("p1", "count", "Int", "", -1, False)], [
            ("p2", "ok", "count > 3", "Bool", "", -1, False),
        ])
        self.assertEqual(tool.initialize({0: variable_rows, 1: program_rows})["code"], 0)
        yes = tool.execute({1: 5})
        self.assertTrue(yes["outputs"][2])
        self.assertFalse(yes["outputs"][3])
        no = tool.execute({1: 1})
        self.assertFalse(no["outputs"][2])
        self.assertTrue(no["outputs"][3])

    def test_branch_accepts_only_true_bool(self):
        tool = BranchTool()
        variable_rows, program_rows = tables([("p1", "count", "Int", "", -1, False)], [
            ("p2", "ok", "count", "Bool", "", -1, False),
        ])
        status = tool.initialize({0: variable_rows, 1: program_rows})
        self.assertEqual(status["code"], 0)
        result = tool.execute({1: 1})
        self.assertNotEqual(result["outputs"][0], 0)
        # An unproduced branch is omitted by the tool; the host supplies the declared default.
        self.assertNotIn(2, result["outputs"])
        self.assertNotIn(3, result["outputs"])

    def test_branch_requires_single_hidden_result(self):
        tool = BranchTool()
        variable_rows, program_rows = tables([], [("p1", "ok", "True", "Bool", "", -1, True)])
        status = tool.initialize({0: variable_rows, 1: program_rows})
        self.assertNotEqual(status["code"], 0)


if __name__ == "__main__":
    unittest.main()
