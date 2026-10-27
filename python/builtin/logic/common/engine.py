"""Restricted expression engine shared by the expression and branch tools.

The vendored simpleeval release supplies parsing, operator tables and AST reuse.
This module adds the VisionRT subset: an explicit function table, list literals,
strict output typing and fixed resource limits. Attribute access, imports,
assignments, comprehensions and lambdas stay unavailable.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass

from .vendor import simpleeval

MAX_EXPRESSION_CHARS = 4096
MAX_AST_NODES = 512
MAX_COLLECTION_ITEMS = 100000
MAX_STRING_CHARS = 1000000
MAX_ARRAY_RANK = 4
# float32 keeps only ~7 significant digits; anything beyond FLT_MAX can not be represented and
# would silently become inf when the value crosses into the native tool boundary.
FLOAT32_MAX = 3.4028234663852886e38

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1

DTYPES = {
    "uint8": (int, 0, 255),
    "int32": (int, -(2**31), 2**31 - 1),
    "int64": (int, INT64_MIN, INT64_MAX),
    "float32": (float, None, None),
    "float64": (float, None, None),
    "bool": (bool, None, None),
    "string": (str, None, None),
}


class EngineError(Exception):
    """Configuration or evaluation failure with a user-facing message."""


class Budget:
    """Shared, cumulative element budget for one evaluation.

    Per-container limits are not enough: nested containers and intermediate results can each stay
    below the limit while the total allocation grows without bound.
    """

    __slots__ = ("remaining",)

    def __init__(self, total: int = MAX_COLLECTION_ITEMS):
        self.remaining = total

    def require(self, count: int, location: str) -> None:
        if count > self.remaining:
            raise EngineError(f"{location} 超出累计元素上限（最多 {MAX_COLLECTION_ITEMS} 个）")

    def spend(self, count: int, location: str) -> None:
        self.require(count, location)
        self.remaining -= count


@dataclass(frozen=True)
class TypeSpec:
    name: str
    dtype: str | None = None
    rank: int = -1

    @classmethod
    def parse(cls, name: str, dtype: str, rank: str) -> "TypeSpec":
        known = {"Int", "Float", "Bool", "String", "Array"}
        if name not in known:
            raise EngineError(f"表达式不支持的类型：{name}")
        if name == "Array":
            if dtype not in DTYPES:
                raise EngineError(f"数组元素类型无效：{dtype or '空'}")
            try:
                parsed_rank = int(rank)
            except (TypeError, ValueError):
                raise EngineError("数组维度必须是整数") from None
            if parsed_rank < 1 or parsed_rank > MAX_ARRAY_RANK:
                raise EngineError(f"数组维度必须在 1 到 {MAX_ARRAY_RANK} 之间")
            return cls(name, dtype, parsed_rank)
        return cls(name)


@dataclass(frozen=True)
class Variable:
    port_id: str
    symbol: str
    spec: TypeSpec
    allow_missing: bool


@dataclass(frozen=True)
class Result:
    port_id: str
    symbol: str
    expression: str
    spec: TypeSpec
    exposed: bool


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_leaf(value, dtype: str, location: str):
    expected, minimum, maximum = DTYPES[dtype]
    if dtype == "float32" or dtype == "float64":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EngineError(f"{location} 需要数值，实际为 {type(value).__name__}")
        converted = float(value)
        if not math.isfinite(converted):
            raise EngineError(f"{location} 必须是有限数值")
        if dtype == "float32" and abs(converted) > FLOAT32_MAX:
            raise EngineError(f"{location} 超出 float32 可表示范围")
        return converted
    if dtype == "bool":
        if not isinstance(value, bool):
            raise EngineError(f"{location} 需要布尔值")
        return value
    if dtype == "string":
        if not isinstance(value, str):
            raise EngineError(f"{location} 需要字符串")
        if len(value) > MAX_STRING_CHARS:
            raise EngineError(f"{location} 字符串超出长度上限")
        return value
    if isinstance(value, bool) or not isinstance(value, int):
        raise EngineError(f"{location} 需要整数")
    if minimum is not None and (value < minimum or value > maximum):
        raise EngineError(f"{location} 超出 {dtype} 可表示范围")
    return value


def check_value(value, spec: TypeSpec, location: str, budget: Budget | None = None):
    """Returns a normalised value or raises EngineError with the exact path."""
    if spec.name == "Int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise EngineError(f"{location} 需要整数，实际为 {type(value).__name__}")
        if value < INT64_MIN or value > INT64_MAX:
            raise EngineError(f"{location} 超出整数可表示范围")
        return value
    if spec.name == "Float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EngineError(f"{location} 需要数值，实际为 {type(value).__name__}")
        converted = float(value)
        if not math.isfinite(converted):
            raise EngineError(f"{location} 必须是有限数值")
        return converted
    if spec.name == "Bool":
        if not isinstance(value, bool):
            raise EngineError(f"{location} 需要布尔值，实际为 {type(value).__name__}")
        return value
    if spec.name == "String":
        if not isinstance(value, str):
            raise EngineError(f"{location} 需要字符串，实际为 {type(value).__name__}")
        if len(value) > MAX_STRING_CHARS:
            raise EngineError(f"{location} 字符串超出长度上限")
        return value
    if spec.name == "Array":
        if not isinstance(value, list):
            raise EngineError(f"{location} 需要数组，实际为 {type(value).__name__}")
        return _check_array(value, spec, 1, location, budget)
    raise EngineError(f"{location} 的类型不受支持：{spec.name}")


def _check_array(value, spec: TypeSpec, depth: int, location: str, budget: Budget | None = None):
    if not isinstance(value, list):
        raise EngineError(f"{location} 需要数组")
    if depth < spec.rank:
        # Rank-mismatch and size checks run before any leaf conversion allocates a copy.
        if len(value) > MAX_COLLECTION_ITEMS:
            raise EngineError(f"{location} 数组元素数量超出上限")
        return [_check_array(item, spec, depth + 1, f"{location}[{index}]", budget)
                for index, item in enumerate(value)]
    leaves = len(value)
    if budget is not None:
        budget.spend(leaves, location)
    elif leaves > MAX_COLLECTION_ITEMS:
        raise EngineError(f"{location} 数组元素数量超出上限")
    return [_check_leaf(item, spec.dtype, f"{location}[{index}]") for index, item in enumerate(value)]


def _array_leaves(value) -> int:
    """Leaf count of a nested list without copying or recursing into scalars."""
    if isinstance(value, list):
        return sum(_array_leaves(item) for item in value)
    return 1


def _numbers(values, location: str):
    if not isinstance(values, list) or not values:
        raise EngineError(f"{location} 需要非空数值数组")
    for index, value in enumerate(values):
        if not _is_number(value):
            raise EngineError(f"{location}[{index}] 不是数值")
    return values


def _mean(values):
    data = _numbers(values, "mean")
    return float(sum(data)) / len(data)


def _median(values):
    data = sorted(float(value) for value in _numbers(values, "median"))
    middle = len(data) // 2
    return data[middle] if len(data) % 2 else (data[middle - 1] + data[middle]) / 2.0


def _std(values):
    data = [float(value) for value in _numbers(values, "std")]
    average = sum(data) / len(data)
    return math.sqrt(sum((item - average) ** 2 for item in data) / len(data))


def _is_empty(value):
    if isinstance(value, (list, str)):
        return len(value) == 0
    raise EngineError("is_empty 只接受数组或字符串")


def _is_missing(value):
    return value is None


def _all_nonempty(values):
    if not isinstance(values, list):
        raise EngineError("all_nonempty 需要数组")
    if not values:
        return False
    return all(bool(item) for item in values)


def _concat(*values, budget: Budget | None = None):
    if not values:
        raise EngineError("concat 至少需要一个参数")
    if all(isinstance(value, str) for value in values):
        total = sum(len(value) for value in values)
        if total > MAX_STRING_CHARS:
            raise EngineError("concat 结果超出长度上限")
        return "".join(values)
    if all(isinstance(value, list) for value in values):
        # Size is checked before extend() allocates, so a huge join is rejected up front.
        total = sum(len(value) for value in values)
        if total > MAX_COLLECTION_ITEMS:
            raise EngineError("concat 结果超出元素数量上限")
        if budget is not None:
            budget.require(total, "concat")
        result = []
        result.extend(values[0])
        for value in values[1:]:
            result.extend(value)
        return result
    raise EngineError("concat 只接受全部字符串或全部数组")


def _vector_operation(left, right, operation, name):
    if not isinstance(left, list):
        raise EngineError(f"{name} 的第一个参数必须是数组")
    if isinstance(right, list):
        if len(left) != len(right):
            raise EngineError(f"{name} 不接受不等长数组")
        pairs = list(zip(left, right, strict=True))
    else:
        pairs = [(item, right) for item in left]
    result = []
    for index, (first, second) in enumerate(pairs):
        if not _is_number(first) or not _is_number(second):
            raise EngineError(f"{name}[{index}] 不是数值")
        result.append(operation(first, second))
    return result


def _divide(first, second):
    if second == 0:
        raise EngineError("向量除法除数为零")
    return first / second


def _vector_compare(operation, name):
    def compare(left, right):
        return _vector_operation(left, right, lambda a, b: bool(operation(a, b)), name)

    return compare


def _select_guard(values, mask, budget: Budget | None = None):
    if not isinstance(values, list) or not isinstance(mask, list):
        raise EngineError("select 需要两个数组")
    if len(values) != len(mask):
        raise EngineError("select 的两个数组长度必须一致")
    if any(not isinstance(item, bool) for item in mask):
        raise EngineError("select 的筛选数组必须是布尔值")
    if budget is not None:
        budget.require(len(values), "select")


def _select(values, mask):
    _select_guard(values, mask)
    return [value for value, keep in zip(values, mask, strict=True) if keep]


def _vector_guarded(left, right, operation, name, budget):
    if budget is not None:
        budget.require(len(left) if isinstance(left, list) else 0, name)
    return _vector_operation(left, right, operation, name)


def functions(budget: Budget | None = None):
    def concat(*values):
        return _concat(*values, budget=budget)

    def select(values, mask):
        _select_guard(values, mask, budget)
        return [value for value, keep in zip(values, mask, strict=True) if keep]

    def unary(operation, name):
        return lambda left, right: _vector_guarded(left, right, operation, name, budget)

    return {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "round": round,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "atan2": math.atan2,
        "floor": math.floor,
        "ceil": math.ceil,
        "mean": _mean,
        "median": _median,
        "std": _std,
        "is_empty": _is_empty,
        "is_missing": _is_missing,
        "all_nonempty": _all_nonempty,
        "any": lambda values: any(bool(item) for item in values) if isinstance(values, list) else bool(values),
        "v_add": unary(lambda a, b: a + b, "v_add"),
        "v_sub": unary(lambda a, b: a - b, "v_sub"),
        "v_mul": unary(lambda a, b: a * b, "v_mul"),
        "v_div": unary(_divide, "v_div"),
        "v_gt": _vector_compare(lambda a, b: a > b, "v_gt"),
        "v_ge": _vector_compare(lambda a, b: a >= b, "v_ge"),
        "v_lt": _vector_compare(lambda a, b: a < b, "v_lt"),
        "v_le": _vector_compare(lambda a, b: a <= b, "v_le"),
        "v_eq": _vector_compare(lambda a, b: a == b, "v_eq"),
        "select": select,
        "concat": concat,
    }


class _Engine(simpleeval.SimpleEval):
    """simpleeval with VisionRT's reduced AST surface."""

    def __init__(self, names, table, budget: Budget | None = None):
        super().__init__(names=names, functions=table)
        for node in (ast.Assign, ast.AugAssign, ast.Import, ast.Attribute, ast.JoinedStr, ast.FormattedValue):
            self.nodes.pop(node, None)
        self.nodes[ast.List] = self._eval_list
        self.nodes[ast.Tuple] = self._eval_list
        self.budget = budget

    def _eval_list(self, node):
        if self.budget is not None:
            self.budget.require(len(node.elts), "列表字面量")
        elif len(node.elts) > MAX_COLLECTION_ITEMS:
            raise EngineError("列表字面量超出元素数量上限")
        values = [self._eval(child) for child in node.elts]
        leaves = sum(_array_leaves(value) for value in values)
        if self.budget is not None:
            self.budget.require(leaves, "列表字面量")
        elif leaves > MAX_COLLECTION_ITEMS:
            raise EngineError("列表字面量超出元素数量上限")
        return values


def _validate_syntax(expression: str):
    if not expression or not expression.strip():
        raise EngineError("表达式不能为空")
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise EngineError(f"表达式超出 {MAX_EXPRESSION_CHARS} 字符上限")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        compile(tree, "<expression>", "eval")
    except SyntaxError as error:
        raise EngineError(f"表达式语法错误：第 {error.lineno} 行第 {error.offset} 列") from None
    if len(list(ast.walk(tree))) > MAX_AST_NODES:
        raise EngineError(f"表达式超出 {MAX_AST_NODES} 个语法节点上限")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Dict,
                             ast.NamedExpr, ast.Await, ast.Yield, ast.Starred)):
            raise EngineError(f"表达式包含不支持的语法：{type(node).__name__}")
    # simpleeval evaluates the expression node itself, not the module wrapper.
    return tree.body


@dataclass
class CompiledResult:
    result: Result
    tree: object  # ast.Expression body node reused by simpleeval


class Program:
    """Compiled variable table plus ordered program."""

    def __init__(self, variables, results, trees):
        self.variables = variables
        self.results = results
        self._trees = trees

    @classmethod
    def from_tables(cls, variable_rows, program_rows, *, condition: bool = False) -> "Program":
        if not isinstance(variable_rows, list) or not isinstance(program_rows, list):
            raise EngineError("初始化表必须是数组")
        variables = []
        symbols = set()
        port_ids = set()
        for row in variable_rows:
            if not isinstance(row, list) or len(row) != 6:
                raise EngineError("变量表每行需要 6 列")
            port_id, symbol, type_name, dtype, rank, allow_missing = row
            if not isinstance(port_id, str) or not isinstance(symbol, str):
                raise EngineError("变量表的前两列必须是字符串")
            if port_id in port_ids:
                raise EngineError(f"变量端口重复：{port_id}")
            if symbol in symbols:
                raise EngineError(f"变量符号重复：{symbol}")
            if allow_missing not in ("true", "false"):
                raise EngineError("变量表的 allow_missing 必须是 true 或 false")
            variables.append(Variable(port_id, symbol, TypeSpec.parse(type_name, dtype, rank),
                                      allow_missing == "true"))
            symbols.add(symbol)
            port_ids.add(port_id)
        results = []
        trees = []
        known = set(symbols)
        reserved = {row[1] for row in program_rows if isinstance(row, list) and len(row) == 7
                    and isinstance(row[1], str)}
        for row in program_rows:
            if not isinstance(row, list) or len(row) != 7:
                raise EngineError("公式表每行需要 7 列")
            port_id, symbol, expression, type_name, dtype, rank, exposed = row
            if not isinstance(port_id, str) or not isinstance(symbol, str) or not isinstance(expression, str):
                raise EngineError("公式表的前三列必须是字符串")
            if port_id in port_ids:
                raise EngineError(f"端口重复：{port_id}")
            if symbol in known:
                raise EngineError(f"符号重复：{symbol}")
            if exposed not in ("true", "false"):
                raise EngineError("公式表的 exposed 必须是 true 或 false")
            tree = _validate_syntax(expression)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in reserved and node.id not in known:
                    raise EngineError(f"表达式引用了后续结果：{node.id}")
            results.append(Result(port_id, symbol, expression, TypeSpec.parse(type_name, dtype, rank),
                                  exposed == "true"))
            trees.append(CompiledResult(results[-1], tree))
            known.add(symbol)
            port_ids.add(port_id)
        if condition:
            if len(results) != 1:
                raise EngineError("条件工具必须恰好一条公式")
            if results[0].spec.name != "Bool" or results[0].exposed:
                raise EngineError("条件公式必须是 Bool 类型且不暴露端口")
        return cls(variables, results, trees)

    def evaluate(self, inputs: dict) -> dict:
        budget = Budget()
        names = {}
        for variable in self.variables:
            if variable.port_id not in inputs:
                raise EngineError(f"缺少输入端口：{variable.port_id}")
            value = inputs[variable.port_id]
            if value is None:
                if not variable.allow_missing:
                    raise EngineError(f"输入变量没有值：{variable.symbol}")
                names[variable.symbol] = None
                continue
            names[variable.symbol] = check_value(value, variable.spec, variable.symbol, budget)
        engine = _Engine(names, functions(budget), budget)
        outputs = {}
        for compiled in self._trees:
            try:
                value = engine.eval(compiled.result.expression, previously_parsed=compiled.tree)
            except EngineError:
                raise
            except simpleeval.FeatureNotAvailable as error:
                raise EngineError(f"表达式包含不支持的语法：{error}") from None
            except ZeroDivisionError:
                raise EngineError("表达式除数为零") from None
            except Exception as error:  # noqa: BLE001 - messages must reach the tool status
                raise EngineError(f"表达式求值失败：{error}") from None
            names[compiled.result.symbol] = value
            if isinstance(value, str) and len(value) > MAX_STRING_CHARS:
                raise EngineError(f"{compiled.result.symbol} 超出字符串长度上限")
            # Hidden results are still type-checked because later formulas use them.
            outputs[compiled.result.port_id] = check_value(value, compiled.result.spec,
                                                           compiled.result.symbol, budget)
        return outputs

    def exposed_ports(self) -> list[str]:
        return [result.port_id for result in self.results if result.exposed]
