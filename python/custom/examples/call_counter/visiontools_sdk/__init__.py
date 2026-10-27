"""VisionTools author SDK: typed parameter helpers for Python tools.

The SDK mirrors the C++ ``visionrt/contracts/Parameters.h`` interface: an
``InputReader``/``OutputWriter`` pair owns the dict keys and the status ports,
and the tool keeps every business decision (upstream-status policy, missing
inputs, in-place versus copy) in its own code. Values — ndarray images and
native lists — are carried by reference; nothing is cloned while wrapping.

The host contract this SDK relies on:
* a missing key and an explicit ``None`` both mean "no value";
* ``upstream_status`` is read but never acted on by the SDK;
* the two status keys belong to the output base class only.
"""

from abc import ABC, abstractmethod

UPSTREAM_STATUS_PORT = "upstream_status"
STATUS_CODE_PORT = "__status_code"
STATUS_MESSAGE_PORT = "__status_message"


class ContractError(ValueError):
    """Raised for author/declaration errors: missing or mistyped ports."""


def is_missing(value):
    """True when the port carries no usable value."""
    return value is None


class InputReader:
    """Typed reads over the same dict the host passes in."""

    def __init__(self, values):
        if not isinstance(values, dict):
            raise ContractError("输入必须是端口字典")
        self._values = values

    @property
    def values(self):
        return self._values

    def _find(self, key):
        value = self._values.get(key)
        return None if is_missing(value) else value

    def present(self, key):
        return self._find(key) is not None

    def _require(self, key):
        value = self._find(key)
        if value is None:
            raise ContractError("输入端口为空：" + key)
        return value

    def integer(self, key):
        value = self._require(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractError("输入端口类型不匹配：" + key + "（期望整数）")
        return value

    def real(self, key):
        value = self._require(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError("输入端口类型不匹配：" + key + "（期望浮点数）")
        return float(value)

    def boolean(self, key):
        value = self._require(key)
        if not isinstance(value, bool):
            raise ContractError("输入端口类型不匹配：" + key + "（期望布尔值）")
        return value

    def text(self, key):
        value = self._require(key)
        if not isinstance(value, str):
            raise ContractError("输入端口类型不匹配：" + key + "（期望字符串）")
        return value

    def array(self, key):
        value = self._require(key)
        if not isinstance(value, list):
            raise ContractError("输入端口类型不匹配：" + key + "（期望列表）")
        return value

    def optional_integer(self, key):
        value = self._find(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractError("输入端口类型不匹配：" + key + "（期望整数）")
        return value

    def optional_real(self, key):
        value = self._find(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError("输入端口类型不匹配：" + key + "（期望浮点数）")
        return float(value)

    def optional_boolean(self, key):
        value = self._find(key)
        if value is None:
            return None
        if not isinstance(value, bool):
            raise ContractError("输入端口类型不匹配：" + key + "（期望布尔值）")
        return value

    def optional_text(self, key):
        value = self._find(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ContractError("输入端口类型不匹配：" + key + "（期望字符串）")
        return value

    def optional_array(self, key):
        value = self._find(key)
        if value is None:
            return None
        if not isinstance(value, list):
            raise ContractError("输入端口类型不匹配：" + key + "（期望列表）")
        return value

    # Raw access for ports a tool inspects without committing to one payload kind.
    def require(self, key):
        return self._require(key)


class OutputWriter:
    """Collects declared business outputs; the status keys stay with the base class."""

    def __init__(self):
        self._values = {}

    def integer(self, key, value):
        self._write(key, value)

    def real(self, key, value):
        self._write(key, float(value))

    def boolean(self, key, value):
        self._write(key, bool(value))

    def text(self, key, value):
        self._write(key, value)

    def array(self, key, value):
        # Stored by reference: images and native lists keep their identity.
        self._write(key, value)

    def missing(self, key):
        # A declared port with no value is written explicitly, never left out.
        self._write(key, None)

    def value(self, key, raw):
        self._write(key, raw)

    def _write(self, key, value):
        if key in (STATUS_CODE_PORT, STATUS_MESSAGE_PORT):
            raise ContractError("状态端口由参数基类写入：" + key)
        if key in self._values:
            raise ContractError("输出端口重复写入：" + key)
        self._values[key] = value

    def take(self):
        return self._values


class InputParametersBase(ABC):
    """Base class for typed input views; reads upstream_status once, centrally."""

    def __init__(self):
        self.upstream_status = 0

    def parse(self, values):
        reader = InputReader(values)
        raw = reader.values.get(UPSTREAM_STATUS_PORT)
        if raw is None:
            self.upstream_status = 0
        elif isinstance(raw, bool) or not isinstance(raw, int):
            raise ContractError("输入端口 '" + UPSTREAM_STATUS_PORT + "' 需要整数状态码")
        else:
            self.upstream_status = raw
        self.read_fields(reader)

    @abstractmethod
    def read_fields(self, reader):
        """Read the declared business inputs from the reader."""


class OutputParametersBase(ABC):
    """Base class for typed outputs; one construction point for status keys."""

    def __init__(self):
        self.status_code = 0
        self.status_message = ""

    def pack(self):
        writer = OutputWriter()
        self.write_fields(writer)
        values = writer.take()
        values[STATUS_CODE_PORT] = self.status_code
        values[STATUS_MESSAGE_PORT] = self.status_message
        return values

    @abstractmethod
    def write_fields(self, writer):
        """Write the declared business outputs through the writer."""
