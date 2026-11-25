"""VisionTools author SDK: typed parameter helpers for Python tools.

A tool imports this SDK next to its own module (``from .visiontools_sdk import ...``).
The native build assembles this module and the generated ``enums.py``; packaging and tests use
that same SDK artifact. Examples keep no SDK copies.

The SDK mirrors the C++ ``visionrt/contracts/Parameters.h`` interface: an
``InputReader``/``OutputWriter`` pair owns the dict keys, and the tool keeps every
business decision (upstream-status policy, missing inputs, in-place versus copy) in
its own code. Values — ndarray images, rectangular numeric arrays as ``numpy.ndarray``,
ragged or string arrays as nested lists — are carried by reference; nothing is cloned
while wrapping.

The host contract this SDK relies on:

* ``execute`` returns ``{"outputs": {port_id: value, 0: code, 1: message}, "detail": detail}``;
* ``initialize`` and ``release`` return the status dict alone (``{"code": code, "message": message, "detail": detail}``);
* a missing key and an explicit ``None`` both mean "no value";
* ``upstream_status`` is read but never acted on by the SDK;
* the status channels are packed alongside outputs: the tool writes them via ``result()``,
  never by manually injecting reserved keys into business outputs.

The three module-level helpers (``success``, ``failure``, ``result``) are the whole
protocol: a functional tool returns them directly, and a class tool uses
``OutputParametersBase`` for the same shape.
"""

from abc import ABC, abstractmethod
from enum import IntEnum
from .enums import CommonError, COMMON_MESSAGES


# NumPy carries rectangular numeric arrays. The host refuses an interpreter without it, so a
# missing import here is a broken environment, not a capability a tool may silently lose.
import numpy as _numpy


def _is_array(value):
    """A declared array arrives as a NumPy ndarray when it is rectangular and numeric, and as a
    nested list when it is ragged, holds strings, or was built from the readable nested form."""
    return isinstance(value, list) or isinstance(value, _numpy.ndarray)


# Ports are integers: input 0 is the upstream status, outputs 0 and 1 are the status code and
# message packed by result(). Business numbers start above them, so one number may appear once per
# direction. A number, not its spelling, is the identity.
UPSTREAM_STATUS_PORT = 0
RESERVED_PORTS = (0, 1)

MAX_PORT_ID = 2 ** 32 - 1


def _port(key):
    """The one place an untrusted key becomes a port number; bool is not an integer here."""
    if type(key) is not int:
        raise ContractError("端口编号必须是整数，实际为 " + repr(key))
    if key < 0 or key > MAX_PORT_ID:
        raise ContractError("端口编号超出范围：" + repr(key))
    return key


def _label(key):
    return "端口 " + str(key)


class ContractError(ValueError):
    """Raised for author/declaration errors: missing or mistyped ports."""


def status(code=0, message="", detail=""):
    """The single status a lifecycle call returns."""
    return {"code": int(code) if isinstance(code, IntEnum) else code, "message": message, "detail": detail}


def result(outputs=None, code=0, message="", detail=""):
    """The execution or training result envelope."""
    res = {} if outputs is None else dict(outputs)
    for key in res:
        _port(key)
        if key in RESERVED_PORTS:
            raise ContractError("状态由宿主发布，工具不得写入输出" + _label(key))
    res[0] = int(code) if isinstance(code, IntEnum) else code
    res[1] = message
    return {"outputs": res, "detail": detail}


def success(outputs=None, message="运行成功"):
    """A successful execution. Declared ports still have to be present in ``outputs``."""
    return result(outputs, 0, message)


def failure(code, message=None, detail="", outputs=None):
    """A business failure. ``outputs`` keeps the declared ports, with ``None`` for no value."""
    return result(outputs, int(code), COMMON_MESSAGES[int(code)] if message is None else message, detail)


def skipped(upstream_status, outputs=None):
    """The answer of a tool whose upstream branch was not selected: nothing ran."""
    return result(outputs, upstream_status, "上游未选中本分支，工具未执行")


def is_missing(value):
    """True when the port carries no usable value."""
    return value is None


class InputReader:
    """Typed reads over the same dict the host passes in."""

    def __init__(self, values, names=None):
        if not isinstance(values, dict):
            raise ContractError("输入必须是端口字典")
        self._values = values
        self._names = {} if names is None else names

    @property
    def values(self):
        return self._values

    def _label(self, key):
        return self._names.get(key, _label(key))

    def _find(self, key):
        value = self._values.get(_port(key))
        return None if is_missing(value) else value

    def present(self, key):
        return self._find(key) is not None

    def _require(self, key):
        value = self._find(key)
        if value is None:
            raise ContractError("输入" + self._label(key) + "缺失")
        return value

    def integer(self, key):
        value = self._require(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractError("输入" + self._label(key) + "类型不匹配（期望整数）")
        return value

    def real(self, key):
        value = self._require(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError("输入" + self._label(key) + "类型不匹配（期望浮点数）")
        return float(value)

    def boolean(self, key):
        value = self._require(key)
        if not isinstance(value, bool):
            raise ContractError("输入" + self._label(key) + "类型不匹配（期望布尔值）")
        return value

    def text(self, key):
        value = self._require(key)
        if not isinstance(value, str):
            raise ContractError("输入" + self._label(key) + "类型不匹配（期望字符串）")
        return value

    def array(self, key):
        value = self._require(key)
        if not _is_array(value):
            raise ContractError("输入" + self._label(key) + "类型不匹配（期望 ndarray 或列表）")
        return value

    image = array
    point = array
    line = array
    rectangle = array
    circle = array
    polygon = array
    direction = array
    coordinate_frame_2d = array

    def training_set(self, key):
        value = self._require(key)
        if not isinstance(value, dict):
            raise ContractError("输入" + self._label(key) + "类型不匹配（期望训练集）")
        return value

    def optional_integer(self, key):
        value = self._find(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractError("输入" + self._label(key) + "类型不匹配（期望整数）")
        return value

    def optional_real(self, key):
        value = self._find(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError("输入" + self._label(key) + "类型不匹配（期望浮点数）")
        return float(value)

    def optional_boolean(self, key):
        value = self._find(key)
        if value is None:
            return None
        if not isinstance(value, bool):
            raise ContractError("输入" + self._label(key) + "类型不匹配（期望布尔值）")
        return value

    def optional_text(self, key):
        value = self._find(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ContractError("输入" + self._label(key) + "类型不匹配（期望字符串）")
        return value

    def optional_array(self, key):
        value = self._find(key)
        if value is None:
            return None
        if not _is_array(value):
            raise ContractError("输入" + self._label(key) + "类型不匹配（期望 ndarray 或列表）")
        return value

    # Raw access for ports a tool inspects without committing to one payload kind.
    def require(self, key):
        return self._require(key)


class OutputWriter:
    """Collects declared business outputs; the status is not a port and stays out."""

    def __init__(self):
        self._values = {}

    def integer(self, key, value):
        self._write(key, value)

    def real(self, key, value):
        # No conversion: a wrong author type is reported by the host's value validation instead of
        # being silently repaired here.
        self._write(key, value)

    def boolean(self, key, value):
        self._write(key, value)

    def text(self, key, value):
        self._write(key, value)

    def array(self, key, value):
        # Stored by reference: images and native lists keep their identity.
        self._write(key, value)

    def value(self, key, raw):
        self._write(key, raw)

    image = array
    point = array
    line = array
    rectangle = array
    circle = array
    polygon = array
    direction = array
    coordinate_frame_2d = array
    training_set = value

    def _write(self, key, value):
        _port(key)
        if key in RESERVED_PORTS:
            raise ContractError("状态由宿主发布，工具不得写入" + _label(key))
        if key in self._values:
            raise ContractError("输出" + _label(key) + "重复写入")
        self._values[key] = value

    def take(self):
        return self._values


class ParametersBase(ABC):
    """A view of arbitrary initialization or training parameters, including port zero."""

    def parse(self, values):
        self.read_fields(InputReader(values, getattr(self, "parameter_names", None)))

    @abstractmethod
    def read_fields(self, reader):
        """Read declared parameters through the shared typed reader."""


InitializationParametersBase = ParametersBase
TrainingParametersBase = ParametersBase


class InputParametersBase(ParametersBase):
    """Base class for typed input views; reads upstream_status once, centrally."""

    def __init__(self):
        self.upstream_status = 0

    def parse(self, values):
        reader = InputReader(values, getattr(self, "parameter_names", None))
        raw = reader.values.get(UPSTREAM_STATUS_PORT)
        if raw is None:
            self.upstream_status = 0
        elif isinstance(raw, bool) or not isinstance(raw, int):
            raise ContractError("输入" + _label(UPSTREAM_STATUS_PORT) + "需要整数状态码")
        else:
            self.upstream_status = raw
        self.read_fields(reader)

    @abstractmethod
    def read_fields(self, reader):
        """Read the declared business inputs from the reader."""


class OutputParametersBase(ABC):
    """Base class for typed outputs; one construction point for the envelope."""

    def __init__(self):
        self.status_code = 0
        self.status_message = ""
        self.status_detail = ""

    def status(self):
        """The status a lifecycle call returns."""
        return status(self.status_code, self.status_message, self.status_detail)

    def pack(self):
        """The envelope execute returns: the written outputs plus this status."""
        writer = OutputWriter()
        self.write_fields(writer)
        return result(writer.take(), self.status_code, self.status_message, self.status_detail)

    @abstractmethod
    def write_fields(self, writer):
        """Write the declared business outputs through the writer."""
