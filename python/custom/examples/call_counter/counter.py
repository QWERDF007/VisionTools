"""Stateful counter example: keeps a class entry and uses the author SDK for outputs.

The counter is the canonical stateful Python class: ``initialize`` resets the count,
``execute`` increments it, ``release`` reports an empty success. Outputs go through the
packaged ``visiontools_sdk`` output subclass so the status keys have one spelling. The
host contract restricts ``initialize``/``release`` returns to the two status keys only,
so lifecycle calls use the bare output base while ``execute`` publishes ``count``.
"""

import sys
from pathlib import Path

try:
    from .visiontools_sdk import OutputParametersBase
except ImportError:  # Direct file loading (tests / debugging) has no package context.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from visiontools_sdk import OutputParametersBase


class LifecycleStatus(OutputParametersBase):
    """The two status keys and nothing else, as the host requires of initialize/release."""

    def write_fields(self, writer):
        pass


class CounterOutputs(OutputParametersBase):
    def __init__(self):
        super().__init__()
        self.count = 0

    def write_fields(self, writer):
        writer.integer("count", self.count)


class CounterTool:
    def __init__(self):
        self.count = 0

    def initialize(self, params):
        self.count = 0
        out = LifecycleStatus()
        out.status_message = ""
        return out.pack()

    def execute(self, inputs):
        self.count += 1
        out = CounterOutputs()
        out.count = self.count
        out.status_message = "运行成功"
        return out.pack()

    def release(self):
        out = LifecycleStatus()
        out.status_message = ""
        return out.pack()
