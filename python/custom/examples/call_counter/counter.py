"""Stateful counter example: keeps a class entry and uses the author SDK for outputs.

The counter is the canonical stateful Python class: ``initialize`` resets the count,
``execute`` increments it, ``release`` reports an empty success. Outputs go through the
packaged ``visiontools_sdk`` output subclass, so the envelope has one spelling. Lifecycle
calls return the status dict alone; only ``execute`` publishes the envelope.
"""

from .visiontools_sdk import OutputParametersBase, status


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
        return status()

    def execute(self, inputs):
        self.count += 1
        out = CounterOutputs()
        out.count = self.count
        out.status_message = "运行成功"
        return out.pack()

    def release(self):
        return status()
