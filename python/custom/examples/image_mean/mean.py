"""Measurement example: reports the mean value of each input image.

Policy for the injected ``upstream_status`` port: this tool never writes the pixels it reads, so it
keeps measuring a batch that arrived from a failed branch and reports its own measurement status.
Tools that modify their inputs must instead stop and publish the upstream status.

Every failure is a returned answer: a declared port stays present with ``None`` where there is no
value, and the status carries the reason. Nothing here raises to control the flow.
"""

import numpy as np

from .visiontools_sdk import ContractError, InputReader, failure, success


# Port-level problems use the shared common codes; a tool with algorithm-specific failures adds
# its own numbers in the same file, never reusing these.
MISSING_INPUT = 1001
INVALID_INPUT = 1002

def _check_batch(images):
    """Returns the first unusable image as a readable sentence, or None for a valid batch."""
    for index, image in enumerate(images):
        if not isinstance(image, np.ndarray):
            return (f"输入端口 'images' 批次索引 {index} 需要 NumPy ndarray，"
                    f"实际 {type(image).__name__}")
        if image.ndim not in (2, 3) or image.size == 0:
            return (f"输入端口 'images' 批次索引 {index} 图像形状 {image.shape} "
                    "必须为非空 HxW 或 HxWxC")
        if image.dtype.kind not in "uif":
            return (f"输入端口 'images' 批次索引 {index} 图像像素需要实数数值类型，"
                    f"实际 {image.dtype}")
    return None

class ToolInstance:
    def initialize(self, params):
        return {"code": 0, "message": "", "detail": ""}

    def execute(self, inputs):
        try:
            images = InputReader(inputs).optional_array("images")
        except ContractError as error:
            return failure(INVALID_INPUT, str(error), outputs={"means": None})
        if images is None:
            return failure(MISSING_INPUT, "输入端口 'images' 缺失", outputs={"means": None})

        problem = _check_batch(images)
        if problem is not None:
            return failure(INVALID_INPUT, problem, outputs={"means": None})

        return success({"means": [float(image.mean()) for image in images]})

    def release(self):
        return {"code": 0, "message": "", "detail": ""}
