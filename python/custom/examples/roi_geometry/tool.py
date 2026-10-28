"""Geometry example: turns each image extent into one rectangle ROI.

The tool only reads, so an unselected upstream branch does not stop it: it reports its own result.
A failure returns the declared port with ``None`` and a reason, never an exception.
"""

import numpy as np

from .visiontools_sdk import failure, success


INVALID_INPUT = 1002

def execute(inputs):
    images = inputs.get("images")
    if not isinstance(images, list):
        return failure(INVALID_INPUT, "输入端口 'images' 需要图像 list", outputs={"rois": None})
    rois = []
    for index, image in enumerate(images):
        if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim not in (2, 3):
            return failure(INVALID_INPUT,
                           f"输入端口 'images' 批次索引 {index} 需要非空 HxW 或 HxWxC ndarray",
                           outputs={"rois": None})
        height, width = int(image.shape[0]), int(image.shape[1])
        rois.append([[2.0, 0.0, 0.0, float(width), float(height), 0.0]])
    return success({"rois": rois})
