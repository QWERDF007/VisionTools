import numpy as np


def execute(inputs):
    images = inputs.get("images")
    if not isinstance(images, list):
        return {
            "__status_code": 1,
            "__status_message": "输入端口 'images' 需要图像 list",
            "rois": None,
        }
    rois = []
    for index, image in enumerate(images):
        if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim not in (2, 3):
            return {
                "__status_code": 1,
                "__status_message": (
                    f"输入端口 'images' 批次索引 {index} 需要非空 HxW 或 HxWxC ndarray"
                ),
                "rois": None,
            }
        height, width = int(image.shape[0]), int(image.shape[1])
        rois.append([[2.0, 0.0, 0.0, float(width), float(height), 0.0]])
    return {"__status_code": 0, "__status_message": "运行成功", "rois": rois}
