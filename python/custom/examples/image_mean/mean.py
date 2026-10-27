import numpy as np

class ToolInstance:
    """Measurement example.

    Policy for ``upstream_status``: this tool never writes the pixels it reads, so it keeps
    measuring a batch that arrived from a failed branch and reports its own measurement status.
    Tools that modify their inputs must instead stop and publish the upstream status.
    """

    def initialize(self, params):
        return {"__status_code": 0, "__status_message": ""}

    def execute(self, inputs):
        images = inputs.get("images")
        if images is None:
            return {"__status_code": 1,
                    "__status_message": "输入端口 'images' 缺失", "means": None}
        if not isinstance(images, list):
            return {"__status_code": 1,
                    "__status_message": "输入端口 'images' 需要图像 list", "means": None}
        for index, image in enumerate(images):
            if not isinstance(image, np.ndarray):
                return {"__status_code": 1,
                        "__status_message": f"输入端口 'images' 批次索引 {index} 需要 NumPy ndarray",
                        "means": None}
            if image.ndim not in (2, 3) or image.size == 0:
                return {"__status_code": 1,
                        "__status_message": (f"输入端口 'images' 批次索引 {index} 图像形状 {image.shape} "
                                             "必须为非空 HxW 或 HxWxC"),
                        "means": None}
            if image.dtype.kind not in "uif":
                return {"__status_code": 1,
                        "__status_message": (f"输入端口 'images' 批次索引 {index} "
                                             f"图像像素需要实数数值类型，实际 {image.dtype}"),
                        "means": None}
        means = [float(image.mean()) for image in images]
        return {"__status_code": 0, "__status_message": "运行成功", "means": means}

    def release(self):
        return {"__status_code": 0, "__status_message": ""}
