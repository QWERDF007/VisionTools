import numpy as np

class ToolInstance:
    def initialize(self, params):
        return {"__status_code": 0, "__status_message": ""}

    def execute(self, inputs):
        images = inputs.get("images")
        if images is None:
            return {"__status_code": 1,
                    "__status_message": "输入端口 'images' 缺失", "means": None}
        # 空批得到shape=(0,)的合法float64输出。
        means = np.asarray([float(image.mean()) for image in images],
                           dtype=np.float64)
        return {"__status_code": 0, "__status_message": "运行成功", "means": means}

    def release(self):
        return {"__status_code": 0, "__status_message": ""}
