"""Example tool: reports the pixel size of each input image.

Policy for the injected ``upstream_status`` port: a read-only measurement keeps running after an
upstream failure and reports its own status; only tools that modify data must stop.
"""


def execute(inputs):
    images = inputs.get("images")
    if not isinstance(images, list):
        return {
            "outputs": {
                "sizes": None,
                "__status_code": 1002,
                "__status_message": "输入端口 'images' 需要图像 list",
            },
            "detail": "",
        }
    return {
        "outputs": {
            "sizes": [[int(image.shape[0]), int(image.shape[1])] for image in images],
            "__status_code": 0,
            "__status_message": "运行成功",
        },
        "detail": "",
    }
