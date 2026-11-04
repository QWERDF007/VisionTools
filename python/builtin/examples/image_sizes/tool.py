"""Example tool: reports the pixel size of each input image.

Policy for the injected ``upstream_status`` port: a read-only measurement keeps running after an
upstream failure and reports its own status; only tools that modify data must stop.
"""

IMAGES = 1
SIZES = 2
STATUS_CODE, STATUS_MESSAGE = 0, 1


def execute(inputs):
    images = inputs.get(IMAGES)
    if not isinstance(images, list):
        return {
            "outputs": {
                SIZES: None,
                STATUS_CODE: 1002,
                STATUS_MESSAGE: "输入端口 1 需要图像 list",
            },
            "detail": "",
        }
    return {
        "outputs": {
            SIZES: [[int(image.shape[0]), int(image.shape[1])] for image in images],
            STATUS_CODE: 0,
            STATUS_MESSAGE: "运行成功",
        },
        "detail": "",
    }
