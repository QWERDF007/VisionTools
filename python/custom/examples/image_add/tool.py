import numpy as np


def error(message):
    return {"__status_code": 1, "__status_message": message, "images": None}


def execute(inputs):
    images = inputs.get("images")
    mode = inputs.get("mode", "copy")
    if not isinstance(images, list):
        return error("输入端口 'images' 需要图像 list")
    if mode not in ("copy", "inplace"):
        return error("输入端口 'mode' 只能是 copy 或 inplace")
    output = []
    for index, image in enumerate(images):
        if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim not in (2, 3):
            return error(f"输入端口 'images' 批次索引 {index} 需要非空 HxW 或 HxWxC ndarray")
        if image.dtype != np.uint8:
            return error(f"输入端口 'images' 批次索引 {index} 需要 uint8 像素")
        if mode == "inplace":
            image[...] = np.minimum(image, 254) + 1
            output.append(image)
        else:
            output.append(np.minimum(image, 254) + 1)
    return {"__status_code": 0, "__status_message": "运行成功", "images": output}
