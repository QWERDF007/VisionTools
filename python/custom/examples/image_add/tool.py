"""Image-processing example: adds one to every pixel.

Policy for the injected ``upstream_status`` port: this tool changes the source image only when the
upstream status is zero. A non-zero status leaves the image untouched and publishes that status with
an empty output, so a failed branch can never look processed.

The tool uses the packaged ``visiontools_sdk`` parameter subclasses: the shared base reads
``upstream_status`` and owns the status keys, while the business decisions below stay here. Images
are carried by reference — ``inplace`` mutates the caller's ndarray, ``copy`` builds new arrays.
"""

import numpy as np

from .visiontools_sdk import ContractError, InputParametersBase, OutputParametersBase


class ImageAddInputs(InputParametersBase):
    def __init__(self):
        super().__init__()
        self.images = None
        self.mode = "copy"

    def read_fields(self, reader):
        self.images = reader.array("images")
        self.mode = reader.text("mode")

class ImageAddOutputs(OutputParametersBase):
    def __init__(self):
        super().__init__()
        self.images = None

    def write_fields(self, writer):
        if self.images is None:
            writer.missing("images")
        else:
            writer.array("images", self.images)

def _validate_batch(images):
    """Returns the per-image problem, or None when the whole batch is valid."""
    for index, image in enumerate(images):
        if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim not in (2, 3):
            return f"输入端口 'images' 批次索引 {index} 需要非空 HxW 或 HxWxC ndarray"
        if image.dtype != np.uint8:
            return f"输入端口 'images' 批次索引 {index} 需要 uint8 像素"
    return None

def execute(values):
    args = ImageAddInputs()
    try:
        args.parse(values)
    except ContractError as error:
        out = ImageAddOutputs()
        # A missing port and a mistyped one are different answers for the caller.
        out.status_code = 1001 if "为空" in str(error) else 1002
        out.status_message = str(error)
        return out.pack()

    out = ImageAddOutputs()
    if args.upstream_status != 0:
        # The tool's own decision: a failed branch is forwarded, never processed. The message says
        # so, because a non-zero status without a reason is not a usable answer.
        out.status_code = args.upstream_status
        out.status_message = "上游未选中本分支，工具未执行"
        return out.pack()

    if args.mode not in ("copy", "inplace"):
        out.status_code = 1003
        out.status_message = "输入端口 'mode' 只能是 copy 或 inplace"
        return out.pack()
    if (problem := _validate_batch(args.images)) is not None:
        out.status_code = 1002
        out.status_message = problem
        return out.pack()

    output = []
    for image in args.images:
        if args.mode == "inplace":
            image[...] = np.minimum(image, 254) + 1
            output.append(image)
        else:
            output.append(np.minimum(image, 254) + 1)
    out.images = output
    out.status_message = "运行成功"
    return out.pack()
