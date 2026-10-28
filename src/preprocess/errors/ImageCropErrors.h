#pragma once

#include "common/errors/ToolError.h"

namespace visiontools::preprocess {

enum class ImageCropError : std::int64_t
{
    RoiStructure = errorCode(domain::kImageCrop, 1),
    OutOfBounds  = errorCode(domain::kImageCrop, 2),
    EmptyRegion  = errorCode(domain::kImageCrop, 3)
};

inline std::string_view message(ImageCropError code) noexcept
{
    switch (code)
    {
    case ImageCropError::RoiStructure:
        return "裁剪 ROI 结构不合法";
    case ImageCropError::OutOfBounds:
        return "裁剪区域超出图像范围";
    case ImageCropError::EmptyRegion:
        return "裁剪区域没有有效像素中心";
    }
    return "图像裁剪失败";
}

} // namespace visiontools::preprocess
