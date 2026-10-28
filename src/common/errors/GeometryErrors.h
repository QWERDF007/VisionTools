#pragma once

#include "common/errors/ToolError.h"

namespace visiontools {

// Failures that belong to the image geometry itself: a transform that cannot be applied to this
// ROI, a region that leaves the image, or an alignment that is not a supported transform.
enum class GeometryError : std::int64_t
{
    AlignmentInvalid  = errorCode(domain::kImageGeometry, 1),
    RoiOutOfImage     = errorCode(domain::kImageGeometry, 2),
    SingularTransform = errorCode(domain::kImageGeometry, 3),
    UnsupportedImage  = errorCode(domain::kImageGeometry, 4)
};

inline std::string_view message(GeometryError code) noexcept
{
    switch (code)
    {
    case GeometryError::AlignmentInvalid:
        return "对齐矩阵不合法";
    case GeometryError::RoiOutOfImage:
        return "ROI 超出图像范围";
    case GeometryError::SingularTransform:
        return "几何变换存在奇点";
    case GeometryError::UnsupportedImage:
        return "图像格式不受支持";
    }
    return "几何处理失败";
}

} // namespace visiontools
