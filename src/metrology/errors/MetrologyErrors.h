#pragma once

#include "common/errors/ToolError.h"

namespace visiontools::metrology {

// Measuring a real feature fails for reasons the parameters alone cannot express: the ROI is not
// the shape the tool measures, the fit has too few points, or the search found no edge.
enum class MetrologyError : std::int64_t
{
    RoiNotConfigured   = errorCode(domain::kMetrology, 1),
    RoiKindMismatch    = errorCode(domain::kMetrology, 2),
    InsufficientPoints = errorCode(domain::kMetrology, 3),
    SingularFit        = errorCode(domain::kMetrology, 4),
    SampleOutOfBounds  = errorCode(domain::kMetrology, 5),
    LineNotFound       = errorCode(domain::kMetrology, 6),
    MeasurementFailed  = errorCode(domain::kMetrology, 7)
};

inline std::string_view message(MetrologyError code) noexcept
{
    switch (code)
    {
    case MetrologyError::RoiNotConfigured:
        return "测量 ROI 未配置";
    case MetrologyError::RoiKindMismatch:
        return "ROI 类型与测量工具不匹配";
    case MetrologyError::InsufficientPoints:
        return "有效拟合点数量不足";
    case MetrologyError::SingularFit:
        return "拟合结果不唯一";
    case MetrologyError::SampleOutOfBounds:
        return "采样位置超出图像范围";
    case MetrologyError::LineNotFound:
        return "未检测到目标边缘";
    case MetrologyError::MeasurementFailed:
        return "测量计算失败";
    }
    return "测量失败";
}

} // namespace visiontools::metrology
