#pragma once

#include "common/InputValues.h"

#include <visionrt/contracts/Tool.h>

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace visiontools {

inline const std::vector<cv::Mat> &requireImages(const visionrt::Values &inputs, std::string_view id = "images")
{
    const auto &value = requireInput(inputs, id);
    const auto *images = std::get_if<visionrt::DataRef<visionrt::Image>>(&value);
    if (!images)
        throw visionrt::ContractError("输入端口 '" + std::string(id) + "' 必须是图像");
    if (images->get().empty())
        throw visionrt::ContractError("输入端口 '" + std::string(id) + "' 没有图像");
    return images->get();
}

// Image tools share the same supported depth/channel set; unsupported kinds fail
// instead of being silently converted.
inline void requireSupportedImage(const cv::Mat &image, const char *tool)
{
    if (image.empty())
        throw visionrt::ContractError(std::string(tool) + " 收到空图像");
    const int depth = image.depth();
    if (depth != CV_8U && depth != CV_16U && depth != CV_16S && depth != CV_32F && depth != CV_64F)
        throw visionrt::ContractError(std::string(tool) + " 不支持该图像深度");
    if (image.channels() != 1 && image.channels() != 3 && image.channels() != 4)
        throw visionrt::ContractError(std::string(tool) + " 只支持 1、3 或 4 通道图像");
}

inline std::pair<double, double> requireDepthRange(int depth)
{
    switch (depth)
    {
    case CV_8U:
        return {0.0, 255.0};
    case CV_16U:
        return {0.0, 65535.0};
    case CV_16S:
        return {-32768.0, 32767.0};
    default:
        return {-1e308, 1e308};
    }
}

// border_value carries one scalar or one value per channel, in the image dtype range.
inline cv::Scalar borderScalar(const visionrt::PortValue &value, int channels, int depth)
{
    if (visionrt::isMissing(value))
        throw visionrt::ContractError("边界填充值缺失");
    const auto &array = std::get<visionrt::Array>(value);
    if (array.type() != visionrt::TypeSpec{visionrt::Type::Array, visionrt::DType::Float64, 1})
        throw visionrt::ContractError("border_value 必须是 Array(float64, rank=1)");
    const auto &values = array.get<visionrt::NestedVector<double, 1>>();
    if (values.size() != 1 && values.size() != static_cast<std::size_t>(channels))
        throw visionrt::ContractError("border_value 长度必须为 1 或图像通道数");
    const auto [minimum, maximum] = requireDepthRange(depth);
    for (const double item : values)
        if (!std::isfinite(item) || item < minimum || item > maximum)
            throw visionrt::ContractError("border_value 超出图像深度可表示范围");
    cv::Scalar scalar{values[0], values[0], values[0], values[0]};
    for (int channel = 0; channel < channels; ++channel)
        scalar[channel] = values.size() == 1 ? values[0] : values[static_cast<std::size_t>(channel)];
    return scalar;
}

} // namespace visiontools
