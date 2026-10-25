#include "RotateImageTool.h"

#include "common/GeometryTransform.h"
#include "common/ImageSupport.h"
#include "common/InputValues.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace visiontools::preprocess {
namespace {
namespace p = ports::image_rotate;

int interpolationFlag(const std::string &value)
{
    if (value == "nearest")
        return cv::INTER_NEAREST;
    if (value == "linear")
        return cv::INTER_LINEAR;
    if (value == "cubic")
        return cv::INTER_CUBIC;
    throw visionrt::ContractError("输入端口 'interpolation' 的取值无效：" + value + "（允许 nearest/linear/cubic）");
}

std::optional<cv::Point2d> requestedCenter(const visionrt::Values &inputs)
{
    const auto &value = requireInput(inputs, p::in::CENTER);
    const auto &array = std::get<visionrt::Array>(value);
    if (array.type() != visionrt::TypeSpec{visionrt::Type::Array, visionrt::DType::Float64, 1})
        throw visionrt::ContractError("center 必须是 Array(float64, rank=1)");
    const auto &values = array.get<visionrt::NestedVector<double, 1>>();
    if (values.empty())
        return std::nullopt;
    if (values.size() != 2)
        throw visionrt::ContractError("center 必须为空或恰好两个有限值");
    if (!std::isfinite(values[0]) || !std::isfinite(values[1]))
        throw visionrt::ContractError("center 必须为有限数值");
    return cv::Point2d{values[0], values[1]};
}

cv::Mat matrixOf(const geometry::Alignment &alignment)
{
    cv::Mat matrix(2, 3, CV_64F);
    matrix.at<double>(0, 0) = alignment.m[0];
    matrix.at<double>(0, 1) = alignment.m[1];
    matrix.at<double>(0, 2) = alignment.m[2];
    matrix.at<double>(1, 0) = alignment.m[3];
    matrix.at<double>(1, 1) = alignment.m[4];
    matrix.at<double>(1, 2) = alignment.m[5];
    return matrix;
}

visionrt::Values missingOutputs()
{
    return {{p::out::IMAGES, visionrt::Missing{}}, {p::out::ALIGNMENT, visionrt::Missing{}}};
}
} // namespace

visionrt::ToolStatus RotateImageTool::initialize(const visionrt::Values &parameters)
{
    if (!parameters.empty())
        return {1, "旋转图像不接受初始化参数"};
    return {};
}

visionrt::ToolResult RotateImageTool::execute(const visionrt::Values &inputs)
{
    try
    {
        const auto &images        = requireImages(inputs, p::in::IMAGES);
        const auto  angle         = requireFloatInRange(inputs, p::in::ANGLE_DEG, -1.0e9, 1.0e9);
        const auto  canvas        = requireString(inputs, p::in::CANVAS);
        const auto  interpolation = interpolationFlag(requireString(inputs, p::in::INTERPOLATION));
        const auto  center        = requestedCenter(inputs);
        const auto  alignments    = geometry::alignmentsFromValue(requireInput(inputs, p::in::ALIGNMENT), images.size());
        if (canvas != "keep" && canvas != "expand")
            throw visionrt::ContractError("输入端口 'canvas' 的取值无效：" + canvas + "（允许 keep/expand）");

        const double normalized = std::fmod(angle, 360.0);
        std::vector<cv::Mat>         output;
        std::vector<geometry::Alignment> outputAlignments;
        output.reserve(images.size());
        outputAlignments.reserve(images.size());
        for (std::size_t index = 0; index < images.size(); ++index)
        {
            const auto &image = images[index];
            requireSupportedImage(image, "旋转图像");
            const auto border = borderScalar(requireInput(inputs, p::in::BORDER_VALUE), image.channels(),
                                             image.depth());
            const cv::Point2d pivot = center.value_or(cv::Point2d{(image.cols - 1) / 2.0, (image.rows - 1) / 2.0});
            const auto rotation = geometry::rotationAround(pivot.x, pivot.y, normalized);
            geometry::Alignment transform = rotation;
            cv::Size            size{image.cols, image.rows};
            if (canvas == "expand")
            {
                double minX = 0, minY = 0, maxX = 0, maxY = 0;
                bool   first = true;
                for (const auto &[x, y] : {std::pair{-0.5, -0.5}, std::pair{image.cols - 0.5, -0.5},
                                           std::pair{image.cols - 0.5, image.rows - 0.5},
                                           std::pair{-0.5, image.rows - 0.5}})
                {
                    double mappedX = 0, mappedY = 0;
                    rotation.apply(x, y, mappedX, mappedY);
                    if (first)
                    {
                        minX = maxX = mappedX;
                        minY = maxY = mappedY;
                        first       = false;
                    }
                    else
                    {
                        minX = std::min(minX, mappedX);
                        minY = std::min(minY, mappedY);
                        maxX = std::max(maxX, mappedX);
                        maxY = std::max(maxY, mappedY);
                    }
                }
                const double width  = maxX - minX;
                const double height = maxY - minY;
                const int    columns = std::max(1, static_cast<int>(std::ceil(width - 1e-6)));
                const int    rows    = std::max(1, static_cast<int>(std::ceil(height - 1e-6)));
                size                 = {columns, rows};
                transform.m[2] += -0.5 - minX;
                transform.m[5] += -0.5 - minY;
            }
            cv::Mat rotated;
            cv::warpAffine(image, rotated, matrixOf(transform), size, interpolation, cv::BORDER_CONSTANT, border);
            output.push_back(std::move(rotated));
            outputAlignments.push_back(transform * alignments[index]);
        }
        visionrt::Values outputs;
        outputs.emplace(p::out::IMAGES,
                        visionrt::DataRef<visionrt::Image>(visionrt::Image(std::move(output)),
                                                           {std::get<visionrt::DataRef<visionrt::Image>>(
                                                                requireInput(inputs, p::in::IMAGES))
                                                                .owner()}));
        outputs.emplace(p::out::ALIGNMENT, geometry::alignmentArray(outputAlignments));
        return {visionrt::executionSuccess(), std::move(outputs)};
    }
    catch (const visionrt::ContractError &error)
    {
        return {{1, error.what()}, missingOutputs()};
    }
    catch (const cv::Exception &error)
    {
        return {{1, "旋转图像失败", error.what()}, missingOutputs()};
    }
    catch (const std::exception &error)
    {
        return {{1, "旋转图像失败", error.what()}, missingOutputs()};
    }
    catch (...)
    {
        return {{-1, "旋转图像失败：未知异常"}, missingOutputs()};
    }
}

visionrt::ToolStatus RotateImageTool::release()
{
    return {};
}
} // namespace visiontools::preprocess
