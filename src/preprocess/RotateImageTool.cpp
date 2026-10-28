#include "RotateImageTool.h"

#include "common/GeometryTransform.h"
#include "common/UpstreamStatus.h"
#include "common/ImageSupport.h"
#include "common/InputValues.h"
#include "common/errors/GeometryErrors.h"
#include "common/errors/ToolError.h"

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
    fail(CommonToolError::InputOutOfRange, "interpolation 只接受 nearest/linear/cubic，实际值：" + value);
}

std::optional<cv::Point2d> requestedCenter(const visionrt::Values &inputs)
{
    const auto found = inputs.find(p::in::CENTER);
    if (found == inputs.end() || visionrt::isMissing(found->second))
        return std::nullopt;
    const auto *array = std::get_if<visionrt::Array>(&found->second);
    if (!array || array->type() != visionrt::TypeSpec{visionrt::Type::Array, visionrt::DType::Float64, 1})
        fail(CommonToolError::InputTypeMismatch, std::string(p::in::CENTER) + " 必须是 Array(float64, rank=1)");
    const auto &values = array->get<visionrt::NestedVector<double, 1>>();
    if (values.empty())
        return std::nullopt;
    if (values.size() != 2)
        fail(CommonToolError::InputOutOfRange, std::string(p::in::CENTER) + " 必须为空或恰好两个有限值");
    if (!std::isfinite(values[0]) || !std::isfinite(values[1]))
        fail(CommonToolError::InputOutOfRange, std::string(p::in::CENTER) + " 必须为有限数值");
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
        return {visionrt::kContractViolationStatus, "旋转图像不接受初始化参数"};
    return {};
}

visionrt::ToolResult RotateImageTool::execute(const visionrt::Values &inputs)
{
    // An unselected upstream branch is not an error: no algorithm runs and nothing is written.
    if (const auto upstream = upstreamStatus(inputs); upstream != 0)
        return skippedByUpstream(upstream, missingOutputs());
    try
    {
        Readings   readings(inputs);
        const auto imagesRef     = readings.image(p::in::IMAGES);
        const auto angle         = readings.real(p::in::ANGLE_DEG);
        const auto canvas        = readings.text(p::in::CANVAS);
        const auto interpolationName = readings.text(p::in::INTERPOLATION);
        const auto alignmentValue = readings.find(p::in::ALIGNMENT);
        const auto borderValue    = readings.find(p::in::BORDER_VALUE);
        if (readings.failed())
            return visionrt::toolResult(missingOutputs(),
                                        readings.status(CommonToolError::InputMissing, CommonToolError::InputTypeMismatch));
        const auto &images = imagesRef.get();
        if (images.empty())
            return failure(missingOutputs(), CommonToolError::InputMissing, std::string(p::in::IMAGES) + " 没有图像");
        if (!std::isfinite(angle) || angle < -1.0e9 || angle > 1.0e9)
            return failure(missingOutputs(), CommonToolError::InputOutOfRange,
                           std::string(p::in::ANGLE_DEG) + " 必须在 [-1e9, 1e9] 内，实际值：" + std::to_string(angle));
        if (canvas != "keep" && canvas != "expand")
            return failure(missingOutputs(), CommonToolError::InputOutOfRange,
                           std::string(p::in::CANVAS) + " 只接受 keep/expand，实际值：" + canvas);
        if (!alignmentValue)
            return failure(missingOutputs(), CommonToolError::InputMissing, std::string(p::in::ALIGNMENT));
        if (!borderValue)
            return failure(missingOutputs(), CommonToolError::InputMissing, std::string(p::in::BORDER_VALUE));
        const auto interpolation = interpolationFlag(interpolationName);
        const auto center        = requestedCenter(inputs);
        const auto alignments    = geometry::alignmentsFromValue(*alignmentValue, images.size());

        const double normalized = std::fmod(angle, 360.0);
        std::vector<cv::Mat>         output;
        std::vector<geometry::Alignment> outputAlignments;
        output.reserve(images.size());
        outputAlignments.reserve(images.size());
        for (std::size_t index = 0; index < images.size(); ++index)
        {
            const auto &image = images[index];
            requireSupportedImage(image, "旋转图像");
            const auto border = borderScalar(*borderValue, image.channels(), image.depth());
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
        // warpAffine allocates every output plane, so the result owns its pixels. Keeping the
        // upstream owner would only pin the reader's decoded cache for as long as this result lives.
        visionrt::Values outputs;
        outputs.emplace(p::out::IMAGES, visionrt::DataRef<visionrt::Image>(visionrt::Image(std::move(output))));
        outputs.emplace(p::out::ALIGNMENT, geometry::alignmentArray(outputAlignments));
        return visionrt::toolResult(std::move(outputs), visionrt::executionSuccess());
    }
    catch (const ToolFailure &error)
    {
        return failureResult(missingOutputs(), error);
    }
    catch (const std::exception &error)
    {
        return failure(missingOutputs(), CommonToolError::OperatorFailed, "旋转图像失败", error.what());
    }
    catch (...)
    {
        return failure(missingOutputs(), CommonToolError::OperatorFailed, "旋转图像失败：未知异常");
    }
}

visionrt::ToolStatus RotateImageTool::release()
{
    return {};
}
} // namespace visiontools::preprocess
