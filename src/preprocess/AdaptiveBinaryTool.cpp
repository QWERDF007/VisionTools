#include "AdaptiveBinaryTool.h"

#include "common/InputValues.h"
#include "common/errors/ToolError.h"
#include "common/UpstreamStatus.h"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace visiontools::preprocess {
namespace {

namespace p = ports::adaptive_binary;

} // namespace

AdaptiveBinaryTool::AdaptiveBinaryTool(const visionrt::ToolServices &services)
    : services_(services)
{
}

visionrt::ToolStatus AdaptiveBinaryTool::initialize(const visionrt::Values &parameters)
{
    if (!parameters.empty())
        return {visionrt::kContractViolationStatus, "自适应二值化不接受初始化参数"};
    return {};
}

namespace {
visionrt::Values missingOutputs()
{
    return {{p::out::IMAGES, visionrt::Missing{}}};
}
} // namespace

visionrt::ToolResult AdaptiveBinaryTool::execute(const visionrt::Values &inputs)
{
    // An unselected upstream branch is not an error: no algorithm runs and nothing is written.
    if (const auto upstream = upstreamStatus(inputs); upstream != 0)
        return skippedByUpstream(upstream, missingOutputs());

    // An empty batch needs no threshold, so the images are read first: a caller that asked for
    // nothing gets an empty answer instead of a complaint about a parameter it never needed.
    Readings   readings(inputs);
    const auto images = readings.image(p::in::IMAGES);
    if (readings.failed())
        return visionrt::toolResult(missingOutputs(),
                                    readings.status(CommonToolError::InputMissing, CommonToolError::InputTypeMismatch));

    const auto batchSize = images.get().size();
    if (batchSize == 0)
    {
        visionrt::Values outputs;
        outputs.emplace(p::out::IMAGES, visionrt::DataRef<visionrt::Image>(visionrt::Image{}));
        return visionrt::toolResult(std::move(outputs), visionrt::executionSuccess());
    }

    const auto blockSize      = readings.integer(p::in::BLOCK_SIZE);
    const auto constantOffset = readings.real(p::in::CONSTANT_OFFSET);
    if (readings.failed())
        return visionrt::toolResult(missingOutputs(),
                                    readings.status(CommonToolError::InputMissing, CommonToolError::InputTypeMismatch));
    if (blockSize < 3 || blockSize > 999 || blockSize % 2 == 0)
        return failure(missingOutputs(), CommonToolError::InputOutOfRange,
                       std::string(p::in::BLOCK_SIZE) + " 必须在 [3, 999] 内且为奇数，实际值："
                           + std::to_string(blockSize));
    if (!std::isfinite(constantOffset))
        return failure(missingOutputs(), CommonToolError::InputOutOfRange,
                       std::string(p::in::CONSTANT_OFFSET) + " 必须为有限数值");
    for (std::size_t index = 0; index < batchSize; ++index)
    {
        const auto &image = images.get()[index];
        if (image.empty())
            return failure(missingOutputs(), CommonToolError::UnsupportedImage,
                           "批次索引 " + std::to_string(index) + " 的图像为空");
        if (image.type() != CV_8UC1)
            return failure(missingOutputs(), CommonToolError::UnsupportedImage,
                           "批次索引 " + std::to_string(index) + " 的图像必须是 CV_8UC1");
    }
    try
    {
        visionrt::Image outputImages;
        outputImages.reserve(batchSize);
        for (const auto &image : images.get())
        {
            cv::Mat binary;
            cv::adaptiveThreshold(image, binary, 255.0, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY,
                                  static_cast<int>(blockSize), constantOffset);
            outputImages.push_back(std::move(binary));
        }
        visionrt::Values outputs;
        outputs.emplace(p::out::IMAGES, visionrt::DataRef<visionrt::Image>(std::move(outputImages)));
        return visionrt::toolResult(std::move(outputs), visionrt::executionSuccess());
    }
    catch (const cv::Exception &error)
    {
        return failure(missingOutputs(), CommonToolError::OperatorFailed, {}, error.what());
    }
    catch (...)
    {
        return failure(missingOutputs(), CommonToolError::OperatorFailed, "未知异常");
    }
}

visionrt::ToolStatus AdaptiveBinaryTool::release()
{
    return {};
}

} // namespace visiontools::preprocess
