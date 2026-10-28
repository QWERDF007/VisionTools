#include "ImagePreprocessTool.h"

#include "common/InputValues.h"
#include "common/UpstreamStatus.h"
#include "common/errors/ToolError.h"

#include <opencv2/imgproc.hpp>

#include <limits>
#include <vector>

namespace visiontools::preprocess {
namespace {

namespace p = ports::image_preprocess;

void requireU8(const cv::Mat &image, std::size_t index)
{
    if (image.empty() || CV_MAT_DEPTH(image.type()) != CV_8U
        || (image.channels() != 1 && image.channels() != 3 && image.channels() != 4))
    {
        fail(CommonToolError::UnsupportedImage, "批次索引 " + std::to_string(index)
                                                   + " 的图像必须是非空 uint8 灰度/BGR/BGRA 图像");
    }
}

cv::Mat toGray(const cv::Mat &image, std::size_t index)
{
    requireU8(image, index);
    if (image.channels() == 1)
        return image;
    cv::Mat result;
    cv::cvtColor(image, result, image.channels() == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY);
    return result;
}

} // namespace

ImagePreprocessTool::ImagePreprocessTool(const visionrt::ToolServices &services)
    : services_(services)
{
}

visionrt::ToolStatus ImagePreprocessTool::initialize(const visionrt::Values &parameters)
{
    if (!parameters.empty())
    {
        return {visionrt::kContractViolationStatus, "图像预处理不接受初始化参数"};
    }
    return {};
}

namespace {
visionrt::Values missingOutputs()
{
    return {{p::out::IMAGES, visionrt::Missing{}}};
}
} // namespace

visionrt::ToolResult ImagePreprocessTool::execute(const visionrt::Values &inputs)
{
    // An unselected upstream branch is not an error: no algorithm runs and nothing is written.
    if (const auto upstream = upstreamStatus(inputs); upstream != 0)
        return skippedByUpstream(upstream, missingOutputs());
    // Only the parameter the chosen operation consumes is read: asking for a threshold while
    // running `identity` would reject a batch that never needed one.
    Readings   readings(inputs);
    const auto images    = readings.image(p::in::IMAGES);
    const auto operation = readings.text(p::in::OPERATION);
    if (readings.failed())
        return visionrt::toolResult(missingOutputs(),
                                    readings.status(CommonToolError::InputMissing, CommonToolError::InputTypeMismatch));
    if (operation != "identity" && operation != "gray" && operation != "gaussian" && operation != "threshold")
        return failure(missingOutputs(), CommonToolError::InputOutOfRange,
                       std::string(p::in::OPERATION) + " 只接受 identity/gray/gaussian/threshold，实际值：" + operation);

    int                kernel    = 0;
    std::int64_t       threshold = 0;
    if (operation == "gaussian")
    {
        const auto kernelValue = readings.integer(p::in::KERNEL);
        if (readings.failed())
            return visionrt::toolResult(missingOutputs(),
                                        readings.status(CommonToolError::InputMissing,
                                                        CommonToolError::InputTypeMismatch));
        if (kernelValue < 1 || kernelValue > std::numeric_limits<int>::max() || kernelValue % 2 == 0)
            return failure(missingOutputs(), CommonToolError::InputOutOfRange,
                           std::string(p::in::KERNEL) + " 必须为正的奇数（高斯核），实际值："
                               + std::to_string(kernelValue));
        kernel = static_cast<int>(kernelValue);
    }
    else if (operation == "threshold")
    {
        const auto thresholdRaw = readings.integer(p::in::THRESHOLD);
        if (readings.failed())
            return visionrt::toolResult(missingOutputs(),
                                        readings.status(CommonToolError::InputMissing,
                                                        CommonToolError::InputTypeMismatch));
        if (thresholdRaw < 0 || thresholdRaw > 255)
            return failure(missingOutputs(), CommonToolError::InputOutOfRange,
                           std::string(p::in::THRESHOLD) + " 必须在 [0, 255] 内，实际值："
                               + std::to_string(thresholdRaw));
        threshold = thresholdRaw;
    }

    try
    {
        if (images.get().empty())
        {
            visionrt::Values outputs;
            outputs.emplace(p::out::IMAGES, visionrt::DataRef<visionrt::Image>(visionrt::Image{}));
            return visionrt::toolResult(std::move(outputs), visionrt::executionSuccess());
        }

        for (std::size_t index = 0; index < images.get().size(); ++index)
        {
            requireU8(images.get()[index], index);
        }

        if (operation == "identity")
        {
            visionrt::Values outputs;
            outputs.emplace(p::out::IMAGES, images);
            return visionrt::toolResult(std::move(outputs), visionrt::executionSuccess());
        }

        visionrt::Image output;
        output.reserve(images.get().size());
        for (std::size_t index = 0; index < images.get().size(); ++index)
        {
            const auto &image = images.get()[index];
            cv::Mat processed;
            if (operation == "gaussian")
            {
                cv::GaussianBlur(image, processed, cv::Size(kernel, kernel), 0, 0, cv::BORDER_REFLECT_101);
            }
            else if (operation == "threshold")
            {
                cv::compare(toGray(image, index), cv::Scalar(static_cast<double>(threshold)), processed, cv::CMP_GE);
            }
            else
            {
                processed = image.channels() == 1 ? image.clone() : toGray(image, index);
            }
            output.push_back(std::move(processed));
        }

        visionrt::Values outputs;
        outputs.emplace(p::out::IMAGES, visionrt::DataRef<visionrt::Image>(std::move(output)));
        return visionrt::toolResult(std::move(outputs), visionrt::executionSuccess());
    }
    catch (const visionrt::ContractError &)
    {
        throw;
    }
    catch (const ToolFailure &error)
    {
        return failureResult(missingOutputs(), error);
    }
    catch (const std::exception &error)
    {
        return failure(missingOutputs(), CommonToolError::OperatorFailed, "图像预处理失败", error.what());
    }
    catch (...)
    {
        return failure(missingOutputs(), CommonToolError::OperatorFailed, "图像预处理失败：未知异常");
    }
}

visionrt::ToolStatus ImagePreprocessTool::release()
{
    return {};
}

} // namespace visiontools::preprocess
