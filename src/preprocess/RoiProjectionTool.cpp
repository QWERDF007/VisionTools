#include "RoiProjectionTool.h"

#include "common/InputValues.h"
#include "common/UpstreamStatus.h"
#include "common/errors/GeometryErrors.h"
#include "common/errors/ToolError.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace visiontools::preprocess {
namespace {

namespace p = ports::roi_projection;

cv::Mat toGray(const cv::Mat &image, std::size_t index)
{
    if (image.empty() || CV_MAT_DEPTH(image.type()) != CV_8U
        || (image.channels() != 1 && image.channels() != 3 && image.channels() != 4))
    {
        fail(CommonToolError::UnsupportedImage, "批次索引 " + std::to_string(index)
                                                   + " 的图像必须是非空 uint8 灰度/BGR/BGRA 图像");
    }
    if (image.channels() == 1)
        return image;
    cv::Mat result;
    cv::cvtColor(image, result, image.channels() == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY);
    return result;
}

} // namespace

RoiProjectionTool::RoiProjectionTool(const visionrt::ToolServices &services)
    : services_(services)
{
}

visionrt::ToolStatus RoiProjectionTool::initialize(const visionrt::Values &parameters)
{
    if (!parameters.empty())
    {
        return {visionrt::kContractViolationStatus, "投影ROI不接受初始化参数"};
    }
    return {};
}

namespace {
visionrt::Values missingOutputs()
{
    return {{p::out::IMAGES, visionrt::Missing{}}, {p::out::ROIS, visionrt::Missing{}}};
}
} // namespace

visionrt::ToolResult RoiProjectionTool::execute(const visionrt::Values &inputs)
{
    // An unselected upstream branch is not an error: no algorithm runs and nothing is written.
    if (const auto upstream = upstreamStatus(inputs); upstream != 0)
        return skippedByUpstream(upstream, missingOutputs());
    Readings   readings(inputs);
    const auto imgBatchRef = readings.image(p::in::IMAGES);
    const auto axis          = readings.text(p::in::AXIS);
    const auto polarity      = readings.text(p::in::POLARITY);
    const auto threshold     = readings.integer(p::in::THRESHOLD);
    const auto minimumPixels = readings.integer(p::in::MIN_PIXELS);
    const auto minimumExtent = readings.integer(p::in::MIN_EXTENT);
    const auto mergeGap      = readings.integer(p::in::MERGE_GAP);
    const auto padding       = readings.integer(p::in::PADDING);
    if (readings.failed())
        return visionrt::toolResult(missingOutputs(),
                                    readings.status(CommonToolError::InputMissing, CommonToolError::InputTypeMismatch));
    if (axis != "x" && axis != "y")
        return failure(missingOutputs(), CommonToolError::InputOutOfRange,
                       std::string(p::in::AXIS) + " 只接受 x/y，实际值：" + axis);
    if (polarity != "bright" && polarity != "dark")
        return failure(missingOutputs(), CommonToolError::InputOutOfRange,
                       std::string(p::in::POLARITY) + " 只接受 bright/dark，实际值：" + polarity);
    if (threshold < 0 || threshold > 255)
        return failure(missingOutputs(), CommonToolError::InputOutOfRange,
                       std::string(p::in::THRESHOLD) + " 必须在 [0, 255] 内，实际值：" + std::to_string(threshold));
    if (minimumPixels < 1)
        return failure(missingOutputs(), CommonToolError::InputOutOfRange,
                       std::string(p::in::MIN_PIXELS) + " 必须为正整数，实际值：" + std::to_string(minimumPixels));
    if (minimumExtent < 1 || minimumExtent > std::numeric_limits<int>::max())
        return failure(missingOutputs(), CommonToolError::InputOutOfRange,
                       std::string(p::in::MIN_EXTENT) + " 超出整数范围，实际值：" + std::to_string(minimumExtent));
    if (mergeGap < 0 || mergeGap > std::numeric_limits<int>::max())
        return failure(missingOutputs(), CommonToolError::InputOutOfRange,
                       std::string(p::in::MERGE_GAP) + " 超出整数范围，实际值：" + std::to_string(mergeGap));
    if (padding < 0 || padding > std::numeric_limits<int>::max())
        return failure(missingOutputs(), CommonToolError::InputOutOfRange,
                       std::string(p::in::PADDING) + " 超出整数范围，实际值：" + std::to_string(padding));
    try
    {
        const auto &imgBatch = imgBatchRef.get();

        const bool horizontal = (axis == "x");
        const bool bright     = (polarity == "bright");
        const int  gap        = static_cast<int>(mergeGap);
        const int  extentMin  = static_cast<int>(minimumExtent);
        const int  paddingPx  = static_cast<int>(padding);

        if (imgBatch.empty())
        {
            visionrt::Values outputs;
            outputs.emplace(p::out::IMAGES, imgBatchRef);
            outputs.emplace(p::out::ROIS, visionrt::DataRef<visionrt::ROI>(visionrt::ROI{}));
            return visionrt::toolResult(std::move(outputs), visionrt::executionSuccess());
        }

        visionrt::ROI output;
        output.reserve(imgBatch.size());

        for (std::size_t index = 0; index < imgBatch.size(); ++index)
        {
            const auto               &image     = imgBatch[index];
            const auto  grayscale = toGray(image, index);
            const int                 length    = horizontal ? image.cols : image.rows;
            std::vector<std::int64_t> counts(static_cast<std::size_t>(length), 0);

            for (int y = 0; y < image.rows; ++y)
            {
                const auto *row = grayscale.ptr<unsigned char>(y);
                for (int x = 0; x < image.cols; ++x)
                {
                    if (bright ? row[x] >= threshold : row[x] < threshold)
                    {
                        ++counts[horizontal ? x : y];
                    }
                }
            }

            std::vector<std::vector<double>> rois;
            const auto append = [&](int start, int end)
            {
                if (end - start < extentMin)
                    return;
                const int left = start - std::min(paddingPx, start);
                const int right = end + std::min(paddingPx, length - end);
                rois.push_back(
                    horizontal ? std::vector<double>{2, static_cast<double>(left), 0, static_cast<double>(right - left),
                                                     static_cast<double>(image.rows), 0}
                               : std::vector<double>{2, 0, static_cast<double>(left), static_cast<double>(image.cols),
                                                     static_cast<double>(right - left), 0});
            };

            int start = -1;
            int end = -1;
            for (int position = 0; position < length; ++position)
            {
                if (counts[static_cast<std::size_t>(position)] < minimumPixels)
                    continue;
                if (start < 0)
                    start = position;
                else if (position - end > gap)
                {
                    append(start, end);
                    start = position;
                }
                end = position + 1;
            }
            if (start >= 0)
                append(start, end);
            output.push_back(std::move(rois));
        }

        visionrt::Values outputs;
        outputs.emplace(p::out::IMAGES, imgBatchRef);
        outputs.emplace(p::out::ROIS, visionrt::DataRef<visionrt::ROI>(std::move(output)));
        return visionrt::toolResult(std::move(outputs), visionrt::executionSuccess());
    }
    catch (const ToolFailure &error)
    {
        return failureResult(missingOutputs(), error);
    }
    catch (const std::exception &error)
    {
        return failure(missingOutputs(), CommonToolError::OperatorFailed, "投影ROI失败", error.what());
    }
    catch (...)
    {
        return failure(missingOutputs(), CommonToolError::OperatorFailed, "投影ROI失败：未知异常");
    }
}

visionrt::ToolStatus RoiProjectionTool::release()
{
    return {};
}

} // namespace visiontools::preprocess
