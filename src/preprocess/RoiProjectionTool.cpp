#include "RoiProjectionTool.h"

#include "common/InputValues.h"
#include "common/UpstreamStatus.h"

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
        throw visionrt::ContractError("批次索引 " + std::to_string(index)
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
        return {1, "投影ROI不接受初始化参数"};
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
    try
    {
        const auto &imgBatch
            = requireTypedInput<visionrt::DataRef<visionrt::Image>>(inputs, p::in::IMAGES, {visionrt::Type::Image});

        const auto axis = requireString(inputs, p::in::AXIS);
        const auto polarity = requireString(inputs, p::in::POLARITY);
        if (axis != "x" && axis != "y")
        {
            throw visionrt::ContractError("输入端口 '" + std::string(p::in::AXIS) + "' 的取值无效：" + axis
                                          + "（允许 x/y）");
        }
        if (polarity != "bright" && polarity != "dark")
        {
            throw visionrt::ContractError("输入端口 '" + std::string(p::in::POLARITY) + "' 的取值无效：" + polarity
                                          + "（允许 bright/dark）");
        }

        const bool        horizontal = (axis == "x");
        const bool        bright = (polarity == "bright");
        const std::int64_t threshold = requireIntInRange(inputs, p::in::THRESHOLD, 0, 255);
        const std::int64_t minimumPixels = requireIntInRange(inputs, p::in::MIN_PIXELS, 1,
                                                             std::numeric_limits<std::int64_t>::max());
        const int minimumExtent = requireInt32(inputs, p::in::MIN_EXTENT, 1, std::numeric_limits<int>::max());
        const int gap = requireInt32(inputs, p::in::MERGE_GAP, 0, std::numeric_limits<int>::max());
        const int padding = requireInt32(inputs, p::in::PADDING, 0, std::numeric_limits<int>::max());

        if (imgBatch.get().empty())
        {
            visionrt::Values outputs;
            outputs.emplace(p::out::IMAGES, imgBatch);
            outputs.emplace(p::out::ROIS, visionrt::DataRef<visionrt::ROI>(visionrt::ROI{}));
            return visionrt::toolResult(std::move(outputs), visionrt::executionSuccess());
        }

        visionrt::ROI output;
        output.reserve(imgBatch.get().size());

        for (std::size_t index = 0; index < imgBatch.get().size(); ++index)
        {
            const auto               &image     = imgBatch.get()[index];
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
                if (end - start < minimumExtent)
                    return;
                const int left = start - std::min(padding, start);
                const int right = end + std::min(padding, length - end);
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
        outputs.emplace(p::out::IMAGES, imgBatch);
        outputs.emplace(p::out::ROIS, visionrt::DataRef<visionrt::ROI>(std::move(output)));
        return visionrt::toolResult(std::move(outputs), visionrt::executionSuccess());
    }
    catch (const visionrt::ContractError &error)
    {
        return visionrt::toolResult(missingOutputs(), {1, error.what()});
    }
    catch (const std::exception &error)
    {
        return visionrt::toolResult(missingOutputs(), {1, "投影ROI失败", error.what()});
    }
    catch (...)
    {
        return visionrt::toolResult(missingOutputs(), {-1, "投影ROI失败：未知异常"});
    }
}

visionrt::ToolStatus RoiProjectionTool::release()
{
    return {};
}

} // namespace visiontools::preprocess
