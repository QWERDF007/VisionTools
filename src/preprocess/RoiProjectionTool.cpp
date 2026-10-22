#include "RoiProjectionTool.h"

#include "InputPorts.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace visiontools::preprocess {
namespace {

namespace p = ports::roi_projection;

cv::Mat toGray(const visionrt::OwnedImage &image, std::size_t index)
{
    if (image.empty() || CV_MAT_DEPTH(image.type()) != CV_8U
        || (image.channels() != 1 && image.channels() != 3 && image.channels() != 4))
    {
        throw visionrt::ContractError("批次索引 " + std::to_string(index)
                                      + " 的图像必须是非空 uint8 灰度/BGR/BGRA 图像");
    }
    if (image.channels() == 1)
        return image.readOnlyMat();
    cv::Mat result;
    cv::cvtColor(image.readOnlyMat(), result, image.channels() == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY);
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

visionrt::ToolResult RoiProjectionTool::execute(const visionrt::Values &inputs)
{
    try
    {
        const auto &imageValue = requireInput(inputs, p::in::IMAGES);
        const auto *imgBatch = std::get_if<visionrt::ImageBatch>(&imageValue);
        if (!imgBatch)
        {
            throw visionrt::ContractError("输入端口 '" + std::string(p::in::IMAGES) + "' 必须是 ImageBatch");
        }

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

        if (imgBatch->images().empty())
        {
            visionrt::Values outputs;
            outputs.emplace(p::out::IMAGES, *imgBatch);
            outputs.emplace(p::out::ROIS, visionrt::RoiBatch{});
            return {visionrt::executionSuccess(), std::move(outputs)};
        }

        std::vector<std::vector<visionrt::Roi>> output;
        output.reserve(imgBatch->images().size());

        for (std::size_t index = 0; index < imgBatch->images().size(); ++index)
        {
            const auto &image = imgBatch->images()[index];
            const auto  grayscale = toGray(image, index);
            const int   length = horizontal ? image.cols() : image.rows();
            std::vector<std::int64_t> counts(static_cast<std::size_t>(length), 0);

            for (int y = 0; y < image.rows(); ++y)
            {
                const auto *row = grayscale.ptr<unsigned char>(y);
                for (int x = 0; x < image.cols(); ++x)
                {
                    if (bright ? row[x] >= threshold : row[x] < threshold)
                    {
                        ++counts[horizontal ? x : y];
                    }
                }
            }

            std::vector<visionrt::Roi> rois;
            const auto append = [&](int start, int end)
            {
                if (end - start < minimumExtent)
                    return;
                const int left = start - std::min(padding, start);
                const int right = end + std::min(padding, length - end);
                rois.push_back(
                    horizontal ? visionrt::Roi::rectangle(static_cast<float>(left), 0, static_cast<float>(right - left),
                                                          static_cast<float>(image.rows()), 0)
                               : visionrt::Roi::rectangle(0, static_cast<float>(left), static_cast<float>(image.cols()),
                                                          static_cast<float>(right - left), 0));
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
        outputs.emplace(p::out::IMAGES, *imgBatch);
        outputs.emplace(p::out::ROIS, visionrt::RoiBatch(std::move(output)));
        return {visionrt::executionSuccess(), std::move(outputs)};
    }
    catch (const visionrt::ContractError &error)
    {
        return {{1, error.what()},
                {{p::out::IMAGES, visionrt::Missing{}}, {p::out::ROIS, visionrt::Missing{}}}};
    }
    catch (const std::exception &error)
    {
        return {
            {1, "投影ROI失败", error.what()},
            {{p::out::IMAGES, visionrt::Missing{}}, {p::out::ROIS, visionrt::Missing{}}}
        };
    }
    catch (...)
    {
        return {{-1, "投影ROI失败：未知异常"},
                {{p::out::IMAGES, visionrt::Missing{}}, {p::out::ROIS, visionrt::Missing{}}}};
    }
}

visionrt::ToolStatus RoiProjectionTool::release()
{
    return {};
}

} // namespace visiontools::preprocess
