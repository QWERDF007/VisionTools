#include "RoiProjectionTool.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace visiontools::preprocess {
namespace {

cv::Mat toGray(const visionrt::OwnedImage &image)
{
    if (image.empty() || CV_MAT_DEPTH(image.type()) != CV_8U
        || (image.channels() != 1 && image.channels() != 3 && image.channels() != 4))
    {
        throw visionrt::ContractError("Expected nonempty uint8 gray/BGR/BGRA image");
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
        return {1, "RoiProjectionTool accepts no init parameters"};
    }
    return {};
}

visionrt::ToolResult RoiProjectionTool::execute(const visionrt::Values &inputs)
{
    try
    {
        auto imgIt = inputs.find("images");
        if (imgIt == inputs.end() || visionrt::isMissing(imgIt->second))
        {
            throw visionrt::ContractError("Missing input: images");
        }
        const auto *imgBatch = std::get_if<visionrt::ImageBatch>(&imgIt->second);
        if (!imgBatch)
            throw visionrt::ContractError("Expected ImageBatch for input: images");

        auto getStr = [&](const std::string &key, const std::string &defVal) -> std::string
        {
            auto it = inputs.find(key);
            if (it == inputs.end() || visionrt::isMissing(it->second))
                return defVal;
            if (auto *u = std::get_if<visionrt::Utf8>(&it->second))
                return u->str();
            throw visionrt::ContractError("Expected string for input: " + key);
        };

        auto getInt = [&](const std::string &key, std::int64_t defVal) -> std::int64_t
        {
            auto it = inputs.find(key);
            if (it == inputs.end() || visionrt::isMissing(it->second))
                return defVal;
            if (auto *v = std::get_if<std::int64_t>(&it->second))
                return *v;
            throw visionrt::ContractError("Expected integer for input: " + key);
        };

        const auto axis     = getStr("axis", "x");
        const auto polarity = getStr("polarity", "bright");
        if (axis != "x" && axis != "y")
            throw visionrt::ContractError("Invalid projection axis");
        if (polarity != "bright" && polarity != "dark")
            throw visionrt::ContractError("Invalid polarity");

        const bool horizontal = (axis == "x");
        const bool bright     = (polarity == "bright");
        const auto threshold  = getInt("threshold", 128);
        if (threshold < 0 || threshold > 255)
            throw visionrt::ContractError("Threshold must be in [0, 255]");

        const auto minimumPixels = getInt("min_pixels", 1);
        const auto minimumExtent = getInt("min_extent", 1);
        const auto gap           = getInt("merge_gap", 0);
        const auto padding       = getInt("padding", 0);

        std::vector<std::vector<visionrt::Roi>> output;
        output.reserve(imgBatch->images().size());

        for (const auto &image : imgBatch->images())
        {
            const auto                grayscale = toGray(image);
            const int                 length    = horizontal ? image.cols() : image.rows();
            std::vector<std::int64_t> counts(length, 0);

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
            const auto                 append = [&](int start, int end)
            {
                if (end - start < minimumExtent)
                    return;
                const int left  = static_cast<int>(start - std::min<std::int64_t>(padding, start));
                const int right = static_cast<int>(end + std::min<std::int64_t>(padding, length - end));
                rois.push_back(
                    horizontal ? visionrt::Roi::rectangle(static_cast<float>(left), 0, static_cast<float>(right - left),
                                                          static_cast<float>(image.rows()), 0)
                               : visionrt::Roi::rectangle(0, static_cast<float>(left), static_cast<float>(image.cols()),
                                                          static_cast<float>(right - left), 0));
            };

            int start = -1;
            int end   = -1;
            for (int index = 0; index < length; ++index)
            {
                if (counts[index] < minimumPixels)
                    continue;
                if (start < 0)
                    start = index;
                else if (index - end > gap)
                {
                    append(start, end);
                    start = index;
                }
                end = index + 1;
            }
            if (start >= 0)
                append(start, end);
            output.push_back(std::move(rois));
        }

        visionrt::Values outputs;
        outputs.emplace("images", *imgBatch);
        outputs.emplace("rois", visionrt::RoiBatch(std::move(output)));
        return {{}, std::move(outputs)};
    }
    catch (const std::exception &e)
    {
        visionrt::Values outputs;
        outputs.emplace("images", visionrt::Missing{});
        outputs.emplace("rois", visionrt::Missing{});
        return {
            {1, e.what()},
            std::move(outputs)
        };
    }
}

visionrt::ToolStatus RoiProjectionTool::release()
{
    return {};
}

} // namespace visiontools::preprocess
