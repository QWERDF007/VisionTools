#include "ImageCropTool.h"

#include "common/InputValues.h"
#include "common/UpstreamStatus.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace visiontools::preprocess {
namespace {

namespace p = ports::image_crop;

bool insidePolygon(std::span<const cv::Point2d> vertices, double x, double y)
{
    bool inside = false;
    for (std::size_t i = 0, previous = vertices.size() - 1; i < vertices.size(); previous = i++)
    {
        const auto  &a         = vertices[previous];
        const auto  &b         = vertices[i];
        const double cross     = (x - a.x) * (b.y - a.y) - (y - a.y) * (b.x - a.x);
        const double tolerance = 32 * std::numeric_limits<double>::epsilon()
                               * std::max(1.0, std::abs((x - a.x) * (b.y - a.y)) + std::abs((y - a.y) * (b.x - a.x)));
        if (std::abs(cross) <= tolerance && x >= std::min(a.x, b.x) && x <= std::max(a.x, b.x)
            && y >= std::min(a.y, b.y) && y <= std::max(a.y, b.y))
            return true;
        if ((a.y > y) != (b.y > y) && x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x)
            inside = !inside;
    }
    return inside;
}

struct MaskEntry
{
    cv::Mat mask;
    bool    any = false;
};

} // namespace

#ifdef VISIONTOOLS_TESTING
namespace testing {
std::size_t maskBuildCount = 0;
} // namespace testing
#endif

ImageCropTool::ImageCropTool(const visionrt::ToolServices &services)
    : services_(services)
{
}

visionrt::ToolStatus ImageCropTool::initialize(const visionrt::Values &parameters)
{
    if (!parameters.empty())
        return {1, "图像裁剪不接受初始化参数"};
    return {};
}

namespace {
visionrt::Values missingOutputs()
{
    return {{p::out::IMAGES, visionrt::Missing{}}, {p::out::ORIGINS, visionrt::Missing{}}};
}
} // namespace

visionrt::ToolResult ImageCropTool::execute(const visionrt::Values &inputs)
{
    // An unselected upstream branch is not an error: no algorithm runs and nothing is written.
    if (const auto upstream = upstreamStatus(inputs); upstream != 0)
        return skippedByUpstream(upstream, missingOutputs());
    try
    {
        const auto &images
            = requireTypedInput<visionrt::DataRef<visionrt::Image>>(inputs, p::in::IMAGES, {visionrt::Type::Image});
        const auto &rectangleValue
            = requireTypedInput<visionrt::DataRef<visionrt::ROI>>(inputs, p::in::RECT, {visionrt::Type::ROI});
        const auto &groups = rectangleValue.get();
        if (groups.size() != 1 || groups[0].size() != 1 || groups[0][0][0] != 2)
            throw visionrt::ContractError("输入端口 '" + std::string(p::in::RECT) + "' 必须包含一组一个矩形 ROI");
        const auto &rectangle = groups[0][0];

        std::array<cv::Point2d, 4> corners;
        try
        {
            corners = visionrt::rectangleCorners(rectangle);
        }
        catch (const std::exception &error)
        {
            throw visionrt::ContractError("输入端口 '" + std::string(p::in::RECT) + "' 的矩形无法解析："
                                          + error.what());
        }
        double minimumX = corners[0].x, maximumX = corners[0].x;
        double minimumY = corners[0].y, maximumY = corners[0].y;
        for (const auto &corner : corners)
        {
            minimumX = std::min(minimumX, corner.x);
            minimumY = std::min(minimumY, corner.y);
            maximumX = std::max(maximumX, corner.x);
            maximumY = std::max(maximumY, corner.y);
        }
        const auto snap = [](double value)
        {
            const double nearest = std::round(value);
            return std::abs(value - nearest)
                        <= 32 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(value))
                     ? nearest : value;
        };
        const bool   axisAligned = std::remainder(static_cast<double>(rectangle[5]), 360.0) == 0.0;
        const double epsilon = 32 * std::numeric_limits<double>::epsilon()
                             * std::max({1.0, std::abs(minimumX), std::abs(minimumY), std::abs(maximumX), std::abs(maximumY)});
        visionrt::Image                   output;
        visionrt::NestedVector<double, 2> origins;
        std::vector<visionrt::Owner>      outputOwners;
        output.reserve(images.get().size());
        origins.reserve(images.get().size());
        std::map<std::array<int, 4>, MaskEntry> masks;

        for (std::size_t index = 0; index < images.get().size(); ++index)
        {
            const auto       &image   = images.get()[index];
            const std::string context = "图像索引 " + std::to_string(index) + "：";
            const int         left
                = static_cast<int>(std::clamp(std::floor(snap(minimumX)), 0.0, static_cast<double>(image.cols)));
            const int top
                = static_cast<int>(std::clamp(std::floor(snap(minimumY)), 0.0, static_cast<double>(image.rows)));
            const int right
                = static_cast<int>(std::clamp(std::ceil(snap(maximumX)), 0.0, static_cast<double>(image.cols)));
            const int bottom
                = static_cast<int>(std::clamp(std::ceil(snap(maximumY)), 0.0, static_cast<double>(image.rows)));
            if (left >= right || top >= bottom)
                throw visionrt::ContractError(context + "裁剪矩形超出图像范围");
            const cv::Rect bounds(left, top, right - left, bottom - top);
            const bool coversEveryPixelCenter = axisAligned && left + 0.5 >= minimumX - epsilon
                                             && right - 0.5 <= maximumX + epsilon && top + 0.5 >= minimumY - epsilon
                                             && bottom - 0.5 <= maximumY + epsilon;
            if (coversEveryPixelCenter)
            {
                output.push_back(image(bounds));
                if (outputOwners.empty())
                    outputOwners.push_back(images.owner());
            }
            else
            {
                const std::array<int, 4> key{left, top, right - left, bottom - top};
                auto entry = masks.find(key);
                if (entry == masks.end())
                {
                    MaskEntry built;
                    built.mask = cv::Mat(bounds.height, bounds.width, CV_8UC1, cv::Scalar(0));
                    for (int rowIndex = 0; rowIndex < bounds.height; ++rowIndex)
                    {
                        auto *row = built.mask.ptr<unsigned char>(rowIndex);
                        for (int column = 0; column < bounds.width; ++column)
                        {
                            if (insidePolygon(corners, left + column + 0.5, top + rowIndex + 0.5))
                            {
                                row[column] = 255;
                                built.any = true;
                            }
                        }
                    }
#ifdef VISIONTOOLS_TESTING
                    ++testing::maskBuildCount;
#endif
                    entry = masks.emplace(key, std::move(built)).first;
                }
                if (!entry->second.any)
                    throw visionrt::ContractError(context + "裁剪区域没有有效像素中心");
                cv::Mat cropped;
                image(bounds).copyTo(cropped, entry->second.mask);
                output.push_back(std::move(cropped));
            }
            origins.push_back({static_cast<double>(left), static_cast<double>(top)});
        }
        visionrt::Values results;
        results.emplace(p::out::IMAGES, visionrt::DataRef<visionrt::Image>(std::move(output), std::move(outputOwners)));
        results.emplace(p::out::ORIGINS, visionrt::Array(std::move(origins)));
        return visionrt::toolResult(std::move(results), visionrt::executionSuccess());
    }
    catch (const visionrt::ContractError &error)
    {
        return visionrt::toolResult(missingOutputs(), {1, error.what()});
    }
    catch (const std::exception &error)
    {
        return visionrt::toolResult(missingOutputs(), {1, "图像裁剪失败", error.what()});
    }
    catch (...)
    {
        return visionrt::toolResult(missingOutputs(), {-1, "图像裁剪失败：未知异常"});
    }
}

visionrt::ToolStatus ImageCropTool::release()
{
    return {};
}

} // namespace visiontools::preprocess
