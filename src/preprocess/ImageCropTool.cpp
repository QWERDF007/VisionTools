#include "ImageCropTool.h"

#include "PortConstants.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace visiontools::preprocess {
namespace {

namespace p = visiontools::ports::visionrt_image_crop;

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

} // namespace

ImageCropTool::ImageCropTool(const visionrt::ToolServices &services)
    : services_(services)
{
}

visionrt::ToolStatus ImageCropTool::initialize(const visionrt::Values &parameters)
{
    if (!parameters.empty())
        return {1, "ImageCropTool accepts no init parameters"};
    return {};
}

visionrt::ToolResult ImageCropTool::execute(const visionrt::Values &inputs)
{
    try
    {
        const auto imageInput = inputs.find(p::in::IMAGES);
        const auto rectangleInput = inputs.find(p::in::RECT);
        if (imageInput == inputs.end() || visionrt::isMissing(imageInput->second))
            throw visionrt::ContractError("Missing input: images");
        if (rectangleInput == inputs.end() || visionrt::isMissing(rectangleInput->second))
            throw visionrt::ContractError("Missing input: rect");
        const auto *images = std::get_if<visionrt::ImageBatch>(&imageInput->second);
        if (!images)
            throw visionrt::ContractError("Expected ImageBatch for input: images");
        visionrt::validateValue({visionrt::Type::Rect}, rectangleInput->second);
        const auto &rectangle = std::get<visionrt::Roi>(rectangleInput->second);
        const auto corners = rectangle.rectangleCorners();
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
        const bool axisAligned = std::remainder(static_cast<double>(rectangle.encoded()[5]), 360.0) == 0.0;
        const double epsilon = 32 * std::numeric_limits<double>::epsilon()
                             * std::max({1.0, std::abs(minimumX), std::abs(minimumY), std::abs(maximumX), std::abs(maximumY)});
        std::vector<visionrt::OwnedImage> output;
        std::vector<double> origins;
        output.reserve(images->images().size());
        origins.reserve(images->images().size() * 2);
        for (std::size_t index = 0; index < images->images().size(); ++index)
        {
            const auto &image = images->images()[index];
            const std::string context = "Crop image=" + std::to_string(index) + ": ";
            const int left = static_cast<int>(std::clamp(std::floor(snap(minimumX)), 0.0, static_cast<double>(image.cols())));
            const int top = static_cast<int>(std::clamp(std::floor(snap(minimumY)), 0.0, static_cast<double>(image.rows())));
            const int right = static_cast<int>(std::clamp(std::ceil(snap(maximumX)), 0.0, static_cast<double>(image.cols())));
            const int bottom = static_cast<int>(std::clamp(std::ceil(snap(maximumY)), 0.0, static_cast<double>(image.rows())));
            if (left >= right || top >= bottom)
                throw visionrt::ContractError(context + "Rect is outside the image");
            const cv::Rect bounds(left, top, right - left, bottom - top);
            const bool coversEveryPixelCenter = axisAligned && left + 0.5 >= minimumX - epsilon
                                             && right - 0.5 <= maximumX + epsilon && top + 0.5 >= minimumY - epsilon
                                             && bottom - 0.5 <= maximumY + epsilon;
            if (coversEveryPixelCenter)
                output.push_back(image.region(bounds));
            else
            {
                cv::Mat mask(bounds.height, bounds.width, CV_8UC1, cv::Scalar(0));
                bool any = false;
                for (int rowIndex = 0; rowIndex < bounds.height; ++rowIndex)
                {
                    auto *row = mask.ptr<unsigned char>(rowIndex);
                    for (int column = 0; column < bounds.width; ++column)
                    {
                        if (insidePolygon(corners, left + column + 0.5, top + rowIndex + 0.5))
                        {
                            row[column] = 255;
                            any = true;
                        }
                    }
                }
                if (!any)
                    throw visionrt::ContractError(context + "Rect has no valid pixel centers");
                cv::Mat cropped;
                image.readOnlyMat()(bounds).copyTo(cropped, mask);
                output.push_back(visionrt::OwnedImage::take(std::move(cropped)));
            }
            origins.push_back(left);
            origins.push_back(top);
        }
        const auto count = static_cast<std::int64_t>(output.size());
        visionrt::Values results;
        results.emplace(p::out::IMAGES, visionrt::ImageBatch(std::move(output)));
        results.emplace(p::out::ORIGINS,
                        visionrt::NumericArray(visionrt::DType::Float64, 2, visionrt::Layout::Dense,
                            {visionrt::NumericBuffer::copy(visionrt::DType::Float64, {count, 2},
                                                          std::as_bytes(std::span(origins)))}));
        return {{}, std::move(results)};
    }
    catch (const std::exception &error)
    {
        return {{1, error.what()}, {{p::out::IMAGES, visionrt::Missing{}}, {p::out::ORIGINS, visionrt::Missing{}}}};
    }
}

visionrt::ToolStatus ImageCropTool::release()
{
    return {};
}

} // namespace visiontools::preprocess
