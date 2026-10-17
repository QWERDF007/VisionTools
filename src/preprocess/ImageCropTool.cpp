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

bool insidePolygon(std::span<const cv::Point2d> vertices, double x, double y) {
    bool inside = false;
    for (std::size_t i = 0, previous = vertices.size() - 1; i < vertices.size(); previous = i++) {
        const auto& a = vertices[previous];
        const auto& b = vertices[i];
        const double cross = (x - a.x) * (b.y - a.y) - (y - a.y) * (b.x - a.x);
        const double tolerance = 32 * std::numeric_limits<double>::epsilon() *
            std::max(1.0, std::abs((x - a.x) * (b.y - a.y)) + std::abs((y - a.y) * (b.x - a.x)));
        if (std::abs(cross) <= tolerance && x >= std::min(a.x, b.x) &&
            x <= std::max(a.x, b.x) && y >= std::min(a.y, b.y) && y <= std::max(a.y, b.y))
            return true;
        if ((a.y > y) != (b.y > y) &&
            x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x)
            inside = !inside;
    }
    return inside;
}

template<class T>
visionrt::NumericBuffer makeBuffer(visionrt::DType dtype, std::vector<std::int64_t> shape, const std::vector<T>& values) {
    return visionrt::NumericBuffer::copy(dtype, std::move(shape), std::as_bytes(std::span(values)));
}

} // namespace

ImageCropTool::ImageCropTool(const visionrt::ToolServices& services)
    : services_(services) {}

visionrt::ToolStatus ImageCropTool::initialize(const visionrt::Values& parameters) {
    if (!parameters.empty()) {
        return {1, "ImageCropTool accepts no init parameters"};
    }
    return {};
}

visionrt::ToolResult ImageCropTool::execute(const visionrt::Values& inputs) {
    try {
        auto imgIt = inputs.find(p::in::IMAGES);
        auto roiIt = inputs.find(p::in::ROIS);
        if (imgIt == inputs.end() || visionrt::isMissing(imgIt->second))
            throw visionrt::ContractError("Missing input: images");
        if (roiIt == inputs.end() || visionrt::isMissing(roiIt->second))
            throw visionrt::ContractError("Missing input: rois");

        const auto* images = std::get_if<visionrt::ImageBatch>(&imgIt->second);
        const auto* rois = std::get_if<visionrt::RoiBatch>(&roiIt->second);
        if (!images) throw visionrt::ContractError("Expected ImageBatch for input: images");
        if (!rois) throw visionrt::ContractError("Expected RoiBatch for input: rois");

        if (images->images().size() != rois->images().size()) {
            throw visionrt::ContractError("Crop image/ROI batch lengths differ");
        }

        std::vector<visionrt::OwnedImage> output;
        std::vector<std::int64_t> indices;
        std::vector<double> origins;

        for (std::size_t b = 0; b < images->images().size(); ++b) {
            const auto& image = images->images()[b];
            for (std::size_t n = 0; n < rois->images()[b].size(); ++n) {
                const auto& roi = rois->images()[b][n];
                const std::string context = "Crop b=" + std::to_string(b) + " n=" + std::to_string(n) + ": ";
                if (roi.kind() == visionrt::RoiKind::Segment || roi.kind() == visionrt::RoiKind::PointSet) {
                    throw visionrt::ContractError(context + "line and point-set ROI have no supported area");
                }
                const auto encoded = roi.encoded();
                std::array<cv::Point2d, 4> corners;
                std::vector<cv::Point2d> polygon;
                std::span<const cv::Point2d> vertices;
                double minimumX, minimumY, maximumX, maximumY;

                if (roi.kind() == visionrt::RoiKind::Circle) {
                    minimumX = static_cast<double>(encoded[1]) - encoded[3];
                    minimumY = static_cast<double>(encoded[2]) - encoded[3];
                    maximumX = static_cast<double>(encoded[1]) + encoded[3];
                    maximumY = static_cast<double>(encoded[2]) + encoded[3];
                } else {
                    if (roi.kind() == visionrt::RoiKind::Rectangle) {
                        corners = roi.rectangleCorners();
                        vertices = corners;
                    } else {
                        polygon.reserve((encoded.size() - 1) / 2);
                        for (std::size_t i = 1; i < encoded.size(); i += 2) {
                            polygon.emplace_back(encoded[i], encoded[i + 1]);
                        }
                        vertices = polygon;
                    }
                    minimumX = maximumX = vertices[0].x;
                    minimumY = maximumY = vertices[0].y;
                    for (const auto& vertex : vertices) {
                        minimumX = std::min(minimumX, vertex.x);
                        minimumY = std::min(minimumY, vertex.y);
                        maximumX = std::max(maximumX, vertex.x);
                        maximumY = std::max(maximumY, vertex.y);
                    }
                }

                const auto snap = [](double value) {
                    const double nearest = std::round(value);
                    return std::abs(value - nearest) <= 32 * std::numeric_limits<double>::epsilon() *
                        std::max(1.0, std::abs(value)) ? nearest : value;
                };

                const int left = static_cast<int>(std::clamp(std::floor(snap(minimumX)), 0.0,
                                                            static_cast<double>(image.cols())));
                const int top = static_cast<int>(std::clamp(std::floor(snap(minimumY)), 0.0,
                                                           static_cast<double>(image.rows())));
                const int right = static_cast<int>(std::clamp(std::ceil(snap(maximumX)), 0.0,
                                                             static_cast<double>(image.cols())));
                const int bottom = static_cast<int>(std::clamp(std::ceil(snap(maximumY)), 0.0,
                                                              static_cast<double>(image.rows())));

                if (left >= right || top >= bottom) {
                    throw visionrt::ContractError(context + "ROI is outside the image");
                }

                const cv::Rect bounds(left, top, right - left, bottom - top);

                // Zero-copy readonly view for axis-aligned rectangles:
                // Shares image.owner() and preserves rowStride() per TS-03 / VS-03 / F06
                const bool isAxisAlignedRect = (roi.kind() == visionrt::RoiKind::Rectangle &&
                    std::remainder(static_cast<double>(encoded[5]), 360.0) == 0.0);

                if (isAxisAlignedRect) {
                    output.push_back(image.region(bounds));
                    indices.push_back(static_cast<std::int64_t>(b));
                    indices.push_back(static_cast<std::int64_t>(n));
                    origins.push_back(left);
                    origins.push_back(top);
                    continue;
                }

                cv::Mat mask(bounds.height, bounds.width, CV_8UC1, cv::Scalar(0));
                bool any = false;

                for (int y = 0; y < bounds.height; ++y) {
                    auto* row = mask.ptr<unsigned char>(y);
                    for (int x = 0; x < bounds.width; ++x) {
                        const double centerX = left + x + 0.5;
                        const double centerY = top + y + 0.5;
                        bool covered;
                        if (roi.kind() == visionrt::RoiKind::Circle) {
                            const double dx = centerX - encoded[1];
                            const double dy = centerY - encoded[2];
                            covered = dx * dx + dy * dy <= static_cast<double>(encoded[3]) * encoded[3];
                        } else {
                            covered = insidePolygon(vertices, centerX, centerY);
                        }
                        if (covered) { row[x] = 255; any = true; }
                    }
                }

                if (!any) {
                    throw visionrt::ContractError(context + "ROI has no valid pixel centers");
                }

                cv::Mat cropped;
                image.readOnlyMat()(bounds).copyTo(cropped, mask);
                output.push_back(visionrt::OwnedImage::take(std::move(cropped)));
                indices.push_back(static_cast<std::int64_t>(b));
                indices.push_back(static_cast<std::int64_t>(n));
                origins.push_back(left);
                origins.push_back(top);
            }
        }

        const auto count = static_cast<std::int64_t>(output.size());
        visionrt::Values results;
        results.emplace(p::out::IMAGES, visionrt::ImageBatch(std::move(output)));
        results.emplace(p::out::SOURCE_INDICES, visionrt::NumericArray(visionrt::DType::Int64, 2, visionrt::Layout::Dense,
            {makeBuffer(visionrt::DType::Int64, {count, 2}, indices)}));
        results.emplace(p::out::ORIGINS, visionrt::NumericArray(visionrt::DType::Float64, 2, visionrt::Layout::Dense,
            {makeBuffer(visionrt::DType::Float64, {count, 2}, origins)}));

        return {{}, std::move(results)};
    } catch (const std::exception& e) {
        visionrt::Values results;
        results.emplace(p::out::IMAGES, visionrt::Missing{});
        results.emplace(p::out::SOURCE_INDICES, visionrt::Missing{});
        results.emplace(p::out::ORIGINS, visionrt::Missing{});
        return {{1, e.what()}, std::move(results)};
    }
}

visionrt::ToolStatus ImageCropTool::release() {
    return {};
}

} // namespace visiontools::preprocess

