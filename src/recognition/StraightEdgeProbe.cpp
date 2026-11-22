#include "StraightEdgeProbe.h"

#include "StraightEdgeLocator.h"

#include "common/EdgeProfile.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

namespace visiontools::recognition {
namespace {

constexpr double kPi = 3.14159265358979323846;

// A picked band is a proposal, not a measurement: it only has to sit on the edge the user pointed
// at, and the search range is a usable default the user can drag afterwards.
constexpr double kBandHeight     = 48.0;
constexpr double kMinCoherence   = 0.2;
constexpr double kMinCompleteness = 0.2;

double degrees(double radians)
{
    return std::remainder(radians * 180.0 / kPi, 360.0);
}

int clampIndex(double value, int low, int high)
{
    return std::clamp(static_cast<int>(std::lround(value)), low, high);
}

// Mean squared gradient over the window: the orientation of an edge is the orientation of the
// gradient rotated by a right angle, which is what the tensor's dominant eigenvector gives.
struct Orientation
{
    bool   valid = false;
    double radians = 0;
    double coherence = 0;
    std::string quality;
};

Orientation estimateOrientation(const cv::Mat &gray, const cv::Rect &window)
{
    Orientation result;
    cv::Mat     gx;
    cv::Mat     gy;
    cv::Sobel(gray(window), gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray(window), gy, CV_32F, 0, 1, 3);

    double jxx = 0;
    double jxy = 0;
    double jyy = 0;
    for (int row = 0; row < gx.rows; ++row)
    {
        const auto *px = gx.ptr<float>(row);
        const auto *py = gy.ptr<float>(row);
        for (int column = 0; column < gx.cols; ++column)
        {
            jxx += static_cast<double>(px[column]) * px[column];
            jxy += static_cast<double>(px[column]) * py[column];
            jyy += static_cast<double>(py[column]) * py[column];
        }
    }
    const double trace = jxx + jyy;
    if (!(trace > 1e-6))
    {
        result.quality = "搜索中心附近没有明显边缘";
        return result;
    }
    result.coherence = std::sqrt((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy) / trace;
    if (result.coherence < kMinCoherence)
    {
        result.quality = "搜索中心附近边缘方向不明确";
        return result;
    }
    // Gradient direction of the dominant structure, then the edge runs across it.
    result.radians = 0.5 * std::atan2(2.0 * jxy, jxx - jyy) + kPi / 2.0;
    result.valid   = true;
    return result;
}

// Estimate the same normal profiles the runtime uses. The first pass measures response width;
// the second pass recomputes contrast at that width, across every supplied reference region.
void estimateParameters(const cv::Mat &gray, const std::vector<roi::Record> &regions,
                        const geometry::Alignment &alignment, EdgeProbeResult &result)
{
    std::vector<edge::IntensityProfile> profiles;
    std::vector<double>                 widths;
    constexpr int                       samplesPerRegion = 16;
    constexpr double                    gaussianFwhm     = 2.354820045;
    for (const auto &region : regions)
    {
        const auto        rectangle = visionrt::RegionView(region).rectangle();
        const double      angle     = rectangle.angle() * kPi / 180.0 + std::atan2(alignment.m[3], alignment.m[0]);
        const cv::Point2d tangent{std::cos(angle), std::sin(angle)};
        const cv::Point2d normal{-tangent.y, tangent.x};
        double            x = 0, y = 0;
        alignment.apply(rectangle.center().x, rectangle.center().y, x, y);
        for (int i = 0; i < samplesPerRegion; ++i)
        {
            const double offset
                = (static_cast<double>(i) / (samplesPerRegion - 1) - 0.5) * rectangle.width() * alignment.scale() * 0.8;
            auto                profile = edge::buildProfile(gray, cv::Point2d{x, y} + tangent * offset, normal,
                                                             rectangle.height() * alignment.scale() / 2.0, tangent, 0);
            edge::ProfileConfig config;
            config.sigma       = 0.5;
            config.minGradient = 1;
            if (const auto response = edge::nearestEdge(edge::detectEdges(profile, config)))
            {
                const auto gradients = edge::gradientProfile(profile, config.sigma);
                const auto peak      = static_cast<size_t>(
                    std::clamp(std::lround((response->offset + profile.halfLength) / profile.step), 0L,
                                    static_cast<long>(gradients.size() - 1)));
                size_t       left = peak, right = peak;
                const double half = std::abs(gradients[peak]) * 0.5;
                while (left > 0 && std::abs(gradients[left - 1]) >= half) --left;
                while (right + 1 < gradients.size() && std::abs(gradients[right + 1]) >= half) ++right;
                widths.push_back((right - left + 1) * profile.step * 2.0 / gaussianFwhm);
                profiles.push_back(std::move(profile));
            }
        }
    }
    if (widths.empty())
        throw visionrt::ContractError("搜索区域没有可用的法向响应");
    std::sort(widths.begin(), widths.end());
    result.width = std::clamp(static_cast<int>(std::lround(widths[widths.size() / 2])), 1, 63);
    std::vector<double> contrasts;
    bool                positive = false, negative = false;
    for (const auto &profile : profiles)
    {
        edge::ProfileConfig config;
        config.sigma       = std::max(0.5, result.width / 2.0);
        config.minGradient = 1;
        if (const auto response = edge::nearestEdge(edge::detectEdges(profile, config)))
        {
            contrasts.push_back(std::abs(response->gradient));
            positive = positive || response->gradient > 0;
            negative = negative || response->gradient < 0;
        }
    }
    if (contrasts.empty())
        throw visionrt::ContractError("平滑后的搜索区域没有有效边缘");
    std::sort(contrasts.begin(), contrasts.end());
    result.contrast = std::max(1.0, contrasts[(contrasts.size() - 1) / 10] * 0.5);
    result.polarity = positive && negative ? "any" : positive ? "dark_to_light" : "light_to_dark";
}

} // namespace

EdgeProbeResult probeStraightEdge(const cv::Mat &gray, const EdgeProbeRequest &request)
{
    EdgeProbeResult result;
    if (gray.empty())
    {
        result.quality = "输入图像为空";
        return result;
    }
    const int      half   = std::clamp(request.searchRadius, 16, 512);
    const int      left   = clampIndex(request.searchCenter.x - half, 0, std::max(0, gray.cols - 1));
    const int      top    = clampIndex(request.searchCenter.y - half, 0, std::max(0, gray.rows - 1));
    const int      right  = clampIndex(request.searchCenter.x + half, 0, std::max(0, gray.cols - 1));
    const int      bottom = clampIndex(request.searchCenter.y + half, 0, std::max(0, gray.rows - 1));
    const cv::Rect window{left, top, right - left + 1, bottom - top + 1};
    if (window.width < 16 || window.height < 16)
    {
        result.quality = "搜索中心附近没有足够的图像区域";
        return result;
    }

    const auto orientation = estimateOrientation(gray, window);
    if (!orientation.valid)
    {
        result.quality = orientation.quality;
        return result;
    }

    const cv::Point2d direction{std::cos(orientation.radians), std::sin(orientation.radians)};
    ReferenceSegment  probe;
    probe.name            = "picked";
    probe.from            = request.searchCenter - direction * half;
    probe.to              = request.searchCenter + direction * half;
    probe.search.center   = request.searchCenter;
    probe.search.width    = 2.0 * half;
    probe.search.height   = static_cast<double>(half);
    probe.search.angle    = degrees(orientation.radians);

    EdgeSearchOptions options;
    options.contrast        = 2.0;
    options.width           = 3;
    options.polarity        = edge::EdgePolarity::Any;
    options.minCompleteness = kMinCompleteness;

    std::optional<LocatedSegment> found;
    try
    {
        found = locateSegment(gray, probe, options);
    }
    catch (const std::exception &error)
    {
        result.quality = std::string("边缘拟合失败：") + error.what();
        return result;
    }
    if (!found)
    {
        result.quality = "搜索中心附近没有可用的直边";
        return result;
    }

    const cv::Point2d middle = (found->start + found->end) * 0.5;
    const double      length = cv::norm(found->end - found->start);
    const double      angle  = std::atan2(found->end.y - found->start.y, found->end.x - found->start.x);
    if (!(length >= 8.0))
    {
        result.quality = "找到的边太短，无法作为参考边";
        return result;
    }

    // The proposal is expressed in reference coordinates; the preview is the same band mapped into
    // the picture the user is looking at. A training reference has no alignment, so they coincide.
    cv::Point2d center = middle;
    double      degreesOut = degrees(angle);
    if (const auto inverse = request.alignment.inverse())
    {
        double x = 0;
        double y = 0;
        inverse->apply(middle.x, middle.y, x, y);
        center     = {x, y};
        degreesOut = degrees(angle - std::atan2(request.alignment.m[3], request.alignment.m[0]));
    }

    result.proposal.push_back(roi::rectangle(center.x, center.y, length, kBandHeight, degreesOut));
    result.preview = {roi::rectangle(middle.x, middle.y, length, kBandHeight, degrees(angle))};
    auto parameterRegions = request.parameterRegions;
    parameterRegions.insert(parameterRegions.end(), result.proposal.begin(), result.proposal.end());
    estimateParameters(gray, parameterRegions, request.alignment, result);
    result.quality  = "已沿边缘方向放置搜索框";
    result.valid    = true;
    return result;
}

} // namespace visiontools::recognition
