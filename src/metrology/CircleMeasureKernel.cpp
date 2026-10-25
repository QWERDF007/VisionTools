#include "CircleMeasureKernel.h"

#include "CircleFit.h"

#include <algorithm>
#include <cmath>

namespace visiontools::metrology {
namespace {
constexpr double kPi = 3.14159265358979323846;
} // namespace

CircleMeasurement measureCircleKernel(const cv::Mat &gray, cv::Point2d center, double radius,
                                      const CircleOptions &options)
{
    CircleMeasurement result;
    ProfileConfig     config;
    config.searchHalfWidth = options.searchHalfWidth;
    config.averageWidth    = options.averageWidth;
    config.sigma           = options.sigma;
    config.minGradient     = options.minGradient;
    config.polarity        = options.polarity;

    for (int caliper = 0; caliper < options.calipers; ++caliper)
    {
        const double angleDegrees = 360.0 * static_cast<double>(caliper) / static_cast<double>(options.calipers);
        const double radians      = angleDegrees * kPi / 180.0;
        const auto   direction    = cv::Point2d{std::cos(radians), std::sin(radians)};
        const auto   nominal      = cv::Point2d{center.x + direction.x * radius, center.y + direction.y * radius};
        const auto   tangent      = cv::Point2d{-direction.y, direction.x};
        const auto   profile      = buildProfile(gray, nominal, direction, options.searchHalfWidth, tangent,
                                                 (options.averageWidth - 1.0) / 2.0, 0.5);
        auto         candidate    = nearestEdge(detectEdges(profile, config));
        if (!candidate)
            continue;
        result.edges.push_back({nominal.x + direction.x * candidate->offset,
                                nominal.y + direction.y * candidate->offset});
        result.angles.push_back(angleDegrees);
        result.edgeOffsets.push_back(candidate->offset);
        if (candidate->gradient > 0.0)
            ++result.positiveEdges;
        else
            ++result.negativeEdges;
    }
    result.completeness = static_cast<double>(result.edges.size()) / static_cast<double>(options.calipers);
    if (result.edges.size() < static_cast<std::size_t>(options.minPoints))
    {
        result.reason = "有效边缘点少于最少有效点";
        return result;
    }
    if (result.completeness < options.minCompleteness)
    {
        result.reason = "有效卡尺占比低于最小完整度";
        return result;
    }
    if (angularCoverage(result.angles) < options.minCoverageDeg)
    {
        result.reason = "有效边缘角覆盖不足";
        return result;
    }
    CircleFitOptions fitOptions;
    fitOptions.maxRms = options.maxRms;
    std::string reason;
    const auto  fit = fitCircle(result.edges, fitOptions, &reason);
    if (!fit)
    {
        result.reason = reason.empty() ? "圆拟合失败" : reason;
        return result;
    }
    if (fit->radius < options.radiusMin || fit->radius > options.radiusMax)
    {
        result.reason = "拟合半径超出允许范围";
        return result;
    }
    result.valid        = true;
    result.centerX      = fit->centerX;
    result.centerY      = fit->centerY;
    result.radius       = fit->radius;
    result.rms          = fit->rms;
    return result;
}
} // namespace visiontools::metrology
