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

    std::vector<bool> edgePositive;
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
        edgePositive.push_back(candidate->gradient > 0.0);
    }
    if (result.edges.size() < static_cast<std::size_t>(options.minPoints))
    {
        result.reason = "有效边缘点少于最少有效点";
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
    // Every quality number below is recomputed on the final inlier subset. Validating the
    // pre-rejection calipers would let a short arc plus scattered outliers pass the gate.
    std::vector<cv::Point2d> inlierEdges;
    std::vector<double>      inlierAngles, inlierOffsets;
    int                      positive = 0, negative = 0;
    inlierEdges.reserve(fit->inlierIndices.size());
    inlierAngles.reserve(fit->inlierIndices.size());
    inlierOffsets.reserve(fit->inlierIndices.size());
    for (const auto index : fit->inlierIndices)
    {
        if (index >= result.edges.size())
            continue;
        inlierEdges.push_back(result.edges[index]);
        inlierAngles.push_back(result.angles[index]);
        inlierOffsets.push_back(result.edgeOffsets[index]);
        if (index < edgePositive.size())
            (edgePositive[index] ? positive : negative) += 1;
    }
    if (inlierEdges.empty())
    {
        result.reason = "圆拟合最终内点为空";
        return result;
    }
    const std::size_t minPoints
        = std::max<std::size_t>(3, static_cast<std::size_t>(std::max(0, options.minPoints)));
    if (inlierEdges.size() < minPoints)
    {
        result.reason = "最终内点少于最少有效点";
        return result;
    }
    const double completeness = static_cast<double>(inlierEdges.size()) / static_cast<double>(options.calipers);
    if (completeness < options.minCompleteness)
    {
        result.reason = "最终内点完整度低于最小完整度";
        return result;
    }
    if (angularCoverage(inlierAngles) < options.minCoverageDeg)
    {
        result.reason = "最终内点角覆盖不足";
        return result;
    }
    if (fit->radius < options.radiusMin || fit->radius > options.radiusMax)
    {
        result.reason = "拟合半径超出允许范围";
        return result;
    }
    result.valid         = true;
    result.centerX       = fit->centerX;
    result.centerY       = fit->centerY;
    result.radius        = fit->radius;
    result.rms           = fit->rms;
    result.completeness  = completeness;
    result.positiveEdges = positive;
    result.negativeEdges = negative;
    // Visualization consumes the same final inliers that produced the geometry.
    result.edges       = std::move(inlierEdges);
    result.angles      = std::move(inlierAngles);
    result.edgeOffsets = std::move(inlierOffsets);
    return result;
}
} // namespace visiontools::metrology
