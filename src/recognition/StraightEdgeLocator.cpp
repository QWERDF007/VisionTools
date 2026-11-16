#include "StraightEdgeLocator.h"

#include "common/errors/ToolError.h"

#include "errors/RecognitionErrors.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace visiontools::recognition {
namespace {
constexpr double kPi = 3.14159265358979323846;

cv::Point2d normalized(cv::Point2d value)
{
    const double length = std::hypot(value.x, value.y);
    return length > 1e-12 ? cv::Point2d{value.x / length, value.y / length} : cv::Point2d{0, 0};
}

double cross(cv::Point2d left, cv::Point2d right)
{
    return left.x * right.y - left.y * right.x;
}

// Intersection of two infinite lines through `origin` along `direction`.
std::optional<cv::Point2d> intersect(cv::Point2d firstOrigin, cv::Point2d firstDirection,
                                     cv::Point2d secondOrigin, cv::Point2d secondDirection)
{
    const double denominator = cross(firstDirection, secondDirection);
    if (std::abs(denominator) < 1e-6)
        return std::nullopt;
    const cv::Point2d delta = secondOrigin - firstOrigin;
    const double      along = cross(delta, secondDirection) / denominator;
    return firstOrigin + firstDirection * along;
}

// Signed angle that maps `from` onto `to`, in VisionRT's x-right / y-down clockwise convention.
double angleBetween(cv::Point2d from, cv::Point2d to)
{
    return std::atan2(cross(from, to), from.dot(to));
}

double wrapAngle(double angle)
{
    while (angle > kPi)
        angle -= 2.0 * kPi;
    while (angle < -kPi)
        angle += 2.0 * kPi;
    return angle;
}

cv::Point2d projectOnto(cv::Point2d point, cv::Point2d origin, cv::Point2d direction)
{
    return origin + direction * (point - origin).dot(direction);
}
} // namespace

std::optional<LocatedSegment> locateSegment(const cv::Mat &gray, const ReferenceSegment &reference,
                                            const EdgeSearchOptions &options)
{
    if (gray.empty() || gray.type() != CV_8UC1)
        fail(RecognitionError::UnsupportedColorImage, "直边定位的输入必须是单通道 8 位灰度图");
    const cv::Point2d direction = normalized(reference.to - reference.from);
    const double      length    = cv::norm(reference.to - reference.from);
    if (!(length > 1.0))
        fail(RecognitionError::AssetFormatInvalid, reference.name, "参考边长度不足 1 像素");

    const cv::Point2d normal{-direction.y, direction.x};
    // The trained band may sit off the nominal segment: its height is the search range across the
    // edge, its width bounds how much of the segment is sampled along the edge.
    const cv::Point2d toBandCenter = reference.search.center - reference.from;
    const double      bandOffset   = toBandCenter.dot(normal);
    const double      searchHalf   = std::max(0.5, reference.search.height / 2.0);
    const double      averageHalf  = std::max(0.0, (static_cast<double>(std::max(1, options.width)) - 1.0) / 2.0);
    const double      sigma        = std::max(0.5, static_cast<double>(std::max(1, options.width)) / 2.0);
    const double      bandCenter   = toBandCenter.dot(direction);
    const double      bandStart    = std::clamp(bandCenter - reference.search.width / 2.0, 0.0, length);
    const double      bandEnd      = std::clamp(bandCenter + reference.search.width / 2.0, 0.0, length);
    if (!(bandEnd - bandStart >= 1.0))
        fail(RecognitionError::AssetFormatInvalid, reference.name, "搜索框没有覆盖参考边的有效长度");

    const int samples = std::clamp(static_cast<int>(std::lround(bandEnd - bandStart)), 8, 4096);
    edge::ProfileConfig config;
    config.searchHalfWidth = searchHalf;
    config.averageWidth    = averageHalf;
    config.sigma           = sigma;
    config.minGradient     = std::max(0.0, options.contrast);
    config.polarity        = options.polarity;

    std::vector<cv::Point2d> points;
    points.reserve(static_cast<std::size_t>(samples));
    for (int index = 0; index < samples; ++index)
    {
        const double u = bandStart + (bandEnd - bandStart) * static_cast<double>(index)
                                          / static_cast<double>(samples - 1);
        const cv::Point2d center = reference.from + direction * u + normal * bandOffset;
        const auto        profile = edge::buildProfile(gray, center, normal, searchHalf, direction, averageHalf);
        if (const auto candidate = edge::nearestEdge(edge::detectEdges(profile, config)))
            points.push_back(center + normal * candidate->offset);
    }
    if (points.size() < 3)
        return std::nullopt;

    cv::Vec4f line;
    cv::fitLine(points, line, cv::DIST_HUBER, 0, 0.01, 0.01);
    // fitLine does not fix the direction sign. The trained direction is the reference, so the
    // reported segment always runs from `from` towards `to` and the pose never comes out reversed.
    cv::Point2d lineDirection{line[0], line[1]};
    if (lineDirection.dot(direction) < 0.0)
        lineDirection = -lineDirection;
    const cv::Point2d lineOrigin{line[2], line[3]};
    std::size_t inliers = 0;
    for (const auto &point : points)
        if (std::abs(cross(point - lineOrigin, lineDirection)) <= 1.5)
            ++inliers;
    const double completeness = static_cast<double>(inliers) / static_cast<double>(samples);
    if (completeness < options.minCompleteness)
        return std::nullopt;

    // Endpoints stay on the fitted line over the sampled extent, so the transform uses the measured
    // pose rather than a noisy outermost sample.
    return LocatedSegment{projectOnto(reference.from + direction * bandStart, lineOrigin, lineDirection),
                          projectOnto(reference.from + direction * bandEnd, lineOrigin, lineDirection), completeness};
}

std::optional<geometry::Alignment> rigidTransform(const ReferenceSegment &firstReference,
                                                  const LocatedSegment  &firstFound,
                                                  const ReferenceSegment &secondReference,
                                                  const LocatedSegment  &secondFound)
{
    const cv::Point2d firstReferenceDirection  = normalized(firstReference.to - firstReference.from);
    const cv::Point2d secondReferenceDirection = normalized(secondReference.to - secondReference.from);
    const cv::Point2d firstFoundDirection      = normalized(firstFound.end - firstFound.start);
    const cv::Point2d secondFoundDirection     = normalized(secondFound.end - secondFound.start);

    const auto referenceIntersection = intersect(firstReference.from, firstReferenceDirection,
                                                 secondReference.from, secondReferenceDirection);
    const auto foundIntersection = intersect(firstFound.start, firstFoundDirection,
                                             secondFound.start, secondFoundDirection);
    if (!referenceIntersection || !foundIntersection)
        return std::nullopt;

    double firstAngle  = angleBetween(firstReferenceDirection, firstFoundDirection);
    double secondAngle = angleBetween(secondReferenceDirection, secondFoundDirection);
    if (std::abs(firstAngle - secondAngle) > kPi)
    {
        // Two estimates on opposite sides of the branch cut still describe the same rotation.
        if (firstAngle > 0.0)
            firstAngle -= 2.0 * kPi;
        else
            secondAngle -= 2.0 * kPi;
    }
    const double angle = wrapAngle((firstAngle + secondAngle) / 2.0);
    const double cos   = std::cos(angle);
    const double sin   = std::sin(angle);

    const cv::Point2d rotatedReference{cos * referenceIntersection->x - sin * referenceIntersection->y,
                                       sin * referenceIntersection->x + cos * referenceIntersection->y};
    const cv::Point2d translation = *foundIntersection - rotatedReference;

    geometry::Alignment alignment;
    alignment.m = {cos, -sin, translation.x, sin, cos, translation.y, 0, 0, 1};
    if (!alignment.isSimilarity() || !alignment.invertible())
        return std::nullopt;
    return alignment;
}

} // namespace visiontools::recognition
