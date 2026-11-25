#include "StraightEdgeLocator.h"

#include "common/errors/ToolError.h"
#include "errors/RecognitionErrors.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace visiontools::recognition {
EdgeSearchOptions EdgeSearchOptions::fromParameters(double contrast, std::int64_t width, const std::string &polarity, double completeness)
{
    if (!(contrast >= 0.0 && contrast <= 1024.0)) fail(CommonToolError::InputOutOfRange, "边缘对比度", "必须在 [0, 1024] 内");
    if (width < 1 || width > 63) fail(CommonToolError::InputOutOfRange, "边缘宽度", "必须在 [1, 63] 内");
    if (!(completeness >= 0.0 && completeness <= 1.0)) fail(CommonToolError::InputOutOfRange, "最低边完整度", "必须在 [0, 1] 内");
    edge::EdgePolarity direction;
    if (polarity == "any") direction = edge::EdgePolarity::Any;
    else if (polarity == "dark_to_light") direction = edge::EdgePolarity::DarkToLight;
    else if (polarity == "light_to_dark") direction = edge::EdgePolarity::LightToDark;
    else fail(CommonToolError::InputOutOfRange, "跃迁方向", "只能是 any、dark_to_light 或 light_to_dark");
    return {contrast, static_cast<int>(width), direction, completeness};
}

namespace {
constexpr double kPi = 3.14159265358979323846;

struct SearchSampling
{
    cv::Point2d normal;
    double halfWidth;
};

SearchSampling searchSampling(const SearchRectangle &band)
{
    const double radians = band.angle * kPi / 180.0;
    return {{-std::sin(radians), std::cos(radians)}, std::max(0.5, band.height / 2.0)};
}

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
std::optional<cv::Point2d> intersect(cv::Point2d firstOrigin, cv::Point2d firstDirection, cv::Point2d secondOrigin,
                                     cv::Point2d secondDirection)
{
    const double denominator = cross(firstDirection, secondDirection);
    if (std::abs(denominator) < kMinEdgeAngleSine)
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
    while (angle > kPi) angle -= 2.0 * kPi;
    while (angle < -kPi) angle += 2.0 * kPi;
    return angle;
}

cv::Point2d projectOnto(cv::Point2d point, cv::Point2d origin, cv::Point2d direction)
{
    return origin + direction * (point - origin).dot(direction);
}
} // namespace

SearchRectangle searchRectangle(const visionrt::EncodedRegion &record, std::size_t index)
{
    const auto view = visionrt::RegionView(record);
    if (view.kind() != visionrt::RoiKind::Rectangle)
        fail(RecognitionError::StraightEdgeRegionsInvalid, "第 " + std::to_string(index + 1) + " 个搜索区域",
             "必须是旋转矩形");
    const auto rectangle = view.rectangle();
    if (!(rectangle.width() > 1.0) || !(rectangle.height() > 0.0))
        fail(RecognitionError::StraightEdgeRegionsInvalid, "搜索矩形", "宽度必须大于 1，高度必须大于 0");
    return {rectangle.center(), std::max(rectangle.width(), rectangle.height()),
            std::min(rectangle.width(), rectangle.height()),
            rectangle.angle() + (rectangle.height() > rectangle.width() ? 90.0 : 0.0)};
}

std::vector<double> searchDirectionRecord(const SearchRectangle &band)
{
    const auto sampling = searchSampling(band);
    const double angle = std::atan2(sampling.normal.y, sampling.normal.x) * 180.0 / kPi;
    return visionrt::directionRecord(band.center - sampling.normal * sampling.halfWidth, 2 * sampling.halfWidth, angle);
}

std::optional<LocatedSegment> locateSegment(const cv::Mat &gray, const SearchRectangle &band,
                                            const EdgeSearchOptions &options)
{
    if (gray.empty() || gray.type() != CV_8UC1)
        fail(RecognitionError::UnsupportedColorImage, "直边定位的输入必须是单通道 8 位灰度图");
    const double      radians = band.angle * kPi / 180.0;
    const cv::Point2d direction{std::cos(radians), std::sin(radians)};
    const double      length = band.width;
    const auto        from   = band.center - direction * (length / 2.0);
    if (!(length > 1.0))
        fail(RecognitionError::StraightEdgeRegionsInvalid, "搜索矩形", "宽度不足 1 像素");

    const auto sampling = searchSampling(band);
    const cv::Point2d normal = sampling.normal;
    const double searchHalf = sampling.halfWidth;
    const double        averageHalf = std::max(0.0, (static_cast<double>(options.width) - 1.0) / 2.0);
    const double        sigma       = std::max(0.5, static_cast<double>(options.width) / 2.0);
    const int           samples     = std::clamp(static_cast<int>(std::lround(length)), 8, 4096);
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
        const double      u       = length * static_cast<double>(index) / static_cast<double>(samples - 1);
        const cv::Point2d center  = from + direction * u;
        const auto        profile = edge::buildProfile(gray, center, normal, searchHalf, direction, averageHalf);
        if (const auto candidate = edge::nearestEdge(edge::detectEdges(profile, config)))
            points.push_back(center + normal * candidate->offset);
    }
    if (points.size() < 3)
        return std::nullopt;

    cv::Vec4f line;
    cv::fitLine(points, line, cv::DIST_HUBER, 0, 0.01, 0.01);
    // fitLine does not fix the direction sign. The search band fixes the direction, so the
    // reported segment always runs from `from` towards `to` and the pose never comes out reversed.
    cv::Point2d lineDirection{line[0], line[1]};
    if (lineDirection.dot(direction) < 0.0)
        lineDirection = -lineDirection;
    const cv::Point2d lineOrigin{line[2], line[3]};
    std::size_t       inliers = 0;
    for (const auto &point : points)
        if (std::abs(cross(point - lineOrigin, lineDirection)) <= 1.5)
            ++inliers;
    const double completeness = static_cast<double>(inliers) / static_cast<double>(samples);
    if (completeness < options.minCompleteness)
        return std::nullopt;

    // Endpoints stay on the fitted line over the sampled extent, so the transform uses the measured
    // pose rather than a noisy outermost sample.
    return LocatedSegment{projectOnto(from, lineOrigin, lineDirection),
                          projectOnto(from + direction * length, lineOrigin, lineDirection), completeness};
}

std::optional<geometry::Alignment> rigidTransform(const ReferenceSegment &firstReference,
                                                  const LocatedSegment   &firstFound,
                                                  const ReferenceSegment &secondReference,
                                                  const LocatedSegment   &secondFound)
{
    const cv::Point2d firstReferenceDirection  = normalized(firstReference.to - firstReference.from);
    const cv::Point2d secondReferenceDirection = normalized(secondReference.to - secondReference.from);
    const cv::Point2d firstFoundDirection      = normalized(firstFound.end - firstFound.start);
    const cv::Point2d secondFoundDirection     = normalized(secondFound.end - secondFound.start);

    const auto referenceIntersection
        = intersect(firstReference.from, firstReferenceDirection, secondReference.from, secondReferenceDirection);
    const auto foundIntersection
        = intersect(firstFound.start, firstFoundDirection, secondFound.start, secondFoundDirection);
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
    alignment.m = {cos, -sin, translation.x, sin, cos, translation.y};
    if (!alignment.isSimilarity() || !alignment.invertible())
        return std::nullopt;
    return alignment;
}

} // namespace visiontools::recognition
