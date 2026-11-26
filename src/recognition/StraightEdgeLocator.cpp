#include "StraightEdgeLocator.h"

#include "common/errors/ToolError.h"
#include "errors/RecognitionErrors.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
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

std::optional<LocatedSegment> locateSegment(const cv::Mat &image, const SearchRectangle &band,
                                            const EdgeSearchOptions &options)
{
    if (image.empty() || image.depth() != CV_8U
        || (image.channels() != 1 && image.channels() != 3 && image.channels() != 4))
        fail(RecognitionError::UnsupportedColorImage, "直边定位的输入必须是 8 位 1/3/4 通道图像");
    const double      radians = band.angle * kPi / 180.0;
    const cv::Point2d direction{std::cos(radians), std::sin(radians)};
    const double      length = band.width;
    const auto        from   = band.center - direction * (length / 2.0);
    if (!(length > 1.0))
        fail(RecognitionError::StraightEdgeRegionsInvalid, "搜索矩形", "宽度不足 1 像素");

    const auto        sampling    = searchSampling(band);
    const cv::Point2d normal      = sampling.normal;
    const double      searchHalf  = sampling.halfWidth;
    const double      averageHalf = std::max(0.0, (static_cast<double>(options.width) - 1.0) / 2.0);
    const double      sigma       = std::max(0.5, static_cast<double>(options.width) / 2.0);
    const int         samples     = std::clamp(static_cast<int>(std::lround(length)), 8, 4096);
    constexpr double  kStep       = 0.5;
    const int         normalCount = static_cast<int>(std::floor(2.0 * searchHalf / kStep)) + 1;
    const int         tangential  = std::max(1, static_cast<int>(std::round(averageHalf * 2.0 + 1.0)));

    const auto dogDouble = edge::derivativeGaussianKernel(sigma, kStep);
    const int  kRadius   = static_cast<int>(dogDouble.size() / 2);
    if (normalCount <= kRadius * 2 + 2)
        return std::nullopt;

    const double halfL = length / 2.0 + averageHalf + 2.0;
    const double halfH = searchHalf + 2.0;
    const double spanX = std::abs(direction.x) * halfL + std::abs(normal.x) * halfH;
    const double spanY = std::abs(direction.y) * halfL + std::abs(normal.y) * halfH;
    const int    x0    = std::clamp(static_cast<int>(std::floor(band.center.x - spanX)), 0, image.cols);
    const int    y0    = std::clamp(static_cast<int>(std::floor(band.center.y - spanY)), 0, image.rows);
    const int    x1    = std::clamp(static_cast<int>(std::ceil(band.center.x + spanX)) + 1, 0, image.cols);
    const int    y1    = std::clamp(static_cast<int>(std::ceil(band.center.y + spanY)) + 1, 0, image.rows);
    if (x1 <= x0 || y1 <= y0)
        return std::nullopt;

    cv::Mat        grayCrop;
    const cv::Rect cropRect(x0, y0, x1 - x0, y1 - y0);
    if (image.channels() == 1)
        grayCrop = image(cropRect);
    else if (image.channels() == 3)
        cv::cvtColor(image(cropRect), grayCrop, cv::COLOR_BGR2GRAY);
    else
        cv::cvtColor(image(cropRect), grayCrop, cv::COLOR_BGRA2GRAY);

    const double uStep    = length / static_cast<double>(samples - 1);
    const int    padT     = tangential / 2;
    const double reqHalfL = length / 2.0 + std::max(averageHalf, static_cast<double>(padT) * uStep);
    const double reqSpanX = std::abs(direction.x) * reqHalfL + std::abs(normal.x) * searchHalf;
    const double reqSpanY = std::abs(direction.y) * reqHalfL + std::abs(normal.y) * searchHalf;
    const bool   fullyInside
        = (band.center.x - reqSpanX >= 0.0 && band.center.y - reqSpanY >= 0.0
           && band.center.x + reqSpanX <= image.cols - 1.0 && band.center.y + reqSpanY <= image.rows - 1.0);

    std::vector<cv::Point2d> points;
    points.reserve(static_cast<std::size_t>(samples));
    const double minGradient = std::max(0.0, options.contrast);
    const int    cStart      = kRadius + 1;
    const int    cEnd        = normalCount - kRadius - 1;

    if (fullyInside)
    {
        const int         totalRows = samples + 2 * padT;
        const cv::Matx23d invMap(normal.x * kStep, direction.x * uStep,
                                 (from.x - x0) - direction.x * padT * uStep - normal.x * searchHalf,
                                 normal.y * kStep, direction.y * uStep,
                                 (from.y - y0) - direction.y * padT * uStep - normal.y * searchHalf);
        cv::Mat warped;
        cv::warpAffine(grayCrop, warped, invMap, cv::Size(normalCount, totalRows),
                       cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);

        std::vector<float> dogFloat(dogDouble.size());
        for (std::size_t i = 0; i < dogDouble.size(); ++i)
            dogFloat[i] = static_cast<float>(dogDouble[i]);
        const cv::Mat kernelX(1, static_cast<int>(dogFloat.size()), CV_32F, dogFloat.data());
        const cv::Mat kernelY(tangential, 1, CV_32F, cv::Scalar(1.0f / static_cast<float>(tangential)));
        cv::Mat       gradFull;
        cv::sepFilter2D(warped, gradFull, CV_32F, kernelX, kernelY, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);

        const float minGradF = static_cast<float>(minGradient);
        for (int r = 0; r < samples; ++r)
        {
            const float *grad          = gradFull.ptr<float>(r + padT);
            bool         found         = false;
            double       bestAbsOffset = std::numeric_limits<double>::infinity();
            double       bestOffset    = 0.0;
            double       bestAbsGrad   = -1.0;
            for (int c = cStart; c < cEnd; ++c)
            {
                const float cur    = grad[c];
                const float absCur = std::abs(cur);
                if (!(absCur > 0.0f) || absCur < minGradF)
                    continue;
                if (options.polarity == edge::EdgePolarity::DarkToLight && cur <= 0.0f)
                    continue;
                if (options.polarity == edge::EdgePolarity::LightToDark && cur >= 0.0f)
                    continue;
                const float prev  = grad[c - 1];
                const float next  = grad[c + 1];
                const bool  isMax = cur > prev && cur >= next;
                const bool  isMin = cur < prev && cur <= next;
                if (!isMax && !isMin)
                    continue;
                const double denom    = static_cast<double>(prev) - 2.0 * cur + next;
                double       subpixel = 0.0;
                if (std::abs(denom) > 1e-12)
                    subpixel = std::clamp(0.5 * (static_cast<double>(prev) - next) / denom, -0.5, 0.5);
                const double offset = -searchHalf + (static_cast<double>(c) + subpixel) * kStep;
                const double absOff = std::abs(offset);
                const bool   better = !found
                                    || (offset != bestOffset ? absOff < bestAbsOffset
                                                             : static_cast<double>(absCur) > bestAbsGrad);
                if (better)
                {
                    found         = true;
                    bestAbsOffset = absOff;
                    bestOffset    = offset;
                    bestAbsGrad   = static_cast<double>(absCur);
                }
            }
            if (found)
                points.push_back(from + direction * (static_cast<double>(r) * uStep) + normal * bestOffset);
        }
    }
    else
    {
        std::vector<double> profileValues(static_cast<std::size_t>(normalCount));
        std::vector<double> gradValues(static_cast<std::size_t>(normalCount));
        const int           cropCols  = grayCrop.cols;
        const int           cropRows  = grayCrop.rows;
        const std::size_t   stepBytes = grayCrop.step;
        const auto         *baseData  = grayCrop.data;
        const double        invT      = 1.0 / static_cast<double>(tangential);
        const double        tCenter   = static_cast<double>(tangential - 1) * 0.5;
        const double        nxStep    = normal.x * kStep;
        const double        nyStep    = normal.y * kStep;

        for (int r = 0; r < samples; ++r)
        {
            const double u  = static_cast<double>(r) * uStep;
            const double cx = (from.x - x0) + direction.x * u - normal.x * searchHalf;
            const double cy = (from.y - y0) + direction.y * u - normal.y * searchHalf;
            for (int c = 0; c < normalCount; ++c)
            {
                const double bx = cx + nxStep * c;
                const double by = cy + nyStep * c;
                double       sum = 0.0;
                bool         ok  = true;
                for (int t = 0; t < tangential; ++t)
                {
                    const double toff = static_cast<double>(t) - tCenter;
                    const double sx   = bx + direction.x * toff;
                    const double sy   = by + direction.y * toff;
                    if (!(sx >= 0.0 && sy >= 0.0 && sx <= cropCols - 1.0 && sy <= cropRows - 1.0))
                    {
                        ok = false;
                        break;
                    }
                    const int   ix   = static_cast<int>(std::floor(sx));
                    const int   iy   = static_cast<int>(std::floor(sy));
                    const int   ix1  = std::min(ix + 1, cropCols - 1);
                    const int   iy1  = std::min(iy + 1, cropRows - 1);
                    const double fx  = sx - ix;
                    const double fy  = sy - iy;
                    const auto *row0 = baseData + static_cast<std::size_t>(iy) * stepBytes;
                    const auto *row1 = baseData + static_cast<std::size_t>(iy1) * stepBytes;
                    const double top = row0[ix] * (1.0 - fx) + row0[ix1] * fx;
                    const double bot = row1[ix] * (1.0 - fx) + row1[ix1] * fx;
                    sum += top * (1.0 - fy) + bot * fy;
                }
                profileValues[static_cast<std::size_t>(c)]
                    = ok ? sum * invT : std::numeric_limits<double>::quiet_NaN();
            }

            std::fill(gradValues.begin(), gradValues.end(), std::numeric_limits<double>::quiet_NaN());
            for (int c = kRadius; c < normalCount - kRadius; ++c)
            {
                bool   valid = true;
                double sum   = 0.0;
                for (int k = -kRadius; k <= kRadius; ++k)
                {
                    const double val = profileValues[static_cast<std::size_t>(c + k)];
                    if (!std::isfinite(val))
                    {
                        valid = false;
                        break;
                    }
                    sum += val * dogDouble[static_cast<std::size_t>(k + kRadius)];
                }
                if (valid)
                    gradValues[static_cast<std::size_t>(c)] = sum;
            }

            bool   found         = false;
            double bestAbsOffset = std::numeric_limits<double>::infinity();
            double bestOffset    = 0.0;
            double bestAbsGrad   = -1.0;
            for (int c = cStart; c < cEnd; ++c)
            {
                const double cur = gradValues[static_cast<std::size_t>(c)];
                if (!std::isfinite(cur))
                    continue;
                const double absCur = std::abs(cur);
                if (!(absCur > 0.0) || absCur < minGradient)
                    continue;
                if (options.polarity == edge::EdgePolarity::DarkToLight && cur <= 0.0)
                    continue;
                if (options.polarity == edge::EdgePolarity::LightToDark && cur >= 0.0)
                    continue;
                const double prev  = gradValues[static_cast<std::size_t>(c - 1)];
                const double next  = gradValues[static_cast<std::size_t>(c + 1)];
                const bool   isMax = cur > prev && cur >= next;
                const bool   isMin = cur < prev && cur <= next;
                if (!isMax && !isMin)
                    continue;
                const double denom    = prev - 2.0 * cur + next;
                double       subpixel = 0.0;
                if (std::abs(denom) > 1e-12)
                    subpixel = std::clamp(0.5 * (prev - next) / denom, -0.5, 0.5);
                const double offset = -searchHalf + (static_cast<double>(c) + subpixel) * kStep;
                const double absOff = std::abs(offset);
                const bool   better = !found || (offset != bestOffset ? absOff < bestAbsOffset : absCur > bestAbsGrad);
                if (better)
                {
                    found         = true;
                    bestAbsOffset = absOff;
                    bestOffset    = offset;
                    bestAbsGrad   = absCur;
                }
            }
            if (found)
                points.push_back(from + direction * u + normal * bestOffset);
        }
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
