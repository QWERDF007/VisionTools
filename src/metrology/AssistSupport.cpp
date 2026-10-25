#include "AssistSupport.h"

#include "CircleMeasureKernel.h"
#include "EdgeProfile.h"
#include "RowSpacingKernel.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <optional>

namespace visiontools::metrology {
namespace {
constexpr double kPi = 3.14159265358979323846;

std::string formatDouble(double value, int decimals = 2)
{
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.*f", decimals, value);
    return buffer;
}

roi::Record toReferenceCircle(const roi::Record &current, const geometry::Alignment &toReference)
{
    const auto inverse = toReference.inverse();
    if (!inverse)
        throw visionrt::ContractError("alignment 线性部分必须可逆");
    double x = 0, y = 0;
    inverse->apply(current.values[0], current.values[1], x, y);
    return roi::circle(x, y, current.values[2] / toReference.scale());
}

roi::Record toReferenceRectangle(const roi::Record &current, const geometry::Alignment &toReference)
{
    const auto inverse = toReference.inverse();
    if (!inverse)
        throw visionrt::ContractError("alignment 线性部分必须可逆");
    const double centerX = current.values[0] + current.values[2] / 2.0;
    const double centerY = current.values[1] + current.values[3] / 2.0;
    double       x = 0, y = 0;
    inverse->apply(centerX, centerY, x, y);
    const double rotation = std::atan2(toReference.m[3], toReference.m[0]) * 180.0 / kPi;
    const double scale    = toReference.scale();
    return roi::rectangle(x, y, current.values[2] / scale, current.values[3] / scale,
                          current.values[4] - rotation);
}

double medianOffset(std::vector<double> values)
{
    if (values.empty())
        return 0;
    for (auto &value : values)
        value = std::abs(value);
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
}

// A cursor-nearby candidate is the whole point of point-and-click: a circle that lives outside the
// local window (or whose centre is further away than the search extent) is never proposed, no
// matter how clean it looks.
constexpr int    kMaxContours   = 256;
constexpr int    kMaxWindowSide = 1024;
constexpr double kMinRadius     = 6.0;

struct LocalSearch
{
    cv::Rect    window;             // full-resolution ROI that was analysed
    cv::Mat     small;              // resized copy (may alias the ROI when no resize happened)
    double      scaleX = 1.0;       // full pixels per small pixel, per axis
    double      scaleY = 1.0;
    cv::Point2d cursor;             // cursor in small coordinates

    double scale() const { return (scaleX + scaleY) / 2.0; }

    // Pixel-centre aware mapping: integer division without the half-pixel offset biases every
    // centre by up to one pixel, which the acceptance report calls out explicitly.
    cv::Point2d toFull(const cv::Point2d &point) const
    {
        return {window.x + (point.x + 0.5) * scaleX - 0.5, window.y + (point.y + 0.5) * scaleY - 0.5};
    }
};

std::optional<LocalSearch> prepareSearch(const cv::Mat &gray, const AssistRequest &request)
{
    const double extent = std::clamp(static_cast<double>(request.searchExtent), 16.0, 4096.0);
    const int    left   = std::max(0, static_cast<int>(std::floor(request.cursor.x - extent)));
    const int    top    = std::max(0, static_cast<int>(std::floor(request.cursor.y - extent)));
    const int    right  = std::min(gray.cols, static_cast<int>(std::ceil(request.cursor.x + extent + 1.0)));
    const int    bottom = std::min(gray.rows, static_cast<int>(std::ceil(request.cursor.y + extent + 1.0)));
    if (right - left < 16 || bottom - top < 16)
        return std::nullopt;
    LocalSearch search;
    search.window = cv::Rect(left, top, right - left, bottom - top);
    const cv::Mat roi = gray(search.window);
    const int     side = std::max(roi.cols, roi.rows);
    // Bounded work: the analysed window never exceeds kMaxWindowSide on its long edge.
    int factor = 1;
    while (side / factor > kMaxWindowSide)
        factor *= 2;
    if (factor == 1)
        search.small = roi;
    else
        cv::resize(roi, search.small,
                   cv::Size(std::max(1, roi.cols / factor), std::max(1, roi.rows / factor)), 0, 0, cv::INTER_AREA);
    search.scaleX = static_cast<double>(search.window.width) / static_cast<double>(search.small.cols);
    search.scaleY = static_cast<double>(search.window.height) / static_cast<double>(search.small.rows);
    search.cursor = {(request.cursor.x - search.window.x - 0.5) / search.scaleX + 0.5,
                     (request.cursor.y - search.window.y - 0.5) / search.scaleY + 0.5};
    return search;
}
} // namespace

AssistResult assistCircle(const cv::Mat &gray, const AssistRequest &request)
{
    AssistResult result;
    if (gray.empty())
    {
        result.quality = "输入图像为空";
        return result;
    }
    // A seed circle already carries the nominal geometry, so the contour search is skipped
    // entirely: the point of the seed is that the user pointed at the target.
    std::optional<cv::Point2d> seededCenter;
    double                     seededRadius = 0;
    for (const auto &seed : request.seeds)
    {
        if (seed.kind != visionrt::RoiKind::Circle || seed.values.size() != 3)
            continue;
        double x = 0, y = 0;
        request.alignment.apply(seed.values[0], seed.values[1], x, y);
        seededCenter = cv::Point2d{x, y};
        seededRadius = seed.values[2] * request.alignment.scale();
        break;
    }

    cv::Point2d nominalCenter{request.cursor.x, request.cursor.y};
    double      nominalRadius = seededRadius;
    if (seededCenter)
    {
        nominalCenter = *seededCenter;
        const double distance = cv::norm(nominalCenter - cv::Point2d{request.cursor.x, request.cursor.y});
        // A stale seed far from the cursor must not hijack the click.
        if (distance > request.searchExtent * 4.0)
        {
            result.quality = "种子区域离光标过远";
            return result;
        }
    }
    else
    {
        const auto search = prepareSearch(gray, request);
        if (!search)
        {
            result.quality = "光标邻域超出图像范围";
            return result;
        }
        cv::Mat edges;
        cv::Canny(search->small, edges, 50, 150);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
        struct Candidate
        {
            cv::Point2d center;
            double      radius;
            double      score;
        };
        std::optional<Candidate> best;
        int                      examined = 0;
        for (const auto &contour : contours)
        {
            if (examined >= kMaxContours)
                break;
            if (contour.size() < 16)
                continue;
            ++examined;
            cv::Point2f center;
            float       enclosing = 0;
            cv::minEnclosingCircle(contour, center, enclosing);
            double meanRadius = 0;
            for (const auto &point : contour)
                meanRadius += cv::norm(cv::Point2f(static_cast<float>(point.x), static_cast<float>(point.y))
                                       - center);
            meanRadius /= static_cast<double>(contour.size());
            double residual = 0;
            for (const auto &point : contour)
                residual += std::abs(cv::norm(cv::Point2f(static_cast<float>(point.x), static_cast<float>(point.y))
                                              - center)
                                     - meanRadius);
            residual /= static_cast<double>(contour.size());
            if (!(meanRadius > 2.0) || residual > 0.18 * meanRadius)
                continue;
            const double area  = cv::contourArea(contour);
            const double ratio = area / (kPi * meanRadius * meanRadius);
            if (ratio < 0.45 || ratio > 1.1)
                continue;
            const cv::Point2d fullCenter = search->toFull({center.x, center.y});
            const double      fullRadius = meanRadius * search->scale();
            // Hard bounds, not score penalties: outside candidates simply do not exist.
            if (fullRadius < kMinRadius || fullRadius > request.searchExtent)
                continue;
            if (cv::norm(fullCenter - cv::Point2d{request.cursor.x, request.cursor.y}) > request.searchExtent)
                continue;
            const double distanceToCenter
                = cv::norm(cv::Point2d{center.x, center.y} - search->cursor);
            const double boundary = std::abs(distanceToCenter - meanRadius) * search->scale();
            const double support  = 1.0 - residual / std::max(1.0, meanRadius);
            const double inside   = distanceToCenter <= meanRadius ? 0.6 : 0.0;
            const double score    = support * 2.0 + inside
                                 - boundary / std::max(1.0, static_cast<double>(request.searchExtent));
            if (!best || score > best->score)
                best = Candidate{fullCenter, fullRadius, score};
        }
        if (!best)
        {
            result.quality = "未找到有效目标";
            return result;
        }
        nominalCenter = best->center;
        nominalRadius = best->radius;
    }
    if (!(nominalRadius > 2.0))
    {
        result.quality = "候选半径过小";
        return result;
    }
    CircleOptions options;
    options.calipers        = request.learn ? 96 : 64;
    options.searchHalfWidth = std::clamp(nominalRadius * 0.15, 3.0, 24.0);
    options.averageWidth    = 3.0;
    options.sigma           = 1.0;
    options.minGradient     = 8.0;
    options.polarity        = EdgePolarity::Any;
    options.minPoints       = 12;
    options.minCompleteness = 0.5;
    options.minCoverageDeg  = 120.0;
    options.maxRms          = 2.0;
    const auto measurement = measureCircleKernel(gray, nominalCenter, nominalRadius, options);
    if (!measurement.valid)
    {
        result.quality = measurement.reason.empty() ? "候选圆精测失败" : measurement.reason;
        return result;
    }
    // The search extent bounds candidate *search*; an existing seed ROI states its own radius.
    if (!seededCenter && measurement.radius > request.searchExtent)
    {
        result.quality = "候选圆超出搜索范围";
        return result;
    }
    result.valid    = true;
    result.preview  = {roi::circle(measurement.centerX, measurement.centerY, measurement.radius)};
    result.proposal = {toReferenceCircle(result.preview.front(), request.alignment)};
    result.polarity = measurement.positiveEdges >= measurement.negativeEdges ? "dark_to_light" : "light_to_dark";
    result.searchHalfWidth
        = std::clamp(medianOffset(measurement.edgeOffsets) * 1.5, 1.0, 50.0);
    result.quality = "半径 " + formatDouble(measurement.radius) + " px，完整度 "
                   + formatDouble(measurement.completeness * 100.0, 0) + "%，拟合误差 "
                   + formatDouble(measurement.rms) + " px";
    return result;
}

AssistResult assistRowSpacing(const cv::Mat &gray, const AssistRequest &request)
{
    AssistResult result;
    // Local structure tensor: the dominant gradient direction crosses the bands.
    const int window = std::clamp(request.searchExtent / 8, 8, 32);
    const int x0     = std::clamp(static_cast<int>(request.cursor.x) - window, 0, gray.cols - 1);
    const int x1     = std::clamp(static_cast<int>(request.cursor.x) + window, 0, gray.cols - 1);
    const int y0     = std::clamp(static_cast<int>(request.cursor.y) - window, 0, gray.rows - 1);
    const int y1     = std::clamp(static_cast<int>(request.cursor.y) + window, 0, gray.rows - 1);
    if (x1 - x0 < 8 || y1 - y0 < 8)
    {
        result.quality = "光标邻域不足以估计方向";
        return result;
    }
    cv::Mat roiImage = gray(cv::Rect(x0, y0, x1 - x0, y1 - y0));
    cv::Mat gx, gy;
    cv::Sobel(roiImage, gx, CV_32F, 1, 0, 3);
    cv::Sobel(roiImage, gy, CV_32F, 0, 1, 3);
    double ixx = 0, iyy = 0, ixy = 0;
    for (int y = 0; y < gx.rows; ++y)
        for (int x = 0; x < gx.cols; ++x)
        {
            const double dx = gx.at<float>(y, x);
            const double dy = gy.at<float>(y, x);
            ixx += dx * dx;
            iyy += dy * dy;
            ixy += dx * dy;
        }
    const double theta = 0.5 * std::atan2(2.0 * ixy, ixx - iyy);
    if (!std::isfinite(theta) || (ixx + iyy) < 1.0)
    {
        result.quality = "光标邻域梯度不足，方向不明确";
        return result;
    }
    const auto direction = cv::Point2d{std::cos(theta), std::sin(theta)};
    const auto normal    = cv::Point2d{-std::sin(theta), std::cos(theta)};

    // Keep the sampling window inside the image.
    const auto limitAlong = [&](const cv::Point2d &axis)
    {
        double half = static_cast<double>(request.searchExtent) / 2.0;
        if (axis.x > 1e-9)
            half = std::min(half, (gray.cols - 1 - request.cursor.x) / axis.x);
        else if (axis.x < -1e-9)
            half = std::min(half, (0.0 - request.cursor.x) / axis.x);
        if (axis.y > 1e-9)
            half = std::min(half, (gray.rows - 1 - request.cursor.y) / axis.y);
        else if (axis.y < -1e-9)
            half = std::min(half, (0.0 - request.cursor.y) / axis.y);
        return std::max(0.0, half);
    };
    const double halfLength = std::min(limitAlong(direction), limitAlong(cv::Point2d{-direction.x, -direction.y}));
    const double halfAcross = std::min(limitAlong(normal), limitAlong(cv::Point2d{-normal.x, -normal.y}));
    if (halfLength < 8.0 || halfAcross < 4.0)
    {
        result.quality = "光标邻域超出图像范围";
        return result;
    }
    ProfileConfig config;
    config.searchHalfWidth = halfLength;
    config.averageWidth    = std::max(1.0, halfAcross * 2.0 - 1.0);
    config.sigma           = 1.0;
    config.minGradient     = 8.0;
    config.polarity        = EdgePolarity::Any;
    const auto profile = buildProfile(gray, request.cursor, direction, halfLength, normal,
                                      std::max(0.0, halfAcross - 1.0), 0.5);
    const auto edges = detectEdges(profile, config);
    if (edges.size() < 2)
    {
        result.quality = "光标邻域未找到边缘对";
        return result;
    }
    // Choose the adjacent opposite-polarity pair closest to the cursor.
    std::optional<std::pair<EdgeCandidate, EdgeCandidate>> pair;
    double bestDistance = 0;
    for (std::size_t index = 0; index + 1 < edges.size(); ++index)
    {
        const auto &first  = edges[index];
        const auto &second = edges[index + 1];
        if ((first.gradient > 0.0) == (second.gradient > 0.0))
            continue;
        if (first.offset > second.offset)
            continue;
        const double distance = std::abs((first.offset + second.offset) / 2.0);
        if (!pair || distance < bestDistance)
        {
            pair         = std::make_pair(first, second);
            bestDistance = distance;
        }
    }
    if (!pair)
    {
        result.quality = "光标邻域未找到完整边缘对";
        return result;
    }
    const double firstWidth = pair->second.offset - pair->first.offset;
    if (!(firstWidth > 0.5))
    {
        result.quality = "候选宽度过小";
        return result;
    }
    const double centerOffset = (pair->first.offset + pair->second.offset) / 2.0;
    const auto   center       = cv::Point2d{request.cursor.x + direction.x * centerOffset,
                                            request.cursor.y + direction.y * centerOffset};
    const double rectWidth    = std::clamp(firstWidth * (request.learn ? 8.0 : 6.0), 16.0,
                                            static_cast<double>(request.searchExtent) * 2.0);
    const double rectHeight   = std::max(3.0, halfAcross * 2.0 - 1.0);
    const auto   rectangle    = roi::rectangle(center.x, center.y, rectWidth, rectHeight,
                                               std::atan2(direction.y, direction.x) * 180.0 / kPi);
    RowOptions options;
    options.sigma        = 1.0;
    options.minGradient  = 8.0;
    options.brightObject = pair->first.gradient > 0.0;
    options.widthMin     = 1.0;
    options.widthMax     = 1.0e6;
    const auto measurement = measureRowRect(gray, rectangle, options);
    if (!measurement.valid)
    {
        result.quality = measurement.reason;
        return result;
    }
    double minimumWidth = measurement.widths.front();
    double maximumWidth = measurement.widths.front();
    for (double value : measurement.widths)
    {
        minimumWidth = std::min(minimumWidth, value);
        maximumWidth = std::max(maximumWidth, value);
    }
    result.valid    = true;
    result.preview  = {rectangle};
    result.proposal = {toReferenceRectangle(rectangle, request.alignment)};
    result.polarity = options.brightObject ? "bright" : "dark";
    result.widthMin = std::max(0.5, minimumWidth * 0.8);
    result.widthMax = std::max(result.widthMin, maximumWidth * 1.2);
    result.quality  = "宽度 " + formatDouble(minimumWidth) + "~" + formatDouble(maximumWidth) + " px，"
                    + std::to_string(measurement.pairs.size()) + " 对边缘";
    return result;
}
} // namespace visiontools::metrology
