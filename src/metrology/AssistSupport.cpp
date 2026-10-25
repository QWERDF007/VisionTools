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
} // namespace

AssistResult assistCircle(const cv::Mat &gray, const AssistRequest &request)
{
    AssistResult result;
    int          factor = 1;
    while (std::max(gray.cols, gray.rows) / factor > 720)
        factor *= 2;
    cv::Mat small;
    if (factor == 1)
        small = gray;
    else
        cv::resize(gray, small, cv::Size(gray.cols / factor, gray.rows / factor), 0, 0, cv::INTER_AREA);
    cv::Mat edges;
    cv::Canny(small, edges, 50, 150);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    struct Candidate
    {
        cv::Point2d center;
        double      radius;
        double      score;
    };
    std::optional<Candidate> best;
    for (const auto &contour : contours)
    {
        if (contour.size() < 16)
            continue;
        cv::Point2f center;
        float       radius = 0;
        cv::minEnclosingCircle(contour, center, radius);
        // The mean radius of the contour points is a far better nominal estimate
        // than the enclosing radius, which is biased outward.
        double meanRadius = 0;
        for (const auto &point : contour)
            meanRadius += cv::norm(cv::Point2f(static_cast<float>(point.x), static_cast<float>(point.y)) - center);
        meanRadius /= static_cast<double>(contour.size());
        radius = static_cast<float>(meanRadius);
        const double scaledRadius = radius * factor;
        if (scaledRadius < 6.0 || scaledRadius > request.searchExtent)
            continue;
        double residual = 0;
        for (const auto &point : contour)
            residual += std::abs(cv::norm(cv::Point2f(static_cast<float>(point.x), static_cast<float>(point.y))
                                          - center)
                                 - radius);
        residual /= static_cast<double>(contour.size());
        if (residual > 0.18 * radius)
            continue;
        const double area = cv::contourArea(contour);
        const double ratio = area / (kPi * radius * radius);
        if (ratio < 0.45 || ratio > 1.1)
            continue;
        const auto   cursor   = cv::Point2d{request.cursor.x / factor, request.cursor.y / factor};
        const double distance = cv::norm(cursor - cv::Point2d{center.x, center.y});
        const double boundary = std::abs(distance - radius) * factor;
        const double support  = 1.0 - residual / std::max(1.0, static_cast<double>(radius));
        const double inside   = distance <= radius ? 0.6 : 0.0;
        const double score    = support * 2.0 + inside - boundary / std::max(1.0, static_cast<double>(request.searchExtent));
        if (!best || score > best->score)
            best = Candidate{{center.x * factor, center.y * factor}, scaledRadius, score};
    }
    if (!best)
    {
        result.quality = "未找到有效目标";
        return result;
    }

    // A seed circle is a stronger nominal input than the contour estimate.
    cv::Point2d nominalCenter = best->center;
    double      nominalRadius = best->radius;
    for (const auto &seed : request.seeds)
    {
        if (seed.kind != visionrt::RoiKind::Circle || seed.values.size() != 3)
            continue;
        double x = 0, y = 0;
        request.alignment.apply(seed.values[0], seed.values[1], x, y);
        nominalCenter = {x, y};
        nominalRadius = seed.values[2] * request.alignment.scale();
        break;
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
    if (measurement.radius > request.searchExtent)
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
