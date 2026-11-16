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
        result.quality = "光标附近没有明显边缘";
        return result;
    }
    result.coherence = std::sqrt((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy) / trace;
    if (result.coherence < kMinCoherence)
    {
        result.quality = "光标附近边缘方向不明确";
        return result;
    }
    // Gradient direction of the dominant structure, then the edge runs across it.
    result.radians = 0.5 * std::atan2(2.0 * jxy, jxx - jyy) + kPi / 2.0;
    result.valid   = true;
    return result;
}

// The polarity the alignment tool would need for this edge: does intensity rise along the probe's
// normal, which the locator defines as the direction rotated a quarter turn from the edge.
std::string polarityAt(const cv::Mat &gray, const cv::Point2d &cursor, double edgeRadians)
{
    const cv::Point2d normal{-std::sin(edgeRadians), std::cos(edgeRadians)};
    const auto        sample = [&](double offset)
    {
        const int x = clampIndex(cursor.x + normal.x * offset, 0, gray.cols - 1);
        const int y = clampIndex(cursor.y + normal.y * offset, 0, gray.rows - 1);
        return static_cast<double>(gray.at<std::uint8_t>(y, x));
    };
    const double rise = sample(3.0) - sample(-3.0);
    if (std::abs(rise) < 1.0)
        return "any";
    return rise > 0.0 ? "dark_to_light" : "light_to_dark";
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
    const int half = std::clamp(request.searchExtent, 16, 512);
    const int left   = clampIndex(request.cursor.x - half, 0, std::max(0, gray.cols - 1));
    const int top    = clampIndex(request.cursor.y - half, 0, std::max(0, gray.rows - 1));
    const int right  = clampIndex(request.cursor.x + half, 0, std::max(0, gray.cols - 1));
    const int bottom = clampIndex(request.cursor.y + half, 0, std::max(0, gray.rows - 1));
    const cv::Rect window{left, top, right - left + 1, bottom - top + 1};
    if (window.width < 16 || window.height < 16)
    {
        result.quality = "光标附近没有足够的图像区域";
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
    probe.from            = request.cursor - direction * half;
    probe.to              = request.cursor + direction * half;
    probe.search.center   = request.cursor;
    probe.search.width    = 2.0 * half;
    probe.search.height   = static_cast<double>(half);
    probe.search.angle    = degrees(orientation.radians);

    EdgeSearchOptions options;
    options.contrast        = 20.0;
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
        result.quality = "光标附近没有可用的直边";
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
    result.polarity = polarityAt(gray, request.cursor, angle);
    result.quality  = "已沿边缘方向放置搜索框";
    result.valid    = true;
    return result;
}

} // namespace visiontools::recognition
