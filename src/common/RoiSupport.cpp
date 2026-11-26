#include "RoiSupport.h"

#include "common/errors/GeometryErrors.h"
#include "common/errors/ToolError.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace visiontools::roi {
namespace {
// Conservative pixel-center tolerance used by the shared geometry component.
constexpr double kBoundaryTolerance = 0.5;

bool pointInPolygon(const visionrt::PolygonView &polygon, double x, double y)
{
    const std::size_t count = polygon.size();
    if (count < 3)
        return false;
    bool inside = false;
    for (std::size_t i = 0, j = count - 1; i < count; j = i++)
    {
        const auto a = polygon.point(j);
        const auto b = polygon.point(i);
        if ((b.y > y) != (a.y > y) && x < (a.x - b.x) * (y - b.y) / (a.y - b.y) + b.x)
            inside = !inside;
    }
    return inside;
}
} // namespace

CircularSearch circularSearch(const visionrt::PortValue &value, const char *portName)
{
    const auto regions = groups(value, 1, portName);
    if (regions.size() != 1 || regions.front().size() != 1)
        fail(CommonToolError::InputOutOfRange, std::string(portName) + " 必须包含一个圆形搜索区域");
    const auto region = visionrt::RegionView(regions.front().front()).circle();
    if (region.radius() < 16 || region.radius() > 2048)
        fail(CommonToolError::InputOutOfRange, std::string(portName) + " 搜索半径必须在 [16, 2048] 内");
    return {region.center(), static_cast<int>(region.radius())};
}

std::vector<std::vector<Record>> groups(const visionrt::PortValue &value, std::size_t imageCount,
                                        const char *portName)
{
    const std::string                name(portName);
    if (visionrt::isMissing(value))
        return {};
    const auto *reference = visionrt::shapeRegions(value);
    if (!reference)
        fail(CommonToolError::InputTypeMismatch, name + " 必须声明具体形状类型");
    if (!reference->empty() && reference->size() != imageCount)
        fail(CommonToolError::InputOutOfRange, name + " 外层长度必须与图像数量一致");
    return *reference;
}

bool anyRecord(const std::vector<std::vector<Record>> &records) noexcept
{
    return std::any_of(records.begin(), records.end(),
                       [](const auto &image) { return !image.empty(); });
}

BoundedMask boundedRegionMask(const std::vector<Record> &records, cv::Size size, const geometry::Alignment &toReference)
{
    if (records.empty() || size.width <= 0 || size.height <= 0)
        return {};
    const auto forward = toReference.inverse();
    if (!forward)
        fail(GeometryError::AlignmentInvalid, "线性部分不可逆");

    struct ClippedBox
    {
        int x0 = 0;
        int y0 = 0;
        int x1 = 0;
        int y1 = 0;
    };
    std::vector<ClippedBox> boxes;
    boxes.reserve(records.size());
    int unionX0 = size.width;
    int unionY0 = size.height;
    int unionX1 = 0;
    int unionY1 = 0;

    for (const auto &record : records)
    {
        double minX = std::numeric_limits<double>::max(), minY = minX;
        double maxX = std::numeric_limits<double>::lowest(), maxY = maxX;
        const auto extend = [&](double referenceX, double referenceY)
        {
            double x = 0, y = 0;
            forward->apply(referenceX, referenceY, x, y);
            minX = std::min(minX, x);
            minY = std::min(minY, y);
            maxX = std::max(maxX, x);
            maxY = std::max(maxY, y);
        };
        const auto view = visionrt::RegionView(record);
        switch (view.kind())
        {
        case visionrt::RoiKind::Rectangle:
            for (const auto &corner : view.rectangle().corners()) extend(corner.x, corner.y);
            break;
        case visionrt::RoiKind::Circle:
        {
            const auto   circle = view.circle();
            const auto   center = circle.center();
            const double radius = circle.radius();
            extend(center.x - radius, center.y - radius);
            extend(center.x + radius, center.y - radius);
            extend(center.x + radius, center.y + radius);
            extend(center.x - radius, center.y + radius);
            break;
        }
        case visionrt::RoiKind::Polygon:
        {
            const auto polygon = view.polygon();
            for (std::size_t index = 0; index < polygon.size(); ++index)
            {
                const auto point = polygon.point(index);
                extend(point.x, point.y);
            }
            break;
        }
        default:
            fail(CommonToolError::InputTypeMismatch, "面积选择只接受旋转矩形、圆和多边形 ROI");
        }
        const int x0 = std::clamp(static_cast<int>(std::floor(minX - 1.0)), 0, size.width);
        const int y0 = std::clamp(static_cast<int>(std::floor(minY - 1.0)), 0, size.height);
        const int x1 = std::clamp(static_cast<int>(std::ceil(maxX + 1.0)), 0, size.width);
        const int y1 = std::clamp(static_cast<int>(std::ceil(maxY + 1.0)), 0, size.height);
        boxes.push_back({x0, y0, x1, y1});
        if (x1 > x0 && y1 > y0)
        {
            unionX0 = std::min(unionX0, x0);
            unionY0 = std::min(unionY0, y0);
            unionX1 = std::max(unionX1, x1);
            unionY1 = std::max(unionY1, y1);
        }
    }
    if (unionX1 <= unionX0 || unionY1 <= unionY0)
        return {};

    BoundedMask result;
    result.bounds = cv::Rect(unionX0, unionY0, unionX1 - unionX0, unionY1 - unionY0);
    result.mask   = cv::Mat::zeros(result.bounds.size(), CV_8UC1);
    const double m00 = toReference.m[0], m01 = toReference.m[1], m02 = toReference.m[2];
    const double m10 = toReference.m[3], m11 = toReference.m[4], m12 = toReference.m[5];

    for (std::size_t index = 0; index < records.size(); ++index)
    {
        const auto &box = boxes[index];
        if (box.x1 <= box.x0 || box.y1 <= box.y0)
            continue;
        const auto view = visionrt::RegionView(records[index]);
        switch (view.kind())
        {
        case visionrt::RoiKind::Rectangle:
        {
            const auto   rectangle  = view.rectangle();
            const auto   center     = rectangle.center();
            const double angle      = rectangle.angle() * 3.14159265358979323846 / 180.0;
            const double cosine     = std::cos(angle);
            const double sine       = std::sin(angle);
            const double halfWidth  = rectangle.width() / 2.0 + kBoundaryTolerance;
            const double halfHeight = rectangle.height() / 2.0 + kBoundaryTolerance;
            for (int y = box.y0; y < box.y1; ++y)
            {
                auto        *dst  = result.mask.ptr<std::uint8_t>(y - unionY0) - unionX0;
                const double rowX = m01 * y + m02 - center.x;
                const double rowY = m11 * y + m12 - center.y;
                for (int x = box.x0; x < box.x1; ++x)
                {
                    const double dx = m00 * x + rowX;
                    const double dy = m10 * x + rowY;
                    if (std::abs(dx * cosine + dy * sine) <= halfWidth
                        && std::abs(-dx * sine + dy * cosine) <= halfHeight)
                        dst[x] = 255;
                }
            }
            break;
        }
        case visionrt::RoiKind::Circle:
        {
            const auto   circle    = view.circle();
            const auto   center    = circle.center();
            const double radiusLim = circle.radius() + kBoundaryTolerance;
            const double radiusSq  = radiusLim * radiusLim;
            for (int y = box.y0; y < box.y1; ++y)
            {
                auto        *dst  = result.mask.ptr<std::uint8_t>(y - unionY0) - unionX0;
                const double rowX = m01 * y + m02 - center.x;
                const double rowY = m11 * y + m12 - center.y;
                for (int x = box.x0; x < box.x1; ++x)
                {
                    const double dx = m00 * x + rowX;
                    const double dy = m10 * x + rowY;
                    if (dx * dx + dy * dy <= radiusSq)
                        dst[x] = 255;
                }
            }
            break;
        }
        case visionrt::RoiKind::Polygon:
        {
            const auto polygon = view.polygon();
            for (int y = box.y0; y < box.y1; ++y)
            {
                auto        *dst  = result.mask.ptr<std::uint8_t>(y - unionY0) - unionX0;
                const double rowX = m01 * y + m02;
                const double rowY = m11 * y + m12;
                for (int x = box.x0; x < box.x1; ++x)
                {
                    if (pointInPolygon(polygon, m00 * x + rowX, m10 * x + rowY))
                        dst[x] = 255;
                }
            }
            break;
        }
        default:
            break;
        }
    }
    return result;
}

cv::Mat regionMask(const std::vector<Record> &records, cv::Size size, const geometry::Alignment &toReference)
{
    cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
    const auto bounded = boundedRegionMask(records, size, toReference);
    if (!bounded.empty())
        bounded.mask.copyTo(mask(bounded.bounds));
    return mask;
}

Record point(double x, double y)
{
    return visionrt::makePoint(x, y);
}

Record circle(double centerX, double centerY, double radius)
{
    return visionrt::makeCircle({centerX, centerY}, radius);
}

Record rectangle(double centerX, double centerY, double width, double height, double angleDegrees)
{
    return visionrt::makeRectangle(centerX - width / 2.0, centerY - height / 2.0, width, height, angleDegrees);
}

Record segmentValue(double x1, double y1, double x2, double y2)
{
    return visionrt::makeSegment({x1, y1}, {x2, y2});
}

Record point(std::vector<cv::Point2d> values)
{
    return visionrt::makePoint(values);
}

} // namespace visiontools::roi
