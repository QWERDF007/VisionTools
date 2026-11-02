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

std::vector<std::vector<Record>> groups(const visionrt::PortValue &value, std::size_t imageCount,
                                        const char *portName)
{
    std::vector<std::vector<Record>> result;
    const std::string                name(portName);
    if (visionrt::isMissing(value))
        return result;
    const auto *reference = std::get_if<visionrt::DataRef<visionrt::ROI>>(&value);
    if (!reference)
        fail(CommonToolError::InputTypeMismatch, name + " 必须是 ROI");
    const auto &images = reference->get();
    if (!images.empty() && images.size() != imageCount)
        fail(CommonToolError::InputOutOfRange, name + " 外层长度必须为 " + std::to_string(imageCount) + " 张图像，实际 "
                                      + std::to_string(images.size()));
    result.resize(images.size());
    for (std::size_t image = 0; image < images.size(); ++image)
    {
        for (const auto &record : images[image])
        {
            if (record.empty())
                fail(GeometryError::RoiOutOfImage, name + " 含空 ROI 记录");
            try
            {
                // One validation per region at the external entry; downstream code receives the
                // record as an already-checked encoding and reads it through typed views.
                (void)visionrt::RegionView(record);
            }
            catch (const visionrt::ContractError &error)
            {
                fail(CommonToolError::InputOutOfRange, name + " 的 ROI 记录不合法：" + error.what());
            }
            result[image].push_back(record);
        }
    }
    return result;
}

bool anyRecord(const std::vector<std::vector<Record>> &records) noexcept
{
    return std::any_of(records.begin(), records.end(),
                       [](const auto &image) { return !image.empty(); });
}

bool contains(const Record &record, double currentX, double currentY, const geometry::Alignment &toReference)
{
    const auto view = visionrt::RegionView(record);
    double     x = 0, y = 0;
    toReference.apply(currentX, currentY, x, y);
    switch (view.kind())
    {
    case visionrt::RoiKind::Rectangle:
    {
        const auto   rectangle = view.rectangle();
        const double angle     = rectangle.angle() * 3.14159265358979323846 / 180.0;
        const double cos       = std::cos(angle), sin = std::sin(angle);
        const auto   center    = rectangle.center();
        const double dx = x - center.x, dy = y - center.y;
        const double localX = dx * cos + dy * sin;
        const double localY = -dx * sin + dy * cos;
        return std::abs(localX) <= rectangle.width() / 2.0 + kBoundaryTolerance
            && std::abs(localY) <= rectangle.height() / 2.0 + kBoundaryTolerance;
    }
    case visionrt::RoiKind::Circle:
    {
        const auto   circle = view.circle();
        const auto   center = circle.center();
        const double dx = x - center.x, dy = y - center.y;
        return std::sqrt(dx * dx + dy * dy) <= circle.radius() + kBoundaryTolerance;
    }
    case visionrt::RoiKind::Polygon:
        return pointInPolygon(view.polygon(), x, y);
    default:
        fail(CommonToolError::InputTypeMismatch, "面积选择只接受旋转矩形、圆和多边形 ROI");
    }
}

cv::Mat regionMask(const std::vector<Record> &records, cv::Size size, const geometry::Alignment &toReference)
{
    cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
    if (records.empty())
        return mask;
    const auto forward = toReference.inverse();
    if (!forward)
        fail(GeometryError::AlignmentInvalid, "线性部分不可逆");
    for (const auto &record : records)
    {
        // Bounding box in current image coordinates: transform the reference
        // bounding box corners and clamp to the image.
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
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x)
                if (contains(record, x, y, toReference))
                    mask.at<std::uint8_t>(y, x) = 255;
    }
    return mask;
}

visionrt::ROI toRoi(std::vector<std::vector<Record>> records)
{
    visionrt::ROI result;
    result.reserve(records.size());
    for (auto &image : records) result.push_back(std::move(image));
    return result;
}

visionrt::PortValue roiValue(std::vector<std::vector<Record>> records)
{
    return visionrt::DataRef<visionrt::ROI>(toRoi(std::move(records)));
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

Record pointSet(std::vector<cv::Point2d> values)
{
    return visionrt::makePointSet(values);
}

} // namespace visiontools::roi
