#include "RoiSupport.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace visiontools::roi {
namespace {
// Conservative pixel-center tolerance used by the shared geometry component.
constexpr double kBoundaryTolerance = 0.5;

bool pointInPolygon(const std::vector<double> &values, std::size_t offset, double x, double y)
{
    const std::size_t count = (values.size() - offset) / 2;
    if (count < 3)
        return false;
    bool inside = false;
    for (std::size_t i = 0, j = count - 1; i < count; j = i++)
    {
        const double xi = values[offset + i * 2], yi = values[offset + i * 2 + 1];
        const double xj = values[offset + j * 2], yj = values[offset + j * 2 + 1];
        if ((yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi)
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
        throw visionrt::ContractError(name + " 必须是 ROI");
    const auto &images = reference->get();
    if (!images.empty() && images.size() != imageCount)
        throw visionrt::ContractError(name + " 外层长度必须为 " + std::to_string(imageCount) + " 张图像，实际 "
                                      + std::to_string(images.size()));
    result.resize(images.size());
    for (std::size_t image = 0; image < images.size(); ++image)
    {
        for (const auto &record : images[image])
        {
            if (record.empty())
                throw visionrt::ContractError(name + " 含空 ROI 记录");
            const int kind = static_cast<int>(record.front());
            if (kind < 0 || kind > 5)
                throw visionrt::ContractError(name + " 的 ROI kind 超出 0-5");
            if (std::any_of(record.begin(), record.end(), [](double v) { return !std::isfinite(v); }))
                throw visionrt::ContractError(name + " 的 ROI 值必须有限");
            Record decoded;
            decoded.kind = static_cast<visionrt::RoiKind>(kind);
            decoded.values.assign(record.begin() + 1, record.end());
            result[image].push_back(std::move(decoded));
        }
    }
    return result;
}

bool anyRecord(const std::vector<std::vector<Record>> &records) noexcept
{
    return std::any_of(records.begin(), records.end(),
                       [](const auto &image) { return !image.empty(); });
}

std::vector<double> encoded(const Record &record)
{
    std::vector<double> result;
    result.reserve(record.values.size() + 1);
    result.push_back(static_cast<double>(record.kind));
    result.insert(result.end(), record.values.begin(), record.values.end());
    return result;
}

bool contains(const Record &record, double currentX, double currentY, const geometry::Alignment &toReference)
{
    double x = 0, y = 0;
    toReference.apply(currentX, currentY, x, y);
    switch (record.kind)
    {
    case visionrt::RoiKind::Rectangle:
    {
        if (record.values.size() != 5)
            throw visionrt::ContractError("矩形 ROI 需要 5 个值");
        const double cx = record.values[0] + record.values[2] / 2.0;
        const double cy = record.values[1] + record.values[3] / 2.0;
        const double angle = record.values[4] * 3.14159265358979323846 / 180.0;
        const double cos   = std::cos(angle), sin = std::sin(angle);
        const double dx    = x - cx, dy = y - cy;
        const double localX = dx * cos + dy * sin;
        const double localY = -dx * sin + dy * cos;
        return std::abs(localX) <= record.values[2] / 2.0 + kBoundaryTolerance
            && std::abs(localY) <= record.values[3] / 2.0 + kBoundaryTolerance;
    }
    case visionrt::RoiKind::Circle:
    {
        if (record.values.size() != 3)
            throw visionrt::ContractError("圆 ROI 需要 3 个值");
        const double dx = x - record.values[0], dy = y - record.values[1];
        return std::sqrt(dx * dx + dy * dy) <= record.values[2] + kBoundaryTolerance;
    }
    case visionrt::RoiKind::Polygon:
        return pointInPolygon(record.values, 0, x, y);
    default:
        throw visionrt::ContractError("面积选择只接受旋转矩形、圆和多边形 ROI");
    }
}

cv::Mat regionMask(const std::vector<Record> &records, cv::Size size, const geometry::Alignment &toReference)
{
    cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
    if (records.empty())
        return mask;
    const auto forward = toReference.inverse();
    if (!forward)
        throw visionrt::ContractError("alignment 线性部分必须可逆");
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
        switch (record.kind)
        {
        case visionrt::RoiKind::Rectangle:
        {
            const double angle = record.values[4] * 3.14159265358979323846 / 180.0;
            const double cx    = record.values[0] + record.values[2] / 2.0;
            const double cy    = record.values[1] + record.values[3] / 2.0;
            const double cos = std::cos(angle), sin = std::sin(angle);
            for (const auto &[sx, sy] : {std::pair{-0.5, -0.5}, std::pair{0.5, -0.5}, std::pair{0.5, 0.5},
                                         std::pair{-0.5, 0.5}})
            {
                const double localX = record.values[2] * sx, localY = record.values[3] * sy;
                extend(cx + localX * cos - localY * sin, cy + localX * sin + localY * cos);
            }
            break;
        }
        case visionrt::RoiKind::Circle:
        {
            const double radius = record.values[2];
            for (const auto &[dx, dy] : {std::pair{-1.0, -1.0}, std::pair{1.0, -1.0}, std::pair{1.0, 1.0},
                                         std::pair{-1.0, 1.0}})
                extend(record.values[0] + dx * radius, record.values[1] + dy * radius);
            break;
        }
        case visionrt::RoiKind::Polygon:
            for (std::size_t index = 0; index + 1 < record.values.size(); index += 2)
                extend(record.values[index], record.values[index + 1]);
            break;
        default:
            throw visionrt::ContractError("面积选择只接受旋转矩形、圆和多边形 ROI");
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
    for (auto &image : records)
    {
        std::vector<std::vector<double>> encodedGroup;
        encodedGroup.reserve(image.size());
        for (const auto &record : image)
            encodedGroup.push_back(encoded(record));
        result.push_back(std::move(encodedGroup));
    }
    return result;
}

visionrt::PortValue roiValue(std::vector<std::vector<Record>> records)
{
    return visionrt::DataRef<visionrt::ROI>(toRoi(std::move(records)));
}

Record point(double x, double y)
{
    return Record{visionrt::RoiKind::Point, {x, y}};
}

Record circle(double centerX, double centerY, double radius)
{
    return Record{visionrt::RoiKind::Circle, {centerX, centerY, radius}};
}

Record rectangle(double centerX, double centerY, double width, double height, double angleDegrees)
{
    return Record{visionrt::RoiKind::Rectangle,
                  {centerX - width / 2.0, centerY - height / 2.0, width, height, angleDegrees}};
}

Record segmentValue(double x1, double y1, double x2, double y2)
{
    return Record{visionrt::RoiKind::Segment, {x1, y1, x2, y2}};
}

Record pointSet(std::vector<cv::Point2d> values)
{
    Record record;
    record.kind = visionrt::RoiKind::PointSet;
    record.values.reserve(values.size() * 2);
    for (const auto &value : values)
    {
        record.values.push_back(value.x);
        record.values.push_back(value.y);
    }
    return record;
}
} // namespace visiontools::roi
