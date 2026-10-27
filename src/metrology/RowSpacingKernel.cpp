#include "RowSpacingKernel.h"

#include "EdgeProfile.h"

#include <algorithm>
#include <cmath>

namespace visiontools::metrology {
namespace {
constexpr double kPi = 3.14159265358979323846;

bool insideImage(const cv::Point2d &point, cv::Size size)
{
    return point.x >= -0.5 && point.y >= -0.5 && point.x <= size.width - 0.5 && point.y <= size.height - 0.5;
}
} // namespace

RowMeasurement measureRowRect(const cv::Mat &gray, const roi::Record &rectangle, const RowOptions &options)
{
    RowMeasurement result;
    if (rectangle.kind != visionrt::RoiKind::Rectangle || rectangle.values.size() != 5)
        throw visionrt::ContractError("行间距测量要求每个 ROI 为 kind=2 的旋转矩形");
    const double width  = rectangle.values[2];
    const double height = rectangle.values[3];
    if (!(width > 0.0) || !(height > 0.0))
        throw visionrt::ContractError("行间距 ROI 宽度和高度必须为正");
    if (width < 4.0 || height < 1.0)
    {
        result.reason = "ROI 宽高不足以支持滤波";
        return result;
    }
    const double angle = rectangle.values[4] * kPi / 180.0;
    const auto   center
        = cv::Point2d{rectangle.values[0] + width / 2.0, rectangle.values[1] + height / 2.0};
    const auto direction = cv::Point2d{std::cos(angle), std::sin(angle)};
    const auto normal    = cv::Point2d{-std::sin(angle), std::cos(angle)};

    for (const auto &[sx, sy] : {std::pair{-0.5, -0.5}, std::pair{0.5, -0.5}, std::pair{0.5, 0.5},
                                 std::pair{-0.5, 0.5}})
    {
        const auto corner = cv::Point2d{center.x + direction.x * width * sx + normal.x * height * sy,
                                        center.y + direction.y * width * sx + normal.y * height * sy};
        if (!insideImage(corner, gray.size()))
        {
            result.reason = "ROI 超出图像范围，支持窗口不足";
            return result;
        }
    }

    ProfileConfig config;
    config.searchHalfWidth = width / 2.0;
    config.averageWidth    = height;
    config.sigma           = options.sigma;
    config.minGradient     = options.minGradient;
    config.polarity        = EdgePolarity::Any;
    if (!(config.searchHalfWidth > 3.0 * options.sigma))
    {
        result.reason = "ROI 宽度不足以支持指定的平滑窗口";
        return result;
    }

    const auto profile = buildProfile(gray, center, direction, width / 2.0, normal, (height - 1.0) / 2.0, 0.5);
    const auto edges   = detectEdges(profile, config);

    // Compress every same-polarity run to its strongest member; ties keep the
    // earlier edge. Pairing never reuses an edge.
    struct Extremum
    {
        double offset;
        double gradient;
    };
    std::vector<Extremum> chosen;
    for (const auto &edge : edges)
    {
        const bool positive = edge.gradient > 0.0;
        if (!chosen.empty() && (chosen.back().gradient > 0.0) == positive)
        {
            if (std::abs(edge.gradient) > std::abs(chosen.back().gradient))
                chosen.back() = {edge.offset, edge.gradient};
            continue;
        }
        chosen.push_back({edge.offset, edge.gradient});
    }

    const bool firstPositive = options.brightObject;
    for (std::size_t index = 0; index + 1 < chosen.size();)
    {
        const bool positive = chosen[index].gradient > 0.0;
        const bool opposite = (chosen[index + 1].gradient > 0.0) != positive;
        if (positive == firstPositive && opposite && chosen[index].offset < chosen[index + 1].offset)
        {
            result.pairs.push_back({chosen[index].offset, chosen[index + 1].offset});
            index += 2;
        }
        else
            ++index;
    }
    result.unpaired = static_cast<int>(chosen.size()) - static_cast<int>(result.pairs.size()) * 2;
    if (result.pairs.empty())
    {
        result.reason = "未找到完整边缘对";
        return result;
    }
    for (const auto &pair : result.pairs)
    {
        const double measured = pair.second - pair.first;
        result.widths.push_back(measured);
        result.accepted.push_back(measured >= options.widthMin && measured <= options.widthMax);
    }
    for (std::size_t index = 0; index + 1 < result.pairs.size(); ++index)
    {
        const auto &current = result.pairs[index];
        const auto &next    = result.pairs[index + 1];
        result.gaps.push_back(next.first - current.second);
        result.pitches.push_back(((next.first + next.second) - (current.first + current.second)) / 2.0);
    }
    result.valid = true;
    return result;
}
} // namespace visiontools::metrology
