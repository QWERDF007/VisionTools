#include "RowSpacingTool.h"

#include "EdgeProfile.h"
#include "RowSpacingKernel.h"

#include "common/GeometryTransform.h"
#include "common/ToolOutputs.h"
#include "common/UpstreamStatus.h"
#include "common/ImageSupport.h"
#include "common/InputValues.h"
#include "common/RoiSupport.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace visiontools::metrology {
namespace {
namespace p = ports::measure_row_spacing;
constexpr double kPi = 3.14159265358979323846;

visionrt::Values missingOutputs()
{
    return missingPorts({p::out::IMAGES, p::out::VALID_ROI, p::out::REASONS, p::out::PAIR_COUNT,
                         p::out::ACCEPTED_COUNT, p::out::UNPAIRED_COUNT, p::out::ACCEPTED, p::out::WIDTH_PX,
                         p::out::GAP_PX, p::out::PITCH_PX, p::out::WIDTH, p::out::GAP, p::out::PITCH,
                         p::out::PAIR_POINTS, p::out::OVERLAY_SEGMENTS, p::out::UNIT});
}
} // namespace

visionrt::ToolStatus RowSpacingTool::initialize(const visionrt::Values &parameters)
{
    if (!parameters.empty())
        return {1, "行间距测量不接受初始化参数"};
    return {};
}

visionrt::ToolResult RowSpacingTool::execute(const visionrt::Values &inputs)
{
    // An unselected upstream branch is not an error: no algorithm runs and nothing is written.
    if (const auto upstream = upstreamStatus(inputs); upstream != 0)
        return skippedByUpstream(upstream, missingOutputs());
    try
    {
        const auto &images = requireImages(inputs, p::in::IMAGES);
        RowOptions options;
        options.sigma          = requireFloatInRange(inputs, p::in::SIGMA, 0.1, 20.0);
        options.minGradient    = requireFloatInRange(inputs, p::in::MIN_GRADIENT, 0.0, 255.0);
        const auto polarity     = requireString(inputs, p::in::OBJECT_POLARITY);
        if (polarity != "bright" && polarity != "dark")
            throw visionrt::ContractError("输入端口 'object_polarity' 的取值无效：" + polarity + "（允许 bright/dark）");
        options.brightObject = polarity == "bright";
        options.widthMin     = requireFloatInRange(inputs, p::in::WIDTH_MIN, 0.0, 1.0e9);
        options.widthMax     = requireFloatInRange(inputs, p::in::WIDTH_MAX, 0.0, 1.0e9);
        if (options.widthMax < options.widthMin)
            throw visionrt::ContractError("width_max 必须不小于 width_min");
        const double unitsPerPixel = requireFloatInRange(inputs, p::in::UNITS_PER_PIXEL, 1.0e-9, 1.0e9);
        const auto   unit          = requireString(inputs, p::in::UNIT);
        if (unit.empty())
            throw visionrt::ContractError("unit 不能为空");

        const auto alignments = geometry::alignmentsFromValue(requireInput(inputs, p::in::ALIGNMENT), images.size());
        const auto records    = roi::groups(requireInput(inputs, p::in::ROIS), images.size(), "rois");
        if (!roi::anyRecord(records))
            throw visionrt::ContractError("行间距 ROI 未配置");

        visionrt::NestedVector<bool, 2>        validRoi;
        visionrt::NestedVector<std::string, 2> reasons;
        visionrt::NestedVector<std::int64_t, 2> pairCounts, acceptedCounts, unpairedCounts;
        visionrt::NestedVector<bool, 3>        accepted;
        visionrt::NestedVector<double, 3>      widthPixels, gapPixels, pitchPixels;
        visionrt::NestedVector<double, 3>      widthValues, gapValues, pitchValues;
        visionrt::NestedVector<double, 4>      pairPoints;
        std::vector<std::vector<roi::Record>>  segments;

        for (std::size_t image = 0; image < images.size(); ++image)
        {
            const auto &source = images[image];
            requireSupportedImage(source, "行间距测量");
            const auto  gray     = grayscale(source);
            const auto &matrix   = alignments[image];
            const auto  inverse  = matrix.inverse();
            if (!inverse)
                throw visionrt::ContractError("alignment 线性部分必须可逆");
            const auto scale = matrix.scale();
            validRoi.emplace_back();
            reasons.emplace_back();
            pairCounts.emplace_back();
            acceptedCounts.emplace_back();
            unpairedCounts.emplace_back();
            accepted.emplace_back();
            widthPixels.emplace_back();
            gapPixels.emplace_back();
            pitchPixels.emplace_back();
            widthValues.emplace_back();
            gapValues.emplace_back();
            pitchValues.emplace_back();
            pairPoints.emplace_back();
            std::vector<roi::Record> imageSegments;

            for (std::size_t index = 0; index < records[image].size(); ++index)
            {
                const auto &record = records[image][index];
                if (record.kind != visionrt::RoiKind::Rectangle || record.values.size() != 5)
                    throw visionrt::ContractError("行间距测量要求每个 ROI 为 kind=2 的旋转矩形");
                // Map the reference rectangle into current image coordinates; a
                // similarity transform keeps the rectangle a rectangle.
                if (!matrix.isSimilarity())
                    throw visionrt::ContractError("行间距测量仅接受保向相似变换的 alignment");
                double centerX = 0, centerY = 0;
                matrix.apply(record.values[0] + record.values[2] / 2.0, record.values[1] + record.values[3] / 2.0,
                             centerX, centerY);
                const double angle = record.values[4] + std::atan2(matrix.m[3], matrix.m[0]) * 180.0 / kPi;
                const auto   current
                    = roi::rectangle(centerX, centerY, record.values[2] * scale, record.values[3] * scale, angle);
                const auto measurement = measureRowRect(gray, current, options);

                validRoi.back().push_back(measurement.valid);
                reasons.back().push_back(measurement.valid ? std::string{} : measurement.reason);
                pairCounts.back().push_back(static_cast<std::int64_t>(measurement.pairs.size()));
                acceptedCounts.back().push_back(
                    static_cast<std::int64_t>(std::count(measurement.accepted.begin(), measurement.accepted.end(), true)));
                unpairedCounts.back().push_back(measurement.unpaired);
                accepted.back().push_back(measurement.accepted);
                widthPixels.back().push_back(measurement.widths);
                gapPixels.back().push_back(measurement.gaps);
                pitchPixels.back().push_back(measurement.pitches);

                std::vector<double> widthConverted, gapConverted, pitchConverted;
                for (double value : measurement.widths)
                    widthConverted.push_back(value * unitsPerPixel);
                for (double value : measurement.gaps)
                    gapConverted.push_back(value * unitsPerPixel);
                for (double value : measurement.pitches)
                    pitchConverted.push_back(value * unitsPerPixel);
                widthValues.back().push_back(std::move(widthConverted));
                gapValues.back().push_back(std::move(gapConverted));
                pitchValues.back().push_back(std::move(pitchConverted));

                const double radians = current.values[4] * kPi / 180.0;
                const auto   direction = cv::Point2d{std::cos(radians), std::sin(radians)};
                const auto   normal    = cv::Point2d{-std::sin(radians), std::cos(radians)};
                const auto   rectCenter
                    = cv::Point2d{current.values[0] + current.values[2] / 2.0,
                                  current.values[1] + current.values[3] / 2.0};
                std::vector<std::vector<double>> points;
                const double                     halfLength = std::max(1.0, std::min(current.values[3] / 2.0, 8.0));
                for (const auto &pair : measurement.pairs)
                {
                    const auto first  = cv::Point2d{rectCenter.x + direction.x * pair.first,
                                                    rectCenter.y + direction.y * pair.first};
                    const auto second = cv::Point2d{rectCenter.x + direction.x * pair.second,
                                                    rectCenter.y + direction.y * pair.second};
                    points.push_back({first.x, first.y, second.x, second.y});
                    for (const auto &edge : {first, second})
                        imageSegments.push_back(roi::segmentValue(edge.x - normal.x * halfLength,
                                                                  edge.y - normal.y * halfLength,
                                                                  edge.x + normal.x * halfLength,
                                                                  edge.y + normal.y * halfLength));
                }
                pairPoints.back().push_back(std::move(points));
            }
            segments.push_back(std::move(imageSegments));
        }

        visionrt::Values outputs;
        outputs.emplace(p::out::IMAGES,
                        visionrt::DataRef<visionrt::Image>(
                            visionrt::Image(images), {std::get<visionrt::DataRef<visionrt::Image>>(
                                                          requireInput(inputs, p::in::IMAGES))
                                                          .owner()}));
        outputs.emplace(p::out::VALID_ROI,
                        visionrt::Array(visionrt::DataRef<visionrt::NestedVector<bool, 2>>(std::move(validRoi))));
        outputs.emplace(p::out::REASONS,
                        visionrt::Array(visionrt::DataRef<visionrt::NestedVector<std::string, 2>>(std::move(reasons))));
        outputs.emplace(p::out::PAIR_COUNT,
                        visionrt::Array(visionrt::DataRef<visionrt::NestedVector<std::int64_t, 2>>(std::move(pairCounts))));
        outputs.emplace(p::out::ACCEPTED_COUNT,
                        visionrt::Array(visionrt::DataRef<visionrt::NestedVector<std::int64_t, 2>>(
                            std::move(acceptedCounts))));
        outputs.emplace(p::out::UNPAIRED_COUNT,
                        visionrt::Array(visionrt::DataRef<visionrt::NestedVector<std::int64_t, 2>>(
                            std::move(unpairedCounts))));
        outputs.emplace(p::out::ACCEPTED,
                        visionrt::Array(visionrt::DataRef<visionrt::NestedVector<bool, 3>>(std::move(accepted))));
        outputs.emplace(p::out::WIDTH_PX,
                        visionrt::Array(visionrt::DataRef<visionrt::NestedVector<double, 3>>(std::move(widthPixels))));
        outputs.emplace(p::out::GAP_PX,
                        visionrt::Array(visionrt::DataRef<visionrt::NestedVector<double, 3>>(std::move(gapPixels))));
        outputs.emplace(p::out::PITCH_PX,
                        visionrt::Array(visionrt::DataRef<visionrt::NestedVector<double, 3>>(std::move(pitchPixels))));
        outputs.emplace(p::out::WIDTH,
                        visionrt::Array(visionrt::DataRef<visionrt::NestedVector<double, 3>>(std::move(widthValues))));
        outputs.emplace(p::out::GAP,
                        visionrt::Array(visionrt::DataRef<visionrt::NestedVector<double, 3>>(std::move(gapValues))));
        outputs.emplace(p::out::PITCH,
                        visionrt::Array(visionrt::DataRef<visionrt::NestedVector<double, 3>>(std::move(pitchValues))));
        outputs.emplace(p::out::PAIR_POINTS,
                        visionrt::Array(visionrt::DataRef<visionrt::NestedVector<double, 4>>(std::move(pairPoints))));
        outputs.emplace(p::out::OVERLAY_SEGMENTS, roi::roiValue(std::move(segments)));
        outputs.emplace(p::out::UNIT, unit);
        return visionrt::toolResult(std::move(outputs), visionrt::executionSuccess());
    }
    catch (const visionrt::ContractError &error)
    {
        return visionrt::toolResult(missingOutputs(), {1, error.what()});
    }
    catch (const std::exception &error)
    {
        return visionrt::toolResult(missingOutputs(), {1, "行间距测量失败", error.what()});
    }
    catch (...)
    {
        return visionrt::toolResult(missingOutputs(), {-1, "行间距测量失败：未知异常"});
    }
}

visionrt::ToolStatus RowSpacingTool::release()
{
    return {};
}
} // namespace visiontools::metrology
