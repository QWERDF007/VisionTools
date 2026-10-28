#include "RowSpacingTool.h"

#include "common/InputValues.h"
#include "errors/MetrologyErrors.h"
#include "common/errors/ToolError.h"

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
        return {visionrt::kContractViolationStatus, "行间距测量不接受初始化参数"};
    return {};
}

visionrt::ToolResult RowSpacingTool::execute(const visionrt::Values &inputs)
{
    // An unselected upstream branch is not an error: no algorithm runs and nothing is written.
    if (const auto upstream = upstreamStatus(inputs); upstream != 0)
        return skippedByUpstream(upstream, missingOutputs());
    Readings   readings(inputs);
    const auto imagesRef = readings.image(p::in::IMAGES);
    RowOptions options;
    options.sigma        = readings.real(p::in::SIGMA);
    options.minGradient  = readings.real(p::in::MIN_GRADIENT);
    const auto polarity  = readings.text(p::in::OBJECT_POLARITY);
    options.widthMin     = readings.real(p::in::WIDTH_MIN);
    options.widthMax     = readings.real(p::in::WIDTH_MAX);
    const auto unitsPerPixel = readings.real(p::in::UNITS_PER_PIXEL);
    const auto unit          = readings.text(p::in::UNIT);
    const auto alignmentValue = readings.find(p::in::ALIGNMENT);
    const auto roiValue       = readings.find(p::in::ROIS);
    if (readings.failed())
        return visionrt::toolResult(missingOutputs(),
                                    readings.status(CommonToolError::InputMissing, CommonToolError::InputTypeMismatch));
    const auto &images = imagesRef.get();
    if (!std::isfinite(options.sigma) || options.sigma < 0.1 || options.sigma > 20.0)
        return failure(missingOutputs(), CommonToolError::InputOutOfRange,
                       std::string(p::in::SIGMA) + " 必须在 [0.1, 20] 内，实际值：" + std::to_string(options.sigma));
    if (!std::isfinite(options.minGradient) || options.minGradient < 0.0 || options.minGradient > 255.0)
        return failure(missingOutputs(), CommonToolError::InputOutOfRange,
                       std::string(p::in::MIN_GRADIENT) + " 必须在 [0, 255] 内，实际值："
                           + std::to_string(options.minGradient));
    if (polarity != "bright" && polarity != "dark")
        return failure(missingOutputs(), CommonToolError::InputOutOfRange,
                       std::string(p::in::OBJECT_POLARITY) + " 只接受 bright/dark，实际值：" + polarity);
    if (!std::isfinite(options.widthMin) || options.widthMin < 0.0 || !std::isfinite(options.widthMax)
        || options.widthMax < 0.0 || options.widthMax < options.widthMin)
        return failure(missingOutputs(), CommonToolError::InputOutOfRange, "width 范围不合法：需满足 0 ≤ width_min ≤ width_max");
    if (!std::isfinite(unitsPerPixel) || unitsPerPixel < 1.0e-9 || unitsPerPixel > 1.0e9)
        return failure(missingOutputs(), CommonToolError::InputOutOfRange, std::string(p::in::UNITS_PER_PIXEL) + " 超出允许范围");
    if (unit.empty())
        return failure(missingOutputs(), CommonToolError::InputOutOfRange, std::string(p::in::UNIT) + " 不能为空");
    if (!alignmentValue || !roiValue)
        return failure(missingOutputs(), CommonToolError::InputMissing, !alignmentValue ? p::in::ALIGNMENT : p::in::ROIS);
    options.brightObject = polarity == "bright";
    try
    {
        const auto alignments = geometry::alignmentsFromValue(*alignmentValue, images.size());
        const auto records    = roi::groups(*roiValue, images.size(), "rois");
        if (!roi::anyRecord(records))
            fail(MetrologyError::RoiNotConfigured);

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
                fail(GeometryError::AlignmentInvalid, "线性部分不可逆");
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
                    fail(MetrologyError::RoiKindMismatch, "第 " + std::to_string(index + 1) + " 个 ROI 必须是 kind=2 的旋转矩形");
                // Map the reference rectangle into current image coordinates; a
                // similarity transform keeps the rectangle a rectangle.
                if (!matrix.isSimilarity())
                    fail(MetrologyError::RoiKindMismatch, "alignment 必须是保向相似变换");
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
                            visionrt::Image(images), {imagesRef.owner()}));
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
    catch (const visionrt::ContractError &)
    {
        throw;
    }
    catch (const ToolFailure &error)
    {
        return failureResult(missingOutputs(), error);
    }
    catch (const std::exception &error)
    {
        return failure(missingOutputs(), MetrologyError::MeasurementFailed, {}, error.what());
    }
    catch (...)
    {
        return failure(missingOutputs(), MetrologyError::MeasurementFailed, "未知异常");
    }
}

visionrt::ToolStatus RowSpacingTool::release()
{
    return {};
}
} // namespace visiontools::metrology
