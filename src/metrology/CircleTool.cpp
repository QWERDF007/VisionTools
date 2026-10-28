#include "CircleTool.h"

#include "CircleMeasureKernel.h"

#include "common/GeometryTransform.h"
#include "common/ToolOutputs.h"
#include "common/UpstreamStatus.h"
#include "common/ImageSupport.h"
#include "common/InputValues.h"
#include "common/RoiSupport.h"
#include "errors/MetrologyErrors.h"
#include "common/errors/ToolError.h"

#include <algorithm>
#include <string>
#include <vector>

namespace visiontools::metrology {
namespace {
namespace p = ports::measure_circle;

visionrt::Values missingOutputs()
{
    return missingPorts({p::out::IMAGES, p::out::VALID, p::out::REASONS, p::out::VALID_COUNT,
                         p::out::SOURCE_ROI_INDICES, p::out::CIRCLES, p::out::CENTERS, p::out::RADIUS_PX,
                         p::out::RADIUS, p::out::RMS, p::out::COMPLETENESS, p::out::USED_EDGES, p::out::UNIT});
}

EdgePolarity parsePolarity(const std::string &value)
{
    if (value == "any")
        return EdgePolarity::Any;
    if (value == "dark_to_light")
        return EdgePolarity::DarkToLight;
    if (value == "light_to_dark")
        return EdgePolarity::LightToDark;
    fail(CommonToolError::InputOutOfRange,
         "polarity 只接受 any/dark_to_light/light_to_dark，实际值：" + value);
}

// One place for "this parameter is usable": the declaration also carries the range, but a direct
// consumer can call the tool without one.
void inRange(double value, double minimum, double maximum, std::string_view id)
{
    if (!std::isfinite(value) || value < minimum || value > maximum)
        fail(CommonToolError::InputOutOfRange,
             std::string(id) + " 必须在 [" + std::to_string(minimum) + ", " + std::to_string(maximum)
                 + "] 内，实际值：" + std::to_string(value));
}

template<class T>
visionrt::Array array1(std::vector<T> values)
{
    return visionrt::Array(visionrt::DataRef<std::vector<T>>(std::move(values)));
}

template<class T>
visionrt::Array array2(std::vector<std::vector<T>> values)
{
    return visionrt::Array(visionrt::DataRef<std::vector<std::vector<T>>>(std::move(values)));
}

template<class T>
visionrt::Array array3(std::vector<std::vector<std::vector<T>>> values)
{
    return visionrt::Array(visionrt::DataRef<std::vector<std::vector<std::vector<T>>>>(std::move(values)));
}
} // namespace

visionrt::ToolStatus CircleTool::initialize(const visionrt::Values &parameters)
{
    if (!parameters.empty())
        return {visionrt::kContractViolationStatus, "圆测量不接受初始化参数"};
    return {};
}

visionrt::ToolResult CircleTool::execute(const visionrt::Values &inputs)
{
    // An unselected upstream branch is not an error: no algorithm runs and nothing is written.
    if (const auto upstream = upstreamStatus(inputs); upstream != 0)
        return skippedByUpstream(upstream, missingOutputs());
    Readings   readings(inputs);
    const auto imagesRef = readings.image(p::in::IMAGES);
    CircleOptions options;
    options.calipers        = readings.integer(p::in::NUM_CALIPERS);
    options.searchHalfWidth = readings.real(p::in::SEARCH_HALF_WIDTH);
    options.averageWidth    = readings.real(p::in::AVERAGE_WIDTH);
    options.sigma           = readings.real(p::in::SIGMA);
    options.minGradient     = readings.real(p::in::MIN_GRADIENT);
    const auto polarityName = readings.text(p::in::POLARITY);
    options.minPoints       = readings.integer(p::in::MIN_POINTS);
    options.minCompleteness = readings.real(p::in::MIN_COMPLETENESS);
    options.minCoverageDeg  = readings.real(p::in::MIN_COVERAGE_DEG);
    options.radiusMin       = readings.real(p::in::RADIUS_MIN);
    options.radiusMax       = readings.real(p::in::RADIUS_MAX);
    options.maxRms          = readings.real(p::in::MAX_RMS);
    const auto unitsPerPixel = readings.real(p::in::UNITS_PER_PIXEL);
    const auto unit          = readings.text(p::in::UNIT);
    const auto alignmentValue = readings.find(p::in::ALIGNMENT);
    const auto roiValue       = readings.find(p::in::ROIS);
    if (readings.failed())
        return visionrt::toolResult(missingOutputs(),
                                    readings.status(CommonToolError::InputMissing, CommonToolError::InputTypeMismatch));
    try
    {
        const auto &images = imagesRef.get();
        inRange(options.calipers, 12, 4096, p::in::NUM_CALIPERS);
        inRange(options.searchHalfWidth, 0.5, 4096.0, p::in::SEARCH_HALF_WIDTH);
        inRange(options.averageWidth, 1.0, 1024.0, p::in::AVERAGE_WIDTH);
        inRange(options.sigma, 0.1, 20.0, p::in::SIGMA);
        inRange(options.minGradient, 0.0, 255.0, p::in::MIN_GRADIENT);
        inRange(options.minPoints, 3, 4096, p::in::MIN_POINTS);
        inRange(options.minCompleteness, 0.0, 1.0, p::in::MIN_COMPLETENESS);
        inRange(options.minCoverageDeg, 0.0, 360.0, p::in::MIN_COVERAGE_DEG);
        inRange(options.radiusMin, 0.0, 1.0e9, p::in::RADIUS_MIN);
        inRange(options.radiusMax, 0.0, 1.0e9, p::in::RADIUS_MAX);
        inRange(options.maxRms, 0.0, 1.0e9, p::in::MAX_RMS);
        inRange(unitsPerPixel, 1.0e-9, 1.0e9, p::in::UNITS_PER_PIXEL);
        options.polarity = parsePolarity(polarityName);
        if (unit.empty())
            fail(CommonToolError::InputOutOfRange, std::string(p::in::UNIT) + " 不能为空");
        if (options.radiusMax < options.radiusMin)
            fail(CommonToolError::InputOutOfRange, "radius_max 必须不小于 radius_min");
        if (options.minPoints > options.calipers)
            fail(CommonToolError::InputOutOfRange, "min_points 不能大于 num_calipers");
        if (!(options.searchHalfWidth > 3.0 * options.sigma))
            fail(CommonToolError::InputOutOfRange, "search_half_width 必须大于 3 倍 sigma，以保留平滑支持窗口");
        const double sampleCount = 2.0 * options.searchHalfWidth / 0.5 + 1.0;
        if (sampleCount * std::max(1.0, options.averageWidth) > 8.0e6)
            fail(CommonToolError::InputOutOfRange, "卡尺采样预算超限，请减小搜索宽度或平均宽度");
        if (!alignmentValue || !roiValue)
            fail(CommonToolError::InputMissing, !alignmentValue ? p::in::ALIGNMENT : p::in::ROIS);

        const auto alignments = geometry::alignmentsFromValue(*alignmentValue, images.size());
        const auto records    = roi::groups(*roiValue, images.size(), "rois");
        if (!roi::anyRecord(records))
            fail(MetrologyError::RoiNotConfigured);

        struct Findings
        {
            std::vector<CircleMeasurement> measurements;
            std::vector<std::size_t>       sources;
        };
        std::vector<Findings>                   findings(images.size());
        visionrt::NestedVector<bool, 2>         valid;
        visionrt::NestedVector<std::string, 2>  reasons;
        visionrt::NestedVector<std::int64_t, 1> validCounts;
        visionrt::NestedVector<std::int64_t, 2> sourceIndices;
        visionrt::NestedVector<double, 3>       centers;
        visionrt::NestedVector<double, 2>       radiusPixels, radii, rmsValues, completenessValues;
        std::vector<std::vector<roi::Record>>   circleGroups;
        std::vector<std::vector<roi::Record>>   edgeGroups;

        for (std::size_t image = 0; image < images.size(); ++image)
        {
            const auto &source = images[image];
            requireSupportedImage(source, "圆测量");
            const auto  gray   = grayscale(source);
            const auto &matrix = alignments[image];
            if (!matrix.isSimilarity())
                fail(MetrologyError::RoiKindMismatch, "alignment 必须是保向相似变换（非均匀缩放或剪切会把圆变成椭圆）");
            const double scale = matrix.scale();
            valid.emplace_back();
            reasons.emplace_back();
            std::vector<roi::Record> circles;
            std::vector<roi::Record> usedEdges;
            for (std::size_t index = 0; index < records[image].size(); ++index)
            {
                const auto &record = records[image][index];
                if (record.kind != visionrt::RoiKind::Circle || record.values.size() != 3)
                    fail(MetrologyError::RoiKindMismatch, "第 " + std::to_string(index + 1) + " 个 ROI 必须是 kind=3 的圆");
                if (!(record.values[2] > 0.0))
                    fail(MetrologyError::RoiKindMismatch, "第 " + std::to_string(index + 1) + " 个圆 ROI 半径必须为正");
                double centerX = 0, centerY = 0;
                matrix.apply(record.values[0], record.values[1], centerX, centerY);
                const double radius = record.values[2] * scale;
                if (!(radius > options.searchHalfWidth))
                    fail(CommonToolError::InputOutOfRange,
                         "第 " + std::to_string(index + 1) + " 个圆 ROI 名义半径必须大于搜索半宽");
                if (radius < options.radiusMin || radius > options.radiusMax)
                {
                    valid.back().push_back(false);
                    reasons.back().push_back("名义半径超出允许范围");
                    continue;
                }
                const auto measurement = measureCircleKernel(gray, {centerX, centerY}, radius, options);
                valid.back().push_back(measurement.valid);
                reasons.back().push_back(measurement.valid ? std::string{} : measurement.reason);
                if (!measurement.valid)
                    continue;
                circles.push_back(roi::circle(measurement.centerX, measurement.centerY, measurement.radius));
                usedEdges.push_back(roi::pointSet(measurement.edges));
                findings[image].measurements.push_back(measurement);
                findings[image].sources.push_back(index);
            }
            validCounts.push_back(static_cast<std::int64_t>(findings[image].measurements.size()));
            sourceIndices.emplace_back();
            centers.emplace_back();
            radiusPixels.emplace_back();
            radii.emplace_back();
            rmsValues.emplace_back();
            completenessValues.emplace_back();
            for (std::size_t k = 0; k < findings[image].measurements.size(); ++k)
            {
                const auto &measurement = findings[image].measurements[k];
                sourceIndices.back().push_back(static_cast<std::int64_t>(findings[image].sources[k]));
                centers.back().push_back({measurement.centerX, measurement.centerY});
                radiusPixels.back().push_back(measurement.radius);
                radii.back().push_back(measurement.radius * unitsPerPixel);
                rmsValues.back().push_back(measurement.rms);
                completenessValues.back().push_back(measurement.completeness);
            }
            circleGroups.push_back(std::move(circles));
            edgeGroups.push_back(std::move(usedEdges));
        }

        visionrt::Values outputs;
        outputs.emplace(p::out::IMAGES,
                        visionrt::DataRef<visionrt::Image>(
                            visionrt::Image(images), {imagesRef.owner()}));
        outputs.emplace(p::out::VALID, array2(std::move(valid)));
        outputs.emplace(p::out::REASONS, array2(std::move(reasons)));
        outputs.emplace(p::out::VALID_COUNT, array1(std::move(validCounts)));
        outputs.emplace(p::out::SOURCE_ROI_INDICES, array2(std::move(sourceIndices)));
        outputs.emplace(p::out::CIRCLES, roi::roiValue(std::move(circleGroups)));
        outputs.emplace(p::out::CENTERS, array3(std::move(centers)));
        outputs.emplace(p::out::RADIUS_PX, array2(std::move(radiusPixels)));
        outputs.emplace(p::out::RADIUS, array2(std::move(radii)));
        outputs.emplace(p::out::RMS, array2(std::move(rmsValues)));
        outputs.emplace(p::out::COMPLETENESS, array2(std::move(completenessValues)));
        outputs.emplace(p::out::USED_EDGES, roi::roiValue(std::move(edgeGroups)));
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

visionrt::ToolStatus CircleTool::release()
{
    return {};
}
} // namespace visiontools::metrology
