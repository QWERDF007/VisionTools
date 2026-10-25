#include "CircleTool.h"

#include "CircleMeasureKernel.h"

#include "common/GeometryTransform.h"
#include "common/ImageSupport.h"
#include "common/InputValues.h"
#include "common/RoiSupport.h"

#include <algorithm>
#include <string>
#include <vector>

namespace visiontools::metrology {
namespace {
namespace p = ports::measure_circle;

visionrt::Values missingOutputs()
{
    visionrt::Values outputs;
    for (const auto *id : {p::out::IMAGES, p::out::VALID, p::out::REASONS, p::out::VALID_COUNT,
                           p::out::SOURCE_ROI_INDICES, p::out::CIRCLES, p::out::CENTERS, p::out::RADIUS_PX,
                           p::out::RADIUS, p::out::RMS, p::out::COMPLETENESS, p::out::USED_EDGES, p::out::UNIT})
        outputs.emplace(id, visionrt::Missing{});
    return outputs;
}

EdgePolarity parsePolarity(const std::string &value)
{
    if (value == "any")
        return EdgePolarity::Any;
    if (value == "dark_to_light")
        return EdgePolarity::DarkToLight;
    if (value == "light_to_dark")
        return EdgePolarity::LightToDark;
    throw visionrt::ContractError("输入端口 'polarity' 的取值无效：" + value
                                  + "（允许 any/dark_to_light/light_to_dark）");
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
        return {1, "圆测量不接受初始化参数"};
    return {};
}

visionrt::ToolResult CircleTool::execute(const visionrt::Values &inputs)
{
    try
    {
        const auto &images = requireImages(inputs, p::in::IMAGES);
        CircleOptions options;
        options.calipers         = requireInt32(inputs, p::in::NUM_CALIPERS, 12, 4096);
        options.searchHalfWidth  = requireFloatInRange(inputs, p::in::SEARCH_HALF_WIDTH, 0.5, 4096.0);
        options.averageWidth     = requireFloatInRange(inputs, p::in::AVERAGE_WIDTH, 1.0, 1024.0);
        options.sigma            = requireFloatInRange(inputs, p::in::SIGMA, 0.1, 20.0);
        options.minGradient      = requireFloatInRange(inputs, p::in::MIN_GRADIENT, 0.0, 255.0);
        options.polarity         = parsePolarity(requireString(inputs, p::in::POLARITY));
        options.minPoints        = requireInt32(inputs, p::in::MIN_POINTS, 3, 4096);
        options.minCompleteness  = requireFloatInRange(inputs, p::in::MIN_COMPLETENESS, 0.0, 1.0);
        options.minCoverageDeg   = requireFloatInRange(inputs, p::in::MIN_COVERAGE_DEG, 0.0, 360.0);
        options.radiusMin        = requireFloatInRange(inputs, p::in::RADIUS_MIN, 0.0, 1.0e9);
        options.radiusMax        = requireFloatInRange(inputs, p::in::RADIUS_MAX, 0.0, 1.0e9);
        options.maxRms           = requireFloatInRange(inputs, p::in::MAX_RMS, 0.0, 1.0e9);
        const double unitsPerPixel = requireFloatInRange(inputs, p::in::UNITS_PER_PIXEL, 1.0e-9, 1.0e9);
        const auto   unit          = requireString(inputs, p::in::UNIT);
        if (unit.empty())
            throw visionrt::ContractError("unit 不能为空");
        if (options.radiusMax < options.radiusMin)
            throw visionrt::ContractError("radius_max 必须不小于 radius_min");
        if (options.minPoints > options.calipers)
            throw visionrt::ContractError("min_points 不能大于 num_calipers");
        if (!(options.searchHalfWidth > 3.0 * options.sigma))
            throw visionrt::ContractError("search_half_width 必须大于 3 倍 sigma，以保留平滑支持窗口");
        const double sampleCount = 2.0 * options.searchHalfWidth / 0.5 + 1.0;
        if (sampleCount * std::max(1.0, options.averageWidth) > 8.0e6)
            throw visionrt::ContractError("卡尺采样预算超限，请减小搜索宽度或平均宽度");

        const auto alignments = geometry::alignmentsFromValue(requireInput(inputs, p::in::ALIGNMENT), images.size());
        const auto records    = roi::groups(requireInput(inputs, p::in::ROIS), images.size(), "rois");
        if (!roi::anyRecord(records))
            throw visionrt::ContractError("圆 ROI 未配置");

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
                throw visionrt::ContractError("圆测量仅接受保向相似变换的 alignment（非均匀缩放或剪切会把圆变成椭圆）");
            const double scale = matrix.scale();
            valid.emplace_back();
            reasons.emplace_back();
            std::vector<roi::Record> circles;
            std::vector<roi::Record> usedEdges;
            for (std::size_t index = 0; index < records[image].size(); ++index)
            {
                const auto &record = records[image][index];
                if (record.kind != visionrt::RoiKind::Circle || record.values.size() != 3)
                    throw visionrt::ContractError("圆测量要求每个 ROI 为 kind=3 的圆");
                if (!(record.values[2] > 0.0))
                    throw visionrt::ContractError("圆 ROI 半径必须为正");
                double centerX = 0, centerY = 0;
                matrix.apply(record.values[0], record.values[1], centerX, centerY);
                const double radius = record.values[2] * scale;
                if (!(radius > options.searchHalfWidth))
                    throw visionrt::ContractError("圆 ROI 名义半径必须大于搜索半宽（第 "
                                                  + std::to_string(index + 1) + " 个 ROI）");
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
                            visionrt::Image(images), {std::get<visionrt::DataRef<visionrt::Image>>(
                                                          requireInput(inputs, p::in::IMAGES))
                                                          .owner()}));
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
        return {visionrt::executionSuccess(), std::move(outputs)};
    }
    catch (const visionrt::ContractError &error)
    {
        return {{1, error.what()}, missingOutputs()};
    }
    catch (const std::exception &error)
    {
        return {{1, "圆测量失败", error.what()}, missingOutputs()};
    }
    catch (...)
    {
        return {{-1, "圆测量失败：未知异常"}, missingOutputs()};
    }
}

visionrt::ToolStatus CircleTool::release()
{
    return {};
}
} // namespace visiontools::metrology
