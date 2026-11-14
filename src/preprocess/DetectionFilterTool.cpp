#include "DetectionFilterTool.h"

#include "common/InputValues.h"
#include "common/ToolOutputs.h"
#include "common/UpstreamStatus.h"
#include "common/errors/ToolError.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace visiontools::preprocess {
namespace {

namespace p = ports::detection_filter;


} // namespace

visionrt::ToolStatus DetectionFilterTool::initialize(const visionrt::Values &parameters)
{
    if (!parameters.empty())
    {
        return {visionrt::kContractViolationStatus, "缺陷过滤不接受初始化参数"};
    }
    return {};
}


namespace {
visionrt::Values missingOutputs()
{
    return missingPorts({p::out::IMAGES, p::out::ROIS, p::out::SCORES, p::out::CLASSES, p::out::CLASS_NAMES,
                         p::out::COUNTS});
}
} // namespace

namespace {
std::size_t rowCount(const visionrt::Array &array)
{
    if (array.layout() == visionrt::ArrayLayout::Dense)
        return array.shape().empty() ? 0 : static_cast<std::size_t>(array.shape()[0]);
    return array.offsets().empty() ? 0 : static_cast<std::size_t>(array.offsets()[0].size() - 1);
}

template<class T>
std::span<const T> rowOf(const visionrt::Array &array, std::size_t row)
{
    if (array.layout() == visionrt::ArrayLayout::Dense)
    {
        const auto columns = array.shape().size() > 1 ? static_cast<std::size_t>(array.shape()[1]) : 0;
        return array.elements<T>().subspan(row * columns, columns);
    }
    const auto &bounds = array.offsets()[0];
    return array.elements<T>().subspan(static_cast<std::size_t>(bounds[row]),
                                       static_cast<std::size_t>(bounds[row + 1] - bounds[row]));
}
} // namespace

visionrt::ToolResult DetectionFilterTool::execute(const visionrt::Values &inputs)
{
    // An unselected upstream branch is not an error: no algorithm runs and nothing is written.
    if (const auto upstream = upstreamStatus(inputs); upstream != 0)
        return skippedByUpstream(upstream, missingOutputs());
    Readings   readings(inputs);
    const auto images       = readings.image(p::in::IMAGES);
    const auto roisRef      = readings.roi(p::in::ROIS);
    const auto scoresRef    = readings.array(p::in::SCORES);
    const auto classesRef   = readings.array(p::in::CLASSES);
    const auto namesRef     = readings.array(p::in::CLASS_NAMES);
    const auto allowedRef   = readings.array(p::in::ALLOWED_CLASSES);
    const auto minimumScore = readings.real(p::in::MIN_SCORE);
    const auto minimumArea  = readings.real(p::in::MIN_AREA);
    if (readings.failed())
        return visionrt::toolResult(missingOutputs(),
                                    readings.status(CommonToolError::InputMissing, CommonToolError::InputTypeMismatch));
    if (!std::isfinite(minimumScore) || minimumScore < 0.0 || minimumScore > 1.0)
        return failure(missingOutputs(), CommonToolError::InputOutOfRange,
                       visionrt::portLabel(p::in::MIN_SCORE) + " 必须在 [0, 1] 内，实际值：" + std::to_string(minimumScore));
    if (!std::isfinite(minimumArea) || minimumArea < 0.0)
        return failure(missingOutputs(), CommonToolError::InputOutOfRange,
                       visionrt::portLabel(p::in::MIN_AREA) + " 必须为非负有限数值，实际值：" + std::to_string(minimumArea));

    const auto &rois      = roisRef.get();
    const auto &scores    = scoresRef;
    const auto &classes   = classesRef;
    const auto &names     = namesRef;
    const auto  allowed   = allowedRef.elements<std::int64_t>();
    const auto  batchSize = images.get().size();
    if (rois.size() != batchSize || rowCount(scores) != batchSize || rowCount(classes) != batchSize
        || rowCount(names) != batchSize)
        return failure(missingOutputs(), CommonToolError::InputOutOfRange,
                       "检测批次长度不一致：images=" + std::to_string(batchSize) + "，rois=" + std::to_string(rois.size())
                           + "，scores=" + std::to_string(rowCount(scores)) + "，classes=" + std::to_string(rowCount(classes))
                           + "，class_names=" + std::to_string(rowCount(names)));
    try
    {

        visionrt::ROI                           filteredRois;
        visionrt::NestedVector<float, 2>        filteredScores;
        visionrt::NestedVector<std::int64_t, 2> filteredClasses;
        visionrt::NestedVector<std::string, 2>  filteredNames;
        std::vector<std::int64_t>               counts;
        filteredRois.reserve(batchSize);
        filteredScores.reserve(batchSize);
        filteredClasses.reserve(batchSize);
        filteredNames.reserve(batchSize);
        counts.reserve(batchSize);

        for (std::size_t batch = 0; batch < batchSize; ++batch)
        {
            const auto &imageRois    = rois[batch];
            const auto  imageScores  = rowOf<float>(scores, batch);
            const auto  imageClasses = rowOf<std::int64_t>(classes, batch);
            const auto  imageNames   = rowOf<std::string>(names, batch);
            if (imageScores.size() != imageRois.size() || imageClasses.size() != imageRois.size()
                || imageNames.size() != imageRois.size())
                fail(CommonToolError::InputOutOfRange,
                     "批次索引 " + std::to_string(batch) + " 的检测字段长度不一致：rois="
                         + std::to_string(imageRois.size()) + "，scores=" + std::to_string(imageScores.size())
                         + "，classes=" + std::to_string(imageClasses.size())
                         + "，class_names=" + std::to_string(imageNames.size()));

            auto &keptRois    = filteredRois.emplace_back();
            auto &keptScores  = filteredScores.emplace_back();
            auto &keptClasses = filteredClasses.emplace_back();
            auto &keptNames   = filteredNames.emplace_back();
            keptRois.reserve(imageRois.size());
            keptScores.reserve(imageRois.size());
            keptClasses.reserve(imageRois.size());
            keptNames.reserve(imageRois.size());
            for (std::size_t detection = 0; detection < imageRois.size(); ++detection)
            {
                if (!std::isfinite(imageScores[detection]) || imageClasses[detection] < 0)
                    fail(CommonToolError::InputOutOfRange,
                         "批次索引 " + std::to_string(batch) + " 的检测索引 " + std::to_string(detection)
                             + " 的得分或类别无效");
                visionrt::RectangleView rectangle;
                try
                {
                    rectangle = visionrt::RegionView(imageRois[detection]).rectangle();
                }
                catch (const visionrt::ContractError &)
                {
                    fail(CommonToolError::InputTypeMismatch,
                         "批次索引 " + std::to_string(batch) + " 的检测索引 " + std::to_string(detection)
                             + " 的 ROI 必须是矩形");
                }
                const double area = rectangle.width() * rectangle.height();
                const bool   classAllowed
                    = allowed.empty()
                   || std::find(allowed.begin(), allowed.end(), imageClasses[detection]) != allowed.end();
                if (imageScores[detection] >= minimumScore && area >= minimumArea && classAllowed)
                {
                    keptRois.push_back(imageRois[detection]);
                    keptScores.push_back(imageScores[detection]);
                    keptClasses.push_back(imageClasses[detection]);
                    keptNames.push_back(imageNames[detection]);
                }
            }
            counts.push_back(static_cast<std::int64_t>(keptRois.size()));
        }

        visionrt::Values outputs;
        outputs.emplace(p::out::IMAGES, images);
        outputs.emplace(p::out::ROIS, visionrt::DataRef<visionrt::ROI>(std::move(filteredRois)));
        outputs.emplace(p::out::SCORES, visionrt::Array::fromNested(std::move(filteredScores)));
        outputs.emplace(p::out::CLASSES, visionrt::Array::fromNested(std::move(filteredClasses)));
        outputs.emplace(p::out::CLASS_NAMES, visionrt::Array::fromNested(std::move(filteredNames)));
        outputs.emplace(p::out::COUNTS, visionrt::Array::fromNested(std::move(counts)));
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
        return failure(missingOutputs(), CommonToolError::OperatorFailed, "缺陷过滤失败", error.what());
    }
    catch (...)
    {
        return failure(missingOutputs(), CommonToolError::OperatorFailed, "缺陷过滤失败：未知异常");
    }
}

visionrt::ToolStatus DetectionFilterTool::release()
{
    return {};
}

} // namespace visiontools::preprocess
