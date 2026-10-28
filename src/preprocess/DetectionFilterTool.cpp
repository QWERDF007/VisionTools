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

DetectionFilterTool::DetectionFilterTool(const visionrt::ToolServices &services)
    : services_(services)
{
}

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

visionrt::ToolResult DetectionFilterTool::execute(const visionrt::Values &inputs)
{
    // An unselected upstream branch is not an error: no algorithm runs and nothing is written.
    if (const auto upstream = upstreamStatus(inputs); upstream != 0)
        return skippedByUpstream(upstream, missingOutputs());
    Readings   readings(inputs);
    const auto images       = readings.image(p::in::IMAGES);
    const auto roisRef      = readings.roi(p::in::ROIS);
    const auto scoresRef    = readings.array<visionrt::NestedVector<float, 2>>(p::in::SCORES);
    const auto classesRef   = readings.array<visionrt::NestedVector<std::int64_t, 2>>(p::in::CLASSES);
    const auto namesRef     = readings.array<visionrt::NestedVector<std::string, 2>>(p::in::CLASS_NAMES);
    const auto allowedRef   = readings.array<std::vector<std::int64_t>>(p::in::ALLOWED_CLASSES);
    const auto minimumScore = readings.real(p::in::MIN_SCORE);
    const auto minimumArea  = readings.real(p::in::MIN_AREA);
    if (readings.failed())
        return visionrt::toolResult(missingOutputs(),
                                    readings.status(CommonToolError::InputMissing, CommonToolError::InputTypeMismatch));
    if (!std::isfinite(minimumScore) || minimumScore < 0.0 || minimumScore > 1.0)
        return failure(missingOutputs(), CommonToolError::InputOutOfRange,
                       std::string(p::in::MIN_SCORE) + " 必须在 [0, 1] 内，实际值：" + std::to_string(minimumScore));
    if (!std::isfinite(minimumArea) || minimumArea < 0.0)
        return failure(missingOutputs(), CommonToolError::InputOutOfRange,
                       std::string(p::in::MIN_AREA) + " 必须为非负有限数值，实际值：" + std::to_string(minimumArea));

    const auto &rois    = roisRef.get();
    const auto &scores  = scoresRef.get();
    const auto &classes = classesRef.get();
    const auto &names   = namesRef.get();
    const auto &allowed = allowedRef.get();
    const auto batchSize = images.get().size();
    if (rois.size() != batchSize || scores.size() != batchSize || classes.size() != batchSize
        || names.size() != batchSize)
        return failure(missingOutputs(), CommonToolError::InputOutOfRange,
                       "检测批次长度不一致：images=" + std::to_string(batchSize) + "，rois=" + std::to_string(rois.size())
                           + "，scores=" + std::to_string(scores.size()) + "，classes=" + std::to_string(classes.size())
                           + "，class_names=" + std::to_string(names.size()));
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
            const auto &imageScores  = scores[batch];
            const auto &imageClasses = classes[batch];
            const auto &imageNames   = names[batch];
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
                if (visionrt::roiKind(imageRois[detection]) != visionrt::RoiKind::Rectangle)
                    fail(CommonToolError::InputTypeMismatch,
                         "批次索引 " + std::to_string(batch) + " 的检测索引 " + std::to_string(detection)
                             + " 的 ROI 必须是矩形");
                const auto  &encoded = imageRois[detection];
                const double area    = encoded[3] * encoded[4];
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
        outputs.emplace(p::out::SCORES, visionrt::Array(std::move(filteredScores)));
        outputs.emplace(p::out::CLASSES, visionrt::Array(std::move(filteredClasses)));
        outputs.emplace(p::out::CLASS_NAMES, visionrt::Array(std::move(filteredNames)));
        outputs.emplace(p::out::COUNTS, visionrt::Array(std::move(counts)));
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
