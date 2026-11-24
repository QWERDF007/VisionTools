#include "AlignStraightBordersTool.h"

#include "StraightEdgeLocator.h"
#include "StraightEdgeModel.h"
#include "common/EdgeProfile.h"
#include "common/InputValues.h"
#include "common/RoiSupport.h"
#include "common/ToolOutputs.h"
#include "common/errors/ToolError.h"
#include "errors/RecognitionErrors.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace visiontools::recognition {
namespace {
namespace p   = ports::align_straight_borders;

visionrt::Values missingOutputs()
{
    return missingPorts({p::out::FOUND_SEGMENTS, p::out::MODEL_PATH});
}

} // namespace

visionrt::ToolResult AlignStraightBordersTool::train(const visionrt::TrainingSet &training, const visionrt::Values &inputs)
{
    Readings readings(inputs, p::TOOL_ID);
    const auto contrast = readings.real(p::train::CONTRAST);
    const auto width = readings.integer(p::train::EDGE_WIDTH);
    const auto polarity = readings.text(p::train::POLARITY);
    const auto completeness = readings.real(p::train::MIN_COMPLETENESS);
    const auto modelPath = readings.text(p::train::MODEL_PATH);
    if (readings.failed())
        return visionrt::toolResult(missingOutputs(), readings.status(CommonToolError::InputMissing, CommonToolError::InputTypeMismatch));
    try
    {
        const auto options = EdgeSearchOptions::fromParameters(contrast, width, polarity, completeness);
        visionrt::validateTrainingSet(training);
        if (training.records.empty())
            fail(RecognitionError::ReferenceSegmentMissing, "训练集");
        if (training.records.size() > 1)
            fail(RecognitionError::ReferenceSegmentMissing, "训练集",
                 "一次只能从一张参考图画两条边，实际 " + std::to_string(training.records.size()) + " 条记录");
        const auto &record = training.records.begin()->second;

        const auto field = record.fields.find(p::train::SEARCH_REGIONS);
        const auto *rois = field == record.fields.end() ? nullptr : std::get_if<visionrt::DataRef<visionrt::ROI>>(&field->second);
        if (!rois || rois->get().size() != 1 || rois->get().front().size() != 2)
            fail(RecognitionError::StraightEdgeRegionsInvalid, "参考边", "一张参考图上只能有两条搜索矩形");

        const auto image = training.images.find(record.imageId);
        if (image == training.images.end())
            fail(RecognitionError::ReferenceSegmentMissing, "参考图");
        const cv::Mat gray = edge::grayscale(image->second.pixels);

        std::vector<SearchRectangle> bands;
        bands.reserve(2);
        for (std::size_t index = 0; index < rois->get().front().size(); ++index)
            bands.push_back(searchRectangle(rois->get().front()[index], index));

        const auto model = StraightEdgeModel::train(gray, bands, options);
        const auto bytes = model.encode();

        std::vector<roi::Record> segments;
        segments.reserve(model.segments().size());
        for (const auto &segment : model.segments())
            segments.push_back(roi::segmentValue(segment.from.x, segment.from.y, segment.to.x, segment.to.y));

        visionrt::Values outputs;
        outputs.emplace(p::out::MODEL_PATH, modelPath);
        outputs.emplace(p::out::FOUND_SEGMENTS, roi::roiValue({std::move(segments)}));
        auto result = visionrt::toolResult(std::move(outputs), {0, "训练成功"});
        result.artifacts.emplace("edge-model", visionrt::ModelArtifact{bytes});
        return result;
    }
    catch (const ToolFailure &error)
    {
        return failureResult(missingOutputs(), error);
    }
    catch (const std::exception &error)
    {
        return failure(missingOutputs(), CommonToolError::OperatorFailed, "直边模型训练失败", error.what());
    }
}

} // namespace visiontools::recognition
