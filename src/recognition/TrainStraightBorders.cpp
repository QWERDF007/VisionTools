#include "TrainStraightBorders.h"

#include "StraightEdgeLocator.h"
#include "StraightEdgeModel.h"
#include "common/EdgeProfile.h"
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
namespace p   = ports::train_straight_borders;
using FieldId = visionrt::TrainingFieldId;

visionrt::Values missingOutputs()
{
    return missingPorts({p::out::MODEL, p::out::SEGMENTS});
}

const std::int64_t *integerOf(const visionrt::Values &values, visionrt::PortId id)
{
    const auto found = values.find(id);
    return found == values.end() ? nullptr : std::get_if<std::int64_t>(&found->second);
}

const double *realOf(const visionrt::Values &values, visionrt::PortId id)
{
    const auto found = values.find(id);
    return found == values.end() ? nullptr : std::get_if<double>(&found->second);
}

const std::string *textOf(const visionrt::Values &values, visionrt::PortId id)
{
    const auto found = values.find(id);
    return found == values.end() ? nullptr : std::get_if<std::string>(&found->second);
}

edge::EdgePolarity parsePolarity(const std::string &value)
{
    if (value == "any")
        return edge::EdgePolarity::Any;
    if (value == "dark_to_light")
        return edge::EdgePolarity::DarkToLight;
    if (value == "light_to_dark")
        return edge::EdgePolarity::LightToDark;
    fail(CommonToolError::InputOutOfRange, "跃迁方向", "只能是 any、dark_to_light 或 light_to_dark");
}

// The drawn band is [kind, left, top, width, height, angle]; the locator works with the centre and
// the long side, so the conversion happens once here for every rectangle.
SearchRectangle toBand(const visionrt::EncodedRegion &record, std::size_t index)
{
    const auto view = visionrt::RegionView(record);
    if (view.kind() != visionrt::RoiKind::Rectangle)
        fail(RecognitionError::StraightEdgeRegionsInvalid, "第 " + std::to_string(index + 1) + " 条参考边",
             "必须是搜索矩形");
    const auto rectangle = view.rectangle();
    if (!(rectangle.width() > 1.0) || !(rectangle.height() > 0.0))
        fail(RecognitionError::StraightEdgeRegionsInvalid, "第 " + std::to_string(index + 1) + " 条参考边",
             "矩形的宽度或高度不合法");
    SearchRectangle band;
    band.width  = std::max(rectangle.width(), rectangle.height());
    band.height = std::min(rectangle.width(), rectangle.height());
    band.angle  = rectangle.angle();
    // A band drawn taller than it is wide still names the same edge; the long side defines the
    // direction, so a quarter turn keeps the rectangle in its place.
    if (rectangle.height() > rectangle.width())
        band.angle += 90.0;
    band.center = {rectangle.left() + rectangle.width() / 2.0, rectangle.top() + rectangle.height() / 2.0};
    return band;
}
} // namespace

visionrt::ToolStatus TrainStraightBordersTool::initialize(const visionrt::Values &)
{
    return {};
}

visionrt::ToolResult TrainStraightBordersTool::execute(const visionrt::Values &inputs)
{
    const auto  setEntry = inputs.find(p::in::SET);
    const auto *set
        = setEntry == inputs.end() ? nullptr : std::get_if<visionrt::DataRef<visionrt::TrainingSet>>(&setEntry->second);
    if (!set)
        return failure(missingOutputs(), CommonToolError::InputTypeMismatch, "训练集");
    const auto *regionValue = integerOf(inputs, p::in::REGION_FIELD);
    if (!regionValue)
        return failure(missingOutputs(), CommonToolError::InputTypeMismatch, "区域字段");
    const auto *contrast     = realOf(inputs, p::in::CONTRAST);
    const auto *edgeWidth    = integerOf(inputs, p::in::EDGE_WIDTH);
    const auto *polarity     = textOf(inputs, p::in::POLARITY);
    const auto *completeness = realOf(inputs, p::in::MIN_COMPLETENESS);
    if (!contrast || !edgeWidth || !polarity || !completeness)
        return failure(missingOutputs(), CommonToolError::InputTypeMismatch, "搜索参数");

    try
    {
        if (!(*contrast >= 0.0) || !(*contrast <= 1024.0))
            fail(CommonToolError::InputOutOfRange, "边缘对比度", "必须在 [0, 1024] 内");
        if (*edgeWidth < 1 || *edgeWidth > 63)
            fail(CommonToolError::InputOutOfRange, "边缘宽度", "必须在 [1, 63] 内");
        if (!(*completeness >= 0.0) || !(*completeness <= 1.0))
            fail(CommonToolError::InputOutOfRange, "最低边完整度", "必须在 [0, 1] 内");

        EdgeSearchOptions options;
        options.contrast        = *contrast;
        options.width           = static_cast<int>(*edgeWidth);
        options.polarity        = parsePolarity(*polarity);
        options.minCompleteness = *completeness;

        const auto &training = set->get();
        visionrt::validateTrainingSet(training);
        if (training.records.empty())
            fail(RecognitionError::ReferenceSegmentMissing, "训练集");
        if (training.records.size() > 1)
            fail(RecognitionError::ReferenceSegmentMissing, "训练集",
                 "一次只能从一张参考图画两条边，实际 " + std::to_string(training.records.size()) + " 条记录");
        const auto &record = training.records.begin()->second;

        const auto  regions = record.fields.find(static_cast<FieldId>(*regionValue));
        const auto *rois    = regions == record.fields.end()
                                ? nullptr
                                : std::get_if<visionrt::DataRef<visionrt::ROI>>(&regions->second);
        if (!rois || rois->get().size() != 1 || rois->get().front().size() != 2)
            fail(RecognitionError::StraightEdgeRegionsInvalid, "参考边", "一张参考图上只能有两条搜索矩形");

        const auto image = training.images.find(record.imageId);
        if (image == training.images.end())
            fail(RecognitionError::ReferenceSegmentMissing, "参考图");
        const cv::Mat gray = edge::grayscale(image->second.pixels);

        std::vector<SearchRectangle> bands;
        bands.reserve(2);
        for (std::size_t index = 0; index < rois->get().front().size(); ++index)
            bands.push_back(toBand(rois->get().front()[index], index));

        const auto model = StraightEdgeModel::train(gray, bands, options);
        const auto bytes = model.encode();

        std::vector<roi::Record> segments;
        segments.reserve(model.segments().size());
        for (const auto &segment : model.segments())
            segments.push_back(roi::segmentValue(segment.from.x, segment.from.y, segment.to.x, segment.to.y));

        visionrt::Values outputs;
        outputs.emplace(p::out::MODEL, visionrt::Array::dense(visionrt::DType::UInt8,
                                                              {static_cast<std::int64_t>(bytes.size())}, bytes));
        outputs.emplace(p::out::SEGMENTS, roi::roiValue({std::move(segments)}));
        return visionrt::toolResult(std::move(outputs), {0, "ok"});
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

visionrt::ToolStatus TrainStraightBordersTool::release()
{
    return {};
}

} // namespace visiontools::recognition
