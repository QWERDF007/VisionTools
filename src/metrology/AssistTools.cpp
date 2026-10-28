#include "AssistTools.h"

#include "AssistSupport.h"
#include "EdgeProfile.h"

#include "common/GeometryTransform.h"
#include "common/ImageSupport.h"
#include "common/InputValues.h"
#include "common/RoiSupport.h"
#include "errors/MetrologyErrors.h"
#include "common/errors/ToolError.h"

#include <utility>
#include <vector>

namespace visiontools::metrology {
namespace {
namespace p = ports::assist;

struct AssistInputs
{
    const std::vector<cv::Mat> &images;
    AssistRequest               request;
};

AssistRequest readRequest(const visionrt::Values &inputs)
{
    AssistRequest request;
    const auto found = inputs.find(p::in::CURSOR);
    if (found == inputs.end() || visionrt::isMissing(found->second))
        fail(CommonToolError::InputMissing, p::in::CURSOR);
    const auto *array = std::get_if<visionrt::Array>(&found->second);
    if (!array || array->type() != visionrt::TypeSpec{visionrt::Type::Array, visionrt::DType::Float64, 1})
        fail(CommonToolError::InputTypeMismatch, std::string(p::in::CURSOR) + " 必须是 Array(float64, rank=1)");
    const auto &values = array->get<visionrt::NestedVector<double, 1>>();
    if (values.size() != 2 || !std::isfinite(values[0]) || !std::isfinite(values[1]))
        fail(CommonToolError::InputOutOfRange, std::string(p::in::CURSOR) + " 必须恰好包含两个有限像素坐标");
    request.cursor = {values[0], values[1]};

    Readings   readings(inputs);
    const auto phase        = readings.text(p::in::PHASE);
    const auto searchExtent = readings.integer(p::in::SEARCH_EXTENT);
    const auto imagesRef    = readings.image(p::in::IMAGES);
    const auto seedValue    = readings.find(p::in::SEED_ROIS);
    const auto alignValue   = readings.find(p::in::ALIGNMENT);
    if (readings.failed())
        fail(CommonToolError::InputMissing, readings.issue()->describe());
    if (phase != "hover" && phase != "learn")
        fail(CommonToolError::InputOutOfRange, std::string(p::in::PHASE) + " 只接受 hover/learn，实际值：" + phase);
    if (searchExtent < 16 || searchExtent > 2048)
        fail(CommonToolError::InputOutOfRange,
             std::string(p::in::SEARCH_EXTENT) + " 必须在 [16, 2048] 内，实际值：" + std::to_string(searchExtent));
    if (!seedValue || !alignValue)
        fail(CommonToolError::InputMissing, !seedValue ? p::in::SEED_ROIS : p::in::ALIGNMENT);
    request.learn        = phase == "learn";
    request.searchExtent = static_cast<int>(searchExtent);
    if (imagesRef.get().size() != 1)
        fail(CommonToolError::InputOutOfRange, "辅助工具只接受单张图像");
    const auto groups = roi::groups(*seedValue, 1, "seed_rois");
    if (!groups.empty())
        request.seeds = groups.front();
    const auto alignments = geometry::alignmentsFromValue(*alignValue, 1);
    request.alignment     = alignments.front();
    if (!request.alignment.isSimilarity())
        fail(MetrologyError::RoiKindMismatch, "alignment 必须是保向相似变换");
    return request;
}

visionrt::Values baseOutputs(const AssistResult &result)
{
    std::vector<std::vector<roi::Record>> preview{result.preview};
    std::vector<std::vector<roi::Record>> proposal{result.proposal};
    if (!result.valid)
    {
        preview.clear();
        proposal.clear();
    }
    visionrt::Values outputs;
    outputs.emplace(p::out::PREVIEW_ROIS, roi::roiValue(std::move(preview)));
    outputs.emplace(p::out::PROPOSAL_ROIS, roi::roiValue(std::move(proposal)));
    outputs.emplace(p::out::CANDIDATE_VALID, result.valid);
    outputs.emplace(p::out::QUALITY, result.quality);
    outputs.emplace(p::out::SUGGESTED_POLARITY, result.polarity);
    return outputs;
}

visionrt::Values missingCircleOutputs()
{
    return {{p::out::PREVIEW_ROIS, visionrt::Missing{}},
            {p::out::PROPOSAL_ROIS, visionrt::Missing{}},
            {p::out::CANDIDATE_VALID, visionrt::Missing{}},
            {p::out::QUALITY, visionrt::Missing{}},
            {p::out::SUGGESTED_POLARITY, visionrt::Missing{}},
            {p::out::SUGGESTED_SEARCH, visionrt::Missing{}}};
}

visionrt::Values missingRowOutputs()
{
    return {{p::out::PREVIEW_ROIS, visionrt::Missing{}},
            {p::out::PROPOSAL_ROIS, visionrt::Missing{}},
            {p::out::CANDIDATE_VALID, visionrt::Missing{}},
            {p::out::QUALITY, visionrt::Missing{}},
            {p::out::SUGGESTED_POLARITY, visionrt::Missing{}},
            {p::out::SUGGESTED_WIDTH_MIN, visionrt::Missing{}},
            {p::out::SUGGESTED_WIDTH_MAX, visionrt::Missing{}}};
}
} // namespace

visionrt::ToolStatus AssistCircleTool::initialize(const visionrt::Values &parameters)
{
    if (!parameters.empty())
        return {visionrt::kContractViolationStatus, "圆智能点选辅助工具不接受初始化参数"};
    return {};
}

visionrt::ToolResult AssistCircleTool::execute(const visionrt::Values &inputs)
{
    try
    {
        const auto request = readRequest(inputs);
        const auto result  = assistCircle(grayscale(requireImages(inputs, p::in::IMAGES).front()), request);
        auto       outputs = baseOutputs(result);
        outputs.emplace(p::out::SUGGESTED_SEARCH, result.searchHalfWidth);
        return visionrt::toolResult(std::move(outputs), visionrt::executionSuccess());
    }
    catch (const ToolFailure &error)
    {
        return failureResult(missingCircleOutputs(), error);
    }
    catch (const std::exception &error)
    {
        return failure(missingCircleOutputs(), MetrologyError::MeasurementFailed, "圆智能点选失败", error.what());
    }
    catch (...)
    {
        return failure(missingCircleOutputs(), MetrologyError::MeasurementFailed, "圆智能点选失败：未知异常");
    }
}

visionrt::ToolStatus AssistCircleTool::release()
{
    return {};
}

visionrt::ToolStatus AssistRowSpacingTool::initialize(const visionrt::Values &parameters)
{
    if (!parameters.empty())
        return {visionrt::kContractViolationStatus, "行间距智能点选辅助工具不接受初始化参数"};
    return {};
}

visionrt::ToolResult AssistRowSpacingTool::execute(const visionrt::Values &inputs)
{
    try
    {
        const auto request = readRequest(inputs);
        const auto result  = assistRowSpacing(grayscale(requireImages(inputs, p::in::IMAGES).front()), request);
        auto       outputs = baseOutputs(result);
        outputs.emplace(p::out::SUGGESTED_WIDTH_MIN, result.widthMin);
        outputs.emplace(p::out::SUGGESTED_WIDTH_MAX, result.widthMax);
        return visionrt::toolResult(std::move(outputs), visionrt::executionSuccess());
    }
    catch (const ToolFailure &error)
    {
        return failureResult(missingRowOutputs(), error);
    }
    catch (const std::exception &error)
    {
        return failure(missingRowOutputs(), MetrologyError::MeasurementFailed, "行间距智能点选失败", error.what());
    }
    catch (...)
    {
        return failure(missingRowOutputs(), MetrologyError::MeasurementFailed, "行间距智能点选失败：未知异常");
    }
}

visionrt::ToolStatus AssistRowSpacingTool::release()
{
    return {};
}
} // namespace visiontools::metrology
