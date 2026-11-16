#include "PickStraightEdgeTool.h"

#include "StraightEdgeProbe.h"

#include "common/EdgeProfile.h"
#include "common/GeometryTransform.h"
#include "common/ImageSupport.h"
#include "common/InputValues.h"
#include "common/RoiSupport.h"
#include "common/errors/ToolError.h"

#include "errors/RecognitionErrors.h"

#include <cmath>
#include <string>
#include <vector>

namespace visiontools::recognition {
namespace {
namespace p = ports::pick_straight_edge;

EdgeProbeRequest readRequest(const visionrt::Values &inputs)
{
    Readings readings(inputs, p::TOOL_ID);
    const auto imagesRef = readings.image(p::in::IMAGES);
    const auto phase     = readings.text(p::in::PHASE);
    const auto extent    = readings.integer(p::in::SEARCH_EXTENT);
    const auto cursor    = readings.find(p::in::CURSOR);
    if (readings.failed())
        fail(CommonToolError::InputMissing, readings.issue()->describe());
    if (!cursor || visionrt::isMissing(*cursor))
        fail(CommonToolError::InputMissing, parameterName(p::TOOL_ID, "inputs", p::in::CURSOR));
    const auto *array = std::get_if<visionrt::Array>(cursor);
    if (!array || array->type() != visionrt::TypeSpec{visionrt::Type::Array, visionrt::DType::Float64, 1})
        fail(CommonToolError::InputTypeMismatch,
             parameterName(p::TOOL_ID, "inputs", p::in::CURSOR) + " 必须是 Array(float64, rank=1)");
    const auto values = array->elements<double>();
    if (values.size() != 2 || !std::isfinite(values[0]) || !std::isfinite(values[1]))
        fail(CommonToolError::InputOutOfRange,
             parameterName(p::TOOL_ID, "inputs", p::in::CURSOR) + " 必须恰好包含两个有限像素坐标");
    if (phase != "hover" && phase != "learn")
        fail(CommonToolError::InputOutOfRange,
             parameterName(p::TOOL_ID, "inputs", p::in::PHASE) + " 只接受 hover/learn，实际值：" + phase);
    if (extent < 16 || extent > 2048)
        fail(CommonToolError::InputOutOfRange,
             parameterName(p::TOOL_ID, "inputs", p::in::SEARCH_EXTENT) + " 必须在 [16, 2048] 内，实际值："
                 + std::to_string(extent));
    if (imagesRef.get().size() != 1)
        fail(CommonToolError::InputOutOfRange, "直边智能点选只接受单张图像");

    EdgeProbeRequest request;
    request.cursor       = {values[0], values[1]};
    request.learn        = phase == "learn";
    request.searchExtent = static_cast<int>(extent);
    if (const auto *seedValue = readings.find(p::in::SEED_ROIS))
        if (const auto groups = roi::groups(*seedValue, 1, "seed_rois"); !groups.empty())
            request.seeds = groups.front();
    if (const auto *alignValue = readings.find(p::in::ALIGNMENT))
    {
        const auto alignments = geometry::alignmentsFromValue(*alignValue, 1);
        request.alignment     = alignments.front();
        if (!request.alignment.isSimilarity())
            fail(RecognitionError::AssetFormatInvalid, "alignment", "必须是保向相似变换");
    }
    return request;
}

visionrt::Values proposalOutputs(const EdgeProbeResult &result)
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
} // namespace

visionrt::ToolStatus PickStraightEdgeTool::initialize(const visionrt::Values &parameters)
{
    if (!parameters.empty())
        return {visionrt::kContractViolationStatus, "直边智能点选辅助工具不接受初始化参数"};
    return {};
}

visionrt::ToolResult PickStraightEdgeTool::execute(const visionrt::Values &inputs)
{
    try
    {
        const auto request = readRequest(inputs);
        const auto gray    = edge::grayscale(requireImages(inputs, p::in::IMAGES, "图像").front());
        return visionrt::toolResult(proposalOutputs(probeStraightEdge(gray, request)),
                                    visionrt::executionSuccess());
    }
    catch (const ToolFailure &error)
    {
        return failureResult(visionrt::Values{}, error);
    }
    catch (const std::exception &error)
    {
        return failure(visionrt::Values{}, CommonToolError::OperatorFailed, "直边智能点选失败", error.what());
    }
    catch (...)
    {
        return failure(visionrt::Values{}, CommonToolError::OperatorFailed, "直边智能点选失败：未知异常");
    }
}

visionrt::ToolStatus PickStraightEdgeTool::release()
{
    return {};
}

} // namespace visiontools::recognition
