#include "LocateStraightEdgeTool.h"

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
namespace p = ports::locate_straight_edge;

EdgeProbeRequest readRequest(const visionrt::Values &inputs)
{
    Readings   readings(inputs, p::TOOL_ID);
    const auto images = readings.image(p::in::IMAGES);
    const auto search = readings.circle(p::in::SEARCH_REGION);
    if (readings.failed())
        fail(CommonToolError::InputMissing, readings.issue()->describe());
    if (images.get().size() != 1)
        fail(CommonToolError::InputOutOfRange, "直边定位只接受单张图像");
    const auto       window = roi::circularSearch(search, "搜索区域");
    EdgeProbeRequest request;
    request.searchCenter = window.center;
    request.searchRadius = window.radius;
    if (const auto alignment = readings.find(p::in::ALIGNMENT))
        request.alignment = geometry::alignmentsFromValue(*alignment, 1).front();
    if (!request.alignment.isSimilarity())
        fail(RecognitionError::AssetFormatInvalid, "alignment", "必须是保向相似变换");
    if (const auto regions = readings.find(p::in::SEED_ROIS))
        if (const auto groups = roi::groups(*regions, 1, "参数估计区域"); !groups.empty())
            request.parameterRegions = groups.front();
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
    outputs.emplace(p::out::PREVIEW_ROIS, roi::value<visionrt::Rectangle>(std::move(preview)));
    outputs.emplace(p::out::PROPOSAL_ROIS, roi::value<visionrt::Rectangle>(std::move(proposal)));
    outputs.emplace(p::out::CANDIDATE_VALID, result.valid);
    outputs.emplace(p::out::QUALITY, result.quality);
    outputs.emplace(p::out::SUGGESTED_POLARITY, result.polarity);
    outputs.emplace(p::out::SUGGESTED_CONTRAST, result.contrast);
    outputs.emplace(p::out::SUGGESTED_WIDTH, static_cast<std::int64_t>(result.width));
    return outputs;
}
} // namespace

visionrt::ToolStatus LocateStraightEdgeTool::initialize(const visionrt::ToolConfiguration &configuration)
{
    const auto &parameters = configuration.parameters;
    if (!parameters.empty())
        return {visionrt::kContractViolationStatus, "直边定位辅助工具不接受初始化参数"};
    return {};
}

visionrt::ToolResult LocateStraightEdgeTool::execute(const visionrt::Values &inputs)
{
    try
    {
        const auto request = readRequest(inputs);
        const auto gray    = edge::grayscale(requireImages(inputs, p::in::IMAGES, "图像").front());
        return visionrt::toolResult(proposalOutputs(probeStraightEdge(gray, request)), visionrt::executionSuccess());
    }
    catch (const ToolFailure &error)
    {
        return failureResult(visionrt::Values{}, error);
    }
    catch (const std::exception &error)
    {
        return failure(visionrt::Values{}, CommonToolError::OperatorFailed, "直边定位失败", error.what());
    }
    catch (...)
    {
        return failure(visionrt::Values{}, CommonToolError::OperatorFailed, "直边定位失败：未知异常");
    }
}

visionrt::ToolStatus LocateStraightEdgeTool::release()
{
    return {};
}

} // namespace visiontools::recognition
