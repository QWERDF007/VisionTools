#include "LocatorTools.h"

#include "ParameterLocation.h"
#include "common/EdgeProfile.h"
#include "common/GeometryTransform.h"
#include "common/ImageSupport.h"
#include "common/InputValues.h"
#include "common/RoiSupport.h"
#include "common/errors/ToolError.h"
#include "errors/MetrologyErrors.h"

#include <utility>
#include <vector>

namespace visiontools::metrology {

using edge::grayscale;

namespace {
namespace p = ports::locate;

LocationRequest readRequest(const visionrt::Values &inputs, std::string_view tool)
{
    Readings   readings(inputs, tool);
    const auto images = readings.image(p::in::IMAGES);
    const auto search = readings.circle(p::in::SEARCH_REGION);
    if (readings.failed())
        fail(CommonToolError::InputMissing, readings.issue()->describe());
    if (images.get().size() != 1)
        fail(CommonToolError::InputOutOfRange, "定位算法只接受单张图像");
    const auto      window = roi::circularSearch(search, "搜索区域");
    LocationRequest request;
    request.searchCenter = window.center;
    request.searchRadius = window.radius;
    if (const auto seed = readings.find(p::in::SEED_ROIS))
        if (const auto regions = roi::groups(*seed, 1, "标称区域"); !regions.empty())
            request.seeds = regions.front();
    if (const auto alignment = readings.find(p::in::ALIGNMENT))
        request.alignment = geometry::alignmentsFromValue(*alignment, 1).front();
    if (!request.alignment.isSimilarity())
        fail(MetrologyError::RoiKindMismatch, "alignment 必须是保向相似变换");
    return request;
}

template<class Shape>
visionrt::Values baseOutputs(const LocationResult &result)
{
    std::vector<std::vector<roi::Record>> preview{result.preview};
    std::vector<std::vector<roi::Record>> proposal{result.proposal};
    if (!result.valid)
    {
        preview.clear();
        proposal.clear();
    }
    visionrt::Values outputs;
    outputs.emplace(p::out::PREVIEW_ROIS, roi::value<Shape>(std::move(preview)));
    outputs.emplace(p::out::PROPOSAL_ROIS, roi::value<Shape>(std::move(proposal)));
    outputs.emplace(p::out::CANDIDATE_VALID, result.valid);
    outputs.emplace(p::out::QUALITY, result.quality);
    outputs.emplace(p::out::SUGGESTED_POLARITY, result.polarity);
    return outputs;
}

} // namespace

visionrt::ToolStatus LocateCircleTool::initialize(const visionrt::ToolConfiguration &configuration)
{
    const auto &parameters = configuration.parameters;
    if (!parameters.empty())
        return {visionrt::kContractViolationStatus, "圆定位辅助工具不接受初始化参数"};
    return {};
}

visionrt::ToolResult LocateCircleTool::execute(const visionrt::Values &inputs)
{
    try
    {
        const auto request = readRequest(inputs, p::CIRCLE_ID);
        const auto result  = locateCircle(grayscale(requireImages(inputs, p::in::IMAGES, "图像").front()), request);
        auto       outputs = baseOutputs<visionrt::Circle>(result);
        outputs.emplace(p::circle_out::SUGGESTED_SEARCH, result.searchHalfWidth);
        outputs.emplace(p::circle_out::SUGGESTED_SIGMA, result.sigma);
        return visionrt::toolResult(std::move(outputs), visionrt::executionSuccess());
    }
    catch (const ToolFailure &error)
    {
        return failureResult(visionrt::Values{}, error);
    }
    catch (const std::exception &error)
    {
        return failure(visionrt::Values{}, MetrologyError::MeasurementFailed, "圆定位失败", error.what());
    }
    catch (...)
    {
        return failure(visionrt::Values{}, MetrologyError::MeasurementFailed, "圆定位失败：未知异常");
    }
}

visionrt::ToolStatus LocateCircleTool::release()
{
    return {};
}

visionrt::ToolStatus LocateRowSpacingTool::initialize(const visionrt::ToolConfiguration &configuration)
{
    const auto &parameters = configuration.parameters;
    if (!parameters.empty())
        return {visionrt::kContractViolationStatus, "行间距定位辅助工具不接受初始化参数"};
    return {};
}

visionrt::ToolResult LocateRowSpacingTool::execute(const visionrt::Values &inputs)
{
    try
    {
        const auto request = readRequest(inputs, p::ROW_ID);
        const auto result = locateRowSpacing(grayscale(requireImages(inputs, p::in::IMAGES, "图像").front()), request);
        auto outputs = baseOutputs<visionrt::Rectangle>(result);
        outputs.emplace(p::row_out::SUGGESTED_WIDTH_MIN, result.widthMin);
        outputs.emplace(p::row_out::SUGGESTED_WIDTH_MAX, result.widthMax);
        return visionrt::toolResult(std::move(outputs), visionrt::executionSuccess());
    }
    catch (const ToolFailure &error)
    {
        return failureResult(visionrt::Values{}, error);
    }
    catch (const std::exception &error)
    {
        return failure(visionrt::Values{}, MetrologyError::MeasurementFailed, "行间距定位失败", error.what());
    }
    catch (...)
    {
        return failure(visionrt::Values{}, MetrologyError::MeasurementFailed, "行间距定位失败：未知异常");
    }
}

visionrt::ToolStatus LocateRowSpacingTool::release()
{
    return {};
}
} // namespace visiontools::metrology
