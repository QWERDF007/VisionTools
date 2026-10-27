#include "AssistTools.h"

#include "AssistSupport.h"
#include "EdgeProfile.h"

#include "common/GeometryTransform.h"
#include "common/ImageSupport.h"
#include "common/InputValues.h"
#include "common/RoiSupport.h"

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
    const auto   &cursor = requireInput(inputs, p::in::CURSOR);
    const auto   *array  = std::get_if<visionrt::Array>(&cursor);
    if (!array || array->type() != visionrt::TypeSpec{visionrt::Type::Array, visionrt::DType::Float64, 1})
        throw visionrt::ContractError("cursor 必须是 Array(float64, rank=1)");
    const auto &values = array->get<visionrt::NestedVector<double, 1>>();
    if (values.size() != 2 || !std::isfinite(values[0]) || !std::isfinite(values[1]))
        throw visionrt::ContractError("cursor 必须恰好包含两个有限像素坐标");
    request.cursor = {values[0], values[1]};
    const auto phase = requireString(inputs, p::in::PHASE);
    if (phase != "hover" && phase != "learn")
        throw visionrt::ContractError("输入端口 'phase' 的取值无效：" + phase + "（允许 hover/learn）");
    request.learn        = phase == "learn";
    request.searchExtent = requireInt32(inputs, p::in::SEARCH_EXTENT, 16, 2048);

    const auto &images = requireImages(inputs, p::in::IMAGES);
    if (images.size() != 1)
        throw visionrt::ContractError("辅助工具只接受单张图像");
    const auto groups
        = roi::groups(requireInput(inputs, p::in::SEED_ROIS), 1, "seed_rois");
    if (!groups.empty())
        request.seeds = groups.front();
    const auto alignments = geometry::alignmentsFromValue(requireInput(inputs, p::in::ALIGNMENT), 1);
    request.alignment     = alignments.front();
    if (!request.alignment.isSimilarity())
        throw visionrt::ContractError("辅助工具仅接受保向相似变换的 alignment");
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
        return {1, "圆智能点选辅助工具不接受初始化参数"};
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
    catch (const visionrt::ContractError &error)
    {
        return visionrt::toolResult(missingCircleOutputs(), {1, error.what()});
    }
    catch (const std::exception &error)
    {
        return visionrt::toolResult(missingCircleOutputs(), {1, "圆智能点选失败", error.what()});
    }
    catch (...)
    {
        return visionrt::toolResult(missingCircleOutputs(), {-1, "圆智能点选失败：未知异常"});
    }
}

visionrt::ToolStatus AssistCircleTool::release()
{
    return {};
}

visionrt::ToolStatus AssistRowSpacingTool::initialize(const visionrt::Values &parameters)
{
    if (!parameters.empty())
        return {1, "行间距智能点选辅助工具不接受初始化参数"};
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
    catch (const visionrt::ContractError &error)
    {
        return visionrt::toolResult(missingRowOutputs(), {1, error.what()});
    }
    catch (const std::exception &error)
    {
        return visionrt::toolResult(missingRowOutputs(), {1, "行间距智能点选失败", error.what()});
    }
    catch (...)
    {
        return visionrt::toolResult(missingRowOutputs(), {-1, "行间距智能点选失败：未知异常"});
    }
}

visionrt::ToolStatus AssistRowSpacingTool::release()
{
    return {};
}
} // namespace visiontools::metrology
