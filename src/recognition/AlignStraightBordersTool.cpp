#include "AlignStraightBordersTool.h"

#include "StraightEdgeLocator.h"
#include "common/EdgeProfile.h"
#include "common/GeometryTransform.h"
#include "common/ImageSupport.h"
#include "common/InputValues.h"
#include "common/ProductPaths.h"
#include "common/RoiSupport.h"
#include "common/ToolOutputs.h"
#include "common/UpstreamStatus.h"
#include "common/errors/ToolError.h"
#include "errors/RecognitionErrors.h"

#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace visiontools::recognition {
namespace {
namespace p = ports::align_straight_borders;

visionrt::Values missingOutputs()
{
    return missingPorts({p::out::ALIGNMENTS, p::out::FOUND_SEGMENTS});
}

} // namespace

visionrt::ToolStatus AlignStraightBordersTool::initialize(const visionrt::ToolConfiguration &configuration)
{
    try
    {
        const auto found = configuration.models.find("edge-model");
        if (found == configuration.models.end() || found->second.empty())
            return status(RecognitionError::StraightEdgeModelPathMissing, "模型路径");
        const auto utf8 = found->second.generic_u8string();
        modelPath_ = std::string(reinterpret_cast<const char*>(utf8.data()), utf8.size());
        std::ifstream file(found->second, std::ios::binary);
        if (!file) return status(RecognitionError::StraightEdgeModelUnreadable, modelPath_);
        const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
        model_ = StraightEdgeModel::decode(bytes);
        return {};
    }
    catch (const ToolFailure &error)
    {
        return status(error);
    }
    catch (const std::exception &error)
    {
        return status(CommonToolError::OperatorFailed, "加载直边模型失败", error.what());
    }
}

visionrt::ToolResult AlignStraightBordersTool::execute(const visionrt::Values &inputs)
{
    if (const auto upstream = upstreamStatus(inputs); upstream != 0)
        return skippedByUpstream(upstream, missingOutputs());
    if (model_.empty())
        return failure(missingOutputs(), RecognitionError::StraightEdgeModelPathMissing, "直边模型未加载");
    const auto &trained = model_.segments();

    Readings          readings(inputs, p::TOOL_ID);
    const auto        imagesRef  = readings.image(p::in::IMAGES);
    const auto        regionsRef = readings.roi(p::in::SEARCH_REGIONS);
    EdgeSearchOptions options;
    options.contrast        = readings.real(p::in::CONTRAST);
    const auto edgeWidth    = readings.integer(p::in::EDGE_WIDTH);
    const auto polarityName = readings.text(p::in::POLARITY);
    options.minCompleteness = readings.real(p::in::MIN_COMPLETENESS);
    if (readings.failed())
        return visionrt::toolResult(missingOutputs(),
                                    readings.status(CommonToolError::InputMissing, CommonToolError::InputTypeMismatch));

    try
    {
        options = EdgeSearchOptions::fromParameters(options.contrast, edgeWidth, polarityName, options.minCompleteness);

        const auto &images  = imagesRef.get();
        const auto  regions = roi::groups(regionsRef, images.size(), "搜索区域");
        if (regions.empty())
            fail(RecognitionError::StraightEdgeRegionsInvalid, "搜索区域", "需要两个有顺序的旋转矩形");
        std::vector<geometry::Alignment>      alignments;
        std::vector<std::vector<roi::Record>> foundSegments;
        alignments.reserve(images.size());
        foundSegments.reserve(images.size());

        for (std::size_t index = 0; index < images.size(); ++index)
        {
            requireSupportedImage(images[index], "通过直边确定对齐");
            if (images[index].depth() != CV_8U)
                fail(CommonToolError::UnsupportedImage, "第 " + std::to_string(index + 1) + " 张图不是 8 位图像");
            const cv::Mat gray = edge::grayscale(images[index]);

            if (regions[index].size() != 2)
                fail(RecognitionError::StraightEdgeRegionsInvalid, "搜索区域", "每张图需要恰好两个旋转矩形");
            const auto first  = locateSegment(gray, searchRectangle(regions[index][0], 0), options);
            const auto second = locateSegment(gray, searchRectangle(regions[index][1], 1), options);
            if (!first)
                fail(RecognitionError::EdgeNotFound, trained[0].name, "第 " + std::to_string(index + 1) + " 张图");
            if (!second)
                fail(RecognitionError::EdgeNotFound, trained[1].name, "第 " + std::to_string(index + 1) + " 张图");

            const auto alignment = rigidTransform(trained[0], *first, trained[1], *second);
            if (!alignment)
                fail(RecognitionError::ReferenceSegmentsParallel, "第 " + std::to_string(index + 1) + " 张图",
                     "参考边或当前找到的两条边近似平行");

            alignments.push_back(*alignment);
            std::vector<roi::Record> segments;
            segments.push_back(roi::segmentValue(first->start.x, first->start.y, first->end.x, first->end.y));
            segments.push_back(roi::segmentValue(second->start.x, second->start.y, second->end.x, second->end.y));
            foundSegments.push_back(std::move(segments));
        }

        visionrt::Values outputs;
        outputs.emplace(p::out::MODEL_PATH, modelPath_);
        outputs.emplace(p::out::ALIGNMENTS, geometry::alignmentArray(alignments));
        outputs.emplace(p::out::FOUND_SEGMENTS, roi::roiValue(std::move(foundSegments)));
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
        return failure(missingOutputs(), CommonToolError::OperatorFailed, "通过直边确定对齐失败", error.what());
    }
}

visionrt::ToolStatus AlignStraightBordersTool::release()
{
    model_ = {};
    return {};
}

} // namespace visiontools::recognition
