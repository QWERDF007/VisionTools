#include "DetectionFilterTool.h"

#include "common/InputValues.h"

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
        return {1, "缺陷过滤不接受初始化参数"};
    }
    return {};
}

visionrt::ToolResult DetectionFilterTool::execute(const visionrt::Values &inputs)
{
    try
    {
        const auto &images
            = requireTypedInput<visionrt::DataRef<visionrt::Image>>(inputs, p::in::IMAGES, {visionrt::Type::Image});
        const auto &rois
            = requireTypedInput<visionrt::DataRef<visionrt::ROI>>(inputs, p::in::ROIS, {visionrt::Type::ROI}).get();
        const auto &scores = requireTypedInput<visionrt::Array>(inputs, p::in::SCORES,
                                                                {visionrt::Type::Array, visionrt::DType::Float32, 2})
                                 .get<visionrt::NestedVector<float, 2>>();
        const auto &classes = requireTypedInput<visionrt::Array>(inputs, p::in::CLASSES,
                                                                 {visionrt::Type::Array, visionrt::DType::Int64, 2})
                                  .get<visionrt::NestedVector<std::int64_t, 2>>();
        const auto &names = requireTypedInput<visionrt::Array>(inputs, p::in::CLASS_NAMES,
                                                               {visionrt::Type::Array, visionrt::DType::String, 2})
                                .get<visionrt::NestedVector<std::string, 2>>();
        const auto &allowed = requireTypedInput<visionrt::Array>(inputs, p::in::ALLOWED_CLASSES,
                                                                 {visionrt::Type::Array, visionrt::DType::Int64, 1})
                                  .get<std::vector<std::int64_t>>();
        const double minimumScore = requireFloatInRange(inputs, p::in::MIN_SCORE, 0.0, 1.0);
        const double minimumArea = requireFloat(inputs, p::in::MIN_AREA);
        if (!std::isfinite(minimumArea) || minimumArea < 0.0)
            throw visionrt::ContractError("输入端口 '" + std::string(p::in::MIN_AREA) + "' 的值 "
                                          + std::to_string(minimumArea) + " 必须为非负有限数值");

        const auto batchSize = images.get().size();
        if (rois.size() != batchSize || scores.size() != batchSize || classes.size() != batchSize
            || names.size() != batchSize)
            throw visionrt::ContractError("检测批次长度不一致：images=" + std::to_string(batchSize) + "，rois="
                                          + std::to_string(rois.size()) + "，scores=" + std::to_string(scores.size())
                                          + "，classes=" + std::to_string(classes.size())
                                          + "，class_names=" + std::to_string(names.size()));

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
                throw visionrt::ContractError(
                    "批次索引 " + std::to_string(batch) + " 的检测字段长度不一致：rois="
                    + std::to_string(imageRois.size()) + "，scores=" + std::to_string(imageScores.size()) + "，classes="
                    + std::to_string(imageClasses.size()) + "，class_names=" + std::to_string(imageNames.size()));

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
                    throw visionrt::ContractError("批次索引 " + std::to_string(batch) + " 的检测索引 "
                                                  + std::to_string(detection) + " 的得分或类别无效");
                if (visionrt::roiKind(imageRois[detection]) != visionrt::RoiKind::Rectangle)
                    throw visionrt::ContractError("批次索引 " + std::to_string(batch) + " 的检测索引 "
                                                  + std::to_string(detection) + " 的 ROI 必须是矩形");
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
        return {visionrt::executionSuccess(), std::move(outputs)};
    }
    catch (const visionrt::ContractError &error)
    {
        return {{1, error.what()},
                {{p::out::IMAGES, visionrt::Missing{}}, {p::out::ROIS, visionrt::Missing{}},
                 {p::out::SCORES, visionrt::Missing{}}, {p::out::CLASSES, visionrt::Missing{}},
                 {p::out::CLASS_NAMES, visionrt::Missing{}}, {p::out::COUNTS, visionrt::Missing{}}}};
    }
    catch (const std::exception &error)
    {
        return {
            {1, "缺陷过滤失败", error.what()},
            {{p::out::IMAGES, visionrt::Missing{}},
             {p::out::ROIS, visionrt::Missing{}},
             {p::out::SCORES, visionrt::Missing{}},
             {p::out::CLASSES, visionrt::Missing{}},
             {p::out::CLASS_NAMES, visionrt::Missing{}},
             {p::out::COUNTS, visionrt::Missing{}}}
        };
    }
    catch (...)
    {
        return {{-1, "缺陷过滤失败：未知异常"},
                {{p::out::IMAGES, visionrt::Missing{}}, {p::out::ROIS, visionrt::Missing{}},
                 {p::out::SCORES, visionrt::Missing{}}, {p::out::CLASSES, visionrt::Missing{}},
                 {p::out::CLASS_NAMES, visionrt::Missing{}}, {p::out::COUNTS, visionrt::Missing{}}}};
    }
}

visionrt::ToolStatus DetectionFilterTool::release()
{
    return {};
}

} // namespace visiontools::preprocess
