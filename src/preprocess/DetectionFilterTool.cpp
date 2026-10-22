#include "DetectionFilterTool.h"

#include "InputPorts.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace visiontools::preprocess {
namespace {

namespace p = ports::detection_filter;

template<class T>
visionrt::NumericBuffer makeBuffer(visionrt::DType dtype, std::vector<std::int64_t> shape, const std::vector<T> &values)
{
    return visionrt::NumericBuffer::copy(dtype, std::move(shape), std::as_bytes(std::span(values)));
}

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
        const auto *images = std::get_if<visionrt::ImageBatch>(&requireInput(inputs, p::in::IMAGES));
        if (!images)
            throw visionrt::ContractError("输入端口 '" + std::string(p::in::IMAGES) + "' 必须是 ImageBatch");
        const auto *rois = std::get_if<visionrt::RoiBatch>(&requireInput(inputs, p::in::ROIS));
        if (!rois)
            throw visionrt::ContractError("输入端口 '" + std::string(p::in::ROIS) + "' 必须是 RoiBatch");
        const auto *scores = std::get_if<visionrt::NumericArray>(&requireInput(inputs, p::in::SCORES));
        if (!scores)
            throw visionrt::ContractError("输入端口 '" + std::string(p::in::SCORES) + "' 必须是 NumericArray");
        const auto *classes = std::get_if<visionrt::NumericArray>(&requireInput(inputs, p::in::CLASSES));
        if (!classes)
            throw visionrt::ContractError("输入端口 '" + std::string(p::in::CLASSES) + "' 必须是 NumericArray");
        const auto *names = std::get_if<visionrt::StringArray>(&requireInput(inputs, p::in::CLASS_NAMES));
        if (!names)
            throw visionrt::ContractError("输入端口 '" + std::string(p::in::CLASS_NAMES) + "' 必须是 StringArray");
        const auto *allowedArray = std::get_if<visionrt::NumericArray>(&requireInput(inputs, p::in::ALLOWED_CLASSES));
        if (!allowedArray)
            throw visionrt::ContractError("输入端口 '" + std::string(p::in::ALLOWED_CLASSES) + "' 必须是 NumericArray");

        if (scores->dtype() != visionrt::DType::Float32 || classes->dtype() != visionrt::DType::Int64
            || allowedArray->dtype() != visionrt::DType::Int64)
        {
            throw visionrt::ContractError("检测过滤输入数组的 dtype 无效：scores 需要 float32，classes/allowed_classes 需要 int64");
        }

        const double minimumScore = requireFloatInRange(inputs, p::in::MIN_SCORE, 0.0, 1.0);
        const double minimumArea = requireFloat(inputs, p::in::MIN_AREA);
        if (!std::isfinite(minimumArea) || minimumArea < 0.0)
        {
            throw visionrt::ContractError("输入端口 '" + std::string(p::in::MIN_AREA) + "' 的值 "
                                          + std::to_string(minimumArea) + " 必须为非负有限数值");
        }

        std::span<const std::int64_t> allowed;
        if (!allowedArray->buffers().empty())
        {
            allowed = allowedArray->buffers()[0].values<std::int64_t>();
        }

        const auto batchSize = images->images().size();
        if (rois->images().size() != batchSize || scores->buffers().size() != batchSize
            || classes->buffers().size() != batchSize || names->buffers().size() != batchSize)
        {
            throw visionrt::ContractError("检测批次长度不一致：images=" + std::to_string(batchSize)
                                          + "，rois=" + std::to_string(rois->images().size())
                                          + "，scores=" + std::to_string(scores->buffers().size())
                                          + "，classes=" + std::to_string(classes->buffers().size())
                                          + "，class_names=" + std::to_string(names->buffers().size()));
        }

        std::vector<std::vector<visionrt::Roi>> filteredRois;
        std::vector<visionrt::NumericBuffer>    filteredScores, filteredClasses;
        std::vector<visionrt::StringBuffer>     filteredNames;
        std::vector<std::int64_t>               counts;

        filteredRois.reserve(batchSize);
        filteredScores.reserve(batchSize);
        filteredClasses.reserve(batchSize);
        filteredNames.reserve(batchSize);
        counts.reserve(batchSize);

        for (std::size_t b = 0; b < batchSize; ++b)
        {
            const auto &imageRois = rois->images()[b];
            const auto  imageScores = scores->buffers()[b].values<float>();
            const auto  imageClasses = classes->buffers()[b].values<std::int64_t>();
            const auto  imageNames = names->buffers()[b].values();

            if (imageScores.size() != imageRois.size() || imageClasses.size() != imageRois.size()
                || imageNames.size() != imageRois.size())
            {
                throw visionrt::ContractError("批次索引 " + std::to_string(b)
                                              + " 的检测字段长度不一致：rois=" + std::to_string(imageRois.size())
                                              + "，scores=" + std::to_string(imageScores.size())
                                              + "，classes=" + std::to_string(imageClasses.size())
                                              + "，class_names=" + std::to_string(imageNames.size()));
            }

            std::vector<visionrt::Roi> keptRois;
            std::vector<float>         keptScores;
            std::vector<std::int64_t>  keptClasses;
            std::vector<std::string>   keptNames;

            keptRois.reserve(imageRois.size());
            keptScores.reserve(imageRois.size());
            keptClasses.reserve(imageRois.size());
            keptNames.reserve(imageRois.size());

            for (std::size_t n = 0; n < imageRois.size(); ++n)
            {
                if (!std::isfinite(imageScores[n]) || imageClasses[n] < 0)
                {
                    throw visionrt::ContractError("批次索引 " + std::to_string(b) + " 的检测索引 "
                                                  + std::to_string(n) + " 的得分或类别无效");
                }
                if (imageRois[n].kind() != visionrt::RoiKind::Rectangle)
                {
                    throw visionrt::ContractError("批次索引 " + std::to_string(b) + " 的检测索引 "
                                                  + std::to_string(n) + " 的 ROI 必须是矩形");
                }
                const auto   encoded = imageRois[n].encoded();
                const double area = static_cast<double>(encoded[3]) * encoded[4];

                const bool classAllowed
                    = allowed.empty() || (std::find(allowed.begin(), allowed.end(), imageClasses[n]) != allowed.end());

                if (imageScores[n] >= minimumScore && area >= minimumArea && classAllowed)
                {
                    keptRois.push_back(imageRois[n]);
                    keptScores.push_back(imageScores[n]);
                    keptClasses.push_back(imageClasses[n]);
                    keptNames.push_back(std::string(imageNames[n]));
                }
            }

            const auto count = static_cast<std::int64_t>(keptRois.size());
            counts.push_back(count);
            filteredRois.push_back(std::move(keptRois));
            filteredScores.push_back(makeBuffer(visionrt::DType::Float32, {count}, keptScores));
            filteredClasses.push_back(makeBuffer(visionrt::DType::Int64, {count}, keptClasses));
            filteredNames.emplace_back(std::vector<std::int64_t>{count}, std::move(keptNames));
        }

        visionrt::Values outputs;
        outputs.emplace(p::out::IMAGES, *images);
        outputs.emplace(p::out::ROIS, visionrt::RoiBatch(std::move(filteredRois)));
        outputs.emplace(p::out::SCORES,
                        visionrt::NumericArray(visionrt::DType::Float32, 2, visionrt::Layout::BatchRagged,
                                               std::move(filteredScores)));
        outputs.emplace(p::out::CLASSES,
                        visionrt::NumericArray(visionrt::DType::Int64, 2, visionrt::Layout::BatchRagged,
                                               std::move(filteredClasses)));
        outputs.emplace(p::out::CLASS_NAMES,
                        visionrt::StringArray(2, visionrt::Layout::BatchRagged, std::move(filteredNames)));
        outputs.emplace(p::out::COUNTS,
                        visionrt::NumericArray(
                            visionrt::DType::Int64, 1, visionrt::Layout::Dense,
                            {makeBuffer(visionrt::DType::Int64, {static_cast<std::int64_t>(batchSize)}, counts)}));

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
