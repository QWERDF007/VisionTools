#include "BrightnessTool.h"

#include "common/GeometryTransform.h"
#include "common/UpstreamStatus.h"
#include "common/ImageSupport.h"
#include "common/InputValues.h"
#include "common/RoiSupport.h"

#include <opencv2/core.hpp>

#include <cmath>
#include <vector>

namespace visiontools::preprocess {
namespace {
namespace p = ports::image_brightness;

visionrt::Values missingOutputs()
{
    return {{p::out::IMAGES, visionrt::Missing{}}, {p::out::APPLIED_MASK, visionrt::Missing{}}};
}

const std::vector<cv::Mat> *optionalMask(const visionrt::Values &inputs)
{
    const auto found = inputs.find(p::in::MASK);
    if (found == inputs.end() || visionrt::isMissing(found->second))
        return nullptr;
    const auto *mask = std::get_if<visionrt::DataRef<visionrt::Image>>(&found->second);
    if (!mask)
        throw visionrt::ContractError("mask 必须是图像");
    return mask->get().empty() ? nullptr : &mask->get();
}

// `wholeImage` distinguishes an unconfigured ROI list (`[]`, the full image) from
// an explicit empty group for one image (`[[]]`, which keeps that image unchanged).
cv::Mat combinedMask(const std::vector<roi::Record> &records, bool wholeImage,
                     const geometry::Alignment &toReference, cv::Size size, const cv::Mat *external)
{
    if (wholeImage)
    {
        if (!external)
            return cv::Mat(size, CV_8UC1, cv::Scalar(255));
        return external->clone();
    }
    if (records.empty())
        return cv::Mat::zeros(size, CV_8UC1);
    auto region = roi::regionMask(records, size, toReference);
    if (!external)
        return region;
    cv::Mat result;
    cv::bitwise_and(region, *external, result);
    return result;
}
} // namespace

visionrt::ToolStatus BrightnessTool::initialize(const visionrt::Values &parameters)
{
    if (!parameters.empty())
        return {1, "调整亮度不接受初始化参数"};
    return {};
}

visionrt::ToolResult BrightnessTool::execute(const visionrt::Values &inputs)
{
    // An unselected upstream branch is not an error: no algorithm runs and nothing is written.
    if (const auto upstream = upstreamStatus(inputs); upstream != 0)
        return skippedByUpstream(upstream, missingOutputs());
    try
    {
        const auto &images = requireImages(inputs, p::in::IMAGES);
        const double gain  = requireFloatInRange(inputs, p::in::GAIN, 0.0, 1.0e12);
        const double offset = requireFloatInRange(inputs, p::in::OFFSET, -1.0e12, 1.0e12);
        const auto *emitValue = std::get_if<bool>(&requireInput(inputs, p::in::EMIT_MASK));
        if (!emitValue)
            throw visionrt::ContractError("emit_mask 必须是布尔值");
        const bool emitMask = *emitValue;
        const auto   alignments
            = geometry::alignmentsFromValue(requireInput(inputs, p::in::ALIGNMENT), images.size());
        const auto records = roi::groups(requireInput(inputs, p::in::ROIS), images.size(), "rois");
        // An empty outer list means "whole image"; per-image empty groups mean
        // "leave this image unchanged".
        const bool wholeImage = records.empty();
        const auto *masks  = optionalMask(inputs);
        if (masks && masks->size() != images.size())
            throw visionrt::ContractError("mask 外层长度必须与图像数量一致");

        visionrt::Image                output;
        visionrt::Image                appliedMasks;
        output.reserve(images.size());
        appliedMasks.reserve(images.size());
        for (std::size_t index = 0; index < images.size(); ++index)
        {
            const auto &image = images[index];
            requireSupportedImage(image, "调整亮度");
            const cv::Mat *external = masks ? &(*masks)[index] : nullptr;
            if (external)
            {
                if (external->empty() || external->type() != CV_8UC1)
                    throw visionrt::ContractError("mask 每张图必须是单通道 8 位图像");
                if (external->size() != image.size())
                    throw visionrt::ContractError("mask 尺寸必须与图像一致");
            }
            const auto toReference = alignments[index].inverse();
            if (!toReference)
                throw visionrt::ContractError("alignment 线性部分必须可逆");
            const auto  region = combinedMask(wholeImage ? std::vector<roi::Record>{} : records[index],
                                              wholeImage, *toReference, image.size(), external);
            cv::Mat     destination = image.clone();
            const bool  constant   = gain == 1.0 && offset == 0.0;
            if (!constant)
            {
                cv::Mat converted;
                cv::Mat scaled;
                image.convertTo(converted, -1, gain, offset);
                if (image.channels() == 4)
                {
                    // Alpha keeps the source value: only BGR is adjusted.
                    std::vector<cv::Mat> sourceChannels, convertedChannels;
                    cv::split(image, sourceChannels);
                    cv::split(converted, convertedChannels);
                    sourceChannels[3].copyTo(convertedChannels[3]);
                    cv::merge(convertedChannels, scaled);
                }
                else
                    scaled = converted;
                scaled.copyTo(destination, region);
            }
            output.push_back(std::move(destination));
            appliedMasks.push_back(region);
        }
        visionrt::Values outputs;
        outputs.emplace(p::out::IMAGES,
                        visionrt::DataRef<visionrt::Image>(std::move(output),
                                                           {std::get<visionrt::DataRef<visionrt::Image>>(
                                                                requireInput(inputs, p::in::IMAGES))
                                                                .owner()}));
        if (emitMask)
            outputs.emplace(p::out::APPLIED_MASK, visionrt::DataRef<visionrt::Image>(std::move(appliedMasks)));
        else
            outputs.emplace(p::out::APPLIED_MASK, visionrt::Missing{});
        return visionrt::toolResult(std::move(outputs), visionrt::executionSuccess());
    }
    catch (const visionrt::ContractError &error)
    {
        return visionrt::toolResult(missingOutputs(), {1, error.what()});
    }
    catch (const cv::Exception &error)
    {
        return visionrt::toolResult(missingOutputs(), {1, "调整亮度失败", error.what()});
    }
    catch (const std::exception &error)
    {
        return visionrt::toolResult(missingOutputs(), {1, "调整亮度失败", error.what()});
    }
    catch (...)
    {
        return visionrt::toolResult(missingOutputs(), {-1, "调整亮度失败：未知异常"});
    }
}

visionrt::ToolStatus BrightnessTool::release()
{
    return {};
}
} // namespace visiontools::preprocess
