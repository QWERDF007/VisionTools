#include "BrightnessTool.h"

#include "BrightnessParameters.h"

#include "common/GeometryTransform.h"
#include "common/ImageSupport.h"
#include "common/RoiSupport.h"

#include <opencv2/core.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace visiontools::preprocess {
namespace {
namespace p = ports::image_brightness;

double ranged(visionrt::InputReader &reader, std::string_view id, double minimum, double maximum)
{
    const double value = reader.real(id);
    if (!std::isfinite(value))
        throw visionrt::ContractError("输入端口 '" + std::string(id) + "' 必须为有限数值");
    if (value < minimum || value > maximum)
        throw visionrt::ContractError("输入端口 '" + std::string(id) + "' 的值 " + std::to_string(value)
                                      + " 超出允许范围 [" + std::to_string(minimum) + ", "
                                      + std::to_string(maximum) + "]");
    return value;
}

const std::vector<cv::Mat> *optionalMask(const std::optional<visionrt::DataRef<visionrt::Image>> &mask)
{
    if (!mask || mask->get().empty())
        return nullptr;
    return &mask->get();
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

void BrightnessInputs::readFields(visionrt::InputReader &reader)
{
    images   = reader.image(p::in::IMAGES);
    gain     = ranged(reader, p::in::GAIN, 0.0, 1.0e12);
    offset   = ranged(reader, p::in::OFFSET, -1.0e12, 1.0e12);
    emitMask = reader.boolean(p::in::EMIT_MASK);
    mask     = reader.nullableImage(p::in::MASK);
    // The shared kernels interpret these two values; the parameter view only keeps the handle.
    rois      = reader.require(p::in::ROIS);
    alignment = reader.require(p::in::ALIGNMENT);
}

void BrightnessOutputs::writeFields(visionrt::OutputWriter &writer)
{
    writer.optionalImage(p::out::IMAGES, images);
    writer.optionalImage(p::out::APPLIED_MASK, appliedMask);
}

visionrt::ToolResult brightnessTool(const visionrt::Values &values, const visionrt::ToolServices &)
{
    BrightnessInputs  inputs;
    BrightnessOutputs outputs;
    try
    {
        inputs.parse(values);
        // An unselected upstream branch is not an error: no algorithm runs and nothing is written.
        if (inputs.upstreamStatus != 0)
        {
            outputs.status = {inputs.upstreamStatus, "上游未选中本分支，工具未执行"};
            return outputs.pack();
        }

        const auto &images = inputs.images.get();
        const auto  records = roi::groups(inputs.rois, images.size(), "rois");
        const auto  alignments = geometry::alignmentsFromValue(inputs.alignment, images.size());
        // An empty outer list means "whole image"; per-image empty groups mean
        // "leave this image unchanged".
        const bool  wholeImage = records.empty();
        const auto *masks      = optionalMask(inputs.mask);
        if (masks && masks->size() != images.size())
            throw visionrt::ContractError("mask 外层长度必须与图像数量一致");

        visionrt::Image output;
        visionrt::Image appliedMasks;
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
            const auto region = combinedMask(wholeImage ? std::vector<roi::Record>{} : records[index],
                                             wholeImage, *toReference, image.size(), external);
            cv::Mat    destination = image.clone();
            const bool constant    = inputs.gain == 1.0 && inputs.offset == 0.0;
            if (!constant)
            {
                cv::Mat converted;
                cv::Mat scaled;
                image.convertTo(converted, -1, inputs.gain, inputs.offset);
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
        // clone() allocates every output plane, so the result owns its pixels and must not pin the
        // reader's decoded cache through the input owner.
        outputs.images = visionrt::DataRef<visionrt::Image>(std::move(output));
        if (inputs.emitMask)
            outputs.appliedMask = visionrt::DataRef<visionrt::Image>(std::move(appliedMasks));
        return outputs.pack();
    }
    catch (const visionrt::ContractError &error)
    {
        outputs.images.reset();
        outputs.appliedMask.reset();
        outputs.status = {1, error.what()};
        return outputs.pack();
    }
    catch (const cv::Exception &error)
    {
        outputs.images.reset();
        outputs.appliedMask.reset();
        outputs.status = {1, "调整亮度失败", error.what()};
        return outputs.pack();
    }
    catch (const std::exception &error)
    {
        outputs.images.reset();
        outputs.appliedMask.reset();
        outputs.status = {1, "调整亮度失败", error.what()};
        return outputs.pack();
    }
    catch (...)
    {
        outputs.images.reset();
        outputs.appliedMask.reset();
        outputs.status = {-1, "调整亮度失败：未知异常"};
        return outputs.pack();
    }
}

} // namespace visiontools::preprocess
