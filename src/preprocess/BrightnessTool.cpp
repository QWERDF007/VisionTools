#include "BrightnessTool.h"

#include "BrightnessParameters.h"

#include "common/GeometryTransform.h"
#include "common/ImageSupport.h"
#include "common/InputValues.h"
#include "common/RoiSupport.h"
#include "common/UpstreamStatus.h"
#include "common/errors/GeometryErrors.h"
#include "common/errors/ToolError.h"

#include <opencv2/core.hpp>

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace visiontools::preprocess {
namespace {
namespace p = ports::image_brightness;

// A declared numeric parameter is only usable inside the range the tool documents; the reader has
// already established that the value is a float of the declared type.
bool inRange(double value, double minimum, double maximum)
{
    return std::isfinite(value) && value >= minimum && value <= maximum;
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

visionrt::ToolStatus BrightnessInputs::readFields(visionrt::InputReader &reader)
{
    Readings readings(reader, "AdjustBrightness");
    images   = readings.nullableImage(p::in::IMAGES);
    gain     = readings.real(p::in::GAIN);
    offset   = readings.real(p::in::OFFSET);
    mask     = readings.nullableImage(p::in::MASK);
    rectangles = readings.rectangle(p::in::ROIS);
    circles = readings.nullableCircle(p::in::CIRCLES);
    polygons = readings.nullablePolygon(p::in::POLYGONS);
    if (readings.failed())
        return readings.status(CommonToolError::InputMissing, CommonToolError::InputTypeMismatch);
    if (!inRange(gain, 0.0, 1.0e12))
        return status(CommonToolError::InputOutOfRange, parameterName("AdjustBrightness", "inputs", p::in::GAIN) + " 必须在 [0, 1e12] 内");
    if (!inRange(offset, -1.0e12, 1.0e12))
        return status(CommonToolError::InputOutOfRange, parameterName("AdjustBrightness", "inputs", p::in::OFFSET) + " 必须在 [-1e12, 1e12] 内");
    // The shared kernels interpret these two values; the parameter view only keeps the handle.
    const auto *alignment = reader.find(p::in::ALIGNMENT);
    if (!alignment)
        return status(CommonToolError::InputMissing, parameterName("AdjustBrightness", "inputs", p::in::ALIGNMENT));
    this->alignment = *alignment;
    return {};
}

void BrightnessOutputs::writeFields(visionrt::OutputWriter &writer)
{
    writer.optionalImage(p::out::IMAGES, images);
    writer.optionalImage(p::out::APPLIED_MASK, appliedMask);
}

visionrt::ToolResult brightnessTool(const visionrt::Values &values)
{
    BrightnessInputs  inputs;
    BrightnessOutputs outputs;

    if (const auto upstream = upstreamStatus(values); upstream != 0)
        return skippedByUpstream(upstream, outputs.pack().outputs);

    const auto parsed = inputs.parse(values);
    if (!parsed)
    {
        outputs.status = parsed;
        return outputs.pack();
    }
    if (!inputs.images)
    {
        outputs.status = status(CommonToolError::InputMissing, parameterName("AdjustBrightness", "inputs", p::in::IMAGES));
        return outputs.pack();
    }

    try
    {
        const auto &images = inputs.images->get();
        std::vector<std::vector<roi::Record>> records;
        const std::array<visionrt::PortValue, 3> regions{
            inputs.rectangles,
            inputs.circles ? visionrt::PortValue{*inputs.circles} : visionrt::PortValue{visionrt::Missing{}},
            inputs.polygons ? visionrt::PortValue{*inputs.polygons} : visionrt::PortValue{visionrt::Missing{}}};
        for (const auto &value : regions)
        {
            const auto groups = roi::groups(value, images.size(), "处理区域");
            if (groups.empty()) continue;
            if (records.empty()) records.resize(images.size());
            for (std::size_t image = 0; image < groups.size(); ++image)
                records[image].insert(records[image].end(), groups[image].begin(), groups[image].end());
        }
        const auto  alignments = geometry::alignmentsFromValue(inputs.alignment, images.size());
        // An empty outer list means "whole image"; per-image empty groups mean
        // "leave this image unchanged".
        const bool  wholeImage = records.empty();
        const auto *masks      = optionalMask(inputs.mask);
        if (masks && masks->size() != images.size())
            fail(CommonToolError::InputOutOfRange, "mask 外层长度必须与图像数量一致");

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
                    fail(CommonToolError::UnsupportedImage, "mask 每张图必须是单通道 8 位图像");
                if (external->size() != image.size())
                    fail(CommonToolError::InputOutOfRange, "mask 尺寸必须与图像一致");
            }
            const auto toReference = alignments[index].inverse();
            if (!toReference)
                fail(GeometryError::AlignmentInvalid, "线性部分不可逆");
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
        outputs.appliedMask = visionrt::DataRef<visionrt::Image>(std::move(appliedMasks));
    }
    catch (const visionrt::ContractError &)
    {
        throw;
    }
    catch (const ToolFailure &error)
    {
        outputs.images.reset();
        outputs.appliedMask.reset();
        outputs.status = status(error);
    }
    catch (const std::exception &error)
    {
        outputs.images.reset();
        outputs.appliedMask.reset();
        outputs.status = status(CommonToolError::OperatorFailed, "调整亮度失败", error.what());
    }
    catch (...)
    {
        outputs.images.reset();
        outputs.appliedMask.reset();
        outputs.status = status(CommonToolError::OperatorFailed, "调整亮度失败：未知异常");
    }
    return outputs.pack();
}

} // namespace visiontools::preprocess
