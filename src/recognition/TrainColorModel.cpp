#include "RecognizeColorTool.h"

#include "ColorModel.h"
#include "common/InputValues.h"
#include "common/ProductPaths.h"
#include "common/ToolOutputs.h"
#include "common/errors/ToolError.h"
#include "errors/RecognitionErrors.h"
#include <visionrt/contracts/TrainingArchive.h>

#include <opencv2/imgproc.hpp>

#include <map>
#include <string>
#include <vector>

namespace visiontools::recognition {
namespace {
namespace p   = ports::recognize_color;

visionrt::Values missingOutputs()
{
    return missingPorts({p::out::MODEL_PATH});
}

// Training frames keep the order they arrived in; the model itself works in BGR. Normalizing here
// is the only place that knows the source order, so neither the model nor its file carries it.
cv::Mat bgrFrame(const visionrt::TrainingImage &image)
{
    switch (image.colorOrder)
    {
    case visionrt::ColorOrder::RGB:
    {
        cv::Mat bgr;
        cv::cvtColor(image.pixels, bgr, cv::COLOR_RGB2BGR);
        return bgr;
    }
    case visionrt::ColorOrder::RGBA:
    {
        cv::Mat bgr;
        cv::cvtColor(image.pixels, bgr, cv::COLOR_RGBA2BGR);
        return bgr;
    }
    case visionrt::ColorOrder::BGRA:
    {
        cv::Mat bgr;
        cv::cvtColor(image.pixels, bgr, cv::COLOR_BGRA2BGR);
        return bgr;
    }
    default:
        return image.pixels;
    }
}
} // namespace

visionrt::ToolResult RecognizeColorTool::train(const visionrt::Values &parameters)
{
    Readings readings(parameters, p::TOOL_ID, visionrt::PortStage::Train);
    const auto set = readings.trainingSet(p::train::DATASET);
    const auto modelPath = readings.text(p::train::MODEL_PATH);
    if (readings.failed()) return visionrt::toolResult(missingOutputs(), readings.status(CommonToolError::InputMissing, CommonToolError::InputTypeMismatch));
    const auto &training = set.get();
    constexpr auto labelField = p::field::LABEL;
    constexpr auto regionField = p::field::REGIONS;
    try
    {
        visionrt::validateTrainingSetPixels(training);
        if (training.records.empty())
            fail(RecognitionError::ColorSamplesMissing, "训练集");

        std::map<visionrt::TrainingImageId, cv::Mat> frames;
        std::vector<ColorModel::Sample>              samples;
        samples.reserve(training.records.size());
        for (const auto &[id, record] : training.records)
        {
            const auto  label = record.fields.find(labelField);
            const auto *text  = label == record.fields.end() ? nullptr : std::get_if<std::string>(&label->second);
            if (!text || text->empty())
                fail(RecognitionError::ColorClassMissing, "记录 " + std::to_string(id));
            const auto  region = record.fields.find(regionField);
            const auto *roi    = region == record.fields.end()
                                   ? nullptr
                                   : std::get_if<visionrt::Rectangle>(&region->second);
            if (!roi || roi->get().empty())
                fail(RecognitionError::ColorSamplesMissing, "记录 " + std::to_string(id));

            if (!record.imageId) fail(RecognitionError::ColorSamplesMissing, "记录 " + std::to_string(id) + " 的图像");
            auto frame = frames.find(*record.imageId);
            if (frame == frames.end())
            {
                const auto image = training.images.find(*record.imageId);
                if (image == training.images.end())
                    fail(RecognitionError::ColorSamplesMissing, "记录 " + std::to_string(id) + " 的图像");
                frame = frames.emplace(*record.imageId, bgrFrame(image->second)).first;
            }

            ColorModel::Sample sample;
            sample.label  = *text;
            sample.pixels = frame->second;
            for (const auto &group : roi->get())
                sample.regions.insert(sample.regions.end(), group.begin(), group.end());
            if (sample.regions.empty())
                fail(RecognitionError::ColorSamplesMissing, "记录 " + std::to_string(id));
            samples.push_back(std::move(sample));
        }

        const auto bytes = ColorModel::train(samples).encode();
        if (modelPath.empty())
            fail(RecognitionError::ColorModelPathMissing, "模型路径");
        std::vector<visionrt::TrainingArchiveImage> archiveImages;
        archiveImages.reserve(training.images.size());
        for (const auto &[id, item] : training.images)
            archiveImages.push_back({p::train::DATASET, id, item.colorOrder, item.pixels});
        visionrt::TrainingArchive::write(common::resolveResourcePath(modelPath),
                                         {{"color-model", bytes}}, archiveImages);
        visionrt::Values outputs;
        outputs.emplace(p::out::MODEL_PATH, modelPath);
        return visionrt::toolResult(std::move(outputs), {0, "训练成功"});
    }
    catch (const ToolFailure &error)
    {
        return failureResult(missingOutputs(), error);
    }
    catch (const std::exception &error)
    {
        return failure(missingOutputs(), CommonToolError::OperatorFailed, "颜色模型训练失败", error.what());
    }
}

} // namespace visiontools::recognition
