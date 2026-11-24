#include "RecognizeColorTool.h"

#include "ColorModel.h"
#include "common/ToolOutputs.h"
#include "common/errors/ToolError.h"
#include "errors/RecognitionErrors.h"

#include <opencv2/imgproc.hpp>

#include <map>
#include <string>
#include <vector>

namespace visiontools::recognition {
namespace {
namespace p   = ports::recognize_color;
using FieldId = visionrt::TrainingFieldId;

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

visionrt::ToolResult RecognizeColorTool::train(const visionrt::TrainingSet &training, const visionrt::Values &parameters)
{
    constexpr auto labelField = p::train::LABEL;
    constexpr auto regionField = p::train::REGIONS;
    try
    {
        visionrt::validateTrainingSet(training);
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
                                   : std::get_if<visionrt::DataRef<visionrt::ROI>>(&region->second);
            if (!roi || roi->get().empty())
                fail(RecognitionError::ColorSamplesMissing, "记录 " + std::to_string(id));

            auto frame = frames.find(record.imageId);
            if (frame == frames.end())
            {
                const auto image = training.images.find(record.imageId);
                if (image == training.images.end())
                    fail(RecognitionError::ColorSamplesMissing, "记录 " + std::to_string(id) + " 的图像");
                frame = frames.emplace(record.imageId, bgrFrame(image->second)).first;
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

        const auto       bytes = ColorModel::train(samples).encode();
        visionrt::Values outputs;
        outputs.emplace(p::out::MODEL_PATH, std::get<std::string>(parameters.at(p::train::MODEL_PATH)));
        auto result = visionrt::toolResult(std::move(outputs), {0, "训练成功"});
        result.artifacts.emplace("color-model", visionrt::ModelArtifact{bytes});
        return result;
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
