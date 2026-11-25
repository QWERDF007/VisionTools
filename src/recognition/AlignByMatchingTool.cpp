#include "AlignByMatchingTool.h"

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

#include <visionrt/contracts/TrainingArchive.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

namespace visiontools::recognition {
namespace {
namespace p = ports::align_by_matching;

cv::Rect boundingRectFromRegions(const std::vector<roi::Record> &records, cv::Size imageSize)
{
    if (records.empty())
        return {0, 0, imageSize.width, imageSize.height};
    double minX = imageSize.width;
    double minY = imageSize.height;
    double maxX = 0.0;
    double maxY = 0.0;
    for (const auto &record : records)
    {
        const auto rect = visionrt::RegionView(record).rectangle();
        for (const auto &corner : rect.corners())
        {
            minX = std::min(minX, corner.x);
            minY = std::min(minY, corner.y);
            maxX = std::max(maxX, corner.x);
            maxY = std::max(maxY, corner.y);
        }
    }
    const int x0 = std::clamp(static_cast<int>(std::floor(minX)), 0, imageSize.width);
    const int y0 = std::clamp(static_cast<int>(std::floor(minY)), 0, imageSize.height);
    const int x1 = std::clamp(static_cast<int>(std::ceil(maxX)), 0, imageSize.width);
    const int y1 = std::clamp(static_cast<int>(std::ceil(maxY)), 0, imageSize.height);
    return {x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0)};
}
} // namespace

visionrt::Values AlignByMatchingTool::missingOutputs()
{
    return missingPorts({p::out::ALIGNMENTS, p::out::SCORES, p::out::MATCH_REGIONS, p::out::MODEL_PATH});
}

visionrt::ToolStatus AlignByMatchingTool::initialize(const visionrt::ToolConfiguration &configuration)
{
    try
    {
        const auto found = configuration.models.find(p::MODEL_ARTIFACT);
        if (found == configuration.models.end() || found->second.empty())
            return status(RecognitionError::MatchModelPathMissing, "对齐模型路径");
        const auto utf8 = found->second.generic_u8string();
        modelPath_.assign(reinterpret_cast<const char *>(utf8.data()), utf8.size());
        if (!std::filesystem::is_regular_file(found->second))
            return status(RecognitionError::MatchModelUnreadable, modelPath_);
        std::vector<std::uint8_t> bytes;
        try
        {
            bytes = visionrt::TrainingArchive::readModel(found->second, p::MODEL_ARTIFACT);
        }
        catch (const visionrt::ContractError &error)
        {
            return status(RecognitionError::MatchModelIncompatible, modelPath_, error.what());
        }
        model_ = TemplateModel::decode(bytes);
        return {};
    }
    catch (const ToolFailure &error)
    {
        return status(error);
    }
    catch (const std::exception &error)
    {
        return status(CommonToolError::OperatorFailed, "加载对齐匹配模型失败", error.what());
    }
}

visionrt::ToolResult AlignByMatchingTool::train(const visionrt::Values &parameters)
{
    Readings readings(parameters, p::TOOL_ID, visionrt::PortStage::Train);
    const auto modelPath = readings.text(p::train::MODEL_PATH);
    const auto methodName = readings.text(p::train::METHOD);
    const auto modelContrast = readings.real(p::train::MODEL_CONTRAST);
    const auto searchContrast = readings.real(p::train::SEARCH_CONTRAST);
    const auto angleTolerance = readings.real(p::train::ANGLE_TOLERANCE);
    const auto angleStep = readings.real(p::train::ANGLE_STEP);
    const auto polarity = readings.text(p::train::POLARITY);
    const auto numFeatures = readings.integer(p::train::NUM_FEATURES);
    const auto setRef = readings.trainingSet(p::train::DATASET);
    if (readings.failed())
        return visionrt::toolResult(missingOutputs(),
                                    readings.status(CommonToolError::InputMissing, CommonToolError::InputTypeMismatch));
    try
    {
        if (modelPath.empty())
            fail(RecognitionError::MatchModelPathMissing, "对齐模型路径不能为空");
        if (!std::isfinite(modelContrast) || modelContrast < 1.0 || modelContrast > 255.0)
            fail(CommonToolError::InputOutOfRange, "模型边缘对比度必须在 [1, 255] 内");
        if (!std::isfinite(searchContrast) || searchContrast < 1.0 || searchContrast > 255.0)
            fail(CommonToolError::InputOutOfRange, "搜索边缘对比度必须在 [1, 255] 内");
        if (!std::isfinite(angleTolerance) || angleTolerance < 0.0 || angleTolerance > 180.0)
            fail(CommonToolError::InputOutOfRange, "方向容差必须在 [0, 180] 内");
        if (!std::isfinite(angleStep) || angleStep < 0.1 || angleStep > 45.0)
            fail(CommonToolError::InputOutOfRange, "角度步长必须在 [0.1, 45] 内");
        if (polarity != "use_polarity" && polarity != "ignore_polarity")
            fail(CommonToolError::InputOutOfRange, "极性策略只接受 use_polarity 或 ignore_polarity");
        if (numFeatures < 16 || numFeatures > 2048)
            fail(CommonToolError::InputOutOfRange, "特征点数量必须在 [16, 2048] 内");

        const auto &training = setRef.get();
        visionrt::validateTrainingSetPixels(training);
        if (training.records.size() != 1)
            fail(RecognitionError::MatchTrainingInvalid, "训练需要恰好一条包含参考图与模板区域的记录");
        const auto &record = training.records.begin()->second;
        if (!record.imageId)
            fail(RecognitionError::MatchTrainingInvalid, "训练记录缺少参考图像");
        const auto imgIt = training.images.find(*record.imageId);
        if (imgIt == training.images.end())
            fail(RecognitionError::MatchTrainingInvalid, "训练记录引用的参考图像不存在");
        requireSupportedImage(imgIt->second.pixels, "通过匹配确定对齐训练");

        const auto fieldIt = record.fields.find(p::field::TEMPLATE_ROI);
        const auto *rois = fieldIt == record.fields.end() ? nullptr : std::get_if<visionrt::Rectangle>(&fieldIt->second);
        if (!rois || rois->get().size() != 1 || rois->get().front().size() != 1)
            fail(RecognitionError::MatchTrainingInvalid, "训练记录需要恰好一个模板矩形区域");

        const auto rect = visionrt::RegionView(rois->get().front().front()).rectangle();
        TemplateTrainOptions options;
        options.method = parseMatchingMethod(methodName);
        options.modelContrast = modelContrast;
        options.searchContrast = searchContrast;
        options.angleToleranceDeg = angleTolerance;
        options.angleStepDeg = angleStep;
        options.ignorePolarity = (polarity == "ignore_polarity");
        options.numFeatures = static_cast<int>(numFeatures);

        const cv::Mat gray = edge::grayscale(imgIt->second.pixels);
        const auto trainedModel = TemplateModel::trainRectangle(
            gray, rect.center(), rect.width(), rect.height(), rect.angle(), options);

        std::vector<visionrt::TrainingArchiveImage> archiveImages;
        archiveImages.reserve(training.images.size());
        for (const auto &[id, item] : training.images)
            archiveImages.push_back({p::train::DATASET, id, item.colorOrder, item.pixels});

        visionrt::TrainingArchive::write(common::resolveResourcePath(modelPath),
                                         {{p::MODEL_ARTIFACT, trainedModel.encode()}}, archiveImages);

        visionrt::Values outputs;
        outputs.emplace(p::out::MODEL_PATH, modelPath);
        outputs.emplace(p::out::MATCH_REGIONS,
                        roi::value<visionrt::Rectangle>({{roi::rectangle(rect.center().x, rect.center().y,
                                                                         rect.width(), rect.height(), rect.angle())}}));
        return visionrt::toolResult(std::move(outputs), visionrt::executionSuccess("匹配对齐模型训练成功"));
    }
    catch (const ToolFailure &error)
    {
        return failureResult(missingOutputs(), error);
    }
    catch (const std::exception &error)
    {
        return failure(missingOutputs(), CommonToolError::OperatorFailed, "匹配对齐模型训练失败", error.what());
    }
}

visionrt::ToolResult AlignByMatchingTool::execute(const visionrt::Values &inputs)
{
    if (const auto upstream = upstreamStatus(inputs); upstream != 0)
        return skippedByUpstream(upstream, missingOutputs());
    if (model_.empty())
        return failure(missingOutputs(), RecognitionError::MatchModelPathMissing, "对齐模型未加载");

    Readings readings(inputs, p::TOOL_ID);
    const auto imagesRef = readings.image(p::in::IMAGES);
    const auto searchRoiRef = readings.rectangle(p::in::SEARCH_ROI);
    const auto minScore = readings.real(p::in::MIN_SCORE);
    const auto angleTolerance = readings.real(p::in::ANGLE_TOLERANCE);
    if (readings.failed())
        return visionrt::toolResult(missingOutputs(),
                                    readings.status(CommonToolError::InputMissing, CommonToolError::InputTypeMismatch));
    try
    {
        if (!std::isfinite(minScore) || minScore < 0.0 || minScore > 1.0)
            fail(CommonToolError::InputOutOfRange, "最低分数必须在 [0, 1] 内");
        if (!std::isfinite(angleTolerance) || angleTolerance < 0.0 || angleTolerance > 180.0)
            fail(CommonToolError::InputOutOfRange, "方向容差必须在 [0, 180] 内");

        const auto &images = imagesRef.get();
        if (images.empty())
            fail(CommonToolError::InputMissing, "图像");
        const auto searchGroups = roi::groups(searchRoiRef, images.size(), "搜索区域");

        std::vector<geometry::Alignment> alignments;
        std::vector<double> scores;
        std::vector<std::vector<roi::Record>> matchRegions;
        alignments.reserve(images.size());
        scores.reserve(images.size());
        matchRegions.reserve(images.size());

        TemplateMatchOptions matchOpts;
        matchOpts.minScore = minScore;
        matchOpts.angleToleranceDeg = std::min(angleTolerance, model_.trainOptions().angleToleranceDeg);
        matchOpts.angleCenterDeg = 0.0;
        matchOpts.maxOverlap = 0.3;
        matchOpts.maxResults = 1;
        matchOpts.refineSubpixel = true;

        for (std::size_t index = 0; index < images.size(); ++index)
        {
            requireSupportedImage(images[index], "通过匹配确定对齐");
            if (images[index].depth() != CV_8U)
                fail(CommonToolError::UnsupportedImage, "第 " + std::to_string(index + 1) + " 张图不是 8 位图像");
            const cv::Mat gray = edge::grayscale(images[index]);
            const cv::Rect roi = searchGroups.empty()
                ? cv::Rect(0, 0, gray.cols, gray.rows)
                : boundingRectFromRegions(searchGroups[index], gray.size());

            const auto hits = model_.match(gray, roi, matchOpts);
            if (hits.empty())
                fail(RecognitionError::MatchNotFound, "第 " + std::to_string(index + 1) + " 张图");

            const auto &best = hits.front();
            alignments.push_back(best.alignment);
            scores.push_back(best.score);
            // Hit angle is CCW; ROI rectangle angle is CW -> subtract CCW angle.
            const double matchedAngle = model_.referenceAngleDeg() - best.angleDeg;
            matchRegions.push_back({roi::rectangle(best.center.x, best.center.y,
                                                   model_.referenceSize().width, model_.referenceSize().height,
                                                   matchedAngle)});
        }

        visionrt::Values outputs;
        outputs.emplace(p::out::ALIGNMENTS, geometry::alignmentArray(alignments));
        outputs.emplace(p::out::SCORES, visionrt::Array::fromNested(std::move(scores)));
        outputs.emplace(p::out::MATCH_REGIONS, roi::value<visionrt::Rectangle>(std::move(matchRegions)));
        outputs.emplace(p::out::MODEL_PATH, modelPath_);
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
        return failure(missingOutputs(), CommonToolError::OperatorFailed, "通过匹配确定对齐失败", error.what());
    }
}

visionrt::ToolStatus AlignByMatchingTool::release()
{
    model_ = {};
    modelPath_.clear();
    return {};
}

} // namespace visiontools::recognition
