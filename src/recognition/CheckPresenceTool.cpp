#include "CheckPresenceTool.h"

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
#include <cstring>
#include <filesystem>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

namespace visiontools::recognition {
namespace {
namespace p = ports::check_presence;

constexpr char kPresenceMagic[4] = {'V', 'P', 'M', '1'};
constexpr std::uint32_t kPresenceVersion = 1;
constexpr double kRadToDeg = 180.0 / std::numbers::pi;

void writeU32(std::vector<std::uint8_t> &out, std::uint32_t v)
{
    for (int i = 0; i < 4; ++i)
        out.push_back(static_cast<std::uint8_t>((v >> (i * 8)) & 0xFFu));
}

void writeF64(std::vector<std::uint8_t> &out, double v)
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    for (int i = 0; i < 8; ++i)
        out.push_back(static_cast<std::uint8_t>((bits >> (i * 8)) & 0xFFu));
}

std::vector<std::uint8_t> encodePresenceModel(const TemplateModel &model,
                                              const std::vector<cv::Point2d> &positions)
{
    const auto templateBytes = model.encode();
    std::vector<std::uint8_t> out;
    out.insert(out.end(), std::begin(kPresenceMagic), std::end(kPresenceMagic));
    writeU32(out, kPresenceVersion);
    writeU32(out, static_cast<std::uint32_t>(templateBytes.size()));
    out.insert(out.end(), templateBytes.begin(), templateBytes.end());
    writeU32(out, static_cast<std::uint32_t>(positions.size()));
    for (const auto &pt : positions)
    {
        writeF64(out, pt.x);
        writeF64(out, pt.y);
    }
    return out;
}

void decodePresenceModel(std::span<const std::uint8_t> bytes,
                         TemplateModel &model,
                         std::vector<cv::Point2d> &positions)
{
    std::size_t pos = 0;
    auto require = [&](std::size_t n)
    {
        if (pos + n > bytes.size())
            fail(RecognitionError::MatchModelIncompatible, "存在性模型字节不完整");
    };
    auto readU32 = [&]() -> std::uint32_t
    {
        require(4);
        std::uint32_t v = 0;
        for (int i = 0; i < 4; ++i)
            v |= static_cast<std::uint32_t>(bytes[pos++]) << (i * 8);
        return v;
    };
    auto readF64 = [&]() -> double
    {
        require(8);
        std::uint64_t bits = 0;
        for (int i = 0; i < 8; ++i)
            bits |= static_cast<std::uint64_t>(bytes[pos++]) << (i * 8);
        double v = 0.0;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    };

    require(4);
    if (std::memcmp(bytes.data(), kPresenceMagic, 4) != 0)
        fail(RecognitionError::MatchModelIncompatible, "存在性模型魔数无效");
    pos = 4;
    if (readU32() != kPresenceVersion)
        fail(RecognitionError::MatchModelIncompatible, "存在性模型版本不受支持");

    const std::uint32_t tmLen = readU32();
    require(tmLen);
    model = TemplateModel::decode(bytes.subspan(pos, tmLen));
    pos += tmLen;

    const std::uint32_t count = readU32();
    if (count == 0)
        fail(RecognitionError::MatchModelIncompatible, "存在性模型缺少参考位置");
    positions.clear();
    positions.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i)
    {
        const double x = readF64();
        const double y = readF64();
        if (!std::isfinite(x) || !std::isfinite(y))
            fail(RecognitionError::MatchModelIncompatible, "存在性模型包含非法参考坐标");
        positions.emplace_back(x, y);
    }
}
} // namespace

visionrt::Values CheckPresenceTool::missingOutputs()
{
    return missingPorts({p::out::MODEL_PATH, p::out::OBJECT_ACCEPTED, p::out::ALL_ACCEPTED,
                         p::out::ABSOLUTE_DISTANCE, p::out::RELATIVE_DISTANCE,
                         p::out::ACCEPTED_OBJECTS, p::out::REJECTED_POSITIONS, p::out::REFERENCE_POSITIONS});
}

visionrt::ToolStatus CheckPresenceTool::initialize(const visionrt::ToolConfiguration &configuration)
{
    try
    {
        const auto found = configuration.models.find(p::MODEL_ARTIFACT);
        if (found == configuration.models.end() || found->second.empty())
            return status(RecognitionError::MatchModelPathMissing, "存在性模型路径");
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
        decodePresenceModel(bytes, objectModel_, referencePositions_);
        return {};
    }
    catch (const ToolFailure &error)
    {
        return status(error);
    }
    catch (const std::exception &error)
    {
        return status(CommonToolError::OperatorFailed, "加载存在性匹配模型失败", error.what());
    }
}

visionrt::ToolResult CheckPresenceTool::train(const visionrt::Values &parameters)
{
    Readings readings(parameters, p::TOOL_ID, visionrt::PortStage::Train);
    const auto modelPath = readings.text(p::train::MODEL_PATH);
    const auto methodName = readings.text(p::train::METHOD);
    const auto trainMinScore = readings.real(p::train::MIN_SCORE);
    const auto trainAngleTol = readings.real(p::train::ANGLE_TOLERANCE);
    const auto trainMaxOverlap = readings.real(p::train::MAX_OVERLAP);
    const auto modelContrast = readings.real(p::train::MODEL_CONTRAST);
    const auto objSetRef = readings.trainingSet(p::train::OBJECT_DATASET);
    const auto posSetRef = readings.trainingSet(p::train::POSITION_DATASET);
    if (readings.failed())
        return visionrt::toolResult(missingOutputs(),
                                    readings.status(CommonToolError::InputMissing, CommonToolError::InputTypeMismatch));
    try
    {
        if (modelPath.empty())
            fail(RecognitionError::MatchModelPathMissing, "存在性模型路径不能为空");
        if (!std::isfinite(trainMinScore) || trainMinScore < 0.05 || trainMinScore > 1.0)
            fail(CommonToolError::InputOutOfRange, "训练最低分数必须在 [0.05, 1] 内");
        if (!std::isfinite(trainAngleTol) || trainAngleTol < 0.0 || trainAngleTol > 180.0)
            fail(CommonToolError::InputOutOfRange, "训练方向容差必须在 [0, 180] 内");
        if (!std::isfinite(trainMaxOverlap) || trainMaxOverlap < 0.0 || trainMaxOverlap > 100.0)
            fail(CommonToolError::InputOutOfRange, "最大重叠比例必须在 [0, 100] 内");
        if (!std::isfinite(modelContrast) || modelContrast < 1.0 || modelContrast > 255.0)
            fail(CommonToolError::InputOutOfRange, "模型边缘对比度必须在 [1, 255] 内");

        const auto &objSet = objSetRef.get();
        visionrt::validateTrainingSetPixels(objSet);
        if (objSet.records.size() != 1)
            fail(RecognitionError::MatchTrainingInvalid, "对象模板样本需要恰好一条记录");
        const auto &objRec = objSet.records.begin()->second;
        if (!objRec.imageId)
            fail(RecognitionError::MatchTrainingInvalid, "对象模板样本缺少参考图像");
        const auto objImgIt = objSet.images.find(*objRec.imageId);
        if (objImgIt == objSet.images.end())
            fail(RecognitionError::MatchTrainingInvalid, "对象模板样本引用的图像不存在");
        requireSupportedImage(objImgIt->second.pixels, "存在性对象模板训练");

        const auto objFieldIt = objRec.fields.find(p::field::OBJECT_ROI);
        const auto *objRois = objFieldIt == objRec.fields.end() ? nullptr : std::get_if<visionrt::Circle>(&objFieldIt->second);
        if (!objRois || objRois->get().size() != 1 || objRois->get().front().size() != 1)
            fail(RecognitionError::MatchTrainingInvalid, "对象模板样本需要恰好一个圆形模板区域");
        const auto objCircle = visionrt::RegionView(objRois->get().front().front()).circle();

        TemplateTrainOptions trainOpts;
        trainOpts.method = parseMatchingMethod(methodName);
        trainOpts.modelContrast = modelContrast;
        trainOpts.searchContrast = std::max(2.0, modelContrast * 0.5);
        trainOpts.angleToleranceDeg = trainAngleTol;
        trainOpts.angleStepDeg = std::max(1.0, std::min(5.0, trainAngleTol > 0.0 ? trainAngleTol * 0.5 : 1.0));
        trainOpts.ignorePolarity = false;
        trainOpts.numFeatures = 64;

        const cv::Mat objGray = edge::grayscale(objImgIt->second.pixels);
        const auto model = TemplateModel::trainCircle(objGray, objCircle.center(), objCircle.radius(), trainOpts);

        // Dataset 2: reference positions (explicit points or automatic template search on reference image).
        const auto &posSet = posSetRef.get();
        visionrt::validateTrainingSetPixels(posSet);
        std::vector<cv::Point2d> refPositions;
        const cv::Mat *posImagePtr = &objImgIt->second.pixels;
        if (!posSet.records.empty())
        {
            if (posSet.records.size() > 1)
                fail(RecognitionError::MatchTrainingInvalid, "参考位置样本最多包含一条记录");
            const auto &posRec = posSet.records.begin()->second;
            if (posRec.imageId)
            {
                const auto posImgIt = posSet.images.find(*posRec.imageId);
                if (posImgIt == posSet.images.end())
                    fail(RecognitionError::MatchTrainingInvalid, "参考位置样本引用的图像不存在");
                requireSupportedImage(posImgIt->second.pixels, "参考位置训练");
                posImagePtr = &posImgIt->second.pixels;
            }
            const auto posFieldIt = posRec.fields.find(p::field::REFERENCE_POINTS);
            if (posFieldIt != posRec.fields.end())
            {
                const auto *pts = std::get_if<visionrt::Point>(&posFieldIt->second);
                if (pts)
                {
                    for (const auto &group : pts->get())
                        for (const auto &rec : group)
                        {
                            const auto pv = visionrt::RegionView(rec).point();
                            for (const auto &pt : pv.points())
                                refPositions.push_back(pt);
                        }
                }
            }
        }

        if (refPositions.empty())
        {
            const cv::Mat posGray = edge::grayscale(*posImagePtr);
            TemplateMatchOptions discoverOpts;
            discoverOpts.minScore = trainMinScore;
            discoverOpts.angleToleranceDeg = trainAngleTol;
            discoverOpts.angleCenterDeg = 0.0;
            discoverOpts.maxOverlap = trainMaxOverlap / 100.0;
            discoverOpts.maxResults = 0;
            discoverOpts.refineSubpixel = true;

            const auto found = model.match(posGray, cv::Rect(0, 0, posGray.cols, posGray.rows), discoverOpts);
            if (found.empty())
                fail(RecognitionError::MatchTrainingInvalid, "在参考图上未能检测到任何参考位置");
            refPositions.reserve(found.size());
            for (const auto &hit : found)
                refPositions.push_back(hit.center);
            // Stable spatial order (left-to-right, then top-to-bottom).
            std::sort(refPositions.begin(), refPositions.end(), [](const cv::Point2d &a, const cv::Point2d &b)
            {
                if (std::abs(a.x - b.x) > 8.0) return a.x < b.x;
                return a.y < b.y;
            });
        }

        std::vector<visionrt::TrainingArchiveImage> archiveImages;
        archiveImages.reserve(objSet.images.size() + posSet.images.size());
        for (const auto &[id, item] : objSet.images)
            archiveImages.push_back({p::train::OBJECT_DATASET, id, item.colorOrder, item.pixels});
        for (const auto &[id, item] : posSet.images)
            archiveImages.push_back({p::train::POSITION_DATASET, id, item.colorOrder, item.pixels});

        visionrt::TrainingArchive::write(common::resolveResourcePath(modelPath),
                                         {{p::MODEL_ARTIFACT, encodePresenceModel(model, refPositions)}},
                                         archiveImages);

        std::vector<roi::Record> pointRecords;
        pointRecords.reserve(refPositions.size());
        for (const auto &pt : refPositions)
            pointRecords.push_back(roi::point(pt.x, pt.y));

        visionrt::Values outputs;
        outputs.emplace(p::out::MODEL_PATH, modelPath);
        outputs.emplace(p::out::REFERENCE_POSITIONS, roi::value<visionrt::Point>({std::move(pointRecords)}));
        return visionrt::toolResult(std::move(outputs),
                                    visionrt::executionSuccess("存在性模型训练成功，共 " + std::to_string(refPositions.size()) + " 个参考位点"));
    }
    catch (const ToolFailure &error)
    {
        return failureResult(missingOutputs(), error);
    }
    catch (const std::exception &error)
    {
        return failure(missingOutputs(), CommonToolError::OperatorFailed, "存在性模型训练失败", error.what());
    }
}

visionrt::ToolResult CheckPresenceTool::execute(const visionrt::Values &inputs)
{
    if (const auto upstream = upstreamStatus(inputs); upstream != 0)
        return skippedByUpstream(upstream, missingOutputs());
    if (objectModel_.empty() || referencePositions_.empty())
        return failure(missingOutputs(), RecognitionError::MatchModelPathMissing, "存在性模型未加载");

    Readings readings(inputs, p::TOOL_ID);
    const auto imagesRef = readings.image(p::in::IMAGES);
    const auto alignmentValue = readings.find(p::in::ALIGNMENTS);
    const auto maxDistance = readings.real(p::in::MAX_DISTANCE);
    const auto minScore = readings.real(p::in::MIN_SCORE);
    const auto maxOverlap = readings.real(p::in::MAX_OVERLAP);
    const auto failedMode = readings.text(p::in::FAILED_PROCESSING);
    if (readings.failed())
        return visionrt::toolResult(missingOutputs(),
                                    readings.status(CommonToolError::InputMissing, CommonToolError::InputTypeMismatch));
    if (!alignmentValue || visionrt::isMissing(*alignmentValue))
        return failure(missingOutputs(), CommonToolError::InputMissing, "对齐数据");
    try
    {
        if (!std::isfinite(maxDistance) || !(maxDistance > 0.0))
            fail(CommonToolError::InputOutOfRange, "最大允许距离必须大于零");
        if (!std::isfinite(minScore) || minScore < 0.0 || minScore > 1.0)
            fail(CommonToolError::InputOutOfRange, "最低分数必须在 [0, 1] 内");
        if (!std::isfinite(maxOverlap) || maxOverlap < 0.0 || maxOverlap > 100.0)
            fail(CommonToolError::InputOutOfRange, "最大重叠比例必须在 [0, 100] 内");
        if (failedMode != "ignore" && failedMode != "nan")
            fail(CommonToolError::InputOutOfRange, "失败处理只接受 ignore 或 nan");

        const auto &images = imagesRef.get();
        if (images.empty())
            fail(CommonToolError::InputMissing, "图像");
        const auto alignments = geometry::alignmentsFromValue(*alignmentValue, images.size());
        const double radius = objectModel_.radius();
        const double minSep = std::max(2.0, 2.0 * radius * (1.0 - maxOverlap / 100.0));
        const double minSepSq = minSep * minSep;
        const bool outputNan = (failedMode == "nan");

        visionrt::NestedVector<bool, 2> objectAccepted;
        visionrt::NestedVector<bool, 1> allAccepted;
        visionrt::NestedVector<double, 2> absoluteDistance;
        visionrt::NestedVector<double, 2> relativeDistance;
        std::vector<std::vector<roi::Record>> acceptedCircles;
        std::vector<std::vector<roi::Record>> rejectedCircles;
        std::vector<std::vector<roi::Record>> referencePoints;

        objectAccepted.reserve(images.size());
        allAccepted.reserve(images.size());
        absoluteDistance.reserve(images.size());
        relativeDistance.reserve(images.size());
        acceptedCircles.reserve(images.size());
        rejectedCircles.reserve(images.size());
        referencePoints.reserve(images.size());

        for (std::size_t imgIdx = 0; imgIdx < images.size(); ++imgIdx)
        {
            requireSupportedImage(images[imgIdx], "通过匹配检查存在状态");
            if (images[imgIdx].depth() != CV_8U)
                fail(CommonToolError::UnsupportedImage, "第 " + std::to_string(imgIdx + 1) + " 张图不是 8 位图像");
            const cv::Mat gray = edge::grayscale(images[imgIdx]);
            const auto &alignment = alignments[imgIdx];
            const double alignAngleCcw = -std::atan2(alignment.m[3], alignment.m[0]) * kRadToDeg;
            const double alignScale = alignment.scale();

            std::vector<bool> imgAccepted;
            std::vector<double> imgAbsDist;
            std::vector<double> imgRelDist;
            std::vector<roi::Record> imgAcceptedCircles;
            std::vector<roi::Record> imgRejectedCircles;
            std::vector<roi::Record> imgRefPoints;
            std::vector<cv::Point2d> claimedCenters;

            imgAccepted.reserve(referencePositions_.size());
            imgRefPoints.reserve(referencePositions_.size());
            bool imgAllOk = true;

            const int halfWin = static_cast<int>(std::ceil(radius * alignScale + maxDistance + 6.0));
            TemplateMatchOptions localOpts;
            localOpts.minScore = minScore;
            localOpts.angleToleranceDeg = std::min(5.0, objectModel_.trainOptions().angleToleranceDeg);
            localOpts.angleCenterDeg = alignAngleCcw;
            localOpts.maxOverlap = maxOverlap / 100.0;
            localOpts.maxResults = 4;
            localOpts.refineSubpixel = true;

            for (const auto &refPt : referencePositions_)
            {
                double expX = 0.0;
                double expY = 0.0;
                alignment.apply(refPt.x, refPt.y, expX, expY);
                const cv::Point2d expected(expX, expY);
                imgRefPoints.push_back(roi::point(expected.x, expected.y));

                const int x0 = std::max(0, static_cast<int>(std::floor(expected.x)) - halfWin);
                const int y0 = std::max(0, static_cast<int>(std::floor(expected.y)) - halfWin);
                const int x1 = std::min(gray.cols, static_cast<int>(std::ceil(expected.x)) + halfWin);
                const int y1 = std::min(gray.rows, static_cast<int>(std::ceil(expected.y)) + halfWin);
                const cv::Rect localRoi(x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0));

                const auto hits = objectModel_.match(gray, localRoi, localOpts);
                const TemplateHit *matched = nullptr;
                double matchedDist = std::numeric_limits<double>::infinity();
                for (const auto &hit : hits)
                {
                    const double dist = cv::norm(hit.center - expected);
                    if (dist > maxDistance)
                        continue;
                    bool overlapsClaimed = false;
                    for (const auto &claimed : claimedCenters)
                    {
                        const cv::Point2d d = hit.center - claimed;
                        if (d.dot(d) < minSepSq)
                        {
                            overlapsClaimed = true;
                            break;
                        }
                    }
                    if (overlapsClaimed)
                        continue;
                    if (!matched || hit.score > matched->score)
                    {
                        matched = &hit;
                        matchedDist = dist;
                    }
                }

                if (matched)
                {
                    imgAccepted.push_back(true);
                    imgAbsDist.push_back(matchedDist);
                    imgRelDist.push_back(matchedDist / maxDistance);
                    imgAcceptedCircles.push_back(roi::circle(matched->center.x, matched->center.y, radius * alignScale));
                    claimedCenters.push_back(matched->center);
                }
                else
                {
                    imgAccepted.push_back(false);
                    imgAllOk = false;
                    if (outputNan)
                    {
                        const double nan = std::numeric_limits<double>::quiet_NaN();
                        imgAbsDist.push_back(nan);
                        imgRelDist.push_back(nan);
                    }
                    imgRejectedCircles.push_back(roi::circle(expected.x, expected.y, radius * alignScale));
                }
            }

            objectAccepted.push_back(std::move(imgAccepted));
            allAccepted.push_back(imgAllOk);
            absoluteDistance.push_back(std::move(imgAbsDist));
            relativeDistance.push_back(std::move(imgRelDist));
            acceptedCircles.push_back(std::move(imgAcceptedCircles));
            rejectedCircles.push_back(std::move(imgRejectedCircles));
            referencePoints.push_back(std::move(imgRefPoints));
        }

        visionrt::Values outputs;
        outputs.emplace(p::out::MODEL_PATH, modelPath_);
        outputs.emplace(p::out::OBJECT_ACCEPTED, visionrt::Array::fromNested(std::move(objectAccepted)));
        outputs.emplace(p::out::ALL_ACCEPTED, visionrt::Array::fromNested(std::move(allAccepted)));
        outputs.emplace(p::out::ABSOLUTE_DISTANCE, visionrt::Array::fromNested(std::move(absoluteDistance)));
        outputs.emplace(p::out::RELATIVE_DISTANCE, visionrt::Array::fromNested(std::move(relativeDistance)));
        outputs.emplace(p::out::ACCEPTED_OBJECTS, roi::value<visionrt::Circle>(std::move(acceptedCircles)));
        outputs.emplace(p::out::REJECTED_POSITIONS, roi::value<visionrt::Circle>(std::move(rejectedCircles)));
        outputs.emplace(p::out::REFERENCE_POSITIONS, roi::value<visionrt::Point>(std::move(referencePoints)));
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
        return failure(missingOutputs(), CommonToolError::OperatorFailed, "通过匹配检查存在状态失败", error.what());
    }
}

visionrt::ToolStatus CheckPresenceTool::release()
{
    objectModel_ = {};
    referencePositions_.clear();
    modelPath_.clear();
    return {};
}

} // namespace visiontools::recognition
