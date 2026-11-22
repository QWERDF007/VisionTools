#include "RecognizeColorTool.h"

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
namespace p = ports::recognize_color;

visionrt::Values missingOutputs()
{
    return missingPorts({p::out::IMAGES, p::out::REGIONS, p::out::COLORS, p::out::SCORES,
                         p::out::RATIOS, p::out::CENTER_X, p::out::CENTER_Y});
}

struct DetectionCircle
{
    cv::Point2d center;
    double      radius = 0;
};

// Circles are trained in reference coordinates. One group is broadcast over every image, which is
// how a single trained reference frame serves a whole batch; a per-image group is also accepted.
std::vector<DetectionCircle> circlesFor(const visionrt::ROI &groups, std::size_t image)
{
    const auto &records = groups.size() == 1 ? groups.front() : groups[image];
    std::vector<DetectionCircle> circles;
    circles.reserve(records.size());
    for (const auto &record : records)
    {
        const visionrt::RegionView view(record);
        if (view.kind() != visionrt::RoiKind::Circle)
            fail(RecognitionError::DetectionRegionNotCircle, "检测区域第 " + std::to_string(circles.size() + 1) + " 项");
        const auto circle = view.circle();
        if (!(circle.radius() > 0.0))
            fail(CommonToolError::InputOutOfRange, "检测区域", "圆半径必须为正数");
        circles.push_back({circle.center(), circle.radius()});
    }
    if (circles.empty())
        fail(RecognitionError::DetectionRegionMissing, "第 " + std::to_string(image + 1) + " 张图");
    return circles;
}

std::vector<std::uint8_t> readModel(const std::filesystem::path &path)
{
    const auto    text = path.generic_u8string();
    std::ifstream file(path, std::ios::binary);
    if (!file)
        fail(RecognitionError::ColorModelUnreadable,
             std::string(reinterpret_cast<const char *>(text.data()), text.size()), "文件不存在或无法打开");
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}
} // namespace

visionrt::ToolStatus RecognizeColorTool::initialize(const visionrt::Values &parameters)
{
    try
    {
        const auto found = parameters.find(p::init::MODEL);
        if (found == parameters.end() || !std::holds_alternative<std::string>(found->second))
            return status(RecognitionError::ColorModelPathMissing, "初始化参数 1");
        const auto &path = std::get<std::string>(found->second);
        if (path.empty())
            return status(RecognitionError::ColorModelPathMissing, "初始化参数 1");
        model_ = ColorModel::decode(readModel(common::resolveResourcePath(path)));
        return {};
    }
    catch (const ToolFailure &error)
    {
        return status(error);
    }
    catch (const std::exception &error)
    {
        return status(CommonToolError::OperatorFailed, "加载颜色模型失败", error.what());
    }
}

visionrt::ToolResult RecognizeColorTool::execute(const visionrt::Values &inputs)
{
    if (const auto upstream = upstreamStatus(inputs); upstream != 0)
        return skippedByUpstream(upstream, missingOutputs());
    if (model_.empty())
        return failure(missingOutputs(), RecognitionError::ColorModelPathMissing, "颜色模型未加载");

    Readings   readings(inputs, p::TOOL_ID);
    const auto imagesRef = readings.image(p::in::IMAGES);
    const auto regions   = readings.roi(p::in::REGIONS);
    const auto minScore  = readings.real(p::in::MIN_SCORE);
    const auto minRatio  = readings.real(p::in::MIN_RATIO);
    const auto alignment = readings.find(p::in::ALIGNMENT);
    if (readings.failed())
        return visionrt::toolResult(missingOutputs(),
                                    readings.status(CommonToolError::InputMissing, CommonToolError::InputTypeMismatch));

    try
    {
        if (!(minScore >= 0.0) || !(minScore <= 1.0))
            fail(CommonToolError::InputOutOfRange, "最低分数", "必须在 [0, 1] 内");
        if (!(minRatio >= 0.0) || !(minRatio <= 1.0))
            fail(CommonToolError::InputOutOfRange, "最低分类像素比例", "必须在 [0, 1] 内");

        const auto &images = imagesRef.get();
        if (images.empty())
            fail(CommonToolError::InputMissing, "图像");
        if (alignment == nullptr)
            fail(CommonToolError::InputMissing, "对齐数据");
        if (regions.get().empty())
            fail(RecognitionError::DetectionRegionMissing, "检测区域");
        if (regions.get().size() != 1 && regions.get().size() != images.size())
            fail(CommonToolError::InputOutOfRange, "检测区域", "需要一组或与图像数量一致");

        const auto alignments = geometry::alignmentsFromValue(*alignment, images.size());

        visionrt::NestedVector<std::string, 2>  colors;
        visionrt::NestedVector<float, 2>        scores, ratios, centersX, centersY;
        std::vector<std::vector<roi::Record>>   currentCircles;
        colors.reserve(images.size());
        scores.reserve(images.size());
        ratios.reserve(images.size());
        centersX.reserve(images.size());
        centersY.reserve(images.size());
        currentCircles.reserve(images.size());

        for (std::size_t index = 0; index < images.size(); ++index)
        {
            const auto &image = images[index];
            requireSupportedImage(image, "识别颜色");
            if (image.type() != CV_8UC3)
                fail(RecognitionError::UnsupportedColorImage, "第 " + std::to_string(index + 1) + " 张图");

            const ChromaImage chroma{image};
            const auto       &alignmentForImage = alignments[index];
            const auto  circles           = circlesFor(regions.get(), index);
            std::vector<std::string> imageColors;
            std::vector<float>       imageScores, imageRatios, imageX, imageY;
            std::vector<roi::Record> imageCircles;
            imageColors.reserve(circles.size());
            imageScores.reserve(circles.size());
            imageRatios.reserve(circles.size());
            imageX.reserve(circles.size());
            imageY.reserve(circles.size());
            imageCircles.reserve(circles.size());

            for (const auto &circle : circles)
            {
                double centerX = 0;
                double centerY = 0;
                alignmentForImage.apply(circle.center.x, circle.center.y, centerX, centerY);
                const double radius = circle.radius * alignmentForImage.scale();
                imageX.push_back(static_cast<float>(centerX));
                imageY.push_back(static_cast<float>(centerY));
                imageCircles.push_back(roi::circle(centerX, centerY, radius));

                const cv::Mat mask
                    = roi::regionMask({imageCircles.back()}, image.size(), geometry::Alignment::identity());
                const auto classification = model_.classify(chroma, mask, minScore, minRatio);
                imageColors.push_back(classification.label);
                imageScores.push_back(static_cast<float>(classification.score));
                imageRatios.push_back(static_cast<float>(classification.ratio));
            }

            colors.push_back(std::move(imageColors));
            scores.push_back(std::move(imageScores));
            ratios.push_back(std::move(imageRatios));
            centersX.push_back(std::move(imageX));
            centersY.push_back(std::move(imageY));
            currentCircles.push_back(std::move(imageCircles));
        }

        visionrt::Values outputs;
        outputs.emplace(p::out::IMAGES, visionrt::DataRef<visionrt::Image>(visionrt::Image(images), {imagesRef.owner()}));
        outputs.emplace(p::out::REGIONS, roi::roiValue(std::move(currentCircles)));
        outputs.emplace(p::out::COLORS, visionrt::Array::fromNested(std::move(colors)));
        outputs.emplace(p::out::SCORES, visionrt::Array::fromNested(std::move(scores)));
        outputs.emplace(p::out::RATIOS, visionrt::Array::fromNested(std::move(ratios)));
        outputs.emplace(p::out::CENTER_X, visionrt::Array::fromNested(std::move(centersX)));
        outputs.emplace(p::out::CENTER_Y, visionrt::Array::fromNested(std::move(centersY)));
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
        return failure(missingOutputs(), CommonToolError::OperatorFailed, "识别颜色失败", error.what());
    }
}

visionrt::ToolStatus RecognizeColorTool::release()
{
    model_ = {};
    return {};
}

} // namespace visiontools::recognition
