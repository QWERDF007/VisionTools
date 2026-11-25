#include "RectifyImageTool.h"

#include "common/ImageSupport.h"
#include "common/InputValues.h"
#include "common/PlanarCalibration.h"
#include "common/ToolOutputs.h"
#include "common/UpstreamStatus.h"
#include "common/errors/GeometryErrors.h"
#include "common/errors/ToolError.h"
#include "errors/MetrologyErrors.h"

#include <visionrt/contracts/GeometryValues.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace visiontools::metrology {
namespace {
namespace p = ports::rectify_image;

constexpr int kBoundaryStep = 8;
constexpr int kMaxRectifiedExtent = 16384;
constexpr std::int64_t kMaxRectifiedPixels = 64000000;

visionrt::Values missingOutputs()
{
    return missingPorts({p::out::RECTIFIED_IMAGES, p::out::ADAPTED_CALIBRATION, p::out::PIXEL_SIZE, p::out::COORDINATE_FRAME});
}

struct RectificationOutcome
{
    cv::Mat rectified;
    geometry::PlanarCalibration adapted;
    double pixelSizeMm = 0.0;
    std::vector<double> coordinateFrame;
};

RectificationOutcome rectifySingle(const cv::Mat &source, const geometry::PlanarCalibration &plane)
{
    if (plane.imageSize != source.size())
        fail(CommonToolError::InputOutOfRange, "输入图像尺寸与标定数据不一致");

    const cv::Point2d center((source.cols - 1) * 0.5, (source.rows - 1) * 0.5);
    const cv::Point2d centerMm = plane.map(center);
    const cv::Point2d dx = plane.map(center + cv::Point2d(1.0, 0.0)) - centerMm;
    const cv::Point2d dy = plane.map(center + cv::Point2d(0.0, 1.0)) - centerMm;
    const double dxNorm = cv::norm(dx);
    const double dyNorm = cv::norm(dy);
    const double pixelSizeMm = 0.5 * (dxNorm + dyNorm);
    if (!std::isfinite(dxNorm) || !std::isfinite(dyNorm) || !(dxNorm > 0.0) || !(dyNorm > 0.0)
        || !std::isfinite(pixelSizeMm) || !(pixelSizeMm > 0.0))
        fail(GeometryError::AlignmentInvalid, "标定数据在图像中心的像素尺度无效");

    const cv::Point2d ex = dx * (1.0 / dxNorm);
    const cv::Point2d eyRaw = dy - ex * dy.dot(ex);
    const double eyNorm = cv::norm(eyRaw);
    if (!std::isfinite(eyNorm) || !(eyNorm > 1e-12))
        fail(GeometryError::AlignmentInvalid, "标定映射在图像中心退化");
    const cv::Point2d ey = eyRaw * (1.0 / eyNorm);

    double minS = std::numeric_limits<double>::infinity();
    double maxS = -std::numeric_limits<double>::infinity();
    double minT = std::numeric_limits<double>::infinity();
    double maxT = -std::numeric_limits<double>::infinity();
    const auto includeBoundaryPoint = [&](double x, double y)
    {
        const cv::Point2d world = plane.map({x, y});
        const double s = world.dot(ex);
        const double t = world.dot(ey);
        minS = std::min(minS, s);
        maxS = std::max(maxS, s);
        minT = std::min(minT, t);
        maxT = std::max(maxT, t);
    };

    const double maxX = static_cast<double>(source.cols - 1);
    const double maxY = static_cast<double>(source.rows - 1);
    for (int x = 0; x < source.cols; x += kBoundaryStep)
    {
        includeBoundaryPoint(static_cast<double>(x), 0.0);
        includeBoundaryPoint(static_cast<double>(x), maxY);
    }
    includeBoundaryPoint(maxX, 0.0);
    includeBoundaryPoint(maxX, maxY);
    for (int y = 0; y < source.rows; y += kBoundaryStep)
    {
        includeBoundaryPoint(0.0, static_cast<double>(y));
        includeBoundaryPoint(maxX, static_cast<double>(y));
    }

    const double spanS = maxS - minS;
    const double spanT = maxT - minT;
    if (!std::isfinite(spanS) || !std::isfinite(spanT) || !(spanS > 0.0) || !(spanT > 0.0))
        fail(GeometryError::AlignmentInvalid, "标定数据无法确定有效的校正视野范围");

    const auto rectW = static_cast<std::int64_t>(std::ceil(spanS / pixelSizeMm)) + 1;
    const auto rectH = static_cast<std::int64_t>(std::ceil(spanT / pixelSizeMm)) + 1;
    if (rectW <= 0 || rectH <= 0 || rectW > kMaxRectifiedExtent || rectH > kMaxRectifiedExtent
        || rectW * rectH > kMaxRectifiedPixels)
        fail(MetrologyError::CalibrationFailed, "校正后图像尺寸超出允许上限");

    const int width = static_cast<int>(rectW);
    const int height = static_cast<int>(rectH);
    const cv::Point2d originMm = ex * minS + ey * minT;
    const cv::Point2d stepX = ex * pixelSizeMm;
    const cv::Point2d stepY = ey * pixelSizeMm;

    const cv::Matx33d worldToCorrected = plane.imageToMillimeters.inv();
    const double normalizer = static_cast<double>(std::max(plane.imageSize.width, plane.imageSize.height));
    const double invNormalizer = 1.0 / normalizer;

    cv::Mat mapX(height, width, CV_32FC1);
    cv::Mat mapY(height, width, CV_32FC1);
    bool anyInside = false;

    for (int row = 0; row < height; ++row)
    {
        float *rowMapX = mapX.ptr<float>(row);
        float *rowMapY = mapY.ptr<float>(row);
        const cv::Point2d rowWorld = originMm + stepY * static_cast<double>(row);
        for (int col = 0; col < width; ++col)
        {
            const cv::Point2d world = rowWorld + stepX * static_cast<double>(col);
            const cv::Vec3d projected = worldToCorrected * cv::Vec3d(world.x, world.y, 1.0);
            if (!std::isfinite(projected[2]) || std::abs(projected[2]) < 1e-15)
            {
                rowMapX[col] = -1.0f;
                rowMapY[col] = -1.0f;
                continue;
            }
            const cv::Point2d corrected(projected[0] / projected[2], projected[1] / projected[2]);
            const cv::Point2d delta = (corrected - plane.center) * invNormalizer;
            const double discriminant = 1.0 - 4.0 * plane.division * delta.dot(delta);
            if (!(discriminant >= 0.0))
            {
                rowMapX[col] = -1.0f;
                rowMapY[col] = -1.0f;
                continue;
            }
            const double factor = normalizer * 2.0 / (1.0 + std::sqrt(discriminant));
            const double srcX = plane.center.x + delta.x * factor;
            const double srcY = plane.center.y + delta.y * factor;
            if (!std::isfinite(srcX) || !std::isfinite(srcY))
            {
                rowMapX[col] = -1.0f;
                rowMapY[col] = -1.0f;
                continue;
            }
            rowMapX[col] = static_cast<float>(srcX);
            rowMapY[col] = static_cast<float>(srcY);
            if (srcX >= 0.0 && srcX <= maxX && srcY >= 0.0 && srcY <= maxY)
                anyInside = true;
        }
    }

    if (!anyInside)
        fail(MetrologyError::CalibrationFailed, "标定映射未覆盖任何有效输入像素");

    cv::Mat rectified;
    cv::remap(source, rectified, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    geometry::PlanarCalibration adapted;
    adapted.imageSize = rectified.size();
    adapted.center = {width * 0.5, height * 0.5};
    adapted.division = 0.0;
    adapted.imageToMillimeters = cv::Matx33d(
        stepX.x, stepY.x, originMm.x,
        stepX.y, stepY.y, originMm.y,
        0.0,     0.0,     1.0);
    adapted.validate();

    const double axisLengthMm = pixelSizeMm * static_cast<double>(std::min(width, height)) * 0.1;
    const cv::Point2d frameOrigin = adapted.project({0.0, 0.0});
    const cv::Point2d frameXAxis = adapted.project({axisLengthMm, 0.0}) - frameOrigin;
    const cv::Point2d frameYAxis = adapted.project({0.0, axisLengthMm}) - frameOrigin;

    return {std::move(rectified), adapted, pixelSizeMm,
            visionrt::coordinateFrameRecord(frameOrigin, frameXAxis, frameYAxis)};
}
} // namespace

visionrt::ToolStatus RectifyImageTool::initialize(const visionrt::ToolConfiguration &configuration)
{
    if (!configuration.parameters.empty())
        return {visionrt::kContractViolationStatus, "图像校正不接受初始化参数"};
    return {};
}

visionrt::ToolResult RectifyImageTool::execute(const visionrt::Values &inputs)
{
    if (const auto upstream = upstreamStatus(inputs); upstream != 0)
        return skippedByUpstream(upstream, missingOutputs());

    Readings readings(inputs, p::TOOL_ID);
    const auto imagesRef = readings.image(p::in::IMAGES);
    const auto calibrationValue = readings.find(p::in::CALIBRATION);
    const auto unit = readings.text(p::in::UNIT);
    if (readings.failed())
        return visionrt::toolResult(missingOutputs(),
                                    readings.status(CommonToolError::InputMissing, CommonToolError::InputTypeMismatch));
    if (!calibrationValue || visionrt::isMissing(*calibrationValue))
        return failure(missingOutputs(), CommonToolError::InputMissing, "标定数据");
    const auto *calibrationArray = std::get_if<visionrt::Array>(calibrationValue);
    if (!calibrationArray)
        return failure(missingOutputs(), CommonToolError::InputTypeMismatch, "标定数据必须是数组");

    try
    {
        const auto &images = imagesRef.get();
        if (images.empty())
            fail(CommonToolError::InputMissing, "图像");
        const double mmPerUnit = geometry::millimetersPerUnit(unit);
        const auto calibrations = geometry::calibrationsFromArray(*calibrationArray, images.size());
        if (calibrations.empty())
            fail(CommonToolError::InputMissing, "标定数据不能为空");

        visionrt::Image rectifiedBatch;
        rectifiedBatch.reserve(images.size());
        std::vector<geometry::PlanarCalibration> adaptedCalibrations;
        adaptedCalibrations.reserve(images.size());
        std::vector<double> pixelSizes;
        pixelSizes.reserve(images.size());
        visionrt::CoordinateFrame2D::Nested frames;
        frames.reserve(images.size());

        for (std::size_t index = 0; index < images.size(); ++index)
        {
            requireSupportedImage(images[index], "图像校正");
            auto outcome = rectifySingle(images[index], calibrations[index]);
            rectifiedBatch.push_back(std::move(outcome.rectified));
            adaptedCalibrations.push_back(outcome.adapted);
            pixelSizes.push_back(outcome.pixelSizeMm / mmPerUnit);
            frames.push_back({std::move(outcome.coordinateFrame)});
        }

        visionrt::Values outputs;
        outputs.emplace(p::out::RECTIFIED_IMAGES, visionrt::DataRef<visionrt::Image>(std::move(rectifiedBatch)));
        outputs.emplace(p::out::ADAPTED_CALIBRATION, geometry::calibrationArray(adaptedCalibrations));
        outputs.emplace(p::out::PIXEL_SIZE, visionrt::Array::fromNested(std::move(pixelSizes)));
        outputs.emplace(p::out::COORDINATE_FRAME, visionrt::CoordinateFrame2D::fromNested(std::move(frames)));
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
        return failure(missingOutputs(), MetrologyError::CalibrationFailed, "图像校正失败", error.what());
    }
}

visionrt::ToolStatus RectifyImageTool::release()
{
    return {};
}

} // namespace visiontools::metrology
