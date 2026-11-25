#include "CalibrateCameraTool.h"

#include "CalibrationFit.h"
#include "common/EdgeProfile.h"
#include "common/ImageSupport.h"
#include "common/InputValues.h"
#include "common/ProductPaths.h"
#include "common/RoiSupport.h"
#include "common/ToolOutputs.h"
#include "common/UpstreamStatus.h"
#include "errors/MetrologyErrors.h"
#include <cmath>
#include <fstream>

namespace visiontools::metrology {
namespace {
namespace p = ports::calibrate_camera;
visionrt::Values missingOutputs()
{
    return missingPorts({p::out::MODEL_PATH, p::out::CALIBRATION, p::out::MARKS, p::out::REPROJECTION_ERROR, p::out::COORDINATE_FRAME});
}
}

visionrt::ToolStatus CalibrateCameraTool::initialize(const visionrt::ToolConfiguration &configuration)
{
    try
    {
        const auto found = configuration.models.find(p::MODEL_ARTIFACT);
        if (found == configuration.models.end() || found->second.empty()) return status(MetrologyError::CalibrationModelMissing);
        std::ifstream file(found->second, std::ios::binary);
        if (!file) return status(MetrologyError::CalibrationModelMissing, "无法读取标定模型");
        const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
        model_ = CalibrationModel::decode(bytes);
        const auto path = found->second.generic_u8string();
        modelPath_.assign(reinterpret_cast<const char *>(path.data()), path.size());
        return {};
    }
    catch (const ToolFailure &error) { return status(error); }
    catch (const std::exception &error) { return status(MetrologyError::CalibrationFailed, "加载标定模型失败", error.what()); }
}

visionrt::ToolResult CalibrateCameraTool::train(const visionrt::Values &parameters)
{
    Readings readings(parameters, p::TOOL_ID, visionrt::PortStage::Train);
    const auto dataset = readings.trainingSet(p::train::DATASET);
    const auto descriptor = readings.text(p::train::PLATE_DESCRIPTION);
    const auto path = readings.text(p::train::MODEL_PATH);
    const auto maximumError = readings.real(p::train::MAX_REPROJECTION_ERROR);
    if (readings.failed()) return visionrt::toolResult(missingOutputs(), readings.status(CommonToolError::InputMissing, CommonToolError::InputTypeMismatch));
    try
    {
        if (!std::isfinite(maximumError) || maximumError <= 0) fail(CommonToolError::InputOutOfRange, "重投影误差阈值必须大于零");
        const auto &set = dataset.get();
        visionrt::validateTrainingSet(set);
        if (set.records.size() != 1 || !set.records.begin()->second.imageId)
            fail(MetrologyError::CalibrationPlateInvalid, "训练需要一条包含标定板原图的记录");
        const auto &image = set.images.at(*set.records.begin()->second.imageId).pixels;
        requireSupportedImage(image, "标定训练");
        const auto plate = CalibrationPlate::read(common::resolveResourcePath(descriptor));
        auto fit = fitCalibration(edge::grayscale(image), plate);
        if (fit.reprojectionError > maximumError)
            fail(MetrologyError::CalibrationQualityRejected, std::to_string(fit.reprojectionError) + " px");
        CalibrationModel model{fit.calibration, fit.reprojectionError, plate.pitch * 3};
        std::vector<roi::Record> marks;
        for (const auto &mark : fit.marks) marks.push_back(roi::circle(mark.image.x, mark.image.y, mark.radius));
        visionrt::Values outputs{{p::out::MODEL_PATH, path}, {p::out::CALIBRATION, geometry::calibrationArray(model.plane)},
            {p::out::REPROJECTION_ERROR, model.reprojectionError}, {p::out::MARKS, roi::value<visionrt::Circle>({std::move(marks)})},
            {p::out::COORDINATE_FRAME, visionrt::CoordinateFrame2D::fromNested({{model.coordinateFrame(model.plane)}})}};
        auto result = visionrt::toolResult(std::move(outputs), visionrt::executionSuccess("标定成功"));
        result.artifacts.emplace(p::MODEL_ARTIFACT, visionrt::ModelArtifact{model.encode()});
        return result;
    }
    catch (const ToolFailure &error) { return failureResult(missingOutputs(), error); }
    catch (const std::exception &error) { return failure(missingOutputs(), MetrologyError::CalibrationFailed, {}, error.what()); }
}

visionrt::ToolResult CalibrateCameraTool::execute(const visionrt::Values &inputs)
{
    if (const auto upstream = upstreamStatus(inputs); upstream != 0) return skippedByUpstream(upstream, missingOutputs());
    if (!model_) return failure(missingOutputs(), MetrologyError::CalibrationModelMissing);
    Readings readings(inputs, p::TOOL_ID);
    const auto images = readings.image(p::in::IMAGES);
    const auto regions = readings.line(p::in::REFERENCE_SEGMENTS);
    const auto length = readings.real(p::in::REFERENCE_LENGTH);
    const auto unit = readings.text(p::in::UNIT);
    if (readings.failed()) return visionrt::toolResult(missingOutputs(), readings.status(CommonToolError::InputMissing, CommonToolError::InputTypeMismatch));
    try
    {
        const auto millimeters = geometry::millimetersPerUnit(unit);
        const auto references = roi::groups(regions, images.get().size(), "已知长度参考线");
        std::vector<geometry::PlanarCalibration> planes;
        visionrt::CoordinateFrame2D::Nested frames;
        for (std::size_t index = 0; index < images.get().size(); ++index)
        {
            if (images.get()[index].size() != model_->plane.imageSize)
                fail(CommonToolError::InputOutOfRange, "测量图像尺寸必须与标定图像一致");
            auto plane = model_->plane;
            if (!references.empty() && !references[index].empty())
            {
                if (references[index].size() != 1 || !std::isfinite(length) || length <= 0)
                    fail(MetrologyError::CalibrationReferenceInvalid, "每张图最多一条非零长度参考线");
                const auto segment = visionrt::RegionView(references[index].front()).segment();
                const auto distance = cv::norm(plane.map(segment.end()) - plane.map(segment.start()));
                if (!(distance > 0)) fail(MetrologyError::CalibrationReferenceInvalid, "参考线端点重合");
                plane = plane.scaled(length * millimeters / distance);
            }
            planes.push_back(plane);
            frames.push_back({model_->coordinateFrame(plane)});
        }
        return visionrt::toolResult({{p::out::MODEL_PATH, modelPath_}, {p::out::CALIBRATION, geometry::calibrationArray(planes)},
            {p::out::REPROJECTION_ERROR, model_->reprojectionError}, {p::out::COORDINATE_FRAME, visionrt::CoordinateFrame2D::fromNested(std::move(frames))}}, visionrt::executionSuccess());
    }
    catch (const ToolFailure &error) { return failureResult(missingOutputs(), error); }
    catch (const std::exception &error) { return failure(missingOutputs(), MetrologyError::CalibrationReferenceInvalid, {}, error.what()); }
}

visionrt::ToolStatus CalibrateCameraTool::release()
{
    model_.reset();
    modelPath_.clear();
    return {};
}
}
