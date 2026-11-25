#include "PlanarCalibration.h"

#include "common/errors/GeometryErrors.h"
#include "common/errors/ToolError.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace visiontools::geometry {
namespace {
cv::Point2d projectPoint(const cv::Matx33d &matrix, cv::Point2d point)
{
    const auto value = matrix * cv::Vec3d(point.x, point.y, 1);
    const cv::Point2d result(value[0] / value[2], value[1] / value[2]);
    if (!std::isfinite(result.x) || !std::isfinite(result.y)) fail(GeometryError::AlignmentInvalid, "标定映射没有有限坐标");
    return result;
}
}

void PlanarCalibration::validate() const
{
    if (imageSize.width <= 0 || imageSize.height <= 0 || !std::isfinite(center.x) || !std::isfinite(center.y)
        || !std::isfinite(division) || !cv::checkRange(cv::Mat(imageToMillimeters)) || cv::determinant(imageToMillimeters) == 0)
        fail(GeometryError::AlignmentInvalid, "标定参数必须是有限可逆平面映射");
}
cv::Point2d PlanarCalibration::correct(cv::Point2d point) const
{
    const auto delta = point - center;
    const double normalizer = std::max(imageSize.width, imageSize.height);
    const double denominator = 1 + division * delta.dot(delta) / (normalizer * normalizer);
    const auto result = center + delta / denominator;
    if (!std::isfinite(result.x) || !std::isfinite(result.y)) fail(GeometryError::AlignmentInvalid, "畸变校正超出定义域");
    return result;
}
cv::Point2d PlanarCalibration::map(cv::Point2d point) const
{
    return projectPoint(imageToMillimeters, correct(point));
}
cv::Point2d PlanarCalibration::project(cv::Point2d millimeters) const
{
    const auto corrected = projectPoint(imageToMillimeters.inv(), millimeters);
    const double normalizer = std::max(imageSize.width, imageSize.height);
    const auto delta = (corrected - center) / normalizer;
    const double discriminant = 1 - 4 * division * delta.dot(delta);
    if (!(discriminant >= 0)) fail(GeometryError::AlignmentInvalid, "标定坐标超出镜头映射范围");
    return center + delta * (normalizer * 2 / (1 + std::sqrt(discriminant)));
}
PlanarCalibration PlanarCalibration::scaled(double factor) const
{
    auto result = *this;
    for (int column = 0; column < ColumnCount; ++column)
    {
        result.imageToMillimeters(0, column) *= factor;
        result.imageToMillimeters(1, column) *= factor;
    }
    return result;
}
visionrt::Array calibrationArray(std::span<const PlanarCalibration> values)
{
    std::vector<double> data;
    data.reserve(values.size() * PlanarCalibration::RowCount * PlanarCalibration::ColumnCount);
    for (const auto &value : values)
    {
        value.validate();
        data.insert(data.end(), {PlanarCalibration::Version, static_cast<double>(value.imageSize.width),
                                 static_cast<double>(value.imageSize.height), value.center.x, value.center.y, value.division});
        data.insert(data.end(), std::begin(value.imageToMillimeters.val), std::end(value.imageToMillimeters.val));
    }
    return visionrt::Array::dense(visionrt::DType::Float64,
        {static_cast<std::int64_t>(values.size()), PlanarCalibration::RowCount, PlanarCalibration::ColumnCount}, std::move(data));
}
visionrt::Array calibrationArray(const PlanarCalibration &value) { return calibrationArray(std::span(&value, 1)); }
std::vector<PlanarCalibration> calibrationsFromArray(const visionrt::Array &value, std::size_t imageCount)
{
    if (value.type() != visionrt::TypeSpec{visionrt::Type::Array, visionrt::DType::Float64, 3})
        fail(CommonToolError::InputTypeMismatch, "标定数据必须是三维浮点数数组");
    if (value.elementCount() == 0) return {};
    const auto nested = visionrt::nestedOf<visionrt::NestedVector<double, 3>>(value);
    if (nested.size() != 1 && nested.size() != imageCount) fail(CommonToolError::InputOutOfRange, "标定数据数量必须为一或与图像数量一致");
    std::vector<PlanarCalibration> result;
    for (const auto &rows : nested)
    {
        if (rows.size() != PlanarCalibration::RowCount
            || std::ranges::any_of(rows, [](const auto &r) { return r.size() != PlanarCalibration::ColumnCount; }))
            fail(CommonToolError::InputOutOfRange, "标定数据每项必须为五行三列");
        const auto &header = rows[PlanarCalibration::Header];
        if (header[0] != PlanarCalibration::Version || header[1] <= 0 || header[2] <= 0
            || header[1] > std::numeric_limits<int>::max() || header[2] > std::numeric_limits<int>::max()
            || std::trunc(header[1]) != header[1] || std::trunc(header[2]) != header[2])
            fail(CommonToolError::InputOutOfRange, "标定数据版本或图像尺寸无效");
        PlanarCalibration calibration;
        calibration.imageSize = {static_cast<int>(header[1]), static_cast<int>(header[2])};
        const auto &lens = rows[PlanarCalibration::Division];
        calibration.center = {lens[0], lens[1]}; calibration.division = lens[2];
        for (int row = 0; row < PlanarCalibration::ColumnCount; ++row)
            for (int column = 0; column < PlanarCalibration::ColumnCount; ++column)
                calibration.imageToMillimeters(row, column) = rows[PlanarCalibration::HomographyX + row][column];
        calibration.validate();
        result.push_back(calibration);
    }
    if (result.size() == 1 && imageCount > 1) result.resize(imageCount, result.front());
    return result;
}
double millimetersPerUnit(std::string_view unit)
{
    for (const auto &[name, factor] : {std::pair<std::string_view, double>{"mm", 1}, {"um", 0.001}, {"cm", 10}, {"m", 1000}})
        if (unit == name) return factor;
    fail(CommonToolError::InputOutOfRange, "单位必须为 mm、um、cm 或 m");
}
}
