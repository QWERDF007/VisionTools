#include "CalibrationModel.h"

#include "errors/MetrologyErrors.h"
#include <visionrt/contracts/ValueJson.h>
#include <array>
#include <cmath>

namespace visiontools::metrology {
namespace {
constexpr std::int64_t kModelVersion = 2;
constexpr visionrt::TypeSpec kCalibrationType{visionrt::Type::Array, visionrt::DType::Float64, 3};
constexpr visionrt::TypeSpec kRealType{visionrt::Type::Float};
constexpr char kVersion[] = "version", kPlane[] = "plane", kError[] = "reprojection_error", kAxis[] = "axis_length_mm";
}

std::vector<std::uint8_t> CalibrationModel::encode() const
{
    const auto json = std::string("{\"") + kVersion + "\":" + std::to_string(kModelVersion)
        + ",\"" + kPlane + "\":" + visionrt::writeValueJson(geometry::calibrationArray(plane), kCalibrationType)
        + ",\"" + kError + "\":" + visionrt::writeValueJson(reprojectionError, kRealType)
        + ",\"" + kAxis + "\":" + visionrt::writeValueJson(axisLengthMillimeters, kRealType) + "}";
    return {json.begin(), json.end()};
}

CalibrationModel CalibrationModel::decode(std::span<const std::uint8_t> bytes)
{
    const visionrt::ValueJsonDocument json(std::string_view(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
    if (std::get<std::int64_t>(json.read(std::array<std::string, 1>{kVersion}, {visionrt::Type::Int})) != kModelVersion)
        fail(MetrologyError::CalibrationFailed, "标定模型版本不受支持");
    CalibrationModel result;
    const auto planes = geometry::calibrationsFromArray(std::get<visionrt::Array>(json.read(std::array<std::string, 1>{kPlane}, kCalibrationType)), 1);
    if (planes.size() != 1) fail(MetrologyError::CalibrationFailed, "标定模型需要一个基准平面");
    result.plane = planes.front();
    result.reprojectionError = std::get<double>(json.read(std::array<std::string, 1>{kError}, kRealType));
    result.axisLengthMillimeters = std::get<double>(json.read(std::array<std::string, 1>{kAxis}, kRealType));
    if (!std::isfinite(result.reprojectionError) || result.reprojectionError < 0
        || !std::isfinite(result.axisLengthMillimeters) || result.axisLengthMillimeters <= 0)
        fail(MetrologyError::CalibrationFailed, "标定模型误差或轴长无效");
    return result;
}

std::vector<double> CalibrationModel::coordinateFrame(const geometry::PlanarCalibration &current) const
{
    const auto origin = current.project({0, 0});
    return visionrt::coordinateFrameRecord(origin, current.project({axisLengthMillimeters, 0}) - origin,
                                           current.project({0, axisLengthMillimeters}) - origin);
}
}
