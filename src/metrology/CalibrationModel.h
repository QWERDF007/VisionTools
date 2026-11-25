#pragma once

#include "common/PlanarCalibration.h"
#include <visionrt/contracts/GeometryValues.h>
#include <cstdint>
#include <span>
#include <vector>

namespace visiontools::metrology {

struct CalibrationModel
{
    geometry::PlanarCalibration plane;
    double reprojectionError = 0;
    double axisLengthMillimeters = 0;

    std::vector<std::uint8_t> encode() const;
    static CalibrationModel decode(std::span<const std::uint8_t> bytes);
    std::vector<double> coordinateFrame(const geometry::PlanarCalibration &current) const;
};

}
