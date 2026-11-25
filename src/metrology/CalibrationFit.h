#pragma once

#include "CalibrationPlate.h"
#include "common/PlanarCalibration.h"

namespace visiontools::metrology {

struct CalibrationFit
{
    geometry::PlanarCalibration calibration;
    double reprojectionError = 0;
    std::vector<CalibrationMark> marks;
};

CalibrationFit fitCalibration(const cv::Mat &gray, const CalibrationPlate &plate);

}
