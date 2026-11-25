#pragma once

#include <opencv2/core.hpp>
#include <filesystem>
#include <vector>

namespace visiontools::metrology {

struct CalibrationPlate
{
    int rows = 0, columns = 0;
    bool lightMarks = true;
    std::vector<cv::Point2i> finders;
    std::vector<cv::Point2d> millimeters;
    double pitch = 0;
    cv::Point2d origin;
    static CalibrationPlate read(const std::filesystem::path &file);
};

struct CalibrationMark
{
    cv::Point2d image;
    cv::Point2d millimeters;
    double radius = 0;
};

std::vector<CalibrationMark> observeCalibrationPlate(const cv::Mat &gray, const CalibrationPlate &plate);

}
