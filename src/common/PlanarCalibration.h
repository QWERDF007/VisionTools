#pragma once

#include <visionrt/contracts/Array.h>
#include <opencv2/core.hpp>
#include <span>
#include <string_view>
#include <vector>

namespace visiontools::geometry {

// Division correction is normalized by max(image width, image height). The homography maps
// corrected pixel coordinates to millimeters, independently of the output display unit.
struct PlanarCalibration
{
    static constexpr int Version = 1;
    enum Row { Header, Division, HomographyX, HomographyY, HomographyZ, RowCount };
    static constexpr int ColumnCount = 3;
    cv::Size imageSize;
    cv::Point2d center;
    double division = 0;
    cv::Matx33d imageToMillimeters = cv::Matx33d::eye();

    void validate() const;
    cv::Point2d correct(cv::Point2d point) const;
    cv::Point2d map(cv::Point2d point) const;
    cv::Point2d project(cv::Point2d millimeters) const;
    PlanarCalibration scaled(double factor) const;
};

visionrt::Array calibrationArray(std::span<const PlanarCalibration> values);
visionrt::Array calibrationArray(const PlanarCalibration &value);
std::vector<PlanarCalibration> calibrationsFromArray(const visionrt::Array &value, std::size_t imageCount);
double millimetersPerUnit(std::string_view unit);

}
