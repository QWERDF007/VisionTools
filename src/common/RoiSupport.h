#pragma once

#include "common/GeometryTransform.h"

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

namespace visiontools::roi {

struct Record
{
    visionrt::RoiKind   kind = visionrt::RoiKind::Point;
    std::vector<double> values; // payload without the kind slot
};

// Decodes ROI[image][record][value] and enforces the outer group rule:
// empty means "no groups configured", otherwise exactly one group per image.
std::vector<std::vector<Record>> groups(const visionrt::PortValue &value, std::size_t imageCount,
                                        const char *portName);

// True when at least one image carries at least one record.
bool anyRecord(const std::vector<std::vector<Record>> &records) noexcept;

std::vector<double> encoded(const Record &record);

// Pixel-center containment in reference coordinates; `toReference` maps current
// image coordinates back to the reference frame. Area kinds only.
bool contains(const Record &record, double currentX, double currentY, const geometry::Alignment &toReference);

// CV_8UC1 mask (0 or 255) of the union of all records of one image, evaluated at
// pixel centers in reference coordinates.
cv::Mat regionMask(const std::vector<Record> &records, cv::Size size, const geometry::Alignment &toReference);

visionrt::PortValue roiValue(std::vector<std::vector<Record>> records);
visionrt::ROI       toRoi(std::vector<std::vector<Record>> records);

Record point(double x, double y);
Record circle(double centerX, double centerY, double radius);
Record rectangle(double centerX, double centerY, double width, double height, double angleDegrees);
Record segmentValue(double x1, double y1, double x2, double y2);
Record pointSet(std::vector<cv::Point2d> values);

} // namespace visiontools::roi
