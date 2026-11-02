#pragma once

#include "common/GeometryTransform.h"

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

namespace visiontools::roi {

// A record is the encoded region itself: the contract's geometry type is the single source for
// layout, validation and named access, so tools never keep a second decoded copy.
using Record = visionrt::EncodedRegion;

// Decodes ROI[image][record][value], validates each region once through the contract's view, and
// enforces the outer group rule: empty means "no groups configured", otherwise exactly one group per
// image. Validated records are copied because the caller may outlive the input DataRef.
std::vector<std::vector<Record>> groups(const visionrt::PortValue &value, std::size_t imageCount,
                                        const char *portName);

// True when at least one image carries at least one record.
bool anyRecord(const std::vector<std::vector<Record>> &records) noexcept;

// Pixel-center containment in reference coordinates; `toReference` maps current
// image coordinates back to the reference frame. Area kinds only.
bool contains(const Record &record, double currentX, double currentY, const geometry::Alignment &toReference);

// CV_8UC1 mask (0 or 255) of the union of all records of one image, evaluated at
// pixel centers in reference coordinates.
cv::Mat regionMask(const std::vector<Record> &records, cv::Size size, const geometry::Alignment &toReference);

visionrt::PortValue roiValue(std::vector<std::vector<Record>> records);
visionrt::ROI       toRoi(std::vector<std::vector<Record>> records);

// Constructors go through the contract so every record is valid by construction.
Record point(double x, double y);
Record circle(double centerX, double centerY, double radius);
// Rectangle fields keep the contract's meaning: unrotated left/top corner plus size and angle.
Record rectangle(double centerX, double centerY, double width, double height, double angleDegrees);
Record segmentValue(double x1, double y1, double x2, double y2);
Record pointSet(std::vector<cv::Point2d> values);

} // namespace visiontools::roi
