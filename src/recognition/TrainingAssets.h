#pragma once

#include <opencv2/core.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace visiontools::recognition {

// A rotated search band the locator samples across. The centre is in reference image coordinates,
// the angle follows the VisionRT rectangle convention (x right, y down, positive clockwise).
struct SearchRectangle
{
    cv::Point2d center;
    double      width  = 0;
    double      height = 0;
    double      angle  = 0;
};

// One trained reference edge: the nominal segment and the band searched across it.
struct ReferenceSegment
{
    std::string     name;
    cv::Point2d     from;
    cv::Point2d     to;
    SearchRectangle search;
};

struct StraightEdgeTraining
{
    std::filesystem::path         referenceImage;
    std::vector<ReferenceSegment> segments;
};

// One labelled colour region on one training image.
struct ColorRegion
{
    cv::Point2d center;
    double      width  = 0;
    double      height = 0;
    double      angle  = 0;
};

struct ColorSample
{
    std::string                id;
    std::string                label;
    std::filesystem::path      image;
    std::vector<ColorRegion>   regions;
};

struct ColorTraining
{
    std::vector<ColorSample> samples;
};

// Both assets are the single source of a tool's training: the file names the images and the
// geometry, and the tool rebuilds its model from it at initialization. A malformed asset raises a
// ToolFailure from this tool's own error domain, so the host reports what to fix.
StraightEdgeTraining loadStraightEdgeTraining(const std::filesystem::path &asset);
ColorTraining        loadColorTraining(const std::filesystem::path &asset);

} // namespace visiontools::recognition
