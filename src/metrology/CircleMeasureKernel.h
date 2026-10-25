#pragma once

#include "EdgeProfile.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace visiontools::metrology {

struct CircleOptions
{
    int          calipers        = 64;
    double       searchHalfWidth = 8.0;
    double       averageWidth    = 3.0;
    double       sigma           = 1.0;
    double       minGradient     = 8.0;
    EdgePolarity polarity        = EdgePolarity::Any;
    int          minPoints       = 12;
    double       minCompleteness = 0.6;
    double       minCoverageDeg  = 180.0;
    double       radiusMin       = 0.5;
    double       radiusMax       = 1.0e6;
    double       maxRms          = 1.0;
};

struct CircleMeasurement
{
    bool                     valid = false;
    std::string              reason;
    double                   centerX = 0, centerY = 0, radius = 0, rms = 0, completeness = 0;
    int                      positiveEdges = 0;
    int                      negativeEdges = 0;
    std::vector<double>      edgeOffsets;
    std::vector<cv::Point2d> edges;
    std::vector<double>      angles;
};

// Radial caliper measurement shared by the formal circle tool and the assist
// helper. All coordinates are current image pixels.
CircleMeasurement measureCircleKernel(const cv::Mat &gray, cv::Point2d center, double radius,
                                      const CircleOptions &options);

} // namespace visiontools::metrology
