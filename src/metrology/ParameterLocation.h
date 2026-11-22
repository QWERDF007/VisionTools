#pragma once

#include "common/GeometryTransform.h"
#include "common/RoiSupport.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace visiontools::metrology {

struct LocationRequest
{
    cv::Point2d              searchCenter;
    std::vector<roi::Record> seeds;
    geometry::Alignment      alignment;
    int                      searchRadius = 256;
};

struct LocationResult
{
    bool                     valid = false;
    std::vector<roi::Record> preview;  // current image coordinates
    std::vector<roi::Record> proposal; // reference coordinates
    std::string              quality;
    std::string              polarity        = "any";
    double                   searchHalfWidth = 8.0;
    double                   sigma           = 1.0;
    double                   widthMin        = 1.0;
    double                   widthMax        = 1.0e6;
};

// Bounded multi-scale candidate search around the search center; the formal measurement
// kernel then refines the best candidate at full resolution.
LocationResult locateCircle(const cv::Mat &gray, const LocationRequest &request);

// Structure tensor orientation, local edge pairing and the shared row kernel.
LocationResult locateRowSpacing(const cv::Mat &gray, const LocationRequest &request);

} // namespace visiontools::metrology
