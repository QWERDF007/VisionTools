#pragma once

#include "common/GeometryTransform.h"
#include "common/RoiSupport.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace visiontools::recognition {

// What the host asks about one point of the picture. The probe reuses the same edge vocabulary as
// the alignment tool, so a band picked here behaves exactly like a trained one.
struct EdgeProbeRequest
{
    cv::Point2d              cursor;
    bool                     learn = false;
    std::vector<roi::Record> seeds;
    geometry::Alignment      alignment;
    int                      searchExtent = 128;
};

struct EdgeProbeResult
{
    bool                     valid = false;
    std::vector<roi::Record> preview;  // current image coordinates
    std::vector<roi::Record> proposal; // reference coordinates
    std::string              quality;
    std::string              polarity = "any";
};

// Estimates the local edge orientation from the structure tensor, then fits the line with the
// shared probe kernel and answers with the band a reference edge would be built from: the long side
// follows the edge, the short side is the search range across it.
EdgeProbeResult probeStraightEdge(const cv::Mat &gray, const EdgeProbeRequest &request);

} // namespace visiontools::recognition
