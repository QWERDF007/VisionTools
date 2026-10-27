#pragma once

#include "common/GeometryTransform.h"
#include "common/RoiSupport.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace visiontools::metrology {

struct AssistRequest
{
    cv::Point2d                     cursor;
    bool                            learn = false;
    std::vector<roi::Record>        seeds;
    geometry::Alignment             alignment;
    int                             searchExtent = 256;
};

struct AssistResult
{
    bool                    valid = false;
    std::vector<roi::Record> preview;  // current image coordinates
    std::vector<roi::Record> proposal; // reference coordinates
    std::string             quality;
    std::string             polarity = "any";
    double                  searchHalfWidth = 8.0;
    double                  widthMin        = 1.0;
    double                  widthMax        = 1.0e6;
};

// Bounded multi-scale candidate search around the cursor; the formal measurement
// kernel then refines the best candidate at full resolution.
AssistResult assistCircle(const cv::Mat &gray, const AssistRequest &request);

// Structure tensor orientation, local edge pairing and the shared row kernel.
AssistResult assistRowSpacing(const cv::Mat &gray, const AssistRequest &request);

} // namespace visiontools::metrology
