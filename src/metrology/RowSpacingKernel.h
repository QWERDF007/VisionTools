#pragma once

#include "common/RoiSupport.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace visiontools::metrology {

struct RowOptions
{
    double sigma        = 1.0;
    double minGradient  = 8.0;
    bool   brightObject = true;
    double widthMin     = 1.0;
    double widthMax     = 1.0e6;
};

struct RowPair
{
    double first  = 0; // leading edge offset from the ROI centre along the scan direction
    double second = 0;
};

struct RowMeasurement
{
    bool                     valid = false;
    std::string              reason;
    std::vector<RowPair>     pairs;
    std::vector<double>      widths;   // pair count
    std::vector<double>      gaps;     // pair count - 1
    std::vector<double>      pitches;  // pair count - 1
    std::vector<bool>        accepted; // pair count
    int                      unpaired = 0;
};

// Measures bright/dark band widths inside one rotated rectangle ROI. The ROI must
// be kind=2 and fully inside the image; an out-of-image ROI is reported as an
// invalid measurement instead of being clipped to a different region.
RowMeasurement measureRowRect(const cv::Mat &gray, const roi::Record &rectangle, const RowOptions &options);

} // namespace visiontools::metrology
