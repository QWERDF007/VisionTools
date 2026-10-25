#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <optional>
#include <vector>

namespace visiontools::metrology {

struct CircleFitResult
{
    double      centerX    = 0;
    double      centerY    = 0;
    double      radius     = 0;
    double      rms        = 0;
    std::size_t inliers    = 0;
    std::size_t points     = 0;
};

struct CircleFitOptions
{
    double maxRms        = 1.0;
    int    maxIterations = 20;
};

// Robust geometric circle fit: algebraic SVD initialisation followed by Huber
// weighted Gauss-Newton. Returns std::nullopt for degenerate or divergent sets.
// An empty result with a reason is reported through `reason` when provided.
std::optional<CircleFitResult> fitCircle(const std::vector<cv::Point2d> &points, const CircleFitOptions &options,
                                         std::string *reason = nullptr);

// Angular coverage in degrees of the sample angles using the largest gap rule.
double angularCoverage(std::vector<double> anglesDegrees);

} // namespace visiontools::metrology
