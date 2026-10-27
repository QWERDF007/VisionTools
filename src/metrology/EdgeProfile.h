#pragma once

#include <opencv2/core.hpp>

#include <optional>
#include <vector>

namespace visiontools::metrology {

enum class EdgePolarity
{
    Any,
    DarkToLight,
    LightToDark
};

struct ProfileConfig
{
    // Search half width and tangential averaging width in pixels.
    double       searchHalfWidth = 8.0;
    double       averageWidth    = 3.0;
    double       sigma           = 1.0;
    double       minGradient     = 8.0;
    EdgePolarity polarity        = EdgePolarity::Any;
};

struct EdgeCandidate
{
    double       offset   = 0; // signed distance from the nominal point along the normal
    double       gradient = 0; // signed intensity derivative per pixel
    cv::Point2d  point;
};

struct IntensityProfile
{
    double              step     = 0.5; // sampling step in pixels
    double              halfLength = 0; // centred at zero
    std::vector<double> values;         // NaN marks a window that left the image
};

// Builds a tangentially averaged intensity profile along `direction`, sampling
// `averageHalfWidth` on both sides of `averageDirection`. Uses bilinear sampling
// and keeps only fully valid windows; out-of-image windows are marked NaN.
IntensityProfile buildProfile(const cv::Mat &gray, cv::Point2d center, cv::Point2d direction, double halfLength,
                              cv::Point2d averageDirection, double averageHalfWidth, double step = 0.5,
                              std::size_t budget = 8'000'000);

// Signed first derivative of the Gaussian-smoothed profile, in intensity/pixel.
std::vector<double> gradientProfile(const IntensityProfile &profile, double sigma);

// Local extrema of the gradient that pass the polarity and threshold rules.
std::vector<EdgeCandidate> detectEdges(const IntensityProfile &profile, const ProfileConfig &config);

// Prefers the candidate closest to the nominal edge, then the stronger gradient,
// then the earlier position so selection never depends on container order.
std::optional<EdgeCandidate> nearestEdge(std::vector<EdgeCandidate> candidates);

cv::Mat grayscale(const cv::Mat &image);

} // namespace visiontools::metrology
