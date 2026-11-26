#pragma once

#include "common/EdgeProfile.h"
#include "common/GeometryTransform.h"

#include <opencv2/core.hpp>

#include <optional>
#include <cstdint>
#include <string>

namespace visiontools::recognition {
inline constexpr double kMinEdgeAngleSine = 1e-3;

// A rotated search band the locator samples across. The centre is in image coordinates,
// the angle follows the VisionRT rectangle convention (x right, y down, positive clockwise).
struct SearchRectangle
{
    cv::Point2d center;
    double      width  = 0;
    double      height = 0;
    double      angle  = 0;
};

// One measured reference edge, fixed until the model is trained again.
struct ReferenceSegment
{
    std::string name;
    cv::Point2d from;
    cv::Point2d to;
};

// How the locator samples one trained segment. Contrast is the minimum intensity derivative along
// the search normal, in gray levels per pixel, so its unit does not depend on the image size.
struct EdgeSearchOptions
{
    double             contrast        = 30.0;
    int                width           = 3;
    edge::EdgePolarity polarity        = edge::EdgePolarity::Any;
    double             minCompleteness = 0.4;
    static EdgeSearchOptions fromParameters(double contrast, std::int64_t width, const std::string &polarity, double completeness);
};

struct LocatedSegment
{
    cv::Point2d start;
    cv::Point2d end;
    double      completeness = 0;
};

// Converts one encoded search region; the caller retains the region order.
SearchRectangle searchRectangle(const visionrt::EncodedRegion &record, std::size_t index);
std::vector<double> searchDirectionRecord(const SearchRectangle &band);

// Fits the edge points in the current search band on an 8-bit 1/3/4-channel image.
// Empty when no sufficiently complete edge exists.
std::optional<LocatedSegment> locateSegment(const cv::Mat &image, const SearchRectangle &band,
                                            const EdgeSearchOptions &options);

// Reference-to-current rigid transform from two non-parallel corresponding segments: the angle
// comes from their directions, the translation from their intersection. Empty when the reference
// segments are parallel.
std::optional<geometry::Alignment> rigidTransform(const ReferenceSegment &firstReference,
                                                  const LocatedSegment   &firstFound,
                                                  const ReferenceSegment &secondReference,
                                                  const LocatedSegment   &secondFound);

} // namespace visiontools::recognition
