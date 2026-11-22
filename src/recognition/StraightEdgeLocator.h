#pragma once

#include "common/EdgeProfile.h"
#include "common/GeometryTransform.h"

#include <opencv2/core.hpp>

#include <optional>
#include <string>

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

// One reference edge: the nominal segment and the band searched across it. The published model
// stores the measured endpoints here, so a node never re-derives them from the band.
struct ReferenceSegment
{
    std::string     name;
    cv::Point2d     from;
    cv::Point2d     to;
    SearchRectangle search;
};

// How the locator samples one trained segment. Contrast is the minimum intensity derivative along
// the search normal, in gray levels per pixel, so its unit does not depend on the image size.
struct EdgeSearchOptions
{
    double              contrast        = 30.0;
    int                 width           = 3;
    edge::EdgePolarity  polarity        = edge::EdgePolarity::Any;
    double              minCompleteness = 0.4;
};

struct LocatedSegment
{
    cv::Point2d start;
    cv::Point2d end;
    double      completeness = 0;
};

// Samples the reference segment across its trained search band and fits a line through the edge
// points that pass the contrast and polarity rules. Empty when the edge is missing or too
// incomplete to trust.
std::optional<LocatedSegment> locateSegment(const cv::Mat &gray, const ReferenceSegment &reference,
                                            const EdgeSearchOptions &options);

// Reference-to-current rigid transform from two non-parallel corresponding segments: the angle
// comes from their directions, the translation from their intersection. Empty when the reference
// segments are parallel.
std::optional<geometry::Alignment> rigidTransform(const ReferenceSegment &firstReference,
                                                  const LocatedSegment  &firstFound,
                                                  const ReferenceSegment &secondReference,
                                                  const LocatedSegment  &secondFound);

} // namespace visiontools::recognition
