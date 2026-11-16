#pragma once

#include "TrainingAssets.h"

#include "common/EdgeProfile.h"
#include "common/GeometryTransform.h"

#include <opencv2/core.hpp>

#include <optional>

namespace visiontools::recognition {

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
