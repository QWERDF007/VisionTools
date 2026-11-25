#pragma once

#include <visionrt/contracts/Affine2D.h>
#include <visionrt/contracts/Value.h>

namespace visiontools::geometry {

using Alignment = visionrt::Affine2D;
using visionrt::alignmentArray;
using visionrt::rotationAround;

// Missing is an absent upstream value. Manual identity comes from the declared six coefficients.
std::vector<Alignment> alignmentsFromValue(const visionrt::PortValue &value, std::size_t imageCount);

}
