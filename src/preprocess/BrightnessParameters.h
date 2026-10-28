#pragma once

#include "BrightnessTool.h"

#include <visionrt/contracts/Parameters.h>

#include <optional>

namespace visiontools::preprocess {

// Typed view of the same port set a class implementation would read. The shared ROI/geometry
// kernels still own the interpretation of their raw values; nothing is cloned or normalised here.
struct BrightnessInputs final : visionrt::InputParametersBase
{
    // `images` is declared optional: a branch that was not selected delivers no image, and the
    // tool - not the parser - decides whether that is a failure. An optional DataRef is exactly
    // that distinction; an empty batch would hide it.
    std::optional<visionrt::DataRef<visionrt::Image>> images;
    double                                          gain    = 1.0;
    double                                          offset  = 0.0;
    bool                                            emitMask = false;
    std::optional<visionrt::DataRef<visionrt::Image>> mask;
    visionrt::PortValue                              rois;
    visionrt::PortValue                              alignment;

protected:
    visionrt::ToolStatus readFields(visionrt::InputReader &reader) override;
};

struct BrightnessOutputs final : visionrt::OutputParametersBase
{
    BrightnessOutputs()
    {
        status = visionrt::executionSuccess();
    }

    std::optional<visionrt::DataRef<visionrt::Image>> images;
    std::optional<visionrt::DataRef<visionrt::Image>> appliedMask;

protected:
    void writeFields(visionrt::OutputWriter &writer) override;
};

} // namespace visiontools::preprocess
