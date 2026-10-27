#pragma once

#include "BrightnessTool.h"

#include <visionrt/contracts/Parameters.h>

#include <optional>

namespace visiontools::preprocess {

// Typed view of the same port set a class implementation would read. The shared ROI/geometry
// kernels still own the interpretation of their raw values; nothing is cloned or normalised here.
struct BrightnessInputs final : visionrt::InputParametersBase
{
    // DataRef has no empty state, so the field starts as an empty batch and is replaced by parse().
    visionrt::DataRef<visionrt::Image>              images{visionrt::Image{}};
    double                                          gain    = 1.0;
    double                                          offset  = 0.0;
    bool                                            emitMask = false;
    std::optional<visionrt::DataRef<visionrt::Image>> mask;
    visionrt::PortValue                              rois;
    visionrt::PortValue                              alignment;

protected:
    void readFields(visionrt::InputReader &reader) override;
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
