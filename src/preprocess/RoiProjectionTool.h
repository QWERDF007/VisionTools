#pragma once

#include <visionrt/contracts/Tool.h>

namespace visiontools::preprocess {

namespace ports::roi_projection {
inline constexpr char TOOL_ID[] = "visionrt.roi.projection";

namespace in {
inline constexpr char AXIS[]       = "axis";
inline constexpr char IMAGES[]     = "images";
inline constexpr char MERGE_GAP[]  = "merge_gap";
inline constexpr char MIN_EXTENT[] = "min_extent";
inline constexpr char MIN_PIXELS[] = "min_pixels";
inline constexpr char PADDING[]    = "padding";
inline constexpr char POLARITY[]   = "polarity";
inline constexpr char THRESHOLD[]  = "threshold";
}

namespace out {
inline constexpr char IMAGES[] = "images";
inline constexpr char ROIS[]   = "rois";
}
}

class RoiProjectionTool final : public visionrt::IToolInstance
{
public:
    explicit RoiProjectionTool(const visionrt::ToolServices &services);
    ~RoiProjectionTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;

private:
    visionrt::ToolServices services_;
};

} // namespace visiontools::preprocess
