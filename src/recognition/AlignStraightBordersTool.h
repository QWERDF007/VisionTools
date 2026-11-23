#pragma once

#include "StraightEdgeModel.h"

#include <visionrt/contracts/Tool.h>

namespace visiontools::recognition {

namespace ports::align_straight_borders {
inline constexpr char TOOL_ID[] = "AlignStraightBorders";

namespace init {
inline constexpr visionrt::PortId MODEL = 1;
}

namespace in {
inline constexpr visionrt::PortId IMAGES           = 1;
inline constexpr visionrt::PortId CONTRAST         = 2;
inline constexpr visionrt::PortId EDGE_WIDTH       = 3;
inline constexpr visionrt::PortId POLARITY         = 4;
inline constexpr visionrt::PortId MIN_COMPLETENESS = 5;
inline constexpr visionrt::PortId SEARCH_REGIONS   = 6;
} // namespace in

namespace out {
inline constexpr visionrt::PortId ALIGNMENTS     = 2;
inline constexpr visionrt::PortId FOUND_SEGMENTS = 3;
} // namespace out
} // namespace ports::align_straight_borders

// Locates the two trained reference edges in every image and publishes the reference-to-current
// rigid transform. The tool never resamples the image: downstream tools apply the matrix.
class AlignStraightBordersTool final : public visionrt::IToolInstance
{
public:
    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;

private:
    StraightEdgeModel model_;
};

} // namespace visiontools::recognition
