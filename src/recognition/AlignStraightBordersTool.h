#pragma once

#include "StraightEdgeModel.h"

#include <visionrt/contracts/Tool.h>

namespace visiontools::recognition {

namespace ports::align_straight_borders {
inline constexpr char TOOL_ID[] = "AlignStraightBorders";

namespace train {
inline constexpr visionrt::PortId MODEL_PATH = 1;
inline constexpr visionrt::PortId CONTRAST = 3;
inline constexpr visionrt::PortId EDGE_WIDTH = 4;
inline constexpr visionrt::PortId POLARITY = 5;
inline constexpr visionrt::PortId MIN_COMPLETENESS = 6;
inline constexpr visionrt::PortId SEARCH_REGIONS = 2;
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
inline constexpr visionrt::PortId MODEL_PATH = 4;
inline constexpr visionrt::PortId ALIGNMENTS     = 2;
inline constexpr visionrt::PortId FOUND_SEGMENTS = 3;
} // namespace out
} // namespace ports::align_straight_borders

// Locates the two trained reference edges in every image and publishes the reference-to-current
// rigid transform. The tool never resamples the image: downstream tools apply the matrix.
class AlignStraightBordersTool final : public visionrt::IToolInstance
{
public:
    visionrt::ToolStatus initialize(const visionrt::ToolConfiguration &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolResult train(const visionrt::TrainingSet &, const visionrt::Values &) override;
    visionrt::ToolStatus release() override;

private:
    std::string modelPath_;
    StraightEdgeModel model_;
};

} // namespace visiontools::recognition
