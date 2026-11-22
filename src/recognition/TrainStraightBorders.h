#pragma once

#include <visionrt/contracts/Tool.h>

namespace visiontools::recognition {

namespace ports::train_straight_borders {
inline constexpr char TOOL_ID[] = "TrainStraightBorders";

namespace in {
inline constexpr visionrt::PortId SET              = 1;
inline constexpr visionrt::PortId REGION_FIELD     = 2;
inline constexpr visionrt::PortId CONTRAST         = 3;
inline constexpr visionrt::PortId EDGE_WIDTH       = 4;
inline constexpr visionrt::PortId POLARITY         = 5;
inline constexpr visionrt::PortId MIN_COMPLETENESS = 6;
} // namespace in

namespace out {
inline constexpr visionrt::PortId MODEL    = 2;
inline constexpr visionrt::PortId SEGMENTS = 3;
} // namespace out
} // namespace ports::train_straight_borders

// Fits the two drawn search bands against the reference picture and publishes the alignment model.
// The pixels decide the endpoints; the drawn bands only say where to look, so a hand-drawn pair and
// a picked pair produce the same model for the same edges.
class TrainStraightBordersTool final : public visionrt::IToolInstance
{
public:
    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;
};

} // namespace visiontools::recognition
