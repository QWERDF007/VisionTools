#pragma once

#include <visionrt/contracts/Tool.h>

namespace visiontools::recognition {

namespace ports::pick_straight_edge {
inline constexpr char TOOL_ID[] = "PickStraightEdge";

namespace in {
inline constexpr visionrt::PortId IMAGES        = 1;
inline constexpr visionrt::PortId CURSOR        = 2;
inline constexpr visionrt::PortId PHASE         = 3;
inline constexpr visionrt::PortId SEED_ROIS     = 4;
inline constexpr visionrt::PortId ALIGNMENT     = 5;
inline constexpr visionrt::PortId SEARCH_EXTENT = 6;
}

namespace out {
inline constexpr visionrt::PortId PREVIEW_ROIS       = 2;
inline constexpr visionrt::PortId PROPOSAL_ROIS      = 3;
inline constexpr visionrt::PortId CANDIDATE_VALID    = 4;
inline constexpr visionrt::PortId QUALITY            = 5;
inline constexpr visionrt::PortId SUGGESTED_POLARITY = 6;
}
} // namespace ports::pick_straight_edge

// Picks one reference edge at the cursor for the alignment tool's training view. It answers the same
// preview/proposal contract the measurement assistants use, so the host needs no special case.
class PickStraightEdgeTool final : public visionrt::IToolInstance
{
public:
    ~PickStraightEdgeTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;
};

} // namespace visiontools::recognition
