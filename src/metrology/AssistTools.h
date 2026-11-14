#pragma once

#include <visionrt/contracts/Tool.h>

namespace visiontools::metrology {

namespace ports::assist {
inline constexpr char CIRCLE_ID[] = "PickCircle";
inline constexpr char ROW_ID[]    = "PickRowSpacing";

// Both assistants publish the same probe inputs, so they share one numbered input set.
namespace in {
inline constexpr visionrt::PortId IMAGES        = 1;
inline constexpr visionrt::PortId CURSOR        = 2;
inline constexpr visionrt::PortId PHASE         = 3;
inline constexpr visionrt::PortId SEED_ROIS     = 4;
inline constexpr visionrt::PortId ALIGNMENT     = 5;
inline constexpr visionrt::PortId SEARCH_EXTENT = 6;
}

// The probes answer the same preview/proposal contract; the suggestion each one produces differs.
namespace out {
inline constexpr visionrt::PortId PREVIEW_ROIS       = 2;
inline constexpr visionrt::PortId PROPOSAL_ROIS      = 3;
inline constexpr visionrt::PortId CANDIDATE_VALID    = 4;
inline constexpr visionrt::PortId QUALITY            = 5;
inline constexpr visionrt::PortId SUGGESTED_POLARITY = 6;
}

namespace circle_out {
inline constexpr visionrt::PortId SUGGESTED_SEARCH = 7;
}

namespace row_out {
inline constexpr visionrt::PortId SUGGESTED_WIDTH_MIN = 7;
inline constexpr visionrt::PortId SUGGESTED_WIDTH_MAX = 8;
}
}

class AssistCircleTool final : public visionrt::IToolInstance
{
public:
    ~AssistCircleTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;
};

class AssistRowSpacingTool final : public visionrt::IToolInstance
{
public:
    ~AssistRowSpacingTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;
};

} // namespace visiontools::metrology
