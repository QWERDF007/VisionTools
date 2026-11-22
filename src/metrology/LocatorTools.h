#pragma once

#include <visionrt/contracts/Tool.h>

namespace visiontools::metrology {

namespace ports::locate {
inline constexpr char CIRCLE_ID[] = "LocateCircle";
inline constexpr char ROW_ID[]    = "LocateRowSpacing";

// Both assistants publish the same probe inputs, so they share one numbered input set.
namespace in {
inline constexpr visionrt::PortId IMAGES        = 1;
inline constexpr visionrt::PortId SEARCH_REGION = 2;
inline constexpr visionrt::PortId SEED_ROIS     = 4;
inline constexpr visionrt::PortId ALIGNMENT     = 5;
} // namespace in

// The probes answer the same preview/proposal contract; the suggestion each one produces differs.
namespace out {
inline constexpr visionrt::PortId PREVIEW_ROIS       = 2;
inline constexpr visionrt::PortId PROPOSAL_ROIS      = 3;
inline constexpr visionrt::PortId CANDIDATE_VALID    = 4;
inline constexpr visionrt::PortId QUALITY            = 5;
inline constexpr visionrt::PortId SUGGESTED_POLARITY = 6;
} // namespace out

namespace circle_out {
inline constexpr visionrt::PortId SUGGESTED_SEARCH = 7;
inline constexpr visionrt::PortId SUGGESTED_SIGMA  = 8;
} // namespace circle_out

namespace row_out {
inline constexpr visionrt::PortId SUGGESTED_WIDTH_MIN = 7;
inline constexpr visionrt::PortId SUGGESTED_WIDTH_MAX = 8;
} // namespace row_out
} // namespace ports::locate

class LocateCircleTool final : public visionrt::IToolInstance
{
public:
    ~LocateCircleTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;
};

class LocateRowSpacingTool final : public visionrt::IToolInstance
{
public:
    ~LocateRowSpacingTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;
};

} // namespace visiontools::metrology
