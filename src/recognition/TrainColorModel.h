#pragma once

#include <visionrt/contracts/Tool.h>

namespace visiontools::recognition {

namespace ports::train_color_model {
inline constexpr char TOOL_ID[] = "TrainColorModel";

namespace in {
inline constexpr visionrt::PortId SET          = 1;
inline constexpr visionrt::PortId LABEL_FIELD  = 2;
inline constexpr visionrt::PortId REGION_FIELD = 3;
} // namespace in

namespace out {
inline constexpr visionrt::PortId MODEL = 2;
}
} // namespace ports::train_color_model

// Turns the labelled records of one training call into the published colour model. It reads the
// pixels the workbench collected and the two field numbers the declaration names; no file, project
// or editor object is involved.
class TrainColorModelTool final : public visionrt::IToolInstance
{
public:
    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;
};

} // namespace visiontools::recognition
