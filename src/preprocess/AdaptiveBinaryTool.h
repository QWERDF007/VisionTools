#pragma once

#include <visionrt/contracts/Tool.h>

namespace visiontools::preprocess {

namespace ports::adaptive_binary {
inline constexpr char TOOL_ID[] = "visiontools.preprocess.adaptive_binary";

namespace in {
inline constexpr visionrt::PortId BLOCK_SIZE = 2;
inline constexpr visionrt::PortId CONSTANT_OFFSET = 3;
inline constexpr visionrt::PortId IMAGES = 1;
}

namespace out {
inline constexpr visionrt::PortId IMAGES = 2;
}
}

class AdaptiveBinaryTool final : public visionrt::IToolInstance
{
public:
    explicit AdaptiveBinaryTool(const visionrt::ToolServices &services);
    ~AdaptiveBinaryTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;

private:
    visionrt::ToolServices services_;
};

} // namespace visiontools::preprocess
