#pragma once

#include <visionrt/contracts/Tool.h>

namespace visiontools::preprocess {

namespace ports::adaptive_binary {
inline constexpr char TOOL_ID[] = "visiontools.preprocess.adaptive_binary";

namespace in {
inline constexpr char BLOCK_SIZE[]      = "block_size";
inline constexpr char CONSTANT_OFFSET[] = "constant_offset";
inline constexpr char IMAGES[]          = "images";
}

namespace out {
inline constexpr char IMAGES[] = "images";
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
