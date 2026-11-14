#pragma once

#include <visionrt/contracts/Tool.h>

namespace visiontools::preprocess {

namespace ports::image_preprocess {
inline constexpr char TOOL_ID[] = "PreprocessImage";

namespace in {
inline constexpr visionrt::PortId IMAGES = 1;
inline constexpr visionrt::PortId KERNEL = 3;
inline constexpr visionrt::PortId OPERATION = 2;
inline constexpr visionrt::PortId THRESHOLD = 4;
}

namespace out {
inline constexpr visionrt::PortId IMAGES = 2;
}
}

class ImagePreprocessTool final : public visionrt::IToolInstance
{
public:
    ~ImagePreprocessTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;

private:
};

} // namespace visiontools::preprocess
