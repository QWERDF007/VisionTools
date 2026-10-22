#pragma once

#include <visionrt/contracts/Tool.h>

namespace visiontools::preprocess {

namespace ports::image_preprocess {
inline constexpr char TOOL_ID[] = "visionrt.image.preprocess";

namespace in {
inline constexpr char IMAGES[]    = "images";
inline constexpr char KERNEL[]    = "kernel";
inline constexpr char OPERATION[] = "operation";
inline constexpr char THRESHOLD[] = "threshold";
}

namespace out {
inline constexpr char IMAGES[] = "images";
}
}

class ImagePreprocessTool final : public visionrt::IToolInstance
{
public:
    explicit ImagePreprocessTool(const visionrt::ToolServices &services);
    ~ImagePreprocessTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;

private:
    visionrt::ToolServices services_;
};

} // namespace visiontools::preprocess
