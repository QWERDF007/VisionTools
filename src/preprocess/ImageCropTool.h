#pragma once

#include <visionrt/contracts/Tool.h>

#include <cstddef>

namespace visiontools::preprocess {

namespace ports::image_crop {
inline constexpr char TOOL_ID[] = "CropImage";

namespace in {
inline constexpr visionrt::PortId IMAGES = 1;
inline constexpr visionrt::PortId RECT = 2;
}

namespace out {
inline constexpr visionrt::PortId IMAGES = 2;
inline constexpr visionrt::PortId ORIGINS = 3;
}
}

class ImageCropTool final : public visionrt::IToolInstance
{
public:
    ~ImageCropTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;

private:
};

#ifdef VISIONTOOLS_TESTING
namespace testing {
// Test-only observation: number of mask builds in the current execute call.
extern std::size_t maskBuildCount;
} // namespace testing
#endif

} // namespace visiontools::preprocess
