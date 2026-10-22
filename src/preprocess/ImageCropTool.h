#pragma once

#include <visionrt/contracts/Tool.h>

#include <cstddef>

namespace visiontools::preprocess {

class ImageCropTool final : public visionrt::IToolInstance
{
public:
    explicit ImageCropTool(const visionrt::ToolServices &services);
    ~ImageCropTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;

private:
    visionrt::ToolServices services_;
};

#ifdef VISIONTOOLS_TESTING
namespace testing {
// Test-only observation: number of mask builds in the current execute call.
extern std::size_t maskBuildCount;
} // namespace testing
#endif

} // namespace visiontools::preprocess
