#pragma once

#include <visionrt/contracts/Tool.h>

#include <filesystem>
#include <functional>

namespace visiontools::preprocess {

class ReadImageTool final : public visionrt::IToolInstance
{
public:
    explicit ReadImageTool(const visionrt::ToolServices &services);
    ~ReadImageTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;

private:
    visionrt::ToolServices services_;
};

#ifdef VISIONTOOLS_TESTING
namespace testing {
// Test-only read boundary: runs after the size checks and before stream.read.
extern std::function<void(const std::filesystem::path &)> beforeRead;
} // namespace testing
#endif

} // namespace visiontools::preprocess
