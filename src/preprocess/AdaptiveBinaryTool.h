#pragma once

#include <visionrt/sdk/Tool.h>

namespace visiontools::preprocess {

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
