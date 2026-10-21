#pragma once

#include <visionrt/contracts/Tool.h>

namespace visiontools::preprocess {

class RoiProjectionTool final : public visionrt::IToolInstance
{
public:
    explicit RoiProjectionTool(const visionrt::ToolServices &services);
    ~RoiProjectionTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;

private:
    visionrt::ToolServices services_;
};

} // namespace visiontools::preprocess
