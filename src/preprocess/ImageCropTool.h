#pragma once

#include <visionrt/sdk/Tool.h>

namespace visiontools::preprocess {

class ImageCropTool final : public visionrt::IToolInstance {
public:
    explicit ImageCropTool(const visionrt::ToolServices& services);
    ~ImageCropTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values& parameters) override;
    visionrt::ToolResult execute(const visionrt::Values& inputs) override;
    visionrt::ToolStatus release() override;

private:
    visionrt::ToolServices services_;
};

} // namespace visiontools::preprocess
