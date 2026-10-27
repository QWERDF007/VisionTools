#include "../ToolFactory.h"
#include "AdaptiveBinaryTool.h"
#include "BrightnessTool.h"
#include "DetectionFilterTool.h"
#include "ImageCropTool.h"
#include "ImagePreprocessTool.h"
#include "ReadImageTool.h"
#include "RotateImageTool.h"
#include "RoiProjectionTool.h"

#include <visionrt/contracts/FunctionTool.h>
#include <visionrt/contracts/Module.h>

namespace visiontools::preprocess { namespace {
visionrt::ToolStatus VT_CALL registerTools(visionrt::IToolRegistry &registry) noexcept
{
    const visionrt::ToolFactoryDescriptor factories[]{
        {      ports::image_read::TOOL_ID,       visiontools::createTool<ReadImageTool>, visiontools::destroyTool},
        {  ports::roi_projection::TOOL_ID,   visiontools::createTool<RoiProjectionTool>, visiontools::destroyTool},
        {      ports::image_crop::TOOL_ID,       visiontools::createTool<ImageCropTool>, visiontools::destroyTool},
        {ports::image_preprocess::TOOL_ID, visiontools::createTool<ImagePreprocessTool>, visiontools::destroyTool},
        {ports::detection_filter::TOOL_ID, visiontools::createTool<DetectionFilterTool>, visiontools::destroyTool},
        { ports::adaptive_binary::TOOL_ID,  visiontools::createTool<AdaptiveBinaryTool>, visiontools::destroyTool},
        {     ports::image_rotate::TOOL_ID,     visiontools::createTool<RotateImageTool>, visiontools::destroyTool},
    };
    for (const auto &factory : factories)
        if (auto status = registry.add(factory); !status)
            return status;
    // 无状态工具走函数入口：同一 IToolInstance 调度，实例不保留任何业务状态。
    return visionrt::registerFunction<&brightnessTool>(registry, ports::image_brightness::TOOL_ID);
}
}}

extern "C" VT_EXPORT const visionrt::ModuleApi *VT_CALL visiontoolsModuleApi() noexcept
{
    static const visionrt::ModuleApi api{visionrt::MODULE_ABI_MAJOR, sizeof(visionrt::ModuleApi),
                                         visiontools::preprocess::registerTools};
    return &api;
}
