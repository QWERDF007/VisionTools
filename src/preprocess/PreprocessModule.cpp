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
    if (auto s = visionrt::registerClass<ReadImageTool>(registry, ports::image_read::TOOL_ID); !s)
        return s;
    if (auto s = visionrt::registerClass<RoiProjectionTool>(registry, ports::roi_projection::TOOL_ID); !s)
        return s;
    if (auto s = visionrt::registerClass<ImageCropTool>(registry, ports::image_crop::TOOL_ID); !s)
        return s;
    if (auto s = visionrt::registerClass<ImagePreprocessTool>(registry, ports::image_preprocess::TOOL_ID); !s)
        return s;
    if (auto s = visionrt::registerClass<DetectionFilterTool>(registry, ports::detection_filter::TOOL_ID); !s)
        return s;
    if (auto s = visionrt::registerClass<AdaptiveBinaryTool>(registry, ports::adaptive_binary::TOOL_ID); !s)
        return s;
    if (auto s = visionrt::registerClass<RotateImageTool>(registry, ports::image_rotate::TOOL_ID); !s)
        return s;
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
