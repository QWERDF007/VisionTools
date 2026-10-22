#include "../ToolFactory.h"
#include "YoloModel.h"

#include <visionrt/contracts/Module.h>

namespace visionrt::infer { namespace {
visionrt::ToolStatus VT_CALL registerTools(visionrt::IToolRegistry &registry) noexcept
{
    const visionrt::ToolFactoryDescriptor factories[]{
        {ports::yolo::TOOL_ID, visiontools::createTool<YoloTool>, visiontools::destroyTool},
    };
    for (const auto &factory : factories)
        if (auto status = registry.add(factory); !status)
            return status;
    return {};
}
}}

extern "C" VT_EXPORT const visionrt::ModuleApi *VT_CALL visiontoolsModuleApi() noexcept
{
    static const visionrt::ModuleApi api{visionrt::MODULE_ABI_MAJOR, sizeof(visionrt::ModuleApi),
                                         visionrt::infer::registerTools};
    return &api;
}
