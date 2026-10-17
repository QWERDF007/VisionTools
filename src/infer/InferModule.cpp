#include <visionrt/sdk/Module.h>
#include "YoloModel.h"

#include <cstring>
#include <string>

namespace visiontools::infer {
namespace {

visionrt::IToolInstance* VT_CALL createInferTool(
    const char* registry_name,
    const visionrt::ToolServices* services,
    visionrt::ToolStatus* status) noexcept {
    if (!registry_name) {
        if (status) {
            status->code = -3;
            status->message = "registry_name is null";
        }
        return nullptr;
    }

    try {
        const visionrt::ToolServices s = services ? *services : visionrt::ToolServices{};

        if (std::strcmp(registry_name, "visionrt.dl.yolo") == 0) {
            if (status) { status->code = 0; status->message.clear(); }
            return new visionrt::infer::YoloTool(s);
        }

        if (status) {
            status->code = -3;
            status->message = std::string("Unknown registry_name: ") + registry_name;
        }
        return nullptr;
    } catch (const std::exception& e) {
        if (status) {
            status->code = -3;
            status->message = e.what();
        }
        return nullptr;
    } catch (...) {
        if (status) {
            status->code = -3;
            status->message = "Unknown exception during infer tool creation";
        }
        return nullptr;
    }
}

void VT_CALL destroyInferTool(visionrt::IToolInstance* instance) noexcept {
    delete instance;
}

const visionrt::ModuleApi s_inferModuleApi = {
    visionrt::MODULE_ABI_MAJOR,
    sizeof(visionrt::ModuleApi),
    &createInferTool,
    &destroyInferTool
};

} // namespace
} // namespace visiontools::infer

extern "C" VT_EXPORT const visionrt::ModuleApi* VT_CALL visiontoolsModuleApi() noexcept {
    return &visiontools::infer::s_inferModuleApi;
}
