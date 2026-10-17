#include <visionrt/sdk/Module.h>

#include "ReadImageTool.h"
#include "RoiProjectionTool.h"
#include "ImageCropTool.h"
#include "ImagePreprocessTool.h"
#include "DetectionFilterTool.h"
#include "AdaptiveBinaryTool.h"

#include <cstring>
#include <string>

namespace visiontools::preprocess {
namespace {

visionrt::IToolInstance* VT_CALL createPreprocessTool(
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

        if (std::strcmp(registry_name, "visionrt.image.read") == 0) {
            if (status) { status->code = 0; status->message.clear(); }
            return new ReadImageTool(s);
        }
        if (std::strcmp(registry_name, "visionrt.roi.projection") == 0) {
            if (status) { status->code = 0; status->message.clear(); }
            return new RoiProjectionTool(s);
        }
        if (std::strcmp(registry_name, "visionrt.image.crop") == 0) {
            if (status) { status->code = 0; status->message.clear(); }
            return new ImageCropTool(s);
        }
        if (std::strcmp(registry_name, "visionrt.image.preprocess") == 0) {
            if (status) { status->code = 0; status->message.clear(); }
            return new ImagePreprocessTool(s);
        }
        if (std::strcmp(registry_name, "visionrt.detection.filter") == 0) {
            if (status) { status->code = 0; status->message.clear(); }
            return new DetectionFilterTool(s);
        }
        if (std::strcmp(registry_name, "visiontools.preprocess.adaptive_binary") == 0) {
            if (status) { status->code = 0; status->message.clear(); }
            return new AdaptiveBinaryTool(s);
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
            status->message = "Unknown exception during tool creation";
        }
        return nullptr;
    }
}

void VT_CALL destroyPreprocessTool(visionrt::IToolInstance* instance) noexcept {
    delete instance;
}

const visionrt::ModuleApi s_preprocessModuleApi = {
    visionrt::MODULE_ABI_MAJOR,
    sizeof(visionrt::ModuleApi),
    &createPreprocessTool,
    &destroyPreprocessTool
};

} // namespace
} // namespace visiontools::preprocess

extern "C" VT_EXPORT const visionrt::ModuleApi* VT_CALL visiontoolsModuleApi() noexcept {
    return &visiontools::preprocess::s_preprocessModuleApi;
}
