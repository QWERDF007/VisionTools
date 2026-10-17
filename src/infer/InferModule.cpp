#include <visionrt/sdk/Module.h>
#include "visiontools_infer_factory.inc"

extern "C" VT_EXPORT const visionrt::ModuleApi* VT_CALL visiontoolsModuleApi() noexcept {
    return &visionrt::infer::s_generatedModuleApi;
}
