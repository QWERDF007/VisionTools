#include <visionrt/sdk/Module.h>

#if __has_include("visiontools_infer_factory.inc")
#include "visiontools_infer_factory.inc"
#elif __has_include("generated/visiontools_infer_factory.inc")
#include "generated/visiontools_infer_factory.inc"
#else
#include "../../src/generated/visiontools_infer_factory.inc"
#endif

extern "C" VT_EXPORT const visionrt::ModuleApi* VT_CALL visiontoolsModuleApi() noexcept {
    return &visionrt::infer::s_generatedModuleApi;
}
