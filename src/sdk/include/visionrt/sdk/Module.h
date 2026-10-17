#pragma once

#include <visionrt/sdk/Tool.h>
#include <cstdint>

#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(VISIONTOOLS_EXPORTS)
    #define VT_EXPORT __declspec(dllexport)
  #else
    #define VT_EXPORT __declspec(dllimport)
  #endif
  #define VT_CALL __cdecl
#else
  #define VT_EXPORT __attribute__((visibility("default")))
  #define VT_CALL
#endif

namespace visionrt {

struct ModuleApi {
    std::uint32_t abi_major;
    std::uint32_t struct_bytes;
    IToolInstance* (VT_CALL *create)(const char* registry_name,
                                     const ToolServices* services,
                                     ToolStatus* status) noexcept;
    void (VT_CALL *destroy)(IToolInstance*) noexcept;
};

inline constexpr std::uint32_t MODULE_ABI_MAJOR = 1;

} // namespace visionrt

extern "C" VT_EXPORT const visionrt::ModuleApi* VT_CALL
visiontoolsModuleApi() noexcept;
