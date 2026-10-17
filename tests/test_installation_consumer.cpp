#include <visionrt/sdk/Module.h>
#include <visionrt/sdk/Tool.h>

#include <iostream>
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <string>

#ifdef _WIN32
#include <windows.h>
using LibHandle = HMODULE;
inline LibHandle loadSharedLib(const std::string& path) { return LoadLibraryA(path.c_str()); }
inline void* getProcAddr(LibHandle h, const char* name) { return reinterpret_cast<void*>(GetProcAddress(h, name)); }
inline void freeSharedLib(LibHandle h) { FreeLibrary(h); }
#else
#include <dlfcn.h>
using LibHandle = void*;
inline LibHandle loadSharedLib(const std::string& path) { return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL); }
inline void* getProcAddr(LibHandle h, const char* name) { return dlsym(h, name); }
inline void freeSharedLib(LibHandle h) { dlclose(h); }
#endif

#define REQUIRE(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed: " << #cond << " (" << msg << ") at line " << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while(0)

int main(int argc, char** argv) {
    std::cout << "[TEST] Running test_installation_consumer (dynamic loading)..." << std::endl;

    std::string dllPath;
    if (argc > 1) {
        dllPath = argv[1];
    } else {
        // Search in common build/dist locations
        std::vector<std::string> candidates = {
            "dist/VisionTools/bin/visiontools_preprocess.dll",
            "../dist/VisionTools/bin/visiontools_preprocess.dll",
            "src/preprocess/visiontools_preprocess.dll",
            "src/preprocess/Release/visiontools_preprocess.dll",
            "bin/visiontools_preprocess.dll",
            "bin/Release/visiontools_preprocess.dll"
        };
        for (const auto& c : candidates) {
            if (std::filesystem::exists(c)) {
                dllPath = c;
                break;
            }
        }
    }

    if (dllPath.empty() || !std::filesystem::exists(dllPath)) {
        std::cout << "[WARN] DLL not found for dynamic loading test. Path candidate searched: " << dllPath << std::endl;
        return 0; // Skip if run without compiled DLL
    }

    std::filesystem::path p(dllPath);
    p = std::filesystem::absolute(p);
#ifdef _WIN32
    SetDllDirectoryA(p.parent_path().string().c_str());
#endif
    std::cout << "[INFO] Loading module dynamically: " << p.string() << std::endl;
    LibHandle handle = loadSharedLib(p.string());
    if (!handle) {
#ifdef _WIN32
        std::cerr << "[ERROR] Failed to load library: " << p.string() << " (GetLastError=" << GetLastError() << ")" << std::endl;
#else
        std::cerr << "[ERROR] Failed to load library: " << p.string() << " (" << dlerror() << ")" << std::endl;
#endif
    }
    REQUIRE(handle != nullptr, "loadSharedLib must succeed");

    using ModuleApiFn = const visionrt::ModuleApi*(*)();
    auto* apiFn = reinterpret_cast<ModuleApiFn>(getProcAddr(handle, "visiontoolsModuleApi"));
    REQUIRE(apiFn != nullptr, "getProcAddr(visiontoolsModuleApi) must succeed");

    const visionrt::ModuleApi* api = apiFn();
    REQUIRE(api != nullptr, "visiontoolsModuleApi() must return non-null pointer");
    REQUIRE(api->abi_major == visionrt::MODULE_ABI_MAJOR, "ABI major must match SDK MODULE_ABI_MAJOR");
    REQUIRE(api->struct_bytes == sizeof(visionrt::ModuleApi), "struct_bytes must match sizeof(visionrt::ModuleApi)");
    REQUIRE(api->create != nullptr, "create pointer must be non-null");
    REQUIRE(api->destroy != nullptr, "destroy pointer must be non-null");

    visionrt::ToolServices services;
    visionrt::ToolStatus status;

    // Unknown registry -> error
    {
        visionrt::IToolInstance* inst = api->create("unknown.tool.registry", &services, &status);
        REQUIRE(inst == nullptr, "Unknown registry must return nullptr");
        REQUIRE(status.code == -3, "Unknown registry must return status code -3");
    }

    // Valid tool creation and execution via dynamic loading
    {
        visionrt::IToolInstance* inst = api->create("visiontools.preprocess.adaptive_binary", &services, &status);
        REQUIRE(inst != nullptr, "Create known tool must return valid instance");

        cv::Mat gray(32, 32, CV_8UC1, cv::Scalar(100));
        visionrt::Values inputs;
        inputs.emplace("images", visionrt::ImageBatch({visionrt::OwnedImage::copyOf(gray)}));
        inputs.emplace("block_size", std::int64_t(5));
        inputs.emplace("constant_offset", 0.0);

        auto res = inst->execute(inputs);
        REQUIRE(res.status.code == 0, "Execution via dynamic loading must succeed");

        inst->release();
        api->destroy(inst);
    }

    // Safe null destroy
    api->destroy(nullptr);

    freeSharedLib(handle);
    std::cout << "[PASS] test_installation_consumer passed successfully." << std::endl;
    return 0;
}
