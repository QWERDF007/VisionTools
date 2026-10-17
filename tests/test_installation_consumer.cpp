#include <visionrt/sdk/Module.h>
#include <visionrt/sdk/Tool.h>

#include <iostream>
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

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

    if (argc < 2) {
        std::cerr << "[ERROR] Missing required DLL or package path argument." << std::endl;
        std::cerr << "Usage: test_installation_consumer <path-to-module-dll-or-package-dir>" << std::endl;
        return 1;
    }

    std::filesystem::path target(argv[1]);
    if (!std::filesystem::exists(target)) {
        std::cerr << "[ERROR] Target path does not exist: " << target.string() << std::endl;
        return 1;
    }

    std::filesystem::path dllPath = target;
    if (std::filesystem::is_directory(target)) {
        dllPath = target / "bin" / "visiontools_preprocess.dll";
        if (!std::filesystem::exists(dllPath)) {
            std::cerr << "[ERROR] visiontools_preprocess.dll not found in package directory: " << target.string() << std::endl;
            return 1;
        }
    }

    std::filesystem::path p = std::filesystem::absolute(dllPath);
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
    int toolsTested = 0;
    int successfulCalls = 0;

    // 1. Unknown registry -> error
    {
        visionrt::IToolInstance* inst = api->create("unknown.tool.registry", &services, &status);
        REQUIRE(inst == nullptr, "Unknown registry must return nullptr");
        REQUIRE(status.code == -3, "Unknown registry must return status code -3");
    }

    // 2. Null registry -> error
    {
        visionrt::IToolInstance* inst = api->create(nullptr, &services, &status);
        REQUIRE(inst == nullptr, "Null registry must return nullptr");
        REQUIRE(status.code == -3, "Null registry must return status code -3");
    }

    // 3. Safe null destroy
    api->destroy(nullptr);

    // 4. Test tool: adaptive_binary (valid + contract error)
    visionrt::OwnedImage snapshotImage;
    {
        visionrt::IToolInstance* inst = api->create("visiontools.preprocess.adaptive_binary", &services, &status);
        REQUIRE(inst != nullptr, "Create adaptive_binary must return valid instance");
        toolsTested++;

        cv::Mat gray(32, 32, CV_8UC1, cv::Scalar(100));
        visionrt::Values inputs;
        inputs.emplace("images", visionrt::ImageBatch({visionrt::OwnedImage::copyOf(gray)}));
        inputs.emplace("block_size", std::int64_t(5));
        inputs.emplace("constant_offset", 0.0);

        auto res = inst->execute(inputs);
        REQUIRE(res.status.code == 0, "Execution of adaptive_binary must succeed");
        successfulCalls++;

        const auto* outBatch = std::get_if<visionrt::ImageBatch>(&res.outputs["images"]);
        REQUIRE(outBatch != nullptr && !outBatch->images().empty(), "Adaptive binary output must contain images");
        snapshotImage = outBatch->images()[0];

        // Contract error test: invalid block_size (non-odd / out of range) or wrong type
        visionrt::Values badInputs;
        badInputs.emplace("images", visionrt::ImageBatch({visionrt::OwnedImage::copyOf(gray)}));
        badInputs.emplace("block_size", "not_an_int"); // wrong type for contract validation
        auto badRes = inst->execute(badInputs);
        REQUIRE(badRes.status.code == -2, "Contract validation failure must return status -2");

        // Release idempotency
        REQUIRE(inst->release().code == 0, "Release 1 must succeed");
        REQUIRE(inst->release().code == 0, "Release 2 must succeed (idempotent)");
        api->destroy(inst);
    }
    // Snapshot persistence after release and destroy
    REQUIRE(snapshotImage.rows() == 32, "Snapshot image must remain valid after instance destroy");
    // Explicitly reset snapshot owner while module DLL is still loaded (RS-05: module resides until owner destroyed)
    snapshotImage = visionrt::OwnedImage();

    // 5. Test tool: crop
    {
        visionrt::IToolInstance* inst = api->create("visionrt.image.crop", &services, &status);
        if (inst != nullptr) {
            toolsTested++;

            cv::Mat colorImg(64, 64, CV_8UC3, cv::Scalar(10, 20, 30));
            visionrt::Values inputs;
            inputs.emplace("images", visionrt::ImageBatch({visionrt::OwnedImage::copyOf(colorImg)}));
            auto roi = visionrt::Roi::rectangle(10.0F, 10.0F, 20.0F, 20.0F, 0.0F);
            inputs.emplace("rois", visionrt::RoiBatch({{roi}}));

            auto res = inst->execute(inputs);
            REQUIRE(res.status.code == 0, "Crop tool execution must succeed");
            successfulCalls++;

            const auto* outBatch = std::get_if<visionrt::ImageBatch>(&res.outputs["images"]);
            REQUIRE(outBatch != nullptr && !outBatch->images().empty(), "Crop output must contain images");
            REQUIRE(outBatch->images()[0].rows() == 20 && outBatch->images()[0].cols() == 20, "Cropped size must match ROI (20x20)");

            REQUIRE(inst->release().code == 0, "Crop release 1 must succeed");
            REQUIRE(inst->release().code == 0, "Crop release 2 must succeed");
            api->destroy(inst);
        }
    }

    // 6. Test tool: detection_filter
    {
        visionrt::IToolInstance* inst = api->create("visionrt.detection.filter", &services, &status);
        if (inst != nullptr) {
            toolsTested++;

            cv::Mat img(100, 100, CV_8UC1, cv::Scalar(0));
            auto roi = visionrt::Roi::rectangle(10.0F, 10.0F, 20.0F, 20.0F, 0.0F);
            std::vector<float> scores = {0.85F, 0.35F};
            std::vector<std::int64_t> classes = {1, 2};
            std::vector<std::string> names = {"cat", "dog"};
            std::vector<std::int64_t> allowed = {1, 2};

            visionrt::Values inputs;
            inputs.emplace("images", visionrt::ImageBatch({visionrt::OwnedImage::copyOf(img)}));
            inputs.emplace("rois", visionrt::RoiBatch({{roi, roi}}));
            inputs.emplace("scores", visionrt::NumericArray(visionrt::DType::Float32, 2, visionrt::Layout::BatchRagged,
                {visionrt::NumericBuffer::copy(visionrt::DType::Float32, {2}, std::as_bytes(std::span(scores)))}));
            inputs.emplace("classes", visionrt::NumericArray(visionrt::DType::Int64, 2, visionrt::Layout::BatchRagged,
                {visionrt::NumericBuffer::copy(visionrt::DType::Int64, {2}, std::as_bytes(std::span(classes)))}));
            inputs.emplace("class_names", visionrt::StringArray(2, visionrt::Layout::BatchRagged, {visionrt::StringBuffer({2}, names)}));
            inputs.emplace("allowed_classes", visionrt::NumericArray(visionrt::DType::Int64, 1, visionrt::Layout::Dense,
                {visionrt::NumericBuffer::copy(visionrt::DType::Int64, {2}, std::as_bytes(std::span(allowed)))}));
            inputs.emplace("min_score", 0.5);
            inputs.emplace("min_area", 0.0);

            auto res = inst->execute(inputs);
            REQUIRE(res.status.code == 0, "Detection filter execution must succeed");
            successfulCalls++;

            // Contract error test: wrong type for min_score
            visionrt::Values badInputs;
            badInputs.emplace("min_score", "not_a_float");
            auto badRes = inst->execute(badInputs);
            REQUIRE(badRes.status.code == -2, "Contract validation failure on detection filter must return -2");

            REQUIRE(inst->release().code == 0, "Filter release 1 must succeed");
            REQUIRE(inst->release().code == 0, "Filter release 2 must succeed");
            api->destroy(inst);
        }
    }

    std::cout << "[INFO] Installation consumer summary:" << std::endl;
    std::cout << "  DLL path: " << p.string() << std::endl;
    std::cout << "  Tools tested: " << toolsTested << std::endl;
    std::cout << "  Successful calls: " << successfulCalls << std::endl;

    freeSharedLib(handle);
    std::cout << "[PASS] test_installation_consumer passed successfully." << std::endl;
    return 0;
}
