#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <visionrt/sdk/Module.h>
#include <visionrt/sdk/Package.h>
#include <visionrt/sdk/Tool.h>

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <span>
#include <string>
#include <vector>

#ifdef _WIN32
#    include <windows.h>
using LibHandle = HMODULE;

inline LibHandle loadSharedLib(const std::filesystem::path &path)
{
    return LoadLibraryW(path.wstring().c_str());
}

inline void *getProcAddr(LibHandle h, const char *name)
{
    return reinterpret_cast<void *>(GetProcAddress(h, name));
}

inline void freeSharedLib(LibHandle h)
{
    FreeLibrary(h);
}
#else
#    include <dlfcn.h>
using LibHandle = void *;

inline LibHandle loadSharedLib(const std::filesystem::path &path)
{
    return dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
}

inline void *getProcAddr(LibHandle h, const char *name)
{
    return dlsym(h, name);
}

inline void freeSharedLib(LibHandle h)
{
    dlclose(h);
}
#endif

#include <visionrt/sdk/SchemaYaml.h>
#include <yaml-cpp/yaml.h>

#define REQUIRE_STRUCTURE(cond, msg)                                                                        \
    do                                                                                                      \
    {                                                                                                       \
        if (!(cond))                                                                                        \
        {                                                                                                   \
            std::cerr << "[STRUCTURE_ERROR] " << (msg) << " (" #cond ") at line " << __LINE__ << std::endl; \
            std::exit(2);                                                                                   \
        }                                                                                                   \
    }                                                                                                       \
    while (0)

#define REQUIRE_LOADING(cond, msg)                                                                        \
    do                                                                                                    \
    {                                                                                                     \
        if (!(cond))                                                                                      \
        {                                                                                                 \
            std::cerr << "[LOADING_ERROR] " << (msg) << " (" #cond ") at line " << __LINE__ << std::endl; \
            std::exit(3);                                                                                 \
        }                                                                                                 \
    }                                                                                                     \
    while (0)

#define REQUIRE_ABI(cond, msg)                                                                        \
    do                                                                                                \
    {                                                                                                 \
        if (!(cond))                                                                                  \
        {                                                                                             \
            std::cerr << "[ABI_ERROR] " << (msg) << " (" #cond ") at line " << __LINE__ << std::endl; \
            std::exit(4);                                                                             \
        }                                                                                             \
    }                                                                                                 \
    while (0)

#define REQUIRE_EXECUTION(cond, msg)                                                                        \
    do                                                                                                      \
    {                                                                                                       \
        if (!(cond))                                                                                        \
        {                                                                                                   \
            std::cerr << "[EXECUTION_ERROR] " << (msg) << " (" #cond ") at line " << __LINE__ << std::endl; \
            std::exit(5);                                                                                   \
        }                                                                                                   \
    }                                                                                                       \
    while (0)

#define REQUIRE(cond, msg) REQUIRE_EXECUTION(cond, msg)

inline std::set<std::string> loadExpectedToolsFromIndex(const std::filesystem::path &indexPath,
                                                        const std::string           &moduleName)
{
    std::set<std::string> registries;
    const auto            packageDir = indexPath.parent_path();
    if (!std::filesystem::exists(packageDir))
        return registries;
    try
    {
        const auto package = visionrt::readToolsPackage(packageDir);
        for (const auto &tool : package.tools)
        {
            if (!tool.moduleLibrary.empty() && (moduleName.empty() || tool.moduleLibrary.stem() == moduleName))
            {
                registries.insert(tool.registryName);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] Failed to read package: " << e.what() << std::endl;
    }
    return registries;
}

inline void verifyInitFailureAndCleanup(const visionrt::ModuleApi *api, const char *reg,
                                        const visionrt::ToolServices *services)
{
    visionrt::ToolStatus     status;
    visionrt::IToolInstance *errInst = api->create(reg, services, &status);
    REQUIRE(errInst != nullptr, "Create for init failure test must succeed");
    visionrt::Values badInit;
    badInit.emplace("bad_param_invalid", std::int64_t(123));
    auto initStatus = errInst->initialize(badInit);
    REQUIRE(initStatus.code == -2, "Unknown init parameter must return status code -2");
    REQUIRE(errInst->release().code == 0, "Release after init failure must succeed");
    api->destroy(errInst);
}

int main(int argc, char **argv)
{
    std::cout << "[TEST] Running test_installation_consumer (dynamic loading)..." << std::endl;

#ifdef _WIN32
    (void)argc;
    (void)argv;
    int     wargc = 0;
    LPWSTR *wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (!wargv || wargc < 2)
    {
        if (wargv)
            LocalFree(wargv);
        std::cerr << "[ERROR] Missing required DLL or package path argument." << std::endl;
        std::cerr << "Usage: test_installation_consumer <path-to-module-dll-or-package-dir>" << std::endl;
        return 1;
    }
    std::filesystem::path target(wargv[1]);
    LocalFree(wargv);
#else
    if (argc < 2)
    {
        std::cerr << "[ERROR] Missing required DLL or package path argument." << std::endl;
        std::cerr << "Usage: test_installation_consumer <path-to-module-dll-or-package-dir>" << std::endl;
        return 1;
    }
    std::filesystem::path target(argv[1]);
#endif
    if (!std::filesystem::exists(target))
    {
        std::cerr << "[ERROR] Target path does not exist: " << target.string() << std::endl;
        return 1;
    }

    std::filesystem::path dllPath = target;
    std::filesystem::path packageDir;
    if (std::filesystem::is_directory(target))
    {
        packageDir                      = target;
        std::filesystem::path indexYaml = packageDir / "module_index.yaml";
        REQUIRE_STRUCTURE(std::filesystem::exists(indexYaml),
                          "Package directory must contain module_index.yaml (YS-01 / YS-04)");
        try
        {
            const auto indexData = visionrt::parseModuleIndexFile(indexYaml);
            auto       it        = indexData.modules.find("visiontools_preprocess");
            REQUIRE_STRUCTURE(it != indexData.modules.end() && !it->second.library.empty(),
                              "module_index.yaml must declare visiontools_preprocess module");
            dllPath = target / it->second.library;
        }
        catch (const std::exception &e)
        {
            std::cerr << "[STRUCTURE_ERROR] Failed reading module_index.yaml: " << e.what() << std::endl;
            return 2;
        }
        if (!std::filesystem::exists(dllPath))
        {
            std::cerr << "[STRUCTURE_ERROR] Module library not found in package directory: " << dllPath.string()
                      << std::endl;
            return 2;
        }
    }
    else
    {
        if (target.parent_path().filename() == "bin")
        {
            packageDir = target.parent_path().parent_path();
        }
    }

    std::filesystem::path p = std::filesystem::absolute(dllPath);
#ifdef _WIN32
    SetDllDirectoryW(p.parent_path().wstring().c_str());
#endif
    std::cout << "[INFO] Loading module dynamically: " << p.string() << std::endl;
    LibHandle handle = loadSharedLib(p);
    if (!handle)
    {
#ifdef _WIN32
        std::cerr << "[LOADING_ERROR] Failed to load library: " << p.string() << " (GetLastError=" << GetLastError()
                  << ")" << std::endl;
#else
        std::cerr << "[LOADING_ERROR] Failed to load library: " << p.string() << " (" << dlerror() << ")" << std::endl;
#endif
    }
    REQUIRE_LOADING(handle != nullptr, "loadSharedLib must succeed");

    using ModuleApiFn = const visionrt::ModuleApi *(*)();
    auto *apiFn       = reinterpret_cast<ModuleApiFn>(getProcAddr(handle, "visiontoolsModuleApi"));
    REQUIRE_ABI(apiFn != nullptr, "getProcAddr(visiontoolsModuleApi) must succeed");

    const visionrt::ModuleApi *api = apiFn();
    REQUIRE_ABI(api != nullptr, "visiontoolsModuleApi() must return non-null pointer");
    REQUIRE_ABI(api->abi_major == visionrt::MODULE_ABI_MAJOR, "ABI major must match SDK MODULE_ABI_MAJOR");
    REQUIRE_ABI(api->struct_bytes == sizeof(visionrt::ModuleApi),
                "struct_bytes must match sizeof(visionrt::ModuleApi)");
    REQUIRE_ABI(api->create != nullptr, "create pointer must be non-null");
    REQUIRE_ABI(api->destroy != nullptr, "destroy pointer must be non-null");

    visionrt::ToolServices services{visionrt::PathResolver(std::filesystem::current_path())};
    visionrt::ToolStatus   status;

    std::set<std::string> expectedTools
        = {"AdaptiveBinaryTool",      "visionrt.image.crop",       "visionrt.detection.filter",
           "visionrt.roi.projection", "visionrt.image.preprocess", "visionrt.image.read"};

    if (!packageDir.empty())
    {
        std::filesystem::path indexYaml = packageDir / "module_index.yaml";
        REQUIRE(std::filesystem::exists(indexYaml), "Package directory must contain module_index.yaml");
        auto indexTools = loadExpectedToolsFromIndex(indexYaml, "visiontools_preprocess");
        REQUIRE(!indexTools.empty(), "module_index.yaml must contain non-empty visiontools_preprocess tools");
        expectedTools = std::move(indexTools);
    }

    std::set<std::string> actualExecutedTools;

    // 1. Unknown registry -> error
    {
        visionrt::IToolInstance *inst = api->create("unknown.tool.registry", &services, &status);
        REQUIRE(inst == nullptr, "Unknown registry must return nullptr");
        REQUIRE(status.code == -3, "Unknown registry must return status code -3");
    }

    // 2. Null registry -> error
    {
        visionrt::IToolInstance *inst = api->create(nullptr, &services, &status);
        REQUIRE(inst == nullptr, "Null registry must return nullptr");
        REQUIRE(status.code == -3, "Null registry must return status code -3");
    }

    // 3. Safe null destroy
    api->destroy(nullptr);

    // 4. Test tool: adaptive_binary
    visionrt::OwnedImage snapshotImage;
    {
        const char *reg = "AdaptiveBinaryTool";
        verifyInitFailureAndCleanup(api, reg, &services);
        visionrt::IToolInstance *inst = api->create(reg, &services, &status);
        REQUIRE(inst != nullptr, "Create adaptive_binary must return valid instance");
        REQUIRE(inst->initialize({}).code == 0, "Initialize adaptive_binary must succeed");

        cv::Mat gray(32, 32, CV_8UC1);
        for (int r = 0; r < 32; ++r)
        {
            for (int c = 0; c < 32; ++c)
            {
                gray.at<unsigned char>(r, c) = static_cast<unsigned char>((r * 5 + c * 3 + 10) % 256);
            }
        }
        visionrt::Values inputs;
        inputs.emplace("images", visionrt::ImageBatch({visionrt::OwnedImage::copyOf(gray)}));
        inputs.emplace("block_size", std::int64_t(5));
        inputs.emplace("constant_offset", 2.0);

        auto res = inst->execute(inputs);
        REQUIRE(res.status.code == 0, "Execution of adaptive_binary must succeed");

        const auto *outBatch = std::get_if<visionrt::ImageBatch>(&res.outputs["images"]);
        REQUIRE(outBatch != nullptr && !outBatch->images().empty(), "Adaptive binary output must contain images");
        snapshotImage = outBatch->images()[0];

        // Contract error test: wrong type for block_size
        visionrt::Values badInputs;
        badInputs.emplace("images", visionrt::ImageBatch({visionrt::OwnedImage::copyOf(gray)}));
        badInputs.emplace("block_size", "not_an_int");
        auto badRes = inst->execute(badInputs);
        REQUIRE(badRes.status.code == -2, "Contract validation failure must return status -2");

        REQUIRE(inst->release().code == 0, "Release 1 must succeed");
        REQUIRE(inst->release().code == 0, "Release 2 must succeed (idempotent)");
        api->destroy(inst);
        actualExecutedTools.insert(reg);
    }
    // Snapshot persistence after release and destroy
    REQUIRE(snapshotImage.rows() == 32, "Snapshot image must remain valid after instance destroy");
    snapshotImage = visionrt::OwnedImage();

    // 5. Test tool: crop
    {
        const char *reg = "visionrt.image.crop";
        verifyInitFailureAndCleanup(api, reg, &services);
        visionrt::IToolInstance *inst = api->create(reg, &services, &status);
        REQUIRE(inst != nullptr, "Create crop must return valid instance");
        REQUIRE(inst->initialize({}).code == 0, "Initialize crop must succeed");

        cv::Mat              colorImg(64, 64, CV_8UC3, cv::Scalar(10, 20, 30));
        visionrt::OwnedImage inImg = visionrt::OwnedImage::copyOf(colorImg);
        visionrt::Values     inputs;
        inputs.emplace("images", visionrt::ImageBatch({inImg}));
        auto roi = visionrt::Roi::rectangle(10.0F, 10.0F, 20.0F, 20.0F, 0.0F);
        inputs.emplace("rois", visionrt::RoiBatch({{roi}}));

        auto res = inst->execute(inputs);
        REQUIRE_EXECUTION(res.status.code == 0, "Crop tool execution must succeed");

        const auto *outBatch = std::get_if<visionrt::ImageBatch>(&res.outputs["images"]);
        REQUIRE_EXECUTION(outBatch != nullptr && !outBatch->images().empty(), "Crop output must contain images");
        const auto &cropped = outBatch->images()[0];
        REQUIRE_EXECUTION(cropped.rows() == 20 && cropped.cols() == 20, "Cropped size must match ROI (20x20)");
        REQUIRE_EXECUTION(cropped.rowStride() == inImg.rowStride(),
                          "Cropped image rowStride must match source rowStride for zero-copy view");
        REQUIRE_EXECUTION(cropped.readOnlyMat().datastart == inImg.readOnlyMat().datastart,
                          "Cropped image must share datastart pointer (zero-copy view)");

        REQUIRE_EXECUTION(inst->release().code == 0, "Crop release must succeed");
        api->destroy(inst);
        actualExecutedTools.insert(reg);
    }

    // 6. Test tool: detection_filter
    {
        const char *reg = "visionrt.detection.filter";
        verifyInitFailureAndCleanup(api, reg, &services);
        visionrt::IToolInstance *inst = api->create(reg, &services, &status);
        REQUIRE(inst != nullptr, "Create detection.filter must return valid instance");
        REQUIRE(inst->initialize({}).code == 0, "Initialize detection.filter must succeed");

        cv::Mat                   img(100, 100, CV_8UC1, cv::Scalar(0));
        auto                      roi     = visionrt::Roi::rectangle(10.0F, 10.0F, 20.0F, 20.0F, 0.0F);
        std::vector<float>        scores  = {0.85F, 0.35F};
        std::vector<std::int64_t> classes = {1, 2};
        std::vector<std::string>  names   = {"cat", "dog"};
        std::vector<std::int64_t> allowed = {1, 2};

        visionrt::Values inputs;
        inputs.emplace("images", visionrt::ImageBatch({visionrt::OwnedImage::copyOf(img)}));
        inputs.emplace("rois", visionrt::RoiBatch({
                                   {roi, roi}
        }));
        inputs.emplace("scores",
                       visionrt::NumericArray(visionrt::DType::Float32, 2, visionrt::Layout::BatchRagged,
                                              {visionrt::NumericBuffer::copy(visionrt::DType::Float32, {2},
                                                                             std::as_bytes(std::span(scores)))}));
        inputs.emplace("classes",
                       visionrt::NumericArray(visionrt::DType::Int64, 2, visionrt::Layout::BatchRagged,
                                              {visionrt::NumericBuffer::copy(visionrt::DType::Int64, {2},
                                                                             std::as_bytes(std::span(classes)))}));
        inputs.emplace("class_names",
                       visionrt::StringArray(2, visionrt::Layout::BatchRagged, {visionrt::StringBuffer({2}, names)}));
        inputs.emplace("allowed_classes",
                       visionrt::NumericArray(visionrt::DType::Int64, 1, visionrt::Layout::Dense,
                                              {visionrt::NumericBuffer::copy(visionrt::DType::Int64, {2},
                                                                             std::as_bytes(std::span(allowed)))}));
        inputs.emplace("min_score", 0.5);
        inputs.emplace("min_area", 0.0);

        auto res = inst->execute(inputs);
        REQUIRE(res.status.code == 0, "Detection filter execution must succeed");

        const auto *outRois = std::get_if<visionrt::RoiBatch>(&res.outputs["rois"]);
        REQUIRE(outRois != nullptr && outRois->images()[0].size() == 1, "Only score >= 0.5 passes");

        // Contract error test
        visionrt::Values badInputs;
        badInputs.emplace("min_score", "not_a_float");
        auto badRes = inst->execute(badInputs);
        REQUIRE(badRes.status.code == -2, "Contract validation failure on detection filter must return -2");

        REQUIRE(inst->release().code == 0, "Filter release must succeed");
        api->destroy(inst);
        actualExecutedTools.insert(reg);
    }

    // 7. Test tool: roi_projection
    {
        const char *reg = "visionrt.roi.projection";
        verifyInitFailureAndCleanup(api, reg, &services);
        visionrt::IToolInstance *inst = api->create(reg, &services, &status);
        REQUIRE(inst != nullptr, "Create roi.projection must return valid instance");
        REQUIRE(inst->initialize({}).code == 0, "Initialize roi.projection must succeed");

        cv::Mat img(100, 100, CV_8UC1, cv::Scalar(0));
        img(cv::Rect(30, 30, 40, 40)).setTo(cv::Scalar(255));

        visionrt::Values inputs;
        inputs.emplace("images", visionrt::ImageBatch({visionrt::OwnedImage::copyOf(img)}));
        inputs.emplace("axis", visionrt::Utf8("x"));
        inputs.emplace("polarity", visionrt::Utf8("bright"));
        inputs.emplace("threshold", std::int64_t(128));
        inputs.emplace("min_pixels", std::int64_t(10));
        inputs.emplace("min_extent", std::int64_t(10));

        auto res = inst->execute(inputs);
        REQUIRE(res.status.code == 0, "Roi projection execution must succeed");

        const auto *rois = std::get_if<visionrt::RoiBatch>(&res.outputs["rois"]);
        REQUIRE(rois != nullptr && rois->images().size() == 1 && rois->images()[0].size() == 1,
                "Expected 1 projected ROI");

        REQUIRE(inst->release().code == 0, "Roi projection release must succeed");
        api->destroy(inst);
        actualExecutedTools.insert(reg);
    }

    // 8. Test tool: image.preprocess
    {
        const char *reg = "visionrt.image.preprocess";
        verifyInitFailureAndCleanup(api, reg, &services);
        visionrt::IToolInstance *inst = api->create(reg, &services, &status);
        REQUIRE(inst != nullptr, "Create image.preprocess must return valid instance");
        REQUIRE(inst->initialize({}).code == 0, "Initialize image.preprocess must succeed");

        cv::Mat          img(50, 50, CV_8UC1, cv::Scalar(150));
        visionrt::Values inputs;
        inputs.emplace("images", visionrt::ImageBatch({visionrt::OwnedImage::copyOf(img)}));
        inputs.emplace("operation", visionrt::Utf8("threshold"));
        inputs.emplace("threshold", std::int64_t(100));

        auto res = inst->execute(inputs);
        REQUIRE(res.status.code == 0, "Image preprocess execution must succeed");

        const auto *out = std::get_if<visionrt::ImageBatch>(&res.outputs["images"]);
        REQUIRE(out != nullptr && !out->images().empty(), "Output must have ImageBatch");
        REQUIRE(out->images()[0].readOnlyMat().at<unsigned char>(0, 0) == 255, "150 >= 100 -> 255");

        REQUIRE(inst->release().code == 0, "Image preprocess release must succeed");
        api->destroy(inst);
        actualExecutedTools.insert(reg);
    }

    // 9. Test tool: image.read
    {
        const char *reg = "visionrt.image.read";
        verifyInitFailureAndCleanup(api, reg, &services);
        visionrt::IToolInstance *inst = api->create(reg, &services, &status);
        REQUIRE(inst != nullptr, "Create image.read must return valid instance");
        REQUIRE(inst->initialize({}).code == 0, "Initialize image.read must succeed");

        std::string tempPath = "test_consumer_temp_image.png";
        cv::Mat     sample(24, 24, CV_8UC3, cv::Scalar(10, 20, 30));
        cv::imwrite(tempPath, sample);

        visionrt::Values inputs;
        inputs.emplace("mode", visionrt::Utf8("file"));
        inputs.emplace("path", visionrt::Utf8(tempPath));
        inputs.emplace("color", visionrt::Utf8("bgr"));

        auto res = inst->execute(inputs);
        std::filesystem::remove(tempPath);

        REQUIRE(res.status.code == 0, "Image read execution must succeed");
        const auto *batch = std::get_if<visionrt::ImageBatch>(&res.outputs["images"]);
        REQUIRE(batch && batch->images().size() == 1, "Must read 1 image");
        REQUIRE(batch->images()[0].cols() == 24 && batch->images()[0].rows() == 24, "Image dims must match 24x24");

        REQUIRE(inst->release().code == 0, "Image read release must succeed");
        api->destroy(inst);
        actualExecutedTools.insert(reg);
    }

    // Verify tool coverage: expected == actual
    std::cout << "[INFO] Installation consumer summary:" << std::endl;
    std::cout << "  DLL path: " << p.string() << std::endl;
    std::cout << "  Expected tools count: " << expectedTools.size() << std::endl;
    std::cout << "  Actual executed tools count: " << actualExecutedTools.size() << std::endl;
    REQUIRE(actualExecutedTools == expectedTools, "All expected tools must be tested and succeed without skipping");

    freeSharedLib(handle);
    std::cout << "[PASS] test_installation_consumer passed successfully." << std::endl;
    return 0;
}
