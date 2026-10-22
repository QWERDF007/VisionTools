#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <visionrt/contracts/Module.h>
#include <visionrt/contracts/Package.h>
#include <visionrt/contracts/Tool.h>

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

#include <visionrt/contracts/SchemaYaml.h>
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
            const auto &entry = package.index.tools.at(tool.toolId);
            if (!tool.moduleLibrary.empty() && (moduleName.empty() || entry.module == moduleName))
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
    REQUIRE(initStatus.message.find("初始化参数") != std::string::npos,
            "Generated wrapper init error must be Chinese");
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
    std::vector<std::filesystem::path> modulePaths;
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
            for (const auto &[name, module] : indexData.modules)
                modulePaths.push_back(target / module.library);
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

    if (modulePaths.empty())
        modulePaths.push_back(dllPath);
    LibHandle handle = nullptr;
    const visionrt::ModuleApi *api = nullptr;
    using ModuleApiFn = const visionrt::ModuleApi *(*)();
    for (const auto &modulePath : modulePaths)
    {
        std::filesystem::path moduleFile = std::filesystem::absolute(modulePath);
#ifdef _WIN32
        SetDllDirectoryW(moduleFile.parent_path().wstring().c_str());
#endif
        std::cout << "[INFO] Loading module dynamically: " << moduleFile.string() << std::endl;
        LibHandle moduleHandle = loadSharedLib(moduleFile);
        if (!moduleHandle)
        {
#ifdef _WIN32
            std::cerr << "[LOADING_ERROR] Failed to load library: " << moduleFile.string() << " (GetLastError=" << GetLastError()
                      << ")" << std::endl;
#else
            std::cerr << "[LOADING_ERROR] Failed to load library: " << moduleFile.string() << " (" << dlerror() << ")" << std::endl;
#endif
        }
        REQUIRE_LOADING(moduleHandle != nullptr, "loadSharedLib must succeed");

        auto *apiFn = reinterpret_cast<ModuleApiFn>(getProcAddr(moduleHandle, "visiontoolsModuleApi"));
        REQUIRE_ABI(apiFn != nullptr, "getProcAddr(visiontoolsModuleApi) must succeed");

        const visionrt::ModuleApi *moduleApi = apiFn();
        REQUIRE_ABI(moduleApi != nullptr, "visiontoolsModuleApi() must return non-null pointer");
        REQUIRE_ABI(moduleApi->abi_major == visionrt::MODULE_ABI_MAJOR, "ABI major must match SDK MODULE_ABI_MAJOR");
        REQUIRE_ABI(moduleApi->struct_bytes == sizeof(visionrt::ModuleApi),
                    "struct_bytes must match sizeof(visionrt::ModuleApi)");
        REQUIRE_ABI(moduleApi->create != nullptr, "create pointer must be non-null");
        REQUIRE_ABI(moduleApi->destroy != nullptr, "destroy pointer must be non-null");
        if (std::filesystem::equivalent(moduleFile, dllPath))
        {
            handle = moduleHandle;
            api = moduleApi;
        }
        else
            freeSharedLib(moduleHandle);
    }
    REQUIRE_ABI(api != nullptr, "Preprocess module must provide an API");
    std::cout << "[INFO] Loaded module ABI contracts: " << modulePaths.size() << std::endl;

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
        REQUIRE(status.message.find("未知实现注册名") != std::string::npos,
                "Unknown registry message must be Chinese");
    }

    // 2. Null registry -> error
    {
        visionrt::IToolInstance *inst = api->create(nullptr, &services, &status);
        REQUIRE(inst == nullptr, "Null registry must return nullptr");
        REQUIRE(status.code == -3, "Null registry must return status code -3");
        REQUIRE(status.message.find("registry_name") != std::string::npos,
                "Null registry message must name registry_name");
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
        REQUIRE(res.status.message == "运行成功", "Adaptive binary success message must be shared");

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
        inputs.emplace("rect", roi);

        auto res = inst->execute(inputs);
        REQUIRE_EXECUTION(res.status.code == 0, "Crop tool execution must succeed");
        REQUIRE_EXECUTION(res.status.message == "运行成功", "Crop success message must be shared");

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
        REQUIRE(res.status.message == "运行成功", "Detection filter success message must be shared");

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
        inputs.emplace("merge_gap", std::int64_t(0));
        inputs.emplace("padding", std::int64_t(0));

        auto res = inst->execute(inputs);
        REQUIRE(res.status.code == 0, "Roi projection execution must succeed");
        REQUIRE(res.status.message == "运行成功", "Roi projection success message must be shared");

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
        REQUIRE(res.status.message == "运行成功", "Image preprocess success message must be shared");

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

        const auto tempPath = std::filesystem::path(u8"test_consumer_中文.png");
        const auto encodedPath = tempPath.generic_u8string();
        const std::string utf8Path(reinterpret_cast<const char *>(encodedPath.data()), encodedPath.size());
        cv::Mat     sample(24, 24, CV_8UC3, cv::Scalar(10, 20, 30));
        std::vector<unsigned char> encoded;
        REQUIRE(cv::imencode(".png", sample, encoded), "Image fixture must encode");
        {
            std::ofstream stream(tempPath, std::ios::binary);
            stream.write(reinterpret_cast<const char *>(encoded.data()), encoded.size());
            REQUIRE(stream.good(), "Unicode image fixture must be written");
        }

        visionrt::Values inputs;
        inputs.emplace("mode", visionrt::Utf8("file"));
        inputs.emplace("path", visionrt::Utf8(utf8Path));
        inputs.emplace("color", visionrt::Utf8("bgr"));
        inputs.emplace("limit", std::int64_t(0));
        inputs.emplace("max_batch_bytes", std::int64_t(0));

        auto res = inst->execute(inputs);
        std::filesystem::remove(tempPath);

        REQUIRE(res.status.code == 0, "Image read execution must succeed");
        REQUIRE(res.status.message == "运行成功", "Image read success message must be shared");
        const auto &paths = std::get<visionrt::StringArray>(res.outputs.at("paths"));
        REQUIRE(paths.buffers()[0].values()[0].ends_with(utf8Path), "Image path must preserve Unicode UTF-8");
        const auto *batch = std::get_if<visionrt::ImageBatch>(&res.outputs["images"]);
        REQUIRE(batch && batch->images().size() == 1, "Must read 1 image");
        REQUIRE(batch->images()[0].cols() == 24 && batch->images()[0].rows() == 24, "Image dims must match 24x24");

        REQUIRE(inst->release().code == 0, "Image read release must succeed");
        api->destroy(inst);
        actualExecutedTools.insert(reg);
    }

    // 10. Generated wrapper enforces declaration options and ranges before the operator runs
    {
        auto create = [&](const char *registry) {
            visionrt::IToolInstance *inst = api->create(registry, &services, &status);
            REQUIRE(inst != nullptr, "Wrapper constraint instance must be created");
            REQUIRE(inst->initialize({}).code == 0, "Wrapper constraint initialize must succeed");
            return inst;
        };

        auto *read = create("visionrt.image.read");
        visionrt::Values badMode;
        badMode.emplace("mode", visionrt::Utf8("invalid"));
        badMode.emplace("path", visionrt::Utf8("unused.png"));
        badMode.emplace("color", visionrt::Utf8("bgr"));
        badMode.emplace("limit", std::int64_t(0));
        badMode.emplace("max_batch_bytes", std::int64_t(0));
        auto modeResult = read->execute(badMode);
        REQUIRE(modeResult.status.code == -2, "Invalid option must be rejected by the generated wrapper");
        REQUIRE(modeResult.status.message.find("mode") != std::string::npos,
                "Option error must name the port");
        visionrt::Values badLimit = badMode;
        badLimit.insert_or_assign("mode", visionrt::Utf8("file"));
        badLimit.insert_or_assign("limit", std::int64_t(-1));
        auto limitResult = read->execute(badLimit);
        REQUIRE(limitResult.status.code == -2, "Range violation must be rejected by the generated wrapper");
        REQUIRE(limitResult.status.message.find("limit") != std::string::npos,
                "Range error must name the port");
        read->release();
        api->destroy(read);

        auto *preprocess = create("visionrt.image.preprocess");
        visionrt::Values badThreshold;
        badThreshold.emplace("images", visionrt::ImageBatch({visionrt::OwnedImage::copyOf(cv::Mat(8, 8, CV_8UC1, cv::Scalar(1)))}));
        badThreshold.emplace("operation", visionrt::Utf8("threshold"));
        badThreshold.emplace("threshold", std::int64_t(300));
        auto thresholdResult = preprocess->execute(badThreshold);
        REQUIRE(thresholdResult.status.code == -2, "Out-of-range threshold must be rejected by the generated wrapper");
        REQUIRE(thresholdResult.status.message.find("threshold") != std::string::npos,
                "Threshold range error must name the port");
        preprocess->release();
        api->destroy(preprocess);

        auto *projection = create("visionrt.roi.projection");
        visionrt::Values badProjection;
        badProjection.emplace("images", visionrt::ImageBatch({visionrt::OwnedImage::copyOf(cv::Mat(8, 8, CV_8UC1, cv::Scalar(1)))}));
        badProjection.emplace("axis", visionrt::Utf8("x"));
        badProjection.emplace("polarity", visionrt::Utf8("bright"));
        badProjection.emplace("threshold", std::int64_t(128));
        badProjection.emplace("min_pixels", std::int64_t(0));
        badProjection.emplace("min_extent", std::int64_t(1));
        badProjection.emplace("merge_gap", std::int64_t(0));
        badProjection.emplace("padding", std::int64_t(0));
        auto projectionResult = projection->execute(badProjection);
        REQUIRE(projectionResult.status.code == -2,
                "Out-of-range min_pixels must be rejected by the generated wrapper");
        projection->release();
        api->destroy(projection);

        auto *adaptive = create("AdaptiveBinaryTool");
        visionrt::Values evenKernel;
        evenKernel.emplace("images", visionrt::ImageBatch({visionrt::OwnedImage::copyOf(cv::Mat(8, 8, CV_8UC1, cv::Scalar(1)))}));
        evenKernel.emplace("block_size", std::int64_t(4));
        evenKernel.emplace("constant_offset", 2.0);
        auto evenResult = adaptive->execute(evenKernel);
        REQUIRE(evenResult.status.code == 1, "Operator-owned odd-kernel rule must still run after wrapper validation");
        REQUIRE(evenResult.status.message.find("奇数") != std::string::npos,
                "Operator-owned math error must be Chinese");
        adaptive->release();
        api->destroy(adaptive);
    }

    // Verify tool coverage: expected == actual
    std::cout << "[INFO] Installation consumer summary:" << std::endl;
    std::cout << "  DLL path: " << std::filesystem::absolute(dllPath).string() << std::endl;
    std::cout << "  Expected tools count: " << expectedTools.size() << std::endl;
    std::cout << "  Actual executed tools count: " << actualExecutedTools.size() << std::endl;
    REQUIRE(actualExecutedTools == expectedTools, "All expected tools must be tested and succeed without skipping");

    freeSharedLib(handle);
    std::cout << "[PASS] test_installation_consumer passed successfully." << std::endl;
    return 0;
}
