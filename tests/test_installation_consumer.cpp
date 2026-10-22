#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <visionrt/contracts/Module.h>
#include "TestRegistry.h"
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
        std::cerr << "Usage: test_installation_consumer <path-to-module-dll>" << std::endl;
        return 1;
    }
    std::filesystem::path target(wargv[1]);
    LocalFree(wargv);
#else
    if (argc < 2)
    {
        std::cerr << "[ERROR] Missing required DLL or package path argument." << std::endl;
        std::cerr << "Usage: test_installation_consumer <path-to-module-dll>" << std::endl;
        return 1;
    }
    std::filesystem::path target(argv[1]);
#endif
    if (!std::filesystem::exists(target))
    {
        std::cerr << "[ERROR] Target path does not exist: " << target.string() << std::endl;
        return 1;
    }

    const auto dllPath = std::filesystem::absolute(target);
    REQUIRE_STRUCTURE(std::filesystem::is_regular_file(dllPath), "Expected a module library file");
#ifdef _WIN32
    SetDllDirectoryW(dllPath.parent_path().wstring().c_str());
#endif
    const auto handle = loadSharedLib(dllPath);
    REQUIRE_LOADING(handle != nullptr, "Module library must load");
    using ModuleApiFn = const visionrt::ModuleApi *(VT_CALL *)() noexcept;
    const auto apiFn = reinterpret_cast<ModuleApiFn>(getProcAddr(handle, "visiontoolsModuleApi"));
    REQUIRE_ABI(apiFn != nullptr, "Module entry must be exported");
    const auto *api = apiFn();
    REQUIRE_ABI(api && api->abi_major == visionrt::MODULE_ABI_MAJOR, "ABI major must match");
    REQUIRE_ABI(api->struct_bytes >= sizeof(visionrt::ModuleApi) && api->register_tools,
                "ABI3 registration callback must be present");
    TestRegistry registry;
    REQUIRE_ABI(api->register_tools(registry).code == 0, "Module registration must succeed");
    const std::set<std::string> expectedTools{
        "visiontools.preprocess.adaptive_binary", "visionrt.image.crop", "visionrt.detection.filter",
        "visionrt.roi.projection", "visionrt.image.preprocess", "visionrt.image.read"};
    REQUIRE_ABI(registry.factories.size() == expectedTools.size(), "All six CPU factories must be present");
    for (const auto &id : expectedTools)
        REQUIRE_ABI(registry.factories.contains(id), "CPU factory must use its stable ID");
    visionrt::ToolServices services{visionrt::PathResolver(std::filesystem::current_path())};
    visionrt::ToolStatus status;
    std::set<std::string> actualExecutedTools;
    // 4. Test tool: adaptive_binary
    visionrt::OwnedImage snapshotImage;
    {
        const char *reg = "visiontools.preprocess.adaptive_binary";
        visionrt::IToolInstance *inst = registry.factories.at(reg).create(&services, &status);
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
        auto badInputs = inputs;
        badInputs.insert_or_assign("block_size", visionrt::Utf8("not_an_int"));
        auto badRes = inst->execute(badInputs);
        REQUIRE(badRes.status.code == 1 && badRes.status.message.find("block_size") != std::string::npos,
                "Invalid block_size must return the operator-owned business error");

        REQUIRE(inst->release().code == 0, "Release 1 must succeed");
        REQUIRE(inst->release().code == 0, "Release 2 must succeed (idempotent)");
        registry.factories.at(reg).destroy(inst);
        actualExecutedTools.insert(reg);
    }
    // Snapshot persistence after release and destroy
    REQUIRE(snapshotImage.rows() == 32, "Snapshot image must remain valid after instance destroy");
    snapshotImage = visionrt::OwnedImage();

    // 5. Test tool: crop
    {
        const char *reg = "visionrt.image.crop";
        visionrt::IToolInstance *inst = registry.factories.at(reg).create(&services, &status);
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
        registry.factories.at(reg).destroy(inst);
        actualExecutedTools.insert(reg);
    }

    // 6. Test tool: detection_filter
    {
        const char *reg = "visionrt.detection.filter";
        visionrt::IToolInstance *inst = registry.factories.at(reg).create(&services, &status);
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
        auto badInputs = inputs;
        badInputs.insert_or_assign("min_score", visionrt::Utf8("not_a_float"));
        auto badRes = inst->execute(badInputs);
        REQUIRE(badRes.status.code == 1 && badRes.status.message.find("min_score") != std::string::npos,
                "Invalid min_score must return the operator-owned business error");

        REQUIRE(inst->release().code == 0, "Filter release must succeed");
        registry.factories.at(reg).destroy(inst);
        actualExecutedTools.insert(reg);
    }

    // 7. Test tool: roi_projection
    {
        const char *reg = "visionrt.roi.projection";
        visionrt::IToolInstance *inst = registry.factories.at(reg).create(&services, &status);
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
        registry.factories.at(reg).destroy(inst);
        actualExecutedTools.insert(reg);
    }

    // 8. Test tool: image.preprocess
    {
        const char *reg = "visionrt.image.preprocess";
        visionrt::IToolInstance *inst = registry.factories.at(reg).create(&services, &status);
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
        registry.factories.at(reg).destroy(inst);
        actualExecutedTools.insert(reg);
    }

    // 9. Test tool: image.read
    {
        const char *reg = "visionrt.image.read";
        visionrt::IToolInstance *inst = registry.factories.at(reg).create(&services, &status);
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
        registry.factories.at(reg).destroy(inst);
        actualExecutedTools.insert(reg);
    }

    // Verify tool coverage: expected == actual
    std::cout << "[INFO] Installation consumer summary:" << std::endl;
    std::cout << "  DLL path: " << std::filesystem::absolute(dllPath).string() << std::endl;
    std::cout << "  Expected tools count: " << expectedTools.size() << std::endl;
    std::cout << "  Actual executed tools count: " << actualExecutedTools.size() << std::endl;
    REQUIRE(actualExecutedTools == expectedTools, "All expected tools must be tested and succeed without skipping");

    snapshotImage = {};
    freeSharedLib(handle);
    std::cout << "[PASS] test_installation_consumer passed successfully." << std::endl;
    return 0;
}
