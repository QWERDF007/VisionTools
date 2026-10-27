#include <opencv2/core.hpp>
#include "result_view.h"
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
        "visionrt.roi.projection", "visionrt.image.preprocess", "visionrt.image.read",
        "visiontools.image.rotate", "visiontools.image.brightness"};
    REQUIRE_ABI(registry.factories.size() == expectedTools.size(), "All CPU factories must be present");
    for (const auto &id : expectedTools)
        REQUIRE_ABI(registry.factories.contains(id), "CPU factory must use its stable ID");
    visionrt::ToolServices services{visionrt::PathResolver(std::filesystem::current_path())};
    visionrt::ToolStatus status;
    std::set<std::string> actualExecutedTools;
    // 4. Test tool: adaptive_binary
    cv::Mat                snapshotImage;
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
        inputs.emplace("images", visionrt::DataRef<visionrt::Image>(visionrt::Image{(gray).clone()}));
        inputs.emplace("block_size", std::int64_t(5));
        inputs.emplace("constant_offset", 2.0);

        auto res = inst->execute(inputs);
        REQUIRE(testview::status(res).code == 0, "Execution of adaptive_binary must succeed");
        REQUIRE(testview::status(res).message == "运行成功", "Adaptive binary success message must be shared");

        const auto *outBatch = std::get_if<visionrt::DataRef<visionrt::Image>>(&res.outputs["images"]);
        REQUIRE(outBatch != nullptr && !outBatch->get().empty(), "Adaptive binary output must contain images");
        snapshotImage = outBatch->get()[0];

        // Contract error test: wrong type for block_size
        auto badInputs = inputs;
        badInputs.insert_or_assign("block_size", std::string("not_an_int"));
        auto badRes = inst->execute(badInputs);
        REQUIRE(testview::status(badRes).code == 1 && testview::status(badRes).message.find("block_size") != std::string::npos,
                "Invalid block_size must return the operator-owned business error");

        REQUIRE(inst->release().code == 0, "Release 1 must succeed");
        REQUIRE(inst->release().code == 0, "Release 2 must succeed (idempotent)");
        registry.factories.at(reg).destroy(inst);
        actualExecutedTools.insert(reg);
    }
    // Snapshot persistence after release and destroy
    REQUIRE(snapshotImage.rows == 32, "Snapshot image must remain valid after instance destroy");
    snapshotImage = cv::Mat();

    // 5. Test tool: crop
    {
        const char *reg = "visionrt.image.crop";
        visionrt::IToolInstance *inst = registry.factories.at(reg).create(&services, &status);
        REQUIRE(inst != nullptr, "Create crop must return valid instance");
        REQUIRE(inst->initialize({}).code == 0, "Initialize crop must succeed");

        cv::Mat              colorImg(64, 64, CV_8UC3, cv::Scalar(10, 20, 30));
        cv::Mat              inImg = (colorImg).clone();
        visionrt::Values     inputs;
        inputs.emplace("images", visionrt::DataRef<visionrt::Image>(visionrt::Image{inImg}));
        auto roi = std::vector<double>{2, 10.0, 10.0, 20.0, 20.0, 0.0};
        inputs.emplace("rect", visionrt::DataRef<visionrt::ROI>(visionrt::ROI{{roi}}));

        auto res = inst->execute(inputs);
        REQUIRE_EXECUTION(testview::status(res).code == 0, "Crop tool execution must succeed");
        REQUIRE_EXECUTION(testview::status(res).message == "运行成功", "Crop success message must be shared");

        const auto *outBatch = std::get_if<visionrt::DataRef<visionrt::Image>>(&res.outputs["images"]);
        REQUIRE_EXECUTION(outBatch != nullptr && !outBatch->get().empty(), "Crop output must contain images");
        const auto &cropped = outBatch->get()[0];
        REQUIRE_EXECUTION(cropped.rows == 20 && cropped.cols == 20, "Cropped size must match ROI (20x20)");
        REQUIRE_EXECUTION(cropped.step[0] == inImg.step[0],
                          "Cropped image rowStride must match source rowStride for zero-copy view");
        REQUIRE_EXECUTION(cropped.datastart == inImg.datastart,
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
        auto                      roi     = std::vector<double>{2, 10.0, 10.0, 20.0, 20.0, 0.0};
        std::vector<float>        scores  = {0.85F, 0.35F};
        std::vector<std::int64_t> classes = {1, 2};
        std::vector<std::string>  names   = {"cat", "dog"};
        std::vector<std::int64_t> allowed = {1, 2};

        visionrt::Values inputs;
        inputs.emplace("images", visionrt::DataRef<visionrt::Image>(visionrt::Image{(img).clone()}));
        inputs.emplace("rois", visionrt::DataRef<visionrt::ROI>(visionrt::ROI{
                                   {roi, roi}
        }));
        inputs.emplace("scores", visionrt::Array(std::vector<std::vector<float>>{scores}));
        inputs.emplace("classes", visionrt::Array(std::vector<std::vector<std::int64_t>>{classes}));
        inputs.emplace("class_names", visionrt::Array(std::vector<std::vector<std::string>>{names}));
        inputs.emplace("allowed_classes", visionrt::Array(std::move(allowed)));
        inputs.emplace("min_score", 0.5);
        inputs.emplace("min_area", 0.0);

        auto res = inst->execute(inputs);
        REQUIRE(testview::status(res).code == 0, "Detection filter execution must succeed");
        REQUIRE(testview::status(res).message == "运行成功", "Detection filter success message must be shared");

        const auto *outRois = std::get_if<visionrt::DataRef<visionrt::ROI>>(&res.outputs["rois"]);
        REQUIRE(outRois != nullptr && outRois->get()[0].size() == 1, "Only score >= 0.5 passes");

        // Contract error test
        auto badInputs = inputs;
        badInputs.insert_or_assign("min_score", std::string("not_a_float"));
        auto badRes = inst->execute(badInputs);
        REQUIRE(testview::status(badRes).code == 1 && testview::status(badRes).message.find("min_score") != std::string::npos,
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
        inputs.emplace("images", visionrt::DataRef<visionrt::Image>(visionrt::Image{(img).clone()}));
        inputs.emplace("axis", std::string("x"));
        inputs.emplace("polarity", std::string("bright"));
        inputs.emplace("threshold", std::int64_t(128));
        inputs.emplace("min_pixels", std::int64_t(10));
        inputs.emplace("min_extent", std::int64_t(10));
        inputs.emplace("merge_gap", std::int64_t(0));
        inputs.emplace("padding", std::int64_t(0));

        auto res = inst->execute(inputs);
        REQUIRE(testview::status(res).code == 0, "Roi projection execution must succeed");
        REQUIRE(testview::status(res).message == "运行成功", "Roi projection success message must be shared");

        const auto *rois = std::get_if<visionrt::DataRef<visionrt::ROI>>(&res.outputs["rois"]);
        REQUIRE(rois != nullptr && rois->get().size() == 1 && rois->get()[0].size() == 1, "Expected 1 projected ROI");

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
        inputs.emplace("images", visionrt::DataRef<visionrt::Image>(visionrt::Image{(img).clone()}));
        inputs.emplace("operation", std::string("threshold"));
        inputs.emplace("threshold", std::int64_t(100));

        auto res = inst->execute(inputs);
        REQUIRE(testview::status(res).code == 0, "Image preprocess execution must succeed");
        REQUIRE(testview::status(res).message == "运行成功", "Image preprocess success message must be shared");

        const auto *out = std::get_if<visionrt::DataRef<visionrt::Image>>(&res.outputs["images"]);
        REQUIRE(out != nullptr && !out->get().empty(), "Output must have ImageBatch");
        REQUIRE(out->get()[0].at<unsigned char>(0, 0) == 255, "150 >= 100 -> 255");

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
        inputs.emplace("path", std::string(utf8Path));
        inputs.emplace("recursive", false);
        inputs.emplace("traversal", std::string("循环"));
        inputs.emplace("order", std::string("顺序（下一张）"));
        inputs.emplace("count", std::int64_t(1));
        inputs.emplace("reload", false);

        auto res = inst->execute(inputs);
        std::filesystem::remove(tempPath);

        REQUIRE(testview::status(res).code == 0, "Image read execution must succeed");
        REQUIRE(testview::status(res).message == "运行成功", "Image read success message must be shared");
        const auto &paths = std::get<visionrt::Array>(res.outputs.at("paths")).get<std::vector<std::string>>();
        REQUIRE(paths[0].ends_with(utf8Path), "Image path must preserve Unicode UTF-8");
        const auto *batch = std::get_if<visionrt::DataRef<visionrt::Image>>(&res.outputs["images"]);
        REQUIRE(batch && batch->get().size() == 1, "Must read 1 image");
        REQUIRE(batch->get()[0].cols == 24 && batch->get()[0].rows == 24, "Image dims must match 24x24");

        REQUIRE(inst->release().code == 0, "Image read release must succeed");
        registry.factories.at(reg).destroy(inst);
        actualExecutedTools.insert(reg);
    }

    // Verify tool coverage: expected == actual
    std::cout << "[INFO] Installation consumer summary:" << std::endl;
    std::cout << "  DLL path: " << std::filesystem::absolute(dllPath).string() << std::endl;
    // 9. Test tool: rotate
    {
        const char *reg = "visiontools.image.rotate";
        visionrt::IToolInstance *inst = registry.factories.at(reg).create(&services, &status);
        REQUIRE_ABI(inst != nullptr, "Create rotate must return valid instance");
        REQUIRE_ABI(inst->initialize({}).code == 0, "Initialize rotate must succeed");
        visionrt::Values inputs;
        inputs.emplace("images", visionrt::DataRef<visionrt::Image>(visionrt::Image{cv::Mat(16, 16, CV_8UC1, cv::Scalar(7))}));
        inputs.emplace("angle_deg", 90.0);
        inputs.emplace("center", visionrt::Array(visionrt::DataRef<visionrt::NestedVector<double, 1>>(
                                      visionrt::NestedVector<double, 1>{})));
        inputs.emplace("canvas", std::string("keep"));
        inputs.emplace("interpolation", std::string("linear"));
        inputs.emplace("border_value", visionrt::Array(visionrt::DataRef<visionrt::NestedVector<double, 1>>(
                                           visionrt::NestedVector<double, 1>{0.0})));
        inputs.emplace("alignment", visionrt::Array(visionrt::DataRef<visionrt::NestedVector<double, 3>>(
                                        visionrt::NestedVector<double, 3>{})));
        auto res = inst->execute(inputs);
        REQUIRE_ABI(testview::status(res).code == 0, "Execution of rotate must succeed");
        const auto *out = std::get_if<visionrt::DataRef<visionrt::Image>>(&res.outputs["images"]);
        REQUIRE_ABI(out != nullptr && !out->get().empty(), "Rotate output must contain images");
        REQUIRE_ABI(inst->release().code == 0, "Release rotate must succeed");
        registry.factories.at(reg).destroy(inst);
        actualExecutedTools.insert(reg);
    }
    // 10. Test tool: brightness
    {
        const char *reg = "visiontools.image.brightness";
        visionrt::IToolInstance *inst = registry.factories.at(reg).create(&services, &status);
        REQUIRE_ABI(inst != nullptr, "Create brightness must return valid instance");
        REQUIRE_ABI(inst->initialize({}).code == 0, "Initialize brightness must succeed");
        visionrt::Values inputs;
        inputs.emplace("images", visionrt::DataRef<visionrt::Image>(visionrt::Image{cv::Mat(8, 8, CV_8UC1, cv::Scalar(20))}));
        inputs.emplace("gain", 1.5);
        inputs.emplace("offset", 0.0);
        inputs.emplace("rois", visionrt::DataRef<visionrt::ROI>(visionrt::ROI{}));
        inputs.emplace("mask", visionrt::Missing{});
        inputs.emplace("alignment", visionrt::Array(visionrt::DataRef<visionrt::NestedVector<double, 3>>(
                                        visionrt::NestedVector<double, 3>{})));
        inputs.emplace("emit_mask", false);
        auto res = inst->execute(inputs);
        REQUIRE_ABI(testview::status(res).code == 0, "Execution of brightness must succeed");
        const auto *out = std::get_if<visionrt::DataRef<visionrt::Image>>(&res.outputs["images"]);
        REQUIRE_ABI(out != nullptr && !out->get().empty(), "Brightness output must contain images");
        REQUIRE_ABI(inst->release().code == 0, "Release brightness must succeed");
        registry.factories.at(reg).destroy(inst);
        actualExecutedTools.insert(reg);
    }
    std::cout << "  Expected tools count: " << expectedTools.size() << std::endl;
    std::cout << "  Actual executed tools count: " << actualExecutedTools.size() << std::endl;
    REQUIRE(actualExecutedTools == expectedTools, "All expected tools must be tested and succeed without skipping");

    snapshotImage = {};
    freeSharedLib(handle);
    std::cout << "[PASS] test_installation_consumer passed successfully." << std::endl;
    return 0;
}
