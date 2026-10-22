#include "src/infer/YoloModel.h"
#include "src/infer/YoloPostprocess.h"
#include "TestRegistry.h"

#include <visionrt/contracts/Module.h>
#include <visionrt/contracts/Tool.h>

#include <cassert>
#include <cmath>
#include <iostream>

#define REQUIRE(cond, msg)                                                                                      \
    do                                                                                                          \
    {                                                                                                           \
        if (!(cond))                                                                                            \
        {                                                                                                       \
            std::cerr << "Assertion failed: " << #cond << " (" << msg << ") at line " << __LINE__ << std::endl; \
            std::exit(1);                                                                                       \
        }                                                                                                       \
    }                                                                                                           \
    while (0)

using namespace visionrt::infer;

void testPostprocessIoU()
{
    std::cout << "[TEST] Running testPostprocessIoU..." << std::endl;
    Box a{0, 0, 10, 10};
    Box b{0, 0, 10, 10};
    REQUIRE(std::abs(intersectionOverUnion(a, b) - 1.0) < 1e-6, "Identical boxes IoU must be 1.0");

    Box c{10, 10, 20, 20};
    REQUIRE(intersectionOverUnion(a, c) == 0.0, "Disjoint boxes IoU must be 0.0");

    Box d{5, 0, 15, 10};
    // Intersection: [5, 0] to [10, 10] -> area 50
    // Total area: 100 + 100 - 50 = 150 -> IoU = 50 / 150 = 1/3
    REQUIRE(std::abs(intersectionOverUnion(a, d) - (1.0 / 3.0)) < 1e-6, "Half overlap IoU must be 1/3");
    std::cout << "[PASS] testPostprocessIoU passed." << std::endl;
}

void testPostprocessGeometry()
{
    std::cout << "[TEST] Running testPostprocessGeometry..." << std::endl;
    LetterboxGeometry geo{
        1920,       1080, // original
        640,        640,  // network input
        640,        360,  // resized
        0,          140,  // padLeft, padTop
        1.0f / 3.0f       // scale
    };

    Box netBox{0, 140, 640, 500};
    Box srcBox = toSource(netBox, geo);
    REQUIRE(std::abs(srcBox.x1 - 0.0) < 1e-3, "src x1 must be 0");
    REQUIRE(std::abs(srcBox.y1 - 0.0) < 1e-3, "src y1 must be 0");
    REQUIRE(std::abs(srcBox.x2 - 1920.0) < 1e-3, "src x2 must be 1920");
    REQUIRE(std::abs(srcBox.y2 - 1080.0) < 1e-3, "src y2 must be 1080");
    std::cout << "[PASS] testPostprocessGeometry passed." << std::endl;
}

void testYoloMissingInitParameters()
{
    std::cout << "[TEST] Running testYoloMissingInitParameters..." << std::endl;
    visionrt::ToolServices services;
    YoloTool               tool(services);

    auto empty = tool.initialize({});
    REQUIRE(empty.code == 1, "Missing init parameters must fail before any GPU work");
    REQUIRE(empty.message.find("缺少参数") != std::string::npos, "Missing init error must be Chinese");
    REQUIRE(empty.message.find("model_name") != std::string::npos, "Missing init error must name the port");

    visionrt::Values partial;
    partial.emplace("model_name", visionrt::Utf8("yolov8n"));
    partial.emplace("load_mode", visionrt::Utf8("engine"));
    partial.emplace("model_path", visionrt::Utf8("missing.engine"));
    partial.emplace("engine_output_path", visionrt::Utf8(""));
    partial.emplace("precision", visionrt::Utf8("fp16"));
    partial.emplace("gpu_index", std::int64_t(0));
    partial.emplace("input_width", std::int64_t(640));
    partial.emplace("input_height", std::int64_t(640));
    partial.emplace("num_classes", std::int64_t(80));
    auto missingLabels = tool.initialize(partial);
    REQUIRE(missingLabels.code == 1, "Missing labels_path must fail before GPU work");
    REQUIRE(missingLabels.message.find("labels_path") != std::string::npos,
            "Missing labels_path must be named");
    std::cout << "[PASS] testYoloMissingInitParameters passed." << std::endl;
}

int main()
{
    std::cout << "[TEST] Running test_infer_contract..." << std::endl;
    try
    {
        const auto *api = visiontoolsModuleApi();
        REQUIRE(api && api->abi_major == visionrt::MODULE_ABI_MAJOR, "Inference module must expose ABI3");
        REQUIRE(api->struct_bytes >= sizeof(visionrt::ModuleApi) && api->register_tools,
                "Inference registration callback must be complete");
        TestRegistry registry;
        REQUIRE(api->register_tools(registry).code == 0, "Inference registration must succeed");
        REQUIRE(registry.factories.size() == 1 && registry.factories.contains("visionrt.dl.yolo"),
                "Inference module must register the stable YOLO ID only");
        const auto &factory = registry.factories.at("visionrt.dl.yolo");
        visionrt::ToolServices services;
        visionrt::ToolStatus status;
        auto *instance = factory.create(&services, &status);
        REQUIRE(instance && status.code == 0, "Construction must not load a model");
        REQUIRE(instance->release().code == 0 && instance->release().code == 0,
                "Uninitialized inference instance must release idempotently");
        factory.destroy(instance);
        testPostprocessIoU();
        testPostprocessGeometry();
        testYoloMissingInitParameters();
        std::cout << "\n[ALL TESTS PASSED] All infer contract tests passed successfully.\n" << std::endl;
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FATAL] Unhandled exception: " << e.what() << std::endl;
        return 1;
    }
}
