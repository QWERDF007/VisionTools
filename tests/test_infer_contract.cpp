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
    partial.emplace("model_name", std::string("yolov8n"));
    partial.emplace("load_mode", std::string("engine"));
    partial.emplace("model_path", std::string("missing.engine"));
    partial.emplace("engine_output_path", std::string(""));
    partial.emplace("precision", std::string("fp16"));
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
        REQUIRE(api && api->abi_major == visionrt::MODULE_ABI_MAJOR, "Inference module must expose ABI4");
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
