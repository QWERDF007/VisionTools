#include "src/infer/YoloModel.h"
#include "ToolResultAssertions.h"
#include "src/infer/YoloPostprocess.h"

#include <visionrt/contracts/Module.h>
#include <visionrt/contracts/Tool.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

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

void testWhiteboxGeometryProbe()
{
    std::cout << "[TEST] Running testWhiteboxGeometryProbe..." << std::endl;
    GeometryProbe pixelProbe = probePreprocessing();
    std::cout << "  maximumPixelError: " << pixelProbe.maximumPixelError << std::endl;
    std::cout << "  paddingPixelError: " << pixelProbe.paddingPixelError << std::endl;
    REQUIRE(pixelProbe.maximumPixelError <= 1.0, "maximumPixelError must be <= 1.0");
    REQUIRE(pixelProbe.paddingPixelError < 0.001, "paddingPixelError must be < 0.001");
    std::cout << "[PASS] testWhiteboxGeometryProbe passed." << std::endl;
}

void testWhiteboxLetterboxRoundtrip()
{
    std::cout << "[TEST] Running testWhiteboxLetterboxRoundtrip..." << std::endl;
    LetterboxGeometry g{1920, 1080, 640, 640, 640, 360, 0, 140};
    Box               netBox{0, 140, 640, 500};
    Box               src = toSource(netBox, g);
    REQUIRE(std::abs(src.x1 - 0.0) < 1e-3, "src x1 match");
    REQUIRE(std::abs(src.y1 - 0.0) < 1e-3, "src y1 match");
    REQUIRE(std::abs(src.x2 - 1920.0) < 1e-3, "src x2 match");
    REQUIRE(std::abs(src.y2 - 1080.0) < 1e-3, "src y2 match");
    std::cout << "[PASS] testWhiteboxLetterboxRoundtrip passed." << std::endl;
}

void testWhiteboxYoloUninitializedErrorPath()
{
    std::cout << "[TEST] Running testWhiteboxYoloUninitializedErrorPath..." << std::endl;
    YoloTool                     tool;

    // 1. Verify tensorDescriptions interface on uninitialized tool
    auto desc = tool.tensorDescriptions();
    std::cout << "  Uninitialized tensor descriptions count: " << desc.size() << std::endl;
    REQUIRE(desc.empty(), "Uninitialized tool must have empty tensor descriptions");

    // 2. Verify executeWithTrace interface fails on uninitialized tool
    std::vector<ImageTrace> traces;
    visionrt::Values        inputs;
    inputs.emplace(1, visionrt::DataRef<visionrt::Image>(visionrt::Image{}));
    inputs.emplace(2, 0.25);
    inputs.emplace(3, 0.45);
    auto res = tool.executeWithTrace(inputs, traces);
    REQUIRE(testview::status(res).code != 0, "Uninitialized tool execution must return non-zero error code");
    REQUIRE(traces.empty(), "Uninitialized tool execution must produce no traces");
    std::cout << "[PASS] testWhiteboxYoloUninitializedErrorPath passed." << std::endl;
}

void testWhiteboxYoloInitializedTrace()
{
    std::cout << "[TEST] Running testWhiteboxYoloInitializedTrace..." << std::endl;
    const char *engineEnv = std::getenv("VISIONTOOLS_TEST_YOLO_ENGINE");
    const char *wtsEnv    = std::getenv("VISIONTOOLS_TEST_YOLO_WTS");

    if (!engineEnv && !wtsEnv)
    {
        std::cout << "  [SKIP] No external YOLO engine or weights specified (set VISIONTOOLS_TEST_YOLO_ENGINE/WTS); "
                     "skipping initialized model trace (exit 77)."
                  << std::endl;
        std::exit(77);
    }

    YoloTool                     tool;

    visionrt::Values initParams;
    initParams.emplace(0, std::string("yolov8n"));
    if (engineEnv)
    {
        if (!std::filesystem::exists(engineEnv))
        {
            std::cerr << "  [ERROR] Specified VISIONTOOLS_TEST_YOLO_ENGINE does not exist: " << engineEnv << std::endl;
            std::exit(1);
        }
        initParams.emplace(1, std::string("engine"));
        initParams.emplace(2, std::string(engineEnv));
        initParams.emplace(3, std::string(""));
    }
    else if (wtsEnv)
    {
        if (!std::filesystem::exists(wtsEnv))
        {
            std::cerr << "  [ERROR] Specified VISIONTOOLS_TEST_YOLO_WTS does not exist: " << wtsEnv << std::endl;
            std::exit(1);
        }
        initParams.emplace(1, std::string("build_wts"));
        initParams.emplace(2, std::string(wtsEnv));
        initParams.emplace(3, std::string("temp_whitebox.engine"));
    }
    initParams.emplace(5, std::string("fp32"));
    initParams.emplace(4, std::string(""));
    initParams.emplace(6, static_cast<std::int64_t>(0));
    initParams.emplace(7, static_cast<std::int64_t>(640));
    initParams.emplace(8, static_cast<std::int64_t>(640));
    initParams.emplace(9, static_cast<std::int64_t>(80));

    auto status = tool.initialize(initParams);
    REQUIRE(status.code == 0, "Model initialization must succeed");

    auto desc = tool.tensorDescriptions();
    REQUIRE(!desc.empty(), "Initialized model must provide tensor descriptions");

    // Execute with trace on non-empty synthetic image
    cv::Mat                           testMat(640, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    visionrt::Image                   imgs;
    imgs.push_back(testMat.clone());
    visionrt::DataRef<visionrt::Image> batch(std::move(imgs));

    std::vector<ImageTrace> traces;
    visionrt::Values        inputs;
    inputs.emplace(1, batch);
    inputs.emplace(2, 0.25);
    inputs.emplace(3, 0.45);
    inputs.emplace(4, static_cast<std::int64_t>(300));

    auto res = tool.executeWithTrace(inputs, traces);
    REQUIRE(testview::status(res).code == 0, "Initialized executeWithTrace must succeed");
    REQUIRE(traces.size() == 1, "Must produce one trace per input image");
    REQUIRE(traces[0].geometry.inputWidth == 640 && traces[0].geometry.inputHeight == 640,
            "Trace geometry network dims match");
    std::cout << "[PASS] testWhiteboxYoloInitializedTrace passed." << std::endl;
}

int main(int argc, char **argv)
{
    // One runner, one scenario per capability: geometry and the uninitialized error path need no
    // model, while the initialized trace needs an external engine or weights and may skip alone.
    std::string scenario = "all";
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if (argument == "--case" && index + 1 < argc)
            scenario = argv[++index];
        else
        {
            std::cerr << "[ERROR] unknown argument: " << argument << std::endl;
            return 1;
        }
    }

    std::cout << "[TEST] infer trace scenario: " << scenario << std::endl;
    if (scenario != "geometry" && scenario != "uninitialized" && scenario != "initialized-trace"
        && scenario != "all")
    {
        std::cerr << "[ERROR] unknown scenario: " << scenario << std::endl;
        return 1;
    }
    try
    {
        if (scenario == "geometry" || scenario == "all")
        {
            testWhiteboxGeometryProbe();
            testWhiteboxLetterboxRoundtrip();
        }
        if (scenario == "uninitialized" || scenario == "all")
            testWhiteboxYoloUninitializedErrorPath();
        if (scenario == "initialized-trace" || scenario == "all")
            testWhiteboxYoloInitializedTrace();
        std::cout << "[ALL TESTS PASSED] infer trace scenario passed." << std::endl;
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] Infer trace test failed: " << e.what() << std::endl;
        return 1;
    }
}
