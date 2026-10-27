#include "src/infer/YoloModel.h"
#include "result_view.h"
#include "src/infer/YoloPostprocess.h"

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
    LetterboxGeometry g{1920, 1080, 640, 640, 640, 360, 0, 140, 1.0f / 3.0f};
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
    const visionrt::ToolServices services{visionrt::PathResolver(std::filesystem::current_path())};
    YoloTool                     tool(services);

    // 1. Verify tensorDescriptions interface on uninitialized tool
    auto desc = tool.tensorDescriptions();
    std::cout << "  Uninitialized tensor descriptions count: " << desc.size() << std::endl;
    REQUIRE(desc.empty(), "Uninitialized tool must have empty tensor descriptions");

    // 2. Verify executeWithTrace interface fails on uninitialized tool
    std::vector<ImageTrace> traces;
    visionrt::Values        inputs;
    inputs.emplace("images", visionrt::DataRef<visionrt::Image>(visionrt::Image{}));
    inputs.emplace("conf_threshold", 0.25);
    inputs.emplace("nms_threshold", 0.45);
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

    const visionrt::ToolServices services{visionrt::PathResolver(std::filesystem::current_path())};
    YoloTool                     tool(services);

    visionrt::Values initParams;
    initParams.emplace("model_name", std::string("yolov8n"));
    if (engineEnv)
    {
        if (!std::filesystem::exists(engineEnv))
        {
            std::cerr << "  [ERROR] Specified VISIONTOOLS_TEST_YOLO_ENGINE does not exist: " << engineEnv << std::endl;
            std::exit(1);
        }
        initParams.emplace("load_mode", std::string("engine"));
        initParams.emplace("model_path", std::string(engineEnv));
        initParams.emplace("engine_output_path", std::string(""));
    }
    else if (wtsEnv)
    {
        if (!std::filesystem::exists(wtsEnv))
        {
            std::cerr << "  [ERROR] Specified VISIONTOOLS_TEST_YOLO_WTS does not exist: " << wtsEnv << std::endl;
            std::exit(1);
        }
        initParams.emplace("load_mode", std::string("build_wts"));
        initParams.emplace("model_path", std::string(wtsEnv));
        initParams.emplace("engine_output_path", std::string("temp_whitebox.engine"));
    }
    initParams.emplace("precision", std::string("fp32"));
    initParams.emplace("labels_path", std::string(""));
    initParams.emplace("gpu_index", static_cast<std::int64_t>(0));
    initParams.emplace("input_width", static_cast<std::int64_t>(640));
    initParams.emplace("input_height", static_cast<std::int64_t>(640));
    initParams.emplace("num_classes", static_cast<std::int64_t>(80));

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
    inputs.emplace("images", batch);
    inputs.emplace("conf_threshold", 0.25);
    inputs.emplace("nms_threshold", 0.45);
    inputs.emplace("max_detections", static_cast<std::int64_t>(300));

    auto res = tool.executeWithTrace(inputs, traces);
    REQUIRE(testview::status(res).code == 0, "Initialized executeWithTrace must succeed");
    REQUIRE(traces.size() == 1, "Must produce one trace per input image");
    REQUIRE(traces[0].geometry.inputWidth == 640 && traces[0].geometry.inputHeight == 640,
            "Trace geometry network dims match");
    std::cout << "[PASS] testWhiteboxYoloInitializedTrace passed." << std::endl;
}

int main()
{
    std::cout << "[TEST] Running test_infer_whitebox_trace (isolated whitebox test)..." << std::endl;
    try
    {
        testWhiteboxGeometryProbe();
        testWhiteboxLetterboxRoundtrip();
        testWhiteboxYoloUninitializedErrorPath();
        testWhiteboxYoloInitializedTrace();
        std::cout << "[ALL TESTS PASSED] test_infer_whitebox_trace passed successfully." << std::endl;
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] Whitebox trace test failed: " << e.what() << std::endl;
        return 1;
    }
}
