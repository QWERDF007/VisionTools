#include <visionrt/sdk/Tool.h>
#include <visionrt/sdk/Module.h>
#include "src/infer/YoloModel.h"
#include "src/infer/YoloPostprocess.h"

#include <iostream>
#include <cassert>
#include <cmath>

#define REQUIRE(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed: " << #cond << " (" << msg << ") at line " << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while(0)

using namespace visionrt::infer;

void testWhiteboxGeometryProbe() {
    std::cout << "[TEST] Running testWhiteboxGeometryProbe..." << std::endl;
    GeometryProbe pixelProbe = probePreprocessing();
    std::cout << "  maximumPixelError: " << pixelProbe.maximumPixelError << std::endl;
    std::cout << "  paddingPixelError: " << pixelProbe.paddingPixelError << std::endl;
    REQUIRE(pixelProbe.maximumPixelError <= 1.0, "maximumPixelError must be <= 1.0");
    REQUIRE(pixelProbe.paddingPixelError < 0.001, "paddingPixelError must be < 0.001");
    std::cout << "[PASS] testWhiteboxGeometryProbe passed." << std::endl;
}

void testWhiteboxLetterboxRoundtrip() {
    std::cout << "[TEST] Running testWhiteboxLetterboxRoundtrip..." << std::endl;
    LetterboxGeometry g{1920, 1080, 640, 640, 640, 360, 0, 140, 1.0f / 3.0f};
    Box netBox{0, 140, 640, 500};
    Box src = toSource(netBox, g);
    REQUIRE(std::abs(src.x1 - 0.0) < 1e-3, "src x1 match");
    REQUIRE(std::abs(src.y1 - 0.0) < 1e-3, "src y1 match");
    REQUIRE(std::abs(src.x2 - 1920.0) < 1e-3, "src x2 match");
    REQUIRE(std::abs(src.y2 - 1080.0) < 1e-3, "src y2 match");
    std::cout << "[PASS] testWhiteboxLetterboxRoundtrip passed." << std::endl;
}

void testWhiteboxYoloTrace() {
    std::cout << "[TEST] Running testWhiteboxYoloTrace..." << std::endl;
    const visionrt::ToolServices services{visionrt::PathResolver(std::filesystem::current_path())};
    YoloTool tool(services);

    // 1. Verify tensorDescriptions interface
    auto desc = tool.tensorDescriptions();
    std::cout << "  Uninitialized tensor descriptions count: " << desc.size() << std::endl;

    // 2. Verify executeWithTrace interface
    std::vector<ImageTrace> traces;
    visionrt::Values inputs;
    inputs.emplace("images", visionrt::ImageBatch{});
    inputs.emplace("conf_threshold", 0.25);
    inputs.emplace("nms_threshold", 0.45);
    auto res = tool.executeWithTrace(inputs, traces);
    REQUIRE(res.status.code != 0 || traces.empty(), "Uninitialized tool execution must not succeed with traces");
    std::cout << "[PASS] testWhiteboxYoloTrace passed." << std::endl;
}

int main() {
    std::cout << "[TEST] Running test_infer_whitebox_trace (isolated whitebox test)..." << std::endl;
    try {
        testWhiteboxGeometryProbe();
        testWhiteboxLetterboxRoundtrip();
        testWhiteboxYoloTrace();
        std::cout << "[ALL TESTS PASSED] test_infer_whitebox_trace passed successfully." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Whitebox trace test failed: " << e.what() << std::endl;
        return 1;
    }
}
