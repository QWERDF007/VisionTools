#include <visionrt/sdk/Tool.h>
#include <visionrt/sdk/Module.h>
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

void testPostprocessIoU() {
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

void testPostprocessGeometry() {
    std::cout << "[TEST] Running testPostprocessGeometry..." << std::endl;
    LetterboxGeometry geo{
        1920, 1080, // original
        640, 640,   // network input
        640, 360,   // resized
        0, 140,     // padLeft, padTop
        1.0f / 3.0f // scale
    };

    Box netBox{0, 140, 640, 500};
    Box srcBox = toSource(netBox, geo);
    REQUIRE(std::abs(srcBox.x1 - 0.0) < 1e-3, "src x1 must be 0");
    REQUIRE(std::abs(srcBox.y1 - 0.0) < 1e-3, "src y1 must be 0");
    REQUIRE(std::abs(srcBox.x2 - 1920.0) < 1e-3, "src x2 must be 1920");
    REQUIRE(std::abs(srcBox.y2 - 1080.0) < 1e-3, "src y2 must be 1080");
    std::cout << "[PASS] testPostprocessGeometry passed." << std::endl;
}

int main() {
    std::cout << "[TEST] Running test_infer_contract..." << std::endl;
    try {
        testPostprocessIoU();
        testPostprocessGeometry();
        std::cout << "\n[ALL TESTS PASSED] All infer contract tests passed successfully.\n" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Unhandled exception: " << e.what() << std::endl;
        return 1;
    }
}
