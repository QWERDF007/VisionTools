// Pure CPU regression for the native YOLO postprocessor. This target links only
// YoloPostprocess.cpp, Contracts and OpenCV so that CPU-only configurations keep
// covering the padding/NMS boundary. Model, module and GPU contracts stay in
// test_infer_contract.
#include "src/infer/YoloPostprocess.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

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

namespace {
constexpr int kGrid = 16;

// Native layout [1, 4 + C, grid] with 4 columns/rows of grid cells.
struct HeadBuffer
{
    std::vector<float> values;

    explicit HeadBuffer(int classes)
        : values(static_cast<std::size_t>(4 + classes) * kGrid, 0.0F)
    {
    }

    void set(int index, float left, float top, float right, float bottom)
    {
        values[index]                   = left;
        values[kGrid + index]           = top;
        values[2 * kGrid + index]       = right;
        values[3 * kGrid + index]       = bottom;
    }

    void setClass(int index, int category, float logit)
    {
        values[(static_cast<std::size_t>(category) + 4) * kGrid + index] = logit;
    }

    HeadView view(int classes) const
    {
        return HeadView{values, 8, 4, 4, classes};
    }
};

// 32x32 network, 16 pixel tall source occupying y=[8,24].
LetterboxGeometry wideGeometry()
{
    return LetterboxGeometry{32, 16, 32, 32, 32, 16, 0, 8, 1.0F};
}

void testIntersectionOverUnion()
{
    const Box a{0, 0, 10, 10};
    REQUIRE(std::abs(intersectionOverUnion(a, a) - 1.0) < 1e-6, "identical boxes must reach IoU 1");
    REQUIRE(intersectionOverUnion(a, Box{10, 10, 20, 20}) == 0.0, "disjoint boxes must reach IoU 0");
    REQUIRE(std::abs(intersectionOverUnion(a, Box{5, 0, 15, 10}) - (1.0 / 3.0)) < 1e-6,
            "half overlap must reach IoU 1/3");
    REQUIRE(intersectionOverUnion(Box{0, 0, 0, 0}, a) == 0.0, "degenerate box must not divide by zero");
}

void testGeometryMapping()
{
    const LetterboxGeometry geometry{1920, 1080, 640, 640, 640, 360, 0, 140, 1.0F / 3.0F};
    const Box               source = toSource(Box{0, 140, 640, 500}, geometry);
    REQUIRE(std::abs(source.x1) < 1e-3 && std::abs(source.y1) < 1e-3, "source origin must map to zero");
    REQUIRE(std::abs(source.x2 - 1920) < 1e-3 && std::abs(source.y2 - 1080) < 1e-3,
            "source extent must map to the original size");

    bool rejected = false;
    try
    {
        toSource(Box{0, 0, 1, 1}, LetterboxGeometry{0, 16, 32, 32, 32, 16, 0, 8, 1.0F});
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    REQUIRE(rejected, "invalid letterbox geometry must be rejected");
}

void testPaddingSuppressionAndLimit()
{
    HeadBuffer disjoint(1);
    disjoint.set(0, 0.5F, 0.5F, 0.5F, 0.5F);   // network [0,0,8,8]: pure padding
    disjoint.set(5, 0.25F, 0.25F, 0.25F, 0.25F); // network [8,8,12,12]: inside the source
    disjoint.setClass(0, 0, 5.0F);
    disjoint.setClass(5, 0, 4.0F);
    const std::array<HeadView, 1> heads{disjoint.view(1)};

    const auto limited = postprocess(heads, wideGeometry(), {0.5, 0.5, 1});
    REQUIRE(limited.size() == 1, "valid box must survive maximum=1 when the padding candidate is filtered");
    const auto unlimited = postprocess(heads, wideGeometry(), {0.5, 0.5, 10});
    REQUIRE(unlimited.size() == 1, "padding candidate must never enter the kept list");

    // Network [0,0,8,16] overlaps the padding box by 0.5 and still covers source
    // pixels; the higher-score padding candidate must not suppress it.
    HeadBuffer overlap(1);
    overlap.set(0, 0.5F, 0.5F, 0.5F, 0.5F);
    overlap.set(4, 0.5F, 1.5F, 0.5F, 0.5F);
    overlap.setClass(0, 0, 5.0F);
    overlap.setClass(4, 0, 4.0F);
    const std::array<HeadView, 1> overwrapped{overlap.view(1)};
    const auto                    suppressed = postprocess(overwrapped, wideGeometry(), {0.5, 0.4, 10});
    REQUIRE(suppressed.size() == 1, "padding candidate must not suppress a valid detection");
    REQUIRE(suppressed.front().candidateIndex == 4, "the surviving detection must be the valid box");
}

void testFullyPaddedImageYieldsNoDetection()
{
    HeadBuffer buffer(1);
    buffer.set(0, 0.5F, 0.5F, 0.5F, 0.5F);
    buffer.setClass(0, 0, 9.0F);
    const std::array<HeadView, 1> heads{buffer.view(1)};
    const auto                    detections = postprocess(heads, wideGeometry(), {0.1, 0.5, 10});
    REQUIRE(detections.empty(), "a fully padded candidate must not produce a detection");
}

void testPartiallyOutOfBoundsKeepsValidArea()
{
    HeadBuffer buffer(1);
    // Network [2,6,14,26] maps to source y=[-2,18] clipped to [0,16]: valid area remains.
    buffer.set(0, 2.0F, 6.0F, 14.0F, 26.0F);
    buffer.setClass(0, 0, 6.0F);
    const std::array<HeadView, 1> heads{buffer.view(1)};
    const auto                    detections = postprocess(heads, wideGeometry(), {0.5, 0.5, 4});
    REQUIRE(detections.size() == 1, "a candidate covering source pixels must be kept");
}

void testScoreOrderIsStable()
{
    HeadBuffer buffer(1);
    buffer.set(5, 0.2F, 0.2F, 0.2F, 0.2F);
    buffer.set(6, 0.2F, 0.2F, 0.2F, 0.2F);
    buffer.setClass(5, 0, 6.0F);
    buffer.setClass(6, 0, 6.0F);
    const std::array<HeadView, 1> heads{buffer.view(1)};
    const auto                    detections = postprocess(heads, wideGeometry(), {0.5, 0.5, 4});
    REQUIRE(detections.size() == 2, "non-overlapping equal scores must both be kept");
    REQUIRE(detections[0].candidateIndex == 5 && detections[1].candidateIndex == 6,
            "equal scores must keep the lower candidate index first");
}

void testDifferentClassesAreNotSuppressed()
{
    HeadBuffer buffer(2);
    buffer.set(5, 0.2F, 0.2F, 0.4F, 0.4F);
    buffer.set(6, 0.2F, 0.2F, 0.4F, 0.4F);
    buffer.setClass(5, 0, 6.0F);
    buffer.setClass(5, 1, -6.0F);
    buffer.setClass(6, 0, -6.0F);
    buffer.setClass(6, 1, 6.0F);
    const std::array<HeadView, 1> heads{buffer.view(2)};
    const auto                    detections = postprocess(heads, wideGeometry(), {0.5, 0.1, 4});
    REQUIRE(detections.size() == 2, "overlapping boxes of different classes must not suppress each other");
    REQUIRE(detections[0].classId != detections[1].classId, "class ids must be reported per detection");
}

void testNonFiniteInputRejected()
{
    HeadBuffer nanBuffer(1);
    nanBuffer.set(5, 0.2F, 0.2F, 0.2F, 0.2F);
    nanBuffer.setClass(5, 0, std::numeric_limits<float>::quiet_NaN());
    const std::array<HeadView, 1> nanHeads{nanBuffer.view(1)};
    bool                          rejected = false;
    try
    {
        postprocess(nanHeads, wideGeometry(), {0.5, 0.5, 4});
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    REQUIRE(rejected, "non-finite class logits must be rejected");

    HeadBuffer negative(1);
    negative.set(5, -0.2F, 0.2F, 0.2F, 0.2F);
    negative.setClass(5, 0, 6.0F);
    const std::array<HeadView, 1> negativeHeads{negative.view(1)};
    rejected = false;
    try
    {
        postprocess(negativeHeads, wideGeometry(), {0.5, 0.5, 4});
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    REQUIRE(rejected, "negative DFL distances must be rejected");
}

void testParameterValidation()
{
    for (const auto *parameters : {new DetectionParameters{-0.1, 0.5, 4}, new DetectionParameters{0.5, 1.5, 4},
                                   new DetectionParameters{0.5, 0.5, 0}})
    {
        bool rejected = false;
        try
        {
            validateParameters(*parameters);
        }
        catch (const std::invalid_argument &)
        {
            rejected = true;
        }
        delete parameters;
        REQUIRE(rejected, "out-of-range detection parameters must be rejected");
    }
    bool ok = true;
    try
    {
        validateParameters({0.25, 0.45, 1});
    }
    catch (const std::invalid_argument &)
    {
        ok = false;
    }
    REQUIRE(ok, "valid detection parameters must be accepted");
}
} // namespace

int main()
{
    testIntersectionOverUnion();
    testGeometryMapping();
    testPaddingSuppressionAndLimit();
    testFullyPaddedImageYieldsNoDetection();
    testPartiallyOutOfBoundsKeepsValidArea();
    testScoreOrderIsStable();
    testDifferentClassesAreNotSuppressed();
    testNonFiniteInputRejected();
    testParameterValidation();
    std::cout << "YOLO postprocess padding, ordering and validation contract passed" << std::endl;
    return 0;
}
