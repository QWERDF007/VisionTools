#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace visionrt::infer {

// Continuous pixel edges, not pixel-center indices (right/bottom may equal W/H).
struct Box { double x1, y1, x2, y2; };
struct LetterboxGeometry {
    int originalWidth, originalHeight;
    int inputWidth, inputHeight;
    int resizedWidth, resizedHeight;
    int padLeft, padTop;
    float sdkScale;
};
struct HeadView {
    std::span<const float> values;
    int stride, gridWidth, gridHeight, numClasses;
};
struct Detection {
    Box box;
    float score;
    std::int64_t classId;
    std::size_t candidateIndex;
};
struct DetectionParameters {
    double confidence = 0.25;
    double nms = 0.45;
    std::int64_t maximum = 100;
};

void validateParameters(const DetectionParameters& parameters);
double intersectionOverUnion(const Box& a, const Box& b) noexcept;
Box toSource(const Box& network, const LetterboxGeometry& geometry);
// Native InferRT 0.0.3 [1,4+C,grid]: integrated DFL in grid units,
// stride 8/16/32, center +0.5, class logits. No second DFL/activation.
// Optional candidates retain network coordinates, including below-threshold
// candidates, solely for the real probe; normal execution allocates no trace.
std::vector<Detection> postprocess(std::span<const HeadView> heads,
    const LetterboxGeometry& geometry, const DetectionParameters& parameters,
    std::vector<Detection>* candidates = nullptr);

} // namespace visionrt::infer
