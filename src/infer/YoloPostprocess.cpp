#include "YoloPostprocess.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace visionrt::infer {
namespace {
float sigmoid(float value) noexcept {
    if (value >= 0) return 1.0F / (1.0F + std::exp(-value));
    const auto exponential = std::exp(value);
    return exponential / (1.0F + exponential);
}
void validateGeometry(const LetterboxGeometry& geometry) {
    if (geometry.originalWidth <= 0 || geometry.originalHeight <= 0 ||
        geometry.inputWidth <= 0 || geometry.inputHeight <= 0 ||
        geometry.resizedWidth <= 0 || geometry.resizedHeight <= 0 ||
        geometry.padLeft < 0 || geometry.padTop < 0 ||
        geometry.resizedWidth > geometry.inputWidth - geometry.padLeft ||
        geometry.resizedHeight > geometry.inputHeight - geometry.padTop)
        throw std::invalid_argument("Invalid actual letterbox geometry");
}
Box inverse(const Box& box, const LetterboxGeometry& geometry) {
    const double inverseX = double(geometry.originalWidth) / geometry.resizedWidth;
    const double inverseY = double(geometry.originalHeight) / geometry.resizedHeight;
    return {std::clamp((box.x1 - geometry.padLeft) * inverseX, 0.0, double(geometry.originalWidth)),
            std::clamp((box.y1 - geometry.padTop) * inverseY, 0.0, double(geometry.originalHeight)),
            std::clamp((box.x2 - geometry.padLeft) * inverseX, 0.0, double(geometry.originalWidth)),
            std::clamp((box.y2 - geometry.padTop) * inverseY, 0.0, double(geometry.originalHeight))};
}
}

void validateParameters(const DetectionParameters& parameters) {
    if (!std::isfinite(parameters.confidence) || parameters.confidence < 0 || parameters.confidence > 1 ||
        !std::isfinite(parameters.nms) || parameters.nms < 0 || parameters.nms > 1 || parameters.maximum < 1)
        throw std::invalid_argument("conf_threshold/nms_threshold must be finite in [0,1]; max_detections must be >= 1");
}

double intersectionOverUnion(const Box& a, const Box& b) noexcept {
    const auto intersection = std::max(0.0, std::min(a.x2, b.x2) - std::max(a.x1, b.x1)) *
                              std::max(0.0, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    const auto total = std::max(0.0, a.x2 - a.x1) * std::max(0.0, a.y2 - a.y1) +
                       std::max(0.0, b.x2 - b.x1) * std::max(0.0, b.y2 - b.y1) - intersection;
    return total > 0 ? intersection / total : 0;
}

Box toSource(const Box& network, const LetterboxGeometry& geometry) {
    validateGeometry(geometry);
    return inverse(network, geometry);
}

std::vector<Detection> postprocess(std::span<const HeadView> heads,
    const LetterboxGeometry& geometry, const DetectionParameters& parameters,
    std::vector<Detection>* candidates) {
    validateParameters(parameters);
    validateGeometry(geometry);
    if (candidates) candidates->clear();
    std::vector<Detection> selected;
    std::size_t offset = 0;
    for (const auto& head : heads) {
        if (head.stride <= 0 || head.gridWidth <= 0 || head.gridHeight <= 0 || head.numClasses <= 0)
            throw std::invalid_argument("Invalid native YOLO head dimensions");
        const auto grid = static_cast<std::size_t>(head.gridWidth) * head.gridHeight;
        const auto channels = static_cast<std::size_t>(head.numClasses) + 4;
        if (grid > std::numeric_limits<std::size_t>::max() / channels || head.values.size() != grid * channels)
            throw std::invalid_argument("Native YOLO head buffer size mismatch");
        for (std::size_t index = 0; index < grid; ++index) {
            std::int64_t bestClass = 0;
            float bestScore = -1;
            for (int category = 0; category < head.numClasses; ++category) {
                const auto logit = head.values[(static_cast<std::size_t>(category) + 4) * grid + index];
                if (!std::isfinite(logit)) throw std::invalid_argument("Non-finite native YOLO class logit");
                const auto score = sigmoid(logit);
                if (score > bestScore) { bestScore = score; bestClass = category; }
            }
            double distances[4];
            for (std::size_t side = 0; side < 4; ++side) {
                distances[side] = head.values[side * grid + index];
                if (!std::isfinite(distances[side]) || distances[side] < 0)
                    throw std::invalid_argument("Invalid integrated DFL distance");
            }
            const double centerX = double(index % head.gridWidth) + 0.5;
            const double centerY = double(index / head.gridWidth) + 0.5;
            Detection detection{{(centerX - distances[0]) * head.stride,
                                  (centerY - distances[1]) * head.stride,
                                  (centerX + distances[2]) * head.stride,
                                  (centerY + distances[3]) * head.stride},
                                  bestScore, bestClass, offset + index};
            if (candidates) candidates->push_back(detection);
            if (bestScore >= parameters.confidence && detection.box.x2 > detection.box.x1 &&
                detection.box.y2 > detection.box.y1) selected.push_back(detection);
        }
        offset += grid;
    }
    std::sort(selected.begin(), selected.end(), [](const Detection& a, const Detection& b) {
        return a.score != b.score ? a.score > b.score : a.candidateIndex < b.candidateIndex;
    });
    std::vector<Detection> kept;
    kept.reserve(std::min(selected.size(), static_cast<std::size_t>(parameters.maximum)));
    for (const auto& candidate : selected) {
        bool suppressed = false;
        for (const auto& previous : kept) {
            if (previous.classId == candidate.classId &&
                intersectionOverUnion(previous.box, candidate.box) > parameters.nms) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(candidate);
            if (kept.size() >= static_cast<std::size_t>(parameters.maximum)) break;
        }
    }
    for (auto& detection : kept) detection.box = inverse(detection.box, geometry);
    std::erase_if(kept, [](const Detection& detection) {
        return detection.box.x2 <= detection.box.x1 || detection.box.y2 <= detection.box.y1;
    });
    return kept;
}

} // namespace visionrt::infer
