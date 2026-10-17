#include "DetectionFilterTool.h"
#include "PortConstants.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace visiontools::preprocess {
namespace {

namespace p = visiontools::ports::visionrt_detection_filter;

template<class T>
visionrt::NumericBuffer makeBuffer(visionrt::DType dtype, std::vector<std::int64_t> shape, const std::vector<T>& values) {
    return visionrt::NumericBuffer::copy(dtype, std::move(shape), std::as_bytes(std::span(values)));
}

} // namespace

DetectionFilterTool::DetectionFilterTool(const visionrt::ToolServices& services)
    : services_(services) {}

visionrt::ToolStatus DetectionFilterTool::initialize(const visionrt::Values& parameters) {
    if (!parameters.empty()) {
        return {1, "DetectionFilterTool accepts no init parameters"};
    }
    return {};
}

visionrt::ToolResult DetectionFilterTool::execute(const visionrt::Values& inputs) {
    try {
        auto getReq = [&](const std::string& key) -> const visionrt::PortValue& {
            auto it = inputs.find(key);
            if (it == inputs.end() || visionrt::isMissing(it->second)) {
                throw visionrt::ContractError("Missing input: " + key);
            }
            return it->second;
        };

        const auto* images = std::get_if<visionrt::ImageBatch>(&getReq(p::in::IMAGES));
        const auto* rois = std::get_if<visionrt::RoiBatch>(&getReq(p::in::ROIS));
        const auto* scores = std::get_if<visionrt::NumericArray>(&getReq(p::in::SCORES));
        const auto* classes = std::get_if<visionrt::NumericArray>(&getReq(p::in::CLASSES));
        const auto* names = std::get_if<visionrt::StringArray>(&getReq(p::in::CLASS_NAMES));
        const auto* allowedArray = std::get_if<visionrt::NumericArray>(&getReq(p::in::ALLOWED_CLASSES));

        if (!images || !rois || !scores || !classes || !names || !allowedArray) {
            throw visionrt::ContractError("Invalid input types for detection filter");
        }

        const auto& scoreVal = getReq(p::in::MIN_SCORE);
        const auto* dScore = std::get_if<double>(&scoreVal);
        if (!dScore) {
            throw visionrt::ContractError("Input port 'min_score' must be Float");
        }
        const double minimumScore = *dScore;
        if (!std::isfinite(minimumScore) || minimumScore < 0.0 || minimumScore > 1.0) {
            throw visionrt::ContractError("min_score out of range [0.0, 1.0]");
        }

        const auto& areaVal = getReq(p::in::MIN_AREA);
        const auto* dArea = std::get_if<double>(&areaVal);
        if (!dArea) {
            throw visionrt::ContractError("Input port 'min_area' must be Float");
        }
        const double minimumArea = *dArea;
        if (!std::isfinite(minimumArea) || minimumArea < 0.0) {
            throw visionrt::ContractError("min_area must be >= 0.0");
        }

        std::span<const std::int64_t> allowed;
        if (!allowedArray->buffers().empty()) {
            allowed = allowedArray->buffers()[0].values<std::int64_t>();
        }

        const auto batchSize = images->images().size();
        if (rois->images().size() != batchSize || scores->buffers().size() != batchSize ||
            classes->buffers().size() != batchSize || names->buffers().size() != batchSize) {
            throw visionrt::ContractError("Detection batch lengths differ");
        }

        std::vector<std::vector<visionrt::Roi>> filteredRois;
        std::vector<visionrt::NumericBuffer> filteredScores, filteredClasses;
        std::vector<visionrt::StringBuffer> filteredNames;
        std::vector<std::int64_t> counts;

        filteredRois.reserve(batchSize);
        filteredScores.reserve(batchSize);
        filteredClasses.reserve(batchSize);
        filteredNames.reserve(batchSize);
        counts.reserve(batchSize);

        for (std::size_t b = 0; b < batchSize; ++b) {
            const auto& imageRois = rois->images()[b];
            const auto imageScores = scores->buffers()[b].values<float>();
            const auto imageClasses = classes->buffers()[b].values<std::int64_t>();
            const auto imageNames = names->buffers()[b].values();

            if (imageScores.size() != imageRois.size() || imageClasses.size() != imageRois.size() ||
                imageNames.size() != imageRois.size()) {
                throw visionrt::ContractError("Detection field lengths differ at b=" + std::to_string(b));
            }

            std::vector<visionrt::Roi> keptRois;
            std::vector<float> keptScores;
            std::vector<std::int64_t> keptClasses;
            std::vector<std::string> keptNames;

            keptRois.reserve(imageRois.size());
            keptScores.reserve(imageRois.size());
            keptClasses.reserve(imageRois.size());
            keptNames.reserve(imageRois.size());

            for (std::size_t n = 0; n < imageRois.size(); ++n) {
                if (!std::isfinite(imageScores[n]) || imageClasses[n] < 0) {
                    throw visionrt::ContractError("Invalid detection score/class at b=" + std::to_string(b) +
                                                  " n=" + std::to_string(n));
                }
                if (imageRois[n].kind() != visionrt::RoiKind::Rectangle) {
                    throw visionrt::ContractError("Detection ROI must be a rectangle at b=" + std::to_string(b) +
                                                  " n=" + std::to_string(n));
                }
                const auto encoded = imageRois[n].encoded();
                const double area = static_cast<double>(encoded[3]) * encoded[4];

                const bool classAllowed = allowed.empty() ||
                    (std::find(allowed.begin(), allowed.end(), imageClasses[n]) != allowed.end());

                if (imageScores[n] >= minimumScore && area >= minimumArea && classAllowed) {
                    keptRois.push_back(imageRois[n]);
                    keptScores.push_back(imageScores[n]);
                    keptClasses.push_back(imageClasses[n]);
                    keptNames.push_back(std::string(imageNames[n]));
                }
            }

            const auto count = static_cast<std::int64_t>(keptRois.size());
            counts.push_back(count);
            filteredRois.push_back(std::move(keptRois));
            filteredScores.push_back(makeBuffer(visionrt::DType::Float32, {count}, keptScores));
            filteredClasses.push_back(makeBuffer(visionrt::DType::Int64, {count}, keptClasses));
            filteredNames.emplace_back(std::vector<std::int64_t>{count}, std::move(keptNames));
        }

        visionrt::Values outputs;
        outputs.emplace(p::out::IMAGES, *images);
        outputs.emplace(p::out::ROIS, visionrt::RoiBatch(std::move(filteredRois)));
        outputs.emplace(p::out::SCORES, visionrt::NumericArray(visionrt::DType::Float32, 2, visionrt::Layout::BatchRagged, std::move(filteredScores)));
        outputs.emplace(p::out::CLASSES, visionrt::NumericArray(visionrt::DType::Int64, 2, visionrt::Layout::BatchRagged, std::move(filteredClasses)));
        outputs.emplace(p::out::CLASS_NAMES, visionrt::StringArray(2, visionrt::Layout::BatchRagged, std::move(filteredNames)));
        outputs.emplace(p::out::COUNTS, visionrt::NumericArray(visionrt::DType::Int64, 1, visionrt::Layout::Dense,
            {makeBuffer(visionrt::DType::Int64, {static_cast<std::int64_t>(batchSize)}, counts)}));

        return {{}, std::move(outputs)};
    } catch (const std::exception& e) {
        visionrt::Values outputs;
        outputs.emplace(p::out::IMAGES, visionrt::Missing{});
        outputs.emplace(p::out::ROIS, visionrt::Missing{});
        outputs.emplace(p::out::SCORES, visionrt::Missing{});
        outputs.emplace(p::out::CLASSES, visionrt::Missing{});
        outputs.emplace(p::out::CLASS_NAMES, visionrt::Missing{});
        outputs.emplace(p::out::COUNTS, visionrt::Missing{});
        return {{1, e.what()}, std::move(outputs)};
    }
}

visionrt::ToolStatus DetectionFilterTool::release() {
    return {};
}

} // namespace visiontools::preprocess
