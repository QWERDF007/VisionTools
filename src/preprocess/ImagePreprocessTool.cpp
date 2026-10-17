#include "ImagePreprocessTool.h"

#include <opencv2/imgproc.hpp>
#include <limits>
#include <vector>

namespace visiontools::preprocess {
namespace {

void requireU8(const visionrt::OwnedImage& image) {
    if (image.empty() || CV_MAT_DEPTH(image.type()) != CV_8U ||
        (image.channels() != 1 && image.channels() != 3 && image.channels() != 4)) {
        throw visionrt::ContractError("Expected nonempty uint8 gray/BGR/BGRA image");
    }
}

cv::Mat toGray(const visionrt::OwnedImage& image) {
    requireU8(image);
    if (image.channels() == 1) return image.readOnlyMat();
    cv::Mat result;
    cv::cvtColor(image.readOnlyMat(), result,
                 image.channels() == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY);
    return result;
}

} // namespace

ImagePreprocessTool::ImagePreprocessTool(const visionrt::ToolServices& services)
    : services_(services) {}

visionrt::ToolStatus ImagePreprocessTool::initialize(const visionrt::Values& parameters) {
    if (!parameters.empty()) {
        return {1, "ImagePreprocessTool accepts no init parameters"};
    }
    return {};
}

visionrt::ToolResult ImagePreprocessTool::execute(const visionrt::Values& inputs) {
    try {
        auto imgIt = inputs.find("images");
        if (imgIt == inputs.end() || visionrt::isMissing(imgIt->second)) {
            throw visionrt::ContractError("Missing input: images");
        }
        const auto* images = std::get_if<visionrt::ImageBatch>(&imgIt->second);
        if (!images) throw visionrt::ContractError("Expected ImageBatch for input: images");

        auto getStr = [&](const std::string& key, const std::string& defVal) -> std::string {
            auto it = inputs.find(key);
            if (it == inputs.end() || visionrt::isMissing(it->second)) return defVal;
            if (auto* u = std::get_if<visionrt::Utf8>(&it->second)) return u->str();
            throw visionrt::ContractError("Expected string for input: " + key);
        };

        auto getInt = [&](const std::string& key, std::int64_t defVal) -> std::int64_t {
            auto it = inputs.find(key);
            if (it == inputs.end() || visionrt::isMissing(it->second)) return defVal;
            if (auto* v = std::get_if<std::int64_t>(&it->second)) return *v;
            throw visionrt::ContractError("Expected integer for input: " + key);
        };

        const auto operation = getStr("operation", "identity");
        if (operation != "identity" && operation != "gray" &&
            operation != "gaussian" && operation != "threshold") {
            throw visionrt::ContractError("Invalid preprocessing operation: " + operation);
        }

        int kernel = 0;
        std::int64_t threshold = 0;
        if (operation == "gaussian") {
            kernel = static_cast<int>(getInt("kernel", 3));
            if (kernel <= 0 || kernel % 2 == 0) {
                throw visionrt::ContractError("Gaussian kernel must be positive and odd");
            }
        }
        if (operation == "threshold") {
            threshold = getInt("threshold", 128);
            if (threshold < 0 || threshold > 255) {
                throw visionrt::ContractError("Threshold must be in [0, 255]");
            }
        }

        for (const auto& image : images->images()) {
            requireU8(image);
        }

        if (operation == "identity") {
            visionrt::Values outputs;
            outputs.emplace("images", *images);
            return {{}, std::move(outputs)};
        }

        std::vector<visionrt::OwnedImage> output;
        output.reserve(images->images().size());

        for (const auto& image : images->images()) {
            cv::Mat processed;
            if (operation == "gaussian") {
                cv::GaussianBlur(image.readOnlyMat(), processed, cv::Size(kernel, kernel), 0, 0,
                                 cv::BORDER_REFLECT_101);
            } else if (operation == "threshold") {
                cv::compare(toGray(image), cv::Scalar(threshold), processed, cv::CMP_GE);
            } else {
                processed = image.channels() == 1 ? image.clone() : toGray(image);
            }
            output.push_back(visionrt::OwnedImage::take(std::move(processed)));
        }

        visionrt::Values outputs;
        outputs.emplace("images", visionrt::ImageBatch(std::move(output)));
        return {{}, std::move(outputs)};
    } catch (const std::exception& e) {
        visionrt::Values outputs;
        outputs.emplace("images", visionrt::Missing{});
        return {{1, e.what()}, std::move(outputs)};
    }
}

visionrt::ToolStatus ImagePreprocessTool::release() {
    return {};
}

} // namespace visiontools::preprocess
