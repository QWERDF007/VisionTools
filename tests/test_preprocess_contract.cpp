#include <visionrt/sdk/Tool.h>
#include <visionrt/sdk/Module.h>

#include "src/preprocess/ReadImageTool.h"
#include "src/preprocess/RoiProjectionTool.h"
#include "src/preprocess/ImageCropTool.h"
#include "src/preprocess/ImagePreprocessTool.h"
#include "src/preprocess/DetectionFilterTool.h"
#include "src/preprocess/AdaptiveBinaryTool.h"

#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <limits>

#define REQUIRE(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed: " << #cond << " (" << msg << ") at line " << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while(0)

using namespace visionrt;
using namespace visiontools::preprocess;

void testAdaptiveBinaryTool() {
    std::cout << "[TEST] Running testAdaptiveBinaryTool..." << std::endl;
    ToolServices services;
    AdaptiveBinaryTool tool(services);

    // 1. Missing input
    {
        Values inputs;
        auto res = tool.execute(inputs);
        REQUIRE(!res.status, "Missing input 'images' must fail");
        REQUIRE(res.status.message.find("images") != std::string::npos, "Error must mention port 'images'");
        REQUIRE(isMissing(res.outputs["images"]), "Output must be Missing");
    }

    // 2. Non-ImageBatch input
    {
        Values inputs;
        inputs.emplace("images", std::int64_t(42));
        auto res = tool.execute(inputs);
        REQUIRE(!res.status, "Non-ImageBatch must fail");
        REQUIRE(res.status.message.find("images") != std::string::npos, "Error must mention port 'images'");
    }

    // 3. B=0 -> Success with empty batch
    {
        Values inputs;
        inputs.emplace("images", ImageBatch{});
        auto res = tool.execute(inputs);
        REQUIRE(res.status, "B=0 must succeed");
        const auto* outBatch = std::get_if<ImageBatch>(&res.outputs["images"]);
        REQUIRE(outBatch != nullptr, "Output must be ImageBatch");
        REQUIRE(outBatch->images().empty(), "Output images must be empty");
    }

    // 4. block_size even -> rejected
    {
        cv::Mat gray(64, 64, CV_8UC1, cv::Scalar(128));
        Values inputs;
        inputs.emplace("images", ImageBatch({OwnedImage::copyOf(gray)}));
        inputs.emplace("block_size", std::int64_t(4)); // Even
        auto res = tool.execute(inputs);
        REQUIRE(!res.status, "Even block_size must be rejected");
        REQUIRE(res.status.message.find("odd") != std::string::npos, "Error must state odd number requirement");
    }

    // 5. block_size < 3 -> rejected
    {
        cv::Mat gray(64, 64, CV_8UC1, cv::Scalar(128));
        Values inputs;
        inputs.emplace("images", ImageBatch({OwnedImage::copyOf(gray)}));
        inputs.emplace("block_size", std::int64_t(1));
        auto res = tool.execute(inputs);
        REQUIRE(!res.status, "block_size < 3 must be rejected");
    }

    // 6. Non-CV_8UC1 (e.g. 3-channel BGR) -> error points out batch index
    {
        cv::Mat bgr(64, 64, CV_8UC3, cv::Scalar(100, 150, 200));
        Values inputs;
        inputs.emplace("images", ImageBatch({OwnedImage::copyOf(bgr)}));
        auto res = tool.execute(inputs);
        REQUIRE(!res.status, "BGR image must be rejected without auto-conversion");
        REQUIRE(res.status.message.find("batch index 0") != std::string::npos, "Error must point out batch index 0");
    }

    // 7. Strict float check: Int constant_offset rejected (VS-03 / TS-04 / F07)
    {
        cv::Mat gray(64, 64, CV_8UC1, cv::Scalar(128));
        Values inputs;
        inputs.emplace("images", ImageBatch({OwnedImage::copyOf(gray)}));
        inputs.emplace("constant_offset", std::int64_t(5)); // Int passed instead of Float
        auto res = tool.execute(inputs);
        REQUIRE(!res.status, "Int constant_offset must be strictly rejected");
        REQUIRE(res.status.message.find("constant_offset") != std::string::npos, "Error must mention constant_offset");
    }

    // 8. Non-finite constant_offset rejected (NaN or Inf)
    {
        cv::Mat gray(64, 64, CV_8UC1, cv::Scalar(128));
        Values inputs;
        inputs.emplace("images", ImageBatch({OwnedImage::copyOf(gray)}));
        inputs.emplace("constant_offset", std::numeric_limits<double>::infinity());
        auto res = tool.execute(inputs);
        REQUIRE(!res.status, "Infinite constant_offset must be rejected");
    }

    // 9. Synthetic 8U grayscale comparison against OpenCV reference
    {
        cv::Mat img1(100, 120, CV_8UC1);
        for (int y = 0; y < img1.rows; ++y) {
            for (int x = 0; x < img1.cols; ++x) {
                img1.at<unsigned char>(y, x) = static_cast<unsigned char>((x * 3 + y * 2) % 256);
            }
        }
        cv::Mat img2(70, 90, CV_8UC1); // Heterogeneous batch
        for (int y = 0; y < img2.rows; ++y) {
            for (int x = 0; x < img2.cols; ++x) {
                img2.at<unsigned char>(y, x) = static_cast<unsigned char>((x * 7 + y * 5) % 256);
            }
        }

        std::int64_t blockSize = 11;
        double constantOffset = 3.5;

        Values inputs;
        inputs.emplace("images", ImageBatch({OwnedImage::copyOf(img1), OwnedImage::copyOf(img2)}));
        inputs.emplace("block_size", blockSize);
        inputs.emplace("constant_offset", constantOffset);

        auto res = tool.execute(inputs);
        REQUIRE(res.status, "Valid adaptive binary execution must succeed");

        const auto* outBatch = std::get_if<ImageBatch>(&res.outputs["images"]);
        REQUIRE(outBatch != nullptr, "Output must be ImageBatch");
        REQUIRE(outBatch->images().size() == 2, "Batch size must be 2");

        // Reference check img1
        cv::Mat ref1;
        cv::adaptiveThreshold(img1, ref1, 255.0, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY,
                             static_cast<int>(blockSize), constantOffset);
        const auto& out1 = outBatch->images()[0].readOnlyMat();
        REQUIRE(cv::countNonZero(ref1 != out1) == 0, "Pixel-for-pixel match required for img1");

        // Reference check img2
        cv::Mat ref2;
        cv::adaptiveThreshold(img2, ref2, 255.0, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY,
                             static_cast<int>(blockSize), constantOffset);
        const auto& out2 = outBatch->images()[1].readOnlyMat();
        REQUIRE(cv::countNonZero(ref2 != out2) == 0, "Pixel-for-pixel match required for img2");
    }

    std::cout << "[PASS] testAdaptiveBinaryTool passed." << std::endl;
}

void testRoiProjectionTool() {
    std::cout << "[TEST] Running testRoiProjectionTool..." << std::endl;
    ToolServices services;
    RoiProjectionTool tool(services);

    // 100x100 black image with a 40x40 white square in center [30, 70]
    cv::Mat img(100, 100, CV_8UC1, cv::Scalar(0));
    img(cv::Rect(30, 30, 40, 40)).setTo(cv::Scalar(255));

    Values inputs;
    inputs.emplace("images", ImageBatch({OwnedImage::copyOf(img)}));
    inputs.emplace("axis", Utf8("x"));
    inputs.emplace("polarity", Utf8("bright"));
    inputs.emplace("threshold", std::int64_t(128));
    inputs.emplace("min_pixels", std::int64_t(10));
    inputs.emplace("min_extent", std::int64_t(10));

    auto res = tool.execute(inputs);
    REQUIRE(res.status, "Projection must succeed");

    const auto* rois = std::get_if<RoiBatch>(&res.outputs["rois"]);
    REQUIRE(rois != nullptr, "Output must contain rois");
    REQUIRE(rois->images().size() == 1, "Rois batch must be 1");
    REQUIRE(rois->images()[0].size() == 1, "Expected 1 projected ROI");

    const auto encoded = rois->images()[0][0].encoded();
    REQUIRE(encoded[0] == 2.0F, "ROI must be rectangle");
    REQUIRE(std::abs(encoded[1] - 30.0F) < 1e-3, "ROI left must be 30");
    REQUIRE(std::abs(encoded[3] - 40.0F) < 1e-3, "ROI width must be 40");

    std::cout << "[PASS] testRoiProjectionTool passed." << std::endl;
}

void testImageCropTool() {
    std::cout << "[TEST] Running testImageCropTool..." << std::endl;
    ToolServices services;
    ImageCropTool tool(services);

    cv::Mat img(100, 100, CV_8UC1, cv::Scalar(100));
    img(cv::Rect(20, 20, 30, 30)).setTo(cv::Scalar(200));

    OwnedImage srcImage = OwnedImage::copyOf(img);
    auto roi = Roi::rectangle(20.0F, 20.0F, 30.0F, 30.0F, 0.0F);

    OwnedImage savedCropped;
    {
        Values inputs;
        inputs.emplace("images", ImageBatch({srcImage}));
        inputs.emplace("rois", RoiBatch({{roi}}));

        auto res = tool.execute(inputs);
        REQUIRE(res.status, "Crop must succeed");

        const auto* cropped = std::get_if<ImageBatch>(&res.outputs["images"]);
        REQUIRE(cropped != nullptr, "Output must contain cropped images");
        REQUIRE(cropped->images().size() == 1, "Expected 1 cropped image");
        REQUIRE(cropped->images()[0].cols() == 30, "Cropped width must be 30");
        REQUIRE(cropped->images()[0].rows() == 30, "Cropped height must be 30");

        // Zero-copy verification: underlying pixel buffer must be shared with input (VS-03 / TS-03 / F06)
        REQUIRE(cropped->images()[0].readOnlyMat().u == srcImage.readOnlyMat().u, "Rect crop must share pixel buffer (u) with input");
        REQUIRE(cropped->images()[0].rowStride() == srcImage.rowStride(), "Rect crop must preserve rowStride");

        savedCropped = cropped->images()[0];
    }

    // Input batch is now out of scope; verify savedCropped is still valid and readable
    REQUIRE(savedCropped.readOnlyMat().at<unsigned char>(0, 0) == 200, "Cropped image must remain readable after input scope exit");

    std::cout << "[PASS] testImageCropTool passed." << std::endl;
}

void testDetectionFilterTool() {
    std::cout << "[TEST] Running testDetectionFilterTool..." << std::endl;
    ToolServices services;
    DetectionFilterTool tool(services);

    cv::Mat img(100, 100, CV_8UC1, cv::Scalar(0));
    auto roi = Roi::rectangle(10.0F, 10.0F, 20.0F, 20.0F, 0.0F);

    std::vector<float> scores = {0.85F, 0.35F};
    std::vector<std::int64_t> classes = {1, 2};
    std::vector<std::string> names = {"cat", "dog"};
    std::vector<std::int64_t> allowed = {1, 2};

    auto makeNumArray = [](DType dtype, auto vals) {
        return NumericArray(dtype, 2, Layout::BatchRagged,
            {NumericBuffer::copy(dtype, {static_cast<std::int64_t>(vals.size())}, std::as_bytes(std::span(vals)))});
    };

    // 1. Invalid threshold type (String instead of double/int) -> rejected
    {
        Values inputs;
        inputs.emplace("images", ImageBatch({OwnedImage::copyOf(img)}));
        inputs.emplace("rois", RoiBatch({{roi, roi}}));
        inputs.emplace("scores", makeNumArray(DType::Float32, scores));
        inputs.emplace("classes", makeNumArray(DType::Int64, classes));
        inputs.emplace("class_names", StringArray(2, Layout::BatchRagged, {StringBuffer({2}, names)}));
        inputs.emplace("allowed_classes", NumericArray(DType::Int64, 1, Layout::Dense,
            {NumericBuffer::copy(DType::Int64, {2}, std::as_bytes(std::span(allowed)))}));
        inputs.emplace("min_score", Utf8("not_a_number")); // String type

        auto res = tool.execute(inputs);
        REQUIRE(!res.status, "Non-numeric min_score must be rejected");
    }

    // 2. Valid execution: filter out score < 0.5
    {
        Values inputs;
        inputs.emplace("images", ImageBatch({OwnedImage::copyOf(img)}));
        inputs.emplace("rois", RoiBatch({{roi, roi}}));
        inputs.emplace("scores", makeNumArray(DType::Float32, scores));
        inputs.emplace("classes", makeNumArray(DType::Int64, classes));
        inputs.emplace("class_names", StringArray(2, Layout::BatchRagged, {StringBuffer({2}, names)}));
        inputs.emplace("allowed_classes", NumericArray(DType::Int64, 1, Layout::Dense,
            {NumericBuffer::copy(DType::Int64, {2}, std::as_bytes(std::span(allowed)))}));
        inputs.emplace("min_score", 0.5);

        auto res = tool.execute(inputs);
        REQUIRE(res.status, "Filter must succeed");
        const auto* outRois = std::get_if<RoiBatch>(&res.outputs["rois"]);
        REQUIRE(outRois != nullptr, "Output must contain rois");
        REQUIRE(outRois->images()[0].size() == 1, "Expected 1 ROI to pass score >= 0.5");
    }

    std::cout << "[PASS] testDetectionFilterTool passed." << std::endl;
}

void testImagePreprocessTool() {
    std::cout << "[TEST] Running testImagePreprocessTool..." << std::endl;
    ToolServices services;
    ImagePreprocessTool tool(services);

    cv::Mat img(50, 50, CV_8UC1, cv::Scalar(150));
    OwnedImage owned = OwnedImage::copyOf(img);

    // 1. Identity pass-through
    {
        Values inputs;
        inputs.emplace("images", ImageBatch({owned}));
        inputs.emplace("operation", Utf8("identity"));
        auto res = tool.execute(inputs);
        REQUIRE(res.status, "Identity must succeed");
        const auto* out = std::get_if<ImageBatch>(&res.outputs["images"]);
        REQUIRE(out != nullptr, "Must have ImageBatch");
        REQUIRE(out->images()[0].data() == owned.data(), "Identity must not copy pixel buffer");
    }

    // 2. Threshold
    {
        Values inputs;
        inputs.emplace("images", ImageBatch({owned}));
        inputs.emplace("operation", Utf8("threshold"));
        inputs.emplace("threshold", std::int64_t(100));
        auto res = tool.execute(inputs);
        REQUIRE(res.status, "Threshold must succeed");
        const auto* out = std::get_if<ImageBatch>(&res.outputs["images"]);
        REQUIRE(out->images()[0].readOnlyMat().at<unsigned char>(0, 0) == 255, "150 >= 100 -> 255");
    }

    std::cout << "[PASS] testImagePreprocessTool passed." << std::endl;
}

int main() {
    try {
        testAdaptiveBinaryTool();
        testRoiProjectionTool();
        testImageCropTool();
        testDetectionFilterTool();
        testImagePreprocessTool();
        std::cout << "\n[ALL TESTS PASSED] All preprocess contract tests passed successfully.\n" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Unhandled exception: " << e.what() << std::endl;
        return 1;
    }
}
