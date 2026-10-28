#include "src/preprocess/AdaptiveBinaryTool.h"
#include "../result_view.h"
#include "TestSupport.h"

using namespace visionrt;
using namespace visiontools::preprocess;
using namespace visiontools::tests;

namespace ab = visiontools::preprocess::ports::adaptive_binary;

void testAdaptiveBinaryTool()
{
    std::cout << "[TEST] Running testAdaptiveBinaryTool..." << std::endl;
    ToolServices       services;
    AdaptiveBinaryTool tool(services);

    static_assert(std::string_view(ab::in::IMAGES) == "images");
    static_assert(std::string_view(ab::in::BLOCK_SIZE) == "block_size");
    static_assert(std::string_view(ab::in::CONSTANT_OFFSET) == "constant_offset");
    static_assert(std::string_view(ab::out::IMAGES) == "images");

    // 1. Missing input
    {
        auto res = tool.execute({});
        requireFailure(res, "images", "Missing input 'images' must fail");
        REQUIRE(testview::status(res).message.find("缺失") != std::string::npos, "Error must be Chinese");
        REQUIRE(isMissing(res.outputs["images"]), "Output must be Missing");
    }

    // 2. Non-Image input
    {
        Values inputs;
        inputs.emplace(ab::in::IMAGES, std::int64_t(42));
        auto res = tool.execute(inputs);
        requireFailure(res, "图像", "Non-Image must fail");
        REQUIRE(testview::status(res).message.find("images") != std::string::npos, "Error must mention port 'images'");
    }

    // 3. B=0 -> success with empty batch
    {
        Values inputs;
        inputs.emplace(ab::in::IMAGES, DataRef<Image>(Image{}));
        auto res = tool.execute(inputs);
        requireSuccess(res, "B=0 must succeed");
        const auto *outBatch = std::get_if<DataRef<Image>>(&res.outputs["images"]);
        REQUIRE(outBatch != nullptr, "Output must be DataRef<Image>");
        REQUIRE(outBatch->get().empty(), "Output images must be empty");
    }

    // 4. Missing values for a non-empty batch
    {
        Values inputs;
        inputs.emplace(ab::in::IMAGES, DataRef<Image>(Image{(gradient(32, 32)).clone()}));
        auto res = tool.execute(inputs);
        requireFailure(res, "block_size", "Missing block_size must fail");
        REQUIRE(testview::status(res).message.find("缺失") != std::string::npos, "Missing value must be Chinese");
    }

    // 5. block_size even -> rejected
    {
        Values inputs;
        inputs.emplace(ab::in::IMAGES, DataRef<Image>(Image{(gradient(32, 32)).clone()}));
        inputs.emplace(ab::in::BLOCK_SIZE, std::int64_t(4));
        inputs.emplace(ab::in::CONSTANT_OFFSET, 2.0);
        auto res = tool.execute(inputs);
        requireFailure(res, "奇数", "Even block_size must be rejected");
        REQUIRE(testview::status(res).message.find("4") != std::string::npos, "Error must include the actual value");
    }

    // 6. block_size outside [3, 999]
    for (const std::int64_t value : {std::int64_t(1), std::int64_t(1000), std::int64_t(1) << 40})
    {
        Values inputs;
        inputs.emplace(ab::in::IMAGES, DataRef<Image>(Image{(gradient(32, 32)).clone()}));
        inputs.emplace(ab::in::BLOCK_SIZE, value);
        inputs.emplace(ab::in::CONSTANT_OFFSET, 2.0);
        auto res = tool.execute(inputs);
        requireFailure(res, "block_size", "block_size outside the declared range must fail");
        REQUIRE(testview::status(res).message.find("[3, 999]") != std::string::npos, "Range failure must state the allowed range");
    }

    // 7. Strict float check: Int constant_offset rejected
    {
        Values inputs;
        inputs.emplace(ab::in::IMAGES, DataRef<Image>(Image{(gradient(32, 32)).clone()}));
        inputs.emplace(ab::in::BLOCK_SIZE, std::int64_t(5));
        inputs.emplace(ab::in::CONSTANT_OFFSET, std::int64_t(5));
        auto res = tool.execute(inputs);
        requireFailure(res, "浮点", "Int constant_offset must be strictly rejected");
        REQUIRE(testview::status(res).message.find("constant_offset") != std::string::npos, "Error must mention constant_offset");
    }

    // 8. Non-finite constant_offset rejected
    {
        Values inputs;
        inputs.emplace(ab::in::IMAGES, DataRef<Image>(Image{(gradient(32, 32)).clone()}));
        inputs.emplace(ab::in::BLOCK_SIZE, std::int64_t(5));
        inputs.emplace(ab::in::CONSTANT_OFFSET, std::numeric_limits<double>::infinity());
        auto res = tool.execute(inputs);
        requireFailure(res, "有限", "Infinite constant_offset must be rejected");
    }

    // 9. Non-CV_8UC1 -> error points out batch index
    {
        cv::Mat bgr(64, 64, CV_8UC3, cv::Scalar(100, 150, 200));
        Values  inputs;
        inputs.emplace(ab::in::IMAGES, DataRef<Image>(Image{(bgr).clone()}));
        inputs.emplace(ab::in::BLOCK_SIZE, std::int64_t(5));
        inputs.emplace(ab::in::CONSTANT_OFFSET, 2.0);
        auto res = tool.execute(inputs);
        requireFailure(res, "批次索引 0", "BGR image must be rejected without auto-conversion");
        REQUIRE(testview::status(res).message.find("CV_8UC1") != std::string::npos, "Error must state the expected type");
    }

    // 10. Synthetic grayscale comparison against OpenCV reference
    {
        cv::Mat img1 = gradient(100, 120);
        cv::Mat img2(70, 90, CV_8UC1);
        for (int y = 0; y < img2.rows; ++y)
            for (int x = 0; x < img2.cols; ++x)
                img2.at<unsigned char>(y, x) = static_cast<unsigned char>((x * 7 + y * 5) % 256);

        Values inputs;
        inputs.emplace(ab::in::IMAGES, DataRef<Image>(Image{(img1).clone(), (img2).clone()}));
        inputs.emplace(ab::in::BLOCK_SIZE, std::int64_t(11));
        inputs.emplace(ab::in::CONSTANT_OFFSET, 3.5);
        auto res = tool.execute(inputs);
        requireSuccess(res, "Valid adaptive binary execution must succeed");
        const auto *outBatch = std::get_if<DataRef<Image>>(&res.outputs["images"]);
        REQUIRE(outBatch != nullptr && outBatch->get().size() == 2, "Batch size must be 2");

        cv::Mat ref1;
        cv::adaptiveThreshold(img1, ref1, 255.0, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 11, 3.5);
        REQUIRE(cv::countNonZero(ref1 != outBatch->get()[0]) == 0, "Pixel-for-pixel match required for img1");
        cv::Mat ref2;
        cv::adaptiveThreshold(img2, ref2, 255.0, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 11, 3.5);
        REQUIRE(cv::countNonZero(ref2 != outBatch->get()[1]) == 0, "Pixel-for-pixel match required for img2");
    }

    std::cout << "[PASS] testAdaptiveBinaryTool passed." << std::endl;
}
