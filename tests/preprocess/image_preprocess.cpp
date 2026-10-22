#include "src/preprocess/ImagePreprocessTool.h"
#include "TestSupport.h"

using namespace visionrt;
using namespace visiontools::preprocess;
using namespace visiontools::tests;

namespace preprocess_ports = visiontools::preprocess::ports::image_preprocess;

void testImagePreprocessTool()
{
    std::cout << "[TEST] Running testImagePreprocessTool..." << std::endl;
    ToolServices        services;
    ImagePreprocessTool tool(services);

    static_assert(std::string_view(preprocess_ports::in::OPERATION) == "operation");
    static_assert(std::string_view(preprocess_ports::in::KERNEL) == "kernel");
    static_assert(std::string_view(preprocess_ports::in::THRESHOLD) == "threshold");
    static_assert(std::string_view(preprocess_ports::out::IMAGES) == "images");

    cv::Mat img(50, 50, CV_8UC1, cv::Scalar(150));
    cv::Mat owned = (img).clone();

    // 1. Identity pass-through
    {
        Values inputs;
        inputs.emplace(preprocess_ports::in::IMAGES, DataRef<Image>(Image{owned}));
        inputs.emplace(preprocess_ports::in::OPERATION, std::string("identity"));
        auto res = tool.execute(inputs);
        requireSuccess(res, "Identity must succeed");
        const auto *out = std::get_if<DataRef<Image>>(&res.outputs[preprocess_ports::out::IMAGES]);
        REQUIRE(out != nullptr && out->get()[0].data == owned.data, "Identity must not copy pixel buffer");
    }

    // 2. Threshold
    {
        Values inputs;
        inputs.emplace(preprocess_ports::in::IMAGES, DataRef<Image>(Image{owned}));
        inputs.emplace(preprocess_ports::in::OPERATION, std::string("threshold"));
        inputs.emplace(preprocess_ports::in::THRESHOLD, std::int64_t(100));
        auto res = tool.execute(inputs);
        requireSuccess(res, "Threshold must succeed");
        const auto *out = std::get_if<DataRef<Image>>(&res.outputs[preprocess_ports::out::IMAGES]);
        REQUIRE(out->get()[0].at<unsigned char>(0, 0) == 255, "150 >= 100 -> 255");
    }

    // 3. Gaussian
    {
        Values inputs;
        inputs.emplace(preprocess_ports::in::IMAGES, DataRef<Image>(Image{owned}));
        inputs.emplace(preprocess_ports::in::OPERATION, std::string("gaussian"));
        inputs.emplace(preprocess_ports::in::KERNEL, std::int64_t(3));
        auto res = tool.execute(inputs);
        requireSuccess(res, "Gaussian must succeed");
        const auto *out = std::get_if<DataRef<Image>>(&res.outputs[preprocess_ports::out::IMAGES]);
        REQUIRE(out->get()[0].cols == 50, "Gaussian must preserve geometry");
    }

    // 4. Missing operation / invalid operation / odd kernel / range
    {
        Values inputs;
        inputs.emplace(preprocess_ports::in::IMAGES, DataRef<Image>(Image{owned}));
        auto res = tool.execute(inputs);
        requireFailure(res, "operation", "Missing operation must fail");
        REQUIRE(res.status.message.find("缺失") != std::string::npos, "Missing value must be Chinese");
    }
    {
        Values inputs;
        inputs.emplace(preprocess_ports::in::IMAGES, DataRef<Image>(Image{owned}));
        inputs.emplace(preprocess_ports::in::OPERATION, std::string("unknown"));
        auto res = tool.execute(inputs);
        requireFailure(res, "operation", "Invalid operation must fail");
        REQUIRE(res.status.message.find("unknown") != std::string::npos, "Error must include the invalid value");
    }
    {
        Values inputs;
        inputs.emplace(preprocess_ports::in::IMAGES, DataRef<Image>(Image{owned}));
        inputs.emplace(preprocess_ports::in::OPERATION, std::string("gaussian"));
        inputs.emplace(preprocess_ports::in::KERNEL, std::int64_t(4));
        auto res = tool.execute(inputs);
        requireFailure(res, "奇数", "Even Gaussian kernel must fail");
    }
    {
        Values inputs;
        inputs.emplace(preprocess_ports::in::IMAGES, DataRef<Image>(Image{owned}));
        inputs.emplace(preprocess_ports::in::OPERATION, std::string("gaussian"));
        inputs.emplace(preprocess_ports::in::KERNEL, std::int64_t(1) << 40);
        auto res = tool.execute(inputs);
        requireFailure(res, "kernel", "Kernel outside int range must fail before narrowing");
    }
    {
        Values inputs;
        inputs.emplace(preprocess_ports::in::IMAGES, DataRef<Image>(Image{owned}));
        inputs.emplace(preprocess_ports::in::OPERATION, std::string("threshold"));
        inputs.emplace(preprocess_ports::in::THRESHOLD, std::int64_t(300));
        auto res = tool.execute(inputs);
        requireFailure(res, "threshold", "Out-of-range threshold must fail");
        REQUIRE(res.status.message.find("[0, 255]") != std::string::npos, "Range must be stated");
    }

    // 5. Invalid image type names the batch index
    {
        cv::Mat invalid(10, 10, CV_16UC1, cv::Scalar(1));
        Values  inputs;
        inputs.emplace(preprocess_ports::in::IMAGES, DataRef<Image>(Image{owned, (invalid).clone()}));
        inputs.emplace(preprocess_ports::in::OPERATION, std::string("gray"));
        auto res = tool.execute(inputs);
        requireFailure(res, "批次索引 1", "Invalid image type must name the batch index");
    }

    // 6. Empty batch
    {
        Values inputs;
        inputs.emplace(preprocess_ports::in::IMAGES, DataRef<Image>(Image{}));
        inputs.emplace(preprocess_ports::in::OPERATION, std::string("identity"));
        auto res = tool.execute(inputs);
        requireSuccess(res, "Empty batch must succeed");
        REQUIRE(std::get<DataRef<Image>>(res.outputs[preprocess_ports::out::IMAGES]).get().empty(),
                "Empty batch must stay empty");
    }

    std::cout << "[PASS] testImagePreprocessTool passed." << std::endl;
}
