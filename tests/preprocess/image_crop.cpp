#include "src/preprocess/ImageCropTool.h"
#include "TestSupport.h"

#include <initializer_list>
#include <string_view>

using namespace visionrt;
using namespace visiontools::preprocess;
using namespace visiontools::tests;

namespace crop_ports = visiontools::preprocess::ports::image_crop;

namespace {

cv::Mat pixelCenterGold(const cv::Mat &image, cv::Point origin, std::initializer_list<std::string_view> rows)
{
    cv::Mat expected
        = cv::Mat::zeros(static_cast<int>(rows.size()), static_cast<int>(rows.begin()->size()), image.type());
    int row = 0;
    for (const auto cells : rows)
    {
        REQUIRE(cells.size() == static_cast<std::size_t>(expected.cols), "Golden mask rows must be rectangular");
        for (int column = 0; column < expected.cols; ++column)
            if (cells[column] == '#')
                expected.at<unsigned char>(row, column) = image.at<unsigned char>(origin.y + row, origin.x + column);
        ++row;
    }
    return expected;
}

} // namespace

void testImageCropTool()
{
    std::cout << "[TEST] Running testImageCropTool..." << std::endl;
    ToolServices  services;
    ImageCropTool tool(services);

    static_assert(std::string_view(crop_ports::in::IMAGES) == "images");
    static_assert(std::string_view(crop_ports::in::RECT) == "rect");
    static_assert(std::string_view(crop_ports::out::ORIGINS) == "origins");

    // 1. Integer rectangle: zero-copy view and input owner release
    cv::Mat savedCropped;
    {
        cv::Mat img(100, 100, CV_8UC1, cv::Scalar(100));
        img(cv::Rect(20, 20, 30, 30)).setTo(cv::Scalar(200));
        cv::Mat srcImage = (img).clone();
        auto    roi      = std::vector<double>{2, 20.0, 20.0, 30.0, 30.0, 0.0};
        Values  inputs;
        inputs.emplace(crop_ports::in::IMAGES, DataRef<Image>(Image{srcImage}));
        inputs.emplace(crop_ports::in::RECT, DataRef<ROI>(ROI{{roi}}));
        auto res = tool.execute(inputs);
        requireSuccess(res, "Crop must succeed");
        const auto *cropped = std::get_if<DataRef<Image>>(&res.outputs[crop_ports::out::IMAGES]);
        REQUIRE(cropped != nullptr && cropped->get().size() == 1, "Expected 1 cropped image");
        REQUIRE(cropped->get()[0].cols == 30 && cropped->get()[0].rows == 30, "Cropped size must be 30x30");
        REQUIRE(cropped->get()[0].u == srcImage.u, "Rect crop must share pixel buffer (u) with input");
        REQUIRE(cropped->get()[0].step[0] == srcImage.step[0], "Rect crop must preserve rowStride");
        savedCropped = cropped->get()[0];
    }
    REQUIRE(savedCropped.at<unsigned char>(0, 0) == 200, "Cropped image must remain readable after input scope exit");

    // 2. Fractional ROI on non-uniform image
    {
        cv::Mat nonUniform(10, 10, CV_8UC1);
        for (int r = 0; r < 10; ++r)
            for (int c = 0; c < 10; ++c)
                nonUniform.at<unsigned char>(r, c) = static_cast<unsigned char>(r * 10 + c + 1);
        auto   roi = std::vector<double>{2, 0.75, 0.0, 1.0, 1.0, 0.0};
        Values inputs;
        inputs.emplace(crop_ports::in::IMAGES, DataRef<Image>(Image{(nonUniform).clone()}));
        inputs.emplace(crop_ports::in::RECT, DataRef<ROI>(ROI{{roi}}));
        auto res = tool.execute(inputs);
        requireSuccess(res, "Fractional crop must succeed via mask path");
        const auto *cropped = std::get_if<DataRef<Image>>(&res.outputs[crop_ports::out::IMAGES]);
        REQUIRE(cropped && cropped->get().size() == 1, "Expected 1 output image");
        const auto &outMat = cropped->get()[0];
        REQUIRE(outMat.cols == 2 && outMat.rows == 1, "Bounds width must be 2, height 1");
        REQUIRE(outMat.at<unsigned char>(0, 0) == 0, "Col 0 center is outside and must be 0");
        REQUIRE(outMat.at<unsigned char>(0, 1) == nonUniform.at<unsigned char>(0, 1), "Col 1 center must be preserved");
    }

    // 3. Narrow fractional ROI has no valid pixel center
    {
        cv::Mat img(10, 10, CV_8UC1, cv::Scalar(100));
        Values  inputs;
        inputs.emplace(crop_ports::in::IMAGES, DataRef<Image>(Image{(img).clone()}));
        inputs.emplace(crop_ports::in::RECT, DataRef<ROI>(ROI{{std::vector<double>{2, 0.1, 0.1, 0.1, 0.1, 0.0}}}));
        auto res = tool.execute(inputs);
        requireFailure(res, "没有有效像素中心", "Narrow ROI with no pixel centers must fail");
        REQUIRE(res.status.message.find("图像索引 0") != std::string::npos, "Error must name the image index");
    }

    // 4. Batch order, clipping and origins
    {
        const auto first  = (cv::Mat(8, 10, CV_8UC1, cv::Scalar(31))).clone();
        const auto second = (cv::Mat(12, 14, CV_8UC1, cv::Scalar(79))).clone();
        Values     inputs{
                {crop_ports::in::IMAGES,                        DataRef<Image>(Image{first, second})},
                {  crop_ports::in::RECT, DataRef<ROI>(ROI{{std::vector<double>{2, -2, 1, 6, 3, 0}}})}
        };
        auto result = tool.execute(inputs);
        requireSuccess(result, "One rectangle must crop every image in batch order");
        const auto &batch = std::get<DataRef<Image>>(result.outputs.at(crop_ports::out::IMAGES)).get();
        REQUIRE(batch.size() == 2, "Crop must preserve batch cardinality");
        REQUIRE(batch[0].cols == 4 && batch[1].rows == 3, "Crop must clip partial bounds");
        REQUIRE(batch[0].at<unsigned char>(0, 0) == 31, "First image order must be retained");
        REQUIRE(batch[1].at<unsigned char>(0, 0) == 79, "Second image order must be retained");
        REQUIRE(result.outputs.count("source_indices") == 0, "Scalar rectangle crop has no expanded ROI indices");
        const auto &origin
            = std::get<Array>(result.outputs.at(crop_ports::out::ORIGINS)).get<std::vector<std::vector<double>>>();
        REQUIRE(origin.size() == 2 && origin[0].size() == 2 && origin[1].size() == 2, "Origins must have B by 2 shape");
        const auto coordinates = origin;
        REQUIRE(coordinates[0][0] == 0 && coordinates[0][1] == 1 && coordinates[1][0] == 0 && coordinates[1][1] == 1,
                "Each origin must correspond to the same-index crop");
    }

    // 5. Invalid inputs never publish partial outputs
    {
        Values inputs{
            {crop_ports::in::IMAGES, DataRef<Image>(Image{(cv::Mat(4, 4, CV_8UC1, cv::Scalar(8))).clone()})}
        };
        for (const PortValue &invalid : std::vector<PortValue>{
                 Missing{},
                 DataRef<ROI>(ROI{{{3, 1, 1, 1}}}
                 ), DataRef<ROI>(ROI{}
                 ),
                 DataRef<ROI>(ROI{{{2, 0, 0, 2, 2, 0}, {2, 0, 0, 2, 2, 0}}}
                 ),
                 DataRef<ROI>(ROI{{{2, 0, 0, 2, 2, 0}}, {{2, 0, 0, 2, 2, 0}}}
                 ),
                 DataRef<ROI>(ROI{{{2, 10, 10, 2, 2, 0}}}
                 )
        })
        {
            inputs.insert_or_assign(crop_ports::in::RECT, invalid);
            auto result = tool.execute(inputs);
            REQUIRE(!result.status, "Missing, non-rectangle, batch and disjoint inputs must fail");
            REQUIRE(isMissing(result.outputs.at(crop_ports::out::IMAGES))
                        && isMissing(result.outputs.at(crop_ports::out::ORIGINS)),
                    "Failure must not publish partial outputs");
        }
        inputs.insert_or_assign(crop_ports::in::IMAGES, DataRef<Image>(Image{}));
        inputs.insert_or_assign(crop_ports::in::RECT, DataRef<ROI>(ROI{{std::vector<double>{2, 0, 0, 2, 2, 0}}}));
        auto empty = tool.execute(inputs);
        requireSuccess(empty, "A valid empty image batch must yield an empty crop batch");
        REQUIRE(std::get<DataRef<Image>>(empty.outputs.at(crop_ports::out::IMAGES)).get().empty(),
                "Empty batch must produce an empty crop batch");
        REQUIRE(
            std::get<Array>(empty.outputs.at(crop_ports::out::ORIGINS)).get<std::vector<std::vector<double>>>().empty(),
            "An empty batch must have zero origin rows");
        inputs.insert_or_assign(crop_ports::in::IMAGES,
                                DataRef<Image>(Image{(cv::Mat(4, 4, CV_8UC1, cv::Scalar(8))).clone(),
                                                     (cv::Mat(1, 1, CV_8UC1, cv::Scalar(9))).clone()}));
        inputs.insert_or_assign(crop_ports::in::RECT, DataRef<ROI>(ROI{{std::vector<double>{2, 2, 2, 2, 2, 45}}}));
        auto partial = tool.execute(inputs);
        REQUIRE(!partial.status && isMissing(partial.outputs.at(crop_ports::out::IMAGES)),
                "A later disjoint image must fail the whole batch without publishing earlier crops");
    }

    // 6. Rotated/subpixel golden pixels and origins
    {
        cv::Mat    image     = gradient(24, 30);
        const auto rectangle = std::vector<double>{2, 2.3, 3.7, 12.4, 9.2, 30.0};
        Values     inputs{
                {crop_ports::in::IMAGES, DataRef<Image>(Image{(image).clone()})},
                {  crop_ports::in::RECT,         DataRef<ROI>(ROI{{rectangle}})}
        };
        auto result = tool.execute(inputs);
        requireSuccess(result, "Rotated crop must succeed");
        const auto &cropped  = std::get<DataRef<Image>>(result.outputs.at(crop_ports::out::IMAGES)).get()[0];
        const auto  expected = pixelCenterGold(image, {0, 1},
                                               {
                                                  ".....#...........",
                                                  ".....###.........",
                                                  "....#####........",
                                                  "....#######......",
                                                  "...##########....",
                                                  "..#############..",
                                                  "..##############.",
                                                  ".###############.",
                                                  ".##############..",
                                                  "...###########...",
                                                  ".....#########...",
                                                  ".......######....",
                                                  "........#####....",
                                                  "..........##.....",
                                                  ".................",
                                              });
        REQUIRE(cropped.rows == expected.rows && cropped.cols == expected.cols, "Rotated bounds must match gold");
        REQUIRE(cv::countNonZero(cropped != expected) == 0, "Rotated crop must match pixel-center gold");
        const auto coordinates
            = std::get<Array>(result.outputs.at(crop_ports::out::ORIGINS)).get<std::vector<std::vector<double>>>()[0];
        REQUIRE(coordinates == std::vector<double>({0., 1.}), "Rotated origin must match the pixel-center gold");
    }

    {
        const auto image = gradient(6, 6);
        Values     inputs{
                {crop_ports::in::IMAGES,                     DataRef<Image>(Image{image})},
                {  crop_ports::in::RECT, DataRef<ROI>(ROI{{{2., -1., -1., 4., 4., 45.}}})},
        };
        const auto result = tool.execute(inputs);
        requireSuccess(result, "Rotated rectangle clipped by two image edges must succeed");
        const auto &cropped  = std::get<DataRef<Image>>(result.outputs.at(crop_ports::out::IMAGES)).get().at(0);
        const auto  expected = pixelCenterGold(image, {0, 0}, {"###.", "###.", "##..", "...."});
        REQUIRE(cropped.size() == expected.size() && cv::countNonZero(cropped != expected) == 0,
                "Clipped rotated pixels must match independent gold");
        REQUIRE((std::get<Array>(result.outputs.at(crop_ports::out::ORIGINS)).get<NestedVector<double, 2>>()
                    == NestedVector<double, 2>{{0., 0.}}), "Clipped rotated origin must match gold");
    }
    {
        const auto   image = gradient(6, 6);
        const double half  = std::sqrt(2.0);
        Values       inputs{
                  {crop_ports::in::IMAGES,                                               DataRef<Image>(Image{image})},
                  {  crop_ports::in::RECT, DataRef<ROI>(ROI{{{2., 2. - half, 2. - half, 2. * half, 2. * half, 45.}}})},
        };
        const auto result = tool.execute(inputs);
        requireSuccess(result, "Pixel centers on rotated edges must be included");
        const auto &cropped  = std::get<DataRef<Image>>(result.outputs.at(crop_ports::out::IMAGES)).get().at(0);
        const auto  expected = pixelCenterGold(image, {0, 0}, {".##.", "####", "####", ".##."});
        REQUIRE(cropped.size() == expected.size() && cv::countNonZero(cropped != expected) == 0,
                "Rotated boundary centers must match independent gold");
        REQUIRE((std::get<Array>(result.outputs.at(crop_ports::out::ORIGINS)).get<NestedVector<double, 2>>()
                    == NestedVector<double, 2>{{0., 0.}}), "Rotated boundary origin must match gold");
    }

    // 7. Mask reuse within one execute, no cache across executes
    {
        const auto rectangle = std::vector<double>{2, 10.0, 10.0, 15.0, 15.0, 15.0};
        {
            Values single{
                {crop_ports::in::IMAGES, DataRef<Image>(Image{(gradient(40, 40)).clone()})},
                {  crop_ports::in::RECT,                    DataRef<ROI>(ROI{{rectangle}})}
            };
            testing::maskBuildCount = 0;
            auto result             = tool.execute(single);
            requireSuccess(result, "1-image rotated crop must succeed");
            REQUIRE(testing::maskBuildCount == 1, "One image must build exactly one mask");
        }
        std::vector<cv::Mat> sameBounds;
        for (int i = 0; i < 8; ++i) sameBounds.push_back((gradient(40, 40)).clone());
        Values inputs{
            {crop_ports::in::IMAGES,     DataRef<Image>(sameBounds)},
            {  crop_ports::in::RECT, DataRef<ROI>(ROI{{rectangle}})}
        };
        testing::maskBuildCount = 0;
        const auto started      = std::chrono::steady_clock::now();
        auto       result       = tool.execute(inputs);
        const auto elapsed
            = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        requireSuccess(result, "8-image rotated crop must succeed");
        REQUIRE(testing::maskBuildCount == 1, "Same bounds must build exactly one mask for 8 images");
        const auto &batch       = std::get<DataRef<Image>>(result.outputs.at(crop_ports::out::IMAGES)).get();
        const auto  expected    = pixelCenterGold(
            gradient(40, 40), {8, 8},
            {
                "....#..............", "....#####..........", "....########.......", "...#############...",
                "...################", "...###############.", "...###############.", "..################.",
                "..################.", "..###############..", ".################..", ".################..",
                ".###############...", ".###############...", "################...", "...#############...",
                ".......########....", "..........#####....", "..............#....",
            });
        std::size_t outputBytes = 0;
        for (const auto &image : batch)
        {
            outputBytes += image.total() * image.elemSize();
            REQUIRE(cv::countNonZero(image != expected) == 0, "Every reused-mask output must match pixel-center gold");
        }
        std::cout << "[METRICS] 8-image rotated crop: " << elapsed << " ms, masks=" << testing::maskBuildCount
                  << ", output_bytes=" << outputBytes << std::endl;

        std::vector<cv::Mat> thirtyTwo;
        for (int i = 0; i < 32; ++i) thirtyTwo.push_back((gradient(40, 40)).clone());
        inputs.insert_or_assign(crop_ports::in::IMAGES, DataRef<Image>(thirtyTwo));
        testing::maskBuildCount = 0;
        const auto started32    = std::chrono::steady_clock::now();
        result                  = tool.execute(inputs);
        const auto elapsed32
            = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started32).count();
        requireSuccess(result, "32-image rotated crop must succeed");
        REQUIRE(testing::maskBuildCount == 1, "Same bounds must build exactly one mask for 32 images");
        const auto &largeBatch = std::get<DataRef<Image>>(result.outputs.at(crop_ports::out::IMAGES)).get();
        REQUIRE(largeBatch.size() == 32, "Reused-mask output batch size must be preserved");
        for (const auto &image : largeBatch)
            REQUIRE(image.size() == expected.size() && cv::countNonZero(image != expected) == 0,
                    "Every 32-image reused-mask output must match independent gold");
        std::cout << "[METRICS] 32-image rotated crop: " << elapsed32 << " ms, masks=" << testing::maskBuildCount
                  << std::endl;

        // Same bounds in a new execute must rebuild instead of using a global cache.
        testing::maskBuildCount = 0;
        result                  = tool.execute(inputs);
        requireSuccess(result, "Repeated execute must succeed");
        REQUIRE(testing::maskBuildCount == 1, "Mask cache must not survive the execute call");

        std::vector<cv::Mat> mixed{(gradient(20, 20)).clone(), (gradient(30, 30)).clone()};
        inputs.insert_or_assign(crop_ports::in::IMAGES, DataRef<Image>(mixed));
        testing::maskBuildCount = 0;
        result                  = tool.execute(inputs);
        requireSuccess(result, "Mixed-size crop must succeed");
        REQUIRE(testing::maskBuildCount == 2, "Different bounds must build independent masks");
    }

    std::cout << "[PASS] testImageCropTool passed." << std::endl;
}
