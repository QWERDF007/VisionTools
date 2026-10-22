#include "src/preprocess/AdaptiveBinaryTool.h"
#include "src/preprocess/DetectionFilterTool.h"
#include "src/preprocess/ImageCropTool.h"
#include "src/preprocess/ImagePreprocessTool.h"
#include "src/preprocess/ReadImageTool.h"
#include "src/preprocess/RoiProjectionTool.h"

#include "PortConstants.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <visionrt/contracts/Module.h>
#include <visionrt/contracts/Tool.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

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

using namespace visionrt;
using namespace visiontools::preprocess;

namespace ab = visiontools::ports::visiontools_preprocess_adaptive_binary;
namespace crop_ports = visiontools::ports::visionrt_image_crop;
namespace filter_ports = visiontools::ports::visionrt_detection_filter;
namespace preprocess_ports = visiontools::ports::visionrt_image_preprocess;
namespace projection_ports = visiontools::ports::visionrt_roi_projection;
namespace read_ports = visiontools::ports::visionrt_image_read;

static void requireSuccess(const ToolResult &result, const char *context)
{
    REQUIRE(result.status.code == 0, context);
    REQUIRE(result.status.message == "运行成功", "Successful execute must use the shared success message");
}

static void requireFailure(const ToolResult &result, std::string_view expected, const char *context)
{
    REQUIRE(result.status.code != 0, context);
    REQUIRE(result.status.message.find(expected) != std::string::npos,
            std::string(context) + ": " + result.status.message);
}

static cv::Mat gradient(int rows, int cols)
{
    cv::Mat image(rows, cols, CV_8UC1);
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            image.at<unsigned char>(y, x) = static_cast<unsigned char>((x * 3 + y * 2) % 256);
    return image;
}

void testAdaptiveBinaryTool()
{
    std::cout << "[TEST] Running testAdaptiveBinaryTool..." << std::endl;
    ToolServices services;
    AdaptiveBinaryTool tool(services);

    static_assert(std::string_view(ab::in::IMAGES) == "images");
    static_assert(std::string_view(ab::in::BLOCK_SIZE) == "block_size");
    static_assert(std::string_view(ab::in::CONSTANT_OFFSET) == "constant_offset");
    static_assert(std::string_view(ab::out::IMAGES) == "images");

    // 1. Missing input
    {
        auto res = tool.execute({});
        requireFailure(res, "images", "Missing input 'images' must fail");
        REQUIRE(res.status.message.find("缺失") != std::string::npos, "Error must be Chinese");
        REQUIRE(isMissing(res.outputs["images"]), "Output must be Missing");
    }

    // 2. Non-ImageBatch input
    {
        Values inputs;
        inputs.emplace(ab::in::IMAGES, std::int64_t(42));
        auto res = tool.execute(inputs);
        requireFailure(res, "ImageBatch", "Non-ImageBatch must fail");
        REQUIRE(res.status.message.find("images") != std::string::npos, "Error must mention port 'images'");
    }

    // 3. B=0 -> success with empty batch
    {
        Values inputs;
        inputs.emplace(ab::in::IMAGES, ImageBatch{});
        auto res = tool.execute(inputs);
        requireSuccess(res, "B=0 must succeed");
        const auto *outBatch = std::get_if<ImageBatch>(&res.outputs["images"]);
        REQUIRE(outBatch != nullptr, "Output must be ImageBatch");
        REQUIRE(outBatch->images().empty(), "Output images must be empty");
    }

    // 4. Missing values for a non-empty batch
    {
        Values inputs;
        inputs.emplace(ab::in::IMAGES, ImageBatch({OwnedImage::copyOf(gradient(32, 32))}));
        auto res = tool.execute(inputs);
        requireFailure(res, "block_size", "Missing block_size must fail");
        REQUIRE(res.status.message.find("缺失") != std::string::npos, "Missing value must be Chinese");
    }

    // 5. block_size even -> rejected
    {
        Values inputs;
        inputs.emplace(ab::in::IMAGES, ImageBatch({OwnedImage::copyOf(gradient(32, 32))}));
        inputs.emplace(ab::in::BLOCK_SIZE, std::int64_t(4));
        inputs.emplace(ab::in::CONSTANT_OFFSET, 2.0);
        auto res = tool.execute(inputs);
        requireFailure(res, "奇数", "Even block_size must be rejected");
        REQUIRE(res.status.message.find("4") != std::string::npos, "Error must include the actual value");
    }

    // 6. block_size outside [3, 999]
    for (const std::int64_t value : {std::int64_t(1), std::int64_t(1000), std::int64_t(1) << 40})
    {
        Values inputs;
        inputs.emplace(ab::in::IMAGES, ImageBatch({OwnedImage::copyOf(gradient(32, 32))}));
        inputs.emplace(ab::in::BLOCK_SIZE, value);
        inputs.emplace(ab::in::CONSTANT_OFFSET, 2.0);
        auto res = tool.execute(inputs);
        requireFailure(res, "block_size", "block_size outside the declared range must fail");
        REQUIRE(res.status.message.find("[3, 999]") != std::string::npos,
                "Range failure must state the allowed range");
    }

    // 7. Strict float check: Int constant_offset rejected
    {
        Values inputs;
        inputs.emplace(ab::in::IMAGES, ImageBatch({OwnedImage::copyOf(gradient(32, 32))}));
        inputs.emplace(ab::in::BLOCK_SIZE, std::int64_t(5));
        inputs.emplace(ab::in::CONSTANT_OFFSET, std::int64_t(5));
        auto res = tool.execute(inputs);
        requireFailure(res, "浮点", "Int constant_offset must be strictly rejected");
        REQUIRE(res.status.message.find("constant_offset") != std::string::npos, "Error must mention constant_offset");
    }

    // 8. Non-finite constant_offset rejected
    {
        Values inputs;
        inputs.emplace(ab::in::IMAGES, ImageBatch({OwnedImage::copyOf(gradient(32, 32))}));
        inputs.emplace(ab::in::BLOCK_SIZE, std::int64_t(5));
        inputs.emplace(ab::in::CONSTANT_OFFSET, std::numeric_limits<double>::infinity());
        auto res = tool.execute(inputs);
        requireFailure(res, "有限", "Infinite constant_offset must be rejected");
    }

    // 9. Non-CV_8UC1 -> error points out batch index
    {
        cv::Mat bgr(64, 64, CV_8UC3, cv::Scalar(100, 150, 200));
        Values inputs;
        inputs.emplace(ab::in::IMAGES, ImageBatch({OwnedImage::copyOf(bgr)}));
        inputs.emplace(ab::in::BLOCK_SIZE, std::int64_t(5));
        inputs.emplace(ab::in::CONSTANT_OFFSET, 2.0);
        auto res = tool.execute(inputs);
        requireFailure(res, "批次索引 0", "BGR image must be rejected without auto-conversion");
        REQUIRE(res.status.message.find("CV_8UC1") != std::string::npos, "Error must state the expected type");
    }

    // 10. Synthetic grayscale comparison against OpenCV reference
    {
        cv::Mat img1 = gradient(100, 120);
        cv::Mat img2(70, 90, CV_8UC1);
        for (int y = 0; y < img2.rows; ++y)
            for (int x = 0; x < img2.cols; ++x)
                img2.at<unsigned char>(y, x) = static_cast<unsigned char>((x * 7 + y * 5) % 256);

        Values inputs;
        inputs.emplace(ab::in::IMAGES, ImageBatch({OwnedImage::copyOf(img1), OwnedImage::copyOf(img2)}));
        inputs.emplace(ab::in::BLOCK_SIZE, std::int64_t(11));
        inputs.emplace(ab::in::CONSTANT_OFFSET, 3.5);
        auto res = tool.execute(inputs);
        requireSuccess(res, "Valid adaptive binary execution must succeed");
        const auto *outBatch = std::get_if<ImageBatch>(&res.outputs["images"]);
        REQUIRE(outBatch != nullptr && outBatch->images().size() == 2, "Batch size must be 2");

        cv::Mat ref1;
        cv::adaptiveThreshold(img1, ref1, 255.0, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 11, 3.5);
        REQUIRE(cv::countNonZero(ref1 != outBatch->images()[0].readOnlyMat()) == 0,
                "Pixel-for-pixel match required for img1");
        cv::Mat ref2;
        cv::adaptiveThreshold(img2, ref2, 255.0, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 11, 3.5);
        REQUIRE(cv::countNonZero(ref2 != outBatch->images()[1].readOnlyMat()) == 0,
                "Pixel-for-pixel match required for img2");
    }

    std::cout << "[PASS] testAdaptiveBinaryTool passed." << std::endl;
}

static Values projectionInputs()
{
    Values inputs;
    inputs.emplace(projection_ports::in::AXIS, Utf8("x"));
    inputs.emplace(projection_ports::in::POLARITY, Utf8("bright"));
    inputs.emplace(projection_ports::in::THRESHOLD, std::int64_t(128));
    inputs.emplace(projection_ports::in::MIN_PIXELS, std::int64_t(1));
    inputs.emplace(projection_ports::in::MIN_EXTENT, std::int64_t(1));
    inputs.emplace(projection_ports::in::MERGE_GAP, std::int64_t(0));
    inputs.emplace(projection_ports::in::PADDING, std::int64_t(0));
    return inputs;
}

void testRoiProjectionTool()
{
    std::cout << "[TEST] Running testRoiProjectionTool..." << std::endl;
    ToolServices services;
    RoiProjectionTool tool(services);

    static_assert(std::string_view(projection_ports::in::AXIS) == "axis");
    static_assert(std::string_view(projection_ports::in::MIN_PIXELS) == "min_pixels");
    static_assert(std::string_view(projection_ports::in::MERGE_GAP) == "merge_gap");
    static_assert(std::string_view(projection_ports::out::ROIS) == "rois");

    cv::Mat img(100, 100, CV_8UC1, cv::Scalar(0));
    img(cv::Rect(30, 30, 40, 40)).setTo(cv::Scalar(255));

    // Missing / invalid values
    {
        Values inputs = projectionInputs();
        inputs.emplace(projection_ports::in::IMAGES, ImageBatch({OwnedImage::copyOf(img)}));
        inputs.erase(projection_ports::in::AXIS);
        auto res = tool.execute(inputs);
        requireFailure(res, "axis", "Missing axis must fail");
        REQUIRE(res.status.message.find("缺失") != std::string::npos, "Missing value must be Chinese");
    }
    {
        Values inputs = projectionInputs();
        inputs.emplace(projection_ports::in::IMAGES, ImageBatch({OwnedImage::copyOf(img)}));
        inputs.insert_or_assign(projection_ports::in::AXIS, Utf8("z"));
        auto res = tool.execute(inputs);
        requireFailure(res, "axis", "Invalid axis must fail");
        REQUIRE(res.status.message.find("无效") != std::string::npos, "Invalid option must be Chinese");
    }
    {
        Values inputs = projectionInputs();
        inputs.emplace(projection_ports::in::IMAGES, ImageBatch({OwnedImage::copyOf(img)}));
        inputs.insert_or_assign(projection_ports::in::POLARITY, Utf8("mixed"));
        auto res = tool.execute(inputs);
        requireFailure(res, "polarity", "Invalid polarity must fail");
    }
    {
        Values inputs = projectionInputs();
        inputs.emplace(projection_ports::in::IMAGES, ImageBatch({OwnedImage::copyOf(img)}));
        inputs.insert_or_assign(projection_ports::in::THRESHOLD, std::int64_t(300));
        auto res = tool.execute(inputs);
        requireFailure(res, "threshold", "Out-of-range threshold must fail");
        REQUIRE(res.status.message.find("[0, 255]") != std::string::npos, "Range must be stated");
    }
    for (const auto &[key, value] : std::vector<std::pair<const char *, std::int64_t>>{
             {projection_ports::in::MIN_PIXELS, 0}, {projection_ports::in::MIN_EXTENT, 0},
             {projection_ports::in::MERGE_GAP, -1}, {projection_ports::in::PADDING, -1}})
    {
        Values inputs = projectionInputs();
        inputs.emplace(projection_ports::in::IMAGES, ImageBatch({OwnedImage::copyOf(img)}));
        inputs.insert_or_assign(key, value);
        auto res = tool.execute(inputs);
        requireFailure(res, key, "Out-of-range projection parameter must fail");
    }

    // Valid projection
    {
        Values inputs = projectionInputs();
        inputs.emplace(projection_ports::in::IMAGES, ImageBatch({OwnedImage::copyOf(img)}));
        inputs.insert_or_assign(projection_ports::in::MIN_PIXELS, std::int64_t(10));
        inputs.insert_or_assign(projection_ports::in::MIN_EXTENT, std::int64_t(10));
        auto res = tool.execute(inputs);
        requireSuccess(res, "Projection must succeed");
        const auto *rois = std::get_if<RoiBatch>(&res.outputs["rois"]);
        REQUIRE(rois != nullptr && rois->images().size() == 1 && rois->images()[0].size() == 1,
                "Expected 1 projected ROI");
        const auto encoded = rois->images()[0][0].encoded();
        REQUIRE(encoded[0] == 2.0F, "ROI must be rectangle");
        REQUIRE(std::abs(encoded[1] - 30.0F) < 1e-3, "ROI left must be 30");
        REQUIRE(std::abs(encoded[3] - 40.0F) < 1e-3, "ROI width must be 40");
    }

    // Empty batch
    {
        Values inputs = projectionInputs();
        inputs.emplace(projection_ports::in::IMAGES, ImageBatch{});
        auto res = tool.execute(inputs);
        requireSuccess(res, "Empty batch must succeed");
        REQUIRE(std::get<ImageBatch>(res.outputs["images"]).images().empty(), "Empty images must stay empty");
        REQUIRE(std::get<RoiBatch>(res.outputs["rois"]).images().empty(), "Empty rois must stay empty");
    }

    std::cout << "[PASS] testRoiProjectionTool passed." << std::endl;
}

namespace {

bool insidePolygon(std::span<const cv::Point2d> vertices, double x, double y)
{
    bool inside = false;
    for (std::size_t i = 0, previous = vertices.size() - 1; i < vertices.size(); previous = i++)
    {
        const auto &a = vertices[previous];
        const auto &b = vertices[i];
        const double cross = (x - a.x) * (b.y - a.y) - (y - a.y) * (b.x - a.x);
        const double tolerance = 32 * std::numeric_limits<double>::epsilon()
                               * std::max(1.0, std::abs((x - a.x) * (b.y - a.y)) + std::abs((y - a.y) * (b.x - a.x)));
        if (std::abs(cross) <= tolerance && x >= std::min(a.x, b.x) && x <= std::max(a.x, b.x)
            && y >= std::min(a.y, b.y) && y <= std::max(a.y, b.y))
            return true;
        if ((a.y > y) != (b.y > y) && x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x)
            inside = !inside;
    }
    return inside;
}

cv::Mat referenceCrop(const cv::Mat &image, const Roi &rectangle)
{
    const auto corners = rectangle.rectangleCorners();
    double minimumX = corners[0].x, maximumX = corners[0].x;
    double minimumY = corners[0].y, maximumY = corners[0].y;
    for (const auto &corner : corners)
    {
        minimumX = std::min(minimumX, corner.x);
        minimumY = std::min(minimumY, corner.y);
        maximumX = std::max(maximumX, corner.x);
        maximumY = std::max(maximumY, corner.y);
    }
    const auto snap = [](double value)
    {
        const double nearest = std::round(value);
        return std::abs(value - nearest) <= 32 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(value))
                   ? nearest
                   : value;
    };
    const int left = static_cast<int>(std::clamp(std::floor(snap(minimumX)), 0.0, static_cast<double>(image.cols)));
    const int top = static_cast<int>(std::clamp(std::floor(snap(minimumY)), 0.0, static_cast<double>(image.rows)));
    const int right = static_cast<int>(std::clamp(std::ceil(snap(maximumX)), 0.0, static_cast<double>(image.cols)));
    const int bottom = static_cast<int>(std::clamp(std::ceil(snap(maximumY)), 0.0, static_cast<double>(image.rows)));
    cv::Mat expected = cv::Mat::zeros(bottom - top, right - left, image.type());
    for (int row = 0; row < expected.rows; ++row)
        for (int column = 0; column < expected.cols; ++column)
            if (insidePolygon(corners, left + column + 0.5, top + row + 0.5))
                expected.at<unsigned char>(row, column) = image.at<unsigned char>(top + row, left + column);
    return expected;
}

} // namespace

void testImageCropTool()
{
    std::cout << "[TEST] Running testImageCropTool..." << std::endl;
    ToolServices services;
    ImageCropTool tool(services);

    static_assert(std::string_view(crop_ports::in::IMAGES) == "images");
    static_assert(std::string_view(crop_ports::in::RECT) == "rect");
    static_assert(std::string_view(crop_ports::out::ORIGINS) == "origins");

    // 1. Integer rectangle: zero-copy view and input owner release
    OwnedImage savedCropped;
    {
        cv::Mat img(100, 100, CV_8UC1, cv::Scalar(100));
        img(cv::Rect(20, 20, 30, 30)).setTo(cv::Scalar(200));
        OwnedImage srcImage = OwnedImage::copyOf(img);
        auto roi = Roi::rectangle(20.0F, 20.0F, 30.0F, 30.0F, 0.0F);
        Values inputs;
        inputs.emplace(crop_ports::in::IMAGES, ImageBatch({srcImage}));
        inputs.emplace(crop_ports::in::RECT, roi);
        auto res = tool.execute(inputs);
        requireSuccess(res, "Crop must succeed");
        const auto *cropped = std::get_if<ImageBatch>(&res.outputs[crop_ports::out::IMAGES]);
        REQUIRE(cropped != nullptr && cropped->images().size() == 1, "Expected 1 cropped image");
        REQUIRE(cropped->images()[0].cols() == 30 && cropped->images()[0].rows() == 30, "Cropped size must be 30x30");
        REQUIRE(cropped->images()[0].readOnlyMat().u == srcImage.readOnlyMat().u,
                "Rect crop must share pixel buffer (u) with input");
        REQUIRE(cropped->images()[0].rowStride() == srcImage.rowStride(), "Rect crop must preserve rowStride");
        savedCropped = cropped->images()[0];
    }
    REQUIRE(savedCropped.readOnlyMat().at<unsigned char>(0, 0) == 200,
            "Cropped image must remain readable after input scope exit");

    // 2. Fractional ROI on non-uniform image
    {
        cv::Mat nonUniform(10, 10, CV_8UC1);
        for (int r = 0; r < 10; ++r)
            for (int c = 0; c < 10; ++c)
                nonUniform.at<unsigned char>(r, c) = static_cast<unsigned char>(r * 10 + c + 1);
        auto roi = Roi::rectangle(0.75F, 0.0F, 1.0F, 1.0F, 0.0F);
        Values inputs;
        inputs.emplace(crop_ports::in::IMAGES, ImageBatch({OwnedImage::copyOf(nonUniform)}));
        inputs.emplace(crop_ports::in::RECT, roi);
        auto res = tool.execute(inputs);
        requireSuccess(res, "Fractional crop must succeed via mask path");
        const auto *cropped = std::get_if<ImageBatch>(&res.outputs[crop_ports::out::IMAGES]);
        REQUIRE(cropped && cropped->images().size() == 1, "Expected 1 output image");
        const auto &outMat = cropped->images()[0].readOnlyMat();
        REQUIRE(outMat.cols == 2 && outMat.rows == 1, "Bounds width must be 2, height 1");
        REQUIRE(outMat.at<unsigned char>(0, 0) == 0, "Col 0 center is outside and must be 0");
        REQUIRE(outMat.at<unsigned char>(0, 1) == nonUniform.at<unsigned char>(0, 1), "Col 1 center must be preserved");
    }

    // 3. Narrow fractional ROI has no valid pixel center
    {
        cv::Mat img(10, 10, CV_8UC1, cv::Scalar(100));
        Values inputs;
        inputs.emplace(crop_ports::in::IMAGES, ImageBatch({OwnedImage::copyOf(img)}));
        inputs.emplace(crop_ports::in::RECT, Roi::rectangle(0.1F, 0.1F, 0.1F, 0.1F, 0.0F));
        auto res = tool.execute(inputs);
        requireFailure(res, "没有有效像素中心", "Narrow ROI with no pixel centers must fail");
        REQUIRE(res.status.message.find("图像索引 0") != std::string::npos, "Error must name the image index");
    }

    // 4. Batch order, clipping and origins
    {
        const auto first = OwnedImage::copyOf(cv::Mat(8, 10, CV_8UC1, cv::Scalar(31)));
        const auto second = OwnedImage::copyOf(cv::Mat(12, 14, CV_8UC1, cv::Scalar(79)));
        Values inputs{{crop_ports::in::IMAGES, ImageBatch({first, second})},
                      {crop_ports::in::RECT, Roi::rectangle(-2, 1, 6, 3, 0)}};
        auto result = tool.execute(inputs);
        requireSuccess(result, "One rectangle must crop every image in batch order");
        const auto &batch = std::get<ImageBatch>(result.outputs.at(crop_ports::out::IMAGES)).images();
        REQUIRE(batch.size() == 2, "Crop must preserve batch cardinality");
        REQUIRE(batch[0].cols() == 4 && batch[1].rows() == 3, "Crop must clip partial bounds");
        REQUIRE(batch[0].readOnlyMat().at<unsigned char>(0, 0) == 31, "First image order must be retained");
        REQUIRE(batch[1].readOnlyMat().at<unsigned char>(0, 0) == 79, "Second image order must be retained");
        REQUIRE(result.outputs.count("source_indices") == 0, "Scalar rectangle crop has no expanded ROI indices");
        const auto &origin = std::get<NumericArray>(result.outputs.at(crop_ports::out::ORIGINS)).buffers()[0];
        REQUIRE(origin.shape().size() == 2 && origin.shape()[0] == 2 && origin.shape()[1] == 2,
                "Origins must have B by 2 shape");
        const auto coordinates = origin.values<double>();
        REQUIRE(coordinates[0] == 0 && coordinates[1] == 1 && coordinates[2] == 0 && coordinates[3] == 1,
                "Each origin must correspond to the same-index crop");
    }

    // 5. Invalid inputs never publish partial outputs
    {
        Values inputs{{crop_ports::in::IMAGES, ImageBatch({OwnedImage::copyOf(cv::Mat(4, 4, CV_8UC1, cv::Scalar(8)))})}};
        for (const PortValue &invalid : std::vector<PortValue>{Missing{}, Roi::circle(1, 1, 1),
                                                               RoiBatch({{Roi::rectangle(0, 0, 2, 2, 0)}}),
                                                               Roi::rectangle(10, 10, 2, 2, 0)})
        {
            inputs.insert_or_assign(crop_ports::in::RECT, invalid);
            auto result = tool.execute(inputs);
            REQUIRE(!result.status, "Missing, non-rectangle, batch and disjoint inputs must fail");
            REQUIRE(isMissing(result.outputs.at(crop_ports::out::IMAGES))
                        && isMissing(result.outputs.at(crop_ports::out::ORIGINS)),
                    "Failure must not publish partial outputs");
        }
        inputs.insert_or_assign(crop_ports::in::IMAGES, ImageBatch{});
        inputs.insert_or_assign(crop_ports::in::RECT, Roi::rectangle(0, 0, 2, 2, 0));
        auto empty = tool.execute(inputs);
        requireSuccess(empty, "A valid empty image batch must yield an empty crop batch");
        REQUIRE(std::get<ImageBatch>(empty.outputs.at(crop_ports::out::IMAGES)).images().empty(),
                "Empty batch must produce an empty crop batch");
        REQUIRE(std::get<NumericArray>(empty.outputs.at(crop_ports::out::ORIGINS)).buffers()[0].shape()[0] == 0,
                "An empty batch must have zero origin rows");
        inputs.insert_or_assign(crop_ports::in::IMAGES,
                                ImageBatch({OwnedImage::copyOf(cv::Mat(4, 4, CV_8UC1, cv::Scalar(8))),
                                            OwnedImage::copyOf(cv::Mat(1, 1, CV_8UC1, cv::Scalar(9)))}));
        inputs.insert_or_assign(crop_ports::in::RECT, Roi::rectangle(2, 2, 2, 2, 45));
        auto partial = tool.execute(inputs);
        REQUIRE(!partial.status && isMissing(partial.outputs.at(crop_ports::out::IMAGES)),
                "A later disjoint image must fail the whole batch without publishing earlier crops");
    }

    // 6. Rotated/subpixel golden pixels and origins
    {
        cv::Mat image = gradient(24, 30);
        const auto rectangle = Roi::rectangle(2.3F, 3.7F, 12.4F, 9.2F, 30.0F);
        Values inputs{{crop_ports::in::IMAGES, ImageBatch({OwnedImage::copyOf(image)})},
                      {crop_ports::in::RECT, rectangle}};
        auto result = tool.execute(inputs);
        requireSuccess(result, "Rotated crop must succeed");
        const auto &cropped = std::get<ImageBatch>(result.outputs.at(crop_ports::out::IMAGES)).images()[0];
        const auto expected = referenceCrop(image, rectangle);
        REQUIRE(cropped.rows() == expected.rows && cropped.cols() == expected.cols, "Rotated bounds must match gold");
        REQUIRE(cv::countNonZero(cropped.readOnlyMat() != expected) == 0,
                "Rotated crop must match pixel-center gold");
        const auto coordinates = std::get<NumericArray>(result.outputs.at(crop_ports::out::ORIGINS)).buffers()[0].values<double>();
        const auto corners = rectangle.rectangleCorners();
        double minimumX = corners[0].x, minimumY = corners[0].y;
        for (const auto &corner : corners)
        {
            minimumX = std::min(minimumX, corner.x);
            minimumY = std::min(minimumY, corner.y);
        }
        const auto snap = [](double value)
        {
            const double nearest = std::round(value);
            return std::abs(value - nearest) <= 32 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(value))
                       ? nearest
                       : value;
        };
        REQUIRE(coordinates[0] == std::floor(snap(minimumX)) && coordinates[1] == std::floor(snap(minimumY)),
                "Rotated origin must be the snapped top-left");
    }

    // 7. Mask reuse within one execute, no cache across executes
    {
        const auto rectangle = Roi::rectangle(10.0F, 10.0F, 15.0F, 15.0F, 15.0F);
        {
            Values single{{crop_ports::in::IMAGES, ImageBatch({OwnedImage::copyOf(gradient(40, 40))})},
                          {crop_ports::in::RECT, rectangle}};
            testing::maskBuildCount = 0;
            auto result = tool.execute(single);
            requireSuccess(result, "1-image rotated crop must succeed");
            REQUIRE(testing::maskBuildCount == 1, "One image must build exactly one mask");
        }
        std::vector<OwnedImage> sameBounds;
        for (int i = 0; i < 8; ++i)
            sameBounds.push_back(OwnedImage::copyOf(gradient(40, 40)));
        Values inputs{{crop_ports::in::IMAGES, ImageBatch(sameBounds)}, {crop_ports::in::RECT, rectangle}};
        testing::maskBuildCount = 0;
        const auto started = std::chrono::steady_clock::now();
        auto result = tool.execute(inputs);
        const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        requireSuccess(result, "8-image rotated crop must succeed");
        REQUIRE(testing::maskBuildCount == 1, "Same bounds must build exactly one mask for 8 images");
        const auto &batch = std::get<ImageBatch>(result.outputs.at(crop_ports::out::IMAGES)).images();
        const auto expected = referenceCrop(gradient(40, 40), rectangle);
        std::size_t outputBytes = 0;
        for (const auto &image : batch)
        {
            outputBytes += image.readOnlyMat().total() * image.readOnlyMat().elemSize();
            REQUIRE(cv::countNonZero(image.readOnlyMat() != expected) == 0,
                    "Every reused-mask output must match pixel-center gold");
        }
        std::cout << "[METRICS] 8-image rotated crop: " << elapsed << " ms, masks=" << testing::maskBuildCount
                  << ", output_bytes=" << outputBytes << std::endl;

        std::vector<OwnedImage> thirtyTwo;
        for (int i = 0; i < 32; ++i)
            thirtyTwo.push_back(OwnedImage::copyOf(gradient(40, 40)));
        inputs.insert_or_assign(crop_ports::in::IMAGES, ImageBatch(thirtyTwo));
        testing::maskBuildCount = 0;
        const auto started32 = std::chrono::steady_clock::now();
        result = tool.execute(inputs);
        const auto elapsed32 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started32).count();
        requireSuccess(result, "32-image rotated crop must succeed");
        REQUIRE(testing::maskBuildCount == 1, "Same bounds must build exactly one mask for 32 images");
        std::cout << "[METRICS] 32-image rotated crop: " << elapsed32 << " ms, masks=" << testing::maskBuildCount
                  << std::endl;

        // Same bounds in a new execute must rebuild instead of using a global cache.
        testing::maskBuildCount = 0;
        result = tool.execute(inputs);
        requireSuccess(result, "Repeated execute must succeed");
        REQUIRE(testing::maskBuildCount == 1, "Mask cache must not survive the execute call");

        std::vector<OwnedImage> mixed{OwnedImage::copyOf(gradient(20, 20)), OwnedImage::copyOf(gradient(30, 30))};
        inputs.insert_or_assign(crop_ports::in::IMAGES, ImageBatch(mixed));
        testing::maskBuildCount = 0;
        result = tool.execute(inputs);
        requireSuccess(result, "Mixed-size crop must succeed");
        REQUIRE(testing::maskBuildCount == 2, "Different bounds must build independent masks");
    }

    std::cout << "[PASS] testImageCropTool passed." << std::endl;
}

namespace {

visionrt::NumericArray numericArray(DType dtype, std::vector<std::int64_t> shape, const auto &values)
{
    return NumericArray(dtype, 2, Layout::BatchRagged,
                        {NumericBuffer::copy(dtype, std::move(shape), std::as_bytes(std::span(values)))});
}

visionrt::NumericArray intArray(std::vector<std::int64_t> values)
{
    return NumericArray(DType::Int64, 1, Layout::Dense,
                        {NumericBuffer::copy(DType::Int64, {static_cast<std::int64_t>(values.size())},
                                             std::as_bytes(std::span(values)))});
}

Values detectionInputs()
{
    const auto roi = Roi::rectangle(10.0F, 10.0F, 20.0F, 20.0F, 0.0F);
    std::vector<float> scores = {0.85F, 0.35F};
    std::vector<std::int64_t> classes = {1, 2};
    std::vector<std::string> names = {"cat", "dog"};
    Values inputs;
    inputs.emplace(filter_ports::in::IMAGES, ImageBatch({OwnedImage::copyOf(gradient(100, 100))}));
    inputs.emplace(filter_ports::in::ROIS, RoiBatch({{roi, roi}}));
    inputs.emplace(filter_ports::in::SCORES, numericArray(DType::Float32, {2}, scores));
    inputs.emplace(filter_ports::in::CLASSES, numericArray(DType::Int64, {2}, classes));
    inputs.emplace(filter_ports::in::CLASS_NAMES, StringArray(2, Layout::BatchRagged, {StringBuffer({2}, names)}));
    inputs.emplace(filter_ports::in::ALLOWED_CLASSES, intArray({1, 2}));
    inputs.emplace(filter_ports::in::MIN_SCORE, 0.5);
    inputs.emplace(filter_ports::in::MIN_AREA, 0.0);
    return inputs;
}

} // namespace

void testDetectionFilterTool()
{
    std::cout << "[TEST] Running testDetectionFilterTool..." << std::endl;
    ToolServices services;
    DetectionFilterTool tool(services);

    static_assert(std::string_view(filter_ports::in::MIN_SCORE) == "min_score");
    static_assert(std::string_view(filter_ports::in::ALLOWED_CLASSES) == "allowed_classes");
    static_assert(std::string_view(filter_ports::out::COUNTS) == "counts");

    {
        auto inputs = detectionInputs();
        inputs.insert_or_assign(filter_ports::in::MIN_SCORE, Utf8("not_a_number"));
        auto res = tool.execute(inputs);
        requireFailure(res, "浮点", "String min_score must be rejected");
        REQUIRE(res.status.message.find("min_score") != std::string::npos, "Error must mention min_score");
    }
    {
        auto inputs = detectionInputs();
        inputs.insert_or_assign(filter_ports::in::MIN_SCORE, std::int64_t(0));
        auto res = tool.execute(inputs);
        requireFailure(res, "浮点", "Int min_score must be rejected");
    }
    {
        auto inputs = detectionInputs();
        inputs.erase(filter_ports::in::MIN_SCORE);
        auto res = tool.execute(inputs);
        requireFailure(res, "min_score", "Missing min_score must be rejected");
        REQUIRE(res.status.message.find("缺失") != std::string::npos, "Missing value must be Chinese");
    }
    {
        auto inputs = detectionInputs();
        inputs.insert_or_assign(filter_ports::in::MIN_SCORE, 1.5);
        auto res = tool.execute(inputs);
        requireFailure(res, "min_score", "Out-of-range min_score must be rejected");
        REQUIRE(res.status.message.find("[0") != std::string::npos, "Range must be stated");
    }
    {
        auto inputs = detectionInputs();
        inputs.insert_or_assign(filter_ports::in::MIN_AREA, -1.0);
        auto res = tool.execute(inputs);
        requireFailure(res, "min_area", "Negative min_area must be rejected");
        REQUIRE(res.status.message.find("非负") != std::string::npos, "Error must state non-negative");
    }
    {
        auto inputs = detectionInputs();
        auto res = tool.execute(inputs);
        requireSuccess(res, "Filter must succeed");
        const auto *rois = std::get_if<RoiBatch>(&res.outputs[filter_ports::out::ROIS]);
        REQUIRE(rois != nullptr && rois->images()[0].size() == 1, "Only score >= 0.5 passes");
        const auto counts = std::get<NumericArray>(res.outputs[filter_ports::out::COUNTS]).buffers()[0].values<std::int64_t>();
        REQUIRE(counts[0] == 1, "Counts must report the filtered count");
    }
    {
        auto inputs = detectionInputs();
        inputs.insert_or_assign(filter_ports::in::ALLOWED_CLASSES, intArray({1}));
        auto res = tool.execute(inputs);
        requireSuccess(res, "Allowed-class filter must succeed");
        const auto *rois = std::get_if<RoiBatch>(&res.outputs[filter_ports::out::ROIS]);
        REQUIRE(rois->images()[0].size() == 1, "Only the allowed class passes");
        const auto names = std::get<StringArray>(res.outputs[filter_ports::out::CLASS_NAMES]).buffers()[0].values();
        REQUIRE(names[0] == "cat", "Filtered class name must be retained");
    }
    {
        auto inputs = detectionInputs();
        std::vector<float> one = {0.5F};
        std::vector<std::int64_t> oneClass = {1};
        std::vector<std::string> oneName = {"cat"};
        inputs.insert_or_assign(filter_ports::in::SCORES, numericArray(DType::Float32, {1}, one));
        inputs.insert_or_assign(filter_ports::in::CLASSES, numericArray(DType::Int64, {1}, oneClass));
        inputs.insert_or_assign(filter_ports::in::CLASS_NAMES,
                                StringArray(2, Layout::BatchRagged, {StringBuffer({1}, oneName)}));
        auto res = tool.execute(inputs);
        requireFailure(res, "长度不一致", "Field length mismatch must fail");
        REQUIRE(res.status.message.find("批次索引 0") != std::string::npos, "Error must name the batch index");
    }
    {
        auto inputs = detectionInputs();
        const auto circle = Roi::circle(5, 5, 2);
        inputs.insert_or_assign(filter_ports::in::ROIS, RoiBatch({{circle, circle}}));
        auto res = tool.execute(inputs);
        requireFailure(res, "矩形", "Non-rectangle ROI must be rejected");
    }
    {
        auto inputs = detectionInputs();
        std::vector<std::int64_t> negative = {-1, 2};
        inputs.insert_or_assign(filter_ports::in::CLASSES, numericArray(DType::Int64, {2}, negative));
        auto res = tool.execute(inputs);
        requireFailure(res, "得分或类别", "Negative class must be rejected");
    }
    {
        Values inputs;
        inputs.emplace(filter_ports::in::IMAGES, ImageBatch{});
        inputs.emplace(filter_ports::in::ROIS, RoiBatch{});
        inputs.emplace(filter_ports::in::SCORES, NumericArray(DType::Float32, 2, Layout::BatchRagged, {}));
        inputs.emplace(filter_ports::in::CLASSES, NumericArray(DType::Int64, 2, Layout::BatchRagged, {}));
        inputs.emplace(filter_ports::in::CLASS_NAMES, StringArray(2, Layout::BatchRagged, {}));
        inputs.emplace(filter_ports::in::ALLOWED_CLASSES, NumericArray(DType::Int64, 1, Layout::Dense,
                                                                       {NumericBuffer::copy(DType::Int64, {0}, {})}));
        inputs.emplace(filter_ports::in::MIN_SCORE, 0.5);
        inputs.emplace(filter_ports::in::MIN_AREA, 0.0);
        auto res = tool.execute(inputs);
        requireSuccess(res, "Empty detection batch must succeed");
        REQUIRE(std::get<RoiBatch>(res.outputs[filter_ports::out::ROIS]).images().empty(),
                "Empty batch must produce empty rois");
    }

    std::cout << "[PASS] testDetectionFilterTool passed." << std::endl;
}

void testImagePreprocessTool()
{
    std::cout << "[TEST] Running testImagePreprocessTool..." << std::endl;
    ToolServices services;
    ImagePreprocessTool tool(services);

    static_assert(std::string_view(preprocess_ports::in::OPERATION) == "operation");
    static_assert(std::string_view(preprocess_ports::in::KERNEL) == "kernel");
    static_assert(std::string_view(preprocess_ports::in::THRESHOLD) == "threshold");
    static_assert(std::string_view(preprocess_ports::out::IMAGES) == "images");

    cv::Mat img(50, 50, CV_8UC1, cv::Scalar(150));
    OwnedImage owned = OwnedImage::copyOf(img);

    // 1. Identity pass-through
    {
        Values inputs;
        inputs.emplace(preprocess_ports::in::IMAGES, ImageBatch({owned}));
        inputs.emplace(preprocess_ports::in::OPERATION, Utf8("identity"));
        auto res = tool.execute(inputs);
        requireSuccess(res, "Identity must succeed");
        const auto *out = std::get_if<ImageBatch>(&res.outputs[preprocess_ports::out::IMAGES]);
        REQUIRE(out != nullptr && out->images()[0].data() == owned.data(), "Identity must not copy pixel buffer");
    }

    // 2. Threshold
    {
        Values inputs;
        inputs.emplace(preprocess_ports::in::IMAGES, ImageBatch({owned}));
        inputs.emplace(preprocess_ports::in::OPERATION, Utf8("threshold"));
        inputs.emplace(preprocess_ports::in::THRESHOLD, std::int64_t(100));
        auto res = tool.execute(inputs);
        requireSuccess(res, "Threshold must succeed");
        const auto *out = std::get_if<ImageBatch>(&res.outputs[preprocess_ports::out::IMAGES]);
        REQUIRE(out->images()[0].readOnlyMat().at<unsigned char>(0, 0) == 255, "150 >= 100 -> 255");
    }

    // 3. Gaussian
    {
        Values inputs;
        inputs.emplace(preprocess_ports::in::IMAGES, ImageBatch({owned}));
        inputs.emplace(preprocess_ports::in::OPERATION, Utf8("gaussian"));
        inputs.emplace(preprocess_ports::in::KERNEL, std::int64_t(3));
        auto res = tool.execute(inputs);
        requireSuccess(res, "Gaussian must succeed");
        const auto *out = std::get_if<ImageBatch>(&res.outputs[preprocess_ports::out::IMAGES]);
        REQUIRE(out->images()[0].cols() == 50, "Gaussian must preserve geometry");
    }

    // 4. Missing operation / invalid operation / odd kernel / range
    {
        Values inputs;
        inputs.emplace(preprocess_ports::in::IMAGES, ImageBatch({owned}));
        auto res = tool.execute(inputs);
        requireFailure(res, "operation", "Missing operation must fail");
        REQUIRE(res.status.message.find("缺失") != std::string::npos, "Missing value must be Chinese");
    }
    {
        Values inputs;
        inputs.emplace(preprocess_ports::in::IMAGES, ImageBatch({owned}));
        inputs.emplace(preprocess_ports::in::OPERATION, Utf8("unknown"));
        auto res = tool.execute(inputs);
        requireFailure(res, "operation", "Invalid operation must fail");
        REQUIRE(res.status.message.find("unknown") != std::string::npos, "Error must include the invalid value");
    }
    {
        Values inputs;
        inputs.emplace(preprocess_ports::in::IMAGES, ImageBatch({owned}));
        inputs.emplace(preprocess_ports::in::OPERATION, Utf8("gaussian"));
        inputs.emplace(preprocess_ports::in::KERNEL, std::int64_t(4));
        auto res = tool.execute(inputs);
        requireFailure(res, "奇数", "Even Gaussian kernel must fail");
    }
    {
        Values inputs;
        inputs.emplace(preprocess_ports::in::IMAGES, ImageBatch({owned}));
        inputs.emplace(preprocess_ports::in::OPERATION, Utf8("gaussian"));
        inputs.emplace(preprocess_ports::in::KERNEL, std::int64_t(1) << 40);
        auto res = tool.execute(inputs);
        requireFailure(res, "kernel", "Kernel outside int range must fail before narrowing");
    }
    {
        Values inputs;
        inputs.emplace(preprocess_ports::in::IMAGES, ImageBatch({owned}));
        inputs.emplace(preprocess_ports::in::OPERATION, Utf8("threshold"));
        inputs.emplace(preprocess_ports::in::THRESHOLD, std::int64_t(300));
        auto res = tool.execute(inputs);
        requireFailure(res, "threshold", "Out-of-range threshold must fail");
        REQUIRE(res.status.message.find("[0, 255]") != std::string::npos, "Range must be stated");
    }

    // 5. Invalid image type names the batch index
    {
        cv::Mat invalid(10, 10, CV_16UC1, cv::Scalar(1));
        Values inputs;
        inputs.emplace(preprocess_ports::in::IMAGES, ImageBatch({owned, OwnedImage::copyOf(invalid)}));
        inputs.emplace(preprocess_ports::in::OPERATION, Utf8("gray"));
        auto res = tool.execute(inputs);
        requireFailure(res, "批次索引 1", "Invalid image type must name the batch index");
    }

    // 6. Empty batch
    {
        Values inputs;
        inputs.emplace(preprocess_ports::in::IMAGES, ImageBatch{});
        inputs.emplace(preprocess_ports::in::OPERATION, Utf8("identity"));
        auto res = tool.execute(inputs);
        requireSuccess(res, "Empty batch must succeed");
        REQUIRE(std::get<ImageBatch>(res.outputs[preprocess_ports::out::IMAGES]).images().empty(),
                "Empty batch must stay empty");
    }

    std::cout << "[PASS] testImagePreprocessTool passed." << std::endl;
}

namespace {

std::string utf8(const std::filesystem::path &path)
{
    const auto value = path.generic_u8string();
    return {reinterpret_cast<const char *>(value.data()), value.size()};
}

Values readInputs(const std::filesystem::path &path)
{
    Values inputs;
    inputs.emplace(read_ports::in::MODE, Utf8("file"));
    inputs.emplace(read_ports::in::PATH, Utf8(utf8(path)));
    inputs.emplace(read_ports::in::COLOR, Utf8("bgr"));
    inputs.emplace(read_ports::in::LIMIT, std::int64_t(0));
    inputs.emplace(read_ports::in::MAX_BATCH_BYTES, std::int64_t(0));
    return inputs;
}

void writePng(const std::filesystem::path &path, const cv::Mat &image)
{
    std::vector<unsigned char> encoded;
    REQUIRE(cv::imencode(".png", image, encoded), "Image fixture must encode");
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char *>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
    REQUIRE(stream.good(), "Image fixture must be written");
}

} // namespace

void testReadImageTool()
{
    std::cout << "[TEST] Running testReadImageTool..." << std::endl;
    ToolServices services{PathResolver(std::filesystem::current_path())};
    ReadImageTool tool(services);

    static_assert(std::string_view(read_ports::in::MAX_BATCH_BYTES) == "max_batch_bytes");
    static_assert(std::string_view(read_ports::out::PATHS) == "paths");

    const auto directory = std::filesystem::current_path() / "test_read_directory";
    const auto files = directory / "files";
    const auto budgetDirectory = directory / "budget";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(files);
    std::filesystem::create_directories(budgetDirectory);

    // 1. Missing / empty / nonexistent path
    {
        Values inputs = readInputs("");
        auto res = tool.execute(inputs);
        requireFailure(res, "path", "Empty path must fail");
        REQUIRE(res.status.message.find("为空") != std::string::npos, "Empty path must be Chinese");
    }
    {
        Values inputs = readInputs(files / "missing.png");
        auto res = tool.execute(inputs);
        requireFailure(res, "文件不存在", "Non-existent path must fail");
        REQUIRE(res.status.message.find("missing.png") != std::string::npos, "Error must include the file path");
    }

    // 2. Valid read
    {
        const auto path = files / "valid.png";
        writePng(path, cv::Mat(24, 24, CV_8UC3, cv::Scalar(10, 20, 30)));
        Values inputs = readInputs(path);
        auto res = tool.execute(inputs);
        requireSuccess(res, "Valid image read must succeed");
        const auto *batch = std::get_if<ImageBatch>(&res.outputs[read_ports::out::IMAGES]);
        REQUIRE(batch && batch->images().size() == 1, "Must read 1 image");
        REQUIRE(batch->images()[0].cols() == 24 && batch->images()[0].rows() == 24, "Image dims must match 24x24");
        const auto paths = std::get<StringArray>(res.outputs[read_ports::out::PATHS]).buffers()[0].values();
        REQUIRE(paths[0].ends_with(utf8(path)), "Returned path must be the resolved file path");
    }

    // 3. Negative limit / budget
    {
        Values inputs = readInputs(files / "valid.png");
        inputs.insert_or_assign(read_ports::in::LIMIT, std::int64_t(-1));
        auto res = tool.execute(inputs);
        requireFailure(res, "limit", "Negative image limit must be rejected");
        REQUIRE(res.status.message.find("非负") != std::string::npos, "Limit error must state non-negative");
    }
    {
        Values inputs = readInputs(files / "valid.png");
        inputs.insert_or_assign(read_ports::in::MAX_BATCH_BYTES, std::int64_t(-1));
        auto res = tool.execute(inputs);
        requireFailure(res, "max_batch_bytes", "Negative budget must be rejected");
        REQUIRE(res.status.message.find("非负") != std::string::npos, "Budget error must state non-negative");
    }

    // 4. Empty file
    {
        const auto path = files / "empty.png";
        std::ofstream(path, std::ios::binary).close();
        Values inputs = readInputs(path);
        auto res = tool.execute(inputs);
        requireFailure(res, "空", "Empty image file must fail");
        REQUIRE(res.status.message.find("empty.png") != std::string::npos, "Error must include the file path");
    }

    // 5. Controlled short read: truncate after the size check, before stream.read
    {
        const auto path = files / "short.png";
        writePng(path, cv::Mat(24, 24, CV_8UC3, cv::Scalar(1, 2, 3)));
        testing::beforeRead = [](const std::filesystem::path &target)
        {
            std::ofstream truncated(target, std::ios::binary | std::ios::trunc);
            truncated << "x";
        };
        Values inputs = readInputs(path);
        auto res = tool.execute(inputs);
        testing::beforeRead = nullptr;
        requireFailure(res, "读取图像文件不完整", "Short read must not decode as a normal image");
        REQUIRE(res.status.message.find("short.png") != std::string::npos, "Short-read error must include the file path");
        REQUIRE(isMissing(res.outputs[read_ports::out::IMAGES]), "Short read must not publish partial images");
    }

    // 6. Directory order and explicit limit
    {
        writePng(directory / "b.png", cv::Mat(8, 8, CV_8UC1, cv::Scalar(2)));
        writePng(directory / "a.png", cv::Mat(8, 8, CV_8UC1, cv::Scalar(1)));
        writePng(directory / "c.png", cv::Mat(8, 8, CV_8UC1, cv::Scalar(3)));
        Values inputs;
        inputs.emplace(read_ports::in::MODE, Utf8("directory"));
        inputs.emplace(read_ports::in::PATH, Utf8(utf8(directory)));
        inputs.emplace(read_ports::in::COLOR, Utf8("gray"));
        inputs.emplace(read_ports::in::LIMIT, std::int64_t(2));
        inputs.emplace(read_ports::in::MAX_BATCH_BYTES, std::int64_t(0));
        auto res = tool.execute(inputs);
        requireSuccess(res, "Directory read must succeed");
        const auto paths = std::get<StringArray>(res.outputs[read_ports::out::PATHS]).buffers()[0].values();
        REQUIRE(paths.size() == 2, "Explicit limit must be honored");
        REQUIRE(paths[0].ends_with("a.png") && paths[1].ends_with("b.png"), "Directory order must be lexical");
    }

    // 7. Cumulative decoded-byte budget
    {
        writePng(budgetDirectory / "a.png", cv::Mat(24, 24, CV_8UC3, cv::Scalar(1, 1, 1)));
        writePng(budgetDirectory / "b.png", cv::Mat(24, 24, CV_8UC3, cv::Scalar(2, 2, 2)));
        Values inputs;
        inputs.emplace(read_ports::in::MODE, Utf8("directory"));
        inputs.emplace(read_ports::in::PATH, Utf8(utf8(budgetDirectory)));
        inputs.emplace(read_ports::in::COLOR, Utf8("bgr"));
        inputs.emplace(read_ports::in::LIMIT, std::int64_t(0));
        inputs.emplace(read_ports::in::MAX_BATCH_BYTES, std::int64_t(2000));
        auto res = tool.execute(inputs);
        requireFailure(res, "预算超限", "Cumulative decoded-byte budget must fail the batch");
        REQUIRE(res.status.message.find("单张解码瞬时峰值") != std::string::npos,
                "Budget error must explain the single-image peak limit");
        REQUIRE(res.status.message.find("b.png") != std::string::npos, "Budget error must name the failing file");
        REQUIRE(isMissing(res.outputs[read_ports::out::IMAGES]), "Budget failure must not publish a truncated batch");
    }

    // 8. Oversized file is rejected before allocation (Windows sparse file)
#ifdef _WIN32
    {
        const auto path = files / "oversized.png";
        HANDLE handle = CreateFileW(path.wstring().c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                                    FILE_ATTRIBUTE_NORMAL, nullptr);
        REQUIRE(handle != INVALID_HANDLE_VALUE, "Oversized fixture must be created");
        LARGE_INTEGER size{};
        size.QuadPart = (static_cast<LONGLONG>(std::numeric_limits<int>::max()) + 4096);
        REQUIRE(SetFilePointerEx(handle, size, nullptr, FILE_BEGIN), "Oversized fixture must seek");
        REQUIRE(SetEndOfFile(handle), "Oversized fixture must set its length");
        CloseHandle(handle);
        Values inputs = readInputs(path);
        auto res = tool.execute(inputs);
        requireFailure(res, "超出可表示范围", "Oversized image must be rejected before allocation");
        std::filesystem::remove(path);
    }
#endif

    std::filesystem::remove_all(directory);
    std::cout << "[PASS] testReadImageTool passed." << std::endl;
}

int main()
{
    try
    {
        testAdaptiveBinaryTool();
        testRoiProjectionTool();
        testImageCropTool();
        testDetectionFilterTool();
        testImagePreprocessTool();
        testReadImageTool();
        std::cout << "\n[ALL TESTS PASSED] All preprocess contract tests passed successfully.\n" << std::endl;
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FATAL] Unhandled exception: " << e.what() << std::endl;
        return 1;
    }
}
