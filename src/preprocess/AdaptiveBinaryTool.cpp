#include "AdaptiveBinaryTool.h"

#include "PortConstants.h"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <vector>

namespace visiontools::preprocess {
namespace {

namespace p = visiontools::ports::visiontools_preprocess_adaptive_binary;

} // namespace

AdaptiveBinaryTool::AdaptiveBinaryTool(const visionrt::ToolServices &services)
    : services_(services)
{
}

visionrt::ToolStatus AdaptiveBinaryTool::initialize(const visionrt::Values &parameters)
{
    if (!parameters.empty())
    {
        return {1, "AdaptiveBinaryTool accepts no init parameters"};
    }
    return {};
}

visionrt::ToolResult AdaptiveBinaryTool::execute(const visionrt::Values &inputs)
{
    try
    {
        // 1. 输入 Missing 或 非 ImageBatch 检验：非零业务状态；错误消息指明 port
        auto imgIt = inputs.find(p::in::IMAGES);
        if (imgIt == inputs.end() || visionrt::isMissing(imgIt->second))
        {
            visionrt::Values outputs;
            outputs.emplace(p::out::IMAGES, visionrt::Missing{});
            return {
                {1, "Input port 'images' is missing"},
                std::move(outputs)
            };
        }
        const auto *images = std::get_if<visionrt::ImageBatch>(&imgIt->second);
        if (!images)
        {
            visionrt::Values outputs;
            outputs.emplace(p::out::IMAGES, visionrt::Missing{});
            return {
                {1, "Input port 'images' must be an ImageBatch"},
                std::move(outputs)
            };
        }

        // 2. B=0 处理：成功，返回空批次
        const auto batchSize = images->images().size();
        if (batchSize == 0)
        {
            visionrt::Values outputs;
            outputs.emplace(p::out::IMAGES, visionrt::ImageBatch{});
            return {
                {0, ""},
                std::move(outputs)
            };
        }

        // 3. block_size 与 constant_offset 校验
        std::int64_t blockSize      = 3;
        double       constantOffset = 5.0;

        auto bsIt = inputs.find(p::in::BLOCK_SIZE);
        if (bsIt != inputs.end() && !visionrt::isMissing(bsIt->second))
        {
            const auto *bsVal = std::get_if<std::int64_t>(&bsIt->second);
            if (!bsVal)
            {
                visionrt::Values outputs;
                outputs.emplace(p::out::IMAGES, visionrt::Missing{});
                return {
                    {1, "Input port 'block_size' must be an integer"},
                    std::move(outputs)
                };
            }
            blockSize = *bsVal;
        }

        if (blockSize < 3 || blockSize > 999)
        {
            visionrt::Values outputs;
            outputs.emplace(p::out::IMAGES, visionrt::Missing{});
            return {
                {1, "Input port 'block_size' must be between 3 and 999"},
                std::move(outputs)
            };
        }
        if (blockSize % 2 == 0)
        {
            visionrt::Values outputs;
            outputs.emplace(p::out::IMAGES, visionrt::Missing{});
            return {
                {1, "Input port 'block_size' must be an odd number (even values are rejected)"},
                std::move(outputs)
            };
        }

        auto coIt = inputs.find(p::in::CONSTANT_OFFSET);
        if (coIt != inputs.end() && !visionrt::isMissing(coIt->second))
        {
            if (const auto *dVal = std::get_if<double>(&coIt->second))
            {
                constantOffset = *dVal;
            }
            else
            {
                // Strict check: Int or non-float types must be rejected (VS-03 / TS-04)
                visionrt::Values outputs;
                outputs.emplace(p::out::IMAGES, visionrt::Missing{});
                return {
                    {1, "Input port 'constant_offset' must be a float (int not accepted)"},
                    std::move(outputs)
                };
            }
        }
        if (!std::isfinite(constantOffset))
        {
            visionrt::Values outputs;
            outputs.emplace(p::out::IMAGES, visionrt::Missing{});
            return {
                {1, "Input port 'constant_offset' must be finite"},
                std::move(outputs)
            };
        }

        // 4. 对 B>0，先验证全部图像为非空 CV_8UC1；不自动灰度转换，错误指出 batch index
        for (std::size_t b = 0; b < batchSize; ++b)
        {
            const auto &img = images->images()[b];
            if (img.empty())
            {
                visionrt::Values outputs;
                outputs.emplace(p::out::IMAGES, visionrt::Missing{});
                return {
                    {1, "Image at batch index " + std::to_string(b) + " is empty"},
                    std::move(outputs)
                };
            }
            if (img.type() != CV_8UC1)
            {
                visionrt::Values outputs;
                outputs.emplace(p::out::IMAGES, visionrt::Missing{});
                return {
                    {1,
                     "Image at batch index " + std::to_string(b) + " must be single-channel 8-bit unsigned (CV_8UC1)"},
                    std::move(outputs)
                };
            }
        }

        // 5. 逐图调用 OpenCV adaptiveThreshold，最大值255、ADAPTIVE_THRESH_MEAN_C、THRESH_BINARY
        std::vector<visionrt::OwnedImage> outputImages;
        outputImages.reserve(batchSize);

        for (std::size_t b = 0; b < batchSize; ++b)
        {
            const auto &inputMat = images->images()[b].readOnlyMat();
            cv::Mat     binaryMat;
            cv::adaptiveThreshold(inputMat, binaryMat, 255.0, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY,
                                  static_cast<int>(blockSize), constantOffset);
            outputImages.push_back(visionrt::OwnedImage::take(std::move(binaryMat)));
        }

        visionrt::Values outputs;
        outputs.emplace(p::out::IMAGES, visionrt::ImageBatch(std::move(outputImages)));
        return {
            {0, ""},
            std::move(outputs)
        };
    }
    catch (const std::exception &e)
    {
        visionrt::Values outputs;
        outputs.emplace(p::out::IMAGES, visionrt::Missing{});
        return {
            {1, e.what()},
            std::move(outputs)
        };
    }
}

visionrt::ToolStatus AdaptiveBinaryTool::release()
{
    return {};
}

} // namespace visiontools::preprocess
