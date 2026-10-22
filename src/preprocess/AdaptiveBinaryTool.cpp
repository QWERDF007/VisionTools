#include "AdaptiveBinaryTool.h"

#include "InputPorts.h"

#include <opencv2/imgproc.hpp>

#include <limits>
#include <vector>

namespace visiontools::preprocess {
namespace {

namespace p = ports::adaptive_binary;

} // namespace

AdaptiveBinaryTool::AdaptiveBinaryTool(const visionrt::ToolServices &services)
    : services_(services)
{
}

visionrt::ToolStatus AdaptiveBinaryTool::initialize(const visionrt::Values &parameters)
{
    if (!parameters.empty())
    {
        return {1, "自适应二值化不接受初始化参数"};
    }
    return {};
}

visionrt::ToolResult AdaptiveBinaryTool::execute(const visionrt::Values &inputs)
{
    try
    {
        const auto &imageValue = requireInput(inputs, p::in::IMAGES);
        const auto *images = std::get_if<visionrt::ImageBatch>(&imageValue);
        if (!images)
        {
            throw visionrt::ContractError("输入端口 '" + std::string(p::in::IMAGES) + "' 必须是 ImageBatch");
        }

        const auto batchSize = images->images().size();
        if (batchSize == 0)
        {
            visionrt::Values outputs;
            outputs.emplace(p::out::IMAGES, visionrt::ImageBatch{});
            return {visionrt::executionSuccess(), std::move(outputs)};
        }

        const auto blockSize = requireIntInRange(inputs, p::in::BLOCK_SIZE, 3, 999);
        if (blockSize % 2 == 0)
        {
            throw visionrt::ContractError("输入端口 '" + std::string(p::in::BLOCK_SIZE) + "' 必须为奇数（拒绝偶数值），实际值："
                                          + std::to_string(blockSize));
        }
        const auto constantOffset = requireFloatInRange(inputs, p::in::CONSTANT_OFFSET,
                                                        -std::numeric_limits<double>::max(),
                                                        std::numeric_limits<double>::max());

        for (std::size_t index = 0; index < batchSize; ++index)
        {
            const auto &image = images->images()[index];
            if (image.empty())
            {
                throw visionrt::ContractError("批次索引 " + std::to_string(index) + " 的图像为空");
            }
            if (image.type() != CV_8UC1)
            {
                throw visionrt::ContractError("批次索引 " + std::to_string(index)
                                              + " 的图像必须是单通道 8 位无符号（CV_8UC1）");
            }
        }

        std::vector<visionrt::OwnedImage> outputImages;
        outputImages.reserve(batchSize);
        for (std::size_t index = 0; index < batchSize; ++index)
        {
            cv::Mat binary;
            cv::adaptiveThreshold(images->images()[index].readOnlyMat(), binary, 255.0, cv::ADAPTIVE_THRESH_MEAN_C,
                                  cv::THRESH_BINARY, static_cast<int>(blockSize), constantOffset);
            outputImages.push_back(visionrt::OwnedImage::take(std::move(binary)));
        }

        visionrt::Values outputs;
        outputs.emplace(p::out::IMAGES, visionrt::ImageBatch(std::move(outputImages)));
        return {visionrt::executionSuccess(), std::move(outputs)};
    }
    catch (const visionrt::ContractError &error)
    {
        return {{1, error.what()}, {{p::out::IMAGES, visionrt::Missing{}}}};
    }
    catch (const std::exception &error)
    {
        return {
            {1, "自适应二值化失败", error.what()},
            {{p::out::IMAGES, visionrt::Missing{}}}
        };
    }
    catch (...)
    {
        return {{-1, "自适应二值化失败：未知异常"}, {{p::out::IMAGES, visionrt::Missing{}}}};
    }
}

visionrt::ToolStatus AdaptiveBinaryTool::release()
{
    return {};
}

} // namespace visiontools::preprocess
