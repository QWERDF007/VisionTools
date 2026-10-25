#include "ImagePreprocessTool.h"

#include "common/InputValues.h"

#include <opencv2/imgproc.hpp>

#include <limits>
#include <vector>

namespace visiontools::preprocess {
namespace {

namespace p = ports::image_preprocess;

void requireU8(const cv::Mat &image, std::size_t index)
{
    if (image.empty() || CV_MAT_DEPTH(image.type()) != CV_8U
        || (image.channels() != 1 && image.channels() != 3 && image.channels() != 4))
    {
        throw visionrt::ContractError("批次索引 " + std::to_string(index)
                                      + " 的图像必须是非空 uint8 灰度/BGR/BGRA 图像");
    }
}

cv::Mat toGray(const cv::Mat &image, std::size_t index)
{
    requireU8(image, index);
    if (image.channels() == 1)
        return image;
    cv::Mat result;
    cv::cvtColor(image, result, image.channels() == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY);
    return result;
}

} // namespace

ImagePreprocessTool::ImagePreprocessTool(const visionrt::ToolServices &services)
    : services_(services)
{
}

visionrt::ToolStatus ImagePreprocessTool::initialize(const visionrt::Values &parameters)
{
    if (!parameters.empty())
    {
        return {1, "图像预处理不接受初始化参数"};
    }
    return {};
}

visionrt::ToolResult ImagePreprocessTool::execute(const visionrt::Values &inputs)
{
    try
    {
        const auto &images
            = requireTypedInput<visionrt::DataRef<visionrt::Image>>(inputs, p::in::IMAGES, {visionrt::Type::Image});

        const auto operation = requireString(inputs, p::in::OPERATION);
        if (operation != "identity" && operation != "gray" && operation != "gaussian" && operation != "threshold")
        {
            throw visionrt::ContractError("输入端口 '" + std::string(p::in::OPERATION) + "' 的取值无效："
                                          + operation + "（允许 identity/gray/gaussian/threshold）");
        }

        int kernel = 0;
        if (operation == "gaussian")
        {
            kernel = requireInt32(inputs, p::in::KERNEL, 1, std::numeric_limits<int>::max());
            if (kernel % 2 == 0)
            {
                throw visionrt::ContractError("输入端口 '" + std::string(p::in::KERNEL)
                                              + "' 必须为奇数（高斯核），实际值：" + std::to_string(kernel));
            }
        }

        std::int64_t threshold = 0;
        if (operation == "threshold")
        {
            threshold = requireIntInRange(inputs, p::in::THRESHOLD, 0, 255);
        }

        if (images.get().empty())
        {
            visionrt::Values outputs;
            outputs.emplace(p::out::IMAGES, visionrt::DataRef<visionrt::Image>(visionrt::Image{}));
            return {visionrt::executionSuccess(), std::move(outputs)};
        }

        for (std::size_t index = 0; index < images.get().size(); ++index)
        {
            requireU8(images.get()[index], index);
        }

        if (operation == "identity")
        {
            visionrt::Values outputs;
            outputs.emplace(p::out::IMAGES, images);
            return {visionrt::executionSuccess(), std::move(outputs)};
        }

        visionrt::Image output;
        output.reserve(images.get().size());
        for (std::size_t index = 0; index < images.get().size(); ++index)
        {
            const auto &image = images.get()[index];
            cv::Mat processed;
            if (operation == "gaussian")
            {
                cv::GaussianBlur(image, processed, cv::Size(kernel, kernel), 0, 0, cv::BORDER_REFLECT_101);
            }
            else if (operation == "threshold")
            {
                cv::compare(toGray(image, index), cv::Scalar(static_cast<double>(threshold)), processed, cv::CMP_GE);
            }
            else
            {
                processed = image.channels() == 1 ? image.clone() : toGray(image, index);
            }
            output.push_back(std::move(processed));
        }

        visionrt::Values outputs;
        outputs.emplace(p::out::IMAGES, visionrt::DataRef<visionrt::Image>(std::move(output)));
        return {visionrt::executionSuccess(), std::move(outputs)};
    }
    catch (const visionrt::ContractError &error)
    {
        return {{1, error.what()}, {{p::out::IMAGES, visionrt::Missing{}}}};
    }
    catch (const std::exception &error)
    {
        return {
            {1, "图像预处理失败", error.what()},
            {{p::out::IMAGES, visionrt::Missing{}}}
        };
    }
    catch (...)
    {
        return {{-1, "图像预处理失败：未知异常"}, {{p::out::IMAGES, visionrt::Missing{}}}};
    }
}

visionrt::ToolStatus ImagePreprocessTool::release()
{
    return {};
}

} // namespace visiontools::preprocess
