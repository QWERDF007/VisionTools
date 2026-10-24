#include "ReadImageTool.h"

#include "InputPorts.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace visiontools::preprocess {
namespace {

namespace p = ports::image_read;

std::string utf8Path(const std::filesystem::path &path)
{
    const auto value = path.generic_u8string();
    return {reinterpret_cast<const char *>(value.data()), value.size()};
}

std::string toLower(std::string_view s)
{
    std::string res;
    res.reserve(s.size());
    for (char c : s) res.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    return res;
}

bool hasExtension(const std::filesystem::path &path, const std::vector<std::string> &extensions)
{
    const auto ext = toLower(utf8Path(path.extension()));
    for (const auto &candidate : extensions)
    {
        if (ext == candidate || ext == ("." + candidate))
            return true;
    }
    return false;
}

std::uint64_t decodedBytes(const cv::Mat &image)
{
    return static_cast<std::uint64_t>(image.total()) * static_cast<std::uint64_t>(image.elemSize());
}

} // namespace

#ifdef VISIONTOOLS_TESTING
namespace testing {
std::function<void(const std::filesystem::path &)> beforeRead;
} // namespace testing
#endif

ReadImageTool::ReadImageTool(const visionrt::ToolServices &services)
    : services_(services)
{
}

visionrt::ToolStatus ReadImageTool::initialize(const visionrt::Values &parameters)
{
    if (!parameters.empty())
    {
        return {1, "图像读取不接受初始化参数"};
    }
    return {};
}

visionrt::ToolResult ReadImageTool::execute(const visionrt::Values &inputs)
{
    try
    {
        const auto mode = requireString(inputs, p::in::MODE);
        const auto color = requireString(inputs, p::in::COLOR);
        const auto rawPath = requireString(inputs, p::in::PATH);
        const auto limit = requireInt(inputs, p::in::LIMIT);
        const auto maxBatchBytes = requireInt(inputs, p::in::MAX_BATCH_BYTES);
        if (limit < 0)
        {
            throw visionrt::ContractError("输入端口 '" + std::string(p::in::LIMIT) + "' 必须为非负整数，实际值："
                                          + std::to_string(limit));
        }
        if (maxBatchBytes < 0)
        {
            throw visionrt::ContractError("输入端口 '" + std::string(p::in::MAX_BATCH_BYTES)
                                          + "' 必须为非负整数，实际值：" + std::to_string(maxBatchBytes));
        }

        int flags = cv::IMREAD_COLOR;
        if (color == "unchanged")
            flags = cv::IMREAD_UNCHANGED;
        else if (color == "bgr")
            flags = cv::IMREAD_COLOR;
        else if (color == "gray")
            flags = cv::IMREAD_GRAYSCALE;
        else
            throw visionrt::ContractError("输入端口 '" + std::string(p::in::COLOR) + "' 的取值无效：" + color
                                          + "（允许 bgr/gray/unchanged）");

        if (rawPath.empty())
        {
            if (mode == "directory")
                throw visionrt::ContractError("目录路径输入为空");
            throw visionrt::ContractError("文件路径输入为空");
        }

        const auto resolved = services_.paths.resolve(rawPath);
        std::vector<std::filesystem::path> files;

        if (mode == "file")
        {
            if (!std::filesystem::exists(resolved) || !std::filesystem::is_regular_file(resolved))
            {
                throw visionrt::ContractError("图像文件不存在：" + utf8Path(resolved));
            }
            files.push_back(resolved);
        }
        else if (mode == "directory")
        {
            if (!std::filesystem::exists(resolved) || !std::filesystem::is_directory(resolved))
            {
                throw visionrt::ContractError("图像目录不存在：" + utf8Path(resolved));
            }
            const std::vector<std::string> validExtensions = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"};
            std::vector<std::filesystem::path> candidates;
            for (const auto &entry : std::filesystem::directory_iterator(resolved))
            {
                if (entry.is_regular_file() && hasExtension(entry.path(), validExtensions))
                {
                    candidates.push_back(entry.path());
                }
            }
            std::sort(candidates.begin(), candidates.end(),
                      [](const auto &a, const auto &b) { return a.filename().u8string() < b.filename().u8string(); });
            for (const auto &candidate : candidates)
            {
                files.push_back(candidate);
                if (limit > 0 && static_cast<std::int64_t>(files.size()) >= limit)
                    break;
            }
        }
        else
        {
            throw visionrt::ContractError("输入端口 '" + std::string(p::in::MODE) + "' 的取值无效：" + mode
                                          + "（允许 file/directory）");
        }

        visionrt::Image          images;
        std::vector<std::string> pathStrings;
        images.reserve(files.size());
        pathStrings.reserve(files.size());
        const auto budget = static_cast<std::uint64_t>(maxBatchBytes);
        std::uint64_t accumulated = 0;

        for (std::size_t index = 0; index < files.size(); ++index)
        {
            const auto &file = files[index];
            std::ifstream stream(file, std::ios::binary | std::ios::ate);
            if (!stream.is_open())
            {
                throw visionrt::ContractError("无法打开图像文件：" + utf8Path(file));
            }
            const auto endPosition = stream.tellg();
            if (endPosition == std::streampos(-1) || !stream)
            {
                throw visionrt::ContractError("无法读取图像文件长度：" + utf8Path(file));
            }
            const auto fileSize = static_cast<std::streamoff>(endPosition);
            if (fileSize <= 0)
            {
                throw visionrt::ContractError("图像文件为空：" + utf8Path(file));
            }
            if (static_cast<std::uintmax_t>(fileSize) > std::numeric_limits<std::size_t>::max()
                || fileSize > std::numeric_limits<std::streamsize>::max()
                || fileSize > std::numeric_limits<int>::max())
            {
                throw visionrt::ContractError("图像文件大小 " + std::to_string(fileSize)
                                              + " 字节超出可表示范围：" + utf8Path(file));
            }

            std::vector<char> buffer(static_cast<std::size_t>(fileSize));
            stream.seekg(0, std::ios::beg);
            if (!stream)
            {
                throw visionrt::ContractError("无法定位图像文件读取位置：" + utf8Path(file));
            }
#ifdef VISIONTOOLS_TESTING
            if (testing::beforeRead)
                testing::beforeRead(file);
#endif
            stream.read(buffer.data(), static_cast<std::streamsize>(fileSize));
            const auto readBytes = stream.gcount();
            if (readBytes != static_cast<std::streamsize>(fileSize) || stream.bad())
            {
                throw visionrt::ContractError("读取图像文件不完整：" + utf8Path(file) + "（期望 "
                                              + std::to_string(fileSize) + " 字节，实际 "
                                              + std::to_string(readBytes) + " 字节）");
            }

            cv::Mat decoded;
            try
            {
                const cv::Mat rawData(1, static_cast<int>(buffer.size()), CV_8UC1, buffer.data());
                decoded = cv::imdecode(rawData, flags);
            }
            catch (const cv::Exception &error)
            {
                return {
                    {1, "图像解码失败：文件 " + utf8Path(file) + "（批次索引 " + std::to_string(index) + "）",
                     error.what()},
                    {{p::out::IMAGES, visionrt::Missing{}}, {p::out::PATHS, visionrt::Missing{}}}
                };
            }
            if (decoded.empty())
            {
                throw visionrt::ContractError("无法解码图像：" + utf8Path(file));
            }

            const auto imageBytes = decodedBytes(decoded);
            if (budget > 0)
            {
                if (imageBytes > budget - accumulated)
                {
                    throw visionrt::ContractError(
                        "批次解码字节预算超限：文件 " + utf8Path(file) + "（批次索引 " + std::to_string(index)
                        + "，本文件解码 " + std::to_string(imageBytes) + " 字节，累计 "
                        + std::to_string(accumulated + imageBytes) + " 字节，上限 " + std::to_string(budget)
                        + " 字节；单张解码瞬时峰值仍可能超过预算）");
                }
                accumulated += imageBytes;
            }

            images.push_back(std::move(decoded));
            pathStrings.push_back(utf8Path(file));
        }

        visionrt::Values outputs;
        outputs.emplace(p::out::IMAGES, visionrt::DataRef<visionrt::Image>(std::move(images)));
        outputs.emplace(p::out::PATHS, visionrt::Array(std::move(pathStrings)));
        return {visionrt::executionSuccess(), std::move(outputs)};
    }
    catch (const visionrt::ContractError &error)
    {
        return {{1, error.what()},
                {{p::out::IMAGES, visionrt::Missing{}}, {p::out::PATHS, visionrt::Missing{}}}};
    }
    catch (const std::exception &error)
    {
        return {
            {1, "图像读取失败", error.what()},
            {{p::out::IMAGES, visionrt::Missing{}}, {p::out::PATHS, visionrt::Missing{}}}
        };
    }
    catch (...)
    {
        return {{-1, "图像读取失败：未知异常"},
                {{p::out::IMAGES, visionrt::Missing{}}, {p::out::PATHS, visionrt::Missing{}}}};
    }
}

visionrt::ToolStatus ReadImageTool::release()
{
    return {};
}

} // namespace visiontools::preprocess
