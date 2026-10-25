#include "ReadImageTool.h"

#include "common/InputValues.h"

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
    for (char c : s)
        res.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    return res;
}

bool hasExtension(const std::filesystem::path &path, const std::vector<std::string> &extensions)
{
    if (extensions.empty())
        return true;
    auto extension = toLower(utf8Path(path.extension()));
    if (!extension.empty() && extension.front() == '.')
        extension.erase(extension.begin());
    return std::find(extensions.begin(), extensions.end(), extension) != extensions.end();
}

std::uint64_t decodedBytes(const cv::Mat &image)
{
    return static_cast<std::uint64_t>(image.total()) * static_cast<std::uint64_t>(image.elemSize());
}

std::vector<std::string> stringArray(const visionrt::Values &inputs, std::string_view id)
{
    const auto &value = requireInput(inputs, id);
    const auto *array = std::get_if<visionrt::Array>(&value);
    if (!array || array->type() != visionrt::TypeSpec{visionrt::Type::Array, visionrt::DType::String, 1})
        throw visionrt::ContractError(std::string(id) + " 必须是 Array(string, rank=1)");
    return array->get<std::vector<std::string>>();
}

std::vector<std::string> normalizedExtensions(const std::vector<std::string> &raw)
{
    std::vector<std::string> result;
    result.reserve(raw.size());
    for (const auto &item : raw)
    {
        std::string value = toLower(item);
        if (!value.empty() && value.front() == '.')
            value.erase(value.begin());
        if (value.empty())
            throw visionrt::ContractError("extensions 不能包含空后缀");
        result.push_back(std::move(value));
    }
    return result;
}

std::string catalogKey(const std::string &mode, const std::string &path, const std::vector<std::string> &paths,
                       bool recursive, const std::vector<std::string> &extensions, std::int64_t limit)
{
    std::string key = mode;
    key += '\x1f';
    key += path;
    key += recursive ? "\x1fR" : "\x1f-";
    for (const auto &item : paths)
    {
        key += '\x1e';
        key += item;
    }
    key += '\x1d';
    for (const auto &item : extensions)
    {
        key += item;
        key += ',';
    }
    key += '\x1d';
    key += std::to_string(limit);
    return key;
}

std::vector<std::filesystem::path> scanDirectory(const std::filesystem::path &root,
                                                 const std::vector<std::string> &extensions, bool recursive,
                                                 std::int64_t limit)
{
    struct Entry
    {
        std::string           lower;
        std::filesystem::path path;
    };
    std::vector<Entry> entries;
    const auto         collect = [&](const std::filesystem::directory_entry &entry)
    {
        if (entry.is_regular_file() && hasExtension(entry.path(), extensions))
            entries.push_back({toLower(utf8Path(entry.path().lexically_relative(root))), entry.path()});
    };
    if (recursive)
    {
        // directory_options::none does not follow directory symlinks, so a link
        // cycle cannot make the scan unbounded.
        for (const auto &entry :
             std::filesystem::recursive_directory_iterator(root, std::filesystem::directory_options::none))
            collect(entry);
    }
    else
    {
        for (const auto &entry : std::filesystem::directory_iterator(root))
            collect(entry);
    }
    std::sort(entries.begin(), entries.end(),
              [](const Entry &left, const Entry &right)
              {
                  if (left.lower != right.lower)
                      return left.lower < right.lower;
                  return left.path.u8string() < right.path.u8string();
              });
    std::vector<std::filesystem::path> files;
    files.reserve(entries.size());
    for (auto &entry : entries)
    {
        files.push_back(std::move(entry.path));
        if (limit > 0 && static_cast<std::int64_t>(files.size()) >= limit)
            break;
    }
    return files;
}

visionrt::Values missingOutputs()
{
    return {{p::out::IMAGES, visionrt::Missing{}},
            {p::out::PATHS, visionrt::Missing{}},
            {p::out::CURRENT_INDEX, visionrt::Missing{}},
            {p::out::TOTAL_COUNT, visionrt::Missing{}},
            {p::out::END_REACHED, visionrt::Missing{}}};
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
        return {1, "图像读取不接受初始化参数"};
    return {};
}

visionrt::ToolResult ReadImageTool::execute(const visionrt::Values &inputs)
{
    try
    {
        const auto  mode          = requireString(inputs, p::in::MODE);
        const auto  readMode      = requireString(inputs, p::in::READ_MODE);
        const auto  color         = requireString(inputs, p::in::COLOR);
        const auto  endPolicy     = requireString(inputs, p::in::END_POLICY);
        const auto  rawPath       = requireString(inputs, p::in::PATH);
        const auto  limit         = requireInt(inputs, p::in::LIMIT);
        const auto  maxBatchBytes = requireInt(inputs, p::in::MAX_BATCH_BYTES);
        const auto *recursiveValue = std::get_if<bool>(&requireInput(inputs, p::in::RECURSIVE));
        if (!recursiveValue)
            throw visionrt::ContractError("recursive 必须是布尔值");
        const bool recursive = *recursiveValue;
        if (limit < 0)
            throw visionrt::ContractError("输入端口 'limit' 必须为非负整数，实际值：" + std::to_string(limit));
        if (maxBatchBytes < 0)
            throw visionrt::ContractError("输入端口 'max_batch_bytes' 必须为非负整数，实际值："
                                          + std::to_string(maxBatchBytes));
        if (readMode != "all" && readMode != "sequence")
            throw visionrt::ContractError("输入端口 'read_mode' 的取值无效：" + readMode + "（允许 all/sequence）");
        if (endPolicy != "wrap" && endPolicy != "hold" && endPolicy != "error")
            throw visionrt::ContractError("输入端口 'end_policy' 的取值无效：" + endPolicy
                                          + "（允许 wrap/hold/error）");

        int flags = cv::IMREAD_COLOR;
        if (color == "unchanged")
            flags = cv::IMREAD_UNCHANGED;
        else if (color == "bgr")
            flags = cv::IMREAD_COLOR;
        else if (color == "gray")
            flags = cv::IMREAD_GRAYSCALE;
        else
            throw visionrt::ContractError("输入端口 'color' 的取值无效：" + color + "（允许 bgr/gray/unchanged）");

        if (mode != "file" && mode != "directory" && mode != "list")
            throw visionrt::ContractError("输入端口 'mode' 的取值无效：" + mode + "（允许 file/directory/list）");

        const auto extensions = normalizedExtensions(stringArray(inputs, p::in::EXTENSIONS));
        const auto listed     = stringArray(inputs, p::in::PATHS_IN);
        if (mode != "list" && rawPath.empty())
            throw visionrt::ContractError(mode == "directory" ? "目录路径输入为空" : "文件路径输入为空");

        const auto key
            = catalogKey(mode, mode == "list" ? std::string{} : rawPath, listed, recursive, extensions, limit);
        if (!catalog_ || catalog_->key != key)
        {
            Catalog rebuilt;
            rebuilt.key = key;
            if (mode == "file")
            {
                const auto resolved = services_.paths.resolve(rawPath);
                if (!std::filesystem::exists(resolved) || !std::filesystem::is_regular_file(resolved))
                    throw visionrt::ContractError("图像文件不存在：" + utf8Path(resolved));
                rebuilt.files.push_back(resolved);
            }
            else if (mode == "directory")
            {
                const auto resolved = services_.paths.resolve(rawPath);
                if (!std::filesystem::exists(resolved) || !std::filesystem::is_directory(resolved))
                    throw visionrt::ContractError("图像目录不存在：" + utf8Path(resolved));
                rebuilt.files = scanDirectory(resolved, extensions, recursive, limit);
            }
            else
            {
                if (listed.empty())
                    throw visionrt::ContractError("文件列表输入为空");
                for (const auto &item : listed)
                {
                    if (item.empty())
                        throw visionrt::ContractError("文件列表不能包含空路径");
                    const auto resolved = services_.paths.resolve(item);
                    if (!std::filesystem::exists(resolved) || !std::filesystem::is_regular_file(resolved))
                        throw visionrt::ContractError("图像文件不存在：" + utf8Path(resolved));
                    rebuilt.files.push_back(resolved);
                    if (limit > 0 && static_cast<std::int64_t>(rebuilt.files.size()) >= limit)
                        break;
                }
            }
            if (rebuilt.files.empty())
                throw visionrt::ContractError("没有可用图像文件");
            catalog_   = std::move(rebuilt);
            nextIndex_ = 0;
        }
        if (readMode != readMode_)
        {
            readMode_  = readMode;
            nextIndex_ = 0;
        }

        const auto  total = catalog_->files.size();
        std::size_t begin = 0;
        std::size_t end   = total;
        if (readMode == "sequence")
        {
            if (nextIndex_ >= total)
            {
                if (endPolicy == "error")
                    throw visionrt::ContractError("图像序列已结束");
                nextIndex_ = endPolicy == "hold" ? total - 1 : 0;
            }
            begin = nextIndex_;
            end   = begin + 1;
        }

        visionrt::Image          images;
        std::vector<std::string> pathStrings;
        std::uint64_t            accumulated = 0;
        const auto               budget      = static_cast<std::uint64_t>(maxBatchBytes);
        for (std::size_t index = begin; index < end; ++index)
        {
            const auto &file = catalog_->files[index];
            std::ifstream stream(file, std::ios::binary | std::ios::ate);
            if (!stream.is_open())
                throw visionrt::ContractError("无法打开图像文件：" + utf8Path(file));
            const auto endPosition = stream.tellg();
            if (endPosition == std::streampos(-1) || !stream)
                throw visionrt::ContractError("无法读取图像文件长度：" + utf8Path(file));
            const auto fileSize = static_cast<std::streamoff>(endPosition);
            if (fileSize <= 0)
                throw visionrt::ContractError("图像文件为空：" + utf8Path(file));
            if (static_cast<std::uintmax_t>(fileSize) > std::numeric_limits<std::size_t>::max()
                || fileSize > std::numeric_limits<std::streamsize>::max()
                || fileSize > std::numeric_limits<int>::max())
                throw visionrt::ContractError("图像文件大小 " + std::to_string(fileSize)
                                              + " 字节超出可表示范围：" + utf8Path(file));

            std::vector<char> buffer(static_cast<std::size_t>(fileSize));
            stream.seekg(0, std::ios::beg);
            if (!stream)
                throw visionrt::ContractError("无法定位图像文件读取位置：" + utf8Path(file));
#ifdef VISIONTOOLS_TESTING
            if (testing::beforeRead)
                testing::beforeRead(file);
#endif
            stream.read(buffer.data(), static_cast<std::streamsize>(fileSize));
            const auto readBytes = stream.gcount();
            if (readBytes != static_cast<std::streamsize>(fileSize) || stream.bad())
                throw visionrt::ContractError("读取图像文件不完整：" + utf8Path(file) + "（期望 "
                                              + std::to_string(fileSize) + " 字节，实际 "
                                              + std::to_string(readBytes) + " 字节）");

            cv::Mat decoded;
            try
            {
                const cv::Mat rawData(1, static_cast<int>(buffer.size()), CV_8UC1, buffer.data());
                decoded = cv::imdecode(rawData, flags);
            }
            catch (const cv::Exception &error)
            {
                return {{1,
                         "图像解码失败：文件 " + utf8Path(file) + "（批次索引 " + std::to_string(index - begin) + "）",
                         error.what()},
                        missingOutputs()};
            }
            if (decoded.empty())
                throw visionrt::ContractError("无法解码图像：" + utf8Path(file));

            const auto imageBytes = decodedBytes(decoded);
            if (budget > 0)
            {
                if (imageBytes > budget - accumulated)
                    throw visionrt::ContractError(
                        "批次解码字节预算超限：文件 " + utf8Path(file) + "（批次索引 " + std::to_string(index - begin)
                        + "，本文件解码 " + std::to_string(imageBytes) + " 字节，累计 "
                        + std::to_string(accumulated + imageBytes) + " 字节，上限 " + std::to_string(budget)
                        + " 字节；单张解码瞬时峰值仍可能超过预算）");
                accumulated += imageBytes;
            }
            images.push_back(std::move(decoded));
            pathStrings.push_back(utf8Path(file));
        }

        const std::size_t currentIndex = readMode == "sequence" ? begin : 0;
        const bool        endReached   = readMode == "sequence" ? begin + 1 >= total : true;
        if (readMode == "sequence")
            nextIndex_ = std::min(begin + 1, total);

        visionrt::Values outputs;
        outputs.emplace(p::out::IMAGES, visionrt::DataRef<visionrt::Image>(std::move(images)));
        outputs.emplace(p::out::PATHS, visionrt::Array(std::move(pathStrings)));
        outputs.emplace(p::out::CURRENT_INDEX, static_cast<std::int64_t>(currentIndex));
        outputs.emplace(p::out::TOTAL_COUNT, static_cast<std::int64_t>(total));
        outputs.emplace(p::out::END_REACHED, endReached);
        return {visionrt::executionSuccess(), std::move(outputs)};
    }
    catch (const visionrt::ContractError &error)
    {
        return {{1, error.what()}, missingOutputs()};
    }
    catch (const std::exception &error)
    {
        return {{1, "图像读取失败", error.what()}, missingOutputs()};
    }
    catch (...)
    {
        return {{-1, "图像读取失败：未知异常"}, missingOutputs()};
    }
}

visionrt::ToolStatus ReadImageTool::release()
{
    catalog_.reset();
    nextIndex_ = 0;
    readMode_.clear();
    return {};
}

} // namespace visiontools::preprocess
