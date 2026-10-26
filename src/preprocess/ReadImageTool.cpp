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

bool hasSupportedExtension(const std::filesystem::path &path)
{
    static const std::vector<std::string> supported = {
        "png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"
    };
    auto extension = toLower(utf8Path(path.extension()));
    if (!extension.empty() && extension.front() == '.')
        extension.erase(extension.begin());
    return std::find(supported.begin(), supported.end(), extension) != supported.end();
}

std::vector<std::filesystem::path> scanDirectory(const std::filesystem::path &root, bool recursive)
{
    struct Entry
    {
        std::string           lower;
        std::filesystem::path path;
    };
    std::vector<Entry> entries;
    const auto         collect = [&](const std::filesystem::directory_entry &entry)
    {
        if (entry.is_regular_file() && hasSupportedExtension(entry.path()))
            entries.push_back({toLower(utf8Path(entry.path().lexically_relative(root))), entry.path()});
    };
    if (recursive)
    {
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
        files.push_back(std::move(entry.path));
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
std::function<void(const std::filesystem::path &)> afterDecode;
std::atomic<std::size_t>                           fileReadCount{0};
std::atomic<std::size_t>                           decodeCount{0};
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
        const auto rawPath = requireString(inputs, p::in::PATH);
        if (rawPath.empty())
            throw visionrt::ContractError("图像路径输入为空");

        const auto *recursiveValue = std::get_if<bool>(&requireInput(inputs, p::in::RECURSIVE));
        if (!recursiveValue)
            throw visionrt::ContractError("recursive 必须是布尔值");
        const bool recursive = *recursiveValue;

        const auto traversal = requireString(inputs, p::in::TRAVERSAL);
        bool isLoop = true;
        if (traversal == "循环" || traversal == "loop")
            isLoop = true;
        else if (traversal == "不循环" || traversal == "once")
            isLoop = false;
        else
            throw visionrt::ContractError("输入端口 'traversal' 的取值无效：" + traversal + "（允许 循环/不循环）");

        const auto order = requireString(inputs, p::in::ORDER);
        bool isReverse = false;
        if (order.find("逆序") != std::string::npos || order == "reverse")
            isReverse = true;
        else if (order.find("顺序") != std::string::npos || order == "forward")
            isReverse = false;
        else
            throw visionrt::ContractError("输入端口 'order' 的取值无效：" + order + "（允许 顺序（下一张）/逆序（上一张））");

        const auto count = requireInt(inputs, p::in::COUNT);
        if (count < 1)
            throw visionrt::ContractError("输入端口 'count' 必须为正整数，实际值：" + std::to_string(count));

        const auto *reloadValue = std::get_if<bool>(&requireInput(inputs, p::in::RELOAD));
        if (!reloadValue)
            throw visionrt::ContractError("reload 必须是布尔值");
        const bool reload = *reloadValue;

        const auto resolved = services_.paths.resolve(rawPath);
        if (!std::filesystem::exists(resolved))
            throw visionrt::ContractError("图像路径不存在：" + utf8Path(resolved));
        const bool isDir = std::filesystem::is_directory(resolved);
        const bool isFile = std::filesystem::is_regular_file(resolved);
        if (!isDir && !isFile)
            throw visionrt::ContractError("图像路径既非文件也非目录：" + utf8Path(resolved));

        const bool needScan = !catalog_
                           || catalog_->rawPath != rawPath
                           || catalog_->recursive != recursive
                           || catalog_->isDirectory != isDir
                           || (reload && isDir);

        if (needScan)
        {
            std::filesystem::path currentFile;
            if (catalog_ && !catalog_->files.empty() && currentIndex_ < catalog_->files.size())
                currentFile = catalog_->files[currentIndex_];

            std::vector<std::filesystem::path> newFiles;
            if (isFile)
            {
                newFiles.push_back(resolved);
            }
            else
            {
                newFiles = scanDirectory(resolved, recursive);
            }

            if (newFiles.empty())
                throw visionrt::ContractError("没有可用图像文件");

            if (catalog_ && catalog_->rawPath == rawPath && catalog_->recursive == recursive && !currentFile.empty())
            {
                auto it = std::find(newFiles.begin(), newFiles.end(), currentFile);
                if (it != newFiles.end())
                    currentIndex_ = static_cast<std::size_t>(std::distance(newFiles.begin(), it));
                else
                    currentIndex_ = 0;
            }
            else
            {
                currentIndex_ = 0;
                fileCache_.clear();
                lastPlannedPaths_.clear();
                lastBatchRef_.reset();
            }

            catalog_ = Catalog{rawPath, isDir, recursive, std::move(newFiles)};
        }

        if (isReverse != lastReverse_)
        {
            lastReverse_ = isReverse;
            currentIndex_ = 0;
        }

        const std::size_t total = catalog_->files.size();
        std::size_t takeCount = 0;
        std::size_t nextIndexAfter = 0;
        bool endReached = false;
        std::vector<std::filesystem::path> plannedFiles;

        if (!isLoop)
        {
            if (currentIndex_ >= total)
                throw visionrt::ContractError("图像序列已结束");

            const std::size_t remaining = total - currentIndex_;
            takeCount = std::min(static_cast<std::size_t>(count), remaining);
            plannedFiles.reserve(takeCount);
            for (std::size_t i = 0; i < takeCount; ++i)
            {
                std::size_t idx = isReverse ? ((total - 1) - (currentIndex_ + i))
                                            : (currentIndex_ + i);
                plannedFiles.push_back(catalog_->files[idx]);
            }
            nextIndexAfter = currentIndex_ + takeCount;
            endReached = (nextIndexAfter >= total);
        }
        else
        {
            takeCount = static_cast<std::size_t>(count);
            plannedFiles.reserve(takeCount);
            for (std::size_t i = 0; i < takeCount; ++i)
            {
                std::size_t offset = (currentIndex_ + i) % total;
                std::size_t idx = isReverse ? ((total - 1) - offset)
                                            : offset;
                plannedFiles.push_back(catalog_->files[idx]);
            }
            nextIndexAfter = (currentIndex_ + takeCount) % total;
            endReached = false;
        }

        std::vector<std::string> plannedPathStrings;
        plannedPathStrings.reserve(plannedFiles.size());
        for (const auto &p : plannedFiles)
            plannedPathStrings.push_back(utf8Path(p));

        const std::size_t reportedCurrentIndex = currentIndex_;

        if (!reload && !lastPlannedPaths_.empty() && plannedPathStrings == lastPlannedPaths_ && lastBatchRef_.has_value())
        {
            currentIndex_ = nextIndexAfter;
            visionrt::Values outputs;
            outputs.emplace(p::out::IMAGES, *lastBatchRef_);
            outputs.emplace(p::out::PATHS, visionrt::Array(std::move(plannedPathStrings)));
            outputs.emplace(p::out::CURRENT_INDEX, static_cast<std::int64_t>(reportedCurrentIndex));
            outputs.emplace(p::out::TOTAL_COUNT, static_cast<std::int64_t>(total));
            outputs.emplace(p::out::END_REACHED, endReached);
            return {visionrt::executionSuccess(), std::move(outputs)};
        }

        std::unordered_map<std::string, cv::Mat> newlyDecoded;
        for (std::size_t i = 0; i < plannedFiles.size(); ++i)
        {
            const auto &file = plannedFiles[i];
            const auto pathStr = plannedPathStrings[i];

            if (!reload && fileCache_.count(pathStr))
                continue;
            if (newlyDecoded.count(pathStr))
                continue;

            std::ifstream stream(file, std::ios::binary | std::ios::ate);
            if (!stream.is_open())
                throw visionrt::ContractError("无法打开图像文件：" + pathStr);
            const auto endPosition = stream.tellg();
            if (endPosition == std::streampos(-1) || !stream)
                throw visionrt::ContractError("无法读取图像文件长度：" + pathStr);
            const auto fileSize = static_cast<std::streamoff>(endPosition);
            if (fileSize <= 0)
                throw visionrt::ContractError("图像文件为空：" + pathStr);
            if (static_cast<std::uintmax_t>(fileSize) > std::numeric_limits<std::size_t>::max()
                || fileSize > std::numeric_limits<std::streamsize>::max()
                || fileSize > std::numeric_limits<int>::max())
                throw visionrt::ContractError("图像文件大小 " + std::to_string(fileSize)
                                              + " 字节超出可表示范围：" + pathStr);

            std::vector<char> buffer(static_cast<std::size_t>(fileSize));
            stream.seekg(0, std::ios::beg);
            if (!stream)
                throw visionrt::ContractError("无法定位图像文件读取位置：" + pathStr);

#ifdef VISIONTOOLS_TESTING
            if (testing::beforeRead)
                testing::beforeRead(file);
            testing::fileReadCount.fetch_add(1);
#endif

            stream.read(buffer.data(), static_cast<std::streamsize>(fileSize));
            const auto readBytes = stream.gcount();
            if (readBytes != static_cast<std::streamsize>(fileSize) || stream.bad())
                throw visionrt::ContractError("读取图像文件不完整：" + pathStr + "（期望 "
                                              + std::to_string(fileSize) + " 字节，实际 "
                                              + std::to_string(readBytes) + " 字节）");

            cv::Mat decoded;
            try
            {
                const cv::Mat rawData(1, static_cast<int>(buffer.size()), CV_8UC1, buffer.data());
                decoded = cv::imdecode(rawData, cv::IMREAD_UNCHANGED);
            }
            catch (const cv::Exception &error)
            {
                return {{1, "图像解码失败：文件 " + pathStr + "（批次索引 " + std::to_string(i) + "）", error.what()},
                        missingOutputs()};
            }
            if (decoded.empty())
                return {{1, "图像解码失败：文件 " + pathStr + "（批次索引 " + std::to_string(i) + "）", "无法解码图像"},
                        missingOutputs()};

#ifdef VISIONTOOLS_TESTING
            if (testing::afterDecode)
                testing::afterDecode(file);
            testing::decodeCount.fetch_add(1);
#endif

            newlyDecoded[pathStr] = std::move(decoded);
        }

        for (auto &pair : newlyDecoded)
            fileCache_[pair.first] = std::move(pair.second);

        visionrt::Image resultImages;
        resultImages.reserve(plannedFiles.size());
        for (const auto &pathStr : plannedPathStrings)
            resultImages.push_back(fileCache_.at(pathStr));

        visionrt::DataRef<visionrt::Image> batchRef(std::move(resultImages));
        lastBatchRef_ = batchRef;
        lastPlannedPaths_ = plannedPathStrings;
        currentIndex_ = nextIndexAfter;

        visionrt::Values outputs;
        outputs.emplace(p::out::IMAGES, batchRef);
        outputs.emplace(p::out::PATHS, visionrt::Array(std::move(plannedPathStrings)));
        outputs.emplace(p::out::CURRENT_INDEX, static_cast<std::int64_t>(reportedCurrentIndex));
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
    currentIndex_ = 0;
    lastReverse_ = false;
    fileCache_.clear();
    lastPlannedPaths_.clear();
    lastBatchRef_.reset();
    return {};
}

} // namespace visiontools::preprocess
