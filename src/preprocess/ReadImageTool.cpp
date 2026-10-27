#include "ReadImageTool.h"

#include "common/InputValues.h"
#include "common/UpstreamStatus.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace visiontools::preprocess {
namespace {

namespace p = ports::image_read;

// Largest batch the tool itself accepts. The declaration publishes the same
// business limit; hosts clamp to it and a direct call is refused here.
constexpr std::int64_t kMaxCount = 64;

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
    static const std::vector<std::string> supported = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"};
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

// A read failure keeps the user-facing message short and carries the decoder's
// technical text separately, so callers can show a precise reason without
// leaking library internals into the primary message.
struct ImageFailure
{
    std::string message;
    std::string detail;
};

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

void ReadImageTool::resetSequence()
{
    cursor_ = {};
    reverse_ = false;
    cache_.clear();
    lastPlannedPaths_.clear();
    lastBatchRef_.reset();
}

ReadImageTool::DecodedBatch ReadImageTool::decode(const Catalog &catalog, const SequencePlan &plan, bool reload,
                                                  PendingInserts &pending)
{
    DecodedBatch batch;
    batch.keys = plan.keys;
    batch.paths.reserve(plan.files.size());
    batch.images.reserve(plan.files.size());

    for (std::size_t i = 0; i < plan.files.size(); ++i)
    {
        const auto &file    = plan.files[i];
        const auto  pathStr = utf8Path(file);
        batch.paths.push_back(pathStr);

        const cv::Mat *cached = (!reload) ? cache_.find(pathStr) : nullptr;
        if (cached)
        {
            batch.images.push_back(*cached);
            continue;
        }

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
            throw visionrt::ContractError("读取图像文件不完整：" + pathStr + "（期望 " + std::to_string(fileSize)
                                          + " 字节，实际 " + std::to_string(readBytes) + " 字节）");

        cv::Mat decoded;
        try
        {
            const cv::Mat rawData(1, static_cast<int>(buffer.size()), CV_8UC1, buffer.data());
            decoded = cv::imdecode(rawData, cv::IMREAD_UNCHANGED);
        }
        catch (const cv::Exception &error)
        {
            throw ImageFailure{"图像解码失败：文件 " + pathStr + "（批次索引 " + std::to_string(i) + "）",
                               error.what()};
        }
        if (decoded.empty())
            throw ImageFailure{"图像解码失败：文件 " + pathStr + "（批次索引 " + std::to_string(i)
                                   + "）：无法解码图像",
                               {}};

#ifdef VISIONTOOLS_TESTING
        if (testing::afterDecode)
            testing::afterDecode(file);
        testing::decodeCount.fetch_add(1);
#endif

        // A cache write is a discardable side effect of a successful batch: nothing is retained
        // until the whole batch decoded, so a failed attempt cannot mix its pixels into history.
        pending.emplace_back(pathStr, decoded);
        ++batch.decoded;
        batch.images.push_back(std::move(decoded));
    }
    return batch;
}

visionrt::ToolResult ReadImageTool::commit(DecodedBatch batch, const SequencePlan &plan, SequenceCursor cursor,
                                           const Catalog *scannedCatalog, bool sourceChanged, bool reverse,
                                           std::size_t totalFiles)
{
    // Commit point: nothing below can fail, so a failed scan, read or decode leaves the previous
    // catalog, cursor, cache and last successful batch exactly as they were.
    if (scannedCatalog)
        catalog_ = *scannedCatalog;
    if (sourceChanged)
        cache_.clear();
    for (auto &[key, image] : batch.pending)
        cache_.insert(key, image);
    cache_.protect(batch.paths);

    lastPlannedPaths_ = batch.paths;
    lastBatchRef_     = visionrt::DataRef<visionrt::Image>(std::move(batch.images));
    reverse_          = reverse;
    cursor_           = std::move(cursor);
    cursor_.started   = true;
    cursor_.nextKey   = plan.nextKey;
    cursor_.lastKey   = plan.lastKey;
    cursor_.exhausted = plan.exhausted;

    visionrt::Values outputs;
    outputs.emplace(p::out::IMAGES, *lastBatchRef_);
    outputs.emplace(p::out::PATHS, visionrt::Array(std::move(batch.paths)));
    outputs.emplace(p::out::CURRENT_INDEX, static_cast<std::int64_t>(plan.startIndex));
    outputs.emplace(p::out::TOTAL_COUNT, static_cast<std::int64_t>(totalFiles));
    outputs.emplace(p::out::END_REACHED, plan.exhausted);
    return visionrt::toolResult(std::move(outputs), visionrt::executionSuccess());
}

visionrt::ToolResult ReadImageTool::execute(const visionrt::Values &inputs)
{
    // An unselected upstream branch is not an error: no algorithm runs and nothing is read, so the
    // decision is taken before the catalog, the cursor or the cache can change.
    if (const auto upstream = upstreamStatus(inputs); upstream != 0)
        return skippedByUpstream(upstream, missingOutputs());

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
        bool       isLoop    = true;
        if (traversal == "循环" || traversal == "loop")
            isLoop = true;
        else if (traversal == "不循环" || traversal == "once")
            isLoop = false;
        else
            throw visionrt::ContractError("输入端口 'traversal' 的取值无效：" + traversal + "（允许 循环/不循环）");

        const auto order     = requireString(inputs, p::in::ORDER);
        bool       isReverse = false;
        if (order.find("逆序") != std::string::npos || order == "reverse")
            isReverse = true;
        else if (order.find("顺序") != std::string::npos || order == "forward")
            isReverse = false;
        else
            throw visionrt::ContractError("输入端口 'order' 的取值无效：" + order
                                          + "（允许 顺序（下一张）/逆序（上一张））");

        const auto count = requireInt(inputs, p::in::COUNT);
        if (count < 1 || count > kMaxCount)
            throw visionrt::ContractError("输入端口 'count' 必须在 1–" + std::to_string(kMaxCount)
                                          + " 之间，实际值：" + std::to_string(count));

        const auto *reloadValue = std::get_if<bool>(&requireInput(inputs, p::in::RELOAD));
        if (!reloadValue)
            throw visionrt::ContractError("reload 必须是布尔值");
        const bool reload = *reloadValue;

        const auto resolved = services_.paths.resolve(rawPath);
        if (!std::filesystem::exists(resolved))
            throw visionrt::ContractError("图像路径不存在：" + utf8Path(resolved));
        const bool isDir  = std::filesystem::is_directory(resolved);
        const bool isFile = std::filesystem::is_regular_file(resolved);
        if (!isDir && !isFile)
            throw visionrt::ContractError("图像路径既非文件也非目录：" + utf8Path(resolved));

        const bool sourceChanged = !catalog_ || catalog_->rawPath != rawPath || catalog_->recursive != recursive
                                || catalog_->isDirectory != isDir;
        const bool directionChanged = isReverse != reverse_;

        // Candidate catalog: a rescan is prepared locally and only becomes the tool's state after a
        // fully successful batch.
        std::optional<Catalog> scanned;
        const Catalog         *catalog = catalog_ ? &*catalog_ : nullptr;
        if (sourceChanged || (reload && isDir) || !catalog)
        {
            Catalog candidate;
            candidate.rawPath     = rawPath;
            candidate.isDirectory = isDir;
            candidate.recursive   = recursive;
            if (isFile)
                candidate.files.push_back(resolved);
            else
                candidate.files = scanDirectory(resolved, recursive);
            if (candidate.files.empty())
                throw visionrt::ContractError("没有可用图像文件");
            scanned = std::move(candidate);
            catalog = &*scanned;
        }

        // Candidate cursor: a changed source or direction starts a fresh sequence, but the committed
        // one is only replaced once this call succeeds.
        SequenceCursor nextCursor = (sourceChanged || directionChanged) ? SequenceCursor{} : cursor_;
        const auto     request
            = SequenceRequest{isReverse, isLoop, static_cast<std::size_t>(count)};
        auto plan = planSequence(catalog->files, nextCursor, request);

        // A file can be deleted between calls without the path changing. Re-scan once so the cursor
        // is resolved against the folder as it is now: a deleted tail must end the sequence (or wrap
        // on request) instead of being retried as a missing file.
        if (isDir)
        {
            const auto missingAtPlan = std::any_of(plan.files.begin(), plan.files.end(),
                                                   [](const auto &file) { return !std::filesystem::exists(file); });
            if (missingAtPlan)
            {
                Catalog refreshed;
                refreshed.rawPath     = rawPath;
                refreshed.isDirectory = true;
                refreshed.recursive   = recursive;
                refreshed.files       = scanDirectory(resolved, recursive);
                if (!refreshed.files.empty())
                {
                    scanned = std::move(refreshed);
                    catalog = &*scanned;
                    plan    = planSequence(catalog->files, nextCursor, request);
                }
            }
        }

        // Re-reading the same unchanged batch keeps the exact previous output reference.
        if (!reload && lastBatchRef_.has_value() && plan.files.size() == lastPlannedPaths_.size())
        {
            std::vector<std::string> planned;
            planned.reserve(plan.files.size());
            for (const auto &file : plan.files)
                planned.push_back(utf8Path(file));
            if (planned == lastPlannedPaths_)
            {
                if (scanned && sourceChanged)
                    cache_.clear();
                if (scanned)
                    catalog_ = std::move(*scanned);
                reverse_ = isReverse;
                cursor_  = std::move(nextCursor);
                cursor_.started   = true;
                cursor_.nextKey   = plan.nextKey;
                cursor_.lastKey   = plan.lastKey;
                cursor_.exhausted = plan.exhausted;
                visionrt::Values outputs;
                outputs.emplace(p::out::IMAGES, *lastBatchRef_);
                outputs.emplace(p::out::PATHS, visionrt::Array(std::move(planned)));
                outputs.emplace(p::out::CURRENT_INDEX, static_cast<std::int64_t>(plan.startIndex));
                outputs.emplace(p::out::TOTAL_COUNT, static_cast<std::int64_t>(catalog_->files.size()));
                outputs.emplace(p::out::END_REACHED, plan.exhausted);
                return visionrt::toolResult(std::move(outputs), visionrt::executionSuccess());
            }
        }

        PendingInserts pending;
        auto           batch = decode(*catalog, plan, reload, pending);
        batch.pending        = std::move(pending);
        return commit(std::move(batch), plan, std::move(nextCursor), scanned ? &*scanned : nullptr, sourceChanged,
                      isReverse, catalog ? catalog->files.size() : 0);
    }
    catch (const ImageFailure &error)
    {
        return visionrt::toolResult(missingOutputs(), {1, error.message, error.detail});
    }
    catch (const visionrt::ContractError &error)
    {
        return visionrt::toolResult(missingOutputs(), {1, error.what()});
    }
    catch (const std::exception &error)
    {
        return visionrt::toolResult(missingOutputs(), {1, "图像读取失败", error.what()});
    }
    catch (...)
    {
        return visionrt::toolResult(missingOutputs(), {-1, "图像读取失败：未知异常"});
    }
}

visionrt::ToolStatus ReadImageTool::release()
{
    catalog_.reset();
    resetSequence();
    return {};
}

} // namespace visiontools::preprocess
