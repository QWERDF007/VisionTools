#include "ReadImageTool.h"

#include "ReadImageParameters.h"
#include "common/InputValues.h"
#include "common/UpstreamStatus.h"
#include "errors/ReadImageErrors.h"
#include "common/errors/ToolError.h"

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

} // namespace

// ---- Parameter subclasses -------------------------------------------------

visionrt::ToolStatus ReadImageInputs::readFields(visionrt::InputReader &reader)
{
    Readings readings(reader);
    path      = readings.text(p::in::PATH);
    recursive = readings.boolean(p::in::RECURSIVE);
    traversal = readings.text(p::in::TRAVERSAL);
    order     = readings.text(p::in::ORDER);
    count     = readings.integer(p::in::COUNT);
    reload    = readings.boolean(p::in::RELOAD);
    if (readings.failed())
        return readings.status(ReadImageError::PathNotFound, CommonToolError::InputTypeMismatch);
    if (path.empty())
        return status(ReadImageError::PathNotFound, std::string(p::in::PATH) + " 不能为空");
    return {};
}

void ReadImageOutputs::writeFields(visionrt::OutputWriter &writer)
{
    if (!present)
    {
        writer.missing(p::out::IMAGES);
        writer.missing(p::out::PATHS);
        writer.missing(p::out::CURRENT_INDEX);
        writer.missing(p::out::TOTAL_COUNT);
        writer.missing(p::out::END_REACHED);
        return;
    }
    writer.optionalImage(p::out::IMAGES, images);
    if (paths)
        writer.array(p::out::PATHS, *paths);
    else
        writer.missing(p::out::PATHS);
    writer.integer(p::out::CURRENT_INDEX, currentIndex);
    writer.integer(p::out::TOTAL_COUNT, totalCount);
    writer.boolean(p::out::END_REACHED, endReached);
}

#ifdef VISIONTOOLS_TESTING
namespace testing {
std::function<void(const std::filesystem::path &)> beforeRead;
std::function<void(const std::filesystem::path &)> afterDecode;
std::function<void()>                              beforeResultBuild;
std::function<void()>                              beforeCacheInsert;
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
        return {visionrt::kContractViolationStatus, "图像读取不接受初始化参数"};
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
            fail(ReadImageError::FileReadFailed, pathStr);
        const auto endPosition = stream.tellg();
        if (endPosition == std::streampos(-1) || !stream)
            fail(ReadImageError::FileReadFailed, pathStr, "无法读取文件长度");
        const auto fileSize = static_cast<std::streamoff>(endPosition);
        if (fileSize <= 0)
            fail(ReadImageError::FileReadFailed, pathStr, "文件为空");
        if (static_cast<std::uintmax_t>(fileSize) > std::numeric_limits<std::size_t>::max()
            || fileSize > std::numeric_limits<std::streamsize>::max()
            || fileSize > std::numeric_limits<int>::max())
            fail(ReadImageError::FileTooLarge, pathStr, "文件大小 " + std::to_string(fileSize) + " 字节超出可表示范围");

        std::vector<char> buffer(static_cast<std::size_t>(fileSize));
        stream.seekg(0, std::ios::beg);
        if (!stream)
            fail(ReadImageError::FileReadFailed, pathStr, "无法定位读取位置");

#ifdef VISIONTOOLS_TESTING
        if (testing::beforeRead)
            testing::beforeRead(file);
        testing::fileReadCount.fetch_add(1);
#endif

        stream.read(buffer.data(), static_cast<std::streamsize>(fileSize));
        const auto readBytes = stream.gcount();
        if (readBytes != static_cast<std::streamsize>(fileSize) || stream.bad())
            fail(ReadImageError::FileReadFailed, pathStr,
                 "期望 " + std::to_string(fileSize) + " 字节，实际 " + std::to_string(readBytes) + " 字节");

        cv::Mat decoded;
        try
        {
            const cv::Mat rawData(1, static_cast<int>(buffer.size()), CV_8UC1, buffer.data());
            decoded = cv::imdecode(rawData, cv::IMREAD_UNCHANGED);
        }
        catch (const cv::Exception &error)
        {
            fail(ReadImageError::DecodeFailed, pathStr + "（批次索引 " + std::to_string(i) + "）", error.what());
        }
        if (decoded.empty())
            fail(ReadImageError::DecodeFailed, pathStr + "（批次索引 " + std::to_string(i) + "）", "无法解码图像");

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
    // Build the complete next state and the complete result before touching any committed member.
    // Constructing the image handle, the Values map and the paths array can still allocate, so none
    // of them may run after the tool state has moved.
#ifdef VISIONTOOLS_TESTING
    if (testing::beforeResultBuild)
        testing::beforeResultBuild();
#endif
    std::optional<Catalog>   nextCatalog;
    if (scannedCatalog)
        nextCatalog = *scannedCatalog;
    auto                     batchRef  = visionrt::DataRef<visionrt::Image>(std::move(batch.images));
    std::vector<std::string> nextPaths = batch.paths;
    ReadImageOutputs         outputs;
    outputs.present      = true;
    outputs.images       = batchRef; // copied handle; batchRef itself moves into lastBatchRef_ below
    outputs.paths        = visionrt::DataRef<std::vector<std::string>>(batch.paths);
    outputs.currentIndex = static_cast<std::int64_t>(plan.startIndex);
    outputs.totalCount   = static_cast<std::int64_t>(totalFiles);
    outputs.endReached   = plan.exhausted;
    auto result          = outputs.pack();

    SequenceCursor nextCursor = std::move(cursor);
    nextCursor.started        = true;
    nextCursor.nextKey        = plan.nextKey;
    nextCursor.lastKey        = plan.lastKey;
    nextCursor.exhausted      = plan.exhausted;

    // Cache maintenance is an optimization, never part of the transaction: an allocation failure
    // keeps the cache internally consistent and leaves the already-built result valid.
    if (sourceChanged)
        cache_.clear();
    try
    {
#ifdef VISIONTOOLS_TESTING
        if (testing::beforeCacheInsert)
            testing::beforeCacheInsert();
#endif
        for (auto &[key, image] : batch.pending)
            cache_.insert(key, image);
        cache_.protect(batch.paths);
    }
    catch (...)
    {
    }

    // Commit point: only non-throwing moves and swaps remain.
    if (scannedCatalog)
        catalog_.swap(nextCatalog);
    lastPlannedPaths_.swap(nextPaths);
    lastBatchRef_ = std::move(batchRef);
    reverse_ = reverse;
    cursor_  = std::move(nextCursor);
    return result;
}

visionrt::ToolResult ReadImageTool::execute(const visionrt::Values &inputs)
{
    // An unselected upstream branch is not an error: no algorithm runs and nothing is read, so the
    // decision is taken before the catalog, the cursor or the cache can change. The typed parse
    // below re-reads the same status; a status type error propagates exactly as before.
    if (const auto upstream = upstreamStatus(inputs); upstream != 0)
        return skippedByUpstream(upstream, ReadImageOutputs::missing().pack().outputs);

    try
    {
        ReadImageInputs args;
        args.parse(inputs);
        const auto &rawPath = args.path;
        visionrt::validateUtf8(rawPath);
        if (rawPath.empty())
            fail(ReadImageError::PathNotFound, std::string(p::in::PATH) + " 不能为空");
        const bool recursive = args.recursive;
        const bool reload    = args.reload;

        const auto &traversal = args.traversal;
        bool       isLoop    = true;
        if (traversal == "循环" || traversal == "loop")
            isLoop = true;
        else if (traversal == "不循环" || traversal == "once")
            isLoop = false;
        else
            fail(CommonToolError::InputOutOfRange, std::string(p::in::TRAVERSAL) + " 只接受 循环/不循环，实际值：" + traversal);

        const auto &order     = args.order;
        bool       isReverse = false;
        if (order.find("逆序") != std::string::npos || order == "reverse")
            isReverse = true;
        else if (order.find("顺序") != std::string::npos || order == "forward")
            isReverse = false;
        else
            fail(CommonToolError::InputOutOfRange,
                 std::string(p::in::ORDER) + " 只接受 顺序（下一张）/逆序（上一张），实际值：" + order);

        const auto count = args.count;
        if (count < 1 || count > kMaxCount)
            fail(CommonToolError::InputOutOfRange,
                 std::string(p::in::COUNT) + " 必须在 1–" + std::to_string(kMaxCount) + " 之间，实际值：" + std::to_string(count));

        const auto resolution = services_.paths.tryResolve(rawPath);
        if (!resolution)
            fail(ReadImageError::PathNotFound, resolution.issue().field, resolution.issue().text);
        const auto &resolved = resolution.value();
        if (!std::filesystem::exists(resolved))
            fail(ReadImageError::PathNotFound, utf8Path(resolved));
        const bool isDir  = std::filesystem::is_directory(resolved);
        const bool isFile = std::filesystem::is_regular_file(resolved);
        if (!isDir && !isFile)
            fail(ReadImageError::PathNotFound, utf8Path(resolved), "路径既非文件也非目录");

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
                fail(ReadImageError::NoImagesFound, utf8Path(resolved));
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
                const auto totalFiles = catalog ? catalog->files.size() : 0;
                // Build the answer first, then move the state: a failed map/array allocation must
                // leave the cursor where it was so the next call returns this same batch.
                ReadImageOutputs outputs;
                outputs.present      = true;
                outputs.images       = *lastBatchRef_;
                outputs.paths        = visionrt::DataRef<std::vector<std::string>>(std::move(planned));
                outputs.currentIndex = static_cast<std::int64_t>(plan.startIndex);
                outputs.totalCount   = static_cast<std::int64_t>(totalFiles);
                outputs.endReached   = plan.exhausted;
                auto result          = outputs.pack();

                SequenceCursor next = std::move(nextCursor);
                next.started        = true;
                next.nextKey        = plan.nextKey;
                next.lastKey        = plan.lastKey;
                next.exhausted      = plan.exhausted;
                std::optional<Catalog> nextCatalog;
                if (scanned)
                    nextCatalog = std::move(*scanned);
                if (scanned && sourceChanged)
                    cache_.clear();
                if (scanned)
                    catalog_.swap(nextCatalog);
                reverse_ = isReverse;
                cursor_  = std::move(next);
                return result;
            }
        }

        PendingInserts pending;
        auto           batch = decode(*catalog, plan, reload, pending);
        batch.pending        = std::move(pending);
        return commit(std::move(batch), plan, std::move(nextCursor), scanned ? &*scanned : nullptr, sourceChanged,
                      isReverse, catalog ? catalog->files.size() : 0);
    }
    catch (const visionrt::ContractError &)
    {
        throw;
    }
    catch (const ToolFailure &error)
    {
        auto outputs = ReadImageOutputs::missing();
        outputs.markFailed(error.code, error.message, error.detail);
        return outputs.pack();
    }
    catch (const std::exception &error)
    {
        auto outputs = ReadImageOutputs::missing();
        const auto failure = status(CommonToolError::OperatorFailed, "图像读取失败", error.what());
        outputs.markFailed(failure.code, failure.message, failure.detail);
        return outputs.pack();
    }
    catch (...)
    {
        auto outputs = ReadImageOutputs::missing();
        const auto failure = status(CommonToolError::OperatorFailed, "图像读取失败：未知异常");
        outputs.markFailed(failure.code, failure.message, failure.detail);
        return outputs.pack();
    }
}

visionrt::ToolStatus ReadImageTool::release()
{
    catalog_.reset();
    resetSequence();
    return {};
}

} // namespace visiontools::preprocess
