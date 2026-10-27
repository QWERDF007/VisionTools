#include "ImageSequence.h"

#include <visionrt/contracts/Tool.h>

#include <algorithm>
#include <optional>

namespace visiontools::preprocess {
namespace {
std::string keyOf(const std::filesystem::path &path)
{
    const auto value = path.generic_u8string();
    return {reinterpret_cast<const char *>(value.data()), value.size()};
}

std::size_t positionOf(const std::vector<std::string> &keys, const std::string &key)
{
    const auto found = std::find(keys.begin(), keys.end(), key);
    return found == keys.end() ? keys.size() : static_cast<std::size_t>(std::distance(keys.begin(), found));
}

// Ordered index sequence for the requested direction.
std::size_t orderIndex(std::size_t step, std::size_t total, bool reverse)
{
    return reverse ? total - 1 - step : step;
}

std::size_t bytesOf(const cv::Mat &image)
{
    return image.empty() ? 0 : image.total() * image.elemSize();
}
} // namespace

SequencePosition startPosition(const std::vector<std::string> &keys, const SequenceCursor &cursor, bool reverse,
                              bool loop)
{
    const std::size_t total = keys.size();
    const auto        find  = [&](const std::string &key) -> std::optional<std::size_t>
    {
        if (key.empty())
            return std::nullopt;
        const auto position = positionOf(keys, key);
        if (position == total)
            return std::nullopt;
        return reverse ? total - 1 - position : position;
    };

    // A cursor that never delivered anything starts at the first file of the direction.
    if (!cursor.started)
        return {SequenceStart::Found, 0};

    // 1) The recorded successor still exists: continue exactly there.
    if (const auto step = find(cursor.nextKey))
        return {SequenceStart::Found, *step};

    // 2) The successor was deleted: continue after the last delivered file.
    if (const auto step = find(cursor.lastKey))
    {
        if (*step + 1 < total)
            return {SequenceStart::Found, *step + 1};
        // The last delivered file is the tail of the new catalog. Wrapping is a decision for the
        // caller, never an implicit start from the first file.
        return loop ? SequencePosition{SequenceStart::RestartRequested, 0}
                    : SequencePosition{SequenceStart::End, 0};
    }

    // 3) Both anchors are gone. A looping sequence restarts on request; a non-looping one ends
    // instead of silently delivering images that were already shown.
    return loop ? SequencePosition{SequenceStart::RestartRequested, 0} : SequencePosition{SequenceStart::End, 0};
}

SequencePlan planSequence(const std::vector<std::filesystem::path> &catalog, const SequenceCursor &cursor,
                          const SequenceRequest &request)
{
    SequencePlan plan;
    plan.keys.reserve(catalog.size());
    for (const auto &path : catalog)
        plan.keys.push_back(keyOf(path));

    const std::size_t total = catalog.size();
    if (total == 0)
        throw visionrt::ContractError("没有可用图像文件");
    if (request.count < 1)
        throw visionrt::ContractError("输入端口 'count' 必须为正整数");

    if (!request.loop && cursor.exhausted)
        throw visionrt::ContractError("图像序列已结束");

    const auto position = startPosition(plan.keys, cursor, request.reverse, request.loop);
    if (position.kind == SequenceStart::End)
        throw visionrt::ContractError("图像序列已结束");
    const std::size_t start = position.kind == SequenceStart::RestartRequested ? 0 : position.step;

    // Looping may legitimately deliver the same file more than once in one batch, so `count` is
    // the batch size, not a cap on the photo count. A non-looping sequence stops at its end.
    const std::size_t take = request.loop ? request.count : std::min(request.count, total);
    std::size_t       step = start;

    plan.files.reserve(take);
    plan.startIndex = orderIndex(step, total, request.reverse);
    for (std::size_t i = 0; i < take; ++i)
    {
        const std::size_t index = orderIndex(step, total, request.reverse);
        plan.files.push_back(catalog[index]);
        plan.lastKey = plan.keys[index];
        if (i + 1 == take)
            break;
        if (request.loop)
            step = (step + 1) % total;
        else
        {
            if (step + 1 >= total)
                break;
            ++step;
        }
    }

    if (request.loop)
    {
        step           = (start + plan.files.size()) % total;
        plan.nextKey   = plan.keys[orderIndex(step, total, request.reverse)];
        plan.exhausted = false;
    }
    else
    {
        const std::size_t lastStep = start + plan.files.size();
        plan.exhausted             = lastStep >= total;
        if (!plan.exhausted)
            plan.nextKey = plan.keys[orderIndex(lastStep, total, request.reverse)];
    }
    return plan;
}

DecodedImageCache::DecodedImageCache(std::size_t budgetBytes)
    : budget_(budgetBytes)
{
}

const cv::Mat *DecodedImageCache::find(const std::string &key)
{
    const auto found = entries_.find(key);
    if (found == entries_.end())
        return nullptr;
    // A hit counts as a use: keep real recency so the next eviction picks the coldest entry.
    order_.splice(order_.begin(), order_, found->second.order);
    return &found->second.image;
}

bool DecodedImageCache::insert(const std::string &key, const cv::Mat &image)
{
    const auto bytes = bytesOf(image);
    if (bytes > budget_)
        return false;
    if (const auto existing = entries_.find(key); existing != entries_.end())
    {
        bytes_ -= existing->second.bytes;
        order_.erase(existing->second.order);
        entries_.erase(existing);
    }
    if (!evictUntilWithinBudget(bytes))
        return false;
    // Keep one order node per entry: if the map insertion fails, the reservation is rolled back so
    // the cache never carries an order key without bytes or an entry without a position.
    bool reserved = false;
    try
    {
        order_.push_front(key);
        reserved = true;
        entries_.emplace(key, Entry{image, bytes, order_.begin()});
    }
    catch (...)
    {
        if (reserved)
            order_.erase(order_.begin());
        throw;
    }
    bytes_ += bytes;
    return true;
}

void DecodedImageCache::protect(const std::vector<std::string> &keys)
{
    protected_.clear();
    for (const auto &key : keys)
        protected_.emplace(key, 0);
}

void DecodedImageCache::clear()
{
    entries_.clear();
    order_.clear();
    protected_.clear();
    bytes_ = 0;
}

bool DecodedImageCache::evictUntilWithinBudget(std::size_t incoming)
{
    // Least recently used first, never the protected batch of the last output. When the protected
    // batch alone occupies the budget the history stays exactly at its budget and the incoming
    // image is simply not retained.
    while (bytes_ + incoming > budget_)
    {
        bool evicted = false;
        for (auto it = order_.end(); it != order_.begin();)
        {
            --it;
            if (protected_.contains(*it))
                continue;
            const auto entry = entries_.find(*it);
            if (entry != entries_.end())
            {
                bytes_ -= entry->second.bytes;
                entries_.erase(entry);
            }
            order_.erase(it);
            evicted = true;
            break;
        }
        if (!evicted)
            return false;
    }
    return true;
}
} // namespace visiontools::preprocess
