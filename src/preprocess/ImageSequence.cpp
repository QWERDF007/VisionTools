#include "ImageSequence.h"

#include <visionrt/contracts/Tool.h>

#include <algorithm>

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

// Step of `key` inside the current direction, or the begin position when it vanished.
std::size_t startStep(const std::vector<std::string> &keys, const SequenceCursor &cursor, bool reverse)
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
    // 1) The recorded successor still exists: continue exactly there.
    if (const auto step = find(cursor.nextKey))
        return *step;
    // 2) The successor was deleted: continue after the last delivered file.
    if (const auto step = find(cursor.lastKey); step && *step + 1 < total)
        return *step + 1;
    // 3) Both vanished or the last delivered file was the end: start the direction again.
    return 0;
}

std::size_t bytesOf(const cv::Mat &image)
{
    return image.empty() ? 0 : image.total() * image.elemSize();
}
} // namespace

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

    // Looping may legitimately deliver the same file more than once in one batch, so `count` is
    // the batch size, not a cap on the photo count. A non-looping sequence stops at its end.
    const std::size_t take = request.loop ? request.count : std::min(request.count, total);
    std::size_t       step = startStep(plan.keys, cursor, request.reverse);

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
        step = (startStep(plan.keys, cursor, request.reverse) + plan.files.size()) % total;
        plan.nextKey = plan.keys[orderIndex(step, total, request.reverse)];
        plan.exhausted = false;
    }
    else
    {
        const std::size_t lastStep = startStep(plan.keys, cursor, request.reverse) + plan.files.size();
        plan.exhausted = lastStep >= total;
        if (!plan.exhausted)
            plan.nextKey = plan.keys[orderIndex(lastStep, total, request.reverse)];
    }
    return plan;
}

DecodedImageCache::DecodedImageCache(std::size_t budgetBytes)
    : budget_(budgetBytes)
{
}

const cv::Mat *DecodedImageCache::find(const std::string &key) const
{
    const auto found = entries_.find(key);
    return found == entries_.end() ? nullptr : &found->second.image;
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
    evictUntilWithinBudget(bytes);
    order_.push_front(key);
    entries_.emplace(key, Entry{image, bytes, order_.begin()});
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

void DecodedImageCache::evictUntilWithinBudget(std::size_t incoming)
{
    // Least recently used first, never the protected batch of the last output.
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
            break; // everything retained is protected by the last successful batch
    }
}
} // namespace visiontools::preprocess
