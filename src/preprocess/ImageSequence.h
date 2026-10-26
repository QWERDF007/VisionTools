#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <list>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace visiontools::preprocess {

// Pure planning result of one image-read call. The cursor is an explicit file
// identity plus an exhausted flag, so a rescan can never silently restart a
// finished non-looping sequence and a reverse scan keeps its real successor.
struct SequencePlan
{
    std::vector<std::filesystem::path> files;
    std::vector<std::string>           keys;
    std::size_t                        startIndex = 0; // 当前项在目录中的正序下标
    std::string                        nextKey;       // 下一项文件标识；耗尽时为空
    std::string                        lastKey;       // 本次最后交付的文件标识
    bool                               exhausted = false;
};

struct SequenceCursor
{
    std::string nextKey;
    std::string lastKey;
    bool        exhausted = false;
};

struct SequenceRequest
{
    bool        reverse = false;
    bool        loop    = false;
    std::size_t count   = 1;
};

// Throws visionrt::ContractError when the sequence is already finished in
// non-looping mode. `catalog` must be sorted in forward order.
SequencePlan planSequence(const std::vector<std::filesystem::path> &catalog, const SequenceCursor &cursor,
                          const SequenceRequest &request);

// Bounded decode cache. It only releases its own references: images handed out
// through DataRef keep their owner alive for as long as a consumer holds them.
class DecodedImageCache
{
public:
    explicit DecodedImageCache(std::size_t budgetBytes);

    const cv::Mat *find(const std::string &key) const;
    // Returns false when the image is larger than the whole budget: the caller may
    // still return it once, but it must not enter the retained history.
    bool insert(const std::string &key, const cv::Mat &image);
    // Protects the keys of the last successful output batch from eviction.
    void protect(const std::vector<std::string> &keys);
    void clear();
    std::size_t bytes() const noexcept { return bytes_; }
    std::size_t size() const noexcept { return entries_.size(); }

private:
    struct Entry
    {
        cv::Mat                   image;
        std::size_t               bytes = 0;
        std::list<std::string>::iterator order;
    };

    void evictUntilWithinBudget(std::size_t incoming);

    std::size_t                                       budget_;
    std::size_t                                       bytes_ = 0;
    std::list<std::string>                            order_; // front = most recent
    std::unordered_map<std::string, Entry>             entries_;
    std::map<std::string, char>                       protected_;
};

} // namespace visiontools::preprocess
